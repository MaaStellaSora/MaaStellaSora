"""Read visible note inventories using Maa's existing icon and OCR resources.

Inputs are client-area images, normalized to the pipeline's 1280 x 720 space.
Only the complete inventory bar is read; other page content is excluded.
"""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
from statistics import median

import cv2
import numpy as np


ICON_NODE = "星塔_背包_识别音符_agent"
COUNT_NODE = "星塔_背包_识别音符数量_agent"
TEXT_NODE = "NoteOCR_Text"
TEXT_RULE_NODES = {
    "bag_count": "星塔_音符采集_背包数量",
    "concerto_level": "星塔_音符采集_协奏等级",
    "event": "星塔_音符采集_事件文本规则",
}


@lru_cache(maxsize=1)
def load_ocr_rules():
    return json.loads(Path(__file__).with_name("note_ocr_rules.json").read_text(encoding="utf-8"))


OCR_RULES = load_ocr_rules()


def text_rules(context, kind):
    """Use the active region's Pipeline rules, with shipped offline defaults."""
    node = getattr(context, "get_node_data", lambda name: None)(TEXT_RULE_NODES[kind]) or {}
    return node.get("attach", {}).get("text_rules", OCR_RULES["text_rules"][kind])


MELODIES = tuple("melody_of_" + name for name in (
    "pummel", "luck", "burst", "stamina", "focus", "skill", "ultimate",
    "ignis", "umbra", "aqua", "ventus", "lux", "terra",
))
COMMON_MELODIES = MELODIES[:7]
TOWER_ELEMENTS = {
    "光土": ("melody_of_lux", "melody_of_terra"),
    "水风": ("melody_of_aqua", "melody_of_ventus"),
    "火暗": ("melody_of_ignis", "melody_of_umbra"),
}


def visible_melodies_for_tower(tower):
    """The tower selects seven common notes and its two element identities."""
    return COMMON_MELODIES + TOWER_ELEMENTS[tower]


# Regions describe layouts, while individual digits remain anchored to icons.
INVENTORY_RULES = OCR_RULES["inventory"]
INVENTORY_REGIONS = {key: tuple(value) for key, value in INVENTORY_RULES["regions"].items()}


def normalize_image(image, client_rect=None, reference_size=None):
    """Crop an explicitly supplied client rectangle, then use Maa coordinates."""
    if client_rect is not None:
        x, y, width, height = client_rect
        if reference_size:
            sx, sy = image.shape[1] / reference_size[0], image.shape[0] / reference_size[1]
            x, y, width, height = round(x * sx), round(y * sy), round(width * sx), round(height * sy)
        image = image[y:y + height, x:x + width]
    return cv2.resize(image, (1280, 720), interpolation=cv2.INTER_CUBIC)


def _records(detail):
    if not detail:
        return []
    return [
        {"box": list(item.box), "score": float(item.score),
         **({"text": item.text} if hasattr(item, "text") else {})}
        for item in detail.all_results
    ]


def valid_text(record, minimum=None):
    minimum = OCR_RULES["ocr"]["threshold"] if minimum is None else minimum
    return record.get("detection_complete", True) and record["score"] >= minimum


def read_text(context, image, roi, only_rec=False, expected=(), *, node=TEXT_NODE):
    params = {"roi": list(roi), "only_rec": only_rec,
              "threshold": OCR_RULES["ocr"]["threshold"], "replace": []}
    if expected or node == TEXT_NODE:
        params["expected"] = list(expected)
    detail = context.run_recognition(node, image, {
        node: {"recognition": {"type": "OCR", "param": params}},
    })
    records = _records(detail)
    if not only_rec:
        for record in records:
            if tuple(record["box"]) == tuple(roi):
                record.update(detection_complete=False, reason="ocr_detection_fallback")
    return records


def _overlaps(first, second):
    x, y, width, height = first
    a, b, w, h = second
    area = max(0, min(x + width, a + w) - max(x, a)) * max(
        0, min(y + height, b + h) - max(y, b))
    return area > OCR_RULES["icons"]["overlap_fraction"] * min(width * height, w * h)


def find_icons(context, image, roi, scales=None, threshold=None, melodies=MELODIES):
    """Use the official templates at explicit image scales, keeping all hits."""
    scales = OCR_RULES["icons"]["default_scales"] if scales is None else scales
    threshold = OCR_RULES["icons"]["default_threshold"] if threshold is None else threshold
    found = []
    for scale in scales:
        if context.tasker.stopping:
            break
        scaled = (image if scale == 1.0 else cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        scaled_roi = [round(value * scale) for value in roi]
        for melody in melodies:
            detail = context.run_recognition(ICON_NODE, scaled, {
                ICON_NODE: {"recognition": {"param": {
                    "template": [OCR_RULES["icons"]["template"].format(melody=melody)],
                    "roi": scaled_roi, "threshold": threshold,
                    "green_mask": True, "order_by": "Score",
                }}},
            })
            for hit in _records(detail):
                if hit["score"] < threshold:
                    continue
                hit["box"] = [round(value / scale) for value in hit["box"]]
                found.append({"melody": melody, **hit})
    unique = []
    for hit in sorted(found, key=lambda item: item["score"], reverse=True):
        if not any(_overlaps(hit["box"], item["box"]) for item in unique):
            unique.append(hit)
    return sorted(unique, key=lambda item: (item["box"][1], item["box"][0]))


def _numeric_values(records):
    return {int(item["text"]) for item in records
            if valid_text(item, OCR_RULES["ocr"]["numeric_confidence"])
            and re.fullmatch(OCR_RULES["ocr"]["numeric_pattern"], item.get("text", ""))}


def white_glyph_image(image, roi):
    """Isolate the existing white count glyphs for the small-number retry."""
    left, top, width, height = roi
    crop = image[top:top + height, left:left + width]
    rules = OCR_RULES["white_glyphs"]
    white = cv2.inRange(crop, tuple(rules["lower"]), tuple(rules["upper"]))
    points = cv2.findNonZero(white)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    border, scale = rules["border"], rules["scale"]
    glyphs = cv2.copyMakeBorder(255 - white[y:y + height, x:x + width],
                                border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(cv2.resize(glyphs, None, fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)


def _locate_inventory_bar(image, icons, visible_melodies, *, append_elements=False):
    """Require the ordered, evenly spaced row before looking for digits."""
    slots = {melody: index for index, melody in enumerate(visible_melodies)}
    candidates = [item for item in icons if item["melody"] in slots]
    evidence = {"candidates": icons, "expected_order": list(visible_melodies),
                "configured_elements": list(visible_melodies[7:])}
    if len(slots) != 9:
        return [], "tower_not_selected", evidence

    def center(item):
        x, y, width, height = item["box"]
        return x + width / 2, y + height / 2

    rules = INVENTORY_RULES
    anchors = [item for item in candidates if item["score"] >= rules["anchor_threshold"]]
    best = None
    # Maa's element selection determines membership, not the UI's last-slot order.
    configured = tuple(visible_melodies)
    for order in (configured, configured[:7] + configured[7:][::-1]):
        slots = {melody: index for index, melody in enumerate(order)}
        for first in anchors:
            a = slots[first["melody"]]
            ax, ay = center(first)
            for last in anchors:
                b = slots[last["melody"]]
                bx, by = center(last)
                if b <= a or abs(ay - by) > rules["y_tolerance"]:
                    continue
                spacing = (bx - ax) / (b - a)
                if not rules["spacing_range"][0] <= spacing <= rules["spacing_range"][1]:
                    continue
                origin, row_y = ax - a * spacing, (ay + by) / 2
                matched = {}
                for item in candidates:
                    index = slots[item["melody"]]
                    cx, cy = center(item)
                    if (abs(cx - origin - index * spacing) > rules["x_tolerance"]
                            or abs(cy - row_y) > rules["y_tolerance"]):
                        continue
                    if index not in matched or item["score"] > matched[index]["score"]:
                        matched[index] = item
                trusted = [index for index, item in matched.items() if item["score"] >= rules["anchor_threshold"]]
                if (len(trusted) < rules["minimum_anchors"]
                        or max(trusted) - min(trusted) < rules["minimum_anchor_span"]):
                    continue
                rank = (len(trusted), len(matched), sum(item["score"] for item in matched.values()))
                if best is None or rank > best[0]:
                    best = rank, matched, origin, row_y, spacing
    if best is None:
        return [], "no_ordered_inventory_row", evidence

    _, matched, origin, row_y, spacing = best
    if append_elements:
        # Build records append other held elements after the tower's nine
        # notes. Their identities come from the actual icons at those slots.
        extra = [item for item in icons if item["melody"] not in visible_melodies]
        used = set(visible_melodies)
        for item in sorted(extra, key=lambda item: item["score"], reverse=True):
            cx, cy = center(item)
            index = round((cx - origin) / spacing)
            if (9 <= index < len(MELODIES) and index not in matched and item["melody"] not in used
                    and abs(cx - origin - index * spacing) <= rules["x_tolerance"]
                    and abs(cy - row_y) <= rules["y_tolerance"]):
                matched[index] = item
                used.add(item["melody"])
    row_length = max(9, max(matched) + 1)
    quality = []
    for index, item in sorted(matched.items()):
        x, y, width, height = item["box"]
        gray = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
        quality.append({"slot": index, "white_level": float(np.percentile(gray, rules["white_percentile"])),
                        "detail": float(cv2.Laplacian(gray, cv2.CV_32F).var())})
    evidence.update(origin_x=origin, row_y=row_y, spacing=spacing,
                    matched_slots=sorted(matched), icon_quality=quality,
                    observed_order=[matched[index]["melody"] if index in matched else None
                                    for index in range(row_length)])
    # Native page waits settle animations before capture. These measurements
    # only reject a severely dimmed or blurred inventory row.
    if (median(item["white_level"] for item in quality) < rules["minimum_white"]
            or median(item["detail"] for item in quality) < rules["minimum_detail"]):
        return [], "inventory_row_not_clear", evidence
    return [matched[index] for index in sorted(matched)], "clear_ordered_inventory_row", evidence


def _read_count(context, image, icon, layout):
    # Use the upstream count node with an icon-relative ROI.
    # Settlement icons vary slightly in vertical alignment. Keep the digit's
    # upper strokes inside the ROI for every icon in that row.
    offset = INVENTORY_RULES["settlement_count_offset" if layout == "settlement" else "count_offset"]
    detail = context.run_recognition(COUNT_NODE, image, {
        COUNT_NODE: {"recognition": {"param": {
            "roi": icon["box"],
            "roi_offset": offset,
            "replace": [], "expected": [],
        }}},
    })
    records = _records(detail)
    values = _numeric_values(records)
    if len(values) != 1:
        x, y, width, height = icon["box"]
        roi = [max(0, x + offset[0]), y + offset[1],
               width + offset[2], height + offset[3]]
        text_image = white_glyph_image(image, roi)
        if text_image is not None:
            retry = context.run_recognition(COUNT_NODE, text_image, {
                COUNT_NODE: {"recognition": {"param": {
                    "roi": [0, 0, text_image.shape[1], text_image.shape[0]],
                    "roi_offset": [0, 0, 0, 0],
                    "replace": [], "expected": [],
                }}},
            })
            retried = _records(retry)
            for item in retried:
                item.update(box=roi, preprocessing="white_glyphs_3x")
            records.extend(retried)
            values.update(_numeric_values(retried))
    return (next(iter(values)) if len(values) == 1 else None), records


def read_inventory(context, image, layout, visible_melodies=(), *, tower=None):
    """Return counts, independent readings, and explicit missing/conflict states.

    ``visible_melodies`` is tower/UI metadata, never a source of count values.
    Unshown types remain unknown; no missing value is converted to zero.
    """
    if tower is not None:
        visible_melodies = visible_melodies_for_tower(tower)
    elif not visible_melodies:
        node = context.get_node_data("星塔_属性塔选择_agent") or {}
        elements = tuple(node.get("attach", {}).get("active_melodies", ()))
        if elements:
            tower = next((name for name, pair in TOWER_ELEMENTS.items() if set(pair) == set(elements)), None)
            visible_melodies = visible_melodies_for_tower(tower) if tower else COMMON_MELODIES + elements
    readings = []
    region = INVENTORY_REGIONS.get(layout)
    bar_reason, bar_evidence, bar_present = "unsupported_layout", {}, False
    if region and visible_melodies:
        append_elements = layout == "settlement"
        candidates = find_icons(context, image, region, scales=INVENTORY_RULES["icon_scales"],
                                threshold=INVENTORY_RULES["icon_threshold"],
                                melodies=MELODIES if append_elements else visible_melodies)
        icons, bar_reason, bar_evidence = _locate_inventory_bar(image, candidates, visible_melodies,
                                                              append_elements=append_elements)
        bar_present = bool(icons)
        for icon in icons:
            current, raw = _read_count(context, image, icon, layout)
            readings.append({"melody": icon["melody"], "source": "inventory",
                             "current": current, "raw": raw, "box": icon["box"],
                             "icon_score": icon["score"],
                             "conflicting_values": sorted(_numeric_values(raw)) if len(_numeric_values(raw)) > 1 else []})
        if append_elements and bar_present:
            visible_melodies = tuple(dict.fromkeys((*visible_melodies, *(item["melody"] for item in icons))))
            for slot, melody in enumerate(bar_evidence["observed_order"]):
                if slot >= 9 and melody is None:
                    readings.append(dict(melody=None, current=None, source="inventory",
                                         raw=[], slot=slot, reason="unresolved_inventory_slot"))
    elif region:
        bar_reason = "tower_not_selected"
    result = combine_readings(readings, layout, visible_melodies)
    result.update(tower=tower, bar_present=bar_present, bar_reason=bar_reason, bar_evidence=bar_evidence)
    result["complete"] = bar_present and result["complete"]
    return result


def combine_readings(readings, layout, visible_melodies=()):
    """Reconcile independent current values without borrowing other semantics."""
    by_melody = defaultdict(list)
    unresolved, skipped = [], []
    for reading in readings:
        if reading.get("kind") == "skipped":
            (skipped if reading.get("reason") and reading.get("alternative") else unresolved).append(reading)
        elif reading.get("melody") not in MELODIES:
            unresolved.append(reading)
        else:
            by_melody[reading["melody"]].append(reading)
    counts = {melody: None for melody in MELODIES}
    conflicts, incomplete, cross_checked = [], [], []
    for melody, items in by_melody.items():
        values = {item["current"] for item in items if item["current"] is not None}
        if len(values) > 1 or any(item.get("conflicting_values") for item in items):
            conflicts.append(melody)
        else:
            if any(item["current"] is None for item in items):
                incomplete.append(melody)
            if values:
                counts[melody] = next(iter(values))
                if len({item["source"] for item in items if item["current"] is not None}) > 1:
                    cross_checked.append(melody)
    missing = [melody for melody in visible_melodies if counts[melody] is None]
    return {"layout": layout, "counts": counts, "readings": readings,
            "scope_melodies": list(visible_melodies),
            "conflicts": conflicts, "incomplete": incomplete, "missing": missing,
            "unresolved_regions": unresolved,
            "skipped_regions": skipped,
            "cross_checked": cross_checked,
            "complete": bool(INVENTORY_REGIONS.get(layout)) and bool(visible_melodies)
                        and not (missing or conflicts or incomplete or unresolved)}
