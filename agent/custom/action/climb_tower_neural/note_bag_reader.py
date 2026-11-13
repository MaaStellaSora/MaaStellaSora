"""Read complete rows in the bag's scrolling note-details list.

Images use the same 1280 x 720 client coordinates as ``note_ocr``. The left
list and selected note's title Lv supply counts; effect descriptions do not.
"""
import re
import unicodedata

import cv2

from .note_ocr import OCR_RULES, TEXT_RULE_NODES, combine_readings, find_icons, read_text, text_rules, valid_text


RULES = OCR_RULES["bag_detail"]
BAG_LIST_REGION = tuple(RULES["list_region"])
BAG_ICON_REGION = tuple(RULES["icon_region"])
BAG_TITLE_ICON_REGION = tuple(RULES["title_icon_region"])


def _count_values(records, context=None):
    rules = text_rules(context, "bag_count")
    values = set()
    for record in records:
        text = unicodedata.normalize("NFKC", record.get("text", "")).strip()
        if valid_text(record) and re.fullmatch(rules["zero"], text, re.I):
            values.add(0)
        match = re.fullmatch(rules["count"], text, re.I)
        if match and valid_text(record, OCR_RULES["ocr"]["numeric_confidence"]):
            values.add(int(next(value for value in match.groups() if value is not None)))
    return values


def _read_count(context, image, icon):
    x, y, _, height = icon["box"]
    left, top, width, extra_height = RULES["count_offset"]
    roi = [x + left, y + top, width, height + extra_height]
    raw = read_text(context, image, roi, only_rec=True, node=TEXT_RULE_NODES["bag_count"])
    values = _count_values(raw, context)
    # The smaller grey "not acquired" label can need a tighter crop than Lv.
    # Keep every attempt, including its original OCR text and confidence.
    retry_regions = [[x + left, y + top, width, height]
                     for left, top, width, height in RULES["count_retry_offsets"]]
    for retry_roi in retry_regions:
        if values:
            break
        retried = read_text(context, image, retry_roi, only_rec=True, node=TEXT_RULE_NODES["bag_count"])
        for record in retried:
            record["count_roi"] = retry_roi
        raw.extend(retried)
        values.update(_count_values(retried, context))
    if not values:
        left, top, width, height = retry_regions[-1]
        crop = image[top:top + height, left:left + width]
        scale = RULES["count_retry_scale"]
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        retried = read_text(context, enlarged, (0, 0, width * scale, height * scale),
                            only_rec=True, node=TEXT_RULE_NODES["bag_count"])
        for record in retried:
            record["ocr_box"] = record["box"]
            a, b, w, h = record["box"]
            record["box"] = [left + round(a / scale), top + round(b / scale), round(w / scale), round(h / scale)]
            record.update(count_roi=list(retry_regions[-1]), preprocessing="count_crop_2x")
        raw.extend(retried)
        values.update(_count_values(retried, context))
    return next(iter(values)) if len(values) == 1 else None, raw, roi, sorted(values) if len(values) > 1 else []


def read_bag_note_page(context, image):
    """Return only fully visible icon-anchored rows from one scroll position.

    ``visible_melodies`` describes complete rows, independently of whether their
    count was recognized. Missing or clipped notes never imply a zero count.
    """
    readings, visible = [], []
    stopped = context.tasker.stopping
    icons = [] if stopped else find_icons(context, image, BAG_ICON_REGION,
                                         scales=RULES["icon_scales"], threshold=RULES["icon_threshold"])
    _, top, _, height = BAG_LIST_REGION
    for index, icon in enumerate(icons):
        if context.tasker.stopping:
            stopped = True
            break
        _, y, _, icon_height = icon["box"]
        padding = RULES["row_padding"]
        row_box = [BAG_LIST_REGION[0], y - padding, BAG_LIST_REGION[2], icon_height + padding * 2]
        if row_box[1] < top or row_box[1] + row_box[3] > top + height:
            readings.append({"melody": icon["melody"], "current": None,
                             "source": f"bag_note_detail:{index}", "kind": "skipped",
                             "reason": "clipped_row", "alternative": "overlapping_scroll_page",
                             "box": row_box, "raw": []})
            continue
        current, raw, count_box, conflicts = _read_count(context, image, icon)
        readings.append({"melody": icon["melody"], "current": current,
                         "source": f"bag_note_detail:{index}", "kind": "bag_note_detail",
                         "box": icon["box"], "row_box": row_box, "count_box": count_box,
                         "icon_score": icon["score"], "raw": raw,
                         "conflicting_values": conflicts})
        visible.append(icon["melody"])
    if not context.tasker.stopping:
        title_icons = find_icons(context, image, BAG_TITLE_ICON_REGION,
                                 scales=RULES["title_icon_scales"], threshold=RULES["title_icon_threshold"])
        if title_icons:
            icon = max(title_icons, key=lambda item: item["score"])
            x, y, _, _ = icon["box"]
            # The JP unacquired caption starts above the acquired Lv baseline.
            # Its upper strokes were cut by the former icon_y - 5 crop.
            left, top, width, height = RULES["title_count_offset"]
            count_box = [x + left, y + top, width, height]
            raw = read_text(context, image, count_box, only_rec=True, node=TEXT_RULE_NODES["bag_count"])
            values = _count_values(raw, context)
            readings.append({"melody": icon["melody"],
                             "current": next(iter(values)) if len(values) == 1 else None,
                             "source": "bag_note_title", "kind": "bag_note_title",
                             "box": icon["box"], "count_box": count_box,
                             "icon_score": icon["score"], "raw": raw,
                             "conflicting_values": sorted(values) if len(values) > 1 else []})
    visible = list(dict.fromkeys(visible))
    result = combine_readings(readings, "bag", visible)
    result.update(layout="bag_note_details", visible_melodies=visible,
                  list_region=list(BAG_LIST_REGION), icon_region=list(BAG_ICON_REGION),
                  title_icon_region=list(BAG_TITLE_ICON_REGION),
                  stopped=stopped, complete=result["complete"] and not stopped)
    return result


def merge_bag_note_pages(pages, expected_melodies, *, repeated=()):
    """Cross-check overlapping pages with a caller-owned required scope.

    The caller supplies the fixed tower identities, held extras and actual
    displayed types. Repeated clear regions keep unread values and conflicts.
    """
    readings, visible = [], []
    for page_index, page in enumerate(pages):
        visible.extend(page["visible_melodies"])
        for index, reading in enumerate(page["readings"]):
            source = "bag_note_title" if reading.get("kind") == "bag_note_title" else "bag_note_detail"
            readings.append(dict(reading, source=f"{source}:{page_index}:{index}",
                                 page_index=page_index))
    readings.extend(repeated)
    result = combine_readings(readings, "bag", tuple(dict.fromkeys(expected_melodies)))
    stopped = any(page.get("stopped", False) for page in pages)
    result.update(layout="bag_note_details", visible_melodies=list(dict.fromkeys(visible)),
                  bar_present=bool(visible), stopped=stopped,
                  bar_reason="clear_bag_note_rows" if visible else "no_complete_bag_note_rows",
                  page_count=len(pages), complete=result["complete"] and not stopped)
    return result
