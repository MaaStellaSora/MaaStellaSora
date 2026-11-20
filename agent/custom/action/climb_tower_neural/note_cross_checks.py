"""Read repeated note counts from the visible result and assist panels."""
from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np

from .note_ocr import OCR_RULES, find_icons, read_text, valid_text


RULES = OCR_RULES["cross_checks"]
REQUIREMENT_REGIONS = {key: tuple(map(tuple, regions))
                       for key, regions in RULES["requirement_regions"].items()}
# Hue medians from the existing official icon templates (OpenCV's 0-179 scale).
ICON_HUES = RULES["icon_hues"]


def _text(records):
    ordered = sorted((item for item in records if valid_text(item)),
                     key=lambda item: (item["box"][1], item["box"][0]))
    return unicodedata.normalize("NFKC", " ".join(item["text"] for item in ordered))


def _level(records):
    # A level crop contains only the Lv label and its number.
    numbers = set()
    for record in records:
        text = _text([record])
        if re.fullmatch(RULES["level_zero_pattern"], text):
            numbers.add(0)
        else:
            numbers.update(int(value) for value in re.findall(RULES["level_pattern"], text))
    return next(iter(numbers)) if len(numbers) == 1 else None


def _read_level(context, image, roi):
    raw = read_text(context, image, roi, only_rec=True)
    value = _level(raw)
    if value is None and re.search(RULES["level_label_pattern"], _text(raw)):
        x, y, width, height = roi
        gray = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
        scale = RULES["level_retry_scale"]
        enlarged = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        retry = read_text(context, enlarged, (0, 0, width * scale, height * scale), only_rec=True)
        for item in retry:
            a, b, w, h = item["box"]
            item["ocr_box"] = item["box"]
            item["box"] = [x + round(a / scale), y + round(b / scale), round(w / scale), round(h / scale)]
            item["preprocessing"] = "level_crop_2x"
        raw += retry
        value = _level(raw)
    return value, raw


def _fraction_values(records):
    """Keep every independently read numerator and denominator in the crop."""
    current, required = set(), set()
    for match in re.finditer(RULES["fraction_pattern"], _text(records).replace("∕", "/").replace("⁄", "/")):
        if match[1] is not None:
            current.add(int(match[1]))
        required.add(int(match[2]))
    return current, required


def _skipped(roi, source="assist_requirements:left", reason="dimmed", alternative="inventory"):
    return {"melody": None, "current": None, "kind": "skipped",
            "source": source, "reason": reason, "alternative": alternative,
            "raw": [], "box": list(roi)}


def _result_cards(image):
    """Locate the actual white horizontal rows, including shifted/partial rows."""
    left, top, width, height = RULES["result_region"]
    pixels = image[top:top + height, left:left + width, :3].astype(np.int16)
    white = ((pixels.min(axis=2) > RULES["result_white_minimum"]) &
             (pixels.max(axis=2) - pixels.min(axis=2) < RULES["result_white_channel_spread"])).astype(np.uint8)
    size = RULES["result_morphology_size"]
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(white)
    cards = []
    for x, y, width, height, area in stats[1:]:
        if (width >= RULES["result_minimum_width"]
                and RULES["result_height_range"][0] <= height <= RULES["result_height_range"][1]
                and area > width * height * RULES["result_minimum_fill"]):
            cards.append((int(x + left), int(y + top), int(width), int(height)))
    return sorted(cards, key=lambda box: box[1])


def _result_levels(context, image):
    cards = _result_cards(image)
    icons = find_icons(context, image, RULES["result_icon_region"],
                       scales=RULES["result_icon_scales"], threshold=RULES["result_icon_threshold"])
    rows = []
    for box in cards:
        x, y, width, height = box
        matches = [icon for icon in icons
                   if x <= icon["box"][0] + icon["box"][2] / 2 <= x + RULES["result_icon_anchor_width"]
                   and y <= icon["box"][1] + icon["box"][3] / 2 <= y + height]
        icon = max(matches, key=lambda item: item["score"], default=None)
        level_y = y + RULES["result_level_y_offset"]
        offset, level_width, level_height = RULES["result_previous_level_offset"]
        previous, previous_raw = _read_level(context, image,
                                             (x + width + offset, level_y, level_width, level_height))
        offset, level_width, level_height = RULES["result_current_level_offset"]
        current, current_raw = _read_level(context, image,
                                         (x + width + offset, level_y, level_width, level_height))
        paired = (re.search(RULES["level_label_pattern"], _text(previous_raw)) is not None and
                  re.search(RULES["level_label_pattern"], _text(current_raw)) is not None)
        if icon is None and paired:
            # Two independently read Lv labels establish a note-result row.
            # Its icon can use a lower threshold without searching background.
            left, top, icon_width, trim = RULES["result_retry_icon_offset"]
            matches = find_icons(context, image, (x + left, y + top, icon_width, height + trim),
                                 scales=RULES["result_icon_scales"], threshold=RULES["result_retry_icon_threshold"])
            icon = max(matches, key=lambda item: item["score"], default=None)
        rows.append((box, icon, previous_raw, current_raw, previous, current, paired))
    if not any(paired for _, _, _, _, _, _, paired in rows):
        return []
    readings = []
    for index, (box, icon, previous_raw, current_raw, previous, current, _) in enumerate(rows):
        reading = {
            "melody": icon["melody"] if icon else None,
            "source": f"result_level:{index}", "kind": "result_level",
            "current": current if icon else None,
            "previous": previous if icon else None,
            "raw": previous_raw + current_raw, "box": list(box),
        }
        if icon is None:
            reading.update(kind="skipped", reason="difficult_redundant",
                           alternative="inventory")
        readings.append(reading)
    return readings


def _requirements(context, image, layout):
    readings = []
    if layout == "shop":
        # The purchase dialog has a continuous turquoise header; the ordinary
        # shelf behind it does not expose the same current/required fields.
        x, y, width, height = RULES["shop_header_region"]
        header = image[y:y + height, x:x + width, :3].astype(np.int16)
        minimum = RULES["shop_header_minimum"]
        turquoise = (header[:, :, 0] > minimum) & (header[:, :, 1] > minimum) & (
            np.minimum(header[:, :, 0], header[:, :, 1]) - header[:, :, 2] > RULES["shop_header_channel_difference"])
        if np.mean(turquoise) < RULES["shop_header_minimum_fraction"]:
            return []
    for region_index, roi in enumerate(REQUIREMENT_REGIONS[layout]):
        if layout == "shop" and region_index == 0:
            x, y, width, height = roi
            foreground = image[y:y + min(RULES["shop_assist_header_height"], height), x:x + width, :3].min(axis=2)
            if np.percentile(foreground, RULES["shop_assist_white_percentile"]) < RULES["shop_assist_minimum_white"]:
                readings.append(_skipped(roi, alternative="purchase_confirmation_and_local_note_rules"))
                continue
        panel_raw = read_text(context, image, roi)
        anchors = [item for item in panel_raw if valid_text(item) and
                   re.search(RULES["fraction_anchor_pattern"], unicodedata.normalize("NFKC", item["text"]))]
        # Settlement cards often show only their own Lv, without note counts.
        if layout == "settlement" and not anchors:
            continue
        icons = find_icons(context, image, roi,
                           scales=RULES["requirement_icon_scales"],
                           threshold=RULES["bag_requirement_icon_threshold" if layout == "bag"
                                           else "requirement_icon_threshold"])
        accepted = []
        for icon in icons:
            x, y, width, height = icon["box"]
            center_y = y + height / 2
            near_fraction = any(abs(center_y - item["box"][1] - item["box"][3] / 2) <= RULES["fraction_y_tolerance"] and
                                item["box"][0] - RULES["fraction_left_margin"] <= x <=
                                item["box"][0] + item["box"][2]
                                for item in anchors)
            patch = image[y:y + height, x:x + width, :3]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hue = ICON_HUES[icon["melody"].removeprefix("melody_of_")]
            distance = np.abs(hsv[:, :, 0].astype(np.int16) - hue)
            distance = np.minimum(distance, 180 - distance)
            colored = np.mean((hsv[:, :, 1] > RULES["icon_minimum_saturation"]) &
                              (hsv[:, :, 2] > RULES["icon_minimum_value"]) &
                              (distance <= RULES["icon_hue_tolerance"]))
            # Small white denominator digits and dark title glyphs can resemble
            # templates after scaling, but do not contain a colored note icon.
            if near_fraction and colored > RULES["icon_minimum_fraction"]:
                accepted.append(icon)
        for index, icon in enumerate(accepted):
            x, y, width, height = icon["box"]
            center_y = y + height / 2
            next_x = min((other["box"][0] for other in accepted
                          if other["box"][0] > x + width and abs(
                              other["box"][1] + other["box"][3] / 2 - center_y) < RULES["adjacent_icon_y_tolerance"]),
                         default=x + width + RULES["count_maximum_width"])
            right = min(x + width + RULES["count_maximum_width"], next_x - RULES["count_right_margin"])
            padding, offset = RULES["count_y_padding"], RULES["count_x_offset"]
            digit_roi = (x + width + offset, max(0, y - padding),
                         max(RULES["count_minimum_width"], right - x - width - offset), height + padding * 2)
            raw = read_text(context, image, digit_roi, only_rec=True)
            current, required = _fraction_values(raw)
            if not required:
                retry = read_text(context, image, digit_roi)
                raw += retry
                current, required = _fraction_values(raw)
            readings.append({
                "melody": icon["melody"],
                "source": f"requirement:{region_index}:{index}", "kind": "requirement",
                "current": next(iter(current)) if len(current) == 1 else None,
                "required": next(iter(required)) if len(required) == 1 else None,
                "conflicting_values": sorted(current) if len(current) > 1 else [],
                "conflicting_required_values": sorted(required) if len(required) > 1 else [],
                "raw": raw, "box": [x, max(0, y - padding), right - x, height + padding * 2],
            })
        for index, item in enumerate(anchors):
            x, y, width, height = item["box"]
            fractions = len(re.findall(RULES["fraction_anchor_pattern"],
                                       unicodedata.normalize("NFKC", item["text"])))
            tolerance = RULES["fraction_y_tolerance"]
            matched = sum(x - RULES["fraction_left_margin"] <= icon["box"][0] <= x + width and
                          y - tolerance <= icon["box"][1] + icon["box"][3] / 2 <= y + height + tolerance
                          for icon in accepted)
            for missing in range(max(0, fractions - matched)):
                readings.append({
                    "melody": None, "source": f"requirement_unresolved:{region_index}:{index}:{missing}",
                    "kind": "requirement", "current": None, "required": None,
                    "raw": [item], "box": item["box"],
                })
    return readings


def collect_cross_checks(context, image, layout):
    """Keep each visible repeated position, including unreadable count values.

    Assist cards showing only their own Lv have no note requirement icons, so
    their card level is never interpreted as note inventory.
    """
    if layout == "result":
        levels = _result_levels(context, image)
        return levels or [_skipped(RULES["result_skipped_region"], "assist_results",
                                   "partial_assist_list", "inventory_and_local_note_rules")]
    if layout == "event":
        return [_skipped(REQUIREMENT_REGIONS[layout][0])]
    if layout in REQUIREMENT_REGIONS:
        return _requirements(context, image, layout)
    return []
