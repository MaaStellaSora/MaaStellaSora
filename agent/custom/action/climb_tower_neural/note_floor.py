"""Read the background floor plaque without inspecting reward content."""
import re

import cv2

from .note_ocr import OCR_RULES, read_text


# Client-area coordinates at 1280 x 720. The inset excludes the plaque border.
RULES = OCR_RULES["floor"]
FLOOR_REGION = tuple(RULES["region"])
FLOOR_DIGITS = tuple(RULES["digits"])
# Only tighten the side margins: trimming vertically removes the blurred lower
# stroke of 6 and makes the native OCR read an otherwise complete 16 as 15.
FLOOR_INNER_DIGITS = tuple(RULES["inner_digits"])
FLOOR_PATTERN = re.compile(RULES["pattern"])


def read_floor_observation(context, image, expected_floors=()):
    """Read the displayed 01–20 floor and retain every OCR observation.

    ``image`` uses Maa's normalized 1280 x 720 client coordinates. The teal
    plaque is dimmed by result overlays. Grayscale and inverted green-channel
    crops expose its digit strokes with two surrounding margins. Confirmation
    requires two numeric readings at >= .80, at least one at >= .85, and no
    conflicting numeric reading at >= .80. When route expectations are given,
    green-channel crops run first and may finish once they confirm an expected
    floor. Unexpected floors use all four crops and retain the observed value.
    Route expectations never supply a missing digit or rewrite OCR text.
    """
    x, y, width, height = FLOOR_REGION
    plaque = image[y:y + height, x:x + width]
    gray = cv2.cvtColor(plaque, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    green = 255 - plaque[:, :, 1]
    expected_floors = list(dict.fromkeys(expected_floors))
    raw = []
    variants = (
        ("gray_normalized_3x", gray, FLOOR_DIGITS),
        ("gray_normalized_inner_3x", gray, FLOOR_INNER_DIGITS),
        ("green_inverted_inner_3x", green, FLOOR_INNER_DIGITS),
        ("green_inverted_3x", green, FLOOR_DIGITS),
    )
    if expected_floors:
        variants = variants[2:] + variants[:2]
    early_completed = False
    for index, (name, plane, region) in enumerate(variants):
        left, top, width, height = region
        digits = plane[top - y:top - y + height, left - x:left - x + width]
        digits = cv2.resize(digits, None, fx=RULES["scale"], fy=RULES["scale"], interpolation=cv2.INTER_CUBIC)
        digits = cv2.cvtColor(digits, cv2.COLOR_GRAY2BGR)
        rows = read_text(context, digits, (0, 0, digits.shape[1], digits.shape[0]), only_rec=True)
        for row in rows:
            # OCR can separate the two seven-segment digits with whitespace.
            # Preserve the raw text; only decimal glyphs can form a floor.
            numeric_text = re.sub(OCR_RULES["ocr"]["whitespace_pattern"], "", row["text"])
            value = int(numeric_text) if FLOOR_PATTERN.fullmatch(numeric_text) else None
            raw.append({**row, "box": list(region), "ocr_box": row["box"],
                        "preprocessing": name, "numeric_text": numeric_text, "floor": value,
                        "accepted": value is not None and row["score"] >= RULES["accepted_confidence"]})
        accepted = [row for row in raw if row["accepted"]]
        floors = sorted({row["floor"] for row in accepted})
        confirmed = (len(floors) == 1
                     and len({row["preprocessing"] for row in accepted}) >= RULES["minimum_variants"]
                     and any(row["score"] >= RULES["confirmed_confidence"] for row in accepted))
        if confirmed and floors[0] in expected_floors:
            early_completed = index + 1 < len(variants)
            break
    floor = floors[0] if confirmed else None
    return {"floor": floor, "observed_floor": floor, "observed_floors": floors,
            "complete": confirmed, "expected_floors": expected_floors,
            "sequence_consistent": floor in expected_floors if floor is not None and expected_floors else None,
            "early_completed": early_completed,
            "method": "floor_hud_ocr_consensus", "source": "floor_hud",
            "box": list(FLOOR_REGION), "coordinate_space": "maa_client_1280x720",
            "raw": raw, "conflicts": floors if len(floors) > 1 else []}


def read_floor(context, image, expected_floors=()):
    """Compatibility API: return the confirmed floor, otherwise None."""
    return read_floor_observation(context, image, expected_floors=expected_floors)["floor"]
