"""Read starting supply and the bag's main discs and concerto requirements."""
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
import json
from pathlib import Path
import re
from statistics import median
import unicodedata

import cv2

from .name_matching import clean_name, match_name
from .note_cross_checks import collect_cross_checks
from .note_ocr import (OCR_RULES, TEXT_RULE_NODES, find_icons, read_text, text_rules,
                       valid_text, white_glyph_image)


SETUP_RULES, CONCERTO_RULES = OCR_RULES["setup"], OCR_RULES["concerto"]
MAIN_NAME_REGIONS = tuple(map(tuple, SETUP_RULES["main_name_regions"]))
SUPPLY_REGION = tuple(SETUP_RULES["supply_region"])
SETTLEMENT_CARD_REGIONS = tuple(map(tuple, CONCERTO_RULES["settlement_card_regions"]))
BAG_MAIN_NAME_REGIONS = tuple(map(tuple, CONCERTO_RULES["main_name_regions"]))
BAG_CONCERTO_REGIONS = tuple(map(tuple, CONCERTO_RULES["card_regions"]))


@lru_cache(maxsize=1)
def load_rules():
    """The shipped catalogue links CN/JP disc names to local concerto needs."""
    return json.loads(Path(__file__).with_name("note_rules.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _disc_aliases():
    aliases = defaultdict(list)
    for disc in load_rules()["discs"]:
        for name in (disc["name_ja"], disc["name_zh"], *disc.get("aliases", ())):
            key = clean_name(name)
            if key and disc not in aliases[key]:
                aliases[key].append(disc)
    return dict(aliases)


def resolve_disc_name(text):
    """Apply the existing name matcher, leaving weak or tied names unknown."""
    aliases = _disc_aliases()
    match = match_name(text, aliases)
    ids = {row["id"] for row in match.items}
    key = clean_name(text)
    accepted = len(ids) == 1
    if match.method == "closest_name":
        rules = OCR_RULES["disc_match"]
        threshold = rules["long_name_threshold"] if len(key) >= rules["long_name_minimum_length"] else rules["short_name_threshold"]
        rivals = [SequenceMatcher(None, key, alias).ratio()
                  for alias, rows in aliases.items()
                  if any(row["id"] not in ids for row in rows)]
        accepted = accepted and match.similarity >= threshold and (
            match.similarity - max(rivals, default=0.) >= rules["minimum_margin"])
    return (match.items[0] if accepted else None), match.method, match.similarity


def _read_main_disc(context, image, slot, roi):
    raw = read_text(context, image, roi)
    text = "".join(row["text"] for row in sorted(raw, key=lambda row: row["box"][0]))
    complete = bool(text) and all(valid_text(row) for row in raw)
    disc, method, similarity = resolve_disc_name(text) if complete else (None, "unconfirmed", 0.)
    return {"slot": slot, "id": disc["id"] if disc else None,
            "name_ja": disc["name_ja"] if disc else None,
            "name_zh": disc["name_zh"] if disc else None,
            "secondary_group_ids": disc["secondary_group_ids"] if disc else [],
            "status": "confirmed" if disc else "unmapped" if complete else "unknown",
            "mapping_status": "catalogue" if disc else "unmapped",
            "complete": complete,
            "text": text, "box": list(roi), "raw": raw,
            "match_method": method, "similarity": similarity}


def _numeric_values(raw):
    return {int(row["text"].strip()) for row in raw
            if valid_text(row, OCR_RULES["ocr"]["numeric_confidence"])
            and re.fullmatch(OCR_RULES["ocr"]["numeric_pattern"], row["text"].strip())}


def _read_supply_count(context, image, icon):
    x, y, width, height = icon["box"]
    rules = SETUP_RULES
    roi = (x + width, y + rules["count_y_offset"],
           min(rules["count_maximum_width"], rules["count_right"] - x - width),
           max(rules["count_minimum_height"], height - rules["count_height_trim"]))
    raw = read_text(context, image, roi, only_rec=True, expected=(OCR_RULES["ocr"]["numeric_pattern"],))
    values = _numeric_values(raw)
    if len(values) != 1:
        enlarged = white_glyph_image(image, roi)
        if enlarged is not None:
            retry = read_text(context, enlarged, (0, 0, enlarged.shape[1], enlarged.shape[0]),
                              only_rec=True, expected=(OCR_RULES["ocr"]["numeric_pattern"],))
            for row in retry:
                row.update(box=list(roi), preprocessing="white_glyphs_3x")
            raw.extend(retry)
            values.update(_numeric_values(retry))
    return (next(iter(values)) if len(values) == 1 else None), raw, list(roi)


def _supply_rows(image, readings):
    """Use the numbered rows to locate the column; keep unread numbers in it."""
    anchored = [row["box"] for row in readings if row["current"] is not None]
    if len(anchored) < 2 or len(anchored) == len(readings):
        return readings, []
    column = median(x + w / 2 for x, _, w, _ in anchored)
    tolerance = median(w for _, _, w, _ in anchored) / SETUP_RULES["column_tolerance_divisor"]
    rows, skipped = [], []
    for row in readings:
        x, _, w, _ = row["box"]
        if row["current"] is None and abs(x + w / 2 - column) > tolerance:
            left, top, width, height = row["count_box"]
            glyphs = cv2.inRange(image[top:top + height, left:left + width],
                                 tuple(OCR_RULES["white_glyphs"]["lower"]),
                                 tuple(OCR_RULES["white_glyphs"]["upper"]))
            if not cv2.countNonZero(glyphs):
                skipped.append(dict(row, reason="outside_supply_column_without_number"))
                continue
        rows.append(row)
    return rows, skipped


def read_setup(context, image, *, read_discs=True):
    """Read a 1280x720 setup image without inspecting auxiliary disc cards.

    ``starting_supply`` contains only displayed rows. Initial zeros for absent
    rows belong to the new-run initialization rule, outside this OCR reader.
    """
    main_discs = ([_read_main_disc(context, image, slot, roi)
                   for slot, roi in enumerate(MAIN_NAME_REGIONS, 1)] if read_discs else [])
    readings = []
    for icon in find_icons(context, image, SUPPLY_REGION,
                           scales=SETUP_RULES["icon_scales"], threshold=SETUP_RULES["icon_threshold"]):
        current, raw, roi = _read_supply_count(context, image, icon)
        values = _numeric_values(raw)
        readings.append({"melody": icon["melody"], "current": current,
                         "source": "starting_supply_total", "box": icon["box"],
                         "icon_score": icon["score"], "count_box": roi, "raw": raw,
                         "conflicting_values": sorted(values) if len(values) > 1 else []})
    readings, skipped = _supply_rows(image, readings)
    by_melody = defaultdict(list)
    for reading in readings:
        by_melody[reading["melody"]].append(reading["current"])
    conflicts = [melody for melody, values in by_melody.items()
                 if len(set(values)) > 1 or any(row["conflicting_values"]
                     for row in readings if row["melody"] == melody)]
    starting_supply = {melody: values[0] if melody not in conflicts else None
                       for melody, values in by_melody.items()}
    unknown_slots = [row["slot"] for row in main_discs if row["id"] is None]
    supply_complete = bool(readings) and all(row["current"] is not None for row in readings) and not conflicts
    return {"main_discs": main_discs, "starting_supply": starting_supply,
            "readings": readings, "raw": {"main_discs": [row["raw"] for row in main_discs],
                                            "starting_supply": readings},
            "unknown_slots": unknown_slots, "conflicts": conflicts, "skipped_regions": skipped,
            "supply_complete": supply_complete, "complete": supply_complete and not unknown_slots}


def _concerto_level(raw, context=None):
    rules = text_rules(context, "concerto_level")
    values = set()
    for row in raw:
        if not valid_text(row):
            continue
        text = unicodedata.normalize("NFKC", row["text"]).strip()
        if re.fullmatch(rules["inactive"], text, re.I):
            values.add(0)
        elif match := re.fullmatch(rules["level"], text, re.I):
            values.add(int(match[1]))
        elif re.fullmatch(rules["maximum"], text, re.I):
            values.add(CONCERTO_RULES["maximum_level"])
    return next(iter(values)) if len(values) == 1 else None


def _card_icon_present(image, box):
    x, y, _, _ = box
    left, top, width, height = CONCERTO_RULES["icon_offset"]
    icon = cv2.cvtColor(image[y + top:y + top + height, x + left:x + left + width], cv2.COLOR_BGR2HSV)
    low, high = CONCERTO_RULES["icon_hue_range"]
    colored = (icon[:, :, 0] >= low) & (icon[:, :, 0] <= high) & (
        icon[:, :, 1] > CONCERTO_RULES["icon_minimum_saturation"]) & (
        icon[:, :, 2] > CONCERTO_RULES["icon_minimum_value"])
    return float(colored.mean()) > CONCERTO_RULES["icon_minimum_fraction"]


def _catalogue_order(main_discs):
    if len(main_discs) != 3 or any(row.get("id") is None for row in main_discs):
        return None
    return [dict(group_id=group, disc_slot=disc["slot"])
            for disc in main_discs for group in disc["secondary_group_ids"]]


def _known_card_mapping(main_discs, cards):
    """Map known prefix/suffix groups without inventing an unknown disc's size."""
    if len(main_discs) != 3 or [row["slot"] for row in cards] != list(range(1, len(cards) + 1)):
        return {}
    unknown = [index for index, row in enumerate(main_discs) if row.get("id") is None]
    if not unknown:
        order = _catalogue_order(main_discs)
        return {row["slot"]: mapped for row, mapped in zip(cards, order)} if len(cards) == len(order) else {}
    prefix = [dict(group_id=group, disc_slot=disc["slot"])
              for disc in main_discs[:unknown[0]] for group in disc["secondary_group_ids"]]
    suffix = [dict(group_id=group, disc_slot=disc["slot"])
              for disc in main_discs[unknown[-1] + 1:] for group in disc["secondary_group_ids"]]
    if len(prefix) + len(suffix) > len(cards):
        return {}
    return {**{row["slot"]: mapped for row, mapped in zip(cards, prefix)},
            **{row["slot"]: mapped for row, mapped in zip(cards[len(cards) - len(suffix):], suffix)}}


def read_bag_skill_page(context, image):
    """Read names, actual levels and requirements; concerto names are unused.

    The game packs groups consecutively, so an unmapped main disc leaves group
    ownership positional. Unknown groups keep only their displayed level and
    next-level requirements. Known groups obtain all other needs from the
    catalogue. Max-level cards hide their requirement row.
    """
    main_discs = [_read_main_disc(context, image, slot, roi)
                  for slot, roi in enumerate(BAG_MAIN_NAME_REGIONS, 1)]
    order = _catalogue_order(main_discs)
    requirements = collect_cross_checks(context, image, "bag")
    cards, empty_slots = [], []
    for slot, box in enumerate(BAG_CONCERTO_REGIONS, 1):
        if context.tasker.stopping:
            break
        x, y, width, height = box
        # The active level shares a baseline with the title. Its right-hand
        # field also contains the inactive pill, without the title's last glyph.
        left, top, trim, level_height = CONCERTO_RULES["level_offset"]
        level_box = [x + left, y + top, width + trim, level_height]
        level_raw = read_text(context, image, level_box, node=TEXT_RULE_NODES["concerto_level"])
        level = _concerto_level(level_raw, context)
        readings = [row for row in requirements if x <= row["box"][0] < x + width
                    and y <= row["box"][1] + row["box"][3] / 2 < y + height]
        expected = order is not None and slot <= len(order)
        if not (expected or level_raw or readings or _card_icon_present(image, box)):
            empty_slots.append(slot)
            continue
        needs, conflicts = {}, []
        for row in readings:
            melody, required = row.get("melody"), row.get("required")
            if melody is None:
                continue
            if row.get("conflicting_required_values"):
                conflicts.append(melody)
            if required is None:
                continue
            if melody in needs and needs[melody] != required:
                conflicts.append(melody)
            needs[melody] = required
        for melody in conflicts:
            needs[melody] = None
        requirements_complete = (bool(needs) and not conflicts
            and all(row.get("melody") and row.get("required") is not None for row in readings))
        maximum = CONCERTO_RULES["maximum_level"]
        cards.append(dict(slot=slot, group_id=None, disc_slot=None, mapping_status="position",
            level=level, source="bag_skill", box=list(box), level_box=level_box,
            displayed_need=needs, requirements=readings, requirements_complete=requirements_complete,
            displayed_need_level=level + 1 if level is not None and level < maximum else None,
            first_level_need=None, first_level_source=None,
            first_level_basis=None, first_level_evidence=None,
            maximum_level=maximum, requirement_conflicts=list(dict.fromkeys(conflicts)),
            raw={"level": level_raw}, complete=level is not None and (level == maximum or requirements_complete)))
    mapping = _known_card_mapping(main_discs, cards)
    for card in cards:
        if mapped := mapping.get(card["slot"]):
            card.update(mapped, mapping_status="catalogue")
            levels = load_rules()["concerto_groups"][str(card["group_id"])]
            card.update(first_level_need=dict(levels[0]["need"]),
                        first_level_source="local_rules", first_level_basis="local_rules_derivation",
                        first_level_evidence=dict(group_id=card["group_id"], source="local_rules"))
    names_complete = all(row["complete"] for row in main_discs)
    scope_complete = bool(cards) or order == []
    return dict(main_discs=main_discs, concertos=cards, empty_slots=empty_slots,
                requirement_readings=requirements,
                unmapped_disc_slots=[row["slot"] for row in main_discs if row["id"] is None],
                names_complete=names_complete, concertos_complete=scope_complete and all(row["complete"] for row in cards),
                mapping_complete=all(row["group_id"] is not None for row in cards),
                complete=names_complete and scope_complete and all(row["complete"] for row in cards)
                    and not context.tasker.stopping, page="bag_skill")


def read_settlement(context, image, concerto_order=None):
    """Read the final score and six visible actual concerto levels."""
    concertos = []
    for slot, (x, y, width, height) in enumerate(SETTLEMENT_CARD_REGIONS, 1):
        if concerto_order is not None:
            previous = next((row for row in concerto_order if row["slot"] == slot), None)
            if previous is None:
                continue
            offset = CONCERTO_RULES["settlement_level_offset_y"]
            levels = read_text(context, image, (x, y + offset,
                CONCERTO_RULES["settlement_level_width"], height - offset), node=TEXT_RULE_NODES["concerto_level"])
            level = _concerto_level(levels, context)
            concertos.append(dict(slot=slot, name=None, level=level,
                group_id=previous.get("group_id"), disc_slot=previous.get("disc_slot"),
                mapping_status=previous.get("mapping_status", "position"),
                source="settlement", box=[x, y, width, height], raw={"level": levels},
                complete=level is not None))
            continue
        names = read_text(context, image, (x, y, width, CONCERTO_RULES["settlement_title_height"]))
        offset = CONCERTO_RULES["settlement_level_offset_y"]
        levels = read_text(context, image, (x, y + offset,
            CONCERTO_RULES["settlement_cold_level_width"], height - offset), node=TEXT_RULE_NODES["concerto_level"])
        name = "".join(row["text"] for row in sorted(names, key=lambda row: row["box"][0]))
        name_complete = bool(name) and all(valid_text(row) for row in names)
        level = _concerto_level(levels, context)
        groups = [int(gid) for gid, names in load_rules()["concerto_names"].items()
                  if name_complete and clean_name(name) in {clean_name(value) for value in names.values() if value}]
        group_id = groups[0] if len(groups) == 1 else None
        concertos.append({"slot": slot, "name": name, "level": level,
                          "group_id": group_id,
                          "mapping_status": "catalogue" if group_id is not None else "unmapped",
                          "source": "settlement", "box": [x, y, width, height],
                          "raw": {"name": names, "level": levels},
                          "complete": name_complete and level is not None})
    score_raw = read_text(context, image, CONCERTO_RULES["settlement_score_region"], only_rec=True,
                          expected=(OCR_RULES["ocr"]["numeric_pattern"],))
    values = _numeric_values(score_raw)
    score = next(iter(values)) if len(values) == 1 else None
    return {"actual_concertos": concertos, "score": score, "score_raw": score_raw,
            "mapping_complete": all(row.get("group_id") is not None for row in concertos),
            "complete": score is not None and all(row["complete"] for row in concertos)}
