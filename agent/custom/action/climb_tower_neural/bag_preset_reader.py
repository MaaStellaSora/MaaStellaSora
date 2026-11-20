"""Read the open tower bag once for targets and actual potential inventory.

The game lists the main character first, then its two supports. Each member
has two five-card specialization rows and six universal cards: four cores,
three rare and nine common ordinary potentials. The bag's display order is
recorded as such; it is not claimed to be the game's hidden draw-pool order.
This module neither opens nor closes the bag and never writes a catalogue.
"""
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import cv2
import numpy as np

from .interactor import Interactor
from .ocr_number import assess_amount

from .name_matching import clean_name
from .note_ocr import read_text
from .ui_flow import read_until, wait_for


HEADER_NODE = "星塔_背包_潜能标记_agent"
THUMB_NODE = "星塔_背包_识别推荐潜能_agent"
CLICK_NODE = "星塔_神经网络_预设点击"
SCROLL_NODE = "星塔_神经网络_背包潜能_向下"
SCROLL_TOP_NODE = "星塔_神经网络_背包潜能_置顶"
LIST_BOX = (352, 95, 592, 591)
COLUMNS = (358, 475, 593, 710, 828)
ROW_OFFSETS = (47, 211, 377, 526)
ROW_CARD_COUNTS = (5, 5, 5, 1)
TEXT_RULES = json.loads(Path(__file__).with_name("bag_preset_rules.json").read_text(encoding="utf-8"))
CONTROLS = TEXT_RULES["controls"]
UNACQUIRED = re.compile(TEXT_RULES["unacquired_pattern"])
OBTAINED_COUNT = re.compile(TEXT_RULES["obtained_count_pattern"])
CLEAR_NAME_SCORE = .98


def _inside(row, box):
    x, y, width, height = box
    a, b, w, h = row["box"]
    return x <= a + w / 2 < x + width and y <= b + h / 2 < y + height


def _text_lines(rows):
    lines = []
    for row in sorted(rows, key=lambda item: (item["box"][1], item["box"][0])):
        if not lines or abs(row["box"][1] - lines[-1][0]["box"][1]) > 9:
            lines.append([row])
        else:
            lines[-1].append(row)
    return lines


def _joined(rows):
    return "".join("".join(row["text"] for row in sorted(line, key=lambda item: item["box"][0]))
                   for line in _text_lines(rows))


def _matches(context, node, image, box, threshold):
    detail = context.run_recognition(node, image, {node: {"recognition": {"param": {
        "roi": list(box), "threshold": threshold, "order_by": "Vertical"}}}})
    return [dict(box=list(row.box), score=float(row.score)) for row in detail.filtered_results
            if row.score >= threshold] if detail else []


def _knob(image, box):
    x, y, width, height = box
    hsv = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 210), (179, 75, 255))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= height * height * .13
                  and stats[i, cv2.CC_STAT_HEIGHT] >= height * .55]
    if not candidates:
        raise ValueError("背包潜能筛选开关未读清")
    index = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    a, b = centers[index]
    left, top, found_width, found_height, _ = stats[index]
    return dict(right=bool(a > width / 2), center=(float(x + a), float(y + b)),
                box=[x + int(left), y + int(top), int(found_width), int(found_height)])


def _knob_right(image, box):
    return _knob(image, box)["right"]


def _unacquired(image, box, texts):
    x, y, width, _ = box
    badge = (x + 5, y + 76, width - 10, 36)
    raw = [row for row in texts if _inside(row, badge)]
    if any(UNACQUIRED.search(clean_name(row["text"])) for row in raw):
        return True, raw
    # The unacquired badge is a dark blue capsule, independently of the
    # darkened artwork and the green four-corner selection frame.
    crop = image[y + 85:y + 102, x + 22:x + width - 20].astype(np.int16)
    blue = ((crop[:, :, 0] > crop[:, :, 1] + 12) & (crop[:, :, 1] > crop[:, :, 2] + 8)
            & (crop[:, :, 0] < 150) & (crop[:, :, 2] < 100))
    return bool(blue.size and blue.mean() > .48), raw


def _rarity_from_background(image, box):
    """Acquired cards move forward, so read pink/rainbow/gold artwork itself."""
    x, y, width, _ = box
    roi = [x + 8, y + 8, width - 16, 66]
    a, b, w, h = roi
    hsv = cv2.cvtColor(image[b:b + h, a:a + w], cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    colored = (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 45)
    fractions = {kind: float(np.mean(colored & band)) for kind, band in (
        ("core", (hue >= 140) & (hue <= 179)),
        ("rare", (hue >= 80) & (hue < 140)),
        ("common", (hue >= 4) & (hue <= 50)))}
    rarity = max(fractions, key=fractions.get)
    if fractions[rarity] < .25:
        raise ValueError("背包潜能品质背景未读清")
    return rarity, dict(box=roi, background_fractions=fractions)


def _content_shift(previous, current):
    """Recover the actual scroll displacement from overlapping list artwork."""
    _, y, _, height = LIST_BOX
    # The final universal row has only its first card. Other columns expose
    # the animated scene through the bag, so they cannot anchor the scroll.
    x, width = COLUMNS[0], 118
    old = cv2.cvtColor(previous[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(current[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    matches = []
    for top in (55, 205, 355, 455):
        strip = old[top:top + 95]
        result = cv2.matchTemplate(new, strip, cv2.TM_CCOEFF_NORMED)
        _, score, _, (_, found_y) = cv2.minMaxLoc(result)
        matches.append((score, found_y - top))
    score, shift = max(matches)
    if score < .75:
        raise ValueError("背包潜能滚动画面未能衔接")
    return shift


def _reflow_name(crop):
    """Put two visible title lines side by side without changing their pixels.

    A lone trailing kana is unreliable when recognized on its own. Recognize
    it together with the preceding line, while retaining both source regions.
    The bottom card outline and the green selection corners are excluded.
    """
    title = crop[:33, 3:-3]
    pixels = title.astype(np.int16)
    ink = ((pixels[:, :, 0] > pixels[:, :, 2] + 10)
           & (pixels[:, :, 0] > pixels[:, :, 1] + 2)
           & (pixels[:, :, 0] < 120) & (pixels[:, :, 1] < 100)
           & (pixels[:, :, 2] < 70))
    # Acquired cards use neutral dark lettering with a white outline on a
    # bright color band; unacquired cards use blue lettering on gray.
    if not ink.any():
        ink = pixels.max(axis=2) < 145
    # The title crop can overlap a badge above or the card outline below.
    # Their wide clipped components and thin clipped lower corners are
    # artwork, not another text line.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    for index in range(1, count):
        _, top, width, height, _ = stats[index]
        if ((width >= ink.shape[1] * .5 and (top <= 1 or top + height >= ink.shape[0] - 1))
                or (height <= 2 and top + height == ink.shape[0])):
            ink[labels == index] = False
    projection = ink.sum(axis=1)
    gaps = [y for y in range(12, 25) if projection[y] == 0
            and projection[:y].sum() > 40 and projection[y + 1:].sum() > 2]
    if not gaps:
        return None
    split = round(float(np.median(gaps)))
    lines, regions = [], []
    for top, bottom in ((0, split), (split, 33)):
        _, xx = np.where(ink[top:bottom])
        if not len(xx):
            return None
        left, right = max(0, int(xx.min()) - 3), min(title.shape[1], int(xx.max()) + 4)
        lines.append(title[top:bottom, left:right])
        regions.append([left + 3, top, right - left, bottom - top])
    strips = []
    height = max(line.shape[0] for line in lines) + 4
    for line in lines:
        strip = np.full((height, line.shape[1], 3), 255, dtype=np.uint8)
        strip[height - line.shape[0]:] = line
        strips.append(strip)
    return np.concatenate(strips, axis=1), regions


@dataclass
class _Card:
    rarity: str
    index: int
    readings: list = field(default_factory=list)
    levels: set = field(default_factory=set)
    targets: set = field(default_factory=set)
    rarities: set = field(default_factory=set)
    observations: list = field(default_factory=list)
    crop: object = None
    crop_box: list | None = None
    crop_frame: int = 0
    name_evidence: dict | None = None

    def name(self):
        if self.name_evidence is not None:
            return self.name_evidence["text"], True
        return self.readings[0]["text"] if self.readings else "", False

    def missing(self):
        return [name for name, complete in (
            ("rarity", len(self.rarities) == 1),
            ("level", len(self.levels) == 1),
            ("target_level", len(self.targets) == 1),
            ("name", self.name()[1])) if not complete]


class _Reader:
    def __init__(self, context, language, capture):
        self.context, self.capture = context, capture
        self.language = language
        self.frame = 0
        self.offset = 0
        self.members = []
        self.display_conditions = {}
        self.missing = []

    def screenshot(self):
        return self._record_frame(Interactor.capture(self.context))

    def _record_frame(self, image):
        if self.context.tasker.stopping:
            raise ValueError("任务已停止")
        if image is None or not image.size:
            raise ValueError("背包潜能截图缺失")
        self.capture_size = image.shape[1], image.shape[0]
        if image.shape[:2] != (720, 1280):
            image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_CUBIC)
        self.frame += 1
        if self.capture is not None:
            self.capture("preset_screenshot", image, frame=self.frame, source_page="bag_potential_collection")
        return image

    def _read_until(self, observe, image=None):
        def read(frame):
            if frame is not image:
                frame = self._record_frame(frame)
            return observe(frame)

        return read_until(self.context, read, initial_image=image)

    def click(self, x, y):
        width, height = self.capture_size
        result = self.context.run_task(CLICK_NODE, {CLICK_NODE: {"action": {"param": {
            "target": [round(x * width / 1280), round(y * height / 720), 1, 1]}}}})
        if not result or not result.status.succeeded:
            raise ValueError("背包潜能页面点击未执行")

    def scroll(self, upward=True):
        width, height = self.capture_size
        node = SCROLL_NODE if upward else SCROLL_TOP_NODE
        overrides = {}
        if (width, height) != (1280, 720):
            params = self.context.get_node_data(node)["action"]["param"]
            points = {name: [round(params[name][0] * width / 1280),
                             round(params[name][1] * height / 720)] for name in ("begin", "end")}
            overrides[node] = {"action": {"param": points}}
        result = self.context.run_task(node, overrides)
        if not result or not result.status.succeeded:
            raise ValueError("背包潜能滚动未执行")

    def headers(self, image):
        return _matches(self.context, HEADER_NODE, image, (830, 80, 115, 608), .8)

    def _ensure_control(self, image, name):
        control = CONTROLS[name]

        def locate(frame):
            try:
                return frame, _knob(frame, control["roi"])
            except ValueError:
                return None

        located = self._read_until(locate, image)
        if located is None:
            raise ValueError(f"背包潜能显示开关未读清：{name}")
        image, knob = located
        if knob["right"] == control["right"]:
            return image
        self.click(*knob["center"])
        latest = image

        def observe(frame):
            nonlocal latest
            latest = self._record_frame(frame)
            try:
                knob = _knob(latest, control["roi"])
            except ValueError:
                return None
            return (latest,) if knob["right"] == control["right"] else None

        def retry(_frame):
            try:
                knob = _knob(latest, control["roi"])
            except ValueError:
                return False
            if knob["right"] == control["right"]:
                return False
            self.click(*knob["center"])
            return True

        result = wait_for(self.context, observe, timeout=3.0, retry=retry)
        if result is None:
            raise ValueError(f"背包潜能显示开关未确认：{name}")
        return result[0]

    def _confirm_display(self, image):
        controls = {name: dict(_knob(image, control["roi"]), roi=control["roi"])
                    for name, control in CONTROLS.items()}
        if any(controls[name]["right"] != control["right"] for name, control in CONTROLS.items()):
            raise ValueError("背包潜能全部卡片及推荐显示条件未确认")
        self.display_conditions[self.frame] = dict(frame=self.frame,
            basis="observed_white_knob_positions", controls=controls)

    def prepare(self):
        image = self.screenshot()

        def locate(frame):
            if self.headers(frame):
                return frame, None
            from .note_ocr import OCR_RULES, TEXT_NODE

            detail = self.context.run_recognition(TEXT_NODE, frame, {
                TEXT_NODE: {"recognition": {"type": "OCR", "param": {
                    "roi": [5, 95, 255, 90], "only_rec": False,
                    "threshold": OCR_RULES["ocr"]["threshold"], "replace": [],
                    "expected": TEXT_RULES["collection_tab_patterns"][self.language]}}}})
            rows = detail.filtered_results if detail and detail.hit else []
            if not rows:
                return None
            return frame, max(rows, key=lambda row: row.score).box

        located = self._read_until(locate, image)
        if located is None:
            raise ValueError("未找到背包潜能收集页签")
        image, tab = located
        if tab is not None:
            x, y, width, height = tab
            self.click(x + width / 2, y + height / 2)
            opened = self._read_until(lambda frame: (frame,) if self.headers(frame) else None)
            if opened is None:
                raise ValueError("背包潜能收集页未确认")
            image = opened[0]
        for name in CONTROLS:
            image = self._ensure_control(image, name)
        self._confirm_display(image)
        for _ in range(7):
            hsv = cv2.cvtColor(image[80:688, 947:954], cv2.COLOR_BGR2HSV)
            yy, _ = np.where(cv2.inRange(hsv, (75, 100, 90), (105, 255, 255)) > 0)
            if len(yy) and yy.min() < 12:
                break
            self.scroll(upward=False)
            following = self.screenshot()
            if np.mean(np.abs(following[95:686, 352:944].astype(np.int16)
                              - image[95:686, 352:944].astype(np.int16))) < 1:
                image = following
                break
            image = following
        return image

    def _digit(self, image, box, maximum):
        x, y, width, height = box
        crop = cv2.resize(image[y:y + height, x:x + width], None, fx=3, fy=3,
                          interpolation=cv2.INTER_CUBIC)
        pattern = TEXT_RULES["single_digit_pattern"].format(maximum=maximum)
        rows = read_text(self.context, crop, (0, 0, width * 3, height * 3),
                         only_rec=True, expected=(pattern,))
        for row in rows:
            a, b, w, h = row["box"]
            row["box"] = [x + round(a / 3), y + round(b / 3), round(w / 3), round(h / 3)]
            row["source_roi"] = list(box)
        value, _, _ = assess_amount(
            (row["text"] for row in rows if row["score"] >= .8
             and re.fullmatch(pattern, row["text"])),
            ((row["text"], row["score"]) for row in rows))
        return value if value is not None and 0 <= value <= maximum else None, rows

    def feed(self, image):
        self._confirm_display(image)
        missing = []
        visible = 0
        texts = read_text(self.context, image, (295, 80, 649, 608))
        for header in self.headers(image):
            x, y, _, _ = header["box"]
            position = y + self.offset
            member = next((row for row in self.members if abs(row["position"] - position) < 12), None)
            if member is None:
                member = dict(position=position, name="", obtained_count=None, header_readings=[],
                              obtained_readings=[],
                              cards=[_Card("", index) for index in range(sum(ROW_CARD_COUNTS))])
                self.members.append(member)
                self.members.sort(key=lambda row: row["position"])
            names = [row for row in texts if _inside(row, (295, y - 10, 460, 35)) and row["score"] >= .65]
            if not names:
                names = read_text(self.context, image, (295, max(80, y - 10), 460, 35))
            name = _joined(names)
            if len(clean_name(name)) > len(clean_name(member["name"])):
                member["name"] = name
            if names:
                reading = names[0] if len(names) == 1 else dict(
                    text=name, score=min(row["score"] for row in names), fragments=names)
                member["header_readings"].append(dict(reading, frame=self.frame))
            if not member["name"]:
                missing.append(f"member@{position}:name")
            if member["obtained_count"] is not None:
                continue
            count_rows = [row for row in texts if _inside(row, (x + 14, y - 10, 78, 35))]
            member["obtained_readings"].extend(dict(row, frame=self.frame) for row in count_rows)
            totals = {int(match[1]) for row in count_rows if row["score"] >= .85
                      and (match := OBTAINED_COUNT.fullmatch(row["text"]))}
            if not totals:
                count_rows = read_text(self.context, image, (x + 14, y - 10, 78, 35), only_rec=True,
                                       expected=(TEXT_RULES["obtained_count_pattern"],))
                member["obtained_readings"].extend(dict(row, frame=self.frame,
                    source="obtained_count_crop") for row in count_rows)
                totals = {int(match[1]) for row in count_rows if row["score"] >= .85
                          and (match := OBTAINED_COUNT.fullmatch(row["text"]))}
            if len(totals) == 1:
                member["obtained_count"] = totals.pop()
            else:
                missing.append(f"member@{position}:obtained_count")
        thumbs = _matches(self.context, THUMB_NODE, image, LIST_BOX, .7)
        for member in self.members:
            origin = member["position"] - self.offset
            index = 0
            for relative, count in zip(ROW_OFFSETS, ROW_CARD_COUNTS):
                top = round(origin + relative)
                for column in range(count):
                    if self.context.tasker.stopping:
                        return ["task_stopped"]
                    card = member["cards"][index]
                    index += 1
                    if top < 98 or top + 148 > 688:
                        continue
                    visible += 1
                    if not card.missing():
                        continue
                    left = COLUMNS[column]
                    box = (left, top, 118, 148)
                    rarity_evidence = None
                    if not card.rarities:
                        try:
                            card.rarity, rarity_evidence = _rarity_from_background(image, box)
                        except ValueError as error:
                            card.observations.append(dict(frame=self.frame, box=list(box), error=str(error)))
                            missing.append(f"member@{member['position']}:card{index}:rarity")
                            continue
                        card.rarities.add(card.rarity)
                    rarity = card.rarity
                    level, level_basis, level_raw, badge_raw = None, None, [], []
                    if not card.levels:
                        unacquired, badge_raw = _unacquired(image, box, texts)
                        if unacquired:
                            level, level_basis = 0, "unacquired_badge"
                        elif rarity == "core":
                            level, level_basis = 1, "core_acquired_badge_absent"
                        else:
                            level, level_raw = self._digit(image, (left + 6, top + 2, 29, 26), 6)
                            level_basis = "left_actual_level"
                    target, target_raw = None, []
                    thumb = [hit for hit in thumbs if _inside(hit, (left + 69, top, 49, 40))]
                    if not card.targets:
                        if not thumb:
                            target = 0
                        elif rarity == "core":
                            target = 1
                        else:
                            target, target_raw = self._digit(image, (left + 102, top + 2, 17, 29), 6)
                            if target == 0:
                                target = None
                    if level is not None:
                        card.levels.add(level)
                    if target is not None:
                        card.targets.add(target)
                    card.observations.append(dict(frame=self.frame, box=list(box),
                        display_condition_frame=self.frame,
                        level=level, level_basis=level_basis, level_readings=level_raw,
                        unacquired_readings=badge_raw, target_level=target,
                        thumb_readings=thumb, target_readings=target_raw,
                        rarity_class=rarity, rarity_basis="bag_card_background_color",
                        rarity_color_evidence=rarity_evidence))
                    if not card.name()[1]:
                        name_box = (left + 3, top + 108, 112, 39)
                        rows = [row for row in texts if _inside(row, name_box) and row["score"] >= .6]
                        self._add_name(card, rows, name_box, "page")
                        a, b, w, h = name_box
                        card.crop = image[b:b + h, a:a + w].copy()
                        card.crop_box, card.crop_frame = list(name_box), self.frame
                        self._confirm_name(card)
                    missing.extend(f"member@{member['position']}:card{index}:{field}"
                                   for field in card.missing())
        if not visible:
            missing.append("visible_card_layout")
        self.missing = missing
        return missing

    def read_page(self, image):
        """Fill this located page before traversing onward; inputs stay outside reads."""
        anchor = image

        def observe(frame):
            try:
                if frame is not anchor and abs(_content_shift(anchor, frame)) >= 3:
                    self.missing = ["stable_visible_page"]
                    return None
                return (frame,) if not self.feed(frame) else None
            except ValueError as error:
                self.missing = [str(error)]
                return None

        result = self._read_until(observe, image)
        if result is None:
            if self.context.tasker.stopping:
                return None
            raise ValueError("背包当前可见页未读全：" + ", ".join(self.missing))
        return result[0]

    def read_scrolled_page(self, previous):
        def observe(frame):
            try:
                return frame, _content_shift(previous, frame)
            except ValueError as error:
                self.missing = [str(error)]
                return None

        result = self._read_until(observe)
        if result is None:
            if self.context.tasker.stopping:
                return None
            raise ValueError("背包潜能滚动画面未能衔接")
        image, shift = result
        self.offset -= shift
        return image, shift

    def _add_name(self, card, rows, box, source, frame=None):
        name = _joined(rows)
        if clean_name(name) and not UNACQUIRED.search(clean_name(name)):
            card.readings.append(dict(text=name, score=min(row["score"] for row in rows),
                box=list(box), frame=self.frame if frame is None else frame, source=source,
                source_roi=list(box), fragments=rows))

    def _confirm_name(self, card):
        """Read original title lines; transformations only fill unread parts."""
        if card.name_evidence is not None:
            return
        x, y, width, height = card.crop_box
        reflowed = _reflow_name(card.crop)
        regions = reflowed[1] if reflowed else [[0, 0, width, 33]]
        regions = [[x + a, y + b, w, h] for a, b, w, h in regions]
        originals = [row for row in card.readings if row["source"] == "page" and row["score"] >= .85
                     and len(_text_lines(row["fragments"])) == len(regions)
                     and (len(regions) == 1 or all(len(clean_name(_joined(line), keep_long_vowel=True)) > 1
                          for line in _text_lines(row["fragments"])))]
        clear_originals = [row for row in originals if row["score"] >= CLEAR_NAME_SCORE]
        if clear_originals:
            originals = clear_originals
        keys = {clean_name(row["text"], keep_long_vowel=True) for row in originals}
        if len(keys) > 1:
            return
        if originals and (max(row["score"] for row in originals) >= CLEAR_NAME_SCORE
                          or len({row["frame"] for row in originals}) >= 2):
            card.name_evidence = dict(text=originals[0]["text"], source="clear_original_region"
                if max(row["score"] for row in originals) >= CLEAR_NAME_SCORE else "independent_original_frames",
                original_regions=[dict(frame=row["frame"], box=row["box"], fragments=row["fragments"])
                                  for row in originals])
            return
        lines = {}
        for scale in (1, 2):
            source = f"name_crop_{scale}x"
            crop = card.crop if scale == 1 else cv2.resize(
                card.crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            rows = read_text(self.context, crop, (0, 0, crop.shape[1], crop.shape[0]))
            for row in rows:
                a, b, w, h = row["box"]
                row["box"] = [x + round(a / scale), y + round(b / scale), round(w / scale), round(h / scale)]
            self._add_name(card, rows, card.crop_box, source, card.crop_frame)
            if len(regions) == 1 and not (scale == 1 and rows
                    and all(row["score"] >= CLEAR_NAME_SCORE for row in rows)):
                located = [row["box"] for row in rows if row["box"] != card.crop_box]
                if located:
                    left = max(x, min(box[0] for box in located) - 2)
                    top = max(y, min(box[1] for box in located) - 2)
                    right = min(x + width, max(box[0] + box[2] for box in located) + 2)
                    bottom = min(y + height, max(box[1] + box[3] for box in located) + 2)
                    rows = read_text(self.context, card.crop,
                        (left - x, top - y, right - left, bottom - top), only_rec=True)
                    for row in rows:
                        a, b, w, h = row["box"]
                        row["box"] = [x + a, y + b, w, h]
                        row["located_text_roi"] = [left, top, right - left, bottom - top]
                    source = f"name_line_from_{scale}x_detection"
                    self._add_name(card, rows, card.crop_box, source, card.crop_frame)
            for index, region in enumerate(regions):
                if index in lines:
                    continue
                fragments = [row for row in rows if _inside(row, region)
                    and row["box"] != card.crop_box
                    and (len(regions) == 1 or (row["box"][1] >= region[1] - 2
                        and row["box"][1] + row["box"][3] <= region[1] + region[3] + 2))]
                # A partial line may be readable yet still contain a small
                # kana error. Only a clear fragment can constrain the later
                # whole-title read; complete fields retain their own gate.
                minimum = CLEAR_NAME_SCORE if len(regions) > 1 else .85
                if (fragments and all(row["score"] >= minimum for row in fragments)
                        and (len(regions) == 1 or len(clean_name(_joined(fragments), keep_long_vowel=True)) > 1)):
                    lines[index] = dict(text=_joined(fragments), box=region,
                        source=source, fragments=fragments)
            if len(lines) == len(regions):
                card.name_evidence = dict(text="".join(lines[index]["text"] for index in range(len(regions))),
                    source_frame=card.crop_frame, source_roi=card.crop_box,
                    source="original_title_lines", lines=[lines[index] for index in range(len(regions))])
                return

        # A wrapped trailing kana can remain unreadable alone. Reflow the
        # original pixels once, retaining every already clear line verbatim.
        strip = reflowed[0] if reflowed else card.crop
        acquired = bool(card.levels - {0})
        if reflowed and acquired and not lines:
            ink = np.where(strip.max(axis=2) < 205, 0, 255).astype(np.uint8)
            strip = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
        crop = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        rows = read_text(self.context, crop, (0, 0, crop.shape[1], crop.shape[0]), only_rec=True)
        source = "name_lines_3x" if reflowed else "name_line_3x"
        for row in rows:
            row["attempt_box"] = row["box"]
            row["box"] = list(card.crop_box)
            row["source_regions"] = regions
            if reflowed and acquired and not lines:
                row["preprocessing"] = "acquired_title_dark_strokes_max205"
        self._add_name(card, rows, card.crop_box, source, card.crop_frame)
        name = _joined(rows)
        key = clean_name(name, keep_long_vowel=True)
        prefix = clean_name(lines[0]["text"], keep_long_vowel=True) if 0 in lines else ""
        suffix = clean_name(lines[1]["text"], keep_long_vowel=True) if 1 in lines else ""
        if (rows and all(row["score"] >= .85 for row in rows)
                and key.startswith(prefix) and key.endswith(suffix)
                and len(key) > sum(len(clean_name(line["text"], keep_long_vowel=True)) for line in lines.values())):
            card.name_evidence = dict(text=name, source_frame=card.crop_frame,
                source_roi=card.crop_box, source=source, source_regions=regions,
                preserved_lines=list(lines.values()))

    def finish_names(self):
        for member in self.members:
            for card in member["cards"]:
                if self.context.tasker.stopping:
                    raise ValueError("任务已停止")
                if card.crop is None:
                    continue
                self._confirm_name(card)

    def result(self):
        party = []
        for role, member in enumerate(self.members, 1):
            cards = []
            for card in member["cards"]:
                if len(card.rarities) != 1:
                    raise ValueError(f"背包角色{role}卡片{card.index + 1}品质未读全或读数冲突")
                if len(card.levels) != 1 or len(card.targets) != 1:
                    raise ValueError(f"背包角色{role}卡片{card.index + 1}等级未读全或读数冲突")
                name, confirmed = card.name()
                target = next(iter(card.targets))
                level = next(iter(card.levels))
                if not confirmed:
                    raise ValueError(f"背包角色{role}卡片{card.index + 1}完整名称未确认")
                cards.append(dict(name=name, rarity_class=card.rarity, target_level=target,
                    level=level, name_confirmed=confirmed,
                    raw=dict(name_readings=card.readings, observations=card.observations,
                             name_confirmation=card.name_evidence,
                             bag_display_index=card.index)))
            if Counter(card["rarity_class"] for card in cards) != {"core": 4, "rare": 3, "common": 9}:
                raise ValueError(f"背包角色{role}未读到完整4/3/9潜能品质")
            if not member["name"] or member["obtained_count"] is None:
                raise ValueError(f"背包角色{role}标题或获得种数未读清")
            if sum(card["level"] > 0 for card in cards) != member["obtained_count"]:
                raise ValueError(f"背包角色{role}实际持有与标题种数不一致")
            party.append(dict(role=role, name=member["name"], obtained_count=member["obtained_count"],
                cards=cards, raw=dict(header_readings=member["header_readings"],
                    obtained_count_readings=member["obtained_readings"],
                    display_conditions=[self.display_conditions[frame] for frame in sorted(
                        {row["frame"] for card in member["cards"] for row in card.observations})],
                    role_basis="game_team_list_order",
                    catalogue_order="bag_display_order", real_draw_pool_order="unknown")))
        if len(party) != 3:
            raise ValueError(f"背包只读到{len(party)}名角色")
        return party


def read_bag_party(context, language, capture=None):
    """Read every member's complete 16-card bag; leave its collection tab open."""
    reader = _Reader(context, language, capture)
    image = reader.prepare()
    for _ in range(18):
        image = reader.read_page(image)
        if image is None:
            return None
        if len(reader.members) == 3 and all(
                not card.missing()
                for member in reader.members for card in member["cards"]):
            break
        reader.scroll()
        following = reader.read_scrolled_page(image)
        if following is None:
            return None
        image, shift = following
        if abs(shift) < 3:
            break
    return reader.result()
