"""Neural-only potential observations, reusing the original layouts and reader."""
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import unicodedata

from custom.reco.climb_tower_potential.context import PotentialContext
from custom.reco.climb_tower_potential.interactor import PotentialInteractor
from custom.reco.climb_tower_potential.main import ChoosePotentialRecognition
from utils import logger as logger_module

from .interactor import Interactor
from .ocr_number import assess_amount


logger = logger_module.get_logger(__name__)
RULES = json.loads(Path(__file__).with_name("potential_ocr_rules.json").read_text(encoding="utf-8"))
LEVEL_ROW = re.compile(RULES["level_row_pattern"], re.IGNORECASE)
RECOMMEND_LEVEL_ROW = re.compile(RULES["recommend_level_row_pattern"], re.IGNORECASE)
LEVEL_NODE = "星塔_神经网络_潜能_识别潜能等级"
RECOMMEND_NODE = "星塔_神经网络_潜能_识别推荐等级"
NAME_NODE = "星塔_神经网络_潜能_识别潜能名称"


@dataclass(slots=True)
class NeuralPotentialContext(PotentialContext):
    def __post_init__(self):
        layouts = deepcopy(self.params.potential_layouts)
        top, height = RULES["level_band"]
        for _, rows in layouts.items():
            for row in rows:
                x, _, width, _ = row.general_potential_level_roi
                row.general_potential_level_roi = [x, top, width, height]
        self.params = replace(self.params, potential_layouts=layouts)

    @property
    def refreshable(self):
        return (self.refresh_button and self.current_coin >= 0
                and self.current_coin - self.params.reserved_coin >= self.refresh_cost
                and self.refresh_count < self.refresh_limit)

    @property
    def refresh_limit(self):
        if self.refresh_cost == 0:
            return self.params.max_refresh_count
        usable = max(0, self.initial_coin - self.params.reserved_coin)
        affordable = usable // self.refresh_cost if self.refresh_button else 0
        return min(self.params.max_refresh_count, affordable)


class NeuralPotentialRecognition(ChoosePotentialRecognition):
    @staticmethod
    def _preload_data(interactor, data):
        data.initial_coin = interactor.get_current_coin()
        data.current_coin = data.initial_coin
        data.core_potential = interactor.check_core_potential()
        data.refresh_cost = (-1 if data.core_potential or data.params.potential_source == "enhance"
                             else interactor.get_refresh_cost())
        data.potential_rarities = interactor.get_potentials_rarity(data.core_potential)
        if data.params.potential_source != "enhance":
            data.team_level = interactor.get_team_level()


class NeuralPotentialInteractor(Interactor, PotentialInteractor):
    def refresh(self):
        result = self.context.run_task("星塔_神经网络_潜能_点击刷新")
        if not result or not result.status.succeeded:
            logger.error("$agent.custom.reco.climb_tower_potential.interactor.refresh.click_failed")
            return False
        self.screenshot()
        return True

    def get_card_box(self, potential):
        node = {
            "core": "星塔_节点_选择潜能_识别核心潜能_agent",
            "common": "星塔_节点_选择潜能_识别普通潜能数量_agent",
            "rare": "星塔_节点_选择潜能_识别稀有潜能数量_agent",
        }[potential.rarity]
        matches = self._recognize(node, roi=potential.potential_roi)
        if len(matches) == 1:
            return list(matches[0].box)
        matches = self.get_recommended_potential(potential.potential_roi)
        if len(matches) == 1:
            return list(matches[0])
        roi = potential.core_potential_name_roi if potential.core else potential.general_potential_name_roi
        detail = self.context.run_recognition(NAME_NODE, self.image, {
            NAME_NODE: {"recognition": {"param": {
                "roi": list(roi), "roi_offset": [0, 0, 0, 0], "only_rec": False}}}})
        rows = detail.filtered_results if detail and detail.hit else []
        rows = [row for row in rows if row.text.strip() and tuple(row.box) != tuple(roi)]
        if not rows or (potential.name and " ".join(row.text for row in rows) != potential.name):
            return None
        return list(rows[0].box)

    def select_potential(self, potential):
        box = self.get_card_box(potential)
        return box is not None and self.click_potential(box)

    def get_refresh_cost(self):
        return self._read_number("星塔_神经网络_识别刷新花费")

    def get_team_level(self):
        node = "星塔_节点_选择潜能_检测是否升级_agent"
        rows = self._recognize(node)
        if self._valid_ocr(node, rows) and rows:
            level = re.search(r"\d+", rows[0].text)
            if level:
                return int(level.group())
        return -1

    def get_potential_name(self, roi):
        return " ".join(row.text for row in self._recognize(NAME_NODE, roi=roi))

    def get_potential_level(self, roi):
        marker = self._recognize("星塔_神经网络_潜能_定位等级上升图标", roi=roi)
        if marker and (width := marker[0].box[0] - roi[0]) > 0:
            roi = [roi[0], roi[1], width, roi[3]]
        transition = self._recognize("星塔_神经网络_潜能_定位等级过渡图标", roi=roi)
        if len(transition) == 1:
            box = transition[0].box
            right = box[0] + box[2]
            before = self._recognize(LEVEL_NODE, roi=[roi[0], roi[1], box[0] - roi[0], roi[3]])
            after = self._recognize("星塔_神经网络_潜能_识别潜能等级后值",
                roi=[right, roi[1], roi[0] + roi[2] - right, roi[3]])
            return self._parse_level_text([" ".join(row.text for row in before), "▶",
                                           " ".join(row.text for row in after)])
        return self._parse_level_text([row.text for row in self._recognize(LEVEL_NODE, roi=roi)])

    @staticmethod
    def _parse_level_text(texts):
        match = LEVEL_ROW.fullmatch(unicodedata.normalize("NFKC", " ".join(texts)))
        if match:
            first, second = match.groups()
            if second is None:
                return 0, int(first)
            if int(first) < int(second):
                return int(first), int(second)
        logger.debug("Potential base levels remain unconfirmed: %s", texts)
        return -1, -1

    def get_recommend_level(self, roi):
        result = self.context.run_recognition(RECOMMEND_NODE, self.image,
            {RECOMMEND_NODE: {"recognition": {"param": {"roi": list(roi)}}}})
        def amount(row):
            match = RECOMMEND_LEVEL_ROW.fullmatch(unicodedata.normalize("NFKC", row.text))
            return match[1] if match else row.text

        value, _, _ = assess_amount(
            (amount(row) for row in result.filtered_results) if result and result.hit else (),
            ((amount(row), row.score) for row in result.all_results) if result else ())
        return -1 if value is None else value

    def get_potentials_rarity(self, core_potential=False):
        if core_potential:
            return ["core"] * 3 if self.check_core_potential() else []
        rows = [(hit.box[0], kind) for kind, node in (
            ("common", "星塔_节点_选择潜能_识别普通潜能数量_agent"),
            ("rare", "星塔_节点_选择潜能_识别稀有潜能数量_agent")) for hit in self._recognize(node)]
        return [kind for _, kind in sorted(rows)] if 1 <= len(rows) <= 3 else []

    def get_selected_potential_index(self, bounds):
        boxes = self.get_select_button()
        return next((i for i, (low, high) in enumerate(bounds)
                     if boxes and low <= boxes[0][0] <= high), -1)
