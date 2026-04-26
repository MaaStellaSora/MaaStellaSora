import re
import time
from dataclasses import dataclass, field
from typing import Optional, Any, Self

import numpy

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential")


MAX_POTENTIAL_LEVEL: int = 6  # 潜能等级上限，condition max_level 字段的默认值

DEFAULT_POTENTIAL_LAYOUTS = {
    1: [
        {
            "core_potential": [530, 425, 220, 40],
            "general_potential": [530, 395, 220, 40],
            "general_potential_level": [530, 425, 220, 40],
            "recommended_level": [670, 165, 140, 50],
            "x_border": [470, 813]
        }
    ],
    2: [
        {
            "core_potential": [358, 425, 220, 40],
            "general_potential": [358, 395, 220, 40],
            "general_potential_level": [358, 425, 220, 40],
            "recommended_level": [490, 165, 140, 50],
            "x_border": [0, 639]
        },
        {
            "core_potential": [703, 425, 220, 40],
            "general_potential": [703, 395, 220, 40],
            "general_potential_level": [703, 425, 220, 40],
            "recommended_level": [840, 165, 140, 50],
            "x_border": [640, 1280]
        }
    ],
    3: [
        {
            "core_potential": [187, 425, 220, 40],
            "general_potential": [187, 395, 220, 40],
            "general_potential_level": [187, 425, 220, 40],
            "recommended_level":[320, 165, 140, 50],
            "x_border": [0, 469]
        },
        {
            "core_potential": [530, 425, 220, 40],
            "general_potential": [530, 395, 220, 40],
            "general_potential_level": [530, 425, 220, 40],
            "recommended_level": [670, 165, 140, 50],
            "x_border": [470, 813]
        },
        {
            "core_potential": [875, 425, 220, 40],
            "general_potential": [875, 395, 220, 40],
            "general_potential_level": [875, 425, 220, 40],
            "recommended_level":[1010, 165, 140, 50],
            "x_border": [814, 1280]
        }
    ]
}


@dataclass(slots=True)
class PotentialLayout:
    core_potential_roi: list[int]
    general_potential_roi: list[int]
    general_potential_level_roi: list[int]
    recommended_level_roi: list[int]
    x_border: list[int]

@dataclass(slots=True)
class PotentialLayouts:
    potential_layouts: dict[int, list[PotentialLayout]] = field(default_factory=lambda: {
        count: [PotentialLayout(**l) for l in layouts]
        for count, layouts in DEFAULT_POTENTIAL_LAYOUTS.items()
    })

    def __getitem__(self, key):
        return self.potential_layouts[key]

    def __iter__(self):
        return iter(self.potential_layouts)

    def items(self):
        return self.potential_layouts.items()

    def get(self, key, default=None):
        return self.potential_layouts.get(key, default)

@dataclass(slots=True, frozen=True)
class Parameters:
    max_refresh_count: int
    reserved_coin: int
    priority_list: list[dict]
    owned_potentials: dict
    potential_layouts: PotentialLayouts = field(default_factory=lambda: PotentialLayouts())
    selected_potential_offset: int = 35

    def __post_init__(self):
        parsed_list = self._parse_priority_raw_list(
            self.priority_list,
            self.owned_potentials,
        )
        object.__setattr__(self, 'priority_list', parsed_list)

    @staticmethod
    def _parse_priority_raw_list(
        potential_priority_raw: list[dict],
        owned_potentials: dict,
    ) -> list[dict[str, Any]]:
        """对原始 priority_list 进行初筛，过滤掉 condition 当前不满足的规则。

        排名使用原始 JSON 的 1-based 行号（index + 1），数值越小排名越高。
        condition 不满足的条目直接跳过，其行号仍保留在原始位置，不影响其他条目的排名。

        Args:
            potential_priority_raw: 原始优先级列表，每个元素结构：
                {
                    "trekker": str,         # 可选，潜能归属角色名
                    "potential": str|list,  # 必填，潜能名称或名称列表
                    "level_span": int,      # 可选，默认 1，最小升级跨度
                    "max_level": int,       # 可选，默认 MAX_POTENTIAL_LEVEL，旧等级上限（不含）
                    "refresh": int,         # 可选，默认 0，已刷新次数必须 >= 该值规则才生效
                    "condition": list       # 可选，生效条件，元素为 dict 时 AND，为 list 时 OR
                }
            owned_potentials: 已拥有潜能状态，按 trekker 分组，结构：
                {"花原": {"飞花乱坠": 1}, "unknown": {"盛大尾奏": 1}}

        Returns:
            list[dict]: 通过初筛的规则列表，每个元素结构：
                {
                    "trekker": str | None,  # 归属角色
                    "names": list[str],     # 目标潜能名称列表
                    "level_span": int,      # 最小升级跨度
                    "max_level": int,       # 旧等级上限（不含）
                    "refresh": int,         # 已刷新次数下限
                    "priority": int         # 原始 JSON 的 1-based 行号，越小排名越高
                }
        """
        owned_map: dict[str, int] = {}
        for _p in owned_potentials.values():
            owned_map.update(_p)

        def _check_single_condition(item: dict) -> bool:
            """检查单个 condition 子项是否满足。"""
            if "count_at_least" in item or "count_at_most" in item:
                potentials = owned_potentials.get(item["trekker"], {})
                level_min = item.get("level_at_least")
                level_max = item.get("level_at_most")
                if level_min is not None or level_max is not None:
                    potentials = {
                        name: level for name, level in potentials.items()
                        if (level_min is None or level >= level_min)
                        and (level_max is None or level <= level_max)
                    }
                count = len(potentials)
                if "count_at_least" in item and count < item["count_at_least"]:
                    return False
                if "count_at_most" in item and count > item["count_at_most"]:
                    return False
                return True
            current = owned_map.get(item["potential"], 0)
            min_ok = current >= item["level_at_least"] if "level_at_least" in item else True
            max_ok = current <= item["level_at_most"] if "level_at_most" in item else True
            return min_ok and max_ok

        def _check_condition(cond: list) -> bool:
            """检查 condition 列表是否满足。

            元素全为 dict 时为 AND 逻辑；含 list 元素时为 OR 逻辑（内层为 AND）。
            """
            if not cond:
                return True
            if all(isinstance(item, dict) for item in cond):
                return all(_check_single_condition(item) for item in cond)
            for branch in cond:
                if isinstance(branch, list):
                    if all(_check_single_condition(item) for item in branch):
                        return True
                elif isinstance(branch, dict):
                    if _check_single_condition(branch):
                        return True
            return False

        valid_entries = []
        for index, raw in enumerate(potential_priority_raw):
            condition = raw.get("condition", [])
            if not isinstance(condition, list):
                continue
            if not _check_condition(condition):
                continue

            potential = raw["potential"]
            names = potential if isinstance(potential, list) else [potential]

            valid_entries.append({
                "trekker": raw.get("trekker"),
                "names": names,
                "level_span": raw.get("level_span", 1),
                "max_level": raw.get("max_level", MAX_POTENTIAL_LEVEL),
                "refresh": raw.get("refresh", 0),
                "priority": index + 1,
            })

        return valid_entries

@dataclass(slots=True)
class Potential:
    index: int
    box: list[int]
    name: str = ""
    old_level: int = 0
    new_level: int = 0
    recommended: bool = False
    recommended_level: int = 0
    rank: int = -1
    sub_rank: int = -1
    trekker: str = ""

    @property
    def level_span(self) -> int:
        return self.new_level - self.old_level


@dataclass(slots=True)
class Data:
    params: Parameters
    current_coin: int = 0
    refresh_cost: int = 0
    potential_count: int = 0
    core_potential: bool = False
    # 需要根据刷新更新的数据
    selected_potential_index: int = 1
    potentials: list[Potential] = field(default_factory=lambda: [])
    # 计数用数据
    refresh_count: int = 0

    @property
    def refreshable(self):
        return self.refresh_cost > 0

    @property
    def available_refreshes(self) -> int:
        usable_coin = max(0, self.current_coin - self.params.reserved_coin)
        affordable = usable_coin // self.refresh_cost if self.refreshable else 0
        return min(self.params.max_refresh_count, affordable)

    @property
    def borders(self) -> list[list[int]]:
        return [l.x_border for l in self.params.potential_layouts[self.potential_count]]

    @property
    def core_potential_name_rois(self) -> list[list[int]]:
        return [l.core_potential_roi for l in self.params.potential_layouts[self.potential_count]]

    @property
    def general_potential_name_rois(self) -> list[list[int]]:
        return [l.general_potential_roi for l in self.params.potential_layouts[self.potential_count]]

    @property
    def general_potential_level_rois(self) -> list[list[int]]:
        return [l.general_potential_level_roi for l in self.params.potential_layouts[self.potential_count]]

    @property
    def recommended_level_rois(self) -> list[list[int]]:
        return [l.recommended_level_roi for l in self.params.potential_layouts[self.potential_count]]


class ScreenDataProcessor:
    def __init__(self, context: Context):
        self.context = context
        self.image = None
        self.max_try = 1

    def screenshot(self):
        self.image = self.context.tasker.controller.post_screencap().wait().get()

    def refresh(self):
        self.context.run_task("星塔_节点_选择潜能_点击刷新_agent")

    def _base_recognition(self, mode, node_name, failed_return, roi=None, image=None, max_try=0) -> Any:
        """
        核心识别逻辑：支持 OCR 和 Template
        识别失败时返回默认值

        Args:
            mode(str): 识别模式，"ocr"或"template"
            node_name(str): 节点名称，用于识别结果的记录和返回
            failed_return(any): 识别失败时的默认值
            roi(tuple): 可选的ROI坐标，用于模板识别
            image(numpy.ndarray): 可选的自定义图像，用于识别
            max_try(int): 可选的最大重试次数，默认为1

        Returns:
            any: 识别到的结果，根据mode返回文本或坐标列表
        """
        if image is None:
            if self.image is None:
                self.image = self.context.tasker.controller.post_screencap().wait().get()
            image = self.image

        actual_max_try = max_try if max_try > 0 else self.max_try

        pipeline_override = {node_name: {"recognition": {"param": {"roi": roi}}}} if roi else {}

        try_count = 0
        while True:
            reco_detail = self.context.run_recognition(node_name, image, pipeline_override)

            if reco_detail and reco_detail.hit:
                if mode == "ocr":
                    # OCR 逻辑：返回文本
                    logger.debug(f"节点{node_name} OCR结果：{[(r.text, r.score) for r in reco_detail.filtered_results]}")
                    results = sorted(reco_detail.filtered_results, key=lambda r: r.score, reverse=True)
                    return [r.text for r in results]
                else:
                    # Template 逻辑：返回坐标列表
                    logger.debug(f"节点{node_name} 模板结果：{[(r.box, r.score) for r in reco_detail.filtered_results]}")
                    results = sorted(reco_detail.filtered_results, key=lambda r: r.score, reverse=True)
                    return [r.box for r in results]

            # 统一的日志记录
            status = "未识别到有效结果" if reco_detail and reco_detail.all_results else "未识别到任何内容"
            logger.debug(f"节点'{node_name}'{status}")

            if self.context.tasker.stopping:
                return failed_return

            try_count += 1
            if try_count >= actual_max_try:
                break

            logger.debug("等待1秒后重新识别")
            time.sleep(1)
            image = self.context.tasker.controller.post_screencap().wait().get()

        logger.debug(f"无法识别节点'{node_name}'，返回默认值 {failed_return}")
        return failed_return

    def _ocr(self, node_name, failed_return, **kwargs):
        return self._base_recognition("ocr", node_name, failed_return, **kwargs)

    def _template(self, node_name, failed_return, **kwargs):
        return self._base_recognition("template", node_name, failed_return, **kwargs)

    def get_current_coin(self, image: Optional[numpy.ndarray] = None, max_try: int = 1) -> int:
        texts = self._ocr("星塔_通用_识别当前金币_agent", ["0"], image=image, max_try=max_try)
        return int(texts[0])

    def get_refresh_cost(self, image: Optional[numpy.ndarray] = None, max_try: int = 1) -> int:
        texts = self._ocr("星塔_通用_识别刷新花费_agent", ["-1"], image=image, max_try=max_try)
        return int(texts[0])

    def check_core_potential(self, image: Optional[numpy.ndarray] = None, max_try: int = 1) -> bool:
        if self._template("星塔_节点_选择潜能_识别核心潜能_agent", [], image=image, max_try=max_try):
            return True
        return False

    def get_potential_name(self, roi: list[int], image: Optional[numpy.ndarray] = None, max_try: int = 1) -> str:
        node_name = "星塔_节点_选择潜能_识别潜能名称_agent"
        texts = self._ocr(node_name, [""], roi=roi, image=image, max_try=max_try)
        return texts[0]

    def get_potential_level(self, roi: list[int], image: Optional[numpy.ndarray] = None, max_try: int = 1) -> list:
        node_name = "星塔_节点_选择潜能_识别潜能等级_agent"
        texts = self._ocr(node_name, [""], roi=roi, image=image, max_try=max_try)
        return texts

    def get_recommend_level(self, roi: list[int], image: Optional[numpy.ndarray] = None, max_try: int = 1) -> int:
        node_name = "星塔_节点_选择潜能_识别推荐等级_agent"
        texts = self._ocr(node_name, ["0"], roi=roi, image=image, max_try=max_try)
        return int(texts[0])

    def check_item_list_visibility(self, max_try: int = 1) -> bool:
        image = self.context.tasker.controller.post_screencap().wait().get()
        return self._template("星塔_节点_选择潜能_检测干扰文字_agent", [], image=image, max_try=max_try)

    def get_potential_count(
            self,
            core_potential: bool = False,
            image: Optional[numpy.ndarray] = None,
            max_try: int = 1
    ) -> int:
        """
            检查可选潜能卡片数量

            Args:
                core_potential(bool): 是否为核心潜能，默认为False
                image(nd.array): 截图
                max_try(int): 可选的最大重试次数，默认为1

            Returns:
                int: 可选潜能卡片数量，识别失败时返回3
        """
        if core_potential:
            return 3

        node_name = "星塔_节点_选择潜能_识别潜能数量_agent"
        failed_return = [1, 2, 3]
        results = self._template(node_name, failed_return, image=image, max_try=max_try)
        if results == failed_return:
            logger.error("潜能数量识别失败，将默认为3个潜能")
        return len(results)

    def get_recommended_potential(
            self,
            borders: list[list],
            image: Optional[numpy.ndarray] = None,
            max_try: int = 1
    ) -> list:
        """
        识别系统推荐图标，返回对应卡片的潜能序数列表。

        推荐图标位于卡片 box 范围内，通过判断图标命中 x 坐标是否落入各卡片
        x_border 区间来确定归属卡片。
        识别失败时返回第一张卡片的 box 作为兜底。

        Args:
            borders: 可选潜能卡片区域列表，每个元素结构：[float, float],  # 卡片 x 轴边界（左闭右闭）
            image: 截图
            max_try: 可选的最大重试次数，默认为1

        Returns:
            list: 包含推荐潜能序数的列表
        """
        recommended_boxes = self._template(
            "星塔_节点_选择潜能_识别推荐图标_agent",
            [],
            image=image,
            max_try=max_try
        )
        hit_xs = [r[0] for r in recommended_boxes]
        matched = [
            i for x in hit_xs
            for i, (low, high) in enumerate(borders)
            if low <= x <= high
        ]

        if not matched:
            logger.debug("推荐图标识别失败，有可能是没有推荐图标，也有可能是识别问题")
        if len(matched) != len(hit_xs):
            logger.error("推荐图标识别位置与潜能数量不匹配，潜能选择可能会出现问题")

        return matched

    def get_selected_potential_index(
            self,
            borders: list[list[int]],
            image: Optional[numpy.ndarray] = None,
            max_try: int = 1
    ) -> int:
        """识别拿走按钮，返回对应卡片的索引。

        拿走按钮位于卡片 box 范围内，通过判断图标命中 x 坐标是否落入各卡片
        x_border 区间（左闭右闭）来确定归属卡片。
        识别失败时返回第一张卡片的索引 作为兜底。

        Args:
            borders(list): 可选潜能卡片的边界框，每个元素为一个列表，包含2个元素，分别是左闭右闭的x轴边界
            image(numpy.ndarray): 截图
            max_try(int): 可选的最大重试次数，默认为1

        Returns:
            int: 目标卡片索引，识别失败时返回0
        """
        result_boxes = self._template(
            "星塔_节点_选择潜能_识别预选潜能位置_agent",
            [],
            image=image,
            max_try=max_try
        )

        hit_x = result_boxes[0][0] if result_boxes else -1
        matched = next((i for i, (low, high) in enumerate(borders) if low <= hit_x <= high), None)

        if matched is None:
            logger.error("拿走按钮识别失败，潜能选择可能会出现问题")
            matched = 0

        return matched


class ChoosePotentialHandler:
    def __init__(self, screen: ScreenDataProcessor, data: Data):
        self.screen = screen
        self.data = data

    def _wait_for_item_list_gone(self):
        while True:
            if self.screen.check_item_list_visibility():
                logger.debug("识别到干扰文字，等待1秒")
                time.sleep(1)
                continue
            break

    def _update_names(self):
        rois = self.data.core_potential_name_rois if self.data.core_potential else self.data.general_potential_name_rois
        adjusted_rois = self._get_adjusted_rois(rois)

        for i, roi in enumerate(adjusted_rois):
            self.data.potentials[i].name = self.screen.get_potential_name(roi)

    def _update_levels(self):
        if self.data.core_potential:
            return

        adjusted_rois = self._get_adjusted_rois(self.data.general_potential_level_rois)

        for i, roi in enumerate(adjusted_rois):
            texts = self.screen.get_potential_level(roi)
            old, new = self._parse_level_text(texts)
            self.data.potentials[i].old_level, self.data.potentials[i].new_level = old, new

    def _update_recommended_potentials(self):
        adjusted_rois = self._get_adjusted_rois(self.data.recommended_level_rois)
        indices = self.screen.get_recommended_potential(self.data.borders)
        for index in indices:
            roi = adjusted_rois[index]
            self.data.potentials[index].recommended = True
            self.data.potentials[index].recommended_level = self.screen.get_recommend_level(roi)

    @staticmethod
    def _parse_level_text(texts: list[str]) -> tuple[int, int]:
        """解析 OCR 返回的等级数字结果集。

        pipeline OCR 使用 \\d+ 匹配并剔除语言关键词，可能返回：
            ["1"]       -> old=0, new=1  （新获得，只有新等级）
            ["4", "5"]  -> old=4, new=5
            ["45"]      -> old=4, new=5  （两位数粘连）
        仅在游戏版本保持最大潜能等级小于10时有效。

        Args:
            texts: OCR filtered_results 中各结果的 text 列表

        Returns:
            tuple[int, int]: (old_level, new_level)，解析失败返回 (0, 0)
        """
        # 将 ["4", "5"] 或 ["45"] 统一转为 "45"
        full_text = "".join(t for t in texts if t.isdigit())

        if len(full_text) == 1:
            return 0, int(full_text)
        if len(full_text) >= 2:
            # 取前两个数字处理粘连
            return int(full_text[0]), int(full_text[1])
        return 0, 0

    def _get_adjusted_rois(self, base_rois: list[list[int]]) -> list[list[int]]:
        """安全地获取偏移后的 ROI 副本，避免污染原始数据"""
        selected_index = self.data.selected_potential_index
        offset = self.data.params.selected_potential_offset

        # 使用列表推导式创建副本并应用偏移
        return [
            [r[0], r[1] - offset, r[2], r[3]] if i == selected_index else list(r)
            for i, r in enumerate(base_rois)
        ]

    def choose_default_potential(self):
        priority_rules = [
            lambda p: p.recommended, # 1. 尝试取系统推荐潜能
            lambda p: p.old_level > 0 # 2. 没有系统推荐潜能，选择之前抓过的潜能
        ]

        for rule in priority_rules:
            candidates = [p for p in self.data.potentials if rule(p)]
            if candidates:
                # 按照等级跨度降序、推荐等级降序、旧等级降序来排序，选择最优的潜能
                return max(candidates, key=lambda p: (p.level_span, p.recommended_level, p.old_level), default=None)

        # 都没有的话，放弃选择，系统选了哪张就哪张
        return Potential(0, [5, 710, 5, 5])

    def refresh(self):
        self.screen.refresh()
        self.data.refresh_count += 1

    @property
    def refreshable(self):
        return self.data.refresh_count <self.data.available_refreshes


class GameRecommendedHandler(ChoosePotentialHandler):
    def __init__(self, screen: ScreenDataProcessor, data: Data):
        super().__init__(screen, data)

    def read_potentials_info(self) -> Self:
        click_boxes = self.data.general_potential_name_rois
        self.data.potentials = [Potential(i, click_boxes[i]) for i in range(self.data.potential_count)]

        self._wait_for_item_list_gone()
        self.screen.screenshot()
        self.data.selected_potential_index = self.screen.get_selected_potential_index(self.data.borders)

        self._update_recommended_potentials()
        self._update_names()
        self._update_levels()

        return self

    def choose(self):
        # 选择排名最高的潜能
        best_potential = self.best_potential
        if best_potential:
            # 输出比较结果
            if self.data.core_potential:
                logger.info(f"[潜能选择] {best_potential.name} | 核心潜能 | 系统推荐")
            else:
                old = best_potential.old_level
                new = best_potential.new_level
                logger.info(f"[潜能选择] {best_potential.name} | 等级 {old}→{new} | 系统推荐")

        return best_potential

    @property
    def best_potential(self) -> Potential | None:
        """
        筛选出最好的系统推荐潜能
        目前规则：
            >升级间距大于等于3级且推荐等级大于等于3级的新卡
            >拥有推荐标记的已抓过的卡
            >升级间距刚好等于推荐等级的新卡
            >升级间距2级且推荐等级大于2级的新卡
        筛选成功后，按照等级跨度降序、推荐等级降序、旧等级降序来排序，取得想要的潜能

        Returns:
            Potential | None: 最好的系统推荐潜能，若没有则返回 None
        """
        priority_rules = [
            lambda p: p.old_level == 0 and p.level_span >= 3 and p.recommended_level >= 3,
            lambda p: p.old_level > 0 and p.recommended,
            lambda p: p.old_level == 0 and p.recommended_level > 0 and p.recommended_level - p.level_span == 0,
            lambda p: p.old_level == 0 and p.level_span == 2 and p.recommended_level > 2,
        ]

        for rule in priority_rules:
            candidates = [p for p in self.data.potentials if rule(p)]
            if candidates:
                return max(candidates, key=lambda p: (p.level_span, p.recommended_level, p.old_level), default=None)

        return None


class AssistantPriorityHandler(ChoosePotentialHandler):
    def __init__(self, screen: ScreenDataProcessor, data: Data):
        super().__init__(screen, data)

    def read_potentials_info(self) -> Self:
        click_boxes = self.data.general_potential_name_rois
        self.data.potentials = [Potential(i, click_boxes[i]) for i in range(self.data.potential_count)]

        self._wait_for_item_list_gone()
        self.screen.screenshot()
        self.data.selected_potential_index = self.screen.get_selected_potential_index(self.data.borders)

        self._update_names()
        self._update_levels()

        return self

    def choose(self):
        # 获得所有潜能的排名
        self._update_priority()

        # 输出比较结果
        for potential in self.data.potentials:
            if self.data.core_potential:
                logger.info(f"[潜能识别] {potential.name} | 核心潜能 | 排名 {potential.rank}")
            else:
                old = potential.old_level
                new = potential.new_level
                logger.info(f"[潜能识别] {potential.name} | 等级 {old}→{new} | 排名 {potential.rank}")

        # 选择排名最高的潜能
        best_potential = self.best_potential
        if best_potential:
            logger.info(f"[潜能选择] {best_potential.name}")

        return best_potential

    def _update_priority(self):
        for potential in self.data.potentials:
            rank, sub_rank, trekker = self._get_potential_priority(potential)
            potential.rank = rank
            potential.sub_rank = sub_rank
            potential.trekker = trekker

    def _get_potential_priority(
        self,
        potential: Potential,
    ) -> tuple[int, int, str | None]:
        """获取单个待选潜能在规则列表中的最高排名及其 trekker 归属。

        遍历 priority_list，找到所有名称匹配且满足 level_span / max_level / refresh
        条件的规则，返回排名数值最小（即优先级最高）的规则对应的排名与 trekker。

        Args:
            potential: 单个待选潜能，结构：
                {"name": str, "old_level": int, "new_level": int, "box": list}

        Returns:
            tuple[int, int, str | None]: (rank, sub_rank, trekker)
                rank 为匹配到的最小排名数值；无匹配时返回 -1
                sub_rank 为命中的 potential 名称在该规则 names 列表中的下标；无匹配时返回 -1
                trekker 为对应规则的归属角色；无匹配时返回空字符
        """
        priority_list = self.data.params.priority_list

        best_entry = None
        best_sub_rank = -1

        for entry in priority_list:
            # 1. 基础剪枝：优先级如果不更高，直接跳过
            if best_entry and entry["priority"] >= best_entry["priority"]:
                continue

            # 2. 匹配名称并获取优先级排名
            sub_rank = self._find_sub_rank(potential.name, entry["names"])
            if sub_rank == -1:
                continue

            # 3. 验证其他规则是否通过
            if not self._is_entry_valid(entry, potential):
                continue

            # 全部通过后，记录该行及副等级
            best_entry = entry
            best_sub_rank = sub_rank

        if not best_entry:
            best_entry = {"rank": -1, "trekker": ""}

        return best_entry["rank"], best_sub_rank, best_entry["trekker"]

    def _find_sub_rank(self, name: str, rule_names: list[str]) -> int:
        """通过潜能名称获取最优排名数值"""
        return next((i for i, r in enumerate(rule_names)
                     if self._match_potential_name(name, r)), -1)

    @staticmethod
    def _match_potential_name(ocr_name: str, rule_name: str) -> bool:
        """比较 OCR 识别的潜能名称与规则中的潜能名称是否匹配。

        OCR 可能产生前后漏字或噪声字符，因此先清洗两边字符串（去除非 Unicode
        字母数字字符），再用 in 检查，覆盖前后漏字的情况。

        中间漏字或错字无法处理，属于 OCR 识别质量问题。

        Args:
            ocr_name: OCR 识别到的潜能名称
            rule_name: 优先级规则中定义的潜能名称

        Returns:
            bool: 两者匹配时返回 True
        """
        cleaned_ocr = re.sub(r'\W', '', ocr_name)
        cleaned_rule = re.sub(r'\W', '', rule_name)

        # 防止"" in "任意字符串" 返回True
        if not cleaned_ocr or not cleaned_rule:
            return False

        chars_to_remove = "ー"
        table = str.maketrans("", "", chars_to_remove)
        cleaned_ocr = cleaned_ocr.translate(table)
        cleaned_rule = cleaned_rule.translate(table)

        return cleaned_ocr in cleaned_rule

    def _is_entry_valid(self, entry: dict, potential: Potential) -> bool:
        """业务规则过滤器：方便未来随意扩展判定条件"""
        # 核心潜能默认全部通过
        if self.data.core_potential:
            return True

        # 普通潜能，组合匹配规则
        checks = [
            potential.old_level < entry["max_level"],
            potential.level_span >= entry["level_span"],
            self.data.refresh_count >= entry["refresh"]
        ]
        return all(checks)

    @property
    def best_potential(self) -> Potential | None:
        """按照排名升序、等级跨度降序、副排名升序三个维度，筛选出最好的潜能"""
        valid_potentials = (p for p in self.data.potentials if p.rank >= 0)
        return min(valid_potentials, key=lambda p: (p.rank, -p.level_span, p.sub_rank), default=None)



@AgentServer.custom_recognition("choose_potential_recognition")
class ChoosePotentialRecognition(CustomRecognition):

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        """自动选择潜能的主流程。

        读取 attach 参数与已拥有潜能状态，识别当前三张潜能卡片，
        根据自定义优先级列表选出最优潜能并返回其 box；
        不满足条件时执行刷新，最终兜底识别系统推荐图标。
        选择完成后将已拥有潜能状态写回节点 attach 持久化。

        Args:
            context: maa.context.Context
            argv: CustomRecognition.AnalyzeArg，含当前截图与节点名

        Returns:
            CustomRecognition.AnalyzeResult: 命中区域为目标潜能卡片的 box
        """
        node_name = argv.node_name
        params = self._get_params(context, node_name)
        data = Data(params=params)
        screen = ScreenDataProcessor(context)

        # 获取只使用一次的数据
        data.current_coin = screen.get_current_coin(argv.image)
        data.refresh_cost = screen.get_refresh_cost(argv.image)
        data.core_potential = screen.check_core_potential(argv.image)
        data.potential_count = screen.get_potential_count(data.core_potential, argv.image)

        # 加载相应的潜能处理类
        if data.params.priority_list:
            handler = AssistantPriorityHandler(screen, data)
        else:
            handler = GameRecommendedHandler(screen, data)

        while True:
            # 获取潜能数据，并选择潜能
            potential = handler.read_potentials_info().choose()
            if potential:
                break
            elif handler.refreshable:
                logger.info("没有找到符合条件的潜能，尝试刷新")
                handler.refresh()
            else:
                logger.info("[潜能选择] 没有找到符合条件的潜能，将按照默认顺序选择")
                potential = handler.choose_default_potential()
                break

        # 如果使用星塔助手优先级模块，更新已拥有潜能记录
        if data.params.priority_list:
            owned = self._update_owned_potentials(
                data.params.owned_potentials,
                potential.name,
                potential.new_level,
                potential.trekker,
            )
            self._save_state(context, node_name, owned)

        return CustomRecognition.AnalyzeResult(box=potential.box, detail={})

    @staticmethod
    def _get_params(context: Context, node_name: str) -> Parameters:
        """获取节点 attach 中的所有参数，缺失时返回安全默认值。

        Args:
            context: maa.context.Context
            node_name: 当前节点名称，用于动态获取节点数据

        Returns:
            Parameters: 包含以下属性的实例：
                - max_refresh_count (int): 最大刷新次数，0 表示禁用
                - reserved_coin (int): 预留金币，计算可用金币时需减去此值
                - priority_list (list): 自定义优先级列表
                - owned_potentials (dict): 已拥有潜能状态，按 trekker 分组
        """

        node_data = context.get_node_data(node_name)
        attach = node_data.get("attach", {})
        params = Parameters(**attach)
        return params

    @staticmethod
    def _update_owned_potentials(
        owned: dict,
        potential_name: str,
        new_level: int,
        trekker: str,
    ) -> dict:
        """将本次选中的潜能写入 owned_potentials 对应的 trekker 分组。

        Args:
            owned: 当前 owned_potentials 字典，按 trekker 分组
            potential_name: 本次选中的潜能名称
            new_level: 本次选中的潜能等级
            trekker: 归属角色名

        Returns:
            dict: 更新后的 owned_potentials
        """
        if not potential_name:
            return owned

        if new_level <= 0:
            logger.debug(f"备份潜能 {potential_name} 时发现等级低于1，将默认为1级")
            new_level = 1

        if not trekker:
            logger.debug(f"备份潜能 {potential_name} 时发现无所属旅人，将默认为unknown")
            trekker = "unknown"

        if trekker not in owned:
            owned[trekker] = {}
        owned[trekker][potential_name] = new_level
        return owned

    @staticmethod
    def _save_state(
        context: Context,
        node_name: str,
        owned: dict,
    ) -> None:
        """将更新后的 owned_potentials 写回节点 attach，通过 override_pipeline 持久化。

        写回失败时记录 ERROR 日志，不中断流程（本次状态丢失）。

        Args:
            context: maa.context.Context
            node_name: 当前节点名称
            owned: 更新后的 owned_potentials
        """
        new_attach = {"owned_potentials": owned}
        success = context.override_pipeline({node_name: {"attach": new_attach}})
        if not success:
            logger.error("保存当前拥有潜能失败，自定义潜能优先级可能无法正常工作")