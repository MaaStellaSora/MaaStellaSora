import random
from typing import Any

import numpy
from maa.context import Context
from maa.define import OCRResult, TemplateMatchResult

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential_interactor")


class PotentialInteractor:
    def __init__(self, context: Context):
        self.context = context
        self.image = self.context.tasker.controller.cached_image

    def screenshot(self):
        self.image = self.context.tasker.controller.post_screencap().wait().get()

    def crop_screenshot(self, roi: list[int]) -> numpy.ndarray:
        return self.image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    def refresh(self):
        self.context.run_task("星塔_节点_选择潜能_点击刷新_agent")
        self.screenshot()

    def click_potential(self, box: list[int]) -> bool:
        """点击指定box，返回点击结果，True表示点击成功，False表示点击失败"""
        click_x = random.randint(box[0], box[0] + box[2])
        click_y = random.randint(box[1], box[1] + box[3])
        return self.context.tasker.controller.post_click(click_x, click_y).wait().succeeded

    def _recognize(
            self,
            node_name: str,
            *,
            roi: list[int] | None = None,
            image: numpy.ndarray | None = None,
            template: str | None = None
    ) -> list[Any]:
        """
        简单识别功能

        Args:
            node_name(str): 节点名称，用于识别结果的记录和返回
            roi(list[int]): 可选的ROI坐标，用于识别
            image(numpy.ndarray): 可选的自定义图像，用于识别
            template(str): 可选的模板名称，用于模板识别

        Returns:
            list[Any]: 识别到的结果，如果为空则返回空列表
        """
        if image is None:
            if self.image is None:
                self.image = self.context.tasker.controller.post_screencap().wait().get()
            image = self.image

        params = {}
        if roi:
            params["roi"] = roi
        if template:
            params["template"] = template
        pipeline_override = {node_name: {"recognition": {"param": params}}} if params else {}

        reco_detail = self.context.run_recognition(node_name, image, pipeline_override)

        # 统一的日志记录
        if reco_detail and reco_detail.all_results:
            if isinstance(reco_detail.all_results[0], OCRResult):
                logger.debug(
                    f"节点{node_name} OCR结果：{[(r.text, r.box, r.score) for r in reco_detail.all_results]}"
                )
            elif isinstance(reco_detail.all_results[0], TemplateMatchResult):
                logger.debug(f"节点'{node_name}'模板结果：{[(r.box, r.score) for r in reco_detail.filtered_results]}")
            else:
                logger.debug(f"节点'{node_name}'识别结果：{[(r.box, r.score) for r in reco_detail.filtered_results]}")
        else:
            logger.debug(f"节点'{node_name}'未识别到任何内容")

        # 输出结果
        return reco_detail.filtered_results if reco_detail else []

    def get_current_coin(self) -> int:
        ocr_results = self._recognize("星塔_通用_识别当前金币_agent")
        try:
            return int(ocr_results[0].text)
        except (ValueError, TypeError, IndexError):
            logger.error("未识别到当前金币，将默认为0")
            return 0

    def get_refresh_cost(self) -> int:
        ocr_results = self._recognize("星塔_通用_识别刷新花费_agent")
        try:
            return int(ocr_results[0].text)
        except (ValueError, TypeError, IndexError):
            logger.error("未识别到刷新花费，将默认为-1")
            return -1

    def check_core_potential(self) -> bool:
        return bool(self._recognize("星塔_节点_选择潜能_识别核心潜能_agent"))

    def check_level_upped(self) -> bool:
        return bool(self._recognize("星塔_节点_选择潜能_检测是否升级_agent"))

    def get_potential_name(self, roi: list[int]) -> str:
        ocr_results = self._recognize("星塔_节点_选择潜能_识别潜能名称_agent", roi=roi)
        return " ".join([t.text for t in ocr_results])

    def get_potential_level(self, roi: list[int]) -> tuple[int, int]:
        ocr_results = self._recognize("星塔_节点_选择潜能_识别潜能等级_agent", roi=roi)
        levels = self._parse_level_text([r.text for r in ocr_results])
        return levels

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
        logger.warning(f"无法解析潜能等级（识别到的等级文本: {texts}）")
        return -1, -1

    def get_recommend_level(self, roi: list[int]) -> int:
        ocr_results = self._recognize("星塔_节点_选择潜能_识别推荐等级_agent", roi=roi)
        try:
            return int(ocr_results[0].text)
        except (ValueError, TypeError, IndexError):
            logger.error("未识别到推荐等级，将默认为-1")
            return -1

    def check_item_list_visibility(self) -> bool:
        node = "星塔_节点_选择潜能_检测干扰文字_agent"
        ocr_results = self._recognize(node)
        if not ocr_results:
            return False
        # 由于maafw的改动，检测模型没有结果时会把整个roi当作检测结果进行识别，导致出现误判情况
        # 所以需要增加“目标ROI是否等于识别ROI”的判断，如果判断成功，则说明没有识别到干扰文字，返回False
        node_data = self.context.get_node_data(node)
        target_roi = node_data.get("recognition", {}).get("param", {}).get("roi", [])
        return tuple(ocr_results[0].box) != tuple(target_roi)

    def get_potential_types(self, core_potential: bool = False) -> list[str]:
        """
            检查可选潜能卡片类型

            Args:
                core_potential(bool): 是否为核心潜能，默认为False

            Returns:
                list: 可选潜能卡片类型列表，识别失败时返回3个普通潜能
        """
        if core_potential:
            return ["core", "core", "core"]

        normal_node_name = "星塔_节点_选择潜能_识别普通潜能数量_agent"
        rare_node_name = "星塔_节点_选择潜能_识别稀有潜能数量_agent"
        # 获取坐标
        normal_potentials = [r.box for r in self._recognize(normal_node_name)]
        rare_potentials = [r.box for r in self._recognize(rare_node_name)]
        # 打标签，按照x坐标排序
        potentials = [["normal", box] for box in normal_potentials] + [["rare", box] for box in rare_potentials]
        potentials.sort(key=lambda x: x[1][0])
        # 去掉坐标，只保留类型标签
        potential_types = [p[0] for p in potentials]
        potential_count = len(potential_types)

        if potential_count == 0:
            logger.error("潜能数量识别失败（没有潜能），将默认为3个普通潜能")
            return ["normal", "normal", "normal"]
        elif potential_count > 3:
            logger.error(f"潜能数量识别失败（识别到{potential_count}个潜能，不符合预期），将默认为3个普通潜能")
            return ["normal", "normal", "normal"]

        return potential_types

    def get_recommended_potential(self, borders: list[list]) -> list:
        """
        识别系统推荐图标，返回对应卡片的潜能序数列表。

        推荐图标位于卡片 box 范围内，通过判断图标命中 x 坐标是否落入各卡片
        x_border 区间来确定归属卡片。

        Args:
            borders: 可选潜能卡片区域列表，每个元素结构：[float, float],  # 卡片 x 轴边界（左闭右闭）

        Returns:
            list: 包含推荐潜顺序数的列表
        """
        reco_results = self._recognize("星塔_节点_选择潜能_识别推荐图标_agent")
        hit_xs = [r.box[0] for r in reco_results]
        matched = [
            i for i, (low, high) in enumerate(borders)
            if any(low <= x <= high for x in hit_xs)
        ]
        unmatched_xs = [
            x for x in hit_xs
            if not any(low <= x <= high for low, high in borders)
        ]

        if not matched:
            logger.debug("推荐图标识别失败，有可能是没有推荐图标，也有可能是识别问题")
        if unmatched_xs:
            logger.error(f"检测到 {len(unmatched_xs)} 个推荐图标超出所有潜能卡片边界: {unmatched_xs}，后续选择将会出现问题")

        return matched

    def check_potential_recommended(self, roi: list[int]) -> bool:
        return bool(self._recognize("星塔_节点_选择潜能_识别推荐图标_agent", roi=roi))

    def get_selected_potential_index(self, borders: list[list[int]]) -> int:
        """识别拿到按钮，返回对应卡片的索引。

        拿到按钮位于卡片 box 范围内，通过判断图标命中 x 坐标是否落入各卡片
        x_border 区间（左闭右闭）来确定归属卡片。
        识别失败时返回第一张卡片的索引 作为兜底。

        Args:
            borders(list): 可选潜能卡片的边界框，每个元素为一个列表，包含2个元素，分别是左闭右闭的x轴边界

        Returns:
            int: 目标卡片索引，识别失败时返回0。这里的索引是0-based的，适合给list使用。
        """
        result_boxes = [r.box for r in self._recognize("星塔_节点_选择潜能_识别预选潜能位置_agent")]

        hit_x = result_boxes[0][0] if result_boxes else -1
        matched = next((i for i, (low, high) in enumerate(borders) if low <= hit_x <= high), None)

        if matched is None:
            logger.error("拿到按钮识别失败，潜能选择可能会出现问题")
            matched = 0

        return matched

    def match_trekker(self, trekker_image: numpy.ndarray, roi: list[int]) -> bool:
        """匹配旅人"""
        self.context.override_image("trekker_image", trekker_image)
        return bool(self._recognize(
            "星塔_节点_选择潜能_识别旅人_agent",
            template="trekker_image",
            roi=roi
        ))
