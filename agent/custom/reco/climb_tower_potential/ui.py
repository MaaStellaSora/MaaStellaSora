import time
from typing import Optional, Any

import numpy
from maa.context import Context

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential_ui")


class UIInteractor:
    def __init__(self, context: Context):
        self.context = context
        self.image = None
        self.max_try = 1

    def load_last_screenshot(self):
        self.image = self.context.tasker.controller.cached_image

    def screenshot(self):
        self.image = self.context.tasker.controller.post_screencap().wait().get()

    def crop_screenshot(self, roi: list[int]) -> numpy.ndarray:
        return self.image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    def refresh(self):
        self.context.run_task("星塔_节点_选择潜能_点击刷新_agent")

    def click_potential(self, box: list[int]) -> bool:
        """点击指定box"""
        pipeline_override = {
            "星塔_节点_选择潜能_点击潜能_agent": {
                "action": {
                    "param": {
                        "target": box
                    }
                }
            }
        }
        run_result = self.context.run_task("星塔_节点_选择潜能_点击潜能_agent", pipeline_override)
        if run_result and run_result.status.succeeded:
            return True
        return False

    def _base_recognition(
            self, mode, node_name, failed_return, *, roi=None, image=None, template=None, max_try=0
    ) -> Any:
        """
        核心识别逻辑：支持 OCR 和 Template
        识别失败时返回默认值

        Args:
            mode(str): 识别模式，"ocr"或"template"或"color"
            node_name(str): 节点名称，用于识别结果的记录和返回
            failed_return(any): 识别失败时的默认值
            roi(tuple): 可选的ROI坐标，用于模板识别
            image(numpy.ndarray): 可选的自定义图像，用于识别
            template(str): 可选的模板名称，用于模板识别
            max_try(int): 可选的最大重试次数，默认为1

        Returns:
            any: 识别到的结果，根据mode返回文本或坐标列表
        """
        if image is None:
            if self.image is None:
                self.image = self.context.tasker.controller.post_screencap().wait().get()
            image = self.image

        actual_max_try = max_try if max_try > 0 else self.max_try

        params = {}
        if roi:
            params["roi"] = roi
        if template and mode == "template":
            params["template"] = template
        pipeline_override = {node_name: {"recognition": {"param": params}}} if params else {}

        try_count = 0
        while True:
            reco_detail = self.context.run_recognition(node_name, image, pipeline_override)

            if reco_detail and reco_detail.hit:
                if mode == "ocr":
                    # OCR 逻辑：返回文本列表，包含坐标和分数
                    logger.debug(
                        f"节点{node_name} OCR结果：{[(r.text, r.box, r.score) for r in reco_detail.filtered_results]}"
                    )
                    results = reco_detail.filtered_results
                    return [(r.text, r.box) for r in results]
                elif mode == "color":
                    # Color 逻辑：返回布尔结果
                    return True if reco_detail.filtered_results else False
                else:
                    # Template 逻辑：返回坐标列表，包含分数
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

    def _ocr(self, node_name, failed_return, **kwargs) -> list[tuple[str, list[int]]]:
        return self._base_recognition("ocr", node_name, failed_return, **kwargs)

    def _template(self, node_name, failed_return, **kwargs) -> list[list[int]]:
        return self._base_recognition("template", node_name, failed_return, **kwargs)

    def _color(self, node_name, **kwargs) -> bool:
        return self._base_recognition("color", node_name, False, **kwargs)

    def get_current_coin(self, image: Optional[numpy.ndarray] = None, max_try: int = 1) -> int:
        ocr_results = self._ocr("星塔_通用_识别当前金币_agent", [["0"]], image=image, max_try=max_try)
        return int(ocr_results[0][0])

    def get_refresh_cost(self, image: Optional[numpy.ndarray] = None, max_try: int = 1) -> int:
        ocr_results = self._ocr("星塔_通用_识别刷新花费_agent", [["-1"]], image=image, max_try=max_try)
        return int(ocr_results[0][0])

    def check_core_potential(self, image: Optional[numpy.ndarray] = None, max_try: int = 1) -> bool:
        if self._template("星塔_节点_选择潜能_识别核心潜能_agent", [], image=image, max_try=max_try):
            return True
        return False

    def get_potential_name(self, roi: list[int], image: Optional[numpy.ndarray] = None, max_try: int = 1) -> str:
        node_name = "星塔_节点_选择潜能_识别潜能名称_agent"
        ocr_results = self._ocr(node_name, [["", []]], roi=roi, image=image, max_try=max_try)
        return " ".join([t for t, _ in ocr_results])

    def get_potential_level(
            self,
            roi: list[int],
            image: Optional[numpy.ndarray] = None,
            max_try: int = 1
    ) -> tuple[int, int]:
        node_name = "星塔_节点_选择潜能_识别潜能等级_agent"
        ocr_results = self._ocr(node_name, [["", []]], roi=roi, image=image, max_try=max_try)
        levels = self._parse_level_text([t for t, _ in ocr_results])
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

    def get_recommend_level(self, roi: list[int], image: Optional[numpy.ndarray] = None, max_try: int = 1) -> int:
        node_name = "星塔_节点_选择潜能_识别推荐等级_agent"
        ocr_results = self._ocr(node_name, [["0"]], roi=roi, image=image, max_try=max_try)
        return int(ocr_results[0][0])

    def check_item_list_visibility(self, max_try: int = 1) -> bool:
        ocr_results = self._ocr("星塔_节点_选择潜能_检测干扰文字_agent", [], max_try=max_try)
        return len(ocr_results) != 0

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

    def check_potential_recommended(
            self,
            roi: list[list],
            image: Optional[numpy.ndarray] = None,
            max_try: int = 1
    ) -> bool:
        node_name = "星塔_节点_选择潜能_识别推荐图标_agent"
        recommended_boxes = self._template(node_name, [], roi=roi, image=image, max_try=max_try)
        if recommended_boxes:
            return True
        return False

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
            int: 目标卡片索引，识别失败时返回0。这里的索引是0-based的，适合给list使用。
        """
        result_boxes = self._template(
            "星塔_节点_选择潜能_识别预选潜能位置_agent",
            [],
            image=image,
            max_try=max_try
        )

        hit_x = result_boxes[0][0] if result_boxes else -1
        matched = next(i for i, (low, high) in enumerate(borders) if low <= hit_x <= high)

        if matched is None:
            logger.error("拿走按钮识别失败，潜能选择可能会出现问题")
            matched = 0

        return matched

    def match_trekker(self, trekker_image: numpy.ndarray, roi: list[int]) -> bool:
        """匹配旅人"""
        self.context.override_image("trekker_image", trekker_image)
        reco_results = self._template(
            "星塔_节点_选择潜能_识别旅人_agent",
            [],
            template="trekker_image",
            roi=roi
        )
        return len(reco_results) != 0

