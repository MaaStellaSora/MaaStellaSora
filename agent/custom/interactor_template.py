"""
通用交互模板
"""

import random
from typing import Any

import numpy as np
from maa.context import Context
from maa.define import OCRResult, TemplateMatchResult, ColorMatchResult

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_interactor")


class Interactor:
    def __init__(self, context: Context):
        self.context = context
        self.image = self.context.tasker.controller.cached_image

    def screenshot(self):
        """截图并保存到self.image"""
        self.image = self.context.tasker.controller.post_screencap().wait().get()

    def update_image(self):
        """通过缓存更新截图到self.image，减轻截图压力"""
        self.image = self.context.tasker.controller.cached_image

    def _recognize(
            self,
            node_name: str,
            *,
            roi: list[int] | None = None,
            image: np.ndarray | None = None,
            template: str | None = None
    ) -> list[Any]:
        """
        带有agent日志输出的简单识别功能，方便开发时查看识别结果

        Args:
            node_name(str): 节点名称，用于识别结果的记录和返回
            roi(list[int]): 可选的ROI坐标，用于识别
            image(numpy.ndarray): 可选的自定义图像，用于识别
            template(str): 可选的模板名称，用于模板识别

        Returns:
            list[Any]: 识别到的结果，如果为空则返回空列表
        """
        if image is None:
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
                results = [(r.text, r.box, r.score) for r in reco_detail.all_results if isinstance(r, OCRResult)]
                logger.debug(f"OCR节点'{node_name}'结果：{results}")
            elif isinstance(reco_detail.all_results[0], TemplateMatchResult):
                results = [(r.box, r.score) for r in reco_detail.all_results if isinstance(r, TemplateMatchResult)]
                logger.debug(f"模板识别节点'{node_name}'结果：{results}")
            elif isinstance(reco_detail.all_results[0], ColorMatchResult):
                results = [(r.box, r.count) for r in reco_detail.all_results if isinstance(r, ColorMatchResult)]
                logger.debug(f"颜色识别节点'{node_name}'结果：{results}")
        else:
            logger.debug(f"节点'{node_name}'未识别到任何内容")

        # 输出结果
        return reco_detail.filtered_results if reco_detail else []

    def click(self, box: list[int]) -> bool:
        """点击指定box，返回点击结果，True表示点击成功，False表示点击失败"""
        click_x = random.randint(box[0], box[0] + box[2])
        click_y = random.randint(box[1], box[1] + box[3])
        return self.context.tasker.controller.post_click(click_x, click_y).wait().succeeded

    def _valid_ocr(self, node: str, ocr_results: list[Any]) -> bool:
        """
        检查OCR结果是否有效

        由于maafw的改动，检测模型没有结果时会把整个roi当作检测结果进行识别，导致出现误判情况
        所以需要额外增加“非仅识别模式时目标ROI是否等于识别ROI”的判断，如果判断成功，说明本次识别到的结果是无效的，返回False

        该函数需要确保pipeline使用v2格式编写才有效
        """
        if not ocr_results: # 检测不到文字基本确认没有误触发only_rec模式，返回True
            return True
        node_data = self.context.get_node_data(node) or {}
        param = node_data.get("recognition", {}).get("param", {})
        target_roi = param.get("roi", [])
        only_rec = param.get("only_rec", False)
        return only_rec or tuple(ocr_results[0].box) != tuple(target_roi)
