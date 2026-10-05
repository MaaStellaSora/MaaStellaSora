from maa.context import Context

from custom.interactor_template import Interactor
from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_shop_interactor")

class ShopInteractor(Interactor):
    def __init__(self, context: Context):
        super().__init__(context)

    def enter_shopping(self):
        """进入商店层购物界面。"""
        run_result = self.context.run_task("星塔_节点_商店_商店层_进入购物界面_agent")
        self.image = self.context.tasker.controller.cached_image
        if not (run_result and run_result.status.succeeded):
            logger.warning("进入购物界面失败")
            return False
        logger.debug("进入购物界面成功")
        return True

    def back_to_main_page(self):
        """返回商店层主界面。"""
        run_result = self.context.run_task("星塔_节点_商店_万能返回商店层_agent")
        if not (run_result and run_result.status.succeeded):
            logger.warning("返回商店层主界面失败")
            return False
        logger.debug("返回商店层主界面成功")
        return True

    def enhance(self) -> bool:
        """强化"""
        pipeline_override = {
            "星塔_节点_选择潜能_agent": {
                "attach": {
                    "potential_source": "enhance"
                }
            }
        }
        run_result = self.context.run_task("星塔_节点_商店_点击强化_agent", pipeline_override)
        if not (run_result and run_result.status.succeeded):
            logger.warning("强化失败，可能已经没有潜能可以强化")
            return False
        logger.info("强化成功")
        return True

    def get_current_coin(self) -> int:
        ocr_results = self._recognize("星塔_通用_识别当前金币_agent")
        try:
            return int(ocr_results[0].text)
        except (ValueError, TypeError, IndexError):
            logger.error("未识别到当前金币，将默认为-1")
            return -1

    def get_enhancement_cost(self) -> int:
        """识别当前强化所需金币数量。"""
        node = "星塔_节点_商店_识别强化所需金币_agent"
        ocr_results = self._recognize(node)
        try:
            return int(max(ocr_results, key=lambda r: r.score).text) if self._valid_ocr(node, ocr_results) else -1
        except (ValueError, TypeError, IndexError):
            logger.error("未识别到强化所需金币，将默认为-1")
            return -1

    def check_shop_type(self) -> str:
        """检查商店类型。中途商店为 regular，最终商店为 final，识别失败返回空字符串。"""
        if self._recognize("星塔_节点_商店_离开商店_agent"):
            return "regular"

        if self._recognize("星塔_节点_商店_离开星塔_agent"):
            return "final"

        return ""

    def get_refresh_remaining(self) -> int:
        """识别商店当前剩余刷新次数。"""
        node = "星塔_节点_商店_购物_识别可刷新次数_agent"
        ocr_results = self._recognize(node)
        try:
            return int(max(ocr_results, key=lambda r: r.score).text) if self._valid_ocr(node, ocr_results) else 0
        except (ValueError, TypeError, IndexError):
            logger.debug("未识别到剩余刷新次数，可能是刷新用完，也有可能识别错误。将默认为0")
            return 0

    def get_refresh_cost(self) -> int:
        """识别当前刷新费用。"""
        node = "星塔_通用_识别刷新花费_agent"
        ocr_results = self._recognize(node)
        try:
            return int(max(ocr_results, key=lambda r: r.score).text) if self._valid_ocr(node, ocr_results) else 0
        except (ValueError, TypeError, IndexError):
            logger.debug("未识别到刷新费用，可能是刷新用完，也有可能识别错误。将默认为0")
            return 0

    def recognize_item_name_quantity(self, roi: list[int]) -> list[str]:
        """识别商店当前物品名称及数量。"""
        ocr_results = self._recognize("星塔_节点_商店_购物_识别物品内容_agent", roi = roi)
        return [r.text for r in ocr_results]

    def recognize_item_price(self, roi: list[int]) -> list[str]:
        """识别商店当前物品价格。"""
        ocr_results = self._recognize("星塔_节点_商店_购物_识别物品价格_agent", roi = roi)
        return [r.text for r in ocr_results]

    def is_specified_drink(self, roi: list[int]) -> bool:
        """检查潜能特饮是否为旅人限定，该识别使用反向识别法"""
        return not self._recognize("星塔_节点_商店_购物_识别非旅人限定特饮_agent", roi = roi)

    def locate_assist_melody_text(self) -> list[dict]:
        """检查协奏技能的音符文本颜色是否为红色，是则返回坐标列表，否则返回空列表。"""
        if filtered_results := self._recognize("星塔_节点_商店_购买协奏音符_核实红色_agent"):
            return [r.box for r in filtered_results]
        return []

    def get_assist_melody_count(self, roi: list[int]) -> str:
        """识别协奏技能的现有音符 / 下一级目标音符数量。"""
        if filtered_results := self._recognize("星塔_节点_商店_购买协奏音符_核实数量_agent", roi = roi):
            return max(filtered_results, key=lambda r: r.score).text
        return ""
