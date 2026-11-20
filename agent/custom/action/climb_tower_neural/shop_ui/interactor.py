import re
import unicodedata

from custom.action.climb_tower_shop.interactor import ShopInteractor as NativeShopInteractor

from ..interactor import Interactor
from ..ocr_number import assess_amount, decimal_amount
from utils import logger as logger_module
logger = logger_module.get_logger(__name__)

REFRESH_COST_NODE = "星塔_神经网络_识别刷新花费"
PRICE_NODE = "星塔_神经网络_商店_识别物品价格"
REMAINING_NODE = "星塔_神经网络_商店_识别可刷新次数"
ENHANCEMENT_NODE = "星塔_神经网络_商店_识别强化所需金币"


class ShopInteractor(Interactor):
    check_shop_type = NativeShopInteractor.check_shop_type
    recognize_item_name_quantity = NativeShopInteractor.recognize_item_name_quantity
    is_specified_drink = NativeShopInteractor.is_specified_drink

    def enter_shopping(self):
        """进入商店层购物界面。"""
        run_result = self.context.run_task("星塔_节点_商店_商店层_进入购物界面_agent")
        self.image = self.context.tasker.controller.cached_image
        if not (run_result and run_result.status.succeeded):
            logger.error(
                "$agent.custom.action.climb_tower_shop.interactor.enter_shopping.shop_open_failed"
            )
            return False
        logger.debug("进入购物界面成功")
        return True

    def back_to_main_page(self):
        """返回商店层主界面。"""
        run_result = self.context.run_task("星塔_节点_商店_万能返回商店层_agent")
        self.update_image()
        if not (run_result and run_result.status.succeeded):
            logger.error(
                "$agent.custom.action.climb_tower_shop.interactor.back_to_main_page.shop_return_failed"
            )
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
            logger.error(
                "$agent.custom.action.climb_tower_shop.interactor.enhance.enhancement_failed"
            )
            return False
        logger.debug(
            "$agent.custom.action.climb_tower_shop.interactor.enhance.enhancement_succeeded"
        )
        return True

    def get_current_coin(self) -> int:
        coin = super().get_current_coin()
        if coin < 0:
            logger.debug("当前商店画面的金币未读完整")
        return coin

    def get_enhancement_cost(self) -> int:
        """识别当前强化所需金币数量。"""
        node = ENHANCEMENT_NODE
        result = self.context.run_recognition(node, self.image)
        cost = None
        if result and result.hit and self._valid_ocr(node, result.all_results):
            params = (self.context.get_node_data(node) or {}).get("recognition", {}).get("param", {})
            threshold = params.get("threshold", .3)
            free_pattern = (self.context.get_node_data(node) or {}).get("attach", {}).get("free_pattern", "")
            amounts, observed = [], []
            for row in result.all_results:
                text = unicodedata.normalize("NFKC", row.text).strip()
                match = re.fullmatch(r".*\(([^()]*)\)\s*", text)
                token = match[1].strip() if match else text
                token = "0" if free_pattern and re.fullmatch(free_pattern, token, re.I) else token
                value = decimal_amount(token)
                amount = str(value) if value is not None else text
                if value is not None and row.score >= threshold:
                    amounts.append(amount)
                observed.append((amount, row.score))
            cost, _, _ = assess_amount(amounts, observed)
        cost = -1 if cost is None else cost
        if cost < 0:
            logger.debug("当前商店画面的强化费未读完整")
        return cost

    def get_refresh_remaining(self) -> int:
        """识别商店当前剩余刷新次数。"""
        return self._read_number(REMAINING_NODE)

    def get_refresh_cost(self) -> int:
        """识别当前刷新费用。"""
        return self._read_number(REFRESH_COST_NODE)

    def recognize_item_price(self, roi: list[int]) -> list[str]:
        """识别商店当前物品价格。"""
        ocr_results = self._recognize(PRICE_NODE, roi = roi)
        return [r.text for r in ocr_results]
