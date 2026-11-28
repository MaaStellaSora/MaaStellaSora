"""Locate a shelf price on the current frame before the purchase pipeline clicks."""
import json

from maa.custom_recognition import CustomRecognition

SHOP_PRICE_RECOGNITION = "stella_value_shop_item_price"


def purchase_override(item, node="星塔_节点_商店_购物_购买道具_agent"):
    return {node: {
        "recognition": {"type": "Custom", "param": {
            "custom_recognition": SHOP_PRICE_RECOGNITION,
            "custom_recognition_param": {"roi": item.price_roi, "price": item.price,
                                         "internal_name": item.internal_name, "quantity": item.quantity},
        }},
        "action": {"type": "Click", "param": {"target": True}},
    }}


class ShopItemPrice(CustomRecognition):
    def analyze(self, context, argv):
        from .interactor import ShopInteractor
        from .price import price_key

        if context.tasker.stopping:
            return None
        params = json.loads(argv.custom_recognition_param)
        roi = params["roi"]
        screen = ShopInteractor(context)
        screen.image = argv.image
        from .context import ITEM_STANDARD_PRICES
        price, box = screen.read_item_price(roi,
            ITEM_STANDARD_PRICES.get(price_key(params["internal_name"], params["quantity"])))
        if price != params["price"]:
            return None
        return CustomRecognition.AnalyzeResult(box=box,
                                               detail={"price": price, "price_roi": roi})
