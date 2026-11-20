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
        from .handler import ShopHandler
        from .interactor import PRICE_NODE, ShopInteractor

        if context.tasker.stopping:
            return None
        params = json.loads(argv.custom_recognition_param)
        roi = params["roi"]
        screen = ShopInteractor(context)
        screen.image = argv.image
        rows = screen._recognize(PRICE_NODE, roi=roi)
        # OCR can fall back to treating its entire ROI as a recognized box.
        if not rows or any(tuple(row.box) == tuple(roi) for row in rows):
            return None
        price_key = ("potential_drink" if params["internal_name"] == "potential_drink"
                     else f"melody_{params['quantity']}")
        from .context import ITEM_STANDARD_PRICES
        price = ShopHandler._clean_and_extract_price(
            [row.text for row in rows], ITEM_STANDARD_PRICES.get(price_key))
        if price != params["price"]:
            return None
        left, top = min(row.box[0] for row in rows), min(row.box[1] for row in rows)
        right = max(row.box[0] + row.box[2] for row in rows)
        bottom = max(row.box[1] + row.box[3] for row in rows)
        return CustomRecognition.AnalyzeResult(box=(left, top, right - left, bottom - top),
                                               detail={"price": price, "price_roi": roi})
