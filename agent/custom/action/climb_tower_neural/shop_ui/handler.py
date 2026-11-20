"""Neural shelf reading, refresh confirmation and enhancement operations."""
import re

from custom.action.climb_tower_shop.handler import ShopHandler as NativeShopHandler

from .context import Item, ShopContext
from .interactor import ShopInteractor
from ..ui_flow import read_until
from utils import logger as logger_module

logger = logger_module.get_logger(__name__)


class ShopHandler:
    _get_item_name_and_quantity = NativeShopHandler._get_item_name_and_quantity
    _parse_item_name = NativeShopHandler._parse_item_name
    update_coin = NativeShopHandler.update_coin
    _update_enhancement_cost = NativeShopHandler._update_enhancement_cost
    _update_shop_type = NativeShopHandler._update_shop_type

    def __init__(self, interactor: ShopInteractor, data: ShopContext):
        self.interactor = interactor
        self.data = data
        self.valid_enhancement_costs = None
        self.refresh_attempted = False

    def read_main_page_info(self, *, read_coin=False, valid_costs=None):
        """在同一主页面补齐强化费；强化后同时确认新的钱包。"""
        if valid_costs is not None:
            self.valid_enhancement_costs = valid_costs
        valid_costs = self.valid_enhancement_costs
        cost_read, coin_read = False, not read_coin
        self.data.current_enhancement_cost = -1
        if read_coin:
            self.data.current_coin = -1

        def observe(image):
            nonlocal cost_read, coin_read
            self.interactor.image = image
            if not self.interactor.is_main_page():
                return None
            if not self.data.shop_type:
                self._update_shop_type()
            if not cost_read:
                self._update_enhancement_cost()
                cost = self.data.current_enhancement_cost
                cost_read = cost in valid_costs if valid_costs is not None else cost >= 0
            if not coin_read:
                self.update_coin()
                coin_read = self.data.current_coin >= 0
            return True if self.data.shop_type and cost_read and coin_read else None

        return read_until(self.interactor.context, observe, initial_image=self.interactor.image) is not None

    def read_goods_page_info(self, *, enter_shopping: bool = True, refresh_exhausted: bool = False):
        """同一稳定货架只补缺项，导航和交易不在重读中重复。"""
        if enter_shopping:
            self.interactor.enter_shopping()
        if refresh_exhausted:
            self.data.refresh_remaining = 0
            self.data.refresh_remaining_observed = -1
            self.data.refresh_remaining_basis = "global_confirmed_refreshes_exhausted"
            self.data.refresh_cost = -1
        coin_read = False
        self.data.current_coin = -1

        def observe(image):
            nonlocal coin_read
            self.interactor.image = image
            if not self.interactor.is_shopping():
                return None
            self._update_refresh_info(missing_only=True)
            if not coin_read:
                self.update_coin()
                coin_read = self.data.current_coin >= 0
            self._update_goods_info(missing_only=True)
            complete = (coin_read and not self.data.goods_missing
                        and 0 <= self.data.refresh_remaining <= 2
                        and (self.data.refresh_remaining == 0 or self.data.refresh_cost > 0))
            return True if complete else None

        return read_until(self.interactor.context, observe, initial_image=self.interactor.image) is not None

    def refresh(self, *, frame_capture=None) -> bool:
        """以次数减少或实付费用与完整货架确认刷新，包括末次控件消失。"""
        self.refresh_attempted = False
        if self.interactor.context.tasker.stopping:
            return False
        # 确保处于商店购物界面
        self.interactor.enter_shopping()

        current_refresh_remaining = self.data.refresh_remaining
        coin_read, cost_read = False, False

        def before_refresh(image):
            nonlocal coin_read, cost_read
            self.interactor.image = image
            if not self.interactor.is_shopping():
                return None
            if not coin_read:
                self.update_coin()
                coin_read = self.data.current_coin >= 0
            if not cost_read:
                self.data.refresh_cost = self.interactor.get_refresh_cost()
                cost_read = self.data.refresh_cost > 0
            if coin_read and cost_read:
                return self.data.current_coin, self.data.refresh_cost
            return None

        before = read_until(self.interactor.context, before_refresh, initial_image=self.interactor.image)
        if before is None:
            return False
        coin_before, price = before
        self.refresh_attempted = True
        self.interactor.context.run_task("星塔_节点_商店_点击刷新_agent")
        if self.interactor.context.tasker.stopping:
            return False

        previous = None
        new_scan = True
        stable = False
        basis = None
        self.data.refresh_cost = -1

        def observed(image):
            nonlocal previous, new_scan, stable, basis
            self.interactor.image = image
            if not self.interactor.is_shopping():
                return None
            if stable:
                if self.data.current_coin < 0:
                    self.update_coin()
                if self.data.refresh_cost <= 0 and current_refresh_remaining > 1:
                    self.data.refresh_cost = self.interactor.get_refresh_cost()
                if self.data.current_coin >= 0 and (current_refresh_remaining == 1 or self.data.refresh_cost > 0):
                    return basis, current_refresh_remaining - 1
                return None
            self.update_coin()
            debit_confirmed = self.data.current_coin >= 0 and coin_before - self.data.current_coin == price
            # The second global refresh removes these controls. A paid,
            # complete shelf can contain the same goods as the previous one.
            remaining = (-1 if debit_confirmed and current_refresh_remaining == 1
                         else self.interactor.get_refresh_remaining())
            self.data.refresh_remaining_observed = remaining
            count_confirmed = remaining >= 0 and remaining == current_refresh_remaining - 1
            if not (count_confirmed or debit_confirmed and remaining < 0):
                previous, new_scan = None, True
                return None
            if new_scan:
                self.data.items = []
                new_scan = False
            self._update_goods_info(missing_only=True, preserve_bought=False)
            complete = (len(self.data.items) == len(self.data.params.item_rois)
                        and {item.grid_num for item in self.data.items}
                        == set(range(1, len(self.data.params.item_rois) + 1))
                        and all(item.internal_name and item.quantity > 0 and item.price > 0
                                for item in self.data.items))
            if not complete:
                return None
            stock = tuple(sorted((item.grid_num, item.internal_name, item.quantity,
                                  item.price, item.trekker_specified) for item in self.data.items))
            basis = ("ocr_after_refresh" if count_confirmed else
                     "confirmed_refresh_debit_and_new_stock")
            result = basis, current_refresh_remaining - 1, stock
            stable = result == previous
            previous, new_scan = result, True
            if not stable:
                return None
            if current_refresh_remaining > 1:
                self.data.refresh_cost = self.interactor.get_refresh_cost()
            return ((basis, current_refresh_remaining - 1)
                    if self.data.current_coin >= 0
                    and (current_refresh_remaining == 1 or self.data.refresh_cost > 0) else None)

        def observe_frame(image):
            return (observed(image) if frame_capture is None else
                    frame_capture(observed, image, new_scan=new_scan and not stable))

        confirmed = read_until(self.interactor.context, observe_frame)
        if confirmed is not None:
            self.data.refresh_remaining_basis, self.data.refresh_remaining = confirmed
            logger.debug("刷新成功")
            return True
        if self.interactor.context.tasker.stopping:
            return False
        logger.debug("商店刷新读数或操作确认仍缺失")
        return False

    def _enhance(self) -> bool:
        """进行一次强化"""
        # 确保在商店层主界面
        self.interactor.back_to_main_page()

        # 开始强化
        if not self.interactor.enhance():
            return False

        # 更新强化费用与辉光币数量
        self.interactor.update_image()
        self.read_main_page_info(read_coin=True)
        return True

    def _update_goods_info(self, *, missing_only=False, preserve_bought=True):
        """识别购物界面 8 个格子的道具信息。

        每格保留名称、数量、实价及缺失字段；补读只填缺口。
        同架保留已确认的购买，刷新则重新读取全部格子。
        """
        previous = {item.grid_num: item for item in self.data.items}
        items = []
        lang = self.data.params.lang

        for i, item_roi in enumerate(self.data.params.item_rois):
            if self.interactor.context.tasker.stopping:
                break
            logger.debug(f"正在识别第 {i + 1} 个格子")
            item = previous.get(i + 1)
            if item is not None and (preserve_bought and item.bought or missing_only and not item.missing):
                items.append(item)
                continue
            if not missing_only or item is None:
                item = Item(grid_num=i + 1, price=-1, missing=["name", "quantity", "price"])
            if "name" in item.missing or "quantity" in item.missing:
                name, quantity = self._get_item_name_and_quantity(item_roi)
                if "name" in item.missing:
                    item.internal_name = name
                if "quantity" in item.missing:
                    item.quantity = quantity
                if item.internal_name == "potential_drink":
                    item.trekker_specified = self.interactor.is_specified_drink(item_roi["item_roi"])
            known_name = item.internal_name in self.data.params.item_translations
            if "price" in item.missing and known_name and item.quantity > 0:
                item.price = self._get_item_price(item_roi, item.internal_name, item.quantity)
            item.missing = (["name", "quantity", "price"] if not known_name else
                            (["quantity"] if item.quantity <= 0 else [])
                            + (["price"] if item.price <= 0 else []))
            item.display_name = self.data.params.item_translations.get(item.internal_name, {}).get(lang, ["?"])[0]
            items.append(item)
            if item.missing:
                logger.debug(f"第 {i + 1} 格商品缺读: {item.missing}, "
                             f"name={item.internal_name!r}, quantity={item.quantity}, price={item.price}")

        logger.debug(f"道具列表: {items}")
        self.data.items = items

    def _get_item_price(self, item_roi: dict[str, list[int]], item_name: str, quantity: int) -> int:
        """通过识别获得商品价格"""
        roi = item_roi["price_roi"]
        reco_result = self.interactor.recognize_item_price(roi)
        price_key = "potential_drink" if item_name == "potential_drink" else f"melody_{quantity}"
        standard_price = (self.data.params.item_standard_prices.get(price_key)
                          if item_name == "potential_drink" or item_name.startswith("melody_of_") else None)
        item_price = self._clean_and_extract_price(reco_result, standard_price)
        return item_price

    @staticmethod
    def _clean_and_extract_price(reco_result: list[str], standard_price: int | None) -> int:
        """按商品原价拆开同框的原价/实价；五折与八折冲突时保持未确认。"""
        if standard_price is None:
            return -1
        discounts = {standard_price // 2, standard_price * 4 // 5}
        values = set()
        original = str(standard_price)
        for text in reco_result:
            text = text.strip()
            if not re.fullmatch(r"[0-9]+", text):
                continue
            text = str(int(text))
            if text.startswith(original) and text[len(original):]:
                reduced = int(text[len(original):])
                if reduced in discounts:
                    values.add(reduced)
            elif int(text) in discounts | {standard_price}:
                values.add(int(text))
        discounted = values & discounts
        if len(discounted) == 1:
            return discounted.pop()
        return standard_price if values == {standard_price} else -1

    def _update_refresh_info(self, *, missing_only=False):
        """更新刷新相关信息"""
        if not missing_only or not 0 <= self.data.refresh_remaining <= 2:
            self.data.refresh_remaining = self.interactor.get_refresh_remaining()
            self.data.refresh_remaining_observed = self.data.refresh_remaining
            self.data.refresh_remaining_basis = "ocr" if self.data.refresh_remaining >= 0 else "unconfirmed"
        if (not missing_only or self.data.refresh_cost <= 0) and self.data.refresh_remaining != 0:
            self.data.refresh_cost = self.interactor.get_refresh_cost()
