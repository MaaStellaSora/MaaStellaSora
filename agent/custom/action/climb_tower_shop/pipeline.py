from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .handler import ShopHandler
from .context import ShopContext, Item, Melody
from .pipeline_buyer import buy_general, buy_assist_melody

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_shop_pipeline")


SortFunc = Callable[[list[Item]], list[Item]]
FilterFunc = Callable[[list[Item], ShopContext], Item | None]
BuyFunc = Callable[[Item, "ShopHandler"], bool]


class BuyPipeline:
    """购买流水线：对数据集进行 排序 -> 筛选 -> 循环尝试买入。"""

    def __init__(self, name: str, sorter: SortFunc, filterer: FilterFunc, buyer: BuyFunc):
        self.name = name
        self.sorter = sorter
        self.filterer = filterer
        self.buyer = buyer

    def step(self, handler: ShopHandler) -> bool:
        """
        执行单次评估与买入。
        返回 True 表示成功进行了一次买入（状态已更新，需要重新循环评估）；
        返回 False 表示当前策略已无可执行目标。
        """
        if handler.interactor.context.tasker.stopping:
            return False

        # 1. 实时排序
        sorted_items = self.sorter(handler.data.items)

        # 2. 实时 Filter：根据最新的 ShopContext 判断出当前【唯一】可买目标
        target_item = self.filterer(sorted_items, handler.data)
        if target_item is None:
            return False

        # 3. 执行 Buy：由 BuyFunc 全权负责 UI 买入 + 后置状态修改（扣金币、减库存、记已买等）
        logger.debug(f"策略 [{self.name}] 选中目标: {target_item.internal_name} (格子 {target_item.grid_num})")
        self.buyer(target_item, handler)
        return True


# ----------------- Sorter -----------------

def sort_by_price(items: list[Item]) -> list[Item]:
    """按 price 顺序对商品列表进行从低到高排序。"""
    items.sort(key=lambda g: g.price)
    return items

def sort_by_price_desc(items: list[Item]) -> list[Item]:
    """按 price 顺序对商品列表进行从高到低排序。"""
    items.sort(key=lambda g: -g.price)
    return items

def sort_drink(items: list[Item]) -> list[Item]:
    """按 price 顺序对饮料商品列表进行从低到高排序，且优先有指定旅人的饮料。"""
    items.sort(key=lambda g: (g.price, not g.trekker_specified))
    return items

# ----------------- Filterer -----------------

def drink(items: list[Item], data: ShopContext) -> Item | None:
    """
    按照以下要求筛选出饮料商品。

    1. 价格符合折扣要求的
    2. 还没卖掉的
    3. 够钱买的
    """
    reserved_coin = data.total_enhancement_cost
    def condition(item: Item) -> bool:
        bool1 = item.discount <= data.params.drink_discount_threshold
        bool2 = (data.current_coin - reserved_coin) >= item.price
        return bool1 and bool2
    return _drink(items, condition)

def all_drink(items: list[Item], data: ShopContext) -> Item | None:
    """
    按照以下要求筛选出饮料商品。

    1. 还没卖掉的
    2. 够钱买的（这里还需要加上额外预留的辉光币）
    """
    reserved_coin = data.total_enhancement_cost + data.dynamic_reserve
    def condition(item: Item) -> bool:
        return (data.current_coin - reserved_coin) >= item.price
    return _drink(items, condition)

def remaining_drink(items: list[Item], data: ShopContext) -> Item | None:
    """
    仅限最终商店且无法刷新时，按照以下要求筛选出饮料商品。

    1. 还没卖掉的
    2. 够钱买的
    3. 在最终商店
    4. 已经不能刷新了
    """
    if data.shop_type != "final":
        return None
    if data.should_refresh:
        return None
    reserved_coin = data.total_enhancement_cost
    def condition(item: Item) -> bool:
        return (data.current_coin - reserved_coin) >= item.price
    return _drink(items, condition)

def _drink(items: list[Item], condition: Callable[[Item], bool] | None = None) -> Item | None:
    """
    按照以下要求筛选出饮料商品。

    1. 还没卖掉的
    2. 其他条件
    """
    items = (
        item for item in items
        if item.internal_name == "potential_drink" and not item.bought
           and (condition is None or condition(item) is True)
    )
    return next(items, None)

def melody(items: list[Item], data: ShopContext) -> Item | None:
    """
    按照以下要求筛选出音符商品。

    1. 价格符合要求的
    2. 还没卖掉的
    3. 当前数量小于目标数量的
    4. 如果限制了购买时机，还要验证购买时机
    """
    if data.params.buy_melody_at_final_only and data.shop_type != "final":
        return None

    def condition(item: Item) -> bool:
        """判断音符是否当前数量小于目标数量"""
        m = data.melodies.get(item.internal_name, Melody())
        return m.count < m.required_count

    return _melody(items, data, condition)

def assist_melody(items: list[Item], data: ShopContext) -> Item | None:
    """
    按照以下要求筛选出协奏音符商品。

    1. 价格符合要求的
    2. 还没卖掉的
    3. 开了购买协奏音符的功能
    4. 没有被check过协奏音符的
    5. 如果限制了购买时机，还要验证购买时机
    """
    if not data.params.buy_assist_melody:
        return None
    if data.params.buy_melody_at_final_only and data.shop_type != "final":
        return None

    def condition(item: Item) -> bool:
        """判断音符是否已经验证过协奏情况，如果验证过则返回False"""
        return not item.checked

    return _melody(items, data, condition)

def _melody(items: list[Item], data: ShopContext, condition: Callable[[Item], bool] | None = None) -> Item | None:
    """
    按照以下要求筛选出音符商品。

    1. 价格符合要求的
    2. 还没卖掉的
    3. 够钱买的
    4. 其他条件
    """
    reserved_coin = data.total_enhancement_cost
    thresholds = {5: data.params.melody_5_discount_threshold, 15: data.params.melody_15_discount_threshold}
    items = (
        item for item in items
        if "melody" in item.internal_name and not item.bought
           and item.discount <= thresholds.get(item.quantity, 1)
           and (condition is None or condition(item) is True)
           and (data.current_coin - reserved_coin) >= item.price
    )
    return next(items, None)

def final_remainder(items: list[Item], data: ShopContext) -> Item | None:
    """
    仅限最终商店且无法刷新时，按照以下要求筛选出零头商品。

    1. 还没卖掉的
    2. 够钱买的
    3. 在最终商店
    4. 已经不能刷新了
    """
    if data.shop_type != "final":
        return None
    if data.should_refresh:
        return None
    reserved_coin = data.greedy_enhancement_cost
    items = (
        item for item in items
        if not item.bought
           and (data.current_coin - reserved_coin) >= item.price
    )
    return next(items, None)


# ----------------- 策略设计 -----------------

DRINK_BUY_STRATEGY = BuyPipeline(name="潜能特饮策略", sorter=sort_drink, filterer=drink, buyer=buy_general)
MELODY_BUY_STRATEGY = BuyPipeline(name="音符策略", sorter=sort_by_price, filterer=melody, buyer=buy_general)
ASSIST_MELODY_STRATEGY = BuyPipeline(name="协奏音符策略", sorter=sort_by_price, filterer=assist_melody, buyer=buy_assist_melody)
HIGH_PRICE_DRINKS_STRATEGY = BuyPipeline(name="高价特饮策略", sorter=sort_drink, filterer=all_drink, buyer=buy_general)
REMAINING_DRINKS_STRATEGY = BuyPipeline(name="剩余特饮策略", sorter=sort_drink, filterer=remaining_drink, buyer=buy_general)
FINAL_REMAINDER_STRATEGY = BuyPipeline(name="零头策略", sorter=sort_by_price_desc, filterer=final_remainder, buyer=buy_general)
