import re
import time
from dataclasses import dataclass, field, fields
from typing import Optional, Any, Self

import numpy

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_action")


def get_current_coin(
    context: Context,
    image: Optional[numpy.ndarray] = None,
    max_try: int = 3,
) -> int:
    """获取当前金币数量。

    Args:
        context: 任务上下文。
        image: 截图，为 None 时自动截图。
        max_try: 最大重试次数。

    Returns:
        int: 当前金币数量，识别失败时返回 0。
    """

    if image is None:
        image = context.tasker.controller.post_screencap().wait().get()

    for _ in range(max_try):
        reco_detail = context.run_recognition("星塔_通用_识别当前金币_agent", image)
        if reco_detail and reco_detail.hit:
            logger.debug(f"识别到当前金币：{[r.text for r in reco_detail.filtered_results]}")
            return int(reco_detail.filtered_results[-1].text)

        if reco_detail and reco_detail.all_results:
            logger.debug(f"识别当前金币结果：{[r.text for r in reco_detail.all_results]}")
        else:
            logger.debug("未识别到任何关于当前金币的内容")
        logger.debug("等待1秒后重新识别")

        if context.tasker.stopping:
            return 0

        time.sleep(1)
        image = context.tasker.controller.post_screencap().wait().get()

    logger.error("无法读取当前金币数量，将当作 0 金币处理")
    return 0


def get_enhancement_cost(
    context: Context,
    image: Optional[numpy.ndarray] = None,
) -> int:
    """识别当前强化所需金币数量。

    Args:
        context: 任务上下文。
        image: 截图，为 None 时自动截图。

    Returns:
        int: 当前强化所需金币；识别失败返回 65535。
    """
    if image is None:
        image = context.tasker.controller.post_screencap().wait().get()

    for _ in range(3):
        reco_detail = context.run_recognition("星塔_节点_商店_识别强化所需金币_agent", image)
        if reco_detail and reco_detail.hit:
            logger.debug(f"识别到强化所需金币：{reco_detail.best_result.text}")
            return int(reco_detail.best_result.text)

        if reco_detail and reco_detail.all_results:
            logger.debug(
                f"识别强化所需金币结果：{[r.text for r in reco_detail.all_results]}"
            )
        else:
            logger.debug("未识别到任何关于强化金币的内容")

        if context.tasker.stopping:
            return 65535

        logger.debug("等待1秒后重试")
        time.sleep(1)

        image = context.tasker.controller.post_screencap().wait().get()

    logger.error("无法读取当前强化所需金币数量")
    return 65535


def calculate_max_enhance(
    current_coin: int,
    current_enhancement_cost: int,
    max_cost: int,
    initial_cost: int,
) -> tuple[int, int]:
    """可强化次数及总消耗金币数量。

    Args:
        current_coin: 当前金币数量。
        current_enhancement_cost: 当前强化所需金币数量。
        max_cost: 允许的单次强化金币上限。
        initial_cost: 除开免费次数，第一次强化消耗的费用。

    Returns:
        tuple[int, int]: 可强化次数，总共消耗的金币数量。
    """
    increment_step = [60, 60, 80, 80, 200, 200, 0]

    def _get_paid_step(current_cost: int) -> int:
        simulated_cost = initial_cost
        step = 0
        while simulated_cost < current_cost:
            i = increment_step[min(step, len(increment_step)-2)]
            simulated_cost += i
            step += 1
        return step

    paid_step = _get_paid_step(current_enhancement_cost)
    total_cost = 0
    free_count = 0
    pay_count = 0

    while (current_coin >= current_enhancement_cost
           and current_enhancement_cost <= max_cost):
        if current_enhancement_cost == 0:
            current_enhancement_cost = initial_cost
            free_count += 1
        else:
            current_coin -= current_enhancement_cost
            total_cost += current_enhancement_cost
            increment = increment_step[min(paid_step+pay_count, len(increment_step)-1)]
            current_enhancement_cost += increment
            pay_count += 1

    return pay_count + free_count - paid_step, total_cost


def check_shop_type(
    context: Context,
    image: Optional[numpy.ndarray] = None,
) -> str:
    """检查商店类型。

    Args:
        context: 任务上下文。
        image: 截图，为 None 时自动截图。

    Returns:
        str: 商店类型，中途商店为 regular，最终商店为 final，识别失败返回空字符串。
    """
    if image is None:
        image = context.tasker.controller.post_screencap().wait().get()

    for _ in range(3):
        reco_detail = context.run_recognition("星塔_节点_商店_离开商店_agent", image)
        if reco_detail and reco_detail.hit:
            return "regular"

        reco_detail = context.run_recognition("星塔_节点_商店_离开星塔_agent", image)
        if reco_detail and reco_detail.hit:
            return "final"

        if context.tasker.stopping:
            return ""

        time.sleep(1)
        image = context.tasker.controller.post_screencap().wait().get()

    return ""


@dataclass
class Parameters:
    """商店层参数类"""
    # 商店设置
    shop_type: str = "regular"
    lang_type: str = "cn"
    priority: list[str] = field(default_factory=lambda: ["drink", "melody"])
    drink_discount_threshold: float = 0.8
    melody_5_discount_threshold: float = 1.0
    melody_15_discount_threshold: float = 0.5
    buy_assist_melody: bool = False
    regular_shop_refresh_threshold: int = 1500
    full_price_buy_reserve_base: int = 500
    melody_of_aqua: bool = False
    melody_of_ignis: bool = False
    melody_of_terra: bool = False
    melody_of_ventus: bool = False
    melody_of_lux: bool = False
    melody_of_umbra: bool = False
    melody_of_focus: bool = False
    melody_of_skill: bool = False
    melody_of_ultimate: bool = False
    melody_of_pummel: bool = False
    melody_of_luck: bool = False
    melody_of_burst: bool = False
    melody_of_stamina: bool = False
    # 强化设置
    initial_cost: int = 60
    max_cost: int = 180
    # 动态参数
    current_coin: int = 0
    refresh_remaining: int = 0
    refresh_cost: int = 65535
    current_cost: int = 65535

    @classmethod
    def get_from_dict(cls, data: dict[str, Any]) -> Self:
        cls_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in cls_fields}
        return cls(**filtered_data)

    def update_from_dict(self, data: dict[str, Any]) -> None:
        cls_fields = {f.name for f in fields(self.__class__)}
        for k, v in data.items():
            if k in cls_fields:
                setattr(self, k, v)

    @property
    def target_melodies(self) -> list[str]:
        return [
            name for name, value in self.__dict__.items()
            if name.startswith("melody_of_") and value is True
        ]

    @property
    def dynamic_reserve(self) -> int:
        return self.full_price_buy_reserve_base * (self.refresh_remaining + 1)

    @property
    def min_buyable_price(self) -> int:
        """基于用户策略计算刷新后理论最低可购买商品价格。

        遍历 ITEM_STANDARD_PRICES，按 priority 和 threshold 筛选用户愿意购买的商品，
        取 标准价 × 对应 threshold 的最小值。

        Returns:
            int: 理论最低可购买价格；无符合条件商品时返回 65535。
        """
        prices = []
        if "drink" in self.priority:
            prices.append(
                ShopAction.ITEM_STANDARD_PRICES["potential_drink"] * self.drink_discount_threshold
            )
        if "melody" in self.priority and (self.target_melodies or self.buy_assist_melody):
            prices.append(
                ShopAction.ITEM_STANDARD_PRICES["melody_5"] * self.melody_5_discount_threshold
            )
            prices.append(
                ShopAction.ITEM_STANDARD_PRICES["melody_15"] * self.melody_15_discount_threshold
            )

        if prices:
            return int(min(prices))
        else:
            logger.error("无法计算理论最低可购买商品价格，本错误将导致无法执行刷新")
            return 65535

    @property
    def _enhancement_data(self) -> tuple[int, int]:
        if self.shop_type == "regular":
            max_cost = self.max_cost
        else:
            max_cost = 65535
        return calculate_max_enhance(self.current_coin, self.current_cost, max_cost, self.initial_cost)

    @property
    def enhancement_cost(self) -> int:
        _, coin = self._enhancement_data
        return coin

    @property
    def enhancement_count(self) -> int:
        count, _ = self._enhancement_data
        return count

    def should_refresh(self) -> bool:
        """判断当前是否满足刷新条件。

        regular 商店额外检查可支配金币是否达到 regular_shop_refresh_threshold；
        两种商店均需满足：刷新次数 > 0 且 可支配金币 ≥ 刷新费用 + min_buyable_price。

        Returns:
            bool: 满足刷新条件返回 True。
        """
        if self.refresh_remaining <= 0:
            logger.info("刷新次数已用完")
            return False

        usable = max(0, self.current_coin - self.enhancement_cost)
        min_threshold = self.refresh_cost + self.min_buyable_price
        if self.shop_type == "regular":
            threshold = max(min_threshold, self.regular_shop_refresh_threshold)
        elif self.shop_type == "final":
            threshold = min_threshold
        else:
            logger.error(f"未知商店类型 {self.shop_type}，本错误将导致无法执行刷新")
            threshold = 65535

        if usable >= threshold:
            logger.info(f"可用金币 {usable} 达到商店刷新标准 {threshold}，尝试刷新")
            return True

        logger.info(f"可用金币 {usable} 未达到刷新标准 {threshold}，跳过刷新")
        return False
    
   
@dataclass
class GridInfo:
    """
    店铺格子信息

    grid_num: 店铺格子编号，1-8。
    item_name: 商品名称，"potential_drink"等。
    item_quantity: 商品数量，1、5 或 15。
    item_price: 商品价格。
    display_name: 商品显示名称，"潜能特饮"等。
    bought: 是否已购买。
    checked: 是否已检查协奏音符。
    buy_type: 购买类型，"normal"、"assist_melody"、"dynamic_drink" 或"final_remainder"。
    buy_priority: 购买优先级，无限递增。
    """
    grid_num: int = 0
    item_name: str = ""
    item_quantity: int = 0
    item_price: int = 0
    display_name: str = ""
    bought: bool = False
    checked: bool = False
    buy_type: str = ""
    buy_priority: int = 0

    @property
    def item_roi(self) -> list[int]:
        """获取当前格子的道具ROI区域。

        Returns:
            list[int, int, int, int]: 道具ROI区域的坐标，(x, y, w, h)。
        """
        return ShopAction.GRID_ROIS[self.grid_num-1]["item_roi"]

    @property
    def price_roi(self) -> list[int]:
        """获取当前格子的道具价格ROI区域。

        Returns:
            list[int, int, int, int]: 道具价格ROI区域的坐标，(x, y, w, h)。
        """
        return ShopAction.GRID_ROIS[self.grid_num-1]["price_roi"]

    @property
    def name_roi(self) -> list[int]:
        """获取当前格子的道具名称ROI区域。

        Returns:
            list[int, int, int, int]: 道具名称ROI区域的坐标，(x, y, w, h)。
        """
        return ShopAction.GRID_ROIS[self.grid_num-1]["name_roi"]

    @property
    def discount(self) -> float:
        """获取物品的折扣比值（实际价格 / 标准价）。

        Returns:
            float: 折扣比值，值越低越划算；无法计算时返回 1.0。
        """
        if self.item_name == "potential_drink":
            std = ShopAction.ITEM_STANDARD_PRICES["potential_drink"]
            return self.item_price / std

        if "melody" in self.item_name and self.item_quantity == 5:
            std = ShopAction.ITEM_STANDARD_PRICES["melody_5"]
            return self.item_price / std

        if "melody" in self.item_name and self.item_quantity == 15:
            std = ShopAction.ITEM_STANDARD_PRICES["melody_15"]
            return self.item_price / std

        return 1.0

    def get_reserved_coin(self, params: Parameters) -> int:
        """根据当前格子的 buy_type，计算需要预留多少钱
        Args:
            params (Parameters): 用户策略参数。

        Returns:
            int: 预留金币数量。
        """
        if self.buy_type in ["normal", "assist_melody", "final_remainder"]:
            return params.enhancement_cost
        if self.buy_type == "dynamic_drink":
            return params.enhancement_cost + params.dynamic_reserve
        logger.error(f"未知购买类型: {self.buy_type}，无法计算预留金币")
        return 0

    def can_afford(self, params: Parameters) -> bool:
        """判定当前格子的钱是否足够购买
        Args:
            params (Parameters): 用户策略参数。

        Returns:
            bool: 如果当前格子的钱足够购买，返回 True；否则返回 False。
        """
        reserve = self.get_reserved_coin(params)
        return (params.current_coin - reserve) >= self.item_price

    def is_match_normal_buy_plan(self, item_type: str, params: Parameters) -> str:
        """判定当前格子是否符合正常购买方案条件

        Args:
            item_type (str): 商品类型，"drink" 或 "melody"。
            params (Parameters): 用户策略参数。

        Returns:
            str: 如果符合正常购买方案，返回 "normal" 或 "assist_melody"；否则返回空字符串。
        """
        if self.bought or not self.item_name:
            return ""

        if item_type == "drink":
            if self.item_name == "potential_drink" and self.discount <= params.drink_discount_threshold:
                return "normal"
            return ""

        if item_type == "melody" and "melody" in self.item_name:
            thresholds = {5: params.melody_5_discount_threshold, 15: params.melody_15_discount_threshold}
            discount_limit = thresholds.get(self.item_quantity)

            if discount_limit is None or self.discount > discount_limit:
                return ""

            if self.item_name in params.target_melodies:
                return "normal"

            if params.buy_assist_melody:
                return "assist_melody"

        return ""
        
        
class GridManager:
    priority_counter: int = 1

    def __init__(self, grids: list[GridInfo], context: Optional[Context] = None, params: Optional[Parameters] = None):
        self._grids = grids
        self.context = context
        self.params = params

    def __iter__(self):
        return iter(self._grids)

    def __getitem__(self, index):
        return self._grids[index]

    def __len__(self):
        return len(self._grids)

    def __setitem__(self, index, value):
        self._grids[index] = value

    def bind(self, context: Context, params: Parameters) -> Self:
        """将执行环境绑定到管家身上"""
        self.context = context
        self.params = params
        return self

    def normal_buy_plan(self) -> Self:
        """按 priority 顺序对格子打标，写入 buy_type 和 buy_priority。

        打标前按价格升序排列。
        命中一个格子 buy_priority 就全局递增 +1，保证优先级唯一。
        melody 类型内优先判断目标音符，不符合时再判断协奏音符。

        Returns:
            Self: 打标后的格子列表。
        """
        self._grids.sort(key=lambda g: g.item_price)

        target_grids = []
        for target_type in self.params.priority:
            for grid in self._grids:
                buy_type = grid.is_match_normal_buy_plan(target_type, self.params)
                if buy_type:
                    grid.buy_type = buy_type
                    grid.buy_priority = self.__class__.priority_counter
                    self.__class__.priority_counter += 1
                    target_grids.append(grid)
        return GridManager(target_grids, self.context, self.params)

    def high_price_drinks_buy_plan(self) -> Self:
        grids = sorted(
            [g for g in self._grids
             if not g.bought
             and g.item_name == "potential_drink"],
            key=lambda g: g.item_price,
        )
        for grid in grids:
            grid.buy_type = "dynamic_drink"
            grid.buy_priority = self.__class__.priority_counter
            self.__class__.priority_counter += 1
        return GridManager(grids, self.context, self.params)

    def remaining_drinks_buy_plan(self) -> Self:
        grids = sorted(
            [g for g in self._grids
             if not g.bought
             and g.item_name == "potential_drink"],
            key=lambda g: g.item_price,
        )
        for grid in grids:
            grid.buy_type = "normal"
            grid.buy_priority = self.__class__.priority_counter
            self.__class__.priority_counter += 1
        return GridManager(grids, self.context, self.params)

    def remainder_buy_plan(self) -> Self:
        grids = sorted(
            [g for g in self._grids if not g.bought],
            key=lambda g: g.item_price,
            reverse=True,
        )
        for grid in grids:
            grid.buy_type = "final_remainder"
            grid.buy_priority = self.__class__.priority_counter
            self.__class__.priority_counter += 1
        return GridManager(grids, self.context, self.params)

    def buy(self) -> bool:
        """对实例中的所有格子依次执行购买操作，购买成功成标记 bought 为 True。
        用户中止时立即返回 False；
        购买失败则输出日志并继续。

        Returns:
            bool: 未被用户中止时返回 True；用户中止时返回 False。
        """

        for grid in self._grids:
            if self.context.tasker.stopping:
                return False

            self.params.current_coin = get_current_coin(self.context)
            if not grid.can_afford(self.params):
                reserved_coin = grid.get_reserved_coin(self.params)
                logger.debug(f"当前金币 {self.params.current_coin}，预留给强化的金币 {reserved_coin}")
                logger.debug(f"需要金币 {grid.item_price}，购买类型 {grid.buy_type}")
                logger.debug(f"金币不足，跳过 {grid.item_name}")
                continue

            if grid.buy_type in ["normal", "dynamic_drink"]:
                success = self._buy_item(grid)
            elif grid.buy_type == "assist_melody":
                if grid.checked:
                    logger.debug(f"跳过已检查的协奏音符 {grid.display_name}")
                    continue
                success = self._buy_assist_melody(grid)
            elif grid.buy_type == "final_remainder":
                success = self._buy_item(grid)
            else:
                logger.error(f"未知的购买类型: {grid.buy_type}")
                continue

            if success:
                grid.bought = True
            else:
                logger.debug(f"购买失败，跳过第{grid.grid_num}个格子")

        return True

    def _buy_item(self, grid: GridInfo) -> bool:
        """执行普通商品购买（潜能特饮 / 音符）。

        将调用方传入的 reserve_coin 注入潜能选择节点，防止潜能选择把预留给强化的金币刷光。
        由 pipeline 按需取用，调用方无需判断商品类型。

        Args:
            grid: 单个格子信息对象。

        Returns:
            bool: 购买任务成功返回 True。
        """
        reserved_coin = grid.get_reserved_coin(self.params)
        override: dict = {
            "星塔_节点_商店_购物_购买道具_agent": {
                "action": {"param": {"target": grid.price_roi}}
            },
            "星塔_节点_选择潜能_agent": {
                "attach": {"reserve_coin": reserved_coin}
            },
        }
        result = self.context.run_task("星塔_节点_商店_购物_购买道具_agent", override)
        if result and result.status.succeeded:
            logger.debug(f"购买 {grid.item_name} 成功")
            return True
        else:
            logger.error(f"购买 {grid.item_name} 过程出现问题")
            return False

    def _buy_assist_melody(self, grid: GridInfo) -> bool:
        """执行协奏音符购买，走单独的协奏音符 pipeline。

        Args:
            grid: 单个格子信息对象。

        Returns:
            bool: 购买任务成功返回 True。
        """
        override: dict = {
            "星塔_节点_商店_购买协奏音符_agent": {
                "action": {"param": {"target": grid.price_roi}}
            },
        }
        run_result = self.context.run_task("星塔_节点_商店_购买协奏音符_agent", override)
        if not(run_result and run_result.status.succeeded):
            logger.error(f"点击协奏音符 {grid.item_name} 过程出现问题")
            return False

        image = self.context.tasker.controller.post_screencap().wait().get()
        reco_detail = self.context.run_recognition("星塔_节点_商店_购买协奏音符_核实协奏_agent", image)
        grid.checked = True
        if reco_detail and reco_detail.hit:
            run_result = self.context.run_task("星塔_节点_商店_购物_购买道具_确认购买_agent")
            if run_result and run_result.status.succeeded:
                logger.debug(f"购买 {grid.item_name} 成功")
                return True
            else:
                logger.error(f"购买 {grid.item_name} 过程出现问题")
                return False
        else:
            logger.debug("该音符不是协奏音符，关闭确认框")
            run_result = self.context.run_task("星塔_节点_商店_购买协奏音符_退出购买_agent")
            if run_result and run_result.status.succeeded:
                logger.debug(f"关闭购买协奏音符 {grid.item_name} 成功")
            else:
                logger.error(f"关闭购买协奏音符 {grid.item_name} 过程出现问题")
            return False


@AgentServer.custom_action("shop_action")
class ShopAction(CustomAction):

    GRID_ROIS = [
        {
            "item_roi": [625, 130, 150, 190],
            "price_roi": [645, 242, 110, 35],
            "name_roi": [645, 275, 110, 25],
        },
        {
            "item_roi": [775, 130, 150, 190],
            "price_roi": [795, 242, 110, 35],
            "name_roi": [795, 275, 110, 25],
        },
        {
            "item_roi": [925, 130, 150, 190],
            "price_roi": [945, 242, 110, 35],
            "name_roi": [945, 275, 110, 25],
        },
        {
            "item_roi": [1075, 130, 150, 190],
            "price_roi": [1095, 242, 110, 35],
            "name_roi": [1095, 275, 110, 25],
        },
        {
            "item_roi": [625, 330, 150, 190],
            "price_roi": [645, 440, 110, 35],
            "name_roi": [645, 475, 110, 25],
        },
        {
            "item_roi": [775, 330, 150, 190],
            "price_roi": [795, 440, 110, 35],
            "name_roi": [795, 475, 110, 25],
        },
        {
            "item_roi": [925, 330, 150, 190],
            "price_roi": [945, 440, 110, 35],
            "name_roi": [945, 475, 110, 25],
        },
        {
            "item_roi": [1075, 330, 150, 190],
            "price_roi": [1095, 440, 110, 35],
            "name_roi": [1095, 475, 110, 25],
        },
    ]

    ITEM_NAMES= {
        "potential_drink": {
            "cn": ["潜能特饮", "能特", "特饮"],
            "tw": ["潛能特飲", "能特"],
            "en": ["Potential Drink", "Drink"],
            "jp": ["素質メザメール","メザ", "メサ", "メール"]
        },
        "melody_of_aqua": {
            "cn": ["水之音"],
            "tw": ["水之音"],
            "en": ["Melody of Water"],
            "jp": ["水の音符"]
        },
        "melody_of_ignis": {
            "cn": ["火之音"],
            "tw": ["火之音"],
            "en": ["Melody of Ignis"],
            "jp": ["火の音符"]
        },
        "melody_of_terra": {
            "cn": ["地之音"],
            "tw": ["地之音"],
            "en": ["Melody of Terra"],
            "jp": ["地の音符"]
        },
        "melody_of_ventus": {
            "cn": ["风之音"],
            "tw": ["風之音"],
            "en": ["Melody of Ventus"],
            "jp": ["風の音符"]
        },
        "melody_of_lux": {
            "cn": ["光之音"],
            "tw": ["光之音"],
            "en": ["Melody of Lux"],
            "jp": ["光の音符"]
        },
        "melody_of_umbra": {
            "cn": ["暗之音"],
            "tw": ["暗之音"],
            "en": ["Melody of Umbra"],
            "jp": ["闇の音符"]
        },
        "melody_of_focus": {
            "cn": ["专注之音"],
            "tw": ["專注之音"],
            "en": ["Melody of Focus"],
            "jp": ["集中の音符"]
        },
        "melody_of_skill": {
            "cn": ["技巧之音"],
            "tw": ["技巧之音"],
            "en": ["Melody of Skill"],
            "jp": ["器用の音符"]
        },
        "melody_of_ultimate": {
            "cn": ["绝招之音"],
            "tw": ["絕招之音"],
            "en": ["Melody of Ultimate"],
            "jp": ["必殺の音符"]
        },
        "melody_of_pummel": {
            "cn": ["强攻之音"],
            "tw": ["強攻之音"],
            "en": ["Melody of Pummel"],
            "jp": ["強撃の音符"]
        },
        "melody_of_luck": {
            "cn": ["幸运之音"],
            "tw": ["幸運之音"],
            "en": ["Melody of Luck"],
            "jp": ["幸運の音符"]
        },
        "melody_of_burst": {
            "cn": ["暴发之音"],
            "tw": ["爆發之音"],
            "en": ["Melody of Burst"],
            "jp": ["爆発の音符"]
        },
        "melody_of_stamina": {
            "cn": ["体力之音"],
            "tw": ["體力之音"],
            "en": ["Melody of Stamina"],
            "jp": ["体力の音符"]
        }
    }

    ITEM_STANDARD_PRICES: dict[str, int] = {
        "potential_drink": 200,
        "melody_5": 90,
        "melody_15": 400,
    }

    DISCOUNT_TEXT = {
        "cn": ["优惠"],
        "tw": ["優惠"],
        "en": ["SALE"],
        "jp": ["割引"]
    }

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """商店楼层自动购买主流程。

        读取配置参数后按 priority 循环购买，每轮结束后执行第二轮溢购饮料，
        满足条件则刷新并重新购买；循环结束后 final 商店追加补买饮料和零头购买。

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 正常完成返回 True；用户中止返回 False。
        """
        params = self._get_params(context, argv.node_name)
        logger.debug(
            f"当前强化费用: {params.current_cost}, "
            f"最大当前强化费用: {params.max_cost}, 初始强化费用: {params.initial_cost}"
        )
        logger.debug(f"商店类型: {params.shop_type}")

        context.run_task("星塔_节点_商店_点击商店购物_agent")

        while True:
            image = context.tasker.controller.post_screencap().wait().get()
            params.refresh_remaining = self._get_refresh_remaining(context, image)
            params.refresh_cost = self._get_refresh_cost(context, image)
            grids = self._get_grids(context, params, image)

            grids.normal_buy_plan().buy()
            grids.high_price_drinks_buy_plan().buy()

            if context.tasker.stopping:
                return False

            params.current_coin = get_current_coin(context)
            if not params.should_refresh():
                break
            context.run_task("星塔_节点_商店_点击刷新_agent")

        if params.shop_type == "final":
            grids.remaining_drinks_buy_plan().buy()
            grids.remainder_buy_plan().buy()

        context.run_task("星塔_节点_商店_购物_返回商店层_agent")
        return True

    @staticmethod
    def _get_params(context: Context, node_name: str) -> Parameters:
        """从节点 attach 读取商店配置参数，缺失时返回安全默认值。

        reserve_coin 由 EnhanceAction 节点的 max_cost 和 initial_cost 计算得出。
        游戏中后续商店层的初始强化费用不一定从 0 起步，以 initial_cost 作为初始费用
        是合理的近似，且此处只取 total_cost，影响微乎其微。
        同时将 max_cost、initial_cost 存为实例变量供后续函数直接读取。

        Args:
            context: 任务上下文。
            node_name: 当前节点名称。

        Returns:
            ShopAction.Parameters: 包含所有商店配置参数
        """
        image = context.tasker.controller.post_screencap().wait().get()

        # 商店参数
        node_data = context.get_node_data(node_name)
        attach = node_data.get("attach", {})
        params = Parameters().get_from_dict(attach)
        params.shop_type = check_shop_type(context, image)

        # 强化参数
        enhance_node_data = context.get_node_data("星塔_节点_商店_强化_agent")
        enhance_attach = enhance_node_data.get("attach", {})
        params.update_from_dict(enhance_attach)

        params.current_cost = get_enhancement_cost(context, image)
        params.current_coin = get_current_coin(context, image)

        return params

    def _get_grids(self, context: Context, params: Parameters, image=None) -> GridManager:
        """识别购物界面 8 个格子的道具信息。

        每个格子识别名称、数量、价格，计算折扣比值后组装为 GridInfo，
        并初始化 bought、buy_type、buy_priority 字段。
        识别完成后按价格升序排列。

        Args:
            context: 任务上下文。
            params: 商店配置参数。

        Returns:
            GridManager: 格子信息列表，每个元素包含 grid_num、item_name、
                item_quantity、item_price等字段。
        """
        grids_info = []
        lang_type = params.lang_type

        for i, grid_roi in enumerate(self.GRID_ROIS):
            logger.debug(f"正在识别第 {i + 1} 个格子")
            item_name, item_quantity, item_price = self._get_single_grid_info(
                context, grid_roi["price_roi"], grid_roi["name_roi"], lang_type, image
            )
            if item_name and item_quantity and item_price:
                grids_info.append(GridInfo(
                    grid_num=i + 1,
                    item_name=item_name,
                    item_quantity=item_quantity,
                    item_price=item_price,
                    display_name=ShopAction.ITEM_NAMES.get(item_name, {}).get(params.lang_type, ["?"])[0]
                ))
            else:
                logger.error(
                    f"第 {i + 1} 个格子内容识别失败："
                    f"item_name={item_name}, item_quantity={item_quantity}, item_price={item_price}"
                )

        logger.debug(f"道具列表: {grids_info}")
        return GridManager(grids_info, context, params)

    def _get_single_grid_info(
        self,
        context: Context,
        price_roi: list[int],
        name_roi: list[int],
        lang_type: str,
        image=None,
    ) -> tuple[str, int, int]:
        """单个格子的名称、数量、价格识别，识别失败时重试。

        Args:
            context: 任务上下文。
            price_roi: 道具价格的识别框范围。
            name_roi: 道具名称的识别框范围。
            lang_type: 识别语言类型。
            image: 截图，默认 None。

        Returns:
            tuple[str, int, int]: (item_name, item_quantity, item_price)，
                识别失败时能返回多少返回多少，完全失败返回 (None, None, None)。
        """
        if image is None:
            image = context.tasker.controller.post_screencap().wait().get()

        results = self._grid_recognition(context, image, price_roi, "price")
        raw_item_price = [r.text for r in results]
        item_price = self._parse_item_price(raw_item_price)
        logger.debug(f"价格从 '{raw_item_price}' 解析为 '{item_price}'")

        results = self._grid_recognition(context, image, name_roi, "name")
        raw_item_name = "".join([r.text for r in results])
        item_name, item_quantity = self._parse_item_name(raw_item_name, lang_type)
        logger.debug(
            f"名称从 '{raw_item_name}' 解析为名称: '{item_name}'，数量: '{item_quantity}'"
        )

        if not (item_name and item_quantity and item_price):
            logger.warning("识别道具格子失败")

        return item_name, item_quantity, item_price

    @staticmethod
    def _grid_recognition(
        context: Context,
        image: numpy.ndarray,
        roi: list[int],
        content: str,
    ) -> list:
        """对指定 ROI 执行单次 OCR 识别，返回识别结果列表。

        Args:
            context: 任务上下文。
            image: 截图。
            roi: 识别框范围。
            content: 识别内容类型，"price" 或 "name"。

        Returns:
            list: filtered_results 列表；识别失败或类型未知时返回空列表。
        """
        if content == "price":
            node_name = "星塔_节点_商店_购物_识别物品价格_agent"
        elif content == "name":
            node_name = "星塔_节点_商店_购物_识别物品内容_agent"
        else:
            logger.error(f"未知的内容类型：{content}")
            return []

        reco_detail = context.run_recognition(node_name, image, {
            node_name: {"recognition": {"param": {"roi": roi}}}
        })
        if reco_detail and reco_detail.hit:
            logger.debug(
                f"识别到物品内容：{[r.text for r in reco_detail.filtered_results]}"
            )
            return reco_detail.filtered_results
        if reco_detail and reco_detail.all_results:
            logger.debug(
                f"识别到的格子内容：{[r.text for r in reco_detail.all_results]}"
            )
        else:
            logger.debug(f"格子 {roi} 未识别到任何内容")
        return []

    def _parse_item_name(self, item_name: str, lang_type: str) -> tuple[str, int]:
        """从 OCR 原始字符串中解析物品内部名称和数量。

        先按 "名称 x数量" 格式拆分，再将语言显示名映射为程序内部通用名称。
        潜能特饮数量统一视为 1。

        Args:
            item_name: OCR 识别到的原始物品名称字符串。
            lang_type: 识别语言类型。

        Returns:
            tuple[str, int]: (item_name, item_quantity)，
                item_name 为内部通用名，item_quantity 为数量（0 表示未识别）。
        """
        mapping = self._get_reverse_mapping(lang_type)
        match = re.match(r"(.*?)\s*[x×]\s*(\d+)$", item_name, re.IGNORECASE)
        item_quantity = 0

        if match:
            item_name = match.group(1).strip()
            if match.group(2):
                item_quantity = int(match.group(2).strip())

        if item_name not in self.ITEM_NAMES:
            for m in mapping:
                if m in item_name:
                    item_name = mapping[m]
                    break

        if item_name == "potential_drink":
            item_quantity = 1

        return item_name, item_quantity

    @staticmethod
    def _parse_item_price(item_price: int | list) -> int:
        """从 OCR 原始数据中提取物品实际价格。

        过滤掉错误识别为 0 的结果，清洗混入的打折价格（4位以上数字截取末段），
        最终取所有候选值中的最小值。

        Args:
            item_price: OCR 识别到的价格原始数据，int 或字符串列表。

        Returns:
            int: 解析后的物品价格；无有效结果时返回 0。
        """
        if isinstance(item_price, int):
            item_price = [str(item_price)]

        parsed_item_price = []

        for p in item_price:
            if p.isdecimal() and (int(p) in [0, 1, 11]):
                continue

            # 清洗混入的打折价格，例如 "09045"->45, "400200"->200
            if len(p) >= 4:
                while True:
                    match = re.search(r'^.*?0(?=[1-9])', p)
                    if match:
                        p = p[match.end():]
                    else:
                        break
                if p.isdecimal():
                    parsed_item_price.append(int(p))

            if p.isdecimal():
                parsed_item_price.append(int(p))

        return min(parsed_item_price, default=0)

    @staticmethod
    def _get_refresh_remaining(context: Context, image=None) -> int:
        """识别商店当前剩余刷新次数。

        Args:
            context: 任务上下文。

        Returns:
            int: 剩余刷新次数；识别失败时返回 0。
        """
        if image is None:
            image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("星塔_节点_商店_购物_识别可刷新次数_agent", image)
        if reco_detail and reco_detail.hit:
            logger.debug(f"识别到刷新次数：{reco_detail.best_result.text}")
            return int(reco_detail.best_result.text)

        logger.debug("刷新次数识别失败，将返回 0")
        if reco_detail and reco_detail.all_results:
            logger.debug(f"识别内容：{[r.text for r in reco_detail.all_results]}")
        else:
            logger.debug("未识别到任何内容")
        return 0

    @staticmethod
    def _get_refresh_cost(context: Context, image=None) -> int:
        """识别当前刷新费用。

        Args:
            context: 任务上下文。

        Returns:
            int: 刷新费用；识别失败时返回 65535 防止误刷新。
        """
        if image is None:
            image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("星塔_通用_识别刷新花费_agent", image)
        if reco_detail and reco_detail.hit:
            logger.debug(f"识别到刷新费用：{[r.text for r in reco_detail.filtered_results]}")
            return int(reco_detail.best_result.text)

        logger.error("无法识别刷新费用，返回 65535")
        return 65535

    def _get_reverse_mapping(self, lang_type: str) -> dict[str, str]:
        """根据语言类型生成显示名到内部名的反向查找表。

        Args:
            lang_type: 游戏语言类型（cn/tw/en/jp）。

        Returns:
            dict[str, str]: 显示名 → 内部通用名的映射字典。
        """
        reverse_map = {}
        for key, translations in self.ITEM_NAMES.items():
            for name in translations.get(lang_type, []):
                reverse_map[name] = key
        return reverse_map


@AgentServer.custom_action("enhance_action")
class EnhanceAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """商店楼层自动强化主流程。

        regular 商店按 max_cost 限制单次强化上限；
        final 商店强制将上限设为 65535，以花光剩余金币为目标。

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 始终返回 True。
        """
        params = self._get_params(context, argv.node_name)

        logger.debug(f"最大强化金币: {params.max_cost}，强化递增金额: {params.initial_cost}")
        logger.debug(f"当前金币: {params.current_coin}，当前强化所需金币: {params.current_cost}")
        logger.debug(f"可强化次数: {params.enhancement_count}")
        for _ in range(params.enhancement_count):
            context.run_task("星塔_节点_商店_点击强化_agent")
        return True

    @staticmethod
    def _get_params(context: Context, node_name: str) -> Parameters:
        """从节点 attach 读取强化配置参数，缺失时返回安全默认值。

        Args:
            context: 任务上下文。
            node_name: 当前节点名称。

        Returns:
            Parameters: 包含更新后的 max_cost, initial_cost, current_cost, current_coin, shop_type。
        """
        image = context.tasker.controller.post_screencap().wait().get()
        # 强化参数
        enhance_node_data = context.get_node_data(node_name)
        enhance_attach = enhance_node_data.get("attach", {})
        params = Parameters().get_from_dict(enhance_attach)

        params.current_cost = get_enhancement_cost(context, image)
        params.current_coin = get_current_coin(context, image)
        params.shop_type = check_shop_type(context, image)

        return params