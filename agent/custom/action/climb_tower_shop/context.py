from __future__ import annotations

from itertools import count, islice
from dataclasses import dataclass, field
from collections.abc import Iterator, ItemsView, ValuesView
from typing import Any, TYPE_CHECKING

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_shop_context")

ITEM_ROIS = [
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

if TYPE_CHECKING:
    class _MelodiesCompletion:
        melody_of_aqua: Melody
        melody_of_ignis: Melody
        melody_of_terra: Melody
        melody_of_ventus: Melody
        melody_of_lux: Melody
        melody_of_umbra: Melody
        melody_of_focus: Melody
        melody_of_skill: Melody
        melody_of_ultimate: Melody
        melody_of_pummel: Melody
        melody_of_luck: Melody
        melody_of_burst: Melody
        melody_of_stamina: Melody

ITEM_TRANSLATIONS = {
    "potential_drink": {
        "cn": ["潜能特饮", "能特", "特饮"],
        "tw": ["潛能特飲", "能特"],
        "en": ["Potential Drink", "Drink"],
        "jp": ["素質メザメール", "メザ", "メサ", "メール"]
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


class EnhancementCalculator:
    INCREMENT_STEPS = (60, 60, 80, 80, 200, 200, 0)

    @classmethod
    def _get_paid_step(cls, current_cost: int, initial_cost: int) -> int:
        """根据当前费用倒推处于第几个付费阶段。"""
        simulated_cost = initial_cost
        paid_step = 0
        max_derive_step = len(cls.INCREMENT_STEPS) - 2

        while simulated_cost < current_cost:
            simulated_cost += cls.INCREMENT_STEPS[min(paid_step, max_derive_step)]
            paid_step += 1

        return paid_step

    @classmethod
    def _enhancement_cost_stream(cls, current_cost: int, initial_cost: int) -> Iterator[int]:
        """生成单次强化费用流。"""
        # 1. 优先处理免费次数
        if current_cost == 0:
            yield 0
            current_cost = initial_cost

        # 2. 确定初始付费阶梯并持续产出费用
        start_step = cls._get_paid_step(current_cost, initial_cost)
        max_step_index = len(cls.INCREMENT_STEPS) - 1

        for step in count(start_step):
            yield current_cost
            current_cost += cls.INCREMENT_STEPS[min(step, max_step_index)]

    def max_enhance_count(
            self, current_coin: int, current_cost: int, max_cost: int, initial_cost: int
    ) -> int:
        """计算最大强化次数。"""
        cnt = 0
        # total_cost = 0

        for cost in self._enhancement_cost_stream(current_cost, initial_cost):
            if cost > max_cost or current_coin < cost:
                break
            current_coin -= cost
            # total_cost += cost
            cnt += 1

        return cnt

    def enhance_total_cost(
            self, count_to_enhance: int, current_cost: int, initial_cost: int
    ) -> int:
        """计算指定强化次数下的总消耗金币数。"""
        stream = self._enhancement_cost_stream(current_cost, initial_cost)
        return sum(islice(stream, count_to_enhance))


@dataclass
class Melody:
    """音符类"""
    display_name: str = ""
    count: int = -1
    required_count: int = -1

class Melodies(_MelodiesCompletion if TYPE_CHECKING else object):
    """音符集合类（基于内部动态字典存储，兼容 ShopParams 框架导入）"""

    def __init__(self, params: ShopParams) -> None:
        """根据 ITEM_TRANSLATIONS 自动装配所有音符，并从 ShopParams 动态读取策略属性。"""
        self._params = params
        self._melodies: dict[str, Melody] = {}

        lang = params.lang
        # 自动遍历配置中的所有音符 ID
        for melody_id, lang_dict in params.item_translations.items():
            if not melody_id.startswith("melody_of_"):
                continue

            # 获取语言显示名
            names = lang_dict.get(lang, [])
            display_name = names[0] if names else melody_id

            # 兼容固定 ShopParams 结构：通过 getattr 动态获取 params 上的同名属性
            # 即使以后框架在 ShopParams 上增加了新音符属性，这里也无需修改
            required_count = getattr(params, melody_id, -1)

            self._melodies[melody_id] = Melody(
                display_name=display_name,
                count=0,
                required_count=required_count
            )

    def update_from_count_dict(self, count_dict: dict[str, int]) -> None:
        """从扫描到的 {音符ID: 当前数量} 字典批量更新音符数量。"""
        for melody_id, cnt in count_dict.items():
            if melody_id in self._melodies:
                self._melodies[melody_id].count = cnt

    # ---- 属性访问代理：实现 melodies.melody_of_aqua ----
    def __getattr__(self, name: str) -> Melody:
        if "_melodies" in self.__dict__ and name in self._melodies:
            return self._melodies[name]
        raise AttributeError(f"'{type(self).__name__}' 对象没有属性 '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_data", "_melodies"):
            super().__setattr__(name, value)
        elif "_melodies" in self.__dict__ and name in self._melodies:
            if isinstance(value, Melody):
                self._melodies[name] = value
            else:
                raise TypeError(f"属性 {name} 的值必须是 Melody 类型对象")
        else:
            super().__setattr__(name, value)

    # ---- 常用容器与字典接口 ----
    def __iter__(self) -> Iterator[str]:
        """迭代器：仅产出需要购买的音符 ID"""
        for melody_id, melody in self._melodies.items():
            if melody.required_count > 0:
                yield melody_id

    def __getitem__(self, key: str) -> Melody:
        try:
            return self._melodies[key]
        except KeyError:
            raise KeyError(f"未找到音符: '{key}'") from None

    def __setitem__(self, key: str, value: Melody) -> None:
        if key not in self._melodies:
            raise KeyError(f"未找到音符: '{key}'")
        self._melodies[key] = value

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key in self._melodies:
            return self._melodies[key].required_count > 0
        return False

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def active_items(self) -> ItemsView[str, Melody]:
        return {
            k: v for k, v in self._melodies.items() if v.required_count > 0
        }.items()

    def active_values(self) -> ValuesView[Melody]:
        return {
            k: v for k, v in self._melodies.items() if v.required_count > 0
        }.values()

    def get(self, key: str, default: Any = None) -> Melody | Any:
        return self._melodies.get(key, default)


@dataclass(frozen=True)
class ShopParams:
    """商店层配置，从attach中导入，不可更改"""
    # 商店设置
    lang: str
    handler: str
    drink_discount_threshold: float
    melody_5_discount_threshold: float
    melody_15_discount_threshold: float
    buy_assist_melody: bool
    buy_assist_before_unlock: bool
    buy_melody_at_final_only: bool
    regular_shop_refresh_threshold: int
    full_price_buy_reserve_base: int
    max_enhancement_cost: int
    initial_enhancement_cost: int
    # 音符策略：每种音符期望购买到的目标数量（0 = 不购买该音符；>0 = 买到该数量为止）
    melody_of_aqua: int
    melody_of_ignis: int
    melody_of_terra: int
    melody_of_ventus: int
    melody_of_lux: int
    melody_of_umbra: int
    melody_of_focus: int
    melody_of_skill: int
    melody_of_ultimate: int
    melody_of_pummel: int
    melody_of_luck: int
    melody_of_burst: int
    melody_of_stamina: int
    # 常量
    item_rois = ITEM_ROIS
    item_standard_prices = ITEM_STANDARD_PRICES
    item_translations = ITEM_TRANSLATIONS
    discount_text = DISCOUNT_TEXT


@dataclass(slots=True)
class ShopContext:
    """商店层数据类，必须显式声明后才能使用，不允许直接实例化。"""
    params: ShopParams
    # 动态参数
    shop_type: str = "" # 商店类型， "regular" or "final"
    current_coin: int = -1 # 当前辉光币数量
    refresh_remaining: int = -1 # 刷新次数剩余数量
    refresh_cost: int = -1 # 刷新消耗的辉光币数量
    current_enhancement_cost: int = -1 # 当前强化消耗的辉光币数量
    melodies: Melodies = field(init=False) # 音符数量，如无设置音符策略则不会更新
    items: list[Item] = field(default_factory=list) # 当前商品格子信息
    enhance_error: int = 0 # 强化错误次数
    # 内部工具
    _item_reverse_maps: dict[str, dict[str, str]] = field(init=False, repr=False)
    _enhancement_calculator: EnhancementCalculator = EnhancementCalculator()

    def __post_init__(self):
        # 初始化后自动构建商品名称的反向查找索引，仅计算一次
        self._item_reverse_maps = {}
        for internal_name, lang_dict in self.params.item_translations.items():
            for lang, name_list in lang_dict.items():
                if lang not in self._item_reverse_maps:
                    self._item_reverse_maps[lang] = {}
                for name in name_list:
                    self._item_reverse_maps[lang][name] = internal_name

        # 初始化Melodies
        self.melodies = Melodies(self.params)

    def parse_item_name(self, text: str) -> str:
        """根据识别的商品名称匹配商品内部名。"""
        reverse_map = self._item_reverse_maps.get(self.params.lang, {})

        # 1. 优先精准匹配
        if text in reverse_map:
            return reverse_map[text]

        # 2. 子字符串模糊包含匹配（按名字长度降序排序，优先匹配长词）
        for name, internal_name in reverse_map.items():
            if name in text:
                return internal_name

        return ""

    @property
    def dynamic_reserve(self) -> int:
        """潜能特饮的动态溢购选项的保留量"""
        return self.params.full_price_buy_reserve_base * (self.refresh_remaining + 1)

    @property
    def min_buyable_price(self) -> int:
        """计算刷新后理论最低可购买商品价格，以保证刷新是有意义的。

        因为当前有绝对购买价值的就是潜能特饮，所以理论最低可购买价格就是潜能特饮的价格。
        为了保证刷新是有意义的，所以取潜能特饮的最高价格。

        Returns:
            int: 理论最低可购买价格；无符合条件商品时返回 65535。
        """
        prices = [ITEM_STANDARD_PRICES["potential_drink"]]

        if prices:
            return int(min(prices))
        else:
            logger.error("无法计算理论最低可购买商品价格，本错误将导致无法执行刷新")
            return 65535

    @property
    def refresh_threshold(self) -> int:
        if self.shop_type == "regular":
            return self.params.regular_shop_refresh_threshold
        elif self.shop_type == "final":
            return self.refresh_cost + self.min_buyable_price
        else:
            logger.error(f"未知商店类型 {self.shop_type}，如你没有修改过代码，请联系开发人员")
            return self.refresh_cost + self.min_buyable_price

    @property
    def should_refresh(self) -> bool:
        """判断当前是否满足刷新条件。"""
        if self.refresh_remaining <= 0:
            return False

        return self.current_coin >= (self.refresh_threshold + self.total_enhancement_cost)

    @property
    def total_enhancement_count(self) -> int:
        return self._enhancement_calculator.max_enhance_count(
            self.current_coin,
            self.current_enhancement_cost,
            self.params.max_enhancement_cost,
            self.params.initial_enhancement_cost
        )

    @property
    def uncapped_enhancement_count(self) -> int:
        return self._enhancement_calculator.max_enhance_count(
            self.current_coin,
            self.current_enhancement_cost,
            65535,
            self.params.initial_enhancement_cost
        )

    @property
    def total_enhancement_cost(self) -> int:
        cost = self._enhancement_calculator.enhance_total_cost(
            self.total_enhancement_count,
            self.current_enhancement_cost,
            self.params.initial_enhancement_cost
        )
        return cost

    @property
    def greedy_enhancement_cost(self) -> int:
        cost = self._enhancement_calculator.enhance_total_cost(
            self.uncapped_enhancement_count,
            self.current_enhancement_cost,
            self.params.initial_enhancement_cost
        )
        return cost


@dataclass(slots=True)
class Item:
    """
    商品格子信息

    grid_num: 商品格子编号，1-8。
    item_name: 商品名称，"potential_drink"等。
    item_quantity: 商品数量，1、5 或 15。
    item_price: 商品价格。
    display_name: 商品显示名称，"潜能特饮"等。
    bought: 是否已购买。
    trekker_specified: 当商品为潜能时，是否已指定 trekker。
    checked: 当商品为音符时，是否已检查协奏音符。
    buy_type: 购买类型，"normal"、"assist_melody"、"dynamic_drink" 或"final_remainder"。
    buy_priority: 购买优先级，供排序使用。
    """
    grid_num: int = 0
    internal_name: str = ""
    quantity: int = 0
    price: int = 0
    display_name: str = ""
    bought: bool = False
    trekker_specified: bool = False
    checked: bool = False
    buy_type: str = ""
    buy_priority: int = 0

    @property
    def item_roi(self) -> list[int]:
        """获取当前格子的道具ROI区域。

        Returns:
            list[int, int, int, int]: 道具ROI区域的坐标，(x, y, w, h)。
        """
        return ITEM_ROIS[self.grid_num - 1]["item_roi"]

    @property
    def price_roi(self) -> list[int]:
        """获取当前格子的道具价格ROI区域。

        Returns:
            list[int, int, int, int]: 道具价格ROI区域的坐标，(x, y, w, h)。
        """
        return ITEM_ROIS[self.grid_num - 1]["price_roi"]

    @property
    def name_roi(self) -> list[int]:
        """获取当前格子的道具名称ROI区域。

        Returns:
            list[int, int, int, int]: 道具名称ROI区域的坐标，(x, y, w, h)。
        """
        return ITEM_ROIS[self.grid_num - 1]["name_roi"]

    @property
    def discount(self) -> float:
        """获取物品的折扣比值（实际价格 / 标准价）。

        Returns:
            float: 折扣比值，值越低越划算；无法计算时返回 1.0。
        """
        if self.internal_name == "potential_drink":
            std = ITEM_STANDARD_PRICES["potential_drink"]
            return self.price / std

        if "melody" in self.internal_name and self.quantity == 5:
            std = ITEM_STANDARD_PRICES["melody_5"]
            return self.price / std

        if "melody" in self.internal_name and self.quantity == 15:
            std = ITEM_STANDARD_PRICES["melody_15"]
            return self.price / std

        return 1.0

