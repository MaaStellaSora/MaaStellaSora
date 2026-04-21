import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_action")


def _get_current_coin(
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


def _get_enhancement_cost(
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
        logger.debug("等待1秒后重试")
        time.sleep(1)
        image = context.tasker.controller.post_screencap().wait().get()

        if context.tasker.stopping:
            return 65535

    logger.error("无法读取当前强化所需金币数量")
    return 65535


def _calculate_max_enhance(
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


def _check_shop_type(
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

    @dataclass
    class _Config:
        # 商店设置
        shop_type: str = "regular"
        lang_type: str = "cn"
        reserve_coin: int = 0
        priority: list[str] = field(default_factory=list)
        drink_discount_threshold: float = 0.8
        melody_5_discount_threshold: float = 1.0
        melody_15_discount_threshold: float = 0.5
        target_melodies: list[str] = field(default_factory=list)
        buy_assist_melody: bool = False
        regular_shop_refresh_threshold: int = 1500
        min_buyable_price: float = float("inf")
        full_price_buy_reserve_base: int = 500
        # 强化设置
        initial_cost: int = 60
        max_cost: int = 180


    def __init__(self) -> None:
        super().__init__()
        self.cfg = self._Config()
        self.dynamic_reserve: int = 0
        self.refresh_remaining: int = 0
        self.current_cost: int = 65535
        self.image = None # 仅用作道具与刷新次数的识别，不作可用金币识别

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
        self.cfg.lang_type = params["lang_type"]
        self.cfg.reserve_coin = params["reserve_coin"]
        self.cfg.priority = params["priority"]
        self.cfg.drink_discount_threshold = params["drink_discount_threshold"]
        self.cfg.melody_5_discount_threshold = params["melody_5_discount_threshold"]
        self.cfg.melody_15_discount_threshold = params["melody_15_discount_threshold"]
        self.cfg.target_melodies = params["target_melodies"]
        self.cfg.buy_assist_melody = params["buy_assist_melody"]
        self.cfg.regular_shop_refresh_threshold = params["regular_shop_refresh_threshold"]
        self.cfg.full_price_buy_reserve_base = params["full_price_buy_reserve_base"]

        self.cfg.shop_type = _check_shop_type(context)
        logger.debug(f"商店类型: {self.cfg.shop_type}, 预留金币: {self.cfg.reserve_coin}")

        context.run_task("星塔_节点_商店_点击商店购物_agent")
        self.cfg.min_buyable_price = self._calc_min_buyable_price()

        while True:
            self.image = context.tasker.controller.post_screencap().wait().get()
            self.refresh_remaining = self._get_refresh_remaining(context)
            grids_info = self._get_grids_info(context)

            self._mark_buy_plan(grids_info)
            if not self._execute_buy_plan(context, grids_info):
                return False

            self._mark_full_price_buy_plan(grids_info)
            if not self._execute_buy_plan(context, grids_info):
                return False

            if not self._should_refresh(context):
                break
            context.run_task("星塔_节点_商店_点击刷新_agent")

        if self.cfg.shop_type == "final":
            self._mark_remaining_drinks_buy_plan(grids_info)
            if not self._execute_buy_plan(context, grids_info):
                return False
            self._mark_remainder_buy_plan(grids_info)
            if not self._execute_buy_plan(context, grids_info):
                return False

        context.run_task("星塔_节点_商店_购物_返回商店层_agent")
        return True

    def _get_params(self, context: Context, node_name: str) -> dict:
        """从节点 attach 读取商店配置参数，缺失时返回安全默认值。

        reserve_coin 由 EnhanceAction 节点的 max_cost 和 initial_cost 计算得出。
        游戏中后续商店层的初始强化费用不一定从 0 起步，以 initial_cost 作为初始费用
        是合理的近似，且此处只取 total_cost，影响微乎其微。
        同时将 max_cost、initial_cost 存为实例变量供后续函数直接读取。

        Args:
            context: 任务上下文。
            node_name: 当前节点名称。

        Returns:
            dict: 包含所有商店配置参数，其中 reserve_coin 为计算所得，
                  target_melodies 为已聚合的目标音符列表。
        """
        defaults = {
            "lang_type": "cn",
            "priority": ["drink", "melody"],
            "drink_discount_threshold": 0.8,
            "melody_5_discount_threshold": 1.0,
            "melody_15_discount_threshold": 0.5,
            "regular_shop_refresh_threshold": 1500,
            "full_price_buy_reserve_base": 500,
            "buy_assist_melody": False,
            "melody_of_aqua": False,
            "melody_of_ignis": False,
            "melody_of_terra": False,
            "melody_of_ventus": False,
            "melody_of_lux": False,
            "melody_of_umbra": False,
            "melody_of_focus": False,
            "melody_of_skill": False,
            "melody_of_ultimate": False,
            "melody_of_pummel": False,
            "melody_of_luck": False,
            "melody_of_burst": False,
            "melody_of_stamina": False,
        }
        node_data = context.get_node_data(node_name)
        if not node_data:
            logger.error("无法获取商店设置，将使用默认参数")
            params = {**defaults, "reserve_coin": 0, "target_melodies": []}
            return params

        attach = node_data.get("attach", {})
        params = {key: attach.get(key, default) for key, default in defaults.items()}

        enhance_node_data = context.get_node_data("星塔_节点_商店_强化_agent")
        if not enhance_node_data:
            logger.error("无法读取强化设置，将使用默认参数")
            params["reserve_coin"] = 0
        else:
            enhance_attach = enhance_node_data.get("attach", {})
            self.max_cost = enhance_attach.get("max_cost", 180)
            self.initial_cost = enhance_attach.get("initial_cost", 60)
            self.current_cost = _get_enhancement_cost(context)
            current_coin = _get_current_coin(context)
            logger.debug(
                f"当前金币: {current_coin}, 当前强化费用: {self.current_cost}, "
                f"最大当前强化费用: {self.max_cost}, 初始强化费用: {self.initial_cost}"
            )
            _, params["reserve_coin"] = _calculate_max_enhance(
                current_coin, self.current_cost, self.max_cost, self.initial_cost
            )

        params["target_melodies"] = [
            m for m in params if m.startswith("melody_of_") and params[m]
        ]
        return params

    def _calc_min_buyable_price(self) -> int:
        """基于用户策略计算刷新后理论最低可购买商品价格。

        遍历 ITEM_STANDARD_PRICES，按 priority 和 threshold 筛选用户愿意购买的商品，
        取 标准价 × 对应 threshold 的最小值。

        Returns:
            int: 理论最低可购买价格；无符合条件商品时返回 65535。
        """
        prices = []
        if "drink" in self.cfg.priority:
            prices.append(
                self.ITEM_STANDARD_PRICES["potential_drink"] * self.cfg.drink_discount_threshold
            )
        if "melody" in self.cfg.priority and (self.cfg.target_melodies or self.cfg.buy_assist_melody):
            prices.append(
                self.ITEM_STANDARD_PRICES["melody_5"] * self.cfg.melody_5_discount_threshold
            )
            prices.append(
                self.ITEM_STANDARD_PRICES["melody_15"] * self.cfg.melody_15_discount_threshold
            )

        if prices:
            return int(min(prices))
        else:
            logger.error("无法计算理论最低可购买商品价格，本错误将导致无法执行刷新")
            return 65535

    def _get_grids_info(self, context: Context) -> list[dict]:
        """识别购物界面 8 个格子的道具信息。

        每个格子识别名称、数量、价格，计算折扣比值后组装为 dict，
        并初始化 bought、buy_type、buy_priority 字段。
        识别完成后按价格升序排列。

        Args:
            context: 任务上下文。

        Returns:
            list[dict]: 格子信息列表，每个元素包含 grid_num、item_name、
                item_quantity、item_price、discount、item_roi、price_roi、
                name_roi、bought、buy_type、buy_priority。
        """
        grids_info = []

        for i, grid_roi in enumerate(self.GRID_ROIS):
            logger.debug(f"正在识别第 {i + 1} 个格子")
            item_name, item_quantity, item_price = self._get_single_grid_info(
                context, grid_roi["price_roi"], grid_roi["name_roi"]
            )
            if item_name and item_quantity and item_price:
                grids_info.append({
                    "grid_num": i + 1,
                    "item_name": item_name,
                    "item_quantity": item_quantity,
                    "item_price": item_price,
                    "discount": self._get_discount(item_name, item_quantity, item_price),
                    "item_roi": grid_roi["item_roi"],
                    "price_roi": grid_roi["price_roi"],
                    "name_roi": grid_roi["name_roi"],
                    "bought": False,
                    "checked": False,
                    "buy_type": None,
                    "buy_priority": 0,
                })
            else:
                logger.error(
                    f"第 {i + 1} 个格子内容识别失败："
                    f"item_name={item_name}, item_quantity={item_quantity}, item_price={item_price}"
                )

        grids_info.sort(key=lambda x: x["item_price"])
        logger.debug(f"排序后道具列表: {grids_info}")
        return grids_info

    def _get_single_grid_info(
        self,
        context: Context,
        price_roi: list[int],
        name_roi: list[int],
        retry_max: int = 3,
    ) -> tuple[str, int, int]:
        """单个格子的名称、数量、价格识别，识别失败时重试。

        Args:
            context: 任务上下文。
            price_roi: 道具价格的识别框范围。
            name_roi: 道具名称的识别框范围。
            retry_max: 最大重试次数，默认为 3。

        Returns:
            tuple[str, int, int]: (item_name, item_quantity, item_price)，
                识别失败时能返回多少返回多少，完全失败返回 (None, None, None)。
        """
        item_price = []
        item_name = ""
        item_quantity = 0

        for count in range(retry_max):
            logger.debug(f"第 {count + 1} 次识别道具格子")
            image = self.image

            if not item_price:
                results = self._grid_recognition(context, image, price_roi, "price")
                raw_item_price = [r.text for r in results]
                item_price = self._parse_item_price(raw_item_price)
                logger.debug(f"价格从 '{raw_item_price}' 解析为 '{item_price}'")

            if not item_name or not item_quantity:
                results = self._grid_recognition(context, image, name_roi, "name")
                raw_item_name = "".join([r.text for r in results])
                item_name_temp, item_quantity_temp = self._parse_item_name(raw_item_name)
                logger.debug(
                    f"名称从 '{raw_item_name}' 解析为名称: '{item_name_temp}'，数量: '{item_quantity_temp}'"
                )
                if not item_name:
                    item_name = item_name_temp
                if not item_quantity:
                    item_quantity = item_quantity_temp

            if item_name and item_quantity and item_price:
                return item_name, item_quantity, item_price

            if context.tasker.stopping:
                return "", 0, 0

            logger.warning("识别道具格子失败，准备重试")
            time.sleep(1)

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

    def _parse_item_name(self, item_name: str) -> tuple[str, int]:
        """从 OCR 原始字符串中解析物品内部名称和数量。

        先按 "名称 x数量" 格式拆分，再将语言显示名映射为程序内部通用名称。
        潜能特饮数量统一视为 1。

        Args:
            item_name: OCR 识别到的原始物品名称字符串。

        Returns:
            tuple[str, int]: (item_name, item_quantity)，
                item_name 为内部通用名，item_quantity 为数量（0 表示未识别）。
        """
        mapping = self._get_reverse_mapping(self.cfg.lang_type)
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
    def _buy_item(context: Context, grid: dict, reserve_coin: int) -> bool:
        """执行普通商品购买（潜能特饮 / 音符）。

        将调用方传入的 reserve_coin 注入潜能选择节点，防止潜能选择把预留给强化的金币刷光。
        由 pipeline 按需取用，调用方无需判断商品类型。

        Args:
            context: 任务上下文。
            grid: 单个格子信息字典。
            reserve_coin: 注入潜能选择节点的预留金币数，由 _execute_single_purchase 按
                buy_type 决定传入值（normal/dynamic_drink 传 self.reserve_coin，
                final_remainder 传强化总消耗 total_cost）。

        Returns:
            bool: 购买任务成功返回 True。
        """
        override: dict = {
            "星塔_节点_商店_购物_购买道具_agent": {
                "action": {"param": {"target": grid["price_roi"]}}
            },
            "星塔_节点_选择潜能_agent": {
                "attach": {"reserve_coin": reserve_coin}
            },
        }
        result = context.run_task("星塔_节点_商店_购物_购买道具_agent", override)
        if result and result.status.succeeded:
            logger.debug(f"购买 {grid['item_name']} 成功")
            return True
        else:
            logger.error(f"购买 {grid['item_name']} 过程出现问题")
            return False

    @staticmethod
    def _buy_assist_melody(context: Context, grid: dict) -> bool:
        """执行协奏音符购买，走单独的协奏音符 pipeline。

        Args:
            context: 任务上下文。
            grid: 单个格子信息字典。

        Returns:
            bool: 购买任务成功返回 True。
        """
        override: dict = {
            "星塔_节点_商店_购买协奏音符_agent": {
                "action": {"param": {"target": grid["price_roi"]}}
            },
        }
        run_result = context.run_task("星塔_节点_商店_购买协奏音符_agent", override)
        if not(run_result and run_result.status.succeeded):
            logger.error(f"点击协奏音符 {grid['item_name']} 过程出现问题")
            return False

        image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("星塔_节点_商店_购买协奏音符_核实协奏_agent", image)
        grid["checked"] = True
        if reco_detail and reco_detail.hit:
            run_result = context.run_task("星塔_节点_商店_购物_购买道具_确认购买_agent")
            if run_result and run_result.status.succeeded:
                logger.debug(f"购买 {grid['item_name']} 成功")
                return True
            else:
                logger.error(f"购买 {grid['item_name']} 过程出现问题")
                return False
        else:
            logger.debug("该音符不是协奏音符，关闭确认框")
            run_result = context.run_task("星塔_节点_商店_购买协奏音符_退出购买_agent")
            if run_result and run_result.status.succeeded:
                logger.debug(f"关闭购买协奏音符 {grid['item_name']} 成功")
            else:
                logger.error(f"关闭购买协奏音符 {grid['item_name']} 过程出现问题")
            return False

    @staticmethod
    def _get_discount(
        item_name: str,
        item_quantity: int,
        item_price: int,
    ) -> float:
        """获取物品的折扣比值（实际价格 / 标准价）。

        Args:
            item_name: 物品内部名称。
            item_quantity: 物品数量。
            item_price: 物品实际价格。

        Returns:
            float: 折扣比值，值越低越划算；无法计算时返回 1.0。
        """
        if item_name == "potential_drink":
            std = ShopAction.ITEM_STANDARD_PRICES["potential_drink"]
            return item_price / std

        if "melody" in item_name and item_quantity == 5:
            std = ShopAction.ITEM_STANDARD_PRICES["melody_5"]
            return item_price / std

        if "melody" in item_name and item_quantity == 15:
            std = ShopAction.ITEM_STANDARD_PRICES["melody_15"]
            return item_price / std

        return 1.0

    def _get_refresh_remaining(self, context: Context) -> int:
        """识别商店当前剩余刷新次数。

        Args:
            context: 任务上下文。

        Returns:
            int: 剩余刷新次数；识别失败时返回 0。
        """
        image = self.image
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

    def _is_target_drink(self, grid: dict) -> bool:
        """判断格子是否为本轮应购买的潜能特饮。

        final 商店不限折扣全买；regular 商店按折扣阈值过滤。

        Args:
            grid: 单个格子信息字典。

        Returns:
            bool: 符合购买条件返回 True。
        """
        if grid["item_name"] != "potential_drink":
            return False
        return grid["discount"] <= self.cfg.drink_discount_threshold

    def _is_target_melody(self, grid: dict) -> bool:
        """判断格子是否为用户指定的目标音符。

        按 target_melodies 过滤商品名，再按数量对应的折扣阈值过滤。

        Args:
            grid: 单个格子信息字典。

        Returns:
            bool: 符合购买条件返回 True。
        """
        if grid["item_name"] not in self.cfg.target_melodies:
            return False
        if grid["item_quantity"] == 5:
            return grid["discount"] <= self.cfg.melody_5_discount_threshold
        if grid["item_quantity"] == 15:
            return grid["discount"] <= self.cfg.melody_15_discount_threshold
        return False

    def _is_target_assist_melody(self, grid: dict) -> bool:
        """判断格子是否为应购买的协奏音符。

        仅在 buy_assist_melody=True 时生效，目标为非用户指定音符中折扣满足阈值的音符。

        Args:
            grid: 单个格子信息字典。

        Returns:
            bool: 符合购买条件返回 True。
        """
        if not self.cfg.buy_assist_melody:
            return False
        if "melody" not in grid["item_name"]:
            return False
        if grid["item_name"] in self.cfg.target_melodies:
            return False
        if grid["item_quantity"] == 5:
            return grid["discount"] <= self.cfg.melody_5_discount_threshold
        if grid["item_quantity"] == 15:
            return grid["discount"] <= self.cfg.melody_15_discount_threshold
        return False

    def _mark_buy_plan(self, grids_info: list[dict]) -> None:
        """按 priority 顺序对格子打标，写入 buy_type 和 buy_priority。

        打标前按价格升序排列，掌握后续执行顺序（执行函数不做价格二次排序）。
        命中一个格子 buy_priority 就全局递增 +1，保证优先级唯一。
        melody 类型内优先判断目标音符，不符合时再判断协奏音符。

        Args:
            grids_info: _get_grids_info() 返回的格子列表。
        """
        grids_info.sort(key=lambda g: g["item_price"])
        for item_type in self.cfg.priority:
            for grid in grids_info:
                if grid["bought"]:
                    continue
                if item_type == "drink" and self._is_target_drink(grid):
                    grid["buy_type"] = "normal"
                    grid["buy_priority"] = max(g["buy_priority"] for g in grids_info) + 1
                elif item_type == "melody":
                    if self._is_target_melody(grid):
                        grid["buy_type"] = "normal"
                        grid["buy_priority"] = max(g["buy_priority"] for g in grids_info) + 1
                    elif self._is_target_assist_melody(grid):
                        grid["buy_type"] = "assist_melody"
                        grid["buy_priority"] = max(g["buy_priority"] for g in grids_info) + 1

    def _execute_buy_plan(self, context: Context, grids_info: list[dict]) -> bool:
        """过滤、排序买单并依次执行购买，统一标记 bought。

        过滤出 buy_priority 非 0 且 buy_type 非 None 且未购买的格子，
        按 buy_priority 升序执行（排序权由各打标函数掌握，此处不做二次排序）。
        购买成功标记 bought=True；用户中止时立即返回 False；购买失败则记录日志并继续。

        Args:
            context: 任务上下文。
            grids_info: 已完成打标的格子列表。

        Returns:
            bool: 未被用户中止时返回 True；用户中止时返回 False。
        """
        plan = sorted(
            [g for g in grids_info
             if g["buy_priority"] != 0 and g["buy_type"] is not None and not g["bought"]],
            key=lambda g: g["buy_priority"],
        )
        for grid in plan:
            current_coin = _get_current_coin(context)
            success = self._execute_single_purchase(context, grid, current_coin)
            if success:
                grid["bought"] = True
            elif context.tasker.stopping:
                return False
            else:
                logger.debug(f"购买失败，跳过第{grid['grid_num']}个格子")
        return True

    def _execute_single_purchase(
        self,
        context: Context,
        grid: dict,
        current_coin: int,
    ) -> bool:
        """按 buy_type 路由执行单格购买，返回是否成功。

        各路由类型在同一分支内完成 usable 计算、金币检查和购买调用，保持低耦合。

        Args:
            context: 任务上下文。
            grid: 单个格子信息字典，含 buy_type、item_price 等字段。
            current_coin: 调用前已取得的当前金币数。

        Returns:
            bool: 购买成功返回 True，失败或金币不足返回 False。
        """
        buy_type = grid["buy_type"]
        item_display = self.ITEM_NAMES.get(grid["item_name"], {}).get(self.cfg.lang_type, ["?"])[0]

        if buy_type == "normal":
            usable = max(0, current_coin - self.cfg.reserve_coin)
            if usable < grid["item_price"]:
                logger.debug(
                    f"可用金币 {usable} 不足，跳过 {item_display}（{grid['item_price']}）"
                )
                return False
            return self._buy_item(context, grid, self.cfg.reserve_coin)

        elif buy_type == "assist_melody":
            if grid["checked"]:
                logger.debug(f"跳过已检查的协奏音符 {item_display}")
                return False
            usable = max(0, current_coin - self.cfg.reserve_coin)
            if usable < grid["item_price"]:
                logger.debug(
                    f"可用金币 {usable} 不足，跳过 {item_display}（{grid['item_price']}）"
                )
                return False
            return self._buy_assist_melody(context, grid)

        elif buy_type == "dynamic_drink":
            usable = max(0, current_coin - self.cfg.reserve_coin - self.dynamic_reserve)
            if usable < grid["item_price"]:
                logger.debug(
                    f"可用金币 {usable} 不足，跳过 {item_display}（{grid['item_price']}）"
                )
                return False
            return self._buy_item(context, grid, self.cfg.reserve_coin)

        elif buy_type == "final_remainder":
            _, total_cost = _calculate_max_enhance(
                current_coin, self.current_cost, 65535, self.initial_cost
            )
            usable = max(0, current_coin - total_cost)
            if usable < grid["item_price"]:
                logger.debug(
                    f"可用金币 {usable} 不足，跳过 {item_display}（{grid['item_price']}）"
                )
                return False
            return self._buy_item(context, grid, total_cost)

        else:
            logger.error(f"未知的购买类型: {buy_type}")
            return False

    @staticmethod
    def _get_refresh_cost(context: Context) -> int:
        """识别当前刷新费用。

        Args:
            context: 任务上下文。

        Returns:
            int: 刷新费用；识别失败时返回 65535 防止误刷新。
        """
        image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("星塔_通用_识别刷新花费_agent", image)
        if reco_detail and reco_detail.hit:
            logger.debug(f"识别到刷新费用：{[r.text for r in reco_detail.filtered_results]}")
            return int(reco_detail.best_result.text)

        logger.error("无法识别刷新费用，返回 65535")
        return 65535

    def _should_refresh(self, context: Context) -> bool:
        """判断当前是否满足刷新条件。

        regular 商店额外检查可支配金币是否达到 regular_shop_refresh_threshold；
        两种商店均需满足：刷新次数 > 0 且 可支配金币 ≥ 刷新费用 + min_buyable_price。

        Args:
            context: 任务上下文。

        Returns:
            bool: 满足刷新条件返回 True。
        """
        usable = max(0, _get_current_coin(context) - self.cfg.reserve_coin)

        if self.cfg.shop_type == "regular" and usable < self.cfg.regular_shop_refresh_threshold:
            logger.info(f"可用金币 {usable} 未达到刷新标准 {self.cfg.regular_shop_refresh_threshold}，跳过刷新")
            return False

        if self.refresh_remaining <= 0:
            logger.info("刷新次数已用完")
            return False

        refresh_cost = self._get_refresh_cost(context)
        min_threshold = refresh_cost + self.cfg.min_buyable_price
        if usable >= min_threshold:
            if self.cfg.shop_type == "regular":
                logger.info(
                    f"可用金币 {usable} 达到商店刷新标准 {max(min_threshold, self.cfg.regular_shop_refresh_threshold)}，尝试刷新"
                )
            elif self.cfg.shop_type == "final":
                logger.info(
                    f"可用金币 {usable} 达到最终商店刷新标准 {min_threshold}，尝试刷新"
                )
            return True

        logger.info(f"可用金币 {usable} 不足以刷新")
        return False

    def _mark_full_price_buy_plan(self, grids_info: list[dict]) -> None:
        """第二轮购买打标：利用动态预留之外的溢出金币买入潜能特饮。

        识别剩余刷新次数，计算动态预留并存为实例变量 self.dynamic_reserve，
        对未购的潜能特饮格子按价格升序排列后逐格递增赋 buy_priority，
        写入 buy_type="dynamic_drink"。

        Args:
            grids_info: 当前轮次的格子列表，复用第一轮识别结果。
        """
        refresh_remaining = self.refresh_remaining
        self.dynamic_reserve = self.cfg.full_price_buy_reserve_base * (refresh_remaining + 1)
        logger.debug(
            f"剩余刷新次数: {refresh_remaining}, 动态预留: {self.dynamic_reserve}"
        )

        drinks = sorted(
            [g for g in grids_info
             if not g["bought"] and g["buy_type"] is None
             and g["item_name"] == "potential_drink"],
            key=lambda g: g["item_price"],
        )
        for grid in drinks:
            grid["buy_type"] = "dynamic_drink"
            grid["buy_priority"] = max(g["buy_priority"] for g in grids_info) + 1

    @staticmethod
    def _mark_remaining_drinks_buy_plan(grids_info: list[dict]) -> None:
        """离开前补买饮料打标（仅 final 商店）：对所有未购潜能特饮打标为 normal。

        按价格升序排列后逐格递增赋 buy_priority。self.reserve_coin 在整个
        ShopAction 过程中始终生效，此阶段不解放。

        Args:
            grids_info: 当前轮次最后一次识别的格子列表。
        """
        drinks = sorted(
            [g for g in grids_info
             if not g["bought"] and g["buy_type"] is None
             and g["item_name"] == "potential_drink"],
            key=lambda g: g["item_price"],
        )
        for grid in drinks:
            grid["buy_type"] = "normal"
            grid["buy_priority"] = max(g["buy_priority"] for g in grids_info) + 1

    @staticmethod
    def _mark_remainder_buy_plan(grids_info: list[dict]) -> None:
        """离开前零头购买打标（仅 final 商店）：对所有未购格子打标为 final_remainder。

        按价格从高到低排列后逐格递增赋 buy_priority（贪心策略，从高到低更易花光零头）。
        不限商品类型，usable 计算和 reserve_coin 传入均由执行函数按 final_remainder 路由处理。

        Args:
            grids_info: 当前轮次最后一次识别的格子列表。
        """
        candidates = sorted(
            [g for g in grids_info if not g["bought"]],
            key=lambda g: g["item_price"],
            reverse=True,
        )
        for grid in candidates:
            grid["buy_type"] = "final_remainder"
            grid["buy_priority"] = max(g["buy_priority"] for g in grids_info) + 1

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
        shop_type = _check_shop_type(context)
        current_coin = _get_current_coin(context)
        current_cost = _get_enhancement_cost(context)
        max_cost = params["max_cost"] if shop_type == "regular" else 65535
        count, _ = _calculate_max_enhance(
            current_coin, current_cost, max_cost, params["initial_cost"]
        )
        logger.debug(
            f"最大强化金币: {max_cost}，强化递增金额: {params['initial_cost']}"
        )
        logger.debug(
            f"当前金币: {current_coin}，当前强化所需金币: {current_cost}，可强化次数: {count}"
        )
        for _ in range(count):
            context.run_task("星塔_节点_商店_点击强化_agent")
        return True

    @staticmethod
    def _get_params(context: Context, node_name: str) -> dict:
        """从节点 attach 读取强化配置参数，缺失时返回安全默认值。

        Args:
            context: 任务上下文。
            node_name: 当前节点名称。

        Returns:
            dict: 包含 max_cost 和 initial_cost。
        """
        defaults = {"max_cost": 180, "initial_cost": 60}
        node_data = context.get_node_data(node_name)
        if not node_data:
            logger.error("无法读取强化设置，将使用默认参数")
            return defaults
        attach = node_data.get("attach", {})
        return {key: attach.get(key, default) for key, default in defaults.items()}