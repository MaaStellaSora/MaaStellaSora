import os
import re
import time
import json
from typing import Optional
from pathlib import Path

import numpy

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger


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
    _logger = logger.get_logger(__name__)

    if image is None:
        image = context.tasker.controller.post_screencap().wait().get()

    for _ in range(max_try):
        reco_detail = context.run_recognition("星塔_通用_识别当前金币_agent", image)
        if reco_detail and reco_detail.hit:
            _logger.debug(f"识别到当前金币：{[r.text for r in reco_detail.filtered_results]}")
            return int(reco_detail.filtered_results[-1].text)
        else:
            if reco_detail and reco_detail.all_results:
                _logger.debug(f"识别当前金币结果：{[r.text for r in reco_detail.all_results]}")
            else:
                _logger.debug("未识别到任何关于当前金币的内容")
            _logger.debug("等待1秒后重新识别")

        if context.tasker.stopping:
            return 0

        time.sleep(1)
        image = context.tasker.controller.post_screencap().wait().get()


    _logger.error("无法读取当前金币数量，将当作 0 金币处理")
    return 0


def _calculate_max_enhance(
    current_coin: int,
    current_enhancement_cost: int,
    max_cost: int,
    enhance_step: int,
) -> tuple[int, int]:
    """计算可强化次数及消耗的金币数量。

    Args:
        current_coin: 当前金币数量。
        current_enhancement_cost: 当前强化所需金币数量。
        max_cost: 允许的单次强化金币上限。
        enhance_step: 每次强化后费用递增的步长。

    Returns:
        tuple(int, int): 可强化次数, 总共消耗的金币数量。
    """
    count = 0
    total_cost = 0
    while (current_coin >= current_enhancement_cost
           and current_enhancement_cost <= max_cost):
        current_coin -= current_enhancement_cost
        total_cost += current_enhancement_cost
        count += 1
        current_enhancement_cost += enhance_step

    return count, total_cost


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

    for _ in range(3): # 最多尝试3次
        reco_detail = context.run_recognition("星塔_节点_商店_离开商店_agent", image)
        if reco_detail and reco_detail.hit:
            return "regular"

        reco_detail = context.run_recognition("星塔_节点_商店_离开星塔_agent", image)
        if reco_detail and reco_detail.hit:
            return "final"

        # 检查是否中断任务
        if context.tasker.stopping:
            return ""

        # 失败时，等待1秒后重试
        time.sleep(1)
        image = context.tasker.controller.post_screencap().wait().get()

    return ""


@AgentServer.custom_action("shop_action")
class ShopAction(CustomAction):

    GRID_ROIS = [
        {
            "item_roi": [625, 130, 150, 190],
            "price_roi": [645, 250, 110, 25],
            "name_roi": [645, 275, 110, 25],
        },
        {
            "item_roi": [775, 130, 150, 190],
            "price_roi": [795, 250, 110, 25],
            "name_roi": [795, 275, 110, 25],
        },
        {
            "item_roi": [925, 130, 150, 190],
            "price_roi": [945, 250, 110, 25],
            "name_roi": [945, 275, 110, 25],
        },
        {
            "item_roi": [1075, 130, 150, 190],
            "price_roi": [1095, 250, 110, 25],
            "name_roi": [1095, 275, 110, 25],
        },
        {
            "item_roi": [625, 330, 150, 190],
            "price_roi": [645, 450, 110, 25],
            "name_roi": [645, 475, 110, 25],
        },
        {
            "item_roi": [775, 330, 150, 190],
            "price_roi": [795, 450, 110, 25],
            "name_roi": [795, 475, 110, 25],
        },
        {
            "item_roi": [925, 330, 150, 190],
            "price_roi": [945, 450, 110, 25],
            "name_roi": [945, 475, 110, 25],
        },
        {
            "item_roi": [1075, 330, 150, 190],
            "price_roi": [1095, 450, 110, 25],
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

    def __init__(self):
        super().__init__()
        self.lang_type = None
        self.buy_secondary_skill_melody = False
        self.logger = logger.get_logger(__name__)

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """商店楼层自动购买主流程。

        读取 attach 参数，判断商店类型，循环执行常规购买；
        最终商店每轮会额外购买剩余的潜能特饮，然后尝试刷新道具并开始新一轮购买。

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 正常完成返回 True；购买异常中止返回 False。
        """
        params = self._get_params(context, argv.node_name)
        self.lang_type = params["lang_type"]
        self.buy_secondary_skill_melody = params["buy_secondary_skill_melody"]
        reserve_coin = params["reserve_coin"]
        priority = params["priority"]
        regular_shop_refresh_threshold = params["regular_shop_refresh_threshold"]
        drink_threshold = params["drink_discount_threshold"]
        melody_5_discount_threshold = params["melody_5_discount_threshold"]
        melody_15_discount_threshold = params["melody_15_discount_threshold"]
        target_melodies = [m for m in params if m.startswith("melody_of_") and params[m]]

        shop_type = _check_shop_type(context)
        self.logger.debug(f"当前商店类型: {shop_type}")
        self.logger.debug(f"预留金币: {reserve_coin}")

        context.run_task("星塔_节点_商店_点击商店购物_agent")
        min_price = self._calc_min_buyable_price(
            priority, drink_threshold, melody_5_discount_threshold, melody_15_discount_threshold, target_melodies
        )

        while True:
            grids_info = self._get_grids_info(context)

            if not self._execute_regular_buy(
                context, grids_info, priority,
                drink_threshold, melody_5_discount_threshold, melody_15_discount_threshold, target_melodies, reserve_coin
            ):
                return False

            current_coin = _get_current_coin(context)
            if shop_type == "regular" and (current_coin-reserve_coin) < regular_shop_refresh_threshold:
                break

            # TODO: 这里不对，要先买光潜能特饮，然后才能买音符，这里先买了潜能特饮跟音符然后才补买是不对的
            if shop_type == "final":
                if not self._execute_drink_supplement_buy(context, grids_info, reserve_coin):
                    return False

            if not self._should_refresh(context, reserve_coin, min_price):
                break

            context.run_task("星塔_通用_点击刷新_agent")

        context.run_task("星塔_节点_商店_购物_返回商店层_agent")
        return True

    def _get_params(self, context: Context, node_name: str) -> dict:
        """从节点 attach 读取商店配置参数，缺失时返回安全默认值。

        reserve_coin 由 EnhanceAction 节点的 max_cost 和 enhance_step 计算得出，
        假设强化费用从 0 开始累加，计算出强化阶段的最大总消耗。

        Args:
            context: 任务上下文。
            node_name: 当前节点名称。

        Returns:
            dict: 包含所有商店配置参数，其中 reserve_coin 为计算所得。
        """
        defaults = {
            "lang_type": "cn",
            "priority": ["drink", "melody"],
            "drink_discount_threshold": 1.0,
            "melody_5_discount_threshold": 1.0,
            "melody_15_discount_threshold": 1.0,
            "regular_shop_refresh_threshold": 1500,
            "buy_secondary_skill_melody": False,
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
            "melody_of_stamina": False
        }
        node_data = context.get_node_data(node_name)
        if not node_data:
            self.logger.error("无法获取商店设置，将使用默认参数")
            return {**defaults, "reserve_coin": 0}
        attach = node_data.get("attach", {})
        params = {key: attach.get(key, default) for key, default in defaults.items()}

        enhance_node_data = context.get_node_data("星塔_节点_商店_强化_agent")
        if not enhance_node_data:
            self.logger.error("无法读取强化设置，将使用默认参数")
            params["reserve_coin"] = 0
            return params

        enhance_attach = enhance_node_data.get("attach", {})
        max_cost = enhance_attach.get("max_cost", 180)
        enhance_step = enhance_attach.get("enhance_step", 60)
        current_coin = _get_current_coin(context)
        print(f"当前金币: {current_coin}, max_cost: {max_cost}, enhance_step: {enhance_step}")
        _, params["reserve_coin"] = _calculate_max_enhance(
            current_coin, enhance_step, max_cost, enhance_step
        )
        return params

    def _calc_min_buyable_price(
        self,
        priority: list[str],
        drink_threshold: float,
        melody_5_discount_threshold: float,
        melody_15_discount_threshold: float,
        target_melodies: list[str],
    ) -> float:
        """基于用户策略计算刷新后理论最低可购买商品价格。

        遍历 ITEM_STANDARD_PRICES，按 priority 和 threshold 筛选用户愿意
        购买的商品，取 标准价 × 对应 threshold 的最小值。

        Args:
            priority: 购买优先级列表。
            drink_threshold: 潜能特饮折扣比值上限。
            melody_5_discount_threshold: 5个音符折扣比值上限。
            melody_15_discount_threshold: 15个音符折扣比值上限。
            target_melodies: 用户指定的目标音符名称列表。

        Returns:
            float: 理论最低可购买价格；无符合条件商品时返回 float("inf")。
        """
        prices = []
        if "drink" in priority:
            prices.append(self.ITEM_STANDARD_PRICES["potential_drink"] * drink_threshold)
        if "melody" in priority and target_melodies:
            prices.append(self.ITEM_STANDARD_PRICES["melody_5"] * melody_5_discount_threshold)
            prices.append(self.ITEM_STANDARD_PRICES["melody_15"] * melody_15_discount_threshold)
        return min(prices) if prices else float("inf")

    def _get_grids_info(self, context):
        """
            获取8个格子的道具信息

            Args:
                context(Context): 上下文对象

            Returns:
                list: 每个格子的道具信息，每个元素为一个字典，包含
                    grid_num(int): 格子编号，从1开始
                    item_name(str): 道具名称
                    item_quantity(int): 道具数量
                    item_price(int): 道具价格
                    discount(float): 折扣信息，没有折扣为空值
        """
        # 初始化结果列表
        grids_info = []

        # 开始8个格子的循环
        for i in range(len(self.GRID_ROIS)):
            self.logger.debug(f"正在识别第{i+1}个格子")
            # 读取识别框范围
            grid = self.GRID_ROIS[i]
            item_roi = grid["item_roi"]
            price_roi = grid["price_roi"]
            name_roi = grid["name_roi"]

            # 获取单个格子的道具信息并添加到结果列表
            item_name, item_quantity, item_price = self._get_single_grid_info(context, price_roi, name_roi)
            if item_name and item_quantity and item_price:
                discount = self._get_discount(item_name, item_quantity, item_price)
                grids_info.append({
                    "grid_num": i+1,
                    "item_name": item_name,
                    "item_quantity": item_quantity,
                    "item_price": item_price,
                    "discount": discount,
                    "item_roi": item_roi,
                    "price_roi": price_roi,
                    "name_roi": name_roi,
                    "bought": False,
                })
            else:
                self.logger.error(f"第{i+1}个格子内容识别失败")
                self.logger.error(f"item_name: {item_name}, item_quantity: {item_quantity}, item_price: {item_price}")

        # 根据价格从低到高排序
        grids_info.sort(key=lambda x: x["item_price"])
        self.logger.debug(f"排序后道具列表: {grids_info}")

        return grids_info

    def _get_single_grid_info(
        self,
        context: Context,
        price_roi: list,
        name_roi: list,
        retry_max: int = 3,
    ) -> tuple:
        """格子内容识别函数。

        Args:
            context: 任务上下文。
            price_roi: 道具价格的识别框范围。
            name_roi: 道具名的识别框范围。
            retry_max: 最大重试次数，默认为 3。

        Returns:
            tuple: (item_name, item_quantity, item_price)，
                   识别失败时各字段为 "", 0, 0。
        """

        item_price = []
        item_name = ""
        item_quantity = 0
        # discount_flag = 0

        for count in range(retry_max):
            self.logger.debug(f"第{count+1}次识别道具格子")
            image = context.tasker.controller.post_screencap().wait().get()
            # 识别价格
            if not item_price:
                results = self._grid_recognition(context, image, price_roi, "price")
                raw_item_price = [r.text for r in results]
                item_price = self._parse_item_price(raw_item_price)
                self.logger.debug(f"将道具价格数据从'{raw_item_price}'解析为'{item_price}'")
            # 识别道具名字及数量
            if not item_name or not item_quantity:
                results = self._grid_recognition(context, image, name_roi, "name")
                raw_item_name = "".join([r.text for r in results])
                item_name_temp, item_quantity_temp = self._parse_item_name(raw_item_name)
                self.logger.debug(f"将道具名称数据从'{raw_item_name}'解析为道具名：'{item_name_temp}'与数量：'{item_quantity_temp}'")
                if not item_name:
                    item_name = item_name_temp
                if not item_quantity:
                    item_quantity = item_quantity_temp

            # 如果都识别到了，直接返回
            if item_name and item_quantity and item_price:
                return item_name, item_quantity, item_price

            # 检查是否中断任务
            if context.tasker.stopping:
                return None, None, None

            # 睡觉
            self.logger.warning(f"识别道具格子失败，准备重试")
            time.sleep(1)

        # 虽然未能识别完整，但能返回多少是多少
        return item_name, item_quantity, item_price

    def _grid_recognition(self, context, image, roi, content):
        """
            格子内容识别函数

            Args:
                context(Context): 上下文对象
                image(Image): 截图
                roi(list): 识别框范围
                content(str): 要识别的内容类型，"price"或"name"

            Returns:
                list: 可遍历的格子内容识别结果
        """
        if content == "price":
            node_name = "星塔_节点_商店_购物_识别物品价格_agent"
        elif content == "name":
            node_name = "星塔_节点_商店_购物_识别物品内容_agent"
        else:
            self.logger.error(f"未知的内容类型：{content}")
            return []

        reco_detail = context.run_recognition(node_name, image, {
            node_name: {
                "recognition": {
                    "param": {
                        "roi": roi
                    }
                }
            }
        })
        if reco_detail and reco_detail.hit:
            self.logger.debug(f"识别到物品内容：{[reco_detail.text for reco_detail in reco_detail.filtered_results]}")
            return reco_detail.filtered_results
        if reco_detail and reco_detail.all_results:
            self.logger.debug(f"识别到的格子内容：{[reco_detail.text for reco_detail in reco_detail.all_results]}")
        else:
            self.logger.debug(f"格子:{roi}未识别到任何内容")
        return []

    def _parse_item_name(self, item_name: str):
        """
            解析物品名称，提取物品名称和数量

            Args:
                item_name(str): 识别到的物品名称

            Returns:
                tuple: 解析后的物品名称和数量
        """
        # 生成反向查找表
        mapping = self._get_reverse_mapping(self.lang_type)

        # 获取物品数量
        match = re.match(r"(.*?)\s*[x×]\s*(\d+)$", item_name, re.IGNORECASE)
        item_quantity = 0

        # 分离物品名称和数量
        if match:
            item_name = match.group(1).strip()
            if match.group(2):
                item_quantity = int(match.group(2).strip())

        # 检查物品名称，尝试转换为程序内部通用名称，以适配不同服务端语言
        if not item_name in self.ITEM_NAMES:
            for m in mapping:
                if m in item_name:
                    item_name = mapping[m]
                    break

        # 针对潜能特饮，把数量改为1
        if item_name == "potential_drink":
            item_quantity = 1

        return item_name, item_quantity

    @staticmethod
    def _parse_item_price(item_price):
        """
            解析并提取物品价格

            Args:
                item_price(int | list): 识别到的物品价格

            Returns:
                int: 解析后的物品价格
        """
        if isinstance(item_price, int):
            item_price = [str(item_price)]

        parsed_item_price = []

        for p in item_price:
            # 排除为0的情况，往往是把硬币图标识别为0了
            if p.isdecimal() and int(p) == 0:
                continue

            # 清洗掉可能识别到的打折价格，往往是跟原价混在一起了
            # 一般价格为2~3位数，最便宜的音符*5是90块，最贵的音符*15是400块
            # 例子："09045" -> 45, "9072" -> 72, "400200" -> 200, "0400320" -> 320
            if len(p) >= 4:
                while True:
                    match = re.search(r'^.*?0(?=[1-9])', p)
                    if match:
                        p = p[match.end():]
                    else:
                        break
                if p.isdecimal():
                    parsed_item_price.append(int(p))

            # 其他情况，直接转换为int
            if p.isdecimal():
                parsed_item_price.append(int(p))

        # 对解析后的价格列表取最低值
        return min(parsed_item_price, default=0)

    @staticmethod
    def _buy_item(
        context: Context,
        roi: list,
        drink_reserve_coin: Optional[int] = None,
    ) -> bool:
        """使用 pipeline 执行购买操作。

        Args:
            context: 任务上下文。
            roi: 道具价格区域的点击范围。
            drink_reserve_coin: 购买潜能特饮时传入，注入选择潜能节点的
                                reserve_coin；购买其他道具时传 None。

        Returns:
            bool: 购买操作成功返回 True。
        """
        override: dict = {
            "星塔_节点_商店_购物_购买道具_agent": {
                "action": {"param": {"target": roi}}
            }
        }
        if drink_reserve_coin is not None:
            override["星塔_节点_选择潜能_agent"] = {
                "attach": {"reserve_coin": drink_reserve_coin}
            }
        run_result = context.run_task("星塔_节点_商店_购物_购买道具_agent", override)
        return bool(run_result and run_result.status.succeeded)

    @staticmethod
    def _buy_secondary_skill_melody(
        context: Context,
        roi: list,
    ) -> bool:
        """使用 pipeline 执行购买操作。

        Args:
            context: 任务上下文。
            roi: 道具价格区域的点击范围。

        Returns:
            bool: 购买操作成功返回 True。
        """
        override: dict = {
            "星塔_节点_商店_购买协奏音符_agent": {
                "action": {"param": {"target": roi}}
            }
        }
        run_result = context.run_task("星塔_节点_商店_购买协奏音符_agent", override)
        return bool(run_result and run_result.status.succeeded)

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

    def _get_refresh_remaining(self, context):
        """
            获取商店可刷新次数

            Args:
                context(Context): 上下文对象

            Returns:
                int: 商店可刷新次数
        """
        image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("星塔_节点_商店_购物_识别可刷新次数_agent", image)
        if reco_detail and reco_detail.hit:
            self.logger.debug(f"识别到刷新次数：{reco_detail.best_result.text}")
            return int(reco_detail.best_result.text)
        self.logger.debug(f"刷新次数识别失败，将返回0")
        if reco_detail and reco_detail.all_results:
            self.logger.debug(f"识别内容：{[result.text for result in reco_detail.all_results]}")
        else:
            self.logger.debug("未识别到任何内容")

        return 0

    def _execute_regular_buy(
        self,
        context: Context,
        grids_info: list[dict],
        priority: list[str],
        drink_threshold: float,
        melody_5_discount_threshold: float,
        melody_15_discount_threshold: float,
        target_melodies: list[str],
        reserve_coin: int,
    ) -> bool:
        """按用户策略遍历格子执行常规购买。

        按 priority 顺序分轮遍历 grids_info，各轮内按筛选条件决定是否购买。
        购买成功后将格子 bought 置 True。

        Args:
            context: 任务上下文。
            grids_info: _get_grids_info() 的返回值，格子按价格升序排列。
            priority: 购买优先级列表。
            drink_threshold: 潜能特饮折扣比值上限。
            melody_5_discount_threshold: 5个音符折扣比值上限。
            melody_15_discount_threshold: 15个音符折扣比值上限。
            target_melodies: 用户指定的目标音符名称列表。
            reserve_coin: 预留金币。

        Returns:
            bool: 所有购买操作正常完成返回 True；操作异常返回 False。
        """
        for item_type in priority:
            for grid in grids_info:
                if grid["bought"]:
                    continue
                target_type = self._is_target_grid(
                    grid, item_type, drink_threshold, melody_5_discount_threshold, melody_15_discount_threshold,
                    target_melodies
                )
                if not target_type:
                    continue
                if target_type == "Special":
                    self._buy_secondary_skill_melody(context, grid["price_roi"])
                    continue
                if not self._try_buy_grid(context, grid, reserve_coin):
                    return False
        return True

    @staticmethod
    def _is_target_grid(
        grid: dict,
        item_type: str,
        drink_threshold: float,
        melody_5_discount_threshold: float,
        melody_15_discount_threshold: float,
        target_melodies: list[str],
    ) -> bool:
        """判断格子是否符合当前轮次的购买条件。

        Args:
            grid: 单个格子信息字典。
            item_type: 当前处理的商品类型（"drink" 或 "melody"）。
            drink_threshold: 潜能特饮折扣比值上限。
            melody_5_discount_threshold: 5个音符折扣比值上限。
            melody_15_discount_threshold: 15个音符折扣比值上限。
            target_melodies: 用户指定的目标音符名称列表。

        Returns:
            bool: 符合条件返回 True。
        """
        if item_type == "drink":
            return (
                grid["item_name"] == "potential_drink"
                and grid["discount"] <= drink_threshold
            )
        if item_type == "melody" and grid["item_quantity"] == 5:
            if grid["item_name"] in target_melodies and grid["discount"] <= melody_5_discount_threshold:
                return True
            elif "melody" in grid["item_name"] and grid["item_name"] not in target_melodies and grid["discount"] <= melody_5_discount_threshold:
                return "Special"
        if item_type == "melody" and grid["item_quantity"] == 15:
            if grid["item_name"] in target_melodies and grid["discount"] <= melody_15_discount_threshold:
                return True
            elif "melody" in grid["item_name"] and grid["item_name"] not in target_melodies and grid["discount"] <= melody_15_discount_threshold:
                return "Special"
        return False

    def _try_buy_grid(
        self,
        context: Context,
        grid: dict,
        reserve_coin: int,
    ) -> bool:
        """检查可支配金币并尝试购买单个格子。

        购买成功后将格子 bought 置 True。

        Args:
            context: 任务上下文。
            grid: 单个格子信息字典。
            reserve_coin: 预留金币。

        Returns:
            bool: 用户没有终止任务时返回 True；用户终止任务时返回 False。
        """
        usable = max(0, _get_current_coin(context) - reserve_coin)
        if usable < grid["item_price"]:
            self.logger.info(
                f"可用金币 {usable} 不足，跳过 {self.ITEM_NAMES[grid['item_name']][self.lang_type][0]}（{grid['item_price']}）"
            )
            return True

        # TODO: 没有必要判断是不是潜能特饮，可以不管三七二十一把reserve_coin传给节点
        is_drink = grid["item_name"] == "potential_drink"
        drink_arg = reserve_coin if is_drink else None
        success = self._buy_item(context, grid["price_roi"], drink_arg)

        if success:
            grid["bought"] = True
            return True

        if not context.tasker.stopping:
            self.logger.error("购买失败，跳过该格子")
            return True
        return False

    def _get_refresh_cost(self, context: Context, max_try: int = 3) -> int:
        """识别当前刷新费用。

        Args:
            context: 任务上下文。
            max_try: 最大重试次数。

        Returns:
            int: 刷新费用；识别失败时返回 65535 防止误刷新。
        """
        for _ in range(max_try):
            image = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition("星塔_通用_识别刷新花费_agent", image)
            if reco_detail and reco_detail.hit:
                return int(reco_detail.best_result.text)
            if context.tasker.stopping:
                return 65535
            time.sleep(1)
        self.logger.error("无法识别刷新费用，返回 65535")
        return 65535

    def _execute_drink_supplement_buy(
        self,
        context: Context,
        grids_info: list[dict],
        reserve_coin: int,
    ) -> bool:
        """最终商店补买阶段：对所有未买的潜能特饮格子执行购买，不限折扣。

        Args:
            context: 任务上下文。
            grids_info: 本轮格子信息（格子按价格升序排列）。
            reserve_coin: 预留金币。

        Returns:
            bool: 操作正常完成返回 True；购买异常返回 False。
        """
        for grid in grids_info:
            if grid["bought"]:
                continue
            if grid["item_name"] != "potential_drink":
                continue
            if not self._try_buy_grid(context, grid, reserve_coin):
                return False
        return True

    def _should_refresh(
        self,
        context: Context,
        reserve_coin: int,
        min_buyable_price: float,
    ) -> bool:
        """判断当前是否满足刷新条件。

        刷新条件：剩余刷新次数 > 0 且 可支配金币 ≥ 刷新费用 + min_buyable_price。

        Args:
            context: 任务上下文。
            reserve_coin: 预留金币。
            min_buyable_price: 预计算的最低理论买单价格。

        Returns:
            bool: 满足刷新条件返回 True。
        """
        refresh_remaining = self._get_refresh_remaining(context)
        if refresh_remaining == 0:
            self.logger.info("刷新次数已用完")
            return False

        refresh_cost = self._get_refresh_cost(context)
        usable = max(0, _get_current_coin(context) - reserve_coin)
        if usable >= refresh_cost + min_buyable_price:
            self.logger.info(f"当前可用金币 {usable} 达到刷新费用标准 {refresh_cost+min_buyable_price}，尝试刷新")
            return True

        self.logger.info("可用金币不足以刷新")
        return False

    def _get_reverse_mapping(self, lang_type):
        """
            根据语言类型生成反向查找表
        """
        reverse_map = {}
        items_dict = self.ITEM_NAMES
        for key, translations in items_dict.items():
            # 获取对应语言的列表，如果不存在则返回空列表
            names = translations.get(lang_type, [])
            for name in names:
                reverse_map[name] = key
        return reverse_map

@AgentServer.custom_action("enhance_action")
class EnhanceAction(CustomAction):

    def __init__(self):
        super().__init__()
        self.logger = logger.get_logger(__name__)

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """商店楼层自动强化主流程。

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 始终返回 True。
        """
        params = self._get_params(context, argv.node_name)
        shop_type = _check_shop_type(context)
        current_coin = _get_current_coin(context)
        current_cost = self._get_enhancement_cost(context)
        if shop_type == "regular":
            max_cost = params["max_cost"]
        else:
            max_cost = 65535
        count, _ = _calculate_max_enhance(
            current_coin, current_cost, max_cost, params["cost_increment"]
        )
        self.logger.debug(f"最大强化金币:{max_cost}，强化递增金额：{params['cost_increment']}")
        self.logger.debug(f"当前金币:{current_coin}，当前强化所需金币:{current_cost}，当前可强化次数：{count}")
        for _ in range(count):
            context.run_task("星塔_节点_商店_点击强化_agent")
        return True

    def _get_params(self, context: Context, node_name: str) -> dict:
        """从节点 attach 读取强化配置参数，缺失时返回安全默认值。

        Args:
            context: 任务上下文。
            node_name: 当前节点名称。

        Returns:
            dict: 包含 max_cost 和 cost_increment。
        """
        defaults = {"max_cost": 180, "cost_increment": 60}
        node_data = context.get_node_data(node_name)
        if not node_data:
            self.logger.error("无法读取强化设置，将使用默认参数")
            return defaults
        attach = node_data.get("attach", {})
        return {key: attach.get(key, default) for key, default in defaults.items()}

    def _get_enhancement_cost(
        self,
        context: Context,
        image: Optional[numpy.ndarray] = None,
    ) -> int:
        """检查当前强化所需金币。

        Args:
            context: 任务上下文。
            image: 截图，为 None 时自动截图。

        Returns:
            int: 当前强化所需金币数量，免费强化返回 0，识别失败返回 65535。
        """
        if image is None:
            image = context.tasker.controller.post_screencap().wait().get()

        for _ in range(3):
            reco_detail = context.run_recognition("星塔_节点_商店_识别强化所需金币_agent", image)
            if reco_detail and reco_detail.hit:
                self.logger.debug(f"识别到强化所需金币：{reco_detail.best_result.text}")
                return int(reco_detail.best_result.text)

            if reco_detail and reco_detail.all_results:
                self.logger.debug(f"识别强化所需金币结果：{[r.text for r in reco_detail.all_results]}")
            else:
                self.logger.debug("未识别到任何关于强化金币的内容")
            self.logger.debug("等待1秒后重试")
            time.sleep(1)
            image = context.tasker.controller.post_screencap().wait().get()

            if context.tasker.stopping:
                return 65535

        self.logger.error("无法读取当前强化所需金币数量")
        return 65535


@AgentServer.custom_action("ascension_preparation")
class AscensionPreparation(CustomAction):

    def __init__(self):
        super().__init__()
        self.logger = logger.get_logger(__name__)

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """检查并导入预设文件，为爬塔流程做准备

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 成功时返回 True，失败时返回 False。
        """

        node_data = context.get_node_data(argv.node_name)
        preset_path = Path(os.path.abspath(__file__)).parent.parent / "presets"
        full_path = ""
        error_flag = False

        try:
            preset_name = node_data["attach"]["preset_name"]
            preset_element_flag = node_data["attach"]["preset_element"]
            preset_melodies_flag = node_data["attach"]["preset_melodies"]

            # 检查有无预设作业
            if not preset_name:
                self.logger.debug("没有提供预设作业，检查选项是否选择了预设")
                if preset_element_flag or preset_melodies_flag:
                    self.logger.error("没有预设作业，请把其他选项里的作业预设改为其他选项")
                    context.tasker.post_stop()
                    return False
                else:
                    self.logger.debug("选项检查通过")
                    return True

            # 读取预设作业
            full_path = (preset_path / preset_name).with_suffix(".json")
            with open(full_path, "r", encoding="utf-8") as f:
                presets = json.load(f)
                priority_list = presets.get("priority_list", [])
                preset_element = presets.get("element", "")
                preset_melodies = presets.get("melodies", [])

            # 检查预设作业是否有预设选项
            if preset_element_flag and not preset_element:
                self.logger.error("作业中没有预设属性选项，请手动选择属性")
                error_flag = True
            if preset_melodies_flag and not preset_melodies:
                self.logger.error("作业中没有预设音符，请手动选择商店购买的音符")
                error_flag = True
            if error_flag:
                return False

        except FileNotFoundError:
            self.logger.error(f"无法找到预设作业文件：{full_path}")
            self.logger.error(f"请核实预设作业名字是否正确，或预设作业是否存在等")
            return False
        except KeyError as e:
            self.logger.error(f"无法读取预设作业信息，错误信息：{e}")
            self.logger.error(f"请核实预设作业名字是否正确，或预设作业是否存在")
            context.tasker.post_stop()
            return False
        except json.decoder.JSONDecodeError as e:
            self.logger.error(f"无法解析作业文件，错误信息：{e}")
            self.logger.error("请核实json内容的格式是否正确")
            context.tasker.post_stop()
            return False

        context.override_pipeline({
            "星塔_节点_选择潜能_agent": {
                "attach": {
                    "priority_list": priority_list
                }
            }
        })

        if preset_element:
            match preset_element:
                case "aqua":
                    context.override_pipeline({
                        "星塔_属性塔选择": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_水风__384_271_129_39__334_221_229_139.png"
                                    ]
                                }
                            }
                        },
                        "星塔_节点_商店_购物_agent": {
                            "attach": {
                                "melody_of_aqua": True
                            }
                        }
                    })
                case "ignis":
                    context.override_pipeline({
                        "星塔_属性塔选择": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_火暗__381_404_129_39__331_354_229_139.png"
                                    ]
                                }
                            }
                        },
                        "星塔_节点_商店_购物_agent": {
                            "attach": {
                                "melody_of_ignis": True
                            }
                        }
                    })
                case "terra":
                    context.override_pipeline({
                        "星塔_属性塔选择": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_光土__387_137_124_45__337_87_224_145.png"
                                    ]
                                }
                            }
                        },
                        "星塔_节点_商店_购物_agent": {
                            "attach": {
                                "melody_of_terra": True
                            }
                        }
                    })
                case "ventus":
                    context.override_pipeline({
                        "星塔_属性塔选择": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_水风__384_271_129_39__334_221_229_139.png"
                                    ]
                                }
                            }
                        },
                        "星塔_节点_商店_购物_agent": {
                            "attach": {
                                "melody_of_ventus": True
                            }
                        }
                    })
                case "lux":
                    context.override_pipeline({
                        "星塔_属性塔选择": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_光土__387_137_124_45__337_87_224_145.png"
                                    ]
                                }
                            }
                        },
                        "星塔_节点_商店_购物_agent": {
                            "attach": {
                                "melody_of_lux": True
                            }
                        }
                    })
                case "umbra":
                    context.override_pipeline({
                        "星塔_属性塔选择": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_火暗__381_404_129_39__331_354_229_139.png"
                                    ]
                                }
                            }
                        },
                        "星塔_节点_商店_购物_agent": {
                            "attach": {
                                "melody_of_umbra": True
                            }
                        }
                    })
                case _:
                    self.logger.error(f"检测到未知属性：{preset_element}，请核实属性名是否符合文档要求")
                    context.tasker.post_stop()
                    return False

        if preset_melodies:
            node_data = context.get_node_data("星塔_节点_商店_购物_agent")
            shop_attachments = node_data.get("attach", {})
            for melody in preset_melodies:
                if melody in shop_attachments:
                    context.override_pipeline({
                        "星塔_属性塔选择_agent": {
                            "attach": {
                                melody: True
                            }
                        }
                    })
                else:
                    self.logger.error(f"未找到音符：{melody}，请核实音符名是否符合文档要求")
                    context.tasker.post_stop()
                    return False

        self.logger.info(f"已导入预设作业：{preset_name}")
        return True

@AgentServer.custom_action("ascension_loop")
class AscensionLoop(CustomAction):

    def __init__(self):
        super().__init__()
        self.logger = logger.get_logger(__name__)

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """检查剩余循环次数，决定是否退出爬塔流程

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 返回 True。
        """

        node_data = context.get_node_data(argv.node_name)
        if not node_data:
            node_data = {}
        attachment = node_data.get("attach", {})
        loop_count = attachment.get("loop_count", 1)
        loop_count -= 1
        if loop_count:
            self.logger.info(f"完成一次爬塔，剩余爬塔次数：{loop_count}")
            context.override_pipeline({
                argv.node_name: {
                    "attach": {
                        "loop_count": loop_count
                    }
                }
            })
        else:
            self.logger.info("爬塔已完成，回到主页")
            context.override_next(argv.node_name, ["星塔_回到主页_agent"])

        return True