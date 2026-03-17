import re
import time
from typing import Optional

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
        _logger.debug(f"识别当前金币结果：{[r.text for r in reco_detail.all_results]}")
        if reco_detail and reco_detail.hit:
            return int(reco_detail.best_result.text)
        time.sleep(1)
        image = context.tasker.controller.post_screencap().wait().get()
        if context.tasker.stopping:
            return 0

    _logger.error("无法读取当前金币数量，将当作 0 金币处理")
    return 0


def _calculate_max_enhance_count(
    current_coin: int,
    current_enhancement_cost: int,
    max_cost: int,
    enhance_step: int,
) -> int:
    """计算可强化次数。

    Args:
        current_coin: 当前金币数量。
        current_enhancement_cost: 当前强化所需金币数量。
        max_cost: 允许的单次强化金币上限。
        enhance_step: 每次强化后费用递增的步长。

    Returns:
        int: 可强化次数。
    """
    count = 0
    while (current_coin >= current_enhancement_cost
           and current_enhancement_cost <= max_cost):
        current_coin -= current_enhancement_cost
        count += 1
        current_enhancement_cost += enhance_step
    return count


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
            "cn": ["潜能特饮"],
            "tw": ["潛能特飲"],
            "en": ["Potential Drink"],
            "jp": ["素質メザメール", "素質"]
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
            "cn": ["土之音"],
            "tw": ["土之音"],
            "en": ["Melody of Terra"],
            "jp": ["土の音符"]
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
        reserve_coin = params["reserve_coin"]
        priority = params["priority"]
        drink_threshold = params["drink_discount_threshold"]
        melody_threshold = params["melody_discount_threshold"]
        target_melodies = params["element_melodies"] + params["stat_melodies"]

        shop_type = self._check_shop_type(context)
        context.run_task("星塔_节点_商店_点击商店购物_agent")
        min_price = self._calc_min_buyable_price(
            priority, drink_threshold, melody_threshold, target_melodies
        )

        while True:
            grids_info = self._get_grids_info(context)

            if not self._execute_regular_buy(
                context, grids_info, priority,
                drink_threshold, melody_threshold, target_melodies, reserve_coin
            ):
                return False

            if shop_type == "regular":
                break

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
            "melody_discount_threshold": 1.0,
            "element_melodies": [],
            "stat_melodies": [],
        }
        node_data = context.get_node_data(node_name)
        if not node_data:
            self.logger.warning("get_node_data 返回 None，使用默认参数")
            return {**defaults, "reserve_coin": 0}
        attach = node_data.get("attach", {})
        params = {key: attach.get(key, default) for key, default in defaults.items()}

        enhance_node_data = context.get_node_data("星塔_节点_商店_强化_agent")
        if not enhance_node_data:
            self.logger.warning("无法读取强化节点数据，reserve_coin 将设为 0")
            params["reserve_coin"] = 0
            return params

        enhance_attach = enhance_node_data.get("attach", {})
        max_cost = enhance_attach.get("max_cost", 180)
        enhance_step = enhance_attach.get("enhance_step", 60)
        params["reserve_coin"] = _calculate_max_enhance_count(
            0, 0, max_cost, enhance_step
        )
        return params

    def _calc_min_buyable_price(
        self,
        priority: list[str],
        drink_threshold: float,
        melody_threshold: float,
        target_melodies: list[str],
    ) -> float:
        """基于用户策略计算刷新后理论最低可购买商品价格。

        遍历 ITEM_STANDARD_PRICES，按 priority 和 threshold 筛选用户愿意
        购买的商品，取 标准价 × 对应 threshold 的最小值。

        Args:
            priority: 购买优先级列表。
            drink_threshold: 潜能特饮折扣比值上限。
            melody_threshold: 音符折扣比值上限。
            target_melodies: 用户指定的目标音符名称列表。

        Returns:
            float: 理论最低可购买价格；无符合条件商品时返回 float("inf")。
        """
        prices = []
        if "drink" in priority:
            prices.append(self.ITEM_STANDARD_PRICES["potential_drink"] * drink_threshold)
        if "melody" in priority and target_melodies:
            prices.append(self.ITEM_STANDARD_PRICES["melody_5"] * melody_threshold)
            prices.append(self.ITEM_STANDARD_PRICES["melody_15"] * melody_threshold)
        return min(prices) if prices else float("inf")

    @staticmethod
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

            # 失败时，等待1秒后重试
            time.sleep(1)
            image = context.tasker.controller.post_screencap().wait().get()

            # 检查是否中断任务
            if context.tasker.stopping:
                return ""

        return ""

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
            # 暂时不需要用discount_flag，所以用_接收
            # TODO(samipale)： 潜能特饮增加对主控专用的检测
            item_name, item_quantity, item_price, _ = self._get_single_grid_info(context, item_roi, price_roi, name_roi)
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
        # TODO(samipale): 实现主控专属检测后，对潜能特饮按主控专属优先、价格升序的复合排序
        grids_info.sort(key=lambda x: int(x["item_price"]))

        return grids_info

    def _get_single_grid_info(
        self,
        context: Context,
        item_roi: list,
        price_roi: list,
        name_roi: list,
        retry_max: int = 3,
    ) -> tuple:
        """格子内容识别函数。

        先对整个 item_roi 区域做 OCR（方法一），若识别不完整则逐区域重试（方法二）。

        Args:
            context: 任务上下文。
            item_roi: 整个道具格子的识别框范围。
            price_roi: 道具价格的识别框范围。
            name_roi: 道具名的识别框范围。
            retry_max: 最大重试次数，默认为 3。

        Returns:
            tuple: (item_name, item_quantity, item_price, discount_flag)，
                   识别失败时各字段为 None。
        """

        # 方法一，通过整个道具格子识别
        raw_item_price = []
        raw_item_name = ""
        discount_flag = 0

        image = context.tasker.controller.post_screencap().wait().get()
        results = self._grid_recognition(context, image, item_roi)
        for r in results:
            if price_roi[0] <= r.box[0] <= price_roi[2] and price_roi[1] <= r.box[1] <= price_roi[3]:
                raw_item_price.append(r.text)
            if name_roi[0] <= r.box[0] <= name_roi[2] and name_roi[1] <= r.box[1] <= name_roi[3]:
                raw_item_name += r.text
            if r.text in self.DISCOUNT_TEXT[self.lang_type]:
                discount_flag = 1

        # 检查并清洗识别结果
        item_name, item_quantity = self._parse_item_name(raw_item_name)
        item_price = self._parse_item_price(raw_item_price)
        self.logger.debug(f"将道具名称数据从'{raw_item_name}'解析为道具名：'{item_name}'与数量：'{item_quantity}'")
        self.logger.debug(f"将道具价格数据从'{raw_item_price}'解析为'{item_price}'")

        # 如果都识别到了，直接返回
        if item_name and item_quantity and item_price:
            return item_name, item_quantity, item_price, discount_flag

        # 未能检测到的内容，使用方法二，识别每个细分区域
        for count in range(retry_max):
            self.logger.debug(f"使用方法二识别，第{count+1}次重试识别道具格子")
            image = context.tasker.controller.post_screencap().wait().get()
            # 识别价格
            if not item_price:
                results = self._grid_recognition(context, image, price_roi)
                raw_item_price = [r.text for r in results]
                item_price = self._parse_item_price(raw_item_price)
                self.logger.debug(f"将道具价格数据从'{raw_item_price}'解析为'{item_price}'")
            # 识别道具名字及数量
            if not item_name or not item_quantity:
                results = self._grid_recognition(context, image, name_roi)
                raw_item_name = "".join([r.text for r in results])
                item_name_temp, item_quantity_temp = self._parse_item_name(raw_item_name)
                self.logger.debug(f"将道具名称数据从'{raw_item_name}'解析为道具名：'{item_name_temp}'与数量：'{item_quantity_temp}'")
                if not item_name:
                    item_name = item_name_temp
                if not item_quantity:
                    item_quantity = item_quantity_temp

            # 如果都识别到了，直接返回
            if item_name and item_quantity and item_price:
                return item_name, item_quantity, item_price, discount_flag

            # 检查是否中断任务
            if context.tasker.stopping:
                return None, None, None, None

            # 睡觉
            self.logger.debug(f"第{count+1}次重试识别道具格子失败，等待1秒后重试")
            time.sleep(1)

        # 两个方法都没办法识别完整，能返回多少是多少
        return item_name, item_quantity, item_price, discount_flag

    def _grid_recognition(self, context, image, roi, expected=r".+"):
        """
            格子内容识别函数

            Args:
                context(Context): 上下文对象
                image(Image): 截图
                roi(list): 识别框范围
                expected(str): 预期的正则表达式，默认为".+"

            Returns:
                list: 可遍历的格子内容识别结果
        """
        reco_detail = context.run_recognition("星塔_节点_商店_购物_识别物品内容_agent", image, {
            "星塔_节点_商店_购物_识别物品内容_agent": {
                "recognition": {
                    "param": {
                        "expected": expected,
                        "roi": roi
                    }
                }
            }
        })
        self.logger.debug(f"识别到的格子内容：{[reco_detail.text for reco_detail in reco_detail.all_results]}")
        if reco_detail and reco_detail.hit:
            return reco_detail.filtered_results
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
            item_name = mapping.get(item_name, "")

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

    def _get_refresh_remaining(self, context, max_count=3):
        """
            获取商店可刷新次数

            Args:
                context(Context): 上下文对象

            Returns:
                int: 商店可刷新次数
        """
        for count in range(max_count):
            image = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition("星塔_节点_商店_购物_识别可刷新次数_agent", image)
            if reco_detail and reco_detail.hit:
                self.logger.debug(f"识别到刷新次数：{reco_detail.best_result.text}")
                return int(reco_detail.best_result.text)
            self.logger.debug(f"第{count+1}次刷新次数识别失败，等待1秒后重试")
            self.logger.debug(f"识别内容：{[result.text for result in reco_detail.all_results]}")
            time.sleep(1)
        return 0

    def _execute_regular_buy(
        self,
        context: Context,
        grids_info: list[dict],
        priority: list[str],
        drink_threshold: float,
        melody_threshold: float,
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
            melody_threshold: 音符折扣比值上限。
            target_melodies: 用户指定的目标音符名称列表。
            reserve_coin: 预留金币。

        Returns:
            bool: 所有购买操作正常完成返回 True；操作异常返回 False。
        """
        for item_type in priority:
            for grid in grids_info:
                if grid["bought"]:
                    continue
                if not self._is_target_grid(
                    grid, item_type, drink_threshold, melody_threshold, target_melodies
                ):
                    continue
                if not self._try_buy_grid(context, grid, reserve_coin):
                    return False
        return True

    @staticmethod
    def _is_target_grid(
        grid: dict,
        item_type: str,
        drink_threshold: float,
        melody_threshold: float,
        target_melodies: list[str],
    ) -> bool:
        """判断格子是否符合当前轮次的购买条件。

        Args:
            grid: 单个格子信息字典。
            item_type: 当前处理的商品类型（"drink" 或 "melody"）。
            drink_threshold: 潜能特饮折扣比值上限。
            melody_threshold: 音符折扣比值上限。
            target_melodies: 用户指定的目标音符名称列表。

        Returns:
            bool: 符合条件返回 True。
        """
        if item_type == "drink":
            return (
                grid["item_name"] == "potential_drink"
                and grid["discount"] <= drink_threshold
            )
        if item_type == "melody":
            return (
                grid["item_name"] in target_melodies
                and grid["discount"] <= melody_threshold
            )
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
            bool: 操作正常完成返回 True；购买异常返回 False。
        """
        usable = max(0, _get_current_coin(context) - reserve_coin)
        if usable < grid["item_price"]:
            self.logger.info(
                f"可支配金币 {usable} 不足，跳过 {grid['item_name']}（{grid['item_price']}）"
            )
            return True

        is_drink = grid["item_name"] == "potential_drink"
        drink_arg = reserve_coin if is_drink else None
        success = self._buy_item(context, grid["price_roi"], drink_arg)

        if success:
            grid["bought"] = True
            return True

        if not context.tasker.stopping:
            # TODO(samipale): 购买操作出现问题改为跳到下一个格子，但需要做一个异常情况的pipeline处理，回到购买界面
            self.logger.error("购买操作出现问题，将中止任务")
            context.tasker.post_stop()
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
            self.logger.info("刷新次数已用完，停止刷新")
            return False

        refresh_cost = self._get_refresh_cost(context)
        usable = max(0, _get_current_coin(context) - reserve_coin)
        if usable >= refresh_cost + min_buyable_price:
            return True

        self.logger.info("可支配金币不足以刷新并购买，停止刷新")
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
        current_coin = _get_current_coin(context)
        current_cost = self._get_enhancement_cost(context)
        count = _calculate_max_enhance_count(
            current_coin, current_cost, params["max_cost"], params["enhance_step"]
        )
        for _ in range(count):
            context.run_task("星塔_节点_商店_点击强化_agent")
        return True

    def _get_params(self, context: Context, node_name: str) -> dict:
        """从节点 attach 读取强化配置参数，缺失时返回安全默认值。

        Args:
            context: 任务上下文。
            node_name: 当前节点名称。

        Returns:
            dict: 包含 max_cost 和 enhance_step。
        """
        defaults = {"max_cost": 180, "enhance_step": 60}
        node_data = context.get_node_data(node_name)
        if not node_data:
            self.logger.warning("get_node_data 返回 None，使用默认参数")
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
            self.logger.debug(f"识别强化所需金币结果：{[r.text for r in reco_detail.all_results]}")
            if reco_detail and reco_detail.hit:
                self.logger.debug(f"识别到强化所需金币：{reco_detail.best_result.text}")
                return int(reco_detail.best_result.text)

            self.logger.debug("识别强化是否免费和所需金币失败，等待1秒后重试")
            time.sleep(1)
            image = context.tasker.controller.post_screencap().wait().get()

            if context.tasker.stopping:
                return 65535

        self.logger.error("无法读取当前强化所需金币数量")
        return 65535