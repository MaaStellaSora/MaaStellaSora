from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
import json
import time
import re
from utils import logger

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

        # 开启debug模式
        # self.logger.debug_mode()

        # 获取资源类型
        param_str = argv.custom_action_param
        param = json.loads(param_str)
        self.lang_type = param.get("lang_type")

        # 先检查是中途的商店还是最终商店
        shop_type = self._check_shop_type(context)

        # 定义初始参数
        # TODO: 通过强化参数预留金币
        reserve_coin = 0

        # 然后进入商店购物的页面
        context.run_task("星塔_节点_商店_点击商店购物_agent")
        # 开始第一轮购买
        while True:
            # 读取8个格子的信息
            grids_infos = self._get_grids_info(context)

            # 按照策略决定购买内容
            # TODO: 其他方案
            # 方案1：只买潜能特饮
            buy_list = [grid for grid in grids_infos if grid["item_name"] == "potential_drink"]
            # 方案2：优先购买潜能特饮，其次根据打折力度购买音符
            # 方案3：优先根据打折力度凑1级附加技能，其次购买潜能特饮
            # 方案4：买到没钱

            # 读取当前金币
            coin = self._get_current_coin(context)
            if coin:
                coin = max(0, coin - reserve_coin)
            else:
                self.logger.error("无法读取金币，为保证爬塔质量，将中止任务")
                context.tasker.post_stop()
                return False

            # 执行购买操作
            for item in buy_list:
                if coin >= item["price"]:
                    # 获取道具所属格子范围
                    buy_result = self._buy_item(context, item["price_roi"])
                    if buy_result:
                        coin = coin - reserve_coin
                    else:
                        if not context.tasker.stopping:
                            self.logger.error("购买操作出现问题，为保证爬塔质量，将中止任务")
                            context.tasker.post_stop()
                        return False

            # 循环完毕后，根据是中途商店还是最终商店，决定是否刷新物品继续新一轮的购买
            if shop_type == "regular":
                break
            else:
                refresh_remaining = self._get_refresh_remaining(context)
                if refresh_remaining > 0 and coin >= 145:
                    context.run_task("星塔_节点_商店_购物_点击刷新_agent")
                elif refresh_remaining == 0:
                    self.logger.info("刷新次数已用完，无法继续刷新")
                    break
                elif coin < 145:
                    self.logger.info("可使用金币不足145，跳过刷新")
                    break

        # 退回商店层主界面
        context.run_task("星塔_节点_商店_购物_返回商店层_agent")
        return True

    @staticmethod
    def _check_shop_type(context, image = None):
        """
            检查商店类型

            Args:
                context(Context): 上下文对象
                image(nd.array): 截图，默认为None

            Returns:
                str: 商店类型，中途商店为regular，最终商店为final，如果没有结果为空值
        """
        if not image:
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

    def _get_current_coin(self, context, image = None, max_try = 3):
        """
            检查当前金币

            Args:
                context(Context): 上下文对象
                image(nd.array): 截图
                max_try(int): 最大尝试次数

            Returns:
                int | None: 当前金币数量，识别失败时返回None
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        for _ in range(max_try):
            reco_detail = context.run_recognition("星塔_通用_识别当前金币_agent", image)
            self.logger.debug(f"识别当前金币结果：{[r.text for r in reco_detail.all_results]}")
            if reco_detail and reco_detail.hit:
                return int(reco_detail.best_result.text)

            # 失败时，等待1秒后重试
            time.sleep(1)
            image = context.tasker.controller.post_screencap().wait().get()

            # 检查是否中断任务
            if context.tasker.stopping:
                return None

        return None

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
            # TODO： 潜能特饮增加对主控专用的检测
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
                })
            else:
                self.logger.error(f"第{i+1}个格子内容识别失败")
                self.logger.error(f"item_name: {item_name}, item_quantity: {item_quantity}, item_price: {item_price}")

            # 根据价格从低到高排序
            grids_info.sort(key=lambda x: int(x["item_price"]))

        return grids_info

    def _get_single_grid_info(self, context, item_roi, price_roi, name_roi, retry_max=3):
        """
            格子内容识别函数

            Args:
                context(Context): 上下文对象
                item_roi(list): 整个道具格子的识别框范围
                price_roi(list): 道具价格的识别框范围
                name_roi(list): 道具名的识别框范围
                retry_max(int): 最大重试次数，默认为3

            Returns:
                tuple: 格子的道具信息，包含item_name, item_quantity, item_price, discount_flag
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
    def _buy_item(context, roi):
        """
            使用pipeline执行购买操作

            Args:
                context(Context): 上下文对象
                roi(list): 道具的点击范围

            Returns:
                bool: 是否完美执行购买操作
        """
        run_result = context.run_task("星塔_节点_商店_购物_购买道具_agent", {
            "星塔_节点_商店_购物_购买道具_agent": {
                "action": {
                    "param": {
                        "target": roi
                    }
                }
            }
        })
        if run_result and run_result.status.succeeded:
            return True
        return False

    @staticmethod
    def _get_discount(item_name, item_quantity, item_price):
        """
            获取物品的折扣信息

            Args:
                item_name(str): 物品名称
                item_quantity(int): 物品数量
                item_price(int): 物品价格

            Returns:
                int: 折扣信息，0表示无折扣，1表示有折扣
        """
        # 检查是否为潜能特饮
        if item_name == "potential_drink":
            return item_price / 200.0

        # 检查是否为音符
        if "melody" in item_name and item_quantity == 5:
            return  item_price / 90.0

        if "melody" in item_name and item_quantity == 15:
            return  item_price / 400.0

        # 其他情况
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

    # TODO：加到pipeline里处理，agent不处理
    FREE_ENHANCE_TEXT = {
        "zh": "免费",
        "tw": "免費",
        "en": "FREE",
        "jp": "無料",
    }

    def __init__(self):
        super().__init__()
        self.logger = logger.get_logger(__name__)

    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:

        # TODO:读取配置中的允许最大强化消耗
        max_cost = 180
        # 检查当前金币
        current_coin = self._get_current_coin(context)
        # 检查当前强化所需金币
        current_enhancement_cost = self._get_enhancement_cost(context)
        # 计算出可强化次数
        # TODO:默认按60算，但如果星塔等级没满就会出问题，配置加个选项会比较好
        max_enhance_count = self._calculate_max_enhance_count(current_coin, current_enhancement_cost, max_cost)

        # 开始循环
        for _ in range(max_enhance_count):
            context.run_task("星塔_节点_商店_点击强化_agent")

        return True

    def _get_current_coin(self, context, image = None, max_try = 3):
        """
            检查当前金币

            Args:
                context(Context): 上下文对象
                image(nd.array): 截图
                max_try(int): 最大尝试次数

            Returns:
                int | None: 当前金币数量，识别失败时返回None
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        for _ in range(max_try):
            reco_detail = context.run_recognition("星塔_通用_识别当前金币_agent", image)
            self.logger.debug(f"识别当前金币结果：{[r.text for r in reco_detail.all_results]}")
            if reco_detail and reco_detail.hit:
                return int(reco_detail.best_result.text)

            # 失败时，等待1秒后重试
            time.sleep(1)
            image = context.tasker.controller.post_screencap().wait().get()

            # 检查是否中断任务
            if context.tasker.stopping:
                return None

        return None

    def _get_enhancement_cost(self, context, image = None):
        """
            检查当前强化所需金币

            Args:
                context(Context): 上下文对象

            Returns:
                int | None: 当前强化所需金币数量，识别失败时返回None
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        for _ in range(3): # 最多尝试3次
            reco_detail = context.run_recognition("星塔_节点_商店_识别强化是否免费_agent", image)
            self.logger.debug(f"识别强化是否免费结果：{[r.text for r in reco_detail.all_results]}")
            if reco_detail and reco_detail.hit:
                self.logger.debug(f"识别到免费强化")
                return 0
            reco_detail = context.run_recognition("星塔_节点_商店_识别强化所需金币_agent", image)
            self.logger.debug(f"识别强化所需金币结果：{[r.text for r in reco_detail.all_results]}")
            if reco_detail and reco_detail.hit:
                self.logger.debug(f"识别到强化所需金币：{reco_detail.best_result.text}")
                return int(reco_detail.best_result.text)

            # 失败时，等待1秒后重试
            self.logger.debug(f"识别强化是否免费和所需金币失败，等待1秒后重试")
            time.sleep(1)
            image = context.tasker.controller.post_screencap().wait().get()

            # 检查是否中断任务
            if context.tasker.stopping:
                return None

        return None

    @staticmethod
    def _calculate_max_enhance_count(current_coin, current_enhancement_cost, max_cost):
        """
            计算出可强化次数

            Args:
                current_coin(int): 当前金币数量
                current_enhancement_cost(int): 当前强化所需金币数量
                max_cost(int): 允许最大强化消耗

            Returns:
                int: 可强化次数
        """
        count = 0
        step = 60

        # 由于次数不多，使用while循环暴力解决
        while current_coin >= current_enhancement_cost and current_enhancement_cost <= max_cost:
            current_coin -= current_enhancement_cost  # 扣钱
            count += 1  # 增加一次升级
            current_enhancement_cost += step  # 下一次的消耗变得更贵

        return count
