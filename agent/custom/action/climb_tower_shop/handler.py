from __future__ import annotations

import re
import time

from . import pipeline
from .context import ShopContext, Item
from .interactor import ShopInteractor
from custom.action.climb_tower_melody_scan import auto_scan_melody_counts
from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_shop_handler")

class ShopHandler:

    def __init__(self, interactor: ShopInteractor, data: ShopContext):
        self.interactor = interactor
        self.data = data

    def execute_buy_plan(self):
        """执行购买计划"""
        self._execute(pipeline.DRINK_BUY_STRATEGY)
        self._execute(pipeline.ASSIST_MELODY_STRATEGY)
        self._execute(pipeline.MELODY_BUY_STRATEGY)
        self._execute(pipeline.HIGH_PRICE_DRINKS_STRATEGY)
        # 仅在最终商店的最后关头执行的策略
        self._execute(pipeline.REMAINING_DRINKS_STRATEGY)
        self._execute(pipeline.FINAL_REMAINDER_STRATEGY)

    def read_shop_page_info(self) -> None:
        """
        在商店层主界面读取商店层信息，更新商店层的上下文对象。
        当前信息包括商店类型、当前强化费用、当前金币数量。
        如果设置了音符策略，还会读取当前音符持有数量。
        """
        # 这里就先不确保处于商店层主界面了，因为在函数起点已经确保了且基本逻辑不会改变，直接识别就行
        self.update_coin()
        self._update_enhancement_cost()
        self._update_shop_type()
        logger.debug(f"商店类型: {self.data.shop_type}")

        # 进商店时读取各音符持有数量；未设置任何音符目标时不读取
        self._update_current_melodies()
        if self.data.melodies:
            logger.info(f"当前有购买需求的音符数量:")
            for melody in self.data.melodies.active_values():
                logger.info(f"{melody.display_name}: {melody.count}/{melody.required_count}")

    def read_goods_page_info(self) -> None:
        """
        在商店层购物界面读取当前页面的信息，更新商店层的上下文对象。
        当前信息包括商品名称、商品价格，还有刷新的剩余次数与费用。
        """
        # 进入商店层购物界面
        self.interactor.enter_shopping()

        # 读取刷新相关信息
        self._update_refresh_info()

        # 读取商品信息
        self._update_goods_info()

    def refresh(self):
        """刷新当前商店"""
        current_refresh_remaining = self.data.refresh_remaining
        self.interactor.context.run_task("星塔_节点_商店_点击刷新_agent")
        for _ in range(20):
            if self.interactor.context.tasker.stopping:
                return
            self.interactor.screenshot()
            if self.interactor.get_refresh_remaining() < current_refresh_remaining:
                logger.debug("刷新成功")
                return
            logger.debug("正在等待刷新完成……")
            time.sleep(1)
        logger.error("等待刷新超时，为保证爬塔质量，将中止任务")
        self.interactor.context.tasker.post_stop()

    def enhance(self):
        """进行强化"""
        # 确保在商店层购物界面
        self.interactor.context.run_task("星塔_节点_商店_万能返回商店层_agent")

        # 计算可强化次数
        count = self.data.total_enhancement_count
        if self.data.shop_type == "final":
            count = self.data.uncapped_enhancement_count
        logger.debug(f"最大强化辉光币: {self.data.params.max_enhancement_cost}"
                     f"当前强化所需辉光币: {self.data.current_enhancement_cost}"
                     f"当前辉光币: {self.data.current_coin}"
                     f"可强化次数: {count}")

        # 开始强化
        for _ in range(count):
            if not self.interactor.enhance():
                logger.error(f"强化出现问题，终止强化")
                return

    def _execute(self, buy_plan: pipeline.BuyPipeline):
        """执行单个购买计划"""
        for _ in range(20):
            if self.interactor.context.tasker.stopping:
                return
            target = buy_plan.step(self)
            if not target:
                return
        logger.error(f"购买计划 {buy_plan.name} 已执行20次且没有完成，为保证爬塔质量，将中止任务")
        self.interactor.context.tasker.post_stop()

    def _update_goods_info(self):
        """识别购物界面 8 个格子的道具信息。

        每个格子识别名称、数量、价格，计算折扣比值后组装为 Item，
        并初始化 bought、buy_type、buy_priority 等字段。
        """
        items = []
        lang = self.data.params.lang

        for i, item_roi in enumerate(self.data.params.item_rois):
            logger.debug(f"正在识别第 {i + 1} 个格子")
            item_name, item_quantity = self._get_item_name_and_quantity(item_roi)
            item_price = self._get_item_price(item_roi)
            trekker_specified = False
            if item_name == "potential_drink":
                trekker_specified = self.interactor.is_specified_drink(item_roi["item_roi"])

            if item_name and item_quantity and item_price:
                display_name = self.data.params.item_translations.get(item_name, {}).get(lang, ["?"])[0]
                items.append(Item(
                    grid_num=i + 1,
                    internal_name=item_name,
                    quantity=item_quantity,
                    price=item_price,
                    trekker_specified=trekker_specified,
                    display_name=display_name
                ))
            else:
                logger.error(
                    f"第 {i + 1} 个格子内容识别失败："
                    f"item_name={item_name}, item_quantity={item_quantity}, item_price={item_price}"
                )

        logger.debug(f"道具列表: {items}")
        self.data.items = items

    def _get_item_name_and_quantity(self, item_roi: dict[str, list[int]]) -> tuple[str, int]:
        """通过识别获得商品名称及数量，因为商品名称及数量在同一个 roi 中"""
        roi = item_roi["name_roi"]
        reco_result = self.interactor.recognize_item_name_quantity(roi)
        split_text = " " if self.data.params.lang == "en" else ""
        item_text = split_text.join(reco_result)
        item_name, item_quantity = self._parse_item_name(item_text)
        return item_name, item_quantity

    def _parse_item_name(self, item_text: str) -> tuple[str, int]:
        """从 OCR 原始字符串中解析物品内部名称和数量。

        先按 "名称 x数量" 格式拆分，再将语言显示名映射为程序内部通用名称。
        潜能特饮数量统一视为 1。

        Args:
            item_text: OCR 识别到的原始物品名称字符串。

        Returns:
            tuple[str, int]: (item_name, item_quantity)，
                item_name 为内部通用名，item_quantity 为数量（0 表示未识别）。
        """
        match = re.match(r"(.*?)\s*[x×]\s*(\d+)$", item_text, re.IGNORECASE)
        item_name = item_text.strip()
        item_quantity = 0

        if match:
            item_name = match.group(1).strip()
            item_quantity = int(match.group(2))

        internal_name = self.data.parse_item_name(item_name) or item_name

        if internal_name == "potential_drink":
            item_quantity = 1

        return internal_name, item_quantity

    def _get_item_price(self, item_roi: dict[str, list[int]]) -> int:
        """通过识别获得商品价格"""
        roi = item_roi["price_roi"]
        reco_result = self.interactor.recognize_item_price(roi)
        item_price = self._clean_and_extract_price(reco_result)
        return item_price

    @staticmethod
    def _clean_and_extract_price(reco_result: list[str]) -> int:
        """从 OCR 原始数据中提取物品实际价格。

        过滤掉错误识别为 0 的结果，清洗混入的原价（4位以上数字截取末段），
        例如 "09045"->45, "400200"->200，
        最终取所有候选值中的最小值。

        Args:
            reco_result: OCR 识别到的价格原始数据，字符串列表。

        Returns:
            int: 解析后的物品价格；无有效结果时返回 0。
        """
        def _clean_price(p: str) -> int | None:
            if not p.isdecimal() or int(p) in {0, 1, 11}:
                return None
            # 4位及以上尝试通过正则截取末段（去除 xx0/x0 系列前缀）
            if len(p) >= 4:
                p = re.sub(r'^.*?0(?=[1-9])', '', p)
            return int(p) if p.isdecimal() else None

        prices = [price for r in reco_result if (price := _clean_price(r)) is not None]
        return min(prices, default=0)

    def _update_refresh_info(self):
        """更新刷新相关信息"""
        self.data.refresh_remaining = self.interactor.get_refresh_remaining()
        self.data.refresh_cost = self.interactor.get_refresh_cost()

    def update_coin(self):
        """更新当前辉光币数量"""
        self.data.current_coin = self.interactor.get_current_coin()

    def _update_enhancement_cost(self):
        """更新当前强化费用"""
        self.data.current_enhancement_cost = self.interactor.get_enhancement_cost()

    def _update_shop_type(self):
        """更新当前商店类型"""
        self.data.shop_type = self.interactor.check_shop_type()

    def _update_current_melodies(self):
        """更新当前有购买需求的音符数量"""
        melody_counts = auto_scan_melody_counts(self.interactor.context, self.data)
        self.data.melodies.update_from_count_dict(melody_counts)

    def can_afford_refresh(self) -> bool:
        """判断当前是否满足刷新条件。与data.should_refresh逻辑保持一致"""
        if self.data.refresh_remaining <= 0:
            logger.info("刷新次数已用完，跳过刷新")
            return False

        coin = self.data.current_coin
        threshold = self.data.refresh_threshold
        enhancement_cost = self.data.total_enhancement_cost
        total = threshold + enhancement_cost

        if self.data.should_refresh:
            logger.info(f"可用金币 {coin} 达到刷新标准: 阈值 {threshold} + 强化消耗 {enhancement_cost} = {total}，尝试刷新")
            return True

        logger.info(f"可用金币 {coin} 未达到刷新标准: 阈值 {threshold} + 强化消耗 {enhancement_cost} = {total}，跳过刷新")
        return False
