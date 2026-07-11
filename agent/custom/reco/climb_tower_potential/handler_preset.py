from math import comb
from typing import Self

from .state import State, OwnedPotential
from .data import Data, Potential
from .ui import UIInteractor
from .handler_default import ChoosePotentialHandler

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential_preset")


class RecommendationHandler(ChoosePotentialHandler):
    HANDLER_TYPE = "preset"

    def __init__(self, screen: UIInteractor, data: Data):
        super().__init__(screen, data)

    def read_potentials_info(self) -> Self:
        self._wait_for_item_list_gone()
        self.data.potentials = self.initialize_potentials()

        self._update_recommended_potentials()
        self._update_names()
        self._update_levels()
        self._update_trekkers()

        # 输出当前潜能列表到日志
        for potential in self.data.potentials:
            recommended_output = f"系统推荐{potential.recommended_level}级" if potential.recommended_level > 0 else "无"
            if self.data.core_potential:
                logger.info(f"[潜能识别] {potential.name} | 核心潜能 | {recommended_output}")
            else:
                old = potential.old_level
                new = potential.new_level
                logger.info(f"[潜能识别] {potential.name} | 等级 {old}→{new} | {recommended_output}")

        return self

    def choose(self) -> Potential | None:
        # 根据参数选择潜能选择器
        if self.data.params.environment.startswith("tower_8"):
            best_potential = self.tower_8_chooser()
        else:
            best_potential = self.choose_fallback_potential()

        if best_potential:
            logger.info(f"[潜能选择] {best_potential.name}")

        return best_potential

    def tower_8_chooser(self) -> Potential | None:
        """
        塔8专用策略，采用打分制
        根据用户设置的最大刷新次数以及刷新分数阈值，判断是否需要刷新

        Returns:
            Potential | None: 最好的系统推荐潜能，若没有则返回 None
        """

        # 核心潜能选择，直接选择推荐潜能，如没有则选择默认潜能
        if self.data.core_potential:
            best_potential = max(
                (p for p in self.data.potentials if p.recommended),
                key=lambda p: p.recommended_level,
                default=self._default_potential,
            )
            self._tower_8_record(best_potential)
            return best_potential

        # 打分
        for p in self.data.potentials:
            p.score = self._tower_8_score(p)

        # 取得最优潜能
        best_potential = max(
            self.data.potentials,
            key=lambda p: (
                p.score,
                p.recommended_level,
                p.level_span,
                p.old_level,
            ),
        )

        # 不能刷：强化潜能、金币不足、刷新次数用尽等，直接选当前最优。
        if not self.data.refresh_botton or not self.data.refreshable:
            self._tower_8_record(best_potential)
            return best_potential

        # 计算刷新阈值
        if self.data.threshold < 0.0:
            self.data.threshold = self._tower_8_threshold()
        threshold = self.data.threshold * (1 - self.data.params.threshold_decay * self.data.refresh_count)

        # 当前牌到达刷新分数阈值，直接选当前最优，否则返回 None 让外层刷新。
        logger.info(f"当前最优牌 {best_potential.name} 得分 {best_potential.score}，阈值 {threshold}")
        if best_potential.score >= threshold:
            self._tower_8_record(best_potential)
            return best_potential

        return None

    def _tower_8_score(self, p: Potential | OwnedPotential, *, level_span: int = 0) -> int | float:
        """
        塔8专用潜能打分核心函数，满分为100分
        由于还没摸透潜能抽取规则，所以只能使用平均概率的方法去计算
        默认目标为6+5+5=16个潜能
        """
        # 已获得潜能可能需要从保存的State类中获得推荐等级
        if isinstance(p, Potential) and p.old_level > 0 and not p.recommended:
            p.recommended_level = State.owned_potentials.find_recommended_level(p.name, mode="FUZZY", trekker=p.trekker)
            p.recommended = True if p.recommended_level >= 0 else False
        # 传入OwnedPotential对象时，需要设置等级跨度，然后转为Potential对象处理
        if isinstance(p, OwnedPotential):
            if level_span == 0:
                logger.warning("传入OwnedPotential对象时未设置等级跨度，将默认为1")
                level_span = 1
            new_level = p.level + level_span
            recommended = True if p.recommended_level > 0 else False
            p = self.dummy_potential(
                name=p.name, trekker=p.trekker, old_level=p.level, new_level=new_level, recommended=recommended,
                recommended_level=p.recommended_level, core=p.core
            )

        # 取得有效升级量，只有在升级到推荐等级上面才是有效升级。
        effective_gain = max(0, min(p.new_level, p.recommended_level) - min(p.old_level, p.recommended_level))

        if p.old_level == 0:
            # 新潜能
            # 在辉光的奇迹buff没用完的情况下，新潜能升级量为3是100分，2是66分，1是33分，0是0分
            # 在辉光的奇迹buff用完的情况下，新潜能升级量为2是100分，1是50分，0是0分
            max_recommended_level = 3 if State.high_level_span_count < 10 else 2
            # 计算有没有有效利用辉光的奇迹buff
            wasted_gain = max(0, p.new_level - p.recommended_level) if max_recommended_level == 3 else 0
            # 计算分数
            score = max(0, effective_gain - wasted_gain) / max_recommended_level * 100
        else:
            # 对于有效升级的老潜能，通过已保有潜能种类数打分，因为永远是以新潜能优先
            score = min(16, State.owned_potentials.count()) / 16 * 100 * effective_gain

        return round(score, 2)

    # TODO: 研究抽取规则，然后重构
    def _tower_8_threshold(self) -> int | float:
        """
        塔8专用刷新阈值计算函数
        由于还没摸透潜能抽取规则，所以暂时只能使用平均概率的方法去计算
        然后通过用户设置的激进程度去调整
        默认目标为6+5+5=16个6级潜能
        """
        # 1. 统计新潜能的种类数
        trekkers = ["0", "1", "2"]
        owned_stats = {
            t: {
                "total": State.owned_potentials.count(trekker=t),
                "leveling": State.owned_potentials.count(trekker=t, leveling_only=True)
            }
            for t in trekkers
        }
        new_potential_counts = [
            0 if stats["total"] >= 5 and stats["leveling"] >= 3
            else (12 - stats["total"])
            for stats in owned_stats.values()
        ]
        new_potential_count = sum(new_potential_counts)

        if new_potential_count == 0:
            return 0

        # 2. 计算新旧潜能的期望分数
        new_potential_scores = self._calculate_new_potential_scores(new_potential_count)
        old_potential_scores = self._calculate_old_potential_scores()
        potential_scores = new_potential_scores + old_potential_scores

        # 3. 计算特定潜能在3次抽取中成为最佳潜能的期望分数
        if len(potential_scores) <= 3: # 卡池只剩3个潜能代表当前显示的3个跟刷新后的3个会一模一样，不需要刷新
            return 0
        potential_scores.sort()
        potential_count = len(potential_scores)
        expected_score = sum(
            potential_scores[i] * comb(i, 2) / comb(potential_count, 3)
            for i in range(2, potential_count)
        )

        # 4. 把刷新价值量化为潜能抽取的期望分数
        # 该系数由潜能特饮价格抽象而来，如果潜能特饮平均买入价格为160，那么一次潜能抽取的转换系数为160/200=0.8
        threshold = expected_score * self.data.params.threshold_coef

        return round(max(0.0, min(100.0, threshold)), 2)

    def _calculate_new_potential_scores(self, new_potential_count: int) -> list[int | float]:
        """计算所有新潜能的总期望分数"""
        # 新潜能等级概率权重: Lv1(30%), Lv2(20%), Lv3(50%)
        level_weights = [(1, 0.3), (2, 0.2), (3, 0.5)] if State.high_level_span_count < 10 else [(1, 0.3), (2, 0.7)]

        recommended_score = sum(
            self._tower_8_score(self.dummy_potential(old_level=0, new_level=lv, recommended_level=6)) * weight
            for lv, weight in level_weights
        )
        # 由于6/5的硬上限限制，所以即使拿到了垃圾潜能，也只能当作拿到推荐潜能算
        new_recommended_potential_count = max(0, 16 - State.owned_potentials.count())
        # 计算期望分数
        recommended_scores = [recommended_score for _ in range(new_recommended_potential_count)]
        unrecommended_scores = [0 for _ in range(new_potential_count - new_recommended_potential_count)]

        return recommended_scores + unrecommended_scores

    def _calculate_old_potential_scores(self) -> list[int | float]:
        """计算所有已有潜能升一级的总分数"""
        return [
            self._tower_8_score(p, level_span=1)
            for p in State.owned_potentials
            if not p.core and p.level < 6
        ]

    @staticmethod
    def _tower_8_record(p: Potential) -> None:
        State.potentials_level_count += p.level_span
        logger.info(f"潜能计数 {State.potentials_level_count}")
        if not p.core:
            if p.old_level == 0 and p.level_span >= 3:
                State.high_level_span_count += 1
                logger.info(f"辉光的奇迹计数 {State.high_level_span_count}/10")

            if p.old_level > 0 and p.level_span == 2:
                State.enhance_high_level_span_count += 1
                logger.info(f"潜能飞升计数 {State.enhance_high_level_span_count}/5")



class RecommendationPlusBagScanHandler(RecommendationHandler):
    def __init__(self, screen: UIInteractor, data: Data):
        super().__init__(screen, data)

    # TODO：
    # def _tower_8_score(self, p: Potential) -> int:
    # 因为能获得整体情况，所以还需要考虑下面问题
    # 还得考虑潜能到达旅人上限的问题，以保证推荐潜能数超过普通上限的情况打分偏向于能够突破上限
    # 推荐等级越高分数越高的修正

    # def _tower_8_threshold(self, threshold: int) -> int:
    # 这个可能不用调整思路，但需要把默认16种潜能改为整个潜能库的概率计算

