from itertools import combinations, permutations, product
from math import exp, prod
from typing import Self, Any

from .state import State, OwnedPotential, Trekker
from .data import Data, Potential
from .interactor import PotentialInteractor
from .handler_default import ChoosePotentialHandler

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential_preset")


class RecommendationHandler(ChoosePotentialHandler):
    HANDLER_TYPE = "preset"
    P_NEW_MAP = (1.0, 2 / 3, 1 / 3, 0.0) # 到达软上限时的新潜能概率映射表，索引为未满级潜能数量，值为新潜能概率

    def __init__(self, screen: PotentialInteractor, data: Data):
        super().__init__(screen, data)

    def read_potentials_info(self) -> Self:
        self.data.potentials = self.initialize_potentials()

        self._update_recommended_potentials()
        self._update_names()
        self._update_levels()
        self._update_trekkers()

        # 输出当前潜能列表到日志
        for potential in self.data.potentials:
            lvl = "?" if potential.recommended_level < 0 and potential.recommended else potential.recommended_level
            recommended_output = f"系统推荐{lvl}级" if lvl > 0 else "无"
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
            best_potential = self.default_chooser() if self.data.refreshable else self.choose_fallback_potential()

        if best_potential:
            logger.info(f"[潜能选择] {best_potential.name}")

        return best_potential

    def default_chooser(self):
        """
        默认策略，根据等级跨度、推荐等级、当前等级排序来选择推荐潜能，如没有推荐潜能则返回 None
        """
        candidates = [p for p in self.data.potentials if p.recommended]
        if candidates:
            # 按照等级跨度降序、推荐等级降序、旧等级降序来排序，选择最优的潜能
            return max(candidates, key=lambda p: (p.level_span, p.recommended_level, p.old_level), default=None)
        return None

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

        # 寻找主旅人的兜底策略
        if self.data.refresh_count == 0 and not State.get_main_trekker():
            owned_potential_count_by_trekker = State.owned_potentials.count_by_trekkers()
            for trekker, count in owned_potential_count_by_trekker.items():
                if count >= 6:
                    trekker.main = True
                    logger.debug("已通过触发潜能种类上限识别到主旅人")
                    break

        # 打分
        for p in self.data.potentials:
            p.score = self._tower_8_score(p)
            p.probability = self._tower_8_probability(p)
            logger.debug(f"[潜能打分] {p.name} | 得分 {p.score} | 概率 {p.probability:.2%}")

        # 取得最优潜能
        best_potential = max(
            self.data.potentials,
            key=lambda p: (
                p.score,
                p.recommended_level,
                -p.probability,
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
        threshold = round(self.data.threshold * (1 - self.data.params.threshold_decay * self.data.refresh_count), 2)
        threshold = max(0.0, threshold)

        # 当前牌到达刷新分数阈值，直接选当前最优，否则返回 None 让外层刷新。
        logger.info(f"当前最优牌 {best_potential.name} 得分 {best_potential.score}，阈值 {threshold}")
        if best_potential.score >= threshold:
            self._tower_8_record(best_potential)
            return best_potential

        return None

    def _tower_8_score(self, p: Potential | OwnedPotential, *, level_span: int = 0) -> float:
        """
        塔8专用潜能打分函数
        """
        # 已获得潜能可能需要从保存的State类中获得推荐等级
        # 如果重新启动过agent再从途中开始，会丢失潜能数据，到时候推荐等级取得一定会有问题，所以要提示用户最好不要中断
        if isinstance(p, Potential) and p.old_level > 0 and not p.recommended:
            p.recommended_level = State.owned_potentials.find_recommended_level(p.name, mode="FUZZY", trekker=p.trekker)
            p.recommended = True if p.recommended_level >= 0 else False
        # 传入OwnedPotential对象时，需要设置等级跨度，然后转为Potential对象处理
        if isinstance(p, OwnedPotential):
            if level_span == 0:
                logger.warning("传入OwnedPotential对象时未设置等级跨度，将默认为1")
                level_span = 1
            p = self.dummy_potential(
                name=p.name, trekker=p.trekker, old_level=p.level, new_level=min(6, p.level + level_span),
                recommended=p.recommended, recommended_level=p.recommended_level, type=p.type
            )

        # 取得有效升级量，只有在升级到推荐等级上面才是有效升级。
        effective_gain = max(0, min(p.new_level, p.recommended_level) - min(p.old_level, p.recommended_level))
        if effective_gain <= 0:
            return 0.0

        # 对新潜能未能有效利用“心花怒放/辉光的奇迹”进行惩罚扣分
        # 但由于本模式默认655都是推荐等级满级，扣分会跟阈值计算对不上，所以暂时注释掉
        # if p.old_level == 0:
        #     max_recommended_level = 3 if State.high_level_span_count < 10 else 2
        #     wasted_gain = max(0, p.new_level - p.recommended_level) if max_recommended_level == 3 else 0
        #     effective_gain = max(0, effective_gain - wasted_gain)

        # 稍微增加一点点推荐等级权重（考虑到本模式推荐等级有造假情况，还是不要加了）
        # recommended_weight = 1 + 0.05 * max(0, p.recommended_level)

        # 计算最终分数
        score = effective_gain

        return round(score, 2)

    def _tower_8_probability(self, p: Potential) -> float:
        """
        塔8专用当前潜能出现概率计算函数
        """
        owned_potential_count_by_trekkers = State.owned_potentials.count_by_trekkers()
        owned_potential_count = owned_potential_count_by_trekkers.get(p.trekker, 0)
        owned_leveling_count = State.owned_potentials.count(trekker=p.trekker, leveling_only=True)

        trekker_weight = self._calculate_trekker_weight(p.trekker, owned_potential_count_by_trekkers)
        holding_weight = self._calculate_holding_weight(p.trekker, p.old_level, owned_potential_count, owned_leveling_count)

        return trekker_weight * holding_weight

    def _calculate_trekker_weight(
            self,
            target_trekker: Trekker,
            owned_potential_count_by_trekkers: dict[Trekker, int]
    ) -> float:
        """
        计算旅人权重
        该公式只是近似拟合数据，而不是游戏的实际公式
        """
        # 主旅人饮料
        if self.data.params.potential_source == "specified_drink":
            return 1.0 if target_trekker.main else 0.0

        # 补充到3个旅人，在游戏初期卡包可能不满3个旅人
        missing_count = 3 - len(owned_potential_count_by_trekkers)
        if missing_count > 0:
            placeholders = {Trekker(index=-(i + 1)): 0 for i in range(missing_count)}
            owned_potential_count_by_trekkers = owned_potential_count_by_trekkers.copy() | placeholders

        scores = []
        # 若匹配不到任何旅人，视为初始未获取状态 (c=0, is_main=0, is_capped=0)，score 为 0
        target_score = 0.0
        for trekker, potential_count in owned_potential_count_by_trekkers.items():
            is_main = 1 if trekker.main else 0
            cap = 6 if is_main else 5
            is_capped = int(potential_count >= cap)
            score = 0.056 * potential_count - 0.092 * potential_count ** 2 + 0.80 * is_main - 2.18 * is_capped
            target_score = score if trekker == target_trekker else target_score
            scores.append(score)

        max_score = max(scores)
        exp_scores = [exp(s - max_score) for s in scores]
        target_exp_score = exp(target_score - max_score)

        return target_exp_score / sum(exp_scores)

    def _calculate_holding_weight(
            self,
            target_trekker: Trekker,
            old_level: int,
            owned_potential_count: int,
            owned_leveling_count: int
    ) -> float:
        """
        计算新旧潜能权重
        该公式只是近似拟合数据，而不是游戏的实际公式
        """
        # 防止极端情况
        remaining = 12 - owned_potential_count
        cap = 6 if target_trekker.main else 5

        # 1. 新/旧潜能类别概率
        if remaining <= 0:
            # 新潜能抽满12种：无法再出新潜能，旧潜能按槽位上限分配
            p_new, p_old = 0.0, min(owned_leveling_count, 3) / 3
        elif owned_potential_count <= 1:
            # 开局阶段（≤1）：机制保底必定全为新潜能
            p_new, p_old = 1.0, 0.0
        elif owned_potential_count >= cap:
            # 到达软上限：查表获取槽位占比（1.0, 2/3, 1/3, 0.0）
            p_new = self.P_NEW_MAP[min(owned_leveling_count, 3)]
            p_old = 1.0 - p_new
        else:
            # 常规阶段：拟合曲线动态过渡
            # 当前拟合公式大概有±5~7个百分点的系统误差
            x = (cap + 1 - owned_potential_count)
            p_new = x / (x + 1.6 * owned_leveling_count)
            p_old = 1.0 - p_new

        # 2. 除以潜能数量获得单潜能真实概率
        prob = p_new if old_level == 0 else p_old
        count = remaining if old_level == 0 else owned_leveling_count

        return prob / count if count > 0 else 0.0

    def _tower_8_threshold(self) -> int | float:
        """
        塔8专用刷新阈值计算函数
        使用两阶段加权模型：
        1. 先根据旅人持有潜能比例，估计下一次抽取属于哪个旅人；
        2. 再根据该旅人当前新/旧潜能比例，在该旅人的池内加权无放回抽取3张；
        3. 以3张中的最高分作为本次抽取收益，求总期望分数作为刷新阈值基础。
        缺点：
        1. 如果不买主控潜能特饮就无法知道谁是主控旅人，所以无法得知谁拥有6潜能种类上限
        2. 无法得知还有多少有效新潜能需要获取，所以只能默认最终目标为6+5+5=16个6级潜能
        3. 仍未推算出精确计算公式，目前的数学模型仍然是近似模型
        """
        # 0. 补充到3个旅人，在游戏初期潜能列表可能不满3个旅人
        trekkers = list(State.trekkers)
        for i in range(len(trekkers), 3):
            trekkers.append(Trekker(index=i))

        # 1. 统计各旅人的新旧潜能的种类数
        owned_stats = {
            trekker: {
                "total": State.owned_potentials.count(trekker=trekker),
                "leveling": State.owned_potentials.count(trekker=trekker, leveling_only=True)
            }
            for trekker in trekkers
        }
        new_potential_counts = {
            trekker:
                0 if stats["total"] >= 5 and stats["leveling"] >= 3 and not trekker.main
                    or stats["total"] >= 6 and stats["leveling"] >= 3 and trekker.main
                else max(0, 12 - stats["total"])
            for trekker, stats in owned_stats.items()
        }
        # 获取未满级的已持有潜能列表
        old_leveling_potentials = {
            trekker: [
                p for p in State.owned_potentials
                if p.trekker == trekker and not p.core and p.level < 6
            ]
            for trekker in trekkers
        }

        total_new_count = sum(new_potential_counts.values())
        total_old_leveling_count = sum(len(potentials) for potentials in old_leveling_potentials.values())

        if total_new_count + total_old_leveling_count == 0:
            return 0

        # 2. 建立带有新旧潜能的基础期望分数的潜能池
        new_potential_pools = self._get_new_potential_pools(new_potential_counts, owned_stats)
        old_potential_pools = self._get_old_potential_pools(old_leveling_potentials, owned_stats)
        potential_pools = {trekker: new_potential_pools[trekker] + old_potential_pools[trekker] for trekker in trekkers}

        # 3. 计算期望分数
        trekker_weights = {}
        trekker_expected_scores = {}

        # 计算每个旅人乘以新旧潜能权重后的综合期望分数
        owned_potential_count_by_trekkers = {t: owned_stats[t]["total"] for t in owned_stats.keys()}
        for t in trekkers:
            if not potential_pools[t]:
                continue
            # 计算选中某个旅人的概率
            trekker_weights[t] = self._calculate_trekker_weight(t, owned_potential_count_by_trekkers)
            # 选中某个旅人后，该旅人的抽取综合期望分数，新旧潜能权重已在potential_pools计算
            trekker_expected_scores[t] = self.expected_best_score(potential_pools[t])

        # 计算每个旅人的旅人权重、新旧潜能权重后的综合期望分数，然后求和
        total_trekker_weight = sum(trekker_weights.values())
        expected_score = sum(
            trekker_expected_scores[t] * trekker_weights[t] / total_trekker_weight
            for t in trekker_weights
        )

        # 4. 把刷新价值量化为潜能抽取的期望分数
        # 该系数由潜能特饮价格抽象而来，如果潜能特饮平均买入价格为160，那么一次潜能抽取的转换系数为160/200=0.8
        threshold = expected_score * self.data.params.threshold_coef

        return round(max(0.0, threshold), 2)

    def _get_new_potential_pools(
            self,
            new_potential_counts: dict[Trekker, int],
            owned_stats: dict[Trekker, dict[str, int]]
    ) -> dict[Trekker, list[dict[str, Any]]]:
        """
        建立带有新潜能的基础期望分数及权重的潜能池

        Args:
            new_potential_counts: 各个旅人新潜能的数量，格式为{旅人对象: 新潜能数量}
            owned_stats: 各个旅人的潜能统计信息，格式为{旅人对象: {"total": 已持有潜能总数, "leveling": 已持有但未满级的潜能数}}

        Returns:
            dict[Trekker, list[dict[str, Any]]]: 各个旅人新潜能的期望分数及权重池
                格式为{旅人对象: [{"score": 期望分数, "weight": 权重}]}
        """
        # 新潜能等级概率权重: Lv1(30%), Lv2(20%), Lv3(50%)
        level_weights = [(1, 0.3), (2, 0.2), (3, 0.5)] if State.high_level_span_count < 10 else [(1, 0.3), (2, 0.7)]

        # 计算还可以获得的新推荐潜能的数量
        # 由于6/5的硬上限限制，且无法预知还有多少有效新潜能需要获取，所以如果不小心拿到了垃圾潜能，也只能当作拿到推荐潜能算
        remaining_recommended_new_count = max(0, 16 - State.owned_potentials.count())

        # 按照平均概率将新推荐潜能分配给各个旅人
        total_new_count = sum(new_potential_counts.values())
        recommended_new_counts = {t: 0 for t in new_potential_counts.keys()}
        if total_new_count > 0 and remaining_recommended_new_count > 0:
            # 通过平均概率计算每个旅人可能获得的新推荐潜能的数量的期望值
            raw_allocations = {
                trekker: remaining_recommended_new_count * new_count / total_new_count
                for trekker, new_count in new_potential_counts.items()
            }
            # 防止数字超过旅人可以获得的潜能种类上限，且移除小数部分
            recommended_new_counts = {
                trekker: min(new_potential_counts[trekker], int(raw_allocations[trekker]))
                for trekker in new_potential_counts.keys()
            }
            # 使用最大余数法分配多出来的潜能
            # 1. 计算剩下的潜能数量
            remainder = remaining_recommended_new_count - sum(recommended_new_counts.values())
            # 2. 按照余数从大到小排序
            allocation_order = sorted(
                new_potential_counts.keys(),
                key=lambda trekker: raw_allocations[trekker] - int(raw_allocations[trekker]),
                reverse=True,
            )
            # 3. 通过余数从大到小分配多出来的潜能
            for trekker in allocation_order:
                if remainder <= 0:
                    break
                if recommended_new_counts[trekker] < new_potential_counts[trekker]:
                    recommended_new_counts[trekker] += 1
                    remainder -= 1

        potential_pools = {trekker: [] for trekker in new_potential_counts.keys()}
        # 给每一张新潜能打分
        for trekker, new_count in new_potential_counts.items():
            # 单个新潜能的期望分数
            recommended_new_results = [
                (self._tower_8_score(
                    self.dummy_potential(trekker=trekker, old_level=0, new_level=lv, recommended_level=6)), weight)
                for lv, weight in level_weights
            ]

            # 推荐目标给期望分，非推荐目标给0分。
            for i in range(new_count):
                potential_pools[trekker].append({
                    "results": recommended_new_results if i < recommended_new_counts[trekker] else [(0.0, 1.0)],
                    "weight": self._calculate_holding_weight(
                        trekker, 0, owned_stats[trekker]["total"], owned_stats[trekker]["leveling"]
                    ),
                })

        return potential_pools

    def _get_old_potential_pools(
            self,
            old_leveling_potentials: dict[Trekker, list[OwnedPotential]],
            owned_stats: dict[Trekker, dict[str, int]]
    ) -> dict[Trekker, list[dict[str, Any]]]:
        """
        建立持有但未满级的潜能升一级时的基础期望分数及权重的潜能池

        Args:
            old_leveling_potentials: 各个旅人已持有但未满级的潜能，格式为{旅人对象: [潜能列表]}
            owned_stats: 各个旅人已持有但未满级的潜能统计信息，格式为{旅人对象: {"leveling": 已持有但未满级的潜能数}}

        Returns:
            dict[Trekker, list[dict[str, Any]]]: 各个旅人已持有但未满级的潜能的期望分数及权重池
                格式为{旅人对象: [{"score": 期望分数, "weight": 权重}]}
        """
        potential_pools = {trekker: [] for trekker in owned_stats.keys()}
        for trekker in owned_stats.keys():
            for p in old_leveling_potentials[trekker]:
                potential_pools[trekker].append({
                    "results": [(self._tower_8_score(p, level_span=1), 1.0)],
                    "weight": self._calculate_holding_weight(
                        trekker, p.level, owned_stats[trekker]["total"], owned_stats[trekker]["leveling"]
                    ),
                })
        return potential_pools

    @staticmethod
    def expected_best_score(pool: list[dict]) -> float:
        """计算单个旅人池内，加权无放回抽3张后的最高分期望。"""
        if not pool:
            return 0.0

        if len(pool) <= 3:
            return sum(
                max(score for score, prob in outcome_combo) * prod(prob for score, prob in outcome_combo)
                for outcome_combo in product(*(card["results"] for card in pool))
            )

        total_weight = sum(card["weight"] for card in pool)
        expected = 0.0

        for comb_cards in combinations(pool, 3):
            comb_probability = 0.0

            for ordered_cards in permutations(comb_cards):
                w1 = ordered_cards[0]["weight"]
                w2 = ordered_cards[1]["weight"]
                w3 = ordered_cards[2]["weight"]

                comb_probability += (
                    w1 / total_weight
                    * w2 / (total_weight - w1)
                    * w3 / (total_weight - w1 - w2)
                )

            comb_expected_max = sum(
                max(score for score, prob in outcome_combo) * prod(prob for score, prob in outcome_combo)
                for outcome_combo in product(*(card["results"] for card in comb_cards))
            )

            expected += comb_probability * comb_expected_max

        return expected

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
