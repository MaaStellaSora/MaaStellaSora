import re
from typing import Any, Self

from custom.reco.climb_tower_potential.state import State, OwnedPotentials
from custom.reco.climb_tower_potential.data import MAX_POTENTIAL_LEVEL, Data, Potential
from custom.reco.climb_tower_potential.ui import UIInteractor
from custom.reco.climb_tower_potential.handler_default import ChoosePotentialHandler

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential_json")


class AssistantPriorityHandler(ChoosePotentialHandler):
    def __init__(self, screen: UIInteractor, data: Data):
        super().__init__(screen, data)

    def read_potentials_info(self) -> Self:
        self.data.parsed_priority_list = self._parse_priority_raw_list(
            self.data.params.priority_list, State.owned_potentials
        )
        self._wait_for_item_list_gone()
        self.data.potentials = self.initialize_potentials()

        self._update_names()
        self._update_levels()

        return self

    def choose(self):
        # 获得所有潜能的排名
        self._update_priority()

        # 输出比较结果
        for potential in self.data.potentials:
            print_rank = potential.rank + 1 if potential.rank >= 0 else "无"
            if self.data.core_potential:
                logger.info(f"[潜能识别] {potential.name} | 核心潜能 | 排名 {print_rank}")
            else:
                old = potential.old_level
                new = potential.new_level
                logger.info(f"[潜能识别] {potential.name} | 等级 {old}→{new} | 排名 {print_rank}")

        # 选择排名最高的潜能
        best_potential = self.best_potential
        if best_potential:
            logger.info(f"[潜能选择] {best_potential.name}")

        return best_potential

    def _update_priority(self):
        for potential in self.data.potentials:
            rank, sub_rank, trekker = self._get_potential_priority(potential)
            potential.rank = rank
            potential.sub_rank = sub_rank
            potential.trekker = trekker

    def _get_potential_priority(
        self,
        potential: Potential,
    ) -> tuple[int, int, str | None]:
        """获取单个待选潜能在规则列表中的最高排名及其 trekker 归属。

        遍历 priority_list，找到所有名称匹配且满足 level_span / max_level / refresh
        条件的规则，返回排名数值最小（即优先级最高）的规则对应的排名与 trekker。

        Args:
            potential: 单个待选潜能，结构：
                {"name": str, "old_level": int, "new_level": int, "box": list}

        Returns:
            tuple[int, int, str | None]: (rank, sub_rank, trekker)
                rank 为匹配到的最小排名数值； rank 从 0 开始。无匹配时返回 -1
                sub_rank 为命中的 potential 名称在该规则 names 列表中的下标；无匹配时返回 -1
                trekker 为对应规则的归属角色；无匹配时返回空字符
        """
        priority_list = self.data.parsed_priority_list

        best_entry = None
        best_rank = -1
        best_sub_rank = -1

        for rank, entry in enumerate(priority_list):
            # 1. 基础剪枝：优先级如果不更高，直接跳过
            if best_entry and entry["priority"] >= best_entry["priority"]:
                continue

            # 2. 匹配名称并获取优先级排名
            sub_rank = self._find_sub_rank(potential.name, entry["names"])
            if sub_rank == -1:
                continue

            # 3. 验证其他规则是否通过
            if not self._is_entry_valid(entry, potential):
                continue

            # 全部通过后，记录该行及副等级
            best_entry = entry
            best_rank = rank
            best_sub_rank = sub_rank

        if not best_entry:
            best_entry = {"trekker": ""}

        return best_rank, best_sub_rank, best_entry["trekker"]

    def _find_sub_rank(self, name: str, rule_names: list[str]) -> int:
        """通过潜能名称获取最优排名数值"""
        return next((i for i, r in enumerate(rule_names)
                     if self._match_potential_name(name, r)), -1)

    @staticmethod
    def _match_potential_name(ocr_name: str, rule_name: str) -> bool:
        """比较 OCR 识别的潜能名称与规则中的潜能名称是否匹配。

        OCR 可能产生前后漏字或噪声字符，因此先清洗两边字符串（去除非 Unicode
        字母数字字符），再用 in 检查，覆盖前后漏字的情况。

        中间漏字或错字无法处理，属于 OCR 识别质量问题。

        Args:
            ocr_name: OCR 识别到的潜能名称
            rule_name: 优先级规则中定义的潜能名称

        Returns:
            bool: 两者匹配时返回 True
        """
        cleaned_ocr = re.sub(r'\W', '', ocr_name)
        cleaned_rule = re.sub(r'\W', '', rule_name)

        # 防止"" in "任意字符串" 返回True
        if not cleaned_ocr or not cleaned_rule:
            return False

        chars_to_remove = "ー"
        table = str.maketrans("", "", chars_to_remove)
        cleaned_ocr = cleaned_ocr.translate(table)
        cleaned_rule = cleaned_rule.translate(table)

        return cleaned_ocr in cleaned_rule

    def _is_entry_valid(self, entry: dict, potential: Potential) -> bool:
        """业务规则过滤器：方便未来随意扩展判定条件"""
        # 核心潜能默认全部通过
        if self.data.core_potential:
            return True

        # 普通潜能，组合匹配规则
        checks = [
            potential.old_level < entry["max_level"],
            potential.level_span >= entry["level_span"],
            self.data.refresh_count >= entry["refresh"]
        ]
        return all(checks)

    @property
    def best_potential(self) -> Potential | None:
        """按照排名升序、等级跨度降序、副排名升序三个维度，筛选出最好的潜能"""
        valid_potentials = (p for p in self.data.potentials if p.rank >= 0)
        return min(valid_potentials, key=lambda p: (p.rank, -p.level_span, p.sub_rank), default=None)

    @staticmethod
    def _parse_priority_raw_list(
        potential_priority_raw: list[dict],
        owned_potentials: OwnedPotentials,
    ) -> list[dict[str, Any]]:
        """对原始 priority_list 进行初筛，过滤掉 condition 当前不满足的规则。

        排名使用原始 JSON 的 1-based 行号（index + 1），数值越小排名越高。
        condition 不满足的条目直接跳过，其行号仍保留在原始位置，不影响其他条目的排名。

        Args:
            potential_priority_raw: 原始优先级列表，每个元素结构：
                {
                    "trekker": str,         # 可选，潜能归属角色名
                    "potential": str|list,  # 必填，潜能名称或名称列表
                    "level_span": int,      # 可选，默认 1，最小升级跨度
                    "max_level": int,       # 可选，默认 MAX_POTENTIAL_LEVEL，旧等级上限（不含）
                    "refresh": int,         # 可选，默认 0，已刷新次数必须 >= 该值规则才生效
                    "condition": list       # 可选，生效条件，元素为 dict 时 AND，为 list 时 OR
                }
            owned_potentials: 已拥有潜能状态。

        Returns:
            list[dict]: 通过初筛的规则列表，每个元素结构：
                {
                    "trekker": str | None,  # 归属角色
                    "names": list[str],     # 目标潜能名称列表
                    "level_span": int,      # 最小升级跨度
                    "max_level": int,       # 旧等级上限（不含）
                    "refresh": int,         # 已刷新次数下限
                    "priority": int         # 原始 JSON 的 1-based 行号，越小排名越高
                }
        """
        def _check_single_condition(item: dict) -> bool:
            """检查单个 condition 子项是否满足。"""
            if "count_at_least" in item or "count_at_most" in item:
                level_min = item.get("level_at_least")
                level_max = item.get("level_at_most")
                count = owned_potentials.count(
                    trekker=item["trekker"],
                    level_at_least=level_min,
                    level_at_most=level_max,
                )
                if "count_at_least" in item and count < item["count_at_least"]:
                    return False
                if "count_at_most" in item and count > item["count_at_most"]:
                    return False
                return True
            current = owned_potentials.find_level(
                item["potential"],
                mode="EXACT",
                trekker=item.get("trekker"),
            )
            min_ok = current >= item["level_at_least"] if "level_at_least" in item else True
            max_ok = current <= item["level_at_most"] if "level_at_most" in item else True
            return min_ok and max_ok

        def _check_condition(cond: list) -> bool:
            """检查 condition 列表是否满足。

            元素全为 dict 时为 AND 逻辑；含 list 元素时为 OR 逻辑（内层为 AND）。
            """
            if not cond:
                return True
            if all(isinstance(item, dict) for item in cond):
                return all(_check_single_condition(item) for item in cond)
            for branch in cond:
                if isinstance(branch, list):
                    if all(_check_single_condition(item) for item in branch):
                        return True
                elif isinstance(branch, dict):
                    if _check_single_condition(branch):
                        return True
            return False

        valid_entries = []
        for index, raw in enumerate(potential_priority_raw):
            condition = raw.get("condition", [])
            if not isinstance(condition, list):
                continue
            if not _check_condition(condition):
                continue

            potential = raw["potential"]
            names = potential if isinstance(potential, list) else [potential]

            valid_entries.append({
                "trekker": raw.get("trekker"),
                "names": names,
                "level_span": raw.get("level_span", 1),
                "max_level": raw.get("max_level", MAX_POTENTIAL_LEVEL),
                "refresh": raw.get("refresh", 0),
                "priority": index + 1,
            })

        return valid_entries

