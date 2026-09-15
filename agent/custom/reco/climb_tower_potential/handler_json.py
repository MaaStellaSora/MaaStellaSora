import re
from typing import Any, Self

from .state import State, OwnedPotentials
from .data import MAX_POTENTIAL_LEVEL, Data, Potential
from .interactor import PotentialInteractor
from .handler_default import ChoosePotentialHandler

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential_json")


class AssistantPriorityHandler(ChoosePotentialHandler):
    HANDLER_TYPE = "json"

    def __init__(self, screen: PotentialInteractor, data: Data):
        super().__init__(screen, data)

    def read_potentials_info(self) -> Self:
        self.data.parsed_priority_list = self._parse_priority_raw_list(
            self.data.params.priority_list, State.owned_potentials
        )
        self.data.potentials = self.initialize_potentials()

        self._update_names()
        self._update_levels()

        return self

    def choose(self):
        # 获得所有潜能的排名
        self._update_priority()

        # 输出比较结果
        for potential in self.data.potentials:
            print_rank = potential.rank if potential.in_preset else "无"
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
            rank, sub_rank, trekker, in_preset = self._get_potential_priority(potential)
            potential.rank = rank
            potential.sub_rank = sub_rank
            potential.trekker.name = trekker
            potential.in_preset = in_preset
            # 是否已拥有：用于优先级相同时优先选择尚未拥有的潜能
            potential.in_owned = bool(
                in_preset
                and trekker
                and State.owned_potentials.find(
                    potential.name, mode="EXACT", trekker_name=trekker
                )
            )

    def _get_potential_priority(
        self,
        potential: Potential,
    ) -> tuple[int, int, str | None, bool]:
        """获取单个待选潜能的实际优先级及其 trekker 归属。

        遍历 priority_list，找到所有名称匹配且满足 level_span / max_level / refresh /
        no_enhance 等条件的规则，取实际优先级最小（即最优先）的那条规则。

        实际优先级 = 基础优先级 - (等级跨度 - 1)：
        一次跃升越多级的潜能越优先；标了 no_upgrade 的规则不做此提升。

        Args:
            potential: 单个待选潜能，结构：
                {"name": str, "old_level": int, "new_level": int, "box": list}

        Returns:
            tuple[int, int, str | None, bool]: (priority, sub_rank, trekker, in_preset)
                priority 为实际优先级，可为负数；无匹配时返回 -1
                sub_rank 为命中的 potential 名称在该规则 names 列表中的下标；无匹配时返回 -1
                trekker 为对应规则的归属角色；无匹配时返回空字符
                in_preset 为是否命中任何一条规则
        """
        priority_list = self.data.parsed_priority_list

        best_entry = None
        best_priority = None
        best_sub_rank = -1

        level_span = max(0, potential.level_span)

        for entry in priority_list:
            # 1. 匹配名称并获取副排名
            sub_rank = self._find_sub_rank(potential.name, entry["names"])
            if sub_rank == -1:
                continue

            # 2. 验证 level_span / max_level / refresh / no_enhance 等条件
            if not self._is_entry_valid(entry, potential):
                continue

            # 3. 计算实际优先级
            #    未配置 priority 的规则：沿用列表行号排名，不做等级跃升提升（与旧版行为完全一致）
            #    配置了 priority 的规则：实际优先级 = 基础优先级 - (等级跨度 - 1)，标了 no_upgrade 的除外
            base_priority = entry["priority"]
            if base_priority is None:
                actual_priority = entry["line_rank"]
            elif entry["no_upgrade"]:
                actual_priority = base_priority
            else:
                actual_priority = base_priority - (level_span - 1)

            # 4. 特殊机制：只在特定等级跨度/刷新次数下抓取，条件不满足则整条规则跳过
            special = entry["special"]
            if special and not self._check_special_rule(special, entry, potential):
                continue

            # 5. 取实际优先级最小（最优先）的规则
            if best_entry is None or actual_priority < best_priority:
                best_entry = entry
                best_priority = actual_priority
                best_sub_rank = sub_rank

        if best_entry is None:
            return -1, -1, "", False

        return best_priority, best_sub_rank, best_entry["trekker"], True

    def _check_special_rule(self, special: str, entry: dict, potential: Potential) -> bool:
        """判断特殊机制规则在当前潜能上是否成立。

        这类潜能的价值取决于「等级跨度」与「刷新次数」，并非无条件抓取；
        达到 special_cap_level 后直接放弃。

        Args:
            special: 特殊机制标记
            entry: 当前规则
            potential: 待选潜能

        Returns:
            bool: True 表示满足条件可参与选择；False 表示跳过该规则
        """
        old = potential.old_level
        span = potential.level_span
        cap = entry["special_cap_level"]

        # 已经抓到上限等级（或以上）就不再抓
        if cap is not None and old >= cap:
            return False

        if special == "dark_afterimage":
            # 暗·蜃影：0→1 的跃升时抓取，或刷新次数足够多时抓取
            return (old == 0 and span == 1) or self.data.refresh_count > 3
        if special == "salute_double":
            # 礼炮双响：仅 0→1 的跃升时抓取
            return old == 0 and span == 1
        if special == "mirror_current":
            # 镜水归潮：低等级起步的跃升（0→2 以内）时抓取
            return old == 0 and span <= 2
        return True

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

        # 强化阶段：标了 no_enhance 的潜能不参与选择
        if self.data.params.potential_source == "enhance" and entry["no_enhance"]:
            return False

        # 普通潜能，组合匹配规则
        checks = [
            potential.old_level < entry["max_level"],
            potential.level_span >= entry["level_span"],
            self.data.refresh_count >= entry["refresh"]
        ]
        return all(checks)

    @property
    def best_potential(self) -> Potential | None:
        """按实际优先级升序选最好的潜能；优先级相同时优先未拥有，再比跨度、副排名。

        只考虑命中优先级规则（in_preset）的潜能，实际优先级可为负数。
        """
        valid_potentials = (p for p in self.data.potentials if p.in_preset)
        return min(
            valid_potentials,
            # 优先级相同 → 未拥有的优先；再按跨度大、副排名小
            key=lambda p: (p.rank, p.in_owned, -p.level_span, p.sub_rank),
            default=None,
        )

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
                    "priority": int,        # 可选，基础优先级（越小越优先，可为负数）；
                                            #   不填则沿用列表行号排名，且不做等级跃升提升
                    "no_upgrade": bool,     # 可选，默认 False，为 True 时等级跃升不提升优先级
                    "no_enhance": bool,     # 可选，默认 False，为 True 时强化阶段不选该潜能
                    "special": str,         # 可选，特殊机制标记，见 _check_special_rule
                    "special_cap_level": int,     # 可选，特殊机制：达到该等级后不再抓取
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
                    "priority": int | None, # 基础优先级，越小越优先，可为负数；None 表示未配置
                    "line_rank": int,       # 未配置 priority 时使用的行号排名
                    "no_upgrade": bool,     # 等级跃升不提升优先级
                    "no_enhance": bool,     # 强化阶段不选该潜能
                    "special": str | None,  # 特殊机制标记
                    "special_cap_level": int | None,     # 特殊机制：达到该等级后不再抓取
                }
        """
        def _check_single_condition(item: dict) -> bool:
            """检查单个 condition 子项是否满足。"""
            if "count_at_least" in item or "count_at_most" in item:
                level_min = item.get("level_at_least")
                level_max = item.get("level_at_most")
                count = owned_potentials.count(
                    trekker_name=item["trekker"],
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
                trekker_name=item.get("trekker"),
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
                # 基础优先级（越小越优先，可为负）；缺省为 None，此时完全沿用行号排名
                "priority": raw.get("priority"),
                # 未配置 priority 时使用的行号排名
                "line_rank": index + 1,
                # 等级跃升时不按公式提升优先级
                "no_upgrade": raw.get("no_upgrade", False),
                # 强化时不选择该潜能
                "no_enhance": raw.get("no_enhance", False),
                # 特殊机制标记：dark_afterimage / salute_double / mirror_current
                "special": raw.get("special"),
                # 特殊机制：达到该等级后不再抓取
                "special_cap_level": raw.get("special_cap_level"),
            })

        return valid_entries
