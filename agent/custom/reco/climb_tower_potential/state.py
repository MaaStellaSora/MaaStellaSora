import re
import json
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
from typing import Literal
from dataclasses import dataclass, field

import numpy

from .data import MAX_POTENTIAL_LEVEL, Data, Potential

from utils import logger as logger_module
from utils.config import DRAW_DATA_SAVE_ENABLED
logger = logger_module.get_logger("climb_tower_potential_state")


@dataclass(slots=True)
class OwnedPotential:
    name: str
    level: int
    recommended_level: int
    trekker: str = "unknown"
    core: bool = False

    @property
    def max_level(self) -> int:
        """潜能等级上限"""
        return 1 if self.core else MAX_POTENTIAL_LEVEL

    @property
    def owned(self) -> bool:
        """是否已持有"""
        return self.level > 0

    @property
    def recommended(self) -> bool:
        """是否为推荐潜能"""
        return self.recommended_level > 0

@dataclass(slots=True)
class OwnedPotentials:
    potentials: list[OwnedPotential] = field(default_factory=list)

    def __iter__(self):
        return iter(self.potentials)

    def __len__(self):
        return len(self.potentials)

    def save(self, potential: Potential, *, handler: str) -> None:
        """将选中的潜能保存到OwnedPotentials中，如果已存在则更新等级和名字"""
        if not potential.name:
            return

        trekker = potential.trekker or "unknown"
        level = max(potential.new_level, 1)
        core = potential.core

        # 先查找潜能是否已存在
        if handler == "json":
            existed = self.find(potential.name, mode="EXACT", trekker=trekker, core=core)
        else:
            if potential.old_level >= 1:
                existed = self.find(potential.name, mode="FUZZY", trekker=trekker, core=core, threshold=0.75)
            elif potential.old_level == -1:
                existed = self.find(potential.name, mode="CONTAINS", trekker=trekker, core=core)
            else:
                existed = None
        # 如果存在则更新等级和名字
        if existed:
            # 防止 OCR 识别失败导致等级倒退，所以至少按照原等级+1处理
            existed.level = min(existed.max_level, max(existed.level + 1, level))
            existed.recommended_level = max(existed.recommended_level, potential.recommended_level)
            # 使用更长的名字，因为更长的名字更接近原名
            existed.name = self._longer_name(existed.name, potential.name)
            return

        # 不存在，就添加到list中
        self.potentials.append(OwnedPotential(
            name=potential.name,
            level=level,
            recommended_level=potential.recommended_level,
            trekker=trekker,
            core=core,
        ))

    def find(
        self,
        name: str,
        *,
        mode: Literal["EXACT", "CONTAINS", "FUZZY"],
        trekker: str | None = None,
        core: bool | None = None,
        threshold: float = 0
    ) -> OwnedPotential | None:
        """
        查找是否已拥有该潜能

        Args:
            name: 潜能名称
            mode: 模式， "EXACT"、"CONTAINS" 或 "FUZZY"
                "EXACT"：精确匹配，只有当名称完全匹配时才返回潜能
                "CONTAINS"：包含匹配，当名称有一方包含另一方时返回潜能
                "FUZZY"：模糊匹配，返回相似度最高的潜能，支持通过阈值限制
            trekker: 指定旅人名称（这里的旅人是代码内部名称，不一定是旅人真实名称）
            core: 指定是否为核心潜能
            threshold: 指定相似度阈值（仅对"FUZZY"模式有效）

        Returns:
            OwnedPotential | None: 如果找到则返回该潜能，否则返回None
        """
        # 模糊匹配
        if mode == "FUZZY":
            results = max(
                (
                    (score, p) for p in self.potentials
                    if (score := self._fuzzy_match(name, p.name)) >= threshold
                ),
                key=lambda p: p[0],
                default=None
            )
            return results[1] if results else None

        # 普通匹配
        contains_mode = mode == "CONTAINS"
        for p in self.potentials:
            if trekker is not None and p.trekker != trekker:
                continue
            if core is not None and p.core != core:
                continue
            if self._match(p.name, name, contains_mode=contains_mode):
                return p
        return None

    def find_level(
        self,
        name: str,
        *,
        mode: Literal["EXACT", "CONTAINS", "FUZZY"],
        trekker: str | None = None,
        fuzzy_threshold: float = 0,
    ) -> int:
        """查找已拥有的某潜能的等级，如果不存在则返回0"""
        owned_potential = self.find(name, mode=mode, trekker=trekker, threshold=fuzzy_threshold)
        return owned_potential.level if owned_potential else 0

    def find_recommended_level(
        self,
        name: str,
        *,
        mode: Literal["EXACT", "CONTAINS", "FUZZY"],
        trekker: str | None = None,
        fuzzy_threshold: float = 0,
    ) -> int:
        """查找已拥有的某潜能的推荐等级，如果不存在则返回0"""
        owned_potential = self.find(name, mode=mode, trekker=trekker, threshold=fuzzy_threshold)
        return owned_potential.recommended_level if owned_potential else 0

    def count(
        self,
        *,
        trekker: str | list = None,
        level_at_least: int | None = None,
        level_at_most: int | None = None,
        recommended_level_at_least: int | None = None,
        recommended_level_at_most: int | None = None,
        include_core: bool = False,
        incomplete_only: bool = False,
        leveling_only: bool = False
    ) -> int:
        """统计符合条件的潜能的数量"""
        if isinstance(trekker, str):
            trekker = (trekker,)

        return sum(
            1 for potential in self.potentials
            if (trekker is None or potential.trekker in trekker)
            and (level_at_least is None or potential.level >= level_at_least)
            and (level_at_most is None or potential.level <= level_at_most)
            and (recommended_level_at_least is None or potential.recommended_level >= recommended_level_at_least)
            and (recommended_level_at_most is None or potential.recommended_level <= recommended_level_at_most)
            and (include_core or not potential.core)
            and (not incomplete_only or potential.level < potential.recommended_level)
            and (not leveling_only or potential.level < potential.max_level)
        )

    @staticmethod
    def _match(a: str, b: str, *, contains_mode: bool) -> bool:
        """
        判断两个字符串是否匹配。
        Args:
            a: 第一个字符串
            b: 第二个字符串
            contains_mode: 是否使用包含匹配模式

        Returns:
            bool: 是否匹配成功
        """
        # 处理特殊符号
        a = re.sub(r"\W", "", a)
        b = re.sub(r"\W", "", b)

        # 精确匹配
        if not contains_mode:
            return a == b

        # 简单处理日语的长音符号，中文的一不处理是因为会出现重名
        a = a.replace("ー", "")
        b = b.replace("ー", "")

        # 通过in进行包含匹配，需要排除空字符串然后再判断包含关系
        return bool(a and b and (a in b or b in a))

    @staticmethod
    def _fuzzy_match(a: str, b: str) -> float:
        """
        计算两个字符串的相似度。
        Args:
            a: 第一个字符串
            b: 第二个字符串

        Returns:
            float: 字符串相似度，范围为[0, 1]
        """
        # 处理特殊符号
        a = re.sub(r"\W", "", a)
        b = re.sub(r"\W", "", b)
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _longer_name(old: str, new: str) -> str:
        """一般来讲，更长的名称更接近原名"""
        return new if len(new) > len(old) else old

@dataclass
class PotentialDrawInfo:
    potential_draws: list[dict] = field(default_factory=list)

    def add(self, data: Data) -> None:
        draws = [
            {
                "name": p.name,
                "trekker": p.trekker,
                "old_level": p.old_level,
                "new_level": p.new_level,
                "recommended_level": p.recommended_level,
            }
            for p in data.potentials
        ]
        owned = [
            {
                "name": p.name,
                "trekker": p.trekker,
                "level": p.level,
                "recommended_level": p.recommended_level,
            }
            for p in State.owned_potentials.potentials if not p.core
        ]
        self.potential_draws.append({
            "draws": draws,
            "owned": owned,
            "trigger_type": data.params.trigger_type,
            "high_level_span_count": State.high_level_span_count,
            "enhance_high_level_span_count": State.enhance_high_level_span_count,
        })

    def export(self) -> None:
        """导出潜能抽取数据到项目根目录下的debug/potential_draws/日期时间.json"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # %f是微秒，取前3位得到毫秒
        filename = f"{timestamp}.json"
        file_path = Path(__file__).resolve().parents[4] / "debug" / "potential_draws" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        json.dump(self.potential_draws, open(file_path, "w"), ensure_ascii=False, indent=4)
        logger.info(f"已导出潜能抽取数据到 {file_path}")

    @property
    def available(self) -> bool:
        return len(self.potential_draws) > 0

class State:
    high_level_span_count: int = 0
    enhance_high_level_span_count: int = 0
    potentials_level_count: int = 0
    main_trekker: str = ""
    trekker_images: list[numpy.ndarray] = []
    owned_potentials: OwnedPotentials = OwnedPotentials()
    potential_draw_info: PotentialDrawInfo = PotentialDrawInfo()

    @classmethod
    def reset(cls):
        cls.high_level_span_count = 0
        cls.enhance_high_level_span_count = 0
        cls.potentials_level_count = 0
        cls.main_trekker = ""
        cls.trekker_images.clear()
        cls.owned_potentials = OwnedPotentials()
        if cls.potential_draw_info.available and DRAW_DATA_SAVE_ENABLED:
            cls.potential_draw_info.export()
            cls.potential_draw_info = PotentialDrawInfo()
