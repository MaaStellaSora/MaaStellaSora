import re
from dataclasses import dataclass, field

import numpy

from custom.reco.climb_tower_potential.data import MAX_POTENTIAL_LEVEL, Potential
from utils import logger as logger_module
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

    def save(self, potential: Potential, *, fuzzy: bool = False) -> None:
        """将选中的潜能保存到OwnedPotentials中，如果已存在则更新等级和名字"""
        if not potential.name:
            return

        trekker = potential.trekker or "unknown"
        level = max(potential.new_level, 1)
        core = potential.core

        # 先查找潜能是否已存在，如果存在则更新等级和名字
        existed = self.find(potential.name, trekker=trekker, core=core, fuzzy=fuzzy)
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
        trekker: str | None = None,
        core: bool | None = None,
        fuzzy: bool = False,
    ) -> OwnedPotential | None:
        """查找是否已拥有该潜能"""
        # TODO: 增加difflib匹配
        for p in self.potentials:
            if trekker is not None and p.trekker != trekker:
                continue
            if core is not None and p.core != core:
                continue
            if self._match(p.name, name, fuzzy=fuzzy):
                return p
        return None

    def find_level(
        self,
        name: str,
        *,
        trekker: str | None = None,
        fuzzy: bool = False,
    ) -> int:
        """查找已拥有的某潜能的等级，如果不存在则返回0"""
        owned_potential = self.find(name, trekker=trekker, fuzzy=fuzzy)
        return owned_potential.level if owned_potential else 0

    def find_recommended_level(
        self,
        name: str,
        *,
        trekker: str | None = None,
        fuzzy: bool = False,
    ) -> int:
        """查找已拥有的某潜能的推荐等级，如果不存在则返回0"""
        owned_potential = self.find(name, trekker=trekker, fuzzy=fuzzy)
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
    def _match(a: str, b: str, *, fuzzy: bool) -> bool:
        """
        判断两个字符串是否匹配。
        Args:
            a: 第一个字符串
            b: 第二个字符串
            fuzzy: 是否开启模糊匹配（通过in方法匹配，只能处理前后有漏的情况）

        Returns:
            bool: 是否匹配成功
        """
        # 处理特殊符号
        a = re.sub(r"\W", "", a)
        b = re.sub(r"\W", "", b)

        # 精确匹配
        if not fuzzy:
            return a == b

        # 简单处理日语的长音符号，中文的一不处理是因为会出现重名
        a = a.replace("ー", "")
        b = b.replace("ー", "")

        # 通过in进行模糊匹配，需要排除空字符串然后再判断包含关系
        return bool(a and b and (a in b or b in a))

    @staticmethod
    def _longer_name(old: str, new: str) -> str:
        """一般来讲，更长的名称更接近原名"""
        return new if len(new) > len(old) else old


class State:
    failed_count: int = 0
    high_level_span_count: int = 0
    enhance_high_level_span_count: int = 0
    potentials_level_count: int = 0
    main_trekker: str = ""
    trekker_images: list[numpy.ndarray] = []
    owned_potentials: OwnedPotentials = OwnedPotentials()

    @classmethod
    def reset(cls):
        cls.failed_count = 0
        cls.high_level_span_count = 0
        cls.enhance_high_level_span_count = 0
        cls.potentials_level_count = 0
        cls.main_trekker = ""
        cls.trekker_images.clear()
        cls.owned_potentials = OwnedPotentials()
