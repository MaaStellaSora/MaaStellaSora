from __future__ import annotations

from dataclasses import dataclass, field

from .ui import UIInteractor


MAX_POTENTIAL_LEVEL: int = 6  # 潜能等级上限，condition max_level 字段的默认值

@dataclass(slots=True, frozen=True)
class Parameters:
    potential_source: str
    max_refresh_count: int
    reserved_coin: int
    priority_list: list[dict]
    handler: str
    environment: str
    threshold_coef_str: str
    threshold_decay_str: str
    # 自带默认值的参数
    potential_layouts: PotentialLayouts = field(default_factory=lambda: PotentialLayouts())
    selected_potential_offset: int = 35

    @property
    def threshold_coef(self) -> float:
        return float(self.threshold_coef_str)

    @property
    def threshold_decay(self) -> float:
        return float(self.threshold_decay_str)

DEFAULT_POTENTIAL_LAYOUTS = {
    1: [
        {
            "core_potential_roi": [530, 405, 220, 60],
            "general_potential_roi": [530, 375, 220, 60],
            "general_potential_level_roi": [530, 425, 220, 40],
            "recommended_level_roi": [670, 165, 140, 50],
            "potential_roi": [470, 0, 343, 720],
            "trekker_roi": [500, 182, 40, 40],
            "x_border": [470, 813]
        }
    ],
    2: [
        {
            "core_potential_roi": [358, 405, 220, 60],
            "general_potential_roi": [358, 375, 220, 60],
            "general_potential_level_roi": [358, 425, 220, 40],
            "recommended_level_roi": [490, 165, 140, 50],
            "potential_roi": [0, 0, 639, 720],
            "trekker_roi": [329, 182, 40, 40],
            "x_border": [0, 639]
        },
        {
            "core_potential_roi": [703, 405, 220, 60],
            "general_potential_roi": [703, 375, 220, 60],
            "general_potential_level_roi": [703, 425, 220, 40],
            "recommended_level_roi": [840, 165, 140, 50],
            "potential_roi": [640, 0, 640, 720],
            "trekker_roi": [673, 182, 40, 40],
            "x_border": [640, 1280]
        }
    ],
    3: [
        {
            "core_potential_roi": [187, 405, 220, 60],
            "general_potential_roi": [187, 375, 220, 60],
            "general_potential_level_roi": [187, 425, 220, 40],
            "recommended_level_roi":[320, 165, 140, 50],
            "potential_roi": [0, 0, 469, 720],
            "trekker_roi": [156, 182, 40, 40],
            "x_border": [0, 469]
        },
        {
            "core_potential_roi": [530, 405, 220, 60],
            "general_potential_roi": [530, 375, 220, 60],
            "general_potential_level_roi": [530, 425, 220, 40],
            "recommended_level_roi": [670, 165, 140, 50],
            "potential_roi": [470, 0, 343, 720],
            "trekker_roi": [500, 182, 40, 40],
            "x_border": [470, 813]
        },
        {
            "core_potential_roi": [875, 405, 220, 60],
            "general_potential_roi": [875, 375, 220, 60],
            "general_potential_level_roi": [875, 425, 220, 40],
            "recommended_level_roi":[1010, 165, 140, 50],
            "potential_roi": [814, 0, 466, 720],
            "trekker_roi": [844, 182, 40, 40],
            "x_border": [814, 1280]
        }
    ]
}

@dataclass(slots=True)
class PotentialLayout:
    core_potential_roi: list[int]
    general_potential_roi: list[int]
    general_potential_level_roi: list[int]
    recommended_level_roi: list[int]
    potential_roi: list[int]
    trekker_roi: list[int]
    x_border: list[int]

@dataclass(slots=True)
class PotentialLayouts:
    potential_layouts: dict[int, list[PotentialLayout]] = field(default_factory=lambda: {
        count: [PotentialLayout(**l) for l in layouts]
        for count, layouts in DEFAULT_POTENTIAL_LAYOUTS.items()
    })

    def __getitem__(self, key):
        return self.potential_layouts[key]

    def __iter__(self):
        return iter(self.potential_layouts)

    def items(self):
        return self.potential_layouts.items()

    def get(self, key, default=None):
        return self.potential_layouts.get(key, default)


# endregion 常量 ===============================================================

# region 统筹所有数据的类 ===============================================================

@dataclass(slots=True)
class Data:
    params: Parameters
    current_coin: int = 0
    refresh_cost: int = 0
    potential_count: int = 0
    core_potential: bool = False
    # 不需要根据刷新更新的数据
    threshold: float = -1.0 # 刷新阈值储存变量
    level_upped: bool = False # 是否通过旅人升级获得潜能
    # 需要根据刷新更新的数据
    selected_potential_index: int = 1
    potentials: list[Potential] = field(default_factory=lambda: [])
    refresh_count: int = 0 # 刷新次数
    parsed_priority_list: list[dict] = field(default_factory=lambda: []) # json匹配用数据

    @property
    def refresh_botton(self) -> bool:
        return self.refresh_cost >= 0

    @property
    def refreshable(self) -> bool:
        return self.refresh_count < self.refresh_limit

    @property
    def refresh_limit(self) -> int:
        usable_coin = max(0, self.current_coin - self.params.reserved_coin)
        affordable = usable_coin // self.refresh_cost if self.refresh_botton else 0
        return min(self.params.max_refresh_count, affordable)

    @property
    def potential_rois(self) -> list[list[int]]:
        return [l.potential_roi for l in self.params.potential_layouts[self.potential_count]]

    @property
    def x_borders(self) -> list[list[int]]:
        return [l.x_border for l in self.params.potential_layouts[self.potential_count]]

    @property
    def core_potential_name_rois(self) -> list[list[int]]:
        return [l.core_potential_roi for l in self.params.potential_layouts[self.potential_count]]

    @property
    def general_potential_name_rois(self) -> list[list[int]]:
        return [l.general_potential_roi for l in self.params.potential_layouts[self.potential_count]]

    @property
    def general_potential_level_rois(self) -> list[list[int]]:
        return [l.general_potential_level_roi for l in self.params.potential_layouts[self.potential_count]]

    @property
    def recommended_level_rois(self) -> list[list[int]]:
        return [l.recommended_level_roi for l in self.params.potential_layouts[self.potential_count]]

    @property
    def trekker_rois(self) -> list[list[int]]:
        return [l.trekker_roi for l in self.params.potential_layouts[self.potential_count]]

@dataclass(slots=True)
class Potential:
    layout: PotentialLayout
    index: int = -1
    core: bool = False
    name: str = ""
    old_level: int = -1
    new_level: int = -1
    recommended: bool = False
    recommended_level: int = -1
    trekker: str = ""
    selected: bool = False
    # 自定义参数
    rank: int = -1
    sub_rank: int = -1
    score: int = 0

    @property
    def level_span(self) -> int:
        return self.new_level - self.old_level

    @property
    def box(self) -> list[int]:
        return self.layout.general_potential_roi

    @property
    def potential_roi(self) -> list[int]:
        return self.layout.potential_roi

    @property
    def x_border(self) -> list[int]:
        return self.layout.x_border

    @property
    def core_potential_name_roi(self) -> list[int]:
        return self.layout.core_potential_roi

    @property
    def general_potential_name_roi(self) -> list[int]:
        return self.layout.general_potential_roi

    @property
    def general_potential_level_roi(self) -> list[int]:
        return self.layout.general_potential_level_roi

    @property
    def recommended_level_roi(self) -> list[int]:
        return self.layout.recommended_level_roi

    def update(self, screen: UIInteractor, data: Data):
        # 更新核心潜能
        if data.core_potential:
            self.core = True

        # 更新潜能数据
        self.name = self._get_name(screen, data)
        self.old_level, self.new_level = self._get_level(screen, data)
        self.recommended, self.recommended_level = self._get_recommended_data(screen, data)

    def _get_name(self, screen: UIInteractor, data: Data) -> str:
        roi = self.core_potential_name_roi if self.core else self.general_potential_name_roi
        adjusted_roi = self._get_adjusted_roi(roi, data.params.selected_potential_offset)
        return screen.get_potential_name(adjusted_roi)

    def _get_level(self, screen: UIInteractor, data: Data) -> tuple[int, int]:
        if self.core:
            return 0, 1
        adjusted_roi = self._get_adjusted_roi(self.general_potential_level_roi, data.params.selected_potential_offset)
        old, new = screen.get_potential_level(adjusted_roi)
        return old, new

    def _get_recommended_data(self, screen: UIInteractor, data: Data) -> tuple[bool, int]:
        roi = self.potential_roi
        adjusted_roi = [self._get_adjusted_roi(roi, data.params.selected_potential_offset)]
        recommended = True if screen.check_potential_recommended(adjusted_roi) else False
        if not recommended:
            return False, 0
        if recommended and data.core_potential:
            return True, 1
        roi = self.recommended_level_roi
        adjusted_roi = self._get_adjusted_roi(roi, data.params.selected_potential_offset)
        level = screen.get_recommend_level(adjusted_roi)
        return recommended, level

    def _get_adjusted_roi(self, roi, offset) -> list[int]:
            return [roi[0], max(0, roi[1] - offset) if self.selected else roi[1], roi[2], roi[3]]
