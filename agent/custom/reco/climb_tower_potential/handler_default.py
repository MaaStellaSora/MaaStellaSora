import time
from typing import Self
from dataclasses import replace

from .data import Data, Potential
from .interactor import PotentialInteractor
from .state import State, Trekker

from utils import logger as logger_module
from utils.config import DEV_IMAGES_SAVE_ENABLED
logger = logger_module.get_logger("climb_tower_potential_default")


class ChoosePotentialHandler:
    HANDLER_TYPE = "default"

    def __init__(self, screen: PotentialInteractor, data: Data):
        self.screen = screen
        self.data = data

    def wait_for_item_list_gone(self):
        for _ in range(5):
            if self.screen.check_item_list_visibility():
                logger.info("识别到干扰文字，等待1秒")
                time.sleep(1)
                self.screen.screenshot()
                continue
            break

    def initialize_potentials(self):
        # 初始化Potential对象，并保存到list中
        potential_layouts = self.data.params.potential_layouts[self.data.potential_count]
        potentials = [Potential(potential_layouts[i]) for i in range(self.data.potential_count)]

        # 给潜能的selected、type字段赋值
        self.data.selected_potential_index = self.screen.get_selected_potential_index(self.data.x_borders)
        for i, p in enumerate(potentials):
            p.index = i
            p.type = self.data.potential_types[i]
            if i == self.data.selected_potential_index:
                p.selected = True
            if p.core:
                p.old_level = 0
                p.new_level = 1

        return potentials

    def read_potentials_info(self) -> Self:
        """最原始的潜能信息识别器，仅识别推荐图标"""
        self.data.potentials = self.initialize_potentials()
        self._update_recommended_potentials()

        return self

    def choose(self) -> Potential | None:
        """最原始的潜能选择，仅靠推荐图标选择潜能"""
        potential = next((p for p in self.data.potentials if p.recommended), None)
        if potential:
            logger.info(f"[潜能选择] 推荐潜能")
        return potential

    def pick(self, potential: Potential) -> bool:
        """点击潜能卡片"""
        if potential.selected:
            return True
        return self.screen.click_potential(potential.box)

    def dummy_potential(self, **kwargs) -> Potential:
        """返回一个虚拟的潜能对象"""
        potential_layout = self.data.params.potential_layouts[1][0]
        base_potential = Potential(layout=potential_layout)
        return replace(base_potential, **kwargs)

    def _update_names(self):
        rois = self.data.core_potential_name_rois if self.data.core_potential else self.data.general_potential_name_rois
        adjusted_rois = self._get_adjusted_rois(rois)

        for i, roi in enumerate(adjusted_rois):
            self.data.potentials[i].name = self.screen.get_potential_name(roi)

    def _update_levels(self):
        if self.data.core_potential:
            return

        adjusted_rois = self._get_adjusted_rois(self.data.general_potential_level_rois)

        for i, roi in enumerate(adjusted_rois):
            old, new = self.screen.get_potential_level(roi)
            self.data.potentials[i].old_level, self.data.potentials[i].new_level = old, new

            if DEV_IMAGES_SAVE_ENABLED and old == -1 and new == -1:
                from utils.image_handler import save_image
                save_image(self.screen.image, f"第{i}个潜能等级识别失败_{roi}")

    def _update_recommended_potentials(self):
        adjusted_rois = self._get_adjusted_rois(self.data.recommended_level_rois)
        indices = self.screen.get_recommended_potential(self.data.x_borders)
        for index in indices:
            roi = adjusted_rois[index]
            self.data.potentials[index].recommended = True
            if self.data.core_potential:
                self.data.potentials[index].recommended_level = 1
            else:
                self.data.potentials[index].recommended_level = self.screen.get_recommend_level(roi)

    def _update_trekkers(self):
        save_rois = self._get_adjusted_rois(self.data.trekker_rois)
        expanded_rois = self._expand_rois(save_rois)

        for potential_i, roi in enumerate(expanded_rois):
            # 先根据现有旅人信息匹配，如果匹配到对应旅人，给潜能赋值对应的trekker对象
            for trekker in State.trekkers:
                if self.screen.match_trekker(trekker.image, roi):
                    self.data.potentials[potential_i].trekker = trekker
                    break

            # 匹配不到时，创建新的Trekker实例，保存到State中，然后再赋值
            if not self.data.potentials[potential_i].trekker:
                cropped_image = self.screen.crop_screenshot(save_rois[potential_i])
                new_trekker = Trekker(index=len(State.trekkers), image=cropped_image)
                State.trekkers.append(new_trekker)
                self.data.potentials[potential_i].trekker = new_trekker

            # 给主控旅人做标记
            if not State.get_main_trekker() and self.data.params.potential_source == "specified_drink":
                matched_trekker = self.data.potentials[potential_i].trekker
                matched_trekker.main = True
                logger.debug("已通过特殊潜能特饮识别到主旅人")

            # 如果trekker超过3个，输出错误日志
            if len(State.trekkers) > 3:
                logger.error("识别到超过3种旅人，后续潜能判断将会出现问题")

    def _get_adjusted_rois(self, base_rois: list[list[int]]) -> list[list[int]]:
        """安全地获取偏移后的 ROI 副本，避免污染原始数据"""
        selected_index = self.data.selected_potential_index
        offset = self.data.params.selected_potential_offset

        # 使用列表推导式创建副本并应用偏移
        return [
            [r[0], max(0, r[1] - offset), r[2], r[3]] if i == selected_index else list(r)
            for i, r in enumerate(base_rois)
        ]

    @staticmethod
    def _expand_rois(base_rois: list[list[int]], px: int = 15) -> list[list[int]]:
        """扩展ROI"""
        return [
            [r[0] - px, r[1] - px, r[2] + px * 2, r[3] + px * 2]
            for r in base_rois
        ]

    def choose_fallback_potential(self):
        priority_rules = [
            lambda p: p.recommended, # 1. 尝试取系统推荐潜能
            lambda p: p.old_level > 0 # 2. 没有系统推荐潜能，选择之前抓过的潜能
        ]

        for rule in priority_rules:
            candidates = [p for p in self.data.potentials if rule(p)]
            if candidates:
                # 按照等级跨度降序、推荐等级降序、旧等级降序来排序，选择最优的潜能
                return max(candidates, key=lambda p: (p.level_span, p.recommended_level, p.old_level), default=None)

        # 都没有的话，放弃选择，系统选了哪张就哪张
        return self._default_potential

    def refresh(self):
        self.screen.refresh()
        self.data.refresh_count += 1

    @property
    def _default_potential(self):
        potential = next(p for p in self.data.potentials if p.selected)
        return potential
