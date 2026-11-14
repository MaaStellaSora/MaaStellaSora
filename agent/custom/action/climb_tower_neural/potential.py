"""Value decisions using Maa's original potential recognition/click loop."""
from dataclasses import dataclass

import numpy as np
from custom.reco.climb_tower_potential.handler_preset import RecommendationHandler
from custom.reco.climb_tower_potential.context import Potential, Trekker
from custom.reco.climb_tower_potential.handler_default import ChoosePotentialHandler

from .decision_log import log_forecast
from .name_matching import clean_name
from .observation import PotentialObservationIncomplete, potential_levels_valid
from .value_model import Action, sim
from .ui_flow import read_until, wait_for
from .pages import BAG_RETURN_PAGES, BY_NAME, classify
from .potential_reading import NeuralPotentialContext as PotentialContext
from utils import logger as logger_module


logger = logger_module.get_logger(__name__)


@dataclass
class PendingPotentialTake:
    potential: object
    selected_offset: int
    layouts: object
    tracker: object
    page_signature: tuple
    page_key: tuple | None = None
    commit_inventory: object = None
    committed: bool = False
    data: object = None
    evidence: dict | None = None


@dataclass
class PotentialTakePage:
    signature: tuple | None = None
    selected_index: int = -1
    button: list | None = None
    visible: bool = False


def _card_signature(item, rarity, old_level, new_level):
    return item["identity"], item["role"], rarity, old_level, new_level


def _trekker_roi(potential, selected_offset):
    roi = list(potential.layout.trekker_roi)
    if potential.selected:
        roi[1] -= selected_offset
    return roi


def _resolve_potential_items(screen, tracker, potentials, selected_offset, *, single_role=False, pending=None):
    """Resolve shared names from this page, held levels and this run's portraits."""
    candidates = [tracker.name_candidates(p.name, rarity=p.rarity) for p in potentials]
    if all(len(items) == 1 for items in candidates):
        return [items[0] for items in candidates]

    def held_levels(item):
        key = item["role"], item["identity"]
        levels = {tracker.identity_levels.get(key, 0)}
        # During Take confirmation, the next page may already include the
        # selected level although its inventory commit is still pending.
        if pending is not None:
            identity, role, _, _, new_level = pending.page_signature[pending.potential.index]
            if key == (role, identity):
                levels.add(new_level)
        return levels

    def same_role():
        if single_role:
            roles = set.intersection(*({item["role"] for item in items} for items in candidates))
            for items in candidates:
                items[:] = [item for item in items if item["role"] in roles]
        if any(not items for items in candidates):
            raise PotentialObservationIncomplete("本帧潜能名称、角色或持有等级尚未对应")

    same_role()
    for potential, items in zip(potentials, candidates):
        if len(items) > 1 and not potential.core and potential.old_level >= 0:
            items[:] = [item for item in items if potential.old_level in held_levels(item)]
    same_role()
    if any(len(items) > 1 for items in candidates):
        portraits = {role: trekker.image for role, trekker in tracker.trekkers.items() if trekker.image.size}
        # A distinct card on a mixed enhancement page can also bind its
        # portrait for another card in the same frame, without changing state.
        for potential, items in zip(potentials, candidates):
            if len(items) == 1 and (potential.core or potential.old_level in held_levels(items[0])):
                role = items[0]["role"]
                if role not in portraits:
                    portraits[role] = screen.crop_screenshot(_trekker_roi(potential, selected_offset)).copy()
        for potential, items in zip(potentials, candidates):
            if len(items) == 1:
                continue
            x, y, width, height = _trekker_roi(potential, selected_offset)
            roi = [x - 15, y - 15, width + 30, height + 30]
            roles = {item["role"] for item in items}
            matched = {role for role in roles if role in portraits and screen.match_trekker(portraits[role], roi)}
            if len(matched) == 1:
                items[:] = [item for item in items if item["role"] in matched]
                same_role()
    if any(len(items) != 1 for items in candidates):
        raise PotentialObservationIncomplete("本帧同名潜能的角色尚未确认")
    return [items[0] for items in candidates]


def _needs_trekker_portrait(tracker, role):
    return any(item["role"] == role and any(other["role"] != role
        and other["rarity_class"] == item["rarity_class"] for other in items)
        for items in tracker.local_names.values() for item in items)


def commit_potential_take(runtime):
    """Commit inventory only after the Take callback has confirmed leaving its page."""
    pending = runtime.pending_take
    if pending is None:
        return False
    if not pending.committed:
        pending.tracker.selected(pending.potential)
        if pending.commit_inventory is not None:
            pending.commit_inventory(pending.potential)
        pending.committed = True
        # A following reward can have similar cards and the same wallet.
        # The confirmed exit, rather than the old signature, opens a new page.
        if pending.tracker is not None:
            pending.tracker._current_cards = None
            pending.tracker._page_key = None
            pending.tracker.selected_page_index = None
            purchase = getattr(pending.tracker, "pending_shop_purchase", None)
            confirmed_page = (pending.evidence or {}).get("take_confirmation")
            if purchase is not None and confirmed_page in (BY_NAME["shop"].node, BY_NAME["shop_main"].node):
                purchase["return_page"] = confirmed_page
    runtime.pending_take = None
    return True


def read_potential_take_page(screen, pending, *, buttons=None, reuse_offer_levels=False, allow_unselected=False,
                             before_take=False):
    """Read actual selection and the complete card page, without changing state."""
    if buttons is None:
        buttons = screen.get_select_button()
    # The selected card is raised. Without its Take anchor, entering/leaving
    # animations can place the level text outside the unselected-card ROI.
    # Wait for the next frame instead of reading names/levels at that offset.
    if len(buttons) != 1 and not (allow_unselected and not buttons):
        return PotentialTakePage(visible=bool(buttons))
    core = screen.check_core_potential()
    rarity_hits = [] if core else sorted(
        (result.box[0], rarity, result.box)
        for rarity, node in (("common", "星塔_节点_选择潜能_识别普通潜能数量_agent"),
                             ("rare", "星塔_节点_选择潜能_识别稀有潜能数量_agent"))
        for result in screen._recognize(node))
    rarities = ["core"] * 3 if core else [rarity for _, rarity, _ in rarity_hits]
    if not 1 <= len(rarities) <= 3:
        rarities = []
    page = PotentialTakePage(visible=bool(buttons or rarities or core))
    if not rarities:
        return page
    layouts = pending.layouts[len(rarities)]
    if buttons:
        page.button = buttons[0]
        page.selected_index = next((i for i, layout in enumerate(layouts)
            if layout.x_bound[0] <= page.button[0] <= layout.x_bound[1]), -1)
        if page.selected_index < 0:
            return page
    if not core and len(rarity_hits) > 1:
        # Take can appear before the selected card has finished rising. Align
        # the existing rarity anchors by the same offset used for text ROIs.
        tops = [box[1] + (pending.selected_offset if i == page.selected_index else 0)
                for i, (_, _, box) in enumerate(rarity_hits)]
        if max(tops) - min(tops) > 3:
            return page
    potentials = []
    for i, (layout, rarity) in enumerate(zip(layouts, rarities)):
        if screen.context.tasker.stopping:
            return page
        offset = pending.selected_offset if i == page.selected_index else 0
        roi = list(layout.core_potential_roi if core else layout.general_potential_roi)
        roi[1] -= offset
        name = screen.get_potential_name(roi)
        if not clean_name(name):
            return page
        potentials.append(Potential(layout, index=i, name=name, rarity=rarity,
                                    selected=i == page.selected_index))
    # Only the frame before our first Take is known to have the original
    # source. A following reward can instead be a mixed enhancement page.
    single_role = (before_take and not core and pending.data is not None
                   and pending.data.params.potential_source != "enhance")
    identities = None
    if reuse_offer_levels:
        try:
            identities = _resolve_potential_items(screen, pending.tracker, potentials,
                pending.selected_offset, single_role=single_role, pending=pending)
        except PotentialObservationIncomplete:
            pass
    # Selecting a card changes its pose, not the already observed offer.
    # Before our first Take, verify every live identity/rarity and the target
    # Take slot, then reuse the complete offer levels instead of OCRing the
    # selected number's animation. Later pages always get fresh level reads.
    reuse_offer_levels = (reuse_offer_levels and page.selected_index == pending.potential.index
        and identities is not None
        and tuple((item["identity"], item["role"], p.rarity) for p, item in zip(potentials, identities))
            == tuple(card[:3] for card in pending.page_signature)
        and all(0 <= card[3] < card[4] <= 6 for card in pending.page_signature))
    for i, potential in enumerate(potentials):
        old, new = 0, 1
        if not core:
            if reuse_offer_levels:
                old, new = pending.page_signature[i][3:]
            else:
                roi = list(potential.layout.general_potential_level_roi)
                roi[1] -= pending.selected_offset if i == page.selected_index else 0
                old, new = screen.get_potential_level(roi)
                if not 0 <= old < new <= 6:
                    return page
        potential.old_level, potential.new_level = old, new
    if identities is None:
        try:
            identities = _resolve_potential_items(screen, pending.tracker, potentials,
                pending.selected_offset, single_role=single_role, pending=pending)
        except PotentialObservationIncomplete:
            return page
    page.signature = tuple(_card_signature(item, p.rarity, p.old_level, p.new_level)
                           for p, item in zip(potentials, identities))
    return page


def outside_potential_page(screen):
    page = classify(screen.context, screen.image,
                    candidates=BAG_RETURN_PAGES | {"settlement"})
    if page not in (BAG_RETURN_PAGES | {"settlement"}) - {"potential"}:
        return None
    if page == "field":
        # The field HUD can also show behind an opening potential overlay.
        if not any((result := screen.context.run_recognition(node, screen.image)) and result.hit
                   for node in ("星塔_神经网络_恢复自动前进", "星塔_神经网络_暂停自动前进")):
            return None
    return BY_NAME[page].node


def _confirm_potential_take(screen, pending, logger, *, timeout, max_clicks, take_sent):
    absent_frames, incomplete_frames = 0, 0
    page_left = False

    def observed(image):
        nonlocal absent_frames, incomplete_frames, page_left
        screen.image = image
        buttons = screen.get_select_button()
        if not take_sent:
            return None
        if not buttons:
            incomplete_frames = 0
            outside = outside_potential_page(screen)
            if outside is not None:
                return "outside", outside
            # A missed Take button alone does not mean the cards left the screen.
            cards_visible = screen.get_potentials_rarity(pending.potential.core)
            absent_frames = 0 if cards_visible else absent_frames + 1
            return None
        page_left = page_left or absent_frames >= 2
        absent_frames = 0
        page = read_potential_take_page(screen, pending, buttons=buttons)
        if page.signature is not None:
            incomplete_frames = 0
            if page.signature != pending.page_signature:
                return "page", page.signature
            page_left = False
        elif page_left and page.selected_index >= 0:
            incomplete_frames += 1
            if incomplete_frames >= 3:
                return "next_take", page.selected_index
        else:
            incomplete_frames = 0
        return None

    def retry(image):
        nonlocal take_sent
        screen.image = image
        if page_left or absent_frames:
            return False
        buttons = screen.get_select_button()
        if len(buttons) != 1:
            return False
        page = read_potential_take_page(screen, pending, buttons=buttons)
        same = page.signature == pending.page_signature and page.selected_index == pending.potential.index
        if not same:
            return False
        logger.debug("Retrying potential %(name)s (attempt %(attempt)s)", {
            "name": pending.potential.name, "attempt": 2})
        sent = bool(screen.click_potential(buttons[0]))
        take_sent = take_sent or sent
        return sent

    result = wait_for(screen.context, observed, timeout=timeout, stable_frames=2,
                      retry=retry if max_clicks > 1 else None, retry_after=.6)
    if result is not None:
        if pending.evidence is not None:
            pending.evidence["take_confirmation"] = "potential_page_reappeared" if result[0] == "next_take" else result[1]
        return True
    return False


def finish_potential_take(screen, pending, logger, *, timeout=15., max_clicks=2, runtime=None):
    """Read the original target, send Take, then commit only its observed exit."""
    def ready(image):
        screen.image = image
        buttons = screen.get_select_button()
        page = read_potential_take_page(screen, pending, buttons=buttons, reuse_offer_levels=True, before_take=True)
        if page.signature is not None:
            return ("ready" if page.signature == pending.page_signature else "changed"), page
        if not page.visible and outside_potential_page(screen) is not None:
            return "changed", page
        return None

    result = wait_for(screen.context, ready, timeout=timeout)
    if result is not None and result[0] == "ready" and result[1].selected_index != pending.potential.index:
        screen.select_potential(pending.potential)

        def selected(image):
            result = ready(image)
            return result if result is not None and (result[0] != "ready"
                or result[1].selected_index == pending.potential.index) else None

        def retry_select(image):
            result = ready(image)
            return bool(result is not None and result[0] == "ready"
                        and result[1].selected_index != pending.potential.index
                        and screen.select_potential(pending.potential))

        result = wait_for(screen.context, selected, timeout=timeout,
                          retry=retry_select if max_clicks > 1 else None, retry_after=.6)
    if result is None or result[0] != "ready" or screen.context.tasker.stopping:
        return False
    take_sent = bool(screen.click_potential(result[1].button))
    return _confirm_potential_take(screen, pending, logger, timeout=timeout, max_clicks=max_clicks,
                                   take_sent=take_sent)


class NeuralRecommendationHandler(ChoosePotentialHandler):
    """Reuse stateless current-page rules and neural navigation/confirmation."""
    _update_recommended_potentials = RecommendationHandler._update_recommended_potentials

    def wait_for_item_list_gone(self):
        if not self.screen.check_item_list_visibility():
            return True
        result = self.screen.context.run_task("星塔_神经网络_潜能_等待干扰文字消失")
        if not result or not result.status.succeeded:
            return False
        self.screen.update_image()
        return True

    def pick(self, potential):
        return potential.selected or self.screen.select_potential(potential)

    def _get_recommended_potential_indices(self, boxes, borders):
        matched = {}
        for box in boxes:
            index = next((i for i, (low, high) in enumerate(borders) if low <= box[0] <= high), None)
            if index is None:
                raise PotentialObservationIncomplete("推荐图标未对应实际潜能卡片")
            matched[index] = box
        return matched

    def refresh(self):
        if self.screen.context.tasker.stopping:
            return False
        old_coin = self.data.current_coin
        expected_coin = old_coin - self.data.refresh_cost
        if not self.screen.refresh():
            return False

        def observed(image):
            self.screen.image = image
            coin = self.screen.get_current_coin()
            if coin != expected_coin:
                return None
            rarities = self.screen.get_potentials_rarity(self.data.core_potential)
            return (coin, rarities) if len(rarities) == self.data.potential_count else None

        result = read_until(self.screen.context, observed, initial_image=self.screen.image)
        if result is not None:
            self.data.current_coin, self.data.potential_rarities = result
            self.data.refresh_count += 1
            return True
        logger.debug("Potential refresh remains unconfirmed: expected=%s", expected_coin)
        return False


class ModelPotentialHandler(NeuralRecommendationHandler):
    def __init__(self, interactor, data, runtime):
        super().__init__(interactor, data)
        self.runtime = runtime
        self.decision_method = "official_policy"

    def pick(self, potential):
        if self.screen.context.tasker.stopping:
            return False
        if super().pick(potential):
            return True
        tracker = self.runtime.require_tracker()
        pending = PendingPotentialTake(potential, self.data.params.selected_potential_offset,
            self.data.params.potential_layouts, tracker, self._page_signature(tracker), data=self.data)

        def selected(image):
            self.screen.image = image
            page = read_potential_take_page(self.screen, pending, allow_unselected=True, before_take=True)
            return bool(page.signature == pending.page_signature and page.selected_index == potential.index)

        def retry(image):
            self.screen.image = image
            page = read_potential_take_page(self.screen, pending, allow_unselected=True, before_take=True)
            if page.signature != pending.page_signature or page.selected_index == potential.index:
                return False
            self.runtime.logger.debug("Retrying potential %(name)s (attempt %(attempt)s)", {
                "name": potential.name, "attempt": 2})
            return bool(self.screen.select_potential(potential))

        return bool(wait_for(self.screen.context, selected, retry=retry, retry_after=.6))

    def read_potentials_info(self):
        last_error = PotentialObservationIncomplete("当前潜能页的卡片布局仍未完整识别")

        def observed(image):
            nonlocal last_error
            self.screen.image = image
            rarities = self.screen.get_potentials_rarity(self.data.core_potential)
            if rarities != self.data.potential_rarities:
                last_error = PotentialObservationIncomplete("当前潜能布局与预读页面尚未一致")
                self.runtime.logger.debug("Potential layout remains unconfirmed: expected=%s actual=%s",
                    self.data.potential_rarities, rarities)
                return None
            if self.screen.get_current_coin() != self.data.current_coin:
                last_error = PotentialObservationIncomplete("当前潜能钱包与预读页面尚未一致")
                return None
            if self.data.refresh_button and self.screen.get_refresh_cost() != self.data.refresh_cost:
                last_error = PotentialObservationIncomplete("当前潜能刷新费用与预读页面尚未一致")
                return None
            self.initialize_potentials()
            try:
                self._read_potentials_info()
            except PotentialObservationIncomplete as error:
                last_error = error
                self.runtime.logger.debug("Potential page incomplete: %(error)s", {"error": error})
                return None
            return True

        result = read_until(self.screen.context, observed, initial_image=self.screen.image)
        if self.screen.context.tasker.stopping:
            raise PotentialObservationIncomplete("任务已停止")
        if result is not None:
            return
        raise last_error

    def _read_potentials_info(self):
        tracker = self.runtime.require_tracker()
        # A unique name and rarity in the bound roster identify the role.
        self._update_recommended_potentials()
        self._update_names()
        self._update_levels()
        for potential in self.data.potentials:
            if not clean_name(potential.name):
                raise PotentialObservationIncomplete("本帧潜能名称为空，等待下一帧识别")
            if not self.data.core_potential and not potential_levels_valid(
                    potential.old_level, potential.new_level, self.data.params.potential_source):
                raise PotentialObservationIncomplete(
                    f"本帧潜能等级未完整识别：{potential.name} {potential.old_level}→{potential.new_level}")
        identities = _resolve_potential_items(self.screen, tracker, self.data.potentials,
            self.data.params.selected_potential_offset,
            single_role=not self.data.core_potential and self.data.params.potential_source != "enhance")
        keys = [(item["role"], item["identity"]) for item in identities]
        if len(set(keys)) != len(keys):
            raise PotentialObservationIncomplete("本帧候选包含重复潜能身份，等待下一帧识别")
        if not self.data.core_potential and any(tracker.identity_levels.get(key, 0) != potential.old_level
                for key, potential in zip(keys, self.data.potentials)):
            raise PotentialObservationIncomplete("本帧潜能等级与已确认持有尚未一致，等待下一帧识别")
        for potential, item in zip(self.data.potentials, identities):
            self._set_trekker(tracker, potential, item["role"])

    def _set_trekker(self, tracker, potential, role):
        if role not in tracker.trekkers:
            image = (self.screen.crop_screenshot(_trekker_roi(potential, self.data.params.selected_potential_offset)).copy()
                     if _needs_trekker_portrait(tracker, role) else np.array([]))
            tracker.trekkers[role] = Trekker(index=role - 1, image=image, is_main=role == 1)
        potential.trekker = tracker.trekkers[role]

    def choose_potential(self):
        tracker = self.runtime.require_tracker()
        signature = self._page_signature(tracker)
        page_key = self._pending_page_key(signature)
        pending = getattr(self.runtime, "pending_take", None)
        if pending is not None and pending.tracker is tracker and pending.page_key == page_key:
            self.decision_method = (pending.evidence or {}).get("decision_method", "official_policy")
            return self.data.potentials[pending.potential.index]
        self.runtime.pending_take = None
        self.decision_method = "official_policy"
        notes = getattr(self.runtime, "notes", None)
        floor_observation = (getattr(notes, "last_floor_observation", None)
                             if self.data.params.potential_source == "default"
                             and notes is not None and notes.observing else None)
        if floor_observation and floor_observation.get("complete") and floor_observation.get("source") == "floor_hud":
            tracker.open_page(self.data, observed_floor=floor_observation)
        else:
            tracker.open_page(self.data)
        if not tracker.decision_route_ready and not tracker.event_reward_page:
            raise PotentialObservationIncomplete("当前路线位置未确认，无法进行模型决策",
                reason="route_position_unconfirmed")
        if self.data.core_potential:
            candidates = tuple(Action("SELECT", index=i) for i in range(1, len(self.data.potentials) + 1))
            decision_context = self.runtime.decision_context("core", {
                "candidates": sim.to_jsonable(candidates),
                "visible_cards": [dict(ui_index=i, name=p.name, role=p.trekker.index + 1,
                                       target_level=tracker.identify(p.name, role=p.trekker.index + 1,
                                                                      rarity=p.rarity)["target_level"])
                                  for i, p in enumerate(self.data.potentials, 1)],
            })
            action = self.runtime.model.prepare_core(tracker.state, candidates, decision_context)
            self.decision_method = "core_target_rule"
            chosen = self.data.potentials[action.index - 1]
            if tracker.pool_order_ready and tracker.expected_total is not None:
                log_forecast(self.runtime.logger, tracker.expected_total,
                             [{"label": p.name, "change": 0., "selected": p is chosen}
                              for p in self.data.potentials])
            return chosen
        candidates = [Action("SELECT", index=i) for i in range(1, len(self.data.potentials) + 1)]
        if self.data.refresh_cost >= 0 and self.data.refreshable:
            candidates.append(Action("REROLL", price=self.data.refresh_cost))
        decision_context = self.runtime.decision_context("potential", {
            "candidates": sim.to_jsonable(candidates),
            "visible_cards": [dict(ui_index=i, name=p.name, role=p.trekker.index + 1,
                                   rarity=p.rarity, old_level=p.old_level, new_level=p.new_level)
                              for i, p in enumerate(self.data.potentials, 1)],
            "refresh_cost": self.data.refresh_cost,
        })
        actions = self.runtime.model.prepare_actions(tracker.state, candidates, decision_context)
        labels = {a: self.data.potentials[a.index-1].name
                  for a in actions if a.kind == "SELECT"}
        action = tracker.choose(tuple(actions), labels, decision_context=decision_context)
        self.decision_method = "model_policy"
        return None if action.kind == "REROLL" else self.data.potentials[action.index-1]

    def cache_potential_data(self, potential):
        tracker = self.runtime.require_tracker()
        signature = self._page_signature(tracker)
        self.runtime.pending_take = PendingPotentialTake(
            potential, self.data.params.selected_potential_offset,
            self.data.params.potential_layouts, tracker, signature,
            self._pending_page_key(signature), self.runtime.cache_potential, data=self.data,
            evidence={"decision_method": self.decision_method})

    def _page_signature(self, tracker):
        return tuple(_card_signature(
            tracker.identify(p.name, role=p.trekker.index + 1, rarity=p.rarity),
            p.rarity, p.old_level, p.new_level) for p in self.data.potentials)

    def _pending_page_key(self, signature):
        return (self.data.params.potential_source, self.data.core_potential,
                self.data.current_coin, signature)
