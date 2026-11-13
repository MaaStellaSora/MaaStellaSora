"""Maa callbacks: reuse official flows and select the model only for its mode."""
from dataclasses import replace
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition

from .interactor import Interactor
from custom.action.climb_tower_preparation import AscensionPreparation
from custom.action.climb_tower_loop import AscensionLoop, HOME_NODE_OFF, HOME_NODE_ON
from custom.action.climb_tower_shop.main import ShopAction
from custom.reco.climb_tower_event.main import EventRecognition as OfficialEventRecognition
from .event_reader import EventRecognition

from .events import _read_event_menu, analyze_model_event
from .event_mapping import is_quiz_consequence
from .potential import (
    ModelPotentialHandler, PotentialObservationIncomplete, commit_potential_take, finish_potential_take,
)
from .potential_reading import (
    NeuralPotentialRecognition as ChoosePotentialRecognition,
    NeuralPotentialContext as PotentialContext,
    NeuralPotentialInteractor as PotentialInteractor,
)
from .shop import ShopNavigationFailed, run_model_shop
from .settlement_reader import SettlementReader
from .tower_sampling_collector import ObservedPotentialHandler
from .pages import BAG_RETURN_PAGES, classify
from .ui_flow import read_until, wait_for


class RuntimeCallback:
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime


class Preparation(RuntimeCallback, AscensionPreparation):
    def run(self, context, argv):
        if not self.runtime.requested(context):
            return super().run(context, argv)
        if not self.runtime.settings.get("enabled", True):
            self.runtime.stop_model(context, "model_disabled", phase="preparation")
            return False
        if self.runtime.language(context) not in ("cn", "jp"):
            self.runtime.logger.error("$agent.custom.action.climb_tower_neural.preset_language_unsupported")
            return False
        environment = context.get_node_data("星塔_神经网络_运行环境") or {}
        if not environment.get("attach", {}).get("native_windows", False):
            self.runtime.logger.error("$agent.custom.action.climb_tower_neural.unsupported_controller")
            return False
        with self.runtime.ui_logs.scope():
            result = super().run(context, argv)
        if not result:
            return False
        self.runtime.begin_task(context)
        return True


class TowerLoop(RuntimeCallback, AscensionLoop):
    def run(self, context, argv):
        if not self.runtime.requested(context):
            return super().run(context, argv)
        self.runtime.notes.finish("run_loop_boundary")
        self.runtime.collector.finish_at_boundary("loop_without_settlement")
        self.runtime.reset_run()
        context.override_pipeline(HOME_NODE_ON)
        # This task stops at its requested sweep count, independent of OCR at settlement.
        settings = (context.get_node_data(argv.node_name) or {}).get("attach", {})
        remaining = settings.get("loop_count", 1) - 1
        if remaining > 0:
            context.override_pipeline({argv.node_name: {"attach": {**settings, "loop_count": remaining}}})
            context.override_pipeline(HOME_NODE_OFF)
            self.runtime.logger.info("$agent.custom.action.climb_tower_loop.run.tower_run_completed",
                                     {"loop_count": remaining})
        else:
            self.runtime.logger.info("$agent.custom.action.climb_tower_loop.run.tower_completed")
        return True


class Potential(RuntimeCallback, ChoosePotentialRecognition):
    def analyze(self, context, argv):
        if not self.runtime.requested(context):
            from custom.reco.climb_tower_potential.main import ChoosePotentialRecognition as NativeRecognition

            return NativeRecognition().analyze(context, argv)
        context = self.runtime.collector.context(context)
        self.runtime.confirm_start(context, argv.node_name)
        self.runtime.collector.ensure_run("potential_page_recognized")
        frame = [argv.image]
        if not self.runtime.ensure_tracker(context, "potential", returned_frame=frame):
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        if frame[0] is None:
            frame[0] = Interactor.capture(context)
        source = (context.get_node_data(argv.node_name) or {}).get("attach", {}).get("potential_source", "default")
        restored, navigated = self.runtime.notes.before_potential(context, frame[0], source)
        if not restored:
            if not context.tasker.stopping:
                self.runtime.stop_model(context, "potential_return_unconfirmed", phase="potential",
                    missing=("original_potential_page",))
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        if navigated:
            frame[0] = Interactor.capture(context)
        argv = replace(argv, image=frame[0])
        context.override_pipeline({argv.node_name: {"next": ["星塔_节点_选择潜能_拿走_agent"]}})
        with self.runtime.ui_logs.scope(self.runtime.active(context)), \
                self.runtime.collector.capture("potential_callback"):
            return self._analyze_page(context, argv)

    def _analyze_page(self, context, argv):
        """Read one offer; Pipeline owns every return after a refresh."""
        handler = None
        try:
            tracker = self.runtime.require_tracker()
            refresh = getattr(self.runtime, "pending_potential_refresh", None)
            data = refresh[0].data if refresh is not None and refresh[1] is tracker and refresh[2] else None
            screen = PotentialInteractor(context)
            screen.image = argv.image
            if data is None:
                data = PotentialContext(params=self._get_params(context, argv.node_name))
                def preload(image):
                    screen.image = image
                    self._preload_data(screen, data)
                    complete = (1 <= data.potential_count <= 3 and data.current_coin >= 0
                        and (data.core_potential or data.params.potential_source == "enhance"
                             or data.refresh_cost >= 0))
                    return True if complete else None
                if read_until(context, preload, initial_image=screen.image) is None:
                    raise PotentialObservationIncomplete("当前潜能页的卡片布局、钱包或刷新费用仍未完整识别")
            handler = self._load_handler(screen, data)
            return self._choose_on_page(context, argv.node_name, handler, data, tracker)
        except (PotentialObservationIncomplete, ValueError) as error:
            self.runtime.logger.debug("Potential page incomplete: %(error)s", {"error": error})
            if not context.tasker.stopping:
                self.runtime.stop_model(context, getattr(error, "reason", "potential_observation_incomplete"), phase="potential",
                    missing=("complete_potential_model_input",), observation_error=str(error))
            return CustomRecognition.AnalyzeResult(box=None, detail={})

    def _choose_on_page(self, context, node_name, handler, data, tracker):
        if handler.wait_for_item_list_gone() is False:
            raise PotentialObservationIncomplete("潜能页干扰文字未在等待后消失")
        handler.initialize_potentials()
        handler.read_potentials_info()
        potential = handler.choose_potential()
        handler.handle_draw_data(potential)
        if potential is None and data.refreshable:
            self.runtime.pending_potential_refresh = (handler, tracker, False)
            context.override_pipeline({node_name: {"next": ["星塔_神经网络_潜能刷新"]}})
            return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail={"refresh_requested": True})
        if potential is None:
            return self._handle_fatal_error(context, "$agent.custom.reco.climb_tower_potential.main.selection_failed")
        if not handler.pick(potential):
            return self._handle_fatal_error(context, "$agent.custom.reco.climb_tower_potential.main.pick_failed")
        self.runtime.pending_potential_refresh = None
        handler.cache_potential_data(potential)
        return CustomRecognition.AnalyzeResult(box=potential.box, detail={})

    def _load_handler(self, interactor, data):
        self.runtime.require_tracker()
        data.params = replace(data.params, environment="tower_8", max_refresh_count=100000, reserved_coin=0)
        handler = ModelPotentialHandler(interactor, data, self.runtime)
        return ObservedPotentialHandler(handler, self.runtime.collector, self.runtime.notes)

    def _handle_fatal_error(self, context, message):
        if self.runtime.requested(context):
            raise PotentialObservationIncomplete(message)
        return super()._handle_fatal_error(context, message)


class PotentialRefresh(RuntimeCallback, CustomAction):
    def run(self, context, argv):
        pending = self.runtime.pending_potential_refresh
        if pending is None or context.tasker.stopping:
            return False
        handler, tracker, confirmed = pending
        if confirmed:
            return True
        handler.screen.context = self.runtime.collector.context(context)
        succeeded = handler.refresh()
        self.runtime.pending_potential_refresh = (handler, tracker, True) if succeeded else None
        if not succeeded and not context.tasker.stopping:
            self.runtime.stop_model(context, "potential_refresh_unconfirmed", phase="potential_refresh",
                missing=("confirmed_refreshed_potential_page",))
        return succeeded


class PotentialTake(RuntimeCallback, CustomAction):
    def run(self, context, argv):
        context = self.runtime.collector.context(context)
        if self.runtime.pending_take is None:
            return False
        screen = PotentialInteractor(context)
        with self.runtime.collector.capture("potential_take"):
            result = finish_potential_take(screen, self.runtime.pending_take, self.runtime.logger, runtime=self.runtime)
        if result:
            commit_potential_take(self.runtime)
        elif (not context.tasker.stopping and self.runtime.collector.enabled
              and self.runtime.collector.config.get("capture_note_screenshots", False)
              and self.runtime.notes.ledger is not None):
            screen.screenshot()
            image_path = self.runtime.notes._save_image(screen.image, "potential_take_unresolved")
            if image_path:
                selection = self.runtime.collector.pending_selection or {}
                self.runtime.collector.emit("potential_take_unresolved", image_path=image_path,
                    operation_confirmed=False, offer_id=selection.get("offer_id"),
                    operation_id=selection.get("operation_id"), reward_chain_id=selection.get("reward_chain_id"))
        if not result and not context.tasker.stopping:
            self.runtime.stop_model(context, "potential_take_unconfirmed", phase="potential_take",
                missing=("confirmed_potential_take",))
        return result


class Shop(RuntimeCallback, ShopAction):
    @staticmethod
    def _init_data(context, node_name):
        from .shop_ui.context import ShopContext, ShopParams

        attachment = (context.get_node_data(node_name) or {}).get("attach", {})
        return ShopContext(ShopParams(**attachment))

    def run(self, context, argv):
        if not self.runtime.requested(context):
            return ShopAction().run(context, argv)
        context = self.runtime.collector.context(context)
        self.runtime.confirm_start(context, argv.node_name)
        self.runtime.collector.ensure_run("shop_page_recognized")
        self.runtime.notes.enter_shop()
        self.runtime.collector.emit("lifecycle_observed", lifecycle_type="shop_entry",
                                   node_name=argv.node_name, owned=self.runtime.collector.owned())
        with self.runtime.ui_logs.scope():
            try:
                if not self.runtime.ensure_tracker(context, "shop"):
                    return False
                return run_model_shop(self, context, argv, self.runtime)
            except ShopNavigationFailed as error:
                self.runtime.stop_model(context, "shop_navigation_unconfirmed", phase="shop",
                    missing=["confirmed_shop_page"], observed_error=str(error))
                return False
            finally:
                self.runtime.notes.after_shop()


class Event(RuntimeCallback, EventRecognition):
    def analyze(self, context, argv):
        if not self.runtime.requested(context):
            return OfficialEventRecognition().analyze(context, argv)
        context = self.runtime.collector.context(context)
        self.runtime.confirm_start(context, argv.node_name)
        if not self.runtime.ensure_tracker(context, "event"):
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        with self.runtime.ui_logs.scope():
            reading = _read_event_menu(self, context, argv, self.runtime)
            if context.tasker.stopping:
                return CustomRecognition.AnalyzeResult(box=None, detail={})
            self.runtime.collector.event_question = reading.question
            if reading.choices and self.runtime.collector.enabled:
                self.runtime.collector.event_menu(reading.choices)
            self.runtime.notes.observe_event(context, reading.image)
            result = analyze_model_event(self, context, argv, self.runtime, reading=reading)
        detail = result.detail if isinstance(result.detail, dict) else {}
        attach = (context.get_node_data(argv.node_name) or {}).get("attach", {})
        self.runtime.collector.event_choice(result, attach)
        if result and result.box is not None:
            event = self.runtime.collector.event or {}
            consequence = attach.get("last_consequence", "")
            self.runtime.notes.before_event(context, reading.image,
                event_id=event.get("event_id"), choice=attach.get("last_choice", ""),
                consequence=consequence, note_effect=detail.get("note_effect"),
                model_event_id=detail.get("event"), option_id=detail.get("option"),
                allow_text_effect=not is_quiz_consequence(consequence))
        return result


class RecordScan(SettlementReader):
    """Thin callback entry; settlement reading lives in its own module."""


def _click_box(context, box):
    result = context.run_action("星塔_神经网络_交互_点击", box=list(box))
    return bool(result and result.success)


def _note_click(context, argv):
    return _click_box(context, argv.box)


class NoteStart(RuntimeCallback, CustomAction):
    def run(self, context, argv):
        if context.tasker.stopping:
            return False
        image = context.tasker.controller.cached_image
        setup = None
        if self.runtime.collector.enabled:
            image, setup, box = self.runtime.notes.read_start_setup(context, image, argv.node_name, argv.box)
            if context.tasker.stopping or box is None:
                return False
            sent = _click_box(context, box)
        else:
            sent = _note_click(context, argv)
        if not sent:
            return False
        self.runtime.start_clicked(argv.node_name, image, setup=setup)
        return True


class NoteContinue(RuntimeCallback, CustomAction):
    def run(self, context, argv):
        if context.tasker.stopping:
            return False
        self.runtime.confirm_start(context, argv.node_name)
        image = Interactor.capture(context)
        if context.tasker.stopping:
            return False
        box = list(argv.box)
        _, restored, navigated = self.runtime.notes.capture(context, image, "result", continue_box=box)
        if not restored or context.tasker.stopping:
            return False
        if not navigated:
            return _click_box(context, box)

        def locate(frame):
            result = context.run_recognition(argv.node_name, frame)
            return result.best_result.box if result and result.hit else None

        box = read_until(context, locate, initial_image=Interactor.capture(context))
        return box is not None and _click_box(context, box)


class NoteScene(RuntimeCallback, CustomAction):
    def run(self, context, argv):
        self.runtime.confirm_start(context, argv.node_name)
        return True


class NoteEventClick(RuntimeCallback, CustomAction):
    def _current_choice(self, context, image, cached):
        if classify(context, image, candidates=BAG_RETURN_PAGES | {"settlement"}) != "event":
            return None
        key, index, _ = cached
        language = self.runtime.language(context)
        question = EventRecognition._get_question_text(context, image, language)
        choices = EventRecognition._get_choice_texts(context, image, language)
        if len(choices) != len(key[1]) or not 0 <= index < len(choices) or choices[index].available is False:
            return None
        if len({tuple(choice.box) for choice in choices}) != len(choices):
            return None
        fields = [(key[0], question)]
        for (label, effect, _), choice in zip(key[1], choices):
            fields.extend(((label, choice.text), (effect, choice.consequence)))
        if any(old and new and old != new for old, new in fields):
            return None
        # A question alone cannot locate an answer. Match the selected option
        # uniquely, using the clear fields on this returned menu.
        old_label, old_effect, _ = key[1][index]
        scores = []
        for choice in choices:
            score = 0
            for field, old in (("text", old_label), ("consequence", old_effect)):
                new = getattr(choice, field)
                if old and new and old != new:
                    score = -1
                    break
                observed = getattr(choice, "observations", {}).get(field, {})
                if old and old == new and observed.get("complete", True):
                    score += 1
            scores.append(score)
        if scores[index] <= 0 or scores.count(max(scores)) != 1 or scores[index] != max(scores):
            return None
        return dict(actual_box=list(choices[index].box), selected_ui_index=index,
                    observations=getattr(choices[index], "observations", {}))

    def run(self, context, argv):
        context = self.runtime.collector.context(context)
        tracker = self.runtime.tracker
        cached = (tracker.event_page_cache if tracker is not None else None) or self.runtime.event_page_cache
        if not cached:
            return False
        restored, inventory_navigation = self.runtime.notes.prepare_event_inventory(context)
        if not restored:
            return False
        if inventory_navigation:
            # A bag visit invalidates the old input frame. Locate the same
            # option again before issuing this event's only click.
            evidence = read_until(context, lambda image: self._current_choice(context, image, cached))
            sent = evidence is not None and _click_box(context, evidence["actual_box"])
            if evidence is not None:
                evidence["selection_basis"] = "returned_menu_recognition"
        else:
            box = list(argv.box)
            if box != cached[2].get("selected_box"):
                return False
            evidence = dict(actual_box=box, selected_ui_index=cached[1], selection_basis="decision_recognition")
            sent = _note_click(context, argv)
        if not sent:
            return False
        def observed(image):
            page = classify(context, image, candidates=BAG_RETURN_PAGES | {"settlement"})
            return page if page in (BAG_RETURN_PAGES | {"settlement"}) - {"event"} else None

        if wait_for(context, observed, timeout=5., stable_frames=2) is None:
            return False
        self.runtime.notes.confirm_event_choice()
        self.runtime.collector.event_clicked(dict(evidence, node_name=argv.node_name,
            decision_box=cached[2].get("selected_box"), confirmation_basis="stable_event_page_transition"))
        if self.runtime.tracker is not None:
            self.runtime.tracker.event_action_confirmed()
        self.runtime.event_page_cache = None
        return True
