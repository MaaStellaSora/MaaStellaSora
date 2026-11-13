"""Agent runtime. Register callbacks without modifying official classes."""
from pathlib import Path
import json

from maa.agent.agent_server import AgentServer
from utils.logger import get_logger
from custom.reco.climb_tower_potential.state import OwnedPotentials

from .log_filter import ModelLogFilter
from .observation import ObservationTracker
from .tower_sampling_collector import SamplingCollector
from .value_model import ValueModel
from .note_runtime import NoteRuntime
from .pages import classify
from .ui_flow import wait_for
from .event_policy import EventPolicy
from .sample_upload import SampleUpload


_STOP_MESSAGES = {
    "model_disabled": "$agent.custom.action.climb_tower_neural.model_disabled",
    "bag_not_opened": "$agent.custom.action.climb_tower_neural.bag_open_failed",
    "bag_preset_unconfirmed": "$agent.custom.action.climb_tower_neural.preset_read_failed",
    "bag_return_unconfirmed": "$agent.custom.action.climb_tower_neural.potential_return_failed",
    "model_start_unavailable": "$agent.custom.action.climb_tower_neural.run_start_unconfirmed",
    "potential_return_unconfirmed": "$agent.custom.action.climb_tower_neural.potential_return_failed",
    "potential_observation_incomplete": "$agent.custom.action.climb_tower_neural.potential_observation_incomplete",
    "route_position_unconfirmed": "$agent.custom.action.climb_tower_neural.route_position_unconfirmed",
    "pool_catalogue_incomplete": "$agent.custom.action.climb_tower_neural.pool_catalogue_incomplete",
    "potential_refresh_unconfirmed": "$agent.custom.action.climb_tower_neural.potential_refresh_failed",
    "potential_take_unconfirmed": "$agent.custom.action.climb_tower_neural.potential_take_failed",
    "shop_navigation_unconfirmed": "$agent.custom.action.climb_tower_neural.shop_navigation_failed",
    "shop_observation_incomplete": "$agent.custom.action.climb_tower_neural.shop_observation_incomplete",
    "shop_route_unresolved": "$agent.custom.action.climb_tower_neural.route_position_unconfirmed",
    "shop_model_inputs_unconfirmed": "$agent.custom.action.climb_tower_neural.route_position_unconfirmed",
    "shop_action_unconfirmed": "$agent.custom.action.climb_tower_neural.shop_action_failed",
    "event_observation_incomplete": "$agent.custom.action.climb_tower_neural.event_observation_incomplete",
    "quiz_answer_unmatched": "$agent.custom.action.climb_tower_neural.quiz_answer_unmatched",
    "event_mapping_unavailable": "$agent.custom.action.climb_tower_neural.event_mapping_unavailable",
    "event_model_inputs_unconfirmed": "$agent.custom.action.climb_tower_neural.event_state_unconfirmed",
}
_SHOP_ACTION_MESSAGES = {
    "open_shop_confirmation": "$agent.custom.action.climb_tower_neural.shop_navigation_failed",
    "restock_confirmation": "$agent.custom.action.climb_tower_neural.shop_refresh_failed",
    "enhance_navigation_confirmation": "$agent.custom.action.climb_tower_neural.shop_navigation_failed",
    "enhance_confirmation": "$agent.custom.action.climb_tower_neural.shop_enhance_failed",
    "purchase_confirmation": "$agent.custom.action.climb_tower_neural.shop_purchase_failed",
}


class ValueModelRuntime:
    def __init__(self, root, app_dir):
        self.root = Path(root)
        self.app_dir = Path(app_dir).resolve()
        self.settings = json.loads((self.root / "config.json").read_text(encoding="utf-8-sig"))
        self.logger = get_logger(__name__)
        self.ui_logs = ModelLogFilter()
        self.owned_cache = OwnedPotentials()
        self.collector = SamplingCollector(
            self.app_dir, get_logger(f"{__package__}.sampling"), self.root,
            owned_provider=lambda: self.owned_cache,
        )
        self.task_open = False
        self.collection_enabled = self.collector.enabled
        self.collector.enabled = False
        self._collector_loaded = False
        self.model = None
        self.event_policy = EventPolicy()
        self.tracker = None
        self.confirmed_preset = None
        self.pending_take = None
        self.pending_potential_refresh = None
        self.pending_start = None
        self.fresh_entry = False
        self.bag_returned = True
        self.event_page_cache = None
        self.notes = NoteRuntime(self)

    def requested(self, context):
        node = context.get_node_data("星塔_节点_选择潜能_agent") or {}
        return node.get("attach", {}).get("handler") == "value_model"

    def active(self, context):
        return self.requested(context)

    def model_ready(self, context):
        return self.active(context) and self.tracker is not None and self.confirmed_preset is not None

    @property
    def policy_mode(self):
        return "model" if self.tracker is not None and self.confirmed_preset is not None else "awaiting_preset"

    def stop_model(self, context, reason, *, phase, missing=(), **observed):
        """Leave the current page in place when its model input cannot be confirmed."""
        message = _STOP_MESSAGES.get(reason, "$agent.custom.action.climb_tower_neural.model_observation_failed")
        variables = {}
        if reason == "potential_take_unconfirmed":
            variables["name"] = self.pending_take.potential.name
        elif reason == "model_start_unavailable" and "first_potential" in missing:
            message = "$agent.custom.action.climb_tower_neural.preset_not_read"
        elif reason in ("shop_model_inputs_unconfirmed", "event_model_inputs_unconfirmed"):
            if "complete_pool_catalogue" in missing:
                message = "$agent.custom.action.climb_tower_neural.pool_catalogue_incomplete"
        elif reason == "shop_observation_incomplete" and "model_state" in missing:
            message = "$agent.custom.action.climb_tower_neural.model_observation_failed"
        elif reason == "shop_action_unconfirmed":
            message = _SHOP_ACTION_MESSAGES.get(missing[0], "$agent.custom.action.climb_tower_neural.shop_action_failed")
        self.pending_take = None
        self.pending_potential_refresh = None
        self.logger.debug("Model stopped: phase=%s reason=%s missing=%s observed=%r",
            phase, reason, list(missing), observed)
        self.collector.emit("runtime_recovery", reason=reason, missing=list(missing),
            recovery_action="stop_task", continuation_basis=phase, policy_mode=self.policy_mode,
            model_decision_recorded=False, continued=False, **observed)
        self.logger.error(message, variables)
        self.notes.finish(reason)
        self.collector.finish_run(reason, completion_observed=False)
        context.tasker.post_stop()

    def begin_task(self, context):
        self.task_open = self.requested(context)
        self.confirmed_preset = None
        if self.collector.upload is not None:
            self.collector.upload.set_consent(False)
        self.notes.reset()
        node = context.get_node_data("星塔_节点_选择潜能_agent") or {}
        options = (context.get_node_data("星塔_神经网络_运行环境") or {}).get("attach", {})
        upload_requested = options.get("research_upload") is True
        self.collector.finish_at_boundary("task_restarted")
        self.collector.enabled = (self.collection_enabled and self.requested(context)
                                  and (upload_requested or options.get("research_recording", True)))
        if self.collector.enabled and not self._collector_loaded:
            self.collector.loaded()
            self._collector_loaded = self.collector.enabled
        self.collector.begin_task(self.language(context), node.get("attach", {}).get("handler"))
        if self.collector.enabled and upload_requested:
            try:
                if self.collector.upload is None:
                    self.collector.upload = SampleUpload(self.app_dir, self.root, self.collector.directory,
                                                         self.collector.logger)
                self.collector.upload.set_consent(True)
                self.collector.upload.start()
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.collector.upload = None
                # Private settings may contain credentials; the reason is enough for diagnostics.
                self.logger.debug("Research uploader configuration unavailable (%s)", type(error).__name__)
        self.owned_cache = OwnedPotentials()
        self.tracker = None
        self.pending_take = None
        self.pending_potential_refresh = None
        self.pending_start = None
        self.fresh_entry = False
        self.bag_returned = True
        self.event_page_cache = None

    @staticmethod
    def language(context):
        node = context.get_node_data("星塔_节点_进行对话选择_agent") or {}
        language = node.get("attach", {}).get("lang_type", "cn")
        return {"zh_cn": "cn", "ja_jp": "jp"}.get(language, language)

    @staticmethod
    def _bag_task(context, node):
        result = context.run_task(node, {node: {"timeout": 3500}})
        return bool(result and result.status.succeeded)

    def _returned_from_bag(self, context, page, *, returned_frame=None):
        """A closing animation or a generic back button is not the original offer."""
        result = wait_for(context,
            lambda image: (image,) if classify(context, image, candidates={page}) == page else None,
            timeout=3.0)
        if result is not None and returned_frame is not None:
            returned_frame[:] = result
        return bool(result)

    def initialize_from_bag(self, context, *, returned_frame=None):
        """Initialize this fresh sweep at its blocking offer; only targets last across loops."""
        from .bag_preset_reader import read_bag_party
        from .preset_tool import compile_bag_preset, read

        page = "potential"
        language = self.language(context)
        preset = self.confirmed_preset
        self.bag_returned = True
        if preset is None or self.notes.observing:
            self.bag_returned = False
            if returned_frame is not None:
                returned_frame[:] = [None]
            if not self._bag_task(context, "星塔_打开背包界面_agent"):
                self.bag_returned = self._returned_from_bag(context, page, returned_frame=returned_frame)
                if not context.tasker.stopping:
                    self.stop_model(context, "bag_not_opened", phase=page, missing=("confirmed_bag_preset",))
                return False
            if preset is None:
                try:
                    party = read_bag_party(context, language, self.collector.capture_preset)
                    if context.tasker.stopping:
                        return False
                    preset = compile_bag_preset(party, language, read(self.root / "potential_name_catalog.json"))
                except (ValueError, RuntimeError, OSError) as exc:
                    if not context.tasker.stopping:
                        self.stop_model(context, "bag_preset_unconfirmed", phase="bag",
                            missing=("confirmed_bag_preset",), observation_error=str(exc))
                    return False
            try:
                if not context.tasker.stopping and self.notes.observing:
                    self.notes.read_bag_setup(context)
            except (ValueError, RuntimeError, OSError) as exc:
                self.collector.emit("note_bag_setup_observed", complete=False, reason=str(exc))
            finally:
                if not context.tasker.stopping and self._bag_task(context, "星塔_关闭背包界面_agent"):
                    self.bag_returned = self._returned_from_bag(context, page, returned_frame=returned_frame)
            if not self.bag_returned:
                if not context.tasker.stopping:
                    self.stop_model(context, "bag_return_unconfirmed", phase=page,
                        missing=("original_potential_page",))
                return False
        try:
            if self.model is None:
                self.model = ValueModel(self.root)
            tracker = ObservationTracker(self.root, preset, self.logger, self.model, self.collector)
            tracker.initialize_bag_inventory()
            self.tracker, self.confirmed_preset = tracker, preset
            self.collector.bind_preset(preset, tracker)
            self.collector.initialize_bag_inventory(preset)
            field = "name_zh_cn" if language == "cn" else "name_ja_jp"
            targets = [total + 2 for total in preset["role_ordinary_totals"]]
            self.logger.info("$agent.custom.action.climb_tower_neural.preset_loaded", {
                "party": " / ".join(member[field] for member in preset["party"]),
                "total": sum(targets), "potential_targets": " + ".join(map(str, targets)),
            })
        except (ValueError, RuntimeError, OSError) as exc:
            self.stop_model(context, "bag_preset_unconfirmed", phase=page,
                missing=("confirmed_bag_preset",), observation_error=str(exc))
            return False
        self.fresh_entry = False
        return True

    def start_clicked(self, node, setup_image, *, setup):
        self.collector.start_intent(node)
        self.pending_start = (setup_image, setup)

    def confirm_start(self, context, evidence):
        """Commit the saved setup only once a real tower screen confirms entry."""
        if self.pending_start is None:
            return
        image, setup = self.pending_start
        self.pending_start = None
        self.fresh_entry = True
        self.collector.confirm_start(evidence)
        self.notes.start(context, image, setup=setup)

    def ensure_tracker(self, context, page="tower", *, returned_frame=None):
        """Keep the callback frame, replacing it only after first-offer bag navigation."""
        if not self.requested(context):
            return False
        if self.fresh_entry and page == "potential":
            return self.initialize_from_bag(context, returned_frame=returned_frame)
        if self.model_ready(context):
            return True
        if not context.tasker.stopping:
            missing = "first_potential" if self.fresh_entry else "confirmed_run_start"
            self.stop_model(context, "model_start_unavailable", phase=page, missing=(missing,))
        return False

    def require_tracker(self):
        if self.tracker is None:
            raise ValueError("本局尚未确认潜能预设。")
        return self.tracker

    def decision_context(self, kind, payload):
        from .decision_context import decision_context
        return decision_context(self, kind, payload)

    def reset_run(self):
        self.owned_cache = OwnedPotentials()
        self.pending_take = None
        self.pending_potential_refresh = None
        self.pending_start = None
        self.fresh_entry = False
        self.bag_returned = True
        self.event_page_cache = None
        self.notes.reset()
        self.tracker = None
        self.collector.tracker = None

    def finish_task(self):
        """Keep only the loaded model and the recorder service between tasks."""
        if not self.task_open:
            return
        self.task_open = False
        self.reset_run()
        self.confirmed_preset = None

    def cache_potential(self, potential):
        """Keep this sweep's confirmed takes separate from the legacy strategy's state."""
        if potential.name:
            previous = self.owned_cache.find(potential.name, mode="EXACT", trekker=potential.trekker)
            if previous:
                previous.update(potential)
            else:
                self.owned_cache.add(potential)

    def register(self):
        from . import _lazy_callbacks, _register_task_sink
        from . import agent_callbacks
        from .sampling_events import TowerContextSink, TowerTaskSink

        self.ui_logs.attach(get_logger("custom"))
        bindings = json.loads((self.root / "agent_bindings.json").read_text(encoding="utf-8"))
        self.callbacks = {}
        for kind, register in (
            ("actions", AgentServer.register_custom_action),
            ("recognitions", AgentServer.register_custom_recognition),
        ):
            for binding in bindings[kind].values():
                callback = getattr(agent_callbacks, binding["handler"])(self)
                self.callbacks[binding["name"]] = callback
                if binding["name"] not in _lazy_callbacks and not register(binding["name"], callback):
                    raise RuntimeError(f"无法注册神经网络节点：{binding['name']}")
        AgentServer.add_context_sink(TowerContextSink(self.collector))
        _register_task_sink(TowerTaskSink(self.collector, self.notes, self))
        self.logger.debug("[神经网络] 独立 Agent 节点已注册；每次任务读取一次游戏当前预设。")
