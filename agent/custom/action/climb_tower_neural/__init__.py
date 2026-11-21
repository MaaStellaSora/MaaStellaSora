"""Register lightweight callbacks; initialize the optional strategy on first use."""
import json
from pathlib import Path

_runtime = None
_task_sink = None
_lazy_callbacks = {}
TASK_ENTRY = "星塔_神经网络_入口"


def register():
    global _runtime
    if _runtime is None:
        from .maa_hook import ValueModelRuntime
        module_dir = Path(__file__).resolve().parent
        _runtime = ValueModelRuntime(module_dir, module_dir.parents[3])
        _runtime.register()
    return _runtime


def register_lazy_callbacks():
    """Keep ordinary tasks independent of the optional strategy and model."""
    try:
        _register_lazy_callbacks()
    except Exception:
        if _runtime_is_available():
            raise


def _register_lazy_callbacks():
    global _task_sink
    from maa.agent.agent_server import AgentServer
    from maa.custom_action import CustomAction
    from maa.custom_recognition import CustomRecognition
    from maa.tasker import TaskerEventSink
    from .interactor import FrameObservation
    from .shop_ui.purchase import SHOP_PRICE_RECOGNITION, ShopItemPrice

    def register_lazy_action(name):
        @AgentServer.custom_action(name)
        class LazyAction(CustomAction):
            def run(self, context, argv):
                runtime = _load_supported_runtime()
                if runtime is None:
                    return False
                return runtime.callbacks[name].run(context, argv)

        return LazyAction

    def register_lazy_recognition(name):
        @AgentServer.custom_recognition(name)
        class LazyRecognition(CustomRecognition):
            def analyze(self, context, argv):
                runtime = _load_supported_runtime()
                if runtime is None:
                    return CustomRecognition.AnalyzeResult(box=None, detail={})
                return runtime.callbacks[name].analyze(context, argv)

        return LazyRecognition

    class LazyTaskSink(TaskerEventSink):
        """Remember the first task's start before its preparation loads the runtime."""
        current = None
        delegate = None

        def on_raw_notification(self, tasker, msg, details):
            if self.delegate is not None:
                self.delegate.on_raw_notification(tasker, msg, details)
            elif msg == "Tasker.Task.Starting":
                self.current = (tasker, msg, dict(details)) if details.get("entry") == TASK_ENTRY else None
            elif (msg in ("Tasker.Task.Succeeded", "Tasker.Task.Failed")
                  and self.current and details.get("task_id") == self.current[2].get("task_id")):
                self.current = None

    bindings = json.loads((Path(__file__).parent / "agent_bindings.json").read_text(encoding="utf-8"))
    for binding in bindings["actions"].values():
        name = binding["name"]
        _lazy_callbacks[name] = register_lazy_action(name)
    for binding in bindings["recognitions"].values():
        name = binding["name"]
        _lazy_callbacks[name] = register_lazy_recognition(name)
    AgentServer.custom_recognition("stella_value_frame_observation")(FrameObservation)
    # Maa exchanges the available names when the client connects. A shop may be
    # hours after that handshake; publish its price reader before loading any model.
    AgentServer.custom_recognition(SHOP_PRICE_RECOGNITION)(ShopItemPrice)
    _task_sink = LazyTaskSink()
    AgentServer.add_tasker_sink(_task_sink)


def _load_supported_runtime():
    if _runtime is not None:
        return _runtime
    if _runtime_is_available():
        return register()
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.error("$agent.custom.action.climb_tower_neural.unsupported_runtime")
    return None


def _runtime_is_available():
    module_dir = Path(__file__).resolve().parent
    metadata_path = module_dir / "model.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = Path(metadata["checkpoint"])
    except (OSError, ValueError, KeyError, TypeError):
        return True
    if checkpoint.is_absolute():
        return True
    checkpoint_path = (module_dir / checkpoint).resolve()
    try:
        checkpoint_path.relative_to(module_dir)
    except ValueError:
        return True
    return checkpoint_path.is_file()


def _register_task_sink(sink):
    from maa.agent.agent_server import AgentServer
    if _task_sink is None:
        AgentServer.add_tasker_sink(sink)
    else:
        _task_sink.delegate = sink
        if _task_sink.current:
            sink.on_raw_notification(*_task_sink.current)
            _task_sink.current = None
