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


def register_if_supported():
    """Keep ordinary tasks independent of OpenCV and optional runtime settings."""
    global _task_sink
    from maa.agent.agent_server import AgentServer
    from maa.custom_action import CustomAction
    from maa.custom_recognition import CustomRecognition
    from maa.tasker import TaskerEventSink
    from .interactor import FrameObservation
    from .shop_ui.purchase import SHOP_PRICE_RECOGNITION, ShopItemPrice

    class LazyAction(CustomAction):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def run(self, context, argv):
            runtime = _load_supported_runtime()
            if runtime is None:
                return False
            return runtime.callbacks[self.name].run(context, argv)

    class LazyRecognition(CustomRecognition):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def analyze(self, context, argv):
            runtime = _load_supported_runtime()
            if runtime is None:
                return CustomRecognition.AnalyzeResult(box=None, detail={})
            return runtime.callbacks[self.name].analyze(context, argv)

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
    for kind, callback, add in (
        ("actions", LazyAction, AgentServer.register_custom_action),
        ("recognitions", LazyRecognition, AgentServer.register_custom_recognition),
    ):
        for binding in bindings[kind].values():
            proxy = callback(binding["name"])
            _lazy_callbacks[binding["name"]] = proxy
            add(binding["name"], proxy)
    AgentServer.register_custom_recognition("stella_value_frame_observation", FrameObservation())
    # Maa exchanges the available names when the client connects. A shop may be
    # hours after that handshake; publish its price reader before loading any model.
    AgentServer.register_custom_recognition(SHOP_PRICE_RECOGNITION, ShopItemPrice())
    _task_sink = LazyTaskSink()
    AgentServer.add_tasker_sink(_task_sink)


def _load_supported_runtime():
    if _runtime is not None:
        return _runtime
    # Check the interpreter architecture, including x64 Python emulated on ARM.
    from sysconfig import get_platform
    if get_platform() == "win-amd64":
        try:
            import cv2  # noqa: F401
        except (ImportError, OSError):
            pass
        else:
            return register()
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.error("$agent.custom.action.climb_tower_neural.unsupported_runtime")
    return None


def _register_task_sink(sink):
    from maa.agent.agent_server import AgentServer
    if _task_sink is None:
        AgentServer.add_tasker_sink(sink)
    else:
        _task_sink.delegate = sink
        if _task_sink.current:
            sink.on_raw_notification(*_task_sink.current)
            _task_sink.current = None
