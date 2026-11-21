import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.define import Rect

from utils import logger as logger_module
logger = logger_module.get_logger(__name__)


RESOURCE_RECOGNITION_NODE = "猎影合围_识别资源数量"


def _get_keep_count(context: Context, key: str) -> int | None:
    node_data = context.get_node_data(RESOURCE_RECOGNITION_NODE)
    if not isinstance(node_data, dict):
        return None
    attach = node_data.get("attach", {})
    if not isinstance(attach, dict):
        return None
    count = attach.get(key, 0)
    if type(count) is not int or count < 0:
        logger.warning(
            "$agent.custom.reco.operation._get_keep_count.invalid_keep_count",
            {
                "value_1": key,
            },
        )
        return None
    return count

def _get_resource_count(context: Context, image: np.ndarray) -> int | None:
    reco_result = context.run_recognition(RESOURCE_RECOGNITION_NODE, image)
    if reco_result and reco_result.hit and reco_result.best_result:
        try:
            return int(reco_result.best_result.text.split("/")[0])
        except (AttributeError, TypeError, ValueError):
            logger.debug("资源数量 OCR 结果无效")
    return None

def _locate_track_button(context: Context, image: np.ndarray) -> list[Rect] | None:
    reco_result = context.run_recognition("猎影合围_追踪目标_追踪按钮", image)
    if reco_result and reco_result.hit:
        return [result.box for result in (reco_result.filtered_results or [])]
    return None

def _locate_hunt_again_button(context: Context, image: np.ndarray) -> Rect | None:
    reco_result = context.run_recognition("__猎影合围_追踪目标_再次讨伐按钮", image)
    if reco_result and reco_result.hit and reco_result.best_result:
        return reco_result.best_result.box
    return None


@AgentServer.custom_recognition("enough_tracking_permit_recognition")
class EnoughTrackingPermitRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        """检查追踪委托书是否大于保留数量及追踪按钮是否可用。"""
        track_button_boxes = _locate_track_button(context, argv.image)
        if not track_button_boxes or len(track_button_boxes) != 2:
            logger.debug("未找到2个追踪按钮")
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        keep_count = _get_keep_count(context, "tracking_permit_keep_count")
        count = _get_resource_count(context, argv.image)
        if count is None or keep_count is None:
            logger.debug("未识别到追踪委托书数量")
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        usable = count - keep_count
        if count <= 0 or usable <= 0:
            logger.debug(f"追踪委托书数量{count}小于等于保留数量{keep_count}或数量为0")
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        
        logger.info(
            "$agent.custom.reco.operation.analyze.dispatch_contract_count",
            {
                "count": count,
                "keep_count": keep_count,
            },
        )
        quick_track_cost = min(5, max(2, count))
        if usable >= quick_track_cost:
            track_button_box = track_button_boxes[0]
        else:
            track_button_box = track_button_boxes[-1]
        return CustomRecognition.AnalyzeResult(box=track_button_box, detail={})

@AgentServer.custom_recognition("lack_of_tracking_permit_recognition")
class LackOfTrackingPermitRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        """检查追踪委托书是否小于保留数量及再次讨伐按钮、追踪按钮是否可用。"""
        keep_count = _get_keep_count(context, "tracking_permit_keep_count")
        count = _get_resource_count(context, argv.image)
        logger.debug(f"结束追踪模块前检查：追踪委托书数量{count}，保留数量{keep_count}")
        if count is None or keep_count is None or (count > keep_count and count > 0):
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        hunt_again_button_box = _locate_hunt_again_button(context, argv.image)
        track_button_box = _locate_track_button(context, argv.image)
        if not hunt_again_button_box and not track_button_box:
            logger.debug("未找到再次讨伐按钮和追踪按钮，继续追踪")
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        logger.debug(f"追踪委托书数量{count}小于保留数量{keep_count}，且找到再次讨伐按钮或追踪按钮，退出追踪")
        return CustomRecognition.AnalyzeResult(box=(1, 1, 1, 1), detail={})

@AgentServer.custom_recognition("enough_hunt_license_recognition")
class EnoughHuntLicenseRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        """检查围猎许可证是否大于保留数量及协助按钮是否可用。"""
        coop_button_box = self._locate_coop_button(context, argv.image)
        if not coop_button_box:
            logger.debug("未找到协助按钮")
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        keep_count = _get_keep_count(context, "hunt_license_keep_count")
        count = _get_resource_count(context, argv.image)
        if count is None or keep_count is None or count <= keep_count or count <= 0:
            logger.debug(f"围猎许可证数量{count}小于等于保留数量{keep_count}或数量为0")
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        
        logger.info(
            "$agent.custom.reco.operation.analyze.hunt_license_count",
            {
                "count": count,
                "keep_count": keep_count,
            },
        )
        return CustomRecognition.AnalyzeResult(box=coop_button_box, detail={})

    @staticmethod
    def _locate_coop_button(context: Context, image: np.ndarray) -> Rect | None:
        reco_result = context.run_recognition("猎影合围_协助_协助讨伐按钮", image)
        if reco_result and reco_result.hit and reco_result.best_result:
            return reco_result.best_result.box
        return None

@AgentServer.custom_recognition("lack_of_hunt_license_recognition")
class LackOfHuntLicenseRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        """检查围猎许可证是否小于保留数量，以及无偿协助是否已完成。"""
        keep_count = _get_keep_count(context, "hunt_license_keep_count")
        count = _get_resource_count(context, argv.image)
        logger.debug(f"结束协助模块前检查：围猎许可证数量{count}，保留数量{keep_count}")
        if count is None or keep_count is None or (count > keep_count and count > 0):
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        if not self._is_free_support_completed(context):
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        return CustomRecognition.AnalyzeResult(box=(1, 1, 1, 1), detail={})

    @staticmethod
    def _is_free_support_completed(context: Context) -> bool:
        """检查无偿协助是否已完成"""
        node_name = "猎影合围_协助_点击无偿协助"
        node_data = context.get_node_data(node_name) or {}
        enabled = node_data.get("enabled", False)
        max_hit = node_data.get("max_hit", 0)
        hit = context.get_hit_count(node_name)
        logger.debug(f"结束协助模块前检查：当前无偿协助开关为{enabled}，当前无偿协助次数{hit}，目标次数{max_hit}")
        return not enabled or hit >= max_hit
