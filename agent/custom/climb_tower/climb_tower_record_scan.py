from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.define import OCRResult, BoxAndScoreResult

from utils import logger as logger_module
logger = logger_module.get_logger(__name__)

# 运行时数值与目标配置的汇合节点（同时也是 loop_count 的载体）
LOOP_NODE = "星塔_循环用节点_agent"
# 结算页的三个识别节点（ROI 在 pipeline 里定义）
LEVEL_NODE = "星塔_记录_识别等级_agent"
POTENTIAL_LOCATION_NODE = "星塔_记录_定位潜能数量位置_agent"
POTENTIAL_RECOGNITION_NODE = "星塔_记录_识别潜能数量_agent"

UNKNOWN = -1


@AgentServer.custom_action("record_scan")
class RecordScanAction(CustomAction):
    """读取结算界面的纪录等级与潜能总数。

    只负责读取，不做任何判定与停止：结果写入 星塔_循环用节点_agent 的 attach，
    由 AscensionLoop 在回到主页后统一判断是否达标。

    读取失败一律记为 -1（未知），避免把识别失败误判成“0 级 / 0 潜能”而误停。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        # 如果未激活纪录等级或纪录潜能数的目标选项，直接返回
        if not self._is_options_active(context, LOOP_NODE):
            return True

        # 读取纪录等级与纪录潜能数
        image = context.tasker.controller.cached_image

        level = self._read_record_level(context, image, LEVEL_NODE)
        parts = self._read_potential_counts(context, image, POTENTIAL_LOCATION_NODE, POTENTIAL_RECOGNITION_NODE)

        attach = (context.get_node_data(LOOP_NODE) or {}).get("attach", {})
        attach["record_level"] = level
        attach["potential_count"] = sum(parts) if parts else UNKNOWN
        context.override_pipeline({LOOP_NODE: {"attach": attach}})

        logger.debug(
            f"[纪录读取] 纪录等级={attach['record_level']} "
            f"潜能数={attach['potential_count']}（分项 {parts}）"
        )
        if attach["record_level"] == UNKNOWN:
            logger.error(
                "$agent.custom.action.climb_tower_record_scan.run.recognition_failed",
                {
                    "level": level,
                },
            )
        if attach["potential_count"] == UNKNOWN:
            logger.error(
                "$agent.custom.action.climb_tower_record_scan.run.potential_recognition_failed",
                {
                    "parts": parts,
                },
            )
        return True

    @staticmethod
    def _is_options_active(context: Context, node: str) -> bool:
        """检查是否激活了纪录等级或潜能数的目标选项。"""
        node_data = context.get_node_data(node) or {}
        attachment = node_data.get("attach", {})
        return attachment.get("min_record_level", 0) > 0 or attachment.get("min_potential_count", 0) > 0

    @staticmethod
    def _read_record_level(context: Context, image, node: str) -> int:
        """按 pipeline 节点识别纪录等级，返回识别到的整数（失败为 -1）。"""
        detail = context.run_recognition(node, image)
        if not (detail and detail.hit and isinstance(detail.best_result, OCRResult)):
            return UNKNOWN
        return int(detail.best_result.text) if detail.best_result.text.isdigit() else UNKNOWN

    @staticmethod
    def _read_potential_counts(context: Context, image, location_node: str, recognition_node: str) -> list[int]:
        """使用 pipeline 节点识别潜能数，先定位记录潜能数量的位置，然后根据位置偏移分别读取潜能数量（失败为空列表）。"""
        location_detail = context.run_recognition(location_node, image)
        if not (location_detail and location_detail.hit):
            logger.error(
                "$agent.custom.action.climb_tower_record_scan._read_potential_counts.potential_count_location_failed"
            )
            return []
        if len(location_detail.filtered_results) != 3:
            logger.error(
                "$agent.custom.action.climb_tower_record_scan._read_potential_counts.recognition_failed",
                {
                    "count": len(location_detail.filtered_results),
                },
            )
            return []

        counts = []
        for i, r in enumerate(location_detail.filtered_results):
            if not isinstance(r, BoxAndScoreResult):
                logger.error(
                    "$agent.custom.action.climb_tower_record_scan._read_potential_counts.invalid_result_type",
                    {
                        "value": i+1,
                        "r": r,
                    },
                )
                return []
            pipeline_override = {recognition_node: {"recognition": {"param": {"roi": r.box}}}}
            reco_result = context.run_recognition(recognition_node, image, pipeline_override)
            if not (reco_result and reco_result.hit and isinstance(reco_result.best_result, OCRResult)):
                logger.error(
                    "$agent.custom.action.climb_tower_record_scan._read_potential_counts.potential_count_recognition_failed",
                    {
                        "value": i+1,
                    },
                )
                return []
            counts.append(int(reco_result.best_result.text))
        return counts
