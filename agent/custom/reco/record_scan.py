from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.define import OCRResult

from utils import logger as logger_module
logger = logger_module.get_logger("record_scan")

# 运行时数值与目标配置的汇合节点（同时也是 loop_count 的载体）
LOOP_NODE = "星塔_循环用节点_agent"
# 结算页的两个识别节点（ROI 在 pipeline 里定义）
LEVEL_NODE = "星塔_记录_识别等级_agent"
POTENTIAL_NODE = "星塔_记录_识别潜能数_agent"

UNKNOWN = -1


@AgentServer.custom_action("record_scan")
class RecordScan(CustomAction):
    """读取结算界面的记录等级与潜能总数。

    只负责读取，不做任何判定与停止：结果写入 星塔_循环用节点_agent 的 attach，
    由 AscensionLoop 在回到主页后统一判断是否达标。

    读取失败一律记为 -1（未知），避免把识别失败误判成“0 级 / 0 潜能”而误停。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        # 如果未激活记录等级或潜能数的目标选项，直接返回
        if not self._is_options_active(context, LOOP_NODE):
            return True

        # 读取记录等级与潜能数
        image = context.tasker.controller.cached_image

        level = self._read_ints(context, image, LEVEL_NODE)
        parts = self._read_ints(context, image, POTENTIAL_NODE)

        attach = dict((context.get_node_data(LOOP_NODE) or {}).get("attach") or {})
        attach["record_level"] = level[0] if level else UNKNOWN
        attach["potential_count"] = sum(parts) if parts else UNKNOWN
        context.override_pipeline({LOOP_NODE: {"attach": attach}})

        logger.debug(
            f"[记录读取] 记录等级={attach['record_level']} "
            f"潜能数={attach['potential_count']}（分项 {parts}）"
        )
        if attach["record_level"] == UNKNOWN:
            logger.error(f"记录等级识别失败：{level}")
        if attach["potential_count"] == UNKNOWN:
            logger.error(f"潜能数识别失败：{parts}")
        return True

    @staticmethod
    def _is_options_active(context: Context, node: str) -> bool:
        """检查是否激活了记录等级或潜能数的目标选项。"""
        node_data = context.get_node_data(node) or {}
        attachment = node_data.get("attach", {})
        return attachment.get("min_record_level", 0) > 0 or attachment.get("min_potential_count", 0) > 0

    @staticmethod
    def _read_ints(context: Context, image, node: str) -> list[int]:
        """按 pipeline 节点识别数字，返回识别到的整数列表（失败为空列表）。"""
        detail = context.run_recognition(node, image)
        if not (detail and detail.hit):
            return []
        texts = [r.text for r in (detail.filtered_results or []) if isinstance(r, OCRResult)]
        return [int(t) for t in texts if t and t.isdigit()]
