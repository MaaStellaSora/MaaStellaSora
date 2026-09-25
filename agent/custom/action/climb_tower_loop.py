from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from custom.reco import climb_tower_potential
from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_loop")

HOME_NODE = "通用_返回主页"
UNKNOWN = -1


@AgentServer.custom_action("ascension_loop")
class AscensionLoop(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """检查剩余循环次数与记录目标，决定是否退出爬塔流程

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 返回 True。
        """
        # 重置潜能状态
        climb_tower_potential.State.reset()

        # 更新循环次数，并判断是否继续爬塔
        node_data = context.get_node_data(argv.node_name)
        if not node_data:
            node_data = {}
        attachment = node_data.get("attach", {})
        loop_count = attachment.get("loop_count", 1)
        loop_count -= 1

        # 记录目标识别失败：直接停止爬塔流程
        if attachment.get("record_level", -1) < 0 or attachment.get("potential_count", -1) < 0:
            logger.error("记录等级或潜能数识别失败，为保证爬塔质量，将结束爬塔")
            context.override_next(argv.node_name, [HOME_NODE])
            return True

        # 已达到记录目标：与爬塔次数用尽一样，正常回到主页结束
        if self._reached_record_target(attachment):
            logger.info(
                f"记录等级 {attachment.get('record_level')}、潜能数 "
                f"{attachment.get('potential_count')} 已达设定目标，结束爬塔"
            )
            context.override_next(argv.node_name, [HOME_NODE])
            return True

        if loop_count > 0:
            logger.info(f"完成一次爬塔，剩余爬塔次数：{loop_count}")
            attachment["loop_count"] = loop_count
            context.override_pipeline({
                argv.node_name: {
                    "attach": attachment
                }
            })
        else:
            logger.info("爬塔已完成，回到主页")
            context.override_next(argv.node_name, [HOME_NODE])

        return True

    @staticmethod
    def _reached_record_target(attachment: dict) -> bool:
        """判断本次结算是否达到设定的记录目标。

        目标值由任务选项写入 attach，未设置（0）的项不参与判断；
        本次未读到（-1）时一律视为不达标，避免识别失败导致误停。

        Args:
            attachment: 循环节点的 attach 数据。

        Returns:
            bool: 达标返回 True。
        """
        def _int(key: str, default: int) -> int:
            try:
                return int(attachment.get(key, default))
            except (TypeError, ValueError):
                return default

        min_level = _int("min_record_level", 0)
        min_count = _int("min_potential_count", 0)
        if min_level <= 0 and min_count <= 0:
            return False

        level = _int("record_level", UNKNOWN)
        count = _int("potential_count", UNKNOWN)
        if level < 0 or count < 0:
            return False

        return (min_level <= 0 or level >= min_level) and (min_count <= 0 or count >= min_count)
