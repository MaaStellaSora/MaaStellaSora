from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from custom.action import climb_tower_potential
from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_loop")


@AgentServer.custom_action("ascension_loop")
class AscensionLoop(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """检查剩余循环次数，决定是否退出爬塔流程

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
        if loop_count > 0:
            logger.info(f"完成一次爬塔，剩余爬塔次数：{loop_count}")
            context.override_pipeline({
                argv.node_name: {
                    "attach": {
                        "loop_count": loop_count
                    }
                }
            })
        else:
            logger.info("爬塔已完成，回到主页")
            context.override_next(argv.node_name, ["星塔_回到主页_agent"])

        return True
