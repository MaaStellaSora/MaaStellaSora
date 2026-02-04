from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

@AgentServer.custom_action("DailyGiftAction")
class DailyGiftAction(CustomAction):
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        # 领奖励的入口节点可以加到这里面
        entrance_node = ["采购_前往采购", "采购_前往商城"]

        # 遍历所有领取每日奖励的入口节点，执行所有领取每日奖励的任务
        for node in entrance_node:
            context.run_task(node)

        # 返回True，执行后续的“通用_返回主页”节点
        return True