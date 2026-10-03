from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from .interactor import ShopInteractor
from .handler import ShopHandler
from .context import ShopContext, ShopParams

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_shop")


@AgentServer.custom_action("shop_action")
class ShopAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """商店楼层自动购买主流程。

        读取配置参数后按 priority 循环购买，每轮结束后执行第二轮溢购饮料，
        满足条件则刷新并重新购买；循环结束后 final 商店追加补买饮料和零头购买。

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 正常完成返回 True；用户中止返回 False。
        """
        data = self._init_data(context, argv.node_name)
        interactor = ShopInteractor(context)
        handler = ShopHandler(interactor, data)

        # 读取商店层信息
        handler.read_shop_page_info()

        while not interactor.context.tasker.stopping:
            # 读取商品信息
            handler.read_goods_page_info()

            # 执行购买计划
            handler.execute_buy_plan()

            if not handler.can_afford_refresh():
                break
            handler.refresh()

        # 购买完成，返回商店层进行强化
        handler.enhance()

        # 结束商店层任务
        return True

    @staticmethod
    def _init_data(context: Context, node_name: str) -> ShopContext:
        """从节点 attach 读取商店配置参数，并生成商店层节点使用的上下文对象。
        为防止与 MaaFramework 的上下文对象名字冲突，将商店上下文对象命名为 data。

        Args:
            context: MaaFramework 的任务上下文。
            node_name: 当前节点名称。

        Returns:
            ShopContext: 包含所有商店配置参数的上下文对象
        """
        # 商店参数
        node_data = context.get_node_data(node_name) or {}
        shop_attach = node_data.get("attach", {})

        # 生成上下文对象，并返回上下文对象
        params = ShopParams(**shop_attach)
        data = ShopContext(params)

        return data
