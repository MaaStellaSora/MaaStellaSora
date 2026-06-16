from __future__ import annotations

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from custom.reco.climb_tower_potential.data import Data, Parameters
from custom.reco.climb_tower_potential.state import State
from custom.reco.climb_tower_potential.ui import UIInteractor
from custom.reco.climb_tower_potential.handler_default import ChoosePotentialHandler
from custom.reco.climb_tower_potential.handler_preset import RecommendationHandler, RecommendationPlusBagScanHandler
from custom.reco.climb_tower_potential.handler_json import AssistantPriorityHandler
from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential")


@AgentServer.custom_recognition("choose_potential_recognition")
class ChoosePotentialRecognition(CustomRecognition):

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        """自动选择潜能的主流程。

        读取 attach 参数与已拥有潜能状态，识别当前三张潜能卡片，
        根据自定义优先级列表选出最优潜能并返回其 box；
        不满足条件时执行刷新，最终兜底识别系统推荐图标。
        选择完成后将已拥有潜能状态写回节点 attach 持久化。

        Args:
            context: maa.context.Context
            argv: CustomAction.RunArg，含当前截图与节点名

        Returns:
            bool: 返回 True
        """
        node_name = argv.node_name
        params = self._get_params(context, node_name)
        data = Data(params=params)
        screen = UIInteractor(context)

        # 获取只使用一次的数据
        image = argv.image
        data.current_coin = screen.get_current_coin(image)
        data.refresh_cost = screen.get_refresh_cost(image)
        data.core_potential = screen.check_core_potential(image)
        data.potential_count = screen.get_potential_count(data.core_potential, image)

        # 加载相应的潜能处理类
        if data.params.handler == "json":
            handler = AssistantPriorityHandler(screen, data)
        elif data.params.handler == "default+":
            handler = RecommendationHandler(screen, data)
        elif data.params.handler == "default++":
            handler = RecommendationPlusBagScanHandler(screen, data)
        else: # default
            handler = ChoosePotentialHandler(screen, data)

        while True:
            # 获取潜能数据，并选择潜能
            potential = handler.read_potentials_info().choose()
            if potential:
                break
            elif data.refreshable:
                logger.info("没有找到符合条件的潜能，尝试刷新")
                handler.refresh()
            else:
                logger.info("[潜能选择] 没有找到符合条件的潜能，将按照保底顺序选择")
                potential = handler.choose_fallback_potential()
                break

        # 点击潜能
        click_result = handler.pick(potential)
        if not click_result:
            logger.error(f"点击潜能失败")

        # 保存已选潜能数据到状态类中
        State.owned_potentials.save(
            potential,
            fuzzy=data.params.handler in {"default+", "default++"},
        )

        return CustomRecognition.AnalyzeResult(box=potential.box, detail={})

    @staticmethod
    def _get_params(context: Context, node_name: str) -> Parameters:
        """获取节点 attach 中的所有参数，缺失时返回安全默认值。

        Args:
            context: maa.context.Context
            node_name: 当前节点名称，用于动态获取节点数据

        Returns:
            Parameters: 包含以下属性的实例：
                - max_refresh_count (int): 最大刷新次数，0 表示禁用
                - reserved_coin (int): 预留金币，计算可用金币时需减去此值
                - priority_list (list): 自定义优先级列表
                - owned_potentials (dict): 已拥有潜能状态，按 trekker 分组
        """
        node_data = context.get_node_data(node_name)
        attach = node_data.get("attach", {})
        params = Parameters(**attach)
        return params

