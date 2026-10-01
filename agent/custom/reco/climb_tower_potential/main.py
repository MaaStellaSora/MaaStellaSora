from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from .data import Data, Parameters
from .interactor import PotentialInteractor
from .handler_default import ChoosePotentialHandler
from .handler_preset import RecommendationHandler
from .handler_json import AssistantPriorityHandler

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
            CustomRecognition.AnalyzeResult: 返回 AnalyzeResult
        """
        # 1. 预加载与初始化数据
        params = self._get_params(context, argv.node_name)
        data = Data(params=params)
        interactor = PotentialInteractor(context)

        self._preload_data(interactor, data)
        handler = self._load_handler(interactor, data)

        # 2. 界面就绪等待
        handler.wait_for_item_list_gone()

        # 3. 进入潜能选择与刷新循环
        while True:
            # 获取潜能数据，并选择潜能
            handler.initialize_potentials()
            handler.read_potentials_info()
            potential = handler.choose_potential()

            # 这里放置一个对当前抽取潜能的数据处理流程，比如开发者需要保存当前抽取潜能的详细信息
            handler.handle_draw_data(potential)

            # 判断是否找到符合条件的潜能，并做出相应处理
            if potential:
                break
            if not handler.data.refreshable: # 兜底选择
                logger.info("[潜能选择] 没有找到符合条件的潜能，按照保底顺序选择")
                potential = handler.choose_fallback_potential()
                break
            logger.info("没有找到符合条件的潜能，尝试刷新")
            if not handler.refresh(): # 刷新潜能
                return self._handle_fatal_error(context, "刷新潜能失败，为保证爬塔质量，将中止任务")

        # 防御性检查
        if not potential:
            return self._handle_fatal_error(context, "潜能选择出现问题，为保证爬塔质量，将中止任务")

        # 4. 点击潜能
        if not handler.pick(potential):
            return self._handle_fatal_error(context, "点击潜能失败，为保证爬塔质量，将中止任务")

        # 5. 保存已选潜能数据到状态类中
        handler.cache_potential_data(potential)

        return CustomRecognition.AnalyzeResult(box=potential.box, detail={})

    @staticmethod
    def _get_params(context: Context, node_name: str) -> Parameters:
        """获取节点 attach 中的所有参数，缺失时返回安全默认值。

        Args:
            context: maa.context.Context
            node_name: 当前节点名称，用于动态获取节点数据

        Returns:
            Parameters: 参数对象，包含常用字段（根据 attach 可选）：
                - potential_source (str): 潜能来源标识（例如 "enhance"）。
                - max_refresh_count (int): 最大刷新次数；0 表示禁用刷新。
                - reserved_coin (int): 预留金币，计算可用金币时需减去此值。
                - priority_list (list): json自定义优先级列表，用于按优先级选择潜能。
                - owned_potentials (dict): 已拥有潜能的状态信息。
                - handler (str): 使用的处理器类型（如 "json"、"preset" 等）。
                - environment (str): 塔难度环境。
                - threshold_coef_str (str): 刷新阈值系数字符串配置。
                - threshold_decay_str (str): 刷新阈值衰减系数字符串配置。
        """
        node_data = context.get_node_data(node_name) or {}
        attach = node_data.get("attach", {})
        params = Parameters(**attach)
        return params

    @staticmethod
    def _preload_data(interactor: PotentialInteractor, data: Data):
        """预加载不受左侧道具列表遮挡、且后续选择潜能流程中不需要再次获取的数据
        包括金币、刷新花费、核心潜能、潜能数量与类型等
        """
        # 读取不需要刷新且没有遮挡的公用数据
        data.initial_coin = interactor.get_current_coin()
        data.current_coin = data.initial_coin
        data.refresh_cost = interactor.get_refresh_cost()
        data.core_potential = interactor.check_core_potential()
        data.potential_types = interactor.get_potential_types(data.core_potential)
        if data.params.potential_source != "enhance":
            data.level_upped = interactor.check_level_upped()

    @staticmethod
    def _load_handler(interactor: PotentialInteractor, data: Data):
        """加载相应的潜能处理类"""
        if data.params.handler == "json":
            handler = AssistantPriorityHandler(interactor, data)
        elif data.params.handler == "preset":
            handler = RecommendationHandler(interactor, data)
        # elif data.params.handler == "preset+bag":
        #     handler = RecommendationPlusBagScanHandler(interactor, data)
        else:
            handler = ChoosePotentialHandler(interactor, data)
        return handler

    @staticmethod
    def _handle_fatal_error(context: Context, message: str):
        logger.error(message)
        context.tasker.post_stop()
        return CustomRecognition.AnalyzeResult(box=None, detail={})