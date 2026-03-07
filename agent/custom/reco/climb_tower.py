from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
import json
import time
from utils import logger



@AgentServer.custom_recognition("choose_potential_recognition")
class ChoosePotentialRecognition(CustomRecognition):

    def __init__(self):
        super().__init__()
        self.logger = logger.get_logger()

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        """
            选择潜能

            json格式说明：
            直接使用list排列方式，按照顺序作为优先级
                name(str): 潜能名称
                level_span(int): 等级跨度，默认为1。例如2表示等级差2级以上时生效
                max_level(int): 最大等级，默认为6。例如6表示5级以下时生效
                condition(list): 其他复杂生效条件，如果列表内元素为list，那么表示and逻辑，如果列表内元素为dict，那么表示or逻辑
                例如：
                    condition: [
                        {"name": "a", "level": 2},
                        {"name": "b", "level": 4}
                    ]
                表示需要a潜能等级为2以上，b潜能等级为4以上时生效

                又例如：
                    condition: [
                        [
                            {"name": "a", "level": 2},
                            {"name": "b", "level": 4}
                        ]
                        [
                            {"name": "c", "level": 6}
                        ]
                    ]
                表示
                    a潜能等级为2以上，以及b潜能等级为4以上时
                    或
                    c潜能等级为6以上时
                生效
        """
        # 读取配置：json作业，刷新次数
        json_example = [
            {"name": "a", "level_span": 2, "max_level": 6},
            {"name": "b", "condition": {}},
        ]
        max_refresh_count = 5
        # 读取当前金币以及刷新需要金币，计算出还能刷新多少次（可选）
        current_coin = self._get_current_coin(context, argv.image)
        refresh_cost = 40
        # 判断是核心潜能还是普通潜能，有没有刷新按钮
        is_core_potential = False
        has_refresh_button = True
        # 读取当前三个潜能的名称，以及等级
        potential = [{"name": "a", "old_level": 0, "new_level": 1},
                     {"name": "b", "old_level": 0, "new_level": 1},
                     {"name": "c", "old_level": 0, "new_level": 1}]
        # 判断选哪个

        #判断不出来，识别系统推荐，并返回系统推荐的box

        return CustomRecognition.AnalyzeResult(
            box=[0, 0, 0, 0],
            detail={}
        )

    def _get_current_coin(self, context, image = None, max_try = 3):
        """
            检查当前金币

            Args:
                context(Context): 上下文对象
                image(nd.array): 截图
                max_try(int): 最大尝试次数

            Returns:
                int | None: 当前金币数量，识别失败时返回None
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        for _ in range(max_try):
            reco_detail = context.run_recognition("星塔_通用_识别当前金币_agent", image)
            self.logger.debug(f"识别当前金币结果：{[r.text for r in reco_detail.all_results]}")
            if reco_detail and reco_detail.hit:
                return int(reco_detail.best_result.text)

            # 失败时，等待1秒后重试
            time.sleep(1)
            image = context.tasker.controller.post_screencap().wait().get()

            # 检查是否中断任务
            if context.tasker.stopping:
                return None

        return None