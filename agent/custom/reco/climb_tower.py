from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
import json
import time
from utils import logger



@AgentServer.custom_recognition("choose_potential_recognition")
class ChoosePotentialRecognition(CustomRecognition):

    POTENTIAL_ROIS = [
        {
            "core_potential": [190, 425, 210, 40],
            "general_potential": [190, 395, 210, 40],
            "general_potential_level": [190, 425, 210, 40]
        },
        {
            "core_potential": [535, 425, 210, 40],
            "general_potential": [535, 395, 210, 40],
            "general_potential_level": [535, 425, 210, 40]
        },
        {
            "core_potential": [880, 425, 210, 40],
            "general_potential": [880, 395, 210, 40],
            "general_potential_level": [880, 425, 210, 40]
        }
    ]

    def __init__(self):
        super().__init__()
        self.logger = logger.get_logger(__name__)

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        """
            选择潜能

            json格式说明：
            直接使用list排列方式，按照顺序作为优先级
                name(str | list): 潜能名称，必选。支持批量选择，当为list时，优先选择等级跨度大的，如条件相当则随机选择
                    如{"name": "a"} 或 {"name": ["a", "b"]}
                level_span(int): 等级跨度，可选，默认为1。例如2表示等级差2级以上时生效
                max_level(int): 最大等级，可选，默认为6。例如6表示6级以下时生效
                condition(list): 其他复杂生效条件，如果列表内元素为list，那么表示or逻辑，如果列表内元素为dict，那么表示and逻辑
                    name(str): 潜能名称，必选。
                    level(int): 等级，必选。例如2表示等级为2以上时生效
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
        # 读取配置：json作业，然后生成潜能优先级列表
        # TODO: 配置文件传递，看看要不要在启动任务时做一个节点生成，直接保留在attach里免得需要多次处理
        json_example = [
            {"name": "a", "level_span": 2, "max_level": 6},
            {"name": "b", "condition": {}},
        ]

        # 读取潜能列表及预留金币
        owned_potentials, reserve_coin = self._get_attachments(context)

        # TODO： 读取刷新次数，配置也要加
        max_refresh_count = 5
        # 判断有没有刷新按钮，有就读取当前金币以及刷新需要金币，然后计算出可以刷新的次数

        refresh_count = 0
        if self._is_refreshable(context, argv.image):
            current_coin = self._get_current_coin(context, argv.image)
            refresh_cost = self._get_refresh_cost(context, argv.image)
            refresh_count = min(max_refresh_count, max(0, current_coin - reserve_coin) // refresh_cost)

        while refresh_count >= 0:
            # 读取当前三个潜能的名称，以及等级
            available_potentials = self._get_available_potentials(context, argv.image)
            # 判断选哪个
            for potential in available_potentials:
                # 根据潜能名称获取优先级
                priority = self._get_potential_priority(potential, json_example)

            #判断不出来，识别系统推荐，并返回系统推荐的box

        return CustomRecognition.AnalyzeResult(
            box=[0, 0, 0, 0],
            detail={}
        )

    def _is_refreshable(self, context, image = None):
        """
            检查是否可刷新

            Args:
                context(Context): 上下文对象
                image(nd.array): 截图

            Returns:
                bool: 是否可刷新
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        reco_detail = context.run_recognition("星塔_通用_点击刷新_agent", image)
        self.logger.debug(f"识别刷新按钮结果：{[r.text for r in reco_detail.all_results]}")
        if reco_detail and reco_detail.hit:
            return True
        return False

    def _get_current_coin(self, context, image=None, max_try=3):
        """
            检查当前金币

            Args:
                context(Context): 上下文对象
                image(nd.array): 截图
                max_try(int): 最大尝试次数

            Returns:
                int: 当前金币数量，识别失败时返回0
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
                return 0

        self.logger.error("无法读取当前金币数量")
        return 0

    def _get_refresh_cost(self, context, image=None, max_try=3):
        """
            检查刷新花费

            Args:
                context(Context): 上下文对象
                image(nd.array): 截图
                max_try(int): 最大尝试次数

            Returns:
                int: 刷新花费，识别失败时返回65535
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        for _ in range(max_try):
            reco_detail = context.run_recognition("星塔_通用_识别刷新花费_agent", image)
            self.logger.debug(f"识别刷新花费结果：{[r.text for r in reco_detail.all_results]}")
            if reco_detail and reco_detail.hit:
                return int(reco_detail.best_result.text)

            # 失败时，等待1秒后重试
            time.sleep(1)
            image = context.tasker.controller.post_screencap().wait().get()

            # 检查是否中断任务
            if context.tasker.stopping:
                return 65535

        self.logger.error("无法读取刷新花费")
        return 65535

    def _get_attachments(self, context: Context) -> tuple[list[str], int]:
        """
            获取attach中的潜能列表及预留金币

            Args:
                context: maa.context.Context

            Returns:
                list[str]: 潜能列表
                int: 预留金币
        """
        attachments = context.get_node_data("星塔_节点_选择潜能_agent")

        try:
            potentials = attachments['attach']['potentials']
            reserve_coin = attachments['attach']['reserve_coin']
        except (TypeError, KeyError, IndexError, AttributeError) as e:
            self.logger.warning(f"提取潜能列表及预留金币的过程中出现问题: {e}")
            potentials = []
            reserve_coin = 0
            return potentials, reserve_coin

        return potentials, reserve_coin

    def _get_available_potentials(self, context: Context, image=None) -> list[dict]:
        """
            获取当前待选潜能

            Args:
                context: maa.context.Context
                image(nd.array): 截图

            Returns:
                list[dict]: 潜能列表，每个元素为{"name": str, "old_level": int, "new_level": int}
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        # 判断是否为核心潜能
        core_potential = True
        reco_detail = context.run_recognition("星塔_节点_选择潜能_识别核心潜能_agent", image)
        if reco_detail and reco_detail.hit:
            core_potential = False

        available_potentials = []
        for i in range(len(self.POTENTIAL_ROIS)):
            rois = self.POTENTIAL_ROIS[i]
            if core_potential:
                name_roi = rois["core_potential"]
            else:
                name_roi = rois["general_potential"]

            # 识别潜能名称
            reco_detail = context.run_recognition("星塔_节点_选择潜能_识别潜能名称_agent", image,{
                "星塔_节点_选择潜能_识别潜能名称_agent": {
                    "recognition": {
                        "param": {
                            "roi": name_roi
                        }
                    }
                }
            })
            if reco_detail and reco_detail.hit:
                potential_name = reco_detail.best_result.text
            else:
                self.logger.error(f"无法识别第{i+1}个潜能的名称")
                potential_name = ""

            # 识别潜能等级
            if core_potential:
                available_potentials.append({"name": potential_name, "old_level": 0, "new_level": 0})
            else:
                level_roi = rois["general_potential_level"]
                reco_detail = context.run_recognition("星塔_节点_选择潜能_识别潜能名称_agent", image, {
                    "星塔_节点_选择潜能_识别潜能名称_agent": {
                        "recognition": {
                            "param": {
                                "roi": level_roi
                            }
                        }
                    }
                })
                if reco_detail and reco_detail.hit:
                    # TODO: 对等级数据进行解析
                    potential_level = reco_detail.best_result.text
                    if len(potential_level) == 1:
                        old_level = 0
                        new_level = int(potential_level)
                    elif len(potential_level) == 2:
                        old_level = int(potential_level[0])
                        new_level = int(potential_level[1])
                    else:
                        self.logger.warning(f"无法解析第{i+1}个潜能的等级：{potential_level}")
                        old_level = -1
                        new_level = -1
                else:
                    self.logger.error(f"无法识别第{i+1}个潜能的等级")
                    old_level = -1
                    new_level = -1

                available_potentials.append({"name": potential_name, "old_level": old_level, "new_level": new_level})

        return available_potentials

    def _get_potential_priority(self, potential: dict, json_example: list[dict]) -> int:
        """
            获取潜能的优先级

            Args:
                potential: 潜能，{"name": str, "old_level": int, "new_level": int}
                json_example: 配置中的潜能列表，每个元素为{"name": str, "level_span": int, "max_level": int}

            Returns:
                int: 优先级，数值越大优先级越高
        """
        priority = 0
        for item in json_example:
            if item["name"] == potential["name"]:
                priority = item["level_span"] * (potential["new_level"] - potential["old_level"])
                break
        return priority
