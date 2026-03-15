from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
import json
import time
import random
from utils import logger



@AgentServer.custom_recognition("choose_potential_recognition")
class ChoosePotentialRecognition(CustomRecognition):

    MAX_POTENTIAL_LEVEL: int = 6  # 潜能等级上限，condition max_level 字段的默认值

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
        # 读取潜能列表及预留金币
        owned_potentials, reserve_coin = self._get_attachments(context)

        # 读取配置：json作业，然后生成潜能优先级列表
        # TODO: 配置文件传递，看看要不要在启动任务时做一个节点生成，直接保留在attach里免得需要多次处理
        # 估计还是要当场处理，因为需要根据当前潜能列表处理
        potential_priority_raw = [
            {"name": "a", "level_span": 2, "max_level": 6},
            {"name": "b", "condition": {}},
        ]
        potential_priority_list = self._parse_priority_raw_list(potential_priority_raw, owned_potentials)

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
                priority = self._get_potential_priority(potential_priority_list, potential)

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

    def _get_attachments(self, context: Context, node_name: str) -> dict:
        """获取节点 attach 中的所有参数，缺失时返回安全默认值。

        Args:
            context: maa.context.Context
            node_name: 当前节点名称，用于动态获取节点数据

        Returns:
            dict: 包含以下键的字典：
                - max_refresh_count (int): 最大刷新次数，0 表示禁用
                - reserve_coin (int): 预留金币，计算可用金币时需减去此值
                - priority_list (list): 自定义优先级列表
                - owned_potentials (dict): 已拥有潜能状态，按 trekker 分组
        """
        defaults = {
            "max_refresh_count": 0,
            "reserve_coin": 0,
            "priority_list": [],
            "owned_potentials": {},
        }
        node_data = context.get_node_data(node_name)
        if not node_data:
            self.logger.warning("get_node_data 返回 None，使用默认参数")
            return defaults

        attach = node_data.get("attach", {})
        return {
            "max_refresh_count": attach.get("max_refresh_count", 0),
            "reserve_coin": attach.get("reserve_coin", 0),
            "priority_list": attach.get("priority_list", []),
            "owned_potentials": attach.get("owned_potentials", {}),
        }

    @staticmethod
    def _parse_level_text(texts: list[str]) -> tuple[int, int]:
        """解析 OCR 返回的等级数字结果集。

        pipeline OCR 使用 \\d+ 匹配并剔除语言关键词，可能返回：
            ["1"]       -> old=0, new=1  （新获得，只有新等级）
            ["4", "5"]  -> old=4, new=5
            ["45"]      -> old=4, new=5  （两位数粘连）

        Args:
            texts: OCR all_results 中各结果的 text 列表

        Returns:
            tuple[int, int]: (old_level, new_level)，解析失败返回 (-1, -1)
        """
        numbers = []
        for t in texts:
            if len(t) == 2 and t.isdigit():
                numbers.extend([int(t[0]), int(t[1])])
            elif len(t) == 1 and t.isdigit():
                numbers.append(int(t))

        if len(numbers) == 1:
            return 0, numbers[0]
        if len(numbers) == 2:
            return numbers[0], numbers[1]
        return -1, -1

    def _get_available_potentials(self, context: Context, image=None) -> list[dict]:
        """获取当前待选的三个潜能卡片信息。

        Args:
            context: maa.context.Context
            image: 截图，为 None 时自动截图

        Returns:
            list[dict]: 潜能列表，每个元素结构：
                {
                    "name": str,        # 潜能名称
                    "old_level": int,   # 升级前等级，核心潜能为 0
                    "new_level": int,   # 升级后等级，核心潜能为 0
                    "is_core": bool,    # 是否为核心潜能
                    "box": list,        # 卡片区域 [x, y, w, h]
                }
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        reco_detail = context.run_recognition(
            "星塔_节点_选择潜能_识别核心潜能_agent", image
        )
        is_core = not (reco_detail and reco_detail.hit)

        available_potentials = []
        for i, rois in enumerate(self.POTENTIAL_ROIS):
            name_roi = rois["core_potential"] if is_core else rois["general_potential"]

            reco_detail = context.run_recognition(
                "星塔_节点_选择潜能_识别潜能名称_agent",
                image,
                {
                    "星塔_节点_选择潜能_识别潜能名称_agent": {
                        "recognition": {"param": {"roi": name_roi}}
                    }
                },
            )
            if reco_detail and reco_detail.hit:
                potential_name = reco_detail.best_result.text
            else:
                self.logger.error(f"无法识别第 {i + 1} 个潜能的名称")
                potential_name = ""

            if is_core:
                available_potentials.append({
                    "name": potential_name,
                    "old_level": 0,
                    "new_level": 0,
                    "is_core": True,
                    "box": rois["general_potential"],
                })
                continue

            reco_detail = context.run_recognition(
                "星塔_节点_选择潜能_识别潜能等级_agent",
                image,
                {
                    "星塔_节点_选择潜能_识别潜能等级_agent": {
                        "recognition": {"param": {"roi": rois["general_potential_level"]}}
                    }
                },
            )
            if reco_detail and reco_detail.hit:
                texts = [r.text for r in reco_detail.filtered_results]
                old_level, new_level = self._parse_level_text(texts)
                if old_level == -1:
                    self.logger.warning(f"无法解析第 {i + 1} 个潜能的等级：{texts}")
            else:
                self.logger.error(f"无法识别第 {i + 1} 个潜能的等级")
                old_level, new_level = -1, -1

            available_potentials.append({
                "name": potential_name,
                "old_level": old_level,
                "new_level": new_level,
                "is_core": False,
                "box": rois["general_potential"],
            })

        return available_potentials

    @staticmethod
    def _parse_priority_raw_list(potential_priority_raw, owned_potentials) -> list[dict]:
        """
            解析潜能优先级配置

            将 json 导入的原始配置转换为可快速索引的优先级列表。
            优先级数值越大越优先，由 raw_list 的 index 从大到小映射（index=0 最高优先级）。
            condition 不满足的条目直接跳过（视为无效，不加入结果）。

            Args:
                potential_priority_raw: 配置中的潜能列表，每个元素结构：
                    {
                        "name": str | list,   # 潜能名称，list 时同优先级内按 level_span 降序选，相等随机
                        "level_span": int,    # 可选，默认 1。new_level - old_level >= level_span 时该条生效
                        "max_level": int,     # 可选，默认 6。old_level < max_level 时该条生效
                        "condition": list     # 可选。列表元素为 dict 时表示 and；元素为 list 时表示 or（内层 and）
                    }
                owned_potentials: 当前已有的潜能列表，每个元素为{"name": str, "level": int}

            Returns:
                list[dict]: 有效的优先级条目列表，每个元素结构：
                    {
                        "names": list[str],  # 该条目匹配的潜能名称（统一转为 list）
                        "level_span": int,   # 等级跨度要求
                        "max_level": int,    # 最大旧等级限制
                        "priority": int      # 优先级数值，数值越大优先级越高
                    }
        """
        # 将 owned_potentials 转为 {name: level} 字典，方便 condition 查询
        owned_map = {p["name"]: p["level"] for p in owned_potentials}

        def _check_condition(condition):
            """
            检查 condition 是否满足。
            condition 为 list：
              - 元素为 dict → and 逻辑，所有 dict 都需满足
              - 元素为 list → or 逻辑，内层列表的 dict 全部满足即该分支满足，任意分支满足则整体满足
            """
            if not condition:
                return True

            # 判断是纯 and 结构（元素全为 dict）还是 or 结构（元素含 list）
            if all(isinstance(item, dict) for item in condition):
                # and 逻辑：所有条件都需满足
                return all(
                    owned_map.get(item["name"], 0) >= item["level"]
                    for item in condition
                )
            else:
                # or 逻辑：任意一个 and 分支满足即可
                for branch in condition:
                    if isinstance(branch, list):
                        if all(owned_map.get(item["name"], 0) >= item["level"] for item in branch):
                            return True
                    elif isinstance(branch, dict):
                        if owned_map.get(branch["name"], 0) >= branch["level"]:
                            return True
                return False

        valid_entries = []
        total = len(potential_priority_raw)

        for index, raw in enumerate(potential_priority_raw):
            # 检查 condition，不满足则跳过
            condition = raw.get("condition", [])
            # condition 为 dict（文档示例中有误用 {} 的情况）时，视为空条件直接通过
            if isinstance(condition, dict):
                condition = list(condition.values()) if condition else []
            if not _check_condition(condition):
                continue

            # 统一 name 为 list
            names = raw["name"] if isinstance(raw["name"], list) else [raw["name"]]
            level_span = raw.get("level_span", 1)
            max_level = raw.get("max_level", 6)

            # index 越小优先级越高，映射为数值：total - index
            priority = total - index

            valid_entries.append({
                "names": names,
                "level_span": level_span,
                "max_level": max_level,
                "priority": priority,
            })

        return valid_entries

    @staticmethod
    def _get_potential_priority(potential_priority_list, potential) -> int:
        """
            获取潜能的优先级

            遍历 priority_list，找到所有名称匹配且满足 level_span / max_level 条件的条目，
            返回其中优先级最高的值。
            当多个同名潜能处于同一条目（name 为 list）且 level_span 相同时，
            由调用方在拿到所有候选的优先级后随机决策；
            本函数只负责返回该 potential 实际能得到的最高优先级分数。

            Args:
                potential_priority_list: _parse_priority_raw_list 的返回值，
                    每个元素为 {"names": list, "level_span": int, "max_level": int, "priority": int}
                potential: 单个待选潜能，{"name": str, "old_level": int, "new_level": int}

            Returns:
                int: 优先级数值，数值越大优先级越高；匹配不到任何规则时返回 -1
        """
        name = potential["name"]
        old_level = potential["old_level"]
        new_level = potential["new_level"]
        level_span = new_level - old_level

        best_priority = -1

        for entry in potential_priority_list:
            # 名称不匹配则跳过
            if name not in entry["names"]:
                continue

            # max_level 条件：old_level < max_level 时生效
            if old_level >= entry["max_level"]:
                continue

            # level_span 条件：实际跨度 >= 要求跨度时生效
            if level_span < entry["level_span"]:
                continue

            # 取所有匹配条目中优先级最高的
            if entry["priority"] > best_priority:
                best_priority = entry["priority"]

        return best_priority