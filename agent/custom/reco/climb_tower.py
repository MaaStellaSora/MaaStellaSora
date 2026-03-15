import re
import time
import random

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

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
        """自动选择潜能的主流程。

        读取 attach 参数与已拥有潜能状态，识别当前三张潜能卡片，
        根据自定义优先级列表选出最优潜能并返回其 box；
        不满足条件时执行刷新，最终兜底识别系统推荐图标。
        选择完成后将已拥有潜能状态写回节点 attach 持久化。

        Args:
            context: maa.context.Context
            argv: CustomRecognition.AnalyzeArg，含当前截图与节点名

        Returns:
            CustomRecognition.AnalyzeResult: 命中区域为目标潜能卡片的 box
        """
        node_name = argv.node_name
        attach = self._get_attachments(context, node_name)
        priority_list = self._parse_priority_raw_list(
            attach["priority_list"],
            attach["owned_potentials"],
        )

        refresh_count = 0
        if self._is_refreshable(context, argv.image):
            current_coin = self._get_current_coin(context, argv.image)
            refresh_cost = self._get_refresh_cost(context, argv.image)
            usable_coin = max(0, current_coin - attach["reserve_coin"])
            affordable = usable_coin // refresh_cost if refresh_cost else 0
            refresh_count = min(attach["max_refresh_count"], affordable)

        target_potential = None
        target_trekker = None

        while True:
            image = context.tasker.controller.post_screencap().wait().get()
            available = self._get_available_potentials(context, image)
            result = self._select_best_potential(available, priority_list)

            if result:
                target_potential, target_trekker = result
                break
            if refresh_count > 0:
                context.run_task("星塔_通用_点击刷新_agent")
                refresh_count -= 1
            else:
                break

        if target_potential:
            target_box = target_potential["box"]
        else:
            target_box = self._get_recommended_box(context, argv.image)

        owned = self._update_owned_potentials(
            attach["owned_potentials"],
            target_potential,
            target_trekker,
        )
        self._save_state(context, node_name, owned)

        return CustomRecognition.AnalyzeResult(box=target_box, detail={})

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
                potential_name = "".join(r.text for r in reco_detail.filtered_results)
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
    def _parse_priority_raw_list(
        potential_priority_raw: list[dict],
        owned_potentials: dict,
    ) -> list[dict]:
        """对原始 priority_list 进行初筛，过滤掉 condition 当前不满足的规则。

        排名使用原始 JSON 的 1-based 行号（index + 1），数值越小排名越高。
        condition 不满足的条目直接跳过，其行号仍保留在原始位置，不影响其他条目的排名。

        Args:
            potential_priority_raw: 原始优先级列表，每个元素结构：
                {
                    "trekker": str,         # 可选，潜能归属角色名
                    "potential": str|list,  # 必填，潜能名称或名称列表
                    "level_span": int,      # 可选，默认 1，最小升级跨度
                    "max_level": int,       # 可选，默认 MAX_POTENTIAL_LEVEL，旧等级上限（不含）
                    "condition": list       # 可选，生效条件，元素为 dict 时 AND，为 list 时 OR
                }
            owned_potentials: 已拥有潜能状态，按 trekker 分组，结构：
                {"花原": {"飞花乱坠": 1}, "unknown": {"盛大尾奏": 1}}

        Returns:
            list[dict]: 通过初筛的规则列表，每个元素结构：
                {
                    "trekker": str | None,  # 归属角色
                    "names": list[str],     # 目标潜能名称列表
                    "level_span": int,      # 最小升级跨度
                    "max_level": int,       # 旧等级上限（不含）
                    "priority": int         # 原始 JSON 的 1-based 行号，越小排名越高
                }
        """
        owned_map: dict[str, int] = {}
        for potentials in owned_potentials.values():
            owned_map.update(potentials)

        trekker_count_map: dict[str, int] = {
            trekker: len(potentials)
            for trekker, potentials in owned_potentials.items()
            if trekker != "unknown"
        }

        def _check_single_condition(item: dict) -> bool:
            """检查单个 condition 子项是否满足。"""
            if "trekker_count" in item:
                return trekker_count_map.get(item["trekker"], 0) >= item["trekker_count"]
            current = owned_map.get(item["potential"], 0)
            return (
                item.get("min_level", 0)
                <= current
                < item.get("max_level", ChoosePotentialRecognition.MAX_POTENTIAL_LEVEL + 1)
            )

        def _check_condition(cond: list) -> bool:
            """检查 condition 列表是否满足。

            元素全为 dict 时为 AND 逻辑；含 list 元素时为 OR 逻辑（内层为 AND）。
            """
            if not cond:
                return True
            if all(isinstance(item, dict) for item in cond):
                return all(_check_single_condition(item) for item in cond)
            for branch in cond:
                if isinstance(branch, list):
                    if all(_check_single_condition(item) for item in branch):
                        return True
                elif isinstance(branch, dict):
                    if _check_single_condition(branch):
                        return True
            return False

        valid_entries = []
        for index, raw in enumerate(potential_priority_raw):
            condition = raw.get("condition", [])
            if isinstance(condition, dict):
                condition = []
            if not _check_condition(condition):
                continue

            potential = raw["potential"]
            names = potential if isinstance(potential, list) else [potential]

            valid_entries.append({
                "trekker": raw.get("trekker"),
                "names": names,
                "level_span": raw.get("level_span", 1),
                "max_level": raw.get("max_level", ChoosePotentialRecognition.MAX_POTENTIAL_LEVEL),
                "priority": index + 1,
            })

        return valid_entries

    @staticmethod
    def _match_potential_name(ocr_name: str, rule_name: str) -> bool:
        """比较 OCR 识别的潜能名称与规则中的潜能名称是否匹配。

        OCR 可能产生前后漏字或噪声字符，因此先清洗两边字符串（去除非 Unicode
        字母数字字符），再用双向 in 检查，覆盖前后漏字的情况。

        中间漏字或错字无法处理，属于 OCR 识别质量问题。

        Args:
            ocr_name: OCR 识别到的潜能名称
            rule_name: 优先级规则中定义的潜能名称

        Returns:
            bool: 两者匹配时返回 True
        """
        cleaned_ocr = re.sub(r'\W', '', ocr_name)
        cleaned_rule = re.sub(r'\W', '', rule_name)
        return cleaned_ocr in cleaned_rule or cleaned_rule in cleaned_ocr

    @staticmethod
    def _get_potential_priority(
        potential_priority_list: list[dict],
        potential: dict,
    ) -> tuple[int, str | None]:
        """获取单个待选潜能在规则列表中的最高排名及其 trekker 归属。

        遍历 priority_list，找到所有名称匹配且满足 level_span / max_level 条件的规则，
        返回排名数值最小（即优先级最高）的规则对应的排名与 trekker。

        Args:
            potential_priority_list: _parse_priority_raw_list 的返回值，每个元素结构：
                {"trekker": str|None, "names": list, "level_span": int,
                 "max_level": int, "priority": int}
            potential: 单个待选潜能，结构：
                {"name": str, "old_level": int, "new_level": int, "is_core": bool, "box": list}

        Returns:
            tuple[int, str | None]: (priority, trekker)
                priority 为匹配到的最小排名数值；无匹配时返回 -1
                trekker 为对应规则的归属角色；无匹配时返回 None
        """
        name = potential["name"]
        old_level = potential["old_level"]
        level_span = potential["new_level"] - old_level

        best_priority = -1
        best_trekker = None

        for entry in potential_priority_list:
            if not any(
                ChoosePotentialRecognition._match_potential_name(name, n)
                for n in entry["names"]
            ):
                continue
            if old_level >= entry["max_level"]:
                continue
            if level_span < entry["level_span"]:
                continue
            if best_priority == -1 or entry["priority"] < best_priority:
                best_priority = entry["priority"]
                best_trekker = entry["trekker"]

        return best_priority, best_trekker

    def _select_best_potential(
        self,
        available: list[dict],
        priority_list: list[dict],
    ) -> tuple[dict, str | None] | None:
        """从待选潜能中选出排名最高的一个，返回潜能与其 trekker 归属。

        同时输出每个待选潜能的识别情况及最终选择的 info 日志。
        若多个潜能匹配到同一条规则（potential 为列表且跨度相同），从并列者中随机选一个。

        Args:
            available: _get_available_potentials 的返回值
            priority_list: _parse_priority_raw_list 的返回值

        Returns:
            tuple[dict, str | None]: (potential, trekker) 若找到匹配规则的潜能；
            None 若所有潜能均无匹配规则
        """
        candidates: list[tuple[dict, int, str | None]] = []
        for potential in available:
            priority, trekker = self._get_potential_priority(priority_list, potential)
            rank_str = str(priority) if priority != -1 else "无"

            if potential["is_core"]:
                self.logger.info(f"[潜能识别] {potential['name']} | 核心潜能 | 排名 {rank_str}")
            else:
                old = potential["old_level"]
                new = potential["new_level"]
                self.logger.info(f"[潜能识别] {potential['name']} | 等级 {old}→{new} | 排名 {rank_str}")

            if priority != -1:
                candidates.append((potential, priority, trekker))

        if not candidates:
            self.logger.info("[潜能选择] 选择系统推荐")
            return None

        best_priority = min(p for _, p, _ in candidates)
        top = [(pot, trek) for pot, p, trek in candidates if p == best_priority]

        best_span = max(pot["new_level"] - pot["old_level"] for pot, _ in top)
        top = [(pot, trek) for pot, trek in top if pot["new_level"] - pot["old_level"] == best_span]

        selected_potential, selected_trekker = random.choice(top)

        self.logger.info(f"[潜能选择] {selected_potential['name']} | 排名 {best_priority}")
        return selected_potential, selected_trekker

    def _get_recommended_box(self, context: Context, image) -> list:
        """识别系统推荐图标，返回对应卡片的 box。

        推荐图标位于卡片 box 范围外，通过计算图标命中 x 坐标与各卡片
        general_potential box 的 x 距离，取最近者对应的卡片。
        识别失败时返回第一张卡片的 box 作为兜底。

        Args:
            context: maa.context.Context
            image: 截图

        Returns:
            list: 目标卡片区域 [x, y, w, h]
        """
        reco_detail = context.run_recognition(
            "星塔_节点_选择潜能_识别推荐图标_agent", image
        )
        if reco_detail and reco_detail.hit:
            hit_x = reco_detail.best_result.box[0]
            closest = min(
                self.POTENTIAL_ROIS,
                key=lambda r: abs(r["general_potential"][0] - hit_x)
            )
            return closest["general_potential"]

        self.logger.warning("推荐图标识别失败，返回第一张卡片")
        return self.POTENTIAL_ROIS[0]["general_potential"]

    def _update_owned_potentials(
        self,
        owned: dict,
        potential: dict | None,
        trekker: str | None,
    ) -> dict:
        """将本次选中的潜能写入 owned_potentials 对应的 trekker 分组。

        Args:
            owned: 当前 owned_potentials 字典，按 trekker 分组
            potential: 本次选中的潜能（含 name, new_level）；为 None 时不更新
            trekker: 归属角色名；为 None 时写入 "unknown" 分组

        Returns:
            dict: 更新后的 owned_potentials
        """
        if potential is None:
            return owned

        name = potential["name"]
        new_level = potential["new_level"]

        if new_level == -1:
            self.logger.warning(f"潜能 {name} 等级解析失败，将默认为1级")
            new_level = 1

        group = trekker if trekker else "unknown"
        if not trekker:
            self.logger.debug(
                f"潜能 {name} 无所属旅人，将默认为unknown"
            )

        if group not in owned:
            owned[group] = {}
        owned[group][name] = new_level
        return owned

    def _save_state(
        self,
        context: Context,
        node_name: str,
        owned: dict,
    ) -> None:
        """将更新后的 owned_potentials 写回节点 attach，通过 override_pipeline 持久化。

        写回失败时记录 ERROR 日志，不中断流程（本次状态丢失）。

        Args:
            context: maa.context.Context
            node_name: 当前节点名称
            owned: 更新后的 owned_potentials
        """
        new_attach = {"owned_potentials": owned}
        success = context.override_pipeline({node_name: {"attach": new_attach}})
        if not success:
            self.logger.error("保存当前拥有潜能失败，自定义潜能优先级可能无法正常工作")