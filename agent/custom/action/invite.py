import difflib
import json
import time
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger

# 角色 -> 特殊选项文本列表，用于回忆收集模式。配置与本文件同目录。
_SPECIAL_OPTIONS_PATH = Path(__file__).resolve().parent / "invite_special_options.json"


def _load_special_options() -> dict[str, list[str]]:
    """从配置文件读取角色 -> 特殊选项映射，读取失败时返回空映射。"""
    try:
        with open(_SPECIAL_OPTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {
            str(name): [str(opt) for opt in options]
            for name, options in data.items()
            if isinstance(options, list)
        }
    except (OSError, json.JSONDecodeError) as exc:
        logger.get_logger("invite").warning(
            f"读取特殊选项配置失败，将不启用特殊选项：{exc}"
        )
        return {}


@AgentServer.custom_action("InviteAuto")
class InviteAuto(CustomAction):
    def __init__(self):
        # 导入logger
        super().__init__()
        self.logger = logger.get_logger()

    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        """
            邀约功能总控制节点
        """

        # 邀约对象的任务列表
        invite_nodes = ["邀约_1号", "邀约_2号", "邀约_3号", "邀约_4号", "邀约_5号"]

        # 是否开启回忆收集模式（由设置界面的 switch 通过 attach 注入）
        inviter_node_data = context.get_node_data(argv.node_name) or {}
        attach = inviter_node_data.get("attach", {})
        memory_mode = bool(attach.get("memory_mode"))
        auto_find = bool(attach.get("auto_find"))

        # 组装待邀约名单：元素 (名字, 送礼类型)
        queue = []

        if auto_find:
            max_find_count = int(attach.get("max_find_count", "5") or "5")
            auto_gift = attach.get("auto_gift", "all") or "all"

            names = self._auto_find_uncollected(context, max_find_count)
            for n in names:
                queue.append((n, auto_gift))
            # 自动查找后重置回顶部，衔接下方 _click_trekker “从顶部开始找”的前提
            self._scroll_to_top(context)

        # 追加用户手填名字（不重复，保持原有 1~5 号顺序）
        # 去重统一按规范化后的名字比较，避免全角/半角括号、空格差异导致重复邀约
        selected = {self._normalize_name(t) for t, _ in queue}
        for node in invite_nodes:
            trekker_name, choose_gift = self._get_trekker_info(context, node)
            if not trekker_name or trekker_name in ("x", "X"):
                continue
            if self._normalize_name(trekker_name) in selected:
                self.logger.info(f"邀约对象 '{trekker_name}' 已由自动查找记录，跳过手填重复项")
                continue
            selected.add(self._normalize_name(trekker_name))
            queue.append((trekker_name, choose_gift))

        # 每日上限 5 人
        queue = queue[:5]
        self.logger.info(f"邀约名单：{", ".join([name for name, _ in queue])}")

        for trekker_name, choose_gift in queue:
            # 检查邀约对象是否达到上限
            image = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition("邀约_达上限", image)
            if reco_detail and reco_detail.hit:
                self.logger.info(f"邀约次数已达到本日上限")
                return True

            # 标记是否需要手动重置位置
            need_reset = False
            # 执行邀约流程
            while not context.tasker.stopping:
                if self._click_trekker(context, trekker_name):
                    # 成功点击邀约对象后，按照choose_gift情况获取送礼流程，然后尝试执行邀约
                    pipeline_override = self._get_choose_gift_pipeline(choose_gift)
                    if memory_mode:
                        pipeline_override = self._merge_memory_override(
                            pipeline_override, trekker_name
                        )
                    res = context.run_task("邀约_开始邀约", pipeline_override)

                    # 成功识别到邀约按钮时，不需要手动重置位置
                    if res and res.status.succeeded:
                        need_reset = False

                    break # 无论任务结果如何，只要点到了人，就停止向下翻页

                # 没找到则滑向下一页，若已到底部则放弃查找
                is_bottom = self._scroll_to_next_page(context)
                if is_bottom:
                    break
                else:
                    need_reset = True

            # 邀约流程完成后，如果需要手动重置位置，则滚动到顶部
            if need_reset:
                self._scroll_to_top(context)

            # 检测任务中止的情况，防止卡死，检测成功时结束函数
            if context.tasker.stopping:
                return False
        # 返回True，执行后续的“通用_返回主页”节点
        return True

    def _get_trekker_info(self, context: Context, node) -> tuple[str, str]:
        """
            获取邀约对象名字及送礼选项

            Args:
                context: maa.context.Context
                node: string，需要提取内容的节点名称

            Returns:
                str: 邀约对象名字
                str: 送礼选项
        """
        trekker_info = context.get_node_data(node)

        try:
            trekker_name = trekker_info['recognition']['param']['expected'][0]
            trekker_name = trekker_name.strip()
            choose_gift = trekker_info['attach']['gift']
        except (TypeError, KeyError, IndexError, AttributeError) as e:
            self.logger.warning(f"提取节点'{node}'的文本过程中出现问题: {e}")
            trekker_name = ""
            choose_gift = ""

        return trekker_name, choose_gift

    def _click_trekker(
            self,
            context: Context,
            trekker_name: str
    ) -> bool:
        """
            识别并点击邀约对象

            Args:
                context: maa.context.Context
                trekker_name: 旅人名字

            Returns:
                bool: 选择到目标对象时返回True，未能选择到目标对象时返回False
        """
        # 参数
        similarity_limit = 0.8 # 文本相似度阈值

        # 处理旅人名字的文本问题，把全角括号都换成半角括号，把空格都取消
        translate_table = str.maketrans({
            '（': '(',
            '）': ')',
            ' ': None,
            '　': None
        })
        formatted_name = trekker_name.translate(translate_table)

        # 识别对象
        image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("邀约_左方识别邀约对象", image)

        # 整理识别结果
        results = self._get_refined_merge(reco_detail.all_results)
        self.logger.debug(f"识别出{len(results)}个结果，开始比较")

        # 比较文本相似程度，如果相似程度高，则点击，并返回True，否则返回False
        for result in results:
            # 使用difflib库计算文本相似度
            formatted_result = result['text'].translate(translate_table)
            similarity = difflib.SequenceMatcher(None, formatted_result, formatted_name).ratio()

            if similarity >= similarity_limit:
                self.logger.debug(f"识别成功！预期: {formatted_name}, 识别结果: {formatted_result}, 相似度: {similarity:.2f}")
                context.tasker.controller.post_click(result['x'], result['y']).wait()
                self.logger.debug(f"点击坐标{result['x']},{result['y']}完成")
                return True
            self.logger.debug(f"识别失败！预期: {formatted_name}, 识别结果: {formatted_result}, 相似度: {similarity:.2f}")
        return False

    @staticmethod
    def _get_refined_merge(results, threshold = 0.7, y_tolerance = 30, x_tolerance = 50):
        """
            处理OCR识别结果，将符合条件的文本块进行合并，并计算最终的点击位置。

            Args:
                results (list): ocr检测结果的列表，每个元素应有 text 和 score 和 box 属性。
                threshold (float): 识别分数阈值，用于过滤识别分数过低的结果
                x_tolerance (int): X 轴方向允许的最大距离，用于判断两个文本框是否属于同一格。
                y_tolerance (int): Y 轴方向允许的最大距离，用于判断当前文本框是否属于同一格。

            Returns:
                list: 合并后的文本及其对应点击坐标的字典列表，每个字典包含 'text'、'x' 和 'y' 键。
        """

        # 排除掉没有识别结果的情况
        if not results:
            return []

        # 过滤掉低于识别阈值的结果，或以P开头的乱码结果
        results = [r for r in results if r.score >= threshold
                   and not (r.text.startswith('P') and not r.text.isascii())
                   and r.text != 'P' and r.text != 'PI']
        # 按 Y 坐标排序，确保从上往下处理
        results.sort(key=lambda r: r.box[1])

        merged_list = []
        for item in results:
            x, y, w, h = item.box
            cx, cy = x + w // 2, y + h // 2

            found = False
            for m in merged_list:
                # 逻辑：X轴距离在格子范围内，且当前块顶部靠近上一个块的底部
                if abs(m['x_ref'] - x) <= x_tolerance and abs(y - m['y_end']) <= y_tolerance:
                    m['text'] += item.text
                    # 简单合并坐标并取整
                    m['x'] = (m['x'] + cx) // 2
                    m['y'] = (m['y'] + cy) // 2
                    m['y_end'] = y + h  # 更新底部边界供下一次合并参考
                    found = True
                    break

            if not found:
                # 没能合并时，创建为新的元素
                merged_list.append({
                    'text': item.text,
                    'x': cx,
                    'y': cy,
                    'x_ref': x,  # 辅助字段：记录起始X
                    'y_end': y + h  # 辅助字段：记录当前底部Y
                })

        # 返回前可以清理掉辅助字段，只保留要的三个键
        return [{'text': i['text'], 'x': i['x'], 'y': i['y']} for i in merged_list]

    def _scroll_to_next_page(self, context: Context, image=None):
        """
            向下滑动到下一页

            Args:
                context: maa.context.Context

            Returns:
                bool: 已滑到底部或无法判断是否划到底部时，返回True；未滑到底部时，返回False
        """
        if not image:
            image = context.tasker.controller.post_screencap().wait().get()

        if not context.override_image("invite_scroll_down_template", image):
            self.logger.error("截图错误，将无法判断是否滑动到底部")
            return True

        context.run_task("邀约_向下滑动")

        image = context.tasker.controller.post_screencap().wait().get()
        reco_result = context.run_recognition("邀约_已滑动到底部", image)
        if reco_result and len(reco_result.all_results) > 0:
            self.logger.debug(f"向下滑动识别分数：{reco_result.all_results[0].score}")
        if reco_result and reco_result.hit:
            self.logger.debug(f"已滑动到底部")
            return True
        else:
            self.logger.debug(f"未滑动到底部")
            return False

    def _scroll_to_top(self, context: Context):
        """
            向上滑动到顶部

            Args:
                context: maa.context.Context

            Returns:
                bool:
                    已滑到顶部时，返回True；
                    未滑到顶部，无法判断是否滑到顶部，又或者任务被中止时，返回False
        """
        image = context.tasker.controller.post_screencap().wait().get()
        while True:
            if not context.override_image("invite_scroll_up_template", image):
                self.logger.error("截图错误，将无法判断是否滑动到顶部")
                return False

            context.run_task("邀约_向上滑动")

            image = context.tasker.controller.post_screencap().wait().get()
            reco_result = context.run_recognition("邀约_已滑动到顶部", image)
            if reco_result and len(reco_result.all_results) > 0:
                self.logger.debug(f"向上滑动识别分数：{reco_result.all_results[0].score}")
            if reco_result and reco_result.hit:
                self.logger.debug(f"已滑动到顶部")
                return True

            # 检测任务中止的情况，防止卡死，检测成功时返回False
            if context.tasker.stopping:
                return False

    @staticmethod
    def _normalize_name(name: str) -> str:
        """名称规范化，与 _click_trekker 的 translate_table 保持一致。"""
        translate_table = str.maketrans({
            '（': '(',
            '）': ')',
            ' ': None,
            '　': None
        })
        return name.translate(translate_table)

    def _auto_find_uncollected(self, context: Context, max_find_count: int) -> list[str]:
        """自动遍历左侧角色列表，找出尚有「未收集回忆」标记的角色。

        逐页 OCR 列表，逐角色点击进入详情态，再对右侧区域匹配「未收集回忆」标记，
        命中则记录角色名，直到查完全部角色或命中数达到上限。

        Args:
            context: maa.context.Context
            max_find_count: 最多查找人数，命中达此数量即提前退出

        Returns:
            list[str]: 命中的角色名（按列表顺序）
        """
        pending = []       # 命中「未收集」的角色名
        seen = set()       # 已点击处理过的角色名（规范化后），仅用于跨翻页去重

        self.logger.info(f"开始自动查找未收集回忆的角色，最多查找 {max_find_count} 人")

        while not context.tasker.stopping:
            image = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition("邀约_左方识别邀约对象", image)
            results = self._get_refined_merge(reco_detail.all_results)
            if not results:
                break

            for r in results:
                key = self._normalize_name(r['text'])
                if key in seen:
                    # 跨半页翻页重复出现的角色，跳过，避免重复点击/重复记录
                    continue
                seen.add(key)

                if len(pending) >= max_find_count:
                    return pending

                context.tasker.controller.post_click(r['x'], r['y']).wait()
                time.sleep(0.5)

                image = context.tasker.controller.post_screencap().wait().get()
                hit = context.run_recognition("邀约_未收集回忆", image)
                if hit and hit.hit:
                    pending.append(r['text'])
                    self.logger.info(f"发现未收集回忆的角色：{r['text']}")

            # 一轮点完，向下翻页，已到底部则停止
            if self._scroll_to_next_page(context):
                break

        return pending

    def _merge_memory_override(self, pipeline_override: dict, trekker_name: str) -> dict:
        """
            回忆收集模式下的额外 pipeline override：
            向回忆收集循环节点注入当前角色的特殊选项文本。
            （地点节点的 next 改道由设置界面的 switch 选项负责）

            Args:
                pipeline_override: 已有 pipeline override（送礼选项相关）
                trekker_name: 当前邀约对象名字

            Returns:
                dict: 合并后的 pipeline override
        """
        special_options = _load_special_options().get(trekker_name, [])
        attach_override = {
            "邀约_回忆收集_循环": {
                "attach": {
                    "special_options": list(special_options)
                }
            }
        }
        node = "邀约_回忆收集_循环"
        if node in pipeline_override:
            pipeline_override[node].update(attach_override[node])
        else:
            pipeline_override[node] = attach_override[node]

        return pipeline_override

    def _get_choose_gift_pipeline(self, choose_gift: str) -> dict:
        """
            根据choose_gift修改送礼流程
            pipeline默认是送最好的礼物

            Args:
                choose_gift: 送礼选项，只有"all"、"favorite"、"no"三种

            Returns:
                dict: 重置的pipeline配置
        """
        if choose_gift == "favorite":
            pipeline_override = {}
        elif choose_gift == "all":
            pipeline_override = {
                "邀约_选择礼物":{
                    "recognition":{
                        "param":{
                            "template":[
                                "Invite/邀约_喜好图标.png",
                                "Invite/邀约_喜好图标2.png",
                                "Invite/邀约_喜好图标3.png"
                            ]
                        }
                    }
                }
            }
        elif choose_gift == "no":
            pipeline_override = {
                "邀约_送礼流程": {
                    "next": [
                        "邀约_还是算了"
                    ]
                }
            }
        else:
            self.logger.error(f"未知的送礼选项：{choose_gift}，将默认只送黄色笑脸")
            return {}
        return pipeline_override


@AgentServer.custom_action("InviteMemory")
class InviteMemory(CustomAction):
    """邀约·回忆收集模式的自定义动作。

    逐句推进对话，遇到选项时优先选择角色特殊选项，否则兜底点首个选项，
    直到识别到「送出礼物」结束面板为止。

    结束面板检测（邀约_送礼 的 OCR）优先级最高；选项状态判定以
    `邀约_对话选项按钮.png` 模板匹配为准；对话推进点击点 [1000, 200]。
    """
    # 推进对话点击点
    ADVANCE_CLICK = (1000, 200)

    # 选项行 Y 坐标（下对齐，从 y=470 起向上铺）
    OPTION_ROWS = (470, 400, 330)
    OPTION_ROW_HEIGHT = 60
    OPTION_ROW_WIDTH = 450
    OPTION_ROW_X = 760

    # 特殊选项匹配阈值
    SIMILARITY_LIMIT = 0.85

    # 最大迭代次数，防止无选项推进时死循环
    MAX_ITERATIONS = 2000

    def __init__(self):
        super().__init__()
        self.logger = logger.get_logger("invite_memory")

    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        node_data = context.get_node_data(argv.node_name) or {}
        special_options = node_data.get("attach", {}).get("special_options", [])
        if not isinstance(special_options, list):
            special_options = []

        iteration = 0
        while not context.tasker.stopping:
            iteration += 1
            if iteration > self.MAX_ITERATIONS:
                self.logger.error("回忆收集循环次数超过上限，强制退出")
                return False

            image = context.tasker.controller.post_screencap().wait().get()

            # 1. 结束面板检测优先级最高
            if self._hit_gift_panel(context, image):
                self.logger.info("识别到送出礼物结束面板，退出回忆收集循环")
                return True

            # 2. 选项状态判定，优先于推进点击
            if self._in_option_state(context, image):
                if self._click_option(context, image, special_options):
                    continue
                # 未匹配到选项（或点击失败），兜底点首个选项
                self._click_first_option(context)
                continue

            # 3. 非选项状态，点推进点
            context.tasker.controller.post_click(*self.ADVANCE_CLICK).wait()

        return context.tasker.stopping is False

    def _hit_gift_panel(self, context: Context, image) -> bool:
        """识别「送出礼物」结束面板，复用 邀约_送礼 的 OCR 识别。"""
        reco_detail = context.run_recognition("邀约_送礼", image)
        return bool(reco_detail and reco_detail.hit)

    def _in_option_state(self, context: Context, image) -> bool:
        """以 `邀约_对话选项按钮.png` 模板匹配判定是否处于选项弹出状态。"""
        reco_detail = context.run_recognition("邀约_对话选项按钮", image)
        return bool(reco_detail and reco_detail.hit)

    def _click_option(self, context: Context, image, special_options: list[str]) -> bool:
        """逐行 OCR 选项文本，命中特殊选项则点击对应行，返回是否命中。"""
        if not special_options:
            return False

        for y in self.OPTION_ROWS:
            texts = self._ocr_row(context, image, y)
            row_text = "".join(texts)
            if not row_text:
                continue
            if self._match_special(row_text, special_options):
                self.logger.info(f"命中特殊选项：{row_text}，点击行 y={y}")
                context.tasker.controller.post_click(
                    self.OPTION_ROW_X + self.OPTION_ROW_WIDTH // 2,
                    y + self.OPTION_ROW_HEIGHT // 2,
                ).wait()
                return True
        return False

    def _click_first_option(self, context: Context) -> None:
        """兜底：点击首个（最下 y=470 那行）选项，保证对话继续。"""
        y = self.OPTION_ROWS[0]
        self.logger.info("未命中特殊选项，点击首个选项")
        context.tasker.controller.post_click(
            self.OPTION_ROW_X + self.OPTION_ROW_WIDTH // 2,
            y + self.OPTION_ROW_HEIGHT // 2,
        ).wait()

    def _ocr_row(self, context: Context, image, y: int) -> list[str]:
        """对指定选项行 ROI 做 OCR，返回识别到的文本列表。"""
        roi = [self.OPTION_ROW_X, y, self.OPTION_ROW_WIDTH, self.OPTION_ROW_HEIGHT]
        pipeline_override = {
            "邀约_读取选项文本": {
                "recognition": {
                    "param": {"roi": roi},
                }
            }
        }
        reco_detail = context.run_recognition("邀约_读取选项文本", image, pipeline_override)
        if not reco_detail or not reco_detail.all_results:
            return []
        return [r.text for r in reco_detail.all_results if r.text]

    @classmethod
    def _match_special(cls, row_text: str, special_options: list[str]) -> bool:
        """判断选项文本是否命中特殊选项，支持精确包含与模糊匹配。"""
        normalized = row_text.strip()
        for opt in special_options:
            opt = opt.strip()
            if not opt:
                continue
            if opt in normalized or normalized in opt:
                return True
            if difflib.SequenceMatcher(None, normalized, opt).ratio() >= cls.SIMILARITY_LIMIT:
                return True
        return False
