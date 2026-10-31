import difflib
import time
from typing import Never

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger

@AgentServer.custom_action("InviteAuto")
class InviteAuto(CustomAction):
    def __init__(self):
        # 导入logger
        super().__init__()
        self.logger = logger.get_logger(__name__)

    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        """
            邀约功能总控制节点
        """
        if context.tasker.stopping:
            return False

        # 邀约对象的任务列表
        invite_nodes = ["邀约_1号", "邀约_2号", "邀约_3号", "邀约_4号", "邀约_5号"]

        # 自动查找配置（由设置界面的 switch 通过 attach 注入）
        inviter_node_data = context.get_node_data(argv.node_name) or {}
        attach = inviter_node_data.get("attach", {})
        auto_find = bool(attach.get("auto_find"))

        # 组装待邀约名单：元素 (名字, 送礼类型)
        queue = []

        if auto_find:
            # 检查邀约对象是否达到上限
            if self._hit_daily_limit(context):
                return True
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
                self.logger.info(
                    "$agent.custom.action.invite.run.duplicate_invitee_skipped",
                    {
                        "trekker_name": trekker_name,
                    },
                )
                continue
            selected.add(self._normalize_name(trekker_name))
            queue.append((trekker_name, choose_gift))

        # 每日上限 5 人
        queue = queue[:5]
        self.logger.info(
            "$agent.custom.action.invite.run.invite_list",
            {
                "value": ", ".join([name for name, _ in queue]),
            },
        )

        for trekker_name, choose_gift in queue:
            if context.tasker.stopping:
                return False
            # 检查邀约对象是否达到上限
            if self._hit_daily_limit(context):
                return True

            # 标记是否需要手动重置位置
            need_reset = False
            # 执行邀约流程
            while not context.tasker.stopping:
                if self._click_trekker(context, trekker_name):
                    # 成功点击邀约对象后，按照choose_gift情况获取送礼流程，然后尝试执行邀约
                    pipeline_override = self._get_choose_gift_pipeline(choose_gift)
                    if context.tasker.stopping:
                        return False
                    # 同一上下文的 max_hit 会累计，每位角色开始邀约前重置对话循环计数
                    context.clear_hit_count("邀约_推进对话")
                    context.run_task("邀约_开始邀约", pipeline_override)

                    break # 无论任务结果如何，只要点到了人，就停止向下翻页

                # 没找到则滑向下一页，若已到底部则放弃查找
                is_bottom = self._scroll_to_next_page(context)
                if is_bottom:
                    break
                else:
                    need_reset = True

            # 检查邀约对象是否达到上限
            if self._hit_daily_limit(context):
                return True

            # 邀约流程完成后，如果需要手动重置位置，则滚动到顶部
            if need_reset:
                self._scroll_to_top(context)

            # 检测任务中止的情况，防止卡死，检测成功时结束函数
            if context.tasker.stopping:
                return False
        # 返回True，执行后续的“通用_返回主页”节点
        return True

    def _hit_daily_limit(self, context: Context) -> bool:
        """识别当日邀约次数是否已达上限。"""
        if context.tasker.stopping:
            return False
        image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("邀约_达上限", image)
        if reco_detail and reco_detail.hit:
            self.logger.info(
                "$agent.custom.action.invite._hit_daily_limit.daily_invite_limit_reached"
            )
            return True
        return False

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
        trekker_info = context.get_node_data(node) or {}

        try:
            trekker_name = trekker_info['recognition']['param']['expected'][0]
            trekker_name = trekker_name.strip()
            choose_gift = trekker_info['attach']['gift']
        except (TypeError, KeyError, IndexError, AttributeError) as e:
            self.logger.warning(
                "$agent.custom.action.invite._get_trekker_info.warning",
                {
                    "node": node,
                    "e": e,
                },
            )
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
                bool: 找到对象返回True，未找到返回False
        """
        # 参数
        similarity_limit = 0.75  # 文本相似度阈值

        # 处理旅人名字的文本问题，把全角括号都换成半角括号，把空格都取消
        formatted_name = self._normalize_name(trekker_name)
        # 文本长度差阈值
        max_allowed_diff = 1 if len(formatted_name) <= 7 else 2

        results = self._recognize_invitees(context)
        self.logger.debug(f"筛选出{len(results)}个结果：{results}，开始比较")

        # 比较文本相似程度，如果相似程度高，则点击，并返回True，否则返回False
        for result in results:
            # 使用difflib库计算文本相似度
            formatted_result = self._normalize_name(result['text'])
            similarity = difflib.SequenceMatcher(None, formatted_result, formatted_name).ratio()
            # 计算文本长度差
            len_diff = abs(len(formatted_result) - len(formatted_name))
            # 检查首尾是否匹配
            is_first_match = formatted_result[0] == formatted_name[0]
            is_last_match = formatted_result[-1] == formatted_name[-1]

            if similarity >= similarity_limit and len_diff <= max_allowed_diff and (is_first_match or is_last_match):
                if context.tasker.stopping:
                    return False
                self.logger.debug(f"识别成功！预期: {formatted_name}, 识别结果: {formatted_result},"
                                  f" 相似度: {similarity:.2f}")
                context.tasker.controller.post_click(result['x'], result['y']).wait()
                self.logger.debug(f"点击坐标{result['x']},{result['y']}完成")
                return True
            self.logger.debug(f"识别失败！预期: {formatted_name}, 识别结果: {formatted_result},"
                              f" 相似度: {similarity:.2f}")
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
                   and not (r.text.startswith('P') and (not r.text.isascii() or r.text.isupper()))]
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

    def _recognize_invitees(self, context: Context, image=None) -> list[dict]:
        """识别并整理左侧邀约对象，可复用调用方已有截图。"""
        if image is None:
            image = context.tasker.controller.post_screencap().wait().get()

        reco_detail = context.run_recognition("邀约_左方识别邀约对象", image)
        if reco_detail is None:
            self._raise_invitee_recognition_error()

        self.logger.debug(f"左方邀约对象识别结果: {reco_detail.all_results}")
        results = self._get_refined_merge(reco_detail.all_results)
        if not results:
            self._raise_invitee_recognition_error()
        return results

    def _raise_invitee_recognition_error(self) -> Never:
        self.logger.error(
            "$agent.custom.action.invite._recognize_invitees.recognition_failed"
        )
        raise RuntimeError("Invitee recognition failed")

    def _scroll_to_next_page(self, context: Context) -> bool:
        """
            向下滑动到下一页

            Args:
                context: maa.context.Context

            Returns:
                bool: 已到底返回True，未到底返回False
        """
        if context.tasker.stopping:
            return True
        before = self._recognize_invitees(context)

        context.run_task("邀约_向下滑动")

        after = self._recognize_invitees(context)

        last_trekker_before = before[-1]['text']
        last_trekker_after = after[-1]['text']
        self.logger.debug(f"check if bottom: '{last_trekker_before}' '{last_trekker_after}'")
        if self._normalize_name(last_trekker_before) == self._normalize_name(last_trekker_after):
            self.logger.debug(f"已滑动到底部")
            return True
        self.logger.debug(f"未滑动到底部")
        return False

    def _scroll_to_top(self, context: Context) -> bool:
        """
            向上滑动到顶部

            Args:
                context: maa.context.Context

            Returns:
                bool: 已滑到顶部返回True，其他情况返回False
        """
        while not context.tasker.stopping:
            before = self._recognize_invitees(context)

            context.run_task("邀约_向上滑动")

            after = self._recognize_invitees(context)

            first_trekker_before = before[0]['text']
            first_trekker_after = after[0]['text']
            self.logger.debug(f"check if top: '{first_trekker_before}' '{first_trekker_after}'")
            if self._normalize_name(first_trekker_before) == self._normalize_name(first_trekker_after):
                self.logger.debug(f"已滑动到顶部")
                return True

        return False

    @staticmethod
    def _normalize_name(name: str) -> str:
        """统一全角括号并移除半角、全角空格。"""
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

        self.logger.info(
            "$agent.custom.action.invite._auto_find_uncollected.started",
            {
                "max_find_count": max_find_count,
            },
        )

        while not context.tasker.stopping:
            results = self._recognize_invitees(context)

            for r in results:
                if context.tasker.stopping:
                    return pending
                key = self._normalize_name(r['text'])
                if key in seen:
                    # 跨半页翻页重复出现的角色，跳过，避免重复点击/重复记录
                    continue
                seen.add(key)

                if len(pending) >= max_find_count:
                    return pending

                context.tasker.controller.post_click(r['x'], r['y']).wait()
                time.sleep(0.5)
                if context.tasker.stopping:
                    return pending

                image = context.tasker.controller.post_screencap().wait().get()
                hit = context.run_recognition("邀约_未收集回忆", image)
                if hit and hit.hit:
                    pending.append(r['text'])
                    self.logger.info(
                        "$agent.custom.action.invite._auto_find_uncollected.uncollected_memory_found",
                        {
                            "text": r['text'],
                        },
                    )

            # 一轮点完，向下翻页，已到底部则停止
            if self._scroll_to_next_page(context):
                break

        return pending

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
            self.logger.error(
                "$agent.custom.action.invite._get_choose_gift_pipeline.unknown_gift_option",
                {
                    "choose_gift": choose_gift,
                },
            )
            return {}
        return pipeline_override
