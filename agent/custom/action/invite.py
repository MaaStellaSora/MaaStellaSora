import difflib
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger

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

        # 开启debug模式
        # logger.debug_mode()

        # 邀约对象的任务列表
        invite_nodes = ["邀约_1号", "邀约_2号", "邀约_3号", "邀约_4号", "邀约_5号"]

        for node in invite_nodes:
            trekker_name, choose_gift = self._get_trekker_info(context, node)
            if not trekker_name:
                self.logger.debug(f"节点'{node}'的邀约对象为空，跳过")
                continue

            # 标记是否需要手动重置位置
            need_reset = True
            # 执行邀约流程
            while not context.tasker.stopping:
                if self._click_trekker(context, trekker_name):
                    # 成功点击邀约对象后，按照choose_gift情况修改送礼流程，然后尝试执行邀约
                    self._change_choose_gift_pipeline(context, choose_gift)
                    res = context.run_task("未邀约")

                    # 只有当结果存在且状态为 succeeded 时，才认为邀约成功。成功时不需要手动重置位置
                    if res and res.status.succeeded:
                        need_reset = False

                    break # 无论任务结果如何，只要点到了人，就停止向下翻页

                # 没找到则滑向下一页，若已到底部则放弃寻找
                if self._scroll_to_next_page(context):
                    break

            # 兜底处理：如果需要手动重置位置，则滚动到顶部
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
        threshold = 0.84 # 由于pipeline的only_rec有bug，所以只能在这里设置阈值
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
        results = self._get_refined_merge(reco_detail.all_results, threshold)
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
    def _get_refined_merge(results, threshold, y_tolerance = 30, x_tolerance = 50):
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

        # 过滤掉低于识别阈值的结果
        results = [r for r in results if r.score >= threshold]
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

    def _change_choose_gift_pipeline(self, context: Context, choose_gift: str):
        """
            根据choose_gift修改送礼流程

            Args:
                context: maa.context.Context
                choose_gift: 送礼选项，只有"all"、"favorite"、"no"三种

            Returns:
                bool: 是否成功修改送礼流程
        """
        if choose_gift == "all":
            context.override_pipeline({
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
            })
            context.override_next("邀约_送礼流程", [
                "邀约_选择礼物成功",
                "[JumpBack]邀约_选择礼物"
            ])
        elif choose_gift == "favorite":
            context.override_pipeline({
                "邀约_选择礼物":{
                    "recognition":{
                        "param":{
                            "template":[
                                "Invite/邀约_喜好图标.png"
                            ]
                        }
                    }
                }
            })
            context.override_next("邀约_送礼流程", [
                "邀约_选择礼物成功",
                "[JumpBack]邀约_选择礼物"
            ])
        elif choose_gift == "no":
            context.override_next("邀约_送礼流程",["邀约_还是算了"])
        else:
            self.logger.error(f"未知的送礼选项：{choose_gift}")
            return False
        return True