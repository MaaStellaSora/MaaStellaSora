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
            仅使用agent进行遍历与判断
            使用pipeline执行邀约流程

            TODO:
                实现拖曳选择角色（用context.override_image跟context.run_recognition_direct去做识别）
                个别送礼选项：笑脸，不限，不送
        """

        # 开启debug模式
        # logger.debug_mode()

        # 邀约对象的任务列表
        invite_nodes = ["邀约_1号", "邀约_2号", "邀约_3号", "邀约_4号", "邀约_5号"]

        for node in invite_nodes:
            self.logger.debug(f"开始分析节点'{node}'")
            # 获取邀约对象信息，并去掉前后的空格
            trekker_name = context.get_node_data(node)
            try:
                trekker_name = trekker_name['recognition']['param']['expected'][0]
                trekker_name = trekker_name.strip()
            except Exception:
                self.logger.warning(f"提取节点'{node}'的文本过程中出现问题，跳过")

            # 判断邀约对象是否为空，为空则跳过
            if trekker_name == "":
                self.logger.debug(f"节点'{node}'的邀约对象为空，跳过")
                continue

            # while True
            # 执行邀约流程
            # 如果成功识别并点击邀约对象，则开始邀约
            if self._select_trekker(context, trekker_name):
                context.run_task("未邀约") # 这里可以根据送礼选项进行pipeline_override，或者在选项里改也行
                # break

            # 如果识别失败，则向下滚动，如果滚动后发现是底部，则break


            # 检测任务中止的情况，防止卡死，检测成功时返回False
            if context.tasker.stopping:
                return False
        # 返回True，执行后续的“通用_返回主页”节点
        return True

    def _select_trekker(
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
        threshold = 0.84
        similarity_limit = 0.8 # 文本相似度阈值

        # 处理旅人名字的文本问题，把全角括号都换成半角括号，把空格都取消
        self.logger.debug(f"将文字'{trekker_name}'统一格式化")
        translate_table = str.maketrans({
            '（': '(',
            '）': ')',
            ' ': None,
            '　': None
        })
        formatted_name = trekker_name.translate(translate_table)

        # 识别对象
        self.logger.debug(f"已格式化为'{formatted_name}'，开始识别")
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

@AgentServer.custom_action("InviteAutoo")
class InviteAutoo(CustomAction):
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        context.tasker.controller.post_swipe(1,2,3,4,5)

        return True