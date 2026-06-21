import time

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from .ui import UIInteractor
from .state import OwnedPotential

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_potential_bag")

# 潜能卡片具体定位参数
POTENTIAL_XS = [358, 475, 593, 710, 828]
POTENTIAL_W = 118
POTENTIAL_H = 148
CORE_POTENTIAL_W = 45
CORE_POTENTIAL_H = 40
POTENTIAL_LEVEL_W = 50
POTENTIAL_LEVEL_H = 40
RECOMMENDED_LEVEL_W = 28
RECOMMENDED_LEVEL_H = 35

# 潜能画面定位参数
UPPER_POTENTIAL_FIRST_ROW_Y_OFFSET = 47
UPPER_POTENTIAL_SECOND_ROW_Y_OFFSET = 211
LOWER_POTENTIAL_FIRST_ROW_Y_OFFSET = 18
LOWER_POTENTIAL_SECOND_ROW_Y_OFFSET = 167
# 划屏结束点
SCROLL_END = [640, 80]

# 旅人名称位置参数（暂时没用）
TREKKER_X = 310
TREKKER_Y_OFFSET = 5
TREKKER_W = 200
TREKKER_H = 23 + TREKKER_Y_OFFSET * 2


class BagUIInteractor(UIInteractor):
    def __init__(self, context: Context):
        super().__init__(context)
        self.image_reverse = None

    def screenshot(self):
        super().screenshot()
        self.image_reverse = ~self.image

    def stop_task(self):
        self.context.tasker.post_stop().wait()

    def open_bag(self):
        self.context.run_task("星塔_打开背包界面_agent")

    def close_bag(self):
        self.context.run_task("星塔_关闭背包界面_agent")

    def click(self, x: int, y: int):
        self.context.tasker.controller.post_click(x, y).wait()

    def scroll_down(self, begin_point, end_point):
        pipeline_override = {"星塔_背包_向下滑动_agent": {"action": {"param": {"begin": begin_point, "end": end_point}}}}
        self.context.run_task("星塔_背包_向下滑动_agent", pipeline_override)

    def get_trekker_name_roi_from_bag(self) -> list[int]:
        results = self._template("星塔_背包_潜能标记_agent", [])
        if len(results) > 1:
            logger.warning("识别到多个潜能标记，无法确定旅人名称位置")
        if len(results) == 0:
            logger.error("未识别到潜能标记，无法确定旅人名称位置")
            return []
        roi = results[0]
        return [TREKKER_X, roi[1] - TREKKER_Y_OFFSET, TREKKER_W, TREKKER_H]

    def get_upper_scroll_begin_point_from_bag(self) -> list[int]:
        results = self._template("星塔_背包_潜能标记_agent", [])
        if not results or len(results) > 1:
            logger.error("无法定位潜能种类标记位置")
            return []
        return [640, results[0][1]]

    def get_lower_scroll_begin_point_from_bag(self) -> list[int]:
        ocr_results = self._ocr("星塔_背包_通用潜能标记_agent", [])
        if not ocr_results or len(ocr_results) > 1:
            logger.error("无法定位通用潜能位置")
            return []
        return [640, ocr_results[0][1][1]]

    def get_upper_potential_ys_from_bag(self) -> tuple[int, int]:
        results = self._template("星塔_背包_潜能标记_agent", [])
        if len(results) > 1:
            logger.warning("识别到多个潜能标记，无法确定潜能行位置")
        if len(results) == 0:
            logger.error("未识别到潜能标记，无法确定潜能行位置")
            return 0, 0
        roi = results[0]
        return roi[1] + UPPER_POTENTIAL_FIRST_ROW_Y_OFFSET, roi[1] + UPPER_POTENTIAL_SECOND_ROW_Y_OFFSET

    def get_lower_potential_ys_from_bag(self) -> tuple[int, int]:
        ocr_results = self._ocr("星塔_背包_通用潜能标记_agent", [])
        if len(ocr_results) == 0:
            logger.error("未识别到潜能标记，无法确定潜能行位置")
            return 0, 0
        roi = ocr_results[0][1]
        return roi[1] + LOWER_POTENTIAL_FIRST_ROW_Y_OFFSET, roi[1] + LOWER_POTENTIAL_SECOND_ROW_Y_OFFSET

    def get_potential_name_from_bag(self, roi: list[int]) -> str:
        ocr_results = self._ocr("星塔_背包_识别潜能名称_agent", "",roi=roi)
        if not ocr_results:
            logger.debug("未识别到潜能名称")
            return ""
        return ocr_results[0][0]

    def get_potential_level_from_bag(self, roi: list[int]) -> int:
        ocr_results = self._ocr("星塔_背包_识别潜能等级_agent", "",roi=roi)
        if not ocr_results:
            logger.warning("未识别到潜能等级，将默认为0")
            return 0
        return int(ocr_results[0][0])

    def get_potential_recommend_level_from_bag(self, roi: list[int]) -> int:
        ocr_results = self._ocr("星塔_背包_识别推荐等级_agent", "",roi=roi, image=self.image_reverse)
        if not ocr_results:
            logger.warning("未识别到推荐等级，将默认为0")
            return 0
        return int(ocr_results[0][0])

    def check_potential_recommended_from_bag(self, roi: list[int]) -> bool:
        return self._color("星塔_背包_识别核心潜能_agent",roi=roi)




class PotentialReader:
    def __init__(self, context: Context):
        self.context = context
        self.screen = BagUIInteractor(context)

    def read_potentials(self) -> list[OwnedPotential]:
        """读取背包中的所有潜能"""
        # 先打开背包，确认处于背包状态
        self.screen.open_bag()

        # 初始化潜能列表
        potentials: list[OwnedPotential] = []

        # 然后开始读取循环
        for i in range(3):
            # 如果是第一次读取潜能，就点一下第三行的潜能移开光标，读取完再点回去
            # 如果是第二次之后，则滚动到潜能图标处于顶部可见区域
            if i == 0:
                self.screen.click(420, 535)
                time.sleep(0.5)
            else:
                begin_point = self.screen.get_upper_scroll_begin_point_from_bag()
                if not begin_point:
                    logger.error("无法定位潜能种类标记位置，扫描背包潜能失败。")
                    return []
                self.screen.scroll_down(begin_point, SCROLL_END)

            # 截图
            self.screen.screenshot()
            # 通过潜能标记定位前两行的潜能
            first_y, second_y = self.screen.get_upper_potential_ys_from_bag()
            if not first_y or not second_y:
                logger.error("无法定位潜能位置，扫描背包潜能失败。")
                return []

            # 生成潜能列表的roi
            potential_rois, core_potential_rois, level_rois, recommend_rois = self._generate_rois(first_y, second_y)

            # 然后读取前两行的潜能
            potentials_upper = self._read_rows(potential_rois, core_potential_rois, level_rois, recommend_rois)

            # 如果是第一次读取潜能，读取完把光标点回去
            if i == 0:
                self.screen.click(420, 205)
                time.sleep(0.5)

            # 读取完后，识别通用潜能的位置，然后向下滚动到刚好通用潜能处于顶部可见区域
            begin_point = self.screen.get_lower_scroll_begin_point_from_bag()
            if not begin_point:
                logger.error("无法定位潜能位置，扫描背包潜能失败。")
                return []
            self.screen.scroll_down(begin_point, SCROLL_END)
            # 截图
            self.screen.screenshot()
            # 然后读取后两行的潜能
            first_y, second_y = self.screen.get_lower_potential_ys_from_bag()
            if not first_y or not second_y:
                logger.error("无法定位潜能位置，扫描背包潜能失败。")
                return []
            potential_rois, core_potential_rois, level_rois, recommend_rois = self._generate_rois(first_y, second_y)
            potentials_lower = self._read_rows(potential_rois, core_potential_rois, level_rois, recommend_rois)

            # 识别完一个旅人之后，标记旅人名称
            potentials_tmp = potentials_upper + potentials_lower
            for potential in potentials_tmp:
                if not potential.trekker:
                    potential.trekker = str(i)

            # 合并临时潜能列表到主潜能列表
            potentials.extend(potentials_tmp)

        # 然后反复三次
        # 读取完潜能后，把数据返回，储存到State中
        self.screen.close_bag()
        logger.info(f"读取到 {len(potentials)} 个潜能")
        return potentials

    def _read_rows(
            self,
            potential_rois: list[list[int]],
            core_potential_rois: list[list[int]],
            level_rois: list[list[int]],
            recommend_rois: list[list[int]]
    ) -> list[OwnedPotential]:
        """读取潜能"""
        potentials = []
        for i, roi in enumerate(potential_rois):
            potential = OwnedPotential("", 0, 0)
            # 读取潜能名称
            potential.name = self.screen.get_potential_name_from_bag(roi)
            if not potential.name:
                continue
            # 读取核心潜能标记
            potential.core = self.screen.check_potential_recommended_from_bag(core_potential_rois[i])
            if not potential.core:
                # 读取潜能等级
                potential.level = self.screen.get_potential_level_from_bag(level_rois[i])
                # 读取推荐等级
                potential.recommended_level = self.screen.get_potential_recommend_level_from_bag(recommend_rois[i])
            # 读取旅人名称（由于意义不大所以不读）
            potential.trekker = ""
            # 保存潜能
            potentials.append(potential)
        return potentials

    @staticmethod
    def _generate_rois(
            y1: int, y2: int
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
        """生成潜能列表的roi"""
        potential_rois = [[POTENTIAL_XS[col], y, POTENTIAL_W, POTENTIAL_H]
                          for y in (y1, y2) for col in range(5)]
        core_potential_rois = [[POTENTIAL_XS[col], y, CORE_POTENTIAL_W, CORE_POTENTIAL_H]
                          for y in (y1, y2) for col in range(5)]
        level_rois = [[POTENTIAL_XS[col], y, POTENTIAL_LEVEL_W, POTENTIAL_LEVEL_H]
                          for y in (y1, y2) for col in range(5)]
        recommend_rois = [
            [POTENTIAL_XS[col] + POTENTIAL_W - RECOMMENDED_LEVEL_W, y, RECOMMENDED_LEVEL_W, RECOMMENDED_LEVEL_H]
            for y in (y1, y2) for col in range(5)]
        return potential_rois, core_potential_rois, level_rois, recommend_rois



@AgentServer.custom_recognition("bag_test_recognition")
class PotentialBagTestRecognition(CustomRecognition):

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        """测试用recognition"""
        reader = PotentialReader(context)
        print(reader.read_potentials())

        return CustomRecognition.AnalyzeResult(box=[1, 1, 1, 1], detail={})
