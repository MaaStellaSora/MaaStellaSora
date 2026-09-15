import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger as logger_module
from utils.image_handler import save_image
logger = logger_module.get_logger("climb_tower_record")


@AgentServer.custom_action("record_stop_check")
class RecordStopCheck(CustomAction):
    """记录结算页判定：刷到指定「潜能数」与「记录等级」就自动停止爬塔。

    潜能数   = 旅人区域红框三个数字之和（识别失败记为 0，避免误停）
    记录等级 = 左上角大徽章数字（识别失败记为 0，避免误停）
    达标条件 = 潜能数 >= min_potential_count 且 记录等级 >= min_record_level
    目标值从节点 attach 读取。

    ROI 为 1280x720 估算值，需按实机校准；识别失败默认“不达标”，安全不误停。
    """
    LEVEL_ROI = [45, 25, 80, 145]
    POTENTIAL_ROIS = [
        [130, 220, 65, 55],
        [225, 220, 65, 55],
        [315, 220, 65, 55],
    ]

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        node_data = context.get_node_data(argv.node_name) or {}
        attach = node_data.get("attach", {})
        try:
            min_pc = int(attach.get("min_potential_count", 80))
            min_lv = int(attach.get("min_record_level", 30))
        except (TypeError, ValueError):
            min_pc, min_lv = 80, 30

        image = context.tasker.controller.post_screencap().wait().get()

        try:
            save_image(image, "记录判定")
        except Exception:
            pass

        # 全屏文字诊断：打印记录页上所有 OCR 识别到的文字+坐标，用于校准坐标
        _diag = context.run_recognition("星塔_记录_识别数字_agent", image, {
            "星塔_记录_识别数字_agent": {"recognition": {"param": {"expected": [".+"], "roi": [0, 0, 1280, 720]}}}
        })
        if _diag:
            for r in (_diag.filtered_results or []):
                logger.info(f"[记录判定] 界面文字 '{r.text}' box={r.box}")

        # 识别记录等级 + 三个潜能数；若没读全，重新截图识别，最多 8 次
        level = None
        parts = [0, 0, 0]
        for _attempt in range(8):
            image = context.tasker.controller.post_screencap().wait().get()
            level = self._number(context, image, self.LEVEL_ROI)
            parts = [self._number(context, image, r) or 0 for r in self.POTENTIAL_ROIS]
            if level is not None and all(p > 0 for p in parts):
                break
            logger.warning(f"[记录判定] 第 {_attempt + 1} 次识别未读全（记录等级={level}, 潜能数={parts}），重试")
            time.sleep(1.5)
        potential_count = sum(parts)

        reached = (
            level is not None
            and level >= min_lv
            and potential_count >= min_pc
        )

        logger.info(
            f"[记录判定] 潜能数={potential_count}({parts}) 目标≥{min_pc} | "
            f"记录等级={level} 目标≥{min_lv} => {'达标，停止爬塔' if reached else '未达标，继续刷'}"
        )

        if reached:
            logger.info("[记录判定] 已刷到目标记录，停止爬塔")
            context.tasker.post_stop()

        return True

    @staticmethod
    def _number(context: Context, image, roi: list) -> int | None:
        detail = context.run_recognition(
            "星塔_记录_识别数字_agent",
            image,
            {"星塔_记录_识别数字_agent": {"recognition": {"param": {"roi": roi}}}},
        )
        if detail and detail.hit and detail.best_result:
            text = detail.best_result.text
            if text.isdigit():
                logger.info(f"[记录判定] roi={roi} 识别到数字 '{text}' box={detail.best_result.box}")
                return int(text)
        alltext = [r.text for r in (detail.all_results or [])]
        logger.info(f"[记录判定] roi={roi} 未命中，该区域识别到 {alltext!r}")
        return None


@AgentServer.custom_action("settle_click_through")
class SettleClickThrough(CustomAction):
    """结算流程：连续点空白处继续，直到出现「未命名记录」（保存纪录）界面。

    用于点掉「探索完成！」「获得道具！」等需要"点击空白处继续"的结算弹窗，
    不依赖 OCR 识别文字（文字识别不稳定），直接点屏幕中部偏上空白，
    每点一次检查一次是否已出现"保存纪录"，出现即停。
    """
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        for _ in range(15):
            if context.tasker.stopping:
                return False
            image = context.tasker.controller.post_screencap().wait().get()
            # 若已出现"保存纪录"（未命名记录界面），停止点空白
            d = context.run_recognition("星塔_检查记录达标_agent", image)
            if d and d.hit:
                logger.info("[结算] 已进入未命名记录界面（检测到保存纪录）")
                return True
            # 尝试识别底部"继续/点击继续"文字并点击
            if self._click_continue(context, image):
                time.sleep(0.6)
                continue
            # 否则点屏幕中部偏上空白（"点击空白处继续"）
            h = image.shape[0]
            w = image.shape[1]
            context.tasker.controller.post_click(int(w * 0.5), int(h * 0.85)).wait()
            time.sleep(0.6)
        logger.warning("[结算] 多次点空白后仍未检测到保存纪录，继续流程")
        return True

    @staticmethod
    def _click_continue(context: Context, image) -> bool:
        """尝试识别底部"点击空白处继续"或"点击A继续"字样并点击；找不到返回 False。

        手柄模式下提示变为"点击 Ⓐ 继续"，鼠标点击提示位置同样有效，故一并识别。
        普通模式点提示行约 42% 处（"空白"二字）；手柄模式同样点约 42% 处（≈"A"图标位置）。
        """
        roi = [200, 560, 900, 160]
        # 第1级：整行"点击…继续"（覆盖普通"点击空白处继续"与手柄"点击A继续"）
        # 第2级：OCR 把提示拆成小块时，兜底只认"继续"二字
        for exprs in (["点击空白处继续", "点击.*继续"], ["继续"]):
            override = {
                "星塔_点击空白处关闭_agent": {
                    "recognition": {"param": {"expected": exprs, "roi": roi}}
                }
            }
            d = context.run_recognition("星塔_点击空白处关闭_agent", image, override)
            if d and d.hit and d.best_result:
                box = d.best_result.box
                text = d.best_result.text
                # 点击提示行约 42% 处：普通="空白"二字，手柄≈"A"图标位置
                cx = int(box[0] + box[2] * 0.42)
                cy = int(box[1] + box[3] / 2)
                logger.info(f"[结算] 识别到 '{text}'，点击 ({cx},{cy})：关闭继续弹窗")
                context.tasker.controller.post_click(cx, cy).wait()
                return True
        return False


@AgentServer.custom_action("click_blank")
class ClickBlank(CustomAction):
    """通用：点屏幕中部偏上空白，关掉"点击空白处继续/继续"弹窗（不依赖OCR）。"""
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        if context.tasker.stopping:
            return False
        image = context.tasker.controller.post_screencap().wait().get()
        # 优先：识别"点击空白处继续/继续"文字并点击（最准，避免点到列表/卡片）
        if SettleClickThrough._click_continue(context, image):
            return True
        # 尝试识别当前弹窗标题（仅用于日志）
        what = "弹窗"
        try:
            d = context.run_recognition("星塔_弹窗点空白_agent", image)
            if d and d.hit and d.best_result and d.best_result.text:
                what = d.best_result.text
        except Exception:
            pass
        h = image.shape[0]
        w = image.shape[1]
        cx = int(w * 0.5)
        cy = int(h * 0.10)
        logger.info(f"[弹窗点空白] 识别到 '{what}'，点击 ({cx},{cy})：未识别到'继续'文字，兜底点顶部空白关闭弹窗")
        context.tasker.controller.post_click(cx, cy).wait()
        return True
