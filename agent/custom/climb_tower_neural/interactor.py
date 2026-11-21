"""Pipeline operations and numeric evidence for the neural tower task."""
from contextvars import copy_context
from itertools import count

from maa.context import Context
from maa.custom_recognition import CustomRecognition

from custom.interactor_template import Interactor as BaseInteractor
from .ocr_number import DECIMAL_AMOUNT, assess_amount


_frame_observers = {}
WALLET_NODE = "星塔_神经网络_识别当前金币"


class FrameObservation(CustomRecognition):
    """Analyze the frame supplied by the Pipeline recognition loop."""

    def analyze(self, context, argv):
        observer = _frame_observers[int(argv.custom_recognition_param)]
        if context.tasker.stopping:
            return None
        try:
            result = observer["scope"].run(observer["observe"], argv.image)
        except Exception as error:
            # Propagate through run_task instead of losing the exception in
            # the native callback boundary.
            observer["error"] = error
            return [0, 0, 1, 1]
        if result is not None and not context.tasker.stopping:
            observer["result"] = result
            return [0, 0, 1, 1]
        return None


class Interactor(BaseInteractor):
    """Extend the stock interactor only for this task's frames and evidence."""

    @staticmethod
    def capture(context: Context):
        """由 Pipeline 截取一帧，沿用框架的画面尺寸与坐标。"""
        return Interactor.wait_for(context, lambda image: image)

    @staticmethod
    def wait_for(context, observe, *, timeout=3.0):
        """Let Maa own screenshot polling, its interval and its deadline."""
        if context.tasker.stopping:
            return None
        slot = next(index for index in count() if index not in _frame_observers)
        observer = dict(observe=observe, scope=copy_context(), result=None, error=None)
        _frame_observers[slot] = observer
        entry, frame = "星塔_神经网络_交互_等待", "星塔_神经网络_交互_观察"
        try:
            context.run_task(entry, {
                entry: {"timeout": max(1, round(timeout * 1000))},
                frame: {"recognition": {"type": "Custom", "param": {
                    "custom_recognition_param": slot}}},
            })
            if observer["error"] is not None:
                raise observer["error"]
            return None if context.tasker.stopping else observer["result"]
        finally:
            del _frame_observers[slot]

    def screenshot(self):
        """截图并保存到self.image"""
        self.image = self.capture(self.context)
        return self.image

    def get_current_coin(self, *, roi=None):
        return self._read_number(WALLET_NODE, roi=roi)

    def _located_wallet_number(self, node_name, params):
        source_roi = list(params["roi"])
        locator_node = node_name + "__文字定位"
        locator = self.context.run_recognition(locator_node, self.image, {
            locator_node: {"recognition": {"type": "OCR", "param": dict(params,
                roi=source_roi, only_rec=False, replace=[], threshold=0., expected=[".*"])},
                "attach": dict(purpose="wallet_text_locator", source_node=node_name,
                    source_roi=source_roi, preprocessing="text_detection_locator")}})
        self._number_locator_observed = tuple((row.text, row.score) for row in locator.all_results) if locator else ()
        self._number_locator_reco_id = locator.reco_id if locator else None
        bounds_reco_id = self._number_locator_reco_id
        boxes = [row.box for row in locator.all_results
                 if tuple(row.box) != tuple(source_roi) and row.box[2] > 0 and row.box[3] > 0] if locator else []
        preprocessing = "detected_text_padding"
        if not boxes:
            digit_node = "星塔_神经网络_定位钱包数字"
            digits = self.context.run_recognition(digit_node, self.image, {
                digit_node: {"recognition": {"param": {"roi": source_roi}},
                    "attach": dict(purpose="wallet_digit_locator", source_node=node_name,
                        source_roi=source_roi, source_locator_reco_id=self._number_locator_reco_id,
                        preprocessing="digit_color_locator")}})
            boxes = [row.box for row in digits.filtered_results] if digits and digits.hit else []
            if boxes:
                bounds_reco_id = digits.reco_id
                preprocessing = "digit_color_padding"
        amount_roi, detected_roi, padding = source_roi, None, 0
        if boxes:
            left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
            right, bottom = max(box[0] + box[2] for box in boxes), max(box[1] + box[3] for box in boxes)
            detected_roi = [left, top, right - left, bottom - top]
            padding = max(2, (bottom - top) // 4)
            x, y, width, height = source_roi
            left, top = max(x, left - padding), max(y, top - padding)
            right, bottom = min(x + width, right + padding), min(y + height, bottom + padding)
            amount_roi = [left, top, right - left, bottom - top]
        return self.context.run_recognition(node_name, self.image, {
            node_name: {"recognition": {"type": "OCR", "param": dict(params,
                roi=amount_roi, only_rec=True, replace=[], threshold=.85,
                expected=["^" + DECIMAL_AMOUNT.pattern + "$"])},
                "attach": dict(purpose="wallet_amount", source_node=node_name,
                    source_roi=source_roi, source_locator_reco_id=bounds_reco_id,
                    detected_text_roi=detected_roi, padding=padding,
                    preprocessing=preprocessing if boxes else "original_roi")}})

    def _number_attempts(self, node_name, params, locate_first):
        if not (node_name == WALLET_NODE and locate_first):
            yield self.context.run_recognition(node_name, self.image)
        if not self.context.tasker.stopping and node_name == WALLET_NODE:
            yield self._located_wallet_number(node_name, params)
        if self.context.tasker.stopping:
            return
        source_roi = list(params["roi"])
        retry_node = node_name + "__数字重读"
        # 独立节点保证重读参数不会改变后续原节点的 only_rec、ROI 或 replace。
        only_rec = not params.get("only_rec", False)
        retry_params = dict(params, roi=source_roi, only_rec=only_rec, replace=[], threshold=self._number_minimum,
                            expected=["^" + DECIMAL_AMOUNT.pattern + "$"])
        yield self.context.run_recognition(retry_node, self.image, {
            retry_node: {"recognition": {"type": "OCR", "param": retry_params},
                "attach": dict(source_node=node_name, source_roi=source_roi,
                    purpose="wallet_amount" if node_name == WALLET_NODE else "number_amount",
                    source_locator_reco_id=self._number_locator_reco_id,
                    preprocessing="single_line" if only_rec else "text_detection")}})

    def _read_number(self, node_name: str, *, roi=None) -> int:
        """只在未读清时重读；同一画面已读到的不同金额不能靠重试覆盖。"""
        node = self.context.get_node_data(node_name) or {}
        params = dict(node.get("recognition", {}).get("param", {}))
        if roi is not None:
            params["roi"] = roi
        attach = node.get("attach", {})
        self._number_minimum = .85 if attach.get("strict_number_reading") else params.get("threshold", .3)
        self._number_locator_observed = ()
        self._number_locator_reco_id = None
        for detail in self._number_attempts(node_name, params, attach.get("wallet_text_locator")):
            if detail is None:
                continue
            value, status, _ = assess_amount(
                (row.text for row in detail.filtered_results if row.score >= self._number_minimum) if detail.hit else (),
                tuple((row.text, row.score) for row in detail.all_results) + self._number_locator_observed)
            if value is not None:
                return value
            if status == "conflict":
                break
        return -1

    def click(self, box: list[int]) -> bool:
        """由框架点击调用方本帧识别得到的目标框。"""
        result = self.context.run_action("星塔_神经网络_交互_点击", box=list(box))
        return bool(result and result.success)
