"""Neural settlement observations, followed by the existing note-tab scan."""
from contextvars import ContextVar
from copy import deepcopy

from maa.custom_action import CustomAction
from maa.define import BoxAndScoreResult

from custom.action.climb_tower_record_scan import (
    LEVEL_NODE, LOOP_NODE, POTENTIAL_LOCATION_NODE, POTENTIAL_RECOGNITION_NODE, RecordScanAction,
)
from .interactor import Interactor
from .ocr_number import assess_ocr_number
from .settlement_counts import check_potential_counts
from .ui_flow import read_until


class SettlementReader(CustomAction):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._observed = ContextVar("record_scan_observations", default=None)

    def run(self, context, argv):
        collector = self.runtime.collector
        requested = self.runtime.requested(context)
        if requested:
            self._prepare_save_recognition(context)
        if not requested or not collector.enabled or not collector.run_confirmed:
            return RecordScanAction().run(context, argv)
        context = collector.context(context)
        observed = {}
        token = self._observed.set(observed)
        try:
            with collector.capture("settlement") as read:
                image = Interactor.capture(context)
                if context.tasker.stopping:
                    return False
                level, parts = -1, []
                first, count_conflicts = True, []

                def observe(frame):
                    nonlocal image, level, parts, first
                    if not first:
                        ready = context.run_recognition("星塔_记录界面_读取数值_agent", frame)
                        if not ready or not ready.hit:
                            return None
                    first = False
                    image = frame
                    level = self._read_record_level(context, image, LEVEL_NODE)
                    parts = self._read_potential_counts(context, image,
                        POTENTIAL_LOCATION_NODE, POTENTIAL_RECOGNITION_NODE, warn=False)
                    for conflict in observed["potential_counts_conflicts"]:
                        if conflict not in count_conflicts:
                            count_conflicts.append(conflict)
                    if count_conflicts:
                        observed["potential_counts_conflicts"] = list(count_conflicts)
                        observed["potential_counts_complete"] = False
                        for conflict in count_conflicts:
                            observed["potential_counts"][conflict["ui_position"] - 1] = None
                        parts = []
                    return True if level >= 0 and parts else None

                read_until(context, observe, initial_image=image)
                if context.tasker.stopping:
                    return False
                if not observed["potential_counts_complete"]:
                    collector.logger.warning(
                        "$agent.custom.action.climb_tower_neural.settlement_potential_counts_unconfirmed")
                observed["read_id"] = read.read_id
                if RecordScanAction._is_options_active(context, LOOP_NODE):
                    attach = (context.get_node_data(LOOP_NODE) or {}).get("attach", {})
                    attach.update(record_level=level, potential_count=sum(parts) if parts else -1)
                    context.override_pipeline({LOOP_NODE: {"attach": attach}})
        finally:
            self._observed.reset(token)
        collector.settlement_result = observed
        collector.emit("settlement_observed", **observed, final_owned=collector.inventory.fields()["owned"],
                       **collector.owned_fields(), tracker_estimate=collector.inferred_state())
        self._read_note_settlement(context, image)
        return True

    @staticmethod
    def _prepare_save_recognition(context):
        save_node = "星塔_保存记录_agent"
        recognition = (context.get_node_data(save_node) or {}).get("recognition", {})
        if recognition.get("type") != "OCR":
            return
        # Copy the current server's original dictionary. The ordinary task
        # keeps that node unchanged; only this neural task adds a fallback.
        text_node = "星塔_神经网络_结算保存文字"
        context.override_pipeline({
            text_node: {"recognition": deepcopy(recognition)},
            save_node: {"recognition": {"type": "Or", "param": {"any_of": [
                text_node, "星塔_神经网络_结算保存图形",
            ]}}},
        })

    def _read_note_settlement(self, context, image=None):
        if image is None:
            image = Interactor.capture(context)
        # Reuse the current server's official tab OCR, in the settlement header.
        # The former random target extended below the tab and could miss it.
        tab_node = "星塔_背包_打开秘纹技能_agent"
        def locate(frame):
            nonlocal image
            image = frame
            tab = context.run_recognition(tab_node, image, {
                tab_node: {"recognition": {"param": {"roi": [835, 45, 400, 60]}}}})
            return tab if tab and tab.hit else None

        tab = read_until(context, locate, initial_image=image)
        if context.tasker.stopping:
            return
        if tab and tab.hit:
            click_node = "星塔_音符采集_结算秘纹页"
            switched = context.run_task(click_node, {
                click_node: {"action": {"param": {"target": list(tab.best_result.box)}}}})
            if not switched or not switched.status.succeeded:
                if not context.tasker.stopping:
                    self.runtime.notes.settlement(context, Interactor.capture(context))
                return
            image = Interactor.capture(context)
            if not context.tasker.stopping:
                self.runtime.notes.settlement(context, image)
        else:
            # Keep the actually missing final observation visible in the data.
            self.runtime.notes.settlement(context, image)

    def _read_record_level(self, context, image, node):
        screen = Interactor(context)
        screen.image = image
        level = screen._read_number(node)
        if (observed := self._observed.get()) is not None:
            observed.update(record_level=level if level >= 0 else None, record_level_complete=level >= 0)
        return level

    def _read_potential_counts(self, context, image, location_node, recognition_node, *, warn=True):
        # Detect text inside Maa's anchored numeric ROIs first. Decoding the
        # entire narrow ROI as one line can lower confidence on clear digits.
        location = context.run_recognition(location_node, image)
        anchors = sorted((item for item in location.filtered_results
                          if isinstance(item, BoxAndScoreResult)), key=lambda item: item.box[0]) if location else []
        params = (context.get_node_data(recognition_node) or {}).get("recognition", {}).get("param", {})
        fields = self.runtime.collector.owned_fields()
        parts, ocr_conflicts = [None, None, None], {}
        if len(anchors) == 3:
            offset = params.get("roi_offset", [0, 0, 0, 0])
            rois = []
            for anchor in anchors:
                roi = [value + shift for value, shift in zip(anchor.box, offset)]
                # The icon's right rim can become a leading "1" with the
                # original one-pixel gap. Retain the configured right edge.
                left = max(roi[0], anchor.box[0] + anchor.box[2] + 2)
                rois.append([left, roi[1], roi[0] + roi[2] - left, roi[3]])
            for index, roi in enumerate(rois):
                parts[index], _, conflicts = self._read_settlement_number(context, image, recognition_node, params, roi)
                if conflicts:
                    ocr_conflicts[index] = conflicts
            checked = check_potential_counts(parts, fields)
            retry = {index for index, value in enumerate(parts) if value is None}
            retry.update(item["ui_position"] - 1 for item in checked["potential_counts_conflicts"])
            for index in sorted(retry - ocr_conflicts.keys()):
                if context.tasker.stopping:
                    break
                value, _, conflicts = self._read_settlement_number(
                    context, image, recognition_node, params, rois[index], detect=False)
                readings = {v for v in (parts[index], value, *conflicts) if v is not None}
                if len(readings) > 1:
                    ocr_conflicts[index] = sorted(readings)
                    parts[index] = None
                elif value is not None:
                    parts[index] = value
        checked = check_potential_counts(parts, fields)
        if ocr_conflicts:
            checked["potential_counts_conflicts"].extend(
                dict(ui_position=index + 1, observed_values=list(values), reason="independent_ocr_conflict")
                for index, values in sorted(ocr_conflicts.items()))
            checked["potential_counts_complete"] = False
        if (observed := self._observed.get()) is not None:
            observed.update(checked)
            observed["potential_counts_order"] = "ui_position_order"
            observed["potential_count_roles"] = None
        if not checked["potential_counts_complete"]:
            if warn:
                self.runtime.collector.logger.warning(
                    "$agent.custom.action.climb_tower_neural.settlement_potential_counts_unconfirmed")
            return []
        return checked["potential_counts"]

    @staticmethod
    def _read_settlement_number(context, image, source_node, params, roi, *, detect=True):
        node = source_node + ("__结算数字" if detect else "__结算数字重读")
        params = dict(deepcopy(params), roi=list(roi), roi_offset=[0, 0, 0, 0],
                      only_rec=not detect, replace=[], expected=["^[0-9]+$"], threshold=.85)
        result = context.run_recognition(node, image, {
            node: {"recognition": {"type": "OCR", "param": params},
                   "attach": dict(source_node=source_node, source_roi=list(roi),
                                  preprocessing="text_detection" if detect else "single_line")}})
        return assess_ocr_number(result)
