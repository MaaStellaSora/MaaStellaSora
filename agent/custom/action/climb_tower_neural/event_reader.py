"""Neural-only evidence and fixed-menu adapters over Maa's event reader."""
from dataclasses import dataclass, field
import re
from types import SimpleNamespace

from maa.custom_recognition import CustomRecognition
from maa.define import BoxAndScoreResult, OCRResult

from custom.reco.climb_tower_event.main import (
    EventRecognition as OfficialEventRecognition, CUSTOM_RULE_REGISTRY, _match_regex,
)
from custom.reco.climb_tower_event.models import Choice as OfficialChoice, EventInfo

from .event_mapping import choice_complete

LOCATION_NODE = "星塔_节点_对话选择_定位选项位置_agent"
CHOICE_NODE = "星塔_节点_对话选择_识别选项文本_agent"
CONSEQUENCE_NODE = "星塔_节点_对话选择_识别选项后果_agent"
QUESTION_NODE = "星塔_节点_对话选择_识别问题文本_agent"


@dataclass(slots=True)
class Choice(OfficialChoice):
    observations: dict = field(default_factory=dict)
    available: bool | None = None


def _observation(result, node, box, replacements):
    rows = [item for item in getattr(result, "filtered_results", ()) if isinstance(item, OCRResult)]
    raw = [item for item in getattr(result, "all_results", ()) if isinstance(item, OCRResult)] or rows
    cleared_boxes = set()
    for original in raw:
        if not original.text.strip():
            continue
        replaced = original.text
        for pattern, replacement in replacements:
            replaced = re.sub(pattern, replacement, replaced)
        if not replaced.strip():
            cleared_boxes.add(tuple(original.box))
    rows = [item for item in rows if item.text.strip() or tuple(item.box) not in cleared_boxes]
    return dict(node_name=node, roi_anchor=box,
                complete=bool(result and result.hit and rows and all(item.text.strip() for item in rows)),
                raw=[dict(text=item.text, box=list(item.box), score=item.score) for item in raw])


class _ReadingContext:
    """Collect the original reader's OCR calls without reading a menu twice."""
    def __init__(self, context):
        self.context = context
        self.observations = {}
        params = (context.get_node_data(CHOICE_NODE) or {}).get("recognition", {}).get("param", {})
        self.title_replacements = params.get("replace", [])

    def __getattr__(self, name):
        return getattr(self.context, name)

    def run_recognition(self, node, image, *args, **kwargs):
        result = self.context.run_recognition(node, image, *args, **kwargs)
        if node == LOCATION_NODE and result and result.hit:
            anchors = [item for item in result.filtered_results if isinstance(item, BoxAndScoreResult)]
            return SimpleNamespace(hit=bool(anchors), filtered_results=anchors)
        if node == QUESTION_NODE:
            self.observations[node] = _observation(result, node, None, [])
        if node in (CHOICE_NODE, CONSEQUENCE_NODE):
            box = kwargs["pipeline_override"][node]["recognition"]["param"]["roi"]
            replacements = self.title_replacements if node == CHOICE_NODE else []
            self.observations[node, tuple(box)] = _observation(result, node, box, replacements)
        return result


class EventRecognition(OfficialEventRecognition):
    @staticmethod
    def _get_question_text(context, image, lang_type):
        observed = _ReadingContext(context)
        text = OfficialEventRecognition._get_question_text(observed, image, lang_type)
        return text if observed.observations.get(QUESTION_NODE, {}).get("complete") else ""

    @staticmethod
    def _get_choice_texts(context, image, lang_type):
        observed = _ReadingContext(context)
        choices = OfficialEventRecognition._get_choice_texts(observed, image, lang_type)
        return [Choice(text=choice.text, consequence=choice.consequence, box=choice.box,
                       observations=dict(text=observed.observations[CHOICE_NODE, tuple(choice.box)],
                                         consequence=observed.observations[CONSEQUENCE_NODE, tuple(choice.box)]))
                for choice in choices]

    def match_from_menu(self, context, argv, question, choices):
        """Read Maa's answer rules without publishing a choice during retries."""
        available = [choice for choice in choices if getattr(choice, "available", None) is not False]
        node = context.get_node_data(argv.node_name) or {}
        info = EventInfo(question=question, choices=available)
        for rule_index, rule in enumerate(node.get("attach", {}).get("rules", [])):
            if rule.get("question") and not _match_regex(question, rule["question"]):
                continue
            if not any(rule.get(field) for field in ("choices", "consequences", "custom")):
                continue
            matched = [choice for choice in available
                       if all(not rule.get(field) or _match_regex(getattr(choice, attribute), rule[field])
                              for field, attribute in (("choices", "text"), ("consequences", "consequence")))]
            if not matched:
                continue
            custom = rule.get("custom")
            if custom:
                selector = CUSTOM_RULE_REGISTRY.get(custom)
                if selector is None:
                    continue
                selected = selector(context, rule, info, matched)
            else:
                selected = matched[0]
            if selected is None:
                continue
            index = choices.index(selected)
            return CustomRecognition.AnalyzeResult(box=selected.box, detail={
                "rule_matched": True, "matched_rule_index": rule_index,
                "selection_basis": "official_answer_rule", "selected_ui_index": index,
                "full_visible_menu": all(choice_complete(choice) for choice in choices)})
        return CustomRecognition.AnalyzeResult(box=None, detail={
            "rule_matched": False, "selection_basis": "official_answer_unmatched"})

    def select_from_menu(self, context, argv, question, choices):
        result = self.match_from_menu(context, argv, question, choices)
        if result.box is not None:
            selected = choices[result.detail["selected_ui_index"]]
            context.override_pipeline({argv.node_name: {"attach": {
                "last_question": question, "last_choice": selected.text,
                "last_consequence": selected.consequence}}})
        return result
