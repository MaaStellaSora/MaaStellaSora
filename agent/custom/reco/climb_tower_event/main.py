import re

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.define import OCRResult, BoxAndScoreResult

from .models import EventInfo, Choice
from .custom_rules import CUSTOM_RULE_REGISTRY
from utils import logger as logger_module
from utils.dev_config import DEV_IMAGES_SAVE_ENABLED

logger = logger_module.get_logger("climb_tower_event")


def _match_regex(actual: str, pattern: str | list | tuple | None) -> bool:
    if pattern is None or actual is None:
        return False

    if isinstance(pattern, (list, tuple)):
        return any(_match_regex(actual, item) for item in pattern)

    expr = str(pattern).strip()
    text = str(actual).strip()
    if not expr or not text:
        return False

    try:
        return re.search(expr, text, flags=re.IGNORECASE) is not None
    except re.error:
        return re.search(re.escape(expr), text, flags=re.IGNORECASE) is not None


@AgentServer.custom_recognition("event_recognition")
class EventRecognition(CustomRecognition):

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        # 获取选项规则，选项规则在AscensionPreparation动作节点读取并存储在本节点的attach中rules中
        node_data = context.get_node_data(argv.node_name) or {}
        rules = node_data.get("attach", {}).get("rules", [])
        lang_type = node_data.get("attach", {}).get("lang_type", "cn")

        # 识别画面中的问题及选项，组合成EventInfo对象
        question_text = self._get_question_text(context, argv.image, lang_type)
        logger.info(f"[对话选择] 问题：{question_text}")
        choices = self._get_choice_texts(context, argv.image, lang_type)
        if not choices:
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        event_info = EventInfo(question=question_text, choices=choices)

        # 根据规则遍历选项列表，找到匹配的选项，匹配方法只使用正则表达式
        # 每一条规则包含 "question"、"choices"、"consequences"、"custom" 四个字段，分别对应问题、选项、选项后果、自定义规则
        # 还有一个 "description" 字段，用于描述该规则的作用
        # 这些字段均为列表，每个元素为一个字符串
        # 如字段不为空，则必须匹配到列表中的任意一个元素才算成功（为空时直接算作匹配成功）
        # 如四个字段都不为空，则必须三个字段都匹配到元素且自定义规则返回True才能算成功
        result_choice = None
        def _match_field(text: str, patterns: list[str] | str | None) -> bool:
            """字段为空或未配置时不作限制（视为匹配成功）；非空时需命中列表中任意一项。"""
            return not patterns or _match_regex(text, patterns)

        for rule in rules:
            rule_q = rule.get("question")
            rule_c = rule.get("choices")
            rule_cq = rule.get("consequences")
            rule_custom = rule.get("custom")
            rule_d = rule.get("description", "")
            matched_choices = []

            # 1. 匹配问题：配置了问题则必须匹配通过
            if not _match_field(question_text, rule_q):
                continue

            # 避免空规则导致的无差别命中（不允许选项或者后果均未配置的空规则，特别是仅配置了问题的规则）
            if not rule_c and not rule_cq and not rule_custom:
                continue

            # 2. 匹配选项与后果：非空字段必须全部满足（AND 关系）
            for choice in event_info.choices:
                if _match_field(choice.text, rule_c) and _match_field(choice.consequence, rule_cq):
                    matched_choices.append(choice)
            if not matched_choices:
                continue

            # 3. 匹配成功时，进行自定义规则判断，自定义规则拥有最高选择权
            if rule_custom:
                custom_function = CUSTOM_RULE_REGISTRY.get(rule_custom)
                if not custom_function:
                    logger.warning(f"未找到自定义规则函数：{rule_custom}")
                    continue
                try:
                    matched_choice = custom_function(context, rule, event_info, matched_choices)
                except Exception as e:
                    logger.error(f"自定义规则函数执行异常：{e}", exc_info=True)
                    continue
            else:
                matched_choice = matched_choices[0]

           # 4. 无法匹配到选项，跳过当前规则
            if not matched_choice:
                continue

            # 5. 所有匹配成功，选择当前选项
            result_choice = matched_choice
            logger.info(f"[对话选择] 命中规则：{rule_d}")
            logger.info(f"[对话选择] 选择选项：{result_choice.text}")
            logger.info(f"[对话选择] 后果: {result_choice.consequence}")
            logger.debug(f"规则内容: {rule}")
            break

        # 兜底：未命中任何规则时选择第一个选项
        if not result_choice:
            result_choice = event_info.choices[0]
            logger.info(f"[对话选择] 未命中任何规则，保底选择第一个选项")
            logger.info(f"[对话选择] 选择选项：{result_choice.text}")
            logger.debug(f"[对话选择] 后果: {result_choice.consequence}")
            if DEV_IMAGES_SAVE_ENABLED:
                from utils.image_handler import save_image
                save_image(argv.image, f"未知选项")

        # 回写attach，以便潜能选择节点使用
        pipeline_override = {
            argv.node_name: {
                "attach":{
                    "last_question": question_text,
                    "last_choice": result_choice.text,
                    "last_consequence": result_choice.consequence,
                }
            }
        }
        context.override_pipeline(pipeline_override)

        # 输出识别结果
        return CustomRecognition.AnalyzeResult(box=result_choice.box, detail={})

    @staticmethod
    def _get_question_text(context: Context, image: np.ndarray, lang_type: str) -> str:
        reco_result = context.run_recognition("星塔_节点_对话选择_定位问题位置_agent", image)
        if not (reco_result and reco_result.hit):
            return ""
        # 在pipeline直接抓取识别结果，所以这里不需要把识别结果传递给下一个节点
        reco_result = context.run_recognition("星塔_节点_对话选择_识别问题文本_agent", image)
        if not (reco_result and reco_result.hit):
            return ""
        # 合并文本，根据语言类型选择不同的分割符
        split_text = " " if lang_type == "en" else ""
        return split_text.join([r.text for r in reco_result.filtered_results if isinstance(r, OCRResult)])

    @staticmethod
    def _get_choice_texts(context: Context, image: np.ndarray, lang_type: str) -> list[Choice]:
        reco_result = context.run_recognition("星塔_节点_对话选择_定位选项位置_agent", image)
        if not reco_result or not reco_result.hit:
            return []

        choices = []
        consequences = []
        choice_boxes = []
        split_text = " " if lang_type == "en" else ""
        for r in reco_result.filtered_results:
            box = list(r.box) if isinstance(r, BoxAndScoreResult) else [0, 0, 0, 0]
            choice_boxes.append(box)

            # 节点覆写，指定roi为当前选项的box
            choice_node = "星塔_节点_对话选择_识别选项文本_agent"
            consequence_node = "星塔_节点_对话选择_识别选项后果_agent"
            pipeline_override_box = {"recognition": {"param": {"roi": box}}}
            override_choice = {choice_node: pipeline_override_box}
            override_consequence = {consequence_node: pipeline_override_box}

            # 开始识别
            reco_choice = context.run_recognition(choice_node, image, pipeline_override=override_choice)
            if reco_choice and reco_choice.hit:
                filtered_texts = [r.text for r in reco_choice.filtered_results if isinstance(r, OCRResult)]
                choices.append(split_text.join(filtered_texts))
            else:
                choices.append("")

            reco_consequence = context.run_recognition(consequence_node, image, pipeline_override=override_consequence)
            if reco_consequence and reco_consequence.hit:
                filtered_texts = [r.text for r in reco_consequence.filtered_results if isinstance(r, OCRResult)]
                consequences.append(split_text.join(filtered_texts))
            else:
                consequences.append("")

        return [
            Choice(text=text, consequence=consequence, box=box)
            for text, consequence, box in zip(choices, consequences, choice_boxes)
        ]
