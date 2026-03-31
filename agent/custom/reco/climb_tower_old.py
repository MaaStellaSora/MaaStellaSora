import json
from typing import Any, Dict, Iterable, List

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition


_FALLBACK_TEMPLATE = (
    "ClimbTower/爬塔_buff推荐图标1__146_389_43_44__96_339_143_144.png"
)
_FALLBACK_THRESHOLD = 0.6


def _normalize_priority_param(param: Any) -> Dict[int, List[str]]:
    """Normalize custom_recognition_param into {priority:int -> [targets:str]}.

    Accepts dict / JSON string / bytes. Ignores invalid priority keys.
    """
    if param is None:
        return {}

    if isinstance(param, (bytes, bytearray)):
        param = param.decode("utf-8", errors="replace")

    if isinstance(param, str):
        if not param.strip():
            return {}
        parsed = json.loads(param)
    else:
        parsed = param

    if not isinstance(parsed, dict):
        raise ValueError("custom_recognition_param must be a dict or JSON string")

    normalized: Dict[int, List[str]] = {}
    for key, value in parsed.items():
        try:
            priority = int(key)
        except (TypeError, ValueError):
            continue

        if isinstance(value, (list, tuple)):
            targets: Iterable[Any] = value
        else:
            targets = [value]

        normalized[priority] = [str(item) for item in targets if str(item).strip()]

    return normalized


def _run_expected_ocr(
    context: Context, image, expected: str
) -> Any:
    return context.run_recognition(
        "OCR",
        image,
        pipeline_override={
            "OCR": {
                "recognition": "OCR",
                "expected": expected,
                "action": "DoNothing",
            }
        },
    )


def _run_fallback_template(
    context: Context, image
) -> Any:
    return context.run_recognition(
        "FallbackTemplate",
        image,
        pipeline_override={
            "FallbackTemplate": {
                "recognition": "TemplateMatch",
                "template": [_FALLBACK_TEMPLATE],
                "green_mask": True,
                "action": "DoNothing",
                "threshold": _FALLBACK_THRESHOLD,
            }
        },
    )


@AgentServer.custom_recognition("auto_tower")
class TowerRecognition(CustomRecognition):

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        if context.tasker.stopping:
            return CustomRecognition.AnalyzeResult(
                box=(0, 0, 0, 0),
                detail="Task Stopped",
            )

        # priority_dict = {
        #     "3": [
        #         "花海·叠浪",
        #         "花海·汹涌",
        #         "花海·爆裂",
        #         "禁行逆风",
        #         "暖风加护",
        #         "森林公主的赐福",
        #         "风蚀坏劫",
        #     ],
        #     "2": [
        #         "风魔种子",
        #         "自我提升",
        #         "花海·侵蚀",
        #         "花海·荟聚",
        #         "全能领导",
        #         "流速紊乱",
        #         "众星拥戴",
        #         "风云无常",
        #         "单科学习强化",
        #         "弱点解析",
        #     ],
        # }
        try:
            priority_dict = _normalize_priority_param(argv.custom_recognition_param)
        except Exception as exc:
            print(f"custom_recognition_param 解析失败: {exc}")
            priority_dict = {}

        for priority in sorted(priority_dict.keys(), reverse=True):
            targets = priority_dict[priority]
            for target in targets:
                if context.tasker.stopping:
                    return CustomRecognition.AnalyzeResult(
                        box=(0, 0, 0, 0),
                        detail="Task Stopped",
                    )

                print(f"正在识别优先级 {priority} 的目标: {target}")
                reco_detail = _run_expected_ocr(context, argv.image, target)
                print(f"识别结果: {reco_detail}")

                if reco_detail and reco_detail.hit and reco_detail.best_result:
                    box = reco_detail.best_result.box
                    print(f"找到目标 {target}，位置: {box}")
                    return CustomRecognition.AnalyzeResult(
                        box=box,
                        detail=f"Found {target} with priority {priority}",
                    )

        print("未找到任何目标，尝试推荐卡片图标")
        reco_detail = _run_fallback_template(context, argv.image)
        if reco_detail and reco_detail.hit and reco_detail.best_result:
            box = reco_detail.best_result.box
            return CustomRecognition.AnalyzeResult(
                box=box,
                detail="use recommend card",
            )

        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 0, 0),
            detail="not found",
        )