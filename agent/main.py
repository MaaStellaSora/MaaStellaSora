import sys

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.custom_action import CustomAction
from maa.context import Context
import json


@AgentServer.custom_recognition("auto_tower")
class TowerRecongition(CustomRecognition):

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        config = argv.custom_recognition_param
        print(config)
        priority_dict = json.loads(config)
        print(priority_dict)
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
            sorted_priorities = sorted(priority_dict.keys(), key=int, reverse=True)

            for priority in sorted_priorities:
                targets = priority_dict[priority]

                for target in targets:
                    print(f"正在识别优先级 {priority} 的目标: {target}")

                    reco_detail = context.run_recognition(
                        "OCR",
                        argv.image,
                        {
                            "OCR": {
                                "recognition": "OCR",
                                "expected": target,
                                "action": "DoNothing",
                            }
                        },
                    )

                    print(f"识别结果: {reco_detail}")

                    if reco_detail and reco_detail.hit:
                        box = reco_detail.best_result.box
                        print(f"找到目标 {target}，位置: {box}")
                        return CustomRecognition.AnalyzeResult(
                            box=(box.x, box.y, box.width, box.height),
                            detail=f"Found {target} with priority {priority}",
                        )
        except Exception as e:
            print(f"发生错误: {e}")
        print("未找到任何目标")
        reco_detail = context.run_recognition(
            "OCR",
            argv.image,
            {
                "OCR": {
                    "recognition": "TemplateMatch",
                    "template": [
                        "ClimbTower/爬塔_buff推荐图标1__146_389_43_44__96_339_143_144.png"
                    ],
                    "green_mask": True,
                    "action": "DoNothing",
                    "threshold": 0.6,
                }
            },
        )
        box = reco_detail.best_result.box
        return CustomRecognition.AnalyzeResult(
            box=box,
            detail=f"use recommend card",
        )


def main():
    Toolkit.init_option("./")

    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)

    socket_id = sys.argv[-1]

    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
