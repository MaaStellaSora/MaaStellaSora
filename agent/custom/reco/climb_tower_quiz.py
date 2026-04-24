from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils import logger as logger_module
logger = logger_module.get_logger("quiz_recognition")


@AgentServer.custom_recognition("quiz_recognition")
class QuizRecognition(CustomRecognition):
    ROIS = {
        2: [670, 300, 590, 230],
        3: [670, 250, 590, 325],
        4: [670, 200, 590, 450]
    }
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        answer_count = 0
        default_box = [0, 0, 0, 0]

        reco_result = context.run_recognition("星塔_节点_随便选择_agent", argv.image)
        if reco_result and reco_result.hit:
            answer_count = len(reco_result.filtered_results)
            default_box = reco_result.best_result.box

        if not answer_count or answer_count not in self.ROIS:
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        roi = self.ROIS[answer_count]
        pipeline_override = {"星塔_节点_进行对话选择_agent":
            {
                 "recognition": {
                     "param": {
                         "roi": roi
                     }
                }
            }
        }
        reco_result = context.run_recognition(
            "星塔_节点_进行对话选择_agent",
            argv.image,
            pipeline_override=pipeline_override
        )
        if reco_result and reco_result.hit:
            target_text = reco_result.best_result.text
            target_box = reco_result.best_result.box
            logger.info(f"[问题选择] 选择答案：{target_text}")
            return CustomRecognition.AnalyzeResult(box=target_box, detail={})
        else:
            logger.info(f"[问题选择] 选择第一个选项")
            return CustomRecognition.AnalyzeResult(box=default_box, detail={})