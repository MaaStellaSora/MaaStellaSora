import random

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger as logger_module


logger = logger_module.get_logger(__name__)


@AgentServer.custom_action("activity_challenge_random_stage")
class ActivityChallengeRandomStage(CustomAction):
    """从当前活动挑战页中随机点击一个可识别的 2-x 关卡。"""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        if context.tasker.stopping:
            return False
        image = context.tasker.controller.post_screencap().wait().get()
        if image is None or image.size == 0:
            logger.error("$agent.custom.action.activity.run.screenshot_failed")
            return False
        reco_detail = context.run_recognition("活动挑战_识别可选关卡", image)

        if not reco_detail or not reco_detail.hit:
            logger.error("$agent.custom.action.activity.run.selectable_stage_not_found")
            return False

        candidates = list(reco_detail.filtered_results or [])
        if not candidates:
            logger.error("$agent.custom.action.activity.run.recognition_result_empty")
            return False

        stage = random.choice(candidates)
        x, y, width, height = stage.box
        click_x = x + width // 2
        click_y = y + height // 2

        if context.tasker.stopping:
            return False
        if not context.tasker.controller.post_click(click_x, click_y).wait().succeeded:
            logger.error("$agent.custom.action.activity.run.click_failed")
            return False
        logger.info(
            "$agent.custom.action.activity.run.stage_selected",
            {
                "value_1": stage.text,
                "value_2": len(candidates),
                "value_3": click_x,
                "value_4": click_y,
            },
        )
        return True


@AgentServer.custom_action("activity_challenge_battle_loop")
class ActivityChallengeBattleLoop(CustomAction):
    """按用户输入次数重复挑战关卡战斗流程。"""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        count = self._parse_count(argv.custom_action_param)
        if count is None:
            return False

        for current in range(1, count + 1):
            if context.tasker.stopping:
                return False

            logger.info(
                "$agent.custom.action.activity.run.challenge_started",
                {
                    "value_1": current,
                    "value_2": count,
                },
            )
            battle_result = context.run_task("活动挑战_单次挑战")
            if not battle_result or not battle_result.status.succeeded:
                logger.error(
                    "$agent.custom.action.activity.run.battle_not_completed",
                    {
                        "value_1": current,
                        "value_2": count,
                    },
                )
                return False

            if context.tasker.stopping:
                return False
            settle_result = context.run_task("活动挑战_结算并返回")
            if not settle_result or not settle_result.status.succeeded:
                logger.error(
                    "$agent.custom.action.activity.run.return_to_stage_list_failed",
                    {
                        "value_1": current,
                        "value_2": count,
                    },
                )
                return False

            logger.info(
                "$agent.custom.action.activity.run.challenge_completed",
                {
                    "value_1": current,
                    "value_2": count,
                },
            )

        return True

    @staticmethod
    def _parse_count(raw: object) -> int | None:
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            count = int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            logger.error(
                "$agent.custom.action.activity._parse_count.invalid_challenge_count",
                {
                    "value_1": raw,
                    "value_2": exc,
                },
            )
            return None

        if count < 1:
            logger.error(
                "$agent.custom.action.activity._parse_count.challenge_count_must_be_positive",
                {
                    "value_1": count,
                },
            )
            return None

        return count
