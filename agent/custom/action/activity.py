import random

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger as logger_module


logger = logger_module.get_logger("activity_challenge")


@AgentServer.custom_action("activity_challenge_random_stage")
class ActivityChallengeRandomStage(CustomAction):
    """从当前活动挑战页中随机点击一个可识别的 2-x 关卡。"""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("活动挑战_识别可选关卡", image)

        if not reco_detail or not reco_detail.hit:
            logger.error("没有识别到可选择的 2-x 活动挑战关卡")
            return False

        candidates = list(reco_detail.filtered_results or [])
        if not candidates:
            logger.error("2-x 活动挑战关卡识别结果为空")
            return False

        stage = random.choice(candidates)
        x, y, width, height = stage.box
        click_x = x + width // 2
        click_y = y + height // 2

        context.tasker.controller.post_click(click_x, click_y).wait()
        logger.info(
            "随机选择活动挑战关卡：%s（候选 %d 个，点击坐标 %d,%d）",
            stage.text,
            len(candidates),
            click_x,
            click_y,
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

            logger.info("开始第 %d/%d 次活动挑战", current, count)
            battle_result = context.run_task("活动挑战_单次挑战")
            if not battle_result or not battle_result.status.succeeded:
                logger.error("第 %d/%d 次活动挑战未能完成战斗", current, count)
                return False

            settle_result = context.run_task("活动挑战_结算并返回")
            if not settle_result or not settle_result.status.succeeded:
                logger.error("第 %d/%d 次活动挑战未能返回关卡列表", current, count)
                return False

            logger.info("已完成第 %d/%d 次活动挑战", current, count)

        return True

    @staticmethod
    def _parse_count(raw: object) -> int | None:
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            count = int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            logger.error("活动挑战次数无效：%r（%s）", raw, exc)
            return None

        if count < 1:
            logger.error("活动挑战次数必须大于 0，实际为 %d", count)
            return None

        return count
