from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction



@AgentServer.custom_action("utool_calc_repeat")
class UToolCalcRepeat(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        if context.tasker.stopping:
            return False

        raw = argv.custom_action_param
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            if isinstance(raw, str):
                raw = raw.strip()
                value = int(raw)
            else:
                value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            print(f"utool_calc_repeat: invalid param {raw!r}: {exc}")
            return False

        if value < 1:
            print("utool_calc_repeat: 次数必须大于 0")
            return False

        if context.tasker.stopping:
            return False
        if value == 1:
            # 单次直接进入开始战斗，保留加次数节点的识别和动作。
            return context.override_next(argv.node_name, ["活动快速战斗_开始战斗"])

        repeat = value - 1
        if not context.override_pipeline({"活动快速战斗_添加战斗次数": {"repeat": repeat}}):
            return False
        print(f"utool_calc_repeat: input={value}, repeat={repeat}")
        return context.override_next(argv.node_name, ["活动快速战斗_添加战斗次数"])
