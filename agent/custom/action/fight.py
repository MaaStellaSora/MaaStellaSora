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
        raw = argv.custom_action_param
        if raw is None:
            return True

        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            if isinstance(raw, str):
                raw = raw.strip()
                if not raw:
                    return True
                value = int(raw)
            else:
                value = int(raw)
        except Exception as exc:
            print(f"utool_calc_repeat: invalid param {raw!r}: {exc}")
            return True

        if value < 1:
            value = 1

        if value <= 1:
            # No extra runs needed: skip the "add times" click and go on.
            context.override_pipeline(
                {
                    "活动_添加战斗次数": {
                        "recognition": {"type": "DirectHit", "param": {}},
                        "action": {"type": "DoNothing", "param": {}},
                        "next": ["活动_确认", "活动_开始战斗"],
                    }
                }
            )
            print("utool_calc_repeat: input=1, skip add times")
            return True

        repeat = value - 1
        context.override_pipeline({"活动_添加战斗次数": {"repeat": repeat}})
        print(f"utool_calc_repeat: input={value}, repeat={repeat}")
        return True
