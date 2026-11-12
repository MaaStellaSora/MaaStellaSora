from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from .interactor_template import Interactor

from utils import logger as logger_module
logger = logger_module.get_logger(__name__)


@AgentServer.custom_action("element_filter")
class ElementFilterAction(CustomAction):
    """根据用户输入的筛选条件，筛选出符合条件的元素，并进行操作。"""

    elements = ("aqua", "ignis", "terra", "ventus", "lux", "umbra")
    valid_elements = set(elements)
    valid_converts = {"", "vulnerability", "non_resistant"}
    vulnerability_elements = {
        "aqua": "terra",
        "ignis": "aqua",
        "terra": "ventus",
        "ventus": "ignis",
        "lux": "umbra",
        "umbra": "lux",
    }

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        # 获取参数
        node_data = context.get_node_data(argv.node_name) or {}
        attachments = node_data.get("attach", {})

        # 导入并检查elements参数
        elements = attachments.get("elements", [])
        invalid_elements = [e for e in elements if e not in self.valid_elements]
        if invalid_elements:
            logger.error(
                "$agent.custom.choose_record.run.invalid_elements",
                {"value_1": invalid_elements, "value_2": self.elements},
            )
            return False

        # 导入并检查convert参数
        convert = attachments.get("convert", "")
        if convert not in self.valid_converts:
            logger.error(
                "$agent.custom.choose_record.run.invalid_convert",
                {"value_1": convert},
            )
            return False

        if convert == "vulnerability":
            elements = self._convert_to_vulnerability(elements)
        elif convert == "non_resistant":
            elements = self._convert_to_non_resistant(elements)

        # 开始点击流程
        run_node_name = "__选择纪录_筛选窗口_选择单个属性"
        maa = Interactor(context)

        logger.debug(f"Choose elements: {elements}")
        for element in elements:
            pipeline_override = self._pipeline_override(run_node_name, element)
            if not maa.run(run_node_name, pipeline_override):
                logger.error(
                    "$agent.custom.choose_record.run.element_selection_failed",
                    {"value_1": element},
                )
                return False
        return True

    @classmethod
    def _convert_to_vulnerability(cls, elements: list[str]) -> list[str]:
        return list(dict.fromkeys(cls.vulnerability_elements[element] for element in elements))

    @classmethod
    def _convert_to_non_resistant(cls, elements: list[str]) -> list[str]:
        return list(dict.fromkeys(
            converted
            for element in elements
            for converted in cls.elements
            if converted != element
        ))

    @staticmethod
    def _pipeline_override(node_name: str, element: str) -> dict:
        return {
            node_name: {
                "recognition": {
                    "param": {
                        "template": f"common/elements/{element}.png"
                    }
                }
            }
        }
