from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
import json


@AgentServer.custom_recognition("shop_recognition")
class ShopRecognition(CustomRecognition):
    """
        商店识别器 - 简化版，仅用于触发动作

        通过在interface.json更改商店节点的next为custom节点，然后在custom节点中通过以下override方式转到此函数处理。
        例子：
        "custom_recognition": "shop_recognition",
            "custom_recognition_param": {
                "type": "complete_shop_flow",
                "shop_type": "regular"
            }
        type: 固定为complete_shop_flow
        shop_type: 分为regular和final两种，分别对应普通商店和最终商店。
    """
    
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        """简化的商店识别，直接返回成功，让动作器处理所有逻辑"""
        
        config = argv.custom_recognition_param
        print(f"商店识别器参数: {config}")

        # 获取具体参数
        box, shop_type, current_state = self._get_shop_type(config)

        # 返回识别结果
        return CustomRecognition.AnalyzeResult(
            box=box,
            detail={
                "current_state": current_state,
                "shop_type": shop_type
            }
        )

    @staticmethod
    def _get_shop_type(config: str) -> tuple[list, str, str]:
        """
            获取商店类型

            Args:
                config(str): 商店识别器参数，默认为json格式的str

            Returns:
                box(list): 识别结果，成功时返回[0, 0, 10, 10]，失败时返回None。成功时返回识别框可以是任意值，让pipeline认为识别成功
                shop_type(str): 商店类型，默认为regular
                current_state(str): 当前状态，默认为ready，运行出错时返回报错原因
        """

        try:
            # 如果 config 是字符串，尝试解析为 JSON 对象
            if isinstance(config, str):
                shop_config = json.loads(config)
            else:
                shop_config = config

            # 取shop_type的值，如取不到，则默认为regular
            shop_type = shop_config.get("shop_type", "regular")
            # 给其他变量赋值
            box = [0, 0, 10, 10]
            current_state = "ready"
            print(f"商店类型: {shop_type}")

        except Exception as e:
            print(f"商店识别器错误: {e}")
            box = None
            shop_type = ""
            current_state = f"Error: {str(e)}"

        return box, shop_type, current_state


@AgentServer.custom_recognition("choose_potential_recognition")
class ChoosePotentialRecognition(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        """
            选择潜能
        """
        # 判断是核心潜能还是普通潜能

        # 读取当前三个潜能的名称

        # 判断选哪个

        #判断不出来，识别系统推荐，并返回系统推荐的box

        return CustomRecognition.AnalyzeResult(
            box=[0, 0, 0, 0],
            detail={}
        )
