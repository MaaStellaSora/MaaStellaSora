from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

# import logging
#
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("debug/agent.log", encoding='utf-8'),
#         logging.StreamHandler()
#     ],
# )

@AgentServer.custom_action("InviteAuto")
class InviteAuto(CustomAction):
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        """
            邀约功能总控制节点
            仅使用agent判断选择对象是否符合要求
            使用pipeline执行邀约流程

            TODO:
                规避名字相撞问题
                实现拖曳选择角色
        """

        # 邀约对象的任务列表
        invite_node = ["邀约_1号", "邀约_2号", "邀约_3号", "邀约_4号", "邀约_5号"]

        # 是否为第一次邀约
        first_invite = True

        for node in invite_node:
            # 获取邀约对象信息，并去掉前后的空格
            # logging.info(f"获取邀约对象'{node}'信息，并去掉前后的空格")
            invite_info = context.get_node_data(node)
            try:
                invite_info = invite_info['recognition']['param']['expected'][0]
                invite_info = invite_info.strip()
            except Exception:
                invite_info = ""

            # 判断邀约对象是否为空，为空则跳过
            if invite_info is None or invite_info == "":
                # logging.info("邀约对象为空，跳过")
                continue

            # 执行邀约流程
            # 确定回到邀约界面
            context.run_task("邀约_回到邀约界面",
                             {
                                 "邀约_主界面":{
                                     "next": []
                                 }
                             })

            # 为了应对目前ocr模型对白色非中文字识别能力较差的问题，第一次邀约需要先尝试识别右上角的文字
            if first_invite:
                image = context.tasker.controller.post_screencap().wait().get()
                reco_detail = context.run_recognition("邀约_识别邀约对象", image,{
                    "邀约_识别邀约对象":{
                        "recognition": {
                            "param": {
                                "expected": [
                                    invite_info
                                ],
                                "roi": [
                                    720,
                                    80,
                                    320,
                                    50
                                ]
                            },
                            "type": "OCR"
                        }
                    }
                })

                if reco_detail and reco_detail.hit:
                    # logging.info(f"邀约对象名字'{invite_info}'通过右上角识别，开始邀约流程")
                    context.run_task("未邀约")
                    continue

            # 第二次及之后的邀约，或者没成功识别右上角文字的第一次邀约
            # logging.info(f"邀约对象为{invite_info}")
            context.run_task(node,
                             {
                                 node: {
                                     "next": ["未邀约"],
                                     "timeout": 5000
                                 }
                             })

        # 返回True，执行后续的“通用_返回主页”节点
        return True