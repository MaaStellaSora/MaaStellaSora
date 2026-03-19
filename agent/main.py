import sys

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

# 导入自定义的action和recognition，以注册到AgentServer
import custom # noqa: F401

# 开启debug_mode
from utils import logger
logger.debug_mode()



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
