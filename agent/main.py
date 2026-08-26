import sys
import os
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

# 添加agent目录，解决便携版python不自动添加脚本目录的问题
AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# 导入自定义的action和recognition，以注册到AgentServer
import custom # noqa: F401

# 仅对爬塔相关模块开启 DEBUG 日志输出（其余模块保持默认 INFO）
from utils import logger
logger.enable_debug_loggers(
    "climb_tower_shop",
    "climb_tower_potential",
    "climb_tower_loop",
    "climb_tower_preparation",
    "climb_tower_quiz",
)

# 全局调试模式（如需所有模块都 DEBUG，可设环境变量 APP_DEBUG=true）
if os.getenv("APP_DEBUG", "false").lower() == "true":
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
