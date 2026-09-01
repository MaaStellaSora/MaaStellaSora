import json
from pathlib import Path
from utils import logger as logger_module
logger = logger_module.get_logger("dev_config")


config_path = Path(__file__).resolve().parents[1] / "agent_config" / "dev_config.json"

DRAW_DATA_SAVE_ENABLED = False
DEV_IMAGES_SAVE_ENABLED = False
DEBUG_MODE = False

try:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    DRAW_DATA_SAVE_ENABLED = data.get("DRAW_DATA_SAVE_ENABLE", False)
    DEV_IMAGES_SAVE_ENABLED = data.get("DEV_IMAGES_SAVE_ENABLE", False)
    DEBUG_MODE = data.get("DEBUG_MODE", False)

except FileNotFoundError:
    logger.debug("agent配置文件不存在")
except json.JSONDecodeError:
    logger.debug("agent配置文件 JSON 格式错误")
except OSError:
    logger.debug("agent配置文件读取时发生文件 I/O 错误")
