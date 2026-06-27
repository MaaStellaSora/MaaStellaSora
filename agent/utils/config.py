import json
from pathlib import Path
from utils import logger as logger_module
logger = logger_module.get_logger("config")


config_path = Path(__file__).resolve().parents[1] / "config.json"

DRAW_DATA_SAVE_ENABLED = False
DEV_IMAGES_SAVE_ENABLED = False

try:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    DRAW_DATA_SAVE_ENABLED = data.get("DRAW_DATA_SAVE_ENABLE", False)
    DEV_IMAGES_SAVE_ENABLED = data.get("DEV_IMAGES_SAVE_ENABLE", False)

except FileNotFoundError:
    logger.debug("agent配置文件不存在")
except json.JSONDecodeError:
    logger.debug("agent配置文件 JSON 格式错误")
except Exception:
    logger.debug("agent配置文件读取时发生未知错误")
