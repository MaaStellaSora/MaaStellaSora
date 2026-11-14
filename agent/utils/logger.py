"""
一个基于MFAAvalonia格式的控制台日志输出功能

使用方法：
    from utils.logger import get_logger, debug_mode

    logger = get_logger(__name__)
    logger.info("这是一条信息")

    # 开启调试模式，调试模式会在项目根目录下的 debug/agent 目录下创建日志文件
    debug_mode()
    logger.debug("这是调试信息")
"""

import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


def _read_translations(language: str) -> dict[str, str] | None:
    translation_file = Path(__file__).resolve().parents[1] / "agent_config" / "i18n" / f"{language}.json"
    try:
        translations = json.loads(translation_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None

    if isinstance(translations, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in translations.items()
    ):
        return translations
    return None


def _load_translations() -> dict[str, str]:
    translations = _read_translations("zh_cn") or {}
    if lang and lang != "zh_cn":
        translations.update(_read_translations(lang) or {})
    return translations


class I18nFilter(logging.Filter):
    """Translate messages written as $key and interpolate {variables}."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_i18n_processed", False):
            return True
        record._i18n_processed = True
        record.i18n_key = "-"

        if not isinstance(record.msg, str) or not record.msg.startswith("$"):
            return True

        record.i18n_key = record.msg[1:]
        translation = _TRANSLATIONS.get(record.i18n_key)
        if translation is None:
            if isinstance(record.args, Mapping):
                record.args = ()
            return True

        variables = record.args if isinstance(record.args, Mapping) else {}
        record.msg = translation.format_map(variables)
        record.args = ()
        return True


class LowerLevelNameFormatter(logging.Formatter):
    """自定义格式化器：将日志级别转换为小写，以适配MFAAvalonia格式"""

    def format(self, record: logging.LogRecord) -> str:
        # 保存原始级别名称
        orig_levelname = record.levelname
        # 转换为小写
        record.levelname = orig_levelname.lower()
        # 格式化
        result = super().format(record)
        # 恢复原始级别名称（避免影响其他handler）
        record.levelname = orig_levelname
        return result


class _ConsoleHandler(logging.StreamHandler):
    pass


class _FileHandler(logging.FileHandler):
    pass


_CUSTOM_LOGGER_NAME = "custom"
_custom_logger_initialized = False
lang = os.environ.get("PI_CLIENT_LANGUAGE", "")
_TRANSLATIONS = _load_translations()


def translate(key: str, **variables: object) -> str:
    """Translate an inline fragment using the same language as user logs."""
    return _TRANSLATIONS[key.removeprefix("$")].format_map(variables)


def get_logger(name: str = "custom") -> logging.Logger:
    """
    获取或创建一个Logger实例

    Args:
        name: logger名称，推荐使用 __name__

    Returns:
        Logger实例

    custom 命名空间的日志通过父 logger 统一输出；控制台默认显示 INFO 及以上，
    文件始终记录 DEBUG 及以上。debug_mode() 会让控制台也显示 DEBUG。
    """
    if name == _CUSTOM_LOGGER_NAME or name.startswith(f"{_CUSTOM_LOGGER_NAME}."):
        _configure_custom_logger()
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    if not any(isinstance(handler, _ConsoleHandler) for handler in logger.handlers):
        logger.setLevel(logging.DEBUG)
        _add_console_handler(logger)
    if not any(isinstance(handler, _FileHandler) for handler in logger.handlers):
        _add_file_handler(logger)
    logger.propagate = False
    return logger


def _add_console_handler(logger: logging.Logger) -> None:
    handler = _ConsoleHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(LowerLevelNameFormatter("%(levelname)s:%(message)s"))
    handler.addFilter(I18nFilter())
    logger.addHandler(handler)


def _add_file_handler(logger: logging.Logger) -> None:
    if any(isinstance(handler, _FileHandler) for handler in logger.handlers):
        return

    log_dir = Path(__file__).resolve().parents[2] / "debug" / "agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    handler = _FileHandler(log_file, encoding="utf-8")
    handler.addFilter(I18nFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s|%(levelname)s|%(name)s|%(i18n_key)s|%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)


def _configure_custom_logger() -> logging.Logger:
    global _custom_logger_initialized

    logger = logging.getLogger(_CUSTOM_LOGGER_NAME)
    if not _custom_logger_initialized:
        logger.setLevel(logging.DEBUG)
        _custom_logger_initialized = True
    if not any(isinstance(handler, _ConsoleHandler) for handler in logger.handlers):
        _add_console_handler(logger)
    if not any(isinstance(handler, _FileHandler) for handler in logger.handlers):
        _add_file_handler(logger)
    return logger


def debug_mode() -> None:
    """开启 custom 命名空间的控制台 DEBUG 日志。"""
    logger = _configure_custom_logger()
    for handler in logger.handlers:
        if isinstance(handler, _ConsoleHandler):
            handler.setLevel(logging.DEBUG)
            break
    else:
        raise RuntimeError("custom logger has no console handler")


def set_log_level(level: int) -> None:
    """设置 custom 命名空间的日志级别。"""
    _configure_custom_logger().setLevel(level)


if __name__ == "__main__":
    _TRANSLATIONS["test.greeting"] = "Hello, {name}!"
    logger = get_logger("custom.example")
    logger.info(f"当前语言: {lang}")
    logger.info("$test.greeting", {"name": "MaaStellaSora"})
    file_logger = logging.getLogger(_CUSTOM_LOGGER_NAME)
    file_handler = next(handler for handler in file_logger.handlers if isinstance(handler, _FileHandler))
    file_handler.flush()
    log_content = Path(file_handler.baseFilename).read_text(encoding="utf-8")
    assert "|test.greeting|Hello, MaaStellaSora!" in log_content

    logger.debug("这条debug不会显示")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")

    logger.info("--- 开启调试模式 ---")
    debug_mode()

    logger.debug("现在debug可以显示了")
    logger.debug("$test.greeting", {"name": "MaaStellaSora"})
    logger.info("调试模式下的信息")

    logger.info("--- 测试不同的custom logger ---")
    logger2 = get_logger("custom.module2")
    logger2.debug("来自module2的debug消息")
    logger2.info("来自module2的info消息")

    logger.info("--- 测试调试模式后新建的custom logger ---")
    logger3 = get_logger("custom.module3")
    logger3.debug("module3在debug_mode()之后创建，debug消息也应正常显示")
    logger3.info("module3在debug_mode()之后创建")
