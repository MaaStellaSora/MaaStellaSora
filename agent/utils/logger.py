import logging
import os
from datetime import datetime

"""
一个基于MFAAvalonia格式的控制台日志输出功能

使用方法：
    from utils.logger import get_logger, debug_mode

    logger = get_logger(__name__)
    logger.info("这是一条信息")

    # 开启调试模式
    debug_mode()
    logger.debug("这是调试信息")
"""


class UIPureTextFormatter(logging.Formatter):
    """自定义格式化器：将日志级别转换为小写"""

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


# 用于追踪已初始化的logger
_initialized_loggers: set[str] = set()

# 全局调试模式状态及日志文件路径，确保后续新建logger也能感知
_debug_mode_enabled: bool = False
_debug_log_file: str | None = None


def get_logger(name: str = "my_app") -> logging.Logger:
    """
    获取或创建一个Logger实例

    Args:
        name: logger名称，推荐使用 __name__

    Returns:
        Logger实例

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("应用启动")
    """
    logger = logging.getLogger(name)

    # 避免重复初始化
    if name not in _initialized_loggers:
        # 设置 logger 级别为 DEBUG，由 handler 控制实际输出级别
        logger.setLevel(logging.DEBUG)

        # 创建控制台handler
        console_handler = logging.StreamHandler()
        # handler 默认级别为 INFO
        console_handler.setLevel(logging.INFO)

        # 设置格式化器：严格遵循 levelname:message 格式
        fmt = "%(levelname)s:%(message)s"
        formatter = UIPureTextFormatter(fmt)
        console_handler.setFormatter(formatter)

        # 添加handler
        logger.addHandler(console_handler)

        # 标记为已初始化
        _initialized_loggers.add(name)

        # 修复：若调试模式已开启，新建logger也应立即应用
        if _debug_mode_enabled and _debug_log_file:
            _apply_debug_to_logger(logger, _debug_log_file)

    return logger


def _apply_debug_to_logger(logger: logging.Logger, log_file: str) -> None:
    """
    对单个logger应用调试模式：添加独立的文件处理器并将所有handler级别设为DEBUG。

    修复：为每个logger创建独立的FileHandler实例，避免多logger共享同一
    handler导致的重复写入问题。

    Args:
        logger:   目标logger实例
        log_file: 日志文件的完整路径
    """
    # 检查是否已经添加了文件处理器（避免重复添加）
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    if not has_file_handler:
        # 修复：每个logger使用独立的FileHandler实例
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        file_fmt = "%(asctime)s|%(levelname)s|%(message)s"
        file_formatter = logging.Formatter(file_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    # 将所有handler（含控制台）级别设为DEBUG
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

    logger.debug(f"Debug mode enabled. Log file: {log_file}")


def debug_mode() -> None:
    """
    开启调试模式：
    - 将所有已创建logger的handler级别设置为DEBUG
    - 为每个logger添加独立的文件处理器，将日志保存到 debug/agent_debug/{年-月-日}.log
    - 记录全局状态，确保之后新建的logger也自动应用调试模式
    """
    global _debug_mode_enabled, _debug_log_file

    # 基于文件自身位置推算项目根目录（/agent/utils/logger.py -> 上两级）
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "debug", "agent")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    # 记录全局状态，供后续 get_logger() 使用
    _debug_mode_enabled = True
    _debug_log_file = log_file

    # 对所有已初始化的logger应用调试模式
    for logger_name in _initialized_loggers:
        logger = logging.getLogger(logger_name)
        _apply_debug_to_logger(logger, log_file)


def set_log_level(level: int) -> None:
    """
    设置所有logger及其handler的日志级别

    修复：同步更新logger自身的level，保证logger与handler级别一致。

    Args:
        level: 日志级别（如 logging.INFO, logging.DEBUG 等）
    """
    for logger_name in _initialized_loggers:
        logger = logging.getLogger(logger_name)
        # 修复：同步更新logger本身的level
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


if __name__ == "__main__":
    # 测试代码
    logger = get_logger(__name__)

    logger.debug("这条debug不会显示")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")

    print("\n--- 开启调试模式 ---")
    debug_mode()

    logger.debug("现在debug可以显示了")
    logger.info("调试模式下的信息")

    print("\n--- 测试不同的logger ---")
    logger2 = get_logger("module2")
    logger2.debug("来自module2的debug消息")
    logger2.info("来自module2的info消息")

    print("\n--- 测试调试模式后新建的logger ---")
    logger3 = get_logger("module3")
    logger3.debug("module3在debug_mode()之后创建，debug消息也应正常显示")
    logger3.info("module3的info消息")