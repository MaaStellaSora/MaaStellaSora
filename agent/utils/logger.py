import logging

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
_initialized_loggers = set()


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

    return logger


def debug_mode() -> None:
    """
    开启调试模式
    将所有已创建的logger的handler级别设置为DEBUG
    """

    # 将所有已初始化的logger的handler级别设为DEBUG
    for logger_name in _initialized_loggers:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")


def set_log_level(level: int) -> None:
    """
    设置所有logger的日志级别

    Args:
        level: 日志级别（如 logging.INFO, logging.DEBUG等）
    """
    for logger_name in _initialized_loggers:
        logger = logging.getLogger(logger_name)
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