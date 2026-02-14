import logging
from maa.tasker import Tasker
from maa.define import LoggingLevelEnum

"""
    一个基于MFAAvalonia格式的控制台日志输出功能
    
    使用方法：
        在需要使用日志输出的地方可以通过以下方法导入
        from utils.logger import get_logger
"""

class UIPureTextFormatter(logging.Formatter):

    def format(self, record):
        # 记录原始级别名称
        orig_levelname = record.levelname
        # 强制转为小写
        record.levelname = orig_levelname.lower()

        # 执行格式化
        result = super().format(record)

        # 还原级别名称（防止影响其他 Handler）
        record.levelname = orig_levelname
        return result


def get_logger(name="my_app"):
    # 更改输出日志级别为 Info 及以上级别
    Tasker.set_stdout_level(LoggingLevelEnum.Info)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console_handler = logging.StreamHandler()

        # 严格遵循 levelname:message 格式
        # 输出示例：info:this is a message
        fmt = "%(levelname)s:%(message)s"
        formatter = UIPureTextFormatter(fmt)

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def debug_mode():
    # debug时用这个函数，就可以输出所有级别的日志
    Tasker.set_stdout_level(LoggingLevelEnum.All)