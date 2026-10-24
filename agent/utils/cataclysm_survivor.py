"""灾变防线的控制器请求等待：轮询状态，避免无限阻塞在 Job.wait。"""

import time

from maa.context import Context
from maa.job import Job


IO_TIMEOUT = 2.0
POLL_INTERVAL = 0.02


def wait_job(context: Context, job: Job, *, timeout: float = IO_TIMEOUT, cleanup: bool = False) -> bool:
    """请求失败、停止或超时返回 False；清理请求在任务停止后仍尝试完成。"""
    deadline = time.monotonic() + max(0.0, timeout)
    try:
        while True:
            if not cleanup and context.tasker.stopping:
                return False
            if job.done:
                return job.succeeded
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(POLL_INTERVAL, remaining))
    except Exception:  # noqa: BLE001 - 控制器停止或断连按请求失败处理
        return False


def grab(context: Context):
    """在有限等待内获取截图；停止、请求失败、超时或空图均返回 None。"""
    try:
        if context.tasker.stopping:
            return None
        job = context.tasker.controller.post_screencap()
        if not wait_job(context, job):
            return None
        image = job.get()
        return image if image is not None and getattr(image, "size", 0) else None
    except Exception:  # noqa: BLE001 - 控制器断连按截图失败处理
        return None
