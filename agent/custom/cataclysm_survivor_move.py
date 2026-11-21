"""灾变防线上下半场之间的传送门走位。

等待阶段每次只比较一帧并返回，让 Pipeline 继续处理命运卡片和结算界面。
画面持续静止后，分阶段按住 W+A、S+D，检测加载黑屏确认进入下半场。
键盘需使用前台控制器的 Seize；每段走位前后释放方向键，停止或异常也须清理。

1280×720 实测静止画面变化率约 1.3%~1.9%；阈值、等待和走位时长见节点 attach。
"""

from __future__ import annotations

import time
from threading import Lock

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger as logger_module
from utils.cataclysm_survivor import IO_TIMEOUT, grab as _grab, wait_job

logger = logger_module.get_logger(__name__)

## Win32 虚拟键码
KEY_W = 0x57
KEY_A = 0x41
KEY_S = 0x53
KEY_D = 0x44
ALL_KEYS = (KEY_W, KEY_A, KEY_S, KEY_D)

## 整屏平均亮度低于该值即认为进了加载（黑屏）
BLACK_MEAN_THRESHOLD = 25.0

## 相邻两次取样间隔超过这么多秒就不比较（间隔太久，动态可能被平均掉）
SAMPLE_GAP_LIMIT = 3.0

## 抽帧步长（每隔几个像素取一个点），够用且快
SAMPLE_STRIDE = 4

## 变化率心跳日志间隔（秒）—— 实机调 still_ratio 的依据
RATIO_LOG_INTERVAL = 10.0

## 松开按键后：先等这么久让松开生效（实测有约 1 秒延迟），再采样这么久判断停没停
SETTLE_WAIT = 1.2
SETTLE_SAMPLE = 0.8

## 走位阶段
PHASE_WAIT = "wait"  ## 等画面静止 = 等上半场打完
PHASE_LEFT = "left"  ## 按住 W+A 走向左上角
PHASE_PORTAL = "portal"  ## 按住 S+D 走向中央传送门
PHASE_DONE = "done"  ## 已进门 / 已达上限

## 走位状态（由 reset 动作清零）
_state: dict = {}
_owner_job = None
_action_lock = Lock()


def _reset_state() -> None:
    _state.clear()
    _state.update(
        started_at=None,
        force_at=0.0,
        phase=PHASE_WAIT,
        attempt=0,
        entered=False,
        last_gray=None,
        last_at=0.0,
        still_since=None,
        ratio_log_at=0.0,
    )


_reset_state()


def _claim_run(context: Context) -> bool:
    """同一 Agent 一次只服务一个灾变任务，不能由另一个运行重置状态或抢占按键。"""
    global _owner_job
    job = context.get_task_job()
    if _owner_job is not None and _owner_job.job_id != job.job_id and not _owner_job.done:
        logger.error(
            "$agent.custom.action.cataclysm_survivor_move._claim_run.concurrent_task_rejected"
        )
        return False
    if _owner_job is None or _owner_job.job_id != job.job_id:
        _reset_state()
        _owner_job = job
    return True


def _drive(context: Context, seconds: float) -> bool:
    """用截图驱动等待（不 sleep），期间顺带响应停止。"""

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if context.tasker.stopping:
            return False
        if _grab(context) is None:
            return False
    return True


def _to_gray(image):
    """抽点 + 转灰度，返回 int16 二维数组。"""

    small = np.asarray(image)[::SAMPLE_STRIDE, ::SAMPLE_STRIDE]
    if small.ndim == 3:
        return small.astype(np.int16).mean(axis=2)
    return small.astype(np.int16)


def _is_black(image) -> bool:
    """整屏平均亮度过低即认为在加载（黑屏）。"""

    if image is None or getattr(image, "size", 0) == 0:
        return False
    return float(np.asarray(image)[::SAMPLE_STRIDE, ::SAMPLE_STRIDE].mean()) < BLACK_MEAN_THRESHOLD


def _diff_ratio(before, after, pixel_diff: int) -> float:
    """两帧的差异像素占比；拿不到图返回 -1。"""

    if before is None or after is None:
        return -1.0
    if getattr(before, "size", 0) == 0 or getattr(after, "size", 0) == 0:
        return -1.0
    ga, gb = _to_gray(before), _to_gray(after)
    if ga.shape != gb.shape:
        return -1.0
    return float((np.abs(ga - gb) > pixel_diff).mean())


def _release_all(context: Context) -> bool:
    """松开全部方向键，并且连发两遍。

    单个 key_up 在实机上出现过「发了但角色还在走」的情况；四个键都发一遍还能
    顺带清掉上一次运行 / 上一阶段遗留的按键状态（本游戏 W+A 与 S+D 同按会抵消，
    残留一个键就会让后续的走位整个失灵）。
    """

    controller = context.tasker.controller
    jobs = []
    for _ in range(2):
        for key in ALL_KEYS:
            try:
                jobs.append((key, controller.post_key_up(key)))
            except Exception:  # noqa: BLE001 - 任务停止时按键任务可能已失效
                pass
    released = set()
    deadline = time.monotonic() + IO_TIMEOUT
    for key, job in jobs:
        if wait_job(context, job, timeout=max(0.0, deadline - time.monotonic()), cleanup=True):
            released.add(key)
    missing = set(ALL_KEYS) - released
    if missing:
        logger.error(
            "$agent.custom.action.cataclysm_survivor_move._release_all.key_release_unconfirmed",
            {
                "value_1": sorted(missing),
            },
        )
    return not missing


def _settle(context: Context, still_ratio: float, pixel_diff: int) -> tuple[bool, float]:
    """松开按键后确认角色真的停下了；缺图或仍在运动均不能推进阶段。

    Returns:
        (是否确认停下, 观测到的画面变化率)
    """

    if not _drive(context, SETTLE_WAIT):
        return False, -1.0
    before = _grab(context)
    if not _drive(context, SETTLE_SAMPLE):
        return False, -1.0
    after = _grab(context)
    ratio = _diff_ratio(before, after, pixel_diff)
    return 0 <= ratio < still_ratio, ratio


def _update_stillness(image, now: float, still_seconds: float, still_ratio: float, pixel_diff: int) -> bool:
    """把这一帧和上一帧比一比，判断画面是否已经连续静止 still_seconds 秒。

    Returns:
        bool: True 表示画面够久了，可以开始尝试进入传送门。
    """

    if image is None or getattr(image, "size", 0) == 0:
        _state["last_gray"] = None
        _state["still_since"] = None
        return False

    ## 加载黑屏是完全静止的，不能算「上半场结束」
    if _is_black(image):
        _state["last_gray"] = None
        _state["still_since"] = None
        return False

    gray = _to_gray(image)
    last = _state.get("last_gray")
    last_at = _state.get("last_at", 0.0)

    if last is not None and last.shape == gray.shape and (now - last_at) <= SAMPLE_GAP_LIMIT:
        ratio = float((np.abs(gray - last) > pixel_diff).mean())

        ## 限流心跳：把实测变化率写进日志，方便按实机数值调 still_ratio
        if now - _state.get("ratio_log_at", 0.0) >= RATIO_LOG_INTERVAL:
            _state["ratio_log_at"] = now
            counting = "，静止计时中" if _state.get("still_since") else ""
            logger.info(
                "$agent.custom.action.cataclysm_survivor_move._update_stillness.screen_change_rate",
                {
                    "ratio": ratio,
                    "still_ratio": still_ratio,
                    "counting": counting,
                },
            )

        if ratio < still_ratio:
            since = _state.get("still_since")
            if since is None:
                _state["still_since"] = now
                logger.info(
                    "$agent.custom.action.cataclysm_survivor_move._update_stillness.screen_still",
                    {
                        "ratio": ratio,
                        "still_seconds": still_seconds,
                    },
                )
            elif now - since >= still_seconds:
                _state["last_gray"] = gray
                _state["last_at"] = now
                return True
        else:
            if _state.get("still_since") is not None:
                logger.info(
                    "$agent.custom.action.cataclysm_survivor_move._update_stillness.screen_motion_resumed",
                    {
                        "ratio": ratio,
                    },
                )
            _state["still_since"] = None

    else:
        # 缺帧、分辨率变化或过大的采样间隔都会中断连续静止计时。
        _state["still_since"] = None

    _state["last_gray"] = gray
    _state["last_at"] = now
    return False


def _hold(context: Context, keys: tuple[int, ...], seconds: float) -> str:
    """按住一组键 seconds 秒（用截图驱动等待），结束一定把四个键都松开。

    Returns:
        str: "entered" 检测到黑屏；"stopped" 用户停止；"timeout" 按满时间；"failed" 输入或截图失败。
    """

    ## 先清干净再按：避免上一阶段残留的键把这一次的走位抵消掉
    if not _release_all(context):
        return "failed"

    controller = context.tasker.controller
    outcome = "timeout"
    try:
        for key in keys:
            if context.tasker.stopping:
                outcome = "stopped"
                break
            if not wait_job(context, controller.post_key_down(key)):
                outcome = "stopped" if context.tasker.stopping else "failed"
                if outcome == "failed":
                    logger.error(
                        "$agent.custom.action.cataclysm_survivor_move._hold.key_press_failed",
                        {
                            "value_1": key,
                        },
                    )
                break

        deadline = time.monotonic() + seconds
        while outcome == "timeout" and time.monotonic() < deadline:
            if context.tasker.stopping:
                outcome = "stopped"
                break
            image = _grab(context)
            if image is None:
                logger.error("$agent.custom.action.cataclysm_survivor_move._hold.screenshot_failed")
                outcome = "failed"
                break
            if _is_black(image):
                outcome = "entered"
                break
    except Exception:  # noqa: BLE001 - 控制器异常时仍须释放已按下的方向键
        logger.exception("灾变防线：走位按键执行异常")
        outcome = "failed"
    finally:
        if not _release_all(context):
            outcome = "failed"
    return outcome


@AgentServer.custom_action("cataclysm_survivor_reset")
class CataclysmSurvivorReset(CustomAction):
    """任务开始时清掉上一次运行留下的走位状态和按键。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        # 反向 RPC 可能重入 Agent 回调；不阻塞等锁，避免阻塞原动作的清理。
        if not _action_lock.acquire(blocking=False):
            logger.error(
                "$agent.custom.action.cataclysm_survivor_move.run.concurrent_reset_rejected"
            )
            return False
        try:
            if not _claim_run(context):
                return False
            _reset_state()
            return _release_all(context)
        finally:
            _action_lock.release()


@AgentServer.custom_action("cataclysm_survivor_move")
class CataclysmSurvivorMove(CustomAction):
    """尝试进入传送门：等画面静止再走，等待阶段秒回。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        if not _action_lock.acquire(blocking=False):
            logger.error(
                "$agent.custom.action.cataclysm_survivor_move.run.concurrent_invocation_rejected"
            )
            return False
        try:
            return self._run(context, argv) if _claim_run(context) else False
        finally:
            _action_lock.release()

    def _run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        if context.tasker.stopping or _state.get("entered"):
            return True

        node_data = context.get_node_data(argv.node_name) or {}
        attach = node_data.get("attach", {}) or {}
        min_wait = float(attach.get("min_wait", 120))
        still_seconds = float(attach.get("still_seconds", 8))
        still_ratio = float(attach.get("still_ratio", 0.05))
        pixel_diff = int(attach.get("pixel_diff", 16))
        max_wait = float(attach.get("max_wait", 330))
        left_seconds = float(attach.get("left_seconds", 12))
        portal_seconds = float(attach.get("portal_seconds", 5))
        attempts = int(attach.get("attempts", 8))

        now = time.monotonic()

        ## 首次被调用：只记时间，立刻让战斗循环回去识别命运卡片
        if _state.get("started_at") is None:
            _state["started_at"] = now
            _state["force_at"] = now + max_wait
            _state["phase"] = PHASE_WAIT
            logger.info(
                "$agent.custom.action.cataclysm_survivor_move._run.combat_loop_started",
                {
                    "min_wait": min_wait,
                    "still_seconds": still_seconds,
                    "max_wait": max_wait,
                },
            )
            return True

        phase = _state.get("phase", PHASE_WAIT)

        ## 等上半场打完（画面静止）：本动作秒回，战斗循环每轮都会重新识别命运卡片
        if phase == PHASE_WAIT:
            elapsed = now - _state["started_at"]
            image = _grab(context)
            ready = _update_stillness(image, time.monotonic(), still_seconds, still_ratio, pixel_diff)
            if image is None:
                logger.error("$agent.custom.action.cataclysm_survivor_move._run.screenshot_failed")
                return False

            ## 刚开始的一段时间不采信静止（避免上半场某段平静时误判）
            if elapsed < min_wait:
                _state["still_since"] = None
                ready = False

            ## 兜底：等太久了就强制试一次；之后重新计一轮
            if not ready and max_wait > 0 and now >= _state.get("force_at", 0.0):
                _state["force_at"] = now + max_wait
                ready = True
                logger.warning(
                    "$agent.custom.action.cataclysm_survivor_move._run.stillness_wait_limit_reached",
                    {
                        "max_wait": max_wait,
                    },
                )

            if ready:
                _state["phase"] = PHASE_LEFT
                _state["attempt"] = _state.get("attempt", 0) + 1
                logger.info(
                    "$agent.custom.action.cataclysm_survivor_move._run.portal_attempt",
                    {
                        "attempt": _state['attempt'],
                        "attempts": attempts,
                    },
                )
            return True

        if phase == PHASE_LEFT:
            keys, seconds, label = (KEY_W, KEY_A), left_seconds, "W+A"
        elif phase == PHASE_PORTAL:
            keys, seconds, label = (KEY_S, KEY_D), portal_seconds, "S+D"
        else:
            return True

        attempt = _state.get("attempt", 1)
        before = _grab(context)
        ## 连续按住方向键；期间每轮截图检查是否进了加载
        outcome = _hold(context, keys, seconds)
        after = _grab(context)

        moved = _diff_ratio(before, after, pixel_diff)
        if moved >= 0:
            hint = "（画面几乎没变，按键可能没生效）" if moved < still_ratio else ""
            logger.info(
                "$agent.custom.action.cataclysm_survivor_move._run.screen_change_rate",
                {
                    "attempt": attempt,
                    "attempts": attempts,
                    "label": label,
                    "seconds": seconds,
                    "moved": moved,
                    "hint": hint,
                },
            )

        if outcome == "stopped":
            return True
        if outcome == "failed":
            return False

        ## 松开自检：确认角色真的停下了，没停就补发松开（实机出现过粘键）
        if outcome != "entered":
            stopped, tail = _settle(context, still_ratio, pixel_diff)
            if not stopped:
                _release_all(context)
                logger.error(
                    "$agent.custom.action.cataclysm_survivor_move._run.movement_not_stable",
                    {
                        "label": label,
                        "tail": tail,
                    },
                )
                return False

        if outcome == "entered":
            _state["entered"] = True
            _state["phase"] = PHASE_DONE
            logger.info("$agent.custom.action.cataclysm_survivor_move._run.portal_entered")
            return True

        if phase == PHASE_LEFT:
            _state["phase"] = PHASE_PORTAL
            logger.info(
                "$agent.custom.action.cataclysm_survivor_move._run.portal_corner_reached",
                {
                    "attempt": attempt,
                    "attempts": attempts,
                },
            )
            return True

        ## PHASE_PORTAL 按满仍未进门
        if attempt >= attempts:
            _state["phase"] = PHASE_DONE
            logger.warning(
                "$agent.custom.action.cataclysm_survivor_move._run.portal_portal_attempt",
                {
                    "attempts": attempts,
                },
            )
        else:
            ## 回到「等画面静止」：若上半场其实没结束，画面会重新动起来，就继续等
            _state["phase"] = PHASE_WAIT
            _state["still_since"] = None
            _state["last_gray"] = None
            logger.info(
                "$agent.custom.action.cataclysm_survivor_move._run.portal_loading_not_detected",
                {
                    "attempt": attempt,
                    "attempts": attempts,
                },
            )
        return True
