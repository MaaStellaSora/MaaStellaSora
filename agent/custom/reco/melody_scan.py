"""进入商店前读取各音符的持有数量。

界面操作全部通过 pipeline 节点完成（打开背包 / 秘纹技能 / 技能音符说明 / 滑动 / 关闭），
本模块只负责调度这些节点、并按垂直位置把“音符名”与“数量”配对，
不发送任何裸坐标点击，也不在代码中出现音符的显示名称。

音符显示名 -> 内部名（aqua 等）的映射写在 pipeline 节点的 replace 里，
因此多服务器只需各自维护资源文件即可。
"""

from maa.context import Context

from utils import logger as logger_module
logger = logger_module.get_logger("melody_scan")

# 与 Data.melody_of_xxx 的 xxx 一一对应
MELODY_KEYS = (
    "aqua", "ignis", "terra", "ventus", "lux", "umbra",
    "focus", "skill", "ultimate", "pummel", "luck", "burst", "stamina",
)

OPEN_BAG_NODE = "星塔_背包_扫描音符_打开背包界面_agent"
CLOSE_BAG_NODE = "星塔_关闭背包界面_agent"
SCROLL_NODE = "星塔_背包_音符效果界面_向下滑动_agent"
NAME_NODE = "星塔_背包_识别音符名称_agent"
COUNT_NODE = "星塔_背包_识别音符数量_agent"

# 一屏放不下全部音符，最多向下翻这么多屏
MAX_SCROLL = 5
# 名字与数量垂直中心相差在此范围内视为同一行
ROW_TOLERANCE = 30


def scan_melody_counts(context: Context, data) -> dict[str, int]:
    """读取背包中各音符的持有数量。

    仅当配置里存在任一 melody_of_xxx > 0 时才读取；否则直接返回空字典，
    不做任何多余操作（不打开背包、不截图）。

    Args:
        context: 任务上下文。
        data: 商店配置数据，用于判断是否设置了音符目标。

    Returns:
        dict[str, int]: {melody_of_xxx: 持有数量}；未设目标或读取失败时为空字典。
    """
    targets = [k for k in MELODY_KEYS if data.get_melody_target(f"melody_of_{k}") > 0]
    if not targets:
        logger.debug("未设置音符数量目标，跳过音符读取")
        return {}

    counts: dict[str, int] = {}
    try:
        if not _open_melody_detail(context):
            return {}
        for _ in range(MAX_SCROLL):
            image = context.tasker.controller.post_screencap().wait().get()
            counts.update(_read_screen(context, image, targets))
            if len(counts) >= len(targets):
                break
            _run(context, SCROLL_NODE)
    except Exception as exc:
        logger.error(f"读取音符数量失败：{exc}")
    finally:
        _run(context, CLOSE_BAG_NODE)

    if counts:
        logger.debug(f"读取到音符数量：{counts}")
    else:
        logger.warning("未能读取到任何音符数量")
    return counts


def _open_melody_detail(context: Context) -> bool:
    """依次打开 背包 -> 秘纹技能 -> 技能音符说明，任一步失败即放弃。"""
    if not _run(context, OPEN_BAG_NODE):
        logger.error("打开音符说明界面失败")
        return False
    return True


def _read_screen(context: Context, image, targets: list[str]) -> dict[str, int]:
    """读取当前一屏内的音符数量。"""
    names = _read_names(context, image)
    numbers = _read_counts(context, image)
    found: dict[str, int] = {}
    for key in targets:
        box = names.get(key)
        if box is None:
            continue
        count = _match_row(box, numbers)
        if count is not None:
            found[f"melody_of_{key}"] = count
    return found


def _read_names(context: Context, image) -> dict[str, list[int]]:
    """识别当前一屏的音符名 -> 名称框；pipeline 已把显示名替换成内部名。"""
    results = _recognize(context, NAME_NODE, image)
    logger.debug(f"识别到音符名称：{results}")
    return {r.text: r.box for r in results if r.text}


def _read_counts(context: Context, image) -> list[tuple[int, list[int]]]:
    """识别当前一屏的数量 -> 数字框。"""
    out: list[tuple[int, list[int]]] = []
    results = _recognize(context, COUNT_NODE, image)
    logger.debug(f"识别到音符数量：{results}")
    for r in results:
        if r.text and r.text.isdigit():
            out.append((int(r.text), r.box))
    return out


def _recognize(context: Context, node: str, image) -> list:
    detail = context.run_recognition(node, image)
    return list(detail.filtered_results) if detail and detail.hit else []


def _match_row(box: list[int], numbers: list[tuple[int, list[int]]]) -> int | None:
    """取与名称同一行（垂直中心最近）的数量。

    Args:
        box: 名称的识别框。
        numbers: 本屏识别到的 (数量, 框) 列表。

    Returns:
        int | None: 匹配到的数量，没有同行项时返回 None。
    """
    center = box[1] + box[3] / 2
    best: tuple[int, float] | None = None
    for count, nbox in numbers:
        ncenter = nbox[1] + nbox[3] / 2
        offset = abs(center - ncenter)
        if offset > ROW_TOLERANCE:
            continue
        if best is None or offset < best[1]:
            best = (count, offset)
    return best[0] if best else None


def _run(context: Context, node: str) -> bool:
    """执行一个 pipeline 节点。"""
    result = context.run_task(node)
    if not result or not result.status.succeeded:
        return False
    return True
