"""进入商店前读取各音符的持有数量。

界面操作全部通过 pipeline 节点完成（打开背包 / 秘纹技能 / 关闭），
本模块只负责调度这些节点，不发送任何裸坐标点击，也不在代码中出现音符的显示名称。
"""
import numpy as np
from maa.context import Context

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_melody_scan")

# 主要音符，与 Data.melody_of_xxx 名字一致，不含属性音符，因为属性音符会根据塔属性动态变化
MAIN_MELODIES = (
    "melody_of_focus",
    "melody_of_skill",
    "melody_of_ultimate",
    "melody_of_pummel",
    "melody_of_luck",
    "melody_of_burst",
    "melody_of_stamina",
)

ELEMENT_NODE = "星塔_属性塔选择_agent"
OPEN_BAG_NODE = "星塔_背包_扫描音符_打开背包界面_agent"
CLOSE_BAG_NODE = "星塔_关闭背包界面_agent"
MELODY_NODE = "星塔_背包_识别音符_agent"
COUNT_NODE = "星塔_背包_识别音符数量_agent"


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
    # 获取属性音符，整合成正确的音符列表
    active_element_melodies = _get_element_melodies(context, ELEMENT_NODE)
    melodies = [*MAIN_MELODIES, *active_element_melodies]

    targets = [m for m in melodies if data.get_melody_target(m) > 0]
    if not targets:
        logger.debug("未设置音符数量目标，跳过音符读取")
        return {}

    counts: dict[str, int] = {}
    try:
        if not _open_melody_detail(context):
            return {}
        image = context.tasker.controller.post_screencap().wait().get()
        for melody in targets:
            counts[melody] = _read_melody_count(context, image, melody)
            if counts[melody] == -1:
                logger.error(f"读取音符{melody}的数量时出现问题，为保证爬塔质量，将结束任务")
                context.tasker.post_stop()
                return {}
    except Exception as exc:
        logger.error(f"读取音符数量时出现程序异常：{exc}，为保证爬塔质量，将结束任务")
        context.tasker.post_stop()
    finally:
        _run(context, CLOSE_BAG_NODE)

    if counts:
        logger.debug(f"读取到音符数量：{counts}")
    else:
        logger.warning("未能读取到任何音符数量")
    return counts


def _get_element_melodies(context: Context, node: str) -> list[str]:
    """获取属性塔节点的属性音符名称，该节点只记录塔里会有什么属性音符，不记录用户设置的音符数量。"""
    node_data = context.get_node_data(node) or {}
    element_melodies = node_data.get("attach", {}).get("active_melodies", [])
    if not element_melodies:
        logger.error(f"属性塔节点未配置属性音符，请检查设置是否正确。如你不是开发者，请联系开发人员")
        context.tasker.post_stop()
    return element_melodies


def _open_melody_detail(context: Context) -> bool:
    """依次打开 背包 -> 秘纹技能，任一步失败即放弃。"""
    if not _run(context, OPEN_BAG_NODE):
        logger.error("打开音符说明界面失败")
        return False
    return True


def _read_melody_count(context: Context, image: np.ndarray, melody: str) -> int:
    """读取指定音符数量，数量位置通过pipeline的节点联动获取"""
    pipeline_override = _make_pipeline_override(MELODY_NODE, melody)
    template_results = _recognize(context, MELODY_NODE, image, pipeline_override)
    if not template_results:
        logger.error(f"未识别到音符{melody}的位置，请检查属性塔设置是否选择正确")
        return -1
    ocr_results = _recognize(context, COUNT_NODE, image)
    if not ocr_results:
        logger.error(f"未识别到音符{melody}的数量")
        return -1
    logger.debug(f"识别到音符{melody}的数量为：{ocr_results}")
    return int(ocr_results[0].text)


def _make_pipeline_override(node: str, melody: str) -> dict:
    """根据音符名称生成 pipeline override。"""
    return {node: {"recognition": {"param": {"template": [f"ClimbTower_agent/melodies/{melody}.png"]}}}}


def _recognize(context: Context, node: str, image: np.ndarray, pipeline_override: dict | None = None) -> list:
    if pipeline_override is None:
        pipeline_override = {}
    detail = context.run_recognition(node, image, pipeline_override)
    return list(detail.filtered_results) if detail and detail.hit else []


def _run(context: Context, node: str) -> bool:
    """执行一个 pipeline 节点。"""
    result = context.run_task(node)
    if not result or not result.status.succeeded:
        return False
    return True
