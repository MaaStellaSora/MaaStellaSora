"""灾变防线：从当前屏幕的纪录卡中选择亮度与饱和度得分最高的一张。

复用 record_position.png 查找徽标，再按相对偏移采样整张卡。不可选的灰卡也有
相同徽标，因此不能直接点击模板匹配排序的第一张。历史样本中，可选卡得分约
215，灰卡最高约 153；保留该评分启发式，不负责下滑搜索或判定全屏均不可选。
"""

from __future__ import annotations

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger as logger_module
from utils.cataclysm_survivor import grab as _grab, wait_job

logger = logger_module.get_logger(__name__)

## 记录卡区域相对「◈ 旅人」徽标的偏移（x, y, w, h）
CARD_OFFSET = (-20, -71, 230, 250)

## 点击位置相对徽标的偏移（取卡片中心）
CLICK_OFFSET = (95, 55)


def _card_score(arr, icon_x: int, icon_y: int) -> float:
    """量一张记录卡「有多亮多彩」；灰卡被蒙灰，分数明显低。"""

    dx, dy, w, h = CARD_OFFSET
    x0 = max(0, icon_x + dx)
    y0 = max(0, icon_y + dy)
    patch = arr[y0:y0 + h, x0:x0 + w]
    if patch.size == 0:
        return -1.0
    p = patch.astype(np.float32)
    mx = p.max(axis=2)
    mn = p.min(axis=2)
    sat = float(((mx - mn) / np.maximum(mx, 1.0)).mean())
    return float(p.mean()) + 100.0 * sat


@AgentServer.custom_action("cataclysm_survivor_pick_record")
class CataclysmSurvivorPickRecord(CustomAction):
    """在「纪录管理」页挑出可选（高亮）的那张记录卡并点击。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        if context.tasker.stopping:
            return False

        image = _grab(context)
        if image is None:
            logger.error("$agent.custom.action.cataclysm_survivor_record.run.screenshot_failed")
            return False

        ## 复用当前节点的模板识别，拿到页面上所有记录卡（每张卡一个小徽标）
        detail = context.run_recognition(argv.node_name, image)
        if detail is None or not detail.hit:
            logger.error(
                "$agent.custom.action.cataclysm_survivor_record.run.record_cards_not_found"
            )
            return False

        candidates = list(detail.filtered_results or detail.all_results or [])
        if not candidates:
            logger.error(
                "$agent.custom.action.cataclysm_survivor_record.run.record_candidates_empty"
            )
            return False

        arr = np.asarray(image)
        scored = []
        for r in candidates:
            x, y, _w, _h = r.box
            scored.append((_card_score(arr, x, y), x, y, r.score))
        scored.sort(key=lambda t: -t[0])

        best_score, best_x, best_y, match_score = scored[0]
        logger.info(
            "$agent.custom.action.cataclysm_survivor_record.run.record_card_scan_summary",
            {
                "value_1": len(scored),
                "value_2": best_score,
                "value_3": match_score,
                "value_4": best_x,
                "value_5": best_y,
            },
        )
        logger.debug("灾变防线：各卡分数 %s", [round(s[0], 1) for s in scored])

        if best_score < 0:
            logger.error(
                "$agent.custom.action.cataclysm_survivor_record.run.record_sampling_failed"
            )
            return False

        click_x = best_x + CLICK_OFFSET[0]
        click_y = best_y + CLICK_OFFSET[1]
        if context.tasker.stopping:
            return False
        if not wait_job(context, context.tasker.controller.post_click(click_x, click_y)):
            logger.error(
                "$agent.custom.action.cataclysm_survivor_record.run.record_card_click_failed"
            )
            return False
        logger.info(
            "$agent.custom.action.cataclysm_survivor_record.run.record_card_selected",
            {
                "value_1": click_x,
                "value_2": click_y,
            },
        )
        return True
