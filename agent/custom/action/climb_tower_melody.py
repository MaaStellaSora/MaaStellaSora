import re
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from custom.reco.climb_tower_potential.state import State
from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_melody")


@AgentServer.custom_action("read_melody_counts")
class ReadMelodyCounts(CustomAction):
    """读取当前各属性音符数量（进商店前：背包-秘纹技能-技能音符说明-左侧属性音符列表）。

    坐标为 1280x720 估算值，需按实机校准。
    """
    BACKPACK_BTN = (140, 37)        # 商店页左上角背包
    SECRET_SKILL_BTN = (110, 213)   # 背包-秘纹技能
    DETAIL_BTN = (1220, 45)         # 右上角"技能音符说明"打开按钮
    LIST_ROI = [300, 205, 320, 445] # 左侧"属性音符"列表区域
    CLOSE_BTN = (1026, 134)         # 说明框右上角关闭 X
    BACK_BTN = (72, 45)             # 背包返回（左上角大返回按钮）

    VALID_SONGS = [
        "水之音", "火之音", "地之音", "土之音", "风之音", "光之音", "暗之音",
        "专注之音", "技巧之音", "绝招之音", "强攻之音", "幸运之音", "暴发之音", "体力之音",
    ]

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            # 只在"遇到了星塔商店，去商店购物吧！"页才读，避免在强化选卡等页误触发乱点
            image0 = context.tasker.controller.post_screencap().wait().get()
            d0 = context.run_recognition("星塔_节点_读协奏音符_agent", image0, {
                "星塔_节点_读协奏音符_agent": {"recognition": {"param": {"expected": ["星塔商店"], "roi": [0, 0, 1280, 720]}}}
            })
            if not (d0 and d0.hit):
                logger.debug("[音符数量] 当前不是星塔商店选择页，跳过读取")
                return True
            self._click(context, self.BACKPACK_BTN)
            time.sleep(1.2)
            self._click(context, self.SECRET_SKILL_BTN)
            time.sleep(1.2)
            self._click(context, self.DETAIL_BTN)
            time.sleep(1.2)
            counts = self._read_list(context)
            State.melody_counts = counts
            logger.info(f"[音符数量] 读取到 {counts}")
            self._log_summary(context, counts)
            self._click(context, self.CLOSE_BTN)
            time.sleep(0.6)
            self._click(context, self.BACK_BTN)
            time.sleep(0.6)
        except Exception as exc:
            logger.warning(f"[音符数量] 读取异常：{exc}")
        return True

    def _click(self, context: Context, xy):
        context.tasker.controller.post_click(int(xy[0]), int(xy[1])).wait()

    def _read_list(self, context: Context) -> dict:
        counts = {}
        node = "星塔_记录_识别数字_agent"
        for i in range(6):
            image = context.tasker.controller.post_screencap().wait().get()
            d = context.run_recognition(node, image, {
                node: {"recognition": {"param": {"expected": [".+"], "roi": self.LIST_ROI}}}
            })
            items = list(d.filtered_results) if d and d.hit else []
            songs = []
            nums = []
            for r in items:
                t = (r.text or "").strip()
                bx = r.box[0]
                if re.fullmatch(r"\d+个?", t) and 430 <= bx <= 500:
                    nums.append((int(re.sub(r"\D", "", t)), r.box))
                elif not re.fullmatch(r"\d+个?", t):
                    songs.append((t, r.box))
            for name, nbox in songs:
                if name not in self.VALID_SONGS:
                    continue
                ny = nbox[1] + nbox[3] / 2
                best = None
                for num, cbox in nums:
                    cy = cbox[1] + cbox[3] / 2
                    if abs(ny - cy) <= 30 and (best is None or abs(ny - cy) < abs(ny - best[1])):
                        best = (num, cy)
                if best:
                    counts[name] = best[0]
            if i < 5:
                # 在"属性音符"列表内向上滑动（从下往上），露出下方更多音符（共13种，一屏显示不全）
                context.tasker.controller.post_swipe(450, 520, 450, 240, 400).wait()
                time.sleep(0.6)
        return counts

    def _log_summary(self, context: Context, counts: dict) -> None:
        """读取结束后输出总结：对比"音符数量目标"，达到设定时额外提示。"""
        try:
            node_data = context.get_node_data("星塔_节点_商店_购物_agent")
            attach = node_data.get("attach", {}) if node_data else {}
            mapping = [
                ("aqua", "水之音"), ("ignis", "火之音"), ("terra", "地之音"),
                ("ventus", "风之音"), ("lux", "光之音"), ("umbra", "暗之音"),
                ("focus", "专注之音"), ("skill", "技巧之音"), ("ultimate", "绝招之音"),
                ("pummel", "强攻之音"), ("luck", "幸运之音"), ("burst", "暴发之音"),
                ("stamina", "体力之音"),
            ]
            targets = {}
            for suffix, cn in mapping:
                try:
                    v = int(attach.get(f"melody_target_{suffix}", 0))
                except (TypeError, ValueError):
                    v = 0
                if v and v > 0:
                    targets[cn] = v
            if not targets:
                logger.info("[音符数量] 未设置音符数量目标，无需判断")
                return
            summary = "，".join(f"{cn}={counts.get(cn, 0)}" for cn in sorted(targets.keys()))
            logger.info(f"[音符数量] 目标对比：{summary}")
            if all(counts.get(cn, 0) >= tgt for cn, tgt in targets.items()):
                logger.info("音符数量已符合设定！")
            else:
                lacks = "，".join(
                    f"{cn}差{max(0, tgt - counts.get(cn, 0))}"
                    for cn, tgt in targets.items() if counts.get(cn, 0) < tgt
                )
                logger.info(f"[音符数量] 尚未符合设定（{lacks}）")
        except Exception as exc:
            logger.warning(f"[音符数量] 总结失败：{exc}")
