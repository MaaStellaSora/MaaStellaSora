"""邀约旅人名字匹配的回归测试（韩文名字与其他语言名字）。"""

import difflib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("invite", ROOT / "agent/custom/action/invite.py")
invite = importlib.util.module_from_spec(spec)
# 单测不启动原生 AgentServer 的消息线程；动作逻辑仍使用真实 Python 绑定类型。
with patch("maa.agent.agent_server.AgentServer.custom_action", side_effect=lambda _name: lambda cls: cls):
    spec.loader.exec_module(invite)
InviteAuto = invite.InviteAuto

# 韩服邀约列表中的旅人名字
KOREAN_NAMES = [
    "윌로(여름)", "토나", "치토세", "프리지아", "아야메", "시먀오", "테레사", "샤통", "라루(이브)",
    "후유카", "치시아", "코하쿠", "플로라", "카시미라", "엘레노어", "레이스", "나츠카", "나노하",
    "세이나", "카나체", "안즈", "스파클", "크루니스(새해)", "그레이", "나즈나", "레이센", "호타루",
    "시아", "미네르바", "라루", "징린", "틸리아", "카린", "오토하", "피렌", "미스티", "코제트",
    "캐러멜", "크루니스", "엘리", "세이나(기원)",
]


def ocr(text, x, y, h=24, score=0.99):
    return SimpleNamespace(text=text, score=score, box=[x, y, 60, h])


class InviteMatchingTests(unittest.TestCase):
    def click_trekker(self, target, ocr_results):
        action = InviteAuto.__new__(InviteAuto)
        action.logger = Mock()
        controller = Mock()
        context = SimpleNamespace(
            tasker=SimpleNamespace(stopping=False, controller=controller),
            run_recognition=Mock(return_value=SimpleNamespace(all_results=ocr_results)),
        )
        found = action._click_trekker(context, target)
        return found, controller.post_click

    def grid(self, names):
        """按游戏列表三列排列，每格之间留足间距，避免名字互相合并。"""
        return [ocr(name, 120 + 125 * (i % 3), 265 + 157 * (i // 3)) for i, name in enumerate(names)]

    def test_korean_name_does_not_match_name_it_is_contained_in(self):
        # 2026-11-01 实机：目标 '치시아' 时第一页的 '시아' 被点击
        found, click = self.click_trekker("치시아", self.grid(["세이나(기원)", "토나", "시아"]))
        self.assertFalse(found)
        click.assert_not_called()
        found, _ = self.click_trekker("시아", self.grid(["치시아"]))
        self.assertFalse(found)

    def test_korean_single_letter_misread_still_matches(self):
        # 实机识别结果：'시먀오' 被识别成 '시야오'/'시마오'，'윌로(여름)' 被识别成 '월로(여름)'
        for target, text in [("시먀오", "시야오"), ("시먀오", "시마오"), ("윌로(여름)", "월로(여름)")]:
            with self.subTest(text=text):
                found, click = self.click_trekker(target, self.grid(["플로라", "라루", text]))
                self.assertTrue(found)
                click.assert_called_once_with(370 + 30, 265 + 12)

    def test_korean_names_are_not_confused_with_each_other(self):
        for target in KOREAN_NAMES:
            for other in KOREAN_NAMES:
                with self.subTest(target=target, other=other):
                    found, _ = self.click_trekker(target, self.grid([other]))
                    self.assertEqual(found, target == other)

    def test_non_korean_similarity_is_unchanged(self):
        for result, name in [("Geraxiux", "Geranium"), ("Chitose", "Chitose"), ("希娅", "缇莉雅"), ("拉露", "拉露(圣夜)")]:
            with self.subTest(result=result):
                self.assertEqual(
                    InviteAuto._similarity(result, name),
                    difflib.SequenceMatcher(None, result, name).ratio(),
                )

    def test_decoration_text_is_not_merged_into_korean_name(self):
        merged = InviteAuto._get_refined_merge([
            ocr("치시아", 120, 570), ocr("PIbPD", 116, 597, h=17, score=0.8),
            ocr("월로", 380, 420), ocr("(여름)", 372, 446, h=16),
        ])
        self.assertEqual([m["text"] for m in merged], ["월로(여름)", "치시아"])

    def test_non_korean_name_merge_is_unchanged(self):
        merged = InviteAuto._get_refined_merge([ocr("Flora", 120, 570), ocr("Summer", 116, 597)])
        self.assertEqual([m["text"] for m in merged], ["FloraSummer"])


if __name__ == "__main__":
    unittest.main()
