"""灾变防线走位的停止、输入失败及连续执行回归测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "catastrophe_defense_move",
    ROOT / "agent/custom/action/catastrophe_defense_move.py",
)
move = importlib.util.module_from_spec(spec)
# 单测不启动原生 AgentServer 的消息线程；动作逻辑仍使用真实 Python 绑定类型。
with patch("maa.agent.agent_server.AgentServer.custom_action", side_effect=lambda _name: lambda cls: cls):
    spec.loader.exec_module(move)


class CatastropheDefenseMovementTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        self.controller.post_key_down.return_value.wait.return_value.succeeded = True
        self.tasker = SimpleNamespace(controller=self.controller, stopping=False)
        self.context = SimpleNamespace(tasker=self.tasker, get_node_data=Mock(return_value={}))
        self.argv = SimpleNamespace(node_name="灾变防线_走位")
        move._reset_state()

    def assert_keys_released(self):
        # 每次走位前与 finally 中各松两遍；失败时仍须完成后一次清理。
        self.assertEqual(
            self.controller.post_key_up.call_args_list,
            [call(key) for key in move.ALL_KEYS] * 4,
        )

    def test_partial_key_down_failure_releases_all_keys(self):
        self.controller.post_key_down.return_value.wait.side_effect = [
            SimpleNamespace(succeeded=True), SimpleNamespace(succeeded=False),
        ]
        with self.assertLogs(move.logger, level="ERROR"):
            result = move._hold(self.context, (move.KEY_W, move.KEY_A), 12)
        self.assertEqual(result, "failed")
        self.assert_keys_released()

    def test_key_down_exception_releases_partially_pressed_keys(self):
        self.controller.post_key_down.side_effect = [
            self.controller.post_key_down.return_value, RuntimeError("controller unavailable"),
        ]
        with self.assertLogs(move.logger, level="ERROR"):
            result = move._hold(self.context, (move.KEY_W, move.KEY_A), 12)
        self.assertEqual(result, "failed")
        self.assert_keys_released()

    def test_wait_exception_releases_keys(self):
        self.controller.post_key_down.return_value.wait.side_effect = RuntimeError("wait failed")
        with self.assertLogs(move.logger, level="ERROR"):
            result = move._hold(self.context, (move.KEY_W, move.KEY_A), 12)
        self.assertEqual(result, "failed")
        self.assert_keys_released()

    def test_stop_before_pressing_sends_no_key_down(self):
        self.tasker.stopping = True
        self.assertEqual(move._hold(self.context, (move.KEY_W, move.KEY_A), 12), "stopped")
        self.controller.post_key_down.assert_not_called()
        self.assert_keys_released()

    def test_stop_between_keys_releases_first_key(self):
        def press(_key):
            self.tasker.stopping = True
            return Mock(wait=Mock(return_value=SimpleNamespace(succeeded=True)))

        self.controller.post_key_down.side_effect = press
        self.assertEqual(move._hold(self.context, (move.KEY_W, move.KEY_A), 12), "stopped")
        self.controller.post_key_down.assert_called_once_with(move.KEY_W)
        self.assert_keys_released()

    def test_stop_during_movement_releases_keys(self):
        def screenshot(_context):
            self.tasker.stopping = True
            return np.full((8, 8, 3), 128, dtype=np.uint8)

        with patch.object(move, "_grab", side_effect=screenshot):
            self.assertEqual(move._hold(self.context, (move.KEY_W, move.KEY_A), 12), "stopped")
        self.assert_keys_released()

    def test_loading_black_screen_releases_keys(self):
        with patch.object(move, "_grab", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
            self.assertEqual(move._hold(self.context, (move.KEY_S, move.KEY_D), 5), "entered")
        self.assert_keys_released()

    def test_missing_screenshot_fails_and_releases_keys(self):
        with patch.object(move, "_grab", return_value=None), self.assertLogs(move.logger, level="ERROR"):
            self.assertEqual(move._hold(self.context, (move.KEY_W, move.KEY_A), 12), "failed")
        self.assert_keys_released()

    def test_hold_timeout_releases_keys(self):
        self.assertEqual(move._hold(self.context, (move.KEY_W, move.KEY_A), 0), "timeout")
        self.assert_keys_released()

    def test_release_failure_does_not_skip_other_keys(self):
        self.controller.post_key_up.side_effect = [RuntimeError("key up failed")] + [Mock()] * 7
        move._release_all(self.context)
        self.assertEqual(
            self.controller.post_key_up.call_args_list,
            [call(key) for key in move.ALL_KEYS] * 2,
        )

    def test_failed_hold_fails_action_without_marking_portal_entered(self):
        move._state.update(started_at=1, phase=move.PHASE_LEFT)
        with patch.object(move, "_grab", return_value=None), patch.object(move, "_hold", return_value="failed"):
            self.assertFalse(move.CatastropheDefenseMove().run(self.context, self.argv))
        self.assertFalse(move._state["entered"])

    def test_reset_allows_second_run_in_same_context(self):
        action = move.CatastropheDefenseMove()
        self.assertTrue(action.run(self.context, self.argv))
        move._state.update(entered=True, phase=move.PHASE_DONE, attempt=8, last_gray=np.zeros((2, 2)))
        self.assertTrue(move.CatastropheDefenseReset().run(self.context, self.argv))
        self.assertFalse(move._state["entered"])
        self.assertEqual(move._state["attempt"], 0)
        self.assertIsNone(move._state["last_gray"])
        self.assertIsNone(move._state["started_at"])
        self.assertTrue(action.run(self.context, self.argv))
        self.assertEqual(move._state["phase"], move.PHASE_WAIT)


if __name__ == "__main__":
    unittest.main()
