"""灾变防线走位的停止、输入失败及连续执行回归测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, call, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "cataclysm_survivor_move",
    ROOT / "agent/custom/action/cataclysm_survivor_move.py",
)
move = importlib.util.module_from_spec(spec)
# 单测不启动原生 AgentServer 的消息线程；动作逻辑仍使用真实 Python 绑定类型。
with patch("maa.agent.agent_server.AgentServer.custom_action", side_effect=lambda _name: lambda cls: cls):
    spec.loader.exec_module(move)


class CataclysmSurvivorMovementTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        for post in (self.controller.post_key_down, self.controller.post_key_up):
            post.return_value.done = True
            post.return_value.succeeded = True
        self.tasker = SimpleNamespace(controller=self.controller, stopping=False)
        self.job = SimpleNamespace(job_id=1, done=False)
        self.context = SimpleNamespace(
            tasker=self.tasker, get_node_data=Mock(return_value={}), get_task_job=lambda: self.job,
        )
        self.argv = SimpleNamespace(node_name="灾变防线_走位")
        move._reset_state()
        move._owner_job = self.job

    def assert_keys_released(self):
        # 每次走位前与 finally 中各松两遍；失败时仍须完成后一次清理。
        self.assertEqual(
            self.controller.post_key_up.call_args_list,
            [call(key) for key in move.ALL_KEYS] * 4,
        )

    def test_partial_key_down_failure_releases_all_keys(self):
        self.controller.post_key_down.side_effect = [
            SimpleNamespace(done=True, succeeded=True), SimpleNamespace(done=True, succeeded=False),
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

    def test_status_query_exception_releases_keys(self):
        job = self.controller.post_key_down.return_value
        type(job).done = PropertyMock(side_effect=RuntimeError("status unavailable"))
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
            return SimpleNamespace(done=True, succeeded=True)

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
        self.controller.post_key_up.side_effect = [RuntimeError("key up failed")] + [
            SimpleNamespace(done=True, succeeded=True),
        ] * 7
        self.assertTrue(move._release_all(self.context))
        self.assertEqual(
            self.controller.post_key_up.call_args_list,
            [call(key) for key in move.ALL_KEYS] * 2,
        )

    def test_failed_hold_fails_action_without_marking_portal_entered(self):
        move._state.update(started_at=1, phase=move.PHASE_LEFT)
        with patch.object(move, "_grab", return_value=None), patch.object(move, "_hold", return_value="failed"):
            self.assertFalse(move.CataclysmSurvivorMove().run(self.context, self.argv))
        self.assertFalse(move._state["entered"])

    def test_reset_allows_second_run_in_same_context(self):
        action = move.CataclysmSurvivorMove()
        self.assertTrue(action.run(self.context, self.argv))
        move._state.update(entered=True, phase=move.PHASE_DONE, attempt=8, last_gray=np.zeros((2, 2)))
        self.assertTrue(move.CataclysmSurvivorReset().run(self.context, self.argv))
        self.assertFalse(move._state["entered"])
        self.assertEqual(move._state["attempt"], 0)
        self.assertIsNone(move._state["last_gray"])
        self.assertIsNone(move._state["started_at"])
        self.assertTrue(action.run(self.context, self.argv))
        self.assertEqual(move._state["phase"], move.PHASE_WAIT)

    def test_persistent_release_failure_prevents_pressing_any_key(self):
        self.controller.post_key_up.return_value.succeeded = False
        with self.assertLogs(move.logger, level="ERROR"):
            self.assertEqual(move._hold(self.context, (move.KEY_W, move.KEY_A), 12), "failed")
        self.controller.post_key_down.assert_not_called()

    def test_final_cleanup_failure_overrides_successful_portal_detection(self):
        with patch.object(move, "_release_all", side_effect=[True, False]), patch.object(
            move, "_grab", return_value=np.zeros((8, 8, 3), dtype=np.uint8),
        ):
            self.assertEqual(move._hold(self.context, (move.KEY_W, move.KEY_A), 12), "failed")

    def test_release_failures_and_exceptions_are_reported(self):
        for result in (RuntimeError("disconnected"), SimpleNamespace(done=True, succeeded=False)):
            with self.subTest(result=result):
                self.controller.post_key_up.side_effect = [result] * 8
                with self.assertLogs(move.logger, level="ERROR"):
                    self.assertFalse(move._release_all(self.context))

    def test_settle_requires_valid_matching_frames(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        invalid = (None, np.empty((0, 0, 3)), np.full((64, 64, 3), 128, dtype=np.uint8))
        for after in invalid:
            with self.subTest(after=after), patch.object(move, "_drive", return_value=True), patch.object(
                move, "_grab", side_effect=[frame, after],
            ):
                self.assertEqual(move._settle(self.context, 0.05, 16), (False, -1.0))

    def test_settle_rejects_motion_and_accepts_matching_still_frames(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        for after, expected in ((frame, True), (np.full_like(frame, 255), False)):
            with self.subTest(expected=expected), patch.object(move, "_drive", return_value=True), patch.object(
                move, "_grab", side_effect=[frame, after],
            ):
                self.assertEqual(move._settle(self.context, 0.05, 16)[0], expected)

    def test_failed_settle_never_advances_either_direction_phase(self):
        for phase in (move.PHASE_LEFT, move.PHASE_PORTAL):
            for ratio in (-1.0, 0.2):
                with self.subTest(phase=phase, ratio=ratio):
                    move._state.update(started_at=1, phase=phase)
                    with patch.object(move, "_grab", return_value=None), patch.object(
                        move, "_hold", return_value="timeout",
                    ), patch.object(move, "_settle", return_value=(False, ratio)), self.assertLogs(
                        move.logger, level="ERROR",
                    ):
                        self.assertFalse(move.CataclysmSurvivorMove().run(self.context, self.argv))
                    self.assertEqual(move._state["phase"], phase)
                    self.assertFalse(move._state["entered"])

    def test_gap_resets_continuous_stillness_before_counting_again(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        move._state.update(last_gray=move._to_gray(frame), last_at=1, still_since=1)
        self.assertFalse(move._update_stillness(frame, 10, 8, 0.05, 16))
        self.assertIsNone(move._state["still_since"])
        for now in (11, 13, 15, 17):
            self.assertFalse(move._update_stillness(frame, now, 8, 0.05, 16))
        self.assertTrue(move._update_stillness(frame, 19, 8, 0.05, 16))

    def test_missing_frame_and_shape_change_break_stillness(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        for image in (None, np.full((64, 64, 3), 128, dtype=np.uint8)):
            move._state.update(last_gray=move._to_gray(frame), last_at=1, still_since=1)
            self.assertFalse(move._update_stillness(image, 2, 8, 0.05, 16))
            self.assertIsNone(move._state["still_since"])

    def test_other_active_run_cannot_reset_or_move_owned_state(self):
        move._state.update(phase=move.PHASE_PORTAL, attempt=3)
        other = SimpleNamespace(**{**vars(self.context), "get_task_job": lambda: SimpleNamespace(job_id=2)})
        for action in (move.CataclysmSurvivorReset(), move.CataclysmSurvivorMove()):
            with self.assertLogs(move.logger, level="ERROR"):
                self.assertFalse(action.run(other, self.argv))
        self.assertEqual(move._state["phase"], move.PHASE_PORTAL)
        self.assertEqual(move._state["attempt"], 3)
        self.controller.post_key_up.assert_not_called()

    def test_completed_run_allows_next_task_to_claim_state(self):
        self.job.done = True
        next_job = SimpleNamespace(job_id=2, done=False)
        self.context.get_task_job = lambda: next_job
        move._state.update(phase=move.PHASE_DONE, entered=True)
        self.assertTrue(move.CataclysmSurvivorReset().run(self.context, self.argv))
        self.assertEqual(move._owner_job.job_id, 2)
        self.assertFalse(move._state["entered"])

    def test_reentrant_callbacks_fail_without_deadlocking_or_releasing_keys(self):
        with move._action_lock:
            for action in (move.CataclysmSurvivorReset(), move.CataclysmSurvivorMove()):
                with self.assertLogs(move.logger, level="ERROR"):
                    self.assertFalse(action.run(self.context, self.argv))
        self.controller.post_key_up.assert_not_called()


if __name__ == "__main__":
    unittest.main()
