"""灾变防线控制器超时、停止和截图错误的回归覆盖。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "agent"))
from utils import cataclysm_survivor as io  # noqa: E402


class ControllerWaitTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        self.tasker = SimpleNamespace(stopping=False, controller=self.controller)
        self.context = SimpleNamespace(tasker=self.tasker)
        self.job = Mock(done=True, succeeded=True)
        self.controller.post_screencap.return_value = self.job

    def test_pending_job_times_out_without_calling_blocking_wait(self):
        self.job.done = False
        with patch.object(io.time, "monotonic", side_effect=[0, 0, 3]), patch.object(io.time, "sleep"):
            self.assertFalse(io.wait_job(self.context, self.job))
        self.job.wait.assert_not_called()

    def test_stop_interrupts_pending_job(self):
        self.job.done = False
        with patch.object(io.time, "sleep", side_effect=lambda _: setattr(self.tasker, "stopping", True)):
            self.assertFalse(io.wait_job(self.context, self.job))
        self.job.wait.assert_not_called()

    def test_cleanup_can_finish_after_task_stopped(self):
        self.tasker.stopping = True
        self.assertFalse(io.wait_job(self.context, self.job))
        self.assertTrue(io.wait_job(self.context, self.job, cleanup=True))

    def test_failed_job_is_not_success(self):
        self.job.succeeded = False
        self.assertFalse(io.wait_job(self.context, self.job))

    def test_status_query_exception_is_handled(self):
        type(self.job).done = PropertyMock(side_effect=RuntimeError("controller unavailable"))
        self.assertFalse(io.wait_job(self.context, self.job))

    def test_zero_timeout_still_accepts_already_completed_job(self):
        self.assertTrue(io.wait_job(self.context, self.job, timeout=0))
        self.job.done = False
        self.assertFalse(io.wait_job(self.context, self.job, timeout=0))

    def test_screenshot_uses_only_successful_completed_job(self):
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        self.job.get.return_value = image
        self.assertIs(io.grab(self.context), image)
        self.job.wait.assert_not_called()

    def test_pending_screenshot_times_out_and_does_not_read_stale_image(self):
        self.job.done = False
        with patch.object(io.time, "monotonic", side_effect=[0, 0, 3]), patch.object(io.time, "sleep"):
            self.assertIsNone(io.grab(self.context))
        self.job.get.assert_not_called()

    def test_empty_and_missing_screenshots_are_failures(self):
        for image in (None, np.empty((0, 0, 3))):
            with self.subTest(image=image):
                self.job.get.return_value = image
                self.assertIsNone(io.grab(self.context))

    def test_stopped_task_does_not_submit_new_screenshot(self):
        self.tasker.stopping = True
        self.assertIsNone(io.grab(self.context))
        self.controller.post_screencap.assert_not_called()

    def test_screenshot_submission_exception_is_handled(self):
        self.controller.post_screencap.side_effect = RuntimeError("disconnected")
        self.assertIsNone(io.grab(self.context))


if __name__ == "__main__":
    unittest.main()
