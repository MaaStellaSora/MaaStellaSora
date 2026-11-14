"""Concurrent collectors and uploaders share real files without losing state."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "agent/custom/action"))
from climb_tower_neural import sample_upload, sampling_data


class TowerNeuralSharedFilesTests(unittest.TestCase):
    def setUp(self):
        temporary = TowerTempDirectory()
        self.addCleanup(temporary.cleanup)
        self.app_dir = Path(temporary.name)

    def uploaders(self):
        directory = self.app_dir / "debug/tower_sampling"
        for folder in ("queue", "retries"):
            (directory / folder).mkdir(parents=True)
        settings = dict(endpoint="https://offline.invalid", access_token="",
                        policy_version="test", recipient="test", purpose="test")
        (self.app_dir / "upload_settings.json").write_text(json.dumps(settings), encoding="utf-8")
        pair = [sample_upload.SampleUpload(self.app_dir, self.app_dir, directory, Mock()) for _ in range(2)]
        for uploader in pair:
            uploader.set_consent(True)
            uploader.wake.clear()
            uploader.send = Mock(side_effect=OSError("offline"))
        return pair

    def ready(self, uploader, run_id, age=1):
        marker = uploader.directory / "queue" / f"{run_id}.ready.json"
        marker.write_text(json.dumps(dict(run_id=run_id,
            research_upload=uploader.consent_for_run(run_id))), encoding="utf-8")
        os.utime(marker, (age, age))
        return marker

    def test_two_collectors_initializing_one_directory_keep_one_identity(self):
        collectors = [sampling_data.SamplingCollector(self.app_dir, Mock()) for _ in range(2)]
        identity = self.app_dir / "config/neural_network/collector_identity.json"
        first_written, second_written, first_finished = (threading.Event() for _ in range(3))
        write_text = Path.write_text

        def staged_write(path, *args, **kwargs):
            result = write_text(path, *args, **kwargs)
            if path.parent == identity.parent and path.suffix == ".tmp":
                if not first_written.is_set():
                    first_written.set()
                    self.assertTrue(second_written.wait(5))
                else:
                    second_written.set()
                    self.assertTrue(first_finished.wait(5))
            return result

        # Both complete their candidate before either publishes; the first
        # collector then finishes before the second attempts publication.
        with patch.object(Path, "write_text", staged_write), ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(collectors[0].loaded)
            self.assertTrue(first_written.wait(5))
            second = workers.submit(collectors[1].loaded)
            try:
                first.result(timeout=5)
            finally:
                first_finished.set()
            second.result(timeout=5)
        participant = json.loads(identity.read_text(encoding="utf-8"))["participant_id"]
        for collector in collectors:
            self.assertTrue(collector.enabled)
            self.assertEqual(collector.participant_id, participant)
            collector.logger.error.assert_not_called()

    def test_two_uploaders_can_publish_retry_metadata_concurrently(self):
        first, second = self.uploaders()
        marker = self.ready(first, "parallel")
        retry = first._retry_path(marker)
        entered, release = threading.Event(), threading.Event()
        replace = Path.replace

        def staged_replace(path, target):
            if target == retry and not entered.is_set():
                entered.set()
                self.assertTrue(release.wait(5))
            return replace(path, target)

        # Hold the first replacement until the competing uploader has written
        # and replaced its own temporary file, without using a network endpoint.
        with (
            patch.object(Path, "replace", staged_replace),
            patch.object(sample_upload.time, "time", return_value=1000),
            patch.object(sample_upload.random, "uniform", side_effect=(.25, .5)),
            ThreadPoolExecutor(max_workers=1) as workers,
        ):
            pending = workers.submit(first.flush)
            self.assertTrue(entered.wait(5))
            try:
                second.flush()
            finally:
                release.set()
            pending.result(timeout=5)
        first.send.assert_called_once()
        second.send.assert_called_once()
        self.assertTrue(marker.exists())
        self.assertEqual(json.loads(retry.read_text(encoding="utf-8")),
                         dict(attempts=1, next_attempt=1003.25))
        self.assertEqual(list(retry.parent.glob("*.tmp")), [])

    def test_corrupt_retry_metadata_does_not_block_older_queued_runs(self):
        uploader, _ = self.uploaders()
        older = self.ready(uploader, "older", age=1)
        newer = self.ready(uploader, "newer", age=2)
        uploader._retry_path(newer).write_text('{"attempts":', encoding="utf-8")
        uploader.flush()
        self.assertEqual([call.args[0] for call in uploader.send.call_args_list], [newer, older])
        for marker in (newer, older):
            self.assertTrue(marker.exists())
            self.assertEqual(json.loads(uploader._retry_path(marker).read_text(encoding="utf-8"))["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
