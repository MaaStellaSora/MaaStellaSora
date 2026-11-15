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

    def test_two_collectors_use_temporary_sessions_without_personal_identity(self):
        collectors = [sampling_data.SamplingCollector(self.app_dir, Mock()) for _ in range(2)]
        identity = self.app_dir / "config/neural_network/collector_identity.json"
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(lambda collector: collector.loaded(), collectors))
        self.assertFalse(identity.exists())
        self.assertEqual(len({collector.session_id for collector in collectors}), 2)
        for collector in collectors:
            self.assertTrue(collector.enabled)
            self.assertFalse(hasattr(collector, "participant_id"))
            collector.begin_task("cn", "test")
            session = collector.directory / "sessions" / f"{collector.session_id}.jsonl"
            rows = [json.loads(line) for line in session.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all("participant_id" not in row for row in rows))
            collector.logger.error.assert_not_called()

    def test_repeated_event_menu_writes_only_changed_semantics(self):
        collector = sampling_data.SamplingCollector(self.app_dir, Mock())
        collector.begin_task("cn", "test")
        collector.start_intent("start")
        collector.confirm_start("entered")
        choices = [dict(text="take", consequence="potential reward", box=[1, 2, 3, 4])]
        collector.event_menu(choices)
        first_event = collector.event["event_id"]
        collector.event_menu(choices)
        self.assertEqual(collector.event["event_id"], first_event)
        collector.event_menu([dict(text="leave", consequence="no reward", box=[1, 2, 3, 4])])
        self.assertNotEqual(collector.event["event_id"], first_event)
        path = collector.directory / "runs" / f"{collector.run['run_id']}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        menus = [row for row in rows if row["event_type"].startswith("event_menu_")]
        self.assertEqual([row["event_type"] for row in menus], ["event_menu_observed"] * 2)
        self.assertEqual([row["choices"][0]["text"] for row in menus], ["take", "leave"])
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
