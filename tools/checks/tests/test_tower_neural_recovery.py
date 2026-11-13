"""Keep a committed run intact when archive recovery overlaps or resumes."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "agent/custom/action"))
from climb_tower_neural import research_store, sampling_data
from climb_tower_neural.research_data import export_payload


class TowerNeuralRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.app_dir = Path(temporary.name)
        self.first = sampling_data.SamplingCollector(self.app_dir, Mock())
        self.second = sampling_data.SamplingCollector(self.app_dir, Mock())
        self.first.begin_task("cn", "test")
        self.run_id = self.first.start_intent("start")
        self.first.confirm_start("entered")
        self.wallet_id = self.first.emit("wallet_observed", coin=410, coin_complete=True)
        self.directory = self.first.directory
        self.marker = self.directory / "queue" / f"{self.run_id}.open.json"
        self.run_path = self.directory / "runs" / f"{self.run_id}.jsonl"
        self.database = self.first.research_store.database_path
        (self.app_dir / "recycled").mkdir()
        recycling = patch.object(research_store, "recycle_file", side_effect=self.recycle)
        recycling.start()
        self.addCleanup(recycling.stop)

    def recycle(self, path):
        # Real file moves keep temporary fixtures out of the Windows Recycle Bin.
        path = Path(path)
        if path.exists():
            path.rename(self.app_dir / "recycled" / path.name)

    def finish_without_cleanup(self):
        # Keep the journals as if the process exited immediately after commit.
        with patch.object(research_store, "recycle_file"):
            self.first.finish_run("settlement_saved", completion_observed=True)
        return export_payload(self.database, self.run_id)

    def assert_archive_unchanged(self, expected):
        actual = export_payload(self.database, self.run_id)
        self.assertEqual(actual["tables"]["runs"][0]["lifecycle_status"], "completed")
        self.assertEqual(actual, expected)
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.run_path.exists())
        self.first.logger.error.assert_not_called()
        self.second.logger.error.assert_not_called()

    def test_two_collectors_recovering_one_run_preserve_all_evidence(self):
        expected = self.finish_without_cleanup()
        entered = False

        def process_alive(pid):
            nonlocal entered
            if not entered:
                entered = True
                # The first collector has read the marker. Let its competitor
                # finish cleanup before it can examine the journal.
                self.second.recover_interrupted()
            return False

        with patch.object(sampling_data, "process_alive", side_effect=process_alive):
            self.first.recover_interrupted()
        self.assertTrue(entered)
        self.assert_archive_unchanged(expected)

    def test_only_marker_remaining_does_not_replace_final_archive_with_empty_end(self):
        expected = self.finish_without_cleanup()
        self.recycle(self.run_path)
        self.recycle(self.directory / "sessions" / f"{self.first.session_id}.jsonl")
        with patch.object(sampling_data, "process_alive", return_value=False):
            self.second.recover_interrupted()
        self.assert_archive_unchanged(expected)

    def test_next_task_retries_its_own_closed_run_while_process_is_alive(self):
        with patch.object(self.first.research_store, "checkpoint", side_effect=sqlite3.OperationalError("busy")):
            self.first.finish_run("settlement_saved", completion_observed=True)
        self.assertEqual(self.first.research_store.run_status(self.run_id), "open")
        self.second.recover_interrupted()
        self.assertTrue(self.marker.exists())
        self.first.begin_task("cn", "next")
        tables = export_payload(self.database, self.run_id)["tables"]
        self.assertEqual(tables["runs"][0]["lifecycle_status"], "completed")
        self.assertIn(self.wallet_id, {row["record_id"] for row in tables["records"]})
        self.assertFalse(self.marker.exists())

    def test_truncated_session_tail_is_flagged_without_changing_run_evidence(self):
        session = self.directory / "sessions" / f"{self.first.session_id}.jsonl"
        data = session.read_bytes() + b'{"unfinished":"private-tail'
        session.write_bytes(data)
        payload = self.finish_without_cleanup()
        tables = payload["tables"]
        issue, = (row for row in tables["quality_issues"] if row["code"] == "truncated_session_tail")
        self.assertIsNone(issue["record_id"])
        self.assertEqual(issue["field_path"], "session.journal")
        self.assertEqual(tables["runs"][0]["lifecycle_status"], "completed")
        self.assertIn(self.wallet_id, {row["record_id"] for row in tables["records"]})
        self.assertNotIn("private-tail", json.dumps(payload))
        self.assertEqual(session.read_bytes(), data)
        session.write_bytes(data + b'\n{"event_type": "task_end"}\n')
        with self.assertRaises(json.JSONDecodeError):
            self.first.research_store.checkpoint(self.run_path)


if __name__ == "__main__":
    unittest.main()
