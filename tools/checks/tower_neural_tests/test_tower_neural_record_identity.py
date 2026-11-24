"""Confirmed inventory must reference the selection record that was persisted."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import research_data, research_store, sampling_data
from custom.climb_tower_neural.note_ledger import NoteLedger
from custom.climb_tower_neural.note_runtime import NoteRuntime
from custom.climb_tower_neural.sampling_data import SamplingCollector
from custom.climb_tower_neural.value_model import replace, sim
from test_tower_neural_wallet_state import tracker_at_quiz


class TowerNeuralRecordIdentityTests(unittest.TestCase):
    def test_route_recovery_keeps_the_persisted_deduplicated_floor_reference(self):
        for direct_observation in (False, True):
            with self.subTest(direct_observation=direct_observation), TowerTempDirectory() as directory:
                collector = SamplingCollector(Path(directory), Mock())
                collector.begin_task("cn", "floor-reference-test")
                collector.start_intent("start")
                collector.confirm_start("entered")
                collector.session.begin_run()
                tracker = tracker_at_quiz(collector=collector, session=collector.session)
                notes = NoteRuntime(SimpleNamespace(session=tracker.session, collector=collector,
                    tracker=tracker, logger=Mock()))
                first = notes._read_floor(None, None, "result", observation=dict(
                    floor=3, complete=True, source="floor_hud"))
                repeated = notes._read_floor(None, None, "result", observation=dict(
                    floor=3, complete=True, source="floor_hud"))
                floor_id = first["floor_observation_id"]
                self.assertEqual(repeated["floor_observation_id"], floor_id)
                self.assertEqual(tracker.session.last_confirmed_floor["floor_observation_id"], floor_id)
                notes._read_floor(None, None, "potential", observation=dict(floor=None,
                    complete=False, source="floor_hud", reason="floor_hud_obscured"))
                # A tracker beyond the remaining route must fall back to the real HUD.
                tracker.state = replace(tracker.state, node=len(sim.ROUTE) + 1, floor=20, phase="ADVANCE")
                tracker.route_candidates = ((tracker.state.node, tracker.state.k, tracker.state.A,
                                             tracker.state.cBel, sim.ONE),)
                tracker.observe_route("DRAW", None, observed_floor=repeated if direct_observation else None)
                self.assertEqual(tracker.state.floor, 3)
                run_id = collector.run["run_id"]
                with patch.object(research_store, "recycle_file"):
                    collector.finish_run("test_finished", completion_observed=True)
                journal = collector.directory / "runs" / f"{run_id}.jsonl"
                rows = [json.loads(line) for line in journal.read_text().splitlines()]
                floors = [row for row in rows if row["event_type"] == "note_floor_observed" and row["floor"] == 3]
                self.assertEqual(len(floors), 1)
                self.assertEqual(floors[0]["record_id"], floor_id)
                recovery = next(row for row in rows if row["event_type"] == "runtime_recovery")
                self.assertEqual(recovery["observed_floor_record_id"], floor_id)
                exported = research_data.export_payload(collector.research_store.database_path, run_id)
                records = {row["record_id"]: row for row in exported["tables"]["records"]}
                payload = json.loads(records[recovery["record_id"]]["payload_json"])
                self.assertEqual(payload["observed_floor_record_id"], floor_id)
                self.assertEqual(records[floor_id]["event_type"], "note_floor_observed")

    def test_failed_floor_write_does_not_borrow_an_older_same_floor_record(self):
        with TowerTempDirectory() as directory:
            collector = SamplingCollector(Path(directory), Mock())
            collector.begin_task("cn", "floor-reference-test")
            collector.start_intent("start")
            collector.confirm_start("entered")
            collector.session.begin_run()
            notes = NoteRuntime(SimpleNamespace(session=collector.session, collector=collector,
                                                tracker=None, logger=Mock()))
            first = notes._read_floor(None, None, "field", observation=dict(
                floor=14, complete=True, source="floor_hud"))
            with patch.object(sampling_data, "append_record", side_effect=OSError("journal unavailable")):
                failed = notes._read_floor(None, None, "result", observation=dict(
                    floor=14, complete=True, source="floor_hud"))
            self.assertEqual(collector.session.last_confirmed_floor["floor"], 14)
            self.assertNotIn("floor_observation_id", collector.session.last_confirmed_floor)
            self.assertNotIn("floor_observation_id", failed)
            self.assertEqual(collector.last_observed_floor_context["record_id"], first["floor_observation_id"])
            notes._read_floor(None, None, "event", observation=dict(floor=None,
                complete=False, source="floor_hud", reason="floor_hud_not_displayed"))
            notes.floor = 14
            notes.ledger = NoteLedger()
            notes.ledger.begin_run([], "光土", run_id=collector.run["run_id"])
            notes.before_event(SimpleNamespace(), None, event_id="next-event", note_effect=dict(operation="gain"))
            observed = notes.ledger.pending["metadata"]["floor_observation"]
            self.assertEqual(observed["floor"], 14)
            self.assertEqual(observed["source"], "previous_clear_floor_hud")
            self.assertNotIn("floor_observation_id", observed)

    def test_confirmed_selection_remains_linked_from_inventory_through_database_export(self):
        with TowerTempDirectory() as directory:
            collector = SamplingCollector(Path(directory), Mock())
            collector.begin_task("cn", "record-identity-test")
            collector.start_intent("start")
            collector.confirm_start("entered")
            run_id = collector.run["run_id"]
            collector.reward = dict(closed=False)
            collector.pending_selection = dict(operation_id="take-1", offer_id="offer-1",
                reward_chain_id="reward-1", source_type="battle", selected=dict(
                    slot=0, role=1, potential_id=101, canonical_rarity="rare", core=False,
                    level_before=0, level_after=1, recognition_status="ok"))

            collector.selection_confirmed("reward_page_closed")

            owned = collector.owned_fields()["owned"]
            self.assertEqual((owned[0]["potential_id"], owned[0]["level"]), (101, 1))
            confirmation_id = owned[0]["confirmation_record_id"]
            self.assertIsNone(collector.pending_selection)
            self.assertTrue(collector.reward["closed"])
            # Keep the actual journal for inspection without invoking Windows' Recycle Bin.
            with patch.object(research_store, "recycle_file"):
                collector.finish_run("settlement_saved", completion_observed=True)

            journal = collector.directory / "runs" / f"{run_id}.jsonl"
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            selection = next(row for row in rows if row["event_type"] == "selection_ui_transition_observed")
            end = next(row for row in rows if row["event_type"] == "exploration_end")
            self.assertEqual(selection["record_id"], confirmation_id)
            self.assertTrue(selection["selection_observation_complete"])
            self.assertEqual(end["final_owned"][0]["confirmation_record_id"], selection["record_id"])

            exported = research_data.export_payload(collector.research_store.database_path, run_id)
            records = {row["record_id"]: row for row in exported["tables"]["records"]}
            self.assertEqual(records[confirmation_id]["event_type"], "selection_ui_transition_observed")
            final = json.loads(records[end["record_id"]]["payload_json"])
            self.assertEqual(final["final_owned"][0]["confirmation_record_id"], confirmation_id)
            selected = json.loads(records[confirmation_id]["payload_json"])["selected"]
            self.assertEqual((selected["potential_id"], selected["level_after"]), (101, 1))


if __name__ == "__main__":
    unittest.main()
