"""Keep state-space observations and their quality when journals are compacted."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import unittest

from tower_temp_directory import TowerTempDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import research_data


def candidate(**overrides):
    value = dict(slot=0, role=1, potential_id=101, character_id=1,
                 canonical_rarity="rare", level_before=0, offered_level_after=1,
                 level_delta=1, target_level=3, recognition_status="ok")
    value.update(overrides)
    return value


def offer(**overrides):
    value = dict(offer_id="offer-1", reward_chain_id="reward-1", source="battle",
                 source_type="battle", floor=1, floor_complete=True,
                 same_page_read_index=1, candidate_read_complete=True,
                 candidates_complete=True, recognition_consistent=True,
                 candidates=[candidate()], owned=[], owned_complete=True)
    value.update(overrides)
    return value


def note_delta(**overrides):
    value = dict(operation_id="battle-1", source_type="battle", floor=1,
                 floor_complete=True, before={"alpha": 0}, after={"alpha": 2},
                 delta={"alpha": 2}, total=2, complete=True,
                 scope_melodies=["alpha"], tracked_melodies=["alpha"],
                 independently_observed=True, analysis_eligible=True,
                 probability_eligible=True)
    value.update(overrides)
    return value


class TowerNeuralResearchDataTests(unittest.TestCase):
    def setUp(self):
        temporary = TowerTempDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.database = self.directory / "research.sqlite3"

    def project(self, *events):
        preset = dict(party=[dict(role=1, character_id=1, targets=[dict(
            potential_id=101, rarity_class="rare", target_level=3)])])
        events = (("start", "exploration_start", dict(
            original_preset=preset, full_run_start_observed=True, owned=[], owned_complete=True)),
            *events, ("end", "exploration_end", dict(
                reason="settlement_saved", completion_observed=True)))
        rows = [dict(schema_version=3, collector_version="test", event_type=event,
            record_type=event, record_id=record_id, run_id="run-1", session_id="session-1",
            task_id="task-1", participant_id="private-participant",
            sequence=sequence, run_sequence=sequence,
            timestamp=f"2026-11-14T00:00:{sequence:02d}+09:00", lang_type="cn",
            target_vector=[3], target_vector_complete=True, **fields)
            for sequence, (record_id, event, fields) in enumerate(events, 1)]
        path = self.directory / "runs/run-1.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        payload = research_data.build_payload(path)
        research_data.import_payload(self.database, payload)
        return research_data.export_payload(self.database, "run-1")

    def payloads(self, payload):
        return {row["record_id"]: json.loads(row["payload_json"])
                for row in payload["tables"]["records"]}

    def query(self, sql):
        with closing(sqlite3.connect(self.database)) as connection:
            return connection.execute(sql).fetchall()

    def test_compact_observations_keep_targets_transitions_and_interval_scope(self):
        selected = candidate(level_after=1)
        combined_sources = [dict(operation_id="event-2", source_type="event"),
                            dict(operation_id="battle-3", source_type="battle")]
        payload = self.project(
            ("wallet", "wallet_observed", dict(coin=410, coin_complete=True,
                recognition_status="ok", parsed_values=[410])),
            ("offer", "offer_observed", offer()),
            ("selected", "selection_ui_transition_observed", dict(
                operation_id="selection-1", offer_id="offer-1", selected=selected,
                source_type="battle", operation_confirmed=True,
                confirmation_basis="reward_page_closed", selection_observation_complete=True,
                owned=[selected], owned_complete=True)),
            ("delta", "note_delta", note_delta()),
            ("combined", "note_delta", note_delta(operation_id="combined-2-3",
                source_type="combined_interval", floor=None, floor_complete=False,
                before={"alpha": 2}, after={"alpha": 5}, delta={"alpha": 3}, total=3,
                reason="stable_page_combined_observation", mixed_sources=combined_sources,
                interval_basis="event_then_battle_without_intermediate_capture",
                independently_observed=False, analysis_eligible=False,
                probability_eligible=False, analysis_exclusion_reason="combined_sources_not_separable")),
            ("settlement", "settlement_observed", dict(
                potential_counts=[1, 0, 0], potential_counts_complete=True,
                score=12, score_complete=True, final_owned=[selected], owned_complete=True)),
        )
        tables, records = payload["tables"], self.payloads(payload)
        self.assertEqual(set(tables), {"presets", "runs", "records", "quality_issues"})
        self.assertEqual(json.loads(tables["presets"][0]["target_vector_json"]), [3])
        self.assertEqual(json.loads(tables["presets"][0]["content_json"])["party"][0]["targets"][0]["target_level"], 3)
        run, = tables["runs"]
        self.assertEqual((run["lifecycle_status"], run["full_run_start_observed"], run["completion_observed"]),
                         ("completed", 1, 1))
        self.assertEqual(records["wallet"]["coin"], 410)
        self.assertEqual(records["selected"]["selected"]["level_before"], 0)
        self.assertEqual(records["selected"]["selected"]["level_after"], 1)
        self.assertTrue(records["selected"]["operation_confirmed"])
        self.assertEqual(records["combined"]["mixed_sources"], combined_sources)
        self.assertEqual(records["combined"]["scope_melodies"], ["alpha"])
        self.assertEqual(records["settlement"]["potential_counts"], [1, 0, 0])
        self.assertEqual(self.query("SELECT record_id,usable_for_draw_statistics FROM potential_draw_validity"),
                         [("offer", 1)])
        self.assertEqual(self.query("SELECT record_id,delta,in_scope,usable_for_interval_statistics "
                                    "FROM note_deltas ORDER BY sequence"),
                         [("delta", 2, 1, 1), ("combined", 3, 1, 0)])

    def test_compaction_drops_personal_identity_and_duplicate_recognition_evidence(self):
        raw = dict(algorithm="OCR", hit=True, all_results=[dict(text="raw-success-only",
            score=.99, box=[1, 2, 3, 4])], filtered_results=[dict(text="raw-success-only",
            score=.99, box=[1, 2, 3, 4])], best_result=dict(text="raw-success-only",
            score=.99, box=[1, 2, 3, 4]), raw_detail="transport-only")
        payload = self.project(
            ("recognition", "context_run_recognition", dict(raw_result=raw,
                arguments=["call-only"], roi=[1, 2, 3, 4], node_name="test-node")),
            ("offer", "offer_observed", offer(candidates=[candidate(raw_ocr=raw,
                recognition_record_ids=["recognition"], name_roi=[1, 2, 3, 4])])),
            ("model", "model_decision", dict(decision=dict(public_state=dict(features=[1, 2, 3])))),
        )
        text = json.dumps(payload)
        for removed in ("participant_id", "private-participant", "raw-success-only", "transport-only",
                        "call-only", "recognition_record_ids", "name_roi", "public_state"):
            self.assertNotIn(removed, text)
        self.assertEqual(set(self.payloads(payload)), {"start", "offer", "end"})
        self.assertEqual(self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"),
                         [("presets",), ("quality_issues",), ("records",), ("runs",)])

    def test_ambiguous_candidates_and_quality_ranges_survive_compaction(self):
        payload = self.project(
            ("prefix", "note_delta", note_delta()),
            ("attempt", "selection_result_attempt", dict(operation_id="unconfirmed-selection")),
            ("offer", "offer_observed", offer(candidates=[candidate(potential_id=None,
                name="ambiguous name", id_candidates=[101, 102], recognition_status="conflict",
                parsed_values=[1, 2], conflicts=[1, 2])])),
            ("after", "note_delta", note_delta(operation_id="battle-4", floor=4)),
        )
        records = self.payloads(payload)
        ambiguous, = records["offer"]["candidates"]
        self.assertEqual(ambiguous["id_candidates"], [101, 102])
        self.assertEqual(ambiguous["parsed_values"], [1, 2])
        self.assertEqual(ambiguous["conflicts"], [1, 2])
        self.assertEqual(ambiguous["name"], "ambiguous name")
        issue, = (row for row in payload["tables"]["quality_issues"]
                  if row["code"] == "unconfirmed_action_tail")
        self.assertEqual(issue["severity"], "partial")
        self.assertEqual(json.loads(issue["evidence_json"])["operation_id"], "unconfirmed-selection")
        self.assertEqual(self.query("SELECT record_id,quality_status,usable_for_interval_statistics "
                                    "FROM note_deltas ORDER BY sequence"),
                         [("prefix", "valid", 1), ("after", "partial", 0)])
        self.assertEqual(self.query("SELECT quality_status,usable_for_draw_statistics FROM potential_draw_validity"),
                         [("suspect", 0)])


if __name__ == "__main__":
    unittest.main()
