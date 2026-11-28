"""Keep state-space observations and their quality when journals are compacted."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import research_data, sampling_data


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

    def test_opening_and_first_battle_keep_the_observed_total_out_of_single_battle_statistics(self):
        from custom.climb_tower_neural.note_runtime import NoteRuntime

        runtime = Mock()
        runtime.collector.run = None
        runtime.tracker = None
        runtime.session.run_confirmed = True
        notes = NoteRuntime(runtime)
        context = Mock()
        context.get_node_data.return_value = {"attach": {"active_melodies": ["melody_of_lux", "melody_of_terra"]}}
        image = Mock(shape=(720, 1280, 3))
        notes.start(context, image, setup=dict(starting_supply={}, readings=[], conflicts=[],
            supply_complete=True, main_discs=[], complete=True))
        counts = dict.fromkeys(notes.ledger.visible_melodies, 0)
        for floor, amount in ((1, 6), (3, 9)):
            if floor == 3:
                notes._battle(floor)
            counts["melody_of_terra"] = amount
            observed = notes.ledger.observe(counts, page="result")
            notes._emit(dict(observed, floor=floor, floor_complete=True))
        notes.finish()
        calls = runtime.collector.emit.call_args_list
        for event in ("note_source_started", "note_inventory_observed"):
            first = next(call.kwargs for call in calls if call.args == (event,))
            self.assertTrue(first["opening_included"])
            self.assertFalse(first["probability_eligible"])
        deltas = [call.kwargs for call in calls if call.args == ("note_delta",)]
        opening, following = deltas
        payload = self.project(("opening", "note_delta", opening), ("following", "note_delta", following))
        records = self.payloads(payload)
        self.assertTrue(records["opening"]["opening_included"])
        self.assertEqual(records["opening"]["analysis_exclusion_reason"], "opening_and_first_battle_not_separable")
        self.assertNotIn("opening_included", records["following"])
        self.assertEqual(self.query("SELECT floor,delta,probability_eligible,usable_for_interval_statistics "
                                    "FROM note_deltas WHERE melody='melody_of_terra' ORDER BY sequence"),
                         [(1, 6, 0, 0), (3, 3, 1, 1)])

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

    def test_collector_keeps_offer_and_derived_pool_history_quality(self):
        collector = sampling_data.SamplingCollector(self.directory, Mock())
        collector.preset = dict(party=[dict(role=1, character_id=1, targets=[dict(
            potential_id=101, rarity_class="rare", target_level=3)])])
        collector.target_vector = [3]
        collector.start_intent("start")
        collector.confirm_start("entered")
        card = SimpleNamespace(index=0, rarity="rare", selected=False)
        data = SimpleNamespace(potentials=[card], current_coin=410,
            refresh_cost=40, refresh_count=0, core_potential=False, level_upped=True,
            params=SimpleNamespace(potential_source="default"))
        read = Mock(read_id="read-1")
        read.matching.return_value = []
        observed = candidate(name="known potential", rarity="rare", is_new=True,
            core=False, level_after=1, identity_name_input_complete=True)
        pool = dict(positions=[[101, None]], unknown_identities=[[102]],
                    projection_unique=False, compatibility="unknown_slots_without_prior")
        with (patch.object(collector, "potential", return_value=observed),
              patch.object(collector.inventory, "fields", side_effect=AssertionError("full inventory rebuilt"))):
            first_offer = collector.observe_offer(data, read)
            collector.model_decision(dict(pool_identity_observation=pool))
            collector.model_decision(dict(pool_identity_observation=pool))
            collector.pending_selection = dict(offer_id=first_offer, operation_id="uncertain-take",
                selected=candidate(core=False, level_after=1, recognition_status="ok",
                    identity_inferred=True, identity_match_validated=False))
            collector.selection_confirmed("potential_page_transition")
            # Even an unchanged pool must carry the newly incomplete history.
            collector.model_decision(dict(pool_identity_observation=pool))
            observed.update(level_before=1, offered_level_after=2, level_after=2, is_new=False)
            second_offer = collector.observe_offer(data, read)
        collector.emit("exploration_end", reason="settlement_saved", completion_observed=True)
        run_path = collector.directory / "runs" / f"{collector.run['run_id']}.jsonl"
        payload = research_data.build_payload(run_path)
        research_data.import_payload(self.database, payload)
        rows = payload["tables"]["records"]
        offers = {json.loads(row["payload_json"])["offer_id"]: row for row in rows
                  if row["event_type"] == "offer_observed"}
        first, second = (json.loads(offers[key]["payload_json"]) for key in (first_offer, second_offer))
        self.assertTrue(first["owned_complete"])
        self.assertFalse(second["owned_complete"])
        self.assertEqual(second["owned_issues"][0]["reason"], "selected_potential_unresolved")
        self.assertNotIn("owned", first)
        self.assertEqual(self.query("SELECT v.usable_for_draw_statistics FROM potential_draw_validity v "
                                    "JOIN records r USING(record_id) ORDER BY r.sequence"),
                         [(1,), (0,)])
        pools = [row for row in rows if row["event_type"] == "pool_observed"]
        self.assertEqual(len(pools), 2)
        self.assertEqual([(row["basis"], row["quality_status"]) for row in pools],
                         [("confirmed_history_reconstruction", "valid"),
                          ("confirmed_history_reconstruction", "partial")])
        self.assertEqual(self.query("SELECT effective_quality_status FROM record_quality "
                                    "WHERE event_type='pool_observed' ORDER BY sequence"),
                         [("valid",), ("partial",)])
        second_pool = json.loads(pools[1]["payload_json"])
        self.assertFalse(second_pool["owned_complete"])
        self.assertEqual(second_pool["owned_issues"], second["owned_issues"])
        self.assertEqual(second_pool["pool_identity_observation"], pool)
        self.assertEqual(research_data.compact_payload(payload), payload)
        # Recompaction preserves recorded provenance, including legacy defaults.
        pools[0]["basis"] = "game_UI_observation"
        pools[1].pop("basis")
        compact_pools = [row for row in research_data.compact_payload(payload)["tables"]["records"]
                         if row["event_type"] == "pool_observed"]
        self.assertEqual([row["basis"] for row in compact_pools], ["game_UI_observation"] * 2)
        collector.logger.error.assert_not_called()

    def test_inventory_history_fields_are_independent_snapshots(self):
        inventory = sampling_data.ObservedInventory(full_start=True)
        inventory.uncertain("selected_potential_unresolved", details=dict(values=[1]))
        history = inventory.owned_history_fields()
        full = inventory.fields()
        inventory.issues[0]["details"]["values"].append(2)
        inventory.issues.append(dict(reason="take_not_confirmed"))
        expected = [dict(reason="selected_potential_unresolved", details=dict(values=[1]))]
        self.assertEqual(history["owned_issues"], expected)
        self.assertEqual(full["owned_issues"], expected)

    def test_confirmed_anonymous_upgrade_after_skipped_take_keeps_partial_inventory(self):
        for old_count, new_count in ((0, 0), (0, 2), (2, 1)):
            with self.subTest(old_count=old_count, new_count=new_count):
                inventory = sampling_data.ObservedInventory(full_start=True)
                selected = candidate(potential_id=None, local_identity="local-zero-target",
                    target_level=0, core=False, level_after=1)
                self.assertFalse(inventory.confirm(dict(selected, identity_inferred=True), "guessed-take"))
                inventory.anonymous[1, "rare", 1] = old_count
                inventory.anonymous[1, "rare", 2] = new_count
                self.assertTrue(inventory.confirm(dict(selected, level_before=1, level_after=2), "confirmed-take"))
                fields = inventory.fields()
                expected = {2: new_count + 1}
                if old_count > 1:
                    expected[1] = old_count - 1
                self.assertEqual({row["level"]: row["count"] for row in fields["owned_anonymous"]}, expected)
                self.assertFalse(fields["owned_complete"])
                self.assertEqual(fields["owned_issues"], [dict(reason="selected_potential_unresolved",
                    confirmation_record_id="guessed-take")])
                payload = self.project(("inventory", "bag_inventory_observed", fields))
                saved = self.payloads(payload)["inventory"]
                self.assertEqual({row["level"]: row["count"] for row in saved["owned_anonymous"]}, expected)
                self.assertFalse(saved["owned_complete"])

    def test_inventory_accepts_validated_fuzzy_identity(self):
        inventory = sampling_data.ObservedInventory(full_start=True)
        self.assertTrue(inventory.confirm(candidate(core=False, level_after=1,
            identity_inferred=False, identity_match_validated=True), "validated-take"))
        self.assertTrue(inventory.complete)
        self.assertEqual(inventory.items[101]["level"], 1)

    def test_inferred_upgrade_does_not_make_later_level_readings_conflict(self):
        inventory = sampling_data.ObservedInventory(full_start=True)
        card = candidate(core=False, level_after=1)
        inventory.confirm(card, "first")
        inventory.confirm(dict(card, potential_id=102), "other")
        self.assertFalse(inventory.confirm(dict(card, level_before=1, level_after=2,
            identity_inferred=True), "uncertain"))

        later = dict(card, level_before=2, level_after=3)
        fuzzy = [dict(later, potential_id=identity, identity_inferred=True) for identity in (101, 102)]
        self.assertEqual(inventory.audit(fuzzy, "fuzzy"), [])
        self.assertEqual(inventory.audit([later], "clear"), [])
        self.assertEqual(inventory.owned_history_fields()["owned_issues"], [
            dict(reason="selected_potential_unresolved", confirmation_record_id="uncertain")])
        self.assertFalse(inventory.complete)
        self.assertEqual(inventory.items[101]["level"], 1)

        conflicts = inventory.audit([dict(later, level_before=3), dict(later, potential_id=102)], "later")
        self.assertEqual([(row["potential_id"], row["history_level"]) for row in conflicts],
                         [(101, 2), (102, 1)])

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
