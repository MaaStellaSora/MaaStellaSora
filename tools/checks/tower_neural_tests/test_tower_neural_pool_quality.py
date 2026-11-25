"""Inferred pool positions retain their provenance until independent reads resolve them."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import Mock

from tower_temp_directory import TowerTempDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.pool_identity import IdentityPools
from custom.climb_tower_neural import research_data, sampling_data


def preset():
    party = [dict(role=role, targets=[dict(potential_id=role * 100 + index,
        rarity_class="common", target_level=index) for index in range(1, 5)],
        catalogue_order=dict(rare=[], common=[role * 100 + index for index in range(1, 5)]))
        for role in (1, 2, 3)]
    return dict(party=party, pools=[[], [1, 2, 3, 4]] * 3)


class TowerNeuralPoolQualityTests(unittest.TestCase):
    def test_swaps_preserve_displaced_uncertainty_and_confirmed_reads_resolve_it(self):
        pools = IdentityPools(preset())
        pools.observe({1: [103]}, inferred={1: [True]})
        self.assertEqual(pools.inferred[1], [True, True, True, True])
        copied = pools.copy()
        copied.observe({1: [104]})
        self.assertEqual(copied.inferred[1], [False, True, True, True])
        self.assertEqual(pools.inferred[1], [True, True, True, True])
        # A clear page in another pool cannot repair the earlier swap.
        copied.observe({3: [204, 203]})
        self.assertIn("position_identity_inferred", copied.snapshot())
        # A complete independent read may also correct uncertain identities.
        copied.observe({1: [104, 102, 103, 101]})
        self.assertNotIn("position_identity_inferred", copied.snapshot())

    def test_partial_reads_restore_only_the_positions_they_independently_confirm(self):
        pools = IdentityPools(preset())
        pools.observe({1: [103]}, inferred={1: [True]})
        pools.observe({1: [103, 102, 101, 104]})
        self.assertNotIn("position_identity_inferred", pools.snapshot())
        pools.observe({1: [104]}, inferred={1: [True]})
        pools.remove(1, 102)
        self.assertIn("position_identity_inferred", pools.snapshot())
        self.assertEqual(len(pools.inferred[1]), len(pools.slots[1]))

    def test_wrong_guessed_known_source_allows_the_actual_clear_full_pool(self):
        pools = IdentityPools(preset())
        # Guessed C at the first slot; the game actually drew D from A B C D.
        pools.observe({1: [103]}, inferred={1: [True]})
        self.assertEqual(pools.slots[1], [103, 102, 101, 104])
        self.assertEqual(pools.inferred[1], [True, True, True, True])
        pools.observe({1: [104, 102, 103, 101]})
        self.assertEqual(pools.slots[1], [104, 102, 103, 101])
        self.assertNotIn("position_identity_inferred", pools.snapshot())

    def test_unknown_swap_only_taints_reachable_suffix_and_compact_clean_snapshots(self):
        pools = IdentityPools(preset())
        self.assertNotIn("position_identity_inferred", pools.snapshot())
        pools.forget_order()
        pools.observe({1: [101, 103]}, inferred={1: [False, True]})
        self.assertEqual(pools.inferred[1], [False, True, True, True])
        pools.observe({1: [101, 103, 102]})
        self.assertEqual(pools.inferred[1], [False, False, False, True])
        pools.observe({1: [101, 103, 102, 104]})
        self.assertNotIn("position_identity_inferred", pools.snapshot())

    def test_unknown_removal_and_order_forgetting_preserve_unresolved_provenance(self):
        pools = IdentityPools(preset())
        pools.forget_order()
        pools.observe({1: [101, 103]}, inferred={1: [False, True]})
        pools.remove(1, 102)
        self.assertEqual(pools.inferred[1], [False, True, True])
        pools.forget_order()
        self.assertEqual(pools.inferred[1], [True, True, True])
        pools.observe({1: [104, 101, 103]})
        self.assertNotIn("position_identity_inferred", pools.snapshot())

    def test_collector_and_sqlite_preserve_quality_across_clear_pages(self):
        with TowerTempDirectory() as directory:
            directory = Path(directory)
            collector = sampling_data.SamplingCollector(directory, Mock())
            collector.preset = preset()
            collector.target_vector = [1, 2, 3, 4] * 3
            collector.start_intent("start")
            collector.confirm_start("entered")
            pools = IdentityPools(collector.preset)
            pools.observe({1: [103]}, inferred={1: [True]})
            collector.model_decision(dict(pool_identity_observation=pools.snapshot()))
            pools.observe({3: [204, 203]})
            collector.model_decision(dict(pool_identity_observation=pools.snapshot()))
            pools.observe({1: [103, 102, 101, 104]})
            collector.model_decision(dict(pool_identity_observation=pools.snapshot()))
            collector.emit("exploration_end", reason="settlement_saved", completion_observed=True)
            path = collector.directory / "runs" / f"{collector.run['run_id']}.jsonl"
            payload = research_data.build_payload(path)
            database = directory / "research.sqlite3"
            research_data.import_payload(database, payload)
            with closing(sqlite3.connect(database)) as connection:
                qualities = connection.execute("SELECT effective_quality_status FROM record_quality "
                    "WHERE event_type='pool_observed' ORDER BY sequence").fetchall()
            self.assertEqual(qualities, [("valid",), ("valid",), ("valid",)])
            records = [row for row in research_data.export_payload(database, collector.run["run_id"])
                       ["tables"]["records"] if row["event_type"] == "pool_observed"]
            self.assertEqual([row["basis"] for row in records], [
                "inferred_identity_history_reconstruction", "inferred_identity_history_reconstruction",
                "confirmed_history_reconstruction"])
            self.assertTrue(all(json.loads(row["payload_json"])["owned_complete"] for row in records))
            observations = [json.loads(row["payload_json"])["pool_identity_observation"] for row in records]
            self.assertEqual(observations[0]["position_identity_inferred"][1], [True] * 4)
            self.assertEqual(observations[1]["position_identity_inferred"][1], [True] * 4)
            self.assertNotIn("position_identity_inferred", observations[2])
            self.assertTrue(all("identity_inferred" not in value and "identity_basis" not in value
                                for value in observations))
            self.assertEqual(research_data.compact_payload(payload), payload)


if __name__ == "__main__":
    unittest.main()
