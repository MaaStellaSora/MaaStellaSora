"""Keep purchase debits separate from nested reward refreshes and recording."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

from tower_temp_directory import TowerTempDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.purchase_evidence import purchase_wallet_evidence
from custom.climb_tower_neural import research_data, sampling_data


def wallet(observation_id, coin, operation_id=None):
    return dict(observation_id=observation_id, coin=coin, coin_complete=coin is not None,
                recognition_status="ok" if coin is not None else "missing", operation_id=operation_id)


def wallet_event(observation):
    fields = dict(observation)
    return fields.pop("observation_id"), "wallet_observed", fields


def project(events):
    events = [("start", "exploration_start", dict(full_run_start_observed=True)),
              *events, ("end", "exploration_end", dict(completion_observed=True))]
    rows = [dict(record_id=record_id, event_type=event, run_id="purchase-test",
                 sequence=sequence, run_sequence=sequence, **fields)
            for sequence, (record_id, event, fields) in enumerate(events, 1)]
    with TowerTempDirectory() as directory:
        path = Path(directory) / "purchase-test.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        payload = research_data.build_payload(path)
    return {row["record_id"]: json.loads(row["payload_json"])
            for row in payload["tables"]["records"]}


class TowerNeuralPurchaseWalletTests(unittest.TestCase):
    def test_collector_database_export_preserves_independent_purchase_outcomes(self):
        # Generated conflicting and missing observations, not a game-log replay.
        cases = [(False, True, 310), (True, False, 310),
                 (True, True, 300), (True, True, None), (True, True, 310)]
        with TowerTempDirectory() as directory:
            collector = sampling_data.SamplingCollector(directory, Mock())
            collector.start_intent("start")
            collector.confirm_start("entered")
            result_ids = []
            for index, (pipeline, committed, after_coin) in enumerate(cases):
                operation = f"buy-{index}"
                before = wallet(f"before-{index}", 410)
                after = wallet(f"after-{index}", after_coin, operation)
                observation_id, event, fields = wallet_event(before)
                collector.emit(event, record_id=observation_id, **fields)
                collector.emit("purchase_attempt", operation_id=operation,
                    wallet_before_observation_id=before["observation_id"])
                observation_id, event, fields = wallet_event(after)
                collector.emit(event, record_id=observation_id, **fields)
                record_id = f"result-{index}"
                result_ids.append(record_id)
                collector.emit("purchase_result", record_id=record_id, operation_id=operation,
                    item=dict(price=100), pipeline_succeeded=pipeline,
                    reward_committed=committed, success=pipeline and committed)
            collector.emit("exploration_end", completion_observed=True)
            run_id = collector.run["run_id"]
            path = collector.directory / "runs" / f"{run_id}.jsonl"
            journal = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            journal_results = {row["record_id"]: row for row in journal}
            payload = research_data.build_payload(path)
            database = Path(directory) / "research.sqlite3"
            research_data.import_payload(database, payload)
            exported = research_data.export_payload(database, run_id)
            records = {row["record_id"]: json.loads(row["payload_json"])
                       for row in exported["tables"]["records"]}
            for record_id, (pipeline, committed, after_coin) in zip(result_ids, cases):
                with self.subTest(record_id=record_id):
                    result = records[record_id]
                    for source in (journal_results[record_id], result):
                        self.assertIs(source["pipeline_succeeded"], pipeline)
                        self.assertIs(source["reward_committed"], committed)
                        self.assertIs(source["success"], pipeline and committed)
                    self.assertIs(result["debit_confirmed"],
                                  None if after_coin is None else after_coin == 310)
                    self.assertNotIn("pipeline_reported_success", result)

    def test_missing_runtime_outcomes_are_not_reconstructed_from_debit(self):
        records = project([wallet_event(wallet("before", 410)),
            ("attempt", "purchase_attempt", dict(operation_id="buy")),
            wallet_event(wallet("after", 310, "buy")),
            ("result", "purchase_result", dict(operation_id="buy", item=dict(price=100), success=False))])
        result = records["result"]
        self.assertTrue(result["debit_confirmed"])
        self.assertFalse(result["success"])
        self.assertNotIn("pipeline_succeeded", result)
        self.assertNotIn("reward_committed", result)
        self.assertNotIn("pipeline_reported_success", result)

    def test_twelve_purchases_keep_first_linked_wallet_separate_from_pipeline_end(self):
        # Construct BUG-018's documented shape: 12 purchases, 9 followed by
        # reward refreshes. The first 740 -> 640 -> 560 case is documented;
        # the other amounts below are generated fixtures, not a raw-log replay.
        events, expected = [], {}
        for index in range(12):
            operation = f"buy-{index}"
            before = wallet(f"before-{index}", 740 + 100 * index)
            readings = [wallet(f"other-{index}", before["coin"] - 200, f"other-op-{index}"),
                        wallet(f"missing-{index}", None, operation),
                        wallet(f"after-{index}", before["coin"] - 100, operation)]
            if index < 9:
                readings.append(wallet(f"refresh-{index}", before["coin"] - 180, f"refresh-op-{index}"))
            evidence = purchase_wallet_evidence(operation, 100, readings,
                before_wallet=before, pipeline_wallet=readings[-1])
            self.assertEqual(evidence["purchase_debit"], 100)
            self.assertTrue(evidence["debit_confirmed"])
            self.assertEqual(evidence["wallet_observation_id"], f"after-{index}")
            self.assertEqual(evidence["pipeline_coins_after"], readings[-1]["coin"])
            expected[f"result-{index}"] = evidence
            events.extend([wallet_event(before),
                (f"attempt-{index}", "purchase_attempt", dict(operation_id=operation)),
                *(wallet_event(reading) for reading in readings),
                (f"result-{index}", "purchase_result", dict(operation_id=operation,
                    item=dict(price=100), pipeline_succeeded=True))])

        records = project(events)
        for record_id, evidence in expected.items():
            with self.subTest(record_id=record_id):
                for key in ("coins_before", "coins_after", "purchase_debit", "debit_confirmed",
                            "wallet_before_observation_id", "wallet_observation_id",
                            "pipeline_coins_after", "pipeline_wallet_observation_id"):
                    self.assertEqual(records[record_id][key], evidence[key])
        self.assertEqual((records["result-0"]["coins_after"],
                          records["result-0"]["pipeline_coins_after"]), (640, 560))

    def test_zero_balance_and_observation_identity_work_without_a_record_id(self):
        before, after = wallet("before", 100), wallet("after", 0, "buy")
        evidence = purchase_wallet_evidence("buy", 100, [after], before_wallet=before, pipeline_wallet=after)
        self.assertEqual((evidence["coins_after"], evidence["pipeline_coins_after"]), (0, 0))
        self.assertTrue(evidence["debit_confirmed"])
        unchanged = purchase_wallet_evidence("buy", 100, [], before_wallet=before, pipeline_wallet=before)
        self.assertIsNone(unchanged["purchase_debit"])
        self.assertIsNone(unchanged["pipeline_coins_after"])
        self.assertIsNone(unchanged["pipeline_wallet_observation_id"])

    def test_legacy_retry_rows_and_legacy_before_selection_still_export(self):
        first = wallet_event(wallet("before", 410))
        missing = wallet_event(wallet("recent-missing", None))
        retry_start = wallet_event(wallet("read-missing", None, "buy"))
        retry_final = wallet_event(wallet("read-complete", 310, "buy"))
        retry_start[2].update(read_id="legacy-read", node_name="星塔_神经网络_识别当前金币")
        retry_final[2].update(read_id="legacy-read", node_name="星塔_神经网络_识别当前金币__数字重读")
        records = project([first, missing,
            ("attempt", "purchase_attempt", dict(operation_id="buy")),
            retry_start, retry_final, wallet_event(wallet("refresh", 290, "refresh-op")),
            ("result", "purchase_result", dict(operation_id="buy", item=dict(price=100)))])
        result = records["result"]
        self.assertEqual((result["coins_before"], result["coins_after"], result["pipeline_coins_after"]),
                         (410, 310, 290))
        self.assertTrue(result["debit_confirmed"])
        self.assertEqual(result["wallet_before_observation_id"], "before")
        self.assertEqual(result["wallet_observation_id"], "read-complete")
        self.assertEqual(records["read-missing"]["wallet_observation_id"], "read-complete")
        self.assertNotIn("read_id", records["read-complete"])

    def test_explicit_before_reference_never_falls_back_to_an_older_complete_wallet(self):
        for reference, coin in (("current", 410), ("current", None), ("lost-record", None), (None, None)):
            with self.subTest(reference=reference, coin=coin):
                attempt = research_data.compact_record("purchase_attempt", dict(
                    operation_id="buy", wallet_before_observation_id=reference))
                self.assertIn("wallet_before_observation_id", attempt)
                records = project([wallet_event(wallet("old-complete", 740)),
                    wallet_event(wallet("current", coin)),
                    ("attempt", "purchase_attempt", attempt),
                    wallet_event(wallet("after", 310, "buy")),
                    ("result", "purchase_result", dict(operation_id="buy", item=dict(price=100)))])
                result = records["result"]
                self.assertEqual(result["coins_before"], coin)
                if coin is None:
                    self.assertIsNone(result["purchase_debit"])
                    self.assertIsNone(result["debit_confirmed"])
                    self.assertEqual(result["confirmation_status"], "wallet_missing")
                else:
                    self.assertTrue(result["debit_confirmed"])
                    self.assertEqual(result["wallet_before_observation_id"], "current")


if __name__ == "__main__":
    unittest.main()
