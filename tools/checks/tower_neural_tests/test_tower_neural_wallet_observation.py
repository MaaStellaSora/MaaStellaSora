"""Publish the final UI wallet read independently of research persistence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.events import _read_event_coins
from custom.climb_tower_neural.interactor import Interactor, WALLET_NODE
from custom.climb_tower_neural.maa_hook import ValueModelRuntime
from custom.climb_tower_neural.purchase_evidence import purchase_wallet_evidence
from custom.climb_tower_neural import sampling_data
from custom.climb_tower_neural.session import RunSession


def result(text=None, *, score=.99, hit=True, reco_id=1):
    rows = ([] if text is None else
            [SimpleNamespace(text=text, score=score, box=[35, 20, 40, 20])])
    return SimpleNamespace(hit=bool(rows) and hit, all_results=rows,
                           filtered_results=rows if hit else [], reco_id=reco_id)


def context_with(*readings, locate_first=False):
    node = dict(recognition=dict(param=dict(roi=[20, 10, 160, 40], only_rec=True,
                                           threshold=.85)),
                attach=dict(strict_number_reading=True, wallet_text_locator=locate_first))
    return SimpleNamespace(
        tasker=SimpleNamespace(stopping=False, controller=SimpleNamespace(cached_image=object())),
        get_node_data=Mock(return_value=node), run_recognition=Mock(side_effect=readings))


def runtime_with(session, collector):
    # Exercise the real runtime boundary without constructing an unrelated model.
    runtime = ValueModelRuntime.__new__(ValueModelRuntime)
    runtime.session, runtime.collector = session, collector
    return runtime


class TowerNeuralWalletObservationTests(unittest.TestCase):
    def test_zero_and_thousands_separator_publish_the_actual_amount_once(self):
        for text, amount in (("0", 0), ("1,150", 1150)):
            with self.subTest(text=text):
                context = context_with(result(text, reco_id=11), result(text, reco_id=12), locate_first=True)
                readings = []
                self.assertEqual(Interactor(context, on_wallet=readings.append).get_current_coin(), amount)
                self.assertEqual(len(readings), 1)
                self.assertEqual((readings[0]["coin"], readings[0]["recognition_status"]), (amount, "ok"))
                self.assertTrue(readings[0]["coin_complete"])
                self.assertEqual(readings[0]["native_reco_id"], 12)
                self.assertEqual([call.args[0] for call in context.run_recognition.call_args_list],
                                 [WALLET_NODE + "__文字定位", WALLET_NODE])

    def test_missing_read_uses_existing_locator_then_retry_and_publishes_only_final_read(self):
        context = context_with(result(), result("o", score=.4, reco_id=2),
                               result(reco_id=3), result("0", reco_id=4))
        readings = []
        self.assertEqual(Interactor(context, on_wallet=readings.append).get_current_coin(), 0)
        self.assertEqual([call.args[0] for call in context.run_recognition.call_args_list],
                         [WALLET_NODE, WALLET_NODE + "__文字定位", WALLET_NODE,
                          WALLET_NODE + "__数字重读"])
        self.assertEqual(len(readings), 1)
        self.assertEqual((readings[0]["coin"], readings[0]["native_reco_id"]), (0, 4))
        retry = context.run_recognition.call_args_list[-1].args[2][WALLET_NODE + "__数字重读"]
        self.assertFalse(retry["recognition"]["param"]["only_rec"])
        self.assertEqual(retry["recognition"]["param"]["roi"], [20, 10, 160, 40])

    def test_trusted_locator_conflict_stops_before_a_retry_can_overwrite_it(self):
        context = context_with(result("410", reco_id=21), result("470", reco_id=22),
                               result("470", reco_id=23), locate_first=True)
        readings = []
        self.assertEqual(Interactor(context, on_wallet=readings.append).get_current_coin(), -1)
        self.assertEqual([call.args[0] for call in context.run_recognition.call_args_list],
                         [WALLET_NODE + "__文字定位", WALLET_NODE])
        self.assertEqual(len(readings), 1)
        self.assertIsNone(readings[0]["coin"])
        self.assertFalse(readings[0]["coin_complete"])
        self.assertEqual(readings[0]["recognition_status"], "conflict")
        self.assertEqual(readings[0]["conflicts"], [410, 470])
        self.assertEqual(readings[0]["native_reco_id"], 22)
        self.assertEqual(set(readings[0]["unresolved_values"]), {"410", "470"})

    def test_exhausted_missing_or_invalid_read_replaces_a_previous_complete_balance(self):
        for final, status in ((result(reco_id=4), "missing"),
                              (result("o", reco_id=4), "invalid")):
            with self.subTest(status=status):
                session = RunSession()
                runtime = runtime_with(session, Mock())
                previous = runtime.observe_wallet(dict(coin=410, coin_complete=True, recognition_status="ok"))
                context = context_with(result(), result("o", score=.4, reco_id=2), result(reco_id=3), final)
                self.assertEqual(Interactor(context, on_wallet=runtime.observe_wallet).get_current_coin(), -1)
                current = session.current_wallet_observation
                self.assertIsNone(current["coin"])
                self.assertEqual(current["recognition_status"], status)
                self.assertNotEqual(current["observation_id"], previous["observation_id"])
                self.assertEqual(previous["coin"], 410)
                self.assertEqual(runtime.collector.record_wallet.call_count, 2)

    def test_missing_event_icon_replaces_old_balance_without_reading_an_amount(self):
        session = RunSession()
        runtime = runtime_with(session, Mock())
        previous = runtime.observe_wallet(dict(coin=740, coin_complete=True, recognition_status="ok"))
        context = context_with(result())
        image = object()
        self.assertIsNone(_read_event_coins(context, image, on_wallet=runtime.observe_wallet))
        context.run_recognition.assert_called_once_with("星塔_神经网络_定位事件钱包", image)
        context.get_node_data.assert_not_called()
        current = session.current_wallet_observation
        self.assertIsNone(current["coin"])
        self.assertEqual(current["recognition_status"], "missing")
        self.assertNotEqual(current["observation_id"], previous["observation_id"])
        self.assertEqual(runtime.collector.record_wallet.call_count, 2)

    def test_live_zero_is_published_before_recording_even_when_disabled_or_append_fails(self):
        for recording in ("enabled", "disabled", "write_failed"):
            with self.subTest(recording=recording), TowerTempDirectory() as directory:
                session = RunSession()
                session.begin_run()
                collector = sampling_data.SamplingCollector(Path(directory), Mock(), session=session)
                collector.enabled = recording != "disabled"
                collector.begin_task("cn", "wallet-test")
                collector.start_intent("start")
                collector.confirm_start("entered")
                runtime = runtime_with(session, collector)
                context = context_with(result("0", reco_id=31))
                append_record = sampling_data.append_record

                def append(path, data):
                    self.assertEqual(session.current_wallet_observation["coin"], 0)
                    if recording == "write_failed":
                        raise OSError("journal unavailable")
                    return append_record(path, data)

                with patch.object(sampling_data, "append_record", side_effect=append) as write:
                    with collector.capture("wallet") as read:
                        self.assertEqual(Interactor(collector.context(context),
                            on_wallet=runtime.observe_wallet).get_current_coin(), 0)
                current = session.current_wallet_observation
                self.assertTrue(current["coin_complete"])
                self.assertTrue(current["observation_id"])
                self.assertNotIn("record_id", current)
                if recording == "disabled":
                    write.assert_not_called()
                    self.assertIsNone(collector.last_wallet)
                else:
                    write.assert_called_once()
                    self.assertEqual(collector.last_wallet["recognition_record_id"],
                                     read.recognition_by_id(31)["record_id"])
                    if recording == "write_failed":
                        collector.logger.error.assert_called()
                        self.assertIsNone(collector.last_wallet["record_id"])
                    else:
                        self.assertEqual(collector.last_wallet["record_id"], current["observation_id"])

    def test_reading_a_refresh_price_does_not_publish_it_as_the_wallet(self):
        context = context_with(result("40"))
        on_wallet = Mock()
        self.assertEqual(Interactor(context, on_wallet=on_wallet)._read_number("星塔_潜能刷新费用"), 40)
        on_wallet.assert_not_called()

    def test_purchase_wallet_from_another_thread_precedes_nested_refresh_without_recording(self):
        with TowerTempDirectory() as directory, ThreadPoolExecutor(max_workers=1) as callback:
            session = RunSession()
            session.begin_run()
            collector = sampling_data.SamplingCollector(Path(directory), Mock(), session=session)
            collector.enabled = False
            runtime = runtime_with(session, collector)
            before = runtime.observe_wallet(dict(coin=410, coin_complete=True, recognition_status="ok"))
            with session.purchase_wallet_capture("buy") as capture:
                with session.action_scope(operation_id="buy"):
                    missing = runtime.observe_wallet(dict(coin=None, coin_complete=False,
                                                         recognition_status="missing"))
                    purchased = callback.submit(runtime.observe_wallet,
                        dict(coin=310, coin_complete=True, recognition_status="ok")).result(timeout=5)
                    repeated = runtime.observe_wallet(dict(coin=310, coin_complete=True, recognition_status="ok"))
                    with session.action_scope(operation_id="refresh"):
                        refreshed = runtime.observe_wallet(dict(coin=290, coin_complete=True,
                                                               recognition_status="ok"))
                evidence = purchase_wallet_evidence("buy", 100, capture["wallets"],
                    before_wallet=capture["before_wallet"], pipeline_wallet=session.current_wallet_observation)
            self.assertEqual(purchased["operation_id"], "buy")
            self.assertEqual(capture["wallets"], [missing, purchased, repeated])
            self.assertEqual(capture["before_wallet"], before)
            self.assertEqual((evidence["coins_before"], evidence["coins_after"],
                              evidence["purchase_debit"], evidence["pipeline_coins_after"]),
                             (410, 310, 100, 290))
            self.assertTrue(evidence["debit_confirmed"])
            self.assertEqual(evidence["wallet_observation_id"], purchased["observation_id"])
            self.assertEqual(evidence["pipeline_wallet_observation_id"], refreshed["observation_id"])
            self.assertIsNone(collector.last_wallet)

    def test_session_operation_overrides_stale_parent_read_and_clears_when_scope_ends(self):
        with TowerTempDirectory() as directory:
            session = RunSession()
            collector = sampling_data.SamplingCollector(Path(directory), Mock(), session=session)
            collector.begin_task("cn", "wallet-test")
            collector.start_intent("start")
            collector.confirm_start("entered")
            runtime = runtime_with(session, collector)
            with collector.capture("old_page", operation_id="old-operation") as parent:
                with session.action_scope(operation_id="current-buy"):
                    self.assertEqual(collector.new_read("wallet").links["operation_id"], "current-buy")
                    runtime.observe_wallet(dict(coin=310, coin_complete=True, recognition_status="ok"))
                    self.assertEqual(collector.last_wallet["operation_id"], "current-buy")
                self.assertIsNone(collector.new_read("after_purchase").links["operation_id"])
                runtime.observe_wallet(dict(coin=290, coin_complete=True, recognition_status="ok"))
                self.assertIsNone(session.current_wallet_observation["operation_id"])
                self.assertIsNone(collector.last_wallet["operation_id"])
                self.assertEqual(parent.links["operation_id"], "old-operation")


if __name__ == "__main__":
    unittest.main()
