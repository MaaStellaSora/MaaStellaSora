"""Nested reward failures must retain the enclosing transaction outcome."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace
from tower_temp_directory import TowerTempDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "agent"))
from custom.climb_tower_neural.sampling_data import SamplingCollector
from custom.climb_tower_neural.tower_sampling_collector import run_observed_operation


def offer(**overrides):
    value = dict(offer_id="offer-1", reward_chain_id="reward-1", source="battle",
        source_type="battle", floor=1, floor_complete=True, same_page_read_index=1,
        candidate_read_complete=True, candidates_complete=True, recognition_consistent=True,
        candidates=[dict(slot=0, role=1, potential_id=101, character_id=1,
            canonical_rarity="rare", level_before=0, offered_level_after=1,
            level_delta=1, target_level=3, recognition_status="ok")], owned=[], owned_complete=True)
    value.update(overrides)
    return value


class DeferredFinishTests(unittest.TestCase):
    def test_nested_failure_keeps_outer_results_before_end(self):
        with TowerTempDirectory() as directory:
            collector = SamplingCollector(directory, Mock())
            collector.start_intent("start")
            collector.confirm_start("entered")
            with collector.defer_finish():
                with collector.defer_finish():
                    collector.finish_run("potential_take_unconfirmed", completion_observed=False)
                self.assertIsNotNone(collector.run)
                result = collector.emit("purchase_result", success=False, pipeline_succeeded=False,
                                        reward_committed=False, operation_id="buy", debit_confirmed=True)
                self.assertIsNotNone(result)
                collector.finish_run("outer_failure", completion_observed=False)
            self.assertIsNone(collector.run)
            self.assertIsNone(collector._deferred_finish)

    def test_failed_enhancement_retains_confirmed_debit(self):
        with TowerTempDirectory() as directory:
            collector = SamplingCollector(directory, Mock())
            collector.start_intent("start")
            collector.confirm_start("entered")
            collector.session.observe_wallet(dict(coin=300, coin_complete=True))
            collector.emit = Mock(wraps=collector.emit)
            handler = SimpleNamespace(interactor=SimpleNamespace(last_enhance_pipeline_succeeded=False), data=SimpleNamespace(current_coin=300,
                current_enhancement_cost=100, params=SimpleNamespace(potential_source="enhance")))

            def enhance():
                collector.session.observe_wallet(dict(coin=200, coin_complete=True))
                with collector.session.action_scope(operation_id="nested-refresh"):
                    collector.session.observe_wallet(dict(coin=160, coin_complete=True))
                collector.finish_run("potential_take_unconfirmed", completion_observed=False)
                return False

            with collector.defer_finish():
                self.assertFalse(run_observed_operation(collector, "enhancement_result", handler,
                                                         enhance, before=True))
            result = next(call.kwargs for call in collector.emit.call_args_list
                          if call.args[0] == "enhancement_result")
            self.assertEqual((result["coins_after"], result["coin_after"]), (200, 160))
            self.assertTrue(result["debit_confirmed"])
            self.assertFalse(result["operation_confirmed"])
            self.assertFalse(result["result"])
            self.assertIsNone(collector.run)

    def test_real_shop_and_take_failure_export_results_and_usable_offer(self):
        from contextlib import nullcontext, closing
        from dataclasses import replace
        from types import MethodType
        import json
        import sqlite3
        from custom.climb_tower_neural import agent_callbacks, shop, research_data
        from custom.climb_tower_neural.maa_hook import ValueModelRuntime
        from custom.climb_tower_neural.tower_sampling_collector import ObservedShopHandler

        for kind, cost, exception, nested in (("BUY", 100, False, "take"), ("BUY", 100, True, "take"),
                ("ENHANCE", 60, False, "take"), ("ENHANCE", 0, False, "take"),
                ("ENHANCE", 60, True, "take"), ("ENHANCE", 60, False, "incomplete"),
                ("ENHANCE", 0, False, "incomplete"), ("ENHANCE", 60, False, "confirmed")):
            with self.subTest(kind=kind, cost=cost, exception=exception, nested=nested), TowerTempDirectory() as directory:
                collector = SamplingCollector(directory, Mock())
                collector.start_intent("start")
                collector.confirm_start("entered")
                run_id = collector.run["run_id"]
                collector.context = lambda context: context
                item = SimpleNamespace(grid_num=1, internal_name="potential_drink", quantity=1,
                    price=100, bought=False, trekker_specified=False, price_roi=[0, 0, 1, 1])
                before_coin = 410
                data = SimpleNamespace(current_coin=before_coin, current_enhancement_cost=60 if kind == "BUY" else cost,
                    shop_type="normal", refresh_remaining=0, refresh_cost=100, goods_missing=[],
                    items=[item], params=SimpleNamespace(potential_source="enhance"))
                tracker = Mock(state=replace(shop.sim.initial_843(), H=((1, 1, 1, 1),)))
                chosen = shop.Action(kind, index=1, price=cost, key="chosen")
                tracker.choose.return_value = chosen
                tracker.identity_pools.available = True
                runtime = SimpleNamespace(collector=collector, session=collector.session, tracker=tracker,
                    confirmed_preset=object(), pending_potential_refresh=None, notes=Mock(), logger=Mock(),
                    policy_mode="model", requested=Mock(return_value=True), confirm_start=Mock(),
                    ensure_tracker=Mock(return_value=True), require_tracker=Mock(return_value=tracker),
                    model_ready=Mock(return_value=True), decision_context=Mock(return_value={}),
                    ui_logs=SimpleNamespace(scope=lambda *args: nullcontext()))
                runtime.stop_model = MethodType(ValueModelRuntime.stop_model, runtime)
                runtime.observe_wallet = lambda reading: collector.record_wallet(collector.session.observe_wallet(reading))
                runtime.model = SimpleNamespace(prepare_actions=lambda state, candidates, context:
                    SimpleNamespace(state=state, actions=candidates, action_slots={"chosen": 1}))
                context = Mock()
                context.tasker.stopping = False
                context.tasker.post_stop.side_effect = lambda: setattr(context.tasker, "stopping", True)
                context.get_node_data.return_value = {}
                runtime.observe_wallet(dict(coin=before_coin, coin_complete=True))

                def task(*args, **kwargs):
                    runtime.observe_wallet(dict(coin=before_coin-cost, coin_complete=True))
                    links = collector.session.current_action_links
                    if nested == "incomplete":
                        with patch.object(agent_callbacks, "read_until", return_value=None), \
                             patch.object(agent_callbacks.Potential, "_get_params", return_value=data.params), \
                             patch.object(agent_callbacks, "PotentialContext", return_value=data):
                            agent_callbacks.Potential(runtime)._analyze_page(context,
                                SimpleNamespace(node_name="potential", image=None))
                    else:
                        source = "enhance" if kind == "ENHANCE" else "drink"
                        with collector.capture("potential_callback", operation_id=links["operation_id"]):
                            chain = collector.reward_chain(source)
                        collector.emit("offer_observed", **offer(source=source,
                            reward_chain_id=chain["reward_chain_id"], origin_operation_id=chain["origin_operation_id"]))
                        collector.pending_selection = dict(offer_id="offer-1", operation_id="take",
                            reward_chain_id=chain["reward_chain_id"], source_type=source, source_raw=source,
                            selected=offer()["candidates"][0])
                        runtime.pending_take = SimpleNamespace(potential=SimpleNamespace(name="test"),
                            tracker=tracker, committed=False, evidence={})
                        with patch.object(agent_callbacks, "finish_potential_take", return_value=nested == "confirmed"):
                            self.assertIs(agent_callbacks.PotentialTake(runtime).run(context, SimpleNamespace()),
                                          nested == "confirmed")
                        if nested == "confirmed":
                            tracker.selected.assert_called_once()
                            self.assertIsNone(runtime.pending_take)
                            runtime.stop_model(context, "test_after_take", phase="potential")
                    if exception:
                        raise RuntimeError("nested failure")
                    return SimpleNamespace(status=SimpleNamespace(succeeded=nested != "take"))

                context.run_task.side_effect = task
                # Keep production purchase/ enhancement orchestration and failure callback;
                # only replace unrelated shelf OCR/navigation and the take recognizer.
                with patch.object(agent_callbacks.Shop, "_init_data", return_value=data), \
                     patch.object(ObservedShopHandler, "read_main_page_info"), \
                     patch.object(ObservedShopHandler, "read_goods_page_info"), \
                     patch.object(shop.ModelShopInteractor, "back_to_main_page", return_value=True), \
                     patch.object(agent_callbacks, "PotentialInteractor", return_value=Mock()):
                    if exception:
                        with self.assertRaisesRegex(RuntimeError, "nested failure"):
                            agent_callbacks.Shop(runtime).run(context, SimpleNamespace(node_name="shop"))
                    else:
                        self.assertFalse(agent_callbacks.Shop(runtime).run(context, SimpleNamespace(node_name="shop")))
                self.assertTrue(context.tasker.stopping)
                self.assertIsNone(collector.run)
                payload = research_data.export_payload(collector.research_store.database_path, run_id)
                rows = payload["tables"]["records"]
                event = "purchase_result" if kind == "BUY" else "enhancement_result"
                outcome = next(row for row in rows if row["event_type"] == event)
                fields = json.loads(outcome["payload_json"])
                if nested == "confirmed":
                    gained = next(row for row in rows if row["event_type"] == "selection_ui_transition_observed")
                    self.assertTrue(json.loads(gained["payload_json"])["operation_confirmed"])
                    self.assertEqual(fields["confirmation_record_id"], gained["record_id"])
                if nested == "incomplete":
                    self.assertFalse(any(row["event_type"] == "offer_observed" for row in rows))
                    self.assertFalse(any(row["event_type"] == "selection_ui_transition_observed" for row in rows))
                    self.assertEqual(payload["tables"]["runs"][0]["termination_reason"], "potential_observation_incomplete")
                self.assertIs(fields["pipeline_succeeded"], None if exception else nested != "take")
                self.assertIs(fields["debit_confirmed"], True if cost else None)
                if kind == "ENHANCE":
                    self.assertIs(fields["operation_confirmed"], nested == "confirmed")
                    self.assertEqual(fields["confirmation_status"], "confirmed" if nested == "confirmed" else "unconfirmed")
                    self.assertEqual((fields["coin_before"], fields["coin_after"]), (before_coin, before_coin-cost))
                else:
                    self.assertFalse(fields["reward_committed"])
                self.assertLess(outcome["sequence"], next(row["sequence"] for row in rows
                                                       if row["event_type"] == "exploration_end"))
                with closing(sqlite3.connect(collector.research_store.database_path)) as connection:
                    self.assertEqual(connection.execute("SELECT usable_for_draw_statistics FROM potential_draw_validity").fetchall(),
                                     [] if nested == "incomplete" else [(1,)])
                    self.assertEqual(connection.execute("SELECT count(*) FROM quality_issues WHERE code='unconfirmed_action_tail'").fetchone()[0], 0)
                    self.assertEqual(connection.execute("SELECT count(*) FROM quality_issues WHERE code='action_not_confirmed'").fetchone()[0], int(kind == "ENHANCE" and nested != "confirmed"))
                    if kind == "ENHANCE" and nested != "confirmed":
                        self.assertNotEqual(connection.execute("SELECT effective_quality_status FROM record_quality WHERE event_type='enhancement_result'").fetchone()[0], "valid")

    def test_legacy_missing_result_only_exempts_its_own_shop_reward(self):
        import json
        import sqlite3
        from contextlib import closing
        from custom.climb_tower_neural import research_data
        with TowerTempDirectory() as directory:
            collector = SamplingCollector(directory, Mock())
            collector.start_intent("start")
            collector.confirm_start("entered")
            run_id = collector.run["run_id"]
            collector.emit("purchase_attempt", operation_id="purchase")
            for offer_id, source, origin in (("related", "drink", "purchase"),
                    ("unrelated", "drink", "another"), ("unknown", "drink", None),
                    ("battle", "battle", "purchase")):
                collector.emit("offer_observed", **offer(offer_id=offer_id, source=source,
                                                         origin_operation_id=origin))
            collector.finish_run("test", completion_observed=False)
            with closing(sqlite3.connect(collector.research_store.database_path)) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM quality_issues WHERE code='unconfirmed_action_tail'").fetchone()[0], 1)
                self.assertEqual(dict(connection.execute("SELECT offer_id,usable_for_draw_statistics FROM potential_draw_validity")),
                    dict(related=1, unrelated=0, unknown=0, battle=0))


if __name__ == "__main__":
    unittest.main()
