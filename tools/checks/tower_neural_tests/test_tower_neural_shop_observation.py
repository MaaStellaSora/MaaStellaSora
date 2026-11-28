"""Keep real shelf state and refresh outcomes independent of their records."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.critic.environment import sim
from custom.climb_tower_neural.interactor import Interactor, WALLET_NODE
from custom.climb_tower_neural.maa_hook import ValueModelRuntime
from custom.climb_tower_neural.potential_adapter import prepare_actions
from custom.climb_tower_neural.sampling_data import SamplingCollector, snapshot
from custom.climb_tower_neural.session import RunSession
from custom.climb_tower_neural.shop_ui.context import Item, ShopContext
from custom.climb_tower_neural.tower_sampling_collector import ObservedShopHandler


class TowerNeuralShopObservationTests(unittest.TestCase):
    def make_handler(self, *, recording=True, final_wallet_linked=True):
        directory = TowerTempDirectory()
        self.addCleanup(directory.cleanup)
        session = RunSession()
        session.begin_run()
        collector = SamplingCollector(Path(directory.name), Mock(), session=session)
        collector.enabled = recording
        collector.begin_task("cn", "shop-observation-test")
        collector.start_intent("start")
        collector.confirm_start("entered")
        collector.emit = Mock(wraps=collector.emit)
        runtime = ValueModelRuntime.__new__(ValueModelRuntime)
        runtime.session, runtime.collector = session, collector
        context = SimpleNamespace(tasker=SimpleNamespace(stopping=False), run_task=Mock())
        interactor = SimpleNamespace(context=context, image="before",
            enter_shopping=Mock(), is_shopping=Mock(return_value=True),
            get_refresh_cost=Mock(return_value=80), get_refresh_remaining=Mock(return_value=1),
            is_specified_drink=Mock(return_value=False), coins={"before": 410, "after-1": 330, "after-2": 330})
        readings = []

        def get_coin():
            amount = interactor.coins[interactor.image]
            native_id = len(readings) + 1
            if final_wallet_linked or interactor.image != "after-2":
                result = SimpleNamespace(reco_id=native_id, hit=True, all_results=[])
                collector.record_recognition(WALLET_NODE, (), {}, result, "cn")
            readings.append(runtime.observe_wallet(dict(coin=amount, coin_complete=True,
                recognition_status="ok", native_reco_id=native_id)))
            return amount

        interactor.get_current_coin = Mock(side_effect=get_coin)
        params = SimpleNamespace(lang="cn", item_translations={"potential_drink": {"cn": ["drink"]}},
            item_rois=[dict(name_roi=[slot, 0, 1, 1], price_roi=[slot, 1, 1, 1],
                            item_roi=[slot, 0, 1, 2]) for slot in range(8)])
        data = ShopContext(params, current_coin=410, refresh_remaining=1,
            refresh_remaining_observed=1, refresh_remaining_basis="ocr", refresh_cost=80,
            items=[Item(grid_num=slot, internal_name="potential_drink", quantity=1,
                        price=100, display_name="drink") for slot in range(1, 9)])
        handler = ObservedShopHandler(interactor, data, collector)
        handler._get_item_name_and_quantity = Mock(return_value=("potential_drink", 1))
        handler._get_item_price = Mock(return_value=100)
        return handler, collector, readings

    def refresh_frames(self, context, observe, *, timeout):
        # Replace only framework frame delivery; run the real refresh observer.
        for frame in ("after-1", "after-2"):
            result = observe(frame)
            if result is not None:
                return result
        return None

    def test_same_shelf_reread_preserves_purchased_and_partially_read_items(self):
        prepared_states = []
        for recording in (True, False):
            with self.subTest(recording=recording):
                handler, collector, _ = self.make_handler(recording=recording)
                self.assertTrue(handler.read_goods_page_info(enter_shopping=False))
                bought, incomplete = handler.data.items[:2]
                state = replace(sim.initial_843(), phase="SHOP", coins=410,
                                stock=((7, 0, 0), (0, 0, 0)))
                bought.bought = True
                incomplete.price, incomplete.missing, incomplete.checked = -1, ["price"], True
                handler.interactor.coins["before"] = 310

                self.assertTrue(handler.read_goods_page_info(enter_shopping=False))
                scan = collector.stock_scan_id
                self.assertTrue(handler.read_goods_page_info(enter_shopping=False))

                self.assertIs(handler.data.items[0], bought)
                self.assertIs(handler.data.items[1], incomplete)
                self.assertTrue(bought.bought)
                self.assertTrue(incomplete.checked)
                self.assertEqual((incomplete.price, incomplete.missing), (100, []))
                handler._get_item_name_and_quantity.assert_not_called()
                handler._get_item_price.assert_called_once()
                self.assertEqual(collector.stock_scan_id, scan)
                items = [dict(slot=item.grid_num, **snapshot(item)) for item in handler.data.items]
                candidates = [sim.Action("BUY", index=item.grid_num, price=item.price)
                              for item in handler.data.items if not item.bought]
                prepared = prepare_actions(state, candidates, dict(kind="shop", visible_items=items,
                    stock_read_complete=True, enhancement_cost=sim.PRICES[0]))
                self.assertEqual(prepared.state.stock, ((7, 0, 0), (0, 0, 0)))
                self.assertEqual(set(prepared.action_slots.values()), {2})
                prepared_states.append(prepared.state)
                stocks = [call for call in collector.emit.call_args_list if call.args[0] == "shop_observed"]
                self.assertEqual(len(stocks), 2 if recording else 0)
        self.assertEqual(*prepared_states)

    def test_last_refresh_accepts_identical_goods_and_links_its_final_frame_wallet(self):
        handler, collector, readings = self.make_handler()
        self.assertTrue(handler.read_goods_page_info(enter_shopping=False))
        before_items = snapshot(handler.data.items)
        before_generation = handler.stock_generation_id
        collector.emit.reset_mock()

        with patch.object(Interactor, "wait_for", side_effect=self.refresh_frames) as wait:
            self.assertTrue(handler.refresh())

        wait.assert_called_once()
        handler.interactor.get_refresh_remaining.assert_not_called()
        handler.interactor.context.run_task.assert_called_once_with("星塔_节点_商店_点击刷新_agent")
        self.assertEqual(snapshot(handler.data.items), before_items)
        self.assertEqual((handler.data.refresh_remaining, handler.data.refresh_remaining_observed), (0, -1))
        self.assertNotEqual(handler.stock_generation_id, before_generation)
        self.assertIsNone(handler.pending_refresh)
        actions = [call for call in collector.emit.call_args_list
                   if call.args[0] in {"shop_refresh_result_attempt", "shop_observed", "shop_refresh_result"}]
        self.assertEqual([call.args[0] for call in actions],
                         ["shop_refresh_result_attempt", "shop_observed", "shop_refresh_result"])
        attempt, stock, result = (call.kwargs for call in actions)
        self.assertTrue(result["success"])
        self.assertFalse(result["shelf_changed"])
        self.assertTrue(result["debit_confirmed"])
        self.assertTrue(result["refresh_remaining_decreased"])
        self.assertEqual((result["coin_before"], result["coin_after"]), (410, 330))
        self.assertEqual(result["next_stock_scan_id"], stock["stock_scan_id"])
        self.assertEqual(result["stock_read_id"], stock["read_id"])
        self.assertEqual(result["wallet_observation_id"], readings[-1]["observation_id"])
        self.assertEqual(result["operation_id"], attempt["operation_id"])
        self.assertEqual({reading["operation_id"] for reading in readings[1:]}, {result["operation_id"]})
        self.assertIsNone(collector.session.current_action_links["operation_id"])

    def test_incomplete_refresh_record_does_not_change_success_or_restore_bought_flags(self):
        handler, collector, _ = self.make_handler(final_wallet_linked=False)
        self.assertTrue(handler.read_goods_page_info(enter_shopping=False))
        old_item = handler.data.items[0]
        old_item.bought, old_item.checked = True, True
        with patch.object(Interactor, "wait_for", side_effect=self.refresh_frames):
            self.assertTrue(handler.refresh())
        self.assertIsNot(handler.data.items[0], old_item)
        self.assertFalse(handler.data.items[0].bought)
        self.assertFalse(handler.data.items[0].checked)
        self.assertIsNone(handler.pending_refresh)
        result = next(call.kwargs for call in reversed(collector.emit.call_args_list)
                      if call.args[0] == "shop_refresh_result")
        self.assertIsNone(result["success"])
        self.assertIsNone(result["coin_after"])
        self.assertIsNone(result["wallet_observation_id"])
        self.assertTrue(result["Maa_return_value"])
        self.assertEqual(handler.data.refresh_remaining, 0)


if __name__ == "__main__":
    unittest.main()
