"""Optional recording must not remove a real refresh's session identity."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.sampling_data import SamplingCollector
from custom.climb_tower_neural.session import RunSession
from custom.climb_tower_neural.shop_ui.handler import ShopHandler
from custom.climb_tower_neural.tower_sampling_collector import ObservedPotentialHandler, ObservedShopHandler


class TowerNeuralObservedOperationsTests(unittest.TestCase):
    def setUp(self):
        directory = TowerTempDirectory()
        self.addCleanup(directory.cleanup)
        self.session = RunSession()
        self.session.begin_run()
        self.collector = SamplingCollector(Path(directory.name), Mock(), session=self.session)
        self.collector.enabled = False
        self.collector.capture = Mock(wraps=self.collector.capture)
        self.collector.emit = Mock(wraps=self.collector.emit)

    def observe_refreshed_wallet(self):
        return self.session.observe_wallet(dict(coin=290, coin_complete=True, recognition_status="ok"))

    def test_potential_refresh_keeps_its_own_scope_and_returns_the_original_result(self):
        result = object()
        readings = []

        def refresh():
            readings.append(self.observe_refreshed_wallet())
            return result

        delegate = SimpleNamespace(data=SimpleNamespace(params=SimpleNamespace(potential_source="drink")),
                                   refresh=Mock(side_effect=refresh))
        handler = ObservedPotentialHandler(delegate, self.collector)
        with self.session.action_scope(operation_id="purchase"):
            self.assertIs(handler.refresh(), result)
            self.assertEqual(self.session.current_action_links["operation_id"], "purchase")
        delegate.refresh.assert_called_once_with()
        self.assertTrue(readings[0]["operation_id"])
        self.assertNotEqual(readings[0]["operation_id"], "purchase")
        self.assertIsNone(self.session.current_action_links["operation_id"])
        self.collector.capture.assert_not_called()
        self.collector.emit.assert_not_called()

    def test_shop_refresh_keeps_its_own_scope_without_inserting_recording_frames(self):
        readings = []
        interactor = Mock()
        handler = ObservedShopHandler(interactor, SimpleNamespace(params=SimpleNamespace()), self.collector)

        def refresh(original_handler):
            self.assertIs(original_handler, handler)
            readings.append(self.observe_refreshed_wallet())
            return False

        with patch.object(ShopHandler, "refresh", autospec=True, side_effect=refresh) as original:
            with self.session.action_scope(operation_id="outer-operation"):
                self.assertIs(handler.refresh(), False)
                self.assertEqual(self.session.current_action_links["operation_id"], "outer-operation")
        original.assert_called_once_with(handler)
        self.assertTrue(readings[0]["operation_id"])
        self.assertNotEqual(readings[0]["operation_id"], "outer-operation")
        self.assertIsNone(self.session.current_action_links["operation_id"])
        self.collector.capture.assert_not_called()
        self.collector.emit.assert_not_called()
        self.assertEqual(interactor.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
