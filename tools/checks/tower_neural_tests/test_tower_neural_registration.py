import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "agent/custom/action"))
from climb_tower_neural import register_lazy_callbacks


class TowerNeuralRegistrationTests(unittest.TestCase):
    def test_registration_error_is_ignored_when_runtime_is_unavailable(self):
        with (
            patch("climb_tower_neural._register_lazy_callbacks", side_effect=RuntimeError),
            patch("climb_tower_neural._runtime_is_available", return_value=False),
        ):
            register_lazy_callbacks()

    def test_registration_error_propagates_when_runtime_is_available(self):
        with (
            patch("climb_tower_neural._register_lazy_callbacks", side_effect=RuntimeError),
            patch("climb_tower_neural._runtime_is_available", return_value=True),
        ):
            with self.assertRaises(RuntimeError):
                register_lazy_callbacks()


if __name__ == "__main__":
    unittest.main()
