"""多张纪录徽标的评分须覆盖全部有效候选，而非只看第一张。"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "cataclysm_survivor_record", ROOT / "agent/custom/action/cataclysm_survivor_record.py",
)
record = importlib.util.module_from_spec(spec)
with patch("maa.agent.agent_server.AgentServer.custom_action", side_effect=lambda _name: lambda cls: cls):
    spec.loader.exec_module(record)


class RecordCandidateTests(unittest.TestCase):
    def test_scores_all_filtered_candidates_and_clicks_brighter_later_card(self):
        image = np.full((400, 640, 3), 64, dtype=np.uint8)
        image[29:279, 330:560] = (255, 180, 80)
        controller = Mock()
        controller.post_click.return_value.done = True
        controller.post_click.return_value.succeeded = True
        candidates = [
            SimpleNamespace(box=(40, 100, 11, 14), score=0.99),
            SimpleNamespace(box=(350, 100, 11, 14), score=0.9),
        ]
        detail = SimpleNamespace(hit=True, filtered_results=candidates, all_results=candidates)
        context = SimpleNamespace(
            tasker=SimpleNamespace(controller=controller, stopping=False),
            run_recognition=Mock(return_value=detail),
        )
        with patch.object(record, "_grab", return_value=image):
            self.assertTrue(record.CataclysmSurvivorPickRecord().run(context, SimpleNamespace(node_name="record")))
        controller.post_click.assert_called_once_with(445, 155)


if __name__ == "__main__":
    unittest.main()
