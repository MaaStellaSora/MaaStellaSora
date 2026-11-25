"""Replay a missed bag-tab click without repeating it on the collection list."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import bag_preset_reader as bag
from custom.climb_tower_neural.note_bag import SKILLS_READY
from custom.climb_tower_neural.note_ocr import TEXT_NODE


class TowerNeuralBagTabTests(unittest.TestCase):
    def test_only_retry_on_confirmed_skills_tab(self):
        initial, loading, skills, collection = range(4)
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        for index, frame in enumerate(frames):
            frame[0, 0, 0] = index
        # Collection scrolled between two headers, with its All Cards knob visible.
        frames[collection][649:677, 86:115] = 255
        frames[collection][82:95, 948:953] = (200, 200, 0)
        boxes = {initial: [30, 120, 60, 20], skills: [80, 130, 60, 20]}
        cases = (
            ([collection], [], True),
            ([initial, loading, collection], [(60, 130)], True),
            ([initial, loading, skills, collection], [(60, 130), (110, 140)], True),
            ([initial, loading, loading], [(60, 130)], False),
            ([initial, skills, skills], [(60, 130), (110, 140)], False),
        )
        for sequence, clicks, opened in cases:
            with self.subTest(sequence=sequence):
                clock = 0
                incoming = iter(sequence[1:])
                context = Mock()
                context.tasker.stopping = False
                reader = bag._Reader(context, "jp", None)
                reader.headers = Mock(return_value=[])
                reader.click = Mock()
                reader._ensure_control = Mock(side_effect=lambda image, name: image)
                reader._confirm_display = Mock()

                def wait_for_frame(context, observe, *, timeout):
                    nonlocal clock
                    for index in incoming:
                        clock += 2
                        result = observe(frames[index])
                        if result is not None:
                            return result
                    return None

                def recognize(node, image, *args):
                    index = int(image[0, 0, 0])
                    if node == SKILLS_READY:
                        return SimpleNamespace(hit=index == skills)
                    self.assertEqual(node, TEXT_NODE)
                    row = SimpleNamespace(score=1., box=boxes[index])
                    return SimpleNamespace(hit=True, filtered_results=[row])

                context.run_recognition.side_effect = recognize
                with (patch.object(bag.Interactor, "capture", return_value=frames[sequence[0]]),
                      patch.object(bag.Interactor, "wait_for", wait_for_frame),
                      patch.object(bag, "monotonic", lambda: clock)):
                    if opened:
                        self.assertIs(reader.prepare(), frames[collection])
                    else:
                        with self.assertRaisesRegex(ValueError, "背包潜能收集页未确认"):
                            reader.prepare()
                self.assertEqual([entry.args for entry in reader.click.call_args_list], clicks)
