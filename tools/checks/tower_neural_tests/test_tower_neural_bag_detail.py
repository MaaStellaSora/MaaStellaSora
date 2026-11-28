"""A bag detail title is evidence only for the card selected on the same page."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import bag_preset_reader as bag
from custom.climb_tower_neural.ui_flow import READ_TIMEOUT


class BagDetailTests(unittest.TestCase):
    def setUp(self):
        self.reader = bag._Reader(SimpleNamespace(tasker=SimpleNamespace(stopping=False)), "cn", None)
        self.card = bag._Card("core", 0, levels={0}, targets={1}, rarities={"core"})
        self.card.observations.append({"box": [358, 137, 118, 148]})
        self.reader.members = [{"position": 90, "cards": [self.card]}]
        self.image = self.reader._record_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.reader.click = Mock()

    def test_confirmed_list_name_needs_no_detail(self):
        self.reader.feed = Mock(return_value=[])
        with patch.object(bag, "read_text") as read:
            self.assertIs(self.reader.read_page(self.image), self.image)
        read.assert_not_called()
        self.reader.click.assert_not_called()

    def replay(self, observations, *, already_selected=False, poll_sizes=None):
        frames = [self.image.copy() for _ in observations]
        index = {id(image): i for i, image in enumerate(frames)}
        state = [(already_selected, 0, "旧标题"), *observations]
        remaining = iter(frames)
        sizes = iter(poll_sizes or [len(frames)])

        def view(frame):
            return state[index[id(frame)] + 1] if frame is not self.image else state[0]

        def feed(_image):
            self.card.crop_frame = self.reader.frame
            self.reader.missing = ["member@90:card1:name"]
            return self.reader.missing

        def poll(context, observe, *, timeout):
            if timeout == READ_TIMEOUT:
                return None
            for _ in range(next(sizes)):
                frame = next(remaining)
                result = observe(frame)
                if result is not None:
                    return result
            return None

        self.reader.feed = feed
        with (
            patch.object(bag.Interactor, "wait_for", side_effect=poll),
            patch.object(bag, "_selected_card", side_effect=lambda image, box:
                         view(image)[0] is True or view(image)[0] == tuple(box)),
            patch.object(bag, "_content_shift", side_effect=lambda anchor, frame: view(frame)[1]),
            patch.object(bag, "read_text", side_effect=lambda ctx, frame, roi: [
                dict(text=view(frame)[2], score=.99, box=[1064, 333, 93, 27])]),
        ):
            return self.reader.read_page(self.image), frames

    def test_waits_for_selected_card_and_new_title(self):
        frame, frames = self.replay([(False, 0, "新标题"), (True, 0, "旧标题"),
                                     (True, 0, "新标题"), (True, 0, "新标题")])
        self.reader.click.assert_called_once_with(417, 211)
        self.assertIs(frame, frames[-1])
        self.assertEqual(self.card.name(), ("新标题", True))

    def test_already_selected_card_needs_no_click(self):
        self.replay([(True, 0, "旧标题")] * 2, already_selected=True)
        self.reader.click.assert_not_called()
        self.assertEqual(self.card.name(), ("旧标题", True))

    def test_same_title_on_newly_selected_card_is_confirmed(self):
        self.card.readings.append(dict(text="旧标题", source="page", score=.9))
        frame, frames = self.replay([(False, 0, "旧标题"), (True, 0, "旧标题"),
                                     (True, 0, "旧标题")])
        self.reader.click.assert_called_once_with(417, 211)
        self.assertIs(frame, frames[-1])
        self.assertEqual(self.card.name(), ("旧标题", True))

    def test_stale_title_waits_for_new_title_without_matching_list_evidence(self):
        frame, frames = self.replay([(True, 0, "旧标题")] * 2
                                   + [(True, 0, "新标题")] * 2)
        self.assertIs(frame, frames[-1])
        self.assertEqual(self.card.name(), ("新标题", True))

    def test_stale_title_alone_remains_unconfirmed(self):
        with self.assertRaisesRegex(ValueError, "card1:name"):
            self.replay([(True, 0, "旧标题")] * 2)
        self.assertIsNone(self.card.name_evidence)

    def test_empty_list_name_uses_confirmed_different_card_as_bridge(self):
        bridge = bag._Card("core", 1, name_evidence={"text": "过渡标题"})
        self.reader.members[0]["cards"].append(bridge)
        self.assertEqual(self.card.readings, [])
        target_box = (358, 137, 118, 148)
        bridge_box = (475, 137, 118, 148)
        frame, frames = self.replay([(target_box, 0, "旧标题")] * 2
                                   + [(bridge_box, 0, "过渡标题")] * 2
                                   + [(target_box, 0, "旧标题")] * 2,
                                   poll_sizes=[2, 2, 2])
        self.assertIs(frame, frames[-1])
        self.assertEqual(self.card.name(), ("旧标题", True))
        self.assertEqual(self.reader.click.call_args_list,
                         [unittest.mock.call(417, 211), unittest.mock.call(534, 211),
                          unittest.mock.call(417, 211)])
        self.assertEqual(self.card.name_evidence["card_box"], list(target_box))

    def test_unreadable_previous_title_does_not_click_or_accept_arbitrary_title(self):
        with (patch.object(bag, "_selected_card", return_value=False),
              patch.object(bag, "read_text", return_value=[]),
              patch.object(bag, "wait_for") as wait):
            self.assertIsNone(self.reader._read_detail_name(self.image, self.card))
        self.reader.click.assert_not_called()
        wait.assert_not_called()

    def test_unselected_card_and_scrolled_page_remain_unconfirmed(self):
        with self.assertRaisesRegex(ValueError, "card1:name"):
            self.replay([(False, 0, "旧标题")] * 2 + [(True, -160, "新标题")] * 2)
        self.assertIsNone(self.card.name_evidence)
