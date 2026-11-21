import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import image_ops


class TowerNeuralImageOpsTests(unittest.TestCase):
    def test_screenshot_colors_use_bgr_and_half_range_hue(self):
        image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255],
                           [60, 177, 221], [250, 250, 250]]], dtype=np.uint8)
        np.testing.assert_array_equal(image_ops.bgr_to_gray(image), [[29, 150, 76, 177, 250]])
        hsv = image_ops.bgr_to_hsv(image)
        np.testing.assert_array_equal(hsv, [[[120, 255, 255], [60, 255, 255], [0, 255, 255],
                                              [22, 186, 221], [0, 0, 250]]])
        np.testing.assert_array_equal(image_ops.in_range(hsv, (0, 0, 210), (179, 75, 255)),
                                      [[False, False, False, False, True]])

    def test_cubic_resize_preserves_uint8_crop_interpolation(self):
        image = np.array([[0, 100], [150, 255]], dtype=np.uint8)
        expected = [[0, 7, 61, 94], [23, 57, 112, 146], [105, 140, 196, 231], [155, 190, 247, 255]]
        result = image_ops.resize_cubic(image, scale=2)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_cubic_resize_roi_matches_full_frame_at_fractional_scales(self):
        image = np.random.default_rng(23).integers(0, 256, (41, 53, 3), dtype=np.uint8)
        for scale in (0.55, 0.9, 1.0, 1.1, 1.95):
            reference = image_ops.resize_cubic(image, scale=scale)
            height, width = reference.shape[:2]
            for x, y in ((0, 0), (13, 7), (width - 6, height - 5)):
                with self.subTest(scale=scale, x=x, y=y):
                    result = image_ops.resize_cubic_for_roi(image, (x, y, 6, 5), scale=scale)
                    self.assertEqual(result.shape, reference.shape)
                    np.testing.assert_array_equal(result[y:y + 5, x:x + 6], reference[y:y + 5, x:x + 6])

    def test_connected_components_include_diagonal_pixels(self):
        mask = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=bool)
        labels, components = image_ops.connected_components(mask)
        self.assertEqual(len(components), 2)
        first, second = components
        self.assertEqual((first.left, first.top, first.width, first.height, first.area), (0, 0, 3, 2, 3))
        self.assertEqual((first.center_x, first.center_y), (1, 1 / 3))
        np.testing.assert_array_equal(labels, [[first.label, 0, first.label, 0],
                                                [0, first.label, 0, 0], [0, 0, 0, 0], [0, 0, 0, second.label]])

    def test_vertical_match_ignores_brightness_changes(self):
        image = np.array([[10, 70, 40], [70, 40, 50], [10, 40, 70], [40, 80, 20],
                          [80, 40, 10], [10, 60, 40]], dtype=np.uint8)
        score, y = image_ops.match_vertical_strip(image, image[2:5] + 35)
        self.assertEqual(y, 2)
        self.assertAlmostEqual(score, 1.0)
