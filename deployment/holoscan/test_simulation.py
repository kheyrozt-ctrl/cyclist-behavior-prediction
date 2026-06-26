from __future__ import annotations

import unittest

import numpy as np

from deployment.holoscan.simulation import synthetic_frame, synthetic_keypoints


class SimulationTest(unittest.TestCase):
    def test_keypoints_are_deterministic_and_normalized(self):
        first = synthetic_keypoints(15, 30)
        second = synthetic_keypoints(15, 30)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 66)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in first))

    def test_frame_shape_and_type(self):
        frame = synthetic_frame(3, 320, 240, 30)
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
