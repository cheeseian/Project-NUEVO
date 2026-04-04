from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


package_root = Path(__file__).resolve().parents[1]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from robot.path_planner import PurePursuitPlanner


class PurePursuitPlannerTests(unittest.TestCase):
    def test_lookahead_uses_ordered_remaining_path(self) -> None:
        planner = PurePursuitPlanner(lookahead_dist=50.0)
        path = np.array([
            [0.0, 500.0],
            [25.0, 500.0],
            [50.0, 500.0],
            [75.0, 500.0],
        ])

        lookahead = planner._lookahead_point(0.0, 480.0, path)

        self.assertEqual(tuple(lookahead), (50.0, 500.0))

    def test_compute_velocity_is_straight_on_for_aligned_target(self) -> None:
        planner = PurePursuitPlanner(lookahead_dist=100.0, max_angular=1.0)

        linear, angular = planner.compute_velocity(
            pose=(0.0, 0.0, 0.0),
            waypoints=np.array([[100.0, 0.0]]),
            max_linear=100.0,
        )

        self.assertAlmostEqual(linear, 100.0, places=3)
        self.assertAlmostEqual(angular, 0.0, places=3)

    def test_compute_velocity_slows_and_turns_for_corner(self) -> None:
        planner = PurePursuitPlanner(lookahead_dist=100.0, max_angular=1.0)

        linear, angular = planner.compute_velocity(
            pose=(0.0, 0.0, 0.0),
            waypoints=np.array([[0.0, 100.0]]),
            max_linear=100.0,
        )

        self.assertAlmostEqual(linear, 0.0, places=3)
        self.assertGreater(angular, 0.0)
        self.assertLessEqual(angular, 1.0)


if __name__ == "__main__":
    unittest.main()
