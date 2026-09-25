import unittest
import numpy as np
from search_weighted import solve_weighted_astar
from search_bfs import replay_program
from dsl import rotate_90, crop_to_content


class TestSearchWeighted(unittest.TestCase):
    def test_solve_weighted_astar(self):
        in1 = np.array([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        out1 = rotate_90(crop_to_content(in1))
        train_pairs = [{"input": in1, "output": out1}]

        res = solve_weighted_astar(train_pairs, weight=2.0, max_depth=2)
        self.assertTrue(res["solved"])
        self.assertIsNotNone(res["program"])

        pred = replay_program(in1, res["program"])
        self.assertTrue(np.array_equal(pred, out1))

    def test_initial_goal_solve(self):
        g = np.array([[2, 3], [4, 5]], dtype=np.int32)
        train_pairs = [{"input": g, "output": g}]
        res = solve_weighted_astar(train_pairs, weight=2.0, max_depth=2)
        self.assertTrue(res["solved"])
        self.assertEqual(res["program"], [])
        self.assertEqual(res["nodes_expanded"], 0)


if __name__ == "__main__":
    unittest.main()
