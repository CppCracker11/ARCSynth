import unittest
import numpy as np
from search_astar import solve_astar, heuristic
from search_bfs import replay_program
from dsl import rotate_180, recolor


class TestSearchAStar(unittest.TestCase):
    def test_pose_and_paint_heuristic(self):
        g1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        g2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        g3 = np.array([[1, 0], [3, 9]], dtype=np.int32)
        g4 = np.array([[1, 2, 3]], dtype=np.int32)

        self.assertEqual(heuristic((g1,), (g2,)), 0)
        self.assertGreater(heuristic((g1,), (g3,)), 0)
        self.assertGreater(heuristic((g1,), (g4,)), 0)

    def test_solve_astar(self):
        in1 = np.array([[1, 2], [3, 0]], dtype=np.int32)
        out1 = recolor(rotate_180(in1), 1, 4)
        train_pairs = [{"input": in1, "output": out1}]

        res = solve_astar(train_pairs, max_depth=2)
        self.assertTrue(res["solved"])
        self.assertIsNotNone(res["program"])
        self.assertGreaterEqual(res["nodes_expanded"], 1)

        pred = replay_program(in1, res["program"])
        self.assertTrue(np.array_equal(pred, out1))

    def test_initial_goal_solve(self):
        g = np.array([[1, 2], [3, 4]], dtype=np.int32)
        train_pairs = [{"input": g, "output": g}]
        res = solve_astar(train_pairs, max_depth=2)
        self.assertTrue(res["solved"])
        self.assertEqual(res["program"], [])
        self.assertEqual(res["nodes_expanded"], 0)


if __name__ == "__main__":
    unittest.main()
