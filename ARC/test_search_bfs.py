import unittest
import numpy as np
from search_bfs import solve_bfs, replay_program
from dsl import rotate_90, flip_h


class TestSearchBFS(unittest.TestCase):
    def test_solve_depth_one(self):
        in1 = np.array([[1, 2], [3, 0]], dtype=np.int32)
        out1 = rotate_90(in1)
        in2 = np.array([[0, 2], [1, 3]], dtype=np.int32)
        out2 = rotate_90(in2)

        train_pairs = [{"input": in1, "output": out1}, {"input": in2, "output": out2}]
        res = solve_bfs(train_pairs, max_depth=1)

        self.assertTrue(res["solved"])
        self.assertEqual(len(res["program"]), 1)
        self.assertEqual(res["program"][0].name, "rotate_90")
        self.assertGreaterEqual(res["nodes_expanded"], 1)

    def test_solve_depth_two_and_replay(self):
        in1 = np.array([[1, 2], [3, 0]], dtype=np.int32)
        out1 = flip_h(rotate_90(in1))
        train_pairs = [{"input": in1, "output": out1}]

        res = solve_bfs(train_pairs, max_depth=2)
        self.assertTrue(res["solved"])

        test_in = np.array([[2, 1], [0, 3]], dtype=np.int32)
        test_out = flip_h(rotate_90(test_in))
        pred = replay_program(test_in, res["program"])
        self.assertTrue(np.array_equal(pred, test_out))

    def test_depth_limit_failure(self):
        # Impossible to solve at depth 0
        in1 = np.array([[1, 2], [3, 0]], dtype=np.int32)
        out1 = rotate_90(in1)
        train_pairs = [{"input": in1, "output": out1}]

        res = solve_bfs(train_pairs, max_depth=0)
        self.assertFalse(res["solved"])
        self.assertIsNone(res["program"])


if __name__ == "__main__":
    unittest.main()
