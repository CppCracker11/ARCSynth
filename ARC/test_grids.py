import os
import unittest
import numpy as np
from grids import grid_hash, generate_random_grid, save_tasks, load_tasks, visualize_task


class TestGrids(unittest.TestCase):
    def test_grid_hash(self):
        g1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        g2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        g3 = np.array([[1, 2], [3, 5]], dtype=np.int32)
        g4 = np.array([[1, 2, 3, 4]], dtype=np.int32)

        self.assertEqual(grid_hash(g1), grid_hash(g2))
        self.assertNotEqual(grid_hash(g1), grid_hash(g3))
        self.assertNotEqual(grid_hash(g1), grid_hash(g4))

    def test_generate_random_grid(self):
        g = generate_random_grid(5, 6, colors=[1, 2, 3], seed=123)
        self.assertEqual(g.shape, (5, 6))
        self.assertEqual(g.dtype, np.int32)
        unique_colors = set(np.unique(g))
        self.assertTrue(unique_colors.issubset({0, 1, 2, 3}))

    def test_save_load_tasks(self):
        tmp_file = "tmp_test_tasks.json"
        try:
            g_in = np.array([[1, 0], [0, 2]], dtype=np.int32)
            g_out = np.array([[2, 0], [0, 1]], dtype=np.int32)
            task_dict = {
                "t1": {
                    "tier": "easy",
                    "program": ["flip_h"],
                    "train": [{"input": g_in, "output": g_out}],
                    "test": [{"input": g_in, "output": g_out}],
                }
            }
            save_tasks(task_dict, tmp_file)
            loaded = load_tasks(tmp_file)

            self.assertIn("t1", loaded)
            self.assertEqual(loaded["t1"]["tier"], "easy")
            self.assertEqual(loaded["t1"]["program"], ["flip_h"])
            self.assertTrue(np.array_equal(loaded["t1"]["train"][0]["input"], g_in))
            self.assertTrue(np.array_equal(loaded["t1"]["train"][0]["output"], g_out))
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def test_visualize_task(self):
        tmp_img = "tmp_visualize_test.png"
        try:
            g_in = np.array([[1, 2], [3, 4]], dtype=np.int32)
            g_out = np.array([[4, 3], [2, 1]], dtype=np.int32)
            task = {
                "train": [{"input": g_in, "output": g_out}],
                "test": [{"input": g_in, "output": g_out}],
            }
            visualize_task(task, save_path=tmp_img, title="Test Visualization")
            self.assertTrue(os.path.exists(tmp_img))
            self.assertGreater(os.path.getsize(tmp_img), 0)
        finally:
            if os.path.exists(tmp_img):
                os.remove(tmp_img)


if __name__ == "__main__":
    unittest.main()
