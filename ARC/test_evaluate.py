import csv
import os
import unittest
import numpy as np
from grids import save_tasks
from dsl import rotate_90
from evaluate import evaluate_benchmark


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.tmp_tasks = "tmp_eval_tasks.json"
        self.tmp_csv = "tmp_eval_results.csv"

        in1 = np.array([[1, 2], [3, 0]], dtype=np.int32)
        out1 = rotate_90(in1)
        tasks = {
            "test_task_1": {
                "tier": "easy",
                "train": [{"input": in1, "output": out1}],
                "test": [{"input": in1, "output": out1}],
            }
        }
        save_tasks(tasks, self.tmp_tasks)

    def tearDown(self):
        for f in [self.tmp_tasks, self.tmp_csv]:
            if os.path.exists(f):
                os.remove(f)

    def test_evaluate_benchmark_run(self):
        records = evaluate_benchmark(tasks_path=self.tmp_tasks, csv_out=self.tmp_csv)
        self.assertTrue(os.path.exists(self.tmp_csv))
        self.assertEqual(len(records), 4)  # 1 task * 4 methods

        expected_methods = {"BFS", "A*", "Weighted A* (w=2.0)", "Q-Learning RL"}
        found_methods = {r["method"] for r in records}
        self.assertEqual(found_methods, expected_methods)

        with open(self.tmp_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 4)
            for r in rows:
                self.assertIn("task_id", r)
                self.assertIn("tier", r)
                self.assertIn("method", r)
                self.assertIn("train_solved", r)
                self.assertIn("test_solved", r)
                self.assertIn("nodes_expanded", r)
                self.assertIn("time_ms", r)
                self.assertIn("program", r)


if __name__ == "__main__":
    unittest.main()
