import unittest
import numpy as np
from benchmark import create_task_specs, generate_benchmark, apply_sequence
from dsl import Action


class TestBenchmark(unittest.TestCase):
    def test_task_specs(self):
        specs = create_task_specs()
        self.assertEqual(len(specs), 24)
        tiers = [s[1] for s in specs]
        self.assertEqual(tiers.count("easy"), 8)
        self.assertEqual(tiers.count("medium"), 8)
        self.assertEqual(tiers.count("hard"), 8)

    def test_benchmark_generation(self):
        tasks = generate_benchmark(seed=123)
        self.assertEqual(len(tasks), 24)

        for task_id, task in tasks.items():
            self.assertIn("tier", task)
            self.assertIn("train", task)
            self.assertIn("test", task)
            self.assertEqual(len(task["train"]), 2)
            self.assertEqual(len(task["test"]), 1)

            # Verify non-triviality
            for pair in task["train"] + task["test"]:
                self.assertFalse(
                    np.array_equal(pair["input"], pair["output"]),
                    f"Trivial identity transformation in {task_id}",
                )


if __name__ == "__main__":
    unittest.main()
