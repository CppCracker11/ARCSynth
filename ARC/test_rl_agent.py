import unittest
import numpy as np
from rl_agent import QLearningAgent, extract_features, action_key
from dsl import Action, rotate_90, build_action_space


class TestRLAgent(unittest.TestCase):
    def test_extract_features(self):
        g1 = np.array([[1, 2], [3, 0]], dtype=np.int32)
        t1 = np.array([[0, 1], [3, 2]], dtype=np.int32)
        feat = extract_features((g1,), (t1,), step=0)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(len(feat), 6)
        for val in feat:
            self.assertIsInstance(val, (int, np.integer))

    def test_action_key(self):
        act1 = Action("rotate_90", rotate_90)
        act2 = Action("recolor(1->2)", lambda g: g)
        self.assertEqual(action_key(act1), "rotate_90")
        self.assertEqual(action_key(act2), "recolor")

    def test_train_and_solve_pipeline(self):
        in1 = np.array([[1, 2], [3, 0]], dtype=np.int32)
        out1 = rotate_90(in1)
        mini_tasks = {
            "mini_01": {
                "tier": "easy",
                "train": [{"input": in1, "output": out1}],
                "test": [{"input": in1, "output": out1}],
            }
        }
        agent = QLearningAgent(alpha=0.5, gamma=0.8)
        self.assertEqual(len(agent.q_table), 0)

        agent.train(mini_tasks, episodes=20, max_steps=2)
        self.assertGreater(len(agent.q_table), 0)

        res = agent.solve(mini_tasks["mini_01"]["train"], max_steps=2)
        self.assertIn("solved", res)
        self.assertIn("program", res)
        self.assertIn("nodes_expanded", res)
        self.assertIn("time_taken", res)


if __name__ == "__main__":
    unittest.main()
