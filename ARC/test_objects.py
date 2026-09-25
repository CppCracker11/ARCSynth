import unittest
import numpy as np
from objects import find_objects, largest_object, count_objects


class TestObjects(unittest.TestCase):
    def test_find_objects_multi_component(self):
        g = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 2],
            [0, 0, 0, 2, 2],
            [3, 3, 0, 0, 0],
        ], dtype=np.int32)
        objs = find_objects(g, bg=0)
        self.assertEqual(len(objs), 3)

        colors = {o["color"] for o in objs}
        self.assertEqual(colors, {1, 2, 3})

        sizes = sorted(o["size"] for o in objs)
        self.assertEqual(sizes, [2, 3, 3])

    def test_largest_object(self):
        g = np.array([
            [1, 0, 2, 2],
            [0, 0, 2, 2],
            [3, 3, 3, 0],
        ], dtype=np.int32)
        lo = largest_object(g, bg=0)
        self.assertIsNotNone(lo)
        self.assertEqual(lo["color"], 2)
        self.assertEqual(lo["size"], 4)
        self.assertEqual(lo["bbox"], (0, 2, 1, 3))

    def test_count_objects(self):
        g = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=np.int32)
        self.assertEqual(count_objects(g, bg=0), 5)

    def test_empty_grid(self):
        g = np.zeros((4, 4), dtype=np.int32)
        self.assertEqual(find_objects(g, bg=0), [])
        self.assertIsNone(largest_object(g, bg=0))
        self.assertEqual(count_objects(g, bg=0), 0)


if __name__ == "__main__":
    unittest.main()
