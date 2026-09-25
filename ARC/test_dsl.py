import unittest
import numpy as np
from dsl import (
    Action,
    rotate_90,
    rotate_180,
    rotate_270,
    flip_h,
    flip_v,
    transpose,
    recolor,
    recolor_by_size,
    invert_colors,
    crop_to_content,
    translate,
    tile,
    mirror_complete_h,
    mirror_complete_v,
    keep_largest_object,
    remove_object,
    count_objects_grid,
    build_action_space,
)


class TestDSL(unittest.TestCase):
    def setUp(self):
        self.g = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.int32)

    def test_geometric(self):
        self.assertTrue(np.array_equal(rotate_90(self.g), np.array([[3, 1], [4, 2]])))
        self.assertTrue(np.array_equal(rotate_180(self.g), np.array([[4, 3], [2, 1]])))
        self.assertTrue(np.array_equal(rotate_270(self.g), np.array([[2, 4], [1, 3]])))
        self.assertTrue(np.array_equal(flip_h(self.g), np.array([[2, 1], [4, 3]])))
        self.assertTrue(np.array_equal(flip_v(self.g), np.array([[3, 4], [1, 2]])))
        self.assertTrue(np.array_equal(transpose(self.g), np.array([[1, 3], [2, 4]])))

    def test_color_primitives(self):
        rec = recolor(self.g, 1, 9)
        self.assertEqual(rec[0, 0], 9)
        self.assertEqual(rec[0, 1], 2)

        inv = invert_colors(np.array([[0, 2], [1, 0]], dtype=np.int32))
        self.assertEqual(inv[0, 0], 0)
        self.assertEqual(inv[0, 1], 8)
        self.assertEqual(inv[1, 0], 9)

        multi_obj = np.array([
            [1, 0, 2, 2],
            [0, 0, 2, 2],
        ], dtype=np.int32)
        r_sz = recolor_by_size(multi_obj, 5)
        self.assertEqual(r_sz[0, 2], 5)
        self.assertEqual(r_sz[1, 3], 5)
        self.assertEqual(r_sz[0, 0], 1)

    def test_spatial_primitives(self):
        padded = np.array([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        cropped = crop_to_content(padded)
        self.assertTrue(np.array_equal(cropped, self.g))

        shifted = translate(self.g, dx=1, dy=0)
        self.assertTrue(np.array_equal(shifted, np.array([[0, 1], [0, 3]])))

        tiled = tile(self.g, 2, 1)
        self.assertEqual(tiled.shape, (4, 2))

        mh = mirror_complete_h(self.g)
        self.assertEqual(mh.shape, (2, 4))
        self.assertTrue(np.array_equal(mh[:, 2:], flip_h(self.g)))

        mv = mirror_complete_v(self.g)
        self.assertEqual(mv.shape, (4, 2))
        self.assertTrue(np.array_equal(mv[2:, :], flip_v(self.g)))

    def test_object_primitives(self):
        multi_obj = np.array([
            [1, 0, 2, 2],
            [0, 0, 2, 2],
        ], dtype=np.int32)
        lo = keep_largest_object(multi_obj)
        self.assertEqual(lo[0, 0], 0)
        self.assertEqual(lo[0, 2], 2)
        self.assertEqual(lo[1, 3], 2)

        rem = remove_object(multi_obj, 2)
        self.assertEqual(rem[0, 2], 0)
        self.assertEqual(rem[0, 0], 1)

        cnt_g = count_objects_grid(multi_obj)
        self.assertEqual(cnt_g.shape, (1, 1))
        self.assertEqual(cnt_g[0, 0], 2)

    def test_build_action_space(self):
        train_pairs = [
            {"input": np.array([[1, 0], [0, 0]]), "output": np.array([[0, 2], [0, 0]])}
        ]
        acts = build_action_space(train_pairs)
        act_names = {a.name for a in acts}
        self.assertIn("rotate_90", act_names)
        self.assertIn("recolor(1->2)", act_names)
        self.assertNotIn("recolor(3->4)", act_names)


if __name__ == "__main__":
    unittest.main()
