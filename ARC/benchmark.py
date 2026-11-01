import numpy as np
try:  # Package import
    from .grids import generate_random_grid, save_tasks
    from .dsl import (
        Action,
        rotate_90,
        rotate_180,
        rotate_270,
        flip_h,
        flip_v,
        recolor,
        recolor_by_size,
        crop_to_content,
        keep_largest_object,
        mirror_complete_h,
        mirror_complete_v,
        build_action_space,
    )
except ImportError:  # Direct script/test import
    from grids import generate_random_grid, save_tasks
    from dsl import (
        Action,
        rotate_90,
        rotate_180,
        rotate_270,
        flip_h,
        flip_v,
        recolor,
        recolor_by_size,
        crop_to_content,
        keep_largest_object,
        mirror_complete_h,
        mirror_complete_v,
        build_action_space,
    )


def apply_sequence(grid, actions):
    cur = grid.copy()
    for act in actions:
        cur = act(cur)
    return cur


def make_grid_for_primitive(prim_type, rng):
    if prim_type == "crop":
        g = np.zeros((6, 6), dtype=np.int32)
        inner = generate_random_grid(3, 3, colors=[1, 2, 3], num_shapes=2, rng=rng)
        r0 = int(rng.integers(1, 3))
        c0 = int(rng.integers(1, 3))
        g[r0 : r0 + 3, c0 : c0 + 3] = inner
        if np.all(g == 0):
            g[1:4, 1:4] = 1
        return g
    elif prim_type == "object":
        g = np.zeros((5, 5), dtype=np.int32)
        g[0:2, 0:2] = 1
        g[3, 3] = 2
        g[3, 4] = 2
        g[4, 3] = 2
        return g
    else:
        g = generate_random_grid(4, 4, colors=[1, 2, 3], num_shapes=3, rng=rng)
        if np.all(g == 0):
            g[1:3, 1:3] = 1
        return g


def create_task_specs():
    specs = [
        # Easy (1 primitive)
        ("easy_01", "easy", [Action("rotate_90", rotate_90)], "geom"),
        ("easy_02", "easy", [Action("flip_h", flip_h)], "geom"),
        ("easy_03", "easy", [Action("flip_v", flip_v)], "geom"),
        ("easy_04", "easy", [Action("recolor(1->2)", lambda g: recolor(g, 1, 2))], "color"),
        ("easy_05", "easy", [Action("recolor_by_size(4)", lambda g: recolor_by_size(g, 4))], "object"),
        ("easy_06", "easy", [Action("crop_to_content", crop_to_content)], "crop"),
        ("easy_07", "easy", [Action("keep_largest_object", keep_largest_object)], "object"),
        ("easy_08", "easy", [Action("mirror_complete_h", mirror_complete_h)], "geom"),

        # Medium (2 primitives)
        ("med_01", "medium", [Action("rotate_90", rotate_90), Action("flip_h", flip_h)], "geom"),
        ("med_02", "medium", [Action("recolor(1->3)", lambda g: recolor(g, 1, 3)), Action("rotate_180", rotate_180)], "color"),
        ("med_03", "medium", [Action("crop_to_content", crop_to_content), Action("recolor(2->4)", lambda g: recolor(g, 2, 4))], "crop"),
        ("med_04", "medium", [Action("flip_v", flip_v), Action("recolor_by_size(3)", lambda g: recolor_by_size(g, 3))], "object"),
        ("med_05", "medium", [Action("keep_largest_object", keep_largest_object), Action("rotate_90", rotate_90)], "object"),
        ("med_06", "medium", [Action("recolor(1->2)", lambda g: recolor(g, 1, 2)), Action("flip_h", flip_h)], "color"),
        ("med_07", "medium", [Action("crop_to_content", crop_to_content), Action("flip_v", flip_v)], "crop"),
        ("med_08", "medium", [Action("rotate_270", rotate_270), Action("recolor(2->1)", lambda g: recolor(g, 2, 1))], "color"),

        # Hard (3 primitives)
        ("hard_01", "hard", [Action("crop_to_content", crop_to_content), Action("recolor(1->3)", lambda g: recolor(g, 1, 3)), Action("rotate_90", rotate_90)], "crop"),
        ("hard_02", "hard", [Action("keep_largest_object", keep_largest_object), Action("recolor(2->4)", lambda g: recolor(g, 2, 4)), Action("flip_h", flip_h)], "object"),
        ("hard_03", "hard", [Action("rotate_90", rotate_90), Action("flip_v", flip_v), Action("recolor(1->2)", lambda g: recolor(g, 1, 2))], "geom"),
        ("hard_04", "hard", [Action("recolor(1->3)", lambda g: recolor(g, 1, 3)), Action("rotate_180", rotate_180), Action("flip_h", flip_h)], "color"),
        ("hard_05", "hard", [Action("crop_to_content", crop_to_content), Action("rotate_90", rotate_90), Action("flip_v", flip_v)], "crop"),
        ("hard_06", "hard", [Action("keep_largest_object", keep_largest_object), Action("flip_v", flip_v), Action("recolor(1->4)", lambda g: recolor(g, 1, 4))], "object"),
        ("hard_07", "hard", [Action("flip_h", flip_h), Action("rotate_90", rotate_90), Action("recolor(2->3)", lambda g: recolor(g, 2, 3))], "geom"),
        ("hard_08", "hard", [Action("crop_to_content", crop_to_content), Action("recolor_by_size(2)", lambda g: recolor_by_size(g, 2)), Action("rotate_180", rotate_180)], "crop"),
    ]
    return specs


def generate_benchmark(seed=None):
    rng = np.random.default_rng(seed)
    specs = create_task_specs()
    bt = {}

    for task_id, tier, actions, grid_type in specs:
        # Build 2 train pairs, 1 test pair
        train_pairs = []
        for _ in range(2):
            for _attempt in range(100):
                inp = make_grid_for_primitive(grid_type, rng)
                cur = inp.copy()
                valid = True
                for act in actions:
                    nxt = act(cur)
                    if np.array_equal(nxt, cur):
                        valid = False
                        break
                    cur = nxt
                if valid:
                    train_pairs.append({"input": inp, "output": cur})
                    break
            else:
                out = apply_sequence(inp, actions)
                train_pairs.append({"input": inp, "output": out})

        test_pairs = []
        for _ in range(1):
            for _attempt in range(100):
                inp = make_grid_for_primitive(grid_type, rng)
                cur = inp.copy()
                valid = True
                for act in actions:
                    nxt = act(cur)
                    if np.array_equal(nxt, cur):
                        valid = False
                        break
                    cur = nxt
                if valid:
                    test_pairs.append({"input": inp, "output": cur})
                    break
            else:
                out = apply_sequence(inp, actions)
                test_pairs.append({"input": inp, "output": out})

        bt[task_id] = {
            "tier": tier,
            "program": [a.name for a in actions],
            "train": train_pairs,
            "test": test_pairs,
        }

    return bt
    """
    bt = benchmark tasks dictionary
    """


if __name__ == "__main__":
    tasks = generate_benchmark()
    save_tasks(tasks, "benchmark_tasks.json")
    print(f"Generated {len(tasks)} benchmark tasks saved to benchmark_tasks.json")
