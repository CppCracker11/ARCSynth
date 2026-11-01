import heapq
import time
import numpy as np
try:  # Package import
    from .grids import grid_hash
    from .dsl import build_action_space
    from .heuristic import pose_and_paint_heuristic
except ImportError:  # Direct script/test import
    from grids import grid_hash
    from dsl import build_action_space
    from heuristic import pose_and_paint_heuristic


def state_hash(state):
    return tuple(grid_hash(g) for g in state)


def heuristic(state, targets):
    return pose_and_paint_heuristic(state, targets)


def is_goal(state, targets):
    return len(state) == len(targets) and all(
        np.array_equal(g, t) for g, t in zip(state, targets)
    )


def solve_astar(train_pairs, max_depth=3, max_nodes=50000):
    st_time = time.perf_counter()
    acts = build_action_space(train_pairs)
    init_state = tuple(p["input"].copy() for p in train_pairs)
    targets = tuple(p["output"] for p in train_pairs)

    h0 = heuristic(init_state, targets)
    if is_goal(init_state, targets):
        return {
            "solved": True,
            "program": [],
            "nodes_expanded": 0,
            "time_taken": time.perf_counter() - st_time,
        }

    uid = 0
    pq = [(h0, h0, 0, uid, init_state, [])]
    cs = {state_hash(init_state): 0}
    nd = 0

    while pq and nd < max_nodes:
        _, _, depth, _, cur_state, program = heapq.heappop(pq)
        nd += 1

        if is_goal(cur_state, targets):
            return {
                "solved": True,
                "program": program,
                "nodes_expanded": nd,
                "time_taken": time.perf_counter() - st_time,
            }

        if depth >= max_depth:
            continue

        for action in acts:
            try:
                next_state = tuple(action(grid) for grid in cur_state)
            except Exception:
                continue

            next_depth = depth + 1
            signature = state_hash(next_state)
            if signature in cs and cs[signature] <= next_depth:
                continue

            cs[signature] = next_depth
            next_h = heuristic(next_state, targets)
            uid += 1
            heapq.heappush(
                pq,
                (
                    next_depth + next_h,
                    next_h,
                    next_depth,
                    uid,
                    next_state,
                    program + [action],
                ),
            )

    return {
        "solved": False,
        "program": None,
        "nodes_expanded": nd,
        "time_taken": time.perf_counter() - st_time,
    }


if __name__ == "__main__":
    try:
        from .grids import load_tasks
    except ImportError:
        from grids import load_tasks

    tasks = load_tasks("benchmark_tasks.json")
    for task_id in ["easy_01", "med_01", "hard_01"]:
        result = solve_astar(tasks[task_id]["train"])
        print(
            f"A* {task_id}: solved={result['solved']}, "
            f"nodes={result['nodes_expanded']}, "
            f"time={result['time_taken'] * 1000:.2f}ms, "
            f"program={result['program']}"
        )
