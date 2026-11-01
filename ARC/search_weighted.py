import heapq
import time
import math
import numpy as np
try:  # Package import
    from .dsl import build_action_space
    from .heuristic import pose_and_paint_heuristic
    from .search_astar import is_goal, state_hash
except ImportError:  # Direct script/test import
    from dsl import build_action_space
    from heuristic import pose_and_paint_heuristic
    from search_astar import is_goal, state_hash


def solve_weighted_astar(
    train_pairs, weight=2.0, max_depth=3, max_nodes=50000
):
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be a finite non-negative number.")

    st_time = time.perf_counter()
    acts = build_action_space(train_pairs)
    init_state = tuple(p["input"].copy() for p in train_pairs)
    targets = tuple(p["output"] for p in train_pairs)

    h0 = pose_and_paint_heuristic(init_state, targets)
    if is_goal(init_state, targets):
        return {
            "solved": True,
            "program": [],
            "nodes_expanded": 0,
            "time_taken": time.perf_counter() - st_time,
        }

    uid = 0
    pq = [(weight * h0, h0, 0, uid, init_state, [])]
    costs = {state_hash(init_state): 0}
    nodes_expanded = 0

    while pq and nodes_expanded < max_nodes:
        _, _, depth, _, current_state, program = heapq.heappop(pq)
        nodes_expanded += 1

        if is_goal(current_state, targets):
            return {
                "solved": True,
                "program": program,
                "nodes_expanded": nodes_expanded,
                "time_taken": time.perf_counter() - st_time,
            }

        if depth >= max_depth:
            continue

        for action in acts:
            try:
                next_state = tuple(action(grid) for grid in current_state)
            except Exception:
                continue

            next_depth = depth + 1
            signature = state_hash(next_state)
            if signature in costs and costs[signature] <= next_depth:
                continue

            costs[signature] = next_depth
            next_h = pose_and_paint_heuristic(next_state, targets)
            uid += 1
            heapq.heappush(
                pq,
                (
                    next_depth + weight * next_h,
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
        "nodes_expanded": nodes_expanded,
        "time_taken": time.perf_counter() - st_time,
    }


if __name__ == "__main__":
    try:
        from .grids import load_tasks
    except ImportError:
        from grids import load_tasks

    tasks = load_tasks("benchmark_tasks.json")
    for task_id in ["easy_01", "med_01", "hard_01"]:
        result = solve_weighted_astar(tasks[task_id]["train"])
        print(
            f"Weighted A* {task_id}: solved={result['solved']}, "
            f"nodes={result['nodes_expanded']}, "
            f"time={result['time_taken'] * 1000:.2f}ms, "
            f"program={result['program']}"
        )
