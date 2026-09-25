import heapq
import time
import numpy as np
from grids import grid_hash
from dsl import build_action_space
from search_astar import heuristic, is_goal, state_hash


def solve_weighted_astar(train_pairs, weight=2.0, max_depth=3, max_nodes=50000):
    st_time = time.perf_counter()
    acts = build_action_space(train_pairs)
    init_state = tuple(p["input"].copy() for p in train_pairs)
    targets = tuple(p["output"] for p in train_pairs)

    h0 = heuristic(init_state, targets)
    if h0 == 0:
        return {
            "solved": True,
            "program": [],
            "nodes_expanded": 0,
            "time_taken": time.perf_counter() - st_time,
        }

    uid = 0
    # Priority queue: (f, h, depth, uid, state, program)
    pq = [(weight * h0, h0, 0, uid, init_state, [])]
    cs = {state_hash(init_state): 0}
    nd = 0

    while pq:
        f, h, d, _, cur_state, prog = heapq.heappop(pq)
        nd += 1

        if h == 0 and is_goal(cur_state, targets):
            return {
                "solved": True,
                "program": prog,
                "nodes_expanded": nd,
                "time_taken": time.perf_counter() - st_time,
            }

        if d >= max_depth or nd > max_nodes:
            continue

        for act in acts:
            try:
                nxt_state = tuple(act(g) for g in cur_state)
            except Exception:
                continue

            nxt_d = d + 1
            nxt_h = heuristic(nxt_state, targets)
            nxt_f = nxt_d + weight * nxt_h
            nxt_prog = prog + [act]

            sh = state_hash(nxt_state)
            if sh in cs and cs[sh] <= nxt_d:
                continue

            cs[sh] = nxt_d
            uid += 1
            heapq.heappush(pq, (nxt_f, nxt_h, nxt_d, uid, nxt_state, nxt_prog))

    return {
        "solved": False,
        "program": None,
        "nodes_expanded": nd,
        "time_taken": time.perf_counter() - st_time,
    }
    """
    cs = cost to reach state
    nd = nodes expanded count
    sh = state hash signature
    """


if __name__ == "__main__":
    from grids import load_tasks
    tasks = load_tasks("benchmark_tasks.json")
    for tid in ["easy_01", "med_01", "hard_01"]:
        res = solve_weighted_astar(tasks[tid]["train"], weight=2.0)
        print(f"Weighted A* {tid}: solved={res['solved']}, nodes={res['nodes_expanded']}, time={res['time_taken']*1000:.2f}ms, prog={res['program']}")
