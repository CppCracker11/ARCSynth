from collections import deque
import time
import numpy as np
from grids import grid_hash
from dsl import build_action_space


def state_hash(state):
    return tuple(grid_hash(g) for g in state)


def is_goal(state, targets):
    return all(np.array_equal(g, t) for g, t in zip(state, targets))


def replay_program(grid, program):
    cur = grid.copy()
    for act in program:
        cur = act(cur)
    return cur


def solve_bfs(train_pairs, max_depth=3, max_nodes=50000):
    st_time = time.perf_counter()
    acts = build_action_space(train_pairs)
    init_state = tuple(p["input"].copy() for p in train_pairs)
    targets = tuple(p["output"] for p in train_pairs)

    if is_goal(init_state, targets):
        return {
            "solved": True,
            "program": [],
            "nodes_expanded": 0,
            "time_taken": time.perf_counter() - st_time,
        }

    vs = {state_hash(init_state)}
    q = deque([(init_state, [])])
    nd = 0

    while q:
        cur_state, prog = q.popleft()
        if len(prog) >= max_depth:
            continue

        nd += 1
        if nd > max_nodes:
            break

        for act in acts:
            try:
                nxt_state = tuple(act(g) for g in cur_state)
            except Exception:
                continue

            nxt_prog = prog + [act]
            if is_goal(nxt_state, targets):
                return {
                    "solved": True,
                    "program": nxt_prog,
                    "nodes_expanded": nd,
                    "time_taken": time.perf_counter() - st_time,
                }

            h = state_hash(nxt_state)
            if h not in vs:
                vs.add(h)
                q.append((nxt_state, nxt_prog))

    return {
        "solved": False,
        "program": None,
        "nodes_expanded": nd,
        "time_taken": time.perf_counter() - st_time,
    }
    """
    vs = visited state signatures
    nd = nodes expanded count
    """


if __name__ == "__main__":
    from grids import load_tasks
    tasks = load_tasks("benchmark_tasks.json")
    for tid in ["easy_01", "med_01", "hard_01"]:
        res = solve_bfs(tasks[tid]["train"])
        print(f"BFS {tid}: solved={res['solved']}, nodes={res['nodes_expanded']}, time={res['time_taken']*1000:.2f}ms, prog={res['program']}")
