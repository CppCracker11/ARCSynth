from collections import defaultdict
import time
import numpy as np
try:  # Package import
    from .dsl import (
        build_action_space,
        rotate_90,
        rotate_180,
        rotate_270,
        flip_h,
        flip_v,
        transpose,
    )
    from .heuristic import pose_and_paint_heuristic as heuristic
    from .search_astar import is_goal
    from .objects import count_objects
except ImportError:  # Direct script/test import
    from dsl import (
        build_action_space,
        rotate_90,
        rotate_180,
        rotate_270,
        flip_h,
        flip_v,
        transpose,
    )
    from heuristic import pose_and_paint_heuristic as heuristic
    from search_astar import is_goal
    from objects import count_objects


GEOM_OPS = [
    ("none", lambda g: g),
    ("r90", rotate_90),
    ("r180", rotate_180),
    ("r270", rotate_270),
    ("fh", flip_h),
    ("fv", flip_v),
    ("tr", transpose),
]


def extract_features(state, targets, step):
    # Shape relation
    same_shape = all(g.shape == t.shape for g, t in zip(state, targets))
    if same_shape:
        shape_code = 0
    elif any(g.shape[0] > t.shape[0] or g.shape[1] > t.shape[1] for g, t in zip(state, targets)):
        shape_code = 1
    else:
        shape_code = 2

    # Pose-and-Paint score bucket
    h_val = heuristic(state, targets)
    if h_val == 0:
        h_bucket = 0
    elif h_val <= 6:
        h_bucket = 1
    elif h_val <= 18:
        h_bucket = 2
    else:
        h_bucket = 3

    # Object count difference
    cur_objs = sum(count_objects(g) for g in state)
    tgt_objs = sum(count_objects(t) for t in targets)
    obj_diff = int(np.clip(cur_objs - tgt_objs, -1, 1))

    # Color difference flag
    cur_colors = set().union(*(np.unique(g).tolist() for g in state))
    tgt_colors = set().union(*(np.unique(t).tolist() for t in targets))
    color_diff = 0 if cur_colors == tgt_colors else 1

    # Geometric alignment signal
    best_geom_idx = 0
    best_red = 0
    if same_shape:
        for idx, (_, op) in enumerate(GEOM_OPS[1:], start=1):
            try:
                nxt = tuple(op(g) for g in state)
                red = h_val - heuristic(nxt, targets)
                if red > best_red:
                    best_red = red
                    best_geom_idx = idx
            except Exception:
                pass

    return (shape_code, h_bucket, obj_diff, color_diff, best_geom_idx, min(step, 3))


def action_key(act):
    return str(getattr(act, "name", act))


class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.8, eps_start=0.9, eps_end=0.05, eps_decay=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.q_table = defaultdict(float)

    def get_q(self, feat, act):
        return self.q_table[(feat, action_key(act))]

    def select_action(self, state, targets, acts, step, explore=True, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        feat = extract_features(state, targets, step)

        if explore and rng.random() < self.eps:
            idx = int(rng.integers(0, len(acts)))
            return acts[idx]

        h_prev = heuristic(state, targets)
        best_act = None
        best_val = -1e9

        for act in acts:
            q_val = self.get_q(feat, act)
            try:
                nxt = tuple(act(g) for g in state)
                h_nxt = heuristic(nxt, targets)
                red = h_prev - h_nxt
                score = q_val + 0.1 * red
            except Exception:
                score = q_val - 1e5

            if score > best_val:
                best_val = score
                best_act = act

        return best_act if best_act is not None else acts[0]

    def train(self, tasks, episodes=400, max_steps=3, seed=42):
        rng = np.random.default_rng(seed)
        task_list = list(tasks.values())

        for _ in range(episodes):
            task = rng.choice(task_list)
            train_pairs = task["train"]
            targets = tuple(p["output"] for p in train_pairs)
            cur_state = tuple(p["input"].copy() for p in train_pairs)
            acts = build_action_space(train_pairs)

            for step in range(max_steps):
                feat = extract_features(cur_state, targets, step)
                h_prev = heuristic(cur_state, targets)
                if h_prev == 0:
                    break

                act = self.select_action(cur_state, targets, acts, step, explore=True, rng=rng)
                try:
                    nxt_state = tuple(act(g) for g in cur_state)
                    h_curr = heuristic(nxt_state, targets)
                except Exception:
                    nxt_state = cur_state
                    h_curr = h_prev + 50

                exact = is_goal(nxt_state, targets)
                reward = float(h_prev - h_curr) - 1.0
                if exact:
                    reward += 100.0

                nxt_feat = extract_features(nxt_state, targets, step + 1)
                max_nxt_q = max((self.get_q(nxt_feat, a) for a in acts), default=0.0)
                cur_q = self.q_table[(feat, action_key(act))]

                self.q_table[(feat, action_key(act))] = cur_q + self.alpha * (
                    reward + self.gamma * max_nxt_q - cur_q
                )

                cur_state = nxt_state
                if exact:
                    break

            self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def solve(self, train_pairs, max_steps=3):
        st_time = time.perf_counter()
        acts = build_action_space(train_pairs)
        cur_state = tuple(p["input"].copy() for p in train_pairs)
        targets = tuple(p["output"] for p in train_pairs)

        if is_goal(cur_state, targets):
            return {
                "solved": True,
                "program": [],
                "nodes_expanded": 0,
                "time_taken": time.perf_counter() - st_time,
            }

        prog = []
        nd = 0
        for step in range(max_steps):
            nd += 1
            act = self.select_action(cur_state, targets, acts, step, explore=False)
            try:
                nxt_state = tuple(act(g) for g in cur_state)
            except Exception:
                break

            prog.append(act)
            cur_state = nxt_state

            if is_goal(cur_state, targets):
                return {
                    "solved": True,
                    "program": prog,
                    "nodes_expanded": nd,
                    "time_taken": time.perf_counter() - st_time,
                }

        return {
            "solved": False,
            "program": None,
            "nodes_expanded": nd,
            "time_taken": time.perf_counter() - st_time,
        }
        """
        nd = nodes expanded count
        """


if __name__ == "__main__":
    try:
        from .grids import load_tasks
    except ImportError:
        from grids import load_tasks
    tasks = load_tasks("benchmark_tasks.json")
    train_tasks = {k: v for k, v in tasks.items() if v["tier"] in ["easy", "medium"]}
    agent = QLearningAgent(alpha=0.4, gamma=0.8)
    print("Training Q-Learning agent on easy/medium tasks...")
    agent.train(train_tasks, episodes=400)
    for tid in ["easy_01", "med_01", "hard_01"]:
        res = agent.solve(tasks[tid]["train"])
        print(f"RL {tid}: solved={res['solved']}, steps={res['nodes_expanded']}, time={res['time_taken']*1000:.2f}ms, prog={res['program']}")
