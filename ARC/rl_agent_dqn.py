from collections import deque
import os
import random
import re
import time

import numpy as np
import torch
from torch import nn

try:  # Package import
    from .dsl import build_action_space
    from .heuristic import pose_and_paint_heuristic
    from .search_astar import is_goal
    from .objects import count_objects
except ImportError:  # Direct script/test import
    from dsl import build_action_space
    from heuristic import pose_and_paint_heuristic
    from search_astar import is_goal
    from objects import count_objects


ACTION_TYPES = (
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_h",
    "flip_v",
    "transpose",
    "recolor",
    "recolor_by_size",
    "invert_colors",
    "crop_to_content",
    "translate",
    "tile",
    "mirror_complete_h",
    "mirror_complete_v",
    "keep_largest_object",
    "remove_object",
    "count_objects",
    "other",
)
STATE_DIM = 12
ACTION_DIM = len(ACTION_TYPES) + 4


def _action_features(action):
    name = str(getattr(action, "name", action))
    category = next(
        (item for item in ACTION_TYPES[:-1]
         if name == item or name.startswith(item + "(")),
        "other",
    )
    vector = np.zeros(ACTION_DIM, dtype=np.float32)
    vector[ACTION_TYPES.index(category)] = 1.0
    params = [int(value) for value in re.findall(r"-?\d+", name[name.find("("):])]
    values = [0.0] * 4

    if category == "recolor" and len(params) >= 2:
        values[:2] = [params[0] / 9.0, params[1] / 9.0]
    elif category in ("recolor_by_size", "remove_object") and params:
        values[0] = params[0] / 9.0
    elif category == "translate" and len(params) >= 2:
        values[:2] = [params[0] / 5.0, params[1] / 5.0]
    elif category == "tile" and len(params) >= 2:
        values[:2] = [params[0] / 3.0, params[1] / 3.0]

    vector[len(ACTION_TYPES):] = np.asarray(values, dtype=np.float32)
    return vector


def _state_features(state, targets, step, max_steps):
    if not state or len(state) != len(targets):
        raise ValueError("State and target collections must be non-empty and aligned.")

    total_cells = sum(max(g.size, t.size) for g, t in zip(state, targets))
    total_cells = max(total_cells, 1)
    shape_mismatches = 0
    shape_distance = 0.0
    occupancy_mismatches = 0
    shared_foreground = 0
    paint_mismatches = 0
    current_objects = target_objects = 0
    current_foreground = target_foreground = 0
    center_distance = 0.0
    current_colors = set()
    target_colors = set()
    target_density = 0.0
    same_shapes = 0

    for current, target in zip(state, targets):
        ch, cw = current.shape
        th, tw = target.shape
        if current.shape == target.shape:
            same_shapes += 1
        else:
            shape_mismatches += 1
        shape_distance += (abs(ch - th) + abs(cw - tw)) / max(ch, cw, th, tw, 1)

        h, w = max(ch, th), max(cw, tw)
        cur = np.zeros((h, w), dtype=np.int32)
        tgt = np.zeros((h, w), dtype=np.int32)
        cur[:ch, :cw] = current
        tgt[:th, :tw] = target
        cur_fg, tgt_fg = cur != 0, tgt != 0
        occupancy_mismatches += int(np.count_nonzero(cur_fg != tgt_fg))
        both = cur_fg & tgt_fg
        shared_foreground += int(np.count_nonzero(both))
        paint_mismatches += int(np.count_nonzero((cur != tgt) & both))

        current_objects += count_objects(current)
        target_objects += count_objects(target)
        current_foreground += int(np.count_nonzero(current))
        target_foreground += int(np.count_nonzero(target))
        current_colors.update(np.unique(current).tolist())
        target_colors.update(np.unique(target).tolist())
        target_density += float(np.count_nonzero(target)) / max(target.size, 1)

        cp = np.argwhere(cur_fg)
        tp = np.argwhere(tgt_fg)
        if len(cp) and len(tp):
            center_distance += float(
                np.abs(cp.mean(axis=0) - tp.mean(axis=0)).sum() / max(h, w, 1)
            )
        elif len(cp) != len(tp):
            center_distance += 1.0

    n = len(state)
    h_value = pose_and_paint_heuristic(state, targets)
    color_union = current_colors | target_colors
    color_difference = len(current_colors ^ target_colors) / max(len(color_union), 1)

    return np.asarray([
        h_value / total_cells,
        shape_mismatches / n,
        shape_distance / n,
        occupancy_mismatches / total_cells,
        paint_mismatches / max(shared_foreground, 1),
        abs(current_objects - target_objects) / max(current_objects + target_objects, 1),
        abs(current_foreground - target_foreground) / total_cells,
        color_difference,
        center_distance / n,
        same_shapes / n,
        target_density / n,
        min(step / max(max_steps, 1), 1.0),
    ], dtype=np.float32)


class _QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(
        self,
        gamma=0.8,
        learning_rate=1e-3,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=0.99,
        batch_size=64,
        replay_capacity=10000,
        target_update=100,
        hidden_dim=128,
        device=None,
    ):
        self.gamma = float(gamma)
        self.eps = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.hidden_dim = int(hidden_dim)
        if self.batch_size < 1 or self.target_update < 1:
            raise ValueError("batch_size and target_update must be positive.")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        input_dim = STATE_DIM + ACTION_DIM
        self.policy_net = _QNetwork(input_dim, self.hidden_dim).to(self.device)
        self.target_net = _QNetwork(input_dim, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=float(learning_rate)
        )
        self.memory = deque(maxlen=int(replay_capacity))
        self.learn_steps = 0

    @staticmethod
    def action_key(action):
        return str(getattr(action, "name", action))

    def _q_values(self, state_features, action_features, network=None):
        if not len(action_features):
            return np.empty(0, dtype=np.float32)
        network = network or self.policy_net
        state_matrix = np.repeat(
            np.asarray(state_features, dtype=np.float32)[None, :],
            len(action_features),
            axis=0,
        )
        inputs = np.concatenate(
            [state_matrix, np.asarray(action_features, dtype=np.float32)], axis=1
        )
        with torch.no_grad():
            values = network(torch.as_tensor(inputs, device=self.device)).squeeze(-1)
        return values.detach().cpu().numpy()

    def select_action(
        self, state, targets, actions, step, explore=True, rng=None, max_steps=3
    ):
        if not actions:
            return None
        if rng is None:
            rng = np.random.default_rng()
        if explore and rng.random() < self.eps:
            return actions[int(rng.integers(len(actions)))]

        state_features = _state_features(state, targets, step, max_steps)
        action_features = np.stack([_action_features(a) for a in actions])
        q_values = self._q_values(state_features, action_features)
        return actions[int(np.argmax(q_values))]

    def _remember(self, state_features, action_features, reward,
                  next_state_features, next_action_features, done):
        self.memory.append((
            np.asarray(state_features, dtype=np.float32),
            np.asarray(action_features, dtype=np.float32),
            float(reward),
            np.asarray(next_state_features, dtype=np.float32),
            np.asarray(next_action_features, dtype=np.float32),
            bool(done),
        ))

    def _learn(self, rng):
        if len(self.memory) < self.batch_size:
            return
        indices = rng.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[int(i)] for i in indices]

        inputs = np.stack([
            np.concatenate([item[0], item[1]]) for item in batch
        ])
        q_pred = self.policy_net(
            torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        ).squeeze(-1)

        targets = []
        for state_feat, action_feat, reward, next_state_feat, next_actions, done in batch:
            target_value = reward
            if not done and len(next_actions):
                next_values = self._q_values(
                    next_state_feat, next_actions, network=self.target_net
                )
                target_value += self.gamma * float(np.max(next_values))
            targets.append(target_value)

        q_target = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        loss = nn.functional.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, tasks, episodes=400, max_steps=3, seed=42):
        if not tasks:
            return self
        rng = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        task_list = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)

        for _ in range(int(episodes)):
            task = task_list[int(rng.integers(len(task_list)))]
            train_pairs = task["train"]
            targets = tuple(pair["output"] for pair in train_pairs)
            state = tuple(pair["input"].copy() for pair in train_pairs)
            actions = build_action_space(train_pairs)
            if not actions:
                continue

            for step in range(int(max_steps)):
                if is_goal(state, targets):
                    break
                state_features = _state_features(state, targets, step, max_steps)
                action = self.select_action(
                    state, targets, actions, step, explore=True,
                    rng=rng, max_steps=max_steps
                )
                try:
                    next_state = tuple(action(grid) for grid in state)
                    next_h = pose_and_paint_heuristic(next_state, targets)
                except Exception:
                    next_state = state
                    next_h = pose_and_paint_heuristic(state, targets) + 10.0

                current_h = pose_and_paint_heuristic(state, targets)
                done = is_goal(next_state, targets)
                reward = current_h - next_h - 1.0
                if done:
                    reward += 100.0

                next_state_features = _state_features(
                    next_state, targets, step + 1, max_steps
                )
                next_action_features = np.stack(
                    [_action_features(a) for a in actions]
                ) if not done else np.empty((0, ACTION_DIM), dtype=np.float32)

                self._remember(
                    state_features, _action_features(action), reward,
                    next_state_features, next_action_features, done
                )
                self._learn(rng)
                state = next_state
                if done:
                    break

            self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return self

    def solve(self, train_pairs, max_steps=3):
        start = time.perf_counter()
        actions = build_action_space(train_pairs)
        state = tuple(pair["input"].copy() for pair in train_pairs)
        targets = tuple(pair["output"] for pair in train_pairs)

        if is_goal(state, targets):
            return {
                "solved": True, "program": [], "nodes_expanded": 0,
                "time_taken": time.perf_counter() - start,
            }

        program = []
        nodes_expanded = 0
        for step in range(int(max_steps)):
            action = self.select_action(
                state, targets, actions, step, explore=False, max_steps=max_steps
            )
            if action is None:
                break
            nodes_expanded += 1
            try:
                next_state = tuple(action(grid) for grid in state)
            except Exception:
                break
            program.append(action)
            state = next_state
            if is_goal(state, targets):
                return {
                    "solved": True, "program": program,
                    "nodes_expanded": nodes_expanded,
                    "time_taken": time.perf_counter() - start,
                }

        return {
            "solved": False, "program": None,
            "nodes_expanded": nodes_expanded,
            "time_taken": time.perf_counter() - start,
        }

    def save_checkpoint(self, path):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "eps": self.eps,
            "learn_steps": self.learn_steps,
            "hidden_dim": self.hidden_dim,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("state_dim") != STATE_DIM or checkpoint.get("action_dim") != ACTION_DIM:
            raise ValueError("Checkpoint feature dimensions do not match this DQN implementation.")
        if checkpoint.get("hidden_dim") != self.hidden_dim:
            raise ValueError("Checkpoint hidden_dim does not match this agent.")
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.eps = float(checkpoint.get("eps", self.eps))
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
        return self


if __name__ == "__main__":
    try:
        from .grids import load_tasks
    except ImportError:
        from grids import load_tasks

    tasks = load_tasks("benchmark_tasks.json")
    training_tasks = {
        key: value for key, value in tasks.items()
        if value.get("tier") in ("easy", "medium")
    }
    agent = DQNAgent()
    agent.train(training_tasks, episodes=400, max_steps=3)
    agent.save_checkpoint("dqn_checkpoint.pt")
