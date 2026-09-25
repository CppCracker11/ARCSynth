import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: grey
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: teal
    "#870C25",  # 9: maroon
]

COLOR_MAP = ListedColormap(ARC_COLORS)
COLOR_NORM = Normalize(vmin=0, vmax=9)


def grid_hash(grid):
    return (grid.shape, grid.tobytes())


def generate_random_grid(h, w, colors=None, num_shapes=3, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    if colors is None:
        colors = [1, 2, 3, 4]

    g = np.zeros((h, w), dtype=np.int32)
    for _ in range(num_shapes):
        c = int(rng.choice(colors))
        st = rng.choice(["rect", "line_h", "line_v", "pixel"])
        r0 = int(rng.integers(0, h))
        c0 = int(rng.integers(0, w))

        if st == "rect":
            rh = int(rng.integers(1, max(2, h // 2)))
            rw = int(rng.integers(1, max(2, w // 2)))
            g[r0 : min(h, r0 + rh), c0 : min(w, c0 + rw)] = c
        elif st == "line_h":
            rw = int(rng.integers(2, max(3, w)))
            g[r0, c0 : min(w, c0 + rw)] = c
        elif st == "line_v":
            rh = int(rng.integers(2, max(3, h)))
            g[r0 : min(h, r0 + rh), c0] = c
        else:
            g[r0, c0] = c

    return g


def save_tasks(tasks, path):
    st = {}
    for task_id, task in tasks.items():
        st[task_id] = {
            "train": [
                {
                    "input": pair["input"].tolist() if isinstance(pair["input"], np.ndarray) else pair["input"],
                    "output": pair["output"].tolist() if isinstance(pair["output"], np.ndarray) else pair["output"],
                }
                for pair in task["train"]
            ],
            "test": [
                {
                    "input": pair["input"].tolist() if isinstance(pair["input"], np.ndarray) else pair["input"],
                    "output": pair["output"].tolist() if isinstance(pair["output"], np.ndarray) else pair["output"],
                }
                for pair in task["test"]
            ],
        }
        if "program" in task:
            st[task_id]["program"] = task["program"]
        if "tier" in task:
            st[task_id]["tier"] = task["tier"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = {}
    for task_id, task in data.items():
        tasks[task_id] = {
            "train": [
                {
                    "input": np.array(p["input"], dtype=np.int32),
                    "output": np.array(p["output"], dtype=np.int32),
                }
                for p in task["train"]
            ],
            "test": [
                {
                    "input": np.array(p["input"], dtype=np.int32),
                    "output": np.array(p["output"], dtype=np.int32),
                }
                for p in task["test"]
            ],
        }
        if "program" in task:
            tasks[task_id]["program"] = task["program"]
        if "tier" in task:
            tasks[task_id]["tier"] = task["tier"]

    return tasks


def visualize_task(task, save_path=None, title=None):
    train_pairs = task["train"]
    test_pairs = task["test"]
    n_rows = len(train_pairs) + len(test_pairs)

    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 2.5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for i, pair in enumerate(train_pairs):
        ax_in, ax_out = axes[i, 0], axes[i, 1]
        ax_in.imshow(pair["input"], cmap=COLOR_MAP, norm=COLOR_NORM)
        ax_in.set_title(f"Train {i+1} In")
        ax_in.set_xticks(np.arange(-0.5, pair["input"].shape[1], 1), minor=True)
        ax_in.set_yticks(np.arange(-0.5, pair["input"].shape[0], 1), minor=True)
        ax_in.grid(which="minor", color="#333333", linewidth=0.5)
        ax_in.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax_out.imshow(pair["output"], cmap=COLOR_MAP, norm=COLOR_NORM)
        ax_out.set_title(f"Train {i+1} Out")
        ax_out.set_xticks(np.arange(-0.5, pair["output"].shape[1], 1), minor=True)
        ax_out.set_yticks(np.arange(-0.5, pair["output"].shape[0], 1), minor=True)
        ax_out.grid(which="minor", color="#333333", linewidth=0.5)
        ax_out.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    row_off = len(train_pairs)
    for j, pair in enumerate(test_pairs):
        ax_in, ax_out = axes[row_off + j, 0], axes[row_off + j, 1]
        ax_in.imshow(pair["input"], cmap=COLOR_MAP, norm=COLOR_NORM)
        ax_in.set_title(f"Test {j+1} In")
        ax_in.set_xticks(np.arange(-0.5, pair["input"].shape[1], 1), minor=True)
        ax_in.set_yticks(np.arange(-0.5, pair["input"].shape[0], 1), minor=True)
        ax_in.grid(which="minor", color="#333333", linewidth=0.5)
        ax_in.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax_out.imshow(pair["output"], cmap=COLOR_MAP, norm=COLOR_NORM)
        ax_out.set_title(f"Test {j+1} Out")
        ax_out.set_xticks(np.arange(-0.5, pair["output"].shape[1], 1), minor=True)
        ax_out.set_yticks(np.arange(-0.5, pair["output"].shape[0], 1), minor=True)
        ax_out.grid(which="minor", color="#333333", linewidth=0.5)
        ax_out.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
