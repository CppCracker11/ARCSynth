import numpy as np
try:  # Package import
    from .objects import find_objects
except ImportError:  # Direct script/test import
    from objects import find_objects


def _content_view(grid):
    points = np.argwhere(grid != 0)
    if not len(points):
        return grid
    low = points.min(axis=0)
    high = points.max(axis=0) + 1
    return grid[low[0]:high[0], low[1]:high[1]]


def _grid_score(current, target):
    current = np.asarray(current)
    target = np.asarray(target)
    ch, cw = current.shape
    th, tw = target.shape
    shape_cost = 2.0 * (abs(ch - th) + abs(cw - tw))

    compare_current, compare_target = current, target
    if current.shape != target.shape:
        cropped_current = _content_view(current)
        cropped_target = _content_view(target)
        if cropped_current.shape == cropped_target.shape:
            compare_current, compare_target = cropped_current, cropped_target

    h = max(compare_current.shape[0], compare_target.shape[0])
    w = max(compare_current.shape[1], compare_target.shape[1])
    cur = np.zeros((h, w), dtype=compare_current.dtype)
    tgt = np.zeros((h, w), dtype=compare_target.dtype)
    cur[:compare_current.shape[0], :compare_current.shape[1]] = compare_current
    tgt[:compare_target.shape[0], :compare_target.shape[1]] = compare_target

    cur_fg = cur != 0
    tgt_fg = tgt != 0
    pose_cost = float(np.count_nonzero(cur_fg != tgt_fg))
    shared_fg = cur_fg & tgt_fg
    paint_cost = float(np.count_nonzero((cur != tgt) & shared_fg))

    cur_objects = find_objects(current)
    tgt_objects = find_objects(target)
    object_cost = 1.5 * abs(len(cur_objects) - len(tgt_objects))

    cur_points = np.argwhere(cur_fg)
    tgt_points = np.argwhere(tgt_fg)
    if len(cur_points) and len(tgt_points):
        cur_center = cur_points.mean(axis=0)
        tgt_center = tgt_points.mean(axis=0)
        center_cost = float(
            np.abs(cur_center - tgt_center).sum() / max(h, w, 1)
        )
    else:
        center_cost = 0.0 if len(cur_points) == len(tgt_points) else 1.0

    cur_colors, cur_counts = np.unique(current[current != 0], return_counts=True)
    tgt_colors, tgt_counts = np.unique(target[target != 0], return_counts=True)
    cur_hist = dict(zip(cur_colors.tolist(), cur_counts.tolist()))
    tgt_hist = dict(zip(tgt_colors.tolist(), tgt_counts.tolist()))
    color_hist_cost = 0.0
    for color in set(cur_hist) | set(tgt_hist):
        color_hist_cost += abs(cur_hist.get(color, 0) - tgt_hist.get(color, 0))
    color_hist_cost *= 0.25

    return (
        shape_cost
        + pose_cost
        + 0.75 * paint_cost
        + object_cost
        + 0.5 * center_cost
        + color_hist_cost
    )

def pose_and_paint_heuristic(state, targets):
    """Estimate structural and color mismatch across all training grids."""
    if len(state) != len(targets):
        raise ValueError("State and target collections must have equal length.")
    return float(sum(_grid_score(grid, target) for grid, target in zip(state, targets)))


def heuristic(state, targets):
    return pose_and_paint_heuristic(state, targets)
