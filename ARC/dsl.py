from functools import partial
import numpy as np
from objects import find_objects, largest_object, count_objects


class Action:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, grid):
        return self.fn(grid)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Action) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


# Geometric primitives
def rotate_90(grid):
    return np.rot90(grid, -1).copy()


def rotate_180(grid):
    return np.rot90(grid, 2).copy()


def rotate_270(grid):
    return np.rot90(grid, 1).copy()


def flip_h(grid):
    return np.fliplr(grid).copy()


def flip_v(grid):
    return np.flipud(grid).copy()


def transpose(grid):
    return grid.T.copy()


# Color primitives
def recolor(grid, old_c, new_c):
    out = grid.copy()
    out[grid == old_c] = new_c
    return out


def recolor_by_size(grid, new_c):
    lo = largest_object(grid)
    out = grid.copy()
    if lo:
        for r, c in lo["coords"]:
            out[r, c] = new_c
    return out


def invert_colors(grid):
    out = grid.copy()
    mask = out > 0
    out[mask] = 10 - out[mask]
    return out


# Spatial primitives
def crop_to_content(grid):
    pts = np.argwhere(grid > 0)
    if len(pts) == 0:
        return grid.copy()
    min_r, min_c = pts.min(axis=0)
    max_r, max_c = pts.max(axis=0)
    return grid[min_r : max_r + 1, min_c : max_c + 1].copy()


def translate(grid, dx, dy):
    h, w = grid.shape
    out = np.zeros_like(grid)
    sr_start = max(0, -dy)
    sr_end = min(h, h - dy)
    sc_start = max(0, -dx)
    sc_end = min(w, w - dx)
    dr_start = max(0, dy)
    dr_end = min(h, h + dy)
    dc_start = max(0, dx)
    dc_end = min(w, w + dx)
    if sr_start < sr_end and sc_start < sc_end:
        out[dr_start:dr_end, dc_start:dc_end] = grid[sr_start:sr_end, sc_start:sc_end]
    return out


def tile(grid, rows, cols):
    return np.tile(grid, (rows, cols)).copy()


def mirror_complete_h(grid):
    return np.hstack([grid, np.fliplr(grid)]).copy()


def mirror_complete_v(grid):
    return np.vstack([grid, np.flipud(grid)]).copy()


# Object-level primitives
def keep_largest_object(grid):
    lo = largest_object(grid)
    out = np.zeros_like(grid)
    if lo:
        for r, c in lo["coords"]:
            out[r, c] = lo["color"]
    return out


def remove_object(grid, color):
    out = grid.copy()
    out[grid == color] = 0
    return out


def count_objects_grid(grid):
    cnt = count_objects(grid)
    return np.array([[min(9, cnt)]], dtype=np.int32)


# Action space builder
def build_action_space(train_pairs):
    cp = []

    # Geometric primitives (always viable)
    cp.append(Action("rotate_90", rotate_90))
    cp.append(Action("rotate_180", rotate_180))
    cp.append(Action("rotate_270", rotate_270))
    cp.append(Action("flip_h", flip_h))
    cp.append(Action("flip_v", flip_v))
    cp.append(Action("transpose", transpose))

    # Inspect colors
    in_colors = set()
    out_colors = set()
    in_shapes = set()
    out_shapes = set()

    for p in train_pairs:
        in_grid = p["input"]
        out_grid = p["output"]
        in_colors.update(np.unique(in_grid).tolist())
        out_colors.update(np.unique(out_grid).tolist())
        in_shapes.add(in_grid.shape)
        out_shapes.add(out_grid.shape)

    # Color recoloring
    for c1 in in_colors:
        for c2 in out_colors:
            if c1 != c2:
                name = f"recolor({c1}->{c2})"
                fn = partial(recolor, old_c=c1, new_c=c2)
                cp.append(Action(name, fn))

    for c2 in out_colors - {0}:
        name = f"recolor_by_size({c2})"
        fn = partial(recolor_by_size, new_c=c2)
        cp.append(Action(name, fn))

    cp.append(Action("invert_colors", invert_colors))

    # Spatial primitives
    cp.append(Action("crop_to_content", crop_to_content))

    # Tiling actions only if target shapes are multiples
    for r in [1, 2, 3]:
        for c in [1, 2, 3]:
            if r == 1 and c == 1:
                continue
            matches = any(
                out_s[0] == in_s[0] * r and out_s[1] == in_s[1] * c
                for in_s in in_shapes
                for out_s in out_shapes
            )
            if matches:
                cp.append(Action(f"tile({r}x{c})", partial(tile, rows=r, cols=c)))

    # Mirroring
    if any(out_s[1] == in_s[1] * 2 for in_s in in_shapes for out_s in out_shapes):
        cp.append(Action("mirror_complete_h", mirror_complete_h))
    if any(out_s[0] == in_s[0] * 2 for in_s in in_shapes for out_s in out_shapes):
        cp.append(Action("mirror_complete_v", mirror_complete_v))

    # Translation
    same_shape = any(in_s == out_s for in_s in in_shapes for out_s in out_shapes)
    if same_shape:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            name = f"translate({dx},{dy})"
            fn = partial(translate, dx=dx, dy=dy)
            cp.append(Action(name, fn))

    # Object-level
    cp.append(Action("keep_largest_object", keep_largest_object))
    for c in in_colors - {0}:
        name = f"remove_object({c})"
        fn = partial(remove_object, color=c)
        cp.append(Action(name, fn))

    if any(out_s == (1, 1) for out_s in out_shapes):
        cp.append(Action("count_objects", count_objects_grid))

    return cp
    """
    cp = candidate parameterized action list
    """
