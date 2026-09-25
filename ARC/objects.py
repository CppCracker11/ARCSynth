from collections import deque
import numpy as np


def find_objects(grid, bg=0):
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    objs = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or vis[r, c]:
                continue
            
            coords = []
            q = deque([(r, c)])
            vis[r, c] = True

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not vis[nr, nc] and grid[nr, nc] == color:
                            vis[nr, nc] = True
                            q.append((nr, nc))

            min_r = min(p[0] for p in coords)
            max_r = max(p[0] for p in coords)
            min_c = min(p[1] for p in coords)
            max_c = max(p[1] for p in coords)

            objs.append({
                "color": color,
                "coords": coords,
                "size": len(coords),
                "bbox": (min_r, min_c, max_r, max_c),
            })

    return objs


def largest_object(grid, bg=0):
    objs = find_objects(grid, bg)
    if not objs:
        return None
    return max(objs, key=lambda o: o["size"])


def count_objects(grid, bg=0):
    return len(find_objects(grid, bg))
