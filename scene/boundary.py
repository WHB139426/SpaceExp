"""Build the ground-plane scene footprint and check horizontal and height limits."""

from collections import deque
import math

import numpy as np


def _morphology(mask, radius, *, erode=False):
    """Binary disk dilation/erosion using only Blender's bundled NumPy."""
    reach = math.ceil(radius)
    padded = np.pad(mask, reach, constant_values=False)
    result = np.ones_like(mask) if erode else np.zeros_like(mask)
    height, width = mask.shape
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dx * dx + dy * dy > radius * radius + 1e-9:
                continue
            shifted = padded[reach + dy:reach + dy + height, reach + dx:reach + dx + width]
            if erode:
                result &= shifted
            else:
                result |= shifted
    return result


def _component(mask, seed):
    """Return the four-connected component containing a row/column seed."""
    selected = np.zeros_like(mask)
    if not mask[seed]:
        return selected
    selected[seed] = True
    pending = deque([seed])
    height, width = mask.shape
    while pending:
        row, column = pending.popleft()
        for y, x in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if 0 <= y < height and 0 <= x < width and mask[y, x] and not selected[y, x]:
                selected[y, x] = True
                pending.append((y, x))
    return selected


def build_boundary(points, *, origin, right, ahead, up, floor, max_height,
                   cell_size=.05, close_radius=.15, camera_radius=.15):
    """Return masks in fixed ground coordinates; this function does not constrain movement."""
    values = (cell_size, close_radius, camera_radius, max_height)
    if not all(math.isfinite(value) for value in values) or cell_size <= 0 or min(values[1:]) < 0:
        raise ValueError('Boundary cell size must be positive; radii and height must be finite and nonnegative')
    origin, up, floor = (np.asarray(value, dtype=float) for value in (origin, up, floor))
    basis = np.column_stack((right, ahead))

    def projected_chunks():
        for start in range(0, len(points), 250_000):
            block = points[start:start + 250_000]
            heights = np.einsum('ij,j->i', block - floor, up)
            keep = np.isfinite(block).all(axis=1) & (heights >= -.1) & (heights <= max_height)
            if keep.any():
                yield np.einsum('ij,jk->ik', block[keep] - origin, basis)

    low, high = np.zeros(2), np.zeros(2)
    point_count = 0
    for xy in projected_chunks():
        low, high = np.minimum(low, xy.min(axis=0)), np.maximum(high, xy.max(axis=0))
        point_count += len(xy)
    if not point_count:
        raise ValueError('No point-cloud points remain in the boundary height range')
    padding = math.ceil((close_radius + camera_radius) / cell_size) + 2
    grid_origin = (np.floor(low / cell_size) - padding) * cell_size
    size = np.ceil((high - grid_origin) / cell_size).astype(int) + padding + 1
    if int(size[0]) * int(size[1]) > 4_000_000:
        raise ValueError('Boundary grid is too large; increase boundary_cell_size')
    scene = np.zeros((size[1], size[0]), dtype=bool)
    for xy in projected_chunks():
        cells = np.floor((xy - grid_origin) / cell_size).astype(int)
        scene[cells[:, 1], cells[:, 0]] = True
    scene = _morphology(_morphology(scene, close_radius / cell_size), close_radius / cell_size, erode=True)

    # Only empty cells connected to the outside remain empty. This fills every
    # enclosed hole while preserving exterior recesses and L-shaped outlines.
    scene = ~_component(~scene, (0, 0))

    start = np.floor(-grid_origin / cell_size).astype(int)
    start_cell = (int(start[1]), int(start[0]))
    start_inside = bool(scene[start_cell])
    if start_inside:
        seed = start_cell
        seed_distance = 0.
    else:
        occupied = np.argwhere(scene)
        if not len(occupied):
            raise ValueError('No scene footprint remains after closing the projected points')
        centers = grid_origin + (occupied[:, ::-1] + .5) * cell_size
        distances = np.linalg.norm(centers, axis=1)
        nearest = int(distances.argmin())
        seed = tuple(occupied[nearest])
        seed_distance = float(distances[nearest])
    scene = _component(scene, seed)
    allowed = _morphology(scene, camera_radius / cell_size, erode=True)
    rows, columns = np.nonzero(scene)
    low = grid_origin + np.array([columns.min(), rows.min()]) * cell_size
    high = grid_origin + np.array([columns.max() + 1, rows.max() + 1]) * cell_size
    return {
        'scene_mask': scene, 'allowed_mask': allowed,
        'grid_origin': grid_origin, 'cell_size': cell_size,
        'origin': origin, 'right': basis[:, 0], 'ahead': basis[:, 1], 'up': up,
        'close_radius': close_radius, 'camera_radius': camera_radius,
        'max_height': max_height, 'outer_only': True,
        'point_count': point_count, 'bounds': np.stack((low, high)),
        'start_inside': start_inside, 'start_allowed': bool(allowed[start_cell]),
        'seed_distance': seed_distance,
    }


def height_block_reason(position, direction, magnitude, *, floor, up, min_height, max_height):
    """Check a straight movement against camera-center heights above the estimated floor."""
    height = float((np.asarray(position, dtype=float) - floor) @ up)
    tolerance = 1e-6  # Allow for Blender's float32 camera coordinates.
    if not math.isfinite(height) or not min_height - tolerance <= height <= max_height + tolerance:
        return f'The current camera height is outside the allowed range [{min_height:g}, {max_height:g}] meters above the estimated floor.'
    vertical = float(np.asarray(direction, dtype=float) @ up)
    # Compare distances before multiplying, so enormous requests cannot overflow coordinates.
    if vertical > 0 and magnitude > (max_height - height + tolerance) / vertical:
        return f'The movement path would exceed the maximum camera height of {max_height:g} meters above the estimated floor.'
    if vertical < 0 and magnitude > (height - min_height + tolerance) / -vertical:
        return f'The movement path would go below the minimum camera height of {min_height:g} meters above the estimated floor.'
    return None


def movement_block_reason(boundary, position, heading, magnitude):
    """Check every grid cell touched by a movement segment, including corner crossings."""
    mask = boundary['allowed_mask']
    height, width = mask.shape
    basis = np.column_stack((boundary['right'], boundary['ahead']))
    start = ((np.asarray(position, dtype=float) - boundary['origin']) @ basis
             - boundary['grid_origin']) / boundary['cell_size']
    if not np.isfinite(start).all() or not (0 <= start[0] < width and 0 <= start[1] < height):
        return 'The current camera position is outside the allowed scene region.'
    column, row = np.floor(start).astype(int)
    if not mask[row, column]:
        return 'The current camera position is outside the allowed scene region, including camera clearance.'
    # A segment longer than the grid diagonal cannot fit. Check before multiplying
    # by the magnitude so very large requests never overflow camera coordinates.
    if magnitude > math.hypot(width, height) * boundary['cell_size']:
        return 'The requested movement path would leave the allowed scene region.'
    delta = np.asarray(heading, dtype=float) @ basis * (magnitude / boundary['cell_size'])
    end = start + delta
    if not np.isfinite(end).all() or not (0 <= end[0] < width and 0 <= end[1] < height):
        return 'The requested movement path would leave the allowed scene region.'
    end_column, end_row = np.floor(end).astype(int)
    reason = 'The movement path crosses the scene boundary or its camera-clearance margin.'

    # A segment lying exactly on a grid line touches cells on both sides.
    offsets = [(0, 0)]
    if delta[0] == 0 and math.isclose(start[0], round(start[0]), abs_tol=1e-9, rel_tol=0):
        offsets.append((-1, 0))
    if delta[1] == 0 and math.isclose(start[1], round(start[1]), abs_tol=1e-9, rel_tol=0):
        offsets.append((0, -1))

    def allowed(x, y):
        return all(0 <= x + dx < width and 0 <= y + dy < height and mask[y + dy, x + dx]
                   for dx, dy in offsets)

    if not allowed(column, row) or not allowed(end_column, end_row):
        return reason
    step_x, step_y = int(np.sign(delta[0])), int(np.sign(delta[1]))
    next_x = (column + (step_x > 0) - start[0]) / delta[0] if step_x else math.inf
    next_y = (row + (step_y > 0) - start[1]) / delta[1] if step_y else math.inf
    interval_x = abs(1 / delta[0]) if step_x else math.inf
    interval_y = abs(1 / delta[1]) if step_y else math.inf
    # Endpoint cells were checked above. Cross only boundaries strictly before
    # the endpoint, avoiding an extra step past a negative-direction endpoint.
    while min(next_x, next_y) < 1 - 1e-12:
        if math.isclose(next_x, next_y, rel_tol=1e-12, abs_tol=1e-12):
            # At a corner, check both side cells as well as the diagonal cell.
            if not allowed(column + step_x, row) or not allowed(column, row + step_y):
                return reason
            column, row = column + step_x, row + step_y
            next_x, next_y = next_x + interval_x, next_y + interval_y
        elif next_x < next_y:
            column += step_x
            next_x += interval_x
        else:
            row += step_y
            next_y += interval_y
        if not allowed(column, row):
            return reason
    return None
