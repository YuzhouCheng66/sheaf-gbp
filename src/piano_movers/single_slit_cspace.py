"""Route-free online C-space planning for one finite single-slit wall.

The planner consumes compound-body rectangles and metric wall geometry.  It
does not consume a shape name, training route, teacher state, or policy hidden
state.  Eliminating y as an interval fiber leaves a small (x, theta) graph;
the returned path is recomputed from geometry and can be queried from the
current pose without a carried progress index.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class SliceField:
    x: np.ndarray
    theta: np.ndarray
    slice_min_y: np.ndarray
    slice_max_y: np.ndarray
    body_min_y: np.ndarray
    body_max_y: np.ndarray
    world_y: float


def build_slice_field(
    rectangles: np.ndarray,
    *,
    x_limit: float,
    x_step: float,
    theta_bins: int,
    slab_half_thickness: float,
    world_y: float,
) -> SliceField:
    rectangles = np.asarray(rectangles, dtype=np.float64)
    if rectangles.ndim != 3 or rectangles.shape[1:] != (4, 2):
        raise ValueError("rectangles must have shape [R,4,2]")
    x = np.arange(-x_limit, x_limit + 0.25 * x_step, x_step, dtype=np.float64)
    theta = np.linspace(-math.pi, math.pi, theta_bins, endpoint=False, dtype=np.float64)
    nx = len(x)
    slice_min = np.full((nx, theta_bins), np.inf, dtype=np.float32)
    slice_max = np.full((nx, theta_bins), -np.inf, dtype=np.float32)
    body_min = np.empty(theta_bins, dtype=np.float32)
    body_max = np.empty(theta_bins, dtype=np.float32)
    for theta_index, angle in enumerate(theta):
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        rotation_transpose = np.asarray(
            ((cosine, sine), (-sine, cosine)), dtype=np.float64
        )
        rotated = rectangles @ rotation_transpose
        local_min = np.full(nx, np.inf, dtype=np.float64)
        local_max = np.full(nx, -np.inf, dtype=np.float64)
        all_vertices = rotated.reshape(-1, 2)
        for rectangle in rotated:
            world_vertex_x = rectangle[:, 0, None] + x[None, :]
            inside = (world_vertex_x >= -slab_half_thickness) & (
                world_vertex_x <= slab_half_thickness
            )
            for vertex_index in range(4):
                mask = inside[vertex_index]
                local_min[mask] = np.minimum(local_min[mask], rectangle[vertex_index, 1])
                local_max[mask] = np.maximum(local_max[mask], rectangle[vertex_index, 1])
            for edge_index in range(4):
                left = rectangle[edge_index]
                right = rectangle[(edge_index + 1) % 4]
                delta_x = right[0] - left[0]
                if abs(delta_x) < 1.0e-14:
                    continue
                for face_x in (-slab_half_thickness, slab_half_thickness):
                    fraction = (face_x - x - left[0]) / delta_x
                    mask = (fraction >= 0.0) & (fraction <= 1.0)
                    intersection_y = left[1] + fraction * (right[1] - left[1])
                    local_min[mask] = np.minimum(local_min[mask], intersection_y[mask])
                    local_max[mask] = np.maximum(local_max[mask], intersection_y[mask])
        slice_min[:, theta_index] = local_min
        slice_max[:, theta_index] = local_max
        body_min[theta_index] = all_vertices[:, 1].min()
        body_max[theta_index] = all_vertices[:, 1].max()
    return SliceField(x, theta, slice_min, slice_max, body_min, body_max, world_y)


def admissible_y_intervals(
    field: SliceField,
    effective_half_aperture: float,
    *,
    boundary_safety: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = field.slice_min_y.shape
    lower = np.broadcast_to(
        -field.world_y + boundary_safety - field.body_min_y, shape
    ).copy()
    upper = np.broadcast_to(
        field.world_y - boundary_safety - field.body_max_y, shape
    ).copy()
    intersects = np.isfinite(field.slice_min_y)
    lower[intersects] = np.maximum(
        lower[intersects], -effective_half_aperture - field.slice_min_y[intersects]
    )
    upper[intersects] = np.minimum(
        upper[intersects], effective_half_aperture - field.slice_max_y[intersects]
    )
    return lower, upper


def _reachable_path(
    lower: np.ndarray, upper: np.ndarray
) -> list[tuple[int, int]]:
    nx, ntheta = lower.shape
    start = (0, ntheta // 2)
    goal = (nx - 1, ntheta // 2)
    valid = lower < upper
    if not valid[start] or not (lower[start] < 0.0 < upper[start]):
        raise RuntimeError("online C-space start is infeasible")
    seen = np.zeros((nx, ntheta), dtype=np.uint8)
    parents = np.full(nx * ntheta, -1, dtype=np.int32)
    seen[start] = 1
    queue: deque[tuple[int, int]] = deque((start,))
    directions = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    reached = False
    while queue:
        current = queue.popleft()
        if current == goal:
            reached = True
            break
        current_lower = lower[current]
        current_upper = upper[current]
        for delta_x, delta_theta in directions:
            next_x = current[0] + delta_x
            if next_x < 0 or next_x >= nx:
                continue
            next_theta = (current[1] + delta_theta) % ntheta
            following = (next_x, next_theta)
            if seen[following] or not valid[following]:
                continue
            if max(current_lower, lower[following]) >= min(current_upper, upper[following]):
                continue
            seen[following] = 1
            parents[next_x * ntheta + next_theta] = current[0] * ntheta + current[1]
            queue.append(following)
    if not reached:
        raise RuntimeError("online C-space goal is unreachable at this resolution")
    start_flat = ntheta // 2
    current_flat = (nx - 1) * ntheta + ntheta // 2
    nodes: list[tuple[int, int]] = []
    while current_flat >= 0:
        nodes.append(divmod(current_flat, ntheta))
        if current_flat == start_flat:
            nodes.reverse()
            return nodes
        current_flat = int(parents[current_flat])
    raise RuntimeError("online C-space parent chain did not reach the start")


def _densify_pose_path(
    poses: list[tuple[float, float, float]],
    *,
    translation_step: float,
    angle_step: float,
) -> np.ndarray:
    """Interpolate safe fiber moves so a Markov pure-pursuit query cannot chatter.

    The interval-fiber construction can legitimately move a long distance in
    ``y`` at one fixed ``(x, theta)``.  Returning only the two endpoints makes
    nearest-pose control discontinuous: after taking one step toward the far
    endpoint, the current pose is again closest to the earlier endpoint and
    the controller reverses forever.  Dense points represent the same
    certified continuous segment without introducing a route/progress state.
    """

    if translation_step <= 0.0 or angle_step <= 0.0:
        raise ValueError("C-space interpolation steps must be positive")
    dense: list[tuple[float, float, float]] = [poses[0]]
    for start, finish in zip(poses[:-1], poses[1:]):
        delta_x = finish[0] - start[0]
        delta_y = finish[1] - start[1]
        delta_theta = math.atan2(
            math.sin(finish[2] - start[2]),
            math.cos(finish[2] - start[2]),
        )
        subdivisions = max(
            1,
            int(
                math.ceil(
                    max(
                        abs(delta_x) / translation_step,
                        abs(delta_y) / translation_step,
                        abs(delta_theta) / angle_step,
                    )
                )
            ),
        )
        for index in range(1, subdivisions + 1):
            fraction = index / subdivisions
            angle = start[2] + fraction * delta_theta
            angle = math.atan2(math.sin(angle), math.cos(angle))
            dense.append(
                (
                    start[0] + fraction * delta_x,
                    start[1] + fraction * delta_y,
                    angle,
                )
            )
    return np.asarray(dense, dtype=np.float32)


def plan_single_slit_cspace(
    rectangles: np.ndarray,
    *,
    effective_half_aperture: float,
    slab_half_thickness: float,
    world_y: float,
    x_limit: float = 0.70,
    x_step: float = 0.004,
    theta_bins: int = 720,
    planning_safety: float = 0.010,
) -> np.ndarray:
    """Compute one geometry-derived SE(2) path in wall-relative coordinates."""

    field = build_slice_field(
        rectangles,
        x_limit=x_limit,
        x_step=x_step,
        theta_bins=theta_bins,
        slab_half_thickness=slab_half_thickness + planning_safety,
        world_y=world_y,
    )
    lower, upper = admissible_y_intervals(
        field,
        effective_half_aperture - planning_safety,
        boundary_safety=planning_safety,
    )
    nodes = _reachable_path(lower, upper)
    poses: list[tuple[float, float, float]] = [(-x_limit, 0.0, 0.0)]
    previous_y = 0.0
    for current, following in zip(nodes[:-1], nodes[1:]):
        overlap_lower = max(lower[current], lower[following])
        overlap_upper = min(upper[current], upper[following])
        edge_y = float(0.5 * (overlap_lower + overlap_upper))
        if abs(edge_y - previous_y) > 1.0e-12:
            poses.append((float(field.x[current[0]]), edge_y, float(field.theta[current[1]])))
        poses.append((float(field.x[following[0]]), edge_y, float(field.theta[following[1]])))
        previous_y = edge_y
    poses.append((x_limit, 0.0, 0.0))
    path = _densify_pose_path(
        poses,
        translation_step=x_step,
        angle_step=2.0 * math.pi / float(theta_bins),
    )
    if path.ndim != 2 or path.shape[1] != 3 or not np.isfinite(path).all():
        raise RuntimeError("online C-space planner emitted a malformed path")
    return path


@lru_cache(maxsize=128)
def cached_single_slit_cspace_solution(
    rectangle_coordinates: tuple[float, ...],
    rectangle_count: int,
    effective_half_aperture: float,
    slab_half_thickness: float,
    world_y: float,
) -> tuple[np.ndarray, float]:
    """Cache a geometry-derived path and its certified planning reserve."""

    rectangles = np.asarray(rectangle_coordinates, dtype=np.float64).reshape(
        rectangle_count, 4, 2
    )
    # Prefer the largest certified margin.  A morphology at the mathematical
    # passage boundary may have no positive-clearance lattice route even
    # though a contact-feedback execution is feasible.  In that exceptional
    # case, progressively relax only the planning reserve; exact swept SAT
    # remains the execution authority and prevents any penetration.
    last_error: RuntimeError | None = None
    # Coarse angular grids can falsely erase a millimetre-scale passage: the
    # canonical T at 0.720 m is the important example.  Refine orientation
    # before giving up geometric margin.  This costs under a second once per
    # morphology/opening and is then cached for all agents and control ticks.
    attempts = (
        (720, 0.010),
        (720, 0.005),
        (720, 0.002),
        (1440, 0.002),
        (1440, 0.001),
        (1440, 0.0005),
        (1440, 0.0001),
        (1440, 0.0),
    )
    for theta_bins, planning_safety in attempts:
        try:
            path = plan_single_slit_cspace(
                rectangles,
                effective_half_aperture=effective_half_aperture,
                slab_half_thickness=slab_half_thickness,
                world_y=world_y,
                theta_bins=theta_bins,
                planning_safety=planning_safety,
            )
            break
        except RuntimeError as error:
            last_error = error
    else:
        assert last_error is not None
        raise last_error
    path.setflags(write=False)
    return path, planning_safety


def cached_single_slit_cspace_path(
    rectangle_coordinates: tuple[float, ...],
    rectangle_count: int,
    effective_half_aperture: float,
    slab_half_thickness: float,
    world_y: float,
) -> np.ndarray:
    """Backward-compatible path-only view of the cached online solution."""

    path, _ = cached_single_slit_cspace_solution(
        rectangle_coordinates,
        rectangle_count,
        effective_half_aperture,
        slab_half_thickness,
        world_y,
    )
    return path


__all__ = [
    "cached_single_slit_cspace_path",
    "cached_single_slit_cspace_solution",
    "plan_single_slit_cspace",
]
