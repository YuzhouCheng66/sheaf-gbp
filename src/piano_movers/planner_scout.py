"""Exact-collision configuration-space scout for non-monotone piano moving.

This module is deliberately a training-data tool, not part of the deployed
policy.  It searches a discretized payload configuration space
``(x, y, theta)`` twice: once with route-axis retreat forbidden and once with
retreat allowed.  A barrier is admitted only when the unrestricted path also
passes a continuous swept-collision audit.  Search exhaustion is kept distinct
from a node-expansion budget, so a timeout can never be reported as evidence
that backtracking is necessary.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2, EnvConfig
from .env import (
    PianoBatch,
    collision_clearance_margin,
    collision_mask,
)


PLANNER_SCOUT_SCHEMA = "piano-movers-se2-backtracking-scout-v1"


@dataclass(frozen=True)
class BarrierCase:
    """One deterministic two-wall geometry for configuration-space search."""

    wall_x: tuple[float, float]
    gap_y: tuple[float, float]
    gap_half: tuple[float, float]
    wall_slab: float = 0.055
    source: tuple[float, float, float] = (-2.65, 0.0, 0.0)
    goal: tuple[float, float, float] = (2.65, 0.0, 0.0)
    world_x: float = 3.90
    world_y: float = 1.60
    payload_shape: str = "T"
    obstacle_rects: tuple[tuple[float, float, float, float], ...] = ()

    def validate(self) -> None:
        if len(self.wall_x) != 2 or self.wall_x[0] >= self.wall_x[1]:
            raise ValueError("wall_x must contain two increasing coordinates")
        if len(self.gap_y) != 2 or len(self.gap_half) != 2:
            raise ValueError("gap_y and gap_half must each contain two values")
        if min(self.gap_half) <= 0.03:
            raise ValueError("gap_half must exceed the collision reserve 0.03")
        if self.wall_slab <= 0.0:
            raise ValueError("wall_slab must be positive")
        if self.source[0] >= self.goal[0]:
            raise ValueError("the v1 scout requires a left-to-right mission")
        if not (-self.world_x < self.source[0] < self.goal[0] < self.world_x):
            raise ValueError("source and goal must lie strictly inside world_x")
        if any(abs(value) >= self.world_y for value in (*self.gap_y, self.source[1], self.goal[1])):
            raise ValueError("gap/source/goal y must lie strictly inside world_y")
        for obstacle in self.obstacle_rects:
            if len(obstacle) != 4:
                raise ValueError("each obstacle must be (x_min, x_max, y_min, y_max)")
            x0, x1, y0, y1 = obstacle
            if not (x0 < x1 and y0 < y1):
                raise ValueError("obstacle bounds must be strictly increasing")
            if x0 < -self.world_x or x1 > self.world_x:
                raise ValueError("obstacle x bounds must lie inside the world")
            if y0 < -self.world_y or y1 > self.world_y:
                raise ValueError("obstacle y bounds must lie inside the world")


@dataclass(frozen=True)
class GridSpec:
    """Uniform state lattice used by both matched searches."""

    x_min: float = -2.70
    x_max: float = 2.70
    x_step: float = 0.04
    y_min: float = -1.04
    y_max: float = 1.04
    y_step: float = 0.04
    theta_bins: int = 90
    rotation_cost_radius: float = 0.35
    occupancy_batch_size: int = 262_144
    max_expansions: int = 3_000_000
    # The 26-neighbour lattice permits coupled translate/rotate primitives.
    diagonal_primitives: bool = True

    def validate(self) -> None:
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("grid bounds must be increasing")
        if self.x_step <= 0.0 or self.y_step <= 0.0:
            raise ValueError("grid steps must be positive")
        if self.theta_bins < 12:
            raise ValueError("theta_bins must be at least 12")
        if self.rotation_cost_radius <= 0.0:
            raise ValueError("rotation_cost_radius must be positive")
        if self.occupancy_batch_size <= 0 or self.max_expansions <= 0:
            raise ValueError("batch size and expansion budget must be positive")


@dataclass(frozen=True)
class SweepSpec:
    """Resolution of the post-search continuous collision audit."""

    translation_resolution: float = 0.010
    angular_resolution: float = math.radians(1.0)
    clearance_epsilon: float = 0.0

    def validate(self) -> None:
        if self.translation_resolution <= 0.0 or self.angular_resolution <= 0.0:
            raise ValueError("sweep resolutions must be positive")
        if self.clearance_epsilon < 0.0:
            raise ValueError("clearance_epsilon must be non-negative")


@dataclass(frozen=True)
class SearchResult:
    """One planner result with fail-closed status semantics."""

    status: str
    monotone_x: bool
    expansions: int
    path_cost: float | None
    path: tuple[tuple[float, float, float], ...]
    generated_nodes: int = 0
    lazy_replans: int = 0
    blocked_directed_edges: int = 0
    swept_valid: bool | None = None
    minimum_clearance: float | None = None
    failing_edge: int | None = None
    cumulative_backtrack: float | None = None
    maximum_backtrack_excursion: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


@dataclass(frozen=True)
class BacktrackingGate:
    """Strict admission thresholds for a backtracking-required map."""

    minimum_cumulative_backtrack: float = 0.15
    minimum_excursion: float = 0.15
    minimum_clearance: float = 0.0

    def validate(self) -> None:
        if self.minimum_cumulative_backtrack <= 0.0 or self.minimum_excursion <= 0.0:
            raise ValueError("backtrack thresholds must be positive")
        if self.minimum_clearance < 0.0:
            raise ValueError("minimum_clearance must be non-negative")


@dataclass(frozen=True)
class ScoutResult:
    case: BarrierCase
    grid: GridSpec
    sweep: SweepSpec
    gate: BacktrackingGate
    monotone: SearchResult
    unrestricted: SearchResult
    admitted: bool
    rejection_reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PLANNER_SCOUT_SCHEMA,
            "case": asdict(self.case),
            "grid": asdict(self.grid),
            "sweep": asdict(self.sweep),
            "gate": asdict(self.gate),
            "monotone": asdict(self.monotone),
            "unrestricted": asdict(self.unrestricted),
            "admitted": self.admitted,
            "rejection_reasons": list(self.rejection_reasons),
        }


class PoseGrid:
    """Index conversion and physically scaled lattice primitives."""

    def __init__(self, spec: GridSpec):
        spec.validate()
        self.spec = spec
        self.x = _inclusive_axis(spec.x_min, spec.x_max, spec.x_step)
        self.y = _inclusive_axis(spec.y_min, spec.y_max, spec.y_step)
        self.theta = np.linspace(-math.pi, math.pi, spec.theta_bins, endpoint=False)
        self.shape = (len(self.x), len(self.y), len(self.theta))
        self.size = int(np.prod(self.shape))
        self._yz = self.shape[1] * self.shape[2]

        offsets: list[tuple[int, int, int]] = []
        if spec.diagonal_primitives:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dt in (-1, 0, 1):
                        if (dx, dy, dt) != (0, 0, 0):
                            offsets.append((dx, dy, dt))
        else:
            offsets = [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ]
        theta_step = 2.0 * math.pi / spec.theta_bins
        self.primitives = tuple(
            (
                dx,
                dy,
                dt,
                math.sqrt(
                    (dx * spec.x_step) ** 2
                    + (dy * spec.y_step) ** 2
                    + (dt * theta_step * spec.rotation_cost_radius) ** 2
                ),
            )
            for dx, dy, dt in offsets
        )

    def flatten(self, ix: int, iy: int, it: int) -> int:
        return (ix * self.shape[1] + iy) * self.shape[2] + (it % self.shape[2])

    def unravel(self, index: int) -> tuple[int, int, int]:
        ix, remainder = divmod(index, self._yz)
        iy, it = divmod(remainder, self.shape[2])
        return ix, iy, it

    def nearest(self, pose: Sequence[float]) -> tuple[int, int, int]:
        ix = int(np.argmin(np.abs(self.x - float(pose[0]))))
        iy = int(np.argmin(np.abs(self.y - float(pose[1]))))
        wrapped = math.atan2(math.sin(float(pose[2])), math.cos(float(pose[2])))
        delta = np.arctan2(np.sin(self.theta - wrapped), np.cos(self.theta - wrapped))
        it = int(np.argmin(np.abs(delta)))
        return ix, iy, it

    def pose(self, index: int) -> tuple[float, float, float]:
        ix, iy, it = self.unravel(index)
        return float(self.x[ix]), float(self.y[iy]), float(self.theta[it])


def _inclusive_axis(lower: float, upper: float, step: float) -> np.ndarray:
    count = int(math.floor((upper - lower) / step + 1e-9)) + 1
    values = lower + step * np.arange(count, dtype=np.float64)
    if values[-1] < upper - 0.25 * step:
        values = np.append(values, upper)
    return values


def env_config_for_case(case: BarrierCase, *, device: str) -> EnvConfig:
    case.validate()
    return EnvConfig(
        n_agents=12,
        handle_layout=HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2,
        payload_shape=case.payload_shape,
        world_x=case.world_x,
        world_y=case.world_y,
        wall_slab=case.wall_slab,
        collision_substeps=4,
        source_x=case.source[0],
        goal_x=case.goal[0],
        bidirectional=False,
        round_trip=False,
        device=device,
    )


def scenario_for_states(states: Tensor, case: BarrierCase) -> PianoBatch:
    """Repeat one deterministic geometry to match a batch of states."""

    batch = states.shape[0]
    source_pose = states.new_tensor(case.source)
    goal_pose = states.new_tensor(case.goal)
    state0 = states.new_zeros((batch, 6))
    state0[:, :3] = source_pose
    obstacles = None
    if case.obstacle_rects:
        obstacles = states.new_tensor(case.obstacle_rects).view(1, -1, 4).expand(batch, -1, -1)
    return PianoBatch(
        state0=state0,
        goal=goal_pose.view(1, 3).expand(batch, -1),
        wall_x=states.new_tensor(case.wall_x).view(1, 2).expand(batch, -1),
        gap_y=states.new_tensor(case.gap_y).view(1, 2).expand(batch, -1),
        gap_half=states.new_tensor(case.gap_half).view(1, 2).expand(batch, -1),
        source=source_pose.view(1, 3).expand(batch, -1),
        leg=None,
        obstacle_rects=obstacles,
    )


@torch.no_grad()
def free_pose_mask(
    grid: PoseGrid,
    case: BarrierCase,
    *,
    device: torch.device | str,
) -> np.ndarray:
    """Evaluate every lattice node with the benchmark's exact SAT collision."""

    compute_device = torch.device(device)
    cfg = env_config_for_case(case, device=str(compute_device))
    output = np.empty(grid.size, dtype=np.bool_)
    theta_count = grid.shape[2]
    yz_count = grid.shape[1] * theta_count
    for start in range(0, grid.size, grid.spec.occupancy_batch_size):
        stop = min(start + grid.spec.occupancy_batch_size, grid.size)
        flat = torch.arange(start, stop, device=compute_device, dtype=torch.long)
        ix = torch.div(flat, yz_count, rounding_mode="floor")
        remainder = flat - ix * yz_count
        iy = torch.div(remainder, theta_count, rounding_mode="floor")
        it = remainder - iy * theta_count
        states = torch.zeros((stop - start, 6), device=compute_device)
        states[:, 0] = states.new_tensor(grid.x)[ix]
        states[:, 1] = states.new_tensor(grid.y)[iy]
        states[:, 2] = states.new_tensor(grid.theta)[it]
        scenario = scenario_for_states(states, case)
        output[start:stop] = (~collision_mask(states, scenario, cfg)).cpu().numpy()
    return output.reshape(grid.shape)


def _heuristic(grid: PoseGrid, node: tuple[int, int, int], goal: tuple[int, int, int]) -> float:
    dx = (node[0] - goal[0]) * grid.spec.x_step
    dy = (node[1] - goal[1]) * grid.spec.y_step
    dt_index = abs(node[2] - goal[2])
    dt_index = min(dt_index, grid.shape[2] - dt_index)
    dt = dt_index * (2.0 * math.pi / grid.shape[2]) * grid.spec.rotation_cost_radius
    return math.sqrt(dx * dx + dy * dy + dt * dt)


def search_pose_grid(
    grid: PoseGrid,
    free: np.ndarray,
    case: BarrierCase,
    *,
    monotone_x: bool,
    blocked_edges: set[tuple[int, int]] | None = None,
    max_expansions: int | None = None,
) -> SearchResult:
    """Run matched A* with or without route-axis retreat primitives."""

    if free.shape != grid.shape:
        raise ValueError("free mask shape does not match grid")
    forbidden = blocked_edges if blocked_edges is not None else set()
    expansion_limit = grid.spec.max_expansions if max_expansions is None else max_expansions
    if expansion_limit <= 0:
        return SearchResult("budget_exhausted", monotone_x, 0, None, ())
    start_node = grid.nearest(case.source)
    goal_node = grid.nearest(case.goal)
    if not bool(free[start_node]):
        return SearchResult("start_collision", monotone_x, 0, None, ())
    if not bool(free[goal_node]):
        return SearchResult("goal_collision", monotone_x, 0, None, ())

    start = grid.flatten(*start_node)
    goal = grid.flatten(*goal_node)
    g_score = np.full(grid.size, np.inf, dtype=np.float64)
    parent = np.full(grid.size, -1, dtype=np.int64)
    closed = np.zeros(grid.size, dtype=np.bool_)
    g_score[start] = 0.0
    queue: list[tuple[float, float, int]] = [(_heuristic(grid, start_node, goal_node), 0.0, start)]
    expansions = 0
    generated_nodes = 1

    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if closed[current] or current_cost > g_score[current] + 1e-12:
            continue
        closed[current] = True
        expansions += 1
        if current == goal:
            path_indices = [current]
            while path_indices[-1] != start:
                path_indices.append(int(parent[path_indices[-1]]))
            path_indices.reverse()
            return SearchResult(
                status="success",
                monotone_x=monotone_x,
                expansions=expansions,
                path_cost=float(current_cost),
                path=tuple(grid.pose(index) for index in path_indices),
                generated_nodes=generated_nodes,
            )
        if expansions >= expansion_limit:
            return SearchResult(
                "budget_exhausted",
                monotone_x,
                expansions,
                None,
                (),
                generated_nodes=generated_nodes,
            )

        ix, iy, it = grid.unravel(current)
        for dx, dy, dt, primitive_cost in grid.primitives:
            if monotone_x and dx < 0:
                continue
            nx = ix + dx
            ny = iy + dy
            nt = (it + dt) % grid.shape[2]
            if nx < 0 or nx >= grid.shape[0] or ny < 0 or ny >= grid.shape[1]:
                continue
            if not free[nx, ny, nt]:
                continue
            neighbour = grid.flatten(nx, ny, nt)
            if (current, neighbour) in forbidden:
                continue
            if closed[neighbour]:
                continue
            tentative = current_cost + primitive_cost
            if tentative + 1e-12 >= g_score[neighbour]:
                continue
            g_score[neighbour] = tentative
            parent[neighbour] = current
            estimate = tentative + _heuristic(grid, (nx, ny, nt), goal_node)
            heapq.heappush(queue, (estimate, tentative, neighbour))
            generated_nodes += 1

    return SearchResult(
        "unreachable",
        monotone_x,
        expansions,
        None,
        (),
        generated_nodes=generated_nodes,
    )


def backtracking_metrics(path: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Return cumulative retreat and largest route-coordinate drawdown."""

    if len(path) < 2:
        return 0.0, 0.0
    route = np.asarray([pose[0] for pose in path], dtype=np.float64)
    increments = np.diff(route)
    cumulative = float(np.maximum(-increments, 0.0).sum())
    running_peak = np.maximum.accumulate(route)
    excursion = float((running_peak - route).max())
    return cumulative, excursion


def _interpolate_path(path: Sequence[Sequence[float]], sweep: SweepSpec) -> tuple[np.ndarray, np.ndarray]:
    samples: list[tuple[float, float, float]] = []
    edge_ids: list[int] = []
    for edge_id, (left, right) in enumerate(zip(path[:-1], path[1:])):
        left_arr = np.asarray(left, dtype=np.float64)
        right_arr = np.asarray(right, dtype=np.float64)
        delta_xy = right_arr[:2] - left_arr[:2]
        delta_theta = math.atan2(
            math.sin(right_arr[2] - left_arr[2]),
            math.cos(right_arr[2] - left_arr[2]),
        )
        steps = max(
            1,
            int(math.ceil(np.linalg.norm(delta_xy) / sweep.translation_resolution)),
            int(math.ceil(abs(delta_theta) / sweep.angular_resolution)),
        )
        for index in range(1, steps + 1):
            fraction = index / steps
            xy = left_arr[:2] + fraction * delta_xy
            theta = math.atan2(
                math.sin(left_arr[2] + fraction * delta_theta),
                math.cos(left_arr[2] + fraction * delta_theta),
            )
            samples.append((float(xy[0]), float(xy[1]), theta))
            edge_ids.append(edge_id)
    return np.asarray(samples, dtype=np.float64), np.asarray(edge_ids, dtype=np.int64)


@torch.no_grad()
def audit_pose_path(
    path: Sequence[Sequence[float]],
    case: BarrierCase,
    sweep: SweepSpec,
    *,
    device: torch.device | str,
) -> tuple[bool, float | None, int | None]:
    """Continuously sample every lattice edge and apply exact SAT + clearance."""

    sweep.validate()
    if len(path) < 2:
        return False, None, None
    invalid_edges, minimum = invalid_pose_path_edges(path, case, sweep, device=device)
    if invalid_edges:
        return False, minimum, invalid_edges[0]
    return True, minimum, None


@torch.no_grad()
def invalid_pose_path_edges(
    path: Sequence[Sequence[float]],
    case: BarrierCase,
    sweep: SweepSpec,
    *,
    device: torch.device | str,
) -> tuple[tuple[int, ...], float | None]:
    """Return every sampled-invalid transition in one candidate path.

    This is the collision oracle used by the optimistic LazySP loop below.
    A sampled collision is conclusive evidence that an edge is invalid.  An
    apparently valid edge remains optimistic until the final recursive
    continuous certificate in :mod:`planner_scout_protocol`.
    """

    sweep.validate()
    if len(path) < 2:
        return (), None
    samples, edge_ids = _interpolate_path(path, sweep)
    compute_device = torch.device(device)
    states = torch.zeros((samples.shape[0], 6), device=compute_device)
    states[:, :3] = torch.as_tensor(samples, dtype=states.dtype, device=compute_device)
    cfg = env_config_for_case(case, device=str(compute_device))
    scenario = scenario_for_states(states, case)
    hit = collision_mask(states, scenario, cfg)
    clearance = collision_clearance_margin(states, scenario, cfg)
    invalid = hit | (clearance <= sweep.clearance_epsilon)
    minimum = float(clearance.amin().item())
    if not bool(invalid.any()):
        return (), minimum
    invalid_samples = torch.where(invalid)[0].detach().cpu().numpy()
    invalid_edge_ids = tuple(
        int(value) for value in np.unique(edge_ids[invalid_samples]).tolist()
    )
    return invalid_edge_ids, minimum


def search_collision_validated_pose_grid(
    grid: PoseGrid,
    free: np.ndarray,
    case: BarrierCase,
    sweep: SweepSpec,
    *,
    monotone_x: bool,
    device: torch.device | str,
) -> SearchResult:
    """LazySP over the lattice with exact-SAT sampled edge rejection.

    Node-only occupancy admits transitions whose endpoints are free but whose
    swept payload crosses a wall.  We therefore plan optimistically, evaluate
    every edge of the proposed path in one GPU batch, block all conclusively
    invalid directed edges (and their reverse), and replan.  Exhausting this
    optimistic supergraph is a valid lattice-relative infeasibility result;
    hitting the aggregate expansion budget is not.
    """

    blocked: set[tuple[int, int]] = set()
    total_expansions = 0
    total_generated = 0
    replans = 0
    while total_expansions < grid.spec.max_expansions:
        remaining = grid.spec.max_expansions - total_expansions
        candidate = search_pose_grid(
            grid,
            free,
            case,
            monotone_x=monotone_x,
            blocked_edges=blocked,
            max_expansions=remaining,
        )
        total_expansions += candidate.expansions
        total_generated += candidate.generated_nodes
        if not candidate.succeeded:
            status = candidate.status
            if status == "unreachable":
                return SearchResult(
                    status,
                    monotone_x,
                    total_expansions,
                    None,
                    (),
                    generated_nodes=total_generated,
                    lazy_replans=replans,
                    blocked_directed_edges=len(blocked),
                )
            return SearchResult(
                "budget_exhausted",
                monotone_x,
                total_expansions,
                None,
                (),
                generated_nodes=total_generated,
                lazy_replans=replans,
                blocked_directed_edges=len(blocked),
            )

        invalid_edges, clearance = invalid_pose_path_edges(
            candidate.path,
            case,
            sweep,
            device=device,
        )
        if not invalid_edges:
            cumulative, excursion = backtracking_metrics(candidate.path)
            return SearchResult(
                status="success",
                monotone_x=monotone_x,
                expansions=total_expansions,
                path_cost=candidate.path_cost,
                path=candidate.path,
                generated_nodes=total_generated,
                lazy_replans=replans,
                blocked_directed_edges=len(blocked),
                swept_valid=True,
                minimum_clearance=clearance,
                failing_edge=None,
                cumulative_backtrack=cumulative,
                maximum_backtrack_excursion=excursion,
            )

        path_nodes = [grid.flatten(*grid.nearest(pose)) for pose in candidate.path]
        previous_count = len(blocked)
        for edge_id in invalid_edges:
            left = path_nodes[edge_id]
            right = path_nodes[edge_id + 1]
            blocked.add((left, right))
            blocked.add((right, left))
        if len(blocked) == previous_count:
            return SearchResult(
                "validation_stalled",
                monotone_x,
                total_expansions,
                None,
                (),
                generated_nodes=total_generated,
                lazy_replans=replans,
                blocked_directed_edges=len(blocked),
            )
        replans += 1

    return SearchResult(
        "budget_exhausted",
        monotone_x,
        total_expansions,
        None,
        (),
        generated_nodes=total_generated,
        lazy_replans=replans,
        blocked_directed_edges=len(blocked),
    )


def classify_backtracking_case(
    case: BarrierCase,
    grid_spec: GridSpec,
    sweep: SweepSpec,
    gate: BacktrackingGate,
    *,
    device: torch.device | str,
) -> ScoutResult:
    """Run matched planners and apply a strict backtracking-required gate."""

    case.validate()
    grid_spec.validate()
    sweep.validate()
    gate.validate()
    grid = PoseGrid(grid_spec)
    free = free_pose_mask(grid, case, device=device)
    unrestricted = search_collision_validated_pose_grid(
        grid,
        free,
        case,
        sweep,
        monotone_x=False,
        device=device,
    )
    if unrestricted.status == "success":
        monotone = search_collision_validated_pose_grid(
            grid,
            free,
            case,
            sweep,
            monotone_x=True,
            device=device,
        )
    else:
        monotone = SearchResult(
            "not_run_unrestricted_failed",
            True,
            0,
            None,
            (),
        )

    reasons: list[str] = []
    if monotone.status != "unreachable":
        reasons.append(f"monotone_status_{monotone.status}")
    if unrestricted.status != "success" or unrestricted.swept_valid is not True:
        reasons.append(f"unrestricted_status_{unrestricted.status}")
    if unrestricted.minimum_clearance is None or unrestricted.minimum_clearance <= gate.minimum_clearance:
        reasons.append("insufficient_clearance")
    if (
        unrestricted.cumulative_backtrack is None
        or unrestricted.cumulative_backtrack < gate.minimum_cumulative_backtrack
    ):
        reasons.append("insufficient_cumulative_backtrack")
    if (
        unrestricted.maximum_backtrack_excursion is None
        or unrestricted.maximum_backtrack_excursion < gate.minimum_excursion
    ):
        reasons.append("insufficient_backtrack_excursion")
    return ScoutResult(
        case=case,
        grid=grid_spec,
        sweep=sweep,
        gate=gate,
        monotone=monotone,
        unrestricted=unrestricted,
        admitted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_barrier_cases(
    *,
    wall_x_abs: Iterable[float],
    wall_slab: Iterable[float],
    gap_half: Iterable[float],
    gap_offset: Iterable[float],
    source_x: float = -2.65,
    goal_x: float = 2.65,
) -> tuple[BarrierCase, ...]:
    """Cartesian candidate family with oppositely offset consecutive gaps."""

    cases = []
    for wall in wall_x_abs:
        for slab in wall_slab:
            if slab >= wall:
                continue
            for half in gap_half:
                for offset in gap_offset:
                    cases.append(
                        BarrierCase(
                            wall_x=(-float(wall), float(wall)),
                            gap_y=(-float(offset), float(offset)),
                            gap_half=(float(half), float(half)),
                            wall_slab=float(slab),
                            source=(float(source_x), -float(offset), 0.0),
                            goal=(float(goal_x), float(offset), 0.0),
                        )
                    )
    return tuple(cases)


def enumerate_serpentine_cases(
    *,
    wall_end_abs: Iterable[float] = (1.15,),
    lane_inner_y: Iterable[float] = (0.60,),
    wall_thickness: Iterable[float] = (0.10,),
    world_x: float = 3.90,
    world_y: float = 2.00,
    source_x: float = -2.65,
    goal_x: float = 2.65,
    lane_center_y: float = 1.30,
) -> tuple[BarrierCase, ...]:
    """Horizontal S-corridors with a topological route-retreat guarantee.

    The lower obstacle is attached to the left world boundary and can only be
    rounded at its right endpoint.  The upper obstacle is attached to the
    right boundary and can only be rounded at its left endpoint.  A route from
    the lower-left source to the upper-right goal must therefore progress
    right, retreat left, and progress right again.  Oversized legacy wall gaps
    disable the original two vertical walls without special collision code.
    """

    cases: list[BarrierCase] = []
    disabled_gap = world_y + 1.0
    for endpoint in wall_end_abs:
        if endpoint <= 0.0 or endpoint >= world_x:
            raise ValueError("wall_end_abs must lie in (0, world_x)")
        for inner in lane_inner_y:
            if inner <= 0.0 or lane_center_y <= inner:
                raise ValueError("lane geometry must satisfy 0 < inner < lane_center_y")
            for thickness in wall_thickness:
                if thickness <= 0.0 or inner + thickness >= world_y:
                    raise ValueError("wall thickness leaves no outer lane")
                cases.append(
                    BarrierCase(
                        wall_x=(-world_x + 0.20, world_x - 0.20),
                        gap_y=(0.0, 0.0),
                        gap_half=(disabled_gap, disabled_gap),
                        wall_slab=0.055,
                        source=(source_x, -lane_center_y, 0.0),
                        goal=(goal_x, lane_center_y, 0.0),
                        world_x=world_x,
                        world_y=world_y,
                        payload_shape="T",
                        obstacle_rects=(
                            (-world_x, endpoint, -inner - thickness, -inner),
                            (-endpoint, world_x, inner, inner + thickness),
                        ),
                    )
                )
    return tuple(cases)


__all__ = [
    "BacktrackingGate",
    "BarrierCase",
    "GridSpec",
    "PLANNER_SCOUT_SCHEMA",
    "PoseGrid",
    "ScoutResult",
    "SearchResult",
    "SweepSpec",
    "audit_pose_path",
    "invalid_pose_path_edges",
    "backtracking_metrics",
    "classify_backtracking_case",
    "enumerate_barrier_cases",
    "enumerate_serpentine_cases",
    "env_config_for_case",
    "free_pose_mask",
    "scenario_for_states",
    "search_collision_validated_pose_grid",
    "search_pose_grid",
    "sha256_file",
]
