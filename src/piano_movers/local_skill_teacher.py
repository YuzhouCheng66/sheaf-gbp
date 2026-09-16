"""Training-only local-target planning and continuously checked force execution.

The configuration-space graph has (x, theta) nodes with continuous feasible y
intervals. Intervals are intersected across every supplied slit; search allows
both signs of translation and rotation. Arbitrary measured start poses and
position-only goals are supported. Graph connectivity proposes a path; only a
separate recursive SAT certificate admits it. Exhaustion is lattice-relative.
No path, node index, or future goal is an input to the deployed learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math

import numpy as np
import torch
from torch import Tensor

from .cem_teacher import feasible_forces_for_absolute_wrench
from .config import EnvConfig
from .env import PianoBatch, collision_clearance_margin, step_dynamics, wrap_angle
from .payload_shapes import payload_rectangles_for_config
from .single_slit_cspace import build_slice_field


@dataclass(frozen=True)
class LocalTeacherConfig:
    x_step: float = 0.01
    theta_bins: int = 360
    planning_margin_m: float = 0.001
    certificate_margin_m: float = 1.0e-6
    certificate_depth: int = 18
    position_tolerance_m: float = 0.015
    execution_position_tolerance_m: float = 0.020
    angle_tolerance_rad: float = 0.020
    # Training-only terminal predicate: the whole body must satisfy
    # nx*x + ny*y > offset. No plane or route index is a policy feature.
    terminal_halfspace: tuple[float, float, float] | None = None
    search_padding_m: float = 0.85
    max_expansions: int = 500_000
    max_lazy_replans: int = 12
    max_speed_mps: float = 0.35
    max_omega_radps: float = 0.8
    force_fraction: float = 0.85


@dataclass(frozen=True)
class ContinuousCertificate:
    passed: bool
    reason: str | None
    failed_segment: int | None
    minimum_sampled_clearance_m: float
    queries: int
    maximum_depth: int


@dataclass(frozen=True)
class LocalPath:
    status: str
    poses: np.ndarray | None
    expansions: int
    lazy_replans: int
    certificate: ContinuousCertificate | None
    planning_margin_m: float | None = None


def repeat_scene(scene: PianoBatch, count: int, reference: Tensor) -> PianoBatch:
    def rep(value):
        return None if value is None else value[:1].to(reference).expand(count, *value.shape[1:])

    return PianoBatch(
        state0=rep(scene.state0),
        source=rep(scene.source),
        goal=rep(scene.goal),
        wall_x=rep(scene.wall_x),
        gap_y=rep(scene.gap_y),
        gap_half=rep(scene.gap_half),
        leg=rep(scene.leg),
        obstacle_rects=rep(scene.obstacle_rects),
    )


@torch.no_grad()
def certify_pose_path(
    path: Tensor,
    scene: PianoBatch,
    cfg: EnvConfig,
    options: LocalTeacherConfig = LocalTeacherConfig(),
) -> ContinuousCertificate:
    """Conservative continuous rigid-motion certificate, beyond sampled SAT.

    A separating axis at an interval endpoint remains separating if its gap
    exceeds ||delta_xy|| + body_radius * |delta_theta| plus the reserve.
    Otherwise bisect. Failure to certify at the depth limit is not success.
    Double precision reduces numerical error; this is not interval arithmetic.
    """

    if path.ndim != 2 or path.shape[-1] not in (3, 6) or len(path) < 2:
        raise ValueError("path must contain at least two pose3/state6 rows")
    q = path[:, :3].to(dtype=torch.float64)
    if not bool(torch.isfinite(q).all()):
        raise ValueError("path must be finite")
    if options.certificate_margin_m <= 0 or options.certificate_depth < 1:
        raise ValueError("certificate requires positive margin and subdivision depth")
    rects = payload_rectangles_for_config(cfg, q.device).to(q)
    radius = torch.linalg.vector_norm(rects.reshape(-1, 2), dim=-1).amax()
    queries = 0
    minimum = math.inf

    def clearance(poses: Tensor) -> Tensor:
        nonlocal queries, minimum
        states = torch.cat((poses, torch.zeros_like(poses)), dim=-1)
        result = collision_clearance_margin(states, repeat_scene(scene, len(states), states), cfg)
        queries += len(states)
        minimum = min(minimum, float(result.min()))
        return result

    margins = clearance(q)
    unsafe = margins <= options.certificate_margin_m
    if bool(unsafe.any()):
        point = int(torch.where(unsafe)[0][0])
        return ContinuousCertificate(
            False, "sampled_clearance_violation", max(0, point - 1), minimum, queries, 0
        )
    left, right = q[:-1], q[1:]
    cl, cr = margins[:-1], margins[1:]
    indices = torch.arange(len(left), device=q.device)
    for depth in range(options.certificate_depth + 1):
        motion = (
            torch.linalg.vector_norm(right[:, :2] - left[:, :2], dim=-1)
            + radius * wrap_angle(right[:, 2] - left[:, 2]).abs()
        )
        certified = torch.maximum(cl, cr) > motion + options.certificate_margin_m
        if bool(certified.all()):
            return ContinuousCertificate(True, None, None, minimum, queries, depth)
        unresolved = ~certified
        left, right, cl, cr, indices = (v[unresolved] for v in (left, right, cl, cr, indices))
        if depth == options.certificate_depth:
            return ContinuousCertificate(
                False, "subdivision_limit", int(indices[0]), minimum, queries, depth
            )
        middle = (left + right) * 0.5
        middle[:, 2] = wrap_angle(left[:, 2] + 0.5 * wrap_angle(right[:, 2] - left[:, 2]))
        cm = clearance(middle)
        unsafe = cm <= options.certificate_margin_m
        if bool(unsafe.any()):
            first = int(torch.where(unsafe)[0][0])
            return ContinuousCertificate(
                False,
                "interior_clearance_violation",
                int(indices[first]),
                minimum,
                queries,
                depth + 1,
            )
        left, right, cl, cr, indices = (
            torch.cat((left, middle)),
            torch.cat((middle, right)),
            torch.cat((cl, cm)),
            torch.cat((cm, cr)),
            torch.cat((indices, indices)),
        )
    raise AssertionError("unreachable")


def _angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


def terminal_halfspace_margin(state: Tensor, cfg: EnvConfig, plane) -> Tensor:
    """Minimum signed body support relative to a training terminal plane."""
    if plane is None:
        return state.new_full((len(state),), torch.inf)
    normal = state.new_tensor(plane[:2])
    length = torch.linalg.vector_norm(normal)
    if not bool(torch.isfinite(state.new_tensor(plane)).all()) or float(length) <= 0:
        raise ValueError("terminal plane needs finite coefficients and a nonzero normal")
    normal = normal / length
    vertices = payload_rectangles_for_config(cfg, state.device).to(state).reshape(-1, 2)
    c, s = state[:, 2:3].cos(), state[:, 2:3].sin()
    x = state[:, 0:1] + c * vertices[:, 0] - s * vertices[:, 1]
    y = state[:, 1:2] + s * vertices[:, 0] + c * vertices[:, 1]
    return (normal[0] * x + normal[1] * y).amin(-1) - plane[2] / length


def plan_local_target(
    state: Tensor,
    goal: Tensor,
    scene: PianoBatch,
    cfg: EnvConfig,
    *,
    orientation_required: bool,
    options: LocalTeacherConfig = LocalTeacherConfig(),
) -> LocalPath:
    """Search one current-target problem; never access a waypoint sequence."""

    if state.shape != (1, 6) or goal.shape != (3,):
        raise ValueError("expected state [1,6] and goal [3]")
    if scene.obstacle_rects is not None and scene.obstacle_rects.numel():
        raise ValueError(
            "fiber search currently supports slit walls; generic obstacles need a different graph"
        )
    start = state[0, :3].detach().cpu().double().numpy()
    target = goal.detach().cpu().double().numpy()
    rects = payload_rectangles_for_config(cfg, "cpu").double().numpy()
    reference = state.detach().double()
    initial_clearance = float(
        collision_clearance_margin(reference, repeat_scene(scene, 1, reference), cfg)[0]
    )
    if initial_clearance <= options.certificate_margin_m:
        return LocalPath("initial_certificate_margin_violation", None, 0, 0, None)
    # Real student failures can lie much closer than the ordinary 1 mm
    # planning reserve. Keep the certification margin fixed, while admitting
    # their actual start to a proportionally thinner search tube.
    margin = min(options.planning_margin_m, 0.1 * initial_clearance)
    pad = options.search_padding_m
    xmin = max(-cfg.world_x, min(start[0], target[0]) - pad)
    xmax = min(cfg.world_x, max(start[0], target[0]) + pad)
    x = np.unique(np.r_[np.arange(xmin, xmax, options.x_step), start[0], target[0], xmax])
    angles = [float(_angle(start[2]))]
    if orientation_required:
        angles.append(float(_angle(target[2])))
    theta = np.unique(
        np.r_[np.linspace(-math.pi, math.pi, options.theta_bins, endpoint=False), angles]
    )
    nx, nt = len(x), len(theta)
    lower = np.full((nx, nt), -cfg.world_y, dtype=np.float64)
    upper = np.full((nx, nt), cfg.world_y, dtype=np.float64)
    for wall_x, gap_y, gap_half in zip(
        scene.wall_x[0].tolist(), scene.gap_y[0].tolist(), scene.gap_half[0].tolist()
    ):
        field = build_slice_field(
            rects,
            x_limit=pad,
            x_step=options.x_step,
            theta_bins=nt,
            slab_half_thickness=cfg.wall_slab + margin,
            world_y=cfg.world_y,
            x_values=x - wall_x,
            theta_values=theta,
        )
        lower = np.maximum(lower, -cfg.world_y + margin - field.body_min_y)
        upper = np.minimum(upper, cfg.world_y - margin - field.body_max_y)
        half = max(0.0, gap_half - 0.03) - margin
        intersects = np.isfinite(field.slice_min_y)
        lower[intersects] = np.maximum(
            lower[intersects], gap_y - half - field.slice_min_y[intersects]
        )
        upper[intersects] = np.minimum(
            upper[intersects], gap_y + half - field.slice_max_y[intersects]
        )
    # The world x boundary matters at any heading, including at connector poses.
    vertices = rects.reshape(-1, 2)
    vx = (
        np.cos(theta[:, None]) * vertices[None, :, 0]
        - np.sin(theta[:, None]) * vertices[None, :, 1]
    )
    valid = (
        (lower < upper)
        & (x[:, None] + vx.min(axis=1) > -cfg.world_x)
        & (x[:, None] + vx.max(axis=1) < cfg.world_x)
    )
    sx = int(np.argmin(abs(x - start[0])))
    st = int(np.argmin(abs(_angle(theta - start[2]))))
    if not valid[sx, st] or not lower[sx, st] <= start[1] <= upper[sx, st]:
        return LocalPath("start_outside_planning_reserve", None, 0, 0, None)
    dx = x - target[0]
    y_radius = np.sqrt(np.maximum(0.0, options.position_tolerance_m**2 - dx**2))
    goal_lower = np.maximum(lower, (target[1] - y_radius)[:, None])
    goal_upper = np.minimum(upper, (target[1] + y_radius)[:, None])
    terminal_valid = np.ones_like(valid)
    if options.terminal_halfspace is not None:
        a, b, offset = options.terminal_halfspace
        length = math.hypot(a, b)
        if not np.isfinite([a, b, offset]).all() or length <= 0:
            raise ValueError("invalid terminal halfspace")
        a, b, offset = a / length, b / length, offset / length
        vy = (np.sin(theta[:, None]) * vertices[None, :, 0]
              + np.cos(theta[:, None]) * vertices[None, :, 1])
        support = (a * vx + b * vy).min(axis=1)
        required_y = offset + options.planning_margin_m - a * x[:, None] - support[None]
        if abs(b) <= 1e-12:
            terminal_valid &= required_y < 0.
        elif b > 0:
            goal_lower = np.maximum(goal_lower, required_y / b)
        else:
            goal_upper = np.minimum(goal_upper, required_y / b)
    goal_mask = (
        valid & terminal_valid & (abs(dx)[:, None] <= options.position_tolerance_m)
        & (goal_lower <= goal_upper)
    )
    if orientation_required:
        goal_mask &= abs(_angle(theta - target[2]))[None, :] <= options.angle_tolerance_rad
    if not goal_mask.any():
        return LocalPath("no_goal_pose_at_resolution", None, 0, 0, None)
    start_id = sx * nt + st
    blocked: set[tuple[int, int]] = set()
    expansions_total = 0
    last_certificate = None
    for replan in range(options.max_lazy_replans + 1):
        costs = np.full(nx * nt, np.inf)
        parents = np.full(nx * nt, -1, dtype=np.int32)
        costs[start_id] = 0.0
        queue = [(0.0, 0.0, start_id)]
        reached = None
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            ix, it = divmod(node, nt)
            expansions_total += 1
            if expansions_total > options.max_expansions:
                return LocalPath(
                    "expansion_budget", None, expansions_total, replan, last_certificate
                )
            if goal_mask[ix, it]:
                reached = node
                break
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                jx, jt = ix + di, (it + dj) % nt
                if not 0 <= jx < nx or not valid[jx, jt]:
                    continue
                following = jx * nt + jt
                if (node, following) in blocked:
                    continue
                lo = max(lower[ix, it], lower[jx, jt])
                hi = min(upper[ix, it], upper[jx, jt])
                if lo >= hi:
                    continue
                step = math.hypot(x[jx] - x[ix], 0.35 * float(_angle(theta[jt] - theta[it])))
                next_cost = cost + step
                if next_cost >= costs[following]:
                    continue
                costs[following], parents[following] = next_cost, node
                gy = np.clip(target[1], lower[jx, jt], upper[jx, jt])
                heuristic = math.hypot(x[jx] - target[0], gy - target[1])
                if orientation_required:
                    heuristic = math.hypot(heuristic, 0.35 * float(_angle(theta[jt] - target[2])))
                heapq.heappush(queue, (next_cost + heuristic, next_cost, following))
        if reached is None:
            return LocalPath("lattice_exhausted", None, expansions_total, replan, last_certificate)
        nodes = [reached]
        while nodes[-1] != start_id:
            nodes.append(int(parents[nodes[-1]]))
        nodes.reverse()
        poses = [start.copy()]
        transitions = []
        previous_y = float(start[1])
        for a, b in zip(nodes[:-1], nodes[1:]):
            ia, ta = divmod(a, nt)
            ib, tb = divmod(b, nt)
            lo, hi = max(lower[ia, ta], lower[ib, tb]), min(upper[ia, ta], upper[ib, tb])
            edge_y = float(np.clip(previous_y, lo, hi))
            if abs(edge_y - previous_y) > 1.0e-10:
                poses.append(np.array([x[ia], edge_y, theta[ta]]))
                transitions.append((a, b))
            poses.append(np.array([x[ib], edge_y, theta[tb]]))
            transitions.append((a, b))
            previous_y = edge_y
        ix, it = divmod(reached, nt)
        terminal_y = float(np.clip(target[1], goal_lower[ix, it], goal_upper[ix, it]))
        poses.append(np.array([x[ix], terminal_y, theta[it]]))
        transitions.append((reached, reached))
        path = np.stack(poses)
        certificate = certify_pose_path(
            torch.from_numpy(path), scene, replace(cfg, device="cpu"), options
        )
        last_certificate = certificate
        if certificate.passed:
            return LocalPath("certified", path, expansions_total, replan, certificate, margin)
        failed = transitions[certificate.failed_segment]
        if failed[0] == failed[1]:
            goal_mask[ix, it] = False
        else:
            blocked.add(failed)
            blocked.add((failed[1], failed[0]))
    return LocalPath(
        "lazy_certificate_budget",
        None,
        expansions_total,
        options.max_lazy_replans,
        last_certificate,
    )


@torch.no_grad()
def executable_teacher_step(
    state: Tensor,
    path: Tensor,
    scene: PianoBatch,
    cfg: EnvConfig,
    options: LocalTeacherConfig = LocalTeacherConfig(),
) -> tuple[Tensor | None, Tensor | None, str]:
    """Choose a bounded-force, continuously certified path-following step.

    Nearest pose is recomputed from current physical state. There is no carried
    progress index. Larger lookaheads are accepted only when their actual force
    transition passes the certificate; braking is a labelled fallback.
    """

    delta = path[:, :2] - state[0, :2]
    distance = delta.square().sum(-1) + (0.35 * wrap_angle(path[:, 2] - state[0, 2])).square()
    nearest = int(distance.argmin())
    offsets = [32, 16, 8, 4, 2, 1, 0, -1, -2, -4]
    candidates = path[
        torch.tensor(
            [max(0, min(len(path) - 1, nearest + k)) for k in offsets], device=state.device
        )
    ]
    desired_xy = 4.0 * (candidates[:, :2] - state[0, :2])
    norm = torch.linalg.vector_norm(desired_xy, dim=-1, keepdim=True).clamp_min(1.0e-9)
    desired_xy *= (options.max_speed_mps / norm).clamp_max(1.0)
    desired_w = (4.0 * wrap_angle(candidates[:, 2] - state[0, 2])).clamp(
        -options.max_omega_radps, options.max_omega_radps
    )
    desired = torch.cat((desired_xy, desired_w[:, None]), dim=-1)
    desired = torch.cat((desired, torch.zeros_like(desired[:1])), dim=0)
    repeated = state.expand(len(desired), -1)
    decay = state.new_tensor(
        [1.0 - cfg.linear_damping * cfg.dt] * 2 + [1.0 - cfg.angular_damping * cfg.dt]
    )
    mass = state.new_tensor([cfg.mass, cfg.mass, cfg.inertia])
    wrench = (desired / decay - state[:, 3:]) * mass / cfg.dt
    forces, _, _ = feasible_forces_for_absolute_wrench(
        repeated, wrench, replace(cfg, max_force=cfg.max_force * options.force_fraction)
    )
    attempted = step_dynamics(repeated, forces, cfg)
    for index in range(len(desired)):
        if index < len(offsets):
            # Certify connection all the way to the reference, not just the
            # next force tick. One-step-safe corner cutting can leave the
            # feasible path tube and brake forever on the following tick.
            bridge = certify_pose_path(
                torch.stack((state[0, :3], candidates[index])), scene, cfg, options
            )
            if not bridge.passed:
                continue
        pair = torch.stack((state[0], attempted[index]))
        certificate = certify_pose_path(pair, scene, cfg, options)
        if certificate.passed:
            mode = (
                "brake" if index == len(offsets) else "advance" if offsets[index] > 0 else "rejoin"
            )
            return forces[index : index + 1], attempted[index : index + 1], mode
    return None, None, "no_certified_force_step"
