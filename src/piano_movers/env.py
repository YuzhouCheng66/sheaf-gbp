"""Vectorized PianoMovers-Force rigid-body benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from torch import Tensor

from .config import (
    HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2,
    HANDLE_LAYOUT_GLYPH_BOUNDARY_V4,
    HANDLE_LAYOUT_LEGACY_OUTRIGGER_V1,
    HANDLE_LAYOUT_PARAMETRIC_T_V3,
    HANDLE_LAYOUTS,
    STRICT_OBSERVATION_SCHEMA_LOCAL_CURRENT_SLIT_V2,
    EnvConfig,
)
from .linear_solve import solve_nonsingular
from .payload_shapes import (
    CANONICAL_T_MORPHOLOGY,
    payload_rectangles_for_config,
    payload_shape_rectangles,
    glyph_boundary_handle_points,
    t_morphology_from_config,
)

# The narrowest procedural slit has ``gap_half == 0.60`` before the exact
# collision model's 0.03 m clearance reserve.  Sweeping the compound T through
# that slit requires about 1.25 rad; keep another small angular margin so the
# target trajectory, rather than only its endpoint, is collision-free.
PASSAGE_TARGET_ANGLE = 1.30


@dataclass
class PianoBatch:
    state0: Tensor
    goal: Tensor
    wall_x: Tensor
    gap_y: Tensor
    gap_half: Tensor
    source: Tensor
    # Active round-trip leg: 0 = source -> goal, 1 = goal -> source.  ``None``
    # preserves the original one-way benchmark and old serialized scenarios.
    leg: Tensor | None = None
    # Optional axis-aligned obstacle rectangles ``[B, O, 4]`` encoded as
    # ``(x_min, x_max, y_min, y_max)``.  ``None`` preserves every legacy
    # two-wall scenario without allocating a padded obstacle tensor.
    obstacle_rects: Tensor | None = None


@dataclass(frozen=True)
class ValidatedGenericObstacleGeometry:
    """One immutable scenario's obstacle table validated for a hot rollout.

    Generic obstacle rectangles are fixed for an entire rollout.  Rechecking
    their ordered bounds with ``bool(cuda_tensor.any())`` at every observation
    forces a device-to-host synchronization even though neither the scenario
    nor its dtype/device changes.  This token binds the validated, converted
    rectangle tensor to the exact source tensor (including its mutation
    version), batch size, observation configuration, dtype, and device.  A
    caller can therefore reuse the validation without weakening the ordinary
    fail-closed public path.
    """

    source_obstacle_rects: Tensor | None
    source_version: int | None
    rects: Tensor | None
    batch_size: int
    obstacle_slots: int
    wall_sensor_range: float
    device: torch.device
    dtype: torch.dtype


def scenario_with_leg(scenario: PianoBatch, leg: Tensor | None) -> PianoBatch:
    """Return a shallow tensor-preserving scenario with a new mission leg."""

    return PianoBatch(
        state0=scenario.state0,
        goal=scenario.goal,
        wall_x=scenario.wall_x,
        gap_y=scenario.gap_y,
        gap_half=scenario.gap_half,
        source=scenario.source,
        leg=leg,
        obstacle_rects=scenario.obstacle_rects,
    )


def mission_leg(scenario: PianoBatch) -> Tensor:
    if scenario.leg is None:
        return torch.zeros(
            scenario.state0.shape[0], dtype=torch.long, device=scenario.state0.device
        )
    return scenario.leg.long()


def active_endpoints(scenario: PianoBatch) -> tuple[Tensor, Tensor]:
    """Return start and goal poses for each episode's active mission leg."""

    returning = mission_leg(scenario).bool().unsqueeze(-1)
    start = torch.where(returning, scenario.goal, scenario.source)
    goal = torch.where(returning, scenario.source, scenario.goal)
    return start, goal


def mission_frame_sign(scenario: PianoBatch) -> Tensor:
    """Return the fixed horizontal orientation of the complete mission.

    Unlike the active leg direction, this sign does not change when a
    round-trip episode switches from outbound to return.  It therefore defines
    one stable coordinate frame for all recurrent state in an episode.
    """

    direction = torch.sign(scenario.goal[:, 0] - scenario.source[:, 0])
    return torch.where(direction == 0, torch.ones_like(direction), direction)


def leg_progress(state: Tensor, scenario: PianoBatch) -> Tensor:
    """Direction-invariant progress from the active start to active goal."""

    start, goal = active_endpoints(scenario)
    route = goal[:, 0:2] - start[:, 0:2]
    displacement = state[:, 0:2] - start[:, 0:2]
    denom = route.square().sum(dim=-1).clamp_min(1e-6)
    return (displacement * route).sum(dim=-1).div(denom).clamp(0.0, 1.0)


def mission_progress(state: Tensor, scenario: PianoBatch, *, round_trip: bool) -> Tensor:
    progress = leg_progress(state, scenario)
    if not round_trip:
        return progress
    return (mission_leg(scenario).to(progress.dtype) + progress) / 2.0


def active_start_state(scenario: PianoBatch) -> Tensor:
    start, _ = active_endpoints(scenario)
    state = torch.zeros_like(scenario.state0)
    state[:, :3] = start
    return state


def sample_training_legs(
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    generator: torch.Generator | None = None,
) -> PianoBatch:
    """Expose both round-trip directions equally during supervised training."""

    if not cfg.round_trip:
        return scenario
    leg = torch.randint(
        0,
        2,
        (scenario.state0.shape[0],),
        device=scenario.state0.device,
        generator=generator,
    )
    return scenario_with_leg(scenario, leg)


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


_LEGACY_OUTRIGGER_HANDLE_POINTS: dict[int, tuple[tuple[float, float], ...]] = {
    4: ((-0.70, 0.48), (0.70, 0.48), (-0.28, -0.55), (0.28, -0.55)),
    6: (
        (-0.78, 0.48),
        (0.00, 0.58),
        (0.78, 0.48),
        (-0.32, -0.35),
        (0.32, -0.35),
        (0.00, -0.92),
    ),
    8: (
        (-0.78, 0.48),
        (-0.28, 0.60),
        (0.28, 0.60),
        (0.78, 0.48),
        (-0.34, 0.02),
        (0.34, 0.02),
        (-0.18, -0.86),
        (0.18, -0.86),
    ),
    10: (
        (-0.78, 0.48),
        (-0.46, 0.58),
        (0.00, 0.62),
        (0.46, 0.58),
        (0.78, 0.48),
        (-0.34, 0.08),
        (0.34, 0.08),
        (-0.24, -0.42),
        (0.24, -0.42),
        (0.00, -0.88),
    ),
    12: (
        (-0.80, 0.47),
        (-0.52, 0.58),
        (-0.18, 0.63),
        (0.18, 0.63),
        (0.52, 0.58),
        (0.80, 0.47),
        (-0.34, 0.12),
        (0.34, 0.12),
        (-0.28, -0.34),
        (0.28, -0.34),
        (-0.16, -0.86),
        (0.16, -0.86),
    ),
}

# Canonical force-application points on the exact boundary of the scaled T
# union used by collision and rendering.  The audited 12-agent table is kept
# verbatim: with radius 0.315 m it forms one connected graph with exactly 14
# undirected links.  Smaller layouts preserve the same symmetric ordering and
# put every handle on that same union boundary.
_BOUNDARY_ATTACHED_HANDLE_POINTS: dict[int, tuple[tuple[float, float], ...]] = {
    4: ((-0.4675, 0.29), (0.4675, 0.29), (-0.08, -0.5225), (0.08, -0.5225)),
    6: (
        (-0.4675, 0.29),
        (0.00, 0.396),
        (0.4675, 0.29),
        (-0.11, -0.12),
        (0.11, -0.12),
        (0.00, -0.5225),
    ),
    8: (
        (-0.4675, 0.29),
        (-0.18, 0.396),
        (0.18, 0.396),
        (0.4675, 0.29),
        (-0.11, 0.05),
        (0.11, 0.05),
        (-0.08, -0.5225),
        (0.08, -0.5225),
    ),
    10: (
        (-0.4675, 0.29),
        (-0.30, 0.396),
        (0.00, 0.396),
        (0.30, 0.396),
        (0.4675, 0.29),
        (-0.11, 0.09),
        (0.11, 0.09),
        (-0.11, -0.25),
        (0.11, -0.25),
        (0.00, -0.5225),
    ),
    12: (
        (-0.4675, 0.29),
        (-0.32, 0.396),
        (-0.11, 0.396),
        (0.11, 0.396),
        (0.32, 0.396),
        (0.4675, 0.29),
        (-0.11, 0.09),
        (0.11, 0.09),
        (-0.11, -0.215),
        (0.11, -0.215),
        (-0.08, -0.5225),
        (0.08, -0.5225),
    ),
}


def _parametric_t_handle_points(config: EnvConfig) -> Tensor:
    """Construct symmetric, boundary-attached handles for every N in [8,20]."""

    morphology = t_morphology_from_config(config)
    n_agents = config.n_agents
    if not 8 <= n_agents <= 20:
        raise ValueError("parametric-t-v3 supports every integer agent count in [8,20]")
    if morphology == CANONICAL_T_MORPHOLOGY and n_agents == 12:
        return torch.tensor(_BOUNDARY_ATTACHED_HANDLE_POINTS[12], dtype=torch.float32)

    half_width = morphology.total_width_m / 2.0
    half_stem = morphology.stem_width_m / 2.0
    body_bottom = -0.06325 - morphology.total_height_m / 2.0
    body_top = -0.06325 + morphology.total_height_m / 2.0
    cap_bottom = body_top - morphology.cap_thickness_m
    cap_center = (cap_bottom + body_top) / 2.0

    top_count = (n_agents + 1) // 2
    points: list[tuple[float, float]] = [(-half_width, cap_center)]
    if top_count > 2:
        # N=8 has only two inner cap handles; pull that pair slightly inward
        # so it links both cap endpoints and the first thin-stem level at the
        # fixed 0.4 m communication radius.  Denser layouts can spread farther.
        inner_extent = (0.45 if top_count == 4 else 0.68) * half_width
        for index in range(top_count - 2):
            fraction = index / max(1, top_count - 3)
            x = -inner_extent + 2.0 * inner_extent * fraction
            points.append((x, body_top))
    points.append((half_width, cap_center))

    remainder = n_agents - top_count
    pair_count = remainder // 2
    has_center = remainder % 2 == 1
    exposed_top = cap_bottom
    for level in range(pair_count):
        # Space the stem levels relative to the complete top-to-bottom span.
        # Clamping only the first dense level to the exposed junction keeps
        # it on the union boundary.  The resulting maximum vertical hop is
        # H/(pair_count+1), which keeps even the sparse N=8 radius graph
        # connected throughout the admitted morphology box.
        y = body_top + (level + 1) * (
            body_bottom - body_top
        ) / (pair_count + 1)
        y = min(exposed_top, y)
        points.extend(((-half_stem, y), (half_stem, y)))
    if has_center:
        points.append((0.0, body_bottom))
    if len(points) != n_agents:
        raise RuntimeError("parametric T handle construction changed cardinality")
    return torch.tensor(points, dtype=torch.float32)


@lru_cache(maxsize=512)
def _cached_glyph_contact_points(
    payload_shape: str,
    n_agents: int,
    communication_radius_m: float,
    glyph_scale: float,
    device_name: str,
) -> Tensor:
    """Cache immutable exact glyph handles on their deployment device."""

    return glyph_boundary_handle_points(
        payload_shape,
        n_agents,
        communication_radius_m=communication_radius_m,
        glyph_scale=glyph_scale,
    ).to(device=torch.device(device_name))


def contact_points(
    config: EnvConfig | int,
    device: torch.device | str | None = None,
    *,
    handle_layout: str | None = None,
) -> Tensor:
    """Return versioned handle locations in payload body coordinates.

    Passing an ``EnvConfig`` is the authoritative path used by the simulator
    and model.  The integer form is retained for external callers and defaults
    to the historical layout, preserving the old public API.
    """

    if isinstance(config, EnvConfig):
        n_agents = config.n_agents
        if handle_layout is not None and handle_layout != config.handle_layout:
            raise ValueError("handle_layout override disagrees with EnvConfig")
        layout = config.handle_layout
    else:
        n_agents = int(config)
        layout = HANDLE_LAYOUT_LEGACY_OUTRIGGER_V1 if handle_layout is None else handle_layout
    if layout == HANDLE_LAYOUT_LEGACY_OUTRIGGER_V1:
        tables = _LEGACY_OUTRIGGER_HANDLE_POINTS
    elif layout == HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2:
        tables = _BOUNDARY_ATTACHED_HANDLE_POINTS
    elif layout == HANDLE_LAYOUT_PARAMETRIC_T_V3:
        if not isinstance(config, EnvConfig):
            raise ValueError("parametric-t-v3 requires a complete EnvConfig")
        return _parametric_t_handle_points(config).to(device=device)
    elif layout == HANDLE_LAYOUT_GLYPH_BOUNDARY_V4:
        if not isinstance(config, EnvConfig):
            raise ValueError("glyph-boundary-v4 requires a complete EnvConfig")
        resolved_device = torch.device("cpu" if device is None else device)
        return _cached_glyph_contact_points(
            config.payload_shape,
            config.n_agents,
            float(config.comm_radius),
            float(config.glyph_scale),
            str(resolved_device),
        )
    else:
        raise ValueError(f"handle_layout must be one of {HANDLE_LAYOUTS}, got {layout!r}")
    if n_agents not in tables:
        raise ValueError("supported n_agents are 4, 6, 8, 10, and 12")
    return torch.tensor(tables[n_agents], dtype=torch.float32, device=device)


def t_shape_points(device: torch.device | str | None = None) -> Tensor:
    """Support points of a compound T body, used for slit collision checks."""

    hull_scale = 0.55
    # Preserve the historical ordering and interior samples exactly: several
    # sealed teachers use this compatibility API in their risk proxy.
    stem = [(-0.20, -0.95), (0.20, -0.95), (-0.20, 0.55), (0.20, 0.55)]
    top = [(-0.85, 0.35), (0.85, 0.35), (-0.85, 0.72), (0.85, 0.72)]
    samples = [(-0.55, 0.54), (0.55, 0.54), (0.0, -0.55), (0.0, 0.0)]
    return hull_scale * torch.tensor(
        stem + top + samples,
        dtype=torch.float32,
        device=device,
    )


def t_shape_rectangles(device: torch.device | str | None = None) -> Tensor:
    """Two local rectangles whose union is the rendered/colliding T payload."""

    return payload_shape_rectangles("T", device)


def rotate(points: Tensor, theta: Tensor) -> Tensor:
    """Rotate `[P,2]` body points by batch angles `[B]`."""

    c = torch.cos(theta).unsqueeze(-1)
    s = torch.sin(theta).unsqueeze(-1)
    x = points[:, 0].view(1, -1)
    y = points[:, 1].view(1, -1)
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def body_to_world(points: Tensor, state: Tensor) -> Tensor:
    rel = rotate(points, state[:, 2])
    return rel + state[:, None, 0:2]


def body_rectangles_to_world(rectangles: Tensor, state: Tensor) -> Tensor:
    """Rotate local rectangles `[R,4,2]` by batch poses `[B,6]`."""

    batch = state.shape[0]
    flat = rectangles.reshape(-1, 2)
    world = body_to_world(flat, state)
    return world.reshape(batch, rectangles.shape[0], rectangles.shape[1], 2)


def _convex_rect_intersects(poly: Tensor, rect: Tensor) -> Tensor:
    """Vectorized SAT intersection for convex quads `[B,4,2]`."""

    axes = []
    for edge_id in (0, 1):
        edge = poly[:, edge_id + 1] - poly[:, edge_id]
        normal = torch.stack((-edge[:, 1], edge[:, 0]), dim=-1)
        axes.append(normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-8))
    x_axis = torch.zeros_like(axes[0])
    x_axis[:, 0] = 1.0
    y_axis = torch.zeros_like(axes[0])
    y_axis[:, 1] = 1.0
    axes.extend((x_axis, y_axis))
    stacked = torch.stack(axes, dim=1)
    poly_proj = torch.einsum("bpd,bad->bpa", poly, stacked)
    rect_proj = torch.einsum("bpd,bad->bpa", rect, stacked)
    separated = (poly_proj.amax(dim=1) < rect_proj.amin(dim=1)) | (
        rect_proj.amax(dim=1) < poly_proj.amin(dim=1)
    )
    return ~separated.any(dim=1)


def sample_scenarios(
    batch_size: int,
    cfg: EnvConfig,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    difficulty: float = 1.0,
) -> PianoBatch:
    """Sample paired procedural three-room tasks."""

    dev = torch.device(device)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, device=dev, generator=generator)

    effective_difficulty = float(difficulty) * float(cfg.scenario_difficulty)
    gap1 = (rand(batch_size) - 0.5) * cfg.gap_y_range
    gap2 = (rand(batch_size) - 0.5) * cfg.gap_y_range
    half = (
        cfg.gap_half_base
        - cfg.gap_half_difficulty_scale * effective_difficulty
        + cfg.gap_half_jitter * rand(batch_size, 2)
    )
    wall_x = (
        torch.tensor([-cfg.wall_x_abs, cfg.wall_x_abs], device=dev)
        .view(1, 2)
        .expand(batch_size, -1)
    )
    gap_y = torch.stack((gap1, gap2), dim=-1)
    left = torch.zeros(batch_size, 3, device=dev)
    left[:, 0] = cfg.source_x
    left[:, 1] = gap1 + 0.10 * (rand(batch_size) - 0.5)
    right = torch.zeros(batch_size, 3, device=dev)
    right[:, 0] = cfg.goal_x
    right[:, 1] = gap2 + 0.10 * (rand(batch_size) - 0.5)
    if cfg.bidirectional:
        reverse = rand(batch_size) < 0.5
        source = torch.where(reverse[:, None], right, left)
        goal = torch.where(reverse[:, None], left, right)
    else:
        source = left
        goal = right
    state0 = torch.zeros(batch_size, 6, device=dev)
    state0[:, :3] = source
    return PianoBatch(
        state0=state0,
        goal=goal,
        wall_x=wall_x,
        gap_y=gap_y,
        gap_half=half,
        source=source,
        leg=(torch.zeros(batch_size, dtype=torch.long, device=dev) if cfg.round_trip else None),
    )


def sample_training_states(
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample states along the task corridor for expert imitation."""

    batch = scenario.state0.shape[0]
    device = scenario.state0.device

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, device=device, generator=generator)

    active_source, active_goal = active_endpoints(scenario)
    state = torch.zeros(batch, 6, device=device)
    progress = rand(batch)
    state[:, 0:2] = active_source[:, 0:2] + progress[:, None] * (
        active_goal[:, 0:2] - active_source[:, 0:2]
    )
    direction = torch.sign(active_goal[:, 0] - active_source[:, 0])
    direction = torch.where(direction == 0, torch.ones_like(direction), direction)
    first_gap = torch.where(direction > 0, scenario.gap_y[:, 0], scenario.gap_y[:, 1])
    second_gap = torch.where(direction > 0, scenario.gap_y[:, 1], scenario.gap_y[:, 0])
    interp_gap = torch.where(progress < 0.48, first_gap, second_gap)
    state[:, 1] = interp_gap + 0.52 * (rand(batch) - 0.5)
    theta1 = 0.80 * direction
    theta2 = -0.80 * direction
    first_wall = torch.where(direction > 0, scenario.wall_x[:, 0], scenario.wall_x[:, 1])
    second_wall = torch.where(direction > 0, scenario.wall_x[:, 1], scenario.wall_x[:, 0])
    route_x = direction * (state[:, 0] - active_source[:, 0])
    first_wall_s = direction * (first_wall - active_source[:, 0])
    second_wall_s = direction * (second_wall - active_source[:, 0])
    nominal_theta = torch.where(
        route_x < first_wall_s + 0.20,
        theta1,
        torch.where(route_x < second_wall_s + 0.20, theta2, 0.0),
    )
    state[:, 2] = wrap_angle(nominal_theta + 0.65 * (rand(batch) - 0.5))
    state[:, 3:5] = 0.25 * torch.randn(batch, 2, device=device, generator=generator)
    state[:, 5] = 0.25 * torch.randn(batch, device=device, generator=generator)
    # Keep a slice of initial states in every batch.
    initial_mask = rand(batch) < 0.20
    active_state0 = torch.zeros_like(state)
    active_state0[:, :3] = active_source
    state = torch.where(initial_mask[:, None], active_state0, state)
    # Random corridor interpolation can place a rotated T inside a wall.  Such
    # states have no physically valid one-step recovery target and previously
    # poisoned about 7% of imitation samples.  Model-induced DAgger supplies
    # genuine near-collision states; invalid synthetic states fall back to the
    # corresponding safe active endpoint.
    invalid = collision_mask(state, scenario, cfg)
    state = torch.where(invalid[:, None], active_state0, state)
    return state


def desired_maneuver(state: Tensor, scenario: PianoBatch) -> tuple[Tensor, Tensor]:
    """Return a direction-equivariant target pose and mission progress."""

    active_source, active_goal = active_endpoints(scenario)
    direction = torch.sign(active_goal[:, 0] - active_source[:, 0])
    direction = torch.where(direction == 0, torch.ones_like(direction), direction)
    first_is_left = direction > 0
    first_wall = torch.where(first_is_left, scenario.wall_x[:, 0], scenario.wall_x[:, 1])
    second_wall = torch.where(first_is_left, scenario.wall_x[:, 1], scenario.wall_x[:, 0])
    first_gap = torch.where(first_is_left, scenario.gap_y[:, 0], scenario.gap_y[:, 1])
    second_gap = torch.where(first_is_left, scenario.gap_y[:, 1], scenario.gap_y[:, 0])

    route_x = direction * (state[:, 0] - active_source[:, 0])
    first_s = direction * (first_wall - active_source[:, 0])
    second_s = direction * (second_wall - active_source[:, 0])
    theta1 = PASSAGE_TARGET_ANGLE * direction
    theta2 = -PASSAGE_TARGET_ANGLE * direction
    target = active_goal.clone()

    before_1 = route_x < first_s - 0.22
    in_1 = (route_x >= first_s - 0.22) & (route_x < first_s + 0.36)
    before_2 = (route_x >= first_s + 0.36) & (route_x < second_s - 0.22)
    in_2 = (route_x >= second_s - 0.22) & (route_x < second_s + 0.36)

    target[:, 0] = torch.where(before_1, first_wall - 0.28 * direction, target[:, 0])
    target[:, 1] = torch.where(before_1, first_gap, target[:, 1])
    target[:, 2] = torch.where(before_1, theta1, target[:, 2])

    target[:, 0] = torch.where(in_1, first_wall + 0.38 * direction, target[:, 0])
    target[:, 1] = torch.where(in_1, first_gap, target[:, 1])
    target[:, 2] = torch.where(in_1, theta1, target[:, 2])

    target[:, 0] = torch.where(before_2, second_wall - 0.28 * direction, target[:, 0])
    target[:, 1] = torch.where(before_2, second_gap, target[:, 1])
    target[:, 2] = torch.where(before_2, theta2, target[:, 2])

    target[:, 0] = torch.where(in_2, second_wall + 0.38 * direction, target[:, 0])
    target[:, 1] = torch.where(in_2, second_gap, target[:, 1])
    target[:, 2] = torch.where(in_2, theta2, target[:, 2])

    progress = leg_progress(state, scenario)
    active_gap = torch.where(route_x < 0.5 * (first_s + second_s), first_gap, second_gap)
    passage_offset = (state[:, 1] - active_gap).clamp(-1.0, 1.0)
    overall_progress = (
        (mission_leg(scenario).to(progress.dtype) + progress) / 2.0
        if scenario.leg is not None
        else progress
    )
    phase = torch.stack((passage_offset, overall_progress), dim=-1)
    return target, phase


def expert_policy(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute local expert forces, normalized shared section, and net wrench."""

    contacts = contact_points(cfg, state.device)
    r_world = rotate(contacts, state[:, 2])
    target, phase = desired_maneuver(state, scenario)
    pos_error = target[:, 0:2] - state[:, 0:2]
    angle_error = wrap_angle(target[:, 2] - state[:, 2])
    desired_twist = torch.cat(
        (
            (1.55 * pos_error - 0.45 * state[:, 3:5]).clamp(-1.7, 1.7),
            (1.80 * angle_error - 0.42 * state[:, 5]).clamp(-1.8, 1.8).unsqueeze(-1),
        ),
        dim=-1,
    )
    force_xy = (4.4 * pos_error - 1.35 * state[:, 3:5]).clamp(
        -cfg.n_agents * cfg.max_force * 0.75,
        cfg.n_agents * cfg.max_force * 0.75,
    )
    tau = (3.2 * angle_error - 0.90 * state[:, 5]).clamp(
        -cfg.n_agents * cfg.max_force * 0.62,
        cfg.n_agents * cfg.max_force * 0.62,
    )
    wrench = torch.cat((force_xy, tau.unsqueeze(-1)), dim=-1)

    batch = state.shape[0]
    grasp = torch.zeros(batch, 3, 2 * cfg.n_agents, device=state.device)
    grasp[:, 0, 0::2] = 1.0
    grasp[:, 1, 1::2] = 1.0
    grasp[:, 2, 0::2] = -r_world[:, :, 1]
    grasp[:, 2, 1::2] = r_world[:, :, 0]
    gram = grasp @ grasp.transpose(-1, -2)
    eye = torch.eye(3, device=state.device).view(1, 3, 3)
    dual = solve_nonsingular(gram + 0.08 * eye, wrench.unsqueeze(-1)).squeeze(-1)
    forces = (grasp.transpose(-1, -2) @ dual.unsqueeze(-1)).squeeze(-1)
    forces = forces.view(batch, cfg.n_agents, 2)
    norms = torch.linalg.vector_norm(forces, dim=-1, keepdim=True).clamp_min(1e-6)
    forces = forces * (cfg.max_force / norms).clamp_max(1.0)

    wrench_scale = cfg.n_agents * cfg.max_force
    section = torch.cat(
        (
            desired_twist[:, 0:2] / 1.7,
            desired_twist[:, 2:3] / 1.8,
            wrench[:, 0:2] / wrench_scale,
            wrench[:, 2:3] / (0.75 * wrench_scale),
            phase,
        ),
        dim=-1,
    )
    return forces, section, wrench


def net_wrench_from_forces(state: Tensor, forces: Tensor, cfg: EnvConfig) -> Tensor:
    """Map per-handle planar forces to the payload net ``(Fx, Fy, tau)``."""

    contacts = contact_points(cfg, state.device)
    r_world = rotate(contacts, state[:, 2])
    force_xy = forces.sum(dim=1)
    torque = (r_world[..., 0] * forces[..., 1] - r_world[..., 1] * forces[..., 0]).sum(dim=1)
    return torch.cat((force_xy, torque.unsqueeze(-1)), dim=-1)


def local_current_slit_y_delta_v2(
    goal_delta_contact: Tensor,
    visible_wall_rel_x: Tensor,
    visible_gap_rel_y: Tensor,
    visibility: Tensor,
    cfg: EnvConfig,
) -> Tensor:
    """Return the V2 local current-slit query in legacy channel-9 units.

    Every input is already present in one agent's strict-local observation:
    active-goal x supplies only the active-leg direction, while wall x, gap y,
    and visibility come from radius-limited sensing.  The continuous weights
    exactly encode the frozen step-300 counterfactual that removed premature
    attraction to the final/second-slit y.  Invisible geometry has zero weight
    and the epsilon clamp makes the no-visible-wall result zero without an
    action gate or a privileged route phase.
    """

    direction = torch.sign(goal_delta_contact[..., 0:1])
    route_relative_x = direction * visible_wall_rel_x * cfg.wall_sensor_range
    weights = (
        visibility
        * torch.sigmoid(12.0 * (route_relative_x + 0.22))
        * torch.exp(-1.5 * route_relative_x.abs())
    )
    local_gap_delta = visible_gap_rel_y * cfg.world_y / 2.6
    return (weights * local_gap_delta).sum(dim=-1, keepdim=True) / weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)


def _generic_obstacle_metadata(
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    batch_size: int,
) -> tuple[Tensor | None, int, int]:
    """Validate cheap obstacle metadata without reading a device scalar."""

    slots = cfg.generic_obstacle_slots
    obstacle_rects = scenario.obstacle_rects
    if obstacle_rects is not None and (
        obstacle_rects.ndim != 3 or obstacle_rects.shape[-1] != 4
    ):
        raise ValueError("obstacle_rects must have shape [batch, obstacles, 4]")
    if obstacle_rects is not None and obstacle_rects.shape[0] != batch_size:
        raise ValueError("obstacle_rects batch does not match state")
    obstacle_count = 0 if obstacle_rects is None else obstacle_rects.shape[1]
    if obstacle_count > slots:
        raise ValueError(
            f"scenario has {obstacle_count} generic obstacles but "
            f"generic_obstacle_slots={slots}"
        )
    return obstacle_rects, obstacle_count, slots


def validate_generic_obstacle_geometry(
    scenario: PianoBatch,
    cfg: EnvConfig,
    reference: Tensor,
) -> ValidatedGenericObstacleGeometry:
    """Validate and bind static generic-obstacle geometry once per rollout.

    The ordered-bound check is intentionally performed after the same
    dtype/device conversion used by :func:`generic_obstacle_observations`, so a
    successful token is output-equivalent to taking the ordinary validation
    branch on every call.
    """

    if reference.ndim < 1:
        raise ValueError("generic obstacle reference must include a batch axis")
    batch_size = int(reference.shape[0])
    obstacle_rects, obstacle_count, slots = _generic_obstacle_metadata(
        scenario,
        cfg,
        batch_size=batch_size,
    )
    if slots > 0 and cfg.wall_sensor_range <= 0.0:
        raise ValueError("wall_sensor_range must be positive when generic obstacle slots are used")
    rects = (
        None
        if obstacle_rects is None or obstacle_count == 0
        else obstacle_rects.to(device=reference.device, dtype=reference.dtype)
    )
    if rects is not None:
        lower = rects[..., (0, 2)]
        upper = rects[..., (1, 3)]
        if bool((lower >= upper).any()):
            raise ValueError("every obstacle rectangle must have increasing bounds")
    return ValidatedGenericObstacleGeometry(
        source_obstacle_rects=obstacle_rects,
        source_version=None if obstacle_rects is None else obstacle_rects._version,
        rects=rects,
        batch_size=batch_size,
        obstacle_slots=slots,
        wall_sensor_range=float(cfg.wall_sensor_range),
        device=reference.device,
        dtype=reference.dtype,
    )


def _validated_obstacle_rects(
    geometry: ValidatedGenericObstacleGeometry,
    obstacle_rects: Tensor | None,
    cfg: EnvConfig,
    contact_world: Tensor,
) -> Tensor | None:
    """Fail closed if a static-geometry token is reused for another rollout."""

    source_version = None if obstacle_rects is None else obstacle_rects._version
    if (
        geometry.source_obstacle_rects is not obstacle_rects
        or geometry.source_version != source_version
        or geometry.batch_size != contact_world.shape[0]
        or geometry.obstacle_slots != cfg.generic_obstacle_slots
        or geometry.wall_sensor_range != float(cfg.wall_sensor_range)
        or geometry.device != contact_world.device
        or geometry.dtype != contact_world.dtype
    ):
        raise ValueError("validated generic obstacle geometry does not match this observation")
    return geometry.rects


def generic_obstacle_observations(
    contact_world: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    validated_geometry: ValidatedGenericObstacleGeometry | None = None,
) -> Tensor:
    """Encode fixed-capacity, radius-limited AABB records per handle.

    Each obstacle slot is ``[visible, centre_dx/range, centre_dy/range,
    half_x/range, half_y/range]``.  Visibility is determined by the shortest
    Euclidean distance from the handle to the AABB (not to its centre), and
    every channel of an invisible or padded slot is exactly zero.
    """

    if contact_world.ndim != 3 or contact_world.shape[-1] != 2:
        raise ValueError("contact_world must have shape [batch, agents, 2]")
    batch, n_agents, _ = contact_world.shape
    obstacle_rects, obstacle_count, slots = _generic_obstacle_metadata(
        scenario,
        cfg,
        batch_size=batch,
    )
    if slots == 0:
        return contact_world.new_zeros((batch, n_agents, 0))
    if cfg.wall_sensor_range <= 0.0:
        raise ValueError("wall_sensor_range must be positive when generic obstacle slots are used")
    if obstacle_rects is None or obstacle_count == 0:
        return contact_world.new_zeros((batch, n_agents, 5 * slots))

    if validated_geometry is None:
        validated_geometry = validate_generic_obstacle_geometry(scenario, cfg, contact_world)
    rects = _validated_obstacle_rects(
        validated_geometry,
        obstacle_rects,
        cfg,
        contact_world,
    )
    if rects is None:  # pragma: no cover - guarded by obstacle_count above
        raise RuntimeError("validated generic obstacle geometry omitted its rectangles")
    lower = rects[..., (0, 2)]
    upper = rects[..., (1, 3)]
    centres = 0.5 * (lower + upper)
    half_extents = 0.5 * (upper - lower)
    handle_delta = centres[:, None, :, :] - contact_world[:, :, None, :]
    distance_to_box = torch.linalg.vector_norm(
        (handle_delta.abs() - half_extents[:, None, :, :]).clamp_min(0.0),
        dim=-1,
    )
    visible = (distance_to_box <= cfg.wall_sensor_range).to(contact_world.dtype)
    scale = float(cfg.wall_sensor_range)
    features = torch.cat(
        (
            visible.unsqueeze(-1),
            handle_delta / scale,
            half_extents[:, None, :, :].expand(-1, n_agents, -1, -1) / scale,
        ),
        dim=-1,
    )
    features = features * visible.unsqueeze(-1)
    if obstacle_count < slots:
        padding = contact_world.new_zeros((batch, n_agents, slots - obstacle_count, 5))
        features = torch.cat((features, padding), dim=2)
    return features.reshape(batch, n_agents, 5 * slots)


def observations(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    validated_obstacle_geometry: ValidatedGenericObstacleGeometry | None = None,
) -> Tensor:
    """Local per-handle observations under the Force abstraction."""

    contacts = contact_points(cfg, state.device)
    contact_world = body_to_world(contacts, state)
    r_world = contact_world - state[:, None, 0:2]
    contact_velocity = torch.stack(
        (
            state[:, None, 3] - state[:, None, 5] * r_world[:, :, 1],
            state[:, None, 4] + state[:, None, 5] * r_world[:, :, 0],
        ),
        dim=-1,
    )
    batch = state.shape[0]
    n = cfg.n_agents
    active_source, active_goal = active_endpoints(scenario)
    target, phase = desired_maneuver(state, scenario)
    goal_delta = active_goal[:, 0:2] - state[:, 0:2]
    source_delta = state[:, 0:2] - active_source[:, 0:2]
    angle_goal = wrap_angle(active_goal[:, 2] - state[:, 2])
    angle_target = wrap_angle(target[:, 2] - state[:, 2])
    generic_obstacles = generic_obstacle_observations(
        contact_world,
        scenario,
        cfg,
        validated_geometry=validated_obstacle_geometry,
    )

    if cfg.strict_local_observation:
        # Strict local sensing intentionally removes the repeated global
        # waypoint/progress/gap features used by earlier experiments.  V1
        # retains the historical final-goal delta.  V2 preserves the same
        # 18-D layout but rebinds only channel 9 to a smooth current-slit query
        # formed independently from each agent's radius-limited wall records.
        repeated = torch.zeros(batch, n, 0, device=state.device)
        visibility = (
            (contact_world[:, :, 0:1] - scenario.wall_x[:, None, :]).abs() <= cfg.wall_sensor_range
        ).float()
        wall_rel_x = (
            scenario.wall_x[:, None, :] - contact_world[:, :, 0:1]
        ) / cfg.wall_sensor_range
        gap_rel_y = (scenario.gap_y[:, None, :] - contact_world[:, :, 1:2]) / cfg.world_y
        gap_half = scenario.gap_half[:, None, :] / cfg.world_y
        visible_wall_rel_x = visibility * wall_rel_x
        visible_gap_rel_y = visibility * gap_rel_y
        local_wall = torch.cat(
            (
                visible_wall_rel_x,
                visible_gap_rel_y,
                visibility * gap_half,
                visibility,
            ),
            dim=-1,
        )
        goal_delta_contact = (active_goal[:, None, 0:2] - contact_world) / 2.6
        if cfg.strict_observation_schema == STRICT_OBSERVATION_SCHEMA_LOCAL_CURRENT_SLIT_V2:
            goal_delta_contact = torch.cat(
                (
                    goal_delta_contact[..., 0:1],
                    local_current_slit_y_delta_v2(
                        goal_delta_contact,
                        visible_wall_rel_x,
                        visible_gap_rel_y,
                        visibility,
                        cfg,
                    ),
                ),
                dim=-1,
            )
        strict_local = torch.cat(
            (
                torch.sin(state[:, 2:3]).unsqueeze(1).expand(-1, n, -1),
                torch.cos(state[:, 2:3]).unsqueeze(1).expand(-1, n, -1),
                contacts.view(1, n, 2).expand(batch, -1, -1),
                contact_world[:, :, 0:1] / 2.6,
                contact_world[:, :, 1:2] / cfg.world_y,
                contact_velocity / 2.0,
                goal_delta_contact,
            ),
            dim=-1,
        )
        return torch.cat((strict_local, local_wall, generic_obstacles), dim=-1)

    if cfg.local_observation:
        progress = leg_progress(state, scenario).unsqueeze(-1)
        global_features = torch.cat(
            (
                state[:, 0:2] / 2.6,
                torch.sin(state[:, 2:3]),
                torch.cos(state[:, 2:3]),
                state[:, 3:5] / 2.0,
                state[:, 5:6] / 2.0,
                goal_delta / 2.6,
                torch.sin(angle_goal).unsqueeze(-1),
                torch.cos(angle_goal).unsqueeze(-1),
                source_delta / 2.6,
                progress,
            ),
            dim=-1,
        )
        repeated = global_features.unsqueeze(1).expand(-1, n, -1)
        visibility = (
            (contact_world[:, :, 0:1] - scenario.wall_x[:, None, :]).abs() <= cfg.wall_sensor_range
        ).float()
        wall_rel_x = (
            scenario.wall_x[:, None, :] - contact_world[:, :, 0:1]
        ) / cfg.wall_sensor_range
        gap_rel_y = (scenario.gap_y[:, None, :] - contact_world[:, :, 1:2]) / cfg.world_y
        gap_half = scenario.gap_half[:, None, :] / cfg.world_y
        local_wall = torch.cat(
            (
                visibility * wall_rel_x,
                visibility * gap_rel_y,
                visibility * gap_half,
                visibility,
            ),
            dim=-1,
        )
    else:
        global_features = torch.cat(
            (
                state[:, 0:2] / 2.6,
                torch.sin(state[:, 2:3]),
                torch.cos(state[:, 2:3]),
                state[:, 3:5] / 2.0,
                state[:, 5:6] / 2.0,
                goal_delta / 2.6,
                torch.sin(angle_goal).unsqueeze(-1),
                torch.cos(angle_goal).unsqueeze(-1),
                source_delta / 2.6,
                (target[:, 0:2] - state[:, 0:2]) / 2.6,
                torch.sin(angle_target).unsqueeze(-1),
                torch.cos(angle_target).unsqueeze(-1),
                phase,
            ),
            dim=-1,
        )
        wall_features = torch.cat(
            (
                (scenario.wall_x - state[:, 0:1]) / 2.6,
                (scenario.gap_y - state[:, 1:2]) / cfg.world_y,
                scenario.gap_half,
            ),
            dim=-1,
        )
        repeated = (
            torch.cat((global_features, wall_features), dim=-1).unsqueeze(1).expand(-1, n, -1)
        )
        local_wall = torch.zeros(batch, n, 0, device=state.device)
    local = torch.cat(
        (
            contacts.view(1, n, 2).expand(batch, -1, -1),
            r_world / 1.2,
            contact_world[:, :, 0:1] / 2.6,
            contact_world[:, :, 1:2] / cfg.world_y,
        ),
        dim=-1,
    )
    return torch.cat((repeated, local_wall, local, generic_obstacles), dim=-1)


def radius_edges(cfg: EnvConfig, *, device: torch.device | str) -> Tensor:
    contacts = contact_points(cfg, device)
    distances = torch.cdist(contacts, contacts)
    rows, cols = torch.where((distances <= cfg.comm_radius) & (distances > 0))
    undirected = rows < cols
    edges = torch.stack((rows[undirected], cols[undirected]), dim=-1)
    if edges.numel() == 0:
        raise ValueError("communication radius produced an empty graph")
    return edges.long()


def step_dynamics(state: Tensor, forces: Tensor, cfg: EnvConfig) -> Tensor:
    contacts = contact_points(cfg, state.device)
    r_world = rotate(contacts, state[:, 2])
    net_force = forces.sum(dim=1)
    torque = (r_world[:, :, 0] * forces[:, :, 1] - r_world[:, :, 1] * forces[:, :, 0]).sum(dim=1)
    next_state = state.clone()
    next_state[:, 3:5] = (next_state[:, 3:5] + cfg.dt * net_force / cfg.mass) * (
        1.0 - cfg.linear_damping * cfg.dt
    )
    next_state[:, 5] = (next_state[:, 5] + cfg.dt * torque / cfg.inertia) * (
        1.0 - cfg.angular_damping * cfg.dt
    )
    next_state[:, 0:2] = next_state[:, 0:2] + cfg.dt * next_state[:, 3:5]
    next_state[:, 2] = wrap_angle(next_state[:, 2] + cfg.dt * next_state[:, 5])
    return next_state


def _reserved_aabb_vertices(obstacle_rects: Tensor, reserve: float = 0.03) -> Tensor:
    """Convert ``[B, O, 4]`` AABBs into reserved SAT quadrilaterals."""

    if obstacle_rects.ndim != 3 or obstacle_rects.shape[-1] != 4:
        raise ValueError("obstacle_rects must have shape [batch, obstacles, 4]")
    x0 = obstacle_rects[..., 0] - reserve
    x1 = obstacle_rects[..., 1] + reserve
    y0 = obstacle_rects[..., 2] - reserve
    y1 = obstacle_rects[..., 3] + reserve
    if bool(((x0 >= x1) | (y0 >= y1)).any()):
        raise ValueError("every obstacle rectangle must have increasing bounds")
    return torch.stack(
        (
            torch.stack((x0, y0), dim=-1),
            torch.stack((x1, y0), dim=-1),
            torch.stack((x1, y1), dim=-1),
            torch.stack((x0, y1), dim=-1),
        ),
        dim=-2,
    )


def _aabb_pair_clearance(
    body_rects: Tensor,
    axes: Tensor,
    obstacle_rects: Tensor,
) -> Tensor:
    """Signed SAT clearance for every compound-body/AABB pair ``[B,R,O]``."""

    obstacles = _reserved_aabb_vertices(obstacle_rects)
    body_projection = torch.einsum("brpd,brad->brpa", body_rects, axes)
    obstacle_projection = torch.einsum("bopd,brad->bropa", obstacles, axes)
    body_projection = body_projection[:, :, None]
    axis_clearance = torch.maximum(
        obstacle_projection.amin(dim=-2) - body_projection.amax(dim=-2),
        body_projection.amin(dim=-2) - obstacle_projection.amax(dim=-2),
    )
    return axis_clearance.amax(dim=-1)


def _collision_mask_geometry(
    state: Tensor,
    wall_x: Tensor,
    gap_y: Tensor,
    gap_half: Tensor,
    cfg: EnvConfig,
    obstacle_rects: Tensor | None = None,
) -> Tensor:
    """Exact T/wall SAT using only the geometry tensors it actually consumes."""

    body_rects = body_rectangles_to_world(
        payload_rectangles_for_config(cfg, state.device), state
    )
    batch = state.shape[0]
    # Evaluate both T-body rectangles against both solid halves of every wall
    # in one SAT batch.  The former nested Python loop launched the same small
    # projection/reduction kernels eight times per collision query (2 body
    # rectangles x 2 walls x 2 wall halves).  Exact collision is a detached,
    # boolean authority in rollout training, so batching those independent SAT
    # pairs preserves its semantics while removing most launch overhead.
    body_edges = body_rects[:, :, 1:3] - body_rects[:, :, :2]
    body_normals = torch.stack((-body_edges[..., 1], body_edges[..., 0]), dim=-1)
    body_normals = body_normals / torch.linalg.vector_norm(
        body_normals, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    cardinal_axes = torch.eye(2, dtype=body_rects.dtype, device=body_rects.device).view(1, 1, 2, 2)
    cardinal_axes = cardinal_axes.expand(batch, body_rects.shape[1], -1, -1)
    axes = torch.cat((body_normals, cardinal_axes), dim=2)

    gap_half = (gap_half - 0.03).clamp_min(0.0)
    x0 = wall_x - cfg.wall_slab
    x1 = wall_x + cfg.wall_slab
    lower_y1 = gap_y - gap_half
    upper_y0 = gap_y + gap_half
    world_bottom = torch.full_like(x0, -cfg.world_y)
    world_top = torch.full_like(x0, cfg.world_y)
    lower = torch.stack(
        (
            torch.stack((x0, world_bottom), dim=-1),
            torch.stack((x1, world_bottom), dim=-1),
            torch.stack((x1, lower_y1), dim=-1),
            torch.stack((x0, lower_y1), dim=-1),
        ),
        dim=2,
    )
    upper = torch.stack(
        (
            torch.stack((x0, upper_y0), dim=-1),
            torch.stack((x1, upper_y0), dim=-1),
            torch.stack((x1, world_top), dim=-1),
            torch.stack((x0, world_top), dim=-1),
        ),
        dim=2,
    )
    wall_rects = torch.stack((lower, upper), dim=2)
    wall_exists = torch.stack((lower_y1 > -cfg.world_y, upper_y0 < cfg.world_y), dim=2)

    body_projection = torch.einsum("brpd,brad->brpa", body_rects, axes)
    wall_projection = torch.einsum("bwhpd,brad->brwhpa", wall_rects, axes)
    body_projection = body_projection[:, :, None, None]
    separated = (body_projection.amax(dim=-2) < wall_projection.amin(dim=-2)) | (
        wall_projection.amax(dim=-2) < body_projection.amin(dim=-2)
    )
    intersects = ~separated.any(dim=-1)
    wall_hit = (intersects & wall_exists[:, None]).any(dim=(1, 2, 3))
    if obstacle_rects is None or obstacle_rects.shape[1] == 0:
        obstacle_hit = torch.zeros(batch, dtype=torch.bool, device=state.device)
    else:
        if obstacle_rects.shape[0] != batch:
            raise ValueError("obstacle_rects batch does not match state")
        obstacle_hit = (_aabb_pair_clearance(body_rects, axes, obstacle_rects) <= 0).any(
            dim=(1, 2)
        )
    vertices = body_rects.reshape(batch, -1, 2)
    y_hit = vertices[:, :, 1].abs().amax(dim=1) > cfg.world_y
    x_hit = vertices[:, :, 0].abs().amax(dim=1) > cfg.world_x
    return wall_hit | obstacle_hit | y_hit | x_hit


def _collision_clearance_geometry(
    state: Tensor,
    wall_x: Tensor,
    gap_y: Tensor,
    gap_half: Tensor,
    cfg: EnvConfig,
    obstacle_rects: Tensor | None = None,
) -> Tensor:
    """Return the exact-SAT signed clearance of each payload pose.

    The margin is measured in world-coordinate metres because every SAT axis
    is normalized.  A positive value certifies separation from every solid
    wall half and world boundary; a negative value means collision.  Zero at
    a wall contact is a collision, matching ``_collision_mask_geometry``.

    World-boundary contact is the sole convention difference in the boolean
    authority: touching ``world_x/world_y`` is allowed while crossing it is
    not.  Exact zero boundary margins are therefore replaced by the smallest
    positive normal value of the dtype.  Consequently ``margin <= 0`` has the
    same discrete semantics as ``collision_mask`` while the function remains
    piecewise differentiable away from contact ties.

    This implementation intentionally does not replace or feed the boolean
    collision authority.  It mirrors the same two body rectangles, wall
    reserve, four separating axes, and world bounds to provide a training-time
    geometric margin with auditable sign equivalence.
    """

    body_rects = body_rectangles_to_world(
        payload_rectangles_for_config(cfg, state.device), state
    )
    batch = state.shape[0]

    body_edges = body_rects[:, :, 1:3] - body_rects[:, :, :2]
    body_normals = torch.stack((-body_edges[..., 1], body_edges[..., 0]), dim=-1)
    body_normals = body_normals / torch.linalg.vector_norm(
        body_normals,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    cardinal_axes = torch.eye(
        2,
        dtype=body_rects.dtype,
        device=body_rects.device,
    ).view(1, 1, 2, 2)
    cardinal_axes = cardinal_axes.expand(batch, body_rects.shape[1], -1, -1)
    axes = torch.cat((body_normals, cardinal_axes), dim=2)

    reserved_gap_half = (gap_half - 0.03).clamp_min(0.0)
    x0 = wall_x - cfg.wall_slab
    x1 = wall_x + cfg.wall_slab
    lower_y1 = gap_y - reserved_gap_half
    upper_y0 = gap_y + reserved_gap_half
    world_bottom = torch.full_like(x0, -cfg.world_y)
    world_top = torch.full_like(x0, cfg.world_y)
    lower = torch.stack(
        (
            torch.stack((x0, world_bottom), dim=-1),
            torch.stack((x1, world_bottom), dim=-1),
            torch.stack((x1, lower_y1), dim=-1),
            torch.stack((x0, lower_y1), dim=-1),
        ),
        dim=2,
    )
    upper = torch.stack(
        (
            torch.stack((x0, upper_y0), dim=-1),
            torch.stack((x1, upper_y0), dim=-1),
            torch.stack((x1, world_top), dim=-1),
            torch.stack((x0, world_top), dim=-1),
        ),
        dim=2,
    )
    wall_rects = torch.stack((lower, upper), dim=2)
    wall_exists = torch.stack(
        (lower_y1 > -cfg.world_y, upper_y0 < cfg.world_y),
        dim=2,
    )

    body_projection = torch.einsum("brpd,brad->brpa", body_rects, axes)
    wall_projection = torch.einsum("bwhpd,brad->brwhpa", wall_rects, axes)
    body_projection = body_projection[:, :, None, None]
    # On one SAT axis, positive means that the two projected intervals are
    # disjoint and gives their separating distance; zero is contact; negative
    # is overlap.  A convex pair is separated by its best axis, while the
    # compound payload is safe only when every existing pair is separated.
    axis_clearance = torch.maximum(
        wall_projection.amin(dim=-2) - body_projection.amax(dim=-2),
        body_projection.amin(dim=-2) - wall_projection.amax(dim=-2),
    )
    pair_clearance = axis_clearance.amax(dim=-1)
    pair_clearance = pair_clearance.masked_fill(
        ~wall_exists[:, None, :, :],
        float("inf"),
    )
    wall_clearance = pair_clearance.amin(dim=(1, 2, 3))

    if obstacle_rects is None or obstacle_rects.shape[1] == 0:
        obstacle_clearance = torch.full_like(wall_clearance, float("inf"))
    else:
        if obstacle_rects.shape[0] != batch:
            raise ValueError("obstacle_rects batch does not match state")
        obstacle_clearance = _aabb_pair_clearance(
            body_rects,
            axes,
            obstacle_rects,
        ).amin(dim=(1, 2))

    vertices = body_rects.reshape(batch, -1, 2)
    y_clearance = cfg.world_y - vertices[:, :, 1].abs().amax(dim=1)
    x_clearance = cfg.world_x - vertices[:, :, 0].abs().amax(dim=1)
    boundary_clearance = torch.minimum(x_clearance, y_clearance)
    # ``collision_mask`` uses strict ``>`` at world bounds. Preserve that
    # measure-zero equality convention without perturbing nearby margins.
    smallest_positive = torch.full_like(
        boundary_clearance,
        torch.finfo(boundary_clearance.dtype).tiny,
    )
    boundary_clearance = torch.where(
        boundary_clearance == 0,
        smallest_positive,
        boundary_clearance,
    )
    return torch.minimum(
        torch.minimum(wall_clearance, obstacle_clearance),
        boundary_clearance,
    )


def collision_clearance_margin(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
) -> Tensor:
    """Return differentiable signed clearance for payload poses ``[B, 6]``.

    ``collision_clearance_margin(...) <= 0`` is exactly equivalent to
    ``collision_mask(...)`` under the benchmark's floating-point geometry.
    The function is intended for losses and diagnostics; exact collision
    decisions continue to use the boolean SAT authority.
    """

    return _collision_clearance_geometry(
        state,
        scenario.wall_x,
        scenario.gap_y,
        scenario.gap_half,
        cfg,
        scenario.obstacle_rects,
    )


def _smooth_min_lower_bound(values: Tensor, temperature: float, *, dim) -> Tensor:
    """Unnormalised soft minimum, always no greater than the hard minimum."""

    return -temperature * torch.logsumexp(-values / temperature, dim=dim)


def _smooth_max_upper_bound(values: Tensor, temperature: float, *, dim) -> Tensor:
    """Unnormalised soft maximum, always no smaller than the hard maximum."""

    return temperature * torch.logsumexp(values / temperature, dim=dim)


def _smooth_max_lower_bound(values: Tensor, temperature: float, *, dim: int) -> Tensor:
    """Log-mean-exp lower bound on a hard maximum."""

    count = values.shape[dim]
    return temperature * (
        torch.logsumexp(values / temperature, dim=dim) - math.log(count)
    )


def _collision_sat_bundle_lower_geometry(
    state: Tensor,
    wall_x: Tensor,
    gap_y: Tensor,
    gap_half: Tensor,
    cfg: EnvConfig,
    obstacle_rects: Tensor | None = None,
    *,
    projection_temperature_m: float,
    axis_temperature_m: float,
    pair_temperature_m: float,
) -> Tensor:
    """Smooth lower bound on the sampled exact-SAT pose clearance.

    Projection minima are replaced by soft lower bounds and projection maxima
    by soft upper bounds, so each directional gap is conservative.  A
    log-mean-exp then blends all eight direction/axis alternatives of one SAT
    pair without exceeding their hard maximum.  Finally, an unnormalised soft
    minimum blends all valid pairs and every world-boundary side.  The result
    is therefore no greater than the authoritative exact-SAT clearance while
    supplying gradients from every nearly active constraint at a contact tie.

    This is a training surrogate only.  Collision decisions and candidate
    acceptance continue to use ``collision_mask``/``collision_clearance_margin``.
    """

    for name, value in (
        ("projection_temperature_m", projection_temperature_m),
        ("axis_temperature_m", axis_temperature_m),
        ("pair_temperature_m", pair_temperature_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    body_rects = body_rectangles_to_world(
        payload_rectangles_for_config(cfg, state.device), state
    )
    batch = state.shape[0]
    body_edges = body_rects[:, :, 1:3] - body_rects[:, :, :2]
    body_normals = torch.stack((-body_edges[..., 1], body_edges[..., 0]), dim=-1)
    body_normals = body_normals / torch.linalg.vector_norm(
        body_normals, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    cardinal_axes = torch.eye(
        2, dtype=body_rects.dtype, device=body_rects.device
    ).view(1, 1, 2, 2)
    cardinal_axes = cardinal_axes.expand(batch, body_rects.shape[1], -1, -1)
    axes = torch.cat((body_normals, cardinal_axes), dim=2)

    reserved_gap_half = (gap_half - 0.03).clamp_min(0.0)
    x0 = wall_x - cfg.wall_slab
    x1 = wall_x + cfg.wall_slab
    lower_y1 = gap_y - reserved_gap_half
    upper_y0 = gap_y + reserved_gap_half
    world_bottom = torch.full_like(x0, -cfg.world_y)
    world_top = torch.full_like(x0, cfg.world_y)
    lower = torch.stack(
        (
            torch.stack((x0, world_bottom), dim=-1),
            torch.stack((x1, world_bottom), dim=-1),
            torch.stack((x1, lower_y1), dim=-1),
            torch.stack((x0, lower_y1), dim=-1),
        ),
        dim=2,
    )
    upper = torch.stack(
        (
            torch.stack((x0, upper_y0), dim=-1),
            torch.stack((x1, upper_y0), dim=-1),
            torch.stack((x1, world_top), dim=-1),
            torch.stack((x0, world_top), dim=-1),
        ),
        dim=2,
    )
    wall_rects = torch.stack((lower, upper), dim=2)
    wall_exists = torch.stack(
        (lower_y1 > -cfg.world_y, upper_y0 < cfg.world_y), dim=2
    )

    body_projection = torch.einsum("brpd,brad->brpa", body_rects, axes)
    wall_projection = torch.einsum("bwhpd,brad->brwhpa", wall_rects, axes)
    body_projection = body_projection[:, :, None, None]
    wall_min_lower = _smooth_min_lower_bound(
        wall_projection, projection_temperature_m, dim=-2
    )
    wall_max_upper = _smooth_max_upper_bound(
        wall_projection, projection_temperature_m, dim=-2
    )
    body_min_lower = _smooth_min_lower_bound(
        body_projection, projection_temperature_m, dim=-2
    )
    body_max_upper = _smooth_max_upper_bound(
        body_projection, projection_temperature_m, dim=-2
    )
    directional_lower = torch.stack(
        (
            wall_min_lower - body_max_upper,
            body_min_lower - wall_max_upper,
        ),
        dim=-1,
    )
    alternatives = directional_lower.flatten(start_dim=-2)
    pair_lower = _smooth_max_lower_bound(
        alternatives, axis_temperature_m, dim=-1
    )
    pair_lower = pair_lower.masked_fill(~wall_exists[:, None, :, :], float("inf"))
    constraint_groups = [pair_lower.reshape(batch, -1)]

    if obstacle_rects is not None and obstacle_rects.shape[1] != 0:
        if obstacle_rects.shape[0] != batch:
            raise ValueError("obstacle_rects batch does not match state")
        # Obstacles are absent from the single-slit task.  Keeping their exact
        # pair margins in the compound soft minimum preserves the global lower
        # bound without claiming a smoothed obstacle-axis gradient.
        obstacle_clearance = _aabb_pair_clearance(
            body_rects, axes, obstacle_rects
        )
        constraint_groups.append(obstacle_clearance.reshape(batch, -1))

    vertices = body_rects.reshape(batch, -1, 2)
    boundary_sides = torch.stack(
        (
            cfg.world_x - vertices[..., 0],
            cfg.world_x + vertices[..., 0],
            cfg.world_y - vertices[..., 1],
            cfg.world_y + vertices[..., 1],
        ),
        dim=-1,
    )
    constraint_groups.append(boundary_sides.reshape(batch, -1))
    compound_constraints = torch.cat(constraint_groups, dim=1)
    if not bool(torch.isfinite(compound_constraints).logical_or(torch.isinf(compound_constraints)).all()):
        raise RuntimeError("SAT bundle constraints contain NaN values")
    return _smooth_min_lower_bound(
        compound_constraints, pair_temperature_m, dim=1
    )


def collision_sat_bundle_lower_bound(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    projection_temperature_m: float = 5.0e-6,
    axis_temperature_m: float = 2.5e-5,
    pair_temperature_m: float = 2.5e-5,
) -> Tensor:
    """Conservative smooth multi-active-axis pose-clearance surrogate."""

    return _collision_sat_bundle_lower_geometry(
        state,
        scenario.wall_x,
        scenario.gap_y,
        scenario.gap_half,
        cfg,
        scenario.obstacle_rects,
        projection_temperature_m=projection_temperature_m,
        axis_temperature_m=axis_temperature_m,
        pair_temperature_m=pair_temperature_m,
    )


def collision_mask(state: Tensor, scenario: PianoBatch, cfg: EnvConfig) -> Tensor:
    return _collision_mask_geometry(
        state,
        scenario.wall_x,
        scenario.gap_y,
        scenario.gap_half,
        cfg,
        scenario.obstacle_rects,
    )


def transition_collision_mask(
    state: Tensor,
    next_state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
) -> Tensor:
    """Audit a dynamics transition with exact SAT at configurable substeps.

    ``collision_substeps=1`` is the legacy endpoint check.  Larger values test
    evenly spaced interpolated poses (including ``next_state``), preventing the
    thin-wall tunnelling that an endpoint-only discrete simulator can miss.
    """

    substeps = max(1, int(cfg.collision_substeps))
    if substeps == 1:
        return collision_mask(next_state, scenario, cfg)
    batch = state.shape[0]
    fractions = torch.arange(
        1,
        substeps + 1,
        device=state.device,
        dtype=state.dtype,
    ) / float(substeps)
    interpolated = state[:, None, :] + fractions.view(1, -1, 1) * (next_state - state)[:, None, :]
    angle_delta = wrap_angle(next_state[:, 2] - state[:, 2])
    interpolated[:, :, 2] = wrap_angle(
        state[:, None, 2] + fractions.view(1, -1) * angle_delta[:, None]
    )
    flat_state = interpolated.reshape(batch * substeps, state.shape[-1])

    return (
        _collision_mask_geometry(
            flat_state,
            scenario.wall_x.repeat_interleave(substeps, dim=0),
            scenario.gap_y.repeat_interleave(substeps, dim=0),
            scenario.gap_half.repeat_interleave(substeps, dim=0),
            cfg,
            None
            if scenario.obstacle_rects is None
            else scenario.obstacle_rects.repeat_interleave(substeps, dim=0),
        )
        .view(batch, substeps)
        .any(dim=1)
    )


def transition_clearance_margin(
    state: Tensor,
    next_state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
) -> Tensor:
    """Return the minimum exact-SAT clearance over one swept transition.

    Interpolation uses the identical substep fractions and shortest wrapped
    angular displacement as ``transition_collision_mask``.  Unlike that
    boolean audit, this path avoids in-place writes so gradients can flow from
    the swept margin to ``next_state`` and, through dynamics, to forces.
    """

    substeps = max(1, int(cfg.collision_substeps))
    if substeps == 1:
        return collision_clearance_margin(next_state, scenario, cfg)
    batch = state.shape[0]
    fractions = torch.arange(
        1,
        substeps + 1,
        device=state.device,
        dtype=state.dtype,
    ) / float(substeps)
    linear_interpolation = (
        state[:, None, :] + fractions.view(1, -1, 1) * (next_state - state)[:, None, :]
    )
    angle_delta = wrap_angle(next_state[:, 2] - state[:, 2])
    interpolated_angle = wrap_angle(
        state[:, None, 2] + fractions.view(1, -1) * angle_delta[:, None]
    )
    interpolated = torch.cat(
        (
            linear_interpolation[:, :, :2],
            interpolated_angle.unsqueeze(-1),
            linear_interpolation[:, :, 3:],
        ),
        dim=-1,
    )
    flat_state = interpolated.reshape(batch * substeps, state.shape[-1])
    margin = _collision_clearance_geometry(
        flat_state,
        scenario.wall_x.repeat_interleave(substeps, dim=0),
        scenario.gap_y.repeat_interleave(substeps, dim=0),
        scenario.gap_half.repeat_interleave(substeps, dim=0),
        cfg,
        None
        if scenario.obstacle_rects is None
        else scenario.obstacle_rects.repeat_interleave(substeps, dim=0),
    )
    return margin.view(batch, substeps).amin(dim=1)


def transition_sat_bundle_lower_bound(
    state: Tensor,
    next_state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    projection_temperature_m: float = 5.0e-6,
    axis_temperature_m: float = 2.5e-5,
    pair_temperature_m: float = 2.5e-5,
    sweep_temperature_m: float = 2.5e-5,
) -> Tensor:
    """Conservative smooth lower bound over the sampled swept SAT margin.

    The pose-level bound blends nearly active SAT axes and convex pairs.  This
    function applies a further unnormalised soft minimum over the exact same
    interpolation samples as ``transition_clearance_margin``.  Therefore its
    value is no greater than the authoritative sampled transition clearance.
    """

    if not math.isfinite(sweep_temperature_m) or sweep_temperature_m <= 0.0:
        raise ValueError("sweep_temperature_m must be finite and positive")
    substeps = max(1, int(cfg.collision_substeps))
    if substeps == 1:
        return collision_sat_bundle_lower_bound(
            next_state,
            scenario,
            cfg,
            projection_temperature_m=projection_temperature_m,
            axis_temperature_m=axis_temperature_m,
            pair_temperature_m=pair_temperature_m,
        )
    batch = state.shape[0]
    fractions = torch.arange(
        1, substeps + 1, device=state.device, dtype=state.dtype
    ) / float(substeps)
    linear_interpolation = (
        state[:, None, :]
        + fractions.view(1, -1, 1) * (next_state - state)[:, None, :]
    )
    angle_delta = wrap_angle(next_state[:, 2] - state[:, 2])
    interpolated_angle = wrap_angle(
        state[:, None, 2] + fractions.view(1, -1) * angle_delta[:, None]
    )
    interpolated = torch.cat(
        (
            linear_interpolation[:, :, :2],
            interpolated_angle.unsqueeze(-1),
            linear_interpolation[:, :, 3:],
        ),
        dim=-1,
    )
    flat_state = interpolated.reshape(batch * substeps, state.shape[-1])
    margins = _collision_sat_bundle_lower_geometry(
        flat_state,
        scenario.wall_x.repeat_interleave(substeps, dim=0),
        scenario.gap_y.repeat_interleave(substeps, dim=0),
        scenario.gap_half.repeat_interleave(substeps, dim=0),
        cfg,
        None
        if scenario.obstacle_rects is None
        else scenario.obstacle_rects.repeat_interleave(substeps, dim=0),
        projection_temperature_m=projection_temperature_m,
        axis_temperature_m=axis_temperature_m,
        pair_temperature_m=pair_temperature_m,
    ).view(batch, substeps)
    return -sweep_temperature_m * torch.logsumexp(
        -margins / sweep_temperature_m, dim=1
    )


def success_mask(state: Tensor, scenario: PianoBatch, cfg: EnvConfig) -> Tensor:
    _, active_goal = active_endpoints(scenario)
    pos_ok = torch.linalg.vector_norm(state[:, 0:2] - active_goal[:, 0:2], dim=-1) < (
        cfg.success_pos_tol
    )
    angle_ok = wrap_angle(state[:, 2] - active_goal[:, 2]).abs() < cfg.success_angle_tol
    return pos_ok & angle_ok


def stopped_at_active_goal(state: Tensor, scenario: PianoBatch, cfg: EnvConfig) -> Tensor:
    """Pose success plus a conservative stop condition for round-trip switching."""

    speed_ok = torch.linalg.vector_norm(state[:, 3:5], dim=-1) < cfg.round_trip_speed_tol
    omega_ok = state[:, 5].abs() < cfg.round_trip_omega_tol
    return success_mask(state, scenario, cfg) & speed_ok & omega_ok


def update_goal_dwell(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    dwell: Tensor,
) -> tuple[Tensor, Tensor]:
    """Advance consecutive stopped-goal counts and report a stable arrival."""

    stable = stopped_at_active_goal(state, scenario, cfg)
    next_dwell = torch.where(stable, dwell + 1, torch.zeros_like(dwell))
    reached = next_dwell >= max(1, cfg.round_trip_dwell_steps)
    return next_dwell, reached
