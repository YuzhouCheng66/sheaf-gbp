"""Physical targets for the private receding-horizon local SE(2) planner."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .config import EnvConfig
from .env import contact_points, wrap_angle
from .payload_shapes import payload_rectangles_for_config, t_morphology_from_config
from .single_slit_cspace import cached_single_slit_cspace_solution


LOCAL_SE2_PLAN_DIM = 8
LOCAL_SE2_POSITION_SCALE_M = 0.05
LOCAL_SE2_ANGLE_SCALE_RAD = 0.10
LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS = 0.65
LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS = 1.40
LOCAL_SE2_TERMINAL_WALL_RESERVE_M = 0.030
LOCAL_SE2_COLLISION_GAP_RESERVE_M = 0.030


def _goal_compatible_exit_center(rotated_min_x: Tensor, target_y: Tensor, cfg: EnvConfig) -> Tensor:
    """Keep the physical reserve, but fit optional tracking slack in the goal.

    At fixed heading and target y, full-body clearance imposes a lower x
    bound, while the unchanged goal disk imposes an upper x bound. A 5 mm
    *extra tracking margin* must not push an otherwise feasible target outside
    their intersection. No safety reserve or supervisor tolerance is reduced.
    Empty intersections retain the old behavior, not a false feasibility claim.
    """
    legacy = float(cfg.wall_slab) + LOCAL_SE2_TERMINAL_WALL_RESERVE_M + .005 - rotated_min_x
    radius = cfg.local_corridor_exit_goal_radius_m
    if radius <= 0:
        return legacy
    minimum_x = float(cfg.wall_slab) + LOCAL_SE2_TERMINAL_WALL_RESERVE_M - rotated_min_x
    dy = target_y - cfg.goal_y
    maximum_x = cfg.goal_x + torch.sqrt((radius * radius - dy.square()).clamp_min(0.))
    slack = maximum_x - minimum_x
    adjust = (dy.abs() < radius) & (slack > 1e-5) & (.5 * slack < .005)
    # Preserve the original floating-point expression on all unchanged nodes.
    return torch.where(adjust, minimum_x + .5 * slack, legacy)


def _sinusoid_arc_extrema(a: Tensor, b: Tensor, start: Tensor, end: Tensor) -> tuple[Tensor, Tensor]:
    """Exact extrema of a*cos(theta)+b*sin(theta) on the closed unwrapped arc."""
    lo, hi = torch.minimum(start, end), torch.maximum(start, end)
    first, last = a * start.cos() + b * start.sin(), a * end.cos() + b * end.sin()
    phase = torch.atan2(b, a)
    period = 2. * torch.pi
    maximum_inside = torch.ceil((lo - phase) / period) <= torch.floor((hi - phase) / period)
    minimum_phase = phase + torch.pi
    minimum_inside = torch.ceil((lo - minimum_phase) / period) <= torch.floor((hi - minimum_phase) / period)
    amplitude = torch.sqrt(a.square() + b.square())
    lower = torch.where(minimum_inside, -amplitude, torch.minimum(first, last))
    upper = torch.where(maximum_inside, amplitude, torch.maximum(first, last))
    return lower, upper


def _position_goal_heading_control(
    vertices: Tensor, center_x: Tensor, center_y: Tensor, theta: Tensor, cfg: EnvConfig,
) -> tuple[Tensor, Tensor]:
    """Independent local interface control, not a trained planner replacement.

    If the unchanged goal disk and full-body exit halfspace have no intersection
    at the current heading, search a short in-place rotation. Every vertex's
    entire angular arc must stay beyond the same reserved wall plane and within
    arena y bounds. A feasible heading must then admit a translation into the
    goal disk at goal-y, retaining the body reserve. No future target is used.
    This is sufficient, not complete: no safe candidate leaves legacy behavior.
    """
    reserve_x = float(cfg.wall_slab) + LOCAL_SE2_TERMINAL_WALL_RESERVE_M
    radius = cfg.local_corridor_exit_goal_radius_m
    min_x = (theta.cos()[..., None] * vertices[:, 0]
             - theta.sin()[..., None] * vertices[:, 1]).amin(-1)
    needs_rotation = (center_x + min_x > reserve_x) & (cfg.goal_x + radius - (reserve_x - min_x) <= 1e-5)
    if radius <= 0 or not bool(needs_rotation.any()):
        return theta, torch.zeros_like(needs_rotation)
    offsets = torch.arange(-36, 37, device=theta.device, dtype=theta.dtype) * (torch.pi / 36.)
    candidate = theta[..., None] + offsets
    start, end = theta[..., None, None], candidate[..., None]
    vx, vy = vertices[:, 0], vertices[:, 1]
    arc_min_x, _ = _sinusoid_arc_extrema(vx, -vy, start, end)
    arc_min_y, arc_max_y = _sinusoid_arc_extrema(vy, vx, start, end)
    safe = (center_x[..., None] + arc_min_x.amin(-1) >= reserve_x + 2e-6)
    safe &= center_y[..., None] + arc_min_y.amin(-1) >= -cfg.world_y + 2e-6
    safe &= center_y[..., None] + arc_max_y.amax(-1) <= cfg.world_y - 2e-6
    end_x = candidate.cos()[..., None] * vx - candidate.sin()[..., None] * vy
    end_y = candidate.sin()[..., None] * vx + candidate.cos()[..., None] * vy
    # Reserve 2 mm inside the unchanged goal disk for subsequent tracking.
    safe &= reserve_x - end_x.amin(-1) <= cfg.goal_x + radius - .002
    safe &= cfg.goal_y + end_y.amin(-1) >= -cfg.world_y + 2e-6
    safe &= cfg.goal_y + end_y.amax(-1) <= cfg.world_y - 2e-6
    score = offsets.abs().expand_as(candidate).masked_fill(~safe, torch.inf)
    selected = candidate.gather(-1, score.argmin(-1, keepdim=True)).squeeze(-1)
    active = needs_rotation & safe.any(-1)
    return torch.where(active, selected, theta), active


def _finite_wall_slice_y_bounds(
    rectangles: Tensor,
    center_x: Tensor,
    center_y: Tensor,
    sine: Tensor,
    cosine: Tensor,
    wall_center_x: Tensor,
    slab_half_thickness: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact vertical extent of a compound body inside one finite wall slab."""

    body_x = rectangles[..., 0]
    body_y = rectangles[..., 1]
    world_x = center_x[..., None, None] + (
        cosine[..., None, None] * body_x
        - sine[..., None, None] * body_y
    )
    world_y = center_y[..., None, None] + (
        sine[..., None, None] * body_x
        + cosine[..., None, None] * body_y
    )
    lower_face = wall_center_x - float(slab_half_thickness)
    upper_face = wall_center_x + float(slab_half_thickness)
    inside = (world_x >= lower_face[..., None, None]) & (
        world_x <= upper_face[..., None, None]
    )
    candidates = [world_y.masked_fill(~inside, torch.nan)]
    next_x = torch.roll(world_x, shifts=-1, dims=-1)
    next_y = torch.roll(world_y, shifts=-1, dims=-1)
    delta_x = next_x - world_x
    safe_delta_x = torch.where(
        delta_x.abs() > 1.0e-12, delta_x, torch.ones_like(delta_x)
    )
    for face in (lower_face, upper_face):
        fraction = (face[..., None, None] - world_x) / safe_delta_x
        valid = (
            (delta_x.abs() > 1.0e-12)
            & (fraction >= 0.0)
            & (fraction <= 1.0)
        )
        intersection_y = world_y + fraction * (next_y - world_y)
        candidates.append(intersection_y.masked_fill(~valid, torch.nan))
    sample_y = torch.cat(candidates, dim=-1).flatten(start_dim=-2)
    intersects = torch.isfinite(sample_y).any(dim=-1)
    minimum = torch.nan_to_num(sample_y, nan=torch.inf).amin(dim=-1)
    maximum = torch.nan_to_num(sample_y, nan=-torch.inf).amax(dim=-1)
    return minimum, maximum, intersects


def _alignment_sweep_is_safe(
    rectangles, center_x, center_y, sine, cosine, wall_x, gap_y, gap_half,
    target_y, target_theta, cfg, *, position_gain, angle_gain, max_speed, max_omega,
):
    """Screen the original x-held alignment from the current local pose.

    Containment of the current wall slice is not permission to rotate it.
    This is only a local candidate screen; the independent continuous-motion
    environment guard remains authoritative after force allocation and GBP.
    """
    samples = cfg.local_cspace_collision_substeps
    if samples < 1:
        raise ValueError("alignment sample count must be positive")
    theta = torch.atan2(sine, cosine)
    vy = (position_gain * (target_y - center_y)).clamp(-max_speed, max_speed)
    omega = (angle_gain * wrap_angle(target_theta - theta)).clamp(-max_omega, max_omega)
    fractions = torch.linspace(1. / samples, 1., samples, device=center_x.device,
                               dtype=center_x.dtype)
    x = center_x[..., None].expand(*center_x.shape, samples)
    y = center_y[..., None] + cfg.dt * vy[..., None] * fractions
    angles = theta[..., None] + cfg.dt * omega[..., None] * fractions
    lo, hi, intersects = _finite_wall_slice_y_bounds(
        rectangles, x, y, angles.sin(), angles.cos(), wall_x[..., None], cfg.wall_slab,
    )
    reserve = 2e-6
    safe = (~intersects) | (
        (lo >= (gap_y - gap_half)[..., None] + reserve)
        & (hi <= (gap_y + gap_half)[..., None] - reserve)
    )
    vertices = rectangles.reshape(-1, 2)
    body_y = (angles.sin()[..., None] * vertices[:, 0]
              + angles.cos()[..., None] * vertices[:, 1] + y[..., None])
    return (safe & (body_y.abs() < cfg.world_y - reserve).all(-1)).all(-1)


def _collision_safe_cspace_targets(
    path: Tensor,
    nearest: Tensor,
    rectangles: Tensor,
    relative_x: Tensor,
    relative_y: Tensor,
    current_theta: Tensor,
    gap_center_y: Tensor,
    gap_half: Tensor,
    cfg: EnvConfig,
    *,
    position_gain: float,
    angle_gain: float,
    max_speed: float,
    max_omega: float,
    lookahead_offsets: tuple[int, ...] = (32, 16, 8, 4, 2, 1, -1, -2, -4),
) -> Tensor:
    """Choose the furthest one-step-safe path target from current geometry.

    This is a tiny receding-horizon collision check, not a progress latch.
    Every candidate is regenerated from the current pose, and the selected
    index is discarded after this control call.  Forward lookaheads are tried
    from long to short; if every forward proposal is blocked, short reverse
    targets implement a geometry-grounded contact retreat.
    """

    offsets = torch.tensor(
        lookahead_offsets,
        dtype=torch.long,
        device=path.device,
    )
    candidate_indices = (
        nearest.unsqueeze(-1) + offsets.unsqueeze(0)
    ).clamp(0, path.shape[0] - 1)
    candidates = path.index_select(0, candidate_indices.reshape(-1)).reshape(
        nearest.shape[0], offsets.shape[0], 3
    )
    position_error = candidates[..., :2] - torch.stack(
        (relative_x, relative_y), dim=-1
    ).unsqueeze(1)
    desired_velocity = position_gain * position_error
    speed = torch.linalg.vector_norm(desired_velocity, dim=-1, keepdim=True).clamp_min(
        1.0e-8
    )
    desired_velocity = desired_velocity * (max_speed / speed).clamp_max(1.0)
    heading_error = wrap_angle(candidates[..., 2] - current_theta.unsqueeze(-1))
    desired_omega = (angle_gain * heading_error).clamp(-max_omega, max_omega)
    predicted_x = relative_x.unsqueeze(-1) + float(cfg.dt) * desired_velocity[..., 0]
    predicted_y = relative_y.unsqueeze(-1) + float(cfg.dt) * desired_velocity[..., 1]
    predicted_theta = current_theta.unsqueeze(-1) + float(cfg.dt) * desired_omega

    samples = cfg.local_cspace_collision_substeps
    if samples < 1:
        raise ValueError("local C-space collision samples must be positive")
    fractions = torch.linspace(
        1. / samples, 1.0, samples, dtype=path.dtype, device=path.device
    ).view(1, 1, -1)
    sample_x = relative_x[:, None, None] + fractions * (
        predicted_x - relative_x.unsqueeze(-1)
    ).unsqueeze(-1)
    sample_y = relative_y[:, None, None] + fractions * (
        predicted_y - relative_y.unsqueeze(-1)
    ).unsqueeze(-1)
    sample_heading_delta = wrap_angle(
        predicted_theta - current_theta.unsqueeze(-1)
    )
    sample_theta = current_theta[:, None, None] + fractions * sample_heading_delta.unsqueeze(-1)
    sine = torch.sin(sample_theta)
    cosine = torch.cos(sample_theta)
    slice_min, slice_max, intersects = _finite_wall_slice_y_bounds(
        rectangles,
        sample_x,
        sample_y,
        sine,
        cosine,
        torch.zeros_like(sample_x),
        float(cfg.wall_slab),
    )
    numerical_reserve = 2.0e-6
    wall_safe = (~intersects) | (
        (slice_min >= -gap_half[:, None, None] + numerical_reserve)
        & (slice_max <= gap_half[:, None, None] - numerical_reserve)
    )
    vertices = rectangles.reshape(-1, 2)
    rotated_y = (
        sine.unsqueeze(-1) * vertices[:, 0]
        + cosine.unsqueeze(-1) * vertices[:, 1]
    )
    world_vertex_y = (
        gap_center_y[:, None, None, None]
        + sample_y.unsqueeze(-1)
        + rotated_y
    )
    boundary_safe = (
        (world_vertex_y > -float(cfg.world_y) + numerical_reserve)
        & (world_vertex_y < float(cfg.world_y) - numerical_reserve)
    ).all(dim=-1)
    safe = (wall_safe & boundary_safe).all(dim=-1)
    any_safe = safe.any(dim=-1)
    first_safe = safe.to(torch.int64).argmax(dim=-1)
    # The current pose is collision-free by the feedback contract.  Holding
    # it is therefore the fail-closed fallback if even a one-step retreat is
    # not certified.
    selected = candidates[
        torch.arange(candidates.shape[0], device=path.device), first_safe
    ]
    current = torch.stack((relative_x, relative_y, current_theta), dim=-1)
    return torch.where(any_safe.unsqueeze(-1), selected, current)


def local_se2_terminal_attractor_plan(
    obs: Tensor,
    learned_plan: Tensor,
    cfg: EnvConfig,
    *,
    position_gain: float,
    angle_gain: float,
    max_speed: float,
    max_omega: float,
    feasible_corridor: bool = False,
    route_free_geometry: bool = False,
    corridor_reserve_m: float = 0.015,
    corridor_max_speed: float = 0.50,
) -> tuple[Tensor, Tensor]:
    """Select a local analytic plan only when geometry proves it is safe.

    Every node reconstructs the payload centre from its own world handle
    position, known body handle coordinate, and observed orientation.  The
    exact compound-payload leftmost x support then supplies a common physical
    event: the complete body is beyond the reserved wall.  Only after that
    event is the learned path plan replaced by a bounded proportional next
    twist toward the shared mission goal
    ``(cfg.goal_x, cfg.goal_y, cfg.goal_theta)``.

    The optional pre-passage corridor mode is also strictly local.  A node
    uses only its first radius-limited wall record to reconstruct the visible
    slit.  It proposes an align--cross--recenter passage iff the complete exact
    compound payload fits in that slit with ``corridor_reserve_m`` on both
    sides.  The target payload y is the point nearest zero in the certified
    feasible interval.  If the current rotated payload is not yet vertically
    contained in the opening, it first moves laterally with zero forward
    velocity.  This matters for one-sided openings: simultaneous diagonal
    motion can reach the wall before completing the required alignment even
    when the final pose itself fits.  Invisible and infeasible rows retain
    ``learned_plan`` exactly; their peers can receive the physical proposal
    only through the ordinary Sheaf-GBP wrench messages downstream.

    No peer observation, global simulator state, teacher label, force, or
    wrench is consumed.  The returned plan is still executed by the ordinary
    local-evidence -> Sheaf-GBP -> exact-grasp force layer.
    """

    if obs.ndim != 3 or obs.shape[-1] < 18:
        raise ValueError("terminal attractor requires strict-local observations")
    if learned_plan.shape != (*obs.shape[:2], LOCAL_SE2_PLAN_DIM):
        raise ValueError("terminal attractor requires plan shape [batch,agent,8]")
    if any(value <= 0.0 for value in (position_gain, angle_gain, max_speed, max_omega)):
        raise ValueError("terminal attractor gains and limits must be positive")
    if corridor_reserve_m <= 0.0 or corridor_max_speed <= 0.0:
        raise ValueError("corridor reserve and speed must be positive")

    sine = obs[..., 0]
    cosine = obs[..., 1]
    handle_body_x = obs[..., 2]
    handle_body_y = obs[..., 3]
    offset_world_x = cosine * handle_body_x - sine * handle_body_y
    offset_world_y = sine * handle_body_x + cosine * handle_body_y
    contact_world_x = 2.6 * obs[..., 4]
    contact_world_y = cfg.world_y * obs[..., 5]
    center_x = contact_world_x - offset_world_x
    center_y = contact_world_y - offset_world_y

    rectangles = payload_rectangles_for_config(cfg, obs.device).to(obs.dtype)
    vertices = rectangles.reshape(-1, 2)
    world_vertex_x = (
        cosine.unsqueeze(-1) * vertices[:, 0]
        - sine.unsqueeze(-1) * vertices[:, 1]
    )
    leftmost_x = center_x + world_vertex_x.amin(dim=-1)
    terminal_active = (
        leftmost_x
        > float(cfg.wall_slab) + LOCAL_SE2_TERMINAL_WALL_RESERVE_M
    )

    # Preserve the sealed historical controller only for the *exact*
    # canonical T morphology.  A thinner, taller, or wider T is a genuine
    # geometry OOD case and must use the same route-free support planner as
    # every other compound body; keying this branch on the letter label made
    # the public shape sliders silently disable generalization.
    shape_corridor_mode = route_free_geometry or not (
        cfg.payload_shape == "T" and t_morphology_from_config(cfg).is_canonical
    )
    corridor_active = torch.zeros_like(terminal_active)
    corridor_target_y = torch.zeros_like(center_y)
    corridor_target_theta = torch.zeros_like(center_y)
    corridor_aligned = torch.zeros_like(terminal_active)
    corridor_rotation_safe = torch.ones_like(terminal_active)
    corridor_safe_center_x = center_x
    corridor_explicit = torch.zeros_like(terminal_active)
    corridor_explicit_target_x = center_x
    corridor_tracking_gain = torch.full_like(
        center_y, max(position_gain, 4.0) if shape_corridor_mode else position_gain
    )
    corridor_tracking_angle_gain = torch.full_like(
        center_y, max(angle_gain, 4.0) if shape_corridor_mode else angle_gain
    )
    if feasible_corridor:
        # The strict-local 18-D schema is
        # [sin, cos, handle_body(2), handle_world(2), handle_velocity(2),
        #  goal_delta(2), wall_dx(2), wall_dy(2), wall_half(2), visible(2)].
        # Hence this reconstruction uses only one node's own visible wall0.
        visible = obs[..., 16] > 0.5
        gap_center_y = contact_world_y + cfg.world_y * obs[..., 12]
        # Exact wall collision contracts the rendered opening by 0.030 m on
        # each side (see env._wall_rectangles).  Certify against that physical
        # free opening, then add the controller's own reserve inside it.
        gap_half = (
            cfg.world_y * obs[..., 14]
            - LOCAL_SE2_COLLISION_GAP_RESERVE_M
        ).clamp_min(0.0)

        body_y = vertices[:, 1]
        body_min_y = body_y.amin()
        body_max_y = body_y.amax()
        feasible_center_lower = (
            gap_center_y - gap_half - body_min_y + corridor_reserve_m
        )
        feasible_center_upper = (
            gap_center_y + gap_half - body_max_y - corridor_reserve_m
        )
        corridor_active = visible & (
            feasible_center_lower <= feasible_center_upper
        )
        mission_goal_y = torch.full_like(center_y, float(cfg.goal_y))
        corridor_target_y = torch.maximum(
            feasible_center_lower,
            torch.minimum(mission_goal_y, feasible_center_upper),
        )

        # Final-pose feasibility alone is not a swept-path certificate for an
        # off-centre slit.  Require the current rotated support to enter a
        # capture corridor before permitting forward motion.  The target has
        # the full configured reserve while this test uses half, giving the
        # bounded controller a finite capture band rather than an asymptotic
        # equality target.
        world_vertex_y = (
            sine.unsqueeze(-1) * vertices[:, 0]
            + cosine.unsqueeze(-1) * vertices[:, 1]
        )
        current_body_min_y = center_y + world_vertex_y.amin(dim=-1)
        current_body_max_y = center_y + world_vertex_y.amax(dim=-1)
        alignment_reserve = 0.5 * corridor_reserve_m
        corridor_aligned = corridor_active & (
            current_body_min_y
            >= gap_center_y - gap_half + alignment_reserve
        ) & (
            current_body_max_y
            <= gap_center_y + gap_half - alignment_reserve
        )

        # A general block glyph may be too tall at the goal orientation but
        # fit after rotating.  Select that passage orientation online from
        # the exact compound geometry.  This is a receding-horizon geometric
        # calculation, not a letter lookup or a stored route: the same search
        # is applied to every non-T body at every visible-wall tick.
        #
        # Canonical and parametric T retain the historical branch above
        # exactly.  This preserves the sealed T controller while giving the
        # new A-Z morphology branch a route-free structural prior.
        if shape_corridor_mode:
            angle_count = 72
            candidate_theta = torch.arange(
                angle_count, device=obs.device, dtype=obs.dtype
            ) * (2.0 * torch.pi / float(angle_count)) - torch.pi
            candidate_cos = torch.cos(candidate_theta)
            candidate_sin = torch.sin(candidate_theta)
            candidate_x = (
                candidate_cos[:, None] * vertices[None, :, 0]
                - candidate_sin[:, None] * vertices[None, :, 1]
            )
            candidate_y = (
                candidate_sin[:, None] * vertices[None, :, 0]
                + candidate_cos[:, None] * vertices[None, :, 1]
            )
            candidate_min_x = candidate_x.amin(dim=-1)
            candidate_max_x = candidate_x.amax(dim=-1)
            candidate_min_y = candidate_y.amin(dim=-1)
            candidate_max_y = candidate_y.amax(dim=-1)
            candidate_span_y = candidate_max_y - candidate_min_y

            # The active goal x is reconstructible at every handle.  Its sign
            # supplies only passage direction; no absolute start, time, route
            # index, or glyph identity is present.
            goal_delta_x = 2.6 * obs[..., 8] + offset_world_x
            direction = torch.where(
                goal_delta_x >= 0.0,
                torch.ones_like(goal_delta_x),
                -torch.ones_like(goal_delta_x),
            )
            # Glyphs are intentionally close to the 0.720 m limit (A has
            # about 0.0125 m physical clearance per side when horizontal).
            # Keep a strict positive geometric reserve, but do not inherit
            # the canonical T controller's much larger 0.050 m dynamic
            # margin when that would incorrectly declare a feasible body
            # impossible.
            minimum_shape_reserve_m = 0.002
            feasible_angle = candidate_span_y.view(1, 1, -1) <= (
                2.0 * (gap_half - minimum_shape_reserve_m)
            ).unsqueeze(-1)
            # First maximize vertical clearance, then prefer the smallest
            # rotation from the final mission heading.  The tiny final term only
            # breaks the +/- pi/2 symmetry consistently with travel direction.
            mission_goal_theta = (
                torch.full_like(center_y, float(cfg.goal_theta))
                if cfg.goal_orientation_required
                else torch.atan2(sine, cosine)
            )
            candidate_goal_angle = wrap_angle(
                candidate_theta.view(1, 1, -1)
                - mission_goal_theta.unsqueeze(-1)
            )
            score = (
                candidate_span_y.view(1, 1, -1)
                + 0.005 * candidate_goal_angle.abs()
                + 1.0e-4
                * direction.unsqueeze(-1)
                * candidate_theta.view(1, 1, -1)
            )
            score = score.masked_fill(~feasible_angle, float("inf"))
            selected_index = score.argmin(dim=-1)
            any_feasible_angle = feasible_angle.any(dim=-1)

            def select(values: Tensor) -> Tensor:
                return values.index_select(0, selected_index.reshape(-1)).view_as(
                    selected_index
                )

            selected_theta = select(candidate_theta)
            selected_min_y = select(candidate_min_y)
            selected_max_y = select(candidate_max_y)
            selected_span_y = selected_max_y - selected_min_y
            available_reserve = 0.45 * (
                2.0 * gap_half - selected_span_y
            ).clamp_min(2.0 * minimum_shape_reserve_m)
            shape_reserve = available_reserve.clamp(
                min=minimum_shape_reserve_m,
                max=corridor_reserve_m,
            )
            shape_lower = (
                gap_center_y - gap_half - selected_min_y + shape_reserve
            )
            shape_upper = (
                gap_center_y + gap_half - selected_max_y - shape_reserve
            )
            shape_target_y = torch.maximum(
                shape_lower,
                torch.minimum(mission_goal_y, shape_upper),
            )
            shape_active = visible & any_feasible_angle

            # Rotate only while the full target-oriented body is clear of the
            # wall slab.  If the learned approach came too close, first back
            # away without rotating; this makes the subsequent in-place
            # alignment a swept-path certificate in x rather than merely a
            # feasible endpoint.
            wall_center_x = contact_world_x + (
                cfg.wall_sensor_range * obs[..., 10]
            )
            rotation_capture_reserve_m = 0.040
            # Certify the complete zero-heading -> passage-heading rotation
            # arc, not merely its endpoint.  This catches intermediate serif
            # support for I/Y without the excessive conservatism of a full
            # bounding circle.  The one-grid-cell pad covers interpolation
            # between the 5-degree support samples.
            angle_step = 2.0 * torch.pi / float(angle_count)
            arc_mask = (
                candidate_theta.view(1, 1, -1)
                * selected_theta.unsqueeze(-1)
                >= -1.0e-7
            ) & (
                candidate_theta.abs().view(1, 1, -1)
                <= selected_theta.abs().unsqueeze(-1) + angle_step
            )
            swept_max_x = candidate_max_x.view(1, 1, -1).masked_fill(
                ~arc_mask, -torch.inf
            ).amax(dim=-1)
            swept_min_x = candidate_min_x.view(1, 1, -1).masked_fill(
                ~arc_mask, torch.inf
            ).amin(dim=-1)
            rightward_safe_x = (
                wall_center_x
                - float(cfg.wall_slab)
                - shape_reserve
                - rotation_capture_reserve_m
                - swept_max_x
            )
            leftward_safe_x = (
                wall_center_x
                + float(cfg.wall_slab)
                + shape_reserve
                + rotation_capture_reserve_m
                - swept_min_x
            )
            shape_safe_x = torch.where(
                direction > 0.0, rightward_safe_x, leftward_safe_x
            )
            # Proportional motion approaches the safe centre asymptotically;
            # use a small capture band inside the already reserved geometric
            # envelope rather than requiring an impossible floating equality.
            shape_rotation_safe = (
                direction * (center_x - shape_safe_x) <= 0.005
            )
            heading_to_passage = wrap_angle(selected_theta - torch.atan2(sine, cosine))

            # The target centre already lies inside ``shape_reserve``.  The
            # crossing guard certifies only the part of the current compound
            # body that intersects the finite wall slab.  Whole-body vertical
            # support is overly conservative for non-convex glyphs whose
            # branches cross the slab sequentially.
            shape_alignment_reserve = torch.minimum(
                0.5 * shape_reserve,
                torch.full_like(shape_reserve, 0.0005),
            )
            (
                current_slice_min_y,
                current_slice_max_y,
                current_slice_intersects,
            ) = _finite_wall_slice_y_bounds(
                rectangles,
                center_x,
                center_y,
                sine,
                cosine,
                wall_center_x,
                float(cfg.wall_slab),
            )
            current_shape_contained = (
                shape_active
                & (
                    ~current_slice_intersects
                    | (
                        (
                            current_slice_min_y
                            >= gap_center_y - gap_half
                        )
                        & (
                            current_slice_max_y
                            <= gap_center_y + gap_half
                        )
                    )
                )
            )
            current_shape_min_y = center_y + world_vertex_y.amin(dim=-1)
            current_shape_max_y = center_y + world_vertex_y.amax(dim=-1)
            current_whole_body_contained = (
                shape_active
                & (
                    current_shape_min_y
                    >= gap_center_y - gap_half + shape_alignment_reserve
                )
                & (
                    current_shape_max_y
                    <= gap_center_y + gap_half - shape_alignment_reserve
                )
            )
            # Entry/crossing requires the robust whole-body corridor that was
            # certified by ``shape_active``.  The finite-slice predicate below
            # is deliberately used only to decide whether re-alignment can be
            # performed in place; it must not authorize forward motion while
            # the next branch is still outside the opening.
            shape_aligned = current_whole_body_contained & (
                heading_to_passage.abs() <= 0.050
            )
            # The retreat line certifies that an initially misaligned body can
            # rotate safely.  Once the *current* exact body is aligned and
            # contained, crossing that historical line must not reactivate a
            # retreat.  Otherwise the controller forms a two-mode limit cycle:
            # advance past the line, retreat, advance again.
            # If small tracking error makes the heading leave its 0.05 rad
            # capture band after crossing begins, hold x and re-align at the
            # current contained configuration; never command a full retreat
            # merely because that instantaneous heading predicate toggled.
            contained_alignment_safe = current_shape_contained
            if cfg.local_support_check_alignment_sweep:
                contained_alignment_safe = contained_alignment_safe & _alignment_sweep_is_safe(
                    rectangles, center_x, center_y, sine, cosine, wall_center_x,
                    gap_center_y, gap_half, shape_target_y, selected_theta, cfg,
                    position_gain=max(position_gain, 4.), angle_gain=max(angle_gain, 4.),
                    max_speed=corridor_max_speed, max_omega=max(max_omega, 1.40),
                )
            shape_rotation_safe = shape_rotation_safe | contained_alignment_safe

            corridor_active = torch.where(shape_active, shape_active, corridor_active)
            corridor_target_y = torch.where(
                shape_active, shape_target_y, corridor_target_y
            )
            corridor_target_theta = torch.where(
                shape_active, selected_theta, corridor_target_theta
            )
            corridor_aligned = torch.where(
                shape_active, shape_aligned, corridor_aligned
            )
            corridor_rotation_safe = torch.where(
                shape_active, shape_rotation_safe, corridor_rotation_safe
            )
            corridor_safe_center_x = torch.where(
                shape_active, shape_safe_x, corridor_safe_center_x
            )

            # Replace the conservative whole-body projection maneuver with a
            # geometry-derived configuration-space route whenever a wall is
            # visible.  This handles genuinely non-convex passages (I/Y),
            # where no single orientation fits wholly inside the opening and
            # different branches must cross the finite slab sequentially.
            # The path is computed online from rectangle coordinates and wall
            # dimensions; it is absent from the checkpoint and queried from
            # the current pose with no carried progress index.
            cspace_active = torch.zeros_like(visible)
            cspace_target_x = center_x.clone()
            cspace_target_y = center_y.clone()
            cspace_target_theta = torch.atan2(sine, cosine).clone()
            geometry_key = tuple(
                float(value)
                for value in rectangles.detach().to("cpu").reshape(-1).tolist()
            )
            # Prefer the smooth support-function maneuver whenever the whole
            # body admits a robust passage orientation.  Invoke the heavier
            # C-space route only for genuinely non-convex threading cases;
            # replacing an already robust maneuver with an arbitrary feasible
            # lattice path needlessly destroys clearance and path efficiency.
            cspace_needed = visible & ~shape_active
            for batch_index in range(obs.shape[0]):
                visible_index = torch.where(cspace_needed[batch_index])[0]
                if int(visible_index.numel()) == 0:
                    continue
                reference_agent = int(visible_index[0])
                effective_half = float(
                    gap_half[batch_index, reference_agent].detach().cpu()
                )
                try:
                    path_numpy, cspace_planning_safety = cached_single_slit_cspace_solution(
                        geometry_key,
                        int(rectangles.shape[0]),
                        round(effective_half, 6),
                        round(float(cfg.wall_slab), 6),
                        round(float(cfg.world_y), 6),
                    )
                except RuntimeError:
                    # The support-function maneuver above remains a safe
                    # fallback for an under-resolved or truly infeasible map.
                    continue
                # The cached NumPy route is deliberately read-only.  Copy it
                # before exposing it to Torch so an accidental in-place op
                # can never mutate the geometry cache.
                path = torch.tensor(
                    path_numpy.copy(), dtype=obs.dtype, device=obs.device
                )
                rows = visible_index
                wall_x_rows = wall_center_x[batch_index, rows]
                gap_y_rows = gap_center_y[batch_index, rows]
                relative_x = center_x[batch_index, rows] - wall_x_rows
                relative_y = center_y[batch_index, rows] - gap_y_rows
                current_theta = torch.atan2(
                    sine[batch_index, rows], cosine[batch_index, rows]
                )
                distance2 = (
                    (path[None, :, 0] - relative_x[:, None]).square()
                    + (path[None, :, 1] - relative_y[:, None]).square()
                    + (
                        0.35
                        * wrap_angle(path[None, :, 2] - current_theta[:, None])
                    ).square()
                )
                nearest = distance2.argmin(dim=-1)
                # Dense C-space fibers remove the old endpoint jump.  Use a
                # fast 16-sample lookahead when the path has metric clearance.
                # At the mathematical feasibility boundary there is no room
                # to cut a lattice corner, so track four samples at a time and
                # let exact collision feedback reject any remaining numerical
                # overshoot.  This decision is based on certified clearance,
                # never on a shape/letter identity.
                if (cspace_planning_safety >= 0.005
                        and not cfg.local_cspace_validate_all_lookaheads):
                    target_index = (nearest + 16).clamp_max(path.shape[0] - 1)
                    target = path.index_select(0, target_index)
                else:
                    target = _collision_safe_cspace_targets(
                        path,
                        nearest,
                        rectangles,
                        relative_x,
                        relative_y,
                        current_theta,
                        gap_center_y[batch_index, rows],
                        gap_half[batch_index, rows],
                        cfg,
                        position_gain=(8.0 if cspace_planning_safety < 0.005 else
                                       float(corridor_tracking_gain[batch_index, reference_agent])),
                        angle_gain=(8.0 if cspace_planning_safety < 0.005 else
                                    float(corridor_tracking_angle_gain[batch_index, reference_agent])),
                        max_speed=corridor_max_speed,
                        max_omega=max(max_omega, 1.40),
                        # A clear reference path does not certify the rejoin
                        # from the actual tracked pose. Preserve its original
                        # gain/16-step lookahead, screening shorter and reverse
                        # alternatives with the same local geometry checker.
                        lookahead_offsets=((32, 16, 8, 4, 2, 1, -1, -2, -4)
                                           if cspace_planning_safety < 0.005 else
                                           (16, 8, 4, 2, 1, -1, -2, -4)),
                    )
                cspace_active[batch_index, rows] = True
                cspace_target_x[batch_index, rows] = wall_x_rows + target[:, 0]
                cspace_target_y[batch_index, rows] = gap_y_rows + target[:, 1]
                cspace_target_theta[batch_index, rows] = target[:, 2]
                if cspace_planning_safety < 0.005:
                    corridor_tracking_gain[batch_index, rows] = 8.0
                    corridor_tracking_angle_gain[batch_index, rows] = 8.0

            corridor_active = torch.where(
                cspace_active, cspace_active, corridor_active
            )
            corridor_explicit = cspace_active
            corridor_explicit_target_x = torch.where(
                cspace_active, cspace_target_x, corridor_explicit_target_x
            )
            corridor_target_y = torch.where(
                cspace_active, cspace_target_y, corridor_target_y
            )
            corridor_target_theta = torch.where(
                cspace_active, cspace_target_theta, corridor_target_theta
            )
            corridor_rotation_safe = torch.where(
                cspace_active, torch.ones_like(cspace_active), corridor_rotation_safe
            )

    # The post-clearance terminal attractor takes precedence and returns to
    # the supplied mission goal. Before clearance, a feasible corridor uses
    # the nearest certified centre to that goal.
    active = terminal_active | corridor_active
    target_y = torch.where(
        terminal_active,
        torch.full_like(center_y, float(cfg.goal_y)),
        corridor_target_y,
    )
    target_x = torch.where(
        terminal_active | (corridor_aligned & ~corridor_explicit),
        torch.full_like(center_x, float(cfg.goal_x)),
        torch.where(
            corridor_explicit,
            corridor_explicit_target_x,
            torch.where(corridor_rotation_safe, center_x, corridor_safe_center_x),
        ),
    )
    if (cfg.complete_local_corridor_exit and not cfg.goal_orientation_required
            and cfg.goal_x > cfg.wall_slab):
        # A position-only handoff can lie before the centre required to clear
        # a large rotated body's trailing support. Do not stop its crossing
        # drive at that x while the supervisor still requires a cleared body.
        # Keep the same local corridor and hold this x lower bound during
        # recentering; otherwise the goal spring immediately pulls the tail
        # back into the just-cleared plane. Explicit C-space entry targets are
        # untouched. Final oriented parking retains the original behavior.
        exit_center_x = _goal_compatible_exit_center(world_vertex_x.amin(dim=-1), target_y, cfg)
        completion = terminal_active | (corridor_active & corridor_aligned & ~corridor_explicit)
        target_x = torch.where(completion, torch.maximum(target_x, exit_center_x), target_x)
    target_y = torch.where(
        corridor_active & ~corridor_rotation_safe,
        center_y,
        target_y,
    )
    target_theta = torch.where(
        terminal_active,
        (
            torch.full_like(center_y, float(cfg.goal_theta))
            if cfg.goal_orientation_required
            else torch.atan2(sine, cosine)
        ),
        torch.where(
            corridor_rotation_safe,
            corridor_target_theta,
            torch.atan2(sine, cosine),
        ),
    )

    if (cfg.local_corridor_goal_heading_search and cfg.complete_local_corridor_exit
            and not cfg.goal_orientation_required):
        recovery_theta, rotate_in_place = _position_goal_heading_control(
            vertices, center_x, center_y, torch.atan2(sine, cosine), cfg,
        )
        # Position-only means no preferred heading, not a prohibition on rotation.
        target_x = torch.where(rotate_in_place, center_x, target_x)
        target_y = torch.where(rotate_in_place, center_y, target_y)
        target_theta = torch.where(rotate_in_place, recovery_theta, target_theta)

    position_error_world = torch.stack(
        (target_x - center_x, target_y - center_y),
        dim=-1,
    )
    default_position_gain = (
        max(position_gain, 4.0) if shape_corridor_mode else position_gain
    )
    effective_position_gain = torch.where(
        corridor_explicit,
        corridor_tracking_gain,
        torch.full_like(center_y, default_position_gain),
    )
    desired_velocity_world = effective_position_gain.unsqueeze(-1) * position_error_world
    speed = torch.linalg.vector_norm(
        desired_velocity_world, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    selected_max_speed = torch.where(
        corridor_active & ~terminal_active,
        torch.full_like(center_x, corridor_max_speed),
        torch.full_like(center_x, max_speed),
    ).unsqueeze(-1)
    desired_velocity_world = desired_velocity_world * (
        selected_max_speed / speed
    ).clamp_max(1.0)
    desired_velocity_body = torch.stack(
        (
            cosine * desired_velocity_world[..., 0]
            + sine * desired_velocity_world[..., 1],
            -sine * desired_velocity_world[..., 0]
            + cosine * desired_velocity_world[..., 1],
        ),
        dim=-1,
    )
    heading_error = wrap_angle(target_theta - torch.atan2(sine, cosine))
    default_angle_gain = max(angle_gain, 4.0) if shape_corridor_mode else angle_gain
    effective_angle_gain = torch.where(
        corridor_explicit,
        corridor_tracking_angle_gain,
        torch.full_like(center_y, default_angle_gain),
    )
    effective_max_omega = max(max_omega, 1.40) if shape_corridor_mode else max_omega
    desired_omega = (effective_angle_gain * heading_error).clamp(
        -effective_max_omega, effective_max_omega
    )
    terminal_six = torch.cat(
        (
            (
                desired_velocity_body
                * cfg.dt
                / LOCAL_SE2_POSITION_SCALE_M
            ).clamp(-1.0, 1.0),
            (
                desired_omega * cfg.dt / LOCAL_SE2_ANGLE_SCALE_RAD
            ).clamp(-1.0, 1.0).unsqueeze(-1),
            (
                desired_velocity_body / LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
            ).clamp(-1.0, 1.0),
            (
                desired_omega / LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS
            ).clamp(-1.0, 1.0).unsqueeze(-1),
        ),
        dim=-1,
    )
    terminal_plan = torch.cat((terminal_six, learned_plan[..., 6:8]), dim=-1)
    active = active.unsqueeze(-1)
    selected = torch.where(active, terminal_plan, learned_plan)
    if not bool(torch.isfinite(selected).all()):
        raise RuntimeError("terminal attractor produced a non-finite plan")
    return selected, active


def local_se2_inverse_dynamics_wrench_proposals(
    obs: Tensor,
    plan: Tensor,
    cfg: EnvConfig,
) -> Tensor:
    """Return strictly-local sufficient statistics for total payload wrench.

    Agent ``i`` observes only the velocity of its own rigid handle,
    ``c_i = A_i [v_x,v_y,omega]``.  The fixed morphology makes the normal
    matrix ``M = sum_j A_j^T A_j`` common knowledge.  Thus
    ``N M^-1 A_i^T c_i`` is a local sufficient statistic whose mean is the
    exact payload twist.  This remains true for an asymmetric T-shaped handle
    layout (and for any full-rank fixed morphology); no zero-centroid shortcut
    is assumed.  Applying one-step inverse dynamics to each local statistic
    makes the mean proposal exactly the desired global wrench.  Sheaf-GBP
    performs that fusion over the real radius graph; this function performs
    no peer aggregation and reads no global state.

    The result uses the correction section's normalized physical convention:
    ``[Fx/(N fmax), Fy/(N fmax), tau/(0.75 N fmax)]``.
    """

    if obs.ndim != 3 or obs.shape[-1] < 18:
        raise ValueError("inverse-dynamics proposals require strict-local observations")
    if plan.shape != (*obs.shape[:2], LOCAL_SE2_PLAN_DIM):
        raise ValueError("local SE(2) plans must have shape [batch,agent,8]")
    if obs.shape[1] != cfg.n_agents:
        raise ValueError("observation agent count does not match environment")
    if not cfg.strict_local_observation:
        raise ValueError("inverse-dynamics proposals require strict_local_observation=True")
    if not bool(torch.isfinite(obs).all()) or not bool(torch.isfinite(plan).all()):
        raise ValueError("inverse-dynamics proposal inputs must be finite")

    sine = obs[..., 0]
    cosine = obs[..., 1]
    handle_body = obs[..., 2:4]
    # Strict observation channels 6:8 store contact velocity divided by two.
    contact_velocity_world = 2.0 * obs[..., 6:8]
    contact_velocity_body = torch.stack(
        (
            cosine * contact_velocity_world[..., 0]
            + sine * contact_velocity_world[..., 1],
            -sine * contact_velocity_world[..., 0]
            + cosine * contact_velocity_world[..., 1],
        ),
        dim=-1,
    )

    desired_velocity_body = (
        plan[..., 3:5] * LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
    )
    desired_omega = plan[..., 5] * LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS

    # The normal matrix is fixed morphology, not a reduction of peer sensing.
    # In the body frame A_i = [[1,0,-r_y],[0,1,r_x]], so the matrix is
    # orientation invariant and every agent can store the same 3x3 inverse.
    fixed_handles = contact_points(cfg, obs.device).to(dtype=obs.dtype)
    fixed_x = fixed_handles[:, 0]
    fixed_y = fixed_handles[:, 1]
    normal = obs.new_tensor(
        (
            (float(cfg.n_agents), 0.0, -float(fixed_y.sum())),
            (0.0, float(cfg.n_agents), float(fixed_x.sum())),
            (
                -float(fixed_y.sum()),
                float(fixed_x.sum()),
                float(fixed_handles.square().sum()),
            ),
        )
    )
    own_normal_rhs = torch.stack(
        (
            contact_velocity_body[..., 0],
            contact_velocity_body[..., 1],
            -handle_body[..., 1] * contact_velocity_body[..., 0]
            + handle_body[..., 0] * contact_velocity_body[..., 1],
        ),
        dim=-1,
    )
    local_current_twist = float(cfg.n_agents) * torch.linalg.solve(
        normal,
        own_normal_rhs.unsqueeze(-1),
    ).squeeze(-1)

    linear_decay = 1.0 - cfg.linear_damping * cfg.dt
    angular_decay = 1.0 - cfg.angular_damping * cfg.dt
    force_body = (
        cfg.mass
        * (desired_velocity_body / linear_decay - local_current_twist[..., :2])
        / cfg.dt
    )
    force = torch.stack(
        (
            cosine * force_body[..., 0] - sine * force_body[..., 1],
            sine * force_body[..., 0] + cosine * force_body[..., 1],
        ),
        dim=-1,
    )
    torque = (
        cfg.inertia
        * (desired_omega / angular_decay - local_current_twist[..., 2])
        / cfg.dt
    )
    scale = obs.new_tensor(
        (
            cfg.n_agents * cfg.max_force,
            cfg.n_agents * cfg.max_force,
            0.75 * cfg.n_agents * cfg.max_force,
        )
    )
    proposals = torch.cat((force, torque.unsqueeze(-1)), dim=-1) / scale
    if proposals.shape != (*obs.shape[:2], 3) or not bool(
        torch.isfinite(proposals).all()
    ):
        raise RuntimeError("inverse-dynamics wrench proposals are malformed")
    return proposals


def local_se2_plan_target(state: Tensor, teacher_label: Any) -> Tensor:
    """Return one normalized payload-frame short-horizon plan.

    Coordinates are the executable teacher's one-step body-frame displacement
    and next twist, followed by cross-track error and ordered path progress.
    This is deliberately a *small next state*, not a distant lookahead pose:
    the force layer is therefore an inverse-dynamics problem rather than an
    ambiguous long-horizon action imitation problem.
    """

    if state.shape != (1, 6):
        raise ValueError("local SE(2) plan target requires state [1,6]")
    next_state = teacher_label.next_state
    semantic8 = teacher_label.semantic8
    if next_state.shape != (6,) or semantic8.shape != (8,):
        raise ValueError("teacher label has malformed next-state or semantic target")
    theta = state[0, 2]
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    delta_world = next_state[:2] - state[0, :2]
    delta_body = torch.stack(
        (
            cosine * delta_world[0] + sine * delta_world[1],
            -sine * delta_world[0] + cosine * delta_world[1],
        )
    )
    normalized_delta = (delta_body / LOCAL_SE2_POSITION_SCALE_M).clamp(-1.0, 1.0)
    normalized_angle = (
        wrap_angle(next_state[2] - state[0, 2]) / LOCAL_SE2_ANGLE_SCALE_RAD
    ).clamp(-1.0, 1.0)

    twist_world = next_state[3:5] / LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
    twist_body = torch.stack(
        (
            cosine * twist_world[0] + sine * twist_world[1],
            -sine * twist_world[0] + cosine * twist_world[1],
        )
    ).clamp(-1.0, 1.0)
    target = torch.cat(
        (
            normalized_delta,
            normalized_angle.view(1),
            twist_body,
            (
                next_state[5:6] / LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS
            ).clamp(-1.0, 1.0),
            semantic8[6:8].clamp(-1.0, 1.0),
        )
    )
    if target.shape != (LOCAL_SE2_PLAN_DIM,) or not bool(torch.isfinite(target).all()):
        raise RuntimeError("local SE(2) plan target is malformed")
    return target


def expand_plan_target_for_agents(target: Tensor, n_agents: int) -> Tensor:
    if target.ndim != 2 or target.shape[-1] != LOCAL_SE2_PLAN_DIM:
        raise ValueError("plan target must have shape [batch,8]")
    if n_agents <= 0:
        raise ValueError("agent count must be positive")
    return target[:, None, :].expand(-1, n_agents, -1)


__all__ = [
    "LOCAL_SE2_PLAN_DIM",
    "LOCAL_SE2_ANGLE_SCALE_RAD",
    "LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS",
    "LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS",
    "LOCAL_SE2_POSITION_SCALE_M",
    "LOCAL_SE2_TERMINAL_WALL_RESERVE_M",
    "expand_plan_target_for_agents",
    "local_se2_inverse_dynamics_wrench_proposals",
    "local_se2_plan_target",
    "local_se2_terminal_attractor_plan",
]
