"""Centralized training-only teachers for PianoMovers-Force.

Nothing in this module is a deployable policy.  In particular,
``predictive_cbf_teacher`` reads the complete payload pose and global wall/slit
geometry.  It is an oracle for producing DAgger labels only; using it during
evaluation would leak privileged state and invalidate policy/communication
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import EnvConfig
from .env import (
    PianoBatch,
    active_endpoints,
    body_rectangles_to_world,
    contact_points,
    net_wrench_from_forces,
    rotate,
    t_shape_rectangles,
)
from .linear_solve import solve_nonsingular


PREDICTIVE_CBF_BRAKE_ACCEL = 0.5
PREDICTIVE_CBF_CLEARANCE_BUFFER = 0.0
_COLLISION_GAP_INSET = 0.03
_GRASP_DAMPING = 1e-5
_FEASIBILITY_BISECTION_STEPS = 12
# Shared admission tolerance for already-clamped fp32 handle actions.  Keep it
# public so cached teacher labels and the teacher's own feasibility checks use
# one boundary convention.
FORCE_LIMIT_TOLERANCE = 2e-5
_FORCE_TOLERANCE = FORCE_LIMIT_TOLERANCE


@dataclass(frozen=True)
class PredictiveCBFDiagnostics:
    """Per-sample diagnostics from the centralized predictive teacher."""

    clearance: Tensor
    v_safe: Tensor
    trigger: Tensor
    alpha: Tensor


@dataclass(frozen=True)
class SourceRelativeCorrectionDiagnostics:
    """Diagnostics for a feasible full-wrench teacher correction.

    ``desired_delta_wrench`` is the rigid-body wrench difference between a
    counterfactual teacher action and the carried-memory source action.
    ``actual_delta_wrench`` is the correction that remains after the
    per-handle force-limit line search.  This distinction matters when a
    source handle is already saturated.
    """

    desired_delta_wrench: Tensor
    actual_delta_wrench: Tensor
    trigger: Tensor
    alpha: Tensor


def _swept_convex_clearance(poly: Tensor, rect: Tensor, route_sign: Tensor) -> Tensor:
    """Return exact first translation distance for a moving convex quadrilateral.

    ``poly`` moves by ``distance * (route_sign, 0)`` while ``rect`` remains
    fixed.  The interval of overlap on every separating axis is intersected,
    which is continuous swept SAT rather than time sampling.
    """

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
    stacked_axes = torch.stack(axes, dim=1)

    poly_projection = torch.einsum("bpd,bad->bpa", poly, stacked_axes)
    rect_projection = torch.einsum("bpd,bad->bpa", rect, stacked_axes)
    poly_min = poly_projection.amin(dim=1)
    poly_max = poly_projection.amax(dim=1)
    rect_min = rect_projection.amin(dim=1)
    rect_max = rect_projection.amax(dim=1)
    projection_rate = route_sign[:, None] * stacked_axes[:, :, 0]

    positive = projection_rate > 1e-8
    negative = projection_rate < -1e-8
    stationary = ~(positive | negative)
    negative_inf = torch.full_like(projection_rate, -torch.inf)
    positive_inf = torch.full_like(projection_rate, torch.inf)

    lower = torch.where(
        positive,
        (rect_min - poly_max) / projection_rate.clamp_min(1e-8),
        negative_inf,
    )
    upper = torch.where(
        positive,
        (rect_max - poly_min) / projection_rate.clamp_min(1e-8),
        positive_inf,
    )
    lower = torch.where(
        negative,
        (rect_max - poly_min) / projection_rate.clamp_max(-1e-8),
        lower,
    )
    upper = torch.where(
        negative,
        (rect_min - poly_max) / projection_rate.clamp_max(-1e-8),
        upper,
    )

    stationary_separation = stationary & ((poly_max < rect_min) | (rect_max < poly_min))
    lower = torch.where(stationary_separation, positive_inf, lower)
    upper = torch.where(stationary_separation, negative_inf, upper)
    enter = lower.amax(dim=-1)
    leave = upper.amin(dim=-1)
    valid = (enter <= leave) & (leave >= 0.0)
    return torch.where(
        valid,
        enter.clamp_min(0.0),
        torch.full_like(enter, torch.inf),
    )


def exact_t_wall_forward_clearance(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
) -> Tensor:
    """Return the exact straight-line clearance to the first wall collision.

    The T payload is the same union of two rotated rectangles used by the
    simulator.  Each rectangle is swept continuously in the active-leg route
    direction against the solid regions above and below both slits.  A result
    of ``+inf`` means the current ``(y, theta)`` pose can translate forward
    through all walls without collision; it does not certify arbitrary future
    rotations or lateral motion.

    This function consumes global geometry and is therefore training-only.
    """

    _validate_state_and_scenario(state, scenario)
    active_start, active_goal = active_endpoints(scenario)
    route_sign = torch.sign(active_goal[:, 0] - active_start[:, 0])
    route_sign = torch.where(route_sign == 0, torch.ones_like(route_sign), route_sign)
    body_rectangles = body_rectangles_to_world(t_shape_rectangles(state.device), state)
    best = torch.full(
        (state.shape[0],),
        torch.inf,
        dtype=state.dtype,
        device=state.device,
    )

    for body_id in range(body_rectangles.shape[1]):
        body = body_rectangles[:, body_id]
        for wall_id in range(scenario.wall_x.shape[1]):
            wall_x = scenario.wall_x[:, wall_id]
            gap_y = scenario.gap_y[:, wall_id]
            gap_half = (scenario.gap_half[:, wall_id] - _COLLISION_GAP_INSET).clamp_min(0.0)
            x0 = wall_x - cfg.wall_slab
            x1 = wall_x + cfg.wall_slab
            lower_y1 = gap_y - gap_half
            upper_y0 = gap_y + gap_half
            lower = torch.stack(
                (
                    torch.stack((x0, torch.full_like(x0, -cfg.world_y)), dim=-1),
                    torch.stack((x1, torch.full_like(x0, -cfg.world_y)), dim=-1),
                    torch.stack((x1, lower_y1), dim=-1),
                    torch.stack((x0, lower_y1), dim=-1),
                ),
                dim=1,
            )
            upper = torch.stack(
                (
                    torch.stack((x0, upper_y0), dim=-1),
                    torch.stack((x1, upper_y0), dim=-1),
                    torch.stack((x1, torch.full_like(x0, cfg.world_y)), dim=-1),
                    torch.stack((x0, torch.full_like(x0, cfg.world_y)), dim=-1),
                ),
                dim=1,
            )
            no_lower_wall = lower_y1 <= -cfg.world_y
            no_upper_wall = upper_y0 >= cfg.world_y
            lower_clearance = _swept_convex_clearance(body, lower, route_sign)
            upper_clearance = _swept_convex_clearance(body, upper, route_sign)
            lower_clearance = torch.where(
                no_lower_wall,
                torch.full_like(lower_clearance, torch.inf),
                lower_clearance,
            )
            upper_clearance = torch.where(
                no_upper_wall,
                torch.full_like(upper_clearance, torch.inf),
                upper_clearance,
            )
            best = torch.minimum(
                best,
                torch.minimum(lower_clearance, upper_clearance),
            )
    return best


def _validate_state_and_scenario(state: Tensor, scenario: PianoBatch) -> None:
    if state.ndim != 2 or state.shape[-1] != 6:
        raise ValueError("state must have shape [batch, 6]")
    batch = state.shape[0]
    tensors = (
        scenario.state0,
        scenario.source,
        scenario.goal,
        scenario.wall_x,
        scenario.gap_y,
        scenario.gap_half,
    )
    if any(value.shape[0] != batch for value in tensors):
        raise ValueError("state and scenario batch dimensions must match")
    if any(value.device != state.device for value in tensors):
        raise ValueError("state and scenario tensors must share one device")
    if scenario.leg is not None:
        if scenario.leg.shape[0] != batch:
            raise ValueError("scenario leg batch dimension must match state")
        if scenario.leg.device != state.device:
            raise ValueError("scenario leg must share the state device")


def _minimum_norm_force_for_wrench(state: Tensor, wrench: Tensor, cfg: EnvConfig) -> Tensor:
    contacts = contact_points(cfg, state.device)
    r_world = rotate(contacts, state[:, 2])
    grasp = torch.zeros(
        state.shape[0],
        3,
        2 * cfg.n_agents,
        dtype=state.dtype,
        device=state.device,
    )
    grasp[:, 0, 0::2] = 1.0
    grasp[:, 1, 1::2] = 1.0
    grasp[:, 2, 0::2] = -r_world[:, :, 1]
    grasp[:, 2, 1::2] = r_world[:, :, 0]
    gram = grasp @ grasp.transpose(-1, -2)
    eye = torch.eye(3, dtype=state.dtype, device=state.device).view(1, 3, 3)
    dual = solve_nonsingular(
        gram + _GRASP_DAMPING * eye,
        wrench.unsqueeze(-1),
    ).squeeze(-1)
    return (
        (grasp.transpose(-1, -2) @ dual.unsqueeze(-1))
        .squeeze(-1)
        .view(state.shape[0], cfg.n_agents, 2)
    )


def _apply_feasible_route_correction(
    raw_forces: Tensor,
    state: Tensor,
    desired_delta_wrench: Tensor,
    cfg: EnvConfig,
) -> tuple[Tensor, Tensor]:
    correction = _minimum_norm_force_for_wrench(state, desired_delta_wrench, cfg)
    lower = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
    upper = torch.ones_like(lower)
    for _ in range(_FEASIBILITY_BISECTION_STEPS):
        midpoint = 0.5 * (lower + upper)
        candidate = raw_forces + midpoint[:, None, None] * correction
        feasible = (
            torch.linalg.vector_norm(candidate, dim=-1).amax(dim=-1)
            <= cfg.max_force + _FORCE_TOLERANCE
        )
        lower = torch.where(feasible, midpoint, lower)
        upper = torch.where(feasible, upper, midpoint)
    no_correction = desired_delta_wrench.abs().amax(dim=-1) <= 1e-12
    alpha = torch.where(no_correction, torch.ones_like(lower), lower)
    corrected = raw_forces + alpha[:, None, None] * correction
    return corrected, alpha


@torch.no_grad()
def feasible_source_relative_correction(
    state: Tensor,
    cfg: EnvConfig,
    source_forces: Tensor,
    teacher_forces: Tensor,
    *,
    check_force_limits: bool = True,
) -> tuple[Tensor, Tensor, SourceRelativeCorrectionDiagnostics]:
    """Project a counterfactual teacher action into a feasible 3-D residual.

    The deployable policy already produces ``source_forces``.  A privileged
    training teacher proposes ``teacher_forces``.  Only their rigid-body net
    wrench difference is retained, because that is exactly the three-channel
    quantity predicted by the return adapter.  The minimum-norm grasp
    correction and scalar feasibility search preserve every handle's
    ``max_force`` constraint.

    The returned normalized target uses the policy convention
    ``[N Fmax, N Fmax, .75 N Fmax]`` for ``[Fx, Fy, tau]``.  It is a complete
    full-wrench target: unlike the predictive CBF target, its lateral and
    torque coordinates are not forced to zero.
    """

    if state.ndim != 2 or state.shape[-1] != 6:
        raise ValueError("state must have shape [batch, 6]")
    expected_shape = (state.shape[0], cfg.n_agents, 2)
    for name, forces in (
        ("source_forces", source_forces),
        ("teacher_forces", teacher_forces),
    ):
        if forces.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if forces.device != state.device:
            raise ValueError(f"{name} and state must share one device")
        if forces.dtype != state.dtype:
            raise ValueError(f"{name} and state must share one dtype")
        if check_force_limits and bool(
            (
                torch.linalg.vector_norm(forces, dim=-1).amax(dim=-1)
                > cfg.max_force + _FORCE_TOLERANCE
            ).any()
        ):
            raise ValueError(f"{name} must already satisfy cfg.max_force")

    source_wrench = net_wrench_from_forces(state, source_forces, cfg)
    teacher_wrench = net_wrench_from_forces(state, teacher_forces, cfg)
    desired_delta_wrench = teacher_wrench - source_wrench
    corrected_forces, alpha = _apply_feasible_route_correction(
        source_forces,
        state,
        desired_delta_wrench,
        cfg,
    )
    corrected_wrench = net_wrench_from_forces(state, corrected_forces, cfg)
    actual_delta_wrench = corrected_wrench - source_wrench
    wrench_scale = state.new_tensor(
        [
            cfg.n_agents * cfg.max_force,
            cfg.n_agents * cfg.max_force,
            0.75 * cfg.n_agents * cfg.max_force,
        ]
    )
    normalized_target = actual_delta_wrench / wrench_scale
    trigger = desired_delta_wrench.abs().amax(dim=-1) > 1e-6
    diagnostics = SourceRelativeCorrectionDiagnostics(
        desired_delta_wrench=desired_delta_wrench,
        actual_delta_wrench=actual_delta_wrench,
        trigger=trigger,
        alpha=alpha,
    )
    return corrected_forces, normalized_target, diagnostics


@torch.no_grad()
def predictive_cbf_teacher(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    raw_forces: Tensor,
) -> tuple[Tensor, Tensor, PredictiveCBFDiagnostics]:
    """Generate centralized predictive-braking labels for DAgger training.

    This oracle must never run inside a deployed policy or evaluation rollout.
    It uses the complete payload state and global wall/slit map to compute an
    exact swept-SAT collision clearance.  With the empirically validated
    constants ``a=0.5 m/s^2`` and ``buffer=0``, it applies the one-step CBF

    ``v_safe = sqrt(2 a max(clearance - v_+ dt, 0))``

    ``F_route_cap = mass/dt * (v_safe/damping - v_route)``.

    Only the active-route world-x wrench is reduced.  The source policy's
    world ``Fy`` and torque are preserved to numerical precision, and a
    feasibility line search keeps every handle force within ``max_force``.

    Returns:
        corrected_forces: Per-handle teacher actions ``[B, N, 2]``.
        normalized_residual_wrench: Shared DAgger target ``[B, 3]`` normalized
            by ``[N*max_force, N*max_force, .75*N*max_force]``.  Its ``Fy`` and
            torque coordinates are exactly zero by construction.
        diagnostics: Clearance, safe route speed, CBF trigger, and the feasible
            fraction ``alpha`` of the requested pure-route correction.
    """

    _validate_state_and_scenario(state, scenario)
    expected_shape = (state.shape[0], cfg.n_agents, 2)
    if raw_forces.shape != expected_shape:
        raise ValueError(f"raw_forces must have shape {expected_shape}")
    if raw_forces.device != state.device:
        raise ValueError("raw_forces and state must share one device")
    if raw_forces.dtype != state.dtype:
        raise ValueError("raw_forces and state must share one dtype")
    if bool(
        (
            torch.linalg.vector_norm(raw_forces, dim=-1).amax(dim=-1)
            > cfg.max_force + _FORCE_TOLERANCE
        ).any()
    ):
        raise ValueError("raw_forces must already satisfy cfg.max_force")

    raw_wrench = net_wrench_from_forces(state, raw_forces, cfg)
    active_start, active_goal = active_endpoints(scenario)
    route_sign = torch.sign(active_goal[:, 0] - active_start[:, 0])
    route_sign = torch.where(route_sign == 0, torch.ones_like(route_sign), route_sign)
    route_speed = route_sign * state[:, 3]
    raw_route_force = route_sign * raw_wrench[:, 0]
    clearance = exact_t_wall_forward_clearance(state, scenario, cfg)

    effective_clearance = (clearance - PREDICTIVE_CBF_CLEARANCE_BUFFER).clamp_min(0.0)
    remaining_after_step = (effective_clearance - route_speed.clamp_min(0.0) * cfg.dt).clamp_min(
        0.0
    )
    v_safe = torch.sqrt(2.0 * PREDICTIVE_CBF_BRAKE_ACCEL * remaining_after_step)
    damping = 1.0 - cfg.linear_damping * cfg.dt
    route_force_cap = cfg.mass / cfg.dt * (v_safe / damping - route_speed)
    finite_clearance = torch.isfinite(clearance)
    corrected_route_force = torch.where(
        finite_clearance,
        torch.minimum(raw_route_force, route_force_cap),
        raw_route_force,
    )
    requested_delta_route = corrected_route_force - raw_route_force
    requested_delta_wrench = torch.zeros_like(raw_wrench)
    requested_delta_wrench[:, 0] = route_sign * requested_delta_route
    corrected_forces, alpha = _apply_feasible_route_correction(
        raw_forces,
        state,
        requested_delta_wrench,
        cfg,
    )

    corrected_wrench = net_wrench_from_forces(state, corrected_forces, cfg)
    actual_delta_wrench = corrected_wrench - raw_wrench
    wrench_scale = state.new_tensor(
        [
            cfg.n_agents * cfg.max_force,
            cfg.n_agents * cfg.max_force,
            0.75 * cfg.n_agents * cfg.max_force,
        ]
    )
    normalized_target = torch.zeros_like(actual_delta_wrench)
    normalized_target[:, 0] = actual_delta_wrench[:, 0] / wrench_scale[0]
    trigger = finite_clearance & (requested_delta_route < -1e-6)
    diagnostics = PredictiveCBFDiagnostics(
        clearance=clearance,
        v_safe=v_safe,
        trigger=trigger,
        alpha=alpha,
    )
    return corrected_forces, normalized_target, diagnostics


__all__ = [
    "FORCE_LIMIT_TOLERANCE",
    "PREDICTIVE_CBF_BRAKE_ACCEL",
    "PREDICTIVE_CBF_CLEARANCE_BUFFER",
    "PredictiveCBFDiagnostics",
    "SourceRelativeCorrectionDiagnostics",
    "exact_t_wall_forward_clearance",
    "feasible_source_relative_correction",
    "predictive_cbf_teacher",
]
