"""Privileged CEM-lite maneuver teacher for PianoMovers-Force.

The teacher searches short sequences of shared three-dimensional payload
wrenches, never per-agent actions.  Controls live in the active-leg canonical
frame as ``[route force, lateral force, route-oriented torque]``.  Every
candidate is mapped back to world coordinates, allocated through the exact
payload grasp map, advanced with the benchmark rigid dynamics, and audited by
the same SAT collision checker used by evaluation.

This module is training-only.  It does not modify the deployable Sheaf-GBP
policy, its edge stalk, its number of GBP rounds, or its communication payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor

from .config import EnvConfig
from .env import (
    PASSAGE_TARGET_ANGLE,
    PianoBatch,
    active_endpoints,
    body_to_world,
    contact_points,
    leg_progress,
    mission_leg,
    net_wrench_from_forces,
    rotate,
    step_dynamics,
    t_shape_points,
    transition_collision_mask,
    wrap_angle,
)
from .linear_solve import solve_nonsingular
from .teachers import (
    SourceRelativeCorrectionDiagnostics,
    feasible_source_relative_correction,
)


_GRASP_DAMPING = 1e-5
CEM_HORIZON_COST_SEMANTICS = "cem-v1-sat4-frozen-state"


@dataclass(frozen=True)
class CEMTeacherConfig:
    """Small batched shooting configuration suitable for DAgger labels."""

    horizon: int = 24
    knots: int = 6
    knot_repeat: int = 4
    population: int = 16
    elite: int = 3
    iterations: int = 2
    initial_std_force: float = 0.24
    initial_std_torque: float = 0.32
    minimum_std: float = 0.04
    elite_update: float = 0.80
    rotate_signs: tuple[float, ...] = (-1.0, 1.0)
    lateral_offsets: tuple[float, ...] = (-0.08, 0.0, 0.08)
    collision_penalty: float = 100_000.0
    reference_weight: float = 0.65
    progress_weight: float = 0.30
    terminal_progress_weight: float = 30.0
    terminal_reference_weight: float = 8.0
    safety_shaping_weight: float = 16.0
    effort_weight: float = 0.008
    smoothness_weight: float = 0.018

    def validate(self) -> None:
        if self.horizon <= 0 or self.knots <= 0 or self.knot_repeat <= 0:
            raise ValueError("CEM horizon, knots, and knot_repeat must be positive")
        if self.horizon > self.knots * self.knot_repeat:
            raise ValueError("CEM knots do not cover the requested horizon")
        if self.population < 3:
            raise ValueError("CEM population must be at least three")
        if not 1 <= self.elite < self.population:
            raise ValueError("CEM elite count must be in [1, population)")
        if self.iterations <= 0:
            raise ValueError("CEM iterations must be positive")
        if not self.rotate_signs or not self.lateral_offsets:
            raise ValueError("CEM requires at least one maneuver hypothesis")
        if any(value not in (-1.0, 1.0) for value in self.rotate_signs):
            raise ValueError("rotate signs must be -1 or +1")
        if not 0.0 < self.elite_update <= 1.0:
            raise ValueError("elite_update must lie in (0, 1]")


@dataclass(frozen=True)
class CEMTeacherDiagnostics:
    """Per-sample plan and source-relative correction diagnostics."""

    selected_cost: Tensor
    selected_plan_collision: Tensor
    selected_rotate_sign: Tensor
    selected_lateral_offset: Tensor
    selected_canonical_wrench: Tensor
    selected_world_wrench: Tensor
    absolute_wrench_feasibility: Tensor
    correction: SourceRelativeCorrectionDiagnostics
    # These optional fields preserve backwards compatibility with tests and
    # downstream tools that construct only the legacy selected-mode record.
    # ``mode_*`` tensors emitted by ``cem_lite_teacher`` are never optional.
    mode_rotate_sign: Tensor | None = None
    mode_lateral_offset: Tensor | None = None
    mode_best_cost: Tensor | None = None
    mode_best_plan_collision: Tensor | None = None
    mode_best_canonical_wrench: Tensor | None = None
    mode_best_world_wrench: Tensor | None = None
    mode_best_target_forces: Tensor | None = None
    mode_best_normalized_correction: Tensor | None = None
    mode_best_source_relative_target: Tensor | None = None
    mode_best_absolute_wrench_feasibility: Tensor | None = None
    mode_best_correction_alpha: Tensor | None = None
    mode_best_correction_trigger: Tensor | None = None
    mode_best_corrected_first_transition_collision: Tensor | None = None


def active_route_sign(scenario: PianoBatch) -> Tensor:
    """World-x sign of the active leg."""

    start, goal = active_endpoints(scenario)
    sign = torch.sign(goal[:, 0] - start[:, 0])
    return torch.where(sign == 0, torch.ones_like(sign), sign)


def canonical_wrench_to_world(normalized_canonical: Tensor, scenario: PianoBatch) -> Tensor:
    """Reflect canonical normalized wrenches into physical world parity.

    Under the x reflection used by active-leg canonicalization, route force
    and planar torque change sign while world-y lateral force does not.
    """

    if normalized_canonical.shape[-1] != 3:
        raise ValueError("canonical wrench must end in three coordinates")
    sign = active_route_sign(scenario)
    while sign.ndim < normalized_canonical.ndim - 1:
        sign = sign.unsqueeze(-1)
    world = normalized_canonical.clone()
    world[..., 0] *= sign
    world[..., 2] *= sign
    return world


def world_wrench_to_canonical(normalized_world: Tensor, scenario: PianoBatch) -> Tensor:
    """World-to-canonical transform; the reflection is self-inverse."""

    return canonical_wrench_to_world(normalized_world, scenario)


def normalized_wrench_scale(cfg: EnvConfig, reference: Tensor) -> Tensor:
    return reference.new_tensor(
        [
            cfg.n_agents * cfg.max_force,
            cfg.n_agents * cfg.max_force,
            0.75 * cfg.n_agents * cfg.max_force,
        ]
    )


@torch.no_grad()
def feasible_forces_for_absolute_wrench(
    state: Tensor,
    world_wrench: Tensor,
    cfg: EnvConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Allocate and uniformly scale an absolute wrench to handle limits."""

    if state.ndim != 2 or state.shape[-1] != 6:
        raise ValueError("state must have shape [batch, 6]")
    if world_wrench.shape != (state.shape[0], 3):
        raise ValueError("world_wrench must have shape [batch, 3]")
    if world_wrench.device != state.device or world_wrench.dtype != state.dtype:
        raise ValueError("state and world_wrench must share device and dtype")

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
        world_wrench.unsqueeze(-1),
    ).squeeze(-1)
    forces = (grasp.transpose(-1, -2) @ dual.unsqueeze(-1)).squeeze(-1)
    forces = forces.view(state.shape[0], cfg.n_agents, 2)
    maximum = torch.linalg.vector_norm(forces, dim=-1).amax(dim=-1)
    feasibility = (cfg.max_force / maximum.clamp_min(1e-8)).clamp_max(1.0)
    zero_wrench = world_wrench.abs().amax(dim=-1) <= 1e-12
    feasibility = torch.where(zero_wrench, torch.ones_like(feasibility), feasibility)
    forces = forces * feasibility[:, None, None]
    realized_wrench = net_wrench_from_forces(state, forces, cfg)
    return forces, realized_wrench, feasibility


def _repeat_scenario(scenario: PianoBatch, repeats: int) -> PianoBatch:
    def repeated(value: Tensor) -> Tensor:
        return (
            value[:, None]
            .expand(value.shape[0], repeats, *value.shape[1:])
            .reshape(
                value.shape[0] * repeats,
                *value.shape[1:],
            )
        )

    return PianoBatch(
        state0=repeated(scenario.state0),
        goal=repeated(scenario.goal),
        wall_x=repeated(scenario.wall_x),
        gap_y=repeated(scenario.gap_y),
        gap_half=repeated(scenario.gap_half),
        source=repeated(scenario.source),
        leg=repeated(scenario.leg) if scenario.leg is not None else None,
    )


def maneuver_hypotheses(
    cfg: CEMTeacherConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Cartesian product in rotation-major, lateral-minor order."""

    pairs = list(product(cfg.rotate_signs, cfg.lateral_offsets))
    rotate_sign = torch.tensor([pair[0] for pair in pairs], device=device, dtype=dtype)
    lateral_offset = torch.tensor([pair[1] for pair in pairs], device=device, dtype=dtype)
    return rotate_sign, lateral_offset


def hypothesis_target(
    state: Tensor,
    scenario: PianoBatch,
    rotate_sign: Tensor,
    lateral_offset: Tensor,
) -> Tensor:
    """Direction-equivariant passage target for a discrete maneuver mode."""

    if rotate_sign.shape != (state.shape[0],) or lateral_offset.shape != (state.shape[0],):
        raise ValueError("one maneuver hypothesis is required per state")
    active_start, active_goal = active_endpoints(scenario)
    direction = active_route_sign(scenario)
    first_is_left = direction > 0
    first_wall = torch.where(first_is_left, scenario.wall_x[:, 0], scenario.wall_x[:, 1])
    second_wall = torch.where(first_is_left, scenario.wall_x[:, 1], scenario.wall_x[:, 0])
    first_gap = torch.where(first_is_left, scenario.gap_y[:, 0], scenario.gap_y[:, 1])
    second_gap = torch.where(first_is_left, scenario.gap_y[:, 1], scenario.gap_y[:, 0])
    first_half = torch.where(
        first_is_left,
        scenario.gap_half[:, 0],
        scenario.gap_half[:, 1],
    )
    second_half = torch.where(
        first_is_left,
        scenario.gap_half[:, 1],
        scenario.gap_half[:, 0],
    )
    first_offset = lateral_offset.clamp(
        min=-(first_half - 0.50).clamp_min(0.0),
        max=(first_half - 0.50).clamp_min(0.0),
    )
    second_offset = lateral_offset.clamp(
        min=-(second_half - 0.50).clamp_min(0.0),
        max=(second_half - 0.50).clamp_min(0.0),
    )

    route_x = direction * (state[:, 0] - active_start[:, 0])
    first_s = direction * (first_wall - active_start[:, 0])
    second_s = direction * (second_wall - active_start[:, 0])
    theta1 = direction * rotate_sign * PASSAGE_TARGET_ANGLE
    theta2 = -theta1
    target = active_goal.clone()

    before_1 = route_x < first_s - 0.22
    in_1 = (route_x >= first_s - 0.22) & (route_x < first_s + 0.36)
    before_2 = (route_x >= first_s + 0.36) & (route_x < second_s - 0.22)
    in_2 = (route_x >= second_s - 0.22) & (route_x < second_s + 0.36)

    target[:, 0] = torch.where(before_1, first_wall - 0.28 * direction, target[:, 0])
    target[:, 1] = torch.where(before_1, first_gap + first_offset, target[:, 1])
    target[:, 2] = torch.where(before_1, theta1, target[:, 2])
    target[:, 0] = torch.where(in_1, first_wall + 0.38 * direction, target[:, 0])
    target[:, 1] = torch.where(in_1, first_gap + first_offset, target[:, 1])
    target[:, 2] = torch.where(in_1, theta1, target[:, 2])
    target[:, 0] = torch.where(before_2, second_wall - 0.28 * direction, target[:, 0])
    target[:, 1] = torch.where(before_2, second_gap + second_offset, target[:, 1])
    target[:, 2] = torch.where(before_2, theta2, target[:, 2])
    target[:, 0] = torch.where(in_2, second_wall + 0.38 * direction, target[:, 0])
    target[:, 1] = torch.where(in_2, second_gap + second_offset, target[:, 1])
    target[:, 2] = torch.where(in_2, theta2, target[:, 2])
    return target


def _reference_cost(state: Tensor, target: Tensor, cfg: EnvConfig) -> Tensor:
    dx = (state[:, 0] - target[:, 0]) / cfg.world_x
    dy = (state[:, 1] - target[:, 1]) / cfg.world_y
    angle = wrap_angle(state[:, 2] - target[:, 2]) / torch.pi
    return dx.square() + dy.square() + 0.70 * angle.square()


def _smooth_wall_risk(state: Tensor, scenario: PianoBatch, cfg: EnvConfig) -> Tensor:
    """Continuous shaping only; exact SAT remains the collision authority."""

    points = body_to_world(t_shape_points(state.device), state)
    wall_dx = (points[:, :, None, 0] - scenario.wall_x[:, None, :]).abs()
    rel_y = (points[:, :, None, 1] - scenario.gap_y[:, None, :]).abs()
    gap_limit = scenario.gap_half[:, None, :] - 0.10
    gap_violation = (rel_y - gap_limit).clamp_min(0.0)
    near_wall = torch.exp(-((wall_dx / (5.0 * cfg.wall_slab)).square()))
    wall_loss = (near_wall * gap_violation.square()).mean(dim=(1, 2))
    y_loss = (points[:, :, 1].abs() - (cfg.world_y - 0.10)).clamp_min(0.0).square().mean(dim=1)
    x_loss = (points[:, :, 0].abs() - (cfg.world_x - 0.12)).clamp_min(0.0).square().mean(dim=1)
    return wall_loss + y_loss + x_loss


def accumulate_cem_horizon_transition(
    current: Tensor,
    proposed: Tensor,
    normalized_canonical_control: Tensor,
    previous_normalized_canonical_control: Tensor,
    cost: Tensor,
    collided: Tensor,
    scenario: PianoBatch,
    rotate_sign: Tensor,
    lateral_offset: Tensor,
    cfg: EnvConfig,
    cem_cfg: CEMTeacherConfig,
    *,
    tick: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply the canonical CEM per-tick cost and frozen-state semantics.

    Both sampled CEM plans and frozen-source policy continuations call this
    function. Keeping collision penalties and every shaping term in one
    ordered implementation prevents source-vs-mode supervision from silently
    using a different objective than the teacher search.
    """

    hit = transition_collision_mask(current, proposed, scenario, cfg)
    first_hit = hit & (~collided)
    cost += first_hit.to(cost.dtype) * (
        cem_cfg.collision_penalty + 100.0 * float(cem_cfg.horizon - tick)
    )
    collided = collided | hit
    next_state = torch.where(collided[:, None], current, proposed)
    target = hypothesis_target(
        next_state,
        scenario,
        rotate_sign,
        lateral_offset,
    )
    progress = leg_progress(next_state, scenario)
    cost += cem_cfg.reference_weight * _reference_cost(next_state, target, cfg)
    cost += cem_cfg.progress_weight * (1.0 - progress).square()
    cost += cem_cfg.safety_shaping_weight * _smooth_wall_risk(
        next_state,
        scenario,
        cfg,
    )
    cost += cem_cfg.effort_weight * normalized_canonical_control.square().mean(dim=-1)
    cost += cem_cfg.smoothness_weight * (
        normalized_canonical_control - previous_normalized_canonical_control
    ).square().mean(dim=-1)
    return next_state, cost, collided, hit


def finalize_cem_horizon_cost(
    current: Tensor,
    cost: Tensor,
    scenario: PianoBatch,
    rotate_sign: Tensor,
    lateral_offset: Tensor,
    cfg: EnvConfig,
    cem_cfg: CEMTeacherConfig,
) -> Tensor:
    """Apply the canonical terminal CEM progress, reference, and stop cost."""

    terminal_target = hypothesis_target(
        current,
        scenario,
        rotate_sign,
        lateral_offset,
    )
    terminal_progress = leg_progress(current, scenario)
    cost += cem_cfg.terminal_progress_weight * (1.0 - terminal_progress).square()
    cost += cem_cfg.terminal_reference_weight * _reference_cost(
        current,
        terminal_target,
        cfg,
    )
    late = terminal_progress > 0.82
    stop_energy = (current[:, 3:5] / 2.0).square().sum(dim=-1) + (current[:, 5] / 2.0).square()
    cost += late.to(cost.dtype) * 2.0 * stop_energy
    return cost


def _reference_canonical_wrench(
    state: Tensor,
    scenario: PianoBatch,
    rotate_sign: Tensor,
    lateral_offset: Tensor,
    cfg: EnvConfig,
) -> Tensor:
    target = hypothesis_target(state, scenario, rotate_sign, lateral_offset)
    pos_error = target[:, 0:2] - state[:, 0:2]
    angle_error = wrap_angle(target[:, 2] - state[:, 2])
    force_xy = 4.4 * pos_error - 1.35 * state[:, 3:5]
    torque = 3.2 * angle_error - 0.90 * state[:, 5]
    scale = normalized_wrench_scale(cfg, state)
    normalized_world = torch.cat((force_xy, torque.unsqueeze(-1)), dim=-1) / scale
    normalized_world = normalized_world.clamp(-1.0, 1.0)
    return world_wrench_to_canonical(normalized_world, scenario)


@torch.no_grad()
def _nominal_knot_sequence(
    state: Tensor,
    scenario: PianoBatch,
    rotate_sign: Tensor,
    lateral_offset: Tensor,
    cfg: EnvConfig,
    cem_cfg: CEMTeacherConfig,
) -> Tensor:
    """Roll the geometric feedback mean once for each maneuver hypothesis."""

    nominal_state = state.clone()
    controls = []
    scale = normalized_wrench_scale(cfg, state)
    for _ in range(cem_cfg.knots):
        canonical = _reference_canonical_wrench(
            nominal_state,
            scenario,
            rotate_sign,
            lateral_offset,
            cfg,
        )
        controls.append(canonical)
        normalized_world = canonical_wrench_to_world(canonical, scenario)
        forces, _, _ = feasible_forces_for_absolute_wrench(
            nominal_state,
            normalized_world * scale,
            cfg,
        )
        for _ in range(cem_cfg.knot_repeat):
            nominal_state = step_dynamics(nominal_state, forces, cfg)
    return torch.stack(controls, dim=1)


@torch.no_grad()
def evaluate_canonical_control_sequences(
    state: Tensor,
    scenario: PianoBatch,
    normalized_canonical_controls: Tensor,
    rotate_sign: Tensor,
    lateral_offset: Tensor,
    cfg: EnvConfig,
    cem_cfg: CEMTeacherConfig,
) -> tuple[Tensor, Tensor]:
    """Score flattened candidate sequences with rigid dynamics and SAT×4."""

    if cfg.collision_substeps != 4:
        raise ValueError("CEM ceiling claims require collision_substeps=4")
    if normalized_canonical_controls.ndim != 3:
        raise ValueError("controls must have shape [candidate, knot, 3]")
    candidates = normalized_canonical_controls.shape[0]
    if state.shape[0] != candidates or scenario.state0.shape[0] != candidates:
        raise ValueError("one state and scenario row are required per candidate")
    if rotate_sign.shape != (candidates,) or lateral_offset.shape != (candidates,):
        raise ValueError("one maneuver hypothesis is required per candidate")

    current = state.clone()
    cost = torch.zeros(candidates, dtype=state.dtype, device=state.device)
    collided = torch.zeros(candidates, dtype=torch.bool, device=state.device)
    previous_control = normalized_canonical_controls[:, 0]
    scale = normalized_wrench_scale(cfg, state)

    for tick in range(cem_cfg.horizon):
        knot = min(tick // cem_cfg.knot_repeat, cem_cfg.knots - 1)
        canonical = normalized_canonical_controls[:, knot].clamp(-1.0, 1.0)
        normalized_world = canonical_wrench_to_world(canonical, scenario)
        forces, _, _ = feasible_forces_for_absolute_wrench(
            current,
            normalized_world * scale,
            cfg,
        )
        proposed = step_dynamics(current, forces, cfg)
        next_state, cost, collided, _ = accumulate_cem_horizon_transition(
            current,
            proposed,
            canonical,
            previous_control,
            cost,
            collided,
            scenario,
            rotate_sign,
            lateral_offset,
            cfg,
            cem_cfg,
            tick=tick,
        )
        previous_control = canonical
        current = next_state

    cost = finalize_cem_horizon_cost(
        current,
        cost,
        scenario,
        rotate_sign,
        lateral_offset,
        cfg,
        cem_cfg,
    )
    return cost, collided


def _gather_population(values: Tensor, indices: Tensor) -> Tensor:
    """Gather population axis 2 from ``[B,H,P,...]`` tensors."""

    trailing = values.shape[3:]
    index = indices.view(*indices.shape, *([1] * len(trailing))).expand(
        *indices.shape,
        *trailing,
    )
    return torch.gather(values, 2, index)


@torch.no_grad()
def cem_lite_teacher(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    source_forces: Tensor,
    *,
    cem_cfg: CEMTeacherConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, CEMTeacherDiagnostics]:
    """Plan and return one feasible source-relative full-wrench correction."""

    options = cem_cfg or CEMTeacherConfig()
    options.validate()
    if cfg.collision_substeps != 4:
        raise ValueError("CEM ceiling claims require collision_substeps=4")
    if scenario.leg is None or not bool((mission_leg(scenario) == 1).all()):
        raise ValueError("CEM-lite teacher accepts return-leg rows only")
    if state.shape[0] != scenario.state0.shape[0]:
        raise ValueError("state and scenario batch dimensions must match")

    batch = state.shape[0]
    rotate_modes, lateral_modes = maneuver_hypotheses(
        options,
        device=state.device,
        dtype=state.dtype,
    )
    hypotheses = rotate_modes.numel()
    hypothesis_repeats = hypotheses
    hypothesis_scenario = _repeat_scenario(scenario, hypothesis_repeats)
    hypothesis_state = state[:, None].expand(batch, hypotheses, 6).reshape(-1, 6)
    rotate = rotate_modes.view(1, hypotheses).expand(batch, -1).reshape(-1)
    lateral = lateral_modes.view(1, hypotheses).expand(batch, -1).reshape(-1)
    nominal = _nominal_knot_sequence(
        hypothesis_state,
        hypothesis_scenario,
        rotate,
        lateral,
        cfg,
        options,
    ).view(batch, hypotheses, options.knots, 3)
    mean = nominal.clone()
    std = (
        state.new_tensor(
            [
                options.initial_std_force,
                options.initial_std_force,
                options.initial_std_torque,
            ]
        )
        .view(1, 1, 1, 3)
        .expand(batch, hypotheses, options.knots, 3)
        .clone()
    )

    incumbent_sequence: Tensor | None = None
    incumbent_cost: Tensor | None = None
    incumbent_collision: Tensor | None = None
    repeats = hypotheses * options.population
    candidate_scenario = _repeat_scenario(scenario, repeats)
    candidate_state = (
        state[:, None, None]
        .expand(
            batch,
            hypotheses,
            options.population,
            6,
        )
        .reshape(-1, 6)
    )
    candidate_rotate = (
        rotate_modes.view(1, hypotheses, 1)
        .expand(
            batch,
            hypotheses,
            options.population,
        )
        .reshape(-1)
    )
    candidate_lateral = (
        lateral_modes.view(1, hypotheses, 1)
        .expand(
            batch,
            hypotheses,
            options.population,
        )
        .reshape(-1)
    )

    for _ in range(options.iterations):
        noise = torch.randn(
            batch,
            hypotheses,
            options.population,
            options.knots,
            3,
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        samples = (mean[:, :, None] + std[:, :, None] * noise).clamp(-1.0, 1.0)
        # Deterministic anchors make each iteration at least as informative as
        # its current geometric mean and a no-force fallback.
        samples[:, :, 0] = mean
        samples[:, :, 1] = 0.0
        flat_samples = samples.reshape(-1, options.knots, 3)
        flat_cost, flat_collision = evaluate_canonical_control_sequences(
            candidate_state,
            candidate_scenario,
            flat_samples,
            candidate_rotate,
            candidate_lateral,
            cfg,
            options,
        )
        costs = flat_cost.view(batch, hypotheses, options.population)
        collisions = flat_collision.view(batch, hypotheses, options.population)
        elite_indices = costs.topk(options.elite, dim=2, largest=False).indices
        elite_samples = _gather_population(samples, elite_indices)
        elite_mean = elite_samples.mean(dim=2)
        elite_std = elite_samples.std(dim=2, unbiased=False).clamp_min(options.minimum_std)
        update = options.elite_update
        mean = (1.0 - update) * mean + update * elite_mean
        std = (1.0 - update) * std + update * elite_std
        iteration_cost, iteration_population = costs.min(dim=2)
        iteration_sequence = _gather_population(
            samples,
            iteration_population.unsqueeze(-1),
        ).squeeze(2)
        iteration_collision = _gather_population(
            collisions,
            iteration_population.unsqueeze(-1),
        ).squeeze(2)
        if incumbent_sequence is None:
            incumbent_sequence = iteration_sequence
            incumbent_cost = iteration_cost
            incumbent_collision = iteration_collision
        else:
            if incumbent_cost is None or incumbent_collision is None:
                raise RuntimeError("incomplete CEM incumbent state")
            improved = iteration_cost < incumbent_cost
            incumbent_sequence = torch.where(
                improved[:, :, None, None],
                iteration_sequence,
                incumbent_sequence,
            )
            incumbent_cost = torch.where(improved, iteration_cost, incumbent_cost)
            incumbent_collision = torch.where(
                improved,
                iteration_collision,
                incumbent_collision,
            )

    if incumbent_sequence is None or incumbent_cost is None or incumbent_collision is None:
        raise RuntimeError("CEM failed to produce candidates")
    selected_hypothesis = incumbent_cost.argmin(dim=1)
    row = torch.arange(batch, device=state.device)
    selected_sequence = incumbent_sequence[row, selected_hypothesis]
    selected_canonical = selected_sequence[:, 0]
    selected_normalized_world = canonical_wrench_to_world(selected_canonical, scenario)
    scale = normalized_wrench_scale(cfg, state)
    teacher_forces, selected_world_wrench, absolute_feasibility = (
        feasible_forces_for_absolute_wrench(
            state,
            selected_normalized_world * scale,
            cfg,
        )
    )
    corrected_forces, normalized_target, correction = feasible_source_relative_correction(
        state,
        cfg,
        source_forces,
        teacher_forces,
    )

    # Retain the complete discrete mixture in canonical product order.  Each
    # first wrench is independently made handle-feasible and converted to the
    # same source-relative three-channel target used by the deployable return
    # adapter.  This is training-only privileged supervision; it never changes
    # inference communication.
    mode_best_canonical = incumbent_sequence[:, :, 0]
    flat_mode_canonical = mode_best_canonical.reshape(batch * hypotheses, 3)
    flat_mode_normalized_world = canonical_wrench_to_world(
        flat_mode_canonical,
        hypothesis_scenario,
    )
    flat_mode_teacher_forces, flat_mode_world_wrench, flat_mode_feasibility = (
        feasible_forces_for_absolute_wrench(
            hypothesis_state,
            flat_mode_normalized_world * normalized_wrench_scale(cfg, state),
            cfg,
        )
    )
    flat_mode_source_forces = (
        source_forces[:, None]
        .expand(
            batch,
            hypotheses,
            cfg.n_agents,
            2,
        )
        .reshape(batch * hypotheses, cfg.n_agents, 2)
    )
    flat_mode_target_forces, flat_mode_source_relative_target, flat_mode_correction = (
        feasible_source_relative_correction(
            hypothesis_state,
            cfg,
            flat_mode_source_forces,
            flat_mode_teacher_forces,
        )
    )
    flat_mode_corrected_next = step_dynamics(
        hypothesis_state,
        flat_mode_target_forces,
        cfg,
    )
    flat_mode_corrected_collision = transition_collision_mask(
        hypothesis_state,
        flat_mode_corrected_next,
        hypothesis_scenario,
        cfg,
    )
    diagnostics = CEMTeacherDiagnostics(
        selected_cost=incumbent_cost[row, selected_hypothesis],
        selected_plan_collision=incumbent_collision[row, selected_hypothesis],
        selected_rotate_sign=rotate_modes[selected_hypothesis],
        selected_lateral_offset=lateral_modes[selected_hypothesis],
        selected_canonical_wrench=selected_canonical,
        selected_world_wrench=selected_world_wrench,
        absolute_wrench_feasibility=absolute_feasibility,
        correction=correction,
        mode_rotate_sign=rotate_modes,
        mode_lateral_offset=lateral_modes,
        mode_best_cost=incumbent_cost,
        mode_best_plan_collision=incumbent_collision,
        mode_best_canonical_wrench=mode_best_canonical,
        mode_best_world_wrench=flat_mode_world_wrench.view(batch, hypotheses, 3),
        mode_best_target_forces=flat_mode_target_forces.view(
            batch,
            hypotheses,
            cfg.n_agents,
            2,
        ),
        mode_best_normalized_correction=flat_mode_source_relative_target.view(
            batch,
            hypotheses,
            3,
        ),
        mode_best_source_relative_target=flat_mode_source_relative_target.view(
            batch,
            hypotheses,
            3,
        ),
        mode_best_absolute_wrench_feasibility=flat_mode_feasibility.view(
            batch,
            hypotheses,
        ),
        mode_best_correction_alpha=flat_mode_correction.alpha.view(batch, hypotheses),
        mode_best_correction_trigger=flat_mode_correction.trigger.view(
            batch,
            hypotheses,
        ),
        mode_best_corrected_first_transition_collision=(
            flat_mode_corrected_collision.view(batch, hypotheses)
        ),
    )
    return corrected_forces, normalized_target, diagnostics


__all__ = [
    "CEM_HORIZON_COST_SEMANTICS",
    "CEMTeacherConfig",
    "CEMTeacherDiagnostics",
    "accumulate_cem_horizon_transition",
    "active_route_sign",
    "canonical_wrench_to_world",
    "cem_lite_teacher",
    "evaluate_canonical_control_sequences",
    "feasible_forces_for_absolute_wrench",
    "finalize_cem_horizon_cost",
    "hypothesis_target",
    "maneuver_hypotheses",
    "normalized_wrench_scale",
    "world_wrench_to_canonical",
]
