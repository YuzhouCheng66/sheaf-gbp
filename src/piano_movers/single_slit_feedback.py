"""Fail-closed contracts for the 0.720 m single-slit feedback task.

The existing serpentine code contains useful mechanics (the exact compound-T
SAT geometry, the ordered-path force teacher, recurrent Sheaf-GBP, and the
semantic section), but its map, horizon, and perturbation contracts do not
describe this task.  This module defines the smaller experiment without
pretending that a collision-certified *kinematic* path is an executable force
teacher trajectory.

The intended preparation order is:

1. load a continuously collision-certified SE(2) reference path;
2. execute ``planner_path_teacher_output`` through the real bounded-force
   dynamics from the exact source;
3. require success and recursively certify the resulting dynamics trajectory;
4. only then construct nominal, clearance-relative tube, near-jam, and
   model-induced DAgger sequences.

The simulator has terminal collision and no unilateral contact/friction
dynamics.  Consequently ``near_jam`` below always means a strictly
positive-clearance state.  It must never be reported as post-contact recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .config import (
    ArmConfig,
    EnvConfig,
    HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2,
    HANDLE_LAYOUT_GLYPH_BOUNDARY_V4,
    HANDLE_LAYOUT_PARAMETRIC_T_V3,
    STRICT_OBSERVATION_SCHEMA_LOCAL_CURRENT_SLIT_V2,
)
from .env import (
    PianoBatch,
    collision_clearance_margin,
    observations,
    radius_edges,
    step_dynamics,
    success_mask,
    transition_clearance_margin,
    transition_collision_mask,
)
from .models import BasePolicy, make_policy
from .planner_path_teacher import (
    PlannerPathTeacherConfig,
    planner_path_teacher_output,
)
from .planner_scout_protocol import BacktrackingGateConfig, audit_swept_pose_path
from .serpentine_bc import _planner_semantic_section


SINGLE_SLIT_FEEDBACK_SCHEMA = "piano-movers-single-slit-feedback-v1"
SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA = "single-slit-strict-local18-v1"
SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA = (
    "single-slit-strict-local18-exact-clearance4-v1"
)
SINGLE_SLIT_OBSERVATION_SCHEMAS = (
    SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA,
    SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA,
)
SINGLE_SLIT_BASE_OBSERVATION_DIM = 18
SINGLE_SLIT_FEEDBACK_FEATURE_DIM = 4
SINGLE_SLIT_FEEDBACK_OBSERVATION_DIM = (
    SINGLE_SLIT_BASE_OBSERVATION_DIM + SINGLE_SLIT_FEEDBACK_FEATURE_DIM
)
# One unit of clearance is one millimetre.  The certified nominal bottleneck
# is only 0.818 mm, so metre-normalized channels lose the task-relevant scale.
SINGLE_SLIT_CLEARANCE_SCALE_M = 1.0e-3
SINGLE_SLIT_CLEARANCE_FD_Y_M = 1.0e-4
SINGLE_SLIT_CLEARANCE_FD_THETA_RAD = 1.0e-4
SINGLE_SLIT_FEEDBACK_FEATURE_CLIP = 32.0
VISIBLE_APERTURE_M = 0.720
WALL_SLAB_HALF_THICKNESS_M = 0.060
COLLISION_RESERVE_EACH_SIDE_M = 0.030
SOURCE_POSE = (-1.05, 0.0, 0.0)
GOAL_POSE = (1.05, 0.0, 0.0)
DUMMY_WALL_X_M = 8.0
DEFAULT_ROLLOUT_HORIZON = 256
DEFAULT_REQUIRED_CONTINUOUS_CLEARANCE_M = 4.0e-4


@dataclass(frozen=True)
class SingleSlitDatasetProtocol:
    """Small two-stage data curriculum for one fixed feedback skill.

    Fractions describe rows, not individual ticks.  Every admitted row is a
    contiguous recurrent sequence with six real context ticks followed by
    sixteen supervised ticks.  Phase B begins only after a Phase-A policy can
    complete the nominal source trajectory without collision.
    """

    burn_in_steps: int = 6
    supervised_steps: int = 16
    phase_a_nominal_fraction: float = 0.35
    phase_a_safe_tube_fraction: float = 0.40
    phase_a_near_jam_fraction: float = 0.25
    phase_a_dagger_fraction: float = 0.0
    phase_b_nominal_fraction: float = 0.25
    phase_b_safe_tube_fraction: float = 0.30
    phase_b_near_jam_fraction: float = 0.25
    phase_b_dagger_fraction: float = 0.20
    # Position/yaw perturbations must scale down with the nominal exact-SAT
    # clearance.  The 0.720 m aperture has only millimetre-scale geometric
    # slack, so the historical 4--100 mm tube ladders are invalid here.
    safe_tube_clearance_fraction: float = 0.20
    safe_tube_position_cap_m: float = 2.5e-4
    safe_tube_angle_cap_rad: float = 5.0e-4
    safe_tube_linear_velocity_cap_mps: float = 5.0e-3
    safe_tube_angular_velocity_cap_radps: float = 8.0e-3
    near_jam_clearance_lower_m: float = 1.0e-4
    near_jam_clearance_upper_m: float = 4.0e-4
    dagger_query_cadence: int = 1
    dagger_rollout_horizon: int = DEFAULT_ROLLOUT_HORIZON

    @property
    def sequence_length(self) -> int:
        return self.burn_in_steps + self.supervised_steps

    def validate(self) -> None:
        if self.burn_in_steps <= 0 or self.supervised_steps < 2:
            raise ValueError("single-slit recurrent windows require context and >=2 labels")
        for name, values in (
            (
                "phase_a",
                (
                    self.phase_a_nominal_fraction,
                    self.phase_a_safe_tube_fraction,
                    self.phase_a_near_jam_fraction,
                    self.phase_a_dagger_fraction,
                ),
            ),
            (
                "phase_b",
                (
                    self.phase_b_nominal_fraction,
                    self.phase_b_safe_tube_fraction,
                    self.phase_b_near_jam_fraction,
                    self.phase_b_dagger_fraction,
                ),
            ),
        ):
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"{name} mixture fractions must be finite and non-negative")
            if not math.isclose(sum(values), 1.0, abs_tol=1.0e-8):
                raise ValueError(f"{name} mixture fractions must sum to one")
        if self.phase_a_dagger_fraction != 0.0:
            raise ValueError("Phase A cannot contain model-induced DAgger rows")
        if not 0.0 < self.safe_tube_clearance_fraction < 1.0:
            raise ValueError("safe-tube clearance fraction must lie in (0,1)")
        positive = (
            self.safe_tube_position_cap_m,
            self.safe_tube_angle_cap_rad,
            self.safe_tube_linear_velocity_cap_mps,
            self.safe_tube_angular_velocity_cap_radps,
            self.near_jam_clearance_lower_m,
            self.near_jam_clearance_upper_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("single-slit tube/jam scales must be finite and positive")
        if self.near_jam_clearance_lower_m >= self.near_jam_clearance_upper_m:
            raise ValueError("near-jam clearance band must be increasing")
        if self.dagger_query_cadence <= 0 or self.dagger_rollout_horizon <= 0:
            raise ValueError("DAgger cadence and horizon must be positive")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SINGLE_SLIT_FEEDBACK_SCHEMA,
            "sequence": {
                "burn_in_steps": self.burn_in_steps,
                "supervised_steps": self.supervised_steps,
                "sequence_length": self.sequence_length,
                "context": "real contiguous dynamics; hidden update valid-mask gated",
                "supervision": "every valid post-burn-in tick has force2-per-handle + semantic8",
            },
            "phase_a_mix": {
                "nominal": self.phase_a_nominal_fraction,
                "safe_tube": self.phase_a_safe_tube_fraction,
                "near_jam": self.phase_a_near_jam_fraction,
                "model_induced_dagger": self.phase_a_dagger_fraction,
            },
            "phase_b_mix": {
                "nominal": self.phase_b_nominal_fraction,
                "safe_tube": self.phase_b_safe_tube_fraction,
                "near_jam": self.phase_b_near_jam_fraction,
                "model_induced_dagger": self.phase_b_dagger_fraction,
            },
            "safe_tube": {
                "clearance_relative_scale": self.safe_tube_clearance_fraction,
                "position_cap_m": self.safe_tube_position_cap_m,
                "angle_cap_rad": self.safe_tube_angle_cap_rad,
                "linear_velocity_cap_mps": self.safe_tube_linear_velocity_cap_mps,
                "angular_velocity_cap_radps": self.safe_tube_angular_velocity_cap_radps,
                "admission": "positive pose clearance and complete teacher continuation safe",
            },
            "near_jam": {
                "strictly_positive_clearance_band_m": [
                    self.near_jam_clearance_lower_m,
                    self.near_jam_clearance_upper_m,
                ],
                "post_contact": False,
                "contact_dynamics_present": False,
                "admission": "exact-SAT positive pose plus complete safe teacher continuation",
            },
            "dagger": {
                "starts_after_nominal_closed_loop_success": True,
                "fixed_query_cadence": self.dagger_query_cadence,
                "rollout_horizon": self.dagger_rollout_horizon,
                "causal_order": (
                    "policy action frozen -> teacher labels visited pre-action state -> "
                    "same frozen policy action advances dynamics"
                ),
                "collision_transition_stored": False,
                "teacher_action_executed_during_collection": False,
            },
            "maneuver_branch": {
                "canonical_rotation_sign_only": True,
                "reason": (
                    "single Gaussian GBP must not average the symmetric +/- rotation modes"
                ),
            },
        }


@dataclass(frozen=True)
class NominalTeacherRollout:
    """One dynamics-executed nominal source trajectory and its labels."""

    state: Tensor
    next_state: Tensor
    target_forces: Tensor
    target_section: Tensor
    minimum_path_index: Tensor
    transition_clearance: Tensor
    outcome: str
    finish_step: int
    recursive_audit: Mapping[str, Any] | None

    @property
    def passed(self) -> bool:
        return (
            self.outcome == "success"
            and self.recursive_audit is not None
            and self.recursive_audit.get("passed") is True
        )

    def to_record(self) -> dict[str, Any]:
        minimum = (
            None
            if self.transition_clearance.numel() == 0
            else float(self.transition_clearance.amin().detach().cpu())
        )
        return {
            "outcome": self.outcome,
            "finish_step": self.finish_step,
            "passed": self.passed,
            "valid_transition_count": int(self.state.shape[0]),
            "minimum_satx4_transition_clearance_m": minimum,
            "recursive_continuous_exact_sat_audit": (
                None if self.recursive_audit is None else dict(self.recursive_audit)
            ),
        }


def single_slit_env_config(
    *,
    device: torch.device | str = "cpu",
    max_steps: int = DEFAULT_ROLLOUT_HORIZON,
) -> EnvConfig:
    """Return the immutable deployment/training environment for this skill."""

    if max_steps <= 0:
        raise ValueError("single-slit max_steps must be positive")
    return EnvConfig(
        n_agents=12,
        handle_layout=HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2,
        payload_shape="T",
        world_x=3.9,
        world_y=1.6,
        wall_slab=WALL_SLAB_HALF_THICKNESS_M,
        collision_substeps=4,
        comm_radius=0.40,
        source_x=SOURCE_POSE[0],
        goal_x=GOAL_POSE[0],
        bidirectional=False,
        round_trip=False,
        local_observation=True,
        strict_local_observation=True,
        strict_observation_schema=STRICT_OBSERVATION_SCHEMA_LOCAL_CURRENT_SLIT_V2,
        generic_obstacle_slots=0,
        max_steps=max_steps,
        device=str(torch.device(device)),
    )


def single_slit_scenario(
    batch_size: int,
    cfg: EnvConfig,
    *,
    initial_state: Tensor | None = None,
    visible_aperture_m: float | Tensor | None = None,
) -> PianoBatch:
    """Materialize one wall plus an always-open dummy slot for the legacy API."""

    if batch_size <= 0:
        raise ValueError("single-slit batch_size must be positive")
    device = torch.device(cfg.device)
    source = torch.tensor(SOURCE_POSE, dtype=torch.float32, device=device).expand(
        batch_size, -1
    )
    goal = torch.tensor(
        (cfg.goal_x, cfg.goal_y, cfg.goal_theta),
        dtype=torch.float32,
        device=device,
    ).expand(batch_size, -1)
    if initial_state is None:
        state0 = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
        state0[:, :3] = source
    else:
        if initial_state.shape != (batch_size, 6):
            raise ValueError("initial_state must have shape [batch,6]")
        state0 = initial_state.to(device=device, dtype=torch.float32).clone()
    wall_x = state0.new_tensor((0.0, DUMMY_WALL_X_M)).expand(batch_size, -1)
    gap_y = state0.new_zeros((batch_size, 2))
    aperture = VISIBLE_APERTURE_M if visible_aperture_m is None else visible_aperture_m
    if isinstance(aperture, Tensor):
        aperture_tensor = aperture.to(device=device, dtype=state0.dtype)
        if aperture_tensor.ndim == 0:
            aperture_tensor = aperture_tensor.expand(batch_size)
        if aperture_tensor.shape != (batch_size,):
            raise ValueError("visible_aperture_m tensor must be scalar or [batch]")
    else:
        aperture_value = float(aperture)
        if not math.isfinite(aperture_value) or aperture_value <= 0.0:
            raise ValueError("visible aperture must be positive and finite")
        aperture_tensor = state0.new_full((batch_size,), aperture_value)
    if not bool(torch.isfinite(aperture_tensor).all()) or bool(
        (aperture_tensor <= 0.0).any()
    ):
        raise ValueError("visible apertures must be positive and finite")
    gap_half = torch.stack(
        (aperture_tensor / 2.0, state0.new_full((batch_size,), 10.0)),
        dim=-1,
    )
    return PianoBatch(
        state0=state0,
        source=source.clone(),
        goal=goal.clone(),
        wall_x=wall_x,
        gap_y=gap_y,
        gap_half=gap_half,
        leg=None,
        obstacle_rects=None,
    )


def _visible_wall_candidate_clearance(
    state: Tensor,
    wall_x: Tensor,
    gap_y: Tensor,
    gap_half: Tensor,
    cfg: EnvConfig,
) -> Tensor:
    """Exact clearance for flattened locally reconstructed wall candidates."""

    if state.ndim != 2 or state.shape[-1] != 6:
        raise ValueError("candidate state must have shape [rows,6]")
    rows = state.shape[0]
    if any(value.shape != (rows,) for value in (wall_x, gap_y, gap_half)):
        raise ValueError("candidate wall geometry must have shape [rows]")
    pose = state[:, :3]
    local_scenario = PianoBatch(
        state0=state,
        source=pose,
        goal=pose,
        wall_x=wall_x[:, None],
        gap_y=gap_y[:, None],
        gap_half=gap_half[:, None],
        leg=None,
        obstacle_rects=None,
    )
    return collision_clearance_margin(state, local_scenario, cfg)


def _single_slit_feedback_features_from_base(
    base: Tensor,
    cfg: EnvConfig,
) -> Tensor:
    """Derive exact-clearance feedback using only one agent's local-18 row.

    The reconstruction does not inspect ``state`` or ``scenario``.  Channels
    0:2 give payload orientation, 2:4 the fixed body-frame contact, 4:6 the
    world contact, and 10:18 the two gated wall records.  Hence a visible
    agent can reconstruct payload centre plus the wall/gap geometry locally.

    Returned channels are:

    0. signed exact-SAT clearance in millimetres;
    1. central finite-difference ``dc/dy`` (equivalently mm per +1 mm);
    2. central finite-difference ``dc/dtheta`` (equivalently mm per +1 mrad);
    3. exact one-tick clearance change in mm if the complete payload were
       translated at this contact's measured velocity with theta held fixed.

    Channel 3 is deliberately a local directional projection, not the true
    rigid-body clearance rate: one contact velocity cannot uniquely recover
    centre velocity and angular velocity.  All four channels are exactly zero
    unless at least one of this agent's wall records is visible.
    """

    if base.ndim != 3 or base.shape[-1] != SINGLE_SLIT_BASE_OBSERVATION_DIM:
        raise ValueError("single-slit base observation must have shape [B,N,18]")
    if base.shape[1] != cfg.n_agents:
        raise ValueError("single-slit base observation agent count mismatch")
    legacy_fixed_t = (
        cfg.handle_layout == HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2
        and cfg.n_agents == 12
        and cfg.payload_shape == "T"
    )
    parametric_t = (
        cfg.handle_layout == HANDLE_LAYOUT_PARAMETRIC_T_V3
        and 8 <= cfg.n_agents <= 20
        and cfg.payload_shape == "T"
    )
    glyph = (
        cfg.handle_layout == HANDLE_LAYOUT_GLYPH_BOUNDARY_V4
        and 8 <= cfg.n_agents <= 20
    )
    if not (legacy_fixed_t or parametric_t or glyph):
        raise ValueError(
            "exact-clearance4 requires canonical boundary-attached 12-T or "
            "parametric-t-v3/glyph-boundary-v4 with 8--20 agents"
        )
    if not cfg.strict_local_observation or cfg.generic_obstacle_slots != 0:
        raise ValueError("exact-clearance4 requires the unextended strict-local18 schema")

    batch, n_agents, _ = base.shape
    wall_slots = 2
    sin_theta = base[..., 0]
    cos_theta = base[..., 1]
    theta = torch.atan2(sin_theta, cos_theta)
    contact_body = base[..., 2:4]
    contact_world = torch.stack(
        (base[..., 4] * 2.6, base[..., 5] * cfg.world_y),
        dim=-1,
    )
    rotated_contact = torch.stack(
        (
            cos_theta * contact_body[..., 0] - sin_theta * contact_body[..., 1],
            sin_theta * contact_body[..., 0] + cos_theta * contact_body[..., 1],
        ),
        dim=-1,
    )
    centre = contact_world - rotated_contact

    visibility = base[..., 16:18] > 0.5
    wall_x = (
        contact_world[..., 0:1]
        + base[..., 10:12] * float(cfg.wall_sensor_range)
    )
    gap_y = contact_world[..., 1:2] + base[..., 12:14] * float(cfg.world_y)
    gap_half = base[..., 14:16] * float(cfg.world_y)

    candidate_state = base.new_zeros((batch, n_agents, wall_slots, 6))
    candidate_state[..., 0:2] = centre.unsqueeze(-2)
    candidate_state[..., 2] = theta.unsqueeze(-1)
    flat_state = candidate_state.reshape(-1, 6)
    flat_wall_x = wall_x.reshape(-1)
    flat_gap_y = gap_y.reshape(-1)
    flat_gap_half = gap_half.reshape(-1)

    def clearance_at(offset_y: float = 0.0, offset_theta: float = 0.0) -> Tensor:
        query = flat_state.clone()
        query[:, 1] += offset_y
        query[:, 2] += offset_theta
        return _visible_wall_candidate_clearance(
            query,
            flat_wall_x,
            flat_gap_y,
            flat_gap_half,
            cfg,
        ).view(batch, n_agents, wall_slots)

    clearance = clearance_at()
    dy = SINGLE_SLIT_CLEARANCE_FD_Y_M
    dtheta = SINGLE_SLIT_CLEARANCE_FD_THETA_RAD
    gradient_y = (clearance_at(dy) - clearance_at(-dy)) / (2.0 * dy)
    gradient_theta = (
        clearance_at(0.0, dtheta) - clearance_at(0.0, -dtheta)
    ) / (2.0 * dtheta)

    # Locally feasible projection: promote this handle's measured world
    # velocity to a hypothetical pure translation of the complete payload.
    contact_velocity = base[..., 6:8] * 2.0
    projected_state = candidate_state.clone()
    projected_state[..., 0] += contact_velocity[..., 0:1] * float(cfg.dt)
    projected_state[..., 1] += contact_velocity[..., 1:2] * float(cfg.dt)
    projected_clearance = _visible_wall_candidate_clearance(
        projected_state.reshape(-1, 6),
        flat_wall_x,
        flat_gap_y,
        flat_gap_half,
        cfg,
    ).view(batch, n_agents, wall_slots)
    projected_delta_mm = (
        projected_clearance - clearance
    ) / SINGLE_SLIT_CLEARANCE_SCALE_M

    masked_clearance = clearance.masked_fill(~visibility, float("inf"))
    active_slot = masked_clearance.argmin(dim=-1, keepdim=True)

    def active(value: Tensor) -> Tensor:
        return value.gather(-1, active_slot).squeeze(-1)

    any_visible = visibility.any(dim=-1)
    features = torch.stack(
        (
            active(clearance) / SINGLE_SLIT_CLEARANCE_SCALE_M,
            active(gradient_y),
            active(gradient_theta),
            active(projected_delta_mm),
        ),
        dim=-1,
    ).clamp(
        min=-SINGLE_SLIT_FEEDBACK_FEATURE_CLIP,
        max=SINGLE_SLIT_FEEDBACK_FEATURE_CLIP,
    )
    return torch.where(any_visible[..., None], features, torch.zeros_like(features))


def single_slit_feedback_observations(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
) -> Tensor:
    """Strict-local18 plus four locally gated millimetre-scale features."""

    if scenario.wall_x.ndim != 2 or scenario.wall_x.shape[1] != 2:
        raise ValueError("single-slit feedback requires exactly two legacy wall slots")
    base = observations(state, scenario, cfg)
    if base.shape[-1] != SINGLE_SLIT_BASE_OBSERVATION_DIM:
        raise RuntimeError(
            "single-slit strict-local base width changed: "
            f"expected 18, got {base.shape[-1]}"
        )
    features = _single_slit_feedback_features_from_base(base, cfg)
    return torch.cat((base, features), dim=-1)


def single_slit_policy_observations(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    observation_schema: str,
) -> Tensor:
    """Dispatch one explicit single-slit observation experiment version."""

    if observation_schema == SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA:
        result = observations(state, scenario, cfg)
    elif observation_schema == SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA:
        result = single_slit_feedback_observations(state, scenario, cfg)
    else:
        raise ValueError(f"unknown single-slit observation schema: {observation_schema!r}")
    expected = (
        SINGLE_SLIT_BASE_OBSERVATION_DIM
        if observation_schema == SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA
        else SINGLE_SLIT_FEEDBACK_OBSERVATION_DIM
    )
    if result.shape[-1] != expected:
        raise RuntimeError(
            f"single-slit schema {observation_schema!r} produced width "
            f"{result.shape[-1]}, expected {expected}"
        )
    return result


def single_slit_observation_schema_for_dim(obs_dim: int) -> str:
    if obs_dim == SINGLE_SLIT_BASE_OBSERVATION_DIM:
        return SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA
    if obs_dim == SINGLE_SLIT_FEEDBACK_OBSERVATION_DIM:
        return SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA
    raise ValueError(f"unsupported single-slit observation width: {obs_dim}")


def single_slit_arm_config(
    cfg: EnvConfig,
    *,
    observation_schema: str = SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA,
) -> ArmConfig:
    """Return the small architecture for an explicit observation experiment.

    The default remains legacy 18D so callers reconstructing existing
    checkpoints are unaffected.  New 0.720 feedback training must request
    ``SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA`` and starts from a fresh model.
    """

    probe = single_slit_scenario(1, cfg)
    obs_dim = int(
        single_slit_policy_observations(
            probe.state0,
            probe,
            cfg,
            observation_schema,
        ).shape[-1]
    )
    edge_count = int(radius_edges(cfg, device=cfg.device).shape[0])
    if edge_count != 24:
        raise RuntimeError(f"single-slit fixed-handle graph must have 24 edges, got {edge_count}")
    return ArmConfig(
        obs_dim=obs_dim,
        hidden_dim=96,
        stalk_dim=32,
        edge_dim=12,
        gbp_steps=4,
        temporal_window=6,
        restriction_rank=4,
        queryable_goal_free_temporal=False,
        route_canonicalization=False,
    )


def build_single_slit_policy(
    cfg: EnvConfig,
    *,
    arm: str = "sheaf_gbp",
    observation_schema: str = SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA,
) -> BasePolicy:
    """Build a fresh policy; every arm receives the identical selected input."""

    return make_policy(
        arm,
        cfg,
        single_slit_arm_config(cfg, observation_schema=observation_schema),
    ).to(cfg.device)


def load_pose_path(path: Path | str) -> Tensor:
    """Load either a ``.npy`` pose array or an NPZ containing ``path``."""

    target = Path(path)
    if not target.is_file():
        raise ValueError(f"single-slit path artifact does not exist: {target}")
    value = np.load(target)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            if "path" not in value.files:
                raise ValueError("single-slit NPZ must contain a 'path' tensor")
            array = value["path"]
        finally:
            value.close()
    else:
        array = value
    result = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if result.ndim != 2 or result.shape[1] not in (3, 6) or result.shape[0] < 2:
        raise ValueError("single-slit path must have shape [N,3] or [N,6], N>=2")
    result = result[:, :3].contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("single-slit path contains non-finite poses")
    return result


def pose_path_sha256(path: Tensor) -> str:
    value = path.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256(b"single-slit-pose-path-v1")
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _trajectory_audit_scenario(states: Tensor, scenario: PianoBatch) -> PianoBatch:
    """Bind recursive-audit endpoints to the executed trajectory itself."""

    return PianoBatch(
        state0=states[:1].clone(),
        source=states[:1, :3].clone(),
        goal=states[-1:, :3].clone(),
        wall_x=scenario.wall_x[:1].clone(),
        gap_y=scenario.gap_y[:1].clone(),
        gap_half=scenario.gap_half[:1].clone(),
        leg=None,
        obstacle_rects=None,
    )


@torch.no_grad()
def rollout_single_slit_teacher(
    path: Tensor,
    cfg: EnvConfig,
    teacher_cfg: PlannerPathTeacherConfig,
    *,
    horizon: int | None = None,
    required_continuous_clearance_m: float = DEFAULT_REQUIRED_CONTINUOUS_CLEARANCE_M,
) -> NominalTeacherRollout:
    """Execute and continuously audit an ordered-path force teacher.

    A kinematic path that is collision-free but cannot be tracked by the real
    force/dynamics loop returns a failed rollout; callers must not use any of
    its labels as nominal training data.
    """

    teacher_cfg.validate()
    if required_continuous_clearance_m < 0.0:
        raise ValueError("required continuous clearance must be non-negative")
    limit = cfg.max_steps if horizon is None else horizon
    if limit <= 0:
        raise ValueError("teacher rollout horizon must be positive")
    device = torch.device(cfg.device)
    reference = path[:, :3].to(device=device, dtype=torch.float32)
    scenario = single_slit_scenario(1, cfg)
    state = scenario.state0.clone()
    padded_path = reference.unsqueeze(0)
    path_lengths = torch.tensor([reference.shape[0]], dtype=torch.long, device=device)
    minimum_index = torch.zeros(1, dtype=torch.long, device=device)

    states: list[Tensor] = []
    next_states: list[Tensor] = []
    forces: list[Tensor] = []
    sections: list[Tensor] = []
    indices: list[Tensor] = []
    clearances: list[Tensor] = []
    outcome = "timeout"
    finish_step = limit
    for tick in range(limit):
        teacher = planner_path_teacher_output(
            state,
            padded_path,
            cfg,
            teacher_cfg,
            path_lengths=path_lengths,
            minimum_path_index=minimum_index,
        )
        section = _planner_semantic_section(
            state,
            padded_path,
            path_lengths,
            teacher,
            cfg,
            teacher_cfg,
        )
        proposed = step_dynamics(state, teacher.forces, cfg)
        clearance = transition_clearance_margin(state, proposed, scenario, cfg)
        hit = transition_collision_mask(state, proposed, scenario, cfg)
        if not torch.equal(hit, clearance <= 0.0):
            raise RuntimeError("single-slit boolean SAT and signed clearance disagree")
        states.append(state[0].clone())
        next_states.append(proposed[0].clone())
        forces.append(teacher.forces[0].clone())
        sections.append(section[0].clone())
        indices.append(minimum_index[0].clone())
        clearances.append(clearance[0].clone())
        finish_step = tick + 1
        if bool(hit[0]):
            outcome = "collision"
            break
        state = proposed
        minimum_index = torch.maximum(minimum_index, teacher.diagnostics.nearest_index)
        if bool(success_mask(state, scenario, cfg)[0]):
            outcome = "success"
            break

    state_tensor = torch.stack(states) if states else state.new_zeros((0, 6))
    next_tensor = torch.stack(next_states) if next_states else state.new_zeros((0, 6))
    force_tensor = (
        torch.stack(forces)
        if forces
        else state.new_zeros((0, cfg.n_agents, 2))
    )
    section_tensor = torch.stack(sections) if sections else state.new_zeros((0, 8))
    index_tensor = (
        torch.stack(indices)
        if indices
        else torch.empty(0, dtype=torch.long, device=device)
    )
    clearance_tensor = torch.stack(clearances) if clearances else state.new_zeros((0,))

    recursive_record: Mapping[str, Any] | None = None
    if outcome == "success" and next_tensor.shape[0] > 0:
        executed = torch.cat((state_tensor[:1], next_tensor), dim=0)
        audit_scenario = _trajectory_audit_scenario(executed, scenario)
        audit = audit_swept_pose_path(
            executed,
            audit_scenario,
            cfg,
            gate_cfg=BacktrackingGateConfig(
                minimum_retreat_excursion_m=1.0e-3,
                endpoint_position_tolerance_m=1.0e-6,
                endpoint_angle_tolerance_rad=1.0e-6,
                required_continuous_clearance_m=required_continuous_clearance_m,
                maximum_subdivision_depth=20,
            ),
        )
        recursive_record = asdict(audit)
    return NominalTeacherRollout(
        state=state_tensor,
        next_state=next_tensor,
        target_forces=force_tensor,
        target_section=section_tensor,
        minimum_path_index=index_tensor,
        transition_clearance=clearance_tensor,
        outcome=outcome,
        finish_step=finish_step,
        recursive_audit=recursive_record,
    )


def checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    cfg: EnvConfig,
    arm_cfg: ArmConfig,
) -> dict[str, Any]:
    """Compare a checkpoint's exact observation/model/physics contract.

    ``device`` is runtime placement and is intentionally ignored.  All other
    fields are part of the continuation contract.  A mismatch does not forbid
    an explicitly labelled weight-initialization experiment, but it does
    forbid calling the checkpoint a resumable single-slit parent.
    """

    checkpoint_env = checkpoint.get("env_cfg")
    checkpoint_arm = checkpoint.get("arm_cfg")
    if not isinstance(checkpoint_env, Mapping) or not isinstance(checkpoint_arm, Mapping):
        return {
            "exact_resume_compatible": False,
            "reason": "checkpoint omitted env_cfg or arm_cfg",
            "env_differences": None,
            "arm_differences": None,
        }
    expected_env = asdict(cfg)
    try:
        normalized_checkpoint_env = asdict(EnvConfig(**dict(checkpoint_env)))
    except (TypeError, ValueError):
        normalized_checkpoint_env = dict(checkpoint_env)
    expected_arm = asdict(arm_cfg)
    env_keys = (set(expected_env) | set(normalized_checkpoint_env)) - {"device"}
    arm_keys = set(expected_arm) | set(checkpoint_arm)
    env_diff = {
        key: {
            "checkpoint": normalized_checkpoint_env.get(key),
            "single_slit": expected_env.get(key),
        }
        for key in sorted(env_keys)
        if normalized_checkpoint_env.get(key) != expected_env.get(key)
    }
    arm_diff = {
        key: {"checkpoint": checkpoint_arm.get(key), "single_slit": expected_arm.get(key)}
        for key in sorted(arm_keys)
        if checkpoint_arm.get(key) != expected_arm.get(key)
    }
    return {
        "exact_resume_compatible": not env_diff and not arm_diff,
        "reason": None if not env_diff and not arm_diff else "configuration mismatch",
        "env_differences": env_diff,
        "arm_differences": arm_diff,
    }


def training_contract_record(
    cfg: EnvConfig,
    arm_cfg: ArmConfig,
    protocol: SingleSlitDatasetProtocol | None = None,
) -> dict[str, Any]:
    options = protocol or SingleSlitDatasetProtocol()
    options.validate()
    return {
        "schema_version": SINGLE_SLIT_FEEDBACK_SCHEMA,
        "claim_status": "training contract; no learned-policy capability claim",
        "geometry": {
            "visible_aperture_m": VISIBLE_APERTURE_M,
            "effective_aperture_after_reserve_m": (
                VISIBLE_APERTURE_M - 2.0 * COLLISION_RESERVE_EACH_SIDE_M
            ),
            "wall_x_m": 0.0,
            "dummy_wall_x_m": DUMMY_WALL_X_M,
            "wall_slab_half_thickness_m": WALL_SLAB_HALF_THICKNESS_M,
            "source_pose": list(SOURCE_POSE),
            "goal_pose": list(GOAL_POSE),
            "one_way": True,
            "fixed_handles": True,
        },
        "environment": asdict(cfg),
        "arm": asdict(arm_cfg),
        "dataset": options.to_record(),
        "hard_gates": [
            "dynamics-executed planner teacher reaches goal within horizon",
            "every nominal transition has positive SATx4 clearance",
            "executed nominal pose path passes recursive continuous exact-SAT audit",
            "each tube/near-jam branch has positive starting clearance and safe teacher continuation",
            "Phase B DAgger starts only after Phase A nominal source success",
            "held-out seeds are never used for training or checkpoint selection",
        ],
    }


__all__ = [
    "COLLISION_RESERVE_EACH_SIDE_M",
    "DEFAULT_REQUIRED_CONTINUOUS_CLEARANCE_M",
    "DEFAULT_ROLLOUT_HORIZON",
    "DUMMY_WALL_X_M",
    "GOAL_POSE",
    "NominalTeacherRollout",
    "SINGLE_SLIT_BASE_OBSERVATION_DIM",
    "SINGLE_SLIT_CLEARANCE_FD_THETA_RAD",
    "SINGLE_SLIT_CLEARANCE_FD_Y_M",
    "SINGLE_SLIT_CLEARANCE_SCALE_M",
    "SINGLE_SLIT_FEEDBACK_FEATURE_CLIP",
    "SINGLE_SLIT_FEEDBACK_FEATURE_DIM",
    "SINGLE_SLIT_FEEDBACK_OBSERVATION_DIM",
    "SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA",
    "SINGLE_SLIT_FEEDBACK_SCHEMA",
    "SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA",
    "SINGLE_SLIT_OBSERVATION_SCHEMAS",
    "SOURCE_POSE",
    "SingleSlitDatasetProtocol",
    "VISIBLE_APERTURE_M",
    "WALL_SLAB_HALF_THICKNESS_M",
    "build_single_slit_policy",
    "checkpoint_compatibility",
    "load_pose_path",
    "pose_path_sha256",
    "rollout_single_slit_teacher",
    "single_slit_arm_config",
    "single_slit_env_config",
    "single_slit_feedback_observations",
    "single_slit_observation_schema_for_dim",
    "single_slit_policy_observations",
    "single_slit_scenario",
    "training_contract_record",
]
