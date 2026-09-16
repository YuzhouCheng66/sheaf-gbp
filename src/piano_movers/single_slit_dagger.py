"""Causal model-induced DAgger for the 0.720 m single-slit skill.

At each visited state the deployable policy is called exactly once.  Its
action and next hidden state are frozen before the privileged feedback teacher
is queried.  The frozen policy action—not the teacher action—then advances the
collection rollout.  A recurrent row is emitted only when that frozen action
would collide, and contains only the last safe pre-action states.  The
colliding proposal is retained solely in explicitly labelled audit arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .cem_teacher import feasible_forces_for_absolute_wrench
from .config import EnvConfig
from .env import (
    collision_clearance_margin,
    collision_mask,
    radius_edges,
    step_dynamics,
    transition_clearance_margin,
    transition_collision_mask,
    wrap_angle,
)
from .planner_path_teacher import PlannerPathTeacherConfig, planner_path_teacher_output
from .single_slit_dataset import (
    ExecutableTeacherArtifact,
    inverse_dynamics_wrench,
)
from .single_slit_feedback import (
    SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA,
    single_slit_observation_schema_for_dim,
    single_slit_policy_observations,
    single_slit_scenario,
)


SINGLE_SLIT_DAGGER_SCHEMA = "piano-movers-single-slit-model-induced-dagger-v2"


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SingleSlitDaggerConfig:
    sequence_length: int = 22
    burn_in_steps: int = 6
    rollout_horizon: int = 256
    query_cadence: int = 1
    maximum_rows: int = 256
    strict_position_tolerance_m: float = 0.03
    strict_angle_tolerance_rad: float = 0.05
    strict_speed_tolerance_mps: float = 0.05
    strict_omega_tolerance_radps: float = 0.08
    strict_stable_steps: int = 5
    minimum_teacher_transition_clearance_m: float = 1.0e-4

    def validate(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("DAgger sequence_length must be at least two")
        if not 0 < self.burn_in_steps < self.sequence_length:
            raise ValueError("DAgger burn-in must lie strictly inside the sequence")
        if self.rollout_horizon < self.sequence_length:
            raise ValueError("DAgger rollout horizon is shorter than one recurrent row")
        if self.query_cadence != 1:
            raise ValueError("dense recurrent DAgger currently requires query_cadence=1")
        if self.maximum_rows <= 0 or self.strict_stable_steps <= 0:
            raise ValueError("DAgger row/stability counts must be positive")
        tolerances = (
            self.strict_position_tolerance_m,
            self.strict_angle_tolerance_rad,
            self.strict_speed_tolerance_mps,
            self.strict_omega_tolerance_radps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("DAgger strict-final tolerances must be finite and positive")
        if (
            not math.isfinite(self.minimum_teacher_transition_clearance_m)
            or self.minimum_teacher_transition_clearance_m < 0.0
        ):
            raise ValueError("minimum teacher-transition clearance must be finite and non-negative")


@dataclass(frozen=True)
class LocalSE2PlanningTeacherConfig:
    """One-step receding-horizon controls for the privileged planning teacher.

    The old feedback teacher asks one PD controller to both choose a route and
    produce forces.  Near a millimetre-clearance branch boundary that coupling
    can leave it braking forever.  This teacher instead chooses one executable
    next payload pose on an ordered, exact-SAT-certified SE(2) path.  Force
    allocation remains a separate downstream problem for Sheaf-GBP.

    ``retreat_offsets`` are only tried when the next ordered waypoint is not a
    safe one-step target.  They make the teacher a genuine local replanner:
    it may recenter on the current path fiber or move to an earlier fiber
    before trying forward motion again.
    """

    forward_waypoints: int = 1
    retreat_offsets: tuple[int, ...] = (0, 1, 2, 4, 8)
    maximum_translation_m: float = 0.045
    maximum_rotation_rad: float = 0.090
    minimum_transition_clearance_m: float = 1.0e-7

    def validate(self) -> None:
        if self.forward_waypoints < 1:
            raise ValueError("forward_waypoints must be positive")
        if not self.retreat_offsets or self.retreat_offsets[0] != 0:
            raise ValueError("retreat_offsets must begin with the current fiber (zero)")
        if any(offset < 0 for offset in self.retreat_offsets):
            raise ValueError("retreat_offsets must be non-negative")
        positive = (
            self.maximum_translation_m,
            self.maximum_rotation_rad,
            self.minimum_transition_clearance_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("local planning scales and clearance must be finite and positive")


@dataclass(frozen=True)
class _TeacherLabel:
    forces: Tensor
    next_state: Tensor
    semantic8: Tensor
    clearance: Tensor
    nearest_index: int
    reference_index: int
    reference_pose: Tensor
    mode: str


@dataclass(frozen=True)
class _VisitedTick:
    state: Tensor
    teacher: _TeacherLabel
    policy_forces: Tensor
    policy_next_state: Tensor
    policy_clearance: Tensor
    policy_collision: bool
    rollout_tick: int
    frozen_action_sha256: str
    hidden_before_sha256: str
    hidden_after_sha256: str
    hidden_before: Tensor


@dataclass(frozen=True)
class SingleSlitDaggerCorpus:
    """Recurrent teacher labels plus causal policy-rollout audit arrays."""

    state: Tensor
    teacher_next_state: Tensor
    teacher_forces: Tensor
    semantic8: Tensor
    teacher_clearance: Tensor
    teacher_nearest_index: Tensor
    teacher_reference_index: Tensor
    teacher_reference_pose: Tensor
    policy_forces: Tensor
    policy_next_state: Tensor
    policy_clearance: Tensor
    policy_collision: Tensor
    rollout_tick: Tensor
    episode_index: Tensor
    trigger_state: Tensor
    trigger_policy_forces: Tensor
    trigger_policy_next_state: Tensor
    trigger_policy_clearance: Tensor
    trigger_rollout_tick: Tensor
    frozen_action_sha256: tuple[tuple[str, ...], ...]
    hidden_before_sha256: tuple[tuple[str, ...], ...]
    hidden_after_sha256: tuple[tuple[str, ...], ...]
    initial_hidden: Tensor
    teacher_mode: tuple[tuple[str, ...], ...]
    checkpoint_sha256: str
    teacher_artifact_sha256: str
    actor: str
    collection_summary: Mapping[str, Any]

    @property
    def rows(self) -> int:
        return int(self.state.shape[0])

    @property
    def time(self) -> int:
        return int(self.state.shape[1])

    def validate(self, cfg: EnvConfig, *, tolerance: float = 3.0e-6) -> None:
        if self.state.ndim != 3 or self.state.shape[-1] != 6:
            raise ValueError("DAgger state must have shape [rows,time,6]")
        rows, time = self.state.shape[:2]
        shapes = {
            "teacher_next_state": (self.teacher_next_state, (rows, time, 6)),
            "teacher_forces": (
                self.teacher_forces,
                (rows, time, cfg.n_agents, 2),
            ),
            "semantic8": (self.semantic8, (rows, time, 8)),
            "teacher_clearance": (self.teacher_clearance, (rows, time)),
            "teacher_nearest_index": (self.teacher_nearest_index, (rows, time)),
            "teacher_reference_index": (self.teacher_reference_index, (rows, time)),
            "teacher_reference_pose": (self.teacher_reference_pose, (rows, time, 3)),
            "policy_forces": (self.policy_forces, (rows, time, cfg.n_agents, 2)),
            "policy_next_state": (self.policy_next_state, (rows, time, 6)),
            "policy_clearance": (self.policy_clearance, (rows, time)),
            "policy_collision": (self.policy_collision, (rows, time)),
            "rollout_tick": (self.rollout_tick, (rows, time)),
            "episode_index": (self.episode_index, (rows,)),
            "trigger_state": (self.trigger_state, (rows, 6)),
            "trigger_policy_forces": (
                self.trigger_policy_forces,
                (rows, cfg.n_agents, 2),
            ),
            "trigger_policy_next_state": (
                self.trigger_policy_next_state,
                (rows, 6),
            ),
            "trigger_policy_clearance": (
                self.trigger_policy_clearance,
                (rows,),
            ),
            "trigger_rollout_tick": (self.trigger_rollout_tick, (rows,)),
        }
        for name, (value, expected) in shapes.items():
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"DAgger {name} has shape {tuple(value.shape)}, expected {expected}"
                )
        string_fields: Sequence[tuple[str, tuple[tuple[str, ...], ...]]] = (
            ("frozen_action_sha256", self.frozen_action_sha256),
            ("hidden_before_sha256", self.hidden_before_sha256),
            ("hidden_after_sha256", self.hidden_after_sha256),
            ("teacher_mode", self.teacher_mode),
        )
        for name, values in string_fields:
            if len(values) != rows or any(len(row) != time for row in values):
                raise ValueError(f"DAgger {name} is not [rows,time]")
        for hashes in (
            self.frozen_action_sha256,
            self.hidden_before_sha256,
            self.hidden_after_sha256,
        ):
            if any(len(value) != 64 for row in hashes for value in row):
                raise ValueError("DAgger tensor audit contains a malformed SHA-256")
        if len(self.checkpoint_sha256) != 64 or len(self.teacher_artifact_sha256) != 64:
            raise ValueError("DAgger provenance SHA-256 is malformed")
        if self.initial_hidden.ndim < 2 or self.initial_hidden.shape[0] != rows:
            raise ValueError("DAgger initial_hidden must have shape [rows,...]")
        if not bool(torch.isfinite(self.initial_hidden).all()):
            raise ValueError("DAgger initial_hidden contains non-finite values")
        for row in range(rows):
            if _tensor_sha256(self.initial_hidden[row : row + 1]) != self.hidden_before_sha256[row][0]:
                raise ValueError("DAgger initial_hidden disagrees with first-tick hidden audit")
        floats = (
            self.state,
            self.teacher_next_state,
            self.teacher_forces,
            self.semantic8,
            self.teacher_clearance,
            self.teacher_reference_pose,
            self.policy_forces,
            self.policy_next_state,
            self.policy_clearance,
            self.trigger_state,
            self.trigger_policy_forces,
            self.trigger_policy_next_state,
            self.trigger_policy_clearance,
        )
        if any(not bool(torch.isfinite(value).all()) for value in floats):
            raise ValueError("DAgger corpus contains non-finite values")
        if bool((self.teacher_clearance <= 0.0).any()):
            raise ValueError("DAgger teacher labels contain an unsafe transition")
        if bool(self.policy_collision[:, :-1].any()):
            raise ValueError("a DAgger training row contains a pre-terminal collision")
        expected_ticks = self.rollout_tick[:, :1] + torch.arange(
            time, dtype=self.rollout_tick.dtype, device=self.rollout_tick.device
        )
        if not torch.equal(self.rollout_tick, expected_ticks):
            raise ValueError("DAgger rollout ticks must be contiguous")
        if bool((self.teacher_nearest_index[:, 1:] < self.teacher_nearest_index[:, :-1]).any()):
            raise ValueError("DAgger teacher progress must be nondecreasing")

        flat_state = self.state.reshape(-1, 6)
        scenario = single_slit_scenario(flat_state.shape[0], cfg, initial_state=flat_state)
        state_clearance = collision_clearance_margin(flat_state, scenario, cfg)
        if bool((state_clearance <= 0.0).any()) or bool(
            collision_mask(flat_state, scenario, cfg).any()
        ):
            raise ValueError("a colliding policy state was stored in the DAgger corpus")

        flat_teacher_next = self.teacher_next_state.reshape(-1, 6)
        replay_teacher = step_dynamics(
            flat_state,
            self.teacher_forces.reshape(-1, cfg.n_agents, 2),
            cfg,
        )
        if float((replay_teacher - flat_teacher_next).abs().amax()) > tolerance:
            raise ValueError("DAgger teacher next-state replay failed")
        teacher_hit = transition_collision_mask(flat_state, flat_teacher_next, scenario, cfg)
        teacher_margin = transition_clearance_margin(flat_state, flat_teacher_next, scenario, cfg)
        if not torch.equal(teacher_hit, teacher_margin <= 0.0) or bool(teacher_hit.any()):
            raise ValueError("DAgger teacher transition SAT audit failed")
        if float((teacher_margin - self.teacher_clearance.reshape(-1)).abs().amax()) > tolerance:
            raise ValueError("DAgger stored teacher clearance disagrees with SAT replay")

        flat_policy_next = self.policy_next_state.reshape(-1, 6)
        replay_policy = step_dynamics(
            flat_state,
            self.policy_forces.reshape(-1, cfg.n_agents, 2),
            cfg,
        )
        if float((replay_policy - flat_policy_next).abs().amax()) > tolerance:
            raise ValueError("DAgger frozen policy-action replay failed")
        policy_hit = transition_collision_mask(flat_state, flat_policy_next, scenario, cfg)
        policy_margin = transition_clearance_margin(flat_state, flat_policy_next, scenario, cfg)
        if not torch.equal(policy_hit, policy_margin <= 0.0):
            raise RuntimeError("DAgger policy boolean/signed SAT authorities disagree")
        if not torch.equal(policy_hit.view(rows, time), self.policy_collision):
            raise ValueError("DAgger stored policy collision flags disagree with replay")
        if float((policy_margin - self.policy_clearance.reshape(-1)).abs().amax()) > tolerance:
            raise ValueError("DAgger stored policy clearance disagrees with SAT replay")

        trigger_scenario = single_slit_scenario(rows, cfg, initial_state=self.trigger_state)
        trigger_state_margin = collision_clearance_margin(self.trigger_state, trigger_scenario, cfg)
        if bool((trigger_state_margin <= 0.0).any()):
            raise ValueError("DAgger collision trigger itself is not a last-safe state")
        trigger_replay = step_dynamics(self.trigger_state, self.trigger_policy_forces, cfg)
        if float((trigger_replay - self.trigger_policy_next_state).abs().amax()) > tolerance:
            raise ValueError("DAgger collision-trigger action replay failed")
        trigger_hit = transition_collision_mask(
            self.trigger_state,
            self.trigger_policy_next_state,
            trigger_scenario,
            cfg,
        )
        trigger_margin = transition_clearance_margin(
            self.trigger_state,
            self.trigger_policy_next_state,
            trigger_scenario,
            cfg,
        )
        if not torch.equal(trigger_hit, trigger_margin <= 0.0) or not bool(trigger_hit.all()):
            raise ValueError("DAgger trigger is not an exact-SAT collision attempt")
        if float((trigger_margin - self.trigger_policy_clearance).abs().amax()) > tolerance:
            raise ValueError("DAgger trigger clearance disagrees with exact SAT replay")
        if bool((self.trigger_rollout_tick < self.rollout_tick[:, -1]).any()):
            raise ValueError("DAgger collision trigger predates its training window")

    def save(self, path: Path | str, cfg: EnvConfig) -> tuple[Path, Path]:
        self.validate(cfg)
        target = Path(path)
        if target.suffix != ".npz":
            raise ValueError("DAgger dataset output must end in .npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            state=self.state.detach().cpu().numpy(),
            teacher_next_state=self.teacher_next_state.detach().cpu().numpy(),
            teacher_forces=self.teacher_forces.detach().cpu().numpy(),
            semantic8=self.semantic8.detach().cpu().numpy(),
            teacher_clearance=self.teacher_clearance.detach().cpu().numpy(),
            teacher_nearest_index=self.teacher_nearest_index.detach().cpu().numpy(),
            teacher_reference_index=self.teacher_reference_index.detach().cpu().numpy(),
            teacher_reference_pose=self.teacher_reference_pose.detach().cpu().numpy(),
            policy_forces=self.policy_forces.detach().cpu().numpy(),
            policy_next_state=self.policy_next_state.detach().cpu().numpy(),
            policy_clearance=self.policy_clearance.detach().cpu().numpy(),
            policy_collision=self.policy_collision.detach().cpu().numpy(),
            rollout_tick=self.rollout_tick.detach().cpu().numpy(),
            episode_index=self.episode_index.detach().cpu().numpy(),
            trigger_state=self.trigger_state.detach().cpu().numpy(),
            trigger_policy_forces=self.trigger_policy_forces.detach().cpu().numpy(),
            trigger_policy_next_state=self.trigger_policy_next_state.detach().cpu().numpy(),
            trigger_policy_clearance=self.trigger_policy_clearance.detach().cpu().numpy(),
            trigger_rollout_tick=self.trigger_rollout_tick.detach().cpu().numpy(),
            frozen_action_sha256=np.asarray(self.frozen_action_sha256),
            hidden_before_sha256=np.asarray(self.hidden_before_sha256),
            hidden_after_sha256=np.asarray(self.hidden_after_sha256),
            initial_hidden=self.initial_hidden.detach().cpu().numpy(),
            teacher_mode=np.asarray(self.teacher_mode),
        )
        sidecar = target.with_suffix(".json")
        record = {
            "schema_version": SINGLE_SLIT_DAGGER_SCHEMA,
            "dataset": str(target),
            "dataset_sha256": _file_sha256(target),
            "actor": self.actor,
            "checkpoint_sha256": self.checkpoint_sha256,
            "teacher_artifact_sha256": self.teacher_artifact_sha256,
            "environment": asdict(cfg),
            "rows": self.rows,
            "time": self.time,
            "collection_summary": dict(self.collection_summary),
            "causal_contract": {
                "policy_action_frozen_before_teacher_query": True,
                "frozen_policy_action_executes_collection_transition": True,
                "teacher_action_executed_during_collection": False,
                "colliding_policy_proposal_and_trigger_state_are_audit_only": True,
                "collision_state_stored_as_training_state": False,
                "recurrent_state_history": (
                    "last contiguous teacher-recoverable safe policy-visited window "
                    "at or before the collision trigger"
                ),
                "hidden_state": "carried across every accepted policy transition",
                "window_initial_hidden": "exact inducing-checkpoint tensor",
            },
            "training_arrays": {
                "state": "state",
                "target_forces": "teacher_forces",
                "target_section": "semantic8",
                "target_transition_clearance": "teacher_clearance",
                "burn_in_supervision_boundary": self.collection_summary.get("burn_in_steps"),
                "initial_hidden": "initial_hidden",
            },
            "binding": {
                "observation_schema": self.collection_summary.get("observation_schema"),
                "hidden_shape_per_row": list(self.initial_hidden.shape[1:]),
                "checkpoint_sha256": self.checkpoint_sha256,
            },
        }
        temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(sidecar)
        return target, sidecar


def _teacher_controls() -> tuple[tuple[str, PlannerPathTeacherConfig], ...]:
    base = PlannerPathTeacherConfig(
        lookahead_distance=0.040,
        maximum_linear_speed=0.30,
        maximum_angular_speed=0.60,
        slowdown_distance=0.24,
        stop_distance=0.008,
        linear_position_gain=4.0,
        linear_velocity_gain=3.0,
        angle_position_gain=8.0,
        angular_velocity_gain=4.0,
    )
    return (
        ("next_reference_feedback", base),
        (
            "short_lookahead_feedback",
            replace(
                base,
                lookahead_distance=0.020,
                maximum_linear_speed=0.18,
                maximum_angular_speed=0.36,
                linear_position_gain=3.0,
                angle_position_gain=6.0,
            ),
        ),
    )


def _semantic8(
    state: Tensor,
    desired_twist: Tensor,
    realized_wrench: Tensor,
    artifact: ExecutableTeacherArtifact,
    path_index: int,
    cfg: EnvConfig,
) -> Tensor:
    left = max(path_index - 1, 0)
    right = min(path_index + 1, artifact.horizon)
    tangent = artifact.reference_poses[right, :2] - artifact.reference_poses[left, :2]
    tangent_norm = torch.linalg.vector_norm(tangent)
    if float(tangent_norm) <= 1.0e-8:
        tangent = state.new_tensor((1.0, 0.0))
    else:
        tangent = tangent / tangent_norm
    offset = state[0, :2] - artifact.reference_poses[path_index, :2]
    cross_track = (tangent[0] * offset[1] - tangent[1] * offset[0]).clamp(-1.0, 1.0)
    twist_scale = state.new_tensor((0.65, 0.65, 1.40))
    wrench_scale = state.new_tensor(
        (
            cfg.n_agents * cfg.max_force,
            cfg.n_agents * cfg.max_force,
            0.75 * cfg.n_agents * cfg.max_force,
        )
    )
    return torch.cat(
        (
            desired_twist / twist_scale,
            realized_wrench[0] / wrench_scale,
            cross_track.view(1),
            state.new_tensor((path_index / float(artifact.horizon),)),
        )
    )


@torch.no_grad()
def query_safe_feedback_teacher(
    state: Tensor,
    minimum_path_index: int,
    artifact: ExecutableTeacherArtifact,
    cfg: EnvConfig,
    *,
    visible_aperture_m: float = 0.720,
) -> tuple[_TeacherLabel | None, str | None]:
    """Query bounded path feedback, with a safe-fiber braking fallback."""

    if state.shape != (1, 6):
        raise ValueError("feedback teacher state must have shape [1,6]")
    path = artifact.reference_poses.unsqueeze(0)
    path_lengths = torch.tensor(
        [artifact.reference_poses.shape[0]], dtype=torch.long, device=state.device
    )
    progress = torch.tensor([minimum_path_index], dtype=torch.long, device=state.device)
    scenario = single_slit_scenario(
        1,
        cfg,
        initial_state=state,
        visible_aperture_m=visible_aperture_m,
    )
    latest = None
    for mode, controls in _teacher_controls():
        output = planner_path_teacher_output(
            state,
            path,
            cfg,
            controls,
            path_lengths=path_lengths,
            minimum_path_index=progress,
        )
        latest = output
        proposed = step_dynamics(state, output.forces, cfg)
        clearance = transition_clearance_margin(state, proposed, scenario, cfg)
        collision = transition_collision_mask(state, proposed, scenario, cfg)
        if not torch.equal(collision, clearance <= 0.0):
            raise RuntimeError("feedback teacher boolean/signed SAT disagreement")
        if not bool(collision[0]):
            nearest = int(output.diagnostics.nearest_index[0].item())
            reference = int(output.diagnostics.reference_index[0].item())
            return (
                _TeacherLabel(
                    forces=output.forces[0].clone(),
                    next_state=proposed[0].clone(),
                    semantic8=_semantic8(
                        state,
                        output.diagnostics.desired_twist[0],
                        output.wrench,
                        artifact,
                        nearest,
                        cfg,
                    ),
                    clearance=clearance[0].clone(),
                    nearest_index=nearest,
                    reference_index=reference,
                    reference_pose=output.diagnostics.reference_pose[0].clone(),
                    mode=mode,
                ),
                None,
            )

    # If forward path feedback cannot keep the next transition in the current
    # connected safe fiber, first brake.  This is still a privileged feedback
    # target, not an executed collection action.
    assert latest is not None
    requested, desired_twist = inverse_dynamics_wrench(state, state[0, :3], cfg, brake=True)
    forces, realized, _scale = feasible_forces_for_absolute_wrench(state, requested, cfg)
    proposed = step_dynamics(state, forces, cfg)
    clearance = transition_clearance_margin(state, proposed, scenario, cfg)
    collision = transition_collision_mask(state, proposed, scenario, cfg)
    if not torch.equal(collision, clearance <= 0.0):
        raise RuntimeError("feedback brake boolean/signed SAT disagreement")
    if bool(collision[0]):
        return None, "no_safe_feedback_action"
    nearest = int(latest.diagnostics.nearest_index[0].item())
    return (
        _TeacherLabel(
            forces=forces[0].clone(),
            next_state=proposed[0].clone(),
            semantic8=_semantic8(state, desired_twist, realized, artifact, nearest, cfg),
            clearance=clearance[0].clone(),
            nearest_index=nearest,
            reference_index=nearest,
            reference_pose=artifact.reference_poses[nearest].clone(),
            mode="safe_fiber_brake",
        ),
        None,
    )


def _bounded_local_pose_target(
    state: Tensor,
    reference_pose: Tensor,
    options: LocalSE2PlanningTeacherConfig,
) -> Tensor:
    """Clip a path reference to one small, locally executable SE(2) target."""

    delta_xy = reference_pose[:2] - state[0, :2]
    distance = torch.linalg.vector_norm(delta_xy)
    if float(distance) > options.maximum_translation_m:
        delta_xy = delta_xy * (options.maximum_translation_m / distance)
    delta_theta = wrap_angle(reference_pose[2] - state[0, 2]).clamp(
        -options.maximum_rotation_rad,
        options.maximum_rotation_rad,
    )
    return torch.cat(
        (
            state[0, :2] + delta_xy,
            wrap_angle(state[0, 2] + delta_theta).view(1),
        )
    )


@torch.no_grad()
def query_exact_local_se2_planning_teacher(
    state: Tensor,
    ordered_path_index: int,
    artifact: ExecutableTeacherArtifact,
    cfg: EnvConfig,
    *,
    visible_aperture_m: float = 0.720,
    options: LocalSE2PlanningTeacherConfig | None = None,
) -> tuple[_TeacherLabel | None, str | None]:
    """Choose an executable next-SE(2) plan, independently of force coordination.

    Candidate targets are converted to bounded forces and real next states
    before exact swept-SAT admission.  Consequently a returned label is not a
    kinematic wish: it is a dynamically executable, collision-free local plan.
    The nominal case advances exactly one certified waypoint.  If that move is
    unsafe off the path fiber, the teacher tries the current and progressively
    earlier fibers, enabling retreat and re-entry without decreasing the
    caller's monotone ordered-progress memory.
    """

    if state.shape != (1, 6):
        raise ValueError("local planning teacher state must have shape [1,6]")
    if not 0 <= ordered_path_index <= artifact.horizon:
        raise ValueError("ordered_path_index lies outside the teacher path")
    controls = options or LocalSE2PlanningTeacherConfig()
    controls.validate()
    scenario = single_slit_scenario(
        1,
        cfg,
        initial_state=state,
        visible_aperture_m=visible_aperture_m,
    )
    forward_index = min(
        artifact.horizon,
        ordered_path_index + controls.forward_waypoints,
    )
    candidates: list[tuple[str, int]] = [("local_se2_forward", forward_index)]
    candidates.extend(
        (
            "local_se2_recenter" if offset == 0 else f"local_se2_retreat_{offset}",
            max(0, ordered_path_index - offset),
        )
        for offset in controls.retreat_offsets
        if forward_index != ordered_path_index or offset != 0
    )

    # Preserve order while removing duplicate path indices near either end.
    unique_candidates: list[tuple[str, int]] = []
    seen_indices: set[int] = set()
    for mode, reference_index in candidates:
        if reference_index in seen_indices:
            continue
        seen_indices.add(reference_index)
        unique_candidates.append((mode, reference_index))

    best_safe: tuple[float, _TeacherLabel] | None = None
    for mode, reference_index in unique_candidates:
        target_pose = _bounded_local_pose_target(
            state,
            artifact.reference_poses[reference_index],
            controls,
        )
        requested, desired_twist = inverse_dynamics_wrench(
            state,
            target_pose,
            cfg,
            brake=False,
        )
        forces, realized, _scale = feasible_forces_for_absolute_wrench(
            state,
            requested,
            cfg,
        )
        proposed = step_dynamics(state, forces, cfg)
        clearance = transition_clearance_margin(state, proposed, scenario, cfg)
        collision = transition_collision_mask(state, proposed, scenario, cfg)
        if not torch.equal(collision, clearance <= 0.0):
            raise RuntimeError("local planner boolean/signed SAT disagreement")
        if (
            bool(collision[0])
            or float(clearance[0]) < controls.minimum_transition_clearance_m
        ):
            continue
        label = _TeacherLabel(
            forces=forces[0].clone(),
            next_state=proposed[0].clone(),
            semantic8=_semantic8(
                state,
                desired_twist,
                realized,
                artifact,
                ordered_path_index,
                cfg,
            ),
            clearance=clearance[0].clone(),
            # Only an accepted forward waypoint advances monotone progress.
            # Recenter/retreat actions retain the caller's current index.
            nearest_index=(
                reference_index if mode == "local_se2_forward" else ordered_path_index
            ),
            reference_index=reference_index,
            reference_pose=target_pose.clone(),
            mode=mode,
        )
        if mode == "local_se2_forward":
            return label, None
        score = float(clearance[0])
        if best_safe is None or score > best_safe[0]:
            best_safe = (score, label)

    if best_safe is not None:
        return best_safe[1], None

    # The current pose can be outside the one-step basin of every ordered path
    # fiber even though a short tangential move increases clearance.  Braking
    # forever in that state is a bad DAgger label: it teaches no recovery.
    # Search a compact local SE(2) lattice, admit candidates through the same
    # bounded-force dynamics and exact swept SAT authority, and prefer moves
    # that create clearance while remaining close to a nearby path fiber.
    # This is privileged training-time replanning only; the executed actor in
    # model-induced collection remains the learned policy.
    translation = controls.maximum_translation_m
    rotation = controls.maximum_rotation_rad
    slow_enough_to_escape = (
        float(torch.linalg.vector_norm(state[0, 3:5])) < 0.08
        and abs(float(state[0, 5])) < 0.12
    )
    local_offsets = tuple(
        (dx * translation, dy * translation, da * rotation)
        for dx in (-1.0, 0.0, 1.0)
        for dy in (-1.0, 0.0, 1.0)
        for da in (-1.0, 0.0, 1.0)
        if (dx, dy, da) != (0.0, 0.0, 0.0)
    ) if slow_enough_to_escape else ()
    path_left = max(0, ordered_path_index - 16)
    path_right = min(artifact.horizon, ordered_path_index + 16)
    nearby_path = artifact.reference_poses[path_left : path_right + 1]
    best_escape: tuple[float, _TeacherLabel] | None = None
    for dx, dy, dtheta in local_offsets:
        target_pose = state[0, :3].clone()
        target_pose[0] += dx
        target_pose[1] += dy
        target_pose[2] = wrap_angle(target_pose[2] + dtheta)
        requested, desired_twist = inverse_dynamics_wrench(
            state,
            target_pose,
            cfg,
            brake=False,
        )
        forces, realized, _scale = feasible_forces_for_absolute_wrench(
            state,
            requested,
            cfg,
        )
        proposed = step_dynamics(state, forces, cfg)
        clearance = transition_clearance_margin(state, proposed, scenario, cfg)
        collision = transition_collision_mask(state, proposed, scenario, cfg)
        if not torch.equal(collision, clearance <= 0.0):
            raise RuntimeError("local escape-lattice boolean/signed SAT disagreement")
        if (
            bool(collision[0])
            or float(clearance[0]) < controls.minimum_transition_clearance_m
        ):
            continue
        delta_xy = nearby_path[:, :2] - proposed[0, :2]
        delta_angle = wrap_angle(nearby_path[:, 2] - proposed[0, 2])
        distance2 = delta_xy.square().sum(-1) + (0.35 * delta_angle).square()
        local_index = int(distance2.argmin())
        matched_index = path_left + local_index
        # Clearance dominates at the immediate contact boundary; path
        # proximity and weak forward preference prevent aimless escape.
        score = (
            8.0 * float(clearance[0])
            - float(distance2[local_index])
            + 1.0e-4 * float(matched_index)
        )
        label = _TeacherLabel(
            forces=forces[0].clone(),
            next_state=proposed[0].clone(),
            semantic8=_semantic8(
                state,
                desired_twist,
                realized,
                artifact,
                ordered_path_index,
                cfg,
            ),
            clearance=clearance[0].clone(),
            nearest_index=ordered_path_index,
            reference_index=matched_index,
            reference_pose=target_pose.clone(),
            mode="local_se2_escape_lattice",
        )
        if best_escape is None or score > best_escape[0]:
            best_escape = (score, label)
    if best_escape is not None:
        return best_escape[1], None

    # A fast off-fiber state can make every pose-tracking candidate unsafe
    # even though cancelling its current twist is safe.  Braking is therefore
    # the final local-planning primitive, not a hidden execution gate: it is
    # returned as an ordinary next-SE(2) supervision target and must itself
    # pass the same bounded dynamics and swept exact-SAT authority.
    requested, desired_twist = inverse_dynamics_wrench(
        state,
        state[0, :3],
        cfg,
        brake=True,
    )
    forces, realized, _scale = feasible_forces_for_absolute_wrench(
        state,
        requested,
        cfg,
    )
    proposed = step_dynamics(state, forces, cfg)
    clearance = transition_clearance_margin(state, proposed, scenario, cfg)
    collision = transition_collision_mask(state, proposed, scenario, cfg)
    if not torch.equal(collision, clearance <= 0.0):
        raise RuntimeError("local planner brake boolean/signed SAT disagreement")
    if (
        not bool(collision[0])
        and float(clearance[0]) >= controls.minimum_transition_clearance_m
    ):
        return (
            _TeacherLabel(
                forces=forces[0].clone(),
                next_state=proposed[0].clone(),
                semantic8=_semantic8(
                    state,
                    desired_twist,
                    realized,
                    artifact,
                    ordered_path_index,
                    cfg,
                ),
                clearance=clearance[0].clone(),
                nearest_index=ordered_path_index,
                reference_index=ordered_path_index,
                reference_pose=state[0, :3].clone(),
                mode="local_se2_brake",
            ),
            None,
        )
    return None, "no_exact_safe_local_se2_plan"


def _strict_goal(state: Tensor, goal: Tensor, options: SingleSlitDaggerConfig) -> Tensor:
    position = torch.linalg.vector_norm(state[:, :2] - goal[:, :2], dim=-1)
    angle = wrap_angle(state[:, 2] - goal[:, 2]).abs()
    speed = torch.linalg.vector_norm(state[:, 3:5], dim=-1)
    omega = state[:, 5].abs()
    return (
        (position <= options.strict_position_tolerance_m)
        & (angle <= options.strict_angle_tolerance_rad)
        & (speed <= options.strict_speed_tolerance_mps)
        & (omega <= options.strict_omega_tolerance_radps)
    )


@torch.no_grad()
def collect_model_induced_dagger(
    policy: Any,
    artifact: ExecutableTeacherArtifact,
    cfg: EnvConfig,
    initial_states: Tensor,
    *,
    checkpoint_sha256: str,
    options: SingleSlitDaggerConfig | None = None,
) -> SingleSlitDaggerCorpus:
    """Collect pre-collision recurrent windows under a frozen deployable policy."""

    protocol = options or SingleSlitDaggerConfig()
    protocol.validate()
    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint_sha256 must be a 64-character digest")
    if initial_states.ndim != 2 or initial_states.shape[-1] != 6:
        raise ValueError("DAgger initial_states must have shape [episodes,6]")
    if initial_states.shape[0] <= 0:
        raise ValueError("DAgger requires at least one rollout")
    initial_states = initial_states.to(device=cfg.device, dtype=torch.float32)
    initial_scenario = single_slit_scenario(
        initial_states.shape[0], cfg, initial_state=initial_states
    )
    initial_margin = collision_clearance_margin(initial_states, initial_scenario, cfg)
    if bool(collision_mask(initial_states, initial_scenario, cfg).any()) or bool(
        (initial_margin <= 0.0).any()
    ):
        raise ValueError("DAgger rollout starts in collision")

    edge_index = radius_edges(cfg, device=cfg.device)
    rows: list[list[_VisitedTick]] = []
    episode_indices: list[int] = []
    trigger_states: list[Tensor] = []
    trigger_forces: list[Tensor] = []
    trigger_next_states: list[Tensor] = []
    trigger_clearances: list[Tensor] = []
    trigger_ticks: list[int] = []
    outcome_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    causal_action_checks = 0
    was_training = policy.training
    arm_cfg = getattr(policy, "arm_cfg", None)
    observation_schema = (
        SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA
        if arm_cfg is None
        else single_slit_observation_schema_for_dim(arm_cfg.obs_dim)
    )
    policy.eval()
    try:
        for episode in range(int(initial_states.shape[0])):
            state = initial_states[episode : episode + 1].clone()
            scenario = single_slit_scenario(1, cfg, initial_state=state)
            hidden = policy.init_hidden(1, state.device)
            minimum_path_index = 0
            stable_steps = 0
            history: list[_VisitedTick] = []
            last_complete_window: list[_VisitedTick] | None = None
            outcome = "timeout"
            for tick in range(protocol.rollout_horizon):
                hidden_before = hidden.clone()
                policy_observation = single_slit_policy_observations(
                    state, scenario, cfg, observation_schema
                )
                output = policy(policy_observation, edge_index, hidden)
                if (
                    output.forces.shape != (1, cfg.n_agents, 2)
                    or output.hidden.shape != hidden.shape
                ):
                    outcome = "invalid_policy_output"
                    break
                if not bool(torch.isfinite(output.forces).all()) or not bool(
                    torch.isfinite(output.hidden).all()
                ):
                    outcome = "invalid_policy_output"
                    break
                if float(torch.linalg.vector_norm(output.forces, dim=-1).amax()) > (
                    cfg.max_force + 1.0e-5
                ):
                    outcome = "invalid_policy_output"
                    break

                # This clone is the causal intervention boundary.  No teacher
                # result can alter the action that advances the rollout.
                frozen_action = output.forces.detach().clone()
                action_digest = _tensor_sha256(frozen_action)
                teacher, rejection = query_safe_feedback_teacher(
                    state, minimum_path_index, artifact, cfg
                )
                if _tensor_sha256(frozen_action) != action_digest:
                    raise RuntimeError("teacher query mutated the frozen policy action")
                causal_action_checks += 1

                policy_proposed = step_dynamics(state, frozen_action, cfg)
                policy_clearance = transition_clearance_margin(
                    state, policy_proposed, scenario, cfg
                )
                policy_hit = transition_collision_mask(state, policy_proposed, scenario, cfg)
                if not torch.equal(policy_hit, policy_clearance <= 0.0):
                    raise RuntimeError("policy boolean/signed SAT disagreement")

                if teacher is None:
                    assert rejection is not None
                    rejection_counts[rejection] = rejection_counts.get(rejection, 0) + 1
                    history.clear()
                elif float(teacher.clearance) < protocol.minimum_teacher_transition_clearance_m:
                    reason = "teacher_transition_below_minimum_clearance"
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    history.clear()
                else:
                    minimum_path_index = max(minimum_path_index, teacher.nearest_index)
                    history.append(
                        _VisitedTick(
                            state=state[0].clone(),
                            teacher=teacher,
                            policy_forces=frozen_action[0].clone(),
                            policy_next_state=policy_proposed[0].clone(),
                            policy_clearance=policy_clearance[0].clone(),
                            policy_collision=bool(policy_hit[0]),
                            rollout_tick=tick,
                            frozen_action_sha256=action_digest,
                            hidden_before_sha256=_tensor_sha256(hidden_before),
                            hidden_after_sha256=_tensor_sha256(output.hidden),
                            hidden_before=hidden_before[0].clone(),
                        )
                    )
                    if len(history) > protocol.sequence_length:
                        history.pop(0)
                    if len(history) == protocol.sequence_length:
                        last_complete_window = list(history)

                if bool(policy_hit[0]):
                    outcome = "collision"
                    if last_complete_window is not None:
                        rows.append(last_complete_window)
                        episode_indices.append(episode)
                        trigger_states.append(state[0].clone())
                        trigger_forces.append(frozen_action[0].clone())
                        trigger_next_states.append(policy_proposed[0].clone())
                        trigger_clearances.append(policy_clearance[0].clone())
                        trigger_ticks.append(tick)
                    else:
                        reason = "collision_before_full_recoverable_history"
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    break

                # Execute exactly the already-frozen policy transition and
                # carry exactly the hidden state produced before teacher query.
                state = policy_proposed
                hidden = output.hidden
                if bool(_strict_goal(state, scenario.goal, protocol)[0]):
                    stable_steps += 1
                else:
                    stable_steps = 0
                if stable_steps >= protocol.strict_stable_steps:
                    outcome = "success"
                    break
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            if len(rows) >= protocol.maximum_rows:
                break
    finally:
        policy.train(was_training)

    if not rows:
        raise RuntimeError(
            "no complete pre-collision recurrent DAgger row was collected; "
            f"outcomes={outcome_counts}, rejected={rejection_counts}"
        )

    def stack_tick(field):
        return torch.stack([torch.stack([field(tick) for tick in row]) for row in rows])

    corpus = SingleSlitDaggerCorpus(
        state=stack_tick(lambda tick: tick.state),
        teacher_next_state=stack_tick(lambda tick: tick.teacher.next_state),
        teacher_forces=stack_tick(lambda tick: tick.teacher.forces),
        semantic8=stack_tick(lambda tick: tick.teacher.semantic8),
        teacher_clearance=stack_tick(lambda tick: tick.teacher.clearance),
        teacher_nearest_index=torch.tensor(
            [[tick.teacher.nearest_index for tick in row] for row in rows],
            dtype=torch.long,
            device=initial_states.device,
        ),
        teacher_reference_index=torch.tensor(
            [[tick.teacher.reference_index for tick in row] for row in rows],
            dtype=torch.long,
            device=initial_states.device,
        ),
        teacher_reference_pose=stack_tick(lambda tick: tick.teacher.reference_pose),
        policy_forces=stack_tick(lambda tick: tick.policy_forces),
        policy_next_state=stack_tick(lambda tick: tick.policy_next_state),
        policy_clearance=stack_tick(lambda tick: tick.policy_clearance),
        policy_collision=torch.tensor(
            [[tick.policy_collision for tick in row] for row in rows],
            dtype=torch.bool,
            device=initial_states.device,
        ),
        rollout_tick=torch.tensor(
            [[tick.rollout_tick for tick in row] for row in rows],
            dtype=torch.long,
            device=initial_states.device,
        ),
        episode_index=torch.tensor(episode_indices, dtype=torch.long, device=initial_states.device),
        trigger_state=torch.stack(trigger_states),
        trigger_policy_forces=torch.stack(trigger_forces),
        trigger_policy_next_state=torch.stack(trigger_next_states),
        trigger_policy_clearance=torch.stack(trigger_clearances),
        trigger_rollout_tick=torch.tensor(
            trigger_ticks, dtype=torch.long, device=initial_states.device
        ),
        frozen_action_sha256=tuple(
            tuple(tick.frozen_action_sha256 for tick in row) for row in rows
        ),
        hidden_before_sha256=tuple(
            tuple(tick.hidden_before_sha256 for tick in row) for row in rows
        ),
        hidden_after_sha256=tuple(tuple(tick.hidden_after_sha256 for tick in row) for row in rows),
        initial_hidden=torch.stack([row[0].hidden_before for row in rows]),
        teacher_mode=tuple(tuple(tick.teacher.mode for tick in row) for row in rows),
        checkpoint_sha256=checkpoint_sha256,
        teacher_artifact_sha256=artifact.sha256,
        actor=str(policy.arm_name),
        collection_summary={
            "rollouts_requested": int(initial_states.shape[0]),
            "rollouts_executed": sum(outcome_counts.values()),
            "rows": len(rows),
            "time": protocol.sequence_length,
            "burn_in_steps": protocol.burn_in_steps,
            "supervised_steps": protocol.sequence_length - protocol.burn_in_steps,
            "outcomes": outcome_counts,
            "rejected": rejection_counts,
            "causal_frozen_action_checks": causal_action_checks,
            "teacher_query_cadence": protocol.query_cadence,
            "minimum_teacher_transition_clearance_admission_m": (
                protocol.minimum_teacher_transition_clearance_m
            ),
            "observation_schema": observation_schema,
            "hidden_shape_per_row": list(rows[0][0].hidden_before.shape),
            "policy_action_executed": True,
            "teacher_action_executed": False,
            "collision_state_stored": False,
            "minimum_teacher_clearance_m": float(
                stack_tick(lambda tick: tick.teacher.clearance).amin().cpu()
            ),
            "terminal_policy_collision_clearance_max_m": float(
                torch.stack(trigger_clearances).amax().cpu()
            ),
            "teacher_recoverability_lag_ticks": [
                trigger_tick - int(row[-1].rollout_tick)
                for trigger_tick, row in zip(trigger_ticks, rows, strict=True)
            ],
        },
    )
    corpus.validate(cfg)
    return corpus


@torch.no_grad()
def load_single_slit_dagger_corpus(
    path: Path | str,
    cfg: EnvConfig,
    *,
    validate: bool = True,
) -> SingleSlitDaggerCorpus:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    sidecar = target.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError("DAgger corpus is missing its audit sidecar")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if record.get("schema_version") != SINGLE_SLIT_DAGGER_SCHEMA:
        raise ValueError("unexpected DAgger corpus schema")
    if record.get("dataset_sha256") != _file_sha256(target):
        raise ValueError("DAgger dataset bytes changed after audit sidecar creation")
    binding = record.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("DAgger v2 sidecar omitted schema/checkpoint/hidden binding")
    if binding.get("checkpoint_sha256") != record.get("checkpoint_sha256"):
        raise ValueError("DAgger checkpoint binding disagrees with provenance")
    observation_schema = binding.get("observation_schema")
    summary = record.get("collection_summary", {})
    if not isinstance(observation_schema, str) or observation_schema != summary.get(
        "observation_schema"
    ):
        raise ValueError("DAgger observation-schema binding disagrees with collection audit")
    device = torch.device(cfg.device)
    with np.load(target) as arrays:
        required = (
            "state",
            "teacher_next_state",
            "teacher_forces",
            "semantic8",
            "teacher_clearance",
            "teacher_nearest_index",
            "teacher_reference_index",
            "teacher_reference_pose",
            "policy_forces",
            "policy_next_state",
            "policy_clearance",
            "policy_collision",
            "rollout_tick",
            "episode_index",
            "trigger_state",
            "trigger_policy_forces",
            "trigger_policy_next_state",
            "trigger_policy_clearance",
            "trigger_rollout_tick",
            "frozen_action_sha256",
            "hidden_before_sha256",
            "hidden_after_sha256",
            "initial_hidden",
            "teacher_mode",
        )
        missing = [name for name in required if name not in arrays.files]
        if missing:
            raise ValueError(f"DAgger corpus omitted arrays: {missing}")

        def floating(name: str) -> Tensor:
            return torch.as_tensor(
                np.asarray(arrays[name]), dtype=torch.float32, device=device
            ).clone()

        def strings(name: str) -> tuple[tuple[str, ...], ...]:
            return tuple(
                tuple(str(item) for item in row) for row in np.asarray(arrays[name]).tolist()
            )

        corpus = SingleSlitDaggerCorpus(
            state=floating("state"),
            teacher_next_state=floating("teacher_next_state"),
            teacher_forces=floating("teacher_forces"),
            semantic8=floating("semantic8"),
            teacher_clearance=floating("teacher_clearance"),
            teacher_nearest_index=torch.as_tensor(
                np.asarray(arrays["teacher_nearest_index"]),
                dtype=torch.long,
                device=device,
            ).clone(),
            teacher_reference_index=torch.as_tensor(
                np.asarray(arrays["teacher_reference_index"]),
                dtype=torch.long,
                device=device,
            ).clone(),
            teacher_reference_pose=floating("teacher_reference_pose"),
            policy_forces=floating("policy_forces"),
            policy_next_state=floating("policy_next_state"),
            policy_clearance=floating("policy_clearance"),
            policy_collision=torch.as_tensor(
                np.asarray(arrays["policy_collision"]),
                dtype=torch.bool,
                device=device,
            ).clone(),
            rollout_tick=torch.as_tensor(
                np.asarray(arrays["rollout_tick"]),
                dtype=torch.long,
                device=device,
            ).clone(),
            episode_index=torch.as_tensor(
                np.asarray(arrays["episode_index"]),
                dtype=torch.long,
                device=device,
            ).clone(),
            trigger_state=floating("trigger_state"),
            trigger_policy_forces=floating("trigger_policy_forces"),
            trigger_policy_next_state=floating("trigger_policy_next_state"),
            trigger_policy_clearance=floating("trigger_policy_clearance"),
            trigger_rollout_tick=torch.as_tensor(
                np.asarray(arrays["trigger_rollout_tick"]),
                dtype=torch.long,
                device=device,
            ).clone(),
            frozen_action_sha256=strings("frozen_action_sha256"),
            hidden_before_sha256=strings("hidden_before_sha256"),
            hidden_after_sha256=strings("hidden_after_sha256"),
            initial_hidden=floating("initial_hidden"),
            teacher_mode=strings("teacher_mode"),
            checkpoint_sha256=str(record.get("checkpoint_sha256")),
            teacher_artifact_sha256=str(record.get("teacher_artifact_sha256")),
            actor=str(record.get("actor")),
            collection_summary=dict(record.get("collection_summary", {})),
        )
    if list(corpus.initial_hidden.shape[1:]) != binding.get("hidden_shape_per_row"):
        raise ValueError("DAgger hidden tensor shape disagrees with sidecar binding")
    if validate:
        corpus.validate(cfg)
    return corpus


@torch.no_grad()
def sample_dagger_batch(
    corpus: SingleSlitDaggerCorpus,
    cfg: EnvConfig,
    *,
    batch_size: int,
    protocol: Any,
    generator: torch.Generator | None = None,
    brake_weight: float = 1.0,
    final_transition_weight: float = 1.0,
    clearance_threshold_m: float = 0.0,
    clearance_weight: float = 1.0,
    stratify_brake_rows: bool = False,
    nominal_anchor_fraction: float = 0.0,
    supervised_time_decay: float = 1.0,
    trigger_stratified_loss: bool = False,
    approach_stratum_weight: float = 0.5,
    boundary_stratum_weight: float = 0.25,
    recovery_stratum_weight: float = 0.25,
) -> Any:
    """Adapt audited DAgger rows to the existing recurrent BC batch API.

    The default path is the historical per-transition weighting byte for byte.
    ``trigger_stratified_loss`` is an explicit alternative for collision-bound
    DAgger rows.  It admits only rows containing the complete relative-tick
    support ``{-4,-3,-2,-1,0}``, then gives each row exactly the declared mass:
    the mean of ticks -4..-2 receives ``approach_stratum_weight`` and ticks -1
    and 0 receive the boundary and recovery masses.  All other direct
    supervision weights are zero, although recurrent hidden state is still
    carried through the full stored prefix.
    """

    from .single_slit_0720_bc import SingleSlitBCBatch

    if batch_size <= 0:
        raise ValueError("DAgger batch_size must be positive")
    if protocol.sequence_length != corpus.time:
        raise ValueError("DAgger corpus time does not match recurrent protocol")
    if protocol.burn_in_steps != corpus.collection_summary.get("burn_in_steps"):
        raise ValueError("DAgger burn-in does not match collection audit")
    weights_to_check = (brake_weight, final_transition_weight, clearance_weight)
    if any(not math.isfinite(value) or value <= 0.0 for value in weights_to_check):
        raise ValueError("DAgger transition weights must be finite and positive")
    if not math.isfinite(clearance_threshold_m) or clearance_threshold_m < 0.0:
        raise ValueError("DAgger clearance threshold must be finite and non-negative")
    if not math.isfinite(nominal_anchor_fraction) or not 0.0 <= nominal_anchor_fraction <= 1.0:
        raise ValueError("DAgger nominal anchor fraction must lie in [0,1]")
    if (
        nominal_anchor_fraction > 0.0
        and corpus.collection_summary.get("collection_kind")
        == "policy-prefix6-teacher-recovery16"
    ):
        raise ValueError(
            "recovery-tube corpora require nominal_anchor_fraction=0; "
            "row zero is a model-induced recovery row, not a nominal source row"
        )
    if not math.isfinite(supervised_time_decay) or not 0.0 < supervised_time_decay <= 1.0:
        raise ValueError("supervised time decay must lie in (0,1]")
    stratum_weights = (
        approach_stratum_weight,
        boundary_stratum_weight,
        recovery_stratum_weight,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in stratum_weights):
        raise ValueError("trigger-stratum weights must be finite and non-negative")
    if trigger_stratified_loss and not math.isclose(
        sum(stratum_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise ValueError("trigger-stratum weights must sum to one")
    if supervised_time_decay != 1.0 and corpus.collection_summary.get(
        "collection_kind"
    ) != "policy-prefix6-teacher-recovery16":
        raise ValueError("supervised time decay is only valid for recovery-tube corpora")
    if trigger_stratified_loss and supervised_time_decay != 1.0:
        raise ValueError("trigger-stratified loss does not combine with time decay")
    trigger_ticks = getattr(corpus, "trigger_rollout_tick", None)
    if trigger_stratified_loss and (
        not isinstance(trigger_ticks, Tensor) or trigger_ticks.shape != (corpus.rows,)
    ):
        raise ValueError(
            "trigger-stratified loss requires per-row trigger_rollout_tick provenance"
        )
    if nominal_anchor_fraction > 0.0:
        if corpus.episode_index.shape != (corpus.rows,) or int(corpus.episode_index[0]) != 0:
            raise ValueError("DAgger nominal anchor row0 must have episode_index==0")
    supervised_slice = range(protocol.burn_in_steps, corpus.time)
    anchor_count = int(round(batch_size * nominal_anchor_fraction))
    remaining = batch_size - anchor_count
    eligible_start = 1 if nominal_anchor_fraction > 0.0 else 0
    eligible_rows = list(range(eligible_start, corpus.rows))
    complete_trigger_rows: set[int] | None = None
    if trigger_stratified_loss:
        complete_trigger_rows = set()
        assert isinstance(trigger_ticks, Tensor)
        for row in range(corpus.rows):
            relative = corpus.rollout_tick[row] - trigger_ticks[row]
            supervised_relative = relative[protocol.burn_in_steps :]
            counts = (
                int(((supervised_relative >= -4) & (supervised_relative <= -2)).sum()),
                int((supervised_relative == -1).sum()),
                int((supervised_relative == 0).sum()),
            )
            if counts == (3, 1, 1):
                complete_trigger_rows.add(row)
        if anchor_count > 0 and 0 not in complete_trigger_rows:
            raise ValueError(
                "trigger-stratified nominal anchor row lacks complete -4..0 support"
            )
        eligible_rows = [row for row in eligible_rows if row in complete_trigger_rows]
    if remaining > 0 and not eligible_rows:
        reason = (
            "no non-anchor DAgger row has complete trigger-relative -4..0 support"
            if trigger_stratified_loss
            else "DAgger nominal anchoring left no non-anchor rows for mixed sampling"
        )
        raise ValueError(reason)
    brake_rows = [
        row for row, modes in enumerate(corpus.teacher_mode)
        if row in eligible_rows
        and any(modes[tick] == "safe_fiber_brake" for tick in supervised_slice)
    ]
    other_rows = [row for row in eligible_rows if row not in set(brake_rows)]
    if remaining == 0:
        remainder_selected = torch.empty(0, dtype=torch.long, device=corpus.state.device)
    elif stratify_brake_rows and brake_rows and other_rows:
        brake_count = (remaining + 1) // 2
        other_count = remaining - brake_count
        pools = (
            torch.tensor(brake_rows, device=corpus.state.device),
            torch.tensor(other_rows, device=corpus.state.device),
        )
        remainder_selected = torch.cat(
            (
                pools[0].index_select(0, torch.randint(len(brake_rows), (brake_count,), device=corpus.state.device, generator=generator)),
                pools[1].index_select(0, torch.randint(len(other_rows), (other_count,), device=corpus.state.device, generator=generator)),
            )
        )
    else:
        eligible = torch.tensor(eligible_rows, dtype=torch.long, device=corpus.state.device)
        remainder_selected = eligible.index_select(
            0,
            torch.randint(len(eligible_rows), (remaining,), device=corpus.state.device, generator=generator),
        )
    selected = torch.cat(
        (
            torch.zeros(anchor_count, dtype=torch.long, device=corpus.state.device),
            remainder_selected,
        )
    )
    if anchor_count > 0 and batch_size > 1:
        selected = selected.index_select(
            0,
            torch.randperm(batch_size, device=corpus.state.device, generator=generator),
        )
    state = corpus.state.index_select(0, selected).transpose(0, 1).contiguous()
    force = corpus.teacher_forces.index_select(0, selected).transpose(0, 1).contiguous()
    section = corpus.semantic8.index_select(0, selected).transpose(0, 1).contiguous()
    clearance = corpus.teacher_clearance.index_select(0, selected).transpose(0, 1).contiguous()
    initial_hidden = corpus.initial_hidden.index_select(0, selected).contiguous()
    valid = torch.ones(corpus.time, batch_size, dtype=torch.bool, device=corpus.state.device)
    context = (
        torch.arange(corpus.time, device=corpus.state.device)[:, None] < protocol.burn_in_steps
    ).expand(-1, batch_size)
    supervision_weight = torch.zeros(
        corpus.time, batch_size, dtype=corpus.state.dtype, device=corpus.state.device
    ) if trigger_stratified_loss else torch.ones(
        corpus.time, batch_size, dtype=corpus.state.dtype, device=corpus.state.device
    )
    selected_rows = selected.detach().cpu().tolist()
    for column, row in enumerate(selected_rows):
        if trigger_stratified_loss:
            assert isinstance(trigger_ticks, Tensor)
            relative = corpus.rollout_tick[row] - trigger_ticks[row]
            approach = (relative >= -4) & (relative <= -2)
            boundary = relative == -1
            recovery = relative == 0
            if (
                int(approach.sum()) != 3
                or int(boundary.sum()) != 1
                or int(recovery.sum()) != 1
            ):
                raise RuntimeError("selected row lost complete trigger-relative support")
            supervision_weight[approach, column] = approach_stratum_weight / 3.0
            supervision_weight[boundary, column] = boundary_stratum_weight
            supervision_weight[recovery, column] = recovery_stratum_weight
            continue
        for tick in supervised_slice:
            candidates = [1.0]
            if corpus.teacher_mode[row][tick] == "safe_fiber_brake":
                candidates.append(brake_weight)
            if tick == corpus.time - 1:
                candidates.append(final_transition_weight)
            if clearance_threshold_m > 0.0 and float(corpus.teacher_clearance[row, tick]) <= clearance_threshold_m:
                candidates.append(clearance_weight)
            # Overlapping audited strata use the strongest declared weight;
            # they are never multiplied without independent justification.
            supervised_age = tick - protocol.burn_in_steps
            supervision_weight[tick, column] = (
                max(candidates) * supervised_time_decay**supervised_age
            )
    supervision_weight[context] = 0.0
    supervision = supervision_weight > 0.0 if trigger_stratified_loss else ~context
    scenario = single_slit_scenario(batch_size, cfg, initial_state=state[0])
    return SingleSlitBCBatch(
        scenario=scenario,
        sequence_state=state,
        sequence_target_forces=force,
        sequence_target_section=section,
        sequence_transition_clearance=clearance,
        sequence_valid_mask=valid,
        sequence_context_mask=context,
        sequence_supervision_mask=supervision,
        supervised_start=corpus.rollout_tick.index_select(0, selected)[:, 0],
        selected_stratum=torch.full((batch_size,), 4, dtype=torch.int8, device=corpus.state.device),
        protocol=protocol,
        initial_hidden=initial_hidden,
        supervision_weight=supervision_weight,
        selected_record_index=selected,
    )


__all__ = [
    "SINGLE_SLIT_DAGGER_SCHEMA",
    "LocalSE2PlanningTeacherConfig",
    "SingleSlitDaggerConfig",
    "SingleSlitDaggerCorpus",
    "collect_model_induced_dagger",
    "load_single_slit_dagger_corpus",
    "query_exact_local_se2_planning_teacher",
    "query_safe_feedback_teacher",
    "sample_dagger_batch",
]
