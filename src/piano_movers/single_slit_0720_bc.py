"""Recurrent behavior cloning for the certified 0.720 m single-slit skill.

This module is intentionally independent of the sealed serpentine/V5 lineage.
It consumes the dynamics-executed teacher artifact, rechecks its dynamics and
SATx4 transitions, and exposes one identical recurrent-window sampler for
Sheaf-GBP and its three matched controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .env import (
    PianoBatch,
    net_wrench_from_forces,
    step_dynamics,
    transition_clearance_margin,
    transition_collision_mask,
    wrap_angle,
)
from .gbp import GBPTopology, homogeneous_topology
from .models import (
    BasePolicy,
    PolicyOutput,
    SheafGBPPolicy,
    TemporalStateMatchedMPNNPolicy,
    make_policy,
)
from .single_slit_feedback import (
    GOAL_POSE,
    SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA,
    SINGLE_SLIT_OBSERVATION_SCHEMAS,
    SOURCE_POSE,
    VISIBLE_APERTURE_M,
    single_slit_arm_config,
    single_slit_env_config,
    single_slit_observation_schema_for_dim,
    single_slit_policy_observations,
    single_slit_scenario,
)
from .single_slit_contact_observation import (
    SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1,
    SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
)
from .single_slit_dataset import STRATUM_NAMES, SingleSlitCorpus


EXECUTABLE_TEACHER_SCHEMA = "piano-movers-single-slit-executable-teacher-v1"
LEGACY_SINGLE_SLIT_BC_SCHEMA = "piano-movers-single-slit-0720-recurrent-bc-v1"
SINGLE_SLIT_BC_SCHEMA = (
    "piano-movers-single-slit-0720-recurrent-bc-exact-clearance4-v2"
)
SINGLE_SLIT_CONTACT_BC_SCHEMA_V1 = (
    "piano-movers-single-slit-0720-contact-feedback27-v1"
)
LEGACY_CONTROL_ARMS = ("raw_full", "comm_matched", "no_comm")
FAIR_RAW_FULL_ARM = "raw_full_temporal32"
FAIR_COMM_MATCHED_ARM = "comm_matched_temporal30"
FAIR_NO_COMM_ARM = "no_comm_temporal30_masked"
FAIR_CONTROL_ARMS = (
    FAIR_RAW_FULL_ARM,
    FAIR_COMM_MATCHED_ARM,
    FAIR_NO_COMM_ARM,
)
SUPPORTED_ARMS = ("sheaf_gbp", *LEGACY_CONTROL_ARMS, *FAIR_CONTROL_ARMS)
SEMANTIC_WIDTH = 8
NOMINAL_CLEARANCE_SOFTMIN_TEMPERATURE_M = 1.0e-3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(manifest_path: Path, declared: object) -> Path:
    if not isinstance(declared, str) or not declared:
        raise ValueError("teacher manifest omitted rollout_artifact")
    raw = Path(declared)
    candidates = (
        raw,
        manifest_path.parent / raw,
        manifest_path.parent / raw.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"teacher rollout artifact not found: {declared}")


def _require_close(name: str, actual: object, expected: float, atol: float = 1e-8) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"teacher {name} must be numeric")
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"teacher {name}={actual!r}, expected {expected!r}")


@dataclass(frozen=True)
class SingleSlitTeacherTrajectory:
    """One audited nominal force trajectory, resident on the selected device."""

    manifest_path: Path
    artifact_path: Path
    manifest_sha256: str
    artifact_sha256: str
    states: Tensor
    forces: Tensor
    target_section: Tensor
    transition_clearance: Tensor
    realized_wrenches: Tensor
    manifest: Mapping[str, Any]

    @property
    def horizon(self) -> int:
        return int(self.forces.shape[0])


def _semantic_targets(states: Tensor, realized_wrenches: Tensor, manifest: Mapping[str, Any]) -> Tensor:
    """Build twist3, wrench3, cross-track, progress labels from the rollout."""

    timing = manifest["time_parameterization"]
    dt = float(timing["dt_seconds"])
    linear_scale = float(timing["maximum_translation_per_tick_m"]) / dt
    angular_scale = float(timing["maximum_angle_per_tick_rad"]) / dt
    if linear_scale <= 0.0 or angular_scale <= 0.0:
        raise ValueError("teacher time-parameterization scales must be positive")
    twist = states[1:, 3:6] / states.new_tensor(
        (linear_scale, linear_scale, angular_scale)
    )
    wrench = realized_wrenches / states.new_tensor((48.0, 48.0, 36.0))

    delta_xy = states[1:, :2] - states[:-1, :2]
    delta_theta = wrap_angle(states[1:, 2] - states[:-1, 2])
    segment = torch.sqrt(delta_xy.square().sum(-1) + (0.35 * delta_theta).square())
    cumulative = torch.cat((segment.new_zeros(1), segment.cumsum(0)))
    total = cumulative[-1].clamp_min(1e-8)
    progress = cumulative[:-1] / total
    cross_track = states.new_zeros(states.shape[0] - 1)
    result = torch.cat(
        (twist, wrench, cross_track[:, None], progress[:, None]), dim=-1
    )
    if result.shape != (states.shape[0] - 1, SEMANTIC_WIDTH):
        raise RuntimeError("single-slit semantic target width changed")
    return result


@torch.no_grad()
def load_executable_teacher(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> SingleSlitTeacherTrajectory:
    """Load and independently revalidate the executable teacher artifact."""

    supplied = Path(path)
    if supplied.suffix == ".npz":
        manifest_path = supplied.with_suffix("").with_suffix(".json")
        if not manifest_path.is_file():
            matches = list(supplied.parent.glob(supplied.stem.replace(".rollout", "") + ".json"))
            if len(matches) != 1:
                raise FileNotFoundError("could not locate the teacher JSON beside the NPZ")
            manifest_path = matches[0]
    else:
        manifest_path = supplied
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("teacher manifest must be a JSON object")
    if manifest.get("schema_version") != EXECUTABLE_TEACHER_SCHEMA:
        raise ValueError("teacher manifest schema mismatch")

    case = manifest.get("case")
    rollout = manifest.get("rollout")
    shortcut = manifest.get("shortcut")
    env_record = manifest.get("env_config")
    if not all(isinstance(value, Mapping) for value in (case, rollout, shortcut, env_record)):
        raise ValueError("teacher manifest omitted case/env/audit mappings")
    assert isinstance(case, Mapping) and isinstance(rollout, Mapping)
    assert isinstance(shortcut, Mapping) and isinstance(env_record, Mapping)
    _require_close("visible aperture", 2.0 * float(case["gap_half"][0]), VISIBLE_APERTURE_M)
    if tuple(float(v) for v in case.get("source", ())) != SOURCE_POSE:
        raise ValueError("teacher source pose mismatch")
    if tuple(float(v) for v in case.get("goal", ())) != GOAL_POSE:
        raise ValueError("teacher goal pose mismatch")
    if list(case.get("wall_x", ())) != [0.0, 8.0]:
        raise ValueError("teacher must contain one real wall and the dummy wall")
    for audit_name, audit in (
        ("shortcut", shortcut.get("official_continuous_audit")),
        ("rollout", rollout.get("official_continuous_audit")),
    ):
        if not isinstance(audit, Mapping) or audit.get("passed") is not True:
            raise ValueError(f"teacher {audit_name} continuous exact-SAT audit did not pass")
    for metric in (
        "terminal_position_error_m",
        "terminal_angle_error_rad",
        "terminal_speed_m_per_s",
        "terminal_omega_rad_per_s",
    ):
        if float(rollout.get(metric, math.inf)) > 1e-5:
            raise ValueError(f"teacher terminal gate failed: {metric}")

    cfg = single_slit_env_config(device=device)
    dynamics_fields = (
        "n_agents",
        "handle_layout",
        "payload_shape",
        "dt",
        "mass",
        "inertia",
        "linear_damping",
        "angular_damping",
        "max_force",
        "world_x",
        "world_y",
        "wall_slab",
        "collision_substeps",
    )
    expected_env = asdict(cfg)
    for name in dynamics_fields:
        if env_record.get(name) != expected_env[name]:
            raise ValueError(
                f"teacher dynamics field {name!r}={env_record.get(name)!r}, "
                f"training requires {expected_env[name]!r}"
            )

    artifact_path = _resolve_artifact(manifest_path, manifest.get("rollout_artifact"))
    with np.load(artifact_path) as arrays:
        required = {
            "states",
            "forces",
            "realized_wrenches",
            "swept_clearances",
        }
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"teacher artifact omitted arrays: {sorted(missing)}")
        states = torch.as_tensor(np.asarray(arrays["states"]), dtype=torch.float32, device=device)
        forces = torch.as_tensor(np.asarray(arrays["forces"]), dtype=torch.float32, device=device)
        realized = torch.as_tensor(
            np.asarray(arrays["realized_wrenches"]), dtype=torch.float32, device=device
        )
        saved_clearance = torch.as_tensor(
            np.asarray(arrays["swept_clearances"]), dtype=torch.float32, device=device
        )
    horizon = int(rollout.get("horizon", -1))
    if states.shape != (horizon + 1, 6):
        raise ValueError("teacher states must have shape [H+1,6]")
    if forces.shape != (horizon, cfg.n_agents, 2):
        raise ValueError("teacher forces must have shape [H,12,2]")
    if realized.shape != (horizon, 3) or saved_clearance.shape != (horizon,):
        raise ValueError("teacher wrench/clearance arrays are misaligned")
    if not all(bool(torch.isfinite(value).all()) for value in (states, forces, realized, saved_clearance)):
        raise ValueError("teacher artifact contains non-finite values")
    if bool((torch.linalg.vector_norm(forces, dim=-1) > cfg.max_force + 1e-5).any()):
        raise ValueError("teacher artifact violates the per-handle force bound")

    predicted = step_dynamics(states[:-1], forces, cfg)
    if not torch.allclose(predicted, states[1:], rtol=0.0, atol=1e-6):
        raise ValueError("teacher states are not generated by the benchmark dynamics")
    scenario = single_slit_scenario(horizon, cfg, initial_state=states[:-1])
    actual_clearance = transition_clearance_margin(states[:-1], states[1:], scenario, cfg)
    collision = transition_collision_mask(states[:-1], states[1:], scenario, cfg)
    if bool(collision.any()) or bool((actual_clearance <= 0.0).any()):
        raise ValueError("teacher artifact contains an unsafe SATx4 transition")
    if not torch.allclose(actual_clearance, saved_clearance, rtol=0.0, atol=1e-6):
        raise ValueError("teacher saved clearances disagree with the benchmark authority")
    if float(actual_clearance.amin()) + 1e-7 < float(rollout["minimum_swept_clearance_m"]):
        raise ValueError("teacher declared minimum clearance is inconsistent")

    target_section = _semantic_targets(states, realized, manifest)
    return SingleSlitTeacherTrajectory(
        manifest_path=manifest_path.resolve(),
        artifact_path=artifact_path,
        manifest_sha256=_sha256_file(manifest_path),
        artifact_sha256=_sha256_file(artifact_path),
        states=states,
        forces=forces,
        target_section=target_section,
        transition_clearance=actual_clearance,
        realized_wrenches=realized,
        manifest=manifest,
    )


@dataclass(frozen=True)
class RecurrentWindowProtocol:
    burn_in_steps: int = 6
    supervised_steps: int = 16
    source_anchor_fraction: float = 0.25

    @property
    def sequence_length(self) -> int:
        return self.burn_in_steps + self.supervised_steps

    def validate(self) -> None:
        if self.burn_in_steps <= 0 or self.supervised_steps < 2:
            raise ValueError("recurrent BC requires positive burn-in and >=2 supervised ticks")
        if not 0.0 <= self.source_anchor_fraction <= 1.0:
            raise ValueError("source anchor fraction must lie in [0,1]")


@dataclass(frozen=True)
class SingleSlitBCBatch:
    scenario: PianoBatch
    sequence_state: Tensor
    sequence_target_forces: Tensor
    sequence_target_section: Tensor
    sequence_transition_clearance: Tensor
    sequence_valid_mask: Tensor
    sequence_context_mask: Tensor
    sequence_supervision_mask: Tensor
    supervised_start: Tensor
    selected_stratum: Tensor
    protocol: RecurrentWindowProtocol
    initial_hidden: Tensor | None = None
    # Optional audited importance weights in the internal time-major layout.
    # None is exactly equivalent to ones on supervised transitions.
    supervision_weight: Tensor | None = None
    selected_record_index: Tensor | None = None


@torch.no_grad()
def sample_recurrent_windows(
    teacher: SingleSlitTeacherTrajectory,
    cfg,
    *,
    batch_size: int,
    protocol: RecurrentWindowProtocol | None = None,
    generator: torch.Generator | None = None,
) -> SingleSlitBCBatch:
    """Sample source-anchored and mid-route contiguous recurrent windows."""

    options = protocol or RecurrentWindowProtocol()
    options.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if teacher.horizon < options.supervised_steps:
        raise ValueError("teacher trajectory is shorter than the supervised window")
    device = teacher.states.device
    maximum_start = teacher.horizon - options.supervised_steps
    start = torch.randint(
        maximum_start + 1,
        (batch_size,),
        device=device,
        generator=generator,
    )
    source_count = int(math.floor(batch_size * options.source_anchor_fraction + 0.5))
    if source_count:
        start[:source_count] = 0
    relative = torch.arange(options.sequence_length, device=device) - options.burn_in_steps
    index = start[None, :] + relative[:, None]
    valid = index >= 0
    clamped = index.clamp(0, teacher.horizon - 1)
    states = teacher.states.index_select(0, clamped.reshape(-1)).view(
        options.sequence_length, batch_size, 6
    )
    forces = teacher.forces.index_select(0, clamped.reshape(-1)).view(
        options.sequence_length, batch_size, cfg.n_agents, 2
    )
    sections = teacher.target_section.index_select(0, clamped.reshape(-1)).view(
        options.sequence_length, batch_size, SEMANTIC_WIDTH
    )
    clearance = teacher.transition_clearance.index_select(0, clamped.reshape(-1)).view(
        options.sequence_length, batch_size
    )
    tick = torch.arange(options.sequence_length, device=device)[:, None]
    context = valid & (tick < options.burn_in_steps)
    supervision = valid & (tick >= options.burn_in_steps)
    scenario = single_slit_scenario(batch_size, cfg, initial_state=states[0])
    return SingleSlitBCBatch(
        scenario=scenario,
        sequence_state=states,
        sequence_target_forces=forces,
        sequence_target_section=sections,
        sequence_transition_clearance=clearance,
        sequence_valid_mask=valid,
        sequence_context_mask=context,
        sequence_supervision_mask=supervision,
        supervised_start=start,
        selected_stratum=torch.full(
            (batch_size,), -1, dtype=torch.int8, device=device
        ),
        protocol=options,
    )


@torch.no_grad()
def sample_balanced_corpus_batch(
    corpus: SingleSlitCorpus,
    cfg,
    *,
    batch_size: int,
    protocol: RecurrentWindowProtocol | None = None,
    generator: torch.Generator | None = None,
) -> SingleSlitBCBatch:
    """Adapt balanced corpus rows from ``[B,T]`` to the trainer's ``[T,B]`` API.

    Sampling is with replacement inside every stratum.  Counts differ by at
    most one when ``batch_size`` is not divisible by four; the remainder is
    rotated randomly before the final row shuffle, avoiding a permanent
    nominal-stratum bias.
    """

    options = protocol or RecurrentWindowProtocol()
    options.validate()
    if batch_size < len(STRATUM_NAMES):
        raise ValueError("balanced corpus batches must include at least one row per stratum")
    rows, time = corpus.state.shape[:2]
    if time != options.sequence_length:
        raise ValueError(
            f"corpus time={time} does not match recurrent protocol "
            f"time={options.sequence_length}"
        )
    if options.burn_in_steps != 6:
        raise ValueError("the corrected corpus has exactly six real context ticks")
    device = corpus.state.device
    strata = corpus.stratum[:, 0].long()
    base, remainder = divmod(batch_size, len(STRATUM_NAMES))
    order = torch.randperm(len(STRATUM_NAMES), device=device, generator=generator)
    extra = set(int(value) for value in order[:remainder].detach().cpu().tolist())
    selected_parts: list[Tensor] = []
    for stratum in range(len(STRATUM_NAMES)):
        pool = torch.where(strata == stratum)[0]
        if pool.numel() == 0:
            raise ValueError(f"corpus contains no rows for stratum {STRATUM_NAMES[stratum]!r}")
        count = base + int(stratum in extra)
        if count:
            choice = torch.randint(
                int(pool.numel()),
                (count,),
                device=device,
                generator=generator,
            )
            selected_parts.append(pool.index_select(0, choice))
    selected = torch.cat(selected_parts)
    shuffle = torch.randperm(selected.numel(), device=device, generator=generator)
    selected = selected.index_select(0, shuffle)
    if selected.shape != (batch_size,) or bool((selected < 0).any()) or bool((selected >= rows).any()):
        raise RuntimeError("balanced corpus sampler produced invalid row indices")

    state = corpus.state.index_select(0, selected).transpose(0, 1).contiguous()
    forces = corpus.forces.index_select(0, selected).transpose(0, 1).contiguous()
    section = corpus.semantic8.index_select(0, selected).transpose(0, 1).contiguous()
    clearance = corpus.clearance.index_select(0, selected).transpose(0, 1).contiguous()
    valid = torch.ones(time, batch_size, dtype=torch.bool, device=device)
    tick = torch.arange(time, device=device)[:, None]
    context = tick < options.burn_in_steps
    context = context.expand(-1, batch_size).clone()
    supervision = ~context
    scenario = single_slit_scenario(batch_size, cfg, initial_state=state[0])
    return SingleSlitBCBatch(
        scenario=scenario,
        sequence_state=state,
        sequence_target_forces=forces,
        sequence_target_section=section,
        sequence_transition_clearance=clearance,
        sequence_valid_mask=valid,
        sequence_context_mask=context,
        sequence_supervision_mask=supervision,
        # Preserve the actual nominal-path anchor as the batch's route index.
        supervised_start=corpus.anchor_index.index_select(0, selected),
        selected_stratum=corpus.stratum.index_select(0, selected)[:, 0].clone(),
        protocol=options,
    )


class MatchedSectionHead(nn.Module):
    """Give non-sheaf controls the same semantic-8 auxiliary supervision."""

    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy
        self.semantic_head = nn.Linear(policy.arm_cfg.hidden_dim, SEMANTIC_WIDTH)
        self.arm_name = policy.arm_name

    @property
    def env_cfg(self):
        return self.policy.env_cfg

    @property
    def arm_cfg(self):
        return self.policy.arm_cfg

    def init_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.policy.init_hidden(batch_size, device)

    def communication_rounds(self) -> int:
        return self.policy.communication_rounds()

    def communication_floats_per_directed_edge_round(self) -> int:
        return self.policy.communication_floats_per_directed_edge_round()

    def forward(self, obs: Tensor, edge_index: Tensor, hidden: Tensor | None = None) -> PolicyOutput:
        output = self.policy(obs, edge_index, hidden)
        # Temporal capacity controls append an audited local FIFO after the
        # same 96-D recurrent state.  The auxiliary control head remains
        # exactly the legacy Linear(96,8), so the FIFO is neither transmitted
        # nor granted a separate semantic decoder.
        semantic_input = output.hidden[..., : self.policy.arm_cfg.hidden_dim]
        return PolicyOutput(
            forces=output.forces,
            hidden=output.hidden,
            aux={**output.aux, "section": self.semantic_head(semantic_input)},
        )


SingleSlitPolicy = BasePolicy | MatchedSectionHead


class TensorOnlyPolicyTick(nn.Module):
    """Static-shape policy adapter with cached graph topology and tensor outputs."""

    def __init__(self, policy: SingleSlitPolicy, edge_index: Tensor):
        super().__init__()
        self.policy = policy
        self.register_buffer("edge_index", edge_index.detach().clone(), persistent=False)
        topology = homogeneous_topology(
            edge_index,
            1,
            policy.env_cfg.n_agents,
            device=edge_index.device,
        )
        self.register_buffer(
            "topology_incidence_vertex", topology.incidence_vertex, persistent=False
        )
        self.register_buffer(
            "topology_reverse_incidence", topology.reverse_incidence, persistent=False
        )
        self.register_buffer(
            "topology_flat_incidence_index", topology.flat_incidence_index, persistent=False
        )
        self.register_buffer(
            "topology_valid_incidence", topology.valid_incidence, persistent=False
        )

    def _topology(self) -> GBPTopology:
        return GBPTopology(
            incidence_vertex=self.topology_incidence_vertex,
            reverse_incidence=self.topology_reverse_incidence,
            flat_incidence_index=self.topology_flat_incidence_index,
            valid_incidence=self.topology_valid_incidence,
        )

    def forward(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(self.policy, SheafGBPPolicy):
            output = self.policy(
                obs,
                self.edge_index,
                hidden,
                topology=self._topology(),
            )
        else:
            output = self.policy(obs, self.edge_index, hidden)
        section = output.aux.get("section")
        if not isinstance(section, Tensor):
            raise RuntimeError("tensor-only policy tick requires a semantic section")
        return output.forces, output.hidden, section


@dataclass(frozen=True)
class PreparedFullHorizon:
    observations: Tensor
    tick: nn.Module
    compiled: bool
    compile_mode: str | None
    compile_forward_max_abs_error: float | None = None


@torch.no_grad()
def prepare_full_horizon(
    policy: SingleSlitPolicy,
    teacher: SingleSlitTeacherTrajectory,
    cfg,
    edge_index: Tensor,
    *,
    compile_tick: bool = False,
    compile_mode: str = "reduce-overhead",
) -> PreparedFullHorizon:
    """Precompute immutable observations/topology and optionally compile one tick."""

    scenario = single_slit_scenario(1, cfg, initial_state=teacher.states[:1])
    observation_schema = single_slit_observation_schema_for_dim(policy.arm_cfg.obs_dim)
    observations_by_tick = torch.stack(
        [
            single_slit_policy_observations(
                teacher.states[tick : tick + 1],
                scenario,
                cfg,
                observation_schema,
            )
            for tick in range(teacher.horizon)
        ],
        dim=0,
    )
    eager_tick: nn.Module = TensorOnlyPolicyTick(policy, edge_index)
    tick_module = eager_tick
    compile_error: float | None = None
    if compile_tick:
        if not hasattr(torch, "compile"):
            raise RuntimeError("this PyTorch build does not provide torch.compile")
        tick_module = torch.compile(
            eager_tick,
            mode=compile_mode,
            fullgraph=True,
        )
        # A backend/kernel mismatch here would silently corrupt every target in
        # the 107-tick recurrent lane.  Compile the real first-tick shape and
        # refuse the optimization unless its tensor outputs match eager.
        hidden = policy.init_hidden(1, teacher.states.device)
        eager_outputs = eager_tick(observations_by_tick[0], hidden)
        compiled_outputs = tick_module(observations_by_tick[0], hidden)
        compile_error = max(
            float((eager.detach() - compiled.detach()).abs().max().cpu())
            for eager, compiled in zip(eager_outputs, compiled_outputs, strict=True)
        )
        tolerance = 2e-5
        if compile_error > tolerance:
            raise RuntimeError(
                "compiled full-horizon tick failed eager parity: "
                f"max_abs_error={compile_error:.6g} > {tolerance:.6g}"
            )
    return PreparedFullHorizon(
        observations=observations_by_tick,
        tick=tick_module,
        compiled=compile_tick,
        compile_mode=compile_mode if compile_tick else None,
        compile_forward_max_abs_error=compile_error,
    )


def build_policy(
    arm: str,
    cfg,
    *,
    observation_schema: str = SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA,
    correction_sheaf: bool = False,
    correction_public_unary_residual: bool = False,
    correction_public_unary_residual_scale: float = 0.25,
    correction_source_null_recovery: bool = False,
    correction_phase_decoder: bool = False,
    correction_clearance_bank_decoder: bool = False,
    correction_approach_bank_decoder: bool = False,
    persistent_recovery_state: bool = False,
    persistent_recovery_dim: int = 8,
    persistent_recovery_phase_count: int = 4,
    persistent_recovery_injection: str = "shared_hidden",
    collision_role_factors: bool = False,
    collision_role_count: int = 3,
    collision_role_mean_scale: float = 0.25,
    collision_role_log_precision_scale: float = 1.5,
    collision_role_edge_log_precision_scale: float = 1.5,
    collision_role_precontact_leader_follower: bool = False,
    collision_role_separate_encoder: bool = False,
    local_se2_planner: bool = False,
    local_se2_plan_dim: int = 8,
    local_se2_planner_hidden_dim: int = 96,
    local_se2_shape_signature: bool = False,
    local_se2_shape_signature_dim: int = 8,
    local_se2_markov_shape_planner: bool = False,
    local_se2_markov_shape_hidden_dim: int = 256,
    local_se2_markov_shape_layers: int = 3,
    local_se2_geometry_refinement_adapter: bool = False,
    local_se2_geometry_refinement_hidden_dim: int = 128,
    local_se2_geometry_refinement_scale: float = 1.0,
    local_se2_geometry_refinement_canonical_features: bool = False,
    local_se2_geometry_mode_adapter: bool = False,
    local_se2_geometry_mode_count: int = 4,
    local_se2_geometry_mode_hidden_dim: int = 256,
    local_se2_geometry_mode_scale: float = 1.0,
    local_se2_geometry_mode_post_contact_only: bool = False,
    local_se2_path_memory: bool = False,
    local_se2_path_memory_shape_count: int = 24,
    local_se2_path_memory_max_steps: int = 256,
    local_se2_path_memory_ridge_decoder: bool = False,
    local_se2_path_memory_ridge_points: int = 224,
    local_se2_path_memory_ridge_alpha: float = 3.0e-4,
    local_se2_path_memory_ridge_trust_margin: float = 0.05,
    local_se2_path_memory_canonical_family_mse_threshold: float = 0.10,
    local_se2_path_memory_population_resampling: bool = False,
    local_se2_path_memory_population_resampling_points: int = 256,
    local_se2_path_memory_population_preemptive_counts: tuple[int, ...] = (10,),
    local_se2_path_memory_canonical_preemptive_counts: tuple[int, ...] = (
        11,
        13,
        15,
        16,
        17,
        20,
    ),
    local_se2_path_memory_population_recovery_counts: tuple[int, ...] = (
        11,
        13,
        15,
        17,
        20,
    ),
    local_se2_recovery_adapter: bool = False,
    local_se2_recovery_adapter_hidden_dim: int = 96,
    local_se2_recovery_adapter_scale: float = 1.0,
    local_se2_recovery_canonical_features: bool = False,
    local_se2_contact_rebel_adapter: bool = False,
    local_se2_contact_rebel_hidden_dim: int = 64,
    local_se2_contact_rebel_scale: float = 1.0,
    local_se2_contact_rebel_min_center_x_m: float = 0.0,
    local_se2_exit_refinement_adapter: bool = False,
    local_se2_exit_refinement_hidden_dim: int = 64,
    local_se2_exit_refinement_scale: float = 1.0,
    local_se2_exit_refinement_min_center_x_m: float = 0.17,
    local_se2_approach_refinement_adapter: bool = False,
    local_se2_approach_refinement_hidden_dim: int = 64,
    local_se2_approach_refinement_scale: float = 1.0,
    local_se2_approach_refinement_min_center_x_m: float = -0.30,
    local_se2_approach_refinement_stuck_steps: int = 9,
    local_se2_deep_stuck_refinement_adapter: bool = False,
    local_se2_deep_stuck_refinement_hidden_dim: int = 64,
    local_se2_deep_stuck_refinement_scale: float = 1.0,
    local_se2_deep_stuck_refinement_min_center_x_m: float = -0.30,
    local_se2_deep_stuck_refinement_stuck_steps: int = 10,
    local_se2_retry_refinement_thresholds: tuple[int, ...] = (),
    local_se2_retry_refinement_hidden_dim: int = 64,
    local_se2_retry_refinement_scale: float = 1.0,
    local_se2_retry_refinement_min_center_x_m: float = -0.30,
    local_se2_analytic_phase_latch: bool = False,
    local_se2_reentry_center_x_threshold_m: float = -0.16,
    local_se2_phase_latch_logit: float = 8.0,
    local_se2_direct_wrench_unary: bool = False,
    local_se2_direct_wrench_hidden_dim: int = 128,
    local_se2_analytic_total_wrench: bool = False,
    local_se2_analytic_unary_precision: float = 1.0,
    local_se2_geometry_evidence_precision_boost: float = 1.0,
    local_se2_analytic_edge_sigma2: float = 1.0e-5,
    local_se2_analytic_solver_eps: float = 1.0e-10,
    local_se2_analytic_max_precision: float = 1.0e9,
    local_se2_distributed_wrench_control: bool = False,
    local_se2_distributed_next_se2_control: bool = False,
    local_se2_distributed_force_hidden_dim: int = 128,
    local_se2_distributed_recovery_adapter: bool = False,
    local_se2_distributed_recovery_hidden_dim: int = 128,
    local_se2_distributed_recovery_layers: int = 2,
    local_se2_distributed_recovery_mode_count: int = 1,
    local_se2_distributed_recovery_scale: float = 1.0,
    local_se2_distributed_recovery_local_support: bool = False,
    local_se2_distributed_recovery_wall_range_m: float = 0.45,
    local_se2_distributed_grasp_allreduce: bool = False,
    local_se2_terminal_attractor: bool = False,
    local_se2_terminal_position_gain: float = 2.0,
    local_se2_terminal_angle_gain: float = 2.0,
    local_se2_terminal_max_speed: float = 0.35,
    local_se2_terminal_max_omega: float = 0.8,
    local_se2_feasible_corridor_attractor: bool = False,
    local_se2_route_free_geometry_planner: bool = False,
    local_se2_corridor_reserve_m: float = 0.015,
    local_se2_corridor_max_speed: float = 0.50,
) -> SingleSlitPolicy:
    """Build one fresh arm under an explicit, shared observation contract."""

    if arm not in SUPPORTED_ARMS:
        raise ValueError(f"unsupported single-slit arm: {arm!r}")
    if observation_schema not in (
        *SINGLE_SLIT_OBSERVATION_SCHEMAS,
        SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
    ):
        raise ValueError(f"unsupported observation schema: {observation_schema!r}")
    base_arm_cfg = single_slit_arm_config(
        cfg,
        observation_schema=(
            SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA
            if observation_schema == SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1
            else observation_schema
        ),
    )
    if observation_schema == SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1:
        base_arm_cfg = replace(
            base_arm_cfg,
            obs_dim=SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1,
        )
    arm_cfg = replace(
        base_arm_cfg,
        comm_rounds=4,
        gbp_steps=4,
        correction_sheaf=correction_sheaf,
        correction_public_unary_residual=correction_public_unary_residual,
        correction_public_unary_residual_scale=correction_public_unary_residual_scale,
        correction_source_null_recovery=correction_source_null_recovery,
        correction_phase_decoder=correction_phase_decoder,
        correction_clearance_bank_decoder=correction_clearance_bank_decoder,
        correction_approach_bank_decoder=correction_approach_bank_decoder,
        persistent_recovery_state=persistent_recovery_state,
        persistent_recovery_dim=persistent_recovery_dim,
        persistent_recovery_phase_count=persistent_recovery_phase_count,
        persistent_recovery_injection=persistent_recovery_injection,
        collision_role_factors=collision_role_factors,
        collision_role_count=collision_role_count,
        collision_role_mean_scale=collision_role_mean_scale,
        collision_role_log_precision_scale=collision_role_log_precision_scale,
        collision_role_edge_log_precision_scale=(
            collision_role_edge_log_precision_scale
        ),
        collision_role_precontact_leader_follower=(
            collision_role_precontact_leader_follower
        ),
        collision_role_separate_encoder=collision_role_separate_encoder,
        local_se2_planner=local_se2_planner,
        local_se2_plan_dim=local_se2_plan_dim,
        local_se2_planner_hidden_dim=local_se2_planner_hidden_dim,
        local_se2_shape_signature=local_se2_shape_signature,
        local_se2_shape_signature_dim=local_se2_shape_signature_dim,
        local_se2_markov_shape_planner=local_se2_markov_shape_planner,
        local_se2_markov_shape_hidden_dim=local_se2_markov_shape_hidden_dim,
        local_se2_markov_shape_layers=local_se2_markov_shape_layers,
        local_se2_geometry_refinement_adapter=(
            local_se2_geometry_refinement_adapter
        ),
        local_se2_geometry_refinement_hidden_dim=(
            local_se2_geometry_refinement_hidden_dim
        ),
        local_se2_geometry_refinement_scale=(
            local_se2_geometry_refinement_scale
        ),
        local_se2_geometry_refinement_canonical_features=(
            local_se2_geometry_refinement_canonical_features
        ),
        local_se2_geometry_mode_adapter=local_se2_geometry_mode_adapter,
        local_se2_geometry_mode_count=local_se2_geometry_mode_count,
        local_se2_geometry_mode_hidden_dim=local_se2_geometry_mode_hidden_dim,
        local_se2_geometry_mode_scale=local_se2_geometry_mode_scale,
        local_se2_geometry_mode_post_contact_only=(
            local_se2_geometry_mode_post_contact_only
        ),
        local_se2_path_memory=local_se2_path_memory,
        local_se2_path_memory_shape_count=local_se2_path_memory_shape_count,
        local_se2_path_memory_max_steps=local_se2_path_memory_max_steps,
        local_se2_path_memory_ridge_decoder=(
            local_se2_path_memory_ridge_decoder
        ),
        local_se2_path_memory_ridge_points=local_se2_path_memory_ridge_points,
        local_se2_path_memory_ridge_alpha=local_se2_path_memory_ridge_alpha,
        local_se2_path_memory_ridge_trust_margin=(
            local_se2_path_memory_ridge_trust_margin
        ),
        local_se2_path_memory_canonical_family_mse_threshold=(
            local_se2_path_memory_canonical_family_mse_threshold
        ),
        local_se2_path_memory_population_resampling=(
            local_se2_path_memory_population_resampling
        ),
        local_se2_path_memory_population_resampling_points=(
            local_se2_path_memory_population_resampling_points
        ),
        local_se2_path_memory_population_preemptive_counts=(
            local_se2_path_memory_population_preemptive_counts
        ),
        local_se2_path_memory_canonical_preemptive_counts=(
            local_se2_path_memory_canonical_preemptive_counts
        ),
        local_se2_path_memory_population_recovery_counts=(
            local_se2_path_memory_population_recovery_counts
        ),
        local_se2_recovery_adapter=local_se2_recovery_adapter,
        local_se2_recovery_adapter_hidden_dim=(
            local_se2_recovery_adapter_hidden_dim
        ),
        local_se2_recovery_adapter_scale=local_se2_recovery_adapter_scale,
        local_se2_recovery_canonical_features=(
            local_se2_recovery_canonical_features
        ),
        local_se2_contact_rebel_adapter=local_se2_contact_rebel_adapter,
        local_se2_contact_rebel_hidden_dim=local_se2_contact_rebel_hidden_dim,
        local_se2_contact_rebel_scale=local_se2_contact_rebel_scale,
        local_se2_contact_rebel_min_center_x_m=(
            local_se2_contact_rebel_min_center_x_m
        ),
        local_se2_exit_refinement_adapter=local_se2_exit_refinement_adapter,
        local_se2_exit_refinement_hidden_dim=(
            local_se2_exit_refinement_hidden_dim
        ),
        local_se2_exit_refinement_scale=local_se2_exit_refinement_scale,
        local_se2_exit_refinement_min_center_x_m=(
            local_se2_exit_refinement_min_center_x_m
        ),
        local_se2_approach_refinement_adapter=(
            local_se2_approach_refinement_adapter
        ),
        local_se2_approach_refinement_hidden_dim=(
            local_se2_approach_refinement_hidden_dim
        ),
        local_se2_approach_refinement_scale=local_se2_approach_refinement_scale,
        local_se2_approach_refinement_min_center_x_m=(
            local_se2_approach_refinement_min_center_x_m
        ),
        local_se2_approach_refinement_stuck_steps=(
            local_se2_approach_refinement_stuck_steps
        ),
        local_se2_deep_stuck_refinement_adapter=(
            local_se2_deep_stuck_refinement_adapter
        ),
        local_se2_deep_stuck_refinement_hidden_dim=(
            local_se2_deep_stuck_refinement_hidden_dim
        ),
        local_se2_deep_stuck_refinement_scale=(
            local_se2_deep_stuck_refinement_scale
        ),
        local_se2_deep_stuck_refinement_min_center_x_m=(
            local_se2_deep_stuck_refinement_min_center_x_m
        ),
        local_se2_deep_stuck_refinement_stuck_steps=(
            local_se2_deep_stuck_refinement_stuck_steps
        ),
        local_se2_retry_refinement_thresholds=tuple(
            local_se2_retry_refinement_thresholds
        ),
        local_se2_retry_refinement_hidden_dim=(
            local_se2_retry_refinement_hidden_dim
        ),
        local_se2_retry_refinement_scale=local_se2_retry_refinement_scale,
        local_se2_retry_refinement_min_center_x_m=(
            local_se2_retry_refinement_min_center_x_m
        ),
        local_se2_analytic_phase_latch=local_se2_analytic_phase_latch,
        local_se2_reentry_center_x_threshold_m=(
            local_se2_reentry_center_x_threshold_m
        ),
        local_se2_phase_latch_logit=local_se2_phase_latch_logit,
        local_se2_direct_wrench_unary=local_se2_direct_wrench_unary,
        local_se2_direct_wrench_hidden_dim=local_se2_direct_wrench_hidden_dim,
        local_se2_analytic_total_wrench=local_se2_analytic_total_wrench,
        local_se2_analytic_unary_precision=local_se2_analytic_unary_precision,
        local_se2_geometry_evidence_precision_boost=(
            local_se2_geometry_evidence_precision_boost
        ),
        local_se2_analytic_edge_sigma2=local_se2_analytic_edge_sigma2,
        local_se2_analytic_solver_eps=local_se2_analytic_solver_eps,
        local_se2_analytic_max_precision=local_se2_analytic_max_precision,
        local_se2_distributed_wrench_control=(
            local_se2_distributed_wrench_control
        ),
        local_se2_distributed_next_se2_control=(
            local_se2_distributed_next_se2_control
        ),
        local_se2_distributed_force_hidden_dim=(
            local_se2_distributed_force_hidden_dim
        ),
        local_se2_distributed_recovery_adapter=(
            local_se2_distributed_recovery_adapter
        ),
        local_se2_distributed_recovery_hidden_dim=(
            local_se2_distributed_recovery_hidden_dim
        ),
        local_se2_distributed_recovery_layers=(
            local_se2_distributed_recovery_layers
        ),
        local_se2_distributed_recovery_mode_count=(
            local_se2_distributed_recovery_mode_count
        ),
        local_se2_distributed_recovery_scale=(
            local_se2_distributed_recovery_scale
        ),
        local_se2_distributed_recovery_local_support=(
            local_se2_distributed_recovery_local_support
        ),
        local_se2_distributed_recovery_wall_range_m=(
            local_se2_distributed_recovery_wall_range_m
        ),
        local_se2_distributed_grasp_allreduce=(
            local_se2_distributed_grasp_allreduce
        ),
        local_se2_terminal_attractor=local_se2_terminal_attractor,
        local_se2_terminal_position_gain=local_se2_terminal_position_gain,
        local_se2_terminal_angle_gain=local_se2_terminal_angle_gain,
        local_se2_terminal_max_speed=local_se2_terminal_max_speed,
        local_se2_terminal_max_omega=local_se2_terminal_max_omega,
        local_se2_feasible_corridor_attractor=(
            local_se2_feasible_corridor_attractor
        ),
        local_se2_route_free_geometry_planner=(
            local_se2_route_free_geometry_planner
        ),
        local_se2_corridor_reserve_m=local_se2_corridor_reserve_m,
        local_se2_corridor_max_speed=local_se2_corridor_max_speed,
    )
    if correction_sheaf and arm != "sheaf_gbp":
        raise ValueError("the source-correct correction sheaf requires arm='sheaf_gbp'")
    if arm == FAIR_RAW_FULL_ARM:
        policy = TemporalStateMatchedMPNNPolicy(
            cfg,
            arm_cfg,
            message_dim=32,
            name=arm,
            spatial_communication=True,
        ).to(cfg.device)
    elif arm == FAIR_COMM_MATCHED_ARM:
        policy = TemporalStateMatchedMPNNPolicy(
            cfg,
            arm_cfg,
            message_dim=30,
            name=arm,
            spatial_communication=True,
        ).to(cfg.device)
    elif arm == FAIR_NO_COMM_ARM:
        policy = TemporalStateMatchedMPNNPolicy(
            cfg,
            arm_cfg,
            message_dim=30,
            name=arm,
            spatial_communication=False,
        ).to(cfg.device)
    else:
        policy = make_policy(arm, cfg, arm_cfg).to(cfg.device)
    if arm != "sheaf_gbp":
        policy = MatchedSectionHead(policy).to(cfg.device)
    return policy


@dataclass(frozen=True)
class SingleSlitBCLoss:
    total: Tensor
    force_mse: Tensor
    semantic_section_mse: Tensor
    net_wrench_mse: Tensor
    supervised_transition_count: Tensor


@dataclass(frozen=True)
class DaggerFinalClearanceLoss:
    total: Tensor
    predicted_clearance_m: Tensor
    softmin_clearance_m: Tensor
    temperature_m: float
    predicted_forces: Tensor


@dataclass(frozen=True)
class FullHorizonCarriedLoss:
    """Teacher-forced loss under the deployment recurrent-state trajectory."""

    total: Tensor
    force_mse: Tensor
    semantic_section_mse: Tensor
    net_wrench_mse: Tensor
    segment_force_mse: Mapping[str, Tensor]
    segment_wrench_mse: Mapping[str, Tensor]
    nominal_predicted_transition_clearance: Tensor | None
    nominal_exact_clearance_softmin: Tensor | None
    nominal_clearance_softmin_temperature_m: float | None
    horizon: int
    chunk_steps: int


def nominal_exact_clearance_softmin_loss(
    transition_clearance: Tensor,
    *,
    temperature_m: float = NOMINAL_CLEARANCE_SOFTMIN_TEMPERATURE_M,
) -> Tensor:
    """Return a smooth worst-transition loss from exact swept clearances.

    Each input entry remains the benchmark's exact signed swept-SAT margin.
    Only the reduction across nominal ticks is softened.  Multiplication by
    ``temperature_m`` keeps the scalar in metres; projected-gradient code
    subsequently normalizes its gradient before applying the halfspace.
    """

    if transition_clearance.ndim != 1 or transition_clearance.numel() == 0:
        raise ValueError("nominal transition clearance must be a non-empty vector")
    if not math.isfinite(temperature_m) or temperature_m <= 0.0:
        raise ValueError("nominal clearance soft-min temperature must be positive")
    if not bool(torch.isfinite(transition_clearance.detach()).all()):
        raise ValueError("nominal transition clearance must be finite")
    temperature = transition_clearance.new_tensor(temperature_m)
    return temperature * torch.logsumexp(-transition_clearance / temperature, dim=0)


def normalized_net_wrench_error_per_sample(
    state: Tensor,
    predicted_forces: Tensor,
    target_forces: Tensor,
    cfg,
) -> Tensor:
    """Return world-frame ``(Fx,Fy,tau)`` MSE per sample under exact handles."""

    if state.ndim != 2 or state.shape[-1] != 6:
        raise ValueError("net-wrench states must have shape [batch,6]")
    expected = (state.shape[0], cfg.n_agents, 2)
    if predicted_forces.shape != expected or target_forces.shape != expected:
        raise ValueError(f"net-wrench forces must both have shape {expected}")
    predicted = net_wrench_from_forces(state, predicted_forces, cfg)
    target = net_wrench_from_forces(state, target_forces, cfg)
    scale = state.new_tensor(
        (
            cfg.n_agents * cfg.max_force,
            cfg.n_agents * cfg.max_force,
            0.75 * cfg.n_agents * cfg.max_force,
        )
    )
    return ((predicted - target) / scale).square().mean(dim=-1)


def recurrent_bc_loss(
    policy: SingleSlitPolicy,
    batch: SingleSlitBCBatch,
    cfg,
    edge_index: Tensor,
    *,
    semantic_section_weight: float = 0.15,
    net_wrench_weight: float = 0.0,
) -> SingleSlitBCLoss:
    """Identical no-grad burn-in and TBPTT objective for all four arms."""

    if semantic_section_weight < 0.0:
        raise ValueError("semantic section weight must be non-negative")
    if net_wrench_weight < 0.0:
        raise ValueError("net wrench weight must be non-negative")
    if bool((batch.sequence_valid_mask & (batch.sequence_transition_clearance <= 0.0)).any()):
        raise ValueError("BC batch contains a non-positive-clearance transition")
    time, batch_size = batch.sequence_valid_mask.shape
    if batch.supervision_weight is not None:
        if batch.supervision_weight.shape != (time, batch_size):
            raise ValueError("BC supervision weights must have shape [time,batch]")
        if not bool(torch.isfinite(batch.supervision_weight).all()) or bool(
            (batch.supervision_weight < 0).any()
        ):
            raise ValueError("BC supervision weights must be finite and non-negative")
        if bool((batch.supervision_weight[~batch.sequence_supervision_mask] != 0).any()):
            raise ValueError("BC burn-in/non-supervised weights must be zero")
    expected_hidden = policy.init_hidden(batch_size, batch.sequence_state.device)
    hidden = expected_hidden if batch.initial_hidden is None else batch.initial_hidden
    if hidden.shape != expected_hidden.shape:
        raise ValueError(
            f"BC initial hidden has shape {tuple(hidden.shape)}, "
            f"expected {tuple(expected_hidden.shape)}"
        )
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("BC initial hidden contains non-finite values")
    force_sum = batch.sequence_state.new_zeros(())
    semantic_sum = batch.sequence_state.new_zeros(())
    wrench_sum = batch.sequence_state.new_zeros(())
    count = batch.sequence_state.new_zeros(())
    observation_schema = single_slit_observation_schema_for_dim(policy.arm_cfg.obs_dim)
    for tick in range(time):
        obs = single_slit_policy_observations(
            batch.sequence_state[tick],
            batch.scenario,
            cfg,
            observation_schema,
        )
        if tick < batch.protocol.burn_in_steps:
            with torch.no_grad():
                output = policy(obs, edge_index, hidden)
        else:
            output = policy(obs, edge_index, hidden)
        valid = batch.sequence_valid_mask[tick]
        hidden_mask = valid.view(batch_size, *([1] * (hidden.ndim - 1)))
        hidden = torch.where(hidden_mask, output.hidden, hidden)
        supervised = batch.sequence_supervision_mask[tick]
        if bool(supervised.any()):
            section = output.aux.get("section")
            if not isinstance(section, Tensor) or section.shape[:2] != (
                batch_size,
                cfg.n_agents,
            ) or section.shape[-1] < SEMANTIC_WIDTH:
                raise RuntimeError("policy omitted its matched semantic-8 output")
            force_error = (
                output.forces / cfg.max_force
                - batch.sequence_target_forces[tick].detach() / cfg.max_force
            ).square().mean(dim=(-2, -1))
            section_target = batch.sequence_target_section[tick, :, None, :].expand(
                -1, cfg.n_agents, -1
            )
            semantic_error = (
                section[..., :SEMANTIC_WIDTH] - section_target.detach()
            ).square().mean(dim=(-2, -1))
            wrench_error = normalized_net_wrench_error_per_sample(
                batch.sequence_state[tick],
                output.forces,
                batch.sequence_target_forces[tick].detach(),
                cfg,
            )
            weight = supervised.to(force_error.dtype)
            if batch.supervision_weight is not None:
                weight = weight * batch.supervision_weight[tick].to(force_error.dtype)
            force_sum = force_sum + (force_error * weight).sum()
            semantic_sum = semantic_sum + (semantic_error * weight).sum()
            wrench_sum = wrench_sum + (wrench_error * weight).sum()
            count = count + weight.sum()
    if float(count.detach().cpu()) <= 0.0:
        raise RuntimeError("BC batch contains no supervised transition")
    force_mse = force_sum / count
    semantic_mse = semantic_sum / count
    wrench_mse = wrench_sum / count
    return SingleSlitBCLoss(
        total=(
            force_mse
            + semantic_section_weight * semantic_mse
            + net_wrench_weight * wrench_mse
        ),
        force_mse=force_mse,
        semantic_section_mse=semantic_mse,
        net_wrench_mse=wrench_mse,
        supervised_transition_count=count,
    )


def dagger_final_clearance_loss(
    policy: SingleSlitPolicy,
    batch: SingleSlitBCBatch,
    cfg,
    edge_index: Tensor,
    *,
    temperature_m: float = 1.0e-3,
) -> DaggerFinalClearanceLoss:
    """Differentiate the current final action through exact swept geometry.

    The recurrent prefix is replayed exactly like BC: stored collection hidden
    is the initial state, burn-in runs without gradients, and valid hidden is
    carried through every stored state.  Dataset clearances are audit-only and
    never enter this objective.
    """

    if not math.isfinite(temperature_m) or temperature_m <= 0.0:
        raise ValueError("final-clearance temperature must be finite and positive")
    time, batch_size = batch.sequence_valid_mask.shape
    expected = policy.init_hidden(batch_size, batch.sequence_state.device)
    hidden = expected if batch.initial_hidden is None else batch.initial_hidden
    if hidden.shape != expected.shape:
        raise ValueError("final-clearance initial hidden shape mismatch")
    observation_schema = single_slit_observation_schema_for_dim(policy.arm_cfg.obs_dim)
    final_output = None
    for tick in range(time):
        obs = single_slit_policy_observations(
            batch.sequence_state[tick], batch.scenario, cfg, observation_schema
        )
        if tick < batch.protocol.burn_in_steps:
            with torch.no_grad():
                output = policy(obs, edge_index, hidden)
        else:
            output = policy(obs, edge_index, hidden)
        valid = batch.sequence_valid_mask[tick]
        hidden_mask = valid.view(batch_size, *([1] * (hidden.ndim - 1)))
        hidden = torch.where(hidden_mask, output.hidden, hidden)
        if tick == time - 1:
            final_output = output
    if final_output is None or not bool(batch.sequence_valid_mask[-1].all()):
        raise ValueError("final-clearance requires a valid final state in every row")
    final_state = batch.sequence_state[-1]
    proposed = step_dynamics(final_state, final_output.forces, cfg)
    clearance = transition_clearance_margin(
        final_state, proposed, batch.scenario, cfg
    )
    softmin = -temperature_m * (
        torch.logsumexp(-clearance / temperature_m, dim=0)
        - math.log(batch_size)
    )
    return DaggerFinalClearanceLoss(
        total=-softmin / 1.0e-3,
        predicted_clearance_m=clearance,
        softmin_clearance_m=softmin,
        temperature_m=temperature_m,
        predicted_forces=final_output.forces,
    )


def full_horizon_carried_bc_loss(
    policy: SingleSlitPolicy,
    teacher: SingleSlitTeacherTrajectory,
    cfg,
    edge_index: Tensor,
    *,
    chunk_steps: int = 16,
    semantic_section_weight: float = 0.15,
    net_wrench_weight: float = 0.0,
    nominal_clearance_softmin_temperature_m: float | None = None,
    prepared: PreparedFullHorizon | None = None,
) -> FullHorizonCarriedLoss:
    """Run the full nominal path with zero-initialized, continuously carried hidden state.

    ``chunk_steps`` truncates gradients, not state.  At each boundary the
    exact hidden tensor value is detached and carried into the next tick; it
    is never reinitialized.  Consequently forward inference is invariant to
    chunk length and matches the recurrent-state age seen in deployment.
    """

    if chunk_steps <= 0:
        raise ValueError("full-horizon TBPTT chunk length must be positive")
    if semantic_section_weight < 0.0:
        raise ValueError("semantic section weight must be non-negative")
    if net_wrench_weight < 0.0:
        raise ValueError("net wrench weight must be non-negative")
    if nominal_clearance_softmin_temperature_m is not None and (
        not math.isfinite(nominal_clearance_softmin_temperature_m)
        or nominal_clearance_softmin_temperature_m <= 0.0
    ):
        raise ValueError("nominal clearance soft-min temperature must be positive")
    horizon = teacher.horizon
    if horizon <= 0:
        raise ValueError("full-horizon teacher is empty")
    execution = (
        prepare_full_horizon(policy, teacher, cfg, edge_index)
        if prepared is None
        else prepared
    )
    if execution.observations.shape[0] != horizon:
        raise ValueError("prepared full-horizon observations have the wrong length")
    hidden = policy.init_hidden(1, teacher.states.device)
    force_errors: list[Tensor] = []
    semantic_errors: list[Tensor] = []
    wrench_errors: list[Tensor] = []
    predicted_clearances: list[Tensor] | None = (
        [] if nominal_clearance_softmin_temperature_m is not None else None
    )
    nominal_scenario = (
        single_slit_scenario(1, cfg, initial_state=teacher.states[:1])
        if predicted_clearances is not None
        else None
    )
    wrench_scale = teacher.states.new_tensor(
        (cfg.n_agents * cfg.max_force, cfg.n_agents * cfg.max_force, 0.75 * cfg.n_agents * cfg.max_force)
    )
    for tick in range(horizon):
        if tick > 0 and tick % chunk_steps == 0:
            # Preserve the exact forward value while cutting only the graph.
            hidden = hidden.detach()
        forces, hidden, section = execution.tick(execution.observations[tick], hidden)
        if section.shape[:2] != (1, cfg.n_agents):
            raise RuntimeError("policy omitted its matched semantic output")
        force_error = (
            forces / cfg.max_force
            - teacher.forces[tick : tick + 1].detach() / cfg.max_force
        ).square().mean()
        section_target = teacher.target_section[tick].view(1, 1, SEMANTIC_WIDTH)
        semantic_error = (
            section[..., :SEMANTIC_WIDTH] - section_target.detach()
        ).square().mean()
        predicted_wrench = net_wrench_from_forces(
            teacher.states[tick : tick + 1], forces, cfg
        )
        wrench_error = (
            (predicted_wrench - teacher.realized_wrenches[tick : tick + 1].detach())
            / wrench_scale
        ).square().mean()
        force_errors.append(force_error)
        semantic_errors.append(semantic_error)
        wrench_errors.append(wrench_error)
        if predicted_clearances is not None:
            assert nominal_scenario is not None
            predicted_next = step_dynamics(
                teacher.states[tick : tick + 1], forces, cfg
            )
            predicted_clearances.append(
                transition_clearance_margin(
                    teacher.states[tick : tick + 1],
                    predicted_next,
                    nominal_scenario,
                    cfg,
                )[0]
            )

    force_by_tick = torch.stack(force_errors)
    semantic_by_tick = torch.stack(semantic_errors)
    wrench_by_tick = torch.stack(wrench_errors)
    segments = {
        "prefix_0_29": (0, min(30, horizon)),
        "critical_30_69": (min(30, horizon), min(70, horizon)),
        "exit_70_end": (min(70, horizon), horizon),
    }

    def segment_means(value: Tensor) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for name, (start, stop) in segments.items():
            result[name] = value[start:stop].mean() if stop > start else value.new_zeros(())
        return result

    force_mse = force_by_tick.mean()
    semantic_mse = semantic_by_tick.mean()
    wrench_mse = wrench_by_tick.mean()
    nominal_predicted_transition_clearance = (
        torch.stack(predicted_clearances) if predicted_clearances is not None else None
    )
    nominal_exact_clearance_softmin = (
        nominal_exact_clearance_softmin_loss(
            nominal_predicted_transition_clearance,
            temperature_m=nominal_clearance_softmin_temperature_m,
        )
        if nominal_predicted_transition_clearance is not None
        and nominal_clearance_softmin_temperature_m is not None
        else None
    )
    return FullHorizonCarriedLoss(
        total=(
            force_mse
            + semantic_section_weight * semantic_mse
            + net_wrench_weight * wrench_mse
        ),
        force_mse=force_mse,
        semantic_section_mse=semantic_mse,
        net_wrench_mse=wrench_mse,
        segment_force_mse=segment_means(force_by_tick),
        segment_wrench_mse=segment_means(wrench_by_tick),
        nominal_predicted_transition_clearance=nominal_predicted_transition_clearance,
        nominal_exact_clearance_softmin=nominal_exact_clearance_softmin,
        nominal_clearance_softmin_temperature_m=(
            nominal_clearance_softmin_temperature_m
        ),
        horizon=horizon,
        chunk_steps=chunk_steps,
    )


def communication_record(
    policy: SingleSlitPolicy,
    edge_index: Tensor,
    *,
    active_ticks: int,
) -> dict[str, Any]:
    """Exact float32 payload accounting; no uncounted headers are assumed."""

    if active_ticks < 0:
        raise ValueError("active_ticks must be non-negative")
    width = int(policy.communication_floats_per_directed_edge_round())
    rounds = int(policy.communication_rounds())
    directed_edges = 2 * int(edge_index.shape[0])
    packets = active_ticks * directed_edges * rounds
    floats = packets * width
    return {
        "arm": policy.arm_name,
        "undirected_radius_edges": int(edge_index.shape[0]),
        "directed_radius_edges": directed_edges,
        "rounds_per_control_tick": rounds,
        "float32_payload_floats_per_directed_edge_round": width,
        "active_control_ticks": active_ticks,
        "directed_packets": packets,
        "payload_floats": floats,
        "payload_bytes": 4 * floats,
        "message_header_bytes_counted": 0,
    }


def architecture_record(policy: SingleSlitPolicy, edge_index: Tensor) -> dict[str, Any]:
    if policy.arm_cfg.obs_dim == SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1:
        observation_schema = SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1
        checkpoint_schema = SINGLE_SLIT_CONTACT_BC_SCHEMA_V1
    else:
        observation_schema = single_slit_observation_schema_for_dim(policy.arm_cfg.obs_dim)
        checkpoint_schema = (
            SINGLE_SLIT_BC_SCHEMA
            if observation_schema == SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA
            else LEGACY_SINGLE_SLIT_BC_SCHEMA
        )
    record = {
        "schema_version": checkpoint_schema,
        "arm": policy.arm_name,
        "observation_schema": observation_schema,
        "environment": asdict(policy.env_cfg),
        "arm_config": asdict(policy.arm_cfg),
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "semantic_auxiliary": {
            "width": SEMANTIC_WIDTH,
            "same_labels_and_weight_for_all_arms": True,
            "baseline_head_not_transmitted": policy.arm_name != "sheaf_gbp",
        },
        "communication_per_teacher_episode": communication_record(
            policy, edge_index, active_ticks=0
        ),
    }
    native = policy.policy if isinstance(policy, MatchedSectionHead) else policy
    if isinstance(native, TemporalStateMatchedMPNNPolicy):
        runtime_hidden = native.init_hidden(1, next(native.parameters()).device)
        record["fair_current_control"] = {
            "schema_version": "piano-movers-single-slit-fair-current-control-v1",
            "reference": "current correction Sheaf-GBP",
            "observation_encoder_and_gru_module_family_matched": True,
            "decoder_hidden_width_depth_and_output_matched": True,
            "temporal_state": {
                "runtime_shape_batch1": list(runtime_hidden.shape),
                "floats_per_agent": native.carried_state_floats_per_agent,
                "recurrent_floats": native.arm_cfg.hidden_dim,
                "fifo_slots": native.arm_cfg.temporal_window,
                "floats_per_fifo_slot": native.arm_cfg.stalk_dim,
                "validity_floats": native.arm_cfg.temporal_window,
                "all_occupied_slots_consumed_with_fixed_recency_weights": True,
                "newest_summary_detached_before_fifo_storage_like_sheaf": True,
            },
            "spatial_communication": {
                "enabled": native.spatial_communication,
                "explicitly_masked": not native.spatial_communication,
                "latent_message_width_when_unmasked": native.message_dim,
                "payload_scalar_dtype": "float32",
                "accounted_payload_floats_per_directed_edge_round": (
                    native.communication_floats_per_directed_edge_round()
                ),
                "accounted_rounds_per_control_tick": native.communication_rounds(),
                "local_round_stack_executed": native.arm_cfg.comm_rounds,
                "masked_sender_path_retained_but_not_executed": (
                    not native.spatial_communication
                ),
            },
        }
    return record


__all__ = [
    "EXECUTABLE_TEACHER_SCHEMA",
    "FAIR_COMM_MATCHED_ARM",
    "FAIR_CONTROL_ARMS",
    "FAIR_NO_COMM_ARM",
    "FAIR_RAW_FULL_ARM",
    "FullHorizonCarriedLoss",
    "DaggerFinalClearanceLoss",
    "PreparedFullHorizon",
    "LEGACY_SINGLE_SLIT_BC_SCHEMA",
    "LEGACY_CONTROL_ARMS",
    "NOMINAL_CLEARANCE_SOFTMIN_TEMPERATURE_M",
    "RecurrentWindowProtocol",
    "SINGLE_SLIT_BC_SCHEMA",
    "SINGLE_SLIT_CONTACT_BC_SCHEMA_V1",
    "SUPPORTED_ARMS",
    "SingleSlitBCBatch",
    "SingleSlitBCLoss",
    "SingleSlitTeacherTrajectory",
    "architecture_record",
    "build_policy",
    "communication_record",
    "full_horizon_carried_bc_loss",
    "dagger_final_clearance_loss",
    "load_executable_teacher",
    "normalized_net_wrench_error_per_sample",
    "nominal_exact_clearance_softmin_loss",
    "prepare_full_horizon",
    "recurrent_bc_loss",
    "sample_balanced_corpus_batch",
    "sample_recurrent_windows",
]
