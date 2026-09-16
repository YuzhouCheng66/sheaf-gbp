"""Behavior cloning for strict, planner-certified serpentine piano maps.

The planner and :mod:`piano_movers.planner_path_teacher` are privileged data
generation tools.  They are never called by :class:`SheafGBPPolicy` during a
policy rollout and their ceiling is reported separately from learned-policy
capability.  Input maps are accepted only through self-hashed strict planner
records whose continuous swept-collision certificate passed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .config import ArmConfig, HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2, EnvConfig
from .env import (
    PianoBatch,
    collision_clearance_margin,
    collision_mask,
    observations,
    radius_edges,
    step_dynamics,
    success_mask,
    transition_clearance_margin,
    transition_collision_mask,
    wrap_angle,
)
from .models import SheafGBPPolicy
from .planner_path_teacher import (
    PlannerPathTeacherConfig,
    PlannerPathTeacherOutput,
    planner_path_teacher_output,
)
from .planner_scout_protocol import route_backtracking_statistics, verify_planner_scout_record


SERPENTINE_SCAN_REPORT_SCHEMA = "piano-movers-backtracking-geometry-scan-v2"
SERPENTINE_BC_SCHEMA = "piano-movers-serpentine-sheaf-gbp-bc-v1"
SEMANTIC_SECTION_WIDTH = 8
SEMANTIC_SECTION_WEIGHT = 0.15
SERPENTINE_UNDIRECTED_EDGES = 24
SERPENTINE_GRAPH_DIAMETER = 4
DEFAULT_BC_BURN_IN_STEPS = 6
DEFAULT_BC_SUPERVISED_STEPS = 10
DEFAULT_BC_SOURCE_ANCHOR_FRACTION = 0.25


@dataclass(frozen=True)
class StrictPlannerExample:
    """One verified map/path pair, stored as unbatched CPU tensors."""

    record_sha256: str
    state0: Tensor
    goal: Tensor
    wall_x: Tensor
    gap_y: Tensor
    gap_half: Tensor
    source: Tensor
    obstacle_rects: Tensor
    path: Tensor
    wall_slab: float


@dataclass(frozen=True)
class PathPerturbationConfig:
    """Small state noise around an exact-SAT-certified planner waypoint."""

    position_std: float = 0.025
    angle_std: float = 0.035
    linear_velocity_std: float = 0.060
    angular_velocity_std: float = 0.080

    def validate(self) -> None:
        values = (
            self.position_std,
            self.angle_std,
            self.linear_velocity_std,
            self.angular_velocity_std,
        )
        if any(value < 0.0 or not math.isfinite(value) for value in values):
            raise ValueError(
                "path perturbation standard deviations must be finite and non-negative"
            )


@dataclass(frozen=True)
class RecurrentBCProtocol:
    """Frozen semantics of one recurrent teacher-forced BC window.

    A mid-route row contains ``burn_in_steps`` real teacher/dynamics context
    transitions followed by ``supervised_steps`` transitions.  A source row
    instead contains invalid left padding during burn-in, so its first
    supervised observation is seen with genuinely empty recurrent and
    temporal state.
    """

    burn_in_steps: int = DEFAULT_BC_BURN_IN_STEPS
    supervised_steps: int = DEFAULT_BC_SUPERVISED_STEPS
    source_anchor_fraction: float = DEFAULT_BC_SOURCE_ANCHOR_FRACTION

    @property
    def sequence_length(self) -> int:
        return self.burn_in_steps + self.supervised_steps

    def validate(self) -> None:
        if self.burn_in_steps <= 0:
            raise ValueError("recurrent BC burn-in must be positive")
        if self.supervised_steps < 2:
            raise ValueError(
                "recurrent BC requires at least two supervised steps; "
                "single-step BC is forbidden"
            )
        if not math.isfinite(self.source_anchor_fraction) or not (
            0.0 <= self.source_anchor_fraction <= 1.0
        ):
            raise ValueError("source anchor fraction must be finite and in [0, 1]")


def recurrent_bc_protocol_record(protocol: RecurrentBCProtocol) -> dict[str, Any]:
    """Return the serialization-safe, fail-closed sequence BC contract."""

    protocol.validate()
    return {
        "schema": "serpentine-recurrent-teacher-forced-bc-v1",
        "burn_in_steps": protocol.burn_in_steps,
        "supervised_steps": protocol.supervised_steps,
        "sequence_length": protocol.sequence_length,
        "source_anchor_fraction": protocol.source_anchor_fraction,
        "midroute_context": (
            "real contiguous planner-teacher transitions through step_dynamics"
        ),
        "midroute_start_pool": (
            "uniform within each selected strict record's nonterminal planner "
            "waypoints after deterministic 16-slot nominal exact-SAT certification"
        ),
        "terminal_padding": (
            "success may end a certified window early; later slots remain invalid"
        ),
        "source_context": "invalid left padding; hidden remains exactly zero",
        "hidden_update": "valid-mask gated",
        "backpropagation": "full supervised unroll; burn-in under no_grad",
        "transition_authority": (
            "transition_collision_mask=false AND "
            "transition_clearance_margin>0 at every valid transition"
        ),
        "unsafe_perturbation": "reject whole window and replay nominal window",
        "teacher_path_index": "carried monotonically across every valid transition",
        "dagger_used": False,
    }


@dataclass(frozen=True)
class PlannerBCBatch:
    """A safe recurrent batch generated only from verified planner records.

    The leading fields retain the first supervised tick for the matched
    one-step baseline API.  Sequence-aware training must use the explicit
    ``sequence_*`` tensors and masks; :func:`serpentine_bc_loss` rejects a
    single-step sequence.
    """

    state: Tensor
    nominal_state: Tensor
    scenario: PianoBatch
    path: Tensor
    path_lengths: Tensor
    minimum_path_index: Tensor
    target_forces: Tensor
    target_section: Tensor
    selected_record_index: Tensor
    perturbation_accepted: Tensor
    sequence_state: Tensor
    sequence_next_state: Tensor
    sequence_nominal_state: Tensor
    sequence_minimum_path_index: Tensor
    sequence_target_forces: Tensor
    sequence_target_section: Tensor
    sequence_transition_clearance: Tensor
    sequence_transition_safe: Tensor
    sequence_valid_mask: Tensor
    sequence_context_mask: Tensor
    sequence_supervision_mask: Tensor
    source_anchored: Tensor
    candidate_pose_rejected: Tensor
    candidate_transition_rejected: Tensor
    protocol: RecurrentBCProtocol


@dataclass(frozen=True)
class BCLoss:
    total: Tensor
    force_mse: Tensor
    semantic_section_mse: Tensor
    temporal_memory_count: Tensor


def serpentine_teacher_config() -> PlannerPathTeacherConfig:
    """Return the path-tracking controls audited at 4/4 on ``train4``.

    The generic teacher defaults cut corners in the tight S corridor.  This
    slower, shorter-lookahead controller tracks the exact certified polyline;
    it is used identically for BC labels and the separately reported ceiling.
    """

    return PlannerPathTeacherConfig(
        lookahead_distance=0.05,
        maximum_linear_speed=0.25,
        maximum_angular_speed=0.6,
        slowdown_distance=0.4,
    )


def _single_row_tensor(
    payload: Mapping[str, Any],
    name: str,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    value = torch.as_tensor(payload.get(name), dtype=dtype)
    if value.shape == (1, width):
        value = value[0]
    if value.shape != (width,) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"strict planner scenario {name!r} must be finite [1, {width}]")
    return value.contiguous()


def _example_from_record(record: Mapping[str, Any]) -> StrictPlannerExample:
    scenario = record.get("scenario")
    environment = record.get("environment")
    if not isinstance(scenario, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("strict planner record is missing scenario/environment mappings")
    required_environment = {
        "n_agents": 12,
        "handle_layout": HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2,
        "payload_shape": "T",
        "collision_substeps": 4,
    }
    for name, expected in required_environment.items():
        if environment.get(name) != expected:
            raise ValueError(
                f"strict planner environment {name!r} must equal {expected!r}, "
                f"got {environment.get(name)!r}"
            )
    for name, expected in (("world_x", 3.9), ("world_y", 2.0)):
        value = environment.get(name)
        if not isinstance(value, (int, float)) or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"strict serpentine environment requires {name}={expected}")

    obstacle_rects = torch.as_tensor(scenario.get("obstacle_rects"), dtype=torch.float32)
    if obstacle_rects.shape == (1, 2, 4):
        obstacle_rects = obstacle_rects[0]
    if obstacle_rects.shape != (2, 4) or not bool(torch.isfinite(obstacle_rects).all()):
        raise ValueError("strict serpentine examples require exactly two finite AABBs")
    if bool(
        (
            (obstacle_rects[:, 0] >= obstacle_rects[:, 1])
            | (obstacle_rects[:, 2] >= obstacle_rects[:, 3])
        ).any()
    ):
        raise ValueError("strict serpentine AABB bounds must be increasing")

    path = torch.as_tensor(record.get("unrestricted_path"), dtype=torch.float32)
    # Strict records serialize collision-audited poses as full rigid states
    # [x,y,theta,vx,vy,omega].  The scout source path can also be pose-only;
    # both forms were covered by the record hash before this projection.
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] not in (3, 6):
        raise ValueError("strict planner path must have shape [waypoints>=2, 3 or 6]")
    if not bool(torch.isfinite(path).all()):
        raise ValueError("strict planner path must be finite")
    path = path[:, :3]
    wall_slab = environment.get("wall_slab")
    if not isinstance(wall_slab, (int, float)) or float(wall_slab) <= 0.0:
        raise ValueError("strict planner environment wall_slab must be positive")
    return StrictPlannerExample(
        record_sha256=str(record["record_sha256"]),
        state0=_single_row_tensor(scenario, "state0", 6),
        goal=_single_row_tensor(scenario, "goal", 3),
        wall_x=_single_row_tensor(scenario, "wall_x", 2),
        gap_y=_single_row_tensor(scenario, "gap_y", 2),
        gap_half=_single_row_tensor(scenario, "gap_half", 2),
        source=_single_row_tensor(scenario, "source", 3),
        obstacle_rects=obstacle_rects.contiguous(),
        path=path.contiguous(),
        wall_slab=float(wall_slab),
    )


def load_strict_eligible_examples(path: Path | str) -> tuple[StrictPlannerExample, ...]:
    """Load a scan report, verify every strict record, and keep only eligible rows.

    Verification occurs before filtering.  Thus changing an ineligible record
    or its eligibility bit cannot silently turn it into ignored input.
    """

    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("planner scan report must be a JSON object")
    if report.get("schema_version") != SERPENTINE_SCAN_REPORT_SCHEMA:
        raise ValueError("planner scan report schema mismatch")
    if report.get("family") != "serpentine":
        raise ValueError("serpentine BC accepts only a serpentine-family scan report")
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ValueError("planner scan report results must be a list")

    examples: list[StrictPlannerExample] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"planner scan result {row_index} must be an object")
        record = row.get("strict_protocol")
        row_eligible = bool(row.get("strict_eligible_for_training", False))
        if record is None:
            if row_eligible:
                raise ValueError("a result cannot be eligible without a strict record")
            continue
        if not isinstance(record, Mapping):
            raise ValueError("strict_protocol must be an object or null")
        verify_planner_scout_record(record)
        record_eligible = bool(record.get("eligible_for_training"))
        if row_eligible != record_eligible:
            raise ValueError("scan-row and strict-record eligibility disagree")
        if record_eligible:
            examples.append(_example_from_record(record))

    declared = report.get("strict_eligible_count")
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise ValueError("planner scan report strict_eligible_count must be an integer")
    if declared != len(examples):
        raise ValueError("planner scan report strict eligible count is inconsistent")
    if not examples:
        raise ValueError("planner scan report contains no strict eligible serpentine maps")
    return tuple(examples)


def serpentine_env_config(
    examples: Sequence[StrictPlannerExample],
    *,
    device: torch.device | str,
    max_steps: int = 550,
) -> EnvConfig:
    """Return the fixed BC/evaluation environment required by this experiment."""

    if not examples:
        raise ValueError("at least one strict planner example is required")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    wall_slab = examples[0].wall_slab
    if any(not math.isclose(item.wall_slab, wall_slab, abs_tol=1e-8) for item in examples):
        raise ValueError("all strict planner examples must use one wall_slab")
    return EnvConfig(
        n_agents=12,
        handle_layout=HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2,
        payload_shape="T",
        world_x=3.9,
        world_y=2.0,
        wall_slab=wall_slab,
        collision_substeps=4,
        # Boundary handles span roughly one metre.  The legacy 1.18 m
        # default makes their graph complete; 0.40 m yields a connected
        # 24-edge radius graph of diameter four, matched to four GBP sweeps.
        comm_radius=0.40,
        source_x=-2.65,
        goal_x=2.65,
        bidirectional=False,
        round_trip=False,
        local_observation=True,
        strict_local_observation=False,
        generic_obstacle_slots=2,
        max_steps=max_steps,
        device=str(torch.device(device)),
    )


def _stack_scenarios(
    examples: Sequence[StrictPlannerExample],
    indices: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> PianoBatch:
    if not indices:
        raise ValueError("scenario batch indices cannot be empty")
    chosen = [examples[index] for index in indices]

    def stack(name: str) -> Tensor:
        return torch.stack([getattr(item, name) for item in chosen]).to(
            device=device, dtype=dtype
        )

    return PianoBatch(
        state0=stack("state0"),
        goal=stack("goal"),
        wall_x=stack("wall_x"),
        gap_y=stack("gap_y"),
        gap_half=stack("gap_half"),
        source=stack("source"),
        leg=None,
        obstacle_rects=stack("obstacle_rects"),
    )


def _padded_paths(
    examples: Sequence[StrictPlannerExample],
    indices: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    chosen = [examples[index].path for index in indices]
    lengths = torch.tensor([path.shape[0] for path in chosen], dtype=torch.long, device=device)
    maximum = int(lengths.max().item())
    padded = torch.empty(len(chosen), maximum, 3, dtype=dtype, device=device)
    for row, path in enumerate(chosen):
        length = path.shape[0]
        value = path.to(device=device, dtype=dtype)
        padded[row, :length] = value
        padded[row, length:] = value[-1]
    return padded, lengths


def build_fresh_serpentine_policy(
    cfg: EnvConfig,
    examples: Sequence[StrictPlannerExample],
) -> SheafGBPPolicy:
    """Instantiate the frozen experiment architecture from fresh parameters."""

    probe = _stack_scenarios(examples, [0], device=cfg.device)
    obs_dim = observations(probe.state0, probe, cfg).shape[-1]
    arm_cfg = ArmConfig(
        obs_dim=obs_dim,
        stalk_dim=32,
        edge_dim=12,
        gbp_steps=4,
        temporal_window=6,
        restriction_rank=4,
        queryable_goal_free_temporal=False,
    )
    return SheafGBPPolicy(cfg, arm_cfg).to(cfg.device)


def _planner_semantic_section(
    state: Tensor,
    path: Tensor,
    path_lengths: Tensor,
    teacher: PlannerPathTeacherOutput,
    cfg: EnvConfig,
    teacher_cfg: PlannerPathTeacherConfig,
) -> Tensor:
    """Build ``twist3,wrench3,cross-track,progress`` supervision.

    Cross-track is signed positive to the left of planner order.  Progress is
    normalized ordered SE(2) arclength, so retreat advances progress even when
    world x decreases.
    """

    batch, waypoint_count, _ = path.shape
    row = torch.arange(batch, device=state.device)
    nearest = teacher.diagnostics.nearest_index
    previous = torch.maximum(nearest - 1, torch.zeros_like(nearest))
    following = torch.minimum(nearest + 1, path_lengths - 1)
    tangent = path[row, following, :2] - path[row, previous, :2]
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    default_tangent = torch.zeros_like(tangent)
    default_tangent[:, 0] = 1.0
    unit_tangent = torch.where(
        tangent_norm > 1e-8,
        tangent / tangent_norm.clamp_min(1e-8),
        default_tangent,
    )
    offset = state[:, :2] - path[row, nearest, :2]
    cross_track = (unit_tangent[:, 0] * offset[:, 1] - unit_tangent[:, 1] * offset[:, 0]).clamp(
        -1.0, 1.0
    )

    if waypoint_count == 1:
        cumulative = state.new_zeros((batch, 1))
    else:
        delta_xy = path[:, 1:, :2] - path[:, :-1, :2]
        delta_theta = wrap_angle(path[:, 1:, 2] - path[:, :-1, 2])
        distance = torch.sqrt(
            delta_xy.square().sum(dim=-1)
            + (teacher_cfg.orientation_distance_scale * delta_theta).square()
        )
        segment_axis = torch.arange(1, waypoint_count, device=state.device).view(1, -1)
        distance = torch.where(
            segment_axis < path_lengths[:, None], distance, torch.zeros_like(distance)
        )
        cumulative = torch.cat((state.new_zeros((batch, 1)), distance.cumsum(dim=-1)), dim=-1)
    current_s = cumulative.gather(1, nearest[:, None]).squeeze(1)
    total_s = cumulative.gather(1, (path_lengths - 1)[:, None]).squeeze(1)
    progress = torch.where(
        total_s > 1e-8,
        current_s / total_s.clamp_min(1e-8),
        torch.ones_like(total_s),
    )

    twist_scale = state.new_tensor(
        [
            teacher_cfg.maximum_linear_speed,
            teacher_cfg.maximum_linear_speed,
            teacher_cfg.maximum_angular_speed,
        ]
    )
    wrench_scale = state.new_tensor(
        [
            cfg.n_agents * cfg.max_force,
            cfg.n_agents * cfg.max_force,
            0.75 * cfg.n_agents * cfg.max_force,
        ]
    )
    section = torch.cat(
        (
            teacher.diagnostics.desired_twist / twist_scale,
            teacher.wrench / wrench_scale,
            cross_track[:, None],
            progress.clamp(0.0, 1.0)[:, None],
        ),
        dim=-1,
    )
    if section.shape != (batch, SEMANTIC_SECTION_WIDTH):
        raise RuntimeError("planner semantic section width changed")
    return section


@dataclass(frozen=True)
class _TeacherSequenceRollout:
    state: Tensor
    next_state: Tensor
    minimum_path_index: Tensor
    target_forces: Tensor
    target_section: Tensor
    transition_clearance: Tensor
    transition_safe: Tensor
    valid_mask: Tensor


@torch.no_grad()
def _rollout_teacher_sequence(
    initial_state: Tensor,
    initial_minimum_path_index: Tensor,
    source_anchored: Tensor,
    scenario: PianoBatch,
    path: Tensor,
    path_lengths: Tensor,
    cfg: EnvConfig,
    controls: PlannerPathTeacherConfig,
    protocol: RecurrentBCProtocol,
) -> _TeacherSequenceRollout:
    """Apply the real teacher/dynamics recurrence and audit every real edge."""

    batch_size = initial_state.shape[0]
    if source_anchored.shape != (batch_size,):
        raise ValueError("source_anchored must have shape [batch]")
    current = initial_state.clone()
    minimum_index = initial_minimum_path_index.clone()
    active = torch.ones(batch_size, dtype=torch.bool, device=current.device)

    states: list[Tensor] = []
    next_states: list[Tensor] = []
    minimum_indices: list[Tensor] = []
    target_forces: list[Tensor] = []
    target_sections: list[Tensor] = []
    clearances: list[Tensor] = []
    safe_masks: list[Tensor] = []
    valid_masks: list[Tensor] = []
    for tick in range(protocol.sequence_length):
        # Source rows are left padded.  They must not move, advance their
        # ordered-path index, or populate model memory during burn-in.
        scheduled = (~source_anchored) | (tick >= protocol.burn_in_steps)
        valid = scheduled & active
        teacher = planner_path_teacher_output(
            current,
            path,
            cfg,
            controls,
            path_lengths=path_lengths,
            minimum_path_index=minimum_index,
        )
        proposed = step_dynamics(current, teacher.forces, cfg)
        hit = transition_collision_mask(current, proposed, scenario, cfg)
        clearance = transition_clearance_margin(current, proposed, scenario, cfg)
        if bool((hit != (clearance <= 0.0)).any()):
            raise RuntimeError(
                "teacher transition boolean and clearance authorities disagree"
            )
        safe = (~hit) & (clearance > 0.0)
        advance = valid & safe
        next_state = torch.where(advance[:, None], proposed, current)

        section = _planner_semantic_section(
            current,
            path,
            path_lengths,
            teacher,
            cfg,
            controls,
        )
        states.append(current.clone())
        next_states.append(next_state.clone())
        minimum_indices.append(minimum_index.clone())
        target_forces.append(
            torch.where(valid[:, None, None], teacher.forces, torch.zeros_like(teacher.forces))
        )
        target_sections.append(
            torch.where(valid[:, None], section, torch.zeros_like(section))
        )
        clearances.append(torch.where(valid, clearance, torch.zeros_like(clearance)))
        safe_masks.append((~valid) | safe)
        valid_masks.append(valid)

        next_minimum = torch.maximum(
            minimum_index,
            teacher.diagnostics.nearest_index,
        )
        minimum_index = torch.where(advance, next_minimum, minimum_index)
        reached = success_mask(proposed, scenario, cfg)
        active = active & ~(valid & ((~safe) | reached))
        current = next_state

    return _TeacherSequenceRollout(
        state=torch.stack(states),
        next_state=torch.stack(next_states),
        minimum_path_index=torch.stack(minimum_indices),
        target_forces=torch.stack(target_forces),
        target_section=torch.stack(target_sections),
        transition_clearance=torch.stack(clearances),
        transition_safe=torch.stack(safe_masks),
        valid_mask=torch.stack(valid_masks),
    )


def _select_rollout_rows(
    candidate: Tensor,
    nominal: Tensor,
    candidate_accepted: Tensor,
) -> Tensor:
    if candidate.ndim < 2 or candidate.shape[:2] != nominal.shape[:2]:
        raise ValueError("candidate and nominal sequence tensors must share [time,batch]")
    selection = candidate_accepted.view(1, -1, *([1] * (candidate.ndim - 2)))
    return torch.where(selection, candidate, nominal)


_MIDROUTE_WAYPOINT_POOL_CACHE: dict[tuple[Any, ...], tuple[Tensor, ...]] = {}


def _midroute_waypoint_pool_cache_key(
    examples: Sequence[StrictPlannerExample],
    cfg: EnvConfig,
    controls: PlannerPathTeacherConfig,
    protocol: RecurrentBCProtocol,
) -> tuple[Any, ...]:
    """Bind cached waypoint certificates to all safety-relevant inputs."""

    return (
        SERPENTINE_BC_SCHEMA,
        tuple(example.record_sha256 for example in examples),
        json.dumps(asdict(cfg), sort_keys=True, allow_nan=False),
        json.dumps(asdict(controls), sort_keys=True, allow_nan=False),
        json.dumps(asdict(protocol), sort_keys=True, allow_nan=False),
    )


@torch.no_grad()
def _precertified_midroute_waypoint_pools(
    examples: Sequence[StrictPlannerExample],
    cfg: EnvConfig,
    controls: PlannerPathTeacherConfig,
    protocol: RecurrentBCProtocol,
    *,
    certification_batch_size: int = 512,
) -> tuple[Tensor, ...]:
    """Return waypoint indices whose every real nominal transition is safe.

    Certification is deterministic and consumes no sampler RNG.  A waypoint
    is admitted only when boolean swept-SAT is false and signed swept-SAT
    clearance is strictly positive at every valid transition.  Legitimate
    padding after ``success_mask`` is retained; the exact final waypoint is
    excluded from the selectable set.  The result is cached for the exact
    verified records, environment, teacher, protocol, and device, so normal
    batches never rescan a planner path.
    """

    if certification_batch_size <= 0:
        raise ValueError("certification_batch_size must be positive")
    if not examples:
        raise ValueError("waypoint certification requires at least one example")
    controls.validate()
    protocol.validate()
    key = _midroute_waypoint_pool_cache_key(examples, cfg, controls, protocol)
    cached = _MIDROUTE_WAYPOINT_POOL_CACHE.get(key)
    if cached is not None:
        return cached

    device = torch.device(cfg.device)
    pools: list[Tensor] = []
    for example_index, example in enumerate(examples):
        certified_chunks: list[Tensor] = []
        # Match the historical sampler's selectable domain: the exact final
        # planner waypoint is a terminal state, not a useful window start.
        selectable_count = int(example.path.shape[0]) - 1
        if selectable_count <= 0:
            raise RuntimeError("strict planner path has no selectable waypoint")
        for start in range(0, selectable_count, certification_batch_size):
            stop = min(selectable_count, start + certification_batch_size)
            waypoint = torch.arange(start, stop, dtype=torch.long, device=device)
            count = int(waypoint.numel())
            repeated_indices = [example_index] * count
            scenario = _stack_scenarios(examples, repeated_indices, device=device)
            path, path_lengths = _padded_paths(
                examples,
                repeated_indices,
                device=device,
            )
            row = torch.arange(count, device=device)
            initial = torch.zeros(count, 6, device=device)
            initial[:, :3] = path[row, waypoint]
            initial_safe = ~collision_mask(initial, scenario, cfg)
            rollout = _rollout_teacher_sequence(
                initial,
                waypoint,
                torch.zeros(count, dtype=torch.bool, device=device),
                scenario,
                path,
                path_lengths,
                cfg,
                controls,
                protocol,
            )
            real_transition_safe = (
                (~rollout.valid_mask)
                | (
                    rollout.transition_safe
                    & (rollout.transition_clearance > 0.0)
                )
            ).all(dim=0)
            protocol_safe = (
                initial_safe
                & real_transition_safe
            )
            certified_chunks.append(waypoint[protocol_safe])
        pool = torch.cat(certified_chunks)
        if pool.numel() == 0:
            raise RuntimeError(
                "strict planner record has no waypoint with a complete safe "
                f"recurrent teacher window: {example.record_sha256}"
            )
        pools.append(pool)

    result = tuple(pools)
    _MIDROUTE_WAYPOINT_POOL_CACHE[key] = result
    return result


@torch.no_grad()
def sample_planner_bc_batch(
    examples: Sequence[StrictPlannerExample],
    cfg: EnvConfig,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    perturbation: PathPerturbationConfig | None = None,
    teacher_cfg: PlannerPathTeacherConfig | None = None,
    protocol: RecurrentBCProtocol | None = None,
) -> PlannerBCBatch:
    """Generate contiguous safe recurrent windows with whole-window fallback.

    A noisy mid-route window is admitted only if its initial pose and every
    real context/supervised transition pass both exact swept-SAT authorities.
    Otherwise the entire row is regenerated from the corresponding nominal
    planner waypoint.  Nominal failure raises instead of leaking an unsafe
    label.  Exactly the predeclared fraction (up to integer rounding) starts
    at the true source with invalid left padding and empty model context.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not examples:
        raise ValueError("at least one strict planner example is required")
    noise_cfg = PathPerturbationConfig() if perturbation is None else perturbation
    controls = serpentine_teacher_config() if teacher_cfg is None else teacher_cfg
    sequence_protocol = RecurrentBCProtocol() if protocol is None else protocol
    noise_cfg.validate()
    controls.validate()
    sequence_protocol.validate()
    device = torch.device(cfg.device)
    selected = torch.randint(
        len(examples),
        (batch_size,),
        device=device,
        generator=generator,
    )
    selected_list = [int(value) for value in selected.detach().cpu().tolist()]
    scenario = _stack_scenarios(examples, selected_list, device=device)
    path, path_lengths = _padded_paths(examples, selected_list, device=device)
    waypoint_pools = _precertified_midroute_waypoint_pools(
        examples,
        cfg,
        controls,
        sequence_protocol,
    )

    source_count = int(
        math.floor(batch_size * sequence_protocol.source_anchor_fraction + 0.5)
    )
    source_anchored = torch.arange(batch_size, device=device) < source_count
    waypoint_fraction = torch.rand(batch_size, device=device, generator=generator)
    waypoint = torch.zeros(batch_size, dtype=torch.long, device=device)
    for example_index, pool in enumerate(waypoint_pools):
        rows_for_example = (selected == example_index) & ~source_anchored
        choice = torch.minimum(
            (waypoint_fraction[rows_for_example] * pool.numel()).long(),
            torch.full_like(waypoint[rows_for_example], pool.numel() - 1),
        )
        waypoint[rows_for_example] = pool[choice]
    row = torch.arange(batch_size, device=device)
    nominal_initial = torch.zeros(batch_size, 6, device=device)
    nominal_initial[:, :3] = path[row, waypoint]
    nominal_initial = torch.where(
        source_anchored[:, None],
        scenario.state0,
        nominal_initial,
    )
    if bool(collision_mask(nominal_initial, scenario, cfg).any()):
        raise RuntimeError(
            "a verified planner waypoint/source collided under the BC environment"
        )

    noise = torch.randn(batch_size, 6, device=device, generator=generator)
    noise[:, :2] *= noise_cfg.position_std
    noise[:, 2] *= noise_cfg.angle_std
    noise[:, 3:5] *= noise_cfg.linear_velocity_std
    noise[:, 5] *= noise_cfg.angular_velocity_std
    candidate_initial = nominal_initial + noise
    candidate_initial[:, 2] = wrap_angle(candidate_initial[:, 2])
    candidate_initial = torch.where(
        source_anchored[:, None],
        nominal_initial,
        candidate_initial,
    )
    candidate_collision = collision_mask(candidate_initial, scenario, cfg)
    candidate_pose_rejected = (~source_anchored) & candidate_collision
    candidate_initial = torch.where(
        candidate_pose_rejected[:, None], nominal_initial, candidate_initial
    )

    candidate_rollout = _rollout_teacher_sequence(
        candidate_initial,
        waypoint,
        source_anchored,
        scenario,
        path,
        path_lengths,
        cfg,
        controls,
        sequence_protocol,
    )
    candidate_transition_rejected = (~source_anchored) & (
        candidate_rollout.valid_mask & ~candidate_rollout.transition_safe
    ).any(dim=0)
    candidate_accepted = (
        (~source_anchored)
        & (~candidate_pose_rejected)
        & (~candidate_transition_rejected)
    )

    nominal_rollout = _rollout_teacher_sequence(
        nominal_initial,
        waypoint,
        source_anchored,
        scenario,
        path,
        path_lengths,
        cfg,
        controls,
        sequence_protocol,
    )
    nominal_unsafe = nominal_rollout.valid_mask & ~nominal_rollout.transition_safe
    if bool(nominal_unsafe.any()):
        bad_rows = torch.where(nominal_unsafe.any(dim=0))[0].detach().cpu().tolist()
        raise RuntimeError(
            "nominal recurrent teacher rollout failed swept-SAT safety for rows "
            f"{bad_rows}"
        )

    state_sequence = _select_rollout_rows(
        candidate_rollout.state, nominal_rollout.state, candidate_accepted
    )
    next_state_sequence = _select_rollout_rows(
        candidate_rollout.next_state, nominal_rollout.next_state, candidate_accepted
    )
    minimum_sequence = _select_rollout_rows(
        candidate_rollout.minimum_path_index,
        nominal_rollout.minimum_path_index,
        candidate_accepted,
    )
    force_sequence = _select_rollout_rows(
        candidate_rollout.target_forces,
        nominal_rollout.target_forces,
        candidate_accepted,
    )
    section_sequence = _select_rollout_rows(
        candidate_rollout.target_section,
        nominal_rollout.target_section,
        candidate_accepted,
    )
    clearance_sequence = _select_rollout_rows(
        candidate_rollout.transition_clearance,
        nominal_rollout.transition_clearance,
        candidate_accepted,
    )
    safe_sequence = _select_rollout_rows(
        candidate_rollout.transition_safe,
        nominal_rollout.transition_safe,
        candidate_accepted,
    )
    valid_sequence = _select_rollout_rows(
        candidate_rollout.valid_mask,
        nominal_rollout.valid_mask,
        candidate_accepted,
    )
    if bool((valid_sequence & ~safe_sequence).any()) or bool(
        (clearance_sequence[valid_sequence] <= 0.0).any()
    ):
        raise RuntimeError("unsafe supervised transition escaped whole-window fallback")

    tick = torch.arange(sequence_protocol.sequence_length, device=device)[:, None]
    context_mask = valid_sequence & (tick < sequence_protocol.burn_in_steps)
    supervision_mask = valid_sequence & (tick >= sequence_protocol.burn_in_steps)
    first_supervised = sequence_protocol.burn_in_steps
    return PlannerBCBatch(
        # Compatibility view for matched one-step baseline code.  The native
        # Sheaf trainer below is sequence-only and validates that invariant.
        state=state_sequence[first_supervised],
        nominal_state=nominal_rollout.state[first_supervised],
        scenario=scenario,
        path=path,
        path_lengths=path_lengths,
        minimum_path_index=minimum_sequence[first_supervised],
        target_forces=force_sequence[first_supervised],
        target_section=section_sequence[first_supervised],
        selected_record_index=selected,
        perturbation_accepted=candidate_accepted,
        sequence_state=state_sequence,
        sequence_next_state=next_state_sequence,
        sequence_nominal_state=nominal_rollout.state,
        sequence_minimum_path_index=minimum_sequence,
        sequence_target_forces=force_sequence,
        sequence_target_section=section_sequence,
        sequence_transition_clearance=clearance_sequence,
        sequence_transition_safe=safe_sequence,
        sequence_valid_mask=valid_sequence,
        sequence_context_mask=context_mask,
        sequence_supervision_mask=supervision_mask,
        source_anchored=source_anchored,
        candidate_pose_rejected=candidate_pose_rejected,
        candidate_transition_rejected=candidate_transition_rejected,
        protocol=sequence_protocol,
    )


def serpentine_bc_loss(
    policy: SheafGBPPolicy,
    batch: PlannerBCBatch,
    cfg: EnvConfig,
    edge_index: Tensor,
    *,
    semantic_section_weight: float = SEMANTIC_SECTION_WEIGHT,
) -> BCLoss:
    """Recurrent teacher-forced BC with no-grad burn-in and full TBPTT."""

    if semantic_section_weight < 0.0:
        raise ValueError("semantic_section_weight must be non-negative")
    batch.protocol.validate()
    sequence_length, batch_size = batch.sequence_state.shape[:2]
    if sequence_length != batch.protocol.sequence_length:
        raise ValueError("batch sequence length disagrees with its frozen protocol")
    if batch.protocol.supervised_steps < 2:
        raise ValueError("single-step recurrent BC is forbidden")
    expected_prefix = (sequence_length, batch_size)
    masks = (
        batch.sequence_valid_mask,
        batch.sequence_context_mask,
        batch.sequence_supervision_mask,
        batch.sequence_transition_safe,
        batch.sequence_transition_clearance,
    )
    if any(value.shape[:2] != expected_prefix for value in masks):
        raise ValueError("sequence masks/clearance must share [time,batch]")
    if bool((batch.sequence_context_mask & batch.sequence_supervision_mask).any()):
        raise ValueError("context and supervision masks must be disjoint")
    if bool(
        (
            (batch.sequence_context_mask | batch.sequence_supervision_mask)
            & ~batch.sequence_valid_mask
        ).any()
    ):
        raise ValueError("context/supervision cannot select padded transitions")
    if bool(
        (
            batch.sequence_valid_mask
            & (
                (~batch.sequence_transition_safe)
                | (batch.sequence_transition_clearance <= 0.0)
            )
        ).any()
    ):
        raise ValueError("recurrent BC batch contains an unsafe transition")

    hidden = policy.init_hidden(batch_size, batch.sequence_state.device)
    force_numerator = batch.sequence_state.new_zeros(())
    semantic_numerator = batch.sequence_state.new_zeros(())
    supervised_count = batch.sequence_state.new_zeros(())
    temporal_counts: list[Tensor] = []
    for tick in range(sequence_length):
        obs = observations(batch.sequence_state[tick], batch.scenario, cfg)
        if tick < batch.protocol.burn_in_steps:
            with torch.no_grad():
                output = policy(obs, edge_index, hidden)
        else:
            output = policy(obs, edge_index, hidden)
        section = output.aux.get("section")
        memory_count = output.aux.get("temporal_memory_count")
        if not isinstance(section, Tensor) or section.shape[-1] < SEMANTIC_SECTION_WIDTH:
            raise RuntimeError("Sheaf-GBP output omitted the 8-D semantic section prefix")
        if not isinstance(memory_count, Tensor) or memory_count.shape != (
            batch_size,
            cfg.n_agents,
        ):
            raise RuntimeError("Sheaf-GBP output omitted temporal memory counts")
        temporal_counts.append(memory_count.detach())

        valid = batch.sequence_valid_mask[tick]
        hidden = torch.where(valid[:, None, None], output.hidden, hidden)
        supervised = batch.sequence_supervision_mask[tick]
        if bool(supervised.any()):
            force_error = (
                output.forces / cfg.max_force
                - batch.sequence_target_forces[tick].detach() / cfg.max_force
            ).square().mean(dim=(-2, -1))
            section_target = batch.sequence_target_section[tick, :, None, :].expand(
                -1, cfg.n_agents, -1
            )
            semantic_error = (
                section[..., :SEMANTIC_SECTION_WIDTH] - section_target.detach()
            ).square().mean(dim=(-2, -1))
            weight = supervised.to(force_error.dtype)
            force_numerator = force_numerator + (force_error * weight).sum()
            semantic_numerator = semantic_numerator + (semantic_error * weight).sum()
            supervised_count = supervised_count + weight.sum()

    if float(supervised_count.detach().cpu().item()) <= 0.0:
        raise RuntimeError("recurrent BC batch contains no valid supervised transition")
    force_mse = force_numerator / supervised_count
    semantic_mse = semantic_numerator / supervised_count
    return BCLoss(
        total=force_mse + semantic_section_weight * semantic_mse,
        force_mse=force_mse,
        semantic_section_mse=semantic_mse,
        temporal_memory_count=torch.stack(temporal_counts),
    )


def architecture_record(policy: SheafGBPPolicy) -> dict[str, Any]:
    """Serialization-safe declaration of the exact first-stage model."""

    return {
        "schema_version": SERPENTINE_BC_SCHEMA,
        "policy": "fresh SheafGBPPolicy",
        "environment": asdict(policy.env_cfg),
        "arm": asdict(policy.arm_cfg),
        "semantic_section": [
            "desired_twist_x",
            "desired_twist_y",
            "desired_twist_omega",
            "realized_wrench_fx",
            "realized_wrench_fy",
            "realized_wrench_tau",
            "signed_cross_track",
            "ordered_path_progress",
        ],
        "teacher": {
            "name": "ordered-SE2-planner-path-force-teacher",
            "config": asdict(serpentine_teacher_config()),
            "training_only_privileged": True,
            "counts_as_policy": False,
            "additional_policy_inference_inputs": 0,
        },
    }


def _graph_distances(edge_index: Tensor, n_agents: int) -> list[list[int]]:
    adjacency: list[set[int]] = [set() for _ in range(n_agents)]
    for source, target in edge_index.detach().cpu().tolist():
        if source == target:
            raise ValueError("serpentine radius graph cannot contain self edges")
        if not (0 <= source < n_agents and 0 <= target < n_agents):
            raise ValueError("serpentine radius graph endpoint is out of range")
        adjacency[source].add(target)
        adjacency[target].add(source)
    distances: list[list[int]] = []
    for root in range(n_agents):
        row = [-1] * n_agents
        row[root] = 0
        queue = [root]
        for node in queue:
            for neighbor in adjacency[node]:
                if row[neighbor] < 0:
                    row[neighbor] = row[node] + 1
                    queue.append(neighbor)
        distances.append(row)
    return distances


def _serpentine_radius_graph_audit(cfg: EnvConfig, edge_index: Tensor) -> dict[str, Any]:
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("radius edge_index must have shape [E,2]")
    canonical = {
        tuple(sorted((int(source), int(target))))
        for source, target in edge_index.detach().cpu().tolist()
    }
    if len(canonical) != int(edge_index.shape[0]):
        raise ValueError("serpentine radius graph contains duplicate undirected edges")
    distances = _graph_distances(edge_index, cfg.n_agents)
    connected = all(value >= 0 for row in distances for value in row)
    diameter = max(value for row in distances for value in row)
    graph = {
        "comm_radius": float(cfg.comm_radius),
        "n_agents": int(cfg.n_agents),
        "undirected_edges": int(edge_index.shape[0]),
        "directed_edges": 2 * int(edge_index.shape[0]),
        "connected": connected,
        "diameter": diameter,
        "edge_index": [
            [int(source), int(target)]
            for source, target in edge_index.detach().cpu().tolist()
        ],
    }
    if not math.isclose(cfg.comm_radius, 0.40, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("serpentine Sheaf evaluation requires comm_radius=0.40")
    if (
        cfg.n_agents != 12
        or graph["undirected_edges"] != SERPENTINE_UNDIRECTED_EDGES
        or not connected
        or diameter != SERPENTINE_GRAPH_DIAMETER
    ):
        raise ValueError("serpentine Sheaf evaluation radius graph audit failed")
    return graph


def _scenario_row(scenario: PianoBatch, index: int) -> PianoBatch:
    def row(value: Tensor | None) -> Tensor | None:
        return None if value is None else value[index : index + 1]

    return PianoBatch(
        state0=row(scenario.state0),  # type: ignore[arg-type]
        goal=row(scenario.goal),  # type: ignore[arg-type]
        wall_x=row(scenario.wall_x),  # type: ignore[arg-type]
        gap_y=row(scenario.gap_y),  # type: ignore[arg-type]
        gap_half=row(scenario.gap_half),  # type: ignore[arg-type]
        source=row(scenario.source),  # type: ignore[arg-type]
        leg=row(scenario.leg),
        obstacle_rects=row(scenario.obstacle_rects),
    )


def _backtracking_from_attempted_trajectories(
    trajectory_states: Tensor,
    trajectory_present: Tensor,
    scenario: PianoBatch,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Audit initial state plus every active tick's attempted endpoint."""

    batch = scenario.state0.shape[0]
    if trajectory_states.ndim != 3 or trajectory_states.shape[1:] != (batch, 6):
        raise ValueError("trajectory_states must have shape [ticks, batch, 6]")
    if trajectory_present.shape != trajectory_states.shape[:2]:
        raise ValueError("trajectory presence mask shape mismatch")
    rows: list[dict[str, float | int]] = []
    for index in range(batch):
        path = trajectory_states[trajectory_present[:, index], index]
        _, maximum_excursion, retreat_edges = route_backtracking_statistics(
            path,
            _scenario_row(scenario, index),
        )
        rows.append(
            {
                "maximum_retreat_excursion_m": float(maximum_excursion),
                "retreat_edge_count": int(retreat_edges),
            }
        )
    excursions = [float(row["maximum_retreat_excursion_m"]) for row in rows]
    edge_counts = [int(row["retreat_edge_count"]) for row in rows]
    aggregate = {
        "measurement": "actual closed-loop trajectory",
        "definition": "planner_scout_protocol.route_backtracking_statistics",
        "route_progress": "unclamped projection along active source-to-goal direction",
        "trajectory": "initial state plus every active tick proposed endpoint",
        "terminal_collision_attempt_included": True,
        "maximum_retreat_excursion_m": {
            "min": min(excursions),
            "mean": sum(excursions) / batch,
            "max": max(excursions),
        },
        "retreat_edge_count": {
            "total": sum(edge_counts),
            "mean": sum(edge_counts) / batch,
            "max": max(edge_counts),
        },
    }
    return rows, aggregate


def _attach_trajectory_metrics(
    result: dict[str, Any],
    *,
    state: Tensor,
    scenario: PianoBatch,
    collided: Tensor,
    minimum_clearance: Tensor,
    trajectory_states: Tensor,
    trajectory_present: Tensor,
) -> None:
    """Attach clearance, goal error, and route-relative retreat metrics."""

    if not torch.equal(collided, minimum_clearance <= 0):
        raise RuntimeError("collision outcome and minimum swept clearance disagree")
    retreat_rows, backtracking = _backtracking_from_attempted_trajectories(
        trajectory_states,
        trajectory_present,
        scenario,
    )
    goal_error = torch.linalg.vector_norm(state[:, :2] - scenario.goal[:, :2], dim=-1)
    angle_error = wrap_angle(state[:, 2] - scenario.goal[:, 2]).abs()
    result["minimum_exact_sat_clearance"] = float(minimum_clearance.min().item())
    result["mean_minimum_exact_sat_clearance"] = float(minimum_clearance.mean().item())
    result["backtracking"] = backtracking
    for index, row in enumerate(result["per_episode"]):
        row.update(
            {
                "minimum_exact_sat_clearance": float(minimum_clearance[index].item()),
                **retreat_rows[index],
                "final_position_error": float(goal_error[index].item()),
                "final_angle_error": float(angle_error[index].item()),
            }
        )


def _outcome_summary(
    *,
    actor: str,
    counts_as_policy: bool,
    record_hashes: Sequence[str],
    succeeded: Tensor,
    collided: Tensor,
    alive: Tensor,
    finish_step: Tensor,
    final_state: Tensor,
) -> dict[str, Any]:
    count = int(succeeded.numel())
    partition = succeeded.to(torch.int8) + collided.to(torch.int8) + alive.to(torch.int8)
    if not bool((partition == 1).all()):
        raise RuntimeError("closed-loop outcomes do not form a partition")
    return {
        "actor": actor,
        "counts_as_policy": counts_as_policy,
        "training_only_privileged": not counts_as_policy,
        "teacher_used_at_policy_inference": False,
        "collision_authority": "exact rectangle SAT at four swept transition substeps",
        "collision_substeps": 4,
        "episode_count": count,
        "success_count": int(succeeded.sum().item()),
        "success_rate": float(succeeded.float().mean().item()),
        "collision_count": int(collided.sum().item()),
        "collision_rate": float(collided.float().mean().item()),
        "timeout_count": int(alive.sum().item()),
        "timeout_rate": float(alive.float().mean().item()),
        "record_sha256": list(record_hashes),
        "per_episode": [
            {
                "record_sha256": record_hashes[index],
                "success": bool(succeeded[index].item()),
                "collision": bool(collided[index].item()),
                "timeout": bool(alive[index].item()),
                "finish_step": int(finish_step[index].item()),
                "final_state": [float(value) for value in final_state[index].tolist()],
            }
            for index in range(count)
        ],
    }


@torch.no_grad()
def evaluate_policy_closed_loop(
    policy: SheafGBPPolicy,
    examples: Sequence[StrictPlannerExample],
    cfg: EnvConfig,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Evaluate carried hidden state, swept SAT, retreat, and actual payload."""

    if not examples:
        raise ValueError("at least one strict planner example is required")
    if cfg.collision_substeps != 4:
        raise ValueError("serpentine evaluation requires collision_substeps=4")
    horizon = cfg.max_steps if max_steps is None else max_steps
    if horizon <= 0:
        raise ValueError("evaluation max_steps must be positive")
    edge_index = radius_edges(cfg, device=cfg.device)
    graph = _serpentine_radius_graph_audit(cfg, edge_index)
    indices = list(range(len(examples)))
    scenario = _stack_scenarios(examples, indices, device=cfg.device)
    state = scenario.state0.clone()
    initial_collision = collision_mask(state, scenario, cfg)
    minimum_clearance = collision_clearance_margin(state, scenario, cfg)
    if not torch.equal(initial_collision, minimum_clearance <= 0):
        raise RuntimeError("boolean SAT and signed clearance authorities disagree")
    if bool(initial_collision.any()):
        raise ValueError("strict serpentine policy evaluation starts in collision")
    episode_count = len(examples)
    hidden = policy.init_hidden(episode_count, cfg.device)
    alive = torch.ones(episode_count, dtype=torch.bool, device=state.device)
    succeeded = torch.zeros_like(alive)
    collided = torch.zeros_like(alive)
    active_ticks = torch.zeros(episode_count, dtype=torch.long, device=state.device)
    finish_step = torch.full_like(active_ticks, horizon)
    trajectory_states = torch.zeros(
        horizon + 1,
        episode_count,
        state.shape[-1],
        dtype=state.dtype,
        device=state.device,
    )
    trajectory_present = torch.zeros(
        horizon + 1,
        episode_count,
        dtype=torch.bool,
        device=state.device,
    )
    trajectory_states[0] = state
    trajectory_present[0] = True
    was_training = policy.training
    policy.eval()
    try:
        for step in range(horizon):
            if not bool(alive.any()):
                break
            active = alive.clone()
            active_ticks += active.to(torch.long)
            output = policy(observations(state, scenario, cfg), edge_index, hidden)
            hidden_mask = active.view(active.shape[0], *([1] * (hidden.ndim - 1)))
            hidden = torch.where(hidden_mask, output.hidden, hidden)
            applied = torch.where(
                active[:, None, None],
                output.forces,
                torch.zeros_like(output.forces),
            )
            proposed = step_dynamics(state, applied, cfg)
            trajectory_states[step + 1] = proposed
            trajectory_present[step + 1] = active
            transition_clearance = transition_clearance_margin(
                state,
                proposed,
                scenario,
                cfg,
            )
            hit = active & transition_collision_mask(state, proposed, scenario, cfg)
            if not torch.equal(hit, active & (transition_clearance <= 0)):
                raise RuntimeError(
                    "boolean SAT and signed swept-clearance authorities disagree"
                )
            minimum_clearance = torch.where(
                active,
                torch.minimum(minimum_clearance, transition_clearance),
                minimum_clearance,
            )
            safe = active & ~hit
            state = torch.where(safe[:, None], proposed, state)
            reached = safe & success_mask(state, scenario, cfg)
            terminal = hit | reached
            finish_step = torch.where(
                terminal,
                torch.full_like(finish_step, step + 1),
                finish_step,
            )
            collided |= hit
            succeeded |= reached
            alive &= ~terminal
    finally:
        policy.train(was_training)
    if not torch.equal(active_ticks, finish_step):
        raise RuntimeError("actual active communication ticks disagree with finish_step")

    width = policy.communication_floats_per_directed_edge_round()
    rounds = policy.communication_rounds()
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or rounds <= 0
    ):
        raise RuntimeError("Sheaf-GBP wire width and communication rounds must be positive ints")
    floats_per_tick = graph["directed_edges"] * rounds * width
    packets_per_tick = graph["directed_edges"] * rounds
    bytes_per_tick = 4 * floats_per_tick
    episode_payload_floats = active_ticks * floats_per_tick
    episode_payload_bytes = active_ticks * bytes_per_tick
    episode_packets = active_ticks * packets_per_tick
    success_count = int(succeeded.sum().item())

    result = _outcome_summary(
        actor="learned_sheaf_gbp_policy",
        counts_as_policy=True,
        record_hashes=[example.record_sha256 for example in examples],
        succeeded=succeeded,
        collided=collided,
        alive=alive,
        finish_step=finish_step,
        final_state=state.detach().cpu(),
    )
    result["schema_version"] = SERPENTINE_BC_SCHEMA
    result["planner_used_at_policy_inference"] = False
    _attach_trajectory_metrics(
        result,
        state=state,
        scenario=scenario,
        collided=collided,
        minimum_clearance=minimum_clearance,
        trajectory_states=trajectory_states,
        trajectory_present=trajectory_present,
    )
    for index, row in enumerate(result["per_episode"]):
        row.update(
            {
                "active_communication_ticks": int(active_ticks[index].item()),
                "actual_payload_floats": int(episode_payload_floats[index].item()),
                "actual_payload_bytes": int(episode_payload_bytes[index].item()),
                "actual_packets": int(episode_packets[index].item()),
            }
        )
    total_bytes = int(episode_payload_bytes.sum().item())
    successful_bytes = episode_payload_bytes[succeeded]
    result["communication"] = {
        "accounting": "actual active control ticks; float32 payload only",
        "message_headers_included": False,
        "event_triggering": False,
        "graph": graph,
        "floats_per_directed_edge_round": width,
        "rounds_per_control_tick": rounds,
        "payload_floats_per_control_tick": floats_per_tick,
        "payload_bytes_per_control_tick": bytes_per_tick,
        "packets_per_control_tick": packets_per_tick,
        "actual_payload_floats_total": int(episode_payload_floats.sum().item()),
        "actual_payload_bytes_total": total_bytes,
        "actual_payload_bytes_per_episode": total_bytes / episode_count,
        "actual_payload_bytes_per_success": (
            total_bytes / success_count if success_count else None
        ),
        "successful_episode_payload_bytes_mean": (
            float(successful_bytes.float().mean().item())
            if successful_bytes.numel()
            else None
        ),
        "actual_packets_total": int(episode_packets.sum().item()),
    }
    result["architecture"] = architecture_record(policy)
    return result


@torch.no_grad()
def evaluate_planner_teacher_ceiling(
    examples: Sequence[StrictPlannerExample],
    cfg: EnvConfig,
    *,
    max_steps: int | None = None,
    teacher_cfg: PlannerPathTeacherConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the centralized teacher as a separately labelled ceiling."""

    if not examples:
        raise ValueError("at least one strict planner example is required")
    if cfg.collision_substeps != 4:
        raise ValueError("serpentine teacher ceiling requires collision_substeps=4")
    horizon = cfg.max_steps if max_steps is None else max_steps
    if horizon <= 0:
        raise ValueError("teacher ceiling max_steps must be positive")
    controls = serpentine_teacher_config() if teacher_cfg is None else teacher_cfg
    controls.validate()
    indices = list(range(len(examples)))
    scenario = _stack_scenarios(examples, indices, device=cfg.device)
    paths, path_lengths = _padded_paths(examples, indices, device=cfg.device)
    state = scenario.state0.clone()
    initial_collision = collision_mask(state, scenario, cfg)
    minimum_clearance = collision_clearance_margin(state, scenario, cfg)
    if not torch.equal(initial_collision, minimum_clearance <= 0):
        raise RuntimeError("boolean SAT and signed clearance authorities disagree")
    if bool(initial_collision.any()):
        raise ValueError("strict serpentine teacher ceiling starts in collision")
    minimum_index = torch.zeros(len(examples), dtype=torch.long, device=state.device)
    alive = torch.ones(len(examples), dtype=torch.bool, device=state.device)
    succeeded = torch.zeros_like(alive)
    collided = torch.zeros_like(alive)
    active_ticks = torch.zeros(len(examples), dtype=torch.long, device=state.device)
    finish_step = torch.full_like(active_ticks, horizon)
    trajectory_states = torch.zeros(
        horizon + 1,
        len(examples),
        state.shape[-1],
        dtype=state.dtype,
        device=state.device,
    )
    trajectory_present = torch.zeros(
        horizon + 1,
        len(examples),
        dtype=torch.bool,
        device=state.device,
    )
    trajectory_states[0] = state
    trajectory_present[0] = True
    for step in range(horizon):
        if not bool(alive.any()):
            break
        active = alive.clone()
        active_ticks += active.to(torch.long)
        output = planner_path_teacher_output(
            state,
            paths,
            cfg,
            controls,
            path_lengths=path_lengths,
            minimum_path_index=minimum_index,
        )
        minimum_index = torch.where(
            active,
            torch.maximum(minimum_index, output.diagnostics.nearest_index),
            minimum_index,
        )
        applied = torch.where(
            active[:, None, None],
            output.forces,
            torch.zeros_like(output.forces),
        )
        proposed = step_dynamics(state, applied, cfg)
        trajectory_states[step + 1] = proposed
        trajectory_present[step + 1] = active
        transition_clearance = transition_clearance_margin(state, proposed, scenario, cfg)
        hit = active & transition_collision_mask(state, proposed, scenario, cfg)
        if not torch.equal(hit, active & (transition_clearance <= 0)):
            raise RuntimeError("boolean SAT and signed swept-clearance authorities disagree")
        minimum_clearance = torch.where(
            active,
            torch.minimum(minimum_clearance, transition_clearance),
            minimum_clearance,
        )
        safe = active & ~hit
        state = torch.where(safe[:, None], proposed, state)
        reached = safe & success_mask(state, scenario, cfg)
        terminal = hit | reached
        finish_step = torch.where(terminal, torch.full_like(finish_step, step + 1), finish_step)
        collided |= hit
        succeeded |= reached
        alive &= ~terminal
    if not torch.equal(active_ticks, finish_step):
        raise RuntimeError("actual active teacher ticks disagree with finish_step")
    result = _outcome_summary(
        actor="centralized_privileged_planner_path_teacher_ceiling",
        counts_as_policy=False,
        record_hashes=[example.record_sha256 for example in examples],
        succeeded=succeeded,
        collided=collided,
        alive=alive,
        finish_step=finish_step,
        final_state=state.detach().cpu(),
    )
    _attach_trajectory_metrics(
        result,
        state=state,
        scenario=scenario,
        collided=collided,
        minimum_clearance=minimum_clearance,
        trajectory_states=trajectory_states,
        trajectory_present=trajectory_present,
    )
    for index, row in enumerate(result["per_episode"]):
        row["active_control_ticks"] = int(active_ticks[index].item())
    result["capability_claim_eligible"] = False
    result["ordered_path_available_at_inference"] = True
    result["planner_used_at_policy_inference"] = True
    result["teacher_config"] = asdict(controls)
    return result


__all__ = [
    "BCLoss",
    "PathPerturbationConfig",
    "PlannerBCBatch",
    "SERPENTINE_BC_SCHEMA",
    "SERPENTINE_GRAPH_DIAMETER",
    "SERPENTINE_UNDIRECTED_EDGES",
    "SEMANTIC_SECTION_WIDTH",
    "SEMANTIC_SECTION_WEIGHT",
    "StrictPlannerExample",
    "architecture_record",
    "build_fresh_serpentine_policy",
    "evaluate_planner_teacher_ceiling",
    "evaluate_policy_closed_loop",
    "load_strict_eligible_examples",
    "sample_planner_bc_batch",
    "serpentine_bc_loss",
    "serpentine_env_config",
    "serpentine_teacher_config",
]
