"""Fail-closed protocol for scouting genuinely non-monotone piano-mover maps.

This module deliberately sits outside the planner implementation.  A map is
not labelled ``backtracking_required`` merely because one successful path
happens to move backwards.  Admission requires all of the following:

* a matched unrestricted search reaches the goal;
* its complete piecewise-SE(2) path has a conservative continuous-clearance
  certificate under the benchmark's exact compound-body SAT geometry;
* a matched monotone search exhausts its open set (a timeout/node limit is not
  evidence of infeasibility); and
* the admitted path contains one real route-coordinate retreat excursion of a
  predeclared size, rather than accumulated grid jitter.

The claim is intentionally lattice-relative.  Exhausting a finite monotone
lattice does not prove that no monotone path exists in continuous SE(2).
Persisted JSON makes that scope explicit and is fail-closed for later training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .config import EnvConfig
from .env import PianoBatch, active_endpoints, collision_clearance_margin, wrap_angle
from .payload_shapes import payload_rectangles_for_config


PLANNER_SCOUT_RECORD_SCHEMA = "piano-movers-backtracking-scout-record-v1"
PLANNER_SCOUT_FAMILY_SCHEMA = "piano-movers-backtracking-scout-family-v1"
COLLISION_AUTHORITY = (
    "exact-compound-sat-walls-aabb-reserve0.03-continuous-clearance-bound-v2"
)
CLAIM_SCOPE = "matched-finite-lattice-monotone-infeasibility;not-continuous-global-proof"


@dataclass(frozen=True)
class BacktrackingGateConfig:
    """Predeclared map-admission thresholds."""

    minimum_retreat_excursion_m: float = 0.15
    endpoint_position_tolerance_m: float = 0.05
    endpoint_angle_tolerance_rad: float = 0.10
    required_continuous_clearance_m: float = 1.0e-4
    maximum_subdivision_depth: int = 16

    def validate(self) -> None:
        values = (
            self.minimum_retreat_excursion_m,
            self.endpoint_position_tolerance_m,
            self.endpoint_angle_tolerance_rad,
            self.required_continuous_clearance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("planner scout thresholds must be finite")
        if self.minimum_retreat_excursion_m <= 0.0:
            raise ValueError("minimum_retreat_excursion_m must be positive")
        if self.endpoint_position_tolerance_m < 0.0:
            raise ValueError("endpoint_position_tolerance_m must be non-negative")
        if self.endpoint_angle_tolerance_rad < 0.0:
            raise ValueError("endpoint_angle_tolerance_rad must be non-negative")
        if self.required_continuous_clearance_m < 0.0:
            raise ValueError("required_continuous_clearance_m must be non-negative")
        if self.maximum_subdivision_depth < 1 or self.maximum_subdivision_depth > 24:
            raise ValueError("maximum_subdivision_depth must be in [1, 24]")


@dataclass(frozen=True)
class PlannerRunEvidence:
    """Matched search evidence, excluding the path tensor itself.

    The two runs must share every fingerprint below.  Their sole intended
    semantic difference is ``mode``: unrestricted transitions versus the
    route-monotone subset of those same transitions.
    """

    mode: str
    success: bool
    termination: str
    open_set_exhausted: bool
    resource_limit_hit: bool
    planner_family: str
    base_lattice_sha256: str
    geometry_sha256: str
    heuristic_sha256: str
    edge_cost_sha256: str
    collision_authority: str
    seed: int
    expanded_nodes: int
    generated_nodes: int | None

    def validate(self) -> None:
        if self.mode not in {"unrestricted", "monotone"}:
            raise ValueError("planner mode must be 'unrestricted' or 'monotone'")
        if self.expanded_nodes < 0 or (
            self.generated_nodes is not None and self.generated_nodes < 0
        ):
            raise ValueError("planner node counts must be non-negative")
        if self.resource_limit_hit and self.open_set_exhausted:
            raise ValueError("a resource-limited search cannot claim open-set exhaustion")
        if self.success:
            if self.termination != "goal_reached":
                raise ValueError("successful planner run must terminate with goal_reached")
            if self.open_set_exhausted or self.resource_limit_hit:
                raise ValueError("successful planner run has inconsistent termination flags")
        elif self.termination == "goal_reached":
            raise ValueError("failed planner run cannot terminate with goal_reached")
        for name in (
            "base_lattice_sha256",
            "geometry_sha256",
            "heuristic_sha256",
            "edge_cost_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.collision_authority != COLLISION_AUTHORITY:
            raise ValueError("planner collision authority does not match the scout protocol")
        if not self.planner_family:
            raise ValueError("planner_family must be non-empty")


@dataclass(frozen=True)
class SweptPathAudit:
    passed: bool
    failure_reason: str | None
    state_count: int
    segment_count: int
    clearance_queries: int
    certified_leaf_segments: int
    deepest_subdivision: int
    minimum_sampled_clearance_m: float | None
    start_position_error_m: float
    start_angle_error_rad: float
    goal_position_error_m: float
    goal_angle_error_rad: float
    total_negative_route_variation_m: float
    maximum_retreat_excursion_m: float
    retreat_edge_count: int
    first_failed_segment: int | None


@dataclass(frozen=True)
class BacktrackingGateDecision:
    admitted: bool
    failed_gates: tuple[str, ...]
    gates: Mapping[str, bool]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def tensor_sha256(label: str, value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(label.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _one_scenario(scenario: PianoBatch) -> None:
    if scenario.state0.shape[0] != 1:
        raise ValueError("planner scout audits exactly one scenario per record")
    for name in ("goal", "wall_x", "gap_y", "gap_half", "source"):
        value = getattr(scenario, name)
        if value.shape[0] != 1:
            raise ValueError(f"scenario.{name} must have batch size one")
    if scenario.leg is not None and scenario.leg.shape != (1,):
        raise ValueError("scenario.leg must be None or shape [1]")
    if scenario.obstacle_rects is not None and (
        scenario.obstacle_rects.ndim != 3
        or scenario.obstacle_rects.shape[0] != 1
        or scenario.obstacle_rects.shape[-1] != 4
    ):
        raise ValueError("scenario.obstacle_rects must be None or shape [1, O, 4]")


def _pose_path(path: Tensor) -> Tensor:
    if not isinstance(path, Tensor) or path.ndim != 2 or path.shape[1] not in (3, 6):
        raise ValueError("path must be a tensor with shape [states, 3] or [states, 6]")
    if path.shape[0] < 2:
        raise ValueError("path must contain at least two states")
    if path.device.type != "cpu":
        path = path.detach().cpu()
    else:
        path = path.detach()
    if not bool(torch.isfinite(path).all()):
        raise ValueError("path must contain only finite values")
    if path.shape[1] == 6:
        return path.clone()
    states = torch.zeros(path.shape[0], 6, dtype=path.dtype)
    states[:, :3] = path
    return states


def _cpu_scenario(scenario: PianoBatch) -> PianoBatch:
    def cpu(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.detach().cpu()

    return PianoBatch(
        state0=cpu(scenario.state0),  # type: ignore[arg-type]
        goal=cpu(scenario.goal),  # type: ignore[arg-type]
        wall_x=cpu(scenario.wall_x),  # type: ignore[arg-type]
        gap_y=cpu(scenario.gap_y),  # type: ignore[arg-type]
        gap_half=cpu(scenario.gap_half),  # type: ignore[arg-type]
        source=cpu(scenario.source),  # type: ignore[arg-type]
        leg=cpu(scenario.leg),
        obstacle_rects=cpu(scenario.obstacle_rects),
    )


def route_backtracking_statistics(
    path: Tensor,
    scenario: PianoBatch,
) -> tuple[float, float, int]:
    """Return total negative variation, largest retreat excursion, edge count.

    Progress is measured in metres along the active start-to-goal route and is
    deliberately *not* clamped.  The excursion statistic prevents many tiny
    lattice oscillations from masquerading as a meaningful retreat maneuver.
    """

    _one_scenario(scenario)
    states = _pose_path(path)
    start, goal = active_endpoints(_cpu_scenario(scenario))
    route = goal[0, :2] - start[0, :2]
    length = float(torch.linalg.vector_norm(route))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("active route must have positive finite length")
    direction = route / length
    progress = (states[:, :2] - start[0, :2]).matmul(direction)
    increments = progress[1:] - progress[:-1]
    negative = (-increments).clamp_min(0.0)
    total_negative = float(negative.sum())
    retreat_edge_count = int((increments < -1.0e-9).sum())

    running_peak = progress[0]
    maximum_excursion = torch.zeros((), dtype=progress.dtype)
    for value in progress[1:]:
        maximum_excursion = torch.maximum(maximum_excursion, running_peak - value)
        running_peak = torch.maximum(running_peak, value)
    return total_negative, float(maximum_excursion), retreat_edge_count


def _payload_radius(cfg: EnvConfig, dtype: torch.dtype) -> float:
    rectangles = payload_rectangles_for_config(cfg, device="cpu").to(dtype=dtype)
    return float(torch.linalg.vector_norm(rectangles.reshape(-1, 2), dim=-1).amax())


def audit_swept_pose_path(
    path: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    gate_cfg: BacktrackingGateConfig | None = None,
) -> SweptPathAudit:
    """Conservatively certify every continuous piecewise-SE(2) transition.

    For one interpolation interval, every payload point moves by at most
    ``||delta_xy|| + radius * |delta_theta|``.  If the exact-SAT clearance at
    an endpoint is larger than that bound plus the required margin, the same
    separating axis remains valid throughout the interval.  Otherwise the
    interval is bisected.  Reaching the predeclared depth without a certificate
    fails closed even when all sampled poses happen to be collision-free.
    """

    options = gate_cfg or BacktrackingGateConfig()
    options.validate()
    _one_scenario(scenario)
    states = _pose_path(path).to(dtype=torch.float32)
    cpu_scenario = _cpu_scenario(scenario)
    if cfg.device != "cpu":
        cfg = EnvConfig(**{**asdict(cfg), "device": "cpu"})

    start, goal = active_endpoints(cpu_scenario)
    start_position_error = float(torch.linalg.vector_norm(states[0, :2] - start[0, :2]))
    start_angle_error = float(wrap_angle(states[0, 2] - start[0, 2]).abs())
    goal_position_error = float(torch.linalg.vector_norm(states[-1, :2] - goal[0, :2]))
    goal_angle_error = float(wrap_angle(states[-1, 2] - goal[0, 2]).abs())
    total_negative, maximum_excursion, retreat_edges = route_backtracking_statistics(
        states, cpu_scenario
    )
    endpoint_failure = None
    if (
        start_position_error > options.endpoint_position_tolerance_m
        or start_angle_error > options.endpoint_angle_tolerance_rad
    ):
        endpoint_failure = "start_endpoint_mismatch"
    elif (
        goal_position_error > options.endpoint_position_tolerance_m
        or goal_angle_error > options.endpoint_angle_tolerance_rad
    ):
        endpoint_failure = "goal_endpoint_mismatch"

    radius = _payload_radius(cfg, states.dtype)
    clearance_queries = 0
    certified_leaves = 0
    deepest = 0
    minimum_clearance = float("inf")
    first_failed_segment: int | None = None
    failure_reason = endpoint_failure

    @torch.no_grad()
    def clearance(pose: Tensor) -> float:
        nonlocal clearance_queries, minimum_clearance
        clearance_queries += 1
        value = float(collision_clearance_margin(pose.view(1, 6), cpu_scenario, cfg)[0])
        minimum_clearance = min(minimum_clearance, value)
        return value

    def midpoint(left: Tensor, right: Tensor) -> Tensor:
        value = (left + right) * 0.5
        angle_delta = wrap_angle(right[2] - left[2])
        value[2] = wrap_angle(left[2] + 0.5 * angle_delta)
        return value

    def certify(left: Tensor, right: Tensor, left_clearance: float, depth: int) -> str | None:
        nonlocal certified_leaves, deepest
        deepest = max(deepest, depth)
        if left_clearance <= options.required_continuous_clearance_m:
            return "sampled_pose_collision_or_clearance_violation"
        angle_delta = abs(float(wrap_angle(right[2] - left[2])))
        point_motion_bound = float(torch.linalg.vector_norm(right[:2] - left[:2])) + (
            radius * angle_delta
        )
        # The endpoint's currently separating SAT axis stays separating for
        # the entire interval under this conservative rigid-point motion bound.
        if left_clearance > point_motion_bound + options.required_continuous_clearance_m:
            certified_leaves += 1
            return None
        right_clearance = clearance(right)
        if right_clearance <= options.required_continuous_clearance_m:
            return "sampled_pose_collision_or_clearance_violation"
        if depth >= options.maximum_subdivision_depth:
            return "continuous_transition_uncertified_at_depth_limit"
        middle = midpoint(left, right)
        middle_clearance = clearance(middle)
        failure = certify(left, middle, left_clearance, depth + 1)
        if failure is not None:
            return failure
        return certify(middle, right, middle_clearance, depth + 1)

    if failure_reason is None:
        for segment_index in range(states.shape[0] - 1):
            left = states[segment_index]
            right = states[segment_index + 1]
            left_clearance = clearance(left)
            failure_reason = certify(left, right, left_clearance, 0)
            if failure_reason is not None:
                first_failed_segment = segment_index
                break

    reported_minimum_clearance = (
        minimum_clearance if math.isfinite(minimum_clearance) else None
    )
    return SweptPathAudit(
        passed=failure_reason is None,
        failure_reason=failure_reason,
        state_count=int(states.shape[0]),
        segment_count=int(states.shape[0] - 1),
        clearance_queries=clearance_queries,
        certified_leaf_segments=certified_leaves,
        deepest_subdivision=deepest,
        minimum_sampled_clearance_m=reported_minimum_clearance,
        start_position_error_m=start_position_error,
        start_angle_error_rad=start_angle_error,
        goal_position_error_m=goal_position_error,
        goal_angle_error_rad=goal_angle_error,
        total_negative_route_variation_m=total_negative,
        maximum_retreat_excursion_m=maximum_excursion,
        retreat_edge_count=retreat_edges,
        first_failed_segment=first_failed_segment,
    )


def decide_backtracking_required(
    unrestricted: PlannerRunEvidence,
    monotone: PlannerRunEvidence,
    path_audit: SweptPathAudit,
    *,
    gate_cfg: BacktrackingGateConfig | None = None,
) -> BacktrackingGateDecision:
    options = gate_cfg or BacktrackingGateConfig()
    options.validate()
    unrestricted.validate()
    monotone.validate()

    matched_fields = (
        "planner_family",
        "base_lattice_sha256",
        "geometry_sha256",
        "heuristic_sha256",
        "edge_cost_sha256",
        "collision_authority",
        "seed",
    )
    matched = all(getattr(unrestricted, name) == getattr(monotone, name) for name in matched_fields)
    gates = {
        "run_modes_are_unrestricted_then_monotone": (
            unrestricted.mode == "unrestricted" and monotone.mode == "monotone"
        ),
        "matched_search_provenance": matched,
        "unrestricted_goal_reached": (
            unrestricted.success and unrestricted.termination == "goal_reached"
        ),
        "unrestricted_path_continuously_certified": path_audit.passed,
        "monotone_goal_not_reached": not monotone.success,
        "monotone_open_set_exhausted": (
            not monotone.success
            and monotone.termination == "open_set_exhausted"
            and monotone.open_set_exhausted
            and not monotone.resource_limit_hit
        ),
        "real_retreat_excursion": (
            path_audit.maximum_retreat_excursion_m
            >= options.minimum_retreat_excursion_m
        ),
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    return BacktrackingGateDecision(not failed, failed, gates)


def _scenario_payload(scenario: PianoBatch) -> dict[str, Any]:
    _one_scenario(scenario)
    payload: dict[str, Any] = {}
    for name in (
        "state0",
        "goal",
        "wall_x",
        "gap_y",
        "gap_half",
        "source",
        "leg",
        "obstacle_rects",
    ):
        value = getattr(scenario, name)
        payload[name] = None if value is None else value.detach().cpu().tolist()
    return payload


def build_planner_scout_record(
    scenario: PianoBatch,
    cfg: EnvConfig,
    unrestricted_path: Tensor,
    unrestricted: PlannerRunEvidence,
    monotone: PlannerRunEvidence,
    *,
    gate_cfg: BacktrackingGateConfig | None = None,
) -> dict[str, Any]:
    """Build a self-hashed JSON-compatible map record."""

    options = gate_cfg or BacktrackingGateConfig()
    audit = audit_swept_pose_path(unrestricted_path, scenario, cfg, gate_cfg=options)
    decision = decide_backtracking_required(unrestricted, monotone, audit, gate_cfg=options)
    path = _pose_path(unrestricted_path).to(dtype=torch.float32)
    scenario_payload = _scenario_payload(scenario)
    env_payload = asdict(cfg)
    record: dict[str, Any] = {
        "schema_version": PLANNER_SCOUT_RECORD_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "eligible_for_training": decision.admitted,
        "failed_gates": list(decision.failed_gates),
        "gates": dict(decision.gates),
        "gate_config": asdict(options),
        "collision_protocol": {
            "authority": COLLISION_AUTHORITY,
            "path_interpolation": "piecewise-linear-xy-shortest-wrapped-angle",
            "certificate": "recursive-exact-SAT-clearance-vs-rigid-point-motion-bound",
            "depth_limit_fails_closed": True,
            "endpoint_only_validation_forbidden": True,
        },
        "environment": env_payload,
        "environment_sha256": canonical_json_sha256(env_payload),
        "scenario": scenario_payload,
        "scenario_sha256": canonical_json_sha256(scenario_payload),
        "unrestricted_run": asdict(unrestricted),
        "monotone_run": asdict(monotone),
        "unrestricted_path": path.tolist(),
        "unrestricted_path_shape": list(path.shape),
        "unrestricted_path_dtype": str(path.dtype),
        "unrestricted_path_sha256": tensor_sha256("unrestricted-path", path),
        "swept_path_audit": asdict(audit),
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def build_record_from_planner_scout(
    result: Any,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Strictly revalidate a :mod:`planner_scout` candidate.

    The broad scout intentionally uses a fast fixed-resolution sweep.  This
    adapter preserves its complete source record but makes the conservative
    recursive certificate in this module authoritative for training
    eligibility.  It accepts any ``ScoutResult``-compatible object to avoid a
    module import cycle.
    """

    from .planner_scout import env_config_for_case, scenario_for_states

    if result.unrestricted.status != "success" or len(result.unrestricted.path) < 2:
        raise ValueError("strict revalidation requires an unrestricted successful path")

    def termination(status: str) -> tuple[bool, str, bool, bool]:
        if status == "success":
            return True, "goal_reached", False, False
        if status == "unreachable":
            return False, "open_set_exhausted", True, False
        if status == "budget_exhausted":
            return False, "node_limit", False, True
        return False, status, False, False

    case_payload = asdict(result.case)
    grid_payload = asdict(result.grid)
    shared = {
        "planner_family": "planner-scout-matched-se2-lazysp-a-star-v2",
        "base_lattice_sha256": canonical_json_sha256(grid_payload),
        "geometry_sha256": canonical_json_sha256(case_payload),
        "heuristic_sha256": canonical_json_sha256(
            {
                "semantics": "euclidean-xy-wrapped-theta-scaled-v1",
                "rotation_cost_radius": result.grid.rotation_cost_radius,
            }
        ),
        "edge_cost_sha256": canonical_json_sha256(
            {
                "semantics": "matched-26-neighbour-euclidean-se2-v1",
                "x_step": result.grid.x_step,
                "y_step": result.grid.y_step,
                "theta_bins": result.grid.theta_bins,
                "rotation_cost_radius": result.grid.rotation_cost_radius,
                "diagonal_primitives": result.grid.diagonal_primitives,
            }
        ),
        "collision_authority": COLLISION_AUTHORITY,
        "seed": int(seed),
    }

    def evidence(mode: str, value: Any) -> PlannerRunEvidence:
        succeeded, ended, exhausted, limited = termination(value.status)
        return PlannerRunEvidence(
            mode=mode,
            success=succeeded,
            termination=ended,
            open_set_exhausted=exhausted,
            resource_limit_hit=limited,
            expanded_nodes=int(value.expansions),
            generated_nodes=(
                int(value.generated_nodes)
                if getattr(value, "generated_nodes", None) is not None
                else None
            ),
            **shared,
        )

    path = torch.tensor(result.unrestricted.path, dtype=torch.float32)
    state = torch.zeros(1, 6, dtype=torch.float32)
    scenario = scenario_for_states(state, result.case)
    cfg = env_config_for_case(result.case, device="cpu")
    options = BacktrackingGateConfig(
        minimum_retreat_excursion_m=float(result.gate.minimum_excursion),
        endpoint_position_tolerance_m=max(result.grid.x_step, result.grid.y_step),
        endpoint_angle_tolerance_rad=math.pi / result.grid.theta_bins + 1.0e-6,
        required_continuous_clearance_m=max(
            1.0e-4,
            float(result.gate.minimum_clearance),
        ),
        maximum_subdivision_depth=16,
    )
    record = build_planner_scout_record(
        scenario,
        cfg,
        path,
        evidence("unrestricted", result.unrestricted),
        evidence("monotone", result.monotone),
        gate_cfg=options,
    )
    record.pop("record_sha256")
    record["source_scout"] = {
        "record": result.to_record(),
        "record_sha256": canonical_json_sha256(result.to_record()),
        "fixed_resolution_sweep_is_advisory_only": True,
        "recursive_continuous_certificate_is_authoritative": True,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def verify_planner_scout_record(record: Mapping[str, Any]) -> None:
    """Fail closed before a scout record is allowed into training data."""

    value = dict(record)
    if value.get("schema_version") != PLANNER_SCOUT_RECORD_SCHEMA:
        raise ValueError("planner scout record schema mismatch")
    expected_hash = value.pop("record_sha256", None)
    if not isinstance(expected_hash, str) or canonical_json_sha256(value) != expected_hash:
        raise ValueError("planner scout record hash mismatch")
    if value.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("planner scout claim scope changed")
    if canonical_json_sha256(value.get("environment")) != value.get("environment_sha256"):
        raise ValueError("planner scout environment hash mismatch")
    if canonical_json_sha256(value.get("scenario")) != value.get("scenario_sha256"):
        raise ValueError("planner scout scenario hash mismatch")
    path = torch.tensor(value.get("unrestricted_path"), dtype=torch.float32)
    if list(path.shape) != value.get("unrestricted_path_shape"):
        raise ValueError("planner scout path shape mismatch")
    if tensor_sha256("unrestricted-path", path) != value.get("unrestricted_path_sha256"):
        raise ValueError("planner scout path hash mismatch")
    gates = value.get("gates")
    if not isinstance(gates, Mapping) or not gates or any(type(item) is not bool for item in gates.values()):
        raise ValueError("planner scout gates are malformed")
    failed = [name for name, passed in gates.items() if not passed]
    if failed != value.get("failed_gates"):
        raise ValueError("planner scout failed-gate list is inconsistent")
    eligible = bool(value.get("eligible_for_training"))
    if eligible != (not failed):
        raise ValueError("planner scout eligibility is inconsistent with gates")
    if eligible and not bool(value.get("swept_path_audit", {}).get("passed")):
        raise ValueError("uncertified path cannot be eligible for training")


def write_planner_scout_record(path: Path, record: Mapping[str, Any]) -> None:
    verify_planner_scout_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize_scout_family(
    records: list[Mapping[str, Any]],
    *,
    minimum_maps: int = 128,
    minimum_unrestricted_success_rate: float = 0.90,
    maximum_monotone_success_rate: float = 0.20,
) -> dict[str, Any]:
    """Apply the pretraining family gate proposed for the planning scout."""

    if minimum_maps <= 0:
        raise ValueError("minimum_maps must be positive")
    if not 0.0 <= minimum_unrestricted_success_rate <= 1.0:
        raise ValueError("minimum_unrestricted_success_rate must be in [0, 1]")
    if not 0.0 <= maximum_monotone_success_rate <= 1.0:
        raise ValueError("maximum_monotone_success_rate must be in [0, 1]")
    for record in records:
        verify_planner_scout_record(record)
    count = len(records)
    unrestricted_successes = sum(
        bool(record["unrestricted_run"]["success"]) for record in records
    )
    monotone_successes = sum(bool(record["monotone_run"]["success"]) for record in records)
    eligible_maps = sum(bool(record["eligible_for_training"]) for record in records)
    unrestricted_rate = unrestricted_successes / count if count else 0.0
    monotone_rate = monotone_successes / count if count else 1.0
    gates = {
        "enough_maps": count >= minimum_maps,
        "unrestricted_success_rate": unrestricted_rate >= minimum_unrestricted_success_rate,
        "monotone_success_rate": monotone_rate <= maximum_monotone_success_rate,
        "at_least_one_strict_backtracking_map": eligible_maps > 0,
    }
    failed = [name for name, passed in gates.items() if not passed]
    report: dict[str, Any] = {
        "schema_version": PLANNER_SCOUT_FAMILY_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "passed": not failed,
        "failed_gates": failed,
        "gates": gates,
        "map_count": count,
        "eligible_backtracking_map_count": eligible_maps,
        "unrestricted_success_count": unrestricted_successes,
        "unrestricted_success_rate": unrestricted_rate,
        "monotone_success_count": monotone_successes,
        "monotone_success_rate": monotone_rate,
        "thresholds": {
            "minimum_maps": minimum_maps,
            "minimum_unrestricted_success_rate": minimum_unrestricted_success_rate,
            "maximum_monotone_success_rate": maximum_monotone_success_rate,
        },
        "ordered_record_sha256": [record["record_sha256"] for record in records],
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report
