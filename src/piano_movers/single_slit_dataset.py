"""Fail-closed recurrent data for the 0.720 m single-slit skill.

The executable-teacher artifact is a dynamics trajectory, not merely a
kinematic path.  This module verifies that artifact against ``step_dynamics``
and the exact compound-payload SAT authority before constructing any data.

Perturbed rows are sampled on the connected, exact-SAT admissible *vertical
fiber* through a nominal pose.  This is deliberate: isotropic pose noise is
not meaningful in a slit with millimetre clearance.  A perturbed state is
admitted only when a bounded inverse-dynamics teacher executes its complete
remaining suffix, every real transition is collision-free, the first
off-nominal continuous segment has a conservative recursive SAT certificate,
and the rollout reaches the artifact's terminal state.  Failed attempts are
counted, never clipped into the corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .cem_teacher import feasible_forces_for_absolute_wrench
from .config import EnvConfig
from .env import (
    collision_clearance_margin,
    step_dynamics,
    transition_clearance_margin,
    transition_collision_mask,
    wrap_angle,
)
from .payload_shapes import payload_rectangles_for_config
from .single_slit_feedback import single_slit_scenario


SINGLE_SLIT_DATASET_SCHEMA = "piano-movers-single-slit-recurrent-corpus-v1"
EXECUTABLE_TEACHER_SCHEMA = "piano-movers-single-slit-executable-teacher-v1"

STRATUM_NOMINAL = 0
STRATUM_TUBE_CORE = 1
STRATUM_WALL_NEAR = 2
STRATUM_NEAR_JAM = 3
STRATUM_NAMES = ("nominal", "tube_core", "wall_near", "near_jam")


@dataclass(frozen=True)
class SingleSlitCorpusConfig:
    """Sampling and admission controls for a recurrent corpus."""

    sequence_length: int = 22
    burn_in_steps: int = 6
    nominal_rows: int = 16
    tube_core_rows: int = 32
    wall_near_rows: int = 32
    near_jam_rows: int = 32
    # The admissible fiber is eroded before normalized rho is sampled.  Thus a
    # rho close to one approaches this positive margin rather than contact.
    fiber_erosion_m: float = 2.0e-4
    fiber_search_limit_m: float = 5.0e-2
    fiber_scan_steps: int = 128
    fiber_bisection_steps: int = 24
    tube_core_rho: tuple[float, float] = (0.05, 0.40)
    wall_near_rho: tuple[float, float] = (0.65, 0.85)
    near_jam_rho: tuple[float, float] = (0.90, 0.98)
    near_jam_clearance_m: tuple[float, float] = (1.5e-4, 5.0e-4)
    vertical_velocity_cap_mps: float = 3.0e-3
    angular_velocity_cap_radps: float = 5.0e-3
    # Exact one-step pose tracking is what makes the off-nominal branch join
    # the already continuously certified nominal suffix.  Even a tiny
    # allocator saturation invalidates that argument, so the default is
    # deliberately exact rather than "nearly one".
    minimum_allocator_scale: float = 1.0
    minimum_continuous_clearance_m: float = 5.0e-5
    continuous_certificate_depth: int = 20
    terminal_state_tolerance: float = 2.0e-5
    dynamics_tolerance: float = 2.0e-6
    maximum_attempts_per_row: int = 256

    @property
    def row_counts(self) -> tuple[int, int, int, int]:
        return (
            self.nominal_rows,
            self.tube_core_rows,
            self.wall_near_rows,
            self.near_jam_rows,
        )

    @property
    def total_rows(self) -> int:
        return sum(self.row_counts)

    def validate(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if not 0 < self.burn_in_steps < self.sequence_length:
            raise ValueError("burn_in_steps must lie strictly inside the sequence")
        if any(isinstance(value, bool) or value < 0 for value in self.row_counts):
            raise ValueError("all stratum row counts must be non-negative integers")
        if self.total_rows <= 0:
            raise ValueError("at least one corpus row is required")
        positive = (
            self.fiber_erosion_m,
            self.fiber_search_limit_m,
            self.vertical_velocity_cap_mps,
            self.angular_velocity_cap_radps,
            self.minimum_continuous_clearance_m,
            self.terminal_state_tolerance,
            self.dynamics_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("corpus scales and tolerances must be finite and positive")
        if self.minimum_continuous_clearance_m >= self.fiber_erosion_m:
            raise ValueError("continuous clearance must be smaller than fiber erosion")
        if self.fiber_scan_steps < 2 or self.fiber_bisection_steps < 1:
            raise ValueError("fiber search needs at least two scan and one bisection step")
        if self.continuous_certificate_depth < 1:
            raise ValueError("continuous certificate depth must be positive")
        if self.maximum_attempts_per_row < 1:
            raise ValueError("maximum_attempts_per_row must be positive")
        if not 0.0 < self.minimum_allocator_scale <= 1.0:
            raise ValueError("minimum_allocator_scale must lie in (0,1]")
        for name, bounds in (
            ("tube_core_rho", self.tube_core_rho),
            ("wall_near_rho", self.wall_near_rho),
            ("near_jam_rho", self.near_jam_rho),
        ):
            if len(bounds) != 2 or not 0.0 <= bounds[0] < bounds[1] < 1.0:
                raise ValueError(f"{name} must be an increasing subinterval of [0,1)")
        if not (
            0.0
            < self.near_jam_clearance_m[0]
            < self.near_jam_clearance_m[1]
        ):
            raise ValueError("near-jam clearance band must be finite and increasing")


@dataclass(frozen=True)
class ExecutableTeacherArtifact:
    """Validated tensors from the executable inverse-dynamics teacher."""

    path: Path
    sha256: str
    reference_poses: Tensor
    states: Tensor
    forces: Tensor
    realized_wrenches: Tensor
    swept_clearances: Tensor
    continuous_clearance_m: float

    @property
    def horizon(self) -> int:
        return int(self.forces.shape[0])


@dataclass(frozen=True)
class SingleSlitCorpus:
    """Dense, padding-free recurrent windows.

    Every leading dimension is ``[rows, time]``.  ``stratum`` is repeated on
    the time axis so every exported transition remains self-describing after
    flattening or sharding.
    """

    state: Tensor
    next_state: Tensor
    forces: Tensor
    semantic8: Tensor
    path_index: Tensor
    clearance: Tensor
    stratum: Tensor
    anchor_index: Tensor
    initial_y_offset: Tensor
    source_artifact_sha256: str
    rejected_attempts: Mapping[str, int]

    def validate(self, cfg: EnvConfig, *, tolerance: float = 3.0e-6) -> None:
        if self.state.ndim != 3 or self.state.shape[-1] != 6:
            raise ValueError("state must have shape [rows,time,6]")
        prefix = self.state.shape[:2]
        expected = {
            "next_state": (self.next_state, (*prefix, 6)),
            "forces": (self.forces, (*prefix, cfg.n_agents, 2)),
            "semantic8": (self.semantic8, (*prefix, 8)),
            "path_index": (self.path_index, prefix),
            "clearance": (self.clearance, prefix),
            "stratum": (self.stratum, prefix),
        }
        for name, (value, shape) in expected.items():
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
        if self.anchor_index.shape != (prefix[0],):
            raise ValueError("anchor_index must have shape [rows]")
        if self.initial_y_offset.shape != (prefix[0],):
            raise ValueError("initial_y_offset must have shape [rows]")
        floats = (
            self.state,
            self.next_state,
            self.forces,
            self.semantic8,
            self.clearance,
            self.initial_y_offset,
        )
        if any(not bool(torch.isfinite(value).all()) for value in floats):
            raise ValueError("corpus contains non-finite values")
        if bool((self.clearance <= 0.0).any()):
            raise ValueError("corpus contains a non-positive exact-SAT clearance")
        valid_strata = (self.stratum >= 0) & (self.stratum < len(STRATUM_NAMES))
        if not bool(valid_strata.all()):
            raise ValueError("corpus contains an unknown stratum id")
        if not bool((self.stratum == self.stratum[:, :1]).all()):
            raise ValueError("stratum must be constant within each recurrent row")
        expected_indices = self.path_index[:, :1] + torch.arange(
            prefix[1], device=self.path_index.device, dtype=self.path_index.dtype
        )
        if not torch.equal(self.path_index, expected_indices):
            raise ValueError("path_index must increase by one on every real transition")

        flat_state = self.state.reshape(-1, 6)
        flat_next = self.next_state.reshape(-1, 6)
        flat_forces = self.forces.reshape(-1, cfg.n_agents, 2)
        replay = step_dynamics(flat_state, flat_forces, cfg)
        if float((replay - flat_next).abs().amax()) > tolerance:
            raise ValueError("stored next_state does not equal actual step_dynamics")
        scenario = single_slit_scenario(flat_state.shape[0], cfg, initial_state=flat_state)
        hit = transition_collision_mask(flat_state, flat_next, scenario, cfg)
        exact_clearance = transition_clearance_margin(flat_state, flat_next, scenario, cfg)
        if not torch.equal(hit, exact_clearance <= 0.0):
            raise RuntimeError("boolean and signed exact-SAT authorities disagree")
        if bool(hit.any()):
            raise ValueError("corpus contains an exact-SAT collision")
        if float((exact_clearance - self.clearance.reshape(-1)).abs().amax()) > tolerance:
            raise ValueError("stored clearance does not match exact transition clearance")

    def save(self, path: Path | str, metadata: Mapping[str, Any]) -> tuple[Path, Path]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "state": self.state.detach().cpu().numpy(),
            "next_state": self.next_state.detach().cpu().numpy(),
            "forces": self.forces.detach().cpu().numpy(),
            "semantic8": self.semantic8.detach().cpu().numpy(),
            "path_index": self.path_index.detach().cpu().numpy(),
            "clearance": self.clearance.detach().cpu().numpy(),
            "stratum": self.stratum.detach().cpu().numpy(),
            "anchor_index": self.anchor_index.detach().cpu().numpy(),
            "initial_y_offset": self.initial_y_offset.detach().cpu().numpy(),
            "stratum_names": np.asarray(STRATUM_NAMES),
        }
        np.savez_compressed(target, **arrays)
        sidecar = target.with_suffix(".json")
        record = dict(metadata)
        record.update(
            {
                "schema_version": SINGLE_SLIT_DATASET_SCHEMA,
                "dataset": str(target),
                "source_artifact_sha256": self.source_artifact_sha256,
                "rejected_attempts": dict(self.rejected_attempts),
                "array_schema": {
                    "state": ["rows", "time", 6],
                    "next_state": ["rows", "time", 6],
                    "forces": ["rows", "time", 12, 2],
                    "semantic8": ["rows", "time", 8],
                    "path_index": ["rows", "time"],
                    "clearance": ["rows", "time"],
                    "stratum": ["rows", "time"],
                    "anchor_index": ["rows"],
                    "initial_y_offset": ["rows"],
                },
            }
        )
        sidecar.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return target, sidecar


@torch.no_grad()
def load_single_slit_corpus(
    path: Path | str,
    cfg: EnvConfig,
    *,
    validate: bool = True,
) -> SingleSlitCorpus:
    """Load a saved corpus on ``cfg.device`` and recheck its hard contracts."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    sidecar = target.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError("single-slit corpus is missing its JSON audit sidecar")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if record.get("schema_version") != SINGLE_SLIT_DATASET_SCHEMA:
        raise ValueError("unexpected single-slit corpus schema")
    source_sha = record.get("source_artifact_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("corpus report omitted the source artifact SHA-256")
    rejected = record.get("rejected_attempts", {})
    if not isinstance(rejected, Mapping):
        raise ValueError("corpus rejected_attempts must be a mapping")
    device = torch.device(cfg.device)
    with np.load(target) as arrays:
        required = (
            "state",
            "next_state",
            "forces",
            "semantic8",
            "path_index",
            "clearance",
            "stratum",
            "anchor_index",
            "initial_y_offset",
            "stratum_names",
        )
        missing = [name for name in required if name not in arrays.files]
        if missing:
            raise ValueError(f"single-slit corpus omitted arrays: {missing}")
        names = tuple(str(value) for value in arrays["stratum_names"].tolist())
        if names != STRATUM_NAMES:
            raise ValueError("single-slit corpus stratum dictionary changed")

        def floating(name: str) -> Tensor:
            return torch.as_tensor(
                np.asarray(arrays[name]), dtype=torch.float32, device=device
            ).clone()

        def integer(name: str, dtype: torch.dtype) -> Tensor:
            return torch.as_tensor(np.asarray(arrays[name]), dtype=dtype, device=device).clone()

        corpus = SingleSlitCorpus(
            state=floating("state"),
            next_state=floating("next_state"),
            forces=floating("forces"),
            semantic8=floating("semantic8"),
            path_index=integer("path_index", torch.long),
            clearance=floating("clearance"),
            stratum=integer("stratum", torch.int8),
            anchor_index=integer("anchor_index", torch.long),
            initial_y_offset=floating("initial_y_offset"),
            source_artifact_sha256=source_sha,
            rejected_attempts={str(key): int(value) for key, value in rejected.items()},
        )
    if validate:
        tolerance = float(record.get("generator", {}).get("dynamics_tolerance", 3.0e-6))
        corpus.validate(cfg, tolerance=tolerance)
    return corpus


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _companion_report(path: Path) -> Path:
    suffix = ".rollout.npz"
    if not path.name.endswith(suffix):
        raise ValueError("executable teacher artifact must end in .rollout.npz")
    return path.with_name(path.name[: -len(suffix)] + ".json")


@torch.no_grad()
def load_executable_teacher_artifact(
    path: Path | str,
    cfg: EnvConfig,
    *,
    dynamics_tolerance: float = 2.0e-6,
) -> ExecutableTeacherArtifact:
    """Load and independently replay the nominal teacher artifact."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    report_path = _companion_report(target)
    if not report_path.is_file():
        raise ValueError("teacher artifact is missing its continuous-audit JSON sidecar")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != EXECUTABLE_TEACHER_SCHEMA:
        raise ValueError("unexpected executable-teacher schema")
    audit = report.get("rollout", {}).get("official_continuous_audit", {})
    continuous_clearance = float(audit.get("minimum_sampled_clearance_m", -math.inf))
    if audit.get("passed") is not True or not continuous_clearance > 0.0:
        raise ValueError("teacher artifact lacks a positive continuous exact-SAT certificate")

    with np.load(target) as source:
        required = (
            "reference_poses",
            "states",
            "forces",
            "realized_wrenches",
            "swept_clearances",
        )
        missing = [name for name in required if name not in source.files]
        if missing:
            raise ValueError(f"teacher artifact is missing arrays: {missing}")
        arrays = {
            name: torch.as_tensor(
                np.asarray(source[name]), dtype=torch.float32, device=torch.device(cfg.device)
            ).clone()
            for name in required
        }

    states = arrays["states"]
    forces = arrays["forces"]
    references = arrays["reference_poses"]
    wrenches = arrays["realized_wrenches"]
    clearances = arrays["swept_clearances"]
    horizon = int(forces.shape[0])
    shapes = {
        "states": (horizon + 1, 6),
        "reference_poses": (horizon + 1, 3),
        "forces": (horizon, cfg.n_agents, 2),
        "realized_wrenches": (horizon, 3),
        "swept_clearances": (horizon,),
    }
    for name, shape in shapes.items():
        if tuple(arrays[name].shape) != shape:
            raise ValueError(f"teacher {name} has shape {tuple(arrays[name].shape)}, expected {shape}")
        if not bool(torch.isfinite(arrays[name]).all()):
            raise ValueError(f"teacher {name} contains non-finite values")
    if horizon < 2:
        raise ValueError("teacher trajectory is too short")

    replay = step_dynamics(states[:-1], forces, cfg)
    replay_error = float((replay - states[1:]).abs().amax())
    if replay_error > dynamics_tolerance:
        raise ValueError(f"teacher artifact fails dynamics replay: {replay_error}")
    scenario = single_slit_scenario(horizon, cfg, initial_state=states[:-1])
    hit = transition_collision_mask(states[:-1], states[1:], scenario, cfg)
    exact_clearance = transition_clearance_margin(states[:-1], states[1:], scenario, cfg)
    if not torch.equal(hit, exact_clearance <= 0.0):
        raise RuntimeError("teacher boolean and signed exact-SAT authorities disagree")
    if bool(hit.any()) or bool((exact_clearance <= 0.0).any()):
        raise ValueError("teacher artifact contains a colliding transition")
    if float((exact_clearance - clearances).abs().amax()) > 3.0e-6:
        raise ValueError("teacher stored swept clearance disagrees with exact SAT replay")
    if float((references - states[:, :3]).abs().amax()) > dynamics_tolerance:
        raise ValueError("teacher reference poses do not equal realized poses")
    if float(torch.linalg.vector_norm(forces, dim=-1).amax()) > cfg.max_force + 1.0e-5:
        raise ValueError("teacher artifact violates the per-handle force bound")

    return ExecutableTeacherArtifact(
        path=target,
        sha256=_file_sha256(target),
        reference_poses=references,
        states=states,
        forces=forces,
        realized_wrenches=wrenches,
        swept_clearances=exact_clearance,
        continuous_clearance_m=continuous_clearance,
    )


def inverse_dynamics_wrench(
    state: Tensor,
    target_pose: Tensor,
    cfg: EnvConfig,
    *,
    brake: bool,
) -> tuple[Tensor, Tensor]:
    """Return one-step wrench and the corresponding desired world twist."""

    if state.shape != (1, 6) or target_pose.shape != (3,):
        raise ValueError("inverse dynamics expects state [1,6] and target_pose [3]")
    linear_decay = 1.0 - cfg.linear_damping * cfg.dt
    angular_decay = 1.0 - cfg.angular_damping * cfg.dt
    if brake:
        desired_twist = state.new_zeros(3)
    else:
        desired_twist = torch.cat(
            (
                (target_pose[:2] - state[0, :2]) / cfg.dt,
                (wrap_angle(target_pose[2] - state[0, 2]) / cfg.dt).view(1),
            )
        )
    wrench = state.new_empty((1, 3))
    wrench[0, :2] = (
        cfg.mass
        * (desired_twist[:2] / linear_decay - state[0, 3:5])
        / cfg.dt
    )
    wrench[0, 2] = (
        cfg.inertia
        * (desired_twist[2] / angular_decay - state[0, 5])
        / cfg.dt
    )
    return wrench, desired_twist


def _semantic8(
    state: Tensor,
    desired_twist: Tensor,
    realized_wrench: Tensor,
    artifact: ExecutableTeacherArtifact,
    path_index: int,
    cfg: EnvConfig,
) -> Tensor:
    """Build twist3,wrench3,cross-track,ordered-progress supervision."""

    horizon = artifact.horizon
    left = max(path_index - 1, 0)
    right = min(path_index + 1, horizon)
    tangent = artifact.reference_poses[right, :2] - artifact.reference_poses[left, :2]
    norm = torch.linalg.vector_norm(tangent)
    if float(norm) <= 1.0e-8:
        tangent = state.new_tensor((1.0, 0.0))
    else:
        tangent = tangent / norm
    offset = state[0, :2] - artifact.reference_poses[path_index, :2]
    cross_track = (tangent[0] * offset[1] - tangent[1] * offset[0]).clamp(-1.0, 1.0)
    twist_scale = state.new_tensor((0.65, 0.65, 1.40))
    wrench_scale = state.new_tensor(
        (cfg.n_agents * cfg.max_force, cfg.n_agents * cfg.max_force, 0.75 * cfg.n_agents * cfg.max_force)
    )
    return torch.cat(
        (
            desired_twist / twist_scale,
            realized_wrench[0] / wrench_scale,
            cross_track.view(1),
            state.new_tensor((path_index / float(horizon),)),
        )
    )


def _continuous_segment_certified(
    left: Tensor,
    right: Tensor,
    cfg: EnvConfig,
    *,
    required_clearance: float,
    maximum_depth: int,
) -> tuple[bool, float]:
    """Conservatively certify one off-nominal SE(2) segment.

    Exact SAT clearances are combined with a payload-point motion bound.  A
    returned ``True`` proves the full continuous segment remains farther than
    ``required_clearance`` from collision, up to floating-point evaluation.
    """

    radius = float(
        torch.linalg.vector_norm(
            payload_rectangles_for_config(cfg, left.device).reshape(-1, 2),
            dim=-1,
        ).amax()
    )
    minimum_sampled = math.inf

    def clearance(pose: Tensor) -> float:
        nonlocal minimum_sampled
        state = left.new_zeros((1, 6))
        state[0, :3] = pose
        scenario = single_slit_scenario(1, cfg, initial_state=state)
        value = float(collision_clearance_margin(state, scenario, cfg)[0])
        minimum_sampled = min(minimum_sampled, value)
        return value

    def midpoint(a: Tensor, b: Tensor) -> Tensor:
        pose = 0.5 * (a + b)
        delta = wrap_angle(b[2] - a[2])
        pose[2] = wrap_angle(a[2] + 0.5 * delta)
        return pose

    def recurse(a: Tensor, b: Tensor, a_clearance: float, depth: int) -> bool:
        delta_angle = abs(float(wrap_angle(b[2] - a[2])))
        motion = float(torch.linalg.vector_norm(b[:2] - a[:2])) + radius * delta_angle
        if a_clearance > motion + required_clearance:
            return True
        b_clearance = clearance(b)
        if min(a_clearance, b_clearance) <= required_clearance or depth >= maximum_depth:
            return False
        middle = midpoint(a, b)
        middle_clearance = clearance(middle)
        return recurse(a, middle, a_clearance, depth + 1) and recurse(
            middle, b, middle_clearance, depth + 1
        )

    first = clearance(left[:3])
    passed = recurse(left[:3], right[:3], first, 0)
    return passed, minimum_sampled


def certify_continuous_se2_segment(
    left: Tensor,
    right: Tensor,
    cfg: EnvConfig,
    *,
    required_clearance: float = 5.0e-5,
    maximum_depth: int = 20,
) -> tuple[bool, float]:
    """Public fail-closed wrapper for recursive continuous SE(2) clearance.

    This is used by parameterized-teacher admission to prove that an
    off-nominal feedback transition rejoins an already certified suffix.  The
    implementation remains shared with recurrent-corpus admission so the two
    gates cannot silently adopt different collision semantics.
    """

    if left.shape != (6,) or right.shape != (6,):
        raise ValueError("continuous SE(2) segment endpoints must be state6 vectors")
    if not math.isfinite(required_clearance) or required_clearance <= 0.0:
        raise ValueError("required continuous clearance must be finite and positive")
    if maximum_depth <= 0:
        raise ValueError("maximum continuous-certificate depth must be positive")
    return _continuous_segment_certified(
        left,
        right,
        cfg,
        required_clearance=required_clearance,
        maximum_depth=maximum_depth,
    )


def _fiber_extent(
    nominal_state: Tensor,
    direction: int,
    cfg: EnvConfig,
    options: SingleSlitCorpusConfig,
) -> float:
    """Find the connected safe y-fiber extent in one direction."""

    if direction not in (-1, 1):
        raise ValueError("fiber direction must be -1 or +1")
    distances = torch.linspace(
        0.0,
        options.fiber_search_limit_m,
        options.fiber_scan_steps + 1,
        device=nominal_state.device,
        dtype=nominal_state.dtype,
    )
    probes = nominal_state.expand(distances.shape[0], -1).clone()
    probes[:, 1] += direction * distances
    scenario = single_slit_scenario(probes.shape[0], cfg, initial_state=probes)
    margins = collision_clearance_margin(probes, scenario, cfg)
    unsafe = margins <= options.fiber_erosion_m
    if bool(unsafe[0]):
        return 0.0
    indices = torch.nonzero(unsafe, as_tuple=False)
    if indices.numel() == 0:
        return options.fiber_search_limit_m
    upper_index = int(indices[0, 0])
    lower = float(distances[upper_index - 1])
    upper = float(distances[upper_index])
    for _ in range(options.fiber_bisection_steps):
        middle = 0.5 * (lower + upper)
        probe = nominal_state.clone()
        probe[1] += direction * middle
        scenario = single_slit_scenario(1, cfg, initial_state=probe.view(1, 6))
        margin = float(collision_clearance_margin(probe.view(1, 6), scenario, cfg)[0])
        if margin > options.fiber_erosion_m:
            lower = middle
        else:
            upper = middle
    return lower


@dataclass(frozen=True)
class _Continuation:
    state: Tensor
    next_state: Tensor
    forces: Tensor
    semantic8: Tensor
    path_index: Tensor
    clearance: Tensor


@torch.no_grad()
def _rollout_complete_continuation(
    initial_state: Tensor,
    anchor: int,
    artifact: ExecutableTeacherArtifact,
    cfg: EnvConfig,
    options: SingleSlitCorpusConfig,
) -> tuple[_Continuation | None, str | None]:
    """Roll out the complete remaining inverse-dynamics teacher suffix."""

    current = initial_state.view(1, 6).clone()
    scenario = single_slit_scenario(1, cfg, initial_state=current)
    if float(collision_clearance_margin(current, scenario, cfg)[0]) <= options.fiber_erosion_m:
        return None, "initial_clearance"
    states: list[Tensor] = []
    next_states: list[Tensor] = []
    forces_log: list[Tensor] = []
    semantic_log: list[Tensor] = []
    index_log: list[int] = []
    clearance_log: list[Tensor] = []
    first_transition: tuple[Tensor, Tensor] | None = None
    for path_index in range(anchor, artifact.horizon):
        requested, desired_twist = inverse_dynamics_wrench(
            current,
            artifact.reference_poses[path_index + 1],
            cfg,
            brake=path_index == artifact.horizon - 1,
        )
        forces, realized, scale = feasible_forces_for_absolute_wrench(current, requested, cfg)
        if float(scale[0]) < options.minimum_allocator_scale:
            return None, "allocator_scale"
        proposed = step_dynamics(current, forces, cfg)
        pose_error = torch.cat(
            (
                proposed[0, :2] - artifact.reference_poses[path_index + 1, :2],
                wrap_angle(
                    proposed[0, 2] - artifact.reference_poses[path_index + 1, 2]
                ).view(1),
            )
        ).abs().amax()
        if float(pose_error) > options.dynamics_tolerance:
            return None, "inverse_pose_tracking"
        hit = transition_collision_mask(current, proposed, scenario, cfg)
        clearance = transition_clearance_margin(current, proposed, scenario, cfg)
        if not torch.equal(hit, clearance <= 0.0):
            raise RuntimeError("continuation boolean and signed SAT authorities disagree")
        if bool(hit[0]) or float(clearance[0]) <= 0.0:
            return None, "transition_collision"
        if first_transition is None:
            first_transition = (current[0].clone(), proposed[0].clone())
        states.append(current[0].clone())
        next_states.append(proposed[0].clone())
        forces_log.append(forces[0].clone())
        semantic_log.append(
            _semantic8(current, desired_twist, realized, artifact, path_index, cfg)
        )
        index_log.append(path_index)
        clearance_log.append(clearance[0].clone())
        current = proposed

    if first_transition is None:
        return None, "empty_continuation"
    continuous, _ = _continuous_segment_certified(
        first_transition[0],
        first_transition[1],
        cfg,
        required_clearance=options.minimum_continuous_clearance_m,
        maximum_depth=options.continuous_certificate_depth,
    )
    if not continuous:
        return None, "continuous_first_transition"
    terminal_error = float((current[0] - artifact.states[-1]).abs().amax())
    if terminal_error > options.terminal_state_tolerance:
        return None, "terminal_state"
    return (
        _Continuation(
            state=torch.stack(states),
            next_state=torch.stack(next_states),
            forces=torch.stack(forces_log),
            semantic8=torch.stack(semantic_log),
            path_index=torch.tensor(index_log, dtype=torch.long, device=current.device),
            clearance=torch.stack(clearance_log),
        ),
        None,
    )


def _anchor_distribution(
    artifact: ExecutableTeacherArtifact,
    minimum_anchor: int,
    maximum_anchor: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Inverse clearance emphasizes the actual bottleneck while retaining full
    # path coverage.  The 1 mm floor prevents a single tick from monopolizing
    # the corpus.
    anchors = np.arange(minimum_anchor, maximum_anchor + 1, dtype=np.int64)
    clearance = artifact.swept_clearances[minimum_anchor : maximum_anchor + 1].detach().cpu().numpy()
    weights = 1.0 / np.maximum(clearance, 1.0e-3)
    weights = 0.25 * np.full_like(weights, weights.mean()) + 0.75 * weights
    return anchors, weights / weights.sum()


def _nominal_context(
    start: int,
    stop: int,
    artifact: ExecutableTeacherArtifact,
    cfg: EnvConfig,
) -> _Continuation:
    """Return audited nominal transitions in ``[start, stop)``."""

    semantic: list[Tensor] = []
    for path_index in range(start, stop):
        _, desired_twist = inverse_dynamics_wrench(
            artifact.states[path_index : path_index + 1],
            artifact.reference_poses[path_index + 1],
            cfg,
            brake=path_index == artifact.horizon - 1,
        )
        semantic.append(
            _semantic8(
                artifact.states[path_index : path_index + 1],
                desired_twist,
                artifact.realized_wrenches[path_index : path_index + 1],
                artifact,
                path_index,
                cfg,
            )
        )
    return _Continuation(
        state=artifact.states[start:stop].clone(),
        next_state=artifact.states[start + 1 : stop + 1].clone(),
        forces=artifact.forces[start:stop].clone(),
        semantic8=torch.stack(semantic),
        path_index=torch.arange(start, stop, dtype=torch.long, device=artifact.states.device),
        clearance=artifact.swept_clearances[start:stop].clone(),
    )


def _sample_initial_state(
    stratum: int,
    anchor: int,
    artifact: ExecutableTeacherArtifact,
    cfg: EnvConfig,
    options: SingleSlitCorpusConfig,
    rng: np.random.Generator,
) -> tuple[Tensor, float] | None:
    nominal = artifact.states[anchor].clone()
    if stratum == STRATUM_NOMINAL:
        return nominal, 0.0
    rho_bounds = {
        STRATUM_TUBE_CORE: options.tube_core_rho,
        STRATUM_WALL_NEAR: options.wall_near_rho,
        STRATUM_NEAR_JAM: options.near_jam_rho,
    }[stratum]
    direction = -1 if rng.integers(0, 2) == 0 else 1
    extent = _fiber_extent(nominal, direction, cfg, options)
    if extent <= 0.0:
        return None
    rho = float(rng.uniform(*rho_bounds))
    offset = direction * rho * extent
    candidate = nominal.clone()
    candidate[1] += offset
    # Velocity noise is structured in the same vertical maneuver coordinate,
    # rather than sampled isotropically in SE(2).  It tests feedback without
    # changing the certified pose fiber.
    candidate[4] += float(
        rng.uniform(-options.vertical_velocity_cap_mps, options.vertical_velocity_cap_mps)
    )
    candidate[5] += float(
        rng.uniform(-options.angular_velocity_cap_radps, options.angular_velocity_cap_radps)
    )
    scenario = single_slit_scenario(1, cfg, initial_state=candidate.view(1, 6))
    pose_clearance = float(
        collision_clearance_margin(candidate.view(1, 6), scenario, cfg)[0]
    )
    if pose_clearance <= options.fiber_erosion_m:
        return None
    if stratum == STRATUM_NEAR_JAM:
        lower, upper = options.near_jam_clearance_m
        if not lower <= pose_clearance <= upper:
            return None
    return candidate, offset


@torch.no_grad()
def generate_single_slit_corpus(
    artifact: ExecutableTeacherArtifact,
    cfg: EnvConfig,
    options: SingleSlitCorpusConfig | None = None,
    *,
    seed: int = 0,
) -> SingleSlitCorpus:
    """Generate exact-sized recurrent strata or fail without a partial set."""

    protocol = options or SingleSlitCorpusConfig()
    protocol.validate()
    if protocol.sequence_length > artifact.horizon:
        raise ValueError("sequence_length exceeds executable teacher horizon")
    supervised_steps = protocol.sequence_length - protocol.burn_in_steps
    minimum_anchor = protocol.burn_in_steps
    maximum_anchor = artifact.horizon - supervised_steps
    anchor_values, anchor_probabilities = _anchor_distribution(
        artifact, minimum_anchor, maximum_anchor
    )
    rng = np.random.default_rng(seed)
    rows: list[_Continuation] = []
    row_strata: list[int] = []
    anchors: list[int] = []
    offsets: list[float] = []
    rejected: dict[str, int] = {}

    for stratum, count in enumerate(protocol.row_counts):
        for _ in range(count):
            admitted: tuple[_Continuation, int, float] | None = None
            for _attempt in range(protocol.maximum_attempts_per_row):
                anchor = int(rng.choice(anchor_values, p=anchor_probabilities))
                sampled = _sample_initial_state(
                    stratum, anchor, artifact, cfg, protocol, rng
                )
                if sampled is None:
                    rejected["fiber_or_stratum"] = rejected.get("fiber_or_stratum", 0) + 1
                    continue
                initial, offset = sampled
                continuation, reason = _rollout_complete_continuation(
                    initial, anchor, artifact, cfg, protocol
                )
                if continuation is None:
                    assert reason is not None
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                admitted = (continuation, anchor, offset)
                break
            if admitted is None:
                raise RuntimeError(
                    f"failed to admit {STRATUM_NAMES[stratum]} row after "
                    f"{protocol.maximum_attempts_per_row} attempts; rejected={rejected}"
                )
            continuation, anchor, offset = admitted
            context = _nominal_context(
                anchor - protocol.burn_in_steps,
                anchor,
                artifact,
                cfg,
            )
            rows.append(
                _Continuation(
                    state=torch.cat(
                        (context.state, continuation.state[:supervised_steps]), dim=0
                    ),
                    next_state=torch.cat(
                        (context.next_state, continuation.next_state[:supervised_steps]), dim=0
                    ),
                    forces=torch.cat(
                        (context.forces, continuation.forces[:supervised_steps]), dim=0
                    ),
                    semantic8=torch.cat(
                        (context.semantic8, continuation.semantic8[:supervised_steps]), dim=0
                    ),
                    path_index=torch.cat(
                        (context.path_index, continuation.path_index[:supervised_steps]), dim=0
                    ),
                    clearance=torch.cat(
                        (context.clearance, continuation.clearance[:supervised_steps]), dim=0
                    ),
                )
            )
            row_strata.append(stratum)
            anchors.append(anchor)
            offsets.append(offset)

    device = artifact.states.device
    time_stratum = torch.tensor(row_strata, dtype=torch.int8, device=device)[:, None].expand(
        -1, protocol.sequence_length
    ).clone()
    corpus = SingleSlitCorpus(
        state=torch.stack([row.state for row in rows]),
        next_state=torch.stack([row.next_state for row in rows]),
        forces=torch.stack([row.forces for row in rows]),
        semantic8=torch.stack([row.semantic8 for row in rows]),
        path_index=torch.stack([row.path_index for row in rows]),
        clearance=torch.stack([row.clearance for row in rows]),
        stratum=time_stratum,
        anchor_index=torch.tensor(anchors, dtype=torch.long, device=device),
        initial_y_offset=torch.tensor(offsets, dtype=torch.float32, device=device),
        source_artifact_sha256=artifact.sha256,
        rejected_attempts=rejected,
    )
    corpus.validate(cfg, tolerance=protocol.dynamics_tolerance)
    return corpus


def corpus_metadata(
    corpus: SingleSlitCorpus,
    artifact: ExecutableTeacherArtifact,
    cfg: EnvConfig,
    options: SingleSlitCorpusConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    counts = {
        name: int((corpus.stratum[:, 0] == index).sum().detach().cpu())
        for index, name in enumerate(STRATUM_NAMES)
    }
    return {
        "claim_status": "teacher corpus only; no learned-policy capability claim",
        "seed": seed,
        "source_artifact": str(artifact.path),
        "source_continuous_minimum_clearance_m": artifact.continuous_clearance_m,
        "environment": asdict(cfg),
        "generator": asdict(options),
        "rows": int(corpus.state.shape[0]),
        "time": int(corpus.state.shape[1]),
        "stratum_counts": counts,
        "minimum_stored_transition_clearance_m": float(
            corpus.clearance.amin().detach().cpu()
        ),
        "maximum_handle_force_n": float(
            torch.linalg.vector_norm(corpus.forces, dim=-1).amax().detach().cpu()
        ),
        "admission": {
            "full_teacher_continuation_executed": True,
            "actual_step_dynamics": True,
            "exact_sat_every_transition": True,
            "continuous_recursive_sat_first_off_nominal_transition": True,
            "isotropic_pose_noise": False,
            "fiber": "connected exact-SAT vertical pose fiber, positively eroded",
            "failed_branches_stored": False,
            "nominal_context_ticks": options.burn_in_steps,
            "external_feedback_disturbance_tick": options.burn_in_steps,
            "recovery_action_is_supervised": True,
        },
        "semantic8": [
            "desired_world_twist_x",
            "desired_world_twist_y",
            "desired_world_omega",
            "realized_world_wrench_fx",
            "realized_world_wrench_fy",
            "realized_world_wrench_tau",
            "signed_cross_track",
            "ordered_progress",
        ],
    }


__all__ = [
    "EXECUTABLE_TEACHER_SCHEMA",
    "ExecutableTeacherArtifact",
    "SINGLE_SLIT_DATASET_SCHEMA",
    "STRATUM_NAMES",
    "STRATUM_NEAR_JAM",
    "STRATUM_NOMINAL",
    "STRATUM_TUBE_CORE",
    "STRATUM_WALL_NEAR",
    "SingleSlitCorpus",
    "SingleSlitCorpusConfig",
    "corpus_metadata",
    "generate_single_slit_corpus",
    "inverse_dynamics_wrench",
    "load_executable_teacher_artifact",
    "load_single_slit_corpus",
]
