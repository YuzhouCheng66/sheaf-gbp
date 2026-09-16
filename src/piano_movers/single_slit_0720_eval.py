"""Fail-closed closed-loop evaluation for the 0.720 m single-slit skill.

The evaluator deliberately does not reuse the loose historical success mask.
An episode succeeds only after five consecutive safe control transitions whose
accepted states satisfy the strict pose and velocity gate declared below.
Every policy call carries recurrent state, applies the policy's actual forces,
and charges every directed radius-edge message executed by that call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

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
from .single_slit_0720_bc import (
    LEGACY_SINGLE_SLIT_BC_SCHEMA,
    SINGLE_SLIT_BC_SCHEMA,
    SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
    SUPPORTED_ARMS,
    SingleSlitPolicy,
    architecture_record,
    build_policy,
)
from .single_slit_contact_observation import (
    SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
)
from .single_slit_feedback import (
    SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA,
    SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA,
    single_slit_env_config,
    single_slit_observation_schema_for_dim,
    single_slit_policy_observations,
    single_slit_scenario,
)


SINGLE_SLIT_EVAL_SCHEMA = "piano-movers-single-slit-0720-closed-loop-eval-v1"
STRICT_POSITION_TOLERANCE_M = 0.03
STRICT_ANGLE_TOLERANCE_RAD = 0.05
STRICT_SPEED_TOLERANCE_M_PER_S = 0.05
STRICT_OMEGA_TOLERANCE_RAD_PER_S = 0.08
STRICT_STABLE_STEPS = 5
DEFAULT_EVALUATION_HORIZON = 256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _config_without_device(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "device"}


@dataclass(frozen=True)
class LoadedSingleSlitCheckpoint:
    """An exactly reconstructed single-slit policy and its byte provenance."""

    path: Path
    checkpoint_sha256: str
    step: int
    arm: str
    observation_schema: str
    policy: SingleSlitPolicy
    cfg: Any
    metadata: Mapping[str, Any]


@torch.no_grad()
def load_single_slit_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> LoadedSingleSlitCheckpoint:
    """Load any of the four matched BC arms with strict architecture checks."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    target_device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=target_device,
        weights_only=True,
    )
    checkpoint = _mapping(checkpoint, "single-slit checkpoint")
    checkpoint_schema = checkpoint.get("schema_version")
    if checkpoint_schema == SINGLE_SLIT_CONTACT_BC_SCHEMA_V1:
        observation_schema = checkpoint.get("observation_schema")
        if observation_schema != SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1:
            raise ValueError("contact-v1 checkpoint omitted its explicit obs27 schema")
    elif checkpoint_schema == SINGLE_SLIT_BC_SCHEMA:
        observation_schema = checkpoint.get("observation_schema")
        if observation_schema != SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA:
            raise ValueError("feedback-v2 checkpoint omitted its exact-clearance4 schema")
    elif checkpoint_schema == LEGACY_SINGLE_SLIT_BC_SCHEMA:
        observation_schema = checkpoint.get(
            "observation_schema", SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA
        )
        if observation_schema != SINGLE_SLIT_LEGACY_OBSERVATION_SCHEMA:
            raise ValueError("legacy-v1 checkpoint must use strict-local18")
    else:
        raise ValueError("single-slit checkpoint schema mismatch")
    arm = checkpoint.get("arm")
    if arm not in SUPPORTED_ARMS:
        raise ValueError(f"unsupported single-slit checkpoint arm: {arm!r}")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("single-slit checkpoint step must be a non-negative integer")

    saved_env = _mapping(checkpoint.get("env_cfg"), "checkpoint env_cfg")
    saved_max_steps = saved_env.get("max_steps")
    if (
        isinstance(saved_max_steps, bool)
        or not isinstance(saved_max_steps, int)
        or saved_max_steps <= 0
    ):
        raise ValueError("checkpoint env_cfg.max_steps must be a positive integer")
    cfg = single_slit_env_config(device=target_device, max_steps=saved_max_steps)
    # Reconstruct records through EnvConfig so checkpoints that predate newly
    # added defaulted geometry fields remain the exact canonical task.
    normalized_saved_env = asdict(EnvConfig(**saved_env))
    if _config_without_device(normalized_saved_env) != _config_without_device(asdict(cfg)):
        raise ValueError("checkpoint environment is not the canonical single-slit task")

    saved_arm = dict(_mapping(checkpoint.get("arm_cfg"), "checkpoint arm_cfg"))
    saved_arm.setdefault("correction_public_unary_residual", False)
    saved_arm.setdefault("correction_public_unary_residual_scale", 0.25)
    saved_arm.setdefault("correction_source_null_recovery", False)
    saved_arm.setdefault("correction_phase_decoder", False)
    saved_arm.setdefault("correction_phase_basis_dim", 6)
    saved_arm.setdefault("correction_phase_content_dim", 12)
    saved_arm.setdefault("correction_phase_wrench_scale", 0.25)
    saved_arm.setdefault("correction_clearance_bank_decoder", False)
    saved_arm.setdefault("correction_clearance_bank_basis_dim", 24)
    saved_arm.setdefault("correction_clearance_bank_content_dim", 12)
    saved_arm.setdefault("correction_clearance_bank_wrench_scale", 0.25)
    saved_arm.setdefault("correction_clearance_bank_centers_mm", (0.0, 5.0, 13.0, 21.0))
    saved_arm.setdefault("correction_clearance_bank_sigma_mm", 3.0)
    saved_arm.setdefault("correction_approach_bank_decoder", False)
    saved_arm.setdefault("correction_approach_bank_basis_dim", 6)
    saved_arm.setdefault("correction_approach_bank_content_dim", 12)
    saved_arm.setdefault("correction_approach_bank_wrench_scale", 0.25)
    saved_arm.setdefault("correction_approach_bank_center_mm", 28.0)
    saved_arm.setdefault("correction_approach_bank_sigma_mm", 2.0)
    # Historical checkpoints predate the opt-in Stage-2 local recovery
    # suffix.  Only absent keys receive defaults; an explicitly serialized
    # incompatible value is still rejected by the exact ArmConfig equality
    # check below.
    saved_arm.setdefault("persistent_recovery_state", False)
    saved_arm.setdefault("persistent_recovery_dim", 8)
    saved_arm.setdefault("persistent_recovery_phase_count", 4)
    saved_arm.setdefault("persistent_recovery_injection", "shared_hidden")
    saved_arm.setdefault("collision_role_factors", False)
    saved_arm.setdefault("collision_role_count", 3)
    saved_arm.setdefault("collision_role_mean_scale", 0.25)
    saved_arm.setdefault("collision_role_log_precision_scale", 1.5)
    saved_arm.setdefault("collision_role_edge_log_precision_scale", 1.5)
    saved_arm.setdefault("collision_role_precontact_leader_follower", False)
    saved_arm.setdefault("collision_role_separate_encoder", False)
    saved_arm.setdefault("local_se2_planner", False)
    saved_arm.setdefault("local_se2_plan_dim", 8)
    saved_arm.setdefault("local_se2_planner_hidden_dim", 96)
    saved_arm.setdefault("local_se2_recovery_adapter", False)
    saved_arm.setdefault("local_se2_recovery_adapter_hidden_dim", 96)
    saved_arm.setdefault("local_se2_recovery_adapter_scale", 1.0)
    saved_arm.setdefault("local_se2_recovery_canonical_features", False)
    saved_arm.setdefault("local_se2_retry_refinement_thresholds", ())
    saved_arm.setdefault("local_se2_retry_refinement_hidden_dim", 64)
    saved_arm.setdefault("local_se2_retry_refinement_scale", 1.0)
    saved_arm.setdefault("local_se2_retry_refinement_min_center_x_m", -0.30)
    saved_arm.setdefault("local_se2_analytic_phase_latch", False)
    saved_arm.setdefault("local_se2_reentry_center_x_threshold_m", -0.16)
    saved_arm.setdefault("local_se2_phase_latch_logit", 8.0)
    saved_arm.setdefault("local_se2_direct_wrench_unary", False)
    saved_arm.setdefault("local_se2_direct_wrench_hidden_dim", 128)
    saved_arm.setdefault("local_se2_analytic_total_wrench", False)
    saved_arm.setdefault("local_se2_analytic_unary_precision", 1.0)
    saved_arm.setdefault("local_se2_geometry_evidence_precision_boost", 1.0)
    saved_arm.setdefault("local_se2_analytic_edge_sigma2", 1.0e-5)
    saved_arm.setdefault("local_se2_analytic_solver_eps", 1.0e-10)
    saved_arm.setdefault("local_se2_analytic_max_precision", 1.0e9)
    saved_arm.setdefault("local_se2_distributed_wrench_control", False)
    saved_arm.setdefault("local_se2_distributed_next_se2_control", False)
    saved_arm.setdefault("local_se2_distributed_force_hidden_dim", 128)
    saved_arm.setdefault("local_se2_distributed_recovery_adapter", False)
    saved_arm.setdefault("local_se2_distributed_recovery_hidden_dim", 128)
    saved_arm.setdefault("local_se2_distributed_recovery_layers", 2)
    saved_arm.setdefault("local_se2_distributed_recovery_mode_count", 1)
    saved_arm.setdefault("local_se2_distributed_recovery_scale", 1.0)
    saved_arm.setdefault("local_se2_distributed_recovery_local_support", False)
    saved_arm.setdefault("local_se2_distributed_recovery_wall_range_m", 0.45)
    saved_arm.setdefault("local_se2_distributed_grasp_allreduce", False)
    saved_arm.setdefault("local_se2_terminal_attractor", False)
    saved_arm.setdefault("local_se2_terminal_position_gain", 2.0)
    saved_arm.setdefault("local_se2_terminal_angle_gain", 2.0)
    saved_arm.setdefault("local_se2_terminal_max_speed", 0.35)
    saved_arm.setdefault("local_se2_terminal_max_omega", 0.8)
    policy = build_policy(
        str(arm),
        cfg,
        observation_schema=observation_schema,
        correction_sheaf=bool(
            saved_arm.get("correction_sheaf", False)
        ),
        correction_public_unary_residual=bool(
            saved_arm.get("correction_public_unary_residual", False)
        ),
        correction_public_unary_residual_scale=float(
            saved_arm.get("correction_public_unary_residual_scale", 0.25)
        ),
        correction_source_null_recovery=bool(
            saved_arm.get("correction_source_null_recovery", False)
        ),
        correction_phase_decoder=bool(
            saved_arm.get("correction_phase_decoder", False)
        ),
        correction_clearance_bank_decoder=bool(
            saved_arm.get("correction_clearance_bank_decoder", False)
        ),
        correction_approach_bank_decoder=bool(
            saved_arm.get("correction_approach_bank_decoder", False)
        ),
        persistent_recovery_state=bool(
            saved_arm.get("persistent_recovery_state", False)
        ),
        persistent_recovery_dim=int(saved_arm.get("persistent_recovery_dim", 8)),
        persistent_recovery_phase_count=int(
            saved_arm.get("persistent_recovery_phase_count", 4)
        ),
        persistent_recovery_injection=str(
            saved_arm.get("persistent_recovery_injection", "shared_hidden")
        ),
        collision_role_factors=bool(saved_arm.get("collision_role_factors", False)),
        collision_role_count=int(saved_arm.get("collision_role_count", 3)),
        collision_role_mean_scale=float(
            saved_arm.get("collision_role_mean_scale", 0.25)
        ),
        collision_role_log_precision_scale=float(
            saved_arm.get("collision_role_log_precision_scale", 1.5)
        ),
        collision_role_edge_log_precision_scale=float(
            saved_arm.get("collision_role_edge_log_precision_scale", 1.5)
        ),
        collision_role_precontact_leader_follower=bool(
            saved_arm.get("collision_role_precontact_leader_follower", False)
        ),
        collision_role_separate_encoder=bool(
            saved_arm.get("collision_role_separate_encoder", False)
        ),
        local_se2_planner=bool(saved_arm.get("local_se2_planner", False)),
        local_se2_plan_dim=int(saved_arm.get("local_se2_plan_dim", 8)),
        local_se2_planner_hidden_dim=int(
            saved_arm.get("local_se2_planner_hidden_dim", 96)
        ),
        local_se2_shape_signature=bool(
            saved_arm.get("local_se2_shape_signature", False)
        ),
        local_se2_shape_signature_dim=int(
            saved_arm.get("local_se2_shape_signature_dim", 8)
        ),
        local_se2_markov_shape_planner=bool(
            saved_arm.get("local_se2_markov_shape_planner", False)
        ),
        local_se2_markov_shape_hidden_dim=int(
            saved_arm.get("local_se2_markov_shape_hidden_dim", 256)
        ),
        local_se2_markov_shape_layers=int(
            saved_arm.get("local_se2_markov_shape_layers", 3)
        ),
        local_se2_geometry_refinement_adapter=bool(
            saved_arm.get("local_se2_geometry_refinement_adapter", False)
        ),
        local_se2_geometry_refinement_hidden_dim=int(
            saved_arm.get("local_se2_geometry_refinement_hidden_dim", 128)
        ),
        local_se2_geometry_refinement_scale=float(
            saved_arm.get("local_se2_geometry_refinement_scale", 1.0)
        ),
        local_se2_geometry_refinement_canonical_features=bool(
            saved_arm.get(
                "local_se2_geometry_refinement_canonical_features", False
            )
        ),
        local_se2_geometry_mode_adapter=bool(
            saved_arm.get("local_se2_geometry_mode_adapter", False)
        ),
        local_se2_geometry_mode_count=int(
            saved_arm.get("local_se2_geometry_mode_count", 4)
        ),
        local_se2_geometry_mode_hidden_dim=int(
            saved_arm.get("local_se2_geometry_mode_hidden_dim", 256)
        ),
        local_se2_geometry_mode_scale=float(
            saved_arm.get("local_se2_geometry_mode_scale", 1.0)
        ),
        local_se2_geometry_mode_post_contact_only=bool(
            saved_arm.get("local_se2_geometry_mode_post_contact_only", False)
        ),
        local_se2_path_memory=bool(
            saved_arm.get("local_se2_path_memory", False)
        ),
        local_se2_path_memory_shape_count=int(
            saved_arm.get("local_se2_path_memory_shape_count", 24)
        ),
        local_se2_path_memory_max_steps=int(
            saved_arm.get("local_se2_path_memory_max_steps", 256)
        ),
        local_se2_path_memory_ridge_decoder=bool(
            saved_arm.get("local_se2_path_memory_ridge_decoder", False)
        ),
        local_se2_path_memory_ridge_points=int(
            saved_arm.get("local_se2_path_memory_ridge_points", 224)
        ),
        local_se2_path_memory_ridge_alpha=float(
            saved_arm.get("local_se2_path_memory_ridge_alpha", 3.0e-4)
        ),
        local_se2_path_memory_ridge_trust_margin=float(
            saved_arm.get("local_se2_path_memory_ridge_trust_margin", 0.05)
        ),
        local_se2_path_memory_canonical_family_mse_threshold=float(
            saved_arm.get(
                "local_se2_path_memory_canonical_family_mse_threshold", 0.10
            )
        ),
        local_se2_path_memory_population_resampling=bool(
            saved_arm.get("local_se2_path_memory_population_resampling", False)
        ),
        local_se2_path_memory_population_resampling_points=int(
            saved_arm.get("local_se2_path_memory_population_resampling_points", 256)
        ),
        local_se2_path_memory_population_preemptive_counts=tuple(
            int(value)
            for value in saved_arm.get(
                "local_se2_path_memory_population_preemptive_counts", (10,)
            )
        ),
        local_se2_path_memory_canonical_preemptive_counts=tuple(
            int(value)
            for value in saved_arm.get(
                "local_se2_path_memory_canonical_preemptive_counts",
                (11, 13, 15, 16, 17, 20),
            )
        ),
        local_se2_path_memory_population_recovery_counts=tuple(
            int(value)
            for value in saved_arm.get(
                "local_se2_path_memory_population_recovery_counts",
                (11, 13, 15, 17, 20),
            )
        ),
        local_se2_recovery_adapter=bool(
            saved_arm.get("local_se2_recovery_adapter", False)
        ),
        local_se2_recovery_adapter_hidden_dim=int(
            saved_arm.get("local_se2_recovery_adapter_hidden_dim", 96)
        ),
        local_se2_recovery_adapter_scale=float(
            saved_arm.get("local_se2_recovery_adapter_scale", 1.0)
        ),
        local_se2_recovery_canonical_features=bool(
            saved_arm.get("local_se2_recovery_canonical_features", False)
        ),
        local_se2_contact_rebel_adapter=bool(
            saved_arm.get("local_se2_contact_rebel_adapter", False)
        ),
        local_se2_contact_rebel_hidden_dim=int(
            saved_arm.get("local_se2_contact_rebel_hidden_dim", 64)
        ),
        local_se2_contact_rebel_scale=float(
            saved_arm.get("local_se2_contact_rebel_scale", 1.0)
        ),
        local_se2_contact_rebel_min_center_x_m=float(
            saved_arm.get("local_se2_contact_rebel_min_center_x_m", 0.0)
        ),
        local_se2_exit_refinement_adapter=bool(
            saved_arm.get("local_se2_exit_refinement_adapter", False)
        ),
        local_se2_exit_refinement_hidden_dim=int(
            saved_arm.get("local_se2_exit_refinement_hidden_dim", 64)
        ),
        local_se2_exit_refinement_scale=float(
            saved_arm.get("local_se2_exit_refinement_scale", 1.0)
        ),
        local_se2_exit_refinement_min_center_x_m=float(
            saved_arm.get("local_se2_exit_refinement_min_center_x_m", 0.17)
        ),
        local_se2_approach_refinement_adapter=bool(
            saved_arm.get("local_se2_approach_refinement_adapter", False)
        ),
        local_se2_approach_refinement_hidden_dim=int(
            saved_arm.get("local_se2_approach_refinement_hidden_dim", 64)
        ),
        local_se2_approach_refinement_scale=float(
            saved_arm.get("local_se2_approach_refinement_scale", 1.0)
        ),
        local_se2_approach_refinement_min_center_x_m=float(
            saved_arm.get("local_se2_approach_refinement_min_center_x_m", -0.30)
        ),
        local_se2_approach_refinement_stuck_steps=int(
            saved_arm.get("local_se2_approach_refinement_stuck_steps", 9)
        ),
        local_se2_deep_stuck_refinement_adapter=bool(
            saved_arm.get("local_se2_deep_stuck_refinement_adapter", False)
        ),
        local_se2_deep_stuck_refinement_hidden_dim=int(
            saved_arm.get("local_se2_deep_stuck_refinement_hidden_dim", 64)
        ),
        local_se2_deep_stuck_refinement_scale=float(
            saved_arm.get("local_se2_deep_stuck_refinement_scale", 1.0)
        ),
        local_se2_deep_stuck_refinement_min_center_x_m=float(
            saved_arm.get("local_se2_deep_stuck_refinement_min_center_x_m", -0.30)
        ),
        local_se2_deep_stuck_refinement_stuck_steps=int(
            saved_arm.get("local_se2_deep_stuck_refinement_stuck_steps", 10)
        ),
        local_se2_retry_refinement_thresholds=tuple(
            int(value)
            for value in saved_arm.get("local_se2_retry_refinement_thresholds", ())
        ),
        local_se2_retry_refinement_hidden_dim=int(
            saved_arm.get("local_se2_retry_refinement_hidden_dim", 64)
        ),
        local_se2_retry_refinement_scale=float(
            saved_arm.get("local_se2_retry_refinement_scale", 1.0)
        ),
        local_se2_retry_refinement_min_center_x_m=float(
            saved_arm.get("local_se2_retry_refinement_min_center_x_m", -0.30)
        ),
        local_se2_analytic_phase_latch=bool(
            saved_arm.get("local_se2_analytic_phase_latch", False)
        ),
        local_se2_reentry_center_x_threshold_m=float(
            saved_arm.get("local_se2_reentry_center_x_threshold_m", -0.16)
        ),
        local_se2_phase_latch_logit=float(
            saved_arm.get("local_se2_phase_latch_logit", 8.0)
        ),
        local_se2_direct_wrench_unary=bool(
            saved_arm.get("local_se2_direct_wrench_unary", False)
        ),
        local_se2_direct_wrench_hidden_dim=int(
            saved_arm.get("local_se2_direct_wrench_hidden_dim", 128)
        ),
        local_se2_analytic_total_wrench=bool(
            saved_arm.get("local_se2_analytic_total_wrench", False)
        ),
        local_se2_analytic_unary_precision=float(
            saved_arm.get("local_se2_analytic_unary_precision", 1.0)
        ),
        local_se2_geometry_evidence_precision_boost=float(
            saved_arm.get("local_se2_geometry_evidence_precision_boost", 1.0)
        ),
        local_se2_analytic_edge_sigma2=float(
            saved_arm.get("local_se2_analytic_edge_sigma2", 1.0e-5)
        ),
        local_se2_analytic_solver_eps=float(
            saved_arm.get("local_se2_analytic_solver_eps", 1.0e-10)
        ),
        local_se2_analytic_max_precision=float(
            saved_arm.get("local_se2_analytic_max_precision", 1.0e9)
        ),
        local_se2_distributed_wrench_control=bool(
            saved_arm.get("local_se2_distributed_wrench_control", False)
        ),
        local_se2_distributed_next_se2_control=bool(
            saved_arm.get("local_se2_distributed_next_se2_control", False)
        ),
        local_se2_distributed_force_hidden_dim=int(
            saved_arm.get("local_se2_distributed_force_hidden_dim", 128)
        ),
        local_se2_distributed_recovery_adapter=bool(
            saved_arm.get("local_se2_distributed_recovery_adapter", False)
        ),
        local_se2_distributed_recovery_hidden_dim=int(
            saved_arm.get("local_se2_distributed_recovery_hidden_dim", 128)
        ),
        local_se2_distributed_recovery_layers=int(
            saved_arm.get("local_se2_distributed_recovery_layers", 2)
        ),
        local_se2_distributed_recovery_mode_count=int(
            saved_arm.get("local_se2_distributed_recovery_mode_count", 1)
        ),
        local_se2_distributed_recovery_scale=float(
            saved_arm.get("local_se2_distributed_recovery_scale", 1.0)
        ),
        local_se2_distributed_recovery_local_support=bool(
            saved_arm.get("local_se2_distributed_recovery_local_support", False)
        ),
        local_se2_distributed_recovery_wall_range_m=float(
            saved_arm.get("local_se2_distributed_recovery_wall_range_m", 0.45)
        ),
        local_se2_distributed_grasp_allreduce=bool(
            saved_arm.get("local_se2_distributed_grasp_allreduce", False)
        ),
        local_se2_terminal_attractor=bool(
            saved_arm.get("local_se2_terminal_attractor", False)
        ),
        local_se2_terminal_position_gain=float(
            saved_arm.get("local_se2_terminal_position_gain", 2.0)
        ),
        local_se2_terminal_angle_gain=float(
            saved_arm.get("local_se2_terminal_angle_gain", 2.0)
        ),
        local_se2_terminal_max_speed=float(
            saved_arm.get("local_se2_terminal_max_speed", 0.35)
        ),
        local_se2_terminal_max_omega=float(
            saved_arm.get("local_se2_terminal_max_omega", 0.8)
        ),
        local_se2_feasible_corridor_attractor=bool(
            saved_arm.get("local_se2_feasible_corridor_attractor", False)
        ),
        local_se2_route_free_geometry_planner=bool(
            saved_arm.get("local_se2_route_free_geometry_planner", False)
        ),
        local_se2_corridor_reserve_m=float(
            saved_arm.get("local_se2_corridor_reserve_m", 0.015)
        ),
        local_se2_corridor_max_speed=float(
            saved_arm.get("local_se2_corridor_max_speed", 0.50)
        ),
    )
    # New source-null modules append default-valued configuration fields.  An
    # older checkpoint that predates such a field is equivalent to explicitly
    # storing its disabled default; unknown or nondefault mismatches still
    # fail closed.
    reconstructed_arm = asdict(policy.arm_cfg)
    normalized_saved_arm = dict(reconstructed_arm)
    normalized_saved_arm.update(saved_arm)
    if normalized_saved_arm != reconstructed_arm:
        raise ValueError("checkpoint arm configuration does not match the reconstructed policy")
    state_dict = _mapping(checkpoint.get("policy_state_dict"), "checkpoint policy_state_dict")
    policy.load_state_dict(dict(state_dict), strict=True)
    policy.eval()
    return LoadedSingleSlitCheckpoint(
        path=checkpoint_path,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        step=step,
        arm=str(arm),
        observation_schema=observation_schema,
        policy=policy,
        cfg=cfg,
        metadata=checkpoint,
    )


def strict_final_components(state: Tensor, goal: Tensor) -> dict[str, Tensor]:
    """Return each component of the strict final-state predicate."""

    if state.ndim != 2 or state.shape[-1] != 6 or goal.shape != (state.shape[0], 3):
        raise ValueError("strict final-state inputs must have shape [B,6] and [B,3]")
    position_error = torch.linalg.vector_norm(state[:, :2] - goal[:, :2], dim=-1)
    angle_error = wrap_angle(state[:, 2] - goal[:, 2]).abs()
    speed = torch.linalg.vector_norm(state[:, 3:5], dim=-1)
    omega = state[:, 5].abs()
    return {
        "position_error_m": position_error,
        "angle_error_rad": angle_error,
        "speed_m_per_s": speed,
        "omega_rad_per_s": omega,
        "position_ok": position_error <= STRICT_POSITION_TOLERANCE_M,
        "angle_ok": angle_error <= STRICT_ANGLE_TOLERANCE_RAD,
        "speed_ok": speed <= STRICT_SPEED_TOLERANCE_M_PER_S,
        "omega_ok": omega <= STRICT_OMEGA_TOLERANCE_RAD_PER_S,
    }


def strict_final_mask(state: Tensor, goal: Tensor) -> Tensor:
    components = strict_final_components(state, goal)
    return (
        components["position_ok"]
        & components["angle_ok"]
        & components["speed_ok"]
        & components["omega_ok"]
    )


def _communication_dimensions(policy: SingleSlitPolicy, edge_index: Tensor) -> dict[str, int]:
    width = policy.communication_floats_per_directed_edge_round()
    rounds = policy.communication_rounds()
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width < 0
        or isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or rounds < 0
    ):
        raise RuntimeError("policy communication width/round count must be non-negative ints")
    if (width == 0) != (rounds == 0):
        raise RuntimeError("a policy must have both zero width and zero rounds, or neither")
    floating_dtypes = {
        parameter.dtype for parameter in policy.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes and floating_dtypes != {torch.float32}:
        raise RuntimeError("single-slit payload accounting requires an FP32 policy")
    directed_edges = 2 * int(edge_index.shape[0])
    gbp_packets_per_tick = directed_edges * rounds
    gbp_floats_per_tick = gbp_packets_per_tick * width
    # The force allocator has a second real distributed protocol, but handle
    # coordinates are fixed in the rigid payload frame.  Its four-float tree
    # reduction is therefore charged once at attachment/setup; it is not
    # repeated at the control rate.  Runtime rotates the cached first moments
    # using each agent's locally observed payload orientation.
    grasp_width = 0
    grasp_rounds = 0
    grasp_directed_edges = 0
    if bool(
        getattr(
            policy.arm_cfg,
            "local_se2_distributed_grasp_allreduce",
            False,
        )
    ):
        n_agents = int(policy.env_cfg.n_agents)
        grasp_width = 4
        grasp_rounds = max(1, n_agents - 1)
        grasp_directed_edges = 2 * (n_agents - 1)
    grasp_packets_per_setup = grasp_directed_edges * grasp_rounds
    grasp_floats_per_setup = grasp_packets_per_setup * grasp_width
    grasp_packets_per_tick = 0
    grasp_floats_per_tick = 0
    packets_per_tick = gbp_packets_per_tick
    floats_per_tick = gbp_floats_per_tick
    return {
        "undirected_edges": int(edge_index.shape[0]),
        "directed_edges": directed_edges,
        "rounds_per_control_tick": rounds,
        "floats_per_directed_edge_round": width,
        "gbp_packets_per_control_tick": gbp_packets_per_tick,
        "gbp_payload_floats_per_control_tick": gbp_floats_per_tick,
        "grasp_directed_tree_edges": grasp_directed_edges,
        "grasp_rounds_per_setup": grasp_rounds,
        # Compatibility keys are retained with their truthful control-rate
        # value: the setup reduction contributes zero packets per later tick.
        "grasp_rounds_per_control_tick": 0,
        "grasp_floats_per_directed_tree_edge_round": grasp_width,
        "grasp_packets_per_setup": grasp_packets_per_setup,
        "grasp_payload_floats_per_setup": grasp_floats_per_setup,
        "grasp_payload_bytes_per_setup": 4 * grasp_floats_per_setup,
        "grasp_packets_per_control_tick": grasp_packets_per_tick,
        "grasp_payload_floats_per_control_tick": grasp_floats_per_tick,
        "packets_per_control_tick": packets_per_tick,
        "payload_floats_per_control_tick": floats_per_tick,
        "payload_bytes_per_control_tick": 4 * floats_per_tick,
    }


def _timeout_conditions(components: Mapping[str, Tensor], index: int, dwell: int) -> list[str]:
    failures: list[str] = []
    for label, key in (
        ("position_tolerance", "position_ok"),
        ("angle_tolerance", "angle_ok"),
        ("speed_tolerance", "speed_ok"),
        ("omega_tolerance", "omega_ok"),
    ):
        if not bool(components[key][index].item()):
            failures.append(label)
    if not failures and dwell < STRICT_STABLE_STEPS:
        failures.append("five_step_stability")
    return failures


@torch.no_grad()
def evaluate_single_slit_closed_loop(
    policy: SingleSlitPolicy,
    cfg: Any,
    *,
    initial_states: Tensor | None = None,
    initial_hidden: Tensor | None = None,
    episodes: int = 1,
    horizon: int = DEFAULT_EVALUATION_HORIZON,
) -> dict[str, Any]:
    """Run actual policy forces with carried hidden state and strict SAT gates.

    Communication per successful mission is *failure charged*: its numerator
    includes the communication consumed by every failed attempt, rather than
    conditioning the byte average on successful episodes.
    """

    if horizon <= 0:
        raise ValueError("evaluation horizon must be positive")
    if cfg.collision_substeps != 4:
        raise ValueError("single-slit evaluation requires four swept SAT substeps")
    if not cfg.strict_local_observation:
        raise ValueError("single-slit evaluation requires strict-local observations")
    if initial_states is None:
        if episodes <= 0:
            raise ValueError("evaluation episodes must be positive")
        scenario = single_slit_scenario(episodes, cfg)
    else:
        if initial_states.ndim != 2 or initial_states.shape[-1] != 6:
            raise ValueError("initial_states must have shape [episodes,6]")
        if initial_states.shape[0] <= 0:
            raise ValueError("initial_states must contain at least one episode")
        scenario = single_slit_scenario(
            int(initial_states.shape[0]), cfg, initial_state=initial_states
        )
    state = scenario.state0.clone()
    episode_count = int(state.shape[0])
    edge_index = radius_edges(cfg, device=state.device)
    comm = _communication_dimensions(policy, edge_index)

    initial_collision = collision_mask(state, scenario, cfg)
    minimum_clearance = collision_clearance_margin(state, scenario, cfg)
    if not torch.equal(initial_collision, minimum_clearance <= 0.0):
        raise RuntimeError("initial boolean SAT and signed-clearance authorities disagree")
    if bool(initial_collision.any()):
        raise ValueError("single-slit evaluation cannot start in collision")

    zero_hidden = policy.init_hidden(episode_count, state.device)
    if initial_hidden is None:
        hidden = zero_hidden
        hidden_initialization = "policy_zero_initial_hidden"
    else:
        if not isinstance(initial_hidden, Tensor):
            raise TypeError("initial_hidden must be a tensor when supplied")
        if initial_hidden.shape != zero_hidden.shape:
            raise ValueError(
                "initial_hidden must match policy.init_hidden(episode_count) shape"
            )
        if initial_hidden.device != zero_hidden.device:
            raise ValueError("initial_hidden must be on the evaluation device")
        if initial_hidden.dtype != zero_hidden.dtype:
            raise ValueError("initial_hidden must use the policy hidden dtype")
        if not bool(torch.isfinite(initial_hidden).all()):
            raise ValueError("initial_hidden contains non-finite values")
        hidden = initial_hidden.detach().clone()
        hidden_initialization = "caller_supplied_authenticated_context"
    observation_schema = single_slit_observation_schema_for_dim(policy.arm_cfg.obs_dim)
    alive = torch.ones(episode_count, dtype=torch.bool, device=state.device)
    succeeded = torch.zeros_like(alive)
    collided = torch.zeros_like(alive)
    invalid_output = torch.zeros_like(alive)
    active_ticks = torch.zeros(episode_count, dtype=torch.long, device=state.device)
    finish_step = torch.full_like(active_ticks, horizon)
    dwell = torch.zeros_like(active_ticks)
    maximum_dwell = torch.zeros_like(active_ticks)
    failure_tick = torch.full_like(active_ticks, -1)
    failure_code = ["" for _ in range(episode_count)]
    terminal_attempt: list[list[float] | None] = [None for _ in range(episode_count)]
    failure_clearance: list[float | None] = [None for _ in range(episode_count)]

    was_training = policy.training
    policy.eval()
    try:
        for tick in range(horizon):
            if not bool(alive.any()):
                break
            active = alive.clone()
            active_ticks += active.to(torch.long)
            policy_observation = single_slit_policy_observations(
                state,
                scenario,
                cfg,
                observation_schema,
            )
            output = policy(policy_observation, edge_index, hidden)
            if output.forces.shape != (episode_count, cfg.n_agents, 2):
                raise RuntimeError("policy force tensor has the wrong shape")
            if output.hidden.shape != hidden.shape:
                raise RuntimeError("policy hidden tensor changed shape during evaluation")
            finite_force = torch.isfinite(output.forces).all(dim=(-2, -1))
            finite_hidden = torch.isfinite(output.hidden).flatten(1).all(dim=1)
            bounded_force = (
                torch.linalg.vector_norm(output.forces, dim=-1).amax(dim=-1)
                <= cfg.max_force + 1.0e-5
            )
            valid_output = finite_force & finite_hidden & bounded_force
            bad = active & ~valid_output
            if bool(bad.any()):
                for index in torch.where(bad)[0].tolist():
                    failure_code[index] = "invalid_policy_output"
                    failure_tick[index] = tick + 1
                    terminal_attempt[index] = [
                        float(value) for value in state[index].detach().cpu().tolist()
                    ]
            invalid_output |= bad

            physics_active = active & valid_output
            hidden_mask = physics_active.view(episode_count, *([1] * (hidden.ndim - 1)))
            hidden = torch.where(hidden_mask, output.hidden, hidden)
            applied = torch.where(
                physics_active[:, None, None],
                output.forces,
                torch.zeros_like(output.forces),
            )
            proposed = step_dynamics(state, applied, cfg)
            transition_clearance = transition_clearance_margin(state, proposed, scenario, cfg)
            hit_authority = transition_collision_mask(state, proposed, scenario, cfg)
            if not torch.equal(hit_authority, transition_clearance <= 0.0):
                raise RuntimeError("boolean SAT and signed swept-clearance authorities disagree")
            hit = physics_active & hit_authority
            minimum_clearance = torch.where(
                physics_active,
                torch.minimum(minimum_clearance, transition_clearance),
                minimum_clearance,
            )
            if bool(hit.any()):
                for index in torch.where(hit)[0].tolist():
                    failure_code[index] = "exact_sat_collision"
                    failure_tick[index] = tick + 1
                    failure_clearance[index] = float(
                        transition_clearance[index].detach().cpu().item()
                    )
                    terminal_attempt[index] = [
                        float(value) for value in proposed[index].detach().cpu().tolist()
                    ]
            collided |= hit

            accepted = physics_active & ~hit
            state = torch.where(accepted[:, None], proposed, state)
            at_goal = accepted & strict_final_mask(state, scenario.goal)
            dwell = torch.where(
                accepted,
                torch.where(at_goal, dwell + 1, torch.zeros_like(dwell)),
                dwell,
            )
            maximum_dwell = torch.maximum(maximum_dwell, dwell)
            reached = accepted & (dwell >= STRICT_STABLE_STEPS)
            succeeded |= reached
            terminal = bad | hit | reached
            finish_step = torch.where(
                terminal & active,
                torch.full_like(finish_step, tick + 1),
                finish_step,
            )
            alive = active & ~terminal
    finally:
        policy.train(was_training)

    timed_out = alive.clone()
    outcome_partition = (
        succeeded.to(torch.int8)
        + collided.to(torch.int8)
        + invalid_output.to(torch.int8)
        + timed_out.to(torch.int8)
    )
    if not bool((outcome_partition == 1).all()):
        raise RuntimeError("single-slit outcomes do not form an exact partition")
    if not torch.equal(active_ticks, finish_step):
        raise RuntimeError("active communication ticks disagree with finish_step")

    components = strict_final_components(state, scenario.goal)
    per_episode: list[dict[str, Any]] = []
    bytes_per_tick = comm["payload_bytes_per_control_tick"]
    floats_per_tick = comm["payload_floats_per_control_tick"]
    packets_per_tick = comm["packets_per_control_tick"]
    setup_bytes = comm["grasp_payload_bytes_per_setup"]
    setup_floats = comm["grasp_payload_floats_per_setup"]
    setup_packets = comm["grasp_packets_per_setup"]
    for index in range(episode_count):
        success = bool(succeeded[index].item())
        collision = bool(collided[index].item())
        invalid = bool(invalid_output[index].item())
        timeout = bool(timed_out[index].item())
        if success:
            outcome = "success"
            reason: str | None = None
            failed_at: int | None = None
            conditions: list[str] = []
        elif collision:
            outcome = "collision"
            reason = failure_code[index]
            failed_at = int(failure_tick[index].item())
            conditions = [reason]
        elif invalid:
            outcome = "invalid_policy_output"
            reason = failure_code[index]
            failed_at = int(failure_tick[index].item())
            conditions = [reason]
        elif timeout:
            outcome = "timeout"
            conditions = _timeout_conditions(components, index, int(dwell[index].item()))
            reason = "timeout:" + ",".join(conditions)
            failed_at = horizon
        else:  # pragma: no cover - guarded by the partition assertion above.
            raise RuntimeError("unclassified single-slit outcome")
        ticks = int(active_ticks[index].item())
        per_episode.append(
            {
                "episode_index": index,
                "outcome": outcome,
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "invalid_policy_output": invalid,
                "finish_step": int(finish_step[index].item()),
                "failure_tick": failed_at,
                "failure_reason": reason,
                "failed_strict_conditions": conditions,
                "final_position_error_m": float(
                    components["position_error_m"][index].detach().cpu().item()
                ),
                "final_angle_error_rad": float(
                    components["angle_error_rad"][index].detach().cpu().item()
                ),
                "final_speed_m_per_s": float(
                    components["speed_m_per_s"][index].detach().cpu().item()
                ),
                "final_omega_rad_per_s": float(
                    components["omega_rad_per_s"][index].detach().cpu().item()
                ),
                "final_consecutive_strict_steps": int(dwell[index].item()),
                "maximum_consecutive_strict_steps": int(maximum_dwell[index].item()),
                "minimum_exact_sat_transition_clearance_m": float(
                    minimum_clearance[index].detach().cpu().item()
                ),
                "failure_transition_clearance_m": failure_clearance[index],
                "final_accepted_state": [
                    float(value) for value in state[index].detach().cpu().tolist()
                ],
                "terminal_attempt_state": terminal_attempt[index],
                "active_communication_ticks": ticks,
                "actual_payload_floats": setup_floats + ticks * floats_per_tick,
                "actual_payload_bytes": setup_bytes + ticks * bytes_per_tick,
                "actual_directed_packets": setup_packets + ticks * packets_per_tick,
                "failure_charged": True,
            }
        )

    success_count = int(succeeded.sum().item())
    collision_count = int(collided.sum().item())
    invalid_count = int(invalid_output.sum().item())
    timeout_count = int(timed_out.sum().item())
    total_active_ticks = int(active_ticks.sum().item())
    total_payload_bytes = episode_count * setup_bytes + total_active_ticks * bytes_per_tick
    total_payload_floats = (
        episode_count * setup_floats + total_active_ticks * floats_per_tick
    )
    total_directed_packets = (
        episode_count * setup_packets + total_active_ticks * packets_per_tick
    )
    fixed_horizon_bytes_per_episode = setup_bytes + horizon * bytes_per_tick
    fixed_horizon_bytes_total = episode_count * fixed_horizon_bytes_per_episode
    return {
        "schema_version": SINGLE_SLIT_EVAL_SCHEMA,
        "actor": policy.arm_name,
        "counts_as_policy": True,
        "teacher_used_at_policy_inference": False,
        "strict_local_observation_used": True,
        "observation_schema": observation_schema,
        "carried_hidden_state_used": True,
        "hidden_initialization": hidden_initialization,
        "actual_policy_forces_applied": True,
        "collision_authority": ("exact compound-rectangle SAT at four swept transition substeps"),
        "protocol": {
            "horizon_control_ticks": horizon,
            "episode_count": episode_count,
            "visible_aperture_m": 0.720,
            "strict_final_gate": {
                "position_error_m_lte": STRICT_POSITION_TOLERANCE_M,
                "angle_error_rad_lte": STRICT_ANGLE_TOLERANCE_RAD,
                "speed_m_per_s_lte": STRICT_SPEED_TOLERANCE_M_PER_S,
                "omega_rad_per_s_lte": STRICT_OMEGA_TOLERANCE_RAD_PER_S,
                "consecutive_safe_steps_required": STRICT_STABLE_STEPS,
            },
            "environment": asdict(cfg),
        },
        "success_count": success_count,
        "success_rate": success_count / episode_count,
        "collision_count": collision_count,
        "collision_rate": collision_count / episode_count,
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / episode_count,
        "invalid_policy_output_count": invalid_count,
        "minimum_exact_sat_transition_clearance_m": float(
            minimum_clearance.amin().detach().cpu().item()
        ),
        "mean_minimum_exact_sat_transition_clearance_m": float(
            minimum_clearance.mean().detach().cpu().item()
        ),
        "communication": {
            "accounting": (
                "all actual active policy calls across successful and failed attempts; "
                "float32 payload only"
            ),
            "failure_charged_definition": (
                "total communication over every attempted episode divided by successful "
                "missions; failed attempts remain in the numerator"
            ),
            "failed_attempts_charged": True,
            "event_triggering": False,
            "message_headers_included": False,
            "payload_scalar_dtype": "float32",
            "payload_scalar_bytes": 4,
            **comm,
            "actual_active_control_ticks_total": total_active_ticks,
            "actual_payload_floats_total": total_payload_floats,
            "actual_payload_bytes_total": total_payload_bytes,
            "actual_directed_packets_total": total_directed_packets,
            "actual_payload_bytes_per_attempt": total_payload_bytes / episode_count,
            "failure_charged_actual_payload_bytes_per_success": (
                total_payload_bytes / success_count if success_count else None
            ),
            "fixed_horizon_payload_bytes_per_episode": fixed_horizon_bytes_per_episode,
            "fixed_horizon_payload_bytes_total": fixed_horizon_bytes_total,
            "failure_charged_fixed_horizon_payload_bytes_per_success": (
                fixed_horizon_bytes_total / success_count if success_count else None
            ),
        },
        "architecture": architecture_record(policy, edge_index),
        "per_episode": per_episode,
    }


__all__ = [
    "DEFAULT_EVALUATION_HORIZON",
    "LoadedSingleSlitCheckpoint",
    "SINGLE_SLIT_EVAL_SCHEMA",
    "STRICT_ANGLE_TOLERANCE_RAD",
    "STRICT_OMEGA_TOLERANCE_RAD_PER_S",
    "STRICT_POSITION_TOLERANCE_M",
    "STRICT_SPEED_TOLERANCE_M_PER_S",
    "STRICT_STABLE_STEPS",
    "evaluate_single_slit_closed_loop",
    "load_single_slit_checkpoint",
    "strict_final_components",
    "strict_final_mask",
]
