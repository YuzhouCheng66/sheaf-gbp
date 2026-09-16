"""Policy arms for PianoMovers-Force."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import (
    ArmConfig,
    EnvConfig,
    RETURN_CANDIDATE_ARCHITECTURE_V1,
    RETURN_CANDIDATE_ARCHITECTURE_V2,
    RETURN_CANDIDATE_ARCHITECTURE_V3,
)
from .env import contact_points, radius_edges
from .gbp import (
    DiagSheafGBP,
    GBPMessageTrace,
    GBPResult,
    GBPTopology,
    homogeneous_topology,
)
from .linear_solve import solve_nonsingular
from .payload_shapes import (
    normalize_payload_shape,
    payload_rectangles_for_config,
    payload_shape_rectangles,
    t_morphology_from_config,
)
from .single_slit_local_planner import (
    LOCAL_SE2_ANGLE_SCALE_RAD,
    LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS,
    LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS,
    LOCAL_SE2_POSITION_SCALE_M,
    local_se2_inverse_dynamics_wrench_proposals,
    local_se2_terminal_attractor_plan,
)


_RETURN_GRASP_DAMPING = 1e-5
_STRICT_PHYSICAL_INDICES = (*range(8), *range(10, 18))
_STRICT_PHYSICAL_DIM = len(_STRICT_PHYSICAL_INDICES)
_STRICT_GOAL_SLICE = slice(8, 10)
_RETURN_SELECTOR_INVARIANT_DIM = 7
_RETURN_SELECTOR_SECTION_DIM = 8
_RETURN_SELECTOR_WRENCH_DIM = 3
_RETURN_SELECTOR_DIM = (
    _RETURN_SELECTOR_INVARIANT_DIM + _RETURN_SELECTOR_SECTION_DIM + _RETURN_SELECTOR_WRENCH_DIM
)
_RETURN_SELECTOR_HIDDEN_DIM = 64
_RETURN_SELECTOR_SOURCE_BIAS = -0.05
_RETURN_CANDIDATE_INPUT_DIM = _RETURN_SELECTOR_INVARIANT_DIM + _RETURN_SELECTOR_SECTION_DIM
_RETURN_CANDIDATE_HIDDEN_DIM = 128
_RETURN_CANDIDATE_OUTPUT_DIM = 3
_RETURN_CANDIDATE_V3_ANALYTIC_DIM = 20
_RETURN_CANDIDATE_V3_SECTION_DIM = 12
_RETURN_CANDIDATE_V3_INPUT_DIM = (
    _RETURN_CANDIDATE_V3_ANALYTIC_DIM + _RETURN_CANDIDATE_V3_SECTION_DIM
)
_RETURN_CANDIDATE_V3_HIDDEN_DIM = 192
_CORRECTION_WRENCH_DIM = 3
_SINGLE_SLIT_EXACT_CLEARANCE_DIM = 22
_SINGLE_SLIT_CONTACT_OBSERVATION_DIM = 27
_SINGLE_SLIT_CONTACT_HIT_INDEX = 22
LOCAL_SE2_RECOVERY_CANONICAL_DIM = 12
SOURCE_FORCES_AUX_KEY = "source_forces"
CORRECTION_SECTION_AUX_KEY = "correction_section"
CORRECTION_WRENCH_AUX_KEY = "correction_wrench"
CORRECTION_FORCE_AUX_KEY = "correction_force"
CORRECTION_PUBLIC_UNARY_DELTA_AUX_KEY = "correction_public_unary_delta"
CORRECTION_PUBLIC_UNARY_PRE_AUX_KEY = "correction_public_unary_mean_pre"
CORRECTION_PUBLIC_UNARY_POST_AUX_KEY = "correction_public_unary_mean_post"
CORRECTION_SOURCE_NULL_RISK_AUX_KEY = "correction_source_null_recovery_rho"
CORRECTION_SOURCE_NULL_DELTA_AUX_KEY = "correction_source_null_recovery_delta"
CORRECTION_SOURCE_NULL_PRE_AUX_KEY = "correction_source_null_recovery_mean_pre"
CORRECTION_SOURCE_NULL_POST_AUX_KEY = "correction_source_null_recovery_mean_post"
PERSISTENT_RECOVERY_STATE_AUX_KEY = "persistent_recovery_state"
PERSISTENT_RECOVERY_ACTIVE_AUX_KEY = "persistent_recovery_active"
PERSISTENT_RECOVERY_PHASE_LOGITS_AUX_KEY = "persistent_recovery_phase_logits"
COLLISION_ROLE_NAMES = ("leader", "follower", "rebel")
COLLISION_ROLE_LOGITS_AUX_KEY = "collision_role_logits"
COLLISION_ROLE_WEIGHTS_AUX_KEY = "collision_role_weights"
COLLISION_ROLE_MEAN_DELTA_AUX_KEY = "collision_role_unary_mean_delta"
COLLISION_ROLE_LOG_PRECISION_DELTA_AUX_KEY = (
    "collision_role_unary_log_precision_delta"
)
COLLISION_ROLE_UNARY_PRECISION_AUX_KEY = "collision_role_unary_precision"
COLLISION_ROLE_FOLLOWER_EDGE_WEIGHT_AUX_KEY = "collision_role_follower_edge_weight"
COLLISION_ROLE_EDGE_SIGMA2_AUX_KEY = "collision_role_edge_sigma2"
COLLISION_ROLE_RESTRICTION_AUX_KEY = "collision_role_node_restriction"
LOCAL_SE2_PLAN_MEAN_AUX_KEY = "local_se2_plan_mean"
LOCAL_SE2_PLAN_PRECISION_AUX_KEY = "local_se2_plan_precision"
LOCAL_SE2_PLAN_USED_AUX_KEY = "local_se2_plan_used"
LOCAL_SE2_PLAN_HIDDEN_DELTA_AUX_KEY = "local_se2_plan_correction_hidden_delta"
LOCAL_SE2_PLAN_BASE_MEAN_AUX_KEY = "local_se2_plan_base_mean"
LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY = "local_se2_recovery_plan_delta"
LOCAL_SE2_RECOVERY_MODE_LOGITS_AUX_KEY = "local_se2_recovery_mode_logits"
LOCAL_SE2_RECOVERY_SELECTED_MODE_AUX_KEY = "local_se2_recovery_selected_mode"
LOCAL_SE2_DISTRIBUTED_RECOVERY_PRE_MEAN_AUX_KEY = (
    "local_se2_distributed_recovery_pre_mean"
)
LOCAL_SE2_DIRECT_WRENCH_PRE_AUX_KEY = "local_se2_direct_wrench_unary_pre"
LOCAL_SE2_DIRECT_WRENCH_DELTA_AUX_KEY = "local_se2_direct_wrench_unary_delta"
LOCAL_SE2_DIRECT_WRENCH_POST_AUX_KEY = "local_se2_direct_wrench_unary_post"
LOCAL_SE2_ANALYTIC_WRENCH_PROPOSAL_AUX_KEY = (
    "local_se2_analytic_total_wrench_proposal"
)
LOCAL_SE2_TERMINAL_PLAN_AUX_KEY = "local_se2_terminal_attractor_plan"
LOCAL_SE2_TERMINAL_ACTIVE_AUX_KEY = "local_se2_terminal_attractor_active"
LOCAL_SE2_DISTRIBUTED_GRASP_STATS_AUX_KEY = "local_se2_distributed_grasp_stats"
LOCAL_SE2_DISTRIBUTED_GRASP_MESSAGE_AUX_KEY = (
    "local_se2_distributed_grasp_message_trace"
)
LOCAL_SE2_DISTRIBUTED_GRASP_SETUP_TRANSMITTED_AUX_KEY = (
    "local_se2_distributed_grasp_setup_transmitted"
)
SLIT_PHASE_CENTERS = (0.60, 0.42, 0.25, 0.08, -0.10, -0.30)
SLIT_PHASE_BASIS_DIM = 6
SLIT_PHASE_CONTENT_DIM = 12


def _minimum_radius_spanning_tree_mask(edge_index: Tensor, n_nodes: int) -> Tensor:
    """Select a deterministic minimum-radius BFS tree from an undirected graph."""

    if edge_index.ndim != 2 or edge_index.shape[-1] != 2:
        raise ValueError("edge_index must have shape [edge,2]")
    pairs = [tuple(int(value) for value in pair) for pair in edge_index.cpu().tolist()]
    adjacency = [set() for _ in range(n_nodes)]
    for left, right in pairs:
        if left == right or not (0 <= left < n_nodes and 0 <= right < n_nodes):
            raise ValueError("invalid undirected radius edge")
        adjacency[left].add(right)
        adjacency[right].add(left)

    def distances(root: int) -> dict[int, int]:
        result = {root: 0}
        queue = [root]
        for node in queue:
            for neighbor in sorted(adjacency[node]):
                if neighbor not in result:
                    result[neighbor] = result[node] + 1
                    queue.append(neighbor)
        return result

    all_distances = [distances(root) for root in range(n_nodes)]
    if any(len(distance) != n_nodes for distance in all_distances):
        raise ValueError("analytic coordination requires a connected radius graph")
    root = min(
        range(n_nodes),
        key=lambda node: (max(all_distances[node].values()), node),
    )
    parents: dict[int, int] = {}
    visited = {root}
    queue = [root]
    for node in queue:
        for neighbor in sorted(adjacency[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = node
                queue.append(neighbor)
    tree = {
        (min(node, parent), max(node, parent))
        for node, parent in parents.items()
    }
    if len(tree) != n_nodes - 1:
        raise RuntimeError("minimum-radius BFS tree is malformed")
    return torch.tensor(
        [(min(left, right), max(left, right)) in tree for left, right in pairs],
        dtype=torch.bool,
        device=edge_index.device,
    )


class _ForwardExactBackwardProposed(torch.autograd.Function):
    """Keep the audited forward tensor bitwise, but train a zero-init proposal.

    A value-based ``torch.where(delta != 0, proposal, reference)`` preserves
    exact checkpoint parity but makes a zero-initialized proposal a dead
    branch: its Jacobian is identically zero.  This custom identity returns
    the already-selected forward tensor verbatim and routes its backward
    covector through ``proposal``.  Once the proposal is nonzero the selected
    forward tensor is the proposal itself, so the same backward contract
    remains correct.
    """

    @staticmethod
    def forward(ctx: object, reference: Tensor, proposal: Tensor) -> Tensor:
        if reference.shape != proposal.shape:
            raise ValueError("forward-exact proposal tensors must share shape")
        return reference.clone()

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[None, Tensor]:
        return None, gradient


def _forward_exact_backward_proposed(reference: Tensor, proposal: Tensor) -> Tensor:
    return _ForwardExactBackwardProposed.apply(reference, proposal)


def local_exact_clearance_observation_prefix(obs: Tensor) -> Tensor:
    """Return the canonical physical/exact-clearance22 observation prefix.

    Contact-feedback27 is a strict suffix extension.  Slit geometry helpers
    must never infer a new physical layout merely because five mechanical
    feedback coordinates were appended.
    """

    if obs.ndim != 3 or obs.shape[-1] not in {
        _SINGLE_SLIT_EXACT_CLEARANCE_DIM,
        _SINGLE_SLIT_CONTACT_OBSERVATION_DIM,
    }:
        raise ValueError(
            "local slit features require exact-clearance22 or contact-feedback27"
        )
    return obs[..., :_SINGLE_SLIT_EXACT_CLEARANCE_DIM]


def contact_conditioned_learned_observation(obs: Tensor) -> Tensor:
    """Gate the obs27 mechanical suffix by its explicit contact bit.

    The previous applied force is nonzero after ordinary safe motion.  It is
    useful only as the action that caused a rejected proposal, so allowing it
    to enter learned affine maps while ``last_hit == 0`` would silently break
    source preservation.  This local algebraic gate has no parameters and no
    wire cost.  obs18/obs22 are returned by identity.
    """

    if obs.ndim != 3:
        raise ValueError("policy observations must have shape [batch,agent,feature]")
    if obs.shape[-1] != _SINGLE_SLIT_CONTACT_OBSERVATION_DIM:
        return obs
    hit = obs[
        ..., _SINGLE_SLIT_CONTACT_HIT_INDEX : _SINGLE_SLIT_CONTACT_HIT_INDEX + 1
    ]
    gated_suffix = torch.cat(
        (hit, hit * obs[..., _SINGLE_SLIT_CONTACT_HIT_INDEX + 1 :]), dim=-1
    )
    return torch.cat(
        (obs[..., :_SINGLE_SLIT_EXACT_CLEARANCE_DIM], gated_suffix), dim=-1
    )


def local_slit_source_null_risk(obs: Tensor) -> Tensor:
    """Return the fixed local recovery envelope ``rho``.

    Clearance channel 18 is expressed in millimetres and clipped at 32 mm.
    The two wall-visibility bits are local sensor records.  Consequently this
    smooth envelope is exactly zero for either an invisible wall or a visible
    far-field observation at the 32 mm clip boundary, and approaches one as
    a visible handle enters the narrow-clearance regime.
    """

    obs = local_exact_clearance_observation_prefix(obs)
    raw_visibility = obs[..., 16] + obs[..., 17]
    visibility = torch.clamp(raw_visibility, 0.0, 1.0)
    clearance_mm = obs[..., 18]
    normalized_distance = (32.0 - clearance_mm) / 4.0
    rho = visibility * (1.0 - torch.exp(-0.5 * normalized_distance.square()))
    # These exact branches document and enforce the structural null set.  In
    # particular, no future algebraic refactor may turn 0 * a learned value
    # into a source perturbation through NaN/rounding behavior.
    null = (raw_visibility == 0.0) | (clearance_mm == 32.0)
    return torch.where(null, torch.zeros_like(rho), rho)


def local_slit_phase_features(
    obs: Tensor, detached_section: Tensor, env_cfg: EnvConfig
) -> tuple[Tensor, Tensor]:
    """Build the common strictly-local 6-RBF x 12-content adapter input.

    This utility is deliberately arm-independent so matched communication,
    raw-full and no-communication controls can receive the identical 216
    parameter decoder.  ``detached_section`` must be the arm's local semantic
    3-vector after its native communication operator.
    """

    obs = local_exact_clearance_observation_prefix(obs)
    if detached_section.shape != (*obs.shape[:-1], 3):
        raise ValueError("phase features require a local section3")
    visibility = obs[..., 16] + obs[..., 17]
    sin_theta, cos_theta = obs[..., 0], obs[..., 1]
    handle_x, handle_y = obs[..., 2], obs[..., 3]
    rotated_x = cos_theta * handle_x - sin_theta * handle_y
    rotated_y = sin_theta * handle_x + cos_theta * handle_y
    relative_x = (
        obs[..., 10] + obs[..., 11]
        + visibility * rotated_x / float(env_cfg.wall_sensor_range)
    )
    relative_y = (
        obs[..., 12] + obs[..., 13]
        + visibility * rotated_y / float(env_cfg.world_y)
    )
    section = torch.tanh(detached_section.detach())
    content = torch.stack(
        (
            torch.ones_like(visibility),
            section[..., 0], section[..., 1], section[..., 2],
            obs[..., 6], obs[..., 7], obs[..., 9], relative_y,
            obs[..., 18] / 32.0, obs[..., 19] / 32.0,
            obs[..., 20] / 32.0, obs[..., 21] / 32.0,
        ),
        dim=-1,
    )
    centers = obs.new_tensor(SLIT_PHASE_CENTERS)
    visibility_basis = torch.exp(-0.5 * ((visibility - 1.0) / 0.25).square())
    clearance_basis = torch.exp(-0.5 * ((obs[..., 18] - 2.0) / 8.0).square())
    position_basis = torch.exp(
        -0.5 * ((relative_x.unsqueeze(-1) - centers) / 0.16).square()
    )
    basis = visibility_basis.unsqueeze(-1) * clearance_basis.unsqueeze(-1) * position_basis
    features = (basis.unsqueeze(-1) * content.unsqueeze(-2)).flatten(-2)
    return features, basis


def local_slit_clearance_bank_features(
    obs: Tensor, detached_section: Tensor, env_cfg: EnvConfig
) -> tuple[Tensor, Tensor]:
    """Build fixed 4-clearance x 6-position RBFs crossed with local content12."""
    obs = local_exact_clearance_observation_prefix(obs)
    if detached_section.shape != (*obs.shape[:-1], 3):
        raise ValueError("clearance-bank features require a local section3")
    visibility=obs[...,16]+obs[...,17]
    sin_theta,cos_theta=obs[...,0],obs[...,1]
    handle_x,handle_y=obs[...,2],obs[...,3]
    rotated_x=cos_theta*handle_x-sin_theta*handle_y
    rotated_y=sin_theta*handle_x+cos_theta*handle_y
    relative_x=obs[...,10]+obs[...,11]+visibility*rotated_x/float(env_cfg.wall_sensor_range)
    relative_y=obs[...,12]+obs[...,13]+visibility*rotated_y/float(env_cfg.world_y)
    section=torch.tanh(detached_section.detach())
    content=torch.stack((
        torch.ones_like(visibility),section[...,0],section[...,1],section[...,2],
        obs[...,6],obs[...,7],obs[...,9],relative_y,
        obs[...,18]/32.0,obs[...,19]/32.0,obs[...,20]/32.0,obs[...,21]/32.0,
    ),dim=-1)
    position_centers=obs.new_tensor(SLIT_PHASE_CENTERS)
    clearance_centers=obs.new_tensor((0.0,5.0,13.0,21.0))
    visibility_basis=torch.exp(-0.5*((visibility-1.0)/0.25).square())
    position_basis=torch.exp(-0.5*((relative_x.unsqueeze(-1)-position_centers)/0.16).square())
    clearance_basis=torch.exp(-0.5*((obs[...,18].unsqueeze(-1)-clearance_centers)/3.0).square())
    basis=(visibility_basis[...,None,None]*clearance_basis.unsqueeze(-1)*position_basis.unsqueeze(-2)).flatten(-2)
    return (basis.unsqueeze(-1)*content.unsqueeze(-2)).flatten(-2),basis


def local_slit_approach_bank_features(
    obs: Tensor, detached_section: Tensor, env_cfg: EnvConfig
) -> tuple[Tensor, Tensor]:
    """Build the fixed 28mm/sigma2 approach RBF x six position bases."""
    obs = local_exact_clearance_observation_prefix(obs)
    if detached_section.shape!=(*obs.shape[:-1],3):
        raise ValueError("approach-bank features require a local section3")
    visibility=obs[...,16]+obs[...,17]
    sin_theta,cos_theta=obs[...,0],obs[...,1]
    handle_x,handle_y=obs[...,2],obs[...,3]
    rotated_x=cos_theta*handle_x-sin_theta*handle_y
    rotated_y=sin_theta*handle_x+cos_theta*handle_y
    relative_x=obs[...,10]+obs[...,11]+visibility*rotated_x/float(env_cfg.wall_sensor_range)
    relative_y=obs[...,12]+obs[...,13]+visibility*rotated_y/float(env_cfg.world_y)
    section=torch.tanh(detached_section.detach())
    content=torch.stack((torch.ones_like(visibility),section[...,0],section[...,1],section[...,2],obs[...,6],obs[...,7],obs[...,9],relative_y,obs[...,18]/32.0,obs[...,19]/32.0,obs[...,20]/32.0,obs[...,21]/32.0),dim=-1)
    position_centers=obs.new_tensor(SLIT_PHASE_CENTERS)
    visibility_basis=torch.exp(-0.5*((visibility-1.0)/0.25).square())
    clearance_basis=torch.exp(-0.5*((obs[...,18]-28.0)/2.0).square())
    position_basis=torch.exp(-0.5*((relative_x.unsqueeze(-1)-position_centers)/0.16).square())
    basis=visibility_basis.unsqueeze(-1)*clearance_basis.unsqueeze(-1)*position_basis
    return (basis.unsqueeze(-1)*content.unsqueeze(-2)).flatten(-2),basis
RETURN_CANDIDATE_CHECKPOINT_SCHEMA = "sheaf-gbp-return-candidate-v2"
RETURN_CANDIDATE_V2_INPUT_SEMANTICS = (
    "sin_theta",
    "cos_theta",
    "payload_center_x_norm",
    "payload_center_y_norm",
    "active_goal_x_norm",
    "active_goal_y_norm",
    "temporal_memory_age_norm",
    *(
        f"detached_geometry_anchored_learned_section_{index}"
        for index in range(_RETURN_SELECTOR_SECTION_DIM)
    ),
)
RETURN_CANDIDATE_V3_INPUT_SEMANTICS = (
    "sin_theta",
    "cos_theta",
    "active_goal_minus_payload_center_x_over_world_x",
    "active_goal_minus_payload_center_y_over_world_y",
    "handle_body_x",
    "handle_body_y",
    "contact_velocity_body_x_over_2",
    "contact_velocity_body_y_over_2",
    "source_force_body_x_over_max_force",
    "source_force_body_y_over_max_force",
    "source_force_norm_over_max_force",
    "temporal_memory_age_norm",
    "wall_0_visible",
    "wall_0_minus_payload_center_x_over_sensor_range",
    "gap_0_minus_payload_center_y_over_world_y",
    "gap_0_half_width_over_world_y",
    "wall_1_visible",
    "wall_1_minus_payload_center_x_over_sensor_range",
    "gap_1_minus_payload_center_y_over_world_y",
    "gap_1_half_width_over_world_y",
    *(f"detached_learned_sheaf_section_{index}" for index in range(12)),
)
# Historical public name retained for V2 checkpoint/audit compatibility.
RETURN_CANDIDATE_INPUT_SEMANTICS = RETURN_CANDIDATE_V2_INPUT_SEMANTICS


def return_candidate_architecture_metadata(
    architecture: str = RETURN_CANDIDATE_ARCHITECTURE_V2,
) -> dict[str, object]:
    """Exact, serialization-safe declaration of a deployable candidate."""

    if architecture == RETURN_CANDIDATE_ARCHITECTURE_V3:
        return {
            "schema_version": "sheaf-gbp-return-candidate-v3",
            "architecture": RETURN_CANDIDATE_ARCHITECTURE_V3,
            "input_dim": _RETURN_CANDIDATE_V3_INPUT_DIM,
            "hidden_dims": [
                _RETURN_CANDIDATE_V3_HIDDEN_DIM,
                _RETURN_CANDIDATE_V3_HIDDEN_DIM,
            ],
            "output_dim": _RETURN_CANDIDATE_OUTPUT_DIM,
            "parameter_count": 43_971,
            "input_semantics": list(RETURN_CANDIDATE_V3_INPUT_SEMANTICS),
            "section_detached": True,
            "opaque_stalk_input": False,
            "raw_decoder_input": False,
            "trainable_selector": False,
            "always_candidate_on_return": True,
            "additional_inference_wire_bytes": 0,
        }
    if architecture != RETURN_CANDIDATE_ARCHITECTURE_V2:
        raise ValueError(f"architecture metadata is unavailable for {architecture!r}")

    return {
        "schema_version": RETURN_CANDIDATE_CHECKPOINT_SCHEMA,
        "architecture": RETURN_CANDIDATE_ARCHITECTURE_V2,
        "input_dim": _RETURN_CANDIDATE_INPUT_DIM,
        "hidden_dims": [_RETURN_CANDIDATE_HIDDEN_DIM, _RETURN_CANDIDATE_HIDDEN_DIM],
        "output_dim": _RETURN_CANDIDATE_OUTPUT_DIM,
        "parameter_count": 18_947,
        "input_semantics": list(RETURN_CANDIDATE_V2_INPUT_SEMANTICS),
        "section_detached": True,
        "opaque_stalk_input": False,
        "raw_decoder_input": False,
        "additional_inference_wire_bytes": 0,
    }


def return_candidate_physical_features_v3(
    obs: Tensor,
    memory_count: Tensor,
    source_forces: Tensor,
    section: Tensor,
    cfg: EnvConfig,
    *,
    temporal_window: int,
) -> Tensor:
    """Build V3's auditable 20 physical + 12 sheaf coordinates.

    All quantities are reconstructed from the canonical strict-local
    observation, the already-computed immutable source action, and the
    existing post-GBP section.  No peer state or new edge message is read.
    Absolute payload/map coordinates are deliberately absent: goals, walls,
    and slits are expressed relative to the payload centre.  The full sheaf
    section is detached because V3 trains only the residual adapter.
    """

    if obs.ndim != 3 or obs.shape[-1] != 18:
        raise ValueError("V3 return candidate requires [batch, agent, 18] strict observations")
    if memory_count.shape != obs.shape[:2]:
        raise ValueError("V3 memory_count must have shape [batch, agent]")
    if source_forces.shape != (*obs.shape[:2], 2):
        raise ValueError("V3 source_forces must have shape [batch, agent, 2]")
    if section.shape != (*obs.shape[:2], _RETURN_CANDIDATE_V3_SECTION_DIM):
        raise ValueError(
            "V3 requires the complete 12-d learned sheaf section with shape "
            f"{(*obs.shape[:2], _RETURN_CANDIDATE_V3_SECTION_DIM)}"
        )
    for name, value in (
        ("memory_count", memory_count),
        ("source_forces", source_forces),
        ("section", section),
    ):
        if value.device != obs.device or value.dtype != obs.dtype:
            raise ValueError(f"V3 {name} must share observation device and dtype")
    if temporal_window < 0:
        raise ValueError("V3 temporal_window must be non-negative")
    if cfg.world_x <= 0.0 or cfg.world_y <= 0.0 or cfg.wall_sensor_range <= 0.0:
        raise ValueError("V3 physical normalizers must be positive")
    if cfg.max_force <= 0.0:
        raise ValueError("V3 max_force must be positive")

    sin_theta = obs[..., 0]
    cos_theta = obs[..., 1]
    body_x = obs[..., 2]
    body_y = obs[..., 3]
    offset_world_x = cos_theta * body_x - sin_theta * body_y
    offset_world_y = sin_theta * body_x + cos_theta * body_y

    # Strict observations store active_goal - handle in channels 8:10.
    # Adding the handle offset recovers active_goal - payload_center.
    goal_center_x = (2.6 * obs[..., 8] + offset_world_x) / cfg.world_x
    goal_center_y = (2.6 * obs[..., 9] + offset_world_y) / cfg.world_y

    # Contact velocity channels are already divided by two.  Rotate both the
    # velocity and exact source force into the payload frame so handle roles
    # transfer across route reflection and payload orientation.
    velocity_body_x = cos_theta * obs[..., 6] + sin_theta * obs[..., 7]
    velocity_body_y = -sin_theta * obs[..., 6] + cos_theta * obs[..., 7]
    source_x = source_forces[..., 0] / cfg.max_force
    source_y = source_forces[..., 1] / cfg.max_force
    source_body_x = cos_theta * source_x + sin_theta * source_y
    source_body_y = -sin_theta * source_x + cos_theta * source_y
    source_saturation = torch.linalg.vector_norm(source_forces, dim=-1) / cfg.max_force
    memory_age = memory_count / float(max(1, temporal_window))

    # Channels 10:18 contain two local wall records grouped by field.  When a
    # wall is invisible every derived coordinate must remain exactly zero;
    # otherwise adding the handle offset expresses it around payload centre.
    visibility = obs[..., 16:18]
    wall_center_x = visibility * (
        obs[..., 10:12] + offset_world_x[..., None] / cfg.wall_sensor_range
    )
    gap_center_y = visibility * (obs[..., 12:14] + offset_world_y[..., None] / cfg.world_y)
    gap_half = obs[..., 14:16]
    wall_features = torch.stack(
        (visibility, wall_center_x, gap_center_y, gap_half),
        dim=-1,
    ).reshape(*obs.shape[:2], 8)

    analytic = torch.cat(
        (
            torch.stack(
                (
                    sin_theta,
                    cos_theta,
                    goal_center_x,
                    goal_center_y,
                    body_x,
                    body_y,
                    velocity_body_x,
                    velocity_body_y,
                    source_body_x,
                    source_body_y,
                    source_saturation,
                    memory_age,
                ),
                dim=-1,
            ),
            wall_features,
        ),
        dim=-1,
    )
    if analytic.shape[-1] != _RETURN_CANDIDATE_V3_ANALYTIC_DIM:
        raise RuntimeError("V3 analytic feature width changed")
    features = torch.cat((analytic, section.detach()), dim=-1)
    if features.shape[-1] != _RETURN_CANDIDATE_V3_INPUT_DIM:
        raise RuntimeError("V3 return-candidate input width changed")
    return features


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        modules.extend((nn.Linear(last, hidden_dim), nn.GELU()))
        last = hidden_dim
    modules.append(nn.Linear(last, out_dim))
    return nn.Sequential(*modules)


class ReturnManeuverAdapter(nn.Module):
    """Hard six-mode return adapter with one shared two-layer trunk."""

    def __init__(self, in_dim: int, hidden_dim: int, modes: int):
        super().__init__()
        if modes != 6:
            raise ValueError("the maneuver adapter currently requires exactly six modes")
        self.modes = modes
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, modes)
        self.wrench_head = nn.Linear(hidden_dim, modes * 3)
        # The adapter is algebraically absent at initialization.  Zero scores
        # also give a deterministic hard tie-break to mode zero.
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        nn.init.zeros_(self.wrench_head.weight)
        nn.init.zeros_(self.wrench_head.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        trunk = self.trunk(features)
        scores = self.score_head(trunk)
        raw_wrenches = self.wrench_head(trunk).view(
            *features.shape[:-1],
            self.modes,
            3,
        )
        return scores, raw_wrenches


class ReturnSourceCandidateAdapter(nn.Module):
    """Source-anchored candidate with exact V1/V2 restore and direct V3.

    Both heads see the same seven locally reconstructed payload invariants and
    the geometry-anchored eight-coordinate prefix of the existing learned
    sheaf section.  (The learned low-rank restriction residual means these are
    not claimed to retain strict coordinate-wise physical semantics.)  In V2
    the candidate is exactly ``15 -> 128 -> 128 -> 3``; it cannot inspect the
    opaque stalk or raw decoder input.  Its bounded wrench is detached before
    it enters the unchanged 18-d selector.  Inputs are detached from the frozen
    coordination core and add no edge communication.  V3 is a direct residual
    ``32 -> 192 -> 192 -> 3`` over explicit dynamics/geometry/source features
    and the complete 12-d section.  It has no selector parameters: its bounded
    residual is always the return-leg candidate.  V1 is retained only so
    historical checkpoints missing an architecture field remain exactly
    reproducible.
    """

    def __init__(
        self,
        architecture: str,
        *,
        legacy_in_dim: int | None = None,
        legacy_hidden_dim: int | None = None,
    ):
        super().__init__()
        if architecture not in {
            RETURN_CANDIDATE_ARCHITECTURE_V1,
            RETURN_CANDIDATE_ARCHITECTURE_V2,
            RETURN_CANDIDATE_ARCHITECTURE_V3,
        }:
            raise ValueError(f"unsupported source-candidate architecture {architecture!r}")
        self.architecture = architecture
        if architecture == RETURN_CANDIDATE_ARCHITECTURE_V1:
            if legacy_in_dim is None or legacy_hidden_dim is None:
                raise ValueError("legacy source candidate requires decoder and hidden widths")
            candidate_in_dim = legacy_in_dim
            candidate_hidden_dim = legacy_hidden_dim
        elif architecture == RETURN_CANDIDATE_ARCHITECTURE_V2:
            candidate_in_dim = _RETURN_CANDIDATE_INPUT_DIM
            candidate_hidden_dim = _RETURN_CANDIDATE_HIDDEN_DIM
        else:
            candidate_in_dim = _RETURN_CANDIDATE_V3_INPUT_DIM
            candidate_hidden_dim = _RETURN_CANDIDATE_V3_HIDDEN_DIM
        self.trunk = nn.Sequential(
            nn.Linear(candidate_in_dim, candidate_hidden_dim),
            nn.GELU(),
            nn.Linear(candidate_hidden_dim, candidate_hidden_dim),
            nn.GELU(),
        )
        if architecture != RETURN_CANDIDATE_ARCHITECTURE_V3:
            self.selector_trunk = nn.Sequential(
                nn.Linear(_RETURN_SELECTOR_DIM, _RETURN_SELECTOR_HIDDEN_DIM),
                nn.GELU(),
            )
            self.advantage_head = nn.Linear(_RETURN_SELECTOR_HIDDEN_DIM, 1)
        self.wrench_head = nn.Linear(
            candidate_hidden_dim,
            _RETURN_CANDIDATE_OUTPUT_DIM,
        )
        # V1/V2 retain their negative source prior exactly.  V3 has no
        # selector; its zero wrench makes the direct residual algebraically
        # absent on both legs at initialization.
        if architecture != RETURN_CANDIDATE_ARCHITECTURE_V3:
            nn.init.zeros_(self.advantage_head.weight)
            nn.init.constant_(self.advantage_head.bias, _RETURN_SELECTOR_SOURCE_BIAS)
        nn.init.zeros_(self.wrench_head.weight)
        nn.init.zeros_(self.wrench_head.bias)

    def forward(
        self,
        decoder_features: Tensor,
        selector_invariants: Tensor,
        section: Tensor,
        *,
        expert_wrench_scale: float,
        obs: Tensor | None = None,
        memory_count: Tensor | None = None,
        source_forces: Tensor | None = None,
        env_cfg: EnvConfig | None = None,
        temporal_window: int | None = None,
    ) -> tuple[Tensor | None, Tensor]:
        if decoder_features.shape[:-1] != section.shape[:-1]:
            raise ValueError("decoder features must share the source-candidate prefix")
        if self.architecture == RETURN_CANDIDATE_ARCHITECTURE_V3:
            if (
                obs is None
                or memory_count is None
                or source_forces is None
                or env_cfg is None
                or temporal_window is None
            ):
                raise ValueError(
                    "V3 source candidate requires canonical obs, memory_count, "
                    "exact source_forces, env_cfg, and temporal_window"
                )
            candidate_features = return_candidate_physical_features_v3(
                obs,
                memory_count,
                source_forces,
                section,
                env_cfg,
                temporal_window=temporal_window,
            ).detach()
        else:
            if selector_invariants.shape != (
                *section.shape[:-1],
                _RETURN_SELECTOR_INVARIANT_DIM,
            ):
                raise ValueError(
                    "source selector invariants must have shape "
                    f"{(*section.shape[:-1], _RETURN_SELECTOR_INVARIANT_DIM)}"
                )
            if section.shape[-1] < _RETURN_SELECTOR_SECTION_DIM:
                raise ValueError(
                    "source candidate section must share the invariant prefix and "
                    f"have at least {_RETURN_SELECTOR_SECTION_DIM} coordinates"
                )
            geometry_section = section[..., :_RETURN_SELECTOR_SECTION_DIM].detach()
        if self.architecture == RETURN_CANDIDATE_ARCHITECTURE_V1:
            candidate_features = decoder_features
        elif self.architecture == RETURN_CANDIDATE_ARCHITECTURE_V2:
            candidate_features = torch.cat(
                (selector_invariants.detach(), geometry_section),
                dim=-1,
            )
            if candidate_features.shape[-1] != _RETURN_CANDIDATE_INPUT_DIM:
                raise RuntimeError("source candidate physical input width changed")
        trunk = self.trunk(candidate_features)
        expert_wrench = expert_wrench_scale * torch.tanh(self.wrench_head(trunk))
        if self.architecture == RETURN_CANDIDATE_ARCHITECTURE_V3:
            return None, expert_wrench
        selector_features = torch.cat(
            (
                selector_invariants.detach(),
                geometry_section,
                expert_wrench.detach(),
            ),
            dim=-1,
        )
        selector = self.selector_trunk(selector_features)
        return self.advantage_head(selector).squeeze(-1), expert_wrench


@dataclass
class PolicyOutput:
    forces: Tensor
    hidden: Tensor
    aux: dict[str, Tensor | GBPMessageTrace]


@dataclass
class PhysicalObservationHistory:
    """Fixed-width FIFO of raw, pre-canonicalization observations."""

    values: Tensor
    valid: Tensor


def mirror_handle_permutation(n_agents: int, device: torch.device | str) -> Tensor:
    """Map each handle to its exact left-right reflected counterpart."""

    if not 4 <= n_agents <= 20:
        raise ValueError(f"no mirror handle permutation for {n_agents} agents")
    top_count = (n_agents + 1) // 2
    permutation = list(reversed(range(top_count)))
    remainder = n_agents - top_count
    for offset in range(0, remainder - remainder % 2, 2):
        left = top_count + offset
        permutation.extend((left + 1, left))
    if remainder % 2:
        permutation.append(n_agents - 1)
    return torch.tensor(permutation, dtype=torch.long, device=device)


def mirror_strict_observations(obs: Tensor, permutation: Tensor) -> Tensor:
    """Reflect 18-d strict-local observations into the opposite route frame."""

    if obs.shape[-1] != 18:
        raise ValueError("route canonicalization currently requires 18-d strict-local observations")
    mirrored = obs.index_select(1, permutation).clone()
    original = mirrored.clone()
    mirrored[..., [0, 2, 4, 6, 8]] *= -1.0
    mirrored[..., 10] = -original[..., 11]
    mirrored[..., 11] = -original[..., 10]
    mirrored[..., 12] = original[..., 13]
    mirrored[..., 13] = original[..., 12]
    mirrored[..., 14] = original[..., 15]
    mirrored[..., 15] = original[..., 14]
    mirrored[..., 16] = original[..., 17]
    mirrored[..., 17] = original[..., 16]
    return mirrored


def return_selector_invariant_features(
    obs: Tensor,
    memory_count: Tensor,
    cfg: EnvConfig,
    *,
    temporal_window: int,
) -> Tensor:
    """Reconstruct the seven shared return-selector features at every handle.

    ``observations`` stores strict-local handle coordinates as follows:

    - body-frame handle position in channels ``2:4``;
    - world x divided by 2.6 and world y divided by ``world_y`` in ``4:6``;
    - both coordinates of the handle-to-goal delta divided by 2.6 in ``8:10``.

    Undoing those exact scales lets every agent reconstruct the same payload
    centre and active final goal without another message or a global reduction.
    The input must already be in the route-canonical frame used by the policy.
    """

    if obs.ndim != 3 or obs.shape[-1] != 18:
        raise ValueError("source selector requires [batch, agent, 18] strict observations")
    if memory_count.shape != obs.shape[:2]:
        raise ValueError("memory_count must have shape [batch, agent]")
    if memory_count.device != obs.device or memory_count.dtype != obs.dtype:
        raise ValueError("memory_count must share observation device and dtype")
    if temporal_window < 0:
        raise ValueError("temporal_window must be non-negative")

    sin_theta = obs[..., 0]
    cos_theta = obs[..., 1]
    body_x = obs[..., 2]
    body_y = obs[..., 3]
    handle_world_x = 2.6 * obs[..., 4]
    handle_world_y = cfg.world_y * obs[..., 5]
    handle_offset_x = cos_theta * body_x - sin_theta * body_y
    handle_offset_y = sin_theta * body_x + cos_theta * body_y
    payload_center_x = handle_world_x - handle_offset_x
    payload_center_y = handle_world_y - handle_offset_y
    # Both goal-delta coordinates use 2.6 in env.observations; notably, the y
    # channel does not use cfg.world_y until the final invariant normalization.
    active_goal_x = handle_world_x + 2.6 * obs[..., 8]
    active_goal_y = handle_world_y + 2.6 * obs[..., 9]
    memory_age = memory_count / float(max(1, temporal_window))
    return torch.stack(
        (
            sin_theta,
            cos_theta,
            payload_center_x / 2.6,
            payload_center_y / cfg.world_y,
            active_goal_x / 2.6,
            active_goal_y / cfg.world_y,
            memory_age,
        ),
        dim=-1,
    )


def local_se2_recovery_canonical_features(obs: Tensor, cfg: EnvConfig) -> Tensor:
    """Reconstruct a shared geometric planning frame from each local handle.

    Every coordinate uses only the existing strict-local observation.  The
    payload centre, goal displacement, and visible slit geometry are therefore
    identical across handles up to floating-point roundoff, while the local
    contact velocity retains the agent's mechanically distinct role.  These
    private features add no edge packet or communication round.
    """

    if obs.ndim != 3 or obs.shape[-1] != _SINGLE_SLIT_CONTACT_OBSERVATION_DIM:
        raise ValueError("canonical recovery features require [batch,agent,27]")
    sin_theta = obs[..., 0]
    cos_theta = obs[..., 1]
    body_x = obs[..., 2]
    body_y = obs[..., 3]
    offset_x = cos_theta * body_x - sin_theta * body_y
    offset_y = sin_theta * body_x + cos_theta * body_y
    handle_world_x = 2.6 * obs[..., 4]
    handle_world_y = cfg.world_y * obs[..., 5]
    centre_x = handle_world_x - offset_x
    centre_y = handle_world_y - offset_y
    goal_delta_x = 2.6 * obs[..., 8] + offset_x
    velocity_body_x = cos_theta * obs[..., 6] + sin_theta * obs[..., 7]
    velocity_body_y = -sin_theta * obs[..., 6] + cos_theta * obs[..., 7]

    visibility = obs[..., 16:18]
    visible_count = visibility.sum(dim=-1).clamp_min(1.0)
    wall_relative_x = (
        visibility
        * (obs[..., 10:12] + offset_x[..., None] / cfg.wall_sensor_range)
    ).sum(dim=-1) / visible_count
    gap_relative_y = (
        visibility
        * (obs[..., 12:14] + offset_y[..., None] / cfg.world_y)
    ).sum(dim=-1) / visible_count
    gap_half = (visibility * obs[..., 14:16]).sum(dim=-1) / visible_count
    any_visible = visibility.amax(dim=-1)
    # Under strict-local-current-slit-v2, channel 9 is the visibility-gated
    # current-slit y displacement in legacy /2.6 units, not final-goal y.
    # Adding the handle offset converts it to slit-minus-payload-centre.
    slit_delta_y = any_visible * (2.6 * obs[..., 9] + offset_y)
    result = torch.stack(
        (
            sin_theta,
            cos_theta,
            centre_x / cfg.world_x,
            centre_y / cfg.world_y,
            goal_delta_x / cfg.world_x,
            slit_delta_y / cfg.world_y,
            velocity_body_x,
            velocity_body_y,
            wall_relative_x,
            gap_relative_y,
            gap_half,
            any_visible,
        ),
        dim=-1,
    )
    if result.shape != (*obs.shape[:2], LOCAL_SE2_RECOVERY_CANONICAL_DIM):
        raise RuntimeError("canonical recovery feature width changed")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("canonical recovery features contain non-finite values")
    return result


def split_strict_physical_goal(obs: Tensor) -> tuple[Tensor, Tensor]:
    """Split the audited strict observation into physical and goal channels.

    Physical sensing is exactly ``obs[..., 0:8, 10:18]`` (16 floats); only
    ``obs[..., 8:10]`` changes when the active endpoint changes.  Keeping this
    split explicit prevents a goal-conditioned latent from being mistaken for
    temporal physical memory.
    """

    if obs.shape[-1] != 18:
        raise ValueError("queryable goal-free temporal memory requires 18-d strict observations")
    return obs[..., _STRICT_PHYSICAL_INDICES], obs[..., _STRICT_GOAL_SLICE]


def mirror_strict_physical_observations(
    physical: Tensor,
    permutation: Tensor,
) -> Tensor:
    """Reflect a 16-d goal-free strict observation into the other route frame."""

    if physical.shape[-1] != _STRICT_PHYSICAL_DIM:
        raise ValueError("strict physical observations must have 16 channels")
    mirrored = physical.index_select(1, permutation).clone()
    original = mirrored.clone()
    mirrored[..., [0, 2, 4, 6]] *= -1.0
    # The two wall channels in each group exchange near/far wall identities
    # under reflection.  Relative x also changes parity; y, width, and
    # visibility only exchange identities.
    mirrored[..., 8] = -original[..., 9]
    mirrored[..., 9] = -original[..., 8]
    mirrored[..., 10] = original[..., 11]
    mirrored[..., 11] = original[..., 10]
    mirrored[..., 12] = original[..., 13]
    mirrored[..., 13] = original[..., 12]
    mirrored[..., 14] = original[..., 15]
    mirrored[..., 15] = original[..., 14]
    return mirrored


def mirror_forces(forces: Tensor, permutation: Tensor) -> Tensor:
    """Reflect canonical per-handle forces back to the physical frame."""

    mirrored = forces.index_select(1, permutation).clone()
    mirrored[..., 0] *= -1.0
    return mirrored


def policy_route_sign(
    policy: "BasePolicy",
    mission_sign: Tensor,
    active_leg: Tensor | None,
) -> Tensor:
    """Choose the controller's canonical route sign.

    The legacy/default behavior keeps one mission frame for the complete
    episode.  Active-leg canonicalization is algebraically identical on leg 0
    and flips only after the stable round-trip turn.
    """

    enabled = bool(
        getattr(
            getattr(policy, "arm_cfg", None),
            "active_leg_canonicalization",
            False,
        )
    )
    if not enabled or active_leg is None:
        return mission_sign
    leg_direction = 1.0 - 2.0 * active_leg.to(mission_sign.dtype)
    return mission_sign * leg_direction


def init_physical_observation_history(
    obs: Tensor,
    steps: int,
) -> PhysicalObservationHistory:
    """Allocate an empty per-episode raw-observation FIFO."""

    if steps < 0:
        raise ValueError("turn replay steps must be non-negative")
    return PhysicalObservationHistory(
        values=obs.new_zeros(obs.shape[0], steps, *obs.shape[1:]),
        valid=torch.zeros(obs.shape[0], steps, dtype=torch.bool, device=obs.device),
    )


def append_physical_observation(
    history: PhysicalObservationHistory,
    obs: Tensor,
    update_mask: Tensor,
) -> PhysicalObservationHistory:
    """Append exact physical observations for selected live outbound rows."""

    if history.values.shape[1] == 0:
        return history
    shifted_values = torch.cat((history.values[:, 1:], obs.unsqueeze(1)), dim=1)
    shifted_valid = torch.cat(
        (
            history.valid[:, 1:],
            torch.ones(obs.shape[0], 1, dtype=torch.bool, device=obs.device),
        ),
        dim=1,
    )
    value_mask = update_mask.view(update_mask.shape[0], 1, 1, 1)
    valid_mask = update_mask.view(update_mask.shape[0], 1)
    return PhysicalObservationHistory(
        values=torch.where(value_mask, shifted_values, history.values),
        valid=torch.where(valid_mask, shifted_valid, history.valid),
    )


def clear_physical_observation_history(
    history: PhysicalObservationHistory,
    clear_mask: Tensor,
) -> PhysicalObservationHistory:
    """Clear completed/reset DAgger chains without affecting other rows."""

    value_mask = clear_mask.view(clear_mask.shape[0], 1, 1, 1)
    valid_mask = clear_mask.view(clear_mask.shape[0], 1)
    return PhysicalObservationHistory(
        values=torch.where(value_mask, torch.zeros_like(history.values), history.values),
        valid=torch.where(valid_mask, torch.zeros_like(history.valid), history.valid),
    )


def mirror_turn_memory(
    policy: "BasePolicy",
    hidden: Tensor,
    turn_mask: Tensor,
) -> Tensor:
    """Carry hidden state into the reflected active-leg frame.

    Only the agent/handle axis is permuted.  Recurrent feature channels are
    left intact: this is a memory-preserving coordinate transition, not a
    hidden reset or a learned parity transform.
    """

    # Queryable goal-free memory stores canonical physical frames together
    # with their active-leg labels.  It is re-expressed lazily when queried,
    # so mutating it at a turn would double-reflect the history.  Crucially,
    # this mode carries no opaque goal-conditioned recurrent latent.
    if bool(
        getattr(
            getattr(policy, "arm_cfg", None),
            "queryable_goal_free_temporal",
            False,
        )
    ):
        return hidden

    permutation = mirror_handle_permutation(policy.env_cfg.n_agents, hidden.device)
    mirrored_hidden = hidden.index_select(1, permutation)
    hidden_mask = turn_mask.view(turn_mask.shape[0], *([1] * (hidden.ndim - 1)))
    return torch.where(hidden_mask, mirrored_hidden, hidden)


def mirror_and_replay_turn_memory(
    policy: "BasePolicy",
    hidden: Tensor,
    history: PhysicalObservationHistory,
    edge_index: Tensor,
    *,
    turn_mask: Tensor,
    route_sign: Tensor,
    active_leg: Tensor,
) -> tuple[Tensor, Tensor]:
    """Carry memory across a frame-changing turn and replay raw history.

    This is deliberately not a memory reset.  Selected rows first retain their
    full carried hidden tensor under the exact handle mirror permutation.  Each
    valid cached physical observation is then processed, oldest to newest,
    through the new active-leg canonical transform.  The returned per-episode
    replay count is used to charge the additional distributed GBP messages.
    """

    rebuilt = mirror_turn_memory(policy, hidden, turn_mask)
    replay_count = torch.zeros(turn_mask.shape[0], dtype=torch.long, device=turn_mask.device)
    for replay_index in range(history.values.shape[1]):
        replay_mask = turn_mask & history.valid[:, replay_index]
        if not bool(replay_mask.any()):
            continue
        replay_output = route_canonical_policy_step(
            policy,
            history.values[:, replay_index],
            edge_index,
            rebuilt,
            route_sign=route_sign,
            active_leg=active_leg,
        )
        replay_hidden_mask = replay_mask.view(replay_mask.shape[0], *([1] * (rebuilt.ndim - 1)))
        rebuilt = torch.where(replay_hidden_mask, replay_output.hidden, rebuilt)
        replay_count = replay_count + replay_mask.to(torch.long)
    return rebuilt, replay_count


def _mixed_frame(original: Tensor, mirrored: Tensor, reverse: Tensor) -> Tensor:
    mask = reverse.view(reverse.shape[0], *([1] * (original.ndim - 1)))
    return torch.where(mask, mirrored, original)


def _mirror_section(section: Tensor, permutation: Tensor) -> Tensor:
    mirrored = section.index_select(1, permutation).clone()
    parity = [index for index in (0, 2, 3, 5) if index < mirrored.shape[-1]]
    if parity:
        mirrored[..., parity] *= -1.0
    return mirrored


def _mirror_wrench(wrench: Tensor, permutation: Tensor) -> Tensor:
    """Reflect a physical ``[Fx, Fy, tau]`` vector and its handle owner."""

    if wrench.shape[-1] != _CORRECTION_WRENCH_DIM:
        raise ValueError("a physical wrench must have exactly three coordinates")
    mirrored = wrench.index_select(1, permutation).clone()
    mirrored[..., [0, 2]] *= -1.0
    return mirrored


def _directed_mirror_permutation(edge_index: Tensor, permutation: Tensor) -> Tensor:
    """Map physical directed-edge slots to canonical directed-edge slots."""

    directed_src = edge_index.reshape(-1).long()
    directed_dst = edge_index.flip(-1).reshape(-1).long()
    canonical_src = permutation.index_select(0, directed_src)
    canonical_dst = permutation.index_select(0, directed_dst)
    matches = (directed_src[None, :] == canonical_src[:, None]) & (
        directed_dst[None, :] == canonical_dst[:, None]
    )
    if not bool(matches.any(dim=1).all()):
        raise ValueError("handle reflection is not an automorphism of the communication graph")
    return matches.to(torch.int64).argmax(dim=1)


def route_canonical_policy_step(
    policy: "BasePolicy",
    obs: Tensor,
    edge_index: Tensor,
    hidden: Tensor | None = None,
    *,
    route_sign: Tensor,
    active_leg: Tensor | None = None,
    trace_messages: bool = False,
) -> PolicyOutput:
    """Run a policy in the controller-selected canonical route frame.

    A negative route sign reflects observations before the encoder and maps
    actions back before dynamics.  The caller owns any recurrent-memory frame
    transition; ``mirror_and_replay_turn_memory`` implements the round-trip
    active-leg protocol.
    """

    enabled = bool(getattr(getattr(policy, "arm_cfg", None), "route_canonicalization", False))
    kwargs = {"trace_messages": True} if trace_messages else {}
    if bool(getattr(getattr(policy, "arm_cfg", None), "return_adapter", False)) or bool(
        getattr(
            getattr(policy, "arm_cfg", None),
            "queryable_goal_free_temporal",
            False,
        )
    ):
        kwargs["active_leg"] = active_leg
    if not enabled:
        return policy(obs, edge_index, hidden, **kwargs)
    if not policy.env_cfg.strict_local_observation:
        raise ValueError("route canonicalization requires --strict-local-observation")

    reverse = route_sign < 0
    permutation = mirror_handle_permutation(policy.env_cfg.n_agents, obs.device)
    canonical_obs = _mixed_frame(
        obs,
        mirror_strict_observations(obs, permutation),
        reverse,
    )
    output = policy(canonical_obs, edge_index, hidden, **kwargs)
    physical_forces = _mixed_frame(
        output.forces,
        mirror_forces(output.forces, permutation),
        reverse,
    )

    aux = dict(output.aux)
    if "section" in aux:
        aux["section"] = _mixed_frame(
            aux["section"], _mirror_section(aux["section"], permutation), reverse
        )
    if "analytic_force" in aux:
        aux["analytic_force"] = _mixed_frame(
            aux["analytic_force"], mirror_forces(aux["analytic_force"], permutation), reverse
        )
    if "base_forces" in aux:
        aux["base_forces"] = _mixed_frame(
            aux["base_forces"], mirror_forces(aux["base_forces"], permutation), reverse
        )
    if SOURCE_FORCES_AUX_KEY in aux:
        source_forces = aux[SOURCE_FORCES_AUX_KEY]
        aux[SOURCE_FORCES_AUX_KEY] = _mixed_frame(
            source_forces,
            mirror_forces(source_forces, permutation),
            reverse,
        )
    if "return_residual" in aux:
        aux["return_residual"] = _mixed_frame(
            aux["return_residual"], mirror_forces(aux["return_residual"], permutation), reverse
        )
    if CORRECTION_FORCE_AUX_KEY in aux:
        correction_force = aux[CORRECTION_FORCE_AUX_KEY]
        aux[CORRECTION_FORCE_AUX_KEY] = _mixed_frame(
            correction_force,
            mirror_forces(correction_force, permutation),
            reverse,
        )
    for key in (
        "return_source_expert_raw_forces",
        "return_source_expert_forces",
    ):
        if key in aux:
            aux[key] = _mixed_frame(
                aux[key],
                mirror_forces(aux[key], permutation),
                reverse,
            )
    for key in (
        "return_wrench_residual",
        "return_selected_wrench",
        "return_source_expert_wrench",
        CORRECTION_SECTION_AUX_KEY,
        CORRECTION_WRENCH_AUX_KEY,
        CORRECTION_PUBLIC_UNARY_DELTA_AUX_KEY,
        CORRECTION_PUBLIC_UNARY_PRE_AUX_KEY,
        CORRECTION_PUBLIC_UNARY_POST_AUX_KEY,
    ):
        if key not in aux:
            continue
        mirrored_wrench = _mirror_wrench(aux[key], permutation)
        aux[key] = _mixed_frame(aux[key], mirrored_wrench, reverse)
    if "return_mode_wrenches" in aux:
        mirrored_mode_wrenches = (
            aux["return_mode_wrenches"]
            .index_select(
                1,
                permutation,
            )
            .clone()
        )
        mirrored_mode_wrenches[..., [0, 2]] *= -1.0
        aux["return_mode_wrenches"] = _mixed_frame(
            aux["return_mode_wrenches"],
            mirrored_mode_wrenches,
            reverse,
        )
    for key in (
        "return_mode_scores",
        "return_selected_mode",
        "return_source_advantage_score",
        "return_selected_candidate",
    ):
        if key in aux:
            aux[key] = _mixed_frame(
                aux[key],
                aux[key].index_select(1, permutation),
                reverse,
            )
    for key in (
        "stalk_mean",
        "stalk_precision",
        "temporal_memory_count",
        "correction_stalk_mean",
        "correction_stalk_precision",
    ):
        if key in aux:
            aux[key] = _mixed_frame(aux[key], aux[key].index_select(1, permutation), reverse)

    directed_permutation: Tensor | None = None
    for key in (
        "message_natural_trace",
        "message_precision_trace",
        "message_effective_precision_trace",
        # Backward-compatible canonical-parameter names.
        "message_xi_trace",
        "message_omega_trace",
        "message_mean_trace",
        "correction_message_natural_trace",
        "correction_message_precision_trace",
        "correction_message_effective_precision_trace",
        "correction_message_mean_trace",
        "communication_bundle_wire_trace",
    ):
        if key not in aux:
            continue
        if directed_permutation is None:
            directed_permutation = _directed_mirror_permutation(edge_index, permutation)
        # Only associate canonical wire slots with their physical sender and
        # receiver.  Do not parity-transform payload coordinates here: the
        # policy really transmitted these floats in its shared canonical route
        # frame, so changing their signs would make the trace visually useful
        # but no longer an exact wire audit.
        mirrored_message = aux[key].index_select(2, directed_permutation)
        aux[key] = _mixed_frame(aux[key], mirrored_message, reverse)
    if "message_intensity_trace" in aux:
        if directed_permutation is None:
            directed_permutation = _directed_mirror_permutation(edge_index, permutation)
        mirrored_intensity = aux["message_intensity_trace"].index_select(2, directed_permutation)
        aux["message_intensity_trace"] = _mixed_frame(
            aux["message_intensity_trace"], mirrored_intensity, reverse
        )
    if "correction_message_intensity_trace" in aux:
        if directed_permutation is None:
            directed_permutation = _directed_mirror_permutation(edge_index, permutation)
        mirrored_intensity = aux["correction_message_intensity_trace"].index_select(
            2, directed_permutation
        )
        aux["correction_message_intensity_trace"] = _mixed_frame(
            aux["correction_message_intensity_trace"], mirrored_intensity, reverse
        )
    if "gbp_message_trace" in aux:
        canonical_trace = aux["gbp_message_trace"]
        if not isinstance(canonical_trace, GBPMessageTrace):
            raise TypeError("aux['gbp_message_trace'] must be a GBPMessageTrace")
        # Rebuild the structured API after associating exact canonical-frame
        # wire floats with physical radius-graph endpoints.
        aux["gbp_message_trace"] = GBPMessageTrace(
            sender=aux["message_sender"],
            receiver=aux["message_receiver"],
            natural=aux["message_natural_trace"],
            precision=aux["message_effective_precision_trace"],
            mean=aux["message_mean_trace"],
            intensity=aux["message_intensity_trace"],
            transmits_precision=canonical_trace.transmits_precision,
        )
    if "correction_gbp_message_trace" in aux:
        canonical_trace = aux["correction_gbp_message_trace"]
        if not isinstance(canonical_trace, GBPMessageTrace):
            raise TypeError("aux['correction_gbp_message_trace'] must be a GBPMessageTrace")
        aux["correction_gbp_message_trace"] = GBPMessageTrace(
            sender=aux["correction_message_sender"],
            receiver=aux["correction_message_receiver"],
            natural=aux["correction_message_natural_trace"],
            precision=aux["correction_message_effective_precision_trace"],
            mean=aux["correction_message_mean_trace"],
            intensity=aux["correction_message_intensity_trace"],
            transmits_precision=canonical_trace.transmits_precision,
        )

    return PolicyOutput(forces=physical_forces, hidden=output.hidden, aux=aux)


class BasePolicy(nn.Module):
    arm_name = "base"

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.arm_cfg = arm_cfg
        self.obs_encoder = mlp(arm_cfg.obs_dim, arm_cfg.hidden_dim, arm_cfg.hidden_dim)
        self.gru = nn.GRUCell(arm_cfg.hidden_dim, arm_cfg.hidden_dim)

    def bind_environment(self, env_cfg: EnvConfig) -> None:
        """Bind a deployment environment without changing learned tensors.

        Agent-invariant policies can be evaluated at a new population size by
        rebuilding their hidden shape and graph from this configuration.  A
        specialized policy may override this hook to refresh nonpersistent,
        analytically derived graph buffers.
        """

        self.env_cfg = env_cfg

    def init_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(
            batch_size,
            self.env_cfg.n_agents,
            self.arm_cfg.hidden_dim,
            device=device,
        )

    def recurrent_encode(self, obs: Tensor, hidden: Tensor | None) -> Tensor:
        batch, n_agents, _ = obs.shape
        encoded = self.obs_encoder(obs)
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)
        flat = self.gru(
            encoded.reshape(batch * n_agents, -1),
            hidden.reshape(batch * n_agents, -1),
        )
        return flat.view(batch, n_agents, -1)

    def communication_floats_per_directed_edge_round(self) -> int:
        return 0

    def communication_rounds(self) -> int:
        return 0


class NoCommPolicy(BasePolicy):
    arm_name = "no_comm"

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig):
        super().__init__(env_cfg, arm_cfg)
        self.decoder = mlp(arm_cfg.hidden_dim + arm_cfg.obs_dim, arm_cfg.hidden_dim, 2)

    def forward(
        self,
        obs: Tensor,
        edge_index: Tensor,
        hidden: Tensor | None = None,
        *,
        active_leg: Tensor | None = None,
    ) -> PolicyOutput:
        del active_leg
        del edge_index
        next_hidden = self.recurrent_encode(obs, hidden)
        raw_force = self.decoder(torch.cat((obs, next_hidden), dim=-1))
        forces = self.env_cfg.max_force * torch.tanh(raw_force)
        return PolicyOutput(forces=forces, hidden=next_hidden, aux={})


class MPNNPolicy(BasePolicy):
    """Recurrent graph baseline with explicit float-message accounting."""

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig, *, message_dim: int, name: str):
        super().__init__(env_cfg, arm_cfg)
        self.message_dim = message_dim
        self.arm_name = name
        self.send = nn.Linear(arm_cfg.hidden_dim, message_dim)
        self.recv = mlp(message_dim, arm_cfg.hidden_dim, arm_cfg.hidden_dim, layers=1)
        self.update = mlp(2 * arm_cfg.hidden_dim, arm_cfg.hidden_dim, arm_cfg.hidden_dim, layers=1)
        self.decoder = mlp(arm_cfg.hidden_dim + arm_cfg.obs_dim, arm_cfg.hidden_dim, 2)

    def communication_floats_per_directed_edge_round(self) -> int:
        return self.message_dim

    def communication_rounds(self) -> int:
        return self.arm_cfg.comm_rounds

    def forward(
        self,
        obs: Tensor,
        edge_index: Tensor,
        hidden: Tensor | None = None,
        *,
        active_leg: Tensor | None = None,
    ) -> PolicyOutput:
        del active_leg
        h = self.recurrent_encode(obs, hidden)
        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        directed_src = torch.cat((src, dst), dim=0)
        directed_dst = torch.cat((dst, src), dim=0)
        for _ in range(self.arm_cfg.comm_rounds):
            message = self.send(h.index_select(1, directed_src))
            agg_msg = h.new_zeros(h.shape[0], h.shape[1], self.message_dim)
            agg_msg.index_add_(1, directed_dst, message)
            degree = h.new_zeros(h.shape[1])
            degree.index_add_(0, directed_dst, torch.ones_like(directed_dst, dtype=h.dtype))
            agg_msg = agg_msg / degree.clamp_min(1.0).view(1, -1, 1)
            agg = self.recv(agg_msg)
            h = h + self.update(torch.cat((h, agg), dim=-1))
        raw_force = self.decoder(torch.cat((obs, h), dim=-1))
        forces = self.env_cfg.max_force * torch.tanh(raw_force)
        return PolicyOutput(forces=forces, hidden=h, aux={})


class TemporalStateMatchedMPNNPolicy(MPNNPolicy):
    """MPNN control with the same carried-state budget as temporal Sheaf-GBP.

    The ordinary MPNN controls carry only their 96-D recurrent node state,
    while the single-slit Sheaf controller additionally carries six 32-D
    posterior stalks and six validity scalars.  This control retains the
    MPNN's native message/update/decoder path, but adds a real, consumed FIFO
    with exactly the same ``96 + K * (32 + 1)`` state layout.  A fixed recency
    weighting makes every occupied FIFO slot part of the current action.

    ``spatial_communication=False`` is an explicit masked-message ablation:
    it retains exactly the same trainable modules and four local update rounds
    as its communication-matched twin, but never reads ``edge_index`` and
    reports zero wire packets.
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        arm_cfg: ArmConfig,
        *,
        message_dim: int,
        name: str,
        spatial_communication: bool = True,
    ) -> None:
        super().__init__(env_cfg, arm_cfg, message_dim=message_dim, name=name)
        if arm_cfg.temporal_window <= 0:
            raise ValueError("temporal-state-matched MPNN requires a positive window")
        if arm_cfg.stalk_dim <= 0:
            raise ValueError("temporal-state-matched MPNN requires a positive history width")
        self.spatial_communication = bool(spatial_communication)
        self.temporal_read = nn.Linear(arm_cfg.stalk_dim, arm_cfg.hidden_dim)
        self.temporal_write = nn.Linear(arm_cfg.hidden_dim, arm_cfg.stalk_dim)

    @property
    def carried_state_floats_per_agent(self) -> int:
        return self.arm_cfg.hidden_dim + self.arm_cfg.temporal_window * (
            self.arm_cfg.stalk_dim + 1
        )

    def init_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(
            batch_size,
            self.env_cfg.n_agents,
            self.carried_state_floats_per_agent,
            device=device,
        )

    def split_temporal_hidden(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected = (
            hidden.shape[0],
            self.env_cfg.n_agents,
            self.carried_state_floats_per_agent,
        )
        if hidden.shape != expected:
            raise ValueError(
                "temporal-state-matched MPNN hidden must have shape "
                f"{expected}, got {tuple(hidden.shape)}"
            )
        recurrent = hidden[..., : self.arm_cfg.hidden_dim]
        history_start = self.arm_cfg.hidden_dim
        history_floats = self.arm_cfg.temporal_window * self.arm_cfg.stalk_dim
        history = hidden[
            ..., history_start : history_start + history_floats
        ].view(
            hidden.shape[0],
            self.env_cfg.n_agents,
            self.arm_cfg.temporal_window,
            self.arm_cfg.stalk_dim,
        )
        valid = hidden[..., history_start + history_floats :]
        return recurrent, history, valid

    def temporal_context(self, history: Tensor, valid: Tensor) -> Tensor:
        expected_history = (
            history.shape[0],
            self.env_cfg.n_agents,
            self.arm_cfg.temporal_window,
            self.arm_cfg.stalk_dim,
        )
        if history.shape != expected_history or valid.shape != expected_history[:-1]:
            raise ValueError("temporal history/mask has the wrong shape")
        # Slot zero is newest, matching SheafGBPPolicy.pack_hidden.  The
        # strictly positive fixed weights make old frames effective without
        # introducing an extra learned positional module.
        recency = 1.0 / (
            1.0
            + torch.arange(
                self.arm_cfg.temporal_window,
                dtype=history.dtype,
                device=history.device,
            )
        )
        weight = valid.to(history.dtype) * recency.view(1, 1, -1)
        denominator = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (history * weight.unsqueeze(-1)).sum(dim=-2) / denominator

    def communication_floats_per_directed_edge_round(self) -> int:
        return self.message_dim if self.spatial_communication else 0

    def communication_rounds(self) -> int:
        return self.arm_cfg.comm_rounds if self.spatial_communication else 0

    def forward(
        self,
        obs: Tensor,
        edge_index: Tensor,
        hidden: Tensor | None = None,
        *,
        active_leg: Tensor | None = None,
    ) -> PolicyOutput:
        del active_leg
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)
        recurrent, history, valid = self.split_temporal_hidden(hidden)
        h = self.recurrent_encode(obs, recurrent)
        context = self.temporal_context(history, valid)
        has_history = (valid > 0.5).any(dim=-1, keepdim=True).to(h.dtype)
        h = h + has_history * torch.tanh(self.temporal_read(context))

        if self.spatial_communication:
            src = edge_index[:, 0].long()
            dst = edge_index[:, 1].long()
            directed_src = torch.cat((src, dst), dim=0)
            directed_dst = torch.cat((dst, src), dim=0)
        for _ in range(self.arm_cfg.comm_rounds):
            if self.spatial_communication:
                message = self.send(h.index_select(1, directed_src))
                aggregate_message = h.new_zeros(
                    batch, self.env_cfg.n_agents, self.message_dim
                )
                aggregate_message.index_add_(1, directed_dst, message)
                degree = h.new_zeros(self.env_cfg.n_agents)
                degree.index_add_(
                    0,
                    directed_dst,
                    torch.ones_like(directed_dst, dtype=h.dtype),
                )
                aggregate_message = aggregate_message / degree.clamp_min(1.0).view(
                    1, -1, 1
                )
            else:
                # Explicitly execute the same local round stack with a zero
                # received packet.  No endpoint, sender state, or edge value
                # is inspected in this branch.
                aggregate_message = h.new_zeros(
                    batch, self.env_cfg.n_agents, self.message_dim
                )
            aggregate = self.recv(aggregate_message)
            h = h + self.update(torch.cat((h, aggregate), dim=-1))

        raw_force = self.decoder(torch.cat((obs, h), dim=-1))
        forces = self.env_cfg.max_force * torch.tanh(raw_force)
        summary = torch.tanh(self.temporal_write(h)).detach()
        next_history = torch.cat((summary.unsqueeze(2), history[:, :, :-1]), dim=2)
        next_valid = torch.cat(
            (torch.ones_like(valid[:, :, :1]), valid[:, :, :-1]), dim=2
        )
        next_hidden = torch.cat(
            (
                h,
                next_history.reshape(batch, self.env_cfg.n_agents, -1),
                next_valid,
            ),
            dim=-1,
        )
        return PolicyOutput(
            forces=forces,
            hidden=next_hidden,
            aux={
                "temporal_memory_count": next_valid.sum(dim=-1),
                "spatial_messages_masked": torch.tensor(
                    not self.spatial_communication, device=obs.device
                ),
            },
        )


class SheafGBPPolicy(BasePolicy):
    """Learned coordination sheaf with geometry-grounded stalk restrictions."""

    arm_name = "sheaf_gbp"

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig):
        super().__init__(env_cfg, arm_cfg)
        if arm_cfg.temporal_window < 0:
            raise ValueError("temporal_window must be non-negative")
        if arm_cfg.edge_dim < 8:
            raise ValueError("sheaf edge_dim must be at least 8 for the physical section")
        if arm_cfg.queryable_goal_free_temporal:
            if not env_cfg.strict_local_observation or arm_cfg.obs_dim != 18:
                raise ValueError(
                    "queryable goal-free temporal memory requires 18-d strict-local observations"
                )
            if arm_cfg.temporal_window <= 0:
                raise ValueError(
                    "queryable goal-free temporal memory requires a positive temporal_window"
                )
            # The physical encoder is shared by the current frame and every
            # cached frame.  Goal channels only enter the query and final
            # fusion, never the persistent FIFO.
            self.goal_free_physical_encoder = mlp(
                _STRICT_PHYSICAL_DIM,
                arm_cfg.hidden_dim,
                arm_cfg.hidden_dim,
                layers=1,
            )
            self.goal_query_encoder = mlp(
                2,
                arm_cfg.hidden_dim,
                arm_cfg.hidden_dim,
                layers=1,
            )
            self.temporal_query = nn.Linear(2 * arm_cfg.hidden_dim, arm_cfg.hidden_dim)
            self.temporal_key = nn.Linear(arm_cfg.hidden_dim, arm_cfg.hidden_dim, bias=False)
            # Local-only age/position signal for the physical FIFO.  Without
            # this, attention is permutation-invariant over the K cached frames
            # and cannot distinguish an approach from the same poses played in
            # reverse.  It never enters the edge section or wire payload.
            self.temporal_age_embedding = nn.Parameter(
                torch.empty(arm_cfg.temporal_window, arm_cfg.hidden_dim)
            )
            nn.init.normal_(self.temporal_age_embedding, mean=0.0, std=0.02)
            self.goal_free_fusion = mlp(
                3 * arm_cfg.hidden_dim + 1,
                arm_cfg.hidden_dim,
                arm_cfg.hidden_dim,
                layers=1,
            )
        else:
            self.goal_free_physical_encoder = None
            self.goal_query_encoder = None
            self.temporal_query = None
            self.temporal_key = None
            self.register_parameter("temporal_age_embedding", None)
            self.goal_free_fusion = None
        self.mean_head = nn.Linear(arm_cfg.hidden_dim, arm_cfg.stalk_dim)
        self.precision_head = nn.Linear(arm_cfg.hidden_dim, arm_cfg.stalk_dim)
        self.u_head = nn.Linear(arm_cfg.hidden_dim, arm_cfg.edge_dim * arm_cfg.restriction_rank)
        self.v_head = nn.Linear(arm_cfg.hidden_dim, arm_cfg.stalk_dim * arm_cfg.restriction_rank)
        if arm_cfg.edge_conditioned_restrictions:
            # Directed restriction residuals can depend on the pair of handles
            # connected by the communication edge.  This keeps the message
            # budget unchanged but lets the sheaf express that different
            # contact pairs should agree on different projected coordinates.
            self.edge_u_head = mlp(
                arm_cfg.hidden_dim + 7,
                arm_cfg.hidden_dim,
                arm_cfg.edge_dim * arm_cfg.restriction_rank,
                layers=1,
            )
        self.gbp = DiagSheafGBP(
            arm_cfg.gbp_steps,
            damping=arm_cfg.damping,
            mean_only_messages=arm_cfg.mean_only_messages,
        )
        decoder_in_dim = arm_cfg.obs_dim + arm_cfg.edge_dim
        if not arm_cfg.decoder_section_only:
            decoder_in_dim += arm_cfg.stalk_dim
        if arm_cfg.decoder_extra_context:
            # analytic force (2) plus temporal memory count (1).  These are
            # locally available after GBP and do not add wireless messages.
            decoder_in_dim += 3
        self.decoder = mlp(decoder_in_dim, arm_cfg.hidden_dim, 2)
        if arm_cfg.return_maneuver_modes not in (0, 1, 6):
            raise ValueError("return_maneuver_modes must be 0, 1, or 6")
        if arm_cfg.return_source_candidate and not arm_cfg.return_adapter:
            raise ValueError("the source candidate requires return_adapter=True")
        if arm_cfg.return_source_candidate and arm_cfg.return_maneuver_modes != 1:
            raise ValueError("the source candidate requires return_maneuver_modes=1")
        if arm_cfg.return_source_candidate and (
            not env_cfg.strict_local_observation or arm_cfg.obs_dim != 18
        ):
            raise ValueError("the source candidate requires 18-d strict-local observations")
        if arm_cfg.return_source_candidate and arm_cfg.return_candidate_architecture not in {
            RETURN_CANDIDATE_ARCHITECTURE_V1,
            RETURN_CANDIDATE_ARCHITECTURE_V2,
            RETURN_CANDIDATE_ARCHITECTURE_V3,
        }:
            raise ValueError(
                "unsupported return_candidate_architecture="
                f"{arm_cfg.return_candidate_architecture!r}"
            )
        if (
            arm_cfg.return_source_candidate
            and arm_cfg.return_candidate_architecture == RETURN_CANDIDATE_ARCHITECTURE_V3
            and arm_cfg.edge_dim != _RETURN_CANDIDATE_V3_SECTION_DIM
        ):
            raise ValueError("V3 return candidate requires edge_dim=12")
        if (
            arm_cfg.return_source_candidate
            and arm_cfg.return_candidate_architecture == RETURN_CANDIDATE_ARCHITECTURE_V3
            and not arm_cfg.exact_return_grasp_allocator
        ):
            raise ValueError("V3 return candidate requires the exact return grasp allocator")
        if arm_cfg.return_maneuver_modes == 6 and not arm_cfg.return_adapter:
            raise ValueError("six return maneuver modes require return_adapter=True")
        if arm_cfg.return_maneuver_modes == 6 and not arm_cfg.exact_return_grasp_allocator:
            raise ValueError("six return maneuver modes require the exact return grasp allocator")
        if arm_cfg.correction_sheaf and arm_cfg.return_adapter:
            raise ValueError(
                "the continuous correction sheaf replaces, rather than stacks with, "
                "the legacy return adapter"
            )
        if arm_cfg.correction_sheaf and (
            not env_cfg.strict_local_observation or arm_cfg.obs_dim not in {18, 22, 27}
        ):
            raise ValueError(
                "the correction sheaf requires 18-d strict-local, exact-clearance22, "
                "or contact-feedback27 observations"
            )
        if arm_cfg.return_adapter:
            # Predict a shared normalized residual wrench, not independent
            # per-handle forces.  The analytic grasp map below turns torque
            # into the required antisymmetric force pattern by construction.
            if arm_cfg.return_source_candidate:
                self.return_adapter = ReturnSourceCandidateAdapter(
                    arm_cfg.return_candidate_architecture,
                    legacy_in_dim=decoder_in_dim,
                    legacy_hidden_dim=arm_cfg.hidden_dim,
                )
            elif arm_cfg.return_maneuver_modes == 6:
                self.return_adapter = ReturnManeuverAdapter(
                    decoder_in_dim,
                    arm_cfg.hidden_dim,
                    arm_cfg.return_maneuver_modes,
                )
            else:
                # Keep the legacy Sequential module and exact state-dict keys
                # for both the default value zero and explicit single mode.
                self.return_adapter = mlp(decoder_in_dim, arm_cfg.hidden_dim, 3)
                final = self.return_adapter[-1]
                if not isinstance(final, nn.Linear):
                    raise TypeError("return adapter must end in a linear layer")
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        else:
            self.return_adapter = None

        if arm_cfg.correction_sheaf:
            # This branch reads only local sensing and the pre-GBP local
            # unary statistics.  It is therefore available at the start of
            # the same communication round as the frozen source packet; no
            # received source section or source decoder feature is an input.
            correction_input_dim = arm_cfg.obs_dim + 2 * arm_cfg.stalk_dim
            self.correction_local_encoder = mlp(
                correction_input_dim,
                arm_cfg.hidden_dim,
                arm_cfg.hidden_dim,
                layers=1,
            )
            if arm_cfg.persistent_recovery_state:
                # The event packet is [failed own force2, rejected own handle
                # motion2].  On later safe ticks these four coordinates are
                # exact zero, while the current physical/clearance22 prefix
                # keeps updating the private recurrent recovery state.
                recovery_input_dim = _SINGLE_SLIT_EXACT_CLEARANCE_DIM + 4
                self.persistent_recovery_gru = nn.GRUCell(
                    recovery_input_dim,
                    arm_cfg.persistent_recovery_dim,
                )
                if arm_cfg.persistent_recovery_injection == "shared_hidden":
                    self.persistent_recovery_fusion = nn.Linear(
                        arm_cfg.persistent_recovery_dim,
                        arm_cfg.hidden_dim,
                        bias=False,
                    )
                    nn.init.zeros_(self.persistent_recovery_fusion.weight)
                    self.persistent_recovery_mean_residual = None
                else:
                    # The recovery state changes only the correction Gaussian
                    # unary mean.  In particular it cannot change confidence
                    # or the learned restriction maps through correction_h.
                    self.persistent_recovery_fusion = None
                    self.persistent_recovery_mean_residual = nn.Linear(
                        arm_cfg.persistent_recovery_dim,
                        arm_cfg.correction_stalk_dim,
                        bias=False,
                    )
                    nn.init.zeros_(self.persistent_recovery_mean_residual.weight)
                # Exact zero is required for bridge-time source parity.  The
                # phase head intentionally retains ordinary initialization:
                # phase supervision then gives the recovery GRU a live first-
                # step gradient while action loss first opens the injection.
                self.persistent_recovery_phase_head = nn.Linear(
                    arm_cfg.persistent_recovery_dim,
                    arm_cfg.persistent_recovery_phase_count,
                )
            if arm_cfg.local_se2_planner:
                planner_input_dim = (
                    arm_cfg.obs_dim
                    + arm_cfg.hidden_dim
                    + arm_cfg.persistent_recovery_dim
                    + (
                        arm_cfg.local_se2_shape_signature_dim
                        if arm_cfg.local_se2_shape_signature
                        else 0
                    )
                )
                self.local_se2_planner_encoder = mlp(
                    planner_input_dim,
                    arm_cfg.local_se2_planner_hidden_dim,
                    arm_cfg.local_se2_planner_hidden_dim,
                    layers=1,
                )
                self.local_se2_plan_mean_head = nn.Linear(
                    arm_cfg.local_se2_planner_hidden_dim,
                    arm_cfg.local_se2_plan_dim,
                )
                self.local_se2_plan_log_precision_head = nn.Linear(
                    arm_cfg.local_se2_planner_hidden_dim,
                    arm_cfg.local_se2_plan_dim,
                )
                if arm_cfg.local_se2_markov_shape_planner:
                    # canonical12 + causal recovery8 + geometry32.  In
                    # particular, the legacy GRU hidden and every path/progress
                    # variable are absent by construction.
                    markov_shape_input_dim = (
                        LOCAL_SE2_RECOVERY_CANONICAL_DIM
                        + arm_cfg.persistent_recovery_dim
                        + arm_cfg.local_se2_shape_signature_dim
                    )
                    self.local_se2_markov_shape_encoder = mlp(
                        markov_shape_input_dim,
                        arm_cfg.local_se2_markov_shape_hidden_dim,
                        arm_cfg.local_se2_markov_shape_hidden_dim,
                        layers=arm_cfg.local_se2_markov_shape_layers,
                    )
                    if arm_cfg.local_se2_markov_goal_context:
                        self.local_se2_markov_goal_projection = nn.Linear(
                            4, arm_cfg.local_se2_markov_shape_hidden_dim, bias=False,
                        )
                        nn.init.zeros_(self.local_se2_markov_goal_projection.weight)
                    self.local_se2_markov_shape_mean_head = nn.Linear(
                        arm_cfg.local_se2_markov_shape_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    self.local_se2_markov_shape_log_precision_head = nn.Linear(
                        arm_cfg.local_se2_markov_shape_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    nn.init.zeros_(self.local_se2_markov_shape_mean_head.weight)
                    nn.init.zeros_(self.local_se2_markov_shape_mean_head.bias)
                    nn.init.zeros_(
                        self.local_se2_markov_shape_log_precision_head.weight
                    )
                    nn.init.zeros_(
                        self.local_se2_markov_shape_log_precision_head.bias
                    )
                if arm_cfg.local_se2_recovery_adapter:
                    recovery_adapter_input_dim = (
                        arm_cfg.obs_dim
                        + arm_cfg.persistent_recovery_dim
                        + arm_cfg.persistent_recovery_phase_count
                        + arm_cfg.local_se2_plan_dim
                        + (
                            LOCAL_SE2_RECOVERY_CANONICAL_DIM
                            if arm_cfg.local_se2_recovery_canonical_features
                            else 0
                        )
                    )
                    self.local_se2_recovery_adapter_encoder = mlp(
                        recovery_adapter_input_dim,
                        arm_cfg.local_se2_recovery_adapter_hidden_dim,
                        arm_cfg.local_se2_recovery_adapter_hidden_dim,
                        layers=2,
                    )
                    self.local_se2_recovery_adapter_head = nn.Linear(
                        arm_cfg.local_se2_recovery_adapter_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    # Exact zero gives bridge-time action and plan parity.  A
                    # supervised phase loss trains the phase path immediately;
                    # plan loss opens this final residual on the first update.
                    nn.init.zeros_(self.local_se2_recovery_adapter_head.weight)
                    nn.init.zeros_(self.local_se2_recovery_adapter_head.bias)
                if arm_cfg.local_se2_contact_rebel_adapter:
                    # canonical12 + exact-clearance4 + mechanical-event5 +
                    # recovery state + phase4 + incumbent local plan8.
                    contact_rebel_input_dim = (
                        LOCAL_SE2_RECOVERY_CANONICAL_DIM
                        + 9
                        + arm_cfg.persistent_recovery_dim
                        + arm_cfg.persistent_recovery_phase_count
                        + arm_cfg.local_se2_plan_dim
                    )
                    self.local_se2_contact_rebel_encoder = mlp(
                        contact_rebel_input_dim,
                        arm_cfg.local_se2_contact_rebel_hidden_dim,
                        arm_cfg.local_se2_contact_rebel_hidden_dim,
                        layers=2,
                    )
                    self.local_se2_contact_rebel_head = nn.Linear(
                        arm_cfg.local_se2_contact_rebel_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    nn.init.zeros_(self.local_se2_contact_rebel_head.weight)
                    nn.init.zeros_(self.local_se2_contact_rebel_head.bias)
                if arm_cfg.local_se2_exit_refinement_adapter:
                    exit_refinement_input_dim = (
                        LOCAL_SE2_RECOVERY_CANONICAL_DIM
                        + 9
                        + arm_cfg.persistent_recovery_dim
                        + arm_cfg.persistent_recovery_phase_count
                        + arm_cfg.local_se2_plan_dim
                    )
                    self.local_se2_exit_refinement_encoder = mlp(
                        exit_refinement_input_dim,
                        arm_cfg.local_se2_exit_refinement_hidden_dim,
                        arm_cfg.local_se2_exit_refinement_hidden_dim,
                        layers=2,
                    )
                    self.local_se2_exit_refinement_head = nn.Linear(
                        arm_cfg.local_se2_exit_refinement_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    nn.init.zeros_(self.local_se2_exit_refinement_head.weight)
                    nn.init.zeros_(self.local_se2_exit_refinement_head.bias)
                if arm_cfg.local_se2_approach_refinement_adapter:
                    approach_refinement_input_dim = (
                        LOCAL_SE2_RECOVERY_CANONICAL_DIM
                        + 9
                        + arm_cfg.persistent_recovery_dim
                        + arm_cfg.persistent_recovery_phase_count
                        + arm_cfg.local_se2_plan_dim
                    )
                    self.local_se2_approach_refinement_encoder = mlp(
                        approach_refinement_input_dim,
                        arm_cfg.local_se2_approach_refinement_hidden_dim,
                        arm_cfg.local_se2_approach_refinement_hidden_dim,
                        layers=2,
                    )
                    self.local_se2_approach_refinement_head = nn.Linear(
                        arm_cfg.local_se2_approach_refinement_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    nn.init.zeros_(self.local_se2_approach_refinement_head.weight)
                    nn.init.zeros_(self.local_se2_approach_refinement_head.bias)
                if arm_cfg.local_se2_deep_stuck_refinement_adapter:
                    deep_stuck_input_dim = (
                        LOCAL_SE2_RECOVERY_CANONICAL_DIM
                        + 9
                        + arm_cfg.persistent_recovery_dim
                        + arm_cfg.persistent_recovery_phase_count
                        + arm_cfg.local_se2_plan_dim
                    )
                    self.local_se2_deep_stuck_refinement_encoder = mlp(
                        deep_stuck_input_dim,
                        arm_cfg.local_se2_deep_stuck_refinement_hidden_dim,
                        arm_cfg.local_se2_deep_stuck_refinement_hidden_dim,
                        layers=2,
                    )
                    self.local_se2_deep_stuck_refinement_head = nn.Linear(
                        arm_cfg.local_se2_deep_stuck_refinement_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    nn.init.zeros_(self.local_se2_deep_stuck_refinement_head.weight)
                    nn.init.zeros_(self.local_se2_deep_stuck_refinement_head.bias)
                if arm_cfg.local_se2_retry_refinement_thresholds:
                    # The extra scalar is the normalized causal contact
                    # streak.  It lets each retry stage express a timed
                    # retreat/re-entry sequence instead of a static response.
                    retry_input_dim = (
                        LOCAL_SE2_RECOVERY_CANONICAL_DIM
                        + 9
                        + arm_cfg.persistent_recovery_dim
                        + arm_cfg.persistent_recovery_phase_count
                        + arm_cfg.local_se2_plan_dim
                        + 1
                    )
                    self.local_se2_retry_refinement_encoders = nn.ModuleList()
                    self.local_se2_retry_refinement_heads = nn.ModuleList()
                    for _threshold in arm_cfg.local_se2_retry_refinement_thresholds:
                        encoder = mlp(
                            retry_input_dim,
                            arm_cfg.local_se2_retry_refinement_hidden_dim,
                            arm_cfg.local_se2_retry_refinement_hidden_dim,
                            layers=2,
                        )
                        head = nn.Linear(
                            arm_cfg.local_se2_retry_refinement_hidden_dim,
                            arm_cfg.local_se2_plan_dim,
                        )
                        nn.init.zeros_(head.weight)
                        nn.init.zeros_(head.bias)
                        self.local_se2_retry_refinement_encoders.append(encoder)
                        self.local_se2_retry_refinement_heads.append(head)
                if arm_cfg.local_se2_geometry_refinement_adapter:
                    geometry_refinement_input_dim = (
                        planner_input_dim + arm_cfg.local_se2_plan_dim
                        + (
                            LOCAL_SE2_RECOVERY_CANONICAL_DIM
                            if arm_cfg.local_se2_geometry_refinement_canonical_features
                            else 0
                        )
                    )
                    self.local_se2_geometry_refinement_encoder = mlp(
                        geometry_refinement_input_dim,
                        arm_cfg.local_se2_geometry_refinement_hidden_dim,
                        arm_cfg.local_se2_geometry_refinement_hidden_dim,
                        layers=2,
                    )
                    self.local_se2_geometry_refinement_head = nn.Linear(
                        arm_cfg.local_se2_geometry_refinement_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    nn.init.zeros_(self.local_se2_geometry_refinement_head.weight)
                    nn.init.zeros_(self.local_se2_geometry_refinement_head.bias)
                if arm_cfg.local_se2_geometry_mode_adapter:
                    geometry_mode_input_dim = (
                        planner_input_dim
                        + arm_cfg.local_se2_plan_dim
                        + LOCAL_SE2_RECOVERY_CANONICAL_DIM
                    )
                    self.local_se2_geometry_mode_encoder = mlp(
                        geometry_mode_input_dim,
                        arm_cfg.local_se2_geometry_mode_hidden_dim,
                        arm_cfg.local_se2_geometry_mode_hidden_dim,
                        layers=3,
                    )
                    self.local_se2_geometry_mode_logits_head = nn.Linear(
                        arm_cfg.local_se2_geometry_mode_hidden_dim,
                        arm_cfg.local_se2_geometry_mode_count,
                    )
                    self.local_se2_geometry_mode_residual_head = nn.Linear(
                        arm_cfg.local_se2_geometry_mode_hidden_dim,
                        arm_cfg.local_se2_geometry_mode_count
                        * arm_cfg.local_se2_plan_dim,
                    )
                    nn.init.zeros_(self.local_se2_geometry_mode_logits_head.weight)
                    nn.init.zeros_(self.local_se2_geometry_mode_logits_head.bias)
                    nn.init.zeros_(self.local_se2_geometry_mode_residual_head.weight)
                    nn.init.zeros_(self.local_se2_geometry_mode_residual_head.bias)
                if arm_cfg.local_se2_path_memory:
                    shape_count = arm_cfg.local_se2_path_memory_shape_count
                    path_points = arm_cfg.local_se2_path_memory_max_steps + 1
                    self.register_buffer(
                        "local_se2_path_memory_shape_signatures",
                        torch.zeros(shape_count, 32),
                    )
                    self.register_buffer(
                        "local_se2_path_memory_reference_poses",
                        torch.zeros(shape_count, path_points, 3),
                    )
                    self.register_buffer(
                        "local_se2_path_memory_horizons",
                        torch.zeros(shape_count, dtype=torch.long),
                    )
                    self.register_buffer(
                        "local_se2_path_memory_letter_codes",
                        torch.zeros(shape_count, dtype=torch.long),
                    )
                    if arm_cfg.local_se2_path_memory_ridge_decoder:
                        # Closed-form train-only morphology decoder.  The
                        # intercept is the final design coordinate.  Weights
                        # cover the full padded path buffer so lookup remains
                        # a single batched tensor operation.
                        self.register_buffer(
                            "local_se2_path_memory_ridge_weights",
                            torch.zeros(33, path_points, 3),
                        )
                        self.register_buffer(
                            "local_se2_path_memory_train_pose_min",
                            torch.zeros(3),
                        )
                        self.register_buffer(
                            "local_se2_path_memory_train_pose_max",
                            torch.zeros(3),
                        )
                    if arm_cfg.local_se2_path_memory_population_resampling:
                        self.register_buffer(
                            "local_se2_path_memory_population_resampling_poses",
                            torch.zeros(shape_count, path_points, 3),
                        )
                self.correction_plan_fusion = nn.Linear(
                    2 * arm_cfg.local_se2_plan_dim,
                    arm_cfg.hidden_dim,
                    bias=False,
                )
                nn.init.zeros_(self.correction_plan_fusion.weight)
                if arm_cfg.local_se2_direct_wrench_unary:
                    self.local_se2_direct_wrench_unary_head = mlp(
                        arm_cfg.hidden_dim + 2 * arm_cfg.local_se2_plan_dim,
                        arm_cfg.local_se2_direct_wrench_hidden_dim,
                        _CORRECTION_WRENCH_DIM,
                        layers=2,
                    )
                    final_direct_wrench = self.local_se2_direct_wrench_unary_head[-1]
                    nn.init.zeros_(final_direct_wrench.weight)
                    nn.init.zeros_(final_direct_wrench.bias)
                if arm_cfg.local_se2_distributed_wrench_control:
                    # This decoder is shared across nodes.  Its inputs are
                    # strictly local except for the 3-D section received via
                    # the real GBP packets.  In particular it never sees the
                    # payload label, global morphology, or a grasp matrix.
                    distributed_force_input_dim = (
                        arm_cfg.obs_dim
                        + arm_cfg.correction_edge_dim
                        + arm_cfg.local_se2_plan_dim
                    )
                    self.local_se2_distributed_force_decoder = mlp(
                        distributed_force_input_dim,
                        arm_cfg.local_se2_distributed_force_hidden_dim,
                        2,
                        layers=3,
                    )
                if arm_cfg.local_se2_distributed_recovery_adapter:
                    distributed_recovery_input_dim = (
                        planner_input_dim + arm_cfg.local_se2_plan_dim
                    )
                    self.local_se2_distributed_recovery_encoder = mlp(
                        distributed_recovery_input_dim,
                        arm_cfg.local_se2_distributed_recovery_hidden_dim,
                        arm_cfg.local_se2_distributed_recovery_hidden_dim,
                        layers=arm_cfg.local_se2_distributed_recovery_layers,
                    )
                    self.local_se2_distributed_recovery_head = nn.Linear(
                        arm_cfg.local_se2_distributed_recovery_hidden_dim,
                        arm_cfg.local_se2_plan_dim,
                    )
                    nn.init.zeros_(
                        self.local_se2_distributed_recovery_head.weight
                    )
                    nn.init.zeros_(
                        self.local_se2_distributed_recovery_head.bias
                    )
                    if arm_cfg.local_se2_distributed_recovery_mode_count > 1:
                        self.local_se2_distributed_recovery_mode_logits_head = (
                            nn.Linear(
                                arm_cfg.local_se2_distributed_recovery_hidden_dim,
                                arm_cfg.local_se2_distributed_recovery_mode_count,
                            )
                        )
                        self.local_se2_distributed_recovery_mode_residual_head = (
                            nn.Linear(
                                arm_cfg.local_se2_distributed_recovery_hidden_dim,
                                arm_cfg.local_se2_distributed_recovery_mode_count
                                * arm_cfg.local_se2_plan_dim,
                            )
                        )
                        nn.init.zeros_(
                            self.local_se2_distributed_recovery_mode_logits_head.weight
                        )
                        nn.init.zeros_(
                            self.local_se2_distributed_recovery_mode_logits_head.bias
                        )
                        nn.init.zeros_(
                            self.local_se2_distributed_recovery_mode_residual_head.weight
                        )
                        nn.init.zeros_(
                            self.local_se2_distributed_recovery_mode_residual_head.bias
                        )
                    distributed_recovery_force_input_dim = (
                        arm_cfg.obs_dim
                        + arm_cfg.correction_edge_dim
                        + arm_cfg.local_se2_plan_dim
                        + arm_cfg.persistent_recovery_dim
                    )
                    self.local_se2_distributed_recovery_force_decoder = mlp(
                        distributed_recovery_force_input_dim,
                        arm_cfg.local_se2_distributed_recovery_hidden_dim,
                        2,
                        layers=3,
                    )
                    final_recovery_force = (
                        self.local_se2_distributed_recovery_force_decoder[-1]
                    )
                    nn.init.zeros_(final_recovery_force.weight)
                    nn.init.zeros_(final_recovery_force.bias)
            if arm_cfg.collision_role_factors:
                # Three local factor families share one 16-D node stalk and
                # one 3-D GBP edge section.  Zero initialization is the
                # bridge contract: roles are visible for supervision at
                # sigmoid(0)=0.5, while their unary/edge interventions are
                # algebraically absent until learned.
                self.correction_role_logits_head = nn.Linear(
                    arm_cfg.hidden_dim,
                    arm_cfg.collision_role_count,
                )
                self.correction_role_mean_head = nn.Linear(
                    arm_cfg.hidden_dim,
                    arm_cfg.collision_role_count * arm_cfg.correction_stalk_dim,
                )
                self.correction_role_log_precision_head = nn.Linear(
                    arm_cfg.hidden_dim,
                    arm_cfg.collision_role_count * arm_cfg.correction_stalk_dim,
                )
                for module in (
                    self.correction_role_logits_head,
                    self.correction_role_mean_head,
                    self.correction_role_log_precision_head,
                ):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
                if arm_cfg.collision_role_separate_encoder:
                    self.correction_role_encoder = mlp(
                        correction_input_dim,
                        arm_cfg.hidden_dim,
                        arm_cfg.hidden_dim,
                        layers=1,
                    )
                else:
                    self.correction_role_encoder = None
            self.correction_mean_head = nn.Linear(
                arm_cfg.hidden_dim,
                arm_cfg.correction_stalk_dim,
            )
            # Stalk coordinates 0:3 are the explicit normalized-wrench
            # proposal and start at exact zero.  The private coordinates 3:16
            # retain ordinary initialization, making the low-rank restriction
            # update live on the very first backward pass.
            with torch.no_grad():
                self.correction_mean_head.weight[:_CORRECTION_WRENCH_DIM].zero_()
                self.correction_mean_head.bias[:_CORRECTION_WRENCH_DIM].zero_()
            self.correction_precision_head = nn.Linear(
                arm_cfg.hidden_dim,
                arm_cfg.correction_stalk_dim,
            )
            self.correction_u_head = nn.Linear(
                arm_cfg.hidden_dim + 2,
                arm_cfg.correction_edge_dim * arm_cfg.correction_restriction_rank,
            )
            self.correction_v_head = nn.Linear(
                arm_cfg.hidden_dim,
                (arm_cfg.correction_stalk_dim - _CORRECTION_WRENCH_DIM)
                * arm_cfg.correction_restriction_rank,
            )
            # A one-sided LoRA zero initialization retains the analytic grasp
            # restriction exactly while preserving a nonzero companion factor.
            # The private stalk posterior makes U live on the first backward
            # pass; V becomes live immediately after U leaves zero.
            nn.init.zeros_(self.correction_u_head.weight)
            nn.init.zeros_(self.correction_u_head.bias)
            self.correction_gbp = DiagSheafGBP(
                arm_cfg.gbp_steps,
                damping=arm_cfg.damping,
                mean_only_messages=False,
            )
            if arm_cfg.local_se2_analytic_total_wrench:
                canonical_edges = radius_edges(env_cfg, device="cpu")
                tree_mask = _minimum_radius_spanning_tree_mask(
                    canonical_edges,
                    env_cfg.n_agents,
                )
                self.register_buffer(
                    "local_se2_analytic_reference_edges",
                    canonical_edges,
                    persistent=False,
                )
                self.register_buffer(
                    "local_se2_analytic_tree_edge_mask",
                    tree_mask,
                    persistent=False,
                )
                # Handle coordinates are rigidly fixed in the payload frame.
                # Their four all-reduced sufficient statistics therefore need
                # to cross the radio graph only once per attachment/layout,
                # rather than once per control tick.  Both caches are local
                # deployment state and deliberately absent from checkpoints.
                self.register_buffer(
                    "local_se2_distributed_grasp_body_local_cache",
                    torch.empty((0, 0, 4), dtype=torch.float32),
                    persistent=False,
                )
                self.register_buffer(
                    "local_se2_distributed_grasp_body_sufficient_cache",
                    torch.empty((0, 0, 4), dtype=torch.float32),
                    persistent=False,
                )
                # The selected factors form a tree, so undamped synchronous
                # GBP reaches the exact Gaussian solution in graph-diameter
                # sweeps.  The single-slit radius graph has minimum radius 2
                # and the configured four sweeps cover its tree diameter 4.
                self.local_se2_analytic_gbp = DiagSheafGBP(
                    arm_cfg.gbp_steps,
                    damping=1.0,
                    eps=arm_cfg.local_se2_analytic_solver_eps,
                    max_precision=arm_cfg.local_se2_analytic_max_precision,
                    mean_only_messages=False,
                )
            else:
                self.register_buffer(
                    "local_se2_analytic_reference_edges",
                    torch.empty((0, 2), dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer(
                    "local_se2_analytic_tree_edge_mask",
                    torch.empty((0,), dtype=torch.bool),
                    persistent=False,
                )
                self.register_buffer(
                    "local_se2_distributed_grasp_body_local_cache",
                    torch.empty((0, 0, 4), dtype=torch.float32),
                    persistent=False,
                )
                self.register_buffer(
                    "local_se2_distributed_grasp_body_sufficient_cache",
                    torch.empty((0, 0, 4), dtype=torch.float32),
                    persistent=False,
                )
                self.local_se2_analytic_gbp = None
            if arm_cfg.correction_public_unary_residual:
                # The correction encoder is 96-D in the single-slit model;
                # use its configured width here so the module remains exact
                # under reconstruction.  A zero final affine map makes this
                # opt-in branch algebraically absent at bridge time.
                self.correction_public_unary_residual_head = nn.Linear(
                    arm_cfg.hidden_dim,
                    _CORRECTION_WRENCH_DIM,
                    bias=True,
                )
                nn.init.zeros_(self.correction_public_unary_residual_head.weight)
                nn.init.zeros_(self.correction_public_unary_residual_head.bias)
            if arm_cfg.correction_source_null_recovery:
                # A 96 -> 3 local proposal with a fixed physical envelope in
                # forward().  Zero initialization makes the opt-in bridge
                # algebraically identical everywhere, while the envelope
                # keeps far-field source ticks identical after training too.
                self.correction_source_null_recovery_head = nn.Linear(
                    arm_cfg.hidden_dim,
                    _CORRECTION_WRENCH_DIM,
                    bias=True,
                )
                nn.init.zeros_(self.correction_source_null_recovery_head.weight)
                nn.init.zeros_(self.correction_source_null_recovery_head.bias)
            if arm_cfg.correction_phase_decoder:
                if arm_cfg.obs_dim not in {22, 27}:
                    raise ValueError(
                        "slit phase decoder requires exact-clearance22 or contact-feedback27"
                    )
                self.correction_phase_wrench_head = nn.Linear(
                    arm_cfg.correction_phase_basis_dim
                    * arm_cfg.correction_phase_content_dim,
                    _CORRECTION_WRENCH_DIM,
                    bias=False,
                )
                nn.init.zeros_(self.correction_phase_wrench_head.weight)
            if arm_cfg.correction_clearance_bank_decoder:
                if arm_cfg.obs_dim not in {22, 27}:
                    raise ValueError(
                        "clearance-bank decoder requires exact-clearance22 or contact-feedback27"
                    )
                self.correction_phase_clearance_wrench_head = nn.Linear(
                    arm_cfg.correction_clearance_bank_basis_dim
                    * arm_cfg.correction_clearance_bank_content_dim,
                    _CORRECTION_WRENCH_DIM,bias=False,
                )
                nn.init.zeros_(self.correction_phase_clearance_wrench_head.weight)
            if arm_cfg.correction_approach_bank_decoder:
                if arm_cfg.obs_dim not in {22, 27}:
                    raise ValueError(
                        "approach bank requires exact-clearance22 or contact-feedback27"
                    )
                self.correction_phase_approach_wrench_head=nn.Linear(
                    arm_cfg.correction_approach_bank_basis_dim*arm_cfg.correction_approach_bank_content_dim,
                    _CORRECTION_WRENCH_DIM,bias=False,
                )
                nn.init.zeros_(self.correction_phase_approach_wrench_head.weight)

    def bind_environment(self, env_cfg: EnvConfig) -> None:
        """Refresh deployment-only graph buffers for a new handle population.

        The analytic total-wrench factors use a deterministic spanning tree of
        the *actual* radius graph.  These buffers are derived geometry, are not
        serialized checkpoint parameters, and therefore may be rebound while
        the learned checkpoint tensors remain bitwise frozen.  Rebinding the
        canonical environment reconstructs the historical buffers exactly.
        """

        super().bind_environment(env_cfg)
        if not self.arm_cfg.local_se2_analytic_total_wrench:
            return
        device = next(self.parameters()).device
        edges = radius_edges(env_cfg, device=device)
        tree_mask = _minimum_radius_spanning_tree_mask(edges, env_cfg.n_agents)
        self.local_se2_analytic_reference_edges = edges
        self.local_se2_analytic_tree_edge_mask = tree_mask
        dtype = next(self.parameters()).dtype
        self.local_se2_distributed_grasp_body_local_cache = torch.empty(
            (0, env_cfg.n_agents, 4), device=device, dtype=dtype
        )
        self.local_se2_distributed_grasp_body_sufficient_cache = torch.empty(
            (0, env_cfg.n_agents, 4), device=device, dtype=dtype
        )

    def legacy_hidden_floats_per_agent(self) -> int:
        """Width of the carried state before the opt-in recovery suffix."""

        if self.arm_cfg.queryable_goal_free_temporal:
            return self.arm_cfg.temporal_window * (_STRICT_PHYSICAL_DIM + 2)
        memory_dim = self.arm_cfg.temporal_window * (self.arm_cfg.stalk_dim + 1)
        return self.arm_cfg.hidden_dim + memory_dim

    def init_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        width = self.legacy_hidden_floats_per_agent()
        if self.arm_cfg.persistent_recovery_state:
            width += self.arm_cfg.persistent_recovery_dim
        if self.arm_cfg.local_se2_approach_refinement_adapter:
            width += 1
        return torch.zeros(
            batch_size,
            self.env_cfg.n_agents,
            width,
            device=device,
        )

    def split_persistent_recovery_hidden(
        self,
        hidden: Tensor | None,
        obs: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Separate the legacy prefix from the opt-in local recovery state."""

        if not self.arm_cfg.persistent_recovery_state:
            return hidden, None, None
        if hidden is None:
            hidden = self.init_hidden(obs.shape[0], obs.device)
        legacy_width = self.legacy_hidden_floats_per_agent()
        approach_width = int(self.arm_cfg.local_se2_approach_refinement_adapter)
        expected = (
            obs.shape[0],
            self.env_cfg.n_agents,
            legacy_width + self.arm_cfg.persistent_recovery_dim + approach_width,
        )
        if hidden.shape != expected:
            raise ValueError(
                "persistent-recovery hidden must have shape "
                f"{expected}, got {tuple(hidden.shape)}"
            )
        recovery_right = legacy_width + self.arm_cfg.persistent_recovery_dim
        approach_streak = (
            hidden[..., recovery_right : recovery_right + 1]
            if approach_width
            else None
        )
        return (
            hidden[..., :legacy_width],
            hidden[..., legacy_width:recovery_right],
            approach_streak,
        )

    def update_approach_contact_streak(
        self,
        obs: Tensor,
        previous: Tensor | None,
    ) -> Tensor | None:
        """Causally count immediately consecutive rejected transitions."""

        if not self.arm_cfg.local_se2_approach_refinement_adapter:
            return None
        if previous is None or previous.shape != (*obs.shape[:2], 1):
            raise ValueError("approach contact streak has the wrong hidden shape")
        hit = obs[
            ...,
            _SINGLE_SLIT_CONTACT_HIT_INDEX : _SINGLE_SLIT_CONTACT_HIT_INDEX + 1,
        ]
        maximum_streak = (
            self.arm_cfg.local_se2_deep_stuck_refinement_stuck_steps
            if self.arm_cfg.local_se2_deep_stuck_refinement_adapter
            else self.arm_cfg.local_se2_approach_refinement_stuck_steps
        )
        if self.arm_cfg.local_se2_retry_refinement_thresholds:
            maximum_streak = self.arm_cfg.local_se2_retry_refinement_thresholds[-1]
        maximum = float(maximum_streak)
        return torch.where(
            hit != 0,
            (previous + 1.0).clamp_max(maximum),
            torch.zeros_like(previous),
        )

    def update_persistent_recovery_state(
        self,
        obs: Tensor,
        previous: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the causal event-triggered recovery recurrence.

        A never-hit exact-zero row is selected around the GRU algebraically.
        Once a hit writes a nonzero state, every following tick consumes the
        current physical observation; a later hit recurrently updates the
        existing state rather than replacing it.
        """

        if obs.shape[-1] != _SINGLE_SLIT_CONTACT_OBSERVATION_DIM:
            raise ValueError("persistent recovery requires contact-feedback27")
        hit = obs[..., _SINGLE_SLIT_CONTACT_HIT_INDEX : _SINGLE_SLIT_CONTACT_HIT_INDEX + 1]
        was_active = (previous != 0).any(dim=-1, keepdim=True)
        active = (hit != 0) | was_active
        event = hit * obs[..., _SINGLE_SLIT_CONTACT_HIT_INDEX + 1 :]
        recovery_input = torch.cat(
            (obs[..., :_SINGLE_SLIT_EXACT_CLEARANCE_DIM], event),
            dim=-1,
        )
        candidate = self.persistent_recovery_gru(
            recovery_input.reshape(-1, recovery_input.shape[-1]),
            previous.reshape(-1, previous.shape[-1]),
        ).view_as(previous)
        state = torch.where(active, candidate, previous)
        logits_candidate = self.persistent_recovery_phase_head(state)
        if self.arm_cfg.local_se2_analytic_phase_latch:
            # Every strict-local handle row reconstructs the same rigid-body
            # centre from its known body coordinate, world contact position,
            # and observed orientation.  A real contact activates retreat;
            # crossing the declared safe centre-x boundary irreversibly
            # latches re-entry.  Channel zero stores only this private latch;
            # no edge packet or globally read simulator state is introduced.
            sine = obs[..., 0]
            cosine = obs[..., 1]
            handle_body_x = obs[..., 2]
            handle_body_y = obs[..., 3]
            rotated_x = cosine * handle_body_x - sine * handle_body_y
            centre_x = 2.6 * obs[..., 4] - rotated_x
            crossed = (
                centre_x
                <= self.arm_cfg.local_se2_reentry_center_x_threshold_m
            ).unsqueeze(-1)
            previously_reentering = previous[..., :1] > 0.0
            reentering = active & (previously_reentering | crossed)
            latch_value = torch.where(
                reentering,
                torch.ones_like(state[..., :1]),
                -torch.ones_like(state[..., :1]),
            )
            state = torch.cat(
                (
                    torch.where(active, latch_value, state[..., :1]),
                    state[..., 1:],
                ),
                dim=-1,
            )
            phase_index = torch.where(
                reentering.squeeze(-1),
                torch.full_like(centre_x, 3, dtype=torch.long),
                torch.full_like(centre_x, 2, dtype=torch.long),
            )
            logits_candidate = (
                self.arm_cfg.local_se2_phase_latch_logit
                * F.one_hot(
                    phase_index,
                    num_classes=self.arm_cfg.persistent_recovery_phase_count,
                ).to(dtype=state.dtype)
            )
        logits = torch.where(active, logits_candidate, torch.zeros_like(logits_candidate))
        if self.arm_cfg.local_se2_distributed_recovery_local_support:
            # Recovery is a local, reversible role rather than a permanent
            # episode mode.  A real rejected transition activates every row
            # for one tick because the rigid load transmits that mechanical
            # event.  Thereafter only handles still close to a wall retain
            # their rebel memory; other handles return exactly to the shared
            # no-contact planner.  No peer row, payload outline, shape ID, or
            # simulator collision query is used here.
            wall_visibility = obs[..., 16:18] > 0.5
            wall_distance_m = (
                obs[..., 10:12].abs() * float(self.env_cfg.wall_sensor_range)
            )
            locally_near_wall = (
                wall_visibility
                & (
                    wall_distance_m
                    <= self.arm_cfg.local_se2_distributed_recovery_wall_range_m
                )
            ).any(dim=-1, keepdim=True)
            local_support = (hit != 0) | locally_near_wall
            active = active & local_support
            state = torch.where(active, state, torch.zeros_like(state))
            logits = torch.where(active, logits, torch.zeros_like(logits))
        return state, active, logits

    def split_hidden(self, hidden: Tensor | None, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if hidden is None:
            hidden = self.init_hidden(obs.shape[0], obs.device)
        hidden_dim = self.arm_cfg.hidden_dim
        rec_hidden = hidden[..., :hidden_dim]
        window = self.arm_cfg.temporal_window
        if window == 0:
            history = obs.new_zeros(obs.shape[0], obs.shape[1], 0, self.arm_cfg.stalk_dim)
            mask = obs.new_zeros(obs.shape[0], obs.shape[1], 0)
            return rec_hidden, history, mask
        offset = hidden_dim
        history_values = window * self.arm_cfg.stalk_dim
        history = hidden[..., offset : offset + history_values].view(
            obs.shape[0],
            obs.shape[1],
            window,
            self.arm_cfg.stalk_dim,
        )
        mask = hidden[..., offset + history_values : offset + history_values + window]
        return rec_hidden, history, mask

    def pack_hidden(
        self, rec_hidden: Tensor, history: Tensor, mask: Tensor, mean: Tensor
    ) -> Tensor:
        window = self.arm_cfg.temporal_window
        if window == 0:
            return rec_hidden
        next_history = torch.cat((mean.detach().unsqueeze(2), history[:, :, :-1]), dim=2)
        next_mask = torch.cat((torch.ones_like(mask[:, :, :1]), mask[:, :, :-1]), dim=2)
        return torch.cat(
            (rec_hidden, next_history.reshape(*rec_hidden.shape[:2], -1), next_mask),
            dim=-1,
        )

    def split_goal_free_hidden(
        self,
        hidden: Tensor | None,
        obs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Unpack the physical FIFO, validity mask, and frame labels."""

        if hidden is None:
            hidden = self.init_hidden(obs.shape[0], obs.device)
        window = self.arm_cfg.temporal_window
        expected = window * (_STRICT_PHYSICAL_DIM + 2)
        if hidden.shape != (obs.shape[0], obs.shape[1], expected):
            raise ValueError(
                "queryable goal-free hidden must have shape "
                f"[{obs.shape[0]}, {obs.shape[1]}, {expected}]"
            )
        value_width = window * _STRICT_PHYSICAL_DIM
        history = hidden[..., :value_width].view(
            obs.shape[0], obs.shape[1], window, _STRICT_PHYSICAL_DIM
        )
        valid = hidden[..., value_width : value_width + window]
        frame_leg = hidden[..., value_width + window : expected]
        return history, valid, frame_leg

    def query_goal_free_history(
        self,
        history: Tensor,
        valid: Tensor,
        frame_leg: Tensor,
        active_leg: Tensor,
    ) -> Tensor:
        """Express each cached physical frame under the current active leg."""

        permutation = mirror_handle_permutation(self.env_cfg.n_agents, history.device)
        queried_frames = []
        current_leg = active_leg.to(history.dtype).view(history.shape[0], 1)
        for frame_index in range(self.arm_cfg.temporal_window):
            frame = history[:, :, frame_index]
            mirrored = mirror_strict_physical_observations(frame, permutation)
            mismatch = (valid[:, :, frame_index] > 0.5) & (
                (frame_leg[:, :, frame_index] > 0.5) != (current_leg > 0.5)
            )
            queried_frames.append(torch.where(mismatch.unsqueeze(-1), mirrored, frame))
        return torch.stack(queried_frames, dim=2)

    def goal_free_temporal_encode(
        self,
        obs: Tensor,
        hidden: Tensor | None,
        active_leg: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Query physical temporal memory with the current physical state/goal."""

        if any(
            module is None
            for module in (
                self.goal_free_physical_encoder,
                self.goal_query_encoder,
                self.temporal_query,
                self.temporal_key,
                self.goal_free_fusion,
            )
        ):
            raise RuntimeError("goal-free temporal modules were not initialized")
        physical, goal = split_strict_physical_goal(obs)
        history, valid, frame_leg = self.split_goal_free_hidden(hidden, obs)
        if active_leg is None or not self.arm_cfg.active_leg_canonicalization:
            # In the legacy mission-fixed frame, a round-trip goal switch does
            # not change coordinates, so every cached frame has label zero.
            active_leg = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        if active_leg.shape != (obs.shape[0],):
            raise ValueError("active_leg must have shape [batch]")

        queried_history = self.query_goal_free_history(
            history,
            valid,
            frame_leg,
            active_leg,
        )
        current_h = self.goal_free_physical_encoder(physical)
        goal_h = self.goal_query_encoder(goal)
        memory_h = self.goal_free_physical_encoder(queried_history)
        query = self.temporal_query(torch.cat((current_h, goal_h), dim=-1))
        if self.temporal_age_embedding is None:
            raise RuntimeError("goal-free temporal age embedding was not initialized")
        age_embedding = self.temporal_age_embedding.view(
            1,
            1,
            self.arm_cfg.temporal_window,
            self.arm_cfg.hidden_dim,
        )
        keys = self.temporal_key(memory_h + age_embedding)
        logits = (keys * query.unsqueeze(2)).sum(dim=-1) / float(self.arm_cfg.hidden_dim) ** 0.5
        valid_bool = valid > 0.5
        weights = torch.softmax(logits.masked_fill(~valid_bool, -1e4), dim=-1)
        weights = weights * valid_bool.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        memory_summary = (weights.unsqueeze(-1) * memory_h).sum(dim=2)
        memory_count = valid.sum(dim=-1)
        count_scale = float(self.arm_cfg.temporal_window)
        local_h = self.goal_free_fusion(
            torch.cat(
                (
                    current_h,
                    goal_h,
                    memory_summary,
                    (memory_count / count_scale).unsqueeze(-1),
                ),
                dim=-1,
            )
        )

        next_history = torch.cat(
            (physical.detach().unsqueeze(2), history[:, :, :-1]),
            dim=2,
        )
        next_valid = torch.cat(
            (torch.ones_like(valid[:, :, :1]), valid[:, :, :-1]),
            dim=2,
        )
        leg_value = active_leg.to(obs.dtype).view(obs.shape[0], 1, 1)
        next_frame_leg = torch.cat(
            (
                leg_value.expand(-1, obs.shape[1], 1),
                frame_leg[:, :, :-1],
            ),
            dim=2,
        )
        next_hidden = torch.cat(
            (
                next_history.reshape(*obs.shape[:2], -1),
                next_valid,
                next_frame_leg,
            ),
            dim=-1,
        )
        return local_h, next_hidden, memory_count

    def temporal_unary(self, history: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.arm_cfg.temporal_window == 0:
            shape = (*history.shape[:2], self.arm_cfg.stalk_dim)
            return history.new_zeros(shape), history.new_zeros(shape)
        weights = mask.unsqueeze(-1)
        precision = self.arm_cfg.temporal_precision * weights.sum(dim=2)
        natural = self.arm_cfg.temporal_precision * (weights * history).sum(dim=2)
        return precision, natural

    def correction_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return the complete, auditable C4.4 trainable parameter partition."""

        if not self.arm_cfg.correction_sheaf:
            return ()
        parameters = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name.startswith("correction_")
        )
        if not parameters:
            raise RuntimeError("the enabled correction sheaf has no parameters")
        return parameters

    def set_correction_only_trainable(self) -> tuple[str, ...]:
        """Freeze the source path and enable gradients only for C4.4 modules."""

        correction = dict(self.correction_named_parameters())
        if not correction:
            raise RuntimeError("correction_sheaf must be enabled before freezing the source")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in correction)
        return tuple(correction)

    def set_correction_phase_only_trainable(self) -> tuple[str, ...]:
        """Freeze every parent tensor and train only the 216-weight adapter."""

        if not self.arm_cfg.correction_sheaf or not self.arm_cfg.correction_phase_decoder:
            raise RuntimeError("correction phase decoder must be enabled")
        expected = "correction_phase_wrench_head.weight"
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name == expected)
        trainable = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )
        if trainable != (expected,) or self.correction_phase_wrench_head.weight.numel() != 216:
            raise RuntimeError("phase-only trainable partition is not the fixed 216 weights")
        return trainable

    def set_correction_clearance_bank_only_trainable(self) -> tuple[str, ...]:
        """Freeze every parent tensor and train only the 864-weight bank."""
        if not self.arm_cfg.correction_clearance_bank_decoder:
            raise RuntimeError("correction clearance-bank decoder must be enabled")
        expected="correction_phase_clearance_wrench_head.weight"
        for name,parameter in self.named_parameters():
            parameter.requires_grad_(name==expected)
        trainable=tuple(name for name,p in self.named_parameters() if p.requires_grad)
        if trainable!=(expected,) or self.correction_phase_clearance_wrench_head.weight.numel()!=864:
            raise RuntimeError("clearance-bank trainable partition is not the fixed 864 weights")
        return trainable

    def set_correction_approach_bank_only_trainable(self) -> tuple[str, ...]:
        """Freeze the parent and train only the 216-weight approach bank."""
        if not self.arm_cfg.correction_approach_bank_decoder:
            raise RuntimeError("approach clearance bank must be enabled")
        expected="correction_phase_approach_wrench_head.weight"
        for name,parameter in self.named_parameters():
            parameter.requires_grad_(name==expected)
        trainable=tuple(name for name,p in self.named_parameters() if p.requires_grad)
        if trainable!=(expected,) or self.correction_phase_approach_wrench_head.weight.numel()!=216:
            raise RuntimeError("approach-bank trainable partition is not fixed 216")
        return trainable

    def set_correction_public_unary_only_trainable(self) -> tuple[str, ...]:
        """Freeze the parent and train only the 291-parameter public unary head."""

        if not self.arm_cfg.correction_public_unary_residual:
            raise RuntimeError("public correction-unary residual must be enabled")
        expected = (
            "correction_public_unary_residual_head.weight",
            "correction_public_unary_residual_head.bias",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in expected)
        trainable = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )
        parameter_count = sum(dict(self.named_parameters())[name].numel() for name in expected)
        if trainable != expected or parameter_count != 291:
            raise RuntimeError(
                "public correction-unary trainable partition is not the fixed 291 parameters"
            )
        return trainable

    def set_correction_source_null_recovery_only_trainable(self) -> tuple[str, ...]:
        """Freeze the parent and train only the 291-parameter recovery head."""

        if not self.arm_cfg.correction_source_null_recovery:
            raise RuntimeError("source-null correction recovery must be enabled")
        expected = (
            "correction_source_null_recovery_head.weight",
            "correction_source_null_recovery_head.bias",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in expected)
        trainable = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )
        parameter_count = sum(dict(self.named_parameters())[name].numel() for name in expected)
        if trainable != expected or parameter_count != 291:
            raise RuntimeError(
                "source-null recovery trainable partition is not the fixed 291 parameters"
            )
        return trainable

    def set_correction_public_phase_clearance_trainable(self) -> tuple[str, ...]:
        """Train the three small, public-output correction adapters jointly.

        This leaves the source encoder, learned restrictions, GBP solver, and
        source decoder frozen.  The 291-D unary residual, 216-D phase decoder,
        and 864-D clearance bank are the only tensors that can move.
        """

        if not (
            self.arm_cfg.correction_public_unary_residual
            and self.arm_cfg.correction_phase_decoder
            and self.arm_cfg.correction_clearance_bank_decoder
        ):
            raise RuntimeError(
                "joint public/phase/clearance training requires all three adapters"
            )
        expected = (
            "correction_public_unary_residual_head.weight",
            "correction_public_unary_residual_head.bias",
            "correction_phase_wrench_head.weight",
            "correction_phase_clearance_wrench_head.weight",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in expected)
        trainable = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )
        parameter_count = sum(dict(self.named_parameters())[name].numel() for name in expected)
        if trainable != expected or parameter_count != 1_371:
            raise RuntimeError(
                "joint public/phase/clearance partition is not the fixed 1371 parameters"
            )
        return trainable

    def correction_analytic_restrictions(self, reference: Tensor) -> Tensor:
        """Return ``[I_3, 0]`` from the 16-D node stalk to physical wrench.

        The first three stalk coordinates have fixed, public semantics:
        normalized ``[Fx, Fy, tau]``.  The remaining thirteen are private
        coordinates available only through the geometry-conditioned learned
        low-rank residual.
        """

        cfg = self.arm_cfg
        if not cfg.correction_sheaf:
            raise RuntimeError("correction_sheaf is disabled")
        if reference.ndim < 2:
            raise ValueError("correction reference must include batch and agent axes")
        restriction = reference.new_zeros(
            reference.shape[0],
            self.env_cfg.n_agents,
            cfg.correction_edge_dim,
            cfg.correction_stalk_dim,
        )
        eye = torch.eye(
            _CORRECTION_WRENCH_DIM,
            dtype=reference.dtype,
            device=reference.device,
        )
        restriction[..., :_CORRECTION_WRENCH_DIM] = eye.view(
            1,
            1,
            _CORRECTION_WRENCH_DIM,
            _CORRECTION_WRENCH_DIM,
        )
        return restriction

    def correction_restrictions(self, correction_h: Tensor) -> Tensor:
        """Geometry-conditioned rank-R residual over private stalk coordinates."""

        cfg = self.arm_cfg
        if not cfg.correction_sheaf:
            raise RuntimeError("correction_sheaf is disabled")
        base = self.correction_analytic_restrictions(correction_h)
        rank = cfg.correction_restriction_rank
        contact_geometry = contact_points(self.env_cfg, correction_h.device).to(
            dtype=correction_h.dtype
        )
        contact_geometry = contact_geometry.view(1, self.env_cfg.n_agents, 2).expand(
            correction_h.shape[0], -1, -1
        )
        u = self.correction_u_head(torch.cat((correction_h, contact_geometry), dim=-1)).view(
            correction_h.shape[0],
            self.env_cfg.n_agents,
            cfg.correction_edge_dim,
            rank,
        )
        private_v = self.correction_v_head(correction_h).view(
            correction_h.shape[0],
            self.env_cfg.n_agents,
            cfg.correction_stalk_dim - _CORRECTION_WRENCH_DIM,
            rank,
        )
        # Padding is structural, not a learned mask: the low-rank update can
        # only reveal private coordinates 3:16 and cannot rotate or erase the
        # public I3 proposal map.
        v = F.pad(private_v, (0, 0, _CORRECTION_WRENCH_DIM, 0))
        residual = torch.einsum("bner,bnvr->bnev", u, v)
        return base + cfg.correction_restriction_residual_scale * residual / float(rank)

    def correction_directed_restrictions(
        self,
        node_r: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Gather node restrictions in GBP's interleaved directed-slot order."""

        directed_src = edge_index.reshape(-1).long()
        return node_r.index_select(1, directed_src)

    def communication_floats_per_directed_edge_round(self) -> int:
        # Default: edge natural plus diagonal precision; no full covariance.
        # Mean-only mode uses fixed precision and only transmits the natural
        # vector, so a 16-d section is comm-matched to a 16-float MPNN message.
        source_floats = (
            self.arm_cfg.edge_dim if self.arm_cfg.mean_only_messages else 2 * self.arm_cfg.edge_dim
        )
        if not self.arm_cfg.correction_sheaf:
            return source_floats
        # The correction solver uses ordinary diagonal Gaussian messages:
        # three natural coordinates plus three diagonal precisions.  These are
        # appended to the source packet in the same sweep, not sent in a new
        # communication round.
        if self.arm_cfg.local_se2_distributed_wrench_control:
            # The legacy source solve is dead and bypassed in this mode.  The
            # only physical wire packet is the correction Sheaf's diagonal
            # Gaussian natural vector plus precision.
            return 2 * self.arm_cfg.correction_edge_dim
        return source_floats + 2 * self.arm_cfg.correction_edge_dim

    def communication_rounds(self) -> int:
        if self.arm_cfg.local_se2_distributed_wrench_control and self.local_se2_analytic_gbp is not None:
            return self.local_se2_analytic_gbp.num_steps
        return self.arm_cfg.gbp_steps

    def analytic_restrictions(self, batch: int, device: torch.device | str) -> Tensor:
        cfg = self.arm_cfg
        contacts = contact_points(self.env_cfg, device)
        r = torch.zeros(batch, self.env_cfg.n_agents, cfg.edge_dim, cfg.stalk_dim, device=device)
        eye = torch.eye(cfg.edge_dim, cfg.stalk_dim, device=device)
        r[:] = eye.view(1, 1, cfg.edge_dim, cfg.stalk_dim)
        # Geometry-grounded wrench row: local force slots 3/4 also imply torque
        # through the handle's body-frame lever arm.
        r[:, :, 5, 3] = -contacts[:, 1].view(1, -1)
        r[:, :, 5, 4] = contacts[:, 0].view(1, -1)
        r[:, :, 5, 5] = 0.35
        return r

    def restrictions(self, h: Tensor) -> Tensor:
        batch = h.shape[0]
        base = self.analytic_restrictions(batch, h.device)
        rank = self.arm_cfg.restriction_rank
        u = self.u_head(h).view(batch, self.env_cfg.n_agents, self.arm_cfg.edge_dim, rank)
        v = self.v_head(h).view(batch, self.env_cfg.n_agents, self.arm_cfg.stalk_dim, rank)
        residual = torch.einsum("bner,bnvr->bnev", u, v)
        return base + self.arm_cfg.restriction_residual_scale * residual / float(rank)

    def directed_restrictions(self, node_r: Tensor, edge_index: Tensor) -> Tensor:
        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        left = node_r.index_select(1, src)
        right = node_r.index_select(1, dst)
        return torch.stack((left, right), dim=2).reshape(
            node_r.shape[0],
            2 * edge_index.shape[0],
            self.arm_cfg.edge_dim,
            self.arm_cfg.stalk_dim,
        )

    def edge_conditioned_directed_restrictions(
        self,
        node_r: Tensor,
        local_h: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # GBP stores directed incidences interleaved per undirected edge:
        # [u0->v0, v0->u0, u1->v1, v1->u1, ...].  Keep restriction slots in
        # exactly that order so each cavity belief is projected by its actual
        # sender's geometry.
        directed_src = edge_index.reshape(-1).long()
        directed_dst = edge_index.flip(-1).reshape(-1).long()
        base = node_r.index_select(1, directed_src)

        contacts = contact_points(self.env_cfg, local_h.device)
        src_contact = contacts.index_select(0, directed_src)
        dst_contact = contacts.index_select(0, directed_dst)
        rel = dst_contact - src_contact
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        geom = torch.cat((src_contact, dst_contact, rel, dist), dim=-1)
        geom = geom.view(1, geom.shape[0], geom.shape[1]).expand(local_h.shape[0], -1, -1)

        h_src = local_h.index_select(1, directed_src)
        rank = self.arm_cfg.restriction_rank
        u = self.edge_u_head(torch.cat((h_src, geom), dim=-1)).view(
            local_h.shape[0],
            directed_src.shape[0],
            self.arm_cfg.edge_dim,
            rank,
        )
        v = self.v_head(h_src).view(
            local_h.shape[0],
            directed_src.shape[0],
            self.arm_cfg.stalk_dim,
            rank,
        )
        residual = torch.einsum("bder,bdvr->bdev", u, v)
        return base + self.arm_cfg.restriction_residual_scale * residual / float(rank)

    def clamp_force_norm(self, forces: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(forces, dim=-1, keepdim=True).clamp_min(1e-6)
        return forces * (self.env_cfg.max_force / norms).clamp_max(1.0)

    def analytic_force_from_section(
        self,
        section: Tensor,
        obs: Tensor,
        *,
        clamp_output: bool = True,
        section_limit: float | None = 1.2,
    ) -> Tensor:
        """Map the consensus wrench slice through the frozen base allocator.

        Section coordinates 3:6 are trained to represent desired net wrench:
        Fx, Fy, tau.  This legacy approximation remains unchanged so loading a
        successful source checkpoint preserves its exact outbound controller.
        The return-only adapter uses ``exact_return_force_from_wrench`` below.
        """

        cfg = self.env_cfg
        safe_section = (
            section.clamp(-section_limit, section_limit) if section_limit is not None else section
        )
        wrench_scale = cfg.n_agents * cfg.max_force
        fx = safe_section[..., 3] * wrench_scale
        fy = safe_section[..., 4] * wrench_scale
        tau = safe_section[..., 5] * (0.75 * wrench_scale)

        contacts = contact_points(cfg, obs.device)
        if cfg.strict_local_observation:
            sin_theta = obs[:, :, 0]
            cos_theta = obs[:, :, 1]
        else:
            sin_theta = obs[:, :, 2]
            cos_theta = obs[:, :, 3]
        cx = contacts[:, 0].view(1, -1)
        cy = contacts[:, 1].view(1, -1)
        rx = cos_theta * cx - sin_theta * cy
        ry = sin_theta * cx + cos_theta * cy
        sum_r2 = (rx.square() + ry.square()).sum(dim=1, keepdim=True)

        fx_per = fx / float(cfg.n_agents + 0.08)
        fy_per = fy / float(cfg.n_agents + 0.08)
        tau_coeff = tau / (sum_r2 + 0.08)
        forces = torch.stack((fx_per - tau_coeff * ry, fy_per + tau_coeff * rx), dim=-1)
        return self.clamp_force_norm(forces) if clamp_output else forces

    def exact_return_force_from_wrench(
        self,
        normalized_wrench: Tensor,
        obs: Tensor,
    ) -> Tensor:
        """Allocate each node's shared residual wrench with the exact grasp map.

        ``normalized_wrench[b, i]`` is agent ``i``'s local estimate of the
        shared ``[Fx, Fy, tau]`` correction. Every agent knows the fixed handle
        geometry and observes payload orientation locally, so it can apply its
        own row of the damped grasp pseudoinverse without another message or a
        centralized aggregation. If node estimates agree, the resulting
        forces match the training-only CBF teacher's allocator to numerical
        precision.

        This method deliberately does not clamp. The caller applies the
        policy's existing final per-handle force limit after adding the frozen
        source action.
        """

        batch, n_agents, wrench_dim = normalized_wrench.shape
        if n_agents != self.env_cfg.n_agents or wrench_dim != 3:
            raise ValueError(
                f"normalized return wrench must have shape [batch, {self.env_cfg.n_agents}, 3]"
            )
        if obs.shape[:2] != (batch, n_agents):
            raise ValueError("observation and return-wrench batches must match")

        cfg = self.env_cfg
        wrench_scale = normalized_wrench.new_tensor(
            [
                cfg.n_agents * cfg.max_force,
                cfg.n_agents * cfg.max_force,
                0.75 * cfg.n_agents * cfg.max_force,
            ]
        )
        wrench = normalized_wrench * wrench_scale

        # Each agent independently combines its locally observed payload
        # orientation with the globally known, fixed handle layout. No peer
        # observation or wireless payload is consumed here.
        if cfg.strict_local_observation:
            sin_theta = obs[:, :, 0]
            cos_theta = obs[:, :, 1]
        else:
            sin_theta = obs[:, :, 2]
            cos_theta = obs[:, :, 3]
        contacts = contact_points(cfg, obs.device)
        cx = contacts[:, 0].view(1, 1, -1)
        cy = contacts[:, 1].view(1, 1, -1)
        rx = cos_theta[:, :, None] * cx - sin_theta[:, :, None] * cy
        ry = sin_theta[:, :, None] * cx + cos_theta[:, :, None] * cy

        grasp = obs.new_zeros((batch, n_agents, 3, 2 * n_agents))
        grasp[:, :, 0, 0::2] = 1.0
        grasp[:, :, 1, 1::2] = 1.0
        grasp[:, :, 2, 0::2] = -ry
        grasp[:, :, 2, 1::2] = rx
        gram = grasp @ grasp.transpose(-1, -2)
        eye = torch.eye(3, dtype=obs.dtype, device=obs.device).view(1, 1, 3, 3)

        # Agent i solves its local 3-D right-hand side, then applies only row i
        # of the resulting grasp solution.
        dual = solve_nonsingular(
            gram + _RETURN_GRASP_DAMPING * eye,
            wrench.unsqueeze(-1),
        ).squeeze(-1)
        own_cx = contacts[:, 0].view(1, -1)
        own_cy = contacts[:, 1].view(1, -1)
        own_rx = cos_theta * own_cx - sin_theta * own_cy
        own_ry = sin_theta * own_cx + cos_theta * own_cy
        return torch.stack(
            (
                dual[:, :, 0] - own_ry * dual[:, :, 2],
                dual[:, :, 1] + own_rx * dual[:, :, 2],
            ),
            dim=-1,
        )

    def distributed_force_from_local_grasp_statistics(
        self,
        normalized_wrench: Tensor,
        obs: Tensor,
        edge_index: Tensor,
        *,
        trace_messages: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None, bool]:
        """Realize a shared wrench after a one-time four-float setup reduction.

        Handle ``i`` knows only its body coordinate from its own observation.
        In world coordinates its planar grasp block is

        ``G_i = [[1,0],[0,1],[-r_y,r_x]]``.

        Hence ``sum_i G_i G_i^T`` depends only on four sums.  The handle
        coordinates are constant in the rigid payload frame, so sum-product
        cavity messages compute ``sum_i [1,b_x,b_y,b_x^2+b_y^2]`` exactly once
        after attachment.  Every later tick rotates the cached two first
        moments by its locally observed payload orientation.  Each node then
        solves the same damped 3-D dual and applies only its own two force
        coordinates.  No node reads another observation, the payload label, a
        compound-body outline, or ``contact_points``.
        """

        batch, n_agents, wrench_dim = normalized_wrench.shape
        if (
            obs.shape[:2] != (batch, n_agents)
            or n_agents != self.env_cfg.n_agents
            or wrench_dim != _CORRECTION_WRENCH_DIM
        ):
            raise ValueError("distributed grasp allocator received malformed tensors")
        if not torch.equal(edge_index, self.local_se2_analytic_reference_edges):
            raise ValueError("distributed grasp allocator received a different radius graph")
        tree_edges = edge_index[self.local_se2_analytic_tree_edge_mask]
        if tree_edges.shape != (n_agents - 1, 2):
            raise RuntimeError("distributed grasp allocator requires a spanning tree")

        sine = obs[..., 0]
        cosine = obs[..., 1]
        body_x = obs[..., 2]
        body_y = obs[..., 3]
        world_x = cosine * body_x - sine * body_y
        world_y = sine * body_x + cosine * body_y
        local_body = torch.stack(
            (
                torch.ones_like(world_x),
                body_x,
                body_y,
                body_x.square() + body_y.square(),
            ),
            dim=-1,
        )

        left = tree_edges[:, 0].long()
        right = tree_edges[:, 1].long()
        source = torch.cat((left, right), dim=0)
        destination = torch.cat((right, left), dim=0)
        directed = int(source.shape[0])
        cache_matches = (
            not self.training
            and self.local_se2_distributed_grasp_body_local_cache.shape
            == (1, n_agents, 4)
            and torch.equal(
                local_body,
                self.local_se2_distributed_grasp_body_local_cache.expand(
                    batch, -1, -1
                ),
            )
        )
        setup_transmitted = not cache_matches
        messages = local_body.new_zeros(batch, directed, 4)
        trace: list[Tensor] = []
        if cache_matches:
            sufficient_body = (
                self.local_se2_distributed_grasp_body_sufficient_cache.expand(
                    batch, -1, -1
                )
            )
        else:
            # On a tree, n-1 synchronous cavity rounds are a conservative
            # exact bound.  This is a setup/reattachment protocol, not a
            # control-rate protocol.
            for _round in range(max(1, n_agents - 1)):
                incoming = local_body.new_zeros(batch, n_agents, 4)
                incoming.index_add_(1, destination, messages)
                reverse = torch.cat(
                    (messages[:, n_agents - 1 :], messages[:, : n_agents - 1]),
                    dim=1,
                )
                messages = (
                    local_body.index_select(1, source)
                    + incoming.index_select(1, source)
                    - reverse
                )
                if trace_messages:
                    trace.append(messages)
            incoming = local_body.new_zeros(batch, n_agents, 4)
            incoming.index_add_(1, destination, messages)
            sufficient_body = local_body + incoming
            if not self.training and bool(
                torch.equal(local_body, local_body[:1].expand_as(local_body))
            ):
                self.local_se2_distributed_grasp_body_local_cache = (
                    local_body[:1].detach().clone()
                )
                self.local_se2_distributed_grasp_body_sufficient_cache = (
                    sufficient_body[:1].detach().clone()
                )

        count = sufficient_body[..., 0]
        sum_body_x = sufficient_body[..., 1]
        sum_body_y = sufficient_body[..., 2]
        sufficient = torch.stack(
            (
                count,
                cosine * sum_body_x - sine * sum_body_y,
                sine * sum_body_x + cosine * sum_body_y,
                sufficient_body[..., 3],
            ),
            dim=-1,
        )

        count = sufficient[..., 0].clamp_min(1.0)
        sum_x = sufficient[..., 1]
        sum_y = sufficient[..., 2]
        sum_r2 = sufficient[..., 3]
        gram = obs.new_zeros(batch, n_agents, 3, 3)
        gram[..., 0, 0] = count
        gram[..., 1, 1] = count
        gram[..., 0, 2] = -sum_y
        gram[..., 2, 0] = -sum_y
        gram[..., 1, 2] = sum_x
        gram[..., 2, 1] = sum_x
        gram[..., 2, 2] = sum_r2
        physical_scale = torch.stack(
            (
                count * float(self.env_cfg.max_force),
                count * float(self.env_cfg.max_force),
                0.75 * count * float(self.env_cfg.max_force),
            ),
            dim=-1,
        )
        physical_wrench = normalized_wrench * physical_scale
        eye = torch.eye(3, dtype=obs.dtype, device=obs.device).view(1, 1, 3, 3)
        dual = solve_nonsingular(
            gram + _RETURN_GRASP_DAMPING * eye,
            physical_wrench.unsqueeze(-1),
        ).squeeze(-1)
        forces = torch.stack(
            (
                dual[..., 0] - world_y * dual[..., 2],
                dual[..., 1] + world_x * dual[..., 2],
            ),
            dim=-1,
        )
        forces = self.clamp_force_norm(forces)
        if trace_messages:
            message_trace = (
                torch.stack(trace, dim=0)
                if trace
                else local_body.new_empty((0, batch, directed, 4))
            )
        else:
            message_trace = None
        return forces, sufficient, message_trace, setup_transmitted

    def _local_se2_plan_from_features(
        self,
        learned_obs: Tensor,
        local_h: Tensor,
        recovery_state: Tensor,
        recovery_active: Tensor,
        recovery_phase_logits: Tensor,
        approach_contact_streak: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        """Return adapted plan, precision, immutable base, and recovery delta.

        The recovery adapter is an explicitly source-null planning branch.  It
        is allowed to change a plan only after this agent has observed a real
        rejected contact transition.  This makes additional post-contact
        capacity compatible with exact preservation of every pre-contact
        source trajectory, independent of the adapter weights.
        """

        planner_parts = [
            learned_obs,
            local_h.detach(),
            recovery_state.detach(),
        ]
        if self.arm_cfg.local_se2_shape_signature:
            planner_parts.append(self.local_se2_shape_geometry_signature(learned_obs))
        planner_inputs = torch.cat(tuple(planner_parts), dim=-1)
        planner_h = self.local_se2_planner_encoder(planner_inputs)
        base_mean = torch.tanh(self.local_se2_plan_mean_head(planner_h))
        plan_log_precision = self.local_se2_plan_log_precision_head(planner_h)
        precision = F.softplus(plan_log_precision) + 1.0e-3
        delta = torch.zeros_like(base_mean)
        mean = base_mean
        distributed_recovery_mode_logits: Tensor | None = None
        distributed_recovery_selected_mode: Tensor | None = None
        distributed_recovery_pre_mean: Tensor | None = None
        if self.arm_cfg.local_se2_recovery_adapter:
            phase_probability = F.softmax(recovery_phase_logits, dim=-1)
            recovery_inputs = torch.cat(
                (
                    learned_obs,
                    recovery_state,
                    phase_probability,
                    base_mean.detach(),
                ),
                dim=-1,
            )
            if self.arm_cfg.local_se2_recovery_canonical_features:
                recovery_inputs = torch.cat(
                    (
                        recovery_inputs,
                        local_se2_recovery_canonical_features(
                            learned_obs, self.env_cfg
                        ),
                    ),
                    dim=-1,
                )
            recovery_h = self.local_se2_recovery_adapter_encoder(recovery_inputs)
            proposed_delta = self.arm_cfg.local_se2_recovery_adapter_scale * torch.tanh(
                self.local_se2_recovery_adapter_head(recovery_h)
            )
            delta = torch.where(
                recovery_active,
                proposed_delta,
                torch.zeros_like(proposed_delta),
            )
            mean = (base_mean + delta).clamp(-1.0, 1.0)
            # The clamp is algebraically the identity on the base tanh when
            # inactive, but make source parity explicit and auditable.
            mean = torch.where(recovery_active, mean, base_mean)
        if self.arm_cfg.local_se2_contact_rebel_adapter:
            if learned_obs.shape[-1] != _SINGLE_SLIT_CONTACT_OBSERVATION_DIM:
                raise ValueError("contact rebel adapter requires observation27")
            phase_probability = F.softmax(recovery_phase_logits, dim=-1)
            canonical_recovery = local_se2_recovery_canonical_features(
                learned_obs, self.env_cfg
            )
            contact_rebel_inputs = torch.cat(
                (
                    canonical_recovery,
                    learned_obs[..., 18:27],
                    recovery_state.detach(),
                    phase_probability.detach(),
                    mean.detach(),
                ),
                dim=-1,
            )
            rebel_h = self.local_se2_contact_rebel_encoder(contact_rebel_inputs)
            proposed_rebel_delta = (
                self.arm_cfg.local_se2_contact_rebel_scale
                * torch.tanh(self.local_se2_contact_rebel_head(rebel_h))
            )
            payload_center_x = canonical_recovery[..., 2:3] * self.env_cfg.world_x
            exit_side = (
                payload_center_x
                >= self.arm_cfg.local_se2_contact_rebel_min_center_x_m
            )
            # Contact history is the causal trigger; the wall-midplane phase
            # support prevents any change to the already successful approach
            # and retreat. Keeping the residual active between impacts lets it
            # execute a multi-tick corrective plan instead of alternating one
            # useful step with one legacy collision.
            rebel_support = recovery_active & exit_side
            rebel_delta = torch.where(
                rebel_support,
                proposed_rebel_delta,
                torch.zeros_like(proposed_rebel_delta),
            )
            proposed_mean = (mean + rebel_delta).clamp(-1.0, 1.0)
            selected_mean = torch.where(rebel_delta != 0.0, proposed_mean, mean)
            mean = _forward_exact_backward_proposed(selected_mean, proposed_mean)
            delta = delta + rebel_delta
        if self.arm_cfg.local_se2_exit_refinement_adapter:
            # The configuration contract guarantees that the first rebel
            # branch computed these causal/local tensors above.
            refinement_inputs = torch.cat(
                (
                    canonical_recovery,
                    learned_obs[..., 18:27],
                    recovery_state.detach(),
                    phase_probability.detach(),
                    mean.detach(),
                ),
                dim=-1,
            )
            refinement_h = self.local_se2_exit_refinement_encoder(
                refinement_inputs
            )
            proposed_refinement_delta = (
                self.arm_cfg.local_se2_exit_refinement_scale
                * torch.tanh(self.local_se2_exit_refinement_head(refinement_h))
            )
            refinement_support = recovery_active & (
                payload_center_x
                >= self.arm_cfg.local_se2_exit_refinement_min_center_x_m
            )
            refinement_delta = torch.where(
                refinement_support,
                proposed_refinement_delta,
                torch.zeros_like(proposed_refinement_delta),
            )
            proposed_mean = (mean + refinement_delta).clamp(-1.0, 1.0)
            selected_mean = torch.where(
                refinement_delta != 0.0, proposed_mean, mean
            )
            mean = _forward_exact_backward_proposed(selected_mean, proposed_mean)
            delta = delta + refinement_delta
        if self.arm_cfg.local_se2_approach_refinement_adapter:
            if approach_contact_streak is None:
                raise RuntimeError("approach refinement omitted its contact streak")
            approach_inputs = torch.cat(
                (
                    canonical_recovery,
                    learned_obs[..., 18:27],
                    recovery_state.detach(),
                    phase_probability.detach(),
                    mean.detach(),
                ),
                dim=-1,
            )
            approach_h = self.local_se2_approach_refinement_encoder(
                approach_inputs
            )
            proposed_approach_delta = (
                self.arm_cfg.local_se2_approach_refinement_scale
                * torch.tanh(self.local_se2_approach_refinement_head(approach_h))
            )
            approach_support = recovery_active & (
                payload_center_x
                >= self.arm_cfg.local_se2_approach_refinement_min_center_x_m
            ) & (
                approach_contact_streak
                >= self.arm_cfg.local_se2_approach_refinement_stuck_steps
            )
            approach_delta = torch.where(
                approach_support,
                proposed_approach_delta,
                torch.zeros_like(proposed_approach_delta),
            )
            proposed_mean = (mean + approach_delta).clamp(-1.0, 1.0)
            selected_mean = torch.where(
                approach_delta != 0.0, proposed_mean, mean
            )
            mean = _forward_exact_backward_proposed(selected_mean, proposed_mean)
            delta = delta + approach_delta
        if self.arm_cfg.local_se2_deep_stuck_refinement_adapter:
            if approach_contact_streak is None:
                raise RuntimeError("deep-stuck refinement omitted its contact streak")
            deep_stuck_inputs = torch.cat(
                (
                    canonical_recovery,
                    learned_obs[..., 18:27],
                    recovery_state.detach(),
                    phase_probability.detach(),
                    mean.detach(),
                ),
                dim=-1,
            )
            deep_stuck_h = self.local_se2_deep_stuck_refinement_encoder(
                deep_stuck_inputs
            )
            proposed_deep_stuck_delta = (
                self.arm_cfg.local_se2_deep_stuck_refinement_scale
                * torch.tanh(
                    self.local_se2_deep_stuck_refinement_head(deep_stuck_h)
                )
            )
            deep_stuck_support = recovery_active & (
                payload_center_x
                >= self.arm_cfg.local_se2_deep_stuck_refinement_min_center_x_m
            ) & (
                approach_contact_streak
                >= self.arm_cfg.local_se2_deep_stuck_refinement_stuck_steps
            )
            deep_stuck_delta = torch.where(
                deep_stuck_support,
                proposed_deep_stuck_delta,
                torch.zeros_like(proposed_deep_stuck_delta),
            )
            proposed_mean = (mean + deep_stuck_delta).clamp(-1.0, 1.0)
            selected_mean = torch.where(
                deep_stuck_delta != 0.0, proposed_mean, mean
            )
            mean = _forward_exact_backward_proposed(selected_mean, proposed_mean)
            delta = delta + deep_stuck_delta
        if self.arm_cfg.local_se2_retry_refinement_thresholds:
            if approach_contact_streak is None:
                raise RuntimeError("retry refinement omitted its contact streak")
            for encoder, head, threshold in zip(
                self.local_se2_retry_refinement_encoders,
                self.local_se2_retry_refinement_heads,
                self.arm_cfg.local_se2_retry_refinement_thresholds,
                strict=True,
            ):
                retry_inputs = torch.cat(
                    (
                        canonical_recovery,
                        learned_obs[..., 18:27],
                        recovery_state.detach(),
                        phase_probability.detach(),
                        mean.detach(),
                        approach_contact_streak.clamp_max(float(threshold))
                        / float(threshold),
                    ),
                    dim=-1,
                )
                retry_h = encoder(retry_inputs)
                proposed_retry_delta = (
                    self.arm_cfg.local_se2_retry_refinement_scale
                    * torch.tanh(head(retry_h))
                )
                retry_support = recovery_active & (
                    payload_center_x
                    >= self.arm_cfg.local_se2_retry_refinement_min_center_x_m
                ) & (approach_contact_streak >= threshold)
                retry_delta = torch.where(
                    retry_support,
                    proposed_retry_delta,
                    torch.zeros_like(proposed_retry_delta),
                )
                proposed_mean = (mean + retry_delta).clamp(-1.0, 1.0)
                selected_mean = torch.where(
                    retry_delta != 0.0, proposed_mean, mean
                )
                mean = _forward_exact_backward_proposed(
                    selected_mean, proposed_mean
                )
                delta = delta + retry_delta
        if self.arm_cfg.local_se2_distributed_recovery_adapter:
            distributed_recovery_pre_mean = mean
            recovery_inputs = torch.cat(
                (planner_inputs, mean.detach()), dim=-1
            )
            recovery_h = self.local_se2_distributed_recovery_encoder(
                recovery_inputs
            )
            if self.arm_cfg.local_se2_distributed_recovery_mode_count == 1:
                proposed_recovery_delta = (
                    self.arm_cfg.local_se2_distributed_recovery_scale
                    * torch.tanh(
                        self.local_se2_distributed_recovery_head(recovery_h)
                    )
                )
            else:
                mode_count = (
                    self.arm_cfg.local_se2_distributed_recovery_mode_count
                )
                mode_logits = (
                    self.local_se2_distributed_recovery_mode_logits_head(
                        recovery_h
                    )
                )
                mode_residuals = (
                    self.arm_cfg.local_se2_distributed_recovery_scale
                    * torch.tanh(
                        self.local_se2_distributed_recovery_mode_residual_head(
                            recovery_h
                        )
                    ).view(
                        *recovery_h.shape[:-1],
                        mode_count,
                        self.arm_cfg.local_se2_plan_dim,
                    )
                )
                selected_mode = mode_logits.argmax(dim=-1)
                distributed_recovery_mode_logits = mode_logits
                distributed_recovery_selected_mode = selected_mode
                selected_index = selected_mode[..., None, None].expand(
                    *selected_mode.shape,
                    1,
                    self.arm_cfg.local_se2_plan_dim,
                )
                proposed_recovery_delta = torch.gather(
                    mode_residuals, -2, selected_index
                ).squeeze(-2)
            distributed_recovery_delta = torch.where(
                recovery_active,
                proposed_recovery_delta,
                torch.zeros_like(proposed_recovery_delta),
            )
            proposed_mean = (mean + distributed_recovery_delta).clamp(-1.0, 1.0)
            mean = torch.where(recovery_active, proposed_mean, mean)
            delta = delta + distributed_recovery_delta
        if self.arm_cfg.local_se2_geometry_refinement_adapter:
            mean, geometry_delta = (
                self.local_se2_geometry_refinement_from_planner_inputs(
                    planner_inputs, mean
                )
            )
            delta = delta + geometry_delta
        if self.arm_cfg.local_se2_geometry_mode_adapter:
            mean, mode_delta, _selected_mode = (
                self.local_se2_geometry_mode_refinement_from_planner_inputs(
                    planner_inputs,
                    mean,
                    active=(
                        recovery_active
                        if self.arm_cfg.local_se2_geometry_mode_post_contact_only
                        else None
                    ),
                )
            )
            delta = delta + mode_delta
        if self.arm_cfg.local_se2_markov_shape_planner:
            markov_mean, markov_precision, markov_log_precision, support = (
                self.local_se2_markov_shape_outputs_from_planner_inputs(
                    planner_inputs
                )
            )
            incumbent_mean = mean
            proposed_mean = (1.0 - support) * mean + support * markov_mean
            proposed_precision = (
                (1.0 - support) * precision + support * markov_precision
            )
            proposed_log_precision = (
                (1.0 - support) * plan_log_precision
                + support * markov_log_precision
            )
            mean = torch.where(support != 0.0, proposed_mean, mean)
            precision = torch.where(
                support != 0.0, proposed_precision, precision
            )
            plan_log_precision = torch.where(
                support != 0.0,
                proposed_log_precision,
                plan_log_precision,
            )
            delta = delta + (mean - incumbent_mean)
        if self.arm_cfg.local_se2_path_memory:
            memory_mean, memory_active = self.local_se2_path_memory_plan(
                planner_inputs,
                mean,
                recovery_active=recovery_active,
            )
            memory_delta = memory_mean - mean
            mean = torch.where(memory_active, memory_mean, mean)
            delta = delta + torch.where(
                memory_active,
                memory_delta,
                torch.zeros_like(memory_delta),
            )
        return (
            mean,
            precision,
            plan_log_precision,
            base_mean,
            delta,
            distributed_recovery_mode_logits,
            distributed_recovery_selected_mode,
            distributed_recovery_pre_mean,
        )

    def local_se2_base_planner_inputs_and_next_private_hidden(
        self,
        obs: Tensor,
        hidden: Tensor | None,
        *,
        active_leg: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return planner input and its minimal recurrent continuation.

        This extraction is intended for frozen-context planner distillation.
        It runs the same observation gate, recurrent encoder, and recovery
        state update as :meth:`forward`, but does not execute GBP or decode a
        force. For the conventional recurrent encoder, the omitted GBP stalk
        history cannot affect any later private-planner input: only the GRU
        state is consumed there. We therefore carry a shape-correct zero stalk
        placeholder while preserving the exact GRU and recovery recurrences.
        Queryable temporal policies use their exact private temporal update.

        The returned input contains no peer message and can be stored compactly
        as an authenticated training record.
        """

        if not self.arm_cfg.local_se2_planner:
            raise ValueError("planner input extraction requires local_se2_planner")
        learned_obs = contact_conditioned_learned_observation(obs)
        legacy_hidden, recovery_previous, approach_previous = (
            self.split_persistent_recovery_hidden(hidden, obs)
        )
        if recovery_previous is None:
            raise RuntimeError("planner input extraction requires persistent recovery")
        recovery_state, _active, _phase = self.update_persistent_recovery_state(
            obs, recovery_previous
        )
        approach_streak = self.update_approach_contact_streak(
            obs, approach_previous
        )
        if self.arm_cfg.queryable_goal_free_temporal:
            local_h, next_legacy_hidden, _memory_count = self.goal_free_temporal_encode(
                obs, legacy_hidden, active_leg
            )
        else:
            rec_hidden, history, history_mask = self.split_hidden(
                legacy_hidden, obs
            )
            encoded = self.obs_encoder(learned_obs)
            batch, n_agents, _ = encoded.shape
            local_h = self.gru(
                encoded.reshape(batch * n_agents, -1),
                rec_hidden.reshape(batch * n_agents, -1),
            ).view(batch, n_agents, -1)
            # The private planner never consumes the GBP stalk-history suffix.
            # A zero placeholder preserves hidden shape without paying for GBP.
            placeholder_mean = obs.new_zeros(
                batch, n_agents, self.arm_cfg.stalk_dim
            )
            next_legacy_hidden = self.pack_hidden(
                local_h, history, history_mask, placeholder_mean
            )
        planner_parts = [learned_obs, local_h.detach(), recovery_state.detach()]
        if self.arm_cfg.local_se2_shape_signature:
            planner_parts.append(self.local_se2_shape_geometry_signature(learned_obs))
        result = torch.cat(tuple(planner_parts), dim=-1)
        expected = self.local_se2_planner_encoder[0].in_features
        if result.shape != (*obs.shape[:2], expected):
            raise RuntimeError(
                f"base planner input has shape {tuple(result.shape)}, expected "
                f"{(*obs.shape[:2], expected)}"
            )
        next_hidden = torch.cat((next_legacy_hidden, recovery_state), dim=-1)
        if self.arm_cfg.local_se2_approach_refinement_adapter:
            if approach_streak is None:
                raise RuntimeError("private planner continuation omitted contact streak")
            next_hidden = torch.cat((next_hidden, approach_streak), dim=-1)
        return result, next_hidden

    def local_se2_base_planner_inputs(
        self,
        obs: Tensor,
        hidden: Tensor | None,
        *,
        active_leg: Tensor | None = None,
    ) -> Tensor:
        """Return the exact private input consumed by the base planner MLP."""

        result, _next_private_hidden = (
            self.local_se2_base_planner_inputs_and_next_private_hidden(
                obs, hidden, active_leg=active_leg
            )
        )
        return result

    def local_se2_shape_geometry_signature(self, reference: Tensor) -> Tensor:
        """Return a private analytic geometric descriptor of the exact body.

        A convex support function would erase exactly the concavities that
        distinguish glyphs such as C/O, E/F, and P/R.  Instead, every exact
        collision rectangle contributes its physical area to eight smooth
        body-frame probes.  The descriptor contains no letter ID, changes
        continuously with geometry, and is identical for all agents carrying
        the same rigid body.  It is private planner context: no coordinate is
        appended to a Sheaf-GBP packet.
        """

        if not self.arm_cfg.local_se2_shape_signature:
            raise ValueError("shape signature is disabled")
        if reference.ndim != 3:
            raise ValueError("shape signature reference must be [batch,agent,feature]")
        rectangles = payload_rectangles_for_config(
            self.env_cfg, reference.device
        ).to(reference.dtype)
        signature = self._local_se2_geometry_signature_from_rectangles(
            rectangles,
            reference,
            dimensions=self.arm_cfg.local_se2_shape_signature_dim,
        )
        return signature.view(1, 1, -1).expand(*reference.shape[:2], -1)

    def local_se2_markov_shape_inputs_from_planner_inputs(
        self, planner_inputs: Tensor
    ) -> Tensor:
        """Extract the route-free morphology-planner input.

        The serialized planner record may contain the legacy recurrent hidden
        for source-policy auditing.  This extractor deliberately skips those
        columns: its output is only current physical/goal geometry, the short
        causal contact state, and the continuous payload descriptor.
        """

        if not self.arm_cfg.local_se2_markov_shape_planner:
            raise ValueError("Markov shape planner is disabled")
        expected = self.local_se2_planner_encoder[0].in_features
        if planner_inputs.ndim != 3 or planner_inputs.shape[-1] != expected:
            raise ValueError("Markov shape planner received malformed planner input")
        obs = planner_inputs[..., : self.arm_cfg.obs_dim]
        recovery_start = self.arm_cfg.obs_dim + self.arm_cfg.hidden_dim
        recovery = planner_inputs[
            ...,
            recovery_start : recovery_start + self.arm_cfg.persistent_recovery_dim,
        ]
        signature = planner_inputs[
            ..., -self.arm_cfg.local_se2_shape_signature_dim :
        ]
        canonical = local_se2_recovery_canonical_features(obs, self.env_cfg)
        result = torch.cat((canonical, recovery, signature), dim=-1)
        expected_markov = self.local_se2_markov_shape_encoder[0].in_features
        if result.shape != (*planner_inputs.shape[:2], expected_markov):
            raise RuntimeError("Markov shape planner input width changed")
        return result

    def local_se2_markov_goal_features(self, features: Tensor) -> Tensor:
        """Existing active mission command, expressed in each node's own frame.

        These four values are private inputs, not extra GBP coordinates. The
        original geometry controller already receives the same goal fields.
        No target index, next target, shape label or route is consumed.
        """
        if features.shape[-1] != self.local_se2_markov_shape_encoder[0].in_features:
            raise ValueError("goal context requires the original morphology features")
        required = float(self.env_cfg.goal_orientation_required)
        goal_angle = features.new_tensor(self.env_cfg.goal_theta)
        current_sine, current_cosine = features[..., 0], features[..., 1]
        return torch.stack((
            float(self.env_cfg.goal_y / self.env_cfg.world_y) - features[..., 3],
            required * (goal_angle.sin() * current_cosine - goal_angle.cos() * current_sine),
            required * (goal_angle.cos() * current_cosine + goal_angle.sin() * current_sine),
            torch.full_like(current_sine, 2. * required - 1.),
        ), dim=-1)

    def local_se2_markov_shape_encode(self, features: Tensor, goal_context: Tensor | None = None) -> Tensor:
        """Same original MLP, with an optional partitioned first affine input.

        W*f + V*g + b is equivalent to appending four input columns. Keeping
        W's GEMM unchanged and selecting its exact result at V*g == 0 avoids
        bridge-time rounding drift while retaining a live gradient for V.
        """
        if not self.arm_cfg.local_se2_markov_goal_context:
            return self.local_se2_markov_shape_encoder(features)
        command = self.local_se2_markov_goal_features(features) if goal_context is None else goal_context
        if command.shape != (*features.shape[:-1], 4):
            raise ValueError("goal command context must contain four values per node")
        base = self.local_se2_markov_shape_encoder[0](features)
        delta = self.local_se2_markov_goal_projection(command)
        proposed = base + delta
        exact = torch.where(delta != 0., proposed, base)
        hidden = _forward_exact_backward_proposed(exact, proposed)
        for layer in list(self.local_se2_markov_shape_encoder)[1:]:
            hidden = layer(hidden)
        return hidden

    def local_se2_noncanonical_geometry_support(
        self, signature: Tensor
    ) -> Tensor:
        """Continuous T-null support for route-free morphology planning."""

        if signature.shape[-1] != self.arm_cfg.local_se2_shape_signature_dim:
            raise ValueError("geometry support received the wrong signature width")
        canonical_rectangles = payload_shape_rectangles(
            "T", signature.device
        ).to(signature.dtype)
        canonical_signature = self._local_se2_geometry_signature_from_rectangles(
            canonical_rectangles,
            signature,
            dimensions=self.arm_cfg.local_se2_shape_signature_dim,
        )
        return (
            torch.linalg.vector_norm(
                signature - canonical_signature.view(1, 1, -1),
                dim=-1,
                keepdim=True,
            )
            / 0.25
        ).clamp(0.0, 1.0)

    def local_se2_markov_shape_outputs_from_planner_inputs(
        self, planner_inputs: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return route-free plan mean/precision and exact T-null support."""

        features = self.local_se2_markov_shape_inputs_from_planner_inputs(
            planner_inputs
        )
        hidden = self.local_se2_markov_shape_encode(features)
        learned_mean = torch.tanh(self.local_se2_markov_shape_mean_head(hidden))
        # Coordinates 6:8 used to be supervised by teacher path progress.
        # A route-free planner must not reproduce that clock.  Replace them
        # with instantaneous, recomputable semantics: local slit cross-track
        # error and spatial goal progress.  Backtracking therefore decreases
        # progress, and changing the goal changes the value immediately.
        cross_track = (4.0 * features[..., 5:6]).clamp(-1.0, 1.0)
        canonical_mission_dx = 2.10 / 2.60
        spatial_goal_progress = (
            1.0 - features[..., 4:5].abs() / canonical_mission_dx
        ).clamp(0.0, 1.0)
        mean = torch.cat(
            (learned_mean[..., :6], cross_track, spatial_goal_progress),
            dim=-1,
        )
        # This value is also fed (through tanh) to the frozen downstream force
        # decoder.  Keep it inside the source planner's calibrated range;
        # an unconstrained Gaussian NLL can otherwise drive raw logits into
        # the hundreds while teacher-state mean loss continues to improve.
        log_precision = 0.5 * torch.tanh(
            self.local_se2_markov_shape_log_precision_head(hidden)
        )
        precision = F.softplus(log_precision) + 1.0e-3
        signature = planner_inputs[
            ..., -self.arm_cfg.local_se2_shape_signature_dim :
        ]
        support = self.local_se2_noncanonical_geometry_support(signature)
        return mean, precision, log_precision, support

    @staticmethod
    def _local_se2_geometry_signature_from_rectangles(
        rectangles: Tensor,
        reference: Tensor,
        *,
        dimensions: int = 8,
    ) -> Tensor:
        """Evaluate the nested V1/V2 descriptor for exact rectangles."""

        if dimensions not in {8, 32}:
            raise ValueError("geometry signature dimensions must be 8 or 32")

        centres = rectangles.mean(dim=1)
        edge_a = rectangles[:, 1] - rectangles[:, 0]
        edge_b = rectangles[:, 3] - rectangles[:, 0]
        areas = torch.abs(
            edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
        )
        probes = reference.new_tensor(
            (
                (-0.22, 0.30),
                (0.00, 0.30),
                (0.22, 0.30),
                (-0.22, -0.045),
                (0.22, -0.045),
                (-0.22, -0.39),
                (0.00, -0.39),
                (0.22, -0.39),
            )
        )
        squared_distance = (
            centres[:, None, :] - probes[None, :, :]
        ).square().sum(-1)
        legacy_density = (
            areas[:, None] * torch.exp(-squared_distance / (2.0 * 0.18**2))
        ).sum(dim=0) / 0.10
        legacy_density = legacy_density.clamp(0.0, 3.0)
        if dimensions == 8:
            return legacy_density

        # Sixteen finer density probes preserve concavity and stroke layout.
        grid_x = reference.new_tensor((-0.30, -0.10, 0.10, 0.30))
        grid_y = reference.new_tensor((0.32, 0.08, -0.16, -0.40))
        fine_probes = torch.cartesian_prod(grid_x, grid_y)
        fine_distance = (
            centres[:, None, :] - fine_probes[None, :, :]
        ).square().sum(-1)
        fine_density = (
            areas[:, None] * torch.exp(-fine_distance / (2.0 * 0.12**2))
        ).sum(dim=0) / 0.06
        fine_density = fine_density.clamp(0.0, 3.0)

        # Eight support samples retain overall scale and aspect while the
        # density probes above distinguish bodies with the same convex hull.
        angles = torch.arange(8, device=reference.device, dtype=reference.dtype)
        angles = angles * (2.0 * torch.pi / 8.0)
        directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        support = torch.einsum("rvc,dc->rvd", rectangles, directions).amax(dim=(0, 1))
        support = (support / 0.60).clamp(-1.0, 1.0)
        return torch.cat((legacy_density, fine_density, support), dim=0)

    def local_se2_geometry_refinement_from_planner_inputs(
        self,
        planner_inputs: Tensor,
        incumbent_mean: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply final noncanonical geometry capacity to a private plan.

        The multiplier is a continuous distance between the observed physical
        geometry descriptor and canonical T. It is exactly zero for canonical
        T under any state or population, so training this branch cannot alter
        the source skill. The branch remains private; its output is converted
        to the existing wrench/sheaf factors downstream.
        """

        if not self.arm_cfg.local_se2_geometry_refinement_adapter:
            raise ValueError("geometry refinement adapter is disabled")
        expected = self.local_se2_planner_encoder[0].in_features
        if planner_inputs.shape != (*incumbent_mean.shape[:2], expected):
            raise ValueError("geometry refinement planner input has the wrong shape")
        if incumbent_mean.shape[-1] != self.arm_cfg.local_se2_plan_dim:
            raise ValueError("geometry refinement incumbent plan has the wrong width")
        current_signature = planner_inputs[
            ..., -self.arm_cfg.local_se2_shape_signature_dim :
        ]
        canonical_rectangles = payload_shape_rectangles(
            "T", planner_inputs.device
        ).to(planner_inputs.dtype)
        canonical_signature = self._local_se2_geometry_signature_from_rectangles(
            canonical_rectangles,
            planner_inputs,
            dimensions=self.arm_cfg.local_se2_shape_signature_dim,
        )
        # Preserve the V1 support/gating scalar exactly when richer private
        # geometry features are bridged in. The additional coordinates are
        # ordinary network inputs, not a new hard gate.
        support_width = min(8, self.arm_cfg.local_se2_shape_signature_dim)
        geometry_support = (
            torch.linalg.vector_norm(
                current_signature[..., :support_width]
                - canonical_signature[:support_width].view(1, 1, support_width),
                dim=-1,
                keepdim=True,
            )
            / 0.25
        ).clamp(0.0, 1.0)
        canonical_family_distance = (
            current_signature - canonical_signature.view(1, 1, -1)
        ).square().mean(dim=-1, keepdim=True)
        preserve_parametric_t_parent = (
            canonical_family_distance
            <= self.arm_cfg.local_se2_path_memory_canonical_family_mse_threshold
        )
        geometry_support = torch.where(
            preserve_parametric_t_parent,
            torch.zeros_like(geometry_support),
            geometry_support,
        )
        feature_parts = [planner_inputs, incumbent_mean]
        if self.arm_cfg.local_se2_geometry_refinement_canonical_features:
            feature_parts.append(
                local_se2_recovery_canonical_features(
                    planner_inputs[..., : self.arm_cfg.obs_dim],
                    self.env_cfg,
                )
            )
        features = torch.cat(tuple(feature_parts), dim=-1)
        hidden = self.local_se2_geometry_refinement_encoder(features)
        proposed = (
            self.arm_cfg.local_se2_geometry_refinement_scale
            * geometry_support
            * torch.tanh(self.local_se2_geometry_refinement_head(hidden))
        )
        refined = (incumbent_mean + proposed).clamp(-1.0, 1.0)
        selected = torch.where(
            geometry_support != 0.0,
            refined,
            incumbent_mean,
        )
        return selected, selected - incumbent_mean

    def local_se2_geometry_mode_outputs(
        self,
        planner_inputs: Tensor,
        incumbent_mean: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return four private maneuver logits, residuals, and T-null support."""

        if not self.arm_cfg.local_se2_geometry_mode_adapter:
            raise ValueError("geometry mode adapter is disabled")
        expected = self.local_se2_planner_encoder[0].in_features
        if planner_inputs.shape != (*incumbent_mean.shape[:2], expected):
            raise ValueError("geometry mode planner input has the wrong shape")
        if incumbent_mean.shape[-1] != self.arm_cfg.local_se2_plan_dim:
            raise ValueError("geometry mode incumbent plan has the wrong width")
        signature_width = self.arm_cfg.local_se2_shape_signature_dim
        current_signature = planner_inputs[..., -signature_width:]
        canonical_rectangles = payload_shape_rectangles(
            "T", planner_inputs.device
        ).to(planner_inputs.dtype)
        canonical_signature = self._local_se2_geometry_signature_from_rectangles(
            canonical_rectangles,
            planner_inputs,
            dimensions=signature_width,
        )
        support_width = min(8, signature_width)
        geometry_support = (
            torch.linalg.vector_norm(
                current_signature[..., :support_width]
                - canonical_signature[:support_width].view(
                    1, 1, support_width
                ),
                dim=-1,
                keepdim=True,
            )
            / 0.25
        ).clamp(0.0, 1.0)
        canonical = local_se2_recovery_canonical_features(
            planner_inputs[..., : self.arm_cfg.obs_dim],
            self.env_cfg,
        )
        features = torch.cat((planner_inputs, incumbent_mean, canonical), dim=-1)
        hidden = self.local_se2_geometry_mode_encoder(features)
        logits = self.local_se2_geometry_mode_logits_head(hidden)
        residuals = self.local_se2_geometry_mode_residual_head(hidden).view(
            *planner_inputs.shape[:2],
            self.arm_cfg.local_se2_geometry_mode_count,
            self.arm_cfg.local_se2_plan_dim,
        )
        residuals = (
            self.arm_cfg.local_se2_geometry_mode_scale
            * geometry_support.unsqueeze(-2)
            * torch.tanh(residuals)
        )
        return logits, residuals, geometry_support

    def local_se2_geometry_mode_refinement_from_planner_inputs(
        self,
        planner_inputs: Tensor,
        incumbent_mean: Tensor,
        *,
        active: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Hard-select one mutually exclusive private maneuver expert."""

        logits, residuals, geometry_support = self.local_se2_geometry_mode_outputs(
            planner_inputs,
            incumbent_mean,
        )
        selected_mode = logits.argmax(dim=-1)
        index = selected_mode[..., None, None].expand(
            *selected_mode.shape,
            1,
            self.arm_cfg.local_se2_plan_dim,
        )
        selected_residual = torch.gather(residuals, -2, index).squeeze(-2)
        proposed = (incumbent_mean + selected_residual).clamp(-1.0, 1.0)
        support = geometry_support != 0.0
        if active is not None:
            if active.shape != geometry_support.shape or active.dtype != torch.bool:
                raise ValueError("geometry mode active mask must be boolean [B,N,1]")
            support = support & active
        selected = torch.where(
            support,
            proposed,
            incumbent_mean,
        )
        return selected, selected - incumbent_mean, selected_mode

    def local_se2_path_memory_plan(
        self,
        planner_inputs: Tensor,
        incumbent_mean: Tensor,
        *,
        recovery_active: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Retrieve a certified path and emit its next local feedback plan.

        Shape lookup uses the continuous 32-D analytic descriptor, never a
        letter ID.  Waypoint lookup uses only the payload pose reconstructed
        from each agent's strict-local observation.  Thus every handle runs
        the same algorithm independently; no hidden all-reduce is introduced.
        """

        if not self.arm_cfg.local_se2_path_memory:
            raise ValueError("local SE(2) path memory is disabled")
        expected = self.local_se2_planner_encoder[0].in_features
        if planner_inputs.shape != (*incumbent_mean.shape[:2], expected):
            raise ValueError("path-memory planner input has the wrong shape")
        if incumbent_mean.shape[-1] != self.arm_cfg.local_se2_plan_dim:
            raise ValueError("path-memory incumbent plan has the wrong width")
        if bool((self.local_se2_path_memory_horizons <= 0).any()):
            raise RuntimeError("path-memory checkpoint contains an empty path")

        signature = planner_inputs[..., -32:]
        shape_distance = (
            signature.unsqueeze(-2)
            - self.local_se2_path_memory_shape_signatures.view(1, 1, -1, 32)
        ).square().mean(dim=-1)
        minimum_shape_distance, shape_index = shape_distance.min(dim=-1)
        exact_train_shape = minimum_shape_distance <= 1.0e-12
        canonical_rectangles = payload_shape_rectangles(
            "T", planner_inputs.device
        ).to(planner_inputs.dtype)
        canonical_signature = self._local_se2_geometry_signature_from_rectangles(
            canonical_rectangles,
            planner_inputs,
            dimensions=32,
        )
        canonical_family_distance = (
            signature - canonical_signature.view(1, 1, 32)
        ).square().mean(dim=-1)
        preserve_parametric_t_parent = (~exact_train_shape) & (
            canonical_family_distance
            <= self.arm_cfg.local_se2_path_memory_canonical_family_mse_threshold
        )
        paths = self.local_se2_path_memory_reference_poses[shape_index]
        horizons = self.local_se2_path_memory_horizons[shape_index]
        population_resampling = torch.zeros_like(exact_train_shape)
        if self.arm_cfg.local_se2_path_memory_ridge_decoder:
            design = torch.cat(
                (signature, torch.ones_like(signature[..., :1])), dim=-1
            )
            ridge_paths = torch.einsum(
                "...f,fpc->...pc",
                design,
                self.local_se2_path_memory_ridge_weights,
            )
            ridge_paths = ridge_paths.clone()
            ridge_paths[..., 0, :] = ridge_paths.new_tensor((-1.05, 0.0, 0.0))
            ridge_points = self.arm_cfg.local_se2_path_memory_ridge_points
            ridge_paths[..., ridge_points - 1, :] = ridge_paths.new_tensor(
                (1.05, 0.0, 0.0)
            )
            lower = (
                self.local_se2_path_memory_train_pose_min
                - self.arm_cfg.local_se2_path_memory_ridge_trust_margin
            )
            upper = (
                self.local_se2_path_memory_train_pose_max
                + self.arm_cfg.local_se2_path_memory_ridge_trust_margin
            )
            ridge_prefix = ridge_paths[..., :ridge_points, :]
            ridge_trusted = (
                torch.isfinite(ridge_prefix).all(dim=(-2, -1))
                & (ridge_prefix >= lower).all(dim=(-2, -1))
                & (ridge_prefix <= upper).all(dim=(-2, -1))
            )
            # Exact train geometries retain their individually certified path.
            # Only genuine continuous-signature OOD can invoke the shared
            # decoder; out-of-envelope extrapolation falls back to nearest
            # certified memory without consulting a held-out identity label.
            use_ridge = (
                (~exact_train_shape)
                & ridge_trusted
                & (~preserve_parametric_t_parent)
            )
            paths = torch.where(use_ridge[..., None, None], ridge_paths, paths)
            ridge_horizon = torch.full_like(horizons, ridge_points - 1)
            horizons = torch.where(use_ridge, ridge_horizon, horizons)
        if self.arm_cfg.local_se2_path_memory_population_resampling:
            if recovery_active is None or recovery_active.shape != (
                *exact_train_shape.shape,
                1,
            ):
                raise ValueError(
                    "population resampling requires [B,N,1] recovery activity"
                )
            preemptive = self.env_cfg.n_agents in (
                self.arm_cfg.local_se2_path_memory_population_preemptive_counts
            )
            canonical_preemptive = (
                canonical_family_distance <= 1.0e-12
            ) & (
                self.env_cfg.n_agents
                in self.arm_cfg.local_se2_path_memory_canonical_preemptive_counts
            )
            recovery_allowed = self.env_cfg.n_agents in (
                self.arm_cfg.local_se2_path_memory_population_recovery_counts
            )
            population_resampling = (
                exact_train_shape
                & (self.env_cfg.n_agents != 12)
                & (
                    (recovery_active.squeeze(-1) & recovery_allowed)
                    | preemptive
                    | canonical_preemptive
                )
            )
            resampled_paths = (
                self.local_se2_path_memory_population_resampling_poses[shape_index]
            )
            paths = torch.where(
                population_resampling[..., None, None], resampled_paths, paths
            )
            resampled_horizon = torch.full_like(
                horizons,
                self.arm_cfg.local_se2_path_memory_population_resampling_points - 1,
            )
            horizons = torch.where(
                population_resampling, resampled_horizon, horizons
            )

        canonical = local_se2_recovery_canonical_features(
            planner_inputs[..., : self.arm_cfg.obs_dim],
            self.env_cfg,
        )
        center_x = canonical[..., 2] * self.env_cfg.world_x
        center_y = canonical[..., 3] * self.env_cfg.world_y
        theta = torch.atan2(canonical[..., 0], canonical[..., 1])
        delta_x = paths[..., 0] - center_x.unsqueeze(-1)
        delta_y = paths[..., 1] - center_y.unsqueeze(-1)
        delta_angle = torch.atan2(
            torch.sin(paths[..., 2] - theta.unsqueeze(-1)),
            torch.cos(paths[..., 2] - theta.unsqueeze(-1)),
        )
        distance2 = delta_x.square() + delta_y.square() + (0.35 * delta_angle).square()
        path_points = self.arm_cfg.local_se2_path_memory_max_steps + 1
        point_index = torch.arange(
            path_points,
            dtype=torch.long,
            device=planner_inputs.device,
        ).view(1, 1, path_points)
        distance2 = distance2.masked_fill(
            point_index > horizons.unsqueeze(-1),
            float("inf"),
        )
        nearest = distance2.argmin(dim=-1)
        target_index = torch.minimum(nearest + 1, horizons)
        target = torch.gather(
            paths,
            2,
            target_index[..., None, None].expand(
                *target_index.shape,
                1,
                3,
            ),
        ).squeeze(2)

        world_dx = target[..., 0] - center_x
        world_dy = target[..., 1] - center_y
        cosine = canonical[..., 1]
        sine = canonical[..., 0]
        body_dx = cosine * world_dx + sine * world_dy
        body_dy = -sine * world_dx + cosine * world_dy
        angle_delta = torch.atan2(
            torch.sin(target[..., 2] - theta),
            torch.cos(target[..., 2] - theta),
        )
        body_delta = torch.stack((body_dx, body_dy), dim=-1)
        desired_velocity_body = body_delta / float(self.env_cfg.dt)
        desired_omega = angle_delta / float(self.env_cfg.dt)
        plan = torch.cat(
            (
                (body_delta / LOCAL_SE2_POSITION_SCALE_M).clamp(-1.0, 1.0),
                (angle_delta / LOCAL_SE2_ANGLE_SCALE_RAD)
                .clamp(-1.0, 1.0)
                .unsqueeze(-1),
                (
                    desired_velocity_body
                    / LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
                ).clamp(-1.0, 1.0),
                (
                    desired_omega / LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS
                ).clamp(-1.0, 1.0).unsqueeze(-1),
                incumbent_mean[..., 6:8],
            ),
            dim=-1,
        )
        geometry_support = (
            torch.linalg.vector_norm(
                signature[..., :8] - canonical_signature[:8].view(1, 1, 8),
                dim=-1,
                keepdim=True,
            )
            / 0.25
        ).clamp(0.0, 1.0)
        active = (
            (geometry_support != 0.0)
            & (~preserve_parametric_t_parent.unsqueeze(-1))
        ) | population_resampling.unsqueeze(-1)
        if plan.shape != incumbent_mean.shape or not bool(torch.isfinite(plan).all()):
            raise RuntimeError("path-memory plan is malformed")
        return plan, active

    def predict_local_se2_recurrent_step(
        self,
        obs: Tensor,
        legacy_hidden: Tensor | None,
        recovery_previous: Tensor,
        approach_streak_previous: Tensor | None,
        *,
        active_leg: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Differentiable planner step with an explicit recovery recurrence."""

        if not self.arm_cfg.local_se2_planner:
            raise ValueError("local SE(2) plan prediction requires the planner arm")
        batch, n_agents, _ = obs.shape
        expected_recovery = (
            batch,
            n_agents,
            self.arm_cfg.persistent_recovery_dim,
        )
        if recovery_previous.shape != expected_recovery:
            raise ValueError(
                f"recovery previous must have shape {expected_recovery}, "
                f"got {tuple(recovery_previous.shape)}"
            )
        learned_obs = contact_conditioned_learned_observation(obs)
        recovery_state, recovery_active, recovery_phase_logits = (
            self.update_persistent_recovery_state(obs, recovery_previous)
        )
        approach_streak = self.update_approach_contact_streak(
            obs, approach_streak_previous
        )
        if self.arm_cfg.queryable_goal_free_temporal:
            local_h, _next_hidden, _memory_count = self.goal_free_temporal_encode(
                obs,
                legacy_hidden,
                active_leg,
            )
        else:
            rec_hidden, _history, _history_mask = self.split_hidden(
                legacy_hidden, obs
            )
            encoded = self.obs_encoder(learned_obs)
            local_h = self.gru(
                encoded.reshape(batch * n_agents, -1),
                rec_hidden.reshape(batch * n_agents, -1),
            ).view(batch, n_agents, -1)
        (
            mean,
            _precision,
            _log_precision,
            _base,
            _delta,
            _mode_logits,
            _mode,
            _pre_recovery,
        ) = (
            self._local_se2_plan_from_features(
                learned_obs,
                local_h,
                recovery_state,
                recovery_active,
                recovery_phase_logits,
                approach_streak,
            )
        )
        return mean, recovery_phase_logits, recovery_state, approach_streak

    def predict_local_se2_plan_mean_and_phase(
        self,
        obs: Tensor,
        hidden: Tensor | None = None,
        *,
        active_leg: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run only the private planner and phase memory, without GBP/decoder.

        This is algebraically identical to ``forward(...).aux[plan_mean]``.
        It exists so planner-head DAgger does not waste most of each update on
        a frozen distributed solver and frozen force decoder.
        """

        if not self.arm_cfg.local_se2_planner:
            raise ValueError("local SE(2) plan prediction requires the planner arm")
        legacy_hidden, recovery_previous, approach_streak_previous = (
            self.split_persistent_recovery_hidden(
            hidden,
            obs,
            )
        )
        if not self.arm_cfg.persistent_recovery_state or recovery_previous is None:
            raise RuntimeError("local SE(2) planner requires persistent recovery state")
        mean, recovery_phase_logits, _recovery_state, _approach_streak = (
            self.predict_local_se2_recurrent_step(
                obs,
                legacy_hidden,
                recovery_previous,
                approach_streak_previous,
                active_leg=active_leg,
            )
        )
        return mean, recovery_phase_logits

    def predict_local_se2_plan_mean(
        self,
        obs: Tensor,
        hidden: Tensor | None = None,
        *,
        active_leg: Tensor | None = None,
    ) -> Tensor:
        """Return the fast-path private plan while preserving the old API."""

        mean, _phase_logits = self.predict_local_se2_plan_mean_and_phase(
            obs,
            hidden,
            active_leg=active_leg,
        )
        return mean

    def forward(
        self,
        obs: Tensor,
        edge_index: Tensor,
        hidden: Tensor | None = None,
        *,
        trace_messages: bool = False,
        active_leg: Tensor | None = None,
        topology: GBPTopology | None = None,
        local_se2_plan_override: Tensor | None = None,
        final_local_se2_plan_override: Tensor | None = None,
        local_se2_wrench_unary_override: Tensor | None = None,
    ) -> PolicyOutput:
        batch, n_agents, _ = obs.shape
        if (
            local_se2_wrench_unary_override is not None
            and not self.arm_cfg.correction_sheaf
        ):
            raise ValueError("wrench-unary override requires the correction Sheaf")
        learned_obs = contact_conditioned_learned_observation(obs)
        legacy_hidden, recovery_previous, approach_streak_previous = (
            self.split_persistent_recovery_hidden(hidden, obs)
        )
        recovery_state: Tensor | None = None
        recovery_active: Tensor | None = None
        recovery_phase_logits: Tensor | None = None
        approach_streak: Tensor | None = None
        if self.arm_cfg.persistent_recovery_state:
            if recovery_previous is None:
                raise RuntimeError("persistent recovery state was not initialized")
            recovery_state, recovery_active, recovery_phase_logits = (
                self.update_persistent_recovery_state(obs, recovery_previous)
            )
            approach_streak = self.update_approach_contact_streak(
                obs, approach_streak_previous
            )
        goal_free_next_hidden: Tensor | None = None
        if self.arm_cfg.queryable_goal_free_temporal:
            local_h, goal_free_next_hidden, memory_count = self.goal_free_temporal_encode(
                obs, legacy_hidden, active_leg
            )
            history = obs.new_zeros(batch, n_agents, 0, self.arm_cfg.stalk_dim)
            history_mask = obs.new_zeros(batch, n_agents, 0)
        else:
            rec_hidden, history, history_mask = self.split_hidden(legacy_hidden, obs)
            encoded = self.obs_encoder(learned_obs)
            local_h = self.gru(
                encoded.reshape(batch * n_agents, -1),
                rec_hidden.reshape(batch * n_agents, -1),
            ).view(batch, n_agents, -1)
            memory_count = history_mask.sum(dim=-1)
        mu = self.mean_head(local_h)
        local_precision = F.softplus(self.precision_head(local_h)) + 1e-3
        precision = local_precision
        natural = precision * mu
        temporal_precision, temporal_natural = self.temporal_unary(history, history_mask)
        precision = precision + temporal_precision
        natural = natural + temporal_natural
        if self.arm_cfg.local_se2_distributed_wrench_control:
            # The legacy 32->8 source Sheaf is not on this controller's action
            # path.  Avoid even constructing its global-handle restriction:
            # the only inter-agent solve below is the 3-D correction Sheaf.
            node_r = None
            directed_r = None
        else:
            node_r = self.restrictions(local_h)
            if self.arm_cfg.edge_conditioned_restrictions:
                directed_r = self.edge_conditioned_directed_restrictions(
                    node_r, local_h, edge_index
                )
            else:
                directed_r = self.directed_restrictions(node_r, edge_index)
        if topology is None:
            topology = homogeneous_topology(
                edge_index,
                obs.shape[0],
                self.env_cfg.n_agents,
                device=obs.device,
            )
        correction_result = None
        correction_section: Tensor | None = None
        correction_wrench: Tensor | None = None
        correction_force: Tensor | None = None
        distributed_grasp_stats: Tensor | None = None
        distributed_grasp_message_trace: Tensor | None = None
        distributed_grasp_setup_transmitted = False
        correction_public_unary_delta: Tensor | None = None
        correction_public_unary_mean_pre: Tensor | None = None
        correction_public_unary_mean_post: Tensor | None = None
        correction_source_null_risk: Tensor | None = None
        correction_source_null_delta: Tensor | None = None
        correction_source_null_mean_pre: Tensor | None = None
        correction_source_null_mean_post: Tensor | None = None
        collision_role_logits: Tensor | None = None
        collision_role_weights: Tensor | None = None
        collision_role_mean_delta: Tensor | None = None
        collision_role_log_precision_delta: Tensor | None = None
        collision_role_follower_edge_weight: Tensor | None = None
        collision_role_edge_sigma2: Tensor | None = None
        collision_role_node_restriction: Tensor | None = None
        local_se2_plan_mean: Tensor | None = None
        local_se2_plan_precision: Tensor | None = None
        local_se2_plan_used: Tensor | None = None
        local_se2_plan_hidden_delta: Tensor | None = None
        local_se2_plan_condition: Tensor | None = None
        local_se2_direct_wrench_pre: Tensor | None = None
        local_se2_direct_wrench_delta: Tensor | None = None
        local_se2_direct_wrench_post: Tensor | None = None
        local_se2_analytic_wrench_proposal: Tensor | None = None
        local_se2_terminal_plan: Tensor | None = None
        local_se2_terminal_active: Tensor | None = None
        local_se2_recovery_mode_logits: Tensor | None = None
        local_se2_recovery_selected_mode: Tensor | None = None
        local_se2_distributed_recovery_pre_mean: Tensor | None = None
        if self.arm_cfg.correction_sheaf:
            # Canonical local observations plus frozen pre-message source
            # unary statistics are the complete correction input.  Detaching
            # the source unary makes the causal/frozen-source boundary true
            # even before a trainer applies set_correction_only_trainable().
            correction_inputs = torch.cat(
                (
                    learned_obs,
                    mu.detach(),
                    local_precision.detach().log(),
                ),
                dim=-1,
            )
            correction_h = self.correction_local_encoder(correction_inputs)
            if (
                self.arm_cfg.persistent_recovery_state
                and self.arm_cfg.persistent_recovery_injection == "shared_hidden"
            ):
                if recovery_state is None or recovery_active is None:
                    raise RuntimeError("persistent recovery update omitted its state")
                if self.persistent_recovery_fusion is None:
                    raise RuntimeError("shared-hidden recovery fusion is missing")
                proposed_correction_h = correction_h + self.persistent_recovery_fusion(
                    recovery_state
                )
                # This explicit selection, rather than an unconditional +0,
                # is the source-parity contract for all never-hit rows.
                correction_h = torch.where(
                    recovery_active,
                    proposed_correction_h,
                    correction_h,
                )
            if self.arm_cfg.local_se2_planner:
                if (
                    recovery_state is None
                    or recovery_active is None
                    or recovery_phase_logits is None
                ):
                    raise RuntimeError("local SE(2) planner requires recovery state")
                (
                    local_se2_plan_mean,
                    local_se2_plan_precision,
                    plan_log_precision,
                    local_se2_plan_base_mean,
                    local_se2_recovery_plan_delta,
                    local_se2_recovery_mode_logits,
                    local_se2_recovery_selected_mode,
                    local_se2_distributed_recovery_pre_mean,
                ) = self._local_se2_plan_from_features(
                    learned_obs,
                    local_h,
                    recovery_state,
                    recovery_active,
                    recovery_phase_logits,
                    approach_streak,
                )
                if local_se2_plan_override is None:
                    local_se2_plan_used = local_se2_plan_mean
                else:
                    if local_se2_plan_override.shape != (
                        batch,
                        n_agents,
                        self.arm_cfg.local_se2_plan_dim,
                    ):
                        raise ValueError("local SE(2) plan override has the wrong shape")
                    if not bool(torch.isfinite(local_se2_plan_override).all()):
                        raise ValueError("local SE(2) plan override must be finite")
                    local_se2_plan_used = local_se2_plan_override
                if self.arm_cfg.local_se2_terminal_attractor:
                    local_se2_plan_used, local_se2_terminal_active = (
                        local_se2_terminal_attractor_plan(
                            obs,
                            local_se2_plan_used,
                            self.env_cfg,
                            position_gain=(
                                self.arm_cfg.local_se2_terminal_position_gain
                            ),
                            angle_gain=self.arm_cfg.local_se2_terminal_angle_gain,
                            max_speed=self.arm_cfg.local_se2_terminal_max_speed,
                            max_omega=self.arm_cfg.local_se2_terminal_max_omega,
                            feasible_corridor=(
                                self.arm_cfg.local_se2_feasible_corridor_attractor
                            ),
                            route_free_geometry=(
                                self.arm_cfg.local_se2_route_free_geometry_planner
                            ),
                            corridor_reserve_m=(
                                self.arm_cfg.local_se2_corridor_reserve_m
                            ),
                            corridor_max_speed=(
                                self.arm_cfg.local_se2_corridor_max_speed
                            ),
                        )
                    )
                    local_se2_terminal_plan = local_se2_plan_used
                if final_local_se2_plan_override is not None:
                    # An independently trained node-local planner must reach
                    # the physical action layer without an analytic plan
                    # silently replacing its supervised output. Historical
                    # callers omit this argument and retain exact behavior.
                    if final_local_se2_plan_override.shape != (batch, n_agents, 8):
                        raise ValueError("final local plan override must be [B,N,8]")
                    if not bool(torch.isfinite(final_local_se2_plan_override).all()):
                        raise ValueError("final local plan override must be finite")
                    local_se2_plan_used = final_local_se2_plan_override
                    local_se2_terminal_active = torch.zeros(
                        (batch,n_agents,1),device=obs.device,dtype=torch.bool
                    )
                plan_condition = torch.cat(
                    (local_se2_plan_used, torch.tanh(plan_log_precision)),
                    dim=-1,
                )
                local_se2_plan_condition = plan_condition
                local_se2_plan_hidden_delta = self.correction_plan_fusion(
                    plan_condition
                )
                proposed_plan_h = correction_h + local_se2_plan_hidden_delta
                selected_plan_h = torch.where(
                    local_se2_plan_hidden_delta != 0.0,
                    proposed_plan_h,
                    correction_h,
                )
                correction_h = _forward_exact_backward_proposed(
                    selected_plan_h,
                    proposed_plan_h,
                )
            correction_mu = self.correction_mean_head(correction_h)
            if self.arm_cfg.collision_role_factors:
                role_count = self.arm_cfg.collision_role_count
                stalk_dim = self.arm_cfg.correction_stalk_dim
                role_h = correction_h
                if self.arm_cfg.collision_role_separate_encoder:
                    if self.correction_role_encoder is None:
                        raise RuntimeError("separate role encoder is missing")
                    role_h = self.correction_role_encoder(correction_inputs)
                    if (
                        recovery_state is None
                        or recovery_active is None
                        or self.persistent_recovery_fusion is None
                    ):
                        raise RuntimeError(
                            "separate role encoder requires shared-hidden recovery"
                        )
                    role_h_with_memory = role_h + self.persistent_recovery_fusion(
                        recovery_state
                    )
                    role_h = torch.where(recovery_active, role_h_with_memory, role_h)
                collision_role_logits = self.correction_role_logits_head(role_h)
                raw_role_weights = torch.sigmoid(collision_role_logits)
                if recovery_active is None:
                    raise RuntimeError("collision roles require persistent recovery state")
                # Leader and follower may coexist before contact.  Rebel is
                # a causal contact-history factor: its learned logit cannot
                # alter the policy until this local row has actually observed
                # a rejected proposal.  This is an input-support contract,
                # not a confidence/action gate.
                collision_role_weights = torch.cat(
                    (
                        raw_role_weights[..., :2],
                        torch.where(
                            recovery_active,
                            raw_role_weights[..., 2:3],
                            torch.zeros_like(raw_role_weights[..., 2:3]),
                        ),
                    ),
                    dim=-1,
                )
                role_mean = self.correction_role_mean_head(role_h).view(
                    batch,
                    n_agents,
                    role_count,
                    stalk_dim,
                )
                role_log_precision = self.correction_role_log_precision_head(
                    role_h
                ).view(
                    batch,
                    n_agents,
                    role_count,
                    stalk_dim,
                )
                if self.arm_cfg.collision_role_precontact_leader_follower:
                    # Curriculum policy: ordinary transport is already a
                    # leader/follower factorization.  Rebel remains exactly
                    # zero before observed contact by construction above.
                    factor_role_weight = collision_role_weights
                else:
                    # Historical source-correct recovery bridge: all learned
                    # role interventions begin only after contact history.
                    factor_role_weight = (
                        collision_role_weights
                        * recovery_active.to(dtype=collision_role_weights.dtype)
                    )
                role_weight = factor_role_weight.unsqueeze(-1)
                collision_role_mean_delta = (
                    self.arm_cfg.collision_role_mean_scale
                    * (role_weight * torch.tanh(role_mean)).sum(dim=-2)
                )
                collision_role_log_precision_delta = (
                    self.arm_cfg.collision_role_log_precision_scale
                    * (role_weight * torch.tanh(role_log_precision)).sum(dim=-2)
                    / float(role_count)
                )
                # Select the untouched tensor on the exact-zero bridge.  This
                # is stronger than merely adding +0 and preserves every
                # historical unary bit before role training.
                proposed_role_mu = correction_mu + collision_role_mean_delta
                selected_role_mu = torch.where(
                    collision_role_mean_delta != 0.0,
                    proposed_role_mu,
                    correction_mu,
                )
                correction_mu = _forward_exact_backward_proposed(
                    selected_role_mu,
                    proposed_role_mu,
                )
            if self.arm_cfg.correction_public_unary_residual:
                correction_public_unary_mean_pre = correction_mu[..., :_CORRECTION_WRENCH_DIM]
                correction_public_unary_delta = (
                    self.arm_cfg.correction_public_unary_residual_scale
                    * torch.tanh(self.correction_public_unary_residual_head(correction_h))
                )
                correction_public_unary_mean_post = (
                    correction_public_unary_mean_pre + correction_public_unary_delta
                )
                # Do not mutate correction_mu in place: the pre-residual
                # public mean is an auditable tensor and autograd must retain
                # both sides of this unary intervention exactly.
                correction_mu = torch.cat(
                    (
                        correction_public_unary_mean_post,
                        correction_mu[..., _CORRECTION_WRENCH_DIM:],
                    ),
                    dim=-1,
                )
            if self.arm_cfg.correction_source_null_recovery:
                correction_source_null_mean_pre = correction_mu[
                    ..., :_CORRECTION_WRENCH_DIM
                ]
                correction_source_null_risk = local_slit_source_null_risk(obs).unsqueeze(-1)
                proposed_delta = (
                    0.25
                    * correction_source_null_risk
                    * torch.tanh(self.correction_source_null_recovery_head(correction_h))
                )
                source_null = correction_source_null_risk == 0.0
                correction_source_null_delta = torch.where(
                    source_null,
                    torch.zeros_like(proposed_delta),
                    proposed_delta,
                )
                proposed_mean_post = (
                    correction_source_null_mean_pre + correction_source_null_delta
                )
                # Selecting the original tensor on the null set is stronger
                # than merely adding +0: it preserves signed zero and every
                # source public-unary bit for arbitrary recovery-head weights.
                correction_source_null_mean_post = torch.where(
                    source_null,
                    correction_source_null_mean_pre,
                    proposed_mean_post,
                )
                correction_mu = torch.cat(
                    (
                        correction_source_null_mean_post,
                        correction_mu[..., _CORRECTION_WRENCH_DIM:],
                    ),
                    dim=-1,
                )
            if (
                self.arm_cfg.persistent_recovery_state
                and self.arm_cfg.persistent_recovery_injection == "correction_mean"
            ):
                if recovery_state is None or recovery_active is None:
                    raise RuntimeError("persistent recovery update omitted its state")
                if self.persistent_recovery_mean_residual is None:
                    raise RuntimeError("correction-mean recovery residual is missing")
                proposed_correction_mu = (
                    correction_mu
                    + self.persistent_recovery_mean_residual(recovery_state)
                )
                # Select the original tensor on never-hit rows.  This is the
                # algebraic source-parity contract even after the residual is
                # trained away from zero.
                correction_mu = torch.where(
                    recovery_active,
                    proposed_correction_mu,
                    correction_mu,
                )
            if self.arm_cfg.local_se2_direct_wrench_unary:
                if local_se2_plan_condition is None:
                    raise RuntimeError("direct wrench unary omitted its plan condition")
                local_se2_direct_wrench_pre = correction_mu[
                    ..., :_CORRECTION_WRENCH_DIM
                ]
                raw_direct_wrench_delta = self.local_se2_direct_wrench_unary_head(
                    torch.cat((correction_h, local_se2_plan_condition), dim=-1)
                )
                # Only agents whose radius-limited sensor currently sees a
                # wall may modify the public wrench.  This activates planning
                # before the much narrower 32 mm contact-clearance envelope;
                # invisible agents remain exact zero and receive the local
                # evidence only through GBP packets.
                direct_wrench_support = torch.clamp(
                    obs[..., 16] + obs[..., 17], 0.0, 1.0
                ).unsqueeze(-1)
                local_se2_direct_wrench_delta = (
                    direct_wrench_support * raw_direct_wrench_delta
                )
                proposed_direct_wrench_post = (
                    local_se2_direct_wrench_pre + local_se2_direct_wrench_delta
                )
                selected_direct_wrench_post = torch.where(
                    local_se2_direct_wrench_delta != 0.0,
                    proposed_direct_wrench_post,
                    local_se2_direct_wrench_pre,
                )
                local_se2_direct_wrench_post = _forward_exact_backward_proposed(
                    selected_direct_wrench_post,
                    proposed_direct_wrench_post,
                )
                correction_mu = torch.cat(
                    (
                        local_se2_direct_wrench_post,
                        correction_mu[..., _CORRECTION_WRENCH_DIM:],
                    ),
                    dim=-1,
                )
            if local_se2_wrench_unary_override is not None:
                expected_override = (batch, n_agents, _CORRECTION_WRENCH_DIM)
                if local_se2_wrench_unary_override.shape != expected_override:
                    raise ValueError(
                        "local SE(2) wrench-unary override has the wrong shape"
                    )
                if not bool(torch.isfinite(local_se2_wrench_unary_override).all()):
                    raise ValueError("local SE(2) wrench-unary override must be finite")
                correction_mu = torch.cat(
                    (
                        local_se2_wrench_unary_override,
                        correction_mu[..., _CORRECTION_WRENCH_DIM:],
                    ),
                    dim=-1,
                )
            if self.arm_cfg.local_se2_analytic_total_wrench:
                if local_se2_plan_used is None:
                    raise RuntimeError("analytic total wrench omitted its local SE(2) plan")
                if local_se2_wrench_unary_override is not None:
                    raise ValueError(
                        "analytic total-wrench mode does not accept a wrench-unary override"
                    )
                if (
                    self.arm_cfg.local_se2_distributed_wrench_control
                    and not self.arm_cfg.local_se2_distributed_next_se2_control
                ):
                    # The recurrent encoder assigns the first three private
                    # plan coordinates the public normalized-wrench meaning.
                    # This proposal reads only one node's local input; the
                    # global belief can arise only through GBP below.
                    local_se2_analytic_wrench_proposal = (
                        local_se2_plan_used[..., :_CORRECTION_WRENCH_DIM]
                    )
                else:
                    # In the physical planner variant all eight coordinates
                    # retain their next-SE(2) meaning.  Each node independently
                    # maps its desired next twist and own measured handle
                    # velocity to a total-wrench sufficient statistic.  GBP
                    # shares only the resulting three physical coordinates.
                    local_se2_analytic_wrench_proposal = (
                        local_se2_inverse_dynamics_wrench_proposals(
                            obs,
                            local_se2_plan_used,
                            self.env_cfg,
                        )
                    )
                correction_mu = torch.cat(
                    (
                        local_se2_analytic_wrench_proposal,
                        correction_mu[..., _CORRECTION_WRENCH_DIM:],
                    ),
                    dim=-1,
                )
            correction_precision = F.softplus(self.correction_precision_head(correction_h)) + 1e-3
            if self.arm_cfg.collision_role_factors:
                if collision_role_log_precision_delta is None:
                    raise RuntimeError("collision role precision delta is missing")
                proposed_role_precision = correction_precision * torch.exp(
                    collision_role_log_precision_delta
                )
                selected_role_precision = torch.where(
                    collision_role_log_precision_delta != 0.0,
                    proposed_role_precision,
                    correction_precision,
                )
                correction_precision = _forward_exact_backward_proposed(
                    selected_role_precision,
                    proposed_role_precision,
                )
            if self.arm_cfg.local_se2_analytic_total_wrench:
                # Role confidence may still be audited, but the physical
                # sufficient statistics receive a common known precision.
                public_precision = torch.full_like(
                    correction_precision[..., :_CORRECTION_WRENCH_DIM],
                    self.arm_cfg.local_se2_analytic_unary_precision,
                )
                evidence_boost = (
                    self.arm_cfg.local_se2_geometry_evidence_precision_boost
                )
                if (
                    evidence_boost > 1.0
                    and not (
                        normalize_payload_shape(self.env_cfg.payload_shape) == "T"
                        and t_morphology_from_config(self.env_cfg).is_canonical
                        and not self.arm_cfg.local_se2_route_free_geometry_planner
                    )
                ):
                    if local_se2_terminal_active is None:
                        raise RuntimeError(
                            "geometry precision boost requires a local geometric plan"
                        )
                    public_precision = public_precision * (
                        1.0
                        + (evidence_boost - 1.0)
                        * local_se2_terminal_active.to(public_precision.dtype)
                    )
                correction_precision = torch.cat(
                    (
                        public_precision,
                        correction_precision[..., _CORRECTION_WRENCH_DIM:],
                    ),
                    dim=-1,
                )
            correction_natural = correction_precision * correction_mu
            correction_node_r = (
                self.correction_analytic_restrictions(correction_h)
                if self.arm_cfg.local_se2_analytic_total_wrench
                else self.correction_restrictions(correction_h)
            )
            if self.arm_cfg.collision_role_factors:
                collision_role_node_restriction = correction_node_r
            correction_directed_r = self.correction_directed_restrictions(
                correction_node_r,
                edge_index,
            )
            correction_sigma2 = obs.new_full(
                (batch, 2 * edge_index.shape[0], self.arm_cfg.correction_edge_dim),
                (
                    self.arm_cfg.local_se2_analytic_edge_sigma2
                    if self.arm_cfg.local_se2_analytic_total_wrench
                    else self.arm_cfg.correction_edge_sigma2
                ),
            )
            if self.arm_cfg.local_se2_analytic_total_wrench:
                if not torch.equal(edge_index, self.local_se2_analytic_reference_edges):
                    raise ValueError(
                        "analytic total-wrench checkpoint received a different radius graph"
                    )
                directed_tree = self.local_se2_analytic_tree_edge_mask.repeat_interleave(2)
                correction_sigma2 = torch.where(
                    directed_tree.view(1, -1, 1),
                    correction_sigma2,
                    torch.full_like(correction_sigma2, 1.0e6),
                )
            if self.arm_cfg.collision_role_factors:
                if collision_role_weights is None:
                    raise RuntimeError("collision role weights are missing")
                # Follower is role index one.  Pairwise compatibility is
                # symmetric: both endpoints vote, then the same factor noise
                # is used by the two directed incidences of that edge.  At
                # the zero bridge both votes are exactly 0.5 and sigma2 is
                # selected bitwise from the legacy tensor.
                if self.arm_cfg.collision_role_precontact_leader_follower:
                    neutral_follower = collision_role_weights[..., 1]
                else:
                    neutral_follower = torch.where(
                        recovery_active.squeeze(-1),
                        collision_role_weights[..., 1],
                        torch.full_like(collision_role_weights[..., 1], 0.5),
                    )
                endpoint_weight = neutral_follower.index_select(
                    1, edge_index.reshape(-1).long()
                ).view(batch, edge_index.shape[0], 2)
                collision_role_follower_edge_weight = endpoint_weight.mean(
                    dim=-1, keepdim=True
                )
                edge_log_sigma2_delta = (
                    -self.arm_cfg.collision_role_edge_log_precision_scale
                    * (2.0 * collision_role_follower_edge_weight - 1.0)
                )
                edge_log_sigma2_delta = edge_log_sigma2_delta.repeat_interleave(
                    2, dim=1
                )
                if not self.arm_cfg.local_se2_analytic_total_wrench:
                    proposed_role_sigma2 = correction_sigma2 * torch.exp(
                        edge_log_sigma2_delta
                    )
                    selected_role_sigma2 = torch.where(
                        edge_log_sigma2_delta != 0.0,
                        proposed_role_sigma2,
                        correction_sigma2,
                    )
                    correction_sigma2 = _forward_exact_backward_proposed(
                        selected_role_sigma2,
                        proposed_role_sigma2,
                    )
                collision_role_edge_sigma2 = correction_sigma2
            correction_solver = (
                self.local_se2_analytic_gbp
                if self.arm_cfg.local_se2_analytic_total_wrench
                else self.correction_gbp
            )
            if correction_solver is None:
                raise RuntimeError("analytic total-wrench GBP solver is missing")
            correction_result = correction_solver(
                correction_precision,
                correction_natural,
                correction_directed_r,
                correction_sigma2,
                topology,
                trace_messages=trace_messages,
            )
            correction_section = torch.einsum(
                "bnev,bnv->bne",
                correction_node_r,
                correction_result.mean,
            )
            if self.arm_cfg.local_se2_analytic_total_wrench:
                # The section already is a normalized *total* wrench.  A tanh
                # here would destroy the linear sufficient-statistic identity.
                correction_wrench = correction_section
            else:
                correction_wrench = self.arm_cfg.correction_wrench_scale * torch.tanh(
                    correction_section
                )
            correction_phase_basis = None
            correction_phase_wrench = None
            if (
                self.arm_cfg.correction_phase_decoder
                and not self.arm_cfg.local_se2_analytic_total_wrench
            ):
                phase_inputs, correction_phase_basis = local_slit_phase_features(
                    obs, correction_section.detach(), self.env_cfg
                )
                correction_phase_wrench = (
                    self.arm_cfg.correction_phase_wrench_scale
                    * torch.tanh(self.correction_phase_wrench_head(phase_inputs))
                )
                correction_wrench = correction_wrench + correction_phase_wrench
            correction_clearance_bank_basis = None
            correction_clearance_bank_wrench = None
            if (
                self.arm_cfg.correction_clearance_bank_decoder
                and not self.arm_cfg.local_se2_analytic_total_wrench
            ):
                bank_inputs,correction_clearance_bank_basis=local_slit_clearance_bank_features(
                    obs,correction_section.detach(),self.env_cfg
                )
                correction_clearance_bank_wrench=(
                    self.arm_cfg.correction_clearance_bank_wrench_scale
                    * torch.tanh(self.correction_phase_clearance_wrench_head(bank_inputs))
                )
                correction_wrench=correction_wrench+correction_clearance_bank_wrench
            correction_approach_bank_basis=None
            correction_approach_bank_wrench=None
            if (
                self.arm_cfg.correction_approach_bank_decoder
                and not self.arm_cfg.local_se2_analytic_total_wrench
            ):
                approach_inputs,correction_approach_bank_basis=local_slit_approach_bank_features(
                    obs,correction_section.detach(),self.env_cfg
                )
                correction_approach_bank_wrench=(self.arm_cfg.correction_approach_bank_wrench_scale*torch.tanh(self.correction_phase_approach_wrench_head(approach_inputs)))
                correction_wrench=correction_wrench+correction_approach_bank_wrench
            if self.arm_cfg.local_se2_distributed_wrench_control:
                if local_se2_plan_used is None:
                    raise RuntimeError("distributed force decode omitted its private plan")
                if self.arm_cfg.local_se2_distributed_grasp_allreduce:
                    (
                        correction_force,
                        distributed_grasp_stats,
                        distributed_grasp_message_trace,
                        distributed_grasp_setup_transmitted,
                    ) = self.distributed_force_from_local_grasp_statistics(
                        correction_section,
                        obs,
                        edge_index,
                        trace_messages=trace_messages,
                    )
                else:
                    distributed_force_input = torch.cat(
                        (learned_obs, correction_section, local_se2_plan_used),
                        dim=-1,
                    )
                    distributed_normalized_force = torch.tanh(
                        self.local_se2_distributed_force_decoder(
                            distributed_force_input
                        )
                    )
                    if self.arm_cfg.local_se2_distributed_recovery_adapter:
                        if recovery_state is None or recovery_active is None:
                            raise RuntimeError(
                                "distributed recovery force omitted local memory"
                            )
                        recovery_force_input = torch.cat(
                            (
                                distributed_force_input,
                                recovery_state.detach(),
                            ),
                            dim=-1,
                        )
                        distributed_recovery_force_delta = (
                            self.arm_cfg.local_se2_distributed_recovery_scale
                            * torch.tanh(
                                self.local_se2_distributed_recovery_force_decoder(
                                    recovery_force_input
                                )
                            )
                        )
                        distributed_normalized_force = torch.where(
                            recovery_active,
                            distributed_normalized_force
                            + distributed_recovery_force_delta,
                            distributed_normalized_force,
                        )
                    correction_force = (
                        self.env_cfg.max_force * distributed_normalized_force
                    )
            else:
                # Historical analytic-total-wrench checkpoints allocate the
                # section through the complete global grasp pseudoinverse.
                correction_force = self.exact_return_force_from_wrench(
                    correction_wrench,
                    obs,
                )
        if self.arm_cfg.local_se2_distributed_wrench_control:
            # Preserve a local recurrent suffix without running or pretending
            # to transmit the dead source GBP.  These tensors are diagnostics
            # only; the correction result above is the sole shared belief.
            directed = 2 * edge_index.shape[0]
            result = GBPResult(
                mean=mu,
                precision=precision,
                omega=obs.new_zeros(batch, directed, self.arm_cfg.edge_dim),
                xi=obs.new_zeros(batch, directed, self.arm_cfg.edge_dim),
                message_trace=None,
            )
            section = obs.new_zeros(batch, n_agents, self.arm_cfg.edge_dim)
            analytic_force = obs.new_zeros(batch, n_agents, 2)
            decoder_features = learned_obs
            residual_force = obs.new_zeros(batch, n_agents, 2)
            base_forces = residual_force
            source_forces = residual_force
        else:
            if node_r is None or directed_r is None:
                raise RuntimeError("source Sheaf restrictions are missing")
            sigma2 = obs.new_full(
                (obs.shape[0], 2 * edge_index.shape[0], self.arm_cfg.edge_dim),
                self.arm_cfg.edge_sigma2,
            )
            result = self.gbp(
                precision,
                natural,
                directed_r,
                sigma2,
                topology,
                trace_messages=trace_messages,
            )
            section = torch.einsum("bnev,bnv->bne", node_r, result.mean)
            analytic_force = self.analytic_force_from_section(section, obs)
            decoder_inputs = [learned_obs, section]
            if not self.arm_cfg.decoder_section_only:
                decoder_inputs.insert(1, result.mean)
            if self.arm_cfg.decoder_extra_context:
                memory_scale = float(max(1, self.arm_cfg.temporal_window))
                decoder_inputs.append(analytic_force / self.env_cfg.max_force)
                decoder_inputs.append((memory_count / memory_scale).unsqueeze(-1))
            decoder_features = torch.cat(decoder_inputs, dim=-1)
            raw_residual = self.decoder(decoder_features)
            residual_force = (
                self.arm_cfg.force_residual_scale
                * self.env_cfg.max_force
                * torch.tanh(raw_residual)
            )
            base_forces = self.clamp_force_norm(
                self.arm_cfg.analytic_force_scale * analytic_force + residual_force
            )
            # Deployment applies a final per-handle clamp after the return branch.
            source_forces = self.clamp_force_norm(base_forces)
        return_residual = torch.zeros_like(base_forces)
        return_wrench_residual = obs.new_zeros((batch, n_agents, 3))
        return_gate = obs.new_zeros((batch, 1, 1))
        return_mode_aux: dict[str, Tensor] = {}
        source_candidate_exact_action = False
        if self.return_adapter is not None:
            if isinstance(self.return_adapter, ReturnSourceCandidateAdapter):
                selector_features = return_selector_invariant_features(
                    obs,
                    memory_count,
                    self.env_cfg,
                    temporal_window=self.arm_cfg.temporal_window,
                )
                (
                    return_source_advantage_score,
                    return_source_expert_wrench,
                ) = self.return_adapter(
                    decoder_features,
                    selector_features,
                    section,
                    expert_wrench_scale=self.arm_cfg.return_adapter_scale,
                    obs=obs,
                    memory_count=memory_count,
                    source_forces=source_forces,
                    env_cfg=self.env_cfg,
                    temporal_window=self.arm_cfg.temporal_window,
                )
                direct_v3 = self.return_adapter.architecture == RETURN_CANDIDATE_ARCHITECTURE_V3
                if direct_v3:
                    # V3 has no learned confidence selector.  Its candidate is
                    # the continuous return residual at every node; the known
                    # active-leg mask below remains the only deployment mask.
                    if return_source_advantage_score is not None:
                        raise RuntimeError("V3 unexpectedly produced a selector score")
                    return_selected_candidate = torch.ones(
                        batch,
                        n_agents,
                        dtype=torch.long,
                        device=obs.device,
                    )
                    return_selected_wrench = return_source_expert_wrench
                else:
                    if return_source_advantage_score is None:
                        raise RuntimeError("V1/V2 source candidate omitted its selector score")
                    # Candidate 0 is fixed at score zero and exact zero wrench.
                    # Strict comparison deliberately sends ties to the source.
                    return_selected_candidate = (return_source_advantage_score > 0.0).long()
                    return_selected_wrench = torch.where(
                        return_selected_candidate[..., None].bool(),
                        return_source_expert_wrench,
                        torch.zeros_like(return_source_expert_wrench),
                    )
                source_candidate_exact_action = self.return_adapter.architecture in {
                    RETURN_CANDIDATE_ARCHITECTURE_V2,
                    RETURN_CANDIDATE_ARCHITECTURE_V3,
                }
                return_wrench_residual = return_selected_wrench
                return_mode_aux = {
                    "return_source_expert_wrench": return_source_expert_wrench,
                    "return_selected_candidate": return_selected_candidate,
                    "return_selected_wrench": return_selected_wrench,
                }
                if return_source_advantage_score is not None:
                    return_mode_aux["return_source_advantage_score"] = return_source_advantage_score
            elif isinstance(self.return_adapter, ReturnManeuverAdapter):
                return_mode_scores, raw_mode_wrenches = self.return_adapter(decoder_features)
                return_mode_wrenches = self.arm_cfg.return_adapter_scale * torch.tanh(
                    raw_mode_wrenches
                )
                return_selected_mode = return_mode_scores.argmax(dim=-1)
                gather_index = return_selected_mode[..., None, None].expand(
                    batch,
                    n_agents,
                    1,
                    3,
                )
                return_selected_wrench = torch.gather(
                    return_mode_wrenches,
                    dim=2,
                    index=gather_index,
                ).squeeze(2)
                return_wrench_residual = return_selected_wrench
                return_mode_aux = {
                    "return_mode_scores": return_mode_scores,
                    "return_mode_wrenches": return_mode_wrenches,
                    "return_selected_mode": return_selected_mode,
                    "return_selected_wrench": return_selected_wrench,
                }
            else:
                raw_return = self.return_adapter(decoder_features)
                return_wrench_residual = self.arm_cfg.return_adapter_scale * torch.tanh(raw_return)
            if self.arm_cfg.exact_return_grasp_allocator:
                return_residual = self.exact_return_force_from_wrench(
                    return_wrench_residual,
                    obs,
                )
            else:
                # Compatibility path for the already-trained return adapter:
                # interpret its three normalized outputs as section wrench
                # coordinates and use the original approximate allocator.
                residual_section = obs.new_zeros(
                    batch,
                    n_agents,
                    max(6, self.arm_cfg.edge_dim),
                )
                residual_section[..., 3:6] = return_wrench_residual
                return_residual = self.analytic_force_from_section(
                    residual_section,
                    obs,
                )
            if active_leg is not None:
                return_gate = active_leg.to(obs.dtype).view(batch, 1, 1).clamp(0.0, 1.0)
        if source_candidate_exact_action:
            # Candidate 1 has one authoritative executable action.  It starts
            # from the exact deployed source action, adds the unselected raw
            # candidate residual, and receives one final clamp.  Selection is
            # then a literal per-handle choice between two already executable
            # actions, making every source-selected handle bitwise exact.
            expert_residual = self.exact_return_force_from_wrench(
                return_source_expert_wrench,
                obs,
            )
            expert_raw_forces = source_forces + expert_residual
            expert_executable_forces = self.clamp_force_norm(expert_raw_forces)
            expert_selected = return_selected_candidate.bool() & return_gate.bool().expand(
                -1,
                n_agents,
                -1,
            ).squeeze(-1)
            forces = torch.where(
                expert_selected[..., None],
                expert_executable_forces,
                source_forces,
            )
            return_mode_aux.update(
                {
                    "return_source_expert_raw_forces": expert_raw_forces,
                    "return_source_expert_forces": expert_executable_forces,
                }
            )
        else:
            # Historical V1 and non-binary return adapters keep their original
            # arithmetic exactly so existing checkpoints remain reproducible.
            if self.arm_cfg.correction_sheaf:
                if correction_force is None:
                    raise RuntimeError("enabled correction sheaf omitted its force")
                if self.arm_cfg.local_se2_analytic_total_wrench:
                    # This branch represents the complete inverse-dynamics
                    # command, not a residual around the legacy action.
                    forces = self.clamp_force_norm(correction_force)
                else:
                    forces = self.clamp_force_norm(
                        base_forces + return_gate * return_residual + correction_force
                    )
            else:
                forces = self.clamp_force_norm(base_forces + return_gate * return_residual)
        next_hidden = (
            goal_free_next_hidden
            if goal_free_next_hidden is not None
            else self.pack_hidden(local_h, history, history_mask, result.mean)
        )
        if self.arm_cfg.persistent_recovery_state:
            if recovery_state is None:
                raise RuntimeError("persistent recovery hidden suffix was not produced")
            next_hidden = torch.cat((next_hidden, recovery_state), dim=-1)
        if self.arm_cfg.local_se2_approach_refinement_adapter:
            if approach_streak is None:
                raise RuntimeError("approach refinement hidden suffix was not produced")
            next_hidden = torch.cat((next_hidden, approach_streak), dim=-1)
        aux = {
            "section": section,
            "analytic_force": analytic_force,
            # The frozen source action before any return-only correction.  It
            # is exposed for training-time source-relative teachers and
            # diagnostics; deployment still sends only the GBP edge message.
            "base_forces": base_forces,
            # Exact final-clamp action of the frozen source controller.  This
            # is the sole source anchor for shadow collection and matched-Q.
            SOURCE_FORCES_AUX_KEY: source_forces,
            "stalk_mean": result.mean,
            "stalk_precision": result.precision,
            "temporal_memory_count": memory_count,
            "return_residual": return_residual,
            "return_wrench_residual": return_wrench_residual,
            "return_gate": return_gate.expand(-1, n_agents, -1),
        }
        aux.update(return_mode_aux)
        if self.arm_cfg.local_se2_distributed_grasp_allreduce:
            if distributed_grasp_stats is None:
                raise RuntimeError("distributed grasp all-reduce omitted statistics")
            aux[LOCAL_SE2_DISTRIBUTED_GRASP_STATS_AUX_KEY] = (
                distributed_grasp_stats
            )
            aux[LOCAL_SE2_DISTRIBUTED_GRASP_SETUP_TRANSMITTED_AUX_KEY] = (
                torch.tensor(
                    distributed_grasp_setup_transmitted,
                    dtype=torch.bool,
                    device=distributed_grasp_stats.device,
                )
            )
            if trace_messages:
                if distributed_grasp_message_trace is None:
                    raise RuntimeError("distributed grasp all-reduce omitted wire trace")
                aux[LOCAL_SE2_DISTRIBUTED_GRASP_MESSAGE_AUX_KEY] = (
                    distributed_grasp_message_trace
                )
        if self.arm_cfg.persistent_recovery_state:
            if (
                recovery_state is None
                or recovery_active is None
                or recovery_phase_logits is None
            ):
                raise RuntimeError("persistent recovery omitted its auxiliary tensors")
            aux.update(
                {
                    PERSISTENT_RECOVERY_STATE_AUX_KEY: recovery_state,
                    PERSISTENT_RECOVERY_ACTIVE_AUX_KEY: recovery_active,
                    PERSISTENT_RECOVERY_PHASE_LOGITS_AUX_KEY: recovery_phase_logits,
                }
            )
            if self.arm_cfg.local_se2_distributed_recovery_mode_count > 1:
                if (
                    local_se2_recovery_mode_logits is None
                    or local_se2_recovery_selected_mode is None
                ):
                    raise RuntimeError("maneuver experts omitted audit tensors")
                aux.update(
                    {
                        LOCAL_SE2_RECOVERY_MODE_LOGITS_AUX_KEY: (
                            local_se2_recovery_mode_logits
                        ),
                        LOCAL_SE2_RECOVERY_SELECTED_MODE_AUX_KEY: (
                            local_se2_recovery_selected_mode
                        ),
                    }
                )
            if self.arm_cfg.local_se2_distributed_recovery_adapter:
                if local_se2_distributed_recovery_pre_mean is None:
                    raise RuntimeError("distributed recovery omitted its pre-plan")
                aux[LOCAL_SE2_DISTRIBUTED_RECOVERY_PRE_MEAN_AUX_KEY] = (
                    local_se2_distributed_recovery_pre_mean
                )
        if self.arm_cfg.local_se2_planner:
            if any(
                value is None
                for value in (
                    local_se2_plan_mean,
                    local_se2_plan_precision,
                    local_se2_plan_used,
                    local_se2_plan_hidden_delta,
                    local_se2_plan_base_mean,
                    local_se2_recovery_plan_delta,
                )
            ):
                raise RuntimeError("local SE(2) planner omitted audit tensors")
            aux.update(
                {
                    LOCAL_SE2_PLAN_MEAN_AUX_KEY: local_se2_plan_mean,
                    LOCAL_SE2_PLAN_PRECISION_AUX_KEY: local_se2_plan_precision,
                    LOCAL_SE2_PLAN_USED_AUX_KEY: local_se2_plan_used,
                    LOCAL_SE2_PLAN_HIDDEN_DELTA_AUX_KEY: local_se2_plan_hidden_delta,
                    LOCAL_SE2_PLAN_BASE_MEAN_AUX_KEY: local_se2_plan_base_mean,
                    LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY: (
                        local_se2_recovery_plan_delta
                    ),
                }
            )
            if self.arm_cfg.local_se2_direct_wrench_unary:
                if any(
                    value is None
                    for value in (
                        local_se2_direct_wrench_pre,
                        local_se2_direct_wrench_delta,
                        local_se2_direct_wrench_post,
                    )
                ):
                    raise RuntimeError("direct wrench unary omitted audit tensors")
                aux.update(
                    {
                        LOCAL_SE2_DIRECT_WRENCH_PRE_AUX_KEY: local_se2_direct_wrench_pre,
                        LOCAL_SE2_DIRECT_WRENCH_DELTA_AUX_KEY: local_se2_direct_wrench_delta,
                        LOCAL_SE2_DIRECT_WRENCH_POST_AUX_KEY: local_se2_direct_wrench_post,
                    }
                )
            if self.arm_cfg.local_se2_analytic_total_wrench:
                if local_se2_analytic_wrench_proposal is None:
                    raise RuntimeError("analytic total wrench omitted its audit proposal")
                aux[LOCAL_SE2_ANALYTIC_WRENCH_PROPOSAL_AUX_KEY] = (
                    local_se2_analytic_wrench_proposal
                )
            if self.arm_cfg.local_se2_terminal_attractor:
                if local_se2_terminal_plan is None or local_se2_terminal_active is None:
                    raise RuntimeError("terminal SE(2) attractor omitted its audit tensors")
                aux[LOCAL_SE2_TERMINAL_PLAN_AUX_KEY] = local_se2_terminal_plan
                aux[LOCAL_SE2_TERMINAL_ACTIVE_AUX_KEY] = local_se2_terminal_active
        if correction_result is not None:
            if correction_section is None or correction_wrench is None or correction_force is None:
                raise RuntimeError("enabled correction sheaf omitted physical outputs")
            aux.update(
                {
                    CORRECTION_SECTION_AUX_KEY: correction_section,
                    CORRECTION_WRENCH_AUX_KEY: correction_wrench,
                    CORRECTION_FORCE_AUX_KEY: correction_force,
                    "correction_stalk_mean": correction_result.mean,
                    "correction_stalk_precision": correction_result.precision,
                }
            )
            if self.arm_cfg.correction_public_unary_residual:
                if (
                    correction_public_unary_delta is None
                    or correction_public_unary_mean_pre is None
                    or correction_public_unary_mean_post is None
                ):
                    raise RuntimeError("public correction-unary residual omitted its audit tensors")
                aux.update(
                    {
                        CORRECTION_PUBLIC_UNARY_DELTA_AUX_KEY: correction_public_unary_delta,
                        CORRECTION_PUBLIC_UNARY_PRE_AUX_KEY: correction_public_unary_mean_pre,
                        CORRECTION_PUBLIC_UNARY_POST_AUX_KEY: correction_public_unary_mean_post,
                    }
                )
            if self.arm_cfg.correction_source_null_recovery:
                if (
                    correction_source_null_risk is None
                    or correction_source_null_delta is None
                    or correction_source_null_mean_pre is None
                    or correction_source_null_mean_post is None
                ):
                    raise RuntimeError("source-null recovery omitted its audit tensors")
                aux.update(
                    {
                        CORRECTION_SOURCE_NULL_RISK_AUX_KEY: correction_source_null_risk,
                        CORRECTION_SOURCE_NULL_DELTA_AUX_KEY: correction_source_null_delta,
                        CORRECTION_SOURCE_NULL_PRE_AUX_KEY: correction_source_null_mean_pre,
                        CORRECTION_SOURCE_NULL_POST_AUX_KEY: correction_source_null_mean_post,
                    }
                )
            if self.arm_cfg.correction_phase_decoder:
                aux["correction_phase_basis"] = correction_phase_basis
                aux["correction_phase_wrench"] = correction_phase_wrench
            if self.arm_cfg.correction_clearance_bank_decoder:
                aux["correction_clearance_bank_basis"] = correction_clearance_bank_basis
                aux["correction_clearance_bank_wrench"] = correction_clearance_bank_wrench
            if self.arm_cfg.correction_approach_bank_decoder:
                aux["correction_approach_bank_basis"]=correction_approach_bank_basis
                aux["correction_approach_bank_wrench"]=correction_approach_bank_wrench
            if self.arm_cfg.collision_role_factors:
                role_values = (
                    collision_role_logits,
                    collision_role_weights,
                    collision_role_mean_delta,
                    collision_role_log_precision_delta,
                    collision_role_follower_edge_weight,
                    collision_role_edge_sigma2,
                    collision_role_node_restriction,
                )
                if any(value is None for value in role_values):
                    raise RuntimeError("collision role factors omitted audit tensors")
                aux.update(
                    {
                        COLLISION_ROLE_LOGITS_AUX_KEY: collision_role_logits,
                        COLLISION_ROLE_WEIGHTS_AUX_KEY: collision_role_weights,
                        COLLISION_ROLE_MEAN_DELTA_AUX_KEY: collision_role_mean_delta,
                        COLLISION_ROLE_LOG_PRECISION_DELTA_AUX_KEY: (
                            collision_role_log_precision_delta
                        ),
                        COLLISION_ROLE_UNARY_PRECISION_AUX_KEY: correction_precision,
                        COLLISION_ROLE_FOLLOWER_EDGE_WEIGHT_AUX_KEY: (
                            collision_role_follower_edge_weight
                        ),
                        COLLISION_ROLE_EDGE_SIGMA2_AUX_KEY: collision_role_edge_sigma2,
                        COLLISION_ROLE_RESTRICTION_AUX_KEY: (
                            collision_role_node_restriction
                        ),
                    }
                )
        if result.message_trace is not None:
            # Natural vector and diagonal precision are the exact wire payload
            # of the default Sheaf-GBP implementation.  The explicit endpoint
            # tensors make the ``2e: u->v, 2e+1: v->u`` contract auditable by
            # downstream recorders instead of requiring them to infer it.
            trace = result.message_trace
            aux["gbp_message_trace"] = trace
            aux["message_sender"] = trace.sender
            aux["message_receiver"] = trace.receiver
            aux["message_natural_trace"] = trace.natural
            aux["message_effective_precision_trace"] = trace.precision
            if trace.transmits_precision:
                aux["message_precision_trace"] = trace.precision
            aux["message_mean_trace"] = trace.mean
            aux["message_intensity_trace"] = trace.intensity
            # Compatibility aliases for existing rollout/video consumers.
            aux["message_xi_trace"] = trace.natural
            aux["message_omega_trace"] = trace.precision
        if correction_result is not None and correction_result.message_trace is not None:
            correction_trace = correction_result.message_trace
            base_trace = result.message_trace
            if self.arm_cfg.local_se2_distributed_wrench_control:
                # In the shape-agnostic controller this is not a correction
                # packet appended to a legacy packet: it is the complete and
                # only real communication payload.
                aux["gbp_message_trace"] = correction_trace
                aux["message_sender"] = correction_trace.sender
                aux["message_receiver"] = correction_trace.receiver
                aux["message_natural_trace"] = correction_trace.natural
                aux["message_effective_precision_trace"] = correction_trace.precision
                aux["message_precision_trace"] = correction_trace.precision
                aux["message_mean_trace"] = correction_trace.mean
                aux["message_intensity_trace"] = correction_trace.intensity
                aux["message_xi_trace"] = correction_trace.natural
                aux["message_omega_trace"] = correction_trace.precision
                aux["correction_gbp_message_trace"] = correction_trace
                aux["correction_message_sender"] = correction_trace.sender
                aux["correction_message_receiver"] = correction_trace.receiver
                aux["correction_message_natural_trace"] = correction_trace.natural
                aux["correction_message_effective_precision_trace"] = (
                    correction_trace.precision
                )
                aux["correction_message_precision_trace"] = correction_trace.precision
                aux["correction_message_mean_trace"] = correction_trace.mean
                aux["correction_message_intensity_trace"] = correction_trace.intensity
                aux["bundled_message_payload"] = torch.cat(
                    (correction_trace.natural, correction_trace.precision), dim=-1
                )
                aux["bundled_message_natural_width"] = (
                    correction_trace.natural.shape[-1]
                )
                aux["bundled_message_precision_width"] = (
                    correction_trace.precision.shape[-1]
                )
                aux["bundled_message_wire_width"] = 2 * (
                    correction_trace.natural.shape[-1]
                )
                base_trace = correction_trace
            if base_trace is None:
                raise RuntimeError("correction tracing requires the bundled source trace")
            if (
                base_trace.natural.shape[:3] != correction_trace.natural.shape[:3]
                or not torch.equal(base_trace.sender, correction_trace.sender)
                or not torch.equal(base_trace.receiver, correction_trace.receiver)
            ):
                raise RuntimeError("source and correction messages do not share endpoints/sweeps")
            aux["correction_gbp_message_trace"] = correction_trace
            aux["correction_message_sender"] = correction_trace.sender
            aux["correction_message_receiver"] = correction_trace.receiver
            aux["correction_message_natural_trace"] = correction_trace.natural
            aux["correction_message_effective_precision_trace"] = correction_trace.precision
            aux["correction_message_precision_trace"] = correction_trace.precision
            aux["correction_message_mean_trace"] = correction_trace.mean
            aux["correction_message_intensity_trace"] = correction_trace.intensity

            # Exact packet layout for audit/serialization.  Both solvers run
            # the same gbp_steps over identical directed endpoints; the
            # correction payload is appended to each source packet rather
            # than creating a second communication round.
            # One direct-sum Gaussian packet: all natural coordinates first,
            # followed by all transmitted precision coordinates.  Keeping
            # the two 15-D blocks contiguous makes the wire schema explicit
            # and prevents a recorder from mistaking the source precision for
            # correction natural parameters.
            if self.arm_cfg.local_se2_distributed_wrench_control:
                # The complete packet was written above; do not append the
                # same correction trace to itself.
                return_output_bundle = False
            else:
                return_output_bundle = True
            natural_width = base_trace.natural.shape[-1] + correction_trace.natural.shape[-1]
            precision_parts = []
            if base_trace.transmits_precision:
                precision_parts.append(base_trace.precision)
            precision_parts.append(correction_trace.precision)
            precision_width = sum(part.shape[-1] for part in precision_parts)
            bundle = torch.cat(
                (
                    base_trace.natural,
                    correction_trace.natural,
                    *precision_parts,
                ),
                dim=-1,
            )
            if not return_output_bundle:
                bundle = aux["bundled_message_payload"]
                natural_width = correction_trace.natural.shape[-1]
                precision_width = correction_trace.precision.shape[-1]
            expected_width = self.communication_floats_per_directed_edge_round()
            if bundle.shape[-1] != expected_width:
                raise RuntimeError(
                    f"bundled wire width {bundle.shape[-1]} != accounting {expected_width}"
                )
            aux["communication_bundle_wire_trace"] = bundle
            aux["communication_bundle_sender"] = correction_trace.sender
            aux["communication_bundle_receiver"] = correction_trace.receiver
            aux["communication_bundle_width"] = torch.tensor(
                expected_width,
                dtype=torch.long,
                device=obs.device,
            )
            aux["communication_bundle_natural_width"] = torch.tensor(
                natural_width,
                dtype=torch.long,
                device=obs.device,
            )
            aux["communication_bundle_precision_width"] = torch.tensor(
                precision_width,
                dtype=torch.long,
                device=obs.device,
            )
            aux["communication_bundle_sweeps"] = torch.tensor(
                self.communication_rounds(),
                dtype=torch.long,
                device=obs.device,
            )
            aux["correction_bundled_same_round"] = torch.tensor(True, device=obs.device)
        return PolicyOutput(
            forces=forces,
            hidden=next_hidden,
            aux=aux,
        )


def make_policy(arm: str, env_cfg: EnvConfig, arm_cfg: ArmConfig) -> BasePolicy:
    if arm_cfg.queryable_goal_free_temporal and arm != "sheaf_gbp":
        raise ValueError("queryable goal-free temporal memory is a Sheaf-GBP feature")
    if arm_cfg.correction_sheaf and arm != "sheaf_gbp":
        raise ValueError("the continuous correction sheaf requires arm='sheaf_gbp'")
    if arm == "no_comm":
        return NoCommPolicy(env_cfg, arm_cfg)
    if arm == "raw_full":
        return MPNNPolicy(env_cfg, arm_cfg, message_dim=32, name="raw_full")
    if arm == "comm_matched":
        return MPNNPolicy(env_cfg, arm_cfg, message_dim=2 * arm_cfg.edge_dim, name="comm_matched")
    if arm == "sheaf_gbp":
        return SheafGBPPolicy(env_cfg, arm_cfg)
    raise ValueError(f"unknown arm: {arm}")
