"""Configuration objects for PianoMovers-Force."""

from __future__ import annotations

from dataclasses import dataclass
import math


RETURN_CANDIDATE_ARCHITECTURE_V1 = "legacy-decoder-feature-mlp-v1"
RETURN_CANDIDATE_ARCHITECTURE_V2 = "physical15-mlp128x2-v2"
RETURN_CANDIDATE_ARCHITECTURE_V3 = "physical32-direct-residual-mlp192x2-v3"
HANDLE_LAYOUT_LEGACY_OUTRIGGER_V1 = "legacy-outrigger-v1"
HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2 = "boundary-attached-v2"
HANDLE_LAYOUT_PARAMETRIC_T_V3 = "parametric-t-v3"
HANDLE_LAYOUT_GLYPH_BOUNDARY_V4 = "glyph-boundary-v4"
HANDLE_LAYOUTS = (
    HANDLE_LAYOUT_LEGACY_OUTRIGGER_V1,
    HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2,
    HANDLE_LAYOUT_PARAMETRIC_T_V3,
    HANDLE_LAYOUT_GLYPH_BOUNDARY_V4,
)
STRICT_OBSERVATION_SCHEMA_FINAL_GOAL_V1 = "strict-local-final-goal-v1"
STRICT_OBSERVATION_SCHEMA_LOCAL_CURRENT_SLIT_V2 = "strict-local-current-slit-v2"
STRICT_OBSERVATION_SCHEMAS = (
    STRICT_OBSERVATION_SCHEMA_FINAL_GOAL_V1,
    STRICT_OBSERVATION_SCHEMA_LOCAL_CURRENT_SLIT_V2,
)


@dataclass(frozen=True)
class EnvConfig:
    n_agents: int = 6
    # Historical checkpoints predate physical handle-layout versioning.  The
    # default must therefore remain the original outrigger geometry so
    # ``EnvConfig(**checkpoint["env_cfg"])`` reproduces their dynamics,
    # observations, restrictions, grasp map, and radius graph exactly.
    handle_layout: str = HANDLE_LAYOUT_LEGACY_OUTRIGGER_V1
    # Rigid collision/render geometry. Historical checkpoints omit this field
    # and therefore reconstruct their exact T morphology through the default.
    payload_shape: str = "T"
    # Optional parametric-T geometry.  The defaults are exactly the historical
    # two-rectangle T.  Keeping these values explicit in new checkpoints makes
    # morphology-conditioned data auditable while checkpoints that predate the
    # fields reconstruct the canonical payload unchanged.
    t_total_width_m: float = 0.935
    t_total_height_m: float = 0.9185
    t_stem_width_m: float = 0.220
    t_cap_thickness_m: float = 0.2035
    dt: float = 0.08
    mass: float = 4.0
    inertia: float = 1.35
    linear_damping: float = 0.18
    angular_damping: float = 0.16
    max_force: float = 4.0
    world_x: float = 3.5
    world_y: float = 1.75
    wall_slab: float = 0.055
    collision_substeps: int = 1
    success_pos_tol: float = 0.22
    success_angle_tol: float = 0.35
    max_steps: int = 90
    comm_radius: float = 1.18
    scenario_difficulty: float = 1.0
    gap_y_range: float = 0.58
    gap_half_base: float = 0.83
    gap_half_difficulty_scale: float = 0.05
    gap_half_jitter: float = 0.08
    wall_x_abs: float = 0.82
    source_x: float = -2.35
    goal_x: float = 2.35
    # Mission command supplied identically to every attached agent.  These
    # are goal coordinates, not a stored route or teacher phase.  Historical
    # checkpoints omit them and therefore retain the original (y,theta)=0
    # endpoint exactly.
    goal_y: float = 0.0
    goal_theta: float = 0.0
    bidirectional: bool = False
    round_trip: bool = False
    round_trip_speed_tol: float = 0.30
    round_trip_omega_tol: float = 0.30
    round_trip_dwell_steps: int = 3
    local_observation: bool = False
    strict_local_observation: bool = False
    # Historical checkpoints omit this field and must reconstruct their exact
    # final-goal-conditioned 18-D observation.  New experiments opt into V2
    # explicitly; changing this value is an EnvConfig mismatch at load time.
    strict_observation_schema: str = STRICT_OBSERVATION_SCHEMA_FINAL_GOAL_V1
    wall_sensor_range: float = 1.05
    # Fixed-capacity, radius-limited records for optional generic AABB
    # obstacles.  Zero preserves every historical observation schema exactly.
    # Each enabled slot appends five channels to every agent observation.
    generic_obstacle_slots: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        from .payload_shapes import TMorphology, normalize_payload_shape

        normalize_payload_shape(self.payload_shape)
        TMorphology(
            total_width_m=self.t_total_width_m,
            total_height_m=self.t_total_height_m,
            stem_width_m=self.t_stem_width_m,
            cap_thickness_m=self.t_cap_thickness_m,
        ).validate()
        if self.handle_layout not in HANDLE_LAYOUTS:
            raise ValueError(
                f"handle_layout must be one of {HANDLE_LAYOUTS}, got {self.handle_layout!r}"
            )
        if self.handle_layout == HANDLE_LAYOUT_PARAMETRIC_T_V3:
            if self.payload_shape != "T":
                raise ValueError("parametric-t-v3 handles require payload_shape='T'")
            if not 8 <= self.n_agents <= 20:
                raise ValueError("parametric-t-v3 supports every integer agent count in [8,20]")
        if self.handle_layout == HANDLE_LAYOUT_GLYPH_BOUNDARY_V4:
            from .payload_shapes import LETTER_PAYLOAD_SHAPES

            if self.payload_shape not in LETTER_PAYLOAD_SHAPES:
                raise ValueError("glyph-boundary-v4 requires an A-Z payload shape")
            if not 8 <= self.n_agents <= 20:
                raise ValueError("glyph-boundary-v4 supports every integer agent count in [8,20]")
        if self.strict_observation_schema not in STRICT_OBSERVATION_SCHEMAS:
            raise ValueError(
                "strict_observation_schema must be one of "
                f"{STRICT_OBSERVATION_SCHEMAS}, got {self.strict_observation_schema!r}"
            )
        if (
            self.strict_observation_schema == STRICT_OBSERVATION_SCHEMA_LOCAL_CURRENT_SLIT_V2
            and not self.strict_local_observation
        ):
            raise ValueError("strict-local-current-slit-v2 requires strict_local_observation=True")
        if isinstance(self.generic_obstacle_slots, bool) or not isinstance(
            self.generic_obstacle_slots, int
        ):
            raise ValueError("generic_obstacle_slots must be an integer")
        if self.generic_obstacle_slots < 0:
            raise ValueError("generic_obstacle_slots must be non-negative")


@dataclass(frozen=True)
class ArmConfig:
    obs_dim: int
    hidden_dim: int = 96
    stalk_dim: int = 32
    edge_dim: int = 8
    message_dim: int = 32
    comm_rounds: int = 3
    gbp_steps: int = 3
    temporal_window: int = 4
    temporal_precision: float = 0.35
    damping: float = 0.55
    restriction_rank: int = 4
    restriction_residual_scale: float = 0.05
    edge_sigma2: float = 0.18
    force_residual_scale: float = 0.60
    analytic_force_scale: float = 1.0
    mean_only_messages: bool = False
    decoder_extra_context: bool = False
    edge_conditioned_restrictions: bool = False
    decoder_section_only: bool = False
    # Replace the legacy goal-conditioned recurrent/stalk FIFO with an
    # auditable K-frame FIFO containing only strict-local physical sensing.
    # The current goal queries this memory after every cached frame has been
    # expressed in the current active-leg frame.  Disabled by default so old
    # checkpoints and controller behavior remain byte-for-byte unchanged.
    queryable_goal_free_temporal: bool = False
    # Reflect an entire right-to-left mission into a fixed left-to-right task
    # frame before encoding, then reflect physical actions back.  By default
    # the frame is tied to the complete mission; the active-leg protocol below
    # performs an explicit memory-preserving frame transition at the turn.
    route_canonicalization: bool = False
    # At a round-trip turn, express the return leg in its own canonical route
    # frame.  The controller mirror-permutes the carried hidden state, then
    # uses exactly one of the replay or real planning-pause mechanisms below.
    active_leg_canonicalization: bool = False
    turn_replay_steps: int = 0
    # Communication-cheaper alternative to historical replay.  At the turn,
    # carry the mirror-permuted hidden state into the new active-leg frame,
    # then spend this many real physical ticks observing/planning under the
    # new goal while applying zero force.  These ticks remain inside the
    # episode horizon and normal communication accounting.
    turn_planning_steps: int = 0
    # Optional round-trip-only residual head.  The pretrained coordination
    # core remains frozen and leg 0 is algebraically identical to the source
    # policy; only the known return leg activates this head.
    return_adapter: bool = False
    return_adapter_scale: float = 2.0
    # Optional hard mixture-of-maneuvers return head.  Values 0 and 1 retain
    # the legacy single-wrench adapter and its checkpoint keys exactly.  The
    # six-mode variant predicts six scores and six candidate wrenches per
    # node, then deploys the single argmax candidate without soft mixing.
    return_maneuver_modes: int = 0
    # Binary source-anchored variant of the explicit single-mode adapter.
    # Candidate zero is the immutable source action; candidate one is a
    # learned wrench correction selected only when its advantage is > 0.
    # Disabled by default so existing single-adapter checkpoints retain their
    # exact module structure and behavior.
    return_source_candidate: bool = False
    # Fail-closed declaration for the candidate architecture.  V2 sees only
    # seven locally reconstructed shared invariants and the detached
    # geometry-anchored 8-d learned sheaf-section prefix.  V3 instead uses 20
    # explicit local physical features plus the complete detached 12-d sheaf
    # section and directly deploys its residual on the return leg; it has no
    # learned source/candidate selector.  Neither architecture can use the
    # opaque stalk or raw decoder feature vector as a shortcut.
    # The dataclass default is deliberately V1: loading a historical
    # checkpoint whose serialized ArmConfig predates this field reconstructs
    # its original  decoder-feature candidate exactly.  New training CLIs and
    # the Windows launcher explicitly select V2.
    return_candidate_architecture: str = RETURN_CANDIDATE_ARCHITECTURE_V1
    # New return-adapter experiments can opt into the exact damped grasp
    # pseudoinverse.  False preserves the approximate analytic section
    # allocator used by existing adapter checkpoints.
    exact_return_grasp_allocator: bool = False
    # C4.4 adds a small, parallel correction sheaf around an immutable source
    # policy.  It is deliberately opt-in: historical checkpoints reconstruct
    # exactly the same module tree and state-dict when this flag is absent.
    correction_sheaf: bool = False
    # The correction node stalk is intentionally smaller than the frozen
    # source stalk.  Its edge section is *exactly* the three normalized wrench
    # coordinates [Fx, Fy, tau], so ordinary diagonal GBP adds six transmitted
    # floats (natural vector + precision) to each bundled source packet.
    correction_stalk_dim: int = 16
    correction_edge_dim: int = 3
    correction_restriction_rank: int = 2
    correction_restriction_residual_scale: float = 0.05
    correction_edge_sigma2: float = 0.18
    correction_wrench_scale: float = 1.0
    # Optional zero-behavior residual on the correction sheaf's public
    # normalized-wrench unary coordinates.  Unlike the slit-specific
    # post-GBP banks below, this proposal is fused by the existing 3-D GBP
    # factor graph and therefore adds no wire coordinates or rounds.
    correction_public_unary_residual: bool = False
    correction_public_unary_residual_scale: float = 0.25
    # Slit-specific, source-null recovery proposal.  The learned 3-D wrench
    # is injected into the correction unary *before* GBP, but only through a
    # fixed local visibility/clearance envelope.  The envelope is exactly
    # zero at the far-field source observation, so arbitrary learned weights
    # cannot perturb those source ticks.  Disabled by default to reconstruct
    # every historical checkpoint without adding parameters or computation.
    correction_source_null_recovery: bool = False
    correction_phase_decoder: bool = False
    correction_phase_basis_dim: int = 6
    correction_phase_content_dim: int = 12
    correction_phase_wrench_scale: float = 0.25
    correction_clearance_bank_decoder: bool = False
    correction_clearance_bank_basis_dim: int = 24
    correction_clearance_bank_content_dim: int = 12
    correction_clearance_bank_wrench_scale: float = 0.25
    correction_clearance_bank_centers_mm: tuple[float, ...] = (0.0, 5.0, 13.0, 21.0)
    correction_clearance_bank_sigma_mm: float = 3.0
    correction_approach_bank_decoder: bool = False
    correction_approach_bank_basis_dim: int = 6
    correction_approach_bank_content_dim: int = 12
    correction_approach_bank_wrench_scale: float = 0.25
    correction_approach_bank_center_mm: float = 28.0
    correction_approach_bank_sigma_mm: float = 2.0
    # Stage-2 contact recovery adds one private, persistent recurrent state per
    # agent.  It is deliberately disabled by default: historical configs keep
    # the original module tree and the original carried-hidden layout.  The
    # state is local-only; it conditions the existing correction sheaf and
    # therefore adds no wireless coordinates or GBP rounds.
    persistent_recovery_state: bool = False
    persistent_recovery_dim: int = 8
    persistent_recovery_phase_count: int = 4
    # Where the private recovery state enters the correction policy.  The
    # historical route perturbs the shared 96-D correction feature and is the
    # default so old serialized ArmConfig dictionaries reconstruct exactly.
    # ``correction_mean`` instead changes only the 16-D Gaussian unary mean;
    # its precision and restriction maps remain functions of the source-local
    # correction feature alone.
    persistent_recovery_injection: str = "shared_hidden"
    # Opt-in training scaffold for the short-horizon contact skill.  These
    # are three *factor families* inside the existing correction graph, not
    # three separately communicated GBP networks: leader (goal drive),
    # follower (edge compatibility), and rebel (contact/stall evidence).
    # Their information-form contributions are fused before the one existing
    # correction GBP solve, so enabling the scaffold adds parameters and
    # diagnostics but no packet coordinates or communication rounds.  The
    # role heads are zero-behaviour initialized; historical checkpoints omit
    # this flag and reconstruct their exact module tree.
    collision_role_factors: bool = False
    collision_role_count: int = 3
    collision_role_mean_scale: float = 0.25
    collision_role_log_precision_scale: float = 1.5
    collision_role_edge_log_precision_scale: float = 1.5
    # Curriculum variant: leader/follower may shape ordinary approach motion;
    # rebel remains strictly supported by observed contact.  False preserves
    # the historical source-correct bridge contract.
    collision_role_precontact_leader_follower: bool = False
    # Give role factors their own local encoder so newly added contact
    # channels can be learned without changing the source correction path.
    # The role heads remain the only route back into the shared GBP factors,
    # so zero heads preserve exact source behaviour at bridge time.
    collision_role_separate_encoder: bool = False
    # Two-timescale local planning. Each agent predicts one private normalized
    # short-horizon SE(2) reference; only the resulting 3-D wrench belief is
    # communicated by the existing correction Sheaf-GBP.
    local_se2_planner: bool = False
    local_se2_plan_dim: int = 8
    local_se2_planner_hidden_dim: int = 96
    # Body-frame geometric samples are fixed local knowledge of the carried
    # rigid body. V1 uses eight Gaussian density probes; V2 retains those
    # coordinates and appends a 4x4 density grid plus eight directional
    # supports (32 total). They condition only the private planner and add no
    # communicated coordinate. Disabled for historical checkpoint parity.
    local_se2_shape_signature: bool = False
    local_se2_shape_signature_dim: int = 8
    # Route-free shared morphology planner.  Unlike the historical recurrent
    # planner and the explicitly forbidden path-memory branch, this network
    # receives only the current locally reconstructed physical frame, the
    # current goal displacement, the short causal contact state, and the
    # analytic payload geometry.  It is reevaluated from scratch every tick.
    # A continuous morphology support is exactly zero on canonical T so the
    # established T skill remains structurally unchanged.
    local_se2_markov_shape_planner: bool = False
    local_se2_markov_shape_hidden_dim: int = 256
    local_se2_markov_shape_layers: int = 3
    # Final, private geometry-conditioned plan capacity for non-canonical
    # payloads. Its continuous geometry support is exactly zero on canonical
    # T, giving a structural source-preservation contract without a confidence
    # gate or any extra communicated coordinate.
    local_se2_geometry_refinement_adapter: bool = False
    local_se2_geometry_refinement_hidden_dim: int = 128
    local_se2_geometry_refinement_scale: float = 1.0
    # Append a shared 12-D payload/slit frame reconstructed independently
    # from each handle's strict-local observation.  This remains private
    # planner context and therefore adds no transmitted coordinate.
    local_se2_geometry_refinement_canonical_features: bool = False
    # A private hard mixture separates mutually exclusive local planning
    # modes (advance, brake, retreat, tangential escape) before their 8-D
    # residuals are converted to the one shared 3-D wrench sheaf.  It is
    # source-null on canonical T and changes neither packets nor GBP rounds.
    local_se2_geometry_mode_adapter: bool = False
    local_se2_geometry_mode_count: int = 4
    local_se2_geometry_mode_hidden_dim: int = 256
    local_se2_geometry_mode_scale: float = 1.0
    # Preserve the incumbent plan until a real rejected transition has
    # entered the local recurrent state, then allow discrete recovery modes.
    local_se2_geometry_mode_post_contact_only: bool = False
    # Retrieval-augmented private local planning.  A continuous morphology
    # signature selects one certified SE(2) path memory; the current payload
    # pose, reconstructed independently from each strict-local handle row,
    # selects the next waypoint.  Only the resulting wrench enters GBP.
    local_se2_path_memory: bool = False
    local_se2_path_memory_shape_count: int = 24
    local_se2_path_memory_max_steps: int = 256
    local_se2_path_memory_ridge_decoder: bool = False
    local_se2_path_memory_ridge_points: int = 224
    local_se2_path_memory_ridge_alpha: float = 3.0e-4
    local_se2_path_memory_ridge_trust_margin: float = 0.05
    local_se2_path_memory_canonical_family_mse_threshold: float = 0.10
    local_se2_path_memory_population_resampling: bool = False
    local_se2_path_memory_population_resampling_points: int = 256
    local_se2_path_memory_population_preemptive_counts: tuple[int, ...] = (10,)
    local_se2_path_memory_canonical_preemptive_counts: tuple[int, ...] = (
        11,
        13,
        15,
        16,
        17,
        20,
    )
    local_se2_path_memory_population_recovery_counts: tuple[int, ...] = (
        11,
        13,
        15,
        17,
        20,
    )
    # Source-null capacity for the planner's genuinely different post-contact
    # distribution.  The adapter is selected only after a locally observed
    # collision has activated the persistent recovery state, so historical
    # pre-contact plans remain bitwise identical.  Its input includes the
    # four supervised recovery-phase probabilities (approach/brake/retreat/
    # re-enter), but it remains private and adds no GBP packet coordinates.
    local_se2_recovery_adapter: bool = False
    local_se2_recovery_adapter_hidden_dim: int = 96
    local_se2_recovery_adapter_scale: float = 1.0
    local_se2_recovery_canonical_features: bool = False
    # A private, contact-triggered residual for the mechanically distinct
    # agents whose rejected handle motion supplies the causal "rebel" signal.
    # Its zero head preserves the parent policy bitwise; after training, its
    # plan correction is converted to wrench evidence and fused by the same
    # communication-matched Sheaf-GBP as every other local proposal.
    local_se2_contact_rebel_adapter: bool = False
    local_se2_contact_rebel_hidden_dim: int = 64
    local_se2_contact_rebel_scale: float = 1.0
    # Natural phase boundary for a left-to-right passage: preserve the learned
    # approach/retreat controller exactly until the payload centre crosses the
    # wall mid-plane, then permit contact-local correction on the exit side.
    local_se2_contact_rebel_min_center_x_m: float = 0.0
    # A second source-null residual can be appended after a trained rebel
    # checkpoint. Its later spatial support makes stagewise frontier learning
    # exact: the already learned entrance and first exit rotation stay bitwise
    # fixed while new capacity learns the next collision basin.
    local_se2_exit_refinement_adapter: bool = False
    local_se2_exit_refinement_hidden_dim: int = 64
    local_se2_exit_refinement_scale: float = 1.0
    local_se2_exit_refinement_min_center_x_m: float = 0.17
    # A third source-null residual adds capacity for perturbation-induced
    # entrance contacts that occur before the two stagewise exit branches are
    # spatially active.  It remains contact-history gated and private.
    local_se2_approach_refinement_adapter: bool = False
    local_se2_approach_refinement_hidden_dim: int = 64
    local_se2_approach_refinement_scale: float = 1.0
    local_se2_approach_refinement_min_center_x_m: float = -0.30
    local_se2_approach_refinement_stuck_steps: int = 9
    # Stagewise capacity for trajectories that remain blocked beyond every
    # already-successful rollout's contact streak.  A strictly higher causal
    # threshold makes the protected success set structurally immutable.
    local_se2_deep_stuck_refinement_adapter: bool = False
    local_se2_deep_stuck_refinement_hidden_dim: int = 64
    local_se2_deep_stuck_refinement_scale: float = 1.0
    local_se2_deep_stuck_refinement_min_center_x_m: float = -0.30
    local_se2_deep_stuck_refinement_stuck_steps: int = 10
    # An append-only bank of private retry residuals.  Each threshold must be
    # strictly later than the preceding one.  Bridging a new zero head beyond
    # every successful rollout's observed contact streak makes those earlier
    # successes structurally immutable while adding capacity only to still
    # blocked trajectories.  The bank changes no communicated GBP fields.
    local_se2_retry_refinement_thresholds: tuple[int, ...] = ()
    local_se2_retry_refinement_hidden_dim: int = 64
    local_se2_retry_refinement_scale: float = 1.0
    local_se2_retry_refinement_min_center_x_m: float = -0.30
    # Temporary interpretable phase scaffold for the short contact skill.
    # After a real rejected transition, retreat remains active until every
    # local handle observation reconstructs a payload centre behind this
    # threshold; re-entry is then latched for the rest of the episode.  The
    # latch is private and source-null before contact, and can later be
    # distilled away into the recurrent recovery state.
    local_se2_analytic_phase_latch: bool = False
    local_se2_reentry_center_x_threshold_m: float = -0.16
    local_se2_phase_latch_logit: float = 8.0
    # Zero-behaviour physical adapter from private plan/context directly to
    # the public [Fx,Fy,tau] correction unary.  It does not add wire fields.
    local_se2_direct_wrench_unary: bool = False
    local_se2_direct_wrench_hidden_dim: int = 128
    # Physically grounded hierarchical controller.  Every node converts its
    # private one-step SE(2) plan and own contact velocity into a local
    # sufficient-statistic proposal for the *total* normalized payload
    # wrench.  The existing 3-D correction Sheaf-GBP fuses those proposals;
    # its exact grasp decoder then replaces (rather than corrects) the legacy
    # action.  No additional coordinates or communication rounds are added.
    local_se2_analytic_total_wrench: bool = False
    local_se2_analytic_unary_precision: float = 1.0
    # Relative Gaussian confidence assigned to a radius-local geometric plan.
    # A value above one lets the few ants that actually see a constraining
    # wall dominate uninformed goal-pull unaries through ordinary GBP, without
    # adding a message field or granting any ant nonlocal observations.
    local_se2_geometry_evidence_precision_boost: float = 1.0
    local_se2_analytic_edge_sigma2: float = 1.0e-5
    local_se2_analytic_solver_eps: float = 1.0e-10
    local_se2_analytic_max_precision: float = 1.0e9
    # Shape-agnostic distributed controller.  The private recurrent planner's
    # first three outputs are interpreted as a local normalized-wrench belief;
    # the existing 3-D correction GBP fuses those beliefs.  A shared decoder
    # maps only [local observation, fused belief, private plan] to this
    # handle's force.  No letter ID, global payload descriptor, path lookup,
    # global grasp matrix, or legacy source message is on the action path.
    # Disabled by default so historical checkpoints reconstruct exactly.
    local_se2_distributed_wrench_control: bool = False
    # Preserve the planner's original physical next-SE(2) semantics while
    # retaining the six-float distributed wire.  Each node converts its 8-D
    # local next-state plan to a 3-D normalized wrench proposal via analytic
    # inverse dynamics; only that physical proposal is fused by Sheaf-GBP.
    # This is distinct from the legacy distributed controller above, which
    # repurposes plan coordinates 0:3 as a direct wrench prediction.
    local_se2_distributed_next_se2_control: bool = False
    local_se2_distributed_force_hidden_dim: int = 128
    # Append-only source-null capacity for model-induced recovery states.  It
    # is selected only after a locally observed rejected transition, so an
    # arbitrary trained adapter cannot alter any pre-contact action.
    local_se2_distributed_recovery_adapter: bool = False
    local_se2_distributed_recovery_hidden_dim: int = 128
    local_se2_distributed_recovery_layers: int = 2
    # One preserves historical continuous regression. Four enables the
    # generic forward/brake/retreat/escape maneuver experts; mode selection is
    # private and the wire protocol remains the same 3-D wrench belief.
    local_se2_distributed_recovery_mode_count: int = 1
    local_se2_distributed_recovery_scale: float = 1.0
    # A rejected transition activates recovery, but a rebel must relinquish
    # that role after its own handle has moved away from every locally visible
    # wall.  This prevents the historical once-hit/active-forever latch while
    # retaining an exact source-null pre-contact action path.
    local_se2_distributed_recovery_local_support: bool = False
    local_se2_distributed_recovery_wall_range_m: float = 0.45
    # Shape-agnostic analytic action realization.  Every handle contributes
    # [1, r_x, r_y, ||r||^2] through an exact sum-product pass on the radius
    # graph's coordination tree.  The resulting four sufficient statistics
    # define the complete planar grasp Gram matrix without a letter, global
    # outline, centralized grasp lookup, or learned force-allocation shortcut.
    local_se2_distributed_grasp_allreduce: bool = False
    # Once the complete rigid body is safely beyond the single wall, replace
    # the learned path plan with a local analytic SE(2) goal attractor.  This
    # is a planner module (not an action/confidence gate): the resulting next
    # twist still goes through the same Sheaf-GBP inverse-dynamics layer.
    local_se2_terminal_attractor: bool = False
    local_se2_terminal_position_gain: float = 2.0
    local_se2_terminal_angle_gain: float = 2.0
    local_se2_terminal_max_speed: float = 0.35
    local_se2_terminal_max_omega: float = 0.8
    # Parameter-free easy-mode planner.  A node selects a direct corridor
    # plan only when its own visible slit record proves that the complete
    # axis-aligned compound payload fits with the requested reserve.  Hard
    # slits therefore retain the learned planner exactly, while easy slits
    # use a short translation/braking solution through the same Sheaf-GBP
    # inverse-dynamics controller.
    local_se2_feasible_corridor_attractor: bool = False
    # Apply the same current-geometry C-space planner to the canonical T as
    # to every morphology OOD body.  False preserves historical checkpoints,
    # whose canonical-T local GRU learned the original fixed training route.
    # New route-free candidates enable this and therefore consume no stored
    # path, teacher phase, shape label, or carried path index at deployment.
    local_se2_route_free_geometry_planner: bool = False
    local_se2_corridor_reserve_m: float = 0.015
    local_se2_corridor_max_speed: float = 0.50

    def __post_init__(self) -> None:
        if self.persistent_recovery_state and not self.correction_sheaf:
            raise ValueError("persistent recovery state requires correction_sheaf=True")
        if self.persistent_recovery_state and self.obs_dim != 27:
            raise ValueError("persistent recovery state requires contact-feedback27")
        if self.persistent_recovery_state and self.hidden_dim != 96:
            raise ValueError("persistent recovery state requires the fixed 96-D hidden")
        if self.persistent_recovery_state and self.queryable_goal_free_temporal:
            raise ValueError("persistent recovery state does not support goal-free temporal memory")
        if self.persistent_recovery_dim <= 0:
            raise ValueError("persistent_recovery_dim must be positive")
        if self.persistent_recovery_phase_count <= 0:
            raise ValueError("persistent_recovery_phase_count must be positive")
        if self.persistent_recovery_injection not in {"shared_hidden", "correction_mean"}:
            raise ValueError(
                "persistent_recovery_injection must be 'shared_hidden' or "
                "'correction_mean'"
            )
        if (
            self.persistent_recovery_injection != "shared_hidden"
            and not self.persistent_recovery_state
        ):
            raise ValueError(
                "non-legacy persistent recovery injection requires "
                "persistent_recovery_state=True"
            )
        if self.collision_role_factors and not (
            self.correction_sheaf
            and self.persistent_recovery_state
            and self.persistent_recovery_injection == "shared_hidden"
        ):
            raise ValueError(
                "collision role factors require the shared-hidden persistent "
                "correction sheaf"
            )
        if self.collision_role_factors and self.collision_role_count != 3:
            raise ValueError(
                "collision role factors require exactly leader/follower/rebel"
            )
        if (
            self.collision_role_precontact_leader_follower
            and not self.collision_role_factors
        ):
            raise ValueError(
                "precontact leader/follower requires collision_role_factors"
            )
        if self.collision_role_separate_encoder and not self.collision_role_factors:
            raise ValueError("separate role encoder requires collision_role_factors")
        if self.local_se2_planner and not (
            self.correction_sheaf
            and self.persistent_recovery_state
            and self.obs_dim == 27
        ):
            raise ValueError(
                "local SE(2) planner requires contact-feedback27 and persistent correction Sheaf"
            )
        if self.local_se2_planner and self.local_se2_plan_dim != 8:
            raise ValueError("local SE(2) planner uses the fixed physical 8-D plan")
        if self.local_se2_planner_hidden_dim <= 0:
            raise ValueError("local SE(2) planner hidden width must be positive")
        if (
            self.local_se2_distributed_recovery_local_support
            and not self.local_se2_distributed_recovery_adapter
        ):
            raise ValueError(
                "local distributed recovery support requires the recovery adapter"
            )
        if (
            not math.isfinite(self.local_se2_distributed_recovery_wall_range_m)
            or self.local_se2_distributed_recovery_wall_range_m <= 0.0
        ):
            raise ValueError(
                "local distributed recovery wall range must be finite and positive"
            )
        if self.local_se2_distributed_grasp_allreduce and not (
            self.local_se2_distributed_wrench_control
            and self.local_se2_analytic_total_wrench
        ):
            raise ValueError(
                "distributed grasp all-reduce requires analytic distributed "
                "wrench control"
            )
        if self.local_se2_recovery_adapter and not self.local_se2_planner:
            raise ValueError("local SE(2) recovery adapter requires the local planner")
        if (
            self.local_se2_recovery_canonical_features
            and not self.local_se2_recovery_adapter
        ):
            raise ValueError(
                "canonical recovery features require the local SE(2) adapter"
            )
        if self.local_se2_recovery_adapter_hidden_dim <= 0:
            raise ValueError("local SE(2) recovery adapter hidden width must be positive")
        if self.local_se2_recovery_adapter_scale <= 0.0:
            raise ValueError("local SE(2) recovery adapter scale must be positive")
        if self.local_se2_contact_rebel_adapter and not (
            self.local_se2_recovery_adapter
            and self.local_se2_recovery_canonical_features
            and self.obs_dim == 27
        ):
            raise ValueError(
                "contact rebel adapter requires canonical contact-feedback recovery"
            )
        if self.local_se2_contact_rebel_hidden_dim <= 0:
            raise ValueError("contact rebel adapter hidden width must be positive")
        if self.local_se2_contact_rebel_scale <= 0.0:
            raise ValueError("contact rebel adapter scale must be positive")
        if not math.isfinite(self.local_se2_contact_rebel_min_center_x_m):
            raise ValueError("contact rebel centre-x boundary must be finite")
        if self.local_se2_exit_refinement_adapter and not (
            self.local_se2_contact_rebel_adapter
            and self.local_se2_recovery_canonical_features
            and self.obs_dim == 27
        ):
            raise ValueError(
                "exit refinement requires the canonical contact rebel planner"
            )
        if self.local_se2_exit_refinement_hidden_dim <= 0:
            raise ValueError("exit refinement hidden width must be positive")
        if self.local_se2_exit_refinement_scale <= 0.0:
            raise ValueError("exit refinement scale must be positive")
        if not math.isfinite(self.local_se2_exit_refinement_min_center_x_m):
            raise ValueError("exit refinement centre-x boundary must be finite")
        if self.local_se2_approach_refinement_adapter and not (
            self.local_se2_exit_refinement_adapter
            and self.local_se2_recovery_canonical_features
            and self.obs_dim == 27
        ):
            raise ValueError(
                "approach refinement requires the canonical exit-refinement planner"
            )
        if self.local_se2_approach_refinement_hidden_dim <= 0:
            raise ValueError("approach refinement hidden width must be positive")
        if self.local_se2_approach_refinement_scale <= 0.0:
            raise ValueError("approach refinement scale must be positive")
        if not math.isfinite(self.local_se2_approach_refinement_min_center_x_m):
            raise ValueError("approach refinement centre-x boundary must be finite")
        if self.local_se2_approach_refinement_stuck_steps <= 0:
            raise ValueError("approach refinement stuck threshold must be positive")
        if self.local_se2_deep_stuck_refinement_adapter and not (
            self.local_se2_approach_refinement_adapter
            and self.local_se2_recovery_canonical_features
            and self.obs_dim == 27
        ):
            raise ValueError(
                "deep-stuck refinement requires the canonical approach planner"
            )
        if self.local_se2_deep_stuck_refinement_hidden_dim <= 0:
            raise ValueError("deep-stuck refinement hidden width must be positive")
        if self.local_se2_deep_stuck_refinement_scale <= 0.0:
            raise ValueError("deep-stuck refinement scale must be positive")
        if not math.isfinite(self.local_se2_deep_stuck_refinement_min_center_x_m):
            raise ValueError("deep-stuck refinement centre-x boundary must be finite")
        if (
            self.local_se2_deep_stuck_refinement_stuck_steps
            <= self.local_se2_approach_refinement_stuck_steps
        ):
            raise ValueError(
                "deep-stuck threshold must exceed the approach stuck threshold"
            )
        retry_thresholds = tuple(self.local_se2_retry_refinement_thresholds)
        if retry_thresholds and not self.local_se2_deep_stuck_refinement_adapter:
            raise ValueError("retry refinements require the deep-stuck planner")
        if self.local_se2_retry_refinement_hidden_dim <= 0:
            raise ValueError("retry refinement hidden width must be positive")
        if self.local_se2_retry_refinement_scale <= 0.0:
            raise ValueError("retry refinement scale must be positive")
        if not math.isfinite(self.local_se2_retry_refinement_min_center_x_m):
            raise ValueError("retry refinement centre-x boundary must be finite")
        previous_retry_threshold = self.local_se2_deep_stuck_refinement_stuck_steps
        for retry_threshold in retry_thresholds:
            if isinstance(retry_threshold, bool) or retry_threshold <= previous_retry_threshold:
                raise ValueError(
                    "retry refinement thresholds must be strictly increasing "
                    "beyond the deep-stuck threshold"
                )
            previous_retry_threshold = retry_threshold
        if self.local_se2_analytic_phase_latch and not (
            self.local_se2_recovery_adapter
            and self.persistent_recovery_state
            and self.persistent_recovery_phase_count == 4
        ):
            raise ValueError(
                "analytic phase latch requires the four-phase recovery adapter"
            )
        if not math.isfinite(self.local_se2_reentry_center_x_threshold_m):
            raise ValueError("re-entry centre-x threshold must be finite")
        if (
            not math.isfinite(self.local_se2_phase_latch_logit)
            or self.local_se2_phase_latch_logit <= 0.0
        ):
            raise ValueError("phase latch logit must be finite and positive")
        if self.local_se2_direct_wrench_unary and not self.local_se2_planner:
            raise ValueError("direct plan-wrench unary requires local_se2_planner=True")
        if self.local_se2_direct_wrench_hidden_dim <= 0:
            raise ValueError("direct plan-wrench unary hidden width must be positive")
        if self.local_se2_analytic_total_wrench and not self.local_se2_planner:
            raise ValueError("analytic total wrench requires local_se2_planner=True")
        if self.local_se2_distributed_wrench_control and not (
            self.local_se2_analytic_total_wrench
            and self.local_se2_planner
            and self.correction_sheaf
        ):
            raise ValueError(
                "distributed wrench control requires local planner, correction "
                "Sheaf, and analytic total-wrench semantics"
            )
        if self.local_se2_distributed_next_se2_control and not (
            self.local_se2_distributed_wrench_control
            and self.local_se2_analytic_total_wrench
            and self.local_se2_planner
        ):
            raise ValueError(
                "distributed next-SE(2) control requires the local planner "
                "and analytic distributed wrench control"
            )
        if (
            self.local_se2_distributed_wrench_control
            and not self.local_se2_distributed_next_se2_control
            and any(
                (
                    self.local_se2_shape_signature,
                    self.local_se2_geometry_refinement_adapter,
                    self.local_se2_geometry_mode_adapter,
                    self.local_se2_path_memory,
                    self.local_se2_terminal_attractor,
                    self.local_se2_feasible_corridor_attractor,
                )
            )
        ):
            raise ValueError(
                "distributed wrench control forbids global geometry, path memory, "
                "and exact-shape attractors"
            )
        if self.local_se2_distributed_force_hidden_dim <= 0:
            raise ValueError("distributed force-decoder width must be positive")
        if self.local_se2_distributed_recovery_adapter and not (
            self.local_se2_distributed_wrench_control
            and self.persistent_recovery_state
        ):
            raise ValueError(
                "distributed recovery requires shape-agnostic wrench control "
                "and persistent local contact memory"
            )
        if self.local_se2_distributed_recovery_hidden_dim <= 0:
            raise ValueError("distributed recovery width must be positive")
        if self.local_se2_distributed_recovery_layers <= 0:
            raise ValueError("distributed recovery depth must be positive")
        if self.local_se2_distributed_recovery_mode_count not in {1, 4}:
            raise ValueError("distributed recovery requires one or four modes")
        if (
            not math.isfinite(self.local_se2_distributed_recovery_scale)
            or self.local_se2_distributed_recovery_scale <= 0.0
        ):
            raise ValueError("distributed recovery scale must be finite and positive")
        if self.local_se2_shape_signature and not self.local_se2_planner:
            raise ValueError("shape signature requires local_se2_planner=True")
        if self.local_se2_shape_signature_dim not in {8, 32}:
            raise ValueError("the geometry support signature requires 8 or 32 dimensions")
        if self.local_se2_markov_shape_planner and not (
            self.local_se2_planner
            and self.local_se2_shape_signature
            and self.local_se2_shape_signature_dim == 32
            and self.persistent_recovery_state
            and self.obs_dim == 27
        ):
            raise ValueError(
                "Markov shape planning requires contact27, recovery state, "
                "and the private 32-D geometry signature"
            )
        if self.local_se2_markov_shape_planner and self.local_se2_path_memory:
            raise ValueError("Markov shape planning forbids path memory")
        if self.local_se2_markov_shape_hidden_dim <= 0:
            raise ValueError("Markov shape planner hidden width must be positive")
        if self.local_se2_markov_shape_layers <= 0:
            raise ValueError("Markov shape planner depth must be positive")
        if self.local_se2_geometry_refinement_adapter and not self.local_se2_shape_signature:
            raise ValueError("geometry refinement requires the private shape signature")
        if (
            self.local_se2_geometry_refinement_canonical_features
            and not self.local_se2_geometry_refinement_adapter
        ):
            raise ValueError(
                "canonical geometry features require the geometry refinement adapter"
            )
        if self.local_se2_geometry_mode_adapter and not (
            self.local_se2_geometry_refinement_adapter
            and self.local_se2_geometry_refinement_canonical_features
        ):
            raise ValueError(
                "geometry mode adapter requires canonical geometry refinement"
            )
        if self.local_se2_geometry_mode_count != 4:
            raise ValueError("geometry mode adapter requires exactly four modes")
        if self.local_se2_geometry_mode_hidden_dim <= 0:
            raise ValueError("geometry mode hidden width must be positive")
        if (
            not math.isfinite(self.local_se2_geometry_mode_scale)
            or self.local_se2_geometry_mode_scale <= 0.0
        ):
            raise ValueError("geometry mode scale must be finite and positive")
        if self.local_se2_path_memory and not (
            self.local_se2_shape_signature
            and self.local_se2_shape_signature_dim == 32
            and self.local_se2_geometry_refinement_canonical_features
            and self.local_se2_analytic_total_wrench
        ):
            raise ValueError(
                "path memory requires signature32, canonical features, and analytic wrench"
            )
        if self.local_se2_path_memory_shape_count <= 0:
            raise ValueError("path memory shape count must be positive")
        if self.local_se2_path_memory_max_steps <= 1:
            raise ValueError("path memory horizon must exceed one step")
        if self.local_se2_path_memory_ridge_decoder and not self.local_se2_path_memory:
            raise ValueError("path-memory ridge decoder requires path memory")
        if self.local_se2_path_memory_ridge_decoder and not 2 <= (
            self.local_se2_path_memory_ridge_points
        ) <= (self.local_se2_path_memory_max_steps + 1):
            raise ValueError("path-memory ridge points must fit the path buffer")
        if (
            not math.isfinite(self.local_se2_path_memory_ridge_alpha)
            or self.local_se2_path_memory_ridge_alpha <= 0.0
        ):
            raise ValueError("path-memory ridge alpha must be finite and positive")
        if (
            not math.isfinite(self.local_se2_path_memory_ridge_trust_margin)
            or self.local_se2_path_memory_ridge_trust_margin < 0.0
        ):
            raise ValueError("path-memory ridge trust margin must be finite and non-negative")
        if (
            not math.isfinite(
                self.local_se2_path_memory_canonical_family_mse_threshold
            )
            or self.local_se2_path_memory_canonical_family_mse_threshold <= 0.0
        ):
            raise ValueError(
                "path-memory canonical-family threshold must be finite and positive"
            )
        if self.local_se2_path_memory_population_resampling and not (
            self.local_se2_path_memory
        ):
            raise ValueError("population path resampling requires path memory")
        if self.local_se2_path_memory_population_resampling and not 2 <= (
            self.local_se2_path_memory_population_resampling_points
        ) <= (self.local_se2_path_memory_max_steps + 1):
            raise ValueError("population resampling points must fit the path buffer")
        preemptive_counts = tuple(
            self.local_se2_path_memory_population_preemptive_counts
        )
        if len(set(preemptive_counts)) != len(preemptive_counts) or any(
            isinstance(value, bool) or not 8 <= value <= 20 or value == 12
            for value in preemptive_counts
        ):
            raise ValueError(
                "population preemptive counts must be unique OOD counts in [8,20]"
            )
        recovery_counts = tuple(
            self.local_se2_path_memory_population_recovery_counts
        )
        canonical_preemptive_counts = tuple(
            self.local_se2_path_memory_canonical_preemptive_counts
        )
        if len(set(canonical_preemptive_counts)) != len(
            canonical_preemptive_counts
        ) or any(
            isinstance(value, bool) or not 8 <= value <= 20 or value == 12
            for value in canonical_preemptive_counts
        ):
            raise ValueError(
                "canonical preemptive counts must be unique OOD counts in [8,20]"
            )
        if len(set(recovery_counts)) != len(recovery_counts) or any(
            isinstance(value, bool) or not 8 <= value <= 20 or value == 12
            for value in recovery_counts
        ):
            raise ValueError(
                "population recovery counts must be unique OOD counts in [8,20]"
            )
        if self.local_se2_geometry_refinement_hidden_dim <= 0:
            raise ValueError("geometry refinement hidden width must be positive")
        if (
            not math.isfinite(self.local_se2_geometry_refinement_scale)
            or self.local_se2_geometry_refinement_scale <= 0.0
        ):
            raise ValueError("geometry refinement scale must be finite and positive")
        if self.local_se2_analytic_total_wrench and self.local_se2_direct_wrench_unary:
            raise ValueError(
                "analytic total wrench and learned direct-wrench unary are alternative controllers"
            )
        if self.local_se2_analytic_total_wrench and self.return_adapter:
            raise ValueError("analytic total wrench does not compose with a return adapter")
        if self.local_se2_analytic_unary_precision <= 0.0:
            raise ValueError("analytic total-wrench unary precision must be positive")
        if (
            not math.isfinite(self.local_se2_geometry_evidence_precision_boost)
            or self.local_se2_geometry_evidence_precision_boost < 1.0
        ):
            raise ValueError(
                "geometry-evidence precision boost must be finite and at least one"
            )
        if self.local_se2_analytic_edge_sigma2 <= 0.0:
            raise ValueError("analytic total-wrench edge sigma2 must be positive")
        if self.local_se2_analytic_solver_eps <= 0.0:
            raise ValueError("analytic total-wrench solver eps must be positive")
        if self.local_se2_analytic_max_precision <= 1.0:
            raise ValueError("analytic total-wrench max precision must exceed one")
        if self.local_se2_terminal_attractor and not self.local_se2_analytic_total_wrench:
            raise ValueError("terminal SE(2) attractor requires analytic total-wrench control")
        if self.local_se2_feasible_corridor_attractor and not (
            self.local_se2_terminal_attractor
            and self.local_se2_analytic_total_wrench
        ):
            raise ValueError(
                "feasible-corridor attractor requires terminal analytic "
                "total-wrench control"
            )
        if self.local_se2_route_free_geometry_planner and not (
            self.local_se2_feasible_corridor_attractor
            and self.local_se2_terminal_attractor
            and self.local_se2_analytic_total_wrench
        ):
            raise ValueError(
                "route-free geometry planner requires the physical feasible "
                "corridor planner"
            )
        if any(
            value <= 0.0
            for value in (
                self.local_se2_terminal_position_gain,
                self.local_se2_terminal_angle_gain,
                self.local_se2_terminal_max_speed,
                self.local_se2_terminal_max_omega,
            )
        ):
            raise ValueError("terminal SE(2) attractor gains and limits must be positive")
        if (
            not math.isfinite(self.local_se2_corridor_reserve_m)
            or self.local_se2_corridor_reserve_m <= 0.0
            or not math.isfinite(self.local_se2_corridor_max_speed)
            or self.local_se2_corridor_max_speed <= 0.0
        ):
            raise ValueError("feasible-corridor reserve and speed must be positive")
        if self.collision_role_factors and (
            self.collision_role_mean_scale <= 0.0
            or self.collision_role_log_precision_scale <= 0.0
            or self.collision_role_edge_log_precision_scale <= 0.0
        ):
            raise ValueError("collision role factor scales must be positive")
        if self.correction_public_unary_residual and not self.correction_sheaf:
            raise ValueError(
                "the public correction-unary residual requires correction_sheaf=True"
            )
        if self.correction_source_null_recovery and not self.correction_sheaf:
            raise ValueError("source-null recovery requires correction_sheaf=True")
        if self.correction_source_null_recovery and self.obs_dim not in {22, 27}:
            raise ValueError(
                "source-null recovery requires local exact-clearance22 or its "
                "versioned contact-feedback27 extension"
            )
        if self.correction_source_null_recovery and self.hidden_dim != 96:
            raise ValueError("source-null recovery requires the fixed 96-D correction hidden")
        if (
            self.correction_public_unary_residual
            and self.correction_public_unary_residual_scale <= 0.0
        ):
            raise ValueError("correction_public_unary_residual_scale must be positive")
        if not self.correction_sheaf:
            return
        if self.correction_stalk_dim != 16:
            raise ValueError("the C4.4 correction sheaf requires correction_stalk_dim=16")
        if self.correction_edge_dim != 3:
            raise ValueError("the C4.4 correction sheaf requires correction_edge_dim=3")
        if self.correction_restriction_rank <= 0:
            raise ValueError("correction_restriction_rank must be positive")
        if self.correction_restriction_residual_scale < 0.0:
            raise ValueError("correction_restriction_residual_scale must be non-negative")
        if self.correction_edge_sigma2 <= 0.0:
            raise ValueError("correction_edge_sigma2 must be positive")
        if self.correction_wrench_scale <= 0.0:
            raise ValueError("correction_wrench_scale must be positive")
        if self.correction_phase_decoder and (
            self.correction_phase_basis_dim != 6
            or self.correction_phase_content_dim != 12
            or self.correction_phase_wrench_scale <= 0.0
        ):
            raise ValueError("correction phase decoder requires the fixed 6x12 design")
        if self.correction_clearance_bank_decoder and (
            not self.correction_sheaf
            or self.correction_clearance_bank_basis_dim != 24
            or self.correction_clearance_bank_content_dim != 12
            or self.correction_clearance_bank_wrench_scale <= 0.0
            or self.correction_clearance_bank_centers_mm != (0.0, 5.0, 13.0, 21.0)
            or self.correction_clearance_bank_sigma_mm != 3.0
        ):
            raise ValueError("clearance-bank decoder requires the fixed 24x12 design")
        if self.correction_approach_bank_decoder and (
            not self.correction_clearance_bank_decoder
            or self.correction_approach_bank_basis_dim != 6
            or self.correction_approach_bank_content_dim != 12
            or self.correction_approach_bank_wrench_scale != 0.25
            or self.correction_approach_bank_center_mm != 28.0
            or self.correction_approach_bank_sigma_mm != 2.0
        ):
            raise ValueError("approach clearance bank requires the fixed 6x12 design")
