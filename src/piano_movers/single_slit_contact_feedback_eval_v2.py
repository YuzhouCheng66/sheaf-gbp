"""Explicit-contact closed-loop evaluation for the single-slit skill.

This V2 evaluator is intentionally separate from both the collision-terminal
legacy evaluator and the implicit-feedback V1 evaluator.  It uses the same
penetration-free rejection-and-stop transition authority as V1, but the next
policy call receives five explicit local mechanical aftereffect channels:
last-hit, the handle's previous force, and its rejected rigid-body motion.

Every valid policy call still advances recurrent/GBP hidden state and incurs
the real directed-edge payload cost.  An invalid output remains terminal;
contact remains nonterminal and can therefore support learned adjustment.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor

from .env import collision_clearance_margin, collision_mask, radius_edges
from .gbp import homogeneous_topology
from .models import (
    CORRECTION_WRENCH_AUX_KEY,
    LOCAL_SE2_PLAN_MEAN_AUX_KEY,
    LOCAL_SE2_RECOVERY_MODE_LOGITS_AUX_KEY,
    LOCAL_SE2_RECOVERY_SELECTED_MODE_AUX_KEY,
    PERSISTENT_RECOVERY_ACTIVE_AUX_KEY,
    SheafGBPPolicy,
)
from .single_slit_0720_bc import SingleSlitPolicy, architecture_record
from .single_slit_0720_eval import (
    DEFAULT_EVALUATION_HORIZON,
    STRICT_ANGLE_TOLERANCE_RAD,
    STRICT_OMEGA_TOLERANCE_RAD_PER_S,
    STRICT_POSITION_TOLERANCE_M,
    STRICT_SPEED_TOLERANCE_M_PER_S,
    STRICT_STABLE_STEPS,
    _communication_dimensions,
    _timeout_conditions,
    strict_final_components,
    strict_final_mask,
)
from .single_slit_contact_feedback import (
    CONTACT_FEEDBACK_CLEARANCE_RESERVE_M,
    single_slit_contact_feedback_step_v1,
)
from .single_slit_contact_observation import (
    SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1,
    SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
    SingleSlitContactObservationContextV1,
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
    single_slit_contact_observations_v1,
)
from .single_slit_feedback import single_slit_scenario


SINGLE_SLIT_CONTACT_FEEDBACK_EVAL_SCHEMA_V2 = (
    "piano-movers-single-slit-contact-feedback-closed-loop-eval-v2"
)
# V2 calls two or more immediately consecutive rejected proposals "stuck".
# This is an evaluator metric, not an observation channel or terminal event.
CONTACT_STUCK_THRESHOLD_STEPS_V2 = 2


def _updated_contact_context(
    previous: SingleSlitContactObservationContextV1,
    proposed: SingleSlitContactObservationContextV1,
    update: Tensor,
) -> SingleSlitContactObservationContextV1:
    """Update only rows that executed a valid physical proposal.

    The transition primitive is evaluated in a static batch, including rows
    that have terminated or produced invalid output.  Those rows did not
    execute a control proposal and therefore must not acquire a fabricated
    zero-force context.
    """

    if update.ndim != 1 or update.dtype != torch.bool:
        raise ValueError("contact-context update mask must be boolean [batch]")
    if update.shape != previous.last_hit.shape:
        raise ValueError("contact-context update mask has the wrong batch shape")
    local_mask = update[:, None, None]
    return SingleSlitContactObservationContextV1(
        last_hit=torch.where(update, proposed.last_hit, previous.last_hit),
        previous_force_normalized=torch.where(
            local_mask,
            proposed.previous_force_normalized,
            previous.previous_force_normalized,
        ),
        rejected_handle_motion_normalized=torch.where(
            local_mask,
            proposed.rejected_handle_motion_normalized,
            previous.rejected_handle_motion_normalized,
        ),
    )


def _optional_scalar(value: Tensor, index: int) -> float | None:
    scalar = value[index]
    if not bool(torch.isfinite(scalar)):
        return None
    return float(scalar.detach().cpu().item())


def _optional_population_min(value: Tensor) -> float | None:
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return None
    return float(finite.amin().detach().cpu().item())


def _optional_population_mean(value: Tensor) -> float | None:
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return None
    return float(finite.mean().detach().cpu().item())


@torch.no_grad()
def evaluate_single_slit_contact_feedback_v2(
    policy: SingleSlitPolicy,
    cfg: Any,
    *,
    initial_states: Tensor | None = None,
    initial_hidden: Tensor | None = None,
    episodes: int = 1,
    horizon: int = DEFAULT_EVALUATION_HORIZON,
    visible_aperture_m: float = 0.720,
    return_state_trace: bool = False,
    return_policy_trace: bool = False,
) -> dict[str, Any]:
    """Run the explicit-observation V2 contact-feedback protocol.

    Every active policy call advances valid recurrent state and incurs the
    policy's normal directed-edge payload cost, including calls that propose a
    blocked transition.  A contact neither terminates nor advances strict goal
    dwell: it resets dwell because the proposed control transition was not
    accepted.  Invalid policy output and horizon exhaustion remain failures.
    """

    if horizon <= 0:
        raise ValueError("evaluation horizon must be positive")
    if cfg.collision_substeps != 4:
        raise ValueError("single-slit evaluation requires four swept SAT substeps")
    if not cfg.strict_local_observation:
        raise ValueError("single-slit evaluation requires strict-local observations")
    if policy.arm_cfg.obs_dim != SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1:
        raise ValueError(
            "explicit-contact V2 requires policy.arm_cfg.obs_dim == "
            f"{SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1}"
        )
    if initial_states is None:
        if episodes <= 0:
            raise ValueError("evaluation episodes must be positive")
        scenario = single_slit_scenario(
            episodes, cfg, visible_aperture_m=visible_aperture_m
        )
    else:
        if initial_states.ndim != 2 or initial_states.shape[-1] != 6:
            raise ValueError("initial_states must have shape [episodes,6]")
        if initial_states.shape[0] <= 0:
            raise ValueError("initial_states must contain at least one episode")
        scenario = single_slit_scenario(
            int(initial_states.shape[0]),
            cfg,
            initial_state=initial_states,
            visible_aperture_m=visible_aperture_m,
        )
    state = scenario.state0.clone()
    episode_count = int(state.shape[0])
    edge_index = radius_edges(cfg, device=state.device)
    comm = _communication_dimensions(policy, edge_index)
    base_policy = getattr(policy, "policy", policy)
    cached_topology = (
        homogeneous_topology(
            edge_index,
            episode_count,
            cfg.n_agents,
            device=state.device,
        )
        if isinstance(base_policy, SheafGBPPolicy)
        else None
    )

    initial_collision = collision_mask(state, scenario, cfg)
    initial_clearance = collision_clearance_margin(state, scenario, cfg)
    if not torch.equal(initial_collision, initial_clearance <= 0.0):
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

    observation_schema = SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1
    contact_context = initial_single_slit_contact_observation_context_v1(state, cfg)
    alive = torch.ones(episode_count, dtype=torch.bool, device=state.device)
    succeeded = torch.zeros_like(alive)
    invalid_output = torch.zeros_like(alive)
    active_ticks = torch.zeros(episode_count, dtype=torch.long, device=state.device)
    finish_step = torch.full_like(active_ticks, horizon)
    failure_tick = torch.full_like(active_ticks, -1)
    failure_code = ["" for _ in range(episode_count)]

    dwell = torch.zeros_like(active_ticks)
    maximum_dwell = torch.zeros_like(active_ticks)
    accepted_motion_count = torch.zeros_like(active_ticks)

    contact_count = torch.zeros_like(active_ticks)
    consecutive_contact = torch.zeros_like(active_ticks)
    maximum_consecutive_contact = torch.zeros_like(active_ticks)
    consecutive_stuck = torch.zeros_like(active_ticks)
    maximum_consecutive_stuck = torch.zeros_like(active_ticks)
    stuck_tick_count = torch.zeros_like(active_ticks)
    stuck_event_count = torch.zeros_like(active_ticks)
    ever_stuck = torch.zeros_like(alive)

    infinity = torch.full(
        (episode_count,), float("inf"), dtype=state.dtype, device=state.device
    )
    minimum_attempted_clearance = infinity.clone()
    minimum_resolved_clearance = infinity.clone()
    contact_events: list[list[dict[str, Any]]] = [
        [] for _ in range(episode_count)
    ]
    state_trace = [state.detach().clone()] if return_state_trace else []
    policy_trace: dict[str, list[Tensor]] = (
        {
            "observation": [],
            "force": [],
            "mode_logits": [],
            "selected_mode": [],
            "local_plan_mean": [],
            "post_gbp_wrench": [],
            "recovery_active": [],
            "attempted_contact": [],
        }
        if return_policy_trace
        else {}
    )

    was_training = policy.training
    policy.eval()
    try:
        for tick in range(horizon):
            if not bool(alive.any()):
                break
            active = alive.clone()
            active_ticks += active.to(torch.long)
            policy_observation = single_slit_contact_observations_v1(
                state,
                scenario,
                cfg,
                contact_context,
            )
            if isinstance(policy, SheafGBPPolicy):
                output = policy(
                    policy_observation,
                    edge_index,
                    hidden,
                    topology=cached_topology,
                )
            else:
                output = policy(policy_observation, edge_index, hidden)
            if return_policy_trace:
                required_aux = {
                    "mode_logits": LOCAL_SE2_RECOVERY_MODE_LOGITS_AUX_KEY,
                    "selected_mode": LOCAL_SE2_RECOVERY_SELECTED_MODE_AUX_KEY,
                    "local_plan_mean": LOCAL_SE2_PLAN_MEAN_AUX_KEY,
                    "post_gbp_wrench": CORRECTION_WRENCH_AUX_KEY,
                    "recovery_active": PERSISTENT_RECOVERY_ACTIVE_AUX_KEY,
                }
                missing = [
                    key for key in required_aux.values() if key not in output.aux
                ]
                if missing:
                    raise RuntimeError(
                        "policy trace requested but auxiliary tensors are missing: "
                        + ", ".join(missing)
                    )
                policy_trace["observation"].append(
                    policy_observation.detach().clone()
                )
                policy_trace["force"].append(output.forces.detach().clone())
                for name, key in required_aux.items():
                    policy_trace[name].append(output.aux[key].detach().clone())
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
            invalid_output |= bad

            physics_active = active & valid_output
            hidden_mask = physics_active.view(
                episode_count, *([1] * (hidden.ndim - 1))
            )
            # Crucially, a blocked contact remains physics_active: its GBP /
            # recurrent result is carried into the next feedback decision.
            hidden = torch.where(hidden_mask, output.hidden, hidden)
            applied = torch.where(
                physics_active[:, None, None],
                output.forces,
                torch.zeros_like(output.forces),
            )
            transition = single_slit_contact_feedback_step_v1(
                state,
                applied,
                scenario,
                cfg,
            )
            proposed_context = single_slit_contact_observation_context_from_result_v1(
                transition,
                applied,
                cfg,
            )
            contact_context = _updated_contact_context(
                contact_context,
                proposed_context,
                physics_active,
            )
            contact = physics_active & transition.attempted_hit
            if return_policy_trace:
                policy_trace["attempted_contact"].append(contact.detach().clone())
            safe_accept = physics_active & ~contact
            minimum_attempted_clearance = torch.where(
                physics_active,
                torch.minimum(
                    minimum_attempted_clearance,
                    transition.attempted_clearance_m,
                ),
                minimum_attempted_clearance,
            )
            minimum_resolved_clearance = torch.where(
                physics_active,
                torch.minimum(
                    minimum_resolved_clearance,
                    transition.resolved_clearance_m,
                ),
                minimum_resolved_clearance,
            )
            if bool(contact.any()):
                for index in torch.where(contact)[0].tolist():
                    contact_events[index].append(
                        {
                            "event_type": "rejected_exact_sat_contact",
                            "control_tick": tick + 1,
                            "attempted_transition_clearance_m": float(
                                transition.attempted_clearance_m[index]
                                .detach()
                                .cpu()
                                .item()
                            ),
                            "attempted_state": [
                                float(value)
                                for value in transition.attempted_state[index]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "resolved_transition_clearance_m": float(
                                transition.resolved_clearance_m[index]
                                .detach()
                                .cpu()
                                .item()
                            ),
                            "resolved_state": [
                                float(value)
                                for value in transition.resolved_state[index]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "resolved_penetration": False,
                        }
                    )

            prior_consecutive_contact = consecutive_contact
            contact_count += contact.to(torch.long)
            accepted_motion_count += safe_accept.to(torch.long)
            consecutive_contact = torch.where(
                physics_active,
                torch.where(
                    contact,
                    consecutive_contact + 1,
                    torch.zeros_like(consecutive_contact),
                ),
                consecutive_contact,
            )
            maximum_consecutive_contact = torch.maximum(
                maximum_consecutive_contact, consecutive_contact
            )
            stuck_now = physics_active & (
                consecutive_contact >= CONTACT_STUCK_THRESHOLD_STEPS_V2
            )
            new_stuck_event = stuck_now & (
                prior_consecutive_contact < CONTACT_STUCK_THRESHOLD_STEPS_V2
            )
            consecutive_stuck = torch.where(
                physics_active,
                torch.where(
                    stuck_now,
                    consecutive_stuck + 1,
                    torch.zeros_like(consecutive_stuck),
                ),
                consecutive_stuck,
            )
            maximum_consecutive_stuck = torch.maximum(
                maximum_consecutive_stuck, consecutive_stuck
            )
            stuck_tick_count += stuck_now.to(torch.long)
            stuck_event_count += new_stuck_event.to(torch.long)
            ever_stuck |= stuck_now

            state = torch.where(
                physics_active[:, None], transition.resolved_state, state
            )
            if return_state_trace:
                state_trace.append(state.detach().clone())
            at_goal = safe_accept & strict_final_mask(state, scenario.goal)
            # Contact is safe after resolution but the proposed transition was
            # rejected, so it cannot count toward the strict accepted-step dwell.
            dwell = torch.where(
                physics_active,
                torch.where(at_goal, dwell + 1, torch.zeros_like(dwell)),
                dwell,
            )
            maximum_dwell = torch.maximum(maximum_dwell, dwell)
            reached = safe_accept & (dwell >= STRICT_STABLE_STEPS)
            succeeded |= reached
            terminal = bad | reached
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
        + invalid_output.to(torch.int8)
        + timed_out.to(torch.int8)
    )
    if not bool((outcome_partition == 1).all()):
        raise RuntimeError("contact-feedback outcomes do not form an exact partition")
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
        invalid = bool(invalid_output[index].item())
        timeout = bool(timed_out[index].item())
        if success:
            outcome = "success"
            reason: str | None = None
            failed_at: int | None = None
            conditions: list[str] = []
        elif invalid:
            outcome = "invalid_policy_output"
            reason = failure_code[index]
            failed_at = int(failure_tick[index].item())
            conditions = [reason]
        elif timeout:
            outcome = "timeout"
            conditions = _timeout_conditions(
                components, index, int(dwell[index].item())
            )
            if int(consecutive_contact[index].item()) >= CONTACT_STUCK_THRESHOLD_STEPS_V2:
                conditions.append("persistent_contact_stuck")
            reason = "timeout:" + ",".join(conditions)
            failed_at = horizon
        else:  # pragma: no cover - guarded by the partition assertion above.
            raise RuntimeError("unclassified contact-feedback outcome")
        ticks = int(active_ticks[index].item())
        events = contact_events[index]
        if len(events) != int(contact_count[index].item()):
            raise RuntimeError("contact trace length disagrees with contact count")
        per_episode.append(
            {
                "episode_index": index,
                "outcome": outcome,
                "success": success,
                # Penetration is impossible by construction; contact is separate.
                "collision": False,
                "collision_field_semantics": "resolved penetration only",
                "attempted_exact_sat_contact": bool(contact_count[index].item()),
                "attempted_exact_sat_contact_count": int(contact_count[index].item()),
                "resolved_penetration": False,
                "contact_occurred": bool(contact_count[index].item()),
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
                "accepted_motion_count": int(accepted_motion_count[index].item()),
                "contact_count": int(contact_count[index].item()),
                "final_consecutive_contact_steps": int(
                    consecutive_contact[index].item()
                ),
                "maximum_consecutive_contact_steps": int(
                    maximum_consecutive_contact[index].item()
                ),
                "stuck": bool(ever_stuck[index].item()),
                "stuck_event_count": int(stuck_event_count[index].item()),
                "stuck_tick_count": int(stuck_tick_count[index].item()),
                "final_consecutive_stuck_steps": int(
                    consecutive_stuck[index].item()
                ),
                "maximum_consecutive_stuck_steps": int(
                    maximum_consecutive_stuck[index].item()
                ),
                "minimum_attempted_exact_sat_transition_clearance_m": (
                    _optional_scalar(minimum_attempted_clearance, index)
                ),
                "minimum_resolved_exact_sat_transition_clearance_m": (
                    _optional_scalar(minimum_resolved_clearance, index)
                ),
                "initial_exact_sat_pose_clearance_m": float(
                    initial_clearance[index].detach().cpu().item()
                ),
                "final_resolved_state": [
                    float(value) for value in state[index].detach().cpu().tolist()
                ],
                "contact_events": events,
                "active_communication_ticks": ticks,
                "actual_directed_messages": setup_packets + ticks * packets_per_tick,
                "actual_directed_packets": setup_packets + ticks * packets_per_tick,
                "actual_payload_floats": setup_floats + ticks * floats_per_tick,
                "actual_payload_bytes": setup_bytes + ticks * bytes_per_tick,
                "actual_transmitted_payload_bytes": setup_bytes + ticks * bytes_per_tick,
                "failure_charged": True,
            }
        )

    success_count = int(succeeded.sum().item())
    invalid_count = int(invalid_output.sum().item())
    timeout_count = int(timed_out.sum().item())
    contact_episode_count = int((contact_count > 0).sum().item())
    stuck_episode_count = int(ever_stuck.sum().item())
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
    result = {
        "schema_version": SINGLE_SLIT_CONTACT_FEEDBACK_EVAL_SCHEMA_V2,
        "actor": policy.arm_name,
        "counts_as_policy": True,
        "teacher_used_at_policy_inference": False,
        "strict_local_observation_used": True,
        "observation_schema": observation_schema,
        "explicit_contact_observation_used": True,
        "carried_hidden_state_used": True,
        "hidden_advances_on_contact": True,
        "hidden_initialization": hidden_initialization,
        "actual_policy_forces_applied": True,
        "collision_authority": (
            "exact compound-rectangle SAT at four swept transition substeps"
        ),
        "contact_resolution": "reject attempted pose, retain prior pose, zero velocity",
        "contact_is_terminal": False,
        "penetration_is_admitted": False,
        "collision_field_semantics": (
            "resolved penetration only; rejected exact-SAT proposals are contact"
        ),
        "protocol": {
            "horizon_control_ticks": horizon,
            "episode_count": episode_count,
            "visible_aperture_m": float(visible_aperture_m),
            "contact_feedback": {
                "proposal_audit": "exact swept SAT x4",
                "numerical_clearance_reserve_m": (
                    CONTACT_FEEDBACK_CLEARANCE_RESERVE_M
                ),
                "blocked_pose_resolution": "prior safe pose",
                "blocked_velocity_resolution": "zero linear and angular velocity",
                "recurrent_hidden_update": "carried from blocked policy call",
                "communication_charged": True,
                "terminal": False,
                "strict_goal_dwell_on_contact": "reset_to_zero",
                "explicit_contact_observation": True,
                "explicit_contact_observation_schema": observation_schema,
                "explicit_contact_feature_order": [
                    "last_hit",
                    "previous_local_force_x_over_max_force",
                    "previous_local_force_y_over_max_force",
                    "rejected_handle_velocity_x_over_2mps",
                    "rejected_handle_velocity_y_over_2mps",
                ],
                "stuck_definition": (
                    "at least two immediately consecutive blocked proposals"
                ),
                "stuck_threshold_consecutive_contacts": (
                    CONTACT_STUCK_THRESHOLD_STEPS_V2
                ),
            },
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
        # Kept explicit for result-table compatibility; exact resolution makes
        # this structurally zero rather than redefining contact as collision.
        "collision_count": 0,
        "collision_rate": 0.0,
        "resolved_penetration_count": 0,
        "resolved_penetration_rate": 0.0,
        "attempted_exact_sat_contact_episode_count": contact_episode_count,
        "attempted_exact_sat_contact_episode_rate": (
            contact_episode_count / episode_count
        ),
        "attempted_exact_sat_contact_count": int(contact_count.sum().item()),
        "contact_episode_count": contact_episode_count,
        "contact_episode_rate": contact_episode_count / episode_count,
        "contact_count": int(contact_count.sum().item()),
        "mean_contacts_per_episode": float(contact_count.float().mean().item()),
        "maximum_consecutive_contact_steps": int(
            maximum_consecutive_contact.amax().item()
        ),
        "stuck_episode_count": stuck_episode_count,
        "stuck_episode_rate": stuck_episode_count / episode_count,
        "stuck_event_count": int(stuck_event_count.sum().item()),
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / episode_count,
        "invalid_policy_output_count": invalid_count,
        "minimum_attempted_exact_sat_transition_clearance_m": (
            _optional_population_min(minimum_attempted_clearance)
        ),
        "mean_minimum_attempted_exact_sat_transition_clearance_m": (
            _optional_population_mean(minimum_attempted_clearance)
        ),
        "minimum_resolved_exact_sat_transition_clearance_m": (
            _optional_population_min(minimum_resolved_clearance)
        ),
        "mean_minimum_resolved_exact_sat_transition_clearance_m": (
            _optional_population_mean(minimum_resolved_clearance)
        ),
        "communication": {
            "accounting": (
                "all actual active policy calls, including contact proposals and failed "
                "attempts; float32 payload only"
            ),
            "failure_charged_definition": (
                "total communication over every attempted episode divided by successful "
                "missions; failed attempts remain in the numerator"
            ),
            "contact_calls_charged": True,
            "failed_attempts_charged": True,
            "event_triggering": False,
            "message_headers_included": False,
            "payload_scalar_dtype": "float32",
            "payload_scalar_bytes": 4,
            **comm,
            "actual_active_control_ticks_total": total_active_ticks,
            "actual_directed_messages_total": total_directed_packets,
            "actual_directed_packets_total": total_directed_packets,
            "actual_payload_floats_total": total_payload_floats,
            "actual_payload_bytes_total": total_payload_bytes,
            "actual_transmitted_payload_bytes_total": total_payload_bytes,
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
    if return_state_trace:
        result["resolved_state_trace"] = torch.stack(state_trace, dim=1)
    if return_policy_trace:
        result["policy_trace"] = {
            key: torch.stack(values, dim=1) for key, values in policy_trace.items()
        }
    return result


__all__ = [
    "CONTACT_STUCK_THRESHOLD_STEPS_V2",
    "SINGLE_SLIT_CONTACT_FEEDBACK_EVAL_SCHEMA_V2",
    "evaluate_single_slit_contact_feedback_v2",
]
