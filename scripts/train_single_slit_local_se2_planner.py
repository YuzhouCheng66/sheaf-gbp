#!/usr/bin/env python3
"""Train private local SE(2) planning and waypoint-conditioned Sheaf force control."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from piano_movers.env import (
    net_wrench_from_forces,
    radius_edges,
    step_dynamics,
    transition_clearance_margin,
    wrap_angle,
)
from piano_movers.models import (
    CORRECTION_WRENCH_AUX_KEY,
    LOCAL_SE2_DIRECT_WRENCH_POST_AUX_KEY,
    LOCAL_SE2_PLAN_MEAN_AUX_KEY,
    PERSISTENT_RECOVERY_PHASE_LOGITS_AUX_KEY,
    SOURCE_FORCES_AUX_KEY,
)
from piano_movers.payload_shapes import payload_shape_rectangles
from piano_movers.single_slit_0720_bc import (
    SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
    architecture_record,
    communication_record,
    load_executable_teacher,
)
from piano_movers.single_slit_0720_eval import (
    STRICT_STABLE_STEPS,
    load_single_slit_checkpoint,
    strict_final_components,
    strict_final_mask,
)
from piano_movers.single_slit_contact_feedback import single_slit_contact_feedback_step_v1
from piano_movers.single_slit_contact_feedback_eval_v2 import evaluate_single_slit_contact_feedback_v2
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_dataset import load_executable_teacher_artifact
from piano_movers.single_slit_dagger import query_exact_local_se2_planning_teacher
from piano_movers.single_slit_feedback import (
    COLLISION_RESERVE_EACH_SIDE_M,
    GOAL_POSE,
    SOURCE_POSE,
    single_slit_feedback_observations,
    single_slit_scenario,
)
from piano_movers.single_slit_local_planner import (
    expand_plan_target_for_agents,
    local_se2_plan_target,
)
from scripts.bridge_single_slit_local_se2_planner import (
    BRIDGE_STAGE,
    planner_parameter_names,
)
from scripts.bridge_single_slit_analytic_total_wrench import (
    ANALYTIC_TOTAL_WRENCH_BRIDGE_STAGE,
)
from scripts.bridge_single_slit_direct_plan_wrench_unary import (
    BRIDGE_STAGE as DIRECT_WRENCH_BRIDGE_STAGE,
    PARAMETER_NAMES as DIRECT_WRENCH_PARAMETER_NAMES,
)
from scripts.bridge_single_slit_terminal_attractor import (
    TERMINAL_ATTRACTOR_BRIDGE_STAGE,
)
from scripts.bridge_single_slit_local_se2_recovery_adapter import (
    BRIDGE_STAGE as RECOVERY_ADAPTER_BRIDGE_STAGE,
    recovery_adapter_parameter_names,
)
from scripts.bridge_single_slit_analytic_phase_latch import (
    BRIDGE_STAGE as ANALYTIC_PHASE_LATCH_BRIDGE_STAGE,
)
from scripts.bridge_single_slit_canonical_recovery_encoder import (
    BRIDGE_STAGE as CANONICAL_RECOVERY_BRIDGE_STAGE,
)
from scripts.bridge_single_slit_contact_rebel_planner import (
    BRIDGE_STAGE as CONTACT_REBEL_BRIDGE_STAGE,
    contact_rebel_parameter_names,
)
from scripts.bridge_single_slit_exit_refinement_planner import (
    BRIDGE_STAGE as EXIT_REFINEMENT_BRIDGE_STAGE,
    exit_refinement_parameter_names,
)
from scripts.bridge_single_slit_approach_refinement_planner import (
    BRIDGE_STAGE as APPROACH_REFINEMENT_BRIDGE_STAGE,
    approach_refinement_parameter_names,
)
from scripts.bridge_single_slit_deep_stuck_refinement_planner import (
    BRIDGE_STAGE as DEEP_STUCK_REFINEMENT_BRIDGE_STAGE,
    deep_stuck_refinement_parameter_names,
)
from scripts.bridge_single_slit_retry_refinement_planner import (
    BRIDGE_STAGE as RETRY_REFINEMENT_BRIDGE_STAGE,
    retry_refinement_parameter_names,
)
from scripts.train_single_slit_multi_aperture_dagger import (
    LEFT_TO_RIGHT_PROGRESS_BIN_M,
    _load_trainable_interpolation,
)


TRAINING_STAGE = "private-local-se2-planner-waypoint-sheaf-v1"
PHASE_APPROACH = 0
PHASE_BRAKE = 1
PHASE_RETREAT = 2
PHASE_REENTER = 3
ORDERED_PATH_POSE_ERROR_BIN = 0.002


def _local_planning_phase(mode: str, post_contact_active: bool) -> int:
    """Map exact local-teacher primitives to the four causal skill phases."""

    if mode == "local_se2_brake":
        return PHASE_BRAKE
    if mode == "local_se2_recenter" or mode.startswith("local_se2_retreat_"):
        return PHASE_RETREAT
    return PHASE_REENTER if post_contact_active else PHASE_APPROACH


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_reference(policy: torch.nn.Module, selected: tuple[str, ...]) -> dict[str, Tensor]:
    trainable = set(selected)
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
        if name not in trainable
    }


def _assert_frozen(policy: torch.nn.Module, reference: Mapping[str, Tensor]) -> None:
    changed = [
        name
        for name, value in reference.items()
        if not torch.equal(policy.state_dict()[name].detach().cpu(), value)
    ]
    if changed:
        raise RuntimeError(f"local planner training changed frozen source tensors: {changed[:3]}")


@torch.no_grad()
def _collect_nominal(
    policy: torch.nn.Module,
    teacher: Any,
    artifact: Any,
    cfg: Any,
    aperture: float,
    *,
    perturbations_per_tick: int = 0,
    perturb_position_std_m: float = 0.01,
    perturb_angle_std_rad: float = 0.015,
    perturb_velocity_std: float = 0.04,
) -> dict[str, Tensor | int]:
    scenario = single_slit_scenario(
        1,
        cfg,
        initial_state=teacher.states[:1],
        visible_aperture_m=aperture,
    )
    edges = radius_edges(cfg, device=teacher.states.device)
    hidden = policy.init_hidden(1, teacher.states.device)
    context = initial_single_slit_contact_observation_context_v1(
        teacher.states[:1], cfg
    )
    progress = 0
    rows: dict[str, list[Tensor]] = {
        "observation": [],
        "hidden": [],
        "state": [],
        "target_force": [],
        "aperture_m": [],
        "last_hit": [],
        "contact_age": [],
        "local_se2_plan_target": [],
        "recovery_target": [],
        "post_contact_active": [],
        "local_se2_phase_target": [],
    }
    query_failures = 0
    was_training = policy.training
    policy.eval()
    try:
        for tick in range(teacher.horizon):
            state = teacher.states[tick : tick + 1]
            # Every label attached to this pre-action state must use the same
            # ordered-path memory.  Advancing progress before perturbation
            # queries would silently turn their one-step targets from t+1
            # into t+2.
            query_progress = progress
            observation = single_slit_contact_observations_v1(
                state, scenario, cfg, context
            )
            label, _error = query_exact_local_se2_planning_teacher(
                state,
                query_progress,
                artifact,
                cfg,
                visible_aperture_m=aperture,
            )
            if label is None:
                query_failures += 1
            else:
                rows["observation"].append(observation[0].clone())
                rows["hidden"].append(hidden[0].clone())
                rows["state"].append(state[0].clone())
                rows["target_force"].append(label.forces.clone())
                rows["aperture_m"].append(state.new_tensor(aperture))
                rows["last_hit"].append(context.last_hit.to(state.dtype)[0].clone())
                rows["contact_age"].append(
                    torch.zeros((), dtype=torch.long, device=state.device)
                )
                rows["local_se2_plan_target"].append(
                    local_se2_plan_target(state, label)
                )
                rows["recovery_target"].append(
                    torch.tensor(
                        label.mode != "local_se2_forward",
                        dtype=torch.bool,
                        device=state.device,
                    )
                )
                rows["post_contact_active"].append(
                    torch.tensor(False, dtype=torch.bool, device=state.device)
                )
                rows["local_se2_phase_target"].append(
                    torch.tensor(
                        _local_planning_phase(label.mode, False),
                        dtype=torch.long,
                        device=state.device,
                    )
                )
                for _sample in range(perturbations_per_tick):
                    perturbed = state.clone()
                    perturbed[:, :2] += perturb_position_std_m * torch.randn_like(
                        perturbed[:, :2]
                    )
                    perturbed[:, 2] = wrap_angle(
                        perturbed[:, 2]
                        + perturb_angle_std_rad * torch.randn_like(perturbed[:, 2])
                    )
                    perturbed[:, 3:] += perturb_velocity_std * torch.randn_like(
                        perturbed[:, 3:]
                    )
                    perturbed_label, _perturbed_error = query_exact_local_se2_planning_teacher(
                        perturbed,
                        query_progress,
                        artifact,
                        cfg,
                        visible_aperture_m=aperture,
                    )
                    if perturbed_label is None:
                        query_failures += 1
                        continue
                    perturbed_scenario = single_slit_scenario(
                        1,
                        cfg,
                        initial_state=perturbed,
                        visible_aperture_m=aperture,
                    )
                    perturbed_observation = single_slit_contact_observations_v1(
                        perturbed, perturbed_scenario, cfg, context
                    )
                    rows["observation"].append(perturbed_observation[0].clone())
                    rows["hidden"].append(hidden[0].clone())
                    rows["state"].append(perturbed[0].clone())
                    rows["target_force"].append(perturbed_label.forces.clone())
                    rows["aperture_m"].append(perturbed.new_tensor(aperture))
                    rows["last_hit"].append(
                        context.last_hit.to(perturbed.dtype)[0].clone()
                    )
                    rows["contact_age"].append(
                        torch.zeros((), dtype=torch.long, device=perturbed.device)
                    )
                    rows["local_se2_plan_target"].append(
                        local_se2_plan_target(perturbed, perturbed_label)
                    )
                    rows["recovery_target"].append(
                        torch.tensor(
                            perturbed_label.mode != "local_se2_forward",
                            dtype=torch.bool,
                            device=perturbed.device,
                        )
                    )
                    rows["post_contact_active"].append(
                        torch.tensor(False, dtype=torch.bool, device=state.device)
                    )
                    rows["local_se2_phase_target"].append(
                        torch.tensor(
                            _local_planning_phase(perturbed_label.mode, False),
                            dtype=torch.long,
                            device=state.device,
                        )
                    )
                progress = max(progress, label.nearest_index)
            output = policy(observation, edges, hidden)
            hidden = output.hidden
            transition = single_slit_contact_feedback_step_v1(
                state,
                teacher.forces[tick : tick + 1],
                scenario,
                cfg,
            )
            if bool(transition.attempted_hit[0]):
                raise RuntimeError("nominal teacher transition collided at bootstrap aperture")
            context = single_slit_contact_observation_context_from_result_v1(
                transition,
                teacher.forces[tick : tick + 1],
                cfg,
            )
    finally:
        policy.train(was_training)
    if not rows["observation"]:
        raise RuntimeError("nominal planner collection produced no rows")
    result: dict[str, Tensor | int] = {
        name: torch.stack(values) for name, values in rows.items()
    }
    result["query_failures"] = query_failures
    return result


@torch.no_grad()
def _collect_oracle_plan_induced(
    policy: torch.nn.Module,
    artifact: Any,
    cfg: Any,
    aperture: float,
    horizon: int,
) -> dict[str, Tensor | int]:
    """Collect states induced by the controller given only oracle waypoints.

    Teacher forces label each visited pre-action state but are never executed.
    The policy executes its own handle forces conditioned on the teacher's
    short-horizon SE(2) target.  This isolates controller covariate shift from
    errors in the private local waypoint predictor.
    """

    scenario = single_slit_scenario(1, cfg, visible_aperture_m=aperture)
    state = scenario.state0.clone()
    edges = radius_edges(cfg, device=state.device)
    hidden = policy.init_hidden(1, state.device)
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    contact_age = torch.zeros(1, dtype=torch.long, device=state.device)
    progress = 0
    rows: dict[str, list[Tensor]] = {
        "observation": [],
        "hidden": [],
        "state": [],
        "target_force": [],
        "aperture_m": [],
        "last_hit": [],
        "contact_age": [],
        "local_se2_plan_target": [],
        "recovery_target": [],
        "post_contact_active": [],
        "local_se2_phase_target": [],
        "ordered_progress": [],
    }
    query_failures = 0
    recovery_active = False
    was_training = policy.training
    policy.eval()
    try:
        for _tick in range(horizon):
            recovery_active = recovery_active or bool(context.last_hit[0])
            observation = single_slit_contact_observations_v1(
                state, scenario, cfg, context
            )
            label, _error = query_exact_local_se2_planning_teacher(
                state,
                progress,
                artifact,
                cfg,
                visible_aperture_m=aperture,
            )
            if label is None:
                query_failures += 1
                break
            progress = max(progress, label.nearest_index)
            target_plan = local_se2_plan_target(state, label)
            rows["observation"].append(observation[0].clone())
            rows["hidden"].append(hidden[0].clone())
            rows["state"].append(state[0].clone())
            rows["target_force"].append(label.forces.clone())
            rows["aperture_m"].append(state.new_tensor(aperture))
            rows["last_hit"].append(context.last_hit.to(state.dtype)[0].clone())
            rows["contact_age"].append(contact_age[0].clone())
            rows["local_se2_plan_target"].append(target_plan)
            rows["recovery_target"].append(
                torch.tensor(
                    label.mode != "local_se2_forward" or recovery_active,
                    dtype=torch.bool,
                    device=state.device,
                )
            )
            rows["post_contact_active"].append(
                torch.tensor(
                    recovery_active,
                    dtype=torch.bool,
                    device=state.device,
                )
            )
            rows["local_se2_phase_target"].append(
                torch.tensor(
                    _local_planning_phase(label.mode, recovery_active),
                    dtype=torch.long,
                    device=state.device,
                )
            )
            rows["ordered_progress"].append(
                torch.tensor(progress, dtype=torch.long, device=state.device)
            )
            override = expand_plan_target_for_agents(
                target_plan.view(1, -1), cfg.n_agents
            )
            output = policy(
                observation,
                edges,
                hidden,
                local_se2_plan_override=override,
            )
            transition = single_slit_contact_feedback_step_v1(
                state, output.forces, scenario, cfg
            )
            context = single_slit_contact_observation_context_from_result_v1(
                transition, output.forces, cfg
            )
            state = transition.resolved_state
            hidden = output.hidden
            contact_age = torch.where(
                transition.attempted_hit,
                contact_age + 1,
                torch.zeros_like(contact_age),
            )
    finally:
        policy.train(was_training)
    if not rows["observation"]:
        raise RuntimeError("oracle-plan DAgger collected no safe teacher labels")
    result: dict[str, Tensor | int] = {
        name: torch.stack(values) for name, values in rows.items()
    }
    result["query_failures"] = query_failures
    return result


@torch.no_grad()
def _collect_corrective_suffix_perturbations(
    suffix: Mapping[str, Tensor | int],
    artifact: Any,
    cfg: Any,
    aperture: float,
    *,
    perturbations_per_tick: int,
    perturb_position_std_m: float,
    perturb_angle_std_rad: float,
    perturb_velocity_std: float,
) -> dict[str, Tensor | int]:
    """Label a local tube around the *post-contact successful* suffix.

    Pre-contact augmentation cannot train a source-null recovery adapter.  A
    small tube around the teacher-executed corrective suffix supplies the
    missing off-trajectory recovery states while retaining the exact carried
    hidden/contact context and ordered-path memory of each source row.
    """

    if perturbations_per_tick <= 0:
        raise ValueError("corrective suffix perturbation count must be positive")
    states = suffix["state"]
    observations = suffix["observation"]
    hidden = suffix["hidden"]
    progress = suffix["ordered_progress"]
    contact_age = suffix["contact_age"]
    if not all(
        isinstance(value, Tensor)
        for value in (states, observations, hidden, progress, contact_age)
    ):
        raise TypeError("corrective suffix omitted tensor state/context")
    rows: dict[str, list[Tensor]] = {
        "observation": [],
        "hidden": [],
        "state": [],
        "target_force": [],
        "aperture_m": [],
        "last_hit": [],
        "contact_age": [],
        "local_se2_plan_target": [],
        "recovery_target": [],
        "post_contact_active": [],
        "local_se2_phase_target": [],
        "ordered_progress": [],
    }
    query_failures = 0
    for row in range(int(states.shape[0])):
        source_state = states[row : row + 1]
        feedback_tail = observations[row : row + 1, :, 22:]
        for _sample in range(perturbations_per_tick):
            perturbed = source_state.clone()
            perturbed[:, :2] += perturb_position_std_m * torch.randn_like(
                perturbed[:, :2]
            )
            perturbed[:, 2] = wrap_angle(
                perturbed[:, 2]
                + perturb_angle_std_rad * torch.randn_like(perturbed[:, 2])
            )
            perturbed[:, 3:] += perturb_velocity_std * torch.randn_like(
                perturbed[:, 3:]
            )
            label, _error = query_exact_local_se2_planning_teacher(
                perturbed,
                int(progress[row]),
                artifact,
                cfg,
                visible_aperture_m=aperture,
            )
            if label is None:
                query_failures += 1
                continue
            perturbed_scenario = single_slit_scenario(
                1,
                cfg,
                initial_state=perturbed,
                visible_aperture_m=aperture,
            )
            base = single_slit_feedback_observations(
                perturbed, perturbed_scenario, cfg
            )
            observation = torch.cat((base, feedback_tail), dim=-1)
            target_plan = local_se2_plan_target(perturbed, label)
            rows["observation"].append(observation[0].clone())
            rows["hidden"].append(hidden[row].clone())
            rows["state"].append(perturbed[0].clone())
            rows["target_force"].append(label.forces.clone())
            rows["aperture_m"].append(perturbed.new_tensor(aperture))
            rows["last_hit"].append(feedback_tail[0, 0, 0].clone())
            rows["contact_age"].append(contact_age[row].clone())
            rows["local_se2_plan_target"].append(target_plan)
            rows["recovery_target"].append(
                torch.tensor(True, dtype=torch.bool, device=perturbed.device)
            )
            rows["post_contact_active"].append(
                torch.tensor(True, dtype=torch.bool, device=perturbed.device)
            )
            rows["local_se2_phase_target"].append(
                torch.tensor(
                    _local_planning_phase(label.mode, True),
                    dtype=torch.long,
                    device=perturbed.device,
                )
            )
            rows["ordered_progress"].append(progress[row].clone())
    if not rows["observation"]:
        raise RuntimeError("corrective suffix perturbations produced no exact labels")
    result: dict[str, Tensor | int] = {
        name: torch.stack(values) for name, values in rows.items()
    }
    result["query_failures"] = query_failures
    return result


def _project_model_induced_progress(
    state: Tensor,
    artifact: Any,
    current_index: int,
    *,
    maximum_forward_search: int = 8,
) -> int:
    """Update privileged path memory from the state actually reached.

    A teacher proposal is not an executed action during DAgger, so its target
    index must never advance progress.  Projection is restricted to a small
    forward window and remains monotone; explicit recovery targets may point
    to earlier fibers without falsifying this memory.
    """

    right = min(int(artifact.horizon), current_index + maximum_forward_search)
    candidates = artifact.reference_poses[current_index : right + 1]
    delta_xy = candidates[:, :2] - state[0, :2]
    delta_theta = wrap_angle(candidates[:, 2] - state[0, 2])
    distance2 = delta_xy.square().sum(dim=-1) + (0.35 * delta_theta).square()
    return current_index + int(distance2.argmin().item())


@torch.no_grad()
def _collect_model_induced_local_plans(
    policy: torch.nn.Module,
    artifact: Any,
    cfg: Any,
    aperture: float,
    horizon: int,
    *,
    initial_state: Tensor | None = None,
    stop_after_consecutive_contact_steps: int = 0,
) -> dict[str, Tensor | int]:
    """Label the learned planner's own recurrent states with exact local plans."""

    if initial_state is not None and initial_state.shape != (1, 6):
        raise ValueError("model-induced initial_state must have shape [1,6]")
    if stop_after_consecutive_contact_steps < 0:
        raise ValueError("contact-tail collection limit must be non-negative")
    scenario = single_slit_scenario(
        1,
        cfg,
        initial_state=initial_state,
        visible_aperture_m=aperture,
    )
    state = scenario.state0.clone()
    edges = radius_edges(cfg, device=state.device)
    hidden = policy.init_hidden(1, state.device)
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    contact_age = torch.zeros(1, dtype=torch.long, device=state.device)
    progress = 0
    rows: dict[str, list[Tensor]] = {
        "observation": [],
        "hidden": [],
        "state": [],
        "target_force": [],
        "aperture_m": [],
        "last_hit": [],
        "contact_age": [],
        "local_se2_plan_target": [],
        "recovery_target": [],
        "post_contact_active": [],
        "local_se2_phase_target": [],
        "ordered_progress": [],
    }
    query_failures = 0
    recovery_active = False
    was_training = policy.training
    policy.eval()
    try:
        for _tick in range(horizon):
            progress = _project_model_induced_progress(state, artifact, progress)
            recovery_active = recovery_active or bool(context.last_hit[0])
            observation = single_slit_contact_observations_v1(
                state,
                scenario,
                cfg,
                context,
            )
            label, _error = query_exact_local_se2_planning_teacher(
                state,
                progress,
                artifact,
                cfg,
                visible_aperture_m=aperture,
            )
            if label is None:
                query_failures += 1
            else:
                rows["observation"].append(observation[0].clone())
                rows["hidden"].append(hidden[0].clone())
                rows["state"].append(state[0].clone())
                rows["target_force"].append(label.forces.clone())
                rows["aperture_m"].append(state.new_tensor(aperture))
                rows["last_hit"].append(context.last_hit.to(state.dtype)[0].clone())
                rows["contact_age"].append(contact_age[0].clone())
                rows["local_se2_plan_target"].append(
                    local_se2_plan_target(state, label)
                )
                rows["recovery_target"].append(
                    torch.tensor(
                        label.mode != "local_se2_forward" or recovery_active,
                        dtype=torch.bool,
                        device=state.device,
                    )
                )
                rows["post_contact_active"].append(
                    torch.tensor(
                        recovery_active,
                        dtype=torch.bool,
                        device=state.device,
                    )
                )
                rows["local_se2_phase_target"].append(
                    torch.tensor(
                        _local_planning_phase(label.mode, recovery_active),
                        dtype=torch.long,
                        device=state.device,
                    )
                )
                rows["ordered_progress"].append(
                    torch.tensor(
                        progress, dtype=torch.long, device=state.device
                    )
                )
            output = policy(observation, edges, hidden)
            transition = single_slit_contact_feedback_step_v1(
                state,
                output.forces,
                scenario,
                cfg,
            )
            context = single_slit_contact_observation_context_from_result_v1(
                transition,
                output.forces,
                cfg,
            )
            state = transition.resolved_state
            hidden = output.hidden
            contact_age = torch.where(
                transition.attempted_hit,
                contact_age + 1,
                torch.zeros_like(contact_age),
            )
            if (
                stop_after_consecutive_contact_steps > 0
                and int(contact_age[0].item())
                >= stop_after_consecutive_contact_steps
            ):
                break
    finally:
        policy.train(was_training)
    if not rows["observation"]:
        raise RuntimeError("model-induced planner collection produced no exact labels")
    result: dict[str, Tensor | int] = {
        name: torch.stack(values) for name, values in rows.items()
    }
    result["query_failures"] = query_failures
    return result


@torch.no_grad()
def _collect_teacher_forced_recovery_suffix(
    policy: torch.nn.Module,
    artifact: Any,
    cfg: Any,
    aperture: float,
    horizon: int,
) -> dict[str, Tensor | int]:
    """Collect a complete corrective demonstration from the policy's own hit.

    The deployable policy first produces a factual prefix and a real rejected
    contact proposal.  Starting from that exact resolved state and carried
    hidden state, the privileged teacher supplies only the next local SE(2)
    plan; the unchanged Sheaf-GBP force layer executes that plan.  This gives
    recurrent training the retreat *and* re-entry states that a failed policy
    cannot visit by itself.  It is deliberately stored as a separate
    teacher-forced corrective suffix, never described as pure DAgger.
    """

    scenario = single_slit_scenario(1, cfg, visible_aperture_m=aperture)
    state = scenario.state0.clone()
    edges = radius_edges(cfg, device=state.device)
    hidden = policy.init_hidden(1, state.device)
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    contact_age = torch.zeros(1, dtype=torch.long, device=state.device)
    progress = 0
    trigger_tick = -1
    rows: dict[str, list[Tensor]] = {
        "observation": [],
        "hidden": [],
        "state": [],
        "target_force": [],
        "aperture_m": [],
        "last_hit": [],
        "contact_age": [],
        "local_se2_plan_target": [],
        "recovery_target": [],
        "post_contact_active": [],
        "local_se2_phase_target": [],
        "ordered_progress": [],
    }
    query_failures = 0
    stable = 0
    succeeded = False
    was_training = policy.training
    policy.eval()
    try:
        # Factual learned prefix: no teacher plan or force is executed here.
        for tick in range(horizon):
            progress = _project_model_induced_progress(state, artifact, progress)
            observation = single_slit_contact_observations_v1(
                state, scenario, cfg, context
            )
            output = policy(observation, edges, hidden)
            transition = single_slit_contact_feedback_step_v1(
                state, output.forces, scenario, cfg
            )
            context = single_slit_contact_observation_context_from_result_v1(
                transition, output.forces, cfg
            )
            state = transition.resolved_state
            hidden = output.hidden
            contact_age = torch.where(
                transition.attempted_hit,
                contact_age + 1,
                torch.zeros_like(contact_age),
            )
            if bool(transition.attempted_hit[0]):
                trigger_tick = tick
                break
        if trigger_tick < 0:
            raise RuntimeError(
                "teacher-forced recovery suffix requires a factual policy contact"
            )

        # Corrective suffix: teacher chooses a safe local pose; the ordinary
        # frozen Sheaf-GBP layer remains the sole force-producing actor.
        remaining = max(1, horizon - trigger_tick - 1)
        for _suffix_tick in range(remaining):
            progress = _project_model_induced_progress(state, artifact, progress)
            query_progress = progress
            observation = single_slit_contact_observations_v1(
                state, scenario, cfg, context
            )
            label, _error = query_exact_local_se2_planning_teacher(
                state,
                query_progress,
                artifact,
                cfg,
                visible_aperture_m=aperture,
            )
            if label is None:
                query_failures += 1
                break
            progress = max(progress, label.nearest_index)
            target_plan = local_se2_plan_target(state, label)
            rows["observation"].append(observation[0].clone())
            rows["hidden"].append(hidden[0].clone())
            rows["state"].append(state[0].clone())
            rows["target_force"].append(label.forces.clone())
            rows["aperture_m"].append(state.new_tensor(aperture))
            rows["last_hit"].append(context.last_hit.to(state.dtype)[0].clone())
            rows["contact_age"].append(contact_age[0].clone())
            rows["local_se2_plan_target"].append(target_plan)
            rows["recovery_target"].append(
                torch.tensor(True, dtype=torch.bool, device=state.device)
            )
            rows["post_contact_active"].append(
                torch.tensor(True, dtype=torch.bool, device=state.device)
            )
            rows["local_se2_phase_target"].append(
                torch.tensor(
                    _local_planning_phase(label.mode, True),
                    dtype=torch.long,
                    device=state.device,
                )
            )
            rows["ordered_progress"].append(
                torch.tensor(
                    query_progress, dtype=torch.long, device=state.device
                )
            )
            override = expand_plan_target_for_agents(
                target_plan.view(1, -1), cfg.n_agents
            )
            output = policy(
                observation,
                edges,
                hidden,
                local_se2_plan_override=override,
            )
            transition = single_slit_contact_feedback_step_v1(
                state, output.forces, scenario, cfg
            )
            context = single_slit_contact_observation_context_from_result_v1(
                transition, output.forces, cfg
            )
            state = transition.resolved_state
            hidden = output.hidden
            contact_age = torch.where(
                transition.attempted_hit,
                contact_age + 1,
                torch.zeros_like(contact_age),
            )
            stable = (
                stable + 1
                if bool(strict_final_mask(state, scenario.goal)[0])
                else 0
            )
            if stable >= STRICT_STABLE_STEPS:
                succeeded = True
                break
    finally:
        policy.train(was_training)
    if not rows["observation"]:
        raise RuntimeError("teacher-forced recovery suffix produced no labels")
    result: dict[str, Tensor | int] = {
        name: torch.stack(values) for name, values in rows.items()
    }
    result["query_failures"] = query_failures
    result["teacher_suffix_success"] = int(succeeded)
    result["teacher_suffix_trigger_tick"] = trigger_tick
    result["teacher_suffix_finish_tick"] = trigger_tick + len(rows["observation"])
    return result


def _merge_replay(*parts: Mapping[str, Tensor | int]) -> dict[str, Tensor]:
    tensor_keys = (
        "observation",
        "hidden",
        "state",
        "target_force",
        "aperture_m",
        "last_hit",
        "contact_age",
        "local_se2_plan_target",
        "recovery_target",
        "post_contact_active",
        "local_se2_phase_target",
    )
    merged = {
        name: torch.cat([part[name] for part in parts], dim=0)  # type: ignore[list-item]
        for name in tensor_keys
    }
    rows = int(merged["observation"].shape[0])
    if any(int(value.shape[0]) != rows for value in merged.values()):
        raise RuntimeError("merged local planner replay has inconsistent row counts")
    return merged


def _move_replay_part(
    part: Mapping[str, Tensor | int], device: torch.device
) -> dict[str, Tensor | int]:
    """Move one authenticated collection to the training device in bulk."""

    return {
        name: value.to(device) if isinstance(value, Tensor) else value
        for name, value in part.items()
    }


@torch.no_grad()
def _cache_incumbent_local_plans(
    policy: torch.nn.Module,
    replay: Mapping[str, Tensor],
    *,
    chunk_size: int = 256,
) -> Tensor:
    """Cache the source-correct planner output before an optimizer proposal."""

    chunks = []
    rows = int(replay["observation"].shape[0])
    was_training = policy.training
    policy.eval()
    try:
        for left in range(0, rows, chunk_size):
            right = min(rows, left + chunk_size)
            chunks.append(
                policy.predict_local_se2_plan_mean(
                    replay["observation"][left:right],
                    replay["hidden"][left:right],
                ).clone()
            )
    finally:
        policy.train(was_training)
    return torch.cat(chunks, dim=0)


def _source_correct_recovery_plan_loss(
    predicted_plan: Tensor,
    exact_target: Tensor,
    incumbent_plan: Tensor,
    recovery_target: Tensor,
    *,
    preservation_weight: float,
    recovery_dimension_weights: Sequence[float] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit recovery rows while preserving the incumbent on learned-safe rows."""

    if recovery_target.ndim != 1 or recovery_target.dtype != torch.bool:
        raise ValueError("recovery_target must be a boolean row mask")
    if predicted_plan.shape != exact_target.shape or predicted_plan.shape != incumbent_plan.shape:
        raise ValueError("source-correct planner tensors must have identical shapes")
    if predicted_plan.shape[0] != recovery_target.shape[0]:
        raise ValueError("source-correct mask length differs from planner batch")
    exact_square = (predicted_plan - exact_target).square()
    if recovery_dimension_weights is not None:
        dimension_weights = predicted_plan.new_tensor(recovery_dimension_weights)
        if dimension_weights.shape != (predicted_plan.shape[-1],):
            raise ValueError("recovery dimension weights must match the plan width")
        if bool((dimension_weights < 0.0).any()) or not bool(
            (dimension_weights > 0.0).any()
        ):
            raise ValueError("recovery dimension weights must be nonnegative and nonzero")
        # Preserve the all-ones loss scale while allowing the executable twist
        # coordinates (3:6) to receive more of the recovery gradient.
        exact_square = exact_square * (
            dimension_weights / dimension_weights.mean()
        ).view(1, 1, -1)
    exact_error = exact_square.mean(dim=(1, 2))
    preservation_error = (predicted_plan - incumbent_plan).square().mean(dim=(1, 2))
    if bool(recovery_target.any()):
        recovery_loss = exact_error[recovery_target].mean()
    else:
        recovery_loss = exact_error.new_zeros(())
    preserve = ~recovery_target
    if bool(preserve.any()):
        preservation_loss = preservation_error[preserve].mean()
    else:
        preservation_loss = preservation_error.new_zeros(())
    total = recovery_loss + preservation_weight * preservation_loss
    return total, recovery_loss, preservation_loss


def _recurrent_sequence_plan_loss(
    policy: torch.nn.Module,
    sequences: list[Mapping[str, Tensor | int]],
    cfg: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    """Unroll the learned recovery state over complete DAgger trajectories."""

    if not sequences:
        raise ValueError("recurrent sequence training requires induced trajectories")
    lengths = [int(sequence["observation"].shape[0]) for sequence in sequences]
    horizon = max(lengths)
    first_observation = sequences[0]["observation"]
    first_hidden = sequences[0]["hidden"]
    batch = len(sequences)
    observation = first_observation.new_zeros(
        batch, horizon, *first_observation.shape[1:]
    )
    stored_hidden = first_hidden.new_zeros(
        batch, horizon, *first_hidden.shape[1:]
    )
    target = first_observation.new_zeros(
        batch, horizon, policy.arm_cfg.local_se2_plan_dim
    )
    post_contact = torch.zeros(
        batch, horizon, dtype=torch.bool, device=first_observation.device
    )
    phase_target = torch.zeros(
        batch, horizon, dtype=torch.long, device=first_observation.device
    )
    for row, (sequence, length) in enumerate(zip(sequences, lengths, strict=True)):
        observation[row, :length] = sequence["observation"]
        stored_hidden[row, :length] = sequence["hidden"]
        target[row, :length] = sequence["local_se2_plan_target"]
        post_contact[row, :length] = sequence["post_contact_active"]
        phase_target[row, :length] = sequence["local_se2_phase_target"]
    batch, horizon, n_agents, _obs_dim = observation.shape
    if n_agents != cfg.n_agents:
        raise ValueError("recurrent sequence agent count differs from environment")
    recovery = observation.new_zeros(
        batch,
        n_agents,
        policy.arm_cfg.persistent_recovery_dim,
    )
    legacy_width = policy.legacy_hidden_floats_per_agent()
    plan_sum = observation.new_zeros(batch)
    phase_sum = observation.new_zeros(batch)
    correct_sum = observation.new_zeros(batch)
    active_rows = post_contact.to(observation.dtype).sum(dim=1)
    if bool((active_rows <= 0).any()):
        raise RuntimeError("recurrent sequence archive contains no post-contact rows")
    for tick in range(horizon):
        legacy = stored_hidden[:, tick, :, :legacy_width]
        predicted, logits, recovery = policy.predict_local_se2_recurrent_step(
            observation[:, tick],
            legacy,
            recovery,
        )
        active_weight = post_contact[:, tick].to(observation.dtype)
        expanded_target = target[:, tick, None, :].expand(-1, n_agents, -1)
        plan_sum = plan_sum + (
            (predicted - expanded_target).square()
            * active_weight[:, None, None]
        ).sum(dim=(1, 2))
        expanded_phase = phase_target[:, tick, None].expand(-1, n_agents)
        phase_per_agent = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            expanded_phase.reshape(-1),
            reduction="none",
        ).reshape(batch, n_agents)
        phase_sum = phase_sum + (
            phase_per_agent * active_weight[:, None]
        ).sum(dim=1)
        correct_sum = correct_sum + (
            (logits.argmax(dim=-1) == expanded_phase).to(observation.dtype)
            * active_weight[:, None]
        ).sum(dim=1)
    weights = observation.new_tensor(
        [float(sequence.get("sequence_weight", 1.0)) for sequence in sequences]
    )
    if bool((~torch.isfinite(weights)).any()) or bool((weights <= 0.0).any()):
        raise ValueError("recurrent sequence weights must be finite and positive")
    weights = weights / weights.sum()
    plan_per_sequence = plan_sum / (
        active_rows * n_agents * policy.arm_cfg.local_se2_plan_dim
    )
    phase_per_sequence = phase_sum / (active_rows * n_agents)
    accuracy_per_sequence = correct_sum / (active_rows * n_agents)
    plan_loss = (weights * plan_per_sequence).sum()
    phase_loss = (weights * phase_per_sequence).sum()
    phase_accuracy = (weights * accuracy_per_sequence).sum()
    return plan_loss, phase_loss, phase_accuracy


def _corrective_suffix_multiple_shooting_loss(
    policy: torch.nn.Module,
    sequence: Mapping[str, Tensor | int],
    cfg: Any,
    aperture: float,
    *,
    rollout_horizon: int,
    window_count: int,
) -> tuple[Tensor, dict[str, float]]:
    """Backpropagate through actual Sheaf-GBP actions for short suffix windows.

    Each window starts on the exact successful corrective suffix with its
    factual carried hidden state.  Subsequent observations are recomputed from
    the policy-induced payload state, and the unchanged full policy produces
    forces for every step.  A first colliding proposal contributes state and
    clearance loss but is rejected exactly as in evaluation.
    """

    states = sequence["state"]
    stored_observation = sequence["observation"]
    stored_hidden = sequence["hidden"]
    if not isinstance(states, Tensor) or not isinstance(stored_observation, Tensor):
        raise TypeError("corrective suffix omitted tensor state/observation")
    if not isinstance(stored_hidden, Tensor):
        raise TypeError("corrective suffix omitted tensor hidden state")
    rows = int(states.shape[0])
    maximum_start = rows - rollout_horizon - 1
    if maximum_start < 0:
        raise ValueError("multi-step horizon exceeds the corrective suffix")
    count = min(window_count, maximum_start + 1)
    starts = torch.linspace(
        0,
        maximum_start,
        count,
        device=states.device,
        dtype=states.dtype,
    ).round().to(torch.long)
    current = states.index_select(0, starts).clone()
    hidden = stored_hidden.index_select(0, starts).clone()
    scenario = single_slit_scenario(
        count,
        cfg,
        initial_state=current,
        visible_aperture_m=aperture,
    )
    edges = radius_edges(cfg, device=states.device)
    state_scale = states.new_tensor((0.03, 0.03, 0.05, 0.10, 0.10, 0.12))
    alive = torch.ones(count, dtype=torch.bool, device=states.device)
    state_sum = states.new_zeros(())
    clearance_sum = states.new_zeros(())
    collision_sum = states.new_zeros(())
    minimum_clearance = states.new_tensor(float("inf"))
    active_count = states.new_zeros(())
    for offset in range(rollout_horizon):
        base = single_slit_feedback_observations(current, scenario, cfg)
        feedback = torch.zeros(
            count,
            cfg.n_agents,
            5,
            device=states.device,
            dtype=states.dtype,
        )
        if offset == 0:
            feedback = stored_observation.index_select(0, starts)[..., 22:].clone()
        observation = torch.cat((base, feedback), dim=-1)
        output = policy(observation, edges, hidden)
        proposed = step_dynamics(current, output.forces, cfg)
        target_index = starts + offset + 1
        target = states.index_select(0, target_index).detach()
        error = proposed - target
        error = torch.cat(
            (
                error[:, :2],
                wrap_angle(error[:, 2]).unsqueeze(-1),
                error[:, 3:],
            ),
            dim=-1,
        )
        active_float = alive.to(states.dtype)
        state_sum = state_sum + (
            (error / state_scale).square().mean(dim=-1) * active_float
        ).sum()
        clearance = transition_clearance_margin(
            current, proposed, scenario, cfg
        )
        minimum_clearance = torch.minimum(
            minimum_clearance,
            torch.where(alive, clearance, torch.full_like(clearance, torch.inf)).amin(),
        )
        deficit = torch.relu(clearance.new_tensor(4.0e-4) - clearance) / 1.0e-3
        clearance_sum = clearance_sum + (deficit.square() * active_float).sum()
        collided = alive & (clearance.detach() <= 0.0)
        collision_sum = collision_sum + (
            collided.to(states.dtype) * float(rollout_horizon - offset)
        ).sum()
        active_count = active_count + active_float.sum()
        next_alive = alive & ~collided
        current = torch.where(next_alive[:, None], proposed, current)
        hidden = torch.where(next_alive[:, None, None], output.hidden, hidden)
        alive = next_alive
    denominator = active_count.clamp_min(1.0)
    state_loss = state_sum / denominator
    clearance_loss = clearance_sum / denominator
    collision_loss = collision_sum / float(count)
    total = state_loss + 0.05 * clearance_loss + 0.25 * collision_loss
    return total, {
        "multi_step_state_mse": float(state_loss.detach().cpu()),
        "multi_step_clearance_hinge": float(clearance_loss.detach().cpu()),
        "multi_step_collision_penalty": float(collision_loss.detach().cpu()),
        "multi_step_minimum_clearance_m": float(minimum_clearance.detach().cpu()),
        "multi_step_active_transitions": float(active_count.detach().cpu()),
    }


def _predicted_evaluation(
    policy: torch.nn.Module,
    cfg: Any,
    aperture: float,
    horizon: int,
    artifact: Any | None = None,
) -> dict[str, Any]:
    evaluation = evaluate_single_slit_contact_feedback_v2(
        policy,
        cfg,
        episodes=1,
        horizon=horizon,
        visible_aperture_m=aperture,
        return_state_trace=artifact is not None,
    )
    episode = evaluation["per_episode"][0]
    ordered_progress = 0
    if artifact is not None:
        state_trace = evaluation["resolved_state_trace"]
        if not isinstance(state_trace, Tensor):
            raise TypeError("contact evaluator omitted its tensor state trace")
        for state in state_trace[0]:
            ordered_progress = _project_model_induced_progress(
                state.view(1, -1), artifact, ordered_progress
            )
    final_x = float(episode["final_resolved_state"][0])
    final_y = float(episode["final_resolved_state"][1])
    final_theta = float(episode["final_resolved_state"][2])
    final_speed = math.hypot(
        float(episode["final_resolved_state"][3]),
        float(episode["final_resolved_state"][4]),
    )
    final_omega = abs(float(episode["final_resolved_state"][5]))
    ordered_path_pose_error = float("inf")
    ordered_path_pose_error_bin = math.inf
    if artifact is not None:
        reference_pose = artifact.reference_poses[ordered_progress]
        final_pose = reference_pose.new_tensor((final_x, final_y, final_theta))
        pose_delta = final_pose - reference_pose
        pose_delta[2] = 0.35 * wrap_angle(pose_delta[2])
        ordered_path_pose_error = float(torch.linalg.vector_norm(pose_delta))
        ordered_path_pose_error_bin = math.floor(
            ordered_path_pose_error / ORDERED_PATH_POSE_ERROR_BIN + 1.0e-6
        )
    progress_bin = math.floor(
        min(final_x, float(GOAL_POSE[0])) / LEFT_TO_RIGHT_PROGRESS_BIN_M + 1.0e-6
    )
    vertices = payload_shape_rectangles(cfg.payload_shape, "cpu").reshape(-1, 2)
    cosine = math.cos(final_theta)
    sine = math.sin(final_theta)
    world_x_extent = max(
        abs(cosine * float(point[0]) - sine * float(point[1]))
        for point in vertices
    )
    fully_right_of_wall = (
        final_x - world_x_extent
        > float(cfg.wall_slab) + COLLISION_RESERVE_EACH_SIDE_M
    )
    # Before passage, route progress is the useful curriculum signal. Once
    # the complete rigid body is right of the wall, rewarding still larger x
    # selects runaway policies. Switch authority at that physical event to
    # terminal pose/twist settling, then use contact only as a tiebreaker.
    if fully_right_of_wall:
        secondary = (
            float(ordered_progress),
            -float(episode["final_position_error_m"]),
            -float(episode["final_angle_error_rad"]),
            -final_speed,
            -final_omega,
            -float(episode["maximum_consecutive_contact_steps"]),
            -float(episode["contact_count"]),
        )
    else:
        secondary = (
            float(ordered_progress),
            -float(ordered_path_pose_error_bin),
            -float(episode["maximum_consecutive_contact_steps"]),
            -float(episode["stuck_tick_count"]),
            -float(episode["contact_count"]),
            -ordered_path_pose_error,
            -float(episode["final_position_error_m"]),
            -float(episode["final_angle_error_rad"]),
        )
    key = (
        float(bool(episode["success"])),
        float(fully_right_of_wall),
        *secondary,
    )
    return {
        "success": bool(episode["success"]),
        "finish_step": int(episode["finish_step"]),
        "final_x_m": final_x,
        "final_y_m": final_y,
        "final_position_error_m": float(episode["final_position_error_m"]),
        "final_angle_error_rad": float(episode["final_angle_error_rad"]),
        "final_speed_m_per_s": final_speed,
        "final_omega_rad_per_s": final_omega,
        "fully_right_of_wall": fully_right_of_wall,
        "ordered_teacher_path_index": ordered_progress,
        "ordered_teacher_path_progress": (
            ordered_progress / max(1, int(artifact.horizon))
            if artifact is not None
            else None
        ),
        "ordered_teacher_path_pose_error": (
            ordered_path_pose_error if artifact is not None else None
        ),
        "ordered_teacher_path_pose_error_bin": (
            int(ordered_path_pose_error_bin) if artifact is not None else None
        ),
        "cartesian_x_progress_bin_diagnostic_only": progress_bin,
        "contact_count": int(episode["contact_count"]),
        "maximum_consecutive_contact_steps": int(
            episode["maximum_consecutive_contact_steps"]
        ),
        "stuck_tick_count": int(episode["stuck_tick_count"]),
        "selection_key": list(key),
        "planner_source": "local_prediction",
    }


@torch.no_grad()
def _oracle_plan_evaluation(
    policy: torch.nn.Module,
    artifact: Any,
    cfg: Any,
    aperture: float,
    horizon: int,
) -> dict[str, Any]:
    scenario = single_slit_scenario(1, cfg, visible_aperture_m=aperture)
    state = scenario.state0.clone()
    edges = radius_edges(cfg, device=state.device)
    hidden = policy.init_hidden(1, state.device)
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    progress = 0
    dwell = 0
    contacts = 0
    maximum_contact_run = 0
    contact_run = 0
    succeeded = False
    query_failures = 0
    finish_step = horizon
    evaluated_ticks = 0
    for tick in range(horizon):
        observation = single_slit_contact_observations_v1(
            state, scenario, cfg, context
        )
        label, _error = query_exact_local_se2_planning_teacher(
            state,
            progress,
            artifact,
            cfg,
            visible_aperture_m=aperture,
        )
        if label is None:
            query_failures += 1
            finish_step = tick
            break
        progress = max(progress, label.nearest_index)
        evaluated_ticks = tick + 1
        target = local_se2_plan_target(state, label).view(1, 1, -1)
        override = target.expand(1, cfg.n_agents, -1)
        output = policy(
            observation,
            edges,
            hidden,
            local_se2_plan_override=override,
        )
        hidden = output.hidden
        transition = single_slit_contact_feedback_step_v1(
            state, output.forces, scenario, cfg
        )
        context = single_slit_contact_observation_context_from_result_v1(
            transition, output.forces, cfg
        )
        contact = bool(transition.attempted_hit[0])
        contacts += int(contact)
        contact_run = contact_run + 1 if contact else 0
        maximum_contact_run = max(maximum_contact_run, contact_run)
        state = transition.resolved_state
        at_goal = (not contact) and bool(strict_final_mask(state, scenario.goal)[0])
        dwell = dwell + 1 if at_goal else 0
        if dwell >= STRICT_STABLE_STEPS:
            succeeded = True
            finish_step = tick + 1
            break
    components = strict_final_components(state, scenario.goal)
    final_x = float(state[0, 0].cpu())
    progress_bin = math.floor(
        min(final_x, float(GOAL_POSE[0])) / LEFT_TO_RIGHT_PROGRESS_BIN_M + 1.0e-6
    )
    # A narrow-passage plan is intentionally non-monotone in x: retreating
    # and re-entering can advance the ordered SE(2) route.  P0 authority must
    # therefore select on ordered teacher-path progress, not Cartesian x.
    key = (
        float(succeeded),
        float(progress),
        -float(query_failures),
        -float(maximum_contact_run),
        -float(contacts),
        float(evaluated_ticks),
        -float(components["position_error_m"][0].cpu()),
        -float(components["angle_error_rad"][0].cpu()),
    )
    return {
        "success": succeeded,
        "finish_step": finish_step,
        "final_x_m": final_x,
        "final_position_error_m": float(components["position_error_m"][0].cpu()),
        "final_angle_error_rad": float(components["angle_error_rad"][0].cpu()),
        "contact_count": contacts,
        "maximum_consecutive_contact_steps": maximum_contact_run,
        "teacher_query_failures": query_failures,
        "evaluated_ticks": evaluated_ticks,
        "ordered_teacher_path_index": progress,
        "ordered_teacher_path_progress": progress
        / max(1, int(artifact.states.shape[0]) - 1),
        "cartesian_x_progress_bin_diagnostic_only": progress_bin,
        "selection_key": list(key),
        "planner_source": "privileged_oracle_waypoint_only",
        "teacher_force_used": False,
    }


def _authority_reports(
    policy: torch.nn.Module,
    cfg: Any,
    artifact_path: Path,
    aperture: float,
    horizon: int,
    authority_device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority_cfg = replace(cfg, device=str(authority_device))
    authority_policy = copy.deepcopy(policy).to(authority_device)
    authority_policy.env_cfg = authority_cfg
    authority_policy.eval()
    artifact = load_executable_teacher_artifact(artifact_path, authority_cfg)
    predicted = _predicted_evaluation(
        authority_policy, authority_cfg, aperture, horizon, artifact
    )
    oracle = _oracle_plan_evaluation(
        authority_policy, artifact, authority_cfg, aperture, horizon
    )
    return predicted, oracle


def _uniform_initial_perturbations(
    episodes: int,
    *,
    seed: int,
    position_cap_m: float,
    angle_cap_rad: float,
    velocity_cap: float,
    angular_velocity_cap: float | None = None,
    device: torch.device,
) -> Tensor:
    """Create a reproducible nominal-plus-local initial-state audit panel."""

    if episodes < 0:
        raise ValueError("initial perturbation episode count must be nonnegative")
    if angular_velocity_cap is None:
        angular_velocity_cap = velocity_cap
    caps = (
        position_cap_m,
        angle_cap_rad,
        velocity_cap,
        angular_velocity_cap,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in caps):
        raise ValueError("initial perturbation caps must be finite and nonnegative")
    source = torch.tensor(
        (*SOURCE_POSE, 0.0, 0.0, 0.0),
        dtype=torch.float32,
        device=device,
    )
    if episodes == 0:
        return source.view(1, 6)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = 2.0 * torch.rand(episodes, 6, generator=generator) - 1.0
    scale = torch.tensor(
        (
            position_cap_m,
            position_cap_m,
            angle_cap_rad,
            velocity_cap,
            velocity_cap,
            angular_velocity_cap,
        )
    )
    perturbed = source.cpu().view(1, 6) + noise * scale
    return torch.cat((source.cpu().view(1, 6), perturbed), dim=0).to(device)


@torch.no_grad()
def _predicted_population_evaluation(
    policy: torch.nn.Module,
    cfg: Any,
    aperture: float,
    horizon: int,
    artifact: Any,
    initial_states: Tensor,
    *,
    workers: int = 1,
    required_success_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """Strict single-episode authority over a fixed perturbation population.

    The millimetre-clearance route is sensitive to batched linear-solve
    arithmetic.  Deployment runs one payload at a time, so every population
    member is deliberately evaluated with batch size one instead of allowing
    the audit batch size to alter the policy trajectory.
    """

    if workers <= 0:
        raise ValueError("population authority workers must be positive")

    def evaluate_one(episode_index: int) -> tuple[Mapping[str, Any], int, float]:
        evaluation = evaluate_single_slit_contact_feedback_v2(
            policy,
            cfg,
            initial_states=initial_states[episode_index : episode_index + 1],
            horizon=horizon,
            visible_aperture_m=aperture,
            return_state_trace=True,
        )
        trace = evaluation["resolved_state_trace"]
        if not isinstance(trace, Tensor) or trace.shape[0] != 1:
            raise TypeError("population authority omitted its single-episode trace")
        progress = 0
        for state in trace[0]:
            progress = _project_model_induced_progress(
                state.view(1, -1), artifact, progress
            )
        reference = artifact.reference_poses[progress]
        final_pose = trace[0, -1, :3]
        delta = final_pose - reference
        delta[2] = 0.35 * wrap_angle(delta[2])
        return (
            evaluation["per_episode"][0],
            progress,
            float(torch.linalg.vector_norm(delta).cpu()),
        )

    episode_count = int(initial_states.shape[0])
    protected = sorted(set(int(index) for index in required_success_indices))
    if any(index < 0 or index >= episode_count for index in protected):
        raise ValueError("required success index is outside the authority panel")
    priority = [0, *[index for index in protected if index != 0]]
    evaluated_by_index: dict[int, tuple[Mapping[str, Any], int, float]] = {}
    if workers == 1 or len(priority) == 1:
        for index in priority:
            evaluated_by_index[index] = evaluate_one(index)
            if not bool(evaluated_by_index[index][0]["success"]):
                break
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(priority))
        ) as pool:
            priority_rows = list(pool.map(evaluate_one, priority))
        evaluated_by_index.update(zip(priority, priority_rows, strict=True))
    protected_failure_short_circuit = any(
        not bool(item[0]["success"])
        for item in evaluated_by_index.values()
    )
    if not protected_failure_short_circuit:
        remaining_indices = [
            index for index in range(episode_count) if index not in evaluated_by_index
        ]
        if workers == 1:
            remaining = [evaluate_one(index) for index in remaining_indices]
        else:
            with ThreadPoolExecutor(
                max_workers=min(workers, max(1, len(remaining_indices)))
            ) as pool:
                remaining = list(pool.map(evaluate_one, remaining_indices))
        evaluated_by_index.update(zip(remaining_indices, remaining, strict=True))
    evaluated_indices = sorted(evaluated_by_index)
    evaluated = [evaluated_by_index[index] for index in evaluated_indices]
    rows = [item[0] for item in evaluated]
    ordered_progress = [item[1] for item in evaluated]
    path_errors = [item[2] for item in evaluated]
    success_count = sum(int(bool(row["success"])) for row in rows)
    nominal_success = bool(rows[0]["success"])
    total_progress = sum(ordered_progress)
    total_max_contact = sum(
        int(row["maximum_consecutive_contact_steps"]) for row in rows
    )
    total_contacts = sum(int(row["contact_count"]) for row in rows)
    total_path_error = sum(path_errors)
    # A robust candidate may never trade away the already established nominal
    # success.  Within that constraint, completed perturbations dominate path
    # advance; contact and path error resolve otherwise equal frontiers.
    key = (
        float(nominal_success),
        float(success_count),
        float(min(ordered_progress)),
        float(total_progress),
        -float(total_max_contact),
        -float(total_contacts),
        -float(total_path_error),
    )
    return {
        "success": (
            not protected_failure_short_circuit and success_count == episode_count
        ),
        "nominal_success": nominal_success,
        "success_count": success_count,
        "episode_count": episode_count,
        "evaluated_episode_count": len(rows),
        "evaluated_episode_indices": evaluated_indices,
        "required_success_indices": protected,
        "protected_success_failure_short_circuit": (
            protected_failure_short_circuit
        ),
        "nominal_failure_short_circuit": (
            protected_failure_short_circuit and 0 in evaluated_by_index
            and not bool(evaluated_by_index[0][0]["success"])
        ),
        "success_indices": [
            index
            for index, row in zip(evaluated_indices, rows, strict=True)
            if bool(row["success"])
        ],
        "success_mask": [bool(row["success"]) for row in rows],
        "finish_steps": [int(row["finish_step"]) for row in rows],
        "contact_counts": [int(row["contact_count"]) for row in rows],
        "maximum_consecutive_contact_steps_per_episode": [
            int(row["maximum_consecutive_contact_steps"]) for row in rows
        ],
        "failed_strict_conditions_per_episode": [
            list(row["failed_strict_conditions"]) for row in rows
        ],
        "final_resolved_states": [list(row["final_resolved_state"]) for row in rows],
        "final_position_errors_m": [
            float(row["final_position_error_m"]) for row in rows
        ],
        "final_angle_errors_rad": [
            float(row["final_angle_error_rad"]) for row in rows
        ],
        "final_speeds_m_per_s": [
            float(row["final_speed_m_per_s"]) for row in rows
        ],
        "final_omegas_rad_per_s": [
            float(row["final_omega_rad_per_s"]) for row in rows
        ],
        "maximum_consecutive_strict_steps_per_episode": [
            int(row["maximum_consecutive_strict_steps"]) for row in rows
        ],
        "success_rate": success_count / episode_count,
        "finish_step": max(int(row["finish_step"]) for row in rows),
        "ordered_teacher_path_index": min(ordered_progress),
        "mean_ordered_teacher_path_index": total_progress / len(rows),
        "ordered_teacher_path_indices": ordered_progress,
        "ordered_teacher_path_pose_errors": path_errors,
        "maximum_consecutive_contact_steps": max(
            int(row["maximum_consecutive_contact_steps"]) for row in rows
        ),
        "contact_count": total_contacts,
        "selection_key": list(key),
        "planner_source": "local_prediction_population_exact_cpu",
    }


def _predicted_authority_report(
    policy: torch.nn.Module,
    cfg: Any,
    artifact_path: Path,
    aperture: float,
    horizon: int,
    authority_device: torch.device,
    initial_states: Tensor | None = None,
    population_workers: int = 1,
    required_success_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """Evaluate a planner candidate without rerunning the frozen oracle arm."""

    authority_cfg = replace(cfg, device=str(authority_device))
    authority_policy = copy.deepcopy(policy).to(authority_device)
    authority_policy.env_cfg = authority_cfg
    authority_policy.eval()
    artifact = load_executable_teacher_artifact(artifact_path, authority_cfg)
    if initial_states is not None:
        return _predicted_population_evaluation(
            authority_policy,
            authority_cfg,
            aperture,
            horizon,
            artifact,
            initial_states.to(authority_device),
            workers=population_workers,
            required_success_indices=required_success_indices,
        )
    return _predicted_evaluation(
        authority_policy, authority_cfg, aperture, horizon, artifact
    )


def _optimization_key(report: Mapping[str, Any], mode: str) -> tuple[float, ...]:
    """Return the training admission key without changing final success semantics."""

    if mode in {"oracle", "predicted"}:
        return tuple(float(value) for value in report["selection_key"])
    if mode != "recovery-curriculum":
        raise ValueError(f"unknown local-planner selection mode: {mode!r}")
    return (
        float(bool(report["success"])),
        float(bool(report["fully_right_of_wall"])),
        -float(report["maximum_consecutive_contact_steps"]),
        -float(report["stuck_tick_count"]),
        -float(report["contact_count"]),
        float(report["final_x_m"]),
        -float(report["final_position_error_m"]),
        -float(report["final_angle_error_rad"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--teacher-rollout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--authority-device", default="cpu")
    parser.add_argument(
        "--collection-device",
        default=None,
        help=(
            "device for factual learned rollouts and corrective suffixes; "
            "defaults to the training device and should match authority when "
            "closed-loop GPU arithmetic is non-deterministic"
        ),
    )
    parser.add_argument("--aperture", type=float, default=0.92)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--updates-per-iteration", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--collection-horizon", type=int, default=256)
    parser.add_argument("--evaluation-horizon", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--plan-loss-weight", type=float, default=1.0)
    parser.add_argument("--phase-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--multi-step-rollout-horizon",
        type=int,
        default=0,
        help="differentiable corrective-suffix rollout length; zero disables it",
    )
    parser.add_argument("--multi-step-window-count", type=int, default=12)
    parser.add_argument("--multi-step-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--frozen-audit-interval",
        type=int,
        default=100,
        help=(
            "optimizer steps between full CPU bitwise audits of frozen tensors; "
            "the complete audit is always repeated before candidate evaluation"
        ),
    )
    parser.add_argument(
        "--planner-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train private plan/phase modules without frozen GBP and force decoding",
    )
    parser.add_argument(
        "--recurrent-sequence-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="backpropagate recovery memory through complete model-induced rollouts",
    )
    parser.add_argument(
        "--teacher-forced-recovery-suffix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after the learned policy's factual first contact, execute exact local "
            "teacher plans through the unchanged Sheaf-GBP force layer and add the "
            "complete corrective suffix as explicitly non-DAgger sequence data"
        ),
    )
    parser.add_argument(
        "--teacher-suffix-sequence-weight",
        type=float,
        default=4.0,
        help=(
            "per-sequence loss weight for the complete corrective suffix; "
            "failed DAgger trajectories remain weight one"
        ),
    )
    parser.add_argument(
        "--teacher-suffix-perturbations-per-tick",
        type=int,
        default=0,
        help=(
            "exact-teacher perturbations around each successful post-contact "
            "suffix state; unlike nominal perturbations these train the "
            "source-null recovery adapter"
        ),
    )
    parser.add_argument("--force-loss-weight", type=float, default=1.0)
    parser.add_argument("--wrench-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--section-wrench-loss-weight",
        type=float,
        default=3.0,
        help=(
            "directly supervise the communicated 3-D correction-wrench section "
            "against teacher net wrench minus the immutable source wrench"
        ),
    )
    parser.add_argument("--direct-unary-loss-weight", type=float, default=5.0)
    parser.add_argument("--clearance-weight", type=float, default=0.01)
    parser.add_argument(
        "--selection-mode",
        choices=("oracle", "predicted", "recovery-curriculum"),
        default="oracle",
    )
    parser.add_argument(
        "--collection-mode",
        choices=("match-selection", "oracle", "predicted"),
        default="match-selection",
        help="which closed-loop policy distribution supplies DAgger states",
    )
    parser.add_argument(
        "--controller-scope",
        choices=(
            "planner-head",
            "planner-recovery-columns",
            "planner-recovery-adapter",
            "planner-recovery-adapter-fixed-phase",
            "planner-recovery-adapter-recurrent",
            "planner-contact-rebel-fixed-phase",
            "planner-exit-refinement-fixed-phase",
            "planner-approach-refinement-fixed-phase",
            "planner-deep-stuck-refinement-fixed-phase",
            "planner-retry-refinement-fixed-phase",
            "planner-only",
            "plan-to-wrench",
            "direct-unary",
            "full-correction",
        ),
        default="full-correction",
        help=(
            "train only the private plan predictor, the complete zero-action "
            "planning bridge, or jointly adapt the complete correction Sheaf"
        ),
    )
    parser.add_argument(
        "--backtracking-alphas",
        type=float,
        nargs="+",
        default=(1.0, 0.3, 0.1, 0.03, 0.01, 0.003),
    )
    parser.add_argument(
        "--break-on-frontier-advance",
        action="store_true",
        help=(
            "accept the first candidate that strictly advances ordered teacher "
            "progress instead of spending more rollouts on same-iteration alphas"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--nominal-perturbations-per-tick", type=int, default=0)
    parser.add_argument("--perturb-position-std-m", type=float, default=0.01)
    parser.add_argument("--perturb-angle-std-rad", type=float, default=0.015)
    parser.add_argument("--perturb-velocity-std", type=float, default=0.04)
    parser.add_argument(
        "--model-induced-initial-perturbation-episodes",
        type=int,
        default=0,
        help=(
            "additional fixed uniformly perturbed initial states collected by "
            "the learned policy and included in strict candidate authority"
        ),
    )
    parser.add_argument(
        "--initial-perturb-position-cap-m", type=float, default=0.002
    )
    parser.add_argument(
        "--initial-perturb-angle-cap-rad", type=float, default=0.004
    )
    parser.add_argument(
        "--initial-perturb-velocity-cap", type=float, default=0.004
    )
    parser.add_argument(
        "--initial-perturb-angular-velocity-cap",
        type=float,
        help="angular-velocity cap; defaults to --initial-perturb-velocity-cap",
    )
    parser.add_argument(
        "--perturbation-authority-workers",
        type=int,
        default=1,
        help="CPU workers; every strict episode still executes at batch size one",
    )
    parser.add_argument(
        "--perturbation-collection-workers",
        type=int,
        default=1,
        help=(
            "parallel CPU model-induced collectors; each trajectory remains "
            "an independent batch-size-one exact-contact rollout"
        ),
    )
    parser.add_argument(
        "--collect-failed-perturbations-only",
        action="store_true",
        help=(
            "collect DAgger labels only for currently failed fixed-panel "
            "perturbations; incumbent successes remain protected by exact "
            "candidate authority and source-preservation distillation"
        ),
    )
    parser.add_argument(
        "--perturbation-collection-contact-tail-steps",
        type=int,
        default=0,
        help=(
            "stop a failed perturbation rollout after this many consecutive "
            "rejected contacts; zero retains the full collection horizon"
        ),
    )
    parser.add_argument(
        "--source-correct-recovery",
        action="store_true",
        help=(
            "fit only exact recenter/retreat/brake/contact-history targets and "
            "distill all other planner rows to the iteration incumbent"
        ),
    )
    parser.add_argument("--source-preservation-weight", type=float, default=10.0)
    parser.add_argument(
        "--recovery-plan-dimension-weights",
        type=float,
        nargs=8,
        default=None,
        metavar=("DX", "DY", "DTHETA", "VX", "VY", "OMEGA", "XTRACK", "PROGRESS"),
        help=(
            "optional eight-coordinate weighting for exact frontier recovery "
            "targets; incumbent preservation remains unweighted"
        ),
    )
    parser.add_argument(
        "--frontier-focused-recovery-width",
        type=int,
        default=0,
        help=(
            "when positive, fit exact post-contact plans only within this many "
            "ordered teacher indices behind the current learned frontier and "
            "through the successful suffix; distill earlier recovery states "
            "to the incumbent"
        ),
    )
    parser.add_argument(
        "--frontier-focused-recovery-lookahead",
        type=int,
        default=4,
        help="ordered teacher indices beyond the learned frontier fitted per iteration",
    )
    parser.add_argument(
        "--aggregate-induced-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "retain every earlier model-induced rollout, as required by DAgger; "
            "disable only for a historical non-aggregated ablation"
        ),
    )
    parser.add_argument(
        "--replay-source-checkpoint",
        type=Path,
        nargs="+",
        default=(),
        help=(
            "optional earlier policy whose closed-loop states seed the DAgger "
            "archive before collecting from the current source"
        ),
    )
    parser.add_argument(
        "--initial-authority-log",
        type=Path,
        help=(
            "reuse the final predicted/oracle record from the parent stage; "
            "intended only for an audited zero-behaviour retry bridge on the "
            "identical deterministic perturbation panel"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(
        value <= 0
        for value in (
            args.iterations,
            args.updates_per_iteration,
            args.batch_size,
            args.collection_horizon,
            args.evaluation_horizon,
        )
    ):
        raise ValueError("local planner integer controls must be positive")
    if args.nominal_perturbations_per_tick < 0:
        raise ValueError("nominal perturbation count must be non-negative")
    if args.teacher_suffix_perturbations_per_tick < 0:
        raise ValueError("teacher suffix perturbation count must be non-negative")
    if args.model_induced_initial_perturbation_episodes < 0:
        raise ValueError("model-induced initial perturbation count must be non-negative")
    if args.perturbation_authority_workers <= 0:
        raise ValueError("perturbation authority workers must be positive")
    if args.perturbation_collection_workers <= 0:
        raise ValueError("perturbation collection workers must be positive")
    if args.perturbation_collection_contact_tail_steps < 0:
        raise ValueError("perturbation contact-tail steps must be non-negative")
    if args.model_induced_initial_perturbation_episodes > 0 and (
        args.selection_mode != "predicted" or args.collection_mode == "oracle"
    ):
        raise ValueError(
            "initial-perturbation DAgger requires predicted selection and collection"
        )
    if (
        args.teacher_suffix_perturbations_per_tick > 0
        and not args.teacher_forced_recovery_suffix
    ):
        raise ValueError("teacher suffix perturbations require a corrective suffix")
    if any(
        value < 0.0
        for value in (
            args.perturb_position_std_m,
            args.perturb_angle_std_rad,
            args.perturb_velocity_std,
            args.initial_perturb_position_cap_m,
            args.initial_perturb_angle_cap_rad,
            args.initial_perturb_velocity_cap,
            (
                args.initial_perturb_velocity_cap
                if args.initial_perturb_angular_velocity_cap is None
                else args.initial_perturb_angular_velocity_cap
            ),
        )
    ):
        raise ValueError("nominal perturbation scales must be non-negative")
    if args.source_preservation_weight <= 0.0:
        raise ValueError("source preservation weight must be positive")
    if args.frontier_focused_recovery_width < 0:
        raise ValueError("frontier recovery width must be non-negative")
    if args.frontier_focused_recovery_lookahead < 0:
        raise ValueError("frontier recovery lookahead must be non-negative")
    if (
        args.frontier_focused_recovery_width > 0
        and not args.source_correct_recovery
    ):
        raise ValueError("frontier recovery requires source-correct recovery")
    if args.phase_loss_weight < 0.0:
        raise ValueError("phase loss weight must be non-negative")
    if args.multi_step_rollout_horizon < 0:
        raise ValueError("multi-step rollout horizon must be non-negative")
    if args.multi_step_window_count <= 0:
        raise ValueError("multi-step window count must be positive")
    if (
        not math.isfinite(args.multi_step_loss_weight)
        or args.multi_step_loss_weight <= 0.0
    ):
        raise ValueError("multi-step loss weight must be finite and positive")
    if args.multi_step_rollout_horizon > 0 and not args.teacher_forced_recovery_suffix:
        raise ValueError("multi-step training requires a corrective teacher suffix")
    if args.multi_step_rollout_horizon > 0 and args.recurrent_sequence_training:
        raise ValueError("multi-step and recurrent-sequence training are separate stages")
    if args.frozen_audit_interval <= 0:
        raise ValueError("frozen audit interval must be positive")
    if (
        not math.isfinite(args.teacher_suffix_sequence_weight)
        or args.teacher_suffix_sequence_weight <= 0.0
    ):
        raise ValueError("teacher suffix sequence weight must be finite and positive")
    if args.planner_fast_path and args.controller_scope not in {
        "planner-recovery-adapter",
        "planner-recovery-adapter-fixed-phase",
        "planner-recovery-adapter-recurrent",
        "planner-contact-rebel-fixed-phase",
        "planner-exit-refinement-fixed-phase",
        "planner-approach-refinement-fixed-phase",
        "planner-deep-stuck-refinement-fixed-phase",
        "planner-retry-refinement-fixed-phase",
    }:
        raise ValueError("planner fast path is restricted to source-null adapters")
    if args.recurrent_sequence_training and not (
        args.planner_fast_path
        and args.controller_scope == "planner-recovery-adapter-recurrent"
    ):
        raise ValueError(
            "recurrent sequence training requires recurrent adapter fast path"
        )
    if args.source_correct_recovery and args.controller_scope not in {
        "planner-head",
        "planner-recovery-columns",
        "planner-recovery-adapter",
        "planner-recovery-adapter-fixed-phase",
        "planner-recovery-adapter-recurrent",
        "planner-contact-rebel-fixed-phase",
        "planner-exit-refinement-fixed-phase",
        "planner-approach-refinement-fixed-phase",
        "planner-deep-stuck-refinement-fixed-phase",
        "planner-retry-refinement-fixed-phase",
    }:
        raise ValueError(
            "source-correct recovery requires planner-head or recovery-column scope"
        )
    alphas = tuple(float(value) for value in args.backtracking_alphas)
    if any(not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("backtracking alphas must lie in (0,1]")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    authority_device = torch.device(args.authority_device)
    collection_device = torch.device(
        args.device if args.collection_device is None else args.collection_device
    )
    if args.perturbation_collection_workers > 1 and collection_device.type != "cpu":
        raise ValueError("parallel perturbation collection is restricted to CPU")
    source_path = args.source_checkpoint.resolve(strict=True)
    teacher_path = args.teacher.resolve(strict=True)
    artifact_path = args.teacher_rollout.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    loaded = load_single_slit_checkpoint(source_path, device=device)
    if not loaded.policy.arm_cfg.local_se2_planner:
        raise ValueError("source checkpoint does not contain the local SE(2) planner")
    if loaded.metadata.get("training_stage") not in {
        BRIDGE_STAGE,
        DIRECT_WRENCH_BRIDGE_STAGE,
        ANALYTIC_TOTAL_WRENCH_BRIDGE_STAGE,
        TERMINAL_ATTRACTOR_BRIDGE_STAGE,
        RECOVERY_ADAPTER_BRIDGE_STAGE,
        ANALYTIC_PHASE_LATCH_BRIDGE_STAGE,
        CANONICAL_RECOVERY_BRIDGE_STAGE,
        CONTACT_REBEL_BRIDGE_STAGE,
        EXIT_REFINEMENT_BRIDGE_STAGE,
        APPROACH_REFINEMENT_BRIDGE_STAGE,
        DEEP_STUCK_REFINEMENT_BRIDGE_STAGE,
        RETRY_REFINEMENT_BRIDGE_STAGE,
        TRAINING_STAGE,
    }:
        raise ValueError("source is not a planner bridge or planner continuation")
    policy = loaded.policy
    planner_names = set(planner_parameter_names(policy))
    recovery_column_name = "local_se2_planner_encoder.0.weight"
    if args.controller_scope == "planner-recovery-columns":
        selected_names = {recovery_column_name}
    elif args.controller_scope == "planner-recovery-adapter":
        if not policy.arm_cfg.local_se2_recovery_adapter:
            raise ValueError("planner recovery adapter scope requires its zero bridge")
        selected_names = set(recovery_adapter_parameter_names(policy)) | {
            name
            for name, _parameter in policy.named_parameters()
            if name.startswith("persistent_recovery_phase_head.")
        }
    elif args.controller_scope == "planner-recovery-adapter-fixed-phase":
        if not policy.arm_cfg.local_se2_recovery_adapter:
            raise ValueError("planner recovery adapter scope requires its zero bridge")
        selected_names = set(recovery_adapter_parameter_names(policy))
    elif args.controller_scope == "planner-recovery-adapter-recurrent":
        if not policy.arm_cfg.local_se2_recovery_adapter:
            raise ValueError("planner recovery adapter scope requires its zero bridge")
        selected_names = set(recovery_adapter_parameter_names(policy)) | {
            name
            for name, _parameter in policy.named_parameters()
            if name.startswith("persistent_recovery_gru.")
            or name.startswith("persistent_recovery_phase_head.")
        }
    elif args.controller_scope == "planner-contact-rebel-fixed-phase":
        if not policy.arm_cfg.local_se2_contact_rebel_adapter:
            raise ValueError("contact rebel scope requires its zero bridge")
        selected_names = set(contact_rebel_parameter_names(policy))
    elif args.controller_scope == "planner-exit-refinement-fixed-phase":
        if not policy.arm_cfg.local_se2_exit_refinement_adapter:
            raise ValueError("exit refinement scope requires its zero bridge")
        selected_names = set(exit_refinement_parameter_names(policy))
    elif args.controller_scope == "planner-approach-refinement-fixed-phase":
        if not policy.arm_cfg.local_se2_approach_refinement_adapter:
            raise ValueError("approach refinement scope requires its zero bridge")
        selected_names = set(approach_refinement_parameter_names(policy))
    elif args.controller_scope == "planner-deep-stuck-refinement-fixed-phase":
        if not policy.arm_cfg.local_se2_deep_stuck_refinement_adapter:
            raise ValueError("deep-stuck refinement scope requires its zero bridge")
        selected_names = set(deep_stuck_refinement_parameter_names(policy))
    elif args.controller_scope == "planner-retry-refinement-fixed-phase":
        if not policy.arm_cfg.local_se2_retry_refinement_thresholds:
            raise ValueError("retry refinement scope requires its zero bridge")
        selected_names = set(retry_refinement_parameter_names(policy))
    elif args.controller_scope == "planner-head":
        selected_names = {
            name
            for name, _parameter in policy.named_parameters()
            if name.startswith("local_se2_planner_encoder.")
            or name.startswith("local_se2_plan_mean_head.")
        }
    elif args.controller_scope == "plan-to-wrench":
        selected_names = {
            name
            for name, _parameter in policy.named_parameters()
            if name == "correction_plan_fusion.weight"
            or name.startswith("correction_mean_head.")
        }
    elif args.controller_scope == "direct-unary":
        if not policy.arm_cfg.local_se2_direct_wrench_unary:
            raise ValueError("direct-unary scope requires its zero bridge")
        selected_names = set(DIRECT_WRENCH_PARAMETER_NAMES)
    elif args.controller_scope == "full-correction":
        selected_names = planner_names | {
            name for name, _parameter in policy.correction_named_parameters()
        }
    else:
        selected_names = planner_names
    selected = tuple(
        name for name, _parameter in policy.named_parameters() if name in selected_names
    )
    if set(selected) != selected_names:
        raise RuntimeError("planner-conditioned controller trainable partition changed")
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name in selected)
    parameters = [dict(policy.named_parameters())[name] for name in selected]
    recovery_column_prefix: Tensor | None = None
    if args.controller_scope == "planner-recovery-columns":
        recovery_weight = dict(policy.named_parameters())[recovery_column_name]
        expected_width = (
            policy.arm_cfg.obs_dim
            + policy.arm_cfg.hidden_dim
            + policy.arm_cfg.persistent_recovery_dim
        )
        if recovery_weight.shape != (
            policy.arm_cfg.local_se2_planner_hidden_dim,
            expected_width,
        ):
            raise RuntimeError("local planner recovery-column matrix shape changed")
        recovery_column_prefix = recovery_weight[
            :, : -policy.arm_cfg.persistent_recovery_dim
        ].detach().clone()
    frozen = _frozen_reference(policy, selected)
    teacher = load_executable_teacher(teacher_path, device=device)
    artifact = load_executable_teacher_artifact(artifact_path, loaded.cfg)
    if collection_device == device:
        collection_cfg = loaded.cfg
        collection_artifact = artifact
    else:
        collection_cfg = replace(loaded.cfg, device=str(collection_device))
        collection_artifact = load_executable_teacher_artifact(
            artifact_path, collection_cfg
        )
    perturbation_panel = _uniform_initial_perturbations(
        args.model_induced_initial_perturbation_episodes,
        seed=args.seed + 7919,
        position_cap_m=args.initial_perturb_position_cap_m,
        angle_cap_rad=args.initial_perturb_angle_cap_rad,
        velocity_cap=args.initial_perturb_velocity_cap,
        angular_velocity_cap=args.initial_perturb_angular_velocity_cap,
        device=authority_device,
    )
    edges = radius_edges(loaded.cfg, device=device)
    if args.initial_authority_log is not None:
        if (
            args.selection_mode != "predicted"
            or args.model_induced_initial_perturbation_episodes <= 0
            or loaded.metadata.get("training_stage") != RETRY_REFINEMENT_BRIDGE_STAGE
        ):
            raise ValueError(
                "initial authority reuse requires a predicted population and "
                "an audited zero retry bridge"
            )
        authority_log_path = args.initial_authority_log.resolve(strict=True)
        authority_text = authority_log_path.read_text(encoding="utf-8")
        authority_lines = [
            line for line in authority_text.splitlines() if line.strip()
        ]
        if not authority_lines:
            raise ValueError("initial authority log is empty")
        try:
            authority_record = json.loads(authority_text)
        except json.JSONDecodeError:
            authority_record = json.loads(authority_lines[-1])
        if "predicted" in authority_record and "oracle" in authority_record:
            predicted = copy.deepcopy(authority_record["predicted"])
            oracle = copy.deepcopy(authority_record["oracle"])
        elif "report" in authority_record:
            predicted = copy.deepcopy(authority_record["report"])
            parent_authority = loaded.metadata.get("authority")
            if not isinstance(parent_authority, Mapping) or "oracle" not in parent_authority:
                raise ValueError("source checkpoint omitted its invariant oracle report")
            oracle = copy.deepcopy(parent_authority["oracle"])
        else:
            raise ValueError("initial authority file omitted predicted/report fields")
        expected_episodes = int(perturbation_panel.shape[0])
        if (
            int(predicted.get("episode_count", -1)) != expected_episodes
            or len(predicted.get("ordered_teacher_path_indices", ()))
            != expected_episodes
            or predicted.get("planner_source")
            != "local_prediction_population_exact_cpu"
            or oracle.get("planner_source") != "privileged_oracle_waypoint_only"
        ):
            raise ValueError("reused authority report differs from this population")
    else:
        predicted, oracle = _authority_reports(
            policy,
            loaded.cfg,
            artifact_path,
            args.aperture,
            args.evaluation_horizon,
            authority_device,
        )
        if args.model_induced_initial_perturbation_episodes > 0:
            predicted = _predicted_authority_report(
                policy,
                loaded.cfg,
                artifact_path,
                args.aperture,
                args.evaluation_horizon,
                authority_device,
                initial_states=perturbation_panel,
                population_workers=args.perturbation_authority_workers,
            )
    incumbent_key = _optimization_key(
        oracle if args.selection_mode == "oracle" else predicted,
        args.selection_mode,
    )
    incumbent_state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    incumbent_reports = (predicted, oracle)
    print(json.dumps({"iteration": 0, "predicted": predicted, "oracle": oracle}), flush=True)
    log_path = output_dir / "train.jsonl"
    global_step = 0
    induced_archive: list[Mapping[str, Tensor | int]] = []
    replay_source_paths: list[Path] = []
    for replay_source_argument in args.replay_source_checkpoint:
        replay_source_path = replay_source_argument.resolve(strict=True)
        replay_source = load_single_slit_checkpoint(replay_source_path, device=device)
        if (
            replay_source.arm != loaded.arm
            or replay_source.observation_schema != loaded.observation_schema
            or replay_source.policy.arm_cfg != loaded.policy.arm_cfg
        ):
            raise ValueError("replay-source checkpoint architecture differs from source")
        if args.collection_mode == "oracle":
            replay_seed = _collect_oracle_plan_induced(
                replay_source.policy,
                artifact,
                loaded.cfg,
                args.aperture,
                args.collection_horizon,
            )
        else:
            replay_seed = _collect_model_induced_local_plans(
                replay_source.policy,
                artifact,
                loaded.cfg,
                args.aperture,
                args.collection_horizon,
            )
        if args.frontier_focused_recovery_width > 0:
            replay_post_contact = replay_seed.get("post_contact_active")
            if not isinstance(replay_post_contact, Tensor):
                raise RuntimeError("replay source omitted post-contact labels")
            # A replay-source argument currently collects its nominal initial
            # state only.  It is therefore an incumbent behavior anchor, not
            # a perturbation frontier target.
            replay_seed["planner_update_target"] = torch.zeros_like(
                replay_post_contact, dtype=torch.bool
            )
        induced_archive.append(replay_seed)
        replay_source_paths.append(replay_source_path)
    for iteration in range(1, args.iterations + 1):
        nominal = _collect_nominal(
            policy,
            teacher,
            artifact,
            loaded.cfg,
            args.aperture,
            perturbations_per_tick=args.nominal_perturbations_per_tick,
            perturb_position_std_m=args.perturb_position_std_m,
            perturb_angle_std_rad=args.perturb_angle_std_rad,
            perturb_velocity_std=args.perturb_velocity_std,
        )
        collection_mode = (
            ("oracle" if args.selection_mode == "oracle" else "predicted")
            if args.collection_mode == "match-selection"
            else args.collection_mode
        )
        if collection_device == device:
            collection_policy = policy
        else:
            collection_policy = copy.deepcopy(policy).to(collection_device)
            collection_policy.env_cfg = collection_cfg
            collection_policy.eval()
        if collection_mode == "oracle":
            collected_induced = _collect_oracle_plan_induced(
                collection_policy,
                collection_artifact,
                collection_cfg,
                args.aperture,
                args.collection_horizon,
            )
        else:
            collected_induced = _collect_model_induced_local_plans(
                collection_policy,
                collection_artifact,
                collection_cfg,
                args.aperture,
                args.collection_horizon,
            )
        induced = _move_replay_part(collected_induced, device)
        perturbed_induced: list[dict[str, Tensor | int]] = []
        collection_panel_indices: list[int] = []
        if args.model_induced_initial_perturbation_episodes > 0:
            collection_panel_indices = list(range(1, int(perturbation_panel.shape[0])))
            if args.collect_failed_perturbations_only:
                incumbent_success_mask = incumbent_reports[0].get("success_mask")
                if (
                    not isinstance(incumbent_success_mask, list)
                    or len(incumbent_success_mask) != int(perturbation_panel.shape[0])
                ):
                    raise RuntimeError(
                        "failed-only DAgger collection requires a complete "
                        "incumbent population success mask"
                    )
                collection_panel_indices = [
                    index
                    for index in collection_panel_indices
                    if not bool(incumbent_success_mask[index])
                ]
            collection_initial_states = list(
                perturbation_panel[collection_panel_indices]
                .to(collection_device)
                .unbind(0)
            )

            def collect_perturbation(
                initial_state: Tensor,
            ) -> dict[str, Tensor | int]:
                collected_part = _collect_model_induced_local_plans(
                    collection_policy,
                    collection_artifact,
                    collection_cfg,
                    args.aperture,
                    args.collection_horizon,
                    initial_state=initial_state.view(1, 6),
                    stop_after_consecutive_contact_steps=(
                        args.perturbation_collection_contact_tail_steps
                    ),
                )
                return _move_replay_part(collected_part, device)

            if not collection_initial_states:
                perturbed_induced = []
            elif args.perturbation_collection_workers == 1:
                perturbed_induced = [
                    collect_perturbation(initial_state)
                    for initial_state in collection_initial_states
                ]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(
                        args.perturbation_collection_workers,
                        len(collection_initial_states),
                    )
                ) as pool:
                    perturbed_induced = list(
                        pool.map(
                            collect_perturbation,
                            collection_initial_states,
                        )
                    )
        new_sequences: list[dict[str, Tensor | int]] = [
            induced,
            *perturbed_induced,
        ]
        teacher_suffix: dict[str, Tensor | int] | None = None
        teacher_suffix_perturbations: dict[str, Tensor | int] | None = None
        if args.teacher_forced_recovery_suffix:
            collected_teacher_suffix = _collect_teacher_forced_recovery_suffix(
                collection_policy,
                collection_artifact,
                collection_cfg,
                args.aperture,
                args.collection_horizon,
            )
            teacher_suffix = _move_replay_part(
                collected_teacher_suffix, device
            )
            teacher_suffix["sequence_weight"] = args.teacher_suffix_sequence_weight
            new_sequences.append(teacher_suffix)
            if args.teacher_suffix_perturbations_per_tick > 0:
                collected_teacher_suffix_perturbations = (
                    _collect_corrective_suffix_perturbations(
                        collected_teacher_suffix,
                        collection_artifact,
                        collection_cfg,
                        args.aperture,
                        perturbations_per_tick=(
                            args.teacher_suffix_perturbations_per_tick
                        ),
                        perturb_position_std_m=args.perturb_position_std_m,
                        perturb_angle_std_rad=args.perturb_angle_std_rad,
                        perturb_velocity_std=args.perturb_velocity_std,
                    )
                )
                teacher_suffix_perturbations = _move_replay_part(
                    collected_teacher_suffix_perturbations, device
                )
                new_sequences.append(teacher_suffix_perturbations)
            if not args.recurrent_sequence_training:
                # The ordinary fast path samples individual rows, so represent
                # the requested corrective-sequence emphasis explicitly in the
                # replay population.  Recurrent training consumes the scalar
                # sequence weight directly and must not duplicate BPTT work.
                copies = max(1, int(round(args.teacher_suffix_sequence_weight)))
                new_sequences.extend([teacher_suffix] * (copies - 1))
        frontier_index: int | None = None
        frontier_threshold: int | None = None
        frontier_upper: int | None = None
        perturbation_frontiers: list[int] = []
        if args.frontier_focused_recovery_width > 0:
            induced_progress = induced.get("ordered_progress")
            if not isinstance(induced_progress, Tensor):
                raise RuntimeError("frontier training requires induced path progress")
            if perturbed_induced:
                for part in perturbed_induced:
                    part_progress = part.get("ordered_progress")
                    if not isinstance(part_progress, Tensor):
                        raise RuntimeError(
                            "perturbed DAgger part omitted ordered progress"
                        )
                    perturbation_frontiers.append(
                        int(part_progress.amax().item())
                    )
                frontier_index = min(perturbation_frontiers)
            else:
                frontier_index = int(induced_progress.amax().item())
            frontier_threshold = max(
                0, frontier_index - args.frontier_focused_recovery_width
            )
            frontier_upper = min(
                int(collection_artifact.horizon),
                frontier_index + args.frontier_focused_recovery_lookahead,
            )
            for part_index, part in enumerate(new_sequences):
                part_progress = part.get("ordered_progress")
                post_contact = part.get("post_contact_active")
                if not isinstance(part_progress, Tensor) or not isinstance(
                    post_contact, Tensor
                ):
                    raise RuntimeError(
                        "frontier training part omitted progress/contact tensors"
                    )
                if perturbed_induced and part_index == 0:
                    # The first sequence is the already-successful nominal
                    # rollout.  Keep it as an incumbent distillation anchor.
                    part["planner_update_target"] = torch.zeros_like(
                        post_contact, dtype=torch.bool
                    )
                else:
                    if 1 <= part_index <= len(perturbed_induced):
                        part_frontier = perturbation_frontiers[part_index - 1]
                        part_threshold = max(
                            0,
                            part_frontier - args.frontier_focused_recovery_width,
                        )
                        part_upper = min(
                            int(collection_artifact.horizon),
                            part_frontier
                            + args.frontier_focused_recovery_lookahead,
                        )
                    else:
                        part_threshold = frontier_threshold
                        part_upper = frontier_upper
                    if (
                        1 <= part_index <= len(perturbed_induced)
                        and part_frontier
                        >= int(collection_artifact.horizon) - 1
                    ):
                        # A perturbation rollout that already reaches the
                        # terminal path is now part of the protected success
                        # set.  Distill the incumbent exactly instead of
                        # re-imposing privileged teacher plans on it.
                        part["planner_update_target"] = torch.zeros_like(
                            post_contact, dtype=torch.bool
                        )
                    else:
                        part["planner_update_target"] = post_contact & (
                            part_progress >= part_threshold
                        ) & (part_progress <= part_upper)
        if args.aggregate_induced_replay:
            induced_archive.extend(new_sequences)
        else:
            induced_archive[:] = new_sequences
        replay = _merge_replay(nominal, *induced_archive)
        if args.frontier_focused_recovery_width > 0:
            replay["planner_update_target"] = torch.cat(
                [
                    torch.zeros(
                        int(nominal["observation"].shape[0]),
                        dtype=torch.bool,
                        device=device,
                    ),
                    *[
                        part["planner_update_target"]
                        for part in induced_archive
                    ],
                ],
                dim=0,
            )
        rows = int(replay["observation"].shape[0])
        if args.source_correct_recovery:
            recovery_mask_name = (
                "planner_update_target"
                if args.frontier_focused_recovery_width > 0
                else (
                    "post_contact_active"
                    if args.controller_scope.startswith("planner-recovery-adapter")
                    else "recovery_target"
                )
            )
            recovery_rows = int(replay[recovery_mask_name].sum().item())
            if not 0 < recovery_rows < rows:
                raise RuntimeError(
                    "source-correct DAgger requires both recovery and preservation rows"
                )
            replay["incumbent_plan"] = _cache_incumbent_local_plans(policy, replay)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=(
                0.0 if args.controller_scope == "planner-recovery-columns" else 1.0e-5
            ),
        )
        metrics: dict[str, float] = {}
        for _ in range(args.updates_per_iteration):
            global_step += 1
            if args.recurrent_sequence_training:
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                plan_loss, phase_loss, phase_accuracy = (
                    _recurrent_sequence_plan_loss(
                        policy,
                        induced_archive,
                        loaded.cfg,
                    )
                )
                loss = (
                    args.plan_loss_weight * plan_loss
                    + args.phase_loss_weight * phase_loss
                )
                loss.backward()
                grad = float(torch.nn.utils.clip_grad_norm_(parameters, 2.0).cpu())
                optimizer.step()
                if global_step % args.frozen_audit_interval == 0:
                    _assert_frozen(policy, frozen)
                metrics = {
                    "loss": float(loss.detach().cpu()),
                    "plan_mse": float(plan_loss.detach().cpu()),
                    "recovery_plan_mse": float(plan_loss.detach().cpu()),
                    "source_preservation_mse": 0.0,
                    "phase_cross_entropy": float(phase_loss.detach().cpu()),
                    "phase_accuracy": float(phase_accuracy.detach().cpu()),
                    "force_mse": 0.0,
                    "wrench_mse": 0.0,
                    "section_wrench_mse": 0.0,
                    "direct_unary_mse": 0.0,
                    "clearance_barrier": 0.0,
                    "gradient_norm_before_clip": grad,
                }
                continue
            if args.batch_size >= rows:
                index = torch.arange(rows, device=device)
            else:
                index = torch.randint(rows, (args.batch_size,), device=device)
            observation = replay["observation"].index_select(0, index)
            hidden = replay["hidden"].index_select(0, index)
            state = replay["state"].index_select(0, index)
            target_force = replay["target_force"].index_select(0, index)
            target_plan = replay["local_se2_plan_target"].index_select(0, index)
            aperture = replay["aperture_m"].index_select(0, index)
            target_plan_agents = expand_plan_target_for_agents(
                target_plan, loaded.cfg.n_agents
            )
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            output = None
            phase_logits = None
            if args.planner_fast_path:
                predicted_plan, phase_logits = (
                    policy.predict_local_se2_plan_mean_and_phase(
                        observation,
                        hidden,
                    )
                )
            else:
                output = policy(
                    observation,
                    edges,
                    hidden,
                    local_se2_plan_override=target_plan_agents,
                )
                predicted_plan = output.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY]
            if args.source_correct_recovery:
                recovery_target = replay[recovery_mask_name].index_select(0, index)
                incumbent_plan = replay["incumbent_plan"].index_select(0, index)
                plan_loss, recovery_plan_loss, preservation_plan_loss = (
                    _source_correct_recovery_plan_loss(
                        predicted_plan,
                        target_plan_agents,
                        incumbent_plan,
                        recovery_target,
                        preservation_weight=args.source_preservation_weight,
                        recovery_dimension_weights=(
                            args.recovery_plan_dimension_weights
                        ),
                    )
                )
            else:
                plan_loss = F.mse_loss(predicted_plan, target_plan_agents)
                recovery_plan_loss = plan_loss.new_zeros(())
                preservation_plan_loss = plan_loss.new_zeros(())
            phase_loss = plan_loss.new_zeros(())
            phase_accuracy = plan_loss.new_zeros(())
            if args.controller_scope in {
                "planner-recovery-adapter",
                "planner-recovery-adapter-recurrent",
            }:
                if phase_logits is None:
                    if output is None:
                        raise RuntimeError("planner phase logits are unavailable")
                    phase_logits = output.aux[PERSISTENT_RECOVERY_PHASE_LOGITS_AUX_KEY]
                post_contact = replay["post_contact_active"].index_select(0, index)
                phase_target = replay["local_se2_phase_target"].index_select(0, index)
                if bool(post_contact.any()):
                    active_logits = phase_logits[post_contact]
                    active_target = phase_target[post_contact].unsqueeze(1).expand(
                        -1, loaded.cfg.n_agents
                    )
                    phase_loss = F.cross_entropy(
                        active_logits.reshape(-1, active_logits.shape[-1]),
                        active_target.reshape(-1),
                    )
                    phase_accuracy = (
                        active_logits.argmax(dim=-1) == active_target
                    ).float().mean()
            if output is None:
                force_loss = plan_loss.new_zeros(())
                wrench_loss = plan_loss.new_zeros(())
                section_wrench_loss = plan_loss.new_zeros(())
                direct_unary_loss = plan_loss.new_zeros(())
                clearance_loss = plan_loss.new_zeros(())
            else:
                force_loss = (
                    ((output.forces - target_force) / loaded.cfg.max_force)
                    .square()
                    .mean()
                )
                predicted_wrench = net_wrench_from_forces(
                    state, output.forces, loaded.cfg
                )
                target_wrench = net_wrench_from_forces(
                    state, target_force, loaded.cfg
                )
                wrench_scale = state.new_tensor(
                    (
                        loaded.cfg.n_agents * loaded.cfg.max_force,
                        loaded.cfg.n_agents * loaded.cfg.max_force,
                        0.75 * loaded.cfg.n_agents * loaded.cfg.max_force,
                    )
                )
                wrench_loss = (
                    ((predicted_wrench - target_wrench) / wrench_scale)
                    .square()
                    .mean()
                )
                source_wrench = net_wrench_from_forces(
                    state, output.aux[SOURCE_FORCES_AUX_KEY], loaded.cfg
                )
                target_correction_wrench = (
                    (target_wrench - source_wrench) / wrench_scale
                ).clamp(-1.0, 1.0)
                target_correction_wrench = target_correction_wrench[:, None, :].expand(
                    -1, loaded.cfg.n_agents, -1
                )
                section_wrench_loss = F.mse_loss(
                    output.aux[CORRECTION_WRENCH_AUX_KEY],
                    target_correction_wrench,
                )
                target_unary = torch.atanh(
                    target_correction_wrench.clamp(-0.995, 0.995)
                )
                if loaded.policy.arm_cfg.local_se2_direct_wrench_unary:
                    direct_unary_loss = F.mse_loss(
                        output.aux[LOCAL_SE2_DIRECT_WRENCH_POST_AUX_KEY],
                        target_unary,
                    )
                else:
                    direct_unary_loss = section_wrench_loss.new_zeros(())
                scenario = single_slit_scenario(
                    int(index.numel()),
                    loaded.cfg,
                    initial_state=state,
                    visible_aperture_m=aperture,
                )
                proposed = step_dynamics(state, output.forces, loaded.cfg)
                clearance = transition_clearance_margin(
                    state, proposed, scenario, loaded.cfg
                )
                clearance_loss = F.softplus((0.001 - clearance) / 0.001).mean()
            loss = args.plan_loss_weight * plan_loss + args.phase_loss_weight * phase_loss
            if output is not None:
                loss = (
                    loss
                    + args.force_loss_weight * force_loss
                    + args.wrench_loss_weight * wrench_loss
                    + args.section_wrench_loss_weight * section_wrench_loss
                    + args.direct_unary_loss_weight * direct_unary_loss
                    + args.clearance_weight * clearance_loss
                )
            multi_step_metrics = {
                "multi_step_state_mse": 0.0,
                "multi_step_clearance_hinge": 0.0,
                "multi_step_collision_penalty": 0.0,
                "multi_step_minimum_clearance_m": 0.0,
                "multi_step_active_transitions": 0.0,
            }
            if args.multi_step_rollout_horizon > 0:
                if teacher_suffix is None:
                    raise RuntimeError("multi-step suffix was not collected")
                multi_step_loss, multi_step_metrics = (
                    _corrective_suffix_multiple_shooting_loss(
                        policy,
                        teacher_suffix,
                        loaded.cfg,
                        args.aperture,
                        rollout_horizon=args.multi_step_rollout_horizon,
                        window_count=args.multi_step_window_count,
                    )
                )
                loss = loss + args.multi_step_loss_weight * multi_step_loss
            loss.backward()
            if args.controller_scope == "planner-recovery-columns":
                recovery_weight = dict(policy.named_parameters())[recovery_column_name]
                if recovery_weight.grad is None:
                    raise RuntimeError("recovery-column adapter received no gradient")
                recovery_weight.grad[
                    :, : -policy.arm_cfg.persistent_recovery_dim
                ].zero_()
            grad = float(torch.nn.utils.clip_grad_norm_(parameters, 2.0).cpu())
            optimizer.step()
            if args.controller_scope == "planner-recovery-columns":
                assert recovery_column_prefix is not None
                recovery_weight = dict(policy.named_parameters())[recovery_column_name]
                if not torch.equal(
                    recovery_weight[
                        :, : -policy.arm_cfg.persistent_recovery_dim
                    ],
                    recovery_column_prefix,
                ):
                    raise RuntimeError("optimizer changed a non-recovery planner column")
            if global_step % args.frozen_audit_interval == 0:
                _assert_frozen(policy, frozen)
            metrics = {
                "loss": float(loss.detach().cpu()),
                "plan_mse": float(plan_loss.detach().cpu()),
                "recovery_plan_mse": float(recovery_plan_loss.detach().cpu()),
                "source_preservation_mse": float(
                    preservation_plan_loss.detach().cpu()
                ),
                "phase_cross_entropy": float(phase_loss.detach().cpu()),
                "phase_accuracy": float(phase_accuracy.detach().cpu()),
                "force_mse": float(force_loss.detach().cpu()),
                "wrench_mse": float(wrench_loss.detach().cpu()),
                "section_wrench_mse": float(section_wrench_loss.detach().cpu()),
                "direct_unary_mse": float(direct_unary_loss.detach().cpu()),
                "clearance_barrier": float(clearance_loss.detach().cpu()),
                "gradient_norm_before_clip": grad,
                **multi_step_metrics,
            }
        _assert_frozen(policy, frozen)
        proposal_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
        best_key = incumbent_key
        best_state = incumbent_state
        best_reports = incumbent_reports
        best_alpha = None
        trials = []
        for alpha in alphas:
            _load_trainable_interpolation(
                policy, incumbent_state, proposal_state, selected, alpha
            )
            _assert_frozen(policy, frozen)
            if args.selection_mode == "oracle":
                trial_predicted, trial_oracle = _authority_reports(
                    policy,
                    loaded.cfg,
                    artifact_path,
                    args.aperture,
                    args.evaluation_horizon,
                    authority_device,
                )
            else:
                trial_predicted = _predicted_authority_report(
                    policy,
                    loaded.cfg,
                    artifact_path,
                    args.aperture,
                    args.evaluation_horizon,
                    authority_device,
                    initial_states=(
                        perturbation_panel
                        if args.model_induced_initial_perturbation_episodes > 0
                        else None
                    ),
                    population_workers=args.perturbation_authority_workers,
                    required_success_indices=tuple(
                        int(index)
                        for index in incumbent_reports[0].get(
                            "success_indices", ()
                        )
                    ),
                )
                # Exact plan override bypasses this adapter and every
                # downstream tensor is bitwise frozen, so this audited oracle
                # report is invariant across candidate interpolations.
                trial_oracle = incumbent_reports[1]
            authority = trial_oracle if args.selection_mode == "oracle" else trial_predicted
            key = _optimization_key(authority, args.selection_mode)
            trials.append(
                {
                    "alpha": alpha,
                    "selection_key": list(key),
                    "strict_selection_key": authority["selection_key"],
                    "predicted_success": trial_predicted["success"],
                    "oracle_success": trial_oracle["success"],
                }
            )
            if key > best_key:
                best_key = key
                best_alpha = alpha
                best_reports = (trial_predicted, trial_oracle)
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                }
            if authority["success"]:
                break
            if (
                args.break_on_frontier_advance
                and args.selection_mode == "predicted"
                and not bool(
                    trial_predicted.get("nominal_failure_short_circuit", False)
                )
                and not bool(
                    trial_predicted.get(
                        "protected_success_failure_short_circuit", False
                    )
                )
                and (
                    int(trial_predicted.get("success_count", 0))
                    > int(incumbent_reports[0].get("success_count", 0))
                    or (
                        int(trial_predicted.get("success_count", 0))
                        == int(incumbent_reports[0].get("success_count", 0))
                        and int(trial_predicted["ordered_teacher_path_index"])
                        > int(incumbent_reports[0]["ordered_teacher_path_index"])
                    )
                )
            ):
                break
        admitted = best_alpha is not None
        policy.load_state_dict(best_state if admitted else incumbent_state, strict=True)
        if admitted:
            incumbent_key = best_key
            incumbent_state = best_state
            incumbent_reports = best_reports
            payload = {
                "schema_version": SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
                "step": global_step,
                "training_stage": TRAINING_STAGE,
                "arm": loaded.arm,
                "observation_schema": loaded.observation_schema,
                "env_cfg": asdict(loaded.cfg),
                "arm_cfg": asdict(policy.arm_cfg),
                "parent": {"path": str(source_path), "sha256": _sha256(source_path)},
                "teacher": {
                    "path": str(teacher_path),
                    "sha256": _sha256(teacher_path),
                    "rollout_path": str(artifact_path),
                    "rollout_sha256": _sha256(artifact_path),
                },
                "selection_mode": args.selection_mode,
                "collection_mode": collection_mode,
                "collection_device": str(collection_device),
                "aggregate_induced_replay": args.aggregate_induced_replay,
                "model_induced_initial_perturbations": {
                    "episodes": args.model_induced_initial_perturbation_episodes,
                    "seed": args.seed + 7919,
                    "position_cap_m": args.initial_perturb_position_cap_m,
                    "angle_cap_rad": args.initial_perturb_angle_cap_rad,
                    "velocity_cap": args.initial_perturb_velocity_cap,
                    "angular_velocity_cap": (
                        args.initial_perturb_velocity_cap
                        if args.initial_perturb_angular_velocity_cap is None
                        else args.initial_perturb_angular_velocity_cap
                    ),
                    "collect_failed_only": args.collect_failed_perturbations_only,
                    "collected_panel_indices": collection_panel_indices,
                    "collection_contact_tail_steps": (
                        args.perturbation_collection_contact_tail_steps
                    ),
                },
                "replay_source_checkpoint": [
                    str(path) for path in replay_source_paths
                ],
                "controller_scope": args.controller_scope,
                "planner_fast_path": args.planner_fast_path,
                "recurrent_sequence_training": args.recurrent_sequence_training,
                "teacher_forced_recovery_suffix": (
                    args.teacher_forced_recovery_suffix
                ),
                "teacher_suffix_sequence_weight": (
                    args.teacher_suffix_sequence_weight
                ),
                "multi_step_training": {
                    "rollout_horizon": args.multi_step_rollout_horizon,
                    "window_count": args.multi_step_window_count,
                    "loss_weight": args.multi_step_loss_weight,
                },
                "frontier_focused_recovery": {
                    "width": args.frontier_focused_recovery_width,
                    "lookahead": args.frontier_focused_recovery_lookahead,
                    "frontier_index": frontier_index,
                    "threshold_index": frontier_threshold,
                    "upper_index": frontier_upper,
                    "perturbation_frontiers": perturbation_frontiers,
                },
                "training_aperture_m": args.aperture,
                "trainable_scope": {
                    "parameter_names": list(selected),
                    "parameter_count": sum(
                        dict(policy.named_parameters())[name].numel()
                        for name in selected
                    ),
                },
                "authority": {
                    "device": str(authority_device),
                    "predicted": best_reports[0],
                    "oracle": best_reports[1],
                },
                "architecture": architecture_record(policy, edges),
                "communication": communication_record(policy, edges, active_ticks=1),
                "policy_state_dict": {
                    name: value.detach().cpu()
                    for name, value in policy.state_dict().items()
                },
            }
            torch.save(payload, output_dir / f"iteration_{iteration:03d}_selected.pt")
        record = {
            "iteration": iteration,
            "global_step": global_step,
            "nominal_rows": int(nominal["observation"].shape[0]),
            "induced_rows": int(induced["observation"].shape[0]),
            "perturbed_induced_episodes": len(perturbed_induced),
            "perturbed_induced_rows": sum(
                int(part["observation"].shape[0]) for part in perturbed_induced
            ),
            "archived_induced_rows": sum(
                int(part["observation"].shape[0]) for part in induced_archive
            ),
            "metrics": metrics,
            "selection_mode": args.selection_mode,
            "collection_mode": collection_mode,
            "collection_device": str(collection_device),
            "aggregate_induced_replay": args.aggregate_induced_replay,
            "replay_source_checkpoint": [str(path) for path in replay_source_paths],
            "controller_scope": args.controller_scope,
            "planner_fast_path": args.planner_fast_path,
            "recurrent_sequence_training": args.recurrent_sequence_training,
            "teacher_forced_recovery_suffix": (
                None
                if teacher_suffix is None
                else {
                    "rows": int(teacher_suffix["observation"].shape[0]),
                    "success": bool(teacher_suffix["teacher_suffix_success"]),
                    "trigger_tick": int(
                        teacher_suffix["teacher_suffix_trigger_tick"]
                    ),
                    "finish_tick": int(
                        teacher_suffix["teacher_suffix_finish_tick"]
                    ),
                    "query_failures": int(teacher_suffix["query_failures"]),
                    "sequence_weight": float(
                        teacher_suffix["sequence_weight"]
                    ),
                }
            ),
            "teacher_suffix_perturbations": (
                None
                if teacher_suffix_perturbations is None
                else {
                    "rows": int(
                        teacher_suffix_perturbations["observation"].shape[0]
                    ),
                    "query_failures": int(
                        teacher_suffix_perturbations["query_failures"]
                    ),
                }
            ),
            "frontier_focused_recovery": {
                "width": args.frontier_focused_recovery_width,
                "lookahead": args.frontier_focused_recovery_lookahead,
                "frontier_index": frontier_index,
                "threshold_index": frontier_threshold,
                "upper_index": frontier_upper,
                "perturbation_frontiers": perturbation_frontiers,
                "exact_update_rows": (
                    int(replay["planner_update_target"].sum().item())
                    if args.frontier_focused_recovery_width > 0
                    else None
                ),
            },
            "multi_step_training": {
                "rollout_horizon": args.multi_step_rollout_horizon,
                "window_count": args.multi_step_window_count,
                "loss_weight": args.multi_step_loss_weight,
            },
            "selected": admitted,
            "selected_alpha": best_alpha,
            "predicted": incumbent_reports[0],
            "oracle": incumbent_reports[1],
            "trials": trials,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        if incumbent_reports[0]["success"]:
            break


if __name__ == "__main__":
    main()
