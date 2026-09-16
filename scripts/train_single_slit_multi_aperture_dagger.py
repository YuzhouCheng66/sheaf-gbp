#!/usr/bin/env python3
"""Model-induced DAgger for multi-aperture role-factor Sheaf-GBP."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from piano_movers.env import (
    net_wrench_from_forces,
    radius_edges,
    step_dynamics,
    transition_clearance_margin,
)
from piano_movers.single_slit_0720_bc import (
    SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
    architecture_record,
    communication_record,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_feedback import single_slit_contact_feedback_step_v1
from piano_movers.single_slit_contact_feedback_eval_v2 import evaluate_single_slit_contact_feedback_v2
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_dagger import query_safe_feedback_teacher
from piano_movers.single_slit_dataset import load_executable_teacher_artifact
from piano_movers.single_slit_feedback import GOAL_POSE, single_slit_scenario
from piano_movers.single_slit_local_planner import local_se2_plan_target
from scripts.bridge_single_slit_collision_role_factors import (
    BRIDGE_STAGE,
    ROLE_PARAMETER_COUNT,
    ROLE_PARAMETER_KEYS,
)
from scripts.bridge_single_slit_separate_role_encoder import (
    BRIDGE_STAGE as SEPARATE_ROLE_ENCODER_BRIDGE_STAGE,
)
from scripts.train_single_slit_multi_aperture_roles import (
    FULL_CORRECTION_TRAINING_STAGE,
    TRAINING_STAGE as NOMINAL_STAGE,
)


TRAINING_STAGE = "collision-role-multi-aperture-model-induced-dagger-v1"
LEFT_TO_RIGHT_PROGRESS_BIN_M = 5.0e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_trainable(
    policy: torch.nn.Module,
    *,
    train_scope: str = "roles",
) -> tuple[str, ...]:
    if train_scope == "roles":
        expected = ROLE_PARAMETER_KEYS
    elif train_scope == "role-encoder":
        expected = tuple(
            name
            for name, _ in policy.named_parameters()
            if name.startswith("correction_role_encoder.")
        )
        if not expected:
            raise RuntimeError("role-encoder scope requires the separate role encoder")
    elif train_scope == "role-branch":
        expected = tuple(
            name
            for name, _ in policy.named_parameters()
            if name.startswith("correction_role_")
        )
        if not any(name.startswith("correction_role_encoder.") for name in expected):
            raise RuntimeError("role-branch scope requires the separate role encoder")
    elif train_scope == "full-correction":
        expected = tuple(name for name, _ in policy.correction_named_parameters())
        if not expected:
            raise RuntimeError("full-correction scope requires an enabled correction sheaf")
    else:
        raise ValueError(f"unknown train scope: {train_scope!r}")
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name in expected)
    selected = tuple(name for name, value in policy.named_parameters() if value.requires_grad)
    count = sum(dict(policy.named_parameters())[name].numel() for name in selected)
    if selected != expected:
        raise RuntimeError(f"DAgger {train_scope} trainable scope changed")
    if train_scope == "roles" and count != ROLE_PARAMETER_COUNT:
        raise RuntimeError("DAgger role parameter count changed")
    return selected


def _frozen_reference(
    policy: torch.nn.Module,
    selected_parameter_names: tuple[str, ...] = ROLE_PARAMETER_KEYS,
) -> dict[str, Tensor]:
    selected = set(selected_parameter_names)
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
        if name not in selected
    }


def _assert_frozen(policy: torch.nn.Module, reference: Mapping[str, Tensor]) -> None:
    if any(
        not torch.equal(policy.state_dict()[name].detach().cpu(), value)
        for name, value in reference.items()
    ):
        raise RuntimeError("DAgger changed a frozen source tensor")


@torch.no_grad()
def _load_trainable_interpolation(
    policy: torch.nn.Module,
    incumbent_state: Mapping[str, Tensor],
    proposal_state: Mapping[str, Tensor],
    selected: tuple[str, ...],
    alpha: float,
) -> None:
    """Load an exact source-frozen point on the optimizer proposal segment."""

    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("backtracking alpha must lie in (0, 1]")
    parameters = dict(policy.named_parameters())
    for name in selected:
        target = incumbent_state[name].to(
            device=parameters[name].device,
            dtype=parameters[name].dtype,
        )
        proposal = proposal_state[name].to(
            device=parameters[name].device,
            dtype=parameters[name].dtype,
        )
        parameters[name].copy_(target + alpha * (proposal - target))


@torch.no_grad()
def _collect(
    policy: torch.nn.Module,
    artifact: Any,
    cfg: Any,
    apertures: tuple[float, ...],
    *,
    horizon: int,
) -> dict[str, Tensor | int]:
    edges = radius_edges(cfg, device=cfg.device)
    observations: list[Tensor] = []
    hiddens: list[Tensor] = []
    states: list[Tensor] = []
    targets: list[Tensor] = []
    aperture_rows: list[Tensor] = []
    contact_rows: list[Tensor] = []
    contact_age_rows: list[Tensor] = []
    local_plan_rows: list[Tensor] = []
    query_failures = 0
    was_training = policy.training
    policy.eval()
    try:
        for aperture in apertures:
            scenario = single_slit_scenario(
                1, cfg, visible_aperture_m=aperture
            )
            state = scenario.state0.clone()
            hidden = policy.init_hidden(1, state.device)
            context = initial_single_slit_contact_observation_context_v1(state, cfg)
            contact_age = torch.zeros(1, dtype=torch.long, device=state.device)
            progress = 0
            for _tick in range(horizon):
                observation = single_slit_contact_observations_v1(
                    state, scenario, cfg, context
                )
                label, error = query_safe_feedback_teacher(
                    state,
                    progress,
                    artifact,
                    cfg,
                    visible_aperture_m=aperture,
                )
                if label is None:
                    query_failures += 1
                else:
                    progress = max(progress, label.nearest_index)
                    observations.append(observation[0].detach().clone())
                    hiddens.append(hidden[0].detach().clone())
                    states.append(state[0].detach().clone())
                    targets.append(label.forces.detach().clone())
                    aperture_rows.append(state.new_tensor(aperture))
                    contact_rows.append(context.last_hit.to(state.dtype)[0].clone())
                    contact_age_rows.append(contact_age[0].clone())
                    local_plan_rows.append(local_se2_plan_target(state, label))
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
    finally:
        policy.train(was_training)
    if not observations:
        raise RuntimeError("model-induced DAgger collected no safe teacher labels")
    return {
        "observation": torch.stack(observations),
        "hidden": torch.stack(hiddens),
        "state": torch.stack(states),
        "target_force": torch.stack(targets),
        "aperture_m": torch.stack(aperture_rows),
        "last_hit": torch.stack(contact_rows),
        "contact_age": torch.stack(contact_age_rows),
        "local_se2_plan_target": torch.stack(local_plan_rows),
        "query_failures": query_failures,
    }


def _evaluate(policy: torch.nn.Module, cfg: Any, apertures: tuple[float, ...], horizon: int) -> dict[str, Any]:
    episodes = []
    for aperture in apertures:
        episode = evaluate_single_slit_contact_feedback_v2(
            policy,
            cfg,
            episodes=1,
            horizon=horizon,
            visible_aperture_m=aperture,
        )["per_episode"][0]
        episodes.append({
            "visible_aperture_m": aperture,
            "success": bool(episode["success"]),
            "finish_step": int(episode["finish_step"]),
            "contact_count": int(episode["contact_count"]),
            "stuck_tick_count": int(episode["stuck_tick_count"]),
            "maximum_consecutive_contact_steps": int(episode["maximum_consecutive_contact_steps"]),
            "final_position_error_m": float(episode["final_position_error_m"]),
            "final_angle_error_rad": float(episode["final_angle_error_rad"]),
            "final_x_m": float(episode["final_resolved_state"][0]),
        })
    success_count = sum(int(value["success"]) for value in episodes)
    progress_bins = sum(
        math.floor(
            min(value["final_x_m"], float(GOAL_POSE[0]))
            / LEFT_TO_RIGHT_PROGRESS_BIN_M
            + 1.0e-6
        )
        for value in episodes
    )
    # First move right in physically meaningful 5 mm increments; within a
    # progress bin reduce jamming before terminal pose errors. Capping at the
    # goal prevents an unstable overshoot from winning.
    key = (
        float(success_count),
        float(progress_bins),
        -float(sum(value["maximum_consecutive_contact_steps"] for value in episodes)),
        -float(sum(value["stuck_tick_count"] for value in episodes)),
        -float(sum(value["contact_count"] for value in episodes)),
        -float(sum(value["final_position_error_m"] for value in episodes)),
        -float(sum(value["final_angle_error_rad"] for value in episodes)),
    )
    return {
        "success_count": success_count,
        "episode_count": len(episodes),
        "success_rate": success_count / len(episodes),
        "per_aperture": episodes,
        "selection_key": list(key),
        "teacher_used_at_inference": False,
        "exact_sat_contact_feedback": True,
    }


def _evaluate_on_authority_device(
    policy: torch.nn.Module,
    cfg: Any,
    apertures: tuple[float, ...],
    horizon: int,
    authority_device: torch.device,
) -> dict[str, Any]:
    """Evaluate a snapshot on a deterministic device, independent of training CUDA."""

    model_device = next(policy.parameters()).device
    if model_device == authority_device and torch.device(cfg.device) == authority_device:
        return _evaluate(policy, cfg, apertures, horizon)
    authority_cfg = replace(cfg, device=str(authority_device))
    authority_policy = copy.deepcopy(policy).to(authority_device)
    authority_policy.env_cfg = authority_cfg
    return _evaluate(authority_policy, authority_cfg, apertures, horizon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-rollout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--authority-device",
        default="cpu",
        help="device used for checkpoint selection and strict exact-SAT replay",
    )
    parser.add_argument("--apertures", type=float, nargs="+", default=(0.80, 0.84, 0.88, 0.92))
    parser.add_argument(
        "--train-scope",
        choices=("roles", "role-encoder", "role-branch", "full-correction"),
        default="roles",
        help="train only role heads or the complete correction Sheaf",
    )
    parser.add_argument(
        "--precontact-leader-follower",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "allow leader/follower factors before contact; disable for an exact "
            "source-preserving contact-triggered Rebel recovery stage"
        ),
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--updates-per-iteration", type=int, default=20)
    parser.add_argument("--collection-horizon", type=int, default=128)
    parser.add_argument("--evaluation-horizon", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--clearance-weight", type=float, default=0.01)
    parser.add_argument(
        "--wrench-weight",
        type=float,
        default=1.0,
        help="weight on normalized shared [Fx,Fy,tau] imitation",
    )
    parser.add_argument(
        "--contact-row-weight",
        type=float,
        default=4.0,
        help="additional weight for model-induced rows with local rejected-contact evidence",
    )
    parser.add_argument(
        "--min-contact-age",
        type=int,
        default=0,
        help="train only replay rows after at least this many consecutive rejected contacts",
    )
    parser.add_argument("--clearance-target-m", type=float, default=1.0e-3)
    parser.add_argument("--clearance-temperature-m", type=float, default=1.0e-3)
    parser.add_argument(
        "--backtracking-alphas",
        type=float,
        nargs="+",
        default=(1.0, 0.3, 0.1, 0.03, 0.01, 0.003),
        help="exact-evaluation line-search fractions of each optimizer proposal",
    )
    parser.add_argument("--stop-on-first-success", action="store_true")
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    integers = (args.iterations, args.updates_per_iteration, args.collection_horizon, args.evaluation_horizon, args.batch_size)
    if any(value <= 0 for value in integers):
        raise ValueError("DAgger integer controls must be positive")
    apertures = tuple(float(value) for value in args.apertures)
    if not apertures or any(not math.isfinite(value) or value <= 0.720 for value in apertures):
        raise ValueError("DAgger apertures must be finite and wider than 0.720")
    backtracking_alphas = tuple(float(value) for value in args.backtracking_alphas)
    if (
        not backtracking_alphas
        or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in backtracking_alphas)
        or len(set(backtracking_alphas)) != len(backtracking_alphas)
    ):
        raise ValueError("backtracking alphas must be unique finite values in (0,1]")
    if args.wrench_weight < 0.0 or args.contact_row_weight < 0.0:
        raise ValueError("wrench and contact-row weights must be non-negative")
    if args.min_contact_age < 0:
        raise ValueError("minimum contact age must be non-negative")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    authority_device = torch.device(args.authority_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    source_path = args.source_checkpoint.resolve(strict=True)
    teacher_path = args.teacher_rollout.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    loaded = load_single_slit_checkpoint(source_path, device=device)
    if loaded.metadata.get("training_stage") not in {
        BRIDGE_STAGE,
        SEPARATE_ROLE_ENCODER_BRIDGE_STAGE,
        NOMINAL_STAGE,
        FULL_CORRECTION_TRAINING_STAGE,
        TRAINING_STAGE,
    }:
        raise ValueError("DAgger requires the role bridge or a selected multi-aperture checkpoint")
    policy = loaded.policy
    # The bridge is forward-identical at zero initialization.  Opting into
    # leader/follower factors before first contact makes the DAgger correction
    # trainable on the model's actual approach states; rebel remains strictly
    # gated by locally observed rejected contact history in the policy.
    policy.arm_cfg = replace(
        policy.arm_cfg,
        collision_role_precontact_leader_follower=args.precontact_leader_follower,
    )
    selected = _configure_trainable(policy, train_scope=args.train_scope)
    selected_count = sum(dict(policy.named_parameters())[name].numel() for name in selected)
    parameters = [dict(policy.named_parameters())[name] for name in selected]
    frozen = _frozen_reference(policy, selected)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1.0e-5)
    artifact = load_executable_teacher_artifact(teacher_path, loaded.cfg)
    edges = radius_edges(loaded.cfg, device=device)
    initial = _evaluate_on_authority_device(
        policy,
        loaded.cfg,
        apertures,
        args.evaluation_horizon,
        authority_device,
    )
    incumbent_key = tuple(float(value) for value in initial["selection_key"])
    incumbent_evaluation = initial
    incumbent_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    print(json.dumps({"iteration": 0, "evaluation": initial}, allow_nan=False), flush=True)
    log_path = output_dir / "train.jsonl"
    global_step = 0
    for iteration in range(1, args.iterations + 1):
        replay = _collect(
            policy,
            artifact,
            loaded.cfg,
            apertures,
            horizon=args.collection_horizon,
        )
        row_count = int(replay["observation"].shape[0])
        eligible = torch.where(replay["contact_age"] >= args.min_contact_age)[0]
        eligible_count = int(eligible.numel())
        if eligible_count == 0:
            raise RuntimeError("no DAgger replay rows satisfy the contact-age curriculum")
        hit_fraction = float(replay["last_hit"].mean().detach().cpu())
        last_loss = math.nan
        last_force_loss = math.nan
        last_wrench_loss = math.nan
        last_grad = math.nan
        for _update in range(args.updates_per_iteration):
            global_step += 1
            if args.batch_size >= eligible_count:
                index = eligible
            else:
                draw = torch.randint(eligible_count, (args.batch_size,), device=device)
                index = eligible.index_select(0, draw)
            observation = replay["observation"].index_select(0, index)
            hidden = replay["hidden"].index_select(0, index)
            state = replay["state"].index_select(0, index)
            target = replay["target_force"].index_select(0, index)
            last_hit = replay["last_hit"].index_select(0, index)
            aperture = replay["aperture_m"].index_select(0, index)
            scenario = single_slit_scenario(
                int(index.numel()),
                loaded.cfg,
                initial_state=state,
                visible_aperture_m=aperture,
            )
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            output = policy(observation, edges, hidden)
            row_contact = last_hit.reshape(last_hit.shape[0], -1).amax(dim=-1)
            row_weight = 1.0 + args.contact_row_weight * row_contact
            row_weight_sum = row_weight.sum().clamp_min(1.0)
            force_per_row = (
                ((output.forces - target) / loaded.cfg.max_force)
                .square()
                .mean(dim=(-2, -1))
            )
            if row_weight.shape != force_per_row.shape:
                raise RuntimeError("contact weights must have exactly one scalar per replay row")
            force_loss = (row_weight * force_per_row).sum() / row_weight_sum
            predicted_wrench = net_wrench_from_forces(
                state,
                output.forces,
                loaded.cfg,
            )
            target_wrench = net_wrench_from_forces(state, target, loaded.cfg)
            wrench_scale = state.new_tensor(
                (
                    loaded.cfg.n_agents * loaded.cfg.max_force,
                    loaded.cfg.n_agents * loaded.cfg.max_force,
                    0.75 * loaded.cfg.n_agents * loaded.cfg.max_force,
                )
            )
            wrench_per_row = (
                ((predicted_wrench - target_wrench) / wrench_scale)
                .square()
                .mean(dim=-1)
            )
            wrench_loss = (row_weight * wrench_per_row).sum() / row_weight_sum
            proposed = step_dynamics(state, output.forces, loaded.cfg)
            clearance = transition_clearance_margin(state, proposed, scenario, loaded.cfg)
            barrier = F.softplus(
                (args.clearance_target_m - clearance) / args.clearance_temperature_m
            ).mean()
            loss = (
                force_loss
                + args.wrench_weight * wrench_loss
                + args.clearance_weight * barrier
            )
            loss.backward()
            last_grad = float(torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm).detach().cpu())
            optimizer.step()
            _assert_frozen(policy, frozen)
            last_loss = float(loss.detach().cpu())
            last_force_loss = float(force_loss.detach().cpu())
            last_wrench_loss = float(wrench_loss.detach().cpu())
        proposal_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
        policy.eval()
        best_key = incumbent_key
        best_state = incumbent_state
        best_evaluation = incumbent_evaluation
        best_alpha: float | None = None
        backtracking_records: list[dict[str, Any]] = []
        for alpha in backtracking_alphas:
            _load_trainable_interpolation(
                policy,
                incumbent_state,
                proposal_state,
                selected,
                alpha,
            )
            _assert_frozen(policy, frozen)
            trial_evaluation = _evaluate_on_authority_device(
                policy,
                loaded.cfg,
                apertures,
                args.evaluation_horizon,
                authority_device,
            )
            trial_key = tuple(
                float(value) for value in trial_evaluation["selection_key"]
            )
            backtracking_records.append(
                {
                    "alpha": alpha,
                    "selection_key": list(trial_key),
                    "success_count": int(trial_evaluation["success_count"]),
                }
            )
            if trial_key > best_key:
                best_key = trial_key
                best_alpha = alpha
                best_evaluation = trial_evaluation
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                }
            if trial_evaluation["success_count"] > 0:
                break
        admitted = best_alpha is not None
        evaluation = best_evaluation
        record = {
            "iteration": iteration,
            "global_step": global_step,
            "collection_rows": row_count,
            "training_rows": eligible_count,
            "minimum_contact_age": args.min_contact_age,
            "collection_hit_fraction": hit_fraction,
            "teacher_query_failures": int(replay["query_failures"]),
            "last_loss": last_loss,
            "last_force_loss": last_force_loss,
            "last_wrench_loss": last_wrench_loss,
            "last_gradient_norm_before_clip": last_grad,
            "evaluation": evaluation,
            "selected": admitted,
            "selected_backtracking_alpha": best_alpha,
            "backtracking": backtracking_records,
            "selection_authority_device": str(authority_device),
        }
        if admitted:
            incumbent_key = best_key
            incumbent_state = best_state
            incumbent_evaluation = best_evaluation
            policy.load_state_dict(incumbent_state, strict=True)
            payload = {
                "schema_version": SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
                "step": global_step,
                "training_stage": TRAINING_STAGE,
                "arm": loaded.arm,
                "observation_schema": loaded.observation_schema,
                "env_cfg": asdict(loaded.cfg),
                "arm_cfg": asdict(policy.arm_cfg),
                "parent": {"path": str(source_path), "sha256": _sha256(source_path)},
                "teacher": {"path": str(teacher_path), "sha256": _sha256(teacher_path)},
                "training_apertures_m": list(apertures),
                "trainable_scope": {
                    "name": args.train_scope,
                    "parameter_names": list(selected),
                    "parameter_count": selected_count,
                },
                "authority": evaluation,
                "architecture": architecture_record(policy, edges),
                "communication": communication_record(policy, edges, active_ticks=1),
                "optimizer_state_dict": optimizer.state_dict(),
                "policy_state_dict": {name: value.detach().cpu() for name, value in policy.state_dict().items()},
            }
            torch.save(payload, output_dir / f"iteration_{iteration:03d}_selected.pt")
        else:
            policy.load_state_dict(incumbent_state, strict=True)
            record["reverted_to_incumbent"] = True
        # Adam moments belong to the full proposal, not necessarily to the
        # admitted line-search point. Restarting keeps the next direction
        # faithful to the currently authenticated policy.
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1.0e-5)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        if args.stop_on_first_success and incumbent_key[0] >= 1.0:
            break


if __name__ == "__main__":
    main()
