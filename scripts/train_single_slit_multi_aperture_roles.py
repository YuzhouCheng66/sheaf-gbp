#!/usr/bin/env python3
"""Bootstrap leader/follower Sheaf factors on a non-extreme aperture curriculum."""

from __future__ import annotations

import argparse
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
    load_executable_teacher,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_feedback_eval_v2 import (
    evaluate_single_slit_contact_feedback_v2,
)
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import GOAL_POSE, single_slit_scenario
from scripts.bridge_single_slit_collision_role_factors import (
    BRIDGE_STAGE,
    ROLE_PARAMETER_COUNT,
    ROLE_PARAMETER_KEYS,
)


TRAINING_STAGE = "collision-role-multi-aperture-nominal-v1"
FULL_CORRECTION_TRAINING_STAGE = "collision-role-full-correction-multi-aperture-nominal-v2"
LEFT_TO_RIGHT_PROGRESS_BIN_M = 5.0e-3
DEFAULT_TEACHER = Path(
    "artifacts/aperture/single_slit_T_executable_teacher_0p720_20260828.json"
)


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
    elif train_scope == "full-correction":
        expected = tuple(name for name, _ in policy.correction_named_parameters())
        if not expected:
            raise RuntimeError("full-correction scope requires an enabled correction sheaf")
    else:
        raise ValueError(f"unknown train scope: {train_scope!r}")
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name in expected)
    selected = tuple(
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    )
    if selected != expected:
        raise RuntimeError(f"multi-aperture {train_scope} trainable scope changed")
    count = sum(dict(policy.named_parameters())[name].numel() for name in selected)
    if train_scope == "roles" and count != ROLE_PARAMETER_COUNT:
        raise RuntimeError("multi-aperture role parameter count changed")
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
    changed = [
        name
        for name, value in reference.items()
        if not torch.equal(policy.state_dict()[name].detach().cpu(), value)
    ]
    if changed:
        raise RuntimeError("multi-aperture training changed frozen source tensors")


def _curriculum_loss(
    policy: torch.nn.Module,
    teacher: Any,
    cfg: Any,
    apertures: Tensor,
    *,
    chunk_steps: int,
    clearance_weight: float,
    clearance_target_m: float,
    clearance_temperature_m: float,
) -> tuple[Tensor, dict[str, float]]:
    batch = int(apertures.numel())
    initial = teacher.states[:1].expand(batch, -1)
    scenario = single_slit_scenario(
        batch,
        cfg,
        initial_state=initial,
        visible_aperture_m=apertures,
    )
    context = initial_single_slit_contact_observation_context_v1(initial, cfg)
    edge_index = radius_edges(cfg, device=teacher.states.device)
    hidden = policy.init_hidden(batch, teacher.states.device)
    force_losses: list[Tensor] = []
    wrench_losses: list[Tensor] = []
    clearance_losses: list[Tensor] = []
    clearances: list[Tensor] = []
    wrench_scale = teacher.states.new_tensor(
        (cfg.n_agents * cfg.max_force, cfg.n_agents * cfg.max_force, 0.75 * cfg.n_agents * cfg.max_force)
    )
    for tick in range(teacher.horizon):
        if tick and tick % chunk_steps == 0:
            hidden = hidden.detach()
        state = teacher.states[tick : tick + 1].expand(batch, -1)
        observation = single_slit_contact_observations_v1(
            state, scenario, cfg, context
        )
        output = policy(observation, edge_index, hidden)
        hidden = output.hidden
        target_force = teacher.forces[tick : tick + 1].expand(batch, -1, -1)
        force_losses.append(
            ((output.forces - target_force) / cfg.max_force).square().mean()
        )
        predicted_wrench = net_wrench_from_forces(state, output.forces, cfg)
        target_wrench = teacher.realized_wrenches[tick : tick + 1].expand(batch, -1)
        wrench_losses.append(
            ((predicted_wrench - target_wrench) / wrench_scale).square().mean()
        )
        proposed = step_dynamics(state, output.forces, cfg)
        clearance = transition_clearance_margin(state, proposed, scenario, cfg)
        clearances.append(clearance)
        clearance_losses.append(
            F.softplus(
                (clearance_target_m - clearance) / clearance_temperature_m
            ).mean()
        )
    force = torch.stack(force_losses).mean()
    wrench = torch.stack(wrench_losses).mean()
    clearance = torch.stack(clearance_losses).mean()
    total = force + 0.25 * wrench + clearance_weight * clearance
    minimum_clearance = torch.stack(clearances).amin()
    if not bool(torch.isfinite(total)):
        raise RuntimeError("multi-aperture curriculum loss is non-finite")
    return total, {
        "force_mse": float(force.detach().cpu()),
        "wrench_mse": float(wrench.detach().cpu()),
        "clearance_barrier": float(clearance.detach().cpu()),
        "minimum_predicted_clearance_m": float(minimum_clearance.detach().cpu()),
    }


def _evaluate(
    policy: torch.nn.Module,
    cfg: Any,
    apertures: tuple[float, ...],
    *,
    horizon: int,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for aperture in apertures:
        report = evaluate_single_slit_contact_feedback_v2(
            policy,
            cfg,
            episodes=1,
            horizon=horizon,
            visible_aperture_m=aperture,
        )
        episode = report["per_episode"][0]
        episodes.append(
            {
                "visible_aperture_m": aperture,
                "success": bool(episode["success"]),
                "finish_step": int(episode["finish_step"]),
                "contact_count": int(episode["contact_count"]),
                "stuck_tick_count": int(episode["stuck_tick_count"]),
                "maximum_consecutive_contact_steps": int(
                    episode["maximum_consecutive_contact_steps"]
                ),
                "final_position_error_m": float(episode["final_position_error_m"]),
                "final_angle_error_rad": float(episode["final_angle_error_rad"]),
                "final_x_m": float(episode["final_resolved_state"][0]),
            }
        )
    success_count = sum(int(value["success"]) for value in episodes)
    progress_bins = sum(
        math.floor(
            min(value["final_x_m"], float(GOAL_POSE[0]))
            / LEFT_TO_RIGHT_PROGRESS_BIN_M
            + 1.0e-6
        )
        for value in episodes
    )
    # Capped, quantized left-to-right progress prevents both retreat and
    # overshoot while avoiding checkpoint churn over sub-millimetre noise.
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
        "exact_sat_contact_feedback": True,
        "teacher_used_at_inference": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--apertures", type=float, nargs="+", default=(0.80, 0.84, 0.88, 0.92))
    parser.add_argument(
        "--train-scope",
        choices=("roles", "full-correction"),
        default="roles",
        help="train only the role heads or the complete correction Sheaf while freezing the source controller",
    )
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--report-interval", type=int, default=5)
    parser.add_argument("--evaluation-horizon", type=int, default=256)
    parser.add_argument("--chunk-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--clearance-weight", type=float, default=0.01)
    parser.add_argument("--clearance-target-m", type=float, default=2.0e-3)
    parser.add_argument("--clearance-temperature-m", type=float, default=2.0e-3)
    parser.add_argument("--stop-on-first-success", action="store_true")
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.report_interval <= 0 or args.chunk_steps <= 0:
        raise ValueError("curriculum integer controls must be positive")
    apertures = tuple(float(value) for value in args.apertures)
    if not apertures or any(
        not math.isfinite(value) or value <= 0.720 for value in apertures
    ):
        raise ValueError("bootstrap apertures must all be finite and wider than 0.720 m")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_path = args.source_checkpoint.resolve(strict=True)
    teacher_path = args.teacher.resolve(strict=True)
    loaded = load_single_slit_checkpoint(source_path, device=device)
    if loaded.metadata.get("training_stage") != BRIDGE_STAGE:
        raise ValueError("multi-aperture bootstrap requires the audited zero role bridge")
    policy = loaded.policy
    policy.arm_cfg = replace(
        policy.arm_cfg,
        collision_role_precontact_leader_follower=True,
    )
    selected = _configure_trainable(policy, train_scope=args.train_scope)
    selected_count = sum(dict(policy.named_parameters())[name].numel() for name in selected)
    frozen = _frozen_reference(policy, selected)
    parameters = [dict(policy.named_parameters())[name] for name in selected]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1.0e-5)
    teacher = load_executable_teacher(teacher_path, device=device)
    aperture_tensor = torch.tensor(apertures, dtype=torch.float32, device=device)
    output_dir.mkdir(parents=True)
    log_path = output_dir / "train.jsonl"
    initial = _evaluate(policy, loaded.cfg, apertures, horizon=args.evaluation_horizon)
    incumbent_key = tuple(float(value) for value in initial["selection_key"])
    incumbent_state = {
        name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
    }
    print(json.dumps({"step": 0, "evaluation": initial}, allow_nan=False), flush=True)
    for step in range(1, args.updates + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        total, components = _curriculum_loss(
            policy,
            teacher,
            loaded.cfg,
            aperture_tensor,
            chunk_steps=args.chunk_steps,
            clearance_weight=args.clearance_weight,
            clearance_target_m=args.clearance_target_m,
            clearance_temperature_m=args.clearance_temperature_m,
        )
        total.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm).detach().cpu()
        )
        optimizer.step()
        _assert_frozen(policy, frozen)
        record: dict[str, Any] = {
            "step": step,
            "loss": float(total.detach().cpu()),
            "components": components,
            "gradient_norm_before_clip": grad_norm,
        }
        if step % args.report_interval == 0 or step == args.updates:
            policy.eval()
            evaluation = _evaluate(
                policy, loaded.cfg, apertures, horizon=args.evaluation_horizon
            )
            candidate_key = tuple(float(value) for value in evaluation["selection_key"])
            selected_candidate = candidate_key > incumbent_key
            record["evaluation"] = evaluation
            record["selected"] = selected_candidate
            if selected_candidate:
                incumbent_key = candidate_key
                incumbent_state = {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                }
                payload = {
                    "schema_version": SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
                    "step": step,
                    "training_stage": (
                        FULL_CORRECTION_TRAINING_STAGE
                        if args.train_scope == "full-correction"
                        else TRAINING_STAGE
                    ),
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
                    "architecture": architecture_record(policy, radius_edges(loaded.cfg, device=device)),
                    "communication": communication_record(policy, radius_edges(loaded.cfg, device=device), active_ticks=1),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "policy_state_dict": {
                        name: value.detach().cpu()
                        for name, value in policy.state_dict().items()
                    },
                }
                torch.save(payload, output_dir / f"step_{step:06d}_selected.pt")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        if args.stop_on_first_success and incumbent_key[0] >= 1.0:
            break
    policy.load_state_dict(incumbent_state, strict=True)
    final = _evaluate(policy, loaded.cfg, apertures, horizon=args.evaluation_horizon)
    (output_dir / "final.json").write_text(
        json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
