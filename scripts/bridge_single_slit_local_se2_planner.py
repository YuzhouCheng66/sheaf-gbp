#!/usr/bin/env python3
"""Bridge a private local SE(2) planner into Sheaf-GBP with exact action parity."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from piano_movers.env import radius_edges
from piano_movers.models import (
    LOCAL_SE2_PLAN_HIDDEN_DELTA_AUX_KEY,
    LOCAL_SE2_PLAN_MEAN_AUX_KEY,
    LOCAL_SE2_PLAN_PRECISION_AUX_KEY,
    make_policy,
)
from piano_movers.single_slit_0720_bc import (
    SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
    architecture_record,
    communication_record,
    load_executable_teacher,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from scripts.bridge_single_slit_collision_role_factors import (
    HIDDEN_WIDTH,
    WIRE_ROUNDS,
    WIRE_WIDTH,
)
from scripts.bridge_single_slit_separate_role_encoder import _general_parity_audit


BRIDGE_STAGE = "zero-action-private-local-se2-planner-sheaf-v1"
PLANNER_PREFIXES = ("local_se2_", "correction_plan_fusion.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def planner_parameter_names(policy: Any) -> tuple[str, ...]:
    return tuple(
        name
        for name, _ in policy.named_parameters()
        if name.startswith(PLANNER_PREFIXES)
    )


def child_from_parent(parent: Any):
    cfg = parent.policy.arm_cfg
    if parent.arm != "sheaf_gbp" or not cfg.correction_sheaf:
        raise ValueError("local planner bridge requires correction Sheaf-GBP")
    if not cfg.persistent_recovery_state or cfg.obs_dim != 27:
        raise ValueError("local planner bridge requires persistent contact-feedback27")
    if cfg.local_se2_planner:
        raise ValueError("local SE(2) planner is already enabled")
    torch.manual_seed(20260830)
    child_cfg = replace(cfg, local_se2_planner=True)
    child = make_policy("sheaf_gbp", parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    expected = set(planner_parameter_names(child))
    if set(incompatible.missing_keys) != expected:
        raise RuntimeError(
            "local planner tensor boundary changed: "
            f"missing={tuple(incompatible.missing_keys)} expected={tuple(sorted(expected))}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError("local planner bridge has unexpected parent keys")
    if torch.count_nonzero(child.correction_plan_fusion.weight).item() != 0:
        raise RuntimeError("planner-to-correction bridge must start at exact zero")
    for name, value in parent.policy.state_dict().items():
        if not torch.equal(child.state_dict()[name], value):
            raise RuntimeError(f"parent tensor {name!r} changed at planner bridge")
    return child


def _gradient_audit(child: Any, obs: torch.Tensor) -> dict[str, Any]:
    edges = radius_edges(child.env_cfg, device="cpu")
    child.train()
    child.zero_grad(set_to_none=True)
    output = child(obs, edges, child.init_hidden(1, "cpu"))
    plan_mean = output.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY]
    plan_precision = output.aux[LOCAL_SE2_PLAN_PRECISION_AUX_KEY]
    target = torch.linspace(-0.7, 0.7, 8).view(1, 1, 8).expand_as(plan_mean)
    plan_loss = ((plan_mean - target).square() * plan_precision.detach()).mean()
    precision_target = torch.full_like(plan_precision, 2.0)
    precision_loss = F.mse_loss(plan_precision, precision_target)
    action_probe = output.forces.square().mean()
    (plan_loss + 0.1 * precision_loss + action_probe).backward()
    gradients = {}
    for name in planner_parameter_names(child):
        gradient = dict(child.named_parameters())[name].grad
        gradients[name] = 0 if gradient is None else int(torch.count_nonzero(gradient).item())
    required = (
        "local_se2_planner_encoder.0.weight",
        "local_se2_plan_mean_head.weight",
        "local_se2_plan_log_precision_head.weight",
        "correction_plan_fusion.weight",
    )
    if any(gradients.get(name, 0) == 0 for name in required):
        raise RuntimeError("local planner bridge contains a dead first-step gradient path")
    if torch.count_nonzero(output.aux[LOCAL_SE2_PLAN_HIDDEN_DELTA_AUX_KEY]).item() != 0:
        raise RuntimeError("zero planner bridge changed correction hidden state")
    child.zero_grad(set_to_none=True)
    child.eval()
    return {"all_required_first_step_gradients_live": True, "nonzero_entries": gradients}


def bridge_local_se2_planner(
    parent_checkpoint: Path | str,
    teacher_path: Path | str,
    output_checkpoint: Path | str,
) -> dict[str, Any]:
    parent_path = Path(parent_checkpoint).resolve(strict=True)
    teacher_path = Path(teacher_path).resolve(strict=True)
    output_path = Path(output_checkpoint).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = load_single_slit_checkpoint(parent_path, device="cpu")
    parent.policy.eval()
    child = child_from_parent(parent)
    child.eval()
    teacher = load_executable_teacher(teacher_path, device="cpu")
    if child.init_hidden(1, "cpu").shape != (1, 12, HIDDEN_WIDTH):
        raise RuntimeError("private local planner changed carried hidden width")
    if child.communication_floats_per_directed_edge_round() != WIRE_WIDTH:
        raise RuntimeError("private local planner changed wire width")
    if child.communication_rounds() != WIRE_ROUNDS:
        raise RuntimeError("private local planner changed GBP rounds")
    parity = _general_parity_audit(parent, child, teacher)
    first_obs = parity.pop("first_obs")
    gradients = _gradient_audit(child, first_obs)
    names = planner_parameter_names(child)
    parameter_count = sum(dict(child.named_parameters())[name].numel() for name in names)
    old_count = sum(value.numel() for value in parent.policy.parameters())
    new_count = sum(value.numel() for value in child.parameters())
    if new_count - old_count != parameter_count:
        raise RuntimeError("local planner added-parameter accounting changed")
    edges = radius_edges(parent.cfg, device="cpu")
    payload: dict[str, Any] = {
        "schema_version": SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
        "step": 0,
        "training_stage": BRIDGE_STAGE,
        "arm": parent.arm,
        "observation_schema": parent.observation_schema,
        "env_cfg": asdict(parent.cfg),
        "arm_cfg": asdict(child.arm_cfg),
        "parent": {
            "path": str(parent_path),
            "sha256": parent.checkpoint_sha256,
            "training_stage": parent.metadata.get("training_stage"),
        },
        "teacher": {
            "path": str(teacher_path),
            "sha256": _sha256(teacher_path),
            "horizon": int(teacher.horizon),
        },
        "local_planner": {
            "plan_coordinates": [
                "body_delta_x",
                "body_delta_y",
                "delta_theta",
                "body_target_vx",
                "body_target_vy",
                "target_omega",
                "cross_track",
                "path_progress",
            ],
            "private_not_transmitted": True,
            "conditioned_shared_section": "3-D net wrench",
            "parameter_names": list(names),
            "parameter_count": parameter_count,
        },
        "bridge_audit": {
            "execution_device": "cpu",
            "parent_tensors_bitwise_equal": True,
            "source_force_hidden_common_aux_bitwise_equal": True,
            "zero_plan_fusion_exact_action_parity": True,
            "wire_floats_per_directed_edge_round": WIRE_WIDTH,
            "wire_rounds": WIRE_ROUNDS,
            "additional_wire_floats": 0,
            "parity": parity,
            "gradients": gradients,
            "old_parameter_count": old_count,
            "new_parameter_count": new_count,
        },
        "architecture": architecture_record(child, edges),
        "communication": communication_record(child, edges, active_ticks=1),
        "policy_state_dict": child.state_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(payload, handle)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bridge_local_se2_planner(args.parent_checkpoint, args.teacher, args.output)


if __name__ == "__main__":
    main()
