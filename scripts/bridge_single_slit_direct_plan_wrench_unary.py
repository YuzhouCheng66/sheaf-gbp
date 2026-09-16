#!/usr/bin/env python3
"""Add a zero-action private-plan to public-wrench unary adapter."""

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
    LOCAL_SE2_DIRECT_WRENCH_DELTA_AUX_KEY,
    make_policy,
)
from piano_movers.single_slit_0720_bc import (
    SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
    architecture_record,
    communication_record,
    load_executable_teacher,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from scripts.bridge_single_slit_collision_role_factors import WIRE_ROUNDS, WIRE_WIDTH
from scripts.bridge_single_slit_separate_role_encoder import _general_parity_audit


BRIDGE_STAGE = "zero-direct-plan-wrench-unary-sheaf-v1"
PARAMETER_NAMES = (
    "local_se2_direct_wrench_unary_head.0.weight",
    "local_se2_direct_wrench_unary_head.0.bias",
    "local_se2_direct_wrench_unary_head.2.weight",
    "local_se2_direct_wrench_unary_head.2.bias",
    "local_se2_direct_wrench_unary_head.4.weight",
    "local_se2_direct_wrench_unary_head.4.bias",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bridge(
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
    cfg = parent.policy.arm_cfg
    if not cfg.local_se2_planner or cfg.local_se2_direct_wrench_unary:
        raise ValueError("parent must have local planning without the direct wrench unary")
    child_cfg = replace(cfg, local_se2_direct_wrench_unary=True)
    child = make_policy("sheaf_gbp", parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    if tuple(incompatible.missing_keys) != PARAMETER_NAMES or incompatible.unexpected_keys:
        raise RuntimeError(
            f"direct-unary tensor boundary changed: {incompatible}"
        )
    for name, value in parent.policy.state_dict().items():
        if not torch.equal(value, child.state_dict()[name]):
            raise RuntimeError(f"parent tensor {name!r} changed")
    final_names = PARAMETER_NAMES[-2:]
    if any(torch.count_nonzero(dict(child.named_parameters())[name]) for name in final_names):
        raise RuntimeError("direct wrench unary final layer is not zero initialized")
    teacher = load_executable_teacher(teacher_path, device="cpu")
    child.eval()
    parity = _general_parity_audit(parent, child, teacher)
    first_obs = parity.pop("first_obs")
    edges = radius_edges(parent.cfg, device="cpu")
    child.train()
    child.zero_grad(set_to_none=True)
    gradient_obs = first_obs.clone()
    gradient_obs[..., 16] = 1.0
    gradient_obs[..., 18] = 0.0
    output = child(gradient_obs, edges, child.init_hidden(1, "cpu"))
    delta = output.aux[LOCAL_SE2_DIRECT_WRENCH_DELTA_AUX_KEY]
    F.mse_loss(delta, torch.full_like(delta, 0.125)).backward()
    gradients = {}
    for name in PARAMETER_NAMES:
        gradient = dict(child.named_parameters())[name].grad
        gradients[name] = 0 if gradient is None else int(torch.count_nonzero(gradient).item())
    if any(gradients[name] == 0 for name in final_names):
        raise RuntimeError("direct wrench unary final layer has a dead first-step gradient")
    # The exact-zero final layer intentionally delays trunk gradients by one
    # optimizer step.  Audit that the trunk becomes live as soon as that
    # layer leaves zero, then restore the exact bridge endpoint.
    child.zero_grad(set_to_none=True)
    final_layer = child.local_se2_direct_wrench_unary_head[-1]
    with torch.no_grad():
        final_layer.weight.fill_(1.0e-3)
    unlocked = child(gradient_obs, edges, child.init_hidden(1, "cpu"))
    F.mse_loss(
        unlocked.aux[LOCAL_SE2_DIRECT_WRENCH_DELTA_AUX_KEY],
        torch.full_like(delta, 0.125),
    ).backward()
    unlocked_gradients = {}
    for name in PARAMETER_NAMES:
        gradient = dict(child.named_parameters())[name].grad
        unlocked_gradients[name] = (
            0 if gradient is None else int(torch.count_nonzero(gradient).item())
        )
    if any(value == 0 for value in unlocked_gradients.values()):
        raise RuntimeError("direct wrench unary trunk did not unlock after the final layer")
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.zero_()
    child.zero_grad(set_to_none=True)
    child.eval()
    parameter_count = sum(dict(child.named_parameters())[name].numel() for name in PARAMETER_NAMES)
    if parameter_count != 31_363:
        raise RuntimeError(
            f"direct unary has {parameter_count} parameters, expected 31363"
        )
    if child.communication_floats_per_directed_edge_round() != WIRE_WIDTH:
        raise RuntimeError("direct unary changed wire width")
    if child.communication_rounds() != WIRE_ROUNDS:
        raise RuntimeError("direct unary changed GBP rounds")
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
        "direct_plan_wrench_unary": {
            "input": "private correction_h96 + private plan/precision16",
            "output": "public normalized [Fx,Fy,tau] unary delta3",
            "parameter_names": list(PARAMETER_NAMES),
            "parameter_count": parameter_count,
            "additional_wire_floats": 0,
        },
        "bridge_audit": {
            "old_tensors_bitwise_equal": True,
            "full_teacher_action_hidden_common_aux_bitwise_equal": True,
            "zero_action_parity": True,
            "first_step_gradients": gradients,
            "post_unlock_gradients": unlocked_gradients,
            "parity": parity,
        },
        "architecture": architecture_record(child, edges),
        "communication": communication_record(child, edges, active_ticks=1),
        "policy_state_dict": child.state_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(payload, handle)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bridge(args.parent_checkpoint, args.teacher, args.output)


if __name__ == "__main__":
    main()
