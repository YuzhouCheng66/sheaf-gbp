#!/usr/bin/env python3
"""Add a private contact-aware role encoder with exact source-action parity."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from typing import Any

import torch

from piano_movers.env import radius_edges
from piano_movers.models import (
    COLLISION_ROLE_LOG_PRECISION_DELTA_AUX_KEY,
    COLLISION_ROLE_MEAN_DELTA_AUX_KEY,
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
    ROLE_PARAMETER_KEYS,
    WIRE_ROUNDS,
    WIRE_WIDTH,
    _equal,
    _hit,
    _obs27,
)


BRIDGE_STAGE = "copied-separate-contact-role-encoder-sheaf-v2"
ROLE_ENCODER_PARAMETER_KEYS = (
    "correction_role_encoder.0.weight",
    "correction_role_encoder.0.bias",
    "correction_role_encoder.2.weight",
    "correction_role_encoder.2.bias",
)
ROLE_ENCODER_PARAMETER_COUNT = 18_144
ROLE_BRANCH_PARAMETER_KEYS = ROLE_PARAMETER_KEYS + ROLE_ENCODER_PARAMETER_KEYS
ROLE_BRANCH_PARAMETER_COUNT = 27_747


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def child_from_zero_role_parent(parent: Any):
    cfg = parent.policy.arm_cfg
    if not cfg.collision_role_factors or cfg.collision_role_separate_encoder:
        raise ValueError("parent must have shared-encoder collision role factors")
    torch.manual_seed(20260830)
    child_cfg = replace(cfg, collision_role_separate_encoder=True)
    child = make_policy("sheaf_gbp", parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    if tuple(incompatible.missing_keys) != ROLE_ENCODER_PARAMETER_KEYS:
        raise RuntimeError(
            "separate role-encoder tensor boundary changed: "
            f"{tuple(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError("separate role-encoder bridge has unexpected parent keys")
    # The copied encoder makes role_h exactly equal to the historical shared
    # correction_h, including after the same persistent-memory fusion. This
    # preserves already-trained role behaviour while opening a private route
    # for future contact-channel learning.
    child.correction_role_encoder.load_state_dict(
        parent.policy.correction_local_encoder.state_dict(),
        strict=True,
    )
    for name, value in parent.policy.state_dict().items():
        _equal(f"parent tensor {name}", child.state_dict()[name], value)
    return child


@torch.no_grad()
def _general_parity_audit(parent: Any, child: Any, teacher: Any) -> dict[str, Any]:
    edges = radius_edges(parent.cfg, device="cpu")
    parent_hidden = parent.policy.init_hidden(1, "cpu")
    child_hidden = child.init_hidden(1, "cpu")
    first_obs = None
    comparisons = 0
    for tick in range(teacher.horizon):
        obs = _obs27(parent, teacher.states[tick : tick + 1])
        first_obs = obs if first_obs is None else first_obs
        before = parent.policy(obs, edges, parent_hidden, trace_messages=True)
        after = child(obs, edges, child_hidden, trace_messages=True)
        _equal(f"source force tick {tick}", after.forces, before.forces)
        _equal(f"source hidden tick {tick}", after.hidden, before.hidden)
        for name, value in before.aux.items():
            if isinstance(value, torch.Tensor):
                _equal(f"source aux {name} tick {tick}", after.aux[name], value)
                comparisons += 1
        parent_hidden = before.hidden
        child_hidden = after.hidden
    if first_obs is None:
        raise RuntimeError("source teacher is empty")
    hit = _hit(first_obs)
    before = parent.policy(hit, edges, parent.policy.init_hidden(1, "cpu"), trace_messages=True)
    after = child(hit, edges, child.init_hidden(1, "cpu"), trace_messages=True)
    _equal("synthetic-hit force", after.forces, before.forces)
    _equal("synthetic-hit hidden", after.hidden, before.hidden)
    for name, value in before.aux.items():
        if isinstance(value, torch.Tensor):
            _equal(f"synthetic-hit aux {name}", after.aux[name], value)
            comparisons += 1
    return {
        "full_source_ticks": int(teacher.horizon),
        "source_and_synthetic_contact_bitwise_equal": True,
        "common_auxiliary_tensor_comparisons": comparisons,
        "first_obs": first_obs,
    }


def _two_step_encoder_gradient_audit(child: Any, obs: torch.Tensor) -> dict[str, Any]:
    """Prove the zero-head bridge opens the role encoder after one head step."""

    probe = copy.deepcopy(child).train()
    edges = radius_edges(probe.env_cfg, device="cpu")
    hit = obs.clone()
    hit[..., 22] = 1.0
    hit[..., 23:27] = torch.tensor((0.51, -0.37, -0.12, 0.09))
    output = probe(hit, edges, probe.init_hidden(1, "cpu"))
    first_loss = (
        output.aux[COLLISION_ROLE_MEAN_DELTA_AUX_KEY].sum()
        + output.aux[COLLISION_ROLE_LOG_PRECISION_DELTA_AUX_KEY].sum()
    )
    first_loss.backward()
    with torch.no_grad():
        for name in ROLE_PARAMETER_KEYS:
            parameter = dict(probe.named_parameters())[name]
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1.0e-3)
    probe.zero_grad(set_to_none=True)
    output = probe(hit, edges, probe.init_hidden(1, "cpu"))
    second_loss = (
        output.aux[COLLISION_ROLE_MEAN_DELTA_AUX_KEY].square().sum()
        + output.aux[COLLISION_ROLE_LOG_PRECISION_DELTA_AUX_KEY].square().sum()
    )
    second_loss.backward()
    nonzero = {}
    for name in ROLE_ENCODER_PARAMETER_KEYS:
        gradient = dict(probe.named_parameters())[name].grad
        if gradient is None:
            raise RuntimeError(f"role encoder tensor {name!r} has no second-step gradient")
        nonzero[name] = int(torch.count_nonzero(gradient).item())
    if not any(value > 0 for value in nonzero.values()):
        raise RuntimeError("separate role encoder gradient path did not open")
    return {
        "zero_heads_open_encoder_after_one_head_update": True,
        "nonzero_second_step_gradient_entries": nonzero,
    }


def bridge_separate_role_encoder(
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
    child = child_from_zero_role_parent(parent)
    child.eval()
    teacher = load_executable_teacher(teacher_path, device="cpu")
    if child.init_hidden(1, "cpu").shape != (1, 12, HIDDEN_WIDTH):
        raise RuntimeError("separate role encoder changed hidden width")
    if child.communication_floats_per_directed_edge_round() != WIRE_WIDTH:
        raise RuntimeError("separate role encoder changed wire width")
    if child.communication_rounds() != WIRE_ROUNDS:
        raise RuntimeError("separate role encoder changed GBP rounds")
    parity = _general_parity_audit(parent, child, teacher)
    first_obs = parity.pop("first_obs")
    gradients = _two_step_encoder_gradient_audit(child, first_obs)
    old_count = sum(value.numel() for value in parent.policy.parameters())
    new_count = sum(value.numel() for value in child.parameters())
    if new_count - old_count != ROLE_ENCODER_PARAMETER_COUNT:
        raise RuntimeError("separate role encoder parameter count changed")
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
        "role_encoder": {
            "input": "local contact-feedback27 + frozen source unary mean/log-precision",
            "private_local_only": True,
            "contact_gated_at_factor_output": True,
            "source_path_changed": False,
            "initialization": "bitwise copy of correction_local_encoder",
            "parameter_names": list(ROLE_ENCODER_PARAMETER_KEYS),
            "parameter_count": ROLE_ENCODER_PARAMETER_COUNT,
            "full_role_branch_parameter_count": ROLE_BRANCH_PARAMETER_COUNT,
        },
        "bridge_audit": {
            "execution_device": "cpu",
            "parent_tensors_bitwise_equal": True,
            "source_forces_hidden_common_aux_bitwise_equal": True,
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
    bridge_separate_role_encoder(args.parent_checkpoint, args.teacher, args.output)


if __name__ == "__main__":
    main()
