#!/usr/bin/env python3
"""Add zero-behaviour leader/follower/rebel factors to persistent Sheaf-GBP.

The three roles are local factor families inside the existing correction
graph.  They do not create three communicated solvers.  This CPU-only bridge
copies every parent tensor bitwise, requires every added tensor to be exact
zero, and publishes only after full-source plus synthetic-contact parity,
gradient, hidden-width, and wire-width audits pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from piano_movers.env import radius_edges
from piano_movers.models import (
    COLLISION_ROLE_LOGITS_AUX_KEY,
    COLLISION_ROLE_LOG_PRECISION_DELTA_AUX_KEY,
    COLLISION_ROLE_MEAN_DELTA_AUX_KEY,
    COLLISION_ROLE_WEIGHTS_AUX_KEY,
    make_policy,
)
from piano_movers.single_slit_0720_bc import (
    SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
    architecture_record,
    communication_record,
    load_executable_teacher,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_observation import (
    SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
)
from piano_movers.single_slit_feedback import (
    SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA,
    single_slit_policy_observations,
    single_slit_scenario,
)


BRIDGE_STAGE = "zero-init-collision-role-factor-sheaf-v1"
ROLE_NAMES = ("leader", "follower", "rebel")
ROLE_PARAMETER_KEYS = (
    "correction_role_logits_head.weight",
    "correction_role_logits_head.bias",
    "correction_role_mean_head.weight",
    "correction_role_mean_head.bias",
    "correction_role_log_precision_head.weight",
    "correction_role_log_precision_head.bias",
)
ROLE_PARAMETER_COUNT = 9_603
HIDDEN_WIDTH = 302
WIRE_WIDTH = 30
WIRE_ROUNDS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _equal(label: str, left: Tensor, right: Tensor) -> None:
    if torch.equal(left, right):
        return
    if left.shape != right.shape:
        raise RuntimeError(f"{label} shape changed: {left.shape} != {right.shape}")
    maximum = float((left - right).abs().amax()) if left.numel() else 0.0
    raise RuntimeError(f"{label} is not bitwise equal; max_abs={maximum:.9g}")


def _obs27(parent: Any, state: Tensor) -> Tensor:
    scenario = single_slit_scenario(1, parent.cfg, initial_state=state)
    obs22 = single_slit_policy_observations(
        state,
        scenario,
        parent.cfg,
        SINGLE_SLIT_FEEDBACK_OBSERVATION_SCHEMA,
    )
    return torch.cat((obs22, obs22.new_zeros((*obs22.shape[:-1], 5))), dim=-1)


def _hit(obs: Tensor) -> Tensor:
    result = obs.clone()
    handle = torch.arange(result.shape[1], dtype=result.dtype).view(1, -1)
    result[..., 22] = 1.0
    result[..., 23] = 0.51
    result[..., 24] = -0.37
    result[..., 25] = -0.12 + 0.02 * handle
    result[..., 26] = 0.09 - 0.015 * handle
    return result


def child_from_parent(parent: Any):
    cfg = parent.policy.arm_cfg
    if parent.arm != "sheaf_gbp":
        raise ValueError("collision-role parent must be Sheaf-GBP")
    if parent.observation_schema != SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1:
        raise ValueError("collision-role parent must use contact-feedback27")
    if not (
        cfg.correction_sheaf
        and cfg.persistent_recovery_state
        and cfg.persistent_recovery_injection == "shared_hidden"
    ):
        raise ValueError("collision roles require the shared-hidden persistent sheaf")
    if cfg.collision_role_factors:
        raise ValueError("collision-role factors are already enabled")
    if parent.policy.init_hidden(1, "cpu").shape != (1, 12, HIDDEN_WIDTH):
        raise ValueError("collision-role parent hidden layout changed")
    child_cfg = replace(cfg, collision_role_factors=True)
    child = make_policy("sheaf_gbp", parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    if tuple(incompatible.missing_keys) != ROLE_PARAMETER_KEYS:
        raise RuntimeError(
            "collision-role tensor boundary changed: "
            f"{tuple(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError("collision-role bridge has unexpected parent keys")
    for name in ROLE_PARAMETER_KEYS:
        if torch.count_nonzero(child.state_dict()[name]).item() != 0:
            raise RuntimeError(f"collision-role bridge tensor {name!r} is not exact zero")
    for name, value in parent.policy.state_dict().items():
        _equal(f"parent tensor {name}", child.state_dict()[name], value)
    return child


@torch.no_grad()
def parity_audit(parent: Any, child: Any, teacher: Any) -> dict[str, Any]:
    edges = radius_edges(parent.cfg, device="cpu")
    parent_hidden = parent.policy.init_hidden(1, "cpu")
    child_hidden = child.init_hidden(1, "cpu")
    common_auxiliary_comparisons = 0
    first_obs: Tensor | None = None
    for tick in range(teacher.horizon):
        obs = _obs27(parent, teacher.states[tick : tick + 1])
        if first_obs is None:
            first_obs = obs
        before = parent.policy(obs, edges, parent_hidden, trace_messages=True)
        after = child(obs, edges, child_hidden, trace_messages=True)
        _equal(f"source force tick {tick}", after.forces, before.forces)
        _equal(f"source hidden tick {tick}", after.hidden, before.hidden)
        for name, value in before.aux.items():
            if isinstance(value, Tensor):
                _equal(f"source aux {name} tick {tick}", after.aux[name], value)
                common_auxiliary_comparisons += 1
        role = after.aux[COLLISION_ROLE_WEIGHTS_AUX_KEY]
        _equal(
            f"source leader/follower role tick {tick}",
            role[..., :2],
            torch.full_like(role[..., :2], 0.5),
        )
        if torch.count_nonzero(role[..., 2]).item() != 0:
            raise RuntimeError("rebel role activated without local contact history")
        parent_hidden = before.hidden
        child_hidden = after.hidden
    if first_obs is None:
        raise RuntimeError("source teacher is empty")

    hit = _hit(first_obs)
    before = parent.policy(
        hit, edges, parent.policy.init_hidden(1, "cpu"), trace_messages=True
    )
    after = child(hit, edges, child.init_hidden(1, "cpu"), trace_messages=True)
    _equal("synthetic-hit force", after.forces, before.forces)
    _equal("synthetic-hit hidden", after.hidden, before.hidden)
    for name, value in before.aux.items():
        if isinstance(value, Tensor):
            _equal(f"synthetic-hit aux {name}", after.aux[name], value)
    _equal(
        "synthetic-hit neutral roles",
        after.aux[COLLISION_ROLE_WEIGHTS_AUX_KEY],
        torch.full_like(after.aux[COLLISION_ROLE_WEIGHTS_AUX_KEY], 0.5),
    )
    if torch.count_nonzero(after.aux[COLLISION_ROLE_MEAN_DELTA_AUX_KEY]).item():
        raise RuntimeError("zero role bridge changed unary mean")
    if torch.count_nonzero(
        after.aux[COLLISION_ROLE_LOG_PRECISION_DELTA_AUX_KEY]
    ).item():
        raise RuntimeError("zero role bridge changed unary precision")
    return {
        "full_source_ticks": int(teacher.horizon),
        "source_forces_hidden_common_aux_bitwise_equal": True,
        "synthetic_contact_forces_hidden_common_aux_bitwise_equal": True,
        "precontact_rebel_exact_zero": True,
        "synthetic_contact_neutral_role_weights": [0.5, 0.5, 0.5],
        "common_auxiliary_tensor_comparisons": common_auxiliary_comparisons,
        "first_obs": first_obs,
    }


def gradient_audit(child: Any, obs: Tensor, edges: Tensor) -> dict[str, Any]:
    child.train()
    child.zero_grad(set_to_none=True)
    output = child(_hit(obs), edges, child.init_hidden(1, "cpu"))
    logits = output.aux[COLLISION_ROLE_LOGITS_AUX_KEY]
    target = torch.zeros_like(logits)
    target[..., 0] = 1.0
    target[..., 1] = 0.7
    target[..., 2] = torch.linspace(0.1, 0.9, logits.shape[1])
    role_loss = F.binary_cross_entropy_with_logits(logits, target)
    factor_probe = (
        output.aux[COLLISION_ROLE_MEAN_DELTA_AUX_KEY].sum()
        + output.aux[COLLISION_ROLE_LOG_PRECISION_DELTA_AUX_KEY].sum()
    )
    (role_loss + factor_probe).backward()
    nonzero: dict[str, int] = {}
    for name in ROLE_PARAMETER_KEYS:
        gradient = dict(child.named_parameters())[name].grad
        if gradient is None:
            raise RuntimeError(f"collision-role tensor {name!r} has no gradient")
        nonzero[name] = int(torch.count_nonzero(gradient).item())
    # At the exact-zero bridge the task-independent factor probe opens the
    # mean/precision heads, while explicit role targets open the logits head.
    required = (
        "correction_role_logits_head.weight",
        "correction_role_mean_head.weight",
        "correction_role_log_precision_head.weight",
    )
    if any(nonzero[name] == 0 for name in required):
        raise RuntimeError("collision-role factor gradient path is not live")
    child.zero_grad(set_to_none=True)
    child.eval()
    return {
        "all_six_added_tensors_have_gradients": True,
        "nonzero_gradient_entries": nonzero,
    }


def bridge_collision_role_factors(
    parent_checkpoint: Path | str,
    teacher_path: Path | str,
    output_checkpoint: Path | str,
) -> dict[str, Any]:
    parent_path = Path(parent_checkpoint).resolve()
    teacher_path = Path(teacher_path).resolve()
    output_path = Path(output_checkpoint).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = load_single_slit_checkpoint(parent_path, device="cpu")
    parent.policy.eval()
    child = child_from_parent(parent)
    child.eval()
    teacher = load_executable_teacher(teacher_path, device="cpu")
    if child.communication_floats_per_directed_edge_round() != WIRE_WIDTH:
        raise RuntimeError("collision-role bridge changed wire width")
    if child.communication_rounds() != WIRE_ROUNDS:
        raise RuntimeError("collision-role bridge changed GBP rounds")
    parity = parity_audit(parent, child, teacher)
    first_obs = parity.pop("first_obs")
    edges = radius_edges(parent.cfg, device="cpu")
    gradients = gradient_audit(child, first_obs, edges)
    old_count = sum(value.numel() for value in parent.policy.parameters())
    new_count = sum(value.numel() for value in child.parameters())
    if new_count - old_count != ROLE_PARAMETER_COUNT:
        raise RuntimeError("collision-role parameter count changed")

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
        "role_factorization": {
            "roles": list(ROLE_NAMES),
            "nonexclusive": True,
            "three_separate_wire_networks": False,
            "information_form_fusion_before_one_gbp": True,
            "rebel_requires_local_contact_history": True,
        },
        "bridge_audit": {
            "execution_device": "cpu",
            "parent_tensors_bitwise_equal": True,
            "added_parameter_names": list(ROLE_PARAMETER_KEYS),
            "added_parameter_count": ROLE_PARAMETER_COUNT,
            "added_tensors_exact_zero": True,
            "old_parameter_count": old_count,
            "new_parameter_count": new_count,
            "hidden_floats_per_agent": HIDDEN_WIDTH,
            "wire_floats_per_directed_edge_round": WIRE_WIDTH,
            "wire_rounds": WIRE_ROUNDS,
            "additional_wire_floats": 0,
            "parity": parity,
            "gradients": gradients,
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
    bridge_collision_role_factors(
        args.parent_checkpoint,
        args.teacher,
        args.output,
    )


if __name__ == "__main__":
    main()
