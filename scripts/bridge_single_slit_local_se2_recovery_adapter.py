#!/usr/bin/env python3
"""Add a zero-behaviour post-contact planner adapter to a trained hierarchy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from piano_movers.env import radius_edges
from piano_movers.models import make_policy
from piano_movers.single_slit_0720_bc import (
    SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
    architecture_record,
    communication_record,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import single_slit_scenario


BRIDGE_STAGE = "zero-source-post-contact-local-se2-recovery-adapter-v1"
ADAPTER_PREFIX = "local_se2_recovery_adapter_"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recovery_adapter_parameter_names(policy: Any) -> tuple[str, ...]:
    return tuple(
        name
        for name, _parameter in policy.named_parameters()
        if name.startswith(ADAPTER_PREFIX)
    )


def child_from_parent(parent: Any):
    cfg = parent.policy.arm_cfg
    if parent.arm != "sheaf_gbp" or not cfg.local_se2_planner:
        raise ValueError("post-contact adapter requires a local-planner Sheaf-GBP")
    if cfg.local_se2_recovery_adapter:
        raise ValueError("post-contact planner adapter is already enabled")
    torch.manual_seed(20260831)
    child_cfg = replace(cfg, local_se2_recovery_adapter=True)
    child = make_policy(parent.arm, parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    expected = set(recovery_adapter_parameter_names(child))
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(
            "post-contact adapter tensor boundary changed: "
            f"missing={tuple(incompatible.missing_keys)} "
            f"unexpected={tuple(incompatible.unexpected_keys)}"
        )
    if not expected:
        raise RuntimeError("post-contact adapter added no parameters")
    head = child.local_se2_recovery_adapter_head
    if torch.count_nonzero(head.weight).item() or torch.count_nonzero(head.bias).item():
        raise RuntimeError("post-contact adapter head must start at exact zero")
    return child


@torch.no_grad()
def _never_hit_parity(parent: Any, child: Any) -> dict[str, Any]:
    scenario = single_slit_scenario(1, parent.cfg, visible_aperture_m=0.720)
    context = initial_single_slit_contact_observation_context_v1(
        scenario.state0, parent.cfg
    )
    obs = single_slit_contact_observations_v1(
        scenario.state0, scenario, parent.cfg, context
    )
    edges = radius_edges(parent.cfg, device="cpu")
    parent_output = parent.policy(obs, edges, parent.policy.init_hidden(1, "cpu"))
    child_output = child(obs, edges, child.init_hidden(1, "cpu"))
    checks = {
        "forces": (parent_output.forces, child_output.forces),
        "plan_mean": (
            parent_output.aux["local_se2_plan_mean"],
            child_output.aux["local_se2_plan_mean"],
        ),
        "hidden": (parent_output.hidden, child_output.hidden),
    }
    unequal = [name for name, pair in checks.items() if not torch.equal(*pair)]
    if unequal:
        raise RuntimeError(f"never-hit source parity failed for {unequal}")
    if torch.count_nonzero(
        child_output.aux["local_se2_recovery_plan_delta"]
    ).item():
        raise RuntimeError("inactive post-contact adapter emitted a nonzero delta")
    return {"bitwise_equal": True, "tensors": sorted(checks)}


def bridge_recovery_adapter(
    parent_checkpoint: Path | str,
    output_checkpoint: Path | str,
) -> dict[str, Any]:
    parent_path = Path(parent_checkpoint).resolve(strict=True)
    output_path = Path(output_checkpoint).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = load_single_slit_checkpoint(parent_path, device="cpu")
    parent.policy.eval()
    child = child_from_parent(parent)
    child.eval()
    parity = _never_hit_parity(parent, child)
    names = recovery_adapter_parameter_names(child)
    edges = radius_edges(parent.cfg, device="cpu")
    payload = {
        "schema_version": SINGLE_SLIT_CONTACT_BC_SCHEMA_V1,
        "step": parent.step,
        "training_stage": BRIDGE_STAGE,
        "arm": parent.arm,
        "observation_schema": parent.observation_schema,
        "env_cfg": asdict(parent.cfg),
        "arm_cfg": asdict(child.arm_cfg),
        "parent": {"path": str(parent_path), "sha256": _sha256(parent_path)},
        "bridge_audit": {
            "never_hit_source_parity": parity,
            "parameter_names": list(names),
            "parameter_count": sum(
                dict(child.named_parameters())[name].numel() for name in names
            ),
            "communication_unchanged": (
                parent.policy.communication_floats_per_directed_edge_round()
                == child.communication_floats_per_directed_edge_round()
                and parent.policy.communication_rounds() == child.communication_rounds()
            ),
        },
        "architecture": architecture_record(child, edges),
        "communication": communication_record(child, edges, active_ticks=1),
        "policy_state_dict": {
            name: value.detach().cpu() for name, value in child.state_dict().items()
        },
    }
    if not payload["bridge_audit"]["communication_unchanged"]:
        raise RuntimeError("private recovery adapter changed communication")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "output": str(output_path),
        "parameter_count": payload["bridge_audit"]["parameter_count"],
        "never_hit_source_parity": True,
        "wire_floats": child.communication_floats_per_directed_edge_round(),
        "wire_rounds": child.communication_rounds(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            bridge_recovery_adapter(args.parent_checkpoint, args.output),
            sort_keys=True,
        )
    )
