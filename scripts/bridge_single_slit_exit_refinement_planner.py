#!/usr/bin/env python3
"""Append a zero-behaviour, later-supported exit refinement planner."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from piano_movers.env import radius_edges
from piano_movers.models import make_policy
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import single_slit_scenario


BRIDGE_STAGE = "zero-source-exit-refinement-planner-v1"
PARAMETER_PREFIX = "local_se2_exit_refinement_"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exit_refinement_parameter_names(policy: Any) -> tuple[str, ...]:
    return tuple(
        name
        for name, _parameter in policy.named_parameters()
        if name.startswith(PARAMETER_PREFIX)
    )


@torch.no_grad()
def _parity(parent: Any, child: Any, min_center_x_m: float) -> dict[str, Any]:
    edges = radius_edges(parent.cfg, device="cpu")
    checks: list[str] = []
    for label, centre_x, hit in (
        ("approach", -1.05, False),
        ("below_support", min_center_x_m - 0.02, True),
        ("inside_support", min_center_x_m + 0.02, True),
    ):
        state = torch.tensor([[centre_x, -0.24, 1.20, 0.0, 0.0, 0.0]])
        scenario = single_slit_scenario(
            1, parent.cfg, initial_state=state, visible_aperture_m=0.720
        )
        context = initial_single_slit_contact_observation_context_v1(
            state, parent.cfg
        )
        observation = single_slit_contact_observations_v1(
            state, scenario, parent.cfg, context
        )
        if hit:
            observation[..., 22] = 1.0
            observation[..., 23:25] = observation.new_tensor((0.2, -0.15))
            observation[..., 25:27] = observation.new_tensor((-0.05, 0.08))
        parent_output = parent.policy(
            observation, edges, parent.policy.init_hidden(1, "cpu")
        )
        child_output = child(
            observation, edges, child.init_hidden(1, "cpu")
        )
        pairs = {
            "forces": (parent_output.forces, child_output.forces),
            "hidden": (parent_output.hidden, child_output.hidden),
            "local_se2_plan_mean": (
                parent_output.aux["local_se2_plan_mean"],
                child_output.aux["local_se2_plan_mean"],
            ),
        }
        unequal = [name for name, pair in pairs.items() if not torch.equal(*pair)]
        if unequal:
            raise RuntimeError(f"{label} exit-refinement parity failed for {unequal}")
        checks.append(label)
    return {"bitwise_equal": True, "conditions": checks}


@torch.no_grad()
def bridge(
    source: Path,
    output: Path,
    *,
    min_center_x_m: float,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = load_single_slit_checkpoint(source, device="cpu")
    cfg = parent.policy.arm_cfg
    if not (
        parent.arm == "sheaf_gbp"
        and cfg.local_se2_contact_rebel_adapter
        and cfg.local_se2_recovery_canonical_features
    ):
        raise ValueError("exit refinement requires a trained canonical rebel policy")
    if cfg.local_se2_exit_refinement_adapter:
        raise ValueError("source already contains exit refinement")
    torch.manual_seed(20260831)
    child_cfg = replace(
        cfg,
        local_se2_exit_refinement_adapter=True,
        local_se2_exit_refinement_min_center_x_m=min_center_x_m,
    )
    child = make_policy(parent.arm, parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    names = exit_refinement_parameter_names(child)
    if set(incompatible.missing_keys) != set(names) or incompatible.unexpected_keys:
        raise RuntimeError(
            "exit refinement tensor boundary changed: "
            f"missing={tuple(incompatible.missing_keys)} "
            f"unexpected={tuple(incompatible.unexpected_keys)}"
        )
    head = child.local_se2_exit_refinement_head
    if torch.count_nonzero(head.weight).item() or torch.count_nonzero(head.bias).item():
        raise RuntimeError("exit refinement head must start at exact zero")
    parent.policy.eval()
    child.eval()
    parity = _parity(parent, child, min_center_x_m)
    communication_unchanged = (
        parent.policy.communication_floats_per_directed_edge_round()
        == child.communication_floats_per_directed_edge_round()
        and parent.policy.communication_rounds() == child.communication_rounds()
    )
    if not communication_unchanged:
        raise RuntimeError("private exit refinement changed communication")
    payload = copy.deepcopy(dict(parent.metadata))
    payload.update(
        {
            "training_stage": BRIDGE_STAGE,
            "arm_cfg": asdict(child_cfg),
            "parent": {"path": str(source), "sha256": _sha256(source)},
            "exit_refinement_bridge": {
                "parameter_names": list(names),
                "parameter_count": sum(
                    dict(child.named_parameters())[name].numel() for name in names
                ),
                "minimum_payload_center_x_m": min_center_x_m,
                "parity": parity,
                "additional_wire_floats": 0,
            },
            "policy_state_dict": {
                name: value.detach().cpu()
                for name, value in child.state_dict().items()
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    reloaded = load_single_slit_checkpoint(output, device="cpu")
    if reloaded.policy.arm_cfg != child_cfg:
        raise RuntimeError("fresh exit-refinement checkpoint changed configuration")
    return {
        "output": str(output),
        "parameter_count": payload["exit_refinement_bridge"]["parameter_count"],
        "parity": parity,
        "wire_floats": child.communication_floats_per_directed_edge_round(),
        "wire_rounds": child.communication_rounds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-center-x-m", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            bridge(
                args.source,
                args.output,
                min_center_x_m=args.min_center_x_m,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
