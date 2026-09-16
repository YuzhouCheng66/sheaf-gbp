#!/usr/bin/env python3
"""Add a zero-behaviour, contact-triggered exit-side planning residual."""

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


BRIDGE_STAGE = "zero-source-contact-triggered-exit-rebel-planner-v2"
PARAMETER_PREFIX = "local_se2_contact_rebel_"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contact_rebel_parameter_names(policy: Any) -> tuple[str, ...]:
    return tuple(
        name
        for name, _parameter in policy.named_parameters()
        if name.startswith(PARAMETER_PREFIX)
    )


@torch.no_grad()
def _parity(parent: Any, child: Any) -> dict[str, Any]:
    scenario = single_slit_scenario(1, parent.cfg, visible_aperture_m=0.720)
    context = initial_single_slit_contact_observation_context_v1(
        scenario.state0, parent.cfg
    )
    safe_obs = single_slit_contact_observations_v1(
        scenario.state0, scenario, parent.cfg, context
    )
    exit_state = scenario.state0.clone()
    exit_state[:, 0] = 0.15
    exit_scenario = single_slit_scenario(
        1, parent.cfg, initial_state=exit_state, visible_aperture_m=0.720
    )
    exit_context = initial_single_slit_contact_observation_context_v1(
        exit_state, parent.cfg
    )
    hit_obs = single_slit_contact_observations_v1(
        exit_state, exit_scenario, parent.cfg, exit_context
    )
    hit_obs[..., 22] = 1.0
    hit_obs[..., 23:25] = safe_obs.new_tensor((0.2, -0.15))
    hit_obs[..., 25:27] = safe_obs.new_tensor((-0.05, 0.08))
    edges = radius_edges(parent.cfg, device="cpu")
    tensors = ("forces", "hidden", "local_se2_plan_mean")
    for label, observation in (
        ("never_hit", safe_obs),
        ("current_hit_exit_side", hit_obs),
    ):
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
            raise RuntimeError(f"{label} bridge parity failed for {unequal}")
    return {
        "bitwise_equal": True,
        "conditions": ["never_hit", "current_hit_exit_side"],
        "tensors": list(tensors),
    }


@torch.no_grad()
def bridge(
    source: Path,
    output: Path,
    *,
    min_center_x_m: float = 0.0,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = load_single_slit_checkpoint(source, device="cpu")
    cfg = parent.policy.arm_cfg
    if not (
        parent.arm == "sheaf_gbp"
        and cfg.local_se2_recovery_adapter
        and cfg.local_se2_recovery_canonical_features
    ):
        raise ValueError("contact rebel bridge requires canonical recovery Sheaf-GBP")
    if cfg.local_se2_contact_rebel_adapter:
        raise ValueError("source already contains a contact rebel adapter")
    torch.manual_seed(20260831)
    child_cfg = replace(
        cfg,
        local_se2_contact_rebel_adapter=True,
        local_se2_contact_rebel_min_center_x_m=min_center_x_m,
    )
    child = make_policy(parent.arm, parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    names = contact_rebel_parameter_names(child)
    if set(incompatible.missing_keys) != set(names) or incompatible.unexpected_keys:
        raise RuntimeError(
            "contact rebel tensor boundary changed: "
            f"missing={tuple(incompatible.missing_keys)} "
            f"unexpected={tuple(incompatible.unexpected_keys)}"
        )
    head = child.local_se2_contact_rebel_head
    if torch.count_nonzero(head.weight).item() or torch.count_nonzero(head.bias).item():
        raise RuntimeError("contact rebel head must start at exact zero")
    parent.policy.eval()
    child.eval()
    parity = _parity(parent, child)
    communication_unchanged = (
        parent.policy.communication_floats_per_directed_edge_round()
        == child.communication_floats_per_directed_edge_round()
        and parent.policy.communication_rounds() == child.communication_rounds()
    )
    if not communication_unchanged:
        raise RuntimeError("private contact rebel branch changed communication")
    payload = copy.deepcopy(dict(parent.metadata))
    payload.update(
        {
            "training_stage": BRIDGE_STAGE,
            "arm_cfg": asdict(child_cfg),
            "parent": {"path": str(source), "sha256": _sha256(source)},
            "contact_rebel_bridge": {
                "parameter_names": list(names),
                "parameter_count": sum(
                    dict(child.named_parameters())[name].numel() for name in names
                ),
                "causal_support": (
                    "latched rejected-contact history and locally reconstructed "
                    "payload centre on the exit side"
                ),
                "parity": parity,
                "additional_wire_floats": 0,
                "minimum_payload_center_x_m": min_center_x_m,
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
        raise RuntimeError("fresh contact rebel checkpoint changed configuration")
    return {
        "output": str(output),
        "parameter_count": payload["contact_rebel_bridge"]["parameter_count"],
        "parity": parity,
        "wire_floats": child.communication_floats_per_directed_edge_round(),
        "wire_rounds": child.communication_rounds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-center-x-m", type=float, default=0.0)
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
