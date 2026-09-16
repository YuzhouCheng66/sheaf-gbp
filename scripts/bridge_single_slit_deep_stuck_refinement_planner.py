#!/usr/bin/env python3
"""Append a zero-behaviour refinement gated beyond protected contact streaks."""

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


BRIDGE_STAGE = "zero-source-deep-stuck-refinement-planner-v1"
PARAMETER_PREFIX = "local_se2_deep_stuck_refinement_"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deep_stuck_refinement_parameter_names(policy: Any) -> tuple[str, ...]:
    return tuple(
        name
        for name, _parameter in policy.named_parameters()
        if name.startswith(PARAMETER_PREFIX)
    )


def _hit_observation(parent: Any, centre_x: float) -> torch.Tensor:
    state = torch.tensor([[centre_x, 0.11, 0.95, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, parent.cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, parent.cfg)
    observation = single_slit_contact_observations_v1(
        state, scenario, parent.cfg, context
    )
    observation[..., 22] = 1.0
    observation[..., 23:25] = observation.new_tensor((0.2, -0.15))
    observation[..., 25:27] = observation.new_tensor((-0.05, 0.08))
    return observation


@torch.no_grad()
def _parity(parent: Any, child: Any, stuck_steps: int) -> dict[str, Any]:
    edges = radius_edges(parent.cfg, device="cpu")
    parent_hidden = parent.policy.init_hidden(1, "cpu")
    child_hidden = child.init_hidden(1, "cpu")
    observation = _hit_observation(parent, -0.176)
    checks: list[str] = []
    for tick in range(1, stuck_steps + 2):
        parent_output = parent.policy(observation, edges, parent_hidden)
        child_output = child(observation, edges, child_hidden)
        pairs = {
            "forces": (parent_output.forces, child_output.forces),
            "hidden_prefix": (
                parent_output.hidden[..., :-1],
                child_output.hidden[..., :-1],
            ),
            "local_se2_plan_mean": (
                parent_output.aux["local_se2_plan_mean"],
                child_output.aux["local_se2_plan_mean"],
            ),
        }
        unequal = [name for name, pair in pairs.items() if not torch.equal(*pair)]
        if unequal:
            raise RuntimeError(f"deep-stuck tick {tick} parity failed: {unequal}")
        expected_parent = float(
            min(tick, parent.policy.arm_cfg.local_se2_approach_refinement_stuck_steps)
        )
        expected_child = float(min(tick, stuck_steps))
        if not torch.equal(
            parent_output.hidden[..., -1:],
            parent_output.hidden.new_full(
                parent_output.hidden[..., -1:].shape, expected_parent
            ),
        ) or not torch.equal(
            child_output.hidden[..., -1:],
            child_output.hidden.new_full(
                child_output.hidden[..., -1:].shape, expected_child
            ),
        ):
            raise RuntimeError(f"deep-stuck tick {tick} counter audit failed")
        parent_hidden = parent_output.hidden
        child_hidden = child_output.hidden
        if tick in {stuck_steps - 1, stuck_steps, stuck_steps + 1}:
            checks.append(f"consecutive_hit_{tick}")
    return {
        "forces_and_plan_bitwise_equal": True,
        "legacy_hidden_prefix_bitwise_equal": True,
        "counter_extension": "parent clips at 9; child clips at 10",
        "conditions": checks,
    }


@torch.no_grad()
def bridge(
    source: Path,
    output: Path,
    *,
    min_center_x_m: float,
    stuck_steps: int,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = load_single_slit_checkpoint(source, device="cpu")
    cfg = parent.policy.arm_cfg
    if not (
        parent.arm == "sheaf_gbp"
        and cfg.local_se2_approach_refinement_adapter
        and cfg.local_se2_recovery_canonical_features
    ):
        raise ValueError("deep-stuck refinement requires a trained approach policy")
    if cfg.local_se2_deep_stuck_refinement_adapter:
        raise ValueError("source already contains deep-stuck refinement")
    if stuck_steps <= cfg.local_se2_approach_refinement_stuck_steps:
        raise ValueError("deep-stuck threshold must exceed the approach threshold")
    torch.manual_seed(20260831)
    child_cfg = replace(
        cfg,
        local_se2_deep_stuck_refinement_adapter=True,
        local_se2_deep_stuck_refinement_min_center_x_m=min_center_x_m,
        local_se2_deep_stuck_refinement_stuck_steps=stuck_steps,
    )
    child = make_policy(parent.arm, parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    names = deep_stuck_refinement_parameter_names(child)
    if set(incompatible.missing_keys) != set(names) or incompatible.unexpected_keys:
        raise RuntimeError(
            "deep-stuck tensor boundary changed: "
            f"missing={tuple(incompatible.missing_keys)} "
            f"unexpected={tuple(incompatible.unexpected_keys)}"
        )
    head = child.local_se2_deep_stuck_refinement_head
    if torch.count_nonzero(head.weight).item() or torch.count_nonzero(head.bias).item():
        raise RuntimeError("deep-stuck head must start at exact zero")
    parent.policy.eval()
    child.eval()
    parity = _parity(parent, child, stuck_steps)
    if (
        parent.policy.communication_floats_per_directed_edge_round()
        != child.communication_floats_per_directed_edge_round()
        or parent.policy.communication_rounds() != child.communication_rounds()
    ):
        raise RuntimeError("private deep-stuck refinement changed communication")
    payload = copy.deepcopy(dict(parent.metadata))
    payload.update(
        {
            "training_stage": BRIDGE_STAGE,
            "arm_cfg": asdict(child_cfg),
            "parent": {"path": str(source), "sha256": _sha256(source)},
            "deep_stuck_refinement_bridge": {
                "parameter_names": list(names),
                "parameter_count": sum(
                    dict(child.named_parameters())[name].numel() for name in names
                ),
                "minimum_payload_center_x_m": min_center_x_m,
                "minimum_consecutive_contact_steps": stuck_steps,
                "parity": parity,
                "additional_wire_floats": 0,
            },
            "policy_state_dict": {
                name: value.detach().cpu() for name, value in child.state_dict().items()
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    reloaded = load_single_slit_checkpoint(output, device="cpu")
    if reloaded.policy.arm_cfg != child_cfg:
        raise RuntimeError("fresh deep-stuck checkpoint changed configuration")
    return {
        "output": str(output),
        "parameter_count": payload["deep_stuck_refinement_bridge"]["parameter_count"],
        "parity": parity,
        "wire_floats": child.communication_floats_per_directed_edge_round(),
        "wire_rounds": child.communication_rounds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-center-x-m", type=float, required=True)
    parser.add_argument("--stuck-steps", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            bridge(
                args.source,
                args.output,
                min_center_x_m=args.min_center_x_m,
                stuck_steps=args.stuck_steps,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
