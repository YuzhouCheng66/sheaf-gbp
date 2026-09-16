#!/usr/bin/env python3
"""Remove the canonical-T learned-route exception without changing weights."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from piano_movers.models import make_policy
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


BRIDGE_STAGE = "route-free-current-geometry-planner-v1"


def bridge_route_free_geometry(
    parent_checkpoint: Path | str,
    output_checkpoint: Path | str,
) -> dict[str, Any]:
    """Enable current-geometry planning for every body, including canonical T."""

    parent_path = Path(parent_checkpoint).resolve(strict=True)
    output_path = Path(output_checkpoint).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = load_single_slit_checkpoint(parent_path, device="cpu")
    parent_cfg = parent.policy.arm_cfg
    if not (
        parent_cfg.local_se2_analytic_total_wrench
        and parent_cfg.local_se2_terminal_attractor
        and parent_cfg.local_se2_feasible_corridor_attractor
    ):
        raise ValueError("route-free bridge requires the physical corridor policy")

    child_cfg = replace(
        parent_cfg,
        local_se2_route_free_geometry_planner=True,
    )
    child = make_policy("sheaf_gbp", parent.cfg, child_cfg).to("cpu")
    child.load_state_dict(parent.policy.state_dict(), strict=True)
    child.eval()
    if tuple(parent.policy.state_dict()) != tuple(child.state_dict()):
        raise RuntimeError("route-free bridge changed state-dict keys")
    for name, value in parent.policy.state_dict().items():
        if not torch.equal(value, child.state_dict()[name]):
            raise RuntimeError(f"route-free bridge changed tensor {name!r}")
    if (
        child.communication_floats_per_directed_edge_round()
        != parent.policy.communication_floats_per_directed_edge_round()
        or child.communication_rounds() != parent.policy.communication_rounds()
    ):
        raise RuntimeError("route-free bridge changed the wire protocol")

    payload = dict(parent.metadata)
    payload.update(
        {
            "training_stage": BRIDGE_STAGE,
            "arm_cfg": asdict(child.arm_cfg),
            "parent": {
                "path": str(parent_path),
                "sha256": parent.checkpoint_sha256,
                "training_stage": parent.metadata.get("training_stage"),
            },
            "route_free_geometry": {
                "canonical_t_learned_route_exception": False,
                "deployment_inputs": (
                    "current radius-local wall observation, current relative "
                    "mission goal, local rigid-handle state, causal contact memory, "
                    "and attached-body rectangle geometry"
                ),
                "letter_id_input": False,
                "stored_route": False,
                "teacher_query": False,
                "carried_path_index": False,
                "additional_parameters": 0,
                "additional_wire_coordinates": 0,
                "additional_communication_rounds": 0,
            },
            "policy_state_dict": child.state_dict(),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(payload, handle)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = bridge_route_free_geometry(args.parent_checkpoint, args.output)
    print(payload["route_free_geometry"])


if __name__ == "__main__":
    main()
