#!/usr/bin/env python3
"""Add the parameter-free, strictly local post-passage SE(2) attractor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from piano_movers.models import make_policy
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


TERMINAL_ATTRACTOR_BRIDGE_STAGE = "local-se2-terminal-attractor-v1"


def bridge_terminal_attractor(
    parent_checkpoint: Path | str,
    output_checkpoint: Path | str,
) -> dict[str, Any]:
    parent_path = Path(parent_checkpoint).resolve(strict=True)
    output_path = Path(output_checkpoint).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = load_single_slit_checkpoint(parent_path, device="cpu")
    cfg = parent.policy.arm_cfg
    if not cfg.local_se2_analytic_total_wrench or not cfg.local_se2_planner:
        raise ValueError("terminal attractor requires hierarchical analytic Sheaf control")
    if cfg.local_se2_terminal_attractor:
        raise ValueError("terminal attractor is already enabled")

    child_cfg = replace(cfg, local_se2_terminal_attractor=True)
    child = make_policy("sheaf_gbp", parent.cfg, child_cfg).to("cpu")
    child.load_state_dict(parent.policy.state_dict(), strict=True)
    child.eval()
    parent_named = dict(parent.policy.named_parameters())
    child_named = dict(child.named_parameters())
    if parent_named.keys() != child_named.keys():
        raise RuntimeError("parameter-free terminal bridge changed parameter names")
    for name, parameter in parent_named.items():
        if not torch.equal(parameter, child_named[name]):
            raise RuntimeError(f"terminal bridge changed parameter {name!r}")
    if (
        child.communication_floats_per_directed_edge_round()
        != parent.policy.communication_floats_per_directed_edge_round()
        or child.communication_rounds() != parent.policy.communication_rounds()
    ):
        raise RuntimeError("terminal bridge changed the wire protocol")

    payload = dict(parent.metadata)
    payload.update(
        {
            "training_stage": TERMINAL_ATTRACTOR_BRIDGE_STAGE,
            "arm_cfg": asdict(child.arm_cfg),
            "parent": {
                "path": str(parent_path),
                "sha256": parent.checkpoint_sha256,
                "training_stage": parent.metadata.get("training_stage"),
            },
            "terminal_attractor": {
                "activation": (
                    "own reconstructed payload leftmost-x strictly beyond "
                    "reserved wall"
                ),
                "inputs": (
                    "own handle world position, own body handle coordinate, "
                    "payload orientation, fixed final goal"
                ),
                "output": "private bounded next SE(2)/twist plan",
                "position_gain": child_cfg.local_se2_terminal_position_gain,
                "angle_gain": child_cfg.local_se2_terminal_angle_gain,
                "max_speed": child_cfg.local_se2_terminal_max_speed,
                "max_omega": child_cfg.local_se2_terminal_max_omega,
                "teacher_information": False,
                "additional_parameters": 0,
                "additional_wire_coordinates": 0,
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
    payload = bridge_terminal_attractor(args.parent_checkpoint, args.output)
    print(payload["terminal_attractor"])


if __name__ == "__main__":
    main()
