#!/usr/bin/env python3
"""Enable the parameter-free local-plan -> total-wrench Sheaf controller."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from piano_movers.models import make_policy
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


ANALYTIC_TOTAL_WRENCH_BRIDGE_STAGE = (
    "local-se2-analytic-total-wrench-sheaf-v1"
)


def bridge_analytic_total_wrench(
    parent_checkpoint: Path | str,
    output_checkpoint: Path | str,
) -> dict[str, Any]:
    parent_path = Path(parent_checkpoint).resolve(strict=True)
    output_path = Path(output_checkpoint).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = load_single_slit_checkpoint(parent_path, device="cpu")
    if parent.arm != "sheaf_gbp":
        raise ValueError("analytic total wrench requires sheaf_gbp")
    arm_cfg = parent.policy.arm_cfg
    if not arm_cfg.local_se2_planner or not arm_cfg.correction_sheaf:
        raise ValueError("parent must contain the local planner and correction Sheaf")
    if arm_cfg.local_se2_direct_wrench_unary:
        raise ValueError("use the pre-direct-unary local planner checkpoint")
    if arm_cfg.local_se2_analytic_total_wrench:
        raise ValueError("analytic total-wrench mode is already enabled")

    child_cfg = replace(arm_cfg, local_se2_analytic_total_wrench=True)
    child = make_policy("sheaf_gbp", parent.cfg, child_cfg).to("cpu")
    child.load_state_dict(parent.policy.state_dict(), strict=True)
    child.eval()
    parent_parameters = dict(parent.policy.named_parameters())
    child_parameters = dict(child.named_parameters())
    if parent_parameters.keys() != child_parameters.keys():
        raise RuntimeError("parameter-free bridge changed parameter names")
    for name, parameter in parent_parameters.items():
        if not torch.equal(parameter, child_parameters[name]):
            raise RuntimeError(f"parameter {name!r} changed across analytic bridge")
    if (
        child.communication_floats_per_directed_edge_round()
        != parent.policy.communication_floats_per_directed_edge_round()
        or child.communication_rounds() != parent.policy.communication_rounds()
    ):
        raise RuntimeError("analytic bridge changed the declared wire protocol")

    tree_edges = child.local_se2_analytic_reference_edges[
        child.local_se2_analytic_tree_edge_mask
    ]
    if int(tree_edges.shape[0]) != parent.cfg.n_agents - 1:
        raise RuntimeError("analytic GBP support is not a spanning tree")
    payload = dict(parent.metadata)
    payload.update(
        {
            "step": parent.step,
            "training_stage": ANALYTIC_TOTAL_WRENCH_BRIDGE_STAGE,
            "arm_cfg": asdict(child.arm_cfg),
            "parent": {
                "path": str(parent_path),
                "sha256": parent.checkpoint_sha256,
                "training_stage": parent.metadata.get("training_stage"),
            },
            "analytic_total_wrench": {
                "planner_output": "private one-step payload-frame SE(2)/twist8",
                "local_evidence": "own handle velocity A_i^T c_i",
                "shared_section": "normalized total payload [Fx,Fy,tau]",
                "restriction": "fixed public I3 over a 16-D node stalk",
                "solver": "four-round radius-edge Sheaf-GBP",
                "active_tree_edges": tree_edges.tolist(),
                "teacher_force_at_inference": False,
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
    payload = bridge_analytic_total_wrench(args.parent_checkpoint, args.output)
    print(payload["analytic_total_wrench"])


if __name__ == "__main__":
    main()
