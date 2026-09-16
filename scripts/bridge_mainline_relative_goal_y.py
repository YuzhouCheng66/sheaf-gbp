#!/usr/bin/env python3
"""Independent SAME-width input-semantic control; no replacement planner.

Column 3 of the existing morphology MLP becomes centre-minus-current-goal-y.
All checkpoint tensors are identical. Unlike a zero residual bridge, this
intentionally changes outputs when goal-y is nonzero and needs retraining and
fresh closed-loop validation. The zero-goal-y case keeps exact old features.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.models import make_policy
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    parent = load_single_slit_checkpoint(args.parent, device="cpu")
    if parent.policy.arm_cfg.local_se2_markov_goal_relative_y:
        raise ValueError("source already uses the relative-y interface")
    arm = replace(parent.policy.arm_cfg, local_se2_markov_goal_relative_y=True)
    child = make_policy("sheaf_gbp", parent.cfg, arm).eval()
    child.load_state_dict(parent.policy.state_dict(), strict=True)
    if any(not torch.equal(v, child.state_dict()[k]) for k, v in parent.policy.state_dict().items()):
        raise RuntimeError("a source tensor changed")
    report = {
        "parent_checkpoint": str(args.parent.resolve()), "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "architecture": "unchanged_original_MLP_with_goal_relative_y_instead_of_absolute_y",
        "private_input_dim": child.local_se2_markov_shape_encoder[0].in_features,
        "changed_column": 3, "additional_parameters": 0,
        "original_tensors_bitwise_unchanged": True,
        "full_behavior_parity_claim": False,
        "encoder_depth_and_width_unchanged": True, "GBP_and_grasp_decoder_unchanged": True,
        "closed_loop_validation": "not_yet_run", "public_replacement": False,
    }
    payload = dict(parent.metadata)
    payload.update(arm_cfg=asdict(arm), policy_state_dict=child.state_dict(),
                   training_stage="original-MLP-goal-relative-y-interface",
                   selection_mode="independent_interface_control_not_released",
                   goal_relative_y_bridge=report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(payload, stream)
    report["checkpoint_sha256"] = sha256(args.output)
    write_json(args.output.with_suffix(".json"), report)
    print(report)


if __name__ == "__main__":
    main()
