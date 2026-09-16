#!/usr/bin/env python3
"""Expose missing mission fields to the original MLP with exact zero-init parity."""

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
    if parent.policy.arm_cfg.local_se2_markov_goal_context:
        raise ValueError("source already has goal command context")
    arm = replace(parent.policy.arm_cfg, local_se2_markov_goal_context=True)
    child = make_policy("sheaf_gbp", parent.cfg, arm).eval()
    extra = "local_se2_markov_goal_projection.weight"
    loaded = child.load_state_dict(parent.policy.state_dict(), strict=False)
    if loaded.missing_keys != [extra] or loaded.unexpected_keys:
        raise RuntimeError("bridge changed more than the declared input projection")
    if any(not torch.equal(v, child.state_dict()[k]) for k, v in parent.policy.state_dict().items()):
        raise RuntimeError("an original tensor changed")
    if bool(child.state_dict()[extra].count_nonzero()):
        raise RuntimeError("goal command projection must start at exact zero")
    torch.manual_seed(7131)
    with torch.no_grad():
        features = torch.randn(9, 12, parent.policy.local_se2_markov_shape_encoder[0].in_features)
        before = parent.policy.local_se2_markov_shape_encode(features)
        after = child.local_se2_markov_shape_encode(features)
        torch.testing.assert_close(before, after, atol=0., rtol=0.)
    report = {
        "parent_checkpoint": str(args.parent.resolve()), "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "architecture": "original_MLP_with_partitioned_extra_goal_input_columns",
        "original_feature_dim": features.shape[-1], "private_goal_command_dim": 4,
        "additional_parameters": child.state_dict()[extra].numel(),
        "original_tensors_bitwise_unchanged": True, "zero_projection_feature_parity": True,
        "encoder_depth_and_width_unchanged": True, "GBP_and_grasp_decoder_unchanged": True,
        "closed_loop_validation": "not_yet_run", "public_replacement": False,
    }
    payload = dict(parent.metadata)
    payload.update(arm_cfg=asdict(arm), policy_state_dict=child.state_dict(),
                   training_stage="original-MLP-goal-command-zero-bridge",
                   selection_mode="independent_interface_control_not_released",
                   goal_context_bridge=report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(payload, stream)
    report["checkpoint_sha256"] = sha256(args.output)
    write_json(args.output.with_suffix(".json"), report)
    print(report)


if __name__ == "__main__":
    main()
