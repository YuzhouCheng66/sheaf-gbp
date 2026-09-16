#!/usr/bin/env python3
"""Append zero-initialized geometry-canonical inputs to the recovery planner."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import torch

from piano_movers.models import (
    LOCAL_SE2_RECOVERY_CANONICAL_DIM,
    make_policy,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from scripts.bridge_single_slit_analytic_phase_latch import _precontact_parity


BRIDGE_STAGE = "canonical-local-se2-recovery-encoder-v1"
FIRST_WEIGHT = "local_se2_recovery_adapter_encoder.0.weight"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def bridge(source_path: Path, output_path: Path) -> None:
    source_path = source_path.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = load_single_slit_checkpoint(source_path, device="cpu")
    if not parent.policy.arm_cfg.local_se2_recovery_adapter:
        raise ValueError("canonical bridge requires a recovery adapter")
    if parent.policy.arm_cfg.local_se2_recovery_canonical_features:
        raise ValueError("source already contains canonical recovery features")
    child_cfg = replace(
        parent.policy.arm_cfg,
        local_se2_recovery_canonical_features=True,
    )
    child = make_policy(parent.arm, parent.cfg, child_cfg).to("cpu")
    source_state = parent.policy.state_dict()
    bridged_state = child.state_dict()
    for name, child_value in bridged_state.items():
        source_value = source_state[name]
        if source_value.shape == child_value.shape:
            child_value.copy_(source_value)
            continue
        if name != FIRST_WEIGHT:
            raise RuntimeError(
                f"canonical bridge found an unexpected changed tensor: {name}"
            )
        if child_value.shape[0] != source_value.shape[0] or (
            child_value.shape[1]
            != source_value.shape[1] + LOCAL_SE2_RECOVERY_CANONICAL_DIM
        ):
            raise RuntimeError("canonical adapter input width changed unexpectedly")
        child_value.zero_()
        child_value[:, : source_value.shape[1]].copy_(source_value)
    child.load_state_dict(bridged_state, strict=True)
    parent.policy.eval()
    child.eval()
    parity = _precontact_parity(parent, child)
    if (
        child.communication_floats_per_directed_edge_round()
        != parent.policy.communication_floats_per_directed_edge_round()
        or child.communication_rounds() != parent.policy.communication_rounds()
    ):
        raise RuntimeError("canonical recovery inputs changed communication")

    payload = copy.deepcopy(dict(parent.metadata))
    payload.update(
        {
            "training_stage": BRIDGE_STAGE,
            "arm_cfg": asdict(child_cfg),
            "parent": {"path": str(source_path), "sha256": _sha256(source_path)},
            "canonical_recovery_encoder": {
                "feature_dim": LOCAL_SE2_RECOVERY_CANONICAL_DIM,
                "source_null_before_real_contact": True,
                "new_input_columns_zero_initialized": True,
                "additional_wire_floats": 0,
                "parity": parity,
            },
            "policy_state_dict": {
                name: value.detach().cpu()
                for name, value in child.state_dict().items()
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    reloaded = load_single_slit_checkpoint(output_path, device="cpu")
    if reloaded.policy.arm_cfg != child_cfg:
        raise RuntimeError("fresh canonical checkpoint changed configuration")
    print(
        {
            "output": str(output_path),
            "precontact_parity": parity,
            "canonical_dim": LOCAL_SE2_RECOVERY_CANONICAL_DIM,
            "wire_floats": child.communication_floats_per_directed_edge_round(),
            "wire_rounds": child.communication_rounds(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bridge(args.source, args.output)


if __name__ == "__main__":
    main()
