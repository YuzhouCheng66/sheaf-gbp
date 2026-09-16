#!/usr/bin/env python3
"""Append one zero-behaviour retry stage beyond all protected successes."""

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


BRIDGE_STAGE = "zero-source-staged-retry-refinement-planner-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def retry_refinement_parameter_names(
    policy: Any, *, stage_index: int | None = None
) -> tuple[str, ...]:
    if stage_index is None:
        stage_index = len(policy.arm_cfg.local_se2_retry_refinement_thresholds) - 1
    prefixes = (
        f"local_se2_retry_refinement_encoders.{stage_index}.",
        f"local_se2_retry_refinement_heads.{stage_index}.",
    )
    return tuple(
        name
        for name, _parameter in policy.named_parameters()
        if name.startswith(prefixes)
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
def _parity(parent: Any, child: Any, threshold: int) -> dict[str, Any]:
    edges = radius_edges(parent.cfg, device="cpu")
    parent_hidden = parent.policy.init_hidden(1, "cpu")
    child_hidden = child.init_hidden(1, "cpu")
    observation = _hit_observation(parent, -0.176)
    old_maximum = (
        parent.policy.arm_cfg.local_se2_retry_refinement_thresholds[-1]
        if parent.policy.arm_cfg.local_se2_retry_refinement_thresholds
        else parent.policy.arm_cfg.local_se2_deep_stuck_refinement_stuck_steps
    )
    for tick in range(1, threshold + 2):
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
            raise RuntimeError(f"retry tick {tick} parity failed: {unequal}")
        expected_parent = float(min(tick, old_maximum))
        expected_child = float(min(tick, threshold))
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
            raise RuntimeError(f"retry tick {tick} counter audit failed")
        parent_hidden = parent_output.hidden
        child_hidden = child_output.hidden
    return {
        "forces_and_plan_bitwise_equal": True,
        "legacy_hidden_prefix_bitwise_equal": True,
        "parent_counter_clip": old_maximum,
        "child_counter_clip": threshold,
    }


@torch.no_grad()
def bridge(
    source: Path,
    output: Path,
    *,
    min_center_x_m: float,
    threshold: int,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = load_single_slit_checkpoint(source, device="cpu")
    cfg = parent.policy.arm_cfg
    if not (
        parent.arm == "sheaf_gbp"
        and cfg.local_se2_deep_stuck_refinement_adapter
        and cfg.local_se2_recovery_canonical_features
    ):
        raise ValueError("retry refinement requires a trained deep-stuck policy")
    previous = (
        cfg.local_se2_retry_refinement_thresholds[-1]
        if cfg.local_se2_retry_refinement_thresholds
        else cfg.local_se2_deep_stuck_refinement_stuck_steps
    )
    if threshold <= previous:
        raise ValueError("retry threshold must exceed the previous causal threshold")
    torch.manual_seed(20260831 + len(cfg.local_se2_retry_refinement_thresholds))
    child_cfg = replace(
        cfg,
        local_se2_retry_refinement_thresholds=(
            *cfg.local_se2_retry_refinement_thresholds,
            threshold,
        ),
        local_se2_retry_refinement_min_center_x_m=min_center_x_m,
    )
    child = make_policy(parent.arm, parent.cfg, child_cfg).to("cpu")
    incompatible = child.load_state_dict(parent.policy.state_dict(), strict=False)
    stage_index = len(child_cfg.local_se2_retry_refinement_thresholds) - 1
    names = retry_refinement_parameter_names(child, stage_index=stage_index)
    if set(incompatible.missing_keys) != set(names) or incompatible.unexpected_keys:
        raise RuntimeError(
            "retry tensor boundary changed: "
            f"missing={tuple(incompatible.missing_keys)} "
            f"unexpected={tuple(incompatible.unexpected_keys)}"
        )
    head = child.local_se2_retry_refinement_heads[stage_index]
    if torch.count_nonzero(head.weight).item() or torch.count_nonzero(head.bias).item():
        raise RuntimeError("retry head must start at exact zero")
    parent.policy.eval()
    child.eval()
    parity = _parity(parent, child, threshold)
    if (
        parent.policy.communication_floats_per_directed_edge_round()
        != child.communication_floats_per_directed_edge_round()
        or parent.policy.communication_rounds() != child.communication_rounds()
    ):
        raise RuntimeError("private retry refinement changed communication")
    payload = copy.deepcopy(dict(parent.metadata))
    payload.update(
        {
            "training_stage": BRIDGE_STAGE,
            "arm_cfg": asdict(child_cfg),
            "parent": {"path": str(source), "sha256": _sha256(source)},
            "retry_refinement_bridge": {
                "stage_index": stage_index,
                "thresholds": list(child_cfg.local_se2_retry_refinement_thresholds),
                "parameter_names": list(names),
                "parameter_count": sum(
                    dict(child.named_parameters())[name].numel() for name in names
                ),
                "minimum_payload_center_x_m": min_center_x_m,
                "minimum_consecutive_contact_steps": threshold,
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
        raise RuntimeError("fresh retry checkpoint changed configuration")
    return {
        "output": str(output),
        "stage_index": stage_index,
        "threshold": threshold,
        "parameter_count": payload["retry_refinement_bridge"]["parameter_count"],
        "parity": parity,
        "wire_floats": child.communication_floats_per_directed_edge_round(),
        "wire_rounds": child.communication_rounds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-center-x-m", type=float, default=-0.30)
    parser.add_argument("--threshold", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            bridge(
                args.source,
                args.output,
                min_center_x_m=args.min_center_x_m,
                threshold=args.threshold,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
