#!/usr/bin/env python3
"""Enable a source-null physical retreat/re-entry latch without new weights."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from typing import Any

import torch

from piano_movers.env import radius_edges
from piano_movers.models import LOCAL_SE2_PLAN_MEAN_AUX_KEY, make_policy
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_feedback import (
    single_slit_contact_feedback_step_v1,
)
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import single_slit_scenario


BRIDGE_STAGE = "analytic-local-se2-phase-latch-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _precontact_parity(parent: Any, child: torch.nn.Module) -> dict[str, Any]:
    cfg = parent.cfg
    scenario = single_slit_scenario(1, cfg)
    state = scenario.state0.clone()
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    edges = radius_edges(cfg, device=state.device)
    parent_hidden = parent.policy.init_hidden(1, state.device)
    child_hidden = child.init_hidden(1, state.device)
    compared = 0
    for tick in range(cfg.max_steps):
        observation = single_slit_contact_observations_v1(
            state, scenario, cfg, context
        )
        parent_output = parent.policy(observation, edges, parent_hidden)
        child_output = child(observation, edges, child_hidden)
        for label, left, right in (
            ("force", parent_output.forces, child_output.forces),
            ("hidden", parent_output.hidden, child_output.hidden),
            (
                "plan",
                parent_output.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY],
                child_output.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY],
            ),
        ):
            if not torch.equal(left, right):
                raise RuntimeError(
                    f"analytic phase latch changed precontact {label} at tick {tick}"
                )
        transition = single_slit_contact_feedback_step_v1(
            state, parent_output.forces, scenario, cfg
        )
        compared += 1
        if bool(transition.attempted_hit[0]):
            return {
                "precontact_and_trigger_outputs_bitwise_equal": True,
                "compared_policy_calls": compared,
                "first_contact_tick": tick,
            }
        context = single_slit_contact_observation_context_from_result_v1(
            transition, parent_output.forces, cfg
        )
        state = transition.resolved_state
        parent_hidden = parent_output.hidden
        child_hidden = child_output.hidden
    raise RuntimeError("source policy never produced the required contact event")


def bridge(
    source_path: Path,
    output_path: Path,
    *,
    threshold_m: float,
    latch_logit: float,
    adapter_scale: float,
) -> None:
    source_path = source_path.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = load_single_slit_checkpoint(source_path, device="cpu")
    if not parent.policy.arm_cfg.local_se2_recovery_adapter:
        raise ValueError("analytic phase latch requires a recovery-adapter source")
    if (
        parent.policy.arm_cfg.local_se2_analytic_phase_latch
        and parent.policy.arm_cfg.local_se2_recovery_adapter_scale == adapter_scale
        and parent.policy.arm_cfg.local_se2_reentry_center_x_threshold_m
        == threshold_m
        and parent.policy.arm_cfg.local_se2_phase_latch_logit == latch_logit
    ):
        raise ValueError("source already has the requested analytic phase latch")
    child_cfg = replace(
        parent.policy.arm_cfg,
        local_se2_analytic_phase_latch=True,
        local_se2_reentry_center_x_threshold_m=threshold_m,
        local_se2_phase_latch_logit=latch_logit,
        local_se2_recovery_adapter_scale=adapter_scale,
    )
    child = make_policy(parent.arm, parent.cfg, child_cfg).to("cpu")
    child.load_state_dict(parent.policy.state_dict(), strict=True)
    child.eval()
    parent.policy.eval()
    if any(
        not torch.equal(child.state_dict()[name], value)
        for name, value in parent.policy.state_dict().items()
    ):
        raise RuntimeError("parameter-free phase bridge changed a source tensor")
    parity = _precontact_parity(parent, child)
    if (
        child.communication_floats_per_directed_edge_round()
        != parent.policy.communication_floats_per_directed_edge_round()
        or child.communication_rounds() != parent.policy.communication_rounds()
    ):
        raise RuntimeError("analytic phase latch changed communication accounting")
    payload = copy.deepcopy(dict(parent.metadata))
    payload.update(
        {
            "training_stage": BRIDGE_STAGE,
            "arm_cfg": asdict(child_cfg),
            "parent": {"path": str(source_path), "sha256": _sha256(source_path)},
            "analytic_phase_latch": {
                "source_null_before_real_contact": True,
                "private_state_only": True,
                "additional_parameters": 0,
                "additional_wire_floats": 0,
                "reentry_center_x_threshold_m": threshold_m,
                "phase_logit": latch_logit,
                "recovery_adapter_scale": adapter_scale,
                "phase_order": "contact -> retreat -> irreversible re-entry",
                "parity": parity,
            },
            "policy_state_dict": {
                name: value.detach().cpu() for name, value in child.state_dict().items()
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    # Fresh reconstruction is part of the bridge transaction.
    reloaded = load_single_slit_checkpoint(output_path, device="cpu")
    if reloaded.policy.arm_cfg != child_cfg:
        raise RuntimeError("fresh phase-latch checkpoint reconstruction changed config")
    print(
        {
            "output": str(output_path),
            "precontact_parity": parity,
            "threshold_m": threshold_m,
            "adapter_scale": adapter_scale,
            "wire_floats": child.communication_floats_per_directed_edge_round(),
            "wire_rounds": child.communication_rounds(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=-0.16)
    parser.add_argument("--latch-logit", type=float, default=8.0)
    parser.add_argument("--adapter-scale", type=float, default=2.0)
    args = parser.parse_args()
    bridge(
        args.source,
        args.output,
        threshold_m=args.threshold_m,
        latch_logit=args.latch_logit,
        adapter_scale=args.adapter_scale,
    )


if __name__ == "__main__":
    main()
