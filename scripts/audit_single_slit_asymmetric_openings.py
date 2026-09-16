#!/usr/bin/env python3
"""Audit the public T skill when either wall half is independently shortened."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import time

import torch

from piano_movers.config import HANDLE_LAYOUT_PARAMETRIC_T_V3
from piano_movers.env import collision_mask, radius_edges
from piano_movers.single_slit_0720_eval import (
    STRICT_STABLE_STEPS,
    load_single_slit_checkpoint,
    strict_final_mask,
)
from piano_movers.single_slit_contact_feedback import (
    single_slit_contact_feedback_step_v1,
)
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import SOURCE_POSE, single_slit_scenario
from piano_movers.single_slit_interactive_demo import (
    BASE_T_CAP_THICKNESS_M,
    BASE_T_HEIGHT_M,
    BASE_T_STEM_WIDTH_M,
    BASE_T_WIDTH_M,
    WORLD_Y_M,
)


DEFAULT_CASES = (
    ("narrow_centered", 0.72, 0.72),
    ("upper_half_open", 0.36, 0.72),
    ("lower_half_open", 0.72, 0.36),
    ("upper_removed", 0.0, 0.72),
    ("lower_removed", 0.72, 0.0),
    ("both_removed", 0.0, 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-agents", type=int, nargs="+", default=(8, 12, 20))
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--thickness-scale", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=384)
    parser.add_argument("--goal-y", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--goal-theta", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--initial-y", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--initial-theta", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(case[0] for case in DEFAULT_CASES),
        default=tuple(case[0] for case in DEFAULT_CASES),
    )
    args = parser.parse_args()
    if any(value < 8 or value > 20 for value in args.n_agents):
        parser.error("agent counts must lie in [8,20]")
    if args.length_scale <= 0.0 or args.thickness_scale <= 0.0:
        parser.error("shape scales must be positive")
    mission_values = (
        *args.goal_y,
        *args.goal_theta,
        *args.initial_y,
        *args.initial_theta,
    )
    if not all(math.isfinite(value) for value in mission_values):
        parser.error("mission and initial-pose panels must be finite")
    return args


def opening_from_wall_lengths(upper: float, lower: float) -> tuple[float, float]:
    upper_bottom = WORLD_Y_M - upper
    lower_top = -WORLD_Y_M + lower
    return upper_bottom - lower_top, 0.5 * (upper_bottom + lower_top)


@torch.inference_mode()
def audit_one(
    policy,
    cfg,
    *,
    upper: float,
    lower: float,
    initial_y: float,
    initial_theta: float,
    horizon: int,
):
    aperture, center = opening_from_wall_lengths(upper, lower)
    policy.bind_environment(cfg)
    state = torch.tensor(
        [[SOURCE_POSE[0], initial_y, initial_theta, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=cfg.device,
    )
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=aperture
    )
    scenario.gap_y[:, 0] = center
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    hidden = policy.init_hidden(1, cfg.device)
    edges = radius_edges(cfg, device=cfg.device)
    stable = 0
    contacts = 0
    penetrations = 0
    path_length = 0.0
    finish_tick = horizon
    outcome = "timeout"
    for tick in range(horizon):
        observation = single_slit_contact_observations_v1(
            state, scenario, cfg, context
        )
        output = policy(observation, edges, hidden)
        transition = single_slit_contact_feedback_step_v1(
            state, output.forces, scenario, cfg
        )
        contacts += int(transition.attempted_hit.sum())
        next_state = transition.resolved_state
        path_length += float(
            torch.linalg.vector_norm(next_state[0, :2] - state[0, :2]).item()
        )
        penetrations += int(collision_mask(next_state, scenario, cfg).sum())
        context = single_slit_contact_observation_context_from_result_v1(
            transition, output.forces, cfg
        )
        state = next_state
        hidden = output.hidden
        stable = stable + 1 if bool(strict_final_mask(state, scenario.goal)[0]) else 0
        if stable >= STRICT_STABLE_STEPS:
            outcome = "success"
            finish_tick = tick + 1
            break
    return {
        "outcome": outcome,
        "finish_tick": finish_tick,
        "attempted_contacts": contacts,
        "exact_penetrations": penetrations,
        "translation_path_m": path_length,
        "aperture_m": aperture,
        "gap_center_m": center,
        "upper_wall_length_m": upper,
        "lower_wall_length_m": lower,
        "initial_y_m": initial_y,
        "initial_theta_rad": initial_theta,
        "goal_y_m": float(cfg.goal_y),
        "goal_theta_rad": float(cfg.goal_theta),
        "final_state": [float(value) for value in state[0].cpu()],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loaded = load_single_slit_checkpoint(
        args.checkpoint.resolve(strict=True), device=device
    )
    policy = getattr(loaded.policy, "policy", loaded.policy).eval()
    records = []
    started = time.perf_counter()
    for n_agents, goal_y, goal_theta, initial_y, initial_theta in product(
        args.n_agents,
        args.goal_y,
        args.goal_theta,
        args.initial_y,
        args.initial_theta,
    ):
        cfg = replace(
            loaded.cfg,
            n_agents=n_agents,
            handle_layout=HANDLE_LAYOUT_PARAMETRIC_T_V3,
            payload_shape="T",
            t_total_width_m=BASE_T_WIDTH_M * args.length_scale,
            t_total_height_m=BASE_T_HEIGHT_M * args.length_scale,
            t_stem_width_m=BASE_T_STEM_WIDTH_M * args.thickness_scale,
            t_cap_thickness_m=BASE_T_CAP_THICKNESS_M * args.thickness_scale,
            goal_y=goal_y,
            goal_theta=goal_theta,
            max_steps=args.horizon,
            device=str(device),
        )
        for label, upper, lower in DEFAULT_CASES:
            if label not in args.cases:
                continue
            record = audit_one(
                policy,
                cfg,
                upper=upper,
                lower=lower,
                initial_y=initial_y,
                initial_theta=initial_theta,
                horizon=args.horizon,
            )
            record.update({"case": label, "n_agents": n_agents})
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    successes = sum(row["outcome"] == "success" for row in records)
    payload = {
        "schema_version": "piano-movers-asymmetric-opening-audit-v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "shape": {
            "letter": "T",
            "length_scale": args.length_scale,
            "thickness_scale": args.thickness_scale,
        },
        "mission_panel": {
            "goal_y_m": list(args.goal_y),
            "goal_theta_rad": list(args.goal_theta),
            "initial_y_m": list(args.initial_y),
            "initial_theta_rad": list(args.initial_theta),
        },
        "records": records,
        "successes": successes,
        "total": len(records),
        "success_rate": successes / len(records),
        "total_exact_penetrations": sum(row["exact_penetrations"] for row in records),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
