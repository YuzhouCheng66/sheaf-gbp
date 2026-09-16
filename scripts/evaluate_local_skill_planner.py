#!/usr/bin/env python3
"""Closed-loop local planner + physical Sheaf-GBP development evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.config import EnvConfig
from piano_movers.env import PianoBatch, collision_mask, wrap_angle
from piano_movers.local_skill_policy import LocalSkillPlanner, LocalSkillController
from piano_movers.local_skill_teacher import certify_pose_path
from piano_movers.multi_waypoint_demo import (
    DEFAULT_COURSE,
    _course_config,
    _full_collision_scenario,
)
from piano_movers.single_slit_0720_eval import (
    load_single_slit_checkpoint,
    strict_final_mask,
    STRICT_STABLE_STEPS,
)
from piano_movers.single_slit_contact_feedback import single_slit_contact_feedback_step_v1
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
)


@torch.inference_mode()
def run(controller, cfg, state, scene, goals, orientation_flags, horizon):
    controller.reset(cfg)
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    stage = dwell = hits = payload = setup = pose_collisions = 0
    states = [state[0].clone()]
    features, plans, forces, stages, hit_flags = [], [], [], [], []
    done = False
    for tick in range(horizon):
        goal = state.new_tensor(goals[stage])
        out, observation, plan = controller.step(
            state, scene, goal, orientation_flags[stage], context
        )
        transition = single_slit_contact_feedback_step_v1(state, out.forces, scene, cfg)
        next_state = transition.resolved_state
        features.append(observation[0].cpu().numpy())
        plans.append(plan[0].cpu().numpy())
        forces.append(out.forces[0].cpu().numpy())
        stages.append(stage)
        hit_flags.append(bool(transition.attempted_hit[0]))
        hits += int(transition.attempted_hit[0])
        pose_collisions += int(collision_mask(next_state, scene, cfg)[0])
        trace = out.aux["gbp_message_trace"]
        payload += trace.wire_float_count * trace.natural.element_size()
        messages = out.aux["local_se2_distributed_grasp_message_trace"]
        setup += messages.numel() * messages.element_size()
        close = bool(torch.linalg.vector_norm(next_state[0, :2] - goal[:2]) <= 0.14)
        if stage < len(goals) - 1 and close:
            stage += 1
            dwell = 0
        elif stage == len(goals) - 1:
            if orientation_flags[stage]:
                passed = bool(strict_final_mask(next_state, goal.view(1, 3))[0])
            else:
                passed = (
                    bool(torch.linalg.vector_norm(next_state[0, :2] - goal[:2]) < 0.05)
                    and bool(torch.linalg.vector_norm(next_state[0, 3:5]) < 0.03)
                    and bool(next_state[0, 5].abs() < 0.04)
                )
            dwell = dwell + 1 if passed and not transition.attempted_hit[0] else 0
            if dwell >= STRICT_STABLE_STEPS:
                done = True
        context = single_slit_contact_observation_context_from_result_v1(
            transition, out.forces, cfg
        )
        state = next_state
        states.append(state[0].clone())
        if done:
            break
    trajectory = torch.stack(states).cpu()
    certificate = certify_pose_path(trajectory, scene, replace(cfg, device="cpu"))
    path_length = float(
        torch.linalg.vector_norm(trajectory[1:, :2] - trajectory[:-1, :2], dim=-1).sum()
    )
    result = {
        "success": done,
        "steps": tick + 1,
        "goals_completed": stage + int(done),
        "contacts": hits,
        "resolved_pose_collisions": pose_collisions,
        "continuous_certificate_passed": certificate.passed,
        "continuous_certificate_reason": certificate.reason,
        "minimum_sampled_clearance_m": certificate.minimum_sampled_clearance_m,
        "path_length_m": path_length,
        "gbp_payload_bytes": payload,
        "grasp_setup_payload_bytes": setup,
        "goal_position_error_m": float(torch.linalg.vector_norm(state[0, :2] - goal[:2])),
        "goal_angle_error_rad": float(wrap_angle(state[0, 2] - goal[2]).abs()),
        "wire_headers_included": False,
    }
    return result, {
        "states": trajectory.numpy(),
        "features": np.stack(features),
        "plan": np.stack(plans),
        "forces": np.stack(forces),
        "stages": np.asarray(stages),
        "hit": np.asarray(hit_flags),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--shapes", nargs="+", default=["A", "B", "L", "T", "X"])
    parser.add_argument("--agents", type=int, nargs="+", default=[12])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--diameter-rounds", action="store_true")
    parser.add_argument("--horizon", type=int, default=1100)
    parser.add_argument("--case-limit", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    parent = load_single_slit_checkpoint(args.parent, device=args.device)
    saved = torch.load(args.planner, map_location=args.device, weights_only=False)
    if saved["parent_checkpoint_sha256"] != parent.checkpoint_sha256:
        raise ValueError("planner/action-parent lineage mismatch")
    planner = LocalSkillPlanner(
        saved["hidden_dim"], goal_residual=saved.get("goal_residual", False)
    ).to(args.device)
    planner.load_state_dict(saved["policy_state_dict"], strict=True)
    controller = LocalSkillController(parent.policy, planner, diameter_rounds=args.diameter_rounds)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    record = {
        "schema": "piano-movers.p6-local-closed-loop.v1",
        "status": "running",
        "split": "development",
        "parent_sha256": parent.checkpoint_sha256,
        "planner_sha256": sha256(args.planner),
        "planner_step": saved["step"],
        "goal_residual": saved.get("goal_residual", False),
        "source_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path("src/piano_movers/local_skill_policy.py"),
                Path("src/piano_movers/local_skill_teacher.py"),
            ]
        },
        "device": args.device,
        "diameter_rounds": args.diameter_rounds,
        "results": [],
    }
    write_json(destination, record)
    tasks = []
    if args.dataset:
        dataset = json.loads(args.dataset.read_text())
        eligible = [
            r
            for r in dataset["results"]
            if r["success"] and r["shape"] in args.shapes and r["agents"] in args.agents
        ]
        record["split"] = "seen_training_cases_smoke"
        for row in eligible[: args.case_limit]:
            cfg = replace(EnvConfig(**row["env_cfg"]), device=args.device)
            state = torch.tensor([row["start_state"]], device=args.device)
            goal = state.new_tensor([row["goal"]])
            walls = row["walls"]
            scene = PianoBatch(
                state0=state,
                source=state[:, :3],
                goal=goal,
                wall_x=state.new_tensor([walls["x"]]),
                gap_y=state.new_tensor([walls["gap_y"]]),
                gap_half=state.new_tensor([walls["gap_half"]]),
            )
            tasks.append(
                (row["case_id"], cfg, state, scene, [row["goal"]], [row["orientation_required"]])
            )
    else:
        for shape in args.shapes:
            for count in args.agents:
                cfg = _course_config(parent.cfg, count, DEFAULT_COURSE, shape)
                scene = _full_collision_scenario(DEFAULT_COURSE, cfg)
                tasks.append(
                    (
                        f"multiwall_{shape}_{count}",
                        cfg,
                        scene.state0,
                        scene,
                        DEFAULT_COURSE.waypoints,
                        [False, False, False, True],
                    )
                )
    for case_id, cfg, state, scene, goals, orientation in tasks:
        started = time.monotonic()
        result, data = run(controller, cfg, state, scene, goals, orientation, args.horizon)
        artifact = args.output / f"{case_id}.npz"
        np.savez_compressed(artifact, **data)
        row = {
            "case_id": case_id,
            "shape": cfg.payload_shape,
            "agents": cfg.n_agents,
            "gbp_rounds": controller.rounds,
            "tree_diameter": controller.tree_diameter,
            "env_cfg": asdict(cfg),
            "goals": [list(g) for g in goals],
            "orientation_required": orientation,
            "walls": {
                "x": scene.wall_x[0].tolist(),
                "gap_y": scene.gap_y[0].tolist(),
                "gap_half": scene.gap_half[0].tolist(),
            },
            "artifact": artifact.name,
            "artifact_sha256": sha256(artifact),
            "elapsed_seconds": time.monotonic() - started,
            **result,
        }
        record["results"].append(row)
        write_json(destination, record)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "case_id",
                        "success",
                        "goals_completed",
                        "contacts",
                        "steps",
                        "goal_position_error_m",
                        "continuous_certificate_passed",
                    )
                }
            ),
            flush=True,
        )
    record["status"] = "complete"
    write_json(destination, record)


if __name__ == "__main__":
    main()
