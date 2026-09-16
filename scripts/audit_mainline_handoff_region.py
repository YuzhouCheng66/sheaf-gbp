#!/usr/bin/env python3
"""Diagnose controller/supervisor goal-region conflicts on archived failures.

This produces pose-level counterfactual witnesses, not executable teachers or
closed-loop successes. It does not relax the supervisor or its body reserve.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.local_skill_teacher import certify_pose_path
from piano_movers.multi_waypoint_demo import (
    Slit, WaypointCourse, _course_config, _full_collision_scenario,
    INTERMEDIATE_POSITION_TOLERANCE_M,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_local_planner import LOCAL_SE2_TERMINAL_WALL_RESERVE_M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--shape", default="L")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    loaded = load_single_slit_checkpoint(args.checkpoint, device="cpu")
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete" or source["checkpoint_sha256"] != loaded.checkpoint_sha256:
        raise ValueError("requires a complete checksum-matched actor audit")
    results = []
    for row in source["results"]:
        if row["shape"] != args.shape or row["result"]["success"] or "trace" not in row:
            continue
        trace = args.rollouts.parent / row["trace"]
        if sha256(trace) != row["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        arrays, h = np.load(trace), row["header"]
        q = arrays["state"][-1].astype(np.float64)
        stage = int(arrays["waypoint_index"][-1])
        if stage == len(h["waypoints"]) - 1:
            continue
        course = WaypointCourse(tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
                                tuple(tuple(g) for g in h["waypoints"]), h["world"]["x"], h["world"]["y"],
                                tuple(h.get("source_velocity", (0., 0., 0.))), h.get("morphology_scale", 1.))
        cfg = _course_config(loaded.cfg, row["agents"], course, row["shape"])
        if row["options"]["nominal_glyph_size"]:
            cfg = replace(cfg, glyph_scale=1.)
        vertices = np.array(h["body_rects"]).reshape(-1, 2)
        offset_min = (vertices[:, 0] * np.cos(q[2]) - vertices[:, 1] * np.sin(q[2])).min()
        goal = np.asarray(h["waypoints"][stage])
        minimum_x = course.walls[min(stage, len(course.walls) - 1)].x + cfg.wall_slab + LOCAL_SE2_TERMINAL_WALL_RESERVE_M - offset_min
        maximum_x = goal[0] + INTERMEDIATE_POSITION_TOLERANCE_M
        slack = maximum_x - minimum_x
        original_target_x = max(goal[0], minimum_x + .005)
        candidate = q[:3].copy()
        candidate[:2] = (max(goal[0], minimum_x + min(.005, .5 * slack)), goal[1])
        certificate = certify_pose_path(torch.from_numpy(np.stack((q[:3], candidate))),
                                        _full_collision_scenario(course, cfg), cfg)
        results.append({
            "case_id": row.get("case_id"), "trace_sha256": row["trace_sha256"],
            "current_pose": q[:3].tolist(), "active_goal": goal.tolist(),
            "position_tolerance_m": INTERMEDIATE_POSITION_TOLERANCE_M,
            "body_clearance_reserve_m": LOCAL_SE2_TERMINAL_WALL_RESERVE_M,
            "fixed_heading_goal_y_feasible_center_interval": [float(minimum_x), float(maximum_x)],
            "positive_interval_slack_m": float(slack), "original_extra_tracking_reserve_m": .005,
            "original_target_x": float(original_target_x),
            "original_target_outside_goal_region": bool(original_target_x > maximum_x),
            "counterfactual_pose": candidate.tolist(),
            "counterfactual_inside_goal_region": bool(np.linalg.norm(candidate[:2] - goal[:2]) < INTERMEDIATE_POSITION_TOLERANCE_M),
            "counterfactual_retains_body_reserve": bool(candidate[0] > minimum_x),
            "counterfactual_path_certificate": asdict(certificate),
            "executable_closed_loop_success": "not_evaluated",
        })
    if not results:
        raise ValueError("no failed intermediate episodes matched")
    report = {"status": "complete", "checkpoint_sha256": loaded.checkpoint_sha256,
              "rollout_sha256": sha256(args.rollouts), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
