#!/usr/bin/env python3
"""Counterfactual current-goal visibility in the original private planner.

Different targets, same geometry/state/private history. This distinguishes
what the learned planner receives from goal-dependent analytic control later
in the original stack; it does not conclude that the whole policy ignores goals.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from collect_mainline_skill_dagger import context_from_original_observation
from piano_movers.env import radius_edges
from piano_movers.multi_waypoint_demo import (
    Slit, WaypointCourse, SkillCompositionOptions, _composition_observation, _course_config,
    INTERMEDIATE_POSITION_TOLERANCE_M,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="L")
    parser.add_argument("--variant", default="coherent_handoff")
    parser.add_argument("--case-id", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = json.loads(args.rollouts.read_text())
    parent = load_single_slit_checkpoint(args.parent, device="cpu")
    if source["checkpoint_sha256"] != parent.checkpoint_sha256:
        raise ValueError("parent checkpoint mismatch")
    row = next(r for r in source["results"] if r["shape"] == args.shape and r["variant"] == args.variant
               and (args.case_id is None or r.get("case_id") == args.case_id))
    path = args.rollouts.parent / row["trace"]
    if sha256(path) != row["trace_sha256"]:
        raise ValueError("source trace mismatch")
    arrays = np.load(path)
    tick = len(arrays["state"]) - 1
    stage = int(arrays["waypoint_index"][tick])
    header = row["header"]
    course = WaypointCourse(tuple(header["source"]), tuple(Slit(**s) for s in header["walls"]), tuple(tuple(g) for g in header["waypoints"]), header["world"]["x"], header["world"]["y"])
    course = replace(course, source_velocity=tuple(header.get("source_velocity", (0., 0., 0.))),
                     morphology_scale=float(header.get("morphology_scale", 1.)))
    options = SkillCompositionOptions(**row["options"])
    state = torch.from_numpy(arrays["state"][tick:tick + 1].copy())
    hidden = torch.from_numpy(arrays["private_hidden"][tick:tick + 1].copy())
    obs = torch.from_numpy(arrays["observation"][tick:tick + 1].copy())
    context = context_from_original_observation(obs)
    cfg = _course_config(parent.cfg, row["agents"], course, row["shape"])
    if options.nominal_glyph_size:
        cfg = replace(cfg, glyph_scale=1.)
    cfg = replace(cfg, local_cspace_collision_substeps=options.planning_substeps,
                  local_cspace_validate_all_lookaheads=options.validate_all_cspace_lookaheads,
                  local_support_check_alignment_sweep=options.check_support_alignment_sweep,
                  local_corridor_exit_goal_radius_m=(INTERMEDIATE_POSITION_TOLERANCE_M
                                                     if options.respect_handoff_goal_region else 0.))
    parent.policy.bind_environment(cfg)
    edges = radius_edges(cfg, device="cpu")
    results = []
    reference = None
    base_theta = course.waypoints[stage][2]
    base_required = stage == len(course.waypoints) - 1 or not options.position_only_control
    for dy, theta, required in ((0., base_theta, base_required), (.3, base_theta, base_required),
                                (-.3, base_theta, base_required), (0., 1., True),
                                (0., base_theta, not base_required)):
        goal = list(course.waypoints[stage])
        goal[1] += dy
        goal[2] = theta
        points = list(course.waypoints)
        points[stage] = tuple(goal)
        alternative = replace(course, waypoints=tuple(points))
        active = course.walls[min(stage, len(course.walls) - 1)]
        local_cfg = replace(cfg, goal_x=goal[0] - active.x, goal_y=goal[1], goal_theta=theta, goal_orientation_required=required, complete_local_corridor_exit=options.complete_corridor_exit)
        parent.policy.env_cfg = local_cfg
        observation = _composition_observation(alternative, state, stage, local_cfg, context, retain_nearby_wall=options.retain_nearby_wall, retain_previous_wall_only=options.retain_previous_wall_only)
        out = parent.policy(observation, edges, hidden)
        tensors = (observation, out.aux["local_se2_plan_mean"], out.aux["local_se2_plan_used"], out.forces)
        if reference is None:
            reference = tuple(x.clone() for x in tensors)
            for actual, field in zip(tensors, ("observation", "learned_plan", "plan", "forces")):
                torch.testing.assert_close(actual, torch.from_numpy(arrays[field][tick:tick + 1]),
                                           atol=2e-5, rtol=2e-5)
        results.append({
            "goal": goal, "orientation_required": required,
            "max_absolute_delta_from_reference": dict(zip(
                ("observation27", "private_plan8", "executed_plan8", "force"),
                [float((a - b).abs().max()) for a, b in zip(tensors, reference)],
            )),
        })
    result = {"schema": "piano-movers.original-goal-observability.v1", "checkpoint_sha256": parent.checkpoint_sha256, "trace_sha256": row["trace_sha256"], "tick": tick, "shape": row["shape"], "source_sha256": {str(p): sha256(p) for p in (Path(__file__), Path("src/piano_movers/models.py"), Path("src/piano_movers/env.py"))}, "results": results}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
