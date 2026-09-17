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
from train_mainline_action_dagger import composition_cfg
from piano_movers.env import radius_edges
from piano_movers.multi_waypoint_demo import (
    Slit, WaypointCourse, SkillCompositionOptions, _composition_observation, _course_config,
    apply_solver_controls,
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
    parser.add_argument("--tick", type=int, default=-1,
                        help="Authenticated source tick, negative indices count from the end")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete":
        raise ValueError("requires completed source rollouts")
    parent = load_single_slit_checkpoint(args.parent, device="cpu")
    if source["checkpoint_sha256"] != parent.checkpoint_sha256:
        raise ValueError("parent checkpoint mismatch")
    row = next(r for r in source["results"] if r["shape"] == args.shape and r["variant"] == args.variant
               and (args.case_id is None or r.get("case_id") == args.case_id))
    path = args.rollouts.parent / row["trace"]
    if sha256(path) != row["trace_sha256"]:
        raise ValueError("source trace mismatch")
    arrays = np.load(path)
    tick = args.tick if args.tick >= 0 else len(arrays["state"]) + args.tick
    if not 0 <= tick < len(arrays["state"]):
        raise ValueError("tick outside source trajectory")
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
    cfg = composition_cfg(replace(cfg, max_steps=row.get("horizon", 1100)), options)
    parent.policy.bind_environment(cfg)
    apply_solver_controls(parent.policy, options)
    edges = radius_edges(cfg, device="cpu")
    results = []
    reference = None
    reference_features = None
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
        planner_inputs = parent.policy.local_se2_base_planner_inputs(observation, hidden)
        features = parent.policy.local_se2_markov_shape_inputs_from_planner_inputs(planner_inputs)
        out = parent.policy(observation, edges, hidden)
        tensors = (observation, out.aux["local_se2_plan_mean"], out.aux["local_se2_plan_used"], out.forces)
        if reference is None:
            reference_features = features.clone()
            reference = tuple(x.clone() for x in tensors)
            for actual, field in zip(tensors, ("observation", "learned_plan", "plan", "forces")):
                torch.testing.assert_close(actual, torch.from_numpy(arrays[field][tick:tick + 1]),
                                           atol=2e-5, rtol=2e-5)
        results.append({
            "goal": goal, "orientation_required": required,
            "private_feature_width": features.shape[-1],
            "private_features_bitwise_equal": torch.equal(features, reference_features),
            "private_feature_max_absolute_delta": float((features - reference_features).abs().max()),
            "private_twist3_max_absolute_delta": float((tensors[1][..., 3:6] - reference[1][..., 3:6]).abs().max()),
            "max_absolute_delta_from_reference": dict(zip(
                ("observation27", "private_plan8", "executed_plan8", "force"),
                [float((a - b).abs().max()) for a, b in zip(tensors, reference)],
            )),
        })
    result = {
        "schema": "piano-movers.original-goal-observability.v2",
        "checkpoint_sha256": parent.checkpoint_sha256,
        "rollout_manifest_sha256": sha256(args.rollouts),
        "trace_sha256": row["trace_sha256"], "tick": tick,
        "case_id": row.get("case_id"), "shape": row["shape"],
        "options": row["options"], "reference_replay_verified": True,
        "scope": "same state and causal memory; one-step command sensitivity, not closed-loop capability",
        "source_sha256": {str(p): sha256(p) for p in (
            Path(__file__), Path("scripts/train_mainline_action_dagger.py"),
            Path("scripts/collect_mainline_skill_dagger.py"),
            *sorted(Path("src/piano_movers").glob("*.py")),
        )}, "results": results,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
