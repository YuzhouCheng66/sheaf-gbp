#!/usr/bin/env python3
"""Initial-state coordinate sensitivity, not a memorization or closed-loop test.

Translate the load, local walls and active goal together while retaining the
same normalization/world-size constants. The outer arena is NOT translated;
therefore this is not a claim of a globally identical physical task. It probes
the response to changing the locally observed scene origin. Every probe starts
with the true initial zero memory, avoiding incorrect transport of recurrent
hidden state. It never writes a checkpoint or changes a public demo.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.env import radius_edges
from piano_movers.multi_waypoint_demo import (
    Slit, WaypointCourse, SkillCompositionOptions, _course_config,
    _composition_observation, apply_solver_controls, INTERMEDIATE_POSITION_TOLERANCE_M,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_observation import initial_single_slit_contact_observation_context_v1


def translated_course(course, dx, dy):
    def pose(p):
        return (p[0] + dx, p[1] + dy, p[2])

    return replace(course, source=pose(course.source),
                   walls=tuple(replace(w, x=w.x + dx, gap_y=w.gap_y + dy) for w in course.walls),
                   waypoints=tuple(pose(g) for g in course.waypoints))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--candidate", type=Path,
                        help="Optional different checkpoint: explicit off-policy initial-state probe, not its rollout")
    parser.add_argument("--shapes", nargs="+", default=list("ABCIOTX"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete":
        raise ValueError("requires a complete actual rollout panel")
    actor_path = Path(source["checkpoint"])
    if sha256(actor_path) != source["checkpoint_sha256"]:
        raise ValueError("actor checkpoint mismatch")
    loaded = load_single_slit_checkpoint(args.candidate or actor_path, device="cpu")
    rows = []
    for row in source["results"]:
        if row["shape"] not in args.shapes or "trace" not in row:
            continue
        h, options = row["header"], SkillCompositionOptions(**row["options"])
        course = WaypointCourse(tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
                                tuple(tuple(g) for g in h["waypoints"]), h["world"]["x"], h["world"]["y"],
                                tuple(h.get("source_velocity", (0., 0., 0.))), h.get("morphology_scale", 1.))
        path = args.rollouts.parent / row["trace"]
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("trace checksum mismatch")
        with np.load(path) as arrays:
            state = torch.from_numpy(arrays["state"][:1].copy())
            actual_obs = torch.from_numpy(arrays["observation"][:1].copy())
            actual_force = torch.from_numpy(arrays["forces"][:1].copy())
            if np.count_nonzero(arrays["private_hidden"][0]):
                raise ValueError("probe requires a true zero-memory start")

        def evaluate(dx, dy, plan_override=None):
            shifted = translated_course(course, dx, dy)
            cfg = _course_config(loaded.cfg, row["agents"], shifted, row["shape"])
            cfg = replace(cfg, goal_orientation_required=not options.position_only_control,
                          complete_local_corridor_exit=options.complete_corridor_exit,
                          local_cspace_collision_substeps=options.planning_substeps,
                          local_cspace_validate_all_lookaheads=options.validate_all_cspace_lookaheads,
                          local_support_check_alignment_sweep=options.check_support_alignment_sweep,
                          local_corridor_exit_goal_radius_m=INTERMEDIATE_POSITION_TOLERANCE_M
                          if options.respect_handoff_goal_region else 0.)
            if options.nominal_glyph_size:
                cfg = replace(cfg, glyph_scale=1.)
            loaded.policy.bind_environment(cfg)
            apply_solver_controls(loaded.policy, options)
            shifted_state = state.clone()
            shifted_state[:, :2] += shifted_state.new_tensor([dx, dy])
            context = initial_single_slit_contact_observation_context_v1(shifted_state, cfg)
            obs = _composition_observation(shifted, shifted_state, 0, cfg, context,
                                           retain_nearby_wall=options.retain_nearby_wall,
                                           retain_previous_wall_only=options.retain_previous_wall_only)
            hidden = loaded.policy.init_hidden(1, "cpu")
            output = loaded.policy(obs, radius_edges(cfg, device="cpu"), hidden,
                                   local_se2_plan_override=plan_override)
            private = loaded.policy.local_se2_base_planner_inputs(obs, hidden)
            features = loaded.policy.local_se2_markov_shape_inputs_from_planner_inputs(private)
            return obs, features, output

        baseline_obs, baseline_features, baseline = evaluate(0., 0.)
        torch.testing.assert_close(baseline_obs, actual_obs, atol=0., rtol=0.)
        if args.candidate is None:
            torch.testing.assert_close(baseline.forces, actual_force, atol=0., rtol=0.)
        for dx, dy in ((.75, 0.), (-.75, 0.), (0., .4), (0., -.4)):
            obs, features, out = evaluate(dx, dy)
            # Override BEFORE the original analytic geometry branch. The
            # final-plan override would erase its confidence mask and would
            # not isolate the private-planner response.
            _, _, same_plan = evaluate(dx, dy, baseline.aux["local_se2_plan_mean"])
            rows.append({
                "case_id": row.get("case_id"), "shape": row["shape"],
                "profile": row.get("profile"), "agents": row["agents"],
                "translation_m": [dx, dy], "source_trace_sha256": row["trace_sha256"],
                "max_observation_delta": float((obs - baseline_obs).abs().max()),
                "changed_private_columns_above_1e_5": torch.nonzero(
                    (features - baseline_features).abs().amax((0, 1)) > 1e-5
                ).flatten().tolist(),
                "max_private_plan_delta": float((out.aux["local_se2_plan_mean"]
                                                 - baseline.aux["local_se2_plan_mean"]).abs().max()),
                "max_executed_plan_delta": float((out.aux["local_se2_plan_used"]
                                                  - baseline.aux["local_se2_plan_used"]).abs().max()),
                "max_force_delta_N": float((out.forces - baseline.forces).abs().max()),
                "force_delta_with_original_private_plan_N": float((same_plan.forces - baseline.forces).abs().max()),
                "same_geometry_mask_with_original_private_plan": bool(torch.equal(
                    same_plan.aux["local_se2_terminal_attractor_active"],
                    baseline.aux["local_se2_terminal_attractor_active"])),
            })
    if not rows:
        raise ValueError("no probes")
    aggregates = {}
    for axis, component in (("x", 0), ("y", 1)):
        subset = [r for r in rows if r["translation_m"][component] != 0.]
        aggregates[axis] = {"probes": len(subset),
                            "force_max_delta_N": max(r["max_force_delta_N"] for r in subset),
                            "force_median_delta_N": float(np.median([r["max_force_delta_N"] for r in subset])),
                            "force_delta_above_1e_3": sum(r["max_force_delta_N"] > 1e-3 for r in subset),
                            "force_max_delta_with_original_private_plan_N": max(r["force_delta_with_original_private_plan_N"] for r in subset),
                            "unchanged_geometry_masks": sum(r["same_geometry_mask_with_original_private_plan"] for r in subset)}
    report = {"status": "complete", "checkpoint_sha256": loaded.checkpoint_sha256,
              "state_source_checkpoint_sha256": source["checkpoint_sha256"],
              "off_policy_initial_state_probe": args.candidate is not None,
              "factual_force_parity_checked": args.candidate is None,
              "rollout_manifest_sha256": sha256(args.rollouts), "results": rows,
              "audit_source_sha256": sha256(Path(__file__)),
              "aggregates": aggregates, "untranslated_outer_arena": True,
              "zero_memory_initial_states_only": True,
              "interpretation": "local coordinate sensitivity; not a memorization diagnosis or a closed-loop SE(2) invariance claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
