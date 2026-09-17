#!/usr/bin/env python3
"""Audit a locked mainline recipe on predeclared, outcome-independent courses.

Results include every requested case. Invalid endpoints and uncertified
feasibility remain visible, not silently filtered. State profiles add stratified
large-angle, moving-entry and morphology controls. They do not fabricate
contact history or constitute a full maze benchmark.
"""

from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_local_skill_composition import Recorder, sha256, write_json
from piano_movers.local_skill_teacher import certify_pose_path
from piano_movers.local_skill_validation import (
    FAMILY_SEEDS, LAYOUT_FAMILIES, STATE_PROFILES, validate_split_seeds, state_profile_course,
    sequence_length_course,
)
from piano_movers.multi_waypoint_demo import (
    MultiWaypointEngine, SkillCompositionOptions, _course_config, _full_collision_scenario,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("configs/p6_mainline_candidate_v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(FAMILY_SEEDS), default="locked_validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[82201])
    parser.add_argument("--families", nargs="+", choices=LAYOUT_FAMILIES, default=LAYOUT_FAMILIES)
    parser.add_argument("--profiles", nargs="+", choices=STATE_PROFILES, default=["nominal"])
    parser.add_argument("--shapes", nargs="+", default=list("ABILOSTZ"))
    parser.add_argument("--agents", nargs="+", type=int, default=[12])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-wall-counts", nargs="+", type=int, choices=(1, 2, 3, 5),
                        help="Independent fixed-arena sequence-length panel, not the historical three-wall draw")
    args = parser.parse_args()
    validate_split_seeds(args.split, args.seeds)
    if args.sequence_wall_counts and (len(set(args.sequence_wall_counts)) != len(args.sequence_wall_counts)
                                     or args.families != ["combined"]):
        raise ValueError("sequence mode requires unique lengths and --families combined")
    for values in (args.shapes, args.agents, args.families, args.profiles):
        if not values or len(set(values)) != len(values):
            raise ValueError("empty or duplicated case dimensions")
    if any(s not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(s) != 1 for s in args.shapes):
        raise ValueError("shapes must be uppercase A-Z")
    if any(n not in range(8, 21) for n in args.agents):
        raise ValueError("agent counts must be 8-20")
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads(args.recipe.read_text())
    checkpoint = (root / recipe["checkpoint"]).resolve(strict=True)
    if sha256(checkpoint) != recipe["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint hash mismatch")
    options = SkillCompositionOptions(**recipe["composition_options"])
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "manifest.json"
    catalog_path = args.output / "case_catalog.json"
    if destination.exists() or catalog_path.exists():
        raise FileExistsError("use a fresh output directory")
    torch.set_num_threads(1)
    cases = [
        {"case_id": (f"{family}_s{seed}_{shape}_n{n}" if profile == "nominal" else
                     f"{family}_{profile}_s{seed}_{shape}_n{n}"), "family": family,
         "seed": seed, "shape": shape, "agents": n,
         "profile": profile, "course": asdict(state_profile_course(seed, family, profile))}
        for family in args.families for seed in args.seeds
        for profile in args.profiles for shape in args.shapes for n in args.agents
    ]
    if args.sequence_wall_counts:
        cases = [
            {"case_id": f"sequence{count}_{profile}_s{seed}_{shape}_n{n}",
             "family": "sequence_length", "seed": seed, "shape": shape, "agents": n,
             "profile": profile, "wall_count": count,
             "course": asdict(sequence_length_course(seed, profile, count))}
            for seed in args.seeds for count in args.sequence_wall_counts
            for profile in args.profiles for shape in args.shapes for n in args.agents
        ]
    # Freeze the exact denominator and provenance before the first simulation.
    write_json(catalog_path, {"split": args.split, "cases": cases})
    record = {
        "schema": "piano-movers.mainline-layout-validation.v1", "status": "running",
        "split": args.split, "recipe": str(args.recipe.resolve()),
        "recipe_sha256": sha256(args.recipe), "recipe_snapshot": recipe,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "options": asdict(options),
        "catalog_sha256": sha256(catalog_path), "requested_cases": len(cases),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in (
            Path(__file__), root / "scripts/audit_local_skill_composition.py",
            *sorted((root / "src/piano_movers").glob("*.py")),
        )},
        "device": args.device, "initial_velocity": "per_case_source_velocity",
        "outcome_based_admission": False, "teacher_labels_present": False,
        "parent_unseen_shape_claim": False, "results": [],
        "sequence_wall_counts": args.sequence_wall_counts,
        "sequence_horizon_rule": "ceil(1100*wall_count/3)" if args.sequence_wall_counts else None,
    }
    write_json(destination, record)
    for case in cases:
        started = time.monotonic()
        course = (sequence_length_course(case["seed"], case["profile"], case["wall_count"])
                  if args.sequence_wall_counts else
                  state_profile_course(case["seed"], case["family"], case["profile"]))
        horizon = (1100 * len(course.walls) + 2) // 3 if args.sequence_wall_counts else 1100
        engine = MultiWaypointEngine(checkpoint, args.device, course, options=options, horizon=horizon)
        cfg = _course_config(engine.loaded.cfg, case["agents"], course, case["shape"])
        if options.nominal_glyph_size:
            cfg = replace(cfg, glyph_scale=1.)
        scene = _full_collision_scenario(course, cfg)
        start_certificate = certify_pose_path(scene.state0.repeat(2, 1), scene, cfg)
        end_certificate = certify_pose_path(scene.goal.repeat(2, 1), scene, cfg)
        row = dict(case, initial_pose_certificate=asdict(start_certificate),
                   goal_pose_certificate=asdict(end_certificate),
                   executable_feasibility="not_queried",
                   variant=("complete_exit_consistent" if options.respect_handoff_goal_region else
                            "complete_exit_swept" if options.check_support_alignment_sweep else
                            "complete_exit_screened" if options.validate_all_cspace_lookaheads else
                            "complete_exit_certified"),
                   options=asdict(options))
        row["horizon"] = horizon
        if not start_certificate.passed or not end_certificate.passed:
            row.update(result={"success": False, "steps": 0,
                               "reason": "invalid_initial_or_final_pose"})
        else:
            recorder = Recorder()
            engine.audit_callback = recorder
            handoffs = []
            for event in engine.events(case["agents"], case["shape"]):
                if event["type"] == "header":
                    row["header"] = event
                elif event["type"] == "frame" and event["waypoint_reached"]:
                    handoffs.append({"step": event["step"], "next_stage": event["waypoint_index"],
                                     "state": event["next_state"]})
                elif event["type"] == "done":
                    row["result"] = event
            path = args.output / (case["case_id"] + ".npz")
            recorder.save(path)
            trajectory = np.concatenate((np.stack(recorder.arrays["state"]),
                                         recorder.arrays["next_state"][-1][None]))
            certificate = certify_pose_path(torch.from_numpy(trajectory), recorder.last_scenario,
                                            replace(recorder.last_cfg, device="cpu"))
            row.update(
                handoffs=handoffs, first_rejection=recorder.first_hit,
                continuous_certificate=asdict(certificate),
                independent_resolved_pose_collisions=recorder.resolved_pose_collisions,
                minimum_resolved_transition_clearance_m=recorder.minimum_clearance,
                path_length_m=recorder.path_length, gbp_float_payload_bytes=recorder.payload_bytes,
                grasp_setup_payload_bytes=recorder.setup_bytes, wire_headers_included=False,
                analytic_plan_agent_step_fraction=recorder.analytic_active / recorder.agent_steps,
                trace=path.name, trace_sha256=sha256(path),
            )
            # A successful certified deployment is itself an executable witness.
            if row["result"]["success"] and certificate.passed:
                row["executable_feasibility"] = "certified_student_witness"
        row["elapsed_seconds"] = time.monotonic() - started
        record["results"].append(row)
        write_json(destination, record)
        print(json.dumps({"case": case["case_id"], **row["result"],
                          "seconds": row["elapsed_seconds"]}), flush=True)
    record["status"] = "complete"
    write_json(destination, record)


if __name__ == "__main__":
    main()
