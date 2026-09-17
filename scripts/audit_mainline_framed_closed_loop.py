#!/usr/bin/env python3
"""Closed-loop fixed-chart interface control with genuinely rotated physics.

This reruns the unchanged policy at every tick; it does not replay forces.
All scenes are rigid copies of the same parallel-wall course. It tests the
supplied coordinate adapter, NOT learned SE(2) equivariance or a mixed-normal
maze. Independent persistent world dynamics/SAT use float64; the identity
world is a mandatory precision/backend-matched control.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_local_skill_composition import Recorder, sha256, write_json
from audit_mainline_rigid_replay import FRAMES
from piano_movers.framed_world_physics import FramedWorldPhysics
from piano_movers.local_skill_validation import FAMILY_SEEDS, STATE_PROFILES, state_profile_course, validate_split_seeds
from piano_movers.multi_waypoint_demo import MultiWaypointEngine, SkillCompositionOptions
from piano_movers.rigid_scene_audit import certify_world_path
from piano_movers.single_slit_0720_eval import strict_final_mask


class FramedAuditEngine(MultiWaypointEngine):
    def __init__(self, *args, frame, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.options.continuous_contact_guard:
            raise ValueError("the framed audit requires a continuous guard")
        self.physics = FramedWorldPhysics(frame)

    def _contact_step(self, state, forces, scenario, cfg):
        return self.physics(state, forces, scenario, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("configs/p6_force_position_heading_control.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(FAMILY_SEEDS), default="development_diagnostics")
    parser.add_argument("--seed", type=int, default=81201)
    parser.add_argument("--shapes", nargs="+", default=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    parser.add_argument("--agents", nargs="+", type=int, default=[12])
    parser.add_argument("--profiles", nargs="+", choices=STATE_PROFILES, default=["entry_reference"])
    args = parser.parse_args()
    validate_split_seeds(args.split, [args.seed])
    for values in (args.shapes, args.agents, args.profiles):
        if not values or len(set(values)) != len(values):
            raise ValueError("case dimensions must be unique and nonempty")
    if any(s not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for s in args.shapes):
        raise ValueError("uppercase letter shapes required")
    if any(n not in range(8, 21) for n in args.agents):
        raise ValueError("agent counts must be 8-20")
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads(args.recipe.read_text())
    checkpoint = (root / recipe["checkpoint"]).resolve(strict=True)
    if sha256(checkpoint) != recipe["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint mismatch")
    options = SkillCompositionOptions(**recipe["composition_options"])
    torch.set_num_threads(1)
    catalog = [{"shape": s, "agents": n, "profile": p, "frame": name,
                "transform": asdict(frame), "course": asdict(state_profile_course(args.seed, "combined", p))}
               for s in args.shapes for n in args.agents for p in args.profiles for name, frame in FRAMES]
    args.output.mkdir(parents=True)
    write_json(args.output / "case_catalog.json", catalog)
    record = {
        "schema": "piano-movers.fixed-chart-closed-loop-audit.v1", "status": "running",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "recipe_snapshot": recipe, "recipe_sha256": sha256(args.recipe),
        "split": args.split, "layout_seed": args.seed, "requested_cases": len(catalog),
        "catalog_sha256": sha256(args.output / "case_catalog.json"),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in (
            Path(__file__), root / "scripts/audit_mainline_rigid_replay.py",
            root / "scripts/audit_local_skill_composition.py", *sorted((root / "src/piano_movers").glob("*.py")))},
        "policy_rerun": True, "fixed_chart_supplied": True, "physics_dtype": "float64",
        "policy_dtype": "float32", "backend_matched_control": "identity",
        "world_collision_checker": "independent_oriented_rectangles_recursive_SAT",
        "not_evidence_for": ["learned_rotation_equivariance", "changing_wall_normals",
                             "frame_change_memory", "independent_layout_sample_count"],
        "outcome_filtered": False, "results": [],
    }
    manifest = args.output / "manifest.json"
    write_json(manifest, record)
    transforms = dict(FRAMES)
    identity = {}
    started = time.monotonic()
    for case in catalog:
        frame = transforms[case["frame"]]
        course = state_profile_course(args.seed, "combined", case["profile"])
        key = f"{case['shape']}_n{case['agents']}_{case['profile']}"
        case_id = f"{key}_{case['frame']}"
        recorder = Recorder()
        engine = FramedAuditEngine(checkpoint, "cpu", course, options=options,
                                   frame=frame, audit_callback=recorder)
        row = dict(case, case_id=case_id)
        for event in engine.events(case["agents"], case["shape"]):
            if event["type"] in ("header", "done"):
                row["header" if event["type"] == "header" else "result"] = event
        trace_path = args.output / (case_id + ".npz")
        recorder.save(trace_path)
        physics = engine.physics
        world_path = np.stack(physics.states)
        world_path_file = args.output / (case_id + "_world.npz")
        np.savez_compressed(world_path_file, states=world_path, forces=np.stack(physics.forces),
                            hit=np.asarray(physics.hits))
        certificate = certify_world_path(world_path, physics.geometry)
        goal = frame.states(np.array([course.waypoints[-1]]))
        world_terminal = bool(strict_final_mask(torch.from_numpy(world_path[-1:]), torch.from_numpy(goal))[0])
        penetration = bool((physics.geometry.clearance(world_path) <= 0).any())
        row.update(continuous_world_certificate=asdict(certificate), world_final_mask=world_terminal,
                   independent_world_penetration=penetration,
                   trace=trace_path.name, trace_sha256=sha256(trace_path),
                   world_trace=world_path_file.name, world_trace_sha256=sha256(world_path_file),
                   gbp_float_payload_bytes=recorder.payload_bytes, grasp_setup_payload_bytes=recorder.setup_bytes,
                   wire_headers_included=False, path_length_m=recorder.path_length,
                   analytic_plan_agent_step_fraction=recorder.analytic_active / recorder.agent_steps)
        row["audited_success"] = row["result"]["success"] and world_terminal and certificate.passed and not penetration
        if case["frame"] == "identity":
            identity[key] = row
        control = identity[key]
        row["paired_identity"] = {"case_id": control["case_id"],
                                  "same_success": row["audited_success"] == control["audited_success"],
                                  "steps_delta": row["result"]["steps"] - control["result"]["steps"],
                                  "gbp_payload_bytes_delta": row["gbp_float_payload_bytes"] - control["gbp_float_payload_bytes"]}
        record["results"].append(row)
        write_json(manifest, record)
        print(json.dumps({"done": len(record["results"]), "total": len(catalog), "case": case_id,
                          "success": row["audited_success"], "steps": row["result"]["steps"],
                          "contacts": row["result"]["contact_events"]}), flush=True)
    record.update(status="complete", successes=sum(r["audited_success"] for r in record["results"]),
                  world_certificates_passed=sum(r["continuous_world_certificate"]["passed"] for r in record["results"]),
                  penetration_episodes=sum(r["independent_world_penetration"] for r in record["results"]),
                  elapsed_seconds=time.monotonic() - started)
    write_json(manifest, record)
    print(json.dumps({k: record[k] for k in ("status", "successes", "requested_cases",
                                            "world_certificates_passed", "penetration_episodes", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
