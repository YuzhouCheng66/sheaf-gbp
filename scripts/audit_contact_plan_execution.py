#!/usr/bin/env python3
"""Separate an unsafe local twist from a GBP/actuation tracking discrepancy.

Read-only offline counterfactuals on authenticated contact DAgger frames.
An ideal next twist is not asserted force-feasible; its path certificate is
only a geometric diagnostic, never a deployment or success claim.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.config import EnvConfig
from piano_movers.env import wrap_angle
from piano_movers.local_skill_physics import certified_contact_step
from piano_movers.local_skill_teacher import certify_pose_path
from piano_movers.multi_waypoint_demo import WaypointCourse, Slit, SkillCompositionOptions, _full_collision_scenario, apply_solver_controls
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_local_planner import LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS, LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS, local_se2_terminal_attractor_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-local-decisions", action="store_true", help="Read-only Python instrumentation of original branch predicates")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.dataset.read_text())
    audit = json.loads(args.source_audit.read_text())
    rollouts = [json.loads(p.read_text()) for p in args.rollouts]
    if data["status"] != "complete" or audit["status"] != "complete" or audit["dataset_sha256"] != sha256(args.dataset):
        raise ValueError("completed matching teacher and counterfactual manifests required")
    if not set(audit["rollouts_sha256"]) <= {sha256(p) for p in args.rollouts}:
        raise ValueError("actor manifest mismatch")
    loaded = load_single_slit_checkpoint(Path(rollouts[0]["checkpoint"]), device="cpu")
    if loaded.checkpoint_sha256 != audit["base_checkpoint_sha256"]:
        raise ValueError("policy identity changed")
    policy = loaded.policy.eval()
    torch.set_num_threads(1)
    original = {r["trace_sha256"]: r for d in rollouts for r in d["results"]}
    queries = {}
    for r in audit["results"]:
        queries.setdefault(r["teacher_case"], []).append(r)
    results = []
    for row in data["results"]:
        if row["case_id"] not in queries:
            continue
        artifact = args.dataset.parent / row["artifact"]
        if sha256(artifact) != row["artifact_sha256"]:
            raise ValueError("teacher artifact changed")
        cfg = EnvConfig(**row["env_cfg"])
        policy.bind_environment(cfg)
        apply_solver_controls(policy, SkillCompositionOptions(**row["actor_composition_options"]))
        h = original[row["actor_trace_sha256"]]["header"]
        course = WaypointCourse(tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
            tuple(tuple(w) for w in h["waypoints"]), h["world"]["x"], h["world"]["y"],
            tuple(h["source_velocity"]), h["morphology_scale"])
        scene = _full_collision_scenario(course, cfg)
        with np.load(artifact) as arrays:
            for query in queries[row["case_id"]]:
                k = query["prefix"]
                state, obs, hidden = (torch.from_numpy(arrays[name][k:k+1].copy()) for name in
                                       ("states", "observation", "private_hidden"))
                decisions = {}

                def trace_decisions(frame, event, arg):
                    if event == "return" and frame.f_code is local_se2_terminal_attractor_plan.__code__:
                        for name in ("shape_rotation_safe", "shape_aligned", "current_shape_contained",
                                     "contained_alignment_safe", "shape_safe_x", "center_x", "center_y",
                                     "target_x", "target_y", "target_theta", "cspace_active", "terminal_active"):
                            value = frame.f_locals.get(name)
                            if isinstance(value, torch.Tensor):
                                decisions[name] = value.detach().clone()

                previous_profile = sys.getprofile()
                try:
                    if args.trace_local_decisions:
                        sys.setprofile(trace_decisions)
                    output = policy(obs, policy.local_se2_analytic_reference_edges, hidden)
                finally:
                    if args.trace_local_decisions:
                        sys.setprofile(previous_profile)
                plan = output.aux["local_se2_plan_used"][0]
                sine, cosine = state[0, 2].sin(), state[0, 2].cos()
                body_v = plan[:, 3:5] * LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
                world_twist = torch.stack((cosine * body_v[:, 0] - sine * body_v[:, 1],
                    sine * body_v[:, 0] + cosine * body_v[:, 1],
                    plan[:, 5] * LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS), -1)
                # Record disagreement explicitly; a mean is not a consensus proof.
                ideal_twist = world_twist.mean(0)
                ideal = state.clone()
                ideal[0, 3:] = ideal_twist
                ideal[0, :3] += cfg.dt * ideal_twist
                ideal[0, 2] = wrap_angle(ideal[0, 2])
                cert = certify_pose_path(torch.cat((state, ideal)), scene, cfg)
                # Counterfactual only: keep the current lateral position and
                # heading while applying the already-proposed longitudinal
                # speed. This tests the crossing/alignment interface, not a
                # new policy or an executable force solution.
                cross = state.clone()
                cross[0, 3:] = 0.
                cross[0, 3] = ideal_twist[0]
                cross[0, 0] += cfg.dt * ideal_twist[0]
                cross_cert = certify_pose_path(torch.cat((state, cross)), scene, cfg)
                transition, _ = certified_contact_step(state, output.forces, scene, cfg)
                results.append({"teacher_case": row["case_id"], "prefix": k, "agents": cfg.n_agents,
                    "source_continuation_success": query["source_success"],
                    "ideal_mean_twist": ideal_twist.tolist(),
                    "maximum_node_twist_disagreement": float((world_twist - ideal_twist).abs().max()),
                    "actual_attempted_twist": transition.attempted_state[0, 3:].tolist(),
                    "maximum_twist_tracking_error": float((transition.attempted_state[0, 3:] - ideal_twist).abs().max()),
                    "force_saturated_nodes": int((output.forces.norm(dim=-1) >= cfg.max_force * (1. - 1e-5)).sum()),
                    "ideal_geometric_certificate": asdict(cert), "actual_attempt_rejected": bool(transition.attempted_hit[0]),
                    "longitudinal_only_counterfactual_certificate": asdict(cross_cert),
                    "local_decisions": {name: value.tolist() for name, value in decisions.items()}})
    report = {"status": "complete", "scope": "offline_local_step_diagnosis_not_success_rate",
        "checkpoint_sha256": loaded.checkpoint_sha256, "dataset_sha256": sha256(args.dataset),
        "counterfactual_sha256": sha256(args.source_audit), "source_sha256": sha256(Path(__file__)), "results": results}
    report["groups"] = [{"source_continuation_success": success, "frames": len(rows),
        "ideal_steps_not_certified": sum(not r["ideal_geometric_certificate"]["passed"] for r in rows),
        "actual_steps_rejected": sum(r["actual_attempt_rejected"] for r in rows),
        "max_twist_disagreement": max(r["maximum_node_twist_disagreement"] for r in rows),
        "max_twist_tracking_error": max(r["maximum_twist_tracking_error"] for r in rows),
        "saturated_frames": sum(r["force_saturated_nodes"] > 0 for r in rows)}
        for success in (False, True) if (rows := [r for r in results if r["source_continuation_success"] == success])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report["groups"]))


if __name__ == "__main__":
    main()
