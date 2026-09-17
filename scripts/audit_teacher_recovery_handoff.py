#!/usr/bin/env python3
"""When can the ORIGINAL skill resume from a real teacher recovery prefix?

This is training-time counterfactual evaluation, not a deployment teacher.
Each source continuation starts with the actual teacher-prefix observation
and causal original-model hidden state; it does not reset memory or position.
Failure means not solved within the declared horizon, not infeasibility.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from collect_mainline_skill_dagger import context_from_original_observation
from piano_movers.config import EnvConfig
from piano_movers.local_skill_physics import certified_contact_step
from piano_movers.local_skill_teacher import certify_pose_path
from piano_movers.multi_waypoint_demo import (
    WaypointCourse, Slit, SkillCompositionOptions, _full_collision_scenario,
    _composition_observation, _intermediate_reached, _body_cleared_wall, apply_solver_controls,
)
from piano_movers.payload_shapes import payload_rectangles_for_config
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint, strict_final_mask, STRICT_STABLE_STEPS
from piano_movers.single_slit_contact_observation import single_slit_contact_observation_context_from_result_v1


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--all-contact-prefixes", action="store_true",
                        help="Evaluate every active-memory teacher frame, including recoverable student equilibria")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.dataset.read_text())
    rollout_records = [json.loads(p.read_text()) for p in args.rollouts]
    rollout_hashes = [sha256(p) for p in args.rollouts]
    required = ([data["rollout_manifest_sha256"]] if data["rollout_manifest_sha256"] is not None else
                [s["rollout_manifest_sha256"] for s in data["source_manifests"]])
    if data["status"] != "complete" or not set(required) <= set(rollout_hashes):
        raise ValueError("teacher/actor provenance mismatch")
    if any(r["status"] != "complete" or r["split"] != "development_diagnostics" for r in rollout_records) or args.horizon < 1:
        raise ValueError("positive horizon and development data required")
    rollouts = rollout_records[0]
    loaded = load_single_slit_checkpoint(Path(rollouts["checkpoint"]), device="cpu")
    actors = data.get("actor_checkpoint_sha256s") or [data["actor_checkpoint_sha256"]]
    if actors != [loaded.checkpoint_sha256] or any(r["checkpoint_sha256"] != loaded.checkpoint_sha256 for r in rollout_records):
        raise ValueError("base checkpoint mismatch")
    policy = loaded.policy.eval()
    torch.set_num_threads(1)
    originals = {r["trace_sha256"]: r for source in rollout_records for r in source["results"] if "trace_sha256" in r}
    selected = [r for r in data["results"] if r["admitted_for_training"] and
                (args.all_contact_prefixes or r["actor_learned_plan_authority_nodes"] == 0)]
    args.output.mkdir(parents=True)
    record = {"status": "running", "schema": "piano-movers.teacher-prefix-source-continuation.v1",
        "dataset_sha256": sha256(args.dataset), "rollouts_sha256": rollout_hashes,
        "base_checkpoint_sha256": loaded.checkpoint_sha256, "horizon": args.horizon,
        "prefix_selection": ("all_active_contact_memory_frames" if args.all_contact_prefixes else
                             "0,4,8,16,32,last_action; fixed before results"),
        "teacher_queries": len(selected), "source_sha256": sha256(Path(__file__)),
        "training_time_only": True, "results": []}
    manifest = args.output / "manifest.json"
    write_json(manifest, record)
    for row in selected:
        artifact = args.dataset.parent / row["artifact"]
        if sha256(artifact) != row["artifact_sha256"]:
            raise ValueError("teacher artifact changed")
        with np.load(artifact) as archive:
            teacher = {k: archive[k].copy() for k in ("states", "observation", "private_hidden", "target_force")}
        h = originals[row["actor_trace_sha256"]]["header"]
        course = WaypointCourse(tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
                                tuple(tuple(w) for w in h["waypoints"]), h["world"]["x"], h["world"]["y"],
                                tuple(h["source_velocity"]), h["morphology_scale"])
        cfg = EnvConfig(**row["env_cfg"])
        options = SkillCompositionOptions(**row["actor_composition_options"])
        scene = _full_collision_scenario(course, cfg)
        stage = row["active_stage_for_audit_only"]
        goal = tuple(row["goal"])
        wall = course.walls[min(stage, len(course.walls) - 1)]
        body = payload_rectangles_for_config(cfg, "cpu")
        prefixes = sorted({min(t, len(teacher["target_force"]) - 1) for t in (0, 4, 8, 16, 32, len(teacher["target_force"]) - 1)})
        if args.all_contact_prefixes:
            policy.bind_environment(cfg)
            obs_array = torch.from_numpy(teacher["observation"])
            recovery = policy.split_persistent_recovery_hidden(torch.from_numpy(teacher["private_hidden"]), obs_array)[1]
            active_memory = recovery.any(-1).any(-1) | obs_array[..., 22].bool().any(-1)
            prefixes = torch.where(active_memory)[0].tolist()
        for prefix in prefixes:
            policy.bind_environment(cfg)
            apply_solver_controls(policy, options)
            state = torch.from_numpy(teacher["states"][prefix:prefix + 1])
            hidden = torch.from_numpy(teacher["private_hidden"][prefix:prefix + 1])
            observation = torch.from_numpy(teacher["observation"][prefix:prefix + 1])
            context = context_from_original_observation(observation)
            path, forces, hits, dwell = [state[0].numpy().copy()], [], [], 0
            success = False
            for tick in range(args.horizon):
                obs = _composition_observation(course, state, stage, cfg, context,
                    retain_nearby_wall=options.retain_nearby_wall, retain_previous_wall_only=options.retain_previous_wall_only)
                if tick == 0:
                    torch.testing.assert_close(obs, observation, atol=2e-6, rtol=2e-6)
                output = policy(obs, policy.local_se2_analytic_reference_edges, hidden)
                transition, _ = certified_contact_step(state, output.forces, scene, cfg)
                state = transition.resolved_state
                hidden = output.hidden
                context = single_slit_contact_observation_context_from_result_v1(transition, output.forces, cfg)
                path.append(state[0].numpy().copy())
                forces.append(output.forces[0].numpy().copy())
                hits.append(bool(transition.attempted_hit[0]))
                if cfg.goal_orientation_required:
                    dwell = dwell + 1 if not hits[-1] and bool(strict_final_mask(state, state.new_tensor([goal]))[0]) else 0
                    success = dwell >= STRICT_STABLE_STEPS
                else:
                    success = (_intermediate_reached(state, goal) and
                               (not options.require_body_clearance or _body_cleared_wall(state, body, wall, cfg)))
                if success:
                    break
            path_array = np.stack(path)
            cert = certify_pose_path(torch.from_numpy(path_array), scene, cfg)
            trace = args.output / f"{row['case_id']}_prefix{prefix}.npz"
            np.savez_compressed(trace, states=path_array, forces=np.stack(forces), hits=np.array(hits))
            result = {"teacher_case": row["case_id"], "teacher_artifact_sha256": row["artifact_sha256"],
                "shape": row["shape"], "agents": row["agents"], "prefix": prefix,
                "teacher_total_steps": len(teacher["target_force"]), "source_success": bool(success and cert.passed),
                "source_steps": len(forces), "source_contact_attempts": sum(hits),
                "continuous_certificate": asdict(cert), "trace": trace.name, "trace_sha256": sha256(trace)}
            record["results"].append(result)
            write_json(manifest, record)
            print(json.dumps({k: result[k] for k in ("teacher_case", "prefix", "source_success", "source_contact_attempts")}), flush=True)
    record["status"] = "complete"
    write_json(manifest, record)


if __name__ == "__main__":
    main()
