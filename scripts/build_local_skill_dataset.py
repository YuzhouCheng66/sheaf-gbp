#!/usr/bin/env python3
"""Generate certified local-skill demonstrations and model-state teacher queries.

Failures remain in the manifest. Training shards contain only fully executable
successful continuations. These are warm-start demonstrations plus queries of
the frozen p5 actor, not claims of a p6 on-policy DAgger iteration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.env import PianoBatch, collision_mask, collision_clearance_margin, wrap_angle
from piano_movers.local_skill_policy import local_skill_features
from piano_movers.local_skill_teacher import (
    LocalTeacherConfig,
    certify_pose_path,
    executable_teacher_step,
    plan_local_target,
)
from piano_movers.multi_waypoint_demo import (
    DEFAULT_COURSE,
    _course_config,
    _full_collision_scenario,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_feedback import SingleSlitContactFeedbackResultV1
from piano_movers.single_slit_contact_observation import (
    SingleSlitContactObservationContextV1,
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
)


TRAIN_SHAPES = tuple(chr(k) for k in range(ord("A"), ord("Z") + 1) if chr(k) not in ("S", "Z"))


def generate_case(
    state, goal, scene, cfg, context, orientation, options, horizon, *, geometry_segments=False
):
    path = None
    planning_attempts = []
    for resolution in (options.theta_bins, options.theta_bins * 2):
        proposed = plan_local_target(
            state,
            goal,
            scene,
            cfg,
            orientation_required=orientation,
            options=replace(options, theta_bins=resolution),
        )
        planning_attempts.append(
            {
                "theta_bins": resolution,
                "status": proposed.status,
                "expansions": proposed.expansions,
                "lazy_replans": proposed.lazy_replans,
            }
        )
        if proposed.poses is not None:
            path = proposed
            break
        if proposed.status in (
            "initial_certificate_margin_violation",
            "start_outside_planning_reserve",
        ):
            break
    report = {
        "planning_attempts": planning_attempts,
        "success": False,
        "terminal_position_tolerance_m": 0.020,
        "terminal_angle_tolerance_rad": 0.030,
        "terminal_speed_tolerance_mps": 0.020,
        "terminal_omega_tolerance_radps": 0.030,
        "terminal_dwell_steps": 6,
    }
    if path is None:
        return report, None
    planned = torch.as_tensor(path.poses, dtype=state.dtype)
    report["geometric_certificate"] = asdict(path.certificate)
    report["planning_margin_m"] = path.planning_margin_m
    features, labels, forces, states = [], [], [], [state[0].clone()]
    modes = []
    dwell = stagnant = 0
    for _ in range(horizon):
        xy = torch.linalg.vector_norm(state[0, :2] - goal[:2])
        close = bool(xy < 0.020) and (
            not orientation or bool(wrap_angle(state[0, 2] - goal[2]).abs() < 0.030)
        )
        rest = bool(torch.linalg.vector_norm(state[0, 3:5]) < 0.02) and bool(
            state[0, 5].abs() < 0.03
        )
        dwell = dwell + 1 if close and rest else 0
        if dwell >= 6:
            report["success"] = True
            break
        f, nxt, mode = executable_teacher_step(
            state, state[:, :3].clone() if close else planned, scene, cfg, options
        )
        if f is None:
            report["execution_failure"] = mode
            break
        if bool(collision_mask(nxt, scene, cfg)[0]):
            raise RuntimeError("independent pose SAT found a teacher collision")
        features.append(
            local_skill_features(
                state,
                scene,
                cfg,
                context,
                goal=goal,
                orientation_required=orientation,
                geometry_segments=geometry_segments,
            )[0]
        )
        s, c = state[0, 2].sin(), state[0, 2].cos()
        body_twist = torch.stack(
            (c * nxt[0, 3] + s * nxt[0, 4], -s * nxt[0, 3] + c * nxt[0, 4], nxt[0, 5])
        )
        labels.append(body_twist / state.new_tensor([0.65, 0.65, 1.4]))
        forces.append(f[0])
        modes.append(mode)
        clearance = collision_clearance_margin(nxt, scene, cfg)
        result = SingleSlitContactFeedbackResultV1(
            nxt, nxt, clearance, torch.zeros(1, dtype=torch.bool), clearance
        )
        context = single_slit_contact_observation_context_from_result_v1(result, f, cfg)
        motion = (
            torch.linalg.vector_norm(nxt[0, :2] - state[0, :2])
            + 0.35 * wrap_angle(nxt[0, 2] - state[0, 2]).abs()
        )
        stagnant = stagnant + 1 if float(motion) < 1.0e-6 and not close else 0
        state = nxt
        states.append(state[0].clone())
        if stagnant >= 20:
            report["execution_failure"] = "teacher_stall_20_steps"
            break
    trajectory = torch.stack(states)
    if len(states) > 1:
        certificate = certify_pose_path(trajectory, scene, cfg, options)
        report["executed_certificate"] = asdict(certificate)
        if not certificate.passed:
            raise RuntimeError("teacher trajectory failed recursive admission")
    report.update(
        {
            "steps": len(forces),
            "position_error_m": float(torch.linalg.vector_norm(state[0, :2] - goal[:2])),
            "angle_error_rad": float(wrap_angle(state[0, 2] - goal[2]).abs()),
            "final_speed_mps": float(torch.linalg.vector_norm(state[0, 3:5])),
            "final_omega_radps": float(state[0, 5].abs()),
            "brake_steps": modes.count("brake"),
            "rejoin_steps": modes.count("rejoin"),
        }
    )
    if not report["success"]:
        report.setdefault("execution_failure", "horizon")
        return report, None
    return report, {
        "features": torch.stack(features).numpy(),
        "target_twist": torch.stack(labels).numpy(),
        "forces": torch.stack(forces).numpy(),
        "states": trajectory.numpy(),
        "path": path.poses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-manifest", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--shapes", nargs="+", default=TRAIN_SHAPES)
    parser.add_argument("--cases-per-shape-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--x-step", type=float, default=0.0125)
    parser.add_argument("--theta-bins", type=int, default=360)
    parser.add_argument("--factorial-entry", action="store_true")
    args = parser.parse_args()
    if set(args.shapes) - set(TRAIN_SHAPES) or any(n % 2 or n < 8 or n > 20 for n in args.counts):
        raise ValueError("warm-start training preserves S/Z and odd-agent holdouts")
    torch.set_num_threads(1)
    parent = load_single_slit_checkpoint(args.checkpoint, device="cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    options = LocalTeacherConfig(x_step=args.x_step, theta_bins=args.theta_bins)
    manifest = {
        "schema": "piano-movers.p6-local-training-set.v1",
        "status": "running",
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "seed": args.seed,
        "source_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path("src/piano_movers/local_skill_teacher.py"),
                Path("src/piano_movers/local_skill_policy.py"),
            ]
        },
        "options": asdict(options),
        "heldout_shapes": ["S", "Z"],
        "heldout_counts": [9, 11, 13, 15, 17, 19],
        "training_shapes": list(args.shapes),
        "training_counts": args.counts,
        "split": "train",
        "fresh_test_episodes_consumed": False,
        "factorial_entry": args.factorial_entry,
        "results": [],
    }
    write_json(manifest_path, manifest)
    rng = np.random.default_rng(args.seed)

    def consume(case_id, state, goal, scene, cfg, context, orientation, origin):
        started = time.monotonic()
        meta = {
            "case_id": case_id,
            "shape": cfg.payload_shape,
            "agents": cfg.n_agents,
            "env_cfg": asdict(cfg),
            "start_state": state[0].tolist(),
            "goal": goal.tolist(),
            "walls": {
                "x": scene.wall_x[0].tolist(),
                "gap_y": scene.gap_y[0].tolist(),
                "gap_half": scene.gap_half[0].tolist(),
            },
            "orientation_required": orientation,
            "origin": origin,
        }
        if bool(collision_mask(state, scene, cfg)[0]):
            report, data = {"success": False, "execution_failure": "initial_state_collision"}, None
        else:
            report, data = generate_case(
                state, goal, scene, cfg, context, orientation, options, args.horizon
            )
        meta.update(report)
        meta["elapsed_seconds"] = time.monotonic() - started
        if data is not None:
            path = args.output / f"{case_id}.npz"
            np.savez_compressed(path, **data)
            meta.update({"artifact": path.name, "artifact_sha256": sha256(path)})
        manifest["results"].append(meta)
        write_json(manifest_path, manifest)
        print(
            json.dumps(
                {k: meta[k] for k in ("case_id", "success", "elapsed_seconds")}
                | {
                    "steps": meta.get("steps"),
                    "failure": meta.get("execution_failure"),
                    "planning": meta.get("planning_attempts"),
                }
            ),
            flush=True,
        )

    # Direct teacher queries of model-induced states retain provenance and
    # contact feedback from the actual frozen-actor rollout, without making
    # inspected development episodes part of the fresh test set.
    if args.rollout_manifest is not None:
        source = json.loads(args.rollout_manifest.read_text())
        if source["checkpoint_sha256"] != parent.checkpoint_sha256:
            raise ValueError("actor lineage mismatch")
        manifest["actor_rollout_manifest_sha256"] = sha256(args.rollout_manifest)
        for row in source["results"]:
            shape, count = row["shape"], row["agents"]
            if shape not in args.shapes or count not in args.counts or row["result"]["success"]:
                continue
            artifact = args.rollout_manifest.parent / row["trace"]
            if sha256(artifact) != row["trace_sha256"]:
                raise ValueError("actor trace SHA mismatch")
            trace = np.load(artifact)
            ticks = [len(trace["state"]) - 1]
            if np.any(trace["hit"]):
                hit = int(np.flatnonzero(trace["hit"])[0])
                ticks.extend([hit, max(0, hit - 12)])
            seen = set()
            for tick in sorted(set(ticks)):
                state = torch.from_numpy(trace["state"][tick].copy()).view(1, 6)
                key = tuple(np.round(state.numpy().flatten(), 4))
                if key in seen:
                    continue
                seen.add(key)
                stage = int(trace["waypoint_index"][tick])
                goal = state.new_tensor(DEFAULT_COURSE.waypoints[stage])
                cfg = _course_config(parent.cfg, count, DEFAULT_COURSE, shape)
                scene = replace(
                    _full_collision_scenario(DEFAULT_COURSE, cfg),
                    state0=state,
                    source=state[:, :3],
                    goal=goal.view(1, 3),
                )
                old = torch.from_numpy(trace["observation"][tick].copy()).unsqueeze(0)
                context = SingleSlitContactObservationContextV1(
                    old[:, 0, 22].bool(), old[..., 23:25], old[..., 25:27]
                )
                consume(
                    f"actor_{shape}_{count}_{tick}",
                    state,
                    goal,
                    scene,
                    cfg,
                    context,
                    stage == 3,
                    {
                        "kind": "p5_model_induced_query_with_teacher_continuation",
                        "trace_sha256": row["trace_sha256"],
                        "tick": tick,
                    },
                )
    for shape in args.shapes:
        for count in args.counts:
            for sample in range(args.cases_per_shape_count):
                case_id = f"seed{args.seed}_{shape}_{count}_{sample}"
                cfg = _course_config(parent.cfg, count, DEFAULT_COURSE, shape)
                size = float(rng.uniform(0.92, 1.0)) if sample % 2 else 1.0
                if shape == "T":
                    cfg = replace(
                        cfg,
                        t_total_width_m=cfg.t_total_width_m * size,
                        t_total_height_m=cfg.t_total_height_m * size,
                        t_stem_width_m=cfg.t_stem_width_m * size,
                        t_cap_thickness_m=cfg.t_cap_thickness_m * size,
                    )
                else:
                    cfg = replace(cfg, glyph_scale=cfg.glyph_scale * size)
                gap = float(rng.uniform(-0.24, 0.24))
                aperture = float(rng.choice([0.72, 0.82, 1.05, 1.6], p=[0.4, 0.2, 0.2, 0.2]))
                theta = float(rng.uniform(-math.pi, math.pi)) if sample % 2 else 0.0
                state = torch.tensor(
                    [
                        [
                            -float(rng.uniform(1.1, 1.45)),
                            float(rng.uniform(-0.2, 0.2)),
                            theta,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ]
                )
                goal = state.new_tensor(
                    [
                        float(rng.uniform(0.65, 1.3)),
                        gap + float(rng.uniform(-0.20, 0.20)),
                        float(rng.uniform(-math.pi, math.pi)),
                    ]
                )
                orientation = bool(sample % 3 == 0)
                if args.factorial_entry:
                    # Cross entry heading, goal semantics and travel direction
                    # independently. The initial four-case pilot accidentally
                    # omitted zero-heading, left-to-right, position-only starts.
                    state[0, 2] = (
                        0.0 if (sample // 4) % 2 == 0 else float(rng.uniform(-math.pi, math.pi))
                    )
                    orientation = bool((sample // 2) % 2)
                    if sample % 2:
                        state[0, 0] *= -1
                        goal[0] *= -1
                elif sample % 4 == 3:
                    # Local terminal maneuver, not a fixed wall-crossing
                    # script: random nearby target and independent heading.
                    state[0, 0] = float(rng.uniform(1.10, 1.35))
                    goal[:2] = state[0, :2] + state.new_tensor(rng.uniform(-0.20, 0.20, size=2))
                    orientation = True
                elif sample % 4 == 2:
                    # Half-turn of a whole scene produces a reverse-direction
                    # family without adding direction labels to the policy.
                    state[0, :2] *= -1
                    state[0, 2] = wrap_angle(state[0, 2] + math.pi)
                    goal[:2] *= -1
                    goal[2] = wrap_angle(goal[2] + math.pi)
                    gap = -gap
                scene = PianoBatch(
                    state0=state,
                    source=state[:, :3],
                    goal=goal.view(1, 3),
                    wall_x=state.new_tensor([[0.0]]),
                    gap_y=state.new_tensor([[gap]]),
                    gap_half=state.new_tensor([[aperture / 2.0]]),
                )
                context = initial_single_slit_contact_observation_context_v1(state, cfg)
                consume(
                    case_id,
                    state,
                    goal,
                    scene,
                    cfg,
                    context,
                    orientation,
                    {"kind": "procedural_training_layout", "seed": args.seed, "sample": sample},
                )
    manifest["status"] = "complete"
    manifest["successes"] = sum(row["success"] for row in manifest["results"])
    manifest["attempts"] = len(manifest["results"])
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
