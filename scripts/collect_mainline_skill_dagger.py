#!/usr/bin/env python3
"""Certified recovery labels on authentic original-architecture student states.

This collector retains the original 27-D sensor, frozen private recurrence and
eight-coordinate plan interface. Only the first six plan coordinates are
supervised: no ordered path-progress target is introduced. A teacher's full
continuation must also meet the student's declared body-clear handoff contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from build_local_skill_dataset import generate_case
from piano_movers.env import collision_clearance_margin
from piano_movers.local_skill_teacher import LocalTeacherConfig
from piano_movers.multi_waypoint_demo import (
    Slit, WaypointCourse, SkillCompositionOptions, _body_cleared_wall,
    _composition_observation, _course_config, _full_collision_scenario,
    INTERMEDIATE_POSITION_TOLERANCE_M,
    apply_solver_controls,
)
from piano_movers.payload_shapes import payload_rectangles_for_config
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_feedback import SingleSlitContactFeedbackResultV1
from piano_movers.single_slit_contact_observation import (
    SingleSlitContactObservationContextV1,
    single_slit_contact_observation_context_from_result_v1,
)
from piano_movers.single_slit_local_planner import local_se2_plan_target
from piano_movers.single_slit_local_planner import LOCAL_SE2_TERMINAL_WALL_RESERVE_M


def context_from_original_observation(obs):
    return SingleSlitContactObservationContextV1(
        obs[:, 0, 22].bool(), obs[..., 23:25], obs[..., 25:27],
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="coherent_handoff")
    parser.add_argument("--queries-per-episode", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete" or source["split"] != "development_diagnostics":
        raise ValueError("requires a complete original-policy development audit")
    parent = load_single_slit_checkpoint(args.parent, device="cpu")
    if source["checkpoint_sha256"] != parent.checkpoint_sha256:
        raise ValueError("student checkpoint lineage mismatch")
    extension = source.get("contact_residual")
    if extension is not None:
        from piano_movers.contact_residual_control import load_contact_residual
        extension_path = Path(extension["path"])
        if sha256(extension_path) != extension["sha256"]:
            raise ValueError("actor contact-residual checkpoint changed")
        load_contact_residual(extension_path, parent.checkpoint_sha256)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    record = {
        "schema": "piano-movers.original-architecture-dagger.v1",
        "status": "running", "split": "train_development_dagger",
        "parent_checkpoint_sha256": parent.metadata.get("p6_mainline_training", {}).get(
            "parent_checkpoint_sha256", parent.checkpoint_sha256,
        ),
        "actor_checkpoint_sha256": parent.checkpoint_sha256,
        "actor_contact_residual": extension,
        "rollout_manifest_sha256": sha256(args.rollouts),
        "source_sha256": {str(p): sha256(p) for p in (
            Path(__file__), Path("scripts/build_local_skill_dataset.py"),
            Path("src/piano_movers/multi_waypoint_demo.py"),
            Path("src/piano_movers/models.py"),
            Path("src/piano_movers/local_skill_teacher.py"),
        )},
        "observation_dim": 27, "plan_dim": 8,
        "supervised_plan_coordinates": [0, 1, 2, 3, 4, 5],
        "heldout_shapes": ["S", "Z"], "heldout_counts": [9, 11, 13, 15, 17, 19],
        "variant": args.variant, "results": [], "skipped_duplicate_queries": 0,
        "unqueried_episodes": [],
    }
    write_json(destination, record)
    for episode in source["results"]:
        if episode["variant"] != args.variant or episode["result"]["success"]:
            continue
        if episode["shape"] in ("S", "Z") or episode["agents"] % 2:
            continue
        if "trace" not in episode:
            record["unqueried_episodes"].append({
                "case_id": episode.get("case_id"), "reason": "no_authentic_state_trace",
                "rollout_result": episode["result"],
            })
            write_json(destination, record)
            continue
        path = args.rollouts.parent / episode["trace"]
        if sha256(path) != episode["trace_sha256"]:
            raise ValueError("student trace hash mismatch")
        data = np.load(path)
        header = episode["header"]
        course = WaypointCourse(
            tuple(header["source"]), tuple(Slit(**s) for s in header["walls"]),
            tuple(tuple(g) for g in header["waypoints"]),
            header["world"]["x"], header["world"]["y"],
            tuple(header.get("source_velocity", (0., 0., 0.))),
            float(header.get("morphology_scale", 1.)),
        )
        composition = SkillCompositionOptions(**episode["options"])
        cfg = _course_config(parent.cfg, episode["agents"], course, episode["shape"])
        if composition.nominal_glyph_size:
            cfg = replace(cfg, glyph_scale=1.)
        if composition.complete_corridor_exit:
            cfg = replace(cfg, complete_local_corridor_exit=True)
        cfg = replace(cfg, local_cspace_collision_substeps=composition.planning_substeps,
                      local_corridor_goal_heading_search=composition.resolve_position_goal_heading,
                      local_cspace_validate_all_lookaheads=composition.validate_all_cspace_lookaheads,
                      local_support_check_alignment_sweep=composition.check_support_alignment_sweep,
                      local_corridor_exit_goal_radius_m=(INTERMEDIATE_POSITION_TOLERANCE_M
                                                         if composition.respect_handoff_goal_region else 0.))
        scene = _full_collision_scenario(course, cfg)
        ticks = list(np.linspace(0, len(data["state"]) - 1, args.queries_per_episode, dtype=int))
        hits = np.flatnonzero(data["hit"])
        if len(hits):
            ticks.extend([max(0, int(hits[0]) - 10), min(len(data["state"]) - 1, int(hits[0]) + 1)])
        seen = set()
        for tick in sorted(set(ticks)):
            stage = int(data["waypoint_index"][tick])
            state = torch.from_numpy(data["state"][tick].copy())[None]
            key = (stage, *np.round(state.numpy().ravel(), 3).tolist())
            if key in seen:
                record["skipped_duplicate_queries"] += 1
                continue
            seen.add(key)
            goal = state.new_tensor(course.waypoints[stage])
            active_wall = course.walls[min(stage, len(course.walls) - 1)]
            oriented = stage == len(course.waypoints) - 1 or not composition.position_only_control
            cfg = replace(cfg, goal_x=float(goal[0]) - active_wall.x,
                          goal_y=float(goal[1]), goal_theta=float(goal[2]),
                          goal_orientation_required=oriented)
            parent.policy.bind_environment(cfg)
            apply_solver_controls(parent.policy, composition)
            observation = torch.from_numpy(data["observation"][tick].copy())[None]
            hidden = torch.from_numpy(data["private_hidden"][tick].copy())[None]
            context = context_from_original_observation(observation)

            def sense(q, feedback):
                return _composition_observation(
                    course, q, stage, cfg, feedback,
                    retain_nearby_wall=composition.retain_nearby_wall,
                    retain_previous_wall_only=composition.retain_previous_wall_only,
                )

            torch.testing.assert_close(sense(state, context), observation, atol=2e-6, rtol=2e-6)
            options = LocalTeacherConfig(x_step=.0125)
            if stage < len(course.waypoints) - 1:
                options = replace(options, position_tolerance_m=.125, execution_position_tolerance_m=.14)
                if composition.require_body_clearance:
                    options = replace(options, terminal_halfspace=(
                        1., 0., active_wall.x + cfg.wall_slab + LOCAL_SE2_TERMINAL_WALL_RESERVE_M,
                    ))
            report, teacher = generate_case(
                state, goal, scene, cfg, context, oriented, options, 512,
            )
            case_id = (f"{args.variant}_{episode['shape']}_{episode['agents']}_"
                       f"{episode['trace_sha256'][:12]}_q{tick}")
            row = {
                "case_id": case_id, "shape": episode["shape"], "agents": episode["agents"],
                "actor_trace_sha256": episode["trace_sha256"], "actor_tick": int(tick),
                "rollout_case_id": episode.get("case_id"),
                "rollout_family": episode.get("family"), "rollout_seed": episode.get("seed"),
                "rollout_state_profile": episode.get("profile"),
                "active_stage_for_audit_only": stage, "env_cfg": asdict(cfg),
                "actor_composition_options": asdict(composition),
                "start_state": state[0].tolist(), "goal": goal.tolist(),
                "teacher_report": report, "admitted_for_training": False,
                "actor_learned_plan_authority_nodes": int(
                    (~data["analytic_active"][tick].astype(bool)).sum()
                ),
                "actor_contact_residual_present": extension is not None,
            }
            if teacher is not None:
                end = torch.from_numpy(teacher["states"][-1:])
                handoff = (stage == len(course.waypoints) - 1
                           or not composition.require_body_clearance
                           or _body_cleared_wall(end, payload_rectangles_for_config(cfg, "cpu"), active_wall, cfg))
                row["teacher_satisfies_handoff"] = bool(handoff)
                row["admitted_for_training"] = bool(handoff)
                observations, inputs, targets, private = [], [], [], []
                for index, force in enumerate(teacher["forces"]):
                    q = torch.from_numpy(teacher["states"][index:index + 1])
                    nxt = torch.from_numpy(teacher["states"][index + 1:index + 2])
                    obs = sense(q, context)
                    private.append(hidden[0].clone().numpy())
                    planner_input, hidden = parent.policy.local_se2_base_planner_inputs_and_next_private_hidden(obs, hidden)
                    observations.append(obs[0].numpy())
                    inputs.append(planner_input[0].numpy())
                    targets.append(local_se2_plan_target(q, SimpleNamespace(
                        next_state=nxt[0], semantic8=torch.zeros(8),
                    )).numpy())
                    clearance = collision_clearance_margin(nxt, scene, cfg)
                    feedback = SingleSlitContactFeedbackResultV1(nxt, nxt, clearance, torch.zeros(1, dtype=torch.bool), clearance)
                    context = single_slit_contact_observation_context_from_result_v1(feedback, torch.from_numpy(force)[None], cfg)
                torch.testing.assert_close(torch.from_numpy(observations[0]), observation[0], atol=2e-6, rtol=2e-6)
                artifact = args.output / f"{case_id}.npz"
                np.savez_compressed(
                    artifact, observation=np.stack(observations), planner_inputs=np.stack(inputs),
                    private_hidden=np.stack(private),
                    target_plan=np.stack(targets), target_force=teacher["forces"], states=teacher["states"],
                    actor_private_hidden=data["private_hidden"][tick],
                    actor_observation=observation[0].numpy(),
                    student_prefix_observation=data["observation"][max(0, tick - 31):tick],
                )
                row.update(artifact=artifact.name, artifact_sha256=sha256(artifact))
            record["results"].append(row)
            write_json(destination, record)
            print(json.dumps({k: row.get(k) for k in (
                "case_id", "admitted_for_training", "teacher_satisfies_handoff",
                "actor_learned_plan_authority_nodes",
            )}), flush=True)
    record["status"] = "complete"
    write_json(destination, record)


if __name__ == "__main__":
    main()
