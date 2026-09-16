#!/usr/bin/env python3
"""Fine-tune the existing p5 morphology planner; never replace its architecture.

The original 52-input MLP and six used motion outputs receive teacher labels.
Frozen-parent successful development observations provide function-preserving
replay. This is private-plan imitation, not a claim that all labelled nodes
control the downstream action. Closed-loop/authority evaluation is mandatory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_local_skill_composition import sha256, write_json
from piano_movers.config import EnvConfig
from piano_movers.multi_waypoint_demo import Slit, WaypointCourse, _course_config
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


def active_anchor_config(base_cfg, course, stage, position_only_control):
    """Reconstruct only the command actually active in an archived frame."""
    if not 0 <= stage < len(course.waypoints):
        raise ValueError("anchor waypoint index outside the archived course")
    goal = course.waypoints[stage]
    wall = course.walls[min(stage, len(course.walls) - 1)]
    return replace(base_cfg, goal_x=goal[0] - wall.x, goal_y=goal[1], goal_theta=goal[2],
                   goal_orientation_required=(stage == len(course.waypoints) - 1
                                              or not position_only_control))


def admitted_anchor_rows(paths, actor_sha):
    """Authenticate successful development replay, preserving held-out splits."""
    sources, rows, seen = [], [], set()
    for path in paths:
        manifest = json.loads(path.read_text())
        if manifest["status"] != "complete" or manifest["split"] != "development_diagnostics":
            raise ValueError("anchors must be complete development data, not held-out evaluation")
        if manifest["checkpoint_sha256"] != actor_sha:
            raise ValueError("anchor actor checkpoint mismatch")
        provenance = {"path": str(path.resolve()), "sha256": sha256(path),
                      "selected_episodes": 0, "excluded_heldout_or_unsuccessful": 0,
                      "duplicate_traces_skipped": 0}
        for row in manifest["results"]:
            if not row["result"]["success"] or row["shape"] in ("S", "Z") or row["agents"] % 2:
                provenance["excluded_heldout_or_unsuccessful"] += 1
                continue
            if not row.get("continuous_certificate", {}).get("passed", False):
                raise ValueError("successful replay requires independent continuous certification")
            if row["trace_sha256"] in seen:
                provenance["duplicate_traces_skipped"] += 1
                continue
            seen.add(row["trace_sha256"])
            provenance["selected_episodes"] += 1
            rows.append((path, provenance["sha256"], row))
        sources.append(provenance)
    return sources, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path, default=None,
                        help="Authenticated actor for replay; defaults to initialization checkpoint")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--extra-anchors", type=Path, action="append", default=[],
                        help="Additional completed success replay from the same source actor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--trainable-scope", choices=("original_modules", "goal_context_only"),
                        default="original_modules")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    parent = load_single_slit_checkpoint(args.parent, device="cpu")
    policy = parent.policy
    replay = (load_single_slit_checkpoint(args.replay_source, device="cpu")
              if args.replay_source else parent)
    data = json.loads(args.dataset.read_text())
    if data["status"] != "complete" or not data["split"].startswith("train"):
        raise ValueError("requires completed training data")
    anchor_sources, replay_rows = admitted_anchor_rows([args.anchors, *args.extra_anchors], replay.checkpoint_sha256)
    root_parent_sha = parent.metadata.get("p6_mainline_training", {}).get(
        "parent_checkpoint_sha256", parent.checkpoint_sha256,
    )
    if data["parent_checkpoint_sha256"] != root_parent_sha:
        raise ValueError("checkpoint lineage mismatch")
    replay_root_sha = replay.metadata.get("p6_mainline_training", {}).get(
        "parent_checkpoint_sha256", replay.checkpoint_sha256,
    )
    if replay_root_sha != root_parent_sha:
        raise ValueError("replay actor has a different root checkpoint")
    if args.trainable_scope == "goal_context_only" and not policy.arm_cfg.local_se2_markov_goal_context:
        raise ValueError("goal-context-only fit requires an explicitly bridged input interface")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "training.json"
    if destination.exists():
        raise FileExistsError(destination)
    features, targets, commands = [], [], []
    anchor_features, anchor_targets, anchor_commands = [], [], []
    teacher_rows = []
    anchor_rows = []
    with torch.no_grad():
        for row in data["results"]:
            if not row["admitted_for_training"]:
                continue
            if row["shape"] in ("S", "Z") or row["agents"] % 2:
                raise ValueError("held-out shape/count in training")
            path = args.dataset.parent / row["artifact"]
            if sha256(path) != row["artifact_sha256"]:
                raise ValueError("teacher artifact hash mismatch")
            arrays = np.load(path)
            policy.bind_environment(EnvConfig(**row["env_cfg"]))
            private = torch.from_numpy(arrays["planner_inputs"])
            feature = policy.local_se2_markov_shape_inputs_from_planner_inputs(private)
            # This pilot targets only the fully supported morphology branch.
            # A fractional/null support requires a declared blended objective.
            support = policy.local_se2_noncanonical_geometry_support(private[..., -32:])
            if not bool((support > .99999).all()):
                raise ValueError("teacher scope requires full existing morphology-plan authority")
            features.append(feature.flatten(0, 1))
            commands.append(policy.local_se2_markov_goal_features(feature).flatten(0, 1))
            targets.append(torch.from_numpy(arrays["target_plan"][:, None, :6]).expand(-1, row["agents"], -1).flatten(0, 1))
            teacher_rows.append(row["case_id"])
        print(json.dumps({"preparation": "teachers_complete", "queries": len(teacher_rows)}), flush=True)
        for anchor_manifest, manifest_sha, row in replay_rows:
            path = anchor_manifest.parent / row["trace"]
            if sha256(path) != row["trace_sha256"]:
                raise ValueError("anchor trace hash mismatch")
            arrays = np.load(path)
            header = row["header"]
            course = WaypointCourse(tuple(header["source"]), tuple(Slit(**s) for s in header["walls"]), tuple(tuple(g) for g in header["waypoints"]), header["world"]["x"], header["world"]["y"])
            course = replace(course, source_velocity=tuple(header.get("source_velocity", (0., 0., 0.))),
                             morphology_scale=float(header.get("morphology_scale", 1.)))
            cfg = _course_config(parent.cfg, row["agents"], course, row["shape"])
            if row["options"]["nominal_glyph_size"]:
                cfg = replace(cfg, glyph_scale=1.)
            # Do not label every replay frame with the first goal. Stage IDs
            # select preprocessing metadata only; they never enter the model.
            for stage in np.unique(arrays["waypoint_index"]):
                active_cfg = active_anchor_config(cfg, course, int(stage), row["options"]["position_only_control"])
                policy.bind_environment(active_cfg)
                replay.policy.bind_environment(active_cfg)
                indices = np.flatnonzero(arrays["waypoint_index"] == stage)
                for start in range(0, len(indices), 128):
                    selected = indices[start:start + 128]
                    obs = torch.from_numpy(arrays["observation"][selected])
                    hidden = torch.from_numpy(arrays["private_hidden"][selected])
                    private = replay.policy.local_se2_base_planner_inputs(obs, hidden)
                    feature = policy.local_se2_markov_shape_inputs_from_planner_inputs(private)
                    mean = replay.policy.local_se2_markov_shape_outputs_from_planner_inputs(private)[0]
                    anchor_features.append(feature.flatten(0, 1))
                    anchor_commands.append(policy.local_se2_markov_goal_features(feature).flatten(0, 1))
                    anchor_targets.append(mean[..., :6].flatten(0, 1))
            anchor_rows.append({"shape": row["shape"], "agents": row["agents"],
                                "trace_sha256": row["trace_sha256"], "manifest_sha256": manifest_sha,
                                "case_id": row.get("case_id"), "profile": row.get("profile", "nominal")})
            if len(anchor_rows) % 50 == 0:
                print(json.dumps({"preparation": "replay", "episodes": len(anchor_rows),
                                  "episodes_total": len(replay_rows)}), flush=True)
    if not features or not anchor_features:
        raise ValueError("empty teacher or function-preserving anchor data")
    x, y, g, ax, ay, ag = (torch.cat(v).cuda() for v in (
        features, targets, commands, anchor_features, anchor_targets, anchor_commands,
    ))
    prefixes = (("local_se2_markov_goal_projection.",) if args.trainable_scope == "goal_context_only"
                else ("local_se2_markov_shape_encoder.", "local_se2_markov_shape_mean_head.",
                      "local_se2_markov_goal_projection."))
    names = [name for name, _ in policy.named_parameters() if name.startswith(prefixes)]
    frozen = {k: v.clone() for k, v in policy.state_dict().items() if k not in names}
    for name, p in policy.named_parameters():
        p.requires_grad_(name in names)
    policy.cuda()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-5)
    record = {
        "schema": "piano-movers.original-markov-dagger-training.v1", "status": "running",
        "parent_checkpoint_sha256": root_parent_sha,
        "initial_checkpoint_sha256": parent.checkpoint_sha256,
        "replay_source_checkpoint_sha256": replay.checkpoint_sha256,
        "initialization_from_previous_mainline": parent.checkpoint_sha256 != root_parent_sha,
        "dataset_sha256": sha256(args.dataset), "anchors_sha256": sha256(args.anchors),
        "anchor_manifests": anchor_sources,
        "teacher_rows": teacher_rows, "anchor_rows": anchor_rows,
        "teacher_agent_frames": len(x), "anchor_agent_frames": len(ax),
        "trainable_parameter_names": names,
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "arm_cfg_unchanged": True, "observation_dim": 27,
        "trainable_scope": args.trainable_scope,
        "private_goal_command_dim": 4 if policy.arm_cfg.local_se2_markov_goal_context else 0,
        "replay_commands_reconstructed_per_active_stage": True,
        "objective": "private_plan_imitation_plus_source_function_replay",
        "goal_relative_y_input": policy.arm_cfg.local_se2_markov_goal_relative_y,
        "downstream_analytic_overrides_unchanged": True,
        "step_requested": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "anchor_weight": args.anchor_weight,
        "seed": args.seed, "gpu": torch.cuda.get_device_name(),
        "source_sha256": {str(p): sha256(p) for p in (Path(__file__), Path("src/piano_movers/models.py"))},
        "history": [], "closed_loop_success": None,
    }
    write_json(destination, record)
    write_json(args.output / "dataset_snapshot.json", data)
    started = time.monotonic()

    def predict(batch, command):
        encoded = policy.local_se2_markov_shape_encode(batch, goal_context=command)
        return torch.tanh(policy.local_se2_markov_shape_mean_head(encoded))[..., :6]

    for step in range(1, args.steps + 1):
        index = torch.randint(len(x), (args.batch_size,), device=x.device)
        anchor_index = torch.randint(len(ax), (args.batch_size,), device=x.device)
        fit = F.smooth_l1_loss(predict(x[index], g[index]), y[index], beta=.05)
        preserve = F.smooth_l1_loss(predict(ax[anchor_index], ag[anchor_index]), ay[anchor_index], beta=.05)
        loss = fit + args.anchor_weight * preserve
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step == 1 or step % 1000 == 0 or step == args.steps:
            point = {"step": step, "fit_loss": float(fit.detach()), "anchor_loss": float(preserve.detach()), "seconds": time.monotonic() - started}
            record["history"].append(point)
            print(json.dumps(point), flush=True)
            state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
            if any(not torch.equal(state[k], value) for k, value in frozen.items()):
                raise RuntimeError("a frozen original tensor changed")
            payload = dict(parent.metadata)
            payload.update(policy_state_dict=state, step=step, training_stage="p6-original-markov-dagger-v1", selection_mode="development_candidate_not_released", p6_mainline_training=record)
            path = args.output / f"checkpoint_step{step}.pt"
            torch.save(payload, path)
            record["checkpoint"] = {"path": path.name, "sha256": sha256(path)}
            record["step"] = step
            write_json(destination, record)
    record["status"] = "complete"
    write_json(destination, record)


if __name__ == "__main__":
    main()
