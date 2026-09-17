#!/usr/bin/env python3
"""Compare original-network plan/force DAgger on identical whole-agent frames.

No deployed architecture change. Actual original GBP, geometry authority and
grasp decoder remain in the differentiable force objective. Frozen-success
replay preserves executed forces, not an arbitrary private-plan null space.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_local_skill_composition import sha256, write_json
from train_mainline_skill_dagger import active_anchor_config, admitted_anchor_rows
from piano_movers.actuation_training import replay_physical_actuation
from piano_movers.config import EnvConfig
from piano_movers.multi_waypoint_demo import (
    Slit, WaypointCourse, SkillCompositionOptions, _course_config, apply_solver_controls,
    INTERMEDIATE_POSITION_TOLERANCE_M,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_local_planner import local_se2_terminal_attractor_plan


FIELDS = ("features", "source_plan", "source_morph", "support", "obs", "geometry",
          "active", "target_force", "target_plan")
BUFFERS = ("local_se2_analytic_reference_edges", "local_se2_analytic_tree_edge_mask",
           "local_se2_distributed_grasp_body_local_cache",
           "local_se2_distributed_grasp_body_sufficient_cache")


def configure_goal_input_control(policy, *, relative_y=False, goal_context=False):
    """Independent input control, leaving original tensors and RNG unchanged.

    The optional four-column affine term uses the existing partitioned-input
    forward path. Relative-y is NOT a behavior-preserving conversion. Preparing
    source targets must happen before this function is called.
    """
    if policy.arm_cfg.local_se2_markov_goal_context or policy.arm_cfg.local_se2_markov_goal_relative_y:
        raise ValueError("input control requires the original source representation")
    additional = 0
    if goal_context:
        first = policy.local_se2_markov_shape_encoder[0]
        if first.weight.device.type != "cpu":
            raise ValueError("construct the audited input bridge on CPU before training")
        with torch.random.fork_rng(devices=[]):
            projection = torch.nn.Linear(4, first.out_features, bias=False,
                                         device=first.weight.device, dtype=first.weight.dtype)
            torch.nn.init.zeros_(projection.weight)
        policy.local_se2_markov_goal_projection = projection
        additional = projection.weight.numel()
    policy.arm_cfg = replace(policy.arm_cfg, local_se2_markov_goal_context=goal_context,
                             local_se2_markov_goal_relative_y=relative_y)
    return additional


def composition_cfg(cfg, options):
    return replace(cfg, complete_local_corridor_exit=options.complete_corridor_exit,
                   local_corridor_goal_heading_search=options.resolve_position_goal_heading,
                   local_cspace_collision_substeps=options.planning_substeps,
                   local_cspace_validate_all_lookaheads=options.validate_all_cspace_lookaheads,
                   local_support_check_alignment_sweep=options.check_support_alignment_sweep,
                   local_support_hold_feasible_crossing=options.hold_feasible_crossing,
                   local_corridor_exit_goal_radius_m=(INTERMEDIATE_POSITION_TOLERANCE_M
                                                      if options.respect_handoff_goal_region else 0.))


def private_from_prediction(prediction, source_plan, source_morph, support):
    """Preserve the frozen canonical branch, including fractional support."""
    updated = source_plan[..., :6] + support * (prediction[..., :6] - source_morph[..., :6])
    updated = torch.where(support == 1., prediction[..., :6], updated)
    updated = torch.where(support == 0., source_plan[..., :6], updated)
    return torch.cat((updated, source_plan[..., 6:]), -1)


def source_correct_plan_target(target, source, geometry, active, preserve):
    """Training-only alternative labels; do not mutate the teacher witness."""
    if preserve.dtype != torch.bool or preserve.shape != target.shape[:1]:
        raise ValueError("preservation mask must be one boolean per whole-agent frame")
    if source.shape != target.shape or geometry.shape != target.shape:
        raise ValueError("source and teacher plans must have identical shapes")
    return torch.where(preserve[:, None, None], torch.where(active, geometry, source), target)


def add_bucket(buckets, cfg, arrays):
    key = json.dumps(asdict(cfg), sort_keys=True)
    if key not in buckets:
        buckets[key] = {"cfg": asdict(cfg), "chunks": {name: [] for name in FIELDS}}
    for name in FIELDS:
        buckets[key]["chunks"][name].append(arrays[name].detach().cpu().clone())


def prepare(args, loaded, options):
    policy = loaded.policy.eval()
    manifest = json.loads(args.dataset.read_text())
    if manifest["status"] != "complete" or not manifest["split"].startswith("train"):
        raise ValueError("requires completed development training data")
    root_sha = loaded.metadata.get("p6_mainline_training", {}).get(
        "parent_checkpoint_sha256", loaded.checkpoint_sha256)
    if manifest["parent_checkpoint_sha256"] != root_sha:
        raise ValueError("teacher lineage mismatch")
    sources, admitted = admitted_anchor_rows(args.anchors, loaded.checkpoint_sha256)
    groups = {"teacher": {}, "replay": {}}
    teacher_rows, anchor_rows, force_errors = [], [], []
    with torch.no_grad():
        for row in manifest["results"]:
            if not row["admitted_for_training"]:
                continue
            if row["shape"] in ("S", "Z") or row["agents"] % 2:
                raise ValueError("held-out teacher data")
            artifact = args.dataset.parent / row["artifact"]
            if sha256(artifact) != row["artifact_sha256"]:
                raise ValueError("teacher artifact changed")
            cfg = composition_cfg(EnvConfig(**row["env_cfg"]), options)
            policy.bind_environment(cfg)
            apply_solver_controls(policy, options)
            with np.load(artifact) as data:
                for start in range(0, len(data["observation"]), 64):
                    sl = slice(start, start + 64)
                    obs = torch.from_numpy(data["observation"][sl].copy())
                    hidden = torch.from_numpy(data["private_hidden"][sl].copy())
                    private = policy.local_se2_base_planner_inputs(obs, hidden)
                    feature = policy.local_se2_markov_shape_inputs_from_planner_inputs(private)
                    old = torch.from_numpy(data["planner_inputs"][sl].copy())
                    torch.testing.assert_close(feature, policy.local_se2_markov_shape_inputs_from_planner_inputs(old),
                                               atol=2e-6, rtol=2e-6)
                    morph, _, _, support = policy.local_se2_markov_shape_outputs_from_planner_inputs(private)
                    if not bool((support == 1.).all()):
                        raise ValueError("teacher requires full existing morphology authority")
                    arm = policy.arm_cfg
                    geom, active = local_se2_terminal_attractor_plan(
                        obs, morph, cfg, position_gain=arm.local_se2_terminal_position_gain,
                        angle_gain=arm.local_se2_terminal_angle_gain, max_speed=arm.local_se2_terminal_max_speed,
                        max_omega=arm.local_se2_terminal_max_omega,
                        feasible_corridor=arm.local_se2_feasible_corridor_attractor,
                        route_free_geometry=arm.local_se2_route_free_geometry_planner,
                        corridor_reserve_m=arm.local_se2_corridor_reserve_m,
                        corridor_max_speed=arm.local_se2_corridor_max_speed)
                    if start == 0:
                        full = policy(obs[:1], policy.local_se2_analytic_reference_edges, hidden[:1])
                        torch.testing.assert_close(morph[:1], full.aux["local_se2_plan_mean"], atol=2e-6, rtol=2e-6)
                        fast = replay_physical_actuation(policy, obs[:1], morph[:1], geom[:1], active[:1])
                        torch.testing.assert_close(fast.forces, full.forces, atol=2e-5, rtol=2e-5)
                        force_errors.append(float((fast.forces - full.forces).abs().max()))
                    target_plan = torch.from_numpy(data["target_plan"][sl].copy())[:, None].expand(-1, cfg.n_agents, -1)
                    force_field = "target_force"
                    if "preserve_source_mask" in data.files:
                        if manifest.get("training_label_schema") != "piano-movers.source-correct-contact-labels.v1":
                            raise ValueError("source-correct labels require explicit audited schema")
                        preserve = torch.from_numpy(data["preserve_source_mask"][sl].copy()).bool()
                        target_plan = source_correct_plan_target(target_plan, morph, geom, active, preserve)
                        force_field = "training_target_force"
                    add_bucket(groups["teacher"], cfg, dict(
                        features=feature, source_plan=morph, source_morph=morph, support=support,
                        obs=obs, geometry=geom, active=active,
                        target_force=torch.from_numpy(data[force_field][sl].copy()), target_plan=target_plan))
            teacher_rows.append(row["case_id"])
            if len(teacher_rows) % 10 == 0:
                print(json.dumps({"prepare": "teachers", "queries": len(teacher_rows)}), flush=True)
        for anchor_manifest, manifest_sha, row in admitted:
            path = anchor_manifest.parent / row["trace"]
            if sha256(path) != row["trace_sha256"]:
                raise ValueError("replay trace changed")
            if SkillCompositionOptions(**row["options"]) != options:
                raise ValueError("force-preserving replay must use the declared deployment recipe")
            h = row["header"]
            course = WaypointCourse(tuple(h["source"]), tuple(Slit(**s) for s in h["walls"]),
                                    tuple(tuple(g) for g in h["waypoints"]), h["world"]["x"], h["world"]["y"],
                                    tuple(h.get("source_velocity", (0., 0., 0.))), h.get("morphology_scale", 1.))
            base = _course_config(loaded.cfg, row["agents"], course, row["shape"])
            if options.nominal_glyph_size:
                base = replace(base, glyph_scale=1.)
            base = composition_cfg(base, options)
            with np.load(path) as data:
                for stage in np.unique(data["waypoint_index"]):
                    cfg = active_anchor_config(base, course, int(stage), options.position_only_control)
                    policy.bind_environment(cfg)
                    apply_solver_controls(policy, options)
                    indices = np.flatnonzero(data["waypoint_index"] == stage)
                    for start in range(0, len(indices), 128):
                        ix = indices[start:start + 128]
                        obs = torch.from_numpy(data["observation"][ix].copy())
                        hidden = torch.from_numpy(data["private_hidden"][ix].copy())
                        private = policy.local_se2_base_planner_inputs(obs, hidden)
                        feature = policy.local_se2_markov_shape_inputs_from_planner_inputs(private)
                        morph, _, _, support = policy.local_se2_markov_shape_outputs_from_planner_inputs(private)
                        source = torch.from_numpy(data["learned_plan"][ix].copy())
                        geometry = torch.from_numpy(data["plan"][ix].copy())
                        active = torch.from_numpy(data["analytic_active"][ix].copy()).bool()
                        force = torch.from_numpy(data["forces"][ix].copy())
                        if start == 0:
                            replay = replay_physical_actuation(policy, obs[:1], source[:1], geometry[:1], active[:1])
                            torch.testing.assert_close(replay.forces, force[:1], atol=2e-5, rtol=2e-5)
                            force_errors.append(float((replay.forces - force[:1]).abs().max()))
                        add_bucket(groups["replay"], cfg, dict(
                            features=feature, source_plan=source, source_morph=morph, support=support,
                            obs=obs, geometry=geometry, active=active, target_force=force, target_plan=source))
            anchor_rows.append({"case_id": row.get("case_id"), "trace_sha256": row["trace_sha256"],
                                "manifest_sha256": manifest_sha})
            if len(anchor_rows) % 50 == 0:
                print(json.dumps({"prepare": "replay", "episodes": len(anchor_rows)}), flush=True)
    packed = {}
    for kind, buckets in groups.items():
        if not buckets:
            raise ValueError("empty training group")
        packed[kind] = [{"cfg": b["cfg"], "arrays": {k: torch.cat(v) for k, v in b["chunks"].items()}}
                        for b in buckets.values()]
    return {"status": "complete", "actor_sha256": loaded.checkpoint_sha256,
            "root_parent_sha256": root_sha, "dataset_sha256": sha256(args.dataset),
            "recipe_sha256": sha256(args.recipe), "anchor_manifests": sources,
            "teacher_rows": teacher_rows, "anchor_rows": anchor_rows, "groups": packed,
            "authenticated_force_states": len(force_errors), "maximum_force_error_N": max(force_errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, nargs="+", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--objective", choices=("force", "plan"), default="force")
    parser.add_argument("--input-column3", choices=("absolute_y", "goal_relative_y"), default="absolute_y",
                        help="Independent same-width semantic control; no change to cached source action targets")
    parser.add_argument("--goal-context", action="store_true",
                        help="Independent +4 goal input columns; original depth, GBP and decoder retained")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64, help="whole multi-agent states per source")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--anchor-weight", type=float, default=3.)
    parser.add_argument("--seed", type=int, default=7102)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    # No TF32 change: preserve the already audited original float32 solver.
    recipe = json.loads(args.recipe.read_text())
    loaded = load_single_slit_checkpoint(Path(recipe["checkpoint"]), device="cpu")
    if loaded.checkpoint_sha256 != recipe["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    policy = loaded.policy.eval()
    if (policy.arm_cfg.local_se2_markov_goal_context or policy.arm_cfg.local_se2_markov_goal_relative_y
            or policy.arm_cfg.local_se2_path_memory):
        raise ValueError("this control targets unchanged original 52-input mainline")
    options = SkillCompositionOptions(**recipe["composition_options"])
    if args.cache.exists():
        cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    else:
        cached = prepare(args, loaded, options)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached, args.cache)
    if (cached["actor_sha256"] != loaded.checkpoint_sha256 or cached["dataset_sha256"] != sha256(args.dataset)
            or cached["recipe_sha256"] != sha256(args.recipe)):
        raise ValueError("cache provenance mismatch")
    if [s["path"] for s in cached["anchor_manifests"]] != [str(p.resolve()) for p in args.anchors]:
        raise ValueError("cache replay sources mismatch")
    for source in cached["anchor_manifests"]:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("cached source manifest changed")
    print(json.dumps({"cache": str(args.cache), "force_parity_max": cached["maximum_force_error_N"],
                      "frames": {k: sum(len(b["arrays"]["features"]) for b in v)
                                 for k, v in cached["groups"].items()}}), flush=True)
    if args.prepare_only:
        return
    if args.output is None:
        raise ValueError("training requires --output")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "training.json"
    if destination.exists():
        raise FileExistsError(destination)
    additional_parameters = configure_goal_input_control(
        policy, relative_y=args.input_column3 == "goal_relative_y", goal_context=args.goal_context)
    names = [name for name, _ in policy.named_parameters()
             if name.startswith(("local_se2_markov_shape_encoder.", "local_se2_markov_shape_mean_head.",
                                 "local_se2_markov_goal_projection."))]
    frozen = {k: v.clone() for k, v in policy.state_dict().items() if k not in names}
    for name, p in policy.named_parameters():
        p.requires_grad_(name in names)
    policy.cuda().eval()  # eval enables the existing attachment-only grasp cache; gradients remain on.
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                  lr=args.learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    groups = cached.pop("groups")
    weights = {}
    for kind, buckets in groups.items():
        weights[kind] = np.array([len(b["arrays"]["features"]) for b in buckets], dtype=float)
        weights[kind] /= weights[kind].sum()
        for bucket in buckets:
            bucket["arrays"] = {k: v.cuda() for k, v in bucket["arrays"].items()}
            if args.input_column3 == "goal_relative_y" and bucket["cfg"]["goal_y"] != 0.:
                bucket["arrays"]["features"][..., 3] -= bucket["cfg"]["goal_y"] / bucket["cfg"]["world_y"]
    record = {**cached, "status": "running", "schema": "piano-movers.original-action-dagger.v1",
              "parent_checkpoint_sha256": cached["root_parent_sha256"],
              "initial_checkpoint_sha256": loaded.checkpoint_sha256,
              "cache_sha256": sha256(args.cache), "objective": args.objective,
              "source_sha256": {str(p): sha256(p) for p in (Path(__file__), Path("src/piano_movers/actuation_training.py"),
                                                             Path("src/piano_movers/models.py"))},
              "steps_requested": args.steps, "batch_size_whole_states_per_source": args.batch_size,
              "learning_rate": args.learning_rate, "anchor_weight": args.anchor_weight, "seed": args.seed,
              "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
              "arm_cfg_unchanged": args.input_column3 == "absolute_y" and not args.goal_context,
              "input_column3": args.input_column3, "additional_parameters": additional_parameters,
              "goal_context": args.goal_context, "goal_projection_zero_initialized": args.goal_context,
              "input_bridge_preserves_rng": True,
              "source_targets_unchanged": True, "downstream_geometry_authority_unchanged": True,
              "gpu": torch.cuda.get_device_name(), "history": [], "closed_loop_success": None}
    write_json(destination, record)

    def sample_loss(kind):
        bucket = groups[kind][rng.choice(len(groups[kind]), p=weights[kind])]
        cfg = EnvConfig(**bucket["cfg"])
        if "buffers" in bucket:
            policy.env_cfg = cfg
            for name, tensor in bucket["buffers"].items():
                setattr(policy, name, tensor)
            policy.local_se2_analytic_gbp.num_steps = bucket["rounds"]
        else:
            policy.bind_environment(cfg)
            bucket["rounds"] = apply_solver_controls(policy, options)["rounds"]
        data = bucket["arrays"]
        ix = torch.randint(len(data["features"]), (args.batch_size,), device="cuda")
        batch = {k: v[ix] for k, v in data.items()}
        encoded = policy.local_se2_markov_shape_encode(batch["features"])
        pred = torch.tanh(policy.local_se2_markov_shape_mean_head(encoded))
        plan = private_from_prediction(pred, batch["source_plan"], batch["source_morph"], batch["support"])
        if args.objective == "force":
            action = replay_physical_actuation(policy, batch["obs"], plan, batch["geometry"], batch["active"])
            loss = F.smooth_l1_loss(action.forces / cfg.max_force, batch["target_force"] / cfg.max_force, beta=.05)
        else:
            loss = F.smooth_l1_loss(plan[..., :6], batch["target_plan"][..., :6], beta=.05)
        bucket["buffers"] = {name: getattr(policy, name) for name in BUFFERS}
        return loss

    start_time = time.monotonic()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        fit = sample_loss("teacher")
        fit.backward()
        replay = sample_loss("replay")
        (args.anchor_weight * replay).backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            point = {"step": step, "fit_loss": float(fit.detach()), "anchor_loss": float(replay.detach()),
                     "seconds": time.monotonic() - start_time}
            record["history"].append(point)
            print(json.dumps(point), flush=True)
            record["step"] = step
            write_json(destination, record)
        if step % 1000 == 0 or step == args.steps:
            state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
            if any(not torch.equal(state[k], value) for k, value in frozen.items()):
                raise RuntimeError("a frozen tensor changed")
            payload = dict(loaded.metadata)
            payload.update(policy_state_dict=state, arm_cfg=asdict(policy.arm_cfg), step=step, training_stage="p6-original-action-dagger-v1",
                           selection_mode="independent_development_control_not_released", p6_mainline_training=record)
            path = args.output / f"checkpoint_step{step}.pt"
            torch.save(payload, path)
            record["checkpoint"] = {"path": path.name, "sha256": sha256(path)}
    record["status"] = "complete"
    write_json(destination, record)


if __name__ == "__main__":
    main()
