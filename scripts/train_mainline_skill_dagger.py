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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=7101)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    parent = load_single_slit_checkpoint(args.parent, device="cpu")
    policy = parent.policy
    data, anchors = (json.loads(p.read_text()) for p in (args.dataset, args.anchors))
    if data["status"] != "complete" or not data["split"].startswith("train"):
        raise ValueError("requires completed training data")
    if anchors["status"] != "complete" or anchors["split"] != "development_diagnostics":
        raise ValueError("anchors must be disclosed development data, not held-out evaluation")
    if data["parent_checkpoint_sha256"] != parent.checkpoint_sha256 or anchors["checkpoint_sha256"] != parent.checkpoint_sha256:
        raise ValueError("checkpoint lineage mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "training.json"
    if destination.exists():
        raise FileExistsError(destination)
    features, targets, anchor_features, anchor_targets = [], [], [], []
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
            targets.append(torch.from_numpy(arrays["target_plan"][:, None, :6]).expand(-1, row["agents"], -1).flatten(0, 1))
            teacher_rows.append(row["case_id"])
        for row in anchors["results"]:
            if not row["result"]["success"] or row["shape"] in ("S", "Z") or row["agents"] % 2:
                continue
            path = args.anchors.parent / row["trace"]
            if sha256(path) != row["trace_sha256"]:
                raise ValueError("anchor trace hash mismatch")
            arrays = np.load(path)
            header = row["header"]
            course = WaypointCourse(tuple(header["source"]), tuple(Slit(**s) for s in header["walls"]), tuple(tuple(g) for g in header["waypoints"]), header["world"]["x"], header["world"]["y"])
            cfg = _course_config(parent.cfg, row["agents"], course, row["shape"])
            if row["options"]["nominal_glyph_size"]:
                cfg = replace(cfg, glyph_scale=1.)
            policy.bind_environment(cfg)
            for start in range(0, len(arrays["state"]), 128):
                obs = torch.from_numpy(arrays["observation"][start:start + 128])
                hidden = torch.from_numpy(arrays["private_hidden"][start:start + 128])
                private = policy.local_se2_base_planner_inputs(obs, hidden)
                feature = policy.local_se2_markov_shape_inputs_from_planner_inputs(private)
                mean = policy.local_se2_markov_shape_outputs_from_planner_inputs(private)[0]
                anchor_features.append(feature.flatten(0, 1))
                anchor_targets.append(mean[..., :6].flatten(0, 1))
            anchor_rows.append({"shape": row["shape"], "agents": row["agents"], "trace_sha256": row["trace_sha256"]})
    if not features or not anchor_features:
        raise ValueError("empty teacher or function-preserving anchor data")
    x, y, ax, ay = (torch.cat(v).cuda() for v in (features, targets, anchor_features, anchor_targets))
    names = [name for name, _ in policy.named_parameters() if name.startswith(("local_se2_markov_shape_encoder.", "local_se2_markov_shape_mean_head."))]
    frozen = {k: v.clone() for k, v in policy.state_dict().items() if k not in names}
    for name, p in policy.named_parameters():
        p.requires_grad_(name in names)
    policy.cuda()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-5)
    record = {
        "schema": "piano-movers.original-markov-dagger-training.v1", "status": "running",
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "dataset_sha256": sha256(args.dataset), "anchors_sha256": sha256(args.anchors),
        "teacher_rows": teacher_rows, "anchor_rows": anchor_rows,
        "teacher_agent_frames": len(x), "anchor_agent_frames": len(ax),
        "trainable_parameter_names": names,
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "arm_cfg_unchanged": True, "observation_dim": 27,
        "objective": "private_plan_imitation_plus_source_function_replay",
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

    def predict(batch):
        return torch.tanh(policy.local_se2_markov_shape_mean_head(policy.local_se2_markov_shape_encoder(batch)))[..., :6]

    for step in range(1, args.steps + 1):
        index = torch.randint(len(x), (args.batch_size,), device=x.device)
        anchor_index = torch.randint(len(ax), (args.batch_size,), device=x.device)
        fit = F.smooth_l1_loss(predict(x[index]), y[index], beta=.05)
        preserve = F.smooth_l1_loss(predict(ax[anchor_index]), ay[anchor_index], beta=.05)
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
