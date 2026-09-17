#!/usr/bin/env python3
"""Offline gradient/representation diagnostics; not a deployment success test."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_local_skill_composition import sha256, write_json
from piano_movers.actuation_training import replay_physical_actuation
from piano_movers.config import EnvConfig
from piano_movers.contact_residual_control import load_contact_residual
from piano_movers.multi_waypoint_demo import SkillCompositionOptions, apply_solver_controls
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


def net_wrench(forces, obs):
    rx = obs[..., 1] * obs[..., 2] - obs[..., 0] * obs[..., 3]
    ry = obs[..., 0] * obs[..., 2] + obs[..., 1] * obs[..., 3]
    return torch.cat((forces.sum(1), (rx * forces[..., 1] - ry * forces[..., 0]).sum(1, keepdim=True)), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=Path("configs/p6_force_position_heading_control.json"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    recipe = json.loads(args.recipe.read_text())
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    loaded = load_single_slit_checkpoint(Path(recipe["checkpoint"]), device="cpu")
    if loaded.checkpoint_sha256 != recipe["checkpoint_sha256"] or cache["actor_sha256"] != loaded.checkpoint_sha256:
        raise ValueError("base checkpoint mismatch")
    policy = loaded.policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    control = load_contact_residual(args.control, loaded.checkpoint_sha256)
    rows, matrices, targets = [], {"difference256": [], "physical52_plus_bias": [], "memory8_plus_bias": [],
                                  "physical52_times_memory8": []}, []
    bounds = torch.tensor([1.2, 1.2, 3.])
    outside = 0
    for b in cache["groups"]["teacher"]:
        cfg = EnvConfig(**b["cfg"])
        policy.bind_environment(cfg)
        apply_solver_controls(policy, SkillCompositionOptions(**recipe["composition_options"]))
        active = b["arrays"]["features"][..., 12:20].any(-1).any(-1)
        for indices in torch.where(active)[0].split(64):
            if not len(indices):
                continue
            a = {k: v[indices] for k, v in b["arrays"].items()}
            executed = torch.where(a["active"], a["geometry"], a["source_plan"])
            oracle = a["target_plan"][..., 3:6] - executed[..., 3:6]
            outside += int((oracle.abs() >= bounds).any(-1).sum())
            with torch.no_grad():
                predicted = control(a["contact_difference"])
            delta = predicted.detach().requires_grad_()
            output = replay_physical_actuation(policy, a["obs"], a["source_plan"], a["geometry"], a["active"],
                                               twist_residual=delta)
            force_per_frame = F.smooth_l1_loss(output.forces / cfg.max_force, a["target_force"] / cfg.max_force,
                                              beta=.05, reduction="none").mean((1, 2))
            g_force = torch.autograd.grad(force_per_frame.sum(), delta, retain_graph=True)[0]
            scale = torch.tensor([cfg.n_agents * cfg.max_force] * 2 + [.75 * cfg.n_agents * cfg.max_force])
            target_wrench = net_wrench(a["target_force"], a["obs"]) / scale
            wrench_per_frame = F.smooth_l1_loss(output.wrench_section, target_wrench[:, None].expand_as(output.wrench_section),
                                               beta=.05, reduction="none").mean((1, 2))
            g_wrench = torch.autograd.grad(wrench_per_frame.sum(), delta)[0]
            direction = (oracle - predicted).flatten(1)
            saturation = (output.forces.detach().norm(dim=-1) >= cfg.max_force * (1. - 1e-5)).float().mean(1)
            for i in range(len(indices)):
                row = {"force_loss": float(force_per_frame[i].detach()), "saturated_node_fraction": float(saturation[i])}
                for name, gradient in (("force", g_force), ("wrench", g_wrench)):
                    g = gradient[i].flatten().detach()
                    row[name + "_gradient_norm"] = float(g.norm())
                    row[name + "_descent_alignment_to_oracle"] = float(
                        -(g * direction[i]).sum() / (g.norm() * direction[i].norm()).clamp_min(1e-15))
                rows.append(row)
            feat = a["features"].reshape(-1, 52).double()
            memory = feat[:, 12:20]
            matrices["difference256"].append(a["contact_difference"].reshape(-1, 256).double())
            matrices["physical52_plus_bias"].append(torch.cat((feat, torch.ones(len(feat), 1)), -1))
            matrices["memory8_plus_bias"].append(torch.cat((memory, torch.ones(len(feat), 1)), -1))
            matrices["physical52_times_memory8"].append((feat[:, :, None] * memory[:, None, :]).flatten(1))
            targets.append(torch.atanh((oracle / bounds).clamp(-.999, .999)).reshape(-1, 3).double())
    target = torch.cat(targets)
    representation = {}
    for name, pieces in matrices.items():
        x = torch.cat(pieces)
        scale = x.square().mean(0).sqrt().clamp_min(1e-4)
        x = x / scale
        ridge = 1e-5
        gram = x.T @ x / len(x) + ridge * torch.eye(x.shape[1], dtype=x.dtype)
        coefficient = torch.linalg.solve(gram, x.T @ target / len(x))
        predicted = (x @ coefficient).tanh() * bounds
        actual = target.tanh() * bounds
        representation[name] = {"feature_dim": x.shape[1], "ridge": ridge,
            "in_sample_twist_MAE": (predicted - actual).abs().mean(0).tolist(),
            "in_sample_raw_target_MSE": float((x @ coefficient - target).square().mean()),
            "scope": "in_sample_only_no_episode_split_no_deployment_claim"}
    summary = {"frames": len(rows), "node_rows": len(target), "oracle_outside_residual_bounds_node_rows": outside,
        "force_zero_gradient_frames": sum(r["force_gradient_norm"] < 1e-8 for r in rows),
        "wrench_zero_gradient_frames": sum(r["wrench_gradient_norm"] < 1e-8 for r in rows),
        "force_wrong_direction_frames": sum(r["force_descent_alignment_to_oracle"] <= 0 for r in rows),
        "wrench_wrong_direction_frames": sum(r["wrench_descent_alignment_to_oracle"] <= 0 for r in rows),
        "mean_saturated_node_fraction": sum(r["saturated_node_fraction"] for r in rows) / len(rows)}
    report = {"status": "complete", "cache_sha256": sha256(args.cache), "control_sha256": sha256(args.control),
              "recipe_sha256": sha256(args.recipe), "source_sha256": sha256(Path(__file__)),
              "summary": summary, "representation": representation, "gradient_frames": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"summary": summary, "representation": representation}))


if __name__ == "__main__":
    main()
