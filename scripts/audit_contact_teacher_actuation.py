#!/usr/bin/env python3
"""Check teacher action reachability before attributing failure to capacity.

The oracle inserts the recorded teacher's next-twist target after geometry;
it is training-only and must never be reported as deployed model success.
Measure individual forces AND net wrench: null-space allocations can differ.
"""

import argparse
import json
from pathlib import Path

import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.actuation_training import replay_physical_actuation
from piano_movers.config import EnvConfig
from piano_movers.contact_residual_control import load_contact_residual
from piano_movers.multi_waypoint_demo import SkillCompositionOptions, apply_solver_controls
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=Path("configs/p6_force_position_heading_control.json"))
    parser.add_argument("--control", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    recipe = json.loads(args.recipe.read_text())
    loaded = load_single_slit_checkpoint(Path(recipe["checkpoint"]), device="cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache["actor_sha256"] != loaded.checkpoint_sha256 or recipe["checkpoint_sha256"] != loaded.checkpoint_sha256:
        raise ValueError("base checkpoint mismatch")
    controls = {str(p): load_contact_residual(p, loaded.checkpoint_sha256) for p in args.control}
    totals = {name: {"frames": 0, "force_coordinates": 0, "absolute_force_error_sum_N": 0.,
                     "net_force_error_norm_sum_N": 0., "torque_error_abs_sum_Nm": 0.,
                     "maximum_force_error_N": 0.} for name in ("base", "teacher_twist_oracle", *controls)}
    for bucket in cache["groups"]["teacher"]:
        cfg = EnvConfig(**bucket["cfg"])
        policy = loaded.policy.eval()
        policy.bind_environment(cfg)
        apply_solver_controls(policy, SkillCompositionOptions(**recipe["composition_options"]))
        data = bucket["arrays"]
        active = data["features"][..., 12:20].any(-1).any(-1)
        for start in range(0, int(active.sum()), 64):
            ix = torch.where(active)[0][start:start + 64]
            b = {k: v[ix] for k, v in data.items()}
            executed = torch.where(b["active"], b["geometry"], b["source_plan"])
            residuals = {"base": None, "teacher_twist_oracle": b["target_plan"][..., 3:6] - executed[..., 3:6]}
            residuals.update({name: control(b["contact_difference"]) for name, control in controls.items()})
            obs = b["obs"]
            rx = obs[..., 1] * obs[..., 2] - obs[..., 0] * obs[..., 3]
            ry = obs[..., 0] * obs[..., 2] + obs[..., 1] * obs[..., 3]
            for name, residual in residuals.items():
                output = replay_physical_actuation(policy, obs, b["source_plan"], b["geometry"], b["active"],
                                                   twist_residual=residual)
                error = output.forces - b["target_force"]
                values = totals[name]
                values["frames"] += len(obs)
                values["force_coordinates"] += error.numel()
                values["absolute_force_error_sum_N"] += float(error.abs().sum())
                values["net_force_error_norm_sum_N"] += float(error.sum(1).norm(dim=-1).sum())
                values["torque_error_abs_sum_Nm"] += float((rx * error[..., 1] - ry * error[..., 0]).sum(1).abs().sum())
                values["maximum_force_error_N"] = max(values["maximum_force_error_N"], float(error.abs().max()))
    for values in totals.values():
        values["force_MAE_N"] = values["absolute_force_error_sum_N"] / values["force_coordinates"]
        values["mean_net_force_error_N"] = values["net_force_error_norm_sum_N"] / values["frames"]
        values["mean_torque_error_Nm"] = values["torque_error_abs_sum_Nm"] / values["frames"]
    report = {"status": "complete", "cache_sha256": sha256(args.cache), "recipe_sha256": sha256(args.recipe),
              "source_sha256": sha256(Path(__file__)), "control_sha256": {str(p): sha256(p) for p in args.control},
              "scope": "offline_teacher_contact_memory_frames_not_closed_loop", "results": totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
