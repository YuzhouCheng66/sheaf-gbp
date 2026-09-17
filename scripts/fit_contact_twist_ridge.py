#!/usr/bin/env python3
"""Same 768-parameter control, closed-form teacher-twist supervision.

This changes the training objective/optimizer, not the deployed architecture.
Original successful contact replay targets zero correction; no-contact output
remains identically zero. All fits are development controls, not releases.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.contact_residual_control import ContactResidualControl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--anchor-weight", type=float, default=2.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.ridge <= 0 or args.anchor_weight < 0:
        raise ValueError("positive ridge and nonnegative anchor weight required")
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache["status"] != "complete":
        raise ValueError("completed authenticated cache required")
    started = time.monotonic()
    groups = {}
    bounds = torch.tensor([1.2, 1.2, 3.], dtype=torch.float64)
    for kind, buckets in cache["groups"].items():
        x, y, weights = [], [], []
        outside, frames = 0, 0
        for b in buckets:
            data = b["arrays"]
            active = data["features"][..., 12:20].any(-1).any(-1)
            if not active.any():
                continue
            feature = data["contact_difference"][active].reshape(-1, 256).double()
            actual = torch.where(data["active"][active], data["geometry"][active], data["source_plan"][active])
            delta = ((data["target_plan"][active][..., 3:6] - actual[..., 3:6]).reshape(-1, 3).double()
                     if kind == "teacher" else torch.zeros(len(feature), 3, dtype=torch.float64))
            outside += int((delta.abs() >= bounds).any(-1).sum())
            frames += int(active.sum())
            x.append(feature)
            y.append(torch.atanh((delta / bounds).clamp(-.999, .999)))
            weights.append(torch.full((len(feature),), 1. / b["cfg"]["n_agents"], dtype=torch.float64))
        if not x:
            raise ValueError("empty active-memory group")
        groups[kind] = {"x": torch.cat(x), "y": torch.cat(y), "w": torch.cat(weights),
                        "frames": frames, "outside": outside}
    teacher = groups["teacher"]
    scale = teacher["x"].square().mean(0).sqrt().clamp_min(1e-4)
    gram = torch.zeros(256, 256, dtype=torch.float64)
    rhs = torch.zeros(256, 3, dtype=torch.float64)
    for kind, g in groups.items():
        x = g["x"] / scale
        w = g["w"] / g["frames"] * (1. if kind == "teacher" else args.anchor_weight)
        gram += x.T @ (w[:, None] * x)
        rhs += x.T @ (w[:, None] * g["y"])
    coefficient = torch.linalg.solve(gram + args.ridge * torch.eye(256, dtype=gram.dtype), rhs)
    control = ContactResidualControl()
    control.feature_scale.copy_(scale.float())
    control.readout.weight.data.copy_(coefficient.T.float())
    metrics = {}
    with torch.no_grad():
        for kind, g in groups.items():
            predicted = control(g["x"].float()).double()
            target = g["y"].tanh() * bounds
            metrics[kind] = {"frames": g["frames"], "node_rows": len(g["x"]),
                "outside_residual_bounds_rows": g["outside"],
                "twist_MAE": (predicted - target).abs().mean(0).tolist(),
                "scope": "in_sample_only_no_deployment_claim"}
    args.output.mkdir(parents=True)
    record = {"status": "complete", "schema": "piano-movers.contact-ridge-fit.v1",
        "actor_sha256": cache["actor_sha256"], "cache_sha256": sha256(args.cache),
        "dataset_sha256": cache["dataset_sha256"], "source_sha256": sha256(Path(__file__)),
        "ridge": args.ridge, "anchor_weight": args.anchor_weight, "trainable_parameters": 768,
        "objective": "weighted_pre_tanh_teacher_twist_and_zero_success_replay",
        "base_policy_frozen": True, "zero_contact_correction": "identically_zero",
        "teacher_oracle_used_only_in_training": True, "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started, "release_status": "independent_control_not_promoted"}
    checkpoint = args.output / "control.pt"
    torch.save({"schema": "piano-movers.contact-residual-control.v1", "actor_sha256": cache["actor_sha256"],
                "hidden_width": 0, "state_dict": control.state_dict(), "training": record}, checkpoint)
    record["checkpoint_sha256"] = sha256(checkpoint)
    write_json(args.output / "training.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
