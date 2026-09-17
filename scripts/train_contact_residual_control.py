#!/usr/bin/env python3
"""Independent contact-authority readout controls on authentic DAgger.

Freeze the original network, GBP and decoder. Fit executed teacher forces
after actual contact, preserving successful source contact trajectories.
No-contact force preservation is algebraic, not merely a replay-loss claim.
The sidecar never overwrites or replaces the mainline checkpoint.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_local_skill_composition import sha256, write_json
from train_mainline_action_dagger import prepare, BUFFERS
from piano_movers.actuation_training import replay_physical_actuation
from piano_movers.config import EnvConfig
from piano_movers.contact_residual_control import ContactResidualControl, contact_encoding_difference, load_contact_residual
from piano_movers.multi_waypoint_demo import SkillCompositionOptions, apply_solver_controls
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("configs/p6_force_position_heading_control.json"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path, required=True)
    parser.add_argument("--replay-variant", default="impulse_reset_feedback")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--anchor-weight", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=7113)
    parser.add_argument("--initial-control", type=Path, help="Continue the independent readout; preserve its feature scale")
    parser.add_argument("--hidden-width", type=int, choices=(0, 64), default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "training.json").exists():
        raise FileExistsError("training already started in this output directory")
    recipe = json.loads(args.recipe.read_text())
    loaded = load_single_slit_checkpoint(Path(recipe["checkpoint"]), device="cpu")
    if loaded.checkpoint_sha256 != recipe["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    options = SkillCompositionOptions(**recipe["composition_options"])
    policy = loaded.policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    # Explicit training-only selection, not a filtered deployment success rate.
    replay = json.loads(args.replay_source.read_text())
    if replay.get("contact_residual") is not None:
        raise ValueError("source-preservation anchors must be the original policy")
    selected = [dict(r, trace=str((args.replay_source.parent / r["trace"]).resolve()))
                for r in replay["results"] if r["variant"] == args.replay_variant]
    selection = {"schema": "piano-movers.explicit-training-replay-selection.v1",
        "status": replay["status"], "split": replay["split"], "checkpoint_sha256": replay["checkpoint_sha256"],
        "source_manifest_sha256": sha256(args.replay_source), "source_total_cases": len(replay["results"]),
        "selection_variant": args.replay_variant, "not_a_deployment_report": True, "results": selected}
    selection_path = args.output / "replay_selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("replay selection changed")
    else:
        write_json(selection_path, selection)
    args.anchors = [selection_path]
    cache_path = args.output / "cache.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        with torch.no_grad():
            cache = prepare(args, loaded, options)
            for buckets in cache["groups"].values():
                for b in buckets:
                    policy.bind_environment(EnvConfig(**b["cfg"]))
                    features = b["arrays"]["features"]
                    parts = [contact_encoding_difference(policy, f) for f in features.split(128)]
                    b["arrays"]["contact_difference"] = torch.cat(parts)
        torch.save(cache, cache_path)
    if (cache["dataset_sha256"] != sha256(args.dataset) or cache["actor_sha256"] != loaded.checkpoint_sha256
            or cache["recipe_sha256"] != sha256(args.recipe)
            or cache["anchor_manifests"][0]["sha256"] != sha256(selection_path)):
        raise ValueError("cache provenance mismatch")
    counts = {k: {"all_frames": sum(len(b["arrays"]["features"]) for b in bs),
                  "contact_memory_frames": sum(int(b["arrays"]["features"][..., 12:20].any(-1).any(-1).sum()) for b in bs)}
              for k, bs in cache["groups"].items()}
    write_json(args.output / "preparation.json", {"status": "complete", "counts": counts,
        "cache_sha256": sha256(cache_path), "dataset_sha256": sha256(args.dataset),
        "maximum_force_replay_error_N": cache["maximum_force_error_N"],
        "feature_encoding_source_sha256": sha256(Path("src/piano_movers/contact_residual_control.py"))})
    print(json.dumps({"prepared": counts}), flush=True)
    if args.prepare_only:
        return
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name():
        raise RuntimeError("this training is authorized only on uk-linux RTX 5090")
    frozen = {n: p.detach().cpu().clone() for n, p in policy.named_parameters()}
    policy.cuda().eval()
    control = (ContactResidualControl(args.hidden_width) if args.initial_control is None else
               load_contact_residual(args.initial_control, loaded.checkpoint_sha256))
    if control.hidden_width != args.hidden_width:
        if control.hidden_width != 0 or args.hidden_width != 64:
            raise ValueError("only an exact zero-initialized linear-to-nonlinear bridge is supported")
        expanded = ContactResidualControl(args.hidden_width)
        expanded.readout.load_state_dict(control.readout.state_dict())
        expanded.feature_scale.copy_(control.feature_scale)
        expanded.twist_scale.copy_(control.twist_scale)
        control = expanded
    control.cuda()
    # Exclude only structurally zero-authority memory frames from fitting.
    # All source data and counts remain recorded above, without relabelling
    # these as task failures or dropping deployment episodes.
    groups = {}
    for kind, buckets in cache["groups"].items():
        groups[kind] = []
        for b in buckets:
            active = b["arrays"]["features"][..., 12:20].any(-1).any(-1)
            if active.any():
                groups[kind].append({"cfg": b["cfg"], "arrays": {k: v[active].cuda() for k, v in b["arrays"].items()}})
        if not groups[kind]:
            raise ValueError(f"no real contact memory in {kind}")
    differences = torch.cat([b["arrays"]["contact_difference"].reshape(-1, 256) for b in groups["teacher"]])
    if args.initial_control is None:
        control.feature_scale.copy_(differences.std(0).clamp_min(1e-3))
    optimizer = torch.optim.AdamW(control.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    weights = {k: np.array([len(b["arrays"]["features"]) for b in bs], float) for k, bs in groups.items()}
    weights = {k: w / w.sum() for k, w in weights.items()}
    record = {"schema": "piano-movers.contact-residual-training.v1", "status": "running",
        "release_status": "independent_control_not_promoted", "actor_sha256": loaded.checkpoint_sha256,
        "initial_control": None if args.initial_control is None else {
            "path": str(args.initial_control), "sha256": sha256(args.initial_control)},
        "dataset_sha256": sha256(args.dataset), "cache_sha256": sha256(cache_path),
        "replay_selection_sha256": sha256(selection_path), "counts": counts,
        "trainable_parameters": sum(p.numel() for p in control.parameters()), "hidden_width": control.hidden_width,
        "base_policy_frozen": True, "zero_contact_correction": "identically_zero",
        "physical_GBP_and_precision_unchanged": True, "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "anchor_weight": args.anchor_weight, "seed": args.seed,
        "gpu": torch.cuda.get_device_name(), "source_sha256": {str(p): sha256(p) for p in (
            Path(__file__), Path("src/piano_movers/models.py"), Path("src/piano_movers/actuation_training.py"),
            Path("src/piano_movers/contact_residual_control.py"))}, "history": [], "checkpoints": []}
    destination = args.output / "training.json"
    write_json(destination, record)

    def loss(kind):
        b = groups[kind][rng.choice(len(groups[kind]), p=weights[kind])]
        cfg = EnvConfig(**b["cfg"])
        if "buffers" in b:
            policy.env_cfg = cfg
            for name, tensor in b["buffers"].items():
                setattr(policy, name, tensor)
            policy.local_se2_analytic_gbp.num_steps = b["rounds"]
        else:
            policy.bind_environment(cfg)
            b["rounds"] = apply_solver_controls(policy, options)["rounds"]
        data = b["arrays"]
        ix = torch.randint(len(data["features"]), (args.batch_size,), device="cuda")
        batch = {k: v[ix] for k, v in data.items()}
        delta = control(batch["contact_difference"])
        action = replay_physical_actuation(policy, batch["obs"], batch["source_plan"],
            batch["geometry"], batch["active"], twist_residual=delta)
        b["buffers"] = {name: getattr(policy, name) for name in BUFFERS}
        return F.smooth_l1_loss(action.forces / cfg.max_force, batch["target_force"] / cfg.max_force, beta=.05) + .001 * delta.square().mean()

    started = time.monotonic()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        fit = loss("teacher")
        fit.backward()
        anchor = loss("replay")
        (args.anchor_weight * anchor).backward()
        torch.nn.utils.clip_grad_norm_(control.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            row = {"step": step, "teacher_force_loss": float(fit.detach()), "anchor_force_loss": float(anchor.detach()),
                   "elapsed_seconds": time.monotonic() - started}
            record["history"].append(row)
            print(json.dumps(row), flush=True)
            write_json(destination, record)
        if step % 500 == 0 or step == args.steps:
            if any(not torch.equal(p.detach().cpu(), frozen[n]) for n, p in policy.named_parameters()):
                raise RuntimeError("frozen base weights changed")
            path = args.output / f"control_step{step}.pt"
            torch.save({"schema": "piano-movers.contact-residual-control.v1", "actor_sha256": loaded.checkpoint_sha256,
                        "hidden_width": control.hidden_width,
                        "state_dict": {k: v.detach().cpu() for k, v in control.state_dict().items()},
                        "training": record, "step": step}, path)
            record["checkpoints"].append({"path": path.name, "sha256": sha256(path)})
    record.update(status="complete", elapsed_seconds=time.monotonic() - started)
    write_json(destination, record)


if __name__ == "__main__":
    main()
