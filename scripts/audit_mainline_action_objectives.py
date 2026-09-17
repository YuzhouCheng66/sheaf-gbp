#!/usr/bin/env python3
"""Deployed-force errors, separating zero trainable-morphology authority."""

import argparse
import json
from pathlib import Path

import torch

from audit_local_skill_composition import sha256, write_json
from train_mainline_action_dagger import private_from_prediction
from piano_movers.actuation_training import replay_physical_actuation
from piano_movers.config import EnvConfig
from piano_movers.multi_waypoint_demo import SkillCompositionOptions, apply_solver_controls
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    recipe = json.loads(args.recipe.read_text())
    if sha256(args.recipe) != cache["recipe_sha256"]:
        raise ValueError("cached actuation recipe changed")
    options = SkillCompositionOptions(**recipe["composition_options"])
    results = []
    for checkpoint in args.checkpoint:
        loaded = load_single_slit_checkpoint(checkpoint, device=args.device)
        actor = cache["actor_sha256"]
        if loaded.checkpoint_sha256 != actor and loaded.metadata.get("p6_mainline_training", {}).get("initial_checkpoint_sha256") != actor:
            raise ValueError("checkpoint does not descend from the cached source")
        totals = {}
        for kind, buckets in cache["groups"].items():
            accumulators = {label: {"frames": 0, "force_components": 0,
                                    "abs_error_sum_N": 0., "squared_error_sum_N2": 0.,
                                    "normalized_huber_sum": 0.}
                            for label in ("all", "no_trainable_authority", "some_trainable_authority")}
            for bucket in buckets:
                cfg = EnvConfig(**bucket["cfg"])
                loaded.policy.bind_environment(cfg)
                apply_solver_controls(loaded.policy, options)
                arrays = bucket["arrays"]
                for start in range(0, len(arrays["features"]), 256):
                    batch = {k: v[start:start + 256].to(args.device) for k, v in arrays.items()}
                    encoded = loaded.policy.local_se2_markov_shape_encode(batch["features"])
                    prediction = torch.tanh(loaded.policy.local_se2_markov_shape_mean_head(encoded))
                    plan = private_from_prediction(prediction, batch["source_plan"], batch["source_morph"], batch["support"])
                    action = replay_physical_actuation(loaded.policy, batch["obs"], plan, batch["geometry"], batch["active"])
                    error = action.forces - batch["target_force"]
                    potential = ((~batch["active"]) & (batch["support"] > 0.)).any(dim=(1, 2))
                    masks = {"all": torch.ones_like(potential), "no_trainable_authority": ~potential,
                             "some_trainable_authority": potential}
                    for label, mask in masks.items():
                        selected = error[mask]
                        acc = accumulators[label]
                        acc["frames"] += int(mask.sum())
                        acc["force_components"] += selected.numel()
                        acc["abs_error_sum_N"] += float(selected.abs().sum())
                        acc["squared_error_sum_N2"] += float(selected.square().sum())
                        normalized = selected / cfg.max_force
                        acc["normalized_huber_sum"] += float(torch.nn.functional.smooth_l1_loss(
                            normalized, torch.zeros_like(normalized), beta=.05, reduction="sum"))
            for acc in accumulators.values():
                count = max(1, acc["force_components"])
                acc["mae_N"] = acc["abs_error_sum_N"] / count
                acc["rmse_N"] = (acc["squared_error_sum_N2"] / count) ** .5
                acc["normalized_huber_mean"] = acc["normalized_huber_sum"] / count
            totals[kind] = accumulators
        results.append({"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": loaded.checkpoint_sha256,
                        "errors": totals})
        print(json.dumps(results[-1]), flush=True)
    report = {"status": "complete", "cache_sha256": sha256(args.cache),
              "recipe_sha256": sha256(args.recipe), "results": results,
              "scope": "off-policy force errors on development training/replay frames; not closed-loop success",
              "authority_caveat": "potential influence of the trained morphology head, not full force Jacobian rank; frozen canonical neural branches are not trainable authority"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
