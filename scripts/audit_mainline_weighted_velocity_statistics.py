#!/usr/bin/env python3
"""Audit a physical sufficient-statistic identity; do not change deployment.

True simulator twist is used ONLY to construct a diagnostic coasting command
requiring zero wrench. Uniform pooling of the current local sufficient
statistics should reproduce this identity. Heterogeneous precision pooling is
tested separately, before finite-round GBP or force clipping. This is not a
closed-loop oracle score or a replacement controller.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.multi_waypoint_demo import Slit, WaypointCourse, _course_config
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.local_wrench_factors import local_wrench_information_factor
from piano_movers.single_slit_local_planner import (
    local_se2_inverse_dynamics_wrench_proposals, LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS,
    LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS,
)


def coasting_plan(obs, twist, cfg):
    """Counterfactual common local-frame command, not a deployable observation."""
    sine, cosine = obs[..., 0], obs[..., 1]
    vx, vy, omega = [twist[..., i:i + 1] for i in range(3)]
    plan = obs.new_zeros((*obs.shape[:2], 8))
    plan[..., 3] = (cosine * vx + sine * vy) * (1. - cfg.linear_damping * cfg.dt) / LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
    plan[..., 4] = (-sine * vx + cosine * vy) * (1. - cfg.linear_damping * cfg.dt) / LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
    plan[..., 5] = omega * (1. - cfg.angular_damping * cfg.dt) / LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS
    return plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=["A", "C", "D", "I", "O", "T", "X"])
    parser.add_argument("--samples-per-episode", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    manifest = json.loads(args.rollouts.read_text())
    if manifest["status"] != "complete":
        raise ValueError("requires complete authentic rollout evidence")
    loaded = load_single_slit_checkpoint(Path(manifest["checkpoint"]), device="cpu")
    if loaded.checkpoint_sha256 != manifest["checkpoint_sha256"]:
        raise ValueError("actor checksum mismatch")
    boost = loaded.policy.arm_cfg.local_se2_geometry_evidence_precision_boost
    rows, errors = [], []
    factor_max = 0.
    for row in manifest["results"]:
        if row["shape"] not in args.shapes or "trace" not in row:
            continue
        path = args.rollouts.parent / row["trace"]
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("trace checksum mismatch")
        arrays, h = np.load(path), row["header"]
        course = WaypointCourse(tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
                                tuple(tuple(g) for g in h["waypoints"]), h["world"]["x"], h["world"]["y"],
                                tuple(h.get("source_velocity", (0., 0., 0.))), h.get("morphology_scale", 1.))
        cfg = _course_config(loaded.cfg, row["agents"], course, row["shape"])
        if row["options"]["nominal_glyph_size"]:
            cfg = replace(cfg, glyph_scale=1.)
        ids = np.unique(np.linspace(0, len(arrays["state"]) - 1, args.samples_per_episode, dtype=int))
        obs = torch.from_numpy(arrays["observation"][ids])
        twist = torch.from_numpy(arrays["state"][ids, 3:6])
        command = coasting_plan(obs, twist, cfg)
        proposed = local_se2_inverse_dynamics_wrench_proposals(obs, command, cfg)
        active = torch.from_numpy(arrays["analytic_active"][ids]).reshape(len(ids), row["agents"])
        precision = 1. + (boost - 1.) * active.to(proposed.dtype)
        uniform = proposed.mean(1)
        weighted = (precision[..., None] * proposed).sum(1) / precision.sum(1, keepdim=True)
        information, natural = local_wrench_information_factor(obs, command, cfg, precision)
        # Central aggregation is used only for this algebraic diagnostic;
        # no deployable GBP or communication-efficiency claim is made.
        factor_estimate = torch.linalg.solve(information.sum(1), natural.sum(1).unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(factor_estimate, torch.zeros_like(factor_estimate), atol=2e-5, rtol=0.)
        factor_max = max(factor_max, float(factor_estimate.abs().max()))
        mixed = precision.amax(1) != precision.amin(1)
        moving = torch.linalg.vector_norm(twist, dim=-1) > .05
        selected = mixed & moving
        torch.testing.assert_close(uniform, torch.zeros_like(uniform), atol=2e-5, rtol=0.)
        chosen = weighted[selected].numpy()
        errors.extend(chosen.tolist())
        worst = int(weighted.abs().amax(-1).argmax())
        rows.append({"case_id": row.get("case_id"), "shape": row["shape"],
                     "trace_sha256": row["trace_sha256"], "sample_count": len(ids),
                     "mixed_precision_moving_samples": int(selected.sum()),
                     "uniform_max_abs_normalized_wrench": float(uniform.abs().max()),
                     "weighted_max_abs_normalized_wrench": float(weighted.abs().max()),
                     "weighted_local_factor_coasting_max_abs": float(factor_estimate.abs().max()),
                     "worst_tick": int(ids[worst]), "worst_twist": twist[worst].tolist(),
                     "worst_coasting_wrench_normalized": weighted[worst].tolist(),
                     "worst_unary_precisions": precision[worst].tolist()})
    if not errors:
        raise ValueError("no mixed-confidence moving samples")
    values = np.abs(np.array(errors))
    result = {"status": "complete", "checkpoint_sha256": loaded.checkpoint_sha256,
              "rollout_sha256": sha256(args.rollouts), "precision_boost": boost,
              "simulator_twist_used_for_diagnostic_only": True,
              "finite_round_GBP_and_force_clipping_tested": False,
              "weighted_local_factor_coasting_max_abs": factor_max,
              "mixed_precision_moving_samples": len(errors),
              "normalized_wrench_abs_mean": values.mean(0).tolist(),
              "normalized_wrench_abs_p95": np.quantile(values, .95, axis=0).tolist(),
              "normalized_wrench_abs_max": values.max(0).tolist(), "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
