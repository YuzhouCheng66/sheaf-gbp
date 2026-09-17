#!/usr/bin/env python3
"""Summarize a complete reserved panel using layout clusters, not IID episodes."""

import argparse
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from audit_local_skill_composition import sha256, write_json
from piano_movers.local_skill_validation import STATE_PROFILES


def strict_success(row):
    return (bool(row["result"]["success"])
            and bool(row.get("continuous_certificate", {}).get("passed", False))
            and not row["result"].get("exact_penetration", False)
            and row.get("independent_resolved_pose_collisions", 0) == 0)


def failure_record(row):
    # Invalid start/goal poses legitimately have no rollout artifact. They
    # remain failures in the denominator, not reporting errors or omissions.
    return {k: row.get(k) for k in ("case_id", "shape", "agents", "seed", "profile", "result", "trace_sha256")}


def summarize(rows):
    if not rows:
        raise ValueError("cannot summarize an empty panel")
    seeds = sorted({r["seed"] for r in rows})
    clusters = [[r for r in rows if r["seed"] == seed] for seed in seeds]
    if len({len(group) for group in clusters}) != 1:
        raise ValueError("the intended layout clusters must have equal panel weights")
    rates = np.array([sum(strict_success(r) for r in group) / len(group) for group in clusters])
    # Four clusters permit exhaustive bootstrap enumeration, without Monte
    # Carlo noise. With only four draws this interval is necessarily fragile.
    means = np.array([rates[list(indices)].mean() for indices in product(range(len(seeds)), repeat=len(seeds))])
    half = math.sqrt(math.log(2. / .05) / (2. * len(seeds)))
    return {
        "episodes": len(rows), "strict_successes": sum(strict_success(r) for r in rows),
        "success_fraction": float(rates.mean()),
        "independent_layout_draws": len(seeds),
        "per_layout_fraction": dict(zip(map(str, seeds), rates.tolist())),
        "layout_cluster_bootstrap_percentile_95": np.quantile(means, [.025, .975]).tolist(),
        "layout_cluster_hoeffding_95": [max(0., float(rates.mean()) - half),
                                         min(1., float(rates.mean()) + half)],
        "continuous_certificates": sum(r.get("continuous_certificate", {}).get("passed", False) for r in rows),
        "resolved_pose_penetrations": sum(r.get("independent_resolved_pose_collisions", 0) for r in rows),
        "collision_attempts": sum(r["result"].get("contact_events", 0) for r in rows),
        "control_steps": sum(r["result"]["steps"] for r in rows),
        "path_length_m": sum(r.get("path_length_m", 0.) for r in rows),
        "gbp_float_payload_bytes": sum(r.get("gbp_float_payload_bytes", 0) for r in rows),
        "grasp_setup_payload_bytes": sum(r.get("grasp_setup_payload_bytes", 0) for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reused-panel", action="store_true",
                        help="Disclose prior inspection of these seeds with earlier candidates")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest["status"] != "complete" or manifest["split"] != "reserved_test":
        raise ValueError("requires complete reserved evaluation")
    for source in manifest["source_manifests"]:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("source shard has changed")
    rows = manifest["results"]
    if len(rows) != 1456 or {r["seed"] for r in rows} != {83201, 83202, 83203, 83204}:
        raise ValueError("unexpected reserved denominator/seeds")
    if {r["shape"] for r in rows} != {"S", "Z"}:
        raise ValueError("unexpected held-out shapes")
    if len({(r["seed"], r["shape"], r["agents"], r["profile"]) for r in rows}) != len(rows):
        raise ValueError("duplicate cases")
    expected = set(product(range(83201, 83205), ("S", "Z"), range(8, 21), STATE_PROFILES[1:]))
    if {(r["seed"], r["shape"], r["agents"], r["profile"]) for r in rows} != expected:
        raise ValueError("reserved panel is not the predeclared complete Cartesian product")
    report = {
        "status": "complete", "manifest_sha256": sha256(args.manifest),
        "report_source_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "recipe_sha256": manifest["recipe_sha256"],
        "overall": summarize(rows),
        "per_agent_count": {str(n): summarize([r for r in rows if r["agents"] == n]) for n in range(8, 21)},
        "per_shape": {shape: summarize([r for r in rows if r["shape"] == shape]) for shape in ("S", "Z")},
        "odd_agent_joint_ood": summarize([r for r in rows if r["agents"] % 2]),
        "failures": [failure_record(r) for r in rows if not strict_success(r)],
        "panel_reused": args.reused_panel,
        "blind_test_claim": False,
        "intervals_selection_adjusted": False,
        "wire_headers_included": False,
        "ancestral_p5_unseen_shape_claim": False,
        "uncertainty_scope": "four independent generator layout/goal draws with correlated shape/count/profile cases per draw",
        "uncertainty_caveat": "percentile bootstrap has only four clusters and is fragile; Hoeffding is conservative under independent bounded layout draws; neither certifies arbitrary mazes",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"overall": report["overall"], "joint_ood": report["odd_agent_joint_ood"],
                      "failures": report["failures"]}, indent=2))


if __name__ == "__main__":
    main()
