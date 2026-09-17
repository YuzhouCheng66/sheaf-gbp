#!/usr/bin/env python3
"""Compare complete legacy single-slit grids with only crossing hold changed."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from audit_local_skill_composition import sha256, write_json
from piano_movers.config import EnvConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before, after = (json.loads(p.read_text()) for p in (args.before, args.after))
    if before["status"] != "complete" or after["status"] != "complete":
        raise ValueError("both complete grids required")
    for key in ("schema", "checkpoint_sha256", "t_profile", "horizon", "continuous_contact_guard",
                "local_cspace_collision_substeps", "local_cspace_validate_all_lookaheads", "cover_coordination_tree_diameter"):
        if before[key] != after[key]:
            raise ValueError(f"unmatched {key}")
    if before.get("local_support_hold_feasible_crossing", False) or not after["local_support_hold_feasible_crossing"]:
        raise ValueError("expected disabled-source versus enabled-candidate")
    indexed = [{r["case_id"]: r for r in d["results"]} for d in (before, after)]
    expected = {f"{s}_{n}" for s in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for n in range(8, 21)}
    if any(len(d["results"]) != 338 or set(ix) != expected for d, ix in zip((before, after), indexed)):
        raise ValueError("both full unique A-Z by 8-20 grids required")
    rows = []
    for key in sorted(expected):
        old, new = (ix[key] for ix in indexed)
        a, b = (asdict(EnvConfig(**r["env_cfg"])) for r in (old, new))
        if a.pop("local_support_hold_feasible_crossing") or not b.pop("local_support_hold_feasible_crossing") or a != b:
            raise ValueError(f"additional environment changes: {key}")
        paths = [args.before.parent / old["artifact"], args.after.parent / new["artifact"]]
        for path, r in zip(paths, (old, new)):
            if sha256(path) != r["artifact_sha256"]:
                raise ValueError("trajectory checksum mismatch")
        with np.load(paths[0]) as x, np.load(paths[1]) as y:
            same = np.array_equal(x["states"], y["states"])
        rows.append({"case_id": key, "shape": new["shape"], "agents": new["agents"],
            "before_success": old["strict_success"], "after_success": new["strict_success"],
            "states_bitwise_equal": same, "before_steps": old["steps"], "after_steps": new["steps"],
            "before_contacts": old["contacts"], "after_contacts": new["contacts"],
            "before_payload_bytes": old["gbp_payload_bytes"], "after_payload_bytes": new["gbp_payload_bytes"],
            "after_certified": new["continuous_certificate"]["passed"],
            "after_penetrations": new["resolved_pose_collisions"]})
    summary = {"cases": len(rows), "before_successes": sum(r["before_success"] for r in rows),
        "after_successes": sum(r["after_success"] for r in rows),
        "regressions": sum(r["before_success"] and not r["after_success"] for r in rows),
        "bitwise_equal_state_paths": sum(r["states_bitwise_equal"] for r in rows),
        "after_certified": sum(r["after_certified"] for r in rows),
        "after_penetrations": sum(r["after_penetrations"] for r in rows)}
    report = {"status": "complete", "before_sha256": sha256(args.before), "after_sha256": sha256(args.after),
        "source_sha256": sha256(Path(__file__)), "summary": summary,
        "only_environment_change": "local_support_hold_feasible_crossing",
        "per_agent_successes": {str(n): sum(r["after_success"] for r in rows if r["agents"] == n) for n in range(8, 21)},
        "results": rows, "wire_headers_included": False,
        "scope": "fixed deterministic regression grid; state parity does not assert individual-force parity"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
