#!/usr/bin/env python3
"""Merge complete disjoint audit shards without copying or relabelling traces."""

import argparse
import json
from pathlib import Path

from audit_local_skill_composition import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    record, keys, rows = None, set(), []
    frozen_grid = None
    layout_grid = None
    for path in args.input:
        source = json.loads(path.read_text())
        if source["status"] != "complete":
            raise ValueError("cannot merge an unfinished audit")
        if record is None:
            record = dict(source)
            frozen_grid = source["schema"] == "piano-movers.frozen-single-slit-grid.v1"
            layout_grid = source["schema"] == "piano-movers.mainline-layout-validation.v1"
        invariant = (("schema", "checkpoint_sha256", "source_sha256", "t_profile", "horizon",
                      "continuous_contact_guard", "local_cspace_collision_substeps")
                     if frozen_grid else ("schema", "split", "checkpoint_sha256", "source_sha256", "device"))
        for key in invariant:
            if source[key] != record[key]:
                raise ValueError(f"mixed audit semantics: {key}")
        if frozen_grid and (source.get("local_cspace_validate_all_lookaheads", False)
                            != record.get("local_cspace_validate_all_lookaheads", False)):
            raise ValueError("mixed nominal rejoin-screen settings")
        if layout_grid:
            for key in ("recipe_sha256", "options", "initial_velocity", "outcome_based_admission"):
                if source[key] != record[key]:
                    raise ValueError(f"mixed validation recipe: {key}")
            catalog = path.parent / "case_catalog.json"
            if sha256(catalog) != source["catalog_sha256"]:
                raise ValueError("validation catalog checksum mismatch")
            requested = json.loads(catalog.read_text())
            expected = {row["case_id"] for row in requested["cases"]}
            actual = {row["case_id"] for row in source["results"]}
            if actual != expected or len(actual) != len(source["results"]):
                raise ValueError("validation results differ from predeclared denominator")
        for row in source["results"]:
            key = (row["case_id"] if layout_grid else
                   (row.get("variant", "nominal"), row["shape"], row["agents"]))
            if key in keys:
                raise ValueError(f"duplicate case: {key}")
            field = "artifact" if frozen_grid else "trace"
            if field in row:
                trace = (path.parent / row[field]).resolve()
                if sha256(trace) != row[field + "_sha256"]:
                    raise ValueError("trace checksum mismatch")
                row = dict(row, **{field: str(trace)})
            elif not layout_grid or row["result"].get("reason") != "invalid_initial_or_final_pose":
                raise ValueError("unexpected missing trajectory")
            keys.add(key)
            rows.append(row)
    if len(rows) != args.expected_cases:
        raise ValueError("case count differs from the declared panel")
    record.update(results=rows, requested={
        "shapes": sorted({r["shape"] for r in rows}),
        "agents": sorted({r["agents"] for r in rows}),
        "variants": sorted({r.get("variant", "nominal") for r in rows}),
    }, source_manifests=[{"path": str(p.resolve()), "sha256": sha256(p)} for p in args.input])
    if layout_grid:
        record.update(catalog_sha256=None, requested_cases=len(rows))
        record["requested"].update(
            seeds=sorted({r["seed"] for r in rows}), families=sorted({r["family"] for r in rows}),
            profiles=sorted({r.get("profile", "nominal") for r in rows}),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, record)
    print(json.dumps({"cases": len(rows), "successes": sum(
        r["strict_success"] if frozen_grid else r["result"]["success"] for r in rows
    )}))


if __name__ == "__main__":
    main()
