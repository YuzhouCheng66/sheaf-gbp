#!/usr/bin/env python3
"""Verify one-step mechanical continuity independently from latent-memory reset."""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_local_skill_composition import sha256, write_json
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete":
        raise ValueError("requires completed rollouts")
    loaded = load_single_slit_checkpoint(Path(source["checkpoint"]), device="cpu")
    if loaded.checkpoint_sha256 != source["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    rows = []
    for row in source["results"]:
        if "trace" not in row:
            continue
        if not row["options"]["preserve_contact_feedback"] or row["options"]["preserve_private_memory"]:
            raise ValueError("this audit expects physical carry with explicit private-memory reset")
        path = args.rollouts.parent / row["trace"]
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("trace mismatch")
        a = np.load(path)
        # These states are in the common simulator frame, not the changing
        # active-wall chart seen by the unchanged policy.
        np.testing.assert_array_equal(a["state"][1:], a["next_state"][:-1])
        switches = np.flatnonzero(np.diff(a["waypoint_index"]) != 0) + 1
        forces = a["forces"][switches - 1] / loaded.cfg.max_force
        np.testing.assert_array_equal(a["observation"][switches, :, 23:25], forces)
        hit = np.broadcast_to(a["hit"][switches - 1].reshape(-1, 1), (len(switches), row["agents"]))
        np.testing.assert_array_equal(a["observation"][switches, :, 22], hit)
        if bool(hit.any()):
            raise ValueError("unexpected contact at handoff; verify rejected-motion carry explicitly")
        assert np.count_nonzero(a["observation"][switches, :, 25:27]) == 0
        assert np.count_nonzero(a["private_hidden"][switches]) == 0
        rows.append({"case_id": row.get("case_id"), "shape": row["shape"], "agents": row["agents"],
                     "trace_sha256": row["trace_sha256"], "switches": len(switches),
                     "nonzero_force_switches": int(np.any(forces != 0, axis=(1, 2)).sum()),
                     "physical_state_and_feedback_continuous": True,
                     "goal_conditioned_hidden_explicitly_reset": True})
    report = {"status": "complete", "rollout_sha256": sha256(args.rollouts),
              "checkpoint_sha256": loaded.checkpoint_sha256, "results": rows,
              "switches_checked": sum(r["switches"] for r in rows),
              "scope": "one-step feedback is preserved; not a proof of long latent-memory transport"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
