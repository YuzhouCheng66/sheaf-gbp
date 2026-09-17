import hashlib
import json
from pathlib import Path
import runpy

import pytest


def test_dagger_aggregation_authenticates_legacy_actor_and_rejects_leakage(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    merge = runpy.run_path(str(scripts / "merge_mainline_dagger.py"))["merge_datasets"]

    def save(name, record):
        p = tmp_path / name
        p.write_text(json.dumps(record))
        return p

    proof = save("rollout.json", {"status": "complete", "checkpoint_sha256": "parent"})
    base = {
        "schema": "piano-movers.original-architecture-dagger.v1", "status": "complete",
        "split": "train_development_dagger", "parent_checkpoint_sha256": "parent",
        "observation_dim": 27, "plan_dim": 8, "supervised_plan_coordinates": list(range(6)),
        "heldout_shapes": ["S", "Z"], "heldout_counts": [9, 11, 13, 15, 17, 19],
        "rollout_manifest_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        # An unsuccessful teacher query must survive aggregation.
        "results": [{"actor_trace_sha256": "trace", "actor_tick": 10,
                     "case_id": "D_10_q10", "shape": "D", "agents": 10,
                     "admitted_for_training": False}],
    }
    legacy = save("legacy.json", base)
    with pytest.raises(ValueError, match="checksum-matched"):
        merge([legacy])
    merged = merge([legacy], [proof])
    assert len(merged["results"]) == 1
    assert not merged["results"][0]["admitted_for_training"]
    assert merged["results"][0]["actor_checkpoint_sha256"] == "parent"
    with pytest.raises(ValueError, match="duplicate"):
        merge([legacy, legacy], [proof])
    validation = save("validation.json", dict(base, split="locked_validation"))
    with pytest.raises(ValueError, match="development DAgger"):
        merge([validation], [proof])
    odd = save("odd.json", dict(base, results=[dict(base["results"][0], agents=9)]))
    with pytest.raises(ValueError, match="held-out"):
        merge([odd], [proof])

    # A frozen base plus a separately learned sidecar is a distinct actor;
    # preserve that identity rather than collapsing it into the base hash.
    controlled = save("controlled.json", dict(base, actor_checkpoint_sha256="parent",
        actor_contact_residual={"path": "control.pt", "sha256": "contact-control"}))
    union = merge([legacy, controlled], [proof])
    assert len(union["actor_identities"]) == 2
    assert union["results"][1]["actor_contact_residual"]["sha256"] == "contact-control"
