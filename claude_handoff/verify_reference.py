"""Read-only source authentication and optional CPU loading of pinned anchors.

No training, CUDA initialization, optimizer/RNG restoration or artifact writes.
Only load this project's trusted checkpoints after their exact file SHA passes.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

SOURCE_SHA = "a0154cb73488ab3bf7f3f34f15d193af6133d422587e46654103835c99185d8d"
ANCHORS = {
    "v310_e12": {
        "file_sha256": "31da2f3962c1a102367088dbcdd4063d11bbcb77ac4f75f11d91f95efefeeb3a",
        "model_sha256": "d1d397b654e6e108b23e1fdabf8f6bb2588d4756835cb9b9d66219be1a369203",
        "module": "scripts.mapping_multiscale_v299_v310",
        "class": "MultiscaleV299",
        "completed_validation_epoch": 12,
    },
    "v299_control": {
        "file_sha256": "28cbc27ca76221c87810508bffd880a32f20c4a2a8a6cd614ee2ceaec165ccb1",
        "model_sha256": "3f52478660b73b2ba6b2032b9a79510be5a7d081ee9a22b7b3c27222e70008db",
        "module": "scripts.mapping_content_readout_v288",
        "class": "ContentReadout64",
        "validated_step": 2000,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=here / "reference_source_v310")
    parser.add_argument("--source-manifest", type=Path,
                        default=here / "source-manifest-v310.json")
    parser.add_argument("--anchor", choices=tuple(ANCHORS))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    require(bool(args.anchor) == bool(args.checkpoint),
            "--anchor and --checkpoint must be provided together")
    source = args.source.resolve(strict=True)
    require(sha256(args.source_manifest) == SOURCE_SHA, "Wrong source manifest SHA")
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    require(len(manifest["files"]) == 70, "Expected the sealed 70-file source")
    for relative, expected in manifest["files"].items():
        path = (source / relative).resolve(strict=True)
        require(path.is_relative_to(source), "Manifest path escapes source root")
        require(sha256(path) == expected, "Source SHA mismatch: " + relative)
    report = dict(source=str(source), source_manifest_sha256=SOURCE_SHA,
                  source_files_verified=70, checkpoint_loaded=False,
                  trained=False, device="CPU only", artifacts_written=False)
    if args.checkpoint:
        pin = ANCHORS[args.anchor]
        checkpoint = args.checkpoint.resolve(strict=True)
        require(sha256(checkpoint) == pin["file_sha256"], "Wrong checkpoint SHA")
        # Insert the authenticated source before mutable workspace modules.
        sys.path[:0] = [str(source), str(source / "exploration_grid")]
        import torch
        torch.set_num_threads(1)
        module = importlib.import_module(pin["module"])
        require(Path(module.__file__).resolve().is_relative_to(source),
                "Model module did not load from the authenticated snapshot")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_payload = payload["model"] if args.anchor == "v310_e12" else payload
        model = getattr(module, pin["class"]).from_payload(model_payload).eval()
        require(all(p.device.type == "cpu" for p in model.parameters()),
                "Unexpected non-CPU model")
        model_sha = model.weights_hash()
        require(model_sha == pin["model_sha256"], "Model tensor-state SHA mismatch")
        report.update(anchor=args.anchor, checkpoint=str(checkpoint),
                      checkpoint_sha256=pin["file_sha256"],
                      model_sha256=model_sha, checkpoint_loaded=True,
                      model_module_file=module.__file__,
                      stored_parameters=sum(p.numel() for p in model.parameters()),
                      torch_version=torch.__version__,
                      **{k: pin[k] for k in ("completed_validation_epoch", "validated_step")
                         if k in pin})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
