#!/usr/bin/env python3
"""Emit an explicit, unpromoted evaluation recipe for a completed input fit."""

import argparse
import json
from pathlib import Path

from audit_local_skill_composition import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--source-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fit = json.loads(args.training.read_text())
    source = json.loads(args.source_recipe.read_text())
    if fit["status"] != "complete" or fit["step"] != fit["steps_requested"]:
        raise ValueError("requires the declared final checkpoint, not selected intermediate weights")
    if fit["recipe_sha256"] != sha256(args.source_recipe):
        raise ValueError("source deployment recipe mismatch")
    if fit["initial_checkpoint_sha256"] != source["checkpoint_sha256"]:
        raise ValueError("source actor mismatch")
    checkpoint = (args.training.parent / fit["checkpoint"]["path"]).resolve(strict=True)
    if sha256(checkpoint) != fit["checkpoint"]["sha256"]:
        raise ValueError("trained checkpoint changed")
    recipe = dict(source)
    recipe.update(
        checkpoint=str(checkpoint), checkpoint_sha256=fit["checkpoint"]["sha256"],
        release_status="independent_goal_input_control_not_promoted",
        training_manifest=str(args.training.resolve()), training_sha256=sha256(args.training),
        source_recipe_sha256=sha256(args.source_recipe),
        source_checkpoint_sha256=source["checkpoint_sha256"],
        input_column3=fit["input_column3"],
        additional_private_goal_inputs=4 if fit["goal_context"] else 0,
        trainable_parameters=fit["trainable_parameters"],
        additional_parameters=fit["additional_parameters"],
        control_description="Matched original-MLP force training; declared goal-input semantics only. "
                            "Geometry interfaces, physical GBP and grasp allocation retained.",
        materializer_sha256=sha256(Path(__file__)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, recipe)
    print(json.dumps({"recipe": str(args.output), "checkpoint_sha256": recipe["checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
