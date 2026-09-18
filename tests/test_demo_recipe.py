from dataclasses import asdict
import hashlib
import json

import pytest

from piano_movers.multi_waypoint_demo import SkillCompositionOptions, load_demo_recipe


def recipe_fixture(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test checkpoint bytes; no model deserialization needed")
    recipe = {"checkpoint": "weights.pt",
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "composition_options": asdict(SkillCompositionOptions(
                  continuous_contact_guard=True, hold_feasible_crossing=True,
                  cover_coordination_tree_diameter=True, resolve_position_goal_heading=True))}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return checkpoint, recipe, path


def test_demo_loads_exact_recipe_without_default_fallback(tmp_path):
    checkpoint, recipe, path = recipe_fixture(tmp_path)
    loaded, options, info = load_demo_recipe(path, tmp_path)
    assert loaded == checkpoint
    assert asdict(options) == recipe["composition_options"]
    assert info["recipe_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info["mode"] == "authenticated_recipe"


def test_demo_rejects_checkpoint_mismatch(tmp_path):
    checkpoint, _, path = recipe_fixture(tmp_path)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_demo_recipe(path, tmp_path)


def test_demo_rejects_omitted_control(tmp_path):
    _, recipe, path = recipe_fixture(tmp_path)
    del recipe["composition_options"]["hold_feasible_crossing"]
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="every composition option"):
        load_demo_recipe(path, tmp_path)
