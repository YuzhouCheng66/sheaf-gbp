from dataclasses import asdict

import pytest

from piano_movers.local_skill_validation import (
    FAMILY_SEEDS, LAYOUT_FAMILIES, validate_split_seeds, validation_course,
)
from piano_movers.multi_waypoint_demo import DEFAULT_COURSE


def test_validation_seed_families_are_disjoint_and_deterministic():
    all_seeds = [seed for seeds in FAMILY_SEEDS.values() for seed in seeds]
    assert len(all_seeds) == len(set(all_seeds))
    for seed in all_seeds:
        for family in LAYOUT_FAMILIES:
            first = validation_course(seed, family)
            first.validate()
            assert asdict(first) == asdict(validation_course(seed, family))
            assert all(w.aperture_m >= .720 for w in first.walls)


def test_isolated_controls_do_not_change_other_components():
    seed = 82201
    entry = validation_course(seed, "entry")
    goal = validation_course(seed, "goal")
    layout = validation_course(seed, "layout")
    combined = validation_course(seed, "combined")
    assert entry.walls == goal.walls == DEFAULT_COURSE.walls
    assert entry.waypoints == DEFAULT_COURSE.waypoints
    assert goal.source == DEFAULT_COURSE.source
    assert layout.source[1:] == DEFAULT_COURSE.source[1:]
    assert combined.walls == layout.walls
    assert combined.source[1:] == entry.source[1:]
    assert combined.waypoints[-1][1:] == goal.waypoints[-1][1:]
    assert layout.walls != DEFAULT_COURSE.walls
    assert entry.source != DEFAULT_COURSE.source
    assert goal.waypoints != DEFAULT_COURSE.waypoints


def test_split_checks_prevent_accidental_seed_relabeling():
    validate_split_seeds("locked_validation", [82201])
    for split, seeds in (("locked_validation", [81201]),
                         ("locked_validation", [82201, 82201]),
                         ("typo", [82201])):
        with pytest.raises(ValueError):
            validate_split_seeds(split, seeds)
