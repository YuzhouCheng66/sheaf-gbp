"""Predeclared course perturbations; these are supervisor inputs, not policy features.

This first family varies poses and goals at zero initial velocity. It does not
claim a velocity/contact-entry, rotated-wall, or full SE(2)-equivariance test.
No outcome-based rejection, teacher admission, or letter-dependent layout is
used when drawing a course.
"""

from dataclasses import replace
import random

from .multi_waypoint_demo import DEFAULT_COURSE, Slit, WaypointCourse


LAYOUT_FAMILIES = ("layout", "entry", "goal", "combined")
FAMILY_SEEDS = {
    "development_diagnostics": (81201, 81202),
    "locked_validation": (82201, 82202),
    "reserved_test": (83201, 83202, 83203, 83204),
}


def validation_course(seed: int, family: str) -> WaypointCourse:
    """Draw matched perturbations with independent, stable RNG substreams.

The same seed gives the same layout/entry/goal perturbation in an isolated
family and the combined family. Shapes and agent counts never affect the draw.
"""
    if family not in LAYOUT_FAMILIES:
        raise ValueError(f"unknown validation family: {family}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    layout_rng = random.Random(seed * 3)
    entry_rng = random.Random(seed * 3 + 1)
    goal_rng = random.Random(seed * 3 + 2)
    course = DEFAULT_COURSE
    if family in ("layout", "combined"):
        x = -1.8 + layout_rng.uniform(-.15, .15)
        walls = []
        for index in range(3):
            if index:
                x += layout_rng.uniform(1.65, 1.95)
            walls.append(Slit(x, layout_rng.uniform(-.22, .22),
                              layout_rng.choice((.720, .800, 1.000))))
        waypoints = tuple((wall.x + .63, wall.gap_y, 0.) for wall in walls)
        course = WaypointCourse(
            source=(walls[0].x - 1.4, 0., 0.), walls=tuple(walls),
            waypoints=(*waypoints, (walls[-1].x + 1.44, 0., 0.)),
            world_x=4.5, world_y=1.6,
        )
    if family in ("entry", "combined"):
        course = replace(course, source=(
            course.source[0] + entry_rng.uniform(-.10, .10),
            course.source[1] + entry_rng.uniform(-.20, .20),
            entry_rng.uniform(-.65, .65),
        ))
    if family in ("goal", "combined"):
        intermediate = tuple(
            (goal[0] + goal_rng.uniform(-.04, .10),
             goal[1] + goal_rng.uniform(-.10, .10), 0.)
            for goal in course.waypoints[:-1]
        )
        final = (course.waypoints[-1][0], goal_rng.uniform(-.25, .25),
                 goal_rng.uniform(-.8, .8))
        course = replace(course, waypoints=(*intermediate, final))
    course.validate()
    return course


def validate_split_seeds(split: str, seeds: list[int]) -> None:
    if split not in FAMILY_SEEDS or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("invalid split or duplicate/empty seed list")
    if not set(seeds) <= set(FAMILY_SEEDS[split]):
        raise ValueError("seeds do not belong to the predeclared split")
