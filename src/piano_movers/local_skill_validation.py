"""Predeclared course perturbations; these are supervisor inputs, not policy features.

This first family varies poses and goals at zero initial velocity. It does not
claim a velocity/contact-entry, rotated-wall, or full SE(2)-equivariance test.
No outcome-based rejection, teacher admission, or letter-dependent layout is
used when drawing a course.
"""

from dataclasses import replace
import math
import random

from .multi_waypoint_demo import DEFAULT_COURSE, Slit, WaypointCourse


LAYOUT_FAMILIES = ("layout", "entry", "goal", "combined")
FAMILY_SEEDS = {
    "development_diagnostics": (81201, 81202),
    "locked_validation": (82201, 82202),
    "reserved_test": (83201, 83202, 83203, 83204),
}
STATE_PROFILES = (
    "nominal", "entry_reference", "heading_neg180", "heading_neg135", "heading_neg90",
    "heading_neg45", "heading_pos45", "heading_pos90", "heading_pos135",
    "velocity_forward", "velocity_backward", "joint_positive", "joint_negative",
    "shape_small", "shape_large",
)
ENTRY_HEADINGS_DEG = {
    "heading_neg180": -180, "heading_neg135": -135, "heading_neg90": -90,
    "heading_neg45": -45, "heading_pos45": 45, "heading_pos90": 90, "heading_pos135": 135,
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


def state_profile_course(seed: int, family: str, profile: str = "nominal") -> WaypointCourse:
    """Stratified large-angle and moving-entry controls, independent of outcomes.

    All non-nominal profiles use the same 1.25 m source-to-wall distance,
    including entry_reference. This leaves room to rotate the declared glyphs
    away from the outer boundary without outcome-based resampling. Geometry
    scales affect body and handles, not mass, force limits or communication
    radius. The large-scale profile scales the physical opening to avoid
    silently comparing a larger body to a provably impossible nominal gap.
    It remains subject to independent feasibility/trajectory checks.
    """
    if profile not in STATE_PROFILES:
        raise ValueError(f"unknown state profile: {profile}")
    course = validation_course(seed, family)
    if profile == "nominal":
        return course
    source = (course.walls[0].x - 1.25, course.source[1], course.source[2])
    course = replace(course, source=source)
    if profile in ENTRY_HEADINGS_DEG:
        course = replace(course, source=(*source[:2], math.radians(ENTRY_HEADINGS_DEG[profile])))
    if profile == "velocity_forward":
        course = replace(course, source_velocity=(.25, .10, .30))
    if profile == "velocity_backward":
        course = replace(course, source_velocity=(-.20, -.10, -.30))
    if profile == "joint_positive":
        course = replace(course, source=(*source[:2], .90), source_velocity=(.20, -.12, -.25))
    if profile == "joint_negative":
        course = replace(course, source=(*source[:2], -.90), source_velocity=(-.20, .12, .25))
    if profile in ("shape_small", "shape_large"):
        scale = .90 if profile == "shape_small" else 1.04
        course = replace(course, morphology_scale=scale)
        if scale > 1.:
            course = replace(course, walls=tuple(
                replace(w, aperture_m=max(w.aperture_m, (.720 - .060) * scale + .060))
                for w in course.walls
            ))
    course.validate()
    return course


def sequence_length_course(seed: int, profile: str, wall_count: int) -> WaypointCourse:
    """New matched 1/2/3/5-wall panel; fixed arena/normalization across lengths.

    Independent random substreams generate each wall and command, so a longer
    course extends an identical local prefix. N/shape/outcomes never enter the
    generator. The 7 m half-arena is an explicit normalization shift vs old
    three-wall tests; the three-wall member isolates that nuisance change.
    The policy still receives only its current local wall and current command.
    """
    if wall_count not in (1, 2, 3, 5):
        raise ValueError("sequence audit supports the predeclared 1/2/3/5-wall lengths")
    reference = state_profile_course(seed, "combined", profile)
    layout_rng = random.Random(seed * 17 + 810)
    target_rng = random.Random(seed * 17 + 811)
    x = -3.5
    walls, waypoints = [], []
    for index in range(wall_count):
        if index:
            x += layout_rng.uniform(1.65, 1.95)
        aperture = layout_rng.choice((.720, .800, 1.000))
        if reference.morphology_scale > 1.:
            aperture = max(aperture, .660 * reference.morphology_scale + .060)
        wall = Slit(x, layout_rng.uniform(-.22, .22), aperture)
        walls.append(wall)
        waypoints.append((x + .63 + target_rng.uniform(-.04, .10),
                          wall.gap_y + target_rng.uniform(-.10, .10), 0.))
    source_distance = reference.walls[0].x - reference.source[0]
    course = replace(reference, source=(walls[0].x - source_distance, *reference.source[1:]),
                     walls=tuple(walls), waypoints=(*waypoints, (walls[-1].x + 1.44,
                                                               *reference.waypoints[-1][1:])),
                     world_x=7., world_y=1.6)
    course.validate()
    return course
