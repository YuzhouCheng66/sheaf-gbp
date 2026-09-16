from dataclasses import replace

import torch

from piano_movers.multi_waypoint_demo import (
    DEFAULT_COURSE,
    _body_cleared_wall,
    _active_observation_scenario,
    _composition_observation,
    _course_config,
    _intermediate_reached,
)
from piano_movers.payload_shapes import payload_rectangles_for_config
from piano_movers.single_slit_feedback import single_slit_env_config
from piano_movers.single_slit_local_planner import local_se2_terminal_attractor_plan
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observations_v1,
)


def test_b_handoff_reproduction_keeps_the_trailing_wall() -> None:
    cfg = _course_config(single_slit_env_config(), 12, DEFAULT_COURSE, "B")
    # Actual frozen-p5 B/12 state at the premature first handoff, step 208.
    state = torch.tensor([[-1.28131914, -0.15365972, -1.57079589, 0.499967, 0.000967, 0.]])
    rectangles = payload_rectangles_for_config(cfg, "cpu")
    assert _intermediate_reached(state, DEFAULT_COURSE.waypoints[0])
    assert not _body_cleared_wall(state, rectangles, DEFAULT_COURSE.walls[0], cfg)
    state[:, 0] += 0.20
    assert _body_cleared_wall(state, rectangles, DEFAULT_COURSE.walls[0], cfg)


def test_position_only_goal_ignores_hidden_angle_but_pose_goal_rotates() -> None:
    cfg = single_slit_env_config()
    # A self-consistent local handle row at payload centre x=1, theta=0.6,
    # wholly beyond the wall; the terminal physical controller is active.
    obs = torch.zeros(1, cfg.n_agents, 27)
    obs[..., 0] = torch.sin(torch.tensor(0.6))
    obs[..., 1] = torch.cos(torch.tensor(0.6))
    obs[..., 4] = 1.0 / 2.6
    plan = torch.zeros(1, cfg.n_agents, 8)

    def run(required: bool, heading: float):
        return local_se2_terminal_attractor_plan(
            obs, plan, replace(cfg, goal_orientation_required=required, goal_theta=heading),
            position_gain=2., angle_gain=2., max_speed=.35, max_omega=.8,
            feasible_corridor=False,
        )

    free_a, active = run(False, -1.0)
    free_b, _ = run(False, 2.0)
    fixed, _ = run(True, -1.0)
    assert active.all()
    torch.testing.assert_close(free_a, free_b, rtol=0, atol=0)
    assert torch.count_nonzero(free_a[..., (2, 5)]) == 0
    assert torch.count_nonzero(fixed[..., (2, 5)]) > 0


def test_optional_neighbor_sensor_preserves_legacy_and_retains_previous_wall() -> None:
    cfg = _course_config(single_slit_env_config(), 12, DEFAULT_COURSE, "B")
    state = torch.tensor([[-1.28131914, -0.15365972, -1.57079589, 0.499967, 0.000967, 0.]])
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    context.previous_force_normalized.fill_(0.2)
    local, old_scene = _active_observation_scenario(DEFAULT_COURSE, state, 1)
    expected = single_slit_contact_observations_v1(local, old_scene, cfg, context)
    legacy = _composition_observation(DEFAULT_COURSE, state, 1, cfg, context)
    torch.testing.assert_close(legacy, expected, rtol=0, atol=0)
    retained = _composition_observation(
        DEFAULT_COURSE, state, 1, cfg, context, retain_nearby_wall=True,
    )
    assert retained.shape == legacy.shape == (1, 12, 27)
    assert not bool(legacy[..., 17].any())
    assert bool(retained[..., 17].any())
    # Local previous force and rejected-motion data are not scene coordinates.
    torch.testing.assert_close(retained[..., 22:], legacy[..., 22:], rtol=0, atol=0)
    # The active wall record remains in exactly the original slots.
    torch.testing.assert_close(retained[..., [10, 12, 14, 16]], legacy[..., [10, 12, 14, 16]], rtol=0, atol=0)


def test_neighbor_sensor_cannot_see_future_goals() -> None:
    cfg = _course_config(single_slit_env_config(), 12, DEFAULT_COURSE, "B")
    state = torch.tensor([[-1.28131914, -0.15365972, -1.57079589, 0., 0., 0.]])
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    altered = replace(DEFAULT_COURSE, waypoints=(
        *DEFAULT_COURSE.waypoints[:2], (2.8, 0.9, 2.), (3.4, -0.7, 1.),
    ))
    first = _composition_observation(DEFAULT_COURSE, state, 1, cfg, context, retain_nearby_wall=True)
    second = _composition_observation(altered, state, 1, cfg, context, retain_nearby_wall=True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_previous_wall_retention_is_bitwise_unchanged_before_first_handoff() -> None:
    cfg = _course_config(single_slit_env_config(), 12, DEFAULT_COURSE, "B")
    state = torch.tensor([[-1.28131914, -0.15365972, -1.57079589, 0., 0., 0.]])
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    old = _composition_observation(DEFAULT_COURSE, state, 0, cfg, context)
    previous = _composition_observation(
        DEFAULT_COURSE, state, 0, cfg, context, retain_previous_wall_only=True,
    )
    torch.testing.assert_close(old, previous, rtol=0, atol=0)
    post = _composition_observation(
        DEFAULT_COURSE, state, 1, cfg, context, retain_previous_wall_only=True,
    )
    assert bool(post[..., 17].any())
