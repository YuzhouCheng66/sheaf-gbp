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


def test_course_twist_and_scale_are_physical_not_extra_policy_features():
    from piano_movers.multi_waypoint_demo import _full_collision_scenario

    base_cfg = single_slit_env_config()
    original = _course_config(base_cfg, 12, DEFAULT_COURSE, "T")
    course = replace(DEFAULT_COURSE, source_velocity=(.2, -.1, .3), morphology_scale=.95)
    course.validate()
    cfg = _course_config(base_cfg, 12, course, "T")
    scene = _full_collision_scenario(course, cfg)
    torch.testing.assert_close(scene.state0[0, 3:], torch.tensor([.2, -.1, .3]))
    old_geometry = payload_rectangles_for_config(original, "cpu")
    new_geometry = payload_rectangles_for_config(cfg, "cpu")
    torch.testing.assert_close(new_geometry, old_geometry * .95)
    context = initial_single_slit_contact_observation_context_v1(scene.state0, cfg)
    obs = _composition_observation(course, scene.state0, 0, cfg, context)
    assert obs.shape == (1, 12, 27)
    assert obs[..., 6:8].abs().max() > .01


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


def test_opt_in_corridor_completion_keeps_driving_a_trailing_body():
    cfg = replace(_course_config(single_slit_env_config(), 12, DEFAULT_COURSE, "L"),
                  goal_orientation_required=False)
    state = torch.tensor([[-1.1690975, .1495785, .9574202, 0., 0., 0.]])
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    observation = _composition_observation(DEFAULT_COURSE, state, 0, cfg, context)
    learned = torch.zeros(1, 12, 8)
    kwargs = dict(position_gain=2., angle_gain=2., max_speed=.35, max_omega=.8,
                  feasible_corridor=True, route_free_geometry=True,
                  corridor_reserve_m=.05, corridor_max_speed=.5)
    old, active = local_se2_terminal_attractor_plan(observation, learned, cfg, **kwargs)
    new, new_active = local_se2_terminal_attractor_plan(
        observation, learned, replace(cfg, complete_local_corridor_exit=True), **kwargs,
    )
    torch.testing.assert_close(active, new_active, rtol=0, atol=0)
    rows = active[0, :, 0]
    assert rows.any()
    c, s = state[0, 2].cos(), state[0, 2].sin()
    old_x = .65 * (c * old[0, :, 3] - s * old[0, :, 4])
    new_x = .65 * (c * new[0, :, 3] - s * new[0, :, 4])
    assert old_x[rows].abs().max() < .01
    assert new_x[rows].min() > .02
    final_cfg = replace(cfg, goal_orientation_required=True)
    final_old, _ = local_se2_terminal_attractor_plan(observation, learned, final_cfg, **kwargs)
    final_new, _ = local_se2_terminal_attractor_plan(
        observation, learned, replace(final_cfg, complete_local_corridor_exit=True), **kwargs,
    )
    torch.testing.assert_close(final_old, final_new, rtol=0, atol=0)


def test_clear_reference_route_does_not_certify_actual_pose_rejoin():
    from piano_movers.local_skill_teacher import certify_pose_path
    from piano_movers.multi_waypoint_demo import _full_collision_scenario

    cfg = replace(_course_config(single_slit_env_config(), 10, DEFAULT_COURSE, "T"),
                  goal_orientation_required=False, complete_local_corridor_exit=True,
                  local_cspace_collision_substeps=32)
    # Actual v2 T/10 deadlock; endpoint-only clearance is positive while the
    # unchecked current-to-lookahead move penetrates during interpolation.
    state = torch.tensor([[-1.80108607, -.33845416, 1.19884074, 0., 0., 0.]])
    scene = _full_collision_scenario(DEFAULT_COURSE, cfg)
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = _composition_observation(DEFAULT_COURSE, state, 0, cfg, context)
    kwargs = dict(position_gain=2., angle_gain=2., max_speed=.35, max_omega=.8,
                  feasible_corridor=True, route_free_geometry=True,
                  corridor_reserve_m=.05, corridor_max_speed=.5)

    def next_pose(checked):
        plan, active = local_se2_terminal_attractor_plan(
            obs, torch.zeros(1, 10, 8),
            replace(cfg, local_cspace_validate_all_lookaheads=checked), **kwargs,
        )
        assert active.all()
        v = plan[0, :, 3:5] * .65
        theta = state[0, 2]
        predicted = state.repeat(10, 1)
        predicted[:, 0] += cfg.dt * (theta.cos() * v[:, 0] - theta.sin() * v[:, 1])
        predicted[:, 1] += cfg.dt * (theta.sin() * v[:, 0] + theta.cos() * v[:, 1])
        predicted[:, 2] += cfg.dt * plan[0, :, 5] * 1.4
        return predicted

    old, checked = next_pose(False), next_pose(True)
    assert not certify_pose_path(torch.cat((state, old[:1])), scene, cfg).passed
    assert all(certify_pose_path(torch.cat((state, q[None])), scene, cfg).passed for q in checked)
    assert torch.linalg.vector_norm(checked[0, :3] - state[0, :3]) > 1e-6
