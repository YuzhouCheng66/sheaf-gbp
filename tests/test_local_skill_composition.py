from dataclasses import replace

import torch

from piano_movers.multi_waypoint_demo import (
    DEFAULT_COURSE,
    _body_cleared_wall,
    _course_config,
    _intermediate_reached,
)
from piano_movers.payload_shapes import payload_rectangles_for_config
from piano_movers.single_slit_feedback import single_slit_env_config
from piano_movers.single_slit_local_planner import local_se2_terminal_attractor_plan


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
