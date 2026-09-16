from dataclasses import replace

import numpy as np
import torch

from piano_movers.env import PianoBatch, step_dynamics, transition_collision_mask
from piano_movers.local_skill_teacher import (
    LocalTeacherConfig,
    certify_pose_path,
    executable_teacher_step,
    plan_local_target,
)
from piano_movers.single_slit_feedback import single_slit_env_config


def scene(state, goal, aperture):
    return PianoBatch(
        state0=state,
        source=state[:, :3].clone(),
        goal=goal.view(1, 3),
        wall_x=torch.tensor([[0.0]]),
        gap_y=torch.tensor([[0.0]]),
        gap_half=torch.tensor([[aperture / 2.0]]),
    )


def test_recursive_certificate_rejects_tunnelling_between_safe_endpoints():
    cfg = replace(single_slit_env_config(), collision_substeps=1)
    left = torch.tensor([[-1.0, 0.90, 0.0, 0.0, 0.0, 0.0]])
    right = left.clone()
    right[:, 0] = 1.0
    geometry = scene(left, right[0, :3], 0.72)
    assert not transition_collision_mask(left, right, geometry, cfg)[0]
    certificate = certify_pose_path(torch.cat((left, right)), geometry, cfg)
    assert not certificate.passed
    assert certificate.reason == "interior_clearance_violation"


def test_local_teacher_position_goal_does_not_depend_on_unused_heading():
    cfg = single_slit_env_config()
    state = torch.tensor([[-0.9, 0.1, 0.4, 0.0, 0.0, 0.0]])
    goal = torch.tensor([0.9, 0.1, -1.0])
    geometry = scene(state, goal, 3.0)
    options = LocalTeacherConfig(x_step=0.10, theta_bins=36, search_padding_m=0.1)
    a = plan_local_target(state, goal, geometry, cfg, orientation_required=False, options=options)
    goal[2] = 2.0
    b = plan_local_target(state, goal, geometry, cfg, orientation_required=False, options=options)
    assert a.status == b.status == "certified"
    np.testing.assert_array_equal(a.poses, b.poses)
    assert a.certificate.passed
    assert np.linalg.norm(a.poses[-1, :2] - goal[:2].numpy()) <= options.position_tolerance_m
    assert abs(a.poses[-1, 2] - float(state[0, 2])) < 1.0e-6


def test_teacher_force_step_replays_exactly_and_respects_force_reserve():
    cfg = single_slit_env_config()
    state = torch.tensor([[-0.9, 0.1, 0.4, 0.0, 0.0, 0.0]])
    goal = torch.tensor([0.9, 0.1, 0.4])
    geometry = scene(state, goal, 3.0)
    path = torch.tensor([[-0.9, 0.1, 0.4], [-0.7, 0.1, 0.4], [0.9, 0.1, 0.4]])
    options = LocalTeacherConfig()
    force, following, status = executable_teacher_step(state, path, geometry, cfg, options)
    assert status == "advance"
    assert (
        float(torch.linalg.vector_norm(force, dim=-1).max())
        <= cfg.max_force * options.force_fraction
    )
    torch.testing.assert_close(following, step_dynamics(state, force, cfg), rtol=0, atol=0)
    assert certify_pose_path(torch.cat((state, following)), geometry, cfg, options).passed
