from dataclasses import replace

import torch

from piano_movers.local_wrench_factors import local_wrench_information_factor
from piano_movers.single_slit_feedback import single_slit_env_config
from piano_movers.single_slit_local_planner import (
    LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS, LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS,
)


def data():
    cfg = replace(single_slit_env_config(), n_agents=8)
    torch.manual_seed(71)
    handles = torch.randn(1, 8, 2, dtype=torch.float64) * .3
    v = torch.tensor([.2, -.1, .4], dtype=torch.float64)
    u = torch.tensor([-.1, .3, -.2], dtype=torch.float64)
    obs = torch.zeros(1, 8, 27, dtype=torch.float64)
    obs[..., 1] = 1.
    obs[..., 2:4] = handles
    obs[..., 6] = (v[0] - handles[..., 1] * v[2]) / 2.
    obs[..., 7] = (v[1] + handles[..., 0] * v[2]) / 2.
    plan = torch.zeros(1, 8, 8, dtype=torch.float64)
    plan[..., 3:5] = u[:2] / LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
    plan[..., 5] = u[2] / LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS
    confidence = torch.tensor([[1., 4., 1., 4., 4., 1., 1., 1.]], dtype=torch.float64)
    return cfg, obs, plan, confidence, v, u


def test_weighted_mechanical_factors_recover_common_command_exactly():
    cfg, obs, plan, confidence, v, u = data()
    precision, natural = local_wrench_information_factor(obs, plan, cfg, confidence)
    z = torch.linalg.solve(precision.sum(1), natural.sum(1).unsqueeze(-1)).squeeze(-1)[0]
    physical = torch.tensor([cfg.mass, cfg.mass, cfg.inertia]) * (u / torch.tensor([
        1. - cfg.linear_damping * cfg.dt, 1. - cfg.linear_damping * cfg.dt,
        1. - cfg.angular_damping * cfg.dt,
    ], dtype=torch.float64) - v) / cfg.dt
    expected = physical / torch.tensor([cfg.n_agents * cfg.max_force] * 2 + [.75 * cfg.n_agents * cfg.max_force])
    torch.testing.assert_close(z, expected, atol=1e-8, rtol=1e-7)
    # Each node observes only two local motion coordinates.
    assert (torch.linalg.matrix_rank(precision) == 2).all()


def test_local_factors_do_not_read_peer_rows_and_are_body_frame_rotation_invariant():
    cfg, obs, plan, confidence, _, _ = data()
    reference = local_wrench_information_factor(obs, plan, cfg, confidence)
    changed = obs.clone()
    changed[:, 1:, 6:8] += 100.
    other = local_wrench_information_factor(changed, plan, cfg, confidence)
    for a, b in zip(reference, other):
        torch.testing.assert_close(a[:, 0], b[:, 0], atol=0., rtol=0.)
    angle = torch.tensor(.8, dtype=torch.float64)
    rotated = obs.clone()
    rotated[..., 0], rotated[..., 1] = angle.sin(), angle.cos()
    rotated[..., 6] = angle.cos() * obs[..., 6] - angle.sin() * obs[..., 7]
    rotated[..., 7] = angle.sin() * obs[..., 6] + angle.cos() * obs[..., 7]
    actual = local_wrench_information_factor(rotated, plan, cfg, confidence)
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
