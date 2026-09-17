from dataclasses import replace

import torch

from piano_movers.config import EnvConfig
from piano_movers.single_slit_local_planner import _crossing_hold_mask


def test_crossing_hold_only_when_current_fit_would_be_lost():
    cfg = replace(EnvConfig(), dt=.1, world_y=2.)
    vertices = torch.tensor([[-.4, -.2], [.4, -.2], [.4, .2], [-.4, .2]], dtype=torch.float64)
    y = torch.tensor([0., 0., 0., .4], dtype=torch.float64)
    zeros = torch.zeros_like(y)
    # Safe straight crossing; unsafe rotation; unsafe drift; already outside.
    result = _crossing_hold_mask(vertices, y, zeros, torch.tensor([0., 0., .1, 0.]),
                                torch.tensor([0., .1, 0., .1]), zeros, zeros + .201, cfg)
    assert result.tolist() == [False, True, True, False]


def test_crossing_bound_contains_dense_exact_vertex_motion():
    cfg = replace(EnvConfig(), dt=.08, world_y=2.)
    generator = torch.Generator().manual_seed(17)
    vertices = torch.tensor([[-.4, -.1], [.4, -.1], [.4, .1], [-.4, .1]], dtype=torch.float64)
    theta = torch.rand(100, generator=generator, dtype=torch.float64) * 6. - 3.
    omega = torch.rand(100, generator=generator, dtype=torch.float64) * 2.8 - 1.4
    vy = torch.rand(100, generator=generator, dtype=torch.float64) - .5
    y = torch.zeros_like(theta)
    half = torch.full_like(theta, .45)
    hold = _crossing_hold_mask(vertices, y, theta, vy, omega, y, half, cfg)
    fraction = torch.linspace(0., 1., 1001, dtype=torch.float64)
    angle = theta[:, None] + cfg.dt * omega[:, None] * fraction
    vertex_y = (angle.sin()[..., None] * vertices[:, 0] + angle.cos()[..., None] * vertices[:, 1]
                + cfg.dt * vy[:, None, None] * fraction[None, :, None])
    sampled_safe = (vertex_y.abs() <= half[:, None, None] - 2e-6).all(-1).all(-1)
    assert bool(sampled_safe[~hold].all())
