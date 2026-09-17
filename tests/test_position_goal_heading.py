from dataclasses import replace

import torch

from piano_movers.config import EnvConfig
from piano_movers.single_slit_local_planner import _sinusoid_arc_extrema, _position_goal_heading_control


def test_arc_extrema_enclose_dense_samples_across_wrap_and_stationary_points():
    torch.manual_seed(281)
    a, b = torch.randn(100, 1, dtype=torch.float64), torch.randn(100, 1, dtype=torch.float64)
    start = 12. * torch.randn(100, 1, dtype=torch.float64)
    end = start + 12. * torch.randn_like(start)
    lower, upper = _sinusoid_arc_extrema(a, b, start, end)
    dense = start + (end - start) * torch.linspace(0., 1., 5001, dtype=torch.float64)
    samples = a * dense.cos() + b * dense.sin()
    assert bool((samples >= lower - 1e-12).all())
    assert bool((samples <= upper + 1e-12).all())
    torch.testing.assert_close(samples.amin(-1, keepdim=True), lower, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(samples.amax(-1, keepdim=True), upper, atol=1e-4, rtol=1e-5)


def test_position_only_rotation_preserves_goal_disk_and_entire_reserved_sweep():
    cfg = replace(EnvConfig(), goal_x=.63, goal_y=0., local_corridor_exit_goal_radius_m=.14)
    vertices = torch.tensor([[-.8, -.1], [-.8, .1], [.8, .1], [.8, -.1]])
    x, y, theta = torch.tensor([[.90]]), torch.tensor([[0.]]), torch.tensor([[0.]])
    chosen, active = _position_goal_heading_control(vertices, x, y, theta, cfg)
    assert bool(active.all())
    angles = torch.linspace(0., float(chosen.item()), 2001)
    rotated_x = vertices[:, 0] * angles[:, None].cos() - vertices[:, 1] * angles[:, None].sin()
    assert float((x + rotated_x).min()) > cfg.wall_slab + .030
    assert cfg.wall_slab + .030 - float(rotated_x[-1].min()) < cfg.goal_x + .14
    _, inactive = _position_goal_heading_control(vertices, x, y, theta,
                                                 replace(cfg, goal_x=1.2))
    assert not bool(inactive.any())
    # Too close to rotate without losing the safety reserve: no fabricated fix.
    _, blocked = _position_goal_heading_control(vertices, x - .0149, y, theta, cfg)
    assert not bool(blocked.any())
