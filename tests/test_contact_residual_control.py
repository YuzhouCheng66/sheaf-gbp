from types import SimpleNamespace

import torch

from piano_movers.contact_residual_control import ContactResidualControl, contact_encoding_difference


def test_zero_memory_means_exactly_zero_residual_even_after_learning():
    torch.manual_seed(728)
    encoder = torch.nn.Sequential(torch.nn.Linear(52, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256))
    policy = SimpleNamespace(arm_cfg=SimpleNamespace(persistent_recovery_dim=8,
        local_se2_markov_goal_context=False, local_se2_markov_goal_relative_y=False),
        local_se2_markov_shape_encode=encoder)
    features = torch.randn(2, 12, 52)
    features[..., 12:20] = 0.
    difference = contact_encoding_difference(policy, features)
    assert difference.count_nonzero() == 0
    control = ContactResidualControl()
    with torch.no_grad():
        control.readout.weight.normal_()
    assert control(difference).count_nonzero() == 0
    assert sum(p.numel() for p in control.parameters()) == 768


def test_zero_initialized_readout_has_live_contact_gradient_and_bounded_output():
    torch.manual_seed(718)
    difference = torch.randn(2, 12, 256)
    control = ContactResidualControl()
    assert control(difference).count_nonzero() == 0
    control(difference).sum().backward()
    assert control.readout.weight.grad.abs().max() > 0
    with torch.no_grad():
        control.readout.weight.normal_(std=3.)
    assert (control(difference).abs() <= control.twist_scale).all()


def test_nonlinear_readout_bridge_preserves_linear_function_and_zero_contact():
    torch.manual_seed(719)
    source = ContactResidualControl()
    with torch.no_grad():
        source.readout.weight.normal_(std=.01)
    expanded = ContactResidualControl(64)
    expanded.readout.load_state_dict(source.readout.state_dict())
    value = torch.randn(2, 12, 256)
    assert torch.equal(source(value), expanded(value))
    with torch.no_grad():
        for p in expanded.parameters():
            p.normal_()
    assert expanded(torch.zeros_like(value)).count_nonzero() == 0
    assert sum(p.numel() for p in expanded.parameters()) == 17344
