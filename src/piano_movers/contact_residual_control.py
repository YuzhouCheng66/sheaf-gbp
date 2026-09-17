"""Independent frozen-trunk contact residual; never a silent mainline replacement.

Subtract the SAME encoder evaluated with zero recovery memory. Thus zero
memory implies identically zero correction, without a confidence gate, mode
selector or position threshold. The linear control has a bias-free 256-to-3
readout (768 weights); the independently tested nonlinear extension adds a
zero-initialized width-64 branch (17,344 weights total). Original message
budgets remain unchanged; local computation and post-geometry twist differ.
"""

import torch
from torch import nn


def contact_encoding_difference(policy, features):
    arm = policy.arm_cfg
    if (features.shape[-1] != 52 or arm.persistent_recovery_dim != 8
            or arm.local_se2_markov_goal_context or arm.local_se2_markov_goal_relative_y):
        raise ValueError("control requires the original 12+8+32 morphology inputs")
    zero_memory = features.clone()
    zero_memory[..., 12:20] = 0.
    return policy.local_se2_markov_shape_encode(features) - policy.local_se2_markov_shape_encode(zero_memory)


class ContactResidualControl(nn.Module):
    def __init__(self, hidden_width=0):
        super().__init__()
        if hidden_width not in (0, 64):
            raise ValueError("declared readout controls are linear or width-64 nonlinear")
        self.hidden_width = hidden_width
        self.readout = nn.Linear(256, 3, bias=False)
        nn.init.zeros_(self.readout.weight)
        self.refinement = (None if hidden_width == 0 else nn.Sequential(
            nn.Linear(256, hidden_width, bias=False), nn.SiLU(), nn.Linear(hidden_width, 3, bias=False)))
        if self.refinement is not None:
            nn.init.zeros_(self.refinement[-1].weight)
        self.register_buffer("feature_scale", torch.ones(256))
        self.register_buffer("twist_scale", torch.tensor([1.2, 1.2, 3.]))

    def forward(self, difference):
        if difference.shape[-1] != 256:
            raise ValueError("contact encoder difference must have width 256")
        value = difference / self.feature_scale
        raw = self.readout(value)
        if self.refinement is not None:
            raw = raw + self.refinement(value)
        return self.twist_scale * torch.tanh(raw)


def load_contact_residual(path, actor_sha256):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != "piano-movers.contact-residual-control.v1" or payload["actor_sha256"] != actor_sha256:
        raise ValueError("contact-control base checkpoint mismatch")
    control = ContactResidualControl(payload.get("hidden_width", 0))
    control.load_state_dict(payload["state_dict"], strict=True)
    return control.eval()
