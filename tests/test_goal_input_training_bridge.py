import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

@dataclass
class InputFlags:
    local_se2_markov_goal_context: bool = False
    local_se2_markov_goal_relative_y: bool = False


def load_bridge():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        return runpy.run_path(str(scripts / "train_mainline_action_dagger.py"))["configure_goal_input_control"]
    finally:
        sys.path.remove(str(scripts))


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("context", [False, True])
def test_training_input_control_preserves_original_tensors_and_rng(relative, context):
    bridge = load_bridge()
    model = nn.Module()
    model.arm_cfg = InputFlags()
    model.local_se2_markov_shape_encoder = nn.Sequential(nn.Linear(52, 16), nn.SiLU())
    old = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    count = bridge(model, relative_y=relative, goal_context=context)
    assert count == (64 if context else 0)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old.items())
    assert model.arm_cfg.local_se2_markov_goal_relative_y == relative
    assert model.arm_cfg.local_se2_markov_goal_context == context
    if context:
        assert not model.local_se2_markov_goal_projection.weight.count_nonzero()
    if relative or context:
        with pytest.raises(ValueError, match="original source"):
            bridge(model, goal_context=True)
