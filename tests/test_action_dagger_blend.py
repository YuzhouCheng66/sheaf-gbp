import importlib.util
from pathlib import Path
import sys

import torch


def test_action_training_preserves_null_branch_and_fractional_support():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("action_trainer", scripts / "train_mainline_action_dagger.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    source = torch.randn(1, 3, 8)
    old_morph = torch.randn_like(source)
    pred = old_morph.detach().clone().requires_grad_()
    support = torch.tensor([[[0.], [.5], [1.]]])
    source[:, 2, :6] = old_morph[:, 2, :6]
    actual = module.private_from_prediction(pred, source, old_morph, support)
    torch.testing.assert_close(actual, source, atol=0., rtol=0.)
    gradient = torch.autograd.grad(actual.sum(), pred)[0]
    torch.testing.assert_close(gradient[..., :6], support.expand(-1, -1, 6))
    assert not gradient[..., 6:].count_nonzero()
    changed = module.private_from_prediction(pred + .1, source, old_morph, support)
    torch.testing.assert_close(changed[:, 0], source[:, 0], atol=0., rtol=0.)
    torch.testing.assert_close(changed[..., 6:], source[..., 6:], atol=0., rtol=0.)
