import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def trainer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("contact_sampler", scripts / "train_contact_residual_control.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_balancing_is_whole_frame_and_does_not_mutate_labels():
    module = trainer()
    mask = torch.tensor([False] * 100 + [True] * 2)
    old = mask.clone()
    for count in (2, 63, 64):
        torch.manual_seed(11)
        indices = module.balanced_contact_indices(mask, count)
        assert indices.shape == (count,)
        assert int(mask[indices].sum()) == count // 2
        torch.manual_seed(11)
        assert torch.equal(indices, module.balanced_contact_indices(mask, count))
        assert torch.equal(old, mask)


@pytest.mark.parametrize("value", [False, True])
def test_single_class_bucket_is_sampled_without_fabricating_labels(value):
    module = trainer()
    indices = module.balanced_contact_indices(torch.full((7,), value), 64)
    assert indices.min() >= 0 and indices.max() < 7


def test_only_executed_twist_difference_counts_as_recovery():
    module = trainer()
    source = torch.zeros(3, 4, 8)
    geometry = torch.ones_like(source)
    active = torch.tensor([True, False, True, False])[None, :, None]
    targets = torch.where(active, geometry, source).clone()
    targets[0, :, :3] += 1.
    targets[1, 0, 4] += .1
    original = targets.clone()
    result = module.contact_target_classes(dict(active=active, geometry=geometry, source_plan=source, target_plan=targets))
    assert result.tolist() == [False, True, False]
    assert torch.equal(original, targets)
