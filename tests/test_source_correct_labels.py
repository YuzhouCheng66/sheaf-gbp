import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def load_trainer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("source_label_trainer", scripts / "train_mainline_action_dagger.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_source_correct_labels_preserve_executed_branch_and_teacher_witness():
    trainer = load_trainer()
    target, source, geometry = (torch.randn(3, 4, 8) for _ in range(3))
    original = target.clone()
    active = torch.tensor([True, False, True, False])[None, :, None].expand(3, -1, -1)
    preserve = torch.tensor([True, False, True])
    actual = trainer.source_correct_plan_target(target, source, geometry, active, preserve)
    assert torch.equal(target, original)
    assert torch.equal(actual[1], original[1])
    assert torch.equal(actual[preserve], torch.where(active, geometry, source)[preserve])
    assert torch.equal(trainer.source_correct_plan_target(target, source, geometry, active, ~preserve & preserve), original)


def test_source_correct_mask_is_a_whole_frame_decision():
    trainer = load_trainer()
    plan = torch.zeros(3, 4, 8)
    with pytest.raises(ValueError, match="one boolean"):
        trainer.source_correct_plan_target(plan, plan, plan, torch.ones(3, 4, 1).bool(), torch.ones(3, 4).bool())
