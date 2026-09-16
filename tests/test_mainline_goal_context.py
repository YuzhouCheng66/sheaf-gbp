from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json
import runpy

import pytest
import torch
from torch import nn

from piano_movers.models import SheafGBPPolicy, mlp
from piano_movers.single_slit_feedback import single_slit_env_config
from piano_movers.multi_waypoint_demo import Slit, WaypointCourse


def small_goal_interface():
    model = SheafGBPPolicy.__new__(SheafGBPPolicy)
    nn.Module.__init__(model)
    model.arm_cfg = SimpleNamespace(local_se2_markov_goal_context=True)
    model.env_cfg = single_slit_env_config()
    model.local_se2_markov_shape_encoder = mlp(52, 16, 16, layers=2)
    model.local_se2_markov_goal_projection = nn.Linear(4, 16, bias=False)
    nn.init.zeros_(model.local_se2_markov_goal_projection.weight)
    return model


def test_zero_goal_projection_has_exact_forward_and_live_gradient():
    torch.manual_seed(72)
    model = small_goal_interface()
    features = torch.randn(4, 12, 52)
    before = model.local_se2_markov_shape_encoder(features)
    after = model.local_se2_markov_shape_encode(features)
    torch.testing.assert_close(before, after, atol=0., rtol=0.)
    after.square().mean().backward()
    assert model.local_se2_markov_goal_projection.weight.grad.abs().max() > 1e-7


def test_goal_context_sees_actual_y_and_masks_unrequested_orientation():
    model = small_goal_interface()
    features = torch.zeros(1, 12, 52)
    features[..., 0], features[..., 1], features[..., 3] = 1., 0., .2
    model.env_cfg = replace(model.env_cfg, goal_y=.4, goal_theta=.7, goal_orientation_required=False)
    free = model.local_se2_markov_goal_features(features)
    assert (free[..., 3] == -1.).all()
    assert (free[..., 1:3] == 0.).all()
    torch.testing.assert_close(free[..., 0], torch.full((1, 12), .4 / model.env_cfg.world_y - .2))
    model.env_cfg = replace(model.env_cfg, goal_theta=-2.)
    torch.testing.assert_close(free, model.local_se2_markov_goal_features(features), atol=0., rtol=0.)
    model.env_cfg = replace(model.env_cfg, goal_theta=0., goal_orientation_required=True)
    fixed = model.local_se2_markov_goal_features(features)
    torch.testing.assert_close(fixed[..., 1:], torch.tensor([-1., 0., 1.]).expand(1, 12, 3))


def test_replay_command_tracks_actual_stage_and_final_orientation(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    active = runpy.run_path(str(scripts / "train_mainline_skill_dagger.py"))["active_anchor_config"]
    course = WaypointCourse(source=(-3., 0., 0.),
                            walls=(Slit(-1., 0., .72), Slit(1., .2, .72)),
                            waypoints=((-0.4, .1, .7), (2., -.2, -.5)))
    first = active(single_slit_env_config(), course, 0, True)
    last = active(single_slit_env_config(), course, 1, True)
    assert first.goal_x == .6 and first.goal_y == .1
    assert not first.goal_orientation_required
    assert last.goal_x == 1. and last.goal_y == -.2
    assert last.goal_theta == -.5 and last.goal_orientation_required


def test_replay_panels_require_correct_actor_development_and_safety(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    admit = runpy.run_path(str(scripts / "train_mainline_skill_dagger.py"))["admitted_anchor_rows"]
    row = {"shape": "A", "agents": 12, "result": {"success": True},
           "continuous_certificate": {"passed": True}, "trace_sha256": "trace"}
    base = {"status": "complete", "split": "development_diagnostics",
            "checkpoint_sha256": "source", "results": [row, dict(row, shape="Z"), dict(row, agents=9)]}
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps(base))
    provenance, rows = admit([path, path], "source")
    assert len(rows) == 1
    assert provenance[0]["excluded_heldout_or_unsuccessful"] == 2
    assert provenance[1]["duplicate_traces_skipped"] == 1
    with pytest.raises(ValueError, match="actor"):
        admit([path], "different")
    path.write_text(json.dumps(dict(base, split="locked_validation")))
    with pytest.raises(ValueError, match="development"):
        admit([path], "source")
    path.write_text(json.dumps(dict(base, results=[dict(row, continuous_certificate={"passed": False})])))
    with pytest.raises(ValueError, match="certification"):
        admit([path], "source")
