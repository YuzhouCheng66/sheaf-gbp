from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from piano_movers.gbp import coordination_tree_diameter
from piano_movers.models import SheafGBPPolicy
from piano_movers.multi_waypoint_demo import SkillCompositionOptions, apply_solver_controls


def test_coordination_tree_diameter_covers_chains_not_only_canonical_star():
    chain = torch.tensor([[i, i + 1] for i in range(6)])
    assert coordination_tree_diameter(chain, torch.ones(6, dtype=torch.bool), 7) == 6
    star = torch.tensor([[0, i] for i in range(1, 7)])
    assert coordination_tree_diameter(star, torch.ones(6, dtype=torch.bool), 7) == 2
    bad = torch.tensor([[0, 1], [1, 2], [2, 0]])
    with pytest.raises(ValueError, match="disconnected"):
        coordination_tree_diameter(bad, torch.ones(3, dtype=torch.bool), 4)


def test_opt_in_budget_and_precision_controls_are_separate():
    @dataclass(frozen=True)
    class Arm:
        gbp_steps: int = 4
        local_se2_geometry_evidence_precision_boost: float = 4.
        correction_edge_dim: int = 3

    policy = SheafGBPPolicy.__new__(SheafGBPPolicy)
    torch.nn.Module.__init__(policy)
    policy.arm_cfg = Arm()
    policy.env_cfg = SimpleNamespace(n_agents=7)
    policy.local_se2_analytic_reference_edges = torch.tensor([[i, i + 1] for i in range(6)])
    policy.local_se2_analytic_tree_edge_mask = torch.ones(6, dtype=torch.bool)
    policy.local_se2_analytic_gbp = SimpleNamespace(num_steps=4)
    policy.communication_floats_per_directed_edge_round = lambda: 6
    before = apply_solver_controls(policy, SkillCompositionOptions())
    assert before["rounds"] == 4 and before["geometry_precision_boost"] == 4.
    longer = apply_solver_controls(policy, SkillCompositionOptions(cover_coordination_tree_diameter=True))
    assert longer["rounds"] == 6 and longer["geometry_precision_boost"] == 4.
    equal = apply_solver_controls(policy, SkillCompositionOptions(geometry_precision_boost_override=1.))
    assert equal["rounds"] == 4 and equal["geometry_precision_boost"] == 1.
    assert equal["wire_floats_per_directed_message"] == 6
