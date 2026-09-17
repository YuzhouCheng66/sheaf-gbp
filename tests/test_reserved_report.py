import importlib.util
from pathlib import Path
import sys

import pytest


def load_report():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("reserved_report", scripts / "report_mainline_reserved_panel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_cluster_uncertainty_does_not_treat_replicated_cases_as_new_layouts():
    module = load_report()
    rows = [{"seed": seed, "result": {"success": True, "steps": 2},
             "continuous_certificate": {"passed": True}} for seed in range(4)]
    one = module.summarize(rows)
    many = module.summarize(rows * 100)
    assert one["layout_cluster_hoeffding_95"] == many["layout_cluster_hoeffding_95"]
    assert many["independent_layout_draws"] == 4
    assert many["layout_cluster_hoeffding_95"][0] < .33
    rows[0]["continuous_certificate"]["passed"] = False
    assert module.summarize(rows)["strict_successes"] == 3
    rows[0]["continuous_certificate"]["passed"] = True
    rows[0]["independent_resolved_pose_collisions"] = 1
    assert not module.strict_success(rows[0])


def test_empty_or_unequal_clusters_are_rejected():
    module = load_report()
    with pytest.raises(ValueError):
        module.summarize([])
    rows = [{"seed": seed, "result": {"success": True, "steps": 2},
             "continuous_certificate": {"passed": True}} for seed in (1, 1, 2)]
    with pytest.raises(ValueError):
        module.summarize(rows)
