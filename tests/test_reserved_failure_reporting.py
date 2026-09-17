import importlib.util
from pathlib import Path
import sys


def reporter():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("reserved_report", scripts / "report_mainline_reserved_panel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_invalid_pose_without_trace_remains_reportable_failure():
    m = reporter()
    row = {"case_id": "invalid", "seed": 83201,
           "result": {"success": False, "steps": 0, "reason": "invalid_initial_or_final_pose"}}
    assert not m.strict_success(row)
    record = m.failure_record(row)
    assert record["trace_sha256"] is None
    assert record["result"]["reason"] == "invalid_initial_or_final_pose"


def test_uncertified_cases_remain_in_equal_cluster_denominator():
    m = reporter()
    rows = []
    for seed in range(83201, 83205):
        rows.extend([{"seed": seed, "result": {"success": False, "steps": 0}},
                     {"seed": seed, "result": {"success": True, "steps": 10},
                      "continuous_certificate": {"passed": True}}])
    summary = m.summarize(rows)
    assert summary["episodes"] == 8
    assert summary["strict_successes"] == 4
    assert summary["independent_layout_draws"] == 4
    assert summary["success_fraction"] == .5
