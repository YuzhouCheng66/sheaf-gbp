import importlib.util
from pathlib import Path
import sys

import pytest
import numpy as np


def comparator():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("trace_comparator", scripts / "compare_mainline_trace_controls.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_case_index_rejects_duplicates_and_supports_counts_catalog():
    m = comparator()
    a = {"shape": "A", "agents": 8}
    assert m.case_key(a) == "A_n8"
    assert m.case_key(dict(a, case_id="layout_A")) == "layout_A"
    assert m.case_key({"case_id": "layout_only"}) == "layout_only"
    with pytest.raises(ValueError, match="duplicate"):
        m.full_case_index([a, a])


def test_only_declared_interface_change_is_allowed():
    m = comparator()
    original = {"options": {}}
    fixed = {"options": {"hold_feasible_crossing": True}}
    assert m.verify_options(original, fixed, "hold_feasible_crossing") == ["hold_feasible_crossing"]
    with pytest.raises(ValueError, match="expected only"):
        m.verify_options(original, {"options": {"hold_feasible_crossing": True, "preserve_private_memory": True}}, "hold_feasible_crossing")


def test_in_place_rotation_is_not_counted_as_stationary():
    m = comparator()
    states = np.zeros((3, 6))
    states[1, 5] = .5
    states[2, 3] = .1
    assert m.stationary_fraction(states) == 1. / 3.
