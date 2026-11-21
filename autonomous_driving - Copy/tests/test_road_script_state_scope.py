import symtable
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ROAD_SCRIPTS = (
    "unstructured_autonomous_road.py",
    "unstructured_autonomous_rough_road.py",
    "unstructured_autonomous_no_pathweaver_road.py",
)


@pytest.mark.parametrize("filename", ROAD_SCRIPTS)
def test_autonomous_loop_uses_module_risk_state(filename):
    path = ROOT / filename
    module_scope = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    loop_scope = next(
        scope
        for scope in module_scope.get_children()
        if scope.get_name() == "autonomous_loop"
    )

    for variable in ("current_risk_level", "current_target_speed"):
        assert loop_scope.lookup(variable).is_global()
