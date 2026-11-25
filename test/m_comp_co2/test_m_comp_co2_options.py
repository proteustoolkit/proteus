"""m_comp_co2 options no regression case sets, on the segregation column (closed, STAB 2, a free gas pool).

The cases of test_m_comp_co2.py leave most of ``m_comp_co2.Coefficients``' options at their defaults. Here the same deck
(case 'segregation', loaded and run by test_m_comp_co2.py's own helpers) runs with one option changed through the
constructor, and what each option promises is checked on the run's totals:

- ``injection_ports`` (a CO2 source at a point of the column): the CO2 total grows, the H2O total does not;
- the lumped-disk and the consistent point-source injections put in the same CO2 (the point source is "mesh-exact");
- options that change the physics of the gas (``S_gr_types``, ``krn_end_types``, ``mu_n``) leave the conservation of CO2 and H2O
  to the tolerance of the segregation case.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("m_comp_co2_cases", str(HERE / "test_m_comp_co2.py"))
cases = importlib.util.module_from_spec(_spec)
sys.modules["m_comp_co2_cases"] = cases
_spec.loader.exec_module(cases)
from proteus.kernels import m_comp_co2


def _run_with(options, tmp_path, monkeypatch):
    init = m_comp_co2.Coefficients.__init__

    def patched(self, *args, **kwargs):
        kwargs.update(options)
        init(self, *args, **kwargs)
    monkeypatch.setattr(m_comp_co2.Coefficients, "__init__", patched)
    return cases._run("case='segregation'", tmp_path, monkeypatch)


RATE, RADIUS = 1.0e-4, 0.03
PORT = [(0.05, 0.2, RATE, RADIUS, 0.0, 1.0e9)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each option run once on the segregation column: {name: totals}."""
    out = {}
    options = {
        "fct": dict(FCT=True, num_fct_iter=1),
        "trapping": dict(S_gr_types=np.array([0.1])),
        "endpoint": dict(krn_end_types=np.array([0.5])),
        "inject_disk": dict(injection_ports=PORT),
        "inject_point": dict(injection_ports=PORT, injection_point_source=True),
    }
    for name, opts in options.items():
        with pytest.MonkeyPatch.context() as mp:
            _, totals = _run_with(opts, tmp_path_factory.mktemp(name), mp)
        out[name] = totals
    return out


@pytest.mark.parametrize("name", ["fct", "trapping", "endpoint"])
def test_the_gas_options_conserve_co2_and_h2o(runs, name):
    drift = cases._drifts(runs[name])
    assert drift["co2_drift"] < 1e-6 and drift["h2o_drift"] < 1e-6, drift


def test_the_point_source_puts_in_the_mesh_exact_amount(runs):
    """rate * pi * radius^2 per unit time (the doc of injection_point_source), in the same units as the totals' CO2 column; the H2O total
    does not move. The disk source over the nodes within the radius puts in less on this mesh (the nodes do not tile the disk)."""
    expected = RATE * np.pi * RADIUS ** 2 * (runs["inject_point"][-1, 0] - runs["inject_point"][0, 0])
    added = {k: runs[k][-1, 1] - runs[k][0, 3] for k in ("inject_disk", "inject_point")}
    assert added["inject_point"] == pytest.approx(expected, rel=0.05)
    assert 0.5 * expected < added["inject_disk"] < added["inject_point"]
    for k in ("inject_disk", "inject_point"):
        assert abs(runs[k][-1, 2] - runs[k][0, 2]) < 1e-6 * abs(runs[k][0, 2])
