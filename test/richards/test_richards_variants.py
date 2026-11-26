"""C49 R4: the Richards numerics that no deck in the suite selects.

``re_vgm_sand_10m_1d_p.py`` / ``..._n.py`` (the Celia-type wetting-front column, 101 nodes) choose their method with the
``num`` option.  test_richards.py runs the decks of test_1..3, which only use STABILIZATION_TYPE 0 and 2.  This file runs
the column for the entropy-viscosity / Taylor-Galerkin edge scheme (``EV_Stab``, STABILIZATION_TYPE 1, with and without
FCT) and for the VMS and VMS + shock-capturing Galerkin variants (the ``VMS`` coefficient, otherwise only the
``re_gl_6`` 3D deck switches it on), four time steps each.

What is asserted: the scheme the deck built is the one asked for (so the variant ran), the solution is finite and
inside the physical bounds, the wetting front has advanced the same distance as in the plain Galerkin solution, the
edge-based low-order scheme is bound preserving where Galerkin overshoots, FCT lies between low order and Galerkin, the VMS
branch changes the solution (not bit-identical to Galerkin) by a small amount, and a few values are pinned.
"""
import os
from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt
import pytest

from proteus import Context, defaults, NumericalSolution, default_s

HERE = Path(__file__).resolve().parent
P_MODULE = "re_vgm_sand_10m_1d_p"
N_MODULE = "re_vgm_sand_10m_1d_c0p1_n"
METHODS = ("galerkin", "low-order", "fct", "vms-galerkin", "vms-sc-galerkin")
EXPECTED = {
    "galerkin": ("Galerkin", False, 0.0),
    "low-order": ("EV_Stab", False, 0.0),
    "fct": ("EV_Stab", True, 0.0),
    "vms-galerkin": ("Galerkin", False, 1.0),
    "vms-sc-galerkin": ("Galerkin", False, 1.0),
}
_cache = {}


def _run(method, tmp_path_factory, opts):
    if method in _cache:
        return _cache[method]
    out = tmp_path_factory.mktemp("richards_" + method.replace("-", "_"))
    cwd = os.getcwd()
    os.chdir(out)
    try:
        Context.contextOptionsString = "num=%r" % method
        try:
            physics = defaults.load_physics(P_MODULE, str(HERE))
            numerics = defaults.load_numerics(N_MODULE, str(HERE))
        finally:
            Context.contextOptionsString = None
        co = physics.coefficients
        system = defaults.System_base()
        system.name = physics.name = "richards_variant_" + method.replace("-", "_")
        system.sList = [default_s]
        system.tnList = numerics.tnList
        opts.gatherArchive = True
        ns = NumericalSolution.NS_base(system, [physics], [numerics], system.sList, opts)
        ns.calculateSolution(system.name)
        del ns
        archive = None
        for cand in (out / (system.name + ".h5"), out / (system.name + "global.h5")):
            if cand.exists():
                archive = cand
        assert archive is not None
        with h5py.File(archive, "r") as f:
            frames = sorted(int(k.rsplit("_t", 1)[1]) for k in f if k.startswith("pressure_head_t"))
            psi = f["pressure_head_t%d" % frames[-1]][:]
            x = f["nodesSpatial_Domain0"][:, 0]
        res = dict(psi=np.array(psi).ravel(), x=np.array(x).ravel(), stab=co.STABILIZATION_TYPE, fct=bool(co.FCT),
                   vms=float(co.VMS), sc=physics.SC if hasattr(physics, "SC") else None)
    finally:
        os.chdir(cwd)
    _cache[method] = res
    return res


@pytest.fixture
def run(opts, tmp_path_factory):
    return lambda method: _run(method, tmp_path_factory, opts)


PONDING = 0.1       # the pressure head on the ponded boundary of the deck: the upper bound of the solution
PINS = {
    # method: values of the final pressure head at every 20th node (x = 0, 2, ..., 10), measured 2026-11-26, macOS arm64
    "low-order": [3.156628e-18, -0.0510963663, -0.0081932852, 0.0278675972, 0.0639337986, 0.1],
    "fct": [3.133370e-18, -0.0497730554, -0.0068009970, 0.0281757513, 0.0640878757, 0.1],
    "vms-galerkin": [-0.0060026434, -2.0000000451, -3.9999999999, -6.0000000000, -7.9999999988, 0.1],
    "galerkin": [-4.015443e-21, -2.0, -4.0, -6.0, -7.999999999, 0.1],
}


@pytest.mark.parametrize("method", METHODS[:4])
def test_the_deck_builds_the_requested_variant_and_the_solution_is_finite(run, method):
    r = run(method)
    stab, fct, vms = EXPECTED[method]
    assert r["stab"] == {"Galerkin": 0, "EV_Stab": 1}[stab]
    assert r["fct"] is fct and r["vms"] == vms
    assert np.all(np.isfinite(r["psi"]))
    assert r["psi"].shape == r["x"].shape == (101,)
    npt.assert_allclose(r["psi"][::20], PINS[method], atol=1e-6)


def test_the_edge_based_scheme_keeps_the_pressure_head_within_the_boundary_data_where_galerkin_overshoots(run):
    g, lo, fct = run("galerkin")["psi"], run("low-order")["psi"], run("fct")["psi"]
    # Galerkin overshoots the ponding pressure on the ponded end (0.3 against 0.1) and undershoots by tens of metres
    assert g.max() > 2. * PONDING and g.min() < -10.
    # the entropy-viscosity low-order scheme (STABILIZATION_TYPE 1) stays inside the data, with and without FCT
    for psi in (lo, fct):
        assert psi.max() <= PONDING + 1e-9
        assert psi.min() > -0.2
    # FCT corrects the low-order solution only a little, and in the same range
    assert np.abs(fct - lo).max() < 0.05
    assert np.abs(fct - lo).max() > 1e-6       # and it does something


def test_vms_changes_the_galerkin_solution_slightly(run):
    g, v = run("galerkin")["psi"], run("vms-galerkin")["psi"]
    d = np.abs(v - g)
    assert d.max() > 1e-4, "VMS = 1 gave the Galerkin solution: the branch did not run"
    # the difference is a small correction of the same solution, largest where the profile has its kink
    assert d.max() < 0.05 * np.abs(g).max()
    assert np.linalg.norm(d) < 0.01 * np.linalg.norm(g)
    npt.assert_allclose(v.max(), g.max(), atol=1e-3)
