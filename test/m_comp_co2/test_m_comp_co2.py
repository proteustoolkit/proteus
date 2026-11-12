#!/usr/bin/env python
"""Regression tests for the compositional CO2-brine model (m_comp_co2).

All cases come from one deck pair, m_comp_co2_p.py / m_comp_co2_n.py, with the
case chosen through proteus's Context; see m_comp_co2_p.py for the physics.

================  ======  =======  ==========================================
test              STAB    top      what is checked
================  ======  =======  ==========================================
static            0       closed   exact steady state: p and z never move
hydrostatic       2       closed   p keeps the hydrostatic gradient, z stays
                                   put, CO2 and H2O conserved
segregation       2       closed   CO2 and H2O conserved, the free gas rises
open_top          0       open     p matches the hydrostatic profile
open_top_stab2    2       open     the same through the EV path, CO2
                                   conserved
seal_holds        2       closed   per-rock closure on: the ESF holds the gas
                                   pool; interface S_g within the per-rock S_r
seal_leaks_...    2       closed   closure off: the same gas crosses the ESF
one_rock_...      2       closed   no rock interfaces: closure on == off
================  ======  =======  ==========================================

Every checked number, its tolerance and the reason for the tolerance are in
ONE file, m_comp_co2_expected.txt.  To regenerate its expected column::

    PROTEUS_SAVE_COMPARISON=1 python -m pytest test_m_comp_co2.py

Saving rewrites only the rows of the tests that ran, so ``-k segregation``
regenerates just those.  Inspect the diff before committing -- a regenerated
file silently blesses whatever the code does today.

The same cases run standalone under parun, e.g.

    parun m_comp_co2_p.py m_comp_co2_n.py -C "case='rock_column' per_rock_closure=False" \\
          -P '-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist'
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from petsc4py import PETSc

from proteus import Context, defaults
from proteus.iproteus import NumericalSolution, default_s, opts
from proteus.m_comp_co2 import m_comp_co2


HERE = Path(__file__).resolve().parent
EXPECTED_FILE = HERE / "m_comp_co2_expected.txt"

P_MODULE = "m_comp_co2_p"
N_MODULE = "m_comp_co2_n"

SAVE_COMPARISON = os.environ.get("PROTEUS_SAVE_COMPARISON", "") not in (
    "", "0", "no", "false", "False",
)

# Deck constants (m_comp_co2_p.py).
P_ATM = 1.013e5      # pressure at the top of the column [Pa]
RHO_W_MASS = 998.0   # brine density of the rest cases' hydrostatic IC [kg/m^3]
G_MAG = 9.81         # gravity magnitude [m/s^2]
LY = 1.0             # rest-case column height [m]
Z0 = 1.0e-5          # initial z of the single-phase rest cases
Y_SEAL0, Y_SEAL1 = 0.10, 0.14   # rock_column ESF bottom and top [m]

# Direct LU, as in parun -P.  The decks' KSP default with linTolFac = 1e-3
# leaves an inexact Newton step: on open_top Newton stalls at |r| = 7.2e-8
# against nl_atol_res = 1e-8 and the run gives up at t = 2.9 s.
PETSC_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "superlu_dist",
}

# Unwrapped, so a test that runs two cases does not wrap the wrapper.
POST_STEP = m_comp_co2.Coefficients.postStep


def _load_case(context):
    """Load the physics/numerics decks under a Context string."""
    for module_name in (P_MODULE, N_MODULE):
        sys.modules.pop(module_name, None)

    saved_path = sys.path[:]
    # The decks call Context.Options at import, which reads this module global --
    # the same channel parun's -C writes (as in test/richards/test_richards.py).
    Context.contextOptionsString = context
    try:
        # load_physics appends the deck directory to sys.path and removes the
        # first occurrence afterwards, so the insert is repeated per loader call.
        sys.path.insert(0, str(HERE))
        physics = defaults.load_physics(P_MODULE, str(HERE))
        # The numerics deck does "from m_comp_co2_p import *"; publish the
        # physics just loaded so it reads back this case's objects.
        sys.modules[P_MODULE] = physics
        sys.path.insert(0, str(HERE))
        numerics = defaults.load_numerics(N_MODULE, str(HERE))
    finally:
        Context.contextOptionsString = None
        sys.path[:] = saved_path
        sys.modules.pop(P_MODULE, None)
    return physics, numerics


def _fields(archive, names=("p", "z")):
    """Node y, the archived frame indices, and {name: {frame: array}}."""
    with h5py.File(archive, "r") as f:
        y = f["nodesSpatial_Domain0"][:, 1]
        frames = sorted(int(k[len("p_t"):]) for k in f.keys()
                        if k.startswith("p_t") and k[len("p_t"):].isdigit())
        # main's archiver names a derived scalar "Sg_t3", 1.9.x "Sg0_t3".
        def key(name, i):
            k = "%s_t%d" % (name, i)
            return k if k in f else "%s_t%d" % (name.rstrip("0") or name, i)
        fields = {name: {i: f[key(name, i)][:] for i in frames}
                  for name in names}
    return y, frames, fields


def _run(context, output_dir, monkeypatch):
    """Run one case to its end time; return its archive and per-step totals.

    totals has one row per step on the STAB=2 path (empty on STAB=0):
    (t, CO2, H2O, CO2 at the step's old time level), from the kernel's conserved
    nodal masses: CO2 = sum_i ML_i mLow_n_i, mLow_n_i = (lumped phi N)_i z_i;
    H2O = sum_i ML_i mLow_i.
    """
    totals = []

    def recording_post_step(self, t, firstStep=False):
        result = POST_STEP(self, t, firstStep)
        model = self.model
        # mLow_n is allocated by the first residual evaluation, so the spin-up
        # call (before any solve) is skipped and step 1 is kept.
        if self.STABILIZATION_TYPE == 2 and getattr(model, "mLow_n", None) is not None:
            totals.append((t,
                           float(np.dot(model.ML, model.mLow_n)),
                           float(np.dot(model.ML, model.mLow)),
                           float(np.dot(model.ML, model.mn_n))))
        return result

    monkeypatch.setattr(m_comp_co2.Coefficients, "postStep", recording_post_step)
    monkeypatch.chdir(output_dir)
    physics, numerics = _load_case(context)

    system = defaults.System_base()
    system.name = physics.name
    system.sList = [default_s]
    # None: NumericalSolution builds the output times from the deck's T, nDTout.
    system.tnList = None

    opts.logLevel = 1
    opts.verbose = False
    opts.profile = False
    opts.gatherArchive = True

    petsc_options = PETSc.Options()
    for key, value in PETSC_OPTIONS.items():
        petsc_options.setValue(key, value)
    try:
        numerical_solution = NumericalSolution.NS_base(
            system, [physics], [numerics], system.sList, opts
        )
        numerical_solution.calculateSolution(system.name)
        del numerical_solution
    finally:
        for key in PETSC_OPTIONS:
            petsc_options.delValue(key)

    archive = None
    for candidate in (output_dir / (system.name + ".h5"),
                      output_dir / (system.name + "global.h5")):
        if candidate.exists():
            archive = candidate
    assert archive is not None, "{0}: no archive in {1}".format(context, output_dir)
    # calculateSolution returns normally when a step fails for good; the archive
    # then stops at the failure, so a missing frame means the run did not reach T.
    _, frames, _ = _fields(archive)
    assert len(frames) == numerics.nDTout + 1, (
        "{0}: archived {1} of {2} frames; the run stopped before T = {3:g} s"
        .format(context, len(frames), numerics.nDTout + 1, numerics.T))
    return archive, np.array(totals)


def _drifts(totals):
    """co2_drift and h2o_drift: max relative change of each total.

    CO2 starts from step 1's old level (the initial state); H2O has no old-level
    total, so it starts from the end of step 1.
    """
    assert len(totals) > 0, "no STAB=2 totals were recorded"
    co2_0, h2o_0 = totals[0, 3], totals[0, 2]
    return {
        "co2_drift": np.abs(totals[:, 1] - co2_0).max() / abs(co2_0),
        "h2o_drift": np.abs(totals[:, 2] - h2o_0).max() / abs(h2o_0),
    }


def _hydrostatic(y):
    return P_ATM + RHO_W_MASS * G_MAG * (LY - y)


def _read_expected():
    """{(case, metric): (expected, tolerance)} from m_comp_co2_expected.txt."""
    rows = {}
    for line in EXPECTED_FILE.read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if fields:
            case, metric, expected, tolerance = fields
            rows[(case, metric)] = (float(expected), float(tolerance))
    return rows


def _save_expected(case, metrics):
    """Rewrite the expected column of this case's rows; keep tolerances, notes."""
    lines = EXPECTED_FILE.read_text().splitlines(keepends=True)
    saved = set()
    for k, line in enumerate(lines):
        data, hash_, note = line.partition("#")
        fields = data.split()
        if fields and fields[0] == case and fields[1] in metrics:
            lines[k] = "{0:<24}{1:<19}{2:<14.6e}{3:<10}{4}{5}".format(
                fields[0], fields[1], metrics[fields[1]], fields[3], hash_,
                note if hash_ else "\n")
            saved.add(fields[1])
    missing = sorted(set(metrics) - saved)
    assert not missing, (
        "{0}: add rows (with a tolerance) to {1} for {2}".format(
            case, EXPECTED_FILE.name, missing))
    EXPECTED_FILE.write_text("".join(lines))


def _check(case, metrics):
    """Compare (or, with PROTEUS_SAVE_COMPARISON, save) this case's metrics."""
    if SAVE_COMPARISON:
        _save_expected(case, metrics)
        pytest.skip("wrote {0} values of {1}".format(len(metrics), EXPECTED_FILE.name))
    rows = _read_expected()
    for metric, value in metrics.items():
        assert (case, metric) in rows, (
            "{0} has no row {1} {2}".format(EXPECTED_FILE.name, case, metric))
        expected, tolerance = rows[(case, metric)]
        assert abs(value - expected) <= tolerance, (
            "{0} {1} = {2:.6e}, expected {3:.6e} +/- {4:.1e}".format(
                case, metric, value, expected, tolerance))


class TestRestAndConservation:

    def test_static(self, tmp_path, monkeypatch):
        archive, _ = _run("case='static'", tmp_path, monkeypatch)
        _, frames, f = _fields(archive)
        _check("static", {
            "p_dev": max(np.abs(f["p"][i] - P_ATM).max() for i in frames),
            "z_dev": max(np.abs(f["z"][i] - Z0).max() for i in frames),
        })

    def test_hydrostatic(self, tmp_path, monkeypatch):
        archive, totals = _run("case='hydrostatic'", tmp_path, monkeypatch)
        y, frames, f = _fields(archive)
        p_h = _hydrostatic(y)
        profile = [f["p"][i] - p_h for i in frames]
        metrics = {
            "p_profile_dev": max(np.abs(d - d.mean()).max() for d in profile),
            "z_dev": max(np.abs(f["z"][i] - Z0).max() for i in frames),
        }
        metrics.update(_drifts(totals))
        _check("hydrostatic", metrics)

    def test_segregation(self, tmp_path, monkeypatch):
        archive, totals = _run("case='segregation'", tmp_path, monkeypatch)
        y, frames, f = _fields(archive, ("p", "Sg0"))
        # Sg is archived from the first step on, so only the end is used.
        Sg = f["Sg0"][frames[-1]]
        metrics = {"gas_centroid": np.dot(Sg, y) / Sg.sum()}
        metrics.update(_drifts(totals))
        _check("segregation", metrics)

    def test_open_top(self, tmp_path, monkeypatch):
        archive, _ = _run("case='open_top'", tmp_path, monkeypatch)
        _check("open_top", self._open_top_metrics(archive))

    def test_open_top_stab2(self, tmp_path, monkeypatch):
        archive, totals = _run("case='open_top_stab2'", tmp_path, monkeypatch)
        metrics = self._open_top_metrics(archive)
        # Brine crosses the open top, so only CO2 is conserved.
        metrics["co2_drift"] = _drifts(totals)["co2_drift"]
        _check("open_top_stab2", metrics)

    @staticmethod
    def _open_top_metrics(archive):
        y, frames, f = _fields(archive)
        p_h = _hydrostatic(y)
        return {
            "p_dev": max(np.abs(f["p"][i] - p_h).max() for i in frames),
            "z_dev": max(np.abs(f["z"][i] - Z0).max() for i in frames),
        }


class TestPerRockClosure:

    def test_seal_holds(self, tmp_path, monkeypatch):
        archive, totals = _run(
            "case='rock_column' per_rock_closure=True one_rock=False T=600.0",
            tmp_path, monkeypatch)
        y, frames, f = _fields(archive, ("p", "Sg0", "Sg_rock_min0", "Sg_rock_max0"))
        last = frames[-1]
        esf = (y > Y_SEAL0 + 1.0e-9) & (y < Y_SEAL1 - 1.0e-9)
        pool = (y > Y_SEAL0 - 0.02) & (y < Y_SEAL0 - 1.0e-9)
        iface = np.isclose(y, Y_SEAL0) | np.isclose(y, Y_SEAL1)
        lower = np.isclose(y, Y_SEAL0)
        # At an interface node the flash S_g is the pore-volume average of the
        # per-rock S_r, so it lies within [min_r S_r, max_r S_r].
        violation = 0.0
        for i in frames:
            Sg = f["Sg0"][i][iface]
            violation = max(violation,
                            np.max(f["Sg_rock_min0"][i][iface] - Sg),
                            np.max(Sg - f["Sg_rock_max0"][i][iface]))
        metrics = {
            "pool_Sg": f["Sg0"][last][pool].max(),
            "esf_Sg": f["Sg0"][last][esf].max(),
            "bracket_violation": max(violation, 0.0),
            "interface_jump": np.min(f["Sg_rock_max0"][last][lower]
                                     - f["Sg_rock_min0"][last][lower]),
        }
        metrics.update(_drifts(totals))
        _check("rock_column", metrics)

    def test_seal_leaks_without_closure(self, tmp_path, monkeypatch):
        archive, totals = _run(
            "case='rock_column' per_rock_closure=False one_rock=False T=600.0",
            tmp_path, monkeypatch)
        y, frames, f = _fields(archive, ("p", "Sg0"))
        metrics = {"above_Sg": f["Sg0"][frames[-1]][y > Y_SEAL1 + 1.0e-9].max()}
        metrics.update(_drifts(totals))
        _check("rock_column_no_closure", metrics)

    def test_one_rock_unchanged(self, tmp_path, monkeypatch):
        runs = {}
        for closure in (True, False):
            output_dir = tmp_path / str(closure)
            output_dir.mkdir()
            archive, _ = _run(
                "case='rock_column' per_rock_closure={0} one_rock=True T=300.0"
                .format(closure), output_dir, monkeypatch)
            runs[closure] = _fields(archive)
        _, frames, on = runs[True]
        _, _, off = runs[False]
        # No node is shared by two rocks, so the closure must change nothing.
        _check("rock_column_one_rock", {
            "p_diff": max(np.abs(on["p"][i] - off["p"][i]).max() for i in frames),
            "z_diff": max(np.abs(on["z"][i] - off["z"][i]).max() for i in frames),
        })
