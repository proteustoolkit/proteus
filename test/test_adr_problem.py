"""proteus.ADRProblem: Proteus problems from a plain-data ADR form.

The unit tests build the ADR form by hand -- ADRProblem never needs sympy.
The end-to-end tests go through scripts/ymf_run and need ymf[symbolic].
"""
import os
import subprocess
import sys

import numpy
import pytest

from proteus import ADRProblem

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "ymf_run")


def poisson_adr(diffusion_depends_on=None):
    """-div(a grad u) = 1 on the unit square, written out as ymf emits it."""
    return {
        "dim": 2, "components": ["u"],
        "equations": [{
            "component": "u",
            "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1], "a": ["1.0", "1.0"],
                                "da": {}, "depends_on": diffusion_depends_on or {}}},
            "reaction": {"r": "-1.0", "dr": {}, "depends_on": {}},
        }],
    }


def test_constant_coefficients_get_proteus_constant_flags():
    c = ADRProblem.ADRCoefficients(poisson_adr())
    assert c.diffusion == {0: {0: {0: 'constant'}}}
    assert c.reaction == {0: {0: 'constant'}}
    assert c.potential == {0: {0: 'u'}}
    assert c.sdInfo[(0, 0)][0].tolist() == [0, 1, 2]


def test_any_dependence_makes_diffusion_nonlinear():
    # Proteus's diffusion flags are only 'constant' or 'nonlinear'
    c = ADRProblem.ADRCoefficients(poisson_adr({"u": "linear"}))
    assert c.diffusion == {0: {0: {0: 'nonlinear'}}}


def test_coefficients_are_evaluated_from_their_code_strings():
    adr = poisson_adr()
    adr["equations"][0]["reaction"]["r"] = "-x*y"
    c = ADRProblem.ADRCoefficients(adr)
    x = numpy.random.default_rng(0).random((3, 4, 3))
    q = {'x': x, ('u', 0): numpy.zeros((3, 4)), ('r', 0): numpy.zeros((3, 4)),
         ('a', 0, 0): numpy.zeros((3, 4, 2))}
    c.evaluate(0.0, q)
    numpy.testing.assert_array_equal(q[('r', 0)], -x[..., 0] * x[..., 1])
    assert (q[('a', 0, 0)] == 1.0).all()


def test_reorder_permutes_components_and_their_equations_together():
    problem = {"adr": {"dim": 2, "components": ["v_0", "v_1", "p"],
                       "equations": [{"component": n} for n in ("v_0", "v_1", "p")]},
               "unknowns": {"v": ["v_0", "v_1"], "p": ["p"]}}
    out = ADRProblem.reorder(problem, ["p", "v_0", "v_1"])
    assert out["adr"]["components"] == ["p", "v_0", "v_1"]
    assert [e["component"] for e in out["adr"]["equations"]] == ["p", "v_0", "v_1"]
    assert problem["adr"]["components"] == ["v_0", "v_1", "p"]      # untouched


def test_supg_pspg_refuses_systems_that_are_not_velocity_pressure():
    with pytest.raises(ValueError, match="velocity-pressure"):
        ADRProblem.velocity_pressure({"unknowns": {"u": ["u"], "w": ["w"]}})


def neumann_poisson():
    """-Laplace(u) = -4 on the unit square, u = x^2 + y^2.

    Dirichlet on three sides; on x = 1 the outward diffusive flux
    q.n = -(grad u).n = -2x. Written as Proteus's ADR residual,
    -div(a grad u) + r = 0, so r = +4.
    """
    return {
        "adr": {
            "dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                "reaction": {"r": "4.0", "dr": {}, "depends_on": {}},
            }],
        },
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "dirichlet": [{"component": "u", "region": "three_sides",
                       "where": "x <= tol or y <= tol or y >= 1.0 - tol",
                       "value": "x**2 + y**2"}],
        "neumann": [{"component": "u", "region": "right",
                     "where": "x >= 1.0 - tol", "value": "-2.0*x"}],
        "exact": {"u": "x**2 + y**2"},
    }


def robin_poisson(alpha="1.0", beta="1.0", g="y**2 - 1.0"):
    """-Laplace(u) = -4 on the unit square, u = x^2 + y^2.

    On x = 1 the outward flux is q.n = -(grad u).n = -2, and u = 1 + y^2, so
    alpha u + beta (q.n) = g holds with alpha = beta = 1 and g = y^2 - 1.
    Dirichlet on the other three sides.
    """
    return {
        "adr": {
            "dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                "reaction": {"r": "4.0", "dr": {}, "depends_on": {}},
            }],
        },
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "dirichlet": [{"component": "u", "region": "three_sides",
                       "where": "x <= tol or y <= tol or y >= 1.0 - tol",
                       "value": "x**2 + y**2"}],
        "robin": [{"component": "u", "region": "right", "where": "x >= 1.0 - tol",
                   "alpha": alpha, "beta": beta, "g": g}],
        "exact": {"u": "x**2 + y**2"},
    }


def _rate(problem, cells=(8, 16), **kw):
    errors = [ADRProblem.run(problem, {"u": ("CG", 1)}, c, **kw)[2]["u"] for c in cells]
    return errors, numpy.log2(errors[0] / errors[1])


def test_a_robin_face_converges_at_the_right_rate():
    errors, rate = _rate(robin_poisson())
    assert errors[0] < 1.0e-2
    assert 1.8 < rate < 2.2, (errors, rate)


def test_robin_coefficients_may_vary_along_the_boundary():
    """alpha and beta are expressions, not just numbers."""
    # alpha = 3, beta = 2  =>  g = 3(1 + y^2) + 2(-2)
    errors, rate = _rate(robin_poisson(alpha="3.0", beta="2.0",
                                       g="3.0*(1.0 + y**2) - 4.0"))
    assert errors[0] < 1.0e-2
    assert 1.8 < rate < 2.2, (errors, rate)


def test_a_wrong_robin_right_hand_side_does_not_converge():
    """The sharp check: g off by one stalls the error instead of refining."""
    errors, rate = _rate(robin_poisson(g="y**2"))
    assert errors[0] > 1.0e-1
    assert rate < 0.5, (errors, rate)


def test_robin_with_no_u_coefficient_is_the_neumann_answer():
    """alpha = 0 reduces to a prescribed flux, down a different code path.

    The Robin class writes the flux itself; a neumann condition is applied by
    OneLevelTransport after the numerical flux runs. Agreeing to the bit is
    what says the Robin flux has the same sign and magnitude.
    """
    robin, _ = _rate(robin_poisson(alpha="0.0", g="-2.0"))
    neumann, _ = _rate(neumann_poisson())
    assert robin == neumann


def test_a_robin_condition_with_no_flux_coefficient_is_refused():
    """beta = 0 is a Dirichlet condition and should be imposed strongly."""
    with pytest.raises(ValueError, match="beta = 0"):
        ADRProblem.run(robin_poisson(beta="0.0"), {"u": ("CG", 1)}, 4)


def test_a_robin_condition_on_a_nonexistent_unknown_is_refused():
    problem = robin_poisson()
    problem["robin"][0]["component"] = "v"
    with pytest.raises(ValueError, match="not an unknown"):
        ADRProblem.physics(problem, "adr")


def hydrograph_heat():
    """du/dt - Laplace(u) = f on the unit square, u = (x^2 + y^2)(1 + t).

    f = (x^2 + y^2) - 4(1 + t), so the ADR reaction (residual form
    m_t + div(-a grad u) + r = 0) is r = -f = 4 + 4t - x^2 - y^2.

    On x = 1 the outward diffusive flux is -(grad u).n = -2x(1 + t) = -2(1+t),
    given as a two-sample hydrograph over t in [0, 1]. Linear interpolation
    is EXACT for it, so the solution still converges to the exact answer --
    which is what makes this a test of the interpolation and not just of the
    plumbing. Evaluating the table at a fixed time (ignoring t) gives -2
    throughout and the solution is wrong.
    """
    return {
        "adr": {
            "dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "mass": {"m": "u", "dm": {"u": "1"}, "depends_on": {"u": "linear"}},
                "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                "reaction": {"r": "4.0 + 4.0*t - x**2 - y**2", "dr": {},
                             "depends_on": {}},
            }],
        },
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "time": [0.0, 1.0],
        "has_mass": True,
        "dirichlet": [{"component": "u", "region": "three_sides",
                       "where": "x <= tol or y <= tol or y >= 1.0 - tol",
                       "value": "(x**2 + y**2)*(1.0 + t)"}],
        "neumann": [{"component": "u", "region": "inflow",
                     "where": "x >= 1.0 - tol",
                     "series": [[0.0, -2.0], [1.0, -4.0]]}],
        "initial": {"u": "x**2 + y**2"},
        "exact": {"u": "(x**2 + y**2)*(1.0 + t)"},
    }


def stage_hydrograph():
    """du/dt - Laplace(u) = x on the unit square, u = (1 + t)x.

    On x = 1 the value is 1 + t -- uniform along the face, varying in time --
    which is what a stage record is: a gauge reading, not a formula. Given as
    the two samples [[0, 1], [1, 2]], for which linear interpolation is
    exact. The other three sides carry the formula.

    This is the case a `type: hydrograph` could not express, because it was
    wired to the flux path. Supplying a value by table is orthogonal to which
    kind of condition it is.
    """
    return {
        "adr": {
            "dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "mass": {"m": "u", "dm": {"u": "1"}, "depends_on": {"u": "linear"}},
                "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                "reaction": {"r": "-x", "dr": {}, "depends_on": {}},
            }],
        },
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "time": [0.0, 1.0],
        "has_mass": True,
        "dirichlet": [
            {"component": "u", "region": "gauge", "where": "x >= 1.0 - tol",
             "series": [[0.0, 1.0], [1.0, 2.0]]},
            {"component": "u", "region": "other_sides",
             "where": "x <= tol or y <= tol or y >= 1.0 - tol",
             "value": "(1.0 + t)*x"},
        ],
        "initial": {"u": "x"},
        "exact": {"u": "(1.0 + t)*x"},
    }


def test_a_dirichlet_value_may_be_tabulated():
    """A stage hydrograph: the gauge face is given by table, not formula.

    u = (1 + t)x is linear in x, so P1 represents it exactly and the table is
    exact for 1 + t; what is left is the solver's tolerance, ~1e-8. A frozen
    table gives >1e-2 (next test), six orders away, so this threshold is not
    a near thing.
    """
    errors, _ = _rate(stage_hydrograph(), time={"dt": 0.05})
    assert errors[0] < 1.0e-7, errors


def test_a_frozen_stage_table_does_not_track_the_solution():
    """The sharp check: hold the gauge at its first reading and it is wrong."""
    problem = stage_hydrograph()
    problem["dirichlet"][0]["series"] = [[0.0, 1.0], [1.0, 1.0]]
    errors, _ = _rate(problem, time={"dt": 0.05})
    assert errors[0] > 1.0e-2, errors


def test_the_emitted_p_file_carries_a_tabulated_dirichlet(tmp_path):
    ADRProblem.emit_pn(stage_hydrograph(), {"u": ("CG", 1)}, 8, time={"dt": 0.5},
                       name="stage", directory=str(tmp_path))
    text = (tmp_path / "stage_p.py").read_text()
    assert "def getDBC_u(X, flag):" in text
    assert "numpy.interp(t, T, Q)" in text


def test_a_hydrograph_is_interpolated_in_time():
    """A two-sample table that is exact for the true flux still converges."""
    problem = hydrograph_heat()
    errors = [ADRProblem.run(problem, {"u": ("CG", 1)}, cells,
                             time={"dt": 0.05})[2]["u"]
              for cells in (8, 16)]
    assert errors[0] < 2.0e-2, errors
    rate = numpy.log2(errors[0] / errors[1])
    assert rate > 1.5, (errors, rate)


def test_a_table_holds_its_end_values():
    """Outside the record the first and last samples hold; nothing extrapolates."""
    flux = ADRProblem._value_of({"series": [[0.0, -2.0], [1.0, -4.0]]}, 2)
    assert flux(None, -5.0) == -2.0
    assert flux(None, 0.5) == -3.0
    assert flux(None, 99.0) == -4.0


def test_the_emitted_p_file_carries_the_hydrograph_table(tmp_path):
    ADRProblem.emit_pn(hydrograph_heat(), {"u": ("CG", 1)}, 8, time={"dt": 0.5},
                       name="hyd", directory=str(tmp_path))
    text = (tmp_path / "hyd_p.py").read_text()
    assert "numpy.interp(t, T, Q)" in text
    assert "[0.0, 1.0]" in text and "[-2.0, -4.0]" in text


def test_a_neumann_face_converges_at_the_right_rate():
    """The sharp check: without the flux the solution does not converge at all.

    Dropping the condition leaves x = 1 with the natural zero-flux condition,
    which solves a different problem -- the error stalls near 2.2e-1 and the
    rate goes slightly negative. Getting the sign wrong is worse still. So
    second-order convergence here is evidence the value AND its sign reach
    Proteus intact.
    """
    problem = neumann_poisson()
    errors = [ADRProblem.run(problem, {"u": ("CG", 1)}, cells)[2]["u"]
              for cells in (8, 16)]
    assert errors[0] < 1.0e-2
    rate = numpy.log2(errors[0] / errors[1])
    assert 1.8 < rate < 2.2, (errors, rate)


def test_a_neumann_condition_needs_something_to_flux():
    problem = neumann_poisson()
    del problem["adr"]["equations"][0]["diffusion"]
    with pytest.raises(ValueError, match="no diffusion term"):
        ADRProblem.physics(problem, "adr")


def test_a_neumann_condition_on_a_nonexistent_unknown_is_refused():
    problem = neumann_poisson()
    problem["neumann"][0]["component"] = "v"
    with pytest.raises(ValueError, match="not an unknown"):
        ADRProblem.physics(problem, "adr")


def test_the_emitted_p_file_carries_the_neumann_flux(tmp_path):
    """parun has to solve the same problem the library just did."""
    ADRProblem.emit_pn(neumann_poisson(), {"u": ("CG", 1)}, 8,
                       name="neu", directory=str(tmp_path))
    text = (tmp_path / "neu_p.py").read_text()
    assert "def getDFBC_u(X, flag):" in text
    assert "diffusiveFluxBoundaryConditions = {0: {0: getDFBC_u}}" in text
    # the value and its sign, not just the hook
    assert "-2.0*x" in text or "-2.0 * x" in text


# --- the spec's linear_solver ------------------------------------------------------


CG1 = {"u": ("CG", 1)}


def test_the_linear_solver_choice_follows_the_spec():
    choice = ADRProblem.linear_solver_choice
    plain, singular = {}, {"up_to_constant": ["u"]}
    # omitted: what Proteus picked before it read the field
    assert choice(plain, CG1) == ("superlu", None)
    assert choice(singular, CG1) == ("mumps", ADRProblem.NULL_SPACE_SOLVER)
    assert choice(plain, CG1, "superlu") == ("superlu", None)
    assert choice(plain, CG1, "mumps") == ("mumps", ADRProblem.MUMPS_SOLVER)
    assert choice(plain, CG1, "petsc") == ("petsc", {})
    # petsc_options are applied as petsc's, laid over mumps's
    assert choice(plain, CG1, "PETSc", {"ksp_type": "cg", "ksp_rtol": 1e-10}) == \
        ("petsc", {"ksp_type": "cg", "ksp_rtol": "1e-10"})
    assert choice(plain, CG1, "mumps", {"mat_mumps_icntl_14": "50"})[1] == \
        dict(ADRProblem.MUMPS_SOLVER, mat_mumps_icntl_14="50")


@pytest.mark.parametrize("args, message", [
    (("umfpack",), "has no Proteus solver"),
    (("superlu", {"ksp_type": "cg"}), "apply to linear_solver petsc or mumps"),
    ((None, {"ksp_type": "cg"}), "not the default"),
])
def test_a_linear_solver_proteus_cannot_honour_is_refused(args, message):
    with pytest.raises(ValueError, match=message):
        ADRProblem.linear_solver_choice({}, CG1, *args)


def test_a_singular_operator_is_not_handed_to_superlu():
    with pytest.raises(ValueError, match="needs linear_solver mumps or petsc"):
        ADRProblem.linear_solver_choice({"up_to_constant": ["p"]}, CG1, "superlu")


@pytest.mark.parametrize("linear_solver, petsc_options", [
    ("mumps", None),
    ("petsc", {"ksp_type": "cg", "pc_type": "jacobi"}),
    ("petsc", None),                      # PETSc's own defaults
])
def test_each_linear_solver_gives_the_superlu_answer(linear_solver, petsc_options):
    """Robin faces too: their flux keeps the Dirichlet DOFs under KSP."""
    expected, rate = _rate(robin_poisson())
    errors, got = _rate(robin_poisson(), linear_solver=linear_solver,
                        petsc_options=petsc_options)
    numpy.testing.assert_allclose(errors, expected, rtol=1e-6)
    assert abs(got - rate) < 1e-3


def test_the_numerics_record_the_solver_proteus_used():
    problem = robin_poisson()
    _, _, n, _ = ADRProblem.setup(problem, CG1, 4, linear_solver="petsc",
                                  petsc_options={"ksp_type": "cg"})
    record = ADRProblem.describe(problem, n)
    assert record["linear_solver"] == "KSP_petsc4py"
    assert record["petsc_options"] == {"adr_ksp_type": "cg"}
    _, _, n, _ = ADRProblem.setup(problem, CG1, 4)
    assert ADRProblem.describe(problem, n)["linear_solver"] == "LU"


# --- end to end, through ymf_run ---------------------------------------------------


@pytest.fixture
def poisson_spec(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    path = tmp_path / "poisson.ymf"
    path.write_text('''
model:
  name: "Poisson"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [u]
    equation_formulation: "Poisson"
    equations: ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    domain: "Ω = [0, 1] × [0, 1]"
    boundary_regions: [{name: wall, where: "∂Ω"}]
    boundary_conditions: [{region: wall, variable: u, type: dirichlet, value: 0.0}]
  weak_forms:
    - label: galerkin
      provenance: human_specified
      derivation: "integrate by parts"
      solution_spaces: {trial: H1, test: H1_0}
      bilinear: "∫ ∇u·∇v"
      linear: "∫ f v"
      stabilization_method: none
solution_paths:
  analytical:
    - name: exact
      provenance: human_specified
      method: closed_form
      solution: {formula: "u = sin(πx) sin(πy)"}
  discretizations:
    - name: P1
      provenance: human_specified
      from_weak_form: galerkin
      finite_element: {fields: {family: CG, order: 1}}
      solver: {type: linear}
''', encoding="utf-8")
    return path


def ymf_run(*args):
    return subprocess.run([sys.executable, SCRIPT, *map(str, args)],
                          capture_output=True, text=True)


def rows(stdout):
    return [line.split() for line in stdout.splitlines() if line.strip()[:1].isdigit()]


def test_a_spec_runs_end_to_end_into_its_archive(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--cells", "4", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    table = rows(out.stdout)
    assert [r[0] for r in table] == ["4", "8"] and [r[-1] for r in table] == ["new", "new"]
    assert 1.8 < float(table[1][2]) < 2.2          # P1 in L2
    from ymf import closure
    spec, outputs, _ = closure.load(tmp_path / "poisson.archive.ymf")
    assert spec["model"]["strong_form"]["equations"] == ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    assert spec["solution_paths"]["discretizations"][0]["mesh"] == {"cells": [4, 8]}
    assert spec["composition"]["overrides"][0]["set_by"] == "ymf_run --cells 4 8"
    key = [k for k in outputs if "/cells=8/" in k][0]
    output = outputs[key]
    assert output["realization"] == {"cells": 8, "levels": 1}
    assert output["input"]["sha256"].startswith(key.rsplit("/", 1)[1])
    discrete = output["transformations"]["adr_to_discrete"]
    assert discrete["spaces"] == {"u": "C0_AffineLinearOnSimplexWithNodalBasis"}
    assert discrete["mesh"] == {"nodes_per_side": 9, "levels": 1}
    assert set(output["verification"]["l2_errors"]) == {"u"}
    files = sorted(p.name for p in tmp_path.iterdir() if p.suffix in (".h5", ".xmf"))
    assert files == sorted(closure.file_stem("poisson", k) + s
                           for k in outputs for s in (".h5", ".xmf"))


def test_like_make_it_computes_only_what_is_missing(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    out = ymf_run(poisson_spec, "--cells", "4", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert [r[-1] for r in rows(out.stdout)] == ["archive", "new"]    # "in the archive"
    assert "2 outputs, 1 new" in out.stdout


def test_an_archive_reruns_to_bitwise_identical_results(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    archive = tmp_path / "poisson.archive.ymf"
    out = ymf_run(archive, "--check")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "reproduced, bitwise" in out.stdout
    from ymf import closure
    _, outputs, _ = closure.load(archive)
    (output,) = outputs.values()
    assert len(output["reproduced"]) == 1


def test_a_differing_rerun_is_refused_unless_kept(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    archive = tmp_path / "poisson.archive.ymf"
    # tamper with the record, as a different machine's result would differ
    from ymf.archive import read_document, write_document
    document = read_document(archive)
    (output,) = document["outputs"].values()
    output["verification"]["l2_errors"]["u"] *= 1.5
    write_document(archive, document)
    out = ymf_run(archive, "--check")
    assert out.returncode == 1 and "NOT REPRODUCED" in out.stdout and "L2(u)" in out.stdout
    out = ymf_run(archive, "--check", "--keep-different")
    assert out.returncode == 0, out.stdout[-2000:]
    keys = list(read_document(archive)["outputs"])
    assert len(keys) == 2 and keys[1] == keys[0] + "~2"


def test_an_edited_spec_does_not_silently_replace_the_archive(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    poisson_spec.write_text(poisson_spec.read_text().replace('"Poisson"', '"Poisson, renamed"'))
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path)
    assert out.returncode != 0 and "earlier version of the input" in out.stdout
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path, "--prune")
    assert out.returncode == 0 and "1 outputs, 1 new" in out.stdout


def test_an_inline_multilevel_run_is_self_contained_and_reproducible(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--cells", "2", "--levels", "3", "--inline", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    from ymf import closure
    from ymf.archive import domain_arrays
    archive = tmp_path / "poisson.archive.ymf"
    _, outputs, _ = closure.load(archive)
    ((key, output),) = outputs.items()
    assert "/cells=2/levels=3/" in key and output["storage"] == "inline"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["poisson.ymf", "poisson.archive.ymf", closure.file_stem("poisson", key) + ".xmf"])
    nodes = [a for k, a in domain_arrays(output["approximation"]).items()
             if k.endswith("Geometry")][-1]
    assert nodes.shape == (81, 3)        # the finest level: 2 cells refined twice -> 8x8
    out = ymf_run(archive, "--check")
    assert out.returncode == 0 and "reproduced, bitwise" in out.stdout, out.stdout[-2000:]


def test_emitted_pn_files_reproduce_the_output_bitwise_under_parun(poisson_spec, tmp_path):
    import numpy
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path, "--emit-pn")
    assert out.returncode == 0, out.stderr[-2000:]
    from ymf import closure
    from ymf.archive import domain_arrays, read_ymf
    _, outputs, _ = closure.load(tmp_path / "poisson.archive.ymf")
    ((key, output),) = outputs.items()
    stem = closure.file_stem("poisson", key)
    work = tmp_path / "pn"
    work.mkdir()
    for suffix in ("_p", "_n", "_so"):
        (work / (stem + suffix + ".py")).write_text((tmp_path / (stem + suffix + ".py")).read_text())
    parun = os.path.join(os.path.dirname(sys.executable), "parun")
    run = subprocess.run([parun, stem + "_so.py"], cwd=work, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    mine = domain_arrays(read_ymf(work / (stem + "_pn.ymf"))[0], work)
    recorded = domain_arrays(output["approximation"], tmp_path)
    assert set(mine) == set(recorded)
    assert all(numpy.array_equal(mine[k], recorded[k]) for k in recorded)


def _with_solver(spec_path, solver):
    text = spec_path.read_text(encoding="utf-8")
    assert text.count("      solver: {type: linear}\n") == 1
    spec_path.write_text(text.replace("      solver: {type: linear}\n",
                                      "      solver: %s\n" % solver), encoding="utf-8")


def test_a_spec_s_linear_solver_is_honoured(poisson_spec, tmp_path):
    _with_solver(poisson_spec, "{type: linear, linear_solver: petsc, "
                               "petsc_options: {ksp_type: cg, pc_type: jacobi}}")
    out = ymf_run(poisson_spec, "--cells", "4", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 1.8 < float(rows(out.stdout)[1][2]) < 2.2
    from ymf import closure
    _, outputs, _ = closure.load(tmp_path / "poisson.archive.ymf")
    for output in outputs.values():
        discrete = output["transformations"]["adr_to_discrete"]
        assert discrete["linear_solver"] == "KSP_petsc4py"
        assert discrete["petsc_options"] == {"adr_ksp_type": "cg", "adr_pc_type": "jacobi"}


def test_a_linear_solver_proteus_lacks_stops_the_run(poisson_spec, tmp_path):
    _with_solver(poisson_spec, "{type: linear, linear_solver: umfpack}")
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path)
    assert out.returncode == 2
    assert "P1: linear_solver 'umfpack' has no Proteus solver" in out.stderr
    assert not rows(out.stdout)
    assert not (tmp_path / "poisson.archive.ymf").exists()      # nothing was run
    # --ignore leaves it out; the archive is written, and is not closed
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path, "--ignore")
    assert out.returncode == 1
    assert "P1 -- left out (--ignore): P1: linear_solver 'umfpack'" in out.stdout
    assert "ymf_state: discretized" in out.stdout


def test_a_branch_without_a_mesh_stops_the_run_with_a_hint(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--outdir", tmp_path)
    assert out.returncode == 2
    assert ("P1 is missing a mesh: give mesh: {h: ...} (or {cells: ...} on a box), "
            "or --h (--cells) on the command line") in out.stderr
    assert "rerun with --ignore" in out.stderr
    assert not (tmp_path / "poisson.archive.ymf").exists()


def test_ignore_runs_the_branches_that_can_close(poisson_spec, tmp_path):
    text = poisson_spec.read_text(encoding="utf-8")
    assert text.endswith("      solver: {type: linear}\n")
    poisson_spec.write_text(text + """\
    - name: P2
      provenance: human_specified
      from_weak_form: galerkin
      finite_element: {fields: {family: CG, order: 2}}
      solver: {type: linear}
      mesh: {cells: 4}
""", encoding="utf-8")
    out = ymf_run(poisson_spec, "--outdir", tmp_path, "--ignore")
    assert out.returncode == 1                 # it ran, but the archive is not closed
    assert "P1 -- left out (--ignore): P1 is missing a mesh" in out.stdout
    assert len(rows(out.stdout)) == 1          # P2 on its one mesh
    assert "ymf_state: well-posed" in out.stdout
    assert "not closed: P1 is missing a mesh" in out.stdout
    archive = (tmp_path / "poisson.archive.ymf").read_text(encoding="utf-8")
    assert "\nymf_state: well-posed\n" in archive
    # given its mesh, P1 runs too, P2 is already there, and the archive closes
    out = ymf_run(poisson_spec, "--outdir", tmp_path, "--discretization", "P1",
                  "--cells", "4")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "ymf_state: closed" in out.stdout


def test_emitted_pn_files_reproduce_a_krylov_solve_under_parun(poisson_spec, tmp_path):
    """The n file carries the KSP settings even when no options are given."""
    _with_solver(poisson_spec, "{type: linear, linear_solver: petsc}")
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path, "--emit-pn")
    assert out.returncode == 0, out.stderr[-2000:]
    from ymf import closure
    from ymf.archive import domain_arrays, read_ymf
    _, outputs, _ = closure.load(tmp_path / "poisson.archive.ymf")
    ((key, output),) = outputs.items()
    stem = closure.file_stem("poisson", key)
    n_file = (tmp_path / (stem + "_n.py")).read_text()
    assert "LinearSolvers.KSP_petsc4py" in n_file and "keepDOFs=True" in n_file
    work = tmp_path / "pn"
    work.mkdir()
    for suffix in ("_p", "_n", "_so"):
        (work / (stem + suffix + ".py")).write_text((tmp_path / (stem + suffix + ".py")).read_text())
    parun = os.path.join(os.path.dirname(sys.executable), "parun")
    run = subprocess.run([parun, stem + "_so.py"], cwd=work, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    mine = domain_arrays(read_ymf(work / (stem + "_pn.ymf"))[0], work)
    recorded = domain_arrays(output["approximation"], tmp_path)
    assert all(numpy.array_equal(mine[k], recorded[k]) for k in recorded)


@pytest.fixture
def kovasznay_spec(tmp_path):
    """Kovasznay flow, Taylor-Hood, velocity given all round: p up to a constant."""
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    path = tmp_path / "kovasznay.ymf"
    path.write_text('''
model:
  name: "Kovasznay"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [{name: v, rank: 1}, p]
    equation_formulation: "Navier-Stokes"
    equations:
      - "∇·(ρ v⊗v) − μΔv + ∇p = 0  in Ω"
      - "∇·v = 0  in Ω"
    domain: "Ω = [-0.5, 1.0] × [-0.5, 1.5]"
    boundary_regions: [{name: outer, where: "∂Ω"}]
    boundary_conditions:
      - {region: outer, variable: v, type: dirichlet,
         formula: "(1 - exp(λx) cos(2πy), λ/(2π) exp(λx) sin(2πy))"}
    coefficients:
      ρ: {value: 1.0}
      μ: {value: 0.025}
      λ: "ρ/(2μ) - sqrt(ρ²/(4μ²) + 4π²)"
  weak_forms:
    - label: mixed
      provenance: human_specified
      derivation: "integrate by parts"
      solution_spaces: {trial: "H1² × L2", test: "H1_0² × L2"}
      bilinear: "..."
      linear: "..."
      stabilization_method: none
solution_paths:
  analytical:
    - name: exact
      provenance: human_specified
      method: closed_form
      solution:
        formula: |-
          v = (1 - exp(λx) cos(2πy), λ/(2π) exp(λx) sin(2πy))
          p = (1 - exp(2λx))/2
  discretizations:
    - name: taylor_hood
      provenance: human_specified
      from_weak_form: mixed
      finite_element: {velocity: {family: CG, order: 2}, pressure: {family: CG, order: 1}}
      mesh: {cells: [4, 8]}
      solver: {type: nonlinear}
''', encoding="utf-8")
    return path


def test_a_pressure_up_to_a_constant_is_solved_in_the_null_space(kovasznay_spec, tmp_path):
    import numpy
    out = ymf_run(kovasznay_spec, "--outdir", tmp_path, "--emit-pn")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "L2(p)*" in out.stdout
    table = rows(out.stdout)
    assert 2.5 < float(table[1][2]) and 1.5 < float(table[1][6])      # v_0 P2, p P1 rates
    from ymf import closure
    from ymf.archive import domain_arrays, read_ymf
    archive = tmp_path / "kovasznay.archive.ymf"
    _, outputs, _ = closure.load(archive)
    output = outputs[[k for k in outputs if "/cells=4/" in k][0]]
    assert output["verification"]["modulo_constants"] == ["p"]
    discrete = output["transformations"]["adr_to_discrete"]
    assert discrete["null_space"] == {"constant": ["p"]}
    assert discrete["linear_solver"] == "KSP_petsc4py"
    assert output["transformations"]["solve"]["by"]["threads"]["OMP_NUM_THREADS"] == "1"
    check = ymf_run(archive, "--check")
    assert check.returncode == 0 and check.stdout.count("reproduced, bitwise") == 2, check.stdout[-2000:]
    # and the emitted files reproduce it under parun, single-threaded
    stem = closure.file_stem("kovasznay", [k for k in outputs if "/cells=4/" in k][0])
    work = tmp_path / "pn"
    work.mkdir()
    for suffix in ("_p", "_n", "_so"):
        (work / (stem + suffix + ".py")).write_text((tmp_path / (stem + suffix + ".py")).read_text())
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    parun = os.path.join(os.path.dirname(sys.executable), "parun")
    run = subprocess.run([parun, stem + "_so.py"], cwd=work, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    mine = domain_arrays(read_ymf(work / (stem + "_pn.ymf"))[0], work)
    recorded = domain_arrays(output["approximation"], tmp_path)
    assert all(numpy.array_equal(mine[k], recorded[k]) for k in recorded)


def test_a_pinned_taylor_hood_pressure_solves_through_mumps(kovasznay_spec, tmp_path):
    """A kept pressure datum needs a diagonal the continuity equation lacks.

    Under KSP the Dirichlet rows stay in the system and become u = g, which
    needs the row's diagonal entry; Taylor-Hood's continuity equation has
    no pressure-pressure term, so without a structural zero it raised
    "Jacobian has a zero row". With it, MUMPS gives SuperLU's answer.
    """
    text = kovasznay_spec.read_text(encoding="utf-8")
    text = text.replace(
        'boundary_regions: [{name: outer, where: "∂Ω"}]',
        'boundary_regions: [{name: outer, where: "∂Ω"},\n'
        '                       {name: datum, where: "x = -0.5 and y = -0.5"}]')
    text = text.replace(
        "    coefficients:\n      ρ: {value: 1.0}",
        "      - {region: datum, variable: p, type: dirichlet, formula: \"(1 - exp(2λx))/2\"}\n"
        "    coefficients:\n      ρ: {value: 1.0}")
    entry = text[text.index("    - name: taylor_hood"):]
    text = text.replace(entry, "\n".join([
        entry.replace("taylor_hood", "th_superlu").replace(
            "solver: {type: nonlinear}", "solver: {type: nonlinear, linear_solver: superlu}").rstrip("\n"),
        entry.replace("taylor_hood", "th_mumps").replace(
            "solver: {type: nonlinear}", "solver: {type: nonlinear, linear_solver: mumps}")]))
    kovasznay_spec.write_text(text, encoding="utf-8")
    out = ymf_run(kovasznay_spec, "--cells", "4", "--outdir", tmp_path)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "L2(p)*" not in out.stdout             # pinned: no null space
    superlu, mumps = rows(out.stdout)
    # one mesh, so no rate columns: cells, L2(v_0), L2(v_1), L2(p), key, status
    numpy.testing.assert_allclose([float(v) for v in mumps[1:4]],
                                  [float(v) for v in superlu[1:4]], rtol=1e-8)


PERIODIC_POISSON = """
model:
  name: "periodic Poisson"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [u]
    equation_formulation: "Poisson"
    equations: ["-Δu = 5π² sin(2πx) sin(πy)  in Ω"]
    domain: "Ω = [0, 1] × [0, 1]"
    boundary_regions:
      - {name: walls, where: "y = 0 or y = 1"}
      - {name: ends, where: "x = 0 or x = 1"}
    boundary_conditions:
      - {region: walls, variable: u, type: dirichlet, value: 0.0}
      - {region: ends, variable: u, type: periodic}
solution_paths:
  analytical:
    - {name: exact, provenance: human_specified, method: closed_form,
       solution: {formula: "u = sin(2πx) sin(πy)"}}
  discretizations:
    - {name: superlu, provenance: human_specified, finite_element: {fields: {family: CG, order: 1}},
       solver: {type: linear, linear_solver: superlu}}
    - {name: mumps, provenance: human_specified, finite_element: {fields: {family: CG, order: 1}},
       solver: {type: linear, linear_solver: mumps}}
    - {name: cg, provenance: human_specified, finite_element: {fields: {family: CG, order: 1}},
       solver: {type: linear, linear_solver: petsc,
                petsc_options: {ksp_type: cg, pc_type: jacobi, ksp_rtol: "1e-12"}}}
"""


def _errors_by_branch(stdout):
    """{branch: [error columns of its first row]} from ymf_run's report."""
    found, branch = {}, None
    for line in stdout.splitlines():
        if " -- " in line and not line.startswith(" "):
            branch = line.split(" -- ")[0]
        elif branch and line.strip()[:1].isdigit() and branch not in found:
            found[branch] = [float(v) for v in line.split()[1:] if "e" in v and "." in v]
    return found


def test_a_periodic_problem_solves_the_same_through_each_linear_solver(tmp_path):
    """The serial periodic path under KSP_petsc4py.

    It used to solve with the periodic faces disconnected (L2 error 0.62
    where SuperLU gives 0.047): keeping the Dirichlet DOFs in the system,
    as KSP needs, skipped building the periodic DOF map altogether, and the
    PETSc vectors were sized by the space's DOFs rather than the merged
    unknowns.
    """
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    spec = tmp_path / "pp.ymf"
    spec.write_text(PERIODIC_POISSON, encoding="utf-8")
    out = ymf_run(spec, "--cells", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    errors = _errors_by_branch(out.stdout)
    assert set(errors) == {"superlu", "mumps", "cg"}
    assert 0.04 < errors["superlu"][0] < 0.05
    for branch in ("mumps", "cg"):
        numpy.testing.assert_allclose(errors[branch], errors["superlu"], rtol=1e-6)


PERIODIC_CHANNEL = """
model:
  name: "plane Poiseuille, periodic"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [{name: v, rank: 1}, p]
    equation_formulation: "incompressible Navier-Stokes"
    equations:
      - "ρ ∂v/∂t + ∇·(ρ v⊗v) − μΔv + ∇p = f  in Ω"
      - "∇·v = 0  in Ω"
    domain: "Ω = [0, 4] × [0, 1]"
    boundary_regions:
      - {name: walls, where: "y = 0 or y = 1"}
      - {name: ends, where: "x = 0 or x = 4"}
      - {name: datum, where: "x = 0 and y = 0 or x = 4 and y = 0"}
    boundary_conditions:
      - {region: walls, variable: v, type: dirichlet, value: 0.0}
      - {region: ends, variable: v, type: periodic}
      - {region: ends, variable: p, type: periodic}
      - {region: datum, variable: p, type: dirichlet, value: 0.0}
    coefficients:
      ρ: {value: 1.0}
      μ: {value: 1.0}
      f: {formula: "(8, 0)"}
  weak_forms:
    - {label: galerkin, provenance: human_specified, derivation: "by parts",
       solution_spaces: {trial: "H1² × L2", test: "H1_0² × L2"},
       bilinear: "...", linear: "...", stabilization_method: none}
    - {label: stabilized, provenance: human_specified, derivation: "by parts",
       solution_spaces: {trial: "H1² × H1", test: "H1_0² × H1"},
       bilinear: "...", linear: "...", stabilization_method: "SUPG/PSPG"}
solution_paths:
  analytical:
    - {name: exact, provenance: human_specified, method: closed_form,
       solution: {formula: "v = (4 y (1 - y), 0)\\np = 0"}}
  discretizations:
    - {name: th_mumps, provenance: human_specified, from_weak_form: galerkin,
       finite_element: {velocity: {family: CG, order: 2}, pressure: {family: CG, order: 1}},
       solver: {type: nonlinear, linear_solver: mumps}}
    - {name: p1_superlu, provenance: human_specified, from_weak_form: stabilized,
       finite_element: {velocity: {family: CG, order: 1}, pressure: {family: CG, order: 1}},
       solver: {type: nonlinear, linear_solver: superlu}}
    - {name: p1_mumps, provenance: human_specified, from_weak_form: stabilized,
       finite_element: {velocity: {family: CG, order: 1}, pressure: {family: CG, order: 1}},
       solver: {type: nonlinear, linear_solver: mumps}}
"""


def test_a_periodic_velocity_pressure_system_solves_through_mumps(tmp_path):
    """Mixed spaces, a pinned periodic pressure, and unequal free counts.

    Pressure is pinned at the periodic corners and velocity at the walls, so
    after merging the periodic pairs the components have different numbers
    of unknowns; Transport must not interleave them then. Taylor-Hood holds
    this flow exactly; SUPG/PSPG must give SuperLU's errors.
    """
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    spec = tmp_path / "channel.ymf"
    spec.write_text(PERIODIC_CHANNEL, encoding="utf-8")
    out = ymf_run(spec, "--cells", "4", "--outdir", tmp_path)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    errors = _errors_by_branch(out.stdout)
    assert set(errors) == {"th_mumps", "p1_superlu", "p1_mumps"}, out.stdout[-2000:]
    assert max(errors["th_mumps"]) < 1e-10
    numpy.testing.assert_allclose(errors["p1_mumps"], errors["p1_superlu"], rtol=1e-6,
                                  atol=1e-12)


# --- periodic conditions on several ranks -------------------------------------------


def _mpiexec():
    """The launcher beside this interpreter: a mismatched one runs N single-rank
    jobs (see test_archive_parallel_metadata), so these tests also check the
    ranks the archive records."""
    import shutil
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _mpiexec()


def ymf_run_on(ranks, *args):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    return subprocess.run([MPIEXEC, "-n", str(ranks), sys.executable, SCRIPT, *map(str, args)],
                          capture_output=True, text=True, timeout=600, env=env)


def _recorded_ranks(archive):
    from ymf import closure
    _, outputs, _ = closure.load(str(archive), check_state=False)
    return {o["transformations"]["solve"]["by"]["ranks"] for o in outputs.values()}


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks", [2, 3])
def test_a_periodic_problem_solves_on_several_ranks_as_on_one(tmp_path, ranks):
    """Proteus's parallel periodic numbering, on an element partition without
    overlap: the mesh must be partitioned that way too (it was meshed as the
    default node partition with overlap, and every overlapping element was
    assembled twice)."""
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    spec = tmp_path / "pp.ymf"
    spec.write_text(PERIODIC_POISSON, encoding="utf-8")
    serial = ymf_run(spec, "--cells", "8", "--outdir", tmp_path / "1")
    errors = _errors_by_branch(serial.stdout)
    out = ymf_run_on(ranks, spec, "--cells", "8", "--discretization", "mumps",
                     "--discretization", "cg", "--outdir", tmp_path / "n")
    assert out.returncode == 1, out.stdout[-2000:] + out.stderr[-2000:]   # superlu left out
    parallel = _errors_by_branch(out.stdout)
    assert set(parallel) == {"mumps", "cg"}, out.stdout[-2000:]
    for branch in ("mumps", "cg"):
        numpy.testing.assert_allclose(parallel[branch], errors[branch], rtol=1e-8)
    assert _recorded_ranks(tmp_path / "n" / "pp.archive.ymf") == {ranks}


ADVECTED_PERIODIC_POISSON = PERIODIC_POISSON.replace(
    '"-Δu = 5π² sin(2πx) sin(πy)  in Ω"',
    '"∇·((1, 0) u) - Δu = 5π² sin(2πx) sin(πy) + 2π cos(2πx) sin(πy)  in Ω"')


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks", [2, 3])
def test_an_advected_periodic_problem_solves_on_several_ranks_as_on_one(tmp_path, ranks):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    spec = tmp_path / "pa.ymf"
    spec.write_text(ADVECTED_PERIODIC_POISSON, encoding="utf-8")
    serial = ymf_run(spec, "--cells", "8", "--outdir", tmp_path / "1")
    errors = _errors_by_branch(serial.stdout)
    out = ymf_run_on(ranks, spec, "--cells", "8", "--discretization", "mumps",
                     "--discretization", "cg", "--outdir", tmp_path / "n")
    assert out.returncode == 1, out.stdout[-2000:] + out.stderr[-2000:]   # superlu left out
    parallel = _errors_by_branch(out.stdout)
    assert set(parallel) == {"mumps", "cg"}, out.stdout[-2000:]
    for branch in ("mumps", "cg"):
        numpy.testing.assert_allclose(parallel[branch], errors[branch], rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_what_the_parallel_periodic_path_cannot_solve_is_refused(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    channel = tmp_path / "pc.ymf"
    channel.write_text(PERIODIC_CHANNEL, encoding="utf-8")
    out = ymf_run_on(2, channel, "--cells", "4", "--discretization", "th_mumps",
                     "--outdir", tmp_path)
    assert out.returncode == 2 and "need every component in CG1" in out.stderr
    assert not list(tmp_path.glob("*.h5"))                    # nothing was solved


# --- advection across a rank interface -----------------------------------------------


ADV_PARTITION_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "adr_advection_partition_mpi.py")


def _adv_partition_run(ranks, cwd, **overrides):
    """One run of adr_advection_partition_mpi.py, with the partition pinned.

    PETSc's default partitioner is not reproducible: the same case came back as
    2.3e-02, 2.9e-02 and 3.1e-02 on three runs, so a regression could not be
    told from the noise. MATPARTITIONINGCURRENT keeps the initial contiguous
    partition, which is the same every time and is in every PETSc build.
    """
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PETSC_OPTIONS="-mat_partitioning_type current",
               PART_TYPE="element", NLAYERS="0", BDIR="y", CELLS="16")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    # even for one rank (test_archive_parallel_metadata.run_ranks says why)
    argv = [sys.executable, ADV_PARTITION_HELPER]
    if MPIEXEC:
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=600, env=env,
                         cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    result = dict(item.split("=") for item in line.split()[1:])
    result["interfaces"] = sum(
        int(dict(i.split("=") for i in l.split()[1:])["interfaces"])
        for l in out.stdout.splitlines() if l.startswith("RANK"))
    return result


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks", [2, 3])
def test_advection_is_not_applied_on_the_faces_between_ranks(ranks, tmp_path):
    """An element partition with no overlap gives the serial answer.

    Every face this rank shares with a neighbour is an exterior face of its
    subdomain, so the advective boundary flux was applied there -- to the
    inside of the domain. Diffusion was already safe (it is driven by
    isDOFBoundary) and so were the Dirichlet conditions (getDBC returns None
    for material type 0), which is why only advection moved. On 16x16 cells
    over 2 ranks the L2 error went from 1.4e-03 to 3.1e-02.
    """
    serial = _adv_partition_run(1, tmp_path)
    assert serial["interfaces"] == 0                       # nothing masked in serial
    parallel = _adv_partition_run(ranks, tmp_path)
    assert parallel["interfaces"] > 0, parallel            # the case really has interfaces
    numpy.testing.assert_allclose(float(parallel["L2_u"]), float(serial["L2_u"]),
                                  rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_advection_along_a_rank_interface_is_untouched(tmp_path):
    """The guard does not fire where it must not.

    With b = (1, 0) the pinned cut runs along the flow, so b.n is zero on every
    interface face and there is nothing to remove. This is why the test above
    advects across the cut: it would pass either way.
    """
    serial = _adv_partition_run(1, tmp_path, BDIR="x")
    parallel = _adv_partition_run(2, tmp_path, BDIR="x")
    numpy.testing.assert_allclose(float(parallel["L2_u"]), float(serial["L2_u"]),
                                  rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("flux", [
    "Advection_DiagonalUpwind",
    "Advection_DiagonalUpwind_IIPG_exterior",
    "Advection_DiagonalUpwind_Diffusion_IIPG_exterior",
    "Advection_DiagonalUpwind_Diffusion_SIPG_exterior",
    "Advection_DiagonalUpwind_Diffusion_NIPG_exterior"])
@pytest.mark.parametrize("advonly", [0, 1])
def test_weak_advective_flux_classes_agree_with_serial_across_a_rank_cut(flux, advonly, tmp_path):
    """The CG weak-boundary upwind classes need no rank-interface mask.

    Measured rather than assumed (C0, F1): with advection across the pinned
    cut, an element partition with no overlap, these classes give the serial
    answer to rounding, with diffusion and with pure advection (Dirichlet on
    the inflow side only, free outflow, so the cut carries both inflow and
    outflow faces). Their exterior advective flux is nonzero on the interface
    faces, but the two ranks' terms there cancel in the assembled residual.
    StrongDirichlet's NoBC kernel does not cancel, which is why it needed #76.
    The test is here so that stays true.
    """
    kw = dict(FLUX=flux, ADVONLY=advonly or "")
    serial = _adv_partition_run(1, tmp_path, **kw)
    parallel = _adv_partition_run(2, tmp_path, **kw)
    assert parallel["interfaces"] > 0, parallel
    numpy.testing.assert_allclose(float(parallel["L2_u"]), float(serial["L2_u"]),
                                  rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("bdir", ["x", "y"])
def test_dg_gives_the_serial_answer_on_several_ranks(bdir, tmp_path):
    """DG needs every face neighbour of an owned element on the rank.

    ADRProblem used to leave the default partition, node with one layer of
    overlap, which does not provide that: the missing neighbours' faces were
    assembled as domain boundary and 16x16 Poisson went from 1.0e-03 on one
    rank to 1.3e-01 on two, with or without advection. The partition is left
    as ADRProblem.setup chose it (empty PART_TYPE / NLAYERS).
    """
    kw = dict(SPACE="DG", PART_TYPE="", NLAYERS="", BDIR=bdir)
    serial = _adv_partition_run(1, tmp_path, **kw)
    parallel = _adv_partition_run(2, tmp_path, **kw)
    numpy.testing.assert_allclose(float(parallel["L2_u"]), float(serial["L2_u"]),
                                  rtol=1e-6)


def test_parallel_periodic_is_chosen_by_rank_count_space_and_advection():
    problem = {"periodic": [{"component": "u"}],
               "adr": {"equations": [{"component": "u", "diffusion": {}}]}}
    cg1, cg2 = {"u": ("CG", 1)}, {"u": ("CG", 2)}
    assert ADRProblem.parallel_periodic(problem, cg1, ranks=1) is False
    assert ADRProblem.parallel_periodic(problem, cg2, ranks=1) is False   # serial: any space
    assert ADRProblem.parallel_periodic(problem, cg1, ranks=2) is True
    assert ADRProblem.parallel_periodic({"periodic": []}, cg2, ranks=2) is False
    with pytest.raises(ValueError, match="need every component in CG1"):
        ADRProblem.parallel_periodic(problem, cg2, ranks=2)
    advected = dict(problem, adr={"equations": [{"component": "u", "advection": {"f": ["u"]}}]})
    assert ADRProblem.parallel_periodic(advected, cg1, ranks=2) is True


def test_an_old_style_model_is_meshed_with_its_n_modules_partition():
    """setDomainFromPhysics copied the mesh size from n but not the
    partition, so n's element partition without overlap was meshed as the
    default node partition with one layer of overlap."""
    from types import SimpleNamespace
    from proteus import MeshTools
    p = SimpleNamespace(domain=None, nd=2, L=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0),
                        name="box", polyfile=None, meshfile=None, genMesh=True)
    n = SimpleNamespace(nnx=5, nny=5, nnz=None, nn=None, nLevels=1, hex=False, quad=False,
                        triangleFlag=0, triangleOptions=None,
                        parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.element,
                        nLayersOfOverlapForParallel=0)
    MeshTools.setDomainFromPhysics(p, n)
    options = MeshTools.getMeshOptions(p.domain)
    assert options.parallelPartitioningType == MeshTools.MeshParallelPartitioningTypes.element
    assert options.nLayersOfOverlapForParallel == 0


# --- regions that are not boxes: unstructured meshes of size h ----------------------

L_SHAPE = '''    domain:
      kind: piecewise_linear
      dimension: 2
      vertices: [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
      segments: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
'''


@pytest.fixture
def lshape_spec(poisson_spec):
    """The Poisson spec on an L-shaped region, on which sin(pi x) sin(pi y)
    still vanishes, meshed by h."""
    text = poisson_spec.read_text(encoding="utf-8")
    box = '    domain: "Ω = [0, 1] × [0, 1]"\n'
    assert box in text
    text = text.replace(box, L_SHAPE).replace(
        "      solver: {type: linear}\n",
        "      solver: {type: linear}\n      mesh: {h: [0.25, 0.125]}\n")
    path = poisson_spec.parent / "lshape.ymf"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_region_is_meshed_by_h_and_a_box_by_cells():
    problem = {"adr": poisson_adr(), "domain": {"lower": [0, 0], "upper": [2, 2],
                                               "region": {"kind": "piecewise_linear"}}}
    with pytest.raises(ValueError, match="meshed by h"):
        ADRProblem.numerics(problem, {"u": ("CG", 1)}, 4)
    n = ADRProblem.numerics(problem, {"u": ("CG", 1)}, None, h=0.5)
    assert n.triangleOptions == "VApq30Dena0.12500000"
    box = {"adr": poisson_adr(), "domain": {"lower": [0, 0], "upper": [1, 1]}}
    with pytest.raises(ValueError, match="cells per side, not h"):
        ADRProblem.numerics(box, {"u": ("CG", 1)}, None, h=0.5)
    assert ADRProblem.mesh_constraint(0.5, 3) == "VApq1.35q12feena%e" % (0.5 ** 3 / 6.0)


def test_an_l_shaped_region_converges_at_second_order(lshape_spec, tmp_path):
    out = ymf_run(lshape_spec, "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    table = rows(out.stdout)
    assert [r[0] for r in table] == ["0.25", "0.125"]
    assert 1.8 < float(table[1][2]) < 2.4               # P1, L2: rate 2
    archive = (tmp_path / "lshape.archive.ymf").read_text(encoding="utf-8")
    assert "P1/h=0.25/" in archive and "P1/h=0.125/" in archive
    assert "generator_options: VApq30Dena" in archive
    assert "\nymf_state: closed\n" in archive    # every planned output is there
