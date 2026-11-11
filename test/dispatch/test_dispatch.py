"""proteus.Dispatch: a ymf spec's (formulation, discretization) -> Proteus modules and flags (C5 step 1).

The acceptance check is object by object: each discretization of dambreak.ymf dispatches to exactly the modules,
coefficient flags and numerics TwoPhaseFlow builds for the deck it describes (damBreakLevelSet.py, with
interface=1, and damBreak.py), and the shallow-water spec to what SWFlow builds (sw_p/sw_n, GN_sw_p/GN_sw_n).
Nothing is solved and no mesh is built: TwoPhaseFlowProblem.initializeAll() builds the p and n modules only.

Then the refusals: a cell no module covers, a model in no family, a sub-problem with no role, an unknown
formulation, a cell without the sub-problems it needs, a sharp function left sharp, and every way a proteus: hint
can disagree (D8).
"""
import contextlib
import copy
import importlib.util
import os
import re
import sys

import pytest

pytest.importorskip("ymf.explicit")

from proteus import Context  # noqa: E402
from proteus.Dispatch import DispatchError, dispatch, resolve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.dirname(HERE)
SWFLOW_MODELS = os.path.join(os.path.dirname(resolve("proteus.SWFlow.SWFlowProblem.__file__")), "models")


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec(name):
    from ymf.closure import load
    spec, _, _ = load(os.path.join(HERE, name))
    return spec


@pytest.fixture(scope="module")
def dambreak():
    return _spec("dambreak.ymf")


@pytest.fixture(scope="module")
def lake():
    return _spec("lake_at_rest.ymf")


@contextlib.contextmanager
def _isolated():
    """Leave PETSc's options database and the global Context as they were: building a deck sets solver options
    (rans2p_ksp_type, ...) that a later test on the same worker would otherwise inherit."""
    from petsc4py import PETSc
    options = PETSc.Options()
    before = options.getAll()
    context = getattr(Context, "context", None)
    try:
        yield
    finally:
        after = options.getAll()
        for key in after:
            if key not in before:
                options.delValue(key)
            elif after[key] != before[key]:
                options.setValue(key, before[key])
        Context.contextOptionsString = None
        Context.set(context)


def _two_phase_deck(deck, options=None):
    """The deck's TwoPhaseFlowProblem with its p and n modules built (initializeAll), nothing solved."""
    with _isolated():
        Context.contextOptionsString = options
        module = _load_module(os.path.join(TEST, "TwoPhaseFlow", deck + ".py"),
                              "c5_dispatch_%s_%s" % (deck, abs(hash(options))))
        Context.contextOptionsString = None
        problem = module.myTpFlowProblem
        problem.initializeAll()
    return problem


#: what every dispatched model must say, so that a comparison cannot pass by saying little
REQUIRED_NUMERICS = {"timeIntegration", "stepController", "numericalFluxType", "subgridError", "shockCapturing",
                     "multilevelNonlinearSolver", "quadrature_order"}


def _compare_model(model, p, n, nd):
    """Every coefficient flag, physics attribute and numerics attribute the dispatch gives, against the objects
    (proteus.DispatchRun.differences: the check a run from the spec makes before it solves)."""
    from proteus.DispatchRun import differences
    where = "%s (%s)" % (model.subproblem, model.cell)
    assert REQUIRED_NUMERICS <= set(model.numerics), where
    assert not differences(model, p, n, nd), "%s: %s" % (where, differences(model, p, n, nd))


def _compare_two_phase(found, problem):
    params = list(problem.SystemPhysics.modelDict.items())
    assert [key for key, _ in params] == [m.subproblem for m in found]
    nd = problem.domain.nd
    for model, (key, param) in zip(found, params):
        assert param.index == model.index
        assert param.name == model.name
        _compare_model(model, param.p, param.n, nd)
        assert param.n.runCFL == found.system["runCFL"]
    so = problem.so
    assert issubclass(so.systemStepControllerType, resolve(found.system["systemStepControllerType"]))
    assert so.systemStepExact == found.system["systemStepExact"]
    assert getattr(so, "modelSpinUpList", []) == found.system.get("modelSpinUpList", [])


# ---- acceptance: object by object --------------------------------------------------------------------------------

def test_level_set_five_is_damBreakLevelSet(dambreak):
    found = dispatch(dambreak, "level_set_five")
    assert found.family == "two_phase_level_set"
    assert found.formulation == "nonconservative_inertia"
    assert [m.module for m in found] == ["proteus.mprans.%s" % m for m in ("RANS2P", "VOF", "NCLS", "RDLS", "MCorr")]
    assert [m.role for m in found] == ["velocity_pressure", "volume_fraction", "level_set", "redistancing",
                                       "mass_correction"]
    flow = found["flow"]
    assert flow.coefficients["NONCONSERVATIVE_FORM"] == 1.0 and flow.sources["NONCONSERVATIVE_FORM"] == "cell"
    assert flow.coefficients["useVF"] == 1.0 and flow.sources["useVF"] == "spec, hint agrees"
    assert flow.coefficients["eb_penalty_constant"] == 1.0e6 and flow.sources["eb_penalty_constant"] == "hint"
    assert found["rdls"].sources["epsFact"] == "spec, hint agrees"
    _compare_two_phase(found, _two_phase_deck("damBreakLevelSet"))


def test_level_set_clsvof_is_damBreakLevelSet_interface_1(dambreak):
    found = dispatch(dambreak, "level_set_clsvof")
    assert [m.cell for m in found] == ["rans2p_nonconservative", "clsvof"]
    _compare_two_phase(found, _two_phase_deck("damBreakLevelSet", "interface=1"))


def test_clsvof_projection_is_damBreak(dambreak):
    found = dispatch(dambreak, "clsvof_projection")
    assert [m.cell for m in found] == ["clsvof", "rans3pf_projection", "presinc", "pres", "presinit"]
    assert found["flow"].coefficients["ARTIFICIAL_VISCOSITY"] == 3
    assert found.system["modelSpinUpList"] == [4]
    _compare_two_phase(found, _two_phase_deck("damBreak"))


def _swflow(deck, p_module, n_module):
    """SWFlow's p and n modules for a deck, as parun --SWEs builds them (SWEs_so), nothing solved."""
    with _isolated():
        module = _load_module(os.path.join(TEST, "SWFlow", deck + ".py"), "c5_dispatch_" + deck)
        Context.setFromModule(module)
        sys.path.insert(0, SWFLOW_MODELS)
        try:
            p = _load_module(os.path.join(SWFLOW_MODELS, p_module + ".py"), p_module)
            sys.modules[p_module] = p              # the n module does "from <p> import *"
            n = _load_module(os.path.join(SWFLOW_MODELS, n_module + ".py"), n_module)
        finally:
            sys.path.remove(SWFLOW_MODELS)
            sys.modules.pop(p_module, None)
    return p, n


def test_shallow_water_is_SWFlow(lake):
    found = dispatch(lake, "convex_limited")
    assert found.family == "shallow_water"
    (model,) = found.models
    assert model.module == "proteus.mprans.SW2DCV"
    assert {k: model.sources[k] for k in ("cE", "LUMPED_MASS_MATRIX", "BOUNDS_RELAXATION")} == \
        dict.fromkeys(("cE", "LUMPED_MASS_MATRIX", "BOUNDS_RELAXATION"), "spec")
    p, n = _swflow("dam3Bumps", "sw_p", "sw_n")
    _compare_model(model, p, n, 2)
    # runCFL is the case's (dam3Bumps runs at 0.25); SWEs_so keeps default_so's controller
    from proteus import default_so
    assert resolve(found.system["systemStepControllerType"]) is default_so.systemStepControllerType


def _dispersive(lake, assumption="dispersive"):
    """The lake at rest declared as the relaxed Serre-Green-Naghdi model. ymf cannot write the dispersive
    equations yet (no D/Dt, C42 section 4.3), so the family is read from the declared assumptions alone."""
    doc = copy.deepcopy(lake)
    doc["model"]["physical_model"]["assumptions"] = ["depth_averaged", assumption]
    entry = doc["solution_paths"]["discretizations"][0]
    entry["discretization"] = {"scheme": "graph_viscosity_convex_limiting", "time_integration": "ssp_rk3",
                               "lumped_mass": "false"}
    entry["subproblems"][0]["proteus"] = {"model": "GN_SW2DCV", "INTEGRATE_BY_PARTS_WALLS": "1"}
    return doc


def test_dispersive_shallow_water_is_SWFlow_GN(lake):
    found = dispatch(_dispersive(lake), "convex_limited")
    assert found.family == "dispersive_shallow_water"
    (model,) = found.models
    assert model.module == "proteus.mprans.GN_SW2DCV"
    p, n = _swflow("dSWEs_steady_state", "GN_sw_p", "GN_sw_n")
    _compare_model(model, p, n, 2)


# ---- ymf's words (C5 step 3): the table is keyed on ymf.vocabulary, a spec's older words are aliases -------------

def test_ymf_words_dispatch_as_their_older_aliases_did(dambreak):
    """The projection discretization with the velocity step's ``vbdf`` written ``bdf2`` (ymf's word) dispatches to
    the same cells and the same flags: nothing the aliases accept changes what runs."""
    spec = copy.deepcopy(dambreak)
    older = copy.deepcopy(dambreak)
    _sub(spec, "clsvof_projection", "flow")["time_integration"] = "bdf2"
    _sub(older, "clsvof_projection", "flow")["time_integration"] = "vbdf"
    found, again = dispatch(spec, "clsvof_projection"), dispatch(older, "clsvof_projection")
    assert found.as_dict() == again.as_dict()
    (flow,) = [m for m in found if m.role == "velocity"]
    assert flow.numerics["timeIntegration"] == "proteus.TimeIntegration.VBDF" and flow.numerics["timeOrder"] == 2
    assert [m.role for m in found] == ["level_set_redistanced", "velocity", "pressure_increment", "pressure",
                                       "pressure_spin_up"]


def test_the_scheme_block_in_ymf_words_and_in_the_older_ones_dispatch_alike(lake):
    """``ssp_rk3`` and ``entropy_viscosity_coefficient`` against ``SSP_RK3`` and ``entropy_viscosity_cE``."""
    def build(time, option):
        doc = copy.deepcopy(lake)
        entry = _entry(doc, "convex_limited")
        entry["discretization"] = {"scheme": "graph_viscosity_convex_limiting", "time_integration": time,
                                   option: "2.5", "lumped_mass": "false"}
        entry["subproblems"][0]["time_integration"] = time
        return dispatch(doc, "convex_limited")
    found, older = build("ssp_rk3", "entropy_viscosity_coefficient"), build("SSP_RK3", "entropy_viscosity_cE")
    assert found.as_dict() == older.as_dict()
    (swe,) = found.models
    assert swe.coefficients["cE"] == 2.5 and swe.numerics["timeOrder"] == 3


def test_the_dispersive_family_is_identified_by_dispersive_and_by_the_older_hyperbolic_relaxation(lake):
    for assumption in ("dispersive", "hyperbolic_relaxation"):
        found = dispatch(_dispersive(lake, assumption), "convex_limited")
        assert found.family == "dispersive_shallow_water" and found.models[0].cell == "gn_sw2dcv"


def test_residual_shock_capturing_is_the_RANS3PF_artificial_viscosity_1(dambreak):
    doc = copy.deepcopy(dambreak)
    _sub(doc, "clsvof_projection", "flow")["stabilization"] = {"method": "residual_shock_capturing"}
    (flow,) = [m for m in dispatch(doc, "clsvof_projection") if m.role == "velocity"]
    assert (flow.coefficients["ARTIFICIAL_VISCOSITY"], flow.coefficients["USE_SUPG"]) == (1, 0)
    doc = copy.deepcopy(dambreak)
    _sub(doc, "clsvof_projection", "flow")["stabilization"] = {"method": "shock_capturing"}     # the older word
    (flow,) = [m for m in dispatch(doc, "clsvof_projection") if m.role == "velocity"]
    assert flow.coefficients["ARTIFICIAL_VISCOSITY"] == 1


# ---- refusals ----------------------------------------------------------------------------------------------------

def _entry(doc, name):
    return next(e for e in doc["solution_paths"]["discretizations"] if e["name"] == name)


def _sub(doc, entry, name):
    return next(s for s in _entry(doc, entry)["subproblems"] if s["name"] == name)


def test_conservative_projection_is_a_cell_no_module_covers(dambreak):
    doc = copy.deepcopy(dambreak)
    _entry(doc, "clsvof_projection")["from_weak_form"] = "conservative_inertia"
    with pytest.raises(DispatchError) as e:
        dispatch(doc, "clsvof_projection")
    assert "no Proteus module covers flow" in str(e.value)
    assert "rans3pf_projection: formulation conservative_inertia, the cell's is nonconservative_inertia" \
        in str(e.value)
    assert e.value.subproblem == "flow"
    # the same form with RANS2P is covered: one module, two cells
    doc = copy.deepcopy(dambreak)
    _entry(doc, "level_set_five")["from_weak_form"] = "conservative_inertia"
    flow = dispatch(doc, "level_set_five")["flow"]
    assert (flow.cell, flow.coefficients["NONCONSERVATIVE_FORM"]) == ("rans2p_conservative", 0.0)


def test_an_uncovered_space_is_refused_by_name(dambreak):
    doc = copy.deepcopy(dambreak)
    _sub(doc, "level_set_five", "flow")["finite_element"]["velocity"]["order"] = 2
    with pytest.raises(DispatchError) as e:
        dispatch(doc, "level_set_five")
    message = str(e.value)
    assert "velocity CG2" in message
    assert "rans2p_nonconservative: velocity is CG2, the cell has CG1" in message


def test_an_uncovered_time_integration_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    _sub(doc, "level_set_five", "vof")["time_integration"] = "ssp_rk3"
    with pytest.raises(DispatchError, match="vof: time integration ssp_rk3, the cell has backward_euler"):
        dispatch(doc, "level_set_five")


def test_a_model_in_no_family_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    doc["model"]["physical_model"]["processes"] = ["incompressible_flow"]
    with pytest.raises(DispatchError, match="matches no family"):
        dispatch(doc, "level_set_five")


def test_a_sub_problem_with_no_role_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    _sub(doc, "level_set_five", "rdls")["enforces"] = "norm(∇φ) = 2"
    with pytest.raises(DispatchError, match="sub-problem rdls .* has no role the table knows"):
        dispatch(doc, "level_set_five")


def test_an_unknown_formulation_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    doc["model"]["weak_forms"][0]["label"] = "skew_symmetric_inertia"
    _entry(doc, "level_set_five")["from_weak_form"] = "skew_symmetric_inertia"
    with pytest.raises(DispatchError, match="formulation the table does not know"):
        dispatch(doc, "level_set_five")


def test_a_cell_without_the_sub_problems_it_needs_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    entry = _entry(doc, "clsvof_projection")
    entry["subproblems"] = [s for s in entry["subproblems"] if s["name"] != "pressureInit"]
    with pytest.raises(DispatchError, match="rans3pf_projection .* needs sub-problems with roles pressure_spin_up"):
        dispatch(doc, "clsvof_projection")


def test_a_sharp_heaviside_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    del _entry(doc, "level_set_five")["regularize"]
    with pytest.raises(DispatchError, match="heaviside is sharp"):
        dispatch(doc, "level_set_five")


def test_an_unread_scheme_option_is_refused(lake):
    doc = copy.deepcopy(lake)
    doc["solution_paths"]["discretizations"][0]["discretization"]["limiter"] = "minmod"
    with pytest.raises(DispatchError, match="does not read the scheme option.s. limiter"):
        dispatch(doc, "convex_limited")


# ---- D8: interim hints -------------------------------------------------------------------------------------------

def test_a_hint_that_disagrees_with_the_spec_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    _sub(doc, "level_set_five", "flow")["proteus"]["useVF"] = "0.0"    # the spec lags χ via vof: useVF = 1
    with pytest.raises(DispatchError, match="hint sets useVF = 0.0, and dispatch infers 1.0 from the spec"):
        dispatch(doc, "level_set_five")


def test_a_hint_that_disagrees_with_the_cell_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    _sub(doc, "level_set_five", "flow")["proteus"]["NONCONSERVATIVE_FORM"] = "0.0"
    with pytest.raises(DispatchError, match="NONCONSERVATIVE_FORM = 0.0, and dispatch infers 1.0 from the table"):
        dispatch(doc, "level_set_five")


def test_a_hint_naming_another_module_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    _sub(doc, "level_set_five", "flow")["proteus"]["model"] = "RANS3PF"
    with pytest.raises(DispatchError, match="hint says model RANS3PF, and dispatch infers RANS2P"):
        dispatch(doc, "level_set_five")
    doc = copy.deepcopy(dambreak)
    _sub(doc, "level_set_five", "rdls")["proteus"]["epsFact"] = "1.5"   # the spec binds εr = 0.75 h
    with pytest.raises(DispatchError, match="rdls: the proteus: hint sets epsFact = 1.5, and dispatch infers 0.75"):
        dispatch(doc, "level_set_five")


def test_a_hint_the_cell_does_not_know_is_refused(dambreak):
    doc = copy.deepcopy(dambreak)
    _sub(doc, "level_set_five", "rdls")["proteus"]["weakDirichlet"] = "zero_level_set"
    with pytest.raises(DispatchError, match="hint sets weakDirichlet, which cell rdls .RDLS. does not know"):
        dispatch(doc, "level_set_five")


def test_without_hints_the_defaults_stand(dambreak):
    doc = copy.deepcopy(dambreak)
    for sub in _entry(doc, "level_set_five")["subproblems"]:
        sub.pop("proteus", None)
    found = dispatch(doc, "level_set_five")
    flow = found["flow"]
    assert (flow.coefficients["eb_penalty_constant"], flow.sources["eb_penalty_constant"]) == (100.0, "default")
    assert (flow.coefficients["useVF"], flow.sources["useVF"]) == (1.0, "spec")


# ---- ymf_run -----------------------------------------------------------------------------------------------------

def test_ymf_run_says_what_a_split_discretization_dispatches_to(tmp_path, capsys):
    """ymf_run runs a split discretization from its dispatch (test_dispatch_run.py compares the run with the deck),
    says what it dispatched to, and records the dispatch and the realization in the archive."""
    from proteus.cli import ymfrun
    from ymf import closure
    assert ymfrun.main([os.path.join(HERE, "dambreak_short.ymf"), "--outdir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "level_set_five -- split, dispatched to flow RANS2P, vof VOF, ncls NCLS, rdls RDLS, mcorr MCorr" in out
    assert "ymf_state: closed" in out
    _, outputs, _ = closure.load(str(tmp_path / "dambreak_short.archive.ymf"), check_state=False)
    (output,) = outputs.values()
    steps = output["transformations"]
    assert [m["cell"] for m in steps["dispatch"]["models"]] == \
        ["rans2p_nonconservative", "vof", "ncls", "rdls", "mcorr"]
    realized = steps["realize"]
    assert realized["boundary"]["y+"] == {"region": "top", "setter": "setAtmosphere", "kwargs": {"vof_air": 1.0}}
    assert {side: b["setter"] for side, b in realized["boundary"].items()} == \
        {"x-": "setFreeSlip", "x+": "setFreeSlip", "y-": "setFreeSlip", "y+": "setAtmosphere"}
    assert realized["initial"]["vof"] == {"vof": "smoothed_heaviside(1.5 h, φ0)"}
    assert realized["system"]["nu_0"] == 1.004e-6 and realized["system"]["rho_1"] == 1.205


def test_ymf_run_refuses_a_split_discretization_no_module_covers(tmp_path, capsys):
    """A cell the table does not cover is refused before anything runs, saying which and why (exit 2)."""
    from proteus.cli import ymfrun
    spec = tmp_path / "conservative_projection.ymf"
    spec.write_text("extends: %s\nsolution_paths:\n  discretizations:\n    - name: clsvof_projection\n"
                    "      from_weak_form: conservative_inertia\n" % os.path.join(HERE, "dambreak.ymf"))
    out = tmp_path / "out"
    assert ymfrun.main([str(spec), "--discretization", "clsvof_projection", "--outdir", str(out)]) == 2
    err = capsys.readouterr().err
    assert "clsvof_projection: split into sub-problems (clsvof, flow, pressureInc, pressure, pressureInit), " \
        "and the dispatch refuses it: no Proteus module covers flow" in err
    assert "rans3pf_projection: formulation conservative_inertia, the cell's is nonconservative_inertia" in err
    assert not os.listdir(str(out))


def test_ymf_run_refuses_a_split_discretization_meshed_by_cells(tmp_path, capsys):
    """The deck machinery meshes a split discretization's region by element size: the lake at rest's cells study
    is refused before anything runs."""
    from proteus.cli import ymfrun
    assert ymfrun.main([os.path.join(HERE, "lake_at_rest.ymf"), "--outdir", str(tmp_path)]) == 2
    assert "convex_limited: a split discretization is meshed by element size, one level (mesh: {h: ...}); " \
        "this one gives cells [50, 100, 200]" in capsys.readouterr().err
    assert not os.listdir(str(tmp_path))


def test_ymf_run_refuses_a_boundary_no_setter_imposes_before_anything_runs(tmp_path, capsys):
    """level_set_clsvof dispatches (RANS2P + CLSVOF) and its atmosphere is a setter's (below); an atmosphere that
    holds the pressure at 1 is none, and is refused up front, exit 2, rather than imposing setAtmosphere's other
    data under the spec's name."""
    from proteus.cli import ymfrun
    spec = tmp_path / "high_pressure.ymf"
    spec.write_text("extends: %s\nmodel:\n  strong_form:\n    condition_sets:\n      atmosphere:\n"
                    "        - {variable: p, type: dirichlet, value: 1.0}\n        - {variable: v, type: open}\n"
                    "        - {variable: χ, type: inflow, value: 1.0}\n"
                    % os.path.join(HERE, "dambreak_short_clsvof.ymf"), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    assert ymfrun.main([str(spec), "--outdir", str(out)]) == 2
    err = capsys.readouterr().err
    assert "level_set_clsvof: split into sub-problems (flow, clsvof). It dispatches to flow RANS2P, clsvof CLSVOF, " \
        "and cannot run: region top (side y+): no BC_RANS setter imposes its closed conditions" in err
    assert not os.listdir(str(out))


def test_the_clsvof_atmosphere_is_setAtmosphere_and_the_run_says_what_it_does_not_impose():
    """With CLSVOF as the interface solver no sub-problem carries χ, and ymf states the atmosphere's inflow on φ
    (sign 1, indicator value 1.0): the top is setAtmosphere(vof_air=1.0), as the deck has it, with a note that
    setAtmosphere sets no CLSVOF data. The same atmosphere with VOF carrying χ is setAtmosphere without a note."""
    from proteus.DispatchRun import tank_sides
    from ymf.symbolic.split import split_problem
    for name, note in (("level_set_clsvof", True), ("level_set_five", False)):
        sides = tank_sides(split_problem(_spec("dambreak.ymf"), name))
        region, setter, kwargs, said = sides["y+"]
        assert (setter, kwargs) == ("setAtmosphere", {"vof_air": 1.0}), name
        assert bool(said) == note and (not note or "sets no CLSVOF boundary data" in said and "1.0" in said), said
        assert sides["x-"][1] == "setFreeSlip" and sides["x-"][3] == ""


def _dispatch_record(path, old, new):
    """Edit the dispatch an archive records (rank 0's write), so that a rerun differs from it."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    head, _, tail = text.partition("      dispatch:\n")
    assert old in tail
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "      dispatch:\n" + tail.replace(old, new, 1))


def test_ymf_run_keeps_reproduces_and_refuses_a_split_output(tmp_path, capsys):
    """make for a split discretization: an output already in the archive is kept; --check reruns it and finds it
    bitwise the same; a rerun that differs from its record is refused (exit 1) unless --keep-different keeps it
    as <key>~2; and pro msh has nothing to mesh for it."""
    from proteus.cli import ymfrun
    from ymf import closure
    spec = os.path.join(HERE, "dambreak_short.ymf")
    out = str(tmp_path)
    archive = os.path.join(out, "dambreak_short.archive.ymf")
    assert ymfrun.main([spec, "--outdir", out]) == 0
    # the output's key is the digest of the spec's inputs: read it, so that a change of the spec does not need a
    # constant edited here
    digest = re.search(r"h 0\.2576, levels 1  ([0-9a-f]{12})", capsys.readouterr().out).group(1)
    assert ymfrun.main([spec, "--outdir", out]) == 0
    assert "h 0.2576, levels 1  %s  in the archive" % digest in capsys.readouterr().out
    assert ymfrun.main([archive, "--check"]) == 0
    text = capsys.readouterr().out
    assert "%s  reproduced, bitwise" % digest in text
    assert "every rerun reproduced its record" in text
    # a record the rerun does not match: refused, then kept
    _dispatch_record(archive, "runCFL: 0.25", "runCFL: 0.3")
    assert ymfrun.main([archive, "--check"]) == 1
    text = capsys.readouterr().out
    assert "%s  DIFFERENT, refused" % digest in text and "the dispatch differs" in text
    assert ymfrun.main([archive, "--check", "--keep-different"]) == 0
    assert "%s~2  DIFFERENT, kept" % digest in capsys.readouterr().out
    _, outputs, _ = closure.load(archive, check_state=False)
    assert sorted(k.split("/")[-1] for k in outputs) == [digest, digest + "~2"]
    assert ymfrun.main([spec, "--outdir", out], mesh_only=True) == 0
    assert "a split discretization meshes as part of its solve: level_set_five" in capsys.readouterr().out


def test_ymf_run_ignore_leaves_out_a_split_it_cannot_run(tmp_path, capsys):
    """--ignore runs the split discretizations that can run and leaves out the others, saying why."""
    from proteus.cli import ymfrun
    spec = tmp_path / "both.ymf"
    spec.write_text("""extends: %s
remove:
  - "solution_paths.discretizations[clsvof_projection]"
model:
  strong_form:
    domain: "Ω = [0, 3.22] × [0, 1.8],  t ∈ (0, 0.01]"
    initial_conditions:
      - {field: v, type: constant, value: 0.0}
      - {field: p, type: constant, value: 0.0}
      - {field: φ, type: function, units: m,
         formula: "hypot(max(x − 1.2, 0), max(y − 0.6, 0)) + min(max(x − 1.2, y − 0.6), 0)"}
solution_paths:
  discretizations:
    - name: level_set_five
      mesh: {h: 0.2576}
      solver: {type: nonlinear, nonlinear_solver: newton, linear_solver: superlu, max_iterations: 50}
    - name: level_set_clsvof              # cannot run: no CLSVOF cell integrates in time by SSP RK3
      subproblems:
        - name: clsvof
          time_integration: ssp_rk3
""" % os.path.join(HERE, "dambreak.ymf"), encoding="utf-8")
    out = tmp_path / "out"
    assert ymfrun.main([str(spec), "--outdir", str(out), "--ignore"]) == 1     # not closed: one is left out
    text = capsys.readouterr().out
    assert "level_set_clsvof -- left out (--ignore): level_set_clsvof: split into sub-problems (flow, clsvof), and " \
        "the dispatch refuses it: no Proteus module covers clsvof" in text
    assert "level_set_five -- split, dispatched to flow RANS2P" in text
    assert "1 outputs, 1 new" in text


# ---- round 2 fixes -----------------------------------------------------------------------------------------------

def test_pro_msh_on_split_discretizations_stops_after_meshing(tmp_path, capsys):
    """pro msh meshes (here: nothing to mesh, a split meshes in its solve), writes no archive and exits 0, whatever
    the discretization kind. Before, it fell through to the run's tail: an archive written, exit 1 ("not closed")."""
    from proteus.cli import ymfrun
    out = tmp_path / "out"
    assert ymfrun.main([os.path.join(HERE, "dambreak_short.ymf"), "--outdir", str(out)], mesh_only=True) == 0
    assert "a split discretization meshes as part of its solve: level_set_five" in capsys.readouterr().out
    assert not [f for f in os.listdir(str(out)) if f.endswith(".archive.ymf")]


def test_petsc_options_restored_puts_back_what_the_run_deleted():
    from petsc4py import PETSc
    from proteus.DispatchRun import petsc_options_restored
    options = PETSc.Options()
    options.setValue("c5test_changed", "a")
    options.setValue("c5test_deleted", "b")
    try:
        with petsc_options_restored():
            options.setValue("c5test_changed", "z")
            options.setValue("c5test_added", "c")
            options.delValue("c5test_deleted")
        found = options.getAll()
        assert found["c5test_changed"] == "a" and found["c5test_deleted"] == "b"
        assert "c5test_added" not in found
    finally:
        for key in ("c5test_changed", "c5test_deleted", "c5test_added"):
            options.delValue(key)


def test_a_constant_density_and_viscosity_are_accepted_and_a_missing_one_is_refused(dambreak):
    from proteus.DispatchRun import RealizeError, _two_phase_values
    doc = copy.deepcopy(dambreak)
    coefficients = doc["model"]["strong_form"]["coefficients"]
    coefficients["ρ"] = {"value": 1000.0, "role": "density"}
    coefficients["μ"] = {"value": 1.0e-3, "role": "dynamic_viscosity"}
    found = _two_phase_values(doc, 2)
    assert (found["rho_0"], found["rho_1"], found["nu_0"], found["nu_1"]) == (1000.0, 1000.0, 1.0e-6, 1.0e-6)
    coefficients["ρ"] = {"units": "kg/m3", "role": "density"}
    with pytest.raises(RealizeError, match="coefficient ρ has neither a formula nor a value.*density"):
        _two_phase_values(doc, 2)


# ---- coefficients by role (C5 step 3) ----------------------------------------------------------------------------

def test_the_two_phase_values_are_found_by_role_not_by_name(dambreak):
    """The density, viscosity and gravity are the coefficients that state those roles, whatever the model calls
    them: the same model with ρ, μ and g renamed (and the formulas renamed with them) gives the same values."""
    from proteus.DispatchRun import _two_phase_values
    doc = copy.deepcopy(dambreak)
    strong = doc["model"]["strong_form"]
    base = _two_phase_values(doc, 2)
    strong["coefficients"] = {{"ρ": "density_of_mixture", "μ": "viscosity_of_mixture", "g": "grav"}.get(k, k): v
                              for k, v in strong["coefficients"].items()}
    for name in ("equations", ):
        for e in strong[name]:
            for old, new in (("ρ", "density_of_mixture"), ("μ", "viscosity_of_mixture"), ("g", "grav")):
                if "equation" in e:
                    e["equation"] = e["equation"].replace(old, new)
    again = _two_phase_values(doc, 2)
    assert {k: again[k] for k in base if k != "indicator"} == {k: base[k] for k in base if k != "indicator"}


@pytest.mark.parametrize("role", ["density", "dynamic_viscosity", "gravity"])
def test_a_two_phase_model_without_a_role_is_refused_naming_it(dambreak, role):
    from proteus.DispatchRun import RealizeError, _two_phase_values
    doc = copy.deepcopy(dambreak)
    for entry in doc["model"]["strong_form"]["coefficients"].values():
        if isinstance(entry, dict) and entry.get("role") == role:
            del entry["role"]
    with pytest.raises(RealizeError, match="needs the model's %s, and no coefficient states role: %s" % (
            role.replace("_", " "), role)):
        _two_phase_values(doc, 2)


@pytest.mark.parametrize("role", ["gravity", "manning_coefficient", "porosity", "linear_drag", "quadratic_drag",
                                  "bed_elevation"])
def test_a_shallow_water_model_without_a_role_is_refused_before_anything_runs(lake, role):
    from proteus.DispatchRun import RealizeError, precheck
    from ymf.symbolic.split import split_problem
    doc = copy.deepcopy(lake)
    for entry in doc["model"]["strong_form"]["coefficients"].values():
        if isinstance(entry, dict) and entry.get("role") == role:
            del entry["role"]
    split = split_problem(doc, "convex_limited")
    with pytest.raises(RealizeError, match="no coefficient states role: %s" % role):
        precheck(doc, dispatch(doc, "convex_limited"), split)
