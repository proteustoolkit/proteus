"""The ``pro`` command line (proteus.cli) and proteus.runtime.

The end-to-end test runs one model with ``parun`` and with ``pro run`` and
requires the two archives to be bitwise identical: ``pro run`` is meant to
reproduce parun's behaviour while having its own command line.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from proteus import runtime
from proteus.cli import build_parser, missing
from proteus.cli.run import model_kind

HERE = os.path.dirname(os.path.abspath(__file__))


def pro(*args, cwd=None):
    """Run ``python -m proteus.cli`` (no installed ``pro`` needed)."""
    return subprocess.run([sys.executable, "-m", "proteus.cli"] + list(args),
                          cwd=cwd, capture_output=True, text=True)


def test_help_does_not_start_mpi_or_petsc():
    code = ("import sys; from proteus.cli import build_parser; "
            "build_parser().format_help(); "
            "print('petsc4py' in sys.modules, 'mpi4py' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_help_lists_the_commands():
    out = pro("--help")
    assert out.returncode == 0
    for command in ("run", "spec", "weak", "dis"):
        assert command in out.stdout


@pytest.mark.skipif(sys.version_info < (3, 14),
                    reason="argparse suggests commands from Python 3.14")
def test_typo_suggests_the_command():
    out = pro("rnu")
    assert out.returncode == 2
    assert "maybe you meant 'run'?" in out.stderr


def test_frontend_command_explains_what_it_needs(monkeypatch, capsys):
    from proteus.cli import dis
    monkeypatch.setattr(dis, "missing", lambda *names: ["sympy"])
    assert dis.run(build_parser().parse_args(["dis", "x.ymf"])) == 2
    assert "ymf[symbolic]" in capsys.readouterr().err


def test_missing():
    assert missing("os", "no_such_module_here", "no_such_pkg.sub") == [
        "no_such_module_here", "no_such_pkg.sub"]


@pytest.mark.parametrize("inputs, message", [
    (["a.py", "b.py", "c.py"], "give one model module"),
    (["model.txt"], "not a .py file"),
    (["no_such_model.py"], "no such file"),
])
def test_run_rejects_bad_inputs(tmp_path, inputs, message):
    out = pro("run", *inputs, cwd=tmp_path)
    assert out.returncode == 2
    assert message in out.stderr


def test_model_kind(tmp_path):
    so = tmp_path / "m_so.py"
    so.write_text("from proteus.default_so import *\npnList = [('p', 'n')]\n")
    case = tmp_path / "case.py"
    case.write_text("# a TwoPhaseFlow case\nmyTpFlowProblem = None\n")
    assert model_kind(str(so)) == "so"
    assert model_kind(str(case)) == "case"


def test_petsc_argv(tmp_path):
    f = tmp_path / "petsc.options"
    f.write_text(textwrap.dedent("""\
        -ksp_type gmres   # the solver
        # a whole-line comment
        -pc_type asm
        """))
    assert runtime.petsc_argv("-pc_type lu", str(f), program="prog") == [
        "prog", "-ksp_type", "gmres", "-pc_type", "asm", "-pc_type", "lu"]
    assert runtime.petsc_argv(["-a", "1"], program="prog") == ["prog", "-a", "1"]
    assert runtime.petsc_argv(program="prog") == ["prog"]


def test_petsc_argv_program_is_never_an_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["-c"])
    assert runtime.petsc_argv() == ["proteus"]
    monkeypatch.setattr(sys, "argv", [])
    assert runtime.petsc_argv() == ["proteus"]


def _parun_defaults():
    """dest -> default of each option parun's parser declares, read from
    scripts/parun without running it (optparse's default is None)."""
    import ast
    path = os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "parun")
    if not os.path.exists(path):
        pytest.skip("needs the source tree's scripts/parun")
    with open(path) as f:
        tree = ast.parse(f.read())
    defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_option"):
            kw = {k.arg: k.value for k in node.keywords}
            if "dest" in kw:
                value = kw.get("default")
                defaults[ast.literal_eval(kw["dest"])] = (
                    None if value is None else eval(compile(ast.Expression(value), path, "eval")))
    return defaults


def test_run_options_match_parun_defaults():
    parun = _parun_defaults()
    defaults = runtime.RunOptions()
    for name in vars(defaults):
        expected = parun[name]
        if name == "hotStart":      # parun's default is None, i.e. off
            expected = bool(expected)
        assert getattr(defaults, name) == expected, name
    assert runtime.RunOptions() is not runtime.RunOptions()


def test_iproteus_is_a_shim_over_runtime(tmp_path):
    code = ("import warnings; warnings.simplefilter('error', DeprecationWarning)\n"
            "try:\n    import proteus.iproteus\nexcept DeprecationWarning as w:\n"
            "    print('warned:', w)\n"
            "warnings.simplefilter('ignore', DeprecationWarning)\n"
            "import sys; sys.modules.pop('proteus.iproteus', None)\n"
            "from proteus.iproteus import *\n"
            "from proteus import runtime\n"
            "print(type(opts) is runtime.RunOptions, comm.size(), "
            "NumericalSolution.__name__, callable(reload))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "warned: proteus.iproteus is deprecated" in out.stdout
    assert out.stdout.splitlines()[-1] == "True 1 proteus.NumericalSolution True"


def _datasets(path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: out.__setitem__(name, np.asarray(obj[()]))
                     if isinstance(obj, h5py.Dataset) else None)
    return out


@pytest.mark.skipif(shutil.which("parun") is None, reason="parun not on PATH")
def test_run_matches_parun_bitwise(tmp_path):
    import numpy as np
    sources = [os.path.join(HERE, "ci", f)
               for f in ("adr.py", "ladr_2d_p.py", "ladr_2d_n.py")]
    for name in ("parun", "pro"):
        (tmp_path / name).mkdir()
        for s in sources:
            shutil.copy(s, tmp_path / name)
    ran = subprocess.run(["parun", "ladr_2d_p.py", "ladr_2d_n.py"],
                         cwd=tmp_path / "parun", capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr[-2000:]
    ran = pro("run", "ladr_2d_p.py", "ladr_2d_n.py", cwd=tmp_path / "pro")
    assert ran.returncode == 0, ran.stderr[-2000:]

    expected = _datasets(tmp_path / "parun" / "ladr_2d.h5")
    actual = _datasets(tmp_path / "pro" / "ladr_2d.h5")
    assert sorted(actual) == sorted(expected)
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name
    assert ((tmp_path / "pro" / "ladr_2d.ymf").read_bytes()
            == (tmp_path / "parun" / "ladr_2d.ymf").read_bytes())


TETGEN_MODEL = ["poisson_3d_tetgen_p.py", "poisson_3d_tetgen_c0p1_n.py"]


def _copy_ci_models(directory):
    directory.mkdir()
    for name in os.listdir(os.path.join(HERE, "ci")):
        if name.endswith(".py"):
            shutil.copy(os.path.join(HERE, "ci", name), directory)


def test_mesh_files_plan():
    from proteus import Domain, MeshTools
    box = Domain.RectangularDomain(L=[1.0, 1.0])
    assert MeshTools.meshFiles(box, MeshTools.getMeshOptions(box)) is None
    plc = Domain.PiecewiseLinearComplexDomain(name="plc")
    plc.polyfile = "plc"
    options = MeshTools.getMeshOptions(plc)
    options.triangleOptions = "VApq1.35q12feena0.01"
    plan = MeshTools.meshFiles(plc, options)
    assert plan["generator"] == "tetgen"
    assert plan["command"] == "tetgen -VApq1.35q12feena0.01 plc.poly"
    assert plan["inputs"] == ["plc.poly"]
    assert plan["outputs"] == ["plc.ele", "plc.node", "plc.face"]


@pytest.mark.skipif(shutil.which("tetgen") is None, reason="tetgen not on PATH")
def test_msh_then_run_without_meshing(tmp_path):
    work = tmp_path / "pro"
    _copy_ci_models(work)
    made = pro("msh", *TETGEN_MODEL, "-l", "0", cwd=work)
    assert made.returncode == 0, made.stderr[-2000:]
    assert "running tetgen" in made.stdout
    assert (work / "meshNoVessel.mesh.ymf").exists()

    again = pro("msh", *TETGEN_MODEL, "-l", "0", cwd=work)
    assert again.returncode == 0 and "up to date" in again.stdout

    # the run reads the mesh pro msh made, and refuses to remesh it
    ran = pro("run", *TETGEN_MODEL, "--no-mesh", cwd=work)
    assert ran.returncode == 0, ran.stderr[-2000:]

    # a context option that changes tetgen's options makes the mesh stale
    stale = pro("run", *TETGEN_MODEL, "--no-mesh", "-C", "Refinement=1", cwd=work)
    assert stale.returncode == 2
    assert "pro msh poisson_3d_tetgen_p.py poisson_3d_tetgen_c0p1_n.py " \
           "-C Refinement=1" in stale.stderr

    if shutil.which("parun") is None:
        return
    reference = tmp_path / "parun"
    _copy_ci_models(reference)
    ran = subprocess.run(["parun"] + TETGEN_MODEL, cwd=reference,
                         capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr[-2000:]
    import numpy as np
    expected = _datasets(reference / "poisson_tetgen_0.h5")
    actual = _datasets(work / "poisson_tetgen_0.h5")
    assert sorted(actual) == sorted(expected)
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name


@pytest.mark.skipif(shutil.which("tetgen") is None, reason="tetgen not on PATH")
def test_edited_mesh_file_is_stale(tmp_path):
    from proteus.cli import mesh
    work = tmp_path / "pro"
    _copy_ci_models(work)
    assert pro("msh", *TETGEN_MODEL, "-l", "0", cwd=work).returncode == 0
    plan = {"generator": "tetgen", "prefix": str(work / "meshNoVessel")}
    import ymf.archive
    record = ymf.archive.read_document(str(work / "meshNoVessel.mesh.ymf"))["mesh"]
    plan["command"] = record["command"]
    plan["inputs"] = [str(work / "meshNoVessel.poly")]
    assert mesh.state(plan) == ("current", "")
    with open(work / "meshNoVessel.node", "a") as f:
        f.write("# edited\n")
    assert mesh.state(plan) == ("stale", "meshNoVessel.node changed since it was generated")


@pytest.mark.skipif(shutil.which("tetgen") is None, reason="tetgen not on PATH")
def test_status(tmp_path):
    work = tmp_path / "pro"
    _copy_ci_models(work)
    absent = pro("status", *TETGEN_MODEL, cwd=work)
    assert absent.returncode == 0, absent.stderr[-2000:]
    assert "mesh:      absent (no meshNoVessel.mesh.ymf)" in absent.stdout
    assert "pro run meshes on rank 0 first" in absent.stdout
    assert not (work / "meshNoVessel.mesh.ymf").exists()
    assert not (work / "meshNoVessel.ele").exists()

    assert pro("msh", *TETGEN_MODEL, "-l", "0", cwd=work).returncode == 0
    current = pro("status", *TETGEN_MODEL, cwd=work)
    assert "mesh:      current" in current.stdout
    assert "next:      pro run reads the mesh files; no meshing" in current.stdout

    stale = pro("status", *TETGEN_MODEL, "-C", "Refinement=1", cwd=work)
    assert "mesh:      stale (the generator's inputs or command changed)" in stale.stdout


def test_status_of_a_mesh_built_in_memory(tmp_path):
    shutil.copytree(os.path.join(HERE, "elliptic_redist", "RDLS"), tmp_path / "rdls")
    out = pro("status", "vortex2D_so.py", cwd=tmp_path / "rdls")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "domain:    RectangularDomain" in out.stdout
    assert "nothing to generate" in out.stdout


POISSON_SPEC = """\
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
"""


def test_spec_goes_to_the_ymf_runner(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    (tmp_path / "poisson.ymf").write_text(POISSON_SPEC, encoding="utf-8")
    out = pro("run", "poisson.ymf", "--cells", "4", "--emit-pn",
              "--outdir", "out", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    archive = (tmp_path / "out" / "poisson.archive.ymf").read_text()
    assert "set_by: pro run --cells 4" in archive
    emitted = sorted(p.name for p in (tmp_path / "out").glob("*_so.py"))
    assert len(emitted) == 1

    # the lean-tier route: the emitted files run with pro run, no spec needed
    work = tmp_path / "compute"
    work.mkdir()
    stem = emitted[0][:-len("_so.py")]
    for suffix in ("_p.py", "_n.py", "_so.py"):
        shutil.copy(tmp_path / "out" / (stem + suffix), work)
    ran = pro("run", stem + "_so.py", cwd=work)
    assert ran.returncode == 0, ran.stderr[-2000:]


def test_spec_without_the_front_end_explains_the_route(monkeypatch, capsys):
    import proteus.cli as cli
    monkeypatch.setattr(cli, "missing", lambda *names: ["sympy"])
    assert cli.main(["run", "spec.ymf", "--cells", "4"]) == 2
    err = capsys.readouterr().err
    assert "lacks sympy" in err
    assert "pro run SPEC.ymf --emit-pn" in err


LADR = ["ladr_2d_p.py", "ladr_2d_n.py"]


def _copy_ladr(directory):
    directory.mkdir()
    for f in ("adr.py",) + tuple(LADR):
        shutil.copy(os.path.join(HERE, "ci", f), directory)


@pytest.mark.skipif(shutil.which("parun") is None, reason="parun not on PATH")
def test_shell_batch_run_matches_parun(tmp_path):
    import numpy as np
    _copy_ladr(tmp_path / "shell")
    (tmp_path / "shell" / "batch.py").write_text(
        "solution = run()\n"
        "print('ran', type(solution).__name__, status, [p.name for p in pList])\n")
    out = pro("shell", *LADR, "--batch", "batch.py", cwd=tmp_path / "shell")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ran NS_base 0 ['ladr_2d']" in out.stdout

    _copy_ladr(tmp_path / "parun")
    ran = subprocess.run(["parun"] + LADR, cwd=tmp_path / "parun",
                         capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr[-2000:]
    expected = _datasets(tmp_path / "parun" / "ladr_2d.h5")
    actual = _datasets(tmp_path / "shell" / "ladr_2d.h5")
    assert sorted(actual) == sorted(expected)
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name


def test_shell_prompt_reads_python(tmp_path):
    _copy_ladr(tmp_path / "shell")
    out = subprocess.run(
        [sys.executable, "-m", "proteus.cli", "shell", *LADR, "-l", "0"],
        cwd=tmp_path / "shell", input="print('model is', so.name, opts.hotStart)\n",
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "model is ladr_2d False" in out.stdout


def test_shell_without_a_model(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "proteus.cli", "shell", "-l", "0"], cwd=tmp_path,
        input="run()\n", capture_output=True, text=True)
    assert "no model loaded: load('model_so.py') first" in out.stderr


@pytest.mark.skipif(shutil.which("triangle") is None, reason="triangle not on PATH")
def test_emitted_region_files_mesh_with_pro_msh(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    box = '    domain: "Ω = [0, 1] × [0, 1]"\n'
    assert box in POISSON_SPEC
    spec = POISSON_SPEC.replace(box, '''    domain:
      kind: piecewise_linear
      dimension: 2
      vertices: [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
      segments: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
''').replace("      solver: {type: linear}\n",
             "      solver: {type: linear}\n      mesh: {h: 0.25}\n")
    (tmp_path / "lshape.ymf").write_text(spec, encoding="utf-8")
    out = pro("run", "lshape.ymf", "--emit-pn", "--outdir", "out", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    (so,) = (tmp_path / "out").glob("*_so.py")
    stem = so.name[:-len("_so.py")]

    # on the compute side: only the emitted files
    work = tmp_path / "compute"
    work.mkdir()
    for suffix in ("_p.py", "_n.py", "_so.py"):
        shutil.copy(tmp_path / "out" / (stem + suffix), work)
    status = pro("status", so.name, cwd=work)
    assert "generator: triangle" in status.stdout
    assert "mesh:      absent" in status.stdout
    made = pro("msh", so.name, "-l", "0", cwd=work)
    assert made.returncode == 0 and "running triangle -VApq30Dena" in made.stdout
    ran = pro("run", so.name, "--no-mesh", cwd=work)
    assert ran.returncode == 0, ran.stderr[-2000:]


LSHAPE = """\
    domain:
      kind: piecewise_linear
      dimension: 2
      vertices: [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
      segments: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
"""


def _lshape(tmp_path):
    """The Poisson spec on an L-shaped region, by P1 and by P2 on one h."""
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    if shutil.which("triangle") is None:
        pytest.skip("triangle not on PATH")
    box = '    domain: "Ω = [0, 1] × [0, 1]"\n'
    spec = POISSON_SPEC.replace(box, LSHAPE).replace(
        "      solver: {type: linear}\n", "      solver: {type: linear}\n      mesh: {h: 0.5}\n")
    spec += ("    - name: P2\n      provenance: human_specified\n      from_weak_form: galerkin\n"
             "      finite_element: {fields: {family: CG, order: 2}}\n"
             "      solver: {type: linear}\n      mesh: {h: 0.5}\n")
    (tmp_path / "lshape.ymf").write_text(spec, encoding="utf-8")


def test_a_spec_run_meshes_a_region_once_and_reuses_it(tmp_path):
    _lshape(tmp_path)
    out = pro("run", "lshape.ymf", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.count("running triangle") == 1        # P1 and P2: one mesh
    assert len(list(tmp_path.glob("lshape_mesh_*.mesh.ymf"))) == 1
    assert "2 outputs, 2 new" in out.stdout
    again = pro("run", "lshape.ymf", "--check", cwd=tmp_path)
    assert again.returncode == 0, again.stderr[-2000:]
    assert "running triangle" not in again.stdout and ": up to date" in again.stdout
    assert again.stdout.count("reproduced, bitwise") == 2


def test_no_mesh_refuses_a_stale_mesh_and_pro_msh_makes_it(tmp_path):
    _lshape(tmp_path)
    assert pro("run", "lshape.ymf", cwd=tmp_path).returncode == 0
    (node,) = tmp_path.glob("lshape_mesh_*.node")
    node.write_text(node.read_text() + "# edited\n")
    out = pro("run", "lshape.ymf", "--check", "--no-mesh", cwd=tmp_path)
    assert out.returncode == 2
    assert "a mesh is missing or stale, and --no-mesh was given" in out.stderr
    assert "changed since it was generated" in out.stderr
    assert "make it with: pro msh lshape.ymf --check" in out.stderr
    assert "reproduced" not in out.stdout                    # nothing was solved
    made = pro("mesh", "lshape.ymf", "--check", cwd=tmp_path)   # the long name
    assert made.returncode == 0 and made.stdout.count("running triangle") == 1
    assert "reproduced" not in made.stdout                    # pro msh solves nothing
    out = pro("run", "lshape.ymf", "--check", "--no-mesh", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.count("reproduced, bitwise") == 2


def test_pro_msh_of_a_spec_says_when_there_is_nothing_to_mesh(tmp_path):
    _spec(tmp_path)                                           # a box: meshed in memory
    out = pro("msh", "poisson.ymf", "--cells", "4", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "nothing to mesh: every output to compute is on a box" in out.stdout
    assert not (tmp_path / "poisson.archive.ymf").exists()
    _lshape(tmp_path)
    assert pro("run", "lshape.ymf", cwd=tmp_path).returncode == 0
    out = pro("msh", "lshape.ymf", cwd=tmp_path)
    assert "nothing to mesh: every output is in the archive (--check" in out.stdout


# --- pro spec: the linter --------------------------------------------------------


def _spec(tmp_path, text=POISSON_SPEC, name="poisson.ymf"):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _report(*args, cwd):
    import json
    out = pro("spec", "--json", *args, cwd=cwd)
    report = json.loads(out.stdout)
    assert report["exit_status"] == out.returncode, out.stderr[-2000:]
    return report


def test_spec_reports_a_well_posed_spec_and_writes_nothing(tmp_path):
    _spec(tmp_path)
    out = pro("spec", "poisson.ymf", cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "state: well-posed" in out.stdout
    assert "P1: runs, given a mesh (--cells or --h)" in out.stdout
    assert sorted(os.listdir(tmp_path)) == ["poisson.ymf"]


def test_spec_json_is_the_same_report_as_data(tmp_path):
    _spec(tmp_path, POISSON_SPEC.replace(
        "      solver: {type: linear}\n",
        "      mesh: {cells: [4, 8]}\n      solver: {type: linear}\n"))
    report = _report("poisson.ymf", cwd=tmp_path)
    assert report["state"] == "discretized" and report["name"] == "Poisson"
    assert report["adr"] == {"unknowns": {"u": ["u"]}, "dimension": 2,
                             "time_dependent": False, "up_to_constant": [],
                             "periodic": []}
    assert report["discretizations"] == {"P1": "runs: 2 outputs"}
    assert not [i for i in report["issues"] if i["severity"] == "error"]


def test_spec_names_the_line_of_a_schema_error(tmp_path):
    _spec(tmp_path, POISSON_SPEC.replace("type: dirichlet", "type: dirichelt"))
    report = _report("poisson.ymf", cwd=tmp_path)
    (issue,) = report["issues"]
    assert issue["check"] == "schema" and issue["severity"] == "error"
    assert issue["location"].endswith("poisson.ymf:11"), issue
    assert report["state"] is None and report["exit_status"] == 1


def test_spec_reports_notation_it_cannot_read(tmp_path):
    _spec(tmp_path, POISSON_SPEC.replace("-Δu = 2π² sin(πx) sin(πy)",
                                         "-Δu = 2π² sin πx sin(πy)"))
    report = _report("poisson.ymf", cwd=tmp_path)
    (issue,) = [i for i in report["issues"] if i["check"] == "adr"]
    assert issue["severity"] == "error" and "sin(x)" in issue["message"]
    assert report["exit_status"] == 1


def test_spec_says_which_branches_pro_run_would_skip(tmp_path):
    _spec(tmp_path, POISSON_SPEC.replace(
        "      solver: {type: linear}\n",
        "      solver: {type: linear, linear_solver: umfpack}\n"
        "    - {name: P1_dg, provenance: human_specified, from_weak_form: galerkin,\n"
        "       finite_element: {fields: {family: RT, order: 1}}, solver: {type: linear}}\n"))
    report = _report("poisson.ymf", cwd=tmp_path)
    assert report["discretizations"]["P1"] == "skipped"
    found = {i["location"]: i["message"] for i in report["issues"]
             if i["check"] == "discretization"}
    assert "has no Proteus solver" in found["solution_paths.discretizations[P1]"]
    assert report["exit_status"] == 0          # a skipped branch warns; it is not an error


def test_spec_refuses_a_state_the_content_does_not_reach(tmp_path):
    _spec(tmp_path, "ymf_state: discretized\n" + POISSON_SPEC)
    report = _report("poisson.ymf", cwd=tmp_path)
    (issue,) = [i for i in report["issues"] if i["check"] == "state"]
    assert "declares ymf_state: discretized, but it is only well-posed" in issue["message"]
    assert report["exit_status"] == 1


def test_spec_of_an_abstract_model_passes_only_when_asked(tmp_path):
    _spec(tmp_path, """\
ymf_state: abstract
model:
  strong_form:
    provenance: human_specified
    unknowns: [u]
    equation_formulation: "Poisson"
    equations: ["-Δu = f  in Ω"]
""", name="model.ymf")
    out = pro("spec", "model.ymf", cwd=tmp_path)
    assert out.returncode == 1 and "a model to be extended" in out.stdout
    assert pro("spec", "model.ymf", "--require", "abstract", cwd=tmp_path).returncode == 0


def test_spec_reports_a_generated_file_that_was_edited(tmp_path):
    """ymf.generated (ymf >= cekees/ymf#14): a tool-written file someone edited."""
    generated = pytest.importorskip("ymf.generated")
    import copy
    from ymf.compose import load_composed
    _spec(tmp_path)
    parent = load_composed(tmp_path / "poisson.ymf")
    weak = dict(copy.deepcopy(parent["model"]["weak_forms"][0]),
                label="galerkin_2", provenance="proteus_derived")
    generated.write(tmp_path / "poisson.weak.ymf", "poisson.ymf", "pro weak (test)",
                    "model_without_weak_forms", parent, {"model": {"weak_forms": [weak]}})
    assert not [i for i in _report("poisson.weak.ymf", cwd=tmp_path)["issues"]
                if i["check"] == "generated"]
    path = tmp_path / "poisson.weak.ymf"
    path.write_text(path.read_text(encoding="utf-8").replace("proteus_derived", "human_edited"),
                    encoding="utf-8")
    (issue,) = [i for i in _report("poisson.weak.ymf", cwd=tmp_path)["issues"]
                if i["check"] == "generated"]
    assert issue["severity"] == "warning" and "edited since pro weak (test)" in issue["message"]


# --- pro weak: the weak forms Proteus would solve -------------------------------

CHANNEL_SPEC = """\
model:
  name: "Channel"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [{name: v, rank: 1}, p]
    equation_formulation: "Navier-Stokes"
    equations:
      - "ρ ∂v/∂t + ∇·(ρ v⊗v) − μΔv + ∇p = 0  in Ω"
      - "∇·v = 0  in Ω"
    domain: "Ω = [0, 4] × [0, 1]"
    boundary_regions: [{name: walls, where: "∂Ω"}]
    boundary_conditions:
      - {region: walls, variable: v, type: dirichlet, formula: "(4 y (1 - y), 0)"}
    coefficients:
      ρ: {value: 1.0}
      μ: {value: 1.0}
solution_paths:
  analytical: []
  discretizations:
    - name: P2P1
      provenance: human_specified
      finite_element: {velocity: {family: CG, order: 2}, pressure: {family: CG, order: 1}}
      solver: {type: nonlinear}
"""


def _weak(*args, cwd):
    import json
    out = pro("weak", "--json", *args, cwd=cwd)
    report = json.loads(out.stdout)
    assert report["exit_status"] == out.returncode, out.stderr[-2000:]
    return report


def test_weak_list_says_what_applies_and_writes_nothing(tmp_path):
    _spec(tmp_path)
    report = _weak("poisson.ymf", "--list", cwd=tmp_path)
    applies = {f["label"]: f["applies"] for f in report["formulations"]}
    assert applies == {"proteus_galerkin": True, "proteus_supg_pspg": False}
    why = [f["why"] for f in report["formulations"] if not f["applies"]][0]
    assert "velocity-pressure" in why
    assert report["written"] is None and sorted(os.listdir(tmp_path)) == ["poisson.ymf"]


def test_weak_writes_a_generated_file_that_composes(tmp_path):
    from ymf import generated
    from ymf.compose import load_composed
    _spec(tmp_path)
    report = _weak("poisson.ymf", cwd=tmp_path)
    assert report["exit_status"] == 0 and report["written"].endswith("poisson.weak.ymf")
    path = tmp_path / "poisson.weak.ymf"
    assert generated.check(path).current
    doc = load_composed(path)
    (weak,) = [w for w in doc["model"]["weak_forms"] if w["label"] == "proteus_galerkin"]
    assert weak["bilinear"] == "u: ∫Ω ∇u·∇w dx"
    assert weak["linear"] == "u: ∫Ω 2 π² sin(π x) sin(π y) w dx"
    assert weak["provenance"] == "proteus_derived"
    assert weak["solution_spaces"] == {"trial": "H1(Ω)", "test": "H1_0(Ω)"}
    assert pro("spec", "poisson.weak.ymf", cwd=tmp_path).returncode == 0


def test_weak_writes_both_formulations_of_a_velocity_pressure_system(tmp_path):
    from ymf.compose import load_composed
    _spec(tmp_path, CHANNEL_SPEC.replace(CHANNEL_SPEC[CHANNEL_SPEC.index("    - name: P2P1"):],
                                         "").replace("  discretizations:\n",
                                                     "  discretizations: []\n"),
          name="channel.ymf")
    report = _weak("channel.ymf", cwd=tmp_path)
    assert report["exit_status"] == 0, report["error"]
    forms = {w["label"]: w for w in
             load_composed(tmp_path / "channel.weak.ymf")["model"]["weak_forms"]}
    assert set(forms) == {"proteus_galerkin", "proteus_supg_pspg"}
    assert forms["proteus_galerkin"]["solution_spaces"]["trial"] == "H1(Ω)² × L2(Ω)"
    assert forms["proteus_supg_pspg"]["solution_spaces"]["trial"] == "H1(Ω)² × H1(Ω)"
    assert forms["proteus_galerkin"]["bilinear"].splitlines()[2] == "p: - ∫Ω (v_0, v_1)·∇q dx"
    assert "Σ_K τ_K" in forms["proteus_supg_pspg"]["bilinear"]
    assert forms["proteus_supg_pspg"]["stabilization_method"] == "SUPG/PSPG"


def test_weak_will_not_make_a_discretization_ambiguous(tmp_path):
    """The spec cannot name a weak form only the generated file has."""
    from ymf.compose import load_composed
    _spec(tmp_path, CHANNEL_SPEC, name="channel.ymf")
    report = _weak("channel.ymf", cwd=tmp_path)
    assert report["exit_status"] == 1 and "P2P1 names no weak form" in report["error"]
    assert "--only" in report["error"] and not (tmp_path / "channel.weak.ymf").exists()
    report = _weak("channel.ymf", "--only", "proteus_galerkin", cwd=tmp_path)
    assert report["exit_status"] == 0, report["error"]
    doc = load_composed(tmp_path / "channel.weak.ymf")
    (entry,) = doc["solution_paths"]["discretizations"]
    assert entry["from_weak_form"] == "proteus_galerkin"      # the only one: defaulted


def test_weak_regenerates_its_own_file_but_not_an_edited_one(tmp_path):
    _spec(tmp_path)
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0      # unchanged: regenerated
    path = tmp_path / "poisson.weak.ymf"
    path.write_text(path.read_text(encoding="utf-8").replace("proteus_derived", "human_edited"),
                    encoding="utf-8")
    report = _weak("poisson.ymf", cwd=tmp_path)
    assert report["exit_status"] == 1 and "was edited since" in report["error"]
    assert "human_edited" in path.read_text(encoding="utf-8")         # left alone
    assert _weak("poisson.ymf", "--force", cwd=tmp_path)["exit_status"] == 0
    assert "human_edited" not in path.read_text(encoding="utf-8")


def test_weak_will_not_overwrite_a_file_no_tool_wrote(tmp_path):
    _spec(tmp_path)
    (tmp_path / "poisson.weak.ymf").write_text("model: {name: mine}\n", encoding="utf-8")
    report = _weak("poisson.ymf", cwd=tmp_path)
    assert report["exit_status"] == 1 and "not written by a tool" in report["error"]


def test_weak_only_writes_what_applies(tmp_path):
    _spec(tmp_path)
    report = _weak("poisson.ymf", "--only", "proteus_supg_pspg", cwd=tmp_path)
    assert report["exit_status"] == 1 and "does not apply" in report["error"]
    report = _weak("poisson.ymf", "--only", "nonesuch", cwd=tmp_path)
    assert report["exit_status"] == 1 and "no formulation nonesuch" in report["error"]


def test_weak_of_an_abstract_model_is_refused(tmp_path):
    _spec(tmp_path, "ymf_state: abstract\nmodel:\n  name: \"m\"\n", name="model.ymf")
    report = _weak("model.ymf", cwd=tmp_path)
    assert report["exit_status"] == 1 and "is abstract" in report["error"]
def test_json_output_survives_what_mpi_prints_to_stdout():
    """Open MPI's UCX layer writes an error line to stdout from C when it
    starts on a machine whose network it cannot use (the locked py3.13 CI
    leg), and that broke --json for any program reading the output."""
    code = ("import os, sys\n"
            "from proteus.cli import json_stdout\n"
            "out = json_stdout()\n"
            "os.write(1, b'[UCX ERROR] from C, at start-up\\n')\n"
            "print('from Python, after')\n"
            "print('{\"ok\": true}', file=out, flush=True)\n"
            "os.write(1, b'[UCX] at exit\\n')\n")
    ran = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout) == {"ok": True}
    assert "UCX ERROR" in ran.stderr and "from Python, after" in ran.stderr
    assert "at exit" in ran.stderr


# --- pro dis --------------------------------------------------------------------------


def _dis(*args, cwd):
    import json
    out = pro("dis", "--json", *args, cwd=cwd)
    report = json.loads(out.stdout)
    assert report["exit_status"] == out.returncode, out.stderr[-2000:]
    return report


def _without_discretizations(text):
    return text[:text.index("  discretizations:")] + "  discretizations: []\n"


def _without_weak_forms(text):
    return text[:text.index("  weak_forms:")] + text[text.index("solution_paths:"):]


def test_dis_needs_the_weak_forms_and_says_how_to_write_them(tmp_path):
    _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    report = _dis("poisson.ymf", cwd=tmp_path)
    assert report["exit_status"] == 1 and "run: pro wk poisson.ymf" in report["error"]
    assert sorted(os.listdir(tmp_path)) == ["poisson.ymf"]


def test_dis_writes_a_discretized_file_that_pro_run_runs(tmp_path):
    from ymf import generated
    from ymf.compose import load_composed
    _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    listed = _dis("poisson.weak.ymf", "--list", cwd=tmp_path)
    assert listed["exit_status"] == 0 and listed["written"] is None
    assert not (tmp_path / "poisson.discrete.ymf").exists()
    report = _dis("poisson.weak.ymf", cwd=tmp_path)
    assert report["exit_status"] == 0, report["error"]
    path = tmp_path / "poisson.discrete.ymf"
    assert report["written"] == str(path) and generated.check(path).current
    doc = load_composed(path)                          # checks ymf_state: discretized
    assert doc["ymf_state"] == "discretized"
    (entry,) = doc["solution_paths"]["discretizations"]
    assert entry["name"] == "galerkin_p1" and entry["from_weak_form"] == "proteus_galerkin"
    assert entry["provenance"] == "proteus_derived"
    assert entry["finite_element"] == {"fields": {"family": "CG", "order": 1}}
    assert entry["mesh"] == {"cells": 16}
    assert entry["solver"]["type"] == "linear"
    assert entry["solver"]["linear_solver"] in ("mumps", "superlu")
    spec = _report("poisson.discrete.ymf", cwd=tmp_path)
    assert spec["state"] == "discretized", spec
    out = pro("run", "poisson.discrete.ymf", cwd=tmp_path)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert (tmp_path / "poisson.discrete.archive.ymf").exists()


def test_dis_of_a_velocity_pressure_system_uses_each_formulations_elements(tmp_path):
    from ymf.compose import load_composed
    _spec(tmp_path, _without_discretizations(CHANNEL_SPEC), name="channel.ymf")
    assert _weak("channel.ymf", cwd=tmp_path)["exit_status"] == 0
    report = _dis("channel.weak.ymf", cwd=tmp_path)
    assert report["exit_status"] == 0, report["error"]
    entries = {d["name"]: d for d in
               load_composed(tmp_path / "channel.discrete.ymf")["solution_paths"]["discretizations"]}
    assert set(entries) == {"galerkin_taylor_hood", "supg_pspg_p1p1"}
    th, eq = entries["galerkin_taylor_hood"], entries["supg_pspg_p1p1"]
    assert th["finite_element"]["velocity"]["order"] == 2
    assert th["finite_element"]["pressure"]["order"] == 1
    assert eq["finite_element"]["velocity"]["order"] == eq["finite_element"]["pressure"]["order"] == 1
    assert eq["from_weak_form"] == "proteus_supg_pspg"
    assert th["solver"]["type"] == "nonlinear" and th["solver"]["nonlinear_solver"] == "newton"


def test_dis_gives_an_existing_discretization_only_what_it_lacks(tmp_path):
    from ymf.compose import load_composed
    _spec(tmp_path)                                    # P1 has no mesh
    report = _dis("poisson.ymf", "--cells", "4", "8", cwd=tmp_path)
    assert report["exit_status"] == 0, report["error"]
    assert report["discretizations"] == [
        {"name": "P1", "new": False, "from_weak_form": "galerkin", "adds": ["mesh"]}]
    (entry,) = load_composed(tmp_path / "poisson.discrete.ymf")["solution_paths"]["discretizations"]
    assert entry["mesh"] == {"cells": [4, 8]}         # a study
    assert entry["provenance"] == "human_specified"    # the rest is the person's
    assert entry["solver"] == {"type": "linear"}


def test_dis_meshes_a_box_by_cells_not_h(tmp_path):
    _spec(tmp_path)
    report = _dis("poisson.ymf", "--h", "0.1", cwd=tmp_path)
    assert report["exit_status"] == 1 and "give --cells, not --h" in report["error"]


def test_dis_picks_a_scalable_solver_for_a_large_problem(tmp_path):
    from ymf.compose import load_composed
    _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    report = _dis("poisson.weak.ymf", "--cells", "400", cwd=tmp_path)    # 401² nodes
    assert report["exit_status"] == 0, report["error"]
    (entry,) = load_composed(tmp_path / "poisson.discrete.ymf")["solution_paths"]["discretizations"]
    assert entry["solver"]["linear_solver"] == "petsc"
    assert "160801 unknowns" in entry["discretization_details"]


HEAT_SPEC = """\
model:
  name: "Heat"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [T]
    equation_formulation: "heat equation"
    equations: ["∂T/∂t = ∇·(κ∇T)  in Ω"]
    domain: "Ω = [0, 1] × [0, 1],  t ∈ (0, 2]"
    boundary_regions: [{name: walls, where: "∂Ω"}]
    initial_conditions: [{field: T, type: function, formula: "sin(πx) sin(πy)"}]
    boundary_conditions: [{region: walls, variable: T, type: dirichlet, value: 0.0}]
    coefficients:
      κ: {value: 1.0e-2}
solution_paths:
  analytical: []
  discretizations: []
"""


def test_dis_steps_a_time_dependent_problem_with_backward_euler(tmp_path):
    from ymf.compose import load_composed
    _spec(tmp_path, HEAT_SPEC, name="heat.ymf")
    assert _weak("heat.ymf", cwd=tmp_path)["exit_status"] == 0
    assert _dis("heat.weak.ymf", cwd=tmp_path)["exit_status"] == 0
    (entry,) = load_composed(tmp_path / "heat.discrete.ymf")["solution_paths"]["discretizations"]
    assert entry["discretization"]["time_integration"] == "backward_euler"
    assert float(entry["discretization"]["dt"]) == pytest.approx(0.02)   # 100 steps


def test_dis_regenerates_its_own_file_but_not_an_edited_one(tmp_path):
    _spec(tmp_path)
    assert _dis("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    assert _dis("poisson.ymf", cwd=tmp_path)["exit_status"] == 0      # its own: regenerated
    path = tmp_path / "poisson.discrete.ymf"
    path.write_text(path.read_text(encoding="utf-8").replace("cells: 16", "cells: 32"),
                    encoding="utf-8")
    report = _dis("poisson.ymf", cwd=tmp_path)
    assert report["exit_status"] == 1 and "was edited since" in report["error"]
    assert _dis("poisson.ymf", "--force", cwd=tmp_path)["exit_status"] == 0


def test_dis_says_when_there_is_nothing_to_write(tmp_path):
    _spec(tmp_path, POISSON_SPEC.replace(
        "      solver: {type: linear}", "      mesh: {cells: 4}\n      solver: {type: linear}"))
    report = _dis("poisson.ymf", cwd=tmp_path)
    assert report["exit_status"] == 1 and "nothing to write" in report["error"]


def test_default_unknowns_are_counted_on_the_finest_mesh_of_a_study():
    from proteus.cli.defaults import estimated_dofs
    problem = {"domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
               "unknowns": {"u": ["u"]}}
    p1 = {"fields": {"family": "CG", "order": 1}}
    p2 = {"fields": {"family": "CG", "order": 2}}
    assert estimated_dofs(problem, p1, {"cells": [4, 8]}) == 81
    assert estimated_dofs(problem, p2, {"cells": 4, "levels": 2}) == 17 ** 2
    region = dict(problem, domain=dict(problem["domain"], region="pslg"))
    assert estimated_dofs(region, p1, {"h": [0.5, 0.25]}) == 25


# --- pro status of a spec: the chain --------------------------------------------------


def _status(path, cwd):
    import json
    out = pro("status", "--json", path, cwd=cwd)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    report = json.loads(out.stdout)
    (step,) = report["next"].values()
    return report, step


def test_status_follows_a_spec_through_its_steps(tmp_path):
    _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    assert _status("poisson.ymf", tmp_path)[1] == "pro wk poisson.ymf"
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    assert _status("poisson.ymf", tmp_path)[1] == "pro dis poisson.weak.ymf"
    assert _dis("poisson.weak.ymf", cwd=tmp_path)["exit_status"] == 0
    report, step = _status("poisson.ymf", tmp_path)
    assert step == "pro run poisson.discrete.ymf"
    disc = report["files"][str(tmp_path / "poisson.discrete.ymf")]
    assert disc["state"] == "discretized" and disc["generated"]["current"]
    assert disc["weak_forms"] == {"proteus_derived": 1} and disc["discretizations"] == 1
    assert pro("run", "poisson.discrete.ymf", cwd=tmp_path).returncode == 0
    report, step = _status("poisson.discrete.ymf", tmp_path)
    assert step.startswith("nothing to do")
    assert [os.path.basename(e["file"]) for e in report["chain"]] == [
        "poisson.ymf", "poisson.weak.ymf", "poisson.discrete.ymf"]
    assert report["chain"][-1]["state"] == "closed"
    assert report["chain"][-1]["archive"]["present"] == 1
    text = pro("status", "poisson.ymf", cwd=tmp_path).stdout
    assert "└ poisson.weak.ymf" in text and "pro dis, current" in text


def test_a_stale_step_is_named_and_pro_run_refuses_it(tmp_path):
    spec = _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    assert _dis("poisson.weak.ymf", cwd=tmp_path)["exit_status"] == 0
    spec.write_text(spec.read_text(encoding="utf-8").replace("2π²", "π²"), encoding="utf-8")
    report, step = _status("poisson.discrete.ymf", tmp_path)
    assert step == "poisson.weak.ymf is stale; regenerate it: pro wk poisson.ymf"
    weak = report["files"][str(tmp_path / "poisson.weak.ymf")]["generated"]
    assert weak["stale"] and not weak["edited"]
    out = pro("run", "poisson.discrete.ymf", cwd=tmp_path)
    assert out.returncode == 2 and "is stale" in out.stderr
    assert "regenerate it: pro dis poisson.weak.ymf" in out.stderr
    assert not (tmp_path / "poisson.discrete.archive.ymf").exists()
    assert pro("run", "poisson.discrete.ymf", "--stale-ok", cwd=tmp_path).returncode == 0


def test_a_hand_edited_step_is_reported_and_still_runs(tmp_path):
    _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    assert _dis("poisson.weak.ymf", cwd=tmp_path)["exit_status"] == 0
    path = tmp_path / "poisson.discrete.ymf"
    path.write_text(path.read_text(encoding="utf-8").replace("cells: 16", "cells: 8"),
                    encoding="utf-8")
    report, step = _status("poisson.discrete.ymf", tmp_path)
    assert step == "pro run poisson.discrete.ymf"           # an edit is the editor's choice
    assert report["chain"][-1]["generated"]["edited"]
    assert "hand-edited" in pro("status", "poisson.discrete.ymf", cwd=tmp_path).stdout
    assert pro("run", "poisson.discrete.ymf", cwd=tmp_path).returncode == 0


def test_regenerate_command_keeps_a_chosen_output_name(tmp_path, monkeypatch):
    from proteus.cli.chain import file_info
    _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    assert _weak("poisson.ymf", "-o", "mine.ymf", cwd=tmp_path)["exit_status"] == 0
    monkeypatch.chdir(tmp_path)
    assert file_info("mine.ymf")["generated"]["regenerate"] == "pro wk poisson.ymf -o mine.ymf"


def test_dis_names_the_method_of_what_it_writes(tmp_path):
    from ymf.compose import load_composed
    _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    assert _dis("poisson.weak.ymf", cwd=tmp_path)["exit_status"] == 0
    (entry,) = load_composed(tmp_path / "poisson.discrete.ymf")["solution_paths"]["discretizations"]
    assert entry["method"] == "mesh"


def test_dis_leaves_a_discretization_of_another_method_alone(tmp_path):
    neural = "      finite_element: {fields: {family: CG, order: 1}}\n"
    spec = POISSON_SPEC.replace(
        "    - name: P1\n", "    - name: NN\n      method: neural\n"
        "      neural: {width: 32, depth: 4, points: 1000}\n").replace(neural, "")
    assert "method: neural" in spec and "finite_element: {fields" not in spec
    _spec(tmp_path, spec)
    report = _dis("poisson.ymf", cwd=tmp_path)
    assert report["skipped"] == [{"discretization": "NN",
                                  "why": "method neural: pro dis fills mesh discretizations only"}]
    assert report["exit_status"] == 1 and "nothing to write" in report["error"]
    assert not (tmp_path / "poisson.discrete.ymf").exists()
    text = pro("dis", "poisson.ymf", "--list", cwd=tmp_path).stdout
    assert "NN" in text and "skipped: method neural" in text


@pytest.mark.parametrize("command", ["weak", "dis"])
def test_a_spec_that_fails_the_schema_is_an_error_not_a_traceback(tmp_path, command):
    import json
    _spec(tmp_path, POISSON_SPEC.replace("      solver: {type: linear}\n", ""))
    out = pro(command, "--json", "poisson.ymf", cwd=tmp_path)
    report = json.loads(out.stdout)
    assert out.returncode == 1 and "Traceback" not in out.stderr
    assert report["error"].startswith("poisson.ymf, line ")
    assert "required key(s) 'solver' not found" in report["error"]
    assert report["error"].endswith("(run pro sp poisson.ymf)")


# --- command names: a short name, and a whole-word long name ---------------------------


def _choices():
    import argparse
    from proteus.cli import build_parser
    (action,) = [a for a in build_parser()._actions
                 if isinstance(a, argparse._SubParsersAction)]
    return action.choices


@pytest.mark.parametrize("short", sorted(__import__("proteus.cli").cli.COMMANDS))
def test_a_command_is_the_same_under_both_its_names(short):
    from proteus.cli import COMMANDS
    module, long = COMMANDS[short]
    choices = _choices()
    assert choices[short] is choices[long]          # one parser: same options, help, handler
    assert choices[short].prog == "pro " + short


@pytest.mark.parametrize("short", sorted(__import__("proteus.cli").cli.COMMANDS))
def test_a_short_name_abbreviates_its_whole_word_long_name(short):
    from proteus.cli import COMMANDS
    long = COMMANDS[short][1]
    assert 2 <= len(short) <= 3 and short.isalpha() and short.islower()
    assert long.isalpha() and long.islower()        # a whole word: no hyphen, no digits
    rest = iter(long)
    assert long[0] == short[0] and all(c in rest for c in short), (short, long)


def test_no_two_commands_share_a_name():
    from proteus.cli import COMMANDS
    names = [n for short, (_, long) in COMMANDS.items() for n in {short, long}]
    assert len(names) == len(set(names))
    assert sorted(_choices()) == sorted(names)


def test_help_lists_each_command_with_its_long_name():
    out = pro("--help")
    assert out.returncode == 0
    for short, long in [("wk", "weak"), ("dis", "discrete"), ("st", "status"),
                        ("sh", "shell"), ("sp", "spec"), ("msh", "mesh")]:
        assert "%s (%s)" % (short, long) in out.stdout
    assert "run (" not in out.stdout                # run has one name


def test_an_option_is_never_matched_by_a_prefix(tmp_path):
    _spec(tmp_path)
    out = pro("dis", "poisson.ymf", "--cel", "4", cwd=tmp_path)
    assert out.returncode == 2 and "unrecognized arguments: --cel" in out.stderr
    out = pro("run", "poisson.ymf", "--discretiz", "P1", cwd=tmp_path)      # the YMF runner's
    assert out.returncode == 2 and "unrecognized arguments: --discretiz" in out.stderr


def test_a_record_written_under_a_long_name_regenerates_under_the_short_one(tmp_path,
                                                                            monkeypatch):
    from proteus.cli.chain import regenerate_command
    _spec(tmp_path, _without_discretizations(_without_weak_forms(POISSON_SPEC)))
    assert _weak("poisson.ymf", cwd=tmp_path)["exit_status"] == 0
    monkeypatch.chdir(tmp_path)
    for by in ("pro weak (proteus 2.0.0.dev0, ymf 0.2.1)", "pro wk (proteus 2.0.0.dev0)"):
        assert regenerate_command("poisson.weak.ymf", {"by": by}) == "pro wk poisson.ymf"
    assert regenerate_command("poisson.weak.ymf", {"by": "pro disc (old)"}) == \
        "pro dis poisson.ymf -o poisson.weak.ymf"


# --- pro run SPEC.ymf on several ranks --------------------------------------------------


def _mpiexec():
    """The launcher beside this interpreter (see test_archive_parallel_metadata:
    a mismatched one runs N single-rank jobs, and a test passes for the wrong
    reason; so these tests also check the ranks the archive records)."""
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _mpiexec()
needs_mpiexec = pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")


def pro_on(ranks, *args, cwd):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    return subprocess.run([MPIEXEC, "-n", str(ranks), sys.executable, "-m", "proteus.cli"]
                          + list(args), cwd=cwd, capture_output=True, text=True,
                          timeout=600, env=env)


def _outputs(path):
    from ymf import closure
    _, outputs, _ = closure.load(str(path), check_state=False)
    return outputs


@needs_mpiexec
def test_a_spec_runs_on_two_ranks_with_the_errors_of_one(tmp_path):
    for ranks in (1, 2):
        (tmp_path / str(ranks)).mkdir()
        _spec(tmp_path / str(ranks))
        out = pro_on(ranks, "run", "poisson.ymf", "--cells", "8", cwd=tmp_path / str(ranks))
        assert out.returncode == 0, out.stderr[-3000:]
        assert out.stdout.count("wrote ") == 1                # rank 0 reports, once
    (one,), (two,) = (_outputs(tmp_path / r / "poisson.archive.ymf").values()
                      for r in ("1", "2"))
    assert two["transformations"]["solve"]["by"]["ranks"] == 2
    a, b = one["verification"]["l2_errors"]["u"], two["verification"]["l2_errors"]["u"]
    assert abs(a - b) <= 1e-12 * a                            # summed over the ranks
    again = pro_on(2, "run", "poisson.ymf", "--cells", "8", "--check", cwd=tmp_path / "2")
    assert again.returncode == 0 and "reproduced, bitwise" in again.stdout
    other = pro_on(2, "run", "poisson.ymf", "--cells", "8", "--check", cwd=tmp_path / "1")
    assert other.returncode == 1
    assert "recorded on 1 rank, rerun on 2: a different partition" in other.stdout


@needs_mpiexec
def test_two_ranks_mesh_a_region_on_rank_0(tmp_path):
    _lshape(tmp_path)
    out = pro_on(2, "run", "lshape.ymf", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.count("running triangle") == 1
    outputs = _outputs(tmp_path / "lshape.archive.ymf")
    assert len(outputs) == 2
    assert {o["transformations"]["solve"]["by"]["ranks"] for o in outputs.values()} == {2}
    msh = pro_on(2, "msh", "lshape.ymf", "--check", cwd=tmp_path)
    assert msh.returncode == 2 and "meshing is serial" in msh.stderr


@needs_mpiexec
def test_a_whole_system_null_space_runs_on_two_ranks(tmp_path):
    spec = (POISSON_SPEC.replace("2π² sin(πx) sin(πy)", "2π² cos(πx) cos(πy)")
            .replace("type: dirichlet, value: 0.0", "type: neumann, value: 0.0")
            .replace("test: H1_0", "test: H1")
            .replace('"u = sin(πx) sin(πy)"', '"u = cos(πx) cos(πy)"'))
    _spec(tmp_path, spec)
    out = pro_on(2, "run", "poisson.ymf", "--cells", "8", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-3000:]
    (output,) = _outputs(tmp_path / "poisson.archive.ymf").values()
    assert output["verification"]["modulo_constants"] == ["u"]
    assert output["verification"]["l2_errors"]["u"] < 3e-2


@needs_mpiexec
def test_superlu_is_refused_on_two_ranks_before_computing(tmp_path):
    _spec(tmp_path, POISSON_SPEC.replace("solver: {type: linear}",
                                         "solver: {type: linear, linear_solver: superlu}"))
    out = pro_on(2, "run", "poisson.ymf", "--cells", "4", cwd=tmp_path)
    assert out.returncode == 2 and "linear_solver superlu is serial" in out.stderr
    assert not list(tmp_path.glob("*.h5"))                    # nothing was solved


POISEUILLE = """\
  analytical:
    - name: poiseuille
      provenance: human_specified
      method: closed_form
      solution:
        formula: |-
          v = (4 y (1 - y), 0)
          p = -8 x
"""


@needs_mpiexec
def test_a_pressure_null_space_runs_on_two_ranks(tmp_path):
    """Taylor-Hood (each rank's DOFs blocked by component) and SUPG/PSPG P1/P1
    (interleaved, block size 3), the pressure fixed only up to a constant."""
    _spec(tmp_path, _without_discretizations(CHANNEL_SPEC).replace(
        "  analytical: []\n", POISEUILLE), name="channel.ymf")
    assert _weak("channel.ymf", cwd=tmp_path)["exit_status"] == 0
    assert _dis("channel.weak.ymf", "--cells", "4", cwd=tmp_path)["exit_status"] == 0
    errors = {}
    for ranks in (1, 2):
        out = pro_on(ranks, "run", "channel.discrete.ymf", "--outdir", str(ranks),
                     cwd=tmp_path)
        assert out.returncode == 0, out.stderr[-3000:]
        outputs = _outputs(tmp_path / str(ranks) / "channel.discrete.archive.ymf")
        assert {o["transformations"]["solve"]["by"]["ranks"] for o in outputs.values()} \
            == {ranks}
        errors[ranks] = {k: o["verification"]["l2_errors"] for k, o in outputs.items()}
    (th,) = [k for k in errors[1] if k.startswith("galerkin_taylor_hood/")]
    (p1,) = [k for k in errors[1] if k.startswith("supg_pspg_p1p1/")]
    for ranks in (1, 2):
        assert max(errors[ranks][th].values()) < 1e-10    # Poiseuille is in the space
    for name, e in errors[1][p1].items():
        assert abs(errors[2][p1][name] - e) <= 1e-10 * e


def test_the_null_space_knows_each_owned_degree_of_freedoms_component():
    import numpy
    from types import SimpleNamespace
    from proteus.LinearSolvers import ConstantComponentsNullSpace as NS
    coefficients = SimpleNamespace(nc=3)
    # equal order, interleaved (block size 3): node by node, p v_0 v_1
    interleaved = SimpleNamespace(coefficients=coefficients, nFreeDOF_global=[4, 4, 4],
                                  offset=[0, 1, 2], stride=[3, 3, 3],
                                  owned_local=numpy.arange(9))     # 3 of 4 nodes owned
    assert list(NS.components_of_owned(interleaved)) == [0, 1, 2] * 3
    # mixed (Taylor-Hood), blocked: p's DOFs, then v_0's, then v_1's; each
    # component's owned ones first, the rest ghosts
    blocked = SimpleNamespace(coefficients=coefficients, nFreeDOF_global=[3, 5, 5],
                              offset=[0, 3, 8], stride=[1, 1, 1],
                              owned_local=numpy.array([0, 1, 3, 4, 5, 6, 8, 9, 10, 11]))
    assert list(NS.components_of_owned(blocked)) == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_the_default_solver_on_several_ranks_is_a_parallel_one():
    from proteus import ADRProblem
    problem = {"adr": {"components": ["u"]}}
    cg1 = {"u": ("CG", 1)}
    assert ADRProblem.linear_solver_choice(problem, cg1, ranks=1)[0] == "superlu"
    assert ADRProblem.linear_solver_choice(problem, cg1, ranks=2)[0] in ("mumps", "petsc")
    with pytest.raises(ValueError, match="superlu is serial"):
        ADRProblem.linear_solver_choice(problem, cg1, "superlu", ranks=2)


def test_a_missing_spec_is_an_argument_error(tmp_path):
    out = pro("run", "nothere.ymf", cwd=tmp_path)
    assert out.returncode == 2 and "nothere.ymf: no such file" in out.stderr
    assert "Traceback" not in out.stderr


@pytest.mark.skipif(shutil.which("parun") is None, reason="parun not on PATH")
def test_run_archive_options_reach_the_archive(tmp_path):
    """--archive-mode, --archive-driver and --archive-document are pro run's too."""
    from ymf.archive import read_ymf
    sources = [os.path.join(HERE, "ci", f)
               for f in ("adr.py", "ladr_2d_p.py", "ladr_2d_n.py")]
    for s in sources:
        shutil.copy(s, tmp_path)
    ran = pro("run", "--archive-mode", "kitchen-sink", "--archive-driver", "serial",
              "ladr_2d_p.py", "ladr_2d_n.py", cwd=tmp_path)
    assert ran.returncode == 0, ran.stderr[-2000:]
    _, extra = read_ymf(tmp_path / "ladr_2d.ymf")
    provenance = extra["proteus"]
    assert provenance["archive_mode"] == "kitchen-sink"
    assert provenance["archive_driver"] == "serial"
    assert "log" in provenance and "command" in provenance
    # the document layout is refused where it makes no sense, by name
    ran = pro("run", "--archive-document", "per-rank", "ladr_2d_p.py", "ladr_2d_n.py",
              cwd=tmp_path)
    assert ran.returncode != 0
    assert "document='per-rank' needs driver='per-rank'" in ran.stderr
