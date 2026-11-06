"""Solve a YMF problem with Proteus, reproducibly: make for archives.

Run as ``pro run SPEC.ymf`` or as the script ``scripts/ymf_run``:

    ymf_run examples/poisson.ymf                     # every discretization
    ymf_run spec.ymf --discretization taylor_hood --cells 4 8 16
    ymf_run out/poisson.archive.ymf                  # fill in what is missing
    ymf_run out/poisson.archive.ymf --check          # rerun all, and compare

The input is a problem specification (monolithic, or composed with
``extends``, possibly of an archive) or an archive. Its archive -- one per
input, ``<outdir>/<stem>.archive.ymf`` -- is the input as realized,
followed by ``outputs``: one per discretization and mesh, keyed by a hash
of the part of the input that produced it (see ymf.closure). Each holds
the transformations that produced it (strong form -> ADR form -> Proteus's
discrete problem -> solve), the L2 errors against the exact solution, and
the approximation itself, its arrays in ``<stem>_<digest>.h5`` beside the
archive or, with ``--inline``, in the archive.

With ``--emit-pn`` it also writes each output's problem as the classic
Proteus files ``<stem>_<digest>_p.py``, ``_n.py`` and ``_so.py``: what
proteus.ADRProblem builds in memory, as readable Python that ``parun``
and ``pro run`` run, bitwise the same. That is how a spec reaches a
compute platform without ymf[symbolic]: emit there where sympy is,
run the files here.

Like make, it computes only the outputs the archive lacks. ``--cells`` and
``--levels`` override the discretizations' ``mesh``, and the archive
records the override. ``--check`` reruns the outputs already there too, and
compares each to its record: the ADR form, the errors and every array,
bitwise. A match is noted under ``reproduced``. A difference is refused,
with the differences listed, unless ``--keep-different`` keeps the rerun
as a second output, ``<key>~2``.

Under ``mpiexec`` every rank solves its part of each output; rank 0 makes
the meshes of regions (:mod:`proteus.cli.ymfmesh`), reads what Proteus
wrote, writes the archive and reports, and the other ranks print nothing.
The archive records the number of ranks, and a rerun on another number is
not expected to be bitwise the same: its partition sums in another order.
Refused on more than one rank, before anything is solved: Proteus's serial
LU (``linear_solver: superlu``; the default there is MUMPS), and periodic
conditions other than in CG1 without advection (see
``ADRProblem.parallel_periodic``).

This is the front end, so it imports sympy through ymf.symbolic. Proteus
itself (proteus.ADRProblem) never does: it is handed plain data.
"""
import os

# Bitwise reproducibility needs a fixed order of floating-point operations.
# Threaded BLAS (inside MUMPS, say) sums in a different order on each run,
# so the threads are pinned to one before anything numeric is loaded --
# unless the caller set them, in which case the record says so.
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS")
for _variable in THREADS:
    os.environ.setdefault(_variable, "1")

import argparse
import datetime
import math
import platform
import subprocess
import sys


def element_spaces(problem, finite_element):
    """Map the spec's finite_element block onto scalar components.

    ``fields`` applies to every unknown; in a velocity-pressure system,
    ``velocity`` applies to the vector unknown and ``pressure`` to the
    scalar one.
    """
    unknowns = problem["unknowns"]
    spaces = {}
    if "fields" in finite_element:
        fe = finite_element["fields"]
        for comps in unknowns.values():
            for c in comps:
                spaces[c] = [fe["family"], int(fe["order"])]
    for key, vector in (("velocity", True), ("pressure", False)):
        if key in finite_element:
            fe = finite_element[key]
            for name, comps in unknowns.items():
                if (len(comps) > 1) == vector:
                    for c in comps:
                        spaces[c] = [fe["family"], int(fe["order"])]
    missing = [c for comps in unknowns.values() for c in comps if c not in spaces]
    if missing:
        raise SystemExit("finite_element gives no space for %s" % ", ".join(missing))
    return spaces


def time_stepping(discretization, problem):
    """None for a steady solve, else {"dt", "outputs"} (of a realized entry)."""
    block = discretization.get("discretization") or {}
    if "time_integration" not in block:
        return None
    if block["time_integration"] != "backward_euler":
        raise SystemExit("time integration %r is not wired up yet (backward_euler is)"
                         % block["time_integration"])
    if not problem.get("time"):
        raise SystemExit("%s steps in time, but the problem's domain gives no time "
                         "interval (', t ∈ (t0, t1]')" % discretization["name"])
    return {"dt": float(block["dt"]), "outputs": int(block.get("outputs", 1))}


def stabilization_of(doc, discretization):
    weak = {w["label"]: w for w in doc["model"].get("weak_forms", [])}.get(
        str(discretization.get("from_weak_form")), {})
    return (weak.get("stabilization_method") or "none").strip().lower()


def software():
    """Versions of everything the result depends on."""
    import numpy
    import sympy
    import ymf
    import proteus
    def commit(module):
        here = os.path.dirname(os.path.abspath(module.__file__))
        try:
            return subprocess.run(["git", "-C", here, "describe", "--always", "--dirty"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None      # an installed package, not a checkout
    return {"ymf": ymf.__version__, "ymf_git": commit(ymf),
            "proteus": proteus.__version__, "proteus_git": commit(proteus),
            "sympy": sympy.__version__, "numpy": numpy.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "threads": {v: os.environ[v] for v in THREADS}, "ranks": _ranks()}


def _ranks():
    from proteus import Comm
    return Comm.comm.size() if Comm.comm is not None else 1


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# --- one output ----------------------------------------------------------------


def run_arguments(doc, problem, name, realization):
    """ADRProblem.run's arguments for one discretization on one realization."""
    from ymf import closure
    entry = closure.realize(next(d for d in doc["solution_paths"]["discretizations"]
                                 if d["name"] == name), realization)
    spaces = element_spaces(problem, entry.get("finite_element") or {})
    solver = entry.get("solver") or {}
    return {"spaces": {c: tuple(s) for c, s in spaces.items()},
            # a box by cells per side, any other region by h (element size)
            "cells": int(realization["cells"]) if "cells" in realization else None,
            "h": float(realization["h"]) if "h" in realization else None,
            "time": time_stepping(entry, problem),
            "stabilization": stabilization_of(doc, entry),
            "levels": int(realization.get("levels", 1)),
            "tolerance": solver.get("tolerance"),
            "max_iterations": solver.get("max_iterations"),
            "linear_solver": solver.get("linear_solver"),
            "petsc_options": solver.get("petsc_options")}


def emit(doc, problem, name, realization, outdir, files, key, archive, base=None):
    """Write the p, n and so files of one output; return their paths. A file domain's relative path is
    taken from ``base``, the spec's directory, as :func:`solve` takes it."""
    from proteus import ADRProblem
    return ADRProblem.emit_pn(
        with_paths_from(problem, base or os.getcwd()), name=files, directory=outdir,
        header="Output %s of %s." % (key, os.path.basename(archive)),
        **run_arguments(doc, problem, name, realization))


def with_paths_from(problem, base):
    """The problem with a file domain's relative path taken from ``base``,
    the spec's directory. Only the solve sees it: the archive keeps the path
    as the spec wrote it."""
    region = problem["domain"].get("region")
    if not region or region.get("kind") != "file" or os.path.isabs(region["path"]):
        return problem
    region = dict(region, path=os.path.join(base, region["path"]))
    return dict(problem, domain=dict(problem["domain"], region=region))


def solve(doc, problem, name, realization, outdir, files, inline, env, mesh=None,
          base=None):
    """Run one discretization on one realization; return its output record.

    ``mesh`` is the prefix of a region's mesh files, already made
    (:mod:`proteus.cli.ymfmesh`); None meshes as part of the solve.

    ``files`` names what Proteus writes (``<files>.h5``); the .ymf it writes
    beside it is dropped, since the archive holds the record.
    """
    from proteus import ADRProblem, Profiling, runtime
    from ymf.archive import inline_domain, read_ymf
    from ymf import closure
    comm = runtime.start()     # one log for the whole run, in the working directory
    Profiling.logLevel = 1
    Profiling.verbose = False
    kwargs = run_arguments(doc, problem, name, realization)
    stabilization = kwargs["stabilization"]
    opts = runtime.RunOptions(dataDir=outdir)
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        ns, _, errors = ADRProblem.run(with_paths_from(problem, base or cwd), name=files,
                                       opts=opts, mesh=mesh, **kwargs)
    finally:
        os.chdir(cwd)
    solved = problem
    if stabilization == "supg/pspg":          # run() solved it pressure first
        velocity, pressure = ADRProblem.velocity_pressure(problem)
        solved = ADRProblem.reorder(problem, [pressure] + velocity)
    discrete = dict(ADRProblem.describe(solved, ns.nList[0]), stabilization=stabilization)

    # The record Proteus wrote beside its .h5, on rank 0 once every rank
    # has finished writing: rank 0 reads it, and every rank gets the output.
    comm.barrier()
    if not comm.isMaster():
        return _bcast(comm, None)
    domain, _ = read_ymf(os.path.join(outdir, files + ".ymf"))
    os.remove(os.path.join(outdir, files + ".ymf"))
    if inline:
        domain = inline_domain(domain, outdir)
        os.remove(os.path.join(outdir, files + ".h5"))
    return _bcast(comm, {
        "input": {"sha256": closure.input_digest(doc, name, realization)},
        "realization": realization,
        "transformations": {
            "strong_to_adr": {"by": "ymf.symbolic %s" % env["ymf"], "adr": problem},
            "adr_to_discrete": dict({"by": "proteus.ADRProblem %s" % env["proteus"]},
                                    **discrete),
            "solve": {"by": env, "created": now()},
        },
        "storage": "inline" if inline else files + ".h5",
        "verification": dict({"l2_errors": errors}, **(
            {"modulo_constants": list(problem["up_to_constant"])}
            if problem.get("up_to_constant") else {})),
        "approximation": domain,
    })


def differences(recorded, rerun, outdir):
    """What differs between two records of one output (empty if nothing)."""
    import numpy
    from ymf.archive import domain_arrays
    out = []
    for step, what in (("strong_to_adr", "the ADR form"), ("dispatch", "the dispatch")):
        a = recorded["transformations"].get(step)
        b = rerun["transformations"].get(step)
        a, b = [{k: v for k, v in (x or {}).items() if k != "by"} for x in (a, b)]
        if a != b:
            out.append("%s differs" % what)
    a = (recorded.get("verification") or {}).get("l2_errors") or {}
    b = (rerun.get("verification") or {}).get("l2_errors") or {}
    for n in sorted(set(a) | set(b)):
        if a.get(n) != b.get(n):
            out.append("L2(%s): recorded %r, rerun %r" % (n, a.get(n), b.get(n)))
    arrays = [domain_arrays(recorded["approximation"], outdir),
              domain_arrays(rerun["approximation"], outdir)]
    if set(arrays[0]) != set(arrays[1]):
        out.append("different fields: %s" % sorted(set(arrays[0]) ^ set(arrays[1])))
    for where in sorted(set(arrays[0]) & set(arrays[1])):
        x, y = arrays[0][where], arrays[1][where]
        if x.dtype != y.dtype or x.shape != y.shape:
            out.append("%s: %s%s, rerun %s%s" % (where, x.dtype, x.shape, y.dtype, y.shape))
        elif not numpy.array_equal(x, y):
            diff = numpy.abs(x.astype(float) - y.astype(float))
            out.append("%s: %d of %d values differ, by up to %.3e"
                       % (where, int((diff > 0).sum()), diff.size, float(diff.max())))
    ranks = [r["transformations"]["solve"]["by"].get("ranks", 1) for r in (recorded, rerun)]
    if out and ranks[0] != ranks[1]:
        # another partition sums in another order: equal to rounding, not bitwise
        out.append("recorded on %d rank%s, rerun on %d: a different partition is not "
                   "expected to reproduce bitwise" % (ranks[0], "" if ranks[0] == 1 else "s",
                                                      ranks[1]))
    return out


def discard(output, outdir):
    """Remove the file of an output that is not being kept."""
    if output["storage"] != "inline":
        os.remove(os.path.join(outdir, output["storage"]))


# --- the archive -------------------------------------------------------------------


def fixes_mesh(doc):
    """Whether the spec's domain is a mesh, or a file of one, so that its
    discretizations give only levels (ymf.discretizations.mesh_from_domain)."""
    try:
        from ymf.discretizations import mesh_from_domain
    except ImportError:                   # a ymf before file domains
        return False
    return mesh_from_domain(doc["model"]["strong_form"].get("domain"))


def override(doc, names, field, values, flag):
    """Set ``mesh.<field>`` of the named discretizations, as a recorded override."""
    value = values if len(values) > 1 else values[0]
    for d in doc["solution_paths"]["discretizations"]:
        if d["name"] not in names:
            continue
        mesh = d.setdefault("mesh", {})
        if field == "levels" and not ("cells" in mesh or "h" in mesh) and not fixes_mesh(doc):
            del d["mesh"]
            continue                       # no mesh to refine; reported later
        if mesh.get(field) == value:
            continue
        doc.setdefault("composition", {}).setdefault("overrides", []).append({
            "path": "solution_paths.discretizations[%s].mesh.%s" % (d["name"], field),
            "was": mesh.get(field), "now": value, "set_by": flag})
        mesh[field] = value


def report(problem, rows):
    """The convergence table of one discretization."""
    names = list(problem["exact"])
    unstructured = any("h" in realization for _, realization, _, _ in rows)
    fixed = not any("h" in r or "cells" in r for _, r, _, _ in rows)   # a mesh file
    print(("      h  " if unstructured else " levels  " if fixed else "  cells  ") + "".join("  L2(%s)%s  rate  " % (
        n, "*" if n in (problem.get("up_to_constant") or []) else " ") for n in names)
          + "  output" + ("   (* modulo constants)" if problem.get("up_to_constant") else ""))
    previous = None
    for key, realization, output, status in rows:
        refinements = 2 ** (int(realization.get("levels", 1)) - 1)
        # the finest mesh's size, relative: 1/cells for a box, h otherwise
        if "h" in realization:
            size = float(realization["h"]) / refinements
            row = "  %5.3g  " % size
        elif "cells" not in realization:
            # the file's own mesh, refined: sizes relative to it
            size = 1.0 / refinements
            row = "  %5d  " % int(realization.get("levels", 1))
        else:
            finest = int(realization["cells"]) * refinements
            size = 1.0 / finest
            row = "  %5d  " % finest
        errors = output["verification"]["l2_errors"]
        for n in names:
            rate = ""
            if (previous and previous[1].get(n, 0) > 0 and errors[n] > 0
                    and previous[0] != size):
                rate = "%5.2f" % (math.log(previous[1][n] / errors[n])
                                  / math.log(previous[0] / size))
            row += "  %.3e %5s  " % (errors[n], rate)
        print(row + "  " + key.split("/")[-1] + "  " + status, flush=True)
        previous = (size, errors)


def dispatches_to(found):
    """``flow RANS2P, vof VOF, ...``: what a dispatched discretization runs."""
    return ", ".join("%s %s" % (m.subproblem, m.module.rpartition(".")[2]) for m in found)


def split_blockers(doc, entry):
    """Why a discretization split into sub-problems cannot run (empty if it
    can): what ymf says it lacks, then what the dispatch (proteus.Dispatch)
    refuses, then a family nothing realizes yet (proteus.DispatchRun)."""
    from ymf.discretizations import missing
    from ymf.symbolic import SymbolicError
    from ymf.symbolic.split import split_problem
    from proteus.Dispatch import DispatchError, dispatch
    from proteus.DispatchRun import precheck
    name = entry["name"]
    head = "%s: split into sub-problems (%s)" % (
        name, ", ".join(s.get("name", "?") for s in entry["subproblems"]))
    try:
        found = dispatch(doc, name)
    except DispatchError as e:
        return ["%s, and the dispatch refuses it: %s" % (head, e)]
    try:
        precheck(doc, found, split_problem(doc, name))
    except (DispatchError, SymbolicError) as e:
        # what the run would be refused on, said before anything runs
        return ["%s. It dispatches to %s, and cannot run: %s" % (head, dispatches_to(found), e)]
    lacking = missing(entry, doc["model"]["strong_form"].get("domain"))
    if lacking:
        return ["%s is missing %s: %s" % (name, m.what, m.hint) for m in lacking]
    mesh = entry.get("mesh") or {}
    if "cells" in mesh or any(int(v) != 1 for v in _listed(mesh.get("levels", 1))):
        # the deck machinery meshes the region with triangle, at element size h, once
        return ["%s: a split discretization is meshed by element size, one level (mesh: {h: ...}); "
                "this one gives %s" % (name, ", ".join("%s %s" % kv for kv in sorted(mesh.items())))]
    return []


def _listed(value):
    return value if isinstance(value, list) else [value]


def solve_split(doc, name, realization, outdir, files, env):
    """Run one split discretization on one realization from its dispatch,
    through the deck machinery (proteus.DispatchRun); return its output
    record: the dispatch and what the realization took from where, and the
    approximation Proteus wrote."""
    from proteus import DispatchRun, Profiling, runtime
    from proteus.Dispatch import dispatch
    from ymf.archive import read_ymf
    from ymf import closure
    from ymf.symbolic.split import split_problem
    comm = runtime.start()
    Profiling.logLevel = 1
    Profiling.verbose = False
    found = dispatch(doc, name)
    split = split_problem(doc, name)
    opts = runtime.RunOptions(dataDir=outdir)
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        with DispatchRun.petsc_options_restored():
            realized = DispatchRun.realize(doc, found, split, realization, files, outdir,
                                           comm.size())
            DispatchRun.run(realized, files, opts)
    finally:
        os.chdir(cwd)
    comm.barrier()
    if not comm.isMaster():
        return _bcast(comm, None)
    domain, _ = read_ymf(os.path.join(outdir, files + ".ymf"))
    os.remove(os.path.join(outdir, files + ".ymf"))
    return _bcast(comm, {
        "input": {"sha256": closure.input_digest(doc, name, realization)},
        "realization": realization,
        "transformations": {
            "dispatch": dict({"by": "proteus.Dispatch %s" % env["proteus"]},
                             **_plain(found.as_dict())),
            "realize": dict({"by": "proteus.DispatchRun %s" % env["proteus"]},
                            **_plain(realized.record)),
            "solve": {"by": env, "created": now()},
        },
        "storage": files + ".h5",
        "approximation": domain,
    })


def _plain(value):
    """``value`` with string keys and lists for tuples, as the archive writes it."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def blockers(doc, problem, entry):
    """Why one discretization cannot close in this run: what ymf says it
    lacks (with ymf's hints), then what Proteus cannot run. Empty if it can.
    """
    from ymf.discretizations import missing
    from proteus.ADRProblem import STABILIZATIONS, linear_solver_choice, parallel_periodic
    name = entry["name"]
    domain = doc["model"]["strong_form"].get("domain")
    try:
        try:
            lacking = missing(entry, domain)   # what a mesh needs depends on it
        except TypeError:                      # a ymf before file domains
            lacking = missing(entry)
    except ValueError as e:                    # an unknown method
        return [str(e)]
    if lacking:
        return ["%s is missing %s: %s" % (name, m.what, m.hint) for m in lacking]
    if entry.get("method", "mesh") != "mesh":
        return ["%s: Proteus solves method mesh only, and this branch's method is %s"
                % (name, entry["method"])]
    if entry.get("subproblems"):
        # solving the whole model as one system would run a different
        # discretization under this branch's name: the dispatch runs it
        return split_blockers(doc, entry)
    method = stabilization_of(doc, entry)
    if method not in STABILIZATIONS:
        # running it without would solve a different weak form under this
        # branch's name
        return ["%s: its weak form needs %s stabilization, which Proteus does not "
                "wire up yet (have %s)" % (name, method, ", ".join(STABILIZATIONS))]
    spaces = element_spaces(problem, entry.get("finite_element") or {})
    solver = entry.get("solver") or {}
    try:
        linear_solver_choice(problem, spaces, solver.get("linear_solver"),
                             solver.get("petsc_options"))
        parallel_periodic(problem, spaces)
    except ValueError as e:
        # running it with another solver would not be what it asks for
        return ["%s: %s" % (name, e)]
    return []


def main(argv=None, prog="ymf_run", mesh_only=False):
    """Run the YMF runner; ``prog`` names it in help and in the archive's
    record of command-line overrides. ``mesh_only`` (``pro msh SPEC.ymf``)
    brings the meshes the run needs up to date, and solves nothing."""
    from . import parser_kwargs
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0],
                                     **parser_kwargs())
    parser.add_argument("input", help="a spec .ymf, or an archive")
    parser.add_argument("--discretization", action="append",
                        help="run only this one (by name; repeatable)")
    parser.add_argument("--cells", type=int, nargs="+",
                        help="override mesh.cells: cells per side of each coarsest mesh")
    parser.add_argument("--h", type=float, nargs="+",
                        help="override mesh.h: element size of each coarsest unstructured "
                             "mesh (a region that is not a box)")
    parser.add_argument("--levels", type=int, nargs="+",
                        help="override mesh.levels: refine uniformly levels-1 times and "
                             "solve on the finest (Proteus nLevels)")
    parser.add_argument("--inline", action="store_true",
                        help="write new outputs' arrays into the archive itself, no .h5")
    parser.add_argument("--outdir",
                        help="where the archive goes (default: beside an archive input, "
                             "else the current directory)")
    parser.add_argument("--check", action="store_true",
                        help="rerun the outputs already in the archive too, and compare")
    parser.add_argument("--keep-different", action="store_true",
                        help="with --check: keep a differing rerun as <key>~N")
    parser.add_argument("--emit-pn", action="store_true",
                        help="also write each output's problem as Proteus p, n and so files, "
                             "<stem>_<digest>_{p,n,so}.py, which parun runs")
    parser.add_argument("--prune", action="store_true",
                        help="drop outputs of an earlier version of the input from the "
                             "archive (their files are left in place)")
    parser.add_argument("--ignore", action="store_true",
                        help="run the discretizations that can close and leave out the "
                             "rest; without it, a run that cannot close stops before "
                             "computing anything")
    parser.add_argument("--stale-ok", action="store_true",
                        help="run even though a generated file in the spec's chain (a "
                             ".weak.ymf or .discrete.ymf) is stale: its input changed since "
                             "it was written")
    parser.add_argument("--no-mesh", action="store_true",
                        help="stop with an error, before solving anything, when the mesh "
                             "of a region (not a box) is missing or out of date, instead "
                             "of making it on rank 0; pro msh SPEC.ymf makes it")
    parser.add_argument("--force", action="store_true",
                        help="remake the meshes of regions even if they are up to date")
    args = parser.parse_args(argv)
    if not os.path.isfile(args.input):
        parser.error("%s: no such file" % (args.input,))
    # MPI starts only now: --help and argument errors are answered without it
    from proteus import runtime
    comm = runtime.start()
    if comm.isMaster():
        return _run(args, prog, mesh_only, comm)
    return _quietly(lambda: _run(args, prog, mesh_only, comm))


def _quietly(run):
    """``run()`` on a rank other than 0, which computes alongside rank 0 but
    prints nothing: rank 0 reports. A failure still says where it was."""
    import contextlib
    import traceback
    with open(os.devnull, "w") as devnull:
        try:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return run()
        except SystemExit as exc:              # rank 0 says why
            raise SystemExit(exc.code if isinstance(exc.code, int) else 1)
        except BaseException:
            traceback.print_exc(file=sys.__stderr__)
            raise


def _bcast(comm, value):
    """Rank 0's ``value`` on every rank."""
    return value if comm.size() == 1 else comm.comm.tompi4py().bcast(value, root=0)


def _run(args, prog, mesh_only, comm):
    """The runner, on every rank. What reads or writes files, and every
    decision made from them, is rank 0's, and broadcast."""
    root = comm.isMaster()

    from ymf import closure
    from ymf.cli import archive2xmf
    from ymf.archive import is_archive_text
    from ymf.symbolic import adr_problem

    with open(args.input, encoding="utf-8") as f:
        input_is_archive = is_archive_text(f.read())
    if not input_is_archive and not args.stale_ok:
        # A file pro weak or pro dis wrote from an input that has changed
        # since describes the old input: running it would solve that.
        from .chain import stale
        found = stale(args.input)
        if found:
            print("%s: refusing to run: a generated file in its chain is stale"
                  % (prog,), file=sys.stderr)
            for info in found:
                g = info["generated"]
                print("  %s: %s" % (os.path.relpath(info["file"]),
                                    "; ".join(g.get("reasons") or [g.get("error", "")])),
                      file=sys.stderr)
                if g.get("regenerate"):
                    print("    regenerate it: %s" % (g["regenerate"],), file=sys.stderr)
            print("or run it as it is with --stale-ok", file=sys.stderr)
            return 2
    stem = closure.archive_stem(args.input)
    if args.outdir:
        outdir = os.path.abspath(args.outdir)
    elif input_is_archive:
        outdir = os.path.dirname(os.path.abspath(args.input))
    else:
        outdir = os.getcwd()
    os.makedirs(outdir, exist_ok=True)
    archive = os.path.join(outdir, stem + ".archive.ymf")

    # Outputs come from the input (an archive, or what it extends) and from
    # the archive of an earlier run of this input. A runner brings an
    # archive up to date, so it loads one whose declared ymf_state its
    # content no longer reaches (an edited input, stale outputs), and the
    # archive it writes records the state it does reach.
    doc, outputs, dropped = closure.load(args.input, check_state=False)
    doc = closure.plain(doc)
    for key, why in dropped.items():
        print("not inherited: %s (%s)" % (key, why))
    if os.path.abspath(args.input) != archive and os.path.exists(archive):
        _, existing, _ = closure.load(archive, check_state=False)
        for key, output in existing.items():
            outputs.setdefault(key, output)

    names = [d["name"] for d in doc["solution_paths"]["discretizations"]]
    selected = args.discretization or names
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise SystemExit("no discretization named %s (have %s)"
                         % (", ".join(unknown), ", ".join(names)))
    flags = "".join(" --discretization " + n for n in args.discretization or [])
    for field in ("cells", "h", "levels"):
        values = getattr(args, field)
        if values:
            override(doc, selected, field, values, "%s%s --%s %s"
                     % (prog, flags, field, " ".join(map(str, values))))

    outputs, stale = closure.partition(doc, outputs)
    if stale:
        print("%s holds outputs of an earlier version of the input:" % archive)
        for key, why in stale.items():
            print("  %s: %s" % (key, why))
        if not args.prune:
            raise SystemExit("refusing to drop them: rerun with --prune, or use another --outdir")

    # A split discretization is not one ADR system, and a model that only a
    # split can solve (two-phase level-set flow) is not one either: it runs
    # from its dispatch (proteus.Dispatch, proteus.DispatchRun), checked
    # before the model is classified as one system, and before anything runs.
    split = [d for d in doc["solution_paths"]["discretizations"]
             if d["name"] in selected and d.get("subproblems")]
    others = [n for n in selected if n not in [d["name"] for d in split]]
    blocked = [(d["name"], split_blockers(doc, d)) for d in split]
    blocked = [(name, reasons) for name, reasons in blocked if reasons]
    if blocked and not args.ignore:
        print("\n%s: %s cannot run, so nothing was run:" % (
            prog, "this discretization" if len(blocked) == 1 else
            "%d discretizations" % len(blocked)), file=sys.stderr)
        for name, reasons in blocked:
            for reason in reasons:
                print("  " + reason, file=sys.stderr)
        print("Give what is missing, choose the branches with --discretization, or "
              "rerun with --ignore to run the ones that can.", file=sys.stderr)
        return 2
    for name, reasons in blocked:
        print("\n%s -- left out (--ignore): %s" % (name, "; ".join(reasons)))
    split = [d for d in split if d["name"] not in dict(blocked)]

    print("%s: %s" % (args.input, doc["model"]["name"]))
    env = software()
    planned = list(closure.planned_outputs(doc))
    new, refused = [], []
    if split and mesh_only:
        print("\n%s: a split discretization meshes as part of its solve: %s"
              % (prog, ", ".join(d["name"] for d in split)))
        split = []
    for entry in split:
        _run_split(args, doc, entry, planned, outputs, new, refused, outdir, stem, env, comm)
    if not others:
        return _finish(args, comm, archive, doc, outputs, new, refused, outdir)

    problem = adr_problem(doc)
    print("unknowns: %s" % ", ".join("%s (%s)" % (n, ", ".join(c))
                                    for n, c in problem["unknowns"].items()))
    selected = others

    # Every selected branch is checked before anything is computed: what
    # ymf says it lacks (ymf.discretizations), and what Proteus can run.
    runnable, blocked = [], []
    for name in selected:
        entry = next(d for d in doc["solution_paths"]["discretizations"] if d["name"] == name)
        reasons = blockers(doc, problem, entry)
        if reasons:
            blocked.append((name, reasons))
        else:
            runnable.append((name, entry, [(k, r) for k, n, r in planned if n == name]))
    if blocked and not args.ignore:
        print("\n%s: %s cannot close, so nothing was run:" % (
            prog, "this discretization" if len(blocked) == 1 else
            "%d discretizations" % len(blocked)), file=sys.stderr)
        for name, reasons in blocked:
            for reason in reasons:
                print("  " + reason, file=sys.stderr)
        print("Give what is missing, choose the branches with --discretization, or "
              "rerun with --ignore to run the ones that can close.", file=sys.stderr)
        return 2
    for name, reasons in blocked:
        print("\n%s -- left out (--ignore): %s" % (name, "; ".join(reasons)))

    # What will be solved, and the meshes it is solved on: those of regions
    # are brought up to date first, on rank 0, before anything is computed
    # (see proteus.cli.ymfmesh). A box is meshed in memory by each solve.
    from . import command, ymfmesh
    todo = [(name, key, realization) for name, _, mine in runnable
            for key, realization in mine if args.check or key not in outputs]
    meshes = ymfmesh.needed(doc, problem, todo, outdir, stem)
    mesh_of = {}
    if meshes:
        from proteus import runtime
        comm = runtime.start()
        if mesh_only and comm.size() > 1:
            if comm.isMaster():
                print("%s: meshing is serial; run it on one process, not under mpiexec"
                      % (prog,), file=sys.stderr)
            return 2
        # the same outputs, so the same meshes: what selects them carries over
        remedy = "%s %s%s%s%s%s" % (
            command("msh"), args.input, flags,
            " --h " + " ".join(map(str, args.h)) if args.h else "",
            " --outdir " + args.outdir if args.outdir else "",
            " --check" if args.check else "")
        print()
        error = ymfmesh.bring_up_to_date(meshes, comm, no_mesh=args.no_mesh and not mesh_only,
                                         force=args.force, remedy=remedy)
        if error:
            sys.stdout.flush()
            print("\n%s: %s" % (prog, error), file=sys.stderr)
            return 2
        mesh_of = {key: prefix for prefix, m in meshes.items() for key in m["outputs"]}
    if mesh_only:
        if not meshes:
            print("\n%s: nothing to mesh: %s" % (prog, "every output to compute is on a box, "
                                                 "meshed in memory" if todo else
                                                 "every output is in the archive (--check "
                                                 "meshes for them too)"))
        return 0

    for name, entry, mine in runnable:
        method = stabilization_of(doc, entry)
        spaces = element_spaces(problem, entry.get("finite_element") or {})
        print("\n%s -- %s%s" % (name, ", ".join("%s %s%d" % (c, f, o)
                                                for c, (f, o) in spaces.items()),
                                "" if method == "none" else ", " + method))
        rows = []
        for key, realization in mine:
            if key in outputs and not args.check:
                rows.append((key, realization, outputs[key], "in the archive"))
                continue
            n = 1
            while ("%s~%d" % (key, n) if n > 1 else key) in outputs:
                n += 1
            slot = "%s~%d" % (key, n) if n > 1 else key
            output = solve(doc, problem, name, realization, outdir,
                           closure.file_stem(stem, slot), args.inline, env,
                           mesh=mesh_of.get(key),
                           base=os.path.dirname(os.path.abspath(args.input)))
            if key not in outputs:
                outputs[key] = output
                new.append(key)
                rows.append((key, realization, output, "new"))
                continue
            found = _bcast(comm, differences(outputs[key], output, outdir) if root else None)
            if not found:
                if root:
                    discard(output, outdir)
                outputs[key].setdefault("reproduced", []).append(
                    {"created": output["transformations"]["solve"]["created"], "by": env})
                rows.append((key, realization, outputs[key], "reproduced, bitwise"))
            elif args.keep_different:
                outputs[slot] = output
                new.append(slot)
                rows.append((slot, realization, output, "DIFFERENT, kept"))
            else:
                if root:
                    discard(output, outdir)
                refused.append((key, found))
                rows.append((key, realization, outputs[key], "DIFFERENT, refused"))
        report(problem, rows)
        if args.emit_pn and root:
            for key, realization, _, status in rows:
                if "refused" in status:
                    continue
                paths = emit(doc, problem, name, realization, outdir,
                             closure.file_stem(stem, key), key, archive,
                             base=os.path.dirname(os.path.abspath(args.input)))
                print("  %s: %s" % (key.split("/")[-1], ", ".join(map(os.path.basename, paths))))

    return _finish(args, comm, archive, doc, outputs, new, refused, outdir)


def _run_split(args, doc, entry, planned, outputs, new, refused, outdir, stem, env, comm):
    """Run the outputs of one split discretization the archive lacks (with
    --check, all of them, compared bitwise with their records), and report."""
    from ymf import closure
    from proteus.Dispatch import dispatch
    name = entry["name"]
    print("\n%s -- split, dispatched to %s" % (name, dispatches_to(dispatch(doc, name))))
    for key, realization in [(k, r) for k, n, r in planned if n == name]:
        mesh = ", ".join("%s %s" % (k, v) for k, v in realization.items())
        if key in outputs and not args.check:
            print("  %s  %s  in the archive" % (mesh, key.split("/")[-1]))
            continue
        n = 1
        while ("%s~%d" % (key, n) if n > 1 else key) in outputs:
            n += 1
        slot = "%s~%d" % (key, n) if n > 1 else key
        output = solve_split(doc, name, realization, outdir, closure.file_stem(stem, slot), env)
        if key not in outputs:
            outputs[key] = output
            new.append(key)
            status = "new"
        else:
            found = _bcast(comm, differences(outputs[key], output, outdir)
                           if comm.isMaster() else None)
            if not found:
                if comm.isMaster():
                    discard(output, outdir)
                outputs[key].setdefault("reproduced", []).append(
                    {"created": output["transformations"]["solve"]["created"], "by": env})
                status = "reproduced, bitwise"
            elif args.keep_different:
                outputs[slot] = output
                new.append(slot)
                status = "DIFFERENT, kept"
            else:
                if comm.isMaster():
                    discard(output, outdir)
                refused.append((key, found))
                status = "DIFFERENT, refused"
        print("  %s  %s  %s" % (mesh, (slot if status == "DIFFERENT, kept" else key).split("/")[-1],
                                status), flush=True)


def _finish(args, comm, archive, doc, outputs, new, refused, outdir):
    """Write the archive (rank 0) and report its state; the exit code."""
    from ymf import closure
    from ymf.cli import archive2xmf
    if comm.isMaster():
        closure.write(archive, doc, outputs)
        if new:
            archive2xmf(archive, outdir, new)
    comm.barrier()
    from ymf.state import reached
    state, why = reached(dict(doc, outputs=outputs))
    print("\nwrote %s: %d outputs, %d new; ymf_state: %s" % (archive, len(outputs),
                                                            len(new), state))
    if refused:
        print("\nNOT REPRODUCED (--keep-different keeps a differing rerun):")
        for key, found in refused:
            print("  %s\n    %s" % (key, "\n    ".join(found)))
        return 1
    if args.check:
        print("every rerun reproduced its record: same ADR form or dispatch, same "
              "errors, every array bitwise identical")
    # 0 only when every run succeeded and the archive is closed. Which
    # non-zero code means what is still to be defined (PROTEUS_2.0_TASK_LIST).
    if state != "closed":
        print("not closed: " + why)
        return 1
    return 0

