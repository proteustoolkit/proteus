"""
The velocities PostProcessingTools produces, held in the FemTools H(div) spaces (C8 round 5).

PostProcessingTools stores a post-processed velocity element by element (RT0 as fluxes or as a + b x, BDM1
in a Lagrange representation), every element with its own copy of each face unknown. `hdiv_vpp_mpi.py` runs
the heterogeneous Poisson problem of test/post_processing unchanged, samples the post-processor's velocity
at the dof points of `Hdiv_AffineOnSimplex` and builds the dof vector of the FemTools space from it. Then,
for each variant:

* the two elements of a face give the same dof (the post-processed normal component is single valued);
* the FemTools field equals the post-processor's own evaluation at random points of every element;
* its face fluxes are the fluxes the model integrates, and |K| div v is the element's source integral
  (to the residual the post-processor leaves);
* on 2 ranks the gathered global dof vector covers every dof, ranks agree where they overlap, and it equals
  the serial vector dof by dof (matched by point and face).

So a post-processed velocity can be carried by one shared unknown per face (RT0) or nd per face (BDM1) without
loss. PostProcessingTools is not modified; the switch is C10b's.
"""
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
HELPER = os.path.join(HERE, "hdiv_vpp_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, variant, order=1, space="c0", overlap=1):
    os.makedirs(str(cwd), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", VPP_VARIANT=variant, VPP_ORDER=str(order), VPP_OVERLAP=str(overlap),
               VPP_SPACE=space, VPP_NN="13", VPP_PROBLEM="smooth")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=900)
    results = [json.loads(line[len("RESULT "):]) for line in out.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, out.stdout[-2000:] + out.stderr[-2000:]
    r = results[0]
    assert r["converged_reason"] > 0, r
    table = np.load(os.path.join(str(cwd), "hdiv_dofs.npz"))
    return r, table["point"], table["value"]


def _check(r, balance_tolerance=1.0e-11):
    scale = max(r["field_scale"], 1.0)
    assert r["family"] == ("bdm1" if "bdm" in r["variant"] else "rt0")
    assert r["nElements"] == 288 and r["nDOF"] == 456 * (2 if r["family"] == "bdm1" else 1), r
    assert r["dofs_covered"] == r["nDOF"], r
    assert r["normal_defect"] < 1.0e-12 * scale, r          # single-valued normal component
    assert r["dofs_with_jump"] == 0, r
    assert r["field_difference"] < 1.0e-12 * scale, r       # the same field
    assert r["flux_difference"] < 1.0e-12 * scale, r        # the model's face fluxes
    assert r["cross_rank_defect"] < 1.0e-12 * scale, r
    # |K| div v = source integral, up to the residual the variant leaves
    assert r["balance"] < balance_tolerance + 2.0 * r["residual"], r


# variant, order of the potential, space of the potential
SERIAL = [("pwl", 1, "c0"), ("pwl", 2, "c0"), ("pwl-bdm", 1, "c0"), ("pwl-bdm", 2, "c0"), ("pwc", 1, "c0"),
          ("sun-rt0", 1, "c0"), ("dg", 1, "dg"), ("dg-bdm", 1, "dg"), ("p1-nc", 1, "nc")]


@pytest.mark.parametrize("variant,order,space", SERIAL, ids=["%s-P%d" % (v, o) for v, o, s in SERIAL])
def test_a_post_processed_velocity_is_a_function_of_the_hdiv_space(tmp_path, variant, order, space):
    r, point, value = _run(1, tmp_path, variant, order, space)
    _check(r)
    assert not np.isnan(value).any()


PARALLEL = [("pwl", 1, "c0", 1), ("pwl", 1, "c0", 0), ("pwl-bdm", 1, "c0", 1), ("pwl-bdm", 1, "c0", 0),
            ("pwl", 2, "c0", 1), ("dg", 1, "dg", 1), ("dg-bdm", 1, "dg", 1), ("p1-nc", 1, "nc", 1), ("p1-nc", 1, "nc", 0)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("variant,order,space,overlap", PARALLEL,
                         ids=["%s-P%d-overlap%d" % (v, o, l) for v, o, s, l in PARALLEL])
def test_two_ranks_give_the_serial_dof_vector(tmp_path, variant, order, space, overlap):
    rs, ps, vs = _run(1, tmp_path / "1", variant, order, space, overlap)
    rp, pp, vp = _run(2, tmp_path / "2", variant, order, space, overlap)
    _check(rs)
    _check(rp)
    assert rp["ranks"] == 2 and rp["nDOF"] == rs["nDOF"]
    # the same dofs, matched by (point, face): the numbering differs, the normals and the values do not
    a, b = np.lexsort(np.round(ps, 9).T[::-1]), np.lexsort(np.round(pp, 9).T[::-1])
    assert np.abs(ps[a] - pp[b]).max() < 1.0e-12
    assert np.abs(vs[a] - vp[b]).max() < 1.0e-9 * max(rs["field_scale"], 1.0), np.abs(vs[a] - vp[b]).max()
