#!/usr/bin/env python
"""Checks of the H(div) face dof map and space (RT0, BDM1) on one rank's subdomain; run by
test_hdiv_dofmap.py, serially and under mpiexec.

Environment: SHAPE (edge|tri|tet), PART_TYPE (node|element), NLAYERS, OUT (directory for the per-rank
tables), RATES (1: also the interpolation errors on this mesh and on the mesh with twice the intervals,
serial only). Prints one line ``RESULT {json}`` per rank and case.

Per case (family, local vertex orders as generated or scrambled per rank):
* the dof map builds (its own check: elements agree on the point of a shared dof and on the face normals);
* dof counts against the face count; owned + ghost layout; subdomain2global is a bijection onto its range;
* H(div): for generic dof values the jump of the normal component across every face with two elements in
  the subdomain (ghost elements included) is round-off, while the tangential component does jump; negative
  control without the sign and ordering rule;
* a field of the space (a + b x for RT0, a linear field for BDM1) is interpolated exactly and its divergence
  is exact, through the space class; element fluxes add up to |K| div v;
* the dofs of that interpolant, with the points that identify each dof, are written for the cross-rank and
  serial-vs-parallel comparison: equal values mean every rank chose the same normal for every face.
"""
import json
import os

from proteus import Comm
comm = Comm.init()

import numpy as np
from proteus import FemTools, HdivSpaces, MeshTools, Quadrature

shape = os.environ.get("SHAPE", "tri")
part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "node")]
layers = int(os.environ.get("NLAYERS", "1"))
out = os.environ.get("OUT", ".")
rank = comm.rank()
nd = {"edge": 1, "tri": 2, "tet": 3}[shape]


def make_mesh(refine=1):
    if shape == "tet":
        m = MeshTools.TetrahedralMesh()
        m.generateTetrahedralMeshFromRectangularGrid(3 * refine + 1, 3 * refine + 1, 3 * refine + 1, 1.0, 1.0, 1.0)
    elif shape == "edge":
        m = MeshTools.EdgeMesh()
        m.generateEdgeMeshFromRectangularGrid(11 * refine + 1, 1.0)
    else:
        m = MeshTools.TriangularMesh()
        m.generateTriangularMeshFromRectangularGrid(6 * refine + 1, 5 * refine + 1, 1.0, 1.0)
    m.partitionMesh(nLayersOfOverlap=layers, parallelPartitioningType=part)
    return m


mesh = make_mesh()
sub = mesh.subdomainMesh
X = np.asarray(sub.nodeArray)
nodeNumbering = np.asarray(mesh.nodeNumbering_subdomain2global)
A0 = np.array([0.3, -0.8, 0.5])[:nd]
B0 = np.array([[0.7, -0.2, 0.4], [0.3, -0.9, 0.1], [-0.5, 0.6, 0.2]])[:nd, :nd]


def in_space(family):
    """A field of the space and its divergence."""
    if family == "rt0":
        return (lambda x: A0 + 0.6 * x[..., :nd]), 0.6 * nd
    return (lambda x: A0 + x[..., :nd] @ B0.T), float(np.trace(B0))


def generic(dm):
    """Dof values that depend only on the dof's point and face, so every rank assigns the same number."""
    x, c = dm.dofPoints, dm.dofFaceCentroids
    return np.cos(3.0 * x[:, 0] + 1.0 + 2.0 * c[:, 1]) + 0.7 * np.sin(1.7 * c[:, 2] - x[:, 1] + 0.4 * c[:, 0])


def jacobians():
    en = np.asarray(sub.elementNodesArray)
    return (X[en[:, 1:]] - X[en[:, :1]])[:, :, :nd].transpose(0, 2, 1)


def evaluate(dm, dof, elements, x):
    """The field with dofs ``dof`` at physical points x (n, 3), one per element of ``elements`` (n,)."""
    space = dm.localFunctionSpace.space
    en = np.asarray(sub.elementNodesArray)[elements]
    J = jacobians()[elements]
    xi = np.linalg.solve(J, (x - X[en[:, 0]])[:, :nd, None])[:, :, 0]
    ref = np.array([space.tabulate(xi[k:k + 1]).values[0] for k in range(len(xi))])   # (n, dim, nd)
    phys = np.einsum("nIi,nki->nkI", J, ref) / np.abs(np.linalg.det(J))[:, None, None]
    return np.einsum("nkI,nk->nI", phys, dm.sign[elements] * dof[dm.l2g[elements]])


def face_jumps(dm, dof):
    """Largest jumps of the normal and of the tangential component across the two-element faces."""
    ebe = np.asarray(sub.elementBoundaryElementsArray)
    faces = np.where(ebe[:, 1] >= 0)[0]
    if len(faces) == 0:
        return 0.0, 0.0, 0
    fn = np.asarray(sub.elementBoundaryNodesArray).reshape(len(ebe), -1)[faces][:, :max(nd, 1)]
    normal = dm.faceAreaNormals[faces][:, :nd]
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    worst_n = worst_t = 0.0
    for wk in np.random.RandomState(2).dirichlet(np.ones(fn.shape[1]), size=4):
        x = np.einsum("k,fkc->fc", wk, X[fn])
        v = [evaluate(dm, dof, ebe[faces, side], x) for side in (0, 1)]
        jump = v[0] - v[1]
        jn = np.einsum("fd,fd->f", jump, normal)
        worst_n = max(worst_n, float(np.abs(jn).max()))
        worst_t = max(worst_t, float(np.abs(jump - jn[:, None] * normal).max()))
    return worst_n, worst_t, int(len(faces))


def run_case(family, scrambled):
    lfs = FemTools.HdivOnSimplex(nd, family)
    dm = FemTools.FaceDOFMap(sub, lfs, nd)
    dpf = len(lfs.space.face_dofs[0])
    res = dict(rank=rank, shape=shape, family=family, scrambled=scrambled, nDOF=int(dm.nDOF),
               nOwned=int(dm.nDOF_subdomain_owned), nAll=int(dm.nDOF_all_processes), defect=dm.conformityDefect,
               offsets=[int(o) for o in dm.dof_offsets_subdomain_owned])
    res["nAll_expected"] = int(dpf * mesh.nElementBoundaries_global)
    s2g = np.asarray(dm.subdomain2global)
    res["s2g_injective"] = bool(len(np.unique(s2g)) == len(s2g) and s2g.min() >= 0 and s2g.max() < dm.nDOF_all_processes)
    off = dm.dof_offsets_subdomain_owned
    res["owned_first"] = bool((s2g[:dm.nDOF_subdomain_owned] == off[rank] + np.arange(dm.nDOF_subdomain_owned)).all()
                              and dm.nDOF_subdomain_owned == off[rank + 1] - off[rank])
    res["ghosts_not_owned"] = bool(((s2g[dm.nDOF_subdomain_owned:] < off[rank]) | (s2g[dm.nDOF_subdomain_owned:] >= off[rank + 1])).all())
    res["l2g_dtype_ok"] = bool(dm.l2g.dtype == FemTools.cmeshTools.INDEX_DTYPE and s2g.dtype == FemTools.cmeshTools.INDEX_DTYPE)
    res["unused_dofs"] = int(dm.nDOF - len(np.unique(dm.l2g)))
    # generic dofs: the normal component is continuous, the tangential one is not
    res["normal_jump"], res["tangential_jump"], res["nFaces2"] = face_jumps(dm, generic(dm))
    bad = FemTools.FaceDOFMap(sub, lfs, nd, orient=False)
    res["normal_jump_unoriented"] = face_jumps(bad, generic(bad))[0]
    res["negative_signs"] = int((dm.sign < 0).sum())
    # how often subdomain-local vertex numbers would give a face the opposite normal, or another dof order
    en = np.asarray(sub.elementNodesArray)
    differs = 0
    for verts in lfs.space.faces:
        loc = en[:, list(verts)]
        differs += int((np.argsort(loc, axis=1) != np.argsort(nodeNumbering[loc], axis=1)).any(axis=1).sum())
    res["element_faces_local_order_differs_from_global"] = differs
    # a field of the space: its dofs, computed element by element, agree; saved for the cross-rank comparison
    field, div = in_space(family)
    values = field(dm.elementDofPoints)
    normals = dm.elementOutwardNormals[:, lfs.space.dof_face, :]
    local = dm.sign * np.einsum("eid,eid->ei", normals, values)
    dof = np.zeros(dm.nDOF)
    dof[dm.l2g] = local
    res["field_dof_defect"] = float(np.abs(dof[dm.l2g] - local).max())
    xq = np.einsum("v,evc->ec", np.full(nd + 1, 1.0 / (nd + 1)), X[en])
    res["field_error"] = float(np.abs(evaluate(dm, dof, np.arange(en.shape[0]), xq) - field(xq)).max())
    if not scrambled:  # AffineMaps wants the generated vertex order
        space = FemTools.Hdiv_AffineOnSimplex(sub, nd, family)
        d2, defect, known = space.dofsFromElementValues(field(space.dofMap.elementDofPoints))
        xi = np.random.RandomState(4).dirichlet(np.ones(nd + 1), size=6)[:, 1:]
        xs = np.einsum("qv,evc->eqc", np.column_stack([1.0 - xi.sum(axis=1), xi]), X[en])
        res["space_class"] = type(space).__name__
        res["space_nDOF"] = int(space.dim)
        res["interpolation_defect"] = defect
        res["interpolation_error"] = float(np.abs(space.evaluate(d2, xi) - field(xs)).max())
        res["divergence_error"] = float(np.abs(space.divergence(d2) - div).max())
        volume = np.abs(np.linalg.det(space.elementJacobians())) / {1: 1.0, 2: 2.0, 3: 6.0}[nd]
        res["flux_balance"] = float(np.abs(space.elementFluxes(d2).sum(axis=1) - volume * space.divergence(d2)).max())
        res["all_known"] = bool(known.all())
        res["same_as_bare_map"] = bool(np.array_equal(space.dofMap.l2g, dm.l2g) and np.abs(d2 - dof).max() < 1.0e-13)
        u = FemTools.FiniteElementFunction(space, name="v")
        res["function_dofs"] = int(u.dof.shape[0])
    tag = "%s_%s_%d_r%d" % (shape, family, int(scrambled), rank)
    used = np.zeros(dm.nDOF, dtype=bool)
    used[np.unique(dm.l2g)] = True
    np.savez(os.path.join(out, tag + ".npz"), s2g=s2g[used], x=np.hstack([dm.dofPoints, dm.dofFaceCentroids])[used],
             dof=dof[used], nOwned=dm.nDOF_subdomain_owned)
    print("RESULT " + json.dumps(res), flush=True)


def interpolation_errors(family, m):
    """L2 errors of the interpolant of a smooth field and of its divergence on mesh m (serial)."""
    s = m.subdomainMesh
    space = FemTools.Hdiv_AffineOnSimplex(s, nd, family)
    k = 2.0
    f = lambda x: np.stack([np.sin(k * x[..., i] + 0.3 * i) * np.cos(k * x[..., (i + 1) % nd] - 0.2) for i in range(nd)], axis=-1)
    divf = lambda x: sum(k * np.cos(k * x[..., i] + 0.3 * i) * np.cos(k * x[..., (i + 1) % nd] - 0.2)
                         - (k * np.sin(k * x[..., i] + 0.3 * i) * np.sin(k * x[..., (i + 1) % nd] - 0.2) if nd == 1 else 0.0)
                         for i in range(nd))
    dof = space.interpolate(f)
    quad = Quadrature.SimplexGaussQuadrature(nd, 6)
    xi = np.array([q[:nd] for q in quad.points])
    w = np.array(quad.weights)
    Xs, en = np.asarray(s.nodeArray), np.asarray(s.elementNodesArray)
    xq = np.einsum("qv,evc->eqc", np.column_stack([1.0 - xi.sum(axis=1), xi]), Xs[en])
    detJ = np.abs(np.linalg.det(space.elementJacobians()))
    e2 = np.einsum("q,e,eqd->", w, detJ, (space.evaluate(dof, xi) - f(xq)) ** 2)
    d2 = np.einsum("q,e,eq->", w, detJ, (space.divergence(dof)[:, None] - divf(xq)) ** 2)
    return float(e2 ** 0.5), float(d2 ** 0.5)


for family in ("rt0", "bdm1"):
    run_case(family, False)
if os.environ.get("RATES") == "1" and comm.size() == 1:
    fine = make_mesh(2)
    for family in ("rt0", "bdm1"):
        c, f = interpolation_errors(family, mesh), interpolation_errors(family, fine)
        print("RATES " + json.dumps(dict(shape=shape, family=family, L2=[c[0], f[0]], div=[c[1], f[1]],
                                         L2_rate=float(np.log2(c[0] / f[0])), div_rate=float(np.log2(c[1] / f[1])))), flush=True)
# scramble the local vertex order of every element (each rank its own way, so a ghost element is listed
# differently on the two ranks that hold it); the dof map looks faces up by their vertices
rs = np.random.RandomState(100 + rank)
en = sub.elementNodesArray
for eN in range(en.shape[0]):
    en[eN, :] = np.asarray(en[eN])[rs.permutation(nd + 1)]
for family in ("rt0", "bdm1"):
    run_case(family, True)
