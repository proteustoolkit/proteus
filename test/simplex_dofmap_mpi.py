#!/usr/bin/env python
"""Checks of the any-order simplex dof map on one rank's subdomain; run by test_simplex_dofmap.py,
serially and under mpiexec.

Environment: SHAPE (edge|tri|tet), PART_TYPE (node|element), NLAYERS, PMAX, OUT (directory for the per-rank
coordinate files). Prints one line ``RESULT {json}`` per rank and case.

Per case (degree p, family, nodes, local vertex orders as generated or scrambled per rank):
* the dof map builds (its own check: every element assigns the same point to a shared dof);
* dof counts against the entity counts; owned + ghost layout; subdomain2global is a bijection onto its range;
* p = 1, 2 on the unscrambled mesh: identical to NodalDOFMap / QuadraticLagrangeDOFMap (l2g, offsets,
  subdomain2global, points), i.e. the closed-form numbering equals the one partitioning.cpp gathers;
* C0: the jump across every face with two elements in the subdomain (ghost elements included) of the
  function with generic dof values is round-off; negative control without the orientation rule;
* interpolation of a degree-p polynomial through the space class is exact;
* the global dof -> point table is written for the cross-rank and serial-vs-parallel comparison.
"""
import json
import os

from proteus import Comm
comm = Comm.init()

import numpy as np
from proteus import FemTools, MeshTools

shape = os.environ.get("SHAPE", "tri")
part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "node")]
layers = int(os.environ.get("NLAYERS", "1"))
out = os.environ.get("OUT", ".")
rank = comm.rank()

if shape == "tet":
    nd = 3
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateTetrahedralMeshFromRectangularGrid(4, 4, 4, 1.0, 1.0, 1.0)
    pmax = int(os.environ.get("PMAX", "4"))
elif shape == "edge":
    nd = 1
    mesh = MeshTools.EdgeMesh()
    mesh.generateEdgeMeshFromRectangularGrid(12, 1.0)
    pmax = int(os.environ.get("PMAX", "5"))
else:
    nd = 2
    mesh = MeshTools.TriangularMesh()
    mesh.generateTriangularMeshFromRectangularGrid(7, 6, 1.0, 1.0)
    pmax = int(os.environ.get("PMAX", "5"))
mesh.partitionMesh(nLayersOfOverlap=layers, parallelPartitioningType=part)
sub = mesh.subdomainMesh
X = np.asarray(sub.nodeArray)
nodeNumbering = np.asarray(mesh.nodeNumbering_subdomain2global)


def generic(x):
    """Dof values that depend only on the dof's point, so every rank assigns the same number."""
    return np.cos(3.0 * x[..., 0] + 1.0) * np.sin(2.0 * x[..., 1] + 0.3) + 0.7 * np.cos(1.7 * x[..., 2] - x[..., 0])


def poly(x, p):
    """A polynomial of total degree p in the nd coordinates."""
    c = np.array([1.0, -0.7, 0.4])[:nd]
    return (0.3 + x[..., :nd] @ c) ** p + 0.5 * x[..., 0] ** p


def reference_coordinates(elements, x):
    """Reference coordinates of physical points x (n, 3) in elements (n,) (affine maps)."""
    en = np.asarray(sub.elementNodesArray)[elements]
    A = (X[en[:, 1:]] - X[en[:, :1]])[:, :, :nd].transpose(0, 2, 1)   # columns are the edge vectors
    return np.linalg.solve(A, (x - X[en[:, 0]])[:, :nd, None])[:, :, 0]


def face_jump(lfs, l2g, dof):
    """Largest jump of u_h across the faces that have both elements in the subdomain."""
    ebe = np.asarray(sub.elementBoundaryElementsArray)
    faces = np.where(ebe[:, 1] >= 0)[0]
    fn = np.asarray(sub.elementBoundaryNodesArray)[faces]
    w = np.random.RandomState(2).dirichlet(np.ones(nd), size=4)       # points on the face
    worst = 0.0
    for wk in w:
        x = np.einsum("k,fkc->fc", wk, X[fn])
        sides = []
        for side in (0, 1):
            e = ebe[faces, side]
            xi = reference_coordinates(e, x)
            sides.append(np.einsum("fi,fi->f", lfs.tabulate(xi, 0).values, dof[l2g[e]]))
        worst = max(worst, np.abs(sides[0] - sides[1]).max())
    return float(worst), int(len(faces))


def run_case(p, family, nodes, scrambled):
    lfs = FemTools.PolynomialOnSimplex(nd, p, family, nodes)
    dm = FemTools.SimplexEntityDOFMap(sub, lfs, nd)
    res = dict(rank=rank, shape=shape, p=p, family=family, nodes=nodes, scrambled=scrambled,
               nDOF=int(dm.nDOF), nOwned=int(dm.nDOF_subdomain_owned), nAll=int(dm.nDOF_all_processes),
               defect=dm.conformityDefect, offsets=[int(o) for o in dm.dof_offsets_subdomain_owned])
    # counts
    nEdges = {1: 0, 2: mesh.nElementBoundaries_global, 3: mesh.nEdges_global}[nd]  # 1D: interior dofs are the cell's
    nFaces = mesh.nElementBoundaries_global if nd == 3 else 0
    dc = len(lfs.space.entity_dofs[nd][0])
    res["nAll_expected"] = int(mesh.nNodes_global + (p - 1) * nEdges + ((p - 1) * (p - 2) // 2) * nFaces
                               + dc * mesh.nElements_global)
    s2g = np.asarray(dm.subdomain2global)
    res["s2g_injective"] = bool(len(np.unique(s2g)) == len(s2g) and s2g.min() >= 0 and s2g.max() < dm.nDOF_all_processes)
    off = dm.dof_offsets_subdomain_owned
    res["owned_first"] = bool((s2g[:dm.nDOF_subdomain_owned] == off[rank] + np.arange(dm.nDOF_subdomain_owned)).all()
                              and dm.nDOF_subdomain_owned == off[rank + 1] - off[rank])
    res["ghosts_not_owned"] = bool(((s2g[dm.nDOF_subdomain_owned:] < off[rank]) | (s2g[dm.nDOF_subdomain_owned:] >= off[rank + 1])).all())
    res["l2g_dtype_ok"] = bool(dm.l2g.dtype == FemTools.cmeshTools.INDEX_DTYPE and s2g.dtype == FemTools.cmeshTools.INDEX_DTYPE)
    res["all_dofs_used"] = bool(len(np.unique(dm.l2g)) == dm.nDOF)
    # generic dof values: C0 across every face
    dof = generic(dm.lagrangeNodesArray)
    res["jump"], res["nFaces2"] = face_jump(lfs, dm.l2g, dof)
    # negative control: without the orientation rule the map is not conforming (p >= 3)
    try:
        bad = FemTools.SimplexEntityDOFMap(sub, lfs, nd, orient=False)
        res["defect_unoriented"] = bad.conformityDefect
        res["jump_unoriented"] = face_jump(lfs, bad.l2g, generic(bad.lagrangeNodesArray))[0]
    except Exception as e:  # pragma: no cover
        res["defect_unoriented"] = repr(e)
    # how often the local frame of an entity is not its global frame, and how often the subdomain-local
    # vertex numbers would orient an edge differently from the global ones
    en = np.asarray(sub.elementNodesArray)
    flips = 0
    for a, b in lfs.space.entity_vertices[1]:
        flips += int((nodeNumbering[en[:, a]] > nodeNumbering[en[:, b]]).sum())
    res["element_edges_reversed"] = flips
    edges = np.asarray(sub.elementBoundaryNodesArray if nd == 2 else sub.edgeNodesArray if nd == 3 else en)
    res["edges_local_order_differs_from_global"] = int(((edges[:, 0] < edges[:, 1]) !=
                                                        (nodeNumbering[edges[:, 0]] < nodeNumbering[edges[:, 1]])).sum())
    if nd == 3:
        fperm = 0
        for verts in lfs.space.entity_vertices[2]:
            g = nodeNumbering[en[:, list(verts)]]
            fperm += int(((np.argsort(g, axis=1) != np.arange(3)).any(axis=1)).sum())
        res["element_faces_permuted"] = fperm
    # the space class: interpolate a polynomial of degree p, exactly
    if not scrambled:  # AffineMaps and the mesh's local boundary arrays want the generated vertex order
        space = FemTools.C0_AffinePolynomialOnSimplex(sub, nd, p, family, nodes)
        u = FemTools.FiniteElementFunction(space, name="u")
        u.projectFromInterpolationConditions(poly(space.interpolationPoints, p))
        xi = np.random.RandomState(4).dirichlet(np.ones(nd + 1), size=6)[:, 1:]
        en_ = np.asarray(sub.elementNodesArray)
        bary = np.column_stack([1.0 - xi.sum(axis=1), xi])
        xq = np.einsum("qv,evc->eqc", bary, X[en_])
        uh = np.einsum("qi,ei->eq", space.getBasisValuesRef(xi), u.dof[space.dofMap.l2g])
        res["interpolation_error"] = float(np.abs(uh - poly(xq, p)).max())
        res["space_nDOF"] = int(space.dim)
        if family == "bernstein" and p >= 2:
            spaceB = FemTools.C0_AffinePolynomialOnSimplex(sub, nd, p, family, nodes, bernsteinOperator=True)
            uB = FemTools.FiniteElementFunction(spaceB, name="uB")
            uB.projectFromInterpolationConditions(poly(spaceB.interpolationPoints, p))
            uhB = np.einsum("qi,ei->eq", spaceB.getBasisValuesRef(xi), uB.dof[spaceB.dofMap.l2g])
            res["bernstein_operator_error"] = float(np.abs(uhB - poly(xq, p)).max())
        # p = 1, 2: the existing maps
        if family == "lagrange" and p <= 2:
            if p == 1:
                old = FemTools.NodalDOFMap(sub)
            else:
                old = FemTools.QuadraticLagrangeDOFMap(sub, FemTools.QuadraticOnSimplexWithNodalBasis(nd), nd)
            same = (np.array_equal(np.asarray(old.l2g), dm.l2g)
                    and np.array_equal(np.asarray(old.subdomain2global), s2g)
                    and np.array_equal(np.asarray(old.dof_offsets_subdomain_owned), np.asarray(off))
                    and old.nDOF_all_processes == dm.nDOF_all_processes
                    and old.max_dof_neighbors == dm.max_dof_neighbors)
            if p == 2:
                same = same and np.abs(old.lagrangeNodesArray - dm.lagrangeNodesArray).max() < 1.0e-14
            res["equals_existing_map"] = bool(same)
    tag = "%s_p%d_%s_%s_%d_r%d" % (shape, p, family, nodes, int(scrambled), rank)
    np.savez(os.path.join(out, tag + ".npz"), s2g=s2g, x=dm.lagrangeNodesArray, nOwned=dm.nDOF_subdomain_owned)
    print("RESULT " + json.dumps(res), flush=True)


cases = [(p, f, n) for p in range(1, pmax + 1) for (f, n) in (("lagrange", "gll"), ("lagrange", "equispaced"), ("bernstein", "gll"))]
for p, f, n in cases:
    run_case(p, f, n, False)
# scramble the local vertex order of every element (each rank its own way, so a ghost element is listed
# differently on the two ranks that hold it); the dof map looks entities up by their vertices
rs = np.random.RandomState(100 + rank)
en = sub.elementNodesArray
for eN in range(en.shape[0]):
    en[eN, :] = en[eN, rs.permutation(nd + 1)]
for p, f, n in cases:
    if p >= 2:
        run_case(p, f, n, True)
