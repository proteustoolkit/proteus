#!/usr/bin/env python
"""Checks of the any-order quadrilateral/hexahedral dof map on one rank's subdomain; run by
test_cube_dofmap.py, serially and under mpiexec.

Environment: SHAPE (quad|hex), PART_TYPE (node|element), NLAYERS, PMAX, OUT (directory for the per-rank
coordinate files). Writes ``result_<tag>.json`` per rank and case to OUT.

Per case (degree p, nodes, local vertex orders as generated or scrambled per rank by a random symmetry of
the square or cube per element):
* the dof map builds (its own check: every element assigns the same point to a shared dof);
* dof counts against the entity counts; owned + ghost layout; subdomain2global is a bijection onto its range;
* p = 1 on the unscrambled mesh: identical to NodalDOFMap (l2g, offsets, subdomain2global); p = 2: the same
  dof points as QuadraticLagrangeCubeDOFMap, and on hexahedra the same numbering;
* C0: the jump across every face with two elements in the subdomain (ghost elements included) of the
  function with generic dof values is round-off; negative control without the orientation rule;
* interpolation of a tensor polynomial of degree p through the space class is exact;
* the global dof -> point table is written for the cross-rank and serial-vs-parallel comparison.
"""
import itertools
import json
import os

from proteus import Comm
comm = Comm.init()

import numpy as np
from proteus import CubeSpaces, FemTools, MeshTools

shape = os.environ.get("SHAPE", "quad")
part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "node")]
layers = int(os.environ.get("NLAYERS", "1"))
out = os.environ.get("OUT", ".")
rank = comm.rank()

if shape == "hex":
    nd = 3
    mesh = MeshTools.HexahedralMesh()
    mesh.generateHexahedralMeshFromRectangularGrid(4, 4, 4, 1.0, 0.8, 1.2)
    pmax = int(os.environ.get("PMAX", "4"))
else:
    nd = 2
    mesh = MeshTools.QuadrilateralMesh()
    mesh.generateQuadrilateralMeshFromRectangularGrid(7, 6, 1.0, 0.8)
    pmax = int(os.environ.get("PMAX", "5"))
mesh.partitionMesh(nLayersOfOverlap=layers, parallelPartitioningType=part)
sub = mesh.subdomainMesh
X = np.asarray(sub.nodeArray)
nodeNumbering = np.asarray(mesh.nodeNumbering_subdomain2global)
CORNERS = 2.0 * np.array(CubeSpaces.CORNERS[nd]) - 1.0


def cell_symmetries():
    """The vertex permutations of the reference cell induced by its 8 (square) or 48 (cube) symmetries."""
    perms = []
    for axes in itertools.permutations(range(nd)):
        for signs in itertools.product((-1.0, 1.0), repeat=nd):
            image = CORNERS[:, list(axes)] * np.array(signs)
            perms.append([int(np.abs(CORNERS - c).sum(axis=1).argmin()) for c in image])
    return np.array(perms)


def generic(x):
    """Dof values that depend only on the dof's point, so every rank assigns the same number."""
    return np.cos(3.0 * x[..., 0] + 1.0) * np.sin(2.0 * x[..., 1] + 0.3) + 0.7 * np.cos(1.7 * x[..., 2] - x[..., 0])


def poly(x, p):
    """A tensor polynomial of degree p in each of the nd coordinates."""
    c = np.array([1.0, -0.7, 0.4])[:nd]
    return (0.3 + x[..., :nd] @ c) ** p + 0.5 * np.prod(x[..., :nd], axis=-1) ** p


def reference_coordinates(elements, x):
    """Reference coordinates of physical points x (n, 3) in elements (n,): the elements are parallelograms
    or parallelepipeds, so the map is affine, x = a + J xi with a the centre and J = sum_v (X_v - a) c_v^T / 2^nd."""
    Xe = X[np.asarray(sub.elementNodesArray)[elements]][:, :, :nd]
    a = Xe.mean(axis=1)
    J = np.einsum("evi,vj->eij", Xe - a[:, None, :], CORNERS) / 2.0 ** nd
    return np.linalg.solve(J, (x[:, :nd] - a)[:, :, None])[:, :, 0]


def face_jump(lfs, l2g, dof):
    """Largest jump of u_h across the faces that have both elements in the subdomain."""
    ebe = np.asarray(sub.elementBoundaryElementsArray)
    faces = np.where(ebe[:, 1] >= 0)[0]
    fn = np.asarray(sub.elementBoundaryNodesArray)[faces]
    w = np.random.RandomState(2).dirichlet(np.ones(fn.shape[1]), size=4)   # points in the (planar) face
    worst = 0.0
    for wk in w:
        x = np.einsum("k,fkc->fc", wk, X[fn])
        sides = []
        for side in (0, 1):
            e = ebe[faces, side]
            xi = reference_coordinates(e, x)
            assert np.abs(xi).max() < 1.0 + 1.0e-12
            sides.append(np.einsum("fi,fi->f", lfs.tabulate(xi, 0).values, dof[l2g[e]]))
        worst = max(worst, np.abs(sides[0] - sides[1]).max())
    return float(worst), int(len(faces))


def run_case(p, nodes, scrambled):
    lfs = FemTools.PolynomialOnCube(nd, p, nodes)
    dm = FemTools.CubeEntityDOFMap(sub, lfs, nd)
    res = dict(rank=rank, shape=shape, p=p, nodes=nodes, scrambled=scrambled,
               nDOF=int(dm.nDOF), nOwned=int(dm.nDOF_subdomain_owned), nAll=int(dm.nDOF_all_processes),
               defect=dm.conformityDefect, offsets=[int(o) for o in dm.dof_offsets_subdomain_owned])
    # counts
    nEdges = mesh.nElementBoundaries_global if nd == 2 else mesh.nEdges_global
    nQuads = mesh.nElements_global if nd == 2 else mesh.nElementBoundaries_global
    res["nAll_expected"] = int(mesh.nNodes_global + (p - 1) * nEdges + (p - 1) ** 2 * nQuads
                               + ((p - 1) ** 3 * mesh.nElements_global if nd == 3 else 0))
    s2g = np.asarray(dm.subdomain2global)
    res["s2g_injective"] = bool(len(np.unique(s2g)) == len(s2g) and s2g.min() >= 0 and s2g.max() < dm.nDOF_all_processes)
    off = dm.dof_offsets_subdomain_owned
    res["owned_first"] = bool((s2g[:dm.nDOF_subdomain_owned] == off[rank] + np.arange(dm.nDOF_subdomain_owned)).all()
                              and dm.nDOF_subdomain_owned == off[rank + 1] - off[rank])
    res["ghosts_not_owned"] = bool(((s2g[dm.nDOF_subdomain_owned:] < off[rank]) | (s2g[dm.nDOF_subdomain_owned:] >= off[rank + 1])).all())
    res["l2g_dtype_ok"] = bool(dm.l2g.dtype == FemTools.cmeshTools.INDEX_DTYPE and s2g.dtype == FemTools.cmeshTools.INDEX_DTYPE)
    # every dof belongs to an element of the subdomain, except the dofs of an edge record that is not an edge:
    # with an element partition and one layer of overlap the hexahedral subdomain mesh can carry one such
    # row, (node, node), in its ghost edges (a partitioner defect, reported; 0 on every other mesh here)
    edgeRows = np.asarray(sub.edgeNodesArray) if nd == 3 else np.zeros((0, 2), int)
    res["junk_edges"] = int((edgeRows[:, 0] == edgeRows[:, 1]).sum())
    res["unused_dofs"] = int(dm.nDOF - len(np.unique(dm.l2g)))
    # generic dof values: C0 across every face
    dof = generic(dm.lagrangeNodesArray)
    res["jump"], res["nFaces2"] = face_jump(lfs, dm.l2g, dof)
    # negative control: without the orientation rule the map is not conforming (p >= 3)
    try:
        bad = FemTools.CubeEntityDOFMap(sub, lfs, nd, orient=False)
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
    edges = np.asarray(sub.elementBoundaryNodesArray if nd == 2 else sub.edgeNodesArray)
    res["edges_local_order_differs_from_global"] = int(((edges[:, 0] < edges[:, 1]) !=
                                                        (nodeNumbering[edges[:, 0]] < nodeNumbering[edges[:, 1]])).sum())
    if nd == 3:
        frames = set()
        moved = 0
        for verts in lfs.space.entity_vertices[2]:
            g = nodeNumbering[en[:, list(verts)]]
            origin = np.argmin(g, axis=1)
            rows = np.arange(len(g))
            first_is_next = g[rows, (origin + 1) % 4] < g[rows, (origin + 3) % 4]
            frames |= set(zip(origin.tolist(), first_is_next.tolist()))
            moved += int(((origin != 0) | ~first_is_next).sum())
        res["element_faces_reoriented"] = moved
        res["face_frames_seen"] = len(frames)     # of the 8 symmetries of the square
    # the space class: interpolate a polynomial of degree p, exactly
    if not scrambled:  # AffineMaps and the mesh's local boundary arrays want the generated vertex order
        space = FemTools.C0_AffinePolynomialOnCube(sub, nd, p, nodes)
        u = FemTools.FiniteElementFunction(space, name="u")
        u.projectFromInterpolationConditions(poly(space.interpolationPoints, p))
        xi = np.random.RandomState(4).uniform(-1.0, 1.0, size=(6, nd))
        xq = np.einsum("qv,evc->eqc", CubeSpaces.vertex_weights(nd, xi), X[en])
        uh = np.einsum("qi,ei->eq", space.getBasisValuesRef(xi), u.dof[space.dofMap.l2g])
        res["interpolation_error"] = float(np.abs(uh - poly(xq, p)).max())
        res["space_nDOF"] = int(space.dim)
        res["space_class"] = type(space).__name__
        if p == 1:
            old = FemTools.NodalDOFMap(sub)
            res["equals_existing_map"] = bool(np.array_equal(np.asarray(old.l2g), dm.l2g)
                                              and np.array_equal(np.asarray(old.subdomain2global), s2g)
                                              and np.array_equal(np.asarray(old.dof_offsets_subdomain_owned), np.asarray(off))
                                              and old.nDOF_all_processes == dm.nDOF_all_processes)
        if p == 2:
            q2 = FemTools.LagrangeOnCubeWithNodalBasis(nd, order=2)
            old = FemTools.QuadraticLagrangeCubeDOFMap(sub, q2, nd)
            a = np.asarray(old.lagrangeNodesArray)
            b = dm.lagrangeNodesArray
            key = lambda x: x[np.lexsort(np.round(x, 9).T[::-1])]
            res["q2_same_points"] = bool(a.shape == b.shape and np.abs(key(a) - key(b)).max() < 1.0e-13
                                         and old.nDOF_all_processes == dm.nDOF_all_processes)
            res["q2_same_numbering"] = bool(np.array_equal(np.asarray(old.l2g), dm.l2g)
                                            and np.array_equal(np.asarray(old.subdomain2global), s2g))
    tag = "%s_p%d_%s_%d_r%d" % (shape, p, nodes, int(scrambled), rank)
    used = np.zeros(dm.nDOF, dtype=bool)
    used[np.unique(dm.l2g)] = True      # leaves out the dofs of an edge record that is not an edge (they have no point)
    np.savez(os.path.join(out, tag + ".npz"), s2g=s2g[used], x=dm.lagrangeNodesArray[used],
             nOwned=dm.nDOF_subdomain_owned, usedOwned=int(used[:dm.nDOF_subdomain_owned].sum()))
    # one file per rank and case: the ranks share stdout, and lines from two of them can interleave
    with open(os.path.join(out, "result_" + tag + ".json"), "w") as fh:
        json.dump(res, fh)


cases = [(p, n) for p in range(1, pmax + 1) for n in ("gll", "equispaced")]
for p, n in cases:
    run_case(p, n, False)
# list every element's vertices through a random symmetry of the cell (each rank its own way, so a ghost
# element is listed differently on the two ranks that hold it); the dof map looks entities up by their vertices
rs = np.random.RandomState(100 + rank)
symmetries = cell_symmetries()
assert len(symmetries) == {2: 8, 3: 48}[nd]
en = sub.elementNodesArray
for eN in range(en.shape[0]):
    en[eN, :] = np.asarray(en[eN])[symmetries[rs.randint(len(symmetries))]]
for p, n in cases:
    if p >= 2:
        run_case(p, n, True)
