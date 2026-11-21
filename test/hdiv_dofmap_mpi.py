#!/usr/bin/env python
"""Checks of the H(div) face dof map and space (RT0, BDM1) on one rank's subdomain; run by
test_hdiv_dofmap.py, serially and under mpiexec.

Environment: SHAPE (edge|tri|tet|quad|hex), PART_TYPE (node|element), NLAYERS, FAMILIES (comma list, default all those
defined for the shape), OUT (directory for the per-rank tables), RATES (1: also the interpolation errors on this mesh and on the mesh with twice the intervals,
serial only). Writes ``result_<tag>.json`` per rank and case to OUT.

Per case (family, local vertex orders as generated or scrambled per rank):
* the dof map builds (its own check: elements agree on the point of a shared dof and on the face normals);
* dof counts against the face count; owned + ghost layout; subdomain2global is a bijection onto its range;
* H(div): for generic dof values the jump of the normal component across every face with two elements in
  the subdomain (ghost elements included) is round-off, while the tangential component does jump; negative
  control without the sign and ordering rule;
* a field of the space (a + b x for RT0, a linear field for BDM1, a + B x + (c.x) x for RT1, a quadratic field
  for BDM2) is interpolated exactly and its divergence is exact, through the space class; element fluxes add
  up to the integral of div v over the element;
* the dofs of that interpolant, with the points that identify each dof, are written for the cross-rank and
  serial-vs-parallel comparison: equal values mean every rank chose the same normal for every face.
"""
import itertools
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
nd = {"edge": 1, "tri": 2, "tet": 3, "quad": 2, "hex": 3}[shape]
cube = shape in ("quad", "hex")
ALL_FAMILIES = ("rt0",) if cube else ("rt0", "bdm1") if nd == 1 else ("rt0", "bdm1", "rt1", "bdm2")
# the cube meshes are sheared (an affine image of the grid), so every element is a parallelogram / parallelepiped
SHEAR = np.array([[1.0, 0.3, 0.1], [0.2, 0.9, 0.15], [0.05, 0.1, 1.1]])[:nd, :nd]
LocalSpace = FemTools.HdivOnCube if cube else FemTools.HdivOnSimplex
DOFMap = FemTools.CubeFaceDOFMap if cube else FemTools.FaceDOFMap
Space = FemTools.Hdiv_AffineOnCube if cube else FemTools.Hdiv_AffineOnSimplex
FAMILIES = [f for f in os.environ.get("FAMILIES", ",".join(ALL_FAMILIES)).split(",") if f in ALL_FAMILIES]


def make_mesh(refine=1):
    if shape == "hex":
        m = MeshTools.HexahedralMesh()
        m.generateHexahedralMeshFromRectangularGrid(3 * refine + 1, 3 * refine + 1, 3 * refine + 1, 1.0, 0.8, 1.2)
    elif shape == "quad":
        m = MeshTools.QuadrilateralMesh()
        m.generateQuadrilateralMeshFromRectangularGrid(6 * refine + 1, 5 * refine + 1, 1.0, 0.8)
    elif shape == "tet":
        m = MeshTools.TetrahedralMesh()
        m.generateTetrahedralMeshFromRectangularGrid(3 * refine + 1, 3 * refine + 1, 3 * refine + 1, 1.0, 1.0, 1.0)
    elif shape == "edge":
        m = MeshTools.EdgeMesh()
        m.generateEdgeMeshFromRectangularGrid(11 * refine + 1, 1.0)
    else:
        m = MeshTools.TriangularMesh()
        m.generateTriangularMeshFromRectangularGrid(6 * refine + 1, 5 * refine + 1, 1.0, 1.0)
    if cube:
        m.nodeArray[:, :nd] = m.nodeArray[:, :nd] @ SHEAR.T
    m.partitionMesh(nLayersOfOverlap=layers, parallelPartitioningType=part)
    return m


mesh = make_mesh()
sub = mesh.subdomainMesh
X = np.asarray(sub.nodeArray)
nodeNumbering = np.asarray(mesh.nodeNumbering_subdomain2global)
A0 = np.array([0.3, -0.8, 0.5])[:nd]
B0 = np.array([[0.7, -0.2, 0.4], [0.3, -0.9, 0.1], [-0.5, 0.6, 0.2]])[:nd, :nd]


# fixed coefficients of the test fields
C0 = np.array([0.4, -0.7, 0.3])[:nd]
S0 = np.array([[[0.5, 0.2, -0.3], [0.2, -0.4, 0.1], [-0.3, 0.1, 0.6]],
               [[-0.2, 0.3, 0.4], [0.3, 0.7, -0.5], [0.4, -0.5, 0.1]],
               [[0.1, -0.6, 0.2], [-0.6, 0.3, 0.5], [0.2, 0.5, -0.4]]])[:nd, :nd, :nd]


def in_space(family):
    """A field of the space and its divergence, both as functions of x (.., 3)."""
    if cube:     # A w(A^{-1} x) / |det A| with w = a + b * x_c the Cartesian RT0 field and A the shear of the grid
        Ainv, detA = np.linalg.inv(SHEAR), abs(np.linalg.det(SHEAR))
        b = np.array([0.6, -0.4, 0.9])[:nd]
        return ((lambda x: (A0 + b * (x[..., :nd] @ Ainv.T)) @ SHEAR.T / detA), (lambda x: float(b.sum()) / detA + 0 * x[..., 0]))
    if family == "rt0":
        return (lambda x: A0 + 0.6 * x[..., :nd]), (lambda x: 0.6 * nd + 0 * x[..., 0])
    if family == "bdm1":
        return (lambda x: A0 + x[..., :nd] @ B0.T), (lambda x: float(np.trace(B0)) + 0 * x[..., 0])
    if family == "rt1":      # a + B x + (c.x) x
        return ((lambda x: A0 + x[..., :nd] @ B0.T + (x[..., :nd] @ C0)[..., None] * x[..., :nd]),
                (lambda x: float(np.trace(B0)) + (nd + 1) * (x[..., :nd] @ C0)))
    # bdm2: a + B x + the quadratic (x^T S_i x)_i
    quad = lambda x: np.einsum("...j,ijk,...k->...i", x[..., :nd], S0, x[..., :nd])
    return ((lambda x: A0 + x[..., :nd] @ B0.T + quad(x)),
            (lambda x: float(np.trace(B0)) + 2.0 * np.einsum("ijk,...k,ij->...", S0, x[..., :nd], np.eye(nd))))


def generic(dm):
    """Dof values that depend only on the dof's point and face, so every rank assigns the same number."""
    x, c = dm.dofPoints, dm.dofFaceCentroids
    return (np.cos(3.0 * x[:, 0] + 1.0 + 2.0 * c[:, 1]) + 0.7 * np.sin(1.7 * c[:, 2] - x[:, 1] + 0.4 * c[:, 0])
            + 0.3 * (dm.dofInteriorIndex + 1.0) * np.cos(2.0 * c[:, 0] + c[:, 1]))


def jacobians(mesh_=None):
    """J, origin with x = origin + J xi over the element's reference cell."""
    mesh_ = sub if mesh_ is None else mesh_
    en, Xm = np.asarray(mesh_.elementNodesArray), np.asarray(mesh_.nodeArray)
    if cube:
        J, centre = FemTools.cubeElementJacobians(Xm, en, nd)
        return J, centre[:, :nd]
    return (Xm[en[:, 1:]] - Xm[en[:, :1]])[:, :, :nd].transpose(0, 2, 1), Xm[en[:, 0]][:, :nd]


def evaluate(dm, dof, elements, x):
    """The field with dofs ``dof`` at physical points x (n, 3), one per element of ``elements`` (n,)."""
    space = dm.localFunctionSpace.space
    J, origin = jacobians()
    J, origin = J[elements], origin[elements]
    xi = np.linalg.solve(J, (x[:, :nd] - origin)[:, :, None])[:, :, 0]
    ref = np.array([space.tabulate(xi[k:k + 1]).values[0] for k in range(len(xi))])   # (n, dim, nd)
    phys = np.einsum("nIi,nki->nkI", J, ref) / np.abs(np.linalg.det(J))[:, None, None]
    return np.einsum("nkI,nk->nI", phys, dm.sign[elements] * dof[dm.l2g[elements]])


def ref_rule(order=10):
    """Quadrature points (nq, nd) and weights on the reference cell (the simplex: weights sum to 1/nd!; the cube
    [-1,1]^nd: to 2^nd)."""
    if cube:
        g, w = np.polynomial.legendre.leggauss(6)
        xi = np.array(list(itertools.product(g, repeat=nd)))
        return xi, np.prod(np.array(list(itertools.product(w, repeat=nd))), axis=1)
    quad = Quadrature.SimplexGaussQuadrature(nd, order)
    return np.array([q[:nd] for q in quad.points]), np.array(quad.weights)


def ref_samples(n, seed):
    """n random points of the reference cell."""
    rs = np.random.RandomState(seed)
    return rs.uniform(-1.0, 1.0, size=(n, nd)) if cube else rs.dirichlet(np.ones(nd + 1), size=n)[:, 1:]


def phys_points(xi, mesh_=None):
    """(nE, nq, 3) physical points of the reference points xi in every element."""
    J, origin = jacobians(mesh_)
    x = np.zeros((J.shape[0], len(xi), 3))
    x[:, :, :nd] = origin[:, None, :] + np.einsum("eIi,qi->eqI", J, xi)
    return x


def field_dofs(dm, field):
    """Dof vector of the interpolant of ``field`` computed here, element by element and independently of
    ``Hdiv_AffineOnSimplex.interpolate``: face nodes: (area normal . value) times the sign; interior dofs: the
    reference-frame moments of |det J| J^{-1} field at the interior rule's points."""
    space = dm.localFunctionSpace.space
    en = np.asarray(sub.elementNodesArray)
    J, origin = jacobians()
    nf = space.n_face_dofs
    normals = dm.elementOutwardNormals[:, space.dof_face[:nf], :]
    local = np.zeros(dm.l2g.shape)
    local[:, :nf] = dm.sign[:, :nf] * np.einsum("eid,eid->ei", normals, field(dm.elementDofPoints[:, :nf, :])[..., :nd])
    if space.n_interior_dofs:
        xi, w = space.interior_points
        xq = origin[:, None, :] + np.einsum("eIi,qi->eqI", J, xi)
        xq3 = np.concatenate([xq, np.zeros(xq.shape[:2] + (3 - nd,))], axis=-1)
        fhat = np.abs(np.linalg.det(J))[:, None, None] * np.linalg.solve(J[:, None], field(xq3)[..., None])[..., 0]
        local[:, nf:] = space.dofs_from_values(np.zeros((en.shape[0], space.dim, nd)), interior=fhat)[:, nf:]
    dof = np.zeros(dm.nDOF)
    dof[dm.l2g] = local
    return dof, float(np.abs(dof[dm.l2g] - local).max())


def face_jumps(dm, dof):
    """Largest jumps of the normal and of the tangential component across the two-element faces."""
    ebe = np.asarray(sub.elementBoundaryElementsArray)
    faces = np.where(ebe[:, 1] >= 0)[0]
    if len(faces) == 0:
        return 0.0, 0.0, 0
    fn = np.asarray(sub.elementBoundaryNodesArray).reshape(len(ebe), -1)[faces][:, :(4 if shape == 'hex' else max(nd, 1))]
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
    lfs = LocalSpace(nd, family)
    dm = DOFMap(sub, lfs, nd)
    dpf = len(lfs.space.face_dofs[0])
    res = dict(rank=rank, shape=shape, family=family, scrambled=scrambled, nDOF=int(dm.nDOF),
               nOwned=int(dm.nDOF_subdomain_owned), nAll=int(dm.nDOF_all_processes), defect=dm.conformityDefect,
               offsets=[int(o) for o in dm.dof_offsets_subdomain_owned])
    res["nAll_expected"] = int(dpf * mesh.nElementBoundaries_global + lfs.space.n_interior_dofs * mesh.nElements_global)
    res["dim"] = int(lfs.space.dim)
    s2g = np.asarray(dm.subdomain2global)
    res["s2g_injective"] = bool(len(np.unique(s2g)) == len(s2g) and s2g.min() >= 0 and s2g.max() < dm.nDOF_all_processes)
    off = dm.dof_offsets_subdomain_owned
    res["owned_first"] = bool((s2g[:dm.nDOF_subdomain_owned] == off[rank] + np.arange(dm.nDOF_subdomain_owned)).all()
                              and dm.nDOF_subdomain_owned == off[rank + 1] - off[rank])
    res["interior_dofs_per_element"] = int(lfs.space.n_interior_dofs)
    res["ghosts_not_owned"] = bool(((s2g[dm.nDOF_subdomain_owned:] < off[rank]) | (s2g[dm.nDOF_subdomain_owned:] >= off[rank + 1])).all())
    res["l2g_dtype_ok"] = bool(dm.l2g.dtype == FemTools.cmeshTools.INDEX_DTYPE and s2g.dtype == FemTools.cmeshTools.INDEX_DTYPE)
    res["unused_dofs"] = int(dm.nDOF - len(np.unique(dm.l2g)))
    # generic dofs: the normal component is continuous, the tangential one is not
    res["normal_jump"], res["tangential_jump"], res["nFaces2"] = face_jumps(dm, generic(dm))
    bad = DOFMap(sub, lfs, nd, orient=False)
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
    dof, res["field_dof_defect"] = field_dofs(dm, field)
    xq = X[en].mean(axis=1)
    res["field_error"] = float(np.abs(evaluate(dm, dof, np.arange(en.shape[0]), xq) - field(xq)).max())
    if not scrambled:  # AffineMaps wants the generated vertex order
        space = Space(sub, nd, family)
        d2, defect, known = space.interpolate(field, returnDefect=True)
        xi = ref_samples(6, 4)
        xs = phys_points(xi)
        res["space_class"] = type(space).__name__
        res["space_nDOF"] = int(space.dim)
        res["interpolation_defect"] = defect
        res["interpolation_error"] = float(np.abs(space.evaluate(d2, xi) - field(xs)).max())
        res["divergence_error"] = float(np.abs(space.divergence(d2, xi) - div(xs)).max())
        xiq, wq = ref_rule()
        integral = np.abs(np.linalg.det(space.elementJacobians())) * np.einsum("q,eq->e", wq, space.divergence(d2, xiq))
        res["flux_balance"] = float(np.abs(space.elementFluxes(d2).sum(axis=1) - integral).max())
        res["all_known"] = bool(known.all())
        res["same_as_bare_map"] = bool(np.array_equal(space.dofMap.l2g, dm.l2g) and np.abs(d2 - dof).max() < 1.0e-13)
        u = FemTools.FiniteElementFunction(space, name="v")
        res["function_dofs"] = int(u.dof.shape[0])
    tag = "%s_%s_%d_r%d" % (shape, family, int(scrambled), rank)
    used = np.zeros(dm.nDOF, dtype=bool)
    used[np.unique(dm.l2g)] = True
    saved = dof.copy()
    if scrambled:       # an interior dof is tied to its element's vertex order, which this scrambling changes per rank
        saved[dm.dofInteriorIndex >= 0] = 0.0
    key = np.hstack([dm.dofPoints, dm.dofFaceCentroids, np.maximum(dm.dofInteriorIndex, 0)[:, None] + 0.0])
    np.savez(os.path.join(out, tag + ".npz"), s2g=s2g[used], x=key[used], dof=saved[used], nOwned=dm.nDOF_subdomain_owned)
    # one file per rank and case: the ranks share stdout, and lines from two of them can interleave
    with open(os.path.join(out, "result_" + tag + ".json"), "w") as fh:
        json.dump(res, fh)


def interpolation_errors(family, m):
    """L2 errors of the interpolant of a smooth field and of its divergence on mesh m (serial)."""
    s = m.subdomainMesh
    space = Space(s, nd, family)
    k = 2.0
    f = lambda x: np.stack([np.sin(k * x[..., i] + 0.3 * i) * np.cos(k * x[..., (i + 1) % nd] - 0.2) for i in range(nd)], axis=-1)
    divf = lambda x: sum(k * np.cos(k * x[..., i] + 0.3 * i) * np.cos(k * x[..., (i + 1) % nd] - 0.2)
                         - (k * np.sin(k * x[..., i] + 0.3 * i) * np.sin(k * x[..., (i + 1) % nd] - 0.2) if nd == 1 else 0.0)
                         for i in range(nd))
    dof = space.interpolate(f)
    xi, w = ref_rule()
    xq = phys_points(xi, s)
    detJ = np.abs(np.linalg.det(space.elementJacobians()))
    e2 = np.einsum("q,e,eqd->", w, detJ, (space.evaluate(dof, xi) - f(xq)) ** 2)
    d2 = np.einsum("q,e,eq->", w, detJ, (space.divergence(dof, xi) - divf(xq)) ** 2)
    return float(e2 ** 0.5), float(d2 ** 0.5)


def projection_errors(family, m):
    """L2 projection of the same field onto the space (serial): errors of the projection and of its divergence."""
    import scipy.sparse as sparse
    import scipy.sparse.linalg as splinalg
    s = m.subdomainMesh
    space = Space(s, nd, family)
    k = 2.0
    f = lambda x: np.stack([np.sin(k * x[..., i] + 0.3 * i) * np.cos(k * x[..., (i + 1) % nd] - 0.2) for i in range(nd)], axis=-1)
    divf = lambda x: sum(k * np.cos(k * x[..., i] + 0.3 * i) * np.cos(k * x[..., (i + 1) % nd] - 0.2)
                         - (k * np.sin(k * x[..., i] + 0.3 * i) * np.sin(k * x[..., (i + 1) % nd] - 0.2) if nd == 1 else 0.0)
                         for i in range(nd))
    xi, w = ref_rule()
    xq = phys_points(xi, s)
    detJ = np.abs(np.linalg.det(space.elementJacobians()))
    phi = space.getVectorBasisValues(xi)                          # (nE, nQ, dim, nd)
    Me = np.einsum("q,e,eqid,eqjd->eij", w, detJ, phi, phi)
    be = np.einsum("q,e,eqid,eqd->ei", w, detJ, phi, f(xq))
    l2g = space.dofMap.l2g
    n = space.dofMap.nDOF
    M = sparse.coo_matrix((Me.ravel(), (np.repeat(l2g, l2g.shape[1], axis=1).ravel(), np.tile(l2g, (1, l2g.shape[1])).ravel())), shape=(n, n)).tocsc()
    b = np.zeros(n)
    np.add.at(b, l2g, be)
    dof = splinalg.spsolve(M, b)
    e2 = np.einsum("q,e,eqd->", w, detJ, (space.evaluate(dof, xi) - f(xq)) ** 2)
    d2 = np.einsum("q,e,eq->", w, detJ, (space.divergence(dof, xi) - divf(xq)) ** 2)
    return float(e2 ** 0.5), float(d2 ** 0.5)


for family in FAMILIES:
    run_case(family, False)
if os.environ.get("RATES") == "1" and comm.size() == 1:
    fine = make_mesh(2)
    for family in FAMILIES:
        c, f = interpolation_errors(family, mesh), interpolation_errors(family, fine)
        pc, pf = projection_errors(family, mesh), projection_errors(family, fine)
        print("RATES " + json.dumps(dict(shape=shape, family=family, L2=[c[0], f[0]], div=[c[1], f[1]],
                                         L2_rate=float(np.log2(c[0] / f[0])), div_rate=float(np.log2(c[1] / f[1])),
                                         proj_L2=[pc[0], pf[0]], proj_div=[pc[1], pf[1]],
                                         proj_L2_rate=float(np.log2(pc[0] / pf[0])), proj_div_rate=float(np.log2(pc[1] / pf[1])))), flush=True)
# scramble the local vertex order of every element (each rank its own way, so a ghost element is listed
# differently on the two ranks that hold it); the dof map looks faces up by their vertices
rs = np.random.RandomState(100 + rank)
en = sub.elementNodesArray
if cube:    # a random symmetry of the cube per element: still a valid listing of a parallelepiped
    from proteus import CubeSpaces
    corners = np.array(CubeSpaces.CORNERS[nd])
    index = {tuple(c): i for i, c in enumerate(corners.tolist())}
    for eN in range(en.shape[0]):
        pi, flip = rs.permutation(nd), rs.randint(0, 2, nd)
        old = np.asarray(en[eN]).copy()
        for v, c in enumerate(corners):
            en[eN, index[tuple((flip ^ c[pi]).tolist())]] = old[v]
else:
    for eN in range(en.shape[0]):
        en[eN, :] = np.asarray(en[eN])[rs.permutation(nd + 1)]
for family in FAMILIES:
    run_case(family, True)
