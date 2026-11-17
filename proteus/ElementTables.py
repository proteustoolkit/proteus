"""
Constexpr reference-element tables for compiled kernels (the C19 engine): C++ headers generated from the local
spaces of :mod:`proteus.SimplexSpaces` and :mod:`proteus.CubeSpaces`.

One header describes one *configuration* ``(cell, family, order, nodes, quadrature degree)`` as a struct of
``static constexpr`` data, in the shape of the engine's ``Space`` (``experimental/adrh/spaces.h``: ``nS``, ``nDOF``,
``nVertices``, ``nFaces``, ``nFaceVertices``, the vertices first among the dofs) and of its quadrature headers::

    namespace proteus_tables {
    struct lagrange_triangle_p3_gll_q8 {
      static constexpr int nS = 2, nDOF = 10, nVertices = 3, nFaces = 3, nFaceVertices = 2, order = 3;
      // element rule on the reference cell
      static constexpr int nQ = 16;
      static constexpr double qPoint[nQ][nS], qWeight[nQ];
      static constexpr double phi[nQ][nDOF];            // basis values at the quadrature points
      static constexpr double dphi[nQ][nDOF][nS];       // reference gradients
      static constexpr double d2phi[nQ][nDOF][nS][nS];  // reference Hessians (hessians=True, the default)
      // face rule: face k with its own coordinates eta, the reference normal scaled for  n dS = |det J| J^{-T} n_ref d(eta)
      static constexpr int nQFace = 6;
      static constexpr double fqPoint[nQFace][nS - 1], fqWeight[nQFace];     // eta, weights on the reference face
      static constexpr double fxi[nFaces][nQFace][nS];                       // the face points in reference coordinates xi
      static constexpr double phiFace[nFaces][nQFace][nDOF];                 // basis at them
      static constexpr double dphiFace[nFaces][nQFace][nDOF][nS];            // reference gradients at them
      static constexpr double nRef[nFaces][nS];
      static constexpr int faceVertex[nFaces][nFaceVertices];
      static constexpr int faceDOF[nFaces][nFaceDOFs];     // local dofs on face k (vertices, edges, face interior), in the order of the dofs
      // dofs
      static constexpr double dofPoint[nDOF][nS];          // the nodes (Lagrange) or control points (Bernstein)
      static constexpr int dofEntityDim[nDOF], dofEntityIndex[nDOF];   // the mesh entity a dof is interior to
      static constexpr int multiIndex[nDOF][nS + 1 or nS];             // simplices: alpha (|alpha| = p); cubes: tensor index
    };
    }

Conventions (the engine's, not proteus's own ``boundaryMapList``):

* simplices (segment, triangle, tetrahedron): reference cell the unit simplex (vertices ``0, e_1, .., e_d``), face ``k``
  opposite vertex ``k`` with vertices in increasing order; the face coordinates ``eta`` live on the standard reference
  ``(nS-1)``-simplex, ``lambda_k = 0``, ``lambda_{faceVertex(k,0)} = 1 - sum(eta)``, ``lambda_{faceVertex(k,m)} =
  eta_{m-1}``, and ``n_ref = (1, .., 1)`` for ``k = 0``, ``-e_{k-1}`` otherwise; weights sum to ``1/(nS-1)!``.
* cubes (quadrilateral, hexahedron): reference cell ``[-1,1]^nS``, face ``k = 2 I + side`` (``xi_I = -1`` for side 0, ``+1``
  for side 1), face coordinates the other reference coordinates in increasing order on ``[-1,1]^(nS-1)``,
  ``n_ref = +-e_I``; weights sum to ``2^(nS-1)``. ``faceProteus[k]`` is the number of the same face in ``CubeSpaces.FACES``.
  The dofs are in the order of :class:`proteus.CubeSpaces.CubeSpace`.
* the segment's faces are its two end points: no face coordinates, ``fqWeight = {1}``; the arrays ``[nQFace][0]`` are
  emitted with one dummy column so that they stay legal C++.

Quadrature. ``degree`` is the polynomial degree integrated exactly. Simplices: :class:`proteus.Quadrature.SimplexGaussQuadrature`
(triangle and tetrahedron, degree up to 20, positive interior or boundary points as that module returns them; a
rule may exceed the request), Gauss-Legendre on ``[0,1]`` for segments and for triangle faces' edges. Cubes: tensor
Gauss-Legendre with ``ceil((degree + 1) / 2)`` points per direction. The header records the rule's degree and the
smallest weight.

Command line::

    python -m proteus.ElementTables --cell triangle --family lagrange --order 3 --nodes gll --degree 8 > tables.h
"""
import argparse
import math
import sys

import numpy as np

CELLS = {"interval": ("simplex", 1), "triangle": ("simplex", 2), "tetrahedron": ("simplex", 3),
         "quadrilateral": ("cube", 2), "hexahedron": ("cube", 3)}


def _space(cell, family, order, nodes):
    kind, d = CELLS[cell]
    if kind == "simplex":
        from .SimplexSpaces import SimplexSpace
        return SimplexSpace(d, order, family, nodes)
    if family != "lagrange":
        raise ValueError("cubes have the lagrange family only")
    from .CubeSpaces import CubeSpace
    return CubeSpace(d, order, nodes)


def _gauss01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1.0) / 2.0, w / 2.0


def _gauss11(n):
    return np.polynomial.legendre.leggauss(n)


def _simplex_rule(m, degree):
    """Points (n, m) and weights (n,) on the reference m-simplex, exact to ``degree``."""
    if m == 0:
        return np.zeros((1, 0)), np.ones(1)
    if m == 1:
        x, w = _gauss01(degree // 2 + 1)
        return x[:, None], w
    from .Quadrature import SimplexGaussQuadrature
    q = SimplexGaussQuadrature(m, degree)
    return np.array([p[:m] for p in q.points], dtype=float), np.array(q.weights, dtype=float)


def _cube_rule(m, degree):
    """Tensor Gauss-Legendre on [-1,1]^m."""
    if m == 0:
        return np.zeros((1, 0)), np.ones(1)
    x, w = _gauss11(degree // 2 + 1)
    import itertools
    pts = np.array(list(itertools.product(x, repeat=m)))
    wts = np.prod(np.array(list(itertools.product(w, repeat=m))), axis=1)
    return pts.reshape(-1, m), wts


def _simplex_faces(nS):
    """Engine convention: per face k, the vertices, n_ref and the map eta -> xi (function of (k, eta (n, nS-1)))."""
    faceVertex = [[m if m < k else m + 1 for m in range(nS)] for k in range(nS + 1)]
    nref = np.zeros((nS + 1, nS))
    nref[0, :] = 1.0
    for k in range(1, nS + 1):
        nref[k, k - 1] = -1.0

    def to_reference(k, eta):
        lam = np.zeros((eta.shape[0], nS + 1))
        lam[:, faceVertex[k][0]] = 1.0
        for m in range(1, nS):
            lam[:, faceVertex[k][m]] = eta[:, m - 1]
            lam[:, faceVertex[k][0]] -= eta[:, m - 1]
        return lam[:, 1:]
    return np.array(faceVertex), nref, to_reference


def _cube_faces(nS):
    from .CubeSpaces import FACES, FACE_PLANE, CORNERS
    sign = lambda b, I: 2 * CORNERS[nS][b][I] - 1
    faceVertex, nref, proteus = [], np.zeros((2 * nS, nS)), []
    for k in range(2 * nS):
        I, side = k // 2, k % 2
        nref[k, I] = 1.0 if side else -1.0
        want = 1 if side else -1
        faceVertex.append([b for b in range(2 ** nS) if sign(b, I) == want])
        proteus.append(next(f for f, (ax, sd) in enumerate(FACE_PLANE[nS]) if (ax, sd) == (I, side)))

    def to_reference(k, eta):
        I, side = k // 2, k % 2
        xi = np.zeros((eta.shape[0], nS))
        m = 0
        for J in range(nS):
            if J == I:
                xi[:, J] = 1.0 if side else -1.0
            else:
                xi[:, J] = eta[:, m]
                m += 1
        return xi
    return np.array(faceVertex), nref, to_reference, proteus


def _fmt(x):
    if float(x) == 0.0:
        return "0.0"
    return repr(float(x))


def _array(name, values, ctype="double", indent="    "):
    """``static constexpr ctype name[d0][d1]..;`` initialiser for a numpy array."""
    a = np.asarray(values)
    dims = "".join("[%d]" % n for n in a.shape)

    def rec(x, depth):
        if x.ndim == 0:
            return _fmt(x) if ctype == "double" else str(int(x))
        if x.ndim == 1:
            return "{" + ", ".join(rec(e, depth + 1) for e in x) + "}"
        pad = "\n" + indent + "  " * (depth + 1)
        return "{" + pad + ("," + pad).join(rec(e, depth + 1) for e in x) + "}"
    return "%sstatic constexpr %s %s%s = %s;" % (indent, ctype, name, dims, rec(a, 0))


def config_name(cell, family, order, nodes, degree):
    nodes = "" if family == "bernstein" else "_" + nodes
    return "%s_%s_p%d%s_q%d" % (family, cell, order, nodes, degree)


def tables(cell, family="lagrange", order=1, nodes="gll", degree=None, hessians=True, face_degree=None):
    """The data of the header as a dict of numpy arrays (what :func:`emit_header` writes)."""
    kind, nS = CELLS[cell]
    space = _space(cell, family, order, nodes)
    degree = 2 * order + 2 if degree is None else int(degree)
    face_degree = degree if face_degree is None else int(face_degree)
    rule = _simplex_rule if kind == "simplex" else _cube_rule
    q, wq = rule(nS, degree)
    tab = space.tabulate(q, derivatives=2 if hessians else 1)
    out = dict(kind=kind, nS=nS, nDOF=space.dim, order=order, degree=degree, face_degree=face_degree,
               qPoint=q, qWeight=wq, phi=tab.values, dphi=tab.grads, dofPoint=space.dof_points)
    if hessians:
        out["d2phi"] = tab.hessians
    if kind == "simplex":
        faceVertex, nref, to_reference = _simplex_faces(nS)
        out["proteusFace"] = list(range(nS + 1))
    else:
        faceVertex, nref, to_reference, proteusFace = _cube_faces(nS)
        out["proteusFace"] = proteusFace
    nFaces = faceVertex.shape[0]
    eta, wf = rule(nS - 1, face_degree)
    fxi = np.array([to_reference(k, eta) for k in range(nFaces)])           # (nFaces, nQFace, nS)
    fphi = np.zeros((nFaces, len(wf), space.dim))
    fdphi = np.zeros((nFaces, len(wf), space.dim, nS))
    for k in range(nFaces):
        t = space.tabulate(fxi[k], derivatives=1)
        fphi[k], fdphi[k] = t.values, t.grads
    # a dummy column for the segment's faces (no face coordinates)
    out.update(nFaces=nFaces, nFaceVertices=faceVertex.shape[1], faceVertex=faceVertex, nRef=nref,
               fqPoint=eta if eta.shape[1] > 0 else np.zeros((len(wf), 1)), fqWeight=wf, fxi=fxi, phiFace=fphi,
               dphiFace=fdphi)
    # the dofs on each face: those whose basis function is not zero there = the dofs interior to the face's entities
    onFace = []
    for k in range(nFaces):
        verts = set(int(v) for v in faceVertex[k])
        dofs = []
        for dim, (vertexLists, dofLists) in enumerate(zip(space.entity_vertices, space.entity_dofs)):
            for vlist, dl in zip(vertexLists, dofLists):
                if set(int(v) for v in vlist) <= verts:
                    dofs += list(dl)
        onFace.append(sorted(dofs))
    n = len(onFace[0])
    assert all(len(x) == n for x in onFace)
    out["faceDOF"] = np.array(onFace)
    dofEntityDim = np.zeros(space.dim, dtype=int)
    dofEntityIndex = np.zeros(space.dim, dtype=int)
    for dim, dofLists in enumerate(space.entity_dofs):
        for e, dl in enumerate(dofLists):
            for i in dl:
                dofEntityDim[i], dofEntityIndex[i] = dim, e
    out["dofEntityDim"], out["dofEntityIndex"] = dofEntityDim, dofEntityIndex
    out["multiIndex"] = np.asarray(space.multi_index, dtype=int)
    out["min_weight"] = float(wq.min())
    return out


def emit_header(cell, family="lagrange", order=1, nodes="gll", degree=None, hessians=True, face_degree=None, guard=True):
    """The C++ header text for one configuration."""
    t = tables(cell, family, order, nodes, degree, hessians, face_degree)
    name = config_name(cell, family, order, nodes, t["degree"])
    nS, nDOF = t["nS"], t["nDOF"]
    L = []
    L.append("// Generated by proteus.ElementTables (python -m proteus.ElementTables --cell %s --family %s --order %d --nodes %s --degree %d%s)."
             % (cell, family, order, nodes, t["degree"], "" if hessians else " --no-hessians"))
    L.append("// Do not edit. Reference-element values and gradients at the quadrature points, see proteus/ElementTables.py.")
    L.append("// Rule: degree %d element rule (%d points, smallest weight %.6g), degree %d face rule (%d points)."
             % (t["degree"], len(t["qWeight"]), t["min_weight"], t["face_degree"], len(t["fqWeight"])))
    if guard:
        L.append("#pragma once")
    L.append("namespace proteus_tables {")
    L.append("struct %s {" % name)
    L.append("    static constexpr int nS = %d, nDOF = %d, nVertices = %d, nFaces = %d, nFaceVertices = %d, order = %d;"
             % (nS, nDOF, 2 ** nS if t["kind"] == "cube" else nS + 1, t["nFaces"], t["nFaceVertices"], order))
    L.append("    static constexpr int nQ = %d, nQFace = %d, nFaceDOF = %d, quadratureDegree = %d, faceQuadratureDegree = %d;"
             % (len(t["qWeight"]), len(t["fqWeight"]), t["faceDOF"].shape[1], t["degree"], t["face_degree"]))
    L.append("    static constexpr bool hasHessians = %s;" % ("true" if hessians else "false"))
    for key in ("qPoint", "qWeight", "phi", "dphi") + (("d2phi",) if hessians else ()):
        L.append(_array(key, t[key]))
    for key in ("fqPoint", "fqWeight", "fxi", "phiFace", "dphiFace", "nRef"):
        L.append(_array(key, t[key]))
    for key in ("faceVertex", "faceDOF", "proteusFace"):
        L.append(_array(key, t[key], "int"))
    L.append(_array("dofPoint", t["dofPoint"]))
    for key in ("dofEntityDim", "dofEntityIndex", "multiIndex"):
        L.append(_array(key, t[key], "int"))
    L.append("};")
    L.append("}  // namespace proteus_tables")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--family", default="lagrange", choices=("lagrange", "bernstein"))
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--nodes", default="gll", choices=("gll", "equispaced"))
    ap.add_argument("--degree", type=int, default=None, help="quadrature degree (default 2 * order + 2)")
    ap.add_argument("--face-degree", type=int, default=None)
    ap.add_argument("--no-hessians", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.write(emit_header(a.cell, a.family, a.order, a.nodes, a.degree, not a.no_hessians, a.face_degree))


if __name__ == "__main__":
    main()
