#!/usr/bin/env python
"""
Generate the pinned reference tables for the H(div) local spaces RT0 and BDM1 (C8 round 5).

Run only to REGENERATE ``hdiv_tables.npz`` (needs sympy). The tests read the pinned file.

    python test/fem_oracle/generate_hdiv_tables.py [--out hdiv_tables.npz]

Construction (independent of ``proteus/HdivSpaces.py``, which uses closed forms): on the unit simplex take a
generating set of the space in monomials (RT0: the constants e_i and the field x; BDM1: e_i and x_j e_i),
build the matrix of the degrees of freedom on it,

    rt0:   sigma_f(v)     = N_f . v(centroid of face f)
    bdm1:  sigma_{f,a}(v) = N_f . v(p_a),  a a vertex of face f

with N_f the area-weighted outward normal computed from the face's vertices (rotation of the edge in 2D, cross
product in 3D, oriented away from the opposite vertex), and invert it exactly in rational arithmetic. The
dual basis is evaluated at fixed points.

Layout of ``hdiv_tables.npz``, keys ``<family>/<cell>/<item>`` (family rt0|bdm1, cell triangle|tetrahedron):
``dof_face`` (n,), ``dof_vertex`` (n,) (-1 for rt0), ``x`` (m, d), ``val`` (m, n, d), ``div`` (n,),
``face_x`` (d+1, q, d) points on each face, ``face_flux`` (d+1, q, n) = N_f . phi_j at those points,
``area_normal`` (d+1, d). Face f is opposite vertex f, as in ``FemTools.ReferenceSimplex``.
"""
import argparse
import os

import numpy as np

SEED = 20261114
DIM = {"triangle": 2, "tetrahedron": 3}


def build(out):
    import sympy as sp
    tables = {"meta/seed": np.array(SEED)}
    for cell, d in DIM.items():
        X = sp.symbols("x y z")[:d]
        verts = [sp.Matrix([0] * d)] + [sp.Matrix([1 if i == j else 0 for i in range(d)]) for j in range(d)]
        faces = [[v for v in range(d + 1) if v != f] for f in range(d + 1)]
        normals = []
        for f in range(d + 1):
            p = [verts[v] for v in faces[f]]
            if d == 2:
                t = p[1] - p[0]
                N = sp.Matrix([t[1], -t[0]])
            else:
                N = (p[1] - p[0]).cross(p[2] - p[0]) / 2
            centroid = sum(p, sp.zeros(d, 1)) / d
            if N.dot(centroid - verts[f]) < 0:
                N = -N
            normals.append(N)
        unit = [sp.Matrix([1 if i == j else 0 for i in range(d)]) for j in range(d)]
        xvec = sp.Matrix(X)
        rs = np.random.RandomState(SEED + d)
        for family in ("rt0", "bdm1"):
            if family == "rt0":
                gens = unit + [xvec]
                dofs = [(f, -1, sum([verts[v] for v in faces[f]], sp.zeros(d, 1)) / d) for f in range(d + 1)]
            else:
                gens = unit + [X[j] * unit[i] for i in range(d) for j in range(d)]
                dofs = [(f, a, verts[a]) for f in range(d + 1) for a in faces[f]]
            n = len(gens)
            assert len(dofs) == n
            A = sp.zeros(n, n)
            for i, (f, a, point) in enumerate(dofs):
                at = dict(zip(X, point))
                for k, g in enumerate(gens):
                    A[i, k] = normals[f].dot(g.subs(at))
            C = A.inv()
            basis = [sum((C[k, j] * gens[k] for k in range(n)), sp.zeros(d, 1)) for j in range(n)]
            m = 9
            x = rs.dirichlet(np.ones(d + 1), size=m)[:, 1:]
            x[0] = 1.0 / (d + 1)
            fn = [sp.lambdify(X, b, "numpy") for b in basis]
            val = np.array([[np.array(fn[j](*pt), dtype=float).reshape(d) for j in range(n)] for pt in x])
            div = np.array([float(sum(sp.diff(b[i], X[i]) for i in range(d))) for b in basis])
            q = 5
            w = rs.dirichlet(np.ones(d), size=q)
            face_x = np.array([[sum(float(wk) * np.array(verts[v], dtype=float).reshape(d) for wk, v in zip(wq, faces[f]))
                                for wq in w] for f in range(d + 1)])
            Nf = np.array([[float(c) for c in N] for N in normals])
            face_flux = np.array([[[Nf[f] @ np.array(fn[j](*pt), dtype=float).reshape(d) for j in range(n)]
                                   for pt in face_x[f]] for f in range(d + 1)])
            key = "%s/%s/" % (family, cell)
            tables[key + "dof_face"] = np.array([f for f, a, p in dofs])
            tables[key + "dof_vertex"] = np.array([a for f, a, p in dofs])
            tables[key + "x"] = x
            tables[key + "val"] = val
            tables[key + "div"] = div
            tables[key + "face_x"] = face_x
            tables[key + "face_flux"] = face_flux
            tables[key + "area_normal"] = Nf
            print("%-5s %-12s n=%2d" % (family, cell, n))
    np.savez_compressed(out, **tables)
    print("wrote", out, "%.1f kB" % (os.path.getsize(out) / 1024.0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdiv_tables.npz"))
    build(ap.parse_args().out)
