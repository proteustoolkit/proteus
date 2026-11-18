"""
Constexpr reference-element tables for the C19 engine (C8 round 7): `proteus.ElementTables`.

Without a compiler (pure numbers): partition of unity, zero gradient and Hessian sums, the element rule integrates
what it should, a basis function vanishes on the faces it is not on (so `faceDOF` and the face mapping are right), the
reference divergence theorem with the engine's `n_ref` and face weights (`int_K d_I phi = sum_k n_ref[k][I] int_Fk phi`),
the entity of every dof, the face rule.

With a C++17 compiler: the generated header compiles as `constexpr`, and a program that includes it prints its tables,
which equal the numpy tabulation to the last digit; and, for the spaces the engine already has (P1 and P2 on
triangles and tetrahedra, Q1 on quadrilaterals and hexahedra), the tables equal what `experimental/adrh/spaces.h` computes
(basis values and gradients at the rule's points and at the face points, `faceToReference`, `referenceNormal`,
`faceVertex`): the header is a drop-in for the engine's `Space` data. `experimental/adrh` is only read.
"""
import itertools
import os
import shutil
import subprocess

import numpy as np
import pytest

from proteus import ElementTables as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ADRH = os.path.join(HERE, "..", "experimental", "adrh")
CXX = shutil.which(os.environ.get("CXX", "c++")) or shutil.which("g++") or shutil.which("clang++")

CONFIGS = [("interval", "lagrange", 2, "gll", 6), ("triangle", "lagrange", 1, "gll", 4), ("triangle", "lagrange", 3, "gll", 8),
           ("triangle", "bernstein", 2, "gll", 6), ("tetrahedron", "lagrange", 2, "gll", 6),
           ("tetrahedron", "lagrange", 3, "gll", 8), ("quadrilateral", "lagrange", 3, "gll", 8),
           ("hexahedron", "lagrange", 2, "gll", 6)]
IDS = ["%s-%s-p%d-q%d" % (c[0], c[1], c[2], c[4]) for c in CONFIGS]


def _cell_volume(cell):
    from math import factorial
    kind, nS = ET.CELLS[cell]
    return 1.0 / factorial(nS) if kind == "simplex" else 2.0 ** nS


def _face_volume(cell):
    from math import factorial
    kind, nS = ET.CELLS[cell]
    return 1.0 / factorial(nS - 1) if kind == "simplex" else 2.0 ** (nS - 1)


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_the_numbers(cfg):
    cell, family, p, nodes, degree = cfg
    t = ET.tables(cell, family, p, nodes, degree)
    nS, nDOF = t["nS"], t["nDOF"]
    q, w = t["qPoint"], t["qWeight"]
    assert abs(w.sum() - _cell_volume(cell)) < 1.0e-14
    # partition of unity and zero derivative sums
    assert np.abs(t["phi"].sum(axis=1) - 1.0).max() < 1.0e-13
    assert np.abs(t["dphi"].sum(axis=1)).max() < 1.0e-12
    assert np.abs(t["d2phi"].sum(axis=1)).max() < 1.0e-10
    assert np.abs(t["phiFace"].sum(axis=2) - 1.0).max() < 1.0e-13
    # the rule integrates monomials of degree <= degree exactly (simplex: prod a! / (|a| + d)!; cube: prod 2/(a+1) or 0)
    from math import factorial
    kind = t["kind"]
    rs = np.random.RandomState(50)
    for _ in range(12):
        a = rs.randint(0, degree + 1, nS)
        if kind == "simplex":
            if a.sum() > degree:
                continue
            exact = np.prod([factorial(i) for i in a]) / factorial(a.sum() + nS)
        else:
            exact = np.prod([2.0 / (i + 1) if i % 2 == 0 else 0.0 for i in a])
        assert abs(w @ np.prod(q ** a, axis=1) - exact) < 1.0e-13, a
    # the face rule: total weight, and the basis traces
    assert abs(t["fqWeight"].sum() - _face_volume(cell)) < 1.0e-14
    nFaces = t["nFaces"]
    for k in range(nFaces):
        off = np.setdiff1d(np.arange(nDOF), t["faceDOF"][k])
        assert np.abs(t["phiFace"][k][:, off]).max() < 1.0e-12, k        # zero on the dofs not on the face
    # reference divergence theorem with the engine's normals: int_K d_I phi_i = sum_k nRef[k][I] int_Fk phi_i
    lhs = np.einsum("q,qiI->iI", w, t["dphi"])
    rhs = np.einsum("kI,f,kfi->iI", t["nRef"], t["fqWeight"], t["phiFace"])
    assert np.abs(lhs - rhs).max() < 1.0e-12
    # the vertices come first (the engine's isoparametric geometry relies on it), then the dofs of edges, faces, the interior
    nv = 2 ** nS if kind == "cube" else nS + 1
    assert (t["dofEntityDim"][:nv] == 0).all() and (t["dofEntityDim"][nv:] > 0).all()
    assert (np.diff(t["dofEntityDim"]) >= 0).all()
    assert t["faceDOF"].shape[0] == nFaces


def _compile_and_run(tmp_path, source, includes=()):
    src = tmp_path / "main.cpp"
    src.write_text(source)
    exe = tmp_path / "main"
    cmd = [CXX, "-std=c++17", "-O0", "-Wall", "-Werror", "-I", str(tmp_path)] + ["-I" + i for i in includes] + [str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True, timeout=300).stdout
    return out


def _dump_program(name, arrays):
    body = []
    for a in arrays:
        body.append("  for (const double *p = reinterpret_cast<const double *>(&T::%s); p < reinterpret_cast<const double *>(&T::%s) + sizeof(T::%s) / sizeof(double); ++p) std::printf(\"%%.17g\\n\", *p);" % (a, a, a))
        body.append("  std::printf(\"--\\n\");")
    return ('#include <cstdio>\n#include "table.h"\nusing T = proteus_tables::%s;\nstatic_assert(T::nQ > 0 && T::nS >= 1, "");\n'
            'int main() {\n%s\n  return 0;\n}\n' % (name, "\n".join(body)))


@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_the_header_compiles_and_holds_the_numpy_tables_to_the_last_digit(tmp_path, cfg):
    cell, family, p, nodes, degree = cfg
    (tmp_path / "table.h").write_text(ET.emit_header(cell, family, p, nodes, degree))
    t = ET.tables(cell, family, p, nodes, degree)
    keys = ["qPoint", "qWeight", "phi", "dphi", "d2phi", "fqPoint", "fqWeight", "fxi", "phiFace", "dphiFace", "nRef", "dofPoint"]
    name = ET.config_name(cell, family, p, nodes, t["degree"])
    out = _compile_and_run(tmp_path, _dump_program(name, keys))
    blocks = out.split("--\n")[:-1]
    assert len(blocks) == len(keys)
    for key, block in zip(keys, blocks):
        got = np.array([float(x) for x in block.split()])
        assert got.shape == np.asarray(t[key]).ravel().shape, key
        assert np.array_equal(got, np.asarray(t[key], dtype=float).ravel()), key


ENGINE = [("triangle", 1, "P1Simplex<2>"), ("triangle", 2, "P2Simplex<2>"), ("tetrahedron", 1, "P1Simplex<3>"),
          ("tetrahedron", 2, "P2Simplex<3>"), ("quadrilateral", 1, "Q1Cube<2>"), ("hexahedron", 1, "Q1Cube<3>")]


@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
@pytest.mark.parametrize("cell,order,space", ENGINE, ids=["%s-p%d" % (c, o) for c, o, s in ENGINE])
def test_the_tables_equal_what_the_engines_space_computes(tmp_path, cell, order, space):
    degree = 6
    (tmp_path / "table.h").write_text(ET.emit_header(cell, "lagrange", order, "gll", degree))
    name = ET.config_name(cell, "lagrange", order, "gll", degree)
    prog = r'''
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "spaces.h"
#include "table.h"
using T = proteus_tables::%(name)s;
using S = adrh::%(space)s;
static_assert(T::nS == S::nS && T::nDOF == S::nDOF && T::nVertices == S::nVertices && T::nFaces == S::nFaces && T::nFaceVertices == S::nFaceVertices, "");
int main() {
  double dphi = 0, dgrad = 0, dface = 0, dfgrad = 0, dxi = 0, dnorm = 0; int dvert = 0;
  for (int q = 0; q < T::nQ; ++q) {
    double w[S::nDOF], gw[S::nDOF][S::nS];
    S::basis(T::qPoint[q], w, gw);
    for (int i = 0; i < S::nDOF; ++i) {
      dphi = std::max(dphi, std::fabs(w[i] - T::phi[q][i]));
      for (int I = 0; I < S::nS; ++I) dgrad = std::max(dgrad, std::fabs(gw[i][I] - T::dphi[q][i][I]));
    }
  }
  for (int k = 0; k < S::nFaces; ++k) {
    double n[S::nS];
    S::referenceNormal(k, n);
    for (int I = 0; I < S::nS; ++I) dnorm = std::max(dnorm, std::fabs(n[I] - T::nRef[k][I]));
    for (int m = 0; m < S::nFaceVertices; ++m) dvert += (S::faceVertex(k, m) != T::faceVertex[k][m]);
    for (int f = 0; f < T::nQFace; ++f) {
      double xi[S::nS];
      S::faceToReference(k, T::fqPoint[f], xi);
      double w[S::nDOF], gw[S::nDOF][S::nS];
      S::basis(xi, w, gw);
      for (int I = 0; I < S::nS; ++I) dxi = std::max(dxi, std::fabs(xi[I] - T::fxi[k][f][I]));
      for (int i = 0; i < S::nDOF; ++i) {
        dface = std::max(dface, std::fabs(w[i] - T::phiFace[k][f][i]));
        for (int I = 0; I < S::nS; ++I) dfgrad = std::max(dfgrad, std::fabs(gw[i][I] - T::dphiFace[k][f][i][I]));
      }
    }
  }
  std::printf("%%.3e %%.3e %%.3e %%.3e %%.3e %%.3e %%d\n", dphi, dgrad, dface, dfgrad, dxi, dnorm, dvert);
  return 0;
}
''' % dict(name=name, space=space)
    out = _compile_and_run(tmp_path, prog, includes=[ADRH]).split()
    assert all(float(x) < 1.0e-13 for x in out[:6]), out
    assert int(out[6]) == 0, out
