"""L2(Omega) error of a Hornung-Messing run at the final archived time.

    python hm_l2_error.py <run_dir> [<run_dir> ...]

Each run_dir holds hm_2d.h5 (P1 nodal pressure head on triangles). The error
is integrated element by element: on triangles (P1) with a degree-8 (16-point)
rule on each of 4 uniform subtriangles, on quadrilaterals (Q1) with 5x5 Gauss
on 2x2 subcells, so quadrature error is negligible next to the
discretization error even across the x - y - t = 0 interface. Prints
h, dt, ||e||_L2 and, for consecutive runs, the observed order.
"""
import pathlib
import sys

import h5py
import numpy as np


def p_exact(x, y, t):
    xi = x - y - t
    e = np.exp(np.minimum(xi, 50.0))
    return np.where(xi < 0.0, -0.5 * xi, -np.tan((e - 1.0) / (e + 1.0)))


# Dunavant degree-8, 16-point rule on the reference triangle (weights sum to 1)
def _rule():
    pts = [(1/3, 1/3, 0.144315607677787)]
    for a, w in [(0.459292588292723, 0.095091634267285),
                 (0.170569307751760, 0.103217370534718),
                 (0.050547228317031, 0.032458497623198)]:
        b = 1 - 2 * a
        pts += [(a, a, w), (a, b, w), (b, a, w)]
    for a, b, w in [(0.008394777409958, 0.263112829634638, 0.027230314174435)]:
        c = 1 - a - b
        pts += [(a, b, w), (b, a, w), (a, c, w), (c, a, w), (b, c, w), (c, b, w)]
    q = np.array(pts)
    return q[:, :2], q[:, 2]


QP, QW = _rule()
assert abs(QW.sum() - 1.0) < 1e-10


def load_final(run_dir):
    with h5py.File(pathlib.Path(run_dir) / "hm_2d.h5", "r") as f:
        k = max(int(s.split("_t")[1]) for s in f if s.startswith("pressure_head_t"))
        X = np.array(f["nodesSpatial_Domain%d" % k])[:, :2]
        E = np.array(f["elementsSpatial_Domain%d" % k])
        U = np.array(f["pressure_head_t%d" % k]).ravel()
    return X, E, U, k


_G5, _W5 = np.polynomial.legendre.leggauss(5)


def integrate_q1(X, E, U, t):
    """Q1 on axis-aligned rectangles: bilinear interpolant, 5x5 Gauss on 2x2 subcells."""
    err2 = 0.0
    g = 0.5 * (_G5 + 1.0)                      # [0,1] points, weights _W5/2
    for quad in E:
        P = X[quad]
        x0, x1 = P[:, 0].min(), P[:, 0].max()
        y0, y1 = P[:, 1].min(), P[:, 1].max()
        # nodal values at the four corners, identified by position
        corner = {}
        for k in range(4):
            corner[(int(np.isclose(P[k, 0], x1)), int(np.isclose(P[k, 1], y1)))] = U[quad[k]]
        for sx in (0.0, 0.5):
            for sy in (0.0, 0.5):
                r = sx + 0.5 * g[:, None] + 0.0 * g[None, :]
                q = sy + 0.5 * g[None, :] + 0.0 * g[:, None]
                uh = (corner[(0, 0)] * (1 - r) * (1 - q) + corner[(1, 0)] * r * (1 - q)
                      + corner[(0, 1)] * (1 - r) * q + corner[(1, 1)] * r * q)
                e = uh - p_exact(x0 + r * (x1 - x0), y0 + q * (y1 - y0), t)
                w = np.outer(_W5, _W5) * 0.25 * 0.25
                err2 += (x1 - x0) * (y1 - y0) * np.sum(w * e * e)
    return np.sqrt(err2)


def integrate(X, E, U, t):
    if E.shape[1] == 4:
        return integrate_q1(X, E, U, t)
    err2 = 0.0
    # 4 uniform subtriangles in barycentric coordinates of the parent
    subs = [((0, 0), (.5, 0), (0, .5)), ((.5, 0), (1, 0), (.5, .5)),
            ((0, .5), (.5, .5), (0, 1)), ((.5, 0), (.5, .5), (0, .5))]
    for tri in E:
        P = X[tri]
        J = np.array([P[1] - P[0], P[2] - P[0]]).T
        area = 0.5 * abs(np.linalg.det(J))
        for s in subs:
            s = np.array(s)
            ref = s[0] + QP[:, :1] * (s[1] - s[0]) + QP[:, 1:2] * (s[2] - s[0])  # ref coords (r, s)
            phys = P[0] + ref @ J.T
            uh = U[tri[0]] * (1 - ref[:, 0] - ref[:, 1]) + U[tri[1]] * ref[:, 0] + U[tri[2]] * ref[:, 1]
            e = uh - p_exact(phys[:, 0], phys[:, 1], t)
            err2 += 0.25 * area * np.dot(QW, e * e)
    return np.sqrt(err2)


if __name__ == "__main__":
    prev = None
    for d in sys.argv[1:]:
        X, E, U, k = load_final(d)
        meta = dict(kv.split("=") for kv in pathlib.Path(d, "context.txt").read_text().split())
        T, nDT, nn = float(meta.get("T", 0.05)), int(meta["nDT"]), int(meta["nn"])
        h, dt = 1.0 / (nn - 1), T / nDT
        assert k == nDT, f"{d}: archive stops at step {k} of {nDT}"
        e = integrate(X, E, U, T)
        rate = "" if prev is None else f"{np.log(prev[1] / e) / np.log(prev[0] / h):6.2f}"
        print(f"{h:9.3e} {dt:9.3e} {e:9.3e} {rate}")
        prev = (h, e)
