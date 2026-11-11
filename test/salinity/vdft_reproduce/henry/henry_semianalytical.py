#!/usr/bin/env python3
"""
Henry semi-analytical reference solution at Proteus h5 nodes.

Implements the Galerkin/Fourier algebraic system from Appendix A of
    Zidane, Younes, Huggenberger, Zechner (2012)
    "The Henry semianalytical solution for saltwater intrusion with
     reduced dispersion", WRR 48, W06533, doi:10.1029/2011WR011157.

Series (Zidane eq. 4-5):
    psi(x,y) = sum_{m=1..MA, n=0..NA} A[m,n] sin(m pi y) cos(n pi x/xi)
    C(x,y)   = sum_{r=0..MB, s=1..NB} B[r,s] cos(r pi y) sin(s pi x/xi)

These are the PERTURBATIONS; physical fields are
    psi_full(x,y) = y    + psi(x,y)
    C_full(x,y)   = x/xi + C(x,y)

Algebraic system (Appendix A) is solved with Levenberg-Marquardt.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.optimize import least_squares
try:
    import h5py
except ImportError:
    h5py = None


BENCHMARK_CASES = {
    "henry": {
        "a": 0.263,
        "b": 0.1,
        "xi": 2.0,
        "description": "Original Henry (1964) benchmark.",
    },
    "simpson_clement": {
        "a": 0.1315,
        "b": 0.2,
        "xi": 2.0,
        "description": "Modified Henry benchmark from Simpson and Clement (2004).",
    },
}


# -------------------------------------------------------------------
# Closed-form integrals from Zidane Appendix A
# -------------------------------------------------------------------

def neg_term(k):
    """((-1)^k - 1) / k for integer k.  Convention: 0 at k = 0."""
    if k == 0:
        return 0.0
    if k % 2 == 1:
        return -2.0 / k
    return 0.0


def N_fn(h, n):
    """N(h, n) = ((-1)^(h+n) - 1)/(h+n) + ((-1)^(h-n) - 1)/(h-n)."""
    return neg_term(h + n) + neg_term(h - n)


def W_fn(arg1, arg2):
    """W(arg1, arg2) = ((-1)^arg1 - 1)/arg1 if arg2 == 0, else 0."""
    if arg2 != 0:
        return 0.0
    return neg_term(arg1)


# -------------------------------------------------------------------
# Levenberg-Marquardt solver for the truncated algebraic system
# -------------------------------------------------------------------

def henry_coefficients(a, b, xi, MA, NA, MB, NB,
                       x0=None, max_nfev=20000, verbose=False,
                       residual_tol=1e-8):
    """Solve Zidane eq. (A1)-(A2) by LM. Returns A (MA, NA+1) and B (MB+1, NB)."""
    pi = np.pi
    pi2 = pi * pi
    nA = MA * (NA + 1)
    nB = (MB + 1) * NB

    # ---------- pre-computed N tables ----------
    # N(g, r) for g in 1..MA, r in 0..MB    -> shape (MA, MB+1)
    N_g_r = np.array([[N_fn(g, r) for r in range(MB + 1)]
                      for g in range(1, MA + 1)])
    # N(h, n) for h in 1..NB, n in 0..NA    -> shape (NB, NA+1)
    N_h_n = np.array([[N_fn(h, n) for n in range(NA + 1)]
                      for h in range(1, NB + 1)])
    # N(h, s) for h in 1..NB, s in 1..NB    -> shape (NB, NB)
    N_h_s = np.array([[N_fn(h, s) for s in range(1, NB + 1)]
                      for h in range(1, NB + 1)])

    # ---------- Quad pre-tensors ----------
    # F[g, m, r] = δ(m-r=g) + δ(r-m=g) - δ(m+r=g)
    # L[g, m, r] = δ(m-r=g) + δ(r-m=g) + δ(m+r=g)
    if verbose:
        print("  building F, L, G, R tensors ...", file=sys.stderr)
    F_grm = np.zeros((MB + 1, MA, MB + 1))
    L_grm = np.zeros((MB + 1, MA, MB + 1))
    for g in range(MB + 1):
        for m in range(1, MA + 1):
            for r in range(MB + 1):
                d_mr = float(m - r == g)
                d_rm = float(r - m == g)
                d_mpr = float(m + r == g)
                F_grm[g, m - 1, r] = d_mr + d_rm - d_mpr
                L_grm[g, m - 1, r] = d_mr + d_rm + d_mpr

    # G[h, n, s] = neg(h+n-s) + neg(h-n+s) - neg(h+n+s) - neg(h-n-s)
    # R[h, n, s] = neg(h+n-s) + neg(h-n+s) + neg(h+n+s) + neg(h-n-s)
    G_hns = np.zeros((NB, NA + 1, NB))
    R_hns = np.zeros((NB, NA + 1, NB))
    for h in range(1, NB + 1):
        for n in range(NA + 1):
            for s in range(1, NB + 1):
                t1 = neg_term(h + n - s)
                t2 = neg_term(h - n + s)
                t3 = neg_term(h + n + s)
                t4 = neg_term(h - n - s)
                G_hns[h - 1, n, s - 1] = t1 + t2 - t3 - t4
                R_hns[h - 1, n, s - 1] = t1 + t2 + t3 + t4

    # ---------- LHS coefficients & W constants ----------
    g1 = np.arange(1, MA + 1)
    h1 = np.arange(0, NA + 1)
    g2 = np.arange(0, MB + 1)
    h2 = np.arange(1, NB + 1)
    eps2 = np.where(h1 == 0, 2.0, 1.0)   # eq A1: ε₂(h)
    eps1 = np.where(g2 == 0, 2.0, 1.0)   # eq A2: ε₁(g)

    G1, H1 = np.meshgrid(g1, h1, indexing="ij")
    lhs1_coef = eps2[None, :] * a * pi2 * (G1 ** 2 + H1 ** 2 / xi ** 2) * xi

    G2, H2 = np.meshgrid(g2, h2, indexing="ij")
    lhs2_coef = eps1[:, None] * b * pi2 * (G2 ** 2 + H2 ** 2 / xi ** 2) * xi

    # (4/π) W(g, h) for eq A1 — only h=0 contributes
    W1 = np.zeros((MA, NA + 1))
    for ig, g in enumerate(g1):
        W1[ig, 0] = (4.0 / pi) * W_fn(g, 0)
    # (4/π) W(h, g) for eq A2 — only g=0 contributes
    W2 = np.zeros((MB + 1, NB))
    for ih, h in enumerate(h2):
        W2[0, ih] = (4.0 / pi) * W_fn(h, 0)

    s_vec = np.arange(1, NB + 1, dtype=float)

    def unpack(x):
        return x[:nA].reshape(MA, NA + 1), x[nA:].reshape(MB + 1, NB)

    def residual(x):
        A, B = unpack(x)

        # ---- Eq A1: F1[g, h] = lhs1·A[g,h] − Σ_r B[r,h] h N(g,r) − W1 ----
        F1 = lhs1_coef * A - W1
        for ih, h_val in enumerate(h1):
            if 1 <= h_val <= NB:
                # B[:, h_val-1] is the column for s = h_val (first index r)
                F1[:, ih] -= h_val * (N_g_r @ B[:, h_val - 1])

        # ---- Eq A2: F2[g, h] = lhs2·B[g,h] − Σ_n A[g,n] g N(h,n) − ε₁ Σ_s B[g,s] s N(h,s) − Quad − W2 ----
        F2 = lhs2_coef * B - W2
        # A-coupling (only g in 1..MA)
        for ig, g_val in enumerate(g2):
            if 1 <= g_val <= MA:
                F2[ig, :] -= g_val * (A[g_val - 1, :] @ N_h_n.T)
        # B-coupling (same row of B)
        for ig in range(MB + 1):
            F2[ig, :] -= eps1[ig] * (B[ig, :] * s_vec) @ N_h_s.T

        # Quad = (π/4) Σ_{m,n,r,s} A[m,n] B[r,s] [m s L[g,m,r] R[h,n,s] − n r F[g,m,r] G[h,n,s]]
        sR = R_hns * s_vec.reshape(1, 1, -1)
        BS = np.einsum("rs,hns->rhn", B, sR)
        m_vec = np.arange(1, MA + 1, dtype=float)
        n_vec = np.arange(0, NA + 1, dtype=float)
        r_vec = np.arange(0, MB + 1, dtype=float)
        T1 = np.einsum("m,mn,gmr,rhn->gh", m_vec, A, L_grm, BS)
        BG = np.einsum("rs,hns->rhn", B, G_hns)
        T2 = np.einsum("n,mn,r,gmr,rhn->gh", n_vec, A, r_vec, F_grm, BG)
        Quad = (pi / 4.0) * (T1 - T2)
        F2 -= Quad

        return np.concatenate([F1.ravel(), F2.ravel()])

    if x0 is None:
        x0 = np.zeros(nA + nB)
    if verbose:
        print(f"  solving {nA + nB} unknowns (LM)...", file=sys.stderr)

    result = least_squares(residual, x0, method="lm",
                           xtol=1e-12, ftol=1e-12, max_nfev=max_nfev)
    res_norm = float(np.linalg.norm(result.fun))
    if verbose:
        print(f"  ||residual|| = {res_norm:.3e}, nfev = {result.nfev}, "
              f"status = {result.status}", file=sys.stderr)
    if not result.success:
        raise RuntimeError(
            f"LM solve failed (status={result.status}): {result.message}. "
            "Try --continuation or adjust the truncation."
        )
    if res_norm > residual_tol:
        raise RuntimeError(
            f"LM solve residual {res_norm:.3e} exceeds tolerance "
            f"{residual_tol:.1e}. Try --continuation or adjust the truncation."
        )

    return unpack(result.x)


# -------------------------------------------------------------------
# Series evaluation (adds back baselines)
# -------------------------------------------------------------------

def evaluate(x, y, A, B, xi):
    pi = np.pi
    MA, NA1 = A.shape
    MB1, NB = B.shape
    psi = y.copy()
    C = x / xi
    for m in range(1, MA + 1):
        sm = np.sin(m * pi * y)
        for n_idx in range(NA1):
            psi += A[m - 1, n_idx] * sm * np.cos(n_idx * pi * x / xi)
    for r in range(MB1):
        cr = np.cos(r * pi * y)
        for s in range(1, NB + 1):
            C += B[r, s - 1] * cr * np.sin(s * pi * x / xi)
    return psi, C


# -------------------------------------------------------------------
# h5 node reader
# -------------------------------------------------------------------

def list_node_datasets(h5_path):
    if h5py is None:
        raise RuntimeError("h5py is required to inspect datasets in an h5 file.")
    found = []

    def visit(name, obj):
        if (isinstance(obj, h5py.Dataset) and obj.ndim == 2
                and obj.shape[1] in (2, 3) and "node" in name.lower()):
            found.append((name, tuple(obj.shape)))

    with h5py.File(h5_path, "r") as f:
        f.visititems(visit)
    by_shape = {}
    for name, shape in found:
        by_shape.setdefault(shape, []).append(name)
    out = []
    for shape, names in by_shape.items():
        names.sort(key=lambda n: (not n.endswith("0"), len(n), n))
        out.append((names[0], shape, len(names)))
    return out


def read_nodes(h5_path, dataset=None):
    if h5py is None:
        raise RuntimeError("h5py is required to read nodes from an h5 file.")
    cands = list_node_datasets(h5_path)
    if not cands:
        raise RuntimeError(f"No node datasets in {h5_path}")
    if dataset is None:
        if len(cands) > 1:
            raise RuntimeError(
                "Multiple candidate node datasets found. "
                "Please rerun with --nodes-dataset or inspect --list-datasets."
            )
        dataset = cands[0][0]
        print(f"using dataset '{dataset}'  shape={cands[0][1]}  "
              f"({cands[0][2]} duplicates collapsed)", file=sys.stderr)
    with h5py.File(h5_path, "r") as f:
        nodes = f[dataset][:]
    return nodes, dataset


def read_nodes_from_npz(npz_path):
    data = np.load(npz_path)
    if "x" not in data or "y" not in data:
        raise RuntimeError(f"{npz_path} must contain 'x' and 'y' arrays.")
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    if x.shape != y.shape:
        raise RuntimeError(f"{npz_path} has incompatible x/y shapes: {x.shape} vs {y.shape}")
    nodes = np.column_stack((x, y))
    return nodes, f"npz:{npz_path}"


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=None)
    p.add_argument("--nodes-npz", default=None,
                   help="NPZ file containing x and y arrays to evaluate on.")
    p.add_argument("--nodes-dataset", default=None)
    p.add_argument("--list-datasets", action="store_true")
    p.add_argument("--case", choices=sorted(BENCHMARK_CASES),
                   default="henry",
                   help="Published benchmark case to run.")
    p.add_argument("--a", type=float, default=None)
    p.add_argument("--b", type=float, default=None)
    p.add_argument("--xi", type=float, default=None)
    p.add_argument("--d", type=float, default=1.0)
    p.add_argument("--MA", type=int, default=7,
                   help="A[1..MA, 0..NA]")
    p.add_argument("--NA", type=int, default=4)
    p.add_argument("--MB", type=int, default=4,
                   help="B[0..MB, 1..NB]")
    p.add_argument("--NB", type=int, default=19)
    p.add_argument("--continuation", action="store_true",
                   help="Step from the Henry case to the target a value.")
    p.add_argument("--residual-tol", type=float, default=1e-8,
                   help="Maximum accepted residual norm for the algebraic solve.")
    p.add_argument("--flip-x", action="store_true",
                   help="Map x to xi - x/ d before evaluation if the mesh is reversed.")
    p.add_argument("--out", default="henry_reference.npz")
    args = p.parse_args()

    if args.h5 is None and args.nodes_npz is None:
        p.error("one of --h5 or --nodes-npz is required")
    if args.h5 is not None and args.nodes_npz is not None:
        p.error("use only one of --h5 or --nodes-npz")

    if args.list_datasets:
        if args.h5 is None:
            p.error("--list-datasets requires --h5")
        for name, shape, ndup in list_node_datasets(args.h5):
            print(f"{name}\tshape={shape}\t(x{ndup} duplicates)")
        return

    case = BENCHMARK_CASES[args.case]
    a = case["a"] if args.a is None else args.a
    b = case["b"] if args.b is None else args.b
    xi = case["xi"] if args.xi is None else args.xi

    nA = args.MA * (args.NA + 1)
    nB = (args.MB + 1) * args.NB
    print(f"benchmark case: {args.case} ({case['description']})", file=sys.stderr)
    print(f"Henry params: a={a}, b={b}, xi={xi}", file=sys.stderr)
    print(f"  truncation: A[1..{args.MA}, 0..{args.NA}] = {nA},  "
          f"B[0..{args.MB}, 1..{args.NB}] = {nB},  total = {nA + nB}",
          file=sys.stderr)

    if args.continuation and abs(a - BENCHMARK_CASES["henry"]["a"]) > 1e-6:
        a_start = BENCHMARK_CASES["henry"]["a"]
        n_steps = max(1, int(np.ceil(abs(a - a_start) / 0.05)))
        a_path = np.linspace(a_start, a, n_steps + 1)
        print(f"  continuation: a path = {a_path}", file=sys.stderr)
        x_curr = np.zeros(nA + nB)
        for a_step in a_path:
            print(f"  -- a = {a_step:.4f}", file=sys.stderr)
            A, B = henry_coefficients(a_step, b, xi,
                                      args.MA, args.NA, args.MB, args.NB,
                                      x0=x_curr, verbose=True,
                                      residual_tol=args.residual_tol)
            x_curr = np.concatenate([A.ravel(), B.ravel()])
    else:
        A, B = henry_coefficients(a, b, xi,
                                  args.MA, args.NA, args.MB, args.NB,
                                  verbose=True,
                                  residual_tol=args.residual_tol)

    print(f"  |A|_inf = {np.abs(A).max():.3e}, |B|_inf = {np.abs(B).max():.3e}",
          file=sys.stderr)

    if args.nodes_npz is not None:
        nodes, ds = read_nodes_from_npz(args.nodes_npz)
    else:
        nodes, ds = read_nodes(args.h5, args.nodes_dataset)
    x_phys, y_phys = nodes[:, 0], nodes[:, 1]
    x = x_phys / args.d
    if args.flip_x:
        x = xi - x
    y = y_phys / args.d
    psi, c = evaluate(x, y, A, B, xi)
    print(f"  c range: [{c.min():.4f}, {c.max():.4f}]", file=sys.stderr)

    np.savez(args.out, x=x_phys, y=y_phys, psi=psi, c=c, A=A, B=B,
             a=a, b=b, xi=xi, d=args.d, dataset=ds, case=args.case)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
