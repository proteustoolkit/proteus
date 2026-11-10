"""Manufactured solution for the coupled Richards-TADR verification.

Fields (mms_coupled_subsection.tex, eqs. mms_psi and mms_c):

    psi_e = psi0 (1 + A_psi s b(x,z)),   b = 16 x(1-x) z(1-z),   s = t/T
        after Radu & Wang (2014), as used by Suciu et al. (2021)
    c_e   = cbar + A_c tanh(exp(-omega t) xi/delta),
    xi    = z - (z0 - U t) + a_f cos(2 pi x)
        after Shunn et al. (2012), Problem 2

on the unit square, z up.  The sources are derived symbolically from the
equations the two kernels discretize (MMS_PLAN.md, "Code facts"):

    f_w = d_t(rho theta) + div(rho v)
    g   = rho theta c_t + rho v.grad(c) - div(theta rho D grad(c))

g is the source of TADR's CANCELLED form; the flow source reaches TADR through
B_i at the node's own c and cancels (TADR.py mms_source docstring).  The
density is dimensionless, rho = 1 + eps c (rho0 = 1), with gravity (0, -1).

Configuration is read from the environment once, at import, so the flow and
transport decks see the same fields:

    MMS_EPS      density contrast eps                         (0.235, UHD)
    MMS_OMEGA    viscosity omega_max; mu/mu0 = 1 + 1.85w - 4.1w^2 + 44.5w^3,
                 w = omega_max c                              (0, constant mu)
    MMS_DISP     0: D = Dm I;  1: mechanical dispersion on    (0)
    MMS_DM       molecular diffusion Dm [m^2/s]               (1e-6: the largest
                 value with mesh Peclet > 1 at the front up to N = 160)
    MMS_ALPHA_L  longitudinal dispersivity [m]; alpha_T = 0.1 alpha_L (1e-3)
    MMS_C        front | bubble | const                       (front)
    MMS_C0       the constant concentration of MMS_C=const    (0.5)
    MMS_STEADY   1: the steady variant (refinement path b)    (0)
    MMS_FLOW_ONLY 1: flow model alone, rho = MMS_RHO_REF      (0)
    MMS_RHO_REF  reference density of a flow-only run         (1)
    MMS_T        final time [s]; default one transit theta_min L / max|v| of the
                 T3 reference configuration (front, eps = 0.235, constant mu), so
                 every test and every point of the eps / omega sweep shares one T
                 and one time grid
    MMS_DZF      total front travel U T [m]                   (0.5 L)

Run this file to execute the self-tests.
"""

import os

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _env(name, default):
    return float(os.environ.get(name, default))


EPS = _env("MMS_EPS", 0.235)
OMEGA_MAX = _env("MMS_OMEGA", 0.0)
DISP = int(os.environ.get("MMS_DISP", "0"))
DM = _env("MMS_DM", 1.0e-6)
ALPHA_L = _env("MMS_ALPHA_L", 1.0e-3) if DISP else 0.0
ALPHA_T = 0.1 * ALPHA_L
C_FIELD = os.environ.get("MMS_C", "front").strip().lower()
C0 = _env("MMS_C0", 0.5)
STEADY = int(os.environ.get("MMS_STEADY", "0"))
FLOW_ONLY = int(os.environ.get("MMS_FLOW_ONLY", "0"))
RHO_REF = _env("MMS_RHO_REF", 1.0)
if C_FIELD not in ("front", "bubble", "const"):
    raise ValueError("MMS_C must be front, bubble or const; got %r" % C_FIELD)

# UHD sand, benchmark_problem/cases.py
L = 1.0
K0 = 0.00112          # m/s
VG_ALPHA = 11.6       # 1/m
VG_N = 2.39
VG_M = 1.0 - 1.0 / VG_N
THETA_R = 0.0162
THETA_SR = 0.3438

# Field parameters, mms_coupled_subsection.tex
PSI0 = -1.0 / VG_ALPHA
A_PSI = 1.0           # < alpha L / 4 = 2.9 keeps v_z < 0 everywhere
CBAR, A_C = 0.5, 0.3
Z0 = 0.75 * L
DELTA = 0.15 * L
A_F = 0.05 * L
DZF = _env("MMS_DZF", 0.5 * L)
WIDEN = np.log(2.0)   # omega T: the front width doubles over the run

# ---------------------------------------------------------------------------
# Symbolic fields and coefficients
# ---------------------------------------------------------------------------
x, z, t, T = sp.symbols("x z t T", real=True)
s = 1 if STEADY else t / T
b = 16 * (x / L) * (1 - x / L) * (z / L) * (1 - z / L)

psi = PSI0 * (1 + A_PSI * s * b)


def _concentration(c_field):
    if c_field == "front":
        z_f = Z0 if STEADY else Z0 - DZF * t / T
        widen = 1 if STEADY else sp.exp(-WIDEN * t / T)
        xi = z - z_f + A_F * sp.cos(2 * sp.pi * x / L)
        return CBAR + A_C * sp.tanh(widen * xi / DELTA)
    if c_field == "bubble":
        return 0.2 + 0.6 * s * b
    return sp.Float(C0)


# VGM in the capillary head psiC = -psi > 0; kr written as in
# pskRelations.h vgm_wetting, which is the Mualem form exactly
psiC = -psi
Se = (1 + (VG_ALPHA * psiC) ** VG_N) ** (-VG_M)
theta = THETA_R + THETA_SR * Se
kr = sp.sqrt(Se) * (1 - (VG_ALPHA * psiC) ** (VG_N - 1) * Se) ** 2


def _darcy(rho_, mu_):
    """Darcy flux -kr (K0/mu)(grad psi - rho g) with g = (0, -1)."""
    return (-kr * (K0 / mu_) * sp.diff(psi, x),
            -kr * (K0 / mu_) * (sp.diff(psi, z) + rho_))


c = _concentration(C_FIELD)
rho = sp.Float(RHO_REF) if FLOW_ONLY else 1 + EPS * c
w = 0 if FLOW_ONLY else OMEGA_MAX * c
mu = 1 + 1.85 * w - 4.1 * w ** 2 + 44.5 * w ** 3
vx, vz = _darcy(rho, mu)


def _transport_terms(cfield):
    """Advective and dispersive parts of the transport operator for cfield."""
    cx, cz = sp.diff(cfield, x), sp.diff(cfield, z)
    # theta rho D, D = Dm I + alpha_L |v_p| vhat vhat + alpha_T |v_p| (I - vhat vhat)
    # with v_p = v/theta, so theta |v_p| = |v|
    vmag = sp.sqrt(vx ** 2 + vz ** 2)
    a_xx = rho * (theta * DM + ALPHA_L * vx * vx / vmag + ALPHA_T * (vmag - vx * vx / vmag))
    a_zz = rho * (theta * DM + ALPHA_L * vz * vz / vmag + ALPHA_T * (vmag - vz * vz / vmag))
    a_xz = rho * (ALPHA_L - ALPHA_T) * vx * vz / vmag
    jx = a_xx * cx + a_xz * cz
    jz = a_xz * cx + a_zz * cz
    div_disp = sp.diff(jx, x) + sp.diff(jz, z)
    return rho * (vx * cx + vz * cz), div_disp, jx, jz


f_w_sym = sp.diff(rho * theta, t) + sp.diff(rho * vx, x) + sp.diff(rho * vz, z)
_adv, _div_disp, _jx, _jz = _transport_terms(c)
g_sym = rho * theta * sp.diff(c, t) + _adv - _div_disp


# ---------------------------------------------------------------------------
# Numerical callables.  The velocity depends on t only through s = t/T, so
# max|v| and theta_min do not depend on T.  VMAX is taken from the T3 reference
# configuration, not from this run's eps / omega_max / c field, so that T and
# the time step (domain_mms.py) are the same for every test.
# ---------------------------------------------------------------------------
def _lambdify(expr, T_value):
    return sp.lambdify((x, z, t), expr.subs(T, T_value), modules="numpy", cse=True)


def _sample_grid(n_xz=101, n_t=21):
    xs = np.linspace(0.0, L, n_xz)
    X, Z, S = np.meshgrid(xs, xs, np.linspace(0.0, 1.0, n_t), indexing="ij")
    return X, Z, S


_X, _Z, _S = _sample_grid()
_vx_ref, _vz_ref = _darcy(1 + 0.235 * _concentration("front"), 1)
VMAX = float(np.max(_lambdify(sp.sqrt(_vx_ref ** 2 + _vz_ref ** 2), 1.0)(_X, _Z, _S)))
THETA_MIN = float(np.min(_lambdify(theta, 1.0)(_X, _Z, _S)))
T_END = _env("MMS_T", THETA_MIN * L / VMAX)

_psi_n = _lambdify(psi, T_END)
_c_n = _lambdify(c, T_END)
_f_w_n = _lambdify(f_w_sym, T_END)
_g_n = _lambdify(g_sym, T_END)


def _on(fn, X, tval):
    """Evaluate fn on points X of shape (..., 3) (nodeArray, a BC point, an IC point)."""
    X = np.asarray(X, dtype=float)
    return np.zeros(X.shape[:-1]) + fn(X[..., 0], X[..., 1], tval)


def psi_e(X, tval):
    return _on(_psi_n, X, tval)


def c_e(X, tval):
    return _on(_c_n, X, tval)


def f_w(X, tval):
    """Flow source [rho/rho0 per second], Richards Coefficients(mms_source=...)."""
    return _on(_f_w_n, X, tval)


def g(X, tval):
    """Cancelled-form transport source, TADR Coefficients(mms_source=...)."""
    return _on(_g_n, X, tval)


def describe():
    return ("MMS c=%s steady=%d flow_only=%d eps=%g omega_max=%g Dm=%g alpha_L=%g "
            "T=%.6e max|v|=%.6e theta_min=%.6f"
            % (C_FIELD, STEADY, FLOW_ONLY, EPS, OMEGA_MAX, DM, ALPHA_L,
               T_END, VMAX, THETA_MIN))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.05, 0.95, size=(50, 2))
    ts = rng.uniform(0.0, T_END, size=50)

    # 1. g = f_s - c f_w, with f_s the conservative transport source
    f_s = (sp.diff(rho * theta * c, t)
           + sp.diff(rho * vx * c - _jx, x) + sp.diff(rho * vz * c - _jz, z))
    resid = _lambdify(g_sym - (f_s - c * f_w_sym), T_END)(pts[:, 0], pts[:, 1], ts)
    scale = np.max(np.abs(_g_n(pts[:, 0], pts[:, 1], ts))) + 1.0e-300
    assert np.max(np.abs(resid)) <= 1.0e-10 * scale, "g != f_s - c f_w"

    # 2. hydrostatic head with constant c: no flow, no source
    c_h = 0.5
    rho_h = 1 + EPS * c_h
    psi_h = -0.1 - rho_h * z
    Se_h = (1 + (VG_ALPHA * -psi_h) ** VG_N) ** (-VG_M)
    theta_h = THETA_R + THETA_SR * Se_h
    kr_h = sp.sqrt(Se_h) * (1 - (VG_ALPHA * -psi_h) ** (VG_N - 1) * Se_h) ** 2
    fw_h = (sp.diff(rho_h * theta_h, t)
            + sp.diff(-rho_h * kr_h * K0 * sp.diff(psi_h, x), x)
            + sp.diff(-rho_h * kr_h * K0 * (sp.diff(psi_h, z) + rho_h), z))
    assert sp.simplify(fw_h) == 0, "f_w != 0 for a hydrostatic head"

    # 3. the field properties the subsection claims (coupled fields only)
    X, Z, S = _sample_grid(201, 41)
    tt = S * T_END
    psi_s = _psi_n(X, Z, tt)
    vz_s = _lambdify(vz, T_END)(X, Z, tt)
    assert psi_s.min() >= -2.0 / VG_ALPHA - 1e-12 and psi_s.max() <= -1.0 / VG_ALPHA + 1e-12
    assert vz_s.max() < 0.0, "v_z must be negative everywhere"
    if C_FIELD == "front" and not FLOW_ONLY:
        c_s = _c_n(X, Z, tt)
        dcdz = _lambdify(sp.diff(c, z), T_END)(X, Z, tt)
        assert 0.2 <= c_s.min() and c_s.max() <= 0.8
        assert dcdz.min() > 0.0, "c_e must be strictly monotone in z"
    print("self-tests passed")
    print(describe())
    print("psi in [%.4f, %.4f] m, max v_z = %.3e m/s, theta_min = %.4f"
          % (psi_s.min(), psi_s.max(), vz_s.max(), THETA_MIN))


if __name__ == "__main__":
    _self_test()
