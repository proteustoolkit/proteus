"""Check on a P1 triangle mesh that, for nodal pressure p,
    gradient form   G_i = sum_j p_j c_ij,             c_ij = int phi_i grad phi_j
    IBP form        I_i = -sum_j p_j c_ji + sum_j n_ij p_j,   n_ij = int_{dOmega} phi_i phi_j nu
agree to round-off (c_ij + c_ji = n_ij), and that replacing the boundary integral by the
lumped p_i n_i (n_i = sum_j n_ij) does not.
"""
import numpy as np
from scipy.spatial import Delaunay

rng = np.random.default_rng(1)
n = 14
gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 0.5, n // 2 + 1))
pts = np.c_[gx.ravel(), gy.ravel()] + 0.01 * rng.standard_normal((gx.size, 2)) * ((gx.ravel() % 1 > 0) & (gx.ravel() < 1) & (gy.ravel() > 0) & (gy.ravel() < 0.5))[:, None]
tri = Delaunay(pts).simplices
N = len(pts)
C = np.zeros((2, N, N)); Nb = np.zeros((2, N, N))
for t in tri:
    P = pts[t]; B = np.array([P[1] - P[0], P[2] - P[0]]).T
    area = 0.5 * abs(np.linalg.det(B)); Bi = np.linalg.inv(B)
    gradphi = np.array([[-1, -1], [1, 0], [0, 1]]) @ Bi          # (3,2): grad phi_a
    for a in range(3):
        for b in range(3):
            C[:, t[a], t[b]] += area / 3.0 * gradphi[b]           # int phi_a = area/3
# boundary edges: edges that belong to one triangle only
from collections import Counter
cnt = Counter(); 
for t in tri:
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cnt[tuple(sorted((t[a], t[b])))] += 1
cent = pts.mean(0)
for (a, b), c in cnt.items():
    if c != 1: continue
    e = pts[b] - pts[a]; L = np.linalg.norm(e); nu = np.array([e[1], -e[0]]) / L
    mid = 0.5 * (pts[a] + pts[b])
    if np.dot(nu, mid - cent) < 0: nu = -nu
    M = L / 6.0 * np.array([[2, 1], [1, 2]]); idx = [a, b]
    for p_, i in enumerate(idx):
        for q_, j in enumerate(idx):
            Nb[:, i, j] += M[p_, q_] * nu
print("max |c_ij + c_ji - n_ij|        :", np.abs(C + C.transpose(0, 2, 1) - Nb).max())
p = rng.random(N)
G = np.einsum("dij,j->di", C, p)
I = -np.einsum("dji,j->di", C, p) + np.einsum("dij,j->di", Nb, p)
Il = -np.einsum("dji,j->di", C, p) + Nb.sum(2) * p
print("max |G - I_exact|                 :", np.abs(G - I).max())
print("max |G - I_lumped|                :", np.abs(G - Il).max(), "  (|G| max %.3g)" % np.abs(G).max())
bnd = np.abs(Nb).sum((0, 2)) > 0
print("rows with a difference (lumped)   :", int((np.abs(G - Il).max(0) > 1e-12).sum()), "of", int(bnd.sum()), "boundary rows")
print("max |sum_j c_ij| (rows sum to 0)  :", np.abs(C.sum(2)).max())
