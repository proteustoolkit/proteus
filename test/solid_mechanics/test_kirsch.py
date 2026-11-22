"""Verification of proteus.elastoplastic in the elastic range against the Kirsch solution (plate with a hole, plane strain)."""
import math
import os
from . import kirsch


def _cleanup(names):
    for name in names:
        for ext in ('h5', 'xmf', 'ymf', 'node', 'ele', 'face'):
            if os.path.exists(name + '.' + ext):
                os.remove(name + '.' + ext)


def test_kirsch_p1_convergence(opts):
    """P1 tetrahedra on nested slab meshes: second order in the L2 displacement error, first order in the stress error,
    and the stress concentration factor of 3 at the hole."""
    results = []
    names = []
    for n in (8, 16):
        prefix = 'kirsch_mesh_%d' % n
        ns, model = kirsch.solve(prefix, n, 1, opts)
        results.append(kirsch.errors(model))
        names += [prefix, 'kirsch_n%d_p1' % n]
        del ns
    _cleanup(names)
    (h0, eu0, es0, _, _), (h1, eu1, es1, _, smax) = results
    rate_u = math.log(eu0/eu1)/math.log(h0/h1)
    rate_s = math.log(es0/es1)/math.log(h0/h1)
    # measured 2026-11-14: errors 8.964e-3, 2.501e-3 (u) and 8.570e-2, 4.586e-2 (stress); rates 1.84 and 0.90
    assert eu1 < 3.0e-3 and es1 < 5.0e-2
    assert rate_u > 1.7
    assert rate_s > 0.85
    assert abs(smax/(3.0*kirsch.S) - 1.0) < 0.01
