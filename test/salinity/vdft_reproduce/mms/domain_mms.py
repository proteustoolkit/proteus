"""Mesh, FE space and time grid shared by the MMS flow and transport decks.

Unit square, structured union-jack triangles (as benchmark_problem/
domain_simmons.py), so the refinement ladder is uniform refinement of one grid.

    MMS_N    intervals per side                                   (10)
    MMS_CO   Courant number dt max|v| / (theta_min h)             (0.5)
    MMS_DT   fixed dt [s]; overrides MMS_CO (dt-halving runs)
    MMS_FCT  1: FCT on in BOTH models                             (1)
"""

import math
import os

from proteus import *
from proteus.default_p import *

try:
    from . import mms_exact
except ImportError:
    import mms_exact

nd = 2

N = int(os.environ.get("MMS_N", "10"))
COURANT = float(os.environ.get("MMS_CO", "0.5"))
FCT = os.environ.get("MMS_FCT", "1") not in ("0", "false", "False")

# Old-style rectangle: NumericalSolution builds the structured mesh.
# RectangularDomain's 2D tags are bottom/right/top/left = 1/2/3/4.
L = (mms_exact.L, mms_exact.L, 1.0)
x0 = (0.0, 0.0, 0.0)
domain = None
polyfile = None
boundaryTags = {"bottom": 1, "right": 2, "top": 3, "left": 4}
nnx = N + 1
nny = N + 1
triangleFlag = 1
h = mms_exact.L / N

# One time grid per (N, Co): VMAX and THETA_MIN come from the T3 reference
# configuration (mms_exact), so every test and sweep point steps alike.
T = mms_exact.T_END
_dt_env = os.environ.get("MMS_DT")
DT = float(_dt_env) if _dt_env else COURANT * mms_exact.THETA_MIN * h / mms_exact.VMAX
NSTEPS = int(math.ceil(T / DT - 1.0e-9))
DT = T / NSTEPS
# The two 1e-8 s startup steps fill the coupled fields before the first real
# step, as in the benchmark and Elder decks.
tnList = [0.0, 1.0e-8, 2.0e-8] + [k * DT for k in range(1, NSTEPS + 1)]

from proteus.StepControl import FixedStep as _FixedStep


class FixedDT_controller(_FixedStep):
    """FixedStep that restores dt = DT before every step.

    Plain FixedStep keeps whatever dt the last step used, and stepExact writes
    the clipped 1e-8 s startup step back into every model, so without the
    reset the whole run would step at 1e-8 s.
    """

    def choose_dt_model(self):
        self.dt_model = DT
        _FixedStep.choose_dt_model(self)


# P1 on simplices for both models, with ONE quadrature rule: TADR reads Richards'
# quadrature arrays point for point (TADR.checkCoupledQuadrature).
femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexLobattoQuadrature(nd, 1)
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd - 1, 1)
subgridError = None


class FieldIC(object):
    """Initial condition from a manufactured field fn(X, t)."""

    def __init__(self, fn):
        self.fn = fn

    def uOfXT(self, x, t):
        return float(self.fn(x, 0.0))

    def uOfX(self, x):
        return float(self.fn(x, 0.0))


def dirichlet_everywhere(fn):
    """Dirichlet data fn(x, t) on all four sides, for dirichletConditions."""
    def getDBC(x, tag):
        if tag in boundaryTags.values():
            return lambda x, t: float(fn(x, t))
    return getDBC


def no_flux_condition(x, tag):
    """Every face is Dirichlet, so no face carries a flux condition."""
    return None
