"""Richards half of the coupled MMS (mms_coupled_subsection.tex).

The paper configuration: entropy-viscosity edge scheme (STABILIZATION_TYPE=2),
FCT from MMS_FCT, weak IIPG Dirichlet psi_e on all four sides, no seepage face,
and the manufactured water source f_w added as a lumped nodal term.  Density is
dimensionless, rho/rho0 with rho0 = 1, supplied by TADR (model 1) unless
MMS_FLOW_ONLY=1, in which case it is the constant MMS_RHO_REF.
"""

import numpy as np
from proteus import *
from proteus.default_p import *
from proteus.richards import Richards

try:
    from .domain_mms import *
    from . import mms_exact
except ImportError:
    from domain_mms import *
    import mms_exact

LevelModelType = Richards.LevelModel

_one = np.ones(1, "d")
# A flow-only run carries a constant density MMS_RHO_REF in the kernel's mass
# and conductivity; the edge loop scales g by rho/rho_ref, so gravity carries
# the same factor to keep the buoyancy term rho g (T1 comparison).
_rho_ref = mms_exact.RHO_REF if mms_exact.FLOW_ONLY else 1.0

coefficients = Richards.Coefficients(
    nd,
    np.array([[mms_exact.K0, mms_exact.K0]]),   # Ksw_types, one material
    _one * mms_exact.VG_N,
    _one * mms_exact.VG_ALPHA,
    _one * mms_exact.THETA_R,
    _one * mms_exact.THETA_SR,
    gravity=np.array([0.0, -_rho_ref, 0.0]),
    density=_rho_ref,
    beta=0.0,
    density_model=None if mms_exact.FLOW_ONLY else 1,
    diagonal_conductivity=True,
    STABILIZATION_TYPE=2,
    ENTROPY_TYPE=1,
    LUMPED_MASS_MATRIX=False,
    FCT=FCT, FCT_BND_DIFF=True, FCT_BND_KRW=True,  # Barua edge-flux runs: his FCT boundary block (both off by default since #173)
    
    num_fct_iter=0,
    cE=1.0,
    uL=0.0,
    uR=1.0,
    cK=1.0,
    outputQuantDOFs=False,
    mms_source=mms_exact.f_w,
)

initialConditions = {0: FieldIC(mms_exact.psi_e)}
dirichletConditions = {0: dirichlet_everywhere(mms_exact.psi_e)}
fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: no_flux_condition}
diffusiveFluxBoundaryConditions = {0: {0: no_flux_condition}}
