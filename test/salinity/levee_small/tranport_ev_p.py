"""The levee deck's transport on an edge-based TADR scheme that advects with Richards' edge fluxes
(LEVEE_STAB: 2-4 explicit SSP stages of order 1, lumped mass; 5 ImplicitEV, backward Euler and Newton), as
test_vdft_compatible.py::_edge_flux_levee_transport sets it. C14 note 3.5 rows a, b, d."""
import os

from tranport_p import *  # noqa: F401,F403

STAB = int(os.environ.get("LEVEE_STAB", "4"))
coefficients = MyCoefficients(aOfX, alpha_L=alpha_L, alpha_T=alpha_T, Dm=Dm, porosity=0.41, V_model=0,
                              specified_velocity=False, checkMass=False, FCT=True,
                              LUMPED_MASS_MATRIX=STAB in (2, 3, 4), STABILIZATION_TYPE=STAB,
                              diagonal_conductivity=True, rho_f=1.0, rho_s=1.025)
coefficients.variableNames = ["u"]
