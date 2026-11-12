"""
Physics deck for the m_comp_co2 regression cases (test_m_comp_co2.py).

The case is a Context option, e.g.

    parun m_comp_co2_p.py m_comp_co2_n.py -C "case='segregation'" \\
          -P '-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist'

Direct LU is required: with the KSP default and linTolFac = 1e-3 the inexact
Newton step stalls just above nl_atol_res (open_top stops at t = 2.9 s).

The model is SI: p in Pa, Ksw_types = k/mu_w [m^2/(Pa s)], beta [1/Pa], gravity
[m/s^2]; T = 20 C and fresh brine are fixed in the kernel.  Slot 1 is z, the
overall CO2 mole fraction: z below X_eq ~ 6e-4 is single-phase brine.

Rest and conservation cases, 0.1 m x 1 m column, FCT off:

  static          STAB 0, no gravity, uniform p and z: an exact steady state
  hydrostatic     STAB 2 (BC), hydrostatic brine at rest, closed
  segregation     STAB 2, gas band at the bottom of a closed column, must rise
  open_top        STAB 0, hydrostatic brine with a pressure-Dirichlet top
  open_top_stab2  STAB 2, the same through the EV boundary path

Per-rock nodal closure (proteus/m_comp_co2/Z_ACROSS_ROCK_INTERFACES.md):

  rock_column     0.02 m x 0.2 m strip, C sand / ESF / C sand, closed, free
                  gas starting 0.02 m below the ESF.  The pooled gas column
                  stays below the 0.12 m that would lift it over the ESF entry
                  pressure (1471.5 Pa against 294.3 Pa for C), so with the
                  closure on the ESF interior stays free of gas.
                  Options: per_rock_closure (True), one_rock (False: True
                  makes the ESF layer C sand, so no node is shared by two
                  rocks), T (end time, 1800 s).
"""
from proteus import Context, Domain
from proteus.default_p import *
from proteus.m_comp_co2 import m_comp_co2

import numpy as np

opts = Context.Options([
    ("case", "static", "static, hydrostatic, segregation, open_top, open_top_stab2, rock_column"),
    ("per_rock_closure", True, "rock_column: per-rock nodal closure at rock interfaces"),
    ("one_rock", False, "rock_column: make the ESF layer C sand (no rock interfaces)"),
    ("T", 1800.0, "rock_column: end time [s]; outputs every 300 s"),
])
CASE = opts.case
REST_CASES = ("static", "hydrostatic", "segregation", "open_top", "open_top_stab2")
if CASE not in REST_CASES + ("rock_column",):
    raise ValueError("unknown case {0!r}".format(CASE))

nd = 2
nLevels = 1
name = "m_comp_co2_" + CASE

P_ATM = 1.013e5              # pressure at the top of every column [Pa]
G_MAG = 9.81                 # gravity magnitude [m/s^2]


def getDBC_none(x, flag):
    return None


def getNoFlow(x, flag):
    return lambda x, t: 0.0


if CASE in REST_CASES:
    # ---- 0.1 m x 1 m column, one material ---------------------------------------
    RHO_W_MASS = 998.0       # brine mass density at 20 C, ~1 bar [kg/m^3]
    MU_W = 1.0e-3            # brine dynamic viscosity [Pa s]
    MU_G = 1.5e-5            # CO2 gas dynamic viscosity [Pa s]
    PERM_M2 = 1.0e-11 if CASE == "segregation" else 1.0e-12   # ~10 / ~1 Darcy [m^2]
    PHI = 0.30               # porosity
    S_WR = 0.10              # residual brine saturation
    LX, LY = 0.10, 1.00      # width, height [m]
    Z0 = 1.0e-5              # single-phase brine (z well below X_eq)

    Ksw_types = np.array([[PERM_M2 / MU_W, PERM_M2 / MU_W]], 'd')
    vgm_n_types = np.array([2.5], 'd')
    vgm_alpha_types = np.array([1.0e-4], 'd')     # 1/entry pressure ~ 1e4 Pa [1/Pa]
    thetaR_types = np.array([S_WR * PHI], 'd')
    thetaSR_types = np.array([(1.0 - S_WR) * PHI], 'd')

    domain = Domain.RectangularDomain(L=(LX, LY), x=(0.0, 0.0), name="box")
    gravity = np.array([0.0, 0.0] if CASE == "static" else [0.0, -G_MAG], 'd')

    LevelModelType = m_comp_co2.LevelModel
    coefficients = m_comp_co2.Coefficients(
        nd,
        Ksw_types,
        vgm_n_types,
        vgm_alpha_types,
        thetaR_types,
        thetaSR_types,
        gravity=gravity,
        density=RHO_W_MASS,
        beta=4.5e-10,                 # brine compressibility [1/Pa]
        rho_n=1.8,                    # CO2 ref density (unused by the compositional flux)
        p_ref_n=0.0,
        PSK_TYPE='BC' if CASE == "hydrostatic" else 'VGM',
        diagonal_conductivity=True,
        STABILIZATION_TYPE=0 if CASE in ("static", "open_top") else 2,
        ENTROPY_TYPE=1,
        LUMPED_MASS_MATRIX=False,
        FCT=False,
        num_fct_iter=0,
        cE=1.0, uL=0.0, uR=1.0, cK=1.0,
        outputQuantDOFs=False,
        injection_ports=None,
        mu_n=MU_G / MU_W,
        reconstruct_velocity_rt0=False,
    )

    class IC_p:
        def uOfXT(self, x, t):
            if CASE == "static":
                return P_ATM
            return P_ATM + RHO_W_MASS * G_MAG * (LY - x[1])

    class IC_z:
        # segregation: gas-rich band (z = 2e-3, S_g ~ 0.65) below y = 0.3 m with a
        # tanh transition of half-width 0.04 m to saturated brine (z = 6e-4 ~ X_eq).
        def uOfXT(self, x, t):
            if CASE == "segregation":
                return 6.0e-4 + 0.5 * (2.0e-3 - 6.0e-4) * (1.0 - np.tanh((x[1] - 0.30) / 0.04))
            return Z0

    initialConditions = {0: IC_p(), 1: IC_z()}

    if CASE.startswith("open_top"):
        # Top face: p pinned to P_ATM; every other face and z are no-flow.
        def _on_top(x):
            return x[1] > LY - 1.0e-6

        def getDBC_p(x, flag):
            if _on_top(x):
                return lambda x, t: P_ATM
            return None

        def getNoFlow_p(x, flag):
            if _on_top(x):
                return None
            return lambda x, t: 0.0

        dirichletConditions = {0: getDBC_p, 1: getDBC_none}
        advectiveFluxBoundaryConditions = {0: getNoFlow_p, 1: getNoFlow}
        diffusiveFluxBoundaryConditions = {0: {0: getNoFlow_p, 1: getNoFlow},
                                           1: {0: getNoFlow_p, 1: getNoFlow}}
    else:
        dirichletConditions = {0: getDBC_none, 1: getDBC_none}
        advectiveFluxBoundaryConditions = {0: getNoFlow, 1: getNoFlow}
        diffusiveFluxBoundaryConditions = {0: {0: getNoFlow, 1: getNoFlow},
                                           1: {0: getNoFlow, 1: getNoFlow}}
    fluxBoundaryConditions = {0: 'noFlow', 1: 'noFlow'}

else:
    # ---- rock_column: one-element strip, node rows 5 mm apart ---------------------
    LX = 0.02                # strip width [m]
    H_LAYER = 0.005          # node-row spacing [m]
    Y_SEAL0, Y_SEAL1, Y_TOP = 0.10, 0.14, 0.20
    NROW = int(round(Y_TOP / H_LAYER)) + 1

    ys = np.linspace(0.0, Y_TOP, NROW)
    vertices = [[0.0, y] for y in ys] + [[LX, y] for y in ys]
    # Boundary flags: 1 bottom, 2 right, 3 top, 4 left; 5 the two rock interfaces.
    vertexFlags = [4] * NROW + [2] * NROW
    vertexFlags[0] = vertexFlags[NROW] = 1
    vertexFlags[NROW - 1] = vertexFlags[2 * NROW - 1] = 3
    segments, segmentFlags = [], []
    for k in range(NROW - 1):
        segments += [[k, k + 1], [NROW + k, NROW + k + 1]]
        segmentFlags += [4, 2]
    segments += [[0, NROW], [NROW - 1, 2 * NROW - 1]]
    segmentFlags += [1, 3]
    for yI in (Y_SEAL0, Y_SEAL1):
        k = int(round(yI / H_LAYER))
        segments.append([k, NROW + k])
        segmentFlags.append(5)
    regions = [[0.5 * LX, 0.5 * Y_SEAL0],
               [0.5 * LX, 0.5 * (Y_SEAL0 + Y_SEAL1)],
               [0.5 * LX, 0.5 * (Y_SEAL1 + Y_TOP)]]
    regionFlags = [1, 1 if opts.one_rock else 2, 1]   # material 1 = C sand, 2 = ESF
    domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
                                                  vertexFlags=vertexFlags,
                                                  segments=segments,
                                                  segmentFlags=segmentFlags,
                                                  regions=regions,
                                                  regionFlags=regionFlags)
    domain.writePoly("rock_column")
    # 'p' PSLG, 'A' region attributes, 'e'/'n' edges and neighbours; no 'q'/'a', so
    # no Steiner points: each 5 mm layer is one rectangle split into two triangles.
    triangleOptions = "pAen"

    # Constants and petrophysics: FluidFlower C sand and ESF
    # (test/m_comp_co2/validation_fluid_flower/validation_FF_moose_like/flow_p.py).
    RHO_W = 998.2041322005837
    MU_W = 1.0e-3
    MU_G = 1.48e-5
    gravity = np.array([0.0, -G_MAG, 0.0])

    # key : (k [m^2], lambdaBC, p_d [Pa], thetaR, thetaSR, krn_end, S_gr)
    SAND = {
        "ESF": (4.3427e-11, 2.0, 1471.5, 0.1392, 0.2958, 0.09, 0.14),
        "C":   (4.6681e-10, 2.0, 294.3,  0.0609, 0.3741, 0.05, 0.10),
    }
    MAT_TO_SAND = {0: "C", 1: "C", 2: "ESF"}   # index 0: fallback the kernel reads
    nMat = 3
    Ksw_types = np.zeros((nMat, nd), 'd')
    vgm_n_types = np.zeros(nMat, 'd')
    vgm_alpha_types = np.zeros(nMat, 'd')
    thetaR_types = np.zeros(nMat, 'd')
    thetaSR_types = np.zeros(nMat, 'd')
    krnEnd_types = np.zeros(nMat, 'd')
    S_gr_types = np.zeros(nMat, 'd')
    for m in range(nMat):
        k_m2, lam, p_d, thR, thSR, krn_end, S_gr = SAND[MAT_TO_SAND[m]]
        Ksw_types[m, :] = k_m2 / MU_W
        vgm_n_types[m] = lam
        vgm_alpha_types[m] = 1.0 / p_d
        thetaR_types[m] = thR
        thetaSR_types[m] = thSR
        krnEnd_types[m] = krn_end
        S_gr_types[m] = S_gr

    LevelModelType = m_comp_co2.LevelModel
    coefficients = m_comp_co2.Coefficients(
        nd,
        Ksw_types,
        vgm_n_types,
        vgm_alpha_types,
        thetaR_types,
        thetaSR_types,
        gravity=gravity,
        density=RHO_W,
        beta=4.5e-10,
        rho_n=1.8,
        p_ref_n=0.0,
        PSK_TYPE='BC',                # per_rock_closure needs BC
        diagonal_conductivity=True,
        STABILIZATION_TYPE=2,         # per_rock_closure needs the EV edge path
        ENTROPY_TYPE=1,
        LUMPED_MASS_MATRIX=False,
        FCT=False,
        num_fct_iter=0,
        cE=0.0, uL=0.0, uR=1.0, cK=1.0,
        outputQuantDOFs=False,
        injection_ports=None,
        krn_end_types=krnEnd_types,
        S_gr_types=S_gr_types,
        mu_n=MU_G / MU_W,
        reconstruct_velocity_rt0=False,
        per_rock_closure=opts.per_rock_closure,
    )

    Y_GAS = 0.08             # top of the initial gas band [m], 0.02 m below the ESF
    Z_GAS = 1.5e-3           # z of the gas band: above X_eq ~ 6e-4, so free gas

    class IC_p:
        """Hydrostatic brine pressure with P_ATM at the top."""
        def uOfXT(self, x, t):
            return P_ATM + RHO_W * G_MAG * (Y_TOP - x[1])

    class IC_z:
        """Gas band below Y_GAS, CO2-free brine above."""
        def uOfXT(self, x, t):
            return Z_GAS if x[1] < Y_GAS + 1.0e-9 else 0.0

    initialConditions = {0: IC_p(), 1: IC_z()}

    # Closed box: no Dirichlet, no-flow on every face, so CO2 and H2O are conserved.
    dirichletConditions = {0: getDBC_none, 1: getDBC_none}
    fluxBoundaryConditions = {0: 'noFlow', 1: 'noFlow'}
    advectiveFluxBoundaryConditions = {0: getNoFlow, 1: getNoFlow}
    diffusiveFluxBoundaryConditions = {0: {0: getNoFlow, 1: getNoFlow},
                                       1: {0: getNoFlow, 1: getNoFlow}}
