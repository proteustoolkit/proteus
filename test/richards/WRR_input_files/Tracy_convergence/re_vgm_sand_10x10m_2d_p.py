from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
from proteus import Context

# Scheme selection.  The defaults are the FCT scheme; the other two are
#   Galerkin              -C "STABILIZATION_TYPE=0 FCT=False"
#   entropy viscosity     -C "STABILIZATION_TYPE=2 FCT=False"
# nnx drives the spatial refinement (used in the _n.py file).
ct = Context.Options([
    ("STABILIZATION_TYPE", 2, "0: Galerkin (no stabilization), 2: entropy viscosity"),
    ("FCT", True, "apply the flux-corrected-transport limiter"),
    ("nnx", 161, "nodes per side: 41/81/161/321/641 = ref_2../ref_6"),
    # IIPG weak-Dirichlet penalty: proteus forms  penalty = penalty_constant / h^power,
    # so the h-scaling is already there and this constant is held FIXED across a
    # refinement sweep.  Nitsche needs penalty >= C*a/h with C above the inverse-trace
    # constant; here a = rho*k_rw*Ks <= Ks = 2.04, and the default penalty_constant=2.0
    # gives C ~ 1 at the wet top -- right at the stability threshold, which shows up as
    # an O(h) violation of the top Dirichlet value.
    ("penalty", 2.0, "penalty_constant for the weak Dirichlet BC (default proteus 2.0)"),
], mutable=True)

nd = 2

L=(10.0,10.0,1.0)
regularGrid=False
domain = Domain.RectangularDomain(L[:2])
if not regularGrid:
    domain.writePoly('sand2d')
    domain = Domain.PlanarStraightLineGraphDomain('sand2d')

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0#density*gravity*4.524e-10
m_per_s_by_m_per_d = 1.1574074e-5
permeability  = (2.04*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #m^2
thetaS        = 0.301   #-
thetaR        = 0.093   #-
mvg_alpha     = 0.164 #0.5 #5.47    #1/m
mvg_n         = 4.264
mvg_m         = 1.0 - 1.0/mvg_n
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)
#make non-dimensional
dimensionless_conductivity  = (timeScale*density*gravity*permeability/(viscosity*lengthScale))/m_per_s_by_m_per_d
#print 'Ks',dimensionless_conductivity
dimensionless_density  = 1.0
dimensionless_gravity  = numpy.array([0.0,
                                        -1.0,
                                        0.0])
dimensionless_alpha    = mvg_alpha*lengthScale
nMediaTypes  = 1
alphaVGtypes = numpy.zeros((nMediaTypes+1,),'d')
nVGtypes     = numpy.zeros((nMediaTypes+1,),'d')
thetaStypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes+1,),'d')
KsTypes      = numpy.zeros((nMediaTypes+1,2),'d')

for i in range(nMediaTypes+1):
    alphaVGtypes[i] = mvg_alpha
    nVGtypes[i]     = mvg_n
    thetaStypes[i]  = thetaS
    thetaRtypes[i]  = thetaR
    thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
    KsTypes[i,:]    = [dimensionless_conductivity,dimensionless_conductivity]#m/d?

LevelModelType = Richards.LevelModel
coefficients = Richards.Coefficients(nd,
                                     KsTypes,
                                     nVGtypes,
                                     alphaVGtypes,
                                     thetaRtypes,
                                     thetaSRtypes,
                                     gravity=dimensionless_gravity,
                                     density=dimensionless_density,
                                     beta=0.0, #0.0001,
                                     diagonal_conductivity=True,
                                     PSK_type='Gardner',
                                     penalty_constant=ct.penalty,
                                     penalty_power=1.0,
                                     STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX=False,
                                     FCT=ct.FCT,
                                     num_fct_iter=0,
                                     # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False)
galerkin = False
# coefficients = ConservativeHeadRichardsMualemVanGenuchten(hydraulicConductivity=dimensionless_conductivity,
#                                                           gravity=dimensionless_gravity,
#                                                           density=dimensionless_density,
#                                                           thetaS=thetaS,
#                                                           thetaR=thetaR,
#                                                           alpha= dimensionless_alpha,
#                                                           n = mvg_n,
#                                                           m = mvg_m,
#                                                           beta = beta)

pondingPressure = 0.1
bottomPressure = 0.0
# ---- Tracy analytical BC/IC parameters (top head forcing) ----
psi_r   = -15.24      # reference head used in Tracy (choose your value, e.g., -2.0)
alpha_g = 0.164      # Gardner alpha [1/m] used in the analytical BC formula
beta_bc = 1.0      # scaling (>1 => "higher" top head); optional convenience
hat_h0  = beta_bc * (1.0 - numpy.exp(alpha_g*psi_r))  # amplitude in transformed variable

a_span  = L[0]     # domain width 'a' (here = 10.0)


#import numpy

def tracy_transient(nodes, t_days, *,
                    a, L, alpha, psi_r,
                    theta_s, theta_r, Ks_day,
                    n_terms=120):
    """
    Transient Tracy solution h(x,z,t) for each node.

    nodes   : array (N,2) with x=nodes[:,0], z=nodes[:,1]
    t_days  : time in days corresponding to this snapshot
    a, L    : domain width and height (scalars)
    alpha   : Gardner alpha [1/m]
    psi_r   : initial pressure head
    theta_s, theta_r : saturations
    Ks_day  : saturated K [m/day]
    """
    x = nodes[:, 0]
    z = nodes[:, 1]

    # --- parameters (Table 3 style) ---
    h0 = 1.0 - numpy.exp(alpha * psi_r)        # \bar h_0
    Ks = Ks_day / 86400.0                      # m/s
    c  = alpha * (theta_s - theta_r) / Ks      # 1/s
    t  = t_days * 86400.0                      # s

    # eigenvalues and decay rates
    k_idx  = numpy.arange(1, n_terms + 1, dtype=float)  # k = 1..n_terms
    lam    = k_idx * numpy.pi / L                       # λ_k
    gamma1 = (lam**2 + (alpha**2) / 4.0) / c           # γ_1
    gamma2 = ((2.0 * numpy.pi / a)**2 + lam**2 + (alpha**2) / 4.0) / c  # γ_2

    psi_tr = numpy.zeros_like(x)

    # precompute constant root for steady term
    root_ss = numpy.sqrt((alpha / 2.0)**2 + (2.0 * numpy.pi / a)**2)

    for i in range(len(x)):
        xi, zi = x[i], z[i]
        if abs(zi - L) < 1e-12:
            psi_tr[i] = (1.0 / alpha) * numpy.log(
                numpy.exp(alpha * psi_r)
                + 0.5 * h0 * (1.0 - numpy.cos(2.0 * numpy.pi * xi / a))
            )
            continue

        # --- steady transformed part \bar h_ss(x,z) ---
        h_bar_ss = (
            0.5 * h0
            * numpy.exp(0.5 * alpha * (L - zi))
            * (
                numpy.sinh(0.5 * alpha * zi) / numpy.sinh(0.5 * alpha * L)
                - numpy.cos(2.0 * numpy.pi * xi / a)
                  * numpy.sinh(root_ss * zi) / numpy.sinh(root_ss * L)
            )
        )

        # --- transient correction \bar\phi(x,z,t) ---
        s = 0.0
        for j in range(n_terms):
            k     = k_idx[j]
            lam_k = lam[j]
            g1    = gamma1[j]
            g2    = gamma2[j]

            term = ((-1.0)**k) * lam_k * (
                (1.0 / g1) * numpy.exp(-g1 * t)
                - (1.0 / g2) * numpy.cos(2.0 * numpy.pi * xi / a) * numpy.exp(-g2 * t)
            ) * numpy.sin(lam_k * zi)

            s += term

        phi_bar = (h0 / (L * c)) * numpy.exp(0.5 * alpha * (L - zi)) * s

        # total transformed variable and back-transform to pressure head
        h_bar = h_bar_ss + phi_bar
        psi_tr[i] = (1.0 / alpha) * numpy.log(numpy.exp(alpha * psi_r) + h_bar)

    return psi_tr

Ks_day = 2.04  # physical Ks in m/day that matches your permeability

def tracy_exact_point(x, t):
    """
    x : [x,z] (meters)
    t : Proteus time (dimensionless). We convert to days via timeScale.
    """
    t_days = t * timeScale
    nodes = numpy.array([[x[0], x[1]]])  # shape (1,2)

    psi_vals = tracy_transient(
        nodes,
        t_days,
        a=L[0],             # domain width  = 10.0
        L=L[1],             # domain height = 10.0
        alpha=mvg_alpha,    # 0.164
        psi_r=psi_r,
        theta_s=thetaS,
        theta_r=thetaR,
        Ks_day=Ks_day,
        n_terms=120,
    )
    return float(psi_vals[0])


eps= 1e-10
def getDBC_Tracy(x, flag= None):
    # Top boundary: prescribed head (Tracy sinusoidal head)
    if abs(x[1] - L[1]) < 1e-12:
        return lambda x, t: (1.0/alpha_g)*numpy.log(
            numpy.exp(alpha_g*psi_r) + 0.5*hat_h0*(1.0 - numpy.cos(2.0*numpy.pi*x[0]/a_span))
        )
    # Bottom boundary: fixed at psi_r
    if abs(x[1] - 0.0) < 1e-12:
        return lambda x, t: psi_r
    # Sides: NO Dirichlet (remain Neumann no-flow)
    # --- SIDE boundaries: x = 0 or x = 10 use full Tracy transient solution ---
    if abs(x[0] - 0.0) < eps or abs(x[0] - L[0]) < eps:
        return lambda x, t: tracy_exact_point(x, t)
#    return None

dirichletConditions = {0: getDBC_Tracy}

class TracyIC:
    def uOfXT(self, x, t):
        bc = getDBC_Tracy(x, 0)
        if bc != None:
            return bc(x, t)
        else:
            return psi_r

initialConditions = {0: TracyIC()}

fluxBoundaryConditions = {0:'noFlow'}

def getFBC_2D_Richards_Shock(x,flag):
   return None
#    if x[1] == L[1]:
#        if (x[0] < L[0]/3.0 or
#            x[0] > 2.0*L[0]/3.0):
#            return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getFBC_2D_Richards_Shock}

diffusiveFluxBoundaryConditions = {0:{0:getFBC_2D_Richards_Shock}}

#T = 0.05/timeScale

T = 0.001/2/timeScale
