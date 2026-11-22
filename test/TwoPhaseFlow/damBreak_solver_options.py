"""
danbreak 2-D
"""
import numpy as np
from proteus import (Domain, Context, LinearSolvers)
from proteus.Profiling import logEvent
from proteus.SpatialTools import Tank2D
from proteus import SpatialTools as st
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
from proteus.Gauges import PointGauges, LineIntegralGauges, LineGauges
import os

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts= Context.Options([
    ("final_time",3.0,"Final time for simulation"),
    ("dt_output",0.01,"Time interval to output solution"),
    ("cfl",0.25,"Desired CFL restriction"),
    ("he",0.01,"he relative to Length of domain in x"),
    ("refinement",3,"level of refinement")
    ])

# ****************** #
# ***** GAUGES ***** #
# ****************** #
height_gauges1 = LineGauges(gauges=((("phi",),
                                        (((2.724, 0.0, 0.0),
                                          (2.724, 1.8, 0.0)), # We consider this one in our paper
                                         ((2.228, 0.0, 0.0),
                                          (2.228, 1.8, 0.0)), # We consider this one in our paper
                                         ((1.732, 0.0, 0.0),
                                          (1.732, 1.8, 0.0)),
                                         ((0.582, 0.0, 0.0),
	                                  (0.582, 1.8, 0.0)))),),
                                        fileName="height1.csv")

height_gauges2 = LineGauges(gauges=((("phi",),
                                     (((0.0, 0.0, 0.0),
                                       (0.0, 0.0, -0.01)),
                                      ((0.0, 0.0, 0.0),
                                        (3.22, 0.0, 0.0)))),),
                            fileName="height2.csv")

pressure_gauges = PointGauges(gauges=((('p',),
                                      ((3.22, 0.16, 0.0), #P1
                                       (3.22, 0.584, 0.0), #P3
                                       (3.22, 0.12, 0.0))),), # This is the one considered in our paper
                                       fileName="pressure.csv")


# *************************** #
# ***** DOMAIN AND MESH ***** #
# ****************** #******* #
tank_dim = (3.22,1.8)
refinement = opts.refinement
structured=False
if structured:
    nny = 5*(2**refinement)+1
    nnx = 2*(nnx-1)+1
    domain = Domain.RectangularDomain(tank_dim)
    boundaryTags = domain.boundaryTags
    triangleFlag=1
else:
    nnx = nny = None
    domain = Domain.PlanarStraightLineGraphDomain()

# ----- TANK ----- #
tank = Tank2D(domain, tank_dim)

# ----- EXTRA BOUNDARY CONDITIONS ----- #
tank.BC['y+'].setAtmosphere()
tank.BC['y-'].setFreeSlip()
tank.BC['x+'].setFreeSlip()
tank.BC['x-'].setFreeSlip()

he = tank_dim[0]*opts.he
domain.MeshOptions.he = he
st.assembleDomain(domain)
domain.polyfile=os.path.dirname(os.path.abspath(__file__))+"/"+"meshDambreak"
domain.MeshOptions.triangleOptions = "VApq30Dena%8.8f" % ((he**2)/2.0,)

# ****************************** #
# ***** INITIAL CONDITIONS ***** #
# ****************************** #
class zero(object):
    def uOfXT(self,x,t):
        return 0.0

waterLine_y = 0.6
waterLine_x = 1.2
class clsvof_init_cond(object):
    def uOfXT(self,x,t):
        if x[0] < waterLine_x and x[1] < waterLine_y:
            return -1.0
        elif x[0] > waterLine_x or x[1] > waterLine_y:
            return 1.0
        else:
            return 0.0

############################################
# ***** Create myTwoPhaseFlowProblem ***** #
############################################
outputStepping = TpFlow.OutputStepping(opts.final_time,dt_output=opts.dt_output)
outputStepping.systemStepExact = True
initialConditions = {'pressure': zero(),
                     'pressure_increment': zero(),
                     'vel_u': zero(),
                     'vel_v': zero(),
                     'clsvof': clsvof_init_cond(),
                     'vof': clsvof_init_cond(),
                     'ncls': zero(),
                     'vof_DBC': zero()}
boundaryConditions = {
    # DIRICHLET BCs #
    'pressure_DBC': lambda x, flag: domain.bc[flag].p_dirichlet.init_cython(),
    'pressure_increment_DBC': lambda x, flag: domain.bc[flag].pInc_dirichlet.init_cython(),
    'vel_u_DBC': lambda x, flag: domain.bc[flag].u_dirichlet.init_cython(),
    'vel_v_DBC': lambda x, flag: domain.bc[flag].v_dirichlet.init_cython(),
    'vel_w_DBC': lambda x, flag: domain.bc[flag].w_dirichlet.init_cython(),
    'clsvof_DBC': lambda x, flag: domain.bc[flag].vof_dirichlet.init_cython(),
    # ADVECTIVE FLUX BCs #
    'pressure_AFBC': lambda x, flag: domain.bc[flag].p_advective.init_cython(),
    'pressure_increment_AFBC': lambda x, flag: domain.bc[flag].pInc_advective.init_cython(),
    'vel_u_AFBC': lambda x, flag: domain.bc[flag].u_advective.init_cython(),
    'vel_v_AFBC': lambda x, flag: domain.bc[flag].v_advective.init_cython(),
    'vel_w_AFBC': lambda x, flag: domain.bc[flag].w_advective.init_cython(),
    'clsvof_AFBC': lambda x, flag: domain.bc[flag].vof_advective.init_cython(),
    # DIFFUSIVE FLUX BCs #
    'pressure_increment_DFBC': lambda x, flag: domain.bc[flag].pInc_diffusive.init_cython(),
    'vel_u_DFBC': lambda x, flag: domain.bc[flag].u_diffusive.init_cython(),
    'vel_v_DFBC': lambda x, flag: domain.bc[flag].v_diffusive.init_cython(),
    'vel_w_DFBC': lambda x, flag: domain.bc[flag].w_diffusive.init_cython(),
    'clsvof_DFBC': lambda x, flag: None}

myTpFlowProblem = TpFlow.TwoPhaseFlowProblem(ns_model=0,
                                             ls_model=1,
                                             nd=2,
                                             cfl=opts.cfl,
                                             outputStepping=outputStepping,
                                             structured=structured,
                                             he=he,
                                             nnx=nnx,
                                             nny=nny,
                                             nnz=None,
                                             domain=domain,
                                             initialConditions=initialConditions,
                                             boundaryConditions=boundaryConditions,
                                             useSuperlu=False)
myTpFlowProblem.Parameters.physical['gravity'] = np.array([0.0,-9.8,0.0])
myTpFlowProblem.useBoundaryConditionsModule = False

m = myTpFlowProblem.Parameters.Models
m.clsvof.p.coefficients.disc_ICs = True
m.clsvof.auxiliaryVariables = [height_gauges1, height_gauges2]
m.pressure.auxiliaryVariables = [pressure_gauges]
m.rans3p.n.ShockCapturingOptions.shockCapturingFactor = 0.5

myTpFlowProblem.Parameters.Models.rans2p.n.linearSmoother = LinearSolvers.NavierStokes_TwoPhasePCD
myTpFlowProblem.Parameters.Models.rans2p.n.linearSmootherOptions = (False, # density_scaling
                                                                    True,  # numerical_viscosity
                                                                    True,  # lumped mass
                                                                    0,     # num_chebyshev_its
                                                                    True,  # laplace_null_space
                                                                    True)  # velocity_block_preconditioner
prefix = myTpFlowProblem.Parameters.Models.rans2p.n.linear_solver_options_prefix
myTpFlowProblem.Parameters.Models.rans2p.OptDB.setValue(prefix+'ksp_atol', 1e-20)
myTpFlowProblem.Parameters.Models.rans2p.OptDB.setValue(prefix+'ksp_rtol', 1e-9)
myTpFlowProblem.Parameters.Models.rans2p.p.coefficients.NONCONSERVATIVE_FORM=0.0
myTpFlowProblem.Parameters.Models.rans2p.p.coefficients.useVF=1.0
myTpFlowProblem.Parameters.Models.rans2p.p.coefficients.eb_penalty_constant = 1e6
myTpFlowProblem.Parameters.Models.rans2p.n.nl_atol_res = 1e-9

myTpFlowProblem.Parameters.mesh.he = he
myTpFlowProblem.Parameters.mesh.triangleOptions = "VApq30Dena%8.8f" % ((he**2)/2.0,)
myTpFlowProblem.Parameters.mesh.genMesh=False
