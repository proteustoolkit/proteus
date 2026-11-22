import numpy as np
from proteus import Domain
from proteus import Context
from proteus import SpatialTools as st
from proteus.mbd import CouplingFSI as fsi
import pychrono
import os
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
import proteus.TwoPhaseFlow.utils.Parameters as Parameters

addedMass = False

#   ____            _            _      ___        _   _
#  / ___|___  _ __ | |_ _____  _| |_   / _ \ _ __ | |_(_) ___  _ __  ___
# | |   / _ \| '_ \| __/ _ \ \/ / __| | | | | '_ \| __| |/ _ \| '_ \/ __|
# | |__| (_) | | | | ||  __/>  <| |_  | |_| | |_) | |_| | (_) | | | \__ \
#  \____\___/|_| |_|\__\___/_/\_\\__|  \___/| .__/ \__|_|\___/|_| |_|___/
#                                           |_|
# Context options
# used in command line directly with option -C
# e.g.: parun [...] -C "g=(0.,-9.81,0.) rho_0=998.2 genMesh=False"
#
# only change/add context options in the "other options section"
# other sections have variables used in _p and _n files

context_options = []
# physical constants
context_options += [
    ("densityA", 998.2, "Water density"),
    ("densityB", 1.205, "Air Densiy"),
    ("kinematicViscosityA", 1.004e-6, "Water kinematic viscosity m/sec^2"),
    ("kinematicViscosityB", 1.5e-5, "Air kinematic viscosity m/sec^2"),
    ("surf_tension_coeff", 0., "Surface tension"),
    ("gravity", (0, -9.81, 0.), "Gravitational acceleration vector"),
    ]
# run time options
context_options += [
    ("T", 1., "Simulation time in s"),
    ("dt_init", 0.001 ,"Value of initial time step"),
    ("dt_fixed", 0.005, "Value of maximum time step"),
    ("archiveAllSteps", False, "archive every steps"),
    ("dt_output", 0.005, "number of saves per second"),
    ("cfl", 0.1 , "Target CFL value"),
    ]
# mesh options
context_options += [
    ("genMesh", False, "Generate the mesh, otherwise use exect mesh exists"),
    ("he", 0.1, "Characteristic element size"),
    ]
# other options
context_options += [
    ("radius", 0.5, "Radius of cylinder"),
    ("use_ball_as_particle", 0., "use ball method"),
    ]

opts = Context.Options(context_options)
he = opts.he
radius = opts.radius

#  ____                        _
# |  _ \  ___  _ __ ___   __ _(_)_ __
# | | | |/ _ \| '_ ` _ \ / _` | | '_ \
# | |_| | (_) | | | | | | (_| | | | | |
# |____/ \___/|_| |_| |_|\__,_|_|_| |_|
# Domain
# All geometrical options go here (but not mesh options)

domain = Domain.PlanarStraightLineGraphDomain()

# ----- SHAPES ----- #

# TANK
tank_dim = np.array([3., 10.])
tank_x = tank_dim[0]
tank_y = tank_dim[1]
water_level=tank_y*2
tank = st.Tank2D(domain, [tank_x, tank_y])

# MESH PARAMETERS
domain.MeshOptions.genMesh = opts.genMesh
domain.MeshOptions.he = opts.he

#  ____                        _                   ____                _ _ _   _
# | __ )  ___  _   _ _ __   __| | __ _ _ __ _   _ / ___|___  _ __   __| (_) |_(_) ___  _ __  ___
# |  _ \ / _ \| | | | '_ \ / _` |/ _` | '__| | | | |   / _ \| '_ \ / _` | | __| |/ _ \| '_ \/ __|
# | |_) | (_) | |_| | | | | (_| | (_| | |  | |_| | |__| (_) | | | | (_| | | |_| | (_) | | | \__ \
# |____/ \___/ \__,_|_| |_|\__,_|\__,_|_|   \__, |\____\___/|_| |_|\__,_|_|\__|_|\___/|_| |_|___/
#                                           |___/
# Boundary Conditions


#tank.BC['y+'].setFreeSlip()
tank.BC['y+'].setAtmosphere()
#tank.BC['y-'].setFreeSlip()
#tank.BC['x+'].setFreeSlip()
#tank.BC['x-'].setFreeSlip()
#tank.BC['y+'].setNoSlip()
tank.BC['y-'].setNoSlip()
tank.BC['x+'].setNoSlip()
tank.BC['x-'].setNoSlip()
#tank.BC['sponge'].setNonMaterial()


#   ____ _
#  / ___| |__  _ __ ___  _ __   ___
# | |   | '_ \| '__/ _ \| '_ \ / _ \
# | |___| | | | | | (_) | | | | (_) |
#  \____|_| |_|_|  \___/|_| |_|\___/
# Chrono
g = np.array(opts.gravity)

# System
system = fsi.ProtChSystem()
system.ChSystem.SetGravitationalAcceleration(pychrono.ChVector3d(g[0], g[1], g[2]))
system.setTimeStep(1e-5)
# Body
body = fsi.ProtChBody(system=system)
body.setBoundaryFlags([0])  # index of particle
#
chbod = body.ChBody
pos = pychrono.ChVector3d(0.5*tank_x, 0.7*tank_y, 0.)
rot = pychrono.ChQuaterniond(1., 0., 0., 0.)
mass = 2000.0#3.14*rho_0*1.5
inertia = pychrono.ChVector3d(1., 1., 1.)
chbod.SetPos(pos)
chbod.SetRot(rot)
chbod.SetMass(mass)
chbod.SetInertiaXX(inertia)
body.setConstraints(free_x=np.array([0., 1., 0.]), free_r=np.array([0., 0., 0.]))
body.setRecordValues(all_values=True)

def sdf(t, x):
    dist = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    if dist < 1.0e-8:
        return -radius, (0., 1., 0.)
    return dist - radius, (x[0]/dist, x[1]/dist, x[2]/dist)

body.setIBM(True,
            radiusIBM=radius,  # used only when particle are balls
            sdfIBM=sdf,  # used only when particle not used as balls
)


#  ___       _ _   _       _    ____                _ _ _   _
# |_ _|_ __ (_) |_(_) __ _| |  / ___|___  _ __   __| (_) |_(_) ___  _ __  ___
#  | || '_ \| | __| |/ _` | | | |   / _ \| '_ \ / _` | | __| |/ _ \| '_ \/ __|
#  | || | | | | |_| | (_| | | | |__| (_) | | | | (_| | | |_| | (_) | | | \__ \
# |___|_| |_|_|\__|_|\__,_|_|  \____\___/|_| |_|\__,_|_|\__|_|\___/|_| |_|___/
# Initial Conditions

from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.ctransportCoefficients import smoothedHeaviside_integral
smoothing = 1.5 * he
nd = domain.nd

class P_IC:
    def uOfXT(self, x, t):
        p = -opts.densityA*g[nd-1]*(tank_dim[1] - x[1])
        return p
class U_IC:
    def uOfXT(self, x, t):
        return 0.0
class V_IC:
    def uOfXT(self, x, t):
        return 0.0
class W_IC:
    def uOfXT(self, x, t):
        return 0.0
class AtRest:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

#  __  __           _        ___        _   _
# |  \/  | ___  ___| |__    / _ \ _ __ | |_(_) ___  _ __  ___
# | |\/| |/ _ \/ __| '_ \  | | | | '_ \| __| |/ _ \| '_ \/ __|
# | |  | |  __/\__ \ | | | | |_| | |_) | |_| | (_) | | | \__ \
# |_|  |_|\___||___/_| |_|  \___/| .__/ \__|_|\___/|_| |_|___/
#                                |_|


domain.MeshOptions.use_gmsh = False
modulepath = os.path.dirname(os.path.abspath(__file__))
mesh_fileprefix=modulepath+'/meshFallingCylinder'
domain.MeshOptions.setOutputFiles(mesh_fileprefix)
st.assembleDomain(domain)
domain.use_gmsh = False
domain.geofile = mesh_fileprefix

domain.BCbyFlag[0].u_diffusive.uOfXT = lambda x, t: 0.
domain.BCbyFlag[0].v_diffusive.uOfXT = lambda x, t: 0.

#  _   _                           _
# | \ | |_   _ _ __ ___   ___ _ __(_) ___ ___
# |  \| | | | | '_ ` _ \ / _ \ '__| |/ __/ __|
# | |\  | |_| | | | | | |  __/ |  | | (__\__ \
# |_| \_|\__,_|_| |_| |_|\___|_|  |_|\___|___/
# Numerics

myTpFlowProblem = TpFlow.TwoPhaseFlowProblem()
myTpFlowProblem.outputStepping.final_time = opts.T
myTpFlowProblem.outputStepping.dt_init = opts.dt_init
myTpFlowProblem.outputStepping.dt_output = opts.dt_output
myTpFlowProblem.outputStepping.dt_fixed = opts.dt_fixed
myTpFlowProblem.outputStepping.archiveAllSteps = opts.archiveAllSteps

myTpFlowProblem.domain = domain

myTpFlowProblem.SystemNumerics.useSuperlu=True
myTpFlowProblem.SystemNumerics.useExact=True
myTpFlowProblem.SystemNumerics.cfl=opts.cfl

# line below needed for relaxation zones
# (!) hack

myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelRANS2P,'flow')
if addedMass is True:
    myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelAddedMass,'addedMass')

m = myTpFlowProblem.SystemPhysics.modelDict
m['flow'].auxiliaryVariables += domain.auxiliaryVariables['twp']

myTpFlowProblem.SystemPhysics.movingDomain = False

params = myTpFlowProblem.SystemPhysics

#initialConditions
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['p']=P_IC()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['u']=U_IC()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['v']=V_IC()
if addedMass is True:
    myTpFlowProblem.SystemPhysics.modelDict['addedMass'].p.initialConditions['addedMass']=AtRest()


# PHYSICAL PARAMETERS

params['rho_0'] = opts.densityA  # water
params['rho_1'] = opts.densityB  # air
params['nu_0'] = opts.kinematicViscosityA  # water
params['nu_1'] = opts.kinematicViscosityB  # air
params['gravity'] = np.array(opts.gravity)
params['surf_tension_coeff'] = 0.

m['flow'].auxiliaryVariables += [system]
m['flow'].p.coefficients.use_ball_as_particle = opts.use_ball_as_particle
m['flow'].p.coefficients.nParticles = 1
m['flow'].n.conservativeFlux=None
