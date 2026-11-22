import numpy as np
from proteus import Domain, Context, Comm
from proteus import SpatialTools as st
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
import proteus.TwoPhaseFlow.utils.Parameters as Parameters
from proteus import WaveTools as wt
from proteus.Profiling import logEvent
from proteus.mbd import CouplingFSI as fsi
import os
import pychrono

rho_0 = 998.2
nu_0 = 1.004e-6
rho_1 = 1.205
nu_1 = 1.5e-5
sigma_01 = 0.
he = 0.2
tank_dim = [1., 1., 1.]
water_level = 0.5
genMesh = False
rhor = 0.5


#  ____                        _
# |  _ \  ___  _ __ ___   __ _(_)_ __
# | | | |/ _ \| '_ ` _ \ / _` | | '_ \
# | |_| | (_) | | | | | | (_| | | | | |
# |____/ \___/|_| |_| |_|\__,_|_|_| |_|
# Domain
# All geometrical options go here (but not mesh options)

domain = Domain.PiecewiseLinearComplexDomain()

# ----- SHAPES ----- #

# TANK
tank = st.Tank3D(domain, tank_dim)

# CAISSON
radius = 0.1
caisson = st.Cuboid(domain,
                    dim=[2*radius, 2*radius, 2*radius],
                    coords=(tank_dim[0]/2., tank_dim[1]/2., water_level+radius/10.),
                    barycenter=(tank_dim[0]/2., tank_dim[1]/2., water_level+radius/10.))
caisson.setHoles([caisson.barycenter])
caisson.holes_ind = np.array([0])
# let gmsh know that the caisson is IN the tank
tank.setChildShape(caisson, 0)


#  ____                        _                   ____                _ _ _   _
# | __ )  ___  _   _ _ __   __| | __ _ _ __ _   _ / ___|___  _ __   __| (_) |_(_) ___  _ __  ___
# |  _ \ / _ \| | | | '_ \ / _` |/ _` | '__| | | | |   / _ \| '_ \ / _` | | __| |/ _ \| '_ \/ __|
# | |_) | (_) | |_| | | | | (_| | (_| | |  | |_| | |__| (_) | | | | (_| | | |_| | (_) | | | \__ \
# |____/ \___/ \__,_|_| |_|\__,_|\__,_|_|   \__, |\____\___/|_| |_|\__,_|_|\__|_|\___/|_| |_|___/
#                                           |___/
# Boundary Conditions

tank.BC['z+'].setAtmosphere()
tank.BC['z-'].setFreeSlip()
tank.BC['y+'].setFreeSlip()
tank.BC['y-'].setFreeSlip()
tank.BC['x+'].setFreeSlip()
tank.BC['x-'].setFreeSlip()
tank.BC['sponge'].setNonMaterial()

for tag, bc in caisson.BC.items():
    bc.setNoSlip()

for tag, bc in tank.BC.items():
    bc.setFixedNodes()


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
        p_L = 0.0
        phi_L = tank_dim[nd-1] - water_level
        phi = x[nd-1] - water_level
        p = p_L -g[nd-1]*(rho_0*(phi_L - phi)
                          +(rho_1 -rho_0)*(smoothedHeaviside_integral(smoothing,phi_L)
                                                -smoothedHeaviside_integral(smoothing,phi)))
        return p

class Zero_IC:
    def uOfXT(self, x, t):
        return 0.0

class U_IC:
    def uOfXT(self, x, t):
        return 0.0
class V_IC:
    def uOfXT(self, x, t):
        return 0.0
class W_IC:
    def uOfXT(self, x, t):
        return 0.0
class VF_IC:
    def uOfXT(self, x, t):
        return smoothedHeaviside(smoothing,x[nd-1]-water_level)
class PHI_IC:
    def uOfXT(self, x, t):
        return x[nd-1] - water_level

#   ____ _
#  / ___| |__  _ __ ___  _ __   ___
# | |   | '_ \| '__/ _ \| '_ \ / _ \
# | |___| | | | | | (_) | | | | (_) |
#  \____|_| |_|_|  \___/|_| |_|\___/
# Chrono

# System
g = np.array([0., 0., -9.81])
system = fsi.ProtChSystem()
system.ChSystem.SetGravitationalAcceleration(pychrono.ChVector3d(g[0], g[1], g[2]))
system.setTimeStep(1e-5)
#system.setCouplingScheme("CSS", prediction="backwardEuler")
# Body
body = fsi.ProtChBody(system=system)
body.attachShape(caisson)
#body.Aij_factor = 1/width
chbod = body.ChBody
x, y, z = caisson.barycenter
pos = pychrono.ChVector3d(x, y, z)
mass = (2.*radius)**3*rho_0*rhor
inertia = pychrono.ChVector3d(1., 1., 1.)
chbod.SetPos(pos)
chbod.SetMass(mass)
chbod.SetInertiaXX(inertia)
#chbod.SetBodyFixed(True)
body.setConstraints(free_x=np.array([1.,1.,1.]), free_r=np.array([1.,1.,1.]))

# body.setInitialRot(rotation_init)
# body.rotation_init=np.array([np.cos(ang/2.), 0., 0., np.sin(ang/2.)*1.])
body.setRecordValues(all_values=True)

#  __  __           _        ___        _   _
# |  \/  | ___  ___| |__    / _ \ _ __ | |_(_) ___  _ __  ___
# | |\/| |/ _ \/ __| '_ \  | | | | '_ \| __| |/ _ \| '_ \/ __|
# | |  | |  __/\__ \ | | | | |_| | |_) | |_| | (_) | | | \__ \
# |_|  |_|\___||___/_| |_|  \___/| .__/ \__|_|\___/|_| |_|___/
#                                |_|


domain.MeshOptions.use_gmsh = genMesh
domain.MeshOptions.genMesh = genMesh
he = he
domain.MeshOptions.he = he
modulepath = os.path.dirname(os.path.abspath(__file__))
mesh_fileprefix=modulepath+'/meshFloatingCube'
domain.MeshOptions.setOutputFiles(mesh_fileprefix)
st.assembleDomain(domain)
domain.use_gmsh = False
domain.geofile = mesh_fileprefix


#  _   _                           _
# | \ | |_   _ _ __ ___   ___ _ __(_) ___ ___
# |  \| | | | | '_ ` _ \ / _ \ '__| |/ __/ __|
# | |\  | |_| | | | | | |  __/ |  | | (__\__ \
# |_| \_|\__,_|_| |_| |_|\___|_|  |_|\___|___/
# Numerics

myTpFlowProblem = TpFlow.TwoPhaseFlowProblem()
myTpFlowProblem.outputStepping.final_time = 0.1
myTpFlowProblem.outputStepping.dt_init = 0.01
myTpFlowProblem.outputStepping.dt_output = 0.1
myTpFlowProblem.outputStepping.dt_fixed = 0.01
myTpFlowProblem.outputStepping.archiveAllSteps = True

myTpFlowProblem.domain = domain

myTpFlowProblem.SystemNumerics.useSuperlu=False
myTpFlowProblem.SystemNumerics.cfl=0.9

myTpFlowProblem.SystemPhysics.setDefaults()
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelMoveMeshElastic,'move')
myTpFlowProblem.SystemPhysics.useDefaultModels()
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelAddedMass,'addedMass')

myTpFlowProblem.SystemPhysics.movingDomain = True
# line below needed for relaxation zones
# (!) hack
m = myTpFlowProblem.SystemPhysics.modelDict
m['flow'].auxiliaryVariables += domain.auxiliaryVariables['twp']

params = myTpFlowProblem.SystemPhysics

#initialConditions
myTpFlowProblem.SystemPhysics.modelDict['move'].p.initialConditions['hx']=Zero_IC()
myTpFlowProblem.SystemPhysics.modelDict['move'].p.initialConditions['hy']=Zero_IC()
myTpFlowProblem.SystemPhysics.modelDict['move'].p.initialConditions['hz']=Zero_IC()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['p']=P_IC()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['u']=U_IC()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['v']=V_IC()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['w']=W_IC()
myTpFlowProblem.SystemPhysics.modelDict['vof'].p.initialConditions['vof']=VF_IC()
myTpFlowProblem.SystemPhysics.modelDict['ncls'].p.initialConditions['phi']=PHI_IC()
myTpFlowProblem.SystemPhysics.modelDict['rdls'].p.initialConditions['phid']=PHI_IC()
myTpFlowProblem.SystemPhysics.modelDict['mcorr'].p.initialConditions['phiCorr']=PHI_IC()
myTpFlowProblem.SystemPhysics.modelDict['addedMass'].p.initialConditions['addedMass']=Zero_IC()

# PHYSICAL PARAMETERS
params['rho_0'] = rho_0  # water
params['rho_1'] = rho_1  # air
params['nu_0'] = nu_0  # water
params['nu_1'] = nu_1  # air
params['gravity'] = np.array(g)
params['surf_tension_coeff'] = sigma_01

m['flow'].auxiliaryVariables += [system]
m['flow'].p.coefficients.eb_bc_penalty_constant = 10.#/nu_0#Re
m['addedMass'].auxiliaryVariables += [system.ProtChAddedMass]
m['flow'].p.coefficients.NONCONSERVATIVE_FORM=0.0
max_flag = 0
max_flag = max(domain.vertexFlags)
max_flag = max(domain.segmentFlags+[max_flag])
max_flag = max(domain.facetFlags+[max_flag])
flags_rigidbody = np.zeros(max_flag+1, dtype='int32')
for s in system.subcomponents:
    if type(s) is fsi.ProtChBody:
        for i in s.boundaryFlags:
            flags_rigidbody[i] = 1
m['addedMass'].p.coefficients.flags_rigidbody = flags_rigidbody
