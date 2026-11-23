import numpy as np
from proteus import Domain
from proteus import SpatialTools as st
from proteus.mbd import CouplingFSI as fsi
import pychrono as chrono
from proteus.TwoPhaseFlow import TwoPhaseFlowProblem as tpf
from proteus.TwoPhaseFlow.utils import Parameters
import os

rho_0 = 1000.
nu_0 = 1.004e-6
rho_1 = 1.205
nu_1 = 1.500e-5
sigma_01 = 0.0
g = [0., 0., -9.81]
he = 2.5
water_level = 2.5

# GEOMETRY

domain = Domain.PiecewiseLinearComplexDomain()
nd=3
tank_dim = [5.,5.,5.]
tank = st.Tank3D(domain, dim=tank_dim)
rect = st.Cuboid(domain, dim=[1.,1.,1.], coords=[tank_dim[0]/2.,
                                                 tank_dim[1]/2.,
                                                 tank_dim[2]/2.])
rect.setHoles(holes=np.array([rect.coords]))

domain.MeshOptions.he = he
# BOUNDARY CONDITIONS

tank.BC['x+'].setNoSlip()
tank.BC['x-'].setNoSlip()
tank.BC['y-'].setNoSlip()
tank.BC['y+'].setNoSlip()
tank.BC['z-'].setNoSlip()
tank.BC['z+'].setAtmosphere()

rect.BC['x+'].setNoSlip()
rect.BC['x-'].setNoSlip()
rect.BC['y+'].setNoSlip()
rect.BC['y-'].setNoSlip()
rect.BC['z+'].setNoSlip()
rect.BC['z-'].setNoSlip()

# CHRONO

system = fsi.ProtChSystem()
system.ChSystem.SetGravitationalAcceleration(chrono.ChVector3d(g[0], g[1], g[2]))
body = fsi.ProtChBody(system=system)
body.attachShape(rect)
body.ChBody.SetMass(500.)
body.ChBody.SetFixed(True)  # fixing body

# OTHER PARAMS
st.assembleDomain(domain)

domain.polyfile=domain.polyfile=os.path.dirname(os.path.abspath(__file__))+"/"+"mesh3D"
domain.MeshOptions.he = he
domain.MeshOptions.genMesh=False

#domain.writePoly("mesh3D")

#  ___       _ _   _       _    ____                _ _ _   _
# |_ _|_ __ (_) |_(_) __ _| |  / ___|___  _ __   __| (_) |_(_) ___  _ __  ___
#  | || '_ \| | __| |/ _` | | | |   / _ \| '_ \ / _` | | __| |/ _ \| '_ \/ __|
#  | || | | | | |_| | (_| | | | |__| (_) | | | | (_| | | |_| | (_) | | | \__ \
# |___|_| |_|_|\__|_|\__,_|_|  \____\___/|_| |_|\__,_|_|\__|_|\___/|_| |_|___/
# Initial Conditions

nd = domain.nd

class PerturbedSurface_p:
    def uOfXT(self, x, t):
        p_L = 0.0
        phi = x[nd-1] - tank_dim[nd-1]
        return p_L-g[nd-1]*(rho_0*phi)

class AtRest:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

#  _   _                           _
# | \ | |_   _ _ __ ___   ___ _ __(_) ___ ___
# |  \| | | | | '_ ` _ \ / _ \ '__| |/ __/ __|
# | |\  | |_| | | | | | |  __/ |  | | (__\__ \
# |_| \_|\__,_|_| |_| |_|\___|_|  |_|\___|___/
# Numerics

myTpFlowProblem = tpf.TwoPhaseFlowProblem()
myTpFlowProblem.domain = domain

myTpFlowProblem.outputStepping.final_time = 0.002
myTpFlowProblem.outputStepping.dt_init = 0.001
myTpFlowProblem.outputStepping.dt_output = 0.001
myTpFlowProblem.outputStepping.dt_fixed = 0.001

myTpFlowProblem.SystemPhysics.setDefaults()

myTpFlowProblem.SystemNumerics.cfl = 0.4
myTpFlowProblem.SystemNumerics.useSuperlu=False
myTpFlowProblem.SystemPhysics.movingDomain = False

params = myTpFlowProblem.SystemPhysics

# PHYSICAL PARAMETERS
params.rho_0 = rho_0  # water
params.rho_1 = rho_1  # air
params.nu_0 = nu_0  # water
params.nu_1 = nu_1  # air
params.surf_tension_coeff = sigma_01

# MODEL PARAMETERS
m = params.modelDict
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelRANS2P,'flow')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelAddedMass,'addedMass')

myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['p']=PerturbedSurface_p()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['u']=AtRest()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['v']=AtRest()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['w']=AtRest()
myTpFlowProblem.SystemPhysics.modelDict['addedMass'].p.initialConditions['addedMass']=AtRest()

m['flow'].p.coefficients.useVF = 1.0
m['flow'].p.coefficients.NONCONSERVATIVE_FORM = 0.0

# auxiliary variables
m['flow'].auxiliaryVariables += [system]
m['addedMass'].auxiliaryVariables += [system.ProtChAddedMass]

flags_rigidbody = np.zeros(20)
for key in rect.boundaryTags_global:
    flags_rigidbody[rect.boundaryTags_global[key]] = 1.

max_flag = 0
max_flag = max(domain.vertexFlags)
max_flag = max(domain.segmentFlags+[max_flag])
max_flag = max(domain.facetFlags+[max_flag])
flags_rigidbody = np.zeros(max_flag+1, dtype='int32')
for s in system.subcomponents:
    if type(s) is fsi.ProtChBody:
        for flag in s.boundaryFlags:
            flags_rigidbody[flag] = 1
m['addedMass'].p.coefficients.flags_rigidbody = flags_rigidbody
m['addedMass'].n.linTolFac=0.0
m['addedMass'].n.l_atol_res=1.0e-10
