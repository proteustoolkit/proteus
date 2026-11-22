import numpy as np
from math import cos
from proteus import (Domain, Context,
                     FemTools as ft,
                     MeshTools as mt,
                     WaveTools as wt)
from proteus import SpatialTools as st
from proteus.Profiling import logEvent
from proteus.SpatialTools import Tank2D, CustomShape
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow


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
    ("rho_0", 998.2, "Water density"),
    ("nu_0", 1.004e-6, "Water kinematic viscosity m/sec^2"),
    #cek hack
    #("rho_1", 998.2, "Water density"),
    #("nu_1", 1.004e-6, "Water kinematic viscosity m/sec^2"),
    #
    ("rho_1", 1.205, "Air Densiy"),
    ("nu_1", 1.500e-5, "Air kinematic viscosity m/sec^2"),
    ("sigma_01", 0., "Surface tension"),
    ("g", (0, -9.81, 0), "Gravitational acceleration vector"),
    ]
# mesh options
context_options += [
    # mesh options
    ("he", 0.0025,"Characteristic element size"),
    ("he_max", 1.,"Maximum characteristic element size"),
    ("use_gmsh", False ,"True: gmsh. Fale: triangle/tetgen"),
    ("constantRefinement", False, "constant refinement"),
    ("grading", 1.0,"Grading of mesh, e.g. 1.1 -> 10% increase from element to next element"),
    ("genMesh", True ,"Generate mesh (True/False)"),
    ("movingDomain", False,"True: domain (mesh) is moving"),
    ("structured", False, "Use a structured mesh"),
    ("useHex", False, "Use a hexahedral structured mesh"),
    ]
# run time options
context_options += [
    ("T", 0.1 ,"Simulation time in s"),
    ("dt_init", 0.001 ,"Value of initial time step"),
    ("dt_fixed", None, "Value of maximum time step"),
    ("archiveAllSteps", False, "archive every steps"),
    ("dt_output", 1., "number of saves per second"),
    ("runCFL", 0.33 ,"Target CFL value"),
    ("cfl", 0.33 ,"Target CFL value"),
    ]
# other options
context_options += [
    ("water_depth_fraction", 0.5, "Nondimensional water level normalised to the tank height"),
    ("amplitude", 0.005, "Nondimensional amplitude normalised to tank height"),
    # tank
    ("tank_dim", (0.1 , 0.1), "Dimensions of the tank (length, height) in m"),
    # gauges
    ("gauge_output", False, "Produce gauge data. A free-surface gauge is located at the far left boundary"),
    # boundary condition
    ("openTop", True,"Enabling atmosphere boundary at the top"),
    # other general options
    ("useSuperlu", True, "useSuperlu"),
    ("ELLIPTIC_REDISTANCING", 0, "elliptic redistancing"),
    ("twoLayer", True, "turn porosity on an off for debugging"),
    ]
# instantiate context options
opts=Context.Options(context_options)

# remove opts. on some optiosn for easier reference later on
rho_0 = opts.rho_0
nu_0 = opts.nu_0
rho_1 = opts.rho_1
nu_1 = opts.nu_1
sigma_01 = opts.sigma_01
g = np.array(opts.g)
he = opts.he
cfl = opts.runCFL


grading = opts.grading  # grading of mesh according to distance function
grading_scale_with_nd = True  # to scale the monitor distance function to relative area
he_min = opts.he
he_max = opts.he_max
ecH = 1.5

# tank
tank_dim = opts.tank_dim

m = 1
km = m*np.pi/tank_dim[0]
water_amplitude = opts.amplitude
water_depth = h = tank_dim[1]*opts.water_depth_fraction

wavelength = tank_dim[0]*2.
k = 2*np.pi/wavelength
h = opts.water_depth_fraction*opts.tank_dim[1]
eps = k*water_amplitude

#  ____                        _
# |  _ \  ___  _ __ ___   __ _(_)_ __
# | | | |/ _ \| '_ ` _ \ / _` | | '_ \
# | |_| | (_) | | | | | | (_| | | | | |
# |____/ \___/|_| |_| |_|\__,_|_|_| |_|
# Domain
# All geometrical options go here (but not mesh options)

if opts.useHex or opts.structured:
    domain = Domain.RectangularDomain(tank_dim)
    boundaryTags = domain.boundaryTags
else:
    domain = Domain.PlanarStraightLineGraphDomain()

# ----- TANK ----- #

justTank=False
if justTank:
    tank = Tank2D(domain, tank_dim)
    tank.facets = np.array([[[0, 1, 2, 3]]])
    tank.facetFlags = np.array([1])
else:
    bt={'x-':1,
        'x+':2,
        'y-':3,
        'y+':4}
    tank = CustomShape(domain,
                       vertices=[[0.0,0.0],
                                 [0.0,tank_dim[1]],
                                 [0.5*tank_dim[0],tank_dim[1]],
                                 [tank_dim[0],tank_dim[1]],
                                 [tank_dim[0],0.0],
                                 [0.5*tank_dim[0],0.0]],
                       vertexFlags=[bt['y-'],
                                    bt['y+'],
                                    bt['y+'],
                                    bt['y+'],
                                    bt['y-'],
                                    bt['y-']],
                       segments=[[0,1],
                                 [1,2],
                                 [2,3],
                                 [3,4],
                                 [4,5],
                                 [5,0],
                                 [2,5]],
                       segmentFlags=[bt['x-'],
                                     bt['y+'],
                                     bt['y+'],
                                     bt['x+'],
                                     bt['y-'],
                                     bt['y-'],
                                     0],
                       regions=[[0.25*tank_dim[0],0.5*tank_dim[1]],
                                [0.75*tank_dim[0],0.5*tank_dim[1]]],
                       regionFlags=[1,2],
                       boundaryTags=bt,
                       boundaryOrientations={
                           'x-': np.array([ -1.0,  0.0,  0.0]),
                           'x+': np.array([  1.0,  0.0,  0.0]),
                           'y-': np.array([  0.0, -1.0,  0.0]),
                           'y+': np.array([  0.0,  1.0,  0.0]),
                       })
    tank.facets = np.array([[[0, 1, 2, 3, 4, 5]]])
    tank.facetFlags = np.array([1])


#  ____                        _                   ____                _ _ _   _
# | __ )  ___  _   _ _ __   __| | __ _ _ __ _   _ / ___|___  _ __   __| (_) |_(_) ___  _ __  ___
# |  _ \ / _ \| | | | '_ \ / _` |/ _` | '__| | | | |   / _ \| '_ \ / _` | | __| |/ _ \| '_ \/ __|
# | |_) | (_) | |_| | | | | (_| | (_| | |  | |_| | |__| (_) | | | | (_| | | |_| | (_) | | | \__ \
# |____/ \___/ \__,_|_| |_|\__,_|\__,_|_|   \__, |\____\___/|_| |_|\__,_|_|\__|_|\___/|_| |_|___/
#                                           |___/
# Boundary Conditions

domain.MeshOptions.he = opts.he
domain.MeshOptions.genMesh = opts.genMesh
domain.MeshOptions.use_gmsh = opts.use_gmsh
meshfile = 'mesh'+str(opts.he)
domain.geofile = meshfile

if opts.useHex:
    domain.MeshOptions.nnx = 4 * refinement + 1
    domain.MeshOptions.nny = 2*refinement+1
elif opts.structured:
    domain.MeshOptions.nnx = 4 * refinement
    domain.MeshOptions.nny = 2 * refinement

twoLayer=opts.twoLayer
domain.porosityTypes =np.array([1.0,1.0,1.0],'d')
domain.dragAlphaTypes=np.array([0.0,0.0,0.0],'d')
domain.dragBetaTypes =np.array([0.0,0.0,0.0],'d')
domain.epsFact_porous =np.array([0.0,0.0,0.0],'d')
if twoLayer:
    domain.porosityTypes[2] = 0.3
    domain.dragAlphaTypes[2] = 1.0e8
    domain.dragBetaTypes[2] = 1.0e8

st.assembleDomain(domain)

if opts.openTop is True:
    tank.BC['y+'].setAtmosphere()
else:
    tank.BC['y+'].setFreeSlip()
tank.BC['y-'].setFreeSlip()
tank.BC['x+'].setFreeSlip()
tank.BC['x-'].setFreeSlip()


#  ___       _ _   _       _    ____                _ _ _   _
# |_ _|_ __ (_) |_(_) __ _| |  / ___|___  _ __   __| (_) |_(_) ___  _ __  ___
#  | || '_ \| | __| |/ _` | | | |   / _ \| '_ \ / _` | | __| |/ _ \| '_ \/ __|
#  | || | | | | |_| | (_| | | | |__| (_) | | | | (_| | | |_| | (_) | | | \__ \
# |___|_| |_|_|\__|_|\__,_|_|  \____\___/|_| |_|\__,_|_|\__|_|\___/|_| |_|___/
# Initial Conditions


class PerturbedSurface_p:
    def __init__(self,waterdepth,amplitude):
        self.waterdepth=waterdepth
        self.amplitude=amplitude
    def uOfXT(self,x,t):
        d = signedDistance(x, 0.)
        if d <= 0:
            #return pressure(x[0], x[1]-self.waterdepth, t, h, eps, rho_0, g, k, 0.)+(tank_dim[1]-(self.waterdepth+eta(x[1], 0.)))*rho_1*(-g[1])
            return (tank_dim[1]-(self.waterdepth+eta(x[0],0.)))*rho_1*(-g[1])+(self.waterdepth+eta(x[0],0.)-x[1])*rho_0*(-g[1])
        else:
            return (tank_dim[1] - x[1])*rho_1*(-g[1])

from proteus.ctransportCoefficients import smoothedHeaviside
class PerturbedSurface_H:
    def uOfXT(self,x,t):
        return smoothedHeaviside(ecH * he, signedDistance(x, 0.))
        #return smoothedHeaviside(0.0, signedDistance(x, 0.))

class PerturbedSurface_phi:
    def uOfXT(self,x,t):
        return signedDistance(x, t)

class AtRest:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

def eta0(x, t):
    eta_0 = np.sin(t)*np.cos(x)
    return eta_0

def phi0(x, y, t, h, w0):
    phi_0 = w0/(np.sinh(h))*np.cos(t)*np.cos(x)*np.cosh(y+h)
    return phi_0

def w0(h):
    w_0 = np.sqrt(np.tanh(h))
    return w_0


def eta1(x, t, w0):
    eta_1 = 1./8.*((w0**2-w0**(-2))+(w0**(-2)-3*w0**(-6))*np.cos(2.*t))*np.cos(2.*x)
    return eta_1

def phi1(x, y, t, h, w0):
    beta0 = 0  # arbitrary
    phi_1 = beta0+1./8.(w0-w0**(-3))*t-1./16.*(3*w0+w0**(-3))*np.sin(2*t)-3./(16.*np.cosh(2.*h))*(w0-w0**(-7))*np.sin(2*t)*np.cos(2*x)*np.cosh(2*(y+h))
    return phi_1

w1 = 0.

def eta2(x, t, w0):
    b11 = 1./32.*(3.*w0**(-8)+6.*w0**(-4)-5.+2.*w0**4)
    b13 = 3./128.*(9.*w0**(-8)+27.*w0**(-4)-15.+w0**4+2*w0**8)
    b31 = 1./128.*(3.*w0**(-8)+18.*w0**(-4)-5.)
    b33 = 3./128.*(-9.*w0**(-12)+3.*w0**(-8)-3.*w0**(-4)+1)
    eta_2 = b11*np.sin(t)*np.cos(x)+b13*np.sin(t)*np.cos(3*x)+b31*np.sin(3*t)*np.cos(x)+b33*np.sin(3*t)*np.cos(3*x)
    return eta_2

def phi2(x, y, t, h, w0):
    beta13 = 1./(128.*np.cosh(3.*h))*(1+3*w0**4)*(3*w0**(-9)-5.*w0**(-1)+2*w0**3)
    beta31 = 1./(128.*np.cosh(h))*(9.*w0**(-9)+62.*w0**(-5)-31.*w0**(-1))
    beta33 = 1./(128.*np.cosh(3.*h))*(1+3.*w0**4)*(-9.*w0**(-13)+22.*w0**(-9)-13.*w0**(-5))
    beta2 = 0.  # arbitrary
    phi_2 = beta2+beta13*np.cos(t)*np.cos(3*x)*np.cosh(3*(y+h))+beta31*np.cos(3*t)*np.cos(x)*np.cosh(y+h)+beta33*np.cos(3*t)*np.cos(3*x)*np.cosh(3*(y+h))
    return phi_2

def eps_eta(x, t, h, eps):
    w_0 = w0(h)
    eta_0 = eta0(x, t)
    eta_1 = eta1(x, t, w_0)
    eta_2 = eta2(x, t, w_0)
    epseta = eps*eta_0+eps**2*eta_1+0.5*eps**3*eta_2
    return epseta

def w2(w0):
    w_2 = 1./32.*(9.*w0**(-7)-12.*w0**(-3)-3*w0-2*w0**5)
    return w_2

def omega(h, eps):
    w_0 = w0(h)
    w_2 = w2(w_0)
    w = w_0+0.5*eps**2*w_2
    return w


def d_phi0_d_t(x, y, t, h, w0):
    return -w0/np.sinh(h)*np.sin(t)*np.cos(x)*np.cosh(y+h)

def d_phi0_d_x(x, y, t, h, w0):
    return -w0/np.sinh(h)*np.cos(t)*np.sin(x)*np.cosh(y+h)

def d_phi0_d_y(x, y, t, h, w0):
    return w0/np.sinh(h)*np.cos(t)*np.cos(x)*np.sinh(y+h)

def d_phi1_d_t(x, y, t, h, w0):
    return 1./8.*(w0-w0**(-3))-2./16.*(3.*w0+w0**(-3))*np.cos(2*t)-2.*3./(16.*np.cosh(2.*h))*(w0-w0**(-7))*np.cos(2.*t)*np.cos(2.*x)*np.cosh(2.*(y+h))

def d_phi1_d_x(x, y, t, h, w0):
    return 2.*3./(16.*np.cosh(2.*h))*(w0-w0**(-7))*np.sin(2.*t)*np.sin(2.*x)*np.cosh(2.*(y+h))

def d_phi1_d_y(x, y, t, h, w0):
    return 2.*3./(16.*np.cosh(2.*h))*(w0-w0**(-7))*np.sin(2.*t)*np.cos(2.*x)*np.sinh(2.*(y+h))

def d_phi2_d_t(x, y, t, h, w0):
    beta13 = 1./(128.*np.cosh(3.*h))*(1+3*w0**4)*(3*w0**(-9)-5.*w0**(-1)+2*w0**3)
    beta31 = 1./(128.*np.cosh(h))*(9.*w0**(-9)+62.*w0**(-5)-31.*w0**(-1))
    beta33 = 1./(128.*np.cosh(3.*h))*(1+3.*w0**4)*(-9.*w0**(-13)+22.*w0**(-9)-13.*w0**(-5))
    return -beta13*np.sin(t)*np.cos(3.*x)*np.cosh(3.*(y+h))-3.*beta31*np.sin(3.*t)*np.cos(x)*np.cosh(y+h)-3.*beta33*np.sin(3.*t)*np.cos(3.*x)*np.cosh(3.*(y+h))

def d_phi2_d_x(x, y, t, h, w0):
    beta13 = 1./(128.*np.cosh(3.*h))*(1+3*w0**4)*(3*w0**(-9)-5.*w0**(-1)+2*w0**3)
    beta31 = 1./(128.*np.cosh(h))*(9.*w0**(-9)+62.*w0**(-5)-31.*w0**(-1))
    beta33 = 1./(128.*np.cosh(3.*h))*(1+3.*w0**4)*(-9.*w0**(-13)+22.*w0**(-9)-13.*w0**(-5))
    return -3.*beta13*np.cos(t)*np.sin(3.*x)*np.cosh(3.*(y+h))-beta31*np.cos(3.*t)*np.sin(x)*np.cosh(y+h)-3.*beta33*np.cos(3.*t)*np.sin(3.*x)*np.cosh(3.*(y+h))

def d_phi2_d_y(x, y, t, h, w0):
    beta13 = 1./(128.*np.cosh(3.*h))*(1+3*w0**4)*(3*w0**(-9)-5.*w0**(-1)+2*w0**3)
    beta31 = 1./(128.*np.cosh(h))*(9.*w0**(-9)+62.*w0**(-5)-31.*w0**(-1))
    beta33 = 1./(128.*np.cosh(3.*h))*(1+3.*w0**4)*(-9.*w0**(-13)+22.*w0**(-9)-13.*w0**(-5))
    return 3.*beta13*np.cos(t)*np.cos(3.*x)*np.sinh(3.*(y+h))+beta31*np.cos(3.*t)*np.cos(x)*np.sinh(y+h)+3.*beta33*np.cos(3.*t)*np.cos(3.*x)*np.sinh(3.*(y+h))

def pressure(x, y, t, h, eps, rho, g, k, p0):
    x_ = x*k
    y_ = y*k
    h_ = h*k
    w_ = omega(h_, eps)
    t_ = (t+2*np.pi/(w_*np.sqrt(k*(-g[1])))*0.25)*(w_*np.sqrt(k*(-g[1])))
    w_0 = w0(h_)
    dt = eps*d_phi0_d_t(x_, y_, t_, h_, w_0)+eps**2*d_phi1_d_t(x_, y_, t_, h_, w_0)+0.5*eps**3*d_phi2_d_t(x_, y_, t_, h_, w_0)
    dx = eps*d_phi0_d_x(x_, y_, t_, h_, w_0)+eps**2*d_phi1_d_x(x_, y_, t_, h_, w_0)+0.5*eps**3*d_phi2_d_x(x_, y_, t_, h_, w_0)
    dy = eps*d_phi0_d_y(x_, y_, t_, h_, w_0)+eps**2*d_phi1_d_y(x_, y_, t_, h_, w_0)+0.5*eps**3*d_phi2_d_y(x_, y_, t_, h_, w_0)
    p = (-y_-dt*w_-0.5*dx**2-0.5*dy**2)*rho*abs(g[1])/k+p0
    return p

def eta(x, t):
    x_ = x*k
    h_ = h*k
    w_ = omega(h_, eps)
    t_ = (t+2*np.pi/(w_*np.sqrt(k*(-g[1])))*0.25)*(w_*np.sqrt(k*(-g[1])))
    eta = eps_eta(x_, t_, h_, eps)/k
    #cek hack
    #eta=0.0
    return eta

def signedDistance(x, t):
    #d=abs(x[1] - water_depth  - water_amplitude * cos(x[0]))
    d = x[1]-(water_depth+eta(x[0], t))
    #d = x[1]-water_depth
    return d



#  _   _                           _
# | \ | |_   _ _ __ ___   ___ _ __(_) ___ ___
# |  \| | | | | '_ ` _ \ / _ \ '__| |/ __/ __|
# | |\  | |_| | | | | | |  __/ |  | | (__\__ \
# |_| \_|\__,_|_| |_| |_|\___|_|  |_|\___|___/
# Numerics


myTpFlowProblem = TpFlow.TwoPhaseFlowProblem()
myTpFlowProblem.outputStepping.final_time = opts.T
myTpFlowProblem.outputStepping.dt_init = 0.01
myTpFlowProblem.outputStepping.dt_output = 0.1
myTpFlowProblem.outputStepping.dt_fixed = 0.01
myTpFlowProblem.outputStepping.archiveAllSteps = opts.archiveAllSteps

myTpFlowProblem.domain = domain

myTpFlowProblem.SystemNumerics.useSuperlu=opts.useSuperlu
myTpFlowProblem.SystemNumerics.cfl=opts.cfl

myTpFlowProblem.SystemPhysics.setDefaults()
myTpFlowProblem.SystemPhysics.useDefaultModels()

myTpFlowProblem.SystemPhysics.movingDomain = opts.movingDomain

m = myTpFlowProblem.SystemPhysics.modelDict

# ELLIPTIC_REDISTANCING
if opts.ELLIPTIC_REDISTANCING != 0:
    m['rdls'].p.ELLIPTIC_REDISTANCING = opts.ELLIPTIC_REDISTANCING
    from proteus import NonlinearSolvers
    m['rdls'].n.levelNonlinearSolver = NonlinearSolvers.TwoStageNewton

params = myTpFlowProblem.SystemPhysics

# PHYSICAL PARAMETERS
params['rho_0'] = opts.rho_0  # water
params['rho_1'] = opts.rho_1  # air
params['nu_0'] = opts.nu_0  # water
params['nu_1'] = opts.nu_1  # air
params['surf_tension_coeff'] = opts.sigma_01

m['flow'].p.coefficients.weak_bc_penalty_constant = 10.

# INITIAL CONDITIONS
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['p']=PerturbedSurface_p(water_depth, water_amplitude)
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['u']=AtRest()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['v']=AtRest()
myTpFlowProblem.SystemPhysics.modelDict['vof'].p.initialConditions['vof']=PerturbedSurface_H()
myTpFlowProblem.SystemPhysics.modelDict['ncls'].p.initialConditions['phi']=PerturbedSurface_phi()
myTpFlowProblem.SystemPhysics.modelDict['rdls'].p.initialConditions['phid']=PerturbedSurface_phi()
myTpFlowProblem.SystemPhysics.modelDict['mcorr'].p.initialConditions['phiCorr']=AtRest()

if opts.gauge_output:
    # ----- GAUGES ----- #
    from proteus import Gauges
    gauge_dx = tank_dim[0]/100.
    probes=np.linspace(0., tank_dim[0], tank_dim[0]/gauge_dx+1)
    PG=[]
    PG2=[]
    LG = []
    zProbes=water_depth*0.5
    for i in probes:
        PG.append((i, zProbes, 0.),)
        PG2.append((i, water_depth, 0.),)
        LG.append([(i, 0., 0.),(i, tank_dim[1], 0.)])

    m['flow'].auxiliaryVariables += [
        Gauges.PointGauges(
            gauges = ((('p',), PG),),
            activeTime=(0, opts.T),
            sampleRate=0,
            fileName='pointGauge_pressure.csv')
    ]
    m['ncls'].auxiliaryVariables += [
        Gauges.PointGauges(
            gauges = ((('phi',), PG2),),
            activeTime=(0, opts.T),
            sampleRate=0,
            fileName='pointGauge_levelset.csv')
    ]
    m['vof'].auxiliaryVariables += [
        Gauges.LineIntegralGauges(
            gauges = ((('vof',), LG),),
            activeTime=(0, opts.T),
            sampleRate=0,
            fileName='lineGauge_vof.csv')
    ]
    m['flow'].auxiliaryVariables += [
        Gauges.LineIntegralGauges(
            gauges = ((('p',), LG),),
            activeTime=(0, opts.T),
            sampleRate=0,
            fileName='lineGauge_pressure.csv')
    ]


#  __  __           _        ___        _   _
# |  \/  | ___  ___| |__    / _ \ _ __ | |_(_) ___  _ __  ___
# | |\/| |/ _ \/ __| '_ \  | | | | '_ \| __| |/ _ \| '_ \/ __|
# | |  | |  __/\__ \ | | | | |_| | |_) | |_| | (_) | | | \__ \
# |_|  |_|\___||___/_| |_|  \___/| .__/ \__|_|\___/|_| |_|___/
#                                |_|

if opts.use_gmsh:
    import py2gmsh
    from py2gmsh import Mesh, Field, geometry2mesh
    mesh = geometry2mesh(domain)
    field_list = []

    def mesh_grading(start, he, grading):
        return '(abs({start})*({grading}-1)+{he})'.format(he=he, start=start, grading=grading)

    def dist_plane(xn, xp, plane='x'):
        x_range = abs(xp-xn)
        dist = '0.5*(abs({plane}-({xn}))+abs({plane}-({xp}))-{x_range})'.format(xn=xn, xp=xp, x_range=x_range, plane=plane)
        return dist

    me1 = Field.MathEval(mesh=mesh)
    y_pos = 'abs(({center_y}+{amplitude}*cos(3.14*abs(x)/({range_x}))))'.format(amplitude=opts.amplitude,
                                                                          center_x=tank_dim[0]/2.,
                                                                          center_y=tank_dim[1]/2.,
                                                                          range_x=tank_dim[0])

    dist = dist_plane(xn=0., xp=0., plane='x')

    #dist = 'sqrt(({dist_z})^2)'.format(dist_z=dist_z)
    me1.F = mesh_grading(start=dist, he=he, grading=grading)
    field_list += [me1]


    # background field
    fmin = Field.Min(mesh=mesh)
    fmin.FieldsList = field_list
    mesh.setBackgroundField(fmin)

    # max element size
    mesh.Options.Mesh.CharacteristicLengthMax = opts.he_max

    mesh.writeGeo(meshfile+'.geo')

def my_func(x, t):
    dist = 1.
    return dist

m['mcorr'].p.coefficients.checkMass=True
m['mcorr'].n.nl_atol_res=1e-10
m['mcorr'].n.tolFac=0.0
