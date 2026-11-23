from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards


import math

nd = 3

regularGrid=False

boundaries = ['outer', 'top', 'bottom', 'pump']
boundaryTags = {key: i + 1 for i, key in enumerate(boundaries)}


def generate_hollow_cylinder(center, inner_radius=0.1, outer_radius=30,
                              height=3.0, n_points=100):
    #cx, cy = center
    cx, cy, cz = center
    angle_step = 2 * math.pi / n_points

    outer_top = []
    outer_bottom = []
    inner_top = []
    inner_bottom = []

    for i in range(n_points):
        angle = i * angle_step
        dx = math.cos(angle)
        dy = math.sin(angle)
        outer_top.append([cx + outer_radius * dx, cy + outer_radius * dy, cz + height/2])
        outer_bottom.append([cx + outer_radius * dx, cy + outer_radius * dy, cz-height/2])
        inner_top.append([cx + inner_radius * dx, cy + inner_radius * dy, cz + height/2])
        inner_bottom.append([cx + inner_radius * dx, cy + inner_radius * dy, cz- height/2])
    
    facets = []
    for i in range(n_points):
        ni = (i + 1) % n_points
        facets.append([i, ni, n_points + ni, n_points + i])  # outer wall
        facets.append([2 * n_points + ni, 2 * n_points + i, 3 * n_points + i, 3 * n_points + ni])  # inner wall
        facets.append([i, ni, 2 * n_points + ni, 2 * n_points + i])  # top face
        facets.append([3 * n_points + i, 3 * n_points + ni, n_points + ni, n_points + i])  # bottom face

    #return vertices, facets

    vertices = outer_top + outer_bottom + inner_top + inner_bottom
    facets_proteus = [[f] for f in facets]
    return vertices, facets_proteus



center=[2.0, 2.0, 1.0]
pump_radius= 0.2
domain_radius= 2.0
height= 2.0
n_points= 31

vertices, facets = generate_hollow_cylinder(center, pump_radius, domain_radius, height,n_points)


vertexFlags = (
    [boundaryTags['top']] * n_points +   # outer_top
    [boundaryTags['bottom']] * n_points +   # outer_bottom
    [boundaryTags['pump']]  * n_points +   # inner_top
    [boundaryTags['pump']]  * n_points     # inner_bottom
    )

facetFlags = []
for _ in range(n_points):
    facetFlags.extend([
        boundaryTags['outer'],   # outer wall
        boundaryTags['pump'],    # inner wall
        boundaryTags['top'],     # top face
        boundaryTags['bottom']   # bottom face
    ])

region_radius = (pump_radius + domain_radius) / 2.0  # (0.6 + 30) / 2 = 15.3
regions = [[center[0] + region_radius, center[1], center[2]]]

regionFlags= [1]

holes = [[2.0, 2.0, 1.0]]

domain = Domain.PiecewiseLinearComplexDomain(vertices=vertices,
                                                  vertexFlags=vertexFlags,
                                                  facets=facets,
                                                  facetFlags=facetFlags,
                                                  holes=holes,
                                                  regions=regions,
                                                  regionFlags=regionFlags)

domain.writePoly("hollow_cylinder")  # Generates hollow_cylinder.poly

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0 #density*gravity*4.524e-10
m_per_s_by_m_per_d = 1.1574074e-5
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)
#make non-dimensional

Transmissivity = 5e-4       # m²/s
b = height     # 3.0 m
Ks = Transmissivity / b     # m/s



# Storage Zone
permeability1  = Ks *viscosity/(gravity*density)  #m^2
#permeability1  = (0.00504)*viscosity/(gravity*density)  #m^2
thetaS1        = 0.4   #-
thetaR1        = 0.05   #-
mvg_alpha1     = 8   #1/m
mvg_n1         = 2.4
mvg_m1         = 1.0 - 1.0/mvg_n1
dimensionless_conductivity1  = (timeScale*density*gravity*permeability1/(viscosity*lengthScale))

#print 'Ks',dimensionless_conductivity
dimensionless_density  = 1.0
dimensionless_gravity  = numpy.array([0.0,
                                        0.0,
                                        -1.0])
#dimensionless_alpha    = mvg_alpha*lengthScale
nMediaTypes  = 1
alphaVGtypes = numpy.zeros((nMediaTypes+1,),'d')
nVGtypes     = numpy.zeros((nMediaTypes+1,),'d')
thetaStypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes+1,),'d')
KsTypes      = numpy.zeros((nMediaTypes+1,2),'d')

for i in range(nMediaTypes+1):
    if i==1:
        alphaVGtypes[i] = mvg_alpha1
        nVGtypes[i]     = mvg_n1
        thetaStypes[i]  = thetaS1
        thetaRtypes[i]  = thetaR1
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity1,dimensionless_conductivity1]#m/d?
    else:
        alphaVGtypes[i] = mvg_alpha1
        nVGtypes[i]     = mvg_n1
        thetaStypes[i]  = thetaS1
        thetaRtypes[i]  = thetaR1
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity1,dimensionless_conductivity1]#m/d?


galerkin = False
#useSeepageFace = True

#def getSeepageFace(flag):
#   if useSeepageFace:
#       if flag == boundaryTags['pump']:
#           return 1
#       else:
#           return 0
#   else:
#       return 0




LevelModelType = Richards.LevelModel
coefficients = Richards.Coefficients(nd,
                                     KsTypes,
                                     nVGtypes,
                                     alphaVGtypes,
                                     thetaRtypes,
                                     thetaSRtypes,
                                     gravity=dimensionless_gravity,
                                     density=dimensionless_density,
                                     beta=0.0001,
                                     diagonal_conductivity=True,
                                     STABILIZATION_TYPE=2,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX= False ,
                                     FCT=True, #False ,#True,
                                     num_fct_iter=0,
                                     # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False,
                                     storavity= 1e-4) #,
                                     #getSeepageFace=getSeepageFace)
#galerkin = False

galerkin = False

######THEIS Head at pump################
#from scipy.special import exp1
#import numpy as np

def exp1_approx(u):
    #if u <= 0:
    #    raise ValueError("Input u must be > 0 for exp1")
    if u <= 1e-30:
        u = 1e-30  # Avoid log(0) or instability

    # Taylor series expansion for small u
    if u < 1.0:
        sum_term = 0.0
        factorial = 1.0
        for k in range(1, 100):
            factorial *= k
            term = (-u)**k / (k * factorial)
            sum_term += term
            if abs(term) < 1e-12:
                break
        result = -math.log(u) - 0.5772156649015329 + sum_term
    else:
        # Continued fraction expansion
        max_iter = 100
        tiny = 1e-30
        b = u + 1.0
        d = 1.0 / max(b, tiny)
        c = b
        h = d

        for i in range(1, max_iter):
            a = -i * i
            b += 2.0
            d = 1.0 / max(a * d + b, tiny)
            c = b + a / max(c, tiny)
            delta = c * d
            h *= delta
            if abs(delta - 1.0) < 1e-10:
                break

        result = math.exp(-u) * h

    # Final safety net: exp1(u) must always be > 0
    return max(result, 0.0)


# Parameters
Q_flux = 1e-5       # pumping rate [m^3/s]
#T = 5e-4        # transmissivity [m^2/s]
Transmissivity = 5e-4 
storavity = 1e-3
#Storavity = 1e-3        # storativity [-]
#h0 = 16.0        # initial head [m]
#pump_radius = 0.6  # distance from center to pump wall


# def theis_head_at_pump(t):
#     if t <= 0.0:
#         return h0
#     u = (pump_radius**2 * S) / (4 * T * t)
#     drawdown = (Q / (4 * numpy.pi * T)) * exp1_approx(u)
#     return h0 - drawdown


# Theis parameters
#Q = 1e-3                     # Pumping rate [m³/s]
#T = 5e-4                      # Transmissivity [m²/s]
#S = 1e-4                      # Storativity [-]
# A = 2 * numpy.pi * pump_radius * height
# q_flux = -Q / A   # negative for extraction


#def getDBC_3D_Theis_by_coordinates(x,flag):
#    dx = x[0] - center[0]
#    dy = x[1] - center[1]
#    r = math.sqrt(dx**2 + dy**2)#

#    # Check if we're near either the pump wall or outer wall
#    is_pump_boundary  = abs(r - pump_radius)  < 1e-2
#    is_outer_boundary = abs(r - domain_radius) < 1e-2
#   
#    if is_pump_boundary:
#        def bc_func(x, t):
#            if t<=0:
#                drawdown=0.0
#            else:
#                u = (pump_radius**2 * S) / (4.0 * T * t) #if t > 0 else 1e12  # large u -> small drawdown
#                drawdown = (Q / (4.0 * math.pi * T)) * exp1_approx(u)
#            return height- x[2]- drawdown # Apply DBC up to water level
#        return bc_func
#    if is_outer_boundary:
#        def bc_func(x, t):
#            if t<=0:
#                drawdown=0.0
#            else:
#                u = (domain_radius**2 * S) / (4.0 * T * t) #if t > 0 else 1e12  # large u -> small drawdown
#                drawdown = (Q / (4.0 * math.pi * T)) * exp1_approx(u)
#            return height- x[2]- drawdown # Apply DBC up to water level
#        return bc_func
#    else:
#        return None

logged_points = set()

def getDBC_3D_Theis_by_coordinates(x, flag):
    dx = x[0] - center[0]
    dy = x[1] - center[1]
    r = math.sqrt(dx**2 + dy**2)

    #global theis_output_file, logged_points
    #try:
    #    theis_output_file
    #except NameError:
    #    theis_output_file = open("dbc_theis_log.txt", "w")
    #    theis_output_file.write("time location x y z r drawdown head\n")

    #is_pump_boundary  = abs(r - pump_radius)  < 0.08
    #is_outer_boundary = abs(r - domain_radius) < 0.08

    #if is_pump_boundary or is_outer_boundary:
    #    location = 'pump' if is_pump_boundary else 'outer'
    if flag == boundaryTags['pump'] or flag == boundaryTags['outer']:
        def bc_func(xp, t):
            dx_local = xp[0] - center[0]
            dy_local = xp[1] - center[1]
            r_local = math.sqrt(dx_local**2 + dy_local**2)

            if t <= 0:
                drawdown = 0.0
            else:
                u = (r_local**2 * storavity) / (4.0 * Transmissivity * t)
                drawdown = (Q_flux / (4.0 * math.pi * Transmissivity)) * exp1_approx(u)

            head = height - xp[2] - drawdown

            #key = (round(t, 5), round(xp[0], 3), round(xp[1], 3), round(xp[2], 3))
            #if key not in logged_points:
            #    logged_points.add(key)
            #    theis_output_file.write(
            #        f"{t:.6e} {location} {xp[0]:.4f} {xp[1]:.4f} {xp[2]:.4f} {r_local:.4f} {drawdown:.6f} {head:.6f}\n"
            #    )
            return head

        return bc_func
    else:
        return None




dirichletConditions = {0:getDBC_3D_Theis_by_coordinates}



   
class ShockIC_2D_Richards:
    def uOfXT(self, x, t):
       # return  height- x[2]
        bc=getDBC_3D_Theis_by_coordinates(x,0)
        if bc != None:
            return bc(x,t)
        else:
            return height - x[2] #h0 - x[2]

initialConditions  = {0:ShockIC_2D_Richards()}
    
def getFBC_by_r_coordinates(x, flag):
    if x[2]==0.0 or x[2]== height:
        return lambda x,t : 0.0
    
fluxBoundaryConditions = {0:'setFlow'}

advectiveFluxBoundaryConditions =  {0:getFBC_by_r_coordinates}

diffusiveFluxBoundaryConditions = {0:{0:getFBC_by_r_coordinates}}

T = 100.0/timeScale

