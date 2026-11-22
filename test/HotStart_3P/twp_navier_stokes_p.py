from proteus import *
from proteus.default_p import *
from NS_hotstart import *
from proteus.kernels import RANS3PF


LevelModelType = RANS3PF.LevelModel

name="momentum_eqn"

coefficients = RANS3PF.Coefficients(epsFact=epsFact_viscosity,
                                    sigma=0.0,
                                    rho_0 = rho_0,
                                    nu_0 = nu_0,
                                    rho_1 = rho_1,
                                    nu_1 = nu_1,
                                    g=g,
                                    nd=nd,
                                    ME_model=0,
                                    PRESSURE_model=2,
                                    SED_model=None,
                                    VOS_model=None,
                                    VOF_model=None,
                                    LS_model=None,
                                    Closure_0_model=None,
                                    Closure_1_model=None,
                                    epsFact_density=epsFact_density,
                                    stokes=False,
                                    useVF=useVF,
                                    useRBLES=useRBLES,
                                    useMetrics=useMetrics,
                                    eb_adjoint_sigma=1.0,
                                    eb_penalty_constant=weak_bc_penalty_constant,
                                    forceStrongDirichlet=ns_forceStrongDirichlet,
                                    turbulenceClosureModel=ns_closure,
                                    movingDomain=movingDomain,
                                    dragAlpha=dragAlpha,
                                    PSTAB=1.0, 
                                    cE=cE,
                                    cMax=cMax, 
                                    CORRECT_VELOCITY=CORRECT_VELOCITY)

#######################
# BOUNDARY CONDITIONS #
#######################
def getDBC_u(x,flag):
    #None
    pi = np.pi
    if (flag==1 or flag==2 or flag==3 or flag==4):
        if manufactured_solution == 1:
            return lambda x,t: np.sin(x[0])*np.sin(x[1]+t)
        else: 
            return lambda x,t: np.sin(pi*x[0])*np.cos(pi*x[1])*np.sin(t)

def getDBC_v(x,flag):
    #None
    pi = np.pi
    if (flag==1 or flag==2 or flag==3 or flag==4):
        if manufactured_solution == 1:
            return lambda x,t: np.cos(x[0])*np.cos(x[1]+t)
        else:
            return lambda x,t: -np.cos(pi*x[0])*np.sin(pi*x[1])*np.sin(t)

def getAFBC_u(x,flag):
    None

def getAFBC_v(x,flag):
    None
    
def getDFBC_u(x,flag):
    # Set grad(u).n    
    None

def getDFBC_v(x,flag):
    None

dirichletConditions = {0:getDBC_u,
                       1:getDBC_v}
advectiveFluxBoundaryConditions =  {0:getAFBC_u,
                                    1:getAFBC_v}
diffusiveFluxBoundaryConditions = {0:{0:getDFBC_u},
                                   1:{1:getDFBC_v}}

######################
# INITIAL CONDITIONS #
######################
class AtRest(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

class velx_at_t0(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        if manufactured_solution == 1:
            return np.sin(x[0])*np.sin(x[1])
        else: 
            return 0.

class vely_at_t0(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        if manufactured_solution == 1:
            return np.cos(x[0])*np.cos(x[1])
        else: 
            return 0.

initialConditions = {0:velx_at_t0(),
                     1:vely_at_t0()}

#############################
# MATERIAL PARAMETER FIELDS #
#############################
def density(X,t):
    x = X[0]
    y = X[1]
    return np.sin(x+y+t)**2+1

mu_constant=False
def dynamic_viscosity(X,t):
    x = X[0]
    y = X[1]
    return mu*(np.cos(x+y+t)**2+1)
    
materialParameters = {'density':density, 
                      'dynamic_viscosity':dynamic_viscosity}

###############
# FORCE TERMS #
###############
def forcex(X,t):
    x = X[0]
    y = X[1]
    rho = density(X,t)
    #pi = np.pi
    if manufactured_solution == 1: #u.n!=0
        return (rho*np.sin(x)*np.cos(y+t) # Time derivative
                + rho*np.sin(x)*np.cos(x) # Non-linearity
                - (0. if KILL_PRESSURE_TERM==True else 1.)*np.sin(x)*np.sin(y+t) # Pressure
                + (2*dynamic_viscosity(X,t)*np.sin(x)*np.sin(y+t) # Diffusion
                   +(0. if mu_constant==True else 1.)*mu*2*np.cos(x+y+t)*np.sin(x+y+t)*(np.cos(x)*np.sin(y+t)+np.sin(x)*np.cos(y+t)) 
                   +(0. if mu_constant==True else 1.)*mu*2*np.cos(x+y+t)*np.sin(x+y+t)*(np.cos(x)*np.sin(y+t)-np.sin(x)*np.cos(y+t)))
            )
    else: # u.n=0
        return (rho*np.sin(pi*x)*np.cos(pi*y)*np.cos(t) # Time derivative                  
                + rho*pi*np.sin(pi*x)*np.cos(pi*x)*np.sin(t)**2 # non-linearity
                - (0. if KILL_PRESSURE_TERM==True else 1.)*np.sin(x)*np.sin(y+t) # Pressure
                - dynamic_viscosity(X,t)*(-2*pi**2*np.sin(pi*x)*np.cos(pi*y)*np.sin(t)) # Diffusion
            )

def forcey(X,t):
    x = X[0]
    y = X[1]
    rho = density(X,t)
    pi = np.pi
    if manufactured_solution == 1: #u.n!=0
        return (-rho*np.cos(x)*np.sin(y+t) # Time derivative
                - rho*np.sin(y+t)*np.cos(y+t) #Non-linearity
                + (0. if KILL_PRESSURE_TERM==True else 1.)*np.cos(x)*np.cos(y+t) #Pressure
                + (2*dynamic_viscosity(X,t)*np.cos(x)*np.cos(y+t) # Diffusion
                   +(0. if mu_constant==True else 1.)*mu*2*np.cos(x+y+t)*np.sin(x+y+t)*(-np.sin(x)*np.cos(y+t)-np.cos(x)*np.sin(y+t))
                   +(0. if mu_constant==True else 1.)*mu*2*np.cos(x+y+t)*np.sin(x+y+t)*(np.sin(x)*np.cos(y+t)-np.cos(x)*np.sin(y+t)))
            )
    else:
        return (-rho*np.cos(pi*x)*np.sin(pi*y)*np.cos(t) # Time derivative
                + rho*pi*np.sin(pi*y)*np.cos(pi*y)*np.sin(t)**2 # non-linearity
                + (0. if KILL_PRESSURE_TERM==True else 1.)*np.cos(x)*np.cos(y+t) #Pressure                
                - dynamic_viscosity(X,t)*(2*pi**2*np.cos(pi*x)*np.sin(pi*y)*np.sin(t)) # Diffusion
            )

forceTerms = {0:forcex,
              1:forcey}

##################
# EXACT SOLUTION #
##################
class velx(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        pi = np.pi
        if manufactured_solution == 1:
            return np.sin(x[0])*np.sin(x[1]+t)
        else: 
            return np.sin(pi*x[0])*np.cos(pi*x[1])*np.sin(t)
    def duOfXT(self,x,t):
        if manufactured_solution == 1:
            return [np.cos(x[0])*np.sin(x[1]+t),
                    np.sin(x[0])*np.cos(x[1]+t)]
        else: 
            return [pi*np.cos(pi*x[0])*np.cos(pi*x[1])*np.sin(t),
                    -pi*np.sin(pi*x[0])*np.sin(pi*x[1])*np.sin(t)]
            
class vely(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        pi = np.pi
        if manufactured_solution == 1:
            return  np.cos(x[0])*np.cos(x[1]+t)
        else:
            return -np.cos(pi*x[0])*np.sin(pi*x[1])*np.sin(t)
    def duOfXT(self,x,t):
        if manufactured_solution == 1:
            return [-np.sin(x[0])*np.cos(x[1]+t),
                    -np.cos(x[0])*np.sin(x[1]+t)]
        else:
            return [pi*np.sin(pi*x[0])*np.sin(pi*x[1])*np.sin(t),
                    -pi*np.cos(pi*x[0])*np.cos(pi*x[1])*np.sin(t)]

analyticalSolution = {0:velx(), 
                      1:vely()}

class pressure(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return np.cos(x[0])*np.sin(x[1]+t)
analyticalPressureSolution={0:pressure()}
