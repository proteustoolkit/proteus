from proteus import *
import os
import numpy
from math import *

details = """
The 2-D Box Problem.  The problem is used to verify velocity consistency for 
the Variable Density Groundwater Flow and Transport in homogeneous porous medium.
There are two different cases that are analyzed with different parameters for the model that are 
outlined in the PETTT Report. Case 1 parameters and set up are taken from Diersch and Kolditz 2002.  
Case 2 parameters and set up are taken from Voss and Souza 1987.   A third case, Case 3 from Voss and Souza 1987,
the horizontal flow case, needs development.
The primitive variables are the mass fraction (mf) and the freshwater pressure head (h)
"""

###########################
# Defining Spatial, Temporal Domain Data and
# Numerical Information common to both nfiles
###########################

# flag to execute in parrallel or structured/unstructed domains
transient=True
parallel=False #True = Parallel, False = Serial
unstructured=False # True = Structured, False = Unstructured
consistentVelocityFlag=int(os.environ.get('GWVD_CV', 0)) # 1= CV, 0 = No CV (flag to implement consistent velocity algorithm)
vel_pp=int(os.environ.get('GWVD_PP', 1))  # 1 = 'PWL', 0 = 'point_eval', Velocity Post Processing Algorithm

#spatial domain/ number of dimensions
L=(20.0,40.0,1) #m
nd = 2

# setting structured/unstructured domains
if (unstructured):
    #myDomain = Domain.RectangularDomain(L=[L[0],L[1]],name="gwdtrans_domain",units="m")
    #polyfile="gwvdtransport_domain_2d"
    #myDomain.writePoly(polyfile)
    polyfile="gwvdtransport_domain_2d_2domain"
    myDomain = Domain.PlanarStraightLineGraphDomain(fileprefix=polyfile)
    triangleOptions = "q30Dena1.0A"
else:
    nnx =32 # 48
    nny =64#96

#time information
T =86400.*float(os.environ.get('GWVD_T_DAYS', 1000.)) #1000 #8.64e5 #1.0e4 #time scale (sec)
nDTout = 100 #number of output timesteps to use
DT=1e4
#number of components of solutions
ncflow = 1
nctrans= 1
gw_quad_order = 3 # quadrature order

###########################
# Setting Material Parameters and Functions
###########################

# Density Relation
rho_fw =  1000.0 # density of freshwater
rho_sw =float(os.environ.get('GWVD_RHO_SW', 1200.0)) # density of saltwater or maximum problem density
mf_fw = 0.0 #  mass fraction, freshwater
mf_sw=1.0 # mass fraction, saltwater of highest density
epsilon =(rho_sw - rho_fw)/rho_fw 
drho_dmf = epsilon*rho_fw # density change with mass fraction

# Dynamic Viscosity Relation
nu = 1.004e-3

# Material Constants
alpha_L =0.0#L[1]/63. # 0.0 # Case 1 # L[1]/63.# Case 2  
alpha_T =alpha_L #alpha_L/10.0 # Case 2 # 0.0 Case 1
omega = 0.3 # porosity
d_mol=1.e-8 # Case 1 # 0.0  Case 2 #molecular diffusion coefficient   
gc = 9.81  #gravitational constant
xi = 1.0e-8  # bulk porous medium compressibility
beta=0.0 # 4.6e-10 # fluid compressibility 

if (transient):
    SP =xi*(1.-omega)+omega*beta   # specific mass storativity wrt to pressure changes
else:
    SP =0.0

#gravity vector (needed for now by consistent velocity approximations)

if nd == 2:
    gravity =numpy.array([0.0,1.0]) # numpy.array([0.0,-gc])
elif nd == 3:
    gravity =numpy.array([0.0,0.0,1.0]) # numpy.array([0.0,0.0,-gc]) # 
else:
    gravity =numpy.array([1.0]) # numpy.array([-gc]) # 
    
# Porous Media Permeability

Kx = 1.0e-11 #5.0e-3# / 2.0**8 #m/s
Ky = Kx #1.0e-6 #m/s

# specific mass storativity related to head changes
S = rho_fw*gc*SP


def hydraulicConductivity(x):
    return numpy.array([[Kx*rho_fw*gc/nu,0.0],
                        [0.0,Ky*rho_fw*gc/nu]],'d')

def fluidSourceTerms(x): #Need to update to add in source term, Q_s
    return  0.0

def bouyancySourceTerm(mf):
    buoyancy = numpy.array([[0.0],[epsilon*mf]])
    A = numpy.reshape(hydraulicConductivity(1), (2,2))
    return numpy.dot(A,buoyancy) 
     
def reactionTerms(x,mf,dmfdt):
    return omega/(rho_fw*(1.+epsilon*mf))*drho_dmf*dmfdt - rho_fw*fluidSourceTerms(x)/(rho_fw*(1.+epsilon*mf))


###########################
# Defining Initial Conditions Functions
###########################

# constant initial condition

class constantIC:
    def __init__(self,cval=0.0):
        self.cval = cval
    def uOfXT(self,x,t):
        return self.cval
    def uOfX(self,x):
        return self.cval

# variable concentration initial condition

class variableIC:
    def __init__(self,val_1 = 0.0,val_2 = 1.0):
        self.val_1 = val_1
        self.val_2 = val_2        
    def uOfXT(self,x,t):
        if x[1] >= L[1]/2.: 
            return self.val_1
        if x[1] < L[1]/2.:  
            return self.val_2
        #return 0.0
    def uOfX(self,x):
        if x[1] >= L[1]/2.:
            return self.val_1
        if x[1] < L[1]/2.:  
            return self.val_2

# hydrostatic freshwater head condition

class hydrostaticIC:
    def __init__(self,val_1 = 0.0,val_2 = 1.0,h_s=0.0):
        self.val_1 = val_1
        self.val_2 = val_2
        self.h_s = h_s
    def uOfXT(self,x,t):
        if x[0] < L[1]/2.: 
            #return self.h_s + (rho_sw-rho_fw)/rho_fw*(L[1]/2. - x[1])   
            return self.h_s + (L[1]/2.-x[1])*epsilon*self.val_2 
        if x[0] >= L[1]/2.: 
            #return self.h_s + (L[1]-x[1])*epsilon*self.val_1
            return self.h_s

###########################
# Setting Boundary Conditions and Initial Conditions for Flow
###########################

# Dummy Boundary Conditions for Pass

def dummyBCs(x,tag):
    #if (x[0] < 1.e-8 and x[1] > L[1]-1.e-8):
       # return lambda x,t: 0.0 
    pass

def dummyflowBCs(x,tag):
    if (x[0]==0.0 and x[1]==L[1]): 
        return lambda x,t: L[1]     
    elif (x[0]==L[0] and x[1] == L[1]):
        return lambda x,t: L[1]
    else:
        pass

# Case 3 Side Flow Problem boundary conditions - For Horizontal Flow Case Not used in report
#
#def flowHeadBCs(x,tag,val_1=0.0,val_2 = 1.0, h_0=0.0):
#    if (x[0] == L[0] and (x[1] >= L[1]/2. and x[1] < L[1]-1.e-8)):
#        return lambda x,t: h_0 + 0.999*(L[1]-x[1])*epsilon*val_1 #m
#    elif ( x[0] == 0.0 and (x[1] >= L[1]/2. and x[1] < L[1]-1.e-8)):
#        return lambda x,t: h_0 + (L[1]-x[1])*epsilon*val_2 #m
#    elif (x[0] == L[0]  and (x[1] > 1.e-8 and x[1] < L[1]/2.)):
#        return lambda x,t: h_0 +0.999*( (L[1]/2.-x[1])*rho_sw/rho_fw + L[1]/2.) #m
#    elif (x[0] == 0.0 and (x[1] > 1.e-8 and x[1] < L[1]/2.)):
#        return lambda x,t: h_0 + (L[1]/2.-x[1])*rho_sw/rho_fw + L[1]/2. #m

#def gettopNoFlowBCs(x,tag):
#    if ( x[1]==0.0 or x[1] == L[1] ):
#        return lambda x,t: 0.0

# Velocity Consistency Verification boundary Conditions For Case 1 and 2

def getNoFlowBCs(x,tag):
    if (x[1] < 1.0e-8 or 
        x[1] > L[1]-1.0e-8 or
        x[0] < 1.0e-8  or
        x[0] > L[0]-1.0e-8):
        return lambda x,t: 0.0
    
###########################
# Setting Concentration and Mass Flux Boundary Conditions
# (Dirichlet, Neumann and Mixed)
###########################

# Dirichlet Boundary Conditions for transport

# For Case 3 Boundary Conditions - Not used 

#def getsideDirichletBCs(x,tag):
#    if (( x[0]==0.0 or x[0]==L[0])
#        and (x[1] >= L[1]/2. and x[1] < L[1]-1.e-8)):
#        return lambda x,t: mf_fw
#    elif (( x[0]==0.0 or x[0]==L[0])
#        and (x[1] <  L[1]/2. and x[1] > 0.0 )):
#        return lambda x,t: mf_sw

#def gettopZeroAdvectiveFlux(x,tag):
#    if (x[1] == L[1] or x[1]==0.0):
#        return lambda x,t: 0.0
   
#def gettopZeroDiffusiveFlux(x,tag):
#    if (x[1] == L[1] or x[1]==0.0):
#        return lambda x,t: 0.0      
 

#Neumann (flux) Boundary Conditions for transport
# Used in Case 1 and Case 2 for the problem

def getZeroAdvectiveFlux(x,tag):
    if (x[0] < 1.0e-8 or x[0] > L[0]-1.0e-8 or
        x[1] < 1.0e-8 or x[1] > L[1]-1.0e-8 ):
        return lambda x,t: 0.0
   
def getZeroDiffusiveFlux(x,tag):
    if (x[0] < 1.0e-8 or x[0] > L[0]-1.0e-8 or 
        x[1] < 1.0e-8 or x[1] > L[1]-1.0e-8):
        return lambda x,t: 0.0

##########################
# Setting Analyical Solutions Flags
##########################

transportAnalyticalSolution = None
velocityAnalyticalSolution = None
flowAnalyticalSolution = None




