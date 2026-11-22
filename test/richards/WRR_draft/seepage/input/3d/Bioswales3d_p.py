from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards


nd = 3

regularGrid=False
## Domain
boundaries = ['bottom','top','left','right','front', 'back',
                'base', 'storage',
                'swale', 
                'drain']
Length= 5.0
slope= 4/100 #/100  # 4/100

boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])
vertices3dt= [[1.0, 0.0, 1.0], #0
             [2.0, 0.0, 1.0],  #1
             [2.5, 0.0, 3.0],  #2
             [0.5, 0.0, 3.0],  #3
             
             [1.0, Length, 1.0-Length*slope], #4
             [2.0, Length, 1.0-Length*slope],  #5
             [2.5, Length, 3.0], #6
             [0.5, Length, 3.0], #7
              
             [1.2, 0.0, 1.4], #8
             [1.7, 0.0, 1.4], #9
             [1.7, 0.0, 1.9], #10
             [1.2, 0.0, 1.9], #11
             
             [1.2, Length, 1.4-Length*slope], #12
             [1.7, Length, 1.4-Length*slope], #13
             [1.7, Length, 1.9-Length*slope], #14
             [1.2, Length, 1.9-Length*slope], #15
             
             
             [0.0,0.0,0.0], #16
             [3.0,0.0,0.0], #17
             [3.0,0.0,3.0], #18
             [0.0,0.0,3.0], #19
             
             [0.0,Length ,0.0-Length*slope], #20
             [3.0,Length,0.0-Length*slope], #21
             [3.0,Length,3.0], #22
             [0.0,Length,3.0], #23
              
             [0.0 ,0.0, 5.0], #24
             [3.0,0.0, 5.0], #25
             [0.0 , Length, 5.0], #26
             [3.0,  Length, 5.0],] #27
           

vertexFlags3dt=[boundaryTags['swale'], #0
                boundaryTags['swale'], #1
                boundaryTags['swale'], #2
                boundaryTags['swale'], #3
                   
                boundaryTags['swale'], #4
                boundaryTags['swale'], #5
                boundaryTags['swale'], #6
                boundaryTags['swale'], #7
                
               
                boundaryTags['drain'], #8
                boundaryTags['drain'], #9
                boundaryTags['drain'], #10
                boundaryTags['drain'], #11

                boundaryTags['drain'], #12
                boundaryTags['drain'], #13
                boundaryTags['drain'], #14
                boundaryTags['drain'], #15
               
                boundaryTags['bottom'], #16
                boundaryTags['bottom'], #17
                boundaryTags['left'], #18
                boundaryTags['left'], #19
                
                boundaryTags['bottom'], #20
                boundaryTags['bottom'], #21
                boundaryTags['right'], #22
                boundaryTags['right'], #23
               
                boundaryTags['top'], #24
                boundaryTags['top'], #25
                boundaryTags['top'], #26
                boundaryTags['top'], #27
                ] 
                
    
facets3dt= [[[0,1,9,8]], # 0 storage left
            [[1,2,10,9]], #1 #storage right
            [[10,11,3,2]], #2 bottom
            [[8,11,3,0]], #3 base right


            [[4,5,13,12]], #4
            [[5,6,14,13]], #5 base bottom
            [[14,15,7,6]], #6  base bottom
            [[12,15,7,4]], #7 back

            [[1,5,13,9]], #8
            [[2,6,14,10]],#9
            [[3,7,15,11]],#10
            [[0,4,12,8]], #11

            [[1,5,6,2]], #12
            [[2,6,7,3]], #13
            [[3,7,4,0]], #14
            [[0,4,5,1]], #15
            [[8,9,13,12]], #16
            [[9,13,14,10]], #17
            [[10,14,15,11]],#18
            [[11,15,12,8]], #19

            [[16,17,1,0]], #20
            [[17,1,2,18]], #21
            [[16,0,3,19]], #22
            
            
            [[20,21,5,4]], #23
            [[21,5,6,22]], #24
            [[20,4,7,23]], #25

            [[16,17,21,20]], #26
            [[17,21,22,18]], #27
            [[18,22,6,2]], #28
            [[3,7,23,19]], #29
            [[19,23,20,16]], #30
            
            
            [[2,18,25]], #31
            [[3,19,24]], #32
            [[2,3,24,25]], #33

            [[6,22,27]], #34
            [[7,23,26]], #35
            [[6,7,26,27]], #36

            [[18,22,27,25]], #37
            [[19,24,26,23]], #38
            [[24,25,27,26]],] #39
            #[[2,6,27,25]], #40
            #[[3,24,26,27]] #41
        
facetFlagsdt=[boundaryTags['left'], #0
              boundaryTags['left'], #1
              boundaryTags['left'], #2
              boundaryTags['left'], #3
              
              boundaryTags['right'], #4
              boundaryTags['right'], #5
              boundaryTags['right'], #6
              boundaryTags['right'], #7
              
              boundaryTags['swale'], #8
              boundaryTags['swale'], #9
              boundaryTags['swale'], #10
              boundaryTags['swale'], #11
              
              boundaryTags['swale'], #12
              boundaryTags['swale'], #13
              boundaryTags['swale'], #14
              boundaryTags['swale'], #15
              
              boundaryTags['drain'], #16
              boundaryTags['drain'], #17
              boundaryTags['drain'], #18
              boundaryTags['drain'], #19
              
              boundaryTags['left'], #20
              boundaryTags['left'], #21
              boundaryTags['left'], #22
                                         
              boundaryTags['right'], #23
              boundaryTags['right'], #24
              boundaryTags['right'], #25
              
              boundaryTags['bottom'], #26
              boundaryTags['front'], #27
              boundaryTags['base'], #28
              boundaryTags['base'], #29
              boundaryTags['back'], #30
             
              boundaryTags['storage'], #31
              boundaryTags['storage'], #32
              boundaryTags['storage'], #33
              
              boundaryTags['storage'], #34
              boundaryTags['storage'], #35
              boundaryTags['storage'], #36
              
              boundaryTags['front'], #37
              boundaryTags['back'], #38
              boundaryTags['top'], #39
             ] 
              

holes3d=[[1.5,0.001, 1.5]]
#regions3dt=[[1.5,0.001,0.001],[0.1,0.001,1.0],[1.99,0.001,1.0],[1,5,0.001,2.9],[1.5,0.001,3.7]]

#regions3dt=[[1.5,0.001,0.001],[1.99,0.001,1.0],[0.001,0.001,1.0],
#            [1.5,0.001,1.1],[1.5,0.001,1.95],[1.,0.001,1.2],[2.,0.001,1.2],
#            [0.001,0.001,3.7],[1.5,0.001,3.7],[2.999,0.001,3.7]]
            
regions3dt=[[1.5,1.0,0.5],[2.6,1.0,1.5],[0.2,1.0,1.5],
            [1.5,1.0,1.2],[1.8,1.0,1.5],[1.5,1.0,2.5],[1.0,1.0,1.5],
            [0.005,1.0,4.0],[1.5,1.0,3.7],[2.9,1.0,4.0]]

regionFlagsdt= [1,1,1,
                2,2,2,2,
                3,3,3]

domain = Domain.PiecewiseLinearComplexDomain(vertices=vertices3dt,
                                                 vertexFlags=vertexFlags3dt,
                                                 facets=facets3dt,
                                                 facetFlags=facetFlagsdt,
                                                 regions=regions3dt,
                                                 regionFlags=regionFlagsdt,#)
                                                 holes= holes3d)

domain.writePoly('bio3d')

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0
#beta          = density*gravity*4.524e-10
m_per_s_by_m_per_d = 1.1574074e-5 

#print 'perm',permeability
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)

# washed site
thetaS3        = 0.4   #-
thetaR3        = 0.05   #-
mvg_alpha3     = 8    #1/m
mvg_n3         = 2.4
mvg_m3         = 1.0 - 1.0/mvg_n3
permeability3  = (5.00*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #m^2
dimensionless_conductivity3  = (timeScale*density*gravity*permeability3/(viscosity*lengthScale))/m_per_s_by_m_per_d


# Swale Channel
thetaS2        = 0.43   #-
thetaR2        = 0.045   #-
mvg_alpha2     = 14.5   #1/m
mvg_n2         = 2.68
mvg_m2         = 1.0 - 1.0/mvg_n2
permeability2  = (7.128*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #m^2
dimensionless_conductivity2  = (timeScale*density*gravity*permeability2/(viscosity*lengthScale))/m_per_s_by_m_per_d

# base
thetaS1        = 0.41   #-
thetaR1        = 0.065   #-
mvg_alpha1     = 7.5   #1/m
mvg_n1         = 1.89
mvg_m1         = 1.0 - 1.0/mvg_n1
permeability1  = (1.06*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #m^2
dimensionless_conductivity1  = (timeScale*density*gravity*permeability1/(viscosity*lengthScale))/m_per_s_by_m_per_d


dimensionless_density  = 1.0
dimensionless_gravity  = numpy.array([0.0,
                                        0.0,
                                        -1.0])
#dimensionless_alpha    = mvg_alpha*lengthScale
nMediaTypes  = 3
alphaVGtypes = numpy.zeros((nMediaTypes+1,),'d')
nVGtypes     = numpy.zeros((nMediaTypes+1,),'d')
thetaStypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes+1,),'d')
KsTypes      = numpy.zeros((nMediaTypes+1,3),'d')

for i in range(nMediaTypes+1):
    if i==1: 
        alphaVGtypes[i] = mvg_alpha1
        nVGtypes[i]     = mvg_n1
        thetaStypes[i]  = thetaS1
        thetaRtypes[i]  = thetaR1
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity1,dimensionless_conductivity1, dimensionless_conductivity1]#m/d?
    elif i==2:
        alphaVGtypes[i] = mvg_alpha2
        nVGtypes[i]     = mvg_n2
        thetaStypes[i]  = thetaS2
        thetaRtypes[i]  = thetaR2
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity2,dimensionless_conductivity2, dimensionless_conductivity2]#m/d? 
        
    elif i==3:
        alphaVGtypes[i] = mvg_alpha3
        nVGtypes[i]     = mvg_n3
        thetaStypes[i]  = thetaS3
        thetaRtypes[i]  = thetaR3
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity3,dimensionless_conductivity3,dimensionless_conductivity3]#m/d?
    else:
        alphaVGtypes[i] = mvg_alpha1
        nVGtypes[i]     = mvg_n1
        thetaStypes[i]  = thetaS1
        thetaRtypes[i]  = thetaR1
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity1,dimensionless_conductivity1,dimensionless_conductivity1]#m/d? 
        
        
galerkin = False   
useSeepageFace = True

def getSeepageFace(flag):
    if useSeepageFace:
        if flag == boundaryTags['drain']:
            return 1
        else:
            return 0
    else:
        return 0
        
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
                                     FCT=False ,#True,
                                     num_fct_iter=0,
                                     # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False,
                                     getSeepageFace=getSeepageFace)



#L=[3.0,5.0,5.0]
pondingPressure= 0.5

def getDBC_3D_Richards_Shock(x,flag):
    if flag == boundaryTags['top'] and 1.0 < x[0] < 2.0:
    #if x[2] == L[2]:
        return lambda x,t: pondingPressure 
    #if flag == boundaryTags['bottom']:
    #    return lambda x,t: 0.0        
    #if x[2] == L[2]:
    #    if (x[1] >= L[1]/3.0 and
    #        x[1] <= 2.0*L[1]/3.0 and
    #        x[1] >= L[1]/3.0 and
    #        x[1] <= 2.0*L[1]/3.0):
    #        return lambda x,t: pondingPressure
    if x[2] == 0.0:
        return lambda x,t: 0.0
#     if (x[0] == 0.0 or
#         x[0] == L[0] or
#         x[1] == 0.0 or
#         x[1] == L[1]):
#         return lambda x,t: x[2]*dimensionless_gravity[2]*dimensionless_density

dirichletConditions = {0:getDBC_3D_Richards_Shock}

class ShockIC_3D_Richards:
    def uOfXT(self,x,t):
        #bc = getDBC_3D_Richards_Shock(x,0)
        #if bc != None:
        #    return bc(x,t)
        #else:
        return x[2]*dimensionless_gravity[2]*dimensionless_density

initialConditions  = {0:ShockIC_3D_Richards()}

def getFBC_3D_Richards_Shock(x,flag):
    return None 
    #if (x[0] == 0.0 or
    #    x[0] == L[0] or
    #    x[1] == 0.0 or
    #    x[1] == L[1]):
    #    return lambda x,t: 0.0
    #if x[2] == L[2]:
    #    if not (x[0] >= L[0]/3.0 and
    #            x[0] <= 2.0*L[0]/3.0 and
    #            x[1] >= L[1]/3.0 and
    #            x[1] <= 2.0*L[1]/3.0):
    #        return lambda x,t: 0.0

fluxBoundaryConditions = {0:'noFlow'}

advectiveFluxBoundaryConditions =  {0:getFBC_3D_Richards_Shock}

diffusiveFluxBoundaryConditions = {0:{0:getFBC_3D_Richards_Shock}}

#T = 0.1/timeScale

T = 0.25/timeScale

