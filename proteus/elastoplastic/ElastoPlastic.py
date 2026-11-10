import proteus
from proteus import FemTools, Archiver
from .cElastoPlastic import *
from math import atan, tan
import numpy as np
from proteus import cmeshTools
from proteus.Transport import OneLevelTransport, sqrt, TC_base
from proteus.Transport import NonlinearEquation, logEvent, memory
from proteus.Transport import FluxBoundaryConditions, Comm, cfemIntegrals
from proteus.Transport import DOFBoundaryConditions, Quadrature
from proteus.mprans import cArgumentsDict
import proteus.Transport
import proteus.TransportCoefficients

class Coefficients(proteus.TransportCoefficients.TC_base):
    def __init__(self,
                 modelType_block,
                 modelParams_block,
                 g=[0.0,0.0,-9.8],#gravitational acceleration
                 rhow=998.2,#kg/m^3 water density (used if pore pressures specified)
                 pa=101325.0,#N/m^2 atmospheric pressure
                 nd=3,
                 meIndex=0,seepageIndex=-1,SRF=1.0,pore_pressure_file_base=None,pore_pressure_field_path=None):
        import copy
        self.modelType_block = modelType_block
        self.modelParams_block = modelParams_block
        self.materialProperties = self.modelParams_block
        self.materialProperties_default = copy.deepcopy(self.modelParams_block)
        self.nMaterialProperties = len(self.materialProperties[-1])
        self.SRF=SRF
        self.g = np.array(g)
        self.gmag = sqrt(sum([gi**2 for gi in g]))
        self.rhow=rhow
        self.pore_fluid_unit_weight = self.gmag*self.rhow
        print("pore_fluid_unit_weight", self.pore_fluid_unit_weight)
        print("soil_unit_weight", self.materialProperties[0,13]*self.gmag)
        self.pa=pa
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        stress={}
        assert(nd==3)
        variableNames=['hx','hy','hz']
        mass = {0:{0:'linear'},
                1:{1:'linear'},
                2:{2:'linear'}}
        reaction = {0:{0:'constant'},
                    1:{1:'constant'},
                    2:{2:'constant'}}
        stress= {0:{0:'linear',1:'linear',2:'linear'},
                 1:{0:'linear',1:'linear',2:'linear'},
                 2:{0:'linear',1:'linear',2:'linear'}}
        TC_base.__init__(self,
                         3,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         stress=stress)
        self.vectorComponents=[0,1,2]
        self.vectorName="displacement"
        self.firstCall=True
        self.gravityStep=1
        self.lastStepWasGravityStep=False
        self.meIndex = meIndex
        self.seepageIndex = seepageIndex
        self.copyInstructions = {'reset_uList':True}
        self.pore_pressure_file_base=pore_pressure_file_base
        self.pore_pressure_field_path=pore_pressure_field_path
    def attachModels(self,modelList):
        self.model = modelList[self.meIndex]
        self.pore_pressure_head = np.zeros((self.model.mesh.nodeArray.shape[0],),'d')
        if self.seepageIndex > 0:
            self.seepageModel = modelList[self.seepageIndex]
            self.pore_pressure_head_save = self.seepageModel.u.dof[0]
            self.pre_pressure_head[:]=self.pore_pressure_head_save
        elif self.pore_pressure_file_base != None:
            import h5py
            archive = h5py.File(self.pore_pressure_file_base+".h5","r")
            permute = np.argsort(self.mesh.globalMesh.nodeNumbering_subdomain2global)
            self.pore_pressure_head[permute] = archive[self.pore_pressure_field_path][self.mesh.globalMesh.nodeNumbering_subdomain2global[permute].tolist()]
            self.pore_pressure_head_save = self.pore_pressure_head.copy()
        else:
            self.pore_pressure_head_save = np.zeros((self.model.mesh.nodeArray.shape[0],),'d')
    def initializeElementQuadrature(self,t,cq):
        """
        Give the TC object access to the element quadrature storage
        """
        if self.firstCall:
            self.firstCall=False
        self.cq=cq
        cq['strain0'] = cq['strain'].copy()
        cq['strain_last'] = cq['strain'].copy()
        cq['plasticStrain'] = cq['strain'].copy()
        cq['plasticStrain_last'] = cq['strain'].copy()
        cq['strain_last'][:]=0.0
        cq['plasticStrain'][:]=0.0
        cq['plasticStrain_last'][:]=0.0
        self.bodyForce = cq['bodyForce']
        for eN in range(self.bodyForce.shape[0]):
            rhos = self.materialProperties[self.mesh.elementMaterialTypes[eN],13]
            for k in range(self.bodyForce.shape[1]):
                self.bodyForce[eN,k,:] = self.g*rhos
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        """
        Give the TC object access to the element quadrature storage
        """
        self.cebqe = cebqe
        cebqe['strain0'] = cebqe['strain'].copy()
        cebqe['strain_last'] = cebqe['strain'].copy()
        cebqe['plasticStrain'] = cebqe['strain'].copy()
        cebqe['plasticStrain_last'] = cebqe['strain'].copy()
        cebqe['strain_last'][:]=0.0
        cebqe['plasticStrain'][:]=0.0
        cebqe['plasticStrain_last'][:]=0.0
    def initializeMesh(self,mesh):
        self.mesh = mesh
    def postStep(self,t,firstStep=False):
        if self.gravityStep:
            if self.gravityStep == 1:
                self.cq['strain0'][:]=self.cq['strain']
                self.cebqe['strain0'][:] = self.cebqe['strain']
                self.gravityStep = 2
            else:
                self.gravityStep = 0
            self.cq['strain_last'][:] = self.cq['strain']
            self.cq['plasticStrain_last'][:] = 0.0#self.cq['plasticStrain']
            self.cebqe['strain_last'][:] = self.cebqe['strain']
            self.cebqe['plasticStrain_last'][:] = 0.0#self.cebqe['plasticStrain']
            self.pore_pressure_head[:] = self.pore_pressure_head_save
            self.lastStepWasGravityStep=True
            #self.model.u[0].dof[:]=0.0
            #self.model.u[1].dof[:]=0.0
            #self.model.u[2].dof[:]=0.0
            for it in range(self.materialProperties.shape[0]):
                #cek hack for veg on levees
                if it != 7:
                    self.materialProperties[it,5] = atan(tan(self.materialProperties_default[it,5])/self.SRF)#phi_mc
                    self.materialProperties[it,6] = self.materialProperties_default[it,6]/self.SRF#c_mc
            self.copyInstructions = {'reset_uList':True}
        else:
            print("Completed========================SRF = "+repr(self.SRF)+"===============================")
            self.lastStepWasGravityStep = False
            self.SRF += 0.05
            for it in range(self.materialProperties.shape[0]):
                if it != 7:
                    self.materialProperties[it,5] = atan(tan(self.materialProperties_default[it,5])/self.SRF)#phi_mc
                    self.materialProperties[it,6] = self.materialProperties_default[it,6]/self.SRF#c_mc
            self.copyInstructions = None
        print("=========Not Updating Mesh=================")
        #self.mesh.nodeArray[:,0]+=self.model.u[0].dof
        #self.mesh.nodeArray[:,1]+=self.model.u[1].dof
        #self.mesh.nodeArray[:,2]+=self.model.u[2].dof
    def preStep(self,t,firstStep=False):
        print("Starting========================SRF = "+repr(self.SRF)+"===============================")
        if self.lastStepWasGravityStep:
            self.model.u[0].dof[:]=0.0
            self.model.u[1].dof[:]=0.0
            self.model.u[2].dof[:]=0.0
        return self.copyInstructions
    def evaluate(self,t,c):
        pass

class LevelModel(proteus.Transport.OneLevelTransport):
    nCalls=0
    def __init__(self,
                 uDict,
                 phiDict,
                 testSpaceDict,
                 matType,
                 dofBoundaryConditionsDict,
                 dofBoundaryConditionsSetterDict,
                 coefficients,
                 elementQuadrature,
                 elementBoundaryQuadrature,
                 fluxBoundaryConditionsDict=None,
                 advectiveFluxBoundaryConditionsSetterDict=None,
                 diffusiveFluxBoundaryConditionsSetterDictDict=None,
                 stressFluxBoundaryConditionsSetterDict=None,
                 stabilization=None,
                 shockCapturing=None,
                 conservativeFluxDict=None,
                 numericalFluxType=None,
                 TimeIntegrationClass=None,
                 massLumping=False,
                 reactionLumping=False,
                 options=None,
                 name='Plasticity',
                 reuse_trial_and_test_quadrature=True,
                 sd = True,
                 movingDomain=False,
                 bdyNullSpace=False):
        self.bdyNullSpace=bdyNullSpace
        #
        #set the objects describing the method and boundary conditions
        #
        self.velocityPostProcessor = None
        self.movingDomain=movingDomain
        self.tLast_mesh=None
        #
        #cek todo clean up these flags in the optimized version
        self.bcsTimeDependent=options.bcsTimeDependent
        self.bcsSet=False
        self.name=name
        self.sd=sd
        self.lowmem=True
        self.timeTerm=True#allow turning off  the  time derivative
        self.testIsTrial=True
        self.phiTrialIsTrial=True            
        self.u = uDict
        self.Hess=False
        if isinstance(self.u[0].femSpace,FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis):
            self.Hess=True
        self.ua = {}#analytical solutions
        self.phi  = phiDict
        self.dphi={}
        self.matType = matType
        #mwf try to reuse test and trial information across components if spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature#True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1,coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        ## Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh #assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList=None #explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict #no velocity post-processing for now
        self.fluxBoundaryConditions=fluxBoundaryConditionsDict
        self.stressFluxBoundaryConditionsSetterDict=stressFluxBoundaryConditionsSetterDict
        #determine whether  the stabilization term is nonlinear
        self.stabilizationIsNonlinear = False
        #cek come back
        if self.stabilization != None:
            for ci in range(self.nc):
                if ci in coefficients.mass:
                    for flag in list(coefficients.mass[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if  ci in coefficients.advection:
                    for  flag  in list(coefficients.advection[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if  ci in coefficients.diffusion:
                    for diffusionDict in list(coefficients.diffusion[ci].values()):
                        for  flag  in list(diffusionDict.values()):
                            if flag != 'constant':
                                self.stabilizationIsNonlinear=True
                if  ci in coefficients.potential:
                    for flag in list(coefficients.potential[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if ci in coefficients.reaction:
                    for flag in list(coefficients.reaction[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if ci in coefficients.hamiltonian:
                    for flag in list(coefficients.hamiltonian[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
        #determine if we need element boundary storage
        self.elementBoundaryIntegrals = {}
        for ci  in range(self.nc):
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux != None) or 
                                                 (numericalFluxType != None) or 
                                                 (self.fluxBoundaryConditions[ci] == 'outFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'mixedFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        #calculate some dimensions
        #
        self.nSpace_global    = self.u[0].femSpace.nSpace_global #assume same space dim for all variables
        self.nDOF_trial_element     = [u_j.femSpace.max_nDOF_element for  u_j in list(self.u.values())]
        self.nDOF_phi_trial_element     = [phi_k.femSpace.max_nDOF_element for  phi_k in list(self.phi.values())]
        self.n_phi_ip_element = [phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for  phi_k in list(self.phi.values())]
        self.nDOF_test_element     = [femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        self.nFreeDOF_global  = [dc.nFreeDOF_global for dc in list(self.dirichletConditions.values())]
        self.nVDOF_element    = sum(self.nDOF_trial_element)
        self.nFreeVDOF_global = sum(self.nFreeDOF_global) 
        #
        NonlinearEquation.__init__(self,self.nFreeVDOF_global)
        #
        #build the quadrature point dictionaries from the input (this
        #is just for convenience so that the input doesn't have to be
        #complete)
        #
        elementQuadratureDict={}
        elemQuadIsDict = isinstance(elementQuadrature,dict)
        if elemQuadIsDict: #set terms manually
            for I in self.coefficients.elementIntegralKeys:
                if I in elementQuadrature:
                    elementQuadratureDict[I] = elementQuadrature[I]
                else:
                    elementQuadratureDict[I] = elementQuadrature['default']
        else:
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[I] = elementQuadrature
        if self.stabilization != None:
            for I in self.coefficients.elementIntegralKeys:
                if elemQuadIsDict:
                    if I in elementQuadrature:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature
        if self.shockCapturing != None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff',ci,ci) in elementQuadrature:
                        elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature[('numDiff',ci,ci)]
                    else:
                        elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature
        if massLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('m',ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',)+I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
        if reactionLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('r',ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',)+I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
        elementBoundaryQuadratureDict={}
        if isinstance(elementBoundaryQuadrature,dict): #set terms manually
            for I in self.coefficients.elementBoundaryIntegralKeys:
                if I in elementBoundaryQuadrature:
                    elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature[I]
                else:
                    elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature['default']
        else:
            for I in self.coefficients.elementBoundaryIntegralKeys: 
                elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature
        #
        # find the union of all element quadrature points and
        # build a quadrature rule for each integral that has a
        # weight at each point in the union
        #mwf include tag telling me which indices are which quadrature rule?
        (self.elementQuadraturePoints,self.elementQuadratureWeights,
         self.elementQuadratureRuleIndeces) = Quadrature.buildUnion(elementQuadratureDict)
        self.nQuadraturePoints_element = self.elementQuadraturePoints.shape[0]
        self.nQuadraturePoints_global = self.nQuadraturePoints_element*self.mesh.nElements_global
        #
        #Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints,
         self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[0]
        self.nElementBoundaryQuadraturePoints_global = (self.mesh.nElements_global*
                                                        self.mesh.nElementBoundaries_element*
                                                        self.nElementBoundaryQuadraturePoints_elementBoundary)
        # if isinstance(self.u[0].femSpace,C0_AffineLinearOnSimplexWithNodalBasis):
        #     print self.nQuadraturePoints_element
        #     if self.nSpace_global == 3:
        #         assert(self.nQuadraturePoints_element == 5)
        #     elif self.nSpace_global == 2:
        #         assert(self.nQuadraturePoints_element == 6)
        #     elif self.nSpace_global == 1:
        #         assert(self.nQuadraturePoints_element == 3)

        #     print self.nElementBoundaryQuadraturePoints_elementBoundary
        #     if self.nSpace_global == 3:
        #         assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 4)
        #     elif self.nSpace_global == 2:
        #         assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 4)
        #     elif self.nSpace_global == 1:
        #         assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 1)
        #
        #simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q={}
        self.ebq={}
        self.ebq_global={}
        self.ebqe={}
        self.phi_ip={}
        #mesh
        #self.q['x'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,3),'d')
        #self.q['det(J)'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        #self.q['abs(det(J))'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        #self.q['J'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global,self.nSpace_global),'d')
        #self.q['inverse(J)'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global,self.nSpace_global),'d')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebqe['g'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
        #                              self.nElementBoundaryQuadraturePoints_elementBoundary,
        #max(1,self.nSpace_global-1),
        #max(1,self.nSpace_global-1)),
        #'d')
        #self.ebqe['inverse(J)'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global,self.nSpace_global),'d')
        #self.ebqe['hat(x)'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebqe['bar(x)'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebqe['sqrt(det(g))'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        #self.ebqe[('n')] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        #ebq for post-processing
        #self.ebq['x'] = np.zeros((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebq['g'] = np.zeros((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,
                                      #  self.nElementBoundaryQuadraturePoints_elementBoundary,
                                      #  max(1,self.nSpace_global-1),
                                      #  max(1,self.nSpace_global-1)),
                                      # 'd')
        #self.ebq['inverse(J)'] = np.zeros((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary,
                                              # self.nSpace_global,self.nSpace_global),'d')
        #self.ebq['hat(x)'] = np.zeros((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebq['bar(x)'] = np.zeros((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebq['sqrt(det(g))'] = np.zeros((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        #self.ebq[('n')] = np.zeros((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        #ebq_global for post-processing
        #self.ebq_global['x'] = np.zeros((self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebq_global['g'] = np.zeros((self.mesh.nElementBoundaries_global,
        #                                    self.nElementBoundaryQuadraturePoints_elementBoundary,
        #                                    max(1,self.nSpace_global-1),
        #                                    max(1,self.nSpace_global-1)),
        #                                   'd')
        #self.ebq_global['inverse(J)'] = np.zeros((self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,
        #                                             self.nSpace_global,self.nSpace_global),'d')
        #self.ebq_global['hat(x)'] = np.zeros((self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebq_global['bar(x)'] = np.zeros((self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        #self.ebq_global['sqrt(det(g))'] = np.zeros((self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        #self.ebq_global[('n')] = np.zeros((self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        #need to calculate
        #shape
        #self.q[('v',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[0]),'d')
        #self.q[('grad(v)',0)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[0],self.nSpace_global),'d')
        #self.q[('w*dV_r',0)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[1]),'d')
        #self.q[('grad(w)*dV_f',0)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[1],self.nSpace_global),'d')
        #self.ebqe[('v',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nDOF_trial_element[0]),'d')
        #self.ebqe[('grad(v)',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nDOF_trial_element[0],self.nSpace_global),'d')
        #self.ebqe[('w*dS_f',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nDOF_trial_element[0]),'d')
        #if self.nDOF_trial_element[1] != self.nDOF_trial_element[0]:
        #    self.q[('v',1)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[1]),'d')
        #    self.q[('grad(v)',1)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[1],self.nSpace_global),'d')
        #    self.q[('w*dV_r',1)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[1]),'d')
        #    self.q[('grad(w)*dV_f',1)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[1],self.nSpace_global),'d')
        #    self.ebqe[('v',1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nDOF_trial_element[1]),'d')
        #    self.ebqe[('grad(v)',1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nDOF_trial_element[1],self.nSpace_global),'d')
        #self.ebqe[('w*dS_f',1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nDOF_trial_element[1]),'d')
        #else:
        #    self.q[('v',1)] = self.q[('v',0)]
        #    self.q[('grad(v)',1)] = self.q[('grad(v)',0)]
        #    self.q[('w*dV_r',1)] =  self.q[('w*dV_r',0)]
        #    self.q[('grad(w)*dV_f',1)] = self.q[('grad(w)*dV_f',0)] 
        #    self.ebqe[('v',1)] = self.ebqe[('v',0)]
        #    self.ebqe[('grad(v)',1)] = self.ebqe[('grad(v)',0)]
        #    self.ebqe[('w*dS_f',1)] = self.ebqe[('w*dS_f',0)]
        #if self.Hess:
        #    self.q[('Hess(w)',1)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[1],self.nSpace_global,self.nSpace_global),'d')
        #    self.q[('Hess(w)*dV_a',1,1)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nDOF_trial_element[1],self.nSpace_global,self.nSpace_global),'d')
        #else:
        #    #dummy
        #    self.q[('Hess(w)',1)] =  np.zeros((1,),'d')
        #    self.q[('Hess(w)*dV_a',1,1)] =  self.q[('Hess(w)',1)]
        #self.q[('Hess(v)',1)] =  self.q[('Hess(w)',1)]
        #self.q[('Hess(v)',2)] =  self.q[('Hess(w)',1)]
        #self.q[('Hess(v)',3)] =  self.q[('Hess(w)',1)]
        #self.q[('Hess(w)',2)] =  self.q[('Hess(w)',1)]
        #self.q[('Hess(w)',3)] =  self.q[('Hess(w)',1)]
        #self.q[('Hess(w)*dV_a',2,2)] =  self.q[('Hess(w)*dV_a',1,1)]
        #self.q[('Hess(w)*dV_a',3,3)] =  self.q[('Hess(w)*dV_a',1,1)]
        #self.q[('v',2)] = self.q[('v',1)]
        #self.q[('grad(v)',2)] = self.q[('grad(v)',1)]
        #self.q[('w*dV_r',2)] =  self.q[('w*dV_r',1)]
        #self.q[('grad(w)*dV_f',2)] = self.q[('grad(w)*dV_f',1)] 
        #self.ebqe[('v',2)] = self.ebqe[('v',1)]
        #self.ebqe[('grad(v)',2)] = self.ebqe[('grad(v)',1)]
        #self.ebqe[('w*dS_f',2)] = self.ebqe[('w*dS_f',1)]
        #self.q[('v',3)] = self.q[('v',1)]
        #self.q[('grad(v)',3)] = self.q[('grad(v)',1)]
        #self.q[('w*dV_r',3)] =  self.q[('w*dV_r',1)]
        #self.q[('grad(w)*dV_f',3)] = self.q[('grad(w)*dV_f',1)] 
        #self.ebqe[('v',3)] = self.ebqe[('v',1)]
        #self.ebqe[('grad(v)',3)] = self.ebqe[('grad(v)',1)]
        #self.ebqe[('w*dS_f',3)] = self.ebqe[('w*dS_f',1)]
        #for ci in range(self.nc):
        #    self.q[('w*dV_m',ci)] = self.q[('w*dV_r',ci)]
        #    self.q[('w',ci)] = self.q[('v',ci)]
        #    self.q[('grad(w)',ci)] = self.q[('grad(v)',ci)]
        #    self.ebqe[('w',ci)] = self.ebqe[('v',ci)]
        #    self.ebqe[('grad(w)',ci)] = self.ebqe[('grad(v)',ci)]
        #self.q[('u',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q['bodyForce'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q['strain'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,6),'d')
        self.q[('u',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.ebqe[('u',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe['penalty'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('strain')] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,6),'d')
        self.ebqe[('u',1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('u',2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('stressFlux_bc_flag',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('stressFlux_bc_flag',1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('stressFlux_bc_flag',2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('stressFlux_bc',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('stressFlux_bc',1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('stressFlux_bc',2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.points_elementBoundaryQuadrature= set()
        self.scalars_elementBoundaryQuadrature= set([('u',ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature= set()
        self.tensors_elementBoundaryQuadrature= set()
        #
        #show quadrature
        #
        logEvent("Dumping quadrature shapes for model %s" % self.name,level=9)
        logEvent("Element quadrature array (q)", level=9)
        for (k,v) in list(self.q.items()): logEvent(str((k,v.shape)),level=9)
        logEvent("Element boundary quadrature (ebq)",level=9) 
        for (k,v) in list(self.ebq.items()): logEvent(str((k,v.shape)),level=9)
        logEvent("Global element boundary quadrature (ebq_global)",level=9)
        for (k,v) in list(self.ebq_global.items()): logEvent(str((k,v.shape)),level=9)
        logEvent("Exterior element boundary quadrature (ebqe)",level=9)
        for (k,v) in list(self.ebqe.items()): logEvent(str((k,v.shape)),level=9)
        logEvent("Interpolation points for nonlinear diffusion potential (phi_ip)",level=9)
        for (k,v) in list(self.phi_ip.items()): logEvent(str((k,v.shape)),level=9)
        #
        # allocate residual and Jacobian storage
        #
        self.elementResidual = [np.zeros(
            (self.mesh.nElements_global,
             self.nDOF_test_element[ci]),
            'd') for ci in range(self.nc)]
        self.elementSpatialResidual = [np.zeros(
            (self.mesh.nElements_global,
             self.nDOF_test_element[ci]),
            'd') for ci in range(self.nc)]
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,),'i')
            self.inflowBoundaryBC_values[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nDOF_trial_element[cj]),'d')
            self.inflowFlux[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.internalNodes = set(range(self.mesh.nNodes_global))
        #identify the internal nodes this is ought to be in mesh
        ##\todo move this to mesh
        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            for i in range(self.mesh.nNodes_element):
                if i != ebN_element:
                    I = self.mesh.elementNodesArray[eN_global,i]
                    self.internalNodes -= set([I])
        self.nNodes_internal = len(self.internalNodes)
        self.internalNodesArray=np.zeros((self.nNodes_internal,),cmeshTools.INDEX_DTYPE)
        for nI,n in enumerate(self.internalNodes):
            self.internalNodesArray[nI]=n
        #
        del self.internalNodes
        self.internalNodes = None
        logEvent("Updating local to global mappings",2)
        self.updateLocal2Global()
        logEvent("Building time integration object",2)
        logEvent(memory("inflowBC, internalNodes,updateLocal2Global","OneLevelTransport"),level=4)
        #mwf for interpolating subgrid error for gradients etc
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(self,integrateInterpolationPoints=True)
        else:
             self.timeIntegration = TimeIntegrationClass(self)
           
        if options != None:
            self.timeIntegration.setFromOptions(options)
        logEvent(memory("TimeIntegration","OneLevelTransport"),level=4)
        logEvent("Calculating numerical quadrature formulas",2)
        self.calculateQuadrature()
        #lay out components/equations contiguously for now
        self.offset = [0]
        for ci in range(1,self.nc):
            self.offset += [self.offset[ci-1]+self.nFreeDOF_global[ci-1]]
        self.stride = [1 for ci in range(self.nc)]
        #use contiguous layout of components for parallel, requires weak DBC's
        comm = Comm.get()
        self.comm=comm
        if comm.size() > 1:
            assert numericalFluxType != None and numericalFluxType.useWeakDirichletConditions,"You must use a numerical flux to apply weak boundary conditions for parallel runs"
            self.offset = [0]
            for ci in range(1,self.nc):
                self.offset += [ci]
            self.stride = [self.nc for ci in range(self.nc)]
        #
        logEvent(memory("stride+offset","OneLevelTransport"),level=4)
        if numericalFluxType != None:
            if options is None or options.periodicDirichletConditions is None:
                self.numericalFlux = numericalFluxType(self,
                                                       dofBoundaryConditionsSetterDict,
                                                       advectiveFluxBoundaryConditionsSetterDict,
                                                       diffusiveFluxBoundaryConditionsSetterDictDict)
            else:
                self.numericalFlux = numericalFluxType(self,
                                                       dofBoundaryConditionsSetterDict,
                                                       advectiveFluxBoundaryConditionsSetterDict,
                                                       diffusiveFluxBoundaryConditionsSetterDictDict,
                                                       options.periodicDirichletConditions)
        else:
            self.numericalFlux = None
        #set penalty terms
        #cek todo move into numerical flux initialization
        if 'penalty' in self.ebq_global:
            for ebN in range(self.mesh.nElementBoundaries_global):
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['penalty'][ebN,k] = self.numericalFlux.penalty_constant/(self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power)
        #penalty term
        #cek move  to Numerical flux initialization
        if 'penalty' in self.ebqe:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebqe['penalty'][ebNE,k] = self.numericalFlux.penalty_constant/self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power
        logEvent(memory("numericalFlux","OneLevelTransport"),level=4)
        self.elementEffectiveDiametersArray  = self.mesh.elementInnerDiametersArray
        #use post processing tools to get conservative fluxes, None by default
        #helper for writing out data storage
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        for ci,sbcObject  in list(self.stressFluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('stressFlux_bc_flag',ci)] = np.zeros(self.ebqe[('stressFlux_bc',ci)].shape,'i')
            for t,g in list(sbcObject.stressFluxBoundaryConditionsDict.items()):
                self.ebqe[('stressFlux_bc',ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                self.ebqe[('stressFlux_bc_flag',ci)][t[0],t[1]] = 1
        self.numericalFlux.setDirichletValues(self.ebqe)
        self.forceStrongConditions=False
        self.dirichletConditionsForceDOF = {}
        if self.forceStrongConditions:
            for cj in range(self.nc):
                self.dirichletConditionsForceDOF[cj] = DOFBoundaryConditions(self.u[cj].femSpace,dofBoundaryConditionsSetterDict[cj],weakDirichletConditions=False)
        compKernelFlag=0
        self.elastoPlastic = cElastoPlastic_base(self.nSpace_global,
                                                 self.nQuadraturePoints_element,
                                                 self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                                 self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                                 self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                 compKernelFlag)
    def getResidual(self,u,r):
        """
        Calculate the element residuals and add in to the global residual
        """
        #Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        #hack
        if self.bcsTimeDependent or not self.bcsSet:
            self.bcsSet=True
            #Dirichlet boundary conditions
            self.numericalFlux.setDirichletValues(self.ebqe)
            #Flux boundary conditions
            for ci,fbcObject  in list(self.stressFluxBoundaryConditionsObjectsDict.items()):
                for t,g in list(fbcObject.stressFluxBoundaryConditionsDict.items()):
                    self.ebqe[('stressFlux_bc',ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                    self.ebqe[('stressFlux_bc_flag',ci)][t[0],t[1]] = 1
        r.fill(0.0)
        self.elementResidual[0].fill(0.0)
        self.elementResidual[1].fill(0.0) 
        self.elementResidual[2].fill(0.0) 
        #import pdb
        #print self.mesh.elementMaterialTypes,
        #print self.coefficients.nMaterialProperties,
        #print self.coefficients.materialProperties,
        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN,g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["disp_trial_ref"] = self.u[0].femSpace.psi
        argsDict["disp_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["disp_test_ref"] = self.u[0].femSpace.psi
        argsDict["disp_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
        argsDict["disp_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["disp_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["disp_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["disp_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["ebqe_penalty"] = self.ebqe['penalty']
        argsDict["gravityStep"] = int(self.coefficients.gravityStep)
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["materialTypes"] = self.mesh.elementMaterialTypes
        argsDict["nMaterialProperties"] = self.coefficients.nMaterialProperties
        argsDict["materialProperties"] = self.coefficients.materialProperties
        argsDict["pore_fluid_unit_weight"] = self.coefficients.pore_fluid_unit_weight
        argsDict["pore_pressure_head_dof"] = self.coefficients.pore_pressure_head
        argsDict["q_strain"] = self.q['strain']
        argsDict["q_strain0"] = self.q['strain0']
        argsDict["q_strain_last"] = self.q['strain_last']
        argsDict["q_plasticStrain"] = self.q['plasticStrain']
        argsDict["q_plasticStrain_last"] = self.q['plasticStrain_last']
        argsDict["ebqe_strain"] = self.ebqe['strain']
        argsDict["ebqe_strain0"] = self.ebqe['strain0']
        argsDict["ebqe_strain_last"] = self.ebqe['strain_last']
        argsDict["ebqe_plasticStrain"] = self.ebqe['plasticStrain']
        argsDict["ebqe_plasticStrain_last"] = self.ebqe['plasticStrain_last']
        argsDict["disp_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.u[0].dof
        argsDict["v_dof"] = self.u[1].dof
        argsDict["w_dof"] = self.u[2].dof
        argsDict["bodyForce"] = self.coefficients.bodyForce
        argsDict["offset_u"] = self.offset[0]
        argsDict["offset_v"] = self.offset[1]
        argsDict["offset_w"] = self.offset[2]
        argsDict["stride_u"] = self.stride[0]
        argsDict["stride_v"] = self.stride[1]
        argsDict["stride_w"] = self.stride[2]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isStressFluxBoundary_u"] = self.ebqe[('stressFlux_bc_flag',0)]
        argsDict["isStressFluxBoundary_v"] = self.ebqe[('stressFlux_bc_flag',1)]
        argsDict["isStressFluxBoundary_w"] = self.ebqe[('stressFlux_bc_flag',2)]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        argsDict["ebqe_bc_v_ext"] = self.numericalFlux.ebqe[('u',1)]
        argsDict["ebqe_bc_w_ext"] = self.numericalFlux.ebqe[('u',2)]
        argsDict["ebqe_bc_stressFlux_u_ext"] = self.ebqe[('stressFlux_bc',0)]
        argsDict["ebqe_bc_stressFlux_v_ext"] = self.ebqe[('stressFlux_bc',1)]
        argsDict["ebqe_bc_stressFlux_w_ext"] = self.ebqe[('stressFlux_bc',2)]
        self.elastoPlastic.calculateResidual(argsDict)
        logEvent("Global residual",level=9,data=r)
        if self.forceStrongConditions:#
            for cj in range(len(self.dirichletConditionsForceDOF)):#
                for dofN,g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                     r[self.offset[cj]+self.stride[cj]*dofN] = 0
        self.nonlinear_function_evaluations += 1
    def getJacobian(self,jacobian,usePicard=False):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["usePicard"] = usePicard
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["disp_trial_ref"] = self.u[0].femSpace.psi
        argsDict["disp_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["disp_test_ref"] = self.u[0].femSpace.psi
        argsDict["disp_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
        argsDict["disp_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["disp_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["disp_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["disp_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["ebqe_penalty"] = self.ebqe['penalty']
        argsDict["gravityStep"] = int(self.coefficients.gravityStep)
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["materialTypes"] = self.mesh.elementMaterialTypes
        argsDict["nMaterialProperties"] = self.coefficients.nMaterialProperties
        argsDict["materialProperties"] = self.coefficients.materialProperties
        argsDict["pore_fluid_unit_weight"] = self.coefficients.pore_fluid_unit_weight
        argsDict["pore_pressure_head_dof"] = self.coefficients.pore_pressure_head
        argsDict["q_strain"] = self.q['strain']
        argsDict["q_strain0"] = self.q['strain0']
        argsDict["q_strain_last"] = self.q['strain_last']
        argsDict["q_plasticStrain"] = self.q['plasticStrain']
        argsDict["q_plasticStrain_last"] = self.q['plasticStrain_last']
        argsDict["ebqe_strain"] = self.ebqe['strain']
        argsDict["ebqe_strain0"] = self.ebqe['strain0']
        argsDict["ebqe_strain_last"] = self.ebqe['strain_last']
        argsDict["ebqe_plasticStrain"] = self.ebqe['plasticStrain']
        argsDict["ebqe_plasticStrain_last"] = self.ebqe['plasticStrain_last']
        argsDict["disp_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.u[0].dof
        argsDict["v_dof"] = self.u[1].dof
        argsDict["w_dof"] = self.u[2].dof
        argsDict["bodyForce"] = self.coefficients.bodyForce
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0,0)]
        argsDict["csrRowIndeces_u_v"] = self.csrRowIndeces[(0,1)]
        argsDict["csrColumnOffsets_u_v"] = self.csrColumnOffsets[(0,1)]
        argsDict["csrRowIndeces_u_w"] = self.csrRowIndeces[(0,2)]
        argsDict["csrColumnOffsets_u_w"] = self.csrColumnOffsets[(0,2)]
        argsDict["csrRowIndeces_v_u"] = self.csrRowIndeces[(1,0)]
        argsDict["csrColumnOffsets_v_u"] = self.csrColumnOffsets[(1,0)]
        argsDict["csrRowIndeces_v_v"] = self.csrRowIndeces[(1,1)]
        argsDict["csrColumnOffsets_v_v"] = self.csrColumnOffsets[(1,1)]
        argsDict["csrRowIndeces_v_w"] = self.csrRowIndeces[(1,2)]
        argsDict["csrColumnOffsets_v_w"] = self.csrColumnOffsets[(1,2)]
        argsDict["csrRowIndeces_w_u"] = self.csrRowIndeces[(2,0)]
        argsDict["csrColumnOffsets_w_u"] = self.csrColumnOffsets[(2,0)]
        argsDict["csrRowIndeces_w_v"] = self.csrRowIndeces[(2,1)]
        argsDict["csrColumnOffsets_w_v"] = self.csrColumnOffsets[(2,1)]
        argsDict["csrRowIndeces_w_w"] = self.csrRowIndeces[(2,2)]
        argsDict["csrColumnOffsets_w_w"] = self.csrColumnOffsets[(2,2)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isStressFluxBoundary_u"] = self.ebqe[('stressFlux_bc_flag',0)]
        argsDict["isStressFluxBoundary_v"] = self.ebqe[('stressFlux_bc_flag',1)]
        argsDict["isStressFluxBoundary_w"] = self.ebqe[('stressFlux_bc_flag',2)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0,0)]
        argsDict["csrColumnOffsets_eb_u_v"] = self.csrColumnOffsets_eb[(0,1)]
        argsDict["csrColumnOffsets_eb_u_w"] = self.csrColumnOffsets_eb[(0,2)]
        argsDict["csrColumnOffsets_eb_v_u"] = self.csrColumnOffsets_eb[(1,0)]
        argsDict["csrColumnOffsets_eb_v_v"] = self.csrColumnOffsets_eb[(1,1)]
        argsDict["csrColumnOffsets_eb_v_w"] = self.csrColumnOffsets_eb[(1,2)]
        argsDict["csrColumnOffsets_eb_w_u"] = self.csrColumnOffsets_eb[(2,0)]
        argsDict["csrColumnOffsets_eb_w_v"] = self.csrColumnOffsets_eb[(2,1)]
        argsDict["csrColumnOffsets_eb_w_w"] = self.csrColumnOffsets_eb[(2,2)]
        self.elastoPlastic.calculateJacobian(argsDict)
        logEvent("Jacobian ",level=10,data=jacobian)
        if self.forceStrongConditions:
            scaling = 1.0#probably want to add some scaling to match non-dirichlet diagonals in linear system 
            for cj in range(self.nc):
                for dofN in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.keys()):
                    global_dofN = self.offset[cj]+self.stride[cj]*dofN
                    for i in range(self.rowptr[global_dofN],self.rowptr[global_dofN+1]):
                        if (self.colind[i] == global_dofN):
                            self.nzval[i] = scaling
                        else:
                            self.nzval[i] = 0.0
        #mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        #jacobian.fwrite("jacobian_p"+`self.nonlinear_function_jacobian_evaluations`)
        return jacobian
    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.
        
        This function should be called only when the mesh changes.
        """
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        #
        #get physical locations of quadrature points and jacobian information there
        #assume all components live on the same mesh
        #
        # self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
        #                                           self.q['x'])
        # if self.movingDomain: 
        #     if self.tLast_mesh != None:
        #         self.q['xt'][:]=self.q['x']
        #         self.q['xt']-=self.q['x_last']
        #         alpha = 1.0/(self.t_mesh - self.tLast_mesh)
        #         self.q['xt']*=alpha
        #     else:
        #         self.q['xt'][:]=0.0
        #     self.q['x_last'][:]=self.q['x']
        # self.u[0].femSpace.elementMaps.getJacobianValues(self.elementQuadraturePoints,
        #                                                  self.q['J'],
        #                                                  self.q['inverse(J)'],
        #                                                  self.q['det(J)'])
        # self.q['abs(det(J))']=np.absolute(self.q['det(J)'])
        # #
        # # get physical space integration weights
        # #
        # self.q['dV'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        # cfemIntegrals.calculateIntegrationWeights(self.q['abs(det(J))'],
        #                                           self.elementQuadratureWeights[('u',0)],
        #                                           self.q['dV'])
        # for ci in range(self.nc): self.q[('dV_u',ci)] = self.q['dV']
        # #
        # #get shape information at the quadrature points
        # #
        # self.testSpace[0].getBasisValues(self.elementQuadraturePoints,
        #                                  self.q[('w',0)])
        # cfemIntegrals.calculateWeightedShape(self.elementQuadratureWeights[('r',1)],
        #                                      self.q['abs(det(J))'],
        #                                      self.q[('w',1)],
        #                                      self.q[('w*dV_r',1)])
        # self.testSpace[1].getBasisGradientValues(self.elementQuadraturePoints,
        #                                           self.q['inverse(J)'],
        #                                           self.q[('grad(w)',1)])
        # cfemIntegrals.calculateWeightedShapeGradients(self.elementQuadratureWeights[('f',0)],
        #                                               self.q['abs(det(J))'],
        #                                               self.q[('grad(w)',0)],
        #                                               self.q[('grad(w)*dV_f',0)])
        # if self.Hess:
        #     self.testSpace[1].getBasisHessianValues(self.elementQuadraturePoints,
        #                                             self.q['inverse(J)'],
        #                                             self.q[('Hess(w)',1)])
        #     cfemIntegrals.calculateWeightedShapeGradients(self.elementQuadratureWeights[('a',1,1)],
        #                                                   self.q['abs(det(J))'],
        #                                                   self.q[('Hess(w)',1)],
        #                                                   self.q[('Hess(w)*dV_a',1,1)])
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t,self.q)
    def calculateElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on element boundaries.

        This function should be called only when the mesh changes.
        """
        #cek todo, a local calculation of the element boundary quadrature permuations
        #
        #get physical locations of element boundary quadrature points
        #
        #assume all components live on the same mesh
        # self.u[0].femSpace.elementMaps.getValuesTrace(self.elementBoundaryQuadraturePoints,
        #                                             self.ebq['x'])
        #
        #get metric tensor and unit normals
        #
        # if self.movingDomain:
        #     if self.tLast_mesh != None:
        #         self.ebq['xt'][:]=self.ebq['x']
        #         self.ebq['xt']-=self.ebq['x_last']
        #         alpha = 1.0/(self.t_mesh - self.tLast_mesh)
        #         self.ebq['xt']*=alpha
        #     else:
        #         self.ebq['xt'][:]=0.0
        #     self.ebq['x_last'][:]=self.ebq['x']
        #     self.u[0].femSpace.elementMaps.getJacobianValuesTrace_movingDomain(self.elementBoundaryQuadraturePoints,
        #                                                                        self.ebq['xt'],
        #                                                                        self.ebq['inverse(J)'],
        #                                                                        self.ebq['g'],
        #                                                                        self.ebq['sqrt(det(g))'],
        #                                                                        self.ebq['n'])
        # else:
        #     self.u[0].femSpace.elementMaps.getJacobianValuesTrace(self.elementBoundaryQuadraturePoints,
        #                                                           self.ebq['inverse(J)'],
        #                                                           self.ebq['g'],
        #                                                           self.ebq['sqrt(det(g))'],
        #                                                           self.ebq['n'])

        # cfemIntegrals.copyLeftElementBoundaryInfo(self.mesh.elementBoundaryElementsArray,
        #                                           self.mesh.elementBoundaryLocalElementBoundariesArray,
        #                                           self.mesh.exteriorElementBoundariesArray,
        #                                           self.mesh.interiorElementBoundariesArray,
        #                                           self.ebq['x'],
        #                                           self.ebq['n'],
        #                                           self.ebq_global['x'],
        #                                           self.ebq_global['n'])
        # if self.movingDomain:
        #     cfemIntegrals.copyLeftElementBoundaryInfo_movingDomain(self.mesh.elementBoundaryElementsArray,
        #                                                            self.mesh.elementBoundaryLocalElementBoundariesArray,
        #                                                            self.mesh.exteriorElementBoundariesArray,
        #                                                            self.mesh.interiorElementBoundariesArray,
        #                                                            self.ebq['xt'])
        # #now map the physical points back to the reference element
        # #assume all components live  on same mesh
        # self.u[0].femSpace.elementMaps.getInverseValuesTrace(self.ebq['inverse(J)'],self.ebq['x'],self.ebq['hat(x)'])
        # self.u[0].femSpace.elementMaps.getPermutations(self.ebq['hat(x)'])
        #
        #since the points on the reference boundary may be reordered on many right element boundaries, we
        #have to use an array of reference boundary points on all element boundaries
        #first copy the left reference element boundary quadrature points from the reference element boundary
        #
        #get the shape information at the reference element boundary quadrature points 
        #
        # self.testSpace[0].getBasisValuesTrace(self.u[0].femSpace.elementMaps.permutations,
        #                                        self.ebq['hat(x)'],
        #                                        self.ebq[('w',0)])
        # cfemIntegrals.calculateWeightedShapeTrace(self.elementBoundaryQuadratureWeights[('u',0)],
        #                                           self.ebq['sqrt(det(g))'],
        #                                           self.ebq[('w',0)],
        #                                           self.ebq[('w*dS_u',0)])
        # self.u[0].femSpace.getBasisGradientValuesTrace(self.u[0].femSpace.elementMaps.permutations,
        #                                                 self.ebq['hat(x)'],
        #                                                 self.ebq['inverse(J)'],
        #                                                 self.ebq[('grad(v)',0)])
        # cfemIntegrals.calculateElementBoundaryIntegrationWeights(self.ebq['sqrt(det(g))'],
        #                                                         self.elementBoundaryQuadratureWeights[('u',0)],
        #                                                         self.ebq[('dS_u',0)])
    def calculateExteriorElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        #
        #get physical locations of element boundary quadrature points
        #
        #assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        # #
        # #get metric tensor and unit normals
        # #
        # if self.movingDomain:
        #     if self.tLast_mesh != None:
        #         self.ebqe['xt'][:]=self.ebqe['x']
        #         self.ebqe['xt']-=self.ebqe['x_last']
        #         alpha = 1.0/(self.t_mesh - self.tLast_mesh)
        #         self.ebqe['xt']*=alpha
        #     else:
        #         self.ebqe['xt'][:]=0.0
        #     self.ebqe['x_last'][:]=self.ebqe['x']
        #     self.u[0].femSpace.elementMaps.getJacobianValuesGlobalExteriorTrace_movingDomain(self.elementBoundaryQuadraturePoints,
        #                                                                                      self.ebqe['xt'],
        #                                                                                      self.ebqe['inverse(J)'],
        #                                                                                      self.ebqe['g'],
        #                                                                                      self.ebqe['sqrt(det(g))'],
        #                                                                                      self.ebqe['n'])
        # else:
        #     self.u[0].femSpace.elementMaps.getJacobianValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
        #                                                                         self.ebqe['inverse(J)'],
        #                                                                         self.ebqe['g'],
        #                                                                         self.ebqe['sqrt(det(g))'],
        #                                                                         self.ebqe['n'])
        # #now map the physical points back to the reference element
        # #assume all components live  on same mesh
        # self.u[0].femSpace.elementMaps.getInverseValuesGlobalExteriorTrace(self.ebqe['inverse(J)'],self.ebqe['x'],self.ebqe['hat(x)'])
        # #
        # #since the points on the reference boundary may be reordered on many right element boundaries, we
        # #have to use an array of reference boundary points on all element boundaries
        # #first copy the left reference element boundary quadrature points from the reference element boundary
        # self.testSpace[0].getBasisValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
        #                                                      self.ebqe[('w',0)])
        # cfemIntegrals.calculateWeightedShapeGlobalExteriorTrace(self.mesh.exteriorElementBoundariesArray,
        #                                                         self.mesh.elementBoundaryElementsArray,
        #                                                         self.mesh.elementBoundaryLocalElementBoundariesArray,
        #                                                         self.elementBoundaryQuadratureWeights[('f',0)],
        #                                                         self.ebqe['sqrt(det(g))'],
        #                                                         self.ebqe[('w',0)],
        #                                                         self.ebqe[('w*dS_f',0)])
        # self.u[0].femSpace.getBasisGradientValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
        #                                                               self.ebqe['inverse(J)'],
        #                                                               self.ebqe[('grad(v)',0)])
        # #setup flux boundary conditions
        self.stressFluxBoundaryConditionsObjectsDict = dict([(cj,FluxBoundaryConditions(self.mesh,
                                                                                        self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                        self.ebqe[('x')],
                                                                                        getStressFluxBoundaryConditions=self.stressFluxBoundaryConditionsSetterDict[cj]))
                                                             for cj in list(self.stressFluxBoundaryConditionsSetterDict.keys())])
        # self.ebqe['dS'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        # cfemIntegrals.calculateIntegrationWeights(self.ebqe['sqrt(det(g))'],
        #                                           self.elementBoundaryQuadratureWeights[('u',0)],
        #                                           self.ebqe['dS'])
        # for ci in range(self.nc): self.ebqe[('dS_u',ci)] = self.ebqe['dS']
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t,self.ebqe)
    def estimate_mt(self):
        pass
    def calculateSolutionAtQuadrature(self):
        pass
    def calculateAuxiliaryQuantitiesAfterStep(self):
        OneLevelTransport.calculateAuxiliaryQuantitiesAfterStep(self)
