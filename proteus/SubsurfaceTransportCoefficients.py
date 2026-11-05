"""
TransportCoefficients for flow and transport in porous media

.. inheritance-diagram:: proteus.SubsurfaceTransportCoefficients
   :parts: 1

"""
from math import *
from .TransportCoefficients import TC_base
import numpy
from .Profiling import logEvent
from proteus import FemTools
from proteus import Transport
from . import subsurfaceTransportFunctions as stfuncs

######################################################################
#Utility classes for dealing with common aspects of flow and transport
######################################################################

class BlockHeterogeneousCoefficients(object):
    """Basic data structures and functionality for keeping track of a
    block heterogeneity

    """
    def __init__(self,mesh):
        self.mesh = mesh
    def initializeMaterialTypes(self):
        """returns material type identifiers for mesh topology in tuple
        element,exterior_element_boundaries,element_boundaries

        note element_boundaries is nElementBoundaries_global x 2 and
        gives the element material property to the left and right of a
        global element boundary

        """
        elementMaterialTypes = self.mesh.elementMaterialTypes
        exteriorElementBoundaryTypes =  numpy.zeros((self.mesh.nExteriorElementBoundaries_global),'i')
        stfuncs.setExteriorElementBoundaryTypes(self.mesh.nExteriorElementBoundaries_global,
                                                self.mesh.exteriorElementBoundariesArray,
                                                self.mesh.elementBoundaryElementsArray,
                                                elementMaterialTypes,
                                                exteriorElementBoundaryTypes)
        elementBoundaryTypes = numpy.zeros((self.mesh.nElementBoundaries_global,2),'i')
        stfuncs.setElementBoundariesArray(self.mesh.nElementBoundaries_global,
                                          self.mesh.elementBoundaryElementsArray,
                                          self.mesh.elementMaterialTypes,
                                          elementBoundaryTypes)
        return elementMaterialTypes,exteriorElementBoundaryTypes,elementBoundaryTypes
##################################################
#Single-phase flow
##################################################
class SinglePhaseDarcyCoefficients(TC_base):
    """:math:`S_s h_t -\\deld ( K_i(x,t) \\grad h_i ) + r(x,t) = 0 i=1,nc`

    """
    def __init__(self,K_types,source_types,S_s_types=None,
                 nc=1,nd=2,
                 timeVaryingCoefficients=False,
                 materialValuesLocallyConstant=False):
        self.K_types = K_types
        self.source_types = source_types
        if S_s_types is None:
            self.S_s_types = {}
            for mat in list(self.K_types.keys()):
                self.S_s_types[mat] = lambda x,t: 0.0
        else:
            self.S_s_types = S_s_types
            assert timeVaryingCoefficients == True, "Transient Simulation (S_s > 0) requires timeVaryingCoefficients == True"
        self.nd = nd
        self.timeVaryingCoefficients=timeVaryingCoefficients
        self.materialValuesLocallyConstant = materialValuesLocallyConstant
        self.K_types_const=None
        self.source_types_const=None
        self.S_s_types_const=None
        if self.materialValuesLocallyConstant:
            self.K_types_const = {}; self.source_types_const= {}; self.S_s_types_const = {}
            x = numpy.zeros((3,),'d'); t0 = 0.0
            for mat in list(self.K_types.keys()):
                self.K_types_const[mat]     =self.K_types[mat](x,t0)
            for mat in list(self.source_types.keys()):
                self.source_types_const[mat]=self.source_types[mat](x,t0)
            for mat in list(self.S_s_types.keys()):
                self.S_s_types_const[mat]=self.S_s_types[mat](x,t0)
        #for matching shapes of quadrature arrays
        self.q_x_shape = None;  self.ebqe_x_shape=None
        self.ebq_global_x_shape=None;  self.ebq_shape=None
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction  = {}
        hamiltonian = {}
        for i in range(nc):
            diffusion[i] = {i : {i:'constant'}}
            reaction[i]  = {i : 'constant'}
            advection[i] = {i : 'constant'} #now include for gravity type terms
            potential[i] = {i : 'u'}
            mass[i] = {i : 'linear'}
        #end i
        sparseDiffusionTensors = {}
        for ci in range(nc):
            sparseDiffusionTensors[(ci,ci)]=(numpy.arange(start=0,stop=nd**2+1,step=nd,dtype='i'),
                                             numpy.array([list(range(nd)) for row in range(nd)],dtype='i'))
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)


    def initializeMesh(self,mesh):
        self.elementMaterialTypes,self.exteriorElementBoundaryTypes,self.elementBoundaryTypes = BlockHeterogeneousCoefficients(mesh).initializeMaterialTypes()
        self.elementBoundariesArray = mesh.elementBoundariesArray
    def initializeElementQuadrature(self,t,cq):
        self.evaluateHeterogeneity_element(t,cq)
        self.q_x_shape = cq['x'].shape
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.evaluateHeterogeneity_globalElementBoundary(t,cebq_global)
        self.ebq_global_x_shape = cebq_global['x'].shape
        self.evaluateHeterogeneity_elementBoundary(t,cebq)
        self.ebq_x_shape = cebq['x'].shape
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.evaluateHeterogeneity_exteriorElementBoundary(t,cebqe)
        self.ebqe_x_shape = cebqe['x'].shape
    def evaluateHeterogeneity_element(self,t,cq):
        if self.materialValuesLocallyConstant:
            x = numpy.zeros((3,),'d'); t0 = 0.0
            for mat in list(self.K_types.keys()):
                self.K_types_const[mat]     =self.K_types[mat](x,t0)
            for mat in list(self.source_types.keys()):
                self.source_types_const[mat]=self.source_types[mat](x,t0)
            for mat in list(self.S_s_types.keys()):
                self.S_s_types_const[mat]=self.S_s_types[mat](x,t0)

            for ci in range(self.nc):
                cq[('f',ci)].flat[:] = 0.0

                stfuncs.setScalarMaterialFunctionOverElements(self.elementMaterialTypes,
                                                              cq[('r',ci)],
                                                              self.source_types_const)
                cq[('r',ci)] *= -1.0
                #sparse matrix storage for tensors
                stfuncs.setVectorMaterialFunctionOverElements(self.elementMaterialTypes,
                                                              cq[('a',ci,ci)],
                                                              self.K_types_const)
                if ('dm',ci,ci) in cq:
                    stfuncs.setScalarMaterialFunctionOverElements(self.elementMaterialTypes,
                                                                  cq[('dm',ci,ci)],
                                                                  self.S_s_types_const)
                    cq[('m',ci)].flat[:] = cq[('u',ci)].flat
                    cq[('m',ci)] *= cq[('dm',ci,ci)]
        else:
            for ci in range(self.nc):
                cq[('f',ci)].flat[:] = 0.0

                stfuncs.evaluateScalarMaterialFunctionOverElements(t,self.elementMaterialTypes,
                                                                  cq['x'],
                                                                  cq[('r',ci)],
                                                                  self.source_types)
                cq[('r',ci)] *= -1.0
                #sparse matrix storage for tensors
                stfuncs.evaluateVectorMaterialFunctionOverElements(t,self.elementMaterialTypes,
                                                                   cq['x'],
                                                                   cq[('a',ci,ci)],
                                                                   self.K_types)

                if ('dm',ci,ci) in cq:
                    stfuncs.evaluateScalarMaterialFunctionOverElements(t,self.elementMaterialTypes,
                                                                       cq['x'],
                                                                       cq[('dm',ci,ci)],
                                                                       self.S_s_types)
                    cq[('m',ci)].flat[:] = cq[('u',ci)].flat
                    cq[('m',ci)] *= cq[('dm',ci,ci)]
    def evaluateHeterogeneity_elementBoundary(self,t,cebq):
        #use harmonic average for a, arith average for f
        if self.materialValuesLocallyConstant:
            x = numpy.zeros((3,),'d'); t0 = 0.0
            for mat in list(self.K_types.keys()):
                self.K_types_const[mat]     =self.K_types[mat](x,t0)
            for mat in list(self.source_types.keys()):
                self.source_types_const[mat]=self.source_types[mat](x,t0)
            for mat in list(self.S_s_types.keys()):
                self.S_s_types_const[mat]=self.S_s_types[mat](x,t0)
            for ci in range(self.nc):
                if ('f',ci) in cebq: cebq[('f',ci)].flat[:] = 0.0
                if ('r',ci) in cebq:
                    stfuncs.setScalarMaterialFunctionOverElementBoundaries_arithmeticAverage(self.elementBoundariesArray,
                                                                                             self.elementBoundaryTypes,
                                                                                             cebq[('r',ci)],
                                                                                             self.source_types_const)
                    cebq[('r',ci)] *= -1.0
                if ('a',ci,ci) in cebq:
                    stfuncs.setSparseTensorMaterialFunctionOverElementBoundaries_harmonicAverage(self.nd,
                                                                                                 self.elementBoundariesArray,
                                                                                                 self.elementBoundaryTypes,
                                                                                                 cebq[('a',ci,ci)],
                                                                                                 self.K_types_const)

                if ('dm',ci,ci) in cebq:
                    stfuncs.setScalarMaterialFunctionOverElementBoundaries_arithmeticAverage(self.elementBoundariesArray,
                                                                                             self.elementBoundaryTypes,
                                                                                             cebq[('dm',ci,ci)],
                                                                                             self.S_s_types_const)
                    cebq[('m',ci)].flat[:] = cebq[('u',ci)].flat
                    cebq[('m',ci)] *= cebq[('dm',ci,ci)]

        else:
            for ci in range(self.nc):
                if ('f',ci) in cebq: cebq[('f',ci)].fill(0.0)
                if ('r',ci) in cebq:
                    stfuncs.evaluateScalarMaterialFunctionOverElementBoundaries_arithmeticAverage(t,
                                                                                                  self.elementBoundariesArray,
                                                                                                  self.elementBoundaryTypes,
                                                                                                  cebq['x'],
                                                                                                  cebq[('r',ci)],
                                                                                                  self.source_types)
                    cebq[('r',ci)] *= -1.0
                if ('a',ci,ci) in cebq:
                    stfuncs.evaluateSparseTensorMaterialFunctionOverElementBoundaries_harmonicAverage(self.nd,
                                                                                                      t,
                                                                                                      self.elementBoundariesArray,
                                                                                                      self.elementBoundaryTypes,
                                                                                                      cebq['x'],
                                                                                                      cebq[('a',ci,ci)],
                                                                                                      self.K_types)
                if ('dm',ci,ci) in cebq:
                    stfuncs.evaluateScalarMaterialFunctionOverElementBoundaries_arithmeticAverage(t,
                                                                                                  self.elementBoundariesArray,
                                                                                                  self.elementBoundaryTypes,
                                                                                                  cebq['x'],
                                                                                                  cebq[('dm',ci,ci)],
                                                                                                  self.S_s_types)

                    cebq[('m',ci)].flat[:] = cebq[('u',ci)].flat
                    cebq[('m',ci)] *= cebq[('dm',ci,ci)]

    def evaluateHeterogeneity_globalElementBoundary(self,t,cebq_global):
        #use harmonic average for a, arith average for f
        if self.materialValuesLocallyConstant:
            x = numpy.zeros((3,),'d'); t0 = 0.0
            for mat in list(self.K_types.keys()):
                self.K_types_const[mat]     =self.K_types[mat](x,t0)
            for mat in list(self.source_types.keys()):
                self.source_types_const[mat]=self.source_types[mat](x,t0)
            for mat in list(self.S_s_types.keys()):
                self.S_s_types_const[mat]=self.S_s_types[mat](x,t0)
            for ci in range(self.nc):
                if ('f',ci) in cebq_global: cebq_global[('f',ci)].flat[:] = 0.0
                if ('r',ci) in cebq_global:
                    stfuncs.setScalarMaterialFunctionOverGlobalElementBoundaries_arithmeticAverage(self.elementBoundariesArray,
                                                                                                   self.elementBoundaryTypes,
                                                                                                   cebq_global[('r',ci)],
                                                                                                   self.source_types_const)
                    cebq_global[('r',ci)] *= -1.0
                if ('a',ci,ci) in cebq_global:
                    stfuncs.setSparseTensorMaterialFunctionOverGlobalElementBoundaries_harmonicAverage(self.nd,
                                                                                                       self.elementBoundariesArray,
                                                                                                       self.elementBoundaryTypes,
                                                                                                       cebq_global[('a',ci,ci)],
                                                                                                       self.K_types_const)

                if ('dm',ci,ci) in cebq_global:
                    stfuncs.setScalarMaterialFunctionOverGlobalElementBoundaries_arithmeticAverage(self.elementBoundariesArray,
                                                                                                   self.elementBoundaryTypes,
                                                                                                   cebq_global[('dm',ci,ci)],
                                                                                                   self.S_s_types_const)
                    cebq_global[('m',ci)].flat[:] = cebq_global[('u',ci)].flat
                    cebq_global[('m',ci)] *= cebq_global[('dm',ci,ci)]
        else:
            for ci in range(self.nc):
                if ('f',ci) in cebq_global: cebq_global[('f',ci)].flat[:] = 0.0
                if ('r',ci) in cebq_global:
                    stfuncs.evaluateScalarMaterialFunctionOverGlobalElementBoundaries_arithmeticAverage(t,
                                                                                                        self.elementBoundariesArray,
                                                                                                        self.elementBoundaryTypes,
                                                                                                        cebq_global['x'],
                                                                                                        cebq_global[('r',ci)],
                                                                                                        self.source_types)
                    cebq_global[('r',ci)] *= -1.0
                if ('a',ci,ci) in cebq_global:
                    stfuncs.evaluateSparseTensorMaterialFunctionOverGlobalElementBoundaries_harmonicAverage(self.nd,
                                                                                                            t,
                                                                                                            self.elementBoundariesArray,
                                                                                                            self.elementBoundaryTypes,
                                                                                                            cebq_global['x'],
                                                                                                            cebq_global[('a',ci,ci)],
                                                                                                            self.K_types)

                if ('dm',ci,ci) in cebq_global:
                    stfuncs.evaluateScalarMaterialFunctionOverGlobalElementBoundaries_arithmeticAverage(t,
                                                                                                        self.elementBoundariesArray,
                                                                                                        self.elementBoundaryTypes,
                                                                                                        cebq_global['x'],
                                                                                                        cebq_global[('dm',ci,ci)],
                                                                                                        self.S_s_types)
                    cebq_global[('m',ci)].flat[:] = cebq_global[('u',ci)].flat
                    cebq_global[('m',ci)] *= cebq_global[('dm',ci,ci)]
    def evaluateHeterogeneity_exteriorElementBoundary(self,t,cebqe):
        if self.materialValuesLocallyConstant:
            x = numpy.zeros((3,),'d'); t0 = 0.0
            for mat in list(self.K_types.keys()):
                self.K_types_const[mat]     =self.K_types[mat](x,t0)
            for mat in list(self.source_types.keys()):
                self.source_types_const[mat]=self.source_types[mat](x,t0)
            for mat in list(self.S_s_types.keys()):
                self.S_s_types_const[mat]=self.S_s_types[mat](x,t0)
            for ci in range(self.nc):
                cebqe[('f',ci)].flat[:] = 0.0

                stfuncs.setScalarMaterialFunctionOverElements(self.exteriorElementBoundaryTypes,
                                                              cebqe[('r',ci)],
                                                              self.source_types_const)
                cebqe[('r',ci)] *= -1.0
                #sparse matrix storage for tensors
                stfuncs.setVectorMaterialFunctionOverElements(self.exteriorElementBoundaryTypes,
                                                              cebqe[('a',ci,ci)],
                                                              self.K_types_const)

                if ('dm',ci,ci) in cebqe:
                    stfuncs.setScalarMaterialFunctionOverElements(self.exteriorElementBoundaryTypes,
                                                                  cebqe[('dm',ci,ci)],
                                                                  self.S_s_types_const)
                    cebqe[('m',ci)].flat[:] = cebqe[('u',ci)].flat
                    cebqe[('m',ci)] *= cebqe[('dm',ci,ci)]
        else:
            for ci in range(self.nc):
                cebqe[('f',ci)].flat[:] = 0.0

                stfuncs.evaluateScalarMaterialFunctionOverElements(t,self.exteriorElementBoundaryTypes,
                                                                  cebqe['x'],
                                                                  cebqe[('r',ci)],
                                                                  self.source_types)
                cebqe[('r',ci)] *= -1.0
                #sparse matrix storage for tensors
                stfuncs.evaluateVectorMaterialFunctionOverElements(t,self.exteriorElementBoundaryTypes,
                                                                   cebqe['x'],
                                                                   cebqe[('a',ci,ci)],
                                                                   self.K_types)

                if ('dm',ci,ci) in cebqe:
                    stfuncs.evaluateScalarMaterialFunctionOverElements(t,self.exteriorElementBoundaryTypes,
                                                                       cebqe['x'],
                                                                       cebqe[('dm',ci,ci)],
                                                                       self.S_s_types)
                    cebqe[('m',ci)].flat[:] = cebqe[('u',ci)].flat
                    cebqe[('m',ci)] *= cebqe[('dm',ci,ci)]

    def evaluate(self,t,c):
        if self.timeVaryingCoefficients == True:
            if c['x'].shape == self.q_x_shape:
                self.evaluateHeterogeneity_element(t,c)
            elif c['x'].shape == self.ebqe_x_shape:
                self.evaluateHeterogeneity_exteriorElementBoundary(t,c)
            elif c['x'].shape == self.ebq_global_x_shape:
                self.evaluateHeterogeneity_globalElementBoundary(t,c)
            elif c['x'].shape == self.ebq_x_shape:
                self.evaluateHeterogeneity_elementBoundary(t,c)
            else:
                raise NotImplementedError
            #mwf debug
            #print "eval c[('a',ci,ci)]= %s" % c[('a',0,0)]
    #end def
########################################
#Richards equation
########################################
class ConservativeHeadRichardsMualemVanGenuchten(TC_base):
    """version of Re where element material type id's used in evals

    """
    from .ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2
    from .ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchten_sd_het
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2 = staticmethod(conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2)
    conservativeHeadRichardsMualemVanGenuchten_sd_het = staticmethod(conservativeHeadRichardsMualemVanGenuchten_sd_het)
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 gravity,
                 density,
                 beta,
                 diagonal_conductivity=True,
                 getSeepageFace=None,
                 pc_eps=1.0e-8):
        variableNames=['pressure_head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        self.getSeepageFace=getSeepageFace
        self.gravity=gravity
        self.rho = density
        self.beta=beta
        self.vgm_n_types = vgm_n_types
        self.vgm_alpha_types = vgm_alpha_types
        self.thetaR_types    = thetaR_types
        self.thetaSR_types   = thetaSR_types
        self.elementMaterialTypes = None
        self.exteriorElementBoundaryTypes  = None
        self.materialTypes_q    = None
        self.materialTypes_ebq  = None
        self.materialTypes_ebqe  = None
        self.pc_eps = pc_eps
        self.nd = nd
        self.nMaterialTypes = len(thetaR_types)
        self.q = {}; self.ebqe = {}; self.ebq = {}; self.ebq_global={}
        #try to allow some flexibility in input of permeability/conductivity tensor
        self.diagonal_conductivity = diagonal_conductivity
        self.Ksw_types_in = Ksw_types
        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

            assert len(Ksw_types.shape) in [1,2], "if diagonal conductivity true then Ksw_types scalar or vector of diagonal entries"
            #allow scalar input Ks
            if len(Ksw_types.shape)==1:
                self.Ksw_types = numpy.zeros((self.nMaterialTypes,self.nd),'d')
                for I in range(self.nd):
                    self.Ksw_types[:,I] = Ksw_types
            else:
                self.Ksw_types = Ksw_types
        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}
            assert len(Ksw_types.shape) in [1,2], "if full tensor conductivity true then Ksw_types scalar or 'flattened' row-major representation of entries"
            if len(Ksw_types.shape)==1:
                self.Ksw_types = numpy.zeros((self.nMaterialTypes,self.nd**2),'d')
                for I in range(self.nd):
                    self.Ksw_types[:,I*self.nd+I] = Ksw_types
            else:
                assert Ksw_types.shape[1] == self.nd**2
                self.Ksw_types = Ksw_types
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)

    def initializeMesh(self,mesh):
        #import pdb
        #pdb.set_trace()
        self.elementMaterialTypes,self.exteriorElementBoundaryTypes,self.elementBoundaryTypes = BlockHeterogeneousCoefficients(mesh).initializeMaterialTypes()
        #want element boundary material types for evaluating heterogeneity
        #not boundary conditions
        self.isSeepageFace = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
        if self.getSeepageFace is not None:
            for ebNE in range(mesh.nExteriorElementBoundaries_global):
                #mwf missing ebNE-->ebN?
                ebN = mesh.exteriorElementBoundariesArray[ebNE]
                #print "eb flag",mesh.elementBoundaryMaterialTypes[ebN]
                #print self.getSeepageFace(mesh.elementBoundaryMaterialTypes[ebN])
                self.isSeepageFace[ebNE] = self.getSeepageFace(mesh.elementBoundaryMaterialTypes[ebN])
        #print self.isSeepageFace
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
#        cq['Ks'] = numpy.zeros(self.q_shape,'d')
#        for k in range(self.q_shape[1]):
#            cq['Ks'][:,k] = self.Ksw_types[self.elementMaterialTypes,0]
        self.q[('vol_frac',0)] = numpy.zeros(self.q_shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for ebN_local in range(self.ebq_shape[1]):
            self.materialTypes_ebq[:,ebN_local] = self.elementMaterialTypes
        self.ebq[('vol_frac',0)] = numpy.zeros(self.ebq_shape,'d')

    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = self.exteriorElementBoundaryTypes
        self.ebqe_shape = cebqe[('u',0)].shape
        self.ebqe[('vol_frac',0)] = numpy.zeros(self.ebqe_shape,'d')
        #
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
            vol_frac = self.q[('vol_frac',0)]
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
            vol_frac = self.ebqe[('vol_frac',0)]
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
            vol_frac = self.ebq[('vol_frac',0)]
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        linearize = 0
        self.conservativeHeadRichardsMualemVanGenuchten_sd_het(self.sdInfo[(0,0)][0],
                                                               self.sdInfo[(0,0)][1],
                                                               materialTypes,
                                                               self.rho,
                                                               self.beta,
                                                               self.gravity,
                                                               self.vgm_alpha_types,
                                                               self.vgm_n_types,
                                                               self.thetaR_types,
                                                               self.thetaSR_types,
                                                               self.Ksw_types,
                                                               c[('u',0)],
                                                               c[('m',0)],
                                                               c[('dm',0,0)],
                                                               c[('f',0)],
                                                               c[('df',0,0)],
                                                               c[('a',0,0)],
                                                               c[('da',0,0,0)],
                                                               vol_frac,
                                                               linearize,
                                                               self.pc_eps)

#         self.conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2(materialTypes,
#                                                                      self.rho,
#                                                                      self.beta,
#                                                                      self.gravity,
#                                                                      self.vgm_alpha_types,
#                                                                      self.vgm_n_types,
#                                                                      self.thetaR_types,
#                                                                      self.thetaSR_types,
#                                                                      self.Ksw_types,
#                                                                      c[('u',0)],
#                                                                      c[('m',0)],
#                                                                      c[('dm',0,0)],
#                                                                      c[('f',0)],
#                                                                      c[('df',0,0)],
#                                                                      c[('a',0,0)],
#                                                                      c[('da',0,0,0)])
        #mwf debug
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('df',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()

#         #mwf debug
#         if c[('u',0)].shape == self.q_shape:
#             c[('visPerm',0)]=c[('a',0,0)][:,:,0,0]

from proteus import cfemIntegrals
class RE_NCP1_OneLevelTransport(Transport.OneLevelTransport):
    """OneLevelTransport designed specifically for Non-Conforming
    :math:`P^1` approximation to RE Approximation uses nodal
    quadrature and upwinding

    """
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
                 name='defaultName',
                 reuse_trial_and_test_quadrature=False,
                 sd = True,
                 movingDomain=False):#,
        Transport.OneLevelTransport.__init__(self,
                                             uDict,
                                             phiDict,
                                             testSpaceDict,
                                             matType,
                                             dofBoundaryConditionsDict,
                                             dofBoundaryConditionsSetterDict,
                                             coefficients,
                                             elementQuadrature,
                                             elementBoundaryQuadrature,
                                             fluxBoundaryConditionsDict=fluxBoundaryConditionsDict,
                                             advectiveFluxBoundaryConditionsSetterDict=advectiveFluxBoundaryConditionsSetterDict,
                                             diffusiveFluxBoundaryConditionsSetterDictDict=diffusiveFluxBoundaryConditionsSetterDictDict,
                                             stressFluxBoundaryConditionsSetterDict=stressFluxBoundaryConditionsSetterDict,
                                             stabilization=stabilization,
                                             shockCapturing=shockCapturing,
                                             conservativeFluxDict=conservativeFluxDict,
                                             numericalFluxType=numericalFluxType,
                                             TimeIntegrationClass=TimeIntegrationClass,
                                             massLumping=massLumping,
                                             reactionLumping=reactionLumping,
                                             options=options,
                                             name=name,
                                             reuse_trial_and_test_quadrature=reuse_trial_and_test_quadrature,
                                             sd = sd,
                                             movingDomain=movingDomain)

        assert self.nQuadraturePoints_element == self.nSpace_global+1
        #particular arrays needed for Darcy flow
        self.q[('k_r',0,0)]    = numpy.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('dk_r',0,0,0)] = numpy.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('f_lin',0)]    = numpy.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('a_lin',0,0)]  = numpy.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.coefficients.sdInfo[(0,0)][0][self.nSpace_global]),'d')
        self.q[('k_r_up',0,0)]    = numpy.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')

        #need to check that points are the interpolation points too

#         #quadrature dictionary for solution interpolation points
#         self.n_u_ip_element = [u_j.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for  u_j in self.u.values()]
#         self.u_ip = {}
#         #coefficients
#         self.points_u_ip =StorageSet(shape=(self.mesh.nElements_global,self.n_u_ip_element[0],3))
#         self.scalars_u_ip=StorageSet(shape=(self.mesh.nElements_global,self.n_u_ip_element[0]))
#         self.vectors_u_ip=StorageSet(shape=(self.mesh.nElements_global,self.n_u_ip_element[0],self.nSpace_global))
#         self.tensors_u_ip=StorageSet(shape={})
#         #shape
#         self.trial_shape_u_ip=StorageSet(shape={})
#         self.test_shape_u_ip = StorageSet(shape={})
#         self.test_shapeGradient_u_ip = StorageSet(shape={})
#         self.trial_shapeGradient_u_ip= StorageSet(shape={})
#         trial_shape_u_ip_duplicate = StorageSet(shape={})
#         trial_shape_u_ip_duplicate_map = {}

#         #
#         # build sets
#         #
#         #u
#         self.scalars_u_ip |= set([('u',ci) for ci  in range(self.nc)])
#         self.vectors_u_ip |= set([('grad(u)',ci) for ci in range(self.nc)])
#         #trial
#         self.trial_shape_u_ip |= set([('v',ci) for ci in self.unique_test_trial_range])
#         trial_shape_u_ip_duplicate |= set([('v',ci) for ci in self.duplicate_test_trial_range])
#         for ci in self.duplicate_test_trial_range:
#             trial_shape_u_ip_duplicate_map[('v',ci)] = ('v',self.unique_test_trial_range[0])
#         #test
#       self.test_shape_u_ip |= set([('w',ci) for ci in range(self.nc)])
#         self.test_shapeGradient_u_ip |= set([('grad(w)',ci) for ci in range(self.nc)])
#         #m
#         self.scalars_u_ip |= set([('m',ci) for ci  in self.coefficients.mass.keys()])
#         self.scalars_u_ip |= set([('mt',ci) for ci  in self.coefficients.mass.keys()])
#         for ci,cjDict in self.coefficients.mass.iteritems():
#             self.scalars_u_ip |= set([('dm',ci,cj) for cj  in cjDict.keys()])
#         #f
#         self.vectors_u_ip |= set([('f',ci) for ci  in self.coefficients.advection.keys()])
#         #special for Darcy flow
#         self.vectors_u_ip |= set([('f_lin',ci) for ci  in self.coefficients.advection.keys()])
#         for ci,cjDict in self.coefficients.advection.iteritems():
#             self.vectors_u_ip |= set([('df',ci,cj) for cj  in cjDict.keys()])
#         #a and phi
#         self.vectors_u_ip |= set([('velocity',ci) for ci  in self.coefficients.diffusion.keys()])
#         self.scalars_u_ip |= set([('phi',ck) for ck  in self.coefficients.potential.keys()])
#         for ci,ckDict in self.coefficients.diffusion.iteritems():
#             self.tensors_u_ip |= set([('a',ci,ck) for ck  in ckDict.keys()])
#             #special for Darcy flow
#             self.tensors_u_ip |= set([('a_lin',ci,ck) for ck  in ckDict.keys()])
#             self.scalars_u_ip |= set([('k_r',ci,ck) for ck  in self.ckDict.keys()])

#             self.vectors_u_ip |= set([('grad(phi)',ck) for ck  in ckDict.keys()])
#             for ck,cjDict in ckDict.iteritems():
#                 self.tensors_u_ip |= set([('da',ci,ck,cj) for cj  in cjDict.keys()])
#                 #special for Darcy flow
#                 self.scalars_u_ip |= set([('dk_r',ci,ck,cj) for cj  in cjDict.keys()])
#                 #mwf trial-dup start
#               #mwf orig self.trial_shape_quadrature |= set([('v',cj) for cj in cjDict.keys()])
#                 unique_cj   = set.intersection(set(self.unique_test_trial_range),set(cjDict.keys()))
#                 duplicate_cj= set.intersection(set(self.duplicate_test_trial_range),set(cjDict.keys()))
#               self.trial_shape_u_ip |= set([('v',cj) for cj in unique_cj])
#               trial_shape_u_ip_duplicate |= set([('v',cj) for cj in duplicate_cj])
#                 #any reason to do this again
#                 for cj in self.duplicate_test_trial_range:
#                     trial_shape_u_ip_duplicate_map[('v',cj)] = ('v',self.unique_test_trial_range[0])
#                 #mwf trial-dup end
#                 for cj in self.coefficients.potential[ck].keys():
#                     self.scalars_u_ip |= set([('dphi',ck,cj)])
#         #mwf what if miss nonlinearities in diffusion that don't match potentials?
#         for comb in self.dphi.keys():
#             self.scalars_u_ip |= set([('dphi',comb[0],comb[1])])
#         #r
#         self.scalars_u_ip |= set([('r',ci) for ci  in self.coefficients.reaction.keys()])
#         for ci,cjDict in self.coefficients.reaction.iteritems():
#             self.scalars_u_ip |= set([('dr',ci,cj) for cj  in cjDict.keys()])
#         #mesh
#         self.points_u_ip |= set([('x',ci) for ci in self.unique_test_trial_range])
#         #
#         memory()
#         #
#         for ci in self.unique_test_trial_range:
#             self.u_ip[('x',ci)] = self.u[ci].femSpace.updateInterpolationPoints()
#         for ci in self.duplicate_test_trial_range:
#             self.u_ip[('x',ci)] = self.u_ip[('x',self.unique_test_trial_range[0])]
#         #
#         #allocations
#         #
#         self.scalars_u_ip.allocate(self.u_ip)
#         #points
#         self.points_u_ip.allocate(self.u_ip)
#         #scalars
#         self.scalars_u_ip.allocate(self.u_ip)
#         #vectors
#         self.vectors_u_ip.allocate(self.u_ip)
#         #allocate tensors element quadrature
#         for k in self.tensors_u_ip:
#             if (self.sd
#                 and k[0] in ['a','da']
#                 and self.coefficients.sdInfo is not None
#                 and (k[1],k[2]) in self.coefficients.sdInfo.keys()):
#                 self.u_ip[k]=numpy.zeros(
#                     (self.mesh.nElements_global,
#                      self.n_u_ip_element[k[2]],
#                      self.coefficients.sdInfo[(k[1],k[2])][0][self.nSpace_global]),
#                     'd')
#             else:
#                 self.u_ip[k]=numpy.zeros(
#                     (self.mesh.nElements_global,
#                      self..n_u_ip_element[k[2]],
#                      self.nSpace_global,
#                      self.nSpace_global),
#                     'd')
#         #
#         # shape
#         #
#         def makeAlias(sd,kw,vstring='v',wstring='w'):
#             aliased=False
#             kv0 = kw[0].replace(wstring,vstring)
#             if len(kw) > 1:
#                 kv = (kv0,)+kw[1:]
#             else:
#                 kv = kv0
#             if self.testIsTrial:
#                 if kv in sd.keys():
#                     logEvent("Shallow copy of trial shape is being used for test shape %s " % kw[0],level=4)
#                     sd[kw] = sd[kv]
#                     aliased=True
#             return aliased
#         def makeAliasForComponent1(sd,k,entryStrings,refi=0):
#             aliased=False
#             #need to add combinatorial options
#             ks = k[0]
#             if ks not in entryStrings:
#                 return False
#             k0 = (ks,)+(refi,)
#             if self.reuse_test_trial_quadrature and refi is not None:
#                 if k0 in sd.keys():
#                     logEvent("Shallow copy of trial shape %s is being used for trial shape %s" % (k0,k),level=4)
#                     sd[k] = sd[k0]
#                     aliased=True
#             return aliased
#         #
#         # element quadrature
#         #
#         #allocate trial shape functions
#         for k in sorted(self.trial_shape_u_ip):
#             if not makeAliasForComponent1(self.u_ip,k,['v'],refi=0):#need to handle multiple component combinations
#                 self.u_ip[k]=numpy.zeros(
#                     (self.mesh.nElements_global,
#                      self.n_u_ip_element,
#                      self.nDOF_trial_element[k[-1]]),
#                     'd')
#         for k in trial_shape_quadrature_duplicate:
#             self.u_ip[k] = self.u_ip[trial_shape_quadrature_duplicate_map[k]]
#         #mwf trial-dup end
#         logEvent(memory("solution interpolation points, test/trial functions trial_shape","OneLevelTransport"),level=4)
#         #allocate test shape functions
#         for k in self.test_shape_quadrature:
#             if not makeAlias(self.u_ip,k):
#                 self.u_ip[k]=numpy.zeros(
#                     (self.mesh.nElements_global,
#                      self.n_u_ip_element,
#                      self.nDOF_test_element[k[-1]]),
#                     'd')
#         logEvent(memory("solution interpolation points, test/trial functions test_shape","OneLevelTransport"),level=4)
#         #allocate trial shape function gradients
#         for k in sorted(self.trial_shapeGradient_quadrature):
#             if not makeAliasForComponent1(self.u_ip,k,['grad(v)'],refi=0):#need to handle multiple component combinations
#                 self.u_ip[k]=numpy.zeros(
#                     (self.mesh.nElements_global,
#                      self.n_u_ip_element,
#                      self.nDOF_trial_element[k[-1]],
#                      self.nSpace_global),
#                     'd')
#         logEvent(memory("solution interpolation points, test/trial functions trial_shapeGradient","OneLevelTransport"),level=4)
#         #allocate test shape function gradients
#         for k in self.test_shapeGradient_quadrature:
#             if not makeAlias(self.u_ip,k):
#                 self.u_ip[k]=numpy.zeros(
#                     (self.mesh.nElements_global,
#                      self.n_u_ip_element,
#                      self.nDOF_test_element[k[-1]],
#                      self.nSpace_global),
#                     'd')
    def calculateElementCoefficients(self):
        """calculate the nonlinear coefficients at the quadrature points and
        nodes include interpolation points explicitly here now

        """

        for cj in range(self.nc):
            self.u[cj].getValues(self.q[('v',cj)],
                                 self.q[('u',cj)])
            if ('grad(u)',cj) in self.q:
                self.u[cj].getGradientValues(self.q[('grad(v)',cj)],
                                             self.q[('grad(u)',cj)])

        #can skip this after first call
        stfuncs.RE_NCP1_evaluateElementCoefficients_Linear(self.coefficients.rho,
                                                           self.coefficients.gravity,
                                                           self.coefficients.sdInfo[(0,0)][0],
                                                           self.coefficients.sdInfo[(0,0)][1],
                                                           self.coefficients.Ksw_types,
                                                           self.nSpace_global,
                                                           self.mesh.nElements_global,
                                                           self.mesh.nElementBoundaries_element,
                                                           self.mesh.elementNeighborsArray,
                                                           self.mesh.elementMaterialTypes,
                                                           self.q[('f_lin',0)],
                                                           self.q[('a_lin',0,0)])

        stfuncs.RE_NCP1_evaluateElementCoefficients_VGM(self.coefficients.rho,
                                                        self.coefficients.beta,
                                                        self.coefficients.gravity,
                                                        self.coefficients.vgm_alpha_types,
                                                        self.coefficients.vgm_n_types,
                                                        self.coefficients.thetaR_types,
                                                        self.coefficients.thetaSR_types,
                                                        self.nSpace_global,
                                                        self.mesh.nElements_global,
                                                        self.mesh.nElementBoundaries_element,
                                                        self.mesh.elementNeighborsArray,
                                                        self.mesh.elementBarycentersArray,
                                                        self.mesh.elementMaterialTypes,
                                                        self.nDOF_trial_element[0],
                                                        self.u[0].femSpace.dofMap.l2g,
                                                        self.u[0].dof,
                                                        self.q['x'],
                                                        self.q[('u',0)],
                                                        self.q[('m',0)],
                                                        self.q[('dm',0,0)],
                                                        self.q[('r',0)],
                                                        self.q[('k_r',0,0)],
                                                        self.q[('dk_r',0,0,0)],
                                                        self.q[('k_r_up',0,0)])

        if self.movingDomain and self.coefficients.movingDomain:
            self.coefficients.updateToMovingDomain(self.timeIntegration.t,self.q)
        if self.timeTerm:
            self.timeIntegration.calculateElementCoefficients(self.q)
        #cek need to clean up calculation of dimless numbers, might as well do it all the time and pass to subgrid error
        #mwf figure out what to do with this
        #what happens if stabilization didn't compute cfl?
        for ci in range(self.nc):
            #for two phase flow would need this
            self.q[('dphi',ci,ci)].fill(1.0)
            if self.sd:
                cfemIntegrals.calculateDimensionlessNumbersADR_sd(self.mesh.nElements_global,
                                                                  self.nQuadraturePoints_element,
                                                                  self.nSpace_global,
                                                                  self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                  self.elementEffectiveDiametersArray,
                                                                  self.q[('df',ci,ci)],
                                                                  self.q[('a',ci,ci)],
                                                                  self.q[('dphi',ci,ci)],
                                                                  self.q[('dr',ci,ci)],
                                                                  self.q[('dmt',ci,ci)],
                                                                  self.q[('pe',ci)],
                                                                  self.q[('cfl',ci)])
            else:
                cfemIntegrals.calculateDimensionlessNumbersADR(self.mesh.nElements_global,
                                                               self.nQuadraturePoints_element,
                                                               self.nSpace_global,
                                                               self.elementEffectiveDiametersArray,
                                                               self.q[('df',ci,ci)],
                                                               self.q[('a',ci,ci)],
                                                               self.q[('dphi',ci,ci)],
                                                               self.q[('dr',ci,ci)],
                                                               self.q[('dmt',ci,ci)],
                                                               self.q[('pe',ci)],
                                                               self.q[('cfl',ci)])

        if self.shockCapturing is not None:
            self.shockCapturing.calculateNumericalDiffusion(self.q)

    def calculateElementResidual(self):
        """Calculate all the element residuals"""
        import pdb
        #mwf debug
        #Transport.OneLevelTransport.calculateElementResidual(self)
        #self.elementResidual_save = numpy.copy(self.elementResidual[0])
        for ci in range(self.nc):
            self.elementResidual[ci].fill(0.0)
        #mwf debug
        #pdb.set_trace()
        stfuncs.RE_NCP1_getElementResidual(self.coefficients.gravity,
                                           self.coefficients.sdInfo[(0,0)][0],
                                           self.coefficients.sdInfo[(0,0)][1],
                                           self.nSpace_global,
                                           self.mesh.nElements_global,
                                           self.mesh.nElementBoundaries_element,
                                           self.mesh.elementNeighborsArray,
                                           self.mesh.elementBarycentersArray,
                                           self.nDOF_test_element[0],
                                           self.q[('u',0)],
                                           self.q[('grad(u)',0)],
                                           self.q[('grad(w)',0)],
                                           self.q['abs(det(J))'],
                                           self.q[('m',0)],
                                           self.q[('mt',0)],
                                           self.q[('r',0)],
                                           self.q[('k_r',0,0)],
                                           self.q[('k_r_up',0,0)],
                                           self.q[('f_lin',0)],
                                           self.q[('a_lin',0,0)],
                                           self.elementResidual[0])
        #mwf debug
        #pdb.set_trace()


    def calculateElementJacobian(self):
        import pdb
        #Transport.OneLevelTransport.calculateElementJacobian(self)
        #self.elementJacobian_save = numpy.copy(self.elementJacobian[0][0])
        for ci in range(self.nc):
            for cj in self.coefficients.stencil[ci]:
                self.elementJacobian[ci][cj].fill(0.0)

        #mwf debug
        #pdb.set_trace()
        stfuncs.RE_NCP1_getElementJacobian(self.coefficients.gravity,
                                           self.coefficients.sdInfo[(0,0)][0],
                                           self.coefficients.sdInfo[(0,0)][1],
                                           self.nSpace_global,
                                           self.mesh.nElements_global,
                                           self.mesh.nElementBoundaries_element,
                                           self.mesh.elementNeighborsArray,
                                           self.mesh.elementBarycentersArray,
                                           self.nDOF_test_element[0],
                                           self.nDOF_trial_element[0],
                                           self.q[('u',0)],
                                           self.q[('grad(u)',0)],
                                           self.q[('grad(w)',0)],
                                           self.q[('grad(v)',0)],
                                           self.q['abs(det(J))'],
                                           self.q[('m',0)],
                                           self.q[('dm',0,0)],
                                           self.q[('mt',0)],
                                           self.q[('dmt',0,0)],
                                           self.q[('r',0)],
                                           self.q[('k_r',0,0)],
                                           self.q[('dk_r',0,0,0)],
                                           self.q[('k_r_up',0,0)],
                                           self.q[('f_lin',0)],
                                           self.q[('a_lin',0,0)],
                                           self.elementJacobian[0][0])

########################################
#Immiscible two-phase flow
########################################
class TwophaseDarcyFlow_base(TC_base):
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_vol_frac,generateSplineTables
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_vol_frac = staticmethod(twophaseDarcy_vol_frac)
    generateSplineTables = staticmethod(generateSplineTables)
    """base class for two-phase flow implementations 

    holds information for:

      * fluid properties
      * eos model tag
      * psk model tag
      * material heterogeneity information (number of types, lookup arrays)

    default fluids are water and air assumed to be incompressible

    """
    default_density_w_parameters = {'model':'Exponential',
                                    'nParameters':3,
                                    'rho_0':998.2,#Water kg/m^3
                                    'psi_0':0.0,
                                    'beta':0.0}
    default_density_n_parameters = {'model':'Exponential',
                                    'nParameters':3,
                                    'rho_0':1.205,#Air   kg/m^3
                                    'psi_0':0.0,
                                    'beta':0.0}
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=default_density_w_parameters,
                 density_n_parameters=default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 diagonal_conductivity=True,
                 nPSKsplineKnots=None): #otherwise Full
        self.nd = nd
        self.g  = dimensionless_gravity
        ##fluid properties
        #normalized density terms
        self.rhon = density_n/density_w
        self.rhow = density_w/density_w
        self.b  = density_n/density_w
        #normalized viscosities
        self.muw= viscosity_w/viscosity_w
        self.mun= viscosity_n/viscosity_w
        #density eos options
        self.density_types={'Exponential':1,
                            'IdealGas':2}
        self.density_w_model = density_w_parameters['model']
        self.density_n_model = density_n_parameters['model']
        assert self.density_w_model in self.density_types
        assert self.density_n_model in self.density_types
        self.density_w_parameters = density_w_parameters
        self.density_n_parameters = density_n_parameters
        for params,rwork in zip([self.density_w_parameters,self.density_n_parameters],
                                ['rwork_density_w','rwork_density_n']):
            if params is not None:
                if params['model'] == 'Exponential':
                    setattr(self,rwork,numpy.array([params['rho_0']/params['rho_0'],#normalize by phase density
                                                    params['psi_0'],
                                                    params['beta']],dtype='d'))
                elif params['model'] == 'IdealGas':
                    setattr(self,rwork,numpy.array([params['T'],
                                                    params['W']/params['rho_0'],#normalize by phase density
                                                    params['R'],
                                                    params['headToPressure'],
                                                    params['rho_0']/params['rho_0'],#normalize by phase density
                                                    params['psi_0']],dtype='d'))
                else:
                    assert False, 'TwophaseDarcy_base density params= %s not found ' % params
        #

        ##psk relations and heterogeneity
        self.psk_model=psk_model
        self.psk_types={'simp':0,
                        'VGM':1,
                        'VGB':2,
                        'BCM':3,
                        'BCB':4,
                        'PSKspline':5}
        self.psk_tolerances={'default':{'eps_small':1.0e-16},
                             'VGM':{'eps_small':1.0e-16,'ns_del':1.0e-8}}
        for psk_model_id in self.psk_types:
            if psk_model_id not in self.psk_tolerances:
                self.psk_tolerances[psk_model_id] = self.psk_tolerances['default']
        self.nPskTolerances=1
        assert(self.psk_model in list(self.psk_types.keys()))
        self.nMaterialTypes = nMaterialTypes
        #psk rwork array lengths
        self.nPSKsplineKnots = nPSKsplineKnots
        self.iwork_psk = numpy.zeros((2,),'i')
        if self.psk_model == 'simp':
            self.nPskParams=2
        elif self.psk_model in ['VGM','VGB']:
            self.nPskParams=4
            if self.psk_model == 'VGM':
                self.nPskTolerances=2
        elif self.psk_model in ['BCM','BCB']:
            self.nPskParams=4
        elif self.psk_model in ['PSKspline']:
            assert self.nPSKsplineKnots is not None
            self.nPskParams=self.nPSKsplineKnots*4
            self.iwork_psk[0] = self.nPSKsplineKnots
            self.iwork_psk[1] = self.nPskParams

        self.rwork_psk = None
        self.rwork_psk_tolerances = numpy.zeros((self.nPskTolerances,),'d')
        for i,tol in enumerate(self.psk_tolerances[self.psk_model].values()):
            self.rwork_psk_tolerances[i] = tol

        self.diagonal_conductivity=diagonal_conductivity
        self.q = {}; self.ebqe = {}; self.ebq = {}; self.ebq_global={}; self.ip = {}
    def initializeMesh(self,mesh):
        self.elementMaterialTypes,self.exteriorElementBoundaryTypes,self.elementBoundaryTypes = BlockHeterogeneousCoefficients(mesh).initializeMaterialTypes()
        self.mesh = mesh #for debugging
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
        #
        cq['psi_n'] = numpy.zeros(cq[('u',0)].shape,'d')
        cq[('dpsi_n',0)] = numpy.zeros(cq[('u',0)].shape,'d')
        cq[('dpsi_n',1)] = numpy.zeros(cq[('u',0)].shape,'d')
        #for visualization
        cq['Ks'] = numpy.zeros(self.q_shape,'d')
        for k in range(self.q_shape[1]):
            cq['Ks'][:,k] = self.Ksw_types[self.elementMaterialTypes,0]
        for ci in range(2):
            self.q[('vol_frac',ci)] = numpy.zeros(self.q_shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for ebN_local in range(self.ebq_shape[1]):
            self.materialTypes_ebq[:,ebN_local] = self.elementMaterialTypes
        cebq['psi_n'] = numpy.zeros(cebq[('u',0)].shape,'d')
        cebq[('dpsi_n',0)] = numpy.zeros(cebq[('u',0)].shape,'d')
        cebq[('dpsi_n',1)] = numpy.zeros(cebq[('u',0)].shape,'d')
        for ci in range(2):
            self.ebq[('vol_frac',ci)] = numpy.zeros(self.ebq_shape,'d')
        if ('u',0) in cebq_global:
            cebq_global['psi_n'] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            cebq_global[('dpsi_n',0)] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            cebq_global[('dpsi_n',1)] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            for ci in range(2):
                self.ebq_global[('vol_frac',ci)] = numpy.zeros(cebq_global[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = self.exteriorElementBoundaryTypes
        self.ebqe_shape = cebqe[('u',0)].shape
        #
        cebqe['psi_n'] = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe[('dpsi_n',0)] = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe[('dpsi_n',1)] = numpy.zeros(cebqe[('u',0)].shape,'d')
        for ci in range(2):
            self.ebqe[('vol_frac',ci)] = numpy.zeros(self.ebqe_shape,'d')
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        self.materialTypes_ip = self.elementMaterialTypes
        self.ip_shape = cip[('u',0)].shape
        #
        cip['psi_n'] = numpy.zeros(cip[('u',0)].shape,'d')
        cip[('dpsi_n',0)] = numpy.zeros(cip[('u',0)].shape,'d')
        cip[('dpsi_n',1)] = numpy.zeros(cip[('u',0)].shape,'d')
        for ci in range(2):
            self.ip[('vol_frac',ci)] = numpy.zeros(self.ip_shape,'d')
    def setMaterialTypes(self,
                         Ksw_types=[1.0],
                         omega_types  = [0.4],
                         Sw_max_types = [1.0],
                         Sw_min_types = [0.0],
                         bc_lambda_types = None,
                         bc_pd_types = None,
                         vg_alpha_types = None,
                         vg_m_types = None,
                         psk_spline_types = None):
        self.nMaterialTypes=len(omega_types)
        self.omega_types= omega_types
        if self.psk_model == 'simp':
            assert self.nPskParams == 2
            self.rwork_psk = numpy.zeros((self.nMaterialTypes,self.nPskParams),'d')
            for Sw_min,Sw_max,i in zip(Sw_min_types,Sw_max_types,list(range(self.nMaterialTypes))):
                self.rwork_psk[i,0] = Sw_min
                self.rwork_psk[i,1] = Sw_max
        elif self.psk_model in ['VGM','VGB']:
            assert(vg_alpha_types is not None and vg_m_types  is not None)
            assert self.nPskParams == 4
            self.rwork_psk = numpy.zeros((self.nMaterialTypes,self.nPskParams),'d')
            for Sw_min,Sw_max,vg_alpha,vg_m,i in zip(Sw_min_types,Sw_max_types,vg_alpha_types,vg_m_types,list(range(self.nMaterialTypes))):
                self.rwork_psk[i,0] = Sw_min
                self.rwork_psk[i,1] = Sw_max
                self.rwork_psk[i,2] = vg_alpha
                self.rwork_psk[i,3] = vg_m
        elif self.psk_model in ['BCM','BCB']:
            assert(bc_lambda_types is not None and bc_lambda_types  is not None)
            assert self.nPskParams == 4
            self.rwork_psk = numpy.zeros((self.nMaterialTypes,self.nPskParams),'d')
            for Sw_min,Sw_max,bc_pd,bc_lambda,i in zip(Sw_min_types,Sw_max_types,bc_pd_types,bc_lambda_types,list(range(self.nMaterialTypes))):
                self.rwork_psk[i,0] = Sw_min
                self.rwork_psk[i,1] = Sw_max
                self.rwork_psk[i,2] = bc_pd
                self.rwork_psk[i,3] = bc_lambda
        elif self.psk_model in ['PSKspline']:
            assert psk_spline_types is not None
            self.rwork_psk = psk_spline_types
            assert self.rwork_psk.shape[0] == self.nMaterialTypes*self.nPskParams

        #try to allow some flexibility in input of permeability/conductivity tensor
        #keep track of input for debugging now
        self.Ksw_types_in = Ksw_types
        if self.diagonal_conductivity:
            assert len(Ksw_types.shape) in [1,2], "if diagonal conductivity true then Ksw_types scalar or vector of diagonal entries"
            #allow scalar input Ks
            if len(Ksw_types.shape)==1:
                self.Ksw_types = numpy.zeros((self.nMaterialTypes,self.nd),'d')
                for I in range(self.nd):
                    self.Ksw_types[:,I] = Ksw_types
            else:
                self.Ksw_types = Ksw_types
        else: #full
            assert len(Ksw_types.shape) in [1,2], "if full tensor conductivity true then Ksw_types scalar or or 'flattened' row-major representation of entries"
            if len(Ksw_types.shape)==1:
                self.Ksw_types = numpy.zeros((self.nMaterialTypes,self.nd**2),'d')
                for I in range(self.nd):
                    self.Ksw_types[:,I*self.nd+I] = Ksw_types
            else:
                assert Ksw_types.shape[1] == self.nd**2
                self.Ksw_types = Ksw_types
        #not the best place for this
        self.rwork_psk_tolerances = numpy.zeros((self.nPskTolerances,),'d')
        for i,tol in enumerate(self.psk_tolerances[self.psk_model].values()):
            self.rwork_psk_tolerances[i] = tol
#         #mwf do some debugging
#         assert self.elementMaterialTypes.max() < self.nMaterialTypes
#         assert self.elementMaterialTypes.min() == 0
#         assert self.exteriorElementBoundaryTypes.max()  < self.nMaterialTypes
#         assert self.exteriorElementBoundaryTypes.min() == 0
#         assert self.elementBoundaryTypes.max() <  self.nMaterialTypes
#         assert self.elementBoundaryTypes.min() == 0
    #

class TwophaseDarcy_fc(TwophaseDarcyFlow_base):
    r"""continuity equation for each phase

    .. math::

        \pd{m_w}{t} - \deld (\ten{a}_w \grad \phi_w) + r_w = 0
        \pd{m_n}{t} - \deld (\ten{a}_n \grad \phi_n) + r_n = 0

    """
    # (normalized) mass for each phase

    # .. math::

    #     m_i = \theta_i \rho_i,  i=w,n

    #     \theta_i = \theta_s S_i

    #     \rho_i   = \varrho_i/\varrho_{i,0}, i=w,n

    # where :math:`S_i` is the saturation for each phase,
    # :math:`\varrho_{i}` is the density and :math:`\varrho_{i,0}` is a
    # reference value

    # (normalized) mass flux for each phase is

    # .. math::

    #     \vec \sigma_i = - \ten{a}_{i}\grad \phi_i,    i=w,n
    #     \ten{a}_i = \rho_i k_{r,i}/\hat{\mu}_i \ten{K}_s

    # and the potentials are defined as

    # .. math::

    #     \phi_{w}  = \psi_{w} - \rho_w \vec g\cdot \vec x
    #     \phi_{n}  = \psi_{n} - b \rho_n \vec g\cdot \vec x

    # where

    # .. math::

    #     \psi_{i} = p_{i}/|\vec g|\rho_{w,0}
    #     b        = \varrho_{n,0}/\varrho_{w,0}
    #     \hat{mu}_{i} = \mu_i/\mu_{w}

    # for the pressure of each phase, :math:`p_i`, and we have the
    # capillary pressure relation

    # .. math::

    #     \psi_{c} = \psi_{n} - \psi_{w}

    # The dependent variables are :math:`S_w`, and :math:`\psi_w`

    # Note :math:`S_n = 1-S_w`

    # needs to be implemented

    # .. math::

    #     r_i = r_i(\vec x,t)

    # slight compressibility assumed in spatial gradients
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_fc_sd_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_fc_sd_het_matType = staticmethod(twophaseDarcy_fc_sd_het_matType)
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=TwophaseDarcyFlow_base.default_density_w_parameters,
                 density_n_parameters=TwophaseDarcyFlow_base.default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 diagonal_conductivity=True,
                 spatialCompressibilityFlag=0,
                 nPSKsplineKnots=None):#0, slight compressibility, full compressibility otherwise
        TwophaseDarcyFlow_base.__init__(self,
                                        nd=nd,
                                        dimensionless_gravity=dimensionless_gravity,
                                        density_w=density_w,
                                        density_n=density_n,
                                        viscosity_w=viscosity_w,
                                        viscosity_n=viscosity_n,
                                        density_w_parameters=density_w_parameters,
                                        density_n_parameters=density_n_parameters,
                                        psk_model=psk_model,
                                        nMaterialTypes=nMaterialTypes,
                                        diagonal_conductivity=diagonal_conductivity,
                                        nPSKsplineKnots=nPSKsplineKnots)

        nc=2
        variableNames=['s_w','psi_w']
        mass = {0:{0:'linear',1:'nonlinear'},
                1:{0:'nonlinear',1:'nonlinear'}}
        advection = {}
        #advection = {0:{0:'nonlinear',1:'nonlinear'},
        #             1:{0:'nonlinear',1:'nonlinear'}} #don't need 1:nonlinear if using slight compressibility
        hamiltonian={}
        potential = {0:{1:'nonlinear',#actually this is just linear if we use slight compressibility
                        0:'nonlinear'},#don't really need saturation dependence for potential
                     1:{0:'nonlinear',
                        1:'nonlinear'}}
        diffusion = {0:{0:{0:'nonlinear',
                           1:'nonlinear'}},
                     1:{1:{0:'nonlinear',
                           1:'nonlinear'}}}
        reaction = {0:{0:'linear'},
                    1:{1:'linear'}}


        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i')),
                                      (1,1):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i')),
                                      (1,1):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}


        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)

        self.spatialCompressibilityFlag = spatialCompressibilityFlag
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
            vol_frac_w = self.q[('vol_frac',0)]
            vol_frac_n = self.q[('vol_frac',1)]
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
            vol_frac_w = self.ebqe[('vol_frac',0)]
            vol_frac_n = self.ebqe[('vol_frac',1)]
        elif c[('u',0)].shape == self.ip_shape:
            materialTypes = self.materialTypes_ip
            vol_frac_w = self.ip[('vol_frac',0)]
            vol_frac_n = self.ip[('vol_frac',1)]
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
            vol_frac_w = self.ebq[('vol_frac',0)]
            vol_frac_n = self.ebq[('vol_frac',1)]
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        assert self.rwork_psk is not None
        assert self.iwork_psk is not None
        #mwf do some debugging
        assert materialTypes.max() < self.nMaterialTypes
        assert materialTypes.min() == 0


        self.twophaseDarcy_fc_sd_het_matType(self.psk_types[self.psk_model],
                                             self.density_types[self.density_w_model],
                                             self.density_types[self.density_n_model],
                                             self.sdInfo[(0,0)][0],
                                             self.sdInfo[(0,0)][1],
                                             materialTypes,
                                             self.muw,
                                             self.mun,
                                             self.omega_types,
                                             self.Ksw_types,
                                             self.b,
                                             self.rwork_psk,self.iwork_psk,
                                             self.rwork_psk_tolerances,
                                             self.rwork_density_w,
                                             self.rwork_density_n,
                                             self.g[:self.nd],#todo get consistent on dimension setting
                                             c['x'],
                                             c[('u',0)],
                                             c[('u',1)],
                                             c[('m',0)],
                                             c[('dm',0,0)],
                                             c[('dm',0,1)],
                                             c[('m',1)],
                                             c[('dm',1,0)],
                                             c[('dm',1,1)],
                                             c['psi_n'],
                                             c[('dpsi_n',0)],
                                             c[('dpsi_n',1)],
                                             c[('phi',0)],
                                             c[('dphi',0,1)],
                                             c[('phi',1)],
                                             c[('dphi',1,1)],
                                             c[('dphi',1,0)],
                                             c[('a',0,0)],
                                             c[('da',0,0,0)],
                                             c[('da',0,0,1)],
                                             c[('a',1,1)],
                                             c[('da',1,1,0)],
                                             c[('da',1,1,1)])

        #mwf hack, need to put this in?
        c[('dphi',0,0)].fill(0.0)
        self.twophaseDarcy_vol_frac(materialTypes,
                                    self.omega_types,
                                    c[('u',0)],
                                    vol_frac_w,
                                    vol_frac_n)

        #mwf debug
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('da',1,1,0)]).any() or
            numpy.isnan(c[('a',1,1)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()
#

class FullyCoupledMualemVanGenuchten(TwophaseDarcy_fc):
    """Formulation using phase continuity equations and Van-Genuchten
    Mualem psk relations

    Basically a convenience wrapper for fully coupled approximation
    with volume-fraction based inputs as in Richards' equation
    formulations

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 density_w_params,
                 density_n_params,
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 use_spline = False,
                 nPSKsplineKnots=None):
        self.use_spline = use_spline
        psk_model = 'VGM'
        if self.use_spline:
            assert nPSKsplineKnots is not None
            psk_model = 'PSKspline'
        TwophaseDarcy_fc.__init__(self,
                                  nd=nd,
                                  dimensionless_gravity=dimensionless_gravity,
                                  density_w=density_w,
                                  density_n=density_n,
                                  viscosity_w=viscosity_w,
                                  viscosity_n=viscosity_n,
                                  density_w_parameters=density_w_params,
                                  density_n_parameters=density_n_params,
                                  psk_model=psk_model,
                                  nMaterialTypes=len(thetaR_types),
                                  diagonal_conductivity=diagonal_conductivity,
                                  nPSKsplineKnots=nPSKsplineKnots)
        for input in [vgm_n_types,vgm_alpha_types,thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        self.psk_tolerances['VGM']['eps_small']=vgm_small_eps
        self.psk_tolerances['VGM']['ns_del']   =vgm_ns_del
        if self.use_spline:
            self.splineTableWork = numpy.zeros((self.nPskParams*self.nMaterialTypes),'d')
            pskCalcFlag = 0 #formulation is f(Sw)
            rwork_tmp = numpy.zeros((4,),'d')
            rwork_tol_tmp = numpy.zeros((2,),'d')
            rwork_tol_tmp[0] = max(vgm_small_eps,1.0e-5); rwork_tol_tmp[1] = vgm_ns_del
            plot_splines = False
            if plot_splines:
                #mwf debug
                import pdb
                import matplotlib
                from matplotlib import pylab

            for i in range(self.nMaterialTypes):
                sw_domain = numpy.zeros((self.nPSKsplineKnots,),'d')
                sw_domain[1:-1] = numpy.linspace(Sw_min_types[i],Sw_max_types[i],num=self.nPSKsplineKnots-2)
                sw_domain[0] = 0.99*Sw_min_types[i]; sw_domain[-1] = 1.01*Sw_max_types[i]
                materialOffset = i*self.nPskParams
                rwork_tmp[0] = Sw_min_types[i]; rwork_tmp[1]  = Sw_max_types[i];
                rwork_tmp[2] = vgm_alpha_types[i]; rwork_tmp[3]= vgm_m_types[i]
                self.generateSplineTables(self.psk_types['VGM'],
                                          materialOffset,
                                          pskCalcFlag,
                                          sw_domain,
                                          rwork_tmp,
                                          self.iwork_psk,
                                          rwork_tol_tmp,
                                          self.splineTableWork)

                #mwf debug
                if plot_splines:
                    pdb.set_trace()
                    pylab.figure(1); pylab.plot(self.splineTableWork[materialOffset:materialOffset+self.nPSKsplineKnots],
                                                self.splineTableWork[materialOffset+self.nPSKsplineKnots:materialOffset+2*self.nPSKsplineKnots],'b--')
                    pylab.figure(2); pylab.plot(self.splineTableWork[materialOffset:materialOffset+self.nPSKsplineKnots],
                                                self.splineTableWork[materialOffset+2*self.nPSKsplineKnots:materialOffset+3*self.nPSKsplineKnots],'r--')
                    pylab.figure(2); pylab.plot(self.splineTableWork[materialOffset:materialOffset+self.nPSKsplineKnots],
                                                self.splineTableWork[materialOffset+3*self.nPSKsplineKnots:materialOffset+4*self.nPSKsplineKnots],'g--')
                    pylab.show()
            #
            self.setMaterialTypes(Ksw_types=Ksw_types,
                                  omega_types=thetaS_types,
                                  Sw_max_types=Sw_max_types,
                                  Sw_min_types=Sw_min_types,
                                  psk_spline_types=self.splineTableWork)


        else:
            self.setMaterialTypes(Ksw_types=Ksw_types,
                                  omega_types=thetaS_types,
                                  Sw_max_types=Sw_max_types,
                                  Sw_min_types=Sw_min_types,
                                  vg_alpha_types=vgm_alpha_types,
                                  vg_m_types=vgm_m_types)


class FullyCoupledSimplePSKs(TwophaseDarcy_fc):
    """Formulation using phase continuity equations and 'simp' quadratic
    rel-perm, linear capillary pressure psk relations

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 density_w_params,
                 density_n_params,
                 diagonal_conductivity=True):

        TwophaseDarcy_fc.__init__(self,
                                  nd=nd,
                                  dimensionless_gravity=dimensionless_gravity,
                                  density_w=density_w,
                                  density_n=density_n,
                                  viscosity_w=viscosity_w,
                                  viscosity_n=viscosity_n,
                                  density_w_parameters=density_w_params,
                                  density_n_parameters=density_n_params,
                                  psk_model='simp',
                                  nMaterialTypes=len(thetaR_types),
                                  diagonal_conductivity=diagonal_conductivity)
        for input in [thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types)


class TwophaseDarcy_fc_pp(TwophaseDarcyFlow_base):
    r"""continuity equation for each phase

    .. math::

      \pd{m_w}{t} - \deld (\ten{a}_w \grad \phi_w) + r_w = 0
      \pd{m_n}{t} - \deld (\ten{a}_n \grad \phi_n) + r_n = 0
    
    """
    
    # (normalized) mass for each phase :math:`m_i = \theta_i \rho_i,
    # i=w,n`

    # .. math::

    #    \theta_i = \theta_s S_i
    #    \rho_i   = \varrho_i/\varrho_{i,0}, i=w,n

    # where :math:`S_i` is the saturation for each phase,
    # :math:`\varrho_{i}` is the density and :math:`\varrho_{i,0}` is a
    # reference value

    # (normalized) mass flux for each phase is

    # .. math::

    #    \vec \sigma_i = - \ten{a}_{i}\grad \phi_i,    i=w,n
    #    \ten{a}_i = \rho_i k_{r,i}/\hat{\mu}_i \ten{K}_s

    # and the potentials are defined as

    # .. math::

    #    \phi_{w}  = \psi_{w} - \rho_w \vec g\cdot \vec x
    #    \phi_{n}  = \psi_{n} - b \rho_n \vec g\cdot \vec x

    # where

    # .. math::

    #    \psi_{i} = p_{i}/|\vec g|\rho_{w,0}
    #    b        = \varrho_{n,0}/\varrho_{w,0}
    #    \hat{mu}_{i} = \mu_i/\mu_{w}

    # for the pressure of each phase, p_i, and we have the capillary pressure relation

    # .. math::

    #    \psi_{c} = \psi_{n} - \psi_{w}

    # The dependent variables are

    # .. math::

    #   \psi_w, and \psi_c

    # Note :math:`S_n = 1-S_w`

    # needs to be implemented

    # .. math::

    #   r_i = r_i(\vec x,t)

    # slight compressibility assumed in spatial gradients

    # """
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_fc_pp_sd_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_fc_pp_sd_het_matType = staticmethod(twophaseDarcy_fc_pp_sd_het_matType)
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=TwophaseDarcyFlow_base.default_density_w_parameters,
                 density_n_parameters=TwophaseDarcyFlow_base.default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 diagonal_conductivity=True,
                 nPSKsplineKnots=None):
        TwophaseDarcyFlow_base.__init__(self,
                                        nd=nd,
                                        dimensionless_gravity=dimensionless_gravity,
                                        density_w=density_w,
                                        density_n=density_n,
                                        viscosity_w=viscosity_w,
                                        viscosity_n=viscosity_n,
                                        density_w_parameters=density_w_parameters,
                                        density_n_parameters=density_n_parameters,
                                        psk_model=psk_model,
                                        nMaterialTypes=nMaterialTypes,
                                        diagonal_conductivity=diagonal_conductivity,
                                        nPSKsplineKnots=nPSKsplineKnots)

        nc=2
        variableNames=['psi_w','psi_c']
        mass = {0:{0:'nonlinear',1:'nonlinear'},
                1:{0:'nonlinear',1:'nonlinear'}}
        advection = {}
        hamiltonian={}
        potential = {0:{1:'nonlinear',#actually this is just linear if we use slight compressibility
                        0:'nonlinear'},#don't really need saturation dependence for potential
                     1:{0:'nonlinear',
                        1:'nonlinear'}}
        diffusion = {0:{0:{0:'nonlinear',
                           1:'nonlinear'}},
                     1:{1:{0:'nonlinear',
                           1:'nonlinear'}}}
        reaction = {0:{0:'linear'},
                    1:{1:'linear'}}


        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i')),
                                      (1,1):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i')),
                                      (1,1):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}


        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)


    def initializeElementQuadrature(self,t,cq):
        TwophaseDarcyFlow_base.initializeElementQuadrature(self,t,cq)
        #
        cq['sw'] = numpy.zeros(cq[('u',0)].shape,'d')
        cq[('dpsi_n',0)].fill(1.0)
        cq[('dpsi_n',1)].fill(1.0)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseDarcyFlow_base.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        if ('u',0) in cebq:
            cebq['sw'] = numpy.zeros(cebq[('u',0)].shape,'d')
            cebq[('dpsi_n',0)].fill(1.0)
            cebq[('dpsi_n',1)].fill(1.0)
        if ('u',0) in cebq_global:
            cebq_global['sw'] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            cebq_global[('dpsi_n',0)].fill(1.0)
            cebq_global[('dpsi_n',1)].fill(1.0)
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseDarcyFlow_base.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        cebqe['sw'] = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe[('dpsi_n',0)].fill(1.0)
        cebqe[('dpsi_n',1)].fill(1.0)
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        TwophaseDarcyFlow_base.initializeGeneralizedInterpolationPointQuadrature(self,t,cip)
        cip['sw'] = numpy.zeros(cip[('u',0)].shape,'d')
        cip[('dpsi_n',0)].fill(1.0)
        cip[('dpsi_n',1)].fill(1.0)
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
            vol_frac_w = self.q[('vol_frac',0)]
            vol_frac_n = self.q[('vol_frac',1)]
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
            vol_frac_w = self.ebqe[('vol_frac',0)]
            vol_frac_n = self.ebqe[('vol_frac',1)]
        elif c[('u',0)].shape == self.ip_shape:
            materialTypes = self.materialTypes_ip
            vol_frac_w = self.ip[('vol_frac',0)]
            vol_frac_n = self.ip[('vol_frac',1)]
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
            vol_frac_w = self.ebq[('vol_frac',0)]
            vol_frac_n = self.ebq[('vol_frac',1)]
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        assert self.rwork_psk is not None
        #mwf do some debugging
        assert materialTypes.max() < self.nMaterialTypes
        assert materialTypes.min() == 0

        self.twophaseDarcy_fc_pp_sd_het_matType(self.psk_types[self.psk_model],
                                                self.density_types[self.density_w_model],
                                                self.density_types[self.density_n_model],
                                                self.sdInfo[(0,0)][0],
                                                self.sdInfo[(0,0)][1],
                                                materialTypes,
                                                self.muw,
                                                self.mun,
                                                self.omega_types,
                                                self.Ksw_types,
                                                self.b,
                                                self.rwork_psk,self.iwork_psk,
                                                self.rwork_psk_tolerances,
                                                self.rwork_density_w,
                                                self.rwork_density_n,
                                                self.g[:self.nd],#todo get consistent on dimension setting
                                                c['x'],
                                                c[('u',0)],
                                                c[('u',1)],
                                                c['sw'],
                                                c[('m',0)],
                                                c[('dm',0,0)],
                                                c[('dm',0,1)],
                                                c[('m',1)],
                                                c[('dm',1,0)],
                                                c[('dm',1,1)],
                                                c[('phi',0)],
                                                c[('dphi',0,0)],
                                                c[('phi',1)],
                                                c[('dphi',1,0)],
                                                c[('dphi',1,1)],
                                                c[('a',0,0)],
                                                c[('da',0,0,0)],
                                                c[('da',0,0,1)],
                                                c[('a',1,1)],
                                                c[('da',1,1,0)],
                                                c[('da',1,1,1)])

        #todo put this back in cpp eval
        #c['psi_n'][:] = c[('u',0)]
        #c['psi_n'] += c[('u',1)]
        #c[('dpsi_n',0)].fill(1.0)
        #c[('dpsi_n',1)].fill(1.0)
        c['psi_n'][:] = c[('phi',1)]
        c[('dpsi_n',0)][:] = c[('dphi',1,0)]
        c[('dpsi_n',1)][:] = c[('dphi',1,1)]
        self.twophaseDarcy_vol_frac(materialTypes,
                                    self.omega_types,
                                    c['sw'],
                                    vol_frac_w,
                                    vol_frac_n)
        #mwf debug
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('da',1,1,0)]).any() or
            numpy.isnan(c[('a',1,1)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()
#
class FullyCoupledPressurePressureMualemVanGenuchten(TwophaseDarcy_fc_pp):
    """Formulation using phase continuity equations, pressure-pressure
    formulation and Van-Genuchten Mualem psk relations

    Basically a convenience wrapper for fully coupled approximation
    with volume-fraction based inputs as in Richards' equation
    formulations

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 density_w_params,
                 density_n_params,
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 use_spline = False,
                 nPSKsplineKnots=None):
        self.use_spline = use_spline
        psk_model = 'VGM'
        if self.use_spline:
            assert nPSKsplineKnots is not None
            psk_model = 'PSKspline'

        TwophaseDarcy_fc_pp.__init__(self,
                                     nd=nd,
                                     dimensionless_gravity=dimensionless_gravity,
                                     density_w=density_w,
                                     density_n=density_n,
                                     viscosity_w=viscosity_w,
                                     viscosity_n=viscosity_n,
                                     density_w_parameters=density_w_params,
                                     density_n_parameters=density_n_params,
                                     psk_model=psk_model,
                                     nMaterialTypes=len(thetaR_types),
                                     diagonal_conductivity=diagonal_conductivity,
                                     nPSKsplineKnots=nPSKsplineKnots)
        for input in [vgm_n_types,vgm_alpha_types,thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        self.psk_tolerances['VGM']['eps_small']=vgm_small_eps
        self.psk_tolerances['VGM']['ns_del']   =vgm_ns_del
        if self.use_spline:
            from . import cpskRelations
            self.splineTableWork = numpy.zeros((self.nPskParams*self.nMaterialTypes),'d')
            pskCalcFlag = 1 #formulation is f(psic)
            rwork_tmp = numpy.zeros((4,),'d')
            rwork_tol_tmp = numpy.zeros((2,),'d')
            rwork_tol_tmp[0] = max(vgm_small_eps,1.0e-5); rwork_tol_tmp[1] = vgm_ns_del
            #mwf come up with this as an input
            psic_min_types=numpy.zeros((self.nMaterialTypes,),'d')
            psic_max_types=numpy.zeros((self.nMaterialTypes,),'d')
            for i in range(self.nMaterialTypes):
                psic,krw,krn,dpsic,dkrw,dkrn= cpskRelations.vgm_calc_from_sw(Sw_min_types[i],Sw_min_types[i],Sw_max_types[i],
                                                                             vgm_alpha_types[i],vgm_m_types[i],vgm_small_eps,vgm_ns_del)
                psic_max_types[i]=3.0#psic*0.1
                psic_min_types[i]=max(vgm_small_eps,1.0e-5)
            #mwf debug
            import pdb
            import matplotlib
            from matplotlib import pylab
            for i in range(self.nMaterialTypes):
                psic_domain = numpy.zeros((self.nPSKsplineKnots,),'d')
                psic_domain[1:-1] = numpy.linspace(psic_min_types[i],psic_max_types[i],num=self.nPSKsplineKnots-2)
                psic_domain[0] = 0.99*psic_min_types[i]; psic_domain[-1] = 1.01*psic_max_types[i]
                materialOffset = i*self.nPskParams
                rwork_tmp[0] = Sw_min_types[i]; rwork_tmp[1]  = Sw_max_types[i];
                rwork_tmp[2] = vgm_alpha_types[i]; rwork_tmp[3]= vgm_m_types[i]
                self.generateSplineTables(self.psk_types['VGM'],
                                          materialOffset,
                                          pskCalcFlag,
                                          psic_domain,
                                          rwork_tmp,
                                          self.iwork_psk,
                                          rwork_tol_tmp,
                                          self.splineTableWork)

                #mwf debug
                pdb.set_trace()
                pylab.figure(1); pylab.plot(self.splineTableWork[materialOffset:materialOffset+self.nPSKsplineKnots],
                                            self.splineTableWork[materialOffset+self.nPSKsplineKnots:materialOffset+2*self.nPSKsplineKnots],'b--')
                pylab.figure(2); pylab.plot(self.splineTableWork[materialOffset:self.nPSKsplineKnots],
                                            self.splineTableWork[materialOffset+2*self.nPSKsplineKnots:materialOffset+3*self.nPSKsplineKnots],'r--')
                pylab.figure(2); pylab.plot(self.splineTableWork[materialOffset:self.nPSKsplineKnots],
                                            self.splineTableWork[materialOffset+3*self.nPSKsplineKnots:materialOffset+4*self.nPSKsplineKnots],'g--')
                pylab.show()
            #
            self.setMaterialTypes(Ksw_types=Ksw_types,
                                  omega_types=thetaS_types,
                                  Sw_max_types=Sw_max_types,
                                  Sw_min_types=Sw_min_types,
                                  psk_spline_types=self.splineTableWork)


        else:
            self.setMaterialTypes(Ksw_types=Ksw_types,
                                  omega_types=thetaS_types,
                                  Sw_max_types=Sw_max_types,
                                  Sw_min_types=Sw_min_types,
                                  vg_alpha_types=vgm_alpha_types,
                                  vg_m_types=vgm_m_types)


class FullyCoupledPressurePressureSimplePSKs(TwophaseDarcy_fc_pp):
    """Formulation using phase continuity equations and 'simp' quadratic
    rel-perm, linear capillary pressure psk relations
    pressure-pressure formulation

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 density_w_params,
                 density_n_params,
                 diagonal_conductivity=True):

        TwophaseDarcy_fc_pp.__init__(self,
                                     nd=nd,
                                     dimensionless_gravity=dimensionless_gravity,
                                     density_w=density_w,
                                     density_n=density_n,
                                     viscosity_w=viscosity_w,
                                     viscosity_n=viscosity_n,
                                     density_w_parameters=density_w_params,
                                     density_n_parameters=density_n_params,
                                     psk_model='simp',
                                     nMaterialTypes=len(thetaR_types),
                                     diagonal_conductivity=diagonal_conductivity)
        for input in [thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types)




###########
class TwophaseDarcy_split_pressure_base(TwophaseDarcyFlow_base):
    """Base class for 'pressure' or total flow conservation equation in
    fractional flow formulations. This

    The primary functionality of the base class is to handle synchronization with a 'saturation' model to
    get the saturation, S_w, and capillary pressure (head), \\psi_c, variables

    """
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=TwophaseDarcyFlow_base.default_density_w_parameters,
                 density_n_parameters=TwophaseDarcyFlow_base.default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nSatModel=1,
                 diagonal_conductivity=True,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcyFlow_base.__init__(self,
                                        nd=nd,
                                        dimensionless_gravity=dimensionless_gravity,
                                        density_w=density_w,
                                        density_n=density_n,
                                        viscosity_w=viscosity_w,
                                        viscosity_n=viscosity_n,
                                        density_w_parameters=density_w_parameters,
                                        density_n_parameters=density_n_parameters,
                                        psk_model=psk_model,
                                        nMaterialTypes=nMaterialTypes,
                                        diagonal_conductivity=diagonal_conductivity)

        self.nc=1
        self.nd=nd
        #
        self.nSatModel=nSatModel
        #for debugging
        self.swConstant=swConstant
        self.capillaryDiffusionScaling=capillaryDiffusionScaling
    def attachModels(self,modelList):
        if self.nSatModel is None:
            print('Warning TwophaseDarcy_split_pressure_base nSatModel is None returning in attachModels')
            return
        #not ideal, but need a way to force nonlinear potential to be evaluated in saturation model
        modelList[self.nSatModel].calculateElementCoefficients()
        self.q_s_w   = modelList[self.nSatModel].q[('u',0)]
        self.ebqe_s_w = modelList[self.nSatModel].ebqe[('u',0)]
        if ('u',0) in modelList[self.nSatModel].ebq:
            self.ebq_s_w = modelList[self.nSatModel].ebq[('u',0)]
        #mwf need to check
        assert ('u',0) in modelList[self.nSatModel].phi_ip
        assert self.ip_s_w.shape ==  modelList[self.nSatModel].phi_ip[('u',0)].shape
        self.ip_s_w = modelList[self.nSatModel].phi_ip[('u',0)]
        #mwf hack 04/03/09 skip
        #assert modelList[self.nSatModel].phi_ip.has_key(('grad(phi)',0))
        self.ip_grad_psic = None#modelList[self.nSatModel].phi_ip[('grad(phi)',0)]
        #mwf end ip stuff
        self.q_grad_psic   = modelList[self.nSatModel].q[('grad(phi)',0)]
        self.ebqe_grad_psic = modelList[self.nSatModel].ebqe[('grad(phi)',0)]
        if ('grad(phi)',0) in modelList[self.nSatModel].ebq:
            self.ebq_grad_psic = modelList[self.nSatModel].ebq[('grad(phi)',0)]
        self.q_psic   = modelList[self.nSatModel].q[('phi',0)]
        self.ebqe_psic= modelList[self.nSatModel].ebqe[('phi',0)]
        if ('phi',0) in modelList[self.nSatModel].ebq:
            self.ebq_psic = modelList[self.nSatModel].ebq[('phi',0)]
        assert ('phi',0) in modelList[self.nSatModel].phi_ip
        assert self.ip_psic.shape ==  modelList[self.nSatModel].phi_ip[('phi',0)].shape
        self.ip_psic = modelList[self.nSatModel].phi_ip[('phi',0)]
        #
        self.q_grad_sw   = modelList[self.nSatModel].q[('grad(u)',0)]
        self.ebqe_grad_sw = modelList[self.nSatModel].ebqe[('grad(u)',0)]
        if ('grad(u)',0) in modelList[self.nSatModel].ebq:
            self.ebq_grad_sw = modelList[self.nSatModel].ebq[('grad(u)',0)]

    def initializeElementQuadrature(self,t,cq):
        TwophaseDarcyFlow_base.initializeElementQuadrature(self,t,cq)
        #set up dummy values in case we're not running the other model
        self.q_s_w   = numpy.zeros(cq[('u',0)].shape,'d')
        self.q_s_w.fill(self.swConstant)
        self.q_grad_psic   = numpy.zeros(cq[('f',0)].shape,'d')
        self.q_psic        = numpy.zeros(cq[('u',0)].shape,'d')
        self.q_grad_sw   = numpy.zeros(cq[('f',0)].shape,'d')
        #mwf not sure if this is ok
        cq['psi_n'] = numpy.zeros(cq[('u',0)].shape,'d')
        cq[('dpsi_n',0)] = numpy.ones(cq[('u',0)].shape,'d')

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseDarcyFlow_base.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        #set up dummy values in case we're not running the other model
        self.ebq_s_w = numpy.zeros(cebq[('u',0)].shape,'d')
        self.ebq_s_w.fill(self.swConstant)
        self.ebq_grad_psic = numpy.zeros(cebq[('f',0)].shape,'d')
        self.ebq_psic = numpy.zeros(cebq[('u',0)].shape,'d')
        self.ebq_grad_sw = numpy.zeros(cebq[('f',0)].shape,'d')
        if ('u',0) in cebq:
            cebq['psi_n'] = numpy.zeros(cebq[('u',0)].shape,'d')
            cebq[('dpsi_n',0)] = numpy.ones(cebq[('u',0)].shape,'d')
        if ('u',0) in cebq_global:
            cebq_global['psi_n'] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            cebq_global[('dpsi_n',0)] = numpy.ones(cebq_global[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseDarcyFlow_base.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        #set up dummy values in case we're not running the other model
        self.ebqe_s_w = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.ebqe_s_w.fill(self.swConstant)
        self.ebqe_grad_psic = numpy.zeros(cebqe[('f',0)].shape,'d')
        self.ebqe_psic = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.ebqe_grad_sw = numpy.zeros(cebqe[('f',0)].shape,'d')
        cebqe['psi_n'] = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe[('dpsi_n',0)] = numpy.ones(cebqe[('u',0)].shape,'d')
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        TwophaseDarcyFlow_base.initializeGeneralizedInterpolationPointQuadrature(self,t,cip)
        #set up dummy values in case we're not running the other model
        self.ip_s_w = numpy.zeros(cip[('u',0)].shape,'d')
        self.ip_s_w.fill(self.swConstant)
        self.ip_grad_psic = numpy.zeros(cip[('f',0)].shape,'d')
        self.ip_psic = numpy.zeros(cip[('u',0)].shape,'d')
        self.ip_grad_sw = numpy.zeros(cip[('f',0)].shape,'d')
        cip['psi_n'] = numpy.zeros(cip[('u',0)].shape,'d')
        cip[('dpsi_n',0)] = numpy.ones(cip[('u',0)].shape,'d')


#
class TwophaseDarcy_split_saturation_base(TwophaseDarcyFlow_base):
    """Base class for aqueous phase mass conservation equation
    (saturation equation) in a fractional flow formulation.

    The primary responsibility of the base class is to handle
    synchronization with the 'pressure' equation to get the total flow
    velocity variable, q_t, and aqueous phase pressure head, psi_w

    """
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=TwophaseDarcyFlow_base.default_density_w_parameters,
                 density_n_parameters=TwophaseDarcyFlow_base.default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nPressModel=0,
                 diagonal_conductivity=True,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcyFlow_base.__init__(self,
                                        nd=nd,
                                        dimensionless_gravity=dimensionless_gravity,
                                        density_w=density_w,
                                        density_n=density_n,
                                        viscosity_w=viscosity_w,
                                        viscosity_n=viscosity_n,
                                        density_w_parameters=density_w_parameters,
                                        density_n_parameters=density_n_parameters,
                                        psk_model=psk_model,
                                        nMaterialTypes=nMaterialTypes,
                                        diagonal_conductivity=diagonal_conductivity)

        self.nc=1
        self.nd=nd
        self.nPressModel=nPressModel
        #for debugging
        self.qScalarConstant=1.0
        self.capillaryDiffusionScaling=capillaryDiffusionScaling
        self.advectionScaling=advectionScaling
    def attachModels(self,modelList):
        if self.nPressModel is None:
            print('Warning TwophaseDarcy_split_saturation_base nPressModel is None returning in attachModels')
            return
        self.flowModel = modelList[self.nPressModel]
        #
        self.q_q_t    = modelList[self.nPressModel].q[('velocity',0)]
        self.ebqe_q_t  = modelList[self.nPressModel].ebqe[('velocity',0)]
        if ('velocity',0) in modelList[self.nPressModel].ebq:
            self.ebq_q_t  = modelList[self.nPressModel].ebq[('velocity',0)]
        #do we really need other model values for q_t in potential calculation?
        assert self.ip_psiw.shape == modelList[self.nPressModel].phi_ip[('u',0)].shape
        self.ip_psiw = modelList[self.nPressModel].phi_ip[('u',0)]
        self.q_psiw    = modelList[self.nPressModel].q[('u',0)]
        self.ebqe_psiw = modelList[self.nPressModel].ebqe[('u',0)]
        if ('u',0) in modelList[self.nPressModel].ebq:
            self.ebq_psiw = modelList[self.nPressModel].ebq[('u',0)]
    def initializeElementQuadrature(self,t,cq):
        TwophaseDarcyFlow_base.initializeElementQuadrature(self,t,cq)
        #set up dummy values in case we're not running the other model
        self.q_q_t   = numpy.zeros(cq[('f',0)].shape,'d')
        self.q_q_t[:] = self.qScalarConstant
        self.q_psiw   = numpy.ones(cq[('u',0)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseDarcyFlow_base.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        #set up dummy values in case we're not running the other model
        self.ebq_q_t = numpy.zeros(cebq[('f',0)].shape,'d')
        self.ebq_q_t[:] = self.qScalarConstant
        self.ebq_psiw = numpy.ones(cebq[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseDarcyFlow_base.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        #set up dummy values in case we're not running the other model
        self.ebqe_q_t = numpy.zeros(cebqe[('f',0)].shape,'d')
        self.ebqe_q_t[:] = self.qScalarConstant
        self.ebqe_psiw = numpy.ones(cebqe[('u',0)].shape,'d')
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        TwophaseDarcyFlow_base.initializeGeneralizedInterpolationPointQuadrature(self,t,cip)
        #set up dummy values in case we're not running the other model
        self.ip_q_t = numpy.zeros(cip[('f',0)].shape,'d')
        self.ip_q_t[:] = self.qScalarConstant
        self.ip_psiw = numpy.ones(cip[('u',0)].shape,'d')

class TwophaseDarcy_incompressible_split_pressure(TwophaseDarcy_split_pressure_base):
    """Total flow conservation equation in an incompressible fractional
    flow formulation

    """
    # Saturation equation

    # .. math::

    #     \begin{eqnarray}
    #     \label{eq:2p-ff-mb-w}
    #     \pd{m_w}{t} + \deld\left(\vec f_w - \ten{a}_{w}\grad \phi_w \right) + r_w &=& 0
    #     \end{eqnarray}
    #     and total flow conservation equation
    #     \begin{eqnarray}
    #     \label{eq:2p-ff-mb-m}
    #     \deld\left(\vec f_m - \ten{a}_m \grad \phi_m \right) + r_m &=& 0
    #     \end{eqnarray}

    #     \begin{table}[ht]
    #     \caption{Coefficient definitions for Two-phase flow, \eqn{2p-ff-mb-w} and \eqn{2p-ff-mb-m}
    #     \label{tab:2p-ff-coef-1}
    #     }
    #     \begin{tabular}{cc}
    #     \hline
    #     Var. & Def. \\
    #     \hline
    #     $u_w$ & $S_w $ \\
    #     $u_n$ & $\psi_w $ \\
    #     $\phi_w$ & $\psi_c$ \\
    #     $\phi_m$ & $\psi_w$  \\
    #     $m_w$ & $\theta_s \rho_{w}S_w$  \\
    #     $\vec f_w$ & $\gvec{\sigma}_t F_w - \ten{K}_s\lambda_wF_n\left(b\rho_n - \rho_w\right)\vec g_u$  \\
    #     $\vec f_m$ &$-\ten{K}_s\lambda_tF_n\grad \psi_c + \ten{K}_s\vec g\lambda_t\left[\rho_w + F_n\left(b\rho_n - \rho_w\right)\right]$\\
    #     $\ten a_{w,w}$ & $-\lambda_wF_n \ten{K}_{s}$ \\
    #     $\ten a_{m,m}$ & $\lambda_t \ten{K}_{s}$ \\
    #     \hline
    #     $F_i $ & $\lambda_i/\lambda_t $, $i=w,n$  \\
    #     $\grad \psi_c $ & $\od{\psi_c}{S_w}\grad S_w $  \\
    #     $\gvec {\sigma}_t$ & $\gvec \sigma_w + \gvec \sigma_n$\\
    #     \hline
    #     \end{tabular}
    #     \end{table}

    #  Here :math:`S_i` is the saturation for each phase,
    #  :math:`\varrho_{i}` is the density and :math:`\varrho_{i,0}` is a
    #  reference value

    # (normalized) mass flux for each phase is

    # .. math::

    #    \vec \sigma_i = - \ten{a}_{i}\grad \phi_i,    i=w,n

    # and
   
    # .. math::
    #    \psi_{i} = p_{i}/|\vec g|\rho_{w,0}
    #    b        = \varrho_{n,0}/\varrho_{w,0}
    #    \hat{mu}_{i} = \mu_i/\mu_{w}

    # for the pressure of each phase, p_i, and we have the capillary pressure relation

    # .. math::

    #    \psi_{c} = \psi_{n} - \psi_{w}

    # The dependent variables are :math:`S_w`, and :math:`\psi_w`

    # Note :math:`S_n = 1-S_w`

    # needs to be implemented

    # .. math::

    #   r_m = r_m(\vec x,t)

    # """
#    TODO:
#
#      Figure out if really need to evaluate potential interpolation points
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_incompressible_split_sd_pressure_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_incompressible_split_sd_pressure_het_matType = staticmethod(twophaseDarcy_incompressible_split_sd_pressure_het_matType)
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nSatModel=1,
                 diagonal_conductivity=True,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcy_split_pressure_base.__init__(self,
                                                   nd=nd,
                                                   dimensionless_gravity=dimensionless_gravity,
                                                   density_w=density_w,
                                                   density_n=density_n,
                                                   viscosity_w=viscosity_w,
                                                   viscosity_n=viscosity_n,
                                                   psk_model=psk_model,
                                                   nMaterialTypes=nMaterialTypes,
                                                   nSatModel=nSatModel,
                                                   diagonal_conductivity=diagonal_conductivity,
                                                   swConstant=swConstant,
                                                   capillaryDiffusionScaling=capillaryDiffusionScaling)

        variableNames=['psi_w']
        #these are only nonlinear for compressible flow
        mass      = {}
        advection = {0:{0:'linear'}}
        hamiltonian={}
        diffusion = {0:{0: {0:'linear'}}}
        potential = {0:{0: 'u'}}
        reaction  = {0:{0:'linear'}}

        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}


        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_s_w.shape:
            materialTypes = self.materialTypes_q
            s_w = self.q_s_w
            grad_psic = self.q_grad_psic
            grad_sw   = self.q_grad_sw
            c['psi_n']= numpy.copy(self.q_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.q[('vol_frac',0)]
            vol_frac_n = self.q[('vol_frac',1)]
            #mwf debug
            #import pdb
            #pdb.set_trace()
            #for eN in range(c['x'].shape[0]):
            #    for k in range(c['x'].shape[1]):
            #        if (1.25 <= c['x'][eN,k,0] and c['x'][eN,k,0] <= 2.25 and
            #            1.25 <= c['x'][eN,k,1] and c['x'][eN,k,1] <= 1.5):
                        #mwf major hack to test initialization
                        #grad_psic[eN,k,0] = 0.0
                        #grad_psic[eN,k,1] = 1.0
            #
            #            print "split press eval inside lens eN=%s k=%s x=%s grad_psiw= %s grad_psic= %s s_w= %s " % (eN,k,c['x'][eN,k],c[('grad(u)',0)][eN,k],grad_psic[eN,k],s_w[eN,k])
            #        else:
            #            print "split press eval outside lens eN=%s k=%s x=%s grad_psiw= %s grad_psic= %s s_w= %s " % (eN,k,c['x'][eN,k],c[('grad(u)',0)][eN,k],grad_psic[eN,k],s_w[eN,k])

            #import pdb
            #pdb.set_trace()
        elif c[('u',0)].shape == self.ebqe_s_w.shape:
            materialTypes = self.materialTypes_ebqe
            s_w = self.ebqe_s_w
            grad_psic = self.ebqe_grad_psic
            grad_sw   = self.ebqe_grad_sw
            c['psi_n']= numpy.copy(self.ebqe_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.ebqe[('vol_frac',0)]
            vol_frac_n = self.ebqe[('vol_frac',1)]
        elif c[('u',0)].shape == self.ip_s_w.shape:
            c['psi_n']= numpy.copy(self.ip_psic)
            c['psi_n'] += c[('u',0)]
            #mwf hack 04/03/09 skip
            return
            materialTypes = self.materialTypes_ip
            s_w = self.ip_s_w
            grad_psic = self.ip_grad_psic
            grad_sw   = self.ip_grad_sw
            vol_frac_w = self.ip[('vol_frac',0)]
            vol_frac_n = self.ip[('vol_frac',1)]
        else:
            assert c[('u',0)].shape == self.ebq_s_w.shape
            materialTypes = self.materialTypes_ebq
            s_w = self.ebq_s_w
            grad_psic = self.ebq_grad_psic
            grad_sw   = self.ebq_grad_sw
            c['psi_n']= numpy.copy(self.ebq_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.ebq[('vol_frac',0)]
            vol_frac_n = self.ebq[('vol_frac',1)]
        assert self.rwork_psk is not None

        self.twophaseDarcy_incompressible_split_sd_pressure_het_matType(self.psk_types[self.psk_model],
                                                                        self.sdInfo[(0,0)][0],
                                                                        self.sdInfo[(0,0)][1],
                                                                        materialTypes,
                                                                        self.muw,
                                                                        self.mun,
                                                                        self.omega_types,
                                                                        self.Ksw_types,
                                                                        self.b,
                                                                        self.capillaryDiffusionScaling,
                                                                        self.rwork_psk,self.iwork_psk,
                                                                        self.rwork_psk_tolerances,
                                                                        self.rwork_density_w,
                                                                        self.rwork_density_n,
                                                                        self.g[:self.nd],
                                                                        s_w,
                                                                        grad_psic,
                                                                        c[('f',0)],
                                                                        c[('a',0,0)])

        self.twophaseDarcy_vol_frac(materialTypes,
                                    self.omega_types,
                                    s_w,
                                    vol_frac_w,
                                    vol_frac_n)
        #mwf debug
        if (numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any()):
            import pdb
            pdb.set_trace()
        #mwf debug
        #import pdb
        #pdb.set_trace()
        if c[('u',0)].shape == self.q_s_w.shape:
            c['Ks']=c[('a',0,0)][:,:,0]

#
class TwophaseDarcy_incompressible_split_saturation(TwophaseDarcy_split_saturation_base):
    """Aqueous phase mass conservation equation (saturation equation) in
    an incompressible fractional flow formulation
 
    """
    
    # Saturation equation

    # .. math::

    #     \begin{eqnarray}
    #     \label{eq:2p-ff-mb-w}
    #     \pd{m_w}{t} + \deld\left(\vec f_w - \ten{a}_{w}\grad \phi_w \right) + r_w &=& 0
    #     \end{eqnarray}
    #     and total flow conservation equation
    #     \begin{eqnarray}
    #     \label{eq:2p-ff-mb-m}
    #     \deld\left(\vec f_m - \ten{a}_m \grad \phi_m \right) + r_m &=& 0
    #     \end{eqnarray}

    #     \begin{table}[ht]
    #     \caption{Coefficient definitions for Two-phase flow, \eqn{2p-ff-mb-w} and \eqn{2p-ff-mb-m}
    #     \label{tab:2p-ff-coef-1}
    #     }
    #     \begin{tabular}{cc}
    #     \hline
    #     Var. & Def. \\
    #     \hline
    #     $u_w$ & $S_w $ \\
    #     $u_n$ & $\psi_w $ \\
    #     $\phi_w$ & $\psi_c$ \\
    #     $\phi_m$ & $\psi_w$  \\
    #     $m_w$ & $\theta_s \rho_{w}S_w$  \\
    #     $\vec f_w$ & $\gvec{\sigma}_t F_w - \ten{K}_s\lambda_wF_n\left(b\rho_n - \rho_w\right)\vec g_u$  \\
    #     $\vec f_m$ &$-\ten{K}_s\lambda_tF_n\grad \psi_c + \ten{K}_s\vec g\lambda_t\left[\rho_w + F_n\left(b\rho_n - \rho_w\right)\right]$\\
    #     $\ten a_{w,w}$ & $-\lambda_wF_n \ten{K}_{s}$ \\
    #     $\ten a_{m,m}$ & $\lambda_t \ten{K}_{s}$ \\
    #     \hline
    #     $F_i $ & $\lambda_i/\lambda_t $, $i=w,n$  \\
    #     $\grad \psi_c $ & $\od{\psi_c}{S_w}\grad S_w $  \\
    #     $\gvec {\sigma}_t$ & $\gvec \sigma_w + \gvec \sigma_n$\\
    #     \hline
    #     \end{tabular}
    #     \end{table}

    #  Here :math:`S_i` is the saturation for each phase, :math:`\varrho_{i}` is the density and
    #  :math:`\varrho_{i,0}` is a reference value

    # (normalized) mass flux for each phase is

    # .. math::

    #    \vec \sigma_i = - \ten{a}_{i}\grad \phi_i,    i=w,n

    # and

    # .. math::

    #    \psi_{i} = p_{i}/|\vec g|\rho_{w,0}
    #    b        = \varrho_{n,0}/\varrho_{w,0}
    #    \hat{mu}_{i} = \mu_i/\mu_{w}

    # for the pressure of each phase, p_i, and we have the capillary pressure relation

    # .. math::

    #    \psi_{c} = \psi_{n} - \psi_{w}

    # The dependent variables are :math:`S_w`, and :math:`\psi_w`

    # Note :math:`S_n = 1-S_w`

    # needs to be implemented

    # .. math::

    #   r_m = r_m(\vec x,t)
    
    # """
#    TODO:
#
#      Figure out if really need to evaluate potential interpolation points
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_incompressible_split_sd_saturation_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_incompressible_split_sd_saturation_het_matType = staticmethod(twophaseDarcy_incompressible_split_sd_saturation_het_matType)
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nPressModel=0,
                 diagonal_conductivity=True,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcy_split_saturation_base.__init__(self,
                                                     nd=nd,
                                                     dimensionless_gravity=dimensionless_gravity,
                                                     density_w=density_w,
                                                     density_n=density_n,
                                                     viscosity_w=viscosity_w,
                                                     viscosity_n=viscosity_n,
                                                     psk_model=psk_model,
                                                     nMaterialTypes=nMaterialTypes,
                                                     nPressModel=nPressModel,
                                                     diagonal_conductivity=diagonal_conductivity,
                                                     qScalarConstant=qScalarConstant,
                                                     capillaryDiffusionScaling=capillaryDiffusionScaling,
                                                     advectionScaling=advectionScaling)

        variableNames=['s_w']
        mass      = {0:{0:'linear'}}
        advection = {0:{0:'nonlinear'}}
        hamiltonian={}
        diffusion = {0:{0:{0:'nonlinear'}}}
        potential = {0:{0: 'nonlinear'}}
        reaction  = {0:{0:'linear'}}

        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}


        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)
    def evaluate(self,t,c):
        if c[('f',0)].shape == self.q_q_t.shape:
            materialTypes = self.materialTypes_q
            q_t = self.q_q_t
            psiw = self.q_psiw
            vol_frac_w = self.q[('vol_frac',0)]
            vol_frac_n = self.q[('vol_frac',1)]
        elif c[('f',0)].shape == self.ebqe_q_t.shape:
            materialTypes = self.materialTypes_ebqe
            q_t = self.ebqe_q_t
            psiw = self.ebqe_psiw
            vol_frac_w = self.ebqe[('vol_frac',0)]
            vol_frac_n = self.ebqe[('vol_frac',1)]
        elif c[('f',0)].shape == self.ip_q_t.shape:
            materialTypes = self.materialTypes_ip
            q_t = self.ip_q_t
            psiw = self.ip_psiw
            vol_frac_w = self.ip[('vol_frac',0)]
            vol_frac_n = self.ip[('vol_frac',1)]
        else:
            assert c[('f',0)].shape == self.ebq_q_t.shape
            materialTypes = self.materialTypes_ebq
            q_t = self.ebq_q_t
            psiw = self.ebq_psiw
            vol_frac_w = self.ebq[('vol_frac',0)]
            vol_frac_n = self.ebq[('vol_frac',1)]
        assert self.rwork_psk is not None
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.twophaseDarcy_incompressible_split_sd_saturation_het_matType(self.psk_types[self.psk_model],
                                                                          self.sdInfo[(0,0)][0],
                                                                          self.sdInfo[(0,0)][1],
                                                                          materialTypes,
                                                                          self.muw,
                                                                          self.mun,
                                                                          self.omega_types,
                                                                          self.Ksw_types,
                                                                          self.b,
                                                                          self.capillaryDiffusionScaling,
                                                                          self.advectionScaling,
                                                                          self.rwork_psk,self.iwork_psk,
                                                                          self.rwork_psk_tolerances,
                                                                          self.rwork_density_w,
                                                                          self.rwork_density_n,
                                                                          self.g[:self.nd],
                                                                          q_t,
                                                                          c[('u',0)],
                                                                          c[('m',0)],
                                                                          c[('dm',0,0)],
                                                                          c[('phi',0)],
                                                                          c[('dphi',0,0)],
                                                                          c[('f',0)],
                                                                          c[('df',0,0)],
                                                                          c[('a',0,0)],
                                                                          c[('da',0,0,0)])

        self.twophaseDarcy_vol_frac(materialTypes,
                                    self.omega_types,
                                    c[('u',0)],
                                    vol_frac_w,
                                    vol_frac_n)

        #mwf debug
#         if c[('f',0)].shape == self.ip_q_t.shape:
#             for eN in range(c['x'].shape[0]):
#                 for k in range(c['x'].shape[1]):
#                     if (1.25 <= c['x'][eN,k,0] and c['x'][eN,k,0] <= 2.25 and
#                         1.25 <= c['x'][eN,k,1] and c['x'][eN,k,1] <= 1.5):
#                         print "split sat eval inside lens eN=%s k=%s x=%s matID=%s psic= %s s_w= %s " % (eN,k,c['x'][eN,k],materialTypes[eN],c[('phi',0)][eN,k],c[('u',0)][eN,k])
#                         if (1.25 < c['x'][eN,k,0] and c['x'][eN,k,0] < 2.25 and
#                             1.25 < c['x'][eN,k,1] and c['x'][eN,k,1] < 1.5):
#                             assert materialTypes[eN] == 1
#                     else:
#                         print "split sat eval outside lens eN=%s k=%s x=%s matID=%s psic= %s s_w= %s " % (eN,k,c['x'][eN,k],materialTypes[eN],c[('phi',0)][eN,k],c[('u',0)][eN,k])
#                         assert materialTypes[eN] == 0


            #import pdb
            #pdb.set_trace()

        #mwf debug
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('phi',0)]).any() or
            numpy.isnan(c[('dphi',0,0)]).any() or
            numpy.isnan(c[('df',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()


class TwophaseDarcy_compressible_split_pressure(TwophaseDarcy_split_pressure_base):
    """Total flow conservation equation in a split compressible
    fractional flow formulation Right now, the options are

    compressibility for the non-wetting phase (compressibleN) :
    compressiblityFlag=1
    
    compressibility for both phases but with the slight compressiblity
    assumption (spatial density gradients are negligible)
    compressiblityFlag=2

    """
    # Saturation equation

    # .. math::

    #     \begin{eqnarray}
    #     \label{eq:2p-c-ff-mb-w}
    #     \pd{m_w}{t} + \deld\left(\vec f_w - \ten{a}_{w}\grad \phi_w \right) + r_w &=& 0
    #     \end{eqnarray}
    #     and total flow conservation equation
    #     \begin{eqnarray}
    #     \label{eq:2p-c-ff-mb-m}
    #     \pd{m_m}{t} + \deld\left(\vec f_m - \ten{a}_m \grad \phi_m \right) + r_m &=& 0
    #     \end{eqnarray}

    #     \begin{table}[ht]
    #     \caption{Coefficient definitions for compressible two-phase fractional flow formulation flow, \eqn{2p-c-ff-mb-w} and \eqn{2p-c-ff-mb-m}
    #     \label{tab:2p-c-ff-coef-1}
    #     }
    #     \begin{tabular}{cc}
    #     \hline
    #     Var. & Def. \\
    #     \hline
    #     $u_w$ & $S_w $ \\
    #     $u_n$ & $\psi_w $ \\
    #     $\phi_w$ & $\psi_c$ \\
    #     $\phi_m$ & $\psi_w$  \\
    #     $m_w$ & $\theta_s \rho_{w}S_w$  \\
    #     $m_m$ & $\theta_s \rho_{w}S_w + \theta_s\rho_{n}(1-S_w)$  \\
    #     $\vec f_w^{\dagger}$ & $\gvec{\sigma}_t F_w - \ten{K}_s\lambda_wF_n\left(b\rho_n - \rho_w\right)\vec g_u$  \\
    #     $\vec f_m^{\dagger}$ &$-\ten{K}_s\lambda_tF_n\grad \psi_c + \ten{K}_s\vec g\lambda_t\left[\rho_w + F_n\left(b\rho_n - \rho_w\right)\right]$\\
    #     $\ten a_{w,w}^{\dagger}$ & $-\lambda_wF_n \ten{K}_{s}$ \\
    #     $\ten a_{m,m}^{\dagger}$ & $\lambda_t \ten{K}_{s}$ \\
    #     \hline
    #     $F_i $ & $\lambda_i/\lambda_t $, $i=w,n$  \\
    #     $\grad \psi_c $ & $\od{\psi_c}{S_w}\grad S_w $  \\
    #     $\gvec {\sigma}_t$ & $\gvec \sigma_w + \gvec \sigma_n$\\
    #     \hline
    #     \multicolumn{2}{l}
    #     {$\dagger$ : $\rho_i \equiv 1$, $\lambda_i= k_{r,i}/\hat{\mu}_{i}$ for slight compressibility assumption}
    #     \end{tabular}
    #     \end{table}

    #  Here :math:`S_i` is the saturation for each phase, :math:`\varrho_{i}` is the
    #  density and :math:`\varrho_{i,0}` is a reference value

    # (normalized) mass flux for each phase is

    # .. math::

    #    \vec \sigma_i = - \ten{a}_{i}\grad \phi_i,    i=w,n

    # and

    # .. math::

    #    \psi_{i} = p_{i}/|\vec g|\rho_{w,0}
    #    b        = \varrho_{n,0}/\varrho_{w,0}
    #    \hat{mu}_{i} = \mu_i/\mu_{w}
    #    \lambda_i= \rho k_{r,i}/\hat{\mu}_{i}
    #    \lambda_t= \lambda_n + \lambda_w

    # for the pressure of each phase, p_i, and we have the capillary pressure relation

    # .. math::

    #    \psi_{c} = \psi_{n} - \psi_{w}

    # The dependent variables are :math:`S_w`, and :math:`\psi_w`

    # Note :math:`S_n = 1-S_w`

    # needs to be implemented

    # .. math::

    #   r_m = r_m(\vec x,t)

    # """
#    TODO:
#
#      Figure out if really need to evaluate potential interpolation points
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_slightCompressible_split_sd_pressure_het_matType
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_compressibleN_split_sd_pressure_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_slightCompressible_split_sd_pressure_het_matType = staticmethod(twophaseDarcy_slightCompressible_split_sd_pressure_het_matType)
    twophaseDarcy_compressibleN_split_sd_pressure_het_matType = staticmethod(twophaseDarcy_compressibleN_split_sd_pressure_het_matType)
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=TwophaseDarcyFlow_base.default_density_w_parameters,
                 density_n_parameters=TwophaseDarcyFlow_base.default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nSatModel=1,
                 compressibilityFlag=2,
                 diagonal_conductivity=True,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcy_split_pressure_base.__init__(self,
                                                   nd=nd,
                                                   dimensionless_gravity=dimensionless_gravity,
                                                   density_w=density_w,
                                                   density_n=density_n,
                                                   viscosity_w=viscosity_w,
                                                   viscosity_n=viscosity_n,
                                                   density_w_parameters=density_w_parameters,
                                                   density_n_parameters=density_n_parameters,
                                                   psk_model=psk_model,
                                                   nMaterialTypes=nMaterialTypes,
                                                   nSatModel=nSatModel,
                                                   diagonal_conductivity=diagonal_conductivity,
                                                   swConstant=swConstant,
                                                   capillaryDiffusionScaling=capillaryDiffusionScaling)

        self.compressibilityFlag=compressibilityFlag
        variableNames=['psi_w']
        #
        mass      = {0:{0:'nonlinear'}}
        advection = {0:{0:'nonlinear'}}
        hamiltonian={}
        diffusion = {0:{0: {0:'nonlinear'}}}
        potential = {0:{0: 'u'}}
        reaction  = {0:{0:'linear'}}
        if self.compressibilityFlag == 2:
            advection = {0:{0:'linear'}}
            diffusion = {0:{0: {0:'linear'}}}

        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}


        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_s_w.shape:
            materialTypes = self.materialTypes_q
            s_w = self.q_s_w
            grad_psic = self.q_grad_psic
            c['psi_n']= numpy.copy(self.q_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.q[('vol_frac',0)]
            vol_frac_n = self.q[('vol_frac',1)]
        elif c[('u',0)].shape == self.ebqe_s_w.shape:
            materialTypes = self.materialTypes_ebqe
            s_w = self.ebqe_s_w
            grad_psic = self.ebqe_grad_psic
            c['psi_n']= numpy.copy(self.ebqe_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.ebqe[('vol_frac',0)]
            vol_frac_n = self.ebqe[('vol_frac',1)]
        elif c[('u',0)].shape == self.ip_s_w.shape:
            c['psi_n']= numpy.copy(self.ip_psic)
            c['psi_n'] += c[('u',0)]
            #mwf hack 04/03/09 skip
            return
            materialTypes = self.materialTypes_ip
            s_w = self.ip_s_w
            grad_psic = self.ip_grad_psic
            vol_frac_w = self.ip[('vol_frac',0)]
            vol_frac_n = self.ip[('vol_frac',1)]
        else:
            assert c[('u',0)].shape == self.ebq_s_w.shape
            materialTypes = self.materialTypes_ebq
            s_w = self.ebq_s_w
            grad_psic = self.ebq_grad_psic
            c['psi_n']= numpy.copy(self.ebq_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.ebq[('vol_frac',0)]
            vol_frac_n = self.ebq[('vol_frac',1)]
        assert self.rwork_psk is not None
        if self.compressibilityFlag == 2:
            self.twophaseDarcy_slightCompressible_split_sd_pressure_het_matType(self.psk_types[self.psk_model],
                                                                                  self.density_types[self.density_w_model],
                                                                                  self.density_types[self.density_n_model],
                                                                                  self.sdInfo[(0,0)][0],
                                                                                  self.sdInfo[(0,0)][1],
                                                                                  materialTypes,
                                                                                  self.muw,
                                                                                  self.mun,
                                                                                  self.omega_types,
                                                                                  self.Ksw_types,
                                                                                  self.b,
                                                                                  self.capillaryDiffusionScaling,
                                                                                  self.rwork_psk,self.iwork_psk,
                                                                                  self.rwork_psk_tolerances,
                                                                                  self.rwork_density_w,
                                                                                  self.rwork_density_n,
                                                                                  self.g[:self.nd],
                                                                                  s_w,
                                                                                  c[('u',0)],
                                                                                  c['psi_n'],
                                                                                  grad_psic,
                                                                                  c[('m',0)],
                                                                                  c[('dm',0,0)],
                                                                                  c[('f',0)],
                                                                                  c[('a',0,0)])

        else:
            self.twophaseDarcy_compressibleN_split_sd_pressure_het_matType(self.psk_types[self.psk_model],
                                                                           self.density_types[self.density_w_model],
                                                                           self.density_types[self.density_n_model],
                                                                           self.sdInfo[(0,0)][0],
                                                                           self.sdInfo[(0,0)][1],
                                                                           materialTypes,
                                                                           self.muw,
                                                                           self.mun,
                                                                           self.omega_types,
                                                                           self.Ksw_types,
                                                                           self.b,
                                                                           self.capillaryDiffusionScaling,
                                                                           self.rwork_psk,self.iwork_psk,
                                                                           self.rwork_psk_tolerances,
                                                                           self.rwork_density_w,
                                                                           self.rwork_density_n,
                                                                           self.g[:self.nd],
                                                                           s_w,
                                                                           c[('u',0)],
                                                                           c['psi_n'],
                                                                           grad_psic,
                                                                           c[('m',0)],
                                                                           c[('dm',0,0)],
                                                                           c[('f',0)],
                                                                           c[('a',0,0)])

        self.twophaseDarcy_vol_frac(materialTypes,
                                    self.omega_types,
                                    s_w,
                                    vol_frac_w,
                                    vol_frac_n)
        #mwf debug
        if (numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any()):
            import pdb
            pdb.set_trace()
        #mwf debug
        #import pdb
        #pdb.set_trace()
        if c[('u',0)].shape == self.q_s_w.shape:
            c['Ks']=c[('a',0,0)][:,:,0]


class TwophaseDarcy_compressible_split_saturation(TwophaseDarcy_split_saturation_base):
    """Aqueous phase mass conservation equation (saturation equation) in a
    compressible fractional flow formulation

    Right now, the options are

    compressibility for the non-wetting phase (compressibleN) :
    compressiblityFlag=1
    
    compressibility for both phases but with the slight compressiblity
    assumption (spatial density gradients are negligible)
    compressiblityFlag=2

    """
    # Saturation equation

    # .. math::

    #     \begin{eqnarray}
    #     \label{eq:2p-c-ff-mb-w}
    #     \pd{m_w}{t} + \deld\left(\vec f_w - \ten{a}_{w}\grad \phi_w \right) + r_w &=& 0
    #     \end{eqnarray}

    # and total flow conservation equation

    # .. math::
    #     \begin{eqnarray}
    #     \label{eq:2p-c-ff-mb-m}
    #     \pd{m_m}{t} + \deld\left(\vec f_m - \ten{a}_m \grad \phi_m \right) + r_m &=& 0
    #     \end{eqnarray}

    #     \begin{table}[ht]
    #     \caption{Coefficient definitions for compressible two-phase fractional flow formulation flow, \eqn{2p-c-ff-mb-w} and \eqn{2p-c-ff-mb-m}
    #     \label{tab:2p-c-ff-coef-1}
    #     }
    #     \begin{tabular}{cc}
    #     \hline
    #     Var. & Def. \\
    #     \hline
    #     $u_w$ & $S_w $ \\
    #     $u_n$ & $\psi_w $ \\
    #     $\phi_w$ & $\psi_c$ \\
    #     $\phi_m$ & $\psi_w$  \\
    #     $m_w$ & $\theta_s \rho_{w}S_w$  \\
    #     $m_m$ & $\theta_s \rho_{w}S_w + \theta_s\rho_{n}(1-S_w)$  \\
    #     $\vec f_w^{\dagger}$ & $\gvec{\sigma}_t F_w - \ten{K}_s\lambda_wF_n\left(b\rho_n - \rho_w\right)\vec g_u$  \\
    #     $\vec f_m^{\dagger}$ &$-\ten{K}_s\lambda_tF_n\grad \psi_c + \ten{K}_s\vec g\lambda_t\left[\rho_w + F_n\left(b\rho_n - \rho_w\right)\right]$\\
    #     $\ten a_{w,w}^{\dagger}$ & $-\lambda_wF_n \ten{K}_{s}$ \\
    #     $\ten a_{m,m}^{\dagger}$ & $\lambda_t \ten{K}_{s}$ \\
    #     \hline
    #     $F_i $ & $\lambda_i/\lambda_t $, $i=w,n$  \\
    #     $\grad \psi_c $ & $\od{\psi_c}{S_w}\grad S_w $  \\
    #     $\gvec {\sigma}_t$ & $\gvec \sigma_w + \gvec \sigma_n$\\
    #     \hline
    #     \multicolumn{2}{l}
    #     {$\dagger$ : $\rho_i \equiv 1$, $\lambda_i= k_{r,i}/\hat{\mu}_{i}$ for slight compressibility assumption}
    #     \end{tabular}
    #     \end{table}

    #  Here :math:`S_i` is the saturation for each phase, :math:`\varrho_{i}` is the density and
    #  :math:`\varrho_{i,0}` is a reference value

    #  (normalized) mass flux for each phase is

    # .. math::

    #    \vec \sigma_i = - \ten{a}_{i}\grad \phi_i,    i=w,n

    # and

    # .. math::

    #    \psi_{i} = p_{i}/|\vec g|\rho_{w,0}
    #    b        = \varrho_{n,0}/\varrho_{w,0}
    #    \hat{mu}_{i} = \mu_i/\mu_{w}
    #    \lambda_i= \rho k_{r,i}/\hat{\mu}_{i}
    #    \lambda_t= \lambda_n + \lambda_w

    # for the pressure of each phase, p_i, and we have the capillary pressure relation

    # .. math::

    #    \psi_{c} = \psi_{n} - \psi_{w}

    # The dependent variables are :math:`S_w`, and :math:`\psi_w`

    # Note :math:`S_n = 1-S_w`

    # needs to be implemented

    # .. math::

    #     r_m = r_m(\vec x,t)

    # """
#    TODO:
#
#      Figure out if really need to evaluate potential interpolation points
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_slightCompressible_split_sd_saturation_het_matType
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_compressibleN_split_sd_saturation_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_slightCompressible_split_sd_saturation_het_matType = staticmethod(twophaseDarcy_slightCompressible_split_sd_saturation_het_matType)
    twophaseDarcy_compressibleN_split_sd_saturation_het_matType = staticmethod(twophaseDarcy_compressibleN_split_sd_saturation_het_matType)
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=TwophaseDarcyFlow_base.default_density_w_parameters,
                 density_n_parameters=TwophaseDarcyFlow_base.default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nPressModel=0,
                 compressibilityFlag=2,
                 diagonal_conductivity=True,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcy_split_saturation_base.__init__(self,
                                                     nd=nd,
                                                     dimensionless_gravity=dimensionless_gravity,
                                                     density_w=density_w,
                                                     density_n=density_n,
                                                     viscosity_w=viscosity_w,
                                                     viscosity_n=viscosity_n,
                                                     density_w_parameters=density_w_parameters,
                                                     density_n_parameters=density_n_parameters,
                                                     psk_model=psk_model,
                                                     nMaterialTypes=nMaterialTypes,
                                                     nPressModel=nPressModel,
                                                     diagonal_conductivity=diagonal_conductivity,
                                                     qScalarConstant=qScalarConstant,
                                                     capillaryDiffusionScaling=capillaryDiffusionScaling,
                                                     advectionScaling=advectionScaling)

        self.compressibilityFlag=compressibilityFlag
        variableNames=['s_w']
        mass      = {0:{0:'linear'}}
        advection = {0:{0:'nonlinear'}}
        hamiltonian={}
        diffusion = {0:{0:{0:'nonlinear'}}}
        potential = {0:{0: 'nonlinear'}}
        reaction  = {0:{0:'linear'}}

        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}


        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)
    def evaluate(self,t,c):
        if c[('f',0)].shape == self.q_q_t.shape:
            materialTypes = self.materialTypes_q
            q_t = self.q_q_t
            psiw = self.q_psiw
            vol_frac_w = self.q[('vol_frac',0)]
            vol_frac_n = self.q[('vol_frac',1)]
        elif c[('f',0)].shape == self.ebqe_q_t.shape:
            materialTypes = self.materialTypes_ebqe
            q_t = self.ebqe_q_t
            psiw = self.ebqe_psiw
            vol_frac_w = self.ebqe[('vol_frac',0)]
            vol_frac_n = self.ebqe[('vol_frac',1)]
        elif c[('f',0)].shape == self.ip_q_t.shape:
            materialTypes = self.materialTypes_ip
            q_t = self.ip_q_t
            psiw = self.ip_psiw
            vol_frac_w = self.ip[('vol_frac',0)]
            vol_frac_n = self.ip[('vol_frac',1)]
        else:
            assert c[('f',0)].shape == self.ebq_q_t.shape
            materialTypes = self.materialTypes_ebq
            q_t = self.ebq_q_t
            psiw = self.ebq_psiw
            vol_frac_w = self.ebq[('vol_frac',0)]
            vol_frac_n = self.ebq[('vol_frac',1)]
        assert self.rwork_psk is not None
        #mwf debug
        #import pdb
        #pdb.set_trace()
        if self.compressibilityFlag == 2:
            self.twophaseDarcy_slightCompressible_split_sd_saturation_het_matType(self.psk_types[self.psk_model],
                                                                                  self.density_types[self.density_w_model],
                                                                                  self.density_types[self.density_n_model],
                                                                                  self.sdInfo[(0,0)][0],
                                                                                  self.sdInfo[(0,0)][1],
                                                                                  materialTypes,
                                                                                  self.muw,
                                                                                  self.mun,
                                                                                  self.omega_types,
                                                                                  self.Ksw_types,
                                                                                  self.b,
                                                                                  self.capillaryDiffusionScaling,
                                                                                  self.advectionScaling,
                                                                                  self.rwork_psk,self.iwork_psk,
                                                                                  self.rwork_psk_tolerances,
                                                                                  self.rwork_density_w,
                                                                                  self.rwork_density_n,
                                                                                  self.g[:self.nd],
                                                                                  q_t,
                                                                                  psiw,
                                                                                  c[('u',0)],
                                                                                  c[('m',0)],
                                                                                  c[('dm',0,0)],
                                                                                  c[('phi',0)],
                                                                                  c[('dphi',0,0)],
                                                                                  c[('f',0)],
                                                                                  c[('df',0,0)],
                                                                                  c[('a',0,0)],
                                                                                  c[('da',0,0,0)])
        else:
            self.twophaseDarcy_compressibleN_split_sd_saturation_het_matType(self.psk_types[self.psk_model],
                                                                             self.density_types[self.density_w_model],
                                                                             self.density_types[self.density_n_model],
                                                                             self.sdInfo[(0,0)][0],
                                                                             self.sdInfo[(0,0)][1],
                                                                             materialTypes,
                                                                             self.muw,
                                                                             self.mun,
                                                                             self.omega_types,
                                                                             self.Ksw_types,
                                                                             self.b,
                                                                             self.capillaryDiffusionScaling,
                                                                             self.advectionScaling,
                                                                             self.rwork_psk,self.iwork_psk,
                                                                             self.rwork_psk_tolerances,
                                                                             self.rwork_density_w,
                                                                             self.rwork_density_n,
                                                                             self.g[:self.nd],
                                                                             q_t,
                                                                             psiw,
                                                                             c[('u',0)],
                                                                             c[('m',0)],
                                                                             c[('dm',0,0)],
                                                                             c[('phi',0)],
                                                                             c[('dphi',0,0)],
                                                                             c[('f',0)],
                                                                             c[('df',0,0)],
                                                                             c[('a',0,0)],
                                                                             c[('da',0,0,0)])

        self.twophaseDarcy_vol_frac(materialTypes,
                                    self.omega_types,
                                    c[('u',0)],
                                    vol_frac_w,
                                    vol_frac_n)
        #material types fi
        #mwf debug
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('phi',0)]).any() or
            numpy.isnan(c[('dphi',0,0)]).any() or
            numpy.isnan(c[('df',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()

###########
class TwophaseDarcy_split_pp_pressure_base(TwophaseDarcyFlow_base):
    """Base class for 'pressure' or total flow conservation equation in
    fractional flow formulations. This

    The primary functionality of the base class is to handle
    synchronization with a 'saturation' model to get the saturation,
    :math:`S_w`, and capillary pressure (head), :math:`\\psi_c`, variables

    This version would allow for capillary pressure to be unknown for
    saturation equation

    """
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=TwophaseDarcyFlow_base.default_density_w_parameters,
                 density_n_parameters=TwophaseDarcyFlow_base.default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nSatModel=1,
                 diagonal_conductivity=True,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcyFlow_base.__init__(self,
                                        nd=nd,
                                        dimensionless_gravity=dimensionless_gravity,
                                        density_w=density_w,
                                        density_n=density_n,
                                        viscosity_w=viscosity_w,
                                        viscosity_n=viscosity_n,
                                        density_w_parameters=density_w_parameters,
                                        density_n_parameters=density_n_parameters,
                                        psk_model=psk_model,
                                        nMaterialTypes=nMaterialTypes,
                                        diagonal_conductivity=diagonal_conductivity)

        self.nc=1
        self.nd=nd
        #
        self.nSatModel=nSatModel
        #for debugging
        self.swConstant=swConstant
        self.capillaryDiffusionScaling=capillaryDiffusionScaling
    def attachModels(self,modelList):
        if self.nSatModel is None:
            print('Warning TwophaseDarcy_split_pressure_base nSatModel is None returning in attachModels')
            return
        #not ideal, but need a way to force nonlinear potential to be evaluated in saturation model
        modelList[self.nSatModel].calculateElementCoefficients()
        self.q_s_w   = modelList[self.nSatModel].q['sw']
        self.ebqe_s_w = modelList[self.nSatModel].ebqe['sw']
        if 'sw' in modelList[self.nSatModel].ebq:
            self.ebq_s_w = modelList[self.nSatModel].ebq['sw']
        self.q_grad_psic   = modelList[self.nSatModel].q[('grad(phi)',0)]
        self.ebqe_grad_psic = modelList[self.nSatModel].ebqe[('grad(phi)',0)]
        if ('grad(phi)',0) in modelList[self.nSatModel].ebq:
            self.ebq_grad_psic = modelList[self.nSatModel].ebq[('grad(phi)',0)]
        self.q_psic   = modelList[self.nSatModel].q[('phi',0)]
        self.ebqe_psic= modelList[self.nSatModel].ebqe[('phi',0)]
        if ('phi',0) in modelList[self.nSatModel].ebq:
            self.ebq_psic = modelList[self.nSatModel].ebq[('phi',0)]
        assert ('phi',0) in modelList[self.nSatModel].phi_ip
        assert self.ip_psic.shape ==  modelList[self.nSatModel].phi_ip[('phi',0)].shape
        self.ip_psic = modelList[self.nSatModel].phi_ip[('phi',0)]

    def initializeElementQuadrature(self,t,cq):
        TwophaseDarcyFlow_base.initializeElementQuadrature(self,t,cq)
        #set up dummy values in case we're not running the other model
        self.q_s_w   = numpy.zeros(cq[('u',0)].shape,'d')
        self.q_s_w[:] = self.swConstant
        for i in range(len(self.q_s_w.flat)//2,len(self.q_s_w.flat)):
            self.q_s_w.flat[i] = 1.0e-4
        self.q_grad_psic   = numpy.zeros(cq[('f',0)].shape,'d')
        self.q_psic        = numpy.zeros(cq[('u',0)].shape,'d')
        cq['psi_n'] = numpy.zeros(cq[('u',0)].shape,'d')
        cq[('dpsi_n',0)] = numpy.ones(cq[('u',0)].shape,'d')

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseDarcyFlow_base.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        #set up dummy values in case we're not running the other model
        self.ebq_s_w = numpy.zeros(cebq[('u',0)].shape,'d')
        self.ebq_s_w.fill(self.swConstant)
        self.ebq_grad_psic = numpy.zeros(cebq[('f',0)].shape,'d')
        self.ebq_psic = numpy.zeros(cebq[('u',0)].shape,'d')
        if ('u',0) in cebq:
            cebq['psi_n'] = numpy.zeros(cebq[('u',0)].shape,'d')
            cebq[('dpsi_n',0)] = numpy.ones(cebq[('u',0)].shape,'d')
        if ('u',0) in cebq_global:
            cebq_global['psi_n'] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            cebq_global[('dpsi_n',0)] = numpy.ones(cebq_global[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseDarcyFlow_base.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        #set up dummy values in case we're not running the other model
        self.ebqe_s_w = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.ebqe_s_w[:]=self.swConstant
        self.ebqe_grad_psic = numpy.zeros(cebqe[('f',0)].shape,'d')
        self.ebqe_psic = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe['psi_n'] = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe[('dpsi_n',0)] = numpy.ones(cebqe[('u',0)].shape,'d')
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        TwophaseDarcyFlow_base.initializeGeneralizedInterpolationPointQuadrature(self,t,cip)
        self.ip_grad_psic = numpy.zeros(cip[('f',0)].shape,'d')
        self.ip_psic = numpy.zeros(cip[('u',0)].shape,'d')
        cip['psi_n'] = numpy.zeros(cip[('u',0)].shape,'d')
        cip[('dpsi_n',0)] = numpy.ones(cip[('u',0)].shape,'d')

#
class TwophaseDarcy_split_pp_saturation_base(TwophaseDarcyFlow_base):
    """Base class for aqueous phase mass conservation equation
    (saturation equation) in a fractional flow formulation.

    The primary responsibility of the base class is to handle
    synchronization with the 'pressure' equation to get the total flow
    velocity variable, :math:`q_t`, and aqueous phase pressure head, :math:`psi_w`

    """
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 density_w_parameters=TwophaseDarcyFlow_base.default_density_w_parameters,
                 density_n_parameters=TwophaseDarcyFlow_base.default_density_n_parameters,
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nPressModel=0,
                 diagonal_conductivity=True,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcyFlow_base.__init__(self,
                                        nd=nd,
                                        dimensionless_gravity=dimensionless_gravity,
                                        density_w=density_w,
                                        density_n=density_n,
                                        viscosity_w=viscosity_w,
                                        viscosity_n=viscosity_n,
                                        density_w_parameters=density_w_parameters,
                                        density_n_parameters=density_n_parameters,
                                        psk_model=psk_model,
                                        nMaterialTypes=nMaterialTypes,
                                        diagonal_conductivity=diagonal_conductivity)

        self.nc=1
        self.nd=nd
        self.nPressModel=nPressModel
        #for debugging
        self.qScalarConstant=1.0
        self.capillaryDiffusionScaling=capillaryDiffusionScaling
        self.advectionScaling=advectionScaling
    def attachModels(self,modelList):
        if self.nPressModel is None:
            print('Warning TwophaseDarcy_split_saturation_base nPressModel is None returning in attachModels')
            return
        self.flowModel = modelList[self.nPressModel]
        #
        self.q_q_t    = modelList[self.nPressModel].q[('velocity',0)]
        self.ebqe_q_t  = modelList[self.nPressModel].ebqe[('velocity',0)]
        if ('velocity',0) in modelList[self.nPressModel].ebq:
            self.ebq_q_t  = modelList[self.nPressModel].ebq[('velocity',0)]
        #do we really need other model values for q_t in potential calculation?
        assert self.ip_psiw.shape == modelList[self.nPressModel].phi_ip[('u',0)].shape
        self.ip_psiw = modelList[self.nPressModel].phi_ip[('u',0)]
        self.q_psiw    = modelList[self.nPressModel].q[('u',0)]
        self.ebqe_psiw = modelList[self.nPressModel].ebqe[('u',0)]
        if ('u',0) in modelList[self.nPressModel].ebq:
            self.ebq_psiw = modelList[self.nPressModel].ebq[('u',0)]
    def initializeElementQuadrature(self,t,cq):
        TwophaseDarcyFlow_base.initializeElementQuadrature(self,t,cq)
        #set up dummy values in case we're not running the other model
        self.q_q_t   = numpy.zeros(cq[('f',0)].shape,'d')
        self.q_q_t[:] = self.qScalarConstant
        self.q_psiw   = numpy.ones(cq[('u',0)].shape,'d')
        cq['sw'] = numpy.zeros(cq[('u',0)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseDarcyFlow_base.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        #set up dummy values in case we're not running the other model
        self.ebq_q_t = numpy.zeros(cebq[('f',0)].shape,'d')
        self.ebq_q_t[:] = self.qScalarConstant
        self.ebq_psiw = numpy.ones(cebq[('u',0)].shape,'d')
        if ('u',0) in cebq:
            cebq['sw'] = numpy.zeros(cebq[('u',0)].shape,'d')
        if ('u',0) in cebq_global:
            cebq_global['sw'] = numpy.zeros(cebq_global[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseDarcyFlow_base.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        #set up dummy values in case we're not running the other model
        self.ebqe_q_t = numpy.zeros(cebqe[('f',0)].shape,'d')
        self.ebqe_q_t[:] = self.qScalarConstant
        self.ebqe_psiw = numpy.ones(cebqe[('u',0)].shape,'d')
        cebqe['sw'] = numpy.zeros(cebqe[('u',0)].shape,'d')
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        TwophaseDarcyFlow_base.initializeGeneralizedInterpolationPointQuadrature(self,t,cip)
        #set up dummy values in case we're not running the other model
        self.ip_q_t = numpy.zeros(cip[('f',0)].shape,'d')
        self.ip_q_t[:] = self.qScalarConstant
        self.ip_psiw = numpy.ones(cip[('u',0)].shape,'d')
        cip['sw'] = numpy.zeros(cip[('u',0)].shape,'d')

class TwophaseDarcy_incompressible_split_pp_pressure(TwophaseDarcy_split_pp_pressure_base):
    """Total flow conservation equation in an incompressible fractional
    flow formulation

    """
    # Saturation equation

    # .. math::
    #     \begin{eqnarray}
    #     \label{eq:2p-ff-mb-w}
    #     \pd{m_w}{t} + \deld\left(\vec f_w - \ten{a}_{w}\grad \phi_w \right) + r_w &=& 0
    #     \end{eqnarray}
    #     and total flow conservation equation
    #     \begin{eqnarray}
    #     \label{eq:2p-ff-mb-m}
    #     \deld\left(\vec f_m - \ten{a}_m \grad \phi_m \right) + r_m &=& 0
    #     \end{eqnarray}

    #     \begin{table}[ht]
    #     \caption{Coefficient definitions for Two-phase flow, \eqn{2p-ff-mb-w} and \eqn{2p-ff-mb-m}
    #     \label{tab:2p-ff-coef-1}
    #     }
    #     \begin{tabular}{cc}
    #     \hline
    #     Var. & Def. \\
    #     \hline
    #     $u_w$ & $S_w $ \\
    #     $u_n$ & $\psi_w $ \\
    #     $\phi_w$ & $\psi_c$ \\
    #     $\phi_m$ & $\psi_w$  \\
    #     $m_w$ & $\theta_s \rho_{w}S_w$  \\
    #     $\vec f_w$ & $\gvec{\sigma}_t F_w - \ten{K}_s\lambda_wF_n\left(b\rho_n - \rho_w\right)\vec g_u$  \\
    #     $\vec f_m$ &$-\ten{K}_s\lambda_tF_n\grad \psi_c + \ten{K}_s\vec g\lambda_t\left[\rho_w + F_n\left(b\rho_n - \rho_w\right)\right]$\\
    #     $\ten a_{w,w}$ & $-\lambda_wF_n \ten{K}_{s}$ \\
    #     $\ten a_{m,m}$ & $\lambda_t \ten{K}_{s}$ \\
    #     \hline
    #     $F_i $ & $\lambda_i/\lambda_t $, $i=w,n$  \\
    #     $\grad \psi_c $ & $\od{\psi_c}{S_w}\grad S_w $  \\
    #     $\gvec {\sigma}_t$ & $\gvec \sigma_w + \gvec \sigma_n$\\
    #     \hline
    #     \end{tabular}
    #     \end{table}

    #  Here :math:`S_i` is the saturation for each phase,
    #  :math:`\varrho_{i}` is the density and :math:`\varrho_{i,0}` is a
    #  reference value

    # (normalized) mass flux for each phase is

    # .. math::

    #    \vec \sigma_i = - \ten{a}_{i}\grad \phi_i,    i=w,n

    # and

    # .. math::

    #    \psi_{i} = p_{i}/|\vec g|\rho_{w,0}
    #    b        = \varrho_{n,0}/\varrho_{w,0}
    #    \hat{mu}_{i} = \mu_i/\mu_{w}

    # for the pressure of each phase, p_i, and we have the capillary pressure relation

    # .. math::

    #    \psi_{c} = \psi_{n} - \psi_{w}

    # The dependent variables are :math:`S_w`, and :math:`\psi_w`

    # Note :math:`S_n = 1-S_w`

    # needs to be implemented

    # .. math::

    #     r_m = r_m(\vec x,t)

    # """
#    TODO:
#
#      Figure out if really need to evaluate potential interpolation points
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_incompressible_split_sd_pressure_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_incompressible_split_sd_pressure_het_matType = staticmethod(twophaseDarcy_incompressible_split_sd_pressure_het_matType)
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nSatModel=1,
                 diagonal_conductivity=True,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcy_split_pp_pressure_base.__init__(self,
                                                      nd=nd,
                                                      dimensionless_gravity=dimensionless_gravity,
                                                      density_w=density_w,
                                                      density_n=density_n,
                                                      viscosity_w=viscosity_w,
                                                      viscosity_n=viscosity_n,
                                                      psk_model=psk_model,
                                                      nMaterialTypes=nMaterialTypes,
                                                      nSatModel=nSatModel,
                                                      diagonal_conductivity=diagonal_conductivity,
                                                      swConstant=swConstant,
                                                      capillaryDiffusionScaling=capillaryDiffusionScaling)

        variableNames=['psi_w']
        #these are only nonlinear for compressible flow
        mass      = {}
        advection = {0:{0:'linear'}}
        hamiltonian={}
        diffusion = {0:{0: {0:'linear'}}}
        potential = {0:{0: 'u'}}
        reaction  = {0:{0:'linear'}}

        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}


        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_s_w.shape:
            materialTypes = self.materialTypes_q
            s_w = self.q_s_w
            grad_psic = self.q_grad_psic

            c['psi_n']= numpy.copy(self.q_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.q[('vol_frac',0)]
            vol_frac_n = self.q[('vol_frac',1)]
            #mwf debug
            #import pdb
            #pdb.set_trace()
            #for eN in range(c['x'].shape[0]):
            #    for k in range(c['x'].shape[1]):
            #        if (1.25 <= c['x'][eN,k,0] and c['x'][eN,k,0] <= 2.25 and
            #            1.25 <= c['x'][eN,k,1] and c['x'][eN,k,1] <= 1.5):
                        #mwf major hack to test initialization
                        #grad_psic[eN,k,0] = 0.0
                        #grad_psic[eN,k,1] = 1.0
            #
            #            print "split press eval inside lens eN=%s k=%s x=%s grad_psiw= %s grad_psic= %s s_w= %s " % (eN,k,c['x'][eN,k],c[('grad(u)',0)][eN,k],grad_psic[eN,k],s_w[eN,k])
            #        else:
            #            print "split press eval outside lens eN=%s k=%s x=%s grad_psiw= %s grad_psic= %s s_w= %s " % (eN,k,c['x'][eN,k],c[('grad(u)',0)][eN,k],grad_psic[eN,k],s_w[eN,k])

            #import pdb
            #pdb.set_trace()
        elif c[('u',0)].shape == self.ebqe_s_w.shape:
            materialTypes = self.materialTypes_ebqe
            s_w = self.ebqe_s_w
            grad_psic = self.ebqe_grad_psic

            c['psi_n']= numpy.copy(self.ebqe_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.ebqe[('vol_frac',0)]
            vol_frac_n = self.ebqe[('vol_frac',1)]
        elif c[('u',0)].shape == self.ip_psic.shape:
            c['psi_n']= numpy.copy(self.ip_psic)
            c['psi_n'] += c[('u',0)]
            return
        else:
            assert c[('u',0)].shape == self.ebq_s_w.shape
            materialTypes = self.materialTypes_ebq
            s_w = self.ebq_s_w
            grad_psic = self.ebq_grad_psic

            c['psi_n']= numpy.copy(self.ebq_psic)
            c['psi_n'] += c[('u',0)]
            vol_frac_w = self.ebq[('vol_frac',0)]
            vol_frac_n = self.ebq[('vol_frac',1)]
        assert self.rwork_psk is not None
        #mwf debug
        import pdb
        pdb.set_trace()

        self.twophaseDarcy_incompressible_split_sd_pressure_het_matType(self.psk_types[self.psk_model],
                                                                        self.sdInfo[(0,0)][0],
                                                                        self.sdInfo[(0,0)][1],
                                                                        materialTypes,
                                                                        self.muw,
                                                                        self.mun,
                                                                        self.omega_types,
                                                                        self.Ksw_types,
                                                                        self.b,
                                                                        self.capillaryDiffusionScaling,
                                                                        self.rwork_psk,self.iwork_psk,
                                                                        self.rwork_psk_tolerances,
                                                                        self.rwork_density_w,
                                                                        self.rwork_density_n,
                                                                        self.g[:self.nd],
                                                                        s_w,
                                                                        grad_psic,
                                                                        c[('f',0)],
                                                                        c[('a',0,0)])

        self.twophaseDarcy_vol_frac(materialTypes,
                                    self.omega_types,
                                    s_w,
                                    vol_frac_w,
                                    vol_frac_n)
        #mwf debug
        if (numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any()):
            import pdb
            pdb.set_trace()
        #mwf debug
        #import pdb
        #pdb.set_trace()
        if c[('u',0)].shape == self.q_s_w.shape:
            c['Ks']=c[('a',0,0)][:,:,0]

#
class TwophaseDarcy_incompressible_split_pp_saturation(TwophaseDarcy_split_pp_saturation_base):
    """Aqueous phase mass conservation equation (saturation equation) in
    an incompressible fractional flow formulation

    """
    # Saturation equation

    # .. math::

    #     \begin{eqnarray}
    #     \label{eq:2p-ff-mb-w}
    #     \pd{m_w}{t} + \deld\left(\vec f_w - \ten{a}_{w}\grad \phi_w \right) + r_w &=& 0
    #     \end{eqnarray}
    #     and total flow conservation equation
    #     \begin{eqnarray}
    #     \label{eq:2p-ff-mb-m}
    #     \deld\left(\vec f_m - \ten{a}_m \grad \phi_m \right) + r_m &=& 0
    #     \end{eqnarray}

    #     \begin{table}[ht]
    #     \caption{Coefficient definitions for Two-phase flow, \eqn{2p-ff-mb-w} and \eqn{2p-ff-mb-m}
    #     \label{tab:2p-ff-coef-1}
    #     }
    #     \begin{tabular}{cc}
    #     \hline
    #     Var. & Def. \\
    #     \hline
    #     $u_w$ & $S_w $ \\
    #     $u_n$ & $\psi_w $ \\
    #     $\phi_w$ & $\psi_c$ \\
    #     $\phi_m$ & $\psi_w$  \\
    #     $m_w$ & $\theta_s \rho_{w}S_w$  \\
    #     $\vec f_w$ & $\gvec{\sigma}_t F_w - \ten{K}_s\lambda_wF_n\left(b\rho_n - \rho_w\right)\vec g_u$  \\
    #     $\vec f_m$ &$-\ten{K}_s\lambda_tF_n\grad \psi_c + \ten{K}_s\vec g\lambda_t\left[\rho_w + F_n\left(b\rho_n - \rho_w\right)\right]$\\
    #     $\ten a_{w,w}$ & $-\lambda_wF_n \ten{K}_{s}$ \\
    #     $\ten a_{m,m}$ & $\lambda_t \ten{K}_{s}$ \\
    #     \hline
    #     $F_i $ & $\lambda_i/\lambda_t $, $i=w,n$  \\
    #     $\grad \psi_c $ & $\od{\psi_c}{S_w}\grad S_w $  \\
    #     $\gvec {\sigma}_t$ & $\gvec \sigma_w + \gvec \sigma_n$\\
    #     \hline
    #     \end{tabular}
    #     \end{table}

    #  Here :math:`S_i` is the saturation for each phase, :math:`\varrho_{i}` is the density and
    #  :math:`\varrho_{i,0}` is a reference value

    # (normalized) mass flux for each phase is

    # .. math::

    #    \vec \sigma_i = - \ten{a}_{i}\grad \phi_i,    i=w,n

    # and

    # .. math::

    #    \psi_{i} = p_{i}/|\vec g|\rho_{w,0}
    #    b        = \varrho_{n,0}/\varrho_{w,0}
    #    \hat{mu}_{i} = \mu_i/\mu_{w}

    # for the pressure of each phase, p_i, and we have the capillary pressure relation

    # .. math::

    #    \psi_{c} = \psi_{n} - \psi_{w}

    # The dependent variables are :math:`S_w`, and :math:`\psi_w`

    # Note :math:`S_n = 1-S_w`

    # needs to be implemented

    # .. math::
    
    #     r_m = r_m(\vec x,t)
    
    # """
#    TODO:
#
#      Figure out if really need to evaluate potential interpolation points
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_incompressible_split_pp_sd_saturation_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_incompressible_split_pp_sd_saturation_het_matType = staticmethod(twophaseDarcy_incompressible_split_pp_sd_saturation_het_matType)
    def __init__(self,
                 nd=1,
                 dimensionless_gravity=[-1.0],
                 density_w=998.2, #Water kg/m^3
                 density_n=1.205, #Air   kg/m^3
                 viscosity_w=8.9e-4, #Water kg/m s
                 viscosity_n=1.81e-5,#Air kg/ m s
                 psk_model='VGM',
                 nMaterialTypes=1,
                 nPressModel=0,
                 diagonal_conductivity=True,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcy_split_pp_saturation_base.__init__(self,
                                                        nd=nd,
                                                        dimensionless_gravity=dimensionless_gravity,
                                                        density_w=density_w,
                                                        density_n=density_n,
                                                        viscosity_w=viscosity_w,
                                                        viscosity_n=viscosity_n,
                                                        psk_model=psk_model,
                                                        nMaterialTypes=nMaterialTypes,
                                                        nPressModel=nPressModel,
                                                        diagonal_conductivity=diagonal_conductivity,
                                                        qScalarConstant=qScalarConstant,
                                                        capillaryDiffusionScaling=capillaryDiffusionScaling,
                                                        advectionScaling=advectionScaling)

        variableNames=['psi_c']
        mass      = {0:{0:'nonlinear'}}
        advection = {0:{0:'nonlinear'}}
        hamiltonian={}
        diffusion = {0:{0:{0:'nonlinear'}}}
        potential = {0:{0: 'nonlinear'}}
        reaction  = {0:{0:'linear'}}

        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd+1,dtype='i'),
                                             numpy.arange(self.nd,dtype='i'))}

        else: #full
            sparseDiffusionTensors = {(0,0):(numpy.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             numpy.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}


        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)
    def evaluate(self,t,c):
        if c[('f',0)].shape == self.q_q_t.shape:
            materialTypes = self.materialTypes_q
            q_t = self.q_q_t
            psiw = self.q_psiw
            vol_frac_w = self.q[('vol_frac',0)]
            vol_frac_n = self.q[('vol_frac',1)]
        elif c[('f',0)].shape == self.ebqe_q_t.shape:
            materialTypes = self.materialTypes_ebqe
            q_t = self.ebqe_q_t
            psiw = self.ebqe_psiw
            vol_frac_w = self.ebqe[('vol_frac',0)]
            vol_frac_n = self.ebqe[('vol_frac',1)]
        elif c[('f',0)].shape == self.ip_q_t.shape:
            materialTypes = self.materialTypes_ip
            q_t = self.ip_q_t
            psiw = self.ip_psiw
            vol_frac_w = self.ip[('vol_frac',0)]
            vol_frac_n = self.ip[('vol_frac',1)]
        else:
            assert c[('f',0)].shape == self.ebq_q_t.shape
            materialTypes = self.materialTypes_ebq
            q_t = self.ebq_q_t
            psiw = self.ebq_psiw
            vol_frac_w = self.ebq[('vol_frac',0)]
            vol_frac_n = self.ebq[('vol_frac',1)]
        assert self.rwork_psk is not None
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.twophaseDarcy_incompressible_split_pp_sd_saturation_het_matType(self.psk_types[self.psk_model],
                                                                             self.sdInfo[(0,0)][0],
                                                                             self.sdInfo[(0,0)][1],
                                                                             materialTypes,
                                                                             self.muw,
                                                                             self.mun,
                                                                             self.omega_types,
                                                                             self.Ksw_types,
                                                                             self.b,
                                                                             self.capillaryDiffusionScaling,
                                                                             self.advectionScaling,
                                                                             self.rwork_psk,self.iwork_psk,
                                                                             self.rwork_psk_tolerances,
                                                                             self.rwork_density_w,
                                                                             self.rwork_density_n,
                                                                             self.g[:self.nd],
                                                                             q_t,
                                                                             c['sw'],
                                                                             c[('u',0)],
                                                                             c[('m',0)],
                                                                             c[('dm',0,0)],
                                                                             c[('phi',0)],
                                                                             c[('dphi',0,0)],
                                                                             c[('f',0)],
                                                                             c[('df',0,0)],
                                                                             c[('a',0,0)],
                                                                             c[('da',0,0,0)])

        self.twophaseDarcy_vol_frac(materialTypes,
                                    self.omega_types,
                                    c['sw'],
                                    vol_frac_w,
                                    vol_frac_n)

########################################
#begin classes for specific psk models
########################################
class IncompressibleFractionalFlowPressureMualemVanGenuchten(TwophaseDarcy_incompressible_split_pressure):
    """Total flow equation coefficients for incompressible flow assuming
    Mualem-Van Genuchten psk's

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 nSatModel=1,
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcy_incompressible_split_pressure.__init__(self,
                                                             nd=nd,
                                                             dimensionless_gravity=dimensionless_gravity,
                                                             density_w=density_w,
                                                             density_n=density_n,
                                                             viscosity_w=viscosity_w,
                                                             viscosity_n=viscosity_n,
                                                             psk_model='VGM',
                                                             nMaterialTypes=len(thetaR_types),
                                                             nSatModel=nSatModel,
                                                             diagonal_conductivity=diagonal_conductivity,
                                                             swConstant=swConstant,
                                                             capillaryDiffusionScaling=capillaryDiffusionScaling)

        for input in [vgm_n_types,vgm_alpha_types,thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.psk_tolerances['VGM']['eps_small']=vgm_small_eps
        self.psk_tolerances['VGM']['ns_del']   =vgm_ns_del
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types,
                              vg_alpha_types=vgm_alpha_types,
                              vg_m_types=vgm_m_types)

#
class IncompressibleFractionalFlowSaturationMualemVanGenuchten(TwophaseDarcy_incompressible_split_saturation):
    """Saturation equation coefficients for incompressible flow assuming
    Mualem-Van Genuchten psk's

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 nPressModel=1,
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcy_incompressible_split_saturation.__init__(self,
                                                               nd=nd,
                                                               dimensionless_gravity=dimensionless_gravity,
                                                               density_w=density_w,
                                                               density_n=density_n,
                                                               viscosity_w=viscosity_w,
                                                               viscosity_n=viscosity_n,
                                                               psk_model='VGM',
                                                               nMaterialTypes=len(thetaR_types),
                                                               nPressModel=nPressModel,
                                                               diagonal_conductivity=diagonal_conductivity,
                                                               qScalarConstant=qScalarConstant,
                                                               capillaryDiffusionScaling=capillaryDiffusionScaling,
                                                               advectionScaling=advectionScaling)

        for input in [vgm_n_types,vgm_alpha_types,thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.psk_tolerances['VGM']['eps_small']=vgm_small_eps
        self.psk_tolerances['VGM']['ns_del']   =vgm_ns_del
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types,
                              vg_alpha_types=vgm_alpha_types,
                              vg_m_types=vgm_m_types)


class IncompressibleFractionalFlowSaturationMualemVanGenuchtenSplitAdvDiff(IncompressibleFractionalFlowSaturationMualemVanGenuchten):
    """Saturation equation coefficients for incompressible flow assuming
    Mualem-Van Genuchten psk's and splitting of advection and
    capillary diffusion terms

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 nPressModel=1,
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0,
                 satModelIndex_me=1,
                 satModelIndex_other=2):
        IncompressibleFractionalFlowSaturationMualemVanGenuchten.__init__(self,
                                                                          nd,
                                                                          Ksw_types,
                                                                          vgm_n_types,
                                                                          vgm_alpha_types,
                                                                          thetaR_types,
                                                                          thetaSR_types,
                                                                          dimensionless_gravity,
                                                                          density_w,
                                                                          density_n,
                                                                          viscosity_w,
                                                                          viscosity_n,
                                                                          nPressModel=nPressModel,
                                                                          diagonal_conductivity=diagonal_conductivity,
                                                                          vgm_small_eps=vgm_small_eps,
                                                                          vgm_ns_del=vgm_ns_del,
                                                                          qScalarConstant=qScalarConstant,
                                                                          capillaryDiffusionScaling=capillaryDiffusionScaling,
                                                                          advectionScaling=advectionScaling)
        self.satModelIndex_other=satModelIndex_other
        self.satModelIndex_me   =satModelIndex_me
        self.satModel_me = None
        self.satModel_other=None
        self.variableNames=['s_w_%s' % self.satModelIndex_me]
        self.u_ip = {}
    def attachModels(self,modelList):
        IncompressibleFractionalFlowSaturationMualemVanGenuchten.attachModels(self,modelList)
        if 0 <= self.satModelIndex_me and self.satModelIndex_me < len(modelList):
            self.satModel_me = modelList[self.satModelIndex_me]
            #interpolation points on physical mesh for this models fem space
            self.u_ip['x']    = self.satModel_me.u[0].femSpace.updateInterpolationPoints()
            #value of other models solution at this models interpolation points
            self.u_ip[('u_other',0)]= numpy.zeros((self.satModel_me.mesh.nElements_global,
                                                   self.satModel_me.u[0].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints),
                                                  'd')
        if 0 <= self.satModelIndex_other and self.satModelIndex_other < len(modelList):
            self.satModel_other = modelList[self.satModelIndex_other]
            #other model's saturation trial functions at this models interpolation points
            self.u_ip[('v_other',0)] = numpy.zeros((self.satModel_me.mesh.nElements_global,
                                                    self.satModel_me.u[0].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                                                    self.satModel_other.nDOF_trial_element[0]),
                                                   'd')
            self.satModel_other.u[0].femSpace.getBasisValues(self.satModel_me.u[0].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                             self.u_ip[('v_other',0)])

    def preStep(self,t,firstStep=False):
        if self.satModel_other is not None:# and self.satModelIndex_me != 1:#mwf hack
            #todo tLast is getting messed up
            #tLastSave =self.satModel_me.timeIntegration.tLast
            #todo need to do make sure mass conserved, handle projection from cg to dg correctly
            #todo ExplicitRK is getting messed up here, going twice as fast
            logEvent("Incomp.FracFlowSatAdvDiff preStep t= %s model %s setting its solution from model %s " % (t,self.satModel_me,self.satModel_other),level=2)
            #if self.satModelIndex_me != 1:#mwf hack
            self.satModel_other.u[0].getValues(self.u_ip[('v_other',0)],
                                               self.u_ip[('u_other',0)])
            #mwf hack to test mass conservation
            #for eN in range(self.u_ip[('u_other',0)].shape[0]):
            #    uAvg = numpy.sum(self.u_ip[('u_other',0)][eN])/float(self.u_ip[('u_other',0)].shape[1])
            #    self.u_ip[('u_other',0)][eN].fill(uAvg)
            #import pdb
            #pdb.set_trace()
            self.satModel_me.u[0].projectFromInterpolationConditions(self.u_ip[('u_other',0)])
            self.satModel_me.calculateCoefficients()
            self.satModel_me.calculateElementResidual()
            #this doesn't work either?
            self.satModel_me.timeIntegration.initializeTimeHistory(resetFromDOF=True)
            #only effects lagging of stabilization and shock-capturing
            self.satModel_me.updateTimeHistory(t,resetFromDOF=True)

            #now do again because of subgrid error lagging
            #\todo modify subgrid error lagging so this won't be necessary
            self.satModel_me.calculateCoefficients()
            self.satModel_me.calculateElementResidual()
            #self.satModel_me.timeIntegration.updateTimeHistory(resetFromDOF=True)
            self.satModel_me.timeIntegration.resetTimeHistory(resetFromDOF=True)
            self.satModel_me.updateTimeHistory(t,resetFromDOF=True)
            #mwf hack tLast is getting messed up
            #self.satModel_me.timeIntegration.tLast = tLastSave
            copyInstructions = {'copy_uList':True,
                                'uList_model':self.satModelIndex_other}
            copyInstructions = {'reset_uList':True}
            #print "advDiff satModel_me= %s after \n u_me= %s " % (self.satModelIndex_me,
            #                                                      self.satModel_me.u[0].dof)
            #

            return copyInstructions
        else:
            return {}

#
class CompressibleFractionalFlowPressureMualemVanGenuchten(TwophaseDarcy_compressible_split_pressure):
    """Total flow equation coefficients for slight compressible flow
    assuming Mualem-Van Genuchten psk's

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 density_w_parameters,
                 density_n_parameters,
                 nSatModel=1,
                 compressibilityFlag=2,#2 slight-compressiblity for N,W, 1 -- compressiblity in N only
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcy_compressible_split_pressure.__init__(self,
                                                           nd=nd,
                                                           dimensionless_gravity=dimensionless_gravity,
                                                           density_w=density_w,
                                                           density_n=density_n,
                                                           viscosity_w=viscosity_w,
                                                           viscosity_n=viscosity_n,
                                                           density_w_parameters=density_w_parameters,
                                                           density_n_parameters=density_n_parameters,
                                                           psk_model='VGM',
                                                           nMaterialTypes=len(thetaR_types),
                                                           nSatModel=nSatModel,
                                                           compressibilityFlag=compressibilityFlag,
                                                           diagonal_conductivity=diagonal_conductivity,
                                                           swConstant=swConstant,
                                                           capillaryDiffusionScaling=capillaryDiffusionScaling)

        for input in [vgm_n_types,vgm_alpha_types,thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.psk_tolerances['VGM']['eps_small']=vgm_small_eps
        self.psk_tolerances['VGM']['ns_del']   =vgm_ns_del
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types,
                              vg_alpha_types=vgm_alpha_types,
                              vg_m_types=vgm_m_types)


#
class CompressibleFractionalFlowSaturationMualemVanGenuchten(TwophaseDarcy_compressible_split_saturation):
    """Saturation equation coefficients for slightly compressible flow
    assuming Mualem-Van Genuchten psk's

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 density_w_parameters,
                 density_n_parameters,
                 nPressModel=1,
                 compressibilityFlag=2,#2 slight-compressiblity for N,W, 1 -- compressiblity in N only
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcy_compressible_split_saturation.__init__(self,
                                                             nd=nd,
                                                             dimensionless_gravity=dimensionless_gravity,
                                                             density_w=density_w,
                                                             density_n=density_n,
                                                             viscosity_w=viscosity_w,
                                                             viscosity_n=viscosity_n,
                                                             density_w_parameters=density_w_parameters,
                                                             density_n_parameters=density_n_parameters,
                                                             psk_model='VGM',
                                                             nMaterialTypes=len(thetaR_types),
                                                             nPressModel=nPressModel,
                                                             compressibilityFlag=compressibilityFlag,
                                                             diagonal_conductivity=diagonal_conductivity,
                                                             qScalarConstant=qScalarConstant,
                                                             capillaryDiffusionScaling=capillaryDiffusionScaling,
                                                             advectionScaling=advectionScaling)

        for input in [vgm_n_types,vgm_alpha_types,thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.psk_tolerances['VGM']['eps_small']=vgm_small_eps
        self.psk_tolerances['VGM']['ns_del']   =vgm_ns_del
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types,
                              vg_alpha_types=vgm_alpha_types,
                              vg_m_types=vgm_m_types)

#
class IncompressibleFractionalFlowPressureSimplePSKs(TwophaseDarcy_incompressible_split_pressure):
    """Total flow equation coefficients for incompressible flow assuming
    'simp' quadratic rel-perm, linear capillary pressure psk relations

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 nSatModel=1,
                 diagonal_conductivity=True,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcy_incompressible_split_pressure.__init__(self,
                                                             nd=nd,
                                                             dimensionless_gravity=dimensionless_gravity,
                                                             density_w=density_w,
                                                             density_n=density_n,
                                                             viscosity_w=viscosity_w,
                                                             viscosity_n=viscosity_n,
                                                             psk_model='simp',
                                                             nMaterialTypes=len(thetaR_types),
                                                             nSatModel=nSatModel,
                                                             diagonal_conductivity=diagonal_conductivity,
                                                             swConstant=swConstant,
                                                             capillaryDiffusionScaling=capillaryDiffusionScaling)

        for input in [thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types)


#
class IncompressibleFractionalFlowSaturationSimplePSKs(TwophaseDarcy_incompressible_split_saturation):
    """Saturation equation coefficients for incompressible flow assuming
    'simp' quadratic rel-perm, linear capillary pressure psk relations

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 nPressModel=1,
                 diagonal_conductivity=True,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcy_incompressible_split_saturation.__init__(self,
                                                               nd=nd,
                                                               dimensionless_gravity=dimensionless_gravity,
                                                               density_w=density_w,
                                                               density_n=density_n,
                                                               viscosity_w=viscosity_w,
                                                               viscosity_n=viscosity_n,
                                                               psk_model='simp',
                                                               nMaterialTypes=len(thetaR_types),
                                                               nPressModel=nPressModel,
                                                               diagonal_conductivity=diagonal_conductivity,
                                                               qScalarConstant=qScalarConstant,
                                                               capillaryDiffusionScaling=capillaryDiffusionScaling,
                                                               advectionScaling=advectionScaling)

        for input in [thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types)




class PressurePressureIncompressibleFractionalFlowPressureMualemVanGenuchten(TwophaseDarcy_incompressible_split_pp_pressure):
    """Total flow equation coefficients for incompressible flow assuming
    Mualem-Van Genuchten psk's

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 nSatModel=1,
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 #for debugging
                 swConstant=1.0,
                 capillaryDiffusionScaling=1.0):
        TwophaseDarcy_incompressible_split_pp_pressure.__init__(self,
                                                                nd=nd,
                                                                dimensionless_gravity=dimensionless_gravity,
                                                                density_w=density_w,
                                                                density_n=density_n,
                                                                viscosity_w=viscosity_w,
                                                                viscosity_n=viscosity_n,
                                                                psk_model='VGM',
                                                                nMaterialTypes=len(thetaR_types),
                                                                nSatModel=nSatModel,
                                                                diagonal_conductivity=diagonal_conductivity,
                                                                swConstant=swConstant,
                                                                capillaryDiffusionScaling=capillaryDiffusionScaling)

        for input in [vgm_n_types,vgm_alpha_types,thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.psk_tolerances['VGM']['eps_small']=vgm_small_eps
        self.psk_tolerances['VGM']['ns_del']   =vgm_ns_del
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types,
                              vg_alpha_types=vgm_alpha_types,
                              vg_m_types=vgm_m_types)

#
class PressurePressureIncompressibleFractionalFlowSaturationMualemVanGenuchten(TwophaseDarcy_incompressible_split_pp_saturation):
    """Saturation equation coefficients for incompressible flow assuming
    Mualem-Van Genuchten psk's

    """
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 dimensionless_gravity,
                 density_w,
                 density_n,
                 viscosity_w,
                 viscosity_n,
                 nPressModel=1,
                 diagonal_conductivity=True,
                 vgm_small_eps=1.0e-16,
                 vgm_ns_del=1.0e-8,
                 #for debugging
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 advectionScaling=1.0):
        TwophaseDarcy_incompressible_split_pp_saturation.__init__(self,
                                                                  nd=nd,
                                                                  dimensionless_gravity=dimensionless_gravity,
                                                                  density_w=density_w,
                                                                  density_n=density_n,
                                                                  viscosity_w=viscosity_w,
                                                                  viscosity_n=viscosity_n,
                                                                  psk_model='VGM',
                                                                  nMaterialTypes=len(thetaR_types),
                                                                  nPressModel=nPressModel,
                                                                  diagonal_conductivity=diagonal_conductivity,
                                                                  qScalarConstant=qScalarConstant,
                                                                  capillaryDiffusionScaling=capillaryDiffusionScaling,
                                                                  advectionScaling=advectionScaling)

        for input in [vgm_n_types,vgm_alpha_types,thetaR_types,thetaSR_types]:
            assert len(input)==self.nMaterialTypes

        vgm_m_types    = 1.0-1.0/vgm_n_types
        thetaS_types   = thetaSR_types + thetaR_types
        Sw_max_types   = numpy.ones((self.nMaterialTypes,),'d')
        Sw_min_types   = thetaR_types/thetaS_types
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.psk_tolerances['VGM']['eps_small']=vgm_small_eps
        self.psk_tolerances['VGM']['ns_del']   =vgm_ns_del
        self.setMaterialTypes(Ksw_types=Ksw_types,
                              omega_types=thetaS_types,
                              Sw_max_types=Sw_max_types,
                              Sw_min_types=Sw_min_types,
                              vg_alpha_types=vgm_alpha_types,
                              vg_m_types=vgm_m_types)

########################################
#Single-phase species transport
########################################
class GroundwaterTransportCoefficients(TC_base):
    from proteus.ctransportCoefficients import groundwaterTransportCoefficientsEvaluate_hetMat
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    groundwaterTransportCoefficientsEvaluate_hetMat = staticmethod(groundwaterTransportCoefficientsEvaluate_hetMat)
    """ groundwater advection-dispersion equation with coefficients
    varying by material type and variable velocity """
    def __init__(self,nc=1,nd=2,
                 omega_types=numpy.array([0.3]),
                 alpha_L_types=numpy.array([1.0]),
                 alpha_T_types=numpy.array([0.1]),
                 d=numpy.array([1.3e-9]),
                 meModelId = 0,
                 flowModelId = None,
                 velocityFunctions = None):

        self.alpha_L_types = alpha_L_types
        self.alpha_T_types = alpha_T_types
        self.omega_types   = omega_types
        self.d = d
        self.nd = nd
        self.flowModelId = flowModelId
        self.flowModel   = None
        self.meModelId   = meModelId

        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction  = {}
        hamiltonian = {}
        for i in range(nc):
            diffusion[i] = {i : {i:'constant'}}
            advection[i] = {i : {i:'linear'}}
            mass[i] = {i : {i:'linear'}}
            reaction[i] = {i : {i:'constant'}}
            potential[i] = {i : 'u'}
        #end i
        sparseDiffusionTensors = {}
        for ci in range(nc):
            sparseDiffusionTensors[(ci,ci)]=(numpy.arange(start=0,stop=nd**2+1,step=nd,dtype='i'),
                                             numpy.array([list(range(nd)) for row in range(nd)],dtype='i'))
        names = ['C_%s' % ci for ci in range(nc)]
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)

        self.q={}; self.ebqe={}; self.ebq ={}; self.ebq_global = {}
        self.velocityFunctions = velocityFunctions
    def initializeMesh(self,mesh):
        self.elementMaterialTypes,self.exteriorElementBoundaryTypes,self.elementBoundaryTypes = BlockHeterogeneousCoefficients(mesh).initializeMaterialTypes()
        self.elementBoundariesArray = mesh.elementBoundariesArray
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
        for ci in range(self.nc):
            self.q[('velocity',ci)] = numpy.zeros((cq['x'].shape[0],cq['x'].shape[1],self.nd),'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for ebN_local in range(self.ebq_shape[1]):
            self.materialTypes_ebq[:,ebN_local] = self.elementMaterialTypes
        for ci in range(self.nc):
            self.ebq[('velocity',ci)] = numpy.zeros((cebq['x'].shape[0],cebq['x'].shape[1],cebq['x'].shape[2],self.nd),'d')
            self.ebq_global[('velocity',ci)] = numpy.zeros((cebq_global['x'].shape[0],cebq_global['x'].shape[1],self.nd),'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = self.exteriorElementBoundaryTypes
        self.ebqe_shape = cebqe[('u',0)].shape
        for ci in range(self.nc):
            self.ebqe[('velocity',ci)] = numpy.zeros((cebqe['x'].shape[0],cebqe['x'].shape[1],self.nd),'d')
    def attachModels(self,modelList):
        self.vt = modelList[self.meModelId]
        if self.flowModelId is not None:
            self.flowModel = modelList[self.flowModelId]
            for ci in range(self.nc):
                self.q[('velocity',ci)]    = self.flowModel.q[('velocity',ci)]
                self.ebqe[('velocity',ci)] = self.flowModel.ebqe[('velocity',ci)]
                if ('velocity',ci) in self.flowModel.ebq:
                    self.ebq[('velocity',ci)] = self.flowModel.ebq[('velocity',ci)]
                if ('velocity',ci) in self.flowModel.ebq_global:
                    self.ebq_global[('velocity',ci)] = self.flowModel.ebq_global[('velocity',ci)]

    def evaluateVelocity(self,t,c):
        if self.velocityFunctions is not None:
            for ci in range(self.nc):
                if len(c['x'].shape) == 3:
                    for i in range(c['x'].shape[0]):
                        for j in range(c['x'].shape[1]):
                            c[('velocity',ci)][i,j,:] = self.velocityFunctions[ci](c['x'][i,j],t)
                elif len(c['x'].shape) == 4:
                    for i in range(c['x'].shape[0]):
                        for j in range(c['x'].shape[1]):
                            for k in range(c['x'].shape[2]):
                                c[('velocity',ci)][i,j,k,:] = self.velocityFunctions[ci](c['x'][i,j,k],t)

    def evaluate(self,t,c):
#        TODO
#          evaluate velocity is currently setting ebqe when c=q but need to make sure this is done
#          before evaluate is called with c=ebqe
        #mwf debug
        #import pdb
        #pdb.set_trace()
        if self.velocityFunctions is not None:
            self.evaluateVelocity(t,c)
        #
        for ci in range(self.nc):
            if self.q[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.q[('velocity',ci)]
                materialTypes = self.materialTypes_q
            elif self.ebqe[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.ebqe[('velocity',ci)]
                materialTypes = self.materialTypes_ebqe
            elif self.ebq[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.ebq[('velocity',ci)]
                materialTypes = self.materialTypes_ebq
            else:
                print(c[('df',ci,ci)].shape)
                print("no v---------------------")
                raise RuntimeError

            self.groundwaterTransportCoefficientsEvaluate_hetMat(self.d[ci],
                                                                 materialTypes,
                                                                 self.omega_types,
                                                                 self.alpha_L_types,
                                                                 self.alpha_T_types,
                                                                 v,
                                                                 c[('u',ci)],
                                                                 c[('m',ci)],
                                                                 c[('dm',ci,ci)],
                                                                 c[('f',ci)],
                                                                 c[('df',ci,ci)],
                                                                 c[('a',ci,ci)])

########################################
#multiphase species transport
########################################
class MultiphaseGroundwaterTransportCoefficients(TC_base):
    from proteus.ctransportCoefficients import variablySaturatedGroundwaterTransportCoefficientsEvaluate_hetMat
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    variablySaturatedGroundwaterTransportCoefficientsEvaluate_hetMat = staticmethod(variablySaturatedGroundwaterTransportCoefficientsEvaluate_hetMat)
    """ groundwater advection-dispersion equation with coefficients
    varying by material type and variable velocity """
    def __init__(self,nc=1,nd=2,
                 omega_types=numpy.array([0.3]),
                 alpha_L_types=numpy.array([1.0]),
                 alpha_T_types=numpy.array([0.1]),
                 d=numpy.array([1.3e-9]),
                 meModelId = 0,
                 flowModelId = None,
                 velocityFunctions = None):

        self.alpha_L_types = alpha_L_types
        self.alpha_T_types = alpha_T_types
        self.omega_types   = omega_types
        self.d = d
        self.nd = nd
        self.flowModelId = flowModelId
        self.flowModel   = None
        self.meModelId   = meModelId

        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction  = {}
        hamiltonian = {}
        for i in range(nc):
            diffusion[i] = {i : {i:'constant'}}
            advection[i] = {i : {i:'linear'}}
            mass[i] = {i : {i:'linear'}}
            reaction[i] = {i : {i:'constant'}}
            potential[i] = {i : 'u'}
        #end i
        sparseDiffusionTensors = {}
        for ci in range(nc):
            sparseDiffusionTensors[(ci,ci)]=(numpy.arange(start=0,stop=nd**2+1,step=nd,dtype='i'),
                                             numpy.array([list(range(nd)) for row in range(nd)],dtype='i'))
        names = ['C_%s' % ci for ci in range(nc)]
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)

        self.q={}; self.ebqe={}; self.ebq ={}; self.ebq_global = {}
        self.velocityFunctions = velocityFunctions
    def initializeMesh(self,mesh):
        self.elementMaterialTypes,self.exteriorElementBoundaryTypes,self.elementBoundaryTypes = BlockHeterogeneousCoefficients(mesh).initializeMaterialTypes()
        self.elementBoundariesArray = mesh.elementBoundariesArray
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
        for ci in range(self.nc):
            self.q[('velocity',ci)] = numpy.zeros((cq['x'].shape[0],cq['x'].shape[1],self.nd),'d')
            self.q[('vol_frac',ci)] = numpy.ones((cq['x'].shape[0],cq['x'].shape[1]),'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for ebN_local in range(self.ebq_shape[1]):
            self.materialTypes_ebq[:,ebN_local] = self.elementMaterialTypes
        for ci in range(self.nc):
            self.ebq[('velocity',ci)] = numpy.zeros((cebq['x'].shape[0],cebq['x'].shape[1],cebq['x'].shape[2],self.nd),'d')
            self.ebq_global[('velocity',ci)] = numpy.zeros((cebq_global['x'].shape[0],cebq_global['x'].shape[1],self.nd),'d')
            self.ebq[('vol_frac',ci)] = numpy.ones((cebq['x'].shape[0],cebq['x'].shape[1],cebq['x'].shape[2]),'d')
            self.ebq_global[('vol_frac',ci)] = numpy.ones((cebq_global['x'].shape[0],cebq_global['x'].shape[1]),'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = self.exteriorElementBoundaryTypes
        self.ebqe_shape = cebqe[('u',0)].shape
        for ci in range(self.nc):
            self.ebqe[('velocity',ci)] = numpy.zeros((cebqe['x'].shape[0],cebqe['x'].shape[1],self.nd),'d')
            self.ebqe[('vol_frac',ci)] = numpy.ones((cebqe['x'].shape[0],cebqe['x'].shape[1]),'d')
    def attachModels(self,modelList):
        self.vt = modelList[self.meModelId]
        if self.flowModelId is not None:
            self.flowModel = modelList[self.flowModelId]
            for ci in range(self.nc):
                self.q[('velocity',ci)]    = self.flowModel.q[('velocity',ci)]
                self.ebqe[('velocity',ci)] = self.flowModel.ebqe[('velocity',ci)]
                if ('velocity',ci) in self.flowModel.ebq:
                    self.ebq[('velocity',ci)] = self.flowModel.ebq[('velocity',ci)]
                if ('velocity',ci) in self.flowModel.ebq_global:
                    self.ebq_global[('velocity',ci)] = self.flowModel.ebq_global[('velocity',ci)]
                self.q[('vol_frac',ci)]    = self.flowModel.coefficients.q[('vol_frac',ci)]
                self.ebqe[('vol_frac',ci)] = self.flowModel.coefficients.ebqe[('vol_frac',ci)]
    def evaluateVelocity(self,t,c):
        if self.velocityFunctions is not None:
            for ci in range(self.nc):
                if len(c['x'].shape) == 3:
                    for i in range(c['x'].shape[0]):
                        for j in range(c['x'].shape[1]):
                            c[('velocity',ci)][i,j,:] = self.velocityFunctions[ci](c['x'][i,j],t)
                elif len(c['x'].shape) == 4:
                    for i in range(c['x'].shape[0]):
                        for j in range(c['x'].shape[1]):
                            for k in range(c['x'].shape[2]):
                                c[('velocity',ci)][i,j,k,:] = self.velocityFunctions[ci](c['x'][i,j,k],t)

    def evaluate(self,t,c):
#        TODO
#          evaluate velocity is currently setting ebqe when c=q but need to make sure this is done
#          before evaluate is called with c=ebqe
        #mwf debug
        #import pdb
        #pdb.set_trace()
        if self.velocityFunctions is not None:
            self.evaluateVelocity(t,c)
        #
        for ci in range(self.nc):
            if self.q[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.q[('velocity',ci)]
                materialTypes = self.materialTypes_q
                vol_frac = self.q[('vol_frac',ci)]
            elif self.ebqe[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.ebqe[('velocity',ci)]
                materialTypes = self.materialTypes_ebqe
                vol_frac = self.ebqe[('vol_frac',ci)]
            elif self.ebq[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.ebq[('velocity',ci)]
                materialTypes = self.materialTypes_ebq
                vol_frac = self.ebq[('vol_frac',ci)]
            else:
                print(c[('df',ci,ci)].shape)
                print("no v---------------------")
                raise RuntimeError

            self.variablySaturatedGroundwaterTransportCoefficientsEvaluate_hetMat(self.d[ci],
                                                                                  materialTypes,
                                                                                  vol_frac,
                                                                                  self.alpha_L_types,
                                                                                  self.alpha_T_types,
                                                                                  v,
                                                                                  c[('u',ci)],
                                                                                  c[('m',ci)],
                                                                                  c[('dm',ci,ci)],
                                                                                  c[('f',ci)],
                                                                                  c[('df',ci,ci)],
                                                                                  c[('a',ci,ci)])


class VariablySaturatedGroundwaterEnergyTransportCoefficients(MultiphaseGroundwaterTransportCoefficients):
    from proteus.ctransportCoefficients import variablySaturatedGroundwaterEnergyTransportCoefficientsEvaluate_hetMat
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    variablySaturatedGroundwaterEnergyTransportCoefficientsEvaluate_hetMat = staticmethod(variablySaturatedGroundwaterEnergyTransportCoefficientsEvaluate_hetMat)
    """ groundwater heat equation with coefficients varying by material
    type and variable velocity """
    def __init__(self,nc=1,nd=2,
                 density_w=998.2, #kg/m^3
                 density_n=1.205, #kg/m^3
                 specificHeat_w=0.04882, #W d / (kg K), 0.001172 W hr/g-C
                 specificHeat_n=0.01446, #W d / (kg K), 0.000347 W hr/g-C need to convert
                 omega_types=numpy.array([0.3]),
                 alpha_L_types=numpy.array([1.0]),
                 alpha_T_types=numpy.array([0.1]),
                 d=numpy.array([1.3e-9]),
                 density_s_types=numpy.array([2.73*998.2]), #kg/m^3
                 specificHeat_s_types=numpy.array([0.004167]), #W d / (kg K) 1.0e-4, W hr/g-C need to convert
                 lambda_sat_types=numpy.array([0.58]),#W/m-K need to convert
                 lambda_dry_types=numpy.array([0.3]),#W/m-K need to convert
                 lambda_ani_types=numpy.array([1.0,1.0,1.0]),
                 meModelId = 0,
                 flowModelId = None,
                 velocityFunctions = None):
        MultiphaseGroundwaterTransportCoefficients.__init__(self,nc=nc,nd=nd,
                                                            omega_types=omega_types,
                                                            alpha_L_types=alpha_L_types,
                                                            alpha_T_types=alpha_T_types,
                                                            d=d,
                                                            meModelId = meModelId,
                                                            flowModelId = flowModelId,
                                                            velocityFunctions = velocityFunctions)
        self.density_w = density_w
        self.density_n = density_n
        self.specificHeat_w = specificHeat_w
        self.specificHeat_n = specificHeat_n
        self.density_s_types = density_s_types
        self.specificHeat_s_types = specificHeat_s_types
        self.lambda_sat_types = lambda_sat_types
        self.lambda_dry_types = lambda_dry_types
        self.lambda_ani_types = lambda_ani_types
        self.nd = nd

    def evaluate(self,t,c):
#        TODO
#          evaluate velocity is currently setting ebqe when c=q but need to make sure this is done
#          before evaluate is called with c=ebqe
        #mwf debug
        #import pdb
        #pdb.set_trace()
        if self.velocityFunctions is not None:
            self.evaluateVelocity(t,c)
        #
        for ci in range(self.nc):
            if self.q[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.q[('velocity',ci)]
                materialTypes = self.materialTypes_q
                vol_frac = self.q[('vol_frac',ci)]
            elif self.ebqe[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.ebqe[('velocity',ci)]
                materialTypes = self.materialTypes_ebqe
                vol_frac = self.ebqe[('vol_frac',ci)]
            elif self.ebq[('velocity',ci)].shape == c[('df',ci,ci)].shape:
                v = self.ebq[('velocity',ci)]
                materialTypes = self.materialTypes_ebq
                vol_frac = self.ebq[('vol_frac',ci)]
            else:
                print(c[('df',ci,ci)].shape)
                print("no v---------------------")
                raise RuntimeError

            self.variablySaturatedGroundwaterEnergyTransportCoefficientsEvaluate_hetMat(self.density_w,
                                                                                        self.density_n,
                                                                                        self.specificHeat_w,
                                                                                        self.specificHeat_n,
                                                                                        materialTypes,
                                                                                        vol_frac,
                                                                                        self.omega_types,
                                                                                        self.alpha_L_types,
                                                                                        self.alpha_T_types,
                                                                                        self.density_s_types,
                                                                                        self.specificHeat_s_types,
                                                                                        self.lambda_sat_types,
                                                                                        self.lambda_dry_types,
                                                                                        self.lambda_ani_types,
                                                                                        v,
                                                                                        c[('u',ci)],
                                                                                        c[('m',ci)],
                                                                                        c[('dm',ci,ci)],
                                                                                        c[('f',ci)],
                                                                                        c[('df',ci,ci)],
                                                                                        c[('a',ci,ci)])

