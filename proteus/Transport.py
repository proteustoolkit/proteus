r"""
This module contains methods for solving

.. math:: m_t^i + \nabla \cdot (f^i - \sum_k a^{ik} \nabla \phi^k) + H^i(\nabla  u) + r^i = 0 \quad i=0,\ldots,nc-1

The solution is the vector of components :math:`u=u_0,\ldots,u_{nc-1}`, and
the nonlinear coefficients are :math:`m^i,f^i,a^{ik},\phi^k, H^i` and :math:`r^i`.

.. inheritance-diagram:: proteus.Transport
   :parts: 1
"""
from math import *
from .EGeometry import *
from .LinearAlgebraTools import *
from .LinearSolvers import *
from .MeshTools import *
from . import MeshTools
from .FemTools import *
from . import Quadrature
from .TimeIntegration import *
from .NonlinearSolvers import *
from . import cfemIntegrals
from .csparsity import CSR_INDEX_DTYPE
from .TransportCoefficients import *
from . import NumericalFlux
from . import cnumericalFlux
from . import Comm
from . import csparsity
from .Profiling import logEvent
from .petsc_optional import PETSc as p4pyPETSc, HAVE_PETSC4PY
from . import superluWrappers
import numpy as np
from proteus import cmeshTools

class StorageSet(set):
    def __init__(self,initializer=[],shape=(0,),storageType='d'):
        set.__init__(self,initializer)
        self.shape = shape
        self.storageType = storageType
    def allocate(self,storageDict):
        for k in self:
            storageDict[k] = numpy.zeros(self.shape,self.storageType)

class _DeclaredFieldFunction(object):
    """Adapts an :class:`~proteus.ArchiveFields.ArchiveField` to the shape
    ``FemSpace.writeFunctionXdmf`` expects: ``.dof``, ``.name``,
    ``.femSpace``.

    ``archiveFiniteElementResiduals`` defines an equivalent ``dummy`` class
    inline. That inline class is why every auxiliary field in Proteus got
    written through a function named "residuals": it was the only adapter
    available. This one is named for what it does and carries the field's
    metadata along, so it can be the general path.
    """

    __slots__ = ("dof", "name", "femSpace", "field")

    def __init__(self, field, name, femSpace):
        self.field = field
        self.dof = field.value
        self.name = name
        self.femSpace = femSpace


class OneLevelTransport(NonlinearEquation):
    r""" A class for finite element discretizations of multicomponent
    advective-diffusive-reactive transport on a single spatial mesh.

    Objects of this type take the initial-boundary value
    problem for

    .. math:: 

            m_t^i + \nabla \cdot (\mathbf{f}^i - \sum_k \mathbf{a}^{ik}
            \nabla \phi^k) + H^i(\nabla u) + r^i = 0

    and turn it into the discrete (nonlinear or linear) algebraic
    problem

    .. math:: F(U) = 0

    where F and U have dimension `self.dim`. The Jacobian of F or an
    approximation for it may also be  provided.

    The NonlinearEquation interface is

    * self.dim

    * getResidual(u,r)

    * getJacobian(jacobian).

    The rest of the functions in this class are either private functions
    or return various other pieces of information.

    Attributes
    ----------
    ebq_global[('velocityAverage',0)] : array
        This attribute stores the average velocity along an edge given 
        a discontinous velocity field.

    """
    #share the storage of weighted shape arrays whose quadrature weights are equal
    #(see makeAliasForEqualWeights in __init__); a class attribute so it can be turned off
    shareWeightedArrays = True
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
                 movingDomain=False,
                 bdyNullSpace=False,
                 hasCutCells=False):#,
        r""" Allocate storage and initialize some variables.

        Parameters
        ----------
        uDict : dict
            Dictionary of 
            :class:`proteus.FemTools.FiniteElementFunction` objects.

        phiDict : dict
            Dictionary of
            :class:`proteus.FemTools.FiniteElementFunction` objects.

        testSpaceDict : dict
            Dictionary of FiniteElementSpace objects

        dofBoundaryConditionsDict : dict
            Dictionary of DOFBoundaryConditions objects for the 
            Dirichlet conditions.

        coefficients : :class:`proteus.TransportCoefficients.TC_base`
            Problem's Transport Coefficients class.

        elementQuadratureDict : dict
            Dictionary of dictionaries of quadrature rules for each 
            element integral in each component equation.

        elementBoundaryQuadratureDict : dict
            Dictionary of dictionaries of quadrature rules for each 
            element boundary integral in each component equation

        stabilization : bool

        shockCapturing : bool

        numericalFlux : bool

        bdyNullSpace : bool
            Indicates whether the boundary conditions create a global
            null space.

        Notes
        -----

        The constructor sets the input arguments, calculates
        dimensions, and allocates storage. The meanings of variable
        suffixes are

        * global          -- per physical domain

        * element         -- per element

        * elementBoundary -- per element boundary

        The prefix n means 'number of'.

        Storage is divided into quantities required at different sets
        of points or geometric entities. Each type of storage has a
        dictionary for all the quantities of that type. The names
        and dimensions of the storage dictionaries are

        * e -- at element

        * q -- at element quadrature, unique to elements

        * ebq -- at element boundary quadrature, unique to elements

        * ebq_global -- at element boundary quadrature, unique to element 
          boundary

        * ebqe -- at element boundary quadrature, unique to global, 
          exterior element boundary

        * phi_ip -- at the generalized interpolation points required to
          build a nonlinear phi
        """
        from . import Comm
        #
        #set the objects describing the method and boundary conditions
        #
        self.hasCutCells=False
        self.movingDomain=movingDomain
        self.tLast_mesh=None
        self.par_info = ParInfo_petsc4py()
        #
        self.name=name
        self.sd=sd
        for u in list(uDict.values()):
            if u.femSpace.referenceFiniteElement.localFunctionSpace.nonzeroHessians:
                if stabilization is not None:
                    self.Hess=True
        self.Hess=True#False
        self.lowmem=True
        self.timeTerm=True#allow turning off  the  time derivative
        #self.lowmem=False
        self.testIsTrial=True
        self.phiTrialIsTrial=True
        #trial-dup try to share trial and test information?
        self.uniform_trial_and_test_spaces = False
        self.unique_test_trial_range = list(range(coefficients.nc))
        self.duplicate_test_trial_range = list(range(0))
        if self.uniform_trial_and_test_spaces:
            self.unique_test_trial_range  = list(range(1))
            self.duplicate_test_trial_range = list(range(1,coefficients.nc))
        self.u = uDict
        self.phi  = phiDict
        self.dphi={}
        for ck,phi in phiDict.items():
            if ck in coefficients.potential:
                for cj in list(coefficients.potential[ck].keys()):
                    self.dphi[(ck,cj)] = FiniteElementFunction(phi.femSpace)
            else:
                self.dphi[(ck,ck)] = FiniteElementFunction(phi.femSpace)
        #check for nonlinearities in the diffusion coefficient that don't match the potential
        for ci,ckDict in coefficients.diffusion.items():
            #for ck,cjDict in coefficients.diffusion.iteritems(): #cek: bug?
            for ck,cjDict in ckDict.items():
                for cj in list(cjDict.keys()):
                    if (ck,cj) not in self.dphi:
                        self.dphi[(ck,cj)] = FiniteElementFunction(phi.femSpace)
        self.matType = matType
        #try to reuse test and trial information across components if spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature#True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1,coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        ## Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh #assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList=None #explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.bdyNullSpace = bdyNullSpace
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict #no velocity post-processing for now
        self.fluxBoundaryConditions=fluxBoundaryConditionsDict
        self.advectiveFluxBoundaryConditionsSetterDict=advectiveFluxBoundaryConditionsSetterDict
        self.diffusiveFluxBoundaryConditionsSetterDictDict = diffusiveFluxBoundaryConditionsSetterDictDict
        self.stressFluxBoundaryConditionsSetterDict = stressFluxBoundaryConditionsSetterDict
        #determine whether  the stabilization term is nonlinear
        self.stabilizationIsNonlinear = False
        if self.stabilization is not None:
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
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux is not None) or
                                                 (numericalFluxType is not None) or
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
        #cek make full diffusion default if not in sdInfo
        if self.sd:
            for ci,ckDict in coefficients.diffusion.items():
                for ck in list(ckDict.keys()):
                    if (ci,ck) not in coefficients.sdInfo:
                        coefficients.sdInfo[(ci,ck)] = (numpy.arange(start=0,stop=self.nSpace_global**2+1,step=self.nSpace_global,dtype='i'),
                                                        numpy.array([list(range(self.nSpace_global)) for row in range(self.nSpace_global)],dtype='i'))
                    logEvent("Sparse diffusion information key "+repr((ci,ck))+' = '+repr(coefficients.sdInfo[(ci,ck)]))
        if self.sd and getattr(coefficients, 'sdInfo', None):
            # the models build these with 'i'; the kernels take the CSR index type
            for key, (sdRowptr, sdColind) in list(coefficients.sdInfo.items()):
                coefficients.sdInfo[key] = (numpy.ascontiguousarray(sdRowptr, dtype=CSR_INDEX_DTYPE),
                                            numpy.ascontiguousarray(sdColind, dtype=CSR_INDEX_DTYPE))
        #
        NonlinearEquation.__init__(self,self.nFreeVDOF_global)
        #
        #build the quadrature point dictionaries from the input (this
        #is just for convenience so that the input doesn't have to be
        #complete)
        #
        self._elementQuadrature = elementQuadrature
        self._elementBoundaryQuadrature = elementBoundaryQuadrature
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
        if self.stabilization is not None:
            for I in self.coefficients.elementIntegralKeys:
                if elemQuadIsDict:
                    if I in elementQuadrature:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature
        if self.shockCapturing is not None:
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
        #mwf include tag telling me which indeces are which quadrature rule?
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
        #
        #storage dictionaries
        self.scalars_element = StorageSet(shape=(self.mesh.nElements_global,))
        #
        # element quadrature
        #
        self.q={}
        #now form sets for the different storage formats in self.q
        #coefficients
        self.points_quadrature = StorageSet(shape=(self.mesh.nElements_global,self.nQuadraturePoints_element,3))
        self.scalars_quadrature = StorageSet(shape=(self.mesh.nElements_global,self.nQuadraturePoints_element))
        self.vectors_quadrature = StorageSet(shape=(self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global))
        self.tensors_quadrature = StorageSet(shape={})
        #shape functions
        self.test_shape_quadrature = StorageSet(shape={})
        self.trial_shape_quadrature = StorageSet(shape={})
        trial_shape_quadrature_duplicate = StorageSet(shape={})
        trial_shape_quadrature_duplicate_map = {}

        self.phi_trial_shape_quadrature = StorageSet(shape={})
        self.test_shapeGradient_quadrature = StorageSet(shape={})
        self.trial_shapeGradient_quadrature = StorageSet(shape={})
        self.phi_trial_shapeGradient_quadrature = StorageSet(shape={})
        self.phi_trial_shapeHessian_quadrature = StorageSet(shape={})
        self.test_shapeHessian_quadrature = StorageSet(shape={})
        self.trial_shape_X_test_shape_quadrature = StorageSet(shape={})
        self.phi_trial_shape_X_test_shape_quadrature = StorageSet(shape={})
        self.trial_shape_X_test_shapeGradient_quadrature = StorageSet(shape={})
        self.trial_shapeGradient_X_test_shape_quadrature = StorageSet(shape={})
        self.phi_trial_shapeGradient_X_test_shape_quadrature = StorageSet(shape={})
        self.gradient_X_test_shapeGradient_quadrature = StorageSet(shape={})
        self.trial_shapeGradient_X_test_shapeGradient_quadrature = StorageSet(shape={})
        self.phi_trial_shapeGradient_X_test_shapeGradient_quadrature = StorageSet(shape={})
        #
        # element boundary quadrature (element trace values)
        #
        self.ebq = {}
        #coefficients
        self.points_elementBoundaryQuadrature = StorageSet(shape=(self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary,3))
        self.scalars_elementBoundaryQuadrature = StorageSet(shape=(self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary))
        self.vectors_elementBoundaryQuadrature = StorageSet(shape=(self.mesh.nElements_global,self.mesh.nElementBoundaries_element,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global))
        self.tensors_elementBoundaryQuadrature = StorageSet(shape={})
        #shape
        self.trial_shape_elementBoundaryQuadrature = StorageSet(shape={})
        self.phi_trial_shape_elementBoundaryQuadrature = StorageSet(shape={})
        self.test_shape_elementBoundaryQuadrature = StorageSet(shape={})
        self.trial_shapeGradient_elementBoundaryQuadrature = StorageSet(shape={})
        self.phi_trial_shapeGradient_elementBoundaryQuadrature = StorageSet(shape={})
        self.trial_shape_X_test_shape_elementBoundaryQuadrature = StorageSet(shape={})
        self.trial_shapeGradient_X_test_shape_elementBoundaryQuadrature = StorageSet(shape={})
        self.phi_trial_shapeGradient_X_test_shape_elementBoundaryQuadrature = StorageSet(shape={})
        self.gradient_X_test_shapeGradient_normal_elementBoundaryQuadrature = StorageSet(shape={})
        #
        # element boundary quadrature (global)
        #
        self.ebq_global = {}
        #coefficients
        self.points_elementBoundaryQuadrature_global = StorageSet(shape=(self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3))
        self.scalars_elementBoundaryQuadrature_global = StorageSet(shape=(self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary))
        self.vectors_elementBoundaryQuadrature_global = StorageSet(shape=(self.mesh.nElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global))
        #
        # phi generalized interpolation points
        #
        self.phi_ip = {}
        #coefficients
        self.points_phi_ip=StorageSet(shape=(self.mesh.nElements_global,self.n_phi_ip_element[0],3))
        self.scalars_phi_ip=StorageSet(shape=(self.mesh.nElements_global,self.n_phi_ip_element[0]))
        self.vectors_phi_ip=StorageSet(shape=(self.mesh.nElements_global,self.n_phi_ip_element[0],self.nSpace_global))
        self.tensors_phi_ip=StorageSet(shape={})
        #shape
        self.trial_shape_phi_ip=StorageSet(shape={})
        #mwf for interpolating residual
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.trial_shapeGradient_phi_ip = StorageSet(shape={})
            self.test_shape_phi_ip = StorageSet(shape={})
            self.test_shapeGradient_phi_ip = StorageSet(shape={})
            self.trial_shapeHessian_phi_ip  = StorageSet(shape={})
            self.test_shapeHessian_phi_ip = StorageSet(shape={})
        #
        # element boundary quadrature (global exterior)
        #
        self.ebqe = {}
        #coefficients
        #for now, just get everything that's in elementBoundaryQuadrature?
        #shape
        #for now, just get everything that's in elementBoundaryQuadrature?
        #
        # build sets
        #
        self.integralKeys = ['m','f','a','r','H','stab','numDiff','u','sigma']
        self.elementBoundaryIntegralKeys = ['f','a','u','H','sigma']
        #u
        self.scalars_quadrature |= set([('u',ci) for ci  in range(self.nc)])
        #trial-dup start
        #orig self.trial_shape_quadrature |= set([('v',ci) for ci in range(self.nc)])
        self.trial_shape_quadrature |= set([('v',ci) for ci in self.unique_test_trial_range])
        trial_shape_quadrature_duplicate |= set([('v',ci) for ci in self.duplicate_test_trial_range])
        for ci in self.duplicate_test_trial_range:
            trial_shape_quadrature_duplicate_map[('v',ci)] = ('v',self.unique_test_trial_range[0])
        #trial-dup end
        self.scalars_elementBoundaryQuadrature |= set([('u',ci) for ci  in range(self.nc)])
        self.trial_shape_elementBoundaryQuadrature |= set([('v',ci) for ci in range(self.nc)])
        self.scalars_phi_ip |= set([('u',ci) for ci  in range(self.nc)])
        self.trial_shape_phi_ip |= set([('v',ci) for ci in range(self.nc)])
        self.scalars_quadrature |= set([('dV_u',ci) for ci  in range(self.nc)])
        self.scalars_elementBoundaryQuadrature |= set([('dS_u',ci) for ci  in range(self.nc)])
        self.test_shape_elementBoundaryQuadrature |= set([('w*dS_u',ci) for ci  in range(self.nc)])
        #m
        self.scalars_quadrature |= set([('m',ci) for ci  in list(self.coefficients.mass.keys())])
        self.scalars_elementBoundaryQuadrature |= set([('m',ci) for ci  in list(self.coefficients.mass.keys())])
        self.scalars_phi_ip |= set([('m',ci) for ci  in list(self.coefficients.mass.keys())])
        self.scalars_quadrature |= set([('mt',ci) for ci  in list(self.coefficients.mass.keys())])
        self.test_shape_quadrature |= set([('w',ci) for ci in list(self.coefficients.mass.keys())])
        self.test_shape_quadrature |= set([('w*dV_m',ci) for ci in list(self.coefficients.mass.keys())])
        for ci,cjDict in self.coefficients.mass.items():
            self.scalars_quadrature |= set([('dm',ci,cj) for cj  in list(cjDict.keys())])
            self.scalars_elementBoundaryQuadrature |= set([('dm',ci,cj) for cj  in list(cjDict.keys())])
            self.scalars_phi_ip |= set([('dm',ci,cj) for cj  in list(cjDict.keys())])
            self.scalars_quadrature |= set([('dmt',ci,cj) for cj  in list(cjDict.keys())])
            self.trial_shape_X_test_shape_quadrature |= set([('vXw*dV_m',cj,ci) for cj in list(cjDict.keys())])
        #f
        self.vectors_quadrature |= set([('f',ci) for ci  in list(self.coefficients.advection.keys())])
        self.vectors_elementBoundaryQuadrature |= set([('f',ci) for ci  in list(self.coefficients.advection.keys())])
        self.vectors_phi_ip |= set([('f',ci) for ci  in list(self.coefficients.advection.keys())])
        self.test_shapeGradient_quadrature |= set([('grad(w)',ci) for ci in list(self.coefficients.advection.keys())])
        self.test_shapeGradient_quadrature |= set([('grad(w)*dV_f',ci) for ci in list(self.coefficients.advection.keys())])
        self.test_shape_elementBoundaryQuadrature |= set([('w',ci) for ci in list(self.coefficients.advection.keys())])
        self.test_shape_elementBoundaryQuadrature |= set([('w*dS_f',ci) for ci in list(self.coefficients.advection.keys())])
        self.vectors_quadrature |= set([('velocity',ci) for ci  in list(self.coefficients.advection.keys())])
        self.vectors_elementBoundaryQuadrature |= set([('velocity',ci) for ci  in list(self.coefficients.advection.keys())])
        for ci,cjDict in self.coefficients.advection.items():
            self.vectors_quadrature |= set([('df',ci,cj) for cj  in list(cjDict.keys())])
            self.vectors_elementBoundaryQuadrature |= set([('df',ci,cj) for cj  in list(cjDict.keys())])
            self.vectors_phi_ip |= set([('df',ci,cj) for cj  in list(cjDict.keys())])
            self.trial_shape_X_test_shapeGradient_quadrature |= set([('vXgrad(w)*dV_f',cj,ci) for cj in list(cjDict.keys())])
            self.trial_shape_X_test_shape_elementBoundaryQuadrature |= set([('vXw*dS_f',cj,ci) for cj in list(cjDict.keys())])
            #cek added these so they would be available for level set computations
            self.vectors_quadrature |= set([('grad(u)',cj) for cj in list(cjDict.keys())])
            self.vectors_elementBoundaryQuadrature |= set([('grad(u)',cj) for cj in list(cjDict.keys())])
            self.trial_shapeGradient_quadrature |= set([('grad(v)',cj) for cj in list(cjDict.keys())  ])
            self.trial_shapeGradient_elementBoundaryQuadrature |= set([('grad(v)',cj) for cj in list(cjDict.keys())])
        #a and phi
        self.vectors_quadrature |= set([('velocity',ci) for ci  in list(self.coefficients.diffusion.keys())])
        self.vectors_elementBoundaryQuadrature |= set([('velocity',ci) for ci  in list(self.coefficients.diffusion.keys())])
        self.test_shapeGradient_quadrature |= set([('grad(w)',ci) for ci in list(self.coefficients.diffusion.keys())])
        self.test_shape_elementBoundaryQuadrature |= set([('w',ci) for ci in list(self.coefficients.diffusion.keys())])
        self.scalars_quadrature |= set([('phi',ck) for ck  in list(self.coefficients.potential.keys())])
        self.scalars_elementBoundaryQuadrature |= set([('phi',ck) for ck  in list(self.coefficients.potential.keys())])
        self.scalars_phi_ip |= set([('phi',ck) for ck  in list(self.coefficients.potential.keys())])
        if numericalFluxType is not None:
            self.test_shape_quadrature |= set([('w',ci) for ci in list(self.coefficients.diffusion.keys())])
        for ci,ckDict in self.coefficients.diffusion.items():
            self.test_shapeGradient_quadrature |= set([('grad(w)*dV_a',ck,ci) for ck in ckDict])
            self.test_shape_elementBoundaryQuadrature |= set([('w*dS_a',ck,ci) for ck in ckDict])
            self.tensors_quadrature |= set([('a',ci,ck) for ck  in list(ckDict.keys())])
            self.tensors_elementBoundaryQuadrature |= set([('a',ci,ck) for ck  in list(ckDict.keys())])
            self.tensors_phi_ip |= set([('a',ci,ck) for ck  in list(ckDict.keys())])
            self.vectors_quadrature |= set([('grad(phi)',ck) for ck  in list(ckDict.keys())])
            self.vectors_quadrature |= set([('grad(u)',ck) for ck  in list(ckDict.keys())])            #\todo remove this when nonlinear phi fully supported
            self.phi_trial_shapeGradient_quadrature |= set([('grad(v)',ck) for ck  in list(ckDict.keys())])
            self.vectors_elementBoundaryQuadrature |= set([('grad(phi)',ck) for ck  in list(ckDict.keys())])
            self.vectors_elementBoundaryQuadrature |= set([('grad(u)',ck) for ck  in list(ckDict.keys())]) #\todo remove when nonlinear phi...
            self.phi_trial_shapeGradient_elementBoundaryQuadrature |= set([('grad(v)',ck) for ck  in list(ckDict.keys())])
            self.gradient_X_test_shapeGradient_quadrature |= set([('grad(phi)Xgrad(w)*dV_a',ck,ci) for ck in list(ckDict.keys())])
            self.gradient_X_test_shapeGradient_quadrature |= set([('grad(u)Xgrad(w)*dV_a',ck,ci) for ck in list(ckDict.keys())]) #\todo remove when nonlinear phi...
            if numericalFluxType is not None:
                self.test_shape_quadrature |= set([('w*dV_a',ck,ci) for ck in list(ckDict.keys())])
            for ck,cjDict in ckDict.items():
                self.tensors_quadrature |= set([('da',ci,ck,cj) for cj  in list(cjDict.keys())])
                self.tensors_elementBoundaryQuadrature |= set([('da',ci,ck,cj) for cj  in list(cjDict.keys())])
                self.tensors_phi_ip |= set([('da',ci,ck,cj) for cj  in list(cjDict.keys())])
                #mwf trial-dup start
                #mwf orig self.trial_shape_quadrature |= set([('v',cj) for cj in cjDict.keys()])
                unique_cj   = set.intersection(set(self.unique_test_trial_range),set(cjDict.keys()))
                duplicate_cj= set.intersection(set(self.duplicate_test_trial_range),set(cjDict.keys()))
                self.trial_shape_quadrature |= set([('v',cj) for cj in unique_cj])
                trial_shape_quadrature_duplicate |= set([('v',cj) for cj in duplicate_cj])
                #any reason to do this again
                for cj in self.duplicate_test_trial_range:
                    trial_shape_quadrature_duplicate_map[('v',cj)] = ('v',self.unique_test_trial_range[0])
                #mwf trial-dup end
                self.trial_shape_elementBoundaryQuadrature |= set([('v',cj) for cj in list(cjDict.keys())])
                #for now we need these even when the potential is not  nonlinear in this variable
                self.phi_trial_shapeGradient_quadrature |= set([('grad(v)',cj) for cj in list(cjDict.keys())])
                self.phi_trial_shapeGradient_X_test_shapeGradient_quadrature |= set([('grad(v)Xgrad(w)*dV_a',ck,cj,ci) for cj in list(cjDict.keys())])
                for cj in list(self.coefficients.potential[ck].keys()):
                    self.scalars_quadrature |= set([('dphi',ck,cj)])
                    self.scalars_elementBoundaryQuadrature |= set([('dphi',ck,cj)])
                    self.scalars_phi_ip |= set([('dphi',ck,cj)])
                    self.phi_trial_shapeGradient_quadrature |= set([('grad(v)',cj)])
                    self.phi_trial_shapeGradient_X_test_shapeGradient_quadrature |= set([('grad(v)Xgrad(w)*dV_a',ck,cj,ci)])
                    self.phi_trial_shapeGradient_X_test_shape_elementBoundaryQuadrature |= set([('grad(v)Xw*dS_a',ck,cj,ci)])
                    if numericalFluxType is not None:
                        self.trial_shape_X_test_shape_quadrature |= set([('vXw*dV_a',ck,cj,ci)])
        #mwf what if miss nonlinearities in diffusion that don't match potentials?
        for comb in list(self.dphi.keys()):
            self.scalars_quadrature |= set([('dphi',comb[0],comb[1])])
            self.scalars_elementBoundaryQuadrature |= set([('dphi',comb[0],comb[1])])
            self.scalars_phi_ip |= set([('dphi',comb[0],comb[1])])
        #r
        self.scalars_quadrature |= set([('r',ci) for ci  in list(self.coefficients.reaction.keys())])
        self.scalars_elementBoundaryQuadrature |= set([('r',ci) for ci  in list(self.coefficients.reaction.keys())])
        self.scalars_phi_ip |= set([('r',ci) for ci  in list(self.coefficients.reaction.keys())])
        self.test_shape_quadrature |= set([('w',ci) for ci in list(self.coefficients.reaction.keys())])
        self.test_shape_quadrature |= set([('w*dV_r',ci) for ci in list(self.coefficients.reaction.keys())])
        for ci,cjDict in self.coefficients.reaction.items():
            self.scalars_quadrature |= set([('dr',ci,cj) for cj  in list(cjDict.keys())])
            self.scalars_elementBoundaryQuadrature |= set([('dr',ci,cj) for cj  in list(cjDict.keys())])
            self.scalars_phi_ip |= set([('dr',ci,cj) for cj  in list(cjDict.keys())])
            self.trial_shape_X_test_shape_quadrature |= set([('vXw*dV_r',cj,ci) for cj in list(cjDict.keys())])
        #H
        self.scalars_quadrature |= set([('H',ci) for ci  in list(self.coefficients.hamiltonian.keys())])
        self.scalars_elementBoundaryQuadrature |= set([('H',ci) for ci  in list(self.coefficients.hamiltonian.keys())])
        self.scalars_phi_ip |= set([('H',ci) for ci  in list(self.coefficients.hamiltonian.keys())])
        self.test_shape_quadrature |= set([('w',ci) for ci in list(self.coefficients.hamiltonian.keys())])
        self.test_shape_quadrature |= set([('w*dV_H',ci) for ci in list(self.coefficients.hamiltonian.keys())])
        self.test_shapeGradient_quadrature |= set([('grad(w)',ci) for ci in list(self.coefficients.hamiltonian.keys())])
        self.test_shapeGradient_quadrature |= set([('grad(w)*dV_H',ci) for ci in list(self.coefficients.hamiltonian.keys())])
        for ci,cjDict in self.coefficients.hamiltonian.items():
            self.vectors_quadrature |= set([('grad(u)',cj) for cj in list(cjDict.keys())])
            self.vectors_elementBoundaryQuadrature |= set([('grad(u)',cj) for cj in list(cjDict.keys())])
            self.trial_shapeGradient_quadrature |= set([('grad(v)',cj) for cj in list(cjDict.keys())  ])
            self.trial_shapeGradient_elementBoundaryQuadrature |= set([('grad(v)',cj) for cj in list(cjDict.keys())])
            self.vectors_quadrature |= set([('dH',ci,cj) for cj  in list(cjDict.keys())])
            self.vectors_elementBoundaryQuadrature |= set([('dH',ci,cj) for cj  in list(cjDict.keys())])
            self.scalars_phi_ip |= set([('dH',ci,cj) for cj  in list(cjDict.keys())])
            for cj in list(cjDict.keys()):
                self.trial_shapeGradient_X_test_shape_quadrature |= set([('grad(v)Xw*dV_H',cj,ci) for cj in list(cjDict.keys())])
        #sigma cek try to keep it simple for now
        #test & trial
        self.trial_shapeGradient_quadrature |= set([('grad(v)',ci) for ci in list(self.coefficients.stress.keys())])
        self.test_shapeGradient_quadrature |= set([('grad(w)',ci) for ci in list(self.coefficients.stress.keys())])
        self.test_shapeGradient_quadrature |= set([('grad(w)*dV_sigma',ci) for ci in list(self.coefficients.stress.keys())])
        self.test_shape_elementBoundaryQuadrature |= set([('w',ci) for ci in list(self.coefficients.stress.keys())])
        self.test_shape_elementBoundaryQuadrature |= set([('w*dS_sigma',ci) for ci in list(self.coefficients.stress.keys())])
        self.phi_trial_shapeGradient_quadrature |= set([('grad(v)',ci) for ci  in list(self.coefficients.stress.keys())])
        self.phi_trial_shapeGradient_elementBoundaryQuadrature |= set([('grad(v)',ci) for ci  in list(self.coefficients.stress.keys())])
        #grad(u)
        self.vectors_quadrature |= set([('grad(u)',ci) for ci  in list(self.coefficients.stress.keys())])
        self.vectors_elementBoundaryQuadrature |= set([('grad(u)',ci) for ci  in list(self.coefficients.stress.keys())])
        #sigma
        if len(list(self.coefficients.stress.keys())) > 0:
            self.tensors_quadrature |= set(['sigma'])
            self.tensors_elementBoundaryQuadrature |= set(['sigma'])
        #dsigma
        for ci,cjDict in self.coefficients.stress.items():
            self.tensors_quadrature |= set([('dsigma',ci,cj) for cj  in list(cjDict.keys())])
            self.tensors_elementBoundaryQuadrature |= set([('dsigma',ci,cj) for cj  in list(cjDict.keys())])
        #TODO
        if numericalFluxType is not None:#put in more specific check
            self.test_shape_elementBoundaryQuadrature |= set([('w',ci) for ci in list(self.coefficients.hamiltonian.keys())])
            self.test_shape_elementBoundaryQuadrature |= set([('w*dS_H',ci) for ci in list(self.coefficients.hamiltonian.keys())])
            for ci,cjDict in self.coefficients.hamiltonian.items():
                self.trial_shape_X_test_shape_elementBoundaryQuadrature |= set([('vXw*dS_H',cj,ci) for cj in list(cjDict.keys())])

        #stabilization
        if self.stabilization is not None:
            for cjDict in list(self.coefficients.advection.values()):
                self.vectors_quadrature |= set([('grad(u)',cj) for cj  in list(cjDict.keys())])
                self.trial_shapeGradient_quadrature |= set([('grad(v)',cj) for cj  in list(cjDict.keys())])
            for cjDict in list(self.coefficients.hamiltonian.values()):
                self.vectors_quadrature |= set([('grad(u)',cj) for cj  in list(cjDict.keys())])
                self.trial_shapeGradient_quadrature |= set([('grad(v)',cj) for cj  in list(cjDict.keys())])
            self.test_shape_quadrature |= set([('w*dV_stab',ci) for ci in list(self.coefficients.mass.keys())])
            self.test_shapeGradient_quadrature |= set([('grad(w)*dV_stab',ci) for ci in list(self.coefficients.advection.keys())])

            for ci,ckDict in self.coefficients.diffusion.items():
                if self.Hess:
                    self.tensors_quadrature |= set([('Hess(phi)',ck) for ck  in list(ckDict.keys())])
                    self.tensors_quadrature |= set([('Hess(u)',ck) for ck  in list(ckDict.keys())])
                    self.phi_trial_shapeHessian_quadrature |= set([('Hess(v)',ck) for ck in list(ckDict.keys())])
                    self.test_shapeHessian_quadrature |= set([('Hess(w)',ci)])
                    self.test_shapeHessian_quadrature |= set([('Hess(w)*dV_stab',ck,ci) for ck in ckDict])
                self.test_shapeGradient_quadrature |= set([('grad(w)*dV_stab',ck,ci) for ck in ckDict])
            self.test_shape_quadrature |= set([('w*dV_stab',ci) for ci in list(self.coefficients.reaction.keys())])
            self.test_shapeGradient_quadrature |= set([('grad(w)*dV_stab',ci) for ci in list(self.coefficients.hamiltonian.keys())])
            for ci in range(self.nc):
                self.scalars_quadrature |= set([('subgridError',ci)])
                self.scalars_quadrature |= set([('pdeResidual',ci)])
                for cj in self.coefficients.stencil[ci]:
                    self.trial_shape_quadrature |= set([('dpdeResidual',ci,cj)])
                    self.test_shape_quadrature |= set([('Lstar*w*dV',cj,ci)])
                for cj in range(self.coefficients.nc):
                    self.trial_shape_quadrature |= set([('dsubgridError',ci,cj)])
            #mwf for interpolating subgrid errorr
            if self.stabilization.usesGradientStabilization:
                for ci,ckDict in self.coefficients.diffusion.items():
                    self.vectors_phi_ip |= set([('grad(phi)',ck) for ck  in list(ckDict.keys())])
                    self.scalars_phi_ip |= set([('mt',ci) for ci  in list(self.coefficients.mass.keys())])
                    for ci,cjDict in self.coefficients.mass.items():
                        self.scalars_phi_ip |= set([('dmt',ci,cj) for cj  in list(cjDict.keys())])

                for cjDict in list(self.coefficients.advection.values()):
                    self.vectors_phi_ip |= set([('grad(u)',cj) for cj  in list(cjDict.keys())])
                    self.trial_shapeGradient_phi_ip |= set([('grad(v)',cj) for cj  in list(cjDict.keys())])
                for cjDict in list(self.coefficients.hamiltonian.values()):
                    self.vectors_phi_ip |= set([('grad(u)',cj) for cj  in list(cjDict.keys())])
                    self.trial_shapeGradient_phi_ip |= set([('grad(v)',cj) for cj  in list(cjDict.keys())])
                self.test_shape_phi_ip |= set([('w*dV_stab',ci) for ci in list(self.coefficients.mass.keys())])
                self.test_shapeGradient_phi_ip |= set([('grad(w)*dV_stab',ci) for ci in list(self.coefficients.advection.keys())])

                for ci,ckDict in self.coefficients.diffusion.items():
                    if self.Hess:
                        self.tensors_phi_ip |= set([('Hess(phi)',ck) for ck  in list(ckDict.keys())])
                        self.tensors_phi_ip |= set([('Hess(u)',ck) for ck  in list(ckDict.keys())])
                        self.trial_shapeHessian_phi_ip |= set([('Hess(v)',ck) for ck in list(ckDict.keys())])
                        self.test_shapeHessian_phi_ip |= set([('Hess(w)',ci)])
                        self.test_shapeHessian_phi_ip |= set([('Hess(w)*dV_stab',ck,ci) for ck in ckDict])
                    self.test_shapeGradient_phi_ip |= set([('grad(w)*dV_stab',ck,ci) for ck in ckDict])
                self.test_shape_phi_ip |= set([('w*dV_stab',ci) for ci in list(self.coefficients.reaction.keys())])
                self.test_shapeGradient_phi_ip |= set([('grad(w)*dV_stab',ci) for ci in list(self.coefficients.hamiltonian.keys())])
                for ci in range(self.nc):
                    self.scalars_phi_ip |= set([('subgridError',ci)])
                    self.scalars_phi_ip |= set([('pdeResidual',ci)])
                    for cj in self.coefficients.stencil[ci]:
                        self.trial_shape_phi_ip |= set([('dpdeResidual',ci,cj)])
                        self.test_shape_phi_ip |= set([('Lstar*w*dV',cj,ci)])
                    for cj in range(self.coefficients.nc):
                        self.trial_shape_phi_ip |= set([('dsubgridError',ci,cj)])
                self.tensors_phi_ip |=  set(['J',
                                             'inverse(J)'])
                self.scalars_phi_ip |= set(['det(J)'])
                for ci in range(self.nc):
                    self.scalars_phi_ip |= set([('cfl',ci),
                                                ('pe',ci)])
            #end usesFEMinterpolant case

        #shock capturing
        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                self.scalars_quadrature |= set([('numDiff',ci,ci)])
                self.scalars_quadrature |= set([('pdeResidual',ci)])
                self.vectors_quadrature |= set([('grad(u)',ci)])
                self.trial_shapeGradient_quadrature |= set([('grad(v)',ci)])
                self.test_shapeGradient_quadrature |= set([('grad(w)',ci)])
                self.test_shapeGradient_quadrature |= set([('grad(w)*dV_numDiff',ci,ci)])
                self.gradient_X_test_shapeGradient_quadrature |= set([('grad(u)Xgrad(w)*dV_numDiff',ci,ci)])
                self.trial_shapeGradient_X_test_shapeGradient_quadrature |= set([('grad(v)Xgrad(w)*dV_numDiff',ci,ci,ci)])
                for cj in self.coefficients.stencil[ci]:
                    self.trial_shape_quadrature |= set([('dpdeResidual',ci,cj)])
        # numerical flux
        #mwf switch quadrature sets for these after get ebqe working?
        for ci in list(self.elementBoundaryIntegrals.keys()):
            for qk in ['massAverage','velocityJump','advectiveFlux','totalFlux']:
                self.scalars_elementBoundaryQuadrature_global |= set([(qk,ci)])
            if ci in self.coefficients.diffusion:
                for ck in self.coefficients.diffusion[ci]:
                    self.scalars_elementBoundaryQuadrature_global |= set(['penalty'])
                    for qk in ['diffusiveFlux']:
                        self.scalars_elementBoundaryQuadrature_global |= set([(qk,ck,ci)])
            for qk in ['massJump','velocityAverage']:
                self.vectors_elementBoundaryQuadrature_global |= set([(qk,ci)])
        for ci in list(self.coefficients.stress.keys()):
            self.scalars_elementBoundaryQuadrature_global |= set(['penalty'])
            self.scalars_elementBoundaryQuadrature_global |= set([('stressFlux',ci)])
        #mesh
        self.scalars_quadrature |= set(['det(J)',
                                        'abs(det(J))'])
        self.tensors_quadrature |= set(['J',
                                        'inverse(J)'])
        self.scalars_elementBoundaryQuadrature |= set(['sqrt(det(g))'])
        self.vectors_elementBoundaryQuadrature |= set(['n'])
        self.vectors_elementBoundaryQuadrature_global |= set(['n'])
        self.points_quadrature |= set(['x'])
        self.points_phi_ip |= set(['x'])
        self.points_elementBoundaryQuadrature |= set(['x','hat(x)','bar(x)'])
        self.points_elementBoundaryQuadrature_global |= set(['x','hat(x)','bar(x)'])
        self.tensors_elementBoundaryQuadrature |= set(['inverse(J)'])
        if self.movingDomain:
            self.points_quadrature |= set(['xt','x_last'])
            self.points_elementBoundaryQuadrature |= set(['xt','x_last'])

        #misc
        for ci in range(self.nc):
            self.scalars_quadrature |= set([('cfl',ci),
                                            ('pe',ci)])
        #mwf add for inflow boundary calculations, wastes space for interor
        self.scalars_elementBoundaryQuadrature |= set([('inflowFlux',ci) for ci in list(self.coefficients.advection.keys())])
        #for 2-sided Hamilton Jacobi fluxes
        if numericalFluxType is not None:#mwf TODO make more specific
            self.scalars_elementBoundaryQuadrature |= set([('HamiltonJacobiFlux',ci) for ci in list(self.coefficients.hamiltonian.keys())])
            #don't have to set exterior explicitly
        #mwf have to go back and separate out logic to remove duplication once things working
        #coefficients
        self.points_elementBoundaryQuadrature_exterior = StorageSet(set.union(self.points_elementBoundaryQuadrature,
                                                                              self.points_elementBoundaryQuadrature_global),
                                                                    shape=(self.mesh.nExteriorElementBoundaries_global,
                                                                           self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                           3))
        self.scalars_elementBoundaryQuadrature_exterior= StorageSet(set.union(self.scalars_elementBoundaryQuadrature,
                                                                              self.scalars_elementBoundaryQuadrature_global),
                                                                    shape=(self.mesh.nExteriorElementBoundaries_global,
                                                                           self.nElementBoundaryQuadraturePoints_elementBoundary))
        self.vectors_elementBoundaryQuadrature_exterior= StorageSet(set.union(self.vectors_elementBoundaryQuadrature,
                                                                              self.vectors_elementBoundaryQuadrature_global),
                                                                    shape=(self.mesh.nExteriorElementBoundaries_global,
                                                                           self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                           self.nSpace_global))
        self.tensors_elementBoundaryQuadrature_exterior= self.tensors_elementBoundaryQuadrature
        #shape
        self.trial_shape_elementBoundaryQuadrature_exterior = self.trial_shape_elementBoundaryQuadrature
        self.phi_trial_shape_elementBoundaryQuadrature_exterior = self.phi_trial_shape_elementBoundaryQuadrature
        self.test_shape_elementBoundaryQuadrature_exterior = self.test_shape_elementBoundaryQuadrature
        self.trial_shapeGradient_elementBoundaryQuadrature_exterior = self.trial_shapeGradient_elementBoundaryQuadrature
        self.phi_trial_shapeGradient_elementBoundaryQuadrature_exterior = self.phi_trial_shapeGradient_elementBoundaryQuadrature
        self.trial_shape_X_test_shape_elementBoundaryQuadrature_exterior = self.trial_shape_X_test_shape_elementBoundaryQuadrature
        self.trial_shapeGradient_X_test_shape_elementBoundaryQuadrature_exterior = self.trial_shapeGradient_X_test_shape_elementBoundaryQuadrature
        self.phi_trial_shapeGradient_X_test_shape_elementBoundaryQuadrature_exterior = self.phi_trial_shapeGradient_X_test_shape_elementBoundaryQuadrature
        self.gradient_X_test_shapeGradient_normal_elementBoundaryQuadrature_exterior = self.gradient_X_test_shapeGradient_normal_elementBoundaryQuadrature

        memory()
        #
        #start allocations
        #
        #element quadrature
        #
        self.scalars_element.allocate(self.q)
        #points element quadrature
        self.points_quadrature.allocate(self.q)
        #allocate scalars element quadrature
        self.scalars_quadrature.allocate(self.q)
        #allocate vectors element quadrature
        self.vectors_quadrature.allocate(self.q)
        #allocate tensors element quadrature
        for k in self.tensors_quadrature:
            if (self.sd
                and k[0] in ['a','da']
                and self.coefficients.sdInfo is not None
                and (k[1],k[2]) in list(self.coefficients.sdInfo.keys())):
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.coefficients.sdInfo[(k[1],k[2])][0][self.nSpace_global]),
                    'd')
            else:
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nSpace_global,
                     self.nSpace_global),
                    'd')
        logEvent(memory("element quadrature","OneLevelTransport"),level=4)
        #
        # element boundary quadrature
        #
        #allocate points element boundary quadrature
        #
        #mwf only need ebq if
        # numerical flux (dg method)
        # postprocessing cg velocity fields
        #
        needEBQ = (numericalFluxType is not None) and numericalFluxType.hasInterior
        if self.conservativeFlux is not None:
            for ci in list(self.conservativeFlux.keys()):
                needEBQ = needEBQ or self.conservativeFlux[ci] is not None
        if options.needEBQ:
            needEBQ = True
        self.needEBQ=needEBQ
        if needEBQ:
            self.points_elementBoundaryQuadrature.allocate(self.ebq)
            #allocate scalars element boundary quadrature
            self.scalars_elementBoundaryQuadrature.allocate(self.ebq)
            #allocate vectors element boundary quadrature
            self.vectors_elementBoundaryQuadrature.allocate(self.ebq)
            #allocate tensors element boundary quadrature
            for k in self.tensors_elementBoundaryQuadrature:
                if (self.sd
                    and k[0] in ['a','da']
                    and self.coefficients.sdInfo is not None
                    and (k[1],k[2]) in list(self.coefficients.sdInfo.keys())):
                    self.ebq[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.coefficients.sdInfo[(k[1],k[2])][0][self.nSpace_global]),
                        'd')
                else:
                    self.ebq[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nSpace_global,
                         self.nSpace_global),
                        'd')
            #allocate the metric tensor
            self.ebq['g'] = numpy.zeros((self.mesh.nElements_global,
                                           self.mesh.nElementBoundaries_element,
                                           self.nElementBoundaryQuadraturePoints_elementBoundary,
                                           max(1,self.nSpace_global-1),
                                           max(1,self.nSpace_global-1)),
                                          'd')
            logEvent(memory("element boundary quadrature","OneLevelTransport"),level=4)
        #
        # exterior element boundary quadrature
        #
        #allocate points element boundary quadrature
        #
        self.points_elementBoundaryQuadrature_exterior.allocate(self.ebqe)
        #allocate scalars element boundary quadrature
        self.scalars_elementBoundaryQuadrature_exterior.allocate(self.ebqe)
        #allocate vectors element boundary quadrature
        self.vectors_elementBoundaryQuadrature_exterior.allocate(self.ebqe)
        #allocate tensors element boundary quadrature
        for k in self.tensors_elementBoundaryQuadrature_exterior:
            if (self.sd
                and k[0] in ['a','da']
                and self.coefficients.sdInfo is not None
                and (k[1],k[2]) in list(self.coefficients.sdInfo.keys())):
                self.ebqe[k]=numpy.zeros(
                    (self.mesh.nExteriorElementBoundaries_global,
                     self.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.coefficients.sdInfo[(k[1],k[2])][0][self.nSpace_global]),
                    'd')
            else:
                self.ebqe[k]=numpy.zeros(
                    (self.mesh.nExteriorElementBoundaries_global,
                     self.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.nSpace_global,
                     self.nSpace_global),
                    'd')
        #allocate the metric tensor
        self.ebqe['g'] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,
                                       self.nElementBoundaryQuadraturePoints_elementBoundary,
                                       max(1,self.nSpace_global-1),
                                       max(1,self.nSpace_global-1)),
                                      'd')
        logEvent(memory("global exterior element boundary quadrature","OneLevelTransport"),level=4)
        self.forceStrongConditions= numericalFluxType.useStrongDirichletConstraints
        self.dirichletConditionsForceDOF = {}
        if self.forceStrongConditions:
            for cj in range(self.nc):
                self.dirichletConditionsForceDOF[cj] = DOFBoundaryConditions(self.u[cj].femSpace,dofBoundaryConditionsSetterDict[cj],weakDirichletConditions=False)
                # A DOF of a periodic pair is one unknown shared with its
                # partner, and periodicity wins over a Dirichlet value there,
                # as it does where Dirichlet DOFs are eliminated.
                for nodeSet in self.dirichletConditions[cj].periodicDOFDict.values():
                    for dofN in nodeSet:
                        self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.pop(dofN,None)
                        self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict.pop(dofN,None)
        #
        # element boundary quadrature global
        #
        #allocate points element boundary quadrature global
        #mwf only need ebq if
        # numerical flux (dg method)
        # postprocessing cg velocity fields
        needEBQ_GLOBAL = numericalFluxType is not None
        if self.conservativeFlux is not None:
            for ci in list(self.conservativeFlux.keys()):
                needEBQ_GLOBAL = needEBQ_GLOBAL or self.conservativeFlux[ci] is not None
        if options.needEBQ_GLOBAL:
            needEBQ_GLOBAL = True
        self.needEBQ_GLOBAL=needEBQ_GLOBAL
        if needEBQ_GLOBAL:
            self.points_elementBoundaryQuadrature_global.allocate(self.ebq_global)
            #allocate scalars element boundary quadrature global
            self.scalars_elementBoundaryQuadrature_global.allocate(self.ebq_global)
            #allocate vectors element boundary quadrature global
            self.vectors_elementBoundaryQuadrature_global.allocate(self.ebq_global)
            logEvent(memory("element boundary quadrature global","OneLevelTransport"),level=4)
        #
        # phi interpolation points
        #
        #allocate scalars phi  interpolation points
        #
        #\todo cek right now  I'm assuming all interpolation for phi_k is same for all k
        self.phi_ip['x'] = self.phi[0].femSpace.updateInterpolationPoints()
        #mwf need to make sure all the keys are here for computing pdeResidual and Lstar*w*dV
        #if trying to use FEM interpolant for subgridError
        self.scalars_phi_ip.allocate(self.phi_ip)
        #allocate vectors phi interpolation points
        self.vectors_phi_ip.allocate(self.phi_ip)
        #allocate tensors phi interpolation points
        for k in self.tensors_phi_ip:
            if (self.sd
                and k[0] in ['a','da']
                and self.coefficients.sdInfo is not None
                and (k[1],k[2]) in list(self.coefficients.sdInfo.keys())):
                self.phi_ip[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.n_phi_ip_element[0],
                     self.coefficients.sdInfo[(k[1],k[2])][0][self.nSpace_global]),
                    'd')
            else:
                self.phi_ip[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.n_phi_ip_element[0],
                     self.nSpace_global,
                     self.nSpace_global),
                    'd')
        #mwf try to share quadrature points here
        for k in self.trial_shape_phi_ip:
            self.phi_ip[k]=numpy.zeros(
                (self.mesh.nElements_global,
                 self.phi[k[-1]].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                 self.nDOF_phi_trial_element[k[-1]]),
                'd')

        if self.stabilization and self.stabilization.usesGradientStabilization:
            for k in sorted(self.trial_shapeGradient_phi_ip):
                self.phi_ip[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.phi[k[-1]].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                     self.nDOF_trial_element[k[-1]],
                     self.nSpace_global),
                    'd')
            for k in self.test_shape_phi_ip:
                self.phi_ip[k] = numpy.zeros(
                    (self.mesh.nElements_global,
                     self.phi[k[-1]].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                     self.nDOF_test_element[k[-1]]),
                    'd')
            for k in self.test_shapeGradient_phi_ip:
                self.phi_ip[k] = numpy.zeros(
                    (self.mesh.nElements_global,
                     self.phi[k[-1]].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                     self.nDOF_test_element[k[-1]],
                     self.nSpace_global),
                    'd')
            if self.Hess:
                #allocate phi trial shape function gradients
                for k in sorted(self.trial_shapeHessian_phi_ip):
                    self.phi_ip[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.phi[k[-1]].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                         self.nDOF_phi_trial_element[k[-1]],
                         self.nSpace_global,
                         self.nSpace_global),
                        'd')
                for k in self.test_shapeHessian_phi_ip:
                    self.phi_ip[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.phi[k[-1]].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                         self.nDOF_phi_trial_element[k[-1]],
                         self.nSpace_global,
                         self.nSpace_global),
                        'd')

        logEvent(memory("interpolation points","OneLevelTransport"),level=4)
        #
        # shape
        #
        def makeAlias(sd,kw,vstring='v',wstring='w'):
            aliased=False
            kv0 = kw[0].replace(wstring,vstring)
            if len(kw) > 1:
                kv = (kv0,)+kw[1:]
            else:
                kv = kv0
            if self.testIsTrial:
                if kv in list(sd.keys()):
                    logEvent("Shallow copy of trial shape is being used for test shape %s " % kw[0],level=4)
                    sd[kw] = sd[kv]
                    aliased=True
            return aliased
        def makeAliasForEqualWeights(sd,k):
            """Share storage with an array already allocated for the same shape
            function (same test component) under another integral (m, f, a, r,
            ...) when both are weighted by the same quadrature weights: the
            values are then identical, and each is a full (elements, points,
            dofs[, space]) array."""
            name = k[0]
            if not self.shareWeightedArrays or 'dV_' not in name:
                return False
            base, I = name.rsplit('dV_',1)
            w = self.elementQuadratureWeights.get((I,)+tuple(reversed(k[1:])))
            if w is None:
                return False
            for kk in sd:
                if (kk == k or not isinstance(kk,tuple) or kk[-1] != k[-1]
                    or not kk[0].startswith(base+'dV_')):
                    continue
                wJ = self.elementQuadratureWeights.get((kk[0][len(base)+3:],)+tuple(reversed(kk[1:])))
                if wJ is not None and numpy.array_equal(w,wJ):
                    logEvent("Shallow copy of %s is being used for %s " % (kk,k),level=4)
                    sd[k] = sd[kk]
                    return True
            return False
        def makeAliasForComponent1(sd,k,entryStrings,refi=0):
            aliased=False
            #need to add combinatorial options
            ks = k[0]
            if ks not in entryStrings:
                return False
            k0 = (ks,)+(refi,)
            if self.reuse_test_trial_quadrature and refi is not None:
                if k0 in list(sd.keys()):
                    logEvent("Shallow copy of trial shape %s is being used for trial shape %s" % (k0,k),level=4)
                    sd[k] = sd[k0]
                    aliased=True
            return aliased
        def findReferenceComponent_phi(diffusion):
            firstComp = None
            for ci,ckDict in diffusion.items():
                tmp = sorted(ckDict.keys())
                if tmp:
                    if firstComp is None or tmp[0] < firstComp:
                        firstComp = tmp[0]
            return firstComp
        #
        # element quadrature
        #
        #allocate trial shape functions
        for k in sorted(self.trial_shape_quadrature):
            if not makeAliasForComponent1(self.q,k,['v'],refi=0):#need to handle multiple component combinations
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_trial_element[k[-1]]),
                    'd')
        for k in trial_shape_quadrature_duplicate:
            self.q[k] = self.q[trial_shape_quadrature_duplicate_map[k]]
        #mwf trial-dup end
        logEvent(memory("element quadrature, test/trial functions trial_shape","OneLevelTransport"),level=4)
        #allocate phi trial shape functions
        refComponent_phi = findReferenceComponent_phi(self.coefficients.diffusion)
        for k in sorted(self.phi_trial_shape_quadrature):
            if not makeAliasForComponent1(self.q,k,['v'],refi=refComponent_phi):
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_phi_trial_element[k[-1]]),
                    'd')
        logEvent(memory("element quadrature, test/trial functions phi_trial_shape","OneLevelTransport"),level=4)
        #allocate test shape functions
        for k in sorted(self.test_shape_quadrature):
            if not makeAlias(self.q,k) and not makeAliasForEqualWeights(self.q,k):
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_test_element[k[-1]]),
                    'd')
        logEvent(memory("element quadrature, test/trial functions test_shape","OneLevelTransport"),level=4)
        #allocate trial shape function gradients
        for k in sorted(self.trial_shapeGradient_quadrature):
            if not makeAliasForComponent1(self.q,k,['grad(v)'],refi=0):#need to handle multiple component combinations
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_trial_element[k[-1]],
                     self.nSpace_global),
                    'd')
        logEvent(memory("element quadrature, test/trial functions trial_shapeGradient","OneLevelTransport"),level=4)
        #allocate phi trial shape function gradients
        for k in sorted(self.phi_trial_shapeGradient_quadrature):
            if not makeAliasForComponent1(self.q,k,['grad(v)'],refi=refComponent_phi):
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_phi_trial_element[k[-1]],
                     self.nSpace_global),
                    'd')
        logEvent(memory("element quadrature, test/trial functions phi_trial_shapeGradient","OneLevelTransport"),level=4)
        #allocate test shape function gradients
        for k in sorted(self.test_shapeGradient_quadrature):
            if not makeAlias(self.q,k) and not makeAliasForEqualWeights(self.q,k):
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_test_element[k[-1]],
                     self.nSpace_global),
                    'd')
        if self.Hess:
            #allocate phi trial shape function gradients
            for k in sorted(self.phi_trial_shapeHessian_quadrature):
                if not makeAliasForComponent1(self.q,k,['Hess(v)'],refi=refComponent_phi):#
                    self.q[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.nQuadraturePoints_element,
                         self.nDOF_phi_trial_element[k[-1]],
                         self.nSpace_global,
                         self.nSpace_global),
                        'd')
            #allocate phi trial shape function gradients
            for k in self.test_shapeHessian_quadrature:
                if not makeAlias(self.q,k,vstring='Hess(v)',wstring='Hess(w)'):#
                    self.q[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.nQuadraturePoints_element,
                         self.nDOF_phi_trial_element[k[-1]],
                         self.nSpace_global,
                         self.nSpace_global),
                        'd')
        logEvent(memory("element quadrature, test/trial functions test_shapeGradient","OneLevelTransport"),level=4)
        if not self.lowmem:
            #allocate trial shape function X test shape functions
            for k in self.trial_shape_X_test_shape_quadrature:
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_trial_element[k[1]],
                     self.nDOF_test_element[k[2]]),
                    'd')
            logEvent(memory("element quadrature, test/trial functions trial_shape_X_test_shape","OneLevelTransport"),level=4)
            #allocate phi trial shape function X test shape functions
            for k in self.phi_trial_shape_X_test_shape_quadrature:
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_phi_trial_element[k[1]],
                     self.nDOF_test_element[k[2]]),
                    'd')
            logEvent(memory("element quadrature, test/trial functions phi_trial_shape_X_test_shape","OneLevelTransport"),level=4)
            #allocate gradient X test shape function gradients
            for k in self.gradient_X_test_shapeGradient_quadrature:
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_test_element[k[1]],
                     self.nSpace_global,
                     self.nSpace_global),
                    'd')
            logEvent(memory("element quadrature, test/trial functions gradient_X_test_shapeGradient","OneLevelTransport"),level=4)
            #allocate trial shape function X shape function gradients
            for k in self.trial_shape_X_test_shapeGradient_quadrature:
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_trial_element[k[1]],
                     self.nDOF_test_element[k[2]],
                     self.nSpace_global),
                    'd')
            logEvent(memory("element quadrature, test/trial functions trial_shape_X_test_shapeGradient","OneLevelTransport"),level=4)
            #
            #allocate trial shape function gradient X test shape function
            for k in self.trial_shapeGradient_X_test_shape_quadrature:
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_trial_element[k[1]],
                     self.nDOF_test_element[k[2]],
                     self.nSpace_global),
                    'd')
            logEvent(memory("element quadrature, test/trial functions trial_shapeGradient_X_test_shape","OneLevelTransport"),level=4)

            #allocate trial shape function gradient X test shape function gradients
            for k in self.trial_shapeGradient_X_test_shapeGradient_quadrature:
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_trial_element[k[2]],
                     self.nDOF_test_element[k[3]],
                     self.nSpace_global,
                     self.nSpace_global),
                    'd')
            logEvent(memory("element quadrature, test/trial functions trial_shapeGradient_X_test_shapeGradient","OneLevelTransport"),level=4)
            #allocate phi trial shape function gradient X test shape function gradients
            for k in self.phi_trial_shapeGradient_X_test_shapeGradient_quadrature:
                self.q[k]=numpy.zeros(
                    (self.mesh.nElements_global,
                     self.nQuadraturePoints_element,
                     self.nDOF_phi_trial_element[k[2]],
                     self.nDOF_test_element[k[3]],
                     self.nSpace_global,
                     self.nSpace_global),
                    'd')
            logEvent(memory("element quadrature, test/trial functions phi_trial_shapeGradient_X_test_shapeGradient","OneLevelTransport"),level=4)
        #
        # element boundary quadrature
        #
        #allocate trial shape element boundary quadrature
        if needEBQ:
            for k in sorted(self.trial_shape_elementBoundaryQuadrature):
                if not makeAliasForComponent1(self.ebq,k,['v'],refi=0):
                    self.ebq[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nDOF_trial_element[k[-1]]),
                        'd')
            logEvent(memory("element boundary quadrature, test/trial functions trial_shape","OneLevelTransport"),level=4)
            #allocate phi trial shape element boundary quadrature
            for k in sorted(self.phi_trial_shape_elementBoundaryQuadrature):
                if not makeAliasForComponent1(self.ebq,k,['v'],refi=refComponent_phi):
                    self.ebq[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nDOF_phi_trial_element[k[-1]]),
                        'd')
            logEvent(memory("element boundary quadrature, test/trial functions phi_trial_shape","OneLevelTransport"),level=4)
            #allocate test shape element boundary quadrature
            for k in self.test_shape_elementBoundaryQuadrature:
                if not makeAlias(self.ebq,k):
                    self.ebq[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nDOF_test_element[k[-1]]),
                        'd')
            logEvent(memory("element boundary quadrature, test/trial functions test_shape","OneLevelTransport"),level=4)
            #allocate trial shape gradient element boundary quadrature
            for k in sorted(self.trial_shapeGradient_elementBoundaryQuadrature):
                if not makeAliasForComponent1(self.ebq,k,['grad(v)'],refi=0):
                    self.ebq[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nDOF_trial_element[k[-1]],
                         self.nSpace_global),
                        'd')
            logEvent(memory("element boundary quadrature, test/trial functions trial_shapeGradient","OneLevelTransport"),level=4)
            #allocate phi trial shape gradient element boundary quadrature
            for k in sorted(self.phi_trial_shapeGradient_elementBoundaryQuadrature):
                if not makeAliasForComponent1(self.ebq,k,['grad(v)'],refi=refComponent_phi):
                    self.ebq[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nDOF_phi_trial_element[k[-1]],
                         self.nSpace_global),
                        'd')
            logEvent(memory("element boundary quadrature, test/trial functions phi_trial_shapeGradient","OneLevelTransport"),level=4)
            #allocate trial shape X test shape element boundary quadratre
            if not self.lowmem:
                for k in self.trial_shape_X_test_shape_elementBoundaryQuadrature:
                    self.ebq[k]=numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nDOF_trial_element[k[1]],
                         self.nDOF_test_element[k[2]]),
                        'd')
                logEvent(memory("element boundary quadrature, test/trial functions trial_shape_X_test_shape","OneLevelTransport"),level=4)
        #
        # global exterior element boundary quadrature
        #
        #allocate trial shape element boundary quadrature
        #
        for k in sorted(self.trial_shape_elementBoundaryQuadrature):
            if not makeAliasForComponent1(self.ebqe,k,['v'],refi=0):
                self.ebqe[k]=numpy.zeros(
                    (self.mesh.nExteriorElementBoundaries_global,
                     self.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.nDOF_trial_element[k[-1]]),
                    'd')
        logEvent(memory("global exterior element boundary quadrature, test/trial functions trial_shape","OneLevelTransport"),level=4)
        #allocate phi trial shape element boundary quadrature
        for k in sorted(self.phi_trial_shape_elementBoundaryQuadrature):
            if not makeAliasForComponent1(self.ebqe,k,['v'],refi=refComponent_phi):
                self.ebqe[k]=numpy.zeros(
                    (self.mesh.nExteriorElementBoundaries_global,
                     self.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.nDOF_phi_trial_element[k[-1]]),
                    'd')
        logEvent(memory("global exterior element boundary quadrature, test/trial functions phi_trial_shape","OneLevelTransport"),level=4)
        #allocate test shape element boundary quadrature
        for k in self.test_shape_elementBoundaryQuadrature:
            if not makeAlias(self.ebqe,k):
                self.ebqe[k]=numpy.zeros(
                    (self.mesh.nExteriorElementBoundaries_global,
                     self.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.nDOF_test_element[k[-1]]),
                    'd')
        logEvent(memory("global exterior element boundary quadrature, test/trial functions test_shape","OneLevelTransport"),level=4)
        #allocate trial shape gradient element boundary quadrature
        for k in sorted(self.trial_shapeGradient_elementBoundaryQuadrature):
            if not makeAliasForComponent1(self.ebqe,k,['grad(v)'],refi=0):
                self.ebqe[k]=numpy.zeros(
                (self.mesh.nExteriorElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nDOF_trial_element[k[-1]],
                 self.nSpace_global),
                'd')
        logEvent(memory("global exterior element boundary quadrature, test/trial functions trial_shapeGradient","OneLevelTransport"),level=4)
        #allocate phi trial shape gradient element boundary quadrature
        for k in sorted(self.phi_trial_shapeGradient_elementBoundaryQuadrature):
            #if not makeAlias(self.ebqe,k):
            if not makeAliasForComponent1(self.ebqe,k,['grad(v)'],refi=refComponent_phi):
                self.ebqe[k]=numpy.zeros(
                    (self.mesh.nExteriorElementBoundaries_global,
                     self.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.nDOF_phi_trial_element[k[-1]],
                     self.nSpace_global),
                    'd')
        logEvent(memory("global exterior element boundary quadrature, test/trial functions phi_trial_shapeGradient","OneLevelTransport"),level=4)
        #allocate trial shape X test shape element boundary quadratre
        if not self.lowmem:
            for k in self.trial_shape_X_test_shape_elementBoundaryQuadrature:
                self.ebqe[k]=numpy.zeros(
                    (self.mesh.nExteriorElementBoundaries_global,
                     self.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.nDOF_trial_element[k[1]],
                     self.nDOF_test_element[k[2]]),
                    'd')
            logEvent(memory("global exterior element boundary quadrature, test/trial functions trial_shape_X_test_shape","OneLevelTransport"),level=4)
        logEvent("Dumping quadrature shapes for model %s" % self.name,level=9)
        logEvent("Element quadrature array (q)", level=9)
        for (k,v) in self.q.items(): logEvent(str((k,v.shape)),level=9)
        logEvent("Element boundary quadrature (ebq)",level=9)
        for (k,v) in self.ebq.items(): logEvent(str((k,v.shape)),level=9)
        logEvent("Global element boundary quadrature (ebq_global)",level=9)
        for (k,v) in self.ebq_global.items(): logEvent(str((k,v.shape)),level=9)
        logEvent("Exterior element boundary quadrature (ebqe)",level=9)
        for (k,v) in self.ebqe.items(): logEvent(str((k,v.shape)),level=9)
        logEvent("Interpolation points for nonlinear diffusion potential (phi_ip)",level=9)
        for (k,v) in self.phi_ip.items(): logEvent(str((k,v.shape)),level=9)
        #
        # allocate residual and Jacobian storage
        #
        self.elementResidual = [numpy.zeros(
                (self.mesh.nElements_global,
                 self.nDOF_test_element[ci]),
                'd') for ci in range(self.nc)]
        self.elementSpatialResidual = [numpy.zeros(
                (self.mesh.nElements_global,
                 self.nDOF_test_element[ci]),
                'd') for ci in range(self.nc)]
        self.elementJacobian = {}
        self.elementJacobian_eb = {}
        for ci in range(self.nc):
            self.elementJacobian[ci]={}
            self.elementJacobian_eb[ci]={}
            for cj in range(self.nc):
                if cj in self.coefficients.stencil[ci]:
                    self.elementJacobian[ci][cj] = numpy.zeros(
                        (self.mesh.nElements_global,
                         self.nDOF_test_element[ci],
                         self.nDOF_trial_element[cj]),
                        'd')
                    self.elementJacobian_eb[ci][cj] = numpy.zeros(
                        (self.mesh.nElements_global,
                         self.mesh.nElementBoundaries_element,
                         self.nDOF_test_element[ci],
                         self.nDOF_trial_element[cj]),
                        'd')
        logEvent(memory("element Jacobian","OneLevelTransport"),level=4)
        self.fluxJacobian = {}
        self.fluxJacobian_eb = {}
        self.fluxJacobian_exterior = {}
        self.fluxJacobian_hj = {}
        self.inflowFlag={}
        for ci in range(self.nc):
            self.fluxJacobian[ci]={}
            self.fluxJacobian_eb[ci]={}
            self.fluxJacobian_exterior[ci]={}
            self.fluxJacobian_hj[ci]={}
            for cj in self.coefficients.stencil[ci]:
                if (self.fluxBoundaryConditions[ci] == 'setFlow' or
                    self.fluxBoundaryConditions[ci] == 'outFlow' or
                    self.fluxBoundaryConditions[ci] == 'mixedFlow' or True):#cek hack
                    #
                    self.fluxJacobian_exterior[ci][cj] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nDOF_trial_element[cj]),
                        'd')

                    self.inflowFlag[ci] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary),
                        'i')
                    self.ebqe[('dadvectiveFlux_left',ci,cj)] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                        self.nElementBoundaryQuadraturePoints_elementBoundary),
                        'd')
                    self.ebqe[('dadvectiveFlux_right',ci,cj)] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                        self.nElementBoundaryQuadraturePoints_elementBoundary),
                        'd')

                if numericalFluxType is not None:
                    self.fluxJacobian_exterior[ci][cj] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nDOF_trial_element[cj]),
                        'd')
                    self.ebqe[('dadvectiveFlux_left',ci,cj)] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary),
                        'd')
                    self.ebqe[('dadvectiveFlux_right',ci,cj)] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary),
                        'd')
                    #mwf hack
                    if numericalFluxType.hasInterior:
                        self.fluxJacobian[ci][cj] = numpy.zeros(
                            (self.mesh.nElementBoundaries_global,
                             2,
                             self.nElementBoundaryQuadraturePoints_elementBoundary,
                             self.nDOF_trial_element[cj]),
                            'd')
                        self.fluxJacobian_eb[ci][cj] = numpy.zeros(
                            (self.mesh.nElementBoundaries_global,
                             2,
                             self.mesh.nElementBoundaries_element,
                             self.nElementBoundaryQuadraturePoints_elementBoundary,
                             self.nDOF_trial_element[cj]),
                            'd')

                        self.ebq_global[('dadvectiveFlux_left',ci,cj)] = numpy.zeros(
                            (self.mesh.nElementBoundaries_global,
                             self.nElementBoundaryQuadraturePoints_elementBoundary),
                            'd')
                        self.ebq_global[('dadvectiveFlux_right',ci,cj)] = numpy.zeros(
                            (self.mesh.nElementBoundaries_global,
                             self.nElementBoundaryQuadraturePoints_elementBoundary),
                            'd')
                    #mwf TODO use fluxJacobian_eb for HamiltonJacobi - two-sided flux
                    #mwf TODO check convention on flux derivatives, should be (cj,ci) to be consistent?
                    #mwf hack
                    #treat as 1-sided flux on exterior boundary
                    self.ebqe[('dHamiltonJacobiFlux_left',ci,cj)] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary),
                        'd')
                    self.ebqe[('dHamiltonJacobiFlux_right',ci,cj)] = numpy.zeros(
                        (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary),
                        'd')
                    if numericalFluxType.hasInterior:
                        self.ebq[('dHamiltonJacobiFlux_left',ci,cj)] = numpy.zeros(
                            (self.mesh.nElements_global,
                             self.mesh.nElementBoundaries_element,
                             self.nElementBoundaryQuadraturePoints_elementBoundary),
                            'd')
                        self.ebq[('dHamiltonJacobiFlux_right',ci,cj)] = numpy.zeros(
                            (self.mesh.nElements_global,
                             self.mesh.nElementBoundaries_element,
                             self.nElementBoundaryQuadraturePoints_elementBoundary),
                            'd')
                        self.fluxJacobian_hj[ci][cj] = numpy.zeros(
                            (self.mesh.nElementBoundaries_global,
                             2,#left flux or right flux
                             2,#left and right neighbor
                             self.nElementBoundaryQuadraturePoints_elementBoundary,
                             self.nDOF_trial_element[cj]),
                            'd')
                  #should be able to reusefluxJacobian_exterior since only 1 flux at boundary
        #
        # Build the node connectivity lists for solvers
        #
        # \todo fix for DG and multicomponent
        #
        #cek not used right now
#         self.mesh.buildNodeStarArray()
#         self.freeNodeStarList=[]
#         for n in range(self.mesh.nNodes_global):
#             self.freeNodeStarList.append([])
#         for n in range(self.mesh.nNodes_global):
#             if self.dirichletConditions[0].global2freeGlobal.has_key(n):
#                 N = self.dirichletConditions[0].global2freeGlobal[n]
#                 self.mesh.nodeStarList[n].sort()
#                 for m in self.mesh.nodeStarList[n]:
#                     if self.dirichletConditions[0].global2freeGlobal.has_key(m):
#                         M = self.dirichletConditions[0].global2freeGlobal[m]
#                         self.freeNodeStarList[N].append(M)
        #
        #
        #
        #
        logEvent(memory("element and element boundary Jacobians","OneLevelTransport"),level=4)
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,),'i')
            self.inflowBoundaryBC_values[cj] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,self.nDOF_trial_element[cj]),'d')
            self.inflowFlux[cj] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
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
        self.internalNodesArray=numpy.zeros((self.nNodes_internal,),cmeshTools.INDEX_DTYPE)
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

        if options is not None:
            self.timeIntegration.setFromOptions(options)
        logEvent(memory("TimeIntegration","OneLevelTransport"),level=4)
        logEvent("Calculating numerical quadrature formulas",2)
        self.calculateQuadrature()

        comm = Comm.get()
        self.comm=comm
        if comm.size() > 1:
            assert numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions,"You must use a numerical flux to apply weak boundary conditions for parallel runs"

        if numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions:
            interleave_DOF=True
            for nDOF_trial_element_ci in self.nDOF_trial_element:
                if nDOF_trial_element_ci != self.nDOF_trial_element[0]:
                    interleave_DOF=False
            # Interleaving needs the same number of unknowns in every
            # component. Equal spaces give that, unless periodic pairs are
            # merged differently: a component that is Dirichlet at a
            # periodic corner keeps both copies there, one that is not
            # merges them.
            if len(set(self.nFreeDOF_global)) > 1:
                interleave_DOF=False
        else:
            interleave_DOF=False
        self.setupFieldStrides(interleaved=interleave_DOF)

        logEvent(memory("stride+offset","OneLevelTransport"),level=4)
        if numericalFluxType is not None:
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
            if options is not None:
                self.numericalFlux.setFromOptions(options)
        else:
            self.numericalFlux = None
        self._refuseDiscontinuousSpaceWithoutFaceNeighbours()
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
        from . import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)
        logEvent(memory("velocity postprocessor","OneLevelTransport"),level=4)
        #helper for writing out data storage
        from . import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        #for model reduction
        self.mass_jacobian = None
        self.space_jacobian = None
        self.nzval_mass = None
        self.nzval_space = None
    #end __init__

    def _refuseDiscontinuousSpaceWithoutFaceNeighbours(self):
        """Raise when a discontinuous space coupled by a numerical flux sits on a partition missing face neighbours.

        A DG face term couples an element to its face neighbour. If a face of an owned element is a rank interface
        (the partitioner flags the faces it cuts -1), the neighbour is not on the rank and the face is assembled as
        domain boundary, so the answer depends on the partition: 16x16 DG Poisson, 1.04e-3 serial, 6.3e-2 on 2 ranks
        with an element partition and no overlap, 1.3e-1 with a node partition and one layer. An element partition
        with one layer has none. LDG needs two: 13x13 LDG P1 on 4 ranks gave an L2
        velocity error of 0.113549 with one layer against 0.113510 serial, and 0.113510 (1e-15) with two. Every rank
        takes part in the sum and every rank raises.
        """
        comm = Comm.get()
        nf = self.numericalFlux
        if comm.size() == 1 or nf is None or not getattr(nf, 'hasInterior', True):
            return
        if isinstance(nf, NumericalFlux.NoFlux):
            return              # zero interior flux: the elements are not coupled through their faces
        if not any(isinstance(u.femSpace.dofMap, (DiscontinuousGalerkinDOFMap, p0DOFMap)) for u in self.u.values()):
            return
        mesh = self.mesh
        marked = int(bool(mesh.__dict__.get('rankInterfacesMarked')))
        if comm.globalSum(marked) < comm.size():
            return              # a partitioner that does not mark its cuts: nothing to conclude
        exterior = np.asarray(mesh.exteriorElementBoundariesArray)
        cut = exterior[MeshTools.rankInterfaceExteriorMask(mesh)]
        # LDG (mixed-form diffusion): the flux through an owned element's face reads the velocity of the neighbour,
        # which is solved from the neighbour's own faces, so the neighbours' neighbours must be on the rank too
        mixed = any(getattr(nf, 'mixedDiffusion', {}).values())
        needed = np.zeros(mesh.nElements_global, bool)
        needed[:mesh.nElements_owned] = True
        if mixed:
            neighbours = np.asarray(mesh.elementNeighborsArray)[:mesh.nElements_owned].ravel()
            needed[neighbours[neighbours >= 0]] = True
        owned = needed[np.asarray(mesh.elementBoundaryElementsArray)[cut, 0]]
        missing = int(comm.globalSum(int(np.count_nonzero(owned))))
        if missing > 0:
            layers = 2 if mixed else 1
            raise RuntimeError(
                "%s: %d faces of %s are rank interfaces, so the discontinuous space coupled by %s would "
                "treat them as domain boundary and the solution would depend on the partition. Use an element "
                "partition with %d layer%s of overlap (parallelPartitioningType = MeshParallelPartitioningTypes.element, "
                "nLayersOfOverlapForParallel = %d, in the n module and in the domain's MeshOptions)."
                % (self.name, missing, "owned elements or their face neighbours (LDG)" if mixed else "owned elements",
                   nf.__class__.__name__, layers, "s" if layers > 1 else "", layers))

    def scatterDOFArray(self, a):
        """Replace the ghost entries of a dof array of component 0's space by the owners' values (in place).

        Kernels write dof-shaped arrays (limiters, smoothness indicators, entropy residuals, interface locators,
        normals) over the elements a rank holds, so their ghost entries are partial; a later kernel that reads them
        at ghost dofs needs the owners' values. One ghosted vector over a buffer of its own per array shape: callers
        pass temporaries, and a vector cached per array kept every one of them alive (1080 per rank after 0.08 s of
        an NCLS run)."""
        comm = getattr(self, "comm", None) or Comm.get()
        if comm.size() == 1:
            return
        vecs = self.__dict__.setdefault("_dofVecs", {})
        cached = vecs.get(a.shape)
        if cached is None:
            dofMap = self.u[0].femSpace.dofMap
            rank = comm.rank()
            nOwned = dofMap.dof_offsets_subdomain_owned[rank + 1] - dofMap.dof_offsets_subdomain_owned[rank]
            buf = a.copy()
            cached = vecs[a.shape] = (buf, ParVec_petsc4py(
                buf, 1, nOwned, dofMap.nDOF_all_processes, dofMap.nDOF_subdomain - nOwned, dofMap.subdomain2global))
        buf, par = cached
        buf[:] = a
        par.scatter_forward_insert()
        a[:] = buf

    def setupFieldStrides(self, interleaved=True):
        """
        Set up the stride/offset layout for this object.  The default layout is interleaved.
        """

        if interleaved:
            self.offset = list(range(self.nc))
            self.stride = [self.nc] * self.nc
        else:
            self.offset = [0]
            for ci in range(1,self.nc):
                self.offset += [self.offset[ci-1]+self.nFreeDOF_global[ci-1]]
            self.stride = [1] * self.nc

    def setInitialConditions(self,getInitialConditionsDict,T=0.0):
        self.timeIntegration.t = T
        #
        #set the initial conditions for the DOF based on the generalized interpolation conditions
        #
        for cj in range(self.nc):
            interpolationValues = numpy.zeros((self.mesh.nElements_global,
                                                 self.u[cj].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints),
                                                'd')
            try:
                for eN in range(self.mesh.nElements_global):
                    materialFlag = self.mesh.elementMaterialTypes[eN]
                    for k in range(self.u[cj].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints):
                        interpolationValues[eN,k] = getInitialConditionsDict[cj].uOfXT(self.u[cj].femSpace.interpolationPoints[eN,k],T,materialFlag)
            except TypeError:
                for eN in range(self.mesh.nElements_global):
                    for k in range(self.u[cj].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints):
                        interpolationValues[eN,k] = getInitialConditionsDict[cj].uOfXT(self.u[cj].femSpace.interpolationPoints[eN,k],T)
            self.u[cj].projectFromInterpolationConditions(interpolationValues)
        #Load the Dirichlet conditions
        for cj in range(self.nc):
            for dofN,g in self.dirichletConditions[cj].DOFBoundaryConditionsDict.items():
                self.u[cj].dof[dofN] = g(self.dirichletConditions[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
    #what about setting initial conditions directly from dofs calculated elsewhere?

    def setInitialConditionsTPF(self,getInitialConditionsDict,idxDict,T=0.0):
        self.timeIntegration.t = T
        #
        #set the initial conditions for the DOF based on the generalized interpolation conditions
        #
        #for key in getInitialConditionsDict.keys():
        for key in idxDict.keys():
            entry = getInitialConditionsDict[key]
            if(entry is None):
                continue
            idx = idxDict[key]
            interpolationValues = numpy.zeros((self.mesh.nElements_global,
                                                 self.u[idx].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints),
                                                'd')
            try:
                for eN in range(self.mesh.nElements_global):
                    materialFlag = self.mesh.elementMaterialTypes[eN]
                    for k in range(self.u[idx].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints):
                        interpolationValues[eN,k] = getInitialConditionsDict[key].uOfXT(self.u[idx].femSpace.interpolationPoints[eN,k],T,materialFlag)
            except TypeError:
                for eN in range(self.mesh.nElements_global):
                    for k in range(self.u[idx].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints):
                        interpolationValues[eN,k] = getInitialConditionsDict[key].uOfXT(self.u[idx].femSpace.interpolationPoints[eN,k],T)
            self.u[idx].projectFromInterpolationConditions(interpolationValues)
        #Load the Dirichlet conditions
        for cj in range(self.nc):
            for dofN,g in self.dirichletConditions[idx].DOFBoundaryConditionsDict.items():
                self.u[idx].dof[dofN] = g(self.dirichletConditions[idx].DOFBoundaryPointDict[dofN],self.timeIntegration.t)

    def archiveAnalyticalSolutions(self,archive,analyticalSolutionsDict,T=0.0,tCount=0):
        import copy
        #
        #set the initial conditions for the DOF based on the generalized interpolation conditions
        #
        if analyticalSolutionsDict is None:
            return
        for cj,sol in analyticalSolutionsDict.items():
            if not hasattr(self,'u_save'):
                self.u_save = {}
                self.u_analytical = {}
            if cj not in self.u_save:
                self.u_save[cj] = (self.u[cj].dof.copy(), self.u[cj].name)
                self.u_analytical[cj] = (self.u[cj].dof.copy(), self.u[cj].name+'_analytical')
            else:
                self.u_save[cj][0][:] = self.u[cj].dof
            interpolationValues = numpy.zeros((self.mesh.nElements_global,
                                               self.u[cj].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints),
                                              'd')
            try:
                for eN in range(self.mesh.nElements_global):
                    materialFlag = self.mesh.elementMaterialTypes[eN]
                    for k in range(self.u[cj].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints):
                        interpolationValues[eN,k] = sol.uOfXT(self.u[cj].femSpace.interpolationPoints[eN,k],T,materialFlag)
            except TypeError:
                for eN in range(self.mesh.nElements_global):
                    for k in range(self.u[cj].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints):
                        interpolationValues[eN,k] = sol.uOfXT(self.u[cj].femSpace.interpolationPoints[eN,k],T)
            self.u[cj].projectFromInterpolationConditions(interpolationValues)
            self.u[cj].name += '_analytical'
            self.u[cj].femSpace.writeFunctionXdmf(archive,self.u[cj],tCount)
            self.u_analytical[cj][0][:] = self.u[cj].dof
            self.u[cj].dof[:] = self.u_save[cj][0]
            self.u[cj].name = self.u_save[cj][1]
    #what about setting initial conditions directly from dofs calculated elsewhere?
    def setInitialConditionsFromDofs(self,getInitialConditionDofs,T=0.0):
        self.timeIntegration.t=T
        for cj in range(self.nc):
            for eN in range(self.mesh.nElements_global):
                for j in self.u[cj].femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                    jg = self.u[cj].femSpace.dofMap.l2g[eN,j]
                    self.u[cj].dof[jg]=getInitialConditionDofs.getICDofs(cj)[jg]
                #end j
            #end eN
        #end cj
        #self.calculateCoefficients()#unremoved
        #cek remove to make behavior consistent with other setInitialConditions...
    #end def
    def initializeTimeIntegration(self,t0,tOut,u0,r0,DTSET=None):
        self.timeIntegration.t=t0
        self.getResidual(u0,r0)
        self.estimate_mt()
        if DTSET is not None:
            self.timeIntegration.chooseInitial_dt(t0,tOut,self.q)
            self.timeIntegration.set_dt(DTSET)
        else:
            self.timeIntegration.chooseInitial_dt(t0,tOut,self.q)
        self.timeIntegration.setInitialStageValues()
        self.timeIntegration.initializeTimeHistory(resetFromDOF=True)
        if self.stabilization is not None:
            self.stabilization.updateSubgridErrorHistory(initializationPhase=True)#mwf added arg for tracking subscales
        if self.shockCapturing is not None:
            self.shockCapturing.updateShockCapturingHistory()
    def initializeTimeHistory(self):
        if self.stabilization is not None:
            self.stabilization.updateSubgridErrorHistory(initializationPhase=True)#mwf added arg for tracking subscales
        if self.shockCapturing is not None:
            self.shockCapturing.updateShockCapturingHistory()
    def updateTimeHistory(self,T,resetFromDOF=False):
        """
        put a version on each level rather than just multilevel?
        """
#         self.timeIntegration.t=T
#         self.timeIntegration.updateTimeHistory(resetFromDOF)
        if self.stabilization is not None:
            self.stabilization.updateSubgridErrorHistory()
        if self.shockCapturing is not None:
            self.shockCapturing.updateShockCapturingHistory()
            #ci
        #if
    def calculateAuxiliaryQuantitiesAfterStep(self):
        if self.conservativeFlux is not None and self.velocityPostProcessor is not None:
            if self.movingDomain:
                for ci in range(self.nc):
                    try:
                        if ci in self.velocityPostProcessor.updateConservationJacobian:
                            self.velocityPostProcessor.updateConservationJacobian[ci]=True
                    except:
                        pass
            self.velocityPostProcessor.postprocess(verbose=0)
#         if self.movingDomain:
#             for ci in range(self.nc):
#                 if self.q.has_key(('velocity',ci)):
#                     for eN in range(self.mesh.nElements_global):
#                         for k in range(self.nQuadraturePoints_element):
#                             for I in range(self.nSpace_global):
#                                 self.q[('velocity',ci)][eN,k,I] += self.q['xt'][eN,k,I]
#                     for eN in range(self.mesh.nElements_global):
#                         for ebN in range(self.mesh.nElementBoundaries_element):
#                             for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                 for I in range(self.nSpace_global):
#                                     self.ebq[('velocity',ci)][eN,ebN,k,I]  += self.ebq['xt'][eN,ebN,k,I]
#                     for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
#                         for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                             for I in range(self.nSpace_global):
#                                 self.ebqe[('velocity',ci)][ebNE,k,I] += self.ebqe['xt'][ebNE,k,I]
    def forcedDOF_global(self,cj,dofN):
        """The global row (and column) of a kept Dirichlet DOF of component cj.

        Its free number, not its DOF number: the two differ when periodic
        DOFs are merged into one unknown, and are the same otherwise.
        """
        return self.offset[cj]+self.stride[cj]*self.dirichletConditions[cj].global2freeGlobal[dofN]
    def getResidual(self,u,r):
        """
        Calculate the element residuals and add in to the global residual
        """
        #cek debug
        #u.tofile("u"+`self.nonlinear_function_evaluations`,sep="\n")
        r.fill(0.0)
        for cj in range(self.nc):
            for dofN,g in self.dirichletConditions[cj].DOFBoundaryConditionsDict.items():
                self.u[cj].dof[dofN] = g(self.dirichletConditions[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN,g in self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items():
                    u[self.forcedDOF_global(cj,dofN)] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)#load the BC value directly into the global array
        #Load the unknowns into the finite element dof
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        self.calculateCoefficients()
        self.calculateElementResidual()
        self.scale_dt = False
        logEvent("Element residual",level=9,data=self.elementResidual)
        if self.timeIntegration.dt < 1.0e-8*self.mesh.h:
            self.scale_dt = True
            logEvent("Rescaling residual for small time steps")
        else:
            self.scale_dt = False
        for ci in range(self.nc):
            if self.scale_dt:
                self.elementResidual[ci]*=self.timeIntegration.dt
            cfemIntegrals.updateGlobalResidualFromElementResidual(self.offset[ci],
                                                                  self.stride[ci],
                                                                  self.l2g[ci]['nFreeDOF'],
                                                                  self.l2g[ci]['freeLocal'],
                                                                  self.l2g[ci]['freeGlobal'],
                                                                  self.elementResidual[ci],
                                                                  r);
        logEvent("Global residual",level=9,data=r)
        if self.forceStrongConditions:#
            for cj in range(len(self.dirichletConditionsForceDOF)):#
                for dofN,g in self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items():
                    r[self.forcedDOF_global(cj,dofN)] = self.u[cj].dof[dofN] - g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
        #for keeping solver statistics
        self.nonlinear_function_evaluations += 1
        #cek debug
        #r.tofile("residual"+`self.nonlinear_function_evaluations`,sep="\n")
        #mwf debug
        #imax = numpy.argmax(r); imin = numpy.argmin(r)
        #print "getResidual max,index r[%s]= %s min,index= r[%s] r= %s " % (imax,r[imax],imin,r[imin])
    def getJacobian(self,jacobian,skipMassTerms=False):
        ##\todo clean up update,calculate,get,intialize usage
        self.calculateElementBoundaryJacobian()
        self.calculateExteriorElementBoundaryJacobian()
        self.calculateElementJacobian(skipMassTerms=skipMassTerms)
        if self.scale_dt:
            for ci in list(self.elementJacobian.keys()):
                for cj in list(self.elementJacobian[ci].keys()):
                    self.elementJacobian[ci][cj] *= self.timeIntegration.dt
            for ci in list(self.elementJacobian_eb.keys()):
                for cj in list(self.elementJacobian_eb[ci].keys()):
                    self.elementJacobian_eb[ci][cj] *= self.timeIntegration.dt
            for ci in list(self.fluxJacobian.keys()):
                for cj in list(self.fluxJacobian[ci].keys()):
                    self.fluxJacobian[ci][cj] *= self.timeIntegration.dt
            for ci in list(self.fluxJacobian_eb.keys()):
                for cj in list(self.fluxJacobian_eb[ci].keys()):
                    self.fluxJacobian_eb[ci][cj] *= self.timeIntegration.dt
            for ci in list(self.fluxJacobian_exterior.keys()):
                for cj in list(self.fluxJacobian_exterior[ci].keys()):
                    self.fluxJacobian_exterior[ci][cj] *= self.timeIntegration.dt
            for ci in list(self.fluxJacobian_hj.keys()):
                for cj in list(self.fluxJacobian_hj[ci].keys()):
                    self.fluxJacobian_hj[ci][cj] *= self.timeIntegration.dt
        logEvent("Element Jacobian ",level=10,data=self.elementJacobian)
        if self.matType == superluWrappers.SparseMatrix:
            self.getJacobian_CSR(jacobian)
        elif self.matType  == numpy.array:
            self.getJacobian_dense(jacobian)
        else:
            raise TypeError("Matrix type must be SparseMatrix or array")
        if self.forceStrongConditions:
            for cj in range(self.nc):
                for dofN in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.keys()):
                    global_dofN = self.forcedDOF_global(cj,dofN)
                    self.nzval[numpy.where(self.colind == global_dofN)] = 0.0 #column
                    self.nzval[self.rowptr[global_dofN]:self.rowptr[global_dofN+1]] = 0.0 #row
                    zeroRow=True
                    for i in range(self.rowptr[global_dofN],self.rowptr[global_dofN+1]):#row
                        if (self.colind[i] == global_dofN):
                            self.nzval[i] = 1.0
                            zeroRow = False
                    if zeroRow:
                        raise RuntimeError("Jacobian has a zero row because sparse matrix has no diagonal entry at row "+repr(global_dofN)+". You probably need add diagonal mass or reaction term")
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian
    def getJacobian_dense(self,jacobian):
        import copy
        jacobian.fill(0.0)
        for ci in range(self.nc):
            for cj in self.coefficients.stencil[ci]:
                #
                #element contributions
                #
                cfemIntegrals.updateGlobalJacobianFromElementJacobian_dense(self.offset[ci],
                                                                            self.stride[ci],
                                                                            self.offset[cj],
                                                                            self.stride[cj],
                                                                            self.nFreeVDOF_global,
                                                                            self.l2g[ci]['nFreeDOF'],
                                                                            self.l2g[ci]['freeLocal'],
                                                                            self.l2g[ci]['freeGlobal'],
                                                                            self.l2g[cj]['nFreeDOF'],
                                                                            self.l2g[cj]['freeLocal'],
                                                                            self.l2g[cj]['freeGlobal'],
                                                                            self.elementJacobian[ci][cj],
                                                                            jacobian)
                if self.numericalFlux is not None and self.numericalFlux.mixedDiffusion[ci] == True:
                    cfemIntegrals.updateGlobalJacobianFromElementJacobian_eb_dense(self.mesh.elementNeighborsArray,
                                                                                   self.offset[ci],
                                                                                   self.stride[ci],
                                                                                   self.offset[cj],
                                                                                   self.stride[cj],
                                                                                   self.nFreeVDOF_global,
                                                                                   self.l2g[ci]['nFreeDOF'],
                                                                                   self.l2g[ci]['freeLocal'],
                                                                                   self.l2g[ci]['freeGlobal'],
                                                                                   self.l2g[cj]['nFreeDOF'],
                                                                                   self.l2g[cj]['freeLocal'],
                                                                                   self.l2g[cj]['freeGlobal'],
                                                                                   self.elementJacobian_eb[ci][cj],
                                                                                   jacobian)
                #
                #element boundary contributions
                #
                if self.numericalFlux is not None and type(self.numericalFlux) != NumericalFlux.DoNothing:
                    if self.numericalFlux.hasInterior:
                        cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_dense(self.offset[ci],
                                                                                                        self.stride[ci],
                                                                                                        self.offset[cj],
                                                                                                        self.stride[cj],
                                                                                                        self.nFreeVDOF_global,
                                                                                                        self.mesh.interiorElementBoundariesArray,
                                                                                                        self.mesh.elementBoundaryElementsArray,
                                                                                                        self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                        self.l2g[ci]['nFreeDOF'],
                                                                                                        self.l2g[ci]['freeLocal'],
                                                                                                        self.l2g[ci]['freeGlobal'],
                                                                                                        self.l2g[cj]['nFreeDOF'],
                                                                                                        self.l2g[cj]['freeLocal'],
                                                                                                        self.l2g[cj]['freeGlobal'],
                                                                                                        self.fluxJacobian[ci][cj],
                                                                                                        self.ebq[('w*dS_f',ci)],
                                                                                                        jacobian)
                        if self.numericalFlux.mixedDiffusion[ci] == True:
                            cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_dense(self.mesh.elementNeighborsArray,
                                                                                                               self.mesh.nElements_global,
                                                                                                               self.offset[ci],
                                                                                                               self.stride[ci],
                                                                                                               self.offset[cj],
                                                                                                               self.stride[cj],
                                                                                                               self.nFreeVDOF_global,
                                                                                                               self.mesh.interiorElementBoundariesArray,
                                                                                                               self.mesh.elementBoundaryElementsArray,
                                                                                                               self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                               self.l2g[ci]['nFreeDOF'],
                                                                                                               self.l2g[ci]['freeLocal'],
                                                                                                               self.l2g[ci]['freeGlobal'],
                                                                                                               self.l2g[cj]['nFreeDOF'],
                                                                                                               self.l2g[cj]['freeLocal'],
                                                                                                               self.l2g[cj]['freeGlobal'],
                                                                                                               self.fluxJacobian_eb[ci][cj],
                                                                                                               self.ebq[('w*dS_f',ci)],
                                                                                                               jacobian)
                        if self.numericalFlux.HamiltonJacobiNumericalFlux[ci] == True:
                            cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_dense(self.offset[ci],
                                                                                                                   self.stride[ci],
                                                                                                                   self.offset[cj],
                                                                                                                   self.stride[cj],
                                                                                                                   self.nFreeVDOF_global,
                                                                                                                   self.mesh.interiorElementBoundariesArray,
                                                                                                                   self.mesh.elementBoundaryElementsArray,
                                                                                                                   self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                                   self.l2g[ci]['nFreeDOF'],
                                                                                                                   self.l2g[ci]['freeLocal'],
                                                                                                                   self.l2g[ci]['freeGlobal'],
                                                                                                                   self.l2g[cj]['nFreeDOF'],
                                                                                                                   self.l2g[cj]['freeLocal'],
                                                                                                                   self.l2g[cj]['freeGlobal'],
                                                                                                                   self.fluxJacobian_hj[ci][cj],
                                                                                                                   self.ebq[('w*dS_f',ci)],
                                                                                                                   jacobian)
                    cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_dense(self.offset[ci],
                                                                                                    self.stride[ci],
                                                                                                    self.offset[cj],
                                                                                                    self.stride[cj],
                                                                                                    self.nFreeVDOF_global,
                                                                                                    self.mesh.exteriorElementBoundariesArray,
                                                                                                    self.mesh.elementBoundaryElementsArray,
                                                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                    self.l2g[ci]['nFreeDOF'],
                                                                                                    self.l2g[ci]['freeLocal'],
                                                                                                    self.l2g[ci]['freeGlobal'],
                                                                                                    self.l2g[cj]['nFreeDOF'],
                                                                                                    self.l2g[cj]['freeLocal'],
                                                                                                    self.l2g[cj]['freeGlobal'],
                                                                                                    self.fluxJacobian_exterior[ci][cj],
                                                                                                    self.ebqe[('w*dS_f',ci)],
                                                                                                    jacobian)
                    if self.numericalFlux.mixedDiffusion[ci] == True:
                        cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_dense(self.mesh.elementNeighborsArray,
                                                                                                           self.mesh.nElements_global,
                                                                                                           self.offset[ci],
                                                                                                           self.stride[ci],
                                                                                                           self.offset[cj],
                                                                                                           self.stride[cj],
                                                                                                           self.nFreeVDOF_global,
                                                                                                           self.mesh.exteriorElementBoundariesArray,
                                                                                                           self.mesh.elementBoundaryElementsArray,
                                                                                                           self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                           self.l2g[ci]['nFreeDOF'],
                                                                                                           self.l2g[ci]['freeLocal'],
                                                                                                           self.l2g[ci]['freeGlobal'],
                                                                                                           self.l2g[cj]['nFreeDOF'],
                                                                                                           self.l2g[cj]['freeLocal'],
                                                                                                           self.l2g[cj]['freeGlobal'],
                                                                                                           self.fluxJacobian_eb[ci][cj],
                                                                                                           self.ebqe[('w*dS_f',ci)],
                                                                                                           jacobian)
                else:
                    #cek these will be gone
                    if (( self.fluxBoundaryConditions[ci] == 'outFlow' or
                          self.fluxBoundaryConditions[ci] == 'mixedFlow')
                        and
                        self.timeIntegration.advectionIsImplicit[ci]):
                        cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_dense(self.offset[ci],
                                                                                                        self.stride[ci],
                                                                                                        self.offset[cj],
                                                                                                        self.stride[cj],
                                                                                                        self.nFreeVDOF_global,
                                                                                                        self.mesh.exteriorElementBoundariesArray,
                                                                                                        self.mesh.elementBoundaryElementsArray,
                                                                                                        self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                        self.l2g[ci]['nFreeDOF'],
                                                                                                        self.l2g[ci]['freeLocal'],
                                                                                                        self.l2g[ci]['freeGlobal'],
                                                                                                        self.l2g[cj]['nFreeDOF'],
                                                                                                        self.l2g[cj]['freeLocal'],
                                                                                                        self.l2g[cj]['freeGlobal'],
                                                                                                        self.fluxJacobian_exterior[ci][cj],
                                                                                                        self.ebqe[('w*dS_f',ci)],
                                                                                                        jacobian)
                    #mwf TODO can't get here in logic?
                    if self.numericalFlux is not None and self.numericalFlux.mixedDiffusion[ci] == True:
                        #
                        cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_dense(self.mesh.elementNeighborsArray,
                                                                                                           self.mesh.nElements_global,
                                                                                                           self.offset[ci],
                                                                                                           self.stride[ci],
                                                                                                           self.offset[cj],
                                                                                                           self.stride[cj],
                                                                                                           self.nFreeVDOF_global,
                                                                                                           self.mesh.exteriorElementBoundariesArray,
                                                                                                           self.mesh.elementBoundaryElementsArray,
                                                                                                           self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                           self.l2g[ci]['nFreeDOF'],
                                                                                                           self.l2g[ci]['freeLocal'],
                                                                                                           self.l2g[ci]['freeGlobal'],
                                                                                                           self.l2g[cj]['nFreeDOF'],
                                                                                                           self.l2g[cj]['freeLocal'],
                                                                                                           self.l2g[cj]['freeGlobal'],
                                                                                                           self.fluxJacobian_eb[ci][cj],
                                                                                                           self.ebqe[('w*dS_f',ci)],
                                                                                                           jacobian)
        #
        #element boundary contributions from diffusion
        #
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck in list(ckDict.keys()):
                if self.numericalFlux.includeBoundaryAdjoint:
                    if self.sd:
                        if self.numericalFlux.hasInterior:
                            cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                                                   self.offset[ci],
                                                                                                                   self.stride[ci],
                                                                                                                   self.offset[ck],
                                                                                                                   self.stride[ck],
                                                                                                                   self.nFreeVDOF_global,
                                                                                                                   self.mesh.interiorElementBoundariesArray,
                                                                                                                   self.mesh.elementBoundaryElementsArray,
                                                                                                                   self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                                   self.l2g[ci]['nFreeDOF'],
                                                                                                                   self.l2g[ci]['freeLocal'],
                                                                                                                   self.l2g[ci]['freeGlobal'],
                                                                                                                   self.l2g[ck]['nFreeDOF'],
                                                                                                                   self.l2g[ck]['freeLocal'],
                                                                                                                   self.l2g[ck]['freeGlobal'],
                                                                                                                   self.numericalFlux.boundaryAdjoint_sigma,
                                                                                                                   self.ebq[('v',ck)],
                                                                                                                   self.ebq['n'],
                                                                                                                   self.ebq[('a',ci,ck)],
                                                                                                                   self.ebq[('grad(v)',ci)],
                                                                                                                   self.ebq[('dS_u',ck)],
                                                                                                                   jacobian)
                        if not self.numericalFlux.includeBoundaryAdjointInteriorOnly: #added to only eval interior tjp
                            cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                                               self.offset[ci],
                                                                                                               self.stride[ci],
                                                                                                               self.offset[ck],
                                                                                                               self.stride[ck],
                                                                                                               self.nFreeVDOF_global,
                                                                                                               self.mesh.exteriorElementBoundariesArray,
                                                                                                               self.mesh.elementBoundaryElementsArray,
                                                                                                               self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                               self.l2g[ci]['nFreeDOF'],
                                                                                                               self.l2g[ci]['freeLocal'],
                                                                                                               self.l2g[ci]['freeGlobal'],
                                                                                                               self.l2g[ck]['nFreeDOF'],
                                                                                                               self.l2g[ck]['freeLocal'],
                                                                                                               self.l2g[ck]['freeGlobal'],
                                                                                                               self.numericalFlux.isDOFBoundary[ck],
                                                                                                               self.numericalFlux.boundaryAdjoint_sigma,
                                                                                                               self.ebqe[('v',ck)],
                                                                                                               self.ebqe['n'],
                                                                                                               self.numericalFlux.ebqe[('a',ci,ck)],
                                                                                                               self.ebqe[('grad(v)',ci)],
                                                                                                               self.ebqe[('dS_u',ck)],
                                                                                                               jacobian)
                    else:
                        if self.numericalFlux.hasInterior:
                            cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense(self.offset[ci],
                                                                                                                self.stride[ci],
                                                                                                                self.offset[ck],
                                                                                                                self.stride[ck],
                                                                                                                self.nFreeVDOF_global,
                                                                                                                self.mesh.interiorElementBoundariesArray,
                                                                                                                self.mesh.elementBoundaryElementsArray,
                                                                                                                self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                                self.l2g[ci]['nFreeDOF'],
                                                                                                                self.l2g[ci]['freeLocal'],
                                                                                                                self.l2g[ci]['freeGlobal'],
                                                                                                                self.l2g[ck]['nFreeDOF'],
                                                                                                                self.l2g[ck]['freeLocal'],
                                                                                                                self.l2g[ck]['freeGlobal'],
                                                                                                                self.numericalFlux.boundaryAdjoint_sigma,
                                                                                                                self.ebq[('v',ck)],
                                                                                                                self.ebq['n'],
                                                                                                                self.ebq[('a',ci,ck)],
                                                                                                                self.ebq[('grad(v)',ci)],
                                                                                                                self.ebq[('dS_u',ck)],
                                                                                                                jacobian)
                        if not self.numericalFlux.includeBoundaryAdjointInteriorOnly: #added to only eval interior tjp
                            cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense(self.offset[ci],
                                                                                                            self.stride[ci],
                                                                                                            self.offset[ck],
                                                                                                            self.stride[ck],
                                                                                                            self.nFreeVDOF_global,
                                                                                                            self.mesh.exteriorElementBoundariesArray,
                                                                                                            self.mesh.elementBoundaryElementsArray,
                                                                                                            self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                            self.l2g[ci]['nFreeDOF'],
                                                                                                            self.l2g[ci]['freeLocal'],
                                                                                                            self.l2g[ci]['freeGlobal'],
                                                                                                            self.l2g[ck]['nFreeDOF'],
                                                                                                            self.l2g[ck]['freeLocal'],
                                                                                                            self.l2g[ck]['freeGlobal'],
                                                                                                            self.numericalFlux.isDOFBoundary[ck],
                                                                                                            self.numericalFlux.boundaryAdjoint_sigma,
                                                                                                            self.ebqe[('v',ck)],
                                                                                                            self.ebqe['n'],
                                                                                                            self.numericalFlux.ebqe[('a',ci,ck)],
                                                                                                            self.ebqe[('grad(v)',ci)],
                                                                                                            self.ebqe[('dS_u',ck)],
                                                                                                            jacobian)
#                     for ebNI,ebN in enumerate(self.mesh.interiorElementBoundariesArray):
#                         left_eN_global = self.mesh.elementBoundaryElementsArray[ebN,0]
#                         right_eN_global = self.mesh.elementBoundaryElementsArray[ebN,1]
#                         left_ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
#                         right_ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
#                         for ii in range(self.l2g[ci]['nFreeDOF'][left_eN_global]):
#                             left_i = self.l2g[ci]['freeLocal'][left_eN_global,ii]
#                             left_I = self.offset[ci] + self.stride[ci]*self.l2g[ci]['freeGlobal'][left_eN_global,ii]
#                             right_i = self.l2g[ci]['freeLocal'][right_eN_global,ii]
#                             right_I = self.offset[ci] + self.stride[ci]*self.l2g[ci]['freeGlobal'][right_eN_global,ii]
#                             for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                 for jj in range(self.l2g[ci]['nFreeDOF'][right_eN_global]):
#                                     left_j = self.l2g[ci]['freeLocal'][left_eN_global,jj]
#                                     left_J = self.offset[ci] + self.stride[ci]*self.l2g[ci]['freeGlobal'][left_eN_global,jj]
#                                     right_j = self.l2g[ci]['freeLocal'][right_eN_global,jj]
#                                     right_J = self.offset[ci] + self.stride[ci]*self.l2g[ci]['freeGlobal'][right_eN_global,jj]
#                                     for II in range(self.nSpace_global):
#                                         jacobian[left_J,left_I] -= self.numericalFlux.boundaryAdjoint_sigma*0.5*(self.ebq[('v',ci)][left_eN_global,left_ebN_element,k,left_j]
#                                                                     *self.ebq['n'][left_eN_global,left_ebN_element,k,II]
#                                                                     *self.ebq[('a',ci,ci)][left_eN_global,left_ebN_element,k,II*self.nSpace_global+II]
#                                                                     *self.ebq[('grad(v)',ci)][left_eN_global,left_ebN_element,k,left_i,II]
#                                                                     *self.ebq[('dS_u',ci)][left_eN_global,left_ebN_element,k])
#                                         jacobian[right_J,left_I] += self.numericalFlux.boundaryAdjoint_sigma*0.5*(self.ebq[('v',ci)][right_eN_global,right_ebN_element,k,right_j]
#                                                                      *self.ebq['n'][left_eN_global,left_ebN_element,k,II]
#                                                                      *self.ebq[('a',ci,ci)][left_eN_global,left_ebN_element,k,II*self.nSpace_global+II]
#                                                                      *self.ebq[('grad(v)',ci)][left_eN_global,left_ebN_element,k,left_i,II]
#                                                                      *self.ebq[('dS_u',ci)][left_eN_global,left_ebN_element,k])
#                                         jacobian[left_J,right_I] -= self.numericalFlux.boundaryAdjoint_sigma*0.5*(self.ebq[('v',ci)][left_eN_global,left_ebN_element,k,left_j]
#                                                                      *self.ebq['n'][left_eN_global,left_ebN_element,k,II]
#                                                                      *self.ebq[('a',ci,ci)][right_eN_global,right_ebN_element,k,II*self.nSpace_global+II]
#                                                                      *self.ebq[('grad(v)',ci)][right_eN_global,right_ebN_element,k,right_i,II]
#                                                                      *self.ebq[('dS_u',ci)][right_eN_global,right_ebN_element,k])
#                                         jacobian[right_J,right_I] += self.numericalFlux.boundaryAdjoint_sigma*0.5*(self.ebq[('v',ci)][right_eN_global,right_ebN_element,k,right_j]
#                                                                       *self.ebq['n'][left_eN_global,left_ebN_element,k,II]
#                                                                       *self.ebq[('a',ci,ci)][right_eN_global,right_ebN_element,k,II*self.nSpace_global+II]
#                                                                       *self.ebq[('grad(v)',ci)][right_eN_global,right_ebN_element,k,right_i,II]
#                                                                       *self.ebq[('dS_u',ci)][right_eN_global,right_ebN_element,k])
#                     for ebNE,ebN in enumerate(self.mesh.exteriorElementBoundariesArray):
#                         eN_global = self.mesh.elementBoundaryElementsArray[ebN,0]
#                         ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
#                         for ii in range(self.l2g[ci]['nFreeDOF'][eN_global]):
#                             i = self.l2g[ci]['freeLocal'][eN_global,ii]
#                             I = self.offset[ci] + self.stride[ci]*self.l2g[ci]['freeGlobal'][eN_global,ii]
#                             for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                 for jj in range(self.l2g[ci]['nFreeDOF'][eN_global]):
#                                     j = self.l2g[ci]['freeLocal'][eN_global,jj]
#                                     J = self.offset[ci] + self.stride[ci]*self.l2g[ci]['freeGlobal'][eN_global,jj]
#                                     for II in range(self.nSpace_global):
#                                         if self.numericalFlux.isDOFBoundary[ci][ebNE,k]:
#                                             jacobian[J,I] -= self.numericalFlux.boundaryAdjoint_sigma*(self.ebqe[('v',ci)][ebNE,k,j]
#                                                                     *self.ebqe['n'][ebNE,k,II]
#                                                                     *self.ebqe[('a',ci,ci)][ebNE,k,II*self.nSpace_global+II]
#                                                                     *self.ebqe[('grad(v)',ci)][ebNE,k,i,II]
#                                                                     *self.ebqe[('dS_u',ci)][ebNE,k])
        return jacobian
    def getJacobian_CSR(self,jacobian):
        """
        Add in the element jacobians to the global jacobian
        """
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)
        for ci in range(self.nc):
            for cj in self.coefficients.stencil[ci]:
                #
                #element contributions
                #
                cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[ci]['nFreeDOF'],
                                                                          self.l2g[ci]['freeLocal'],
                                                                          self.l2g[cj]['nFreeDOF'],
                                                                          self.l2g[cj]['freeLocal'],
                                                                          self.csrRowIndeces[(ci,cj)],
                                                                          self.csrColumnOffsets[(ci,cj)],
                                                                          self.elementJacobian[ci][cj],
                                                                          jacobian)
                if self.numericalFlux is not None and self.numericalFlux.mixedDiffusion[ci] == True:
                    cfemIntegrals.updateGlobalJacobianFromElementJacobian_eb_CSR(self.mesh.elementNeighborsArray,
                                                                                 self.l2g[ci]['nFreeDOF'],
                                                                                 self.l2g[ci]['freeLocal'],
                                                                                 self.l2g[cj]['nFreeDOF'],
                                                                                 self.l2g[cj]['freeLocal'],
                                                                                 self.csrRowIndeces[(ci,cj)],
                                                                                 self.csrColumnOffsets_eNebN[(ci,cj)],
                                                                                 self.elementJacobian_eb[ci][cj],
                                                                                 jacobian)
                #
                #element boundary contributions
                #
                if self.numericalFlux is not None and not isinstance(self.numericalFlux, NumericalFlux.DoNothing):
                    if self.numericalFlux.hasInterior:
                        cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_CSR(self.mesh.interiorElementBoundariesArray,
                                                                                                      self.mesh.elementBoundaryElementsArray,
                                                                                                      self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                      self.l2g[ci]['nFreeDOF'],
                                                                                                      self.l2g[ci]['freeLocal'],
                                                                                                      self.l2g[cj]['nFreeDOF'],
                                                                                                      self.l2g[cj]['freeLocal'],
                                                                                                      self.csrRowIndeces[(ci,cj)],
                                                                                                      self.csrColumnOffsets_eb[(ci,cj)],
                                                                                                      self.fluxJacobian[ci][cj],
                                                                                                      self.ebq[('w*dS_f',ci)],
                                                                                                      jacobian)
                        if self.numericalFlux.mixedDiffusion[ci] == True:
                            cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_CSR(self.mesh.elementNeighborsArray,
                                                                                                             self.mesh.interiorElementBoundariesArray,
                                                                                                             self.mesh.elementBoundaryElementsArray,
                                                                                                             self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                             self.l2g[ci]['nFreeDOF'],
                                                                                                             self.l2g[ci]['freeLocal'],
                                                                                                             self.l2g[cj]['nFreeDOF'],
                                                                                                             self.l2g[cj]['freeLocal'],
                                                                                                             self.csrRowIndeces[(ci,cj)],
                                                                                                             self.csrColumnOffsets_eb_eNebN[(ci,cj)],
                                                                                                             self.fluxJacobian_eb[ci][cj],
                                                                                                             self.ebq[('w*dS_f',ci)],
                                                                                                             jacobian)
                        if self.numericalFlux.HamiltonJacobiNumericalFlux[ci] == True:
                            cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_CSR(self.mesh.interiorElementBoundariesArray,
                                                                                                                 self.mesh.elementBoundaryElementsArray,
                                                                                                                 self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                                 self.l2g[ci]['nFreeDOF'],
                                                                                                                 self.l2g[ci]['freeLocal'],
                                                                                                                 self.l2g[cj]['nFreeDOF'],
                                                                                                                 self.l2g[cj]['freeLocal'],
                                                                                                                 self.csrRowIndeces[(ci,cj)],
                                                                                                                 self.csrColumnOffsets_eb[(ci,cj)],
                                                                                                                 self.fluxJacobian_hj[ci][cj],
                                                                                                                 self.ebq[('w*dS_f',ci)],
                                                                                                                 jacobian)
                    cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_CSR(self.mesh.exteriorElementBoundariesArray,
                                                                                                  self.mesh.elementBoundaryElementsArray,
                                                                                                  self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                  self.l2g[ci]['nFreeDOF'],
                                                                                                  self.l2g[ci]['freeLocal'],
                                                                                                  self.l2g[cj]['nFreeDOF'],
                                                                                                  self.l2g[cj]['freeLocal'],
                                                                                                  self.csrRowIndeces[(ci,cj)],
                                                                                                  self.csrColumnOffsets_eb[(ci,cj)],
                                                                                                  self.fluxJacobian_exterior[ci][cj],
                                                                                                  self.ebqe[('w*dS_f',ci)],
                                                                                                  jacobian)
                    if self.numericalFlux.mixedDiffusion[ci] == True:
                        cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_CSR(self.mesh.elementNeighborsArray,
                                                                                                         self.mesh.exteriorElementBoundariesArray,
                                                                                                         self.mesh.elementBoundaryElementsArray,
                                                                                                         self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                         self.l2g[ci]['nFreeDOF'],
                                                                                                         self.l2g[ci]['freeLocal'],
                                                                                                         self.l2g[cj]['nFreeDOF'],
                                                                                                         self.l2g[cj]['freeLocal'],
                                                                                                         self.csrRowIndeces[(ci,cj)],
                                                                                                         self.csrColumnOffsets_eb_eNebN[(ci,cj)],
                                                                                                         self.fluxJacobian_eb[ci][cj],
                                                                                                         self.ebqe[('w*dS_f',ci)],
                                                                                                         jacobian)
                else:
                    #this will go away
                    if ((self.fluxBoundaryConditions[ci] == 'outFlow' or
                         self.fluxBoundaryConditions[ci] == 'mixedFlow')
                        and
                        self.timeIntegration.advectionIsImplicit[ci]):
                        if ('w*dS_f',ci) in self.ebqe:
                            #
                            cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_CSR(self.mesh.exteriorElementBoundariesArray,
                                                                                                          self.mesh.elementBoundaryElementsArray,
                                                                                                          self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                          self.l2g[ci]['nFreeDOF'],
                                                                                                          self.l2g[ci]['freeLocal'],
                                                                                                          self.l2g[cj]['nFreeDOF'],
                                                                                                          self.l2g[cj]['freeLocal'],
                                                                                                          self.csrRowIndeces[(ci,cj)],
                                                                                                          self.csrColumnOffsets_eb[(ci,cj)],
                                                                                                          self.fluxJacobian_exterior[ci][cj],
                                                                                                          self.ebqe[('w*dS_f',ci)],
                                                                                                          jacobian)
                            #mwf TODO can't get here with current logic?
                            if self.numericalFlux is not None and self.numericalFlux.mixedDiffusion[ci] == True:
                            #
                                cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_CSR(self.mesh.exteriorElementBoundariesArray,
                                                                                                                 self.mesh.elementBoundaryElementsArray,
                                                                                                                 self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                                 self.l2g[ci]['nFreeDOF'],
                                                                                                                 self.l2g[ci]['freeLocal'],
                                                                                                                 self.l2g[cj]['nFreeDOF'],
                                                                                                                 self.l2g[cj]['freeLocal'],
                                                                                                                 self.csrRowIndeces[(ci,cj)],
                                                                                                                 self.csrColumnOffsets_eb_eNebN[(ci,cj)],
                                                                                                                 self.fluxJacobian_eb[ci][cj],
                                                                                                                 self.ebqe[('w*dS_f',ci)],
                                                                                                                 jacobian)

        #
        #element boundary contributions from diffusion
        #
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck in list(ckDict.keys()):
                if self.numericalFlux.includeBoundaryAdjoint:
                    if self.sd:
                        if self.numericalFlux.hasInterior:
                            cfemIntegrals.updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_CSR_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                                                 self.offset[ci],
                                                                                                                 self.stride[ci],
                                                                                                                 self.offset[ck],
                                                                                                                 self.stride[ck],
                                                                                                                 self.nFreeVDOF_global,
                                                                                                                 self.mesh.interiorElementBoundariesArray,
                                                                                                                 self.mesh.elementBoundaryElementsArray,
                                                                                                                 self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                                 self.l2g[ci]['nFreeDOF'],
                                                                                                                 self.l2g[ci]['freeLocal'],
                                                                                                                 self.l2g[ci]['freeGlobal'],
                                                                                                                 self.l2g[ck]['nFreeDOF'],
                                                                                                                 self.l2g[ck]['freeLocal'],
                                                                                                                 self.l2g[ck]['freeGlobal'],
                                                                                                                 self.csrRowIndeces[(ci,ck)],
                                                                                                                 self.csrColumnOffsets_eb[(ci,ck)],
                                                                                                                 self.numericalFlux.boundaryAdjoint_sigma,
                                                                                                                 self.ebq[('v',ck)],
                                                                                                                 self.ebq['n'],
                                                                                                                 self.ebq[('a',ci,ck)],
                                                                                                                 self.ebq[('grad(v)',ci)],
                                                                                                                 self.ebq[('dS_u',ck)],
                                                                                                                 jacobian)
                        if not self.numericalFlux.includeBoundaryAdjointInteriorOnly: #added to only eval interior tjp
                            cfemIntegrals.updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_CSR_sd(self.coefficients.sdInfo[(ci,ck)][0],
                                                                                                             self.coefficients.sdInfo[(ci,ck)][1],
                                                                                                             self.offset[ci],
                                                                                                             self.stride[ci],
                                                                                                             self.offset[ck],
                                                                                                             self.stride[ck],
                                                                                                             self.nFreeVDOF_global,
                                                                                                             self.mesh.exteriorElementBoundariesArray,
                                                                                                             self.mesh.elementBoundaryElementsArray,
                                                                                                             self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                             self.l2g[ci]['nFreeDOF'],
                                                                                                             self.l2g[ci]['freeLocal'],
                                                                                                             self.l2g[ci]['freeGlobal'],
                                                                                                             self.l2g[ck]['nFreeDOF'],
                                                                                                             self.l2g[ck]['freeLocal'],
                                                                                                             self.l2g[ck]['freeGlobal'],
                                                                                                             self.csrRowIndeces[(ci,ck)],
                                                                                                             self.csrColumnOffsets_eb[(ci,ck)],
                                                                                                             self.numericalFlux.isDOFBoundary[ck],
                                                                                                             self.numericalFlux.boundaryAdjoint_sigma,
                                                                                                             self.ebqe[('v',ck)],
                                                                                                             self.ebqe['n'],
                                                                                                             self.numericalFlux.ebqe[('a',ci,ck)],
                                                                                                             self.ebqe[('grad(v)',ci)],
                                                                                                             self.ebqe[('dS_u',ck)],
                                                                                                             jacobian)
                    else:
                        raise RuntimeError("boundary adjoint terms with CSR jacobian and dense diffusion tensor not implemented")
        return jacobian
    def calculateElementResidual(self):
        """Calculate all the element residuals"""
        for ci in range(self.nc):
            self.elementResidual[ci].fill(0.0)
        for ci  in list(self.coefficients.advection.keys()):
            cfemIntegrals.updateAdvection_weak(self.q[('f',ci)],
                                               self.q[('grad(w)*dV_f',ci)],
                                               self.elementResidual[ci])
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck in list(ckDict.keys()):
                if self.numericalFlux is None or self.numericalFlux.mixedDiffusion[ci] == False:
                    if self.sd:
                        cfemIntegrals.updateDiffusion_weak_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                              self.q[('a',ci,ck)],
                                                              self.q[('grad(phi)',ck)],
                                                              self.q[('grad(w)*dV_a',ck,ci)],
                                                              self.elementResidual[ci])
                    elif self.lowmem:
                        cfemIntegrals.updateDiffusion_weak_lowmem(self.q[('a',ci,ck)],
                                                                  self.q[('grad(phi)',ck)],
                                                                  self.q[('grad(w)*dV_a',ck,ci)],
                                                                  self.elementResidual[ci])
                    else:
                        cfemIntegrals.updateDiffusion_weak(self.q[('a',ci,ck)],
                                                           self.q[('grad(phi)Xgrad(w)*dV_a',ck,ci)],
                                                           self.elementResidual[ci])

                else:
                    if ('grad(w)*dV_f',ck) not in self.q:
                        self.q[('grad(w)*dV_f',ck)] = self.q[('grad(w)*dV_a',ci,ck)]
                    if 'rho_split' in dir(self.numericalFlux):
                        rho_split = self.numericalFlux.rho_split
                    else:
                        rho_split = 0
                    if self.sd:
                        self.q[('velocity',ck)].fill(0.0)
                        cfemIntegrals.updateDiffusion_MixedForm_weak_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                        self.numericalFlux.aTilde[(ci,ck)],
                                                                        self.numericalFlux.qV[ck],
                                                                        self.q[('grad(w)*dV_f',ck)],
                                                                        self.q[('velocity',ck)],#added for ldg coupling tjp
                                                                        self.elementResidual[ci],
                                                                        rho_split)
                    else:
                        cfemIntegrals.updateDiffusion_MixedForm_weak(self.numericalFlux.aTilde[(ci,ck)],
                                                                     self.numericalFlux.qV[ck],
                                                                     self.q[('grad(w)*dV_f',ck)],
                                                                     self.elementResidual[ci])
        for ci in list(self.coefficients.reaction.keys()):
            cfemIntegrals.updateReaction_weak(self.q[('r',ci)],
                                              self.q[('w*dV_r',ci)],
                                              self.elementResidual[ci])
        for ci in list(self.coefficients.hamiltonian.keys()):
            cfemIntegrals.updateHamiltonian_weak(self.q[('H',ci)],
                                                 self.q[('w*dV_H',ci)],
                                                 self.elementResidual[ci])
        if len(list(self.coefficients.stress.keys()))>0:
#             import copy
#             sc = copy.deepcopy(self.q['sigma'])
#             sc.flat[:]=0.0
#             for ci in range(3):
#                 for ck in range(3):
#                     for eN in range(self.q[('dsigma',ci,ck)].shape[0]):
#                         for i in range(self.elementResidual[ci].shape[1]):
#                             for k in range(self.q[('dsigma',ci,ck)].shape[1]):
#                                 for I in range(3):
#                                     for J in range(3):
#                                         if i==0:
#                                             sc[eN,k,I,ci] += self.q[('dsigma',ci,ck)][eN,k,I,J]*self.q[('grad(u)',ck)][eN,k,J]
#                                         #self.elementResidual[ci][eN,i] += self.q[('dsigma',ci,ck)][eN,k,I,J]*self.q[('grad(u)',ck)][eN,k,J]*self.q[('grad(w)*dV_sigma',ci)][eN,k,i,I]
#             for ci in range(3):
#                 for eN in range(self.q[('dsigma',ci,ck)].shape[0]):
#                     for i in range(self.elementResidual[ci].shape[1]):
#                         for k in range(self.q[('dsigma',ci,ck)].shape[1]):
#                             for I in range(3):
#                                 self.elementResidual[ci][eN,i] += sc[eN,k,I,ci]*self.q[('grad(w)*dV_sigma',ci)][eN,k,i,I]
#                     cfemIntegrals.updateDiffusion_weak_lowmem(self.q[('dsigma',ci,ck)],
#                                                               self.q[('grad(u)',ck)],
#                                                               self.q[('grad(w)*dV_sigma',0)],
#                                                               self.elementResidual[ci])
#             #print "copy",sc
#             #print "diff",
#             print "diff",self.q['sigma'] - sc
            cfemIntegrals.updateStress_weak(self.q['sigma'],
                                            self.q[('grad(w)*dV_sigma',0)],
                                            self.elementResidual[0],
                                            self.elementResidual[1],
                                            self.elementResidual[2])
        if  self.stabilization is not None:
            for ci in range(self.nc):
                for cj in self.coefficients.stencil[ci]:
                    cfemIntegrals.updateSubgridError(self.q[('subgridError',cj)],
                                                     self.q[('Lstar*w*dV',cj,ci)],
                                                     self.elementResidual[ci])
                    #now incorporate gradient stabilization if exists
                    if self.stabilization.usesGradientStabilization == True:
                        #mwf hack for now assume only term surviving gradient of
                        #L*w_h are mass and reaction terms
                        if self.lowmem:
                            cfemIntegrals.updateNumericalDiffusion_lowmem(self.q[('dmt_sge',ci,cj)],#check this
                                                                          self.q[('grad(subgridError)',ci)],
                                                                          self.q[('grad(w)*dV_f',ci)],
                                                                          self.elementResidual[ci])
                            #now try to manually insert to double check
                            #mwf also need dr_sge ...
                            cfemIntegrals.updateNumericalDiffusion_lowmem(self.q[('dr',ci,cj)],#check this
                                                                          self.q[('grad(subgridError)',ci)],
                                                                          self.q[('grad(w)*dV_f',ci)],
                                                                          self.elementResidual[ci])
                        else:
                            assert False, "need self.q[('grad(subgridError)Xgrad(w)*dV_numDiff',ci,ci)] if not self.lowmem "

        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                if self.lowmem:
                    cfemIntegrals.updateNumericalDiffusion_lowmem(self.q[('numDiff',ci,ci)],
                                                                  self.q[('grad(u)',ci)],
                                                                  self.q[('grad(w)*dV_numDiff',ci,ci)],
                                                                  self.elementResidual[ci])
                else:
                    cfemIntegrals.updateNumericalDiffusion(self.q[('numDiff',ci,ci)],
                                                           self.q[('grad(u)Xgrad(w)*dV_numDiff',ci,ci)],
                                                           self.elementResidual[ci])
        if self.numericalFlux is not None and not isinstance(self.numericalFlux, NumericalFlux.DoNothing):
            for ci in range(self.nc):
                self.ebq_global[('totalFlux',ci)].fill(0.0)
                self.ebqe[('totalFlux',ci)].fill(0.0)
            for ci in list(self.coefficients.advection.keys()):
                if (ci in self.numericalFlux.advectiveNumericalFlux and
                    self.numericalFlux.advectiveNumericalFlux[ci]) == True:
                    #cek do we let the numerical flux decide if interior is updated? yes we do. no we don't
                    if self.numericalFlux.hasInterior:
                        cfemIntegrals.updateInteriorElementBoundaryFlux(self.mesh.interiorElementBoundariesArray,
                                                                        self.mesh.elementBoundaryElementsArray,
                                                                        self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                        self.ebq_global[('advectiveFlux',ci)],
                                                                        self.ebq[('w*dS_f',ci)],
                                                                        self.elementResidual[ci])
                    cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                    self.mesh.elementBoundaryElementsArray,
                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                    self.ebqe[('advectiveFlux',ci)],
                                                                    self.ebqe[('w*dS_f',ci)],
                                                                    self.elementResidual[ci])
                    if self.numericalFlux.hasInterior:
                        self.ebq_global[('totalFlux',ci)] += self.ebq_global[('advectiveFlux',ci)]
                    self.ebqe[('totalFlux',ci)]       += self.ebqe[('advectiveFlux',ci)]
            for ci in list(self.coefficients.diffusion.keys()):
                if (ci in self.numericalFlux.diffusiveNumericalFlux and
                    self.numericalFlux.diffusiveNumericalFlux[ci]) == True:
                    for ck in self.coefficients.diffusion[ci]:
                        if self.numericalFlux.hasInterior:
                            cfemIntegrals.updateInteriorElementBoundaryFlux(self.mesh.interiorElementBoundariesArray,
                                                                            self.mesh.elementBoundaryElementsArray,
                                                                            self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                            self.ebq_global[('diffusiveFlux',ck,ci)],
                                                                            self.ebq[('w*dS_a',ck,ci)],
                                                                        self.elementResidual[ci])
                        cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                        self.mesh.elementBoundaryElementsArray,
                                                                        self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                        self.ebqe[('diffusiveFlux',ck,ci)],
                                                                        self.ebqe[('w*dS_a',ck,ci)],
                                                                        self.elementResidual[ci])
                        if self.numericalFlux.hasInterior:
                            self.ebq_global[('totalFlux',ci)] += self.ebq_global[('diffusiveFlux',ck,ci)]
                        self.ebqe[('totalFlux',ci)]       += self.ebqe[('diffusiveFlux',ck,ci)]
                        if self.numericalFlux.includeBoundaryAdjoint:
                            if self.sd:
                                if self.numericalFlux.hasInterior:
                                    cfemIntegrals.updateInteriorElementBoundaryDiffusionAdjoint_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                                   self.mesh.interiorElementBoundariesArray,
                                                                                                   self.mesh.elementBoundaryElementsArray,
                                                                                                   self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                   self.numericalFlux.boundaryAdjoint_sigma,
                                                                                                   self.ebq[('u',ck)],
                                                                                                   self.ebq['n'],
                                                                                                   self.ebq[('a',ci,ck)],
                                                                                                   self.ebq[('grad(v)',ci)],#cek should be grad(w)
                                                                                                   self.ebq[('dS_u',ci)],
                                                                                                   self.elementResidual[ci])
                                if not self.numericalFlux.includeBoundaryAdjointInteriorOnly: #added to only eval interior tjp
                                    cfemIntegrals.updateExteriorElementBoundaryDiffusionAdjoint_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                               self.numericalFlux.isDOFBoundary[ck],
                                                                                               self.mesh.exteriorElementBoundariesArray,
                                                                                               self.mesh.elementBoundaryElementsArray,
                                                                                               self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                               self.numericalFlux.boundaryAdjoint_sigma,
                                                                                               self.ebqe[('u',ck)],
                                                                                               self.numericalFlux.ebqe[('u',ck)],
                                                                                               self.ebqe['n'],
                                                                                               self.numericalFlux.ebqe[('a',ci,ck)],
                                                                                               self.ebqe[('grad(v)',ci)],#cek grad w
                                                                                               self.ebqe[('dS_u',ci)],
                                                                                               self.elementResidual[ci])
                            else:
                                if self.numericalFlux.hasInterior:
                                    cfemIntegrals.updateInteriorElementBoundaryDiffusionAdjoint(self.mesh.interiorElementBoundariesArray,
                                                                                                self.mesh.elementBoundaryElementsArray,
                                                                                                self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                self.numericalFlux.boundaryAdjoint_sigma,
                                                                                                self.ebq[('u',ck)],
                                                                                                self.ebq['n'],
                                                                                                self.ebq[('a',ci,ck)],
                                                                                                self.ebq[('grad(v)',ci)],#cek grad w
                                                                                                self.ebq[('dS_u',ci)],
                                                                                                self.elementResidual[ci])
                                if not self.numericalFlux.includeBoundaryAdjointInteriorOnly: #added to only eval interior tjp
                                    cfemIntegrals.updateExteriorElementBoundaryDiffusionAdjoint(self.numericalFlux.isDOFBoundary[ck],
                                                                                            self.mesh.exteriorElementBoundariesArray,
                                                                                            self.mesh.elementBoundaryElementsArray,
                                                                                            self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                            self.numericalFlux.boundaryAdjoint_sigma,
                                                                                            self.ebqe[('u',ck)],
                                                                                            self.numericalFlux.ebqe[('u',ck)],
                                                                                            self.ebqe['n'],
                                                                                            self.numericalFlux.ebqe[('a',ci,ck)],
                                                                                            self.ebqe[('grad(v)',ci)],#cek grad w
                                                                                            self.ebqe[('dS_u',ci)],
                                                                                            self.elementResidual[ci])
#                             for ebNI,ebN in enumerate(self.mesh.interiorElementBoundariesArray):
#                                 left_eN_global = self.mesh.elementBoundaryElementsArray[ebN,0]
#                                 right_eN_global = self.mesh.elementBoundaryElementsArray[ebN,1]
#                                 left_ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
#                                 right_ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
#                                 for i in range(self.nDOF_test_element[ci]):
#                                     for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                         for I in range(self.nSpace_global):
#                                             self.elementResidual[ci][left_eN_global,i] -= self.numericalFlux.boundaryAdjoint_sigma*0.5*((self.ebq[('u',ci)][left_eN_global,left_ebN_element,k]
#                                                                                                                                          -self.ebq[('u',ci)][right_eN_global,right_ebN_element,k])
#                                                                                                                                         *self.ebq['n'][left_eN_global,left_ebN_element,k,I]
#                                                                                                                                         *self.ebq[('a',ci,ci)][left_eN_global,left_ebN_element,k,I*self.nSpace_global+I]
#                                                                                                                                         *self.ebq[('grad(v)',ci)][left_eN_global,left_ebN_element,k,i,I]
#                                                                                                                                         *self.ebq[('dS_u',ci)][left_eN_global,left_ebN_element,k])
#                                             self.elementResidual[ci][right_eN_global,i] -= self.numericalFlux.boundaryAdjoint_sigma*0.5*((self.ebq[('u',ci)][left_eN_global,left_ebN_element,k]
#                                                                                                                                           -self.ebq[('u',ci)][right_eN_global,right_ebN_element,k])
#                                                                                                                                          *self.ebq['n'][left_eN_global,left_ebN_element,k,I]
#                                                                                                                                          *self.ebq[('a',ci,ci)][right_eN_global,right_ebN_element,k,I*self.nSpace_global+I]
#                                                                                                                                          *self.ebq[('grad(v)',ci)][right_eN_global,right_ebN_element,k,i,I]
#                                                                                                                                          *self.ebq[('dS_u',ci)][right_eN_global,right_ebN_element,k])
#                             for ebNE,ebN in enumerate(self.mesh.exteriorElementBoundariesArray):
#                                 eN_global = self.mesh.elementBoundaryElementsArray[ebN,0]
#                                 ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
#                                 for i in range(self.nDOF_test_element[ci]):
#                                     for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                         for I in range(self.nSpace_global):
#                                             if self.numericalFlux.isDOFBoundary[ci][ebNE,k]:
#                                                 self.elementResidual[ci][eN_global,i] -= self.numericalFlux.boundaryAdjoint_sigma*((self.ebqe[('u',ci)][ebNE,k]
#                                                                                                                                     -self.numericalFlux.ebqe[('u',ci)][ebNE,k])
#                                                                                                                                    *self.ebqe['n'][ebNE,k,I]
#                                                                                                                                    *self.ebqe[('a',ci,ci)][ebNE,k,I*self.nSpace_global+I]
#                                                                                                                                    *self.ebqe[('grad(v)',ci)][ebNE,k,i,I]
#                                                                                                                                    *self.ebqe[('dS_u',ci)][ebNE,k])
            for ci in range(self.nc):
                if self.numericalFlux.hasInterior or self.conservativeFlux is not None:
                    cfemIntegrals.copyExteriorElementBoundaryValuesToGlobalElementBoundaryValues(self.mesh.exteriorElementBoundariesArray,
                                                                                                 self.mesh.elementBoundaryElementsArray,
                                                                                                 self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                 self.ebqe[('totalFlux',ci)],
                                                                                                 self.ebq_global[('totalFlux',ci)])
            for ci in list(self.coefficients.hamiltonian.keys()):
                if (ci in self.numericalFlux.HamiltonJacobiNumericalFlux and
                    self.numericalFlux.HamiltonJacobiNumericalFlux[ci] == True):
                    if self.numericalFlux.hasInterior:
                        cfemIntegrals.updateInteriorTwoSidedElementBoundaryFlux(self.mesh.interiorElementBoundariesArray,
                                                                                self.mesh.elementBoundaryElementsArray,
                                                                                self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                self.ebq[('HamiltonJacobiFlux',ci)],
                                                                                self.ebq[('w*dS_H',ci)],
                                                                                self.elementResidual[ci])
                    #1-sided on exterior boundary
                    cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                    self.mesh.elementBoundaryElementsArray,
                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                    self.ebqe[('HamiltonJacobiFlux',ci)],
                                                                    self.ebqe[('w*dS_H',ci)],
                                                                    self.elementResidual[ci])
            for ci in list(self.coefficients.stress.keys()):
                cfemIntegrals.updateExteriorElementBoundaryStressFlux(self.mesh.exteriorElementBoundariesArray,
                                                                      self.mesh.elementBoundaryElementsArray,
                                                                      self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                      self.ebqe[('stressFlux',ci)],
                                                                      self.ebqe[('w*dS_sigma',ci)],
                                                                      self.elementResidual[ci])
        else:
            #cek this will go away
            #cek need to clean up how BC's interact with numerical fluxes
            #outflow <=> zero diffusion, upwind advection
            #mixedflow <=> you set, otherwise calculated as free
            #setflow <=
            for ci,flag in self.fluxBoundaryConditions.items():
                #put in total flux here as well?
                self.ebqe[('totalFlux',ci)].fill(0.0)
                if self.conservativeFlux is not None and ci in self.conservativeFlux and self.conservativeFlux[ci] is not None and self.numericalFlux.hasInterior:
                    self.ebq_global[('totalFlux',ci)].fill(0.0)
                if (flag == 'outFlow' or
                    flag == 'mixedFlow' or
                    flag == 'setFlow'):
                    if ci in self.coefficients.advection:
                        cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                        self.mesh.elementBoundaryElementsArray,
                                                                        self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                        self.ebqe[('advectiveFlux',ci)],
                                                                        self.ebqe[('w*dS_f',ci)],
                                                                        self.elementResidual[ci])
                        self.ebqe[('totalFlux',ci)] += self.ebqe[('advectiveFlux',ci)]
                if (flag == 'mixedFlow' or
                    flag == 'setFlow'):
                    if  ci in self.coefficients.diffusion:
                        for ck in self.coefficients.diffusion[ci]:
                            #
                            cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                            self.mesh.elementBoundaryElementsArray,
                                                                            self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                            self.ebqe[('diffusiveFlux',ck,ci)],
                                                                            self.ebqe[('w*dS_a',ck,ci)],
                                                                            self.elementResidual[ci])
                            self.ebqe[('totalFlux',ci)]       += self.ebqe[('diffusiveFlux',ck,ci)]
                if self.conservativeFlux is not None and ci in self.conservativeFlux and self.conservativeFlux[ci] is not None:
                    cfemIntegrals.copyExteriorElementBoundaryValuesToGlobalElementBoundaryValues(self.mesh.exteriorElementBoundariesArray,
                                                                                                 self.mesh.elementBoundaryElementsArray,
                                                                                                 self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                 self.ebqe[('totalFlux',ci)],
                                                                                                 self.ebq_global[('totalFlux',ci)])
        self.elementSpatialResidual[ci][:]=self.elementResidual[ci]
        self.timeIntegration.calculateElementSpatialResidual(self.elementResidual)
        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        for ci in list(self.coefficients.mass.keys()):
            cfemIntegrals.updateMass_weak(self.q[('mt',ci)],
                                          self.q[('w*dV_m',ci)],
                                          self.elementResidual[ci])
        if self.dirichletNodeSetList is not None:
            for cj,nodeSetList in self.dirichletNodeSetList.items():
                for eN in range(self.mesh.nElements_global):
                    for j in nodeSetList[eN]:
                        J = self.u[cj].femSpace.dofMap.l2g[eN,j]
                        self.weakFactor=3.0
                        self.elementResidual[cj][eN,j] = (self.u[cj].dof[J]-self.dirichletValues[cj][(eN,j)])*self.weakFactor*self.mesh.elementDiametersArray[eN]
                        #self.u[cj].dof[J] = self.dirichletValues[cj][(eN,j)]#
                        #self.elementResidual[cj][eN,j] = self.u[cj].dof[J]-self.dirichletValues[cj][(eN,j)]
    def estimate_mt(self):
        for ci in list(self.coefficients.mass.keys()):
            cfemIntegrals.estimate_mt_lowmem(self.q[('w',ci)],
                                             self.q[('w*dV_m',ci)],
                                             self.elementSpatialResidual[ci],
                                             self.q[('mt',ci)])
    def calculateElementJacobian(self,skipMassTerms=False):
        for ci in range(self.nc):
            for cj in self.coefficients.stencil[ci]:
                self.elementJacobian[ci][cj].fill(0.0)
                self.elementJacobian_eb[ci][cj].fill(0.0)
        for ci,cjDict in self.coefficients.advection.items():
            for cj in cjDict:
                if self.timeIntegration.advectionIsImplicit[ci]:
                    if self.lowmem:
                        cfemIntegrals.updateAdvectionJacobian_weak_lowmem(self.q[('df',ci,cj)],
                                                                          self.q[('v',cj)],
                                                                          self.q[('grad(w)*dV_f',ci)],
                                                                          self.elementJacobian[ci][cj])
                    else:
                        cfemIntegrals.updateAdvectionJacobian_weak(self.q[('df',ci,cj)],
                                                                   self.q[('vXgrad(w)*dV_f',cj,ci)],
                                                                   self.elementJacobian[ci][cj])
        ##\todo optimize nonlinear diffusion Jacobian calculation for the  different combinations of nonlinear a and phi
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                for cj in set(list(cjDict.keys())+list(self.coefficients.potential[ck].keys())):
                    if self.timeIntegration.diffusionIsImplicit[ci]:
                        if self.numericalFlux is None or self.numericalFlux.mixedDiffusion[ci] == False:
                            if self.sd:
                                cfemIntegrals.updateDiffusionJacobian_weak_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                              self.phi[ck].femSpace.dofMap.l2g,
                                                                              self.q[('a',ci,ck)],
                                                                              self.q[('da',ci,ck,cj)],
                                                                              self.q[('grad(phi)',ck)],
                                                                              self.q[('grad(w)*dV_a',ck,ci)],
                                                                              self.dphi[(ck,cj)].dof,
                                                                              self.q[('v',cj)],
                                                                              self.q[('grad(v)',cj)],
                                                                              self.elementJacobian[ci][cj])
                            elif self.lowmem:
                                #mwf getting a problem right now when have multiple potentials since
                                #femIntegral routine uses nDOF_trial_element for accessing phi but this might not
                                #be the same when there are different spaces for different components
                                #e.g. when computing jacobian[0][1], 0 component has local dim 3, (trial space)
                                #component 1 has local dim 2 and potential, ck = 1 so phi has local dim 2
                                cfemIntegrals.updateDiffusionJacobian_weak_lowmem(self.phi[ck].femSpace.dofMap.l2g,
                                                                                  self.q[('a',ci,ck)],
                                                                                  self.q[('da',ci,ck,cj)],
                                                                                  self.q[('grad(phi)',ck)],
                                                                                  self.q[('grad(w)*dV_a',ck,ci)],
                                                                                  self.dphi[(ck,cj)].dof,
                                                                                  self.q[('v',cj)],
                                                                                  self.q[('grad(v)',cj)],
                                                                                  self.elementJacobian[ci][cj])
                            else:
                                cfemIntegrals.updateDiffusionJacobian_weak(self.phi[ck].femSpace.dofMap.l2g,
                                                                           self.q[('a',ci,ck)],
                                                                           self.q[('da',ci,ck,cj)],
                                                                           self.q[('grad(phi)Xgrad(w)*dV_a',ck,ci)],
                                                                           self.dphi[(ck,cj)].dof,
                                                                           self.q[('v',cj)],
                                                                           self.q[('grad(v)Xgrad(w)*dV_a',ck,cj,ci)],
                                                                           self.elementJacobian[ci][cj])
                        else:
                            if self.sd:
                                cfemIntegrals.updateDiffusionJacobian_MixedForm_weak_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                        self.numericalFlux.aTilde[(ci,ck)],
                                                                                        #self.q[('a',ci,ck)],
                                                                                        self.q[('da',ci,ck,cj)],
                                                                                        self.numericalFlux.qV[ck],
                                                                                        self.numericalFlux.qDV[ck],
                                                                                        self.numericalFlux.qDV_eb[ck],
                                                                                        self.q[('grad(w)*dV_f',ck)],
                                                                                        self.q[('v',cj)],
                                                                                        self.elementJacobian[ci][cj],
                                                                                        self.elementJacobian_eb[ci][cj])
                            else:
                                cfemIntegrals.updateDiffusionJacobian_MixedForm_weak(self.numericalFlux.aTilde[(ci,ck)],
                                                                                     #self.q[('a',ci,ck)],
                                                                                     self.q[('da',ci,ck,cj)],
                                                                                     self.numericalFlux.qV[ck],
                                                                                     self.numericalFlux.qDV[ck],
                                                                                     self.numericalFlux.qDV_eb[ck],
                                                                                     self.q[('grad(w)*dV_f',ck)],
                                                                                     self.q[('v',cj)],
                                                                                     self.elementJacobian[ci][cj],
                                                                                     self.elementJacobian_eb[ci][cj])
        for ci,cjDict in self.coefficients.reaction.items():
            for cj in cjDict:
                if self.timeIntegration.reactionIsImplicit[ci]:
                    if self.lowmem:
                        cfemIntegrals.updateReactionJacobian_weak_lowmem(self.q[('dr',ci,cj)],
                                                                         self.q[('v',cj)],
                                                                         self.q[('w*dV_r',ci)],
                                                                         self.elementJacobian[ci][cj])
                    else:
                        cfemIntegrals.updateReactionJacobian_weak(self.q[('dr',ci,cj)],
                                                                  self.q[('vXw*dV_r',cj,ci)],
                                                                  self.elementJacobian[ci][cj])
        for ci,cjDict in self.coefficients.hamiltonian.items():
            for cj in cjDict:
                if self.timeIntegration.hamiltonianIsImplicit[ci]:
                    if self.lowmem:
                        cfemIntegrals.updateHamiltonianJacobian_weak_lowmem(self.q[('dH',ci,cj)],
                                                                            self.q[('grad(v)',cj)],
                                                                            self.q[('w*dV_H',ci)],
                                                                            self.elementJacobian[ci][cj])
                    else:
                        cfemIntegrals.updateHamiltonianJacobian_weak(self.q[('dH',ci,cj)],
                                                                     self.q[('grad(v)Xw*dV_H',cj,ci)],
                                                                     self.elementJacobian[ci][cj])
        if len(self.coefficients.stress) > 0:
            cfemIntegrals.updateStressJacobian_weak(self.q[('dsigma',0,0)],
                                                    self.q[('dsigma',0,1)],
                                                    self.q[('dsigma',0,2)],
                                                    self.q[('dsigma',1,0)],
                                                    self.q[('dsigma',1,1)],
                                                    self.q[('dsigma',1,2)],
                                                    self.q[('dsigma',2,0)],
                                                    self.q[('dsigma',2,1)],
                                                    self.q[('dsigma',2,2)],
                                                    self.q[('grad(v)',0)],
                                                    self.q[('grad(w)*dV_sigma',0)],
                                                    self.elementJacobian[0][0],
                                                    self.elementJacobian[0][1],
                                                    self.elementJacobian[0][2],
                                                    self.elementJacobian[1][0],
                                                    self.elementJacobian[1][1],
                                                    self.elementJacobian[1][2],
                                                    self.elementJacobian[2][0],
                                                    self.elementJacobian[2][1],
                                                    self.elementJacobian[2][2])
        if self.stabilization is not None:
            for ci in range(self.coefficients.nc):
                if self.timeIntegration.stabilizationIsImplicit[ci]:
                    for cj in self.coefficients.stencil[ci]:
                        for cjj in self.coefficients.stencil[ci]:
                            cfemIntegrals.updateSubgridErrorJacobian(self.q[('dsubgridError',cj,cjj)],
                                                                     self.q[('Lstar*w*dV',cj,ci)],
                                                                     self.elementJacobian[ci][cjj])
                    if self.stabilization.usesGradientStabilization == True:
                        for cj in self.coefficients.stencil[ci]:
                            if self.lowmem:
                                #mwf hack to try to get something running for linear problems
                                cfemIntegrals.updateNumericalDiffusionJacobian_lowmem(self.q[('dgrad(subgridError)',ci,cj)],
                                                                                      self.q[('grad(v)',ci)],
                                                                                      self.q[('grad(w)*dV_f',ci)],
                                                                                      self.elementJacobian[ci][ci])
                            else:
                                assert False
        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                if self.timeIntegration.shockCapturingIsImplicit[ci]:
                    if self.lowmem:
                        cfemIntegrals.updateNumericalDiffusionJacobian_lowmem(self.q[('numDiff',ci,ci)],
                                                                              self.q[('grad(v)',ci)],
                                                                              self.q[('grad(w)*dV_numDiff',ci,ci)],
                                                                              self.elementJacobian[ci][ci])
                    else:
                        cfemIntegrals.updateNumericalDiffusionJacobian(self.q[('numDiff',ci,ci)],
                                                                       self.q[('grad(v)Xgrad(w)*dV_numDiff',ci,ci,ci)],
                                                                       self.elementJacobian[ci][ci])
        self.timeIntegration.calculateElementSpatialJacobian(self.elementJacobian)
        if not skipMassTerms:
            for ci,cjDict in self.coefficients.mass.items():
                for cj in cjDict:
                    if self.timeIntegration.massIsImplicit[ci]:
                        if self.lowmem:
                            cfemIntegrals.updateMassJacobian_weak_lowmem(self.q[('dmt',ci,cj)],
                                                                         self.q[('v',cj)],
                                                                         self.q[('w*dV_m',ci)],
                                                                         self.elementJacobian[ci][cj])
                        else:
                            cfemIntegrals.updateMassJacobian_weak(self.q[('dmt',ci,cj)],
                                                                  self.q[('vXw*dV_m',cj,ci)],
                                                                  self.elementJacobian[ci][cj])
            for cj in range(self.nc):
                if self.timeIntegration.duStar_du[cj] is not None:
                    self.elementJacobian[ci][cj] *= self.timeIntegration.duStar_du[cj]
        if self.dirichletNodeSetList is not None:
            for cj,nodeSetList in self.dirichletNodeSetList.items():
                for eN in range(self.mesh.nElements_global):
                    for j in nodeSetList[eN]:
                        self.elementJacobian[cj][cj][eN,j,:]=0.0
                        self.elementJacobian[cj][cj][eN,j,j]=self.weakFactor*self.mesh.elementDiametersArray[eN]
    def calculateElementMassJacobian(self):
        """
        calculate just the mass matrix terms for element jacobian (i.e., those that multiply the accumulation term)
        does not include dt
        """
        for ci in range(self.nc):
            for cj in self.coefficients.stencil[ci]:
                self.elementJacobian[ci][cj].fill(0.0)
        for ci,cjDict in self.coefficients.mass.items():
            for cj in cjDict:
                if self.timeIntegration.massIsImplicit[ci]:
                    if self.lowmem:
                        cfemIntegrals.updateMassJacobian_weak_lowmem(self.q[('dm',ci,cj)],
                                                                     self.q[('v',cj)],
                                                                     self.q[('w*dV_m',ci)],
                                                                     self.elementJacobian[ci][cj])
                    else:
                        cfemIntegrals.updateMassJacobian_weak(self.q[('dm',ci,cj)],
                                                              self.q[('vXw*dV_m',cj,ci)],
                                                              self.elementJacobian[ci][cj])
        for cj in range(self.nc):
            if self.timeIntegration.duStar_du[cj] is not None:
                self.elementJacobian[ci][cj] *= self.timeIntegration.duStar_du[cj]
        if self.dirichletNodeSetList is not None:
            for cj,nodeSetList in self.dirichletNodeSetList.items():
                for eN in range(self.mesh.nElements_global):
                    for j in nodeSetList[eN]:
                        self.elementJacobian[cj][cj][eN,j,:]=0.0
                        self.elementJacobian[cj][cj][eN,j,j]=self.weakFactor*self.mesh.elementDiametersArray[eN]


    def calculateElementBoundaryJacobian(self):
        evalElementBoundaryJacobian = False; evalElementBoundaryJacobian_hj = False
        for jDict in list(self.fluxJacobian.values()):
            for j in list(jDict.values()):
                evalElementBoundaryJacobian = True
                j.fill(0.0)
        for jDict in list(self.fluxJacobian_eb.values()):
            for j in list(jDict.values()):
                j.fill(0.0)
        for jDict in list(self.fluxJacobian_hj.values()):
            for j in list(jDict.values()):
                evalElementBoundaryJacobian_hj = True
                j.fill(0.0)
        #cek clean up this logic using numerical flux
        if self.numericalFlux is not None and not isinstance(self.numericalFlux, NumericalFlux.DoNothing):
            #
            self.numericalFlux.updateInteriorNumericalFluxJacobian(self.l2g,self.q,self.ebq,self.ebq_global,self.dphi,
                                                                   self.fluxJacobian,self.fluxJacobian_eb,self.fluxJacobian_hj)
        #end else
        if evalElementBoundaryJacobian:
            self.timeIntegration.calculateElementSpatialBoundaryJacobian(self.fluxJacobian,self.fluxJacobian_eb,)
        #mwf TODO what about fluxJacobian_hj
        if evalElementBoundaryJacobian_hj:
            self.timeIntegration.calculateElementSpatialBoundaryJacobian(self.fluxJacobian_hj,self.fluxJacobian_eb)#cek hack added eb, need to provide hj function

    def calculateExteriorElementBoundaryJacobian(self):
        for jDict in list(self.fluxJacobian_exterior.values()):
            for j in list(jDict.values()):
                j.fill(0.0)
        if self.numericalFlux is not None and not isinstance(self.numericalFlux, NumericalFlux.DoNothing):
            self.numericalFlux.updateExteriorNumericalFluxJacobian(self.l2g,self.inflowFlag,self.q,self.ebqe,self.dphi,
                                                                   self.fluxJacobian_exterior,self.fluxJacobian_eb,self.fluxJacobian_hj)
        else:
            for ci,cjDict in self.coefficients.advection.items():
                if ((self.fluxBoundaryConditions[ci] == 'outFlow' or
                     self.fluxBoundaryConditions[ci] == 'mixedFlow') and
                    self.timeIntegration.advectionIsImplicit[ci]):
                    for cj in cjDict:
                        cnumericalFlux.updateExteriorNumericalAdvectiveFluxJacobian(self.mesh.exteriorElementBoundariesArray,
                                                                                    self.mesh.elementBoundaryElementsArray,
                                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                    self.inflowFlag[ci],
                                                                                    self.ebqe[('dadvectiveFlux_left',ci,cj)],
                                                                                    self.ebqe[('v',cj)],
                                                                                    self.fluxJacobian_exterior[ci][cj])
        #end else
        self.timeIntegration.calculateExteriorElementSpatialBoundaryJacobian(self.fluxJacobian_exterior)

    def calculateCoefficients(self):
        #cek could put logic her for element/internal eb/external eb
        self.calculateElementCoefficients()
        if self.needEBQ:
            self.calculateElementBoundaryCoefficients()
        self.calculateExteriorElementBoundaryCoefficients()
    def calculateSolutionAtQuadrature(self):
        for cj in range(self.nc):
            self.u[cj].getValues(self.q[('v',cj)],
                                 self.q[('u',cj)])
            if ('grad(u)',cj) in self.q:
                self.u[cj].getGradientValues(self.q[('grad(v)',cj)],
                                             self.q[('grad(u)',cj)])
            self.u[cj].getValuesGlobalExteriorTrace(self.ebqe[('v',cj)],
                                                    self.ebqe[('u',cj)])
            if ('grad(u)',cj) in self.ebqe:
                self.u[cj].getGradientValuesGlobalExteriorTrace(self.ebqe[('grad(v)',cj)],
                                                                self.ebqe[('grad(u)',cj)])
        if self.needEBQ:
            for cj in range(self.nc):
                self.u[cj].getValuesTrace(self.ebq[('v',cj)],
                                     self.ebq[('u',cj)])
                if ('grad(u)',cj) in self.ebq:
                    self.u[cj].getGradientValuesTrace(self.ebq[('grad(v)',cj)],
                                                 self.ebq[('grad(u)',cj)])

    def calculateStrongResidualAndAdjoint(self,cq):
        """
        break off computation of strong residual and adjoint so that we can do this at different quadrature locations?
        """
        if (self.stabilization or self.shockCapturing):
            for ci in range(self.nc):
                cq[('pdeResidual',ci)].fill(0.0)
                for cj in self.coefficients.stencil[ci]:
                    if self.stabilization:
                        cq[('Lstar*w*dV',cj,ci)].fill(0.0)
                    cq[('dpdeResidual',ci,cj)].fill(0.0)
            for ci,cjDict  in self.coefficients.advection.items():
                for cj in cjDict:
                    if ('df_sge',ci,cj) in cq:
                        cfemIntegrals.updateAdvection_strong(cq[('df_sge',ci,cj)],
                                                             cq[('grad(u)',cj)],
                                                             cq[('pdeResidual',ci)])
                        if self.stabilization:
                            cfemIntegrals.updateAdvection_adjoint(cq[('df_sge',ci,cj)],
                                                                  cq[('grad(w)*dV_stab',ci)],
                                                                  cq[('Lstar*w*dV',cj,ci)])
                        if self.timeIntegration.advectionIsImplicit[ci]:
                            cfemIntegrals.updateAdvectionJacobian_strong(cq[('df_sge',ci,cj)],
                                                                         cq[('grad(v)',cj)],
                                                                         cq[('dpdeResidual',ci,cj)])
            for ci,ckDict in self.coefficients.diffusion.items():
                for ck,cjDict in ckDict.items():
                    if self.Hess and self.stabilization is not None:
                        if self.sd:
                            cfemIntegrals.updateDiffusion2_strong_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                     cq[('a',ci,ck)],
                                                                     cq[('Hess(phi)',ck)],
                                                                     cq[('pdeResidual',ci)])
                        else:
                            cfemIntegrals.updateDiffusion2_strong(cq[('a',ci,ck)],
                                                                  cq[('Hess(phi)',ck)],
                                                                  cq[('pdeResidual',ci)])
                        if self.stabilization:
                            if self.sd:
                                cfemIntegrals.updateDiffusion2_adjoint_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                          cq[('a',ci,ck)],
                                                                          cq[('Hess(w)*dV_stab',ck,ci)],
                                                                          cq[('Lstar*w*dV',ck,ci)])
                            else:
                                cfemIntegrals.updateDiffusion2_adjoint(cq[('a',ci,ck)],
                                                                       cq[('Hess(w)*dV_stab',ck,ci)],
                                                                       cq[('Lstar*w*dV',ck,ci)])
                    for cj in list(cjDict.keys()):
                        if self.sd:
                            cfemIntegrals.updateDiffusion_strong_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                    cq[('da_sge',ci,ck,cj)],
                                                                    cq[('grad(phi)_sge',ck)],
                                                                    cq[('grad(u)',cj)],
                                                                    cq[('pdeResidual',ci)])
                        else:
                            cfemIntegrals.updateDiffusion_strong(cq[('da_sge',ci,ck,cj)],
                                                                 cq[('grad(phi)_sge',ck)],
                                                                 cq[('grad(u)',cj)],
                                                                 cq[('pdeResidual',ci)])
                        if self.stabilization:
                            if self.sd:
                                cfemIntegrals.updateDiffusion_adjoint_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                         cq[('da_sge',ci,ck,cj)],
                                                                         cq[('grad(phi)_sge',ck)],
                                                                         cq[('grad(w)*dV_stab',ck,ci)],
                                                                         cq[('Lstar*w*dV',cj,ci)])
                            else:
                                cfemIntegrals.updateDiffusion_adjoint(cq[('da_sge',ci,ck,cj)],
                                                                      cq[('grad(phi)_sge',ck)],
                                                                      cq[('grad(w)*dV_stab',ck,ci)],
                                                                      cq[('Lstar*w*dV',cj,ci)])
                        if self.timeIntegration.diffusionIsImplicit[ci]:
                            if self.sd:
                                cfemIntegrals.updateDiffusionJacobian_strong_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                self.phi[ck].femSpace.dofMap.l2g,
                                                                                cq[('da_sge',ci,ck,cj)],
                                                                                cq[('dphi_sge',ck,cj)],
                                                                                cq[('grad(phi)_sge',ck)],
                                                                                cq[('grad(u)',cj)],
                                                                                cq[('grad(v)',cj)],
                                                                                cq[('dpdeResidual',ci,cj)])
                            else:
                                cfemIntegrals.updateDiffusionJacobian_strong(self.phi[ck].femSpace.dofMap.l2g,
                                                                             cq[('da_sge',ci,ck,cj)],
                                                                             cq[('dphi_sge',ck,cj)],
                                                                             cq[('grad(phi)_sge',ck)],
                                                                             cq[('grad(u)',cj)],
                                                                             cq[('grad(v)',cj)],
                                                                             cq[('dpdeResidual',ci,cj)])
                    if self.Hess and self.stabilization is not None:
                        for cj in list(self.coefficients.potential[ck].keys()):
                            if self.sd:
                                cfemIntegrals.updateDiffusionJacobian2_strong_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                 self.phi[ck].femSpace.dofMap.l2g,
                                                                                 cq[('a',ci,ck)],
                                                                                 cq[('da',ci,ck,cj)],#cek hack
                                                                                 cq[('v',ck)],#cek hack, need to split the nonlinear case
                                                                                 cq[('Hess(phi)',ck)],
                                                                                 cq[('dphi',ck,cj)],
                                                                                 cq[('Hess(v)',cj)],
                                                                                 cq[('dpdeResidual',ci,cj)])
                            else:
                                cfemIntegrals.updateDiffusionJacobian2_strong(self.phi[ck].femSpace.dofMap.l2g,
                                                                              cq[('a',ci,ck)],
                                                                              cq[('da',ci,ck,cj)],#cek hack
                                                                              cq[('v',ck)],#cek hack, need to split the nonlinear case
                                                                              cq[('Hess(phi)',ck)],
                                                                              cq[('dphi',ck,cj)],
                                                                              cq[('Hess(v)',cj)],
                                                                              cq[('dpdeResidual',ci,cj)])
            for ci,cjDict in self.coefficients.reaction.items():
                cfemIntegrals.updateReaction_strong(cq[('r',ci)],
                                                    cq[('pdeResidual',ci)])
                for cj in cjDict:
                    if self.stabilization:
                        cfemIntegrals.updateReaction_adjoint(cq[('dr',ci,cj)],
                                                             cq[('w*dV_stab',ci)],
                                                             cq[('Lstar*w*dV',cj,ci)])
                    if self.timeIntegration.reactionIsImplicit[ci]:
                        cfemIntegrals.updateReactionJacobian_strong(cq[('dr',ci,cj)],
                                                                    cq[('v',cj)],
                                                                    cq[('dpdeResidual',ci,cj)])
            for ci,cjDict in self.coefficients.hamiltonian.items():
                for cj in cjDict:
                    #cek todo, need dH_sge here
                    cfemIntegrals.updateHamiltonian_strong(cq[('dH_sge',ci,cj)],
                                                           cq[('grad(u)',cj)],
                                                           cq[('pdeResidual',ci)])
                    if self.stabilization:
                        cfemIntegrals.updateHamiltonian_adjoint(cq[('dH_sge',ci,cj)],
                                                                cq[('grad(w)*dV_stab',cj)],
                                                                cq[('Lstar*w*dV',cj,ci)])
                    if True:#self.timeIntegration.hamiltonianIsImplicit[ci]:
                        cfemIntegrals.updateHamiltonianJacobian_strong(cq[('dH_sge',ci,cj)],
                                                                       cq[('grad(v)',cj)],
                                                                       cq[('dpdeResidual',ci,cj)])
            self.timeIntegration.calculateStrongElementSpatialResidual(cq)
            #cek do we need both these sets of loops
            for ci,cjDict in self.coefficients.mass.items():
                #mwf assume mt never lagged (what about setting to zero if lagged?)
                cfemIntegrals.updateMass_strong(cq[('mt',ci)],
                                                cq[('pdeResidual',ci)])
                for cj in cjDict:
                    #mwf May 1 2009 uncommented, then moved below
                    #if self.stabilization:
                    #    #mwf this needs to be dmt_sge, but needs to go after calculateSubgridError then
                    #    cfemIntegrals.updateMass_adjoint(cq[('dmt',ci,cj)],
                    #                                     cq[('w*dV_stab',ci)],
                    #                                     cq[('Lstar*w*dV',cj,ci)])
                    if self.timeIntegration.massIsImplicit[ci]:
                        cfemIntegrals.updateMassJacobian_strong(cq[('dmt',ci,cj)],
                                                                cq[('v',cj)],
                                                                cq[('dpdeResidual',ci,cj)])
            #now try to handle dmt term in adjoint
            for ci,cjDict in self.coefficients.mass.items():
                for cj in cjDict:
                    #mwf May 1 2009 uncommented, try to move after calculateSubgridError for stablization to track
                    #subscales
                    if self.stabilization and self.stabilization.trackSubScales:# or self.stabilization.usesGradientStabilization):
                        #mwf this needs to be dmt_sge, but needs to go after calculateSubgridError then
                        #mwf debug
                        logEvent("Transport adding adjoint mass term dmt_sge(%s,%s) max=%s min=%s " % (ci,cj,cq[('dmt_sge',ci,cj)].max(),cq[('dmt_sge',ci,cj)].min()))
                        cfemIntegrals.updateMass_adjoint(cq[('dmt_sge',ci,cj)],
                                                         cq[('w*dV_stab',ci)],
                                                         cq[('Lstar*w*dV',cj,ci)])


    def calculateElementCoefficients(self):
        """
        calculate the nonlinear coefficients at the quadrature points and nodes
        """
        #
        #get u,grad(u), and grad(u)Xgrad(w) at the quadrature points
        #
        for cj in range(self.nc):
            self.u[cj].getValues(self.q[('v',cj)],
                                 self.q[('u',cj)])
            if ('grad(u)',cj) in self.q:
                self.u[cj].getGradientValues(self.q[('grad(v)',cj)],
                                             self.q[('grad(u)',cj)])
            if self.stabilization is not None:
                if ('Hess(u)',cj) in self.q:
                    self.u[cj].getHessianValues(self.q[('Hess(v)',cj)],
                                                self.q[('Hess(u)',cj)])
            if not self.lowmem:
                for ci in range(self.nc):
                    for I in self.integralKeys:
                        if ('grad(u)Xgrad(w)*dV_'+I,cj,ci) in self.q:
                            self.u[cj].getGradientTensorValues(self.q[('grad(v)Xgrad(w)*dV_'+I,cj,cj,ci)],
                                                               self.q[('grad(u)Xgrad(w)*dV_'+I,cj,ci)])
        #
        #get functions of (t,x,u) at the quadrature points
        #
        self.coefficients.evaluate(self.timeIntegration.t,self.q)
        if self.movingDomain and self.coefficients.movingDomain:
            self.coefficients.updateToMovingDomain(self.timeIntegration.t,self.q)
#             for ci in range(self.nc):
#                 if self.q.has_key(('f',ci)):
#                     for eN in range(self.mesh.nElements_global):
#                         for k in range(self.nQuadraturePoints_element):
#                             for I in range(self.nSpace_global):
#                                 if self.q.has_key(('m',ci)):
#                                     self.q[('f',ci)][eN,k,I]-=self.q[('m',ci)][eN,k]*self.q['xt'][eN,k,I]
#                                 else:
#                                     self.q[('f',ci)][eN,k,I]-=self.q['xt'][eN,k,I]
#                     for cj in range(self.nc):
#                         if self.q.has_key(('dm',ci,cj)):
#                             for eN in range(self.mesh.nElements_global):
#                                 for k in range(self.nQuadraturePoints_element):
#                                     for I in range(self.nSpace_global):
#                                         self.q[('df',ci,cj)][eN,k,I]-=self.q[('dm',ci,cj)][eN,k]*self.q['xt'][eN,k,I]
#                     print "f",ci,self.q[('f',ci)]
        logEvent("Coefficients on element",level=10,data=self.q)
        #
        # let the time integrator calculate m_t and possibly other coefficients at the quadrature points
        #
        if self.timeTerm:
            self.timeIntegration.calculateElementCoefficients(self.q)
        #
        #cek and mwf need to go through this section to clean up, some of next two blocks could go to calcQuad
        #
        for cj in range(self.nc):
            if self.stabilization and self.stabilization.usesGradientStabilization:
                self.u[cj].femSpace.getBasisValues(self.phi[cj].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                   self.phi_ip[('v',cj)])
                self.u[cj].femSpace.elementMaps.getValues(self.phi[cj].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                          self.phi_ip[('x')])

                self.u[cj].getValues(self.phi_ip[('v',cj)],
                                     self.phi_ip[('u',cj)])
                self.u[cj].getGradientValues(self.phi_ip[('grad(v)',cj)],
                                             self.phi_ip[('grad(u)',cj)])
                self.coefficients.evaluate(self.timeIntegration.t,self.phi_ip)
        phiIsNonlinear=False
        for ck,cjDict in self.coefficients.potential.items():
            for cj,flag in cjDict.items():
                if flag == 'nonlinear':
                    phiIsNonlinear= True
                    for cj in range(self.coefficients.nc):
                        self.u[cj].femSpace.getBasisValues(self.phi[cj].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                           self.phi_ip[('v',cj)])
                        self.u[cj].femSpace.elementMaps.getValues(self.phi[cj].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                                  self.phi_ip[('x')])

                        self.u[cj].getValues(self.phi_ip[('v',cj)],
                                             self.phi_ip[('u',cj)])
                    #mwf check with Chris if we can pull this out of ck loop and call only if flag=='nonlinear' for some potential
                    self.coefficients.evaluate(self.timeIntegration.t,self.phi_ip)
                    break
        for ck,cjDict in self.coefficients.potential.items():
            for cj,flag in cjDict.items():
                if flag == 'nonlinear':
                    #mwf
                    self.phi[ck].projectFromInterpolationConditions(self.phi_ip[('phi',ck)])
                    self.dphi[(ck,cj)].projectFromInterpolationConditions(self.phi_ip[('dphi',ck,cj)])
                    #mwf need to communicate values across processors if have spatial heterogeneity and spatial dependence of
                    #potential
                    if self.phi[ck].par_dof is not None:
                        self.phi[ck].par_dof.scatter_forward_insert()
                    if self.dphi[(ck,cj)].par_dof is not None:
                        self.dphi[(ck,cj)].par_dof.scatter_forward_insert()

                    self.phi[ck].getValues(self.q[('v',ck)],
                                           self.q[('phi',ck)])
                    self.phi[ck].getGradientValues(self.q[('grad(v)',ck)],
                                                   self.q[('grad(phi)',ck)])
                    if self.stabilization is not None and self.Hess:
                        self.phi[ck].getHessianValues(self.q[('Hess(v)',ck)],
                                                      self.q[('Hess(phi)',ck)])
                    for ci in list(self.coefficients.diffusion.keys()):
                        if ck in self.coefficients.diffusion[ci]:
                            if not self.lowmem:
                                self.phi[ck].getGradientTensorValues(self.q[('grad(v)Xgrad(w)*dV_a',ck,cj,ci)],
                                                                     self.q[('grad(phi)Xgrad(w)*dV_a',ck,ci)])
                else:
                    self.phi[ck].dof[:]=self.u[ck].dof
                    self.dphi[(ck,cj)].dof.fill(1.0)
                    self.q[('phi',ck)][:]=self.q[('u',ck)]
                    self.q[('dphi',ck,cj)].fill(1.0)
                    self.q[('grad(phi)',ck)][:]=self.q[('grad(u)',ck)]
                    if self.stabilization is not None and self.Hess:
                        self.q[('Hess(phi)',ck)][:]=self.q[('Hess(u)',ck)]
                    for ci in list(self.coefficients.diffusion.keys()):
                        if ck in self.coefficients.diffusion[ci]:
                            if not self.lowmem:
                                self.q[('grad(phi)Xgrad(w)*dV_a',ck,ci)][:]=self.q[('grad(u)Xgrad(w)*dV_a',ck,ci)]

        if (self.stabilization or self.shockCapturing):
            if self.shockCapturing and not self.stabilization:
                self.calculateStrongResidualAndAdjoint(self.q)
            if self.stabilization:
                if self.stabilization.usesGradientStabilization:

                    #mwf hack pick up adjoint terms first
                    self.calculateStrongResidualAndAdjoint(self.q)
                    self.coefficients.evaluate(self.timeIntegration.t,self.phi_ip)
                    if self.timeTerm:
                        self.timeIntegration.calculateGeneralizedInterpolationCoefficients(self.phi_ip)
                    self.calculateStrongResidualAndAdjoint(self.phi_ip)
                else:
                    self.calculateStrongResidualAndAdjoint(self.q)

            if self.stabilization:
                self.stabilization.calculateSubgridError(self.q)

        else:
            #cek need to clean up calculation of dimless numbers, might as well do it all the time and pass to subgrid error
            #mwf figure out what to do with this
            #what happens if stabilization didn't compute cfl?
            for ci in range(self.nc):
                #cek try to skip this for stokes
                if (('df',ci,ci) in self.q and ('a',ci,ci) in self.q and
                    ('dr',ci,ci) in self.q and ('dmt',ci,ci) in self.q and
                    ('a',ci,ci) in self.q and ('dphi',ci,ci) in self.q):
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
                elif (('df',ci,ci) in self.q and ('dH',ci,ci) in self.q and
                      ('dm',ci,ci) in self.q):
                    #not likely?
                    cfemIntegrals.calculateCFLADR2speeds(self.elementEffectiveDiametersArray,
                                                         self.q[('dm',ci,ci)],
                                                         self.q[('df',ci,ci)],
                                                         self.q[('dH',ci,ci)],
                                                         self.q[('cfl',ci)])

                elif (('df',ci,ci) in self.q and ('dm',ci,ci) in self.q):
                    cfemIntegrals.calculateCFLADR(self.elementEffectiveDiametersArray,
                                                  self.q[('dm',ci,ci)],
                                                  self.q[('df',ci,ci)],
                                                  self.q[('cfl',ci)])
                elif (('dH',ci,ci) in self.q and ('dm',ci,ci) in self.q):
                    cfemIntegrals.calculateCFLADR(self.elementEffectiveDiametersArray,
                                                  self.q[('dm',ci,ci)],
                                                  self.q[('dH',ci,ci)],
                                                  self.q[('cfl',ci)])

        if self.shockCapturing is not None:
            self.shockCapturing.calculateNumericalDiffusion(self.q)
    def calculateElementBoundaryCoefficients(self):
        """
        Calculate the nonlinear coefficients at the element boundary quadrature points
        """
        #mwf add check for ebq
        #cek fix trickiness
        if 'x' not in list(self.ebq.keys()):
            return
        #
        #get u and grad(u) at the quadrature points
        #
        for ci in range(self.nc):
            self.u[ci].getValuesTrace(self.ebq[('v',ci)],self.ebq[('u',ci)])
            if ('grad(u)',ci) in self.ebq:
                self.u[ci].getGradientValuesTrace(self.ebq[('grad(v)',ci)],self.ebq[('grad(u)',ci)])
        #
        #get coefficients at the element boundary quadrature points
        #
        self.coefficients.evaluate(t = self.timeIntegration.t, c = self.ebq)
        if self.movingDomain and self.coefficients.movingDomain:
            self.coefficients.updateToMovingDomain(self.timeIntegration.t,self.ebq)
#             for ci in range(self.nc):
#                 if self.ebq.has_key(('f',ci)):
#                     for eN in range(self.mesh.nElements_global):
#                         for ebN in range(self.mesh.nElementBoundaries_element):
#                             for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                 for I in range(self.nSpace_global):
#                                     if self.ebq.has_key(('m',ci)):
#                                         self.ebq[('f',ci)][eN,ebN,k,I]-=self.ebq[('m',ci)][eN,ebN,k]*self.ebq['xt'][eN,ebN,k,I]
#                                     else:
#                                         self.ebq[('f',ci)][eN,ebN,k,I]-=self.ebq['xt'][eN,ebN,k,I]
#                     for cj in range(self.nc):
#                         if self.ebq.has_key(('dm',ci,cj)):
#                             for eN in range(self.mesh.nElements_global):
#                                 for ebN in range(self.mesh.nElementBoundaries_element):
#                                     for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                         for I in range(self.nSpace_global):
#                                             self.ebq[('df',ci,cj)][eN,ebN,k,I]-=self.ebq[('dm',ci,cj)][eN,ebN,k]*self.ebq['xt'][eN,ebN,k,I]
        #
        #time integration
        #
        self.timeIntegration.calculateElementBoundaryCoefficients(self.ebq)
        #
        #get phi at the quadrature points if necessary
        #
        for ck,cjDict in self.coefficients.potential.items():
            for cj,flag in cjDict.items():
                if flag == 'nonlinear':
                    self.phi[ck].getValuesTrace(self.ebq[('v',ck)],
                                                self.ebq[('phi',ck)])
                    self.phi[ck].getGradientValuesTrace(self.ebq[('grad(v)',ck)],
                                                        self.ebq[('grad(phi)',ck)])
                else:
                    self.ebq[('phi',ck)][:]=self.ebq[('u',ck)]
                    self.ebq[('dphi',ck,cj)].fill(1.0)
                    self.ebq[('grad(phi)',ck)][:]=self.ebq[('grad(u)',ck)]
        #
        # calculate the averages and jumps at element boundaries
        #
        if self.numericalFlux is not None and not isinstance(self.numericalFlux, NumericalFlux.DoNothing):
            self.numericalFlux.calculateInteriorNumericalFlux(self.q,self.ebq,self.ebq_global)
        if self.conservativeFlux is not None:
            for ci in list(self.conservativeFlux.keys()):
                #don't need for p1-nc?
                if self.conservativeFlux[ci] not in ['p1-nc','dg','dg-bdm','dg-point-eval'] and self.conservativeFlux[ci] is not None:
                    self.ebq[('velocity',ci)].fill(0.0)
                    #self.ebq_global[('velocity',ci)].fill(0.0)
                    self.ebq_global[('velocityAverage',ci)].fill(0.0)
                    if ci in self.coefficients.diffusion:
                        for ck in list(self.coefficients.diffusion[ci].keys()):
                            if self.sd:
                                cfemIntegrals.updateInteriorElementBoundaryDiffusiveVelocity_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                                self.mesh.interiorElementBoundariesArray,
                                                                                                self.mesh.elementBoundaryElementsArray,
                                                                                                self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                self.ebq[('a',ci,ck)],
                                                                                                self.ebq[('grad(phi)',ck)],
                                                                                                self.ebq[('velocity',ci)])
                                cfemIntegrals.updateExteriorElementBoundaryDiffusiveVelocity_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                                self.mesh.exteriorElementBoundariesArray,
                                                                                                self.mesh.elementBoundaryElementsArray,
                                                                                                self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                self.ebq[('a',ci,ck)],
                                                                                                self.ebq[('grad(phi)',ck)],
                                                                                                self.ebq[('velocity',ci)])
                            else:
                                cfemIntegrals.updateInteriorElementBoundaryDiffusiveVelocity(self.mesh.interiorElementBoundariesArray,
                                                                                             self.mesh.elementBoundaryElementsArray,
                                                                                             self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                             self.ebq[('a',ci,ck)],
                                                                                             self.ebq[('grad(phi)',ck)],
                                                                                             self.ebq[('velocity',ci)])
                                cfemIntegrals.updateExteriorElementBoundaryDiffusiveVelocity(self.mesh.exteriorElementBoundariesArray,
                                                                                             self.mesh.elementBoundaryElementsArray,
                                                                                             self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                             self.ebq[('a',ci,ck)],
                                                                                             self.ebq[('grad(phi)',ck)],
                                                                                             self.ebq[('velocity',ci)])
                    if ci in self.coefficients.advection:
                        cfemIntegrals.updateInteriorElementBoundaryAdvectiveVelocity(self.mesh.interiorElementBoundariesArray,
                                                                                     self.mesh.elementBoundaryElementsArray,
                                                                                     self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                     self.ebq[('f',ci)],
                                                                                     self.ebq[('velocity',ci)])
                        cfemIntegrals.updateExteriorElementBoundaryAdvectiveVelocity(self.mesh.exteriorElementBoundariesArray,
                                                                                     self.mesh.elementBoundaryElementsArray,
                                                                                     self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                     self.ebq[('f',ci)],
                                                                                     self.ebq[('velocity',ci)])
                    #
                    #cek can we get rid fo this flag if we're always doing it
                    includeShockCapturingVelocity = True
                    if self.shockCapturing is not None and includeShockCapturingVelocity:
                        fact=1.0
                        cfemIntegrals.updateInteriorElementBoundaryShockCapturingVelocity(self.mesh.interiorElementBoundariesArray,
                                                                                          self.mesh.elementBoundaryElementsArray,
                                                                                          self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                          fact*self.q[('numDiff',ci,ci)],
                                                                                          self.ebq[('grad(u)',ci)],
                                                                                          self.ebq[('velocity',ci)])
                        #
                        cfemIntegrals.updateExteriorElementBoundaryShockCapturingVelocity(self.mesh.exteriorElementBoundariesArray,
                                                                                          self.mesh.elementBoundaryElementsArray,
                                                                                          self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                          fact*self.q[('numDiff',ci,ci)],
                                                                                          self.ebq[('grad(u)',ci)],
                                                                                          self.ebq[('velocity',ci)])
                    cfemIntegrals.calculateInteriorElementBoundaryAverageVelocity(self.mesh.interiorElementBoundariesArray,
                                                                                  self.mesh.elementBoundaryElementsArray,
                                                                                  self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                  self.ebq[('velocity',ci)],
                                                                                  self.ebq_global[('velocityAverage',ci)])
                    #mwf can never get here! Need to fix
                    #if self.numericalFlux is None:
                    #cfemIntegrals.calculateExteriorElementBoundaryAverageVelocity(self.mesh.exteriorElementBoundariesArray,
                    #                                                              self.mesh.elementBoundaryElementsArray,
                    #                                                              self.mesh.elementBoundaryLocalElementBoundariesArray,
                    #                                                              self.ebq[('velocity',ci)],
                    #                                                              self.ebq_global[('velocityAverage',ci)])
                #end not p1-nc
            #end ci
        #end cons flux
    def calculateExteriorElementBoundaryCoefficients(self):
        """
        Calculate the nonlinear coefficients at global exterior element boundary quadrature points
        """
        #
        #get u and grad(u) at the quadrature points
        #
        for ci in range(self.nc):
            self.u[ci].getValuesGlobalExteriorTrace(self.ebqe[('v',ci)],self.ebqe[('u',ci)])
            if ('grad(u)',ci) in self.ebqe:
                self.u[ci].getGradientValuesGlobalExteriorTrace(self.ebqe[('grad(v)',ci)],self.ebqe[('grad(u)',ci)])
        #
        #get coefficients at the element boundary quadrature points
        #
        self.coefficients.evaluate(t = self.timeIntegration.t, c = self.ebqe)
        if self.movingDomain and self.coefficients.movingDomain:
            self.coefficients.updateToMovingDomain(self.timeIntegration.t,self.ebqe)
#             for ci in range(self.nc):
#                 if self.ebqe.has_key(('f',ci)):
#                     for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
#                         for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                             for I in range(self.nSpace_global):
#                                 if self.ebqe.has_key(('m',ci)):
#                                     self.ebqe[('f',ci)][ebNE,k,I]-=self.ebqe[('m',ci)][ebNE,k]*self.ebqe['xt'][ebNE,k,I]
#                                 else:
#                                     self.ebqe[('f',ci)][ebNE,k,I]-=self.ebqe['xt'][ebNE,k,I]
#                     for cj in range(self.nc):
#                         if self.ebqe.has_key(('dm',ci,cj)):
#                             for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
#                                 for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                     for I in range(self.nSpace_global):
#                                         self.ebqe[('df',ci,cj)][ebNE,k,I]-=self.ebqe[('dm',ci,cj)][ebNE,k]*self.ebqe['xt'][ebNE,k,I]
        #
        #time integration
        #
        self.timeIntegration.calculateExteriorElementBoundaryCoefficients(self.ebqe)
        #
        #get phi at the quadrature points if necessary
        #
        for ck,cjDict in self.coefficients.potential.items():
            for cj,flag in cjDict.items():
                if flag == 'nonlinear':
                    self.phi[ck].getValuesGlobalExteriorTrace(self.ebqe[('v',ck)],
                                                self.ebqe[('phi',ck)])
                    self.phi[ck].getGradientValuesGlobalExteriorTrace(self.ebqe[('grad(v)',ck)],
                                                                      self.ebqe[('grad(phi)',ck)])
                else:
                    self.ebqe[('phi',ck)][:]=self.ebqe[('u',ck)]
                    self.ebqe[('dphi',ck,cj)].fill(1.0)
                    self.ebqe[('grad(phi)',ck)][:]=self.ebqe[('grad(u)',ck)]
        #
        # calculate the averages and jumps at element boundaries
        #
        if self.numericalFlux is not None and not isinstance(self.numericalFlux, NumericalFlux.DoNothing):
            self.numericalFlux.calculateExteriorNumericalFlux(self.inflowFlag,self.q,self.ebqe)
        else:
            #cek this wll go away
            for ci,cjDict in self.coefficients.advection.items():
                if (self.fluxBoundaryConditions[ci] == 'outFlow' or
                    self.fluxBoundaryConditions[ci] == 'mixedFlow'):
                    for cj in cjDict:
                        cnumericalFlux.calculateExteriorNumericalAdvectiveFlux_NoBC(self.mesh.exteriorElementBoundariesArray,
                                                                                    self.mesh.elementBoundaryElementsArray,
                                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                    self.inflowFlag[ci],
                                                                                    self.ebqe['n'],
                                                                                    self.ebqe[('f',ci)],
                                                                                    self.ebqe[('df',ci,cj)],
                                                                                    self.ebqe[('advectiveFlux',ci)],
                                                                                    self.ebqe[('dadvectiveFlux_left',ci,cj)])
                        # The faces shared with another rank are exterior
                        # faces of this subdomain and interior to the global
                        # mesh, so the flux just computed on them is applied
                        # to the inside of the domain. Mask it off afterwards:
                        # the kernel indexes every ebqe array by its own loop
                        # counter, so it cannot be given a shorter face list.
                        _mask = MeshTools.rankInterfaceExteriorMask(self.mesh)
                        if _mask is not None:
                            self.ebqe[('advectiveFlux',ci)][_mask] = 0.0
                            self.ebqe[('dadvectiveFlux_left',ci,cj)][_mask] = 0.0
        for ci,fbcObject  in self.fluxBoundaryConditionsObjectsDict.items():
            if ci in self.coefficients.advection:
                i0,i1 = fbcObject.evaluate('advective',fbcObject.advectiveFluxBoundaryConditionsDict,
                                           self.ebqe['x'],self.ebqe[('advectiveFlux',ci)],self.timeIntegration.t)
                for cj in list(self.coefficients.advection[ci].keys()):
                    self.ebqe[('dadvectiveFlux_left',ci,cj)][i0,i1] = 0.0
            for ck,diffusiveFluxBoundaryConditionsDict in fbcObject.diffusiveFluxBoundaryConditionsDictDict.items():
                fbcObject.evaluate(('diffusive',ck),diffusiveFluxBoundaryConditionsDict,
                                   self.ebqe['x'],self.ebqe[('diffusiveFlux',ck,ci)],self.timeIntegration.t)
        for ci,sbcObject  in self.stressFluxBoundaryConditionsObjectsDict.items():
            sbcObject.evaluate('stress',sbcObject.stressFluxBoundaryConditionsDict,
                               self.ebqe['x'],self.ebqe[('stressFlux',ci)],self.timeIntegration.t)

    def calculateQuadrature(self):
        logEvent(memory("ElementQuadrature","OneLevelTransport"),level=4)
        logEvent("Element Quadrature",level=3)
        self.calculateElementQuadrature()
        logEvent(memory("ElementBoundaryQuadrature","OneLevelTransport"),level=4)
        logEvent("Element Boundary Quadrature",level=3)
        self.calculateElementBoundaryQuadrature()
        logEvent(memory("Global ExteriorElementBoundaryQuadrature","OneLevelTransport"),level=4)
        logEvent("Global Exterior Element Boundary Quadrature",level=3)
        self.calculateExteriorElementBoundaryQuadrature()
        logEvent(memory("exiting calcualteQuadrature","OneLevelTransport"),level=4)
    def updateAfterMeshMotion(self):
        self.calculateQuadrature()#not always the right thing to do (e.g. for optimized models)
    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        #
        #get physical locations of quadrature points and jacobian information there
        #assume all components live on the same mesh
        #
        self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
                                                  self.q['x'])
        
        if self.movingDomain:
            if self.tLast_mesh is not None:
                self.q['xt'][:]=self.q['x']
                self.q['xt']-=self.q['x_last']
                alpha = 1.0/(self.t_mesh-self.tLast_mesh)
                self.q['xt']*=alpha
            else:
                self.q['xt'][:]=0.0
            self.q['x_last'][:]=self.q['x']
        self.u[0].femSpace.elementMaps.getJacobianValues(self.elementQuadraturePoints,
                                                         self.q['J'],
                                                         self.q['inverse(J)'],
                                                         self.q['det(J)'])
        self.q['abs(det(J))']=numpy.absolute(self.q['det(J)'])
        #
        #
        for ci in range(self.nc):
            if ('dV_u',ci) in self.q:
                cfemIntegrals.calculateIntegrationWeights(self.q['abs(det(J))'],
                                                          self.elementQuadratureWeights[('u',ci)],
                                                          self.q[('dV_u',ci)])

            #
            if 'dV' not in self.q and ('dV_u',ci) in self.q:
                self.q['dV'] = self.q[('dV_u',ci)]

        #
        #get shape information at the quadrature points
        #
        for ci in range(self.nc):
            if ('w',ci) in self.q:
                self.testSpace[ci].getBasisValues(self.elementQuadraturePoints,
                                                  self.q[('w',ci)])
                for I in self.integralKeys:
                    if ('w*dV_'+I,ci) in self.q:
                        cfemIntegrals.calculateWeightedShape(self.elementQuadratureWeights[(I,ci)],
                                                             self.q['abs(det(J))'],
                                                             self.q[('w',ci)],
                                                             self.q[('w*dV_'+I,ci)])
                    for ck in range(self.nc):
                        if ('w*dV_'+I,ck,ci) in self.q:
                            cfemIntegrals.calculateWeightedShape(self.elementQuadratureWeights[(I,ci,ck)],
                                                                 self.q['abs(det(J))'],
                                                                 self.q[('w',ci)],
                                                                 self.q[('w*dV_'+I,ck,ci)])
            if ('grad(w)',ci) in self.q:
                self.testSpace[ci].getBasisGradientValues(self.elementQuadraturePoints,
                                                          self.q['inverse(J)'],
                                                          self.q[('grad(w)',ci)])
                for I in self.integralKeys:
                    if ('grad(w)*dV_'+I,ci) in self.q:
                        cfemIntegrals.calculateWeightedShapeGradients(self.elementQuadratureWeights[(I,ci)],
                                                                      self.q['abs(det(J))'],
                                                                      self.q[('grad(w)',ci)],
                                                                      self.q[('grad(w)*dV_'+I,ci)])
                    for ck in range(self.nc):
                        if ('grad(w)*dV_'+I,ck,ci) in self.q:
                            cfemIntegrals.calculateWeightedShapeGradients(self.elementQuadratureWeights[(I,ci,ck)],
                                                                          self.q['abs(det(J))'],
                                                                          self.q[('grad(w)',ci)],
                                                                          self.q[('grad(w)*dV_'+I,ck,ci)])
            if ('Hess(w)',ci) in self.q:
                self.testSpace[ci].getBasisHessianValues(self.elementQuadraturePoints,
                                                         self.q['inverse(J)'],
                                                         self.q[('Hess(w)',ci)])
                for I in self.integralKeys:
                    for ck in range(self.nc):
                        if ('Hess(w)*dV_'+I,ck,ci) in self.q:
                            cfemIntegrals.calculateWeightedShapeHessians(self.elementQuadratureWeights[(I,ci,ck)],
                                                                         self.q['abs(det(J))'],
                                                                         self.q[('Hess(w)',ci)],
                                                                         self.q[('Hess(w)*dV_'+I,ck,ci)])
        for cj in range(self.nc):
            if ('v',cj) in self.q:
                self.u[cj].femSpace.getBasisValues(self.elementQuadraturePoints,
                                                   self.q[('v',cj)])
            if ('grad(v)',cj) in self.q:
                self.u[cj].femSpace.getBasisGradientValues(self.elementQuadraturePoints,
                                                           self.q['inverse(J)'],
                                                           self.q[('grad(v)',cj)])
            if ('Hess(v)',cj) in self.q:
                self.u[cj].femSpace.getBasisHessianValues(self.elementQuadraturePoints,
                                                          self.q['inverse(J)'],
                                                          self.q[('Hess(v)',cj)])
        #
        #get tensor products of shape information
        #
        if not self.lowmem:
            for ci in range(self.nc):
                for cj in range(self.nc):
                    for I in self.integralKeys:
                        if ('vXw*dV_'+I,cj,ci) in self.q:
                            cfemIntegrals.calculateShape_X_weightedShape(self.q[('v',cj)],
                                                                         self.q[('w*dV_'+I,ci)],
                                                                         self.q[('vXw*dV_'+I,cj,ci)])
                        if ('vXgrad(w)*dV_'+I,cj,ci) in self.q:
                            cfemIntegrals.calculateShape_X_weightedGradShape(self.q[('v',cj)],
                                                                             self.q[('grad(w)*dV_'+I,ci)],
                                                                             self.q[('vXgrad(w)*dV_'+I,cj,ci)])
                        if  ('grad(v)Xw*dV_'+I,cj,ci) in self.q:
                            cfemIntegrals.calculateGradShape_X_weightedShape(self.q[('grad(v)',cj)],
                                                                             self.q[('w*dV_'+I,ci)],
                                                                             self.q[('grad(v)Xw*dV_'+I,cj,ci)])
                        for ck in range(self.nc):
                            if ('vXw*dV_'+I,ck,cj,ci) in self.q:
                                cfemIntegrals.calculateShape_X_weightedShape(self.q[('v',cj)],
                                                                             self.q[('w*dV_'+I,ck,ci)],
                                                                             self.q[('vXw*dV_'+I,ck,cj,ci)])
                            if ('grad(v)Xgrad(w)*dV_'+I,ck,cj,ci) in self.q:
                                cfemIntegrals.calculateGradShape_X_weightedGradShape(self.q[('grad(v)',cj)],
                                                                                     self.q[('grad(w)*dV_'+I,ck,ci)],
                                                                                     self.q[('grad(v)Xgrad(w)*dV_'+I,ck,cj,ci)])
        #
        #cek this is where we need check for nonlinear phi, I think
        #
        for ci in range(self.nc):
            if ci in self.coefficients.potential:
                ##\todo put back in ability to leave out interpolation points
                pass

                #if self.coefficients.potential[ci][ci] != 'u':
                #    self.phi[ci].femSpace.updateInterpolationPoints()
                #self.dphi[ci].femSpace.interpolationPoints = self.phi.femSpace.interpolationPoints
        #cek todo, I think this needs to be moved, we don't want to re-initialize variables, just update Eulerian coordinates and quadrature formulas
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t,self.q)
        if self.stabilization is not None:
            #mwf hack interpolating subgrid error
            if self.stabilization.usesGradientStabilization:
                self.phi[0].femSpace.elementMaps.getJacobianValues(self.phi[0].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                                   self.phi_ip['J'],
                                                                   self.phi_ip['inverse(J)'],
                                                                   self.phi_ip['det(J)'])
                for cj in range(self.nc):
                    if ('v',cj) in self.phi_ip:
                        self.phi[cj].femSpace.getBasisValues(self.phi[cj].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                             self.phi_ip[('v',cj)])
                    if ('grad(v)',cj) in self.phi_ip:
                        self.phi[cj].femSpace.getBasisGradientValues(self.phi[cj].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                                   self.phi_ip['inverse(J)'],
                                                                   self.phi_ip[('grad(v)',cj)])
                    if ('Hess(v)',cj) in self.phi_ip:
                        self.phi[cj].femSpace.getBasisHessianValues(self.phi[cj].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                                                  self.phi_ip['inverse(J)'],
                                                                  self.phi_ip[('Hess(v)',cj)])

                #mwf check to see if this needs to go after coefficients.initializeGeneralizedInterpolationPointQuadrature
                self.stabilization.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q,self.phi_ip)
            else:
                self.stabilization.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing is not None:
            self.shockCapturing.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q)
        #update interpolation points here?
        ##\todo put logic in to evaluate only when necessary
        self.coefficients.initializeGeneralizedInterpolationPointQuadrature(self.timeIntegration.t,self.phi_ip)
    def calculateElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on element boundaries.

        This function should be called only when the mesh changes.
        """
        #
        #cek get rid of trickiness
        #
        if 'x' not in list(self.ebq.keys()):
            return
        #
        #get physical locations of element boundary quadrature points
        #
        #assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getValuesTrace(self.elementBoundaryQuadraturePoints,
                                                      self.ebq['x'])
        #
        #get metric tensor and unit normals
        #
        if self.movingDomain:
            if self.tLast_mesh is not None:
                self.ebq['xt'][:]=self.ebq['x']
                self.ebq['xt']-=self.ebq['x_last']
                alpha = 1.0/(self.t_mesh-self.tLast_mesh)
                self.ebq['xt']*=alpha
            else:
                self.ebq['xt'][:]=0.0
            self.ebq['x_last'][:]=self.ebq['x']
            self.u[0].femSpace.elementMaps.getJacobianValuesTrace_movingDomain(self.elementBoundaryQuadraturePoints,
                                                                               self.ebq['xt'],
                                                                               self.ebq['inverse(J)'],
                                                                               self.ebq['g'],
                                                                               self.ebq['sqrt(det(g))'],
                                                                               self.ebq['n'])
#             self.u[0].femSpace.elementMaps.getJacobianValuesTrace(self.elementBoundaryQuadraturePoints,
#                                                                   self.ebq['inverse(J)'],
#                                                                   self.ebq['g'],
#                                                                   self.ebq['sqrt(det(g))'],
#                                                                   self.ebq['n'])
        else:
            self.u[0].femSpace.elementMaps.getJacobianValuesTrace(self.elementBoundaryQuadraturePoints,
                                                                  self.ebq['inverse(J)'],
                                                                  self.ebq['g'],
                                                                  self.ebq['sqrt(det(g))'],
                                                                  self.ebq['n'])
        useC=True
        #
        #the boundary quadrature points will correspond to the ones on the "left side"
        #of each element boundary so we need to fix the "right side"
        #
        #first copy left information into ebq_global storage
        if not useC:
            for ebN in range(self.mesh.nElementBoundaries_global):
                left_eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
                left_ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['x'][ebN,k] = self.ebq['x'][left_eN_global,left_ebN_element,k]
                    self.ebq_global['n'][ebN,k,:] = self.ebq['n'][left_eN_global,left_ebN_element,k]
            #now copy left information into right physical points on interior element boundaries
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                right_eN_global  = self.mesh.elementBoundaryElementsArray[ebN,1]
                right_ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq['x'][right_eN_global,right_ebN_element,k,:] = self.ebq_global['x'][ebN,k]
                    self.ebq['n'][right_eN_global,right_ebN_element,k,:] = - self.ebq_global['n'][ebN,k,:]
        else:
            cfemIntegrals.copyLeftElementBoundaryInfo(self.mesh.elementBoundaryElementsArray,
                                                      self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                      self.mesh.exteriorElementBoundariesArray,
                                                      self.mesh.interiorElementBoundariesArray,
                                                      self.ebq['x'],
                                                      self.ebq['n'],
                                                      self.ebq_global['x'],
                                                      self.ebq_global['n'])
        if self.movingDomain:
            cfemIntegrals.copyLeftElementBoundaryInfo_movingDomain(self.mesh.elementBoundaryElementsArray,
                                                                   self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                   self.mesh.exteriorElementBoundariesArray,
                                                                   self.mesh.interiorElementBoundariesArray,
                                                                   self.ebq['xt'])

#         if self.movingDomain:
#             if self.tLast_mesh is not None:
#                 self.ebq['xt']-=self.ebq['x']
#                 alpha = -1.0/(self.t_mesh - self.tLast_mesh)
#                 self.ebq['xt']*=alpha
#                 #modify the metric tensor and normal
#                 if self.useC_md:
#                     cfemIntegrals.updateBoundaryInfo_movingDomain(self.ebq['xt'],self.ebq['n'],self.ebq['sqrt(det(g))'])
#                 else:
#                     for eN in range(self.mesh.nElements_global):
#                         for ebN in range(self.mesh.nElementBoundaries_element):
#                             for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                                 n_dot_xt=0.0
#                                 xt_dot_xt=0.0
#                                 for I in range(self.nSpace_global):
#                                     n_dot_xt += self.ebq['n'][eN,ebN,k,I]*self.ebq['xt'][eN,ebN,k,I]
#                                     xt_dot_xt += self.ebq['xt'][eN,ebN,k,I]*self.ebq['xt'][eN,ebN,k,I]
#                                 self.ebq['sqrt(det(g))'][eN,ebN,k]*=math.sqrt(1.0+xt_dot_xt)
#                                 self.ebq['n'][eN,ebN,k]/=math.sqrt(1.0+n_dot_xt*n_dot_xt)
#             else:
#                 self.ebq['xt'][:]=0.0
        #now map the physical points back to the reference element
        #assume all components live  on same mesh
        self.u[0].femSpace.elementMaps.getInverseValuesTrace(self.ebq['inverse(J)'],self.ebq['x'],self.ebq['hat(x)'])
        self.u[0].femSpace.elementMaps.getPermutations(self.ebq['hat(x)'])
        #
        #since the points on the reference boundary may be reordered on many right element boundaries, we
        #have to use an array of reference boundary points on all element boundaries
        #first copy the left reference element boundary quadrature points from the reference element boundary
        useC = True
        if useC == False:
            for ebN in range(self.mesh.nElementBoundaries_global):
                left_eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
                left_ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq['bar(x)'][left_eN_global,left_ebN_element,k,:] = self.elementBoundaryQuadraturePoints[k]
            #now get the right element boundary quadrature points and normals from the inverse map
            #note that the inverses are only defined on the boundaries of the reference element
            boundaryMapInverseList = self.u[0].femSpace.referenceFiniteElement.referenceElement.boundaryMapInverseList
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                left_eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
                right_eN_global  = self.mesh.elementBoundaryElementsArray[ebN,1]
                left_ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                right_ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq['bar(x)'][right_eN_global,right_ebN_element,k,:] = boundaryMapInverseList[right_ebN_element](
                        self.ebq['hat(x)'][right_eN_global,right_ebN_element,k])
        #
        #get the shape information at the reference element boundary quadrature points
        #
        for ci in range(self.nc):
            if ('w',ci) in self.ebq:
                if useC == True:
                    self.testSpace[ci].getBasisValuesTrace(self.u[0].femSpace.elementMaps.permutations,
                                                           self.ebq['hat(x)'],
                                                           self.ebq[('w',ci)])
                else:
                    self.testSpace[ci].getBasisValuesTraceAtArray(self.ebq['bar(x)'],
                                                                  self.ebq[('w',ci)])
                for I in self.elementBoundaryIntegralKeys:
                    if (I,ci) in self.ebq:
                        cfemIntegrals.calculateWeightedShapeTrace(self.elementBoundaryQuadratureWeights[(I,ci)],
                                                                  self.ebq['sqrt(det(g))'],
                                                                  self.ebq[('w',ci)],
                                                                  self.ebq[('w*dS_'+I,ci)])
                    #for w*dS_a
                    for ck in range(self.nc):
                        if (I,ci,ck) in self.ebq:
                            cfemIntegrals.calculateWeightedShapeTrace(self.elementBoundaryQuadratureWeights[(I,ci,ck)],
                                                                      self.ebq['sqrt(det(g))'],
                                                                      self.ebq[('w',ci)],
                                                                      self.ebq[('w*dS_'+I,ci,ck)])
                if ('w*dS_f',ci) not in self.ebq and len(self.elementBoundaryIntegralKeys) > 0:
                    for ck in range(self.nc):
                        try:
                            self.ebq[('w*dS_f',ci)] = self.ebq[('w*dS_a',ci,ck)]
                            break
                        except:
                            try:
                                self.ebq[('w*dS_f',ci)] = self.ebq[('w*dS_H',ci)]
                            except:
                                pass
                    assert ('w*dS_f',ci) in self.ebq
        for cj in range(self.nc):
            if ('v',cj) in self.ebq:
                if useC == True:
                    self.u[cj].femSpace.getBasisValuesTrace(self.u[0].femSpace.elementMaps.permutations,
                                                            self.ebq['hat(x)'],
                                                            self.ebq[('v',cj)])
                else:
                    self.u[cj].femSpace.getBasisValuesTraceAtArray(self.ebq['bar(x)'],
                                                                   self.ebq[('v',cj)])
            if ('grad(v)',cj) in self.ebq:
                if useC == True:
                    self.u[cj].femSpace.getBasisGradientValuesTrace(self.u[0].femSpace.elementMaps.permutations,
                                                                    self.ebq['hat(x)'],
                                                                    self.ebq['inverse(J)'],
                                                                    self.ebq[('grad(v)',cj)])
                else:
                    self.u[cj].femSpace.getBasisGradientValuesTraceAtArray(self.ebq['bar(x)'],
                                                                           self.ebq['inverse(J)'],
                                                                           self.ebq[('grad(v)',cj)])
        #tensor products of shape information
        if not self.lowmem:
            for ci in range(self.nc):
                for cj in range(self.nc):
                    for I in self.elementBoundaryIntegralKeys:
                        if ('vXw*dS_'+I,cj,ci) in self.ebq:
                            cfemIntegrals.calculateShape_X_weightedShapeTrace(self.ebq[('v',cj)],
                                                                              self.ebq[('w*dS_'+I,ci)],
                                                                              self.ebq[('vXw*dS_'+I,cj,ci)])
                        for ck in range(self.nc):
                            if ('grad(v)Xw*dS_'+I,ck,cj,ci) in self.ebq:
                                cfemIntegrals.calculateGradShape_X_weightedShapeTrace(self.ebq[('grad(v)',cj)],
                                                                                      self.ebq[('w*dS_'+I,ck,ci)],
                                                                                      self.ebq[('grad(v)Xw*dS_'+I,ck,cj,ci)])

        #setup flux boundary conditions
        #note for now requires advectiveFluxBoundaryConditions to have key if want
        #to set diffusiveFluxBCs
        #mwf should be able to get rid of this too
        #setup flux boundary conditions
        fluxBoundaryCondition_components = set()
        if self.advectiveFluxBoundaryConditionsSetterDict is not None:
            fluxBoundaryCondition_components = fluxBoundaryCondition_components.union(set(self.advectiveFluxBoundaryConditionsSetterDict.keys()))
        if self.diffusiveFluxBoundaryConditionsSetterDictDict is not None:
            fluxBoundaryCondition_components = fluxBoundaryCondition_components.union(set(self.diffusiveFluxBoundaryConditionsSetterDictDict.keys()))
        self.fluxBoundaryConditionsObjectsDictGlobalElementBoundary = {}
        for ci in fluxBoundaryCondition_components:
            if ci in self.advectiveFluxBoundaryConditionsSetterDict:
                advectiveFluxBC = self.advectiveFluxBoundaryConditionsSetterDict[ci]
            else:
                advectiveFluxBC = None
            if ci in self.diffusiveFluxBoundaryConditionsSetterDictDict:
                diffusiveFluxBC = self.diffusiveFluxBoundaryConditionsSetterDictDict[ci]
            else:
                diffusiveFluxBC = {}
            self.fluxBoundaryConditionsObjectsDictGlobalElementBoundary[ci] = FluxBoundaryConditionsGlobalElementBoundaries(self.mesh,
                                                                                                                          self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                                                          self.ebq_global[('x')],
                                                                                                                          advectiveFluxBC,
                                                                                                                          diffusiveFluxBC)
        #
        for ci in range(self.nc):
            if ('dS_u',ci) in self.ebq:
                cfemIntegrals.calculateElementBoundaryIntegrationWeights(self.ebq['sqrt(det(g))'],
                                                                         self.elementBoundaryQuadratureWeights[('u',ci)],
                                                                         self.ebq[('dS_u',ci)])
        self.coefficients.initializeElementBoundaryQuadrature(self.timeIntegration.t,self.ebq,self.ebq_global)
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
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        #
        #get metric tensor and unit normals
        #
        if self.movingDomain:
            if self.tLast_mesh is not None:
                self.ebqe['xt'][:]=self.ebqe['x']
                self.ebqe['xt']-=self.ebqe['x_last']
                alpha = 1.0/(self.t_mesh-self.tLast_mesh)
                self.ebqe['xt']*=alpha
            else:
                self.ebqe['xt'][:]=0.0
            self.ebqe['x_last'][:]=self.ebqe['x']
            self.u[0].femSpace.elementMaps.getJacobianValuesGlobalExteriorTrace_movingDomain(self.elementBoundaryQuadraturePoints,
                                                                                             self.ebqe['xt'],
                                                                                             self.ebqe['inverse(J)'],
                                                                                             self.ebqe['g'],
                                                                                             self.ebqe['sqrt(det(g))'],
                                                                                             self.ebqe['n'])
        else:
            self.u[0].femSpace.elementMaps.getJacobianValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                                self.ebqe['inverse(J)'],
                                                                                self.ebqe['g'],
                                                                                self.ebqe['sqrt(det(g))'],
                                                                                self.ebqe['n'])
#         if self.movingDomain:
#             if self.tLast_mesh is not None:
#                 self.ebqe['xt']-=self.ebqe['x']
#                 alpha = -1.0/(self.t_mesh - self.tLast_mesh)
#                 self.ebqe['xt']*=alpha
#                 #modify the metric tensor and normal
#                 if self.useC_md:
#                     cfemIntegrals.updateBoundaryInfo_movingDomain(self.ebqe['xt'],self.ebqe['n'],self.ebqe['sqrt(det(g))'])
#                 else:
#                     for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
#                         for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
#                             n_dot_xt=0.0
#                             xt_dot_xt=0.0
#                             for I in range(self.nSpace_global):
#                                 n_dot_xt += self.ebqe['n'][ebNE,k,I]*self.ebqe['xt'][ebNE,k,I]
#                                 xt_dot_xt += self.ebqe['xt'][ebNE,k,I]*self.ebqe['xt'][ebNE,k,I]
#                             self.ebqe['sqrt(det(g))'][ebNE,k] *= math.sqrt(1.0+xt_dot_xt)
#                             self.ebqe['n'][ebNE,k] /= math.sqrt(1.0+n_dot_xt*n_dot_xt)
#             else:
#                 self.ebqe['xt'][:]=0.0
        #now map the physical points back to the reference element
        #assume all components live  on same mesh
        #self.u[0].femSpace.elementMaps.getInverseValuesGlobalExteriorTrace(self.ebqe['inverse(J)'],self.ebqe['x'],self.ebqe['hat(x)'])
        #
        #since the points on the reference boundary may be reordered on many right element boundaries, we
        #have to use an array of reference boundary points on all element boundaries
        #first copy the left reference element boundary quadrature points from the reference element boundary
        #mwf I think it's safe to get rid of bar(x) for ebqe ...
        useC = True
        #if useC == False:
        #    for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
        #        for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
        #            self.ebqe['bar(x)'][ebNE,k,:] = self.elementBoundaryQuadraturePoints[k]
        #
        #get the shape information at the reference element boundary quadrature points
        #
        for ci in range(self.nc):
            if ('w',ci) in self.ebqe:
                if useC == True:
                    self.testSpace[ci].getBasisValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                         self.ebqe[('w',ci)])
                else:
                    self.testSpace[ci].getBasisValuesGlobalExteriorTraceAtArray(self.ebqe['bar(x)'],
                                                                                self.ebqe[('w',ci)])

                for I in self.elementBoundaryIntegralKeys:
                    if ('w*dS_'+I,ci) in self.ebqe and (I,ci) in self.elementBoundaryQuadratureWeights:
                        cfemIntegrals.calculateWeightedShapeGlobalExteriorTrace(self.mesh.exteriorElementBoundariesArray,
                                                                                self.mesh.elementBoundaryElementsArray,
                                                                                self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                self.elementBoundaryQuadratureWeights[(I,ci)],
                                                                                self.ebqe['sqrt(det(g))'],
                                                                                self.ebqe[('w',ci)],
                                                                                self.ebqe[('w*dS_'+I,ci)])

                    #for w*dS_a
                    for ck in range(self.nc):
                        if ('w*dS_'+I,ci,ck) in self.ebqe:
                            cfemIntegrals.calculateWeightedShapeGlobalExteriorTrace(self.mesh.exteriorElementBoundariesArray,
                                                                                    self.mesh.elementBoundaryElementsArray,
                                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                    self.elementBoundaryQuadratureWeights[(I,ci,ck)],
                                                                                    self.ebqe['sqrt(det(g))'],
                                                                                    self.ebqe[('w',ci)],
                                                                                    self.ebqe[('w*dS_'+I,ci,ck)])
                if ('w*dS_f',ci) not in self.ebqe:
                    for ck in range(self.nc):
                        try:
                            self.ebqe[('w*dS_f',ci)] = self.ebqe[('w*dS_a',ci,ck)]
                            break
                        except:
                            try:
                                self.ebqe[('w*dS_f',ci)] = self.ebqe[('w*dS_H',ci)]
                            except:
                                pass
                        try:
                            self.ebqe[('w*dS_f',ci)] = self.ebqe[('w*dS_sigma',ci)]
                        except:
                            pass
                    assert ('w*dS_f',ci) in self.ebqe

        for cj in range(self.nc):
            if ('v',cj) in self.ebqe:
                if useC == True:
                    self.u[cj].femSpace.getBasisValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                          self.ebqe[('v',cj)])
                else:
                    self.u[cj].femSpace.getBasisValuesGlobalExteriorTraceAtArray(self.ebqe['bar(x)'],
                                                                                 self.ebqe[('v',cj)])
            if ('grad(v)',cj) in self.ebqe:
                if useC == True:
                    self.u[cj].femSpace.getBasisGradientValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                                  self.ebqe['inverse(J)'],
                                                                                  self.ebqe[('grad(v)',cj)])
                else:
                    self.u[cj].femSpace.getBasisGradientValuesGlobalExteriorTraceAtArray(self.ebqe['bar(x)'],
                                                                                         self.ebqe['inverse(J)'],
                                                                                         self.ebqe[('grad(v)',cj)])
        #tensor products of shape information
        if not self.lowmem:
            for ci in range(self.nc):
                for cj in range(self.nc):
                    for I in self.elementBoundaryIntegralKeys:
                        if ('vXw*dS_'+I,cj,ci) in self.ebqe:
                            cfemIntegrals.calculateShape_X_weightedShapeGlobalExteriorTrace(self.mesh.exteriorElementBoundariesArray,
                                                                                            self.mesh.elementBoundaryElementsArray,
                                                                                            self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                            self.ebqe[('v',cj)],
                                                                                            self.ebqe[('w*dS_'+I,ci)],
                                                                                            self.ebqe[('vXw*dS_'+I,cj,ci)])
                        for ck in range(self.nc):
                            if ('grad(v)Xw*dS_'+I,ck,cj,ci) in self.ebqe:
                                cfemIntegrals.calculateGradShape_X_weightedShapeGlobalExteriorTrace(self.mesh.exteriorElementBoundariesArray,
                                                                                                    self.mesh.elementBoundaryElementsArray,
                                                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                                    self.ebqe[('grad(v)',cj)],
                                                                                                    self.ebqe[('w*dS_'+I,ck,ci)],
                                                                                                    self.ebqe[('grad(v)Xw*dS_'+I,ck,cj,ci)])

        #setup flux boundary conditions
        fluxBoundaryCondition_components = set()
        if self.advectiveFluxBoundaryConditionsSetterDict is not None:
            fluxBoundaryCondition_components = fluxBoundaryCondition_components.union(set(self.advectiveFluxBoundaryConditionsSetterDict.keys()))
        if self.diffusiveFluxBoundaryConditionsSetterDictDict is not None:
            fluxBoundaryCondition_components = fluxBoundaryCondition_components.union(set(self.diffusiveFluxBoundaryConditionsSetterDictDict.keys()))
        self.fluxBoundaryConditionsObjectsDict = {}
        for ci in fluxBoundaryCondition_components:
            if ci in self.advectiveFluxBoundaryConditionsSetterDict:
                advectiveFluxBC = self.advectiveFluxBoundaryConditionsSetterDict[ci]
            else:
                advectiveFluxBC = None
            if ci in self.diffusiveFluxBoundaryConditionsSetterDictDict:
                diffusiveFluxBC = self.diffusiveFluxBoundaryConditionsSetterDictDict[ci]
            else:
                diffusiveFluxBC = {}
            self.fluxBoundaryConditionsObjectsDict[ci] = FluxBoundaryConditions(self.mesh,
                                                                                self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                self.ebqe[('x')],
                                                                                advectiveFluxBC,
                                                                                diffusiveFluxBC)
        self.stressFluxBoundaryConditionsObjectsDict = dict([(cj,FluxBoundaryConditions(self.mesh,
                                                                                        self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                        self.ebqe[('x')],
                                                                                        getStressFluxBoundaryConditions=self.stressFluxBoundaryConditionsSetterDict[cj]))
                                                         for cj in list(self.stressFluxBoundaryConditionsSetterDict.keys())])
        #mwf what do I need for getting raw weights in physical space on elements?
        #
        for ci in range(self.nc):
            if ('dS_u',ci) in self.ebqe:
                cfemIntegrals.calculateIntegrationWeights(self.ebqe['sqrt(det(g))'],
                                                          self.elementBoundaryQuadratureWeights[('u',ci)],
                                                          self.ebqe[('dS_u',ci)])
            if 'dS' not in self.ebqe and ('dS_u',ci) in self.ebqe:
                self.ebqe['dS'] = self.ebqe[('dS_u',ci)]
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t,self.ebqe)
    def setFreeDOF(self,free_u):
        """
        Set the free(non-Dirichlet) DOF from the global DOF
        """
        for cj in range(self.nc):
            fromFreeToGlobal=0 #direction copying
            cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                                   self.offset[cj],
                                                                   self.stride[cj],
                                                                   self.dirichletConditions[cj].global2freeGlobal_global_dofs,
                                                                   self.dirichletConditions[cj].global2freeGlobal_free_dofs,
                                                                   free_u,
                                                                   self.u[cj].dof)
    def updateLocal2Global(self):
        """
        Build a mapping between local DOF numbers and global free DOF
        numbers, that is, EXCLUDING Dirichlet DOF.  We have to use a
        list of local dof because of the exclusion of Dirichlet nodes
        (otherwise we could just loop over range(self.nDOF_element).
        """
        self.l2g=[{'nFreeDOF':numpy.zeros((self.mesh.nElements_global,),'i'),
                   'freeLocal':numpy.zeros((self.mesh.nElements_global,self.nDOF_trial_element[cj]),'i'),
                   'freeGlobal':numpy.zeros((self.mesh.nElements_global,self.nDOF_trial_element[cj]),cmeshTools.INDEX_DTYPE)} for cj in range(self.nc)]
        for cj in range(self.nc):
            for eN in range(self.mesh.nElements_global):
                nFreeDOF=0
                for j in range(self.nDOF_trial_element[cj]):
                    J = self.u[cj].femSpace.dofMap.l2g[eN,j]
                    if J in self.dirichletConditions[cj].global2freeGlobal:
                        self.l2g[cj]['freeLocal'][eN,nFreeDOF]=j
                        self.l2g[cj]['freeGlobal'][eN,nFreeDOF]=self.dirichletConditions[cj].global2freeGlobal[J]
                        nFreeDOF+=1
                self.l2g[cj]['nFreeDOF'][eN]=nFreeDOF
    def setInflowFlux(self):
        for ci in range(self.nc):
            cfemIntegrals.setInflowFlux(self.mesh.exteriorElementBoundariesArray,
                                        self.ebqe[('advectiveFlux',ci)],
                                        self.inflowFlux[ci])
    def setUnknowns(self,free_u):
        # copyBetweenFreeUnknownsAndGlobalUnknowns indexes free_u from C with no
        # bounds checking, so handing it a vector shorter than this level needs
        # corrupts whatever happens to follow it in memory instead of raising --
        # typically the dof arrays' own headers, which then surface far away as
        # a nonsensical shape. Callers iterating a multilevel model have passed
        # uList[0] for every level more than once, so check the contract here.
        if free_u.size < self.dim:
            raise ValueError(
                "setUnknowns needs at least %i free unknowns for this level but "
                "was given %i; a caller is probably passing the coarsest level's "
                "vector for every level" % (self.dim, free_u.size))
        for cj in range(self.nc):
            assert len(self.dirichletConditions[cj].global2freeGlobal_global_dofs) == len(self.dirichletConditions[cj].global2freeGlobal_free_dofs)
            fromFreeToGlobal=1 #direction copying
            cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                                   self.offset[cj],
                                                                   self.stride[cj],
                                                                   self.dirichletConditions[cj].global2freeGlobal_global_dofs,
                                                                   self.dirichletConditions[cj].global2freeGlobal_free_dofs,
                                                                   free_u,
                                                                   self.u[cj].dof)
    def initializeJacobian(self):
        #
        #build sparse  matrix and extract the stuff we  need  to directly load/update csr matrices
        #from element matrices
        #
        needNumericalFluxJacobian = False; needOutflowJacobian = False
        for ci in range(self.nc):
            #start off assuming anything implicit means have to numerical flux jacobian, then scale back
            #just to advection,diffusion,hamiltonian
            needNumericalFluxJacobian = (needNumericalFluxJacobian                         or
                                         self.timeIntegration.advectionIsImplicit[ci]      or
                                         self.timeIntegration.diffusionIsImplicit[ci]      or
                                         self.timeIntegration.hamiltonianIsImplicit[ci]    or
                                         self.timeIntegration.reactionIsImplicit[ci]       or
                                         self.timeIntegration.stabilizationIsImplicit[ci]  or
                                         self.timeIntegration.shockCapturingIsImplicit[ci])
            #for now assume needOutflowJacobian if anything is implicit for space
            needOutflowJacobian      = (needOutflowJacobian                               or
                                        self.timeIntegration.advectionIsImplicit[ci]      or
                                        self.timeIntegration.diffusionIsImplicit[ci]      or
                                        self.timeIntegration.hamiltonianIsImplicit[ci]    or
                                        self.timeIntegration.reactionIsImplicit[ci]       or
                                        self.timeIntegration.stabilizationIsImplicit[ci]  or
                                        self.timeIntegration.shockCapturingIsImplicit[ci])

        columnIndecesDict={}#replace with C++ map (this  collects column indeces for each row)
        logEvent("Building sparse matrix structure",level=2)
        self.sparsityInfo = csparsity.PySparsityInfo()
        useC=True
        for ci in range(self.nc):
            for cj in self.coefficients.stencil[ci]: #if we make stencil an array this can pass to C++
                if useC:
                    #immersed/embedded
                    try:
                        if self.hasCutCells:
                            hasNumericalFlux=1#edge-based ghost penalties are like numerical fluxes
                            needNumericalFluxJacobian_int = 1
                            hasDiffusionInMixedForm = 0
                            hasOutflowBoundary = 0
                            needsOutflowJacobian_int = 0
                    except:
                        self.hasCutCells=False
                    if not self.hasCutCells:
                        hasNumericalFlux=0
                        if self.numericalFlux is not None and self.numericalFlux.hasInterior:
                            hasNumericalFlux = 1
                        hasDiffusionInMixedForm = int(self.numericalFlux is not None and  self.numericalFlux.mixedDiffusion[ci] == True)
                        needNumericalFluxJacobian_int = int(needNumericalFluxJacobian)
                        hasOutflowBoundary = int(self.fluxBoundaryConditions[ci] == 'outFlow')
                        needsOutflowJacobian_int = int(needOutflowJacobian == True)
                    self.sparsityInfo.findNonzeros(self.mesh.nElements_global,
                                                   self.nDOF_test_element[ci],
                                                   self.nDOF_trial_element[cj],
                                                   self.l2g[ci]['nFreeDOF'],
                                                   self.l2g[ci]['freeGlobal'],
                                                   self.l2g[cj]['nFreeDOF'],
                                                   self.l2g[cj]['freeGlobal'],
                                                   self.offset[ci],
                                                   self.stride[ci],
                                                   self.offset[cj],
                                                   self.stride[cj],
                                                   hasNumericalFlux,
                                                   hasDiffusionInMixedForm,
                                                   needNumericalFluxJacobian_int,
                                                   self.mesh.nElementBoundaries_element,
                                                   self.mesh.elementNeighborsArray,
                                                   self.mesh.nInteriorElementBoundaries_global,
                                                   self.mesh.interiorElementBoundariesArray,
                                                   self.mesh.elementBoundaryElementsArray,
                                                   self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                   self.fluxBoundaryConditions[ci] == 'outFlow',
                                                   self.mesh.nExteriorElementBoundaries_global,
                                                   self.mesh.exteriorElementBoundariesArray,
                                                   hasOutflowBoundary,
                                                   needsOutflowJacobian_int)
                else:
                    for eN in range(self.mesh.nElements_global):
                        for ii in range(self.l2g[ci]['nFreeDOF'][eN]): #l2g is an array
                            I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][eN,ii] #offset  and stride can be arrays
                            if I not in columnIndecesDict:
                                columnIndecesDict[I]=set() #use C++ set
                            for jj in range(self.l2g[cj]['nFreeDOF'][eN]):
                                J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][eN,jj]
                                columnIndecesDict[I].add(J)
                        if (self.numericalFlux is not None and self.numericalFlux.mixedDiffusion[ci] == True and
                            needNumericalFluxJacobian == True):#pass in array of flags for numflux and mixed
                            for ebN in range(self.mesh.nElementBoundaries_element):
                                eN_ebN = self.mesh.elementNeighborsArray[eN,ebN]
                                if eN_ebN >= 0:
                                    for ii in range(self.l2g[ci]['nFreeDOF'][eN]):
                                        I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][eN,ii]
                                        for jj in range(self.l2g[cj]['nFreeDOF'][eN_ebN]):
                                            J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][eN_ebN,jj]
                                            columnIndecesDict[I].add(J)
                    if self.numericalFlux is not None and needNumericalFluxJacobian == True:
                        for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                            ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                            left_eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
                            right_eN_global  = self.mesh.elementBoundaryElementsArray[ebN,1]
                            left_ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                            right_ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
                            for ii in range(self.l2g[ci]['nFreeDOF'][left_eN_global]):
                                left_I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][left_eN_global,ii]
                                if left_I not in columnIndecesDict:
                                    columnIndecesDict[left_I]=set()
                                for jj in range(self.l2g[cj]['nFreeDOF'][left_eN_global]):
                                    left_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][left_eN_global,jj]
                                    columnIndecesDict[left_I].add(left_J)
                                for jj in range(self.l2g[cj]['nFreeDOF'][right_eN_global]):
                                    right_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][right_eN_global,jj]
                                    columnIndecesDict[left_I].add(right_J)
                            for ii in range(self.l2g[ci]['nFreeDOF'][right_eN_global]):
                                right_I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][right_eN_global,ii]
                                if right_I not in columnIndecesDict:
                                    columnIndecesDict[right_I]=set()
                                for jj in range(self.l2g[cj]['nFreeDOF'][left_eN_global]):
                                    left_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][left_eN_global,jj]
                                    columnIndecesDict[right_I].add(left_J)
                                for  jj  in range(self.l2g[cj]['nFreeDOF'][right_eN_global]):
                                    right_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][right_eN_global,jj]
                                    columnIndecesDict[right_I].add(right_J)
                            if self.numericalFlux.mixedDiffusion[ci] == True:
                                for ebN_eN in range(self.mesh.nElementBoundaries_element):
                                    left_eN_ebN = self.mesh.elementNeighborsArray[left_eN_global,ebN_eN]
                                    right_eN_ebN = self.mesh.elementNeighborsArray[right_eN_global,ebN_eN]
                                    for ii in range(self.l2g[ci]['nFreeDOF'][left_eN_global]):
                                        left_I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][left_eN_global,ii]
                                        if left_eN_ebN >= 0:
                                            for jj in range(self.l2g[cj]['nFreeDOF'][left_eN_ebN]):
                                                left_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][left_eN_ebN,jj]
                                                columnIndecesDict[left_I].add(left_J)
                                        if right_eN_ebN >= 0:
                                            for jj in range(self.l2g[cj]['nFreeDOF'][right_eN_ebN]):
                                                right_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][right_eN_ebN,jj]
                                                columnIndecesDict[left_I].add(right_J)
                                    for ii in range(self.l2g[ci]['nFreeDOF'][right_eN_global]):
                                        right_I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][right_eN_global,ii]
                                        if left_eN_ebN >= 0:
                                            for jj in range(self.l2g[cj]['nFreeDOF'][left_eN_ebN]):
                                                left_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][left_eN_ebN,jj]
                                                columnIndecesDict[right_I].add(left_J)
                                        if right_eN_ebN >= 0:
                                            for  jj  in range(self.l2g[cj]['nFreeDOF'][right_eN_ebN]):
                                                right_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][right_eN_ebN,jj]
                                                columnIndecesDict[right_I].add(right_J)
                    if ((self.numericalFlux is not None and needNumericalFluxJacobian == True) or
                        (self.fluxBoundaryConditions[ci] == 'outFlow' and needOutflowJacobian == True)):#BC flag to array
                        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                            eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
                            ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                            for  ii  in range(self.l2g[ci]['nFreeDOF'][eN_global]):
                                I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][eN_global,ii]
                                if I not in columnIndecesDict:
                                    columnIndecesDict[I]=set()
                                for  jj  in range(self.l2g[cj]['nFreeDOF'][eN_global]):
                                    J = self.offset[cj] + self.stride[cj]*self.l2g[cj]['freeGlobal'][eN_global,jj]
                                    columnIndecesDict[I].add(J)
                            if self.numericalFlux is not None and self.numericalFlux.mixedDiffusion[ci] == True:
                                for ebN_eN in range(self.mesh.nElementBoundaries_element):
                                    eN_ebN = self.mesh.elementNeighborsArray[eN_global,ebN_eN]
                                    for  ii  in range(self.l2g[ci]['nFreeDOF'][eN_global]):
                                        I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][eN_global,ii]
                                    if eN_ebN >= 0:
                                        for jj in range(self.l2g[cj]['nFreeDOF'][eN_ebN]):
                                            J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][eN_ebN,jj]
                                            columnIndecesDict[I].add(J)
        if useC:
            (self.rowptr,self.colind,self.nnz,self.nzval)  = self.sparsityInfo.getCSR()
        else:
            self.nnz = 0
            self.rowptr = numpy.zeros(self.nFreeVDOF_global+1,CSR_INDEX_DTYPE)
            lastIndex=0
            columnOffsetDict={}
            for I in range(self.nFreeVDOF_global):
                columnIndeces=list(columnIndecesDict[I])
                columnIndeces.sort()
                self.rowptr[I]=lastIndex
                lastIndex += len(columnIndeces)
            self.rowptr[self.nFreeVDOF_global]=lastIndex
            self.nnz = lastIndex
            self.colind = numpy.zeros((self.nnz,),CSR_INDEX_DTYPE)
            self.connectionList = [list(columnIndecesDict[I]-set([I])) for I in range(self.nFreeVDOF_global)]
            for I in range(self.nFreeVDOF_global):
                columnIndeces=list(columnIndecesDict[I])
                columnIndeces.sort()
                for columnOffset,J in enumerate(columnIndeces):
                    columnOffsetDict[(I,J)] = columnOffset
                    self.colind[self.rowptr[I]+columnOffset]=J
            self.nzval = numpy.zeros((self.nnz,),'d')
        #end first pass at translation to C, return rowptr,nnz,colind, connection list can be replaced by colind and rowptr
        if self.matType == superluWrappers.SparseMatrix:
            self.jacobian = SparseMat(self.nFreeVDOF_global,self.nFreeVDOF_global,self.nnz,self.nzval,self.colind,self.rowptr)
        elif self.matType == numpy.array:
            self.jacobian = Mat(self.nFreeVDOF_global,self.nFreeVDOF_global)
        else:
            raise TypeError("Matrix type must be sparse matrix or array")
        self.csrRowIndeces = {}
        self.csrColumnOffsets = {}
        self.csrColumnOffsets_eNebN = {}
        self.csrColumnOffsets_eb = {}
        self.csrColumnOffsets_eb_eNebN = {}
        for ci in range(self.nc):
            for cj in self.coefficients.stencil[ci]:
                hasNumericalFlux = int(self.numericalFlux is not None)
                hasDiffusionInMixedForm = int(self.numericalFlux is not None and  self.numericalFlux.mixedDiffusion[ci] == True)
                needNumericalFluxJacobian_int = int(needNumericalFluxJacobian)
                hasOutflowBoundary = int(self.fluxBoundaryConditions[ci] == 'outFlow')
                needsOutflowJacobian_int = int(needOutflowJacobian == True)
                memory()
                self.csrRowIndeces[(ci,cj)] = numpy.zeros((self.mesh.nElements_global,
                                                             self.nDOF_test_element[ci]),CSR_INDEX_DTYPE)
                logEvent(memory("csrRowIndeces","OneLevelTransport"),level=4)
                self.csrColumnOffsets[(ci,cj)] = numpy.zeros((self.mesh.nElements_global,
                                                                self.nDOF_test_element[ci],self.nDOF_trial_element[cj]),CSR_INDEX_DTYPE)
                logEvent(memory("csrColumnOffsets","OneLevelTransport"),level=4)
                if hasDiffusionInMixedForm:
                    self.csrColumnOffsets_eNebN[(ci,cj)] = numpy.zeros((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,
                                                                        self.nDOF_test_element[ci],self.nDOF_trial_element[cj]),CSR_INDEX_DTYPE)
                else:
                    self.csrColumnOffsets_eNebN[(ci,cj)] = numpy.zeros((0,),CSR_INDEX_DTYPE)
                logEvent(memory("csrColumnOffsets_eNebN","OneLevelTransport"),level=4)
                self.csrColumnOffsets_eb[(ci,cj)] = numpy.zeros((self.mesh.nElementBoundaries_global,2,2,self.nDOF_test_element[ci],self.nDOF_trial_element[cj]),CSR_INDEX_DTYPE)
                logEvent(memory("csrColumnOffsets_eb","OneLevelTransport"),level=4)
                if hasDiffusionInMixedForm:
                    self.csrColumnOffsets_eb_eNebN[(ci,cj)] = numpy.zeros((self.mesh.nElementBoundaries_global,2,2,self.mesh.nElementBoundaries_element,
                                                                           self.nDOF_test_element[ci],self.nDOF_trial_element[cj]),CSR_INDEX_DTYPE)
                else:
                    self.csrColumnOffsets_eb_eNebN[(ci,cj)] = numpy.zeros((0,),CSR_INDEX_DTYPE)
                logEvent(memory("csrColumnOffsets_eb_eNebN","OneLevelTransport"),level=4)
                if useC:
                    self.sparsityInfo.getOffsets_CSR(self.mesh.nElements_global,
                                                     self.nDOF_test_element[ci],
                                                     self.nDOF_trial_element[cj],
                                                     self.l2g[ci]['nFreeDOF'],
                                                     self.l2g[ci]['freeGlobal'],
                                                     self.l2g[cj]['nFreeDOF'],
                                                     self.l2g[cj]['freeGlobal'],
                                                     self.offset[ci],
                                                     self.stride[ci],
                                                     self.offset[cj],
                                                     self.stride[cj],
                                                     hasNumericalFlux,
                                                     hasDiffusionInMixedForm,
                                                     needNumericalFluxJacobian_int,
                                                     self.mesh.nElementBoundaries_element,
                                                     self.mesh.elementNeighborsArray,
                                                     self.mesh.nInteriorElementBoundaries_global,
                                                     self.mesh.interiorElementBoundariesArray,
                                                     self.mesh.elementBoundaryElementsArray,
                                                     self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                     self.fluxBoundaryConditions[ci] == 'outFlow',
                                                     self.mesh.nExteriorElementBoundaries_global,
                                                     self.mesh.exteriorElementBoundariesArray,
                                                     hasOutflowBoundary,
                                                     needsOutflowJacobian_int,
                                                     self.rowptr,
                                                     self.csrRowIndeces[(ci,cj)],
                                                     self.csrColumnOffsets[(ci,cj)],
                                                     self.csrColumnOffsets_eNebN[(ci,cj)],
                                                     self.csrColumnOffsets_eb[(ci,cj)],
                                                     self.csrColumnOffsets_eb_eNebN[(ci,cj)])
                else:
                    for eN in range(self.mesh.nElements_global):
                        for ii in range(self.l2g[ci]['nFreeDOF'][eN]):
                            I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][eN,ii]
                            self.csrRowIndeces[(ci,cj)][eN,ii]=self.rowptr[I]
                            for jj in range(self.l2g[cj]['nFreeDOF'][eN]):
                                J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][eN,jj]
                                self.csrColumnOffsets[(ci,cj)][eN,ii,jj] = columnOffsetDict[(I,J)]
                        if (self.numericalFlux is not None and self.numericalFlux.mixedDiffusion[ci] == True):
                            for ebN in range(self.mesh.nElementBoundaries_element):
                                eN_ebN = self.mesh.elementNeighborsArray[eN,ebN]
                                if eN_ebN >= 0:
                                    for ii in range(self.l2g[ci]['nFreeDOF'][eN]):
                                        I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][eN,ii]
                                        for jj in range(self.l2g[cj]['nFreeDOF'][eN_ebN]):
                                            J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][eN_ebN,jj]
                                            self.csrColumnOffsets_eNebN[(ci,cj)][eN,ebN,ii,jj] = columnOffsetDict[(I,J)]
                    if self.numericalFlux is not None and needNumericalFluxJacobian == True:
                        for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                            ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                            left_eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
                            right_eN_global  = self.mesh.elementBoundaryElementsArray[ebN,1]
                            left_ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                            right_ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
                            for ii in range(self.l2g[ci]['nFreeDOF'][left_eN_global]):
                                left_I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][left_eN_global,ii]
                                for jj in range(self.l2g[cj]['nFreeDOF'][left_eN_global]):
                                    left_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][left_eN_global,jj]
                                    self.csrColumnOffsets_eb[(ci,cj)][ebN,0,0,ii,jj] = columnOffsetDict[(left_I,left_J)]
                                for jj in range(self.l2g[cj]['nFreeDOF'][right_eN_global]):
                                    right_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][right_eN_global,jj]
                                    self.csrColumnOffsets_eb[(ci,cj)][ebN,0,1,ii,jj] = columnOffsetDict[(left_I,right_J)]
                            for ii in range(self.l2g[ci]['nFreeDOF'][right_eN_global]):
                                right_I = self.offset[ci] + self.stride[ci]*self.l2g[ci]['freeGlobal'][right_eN_global,ii]
                                for jj in range(self.l2g[cj]['nFreeDOF'][left_eN_global]):
                                    left_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][left_eN_global,jj]
                                    self.csrColumnOffsets_eb[(ci,cj)][ebN,1,0,ii,jj] = columnOffsetDict[(right_I,left_J)]
                                for  jj  in range(self.l2g[cj]['nFreeDOF'][right_eN_global]):
                                    right_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][right_eN_global,jj]
                                    self.csrColumnOffsets_eb[(ci,cj)][ebN,1,1,ii,jj] = columnOffsetDict[(right_I,right_J)]
                            if self.numericalFlux.mixedDiffusion[ci] == True:
                                for ebN_eN in range(self.mesh.nElementBoundaries_element):
                                    left_eN_ebN = self.mesh.elementNeighborsArray[left_eN_global,ebN_eN]
                                    right_eN_ebN = self.mesh.elementNeighborsArray[right_eN_global,ebN_eN]
                                    for ii in range(self.l2g[ci]['nFreeDOF'][left_eN_global]):
                                        left_I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][left_eN_global,ii]
                                        if left_eN_ebN >= 0:
                                            for jj in range(self.l2g[cj]['nFreeDOF'][left_eN_ebN]):
                                                left_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][left_eN_ebN,jj]
                                                self.csrColumnOffsets_eb_eNebN[(ci,cj)][ebN,0,0,ebN_eN,ii,jj] = columnOffsetDict[(left_I,left_J)]
                                        if right_eN_ebN >= 0:
                                            for jj in range(self.l2g[cj]['nFreeDOF'][right_eN_ebN]):
                                                right_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][right_eN_ebN,jj]
                                                self.csrColumnOffsets_eb_eNebN[(ci,cj)][ebN,0,1,ebN_eN,ii,jj] = columnOffsetDict[(left_I,right_J)]
                                    for ii in range(self.l2g[ci]['nFreeDOF'][right_eN_global]):
                                        right_I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][right_eN_global,ii]
                                        if left_eN_ebN >= 0:
                                            for jj in range(self.l2g[cj]['nFreeDOF'][left_eN_ebN]):
                                                left_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][left_eN_ebN,jj]
                                                self.csrColumnOffsets_eb_eNebN[(ci,cj)][ebN,1,0,ebN_eN,ii,jj] = columnOffsetDict[(right_I,left_J)]
                                        if right_eN_ebN >= 0:
                                            for  jj  in range(self.l2g[cj]['nFreeDOF'][right_eN_ebN]):
                                                right_J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][right_eN_ebN,jj]
                                                self.csrColumnOffsets_eb_eNebN[(ci,cj)][ebN,1,1,ebN_eN,ii,jj] = columnOffsetDict[(right_I,right_J)]
                    if ((self.numericalFlux is not None and needNumericalFluxJacobian == True) or
                        (self.fluxBoundaryConditions[ci] == 'outFlow' and needOutflowJacobian  == True)):
                        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                            eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
                            ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                            for  ii  in range(self.l2g[ci]['nFreeDOF'][eN_global]):
                                I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][eN_global,ii]
                                for  jj  in range(self.l2g[cj]['nFreeDOF'][eN_global]):
                                    J = self.offset[cj] + self.stride[cj]*self.l2g[cj]['freeGlobal'][eN_global,jj]
                                    self.csrColumnOffsets_eb[(ci,cj)][ebN,0,0,ii,jj] = columnOffsetDict[(I,J)]
                            if self.numericalFlux is not None and self.numericalFlux.mixedDiffusion[ci] == True:
                                for ebN_eN in range(self.mesh.nElementBoundaries_element):
                                    eN_ebN = self.mesh.elementNeighborsArray[eN_global,ebN_eN]
                                    for ii in range(self.l2g[ci]['nFreeDOF'][eN_global]):
                                        I = self.offset[ci]+self.stride[ci]*self.l2g[ci]['freeGlobal'][eN_global,ii]
                                        if eN_ebN >= 0:
                                            for jj in range(self.l2g[cj]['nFreeDOF'][eN_ebN]):
                                                J = self.offset[cj]+self.stride[cj]*self.l2g[cj]['freeGlobal'][eN_ebN,jj]
                                                self.csrColumnOffsets_eb_eNebN[(ci,cj)][ebN,0,0,ebN_eN,ii,jj] = columnOffsetDict[(I,J)]
        self.nNonzerosInJacobian = self.nnz
        assert(self.nNonzerosInJacobian > 0)
        return self.jacobian
    def viewSolution(self,plotOffSet=None,titleModifier='',dgridnx=50,dgridny=50,dgridp=16.,pause=False):
        #tmp add pause arg for vtk
        #cek low priority clean up, could get gnuplot/matlab/vtk/asymptote? Maybe should seperate off
        from . import Viewers
        if Viewers.viewerType == 'vtk':
            return self.viewSolutionVTK(plotOffSet=plotOffSet,titleModifier=titleModifier,dgridnx=dgridnx,
                                        dgridny=dgridny,dgridp=dgridp,pause=pause)
        if plotOffSet is not None:
            windowNumberSave = Viewers.windowNumber
            Viewers.windowNumber=plotOffSet
        else:
            windowNumberSave = None
        if Viewers.viewerType == 'gnuplot':
            if self.nSpace_global == 1:
                for ci in range(self.coefficients.nc):
                    if (isinstance(self.u[ci].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) or
                        isinstance(self.u[ci].femSpace,NC_AffineLinearOnSimplexWithNodalBasis)): #CrR same in 1d
                        #
                        xandu = [(x,u) for x,u in zip(self.mesh.nodeArray[:,0],self.u[ci].dof)]
                        xandu.sort()#sorts based on first entry by default I believe
                        #for x,u in zip(self.mesh.nodeArray[:,0],self.u[ci].dof):
                        for xu in xandu:
                            Viewers.datFile.write("%12.5e %12.5e \n" % (xu[0],xu[1]))
                        Viewers.datFile.write("\n \n")
                        cmd = "set term x11 %i; plot \'%s\' index %i with linespoints title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                          Viewers.datFilename,
                                                                                                          Viewers.plotNumber,
                                                                                                          self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()
                    #
                    elif (isinstance(self.u[ci].femSpace,DG_AffineLinearOnSimplexWithNodalBasis) or
                          isinstance(self.u[ci].femSpace,DG_AffineQuadraticOnSimplexWithNodalBasis)) :
                        ldim = self.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim
                        #sort for now
                        xandu = []
                        for eN in range(self.mesh.nElements_global):
                            for nN in range(self.mesh.nNodes_element):
                                #for sorting have to move just a little bit into the element
                                x=self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][0]*(1.0 + 1.0e-8*(1-2.0*nN))
                                xandu.append((x,self.u[ci].dof[eN*ldim+nN]))
                        xandu.sort()
                        for xu in xandu:
                            Viewers.datFile.write("%12.5e %12.5e \n" % (xu[0],xu[1]))
                        Viewers.datFile.write("\n \n")
                        cmd = "set term x11 %i; plot \'%s\' index %i with linespoints title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                          Viewers.datFilename,
                                                                                                          Viewers.plotNumber,
                                                                                                          self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()
                    elif (isinstance(self.u[ci].femSpace,DG_AffinePolynomialsOnSimplexWithMonomialBasis)):
                        npoints  = self.q['x'].shape[0]*self.q['x'].shape[1]
                        xandu = [(self.q['x'].flat[i*3+0],self.q[('u',ci)].flat[i]) for i in range(npoints)]
                        xandu.sort()
                        for xu in xandu:
                            Viewers.datFile.write("%12.5e %12.5e \n" % (xu[0],xu[1]))
                        Viewers.datFile.write("\n \n")
                        cmd = "set term x11 %i; plot \'%s\' index %i with linespoints title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                          Viewers.datFilename,
                                                                                                          Viewers.plotNumber,
                                                                                                          self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()

                    elif  (isinstance(self.u[ci].femSpace,DG_Constants) or
                           isinstance(self.u[ci].femSpace,DG_AffineP0_OnSimplexWithMonomialBasis)):
                        #sort for now
                        xandu = []
                        for eN in range(self.mesh.nElements_global):
                            for nN in range(self.mesh.nNodes_element):
                                x=self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][0]*(1.0 + 1.0e-8*(1-2.0*nN))
                                xandu.append((x,self.u[ci].dof[eN]))
                        xandu.sort()
                        for xu in xandu:
                            Viewers.datFile.write("%12.5e %12.5e \n" % (xu[0],xu[1]))
                        Viewers.datFile.write("\n \n")
                        cmd = "set term x11 %i; plot \'%s\' index %i with linespoints title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                          Viewers.datFilename,
                                                                                                          Viewers.plotNumber,
                                                                                                          self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()
                    elif isinstance(self.u[ci].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis):
                        xandu = []
                        for eN in range(self.mesh.nElements_global):
                            for i in range(self.u[ci].femSpace.dofMap.l2g.shape[1]):
                                ig = self.u[ci].femSpace.dofMap.l2g[eN,i]
                                xandu.append((self.u[ci].femSpace.dofMap.lagrangeNodesArray[ig,0],self.u[ci].dof[ig]))
                        xandu.sort()
                        for xu in xandu:
                            Viewers.datFile.write("%12.5e %12.5e \n" % (xu[0],xu[1]))
                        Viewers.datFile.write("\n \n")
                        cmd = "set term x11 %i; plot \'%s\' index %i with linespoints title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                          Viewers.datFilename,
                                                                                                          Viewers.plotNumber,
                                                                                                          self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()

            elif self.nSpace_global == 2:
                for ci in range(self.coefficients.nc):
                    #
                    if (isinstance(self.u[ci].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) or
                        isinstance(self.u[ci].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis)):
                        #mwf todo 10/17/09 won't work in parallel for C0_AffineQuadratic
                        for x,y,u in zip(self.mesh.nodeArray[:,0],self.mesh.nodeArray[:,1],self.u[ci].dof[:self.mesh.nodeArray.shape[0]]):
                            Viewers.datFile.write("%12.5e %12.5e %12.5e \n" % (x,y,u))
                        Viewers.datFile.write("\n \n")
                        cmd = "set dgrid3d %d,%d,%g; set contour base; set term x11 %i; splot \'%s\' index %i with lines title \"%s\" \n" % (dgridnx,dgridny,dgridp,Viewers.windowNumber,
                                                                                                                                             Viewers.datFilename,
                                                                                                                                             Viewers.plotNumber,
                                                                                                                                             self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()

                    #
                    elif (isinstance(self.u[ci].femSpace,DG_AffineLinearOnSimplexWithNodalBasis) or
                          isinstance(self.u[ci].femSpace,DG_AffineQuadraticOnSimplexWithNodalBasis)):
                        for eN in range(self.mesh.nElements_global):
                            for nN in range(self.mesh.nNodes_element):
                                ldim = self.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim
                                Viewers.datFile.write("%12.5e %12.5e %12.5e \n" % (self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][0],
                                                                                       self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][1],
                                                                                       self.u[ci].dof[eN*ldim+nN]))
                        Viewers.datFile.write("\n \n")
                        cmd = "set dgrid3d %d,%d,%g; set contour base; set term x11 %i; splot \'%s\' index %i with lines title \"%s\" \n" % (dgridnx,dgridny,dgridp,Viewers.windowNumber,
                                                                                                                                    Viewers.datFilename,
                                                                                                                                    Viewers.plotNumber,
                                                                                                                                    self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()
                    elif  (isinstance(self.u[ci].femSpace,DG_Constants) or
                           isinstance(self.u[ci].femSpace,DG_AffineP0_OnSimplexWithMonomialBasis)):
                        for eN in range(self.mesh.nElements_global):
                            for nN in range(self.mesh.nNodes_element):
                                Viewers.datFile.write("%12.5e %12.5e %12.5e \n" % (self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][0],
                                                                                   self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][1],
                                                                                   self.u[ci].dof[eN]))
                        Viewers.datFile.write("\n \n")
                        cmd = "set dgrid3d %d,%d,%g; set contour base; set term x11 %i; splot \'%s\' index %i with lines title \"%s\" \n" % (dgridnx,dgridny,dgridp,Viewers.windowNumber,
                                                                                                                                    Viewers.datFilename,
                                                                                                                                    Viewers.plotNumber,
                                                                                                                                    self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()
                    elif isinstance(self.u[ci].femSpace,NC_AffineLinearOnSimplexWithNodalBasis):
                        #dofs are values at edge midpoints
                        for eN in range(self.mesh.nElements_global):
                            for nN in range(self.mesh.nNodes_element):
                                nipj = [int(fmod(nN+j,self.mesh.nNodes_element)) for j in range(1,self.mesh.nNodes_element)]
                                gNipj= [self.mesh.elementNodesArray[eN,nipj[j]]  for j in range(0,self.mesh.nNodes_element-1)]
                                xM   = 0.5*(self.mesh.nodeArray[gNipj[0],:]+self.mesh.nodeArray[gNipj[1],:])
                                ig   = self.u[ci].femSpace.dofMap.l2g[eN,nN]
                                Viewers.datFile.write("%12.5e %12.5e %12.5e \n" % (xM[0],xM[1],self.u[ci].dof[ig]))
                        Viewers.datFile.write("\n \n")
                        cmd = "set dgrid3d %d,%d,%g; set contour base; set term x11 %i; splot \'%s\' index %i with lines title \"%s\" \n" % (dgridnx,dgridny,dgridp,Viewers.windowNumber,
                                                                                                                                    Viewers.datFilename,
                                                                                                                                    Viewers.plotNumber,
                                                                                                                                    self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()

                if self.coefficients.vectorComponents is not None:
                    scale_x = max(numpy.absolute(self.u[self.coefficients.vectorComponents[0]].dof.flat))
                    scale_y = max(numpy.absolute(self.u[self.coefficients.vectorComponents[1]].dof.flat))
                    L = min((max(self.mesh.nodeArray[:,0]),max(self.mesh.nodeArray[:,1])))
                    scale=10.0*max((scale_x,scale_y,1.0e-16))/L
                    for x,y,u,v in zip(self.mesh.nodeArray[:,0],self.mesh.nodeArray[:,1],self.u[self.coefficients.vectorComponents[0]].dof,self.u[self.coefficients.vectorComponents[1]].dof):
                        Viewers.datFile.write("%12.5e %12.5e %12.5e %12.5e \n" % (x,y,u/scale,v/scale))
                    Viewers.datFile.write("\n \n")
                    cmd = "set term x11 %i; plot \'%s\' index %i with vectors title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                  Viewers.datFilename,
                                                                                                  Viewers.plotNumber,
                                                                                                  self.coefficients.vectorName+titleModifier)
                    Viewers.cmdFile.write(cmd)
                    Viewers.viewerPipe.write(cmd)
                    Viewers.newPlot()
                    Viewers.newWindow()
            elif self.nSpace_global == 3:
                (slice_x,slice_y,slice_z) = self.mesh.nodeArray[self.mesh.nodeArray.shape[0]//2,:]
                for ci in range(self.coefficients.nc):
                    if (isinstance(self.u[ci].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) or
                        isinstance(self.u[ci].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis)):
                        #mwf todo 10/17/09 won't work in parallel for C0_AffineQuadratic
                        for x,y,z,u in zip(self.mesh.nodeArray[:,0],self.mesh.nodeArray[:,1],self.mesh.nodeArray[:,2],
                                           self.u[ci].dof[:self.mesh.nodeArray.shape[0]]):
                            if x == slice_x:
                                Viewers.datFile.write("%12.5e %12.5e %12.5e \n" % (y,z,u))
                        Viewers.datFile.write("\n \n")
                        cmd = "set dgrid3d; set contour base; set term x11 %i; splot \'%s\' index %i with lines title \"%s-x-slice\" \n" % (Viewers.windowNumber,
                                                                                                                                            Viewers.datFilename,
                                                                                                                                            Viewers.plotNumber,
                                                                                                                                            self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()
                        for x,y,z,u in zip(self.mesh.nodeArray[:,0],self.mesh.nodeArray[:,1],self.mesh.nodeArray[:,2],
                                           self.u[ci].dof[:self.mesh.nodeArray.shape[0]]):
                            if y == slice_y:
                                Viewers.datFile.write("%12.5e %12.5e %12.5e \n" % (x,z,u))
                        Viewers.datFile.write("\n \n")
                        cmd = "set dgrid3d; set contour base; set term x11 %i; splot \'%s\' index %i with lines title \"%s-y-slice\" \n" % (Viewers.windowNumber,
                                                                                                                                                    Viewers.datFilename,
                                                                                                                                                    Viewers.plotNumber,
                                                                                                                                                    self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()
                        for x,y,z,u in zip(self.mesh.nodeArray[:,0],self.mesh.nodeArray[:,1],self.mesh.nodeArray[:,2],
                                           self.u[ci].dof[:self.mesh.nodeArray.shape[0]]):
                            if z == slice_z:
                                Viewers.datFile.write("%12.5e %12.5e %12.5e \n" % (x,y,u))
                        Viewers.datFile.write("\n \n")
                        cmd = "set dgrid3d; set contour base; set term x11 %i; splot \'%s\' index %i with lines title \"%s-z-slice\" \n" % (Viewers.windowNumber,
                                                                                                                                            Viewers.datFilename,
                                                                                                                                            Viewers.plotNumber,
                                                                                                                                            self.coefficients.variableNames[ci]+titleModifier)
                        Viewers.cmdFile.write(cmd)
                        Viewers.viewerPipe.write(cmd)
                        Viewers.newPlot()
                        Viewers.newWindow()

                if (self.coefficients.vectorComponents is not None and
                    (isinstance(self.u[0].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) and
                     isinstance(self.u[1].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) and
                     isinstance(self.u[2].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) or
                     isinstance(self.u[0].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis) and
                     isinstance(self.u[1].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis) and
                     isinstance(self.u[2].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis))):
                    #mwf todo 10/17/09 won't work in parallel for C0_AffineQuadratic
                    for x,y,z,u,v,w in zip(self.mesh.nodeArray[:,0],
                                           self.mesh.nodeArray[:,1],
                                           self.mesh.nodeArray[:,2],
                                           self.u[self.coefficients.vectorComponents[0]].dof[:self.mesh.nodeArray.shape[0]],
                                           self.u[self.coefficients.vectorComponents[1]].dof[:self.mesh.nodeArray.shape[0]],
                                           self.u[self.coefficients.vectorComponents[2]].dof[:self.mesh.nodeArray.shape[0]]):
                        if x == slice_x:
                            Viewers.datFile.write("%12.5e %12.5e %12.5e %12.5e \n" % (y,z,v,w))
                    Viewers.datFile.write("\n \n")
                    cmd = "set term x11 %i; plot \'%s\' index %i with vectors title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                  Viewers.datFilename,
                                                                                                  Viewers.plotNumber,
                                                                                                  self.coefficients.vectorName+": x-slice"+titleModifier)
                    Viewers.cmdFile.write(cmd)
                    Viewers.viewerPipe.write(cmd)
                    Viewers.newPlot()
                    Viewers.newWindow()
                    for x,y,z,u,v,w in zip(self.mesh.nodeArray[:,0],
                                           self.mesh.nodeArray[:,1],
                                           self.mesh.nodeArray[:,2],
                                           self.u[self.coefficients.vectorComponents[0]].dof[:self.mesh.nodeArray.shape[0]],
                                           self.u[self.coefficients.vectorComponents[1]].dof[:self.mesh.nodeArray.shape[0]],
                                           self.u[self.coefficients.vectorComponents[2]].dof[:self.mesh.nodeArray.shape[0]]):
                        if y == slice_y:
                            Viewers.datFile.write("%12.5e %12.5e %12.5e %12.5e \n" % (x,z,u,w))
                    Viewers.datFile.write("\n \n")
                    cmd = "set term x11 %i; plot \'%s\' index %i with vectors title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                  Viewers.datFilename,
                                                                                                  Viewers.plotNumber,
                                                                                                  self.coefficients.vectorName+": y-slice,"+titleModifier)
                    Viewers.cmdFile.write(cmd)
                    Viewers.viewerPipe.write(cmd)
                    Viewers.newPlot()
                    Viewers.newWindow()
                    for x,y,z,u,v,w in zip(self.mesh.nodeArray[:,0],
                                           self.mesh.nodeArray[:,1],
                                           self.mesh.nodeArray[:,2],
                                           self.u[self.coefficients.vectorComponents[0]].dof[:self.mesh.nodeArray.shape[0]],
                                           self.u[self.coefficients.vectorComponents[1]].dof[:self.mesh.nodeArray.shape[0]],
                                           self.u[self.coefficients.vectorComponents[2]].dof[:self.mesh.nodeArray.shape[0]]):
                        if z == slice_z:
                            Viewers.datFile.write("%12.5e %12.5e %12.5e %12.5e \n" % (x,y,u,v))
                    Viewers.datFile.write("\n \n")
                    cmd = "set term x11 %i; plot \'%s\' index %i with vectors title \"%s\" \n" % (Viewers.windowNumber,
                                                                                                  Viewers.datFilename,
                                                                                                  Viewers.plotNumber,
                                                                                                  self.coefficients.vectorName+": z-slice,"+titleModifier)
                    Viewers.cmdFile.write(cmd)
                    Viewers.viewerPipe.write(cmd)
                    Viewers.newPlot()
                    Viewers.newWindow()
        if Viewers.viewerType == 'matlab':
            for ci in range(self.coefficients.nc):
                nplotted = self.u[ci].writeFunctionMatlab(Viewers.cmdFile,append=False,
                                                          storeMeshData= not Viewers.meshDataStructuresWritten,
                                                          figureOffset=Viewers.windowNumber+1)
                Viewers.windowNumber += nplotted
            #
            if self.nSpace_global == 2:
                if self.coefficients.vectorComponents is not None:
                    vci0 = self.coefficients.vectorComponents[0]
                    vci1 = self.coefficients.vectorComponents[1]

                    if (isinstance(self.u[vci0].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) and
                        isinstance(self.u[vci1].femSpace,C0_AffineLinearOnSimplexWithNodalBasis)):

                        writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
                        nplotted = writer.viewVector_LagrangeC0P1(Viewers.cmdFile,self.nSpace_global,
                                                                  self.mesh.nodeArray,self.mesh.elementNodesArray,
                                                                  self.u[vci0].dof,self.u[vci1].dof,name="velocity_dof",
                                                                  storeMeshData= not Viewers.meshDataStructuresWritten,
                                                                  figureNumber = Viewers.windowNumber+1,
                                                                  title = 'velocity_{dof}' + titleModifier)
                        Viewers.windowNumber += nplotted
                    elif (isinstance(self.u[vci0].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis) and
                          isinstance(self.u[vci1].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis)):

                        writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
                        nplotted = writer.viewVector_LagrangeC0P2(Viewers.cmdFile,self.nSpace_global,
                                                                  self.u[vci0].femSpace.dofMap.lagrangeNodesArray,self.mesh.elementNodesArray,
                                                                  self.u[vci0].femSpace.dofMap.l2g,
                                                                  self.u[vci0].dof,self.u[vci1].dof,
                                                                  name="velocity_dof",
                                                                  storeMeshData= not Viewers.meshDataStructuresWritten,
                                                                  figureNumber = Viewers.windowNumber+1,
                                                                  title = 'velocity_{dof}' + titleModifier)
                        Viewers.windowNumber += nplotted
            elif self.nSpace_global == 3:
                if self.coefficients.vectorComponents is not None:
                    vci0 = self.coefficients.vectorComponents[0]
                    vci1 = self.coefficients.vectorComponents[1]
                    vci2 = self.coefficients.vectorComponents[2]

                    if (isinstance(self.u[vci0].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) and
                        isinstance(self.u[vci1].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) and
                        isinstance(self.u[vci2].femSpace,C0_AffineLinearOnSimplexWithNodalBasis)):

                        writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
                        nplotted = writer.viewVector_LagrangeC0P1(Viewers.cmdFile,self.nSpace_global,
                                                                  self.mesh.nodeArray,self.mesh.elementNodesArray,
                                                                  self.u[vci0].dof,self.u[vci1].dof,self.u[vci2].dof,
                                                                  name="velocity_dof",
                                                                  storeMeshData= not Viewers.meshDataStructuresWritten,
                                                                  figureNumber = Viewers.windowNumber+1,
                                                                  title = 'velocity_{dof}' + titleModifier)
                        Viewers.windowNumber += nplotted
                    elif (isinstance(self.u[vci0].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis) and
                          isinstance(self.u[vci1].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis) and
                          isinstance(self.u[vci2].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis)):
                        writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
                        nplotted = writer.viewVector_LagrangeC0P2(Viewers.cmdFile,self.nSpace_global,
                                                                  self.u[vci0].femSpace.dofMap.lagrangeNodesArray,self.mesh.elementNodesArray,
                                                                  self.u[vci0].femSpace.dofMap.l2g,
                                                                  self.u[vci0].dof,self.u[vci1].dof,self.u[vci2].dof,
                                                                  name="velocity_dof",
                                                                  storeMeshData= not Viewers.meshDataStructuresWritten,
                                                                  figureNumber = Viewers.windowNumber+1,
                                                                  title = 'velocity_{dof}' + titleModifier)
                        Viewers.windowNumber += nplotted

        #if matlab
        if windowNumberSave is not None:
            Viewers.windowNumber = windowNumberSave
        return Viewers.windowNumber

    #this is viewSolutionVTK
    def viewSolutionVTK(self,plotOffSet=None,titleModifier='',dgridnx=50,dgridny=50,dgridp=16.,
                        pause=False):
        from . import Viewers
        from proteusGraphical import vtkViewers
        if plotOffSet is not None:
            windowNumberSave = Viewers.windowNumber
            Viewers.windowNumber=plotOffSet
        else:
            windowNumberSave = None
        if self.nSpace_global == 1:
            for ci in range(self.coefficients.nc):
                title = self.coefficients.variableNames[ci]+titleModifier
                if isinstance(self.u[ci].femSpace,C0_AffineLinearOnSimplexWithNodalBasis):
                    vtkViewers.viewScalar_1D(self.mesh.nodeArray[:,0],self.u[ci].dof,"x",self.u[ci].name,title,Viewers.windowNumber,
                                                Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,NC_AffineLinearOnSimplexWithNodalBasis):
                    vtkViewers.viewScalar_1D(self.mesh.elementBoundaryBarycentersArray[:,0],self.u[ci].dof,"x",self.u[ci].name,title,Viewers.windowNumber,
                                                Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()

                elif isinstance(self.u[ci].femSpace,DG_AffineLinearOnSimplexWithNodalBasis):
                    ldim = self.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim
                    xvals = []; yvals = []
                    for eN in range(self.mesh.nElements_global):
                        for nN in range(self.mesh.nNodes_element):
                            #for sorting have to move just a little bit into the element
                            x=self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][0]*(1.0 + 1.0e-8*(1-2.0*nN))
                            xvals.append(x); yvals.append(self.u[ci].dof[eN*ldim+nN])
                    #eN
                    vtkViewers.viewScalar_1D(numpy.array(xvals),numpy.array(yvals),"x",self.u[ci].name,title,Viewers.windowNumber,
                                             Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,DG_AffineQuadraticOnSimplexWithNodalBasis):
                    ldim = self.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim
                    xvals = []; yvals = []
                    for eN in range(self.mesh.nElements_global):
                        for nN in range(self.mesh.nNodes_element):
                            #for sorting have to move just a little bit into the element
                            x=self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][0]*(1.0 + 1.0e-8*(1-2.0*nN))
                            xvals.append(x); yvals.append(self.u[ci].dof[eN*ldim+nN])
                        nN = self.mesh.nNodes_element
                        x  = 0.5*(self.mesh.nodeArray[self.mesh.elementNodesArray[eN,0]][0]+
                                  self.mesh.nodeArray[self.mesh.elementNodesArray[eN,1]][0])
                        xvals.append(x); yvals.append(self.u[ci].dof[eN*ldim+nN])
                    #eN
                    vtkViewers.viewScalar_1D(numpy.array(xvals),numpy.array(yvals),"x",self.u[ci].name,title,Viewers.windowNumber,
                                             Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()

                elif (isinstance(self.u[ci].femSpace,DG_Constants) or
                      isinstance(self.u[ci].femSpace,DG_AffineP0_OnSimplexWithMonomialBasis)):
                    xvals = []; yvals = []
                    for eN in range(self.mesh.nElements_global):
                        for nN in range(self.mesh.nNodes_element):
                            x=self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN]][0]*(1.0 + 1.0e-8*(1-2.0*nN))
                            xvals.append(x); yvals.append(self.u[ci].dof[eN])
                    vtkViewers.viewScalar_1D(numpy.array(xvals),numpy.array(yvals),"x",self.u[ci].name,title,Viewers.windowNumber,
                                             Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif (isinstance(self.u[ci].femSpace,DG_AffinePolynomialsOnSimplexWithMonomialBasis)):
                    npoints  = self.q['x'].shape[0]*self.q['x'].shape[1]
                    xvals = [self.q['x'].flat[i*3+0] for i in range(npoints)]
                    yvals = [self.q[('u',ci)].flat[i] for i in range(npoints)]
                    vtkViewers.viewScalar_1D(numpy.array(xvals),numpy.array(yvals),"x",self.u[ci].name,title,Viewers.windowNumber,
                                             Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis):
                    #mwf switch order for parallel now
#                     vtkViewers.viewScalar_1D(numpy.append(self.mesh.nodeArray,
#                                                           self.u[ci].femSpace.dofMap.lagrangeNodesArray)[::3],
#                                              self.u[ci].dof,
#                                              "x",
#                                              self.u[ci].name,
#                                              title,
#                                              Viewers.windowNumber,
#                                              Pause=pause,sortPoints=True)
                    vtkViewers.viewScalar_1D(self.u[ci].femSpace.dofMap.lagrangeNodesArray[:,0],
                                             self.u[ci].dof,
                                             "x",
                                             self.u[ci].name,
                                             title,
                                             Viewers.windowNumber,
                                             Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif (isinstance(self.u[ci].femSpace,C0_AffineP1P0BubbleOnSimplexWithNodalBasis)):
                    npoints  = self.q['x'].shape[0]*self.q['x'].shape[1]
                    xvals = [self.q['x'].flat[i*3+0] for i in range(npoints)]
                    yvals = [self.q[('u',ci)].flat[i] for i in range(npoints)]
                    vtkViewers.viewScalar_1D(numpy.array(xvals),numpy.array(yvals),"x",self.u[ci].name,title,Viewers.windowNumber,
                                             Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif (isinstance(self.u[ci].femSpace,C0_AffineP1BubbleOnSimplexWithNodalBasis)):
                    vtkViewers.viewScalar_1D(self.mesh.nodeArray[:,0],self.u[ci].dof,"x",self.u[ci].name,title,Viewers.windowNumber,
                                             Pause=pause,sortPoints=True)
                    #full solution at quadrature points
                    #npoints  = self.q['x'].shape[0]*self.q['x'].shape[1]
                    #xvals = [self.q['x'].flat[i*3+0] for i in range(npoints)]
                    #yvals = [self.q[('u',ci)].flat[i] for i in range(npoints)]
                    #vtkViewers.viewScalar_1D(numpy.array(xvals),numpy.array(yvals),"x",self.u[ci].name,title,Viewers.windowNumber,
                    #                         Pause=pause,sortPoints=True)
                    Viewers.newPlot()
                    Viewers.newWindow()
        elif self.nSpace_global == 2:
            for ci in range(self.coefficients.nc):
                title = self.coefficients.variableNames[ci]+titleModifier
                if (isinstance(self.u[ci].femSpace,C0_AffineLinearOnSimplexWithNodalBasis)):
                    vtkViewers.viewScalar_tri3_2D(self.mesh,
                                                  self.u[ci].dof[:self.mesh.nodeArray.shape[0]],
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber,
                                                  viewTypes=['colorMapped'],#,'contour','warp'],
                                                  IsoSurface=True,
                                                  Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis):
                    vtkViewers.viewScalar_tri6_2D(self.mesh,
                                                  self.u[ci].femSpace.dofMap,
                                                  self.u[ci].dof,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber,
                                                  viewTypes=['colorMapped'],#,'contour','warp'],
                                                  IsoSurface=True,
                                                  Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,DG_AffineLinearOnSimplexWithNodalBasis):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tri3_2D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber,
                                                  IsoSurface=True,
                                                  Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()

                elif isinstance(self.u[ci].femSpace,DG_AffineQuadraticOnSimplexWithNodalBasis):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tri3_2D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber,
                                                  IsoSurface=True,
                                                  Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif  (isinstance(self.u[ci].femSpace,DG_Constants) or
                       isinstance(self.u[ci].femSpace,DG_AffineP0_OnSimplexWithMonomialBasis)):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tri3_2D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber,
                                                  IsoSurface=True,
                                                  Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,NC_AffineLinearOnSimplexWithNodalBasis):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tri3_2D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber,
                                                  IsoSurface=True,
                                                  Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()

                elif (isinstance(self.u[ci].femSpace,DG_AffinePolynomialsOnSimplexWithMonomialBasis)):
                    vtkViewers.viewScalar_pointSet_2D(self.q['x'],
                                                      self.q[('u',ci)],
                                                      title,
                                                      Viewers.windowNumber,
                                                      IsoSurface=True,Pause=pause)

                    Viewers.newPlot()
                    Viewers.newWindow()
                elif (isinstance(self.u[ci].femSpace,C0_AffineP1P0BubbleOnSimplexWithNodalBasis)):
                    vtkViewers.viewScalar_pointSet_2D(self.q['x'],
                                                      self.q[('u',ci)],
                                                      title,
                                                      Viewers.windowNumber,
                                                      IsoSurface=True,Pause=pause)

                    Viewers.newPlot()
                    Viewers.newWindow()
                elif (isinstance(self.u[ci].femSpace,C0_AffineP1BubbleOnSimplexWithNodalBasis)):
                    vtkViewers.viewScalar_tri3_2D(self.mesh,
                                                  self.u[ci].dof[:self.mesh.nodeArray.shape[0]],
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber,
                                                  viewTypes=['colorMapped'],#,'contour','warp'],
                                                  IsoSurface=True,
                                                  Pause=pause)
                    #full solution at quadrature points
                    #vtkViewers.viewScalar_pointSet_2D(self.q['x'],
                    #                                  self.q[('u',ci)],
                    #                                  title,
                    #                                  Viewers.windowNumber,
                    #                                  IsoSurface=True,Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()

                #if has vel
            if self.coefficients.vectorComponents is not None:
                #
                scale_x = max(numpy.absolute(self.u[self.coefficients.vectorComponents[0]].dof.flat))
                scale_y = max(numpy.absolute(self.u[self.coefficients.vectorComponents[1]].dof.flat))
                L = min((max(self.mesh.nodeArray[:,0]),max(self.mesh.nodeArray[:,1])))
                scale=10.0*max((scale_x,scale_y,1.0e-16))/L
                #assume all components the same FemSpace for now
                if isinstance(self.u[self.coefficients.vectorComponents[0]].femSpace,C0_AffineLinearOnSimplexWithNodalBasis):
                    vtkViewers.viewVector_tri3_2D(self.mesh,
                                                  self.u[self.coefficients.vectorComponents[0]].dof,
                                                  self.u[self.coefficients.vectorComponents[1]].dof,
                                                  self.coefficients.vectorName+titleModifier)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[self.coefficients.vectorComponents[0]].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis):
                    vtkViewers.viewVector_tri6_2D(self.mesh,
                                                  self.u[ci].femSpace.dofMap,
                                                  self.u[self.coefficients.vectorComponents[0]].dof,
                                                  self.u[self.coefficients.vectorComponents[1]].dof,
                                                  self.coefficients.vectorName+titleModifier)
                    Viewers.newPlot()
                    Viewers.newWindow()
        elif self.nSpace_global == 3:
            (slice_x,slice_y,slice_z) = self.mesh.nodeArray[self.mesh.nodeArray.shape[0]//2,:]
            for ci in range(self.coefficients.nc):
                title = self.coefficients.variableNames[ci]+titleModifier
                if isinstance(self.u[ci].femSpace,C0_AffineLinearOnSimplexWithNodalBasis):
                    vtkViewers.viewScalar_tet4_3D(self.mesh,
                                                  self.u[ci].dof,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber, Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis):
                    vtkViewers.viewScalar_tet10_3D(self.mesh,
                                                   self.u[ci].femSpace.dofMap,
                                                   self.u[ci].dof,
                                                   self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber, Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,DG_AffineLinearOnSimplexWithNodalBasis):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tet4_3D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber, Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,DG_AffineQuadraticOnSimplexWithNodalBasis):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tet4_3D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber, Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()

                elif  isinstance(self.u[ci].femSpace,DG_Constants):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tet4_3D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber, Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif isinstance(self.u[ci].femSpace,NC_AffineLinearOnSimplexWithNodalBasis):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tet4_3D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber, Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif (isinstance(self.u[ci].femSpace,DG_AffinePolynomialsOnSimplexWithMonomialBasis)):
                    vtkViewers.vtkDisplay3DScalarMeshGeneric(self.q['x'],self.q[('u',ci)].flat,title,Viewers.windowNumber,Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif (isinstance(self.u[ci].femSpace,C0_AffineP1P0BubbleOnSimplexWithNodalBasis)):
                    vtkViewers.vtkDisplay3DScalarMeshGeneric(self.q['x'],self.q[('u',ci)].flat,title,Viewers.windowNumber,Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()
                elif (isinstance(self.u[ci].femSpace,C0_AffineP1BubbleOnSimplexWithNodalBasis)):
                    self.u[ci].calculateValuesAtMeshNodes()
                    vtkViewers.viewScalar_tet4_3D(self.mesh,
                                                  self.u[ci].meshNodeValues,
                                                  self.coefficients.variableNames[ci]+titleModifier,
                                                  Viewers.windowNumber, Pause=pause)
                    Viewers.newPlot()
                    Viewers.newWindow()

            if self.coefficients.vectorComponents is not None:
                #
                scale_x = max(numpy.absolute(self.u[self.coefficients.vectorComponents[0]].dof.flat))
                scale_y = max(numpy.absolute(self.u[self.coefficients.vectorComponents[1]].dof.flat))
                scale_z = max(numpy.absolute(self.u[self.coefficients.vectorComponents[2]].dof.flat))
                L = min((max(self.mesh.nodeArray[:,0]),max(self.mesh.nodeArray[:,1]),max(self.mesh.nodeArray[:,2])))
                scale=10.0*max((scale_x,scale_y,scale_z,1.0e-16))/L
                #assume all components the same FemSpace for now
                if (isinstance(self.u[self.coefficients.vectorComponents[0]].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) and
                    isinstance(self.u[self.coefficients.vectorComponents[1]].femSpace,C0_AffineLinearOnSimplexWithNodalBasis) and
                    isinstance(self.u[self.coefficients.vectorComponents[2]].femSpace,C0_AffineLinearOnSimplexWithNodalBasis)) :
                    vtkViewers.viewVector_tet4_3D(self.mesh,
                                                  self.u[self.coefficients.vectorComponents[0]].dof,
                                                  self.u[self.coefficients.vectorComponents[1]].dof,
                                                  self.u[self.coefficients.vectorComponents[2]].dof,
                                                  self.coefficients.vectorName+titleModifier)
                    Viewers.newPlot()
                    Viewers.newWindow()

                elif (isinstance(self.u[self.coefficients.vectorComponents[0]].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis) and
                      isinstance(self.u[self.coefficients.vectorComponents[1]].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis) and
                      isinstance(self.u[self.coefficients.vectorComponents[2]].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis)):
                    vtkViewers.viewVector_tet10_3D(self.mesh,
                                                   self.u[ci].femSpace.dofMap,
                                                   self.u[self.coefficients.vectorComponents[0]].dof,
                                                   self.u[self.coefficients.vectorComponents[1]].dof,
                                                   self.u[self.coefficients.vectorComponents[2]].dof,
                                                   self.coefficients.vectorName+titleModifier)
                    Viewers.newPlot()
                    Viewers.newWindow()
                #
            #vector components
        if windowNumberSave is not None:
            Viewers.windowNumber = windowNumberSave
        return Viewers.windowNumber
    #### start routines for saving/writing data
    def saveSolution(self):
        pass
    def write_ebq_geo_Ensight(self,filename):
        caseOut=open(filename+'.case','a')
        caseOut.write('measured: '+filename+'ebq.geo\n')
        caseOut.close()
        meshOut=open(filename+'ebq.geo','w')
        meshOut.write('Element quadrature\n')
        meshOut.write('particle coordinates\n')
        meshOut.write('%8i\n' % (self.mesh.nElements_global*self.nQuadraturePoints_element,))
        pN=1
        for eN in range(self.mesh.nElements_global):
            for k in range(self.nQuadraturePoints_element):
                meshOut.write('%8i%12.5E%12.5E%12.5E\n' % (pN,
                                                            self.q['x'][eN,k,0],
                                                            self.q['x'][eN,k,1],
                                                            self.q['x'][eN,k,2]))
                pN+=1
        meshOut.close()
    def write_ebq_velocity_Ensight(self,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per measured node: '+
                              'velocity '+filename+'velocity'+'.vec****\n')
        uOut=open(filename+'velocity'+'.vec%4.4i' % nOutput,'w')
        uOut.write('velocity\n')
        n=0
        vmax =max(self.q[('velocity',0)].flat)+1.0e-8
        for eN in range(self.mesh.nElements_global):
            for k in range(self.nQuadraturePoints_element):
                uOut.write('%12.5e%12.5e%12.5e' % (self.q[('velocity',0)][eN,k,0]/vmax,
                                                   self.q[('velocity',0)][eN,k,1]/vmax,
                                                   0.0))
                if n%2==1:
                    uOut.write('\n')
                n+=1
        if n%2==1:
            uOut.write('\n')
        uOut.close()
    def archiveFiniteElementSolutions(self,archive,t,tCount,initialPhase=False,writeVectors=True,meshChanged=False,femSpaceWritten={},writeVelocityPostProcessor=True):
        """
        write finite element solutions to archive at time t, tries to group finite element
        functions by their space

        """
        for ci in range(self.coefficients.nc):
            femSpaceType_ci = self.u[ci].femSpace.__class__
            alreadyWritten = femSpaceType_ci in femSpaceWritten
            if initialPhase:
                if alreadyWritten:
                    femSpaceWritten[femSpaceType_ci]= self.u[ci].femSpace.writeMeshXdmf(archive,self.u[ci].name,
                                                                                        t,init=False,
                                                                                        meshChanged=meshChanged,
                                                                                        arGrid=femSpaceWritten[femSpaceType_ci],tCount=tCount)
                else:
                    femSpaceWritten[femSpaceType_ci]= self.u[ci].femSpace.writeMeshXdmf(archive,self.u[ci].name,
                                                                                        t,init=True,meshChanged=meshChanged,tCount=tCount)
            else:
                if alreadyWritten:
                    femSpaceWritten[femSpaceType_ci]= self.u[ci].femSpace.writeMeshXdmf(archive,self.u[ci].name,
                                                                                        t,init=False,
                                                                                        meshChanged=meshChanged,
                                                                                        arGrid=femSpaceWritten[femSpaceType_ci],tCount=tCount)
                else:
                    femSpaceWritten[femSpaceType_ci]= self.u[ci].femSpace.writeMeshXdmf(archive,self.u[ci].name,
                                                                                        t,init=False,meshChanged=meshChanged,tCount=tCount)

            self.u[ci].femSpace.writeFunctionXdmf(archive,self.u[ci],tCount)
        if writeVectors and self.coefficients.vectorComponents is not None:
            c0 = self.coefficients.vectorComponents[0]
            self.u[c0].femSpace.writeVectorFunctionXdmf(archive,
                                                        self.u,
                                                        self.coefficients.vectorComponents,
                                                        self.coefficients.vectorName,
                                                        tCount)

        if writeVelocityPostProcessor and self.velocityPostProcessor is not None:
            self.velocityPostProcessor.archiveVelocityValues(archive,t,tCount,initialPhase=initialPhase,meshChanged=meshChanged)
    def archiveElementQuadratureValues(self,archive,t,tCount,scalarKeys=None,vectorKeys=None,
                                       tensorKeys=None,
                                       initialPhase=False,meshChanged=False):
        """
        write element quadrature dictionary values to archive at time t, groups all entries
        under same mesh at a given time level

        """
        self.elementQuadratureDictionaryWriter.writeMeshXdmf_elementQuadrature(archive,
                                                                               self.mesh,
                                                                               self.nSpace_global,
                                                                               self.q['x'],
                                                                               t=t,init=initialPhase,
                                                                               meshChanged=meshChanged,arGrid=None,
                                                                               tCount=tCount)
        if scalarKeys is not None:
            for key in scalarKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.q, "key = %s not found" % str(key)
                self.elementQuadratureDictionaryWriter.writeScalarXdmf_elementQuadrature(archive,
                                                                                         self.q[key],
                                                                                         name,
                                                                                         tCount)
        if vectorKeys is not None:
            for key in vectorKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.q, "key = %s not found" % str(key)
                self.elementQuadratureDictionaryWriter.writeVectorXdmf_elementQuadrature(archive,
                                                                                         self.q[key],
                                                                                         name,
                                                                                         tCount)
        if tensorKeys is not None:
            for key in tensorKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.q, "key = %s not found" % str(key)
                self.elementQuadratureDictionaryWriter.writeTensorXdmf_elementQuadrature(archive,
                                                                                         self.q[key],
                                                                                         name,
                                                                                         tCount)
    def archiveElementBoundaryQuadratureValues(self,archive,t,tCount,scalarKeys=None,vectorKeys=None,
                                               tensorKeys=None,
                                               initialPhase=False,meshChanged=False):
        """
        write elementBoundary quadrature dictionary values to archive at time t, groups all entries
        under same mesh at a given time level

        """
        if 'x' not in self.ebq_global:
            return
        self.elementBoundaryQuadratureDictionaryWriter.writeMeshXdmf_elementBoundaryQuadrature(archive,
                                                                                               self.mesh,
                                                                                               self.nSpace_global,
                                                                                               self.ebq_global['x'],
                                                                                               t=t,init=initialPhase,
                                                                                               meshChanged=meshChanged,arGrid=None,
                                                                                               tCount=tCount)

        if scalarKeys is not None:
            for key in scalarKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.ebq_global, "key = %s not found" % str(key)
                self.elementBoundaryQuadratureDictionaryWriter.writeScalarXdmf_elementBoundaryQuadrature(archive,
                                                                                                         self.ebq_global[key],
                                                                                                         name,
                                                                                                         tCount)
        if vectorKeys is not None:
            for key in vectorKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.ebq_global, "key = %s not found" % str(key)
                self.elementBoundaryQuadratureDictionaryWriter.writeVectorXdmf_elementBoundaryQuadrature(archive,
                                                                                                         self.ebq_global[key],
                                                                                                         name,
                                                                                                         tCount)
        if tensorKeys is not None:
            for key in tensorKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.ebq_global, "key = %s not found" % str(key)
                self.elementBoundaryQuadratureDictionaryWriter.writeTensorXdmf_elementBoundaryQuadrature(archive,
                                                                                                         self.ebq_global[key],
                                                                                                         name,
                                                                                                         tCount)
    def archiveExteriorElementBoundaryQuadratureValues(self,archive,t,tCount,scalarKeys=None,vectorKeys=None,
                                                       tensorKeys=None,
                                                       initialPhase=False,meshChanged=False):
        """
        write exteriorElementBoundary quadrature dictionary values to archive at time t, groups all entries
        under same mesh at a given time level

        """
        if 'x' not in self.ebqe:
            return
        self.exteriorElementBoundaryQuadratureDictionaryWriter.writeMeshXdmf_exteriorElementBoundaryQuadrature(archive,
                                                                                                               self.mesh,
                                                                                                               self.nSpace_global,
                                                                                                               self.ebqe['x'],
                                                                                                               t=t,init=initialPhase,
                                                                                                               meshChanged=meshChanged,arGrid=None,
                                                                                                               tCount=tCount)
        if scalarKeys is not None:
            for key in scalarKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.ebqe, "key = %s not found" % str(key)
                self.exteriorElementBoundaryQuadratureDictionaryWriter.writeScalarXdmf_exteriorElementBoundaryQuadrature(archive,
                                                                                                                         self.ebqe[key],
                                                                                                                         name,
                                                                                                                         tCount)
        if vectorKeys is not None:
            for key in vectorKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.ebqe, "key = %s not found" % str(key)
                self.exteriorElementBoundaryQuadratureDictionaryWriter.writeVectorXdmf_exteriorElementBoundaryQuadrature(archive,
                                                                                                                         self.ebqe[key],
                                                                                                                         name,
                                                                                                                         tCount)
        if tensorKeys is not None:
            for key in tensorKeys:
                if isinstance(key,str):
                    name = key
                else:
                    name = "%s" % key[0]
                    for comp in key[1:]:
                        name += "_%s" % comp
                assert key in self.ebqe, "key = %s not found" % str(key)
                self.exteriorElementBoundaryQuadratureDictionaryWriter.writeTensorXdmf_exteriorElementBoundaryQuadrature(archive,
                                                                                                                         self.ebqe[key],
                                                                                                                         name,
                                                                                                                         tCount)
    def archiveFiniteElementResiduals(self,archive,t,tCount,res_dict,res_name_base='spatial_residual'):
        """
        write assembled spatial residual stored in r at time t

        ASSUMES archiveFiniteElementSolutions has already been called for t and tCount!!!

        """
        class dummy(object):
            """
            needed to satisfy api for writeFunctionXdmf
            """
            def __init__(self,ci,r,femSpace):
                self.dof=r
                self.name=res_name_base+'{0}'.format(ci)
                self.femSpace=femSpace
        for ci in res_dict.keys():
            self.u[ci].femSpace.writeFunctionXdmf(archive,dummy(ci,res_dict[ci],self.u[ci].femSpace),tCount)

    def archiveDeclaredFields(self, archive, t, tCount, model_name=None):
        """Write every field this model's coefficients declare.

        Replaces the sixteen hand-rolled ``try/except: pass`` blocks that
        used to live in ``NumericalSolution.py`` -- see
        :mod:`proteus.ArchiveFields` for what was wrong with them.

        Errors are deliberately not caught. A field that was declared and
        silently not written is exactly the failure this replaces; if a
        declaration is broken, the run should say so.

        ``model_name`` is the owning multilevel model's name; it expands
        ``{model}`` in field names, which matters when several models share
        one archive. Defaults to this level model's own name.

        ASSUMES archiveFiniteElementSolutions has already been called for
        ``t``/``tCount`` -- the mesh and grid elements for this timestep
        must exist before fields can be attached to them.
        """
        from .ArchiveFields import archive_fields_for, ArchiveFieldError

        for field in archive_fields_for(self, model_name or getattr(self, "name", None)):
            femSpace = field.resolve_fem_space(self)
            name = field.archive_name
            try:
                if field.on_mesh_nodes:
                    #A field defined on the mesh itself, not on a solution
                    #space: write it straight onto the mesh's own grid. Same
                    #shape as the nodal writer in FemTools -- the DataItem
                    #covers the global node count while each rank
                    #contributes the nodes it owns.
                    mesh = femSpace.mesh
                    if archive.global_sync:
                        owned = mesh.globalMesh.nodeOffsets_subdomain_owned
                        rank = archive.comm.rank()
                        archive.write_field(
                            mesh.arGrid, name, field.value, tCount,
                            center=field.center, rank=field.rank,
                            dimensions=[mesh.globalMesh.nNodes_global],
                            sync_offsets=owned,
                            sync_data=field.value[:(owned[rank+1] - owned[rank])])
                    else:
                        archive.write_field(
                            mesh.arGrid, name, field.value, tCount,
                            center=field.center, rank=field.rank,
                            dimensions=[mesh.nNodes_global])
                else:
                    femSpace.writeFunctionXdmf(
                        archive, _DeclaredFieldFunction(field, name, femSpace), tCount
                    )
            except Exception as exc:
                raise ArchiveFieldError(
                    "failed writing archive field %r at t=%s (tCount=%s): %s"
                    % (name, t, tCount, exc)
                ) from exc
            logEvent(
                "Wrote archive field %s for model %s at t=%s"
                % (name, getattr(self, "name", "?"), t),
                level=3,
            )

    def initializeMassJacobian(self):
        """
        Setup the storage for the mass jacobian and return as a ```SparseMat``` or ```Mat``` based on self.matType
        """
        if self.mass_jacobian is not None:
            return self.mass_jacobian

        from . import superluWrappers
        if self.matType == superluWrappers.SparseMatrix:
            self.nzval_mass = self.nzval.copy()
            self.mass_jacobian = SparseMat(self.nFreeVDOF_global,self.nFreeVDOF_global,self.nnz,
                                           self.nzval_mass,self.colind,self.rowptr)
        elif self.matType == numpy.array:
            self.mass_jacobian = Mat(self.nFreeVDOF_global,self.nFreeVDOF_global)
        else:
            raise TypeError("Matrix type must be sparse matrix or array")
        return self.mass_jacobian

    def initializeSpatialJacobian(self):
        """
        Setup the storage for the spatial jacobian and return as a ```SparseMat``` or ```Mat``` based on self.matType
        """
        if self.space_jacobian is not None:
            return self.space_jacobian
        from . import superluWrappers
        if self.matType == superluWrappers.SparseMatrix:
            self.nzval_space = self.nzval.copy()
            self.space_jacobian = SparseMat(self.nFreeVDOF_global,self.nFreeVDOF_global,self.nnz,
                                           self.nzval_space,self.colind,self.rowptr)
        elif self.matType == numpy.array:
            self.space_jacobian = Mat(self.nFreeVDOF_global,self.nFreeVDOF_global)
        else:
            raise TypeError("Matrix type must be sparse matrix or array")
        return self.space_jacobian


    def calculateElementLoadCoefficients_inhomogeneous(self):
        """
        Calculate the non-solution dependent portion of the Reaction term, 'r'
        Temporary fix for model reduction for linear problems
        Just Zero's the solution and calls the usual update
        """
        #
        #zero u,grad(u), and grad(u)Xgrad(w) at the quadrature points
        # but save the values
        #
        q_save = {}
        for cj in range(self.nc):
            q_save[('u',cj)]=self.q[('u',cj)].copy()
            self.q[('u',cj)].fill(0.0)
            if ('grad(u)',cj) in self.q:
                q_save[('grad(u)',cj)]=self.q[('grad(u)',cj)].copy()
                self.q[('grad(u)',cj)].fill(0.)

        #
        #get functions of (t,x,u) at the quadrature points
        #
        self.coefficients.evaluate(self.timeIntegration.t,self.q)
        if self.movingDomain and self.coefficients.movingDomain:
            self.coefficients.updateToMovingDomain(self.timeIntegration.t,self.q)
        #
        for key in list(q_save.keys()):
            self.q[key].flat[:] = q_save[key].flat[:]
        logEvent("Coefficients on element",level=10,data=self.q)
        ## exterior element boundaries
    def calculateElementLoad_inhomogeneous(self):
        """
        Calculate all the portion of the weak element residual corresponding to terms that
        the 'inhomogeneous' or constant portion of the traditional load vector. This includes purely spatio-temporal
        portions of the reaction term, 'r', and boundary condition terms
        This is a temporary fix for linear model reduction.
        """
        for ci in range(self.nc):
            self.elementResidual[ci].fill(0.0)
        for ci in list(self.coefficients.reaction.keys()):
            cfemIntegrals.updateReaction_weak(self.q[('r',ci)],
                                              self.q[('w*dV_r',ci)],
                                              self.elementResidual[ci])

        if self.numericalFlux is not None:
            for ci in list(self.coefficients.advection.keys()):
                if (ci in self.numericalFlux.advectiveNumericalFlux and
                    self.numericalFlux.advectiveNumericalFlux[ci]) == True:
                    cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                    self.mesh.elementBoundaryElementsArray,
                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                    self.ebqe[('advectiveFlux',ci)],
                                                                    self.ebqe[('w*dS_f',ci)],
                                                                    self.elementResidual[ci])
            for ci in list(self.coefficients.diffusion.keys()):
                if (ci in self.numericalFlux.diffusiveNumericalFlux and
                    self.numericalFlux.diffusiveNumericalFlux[ci]) == True:
                    for ck in self.coefficients.diffusion[ci]:
                        cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                        self.mesh.elementBoundaryElementsArray,
                                                                        self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                        self.ebqe[('diffusiveFlux',ck,ci)],
                                                                        self.ebqe[('w*dS_a',ck,ci)],
                                                                        self.elementResidual[ci])
                        if self.numericalFlux.includeBoundaryAdjoint:
                            if self.sd:
                                if not self.numericalFlux.includeBoundaryAdjointInteriorOnly: #added to only eval interior tjp
                                    cfemIntegrals.updateExteriorElementBoundaryDiffusionAdjoint_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                                               self.numericalFlux.isDOFBoundary[ck],
                                                                                               self.mesh.exteriorElementBoundariesArray,
                                                                                               self.mesh.elementBoundaryElementsArray,
                                                                                               self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                               self.numericalFlux.boundaryAdjoint_sigma,
                                                                                               self.ebqe[('u',ck)],
                                                                                               self.numericalFlux.ebqe[('u',ck)],
                                                                                               self.ebqe['n'],
                                                                                               self.numericalFlux.ebqe[('a',ci,ck)],
                                                                                               self.ebqe[('grad(v)',ci)],#cek grad w
                                                                                               self.ebqe[('dS_u',ci)],
                                                                                               self.elementResidual[ci])
                            else:
                                if not self.numericalFlux.includeBoundaryAdjointInteriorOnly: #added to only eval interior tjp
                                    cfemIntegrals.updateExteriorElementBoundaryDiffusionAdjoint(self.numericalFlux.isDOFBoundary[ck],
                                                                                            self.mesh.exteriorElementBoundariesArray,
                                                                                            self.mesh.elementBoundaryElementsArray,
                                                                                            self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                            self.numericalFlux.boundaryAdjoint_sigma,
                                                                                            self.ebqe[('u',ck)],
                                                                                            self.numericalFlux.ebqe[('u',ck)],
                                                                                            self.ebqe['n'],
                                                                                            self.numericalFlux.ebqe[('a',ci,ck)],
                                                                                            self.ebqe[('grad(v)',ci)],#cek grad w
                                                                                            self.ebqe[('dS_u',ci)],
                                                                                            self.elementResidual[ci])
            for ci in list(self.coefficients.hamiltonian.keys()):
                if (ci in self.numericalFlux.HamiltonJacobiNumericalFlux and
                    self.numericalFlux.HamiltonJacobiNumericalFlux[ci] == True):
                    #1-sided on exterior boundary
                    cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                    self.mesh.elementBoundaryElementsArray,
                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                    self.ebqe[('HamiltonJacobiFlux',ci)],
                                                                    self.ebqe[('w*dS_H',ci)],
                                                                    self.elementResidual[ci])
            for ci in list(self.coefficients.stress.keys()):
                cfemIntegrals.updateExteriorElementBoundaryStressFlux(self.mesh.exteriorElementBoundariesArray,
                                                                      self.mesh.elementBoundaryElementsArray,
                                                                      self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                      self.ebqe[('stressFlux',ci)],
                                                                      self.ebqe[('w*dS_sigma',ci)],
                                                                      self.elementResidual[ci])
        else:
            #cek this will go away
            #cek need to clean up how BC's interact with numerical fluxes
            #outflow <=> zero diffusion, upwind advection
            #mixedflow <=> you set, otherwise calculated as free
            #setflow <=
            for ci,flag in self.fluxBoundaryConditions.items():
                if (flag == 'outFlow' or
                    flag == 'mixedFlow' or
                    flag == 'setFlow'):
                    if ci in self.coefficients.advection:
                        cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                        self.mesh.elementBoundaryElementsArray,
                                                                        self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                        self.ebqe[('advectiveFlux',ci)],
                                                                        self.ebqe[('w*dS_f',ci)],
                                                                        self.elementResidual[ci])
                if (flag == 'mixedFlow' or
                    flag == 'setFlow'):
                    if  ci in self.coefficients.diffusion:
                        for ck in self.coefficients.diffusion[ci]:
                            #
                            cfemIntegrals.updateExteriorElementBoundaryFlux(self.mesh.exteriorElementBoundariesArray,
                                                                            self.mesh.elementBoundaryElementsArray,
                                                                            self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                            self.ebqe[('diffusiveFlux',ck,ci)],
                                                                            self.ebqe[('w*dS_a',ck,ci)],
                                                                            self.elementResidual[ci])

    def calculateExteriorElementBoundaryCoefficients_inhomogeneous(self):
        """
        Calculate the coefficients at global exterior element boundary quadrature points for
        terms that are independent of the solution, i.e., space and time dependent portions of boundary integrals
        Just sets the solution to zero and evaluates the boundary integrals
        This is a temporary fix for linear model reduction
        """
        ebqe_save = {}
        #
        #get u and grad(u) at the quadrature points
        #
        for ci in range(self.nc):
            ebqe_save[('u',ci)]=self.ebqe[('u',ci)].copy()
            self.ebqe[('u',ci)].fill(0.0)
            if ('grad(u)',ci) in self.ebqe:
                ebqe_save[('grad(u)',ci)]=self.ebqe[('grad(u)',ci)].copy()
                self.ebqe[('grad(u)',ci)].fill(0.0)
        #
        #get coefficients at the element boundary quadrature points
        #
        self.coefficients.evaluate(t = self.timeIntegration.t, c = self.ebqe)
        if self.movingDomain and self.coefficients.movingDomain:
            self.coefficients.updateToMovingDomain(self.timeIntegration.t,self.ebqe)
        #
        #time integration
        #
        self.timeIntegration.calculateExteriorElementBoundaryCoefficients(self.ebqe)
        #
        #get phi at the quadrature points if necessary
        #
        for ck,cjDict in self.coefficients.potential.items():
            for cj,flag in cjDict.items():
                ebqe_save[('phi',ck)]=self.ebqe[('phi',ck)].copy()
                self.ebqe[('phi',ck)].fill(0.)
                ebqe_save[('grad(phi)',ci)]=self.ebqe[('grad(phi)',ci)].copy()
                self.ebqe[('grad(phi)',ci)].fill(0.0)
        #
        # calculate the averages and jumps at element boundaries
        #
        if self.numericalFlux is not None:
            self.numericalFlux.calculateExteriorNumericalFlux(self.inflowFlag,self.q,self.ebqe)
        else:
            #cek this wll go away
            for ci,cjDict in self.coefficients.advection.items():
                if (self.fluxBoundaryConditions[ci] == 'outFlow' or
                    self.fluxBoundaryConditions[ci] == 'mixedFlow'):
                    for cj in cjDict:
                        cnumericalFlux.calculateExteriorNumericalAdvectiveFlux_NoBC(self.mesh.exteriorElementBoundariesArray,
                                                                                    self.mesh.elementBoundaryElementsArray,
                                                                                    self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                    self.inflowFlag[ci],
                                                                                    self.ebqe['n'],
                                                                                    self.ebqe[('f',ci)],
                                                                                    self.ebqe[('df',ci,cj)],
                                                                                    self.ebqe[('advectiveFlux',ci)],
                                                                                    self.ebqe[('dadvectiveFlux_left',ci,cj)])
                        # The faces shared with another rank are exterior
                        # faces of this subdomain and interior to the global
                        # mesh, so the flux just computed on them is applied
                        # to the inside of the domain. Mask it off afterwards:
                        # the kernel indexes every ebqe array by its own loop
                        # counter, so it cannot be given a shorter face list.
                        _mask = MeshTools.rankInterfaceExteriorMask(self.mesh)
                        if _mask is not None:
                            self.ebqe[('advectiveFlux',ci)][_mask] = 0.0
                            self.ebqe[('dadvectiveFlux_left',ci,cj)][_mask] = 0.0
        for ci,fbcObject  in self.fluxBoundaryConditionsObjectsDict.items():
            if ci in self.coefficients.advection:
                i0,i1 = fbcObject.evaluate('advective',fbcObject.advectiveFluxBoundaryConditionsDict,
                                           self.ebqe['x'],self.ebqe[('advectiveFlux',ci)],self.timeIntegration.t)
                for cj in list(self.coefficients.advection[ci].keys()):
                    self.ebqe[('dadvectiveFlux_left',ci,cj)][i0,i1] = 0.0
            for ck,diffusiveFluxBoundaryConditionsDict in fbcObject.diffusiveFluxBoundaryConditionsDictDict.items():
                fbcObject.evaluate(('diffusive',ck),diffusiveFluxBoundaryConditionsDict,
                                   self.ebqe['x'],self.ebqe[('diffusiveFlux',ck,ci)],self.timeIntegration.t)
        for ci,sbcObject  in self.stressFluxBoundaryConditionsObjectsDict.items():
            sbcObject.evaluate('stress',sbcObject.stressFluxBoundaryConditionsDict,
                               self.ebqe['x'],self.ebqe[('stressFlux',ci)],self.timeIntegration.t)

        #
        for key in list(ebqe_save.keys()):
            self.ebqe[key].flat[:] = ebqe_save[key].flat[:]

    def getLoadVector(self,f):
        """
        Return the non-solution dependent portion of the Reaction term, 'r' and flux boundary conditions
        """
        f.fill(0.)
        self.calculateElementLoadCoefficients_inhomogeneous()
        self.calculateExteriorElementBoundaryCoefficients_inhomogeneous()
        self.calculateElementLoad_inhomogeneous()
        for ci in range(self.nc):
            cfemIntegrals.updateGlobalResidualFromElementResidual(self.offset[ci],
                                                                  self.stride[ci],
                                                                  self.l2g[ci]['nFreeDOF'],
                                                                  self.l2g[ci]['freeLocal'],
                                                                  self.l2g[ci]['freeGlobal'],
                                                                  self.elementResidual[ci],
                                                                  f);
        logEvent("Global Load Vector",level=9,data=f)


    def getMassJacobian(self,jacobian):
        """
        assemble the portion of the jacobian coming from the time derivative terms
        """
        from . import superluWrappers
        import numpy
        self.calculateElementMassJacobian()
        logEvent("Element Mass Jacobian ",level=10,data=self.elementJacobian)
        if self.matType == superluWrappers.SparseMatrix:
            cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                           jacobian)
            for ci in range(self.nc):
                for cj in self.coefficients.stencil[ci]:
                    #
                    #element contributions
                    #
                    cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[ci]['nFreeDOF'],
                                                                              self.l2g[ci]['freeLocal'],
                                                                              self.l2g[cj]['nFreeDOF'],
                                                                              self.l2g[cj]['freeLocal'],
                                                                              self.csrRowIndeces[(ci,cj)],
                                                                              self.csrColumnOffsets[(ci,cj)],
                                                                              self.elementJacobian[ci][cj],
                                                                              jacobian)

        elif self.matType  == numpy.array:
            jacobian.fill(0.)
            for ci in range(self.nc):
                for cj in self.coefficients.stencil[ci]:
                    #
                    #element contributions
                    #
                    cfemIntegrals.updateGlobalJacobianFromElementJacobian_dense(self.offset[ci],
                                                                                self.stride[ci],
                                                                                self.offset[cj],
                                                                                self.stride[cj],
                                                                                self.nFreeVDOF_global,
                                                                                self.l2g[ci]['nFreeDOF'],
                                                                                self.l2g[ci]['freeLocal'],
                                                                                self.l2g[ci]['freeGlobal'],
                                                                                self.l2g[cj]['nFreeDOF'],
                                                                                self.l2g[cj]['freeLocal'],
                                                                                self.l2g[cj]['freeGlobal'],
                                                                                self.elementJacobian[ci][cj],
                                                                                jacobian)

        else:
            raise TypeError("Matrix type must be SparseMatrix or array")
        logEvent("Mass Jacobian ",level=10,data=jacobian)
        if self.forceStrongConditions:
            for cj in range(self.nc):
                for dofN in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.keys()):
                    global_dofN = self.forcedDOF_global(cj,dofN)
                    for i in range(self.rowptr[global_dofN],self.rowptr[global_dofN+1]):
                        if (self.colind[i] == global_dofN):
                            self.nzval[i] = 1.0
                        else:
                            self.nzval[i] = 0.0
        return jacobian
    #
    def getSpatialJacobian(self,jacobian):
        return self.getJacobian(jacobian,skipMassTerms=True)

    def getSpatialResidual(self,u,r):
        """
        Calculate the element spatial residuals and add in to the global residual
        This is being used right now to test nonlinear pod and deim
        """
        r.fill(0.0)
        #Load the Dirichlet conditions
        for cj in range(self.nc):
            for dofN,g in self.dirichletConditions[cj].DOFBoundaryConditionsDict.items():
                self.u[cj].dof[dofN] = g(self.dirichletConditions[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN,g in self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items():
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
        #Load the unknowns into the finite element dof
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        self.calculateCoefficients()
        self.calculateElementResidual()
        self.scale_dt = False
        logEvent("Element spatial residual",level=9,data=self.elementSpatialResidual)
        if self.timeIntegration.dt < 1.0e-8*self.mesh.h:
            self.scale_dt = True
            logEvent("Rescaling residual for small time steps")
        else:
            self.scale_dt = False
        for ci in range(self.nc):
            if self.scale_dt:
                self.elementSpatialResidual[ci]*=self.timeIntegration.dt
            cfemIntegrals.updateGlobalResidualFromElementResidual(self.offset[ci],
                                                                  self.stride[ci],
                                                                  self.l2g[ci]['nFreeDOF'],
                                                                  self.l2g[ci]['freeLocal'],
                                                                  self.l2g[ci]['freeGlobal'],
                                                                  self.elementSpatialResidual[ci],
                                                                  r);
        logEvent("Global spatial residual",level=9,data=r)
        if self.forceStrongConditions:#
            for cj in range(len(self.dirichletConditionsForceDOF)):#
                for dofN,g in self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items():
                    r[self.forcedDOF_global(cj,dofN)] = 0
    def getMassResidual(self,u,r):
        """
        Calculate the portion of the element residuals associated with the temporal term and assemble into a global residual
        This is being used right now to test nonlinear pod and deim and is inefficient
        """
        r.fill(0.0)
        #Load the Dirichlet conditions
        for cj in range(self.nc):
            for dofN,g in self.dirichletConditions[cj].DOFBoundaryConditionsDict.items():
                self.u[cj].dof[dofN] = g(self.dirichletConditions[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN,g in self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items():
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
        #Load the unknowns into the finite element dof
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        self.calculateCoefficients()
        self.calculateElementResidual()
        elementMassResidual = {}
        for ci in range(self.nc):
            elementMassResidual[ci]=np.copy(self.elementResidual[ci])
            elementMassResidual[ci]-= self.elementSpatialResidual[ci]

        self.scale_dt = False
        logEvent("Element Mass residual",level=9,data=elementMassResidual)
        if self.timeIntegration.dt < 1.0e-8*self.mesh.h:
            self.scale_dt = True
            logEvent("Rescaling residual for small time steps")
        else:
            self.scale_dt = False
        for ci in range(self.nc):
            if self.scale_dt:
                elementMassResidual[ci]*=self.timeIntegration.dt
            cfemIntegrals.updateGlobalResidualFromElementResidual(self.offset[ci],
                                                                  self.stride[ci],
                                                                  self.l2g[ci]['nFreeDOF'],
                                                                  self.l2g[ci]['freeLocal'],
                                                                  self.l2g[ci]['freeGlobal'],
                                                                  elementMassResidual[ci],
                                                                  r);
        logEvent("Global mass residual",level=9,data=r)
        if self.forceStrongConditions:#
            for cj in range(len(self.dirichletConditionsForceDOF)):#
                for dofN,g in self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items():
                    r[self.forcedDOF_global(cj,dofN)] = 0

    def attachLaplaceOperator(self,nu=1.0):
        """Attach a Discrete Laplace Operator to the Transport Class.

        Arguments
        ---------
        nu : float
             Viscosity parameter for the Laplace operator.

        Notes
        -----
        This function creates a Laplace matrix and stores the result in the 
        :class:`proteus.OneLevelTransport` class to assist in the construction 
        of physics based preconditione
        """

        self.laplace_val = self.nzval.copy()
        self.LaplaceOperator = SparseMat(self.nFreeVDOF_global,
                                         self.nFreeVDOF_global,
                                         self.nnz,
                                         self.laplace_val,
                                         self.colind,
                                         self.rowptr)

        _nd = self.coefficients.nd
        if self.coefficients.nu is not None:
            _nu = self.coefficients.nu
        self.LaplaceOperatorCoeff = DiscreteLaplaceOperator(nd=_nd)
        _t = 1.0

        Laplace_phi = {}
        Laplace_dphi = {}

        for ci,space in self.testSpace.items():
            Laplace_phi[ci] = FiniteElementFunction(space)

        for ck,phi in Laplace_phi.items():
            Laplace_dphi[(ck,ck)] = FiniteElementFunction(Laplace_phi[ck].femSpace)

        for ci,dphi in Laplace_dphi.items():
            dphi.dof.fill(1.0)

        self.Laplace_q = {}
        scalar_quad = StorageSet(shape=(self.mesh.nElements_global,
                                        self.nQuadraturePoints_element))
        tensors_quad = StorageSet(shape={})
        
        scalar_quad |= set([('u',ci) for ci in range(self.nc)])
        tensors_quad |= set([('a',ci,ci) for ci in range(self.nc)])
        tensors_quad |= set([('da',ci,ci,ci) for ci in range(self.nc)])

        scalar_quad.allocate(self.Laplace_q)
        
        for k in tensors_quad:
            self.Laplace_q[k] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.nQuadraturePoints_element,
                 self.LaplaceOperatorCoeff.sdInfo[(k[1],k[2])][0][self.nSpace_global]),
                'd')

        if _nd == 2:
            self.LaplaceOperatorCoeff.evaluate(_t,self.Laplace_q)
        
        LaplaceJacobian = {}
        for ci in range(self.nc):
            LaplaceJacobian[ci] = {}
            for cj in range(self.nc):
                if cj in self.LaplaceOperatorCoeff.stencil[ci]:
                    LaplaceJacobian[ci][cj] = numpy.zeros(
                        (self.mesh.nElements_global,
                         self.nDOF_test_element[ci],
                         self.nDOF_trial_element[cj]),
                        'd')

        for ci,ckDict in self.LaplaceOperatorCoeff.diffusion.items():
            for ck,cjDict in ckDict.items():
                for cj in set(list(cjDict.keys())+list(self.LaplaceOperatorCoeff.potential[ck].keys())):
                    cfemIntegrals.updateDiffusionJacobian_weak_sd(self.LaplaceOperatorCoeff.sdInfo[(ci,ck)][0],
                                                                  self.LaplaceOperatorCoeff.sdInfo[(ci,ck)][1],
                                                                  self.phi[ck].femSpace.dofMap.l2g,
                                                                  self.Laplace_q[('a',ci,ck)],
                                                                  self.Laplace_q[('da',ci,ck,cj)],
                                                                  self.q[('grad(phi)',ck)],
                                                                  self.q[('grad(w)*dV_a',ck,ci)],
                                                                  Laplace_dphi[(ck,cj)].dof,
                                                                  self.q[('v',cj)],
                                                                  self.q[('grad(v)',cj)],
                                                                  LaplaceJacobian[ci][cj])
        for ci in range(self.nc):
            for cj in self.LaplaceOperatorCoeff.stencil[ci]:
                cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[ci]['nFreeDOF'],
                                                                          self.l2g[ci]['freeLocal'],
                                                                          self.l2g[cj]['nFreeDOF'],
                                                                          self.l2g[cj]['freeLocal'],
                                                                          self.csrRowIndeces[(ci,cj)],
                                                                          self.csrColumnOffsets[(ci,cj)],
                                                                          LaplaceJacobian[ci][cj],
                                                                          self.LaplaceOperator)

        self.LaplaceOperatorpetsc = superlu_2_petsc4py(self.LaplaceOperator)
        
        var_range = []
        isp_list = []
        starting_idx = 0
        for comp in range(self.nc):
            comp_num_dof = self.phi[comp].femSpace.dofMap.nDOF
            var_range.append(numpy.arange(start=starting_idx,
                                          stop=starting_idx+comp_num_dof,
                                          dtype='i'))
            isp_list.append(p4pyPETSc.IS())
            isp_list[comp].createGeneral(var_range[comp])
            starting_idx+=comp_num_dof

        self.LaplaceOperatorList = []
        for comp in range(self.nc):
            self.LaplaceOperatorList.append(self.LaplaceOperatorpetsc.getSubMatrix
                                            (isp_list[comp],
                                             isp_list[comp]))

#end Transport definition
class MultilevelTransport(object):
    """Nonlinear ADR on a multilevel mesh"""
    #None or False: distributed transfer operators only when there is more than one rank
    distributedTransfers = False
    def __init__(self,problem,numerics,mlMesh,OneLevelTransportType=OneLevelTransport):
        self.name = problem.name
        #cek temporary fix to get everything weaned off the old BC's
        if numerics.numericalFluxType is None:
            numerics.numericalFluxType = NumericalFlux.StrongDirichletFactory(problem.fluxBoundaryConditions)
        self.OneLevelTransportType=OneLevelTransportType
        phiSpaces = None
        if 'phiSpaces' in dir(numerics):
            phiSpaces = numerics.phiSpaces
        self.initialize(
            problem.nd,
            mlMesh,
            numerics.femSpaces,
            numerics.femSpaces,
            numerics.matrix,
            problem.dirichletConditions,
            problem.coefficients,
            numerics.elementQuadrature,
            numerics.elementBoundaryQuadrature,
            problem.fluxBoundaryConditions,
            problem.advectiveFluxBoundaryConditions,
            problem.diffusiveFluxBoundaryConditions,
            problem.stressFluxBoundaryConditions,
            numerics.subgridError,
            numerics.shockCapturing,
            numerics.conservativeFlux,
            numerics.numericalFluxType,
            numerics.timeIntegration,
            numerics.massLumping,
            numerics.reactionLumping,
            problem.weakDirichletConditions,
            numerics,
            problem.sd,
            problem.movingDomain,
            PhiSpaceTypeDict=phiSpaces)
    def initialize(self,
                   nd,
                   mlMesh,
                   TrialSpaceTypeDict,
                   TestSpaceTypeDict,
                   matType,
                   dirichletConditionsSetterDict,
                   coefficients,
                   quadratureDict,
                   elementBoundaryQuadratureDict,
                   fluxBoundaryConditionsDict='noFlow',#'outFlow',
                   advectiveFluxBoundaryConditionsSetterDict={},
                   diffusiveFluxBoundaryConditionsSetterDictDict={},
                   stressFluxBoundaryConditionsSetterDict={},
                   stabilization=None,
                   shockCapturing=None,
                   conservativeFlux=None,
                   numericalFluxType=None,
                   TimeIntegrationClass = BackwardEuler,
                   massLumping=False,
                   reactionLumping=False,
                   weakDirichletConditions=None,
                   options=None,
                   useSparseDiffusion=True,
                   movingDomain=False,
                   PhiSpaceTypeDict=None):
        import copy
        """read in the multilevel mesh, mesh independent boundary
        conditions, and types for test and trial spaces and the
        jacobian. Pass through the rest to the models on each mesh"""
        if bool(TrialSpaceTypeDict) == False:
            raise Exception('Proteus is trying to create a' \
            ' Multilevel Transport object with no trial space.  Make' \
            ' sure femSpaces is properly specified in your numerics.')
        
        self.weakDirichletConditions=weakDirichletConditions
        self.jacobianList=[]
        self.par_jacobianList=[]
        self.uDictList=[]
        self.uList=[]
        self.duList=[]
        self.rList=[]
        self.par_uList=[]
        self.par_duList=[]
        self.par_rList=[]
        self.trialSpaceDictList=[]
        self.trialSpaceListDict={}
        self.testSpaceDictList=[]
        self.bcDictList=[]
        self.bcListDict={}
        self.levelModelList=[]
        self.offsetListList=[]
        self.strideListList=[]
        self.matType = matType
        #mwf debug
        if PhiSpaceTypeDict is None: #by default phi in same space as u
            PhiSpaceTypeDict = TrialSpaceTypeDict
        self.phiSpaceDictList = []
        #self.phiSpaceListDict = {}

        logEvent("Building Transport for each mesh",level=2)
        for  cj in list(TrialSpaceTypeDict.keys()):
            self.trialSpaceListDict[cj]=[]
            self.bcListDict[cj]=[]
        # In parallel the transfer operators are built per rank from the
        # subdomain spaces (see useParallelTransfers); a single level has no
        # transfers at all.
        from . import Comm
        _comm = Comm.get()
        #Distributed (PETSc Mat) transfers are what parallel runs need. A serial run
        #uses the global SparseMatrix ones unless distributedTransfers is True, which
        #lets a one-rank run use exactly the code a many-rank run does -- PETSc's PCMG
        #needs it, and so does a scaling study with a one-rank baseline.
        self.parallelTransfers = ((_comm.size() > 1 or self.distributedTransfers)
                                  and len(mlMesh.meshList) > 1)
        for mesh in mlMesh.meshList:
            
            sdmesh = mesh.subdomainMesh # Linoj: mesh.subdomainMesh.nodeArray[:,2]=all zeros
            
            memory()
            logEvent("Generating Trial Space",level=2)
            trialSpaceDict = dict([ (cj,TrialSpaceType(sdmesh,nd)) for (cj,TrialSpaceType) in TrialSpaceTypeDict.items()])
            self.trialSpaceDictList.append(trialSpaceDict)
            logEvent("Generating Test Space",level=2)
            testSpaceDict = dict([(ci,TestSpaceType(sdmesh,nd)) for (ci,TestSpaceType) in TestSpaceTypeDict.items()])
            self.testSpaceDictList.append(testSpaceDict)
            logEvent("Allocating u",level=2)
            uDict = dict([(cj,FiniteElementFunction(trialSpace,name=coefficients.variableNames[cj])) for (cj,trialSpace) in trialSpaceDict.items()])
            #11/11/09 allow FiniteElementFunction to communicate uknowns across procs
            for cj in list(uDict.keys()):
                uDict[cj].setupParallelCommunication()
            self.uDictList.append(uDict)
            logEvent("Allocating phi")
            phiSpaceDict = dict([ (cj,PhiSpaceType(sdmesh,nd)) for (cj,PhiSpaceType) in PhiSpaceTypeDict.items()])
            self.phiSpaceDictList.append(phiSpaceDict)
            phiDict = dict([(cj,FiniteElementFunction(phiSpace)) for (cj,phiSpace) in phiSpaceDict.items()])
            #need to communicate phi if nonlinear potential and there is spatial dependence in potential function
            for cj in list(phiDict.keys()):
                phiDict[cj].setupParallelCommunication()
            logEvent(memory("finite element spaces","MultilevelTransport"),level=4)
            logEvent("Setting Boundary Conditions")
            if numericalFluxType is None:
                useWeakDirichletConditions=False
            else:
                useWeakDirichletConditions=numericalFluxType.useWeakDirichletConditions
            logEvent("Setting Boundary Conditions-1")
            for cj in list(trialSpaceDict.keys()):
                if cj not in dirichletConditionsSetterDict:

                    dirichletConditionsSetterDict[cj] = None
                if cj not in fluxBoundaryConditionsDict:
                    fluxBoundaryConditionsDict[cj] = None
            logEvent("Setting Boundary Conditions-2")
            if options is None or options.periodicDirichletConditions is None or options.parallelPeriodic==True:
                logEvent("Setting Boundary Conditions-2a")
                dirichletConditionsDict=dict([(cj,DOFBoundaryConditions(
                    trialSpace,dirichletConditionsSetterDict[cj],useWeakDirichletConditions))
                                              for (cj,trialSpace) in trialSpaceDict.items()])
            else:
                logEvent("Setting Boundary Conditions-2b")
                # Strong Dirichlet constraints kept in the system
                # (StrongDirichletFactory(keepDOFs=True), for KSP_petsc4py)
                # still merge periodic pairs, and so does a weak-Dirichlet flux
                # that asks for it (NF_base.mergesPeriodicDOFs: it then treats
                # the seam as interior); any other weak-Dirichlet flux leaves
                # periodicity to the flux, as it always has.
                keptStrong = (numericalFluxType is not None and
                              (getattr(numericalFluxType,'useStrongDirichletConstraints',False) or
                               (hasattr(numericalFluxType,'mergesPeriodicDOFs') and
                                numericalFluxType.mergesPeriodicDOFs())))
                dirichletConditionsDict=dict([(cj,DOFBoundaryConditions(
                    trialSpace,dirichletConditionsSetterDict[cj],useWeakDirichletConditions,options.periodicDirichletConditions[cj],
                    mergeKeptPeriodicDOFs=keptStrong))
                                              for (cj,trialSpace) in trialSpaceDict.items()])
            logEvent("Setting Boundary Conditions-3")
            self.bcDictList.append(dirichletConditionsDict)
            logEvent("Setting Boundary Conditions-4")
            for cj in list(TrialSpaceTypeDict.keys()):
                self.trialSpaceListDict[cj].append(trialSpaceDict[cj])
                self.bcListDict[cj].append(dirichletConditionsDict[cj])
            #cek try setting parallel periodic conditions
            #
            from . import Comm
            comm = Comm.get()
            if options.periodicDirichletConditions and options.parallelPeriodic:
                logEvent("Generating Trial Space--Parallel Periodic",level=2)
                #the global mesh has been renumbered so all this is in the new (partitioned) numberings
                trialSpace_global = TrialSpaceTypeDict[0](mesh,nd)
                logEvent("Setting Boundary Conditions--Parallel Periodic")
                #get new global numbering of DOF that maps periodic DOF to a single master DOF
                periodicConditions_global=DOFBoundaryConditions(trialSpace_global,
                                                                lambda x,flag: None,
                                                                False,
                                                                options.periodicDirichletConditions[0])
                #collect the free DOF
                ownedDOF=set()
                globalDOF=set()
                trialSpaceDict[0].dofMap.nDOF_all_processes = periodicConditions_global.nFreeDOF_global
                for nN_subdomain,nN_global in enumerate(trialSpaceDict[0].dofMap.subdomain2global):
                    nN_free_global   = periodicConditions_global.global2freeGlobal[nN_global]
                    nN_global_global = periodicConditions_global.myFreeDOF[nN_global]
                    if nN_global_global != nN_global:#this node is now pointing to a different global node so it's periodic
                        if (nN_global_global < trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned[comm.rank()] or
                            nN_global_global >= trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned[comm.rank()+1]):#the node it points to is off proc
                            globalDOF.add(nN_free_global)#globalDOF.append(nN_free_global)
                    elif (nN_global < trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned[comm.rank()] or
                          nN_global >= trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned[comm.rank()+1]):#this node is a regular unknown off processor
                        globalDOF.add(nN_free_global)#globalDOF.append(nN_free_global)
                    else:#this node is a regular unknown DOF
                        ownedDOF.add(nN_free_global)#ownedDOF.append(nN_free_global)
                #sort DOF's separately so ghost DOF stay at end
                globalDOF = list(globalDOF)
                globalDOF.sort()
                ownedDOF = list(ownedDOF)
                ownedDOF.sort()
                dofList = ownedDOF + globalDOF
                #set the number of DOF owned on subdomain and the global offsets
                trialSpaceDict[0].dofMap.nDOF_subdomain_owned = len(ownedDOF)
                trialSpaceDict[0].dofMap.nDOF_subdomain = len(dofList)
                trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned = mesh.nodeOffsets_subdomain_owned.copy()
                trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned[comm.rank()] = ownedDOF[0]
                trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned[comm.rank()+1] = ownedDOF[-1]+1
                trialSpaceDict[0].dofMap.nDOF = trialSpaceDict[0].dofMap.nDOF_subdomain
                trialSpaceDict[0].dofMap.range_nDOF = range(trialSpaceDict[0].dofMap.nDOF)
                #build a mapping of the subdomain DOF to the free subdomain DOF
                subdomain_global2freeGlobal={}
                for nN_subdomain,nN_global in enumerate(trialSpaceDict[0].dofMap.subdomain2global):
                    nN_free_global = periodicConditions_global.global2freeGlobal[nN_global]
                    subdomain_global2freeGlobal[nN_subdomain] = dofList.index(nN_free_global)
                #build the new mapping of free subdomain dof to global dof
                trialSpaceDict[0].dofMap.subdomain2global = numpy.zeros((trialSpaceDict[0].dofMap.nDOF_subdomain,),cmeshTools.INDEX_DTYPE)
                for nN_subdomain in range(trialSpaceDict[0].dofMap.nDOF_subdomain):
                    trialSpaceDict[0].dofMap.subdomain2global[nN_subdomain] = dofList[nN_subdomain]#maps free dof on subdomain to global free dof
                #build arrays exactly as in FemTools but using subdomain_globa2freeGlobal we built above
                bc = self.bcDictList[-1][0]
                bc.nFreeDOF_global = len(dofList)
                bc.global2freeGlobal = subdomain_global2freeGlobal
                for i,dofN in enumerate(bc.global2freeGlobal.keys()):
                    bc.global2freeGlobal_global_dofs[i] = dofN#map each of the unknown DOF's to the original node number
                    bc.global2freeGlobal_free_dofs[i] = subdomain_global2freeGlobal[dofN]#map each of the unknown DOF's to the free unknown number
                #now modify the remaining bc and testspace objects
                for ci in range(1,len(trialSpaceDict)):
                    self.bcDictList[-1][ci].global2freeGlobal_free_dofs = bc.global2freeGlobal_free_dofs
                    self.bcDictList[-1][ci].global2freeGlobal_global_dofs = bc.global2freeGlobal_global_dofs
                    self.bcDictList[-1][ci].global2freeGlobal = subdomain_global2freeGlobal
                    self.bcDictList[-1][ci].nFreeDOF_global = len(dofList)
                for ci in range(1,len(trialSpaceDict)):
                    trialSpaceDict[ci].dofMap.nDOF_subdomain = trialSpaceDict[0].dofMap.nDOF_subdomain
                    trialSpaceDict[ci].dofMap.subdomain2global = trialSpaceDict[0].dofMap.subdomain2global
                    trialSpaceDict[ci].dofMap.dof_offsets_subdomain_owned = trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned
                    trialSpaceDict[ci].dofMap.nDOF_all_processes = trialSpaceDict[0].dofMap.nDOF_all_processes
                    trialSpaceDict[ci].dofMap.nDOF_subdomain_owned = trialSpaceDict[0].dofMap.nDOF_subdomain_owned
                    trialSpaceDict[ci].dofMap.nDOF = trialSpaceDict[ci].dofMap.nDOF_subdomain
                    trialSpaceDict[ci].dofMap.range_nDOF = range(trialSpaceDict[ci].dofMap.nDOF)
                # one unknown per periodic pair now, as the serial merge makes
                # (NF_base.seamIsInterior reads this)
                for ci in range(len(trialSpaceDict)):
                    self.bcDictList[-1][ci].periodicDOFsMerged = True
                for ci in range(len(testSpaceDict)):
                    testSpaceDict[ci].dofMap.nDOF_subdomain = trialSpaceDict[0].dofMap.nDOF_subdomain
                    testSpaceDict[ci].dofMap.subdomain2global = trialSpaceDict[0].dofMap.subdomain2global
                    testSpaceDict[ci].dofMap.dof_offsets_subdomain_owned = trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned
                    testSpaceDict[ci].dofMap.nDOF_all_processes = trialSpaceDict[0].dofMap.nDOF_all_processes
                    testSpaceDict[ci].dofMap.nDOF_subdomain_owned = trialSpaceDict[0].dofMap.nDOF_subdomain_owned
                    testSpaceDict[ci].dofMap.nDOF = testSpaceDict[ci].dofMap.nDOF_subdomain
                    testSpaceDict[ci].dofMap.range_nDOF = range(testSpaceDict[ci].dofMap.nDOF)
            #
            #
            logEvent(memory("boundary conditions","MultilevelTransport"),level=4)
            logEvent("Initializing OneLevelTransport",level=2)
            
            uDict[0].femSpace.mesh.nLayersOfOverlap = mesh.nLayersOfOverlap
            uDict[0].femSpace.mesh.parallelPartitioningType = mesh.parallelPartitioningType
            uDict[0].femSpace.mesh.nodeNumbering_subdomain2global = mesh.nodeNumbering_subdomain2global     ###NEW LINE
            #uDict[0].femSpace.mesh.globalMesh.volume = mesh.volume
            #uDict[0].femSpace.mesh.nodeArray = mesh.nodeArray[mesh.nodeNumbering_subdomain2global,:]        ###NEW LINE
            
            transport=self.OneLevelTransportType(uDict,
                                            phiDict,
                                            testSpaceDict,
                                            matType,
                                            dirichletConditionsDict,
                                            dirichletConditionsSetterDict,
                                            copy.deepcopy(coefficients),
                                            quadratureDict,
                                            elementBoundaryQuadratureDict,
                                            fluxBoundaryConditionsDict,
                                            advectiveFluxBoundaryConditionsSetterDict,
                                            diffusiveFluxBoundaryConditionsSetterDictDict,
                                            stressFluxBoundaryConditionsSetterDict,
                                            copy.deepcopy(stabilization),
                                            copy.deepcopy(shockCapturing),
                                            conservativeFlux,
                                            numericalFluxType,
                                            TimeIntegrationClass,
                                            massLumping,
                                            reactionLumping,
                                            options,
                                            self.name + str(len(self.levelModelList)),
                                            sd=useSparseDiffusion,
                                            movingDomain=movingDomain)
            
            self.offsetListList.append(transport.offset)
            self.strideListList.append(transport.stride)
            memory()
            self.levelModelList.append(transport)
            logEvent("Allocating residual and solution vectors",level=2)
            u = numpy.zeros((transport.dim,),'d')
            du = numpy.zeros((transport.dim,),'d')
            r = numpy.zeros((transport.dim,),'d')
            self.uList.append(u)
            self.duList.append(du)
            self.rList.append(r)
            logEvent("Allocating Jacobian",level=2)
            jacobian = transport.initializeJacobian()
            self.jacobianList.append(jacobian)
            par_bs = transport.coefficients.nc
            logEvent("Allocating parallel storage",level=2)
            comm = Comm.get()
            self.comm=comm               
            
            if (comm.size() > 1):
                for ci in range(transport.nc):
                    assert trialSpaceDict[ci].dofMap.dof_offsets_subdomain_owned is not None, "trial space %s needs subdomain -> global mappings " % trialSpaceDict
                
                #initially assume all the spaces can share the same l2g information ...
                par_n = trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned[comm.rank()+1] - trialSpaceDict[0].dofMap.dof_offsets_subdomain_owned[comm.rank()]
                par_N = trialSpaceDict[0].dofMap.nDOF_all_processes
                
                mixed = False
                for ts in list(trialSpaceDict.values()):
                    if ts.dofMap.nDOF_all_processes != par_N:
                        mixed=True
                #proteus's own smoothers refresh the ghost entries of their
                #iterate through par_u after each local update, so they are
                #usable in parallel: Jacobi's update is pointwise and gives the
                #same iterates as in serial, while GaussSeidel becomes
                #processor-block Gauss-Seidel and StarILU/StarBILU are already
                #Schwarz methods. MGM and NI drive those.
                #
                #LU is still refused. It would factor the *subdomain* matrix,
                #which is a block-Jacobi approximation rather than the direct
                #solve the caller asked for, and would quietly return a
                #different answer in parallel than in serial.
                #StarILU is usable once two things hold: its sweep is
                #restricted to stars centred on nodes this rank owns, and the
                #residual's ghost entries carry their owners' values before the
                #local solves read them. Both are handled in LinearSolvers.
                #
                #StarBILU is absent because it returns a zero solution even in
                #serial on the problems tested here, which is its own question;
                #there is nothing to preserve in parallel until that is fixed.
                #
                #LU is absent for a different reason: it would factor the
                #subdomain matrix, which is a block-Jacobi approximation rather
                #than the direct solve the caller asked for.
                parallelCapableSolvers = (KSP_petsc4py,Jacobi,GaussSeidel,StarILU,MGM,NI,petsc_MG)
                for (what,solver) in (("multilevelLinearSolver",options.multilevelLinearSolver),
                                      ("levelLinearSolver",options.levelLinearSolver),
                                      ("linearSmoother",getattr(options,'linearSmoother',None))):
                    if solver is None:
                        continue
                    #the KSP_Preconditioner family (petsc_ASM, petsc_LU, the Schur
                    #complement preconditioners such as SimpleNavierStokes2D) sets up
                    #PETSc preconditioners on the global parallel matrix for
                    #KSP_petsc4py; it is how parallel Navier-Stokes is run
                    assert (solver in parallelCapableSolvers or
                            (isinstance(solver,type) and issubclass(solver,KSP_Preconditioner))), (
                        "%s=%s cannot be used in parallel: it would return a "
                        "different answer than in serial. Use one of %s."
                        % (what,getattr(solver,'__name__',solver),
                           [c.__name__ for c in parallelCapableSolvers]+["KSP_Preconditioner subclasses"]))
                if mixed:
                    #
                    #calculate dimensions mixed multicomponent system
                    #since it's not nc*dim
                    #
                    #first create list of global dof dim for each component
                    #this is the proteus component global numbering
                    
                    par_N_list = [ts.dofMap.nDOF_all_processes for ts in list(trialSpaceDict.values())]
                    #sum to get total global dimension
                    par_N = sum(par_N_list)
                    #calculate list of  dimensions of the locally owned
                    #dof for  each component
                    par_n_list = [ts.dofMap.dof_offsets_subdomain_owned[comm.rank()+1] -
                                  ts.dofMap.dof_offsets_subdomain_owned[comm.rank()]
                                  for ts in list(trialSpaceDict.values())]
                    #sum to get total local dimension
                    par_n = sum(par_n_list)
                    #calculate number of  ghosts  fo r each component
                    par_nghost_list = [ts.dofMap.nDOF_subdomain - nDOF_owned
                                       for ts,nDOF_owned in zip(list(trialSpaceDict.values()),
                                                                par_n_list)]
                    #sum to  get total ghost dimension
                    par_nghost = sum(par_nghost_list)
                    #
                    #calculate offsets
                    #
                    #first for petsc global (natural) numbering offsets
                    # if p and q are components and ^o and ^g are
                    # owned and ghost this numbering has
                    # p^o_0, p^o_1,...p^o_{np}, q^o_0,...q^0_{nq}, p^g_0...
                    # i.e. all owned, then all ghost
                    # we'll eventually need the processor and local
                    # component index  of each ghost too
                    petsc_global_offsets=[0]
                    petsc_component_offsets=[[0]]
                    component_ghost_proc={}
                    component_ghost_local_index={}
                    total_dof_proc=0
                    for proc in range(comm.size()):
                        for ci,ts in enumerate(trialSpaceDict.values()):
                            total_dof_proc += (ts.dofMap.dof_offsets_subdomain_owned[proc+1] -
                                               ts.dofMap.dof_offsets_subdomain_owned[proc])
                            petsc_component_offsets[proc].append(total_dof_proc)
                            for g in ts.dofMap.subdomain2global[par_n_list[ci]:ts.dofMap.nDOF_subdomain]:
                                if g >= ts.dofMap.dof_offsets_subdomain_owned[proc] and g < ts.dofMap.dof_offsets_subdomain_owned[proc+1]:
                                    # key by (component, global-id): different
                                    # components have INDEPENDENT global numberings
                                    # (each starts at 0), so the same bare id g can
                                    # be a ghost in two components with DIFFERENT
                                    # owners/local-indices.  Keying by g alone makes
                                    # the later component clobber the earlier one ->
                                    # scrambled petsc ghost columns -> singular
                                    # Jacobian (the node-split-z mixed system hits
                                    # exactly this overlap).
                                    component_ghost_proc[(ci,g)] = proc
                                    component_ghost_local_index[(ci,g)] = g -  ts.dofMap.dof_offsets_subdomain_owned[proc]
                        petsc_component_offsets.append([petsc_component_offsets[-1][-1]])
                        petsc_global_offsets.append(total_dof_proc)
                    #calculate proteus global (end-to-end) offset for each component
                    global_component_offset = [0]
                    for ts in list(trialSpaceDict.values()):
                        global_component_offset.append(global_component_offset[-1] + ts.dofMap.nDOF_all_processes)
                    #
                    #now create subdomain2global for stacked ghosted component DOF by shifting global DOF  numbers  by global offsets
                    #
                    subdomain2global = numpy.hstack([offset+ts.dofMap.subdomain2global
                                                     for offset,ts in
                                                     zip(global_component_offset, list(trialSpaceDict.values()))])
                    
                    #
                    #store ghost proc w.r.t. proteus  global number
                    #store petsc global w.r.t. proteus global number
                    ghost_proc={}
                    ghost_global2petsc={}
                    for ci,offset, ts in zip(list(range(transport.nc)), global_component_offset, list(trialSpaceDict.values())):
                        # iterate ONLY this component's ghosts (component-local
                        # global ids) -- not the merged dict -- and look them up
                        # by (ci,g) so an id shared across components resolves to
                        # the right owner/local-index for THIS component.
                        for g in ts.dofMap.subdomain2global[par_n_list[ci]:ts.dofMap.nDOF_subdomain]:
                            gproc = component_ghost_proc[(ci,g)]
                            ghost_proc[offset+g] = gproc
                            ghost_global2petsc[offset+g] = petsc_component_offsets[gproc][ci] + component_ghost_local_index[(ci,g)]
                    #
                    #build subdomain2global mappings
                    #
                    subdomain2global_list = [ts.dofMap.subdomain2global for ts in  list(trialSpaceDict.values())]
                    #
                    #form lists of owned and ghosts elements
                    #
                    ghosts_list=[]
                    owned_list=[]
                    owned = numpy.zeros((par_n),'i')
                    owned_local = numpy.zeros((par_n),'i')
                    ghosts = numpy.zeros((par_nghost),'i')
                    ghosts_start = 0
                    owned_start = 0
                    for ci in range(transport.nc):
                        #owned
                        owned_ci = subdomain2global[transport.offset[ci]:transport.offset[ci]+par_n_list[ci]]
                        owned_local[owned_start:owned_start + par_n_list[ci]] = numpy.arange(transport.offset[ci],transport.offset[ci]+par_n_list[ci])
                        owned[owned_start:owned_start + par_n_list[ci]] = owned_ci
                        owned_list.append(owned_ci)
                        owned_start = owned_start + par_n_list[ci]
                        #ghosts
                        ghosts_ci = subdomain2global[transport.offset[ci]+par_n_list[ci]:transport.offset[ci]+par_n_list[ci]+par_nghost_list[ci]]
                        ghosts[ghosts_start:ghosts_start + par_nghost_list[ci]] = ghosts_ci
                        ghosts_list.append(ghosts_ci)
                        ghosts_start = ghosts_start + par_nghost_list[ci]
                    transport.owned_local = owned_local
                    #stack all owned and all ghost
                    petsc_subdomain2global = np.hstack(owned_list+ghosts_list)
                    #
                    # now find mappings (permutations) between proteus and
                    # petsc orderings
                    perm_petsc_subdomain2global = petsc_subdomain2global.argsort()
                    perm_subdomain2global = subdomain2global.argsort()
                    proteus2petsc_subdomain = np.array(perm_petsc_subdomain2global[perm_subdomain2global.argsort()],'i')
                    petsc2proteus_subdomain = np.array(perm_subdomain2global[perm_petsc_subdomain2global.argsort()],'i')
                    assert((petsc_subdomain2global[perm_petsc_subdomain2global] == subdomain2global[perm_subdomain2global]).all())
                    assert((petsc_subdomain2global[proteus2petsc_subdomain] == subdomain2global).all())
                    assert((subdomain2global[petsc2proteus_subdomain] == petsc_subdomain2global).all())
                    petsc_subdomain2global_petsc = subdomain2global.copy()
                    petsc_subdomain2global_petsc[:]=-1
                    for i in range(par_n):
                        petsc_subdomain2global_petsc[i] = petsc_global_offsets[comm.rank()]+i
                    petsc_ghosts=ghosts.copy()
                    for gi,i in enumerate(range(par_n,par_n+par_nghost)):
                        global_proteus = subdomain2global[petsc2proteus_subdomain[i]]
                        petsc_subdomain2global_petsc[i] = ghost_global2petsc[global_proteus]
                        petsc_ghosts[gi] = petsc_subdomain2global_petsc[i]
                    if comm.size() == 1:
                        assert((subdomain2global == petsc_subdomain2global).all())
                        assert((subdomain2global == petsc_subdomain2global_petsc).all())
                    par_u  = ParVec_petsc4py(u, 1, par_n, par_N, par_nghost,
                                             petsc_subdomain2global_petsc, ghosts=petsc_ghosts,
                                             proteus2petsc_subdomain=proteus2petsc_subdomain,
                                             petsc2proteus_subdomain=petsc2proteus_subdomain)
                    par_r  = ParVec_petsc4py(r, 1, par_n, par_N, par_nghost,
                                             petsc_subdomain2global_petsc, ghosts=petsc_ghosts,
                                             proteus2petsc_subdomain=proteus2petsc_subdomain,
                                             petsc2proteus_subdomain=petsc2proteus_subdomain)
                    par_du = ParVec_petsc4py(du, 1, par_n, par_N, par_nghost,
                                             petsc_subdomain2global_petsc, ghosts=petsc_ghosts,
                                             proteus2petsc_subdomain=proteus2petsc_subdomain,
                                             petsc2proteus_subdomain=petsc2proteus_subdomain)
                    rowptr, colind, nzval = jacobian.getCSRrepresentation()
                    rowptr_petsc = rowptr.copy()
                    colind_petsc = colind.copy()
                    nzval[:] = np.arange(nzval.shape[0])
                    nzval_petsc = nzval.copy()
                    nzval_proteus2petsc=colind.copy()
                    nzval_petsc2proteus=colind.copy()
                    rowptr_petsc[0] = 0
                    comm.beginSequential()
                    for i in range(par_n+par_nghost):
                        start_proteus = rowptr[petsc2proteus_subdomain[i]]
                        end_proteus = rowptr[petsc2proteus_subdomain[i]+1]
                        nzrow =  end_proteus - start_proteus
                        rowptr_petsc[i+1] = rowptr_petsc[i] + nzrow
                        start_petsc = rowptr_petsc[i]
                        end_petsc = rowptr_petsc[i+1]
                        #print "proteus_colind", colind[start_proteus:end_proteus]
                        petsc_cols_i = proteus2petsc_subdomain[colind[start_proteus:end_proteus]]
                        j_sorted = petsc_cols_i.argsort()
                        colind_petsc[start_petsc:end_petsc] = petsc_cols_i[j_sorted]
                        nzval_petsc[start_petsc:end_petsc] = nzval[start_proteus:end_proteus][j_sorted]
                        for j_petsc, j_proteus in zip(np.arange(start_petsc,end_petsc),
                                                      np.arange(start_proteus,end_proteus)[j_sorted]):
                            nzval_petsc2proteus[j_petsc] = j_proteus
                            nzval_proteus2petsc[j_proteus] = j_petsc
                    assert((nzval_petsc[nzval_proteus2petsc] == nzval).all())
                    assert((nzval[nzval_petsc2proteus] == nzval_petsc).all())
                    comm.endSequential()
                    assert(nzval_petsc.shape[0] == colind_petsc.shape[0] == rowptr_petsc[-1] - rowptr_petsc[0])
                    petsc_a = {}
                    proteus_a = {}
                    for i in range(transport.dim):
                        for j,k in zip(colind[rowptr[i]:rowptr[i+1]],list(range(rowptr[i],rowptr[i+1]))):
                            nzval[k] = i*transport.dim+j
                            proteus_a[i,j] = nzval[k]
                            petsc_a[proteus2petsc_subdomain[i],proteus2petsc_subdomain[j]] = nzval[k]
                    for i in range(transport.dim):
                        for j,k in zip(colind_petsc[rowptr_petsc[i]:rowptr_petsc[i+1]],list(range(rowptr_petsc[i],rowptr_petsc[i+1]))):
                            nzval_petsc[k] = petsc_a[i,j]
                    assert((nzval_petsc[nzval_proteus2petsc] == nzval).all())
                    assert((nzval[nzval_petsc2proteus] == nzval_petsc).all())
                    assert (all(proteus_a[(petsc2proteus_subdomain[k[0]],petsc2proteus_subdomain[k[1]])] == v for k,v in list(petsc_a.items())))
                    assert (all(petsc_a[(proteus2petsc_subdomain[k[0]],proteus2petsc_subdomain[k[1]])] == v for k,v in list(proteus_a.items())))
                    transport.nzval_petsc = nzval_petsc
                    transport.colind_petsc = colind_petsc
                    transport.rowptr_petsc = rowptr_petsc
                    petsc_jacobian = SparseMat(transport.dim,transport.dim,nzval_petsc.shape[0], nzval_petsc, colind_petsc, rowptr_petsc)
                    transport.petsc_jacobian = petsc_jacobian#petsc_jacobian = jacobian
                    if  comm.size() == 1:
                        assert (nzval_petsc == nzval).all()
                        assert (colind_petsc == colind).all()
                        assert (rowptr_petsc == rowptr).all()
                    assert(colind.max() <= par_n+par_nghost)
                    assert(colind_petsc.max() <= par_n + par_nghost)
                    try:
                        transport.par_info.par_bs = 1
                        transport.par_info.par_n = par_n
                        transport.par_info.par_n_lst = par_n_list
                        transport.par_info.par_N = par_N
                        transport.par_info.par_nghost = par_nghost
                        transport.par_info.par_nghost_lst = par_nghost_list
                        transport.par_info.petsc_subdomain2global_petsc = petsc_subdomain2global_petsc
                        transport.par_info.proteus2petsc_subdomain = proteus2petsc_subdomain
                        transport.par_info.petsc2proteus_subdomain = petsc2proteus_subdomain
                        transport.par_info.subdomain2global = subdomain2global
                        transport.par_info.dim = transport.dim
                        transport.par_info.nzval_proteus2petsc = nzval_proteus2petsc
                        transport.par_info.mixed = mixed
                    except AttributeError:
                        logEvent("Transport class has no ParInfo_petsc4py class to store parallel data.",level=4)
                    par_jacobian = ParMat_petsc4py(petsc_jacobian,1,par_n,par_N,par_nghost,
                                                   petsc_subdomain2global_petsc,pde=transport,
                                                   proteus_jacobian=jacobian, nzval_proteus2petsc=nzval_proteus2petsc)
                else:
                    transport.owned_local = numpy.arange(par_n*par_bs)
                    par_nghost = trialSpaceDict[0].dofMap.nDOF_subdomain - par_n
                    subdomain2global = trialSpaceDict[0].dofMap.subdomain2global
                    logEvent("Allocating ghosted parallel vectors on rank %i" % comm.rank(),level=2)
                    par_u = ParVec_petsc4py(u,par_bs,par_n,par_N,par_nghost,subdomain2global)
                    par_r = ParVec_petsc4py(r,par_bs,par_n,par_N,par_nghost,subdomain2global)
                    logEvent("Allocating ghosted parallel vectors on rank %i" % comm.rank(),level=2)
                    #du is ghosted like u and r. KSP never needed it to be --
                    #it only ever reads the owned range -- but proteus's own
                    #smoothers compute residuals with the subdomain matrix,
                    #whose owned rows couple to ghost columns, so they have to
                    #be able to refresh du's ghost entries after each update.
                    par_du = ParVec_petsc4py(du,par_bs,par_n,par_N,par_nghost,subdomain2global)
                    logEvent("Allocating matrix on rank %i" % comm.rank(),level=2)
                    
                    try:
                        transport.par_info.par_bs = par_bs
                        transport.par_info.par_n = par_n
                        transport.par_info.par_N = par_N
                        transport.par_info.par_nghost = par_nghost
                        transport.par_info.subdomain2global = subdomain2global
                        transport.par_info.dim = transport.dim
                        transport.par_info.mixed = mixed
                    except AttributeError:
                        logEvent("Transport class has no ParInfo_petsc4py class to store parallel data.",level=4)
                    par_jacobian = ParMat_petsc4py(jacobian,par_bs,par_n,par_N,par_nghost,subdomain2global,pde=transport)

            elif  (options.multilevelLinearSolver == KSP_petsc4py or
                   options.levelLinearSolver == KSP_petsc4py):
                assert trialSpaceDict[0].dofMap.subdomain2global is not None, "need trivial subdomain2global in dofMap for running PETSc"
                assert trialSpaceDict[0].dofMap.max_dof_neighbors is not None, "need max_dof_neighbors in dofMap for running PETSc"
                par_N = par_n =  trialSpaceDict[0].dofMap.nDOF_all_processes
                mixed = False
                for ts in list(trialSpaceDict.values()):
                    if ts.dofMap.nDOF_all_processes != par_N:
                        mixed=True
                # Serial periodic conditions on kept strong Dirichlet DOFs
                # merge each periodic pair into one unknown
                # (DOFBoundaryConditions), so the system has
                # transport.nFreeDOF_global[ci] unknowns per component, not
                # the space's DOF count. In serial its numbering is the
                # identity, whatever the component layout, so it goes
                # through the per-component (mixed) path with block size 1.
                # Where nothing was merged, nothing changes.
                serialPeriodic = any(transport.nFreeDOF_global[ci] != ts.dofMap.nDOF_all_processes
                                     for ci,ts in trialSpaceDict.items())
                if serialPeriodic:
                    mixed = True
                par_nghost = 0
                subdomain2global = trialSpaceDict[0].dofMap.subdomain2global
                max_dof_neighbors= trialSpaceDict[0].dofMap.max_dof_neighbors
                logEvent("Allocating ghosted parallel vectors on rank %i" % comm.rank(),level=2)
                
                if mixed:
                    if serialPeriodic:
                        par_N = par_n = sum(transport.nFreeDOF_global)
                        subdomain2global = numpy.arange(par_n, dtype=cmeshTools.INDEX_DTYPE)
                    else:
                        par_N = par_n = sum([ts.dofMap.nDOF_all_processes for ts in list(trialSpaceDict.values())])
                        subdomain2global = numpy.hstack([offset+ts.dofMap.subdomain2global for
                                                         offset,ts in zip(transport.offset,list(trialSpaceDict.values()))])
                    transport.owned_local = numpy.arange(par_n)
                    par_u = ParVec_petsc4py(u,1,par_n,par_N,par_nghost,subdomain2global[:par_n])
                    par_r = ParVec_petsc4py(r,1,par_n,par_N,par_nghost,subdomain2global[:par_n])
                    logEvent("Allocating un-ghosted parallel vectors on rank %i" % comm.rank(),level=2)
                    par_du = ParVec_petsc4py(du,1,par_n,par_N)
                    logEvent("Allocating matrix on rank %i" % comm.rank(),level=2)
                    par_jacobian = ParMat_petsc4py(jacobian,1,par_n,par_N,par_nghost,subdomain2global,pde=transport)
                    try:
                        transport.par_info.par_bs = par_bs
                        transport.par_info.mixed = mixed
                        transport.par_info.par_n = par_n
                        transport.par_info.par_N = par_N
                        transport.par_info.par_nghost = par_nghost
                        transport.par_info.subdomain2global = subdomain2global
                        transport.par_info.dim = transport.dim
                    except AttributeError:
                        logEvent("Transport class has no ParInfo_petsc4py class to store parallel data.",level=4)
                else:
                    transport.owned_local = numpy.arange(par_n*par_bs)
                    subdomain2global = trialSpaceDict[0].dofMap.subdomain2global
                    max_dof_neighbors= trialSpaceDict[0].dofMap.max_dof_neighbors
                    par_u = ParVec_petsc4py(u,par_bs,par_n,par_N,par_nghost,subdomain2global[:par_n])
                    par_r = ParVec_petsc4py(r,par_bs,par_n,par_N,par_nghost,subdomain2global[:par_n])
                    logEvent("Allocating un-ghosted parallel vectors on rank %i" % comm.rank(),level=2)
                    par_du = ParVec_petsc4py(du,par_bs,par_n,par_N)
                    logEvent("Allocating matrix on rank %i" % comm.rank(),level=2)
                    par_jacobian = ParMat_petsc4py(jacobian,par_bs,par_n,par_N,par_nghost,subdomain2global,pde=transport)
                    try:
                        transport.par_info.par_bs = par_bs
                        transport.par_info.mixed = mixed
                        transport.par_info.par_n = par_n
                        transport.par_info.par_N = par_N
                        transport.par_info.par_nghost = par_nghost
                        transport.par_info.subdomain2global = subdomain2global
                        transport.par_info.dim = transport.dim
                    except AttributeError:
                        logEvent("Transport class has no ParInfo_petsc4py class to store parallel data.",level=4)
            else:
                transport.owned_local = numpy.arange(transport.dim)                
                par_u = None
                par_r = None
                par_du = None
                par_jacobian = None
            
            # parallel periodic numbering renumbers the subdomain dofs (merged pairs): left as it was
            if par_r is not None and not (options.periodicDirichletConditions and getattr(options, 'parallelPeriodic', False)):
                attachStrongDirichletAuthority(transport, par_r, par_jacobian)
            self.par_uList.append(par_u)
            self.par_duList.append(par_du)
            self.par_rList.append(par_r)
            self.par_jacobianList.append(par_jacobian)
            logEvent(memory("global Jacobian and vectors","MultilevelTransport"),level=4)
            ##transport.mesh.nodeArray[:,2] is partitioned bathymetry
            ## transport.coefficients.bathymetry is global bathymetry
        logEvent("Building Mesh Transfers",level=2)
        
        MultilevelProjectionOperatorType = MultilevelProjectionOperators
        if self.parallelTransfers:
            #only the container: the distributed operators are installed into it
            self.meshTransfers = MultilevelProjectionOperatorType(
                mlMesh,
                self.trialSpaceDictList,
                self.offsetListList,
                self.strideListList,
                self.bcDictList,
                build=False)
            self.useParallelTransfers(mlMesh)
        else:
            #a hierarchy generated in pieces has no global levels; on one rank its
            #pieces are the whole levels, related by subdomainHierarchy
            self.meshTransfers = MultilevelProjectionOperatorType(
                mlMesh.subdomainHierarchy() if getattr(mlMesh,'generated',False) else mlMesh,
                self.trialSpaceDictList,
                self.offsetListList,
                self.strideListList,
                self.bcDictList)
        logEvent(memory("mesh transfers","MultilevelTransport"),level=4)
        #mwf hack keep reference to mlMesh in Transport ctor for now
        self.mlMeshSave = mlMesh
        
    def useParallelTransfers(self,mlMesh):
        """Replace the transfer operators with distributed ones where possible.

        MultilevelProjectionOperators builds global matrices, which cannot act
        on a subdomain's vectors. For a nodally nested hierarchy of spaces whose
        dofs are the mesh vertices the operators follow from the node parents
        alone, so each rank can assemble the rows it owns and PETSc applies
        them -- see LinearAlgebraTools.nodalProlongationMat. The parents have to
        be read in the partitioned node numbering, which is what
        partitionedNodeParents does.

        There are two families of operators and they act on different vectors.
        prolongList/restrictList act on the *free* unknowns of the stacked
        system, which is what the linear multigrid solvers see. The
        *_bcListDict operators act on each component's full dof array,
        Dirichlet dofs included, which is what the nonlinear nested iteration
        and the post-processing see. Parallel runs require weak Dirichlet
        conditions (OneLevelTransport asserts it), so no dof is ever eliminated
        and the two families are the same matrix up to how the vectors are laid
        out -- which is why one prolongation Mat serves both. Strong
        elimination of Dirichlet dofs is a serial-only variant: the free
        numbering stops coinciding with the vertex numbering, and
        nodalTransfersApply declines it.

        Anything that does not qualify -- higher-order, modal, discontinuous
        and non-conforming spaces, hierarchies that are not nodally nested --
        takes useGeneralParallelTransfers.
        """
        #the nodal route reads node parents through the partitioner's node
        #renumbering, which only node partitioning records
        if hasattr(mlMesh,'hasPartitionedNumbering'):
            permuted = mlMesh.hasPartitionedNumbering()
        else:
            permuted = all(getattr(mesh,'nodeNumbering_global_old2new',None) is not None
                           for mesh in mlMesh.meshList)
        if not permuted or not FemTools.nodalTransfersApply(mlMesh,
                                                            self.trialSpaceDictList,
                                                            self.bcDictList):
            self.useGeneralParallelTransfers(mlMesh)
            return
        if not getattr(mlMesh,'generated',False):
            mlMesh.vertexInterpolation()
        nc = len(self.trialSpaceDictList[0])
        cache = {}
        def levelMats(l):
            if l not in cache:
                interpolation = mlMesh.partitionedVertexInterpolation(l)
                coarseOffsets = np.asarray(mlMesh.meshList[l].nodeOffsets_subdomain_owned)
                fineOffsets = np.asarray(mlMesh.meshList[l+1].nodeOffsets_subdomain_owned)
                prolongMat = vertexProlongationMat(*interpolation,
                                                   coarseNodeOffsets=coarseOffsets,
                                                   fineNodeOffsets=fineOffsets)
                injectionMat = vertexProlongationMat(*interpolation,
                                                     coarseNodeOffsets=coarseOffsets,
                                                     fineNodeOffsets=fineOffsets,
                                                     inheritedOnly=True)
                rowSums = nodalRestrictionScaling(prolongMat)
                stackedMat = prolongMat if nc == 1 else vertexProlongationMat(
                    *interpolation,coarseNodeOffsets=coarseOffsets,
                    fineNodeOffsets=fineOffsets,blockSize=nc)
                cache[l] = (prolongMat,injectionMat,rowSums,stackedMat)
            return cache[l]
        self.installParallelTransfers(mlMesh,
                                      lambda l,cj: levelMats(l)[:3],
                                      lambda l: levelMats(l)[3])

    def useGeneralParallelTransfers(self,mlMesh):
        """Distributed transfers for any space, built from each rank's own subdomain.

        The nodal route needs a dof to be a vertex. Otherwise the operators come
        from the geometric construction in MultilevelProjectionOperators --
        coarse basis functions evaluated at the fine interpolation points --
        run on the *subdomain* spaces over the subdomain parent/child relations
        (MultilevelMesh.subdomainHierarchy). That gives each rank a prolongation
        between its subdomain dofs, complete on the rows it owns, because a finer
        level inherits the coarser one's partition (MultilevelMesh.partitionLevels)
        and so every fine element touching an owned dof has its parent here.
        The owned rows are translated to global dof numbers and assembled into a
        distributed Mat (LinearAlgebraTools.subdomainProlongationMat); nothing is
        gathered, and no space or operator is built on the global mesh.

        An owned row with no entries means the partitions were not
        level-consistent, and raises rather than building a wrong operator.
        """
        nc = len(self.trialSpaceDictList[0])
        hierarchy = mlMesh.subdomainHierarchy()
        ops = FemTools.MultilevelProjectionOperators(
            hierarchy,self.trialSpaceDictList,self.offsetListList,self.strideListList,
            self.bcDictList)
        def csr(sparseMat):
            rowptr,colind,nzval = sparseMat.getCSRrepresentation()
            return (np.asarray(rowptr),np.asarray(colind),np.asarray(nzval))
        def maps(l,cj):
            fine = self.trialSpaceDictList[l+1][cj].dofMap
            coarse = self.trialSpaceDictList[l][cj].dofMap
            return (fine.subdomain2global,coarse.subdomain2global,
                    np.asarray(fine.dof_offsets_subdomain_owned),
                    np.asarray(coarse.dof_offsets_subdomain_owned))
        cache = {}
        def mats(l,cj):
            if (l,cj) not in cache:
                P = csr(ops.prolong_bcListDict[cj][l+1])
                build = lambda c,**kw: subdomainProlongationMat(*c,*maps(l,cj),**kw)
                prolongMat = build(P)
                #entries equal to one are what the injection keeps, as in the
                #global operators' interp_bc
                injectionMat = build(filterCSR(*P,lambda w: np.abs(w-1.0) < 1.0e-8),
                                     allowEmptyRows=True)
                #restricted solutions are divided by the row sums of |P|
                absMat = build(filterCSR(*P,lambda w: np.ones(w.shape,dtype=bool),np.abs))
                cache[(l,cj)] = (prolongMat,injectionMat,nodalRestrictionScaling(absMat))
            return cache[(l,cj)]
        def stackedMat(l):
            if nc == 1:
                return mats(l,0)[0]
            fineS2G,coarseS2G,fineOff,coarseOff = maps(l,0)
            return subdomainProlongationMat(*csr(ops.prolong_bcListDict[0][l+1]),
                                            fineS2G,coarseS2G,fineOff,coarseOff,blockSize=nc)
        self.installParallelTransfers(mlMesh,mats,stackedMat,sameSpaces=len(
            set((type(ts.dofMap).__name__) for ts in self.trialSpaceDictList[0].values())) == 1)

    def installParallelTransfers(self,mlMesh,mats,stackedMat,sameSpaces=True):
        """Put distributed operators into the mesh transfers.

        mats(l,cj) gives (prolongation, injection, restriction row sums) from
        level l to l+1 for component cj, as distributed Mats; stackedMat(l)
        gives the prolongation of the whole interleaved system.
        """
        nc = len(self.trialSpaceDictList[0])
        #The stacked system's operator is P (x) I only if its vectors interleave
        #the components node by node, so that component c of node i is entry
        #nc*i + c on every level. Nodal spaces give that whenever weak Dirichlet
        #conditions apply, which parallel runs require, but it is a property of
        #how Transport laid the system out, so read it back rather than assume.
        stacked = sameSpaces and all(
            list(self.offsetListList[l]) == list(range(nc)) and
            list(self.strideListList[l]) == [nc]*nc and
            getattr(self.par_uList[l],'bs',1) == nc
            for l in range(len(mlMesh.meshList)))
        if not stacked:
            logEvent("free-unknown transfers stay global: the system's "
                     "components are not interleaved node by node, which the "
                     "distributed stacked operator assumes (the per-component "
                     "ones are built)",level=2)
        mt = self.meshTransfers
        prolongList = [[]]
        restrictList = [[]]
        restrictSumList = [[]]
        prolong_bc = dict((cj,[[]]) for cj in range(nc))
        restrict_bc = dict((cj,[[]]) for cj in range(nc))
        scaled_restrict_bc = dict((cj,[[]]) for cj in range(nc))
        interp_bc = dict((cj,[[]]) for cj in range(nc))
        restrict_bcSum = dict((cj,[[]]) for cj in range(nc))
        for l in range(len(mlMesh.meshList)-1):
            if stacked:
                stackedP = stackedMat(l)
                prolongList.append(ParTransferOperator(
                    stackedP,self.par_uList[l],self.par_uList[l+1]))
                restrictList.append(ParTransposeTransferOperator(
                    stackedP,self.par_uList[l],self.par_uList[l+1]))
                #FAS divides a restricted *solution* by these row sums (the paper's
                #I-tilde); ghosts included, since the next residual reads them
                coarseU = self.par_uList[l]
                rowSums = np.zeros(coarseU.proteus_array.size)
                rowSums[:coarseU.dim_proc] = nodalRestrictionScaling(stackedP).getArray()
                coarseU.duplicate_with_array(rowSums).scatter_forward_insert()
                restrictSumList.append(rowSums)
            for cj in range(nc):
                prolongMat,injectionMat,rowSums = mats(l,cj)
                coarse = self.uDictList[l][cj].par_dof
                fine = self.uDictList[l+1][cj].par_dof
                prolong_bc[cj].append(ParTransferOperator(prolongMat,coarse,fine))
                restrict_bc[cj].append(ParTransposeTransferOperator(prolongMat,coarse,fine))
                scaled_restrict_bc[cj].append(
                    ParScaledTransposeTransferOperator(prolongMat,coarse,fine,rowSums))
                #the injection's transpose, so it lands coarse <- fine like the
                #restrictions; entries are 1 on the nodes the refinement inherited
                interp_bc[cj].append(ParTransposeTransferOperator(injectionMat,coarse,fine))
                #NLNI divides a restricted solution by these, ghosts included
                sums = np.zeros(coarse.proteus_array.size)
                sums[:coarse.dim_proc] = rowSums.getArray()
                coarse.duplicate_with_array(sums).scatter_forward_insert()
                restrict_bcSum[cj].append(sums)
        if stacked:
            mt.prolongList = prolongList
            mt.restrictList = restrictList
            mt.restrictSumList = restrictSumList
        mt.prolong_bcListDict = prolong_bc
        mt.restrict_bcListDict = restrict_bc
        mt.scaled_restrict_bcListDict = scaled_restrict_bc
        mt.interp_bcListDict = interp_bc
        mt.restrict_bcSumListDict = restrict_bcSum
        logEvent("mesh transfers assembled as distributed matrices over %i levels"
                 % (len(mlMesh.meshList),),level=2)

    def setInitialConditions(self,getInitialConditionsDict,T=0.0):
        logEvent("Setting initial conditions on model "+self.name)
        self.t=T
        for m,u in zip(self.levelModelList,self.uList):
            #separate functions to maintain backwards compatibility
            try:    
                m.setInitialConditionsTPF(getInitialConditionsDict.__dict__,m.var2idxDict,T)
            except:
                m.setInitialConditions(getInitialConditionsDict,T)

            m.setFreeDOF(u)
    def setInitialConditionsFromDofs(self,T):
        self.t=T
        #assumes free dofs already determined
        for m in self.levelModelList:
            m.setInitialConditionsFromDofs(m.coefficients,T)
    def calculateAuxiliaryQuantitiesAfterStep(self):
        for m in self.levelModelList:
            m.calculateAuxiliaryQuantitiesAfterStep()
    def attachModels(self,modelList):
        
        for l,m in enumerate(self.levelModelList):
            models=[]
            for mOther in modelList:
                models.append(mOther.levelModelList[l])
            m.coefficients.attachModels(models)

    def viewJacobian(self, file_prefix='./dump_', b=None):
        """
        Save parallel Jacobian list and (optionally) right-hand side b to disk if they
        possess save methods, otherwise, do nothing.
        """

        for idx, par_jacobian in enumerate(self.par_jacobianList):
            if hasattr(par_jacobian, 'save'):
                filename = file_prefix + 'par_j' + '_' + str(idx)
                logEvent('Saving Parallel Jacobian to %s' % filename)
                par_jacobian.save(filename)
        if b and hasattr(b, 'save'):
            filename = file_prefix + 'b'
            logEvent('Saving right-hand-side to %s' % filename)
            b.save(filename)

## @}
