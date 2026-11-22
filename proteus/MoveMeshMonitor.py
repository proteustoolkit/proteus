import numpy as np
import copy
from proteus.Profiling import logEvent
from proteus import TransportCoefficients
from proteus.mprans import MeshSmoothing as ms
from proteus.mprans import cMoveMeshMonitor as cmm

class Coefficients(TransportCoefficients.PoissonEquationCoefficients):
    """
    Coefficients for deforming the mesh according to an area function

    Parameters
    ----------
    func: function
        function of x defining element area
    he_min: double
        minimum characteristic element size
    he_max: double
        maximum characteristic element size
    ME_MODEL: int
        index of this model
    LS_MODEL: int
        index of LS model for retrieving phi values and refine around free
        surface
    nd: int
        number of dimensions
    fixedNodeMaterialTypes: int[:]
        array of length of all domain node material types, with indexes
        corresponding to the material type set as: 0 -> not fixed, 1 -> fixed.
        e.g. if nodes of material type 3 should be fixed:
        fixedNodeMaterialTypes[3]=1   (!) must be integers
    fixedElementMaterialTypes: int[:]
        array of length of all domain element material types, with indexes
        corresponding to the material type set as: 0 -> not fixed, 1 -> fixed.
        all nodes that have at least one element with fixed material type around
        them will be fixed
    noNodeVelocityNodeMaterialTypes: int[:]
        array of length of all domain node material types, with indexes
        corresponding to the material type set as: 0 -> not fixed, 1 -> fixed.
        e.g. if nodes of material type 3 should not have a node velocity:
        fixedNodeMaterialTypes[3]=1   (!) must be integers
    nSmoothOut: int
        number of smoothing steps after finishing pseudo time stepping
    nSmoothIn: int
        number of smoothing steps after finishing pseudo time stepping
    epsFact_density: double
        epsFact*he_min where refinement is the same
    epsTimeStep: double
        pseudo time step size (0 < epsTimeStep <= 1)
    grading: double
        grading of mesh with distance function. 1.-> no grading, 1.1->10% increase
        (!) if using grading, func must be a distance function
        otherwise func is an area function
    grading_type: int
        type of grading:
        0 -> no grading
        1 -> hyperbolic
        2 -> log
    resetNodeVelocityArray: bool
        reset nodeVelocityArray to zero during prestep if true, otherwise adds
        up velocity to the existing node velocity. Can be used as False if this
        model is used in conjunction with another moving mesh model.
    """
    def __init__(self,
                 func,
                 he_min,
                 he_max,
                 ME_MODEL,
                 LS_MODEL=None,
                 useLS=True,
                 nd=2,
                 fixedNodeMaterialTypes=None,
                 fixedElementMaterialTypes=None,
                 noNodeVelocityNodeMaterialTypes=None,
                 nSmoothOut=0,
                 nSmoothIn=0,
                 epsFact_density=0,
                 epsTimeStep=1.,
                 ntimes_solved=1,
                 grading=1.1,
                 grading_type=0,
                 resetNodeVelocityArray=True,
                 scale_with_nd=False,
                 do_firstStep=True,
                 nullSpace="ConstantNullSpace",
                 initialize=True):
        self.nullSpace = nullSpace
        self.myfunc = func
        self.LS_MODEL = LS_MODEL
        self.useLS = useLS
        self.ME_MODEL = ME_MODEL
        self.he_min = he_min
        self.he_max = he_max
        self.grading = grading
        self.grading_type = grading_type
        self.C = 1.  # scaling coefficient for f (computed in preStep)
        self.integral_area = 1.
        self.nd = nd
        self.epsTimeStep = epsTimeStep
        self.epsFact_density = epsFact_density
        self.fixedNodeMaterialTypes = fixedNodeMaterialTypes
        self.fixedElementMaterialTypes = fixedElementMaterialTypes
        self.nSmoothOut = nSmoothOut
        self.nSmoothIn = nSmoothIn
        self.dt_last = None
        self.u_phi = None
        self.nt = 0
        self.t = 0.
        self.t_last = 0.
        self.poststepdone = False
        self.noNodeVelocityNodeMaterialTypes = noNodeVelocityNodeMaterialTypes
        self.ntimes_solved = ntimes_solved
        self.resetNodeVelocityArray = resetNodeVelocityArray
        self.ntimes_i = 0
        self.do_firstStep = True
        self.scale_with_nd = scale_with_nd
        if initialize:
            self.initialize()

    def initialize(self):
        assert 0 < self.epsTimeStep <= 1, 'epsTimeStep must be between 0. and 1.'
        if self.useLS is False:
            self.LS_MODEL = None
        if self.scale_with_nd:
            if self.nd == 2:
                self.fFactor = lambda f: f**2
            elif self.nd == 3:
                self.fFactor = lambda f: f**3
        else:
            self.fFactor = lambda f: f
        aOfX = [lambda x: np.array([[1., 0.], [0., 1.]])]
        fOfX = [self.uOfX]  # scaled function reciprocal
        TransportCoefficients.PoissonEquationCoefficients.__init__(self, aOfX, fOfX)


    def initializeMesh(self, mesh):
        self.mesh = mesh
        self.areas_nodes = np.zeros(self.mesh.elementNodesArray.shape[0])
        self.areas = np.zeros(self.mesh.elementNodesArray.shape[0])
        # get Normals
        if self.nd == 2:
            # triangle
            self.mesh.elementBoundaryNormalsArray = np.zeros((len(self.mesh.elementBoundariesArray), 3, 3))
            self.mesh.elementBoundaryNormalsArray0 = np.zeros((len(self.mesh.elementBoundariesArray), 3, 3))
        elif self.nd == 3:
            # tetra
            self.mesh.elementBoundaryNormalsArray = np.zeros((len(self.mesh.elementBoundariesArray), 4, 3))
            self.mesh.elementBoundaryNormalsArray0 = np.zeros((len(self.mesh.elementBoundariesArray), 4, 3))
        self.mesh.elementBoundaryBarycentersArray0 = np.zeros(self.mesh.elementBoundaryBarycentersArray.shape)
        self.mesh.elementBarycentersArray0 = np.zeros(self.mesh.elementBarycentersArray.shape)
        if self.nd == 2:
            ms.updateElementBoundaryNormalsTriangle(self.mesh.elementBoundaryNormalsArray,
                                                    self.mesh.nodeArray,
                                                    self.mesh.elementBoundariesArray,
                                                    self.mesh.elementBoundaryNodesArray,
                                                    self.mesh.elementBoundaryBarycentersArray,
                                                    self.mesh.elementBarycentersArray,
                                                    self.mesh.nElements_global)
            # ms.updateElementBoundaryNormalsTriangle(elementBoundaryNormalsArray=self.mesh.elementBoundaryNormalsArray,
            #                                         nodeArray=self.mesh.nodeArray,
            #                                         elementBoundariesArray=self.mesh.elementBoundariesArray,
            #                                         elementBoundaryNodesArray=self.mesh.elementBoundaryNodesArray,
            #                                         elementBoundaryBarycentersArray=self.mesh.elementBoundaryBarycentersArray,
            #                                         elementBarycentersArray=self.mesh.elementBarycentersArray,
            #                                         nElements=self.mesh.nElements_global)
        elif self.nd == 3:
            ms.updateElementBoundaryNormalsTetra(elementBoundaryNormalsArray=self.mesh.elementBoundaryNormalsArray,
                                                 nodeArray=self.mesh.nodeArray,
                                                 elementBoundariesArray=self.mesh.elementBoundariesArray,
                                                 elementBoundaryNodesArray=self.mesh.elementBoundaryNodesArray,
                                                 elementBoundaryBarycentersArray=self.mesh.elementBoundaryBarycentersArray,
                                                 elementBarycentersArray=self.mesh.elementBarycentersArray,
                                                 nElements=self.mesh.nElements_global)
        # nodes to keep fixed
        self.mesh.fixedNodesBoolArray = np.zeros(len(self.mesh.nodeArray), dtype=np.int32)
        # find corner nodes
        self.mesh.cornerNodes = ms.getCornerNodesTriangle(nodeArray=self.mesh.nodeArray,
                                                          nodeStarArray=self.mesh.nodeStarArray,
                                                          nodeStarOffsets=self.mesh.nodeStarOffsets,
                                                          nodeMaterialTypes=self.mesh.nodeMaterialTypes,
                                                          nNodes=self.mesh.nNodes_owned)
        for cn in self.mesh.cornerNodes:
            self.mesh.fixedNodesBoolArray[cn] = 1
        if self.fixedNodeMaterialTypes is not None:
            for node in range(len(self.mesh.nodeMaterialTypes)):
                if self.fixedNodeMaterialTypes[self.mesh.nodeMaterialTypes[node]] == 1:
                    self.mesh.fixedNodesBoolArray[node] = 1
        if self.fixedElementMaterialTypes is not None:
            # if node has a fixed element material type around them, fix the node
            for node in range(len(self.mesh.nodeArray)):
                for eOffset in range(self.mesh.nodeElementOffsets[node], self.mesh.nodeElementOffsets[node+1]):
                    eN = self.mesh.nodeElementsArray[eOffset]
                    if self.fixedElementMaterialTypes[self.mesh.elementMaterialTypes[eN]] == 1:
                        self.mesh.fixedNodesBoolArray[node] = 1
        self.mesh.nodeArray0 = np.zeros(self.mesh.nodeArray.shape)

    def attachModels(self,modelList):
        self.model = modelList[self.ME_MODEL]
        self.model.bdyNullSpace = True
        self.q = self.model.q
        self.model.mesh_array0 = copy.deepcopy(self.mesh.nodeArray)
        self.PHI = np.zeros_like(self.model.mesh.nodeArray)
        self.grads = np.zeros((len(self.model.u[0].dof),2))
        self.areas_nodes = np.zeros(len(self.mesh.nodeArray))
        self.areas = np.zeros(self.mesh.elementNodesArray.shape[0])
        self.nearest_nodes = np.array([i for i in range(len(self.mesh.nodeArray))])
        self.eN_phi = np.array([None for i in range(len(self.mesh.nodeArray))])  # element containing displaced phi (guess)
        self.nearest_nodes0 = np.array([i for i in range(len(self.mesh.nodeArray))])
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape,'d')
        self.mesh.nodeDisplacementArray = np.zeros(self.mesh.nodeArray.shape,'d')
        self.uOfXTatNodes = np.zeros(len(self.mesh.nodeArray))
        self.uOfXTatQuadrature = np.zeros((self.model.q['x'].shape[0], self.model.q['x'].shape[1]))
        if self.LS_MODEL is not None:
            self.model_ls = modelList[self.LS_MODEL]
            self.u_phi = self.model_ls.u[0].dof
            self.q_phi = self.model_ls.q[('u', 0)]
        else:
            self.u_phi = np.zeros(self.model.u[0].dof.shape)+1e12
        self.cCoefficients = cmm.cCoefficients()
        self.cCoefficients.attachPyCoefficients(self)

    def preStep(self, t, firstStep=False):
        logEvent("Updating area function scaling coefficient and quadrature",
                 level=3)
        # careful for nodeArray0 if mesh was moved with another moving mesh model
        self.mesh.nodeArray0[:] = self.mesh.nodeArray[:]
        self.mesh.elementBoundaryNormalsArray0[:] = self.mesh.elementBoundaryNormalsArray[:]
        self.evaluateFunAtQuadraturePoints()
        self.cCoefficients.preStep()
        self.C = self.cCoefficients.C
        self.model.areas = self.areas
        if self.ntimes_i == 0:
            self.mesh.nodeArray0[:] = self.mesh.nodeArray[:]
        logEvent("Finished updating area function",
                 level=3)
        if self.t != self.model.t_mesh:
            self.nt += 1
            self.t_last = self.t
            self.t = self.model.t_mesh
            self.poststepdone = False
            if self.resetNodeVelocityArray is True and self.ntimes_i == 0:
                self.mesh.nodeVelocityArray[:] = 0.

    def postStep(self, t, firstStep=False):
        # if self.t > 0 or self.do_firstStep is True and firstStep is True:
        # gradient recovery
        logEvent("Gradient recovery at mesh nodes", level=3)
        # self.grads[:] = ms.pyVectorRecoveryAtNodes(vectors=self.model.q[('grad(u)', 0)][:,0,:],
        #                                            nodeElementsArray=self.mesh.nodeElementsArray,
        #                                            nodeElementOffsets=self.mesh.nodeElementOffsets,
        #                                            nd=self.nd)
        self.grads[:] = cmm.gradientRecoveryAtNodes(grads=self.model.q[('grad(u)', 0)],
                                                    nodeElementsArray=self.mesh.nodeElementsArray,
                                                    nodeElementOffsets=self.mesh.nodeElementOffsets,
                                                    nd=self.nd)
        # print(self.model.q[('grad(u)', 0)])
        # print(self.grads)
        logEvent("Area recovery at mesh nodes", level=3)
        self.areas_nodes[:] = cmm.recoveryAtNodes(scalars=self.areas,
                                                  nodeElementsArray=self.mesh.nodeElementsArray,
                                                  nodeElementOffsets=self.mesh.nodeElementOffsets)
        nNodes_owned = self.mesh.nNodes_owned
        nNodes_global = self.mesh.nNodes_global
        nElements_owned = self.mesh.nElements_owned
        nElements_global = self.mesh.nElements_global
        nodeNumbering_subdomain2global = self.mesh.globalMesh.nodeNumbering_subdomain2global
        nodeOffsets_subdomain_owned = self.mesh.globalMesh.nodeOffsets_subdomain_owned
        ms.getNonOwnedNodeValues(args_=self.grads,
                                 nNodes_owned=nNodes_owned,
                                 nNodes_global=nNodes_global,
                                 nodeNumbering_subdomain2global=nodeNumbering_subdomain2global,
                                 nodeOffsets_subdomain_owned=nodeOffsets_subdomain_owned)
        ms.getNonOwnedNodeValues(args_=self.areas_nodes,
                                 nNodes_owned=nNodes_owned,
                                 nNodes_global=nNodes_global,
                                 nodeNumbering_subdomain2global=nodeNumbering_subdomain2global,
                                 nodeOffsets_subdomain_owned=nodeOffsets_subdomain_owned)
        # ms.getNonOwnedNodeValues(self.u_phi,
        #                          nNodes_owned,
        #                          nNodes_global,
        #                          nodeNumbering_subdomain2global,
        #                          nodeOffsets_subdomain_owned)
        self.model.grads = self.grads
        self.model.areas_nodes = self.areas_nodes
        # evaluate f at nodes
        self.evaluateFunAtNodes()
        # gamma parameter
        f_over_g = self.uOfXTatNodes/self.areas_nodes
        f_over_g_max = max(f_over_g)
        f_over_g_min = min(f_over_g)
        self.gamma = f_over_g_max/f_over_g_min
        self.gamma0 = 10.  # user defined parameter
        self.na = np.log(self.gamma)/np.log(self.gamma0)
        nNodes_owned = self.mesh.nNodes_owned
        # if self.dt_last is not None or firstStep is False:
        # pseudo-time step
        self.PHI[:] = self.mesh.nodeArray[:]
        self.cCoefficients.pseudoTimeStepping(eps=self.epsTimeStep,
                                              xx=self.PHI)
        if self.dt_last is not None:
            dt = self.dt_last
        else:
            dt = self.model.timeIntegration.dt
        self.mesh.nodeVelocityArray[:] += (self.PHI[:]-self.mesh.nodeArray[:])/dt
        # self.model.mesh.nodeVelocityArray[:] = self.model.mesh.nodeDisplacementArray[:]/dt
        self.mesh.nodeArray[:] = self.PHI[:]
        self.nearest_nodes[:] = self.nearest_nodes0[:]
        self.eN_phi[:] = None
        # # tri hack: remove mesh velocity when dirichlet imposed on boundaries
        if self.noNodeVelocityNodeMaterialTypes is not None:
            for node in range(nNodes_global):
                if self.noNodeVelocityNodeMaterialTypes[self.mesh.nodeMaterialTypes[node]] == 1:
                    self.mesh.nodeVelocityArray[node, :] = 0.
        self.cCoefficients.postStep()
        # IMR_nodes = ms.getInverseMeanRatioTriangleNodes(nodeArray=self.mesh.nodeArray,
        #                                                 elementNodesArray=self.mesh.elementNodesArray,
        #                                                 nodeElementOffsets=self.mesh.nodeElementOffsets,
        #                                                 nodeElementsArray=self.mesh.nodeElementsArray,
        #                                                 el_average=False,
        #                                                 nElements=nElements_global,
        #                                                 nNodes=nNodes_global)
        # self.model.u[0].dof[:] = IMR_nodes
        self.dt_last = self.model.timeIntegration.dt
        self.poststepdone = True
        if self.ntimes_i == self.ntimes_solved-1:
            # reset for next time step
            self.ntimes_i = 0

    def uOfX(self, x, eN=None):
        """
        reaction term
        """
        # f_scaled = (self.myfunc(x)*self.C)
        # f_scaled_reciprocal = 1./f_scaled
        if eN is None:
            area = 1.
        else:
            area = self.areas[eN]
        f = 1./(self.evaluateFunAtX(x)*self.C)-1./area
        return f

    def dist_search(self, x, node, eN=-1):
        """
        search element containing containing coords x starting with a guessed
        nearest node

        Parameters
        ----------
        x: array_like
            coordinates
        node: int
            first guess of the nearest node
        """
        femSpace = self.model.u[0].femSpace
        checkedElements=[]
        patchBoundaryNodes=set()

        nearest_node = node
        min_dist = np.sqrt(np.sum((x-self.mesh.nodeArray[node])**2))
        found_node = False
        element = None
        xx = np.zeros(3)
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        x = xx
        if eN >= 0:
            xi = femSpace.elementMaps.getInverseValue(eN, x)
            # query whether xi lies within the reference element
            if femSpace.elementMaps.referenceElement.onElement(xi):
                element = eN
        if element < 0:
            while found_node is False:
                nearest_node0 = nearest_node
                for nOffset in range(femSpace.mesh.nodeStarOffsets[nearest_node0],
                                     femSpace.mesh.nodeStarOffsets[nearest_node0+1]):
                    node = self.mesh.nodeStarArray[nOffset]
                    node_coords = self.mesh.nodeArray[node]
                    dist = np.sqrt(np.sum((node_coords-x)**2))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = node
                if nearest_node0 == nearest_node:
                    found_node = True
            # now find element containing x around nearest_node
            for eOffset in range(self.mesh.nodeElementOffsets[nearest_node],
                                 self.mesh.nodeElementOffsets[nearest_node+1]):
                eN = self.mesh.nodeElementsArray[eOffset]
                checkedElements.append(eN)
                # union of set
                nodes = self.mesh.nodeStarArray[femSpace.mesh.nodeStarOffsets[nearest_node]:femSpace.mesh.nodeStarOffsets[nearest_node+1]]
                patchBoundaryNodes|=set(nodes)
                # evaluate the inverse map for element eN (global to local)
                xi = femSpace.elementMaps.getInverseValue(eN, x)
                # query whether xi lies within the reference element
                if femSpace.elementMaps.referenceElement.onElement(xi):
                    element = eN
            if element < 0:
                # extra loop if case coords is in neighbour element
                for node in patchBoundaryNodes:
                    for eOffset in range(self.mesh.nodeElementOffsets[node],
                                         self.mesh.nodeElementOffsets[node+1]):
                        eN = self.mesh.nodeElementsArray[eOffset]
            #            patchBoundaryNodes|=set(femSpace.mesh.elementNodesArray[eN])
                        if eN not in checkedElements:
                            checkedElements.append(eN)
                            # evaluate the inverse map for element eN
                            xi = femSpace.elementMaps.getInverseValue(eN, x)
                            # query whether xi lies within the reference element
                            if femSpace.elementMaps.referenceElement.onElement(xi):
                                element = eN
        if element is None:
            element = -1
        return element, nearest_node, xi

    def evaluateFunAtX(self, x, ls_phi=None, debug=False):
        """
        reaction term
        """
        # f_scaled = (self.myfunc(x)*self.C)
        # f_scaled_reciprocal = 1./f_scaled
        if ls_phi is not None:
            f = min(abs(ls_phi), self.myfunc(x, self.t))
            # f = max(self.he_min, f)
            # f = min(self.he_max, f)
        else:
            f = self.myfunc(x, self.t)
            # f = min(self.he_max, f)
        f = f-self.epsFact_density*self.he_min
        if f < 0:
            f = 0.
        if self.grading_type == 1:
            f = self.he_min*self.grading**(f/self.he_min)
        if self.grading_type == 2:
            f = ((self.grading-1)*f+self.he_min)/self.grading
        f = min(self.he_max, f)
        f = self.fFactor(f)
        return f

    def evaluateFunAtNodes(self):
        for i in range(len(self.mesh.nodeArray)):
            if self.LS_MODEL is not None:
                f = min(abs(self.u_phi[i]),
                        self.myfunc(self.mesh.nodeArray[i], self.t))
                # f = max(self.he_min, f)
                # f = min(self.he_max, f)
            else:
                f = self.myfunc(self.mesh.nodeArray[i], self.t)
                # f = min(self.he_max, f)
            f = f-self.epsFact_density*self.he_min
            if f < 0:
                f = 0.
            if self.grading_type == 1:
                f = self.he_min*self.grading**(f/self.he_min)
            if self.grading_type == 2:
                f = (((self.grading-1)*f+self.he_min)/self.grading)
            f = min(self.he_max, f)
            f = self.fFactor(f)
            self.uOfXTatNodes[i] = f

    def evaluateFunAtQuadraturePoints(self):
        xx = self.model.q['x']
        N_k = self.model.q['x'].shape[1]
        for e in range(self.mesh.elementNodesArray.shape[0]):
            for k in range(N_k):
                if self.LS_MODEL is not None:
                    f = min(abs(self.q_phi[e, k]),
                            self.myfunc(xx[e, k], self.t))
                    # f = max(self.he_min, f)
                    # f = min(self.he_max, f)
                else:
                    f = self.myfunc(xx[e, k], self.t)
                    # f = min(self.he_max, f)
                f = f-self.epsFact_density*self.he_min
                if f < 0:
                    f = 0.
                if self.grading_type == 1:
                    f = self.he_min*self.grading**(f/self.he_min)
                if self.grading_type == 2:
                    f = ((self.grading-1)*f+self.he_min)/self.grading
                f = min(self.he_max, f)
                f = self.fFactor(f)
                self.uOfXTatQuadrature[e, k] = f

    def getLevelSetValue(self, eN, xi):
        value = self.model_ls.u[0].getValue(eN, xi)
        return value

    def getGradientValue(self, eN, xi):
        femSpace = self.model.u[0].femSpace
        value = 0.0
        #  tridelat hack: the following 3 lines do not work in parallel
        for i,psi in zip(femSpace.dofMap.l2g[eN],
                         femSpace.elementMaps.localFunctionSpace.basis):
            value+=self.grads[i]*psi(xi)
        # reverting to grad value within element
        # value = self.model.q[('grad(u)', 0)][eN, 0]
        return value

    def getAreaValue(self, eN, xi):
        femSpace = self.model.u[0].femSpace
        value = 0.0
        #  tridelat hack: the following 3 lines do not work in parallel
        for i,psi in zip(femSpace.dofMap.l2g[eN],
                         femSpace.elementMaps.localFunctionSpace.basis):
            value+=self.areas_nodes[i]*psi(xi)
        # reverting to grad value within element
        # value = self.model.q[('grad(u)', 0)][eN, 0]
        return value

    def getLevelSetValue(self, eN, xi):
        femSpace = self.model.u[0].femSpace
        value = 0.0
        #  tridelat hack: the following 3 lines do not work in parallel
        for i,psi in zip(femSpace.dofMap.l2g[eN],
                         femSpace.elementMaps.localFunctionSpace.basis):
            value+=self.u_phi[i]*psi(xi)
        # reverting to grad value within element
        # value = self.model.q[('grad(u)', 0)][eN, 0]
        return value

def calculate_areas(x, detJ, weights):
        N_k = len(weights)
        areas = np.zeros(len(x))
        for e in range(len(x)):
            area = 0
            for k in range(N_k):
                area += detJ[e, k]*weights[k]
            areas[e] = area
        return area

def calculate_area(x, detJ, weights):
        N_k = len(weights)
        area = 0
        for k in range(N_k):
            area += detJ[k]*weights[k]
        return area
