#!python
# cython: wraparound=True, boundscheck=False, initializedcheck=False, cdivision=True
# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
cdef extern from "index_type.h":
    ctypedef int proteus_index_t
from libcpp cimport bool
from proteus.Profiling import logEvent
from proteus.mprans import MeshSmoothing as ms
from proteus.mprans cimport MeshSmoothing as ms
from proteus import Comm
from mpi4py import MPI


cdef class cCoefficients:
    cdef public:
        object pyCoefficients
        double C

    def __cinit__(self):
        self.C = 1.

    def attachPyCoefficients(self,
                             object pyCoefficients):
        self.pyCoefficients = pyCoefficients

    def preStep(self):
        pc = self.pyCoefficients
        self.cppPreStep(q_uOfX=pc.uOfXTatQuadrature,
                        q_J=pc.model.q['abs(det(J))'],
                        q_weights=pc.model.elementQuadratureWeights[('u', 0)],
                        areas_=pc.areas,
                        q_rci=pc.model.q[('r', 0)],
                        q_fci=pc.model.q[('f', 0)],
                        nElements_owned=pc.mesh.nElements_owned)

    cdef cppPreStep(self,
                    double[:,:] q_uOfX,
                    double[:,:] q_J,
                    double[:] q_weights,
                    double[:] areas_,
                    double[:, :] q_rci,
                    double[:, :, :] q_fci,
                    proteus_index_t nElements_owned):
        cdef double integral_1_over_f = 0.
        cdef int N_eN = q_J.shape[0]
        cdef int nE = 0
        cdef double area = 0
        cdef int nJ = len(q_J)
        cdef int nk = len(q_weights)
        for eN in xrange(nJ):
            area = 0
            for k in xrange(nk):
                area += q_J[eN, k]*q_weights[k]
                if eN < nElements_owned:
                    integral_1_over_f += q_J[eN, k]*q_weights[k]/q_uOfX[eN, k]
            areas_[eN] = area
            if eN < nElements_owned:
                nE += 1
        comm = Comm.get().comm.tompi4py()
        if comm.size > 1:
            integral_1_over_f = comm.allreduce(integral_1_over_f, op=MPI.SUM)
            nE = comm.allreduce(nE, op=MPI.SUM)
        cdef double C = integral_1_over_f/nE  # update scaling coefficient
        self.C = C
        cdef int nrci = len(q_rci)
        for eN in range(nrci):
            for k in range(nk):
                q_fci[eN, k, :] = 0.0
                q_rci[eN, k] = -(1./(q_uOfX[eN, k]*C)-1./areas_[eN])

        # update other element values
        cdef object pc = self.pyCoefficients
        cdef object mesh = pc.mesh
        cdef int nd = pc.nd
        cdef double[:,:] elementBoundaryBarycentersArray = mesh.elementBoundaryBarycentersArray
        cdef double[:,:] elementBarycentersArray = mesh.elementBarycentersArray
        cdef double[:,:] nodeArray = mesh.nodeArray
        cdef proteus_index_t[:,:] elementBoundaryNodesArray = mesh.elementBoundaryNodesArray
        cdef proteus_index_t[:,:] elementNodesArray = mesh.elementNodesArray
        cdef proteus_index_t[:,:] elementBoundariesArray = mesh.elementBoundariesArray
        cdef proteus_index_t nElementBoundaries_global = mesh.nElementBoundaries_global
        cdef int nNodes_elementBoundary = mesh.nNodes_elementBoundary
        cdef proteus_index_t nElements_global = mesh.nElements_global
        cdef int nNodes_element = mesh.nNodes_element
        cdef double[:,:,:] elementBoundaryNormalsArray = pc.mesh.elementBoundaryNormalsArray
        # update element boundary barycenters
        for ebN in range(nElementBoundaries_global):
            elementBoundaryBarycentersArray[ebN, 0] = 0.
            elementBoundaryBarycentersArray[ebN, 1] = 0.
            elementBoundaryBarycentersArray[ebN, 2] = 0.
            for nN in range(nNodes_elementBoundary):
                elementBoundaryBarycentersArray[ebN, 0] += nodeArray[elementBoundaryNodesArray[ebN, nN], 0]
                elementBoundaryBarycentersArray[ebN, 1] += nodeArray[elementBoundaryNodesArray[ebN, nN], 1]
                elementBoundaryBarycentersArray[ebN, 2] += nodeArray[elementBoundaryNodesArray[ebN, nN], 2]
            elementBoundaryBarycentersArray[ebN, 0] /= nNodes_elementBoundary
            elementBoundaryBarycentersArray[ebN, 1] /= nNodes_elementBoundary
            elementBoundaryBarycentersArray[ebN, 2] /= nNodes_elementBoundary
        # update element barycenters
        for eN in range(nElements_global):
            elementBarycentersArray[eN, 0] = 0.
            elementBarycentersArray[eN, 1] = 0.
            elementBarycentersArray[eN, 2] = 0.
            for ebN in range(nNodes_element):
                elementBarycentersArray[eN, 0] += nodeArray[elementNodesArray[eN, ebN], 0]
                elementBarycentersArray[eN, 1] += nodeArray[elementNodesArray[eN, ebN], 1]
                elementBarycentersArray[eN, 2] += nodeArray[elementNodesArray[eN, ebN], 2]
            elementBarycentersArray[eN, 0] /= nNodes_element
            elementBarycentersArray[eN, 1] /= nNodes_element
            elementBarycentersArray[eN, 2] /= nNodes_element
        # update normals
        if nd == 2:
            # triangle
            ms.pyxUpdateElementBoundaryNormalsTriangle(elementBoundaryNormalsArray_=elementBoundaryNormalsArray,
                                                       nodeArray=nodeArray,
                                                       elementBoundariesArray=elementBoundariesArray,
                                                       elementBoundaryNodesArray=elementBoundaryNodesArray,
                                                       elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                                       elementBarycentersArray=elementBarycentersArray,
                                                       nElements=nElements_global)
        elif nd == 3:
            # tetra
            ms.pyxUpdateElementBoundaryNormalsTetra(elementBoundaryNormalsArray_=elementBoundaryNormalsArray,
                                                    nodeArray=nodeArray,
                                                    elementBoundariesArray=elementBoundariesArray,
                                                    elementBoundaryNodesArray=elementBoundaryNodesArray,
                                                    elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                                    elementBarycentersArray=elementBarycentersArray,
                                                    nElements=nElements_global)


    def postStep(self):
        pc = self.pyCoefficients
        self.cppPostStep()

    cdef cppPostStep(self):
        cdef object pc = self.pyCoefficients
        cdef object mesh = pc.mesh
        cdef int nd = pc.nd
        cdef double[:,:] elementBoundaryBarycentersArray = mesh.elementBoundaryBarycentersArray
        cdef double[:,:] elementBarycentersArray = mesh.elementBarycentersArray
        cdef double[:,:] nodeArray = mesh.nodeArray
        cdef proteus_index_t[:,:] elementBoundaryNodesArray = mesh.elementBoundaryNodesArray
        cdef proteus_index_t[:,:] elementNodesArray = mesh.elementNodesArray
        cdef proteus_index_t[:,:] elementBoundariesArray = mesh.elementBoundariesArray
        cdef proteus_index_t nElementBoundaries_global = mesh.nElementBoundaries_global
        cdef int nNodes_elementBoundary = mesh.nNodes_elementBoundary
        cdef proteus_index_t nElements_global = mesh.nElements_global
        cdef int nNodes_element = mesh.nNodes_element
        cdef double[:,:,:] elementBoundaryNormalsArray = pc.mesh.elementBoundaryNormalsArray
        # update element boundary barycenters
        for ebN in range(nElementBoundaries_global):
            elementBoundaryBarycentersArray[ebN, 0] = 0.
            elementBoundaryBarycentersArray[ebN, 1] = 0.
            elementBoundaryBarycentersArray[ebN, 2] = 0.
            for nN in range(nNodes_elementBoundary):
                elementBoundaryBarycentersArray[ebN, 0] += nodeArray[elementBoundaryNodesArray[ebN, nN], 0]
                elementBoundaryBarycentersArray[ebN, 1] += nodeArray[elementBoundaryNodesArray[ebN, nN], 1]
                elementBoundaryBarycentersArray[ebN, 2] += nodeArray[elementBoundaryNodesArray[ebN, nN], 2]
            elementBoundaryBarycentersArray[ebN, 0] /= nNodes_elementBoundary
            elementBoundaryBarycentersArray[ebN, 1] /= nNodes_elementBoundary
            elementBoundaryBarycentersArray[ebN, 2] /= nNodes_elementBoundary
        # update element barycenters
        for eN in range(nElements_global):
            elementBarycentersArray[eN, 0] = 0.
            elementBarycentersArray[eN, 1] = 0.
            elementBarycentersArray[eN, 2] = 0.
            for ebN in range(nNodes_element):
                elementBarycentersArray[eN, 0] += nodeArray[elementNodesArray[eN, ebN], 0]
                elementBarycentersArray[eN, 1] += nodeArray[elementNodesArray[eN, ebN], 1]
                elementBarycentersArray[eN, 2] += nodeArray[elementNodesArray[eN, ebN], 2]
            elementBarycentersArray[eN, 0] /= nNodes_element
            elementBarycentersArray[eN, 1] /= nNodes_element
            elementBarycentersArray[eN, 2] /= nNodes_element
        # update normals
        if nd == 2:
            # triangle
            ms.pyxUpdateElementBoundaryNormalsTriangle(elementBoundaryNormalsArray_=elementBoundaryNormalsArray,
                                                       nodeArray=nodeArray,
                                                       elementBoundariesArray=elementBoundariesArray,
                                                       elementBoundaryNodesArray=elementBoundaryNodesArray,
                                                       elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                                       elementBarycentersArray=elementBarycentersArray,
                                                       nElements=nElements_global)
        elif nd == 3:
            # tetra
            ms.pyxUpdateElementBoundaryNormalsTetra(elementBoundaryNormalsArray_=elementBoundaryNormalsArray,
                                                    nodeArray=nodeArray,
                                                    elementBoundariesArray=elementBoundariesArray,
                                                    elementBoundaryNodesArray=elementBoundaryNodesArray,
                                                    elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                                    elementBarycentersArray=elementBarycentersArray,
                                                    nElements=nElements_global)

    def pseudoTimeStepping(self,
                           xx,
                           eps=1.):
        pc = self.pyCoefficients
        return self.cppPseudoTimeSteppingParallel(xx=xx,
                                                  eps=eps)

    cdef cppPseudoTimeSteppingParallel(self,
                                       double[:,:] xx,
                                       double eps):
        pc = self.pyCoefficients
        cdef int ntimes_solved = pc.ntimes_solved
        cdef int ntimes_i = pc.ntimes_i
        cdef double t_min = 1./ntimes_solved*(ntimes_i)
        cdef double t_max = t_min+1./ntimes_solved
        logEvent("Pseudo-time stepping with dt={eps} ({t_min}<t<{t_max})".format(eps=eps, t_max=t_max, t_min=t_min),
                 level=3)
        cdef double[:] t_range = np.linspace(t_min, t_max, int((t_max-t_min)/eps+1))[1:]
        cdef int ntimesteps = len(t_range)
        cdef int[:] eN_phi = np.zeros(len(xx), dtype=np.int32)
        cdef double[:] normal = np.zeros(3)
        cdef double t_last = t_min
        cdef double dt = 0
        cdef int flag
        cdef bool fixed
        cdef double area = 0
        cdef proteus_index_t eN, eeN
        cdef double ls_phi
        cdef double f
        cdef double Ccoeff = pc.C
        cdef int nd = pc.nd
        cdef double[:] dphi = np.zeros(nd)
        cdef double[:,:] grads = pc.grads
        cdef double[:] v_grad = np.zeros(3)
        cdef double[:] areas_nodes = pc.areas_nodes,
        cdef double[:] areas = pc.areas
        cdef double[:] u_phi = pc.u_phi
        cdef int i, ii
        cdef proteus_index_t iN
        cdef object femSpace = pc.model.u[0].femSpace
        cdef bool find_nearest_node
        # initialise mesh memoryview before loop
        cdef double[:,:] nodeArray = pc.mesh.nodeArray
        cdef proteus_index_t[:] nodeStarOffsets = pc.mesh.nodeStarOffsets
        cdef proteus_index_t[:] nodeStarArray = pc.mesh.nodeStarArray
        cdef proteus_index_t[:] nodeElementOffsets = pc.mesh.nodeElementOffsets
        cdef proteus_index_t[:] nodeElementsArray = pc.mesh.nodeElementsArray
        cdef int[:] nodeMaterialTypes = pc.mesh.nodeMaterialTypes
        cdef double[:,:] elementBarycentersArray = pc.mesh.elementBarycentersArray
        cdef proteus_index_t[:,:] elementBoundaryElementsArray = pc.mesh.elementBoundaryElementsArray
        cdef proteus_index_t[:,:] elementBoundariesArray = pc.mesh.elementBoundariesArray
        cdef proteus_index_t[:,:] elementNodesArray = pc.mesh.elementNodesArray
        cdef double[:,:] elementBoundaryBarycentersArray = pc.mesh.elementBoundaryBarycentersArray
        cdef int[:] exteriorElementBoundariesBoolArray = np.zeros(pc.mesh.nElementBoundaries_global, dtype=np.int32)
        cdef int bb_i
        for bb_i in pc.mesh.exteriorElementBoundariesArray:
            exteriorElementBoundariesBoolArray[bb_i] = 1
        cdef double[:,:,:] elementBoundaryNormalsArray = pc.mesh.elementBoundaryNormalsArray
        cdef proteus_index_t nNodes_owned = pc.mesh.nNodes_owned
        cdef proteus_index_t nNodes_global = pc.mesh.nNodes_global
        cdef proteus_index_t nElements_owned = pc.mesh.nElements_owned
        cdef proteus_index_t nElements_global = pc.mesh.nElements_global
        cdef proteus_index_t[:] nodeNumbering_subdomain2global = pc.mesh.globalMesh.nodeNumbering_subdomain2global
        cdef proteus_index_t[:] elementNumbering_subdomain2global = pc.mesh.globalMesh.elementNumbering_subdomain2global
        cdef proteus_index_t[:] nodeOffsets_subdomain_owned = pc.mesh.globalMesh.nodeOffsets_subdomain_owned
        cdef proteus_index_t[:] elementOffsets_subdomain_owned = pc.mesh.globalMesh.elementOffsets_subdomain_owned
        cdef int[:] nearestNArray = np.array([i for i in range(len(xx))], dtype=np.int32)
        cdef int[:] typeNArray = np.zeros(nNodes_global, dtype=np.int32)
        cdef proteus_index_t[:,:] elementNeighborsArray = pc.mesh.elementNeighborsArray
        cdef proteus_index_t nElementBoundaries_owned = pc.mesh.nElementBoundaries_owned
        cdef proteus_index_t[:] elementBoundaryNumbering_subdomain2global = pc.mesh.globalMesh.elementBoundaryNumbering_subdomain2global
        cdef proteus_index_t[:] elementBoundaryOffsets_subdomain_owned = pc.mesh.globalMesh.elementBoundaryOffsets_subdomain_owned
        cdef proteus_index_t[:,:] elementBoundaryNodesArray = pc.mesh.elementBoundaryNodesArray
        cdef int[:] fixedNodesBoolArray = pc.mesh.fixedNodesBoolArray
        cdef int nSmoothOut = pc.nSmoothOut
        cdef int nSmoothIn = pc.nSmoothIn
        cdef int nSmooth = 0
        cdef bool inside_eN = False
        cdef double[:] vec = np.zeros(3)
        cdef double[:] vec2 = np.zeros(3)
        cdef double vec_dist
        cdef double[:] fixed_dir = np.ones(3)
        cdef proteus_index_t nearestN
        cdef int typeN
        comm = Comm.get().comm.tompi4py()
        comm.barrier()
        cdef int my_rank = comm.rank
        cdef int rank=0
        cdef int comm_size = comm.size
        cdef int[:] counts_local = np.zeros(comm_size, dtype=np.int32)
        cdef int[:,:] counts_total = np.zeros((comm_size, comm_size), dtype=np.int32)

        # the counts and displacements for nodes coming in from other processors
        cdef int[:] counts_in = np.zeros(comm_size, dtype=np.int32)
        cdef int[:] displacements_in = np.zeros(comm_size, dtype=np.int32)
        # the counts and displacements args_ coming back from other processors
        cdef int[:] counts_out = np.zeros(comm_size, dtype=np.int32)
        cdef int[:] displacements_out = np.zeros(comm_size, dtype=np.int32)
        # number of communications
        cdef int ncomm = 0
        # coords to send
        cdef dict coords_2rank = {}
        # nodes to send
        cdef dict nodes0_2rank = {}
        # original rank of sending rank
        cdef dict rank0_2rank = {}
        # original node number on sending rank
        cdef dict nearestN_2rank = {}
        # elements to send
        cdef dict typeN_2rank = {}
        # direction to rank (for sliding nodes)
        cdef dict dir_2rank = {}
        cdef dict b_i_2rank = {}
        # solutions
        cdef bool solFound
        cdef bool sendBack = False
        cdef int nPending_disp = 0  # number of pending solutions from other ranks
        cdef int nPending_disp_total = 0  # number of pending solutions from other ranks
        # declare variables for speed up
        cdef proteus_index_t node
        cdef proteus_index_t node0
        cdef proteus_index_t nNodes
        cdef proteus_index_t nOffset
        cdef double dot
        cdef double t
        cdef int j
        cdef int i_time
        cdef int parallel_steps
        cdef int new_rank
        cdef int nodeEl
        cdef double[:] coords = np.zeros(3)
        cdef proteus_index_t b_i, b_i_global
        cdef double[:] bound_bar
        cdef int[3] result
        cdef int[:] result2
        cdef bool found
        cdef double array_size_local = 0
        if comm_size > 1:
            for rank in range(comm_size):
                # things to send to other processors to get solution
                # things to send to other processors to get solution
                # coords_2rank[rank]:
                # 0: coords x
                # 1: coords y
                # 2: coords z
                # 3: nearestN
                # 4: typeN
                # 5: v_grad x
                # 6: v_grad y
                # 7: v_grad z
                # 8: area
                # 9: ls
                # 10: node0
                # 11: rank0
                # 12: fixed_dir[0]
                # 13: fixed_dir[1]
                # 14: fixed_dir[2]
                coords_2rank[rank] = np.zeros((0, 15))
        cdef double[:,:] coords_2doArray = np.zeros((0, 15))
        # cdef int[:] solFound_2do
        # cdef int[:] solFound_2doArray
        cdef double[:] nodesSentBoolArray = np.zeros(len(xx))
        cdef bool pending = False
        cdef double[:] starting_coords = np.zeros(3)
        cdef proteus_index_t nEbn  # number of boundaries per elements
        cdef double[:] resultd
        if nd == 2:
            nEbn = 3
        elif nd == 3:
            nEbn = 4
        cdef proteus_index_t nNel  # number of nodes per elements
        if nd == 2:
            nNel = 3
        elif nd == 3:
            nNel = 4
        for i_time in range(ntimesteps):
            t = t_range[i_time]
            logEvent("Pseudo-time stepping t={t}".format(t=t), level=3)
            nPending_disp = 0
            dt = t-t_last
            t_last = t
            for node in range(nNodes_owned):
                new_rank = my_rank
                # coords = xx[node]
                coords[0] = xx[node, 0]
                coords[1] = xx[node, 1]
                coords[2] = xx[node, 2]
                fixed = False
                pending = False
                area = 0.
                flag = nodeMaterialTypes[node]

                fixed_dir[0] = 1.
                fixed_dir[1] = 1.
                fixed_dir[2] = 1.
                b_i_global = -1
                nearestN = nearestNArray[node]
                typeN = typeNArray[node]
                for ndi in range(nd):
                    v_grad[ndi] = 0.
                    dphi[ndi] = 0.
                if nodesSentBoolArray[node] == 1:
                    pending = True
                    nPending_disp += 1
                elif nodesSentBoolArray[node] == 0:
                    if flag != 0:
                        if fixedNodesBoolArray[node] == 1:
                            fixed = True
                        else:
                            ms.cyFindBoundaryDirectionTriangle(
                                dir_=fixed_dir,
                                node=node,
                                nodeArray=nodeArray,
                                nodeStarOffsets=nodeStarOffsets,
                                nodeStarArray=nodeStarArray,
                                nodeMaterialTypes=nodeMaterialTypes,
                            )
                    if not fixed:  # either flag==0 or not fixed
                        if coords[0] == nodeArray[node, 0] and coords[1] == nodeArray[node, 1] and coords[2] == nodeArray[node, 2]:  # nodes are at original position (no search)
                        # if i_time == 0:
                            for ndi in range(nd):
                                v_grad[ndi] = grads[node, ndi]
                            area = areas_nodes[node]
                            eN = -1
                            eN_phi[node] = eN
                            if u_phi is not None:
                                ls_phi = u_phi[node]
                            else:
                                ls_phi = 1e12
                            f = pc.evaluateFunAtX(x=coords, ls_phi=ls_phi)
                            for ndi in range(nd):
                                dphi[ndi] = v_grad[ndi]/(t*1./(f*Ccoeff)+(1-t)*1./area)
                                coords[ndi] += dphi[ndi]*dt*fixed_dir[ndi]
                                xx[node, ndi] = coords[ndi]
                        else:  # find node that moved already (search)
                            findN(coords=coords,
                                           nodeArray=nodeArray,
                                           nodeStarOffsets=nodeStarOffsets,
                                           nodeStarArray=nodeStarArray,
                                           nearestN=nearestNArray[node],
                                           typeN = typeNArray[node],
                                           my_rank=my_rank,
                                           nNodes_owned=nNodes_owned,
                                           nodeNumbering_subdomain2global=nodeNumbering_subdomain2global,
                                           nodeOffsets_subdomain_owned=nodeOffsets_subdomain_owned,
                                           nodeElementOffsets=nodeElementOffsets,
                                           nodeElementsArray=nodeElementsArray,
                                           elementBarycentersArray=elementBarycentersArray,
                                           elementBoundaryNormalsArray=elementBoundaryNormalsArray,
                                           elementBoundariesArray=elementBoundariesArray,
                                           elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                           elementBoundaryElementsArray=elementBoundaryElementsArray,
                                           exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
                                           nElements_owned=nElements_owned,
                                           elementNumbering_subdomain2global=elementNumbering_subdomain2global,
                                           elementOffsets_subdomain_owned=elementOffsets_subdomain_owned,
                                           nElementBoundaries_owned=nElementBoundaries_owned,
                                           elementBoundaryNumbering_subdomain2global=elementBoundaryNumbering_subdomain2global,
                                           elementBoundaryOffsets_subdomain_owned=elementBoundaryOffsets_subdomain_owned,
                                           nodeMaterialTypes=nodeMaterialTypes,
                                           elementNodesArray=elementNodesArray,
                                           nNel=nNel,
                                  result_out = result)
                            nearestN = result[0]
                            typeN = result[1]
                            new_rank = result[2]
                            if new_rank == my_rank:
                                inside_eN = True
                                nearestNArray[node] = nearestN
                                typeNArray[node] = typeN
                                if typeN == 2 and flag != 0:
                                    # get an element from there
                                    if elementBoundaryElementsArray[nearestN, 0] == -1 or elementBoundaryElementsArray[nearestN, 0] > nElements_owned:
                                        nearestN = elementBoundaryElementsArray[nearestN, 1]
                                        typeN = 1
                                    elif elementBoundaryElementsArray[nearestN, 1] == -1 or elementBoundaryElementsArray[nearestN, 1] > nElements_owned:
                                        nearestN = elementBoundaryElementsArray[nearestN, 0]
                                        typeN = 1
                                    else:
                                        nearestN = elementBoundaryElementsArray[nearestN, 0]
                                        typeN = 1

                                        # fixed_dir[0] = nodeArray[elementBoundaryNodesArray[1], 0]-nodeArray[elementBoudnaryNodesArray[0], 0]
                                        # fixed_dir[1] = nodeArray[elementBoundaryNodesArray[1], 1]-nodeArray[elementBoudnaryNodesArray[0], 1]
                                        # fixed_dir[2] = nodeArray[elementBoundaryNodesArray[1], 2]-nodeArray[elementBoudnaryNodesArray[0], 2]

                                for j in range(nEbn):
                                    bb_i = elementBoundariesArray[nearestN, j]
                                    normal[0] = elementBoundaryNormalsArray[nearestN, j, 0]
                                    normal[1] = elementBoundaryNormalsArray[nearestN, j, 1]
                                    normal[2] = elementBoundaryNormalsArray[nearestN, j, 2]
                                    bound_bar = elementBoundaryBarycentersArray[bb_i]
                                    dot = (bound_bar[0]-coords[0])*normal[0]+(bound_bar[1]-coords[1])*normal[1]+(bound_bar[2]-coords[2])*normal[2]
                                    if dot < 0.:
                                        inside_eN = False
                                        try:
                                            raise AssertionError('did not find containing element! coords outside domain?')
                                        except AssertionError as exc:
                                            print('Error: ', exc)
                                            print('node number', node)
                                            print('node materialType: ',
                                                nodeMaterialTypes[node])
                                            print('node old coordinates: ',
                                                nodeArray[node, 0],
                                                nodeArray[node, 1],
                                                nodeArray[node, 2])
                                            print('node new coordinates: ',
                                                coords[0],
                                                coords[1],
                                                coords[2])
                                            print('nearestN (element): ',
                                                nearestN,
                                                'type (should be 1): ',
                                                typeN)
                                            print('element barycenter: ',
                                                elementBarycentersArray[nearestN,0],
                                                elementBarycentersArray[nearestN,1],
                                                elementBarycentersArray[nearestN,2])
                                            for ii in range(elementNodesArray.shape[1]):
                                                print('element node: ',
                                                    elementNodesArray[nearestN, ii],
                                                    nodeArray[elementNodesArray[nearestN, ii], 0],
                                                    nodeArray[elementNodesArray[nearestN, ii], 1],
                                                    nodeArray[elementNodesArray[nearestN, ii], 2])
                                            print('fixed_dir: ',
                                                fixed_dir[0],
                                                fixed_dir[1],
                                                fixed_dir[2])
                                            print('dot: ', dot)
                                            comm.Abort()
                                if inside_eN:
                                    xi = femSpace.elementMaps.getInverseValue(nearestN, coords)
                                    # line below needs optimisation:
                                    v_grad = pc.getGradientValue(nearestN, xi)
                                    area = pc.getAreaValue(nearestN, xi)
                                    if u_phi is not None:
                                        # line below needs optimisation:
                                        ls_phi = pc.getLevelSetValue(nearestN, coords)
                                    else:
                                        ls_phi = 1e12
                                    f = pc.evaluateFunAtX(x=coords, ls_phi=ls_phi)
                                    for ndi in range(nd):
                                        dphi[ndi] = v_grad[ndi]/(t*1./(f*Ccoeff)+(1-t)*1./area)
                                        coords[ndi] += dphi[ndi]*dt*fixed_dir[ndi]
                                        xx[node, ndi] = coords[ndi]
                            else:  # info to send to other processor
                                coords_2rank[new_rank] = np.append(coords_2rank[new_rank],
                                                                   [[coords[0],
                                                                     coords[1],
                                                                     coords[2],
                                                                     nearestN,
                                                                     typeN,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     node,
                                                                     my_rank,
                                                                     fixed_dir[0],
                                                                     fixed_dir[1],
                                                                     fixed_dir[2]]],
                                                                   axis=0)
                                nodesSentBoolArray[node] = 1
                                nPending_disp += 1
            # parallel comm
            comm.barrier()
            if comm_size > 1:
                # number of pending solutions to send to other processors
                nPending_disp_total = comm.allreduce(nPending_disp)
                parallel_steps = -1
                while nPending_disp_total > 0:
                    parallel_steps += 1
                    nPending_disp = 0
                    sendBack = True
                    # initialize solution from previously found solutions
                    # coords_2doArray = coords_2rank[my_rank]
                    # gather counts info from all processors
                    for rank in range(comm_size):
                        counts_local[rank] = len(coords_2rank[rank])
                    comm.Allgatherv([counts_local, MPI.INT],
                                    [counts_total,
                                     [comm_size for i in range(comm_size)],
                                     [comm_size*i for i in range(comm_size)],
                                     MPI.INT])
                    comm.barrier()
                    # calculate array_size for current rank
                    for rank_recv in range(comm_size):
                        array_size = 0
                        for rank in range(comm_size):
                            array_size += counts_total[rank, rank_recv]
                            counts_in[rank] = counts_total[rank, rank_recv]
                            if rank > 0:
                                displacements_in[rank] = displacements_in[rank-1]+counts_in[rank-1]
                        # check if another parallel comm is needed
                        if array_size-counts_total[rank_recv, rank_recv] > 0:
                            if my_rank == rank_recv:
                                # initialise coords_2doArray only on receiving processor
                                coords_2doArray = np.zeros((array_size, 15))
                            # -----
                            # get the coords_2doArray (nodes where to retrieve values for arg)
                            datatype = MPI.DOUBLE.Create_contiguous(15).Commit()
                            comm.Gatherv(coords_2rank[rank_recv],
                                         [coords_2doArray,
                                          tuple(counts_in[i]*coords_2doArray.shape[1] for i in range(comm_size)),
                                          tuple(displacements_in[i]*coords_2doArray.shape[1] for i in range(comm_size)),
                                          MPI.DOUBLE],
                                         root=rank_recv)
                            ncomm += 1
                            comm.barrier()
                        else:
                            comm.barrier()
                            if rank_recv == my_rank: 
                                coords_2doArray = coords_2rank[my_rank]
                            comm.barrier()
                        if my_rank == rank_recv:
                            solFound_2doArray = np.zeros(array_size, dtype=np.int32)
                            if parallel_steps > 0:
                                # if coming from this rank, solution was already found
                                solFound_2doArray[displacements_in[my_rank]:displacements_in[my_rank]+counts_in[my_rank]] = 1
                    # COMMUNICATION FINISHED
                    # wipe dicts out
                    for rank in range(comm_size):
                        coords_2rank[rank] = np.zeros((0, 15))
                    nNodes = len(coords_2doArray)
                    for iN in range(nNodes):
                        coords[0] = coords_2doArray[iN, 0]
                        coords[1] = coords_2doArray[iN, 1]
                        coords[2] = coords_2doArray[iN, 2]
                        nearestN = int(coords_2doArray[iN, 3])
                        typeN = int(coords_2doArray[iN, 4])
                        v_grad[0] = coords_2doArray[iN, 5]
                        v_grad[1] = coords_2doArray[iN, 6]
                        v_grad[2] = coords_2doArray[iN, 7]
                        area = coords_2doArray[iN, 8]
                        ls_phi = coords_2doArray[iN, 9]
                        node0 = int(coords_2doArray[iN, 10])
                        rank0  = int(coords_2doArray[iN, 11])
                        new_rank = my_rank
                        fixed_dir[0] = coords_2doArray[iN, 12]
                        fixed_dir[1] = coords_2doArray[iN, 13]
                        fixed_dir[2] = coords_2doArray[iN, 14]
                        if solFound_2doArray[iN] == 0:
                            findN(coords=coords,
                                           nodeArray=nodeArray,
                                           nodeStarOffsets=nodeStarOffsets,
                                           nodeStarArray=nodeStarArray,
                                           nearestN=nearestN,
                                           typeN = typeN,
                                           my_rank=my_rank,
                                           nNodes_owned=nNodes_owned,
                                           nodeNumbering_subdomain2global=nodeNumbering_subdomain2global,
                                           nodeOffsets_subdomain_owned=nodeOffsets_subdomain_owned,
                                           nodeElementOffsets=nodeElementOffsets,
                                           nodeElementsArray=nodeElementsArray,
                                           elementBarycentersArray=elementBarycentersArray,
                                           elementBoundaryNormalsArray=elementBoundaryNormalsArray,
                                           elementBoundariesArray=elementBoundariesArray,
                                           elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                           elementBoundaryElementsArray=elementBoundaryElementsArray,
                                           exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
                                           nElements_owned=nElements_owned,
                                           elementNumbering_subdomain2global=elementNumbering_subdomain2global,
                                           elementOffsets_subdomain_owned=elementOffsets_subdomain_owned,
                                           nElementBoundaries_owned=nElementBoundaries_owned,
                                           elementBoundaryNumbering_subdomain2global=elementBoundaryNumbering_subdomain2global,
                                           elementBoundaryOffsets_subdomain_owned=elementBoundaryOffsets_subdomain_owned,
                                           nodeMaterialTypes=nodeMaterialTypes,
                                           elementNodesArray=elementNodesArray,
                                           nNel=nNel,
                                  result_out=result)
                            nearestN = result[0]
                            typeN = result[1]
                            new_rank = result[2]
                            if my_rank != new_rank:
                                nPending_disp += 1
                            else:
                                solFound_2doArray[iN] += 1
                                inside_eN = True  # checking if actually true
                                if typeN == 2:
                                    # get an element from there
                                    if elementBoundaryElementsArray[nearestN, 0] == -1:
                                        nearestN = elementBoundaryElementsArray[nearestN, 1]
                                        typeN = 1
                                    elif elementBoundaryElementsArray[nearestN, 1] == -1:
                                        nearestN = elementBoundaryElementsArray[nearestN, 0]
                                        typeN = 1
                                for ii in range(nEbn):
                                    bb_i = elementBoundariesArray[nearestN, ii]
                                    normal[0] = elementBoundaryNormalsArray[nearestN, ii, 0]
                                    normal[1] = elementBoundaryNormalsArray[nearestN, ii, 1]
                                    normal[2] = elementBoundaryNormalsArray[nearestN, ii, 2]
                                    bound_bar = elementBoundaryBarycentersArray[bb_i]
                                    dot = (bound_bar[0]-coords[0])*normal[0]+(bound_bar[1]-coords[1])*normal[1]+(bound_bar[2]-coords[2])*normal[2]
                                    if dot < 0.:
                                        inside_eN = False
                                        try:
                                            raise AssertionError('did not find containing element! coords outside domain??')
                                        except AssertionError as exc:
                                            print('Error: ', exc)
                                            print('node number', node)
                                            print('node materialType: ',
                                                nodeMaterialTypes[node])
                                            print('node old coordinates: ',
                                                nodeArray[node, 0],
                                                nodeArray[node, 1],
                                                nodeArray[node, 2])
                                            print('node new coordinates: ',
                                                coords[0],
                                                coords[1],
                                                coords[2])
                                            print('nearestN (element): ',
                                                nearestN,
                                                'type (should be 1): ',
                                                typeN)
                                            print('element barycenter: ',
                                                elementBarycentersArray[nearestN,0],
                                                elementBarycentersArray[nearestN,1],
                                                elementBarycentersArray[nearestN,2])
                                            for ii in range(elementNodesArray.shape[1]):
                                                print('element node: ',
                                                    elementNodesArray[nearestN, ii],
                                                    nodeArray[elementNodesArray[nearestN, ii], 0],
                                                    nodeArray[elementNodesArray[nearestN, ii], 1],
                                                    nodeArray[elementNodesArray[nearestN, ii], 2])
                                            print('fixed_dir: ',
                                                fixed_dir[0],
                                                fixed_dir[1],
                                                fixed_dir[2])
                                            print('dot: ', dot)
                                            comm.Abort()
                                if inside_eN:
                                    xi = femSpace.elementMaps.getInverseValue(nearestN, coords)
                                    # line below needs optimisation:
                                    v_grad = pc.getGradientValue(nearestN, xi)
                                    area = pc.getAreaValue(nearestN, xi)
                                    if u_phi is not None:
                                        # line below needs optimisation:
                                        ls_phi = pc.getLevelSetValue(nearestN, coords)
                                    else:
                                        ls_phi = 1e12
                                    f = pc.evaluateFunAtX(x=coords, ls_phi=ls_phi)
                                    for ndi in range(nd):
                                        dphi[ndi] = v_grad[ndi]/(t*1./(f*Ccoeff)+(1-t)*1./area)
                                        coords[ndi] += dphi[ndi]*fixed_dir[ndi]*dt
                        coords_2rank[new_rank] = np.append(coords_2rank[new_rank],
                                                           [[coords[0],
                                                             coords[1],
                                                             coords[2],
                                                             nearestN,
                                                             typeN,
                                                             v_grad[0],
                                                             v_grad[1],
                                                             v_grad[2],
                                                             area,
                                                             ls_phi,
                                                             node0,
                                                             rank0,
                                                             fixed_dir[0],
                                                             fixed_dir[1],
                                                             fixed_dir[2]]],
                                                           axis=0)
                    nPending_disp_total = comm.allreduce(nPending_disp)

        comm.barrier()
        # SEND NODES POSITION SOLUTION BACK TO ORIGINAL PROCESSORS
        if sendBack is True:
            coords_2doArray = coords_2rank[my_rank]
            for rank in range(comm.size):
                # things to send to other processors to get solution
                coords_2rank[rank] = np.zeros((0, 15))
            nNodes = len(coords_2doArray)
            for iN in range(nNodes):
                rank0 = coords_2doArray[iN, 11]
                coords_2rank[rank0] = np.append(coords_2rank[rank0], [coords_2doArray[iN]], axis=0)
            # gather counts info from all processors
            for rank in range(comm_size):
                counts_local[rank] = len(coords_2rank[rank])
            comm.Allgatherv([counts_local, MPI.INT],
                            [counts_total,
                             [comm_size for i in range(comm_size)],
                             [comm_size*i for i in range(comm_size)],
                             MPI.INT])
            # calculate array_size for current rank
            for rank_recv in range(comm_size):
                array_size = 0
                for rank in range(comm_size):
                    array_size += counts_total[rank, rank_recv]
                    counts_in[rank] = counts_total[rank, rank_recv]
                    if rank > 0:
                        displacements_in[rank] = displacements_in[rank-1]+counts_in[rank-1]
                # check if another parallel comm is needed
                if array_size-counts_total[rank_recv, rank_recv] > 0:
                    if my_rank == rank_recv:
                        # initialise coords_2doArray only on receiving processor
                        coords_2doArray = np.zeros((array_size, 15))
                    # -----
                    # get the coords_2doArray (nodes where to retrieve values for arg)
                    datatype = MPI.DOUBLE.Create_contiguous(15).Commit()
                    comm.Gatherv(coords_2rank[rank_recv],
                                 [coords_2doArray,
                                  tuple(counts_in[i]*coords_2doArray.shape[1] for i in range(comm_size)),
                                  tuple(displacements_in[i]*coords_2doArray.shape[1] for i in range(comm_size)),
                                  MPI.DOUBLE],
                                 root=rank_recv)
                    ncomm += 1
                else:
                    if rank_recv == my_rank:
                        coords_2doArray = coords_2rank[my_rank]
            nNodes = len(coords_2doArray)
            for iN in range(nNodes):
                node0 = int(coords_2doArray[iN, 10])
                coords[0] = coords_2doArray[iN, 0]
                coords[1] = coords_2doArray[iN, 1]
                coords[2] = coords_2doArray[iN, 2]
                for ind in range(nd):
                    xx[node0, ind] = coords[ind]

        comm.barrier()
        # FINAL STEP: GET NON-OWNED NODES POSITION FOR CONSISTENCY
        # BUILD NON OWNED NODES ARRAY TO RETRIEVE SOLUTION
        ms.getNonOwnedNodeValues(xx,
                                 nNodes_owned,
                                 nNodes_global,
                                 nodeNumbering_subdomain2global,
                                 nodeOffsets_subdomain_owned)


        if nSmoothIn > 0:
            nSmooth = nSmoothIn
        elif nSmoothOut > 0 and ntimes_i == ntimes_solved-1:
            nSmooth = nSmoothOut
        comm.barrier()
        if nSmooth > 0:
            logEvent('Smoothing Mesh with Laplace Smoothing - '+str(nSmooth))
            for iS in range(nSmooth):
                # elementVolumesArray = self.model.q['abs(det(J))'][:,0]
                # elementBarycentersArray = self.mesh.elementBarycentersArray
                # ms.updateElementVolumes(elementVolumesArray_=elementVolumesArray,
                #                       elementNodesArray=self.mesh.elementNodesArray,
                #                       nodeArray=self.PHI)
                # ms.updateElementBarycenters(elementBarycentersArray_=elementBarycentersArray,
                #                             elementNodesArray=self.mesh.elementNodesArray,
                #                             nodeArray=self.PHI)
                simultaneous = True
                ms.smoothNodesLaplace(nodeArray_=xx,
                                      nodeStarOffsets=nodeStarOffsets,
                                      nodeStarArray=nodeStarArray,
                                      nodeMaterialTypes=nodeMaterialTypes,
                                      nNodes_owned=nNodes_owned,
                                      nd=nd,
                                      simultaneous=simultaneous,
                                      smoothBoundaries=True,
                                      fixedNodesBoolArray=fixedNodesBoolArray,
                                      alpha=0.)
                comm.barrier()
                ms.getNonOwnedNodeValues(xx,
                                        nNodes_owned,
                                        nNodes_global,
                                        nodeNumbering_subdomain2global,
                                        nodeOffsets_subdomain_owned)
            logEvent('Done smoothing')

        logEvent('Done pseudo-timestepping')

    # def evaluateFunAtX(self, x, ls_phi=None):
    #     pc = self.pyCoefficients
    #     f = self.cppEvaluateFunAtX(x=x,
    #                                he_min=pc.he_min,
    #                                he_max=pc.he_max,
    #                                ls_phi=ls_phi,
    #                                fun=pc.fun,
    #                                t=pc.t)

    # cdef double cppEvaluateFunAtX(self,
    #                               double[:] x,
    #                               double he_min,
    #                               double he_max,
    #                               double ls_phi=None,
    #                               object fun=None,
    #                               double t=None):
    #     cdef double f
    #     if fun:
    #         f = fun(x, self.t)
    #     else:
    #         f = 0.
    #     if ls_phi is not None:
    #         f = min(abs(ls_phi), f)
    #     f = max(he_min, f)
    #     f = min(he_max, f)
    #     return f

    # def evaluateFunAtNodes(self):
    #     pc = self.pyCoefficients
    #     self.cppEvaluateFunAtNodes(nodeArray=pc.mesh.nodeArray,
    #                                uOfXTatNodes=pc.uOfXTatNodes,
    #                                he_min=pc.he_min,
    #                                he_max=pc.he_max,
    #                                fun=pc.func,
    #                                t=pc.t,
    #                                u_phi=pc.u_phi)

    # cdef cppEvaluateFunAtNodes(self,
    #                            double[:] nodeArray,
    #                            double[:] uOfXTatNodes,
    #                            double he_min,
    #                            double he_max,
    #                            object fun=None,
    #                            double t=None,
    #                            double[:] u_phi=None):
    #     cdef double f
    #     for i in range(len(self.mesh.nodeArray)):
    #         if fun:
    #             f = fun(nodeArray[i], t)
    #         else:
    #             f = 0.
    #         if u_phi is not None:
    #             f = min(abs(u_phi[i]), f)
    #         f = max(he_min, f)
    #         f = min(he_max, f)
    #         uOfXTatNodes[i] = f

    # def evaluateFunAtQuadraturePoints(self):
    #     pc = self.pyCoefficients
    #     self.cppEvaluateFunAtQuadraturePoints(qx=pc.model.q['x'],
    #                                           uOfXTatQuadrature=pc.uOfXTatQuadrature,
    #                                           he_min=pc.he_min,
    #                                           he_max=pc.he_max,
    #                                           q_phi=pc.q_phi,
    #                                           fun=pc.myfunc,
    #                                           t=pc.t)

    # cdef cppEvaluateFunAtQuadraturePoints(self,
    #                                       double[:,:,:] qx,
    #                                       double[:,:] uOfXTatQuadrature,
    #                                       double he_min,
    #                                       double he_max,
    #                                       double[:,:] q_phi=None,
    #                                       fun=None,
    #                                       t=None):
    #     cdef int N_k = qx.shape[1]
    #     cdef double f
    #     for e in xrange(len(qx)):
    #         for k in xrange(N_k):
    #             if fun:
    #                 f = fun(qx[e, k], self.t)
    #             if q_phi is not None:
    #                 f = min(abs(q_phi[e, k]), f)
    #             f = max(he_min, f)
    #             f = min(he_max, f)
    #             self.uOfXTatQuadrature[e, k] = f

def recoveryAtNodes(double[:] scalars,
                    proteus_index_t[:] nodeElementsArray,
                    proteus_index_t[:] nodeElementOffsets):
    return cppRecoveryAtNodes(scalars=scalars,
                              nodeElementsArray=nodeElementsArray,
                              nodeElementOffsets=nodeElementOffsets)

cdef double[:] cppRecoveryAtNodes(double[:] scalars,
                                  proteus_index_t[:] nodeElementsArray,
                                  proteus_index_t[:] nodeElementOffsets):
    """
    scalar:
         Scalar in element
    """
    cdef double[:] recovered_scalars = np.zeros(len(nodeElementOffsets)-1)
    cdef proteus_index_t nb_el
    cdef proteus_index_t eOffset
    cdef double var_sum
    for node in range(len(nodeElementOffsets)-1):
        nb_el = 0
        var_sum = 0
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            var_sum += scalars[nodeElementsArray[eOffset]]
        recovered_scalars[node] = var_sum/nb_el
    return recovered_scalars

def recoveryAtNodesWeighted(double[:] scalars,
                            proteus_index_t[:] nodeElementsArray,
                            proteus_index_t[:] nodeElementOffsets,
                            double[:] detJ_array,
                            proteus_index_t nNodes):
    return cppRecoveryAtNodesWeighted(scalars=scalars,
                                      nodeElementsArray=nodeElementsArray,
                                      nodeElementOffsets=nodeElementOffsets,
                                      detJ_array=detJ_array,
                                      nNodes=nNodes)

cdef double[:] cppRecoveryAtNodesWeighted(double[:] scalars,
                                          proteus_index_t[:] nodeElementsArray,
                                          proteus_index_t[:] nodeElementOffsets,
                                          double[:] detJ_array,
                                          proteus_index_t nNodes):
    cdef double[:] recovered_scalars = np.zeros(nNodes)
    cdef proteus_index_t nb_el
    cdef double detJ_patch = 0.
    cdef double scalar_sum = 0.
    cdef proteus_index_t node
    cdef proteus_index_t eOffset
    cdef proteus_index_t eN
    for node in range(nNodes):
        nb_el = 0
        detJ_patch = 0.
        scalar_sum = 0.
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            eN = nodeElementsArray[eOffset]
            # for k in range(n_quad):
            #     scalar_k = gradrads[eN, k]
            #     scalar_eN_av += gradrad_k
            # scalar_eN_av /= n_quad
            detJ_patch += detJ_array[eN]
            scalar_sum += detJ_array[eN]*scalars[eN]  # same value at all quad points
        recovered_scalars[node] = scalar_sum/detJ_patch
    return recovered_scalars


def gradientRecoveryAtNodes(grads,
                            nodeElementsArray,
                            nodeElementOffsets,
                            nd):
    return cppGradientRecoveryAtNodes(grads=grads,
                                      nodeElementsArray=nodeElementsArray,
                                      nodeElementOffsets=nodeElementOffsets,
                                      nd=nd)

cdef double[:,:] cppGradientRecoveryAtNodes(double[:,:,:] grads,
                                            proteus_index_t[:] nodeElementsArray,
                                            proteus_index_t[:] nodeElementOffsets,
                                            int nd):
    cdef double[:, :] recovered_grads = np.zeros((len(nodeElementOffsets)-1, nd))
    cdef proteus_index_t nb_el
    cdef double[:] grad_sum = np.zeros(nd)
    for node in range(len(nodeElementOffsets)-1):
        nb_el = 0
        for ndi in range(nd):
            grad_sum[ndi] = 0.
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            eN = nodeElementsArray[eOffset]
            # for k in range(n_quad):
            #     grad_k = grads[eN, k]
            #     grad_eN_av += grad_k
            # grad_eN_av /= n_quad
            for ndi in range(nd):
                grad_sum[ndi] += grads[eN, 0, ndi]  # same value at all quad points
        for ndi in range(nd):
            recovered_grads[node, ndi] = grad_sum[ndi]/nb_el
    return recovered_grads

def gradientRecoveryAtNodesWeighted(double[:,:,:] grads,
                                    proteus_index_t[:] nodeElementsArray,
                                    proteus_index_t[:] nodeElementOffsets,
                                    double[:,:] detJ_array,
                                    int nd):
    return cppGradientRecoveryAtNodesWeighted(grads=grads,
                                              nodeElementsArray=nodeElementsArray,
                                              nodeElementOffsets=nodeElementOffsets,
                                              detJ_array=detJ_array,
                                              nd=nd)

cdef double[:,:] cppGradientRecoveryAtNodesWeighted(double[:,:,:] grads,
                                                    proteus_index_t[:] nodeElementsArray,
                                                    proteus_index_t[:] nodeElementOffsets,
                                                    double[:,:] detJ_array,
                                                    int nd):
    cdef double[:, :] recovered_grads = np.zeros((len(nodeElementOffsets)-1, nd))
    cdef proteus_index_t nb_el
    cdef double detJ_patch = 0.
    cdef double[:] grad_sum = np.zeros(nd)
    for node in range(len(nodeElementOffsets)-1):
        nb_el = 0
        detJ_patch = 0.
        for ndi in range(nd):
            grad_sum[ndi] = 0.
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            eN = nodeElementsArray[eOffset]
            # for k in range(n_quad):
            #     grad_k = grads[eN, k]
            #     grad_eN_av += grad_k
            # grad_eN_av /= n_quad
            detJ_patch += detJ_array[eN,0]
            for ndi in range(nd):
                grad_sum[ndi] += detJ_array[eN,0]*grads[eN, 0, ndi]  # same value at all quad points
        for ndi in range(nd):
            recovered_grads[node, ndi] = grad_sum[ndi]/detJ_patch
    return recovered_grads



# cdef tuple pyxSearchNearestNodeElementFromMeshObject(object mesh,
#                                                      double[:] x,
#                                                      int[:] exteriorElementBoundariesBoolArray,
#                                                      double[:,:,:] elementBoundaryNormalsArray,
#                                                      proteus_index_t nearest_node,
#                                                      proteus_index_t eN):
#     nearest_node = ms.pyxGetLocalNearestNode(coords=x,
#                                              nodeArray=mesh.nodeArray,
#                                              nodeStarOffsets=mesh.nodeStarOffsets,
#                                              nodeStarArray=mesh.nodeStarArray,
#                                              node=nearest_node)
#     eN = ms.pyxGetLocalNearestElementAroundNode(coords=x,
#                                                 nodeElementOffsets=mesh.nodeElementOffsets,
#                                                 nodeElementsArray=mesh.nodeElementsArray,
#                                                 elementBarycenterArray=mesh.elementBarycentersArray,
#                                                 node=nearest_node)
#     eN = ms.pyxGetLocalNearestElementIntersection(coords=x,
#                                                   elementBoundaryNormalsArray=elementBoundaryNormalsArray,
#                                                   elementBoundariesArray=mesh.elementBoundariesArray,
#                                                   elementBarycentersArray=mesh.elementBoundaryBarycentersArray,
#                                                   elementBoundaryElementsArray=mesh.elementBoundaryElementsArray,
#                                                   elementBarycentersArray=mesh.elementBarycentersArray,
#                                                   exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
#                                                   eN=eN)
#     # check if actually inside eN
#     inside_eN = True
#     for j, b_i in enumerate(mesh.elementBoundariesArray[eN]):
#         normal = mesh.elementBoundaryNormalsArray[eN, j]
#         bound_bar = mesh.elementBoundaryBarycentersArray[b_i]
#         dot = (bound_bar[0]-x[0])*normal[0]+(bound_bar[1]-x[1])*normal[1]+(bound_bar[2]-x[2])*normal[2]
#         if dot < 0:
#             inside_eN = False
#     return eN, nearest_node, inside_eN


# cdef tuple pyxSearchNearestNodeElement(double[:] x,
#                                        double[:,:] nodeArray,
#                                        proteus_index_t[:] nodeStarOffsets,
#                                        proteus_index_t[:] nodeStarArray,
#                                        proteus_index_t[:] nodeElementOffsets,
#                                        proteus_index_t[:] nodeElementsArray,
#                                        double[:,:] elementBarycentersArray,
#                                        proteus_index_t[:,:] elementBoundaryElementsArray,
#                                        proteus_index_t[:,:] elementBoundariesArray,
#                                        double[:,:] elementBoundaryBarycentersArray,
#                                        int[:] exteriorElementBoundariesBoolArray,
#                                        double[:,:,:] elementBoundaryNormalsArray,
#                                        proteus_index_t nearest_node,
#                                        proteus_index_t eN,
#                                        bool find_nearest_node=True):
#     if find_nearest_node is True:
#         nearest_node = ms.pyxGetLocalNearestNode(coords=x,
#                                                  nodeArray=nodeArray,
#                                                 nodeStarOffsets=nodeStarOffsets,
#                                                  nodeStarArray=nodeStarArray,
#                                                  node=nearest_node)
#         eN = ms.pyxGetLocalNearestElementAroundNode(coords=x,
#                                                     nodeElementOffsets=nodeElementOffsets,
#                                                     nodeElementsArray=nodeElementsArray,
#                                                     elementBarycentersArray=elementBarycentersArray,
#                                                     node=nearest_node)
#         eN = ms.pyxGetLocalNearestElementIntersection(coords=x,
#                                                       elementBoundaryNormalsArray=elementBoundaryNormalsArray,
#                                                       elementBoundariesArray=elementBoundariesArray,
#                                                       elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
#                                                       elementBoundaryElementsArray=elementBoundaryElementsArray,
#                                                       elementBarycentersArray=elementBarycentersArray,
#                                                       exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
#                                                       eN=eN)
#     # check if actually inside eN
#     cdef bool inside_eN = True
#     cdef double[:] normal = np.zeros(3)
#     for j, b_i in enumerate(elementBoundariesArray[eN]):
#         normal[0] = elementBoundaryNormalsArray[eN, j, 0]
#         normal[1] = elementBoundaryNormalsArray[eN, j, 1]
#         normal[2] = elementBoundaryNormalsArray[eN, j, 2]
#         bound_bar = elementBoundaryBarycentersArray[b_i]
#         dot = (bound_bar[0]-x[0])*normal[0]+(bound_bar[1]-x[1])*normal[1]+(bound_bar[2]-x[2])*normal[2]
#         if dot < 0:
#             inside_eN = False
#     return eN, nearest_node, inside_eN


def pyCheckOwnedVariable(proteus_index_t variable_nb_local,
                         int rank,
                         proteus_index_t nVariables_owned,
                         proteus_index_t[:] variableNumbering_subdomain2global,
                         proteus_index_t[:] variableOffsets_subdomain_owned):
    return checkOwnedVariable(variable_nb_local=variable_nb_local,
                              rank=rank,
                              nVariables_owned=nVariables_owned,
                              variableNumbering_subdomain2global=variableNumbering_subdomain2global,
                              variableOffsets_subdomain_owned=variableOffsets_subdomain_owned)

cdef tuple checkOwnedVariable(proteus_index_t variable_nb_local,
                              int rank,
                              proteus_index_t nVariables_owned,
                              proteus_index_t[:] variableNumbering_subdomain2global,
                              proteus_index_t[:] variableOffsets_subdomain_owned):
    cdef int nSubdomains = len(variableOffsets_subdomain_owned)-1
    cdef proteus_index_t variable_nb_global
    cdef proteus_index_t new_variable_nb_local
    cdef int new_rank = -2  # initialised as fake rank
    if variable_nb_local >= nVariables_owned:
        # change rank ownership
        variable_nb_global = variableNumbering_subdomain2global[variable_nb_local]
        if not variableOffsets_subdomain_owned[rank] <= variable_nb_global < variableOffsets_subdomain_owned[rank+1]:
            for i in range(nSubdomains+1):
                if variableOffsets_subdomain_owned[i] > variable_nb_global:
                    # changing processor
                    if new_rank == -2:
                        new_rank = i-1
    # getting nearest variable number on new rank
    if new_rank >= 0:
        new_variable_nb_local = variable_nb_global-variableOffsets_subdomain_owned[new_rank]
    else:
        new_rank = rank
        new_variable_nb_local = variable_nb_local
    return new_variable_nb_local, new_rank

# cdef retrieveSolution(double[:]x,
#                       double[:,:] nodeArray,
#                       proteus_index_t[:] nodeStarOffsets,
#                       proteus_index_t[:] nodeStarArray,
#                       proteus_index_t nearest_node,
#                       int rank,
#                       proteus_index_t nNodes_owned,
# )
#                         nearest_node = ms.getLocalNearestNode(coords=x,
#                                                               nodeArray=nodeArray,
#                                                               nodeStarOffsets=nodeStarOffsets,
#                                                               nodeStarArray=nodeStarArray,
#                                                               node=nearest_node)
#                         if nearest_node >= nNodes_owned:
#                             nearest_node_new_rank, new_rank = checkOwnedVariable(variable_nb_local=nearest_node,
#                                                                                  rank=my_rank,
#                                                                                  nVariables_owned=nNodes_owned,
#                                                                                  variablesNumering_subdomain2global=nodeNumbering_subdomain2global,
#                                                                                  variableOffsets_subdomain_owned=nodeOffsets_subdomain_owned)
#                             nodes0_2rank[new_rank] = np.append(nodes0_2rank[new_rank], nearest_node_new_rank)
#                             nPending_disp += 1
#                         else:

#                             nearest_eN = ms.getLocalNearestElementAroundNode(coords=x,
#                                                                              nodeElementOffsets=nodeElementOffsets,
#                                                                              nodeElementsArray=nodeElementsArray,
#                                                                              elementBarycentersArray=elementBarycentersArray,
#                                                                              node=nearest_node)
#                             nPending_disp += 1
#                             if nearest_eN >= nElements_owned:
#                                 nearest_eN_new_rank, new_rank = checkOwnedVariable(variable_nb_local=nearest_eN,
#                                                                                    rank=my_rank,
#                                                                                    nVariables_owned=nENs_owned,
#                                                                                    variablesNumering_subdomain2global=eNNumbering_subdomain2global,
#                                                                                    variableOffsets_subdomain_owned=eNOffsets_subdomain_owned)
#                                 elements2rank[new_rank] = np.append(elements2rank[new_rank], nearest_eN_new_rank)
#                                 nPending_disp += 1
#                             else:
#                                 nearest_eN = ms.getLocalNearestElementIntersection(coords=x,
#                                                                                    elementBoundaryNormalsArray=elementBoundaryNormalsArray,
#                                                                                    elementBoundariesArray=elementBoundariesArray,
#                                                                                    elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
#                                                                                    elementNeighborsArray=elementNeighborsArray,
#                                                                                    elementBarycentersArray=elementBarycentersArray,
#                                                                                    exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
#                                                                                    eN=enearest_eN)
#                                 if nearest_eN >= nElements_owned:
#                                     nearest_eN_new_rank, new_rank = checkOwnedVariable(variable_nb_local=nearest_eN,
#                                                                                        rank=my_rank,
#                                                                                        nVariables_owned=nENs_owned,
#                                                                                        variablesNumering_subdomain2global=eNNumbering_subdomain2global,
#                                                                                        variableOffsets_subdomain_owned=eNOffsets_subdomain_owned)
#                                     elements2rank[new_rank] = np.append(elements2rank[new_rank], nearest_eN_new_rank)
#                                     nPending_disp += 1

























# coords_2rank = coords_2rank,
# tofind_2rank = 
#     cdef dict ls_coords_2rank = {}
#     # nodes to send
#     cdef dict ls_nodes0_2rank = {}
#     # original rank of sending rank
#     cdef dict ls_rank0_2rank = {}
#     # original node number on sending rank
#     cdef dict ls_nearestN_2rank = {}
#     # elements to send
#     cdef dict ls_typeN_2rank = {}
#     # direction to rank (for sliding nodes)
#     cdef dict ls_b_i_2rank = {}
#     # solutions
#     cdef int nPending_disp = 0  # number of pending solutions from other ranks
#     cdef int nPending_disp_total = 0  # number of pending solutions from other ranks
#     cdef double[:,:] nodeArray0 = mesh.nodeArray0
#     cdef double[:,:] elementBarycentersArray0 = mesh.elementBarycentersArray0
#     cdef double[:,:,:] elementBoundaryNormalsArray0 = mesh.elementBoundaryNormalsArray0
#     cdef double[:,:] elementBoundaryBarycentersArray0 = mesh.elementBoundaryBarycentersArray0
#     # declare variables for speed up
#     if comm_size > 1:
#         for rank in range(comm_size):
#             # things to send to other processors to get solution
#             ls_coords_2rank[rank] = np.zeros((0, 3))
#             ls_b_i_2rank[rank] = np.zeros(0, dtype=np.int32)
#             ls_nodes0_2rank[rank] = np.zeros(0, dtype=np.int32)
#             ls_rank0_2rank[rank] = np.zeros(0, dtype=np.int32)
#             ls_nearestN_2rank[rank] = np.zeros(0, dtype=np.int32)
#             ls_typeN_2rank[rank] = np.zeros(0, dtype=np.int32)
#     ls_coords_2rank[my_rank] = coords_2rank[my_rank]
#     ls_2rank[my_rank] = np.zeros(len(coords_2rank[my_rank]))
#     ls_b_i_2rank[my_rank] = b_i_2rank[my_rank]
#     ls_rank0_2rank[my_rank] = rank0_2rank[my_rank]
#     ls_nodes0_2rank[my_rank] = nodes0_2rank[my_rank]
#     ls_typeN_2rank[my_rank] = typeN_2rank[my_rank]
#     ls_nearestN_2rank[my_rank] = nearestN_2rank[my_rank]
# getMPI(string dtype='int',
#        tofind_2rank=ls_2rank,
#        coords_2rank=ls_coords_2rank,
#        nodes0_2rank=ls_nodes0_2rank,
#        b_i_2rank=ls_b_i_2rank,
#        rank0_2rank=ls_rank0_2rank,
#        nearestN_2rank=ls_nearestN_2rank,
#        typeN_2rank=ls_typeN_2rank,
#        femSpace=,
#        tofind_function=pc.getLevelSetValue,
#        nodeArray=nodeArray0,
#        nodeStarOffsets=nodeStarOffsets,
#        nodeStarArray=nodeStarArray,
#        nNodes_owned=nNodes_owned,
#        nodeNumbering_subdomain2global=nodeNumbering_subdomain2global,
#        nodeOffsets_subdomain_owned=nodeOffsets_subdomain_owned,
#        nodeElementOffsets=nodeElementOffsets,
#        nodeElementsArray=nodeElementsArray,
#        elementBarycentersArray=elementBarycentersArray0,
#        elementBoundaryNormalsArray=elementBoundaryNormalsArray0,
#        elementBoundariesArray=elementBoundariesArray,
#        elementBoudnariesBoolArray=elementBoudnariesBoolArray,
# )
# def getMPI(string dtype,
#            dict tofind_2rank,
#            dict coords_2rank,
#            dict nodes0_2rank,
#            dict b_i_2rank,
#            dict rank0_2rank,
#            dict nearestN_2rank,
#            dict typeN_2rank,
#            object femSpace,
#            object tofind_function,
#            # mesh variables
#            double[:,:] nodeArray,
#            proteus_index_t[:] nodeStarOffsets,
#            proteus_index_t[:] nodeStarArray,
#            proteus_index_t nNodes_owned,
#            proteus_index_t[:] nodeNumbering_subdomain2global,
#            proteus_index_t[:] nodeOffsets_subdomain_owned,
#            proteus_index_t[:] nodeElementOffsets,
#            proteus_index_t[:,:] nodeElementsArray,
#            double[:,:] elementBarycentersArray,
#            double[:,:,:] elementBoundaryNormalsArray,
#            proteus_index_t[:,:] elementBoundariesArray,
#            double[:,:] elementBoundaryBarycentersArray,
#            proteus_index_t[:,:] elementBoundaryElementsArray,
#            int[:] exteriorElementBoundariesBoolArray,
# ):
#     if comm_size > 1:
#         # number of pending solutions to send to other processors
#         nPending_disp_total = comm.allreduce(nPending_disp)
#         parallel_steps = -1
#         while nPending_disp_total > 0:
#             parallel_steps += 1
#             nPending_disp = 0
#             sendBack = True
#             # initialize solution from previously found solutions
#             tofind_2doArray = tofind_2rank[my_rank]
#             coords_2doArray = coords_2rank[my_rank]
#             nodes0_2doArray = nodes0_2rank[my_rank]
#             b_i_2doArray = b_i_2rank[my_rank]
#             rank0_2doArray = rank0_2rank[my_rank]
#             nearestN_2doArray = nearestN_2rank[my_rank]
#             typeN_2doArray = typeN_2rank[my_rank]
#             if parallel_steps > 0:
#                 # if coming from this rank, solution was already found
#                 solFound_2doArray = np.ones(len(coords_2doArray), dtype=np.int32)
#             else:
#                 solFound_2doArray = np.zeros(len(coords_2doArray), dtype=np.int32)
#             # wipe dicts out
#             if dtype == 'int':
#                 tofind_2rank[my_rank] = np.zeros(0, dtype=np.int32)
#             elif dtype == 'intVector':
#                 tofind_2rank[my_rank] = np.zeros((0, 3), dtype=np.int32)
#             elif dtype == 'double':
#                 tofind_2rank[my_rank] = np.zeros(0)
#             elif dtype == 'doubleVector':
#                 tofind_2rank[my_rank] = np.zeros((0, 3))
#             coords_2rank[my_rank] = np.zeros((0, 3))
#             nodes0_2rank[my_rank] = np.zeros(0, dtype=np.int32)
#             rank0_2rank[my_rank] = np.zeros(0, dtype=np.int32)
#             nearestN_2rank[my_rank] = np.zeros(0, dtype=np.int32)
#             typeN_2rank[my_rank] = np.zeros(0, dtype=np.int32)
#             b_i_2rank[my_rank] = np.zeros(0, dtype=np.int32)
#             for rank_recv in range(comm.size):
#                 for rank_send in range(comm.size):
#                     if rank_send != rank_recv and rank_send == my_rank:
#                         comm.send(coords_2rank[rank_recv].size, dest=rank_recv, tag=0)
#                         if coords_2rank[rank_recv].size > 0:
#                             # tofind
#                             comm.send(tofind_2rank[rank_recv],  dest=rank_recv, tag=100)
#                             # coords
#                             comm.send(coords_2rank[rank_recv],  dest=rank_recv, tag=1)
#                             # current nearest nodes
#                             comm.send(nodes0_2rank[rank_recv], dest=rank_recv, tag=2)
#                             # rank for original nodes (to send back final solution)
#                             comm.send(rank0_2rank[rank_recv], dest=rank_recv, tag=3)
#                             # node number of nodes on original rank send back final solution)
#                             comm.send(nearestN_2rank[rank_recv], dest=rank_recv, tag=4)
#                             # current nearest elements
#                             comm.send(typeN_2rank[rank_recv], dest=rank_recv, tag=5)
#                             # node direction (if boundary nodes)
#                             comm.send(b_i_2rank[rank_recv], dest=rank_recv, tag=7)
#                             # wipe the dictionary items now that it has been sent
#                         if dtype == 'int':
#                             tofind_2rank[rank_recv] = np.zeros(0, dtype=np.int32)
#                         elif dtype == 'intVector':
#                             tofind_2rank[rank_recv] = np.zeros((0, 3), dtype=np.int32)
#                         elif dtype == 'double':
#                             tofind_2rank[rank_recv] = np.zeros(0)
#                         elif dtype == 'doubleVector':
#                             tofind_2rank[rank_recv] = np.zeros((0, 3))
#                         coords_2rank[rank_recv] = np.zeros((0, 3))
#                         nodes0_2rank[rank_recv] = np.zeros(0, dtype=np.int32)
#                         rank0_2rank[rank_recv] = np.zeros(0, dtype=np.int32)
#                         nearestN_2rank[rank_recv] = np.zeros(0, dtype=np.int32)
#                         typeN_2rank[rank_recv] = np.zeros(0, dtype=np.int32)
#                         b_i_2rank[rank_recv] = np.zeros(0, dtype=np.int32)
#                     elif rank_send!= rank_recv and rank_recv == my_rank:
#                         size = comm.recv(source=rank_send, tag=0)
#                         if size > 0:
#                             tofind_2do = comm.recv(source=rank_send, tag=100)
#                             tofind_2doArray = np.append(tofind_2doArray, tofind_2do, axis=0)
#                             # coords
#                             coords_2do = comm.recv(source=rank_send, tag=1)
#                             coords_2doArray = np.append(coords_2doArray, coords_2do, axis=0)
#                             # node number on original rank
#                             nodes0_2do = comm.recv(source=rank_send, tag=2)
#                             nodes0_2doArray = np.append(nodes0_2doArray, nodes0_2do)
#                             # original rank
#                             rank0_2do = comm.recv(source=rank_send, tag=3)
#                             rank0_2doArray = np.append(rank0_2doArray, rank0_2do)
#                             # current nearest N
#                             nearestN_2do = comm.recv(source=rank_send, tag=4)
#                             nearestN_2doArray = np.append(nearestN_2doArray, nearestN_2do)
#                             # is it node (0) or element (1)
#                             typeN_2do = comm.recv(source=rank_send, tag=5)
#                             typeN_2doArray = np.append(typeN_2doArray, typeN_2do)
#                             b_i_2do = comm.recv(source=rank_send, tag=7)
#                             b_i_2doArray = np.append(b_i_2doArray, b_i_2do)
#                             # sent information means solution was not found
#                             solFound_2doArray = np.append(solFound_2doArray, np.zeros(len(coords_2do), dtype=np.int32))
#                     comm.barrier()
#             nNodes = len(nearestN_2doArray)
#             for iN in range(nNodes):
#                 coords = np.zeros(3)
#                 coords[0] = coords_2doArray[iN, 0]
#                 coords[1] = coords_2doArray[iN, 1]
#                 coords[2] = coords_2doArray[iN, 2]
#                 nearestN = nearestN_2doArray[iN]
#                 typeN = typeN_2doArray[iN]
#                 new_rank = my_rank
#                 b_i_global = b_i_2doArray[iN]
#                 node0 = nodes0_2doArray[iN]
#                 if solFound_2doArray[iN] == 0:
#                     if typeN == 0:
#                         nearestN = ms.pyxGetLocalNearestNode(coords=coords,
#                                                                 nodeArray=nodeArray,
#                                                                 nodeStarOffsets=nodeStarOffsets,
#                                                                 nodeStarArray=nodeStarArray,
#                                                                 node=nearestN)
#                         if nearestN >= nNodes_owned:
#                             result = ms.cyCheckOwnedVariable(variable_nb_local=nearestN,
#                                                                 rank=my_rank,
#                                                                 nVariables_owned=nNodes_owned,
#                                                                 variableNumbering_subdomain2global=nodeNumbering_subdomain2global,
#                                                                 variableOffsets_subdomain_owned=nodeOffsets_subdomain_owned)
#                             nearestN = result[0]
#                             new_rank = result[1]
#                         else:
#                             typeN = 1
#                             nearestN = ms.pyxGetLocalNearestElementAroundNode(coords=coords,
#                                                                                 nodeElementOffsets=nodeElementOffsets,
#                                                                                 nodeElementsArray=nodeElementsArray,
#                                                                                 elementBarycentersArray=elementBarycentersArray,
#                                                                                 node=nearestN)
#                     if typeN == 1:
#                         if nearestN == -1:
#                             if b_i == -1:
#                                 import pdb; pdb.set_trace()
#                             b_i = ms.cyGetLocalVariable(variable_nb_global=b_i_global,
#                                                         rank=my_rank,
#                                                         nVariables_owned=nElementBoundaries_owned,
#                                                         variableNumbering_subdomain2global=elementBoundaryNumbering_subdomain2global,
#                                                         variableOffsets_subdomain_owned=elementBoundaryOffsets_subdomain_owned)
#                             for ii in range(2):
#                                 eeN = elementBoundaryElementsArray[b_i, ii]
#                                 if eeN == -1:
#                                     nearestN = elementBoundaryElementsArray[b_i, ii-1]
#                             if nearestN == -1:
#                                 nearestN = eeN
#                             assert nearestN != -1, 'did not find nearestN! {x}, {y}'.format(x=coords[0], y=coords[1])
#                         starting_coords[0] = elementBarycentersArray[nearestN, 0]
#                         starting_coords[1] = elementBarycentersArray[nearestN, 1]
#                         starting_coords[2] = elementBarycentersArray[nearestN, 2]
#                         result = ms.pyxGetLocalNearestElementIntersection(coords=coords,
#                                                                             starting_coords=starting_coords,
#                                                                             elementBoundaryNormalsArray=elementBoundaryNormalsArray,
#                                                                             elementBoundariesArray=elementBoundariesArray,
#                                                                             elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
#                                                                             elementBoundaryElementsArray=elementBoundaryElementsArray,
#                                                                             exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
#                                                                             eN=nearestN)
#                         nearestN = result[0]
#                         b_i = result[1]
#                         if nearestN >= nElements_owned:
#                             result = ms.cyCheckOwnedVariable(variable_nb_local=nearestN,
#                                                                 rank=my_rank,
#                                                                 nVariables_owned=nElements_owned,
#                                                                 variableNumbering_subdomain2global=elementNumbering_subdomain2global,
#                                                                 variableOffsets_subdomain_owned=elementOffsets_subdomain_owned)
#                             nearestN = result[0]
#                             new_rank = result[1]
#                         if nearestN == -1:
#                             result = ms.cyGetGlobalVariable(variable_nb_local=b_i,
#                                                             nVariables_owned=nElementBoundaries_owned,
#                                                             variableNumbering_subdomain2global=elementBoundaryNumbering_subdomain2global,
#                                                             variableOffsets_subdomain_owned=elementBoundaryOffsets_subdomain_owned)
#                             b_i_global = result[0]
#                             new_rank = result[1]
#                         else:
#                             # directly using nearestN for next time
#                             # (no need for b_i_global)
#                             b_i_global = -1
#                     if new_rank != my_rank:
#                         pending = True
#                     else:
#                         pending = False
#                     if pending is True:
#                         nPending_disp += 1
#                     else:
#                         solFound_2doArray[iN] += 1
#                         inside_eN = True  # checking if actually true
#                         for ii in range(nEbn):
#                             bb_i = elementBoundariesArray[nearestN, ii]
#                             normal[0] = elementBoundaryNormalsArray[nearestN, ii, 0]
#                             normal[1] = elementBoundaryNormalsArray[nearestN, ii, 1]
#                             normal[2] = elementBoundaryNormalsArray[nearestN, ii, 2]
#                             bound_bar = elementBoundaryBarycentersArray[bb_i]
#                             dot = (bound_bar[0]-coords[0])*normal[0]+(bound_bar[1]-coords[1])*normal[1]+(bound_bar[2]-coords[2])*normal[2]
#                             if dot < 0:
#                                 inside_eN = False
#                         if inside_eN:
#                             xi = femSpace.elementMaps.getInverseValue(nearestN, coords)
#                             # line below needs optimisation:
#                             tofind = tofind_function(nearestN, xi)
#                         else:
#                             print('did not find coords', coords[0], coords[1])
#                 tofind_2rank[new_rank] = np.append(tofind_2rank[new_rank], [tofind], axis=0)
#                 coords_2rank[new_rank] = np.append(coords_2rank[new_rank], [coords], axis=0)
#                 nodes0_2rank[new_rank] = np.append(nodes0_2rank[new_rank], nodes0_2doArray[iN])
#                 rank0_2rank[new_rank] = np.append(rank0_2rank[new_rank], rank0_2doArray[iN])
#                 nearestN_2rank[new_rank] = np.append(nearestN_2rank[new_rank], nearestN)
#                 typeN_2rank[new_rank] = np.append(typeN_2rank[new_rank], typeN)
#                 b_i_2rank[new_rank] = np.append(b_i_2rank[new_rank], b_i_global)
#             nPending_disp_total = comm.allreduce(nPending_disp)


#     # SEND NODES POSITION SOLUTION BACK TO ORIGINAL PROCESSORS
#     if sendBack is True:
#         tofind_2doArray = tofind_2rank[my_rank]
#         nodes0_2doArray = nodes0_2rank[my_rank]
#         rank0_2doArray = rank0_2rank[my_rank]
#         for rank in range(comm.size):
#             # things to send to other processors to get solution
#             if dtype == 'int':
#                 tofind_2rank[rank] = np.zeros(0, dtype=np.int32)
#             elif dtype == 'intVector':
#                 tofind_2rank[rank] = np.zeros((0, 3), dtype=np.int32)
#             elif dtype == 'double':
#                 tofind_2rank[rank] = np.zeros(0)
#             elif dtype == 'doubleVector':
#                 tofind_2rank[rank] = np.zeros((0, 3))
#             nodes0_2rank[rank] = np.zeros(0, dtype=np.int32)
#             rank0_2rank[rank] = np.zeros(0, dtype=np.int32)
#         for iN in range(len(coords_2doArray)):
#             if 'Vector' in dtype:
#                 tofind_2rank[rank0_2doArray[iN]] = np.append(tofind_2rank[rank0_2doArray[iN]], [coords_2doArray[iN]], axis=0)
#             else:
#                 tofind_2rank[rank0_2doArray[iN]] = np.append(tofind_2rank[rank0_2doArray[iN]], coords_2doArray[iN])
#             nodes0_2rank[rank0_2doArray[iN]] = np.append(nodes0_2rank[rank0_2doArray[iN]], nodes0_2doArray[iN])
#             rank0_2rank[rank0_2doArray[iN]] = np.append(rank0_2rank[rank0_2doArray[iN]], rank0_2doArray[iN])
#         tofind_2doArray = tofind_2rank[my_rank]
#         nodes0_2doArray = nodes0_2rank[my_rank]
#         for rank_recv in range(comm.size):
#             for rank_send in range(comm.size):
#                 if rank_send != rank_recv and rank_send == my_rank:
#                     comm.send(tofind_2rank[rank_recv].size, dest=rank_recv, tag=0)
#                     if tofind_2rank[rank_recv].size > 0:
#                         # coords
#                         comm.send(tofind_2rank[rank_recv],  dest=rank_recv, tag=1)
#                         # original nodes
#                         comm.send(nodes0_2rank[rank_recv], dest=rank_recv, tag=2)
#                 elif rank_send!= rank_recv and rank_recv == my_rank:
#                     size = comm.recv(source=rank_send, tag=0)
#                     if size > 0:
#                         # coords
#                         tofind_2do = comm.recv(source=rank_send, tag=1)
#                         tofind_2doArray = np.append(tofind_2doArray, tofind_2do, axis=0)
#                         # original nodes
#                         nodes0_2do = comm.recv(source=rank_send, tag=2)
#                         nodes0_2doArray = np.append(nodes0_2doArray, nodes0_2do)
#         return tofind_2do


cdef void findN(double[:] coords,
                  double[:,:] nodeArray,
                  proteus_index_t[:] nodeStarOffsets,
                  proteus_index_t[:] nodeStarArray,
                  proteus_index_t nearestN,
                  int typeN,
                  int my_rank,
                  proteus_index_t nNodes_owned,
                  proteus_index_t[:] nodeNumbering_subdomain2global,
                  proteus_index_t[:] nodeOffsets_subdomain_owned,
                  proteus_index_t[:] nodeElementOffsets,
                  proteus_index_t[:] nodeElementsArray,
                  double[:,:] elementBarycentersArray,
                  double[:,:,:] elementBoundaryNormalsArray,
                  proteus_index_t[:,:] elementBoundariesArray,
                  double[:,:] elementBoundaryBarycentersArray,
                  proteus_index_t[:,:] elementBoundaryElementsArray,
                  int[:] exteriorElementBoundariesBoolArray,
                  proteus_index_t nElements_owned,
                  proteus_index_t[:] elementNumbering_subdomain2global,
                  proteus_index_t[:] elementOffsets_subdomain_owned,
                  proteus_index_t nElementBoundaries_owned,
                  proteus_index_t[:] elementBoundaryNumbering_subdomain2global,
                  proteus_index_t[:] elementBoundaryOffsets_subdomain_owned,
                  proteus_index_t[:,:] elementNodesArray,
                  int[:] nodeMaterialTypes,
                  proteus_index_t nNel,
                  int[3] result_out):
    cdef int[:] result_in
    cdef double[:] starting_coords = np.zeros(3)
    cdef int rank = my_rank # rank of owning processor
    cdef bool stop = False
    if typeN == 0:  # node
        # find closest node to coords
        nearestN = ms.pyxGetLocalNearestNode(coords=coords,
                                             nodeArray=nodeArray,
                                             nodeStarOffsets=nodeStarOffsets,
                                             nodeStarArray=nodeStarArray,
                                             node=nearestN)
        # check if closest node is owned
        if nearestN >= nNodes_owned:
            result_in = ms.cyCheckOwnedVariable(variable_nb_local=nearestN,
                                                rank=my_rank,
                                                nVariables_owned=nNodes_owned,
                                                variableNumbering_subdomain2global=nodeNumbering_subdomain2global,
                                                variableOffsets_subdomain_owned=nodeOffsets_subdomain_owned)
            nearestN = result_in[0]
            rank = result_in[1]
        else:
            # if owned, find closest element barycenter to coords
            nearestN = ms.pyxGetLocalNearestElementAroundNode(coords=coords,
                                                              nodeElementOffsets=nodeElementOffsets,
                                                              nodeElementsArray=nodeElementsArray,
                                                              elementBarycentersArray=elementBarycentersArray,
                                                              node=nearestN)
            typeN = 1
    if typeN == 2:  # element boundary
        # get local number
        nearestN = ms.cyGetLocalVariable(variable_nb_global=nearestN,
                                         rank=my_rank,
                                         nVariables_owned=nElementBoundaries_owned,
                                         variableNumbering_subdomain2global=elementBoundaryNumbering_subdomain2global,
                                         variableOffsets_subdomain_owned=elementBoundaryOffsets_subdomain_owned)
        # get an element from there
        if elementBoundaryElementsArray[nearestN, 0] == -1 or elementBoundaryElementsArray[nearestN, 0] >= nElements_owned:
            nearestN = elementBoundaryElementsArray[nearestN, 1]
            typeN = 1
        elif elementBoundaryElementsArray[nearestN, 1] == -1 or elementBoundaryElementsArray[nearestN, 1] >= nElements_owned:
            nearestN = elementBoundaryElementsArray[nearestN, 0]
            typeN = 1
        else:
            # hack
            # must be on element boundary between processors
            # take arbitrary element sharing boundary
            nearestN = elementBoundaryElementsArray[nearestN, 0]
            typeN = 1

        if nearestN >= nElements_owned:
            result_in = ms.cyCheckOwnedVariable(variable_nb_local=nearestN,
                                                rank=my_rank,
                                                nVariables_owned=nElements_owned,
                                                variableNumbering_subdomain2global=elementNumbering_subdomain2global,
                                                variableOffsets_subdomain_owned=elementOffsets_subdomain_owned)
            nearestN = result_in[0]
            rank = result_in[1]
    if typeN == 1:  # element
        # find closest element through boundary intersection
        starting_coords[0] = elementBarycentersArray[nearestN, 0]
        starting_coords[1] = elementBarycentersArray[nearestN, 1]
        starting_coords[2] = elementBarycentersArray[nearestN, 2]
        result_in = ms.pyxGetLocalNearestElementIntersection(coords=coords,
                                                             starting_coords=starting_coords,
                                                             elementBoundaryNormalsArray=elementBoundaryNormalsArray,
                                                             elementBoundariesArray=elementBoundariesArray,
                                                             elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                                             elementBoundaryElementsArray=elementBoundaryElementsArray,
                                                             exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
                                                             eN=nearestN)
        nearestN = result_in[0]
        # check if owned
        if nearestN >= nElements_owned:
            result_in = ms.cyCheckOwnedVariable(variable_nb_local=nearestN,
                                                rank=my_rank,
                                                nVariables_owned=nElements_owned,
                                                variableNumbering_subdomain2global=elementNumbering_subdomain2global,
                                                variableOffsets_subdomain_owned=elementOffsets_subdomain_owned)
            nearestN = result_in[0]
            rank = result_in[1]
        # return boundary if element was not found
        elif nearestN == -1:
            assert result_in[1] != -1, 'b_i and nearestN cannot be both -1'
            nearestN = result_in[1]  # boundary number
            typeN = 2
            if nearestN >= nElementBoundaries_owned:
                result_in = ms.cyGetGlobalVariable(variable_nb_local=nearestN,
                                                   nVariables_owned=nElementBoundaries_owned,
                                                   variableNumbering_subdomain2global=elementBoundaryNumbering_subdomain2global,
                                                   variableOffsets_subdomain_owned=elementBoundaryOffsets_subdomain_owned)
                nearestN = result_in[0]
                rank = result_in[1]
    # return results
    result_out[0] = nearestN
    result_out[1] = typeN
    result_out[2] = rank

# def mpicomm(dict_2rank, ):
#     comm = Comm.get().comm.tompi4py()
#     comm_size = comm.size
#     cdef dict counts_local = {}
#     cdef 
#     # initialize solution from previously found solutions
#     dict_2doArray = dict_2rank[my_rank]
#     # gather counts info from all processors
#     for rank in range(comm_size):
#         counts_local[rank] = len(dict_2rank[rank])
#     comm.Allgatherv([counts_local, MPI.INT],
#                     [counts_total,
#                      [comm_size for i in range(comm_size)],
#                      [comm_size*i for i in range(comm_size)],
#                      MPI.INT])
#     # calculate array_size for current rank
#     for rank_recv in range(comm_size):
#         array_size = 0
#         for rank in range(comm_size):
#             array_size += counts_total[rank, rank_recv]
#             counts_in[rank] = counts_total[rank, rank_recv]
#             if rank > 0:
#                 displacements_in[rank] = displacements_in[rank-1]+counts_in[rank-1]
#         if my_rank == rank_recv:
#             # initialise dict_2doArray only on receiving processor
#             dict_2doArray = np.zeros((array_size, 12))
#             if parallel_steps > 0:
#                 # if coming from this rank, solution was already found
#                 solFound_2doArray = np.ones(len(dict_2doArray), dtype=np.int32)
#             else:
#                 solFound_2doArray = np.zeros(len(dict_2doArray), dtype=np.int32)
#         # check if another parallel comm is needed
#         if array_size-counts_total[rank_recv, rank_recv] > 0:
#             # -----
#             # get the dict_2doArray (nodes where to retrieve values for arg)
#             datatype = MPI.DOUBLE.Create_contiguous(12).Commit() 
#             comm.Gatherv(dict_2rank[rank_recv],
#                             [dict_2doArray,
#                             tuple(counts_in[i]*dict_2doArray.shape[1] for i in range(comm_size)),
#                             tuple(displacements_in[i]*dict_2doArray.shape[1] for i in range(comm_size)),
#                             MPI.DOUBLE],
#                             root=rank_recv)
#             ncomm += 1
#             if my_rank == 0 and rank_recv ==0:
#                 print([counts_in[i] for i in range(len(counts_in))],
#                     [[dict_2doArray[i, 0], dict_2doArray[i, 1] ] for i in range(len(dict_2doArray))])
#             comm.barrier()
#         else:
#             if rank_recv == my_rank: 
#                 dict_2doArray[:] = dict_2rank[my_rank]


