import proteus
import sys
import numpy
from proteus.Profiling import logEvent
from proteus import MeshTools
from proteus import SimTools
import proteus.Domain
import proteus.kernels

def gatherGlobalMesh(sub, mpicomm, nd, vertices=None):
    """Assemble the whole (partitioned-numbering) mesh on rank 0 from the owned
    entities of every subdomain.

    Returns (nodes, elements, owner, isModelVert, bFaces) as C-contiguous
    float64/int32 arrays on rank 0 and empty ones of the same rank elsewhere.
    owner[i] is the rank that owns global element i. bFaces rows are
    (tag, element, node, ...) over the true domain boundary: faces a rank shares
    with a neighbour (flagged -1 by the partitioner, or interior to the owned
    elements) are left out, since they are not model faces.
    """
    nE, nN = sub.nElements_owned, sub.nNodes_owned
    #the subdomain-to-global numbering lives on the global mesh the subdomain came from
    nodeGid = numpy.asarray(sub.globalMesh.nodeNumbering_subdomain2global)
    elemGid = numpy.asarray(sub.globalMesh.elementNumbering_subdomain2global)
    elements = nodeGid[numpy.asarray(sub.elementNodesArray)[:nE]]
    ext = numpy.asarray(sub.exteriorElementBoundariesArray)
    ebe = numpy.asarray(sub.elementBoundaryElementsArray)[ext, 0]
    keep = (ebe < nE) & (numpy.asarray(sub.elementBoundaryMaterialTypes)[ext] != -1)
    ext, ebe = ext[keep], ebe[keep]
    bFaces = numpy.zeros((len(ext), 2 + nd), dtype=numpy.int32)
    bFaces[:, 0] = numpy.asarray(sub.elementBoundaryMaterialTypes)[ext]
    bFaces[:, 1] = elemGid[ebe]
    bFaces[:, 2:] = nodeGid[numpy.asarray(sub.elementBoundaryNodesArray)[ext][:, :nd]]
    piece = (nodeGid[:nN], numpy.asarray(sub.nodeArray)[:nN],
             elemGid[:nE], elements, mpicomm.rank, bFaces)
    pieces = mpicomm.gather(piece, root=0)
    if mpicomm.rank != 0:
        return (numpy.zeros((0, 3)), numpy.zeros((0, nd + 1), dtype=numpy.int32),
                numpy.zeros(0, dtype=numpy.int32), numpy.zeros(0, dtype=numpy.int32),
                numpy.zeros((0, 2 + nd), dtype=numpy.int32))
    nNodes = sum(len(pc[0]) for pc in pieces)
    nElements = sum(len(pc[2]) for pc in pieces)
    nodes = numpy.zeros((nNodes, 3))
    elems = numpy.zeros((nElements, nd + 1), dtype=numpy.int32)
    owner = numpy.zeros(nElements, dtype=numpy.int32)
    for ngid, xyz, egid, enodes, rank, _ in pieces:
        nodes[ngid] = xyz
        elems[egid] = enodes
        owner[egid] = rank
    bAll = numpy.ascontiguousarray(numpy.concatenate([pc[5] for pc in pieces]), dtype=numpy.int32)
    isModelVert = numpy.zeros(nNodes, dtype=numpy.int32)
    if vertices is not None:
        from scipy import spatial
        tree = spatial.cKDTree(nodes)
        for vertex in vertices:
            v = list(vertex) + [0.0] * (3 - len(vertex))
            isModelVert[tree.query(v)[1]] = 1
    return nodes, numpy.ascontiguousarray(elems), owner, isModelVert, bAll


#try making a class that's build on top of NS_base for these functions 
class PUMI_helper:
     def __init__(self,NS_Base):
        self.solver = NS_Base
        self.rho_0=None
        self.rho_1=None
        self.nu_0=None
        self.nu_1=None
        self.g = None
        self.domain=None
        self.epsFact_density=None
     def __getattr__(self, attr):
        return getattr(self.solver, attr)

     def setProperties(self):
        self.domain = self.pList[0].domain
        self.rho_0 = self.modelList[self.flowIdx].levelModelList[0].coefficients.rho_0
        self.rho_1 = self.modelList[self.flowIdx].levelModelList[0].coefficients.rho_1
        self.nu_0 = self.modelList[self.flowIdx].levelModelList[0].coefficients.nu_0
        self.nu_1 = self.modelList[self.flowIdx].levelModelList[0].coefficients.nu_1
        self.g = self.modelList[self.flowIdx].levelModelList[0].coefficients.g
        self.epsFact_density = self.modelList[self.flowIdx].levelModelList[0].coefficients.epsFact_density

        self.domain.AdaptManager.PUMIAdapter.setAdaptProperties(self.domain.AdaptManager)

     def getModels(self,modelDict):
        #indices are better than models since during adaptation, models are destroyed and recreated
        #using indices avoids having to call this function multiple times
        try:
            self.flowIdx = modelDict['flow']
        except:
            print("flow index not defined properly")
        try:    
            #phaseIdx used for determining material properties
            self.phaseIdx = modelDict['phase'] 
        except:
            print("phase index not defined, using flowIdx")
            self.phaseIdx = self.flowIdx
        try:
            self.correctionsIdxs = modelDict['corrections']
        except:
            self.correctionsIdxs=[]
        
     def reconstructMesh(self,domain,mesh):

        if hasattr(domain,"AdaptManager") and not isinstance(domain,proteus.Domain.PUMIDomain) :

          logEvent("Reconstruct based on Proteus, convert PUMI mesh to Proteus")

          nd = domain.nd
          mpicomm = self.comm.comm.tompi4py() if self.comm.size() > 1 else None
          if mpicomm is not None:
            assert domain.vertices, "model vertices (domain.vertices) were not specified"
            args = gatherGlobalMesh(mesh, mpicomm, nd, domain.vertices)
            domain.AdaptManager.PUMIAdapter.reconstructFromProteusGlobal(nd, *args)
            logEvent("Finished Reconstruction in parallel, terminating program. Rerun with PUMI workflow")
            sys.exit(0)
          from scipy import spatial
          meshVertexTree = spatial.cKDTree(mesh.nodeArray)
          meshVertex2Model= [0]*mesh.nNodes_owned

          assert domain.vertices, "model vertices (domain.vertices) were not specified"
          assert domain.vertexFlags, "model classification (domain.vertexFlags) needs to be specified"

          for idx,vertex in enumerate(domain.vertices):
            if(nd==2 and len(vertex) == 2): #there might be a smarter way to do this
              vertex.append(0.0) #need to make a 3D coordinate
            closestVertex = meshVertexTree.query(vertex)
            meshVertex2Model[closestVertex[1]] = 1

          isModelVert = numpy.asarray(meshVertex2Model).astype("i")

          meshBoundaryConnectivity = numpy.zeros((mesh.nExteriorElementBoundaries_global,2+nd),dtype=numpy.int32)
          for elementBdyIdx in range(len(mesh.exteriorElementBoundariesArray)):
            exteriorIdx = mesh.exteriorElementBoundariesArray[elementBdyIdx]
            meshBoundaryConnectivity[elementBdyIdx][0] = mesh.elementBoundaryMaterialTypes[exteriorIdx]
            meshBoundaryConnectivity[elementBdyIdx][1] = mesh.elementBoundaryElementsArray[exteriorIdx][0]
            meshBoundaryConnectivity[elementBdyIdx][2] = mesh.elementBoundaryNodesArray[exteriorIdx][0]
            meshBoundaryConnectivity[elementBdyIdx][3] = mesh.elementBoundaryNodesArray[exteriorIdx][1]
            if(nd==3):
              meshBoundaryConnectivity[elementBdyIdx][4] = mesh.elementBoundaryNodesArray[exteriorIdx][2]

          domain.AdaptManager.PUMIAdapter.reconstructFromProteus2(mesh.cmesh,isModelVert,meshBoundaryConnectivity)

     def PUMI_reallocate(self,mesh):
        p0 = self.pList[0]
        n0 = self.nList[0]

        nLevels = n0.nLevels
        nLayersOfOverlapForParallel = n0.nLayersOfOverlapForParallel
        parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.element
        domain = p0.domain
        MeshTools.getMeshOptions(domain).setParallelPartitioningType('element')

        logEvent("Generating %i-level mesh from PUMI mesh" % (nLevels,))
        if domain.nd == 3:
          mlMesh = MeshTools.MultilevelTetrahedralMesh(
              0,0,0,skipInit=True,
              nLayersOfOverlap=nLayersOfOverlapForParallel,
              parallelPartitioningType=parallelPartitioningType)
        if domain.nd == 2:
          mlMesh = MeshTools.MultilevelTriangularMesh(
              0,0,0,skipInit=True,
              nLayersOfOverlap=nLayersOfOverlapForParallel,
              parallelPartitioningType=parallelPartitioningType)
        if self.comm.size()==1:
            mlMesh.generateFromExistingCoarseMesh(
                mesh,nLevels,
                nLayersOfOverlap=nLayersOfOverlapForParallel,
                parallelPartitioningType=parallelPartitioningType)
        else:
            mlMesh.generatePartitionedMeshFromPUMI(
                mesh,nLevels,
                nLayersOfOverlap=nLayersOfOverlapForParallel)
        
        #need to remove old mlMesh references to ensure the number of mesh entities is properly updated
        self.mlMesh_nList.clear()
        for p in self.pList:
            self.mlMesh_nList.append(mlMesh)
        if (b"isotropicProteus" in self.domain.AdaptManager.sizeInputs):
            mlMesh.meshList[0].subdomainMesh.size_field = numpy.ones((mlMesh.meshList[0].subdomainMesh.nNodes_global,1),'d')*1.0e-1
        if (b'anisotropicProteus' in self.domain.AdaptManager.sizeInputs):
            mlMesh.meshList[0].subdomainMesh.size_scale = numpy.ones((mlMesh.meshList[0].subdomainMesh.nNodes_global,3),'d')
            mlMesh.meshList[0].subdomainMesh.size_frame = numpy.ones((mlMesh.meshList[0].subdomainMesh.nNodes_global,9),'d')

        #may want to trigger garbage collection here
        self.modelListOld = self.modelList
        logEvent("Allocating models on new mesh")
        self.allocateModels()

     def PUMI_recomputeStructures(self,modelListOld):

        ##This section is to correct any differences in the quadrature point field from the old model

        #Shock capturing lagging needs to be matched

        import copy

        ###Details for solution transfer
        #To get shock capturing lagging correct, the numDiff array needs to be computed correctly with the u^{n} solution.
        #numDiff depends on the PDE residual and can depend on the subgrid error (SGE)
        #the PDE residual depends on the alpha and beta_bdf terms which depend on m_tmp from u^{n-1} as well as VOF or LS fields.
        #getResidual() is used to populate m_tmp, numDiff.
        #The goal is therefore to populate the nodal fields with the old solution, get m_tmp properly and lagged sge properly.
        #Mimic the self stagger with a new loop to repopulate the nodal fields with u^{n} solution. This is necessary because NS relies on the u^{n-1} field for VOF/LS

        ###This loop stores the current solution (u^n) and loads in the previous timestep solution (u^{n-1}
        for m,mOld in zip(self.modelList, modelListOld):
            for lm, lu, lr, lmOld in zip(m.levelModelList, m.uList, m.rList, mOld.levelModelList):
                #lm.coefficients.postAdaptStep() #MCorr needs this at the moment
                lm.u_store = lm.u.copy()
                for ci in range(0,lm.coefficients.nc):
                    lm.u_store[ci] = lm.u[ci].copy()
                lm.dt_store = copy.deepcopy(lm.timeIntegration.dt)
                for ci in range(0,lm.coefficients.nc):
                    lm.u[ci].dof[:] = lm.u[ci].dof_last
                lm.setFreeDOF(lu)

        #All solution fields are now in state u^{n-1} and used to get m_tmp and u_sge
        for m,mOld in zip(self.modelList, modelListOld):
            for lm, lu, lr, lmOld in zip(m.levelModelList, m.uList, m.rList, mOld.levelModelList):
                lm.getResidual(lu,lr)

                #This gets the subgrid error history correct
                if(modelListOld[self.flowIdx].levelModelList[0].stabilization.lag and ((modelListOld[self.flowIdx].levelModelList[0].stabilization.nSteps - 1) > modelListOld[self.flowIdx].levelModelList[0].stabilization.nStepsToDelay) ):
                    self.modelList[self.flowIdx].levelModelList[0].stabilization.nSteps = self.modelList[self.flowIdx].levelModelList[0].stabilization.nStepsToDelay
                    self.modelList[self.flowIdx].levelModelList[0].stabilization.updateSubgridErrorHistory()

                #update the eddy-viscosity history
                lm.calculateAuxiliaryQuantitiesAfterStep()


        #shock capturing depends on m_tmp or m_last (if lagged). m_tmp is modified by mass-correction and is pushed into m_last during updateTimeHistory().
        #This leads to a situation where m_last comes from the mass-corrected solutions so post-step is needed to get this behavior.
        #If adapt is called after the first time-step, then skip the post-step for the old solution
        if( (abs(self.systemStepController.t_system_last - self.tnList[1])> 1e-12 and  abs(self.systemStepController.t_system_last - self.tnList[0])> 1e-12 )
          or self.opts.hotStart):

            for idx in self.correctionsIdxs:
                model = self.modelList[idx]
                self.preStep(model)
                self.setWeakDirichletConditions(model)
                model.stepController.setInitialGuess(model.uList,model.rList)
                solverFailed = model.solver.solveMultilevel(uList=model.uList,
                                                    rList=model.rList,
                                                    par_uList=model.par_uList,
                                                    par_rList=model.par_rList)
                self.postStep(model)

        for m,mOld in zip(self.modelList, modelListOld):
            for lm, lu, lr, lmOld in zip(m.levelModelList, m.uList, m.rList, mOld.levelModelList):
                lm.timeIntegration.postAdaptUpdate(lmOld.timeIntegration)

                if(hasattr(lm.timeIntegration,"dtLast") and lm.timeIntegration.dtLast is not None):
                    lm.timeIntegration.dt = lm.timeIntegration.dtLast

        ###This loop reloads the current solution and the previous solution into proper places
        for m,mOld in zip(self.modelList, modelListOld):
            for lm, lu, lr, lmOld in zip(m.levelModelList, m.uList, m.rList, mOld.levelModelList):
                for ci in range(0,lm.coefficients.nc):
                    lm.u[ci].dof[:] = lm.u_store[ci].dof
                    lm.u[ci].dof_last[:] = lm.u_store[ci].dof_last

                lm.setFreeDOF(lu)
                lm.getResidual(lu,lr)

                #This gets the subgrid error history correct
                if(modelListOld[self.flowIdx].levelModelList[0].stabilization.lag and modelListOld[self.flowIdx].levelModelList[0].stabilization.nSteps > modelListOld[self.flowIdx].levelModelList[0].stabilization.nStepsToDelay):
                    self.modelList[self.flowIdx].levelModelList[0].stabilization.nSteps = self.modelList[self.flowIdx].levelModelList[0].stabilization.nStepsToDelay
                    self.modelList[self.flowIdx].levelModelList[0].stabilization.updateSubgridErrorHistory()
        ###

        ###need to re-distance and mass correct
        if( (abs(self.systemStepController.t_system_last - self.tnList[0])> 1e-12) or self.opts.hotStart  ):
            for idx in self.correctionsIdxs:
                model = self.modelList[idx]
                self.preStep(model)
                self.setWeakDirichletConditions(model)
                model.stepController.setInitialGuess(model.uList,model.rList)
                solverFailed = model.solver.solveMultilevel(uList=model.uList,
                                                            rList=model.rList,
                                                            par_uList=model.par_uList,
                                                            par_rList=model.par_rList)
                self.postStep(model)

            #clsvof needs to call poststep as well once, disc_ICs need to be turned off after initial time step
            try:
                from proteus.kernels import CLSVOF  # call time: proteus.kernels.CLSVOF imports proteus, which imports this module
                if(isinstance(self.modelList[self.phaseIdx].levelModelList[0],CLSVOF.LevelModel)):
                    model = self.modelList[self.phaseIdx]
                    self.postStep(model)
                    model.levelModelList[0].coefficients.disc_ICs = modelListOld[self.phaseIdx].levelModelList[0].coefficients.disc_ICs
            except:
                pass

        for m,mOld in zip(self.modelList, modelListOld):
            for lm, lu, lr, lmOld in zip(m.levelModelList, m.uList, m.rList, mOld.levelModelList):

              lm.timeIntegration.postAdaptUpdate(lmOld.timeIntegration)
              lm.timeIntegration.dt = lm.dt_store
              
        ###Shock capturing update happens with the time history update
              if(lmOld.shockCapturing and lmOld.shockCapturing.nStepsToDelay is not None and lmOld.shockCapturing.nSteps > lmOld.shockCapturing.nStepsToDelay):
                    lm.shockCapturing.nSteps=lm.shockCapturing.nStepsToDelay
                    lm.shockCapturing.updateShockCapturingHistory()


              #update the eddy-viscosity history
              lm.calculateAuxiliaryQuantitiesAfterStep()

     def PUMI2Proteus(self,domain):

        modelListOld = self.modelListOld
        logEvent("Attach auxiliary variables to new models")
        #(cut and pasted from init, need to cleanup)
        self.simOutputList = []
        self.auxiliaryVariables = {}
        self.newAuxiliaryVariables = {}
        if self.simFlagsList is not None:
            for p, n, simFlags, model, index in zip(
                    self.pList,
                    self.nList,
                    self.simFlagsList,
                    self.modelList,
                    list(range(len(self.pList)))):
                self.simOutputList.append(
                    SimTools.SimulationProcessor(
                        flags=simFlags,
                        nLevels=n.nLevels,
                        pFile=p,
                        nFile=n,
                        analyticalSolution=p.analyticalSolution))
                model.simTools = self.simOutputList[-1]

                #Code to refresh attached gauges. The goal is to first purge
                #existing point gauge node associations as that may have changed
                #If there is a line gauge, then all the points must be deleted
                #and remade.
                from collections import OrderedDict
                for av in n.auxiliaryVariables:
                  if hasattr(av,'adapted'):
                    av.adapted=True
                    for point, l_d in av.points.items():
                      if 'nearest_node' in l_d:
                        l_d.pop('nearest_node')
                    if(av.isLineGauge or av.isLineIntegralGauge): #if line gauges, need to remove all points
                      av.points = OrderedDict()
                    if(av.isGaugeOwner):
                      if(self.comm.rank()==0 and not av.file.closed):
                        av.file.close()
                      for item in av.pointGaugeVecs:
                        item.destroy()
                      for item in av.pointGaugeMats:
                        item.destroy()
                      for item in av.dofsVecs:
                        item.destroy()

                      av.pointGaugeVecs = []
                      av.pointGaugeMats = []
                      av.dofsVecs = []
                      av.field_ids=[]
                      av.isGaugeOwner=False
                ##reinitialize auxiliaryVariables
                self.auxiliaryVariables[model.name]= [av.attachModel(model,self.ar[index]) for av in n.auxiliaryVariables]
        else:
            for p,n,s,model,index in zip(
                    self.pList,
                    self.nList,
                    self.sList,
                    self.modelList,
                    list(range(len(self.pList)))):
                self.simOutputList.append(SimTools.SimulationProcessor(pFile=p,nFile=n))
                model.simTools = self.simOutputList[-1]
                model.viewer = Viewers.V_base(p,n,s)
                self.auxiliaryVariables[model.name]= [av.attachModel(model,self.ar[index]) for av in n.auxiliaryVariables]
        for avList in list(self.auxiliaryVariables.values()):
            for av in avList:
                av.attachAuxiliaryVariables(self.auxiliaryVariables)

        #just added, need to initialize structures
        for model in self.modelList:
            logEvent("Auxiliary variable calculate_init for model %s" % (model.name,))
            for av in self.auxiliaryVariables[model.name]:
                #gauges do not need to have aux variables recomputed as this would add additional lines to the output unnecessarily
                if hasattr(av,"isGaugeOwner"):
                    pass
                else:
                    av.calculate_init()

        logEvent("Transfering fields from PUMI to Proteus")
        for m in self.modelList:
          for lm in m.levelModelList:
            coef = lm.coefficients
            if coef.vectorComponents is not None:
              vector=numpy.zeros((lm.mesh.nNodes_global,3),'d')
              domain.AdaptManager.PUMIAdapter.transferFieldToProteus(
                     coef.vectorName.encode('utf-8'), vector)
              for vci in range(len(coef.vectorComponents)):
                lm.u[coef.vectorComponents[vci]].dof[:] = vector[:,vci]
              domain.AdaptManager.PUMIAdapter.transferFieldToProteus(
                     coef.vectorName.encode('utf-8')+b"_old", vector)
              for vci in range(len(coef.vectorComponents)):
                lm.u[coef.vectorComponents[vci]].dof_last[:] = vector[:,vci]
              domain.AdaptManager.PUMIAdapter.transferFieldToProteus(
                     coef.vectorName.encode('utf-8')+b"_old_old", vector)
              for vci in range(len(coef.vectorComponents)):
                lm.u[coef.vectorComponents[vci]].dof_last_last[:] = vector[:,vci]

              del vector
            for ci in range(coef.nc):
              if coef.vectorComponents is None or \
                 ci not in coef.vectorComponents:
                scalar=numpy.zeros((lm.mesh.nNodes_global,1),'d')
                domain.AdaptManager.PUMIAdapter.transferFieldToProteus(
                    coef.variableNames[ci].encode('utf-8'), scalar)
                lm.u[ci].dof[:] = scalar[:,0]
                domain.AdaptManager.PUMIAdapter.transferFieldToProteus(
                    coef.variableNames[ci].encode('utf-8')+b"_old", scalar)
                lm.u[ci].dof_last[:] = scalar[:,0]
                domain.AdaptManager.PUMIAdapter.transferFieldToProteus(
                    coef.variableNames[ci].encode('utf-8')+b"_old_old", scalar)
                lm.u[ci].dof_last_last[:] = scalar[:,0]

                del scalar

        if(b'ibm' in self.domain.AdaptManager.sizeInputs):
            scalar=numpy.zeros((lm.mesh.nNodes_global,1),'d')
            domain.AdaptManager.PUMIAdapter.transferFieldToProteus(
                b'phi', scalar)
            self.modelList[self.flowIdx].levelModelList[0].coefficients.phi_s[:]=scalar[:,0]
            del scalar

        logEvent("Attaching models on new mesh to each other")
        for m,ptmp,mOld in zip(self.modelList, self.pList, modelListOld):
            for lm, lu, lr, lmOld in zip(m.levelModelList, m.uList, m.rList,mOld.levelModelList):
                #save_dof=[]
                #for ci in range(lm.coefficients.nc):
                #    save_dof.append( lm.u[ci].dof.copy())
                #    lm.u[ci].dof_last = lm.u[ci].dof.copy()
                lm.setFreeDOF(lu)
                #for ci in range(lm.coefficients.nc):
                #    assert((save_dof[ci] == lm.u[ci].dof).all())
                lm.calculateSolutionAtQuadrature()
                lm.timeIntegration.tLast = lmOld.timeIntegration.tLast
                lm.timeIntegration.t = lmOld.timeIntegration.t
                lm.timeIntegration.dt = lmOld.timeIntegration.dt
                assert(lmOld.timeIntegration.tLast == lm.timeIntegration.tLast)
                assert(lmOld.timeIntegration.t == lm.timeIntegration.t)
                assert(lmOld.timeIntegration.dt == lm.timeIntegration.dt)
            m.stepController.dt_model = mOld.stepController.dt_model
            m.stepController.t_model = mOld.stepController.t_model
            m.stepController.t_model_last = mOld.stepController.t_model_last
            m.stepController.substeps = mOld.stepController.substeps

        #if first time-step / initial adapt & not hotstarted
        if(abs(self.systemStepController.t_system_last - self.tnList[0])< 1e-12 and not self.opts.hotStart):
            for index,p,n,m,simOutput in zip(range(len(self.modelList)),self.pList,self.nList,self.modelList,self.simOutputList):
                if p.initialConditions is not None:
                    logEvent("Setting initial conditions for "+p.name)
                    m.setInitialConditions(p.initialConditions,self.tnList[0])


        #Attach models and do sample residual calculation. The results are usually irrelevant.
        #What's important right now is to re-establish the relationships between data structures.
        #The necessary values will be written in later.
        for m,ptmp,mOld in zip(self.modelList, self.pList, modelListOld):
            logEvent("Attaching models to model "+ptmp.name)
            m.attachModels(self.modelList)
        logEvent("Evaluating residuals and time integration")

        for m,ptmp,mOld in zip(self.modelList, self.pList, modelListOld):
            for lm, lu, lr, lmOld in zip(m.levelModelList, m.uList, m.rList, mOld.levelModelList):
                lm.timeTerm=True
                lm.getResidual(lu,lr)
                lm.timeIntegration.initializeTimeHistory(resetFromDOF=True)
                lm.initializeTimeHistory()
                lm.timeIntegration.initializeSpaceHistory()
                lm.getResidual(lu,lr)
                #lm.estimate_mt() #function is empty in all models
            assert(m.stepController.dt_model == mOld.stepController.dt_model)
            assert(m.stepController.t_model == mOld.stepController.t_model)
            assert(m.stepController.t_model_last == mOld.stepController.t_model_last)
            logEvent("Initializing time history for model step controller")
            if(not self.opts.hotStart):
              m.stepController.initializeTimeHistory()
        #p0.domain.initFlag=True #For next step to take initial conditions from solution, only used on restarts

            #m.stepController.initializeTimeHistory()
        #domain.initFlag=True #For next step to take initial conditions from solution, only used on restarts

        self.systemStepController.modelList = self.modelList
        self.systemStepController.exitModelStep = {}
        self.systemStepController.controllerList = []
        for model in self.modelList:
            self.systemStepController.exitModelStep[model] = False
            if model.levelModelList[-1].timeIntegration.isAdaptive:
                self.systemStepController.controllerList.append(model)
                self.systemStepController.maxFailures = model.stepController.maxSolverFailures

        #this sets the timeIntegration time, which might be unnecessary for restart
        if(self.opts.hotStart):
          self.systemStepController.stepSequence=[(self.systemStepController.t_system,m) for m in self.systemStepController.modelList]
        else:
          self.systemStepController.choose_dt_system()

        #Don't do anything if this is the initial adapt
        if(abs(self.systemStepController.t_system_last - self.tnList[0])> 1e-12  or
          (abs(self.systemStepController.t_system_last - self.tnList[0]) < 1e-12 and self.opts.hotStart)):
            self.PUMI_recomputeStructures(modelListOld)

            #something different is needed for initial conditions
            #do nothing if archive sequence step because there will be an archive
            #if self.archiveFlag != ArchiveFlags.EVERY_SEQUENCE_STEP:
            #  self.tCount+=1
            #  for index,model in enumerate(self.modelList):
            #    #import pdb; pdb.set_trace()
            #    self.archiveSolution(
            #      model,
            #      index,
            #      #self.systemStepController.t_system_last+1.0e-6)
            #      self.systemStepController.t_system)

            #This logic won't account for if final step doesn't match frequency or if adapt isn't being called
            if((self.PUMIcheckpointer.frequency>0) and ( (domain.AdaptManager.PUMIAdapter.nAdapt()!=0) and (domain.AdaptManager.PUMIAdapter.nAdapt() % self.PUMIcheckpointer.frequency==0 ) or self.systemStepController.t_system_last==self.tnList[-1])):

              self.PUMIcheckpointer.checkpoint(self.systemStepController.t_system_last)

        #del modelListOld to free up memory
        del modelListOld
        import gc;
        gc.disable()
        gc.collect()
        self.comm.barrier()

     def PUMI_transferFields(self):

        domain = self.domain
       
        logEvent("Copying coordinates to PUMI")
        
        #might be an arbitrary choice in terms of which model to use, but guaranteed that the 0th model exists
        domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(b"coordinates",
            self.modelList[0].levelModelList[0].mesh.nodeArray)

        #I want to compute the density and viscosity arrays here
        #arrays are length = number of elements and will correspond to density at center of element
        rho_transfer = numpy.zeros((self.modelList[0].levelModelList[0].mesh.nElements_owned),'d')      
        nu_transfer = numpy.zeros((self.modelList[0].levelModelList[0].mesh.nElements_owned),'d')      
        
        #obtain the level-set or vof value at each element centroid
        #pass through heaviside function to get material property
        #only need to do this if using combined adapt

        if(b'error_vms' in self.domain.AdaptManager.sizeInputs or b'error_erm' in self.domain.AdaptManager.sizeInputs):
            self.computeElementMaterials(rho_transfer,nu_transfer)

        #put the solution field as uList
        #VOF and LS needs to reset the u.dof array for proper transfer
        #but needs to be returned to the original form if not actually adapting....be careful with the following statements, unsure if this doesn't break something else
        import copy
        for m in self.modelList:
            for lm in m.levelModelList:
                lm.u_store = lm.u.copy()
                for ci in range(0,lm.coefficients.nc):
                    lm.u_store[ci] = lm.u[ci].copy()

        #for each type of phase tracking model, this will need to be done
        from proteus.kernels import VOF, NCLS  # call time, as above
        for m in self.modelList:
            for lm in m.levelModelList:
                try:
                    if(isinstance(lm,VOF.LevelModel) or 
                        isinstance(lm,NCLS.LevelModel)
                    ):
                        lm.setUnknowns(m.uList[0])
                except:
                    pass

        logEvent("Copying DOF and parameters to PUMI")
        for m in self.modelList:
          for lm in m.levelModelList:
            coef = lm.coefficients
            if coef.vectorComponents is not None:
              vector=numpy.zeros((lm.mesh.nNodes_global,3),'d')
              for vci in range(len(coef.vectorComponents)):
                vector[:,vci] = lm.u[coef.vectorComponents[vci]].dof[:]

              domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                  coef.vectorName.encode('utf-8'), vector)
              #Transfer dof_last
              for vci in range(len(coef.vectorComponents)):
                vector[:,vci] = lm.u[coef.vectorComponents[vci]].dof_last[:]
              domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                     coef.vectorName.encode('utf-8')+b"_old", vector)
              #Transfer dof_last_last
              for vci in range(len(coef.vectorComponents)):
                vector[:,vci] = lm.u[coef.vectorComponents[vci]].dof_last_last[:]
                domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                     coef.vectorName.encode('utf-8')+b"_old_old", vector)

              del vector
            for ci in range(coef.nc):
              if coef.vectorComponents is None or \
                 ci not in coef.vectorComponents:
                scalar=numpy.zeros((lm.mesh.nNodes_global,1),'d')
                scalar[:,0] = lm.u[ci].dof[:]
                domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                    coef.variableNames[ci].encode('utf-8'), scalar)

                #Transfer dof_last
                scalar[:,0] = lm.u[ci].dof_last[:]
                domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                     coef.variableNames[ci].encode('utf-8')+b"_old", scalar)
                #Transfer dof_last_last
                scalar[:,0] = lm.u[ci].dof_last_last[:]
                domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                     coef.variableNames[ci].encode('utf-8')+b"_old_old", scalar)

                del scalar

        #if chrono solid phi_s field is present this needs to have a corresponding mirror transfer back function
        #also should maybe rely on a separate sf config scheme
        if(b'ibm' in self.domain.AdaptManager.sizeInputs):
            scalar=numpy.zeros((lm.mesh.nNodes_global,1),'d')
            scalar[:,0] = self.modelList[self.flowIdx].levelModelList[0].coefficients.phi_s[:]
            domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                b'phi', scalar)

            del scalar

        #Get Physical Parameters
        #Can we do this in a problem-independent  way?

        g = numpy.asarray(self.g)

        #This condition is to account for adapting before the simulation started
        if(hasattr(self,"tn")):
            #deltaT = self.tn-self.tn_last
            #is actually the time step for next step, self.tn and self.tn_last refer to entries in tnList
            #deltaT = self.systemStepController.dt_system 
            deltaT=self.modelList[self.flowIdx].levelModelList[0].timeIntegration.dtLast
            T_current = self.systemStepController.t_system_last
            deltaT_next = self.systemStepController.dt_system 
        else:
            deltaT = 0
            deltaT_next = 0.0
            T_current = 0.0

        domain.AdaptManager.PUMIAdapter.transferPropertiesToPUMI(rho_transfer,nu_transfer,g,deltaT,deltaT_next,T_current,self.epsFact_density)

     def PUMI_estimateError(self):
        """
        Estimate the error using the classical element residual method by
        Ainsworth and Oden and generates a corresponding error field.
        """

        adaptMeshNow = False
        domain = self.domain

        if (hasattr(domain, 'AdaptManager') and
            domain.AdaptManager.PUMIAdapter.adaptMesh() and
            self.so.useOneMesh): #and
            #self.nSolveSteps%domain.AdaptManager.PUMIAdapter.numAdaptSteps()==0):
            if (b"isotropicProteus" in self.domain.AdaptManager.sizeInputs):
                domain.AdaptManager.PUMIAdapter.transferFieldToPUMI("proteus_size_user",
                                                       self.modelList[0].levelModelList[0].mesh.size_field)
            if (b'anisotropicProteus' in self.domain.AdaptManager.sizeInputs):
                #Insert a function to define the size_scale/size_frame fields here.
                #For a given vertex, the i-th size_scale is roughly the desired edge length along the i-th direction specified by the size_frame
                for i in range(len(self.modelList[0].levelModelList[0].mesh.size_scale)):
                  self.modelList[0].levelModelList[0].mesh.size_scale[i,0] =  1e-1
                  self.modelList[0].levelModelList[0].mesh.size_scale[i,1] =  (self.modelList[0].levelModelList[0].mesh.nodeArray[i,1]/0.584)*1e-1
                  for j in range(3):
                    for k in range(3):
                      if(j==k):
                        self.modelList[0].levelModelList[0].mesh.size_frame[i,3*j+k] = 1.0
                      else:
                        self.modelList[0].levelModelList[0].mesh.size_frame[i,3*j+k] = 0.0
                self.modelList[0].levelModelList[0].mesh.size_scale
                domain.AdaptManager.PUMIAdapter.transferFieldToPUMI("proteus_sizeScale", self.modelList[0].levelModelList[0].mesh.size_scale)
                domain.AdaptManager.PUMIAdapter.transferFieldToPUMI("proteus_sizeFrame", self.modelList[0].levelModelList[0].mesh.size_frame)

            self.PUMI_transferFields()


            logEvent("Estimate Error")
            if(b"error_erm" in self.domain.AdaptManager.sizeInputs):
              errorTotal= domain.AdaptManager.PUMIAdapter.get_local_error()
              if(domain.AdaptManager.PUMIAdapter.willAdapt()):
                adaptMeshNow=True
                logEvent("Need to Adapt")
            if(b"error_vms" in self.domain.AdaptManager.sizeInputs):
              errorTotal = domain.AdaptManager.PUMIAdapter.get_VMS_error()
              if(domain.AdaptManager.PUMIAdapter.willAdapt()):
                adaptMeshNow=True
                logEvent("Need to Adapt")
              if(self.nSolveSteps <= 5): #the first few time steps are ignored for adaptivity
                adaptMeshNow=False
            if(b"interface" in self.domain.AdaptManager.sizeInputs or b"ibm" in self.domain.AdaptManager.sizeInputs):
              if(domain.AdaptManager.PUMIAdapter.willInterfaceAdapt()):
                  adaptMeshNow=True
                  logEvent("Need to Adapt")
                  logEvent('numSolveSteps %f ' % self.nSolveSteps)
            if(b'meshQuality' in self.domain.AdaptManager.sizeInputs):
              minQual = domain.AdaptManager.PUMIAdapter.getMinimumQuality()
              logEvent('The quality is %f ' % (minQual**(1./3.)))
              #adaptMeshNow=True
              if(minQual**(1./3.)<0.25):
                adaptMeshNow=True
                logEvent("Need to Adapt")

              if (self.auxiliaryVariables['rans2p'][0].subcomponents[0].__class__.__name__== 'ProtChBody'):
                sphereCoords = numpy.asarray(self.auxiliaryVariables['rans2p'][0].subcomponents[0].position)
                domain.AdaptManager.PUMIAdapter.updateSphereCoordinates(sphereCoords)
                logEvent("Updated the sphere coordinates %f %f %f" % (sphereCoords[0],sphereCoords[1],sphereCoords[2]))
              else:
                sys.exit("Haven't been implemented code yet to cover this behavior.")
            
            if(b'error_spr' in self.domain.AdaptManager.sizeInputs and
               self.nSolveSteps % domain.AdaptManager.numAdaptSteps == 0):
                adaptMeshNow=True
                logEvent("Need to Adapt")
            if(b'pseudo' in self.domain.AdaptManager.sizeInputs):
                adaptMeshNow=True

            #if not adapting need to return data structures to original form which was modified by PUMI_transferFields()
            if(adaptMeshNow == False):
                for m in self.modelList:
                    for lm in m.levelModelList:
                        lm.u[0].dof[:]=lm.u_store[0].dof

        return adaptMeshNow

     def PUMI_adaptMesh(self,inputString=b""):
        """
        Uses a computed error field to construct a size field and adapts
        the mesh using SCOREC tools (a.k.a. MeshAdapt)
        """
        ##

        domain = self.domain

        if(hasattr(self,"nSolveSteps")):
          logEvent("h-adapt mesh by calling AdaptAdaptManager.PUMIAdapter at step %s" % self.nSolveSteps)
        if(b"pseudo" in self.domain.AdaptManager.sizeInputs):
            logEvent("Testing solution transfer and restart feature of adaptation. No actual mesh adaptation!")
        else:
            domain.AdaptManager.PUMIAdapter.adaptPUMIMesh(inputString)

        logEvent("Converting PUMI mesh to Proteus")
        #ibaned: PUMI conversion #2
        #TODO: this code is nearly identical to
        #PUMI conversion #1, they should be merged
        #into a function
        if domain.nd == 3:
          mesh = MeshTools.TetrahedralMesh()
        else:
          mesh = MeshTools.TriangularMesh()

        mesh.convertFromPUMI(domain,
                             domain.AdaptManager.PUMIAdapter,
                             domain.faceList,
                             domain.regList,
                             parallel = self.comm.size() > 1,
                             dim = domain.nd)

        self.PUMI_reallocate(mesh)
        self.PUMI2Proteus(domain)
      ##chitak end Adapt

     def hotstartWithPUMI(self):
         #Call restart functions
         logEvent("Converting PUMI mesh to Proteus")
         if self.pList[0].domain.nd == 3:
             mesh = MeshTools.TetrahedralMesh()
         else:
             mesh = MeshTools.TriangularMesh()

         mesh.convertFromPUMI(self.pList[0].domain.AdaptManager.PUMIAdapter,
                             self.pList[0].domain.faceList,
                             self.pList[0].domain.regList,
                             parallel = self.comm.size() > 1,
                             dim = self.pList[0].domain.nd)

         if(self.pList[0].domain.checkpointInfo==None):
             sys.exit("Need to specify checkpointInfo file in inputs")
         else:
             self.PUMIcheckpointer.DecodeModel(self.pList[0].domain.checkpointInfo)

         self.PUMI_reallocate(mesh) #need to double check if this call is necessaryor if it can be simplified to a shorter call
         PUMI2Proteus(self,self.pList[0].domain)

     def initialAdapt(self):
        if (hasattr(self.pList[0].domain.AdaptManager, 'PUMIAdapter') and
            self.pList[0].domain.AdaptManager.PUMIAdapter.adaptMesh() and
            (b"pseudo" in self.domain.AdaptManager.sizeInputs or b"interface" in self.domain.AdaptManager.sizeInputs or b"ibm" in self.domain.AdaptManager.sizeInputs) and
            self.so.useOneMesh and not self.opts.hotStart):

            self.PUMI_transferFields()
            logEvent("Initial Adapt before Solve")
            self.PUMI_adaptMesh(b"interface")
            self.PUMI_transferFields()
            logEvent("Initial Adapt 2 before Solve")
            self.PUMI_adaptMesh(b"interface")

     def computeElementMaterials(self,rho_transfer,nu_transfer):

        #get quadrature points at element centroid and evaluate at shape functions
        from proteus import Quadrature
        transferQpt = Quadrature.SimplexGaussQuadrature(self.domain.nd,1)
        qpt_centroid = numpy.asarray([transferQpt.points[0]])
        materialSpace = self.nList[0].femSpaces[0](self.modelList[0].levelModelList[0].mesh.subdomainMesh,self.domain.nd)
        materialSpace.getBasisValuesRef(qpt_centroid)


        from proteus.ctransportCoefficients import smoothedHeaviside
        
        IEN = self.modelList[self.phaseIdx].levelModelList[0].u[0].femSpace.dofMap.l2g
        for (eID, dofs) in enumerate(IEN):
            phi_val = 0.0
            for idx in range(len(dofs)):
                phi_val += materialSpace.psi[0][idx]*self.modelList[self.phaseIdx].levelModelList[0].u[0].dof[dofs[idx]]
            
            #heaviside
            h_phi=0.0;
            for idx in range(len(dofs)):
                h_phi += (materialSpace.psi[0][idx])*(self.modelList[self.phaseIdx].levelModelList[0].mesh.nodeDiametersArray[dofs[idx]]);
            eps_rho = self.epsFact_density*h_phi
            smoothed_phi_val = smoothedHeaviside(eps_rho,phi_val)

            rho_transfer[eID] = (1.0-smoothed_phi_val)*self.rho_0 + smoothed_phi_val*self.rho_1
            nu_transfer[eID] = (1.0-smoothed_phi_val)*self.nu_0 + smoothed_phi_val*self.nu_1
