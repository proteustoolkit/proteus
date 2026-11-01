"""
A hierarchy of classes for managing complete numerical solution implementations

.. inheritance-diagram:: proteus.NumericalSolution
   :parts: 1
"""
import os
import numpy
from subprocess import check_call, check_output

from . import LinearSolvers
from . import NonlinearSolvers
from . import MeshTools
from . import Profiling
from . import Transport
from . import SimTools
from . import Archiver
from . import Viewers
from .Archiver import ArchiveFlags
from . import Domain
from .MeshAdaptPUMI import Checkpoint,AdaptHelper
from .Profiling import logEvent

# Global to control whether the kernel starting is active.
embed_ok = True


class NS_base(object):  # (HasTraits):
    r"""
    The base class for managing the numerical solution of  PDE's.

    The constructor must build all the objects required by a numerical
    method to approximate the solution over a sequence of time intervals.

    calculateSolution(runName) carries out the numerical solution.

    .. graphviz::

       digraph NumericalSolutionHasA {
       node [shape=record, fontname=Helvetica, fontsize=12];
       NS   [label="NumericalSolution" URL="\ref NumericalSolution", style="filled", fillcolor="gray"];
       mList [label="MultilevelTranportModel [n]" URL="\ref proteus::Transport::MultilevelTransport"];
       nsList [label="NonLinearSolver [n] " URL="\ref proteus::NonLinearSolver"];
       lsList [label="LinearSolver [n] " URL="\ref proteus::LinearSolver"];
       pList [label="Problem Specification [n]" URL="\ref proteus::default_p"];
       nList [label="Numerics Specifiation [n]" URL="\ref proteus::default_n"];
       sList [label="Output Specification [n]" URL="\ref proteus::SimTools"];
       so [label="Coupling Specification " URL="\ref proteus::SO_base"];
       ar [label="Archiver" URL="\ref proteus::AR_base"];
       NS -> pList [arrowhead="normal", style="dashed", color="purple"];
       NS -> nList [arrowhead="normal", style="dashed", color="purple"];
       NS -> so [arrowhead="normal", style="dashed", color="purple"];
       NS -> sList [arrowhead="normal", style="dashed", color="purple"];
       NS -> mList [arrowhead="normal", style="dashed", color="purple"];
       NS -> nsList [arrowhead="normal", style="dashed", color="purple"];
       NS -> lsList [arrowhead="normal", style="dashed", color="purple"];
       NS -> ar [arrowhead="normal", style="dashed", color="purple"];
       }
    """

    def __init__(self,so,pList,nList,sList,opts,simFlagsList=None,TwoPhaseFlow=False):
        from . import Comm
        comm=Comm.get()
        self.comm=comm
        message = "Initializing NumericalSolution for "+so.name+"\n System includes: \n"
        for p in pList:
            message += p.name+"\n"
        logEvent(message)
        #: SplitOperator initialize file
        self.so=so
        #: List of physics initialize files
        self.pList=pList
        #: List of numerics initialize files
        self.nList=nList
        #: Dictionary of command line arguments
        self.opts=opts
        self.simFlagsList=simFlagsList
        self.TwoPhaseFlow=TwoPhaseFlow
        self.timeValues={}
        Profiling.memory("Memory used before initializing"+so.name)
        memBase = Profiling.memLast #save current memory usage for later
        if not so.useOneMesh:
            so.useOneArchive=False
        logEvent("Setting Archiver(s)")

        if hasattr(self.so,"fastArchive"):
            self.fastArchive = self.so.fastArchive
        else:
            self.fastArchive = False

        if so.useOneArchive:
            self.femSpaceWritten={}
            tmp  = Archiver.XdmfArchive(opts.dataDir,so.name,useTextArchive=opts.useTextArchive,
                                        gatherAtClose=opts.gatherArchive,hotStart=opts.hotStart,
                                        useGlobalXMF=(not opts.subdomainArchives),
                                        global_sync=opts.global_sync)
            if self.fastArchive==True:
                self.ar = dict([(0,tmp)])
            else:
                self.ar = dict([(i,tmp) for i in range(len(self.pList))])
        elif len(self.pList) == 1:
            self.ar = {0:Archiver.XdmfArchive(opts.dataDir,so.name,useTextArchive=opts.useTextArchive,
                                              gatherAtClose=opts.gatherArchive,hotStart=opts.hotStart)} #reuse so.name if possible
        else:
            self.ar = dict([(i,Archiver.XdmfArchive(opts.dataDir,p.name,useTextArchive=opts.useTextArchive,
                                                    gatherAtClose=opts.gatherArchive,hotStart=opts.hotStart)) for i,p in enumerate(self.pList)])
        #by default do not save quadrature point info
        self.archive_q                 = dict([(i,False) for i in range(len(self.pList))]);
        self.archive_ebq_global        = dict([(i,False) for i in range(len(self.pList))]);
        self.archive_ebqe              = dict([(i,False) for i in range(len(self.pList))]);
        self.archive_pod_residuals = dict([(i,False) for i in range(len(self.pList))]);
        if simFlagsList is not None:
            assert len(simFlagsList) == len(self.pList), "len(simFlagsList) = %s should be %s " % (len(simFlagsList),len(self.pList))
            for index in range(len(self.pList)):
                if 'storeQuantities' in simFlagsList[index]:
                    for quant in [a for a in simFlagsList[index]['storeQuantities'] if a is not None]:
                        recType = quant.split(':')
                        if len(recType) > 1 and recType[0] == 'q':
                            self.archive_q[index] = True
                        elif len(recType) > 1 and recType[0] == 'ebq_global':
                            self.archive_ebq_global[index] = True
                        elif len(recType) > 1 and recType[0] == 'ebqe':
                            self.archive_ebqe[index] = True
                        #
                        elif recType[0] == 'pod_residuals':
                            self.archive_pod_residuals[index]=True
                        else:
                            logEvent("Warning Numerical Solution storeQuantity = %s not recognized won't archive" % quant)
                    #
                #
            #
        #
        logEvent("Setting up MultilevelMesh")
        mlMesh_nList = []

        if so.useOneMesh:
            logEvent("Building one multilevel mesh for all models")
            nListForMeshGeneration=[nList[0]]
            pListForMeshGeneration=[pList[0]]
        else:
            logEvent("Building seperate meshes for each model")
            nListForMeshGeneration=nList
            pListForMeshGeneration=pList
        for p,n in zip(pListForMeshGeneration,nListForMeshGeneration):
            if opts.hotStart:
                p.genMesh = False
                logEvent("Hotstarting, using existing mesh "+p.name)
            else:
                logEvent("Generating mesh for "+p.name)
            #support for old-style domain input
            if p.domain is None:
                if p.nd == 1:
                    p.domain = Domain.RectangularDomain(L=p.L[:1],
                                                        x=p.x0[:1],
                                                        name=p.name)
                elif p.nd == 2:
                    if p.polyfile is not None:
                        p.domain = Domain.PlanarStraightLineGraphDomain(fileprefix=p.polyfile,name=p.polyfile)
                    elif p.meshfile != None:
                        p.domain = Domain.Mesh2DMDomain(p.meshfile)
                    else:
                        p.domain = Domain.RectangularDomain(L=p.L[:2],
                                                            x=p.x0[:2],
                                                            name=p.name)
                elif p.nd == 3:
                    if p.polyfile is not None:
                        p.domain = Domain.PiecewiseLinearComplexDomain(fileprefix=p.polyfile,name=p.polyfile)
                    elif p.meshfile is not None:
                        p.domain = Domain.Mesh3DMDomain(p.meshfile)
                    else:
                        p.domain = Domain.RectangularDomain(L=p.L[:3],
                                                            x=p.x0[:3],
                                                            name=p.name)
                else:
                    raise RuntimeError("No support for domains in more than three dimensions")

                #Ensuring backwards compatibility with isolated mesh generation routines
                p.domain.MeshOptions.nnx, p.domain.MeshOptions.nny,p.domain.MeshOptions.nnz= n.nnx,n.nny,n.nnz
                p.domain.MeshOptions.nn = n.nn 
                p.domain.MeshOptions.nLevels = n.nLevels
                p.domain.MeshOptions.hex = n.hex
                p.domain.MeshOptions.quad = n.quad
                p.domain.MeshOptions.triangleFlag=n.triangleFlag
                p.domain.MeshOptions.triangleOptions=n.triangleOptions
                p.domain.MeshOptions.genMesh=p.genMesh
                p.domain.MeshOptions.parallelPartitioningType=n.parallelPartitioningType
                p.domain.MeshOptions.nLayersOfOverlapForParallel=n.nLayersOfOverlapForParallel
        try:
            mlMesh = self.pList[0].domain.mesh
        except:
            mlMesh = MeshTools.generateMesh(self.pList[0],self.nList[0],opts.generatePartitionedMeshFromFiles)

        theMesh = mlMesh.meshList[0].subdomainMesh
        
        theDomain = self.pList[0].domain

        if(hasattr(theDomain,'AdaptManager')):
            # attach the checkpointer
            if isinstance(theDomain,Domain.PUMIDomain):
                self.PUMIcheckpointer = Checkpoint.Checkpointer(self, theDomain.checkpointFrequency)
            self.AdaptHelper = AdaptHelper.PUMI_helper(self)
            self.AdaptHelper.reconstructMesh(theDomain,theMesh)
            self.AdaptHelper.getModels(theDomain.AdaptManager.modelDict)


        #TODO: this needs to be isolated MeshAdaptPUMI
        if so.useOneMesh:
            try:
                if (b"isotropicProteus" in theDomain.AdaptManager.sizeInputs):
                    mlMesh.meshList[0].subdomainMesh.size_field = numpy.ones((mlMesh.meshList[0].subdomainMesh.nNodes_global,1),'d')*1.0e-1
                if (b"anisotropicProteus" in theDomain.AdaptManager.sizeInputs):
                    mlMesh.meshList[0].subdomainMesh.size_scale = numpy.ones((mlMesh.meshList[0].subdomainMesh.nNodes_global,3),'d')
                    mlMesh.meshList[0].subdomainMesh.size_frame = numpy.ones((mlMesh.meshList[0].subdomainMesh.nNodes_global,9),'d')
            except:
                pass
        #
        mlMesh_nList.append(mlMesh)
        if so.useOneMesh:
            for p in pList[1:]: mlMesh_nList.append(mlMesh)

        if opts.viewMesh:
            logEvent("Attempting to visualize mesh")
            try:
                from proteusGraphical import vtkViewers
                vtkViewers.ViewMesh(mlMesh.meshList[0],viewMaterialTypes=True)
                vtkViewers.ViewBoundaryMesh(mlMesh.meshList[0],viewBoundaryMaterialTypes=True)
            except:
                logEvent("NumericalSolution ViewMesh failed for coarse mesh")
        for l in range(n.nLevels):
            try:
                logEvent(mlMesh.meshList[l].meshInfo())
            except:
                logEvent("meshInfo() method not implemented for this mesh type")
            if opts.viewMesh and opts.viewLevels and l > 0:
                logEvent("Attempting to visualize mesh")
                try:
                    from proteusGraphical import vtkViewers
                    vtkViewers.ViewMesh(mlMesh.meshList[l],title="mesh level %s " % l,
                                        viewMaterialTypes=True)
                    vtkViewers.ViewBoundaryMesh(mlMesh.meshList[l],title="boundary mesh level %s " % l,
                                                viewBoundaryMaterialTypes=True)
                except:
                    logEvent("NumericalSolution ViewMesh failed for mesh level %s" % l)

        Profiling.memory("Mesh")
        from collections import OrderedDict
        self.modelSpinUp = OrderedDict()
        for p in pList:
            p.coefficients.opts = self.opts
            if p.coefficients.sdInfo == {}:
                for ci,ckDict in p.coefficients.diffusion.items():
                    for ck in list(ckDict.keys()):
                        if (ci,ck) not in p.coefficients.sdInfo:
                            p.coefficients.sdInfo[(ci,ck)] = (numpy.arange(start=0,stop=p.nd**2+1,step=p.nd,dtype='i'),
                                                              numpy.array([list(range(p.nd)) for row in range(p.nd)],dtype='i').flatten())
                            logEvent("Numerical Solution Sparse diffusion information key "+repr((ci,ck))+' = '+repr(p.coefficients.sdInfo[(ci,ck)]))
        self.sList = sList
        self.mlMesh_nList = mlMesh_nList
        self.allocateModels()
        #collect models to be used for spin up
        for index in so.modelSpinUpList:
            self.modelSpinUp[index] = self.modelList[index]
        logEvent("Finished setting up models and solvers")
        if self.opts.save_dof:
            for m in self.modelList:
                for lm in m.levelModelList:
                    for ci in range(lm.coefficients.nc):
                        lm.u[ci].dof_last = lm.u[ci].dof.copy()
        self.archiveFlag= so.archiveFlag
        logEvent("Setting up SimTools for "+p.name)
        self.simOutputList = []
        self.auxiliaryVariables = {}
        if self.simFlagsList is not None:
            for p,n,simFlags,model,index in zip(pList,nList,simFlagsList,self.modelList,list(range(len(pList)))):
                self.simOutputList.append(SimTools.SimulationProcessor(flags=simFlags,nLevels=n.nLevels,
                                                                       pFile=p,nFile=n,
                                                                       analyticalSolution=p.analyticalSolution))
                model.simTools = self.simOutputList[-1]
                self.auxiliaryVariables[model.name]= [av.attachModel(model,self.ar[index]) for av in n.auxiliaryVariables]
        else:
            for p,n,s,model,index in zip(pList,nList,sList,self.modelList,list(range(len(pList)))):
                self.simOutputList.append(SimTools.SimulationProcessor(pFile=p,nFile=n))
                model.simTools = self.simOutputList[-1]
                model.viewer = Viewers.V_base(p,n,s)
                self.auxiliaryVariables[model.name]= [av.attachModel(model,self.ar[index]) for av in n.auxiliaryVariables]
        for avList in list(self.auxiliaryVariables.values()):
            for av in avList:
                av.attachAuxiliaryVariables(self.auxiliaryVariables)
        for model in self.modelList:
            logEvent("Auxiliary variable calculate_init for model %s" % (model.name,))
            for av in self.auxiliaryVariables[model.name]:
                av.calculate_init()
        logEvent(Profiling.memory("NumericalSolution memory",className='NumericalSolution',memSaved=memBase))
        if so.tnList is None:
            logEvent("Building tnList from model = "+pList[0].name+" nDTout = "+repr(nList[0].nDTout))
            self.tnList=[float(n)*nList[0].T/float(nList[0].nDTout)
                         for n in range(nList[0].nDTout+1)]
        else:
            logEvent("Using tnList from so = "+so.name)
            self.tnList = so.tnList
        logEvent("Time sequence"+repr(self.tnList))
        logEvent("NAHeader Num Time Steps "+repr(len(self.tnList)-1))
        logEvent("Setting "+so.name+" systemStepController to object of type "+str(so.systemStepControllerType))
        self.systemStepController = so.systemStepControllerType(self.modelList,stepExact=so.systemStepExact)
        self.systemStepController.setFromOptions(so)
        logEvent("Finished NumericalSolution initialization")

    def allocateModels(self):
        self.modelList=[]
        self.lsList=[]
        self.nlsList=[]

        for p,n,s,mlMesh,index \
            in zip(self.pList,self.nList,self.sList,self.mlMesh_nList,list(range(len(self.pList)))):

            if self.so.needEBQ_GLOBAL:
                n.needEBQ_GLOBAL = True

            if self.so.needEBQ:
                n.needEBQ = True
            ## \todo clean up tolerances: use rtol_u,atol_u and rtol_res, atol_res; allow scaling by mesh diameter
            ## \todo pass in options = (p,n) instead of using monster ctor signature
            tolList=[]
            linTolList=[]
            for l in range(n.nLevels):
                #if mlMesh.meshList[l].hasGeometricInfo != True:
                #    mlMesh.meshList[l].computeGeometricInfo()

                #fac = (mlMesh.meshList[l].h/mlMesh.meshList[0].h)**2
                fac = 1.0
                tolList.append(n.tolFac*fac)
                linTolList.append(n.linTolFac*fac)

            logEvent("Setting up MultilevelTransport for "+p.name)
            model \
                = Transport.MultilevelTransport(p,
                                                n,
                                                mlMesh,
                                                OneLevelTransportType=p.LevelModelType)
            self.modelList.append(model)

            model.name = p.name
            logEvent("Setting "+model.name+" stepController to "+str(n.stepController))
            model.stepController = n.stepController(model,n)

            Profiling.memory("MultilevelTransport for "+p.name)
            logEvent("Setting up MultilevelLinearSolver for"+p.name)

            #allow options database to set model specific parameters?
            linear_solver_options_prefix = None
            if 'linear_solver_options_prefix' in dir(n):
                linear_solver_options_prefix = n.linear_solver_options_prefix

            (multilevelLinearSolver,directSolverFlag) = LinearSolvers.multilevelLinearSolverChooser(
                linearOperatorList = model.jacobianList,
                par_linearOperatorList = model.par_jacobianList,
                multilevelLinearSolverType = n.multilevelLinearSolver,
                computeSolverRates=n.computeLinearSolverRates,
                printSolverInfo=n.printLinearSolverInfo,
                levelLinearSolverType = n.levelLinearSolver,
                computeLevelSolverRates=n.computeLevelLinearSolverRates,
                printLevelSolverInfo=n.printLevelLinearSolverInfo,
                smootherType = n.linearSmoother,
                computeSmootherRates=n.computeLinearSmootherRates,
                printSmootherInfo=n.printLinearSmootherInfo,
                prolongList = model.meshTransfers.prolongList,
                restrictList = model.meshTransfers.restrictList,
                connectivityListList = [model.levelModelList[l].sparsityInfo for l in range(n.nLevels)],
                relativeToleranceList = linTolList,
                absoluteTolerance = n.l_atol_res,
                solverMaxIts = n.linearSolverMaxIts,
                solverConvergenceTest=n.linearSolverConvergenceTest,
                cycles=n.linearWCycles,
                preSmooths=n.linearPreSmooths,
                postSmooths=n.linearPostSmooths,
                ##\todo logic needs to handle element boundary partition too
                parallelUsesFullOverlap=(n.nLayersOfOverlapForParallel > 0 or n.parallelPartitioningType == MeshTools.MeshParallelPartitioningTypes.node),
                par_duList=model.par_duList,
                solver_options_prefix=linear_solver_options_prefix,
                computeEigenvalues = n.computeEigenvalues,
                linearSmootherOptions = n.linearSmootherOptions)
            self.lsList.append(multilevelLinearSolver)
            Profiling.memory("MultilevelLinearSolver for "+p.name)
            logEvent("Setting up MultilevelNonLinearSolver for "+p.name)
            self.nlsList.append(NonlinearSolvers.multilevelNonlinearSolverChooser(
                model.levelModelList,
                model.jacobianList,
                model.par_jacobianList,
                duList=model.duList,
                par_duList=model.par_duList,
                multilevelNonlinearSolverType = n.multilevelNonlinearSolver,
                computeSolverRates=n.computeNonlinearSolverRates,
                solverConvergenceTest=n.nonlinearSolverConvergenceTest,
                levelSolverConvergenceTest=n.levelNonlinearSolverConvergenceTest,
                printSolverInfo=n.printNonlinearSolverInfo,
                relativeToleranceList = tolList,
                absoluteTolerance = n.nl_atol_res,
                levelNonlinearSolverType=n.levelNonlinearSolver,
                computeLevelSolverRates=n.computeNonlinearLevelSolverRates,
                printLevelSolverInfo=n.printNonlinearLevelSolverInfo,
                smootherType = n.nonlinearSmoother,
                computeSmootherRates=n.computeNonlinearSmootherRates,
                printSmootherInfo=n.printNonlinearSmootherInfo,
                preSmooths=n.nonlinearPreSmooths,
                postSmooths=n.nonlinearPostSmooths,
                cycles=n.nonlinearWCycles,
                maxSolverIts=n.maxNonlinearIts,
                prolong_bcList = model.meshTransfers.prolong_bcListDict,
                restrict_bcList = model.meshTransfers.restrict_bcListDict,
                restrict_bcSumList = model.meshTransfers.restrict_bcSumListDict,
                prolongList = model.meshTransfers.prolongList,
                restrictList = model.meshTransfers.restrictList,
                restrictionRowSumList = model.meshTransfers.restrictSumList,
                connectionListList=[model.levelModelList[l].sparsityInfo for l in range(n.nLevels)],
                linearSolverList=multilevelLinearSolver.solverList,
                linearDirectSolverFlag=directSolverFlag,
                solverFullNewtonFlag=n.fullNewtonFlag,
                levelSolverFullNewtonFlag=n.fullNewtonFlag,
                smootherFullNewtonFlag=n.fullNewtonFlag,
                EWtol=n.useEisenstatWalker,
                maxLSits=n.maxLineSearches,
                #\todo need to add logic in multilevel NL solver chooser to account for numerical method's stencil as well
                parallelUsesFullOverlap=(n.nLayersOfOverlapForParallel > 0 or n.parallelPartitioningType == MeshTools.MeshParallelPartitioningTypes.node),
                nonlinearSolverNorm = n.nonlinearSolverNorm,
                nonlinearSolverUNorm = getattr(n, 'nonlinearSolverUNorm', None),
                relativeToleranceDu = getattr(n, 'nl_rtol_du', 1.0e-4),
                absoluteToleranceDu = getattr(n, 'nl_atol_du', 1.0e-16)))
            model.solver=self.nlsList[-1]
            model.viewer = Viewers.V_base(p,n,s)
            Profiling.memory("MultilevelNonlinearSolver for"+p.name)

    ## compute the solution

    def calculateSolution(self,runName):
        """ Cacluate the PDEs numerical solution.

        Parameters
        ----------
        runName : str
            A name for the calculated solution.
        """

        logEvent("Setting initial conditions",level=0)
        for index,p,n,m,simOutput in zip(list(range(len(self.modelList))),self.pList,self.nList,self.modelList,self.simOutputList):
            if self.opts.hotStart:
                logEvent("Setting initial conditions from hot start file for "+p.name)
                tCount = int(self.ar[index].tree.getroot()[-1][-1][-1][0].attrib['Name'])
                offset=0
                while tCount > 0:
                    time = float(self.ar[index].tree.getroot()[-1][-1][-1-offset][0].attrib['Value'])
                    if time <= self.opts.hotStartTime:
                        break
                    else:
                        tCount -=1
                        offset +=1
                self.ar[index].n_datasets = tCount + 1
                if len(self.ar[index].tree.getroot()[-1][-1]) - offset - 1 > 0:
                    dt = time - float(self.ar[index].tree.getroot()[-1][-1][-1-offset-1][0].attrib['Value'])
                else:
                    logEvent("Not enough steps in hot start file set set dt, setting dt to 1.0")
                    dt = 1.0
                logEvent("Hot starting from time step t = "+repr(time))
                #the number of nodes in an adapted mesh is not necessarily going to be the same as that of the solution field when archived...but it's not important because things should be bookkept correctly later on
                #if not isinstance(p.domain,Domain.PUMIDomain):

                m.stepController.t_model_last=time
                for lm,lu,lr in zip(m.levelModelList,m.uList,m.rList):
                    for cj in range(lm.coefficients.nc):

                        if not isinstance(self.pList[0].domain,Domain.PUMIDomain):
                          lm.u[cj].femSpace.readFunctionXdmf(self.ar[index],lm.u[cj],tCount)
                        lm.setFreeDOF(lu)
                        lm.timeIntegration.tLast = time
                        lm.timeIntegration.t = time
                        lm.timeIntegration.dt = dt
                self.tCount = tCount
            elif p.initialConditions is not None:
                logEvent("Setting initial conditions for "+p.name)
                m.setInitialConditions(p.initialConditions,self.tnList[0])
                #It's only safe to calculate the solution and solution
                #gradients because the models aren't attached yet
                for lm in m.levelModelList:
                    lm.calculateSolutionAtQuadrature()
            else:
                logEvent("No initial conditions provided for model "+p.name)
        if self.opts.hotStart:
            if time >= self.tnList[-1] - 1.0e-5:
                logEvent("Modifying time interval to be tnList[-1] + tnList since tnList hasn't been modified already")
                ndtout = len(self.tnList)
                self.tnList = [time + i for i in self.tnList]
                self.tnList.insert(1, 0.9*self.tnList[0]+0.1*self.tnList[1])

            else:
                tnListNew=[time]
                for n,t in enumerate(self.tnList):
                    if time < t-1.0e-8:
                        tnListNew.append(t)
                self.tnList=tnListNew
                logEvent("Hotstarting, new tnList is"+repr(self.tnList))
        else:
            self.tCount=0#time step counter


        logEvent("Attaching models and running spin-up step if requested")
        self.firstStep = True ##\todo get rid of firstStep flag in NumericalSolution if possible?
        spinup = []
        if (not self.opts.hotStart) or (not self.so.skipSpinupOnHotstart):
            for index,m in self.modelSpinUp.items():
                spinup.append((self.pList[index],self.nList[index],m,self.simOutputList[index]))
        for index,m in enumerate(self.modelList):
            logEvent("Attaching models to model "+m.name)
            m.attachModels(self.modelList)
            if index not in self.modelSpinUp:
                spinup.append((self.pList[index],self.nList[index],m,self.simOutputList[index]))
        for m in self.modelList:
            for lm,lu,lr in zip(m.levelModelList,
                                m.uList,
                                m.rList):
                #calculate the coefficients, any explicit-in-time
                #terms will be wrong
                lm.getResidual(lu,lr)
        for p,n,m,simOutput in spinup:
            logEvent("Attaching models to model "+p.name)
            m.attachModels(self.modelList)
            if m in list(self.modelSpinUp.values()):
                logEvent("Spin-Up Estimating initial time derivative and initializing time history for model "+p.name)
                #now the models are attached so we can calculate the coefficients
                for lm,lu,lr in zip(m.levelModelList,
                                    m.uList,
                                    m.rList):
                    #calculate the coefficients, any explicit-in-time
                    #terms will be wrong
                    lm.getResidual(lu,lr)
                    #post-process velocity
                    #lm.calculateAuxiliaryQuantitiesAfterStep()
                    #load in the initial conditions into time
                    #integration history to get explict terms right
                    lm.initializeTimeHistory()
                    lm.timeIntegration.initializeSpaceHistory()
                    #recalculate coefficients
                    lm.getResidual(lu,lr)
                    #calculate consistent time derivative
                    lm.estimate_mt()
                    #post-process velocity
                    lm.calculateAuxiliaryQuantitiesAfterStep()
                logEvent("Spin-Up Choosing initial time step for model "+p.name)
                m.stepController.initialize_dt_model(self.tnList[0],self.tnList[1])
                #mwf what if user wants spin-up to be over (t_0,t_1)?

                if m.stepController.stepExact and m.stepController.t_model_last != self.tnList[1]:
                    logEvent("Spin-up step exact called for model %s" % (m.name,),level=3)
                    m.stepController.stepExact_model(self.tnList[1])
                logEvent("Spin-Up Initializing time history for model step controller")

                m.stepController.initializeTimeHistory()
                m.stepController.setInitialGuess(m.uList,m.rList)

                solverFailed = m.solver.solveMultilevel(uList=m.uList,
                                                        rList=m.rList,
                                                        par_uList=m.par_uList,
                                                        par_rList=m.par_rList)
                Profiling.memory("solver.solveMultilevel")
                if solverFailed:
                    logEvent("Spin-Up Step Failed t=%12.5e, dt=%12.5e for model %s, CONTINUING ANYWAY!" %  (m.stepController.t_model,
                                                                                                     m.stepController.dt_model,
                                                                                                     m.name))

                else:
                    if n.restrictFineSolutionToAllMeshes:
                        logEvent("Using interpolant of fine mesh an all meshes")
                        self.restrictFromFineMesh(m)
                    self.postStep(m)
                    self.systemStepController.modelStepTaken(m,self.tnList[0])
                    logEvent("Spin-Up Step Taken, Model step t=%12.5e, dt=%12.5e for model %s" % (m.stepController.t_model,
                                                                                             m.stepController.dt_model,
                                                                                             m.name))

        for p,n,m,simOutput,index in zip(self.pList,self.nList,self.modelList,self.simOutputList,list(range(len(self.pList)))):
            if not self.opts.hotStart:
                logEvent("Archiving initial conditions")
                self.archiveInitialSolution(m,index)
            else:
                self.ar[index].domain = self.ar[index].tree.find("Domain")
            #if(not hasattr(self.pList[0].domain,'PUMIMesh') and not self.opts.hotStart):
            self.initializeViewSolution(m)
            logEvent("Estimating initial time derivative and initializing time history for model "+p.name)
            #now the models are attached so we can calculate the coefficients
            for lm,lu,lr in zip(m.levelModelList,
                                m.uList,
                                m.rList):
                if self.opts.save_dof:
                    import copy
                    lm.u_store = lm.u.copy()
                    for ci in range(0,lm.coefficients.nc):
                        lm.u_store[ci] = lm.u[ci].copy()

                    lm.setUnknowns(m.uList[0])
                    for ci in range(lm.coefficients.nc):
                        lm.u[ci].dof_last_last[:] = lm.u[ci].dof_last
                        lm.u[ci].dof_last[:] = lm.u[ci].dof
                        lm.u[ci].dof[:] = lm.u_store[ci].dof

                #calculate the coefficients, any explicit terms will be wrong
                lm.timeTerm=False
                lm.getResidual(lu,lr)
                #post-process velocity
                #lm.calculateAuxiliaryQuantitiesAfterStep()
                #load in the initial conditions into time integration history to get explict terms right
                lm.initializeTimeHistory()
                lm.timeIntegration.initializeSpaceHistory()
                #recalculate  coefficients with the explicit terms correct
                lm.getResidual(lu,lr)
                #post-process velocity
                #lm.calculateAuxiliaryQuantitiesAfterStep()
                lm.timeTerm=True
                #calculate consistent
                lm.estimate_mt()
                #
            logEvent("Choosing initial time step for model "+p.name)
            m.stepController.initialize_dt_model(self.tnList[0],self.tnList[1])
            #recalculate  with all terms ready
            for lm,lu,lr in zip(m.levelModelList,
                                m.uList,
                                m.rList):
                lm.getResidual(lu,lr)
            logEvent("Initializing time history for model step controller")
            m.stepController.initializeTimeHistory()
        self.systemStepController.initialize_dt_system(self.tnList[0],self.tnList[1]) #may reset other dt's
        logEvent("Starting time stepping",level=0)
        systemStepFailed=False
        stepFailed=False


        if(hasattr(self.pList[0].domain,"AdaptManager")):
            self.AdaptHelper.setProperties()
        #### Perform an initial adapt after applying initial conditions ####
        # The initial adapt is based on interface, but will eventually be generalized to any sort of initialization
        # Needs to be placed here at this time because of the post-adapt routine requirements

        if(hasattr(self.pList[0].domain,"AdaptManager")):
            self.AdaptHelper.initialAdapt()

        #NS_base has a fairly complicated time stepping loop structure
        #to accommodate fairly general split operator approaches. The
        #outer loop is over user defined time intervals for the entire
        #system of models. The next loop is over potentially adaptive
        #steps for the entire system. The next loop is for iterations
        #over the entire system such as for interactive split
        #operator. The next loop is for a sequence of model steps such
        #as for alternating split operator or fractional step
        #schemes. The next loop is for each model to step, potentially
        #adaptively, to the time in the stepSequence. Lastly there is
        #a loop for substeps(stages).

       # for p,n,m,simOutput,index in zip(self.pList,self.nList,self.modelList,self.simOutputList,range(len(self.pList))):
       #   for lm,lu,lr in zip(m.levelModelList,
       #                         m.uList,
       #                         m.rList):
       #     lm.getResidual(lu,lr)
       #     print "Initial Field \n %s" % lu
       #     print "Initial Residual \n %s" % lr
       #     print "Min / Max residual %s / %s" %(lr.min(),lr.max())

        self.nSequenceSteps = 0
        nSequenceStepsLast=self.nSequenceSteps # prevent archiving the same solution twice
        self.nSolveSteps=0


        self.opts.save_dof = True
        if self.opts.save_dof:
            import copy
            for m in self.modelList:
                for lm in m.levelModelList:
                    lm.u_store = lm.u.copy()
                    for ci in range(0,lm.coefficients.nc):
                        lm.u_store[ci] = lm.u[ci].copy()

                    lm.setUnknowns(m.uList[0])
                    for ci in range(lm.coefficients.nc):
                        lm.u[ci].dof_last_last[:] = lm.u[ci].dof_last
                        lm.u[ci].dof_last[:] = lm.u[ci].dof
                        lm.u[ci].dof[:] = lm.u_store[ci].dof

        #### If PUMI and hotstarting then decode info and proceed with restart ####
        #### This has to be done after the dof histories are saved because DOF histories are already present on the mesh ####

        if (hasattr(self.pList[0].domain, 'AdaptManager') and self.opts.hotStart):
          f = open(self.pList[0].domain.checkpointInfo, 'r')
          import json
          previousInfo = json.load(f)
          f.close()
          if(previousInfo["checkpoint_status"]=="endsystem"):
            self.AdaptHelper.hotstartWithPUMI()
            self.opts.hotStart = False
            #Need to clean mesh for output again
            self.pList[0].domain.AdaptManager.cleanMesh()
        ####

        import time
        if hasattr(self.so,'measureSpeedOfCode'):
            measureSpeed = self.so.measureSpeedOfCode
        elif hasattr(n,'measureSpeedOfCode'):
            measureSpeed = n.measureSpeedOfCode
        else:
            measureSpeed = False
        #
        append=False
        startToMeasureSpeed = False
        numTimeSteps=0
        start=0

        for (self.tn_last,self.tn) in zip(self.tnList[:-1],self.tnList[1:]):
            logEvent("==============================================================",level=0)
            logEvent("Solving over interval [%12.5e,%12.5e]" % (self.tn_last,self.tn),level=0)
            logEvent("==============================================================",level=0)
            if measureSpeed and startToMeasureSpeed and numTimeSteps==0 and self.comm.isMaster():
                start = time.time()
                logEvent("**********... start measuring speed of the code",level=1)
            #
#            logEvent("NumericalAnalytics Time Step " + `self.tn`, level=0)

            if self.systemStepController.stepExact and self.systemStepController.t_system_last != self.tn:
                self.systemStepController.stepExact_system(self.tn)
            while self.systemStepController.t_system_last < self.tn:
                logEvent("System time step t=%12.5e, dt=%12.5e" % (self.systemStepController.t_system,
                                                              self.systemStepController.dt_system),level=3)

                while (not self.systemStepController.converged() and
                       not systemStepFailed):


                    if (hasattr(self.pList[0].domain, 'AdaptManager') and self.opts.hotStart):
                      self.hotstartWithPUMI()
                      self.opts.hotStart = False
                      #Need to clean mesh for output again
                      self.pList[0].domain.AdaptManager.cleanMesh()

                    #This should be the only place dofs are saved otherwise there might be a double-shift for last_last
                    self.opts.save_dof = True
                    if self.opts.save_dof:
                        import copy
                        for m in self.modelList:
                            for lm in m.levelModelList:
                                lm.u_store = lm.u.copy()
                                for ci in range(lm.coefficients.nc):
                                    lm.u_store[ci] = lm.u[ci].copy()
                                lm.setUnknowns(m.uList[0])
                                for ci in range(lm.coefficients.nc):
                                    lm.u[ci].dof_last_last[:] = lm.u[ci].dof_last
                                    lm.u[ci].dof_last[:] = lm.u[ci].dof
                                for ci in range(lm.coefficients.nc):
                                    lm.u[ci].dof[:] = lm.u_store[ci].dof
                                #lm.setFreeDOF(m.uList[0])
                        logEvent("saving previous velocity dofs %s" % self.nSolveSteps)


                    logEvent("Split operator iteration %i" % (self.systemStepController.its,),level=3)
                    self.nSequenceSteps += 1
                    for (self.t_stepSequence,model) in self.systemStepController.stepSequence:

                        logEvent("NumericalAnalytics Model %s " % (model.name), level=5)
                        logEvent("Model: %s" % (model.name),level=1)
                        logEvent("NumericalAnalytics Time Step " + repr(self.t_stepSequence), level=7)
                        logEvent("Fractional step %12.5e for model %s" % (self.t_stepSequence,model.name),level=3)
                        for m in model.levelModelList:
                            if m.movingDomain and m.tLast_mesh != self.systemStepController.t_system_last:
                                m.t_mesh = self.systemStepController.t_system_last
                                m.updateAfterMeshMotion()
                                m.tLast_mesh = m.t_mesh

                        self.preStep(model)
                        self.setWeakDirichletConditions(model)

                        stepFailed = False
                        if model.stepController.stepExact and model.stepController.t_model_last != self.t_stepSequence:
                            logEvent("Step exact called for model %s" % (model.name,),level=3)
                            model.stepController.stepExact_model(self.t_stepSequence)
                        while (model.stepController.t_model_last < self.t_stepSequence and
                               not stepFailed and
                               not self.systemStepController.exitModelStep[model]):
                            logEvent("Model step t=%12.5e, dt=%12.5e for model %s" % (model.stepController.t_model,
                                                                                 model.stepController.dt_model,
                                                                                 model.name),level=3)
                            for self.tSubstep in model.stepController.substeps:

                                logEvent("Model substep t=%12.5e for model %s" % (self.tSubstep,model.name),level=3)
                                #TODO: model.stepController.substeps doesn't seem to be updated after a solver failure unless model.stepController.stepExact is true
                                logEvent("Model substep t=%12.5e for model %s model.timeIntegration.t= %12.5e" % (self.tSubstep,model.name,model.levelModelList[-1].timeIntegration.t),level=3)



                                model.stepController.setInitialGuess(model.uList,model.rList)
                                solverFailed = model.solver.solveMultilevel(uList=model.uList,
                                                                            rList=model.rList,
                                                                            par_uList=model.par_uList,
                                                                            par_rList=model.par_rList)
                                if Profiling.logLevel > 10:
                                    model.viewJacobian(file_prefix='dump_{0:s}_{1:s}_{2:s}'.format(runName, model.name,repr(self.tSubstep)))

                                Profiling.memory("solver.solveMultilevel")
                                if self.opts.wait:
                                    input("Hit any key to continue")
                                if solverFailed:
                                    break
                                else:
                                    if n.restrictFineSolutionToAllMeshes:
                                        logEvent("Using interpolant of fine mesh an all meshes")
                                        self.restrictFromFineMesh(model)
                                    model.stepController.updateSubstep()
                            #end model substeps
                            if solverFailed:
                                logEvent("Step failed due to solver failure")
                                stepFailed = not self.systemStepController.retryModelStep_solverFailure(model)
                            elif model.stepController.errorFailure():
                                logEvent("Step failed due to error failure")
                                stepFailed = not self.systemStepController.retryModelStep_errorFailure(model)
                            else:
                                #set up next step
                                self.systemStepController.modelStepTaken(model,self.t_stepSequence)
                                logEvent("Step Taken, t_stepSequence= %s Model step t=%12.5e, dt=%12.5e for model %s" % (self.t_stepSequence,
                                                                                                                         model.stepController.t_model,
                                                                                                                         model.stepController.dt_model,
                                                                                                                         model.name),level=3)

                        #end model step
                        if stepFailed:
                            logEvent("Sequence step failed")
                            if not self.systemStepController.ignoreSequenceStepFailure(model):
                                break
                            else:
                                logEvent("IGNORING STEP FAILURE")
                                self.postStep(model)
                                self.systemStepController.sequenceStepTaken(model)
                        else:
                            self.postStep(model)
                            self.systemStepController.sequenceStepTaken(model)
                    #end model split operator step
                    if stepFailed:
                        systemStepFailed = not self.systemStepController.retrySequence_modelStepFailure()
                        if not systemStepFailed:
                            stepFailed=False
                            logEvent("Retrying sequence")
                        else:
                            logEvent("Sequence failed")
                    else:
                        self.firstStep=False
                        systemStepFailed=False
                        logEvent("Step Taken, Model step t=%12.5e, dt=%12.5e for model %s" % (model.stepController.t_model,
                                                                                              model.stepController.dt_model,
                                                                                              model.name))
                        self.systemStepController.sequenceTaken()
                        for index,model in enumerate(self.modelList):
                            self.viewSolution(model,index)
                        if self.archiveFlag == ArchiveFlags.EVERY_MODEL_STEP:
                            self.tCount+=1
                            for index,model in enumerate(self.modelList):
                                self.archiveSolution(model,index,self.systemStepController.t_system)
                #end system split operator sequence
                if systemStepFailed:
                    logEvent("System Step Failed")
                    #go ahead and update as if the time step had succeeded
                    self.postStep(model)
                    self.systemStepController.modelStepTaken(model,self.t_stepSequence)
                    self.systemStepController.sequenceTaken()
                    self.systemStepController.updateTimeHistory()
                    #you're dead if retrySequence didn't work
                    logEvent("Step Failed, Model step t=%12.5e, dt=%12.5e for model %s" % (model.stepController.t_model,
                                                                                           model.stepController.dt_model,
                                                                                           model.name))
                    break
                else:
                    self.systemStepController.updateTimeHistory()
                    logEvent("Step Taken, System time step t=%12.5e, dt=%12.5e" % (self.systemStepController.t_system,
                                                                                   self.systemStepController.dt_system))
                    self.systemStepController.choose_dt_system()
                    logEvent("Potential System time step t=%12.5e, dt=%12.5e for next step" % (self.systemStepController.t_system,
                                                                                               self.systemStepController.dt_system))
                    if self.systemStepController.stepExact and self.systemStepController.t_system_last != self.tn:
                        self.systemStepController.stepExact_system(self.tn)
                #
                for model in self.modelList:
                    for av in self.auxiliaryVariables[model.name]:
                        av.calculate()
                if self.archiveFlag == ArchiveFlags.EVERY_SEQUENCE_STEP:
                    self.tCount+=1
                    for index,model in enumerate(self.modelList):
                        self.archiveSolution(model,index,self.systemStepController.t_system_last)

                #can only handle PUMIDomain's for now
                #if(self.tn < 0.05):
                #  self.nSolveSteps=0#self.nList[0].adaptMesh_nSteps-2
                self.nSolveSteps += 1
                #import gc; gc.collect()
                if(hasattr(self.pList[0].domain,"AdaptManager")):
                    if(self.AdaptHelper.PUMI_estimateError()):
                        self.AdaptHelper.PUMI_adaptMesh()
                #
                if measureSpeed and startToMeasureSpeed and self.comm.isMaster():
                    numTimeSteps += 1
                    logEvent("**********... end of time step. Number of time steps (to measure speed of the code): " + str(numTimeSteps),level=1)
                if measureSpeed and numTimeSteps==100 and self.comm.isMaster():
                    end = time.time()
                    Nproc = self.comm.size()
                    NDOFs=0
                    for i,mod in enumerate(self.modelList):
                        if (i in self.so.modelSpinUpList) == False: #To remove spin up models
                            NDOFs += mod.par_uList[0].size if mod.par_uList[0] is not None else len(mod.uList[0])
                    #
                    if append==False:
                        mode="w"
                    else:
                        mode="a"
                    with open ("speed_measurement.txt",mode) as file:
                        append=True
                        # write file and log this event
                        multiple_line_string = """ ******************** Measurements of speed ********************
                        Num of time steps: {nts:d}
                        Total time: {t:f}
                        Num of processors: {Nproc:d}
                        Total num of DOFs: {NDOFs:d}
                        Num of DOFs per processor: {aux1:d}
                        Time per time step, per DOF, per processor: {aux2:.4E} \n""".format(nts=numTimeSteps,
                                                                                            t=(end-start),
                                                                                            Nproc=Nproc,
                                                                                            NDOFs=NDOFs,
                                                                                            aux1=int(NDOFs/Nproc),
                                                                                            aux2=(end-start)/numTimeSteps*Nproc/NDOFs)
                        file.write(multiple_line_string)
                        logEvent(multiple_line_string,level=4)
                    #
                    measureSpeed = False
                #
            if measureSpeed and startToMeasureSpeed and self.comm.isMaster():
                end = time.time()
                Nproc = self.comm.size()
                NDOFs=0
                for i,mod in enumerate(self.modelList):
                    if (i in self.so.modelSpinUpList) == False:
                        NDOFs += mod.par_uList[0].size if mod.par_uList[0] is not None else len(mod.uList[0])
                #
                if append==False:
                    mode="w"
                else:
                    mode="a"
                with open ("speed_measurement.txt",mode) as file:
                    append=True
                    # write file and log this event
                    multiple_line_string = """ ******************** Measurements of speed ********************
                    Num of time steps: {nts:d}
                    Total time: {t:f}
                    Num of processors: {Nproc:d}
                    Total num of DOFs: {NDOFs:d}
                    Num of DOFs per processor: {aux1:d}
                    Time per time step, per DOF, per processor: {aux2:.4E} \n""".format(nts=numTimeSteps,
                               t=(end-start),
                               Nproc=Nproc,
                               NDOFs=NDOFs,
                               aux1=int(NDOFs/Nproc),
                               aux2=(end-start)/numTimeSteps*Nproc/NDOFs)
                    file.write(multiple_line_string)
                    logEvent(multiple_line_string,level=4)
                #
                measureSpeed = False
            #
            #end system step iterations
            if self.archiveFlag == ArchiveFlags.EVERY_USER_STEP and self.nSequenceSteps > nSequenceStepsLast:
                nSequenceStepsLast = self.nSequenceSteps
                self.tCount+=1
                for index,model in enumerate(self.modelList):
                    self.archiveSolution(model,index,self.systemStepController.t_system_last)

            if systemStepFailed:
                break

            if measureSpeed and self.comm.isMaster():
                startToMeasureSpeed = True
            if measureSpeed==False and append==True:
                measureSpeed=True
                numTimeSteps=0
            #
        logEvent("Finished calculating solution",level=3)
        # compute auxiliary quantities at last time step
        for index,model in enumerate(self.modelList):
            if hasattr(model.levelModelList[-1],'runAtEOS'):
                model.levelModelList[-1].runAtEOS()

        if(hasattr(self.pList[0].domain,"AdaptManager")):
        #Transfer solution to PUMI mesh for output
          self.pList[0].domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(b"coordinates",
            self.modelList[0].levelModelList[0].mesh.nodeArray)

          for m in self.modelList:
            for lm in m.levelModelList:
              coef = lm.coefficients
              if coef.vectorComponents is not None:
                vector=numpy.zeros((lm.mesh.nNodes_global,3),'d')
                for vci in range(len(coef.vectorComponents)):
                  vector[:,vci] = lm.u[coef.vectorComponents[vci]].dof[:]
                self.pList[0].domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                   coef.vectorName.encode('utf-8'), vector)
                #Transfer dof_last
                for vci in range(len(coef.vectorComponents)):
                  vector[:,vci] = lm.u[coef.vectorComponents[vci]].dof_last[:]
                self.pList[0].domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                     coef.vectorName.encode('utf-8')+b"_old", vector)
                #Transfer dof_last_last
                for vci in range(len(coef.vectorComponents)):
                  vector[:,vci] = lm.u[coef.vectorComponents[vci]].dof_last_last[:]
                self.pList[0].domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                     coef.vectorName.encode('utf-8')+b"_old_old", vector)
                del vector

              for ci in range(coef.nc):
                if coef.vectorComponents is None or \
                  ci not in coef.vectorComponents:
                  scalar=numpy.zeros((lm.mesh.nNodes_global,1),'d')
                  scalar[:,0] = lm.u[ci].dof[:]
                  self.pList[0].domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                      coef.variableNames[ci].encode('utf-8'), scalar)
                  #Transfer dof_last
                  scalar[:,0] = lm.u[ci].dof_last[:]
                  self.pList[0].domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                     coef.variableNames[ci].encode('utf-8')+b"_old", scalar)
                  #Transfer dof_last_last
                  scalar[:,0] = lm.u[ci].dof_last_last[:]
                  self.pList[0].domain.AdaptManager.PUMIAdapter.transferFieldToPUMI(
                     coef.variableNames[ci].encode('utf-8')+b"_old_old", scalar)
                  del scalar

          self.pList[0].domain.AdaptManager.PUMIAdapter.writeMesh(b"finalMesh.smb")
          if((self.PUMIcheckpointer.frequency>0) ):
            self.modelListOld = self.modelList
            self.PUMIcheckpointer.checkpoint(self.systemStepController.t_system_last)

        for index,model in enumerate(self.modelList):
            self.finalizeViewSolution(model)
            self.closeArchive(model,index)

        return systemStepFailed
    #
    #try to make preStep and postStep just manipulate "current values" and let the step controllers manage the history setting
    ##intermodel transfer before a solution step
    def preStep(self,model):
        for level,levelModel in enumerate(model.levelModelList):
            preCopy = levelModel.coefficients.preStep(model.stepController.t_model,firstStep=self.firstStep)
            if (preCopy is not None and ('copy_uList') in preCopy and preCopy['copy_uList'] == True):
                for u_ci_lhs,u_ci_rhs in zip(list(levelModel.u.values()),list(self.modelList[preCopy['uList_model']].levelModelList[level].u.values())):
                    u_ci_lhs.dof[:] = u_ci_rhs.dof
                levelModel.setFreeDOF(model.uList[level])
            if preCopy is not None and ('clear_uList') in preCopy and preCopy['clear_uList'] == True:
                for u_ci_lhs in list(levelModel.u.values()):
                    u_ci_lhs.dof[:] = 0.0
                levelModel.setFreeDOF(model.uList[level])
            if preCopy is not None and ('reset_uList') in preCopy and preCopy['reset_uList'] == True:
                levelModel.setFreeDOF(model.uList[level])
                levelModel.getResidual(model.uList[level],model.rList[level])

    ##intermodel transfer after a step
    def postStep(self,model):
        for level,levelModel in enumerate(model.levelModelList):
            postCopy = levelModel.coefficients.postStep(model.stepController.t_model,firstStep=self.firstStep)
            if postCopy is not None and ('copy_uList') in postCopy and postCopy['copy_uList'] == True:
                for u_ci_lhs,u_ci_rhs in zip(list(self.modelList[postCopy['uList_model']].levelModelList[level].u.values()),list(model.levelModelList[level].u.values())):
                    u_ci_lhs.dof[:] = u_ci_rhs.dof
                self.modelList[postCopy['uList_model']].levelModelList[level].setFreeDOF(self.modelList[postCopy['uList_model']].uList[level])

    def setWeakDirichletConditions(self,model):
        if model.weakDirichletConditions is not None:
            for levelModel in model.levelModelList:
                levelModel.dirichletNodeSetList={}
                levelModel.dirichletGlobalNodeSet={}
                levelModel.dirichletValues={}
            for ci in model.weakDirichletConditions:
                for levelModel in model.levelModelList:
                    model.weakDirichletConditions[ci](levelModel)

    def restrictFromFineMesh(self,model):
        for level in range(len(model.levelModelList)-1,0,-1):
            for cj in range(model.levelModelList[-1].coefficients.nc):
                model.meshTransfers.interp_bcListDict[cj][level].matvec(model.levelModelList[level].u[cj].dof,
                                                                                 model.levelModelList[level-1].u[cj].dof)
            model.levelModelList[level-1].setFreeDOF(model.uList[level-1])
            model.levelModelList[level-1].calculateCoefficients()

    ##save model's initial solution values to archive
    def archiveInitialSolution(self,model,index):
        if True if self.fastArchive == False else 'clsvof' in model.name:
            import xml.etree.ElementTree as ElementTree
            if self.archiveFlag == ArchiveFlags.UNDEFINED:
                return
            logEvent("Writing initial mesh for  model = "+model.name,level=3)
            logEvent("Writing initial conditions for  model = "+model.name,level=3)
            if not self.so.useOneArchive or index==0:
                self.ar[index].domain = ElementTree.SubElement(self.ar[index].tree.getroot(),"Domain")
            if self.so.useOneArchive:
                model.levelModelList[-1].archiveFiniteElementSolutions(self.ar[index],self.tnList[0],self.tCount,initialPhase=True,
                                                                       writeVectors=True,meshChanged=True,femSpaceWritten=self.femSpaceWritten,
                                                                       writeVelocityPostProcessor=self.opts.writeVPP)
            else:
                model.levelModelList[-1].archiveFiniteElementSolutions(self.ar[index],self.tnList[0],self.tCount,initialPhase=True,
                                                                       writeVectors=True,meshChanged=True,femSpaceWritten={},
                                                                       writeVelocityPostProcessor=self.opts.writeVPP)
            model.levelModelList[-1].archiveAnalyticalSolutions(self.ar[index],self.pList[index].analyticalSolution,
                                                                self.tnList[0],
                                                                self.tCount)
            #could just pull the code and flags out from SimTools rathter than asking it to parse them
            #uses values in simFlags['storeQuantities']
            #q dictionary
            if self.archive_q[index] == True:
                scalarKeys = model.simTools.getScalarElementStorageKeys(model,self.tnList[0])
                vectorKeys = model.simTools.getVectorElementStorageKeys(model,self.tnList[0])
                tensorKeys = model.simTools.getTensorElementStorageKeys(model,self.tnList[0])
                model.levelModelList[-1].archiveElementQuadratureValues(self.ar[index],self.tnList[0],self.tCount,
                                                                        scalarKeys=scalarKeys,vectorKeys=vectorKeys,tensorKeys=tensorKeys,
                                                                        initialPhase=True,meshChanged=True)
            if self.archive_ebq_global[index] == True:
                #ebq_global dictionary
                scalarKeys = model.simTools.getScalarElementBoundaryStorageKeys(model,self.tnList[0])
                vectorKeys = model.simTools.getVectorElementBoundaryStorageKeys(model,self.tnList[0])
                tensorKeys = model.simTools.getTensorElementBoundaryStorageKeys(model,self.tnList[0])
                model.levelModelList[-1].archiveElementBoundaryQuadratureValues(self.ar[index],self.tnList[0],self.tCount,
                                                                                scalarKeys=scalarKeys,vectorKeys=vectorKeys,tensorKeys=tensorKeys,
                                                                                initialPhase=True,meshChanged=True)
            if self.archive_ebqe[index] == True:
                #ebqe dictionary
                scalarKeys = model.simTools.getScalarExteriorElementBoundaryStorageKeys(model,self.tnList[0])
                vectorKeys = model.simTools.getVectorExteriorElementBoundaryStorageKeys(model,self.tnList[0])
                tensorKeys = model.simTools.getTensorExteriorElementBoundaryStorageKeys(model,self.tnList[0])
                model.levelModelList[-1].archiveExteriorElementBoundaryQuadratureValues(self.ar[index],self.tnList[0],self.tCount,
                                                                                        scalarKeys=scalarKeys,vectorKeys=vectorKeys,tensorKeys=tensorKeys,
                                                                                        initialPhase=True,meshChanged=True)
            try:
                phi_s = {}
                phi_s[0] = model.levelModelList[-1].coefficients.phi_s
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                       self.tnList[0],
                                                                       self.tCount,
                                                                       phi_s,
                                                                       res_name_base='phi_s')
                logEvent("Writing initial phi_s at DOFs for = "+model.name+" at time t="+str(t),level=3)
            except:
                pass
            try:
                phi_sp = {}
                phi_sp[0] = model.levelModelList[-1].coefficients.phi_sp
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                       self.tnList[0],
                                                                       self.tCount,
                                                                       phi_sp,
                                                                       res_name_base='phi_sp')
                logEvent("Writing initial phi_sp at DOFs for = "+model.name+" at time t="+str(self.tnList[0]),level=3)
            except:
                pass
            if 'clsvof' in model.name:
                vofDOFs = {}
                vofDOFs[0] = model.levelModelList[-1].vofDOFs
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                       self.tnList[0],
                                                                       self.tCount,
                                                                       vofDOFs,
                                                                       res_name_base='vof')
                logEvent("Writing initial vof from clsvof at time t="+str(0),level=3)
            #For aux quantity of interest (MQL)
            try:
                if model.levelModelList[-1].coefficients.outputQuantDOFs==True:
                    quantDOFs = {}
                    quantDOFs[0] = model.levelModelList[-1].quantDOFs
                    model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                           self.tnList[0],
                                                                           self.tCount,
                                                                           quantDOFs,
                                                                           res_name_base='quantDOFs_for_'+model.name)
                    logEvent("Writing initial quantity of interest at DOFs for = "+model.name+" at time t="+str(0),level=3)
            except:
                pass
            #Extra model-supplied derived nodal scalars (e.g. m_comp_co2 flash
            #fields Sg/X/c_brine) -> XDMF, alongside the primary unknowns.
            try:
                extra = getattr(model.levelModelList[-1].coefficients,
                                'archive_scalar_dofs', None)
                if extra:
                    for _nm, _arr in extra.items():
                        model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                               self.tnList[0],
                                                                               self.tCount,
                                                                               {0: _arr},
                                                                               res_name_base=_nm)
            except:
                pass
            #Write bathymetry for Shallow water equations (MQL)
            try:
                bathymetry = {}
                bathymetry[0] = model.levelModelList[-1].coefficients.b.dof
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                       self.tnList[0],
                                                                       self.tCount,
                                                                       bathymetry,
                                                                       res_name_base='bathymetry')
                logEvent("Writing bathymetry for = "+model.name,level=3)
            except:
                pass
            #write eta=h+bathymetry for SWEs (MQL)
            try:
                eta = {}
                eta[0] = model.levelModelList[-1].coefficients.b.dof+model.levelModelList[-1].u[0].dof
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                       self.tnList[0],
                                                                       self.tCount,
                                                                       eta,
                                                                       res_name_base='eta')
                logEvent("Writing bathymetry for = "+model.name,level=3)
            except:
                pass
            #for nonlinear POD
            if self.archive_pod_residuals[index] == True:
                res_space = {}; res_mass = {}
                for ci in range(model.levelModelList[-1].coefficients.nc):
                    res_space[ci] = numpy.zeros(model.levelModelList[-1].u[ci].dof.shape,'d')
                    model.levelModelList[-1].getSpatialResidual(model.levelModelList[-1].u[ci].dof,res_space[ci])
                    res_mass[ci] = numpy.zeros(model.levelModelList[-1].u[ci].dof.shape,'d')
                    model.levelModelList[-1].getMassResidual(model.levelModelList[-1].u[ci].dof,res_mass[ci])
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],self.tnList[0],self.tCount,res_space,res_name_base='spatial_residual')
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],self.tnList[0],self.tCount,res_mass,res_name_base='mass_residual')
            if not self.opts.cacheArchive:
                if not self.so.useOneArchive:
                    self.ar[index].sync()
                else:
                    if index == len(self.ar) - 1:
                        self.ar[index].sync()

    ##save model's solution values to archive
    def archiveSolution(self,model,index,t=None):
        if True if self.fastArchive == False else 'clsvof' in model.name:
            if self.archiveFlag == ArchiveFlags.UNDEFINED:
                return
            if t is None:
                t = self.systemStepController.t_system
            logEvent("Writing mesh header for  model = "+model.name+" at time t="+str(t),level=3)
            logEvent("Writing solution for  model = "+model.name,level=3)
            if self.so.useOneArchive:
                if index==0:
                    self.femSpaceWritten={}
                model.levelModelList[-1].archiveFiniteElementSolutions(self.ar[index],t,self.tCount,
                                                                       initialPhase=False,
                                                                       writeVectors=True,meshChanged=True,femSpaceWritten=self.femSpaceWritten,
                                                                       writeVelocityPostProcessor=self.opts.writeVPP)
            else:
                model.levelModelList[-1].archiveFiniteElementSolutions(self.ar[index],t,self.tCount,
                                                                       initialPhase=False,
                                                                       writeVectors=True,meshChanged=True,femSpaceWritten={},
                                                                       writeVelocityPostProcessor=self.opts.writeVPP)
            model.levelModelList[-1].archiveAnalyticalSolutions(self.ar[index],self.pList[index].analyticalSolution,
                                                                t,
                                                                self.tCount)
            #uses values in simFlags['storeQuantities']
            #q dictionary
            if self.archive_q[index] == True and self.fastArchive==False:
                scalarKeys = model.simTools.getScalarElementStorageKeys(model,t)
                vectorKeys = model.simTools.getVectorElementStorageKeys(model,t)
                tensorKeys = model.simTools.getTensorElementStorageKeys(model,t)
                model.levelModelList[-1].archiveElementQuadratureValues(self.ar[index],t,self.tCount,
                                                                        scalarKeys=scalarKeys,vectorKeys=vectorKeys,tensorKeys=tensorKeys,
                                                                        initialPhase=False,meshChanged=True)
            #ebq_global dictionary
            if self.archive_ebq_global[index] == True and self.fastArchive==False:
                scalarKeys = model.simTools.getScalarElementBoundaryStorageKeys(model,t)
                vectorKeys = model.simTools.getVectorElementBoundaryStorageKeys(model,t)
                tensorKeys = model.simTools.getTensorElementBoundaryStorageKeys(model,t)
                model.levelModelList[-1].archiveElementBoundaryQuadratureValues(self.ar[index],t,self.tCount,
                                                                                scalarKeys=scalarKeys,vectorKeys=vectorKeys,tensorKeys=tensorKeys,
                                                                                initialPhase=False,meshChanged=True)
            if self.archive_ebqe[index] == True and self.fastArchive==False:
                #ebqe dictionary
                scalarKeys = model.simTools.getScalarExteriorElementBoundaryStorageKeys(model,t)
                vectorKeys = model.simTools.getVectorExteriorElementBoundaryStorageKeys(model,t)
                tensorKeys = model.simTools.getTensorExteriorElementBoundaryStorageKeys(model,t)
                model.levelModelList[-1].archiveExteriorElementBoundaryQuadratureValues(self.ar[index],t,self.tCount,
                                                                                        scalarKeys=scalarKeys,vectorKeys=vectorKeys,tensorKeys=tensorKeys,
                                                                                        initialPhase=False,meshChanged=True)
            if self.fastArchive==False:
                try:
                    phi_s = {}
                    phi_s[0] = model.levelModelList[-1].coefficients.phi_s
                    model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                           self.tnList[0],
                                                                           self.tCount,
                                                                           phi_s,
                                                                           res_name_base='phi_s')
                    logEvent("Writing phi_s at DOFs for = "+model.name+" at time t="+str(t),level=3)
                except:
                    pass
                try:
                    phi_sp = {}
                    phi_sp[0] = model.levelModelList[-1].coefficients.phi_sp
                    model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                           t,
                                                                           self.tCount,
                                                                           phi_sp,
                                                                           res_name_base='phi_sp')
                    logEvent("Writing phi_sp at DOFs for = "+model.name+" at time t="+str(t),level=3)
                except:
                    pass
            if 'clsvof' in model.name and self.fastArchive==False:
                vofDOFs = {}
                vofDOFs[0] = model.levelModelList[-1].vofDOFs
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                       self.tnList[0],
                                                                       self.tCount,
                                                                       vofDOFs,
                                                                       res_name_base='vof')
                logEvent("Writing initial vof from clsvof at time t="+str(t),level=3)
            if self.fastArchive==False:
                try:
                    if model.levelModelList[-1].coefficients.outputQuantDOFs==True:
                        quantDOFs = {}
                        quantDOFs[0] = model.levelModelList[-1].quantDOFs
                        model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                               self.tnList[0],
                                                                               self.tCount,
                                                                               quantDOFs,
                                                                               res_name_base='quantDOFs_for_'+model.name)
                        logEvent("Writing quantity of interest at DOFs for = "+model.name+" at time t="+str(t),level=3)
                except:
                    pass
            #Extra model-supplied derived nodal scalars (e.g. m_comp_co2 flash
            #fields Sg/X/c_brine) -> XDMF, alongside the primary unknowns.
            if self.fastArchive==False:
                try:
                    extra = getattr(model.levelModelList[-1].coefficients,
                                    'archive_scalar_dofs', None)
                    if extra:
                        for _nm, _arr in extra.items():
                            model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                                   self.tnList[0],
                                                                                   self.tCount,
                                                                                   {0: _arr},
                                                                                   res_name_base=_nm)
                except:
                    pass
            #Write bathymetry for Shallow water equations (MQL)
            if self.fastArchive==False:
                try:
                    bathymetry = {}
                    bathymetry[0] = model.levelModelList[-1].coefficients.b.dof
                    model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                           self.tnList[0],
                                                                           self.tCount,
                                                                           bathymetry,
                                                                           res_name_base='bathymetry')
                    logEvent("Writing bathymetry for = "+model.name,level=3)
                except:
                    pass
                #write eta=h+bathymetry for SWEs (MQL)
                try:
                    eta = {}
                    eta[0] = model.levelModelList[-1].coefficients.b.dof+model.levelModelList[-1].u[0].dof
                    model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],
                                                                           self.tnList[0],
                                                                           self.tCount,
                                                                           eta,
                                                                           res_name_base='eta')
                    logEvent("Writing bathymetry for = "+model.name,level=3)
                except:
                    pass

            #for nonlinear POD
            if self.archive_pod_residuals[index] == True and self.fastArchive==False:
                res_space = {}; res_mass = {}
                for ci in range(model.levelModelList[-1].coefficients.nc):
                    res_space[ci] = numpy.zeros(model.levelModelList[-1].u[ci].dof.shape,'d')
                    model.levelModelList[-1].getSpatialResidual(model.levelModelList[-1].u[ci].dof,res_space[ci])
                    res_mass[ci] = numpy.zeros(model.levelModelList[-1].u[ci].dof.shape,'d')
                    model.levelModelList[-1].getMassResidual(model.levelModelList[-1].u[ci].dof,res_mass[ci])

                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],t,self.tCount,res_space,res_name_base='spatial_residual')
                model.levelModelList[-1].archiveFiniteElementResiduals(self.ar[index],t,self.tCount,res_mass,res_name_base='mass_residual')
            if not self.opts.cacheArchive:
                if not self.so.useOneArchive:
                    self.ar[index].sync()
                else:
                    if index == len(self.ar) - 1:
                        self.ar[index].sync()

    ## clean up archive
    def closeArchive(self,model,index):
        if self.archiveFlag is None:
            return
        if self.so.useOneArchive:
            if index==0:
                logEvent("Closing solution archive for "+self.so.name)
                self.ar[index].close()
        else:
            logEvent("Closing solution archive for "+model.name)
            self.ar[index].close()

    def initializeViewSolution(self,model):
        """
        """
        model.viewer.preprocess(model,model.stepController.t_model_last)
        model.simTools.preprocess(model,model.stepController.t_model_last)

    ## run time visualization for modela
    def viewSolution(self,model,initialCondition=False):
        """

        """
        #mwf looking at last solns
        if (model.viewer.viewerType != 'matlab' or model.stepController.t_model_last <= self.tnList[0] or
            model.stepController.t_model_last >= self.tnList[-1]):
            model.viewer.processTimeLevel(model,model.stepController.t_model_last)
            model.simTools.processTimeLevel(model,model.stepController.t_model_last)


    ## clean up runtime visualization
    def finalizeViewSolution(self,model):
        model.viewer.postprocess(model,model.stepController.t_model_last)
        model.simTools.postprocess(model,model.stepController.t_model_last)
