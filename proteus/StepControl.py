"""
A class hierarchy for methods of controlling the step size

.. inheritance-diagram:: proteus.StepControl
   :parts: 1
"""
from .Profiling import logEvent
from . import Comm
from .petsc_optional import PETSc
from .Comm import globalMax

class SC_base(object):
    """
    A simple fixed time stepping controller with no error/stability
    control and reduction by 1/2 in case of solver failures.

    TODO:
     Implementation:
      Get data file format straight

    """
    def __init__(self,model,nOptions):
        import copy
        self.model   = model
        self.uListSave = []
        self.rListSave = []
        for mi,u,r in zip(self.model.levelModelList,self.model.uList,self.model.rList):
            mi.timeIntegration.setFromOptions(nOptions)
            self.uListSave.append(copy.deepcopy(u))
            self.rListSave.append(copy.deepcopy(r))
        self.nStages = self.model.levelModelList[-1].timeIntegration.nStages
        self.t_model_last = 0.0
        self.dt_model = 1.0
        self.t_model=self.t_model_last+self.dt_model
        self.solverFailures=0
        self.errorFailures=0
        self.maxSolverFailures=nOptions.maxSolverFailures
        self.maxErrorFailures=nOptions.maxErrorFailures
        self.stepExact = True
        #keep track of time step history ...
        #should push this into base class at some point
        #comm  = Comm.get()
        #rank = comm.rank()
        #dname = "data_dt_%s_p%d.txt" % (model.levelModelList[0].name,rank)
        #self.datafile = open(dname,'w')
        #dheader = \
#"""
# k |  Step Size   |      tn      |errf|nlsf|ilsf|func|jacs|NLit|L_IT|linS|"""
        #self.datafile.write(dheader)
    def state(self):
        """Nothing to carry beyond the solution (the state contract, see TimeIntegration.TI_base.state)."""
        return None

    def set_state(self, state):
        """Restore what ``state()`` returned; see TimeIntegration.TI_base.state."""
        pass

    def setInitialGuess(self,uList,rList):
        pass
    def set_dt_allLevels(self):
        self.t_model = self.t_model_last + self.dt_model
        for mi in self.model.levelModelList:
            mi.timeIntegration.set_dt(self.dt_model)
    def resetSolution(self):
        for u,r,uSave,rSave in zip(self.model.uList,
                                   self.model.rList,
                                   self.uListSave,
                                   self.rListSave):
            u[:]=uSave
            r[:]=rSave
    def saveSolution(self):
        for u,r,uSave,rSave in zip(self.model.uList,
                                   self.model.rList,
                                   self.uListSave,
                                   self.rListSave):
            uSave[:]=u
            rSave[:]=r
    def retryStep_solverFailure(self):
        self.solverFailures += 1
        retry = False
        if self.solverFailures < self.maxSolverFailures:
            self.resetSolution()
            self.dt_model *= 0.5
            if self.dt_model > self.t_model_last*1.0e-8:
                self.set_dt_allLevels()
                retry = True
            else:
                logEvent("Time step reduced to machine precision",level=1)
        return retry
    def retryStep_errorFailure(self):
        self.errorFailures += 1
        retry = False
        if self.errorFailures < self.maxErrorFailures:
            self.resetSolution()
            self.dt_model *= 0.5
            if self.dt_model > self.t_model_last*1.0e-8:
                self.set_dt_allLevels()
                retry = True
            else:
                logEvent("Time step reduced to machine precision",level=1)
        return retry
    def stepExact_model(self,tOut):
        if self.t_model > tOut - tOut*1.0e-8:
            logEvent("StepControl base stepExact t_model= %s tOut= %s t_model_last= %s dt_model= %s setting to %s " % (self.t_model,tOut,self.t_model_last,
                                                                                                                  self.dt_model,tOut-self.t_model_last),1)
            self.dt_model = tOut - self.t_model_last
            self.set_dt_allLevels()
            #self.substeps = [self.t_model]
            self.setSubsteps([self.t_model])
    def choose_dt_model(self):
        self.set_dt_allLevels()
        #self.substeps = [self.t_model]
        self.setSubsteps([self.t_model])
    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last=t0
        self.dt_model = tOut - t0
        self.set_dt_allLevels()
        #self.substeps = [self.t_model]
        self.setSubsteps([self.t_model])
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def updateSubstep(self):
        for m in self.model.levelModelList:
            m.timeIntegration.updateStage()
    def setSubsteps(self,tList):
        """
        default controller just sets substeps to be input list
        without regard to time integration scheme
        """
        self.substeps = [t for t in tList]
    def initializeTimeHistory(self):
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        #self.dt_model = 1.0
        self.set_dt_allLevels()
        self.setSubsteps([self.t_model])
        for m,u,r in zip(self.model.levelModelList,
                         self.model.uList,
                         self.model.rList):
            #m.getResidual(u,r)#cek todo move these out so history can be reset with calculating these
            #m.estimate_mt()
            #m.initializeTimeHistory()
            m.timeIntegration.initializeTimeHistory(resetFromDOF=True)
    def updateTimeHistory(self,resetFromDOF=False):
        self.writeSolverStatisticsForStep()
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        self.t_model_last = self.t_model
        for m in self.model.levelModelList:
            m.updateTimeHistory(resetFromDOF)
            m.timeIntegration.updateTimeHistory(resetFromDOF)
    def errorFailure(self):
        return False
    def writeSolverStatisticsForStep(self):
        """
        record some relevant time step and solver statistics
        uses finest level to get this information
        called before updating t_model --> t_model_last etc

        """
#         k |  Step Size   |      tn      |errf|nlsf|ilsf|func|jacs|NLit|L_IT|linS|
        #cek hack had to comment out to get some other things working
        pass
#         lineform = \
# """
#  %d | %12.5e | %12.5e | %d | %d | %d | %d | %d | %d | %d | %d |"""
#         mfine = self.model.levelModelList[-1]
#         line = lineform % (mfine.timeIntegration.timeOrder,
#                            self.dt_model,
#                            self.t_model,
#                            self.errorFailures,
#                            self.solverFailures,
#                            self.model.solver.solverList[-1].linearSolver.solveCalls_failed,  #solveCalls_failed
#                            mfine.nonlinear_function_evaluations,
#                            mfine.nonlinear_function_jacobian_evaluations,
#                            self.model.solver.solverList[-1].its,
#                            self.model.solver.solverList[-1].linearSolver.its, #should be recordedIts if know linear solver computeAverages is called
#                            -1)#need to record line searches

#         #this is unpleasant but need to zero nonlinear function statistics somewhere
#         for m in self.model.levelModelList:
#             m.resetNonlinearFunctionStatistics()

#         self.datafile.write(line)
    #

FixedStep = SC_base

class Newton_controller(SC_base):
    """
    Same as SC_base but since there is no dt we have not  way to retry
    """
    def __init__(self,model,nOptions):
        SC_base.__init__(self,model,nOptions)
    def retryStep_solverFailure(self):
        return False
    def retryStep_errorFailure(self):
        return False

class PsiTCtte_controller(SC_base):
    def __init__(self,model,nOptions):
        SC_base.__init__(self,model,nOptions)
        #cek todo have to just pick a component for tols until I get res norm right
        for ci in list(nOptions.atol_res.keys()):
            self.atol = nOptions.atol_res[ci]
            self.rtol = nOptions.rtol_res[ci]
        self.stepExact = True
    def stepExact_model(self,tOut):
        self.dt_model = tOut - self.t_model_last
    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last = t0
        self.t_model = tOut
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0,tOut,m.q)
        #set the starting time steps
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        #now set back to full step
        self.dt_model = tOut - t0
        self.t = self.t_model_last + self.dt_model
        #self.substeps = [self.t_model_last]
        self.setSubsteps([self.t_model_last])
        self.nssteps=0
        #reset the nonlinear solver to do 1 iteration
        self.model.solver.maxIts=1
        self.model.solver.convergenceTest = 'its'
        self.model.solver.tolList = None
        for levelSolver in self.model.solver.solverList:
            levelSolver.maxIts=1
            levelSolver.convergenceTest = 'its'
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def setInitialGuess(self,uList,rList):
        pass#leave at last solution
    def updateSubstep(self):
        #choose  a new dt and add a substep without increasing t
        #if the steady state has been reached then append the new  t to the  substeps
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        #here we just need to test the error and set to tOut if steady state
        if self.nssteps == 0:
            self.res0 = self.model.solver.solverList[-1].norm_r0
        res = self.model.solver.solverList[-1].norm_r0
        #print "res dt",res,self.dt_model,self.res0,self.rtol,self.atol
        ssError = res/(self.res0*self.rtol + self.atol)
        #print "ssError",ssError
        for m in self.model.levelModelList:
            m.updateTimeHistory(self.t_model)
            #m.timeIntegration.updateTimeHistory()
        if ssError >= 1.0:
            for m in self.model.levelModelList:
                m.timeIntegration.choose_dt()
            dt_model_save = self.dt_model
            self.dt_model = m.timeIntegration.dt
            self.t_model = self.t_model_last
            for mi in self.model.levelModelList:
                mi.timeIntegration.set_dt(self.dt_model)
                mi.timeIntegration.t = self.t_model_last
            self.dt_model = dt_model_save
            self.substeps.append(self.t_model_last)#set to last time step
        else:
            self.t_model = self.t_model_last + self.dt_model
            if self.substeps[-1] != self.t_model:
                self.substeps.append(self.t_model)#set to new  time step
            else:
                logEvent("PsiTC converged ||res|| = %12.5e" % res)
        self.nssteps +=1
    def choose_dt_model(self):
        #don't modify dt_model
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.set_dt_allLevels()
        #self.substeps=[self.t_model_last]#set to last time step
        self.setSubsteps([self.t_model_last])#set to last time step
class Osher_controller(SC_base):
    def __init__(self,model,nOptions):
        SC_base.__init__(self,model,nOptions)
        #cek todo have to just pick a component for tols until I get res norm right
        for ci in list(nOptions.atol_res.keys()):
            self.atol = nOptions.atol_res[ci]
            self.rtol = nOptions.rtol_res[ci]
        self.stepExact = True
        for m in model.levelModelList:
            m.timeIntegration.isAdaptive=False
        self.nSteps=0
        self.nStepsMax=30#10
    def stepExact_model(self,tOut):
        self.dt_model = tOut - self.t_model_last
    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last = t0
        self.t_model = tOut
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0,tOut,m.q)
        #set the starting time steps
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        #now set back to full step
        self.dt_model = tOut - t0
        self.t = self.t_model_last + self.dt_model
        #self.substeps = [self.t_model_last]
        self.setSubsteps([self.t_model_last])
        self.nSteps=0
        #cek using newton
        #reset the nonlinear solver to do 1 iteration
        #self.model.solver.maxIts=1
        #self.model.solver.convergenceTest = 'its'
        #self.model.solver.tolList = None
        #for levelSolver in self.model.solver.solverList:
        #    levelSolver.maxIts=1
        #    levelSolver.convergenceTest = 'its'
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def updateSubstep(self):
        #choose  a new dt and add a substep without increasing t
        #if the steady state has been reached then append the new  t to the  substeps
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        #here we just need to test the error and set to tOut if steady state
        if self.nSteps == 0:
            self.res0 = self.model.solver.solverList[-1].norm_r0
        res = self.model.solver.solverList[-1].norm_r0
        #print "res0",res
        ssError = res/(self.res0*self.rtol + self.atol)
        #print "ssError",ssError
        for m in self.model.levelModelList:
            m.updateTimeHistory(self.t_model)
            m.timeIntegration.updateTimeHistory()
        #if ssError >= 1.0 or self.nssteps < self.nStepsMax:
        if self.nSteps < self.nStepsMax:
            for m in self.model.levelModelList:
                m.timeIntegration.choose_dt()
            dt_model_save = self.dt_model
            #mwf hack, play with picking time step after a certain number of steps
            if self.nSteps > 4:
                self.dt_model = m.timeIntegration.dt*2.0**(self.nSteps-4)
            self.set_dt_allLevels()
            self.dt_model = dt_model_save
            self.substeps.append(self.t_model_last)#set to last time step
            self.nSteps+=1
        else:
            if self.substeps[-1] != self.t_model:
                self.substeps[-1] = self.t_model#set to new time step#.append(self.t_model)#set to new  time step
            else:
                logEvent("Osher converged %12.5e" % (ssError))
            self.nSteps=0

    def choose_dt_model(self):
        #don't modify dt_model
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.set_dt_allLevels()
        #self.substeps=[self.t_model_last]#set to last time step
        self.setSubsteps([self.t_model_last])#set to last time step

class Osher_PsiTC_controller(SC_base):
    def __init__(self,model,nOptions):
        SC_base.__init__(self,model,nOptions)
        for ci in list(nOptions.atol_res.keys()):
            self.atol = nOptions.atol_res[ci]
            self.rtol = nOptions.rtol_res[ci]
        self.stepExact = True
        for m in model.levelModelList:
            m.timeIntegration.isAdaptive=False
        self.nSteps=0
        self.nStepsOsher=nOptions.psitc['nStepsForce']
        self.nStepsMax=nOptions.psitc['nStepsMax']
    def stepExact_model(self,tOut):
        #pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        #physical time step
        self.t_model=tOut
        self.setSubsteps([tOut])
    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last = t0
        self.t_model = tOut
        #pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0,tOut,m.q)
        #set the starting time steps
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        #physical time step
        self.t = tOut
        self.setSubsteps([tOut])
        self.nSteps=0
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def updateSubstep(self):
        #choose  a new dt and add a substep without increasing t
        #if the steady state has been reached then append the new  t to the  substeps
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        #here we just need to test the error and set to tOut if steady state
        if self.nSteps == 0:
            self.res0 = self.model.solver.solverList[-1].norm_r0
            for m in self.model.levelModelList:
                m.timeIntegration.choose_dt()
        res = self.model.solver.solverList[-1].norm_r0
        ssError = res/(self.res0*self.rtol + self.atol)
        for m in self.model.levelModelList:
            m.updateTimeHistory(self.t_model)
            m.timeIntegration.updateTimeHistory()
        if ((self.nSteps < self.nStepsOsher or
             ssError >= 1.0) and
            self.nSteps < self.nStepsMax):
            self.nSteps+=1
            self.dt_model = m.timeIntegration.dt
            if self.nSteps >= self.nStepsOsher:#start ramping up the time step
                self.dt_model = m.timeIntegration.dt*2.0**(self.nSteps-self.nStepsOsher)
            logEvent("Osher-PsiTC dt %12.5e" %(self.dt_model),level=1)
            self.set_dt_allLevels()
            #physical time step
            self.t_model = self.substeps[0]
            self.substeps.append(self.substeps[0])
            logEvent("Osher-PsiTC iteration %d |res| = %12.5e" %(self.nSteps,res),level=1)
        elif self.nSteps >= self.nStepsMax:
            logEvent("Osher-PsiTC DID NOT Converge |res| = %12.5e but quitting anyway" %(res,))
            self.nSteps=0
        else:
            logEvent("Osher-PsiTC converged |res| = %12.5e" %(res,))
            self.nSteps=0
    def choose_dt_model(self):
        #don't modify dt_model
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        #pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        #physical time step
        self.t_model = self.substeps[0]
        self.setSubsteps([self.substeps[0]])


class Osher_PsiTC_controller2(SC_base):
    def __init__(self,model,nOptions):
        SC_base.__init__(self,model,nOptions)
        for ci in list(nOptions.atol_res.keys()):
            self.atol = nOptions.atol_res[ci]
            self.rtol = nOptions.rtol_res[ci]
        self.stepExact = True
        for m in model.levelModelList:
            m.timeIntegration.isAdaptive=False
        self.nSteps=0
        self.nStepsOsher=nOptions.psitc['nStepsForce']
        self.red_ratio=nOptions.psitc['reduceRatio']
        self.start_ratio=nOptions.psitc['startRatio']
        self.nStepsMax=nOptions.psitc['nStepsMax']
    def stepExact_model(self,tOut):
        #pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = self.start_ratio*m.timeIntegration.dt
        self.set_dt_allLevels()
        #physical time step
        self.t_model=tOut
        self.setSubsteps([tOut])
    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last = t0
        self.t_model = tOut
        #pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0,tOut,m.q)
        #set the starting time steps
        self.dt_model = self.start_ratio*m.timeIntegration.dt
        self.set_dt_allLevels()
        #physical time step
        self.t = tOut
        self.setSubsteps([tOut])
        self.nSteps=0



        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def updateSubstep(self):
        #choose  a new dt and add a substep without increasing t
        #if the steady state has been reached then append the new  t to the  substeps
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        #here we just need to test the error and set to tOut if steady state
        if self.nSteps == 0:
            self.res0 = self.model.solver.solverList[-1].norm_r0
            for m in self.model.levelModelList:
                m.timeIntegration.choose_dt()

            self.dt_model = self.start_ratio*self.model.levelModelList[0].timeIntegration.dt
        res = self.model.solver.solverList[-1].norm_r0
        ssError = res/(self.res0*self.rtol + self.atol)
        for m in self.model.levelModelList:
            m.updateTimeHistory(self.t_model)
            m.timeIntegration.updateTimeHistory()
        if ((self.nSteps < self.nStepsOsher or
             ssError >= 1.0) and
            self.nSteps < self.nStepsMax):
            self.nSteps+=1
            self.dt_model = m.timeIntegration.dt
            if self.nSteps >= self.nStepsOsher:#start ramping up the time step
                self.dt_model = self.dt_model*self.red_ratio
            #logEvent("Osher-PsiTC dt %12.5e" %(self.dt_model),level=1)
            self.set_dt_allLevels()
            #physical time step
            self.t_model = self.substeps[0]
            self.substeps.append(self.substeps[0])


            logEvent("Osher-PsiTC iteration %d  dt = %12.5e  |res| = %12.5e %g  " %(self.nSteps,self.dt_model,res,(res/self.res0)*100.0),level=1)
        elif self.nSteps >= self.nStepsMax:
            logEvent("Osher-PsiTC DID NOT Converge |res| = %12.5e but quitting anyway" %(res,))
            self.nSteps=0
        else:
            logEvent("Osher-PsiTC converged |res| = %12.5e %12.5e" %(res,ssError*100.0))
            self.nSteps=0
    def choose_dt_model(self):
        #don't modify dt_model
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        #pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = self.start_ratio*m.timeIntegration.dt
        self.set_dt_allLevels()
        #physical time step
        self.t_model = self.substeps[0]
        self.setSubsteps([self.substeps[0]])

class Min_dt_controller(SC_base):
    def __init__(self,model,nOptions):
        SC_base.__init__(self,model,nOptions)
        self.stepExact = False
    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last = t0
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0,tOut,m.q)
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        self.substeps = [self.t_model]
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def choose_dt_model(self):
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        #self.substeps=[self.t_model]
        self.setSubsteps([self.t_model])
    def setSubsteps(self,tList):
        """
        allow time intergration scheme to pick 'substeps' for interval. This would be useful
        for multistage schemes
        """
        for m in self.model.levelModelList:
            m.timeIntegration.generateSubsteps(tList)
        self.substeps = self.model.levelModelList[-1].timeIntegration.substeps
        logEvent("Min_dt_controller setSubsteps tList=%s self.t_model=%s self.substeps= %s " % (tList,self.t_model,self.substeps))
class Min_dt_RKcontroller(SC_base):
    def __init__(self,model,nOptions):
        SC_base.__init__(self,model,nOptions)
        self.stepExact = False#True#False
    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last = t0
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0,tOut,m.q)
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        #self.substeps = m.timeIntegration.substeps
        self.setSubsteps([self.t_model_last + self.dt_model])
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def choose_dt_model(self):
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = m.timeIntegration.dt
        self.set_dt_allLevels()
        #self.substeps=m.timeIntegration.substeps
        self.setSubsteps([self.t_model_last + self.dt_model])
    def stepExact_model(self,tOut):
        if self.t_model > tOut - tOut*1.0e-8:
            self.dt_model = tOut - self.t_model_last
            self.set_dt_allLevels()
            #self.substeps = self.model.levelModelList[-1].timeIntegration.substeps
            self.setSubsteps([self.t_model_last + self.dt_model])
    def setSubsteps(self,tList):
        """
        allow time intergration scheme to pick 'substeps' for interval. this is necessary for
        RK since stages are substeps
        """
        for m in self.model.levelModelList:
            m.timeIntegration.generateSubsteps(tList)
        self.substeps = self.model.levelModelList[-1].timeIntegration.substeps
        #mwf where to set t?
        #import pdb
        #pdb.set_trace()
        for m in self.model.levelModelList:
            m.timeIntegration.t = self.substeps[0]
class Min_dt_cfl_controller(Min_dt_controller):
    def __init__(self,model,nOptions):
        Min_dt_controller.__init__(self,model,nOptions)
        self.runCFL = nOptions.runCFL
        self.dt_model_last = None
        self.dt_ratio_max = 2.0
        self.cfl = {}
        for ci in range(model.levelModelList[-1].nc):
            if ('cfl',ci) in model.levelModelList[-1].q:
                self.cfl[ci] = model.levelModelList[-1].q[('cfl',ci)]

    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last = t0
        m = self.model.levelModelList[-1]
        maxCFL = 1.0e-6
        for ci in range(m.nc):
            if ci in self.cfl:
                maxCFL = max(maxCFL,globalMax(self.cfl[ci].max()))
        self.dt_model = self.runCFL/maxCFL
        if self.dt_model_last is None:
            self.dt_model_last = self.dt_model
        if self.dt_model/self.dt_model_last  > self.dt_ratio_max:
            self.dt_model = self.dt_model_last*self.dt_ratio_max
        self.set_dt_allLevels()
        self.substeps = [self.t_model]
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def choose_dt_model(self):
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        m = self.model.levelModelList[-1]
        maxCFL = 1.0e-6
        for ci in range(m.nc):
            if ci in self.cfl:
                maxCFL = max(maxCFL,globalMax(self.cfl[ci].max()))
        self.dt_model = self.runCFL/maxCFL
        if self.dt_model_last is None:
            self.dt_model_last = self.dt_model
        if self.dt_model/self.dt_model_last  > self.dt_ratio_max:
            self.dt_model = self.dt_model_last*self.dt_ratio_max
        self.set_dt_allLevels()
        #self.substeps=[self.t_model]
        self.setSubsteps([self.t_model])
    def updateTimeHistory(self,resetFromDOF=False):
        Min_dt_controller.updateTimeHistory(self,resetFromDOF=resetFromDOF)
        self.dt_model_last = self.dt_model

class Min_dt_controller_FCT(Min_dt_controller):
    """
    controller try and implement a piece of FCT methodology where first step is
    a low order solution and the next step corrects is

    """
    def __init__(self,model,nOptions):
        Min_dt_controller.__init__(self,model,nOptions)

    def initialize_dt_model(self,t0,tOut):
        Min_dt_controller.initialize_dt_model(self,t0,tOut)

    def setInitialGuess(self,uList,rList):
        for m in self.model.levelModelList:
            m.nonlinear_function_evaluations = 0

    def errorFailure(self):
        #redo step if it was the low order (step)
        low_order_step = False
        for m in self.model.levelModelList:
            if m.timeIntegration.low_order_step == True:
                low_order_step = True
                m.timeIntegration.low_order_step = False
        return low_order_step
    def retryStep_errorFailure(self):
        self.errorFailures += 1
        #need to one solution from first iteration as the
        #low order one
        if self.errorFailures == 1:
            for m in self.model.levelModelList:
                for ci in range(m.nc):
                    m.timeIntegration.u_dof_low_order[ci].flat[:] = m.u[ci].dof
            return True
        return False
class FLCBDF_controller(SC_base):
    def __init__(self,model,nOptions):
        import numpy
        self.ignoreErrorControl=True#if mincfl kicks in this gets set to True until time step is larger than min time step
        self.dtRatioMax=2.0
        SC_base.__init__(self,model,nOptions)
        self.dt_min=getattr(nOptions,'dt_min',0.0)
        self.stepExact=False
        self.massComponents = list(self.model.levelModelList[-1].coefficients.mass.keys())
        for mi in self.model.levelModelList:
            mi.timeIntegration.setFromOptions(nOptions)
        self.nStages = self.model.levelModelList[-1].timeIntegration.nStages
        self.t_model=0.0
        self.flcbdfList=[]
        #mwf change to take account of level and component
        self.one={}#self.one=[]
        #mwf add level index for one?
        for l,m in enumerate(model.levelModelList):
            self.flcbdfList.append(m.timeIntegration.flcbdf)
            for ci in m.timeIntegration.massComponents:
                try:
                    #the error norm sums over the ranks: the ghost elements (after the owned ones) are another
                    #rank's and must not count twice, or the step sequence depends on the number of ranks
                    dV = m.q[('dV_u',ci)].copy()
                    dV[m.mesh.nElements_owned:] = 0.0
                    m.timeIntegration.flcbdf[ci].setTolerances(nOptions.atol_u[ci],nOptions.rtol_u[ci],dV)
                except:
                    logEvent("WARNING: couldn't find integration weights for FLCBDF norm")
                    m.timeIntegration.flcbdf[ci].setTolerances(nOptions.atol_u[ci],nOptions.rtol_u[ci],numpy.ones_like(m.q[('m',0)]))
            for ci in range(m.coefficients.nc):
                #mwf this gets messed up because of multiple levels, ci doesn't grab one on right level?
                ##self.one[ci] /= float(len(m.u[ci].dof.flat)) #make L2 into wrms until we make real L2 for dof
                #self.one.append(numpy.ones(m.u[ci].dof.shape,'d'))
                #self.one[-1] /= float(len(m.u[ci].dof.flat)) #make L2 into wrms until we make real L2 for dof
                #mwf debug
                #import pdb
                #pdb.set_trace()
                #print "FLCBDF step control calling setTolerances with self.one[ci]"
                #mwf problem with self.one[ci] being right shape on different levels?
                ##m.timeIntegration.flcbdf[('u',ci)].setTolerances(1.0,1.0,self.one[ci])
                #m.timeIntegration.flcbdf[('u',ci)].setTolerances(1.0,1.0,self.one[-1])
                self.one[(l,ci)]  = numpy.ones(m.u[ci].dof.shape,'d')
                #self.one[(l,ci)]  = numpy.zeros(m.u[ci].dof.shape,'d') #cek hack changed to zeros
                #self.one[(l,ci)] /= float(len(m.u[ci].dof.flat))
                m.timeIntegration.flcbdf[('u',ci)].setTolerances(1.0e8,1.0e8,self.one[(l,ci)])
        self.flcbdfListFlat=[]
        for flcbdfDict in self.flcbdfList:
            for flcbdf in list(flcbdfDict.values()):
                self.flcbdfListFlat.append(flcbdf)
    def setInitialGuess(self,uList,rList):
        for m,u,r in zip(self.model.levelModelList,uList,rList):
            for ci in range(m.coefficients.nc):
                #pass#cek hack, looks like initial guess is bad
                m.timeIntegration.flcbdf[('u',ci)].setInitialGuess(m.u[ci].dof)
            m.setFreeDOF(u)
            m.getResidual(u,r)
    def set_dt_allLevels(self):
        self.t_model = self.t_model_last + self.dt_model
        for m in self.model.levelModelList:
            m.timeIntegration.set_dt(self.dt_model)
        self.dt_model_last = self.dt_model
    def retryStep_solverFailure(self):
        self.solverFailures+=1
        retry = False
        if self.solverFailures < self.maxSolverFailures:
            self.resetSolution()
            self.dt_model = min([flcbdf.retryStep_solverFailure() for flcbdf in self.flcbdfListFlat])
            self.set_dt_allLevels()
            self.setSubsteps([self.t_model])
            retry = True
        return retry
    def retryStep_errorFailure(self):
        self.errorFailures+=1
        retry = False
        if self.errorFailures < self.maxErrorFailures:
            self.resetSolution()
            self.dt_model = max(self.dt_min,min([flcbdf.retryStep_errorFailure() for flcbdf in self.flcbdfListFlat]))
            self.set_dt_allLevels()
            self.setSubsteps([self.t_model])
            retry = True
        return retry
    def errorFailure(self):
        ERROR_OK=False
        for model in self.model.levelModelList:
            ERROR_OK = model.timeIntegration.lastStepErrorOk()
        if self.ignoreErrorControl==True:
            ERROR_OK=True
        if self.dt_min > 0.0 and self.dt_model <= self.dt_min*(1.0+1.0e-12):
            ERROR_OK=True#at the floor: accept
        return not ERROR_OK
    def syncAfterRestore(self):
        """Hot start: the step the archived run chose next is the integrator's
        restored dt, not the one initialize_dt_model estimated from the restart
        state; make the controller agree (the system controller then chooses
        from it, as it does at the end of every step)."""
        for m in self.model.levelModelList:
            self.dt_model = m.timeIntegration.dt
        self.dt_model_last = self.dt_model
        self.t_model = self.t_model_last + self.dt_model
    def initialize_dt_model(self,t0,tOut):
        self.saveSolution()
        self.t_model_last = t0
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0,tOut,m.q)
        self.dt_model = m.timeIntegration.dt
        self.dt_model_last = self.dt_model
        self.set_dt_allLevels()
        self.setSubsteps([self.t_model])
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                   self.dt_model),
            level=1)
    def choose_dt_model(self):
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = max(self.dt_min,m.timeIntegration.dt)
#         print "Error control dt======================================",m.name,self.dt_model
#         #add check of cfl to make sure time step doesn't get too small
#         maxCFL = 1.0e-6
#         minRunCFL = 0.33
#         for ci in range(self.model.levelModelList[-1].nc):
#             if self.model.levelModelList[-1].q.has_key(('cfl',ci)):
#                 maxCFL=max(maxCFL,globalMax(self.model.levelModelList[-1].q[('cfl',ci)].max()))
#         self.dt_model = max(minRunCFL/maxCFL,m.timeIntegration.dt)
#         if self.dt_model/self.dt_model_last  > self.dtRatioMax:
#             self.dt_model = self.dt_model_last*self.dtRatioMax
#         print "Actual dt======================================",m.name,self.dt_model
#         print "Actual CFL=====================================",m.name,self.dt_model*maxCFL
#         #
#         if self.dt_model != m.timeIntegration.dt:
#             m.timeIntegration.dt = self.dt_model
#             self.ignoreErrorControl=True
#         else:
#             self.ignoreErrorControl=False
#         self.ignoreErrorControl=True#False
        self.dt_model_last = self.dt_model
        self.set_dt_allLevels()
        self.setSubsteps([self.t_model])

class FLCBDF_controller_sys(FLCBDF_controller):
    def __init__(self,model,nOptions):
        FLCBDF_controller.__init__(self,model,nOptions)
    def retryStep_solverFailure(self):
        self.solverFailures+=1
        retry = False
        if self.solverFailures < self.maxSolverFailures:
            self.resetSolution()
            self.dt_model = min([flcbdf.retryStep_solverFailure() for flcbdf in self.flcbdfListFlat])
            self.set_dt_allLevels()
            self.setSubsteps([self.t_model])
            retry = True #used to kick out to system level but now we use this to decide
        return retry
    def retryStep_errorFailure(self):
        self.errorFailures+=1
        retry = False
        if self.errorFailures < self.maxErrorFailures:
            self.resetSolution()
            self.dt_model = max(self.dt_min,min([flcbdf.retryStep_errorFailure() for flcbdf in self.flcbdfListFlat]))
            self.set_dt_allLevels()
            #self.substeps = [self.t_model]
            self.setSubsteps([self.t_model])
            retry = True #used to kick out to system level but now we use this to decide
        return retry

class HeuristicNL_dt_controller(SC_base):
    r"""
    Classical Heuristic step controller that picks time step based on threshholds in nonlinear
    solver iterations::

      if nnl < nonlinearIterationsFloor:
         dt *= dtNLgrowFactor
      else if nnl > nonlinearIterationsCeil:
         dt *= dtNLreduceFactor
      end

    if the nonlinear solver fails, the time step is modified using::

      dt *= dtNLfailureReduceFactor

    Also includes simple linear predictor for initial guess::

      y^{n+1,p} = y^{n} + (y^{n}-y^{n-1})/(\Delta t^{n})(t - t^{n-1})

    TODO:
     Implementation:

     Algorithm:
      Decide if retryStep_solverFailure should retry predictor as well, right now does not
    """
    def __init__(self,model,nOptions):
        import copy
        SC_base.__init__(self,model,nOptions)
        self.nonlinearIterationsFloor = 4
        self.nonlinearIterationsCeil  = 8
        self.dtNLgrowFactor  = 2.0
        self.dtNLreduceFactor= 0.5
        self.dtNLfailureReduceFactor = 0.5
        self.useInitialGuessPredictor = False
        self.predictorHistoryIsValid  = False
        for flag in ['nonlinearIterationsFloor',
                     'nonlinearIterationsCeil',
                     'dtNLgrowFactor',
                     'dtNLreduceFactor',
                     'dtNLfailureReduceFactor',
                     'useInitialGuessPredictor',
                     'stepExact']:
            if flag in dir(nOptions):
                val = getattr(nOptions,flag)
                setattr(self,flag,val)
        self.dt_nm1 = None
        self.unm1ListSave = []
        for mi,u in zip(self.model.levelModelList,self.model.uList):
            self.unm1ListSave.append(copy.deepcopy(u))

    def setInitialGuess(self,uList,rList):
        if self.useInitialGuessPredictor and self.predictorHistoryIsValid:
            assert len(self.unm1ListSave) == len(uList) and len(uList) == len(self.uListSave)
            for m,r,u,un,unm1 in zip(self.model.levelModelList,rList,uList,self.uListSave,self.unm1ListSave):
                u[:] = un[:]
                u   -= unm1
                u   *= (self.t_model-self.t_model_last)/(self.dt_nm1 + 1.0e-16)
                u   += un
                m.setFreeDOF(u)
                m.getResidual(u,r)
    def saveSolution(self):
        for u,r,uSave,rSave,unSave in zip(self.model.uList,
                                          self.model.rList,
                                          self.uListSave,
                                          self.rListSave,
                                          self.unm1ListSave):
            unSave[:] = uSave
            uSave[:]=u
            rSave[:]=r


    def retryStep_solverFailure(self):
        """
        nonlinear solver failure just use cut by dtNLfailureReduceFactor
        """
        self.solverFailures += 1
        retry = False
        if self.solverFailures < self.maxSolverFailures:
            self.resetSolution()
            #basic heuristic step selection just reduce by "failure factor"
            self.dt_model *= self.dtNLfailureReduceFactor
            if self.dt_model > self.t_model_last*1.0e-8:
                self.set_dt_allLevels()
                self.setSubsteps([self.t_model])
                retry = True
            else:
                logEvent("Time step reduced to machine precision",level=1)
        return retry
    def retryStep_errorFailure(self):
        self.errorFailures += 1
        retry = False
        #do not allow adaption based on error failure
        return retry
    def choose_dt_model(self):
        """
        choose dt after successful step
        use finest level to pick for now
        """
        ratio = 1.0
        if self.model.solver.solverList[-1].its < self.nonlinearIterationsFloor:
            ratio = self.dtNLgrowFactor
        elif self.model.solver.solverList[-1].its > self.nonlinearIterationsCeil:
            ratio = self.dtNLreduceFactor
        #
        self.dt_model *= ratio
        self.set_dt_allLevels()
        #self.substeps = [self.t_model]
        self.setSubsteps([self.t_model])
    def updateTimeHistory(self,resetFromDOF=False):
        self.writeSolverStatisticsForStep()
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        self.predictorHistoryIsValid = True
        self.dt_nm1 = self.dt_model
        self.t_model_last = self.t_model
        for m in self.model.levelModelList:
            m.updateTimeHistory(resetFromDOF)
            m.timeIntegration.updateTimeHistory(resetFromDOF)
    def errorFailure(self):
        return False

class GustafssonFullNewton_dt_controller(SC_base):
    r"""
    Try version of basic Gustafsson and Soederlind 97 time step selection strategy
      that accounts for nonlinear solver performance assuming a full Newton nonlinear
      solver

    Also includes error control based on classical "deadbeat" control
    Right now, decisions based on finest level solve::

      input: dt_prev
      output: dt

      get time step estimate based on temporal error --> dt_e
      get convergence rate estimate from nonlinear solver --> a
      get number of iterations from nonlinear solver --> nnl

      if nonlinear solver converges

         r_a = phi(a_ref/a)
         r_a = min(r_a_max,max(r_a,r_a_min))

      else

         if a_ref < a    #convergence rate ok but took too many iterations anyway
           r_a = phi(nnl_ref/nnl)
         else
           r_a = phi(a_ref/a)
         #
         r_a = min(r_a_max,max(r_a,r_a_min))
      #

      dt = min(dt_e,r_a dt_prev)

    Here,

     a_ref   -- target convergence rate
     nnl_ref -- target number of nonlinear iterations
     r_a_max -- max growth rate
     r_a_min -- min growth rate

     phi     -- limiter function, defaut is phi(x) = x

    Also includes simple linear predictor for initial guess::

      y^{n+1,p} = y^{n} + (y^{n}-y^{n-1})/(\Delta t^{n})(t - t^{n-1})

    TODO:
     Implementation:
      Figure out if need to change initialize_dt_model
      Incorporate dt from timeIntegrator
     Algorithm:
      Decide if retryStep_solverFailure should retry predictor as well, right now does not
    """
    def __init__(self,model,nOptions):
        from .LinearAlgebraTools import WeightedNorm
        import copy
        SC_base.__init__(self,model,nOptions)
        self.nonlinearGrowthRateMax = 2    #r_a_max
        self.nonlinearGrowthRateMin = 0.5  #r_a_min
        self.nonlinearConvergenceRate_ref  = 0.2  #alpha_ref
        self.nonlinearIterations_ref       = 4    #nnl_ref
        self.useInitialGuessPredictor = False
        self.predictorHistoryIsValid  = False
        #error controll stuff
        self.errorGrowthRateMax = 2    #r_e_max
        self.errorGrowthRateMin = 0.5  #r_e_min
        self.errorSafetyFactor  = 0.8  #
        self.atol_u = nOptions.atol_u  #integration tolerances
        self.rtol_u = nOptions.rtol_u
        self.timeEps = 1.0e-12
        for flag in ['nonlinearGrowthRateMax',
                     'nonlinearGrowthRateMin',
                     'nonlinearConvergenceRate_ref',
                     'nonlinearIterations_ref',
                     'useInitialGuessPredictor',
                     'stepExact',
                     'errorGrowthRateMax',
                     'errorGrowthRateMin',
                     'errorSafetyFactor']:
            if flag in dir(nOptions):
                val = getattr(nOptions,flag)
                setattr(self,flag,val)
        self.dt_nm1 = None
        self.unm1ListSave = []
        for mi,u in zip(self.model.levelModelList,self.model.uList):
            self.unm1ListSave.append(copy.deepcopy(u))
        #force this
        nOptions.computeNonlinearSolverRates = True
        self.useTemporalErrorEstimate = self.model.levelModelList[-1].timeIntegration.isAdaptive
        self.errorNorm = None
        if self.useTemporalErrorEstimate and self.model.levelModelList[-1].timeIntegration.error_estimate is not None:
            self.errorNorm = {}
            for ci in list(self.model.levelModelList[-1].timeIntegration.error_estimate.keys()):
                self.errorNorm[ci] = WeightedNorm(self.model.levelModelList[-1].timeIntegration.error_estimate[ci].shape,
                                                  self.atol_u[ci],self.rtol_u[ci])

        #if using weighted norms
        self.timeErrorTolerance = 1.0
        #how to pick initial time step
        #can use norm of mt or cfl or something else
        self.use_cfl_for_initial_dt = True
        self.cfl_for_initial_dt = 0.001
    def setInitialGuess(self,uList,rList):
        """
        for now ignore time integrations predictor since that is for m and not u by default ...

        TODO:
           Need a good place to trigger predictor calculation in timeIntegration
        """
        #this would set m^{n+1,p} maybe u^{n+1,p}?
        for m in self.model.levelModelList:
            m.timeIntegration.setInitialGuess()
        #This sets the predictor for the solution, could overwrite u^{n+1,p} if done
        #in timeIntegration, will effect nonlinear solve
        if self.useInitialGuessPredictor and self.predictorHistoryIsValid:
            assert len(self.unm1ListSave) == len(uList) and len(uList) == len(self.uListSave)
            for m,r,u,un,unm1 in zip(self.model.levelModelList,rList,uList,self.uListSave,self.unm1ListSave):
                u[:] = un[:]
                u   -= unm1
                u   *= (self.t_model-self.t_model_last)/(self.dt_nm1 + 1.0e-16)
                u   += un
                m.setFreeDOF(u)
                m.getResidual(u,r)
    def saveSolution(self):
        for u,r,uSave,rSave,unSave in zip(self.model.uList,
                                          self.model.rList,
                                          self.uListSave,
                                          self.rListSave,
                                          self.unm1ListSave):
            unSave[:] = uSave
            uSave[:]=u
            rSave[:]=r


    def retryStep_solverFailure(self):
        """
        nonlinear solver failure
        TODO:
           make sure predictor gets called again after this to get error
           estimate correct for next solve
        """
        self.solverFailures += 1
        retry = False
        if self.solverFailures < self.maxSolverFailures:
            self.resetSolution()
            self.dt_model = self.choose_dt_solverFailure(self.dt_model)
            if self.dt_model > self.t_model_last*1.0e-8:
                self.set_dt_allLevels()
                self.setSubsteps([self.t_model])
                retry = True
                #mwf this has got to go somewhere else
                #for m in self.model.levelModelList:
                #    m.timeIntegration.setInitialGuess()
            else:
                logEvent("Time step reduced to machine precision",level=1)
                self.writeSolverStatisticsForStep()
        else:
            self.writeSolverStatisticsForStep()
        return retry
    def choose_dt_solverFailure(self,dt):
        alpha = self.model.solver.solverList[-1].gustafsson_alpha
        nnl   = self.model.solver.solverList[-1].its
        alpha_ref = self.nonlinearConvergenceRate_ref
        nnl_ref   = float(self.nonlinearIterations_ref)
        r_a   = 1.0
        if alpha_ref > alpha:
            assert nnl > 0.0
            r_a = self.phi(nnl_ref/nnl)
        else:
            assert alpha > 0.0
            r_a = self.phi(alpha_ref/alpha)
        r = min(self.nonlinearGrowthRateMax,max(self.nonlinearGrowthRateMin,r_a))
        dtout = dt*r
        #mwf debug
        if r >= 1.0:
            import pdb
            pdb.set_trace()
        assert r < 1.0, "Gustaffson solver failure r= %s should have r decrease dt_in= %s alpha=%s alpha_ref=%s nnl=%s nnl_ref=%s r_a=%s, dtout=%s " % (r,dt,alpha,alpha_ref,nnl,nnl_ref,r_a,dtout)
        logEvent("Gustafsson solver failure dt_in= %s alpha=%s alpha_ref=%s nnl=%s nnl_ref=%s r_a=%s, dtout=%s " % (dt,alpha,alpha_ref,nnl,nnl_ref,r_a,dtout),level=1)
        return dtout
    def choose_dt_solverSuccess(self,dt):
        alpha = self.model.solver.solverList[-1].gustafsson_alpha
        nnl   = self.model.solver.solverList[-1].its
        alpha_ref = self.nonlinearConvergenceRate_ref
        nnl_ref   = float(self.nonlinearIterations_ref)
        assert alpha >= 0.0 or nnl == 1, "invalid alpha = %s nnl = %d " % (alpha,nnl)
        if alpha <= 0.0:
            r_a = self.nonlinearGrowthRateMax #could use nnl here
        else:
            r_a = self.phi(alpha_ref/alpha)
        r = min(self.nonlinearGrowthRateMax,max(self.nonlinearGrowthRateMin,r_a))
        dtout = dt*r
        logEvent("Gustafsson solver success dt_in= %s alpha=%s alpha_ref=%s nnl=%s nnl_ref=%s r_a=%s, dtout=%s " % (dt,alpha,alpha_ref,nnl,nnl_ref,r_a,dtout),level=1)
        return dtout

    def retryStep_errorFailure(self):
        """
        figure out where to make sure that predictor gets called by timeIntegration to setup error estimates
        """
        self.errorFailures += 1
        retry = False
        if self.errorFailures < self.maxErrorFailures:
            self.resetSolution()
            #find min (over levels) time step for error considerations
            dt_e = self.choose_dt_fromError(self.dt_model)
            dt_a = self.choose_dt_solverSuccess(self.dt_model)
            if self.useTemporalErrorEstimate:
                self.dt_model = min(dt_e,dt_a)
            else:
                self.dt_model = dt_a
            if self.dt_model > self.t_model_last*1.0e-8:
                self.set_dt_allLevels()
                self.setSubsteps([self.t_model])
                #mwf this has got to go somewhere else
                #for m in self.model.levelModelList:
                #    m.timeIntegration.setInitialGuess()

                logEvent("Gustafsson error failure dt_e= %s dt_a=%s dt_model=%s" % (dt_e,dt_a,self.dt_model),level=1)
                retry = True
            else:
                logEvent("Time step reduced to machine precision",level=1)
                self.writeSolverStatisticsForStep()

        else:
            self.writeSolverStatisticsForStep()

        return retry
    def estimateError(self):
        #mwf debug
        #import pdb
        #pdb.set_trace()
        #put call for time integrator's estimate error directly?
        mFine = self.model.levelModelList[-1]
        if (not mFine.timeIntegration.provides_dt_estimate and
            mFine.timeIntegration.error_estimate is not None and
            self.useTemporalErrorEstimate):
            error = mFine.timeIntegration.error_estimate
            localError = {}
            for ci in list(error.keys()):
                localError[ci] = self.errorNorm[ci].norm(error[ci],2)
            self.errorEstimate = max(localError.values())
        else:
            self.errorEstimate = None
        logEvent("Gustafsson estimateError t=%s dt=%s error= %s" % (self.t_model,self.dt_model,self.errorEstimate))

    def choose_dt_fromError(self,dtIn):
        """
        pick dt based on error considerations, assumes error is already calculated
        """
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        mFine = self.model.levelModelList[-1]

        if (not mFine.timeIntegration.provides_dt_estimate and
            mFine.timeIntegration.error_estimate is not None):
            ordInv = 1.0/(mFine.timeIntegration.timeOrder+1.)
            minErr = max(self.errorEstimate,self.timeEps)
            r  = self.errorSafetyFactor*(self.timeErrorTolerance/minErr)**ordInv
            r_e = min(self.errorGrowthRateMax,max(self.errorGrowthRateMin,r))
            dt_e = r_e*dtIn
            logEvent("Gustafsson choose_dt_fromError self t=%s dt=%s error= %s minErr= %s r_e=%s r= %s" % (self.t_model,self.dt_model,
                                                                                                      self.errorEstimate,
                                                                                                      minErr,
                                                                                                      r_e,r))
            return dt_e
        else:
            assert mFine.timeIntegration.provides_dt_estimate
            return mFine.timeIntegration.dt
    def initialize_dt_model(self,t0,tOut):
        """
        TODO: Figure out good strategy for picking initial dt since we don't necessarily want
         time integration to be responsible for this right now
        """
        self.saveSolution()
        self.t_model_last=t0
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0,tOut,m.q)
        if self.use_cfl_for_initial_dt:
            maxCFL = 1.0e-6
            for ci in range(m.nc):
                if ('cfl',ci) in m.q:
                    maxCFL=max(maxCFL,globalMax(m.q[('cfl',ci)].max()))
                    #mwf debug
                    logEvent("Gustafsson cfl initial step ci = %s maxCFL= %s " % (ci,maxCFL))
            self.dt_model = min(self.cfl_for_initial_dt/maxCFL,m.timeIntegration.dt)
        else:
            self.dt_model = m.timeIntegration.dt
        #put safety factor in as in FLCBDF?
        self.dt_model = min(self.dt_model,1.0e-3*(tOut-t0))
        self.set_dt_allLevels()
        self.setSubsteps([self.t_model])
        logEvent("Gustafsson Initializing time step on model %s to dt = %12.5e t_model_last= %s" % (self.model.name,
                                                                                               self.dt_model,
                                                                                               self.t_model_last),
            level=1)
    def choose_dt_model(self):
        """
        choose dt after successful step
        use finest level to pick for now

        assumes estimateError already called in errorFailure
        """
        #find min (over levels) time step for error considerations
        dt_e = self.choose_dt_fromError(self.dt_model)
        dt_a = self.choose_dt_solverSuccess(self.dt_model)
        if self.useTemporalErrorEstimate:
            self.dt_model = min(dt_e,dt_a)
        else:
            self.dt_model = dt_a
        self.set_dt_allLevels()
        self.setSubsteps([self.t_model])
        logEvent("Gustafsson choose_dt_model dt_e= %s dt_a=%s t_model_last= %s dt_model=%s t_model= %s" % (dt_e,dt_a,self.t_model_last,self.dt_model,self.t_model),level=1)
    def updateTimeHistory(self,resetFromDOF=False):
        logEvent("Gustafsson updateTimeHistory t_model_last= %s dt_model=%s t_model=%s" % (self.t_model_last,self.dt_model,self.t_model),level=1)
        self.writeSolverStatisticsForStep()
        self.solverFailures=0
        self.errorFailures=0
        self.saveSolution()
        self.predictorHistoryIsValid = True
        self.dt_nm1 = self.dt_model
        self.t_model_last = self.t_model
        for m in self.model.levelModelList:
            m.updateTimeHistory(resetFromDOF)
            m.timeIntegration.updateTimeHistory(resetFromDOF)
        #this needs to be more general to allow not just setting based on m
        if self.errorNorm is not None:
            for ci in range(self.model.levelModelList[-1].nc):
                if ('m',ci) in self.model.levelModelList[-1].q:
                    self.errorNorm[ci].setWeight(self.model.levelModelList[-1].q[('m',ci)])
    def errorFailure(self):
        #figure out how to pick error estimates, just try fine for now
        mFine = self.model.levelModelList[-1]
        #need to go through all the levels
        timeIntegratorOk = mFine.timeIntegration.lastStepErrorOk()
        self.estimateError()
        if mFine.timeIntegration.provides_dt_estimate:
            return not timeIntegratorOk
        return self.errorEstimate >= self.timeErrorTolerance
    def phi(self,x):
        return float(x)
    def set_dt_allLevels(self):
        self.t_model = self.t_model_last + self.dt_model
        for m in self.model.levelModelList:
            m.timeIntegration.set_dt(self.dt_model)

    def stepExact_model(self,tOut):
        if self.t_model > tOut - tOut*1.0e-8:
            logEvent("StepControl Gustafsson stepExact t_model= %s tOut= %s t_model_last= %s dt_model= %s setting to %s " % (self.t_model,tOut,self.t_model_last,
                                                                                                                  self.dt_model,tOut-self.t_model_last),1)
            self.dt_model = tOut - self.t_model_last
            self.set_dt_allLevels()
            #self.substeps = [self.t_model]
            self.setSubsteps([self.t_model])
