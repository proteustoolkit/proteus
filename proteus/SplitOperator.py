"""
A class hierarchy for split operator methods.

.. inheritance-diagram:: proteus.SplitOperator
   :parts: 1
"""
from .Profiling import logEvent 

class System(object):
    def __init__(self):
        self.name="Default System"

defaultSystem = System()

class SO_base(object):
    """Base class for operating splitting methods for systems.

    The base class implements sequential splitting with a fixed time
    step based on the input parameter `default_so.dt_system`. If
    `default_so.systemStepExact` is True then the time step will be
    reduced when needed to match the output times in
    `default_so.tnList`, otherwise the output will be the first time
    step after each step in `tnList`.

    Here each model takes the same fixed time step

    """
    def __init__(self,modelList,system=defaultSystem,stepExact=True):
        self.system=system
        self.dt_system = 1.0
        self.t_system_last=0.0
        self.t_system=1.0
        self.modelList=modelList
        self.exitModelStep={}
        for model in self.modelList:
            self.exitModelStep[model] = False
        self.stepExact = stepExact
        self.updateAfterModelStep=True
        self.stepExactEps = 1.0e-12
        self.stepFailures = 0
        #self.atol_iso=1.e-8 #tjp added for iterative SO routine tolerance
    def state(self):
        """Nothing to carry beyond the solution (the state contract, see TimeIntegration.TI_base.state)."""
        return None

    def set_state(self, state):
        """Restore what ``state()`` returned; see TimeIntegration.TI_base.state."""
        pass

    def converged(self):
        #no iteration
        if self.its > 0:
            self.its=0
            return True
        else:
            return False
    def stepExact_system(self,tExact):
        if (self.dt_system > 0.0):
            if(self.t_system_last + self.dt_system >= tExact*(1.0-self.stepExactEps)):
                logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                self.dt_system = tExact - self.t_system_last
                logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            elif( tExact*(1.0+self.stepExactEps) - (self.t_system_last + self.dt_system) < self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                self.dt_system = (tExact - self.t_system_last)/2.0
                logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
        if (self.dt_system < 0.0):
            if(self.t_system_last + self.dt_system <= tExact*(1.0 + self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
            elif( tExact - (self.t_system_last + self.dt_system) > self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                self.dt_system = (tExact - self.t_system_last)/2.0
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            model.stepController.setSubsteps([self.t_system])
    def choose_dt_system(self):
        #fixed step
        self.dt_system = self.dt_system_fixed
        self.t_system = self.t_system_last+self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            model.stepController.setSubsteps([self.t_system])
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = self.dt_system_fixed
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,self.dt_system),level=1)
        logEvent("Initializing step sequence  for system %s to %s" %
            (self.system.name,self.stepSequence),level=1)
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
    def updateTimeHistory(self):
        #update step
        self.t_system_last = self.t_system
    def retryModelStep_solverFailure(self,model):
        self.retry = model.stepController.retryStep_solverFailure()
        self.maxFailures = model.stepController.maxSolverFailures
        return False#kick out to sequence level
    def retryModelStep_errorFailure(self,model):
        self.retry = model.stepController.retryStep_errorFailure()
        self.maxFailures = model.stepController.maxErrorFailures
        return False
    def ignoreSequenceStepFailure(self,model):
        return False#kick out to sequence level
    def retrySequence_modelStepFailure(self):
        #retry whole sequence with new min time step
        self.stepFailures +=1
        if (self.stepFailures < self.maxFailures and self.retry):
            self.choose_dt_system()
            return True
        else:
            return False
    # def retryModelStep_solverFailure(self,model):
    #     return model.stepController.retryStep_solverFailure()
    # def retryModelStep_errorFailure(self,model):
    #     return model.stepController.retryStep_errorFailure()
    # def ignoreSequenceStepFailure(self,model):
    #     return False#don't try to recover
    # def retrySequence_modelStepFailure(self):
    #     return False#don't try to recover
    def modelStepTaken(self,model,t_stepSequence):
        logEvent("SO_base modelStepTaken for model= %s t_system_last= %s t_model_last= %s  setting to t_stepSequence= %s " % (model.name,
                                                                                                                         self.t_system_last,
                                                                                                                         model.stepController.t_model_last,
                                                                                                                         t_stepSequence),level=3)
        self.stepFailures=0
        model.calculateAuxiliaryQuantitiesAfterStep()
        model.stepController.t_model_last = t_stepSequence

    def sequenceStepTaken(self,model):
        self.stepFailures=0
    def sequenceTaken(self):
        #this is called when the sequence of steps  is done
        self.its += 1
        for model in self.modelList:
            model.stepController.updateTimeHistory()
            model.stepController.choose_dt_model()
    # tjp added for split operator class
    def SysNorm(self,rSys=0):
        """ Compute the maximum discrete residual value from both models"""
        pass
    def setFromOptions(self,soOptions):
        """
        allow classes to set various numerical parameters
        """
        self.stepExact = soOptions.systemStepExact
        self.dt_system_fixed = soOptions.dt_system_fixed

Sequential_FixedStep = SO_base

class Sequential_tnList(SO_base):
    def __init__(self,modelList,system=defaultSystem,stepExact=True):
        stepExact=True
        SO_base.__init__(self,modelList,system,stepExact)
        for m in modelList:
            m.stepController.stepExact=True
    def stepExact_system(self,tExact):
        if (self.dt_system > 0.0):
            if(self.t_system_last + self.dt_system >= tExact*(1.0-self.stepExactEps)):
                logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                self.dt_system = tExact - self.t_system_last
                logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            elif( self.t_system_last + self.dt_system < tExact ): 
                logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                self.dt_system = tExact - self.t_system_last
                logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
        if (self.dt_system < 0.0):
            if(self.t_system_last + self.dt_system <= tExact*(1.0 + self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
            elif( tExact - (self.t_system_last + self.dt_system) > tExact ):
                self.dt_system = (tExact - self.t_system_last)
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            model.stepController.setSubsteps([self.t_system])
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = tOut - self.t_system_last
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,self.dt_system),level=1)
        logEvent("Initializing step sequence  for system %s to %s" %
            (self.system.name,self.stepSequence),level=1)
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
    def choose_dt_system(self):
        #fixed step
        self.t_system = self.t_system_last+self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
    
Sequential_FixedStep_Simple = Sequential_tnList

class Sequential_NonUniformFixedStep(SO_base):
    """
    Just step to system output time levels, allowing models to
    substep if necessary to system steps

    """
    def __init__(self,modelList,system=defaultSystem,stepExact=True):
        SO_base.__init__(self,modelList,system,stepExact=True)#force?
        for model in self.modelList:
            model.stepController.stepExact = True#mwf should be True
    def stepExact_system(self,tExact):
        old = False
        if old:
            if (self.dt_system > 0.0 and
                self.t_system_last + self.dt_system >=  tExact*(1.0-self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
            elif (self.dt_system < 0.0 and
                self.t_system_last + self.dt_system <=  tExact*(1.0 + self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
        else:
            if (self.dt_system > 0.0):
                if(self.t_system_last + self.dt_system >= tExact*(1.0-self.stepExactEps)):
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = tExact - self.t_system_last
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
                elif( tExact - (self.t_system_last + self.dt_system) < self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = (tExact - self.t_system_last)/2.0
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            if (self.dt_system < 0.0):
                if(self.t_system_last + self.dt_system <= tExact*(1.0 + self.stepExactEps)):
                    self.dt_system = tExact - self.t_system_last
                elif( tExact - (self.t_system_last + self.dt_system) > self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    self.dt_system = (tExact - self.t_system_last)/2.0
        self.dt_system = tExact - self.t_system_last
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
    def choose_dt_system(self):
        #fixed system step
        self.t_system = self.t_system_last+self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.set_dt_allLevels()
    def modelStepTaken(self,model,t_stepSequence):
        self.stepFailures=0
        model.calculateAuxiliaryQuantitiesAfterStep()
        #model.stepController.t_model_last = t_stepSequence
        model.stepController.updateTimeHistory()
        model.stepController.choose_dt_model()
    def sequenceTaken(self):
        #this is called when the sequence of steps  is done
        self.its += 1
        #for model in self.modelList:
        #    model.stepController.updateTimeHistory()
        #    model.stepController.choose_dt_model()


class Sequential_MinModelStep(SO_base):
    """
    Look at the minimum model step and make that the system step as
    well as all the model steps

    Force models to step exactly to each with substepping if necessary,
      but this isn't strictly necessary
    """
    def __init__(self,modelList,system=defaultSystem,stepExact=True):
        SO_base.__init__(self,modelList,system,stepExact)
        for model in self.modelList:
            model.stepController.stepExact = True
    def converged(self):
        #no iteration
        if self.its > 0:
            self.its=0
            return True
        else:
            return False
    def stepExact_system(self,tExact):
        old = False
        if old:
            if (self.dt_system > 0.0 and
                self.t_system_last + self.dt_system >=  tExact*(1.0-self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
            elif (self.dt_system < 0.0 and
                self.t_system_last + self.dt_system <=  tExact*(1.0 + self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
        else:
            if (self.dt_system > 0.0):
                if(self.t_system_last + self.dt_system >= tExact*(1.0-self.stepExactEps)):
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = tExact - self.t_system_last
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
                elif( tExact - (self.t_system_last + self.dt_system) < self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = (tExact - self.t_system_last)/2.0
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            if (self.dt_system < 0.0):
                if(self.t_system_last + self.dt_system <= tExact*(1.0 + self.stepExactEps)):
                    self.dt_system = tExact - self.t_system_last
                elif( tExact - (self.t_system_last + self.dt_system) > self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    self.dt_system = (tExact - self.t_system_last)/2.0
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            #model.stepController.substeps= [self.t_system]
            model.stepController.setSubsteps([self.t_system])

    def choose_dt_system(self):
        self.dt_system = min([model.stepController.dt_model for model  in self.modelList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,model) for model in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
        logEvent("SplitOperator_Min choose_dt_system t_system_last= %s dt_system= %s t_system= %s " % (self.t_system_last,
                                                                                                  self.dt_system,
                                                                                                  self.t_system),3)
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = min(min([model.stepController.dt_model for model in self.modelList]),tOut-t0)
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.t_model = self.t_system
            model.stepController.set_dt_allLevels()
            model.stepController.initializeTimeHistory()
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,
             self.dt_system),
            level=1)
        logEvent("Initializing step sequence for system %s to %s" %
            (self.system.name,
             self.stepSequence),
            level=1)
class Sequential_MinFLCBDFModelStep(SO_base):
    """
    Look at the minimum model step and make that the system step as
    well as all the model steps
    """
    def __init__(self,modelList,system=defaultSystem,stepExact=False):
        from .StepControl import FLCBDF_controller
        SO_base.__init__(self,modelList,system,stepExact)
        self.flcbdfList = []
        for model  in self.modelList:
            if isinstance(model.stepController,FLCBDF_controller):
                self.flcbdfList.append(model)
                self.maxFailures = model.stepController.maxSolverFailures
        self.stepFailures=0
        self.updateAfterModelStep=False
    def converged(self):
        #no iteration
        if self.its > 0:
            self.its=0
            return True
        else:
            return False
    def stepExact_system(self,tExact):
        #mwf needs to be checked
        if (self.dt_system > 0.0 and
            self.t_system_last + self.dt_system >=  tExact*(1.0-self.stepExactEps)):
            self.dt_system = tExact - self.t_system_last
        elif (self.dt_system < 0.0 and
            self.t_system_last + self.dt_system <=  tExact*(1.0 + self.stepExactEps)):
            self.dt_system = tExact - self.t_system_last
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            #model.stepController.substeps= [self.t_system]
            model.stepController.setSubsteps([self.t_system])

    def choose_dt_system(self):
        self.dt_system = min([model.stepController.dt_model for model  in self.flcbdfList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,model) for model in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = min([model.stepController.dt_model for model in self.modelList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.t_model = self.t_system
            model.stepController.set_dt_allLevels()
            model.stepController.initializeTimeHistory()
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,
             self.dt_system),
            level=1)
        logEvent("Initializing step sequence for system %s to %s" %
            (self.system.name,
             self.stepSequence),
            level=1)
    def retryModelStep_solverFailure(self,model):
        self.retry = self._retryStep(model,"retryStep_solverFailure")
        return False#kick out to sequence level
    def retryModelStep_errorFailure(self,model):
        self.retry = self._retryStep(model,"retryStep_errorFailure")
        return False
    def _retryStep(self,model,method):
        """Retry the sequence with a smaller step. A model whose controller is
        not FLCBDF (redistancing, mass correction) cannot shorten the step
        itself: the FLCBDF models do, and the sequence runs again from the
        solutions they restore (the others restart from those)."""
        if model in self.flcbdfList:
            return getattr(model.stepController,method)()
        return all([getattr(m.stepController,method)() for m in self.flcbdfList])
    def ignoreSequenceStepFailure(self,model):
        return False#kick out to sequence level
    def retrySequence_modelStepFailure(self):
        #retry whole sequence with new min time step
        self.stepFailures +=1
        if (self.stepFailures < self.maxFailures and self.retry):
            self.choose_dt_system()
            return True
        else:
            return False
    def modelStepTaken(self,model,t_stepSequence):
        self.stepFailures=0
        if model.stepController.t_model >= t_stepSequence:
            self.exitModelStep[model] = True
        model.calculateAuxiliaryQuantitiesAfterStep()
    def sequenceStepTaken(self,model):
        self.stepFailures=0
        self.exitModelStep[model] = False #reset to False
    def sequenceTaken(self):
        #this is called when the sequence of steps  is done
        self.its += 1
        #do not need to do step exact here, because taken care of by
        #systemStepController forcing lock-step?
        for model in self.modelList:
            model.stepController.updateTimeHistory()
            model.stepController.choose_dt_model()
            #recompute auxiliary variables here?
class Sequential_MinAdaptiveModelStep(SO_base):
    """
    Look at the minimum model step and make that the system step as
    well as all the model steps
    """
    def __init__(self,modelList,system=defaultSystem,stepExact=False):
        from .StepControl import FLCBDF_controller
        SO_base.__init__(self,modelList,system,stepExact)
        self.controllerList = []
        for model  in self.modelList:
            if model.levelModelList[-1].timeIntegration.isAdaptive:
                self.controllerList.append(model)
                self.maxFailures = model.stepController.maxSolverFailures
        #print "controllers",[c.name for c in self.controllerList]
        self.stepFailures=0
        self.updateAfterModelStep=False
    def converged(self):
        #no iteration
        if self.its > 0:
            self.its=0
            return True
        else:
            return False
    def stepExact_system(self,tExact):
        #cek try to prevent cutting dt by more  than a factor of 2
        #todo, make standard approach?
        old=False
        if old:
            if (self.dt_system > 0.0 and
                self.t_system_last + self.dt_system >=  tExact*(1.0-self.stepExactEps)):
                logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                self.dt_system = tExact - self.t_system_last
                logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            elif (self.dt_system < 0.0 and
                  self.t_system_last + self.dt_system <=  tExact*(1.0 + self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
        else:
            if (self.dt_system > 0.0):
                if(self.t_system_last + self.dt_system >= tExact*(1.0-self.stepExactEps)):
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = tExact - self.t_system_last
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
                elif( tExact - (self.t_system_last + self.dt_system) < self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = (tExact - self.t_system_last)/2.0
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            if (self.dt_system < 0.0):
                if(self.t_system_last + self.dt_system <= tExact*(1.0 + self.stepExactEps)):
                    self.dt_system = tExact - self.t_system_last
                elif( tExact - (self.t_system_last + self.dt_system) > self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    self.dt_system = (tExact - self.t_system_last)/2.0
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            #model.stepController.substeps= [self.t_system]
            model.stepController.setSubsteps([self.t_system])

    def choose_dt_system(self):
        if self.dt_system_fixed is not None:
            self.dt_system = min([model.stepController.dt_model for model  in self.controllerList]+[self.dt_system_fixed])
        else:
            self.dt_system = min([model.stepController.dt_model for model  in self.controllerList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,model) for model in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            #mwf this is not done in flcbdf system controller, should it be?
            #model.stepController.t_model = self.t_system
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = min(tOut-t0,min([model.stepController.dt_model for model in self.controllerList]))
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.t_model = self.t_system
            model.stepController.set_dt_allLevels()
            model.stepController.initializeTimeHistory()
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,
             self.dt_system),
            level=1)
        logEvent("Initializing step sequence for system %s to %s" %
            (self.system.name,
             self.stepSequence),
            level=1)
    def retryModelStep_solverFailure(self,model):
        self.failureType='solver'
        return False#kick out to sequence level
    def retryModelStep_errorFailure(self,model):
        self.failureType='error'
        return False
    def ignoreSequenceStepFailure(self,model):
        return False#kick out to sequence level
    def retrySequence_modelStepFailure(self):
        self.retry=True
        for model in self.controllerList:
            if self.failureType == 'solver':
                self.retry = self.retry and model.stepController.retryStep_solverFailure()
            else:
                self.retry = self.retry and model.stepController.retryStep_errorFailure()
        if self.retry:
            self.stepFailures +=1
            if (self.stepFailures < self.maxFailures and self.retry):
                self.choose_dt_system()
                return True
            else:
                return False
    def modelStepTaken(self,model,t_stepSequence):
        self.stepFailures=0
        if model.stepController.t_model >= t_stepSequence:
            self.exitModelStep[model] = True
        model.calculateAuxiliaryQuantitiesAfterStep()
    def sequenceStepTaken(self,model):
        self.stepFailures=0
        self.exitModelStep[model] = False #reset to False
    def sequenceTaken(self):
        #this is called when the sequence of steps  is done
        #do not need to do step exact here, because taken care of by
        #systemStepController forcing lock-step?
        self.its += 1
        for model in self.modelList:
            model.stepController.updateTimeHistory()
            model.stepController.choose_dt_model()
            #recompute auxiliary variables here?

class ISO_fixed_MinAdaptiveModelStep(SO_base):
    """
    Look at the minimum model step and make that the system step as
    well as all the model steps
    """
    def __init__(self,modelList,system=defaultSystem,stepExact=False):
        from .StepControl import FLCBDF_controller
        SO_base.__init__(self,modelList,system,stepExact)
        self.controllerList = []
        for model  in self.modelList:
            if model.levelModelList[-1].timeIntegration.isAdaptive:
                self.controllerList.append(model)
                self.maxFailures = model.stepController.maxSolverFailures
        #print "controllers",[c.name for c in self.controllerList]
        self.stepFailures=0
        self.updateAfterModelStep=False
    def converged(self):
        #no iteration
        if self.its > 2:
            self.its=0
            return True
        else:
            return False
    def stepExact_system(self,tExact):
        #cek try to prevent cutting dt by more  than a factor of 2
        #todo, make standard approach?
        old=False
        if old:
            if (self.dt_system > 0.0 and
                self.t_system_last + self.dt_system >=  tExact*(1.0-self.stepExactEps)):
                logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                self.dt_system = tExact - self.t_system_last
                logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            elif (self.dt_system < 0.0 and
                  self.t_system_last + self.dt_system <=  tExact*(1.0 + self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
        else:
            if (self.dt_system > 0.0):
                if(self.t_system_last + self.dt_system >= tExact*(1.0-self.stepExactEps)):
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = tExact - self.t_system_last
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
                elif( tExact - (self.t_system_last + self.dt_system) < self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = (tExact - self.t_system_last)/2.0
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            if (self.dt_system < 0.0):
                if(self.t_system_last + self.dt_system <= tExact*(1.0 + self.stepExactEps)):
                    self.dt_system = tExact - self.t_system_last
                elif( tExact - (self.t_system_last + self.dt_system) > self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    self.dt_system = (tExact - self.t_system_last)/2.0
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            #model.stepController.substeps= [self.t_system]
            model.stepController.setSubsteps([self.t_system])

    def choose_dt_system(self):
        self.dt_system = min([model.stepController.dt_model for model  in self.controllerList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,model) for model in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            #mwf this is not done in flcbdf system controller, should it be?
            #model.stepController.t_model = self.t_system
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = min([model.stepController.dt_model for model in self.controllerList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.t_model = self.t_system
            model.stepController.set_dt_allLevels()
            model.stepController.initializeTimeHistory()
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,
             self.dt_system),
            level=1)
        logEvent("Initializing step sequence for system %s to %s" %
            (self.system.name,
             self.stepSequence),
            level=1)
    def retryModelStep_solverFailure(self,model):
        self.failureType='solver'
        return False#kick out to sequence level
    def retryModelStep_errorFailure(self,model):
        self.failureType='error'
        return False
    def ignoreSequenceStepFailure(self,model):
        return False#kick out to sequence level
    def retrySequence_modelStepFailure(self):
        self.retry=True
        for model in self.modelList:
            if self.failureType == 'solver':
                self.retry = self.retry and model.stepController.retryStep_solverFailure()
            else:
                self.retry = self.retry and model.stepController.retryStep_errorFailure()
        if self.retry:
            self.stepFailures +=1
            if (self.stepFailures < self.maxFailures and self.retry):
                self.choose_dt_system()
                return True
            else:
                return False
    def modelStepTaken(self,model,t_stepSequence):
        self.stepFailures=0
        if model.stepController.t_model >= t_stepSequence:
            self.exitModelStep[model] = True
        model.calculateAuxiliaryQuantitiesAfterStep()
    def sequenceStepTaken(self,model):
        self.stepFailures=0
        self.exitModelStep[model] = False #reset to False
    def sequenceTaken(self):
        #this is called when the sequence of steps  is done
        #do not need to do step exact here, because taken care of by
        #systemStepController forcing lock-step?
        self.its += 1
        if self.its > 2:
            for model in self.modelList:
                model.stepController.updateTimeHistory()
                model.stepController.choose_dt_model()
            #recompute auxiliary variables here?

class Sequential_MinAdaptiveModelStep_SS(SO_base):
    """
    Look at the minimum model step and make that the system step as
    well as all the model steps
    """
    def __init__(self,modelList,system=defaultSystem,stepExact=False):
        from .StepControl import FLCBDF_controller
        SO_base.__init__(self,modelList,system,stepExact)
        self.controllerList = []
        for model  in self.modelList:
            if model.levelModelList[-1].timeIntegration.isAdaptive:
                self.controllerList.append(model)
                self.maxFailures = model.stepController.maxSolverFailures
        #print "controllers",[c.name for c in self.controllerList]
        self.stepFailures=0
        self.updateAfterModelStep=False
    def converged(self):
        #no iteration
        if self.its > 0:
            self.its=0
            return True
        else:
            return False
    def stepExact_system(self,tExact):
        #cek try to prevent cutting dt by more  than a factor of 2
        #todo, make standard approach?
        old=False
        if old:
            if (self.dt_system > 0.0 and
                self.t_system_last + self.dt_system >=  tExact*(1.0-self.stepExactEps)):
                logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                self.dt_system = tExact - self.t_system_last
                logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            elif (self.dt_system < 0.0 and
                  self.t_system_last + self.dt_system <=  tExact*(1.0 + self.stepExactEps)):
                self.dt_system = tExact - self.t_system_last
        else:
            if (self.dt_system > 0.0):
                if(self.t_system_last + self.dt_system >= tExact*(1.0-self.stepExactEps)):
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = tExact - self.t_system_last
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
                elif( tExact - (self.t_system_last + self.dt_system) < self.dt_system/2.0 ): #if next step would be within dt/2 ball go ahead and cut a little bit
                    logEvent("===========================================================dt system orig" + str(self.dt_system),level=5)
                    self.dt_system = (tExact - self.t_system_last)/2.0
                    logEvent("=========================================================dt system final" + str(self.dt_system),level=5)
            if (self.dt_system < 0.0):
                if(self.t_system_last + self.dt_system <= tExact*(1.0 + self.stepExactEps)):
                    self.dt_system = tExact - self.t_system_last
                elif( tExact - (self.t_system_last + self.dt_system) > self.dt_system/2.0): #if next step would be within dt/2 ball go ahead and cut a little bit
                    self.dt_system = (tExact - self.t_system_last)/2.0
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            #model.stepController.substeps= [self.t_system]
            model.stepController.setSubsteps([self.t_system])

    def choose_dt_system(self):
        self.dt_system = min([model.stepController.dt_model for model  in self.controllerList])
        self.max_its = max([model.solver.solverList[-1].its for model  in self.controllerList])
        if self.max_its < 3:
            self.dt_system = self.dt_system_last*1.1
        self.dt_system_last = self.dt_system
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,model) for model in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            #mwf this is not done in flcbdf system controller, should it be?
            #model.stepController.t_model = self.t_system
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = min([model.stepController.dt_model for model in self.controllerList])
        self.dt_system_last = self.dt_system
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.t_model = self.t_system
            model.stepController.set_dt_allLevels()
            model.stepController.initializeTimeHistory()
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,
             self.dt_system),
            level=1)
        logEvent("Initializing step sequence for system %s to %s" %
            (self.system.name,
             self.stepSequence),
            level=1)
    def retryModelStep_solverFailure(self,model):
        self.failureType='solver'
        return False#kick out to sequence level
    def retryModelStep_errorFailure(self,model):
        self.failureType='error'
        return False
    def ignoreSequenceStepFailure(self,model):
        return False#kick out to sequence level
    def retrySequence_modelStepFailure(self):
        self.retry=True
        for model in self.modelList:
            if self.failureType == 'solver':
                self.retry = self.retry and model.stepController.retryStep_solverFailure()
            else:
                self.retry = self.retry and model.stepController.retryStep_errorFailure()
        if self.retry:
            self.stepFailures +=1
            if (self.stepFailures < self.maxFailures and self.retry):
                self.choose_dt_system()
                return True
            else:
                return False
    def modelStepTaken(self,model,t_stepSequence):
        self.stepFailures=0
        if model.stepController.t_model >= t_stepSequence:
            self.exitModelStep[model] = True
        model.calculateAuxiliaryQuantitiesAfterStep()
    def sequenceStepTaken(self,model):
        self.stepFailures=0
        self.exitModelStep[model] = False #reset to False
    def sequenceTaken(self):
        #this is called when the sequence of steps  is done
        #do not need to do step exact here, because taken care of by
        #systemStepController forcing lock-step?
        self.its += 1
        for model in self.modelList:
            model.stepController.updateTimeHistory()
            model.stepController.choose_dt_model()
            #recompute auxiliary variables here?

class SequentialNotInOrder_MinFLCBDFModelStep(Sequential_MinFLCBDFModelStep):
    """
    loop through the models in some specified order
    """
    #mwf hack set list for testing
    def __init__(self,modelList,modelSequenceList=[1],system=defaultSystem,stepExact=False):
        Sequential_MinFLCBDFModelStep.__init__(self,modelList,system=system,stepExact=stepExact)
        if modelSequenceList is None:
            self.modelSequenceList = [i for i in range(len(modelList))]
        else:
            self.modelSequenceList = modelSequenceList

    def stepExact_system(self,tExact):
        #mwf needs to be checked
        if (self.dt_system > 0.0 and
            self.t_system_last + self.dt_system >=  tExact*(1.0-self.stepExactEps)):
            self.dt_system = tExact - self.t_system_last
        elif (self.dt_system < 0.0 and
            self.t_system_last + self.dt_system <=  tExact*(1.0 + self.stepExactEps)):
            self.dt_system = tExact - self.t_system_last
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,self.modelList[i]) for i in self.modelSequenceList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            #model.stepController.substeps= [self.t_system]
            model.stepController.setSubsteps([self.t_system])

    def choose_dt_system(self):
        self.dt_system = min([model.stepController.dt_model for model  in self.flcbdfList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,self.modelList[i]) for i in self.modelSequenceList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = min([model.stepController.dt_model for model in self.modelList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,self.modelList[i]) for i in self.modelSequenceList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.t_model = self.t_system
            model.stepController.set_dt_allLevels()
            model.stepController.initializeTimeHistory()
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,
             self.dt_system),
            level=1)
        logEvent("Initializing step sequence for system %s to %s" %
            (self.system.name,
             self.stepSequence),
            level=1)

class SequentialNotInOrder_MinAdaptiveModelStep(Sequential_MinAdaptiveModelStep):
    """
    loop through the models in some specified order
    """
    #mwf hack set default list for testing
    def __init__(self,modelList,modelSequenceList=[1],system=defaultSystem,stepExact=False):
        Sequential_MinAdaptiveModelStep.__init__(self,modelList,system=system,stepExact=stepExact)
        if modelSequenceList is None:
            self.modelSequenceList = [i for i in range(len(modelList))]
        else:
            self.modelSequenceList = modelSequenceList

    def stepExact_system(self,tExact):
        #mwf needs to be checked
        if (self.dt_system > 0.0 and
            self.t_system_last + self.dt_system >=  tExact*(1.0-self.stepExactEps)):
            self.dt_system = tExact - self.t_system_last
        elif (self.dt_system < 0.0 and
            self.t_system_last + self.dt_system <=  tExact*(1.0 + self.stepExactEps)):
            self.dt_system = tExact - self.t_system_last
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,self.modelList[i]) for i in self.modelSequenceList]
        #self.stepSequence=[(self.t_system,m) for m in self.modelList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            model.stepController.t_model = self.t_system
            #model.stepController.substeps= [self.t_system]
            model.stepController.setSubsteps([self.t_system])

    def choose_dt_system(self):
        self.dt_system = min([model.stepController.dt_model for model  in self.controllerList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,self.modelList[i]) for i in self.modelSequenceList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.set_dt_allLevels()
            #mwf this is not done in flcbdf system controller, should it be?
            #model.stepController.t_model = self.t_system
    def initialize_dt_system(self,t0,tOut):
        self.its=0
        self.t_system_last = t0
        self.dt_system = min([model.stepController.dt_model for model in self.modelList])
        self.t_system = self.t_system_last + self.dt_system
        self.stepSequence=[(self.t_system,self.modelList[i]) for i in self.modelSequenceList]
        for model in self.modelList:
            model.stepController.dt_model = self.dt_system
            model.stepController.t_model = self.t_system
            model.stepController.set_dt_allLevels()
            model.stepController.initializeTimeHistory()
        logEvent("Initializing time step on system %s to dt = %12.5e" %
            (self.system.name,
             self.dt_system),
            level=1)
        logEvent("Initializing step sequence for system %s to %s" %
            (self.system.name,
             self.stepSequence),
            level=1)
