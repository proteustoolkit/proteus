"""
A class hierarchy of time discretizations

.. inheritance-diagram:: proteus.TimeIntegration
   :parts: 1
"""
from .LinearAlgebraTools import *
import sys,math
from . import Profiling
from .Profiling import logEvent
from .Comm import globalMax

class TI_base(object):
    r"""
    The base class for time discretizations of tightly coupled systems
    of transport equations.

    The purpose of time integration objects is to construct
    approximations::

      m_t, dm_t -- the time derivatives and their derivatives  w.r.t. u
      u_*, du_* -- the solution variables and their derivatives w.r.t. u
      f_*, df_*,... --the coefficients and their derivatives w.r.t. u

    The objects provide a set_dt function for setting the time step as
    well as functions for SUGGESTING the time step. The StepController
    objects have the final responsibility for choosing the time step
    (by calling set_dt).

    The objects can also enforce a tolerance on an a posteriori error estimate
    using the lastStepErrorOK member function.

    If any of the coefficient approximations are not implicit, a flag
    can be set to pass that information to the transport object.

    This base class implements no time integration
    method and results in the steady state scalar
    transport equation:

    .. math::

       \deld ( f - a \grad \phi ) + r = 0

    NoIntegration is an alias for this class.
    """
    def __init__(self,transport,integrateInterpolationPoints=False):
        """
        Set flags that indicate that all terms
        are implicit.
        """
        self.tLast = 0.0
        self.dt = 1.0
        self.t  = self.tLast + self.dt
        self.transport = transport
        self.nc = transport.coefficients.nc
        self.massComponents = list(transport.coefficients.mass.keys())
        self.u ={}
        self.duStar_du = {}
        self.massIsImplicit = {}
        self.advectionIsImplicit = {}
        self.diffusionIsImplicit = {}
        self.reactionIsImplicit = {}
        self.hamiltonianIsImplicit = {}
        self.stabilizationIsImplicit  = {}
        self.shockCapturingIsImplicit= {}
        for ci in range(self.nc):
            self.u[ci]                        = None
            self.duStar_du[ci]                = None
            self.massIsImplicit[ci]           = True
            self.advectionIsImplicit[ci]      = True
            self.diffusionIsImplicit[ci]      = True
            self.reactionIsImplicit[ci]       = True
            self.hamiltonianIsImplicit[ci]    = True
            self.stabilizationIsImplicit[ci]  = True
            self.shockCapturingIsImplicit[ci] = True
        self.nStages = 1
        self.isAdaptive=False
        self.timeOrder = 1
        self.error_estimate = None#want to calculate this now instead of dt in time integration classes
        self.provides_dt_estimate = True #whether or not
                                         #discretization can compute
                                         #own "dt" based on error or
                                         #stability. Eventually move
                                         #this capability out of
                                         #TimeIntegration
        self.provides_initialGuess = False #does this method compute its own initialGuess for new solution
        self.isSSP=False
        self.alpha_bdf = 0.0
        self.beta_bdf  = {}
        self.m_tmp  = {}
        for ci in self.massComponents:
            if ('m',ci) in transport.q:
                self.m_tmp[ci] = transport.q[('m',ci)].copy()
                self.beta_bdf[ci] = transport.q[('m',ci)].copy()
                self.beta_bdf[ci][:]=0.0
    def calculateU(self,u):
        """
        Generate u_*
        """
        self.u = u
    def calculateElementCoefficients(self,q):
        """
        Calculate m_t, and dm_t and recalculate any of the other coefficients on element interiors
        """
        for ci in range(self.nc):
            if ('mt',ci) in q:
                q[('mt',ci)].fill(0.0)
                for cj in range(self.nc):
                    if ('dmt',ci,cj) in q:
                        q[('dmt',ci,cj)].fill(0.0)
    def calculateElementBoundaryCoefficients(self,ebq):
        """
        Calculate m_t, and dm_t and recalculate any of the other coefficients on element boundaries
        """
        pass
    def calculateExteriorElementBoundaryCoefficients(self,ebqe):
        """
        Calculate m_t, and dm_t and recalculate any of the other coefficients on global element boundaries
        """
        pass
    def calculateStrongElementSpatialResidual(self,q):
        """
        Recalculate the part of the element residual due to spatial terms
        """
        pass
    def calculateElementSpatialResidual(self,elementResidual):
        """
        Recalculate the part of the element residual due to spatial terms
        """
        pass
    def calculateElementSpatialJacobian(self,elementJacobian):
        """
        Recalculate the part of the element Jacobian due to the spatial terms
        """
        pass
    def calculateElementSpatialBoundaryJacobian(self,elementBoundaryJacobian,elementBoundaryJacobian_eb):
        """
        Recalculate the part of the element Jacobian due to the spatial terms
        along element boundaries
        """
        pass
    def calculateExteriorElementSpatialBoundaryJacobian(self,exteriorElementBoundaryJacobian):
        """
        Recalculate the part of the element Jacobian due to the spatial terms
        along element boundaries
        """
        pass
    def calculateGeneralizedInterpolationCoefficients(self,cip):
        """
        Calculate m_t, and dm_t and recalculate any of the other coefficients on element interiors
        """
        pass
    def initialize_dt(self,t0,tOut,q):
        """
        Modify self.dt
        """
        self.tLast=t0
        self.t = tOut
        self.dt = tOut - t0
    def choose_dt(self):
        """
        Modify self.dt
        """
        self.t = self.tLast + self.dt
    def set_dt(self,DTSET):
        self.dt=DTSET
        self.t = self.tLast + self.dt
    def generateSubsteps(self,tList):
        """
        create list of substeps over time values given in tList.
        """
        self.substeps = [t for t in tList]
    def initializeSpaceHistory(self,resetFromDOF=False):
        pass
    def updateTimeHistory(self,resetFromDOF=False):
        """
        Push necessary information into time history arrays
        """
        self.tLast = self.t
    def resetTimeHistory(self,resetFromDOF=False):
        pass #do something to the history
    def initializeTimeHistory(self,resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        #mwf do not want to treat initialization just like accepted step
        #e.g., updating self.tLast = self.t
        #self.updateTimeHistory(resetFromDOF)
        pass
    def updateStage(self):
        """
        increment stage counters and push necessary information
        into stage arrays
        """
        pass
    def setInitialStageValues(self):
        """
        set the stage values assuming this is the first step
        after a problem reset
        """
        pass
    def setInitialGuess(self):
        """
        set an initial guess for the next step
        """
        pass
    def setLastSolveFailed(self,lastSolveFailed):
        """
        tell integrator last attempted time step failed or not
        """
        pass
    def lastStepErrorOk(self):
        """
        was the last time step acceptable or not
        """
        return True
    def calculateCoefs(self):
        """
        calculate any coefficients that depend on new time step
        """
    def state(self):
        """The state contract: what a hot start restores to continue this object exactly.

        Plain data (python scalars and numpy arrays; an alias between two arrays is recorded as a
        flag, not as two arrays) that the solution does not determine and that the spin-up of a
        restart would rebuild differently: the value from the last evaluation inside the previous
        step, a state machine's counters, a multistep history. None when there is nothing to carry.
        ``set_state(state)`` puts it back after the spin-up. NumericalSolution archives it per rank
        (``solver_state/<model>/<owner>``) and only the same partition reads it back.

        This default carries nothing; whether that is right for a class is recorded in
        test/state_contract_inventory.txt, which test/test_state_contract.py keeps complete.
        """
        return None

    def set_state(self, state):
        """Restore what ``state()`` returned; see the contract there."""
        pass

    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        pass
    def postAdaptUpdate(self,oldTime):
        """
        ensure array histories are properly set after adapt. This should only work for backwards euler and vbdf at the moment.
        """
        logEvent("WARNING: calling abstract function that doesn't have a definition for your object so just passing for now")
        pass

NoIntegration = TI_base

class BackwardEuler(TI_base):
    def __init__(self,transport,integrateInterpolationPoints=False):
        TI_base.__init__(self,transport)
        self.integrateInterpolationPoints = integrateInterpolationPoints
        self.m_last={}
        self.m_tmp={}
        self.m_ip_last={}
        self.m_ip_tmp={}
        for ci in self.massComponents:
            if ('m_last',ci) in transport.q:
                self.m_last[ci] = transport.q[('m_last',ci)]
            else:
                self.m_last[ci] = numpy.array(transport.q[('m',ci)])
            if ('m_tmp',ci) in transport.q:
                self.m_tmp[ci] = transport.q[('m_tmp',ci)]
            else:
                self.m_tmp[ci] = numpy.array(transport.q[('m',ci)])
            if self.integrateInterpolationPoints:
                self.m_ip_last[ci] = numpy.array(transport.phi_ip[('m',ci)])
                self.m_ip_tmp[ci]  = numpy.array(transport.phi_ip[('m',ci)])
        #moving to generic bdf interface
        self.m_history = []; self.m_history.append({})
        self.alpha_bdf = 1.0
        self.beta_bdf  = {}
        for ci in self.massComponents:
            self.m_history[0][ci] = self.m_last[ci]
            self.beta_bdf[ci] = numpy.copy(self.m_tmp[ci])

    def calculateElementCoefficients(self,q):
        #for bdf interface
        self.calculateCoefs()
        for ci in self.massComponents:
            self.m_tmp[ci][:] = q[('m',ci)]
            q[('mt',ci)][:]   = q[('m',ci)]
            q[('mt',ci)] -= self.m_last[ci]
            q[('mt',ci)] /= self.dt
            for cj in range(self.nc):
                if ('dmt',ci,cj) in q:
                    q[('dmt',ci,cj)][:] = q[('dm',ci,cj)]
                    q[('dmt',ci,cj)] /= self.dt
                if ('dm_sge',ci,cj) in q and ('dmt_sge',ci,cj) in q:
                    q[('dmt_sge',ci,cj)][:] = q[('dm_sge',ci,cj)]
                    q[('dmt_sge',ci,cj)] /= self.dt
            #print q[('mt',ci)]
    def calculateGeneralizedInterpolationCoefficients(self,cip):
        if not self.integrateInterpolationPoints:
            return
        for ci in list(self.m_ip_last.keys()):
            self.m_ip_tmp[ci][:] = cip[('m',ci)]
            cip[('mt',ci)][:]   = cip[('m',ci)]
            cip[('mt',ci)] -= self.m_ip_last[ci]
            cip[('mt',ci)] /= self.dt
            for cj in range(self.nc):
                if ('dmt',ci,cj) in cip:
                    cip[('dmt',ci,cj)][:] = cip[('dm',ci,cj)]
                    cip[('dmt',ci,cj)] /= self.dt
                if ('dmt_sge',ci,cj) in cip:
                    cip[('dmt_sge',ci,cj)][:] = cip[('dm_sge',ci,cj)]
                    cip[('dmt_sge',ci,cj)] /= self.dt

    def initializeTimeHistory(self,resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        for ci in self.massComponents:
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat
        if self.integrateInterpolationPoints:
            for ci in list(self.m_last.keys()):
                self.m_ip_last[ci].flat[:] = self.m_ip_tmp[ci].flat
    def updateTimeHistory(self,resetFromDOF=False):
        self.tLast = self.t
        for ci in self.massComponents:
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat
        if self.integrateInterpolationPoints:
            for ci in list(self.m_last.keys()):
                self.m_ip_last[ci].flat[:] = self.m_ip_tmp[ci].flat
    def calculateCoefs(self):
        #for bdf interface
        dtInv = 1.0/self.dt
        self.alpha_bdf = dtInv
        for ci in list(self.m_last.keys()):
            self.beta_bdf[ci].flat[:] = self.m_last[ci].flat
            self.beta_bdf[ci] *= -dtInv

class BackwardEuler_cfl(BackwardEuler):
    """
    Take a fraction of the max (over all components and elements)
    """
    def __init__(self,transport,runCFL=0.9,integrateInterpolationPoints=False):
        BackwardEuler.__init__(self,transport,integrateInterpolationPoints=integrateInterpolationPoints)
        self.runCFL=runCFL
        self.dtLast=None
        self.dtRatioMax = 2.0
        self.cfl = {}
        for ci in range(self.nc):
            if ('cfl',ci) in transport.q:
                self.cfl[ci] = transport.q[('cfl',ci)]
        self.isAdaptive=True
    def choose_dt(self):
        maxCFL = 1.0e-6
        for ci in range(self.nc):
            if ci in self.cfl:
                maxCFL=max(maxCFL,globalMax(self.cfl[ci].max()))
                #print "mac cfl component ci",maxCFL,ci
        self.dt = self.runCFL/maxCFL
        if self.dtLast is None:
            self.dtLast = self.dt
        if self.dt/self.dtLast  > self.dtRatioMax:
            self.dt = self.dtLast*self.dtRatioMax
        self.t = self.tLast + self.dt
    def initialize_dt(self,t0,tOut,q):
        """
        Modify self.dt
        """
        self.tLast=t0
        self.choose_dt()
        self.t = t0+self.dt
    def updateTimeHistory(self,resetFromDOF=False):
        for ci in self.massComponents:
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat
        self.dtLast = self.dt
        self.tLast = self.t
        if self.integrateInterpolationPoints:
            for ci in list(self.m_last.keys()):
                self.m_ip_last[ci].flat[:] = self.m_ip_tmp[ci].flat
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL
    def postAdaptUpdate(self,oldTime):
        """This looks exactly like updateTimeHistory with the exception of setting self.tLast=self.t"""
        for ci in self.massComponents:
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat

        #unclear why this is needed; a change of machine precision in these values yield larger deviations in the gauges
        self.dtLast = oldTime.dtLast


class SSP(BackwardEuler_cfl):
    def __init__(self,transport,runCFL=0.9,integrateInterpolationPoints=False):
        BackwardEuler.__init__(self,transport,integrateInterpolationPoints=integrateInterpolationPoints)
        self.runCFL=runCFL
        self.dtLast=None
        self.dtRatioMax = 2.0
        self.cfl = {}
        for ci in range(self.nc):
            if ('cfl',ci) in transport.q:
                self.cfl[ci] = transport.q[('cfl',ci)]
        self.isAdaptive=True
        self.isSSP=True

class FLCBDF(TI_base):
    def __init__(self,transport):
        try:
            from . import flcbdfWrappers
        except ImportError:
            #experimental build: experimental/flcbdf/setup.py, on PYTHONPATH
            import flcbdfWrappers
        TI_base.__init__(self,transport)
        self.initCalls=0
        #masses
        self.flcbdf = dict([(ci,flcbdfWrappers.FLCBDF_integrator(transport.q[('m',ci)],"{0:s}m{1:d}".format(transport.name,ci))) for ci in self.massComponents])
        #dof cek could turn this off to save storage and/or look at robustness of predictor
        for ci in range(self.nc):
            self.flcbdf[('u',ci)] = flcbdfWrappers.FLCBDF_integrator(transport.u[ci].dof,transport.u[ci].name)
        self.Ddof_Dt= dict([(ci,numpy.zeros(self.transport.u[ci].dof.shape,'d'))
                            for ci in range(self.nc)])
        #cek todo get rid of dummy
        self.dummy= dict([(ci,numpy.ones(self.transport.u[ci].dof.shape,'d'))
                            for ci in range(self.nc)])
        self.calls=0
        self.m={}
        self.m_tmp={}
        self.beta_bdf={}
        self.alpha_bdf=0.0
        for ci in self.massComponents:
            self.m_tmp[ci] = transport.q[('m',ci)]
            self.m[ci]=transport.q[('m',ci)]
            self.beta_bdf[ci] = transport.q[('m',ci)].copy()
            self.beta_bdf[ci][:]=0.0
        self.beta_bdf_dummy = self.beta_bdf[ci].copy()#cek hack assumes all components have same quadrature
        self.beta_bdf_dummy[:]=0.0
        self.beta_bdf_dummy2 = self.beta_bdf[ci].copy()#cek hack assumes all components have same quadrature
        self.beta_bdf_dummy2[:]=0.0
        self.isAdaptive=True
        self.provides_initialGuess = True
    def initialize_dt(self,t0,tOut,q):
        self.tLast = t0
        self.dt = min([self.flcbdf[ci].initialize_dt(t0,tOut,q[('m',ci)],q[('mt',ci)]) for ci in self.massComponents])
        logEvent("Warning!!! FLDBDF initial dt will be different in parallel until we fix estimate_mt!")
        #mwf hack
        #import Comm
        #comm = Comm.get()
        #print "rank= %s FLCBDF chose dt= %s |mt,0|_max= %s |mt,1|_max= %s  forcing to (tOut-t)/1000 = %s " % (comm.rank(),self.dt,
        #                                                                                                      numpy.absolute(q[('mt',0)].flat).max(),
        #                                                                                                      numpy.absolute(q[('mt',1)].flat).max(),
        #                                                                                                      (tOut-t0)*1.0e-3)
        #self.dt = (tOut-t0)*1.0e-3
        for ci in range(self.nc): self.flcbdf[('u',ci)].initialize_dt(t0,tOut,self.transport.u[ci].dof,self.Ddof_Dt[ci])
        self.t = self.tLast + self.dt
    def initializeTimeHistory(self,resetFromDOF=False):
        logEvent("Initializing FLCBDF time history")
        self.initCalls +=1
        for ci in self.massComponents:
            self.flcbdf[ci].initializeTimeHistory(self.transport.q[('m',ci)],self.transport.q[('mt',ci)])
        for ci in range(self.nc): self.flcbdf[('u',ci)].initializeTimeHistory(self.transport.u[ci].dof,self.Ddof_Dt[ci])
    def calculateCoefs(self):
        logEvent("Calculating alpha_bdf and beta_bdf for FLCBDF")
        for ci in self.massComponents:
            self.alpha_bdf = self.flcbdf[ci].getCurrentAlpha()
            self.flcbdf[ci].calculate_yprime(self.beta_bdf_dummy,self.beta_bdf_dummy,self.beta_bdf[ci],self.beta_bdf_dummy2)
    def calculateElementCoefficients(self,q):
        for ci in self.massComponents:
            #! \todo fix dm,dmt calculation for non-diagonal nonlinearity
            #mwf hack what to do if no diagonal dependence at all?
            if ('dm',ci,ci) in q:
                self.flcbdf[ci].calculate_yprime(q[('m',ci)],q[('dm',ci,ci)],q[('mt',ci)],q[('dmt',ci,ci)])

            else:
                if ('dm_q',ci,ci) not in self.dummy:
                    self.dummy[('dm_q',ci,ci)] = numpy.zeros(q[('m',ci)].shape,'d')
                    self.dummy[('dmt_q',ci,ci)]    = numpy.zeros(q[('m',ci)].shape,'d')
                self.flcbdf[ci].calculate_yprime(q[('m',ci)],self.dummy[('dm_q',ci,ci)],q[('mt',ci)],self.dummy[('dmt_q',ci,ci)])
            #mwf duplicate for subgrid error
            for cj in range(self.nc):
                #mwf what happens if off diagonal dmt calculation?
                if ('dmt',ci,cj) in q:
                    #mwf debug
                    #import pdb
                    #pdb.set_trace()
                    alpha = self.flcbdf[ci].getCurrentAlpha()
                    self.alpha_bdf = alpha
                    q[('dmt',ci,cj)].flat[:] = q[('dm',ci,cj)].flat
                    q[('dmt',ci,cj)] *= alpha
                    #mwf debug
                    logEvent("FLCBDF calculateElementCoefficients t= %s dt= %s alpha= %s " % (self.t,self.dt,alpha))
                if ('dmt_sge',ci,cj) in q:
                    #mwf debug
                    #import pdb
                    #pdb.set_trace()
                    alpha = self.flcbdf[ci].getCurrentAlpha()
                    self.alpha_bdf = alpha
                    q[('dmt_sge',ci,cj)].flat[:] = q[('dm_sge',ci,cj)].flat
                    q[('dmt_sge',ci,cj)] *= alpha


#                 mwfHackedAyprime = False
#                 for cj in range(self.nc):
#                     try:
#                         self.flcbdf[ci].calculate_yprime(q[('m',ci)],q[('dm',ci,cj)],q[('mt',ci)],q[('dmt',ci,cj)])
#                         mwfHackedAyprime=True
#                         break
#                     except:
#                         pass
#                 assert mwfHackedAyprime, "couldn't find a key for dm,%s " % ci
            if numpy.isnan(q[('mt',ci)]).any():
                import pdb
                pdb.set_trace()
        for ci in range(self.nc):
            self.flcbdf[('u',ci)].calculate_yprime(self.transport.u[ci].dof,self.dummy[ci],self.Ddof_Dt[ci],self.dummy[ci])
    def choose_dt(self):
        logEvent("choosing dt for FLCBDF")
        self.dt = min([self.flcbdf[ci].choose_dt(self.tLast,self.tLast+100.0*self.dt) for ci in self.massComponents])
        for ci in range(self.nc): self.flcbdf[('u',ci)].choose_dt(self.tLast,self.tLast+100.0*self.dt)
        self.t = self.tLast + self.dt
    def set_dt(self,DTSET):
        logEvent("setting dt for FLCBDF")
        self.dt = DTSET
        self.t = self.tLast + self.dt
        for ci in self.massComponents: self.flcbdf[ci].set_dt(DTSET)
        for ci in range(self.nc): self.flcbdf[('u',ci)].set_dt(DTSET)
    def updateTimeHistory(self,resetFromDOF=False):
        logEvent("updating time history for FLCBDF")
        self.tLast = self.t
        #models that follow this one in the step (redistancing, mass correction) may have changed the
        #solution since the controller checked its error; the history has to start the next step from what
        #they left, as backward Euler's m_last does (without this the level set drifts from backward Euler
        #by O(t) at order 1 and fails Newton). Mass correction refreshes q[('m_tmp',ci)].
        mfinal = dict((ci,self.transport.q.get(('m_tmp',ci),self.m[ci])) for ci in self.massComponents)
        #the level set is stored at the quadrature points as u itself (m_tmp is q u), but redistancing
        #changes only its dofs (postStep copies them): bring the quadrature values up to the dofs, or the
        #history differs from where the next step starts by the redistancing jump, whatever the step size
        for ci in self.massComponents:
            q = self.transport.q
            if mfinal[ci] is q.get(('u',ci)):
                fs = self.transport.u[ci].femSpace
                psi = fs.getBasisValuesRef(self.transport.elementQuadraturePoints)
                q[('u',ci)][:] = numpy.dot(self.transport.u[ci].dof[fs.dofMap.l2g],psi.T)
        #the error check of the step was made by the controller on the solution as the step left it. A
        #projection moves only the level of the history (the integrator keeps the step's derivative), so a
        #change that does not shrink with the step does not appear as a derivative and an error floor. That
        #is right where the projected state is one the dofs can represent: the level set. The volume of
        #fluid's mass correction overwrites its quadrature values with the Heaviside of the corrected level
        #set, which its finite element space cannot represent; shifting to that value leaves an error of the
        #representation, independent of the step, so its history is left as the integrator took it.
        for ci in self.massComponents:
            if mfinal[ci] is self.transport.q.get(('u',ci)): self.flcbdf[ci].shiftLevel(mfinal[ci])
        for ci in range(self.nc): self.flcbdf[('u',ci)].shiftLevel(self.transport.u[ci].dof)
        for ci in self.massComponents: self.flcbdf[ci].stepTaken(mfinal[ci])
        for ci in range(self.nc): self.flcbdf[('u',ci)].stepTaken(self.transport.u[ci].dof)
    def _flcbdf_key(self,key):
        return "m%d" % key if isinstance(key,int) else "u%d" % key[1]
    def state(self):
        """What a hot start needs to continue with the same history, order and
        step size: each integrator's full state, and the current time and dt."""
        st = {"tLast":float(self.tLast),"dt":float(self.dt)}
        for key,f in self.flcbdf.items():
            st["flcbdf_"+self._flcbdf_key(key)] = f.state()
        return st
    def set_state(self,st):
        self.tLast = st["tLast"]
        self.dt = st["dt"]
        self.t = self.tLast + self.dt
        for key,f in self.flcbdf.items():
            f.set_state(numpy.ascontiguousarray(st["flcbdf_"+self._flcbdf_key(key)],dtype='d'))
    def updateStage(self):
        """
        increment stage counters and push necessary information
        into stage arrays
        """
        pass
    def setInitialStageValues(self):
        """
        set the all stage values assuming this is the first step
        after a problem reset
        """
        pass
    def lastStepErrorOk(self):
        logEvent("checking error for FLCBDF")
        OK=True
        for ci in self.massComponents:
            OK = (OK and bool(self.flcbdf[ci].lastStepErrorOk(self.m[ci])))
        for ci in range(self.nc): self.flcbdf[('u',ci)].lastStepErrorOk(self.transport.u[ci].dof)
        return OK

class PsiTCtte(BackwardEuler_cfl):
    from .ctimeIntegration import psiTCtteDT
    def __init__(self,transport):
        runCFL=0.1
        tau=1.0e-2
        BackwardEuler_cfl.__init__(self,transport)
        self.tau = tau
        self.mt_history={}
        self.mt_tmp = {}
        nhist = 2
        self.nsteps = 0
        self.dt_history = numpy.ones(nhist,'d')
        self.dt_history.fill(-12345.0)
        self.dt_tmp = -12345.0
        for ci in self.massComponents:
            self.mt_tmp[ci] = numpy.array(transport.q[('mt',ci)])
        for it in range(nhist):
            self.mt_history[it] = {}
            for ci in self.massComponents:
                self.mt_history[it][ci] = numpy.array(transport.q[('mt',ci)])
        #mwf for keeping track of calls to update history calls
        #cek todo should get rid of this flag if no longer needed
        self.updateHistoryJustCalled = False
        self.isAdaptive=False
    def initializeTimeHistory(self,resetFromDOF=True):
        self.updateTimeHistory(resetFromDOF)
    def calculateElementCoefficients(self,q):
        BackwardEuler_cfl.calculateElementCoefficients(self,q)
        for ci in list(self.m_last.keys()):
            self.mt_tmp[ci].flat[:] = q[('mt',ci)].flat[:]
        self.dt_tmp = self.dt
        #now tmp values are for new time step
        self.updateHistoryJustCalled = False
    def choose_dt(self):
        """
        Modify self.dt
        """
        #I think update history is getting called more than once
        if self.dt_history[0] <= 0.0 or self.dt_history[1] < 0.0 or self.nsteps < 2:
            BackwardEuler_cfl.choose_dt(self)
            return
        dtmax = 0.0
        for ci in self.massComponents:
            dtci = self.psiTCtteDT(self.tau,self.dt_history[0],self.dt_history[1],
                                   self.m_last[ci],
                                   self.mt_history[0][ci],self.mt_history[1][ci])
            dtmax = max(dtmax,dtci)
        self.dt = dtmax
        self.t = self.tLast + self.dt
    def updateTimeHistory(self,resetFromDOF=False):
        self.tLast = self.t
        #mwf to check how many times update history called
        if self.updateHistoryJustCalled:
            logEvent("""WARNING PsiTCtte in updateHistory historyJustCalled is True
dt_history[0]=%g dt_history[1]=%g nsteps=%d """ %(self.dt_history[0],self.dt_history[1],
                                                  self.nsteps))

        BackwardEuler_cfl.updateTimeHistory(self,resetFromDOF=False)
        self.dt_history[1] = self.dt_history[0]
        self.dt_history[0] = self.dt_tmp
        self.nsteps += 1
        for ci in list(self.m_last.keys()):
            self.mt_history[1][ci].flat[:]= self.mt_history[0][ci].flat[:]
            self.mt_history[0][ci].flat[:]= self.mt_tmp[ci].flat[:]
        #end ci
        self.updateHistoryJustCalled = True
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL

class PsiTCtte_new(BackwardEuler):
    from .ctimeIntegration import psiTCtteDT
    def __init__(self,transport):
        runCFL=0.9
        tau=1.0e-2
        BackwardEuler.__init__(self,transport)
        self.tau = tau
        self.mt_history={}
        self.mt_tmp = {}
        nhist = 2
        self.nsteps = 0
        self.dt_history = numpy.ones(nhist,'d')
        self.dt_history.fill(-12345.0)
        self.dt_tmp = -12345.0
        for ci in self.massComponents:
            self.mt_tmp[ci] = numpy.array(transport.q[('mt',ci)])
        for it in range(nhist):
            self.mt_history[it] = {}
            for ci in self.massComponents:
                self.mt_history[it][ci] = numpy.array(transport.q[('mt',ci)])
    def calculateElementCoefficients(self,q):
        BackwardEuler.calculateElementCoefficients(self,q)
        for ci in list(self.m_last.keys()):
            self.mt_tmp[ci].flat[:] = q[('mt',ci)].flat[:]
        self.dt_tmp = self.dt
    def choose_dt(self):
        #choose new dt
        if self.nsteps < 2:
            self.dt_history[1] = self.dt_history[0]
            self.dt_history[0] = self.dt
            self.nsteps+=1
            logEvent("PsiTC (first 2 steps) choose_dt dt=%f" % self.dt)
            return
        dtmax = 0.0
        for ci in self.massComponents:
            dtci = self.psiTCtteDT(self.tau,self.dt_history[0],self.dt_history[1],
                                   self.m_last[ci],
                                   self.mt_history[0][ci],self.mt_history[1][ci])
            #print "dtci dtmax", dtci, dtmax
            dtmax = max(dtmax,dtci)
        self.dt = dtmax
        self.dt = min(self.dt,10.0*self.dt_history[0])
        self.dt_history[1] = self.dt_history[0]
        self.dt_history[0] = self.dt_tmp
        self.nsteps += 1
        for ci in self.massComponents:
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat
            self.mt_history[1][ci].flat[:]= self.mt_history[0][ci].flat[:]
            self.mt_history[0][ci].flat[:]= self.mt_tmp[ci].flat[:]
        logEvent("PsiTC choose_dt dt=%f" % self.dt)
    def resetTimeHistory(self,resetFromDOF=False):
        for ci in self.massComponents:
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat
    def updateTimeHistory(self,resetFromDOF=False):
        self.nsteps=0
        self.tLast = self.t
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL
        if 'PsiTCtte_tau' in dir(nOptions):
            tau=nOptions.PsiTCtte_tau

class ForwardEuler(TI_base):
    def __init__(self,transport,runCFL=0.45):
        TI_base.__init__(self,transport)
        self.m_last={}
        self.m_tmp={}
        self.r_last={}
        self.r_tmp={}
        if transport.stabilization is not None:
            self.strong_r_last={}
            self.strong_r_tmp={}
        self.runCFL=runCFL
        self.dtLast=None
        self.dtRatioMax = 2.0 #come up with better value
        for ci in self.massComponents:
            if ('m',ci) in transport.q:
                self.m_last[ci] = numpy.array(transport.q[('m',ci)])
                self.m_tmp[ci] = numpy.array(transport.q[('m',ci)])
                self.r_last[ci] = numpy.array(transport.elementResidual[ci])
                self.r_tmp[ci] = numpy.array(transport.elementResidual[ci])
                if self.transport.stabilization is not None:
                    self.strong_r_last[ci] = numpy.array(transport.q[('pdeResidual',ci)])
                    self.strong_r_tmp[ci] = numpy.array(transport.q[('pdeResidual',ci)])
                self.advectionIsImplicit[ci]      = False
                self.diffusionIsImplicit[ci]      = False
                self.reactionIsImplicit[ci]       = False
                self.hamiltonianIsImplicit[ci]    = False
                self.stabilizationIsImplicit[ci]  = False
                self.shockCapturingIsImplicit[ci] = False
        self.cfl = {}
        for ci in range(self.nc):
            if ('cfl',ci) in transport.q:
                self.cfl[ci] = transport.q[('cfl',ci)]
        self.isAdaptive=True
    def calculateElementSpatialJacobian(self,elementJacobian):
        for ci in list(elementJacobian.keys()):
            for cj in list(elementJacobian[ci].keys()):
                elementJacobian[ci][cj].fill(0.0)
    def calculateElementBoundaryCoefficients(self,ebq):
        """
        Calculate m_t, and dm_t and recalculate any of the other coefficients on element boundaries
        do I need these to be synchronized with q or do the elementBoundary quad values get
        lagged in the spatial residual
        """
        pass
    def calculateElementSpatialBoundaryJacobian(self,elementBoundaryJacobian,elementBoundaryJacobian_eb):
        for ci in list(elementBoundaryJacobian.keys()):
            for cj in list(elementBoundaryJacobian[ci].keys()):
                elementBoundaryJacobian[ci][cj].fill(0.0)
        for ci in list(elementBoundaryJacobian_eb.keys()):
            for cj in list(elementBoundaryJacobian_eb[ci].keys()):
                elementBoundaryJacobian_eb[ci][cj].fill(0.0)
    def calculateExteriorElementSpatialBoundaryJacobian(self,elementBoundaryJacobian):
        for ci in list(elementBoundaryJacobian.keys()):
            for cj in list(elementBoundaryJacobian[ci].keys()):
                elementBoundaryJacobian[ci][cj].fill(0.0)
    def calculateElementCoefficients(self,q):
        for ci in self.massComponents:
            if ('m',ci) in q:
                self.m_tmp[ci] = q[('m',ci)]
                q[('mt',ci)][:] =q[('m',ci)]
                q[('mt',ci)] -= self.m_last[ci]
                q[('mt',ci)] /= self.dt
                for cj in range(self.nc):
                    if ('dm',ci,cj) in q:
                        q[('dmt',ci,cj)][:] = q[('dm',ci,cj)]
                        q[('dmt',ci,cj)] /= self.dt
    def calculateStrongElementSpatialResidual(self,q):
        if self.transport.stabilization is not None:
            for ci in list(self.r_tmp.keys()):
                self.strong_r_tmp[ci][:]=q[('pdeResidual',ci)]
                q[('pdeResidual',ci)][:]=self.strong_r_last[ci]
                for cj in list(self.r_tmp.keys()):
                    if ('dpdeResidual',ci,cj) in q:
                        q[('dpdeResidual',ci,cj)].fill(0.0)
    def calculateElementSpatialResidual(self,elementResidual):
        for ci in list(self.r_tmp.keys()):
            self.r_tmp[ci][:]=elementResidual[ci]
            elementResidual[ci][:]=self.r_last[ci]
    def choose_dt(self):
        maxCFL=1.0e-6
        for ci in range(self.nc):
            if ci in self.cfl:
                maxCFL=max(maxCFL,globalMax(self.cfl[ci].max()))
        self.dt = self.runCFL/maxCFL
        if self.dtLast is None:
            self.dtLast = self.dt
        if self.dt/self.dtLast  > self.dtRatioMax:
            self.dt = self.dtLast*self.dtRatioMax
        self.t = self.tLast + self.dt
    def initialize_dt(self,t0,tOut,q):
        """
        Modify self.dt
        """
        self.tLast=t0
        self.choose_dt()
        self.t = t0+self.dt
    def initializeSpaceHistory(self,resetFromDOF=False):
        self.tLast = self.t
        self.dtLast = self.dt
        for ci in list(self.m_last.keys()):
            self.r_last[ci].flat[:] = self.r_tmp[ci].flat
            if self.transport.stabilization is not None:
                self.strong_r_last[ci].flat[:] = self.strong_r_tmp[ci].flat
    def updateTimeHistory(self,resetFromDOF=False):
        self.tLast = self.t
        self.dtLast = self.dt
        for ci in list(self.m_last.keys()):
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat
            self.r_last[ci].flat[:] = self.r_tmp[ci].flat
            if self.transport.stabilization is not None:
                self.strong_r_last[ci].flat[:] = self.strong_r_tmp[ci].flat
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL
#cek todo rename this. It's forward euler in advection only so it's a special IMEX method
class ForwardEuler_A(TI_base):
    def __init__(self,transport,runCFL=0.45,limiterType=None):
        TI_base.__init__(self,transport)
        self.m_last = {}
        self.m_tmp = {}
        self.f_last ={}
        self.f_tmp = {}
        self.bf_last = {}
        self.bf_tmp = {}
        self.ebf_last = {}
        self.ebf_tmp  = {}
        #Necessary for Rusanov flux at least
        #Problem is how to allow for implicit u in diffusion
        self.ebq_u_last = {} ; self.ebq_u_tmp = {}
        self.ebqe_u_last = {} ; self.ebqe_u_tmp = {}
        self.q_u_last = {} ; self.q_u_tmp = {}
        self.q_cfl_last= {} ; self.q_cfl_tmp = {}
        self.q_df_tmp = {} ;    self.q_df_last= {}
        self.ebq_df_tmp= {} ;  self.ebq_df_last={}
        self.ebqe_df_tmp= {} ;  self.ebqe_df_last={}

        #
        self.limiter=None; self.uLimDof = None
        self.limiterType = limiterType
        if self.limiterType is not None:
            self.limiter = limiterType(transport.mesh,transport.nSpace_global,
                                       transport.u,
                                       transport=transport)
            self.uLimDof = {}
            for ci in range(self.nc):
                self.uLimDof[ci]=numpy.zeros((transport.u[ci].dof.shape),'d')

        for ci in self.massComponents:
            self.m_last[ci] = numpy.array(transport.q[('m',ci)])
            self.m_tmp[ci] = numpy.array(transport.q[('m',ci)])

            if ci in transport.coefficients.advection:
                self.f_last[ci] = numpy.array(transport.q[('f',ci)])
                self.f_tmp[ci] = numpy.array(transport.q[('f',ci)])
                if ('f',ci) in transport.ebq:
                    self.bf_last[ci] = numpy.array(transport.ebq[('f',ci)])
                    self.bf_tmp[ci] = numpy.array(transport.ebq[('f',ci)])
                self.ebf_last[ci] = numpy.array(transport.ebqe[('f',ci)])
                self.ebf_tmp[ci] = numpy.array(transport.ebqe[('f',ci)])
                #u
                if ('u',ci) in transport.ebq:
                    self.ebq_u_last[ci] = numpy.array(transport.ebq[('u',ci)])
                    self.ebq_u_tmp[ci] = numpy.array(transport.ebq[('u',ci)])
                if ('u',ci) in transport.ebqe:
                    self.ebqe_u_last[ci] = numpy.array(transport.ebqe[('u',ci)])
                    self.ebqe_u_tmp[ci] = numpy.array(transport.ebqe[('u',ci)])
                if ('u',ci) in transport.q:
                    self.q_u_last[ci] = numpy.array(transport.q[('u',ci)])
                    self.q_u_tmp[ci] = numpy.array(transport.q[('u',ci)])
                if ('cfl',ci) in transport.q:
                    self.q_cfl_last[ci] = numpy.array(transport.q[('cfl',ci)])
                    self.q_cfl_tmp[ci] = numpy.array(transport.q[('cfl',ci)])

                #df
                for cj in range(self.nc):
                    if ('df',ci,cj) in transport.q:
                        self.q_df_last[(ci,cj)] = numpy.array(transport.q[('df',ci,cj)])
                        self.q_df_tmp[(ci,cj)]  = numpy.array(transport.q[('df',ci,cj)])
                    if ('df',ci,cj) in transport.ebq:
                        self.ebq_df_last[(ci,cj)] = numpy.array(transport.ebq[('df',ci,cj)])
                        self.ebq_df_tmp[(ci,cj)]  = numpy.array(transport.ebq[('df',ci,cj)])
                    if ('df',ci,cj) in transport.ebqe:
                        self.ebqe_df_last[(ci,cj)] = numpy.array(transport.ebqe[('df',ci,cj)])
                        self.ebqe_df_tmp[(ci,cj)]  = numpy.array(transport.ebqe[('df',ci,cj)])

            self.advectionIsImplicit[ci] = False
        self.runCFL=runCFL
        self.dtLast=None
        self.dtRatioMax = 2.0 #come up with better value
        self.cfl = {}
        for ci in range(self.nc):
            if ('cfl',ci) in transport.q:
                self.cfl[ci] = transport.q[('cfl',ci)]
        self.isAdaptive=True
    def calculateElementCoefficients(self,q):
        for ci in self.massComponents:
            self.m_tmp[ci][:] = q[('m',ci)]
            q[('mt',ci)][:] =q[('m',ci)]
            q[('mt',ci)] -= self.m_last[ci]
            q[('mt',ci)] /= self.dt
            q[('dmt',ci,ci)][:] = q[('dm',ci,ci)]
            q[('dmt',ci,ci)] /= self.dt
        for ci in list(self.f_tmp.keys()):
            self.f_tmp[ci][:] = q[('f',ci)]
            q[('f',ci)][:] = self.f_last[ci]
            #for numerical fluxes like Rusanov that need some element information
            for cj in range(self.nc):
                if ('df_advectiveNumericalFlux',ci,cj) in q:
                    self.q_df_tmp[(ci,cj)][:] = q[('df',ci,cj)]
                    q[('df_advectiveNumericalFlux',ci,cj)][:] = self.q_df_last[(ci,cj)]
                    #zero for jacobian? will get skipped because of advectionIsImplicit?
                    #q[('df',ci,cj)].fill(0.0)
        for ci in list(self.q_u_tmp.keys()):
            self.q_u_tmp[ci][:] = q[('u',ci)]
            if ('u_advectiveNumericalFlux',ci) in q:
                q[('u_advectiveNumericalFlux',ci)][:] = self.q_u_last[ci]
        for ci in list(self.q_cfl_tmp.keys()):
            self.q_cfl_tmp[ci][:] = q[('cfl',ci)]
            if ('cfl_advectiveNumericalFlux',ci) in q:
                q[('cfl_advectiveNumericalFlux',ci)][:] = self.q_cfl_last[ci]

    def calculateElementBoundaryCoefficients(self,ebq):
        for ci in list(self.bf_tmp.keys()):
            self.bf_tmp[ci][:] = ebq[('f',ci)]
            ebq[('f',ci)][:] = self.bf_last[ci]
            #
            if ('u_advectiveNumericalFlux',ci) in ebq:
                self.ebq_u_tmp[ci][:] = ebq[('u',ci)]
                ebq[('u_advectiveNumericalFlux',ci)][:] = self.ebq_u_last[ci]
            for cj in range(self.nc):
                if ('df_advectiveNumericalFlux',ci,cj) in ebq:
                    self.ebq_df_tmp[(ci,cj)][:] = ebq[('df',ci,cj)]
                    ebq[('df_advectiveNumericalFlux',ci,cj)][:] = self.ebq_df_last[(ci,cj)]
    def calculateExteriorElementBoundaryCoefficients(self,ebqe):
        for ci in list(self.ebf_tmp.keys()):
            self.ebf_tmp[ci][:] = ebqe[('f',ci)]
            ebqe[('f',ci)][:] = self.ebf_last[ci]
            #
            if ('u_advectiveNumericalFlux',ci) in ebqe:
                self.ebqe_u_tmp[ci][:] = ebqe[('u',ci)]
                ebqe[('u_advectiveNumericalFlux',ci)][:] = self.ebqe_u_last[ci]
            for cj in range(self.nc):
                if ('df_advectiveNumericalFlux',ci,cj) in ebqe:
                    self.ebqe_df_tmp[(ci,cj)][:] = ebqe[('df',ci,cj)]
                    ebqe[('df_advectiveNumericalFlux',ci,cj)][:] = self.ebqe_df_last[(ci,cj)]
    def initialize_dt(self,t0,tOut,q):
        """
        Modify self.dt
        """
        self.tLast=t0
        self.choose_dt()
        self.t = t0+self.dt
    def choose_dt(self):
        """
        Modify self.dt
        mwf needs to be checked
        """
        maxCFL=1.0e-6
        for ci in range(self.nc):
            if ci in self.cfl:
                maxCFL=max(maxCFL,globalMax(self.cfl[ci].max()))
        self.dt = self.runCFL/maxCFL
        if self.dtLast is None:
            self.dtLast = self.dt
        if self.dt/self.dtLast  > self.dtRatioMax:
            self.dt = self.dtLast*self.dtRatioMax
        self.t = self.tLast + self.dt
    def updateStage(self):
        """
        if using limiting
        """
        #mwf debug
        #import pdb
        #pdb.set_trace()
        #mwf hack
        #grad_phi = numpy.array(self.transport.q[('grad(phi)',0)])
        if self.limiterType is not None:
            self.limiter.applySlopeLimiting(self.transport.u,
                                            self.uLimDof)
            for ci in range(self.nc):
                self.transport.u[ci].dof.flat[:] = self.uLimDof[ci].flat[:]
                assert self.transport.u[ci].par_dof is not None
                self.transport.u[ci].par_dof.scatter_forward_insert()
        #if

        #update coefficients how?
        self.transport.calculateCoefficients()
        #mwf hack
        #self.transport.q[('grad(phi)',0)][:]= grad_phi
        #time integration loads in new residual in this call
        #self.transport.calculateElementResidual()

    def updateTimeHistory(self,resetFromDOF=False):
        self.tLast = self.t
        self.dtLast = self.dt
        for ci in self.massComponents:
            self.m_last[ci][:] = self.m_tmp[ci]
        for ci in list(self.f_last.keys()):
            self.f_last[ci][:] = self.f_tmp[ci]
            self.ebf_last[ci][:] = self.ebf_tmp[ci]
        for ci in list(self.bf_last.keys()):
            self.bf_last[ci][:] = self.bf_tmp[ci]
        #
        for ci in list(self.ebq_u_last.keys()):
            self.ebq_u_last[ci][:] = self.ebq_u_tmp[ci]
        for ci in list(self.ebqe_u_last.keys()):
            self.ebqe_u_last[ci][:] = self.ebqe_u_tmp[ci]
        for k in list(self.q_df_last.keys()):
            self.q_df_last[k][:] = self.q_df_tmp[k]
        for k in list(self.q_df_last.keys()):
            self.q_df_last[k][:] = self.q_df_tmp[k]
        for k in list(self.ebq_df_last.keys()):
            self.ebq_df_last[k][:] = self.ebq_df_tmp[k]
        for k in list(self.ebqe_df_last.keys()):
            self.ebqe_df_last[k][:] = self.ebqe_df_tmp[k]
        #
        for ci in list(self.q_u_last.keys()):
            self.q_u_last[ci][:] = self.q_u_tmp[ci]
        for ci in list(self.q_cfl_last.keys()):
            self.q_cfl_last[ci][:] = self.q_cfl_tmp[ci]
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL

        if 'limiterType' in dir(nOptions) and nOptions.limiterType is not None:
            self.limiter = None; self.limiterType = None; self.uLimDof = {}
            self.limiterType = nOptions.limiterType
            self.limiter = self.limiterType(self.transport.mesh,
                                            self.transport.nSpace_global,
                                            self.transport.u,
                                            transport=self.transport)

            for ci in range(self.nc):
                #for limiting, in values
                self.uLimDof[ci] = numpy.zeros((self.transport.u[ci].dof.shape),'d')

            #
            self.limiter.setFromOptions(nOptions)
        #
    def initializeSpaceHistory(self,resetFromDOF=True):
        self.tLast = self.t
        self.dtLast = self.dt
        for ci in list(self.f_last.keys()):
            self.f_last[ci][:] = self.f_tmp[ci]
            self.ebf_last[ci][:] = self.ebf_tmp[ci]
        for ci in list(self.bf_last.keys()):
            self.bf_last[ci][:] = self.bf_tmp[ci]
        #
        for ci in list(self.ebq_u_last.keys()):
            self.ebq_u_last[ci][:] = self.ebq_u_tmp[ci]
        for ci in list(self.ebqe_u_last.keys()):
            self.ebqe_u_last[ci][:] = self.ebqe_u_tmp[ci]
        for k in list(self.q_df_last.keys()):
            self.q_df_last[k][:] = self.q_df_tmp[k]
        for k in list(self.q_df_last.keys()):
            self.q_df_last[k][:] = self.q_df_tmp[k]
        for k in list(self.ebq_df_last.keys()):
            self.ebq_df_last[k][:] = self.ebq_df_tmp[k]
        for k in list(self.ebqe_df_last.keys()):
            self.ebqe_df_last[k][:] = self.ebqe_df_tmp[k]
        #
        for ci in list(self.q_u_last.keys()):
            self.q_u_last[ci][:] = self.q_u_tmp[ci]
        for ci in list(self.q_cfl_last.keys()):
            self.q_cfl_last[ci][:] = self.q_cfl_tmp[ci]
    def initializeTimeHistory(self,resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        for ci in self.massComponents:
            self.m_last[ci][:] = self.m_tmp[ci]
        for ci in list(self.f_last.keys()):
            self.f_last[ci][:] = self.f_tmp[ci]
            self.ebf_last[ci][:] = self.ebf_tmp[ci]
        for ci in list(self.bf_last.keys()):
            self.bf_last[ci][:] = self.bf_tmp[ci]
        #
        for ci in list(self.ebq_u_last.keys()):
            self.ebq_u_last[ci][:] = self.ebq_u_tmp[ci]
        for ci in list(self.ebqe_u_last.keys()):
            self.ebqe_u_last[ci][:] = self.ebqe_u_tmp[ci]
        for k in list(self.q_df_last.keys()):
            self.q_df_last[k][:] = self.q_df_tmp[k]
        for k in list(self.q_df_last.keys()):
            self.q_df_last[k][:] = self.q_df_tmp[k]
        for k in list(self.ebq_df_last.keys()):
            self.ebq_df_last[k][:] = self.ebq_df_tmp[k]
        for k in list(self.ebqe_df_last.keys()):
            self.ebqe_df_last[k][:] = self.ebqe_df_tmp[k]
        #
        for ci in list(self.q_u_last.keys()):
            self.q_u_last[ci][:] = self.q_u_tmp[ci]
        for ci in list(self.q_cfl_last.keys()):
            self.q_cfl_last[ci][:] = self.q_cfl_tmp[ci]


class ForwardEuler_H(TI_base):
    def __init__(self,transport,runCFL=0.45):
        TI_base.__init__(self,transport)
        self.m_last = numpy.array(transport.q[('m',0)])
        self.m_tmp = numpy.array(transport.q[('m',0)])
        self.H_last = numpy.array(transport.q[('H',0)])
        self.H_tmp = numpy.array(transport.q[('H',0)])
        self.dH_last = numpy.array(transport.q[('dH',0,0)])
        self.dH_tmp = numpy.array(transport.q[('dH',0,0)])
        if ('grad(u)',0) in transport.q:
            self.grad_u_last = numpy.array(transport.q[('grad(u)',0)])
            self.grad_u_tmp = numpy.array(transport.q[('grad(u)',0)])
        else:
            self.grad_u_last = None
            self.grad_u_tmp = None
        self.hamiltonianIsImplicit[0] = False
        self.runCFL=runCFL
        self.dtLast=None
        self.dtRatioMax = 2.0 #come up with better value
        #mwf allow initial calculation of cfl and stabilization quantities
        self.nCalls = 0
        self.cfl = {}
        for ci in range(self.nc):
            if ('cfl',ci) in transport.q:
                self.cfl[ci] = transport.q[('cfl',ci)]
            #end if
        #end for

    def calculateU(self,u):
        self.u = u
    def calculateElementCoefficients(self,q):
        self.m_tmp.flat[:] = q[('m',0)].flat
        q[('mt',0)].flat[:] =q[('m',0)].flat
        q[('mt',0)] -= self.m_last
        q[('mt',0)] /= self.dt
        q[('dmt',0,0)].flat[:] = q[('dm',0,0)].flat
        q[('dmt',0,0)] /= self.dt
        self.H_tmp.flat[:] = q[('H',0)].flat
        self.dH_tmp.flat[:] = q[('dH',0,0)].flat
        if self.nCalls > 1:
            q[('H',0)].flat[:] = self.H_last.flat
            q[('dH',0,0)].flat[:] = self.dH_last.flat
#         if q.has_key(('grad(u)',0)):
#             self.grad_u_tmp.flat[:]  = q[('grad(u)',0)].flat
#             if not self.nCalls > 1:
#                 q[('grad(u)',0)].flat[:] = self.grad_u_last.flat
        self.nCalls += 1
    def calculateElementBoundaryCoefficients(self,ebq):
        pass
    def calculateElementSpatialResidual(self,elementResidual):
        pass
    def choose_dt(self):
        """
        Modify self.dt
        mwf needs to be checked
        """
        maxCFL=1.0e-6
        for ci in range(self.nc):
            if ci in self.cfl:
                maxCFL=globalMax(self.cfl[ci].max())
            #end has key
        self.dt = self.runCFL/maxCFL
        if self.dtLast is None:
            self.dtLast = self.dt
        if self.dt/self.dtLast  > self.dtRatioMax:
            self.dt = self.dtLast*self.dtRatioMax
        self.t = self.tLast + self.dt
    def updateTimeHistory(self,resetFromDOF=False):
        self.tLast = self.t
        self.dtLast = self.dt
        self.m_last.flat[:] = self.m_tmp.flat
        self.H_last.flat[:] = self.H_tmp.flat
        self.dH_last.flat[:] = self.dH_tmp.flat
        if self.grad_u_last is not None:
            self.grad_u_last.flat[:] = self.grad_u_tmp.flat
    def initializeSpaceHistory(self,resetFromDOF=False):
        self.tLast = self.t
        self.dtLast = self.dt
        self.H_last.flat[:] = self.H_tmp.flat
        self.dH_last.flat[:] = self.dH_tmp.flat
        if self.grad_u_last is not None:
            self.grad_u_last.flat[:] = self.grad_u_tmp.flat
    def updateStage(self):
        """
        increment stage counters and push necessary information
        into stage arrays
        """
        pass
    def setInitialStageValues(self):
        """
        set the all stage values assuming this is the first step
        after a problem reset
        """
        pass
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL

class OuterTheta(TI_base):
    def __init__(self,transport,
                 runCFL=0.9,
                 thetaAdvection=0.5,
                 thetaDiffusion=0.5,
                 thetaReaction=0.5,
                 thetaHamiltonian=0.5):
        TI_base.__init__(self,transport)
        self.runCFL = 0.1
        self.thetaAdvection=thetaAdvection
        self.thetaDiffusion=thetaDiffusion
        self.thetaReaction=thetaReaction
        self.thetaHamiltonian=thetaHamiltonian
        self.cfl = {}
        for ci in range(self.nc):
            if ('cfl',ci) in transport.q:
                self.cfl[ci] = transport.q[('cfl',ci)]
        self.dt = 1.0
        self.dtLast = None
        self.dtRatioMax = 2.0 #come up with better value
        self.q_last={}
        self.q_tmp={}
        self.m_last={}
        for ci in self.massComponents:
            self.massIsImplicit[ci] = True
            self.q_last[('m',ci)] = numpy.array(transport.q[('m',ci)])
            self.q_tmp[('m',ci)] = numpy.array(transport.q[('m',ci)])
            self.m_last[ci] = self.q_last[('m',ci)]
        for ci in list(transport.coefficients.advection.keys()):
            self.advectionIsImplicit[ci] = True
            self.q_last[('f',ci)] = numpy.array(transport.q[('f',ci)])
            self.q_tmp[('f',ci)] = numpy.array(transport.q[('f',ci)])
        for ci in list(transport.coefficients.reaction.keys()):
            self.reactionIsImplicit[ci] = True
            self.q_last[('r',ci)] = numpy.array(transport.q[('r',ci)])
            self.q_tmp[('r',ci)] = numpy.array(transport.q[('r',ci)])
        for ci in list(transport.coefficients.hamiltonian.keys()):
            self.q_last[('H',ci)] = numpy.array(transport.q[('H',ci)])
            self.q_tmp[('H',ci)] = numpy.array(transport.q[('H',ci)])
        for ci,ckDict in transport.coefficients.diffusion.items():
            self.diffusionIsImplicit[ci] = True
            for ck in list(ckDict.keys()):
                self.q_last[('a',ci,ck)] = numpy.array(transport.q[('a',ci,ck)])
                self.q_tmp[('a',ci,ck)] = numpy.array(transport.q[('a',ci,ck)])
        self.ebq_last={}
        self.ebq_tmp={}
        for ci in list(transport.coefficients.advection.keys()):
            if ('f',ci) in transport.ebq:
                self.ebq_last[('f',ci)] = numpy.array(transport.ebq[('f',ci)])
                self.ebq_tmp[('f',ci)] = numpy.array(transport.ebq[('f',ci)])
        for ci,ckDict in transport.coefficients.diffusion.items():
            for ck in list(ckDict.keys()):
                if ('a',ci,ck) in transport.ebq:
                    self.ebq_last[('a',ci,ck)] = numpy.array(transport.ebq[('a',ci,ck)])
                    self.ebq_tmp[('a',ci,ck)] = numpy.array(transport.ebq[('a',ci,ck)])
    def calculateU(self,u):
        self.u = u
    def calculateElementCoefficients(self,q):
        for ci,cjDict in self.transport.coefficients.mass.items():
            self.q_tmp[('m',ci)][:] = q[('m',ci)]
            q[('mt',ci)][:] = q[('m',ci)]
            q[('mt',ci)] -= self.q_last[('m',ci)]
            q[('mt',ci)] /= self.dt
            for cj in list(cjDict.keys()):
                q[('dmt',ci,cj)][:] = q[('dm',ci,cj)]
                q[('dmt',ci,cj)] /= self.dt
        for ci,cjDict in self.transport.coefficients.advection.items():
            self.q_tmp[('f',ci)][:] = q[('f',ci)]
            q[('f',ci)] *= self.thetaAdvection
            q[('f',ci)] += self.q_last[('f',ci)]
            for  cj in list(cjDict.keys()):
                q[('df',ci,cj)] *= self.thetaAdvection
        for ci,cjDict in self.transport.coefficients.reaction.items():
            self.q_tmp[('r',ci)][:] = q[('r',ci)]
            q[('r',ci)] *= self.thetaReaction
            q[('r',ci)] += self.q_last[('r',ci)]
            for cj in list(cjDict.keys()):
                q[('dr',ci,cj)] *= self.thetaReaction
        for ci,cj in self.transport.coefficients.hamiltonian.items():
            self.q_tmp[('H',ci)][:] = q[('H',ci)]
            q[('H',ci)] *= self.thetaHamiltonian
            q[('H',ci)] += self.q_last[('H',ci)]
            for cj in list(cjDict.keys()):
                q[('dH',ci,cj)] *= self.thetaHamiltonian
        for ci,ckDict in self.transport.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                self.q_tmp[('a',ci,ck)][:] = q[('a',ci,ck)]
                q[('a',ci,ck)] *= self.thetaDiffusion
                q[('a',ci,ck)] += self.q_last[('a',ci,ck)]
                for cj in list(cjDict.keys()):
                    q[('da',ci,ck,cj)] *= self.thetaDiffusion
    def calculateElementBoundaryCoefficients(self,ebq):
        for ci,cjDict in self.transport.coefficients.advection.items():
            self.ebq_tmp[('f',ci)][:] = ebq[('f',ci)]
            ebq[('f',ci)] *= self.thetaAdvection
            ebq[('f',ci)] += self.ebq_last[('f',ci)]
            for  cj in list(cjDict.keys()):
                ebq[('df',ci,cj)] *= self.thetaAdvection
        for ci,ckDict in self.transport.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                self.ebq_tmp[('a',ci,ck)][:] = ebq[('a',ci,ck)]
                ebq[('a',ci,ck)] *= self.thetaDiffusion
                ebq[('a',ci,ck)] += self.ebq_last[('a',ci,ck)]
                for cj in list(cjDict.keys()):
                    ebq[('da',ci,ck,cj)] *= self.thetaDiffusion
    def choose_dt(self):
        """
        Modify self.dt
        mwf needs to be checked
        """
        maxCFL=1.0e-6
        for ci in range(self.nc):
            if ci in self.cfl:
                maxCFL=max(maxCFL,globalMax(self.cfl[ci].max()))
        self.dt = self.runCFL/maxCFL
        if self.dtLast is None:
            self.dtLast = self.dt
        if self.dt/self.dtLast  > self.dtRatioMax:
            self.dt = self.dtLast*self.dtRatioMax
        self.t = self.tLast + self.dt
    def updateTimeHistory(self,resetFromDOF=False):
        self.tLast = self.t
        self.dtLast = self.dt
        for ci in self.self.massComponents:
            self.q_last[('m',ci)][:] = self.q_tmp[('m',ci)]
        for ci in list(self.transport.coefficients.advection.keys()):
            self.q_last[('f',ci)][:] = self.q_tmp[('f',ci)]
            self.q_last[('f',ci)] *= (1.0-self.thetaAdvection)
        for ci in list(self.transport.coefficients.reaction.keys()):
            self.q_last[('r',ci)][:] = self.q_tmp[('r',ci)]
            self.q_last[('r',ci)] *= (1.0-self.thetaReaction)
        for ci in list(self.transport.coefficients.hamiltonian.keys()):
            self.q_last[('H',ci)][:] = self.q_tmp[('H',ci)]
            self.q_last[('H',ci)] *= (1.0-self.thetaHamiltonian)
        for ci,ckDict in self.transport.coefficients.diffusion.items():
            for ck in list(ckDict.keys()):
                self.q_last[('a',ci,ck)][:] = self.q_tmp[('a',ci,ck)]
                self.q_last[('a',ci,ck)] *= (1.0-self.thetaDiffusion)
        for ci in list(self.transport.coefficients.advection.keys()):
            if ('f',ci) in self.ebq_last:
                self.ebq_last[('f',ci)][:] = self.ebq_tmp[('f',ci)]
                self.ebq_last[('f',ci)] *= (1.0-self.thetaAdvection)
        for ci,ckDict in self.transport.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                if ('a',ci,ck) in self.ebq_last:
                    self.ebq_last[('a',ci,ck)][:]  = self.ebq_tmp[('a',ci,ck)]
                    self.ebq_last[('a',ci,ck)] *= (1.0-self.thetaDiffusion)
    def initializeSpaceHistory(self,resetFromDOF=False):
        self.tLast = self.t
        self.dtLast = self.dt
        for ci in list(self.transport.coefficients.advection.keys()):
            self.q_last[('f',ci)][:] = self.q_tmp[('f',ci)]
            self.q_last[('f',ci)] *= (1.0-self.thetaAdvection)
        for ci in list(self.transport.coefficients.reaction.keys()):
            self.q_last[('r',ci)][:] = self.q_tmp[('r',ci)]
            self.q_last[('r',ci)] *= (1.0-self.thetaReaction)
        for ci in list(self.transport.coefficients.hamiltonian.keys()):
            self.q_last[('H',ci)][:] = self.q_tmp[('H',ci)]
            self.q_last[('H',ci)] *= (1.0-self.thetaHamiltonian)
        for ci,ckDict in self.transport.coefficients.diffusion.items():
            for ck in list(ckDict.keys()):
                self.q_last[('a',ci,ck)][:] = self.q_tmp[('a',ci,ck)]
                self.q_last[('a',ci,ck)] *= (1.0-self.thetaDiffusion)
        for ci in list(self.transport.coefficients.advection.keys()):
            if ('f',ci) in self.ebq_last:
                self.ebq_last[('f',ci)][:] = self.ebq_tmp[('f',ci)]
                self.ebq_last[('f',ci)] *= (1.0-self.thetaAdvection)
        for ci,ckDict in self.transport.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                if ('a',ci,ck) in self.ebq_last:
                    self.ebq_last[('a',ci,ck)][:]  = self.ebq_tmp[('a',ci,ck)]
                    self.ebq_last[('a',ci,ck)] *= (1.0-self.thetaDiffusion)
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL

IMEX_AD_Euler = ForwardEuler_A
IMEX_HJ_Euler = ForwardEuler_H

class VBDF(TI_base):
    """
    Variable coefficient bdf's just first and second order
    """
    def __init__(self,transport,timeOrder=2,integrateInterpolationPoints=False):
        TI_base.__init__(self,transport)
        self.isAdaptive = True
        self.max_order = 2
        self.m_history=[]       #history for mass variable, m^n,m^{n-1}, ... n-k+1
        self.mt_history=[]      #history for mass derivative m_t, m_t^{n},
        self.m_pred   = {}     #predictor for mass at t^{n+1}, m^{n+1,p}
        self.error_estimate    = {}     #\vec e^{n+1} for mass term
        self.alpha_bdf = 1.0    #m_t \approx \alpha m^{n+1} + \beta^n
        self.beta_bdf  = {}
        self.dt_history=numpy.zeros((self.max_order,),'d')
        self.m_tmp     ={}      #storage for m,m_t at last approximation
        self.mt_tmp    ={}
        self.work      ={}      #need to clean up storage
        #for now add m_last to keep same interface as BackwardEuler for ns-ls-two phase flow prestep and poststep
        self.m_last    ={}
        self.targetTimeOrder = timeOrder #target order
        assert self.targetTimeOrder <= self.max_order
        self.timeOrder = 1
        self.nUpdatesTimeHistoryCalled   = 0
        self.integrateInterpolationPoints = integrateInterpolationPoints

        self.predictorPolynomialOrder = 1#predictorPolynomialOrder
        self.provides_dt_estimate     = False
        self.provides_initialGuess    = True
        self.needToCalculateBDFCoefs= True
        for n in range(self.max_order):
            self.m_history.append({})
            self.mt_history.append({})
            for ci in self.massComponents:
                if ('m',ci) in transport.q:
                    self.m_history[n][ci] = numpy.array(transport.q[('m',ci)])
                    self.mt_history[n][ci] = numpy.zeros(transport.q[('m',ci)].shape,'d')
                else:
                    assert False, "transport.q needs key ('m',%s) " % ci
        for ci in self.massComponents:
            self.m_pred[ci] = numpy.array(transport.q[('m',ci)])
            self.error_estimate[ci]  = numpy.array(transport.q[('m',ci)])
            self.m_tmp[ci]  = numpy.array(transport.q[('m',ci)])
            self.mt_tmp[ci] = numpy.zeros(transport.q[('m',ci)].shape,'d')
            self.beta_bdf[ci] = numpy.array(transport.q[('m',ci)])
            self.work[ci]     = numpy.zeros(transport.q[('m',ci)].shape,'d')
            self.m_last[ci] = self.m_history[0][ci]
    def calculateElementCoefficients(self,q):
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.calculateCoefs()
        #mwf debug
        logEvent("VBDF calculateElementCoefficients t= %s dt= %s alpha= %s " % (self.t,self.dt,self.alpha_bdf))
        for ci in self.massComponents:
            self.m_tmp[ci][:] = q[('m',ci)]
            self.mt_tmp[ci][:]= q[('m',ci)]#self.alpha_bdf*q[('m',ci)]
            self.mt_tmp[ci] *= self.alpha_bdf
            self.mt_tmp[ci]  += self.beta_bdf[ci]
            q[('mt',ci)][:]   = self.mt_tmp[ci]
            for cj in range(self.nc):
                if ('dmt',ci,cj) in q:
                    q[('dmt',ci,cj)][:] = q[('dm',ci,cj)]
                    q[('dmt',ci,cj)] *= self.alpha_bdf
                if ('dmt_sge',ci,cj) in q:
                    q[('dmt_sge',ci,cj)][:] = q[('dm_sge',ci,cj)]
                    q[('dmt_sge',ci,cj)] *= self.alpha_bdf

    def initialize_dt(self,t0,tOut,q):
        """
        This routine is bad right now, because we don't want to compute an internal dt anymore
        just the error estimates

        Controller now has to take care of self.t and self.dt
        """
        logEvent("VBDF initialize_dt tLast= %s t=%s dt=%s t0=%s tOut=%s " % (self.tLast,self.t,self.dt,t0,tOut))
        self.tLast=t0
        self.t = tOut
        self.dt = tOut - t0
        self.needToCalculateBDFCoefs = True

    def choose_dt(self):
        """
        This routine is bad right now, because we don't want to compute an internal dt anymore
        just the error estimates

        Controller now has to take care of self.t and self.dt
        """
        self.needToCalculateBDFCoefs = True

    def setInitialGuess(self):
        """
        set an initial guess for the next step
        need to figure out how to synchronize this

        predictor necessary for error estimates
        """
        self.calculatePredictor()

    def lastStepErrorOk(self):
        """
        was the last time step acceptable or not
        This routine now just computes the error estimate ...
        """
        self.computeErrorEstimate()
        return True
    def initializeTimeHistory(self,resetFromDOF=True):
        logEvent("VBDF initialTimeHistory call %s tLast=%s t=%s dt= %s dt_history[0] = %s " % (self.nUpdatesTimeHistoryCalled,
                                                                                          self.tLast,self.t,self.dt,self.dt_history[0]),1)
        for n in range(self.max_order-1):
            for ci in self.massComponents:
                self.m_history[n+1][ci].flat[:] =self.m_history[n][ci].flat
                self.mt_history[n+1][ci].flat[:]=self.mt_history[n][ci].flat
            #
        for ci in self.massComponents:
            self.m_history[0][ci].flat[:] = self.m_tmp[ci].flat
            self.mt_history[0][ci].flat[:]= self.mt_tmp[ci].flat
        self.needToCalculateBDFCoefs = True

    def updateTimeHistory(self,resetFromDOF=False):
        #go ahead and make sure error estimate for last step is computed before updating?
        #self.computeErrorEstimate()
        self.nUpdatesTimeHistoryCalled  += 1
        #what to do about first call?, getting multiple calls
        logEvent("VBDF updateTimeHistory call %s tLast=%s t=%s dt= %s dt_history[0] = %s " % (self.nUpdatesTimeHistoryCalled,
                                                                                               self.tLast,self.t,self.dt,self.dt_history[0]),1)
        self.tLast = self.t
        for n in range(self.max_order-1):
            for ci in self.massComponents:
                self.m_history[n+1][ci].flat[:] =self.m_history[n][ci].flat
                self.mt_history[n+1][ci].flat[:]=self.mt_history[n][ci].flat
            #
            self.dt_history[n+1]   =self.dt_history[n]
        #
        self.dt_history[0] = self.dt
        for ci in self.massComponents:
            self.m_history[0][ci].flat[:] = self.m_tmp[ci].flat
            self.mt_history[0][ci].flat[:]= self.mt_tmp[ci].flat

        self.needToCalculateBDFCoefs = True
        #decide where this goes
        self.chooseOrder()
    #
    def postAdaptUpdate(self,oldTime):
        """This looks exactly like updateTimeHistory with the exeption of setting self.tLast=self.t"""
        self.nUpdatesTimeHistoryCalled  = oldTime.nUpdatesTimeHistoryCalled
        for n in range(self.max_order-1):
            for ci in self.massComponents:
                self.m_history[n+1][ci].flat[:] =self.m_history[n][ci].flat
                self.mt_history[n+1][ci].flat[:]=self.mt_history[n][ci].flat
            #
            self.dt_history[n+1]   = self.dt_history[n]
        #
        self.dt_history[0] = self.dt
        for ci in self.massComponents:
            self.m_history[0][ci].flat[:] = self.m_tmp[ci].flat
            self.mt_history[0][ci].flat[:]= self.mt_tmp[ci].flat

        self.needToCalculateBDFCoefs = True
        #decide where this goes
        self.chooseOrder()

        #unclear why this is needed; a change of machine precision in these values yield larger deviations in the gauges
        self.dt_history[:] = oldTime.dt_history[:]

    def computeErrorEstimate(self):
        r"""calculate :math:`\vec{e}^{n+1}`

        depending on order of approximation To be consistent, this
        must be called after step taken but before update time history

        Initially, use

        .. math::

            (\vec y^{n+1}-\vec y^{n+1,p})/2

        for first order and

        .. math::

            r/(1+r)(\vec y^{n+1}-(1+r)\vec y^{n}+r\vec y^{n-1})

        for second order.
        """
        #mwf debug
        #import pdb
        #pdb.set_trace()
        if self.timeOrder == 2:
            r = self.dt/self.dt_history[0]
            for ci in list(self.mt_tmp.keys()):
                self.error_estimate[ci].flat[:] = self.m_history[1][ci].flat
                self.error_estimate[ci]   *= r
                self.work[ci].flat[:]           = self.m_history[0][ci].flat
                self.work[ci]             *= -(1.0+r)
                self.error_estimate[ci]   += self.work[ci]
                self.error_estimate[ci]   += self.m_tmp[ci]
                self.error_estimate[ci]   *= r/(1.0+r)
            #
        else:
            for ci in list(self.mt_tmp.keys()):
                self.error_estimate[ci].flat[:] = self.m_tmp[ci].flat
                self.error_estimate[ci]   -= self.m_pred[ci]
                self.error_estimate[ci]   *= 0.5

    def calculatePredictor(self):
        #mwf debug
        #import pdb
        #pdb.set_trace()
        #for now just first order predictor
        for ci in self.massComponents:
            self.m_pred[ci].flat[:] = self.mt_history[0][ci].flat
            self.m_pred[ci]   *= self.dt
            self.m_pred[ci]   += self.m_history[0][ci]
        logEvent("VBDF calculatePredictor tLast= %s t= %s dt= %s " % (self.tLast,self.t,self.dt),1)

    def calculateCoefs(self):
        if self.needToCalculateBDFCoefs:
            dtInv = 1.0/self.dt
            if self.timeOrder == 2:
                r         = self.dt/self.dt_history[0]
                self.alpha_bdf = (1.0+2.0*r)/(1.0+r)*dtInv

                b0        =-(1.0+r)*dtInv
                b1        =r*r/(1.0+r)*dtInv
                for ci in self.massComponents:
                    self.beta_bdf[ci].flat[:] = self.m_history[0][ci].flat
                    self.beta_bdf[ci]   *= b0
                    self.work[ci].flat[:]     = self.m_history[1][ci].flat
                    self.work[ci]       *= b1
                    self.beta_bdf[ci]   += self.work[ci]
            else: #default is first order
                self.alpha_bdf = dtInv
                for ci in self.massComponents:
                    self.beta_bdf[ci].flat[:] = self.m_history[0][ci].flat
                    self.beta_bdf[ci]   *= -dtInv
                #
            #
            #mwf debug
            self.needToCalculateBDFCoefs = False
    #
    def state(self):
        """What a hot start needs to continue with the same history: the masses and their
        derivatives at the last steps, the last step sizes, and how many steps have been
        taken (which sets the order). The spin-up rebuilds all of it from the restart
        state, as a first-order start."""
        st = {"nUpdatesTimeHistoryCalled":int(self.nUpdatesTimeHistoryCalled),
              "dt_history":self.dt_history.copy()}
        for n in range(self.max_order):
            for ci in self.massComponents:
                st["m_history_%d_%d" % (n,ci)] = self.m_history[n][ci].copy()
                st["mt_history_%d_%d" % (n,ci)] = self.mt_history[n][ci].copy()
        return st
    def set_state(self,st):
        self.nUpdatesTimeHistoryCalled = int(st["nUpdatesTimeHistoryCalled"])
        self.dt_history[:] = st["dt_history"]
        for n in range(self.max_order):
            for ci in self.massComponents:
                #in place: m_last is m_history[0]
                self.m_history[n][ci][:] = st["m_history_%d_%d" % (n,ci)]
                self.mt_history[n][ci][:] = st["mt_history_%d_%d" % (n,ci)]
        self.chooseOrder()
        self.needToCalculateBDFCoefs = True
    def chooseOrder(self):
        if self.nUpdatesTimeHistoryCalled  < 1:
            self.timeOrder = 1
        else:
            self.timeOrder = self.targetTimeOrder
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        flags = ['timeOrder']
        for flag in flags:
            if flag in dir(nOptions):
                val = getattr(nOptions,flag)
                if flag == 'timeOrder':
                    setattr(self,'targetTimeOrder',val)
                else:
                    setattr(self,flag,val)

        self.predictorPolynomialOrder = self.timeOrder-1

#mwf hack for cfl calculations
from . import cfemIntegrals
class ExplicitRK_base(TI_base):
    r"""
    base class for explicit RK schemes which can be written in
    the Shu Osher 88 (Cockburn Shu 89) style generic formula

    u^0 = u^n
    u^l = \sum_{k=0}^{l-1}\alpha_{l,k}u^k + \beta_{l,k} \Delta t L(u^{k},t^n + d_{k} \Delta t^n)
    u^{n+1} = u^{m}

    We will try to include these schemes in a Rothe framework rather than MOL

    so in calculate element coefficients, at stage number l, u_t will be replaced with

      u_t --> (u - \sum_{k=0}^{l-1} \alpha_{l,k} u^k)/\Delta t

    or

      u_t -->  (u - \sum_{k=0}^{l-1} \alpha_{l,k} u^k)

    depending on whether \Delta t is assumed to be in the mass matrix or not

    The spatial residual at stage l will be replaced with (in calculateElementSpatialResidual)

      r --> \sum_{k=0,l-1} \beta_{l,k} r_k

    or

      r --> \Delta t \sum_{k=0,l-1} \beta_{l,k} r_k

    depending on whether \Delta t is assumed to be in the mass matrix or not
    Here r_k corresponds to

       L(u^{k},t^n + d_{k} \Delta t^n)


    Substeps in NumericalSolution framework correspond to stage values
    in the RK integration. These are setup in generateSubsteps. The
    indexing is a little confusing, since the residuals are 'cached'
    when they come into calculateElementResidual in order to be used
    at the next stage. This means the time levels in substep have to
    be for the following stage level.  This is modulo the number of
    stages so t^{n+1} at the last stage --> t^n for the next time step



    """
    def __init__(self,transport,timeOrder=1,nStages=1,runCFL=0.1,usingDTinMass=True):
        TI_base.__init__(self,transport)
        self.isAdaptive=True
        self.timeOrder  = timeOrder   #order of approximation
        self.nStages= nStages #number of stages total
        self.lstage = 0       #last stage completed
        self.setCoefficients() #set alpha,beta,d
        self.usingDTinMass = usingDTinMass

        self.eqTerms = ['m','res']
        self.stageValues = {}
        for ci in range(self.nc):
            if ('m',ci) in transport.q:
                self.massIsImplicit[ci]       = True #assume linearly implicit disc.
                self.advectionIsImplicit[ci]  = False
                self.diffusionIsImplicit[ci]  = False
                self.reactionIsImplicit[ci]   = False
                self.hamiltonianIsImplicit[ci]= False
                self.stabilizationIsImplicit[ci]   = False
                self.shockCapturingIsImplicit[ci] = False
            #end if
        #end ci
        for term in ['m']:
            self.stageValues[term] = {}
            for ci in range(self.nc):
                if (term,ci) in transport.q:
                    self.stageValues[term][ci] = [] #stage values for u, u[0]= u^n not U^1
                    for i in range(self.nStages+1):
                        self.stageValues[term][ci].append(
                            numpy.array(transport.q[(term,ci)]))
                    #end i
                #end if
            #end ci
        #end element quadrature terms
        #for term in ['m']:
        #    bndterm = 'eb.'+term
        #    self.stageValues[bndterm] = {}
        #    for ci in range(self.nc):
        #        if transport.ebq.has_key((term,ci)):
        #            self.stageValues[bndterm][ci] = [] #stage values for u, u[0]= u^n not U^1
        #            for i in range(self.nStages+1):
        #                self.stageValues[bndterm][ci].append(
        #                    numpy.array(transport.ebq[(term,ci)]))
        #            #end i
        #        #end if
        #    #end ci
        #end boundary quadrature terms
        #now take care of residual, keep copy if 'm' is in q
        term = 'res'
        self.stageValues[term] = {}
        for ci in range(self.nc):
            if ('m',ci) in transport.q:
                self.stageValues[term][ci] = [] #stage values for u, u[0]= u^n not U^1
                for i in range(self.nStages+1):
                    self.stageValues[term][ci].append(
                        numpy.array(transport.elementResidual[ci]))
                #end i
            #end if
        #end ci
        #now get cfl as shallow copy
        self.cfl = {}
        for ci in range(self.nc):
            if ('cfl',ci) in transport.q:
                self.cfl[ci] = transport.q[('cfl',ci)]
            #end if
        #end for
        self.runCFL=runCFL
        self.dtLast=None
        self.dtRatioMax = 2.0 #come up with better value

    #
    def set_dt(self,DTSET):
        self.dt=DTSET
        #do not change t because of substeps, default self.t = self.tLast + self.dt
    def resetOrder(self,order,stages):
        """
        resize and reinitialize if necessary
        """
        failed = False
        if order == self.timeOrder:
            return failed
        self.timeOrder = order
        self.nStages = stages
        self.lstage  = 0
        self.setCoefficients()#alpha  = self.getAlphaCoefs(order)
        assert self.eqTerms == ['m','res'], "something wrong with self.eqTerms=%s " % self.eqTerms

        #get stabe value keys and sizes and keep those
        stageValShapes = {};
        for term in self.eqTerms:
            stageValShapes[term] = {}
            for ci in range(self.nc):
                if ci in self.stageValues[term]:
                    stageValShapes[term][ci] = self.stageValues[term][ci][0].shape #assume all stages same shape
        self.stageValues = {}
        for term in ['m']:
            self.stageValues[term] = {}
            for ci in range(self.nc):
                if ci in stageValShapes[term]: #transport.q.has_key((term,ci)):
                    self.stageValues[term][ci] = [] #stage values for u, u[0]= u^n not U^1
                    for i in range(self.nStages+1):
                        self.stageValues[term][ci].append(
                            numpy.zeros(stageValShapes[term][ci],'d'))
                    #end i
                #end if
            #end ci
        #end element quadrature terms

        #now take care of residual, keep copy if 'm' is in q
        term = 'res'
        self.stageValues[term] = {}
        for ci in range(self.nc):
            if ci in stageValShapes[term]: #if transport.q.has_key(('m',ci)):
                self.stageValues[term][ci] = [] #stage values for u, u[0]= u^n not U^1
                for i in range(self.nStages+1):
                    self.stageValues[term][ci].append(
                        numpy.zeros(stageValShapes[term][ci],'d'))
                #end i
            #end if
        #end ci
    #
    def setCoefficients(self):
        r"""set alpha,beta, d for

        .. math::

            u^0 = u^n
            u^l = \sum_{k=0}^{l-1}\alpha_{l,k}u^k + \beta_{l,k} \Delta t L(u^{k},t^n + d_{k} \Delta t^n)
            u^{n+1} = u^{m}

        must be implemented in derived class

        Note that

        .. math::

            alpha[l,k] -->  alpha_{l+1,k}
            beta[l,k]  -->  beta_{l+1,k}
            dcoefs[k]  -->  d_{k+1} 

        because of the whole caching\delayed eval deal second index
        (k) is actual level

        """
        self.alpha = None
        self.beta  = None
        self.dcoefs= None
    def updateStage(self):
        """
        increment stage counter by 1.
        lstage here is last stage completed
        """
        if self.lstage < self.nStages-1:
            self.lstage +=1
        if self.lstage < self.nStages:
            self.t = self.substeps[self.lstage]
        if self.lstage < 0 or self.lstage > self.nStages:
            logEvent("""stage= %d out of allowed bounds [0,%d]""" % (self.lstage,self.nStages))
    def calculateElementCoefficients(self,q):
        r"""
        set m_t as

              m_t --> (m - \sum_{k=0}^{l-1} \alpha_{l,k} m^k)/\Delta t

        """
        if self.dt <= 1.0e-24:
            logEvent('WARNING dt= ',self.dt,' too small in updateElementCoefficients quitting ')
            sys.exit(1)
        if self.usingDTinMass == False:
            return self.calculateElementCoefficientsNoDtInMass(q)
        #mwf debug
        logEvent("""ExplicitRKbase calcElemenCoefs lstage= %d nStages= %d dt= %s """ % (self.lstage,self.nStages,self.dt))
        #import pdb
        #pdb.set_trace()
        for ci in range(self.nc):#loop through components
            if ('m',ci) in q:
                self.stageValues['m'][ci][self.lstage+1].flat[:] = q[('m',ci)].flat[:]
                q[('mt',ci)].flat[:] = q[('m',ci)].flat[:]
                for i in range(self.lstage+1): #go from 0 to l-1
                    q[('mt',ci)].flat[:]-= self.alpha[self.lstage,i]*self.stageValues['m'][ci][i].flat[:]
                #end i loop through stages
                #mwf debug
                logEvent("""ExplicitRKbase calcElemenCoefs lstage= %d nStages= %d max dt diff= %s """ % (self.lstage,self.nStages,max(numpy.absolute(q[('mt',0)].flat))))
                #if max(numpy.absolute(q[('mt',0)].flat)) > 1.0e-6:
                #    import pdb
                #    pdb.set_trace()
                q[('mt',ci)]/= self.dt
                for cj in range(self.nc):
                    if ('dm',ci,cj) in q:
                        q[('dmt',ci,cj)][:]=q[('dm',ci,cj)][:]
                        q[('dmt',ci,cj)][:]/=self.dt
                    #end q has key
                #end cj
            #end if q has key
        #end ci
    def calculateElementCoefficientsNoDtInMass(self,q):
        r"""
        set m_t as

              m_t --> (m - \sum_{k=0}^{l-1} \alpha_{l,k} m^k)

        """
        assert self.dt > 1.0e-24,"dt = %d too small in calculateElementCoefficients" % self.dt
        #mwf debug
        logEvent("""ExplicitRK calcElemenCoefsNoDt lstage= %d nStages= %d""" % (self.lstage,self.nStages))
        for ci in range(self.nc):#loop through components
            if ('m',ci) in q:
                self.stageValues['m'][ci][self.lstage+1].flat[:] = q[('m',ci)].flat[:]
                q[('mt',ci)].flat[:] = q[('m',ci)].flat[:]
                for i in range(self.nStages): #go from 0 to s-1
                    q[('mt',ci)].flat[:]-= self.alpha[self.lstage,i]*self.stageValues['m'][ci][i].flat[:]
                #end i loop through stages
                for cj in range(self.nc):
                    if ('dm',ci,cj) in q:
                        q[('dmt',ci,cj)][:]=q[('dm',ci,cj)][:]
                    #end q has key
                #end cj
            #end if q has key
        #end ci

    def calculateElementSpatialResidual(self,elementResidual):
        r"""
        The spatial residual at stage l will be replaced with (in calculateElementSpatialResidual)

           r --> \sum_{k=0,l-1} \beta_{l,k} r_k


        """
        if self.usingDTinMass == False:
            return self.calculateElementSpatialResidualNoDtInMass(elementResidual)
        for ci in list(self.stageValues['res'].keys()):
            self.stageValues['res'][ci][self.lstage+1].flat[:]=elementResidual[ci].flat[:]
            #start at bottom
            elementResidual[ci].flat[:] = self.stageValues['res'][ci][0].flat[:]
            elementResidual[ci].flat[:]*= self.beta[self.lstage,0]
            for i in range(1,self.nStages):
                #no good way to avoid temporary here yet
                elementResidual[ci].flat[:] += self.beta[self.lstage,i]*self.stageValues['res'][ci][i].flat[:]
        #end ci
    #end calculateElementSpatialResidual
    def calculateElementSpatialResidualNoDtInMass(self,elementResidual):
        r"""
        The spatial residual at stage l will be replaced with (in calculateElementSpatialResidual)

          r --> \Delta t \sum_{k=0,l-1} \beta_{l,k} r_k


        """
        for ci in list(self.stageValues['res'].keys()):
            self.stageValues['res'][ci][self.lstage+1].flat[:]=elementResidual[ci].flat[:]
            #start at bottom
            elementResidual[ci].flat[:] = self.stageValues['res'][ci][0].flat[:]
            elementResidual[ci].flat[:]*= self.dt*self.beta[self.lstage,0]
            for i in range(1,self.nStages):
                #no good way to avoid temporary here yet
                elementResidual[ci].flat[:] += self.dt*self.beta[self.lstage,i]*self.stageValues['res'][ci][i].flat[:]
        #end ci
    #end calculateElementSpatialResidual
    def calculateElementSpatialJacobian(self,elementJacobian):
        """
        Recalculate the part of the element Jacobian due to the spatial terms
        """
        #mwf should not be called if everything is explicit
        #pass
        #mwf debug
        #print "Explicit RK zeroing  jacobian"
        for ci in list(elementJacobian.keys()):
            for cj in list(elementJacobian[ci].keys()):
                elementJacobian[ci][cj].flat[:] = 0.0
            #end cj
        #end ci
    def calculateElementBoundaryCoefficients(self,ebq):
        """
        Calculate m_t, and dm_t and recalculate any of the other coefficients on element boundaries
        do I need these to be synchronized with q or do the elementBoundary quad values get
        lagged in the spatial residual
        """
        pass

    def calculateElementSpatialBoundaryJacobian(self,elementBoundaryJacobian,elementBoundaryJacobian_eb):
        """
        Recalculate the part of the element Jacobian due to the spatial terms
        along element boundaries
        """
        #should not be used if everthing is explicit
        #mwf right now any numericalFlux triggers Jacobian evaluate for fluxex
        #pass
        #mwf debug
        #print "Explicit RK zeroing boundary jacobian"
        for ci in list(elementBoundaryJacobian.keys()):
            for cj in list(elementBoundaryJacobian[ci].keys()):
                elementBoundaryJacobian[ci][cj].fill(0.0)
            #end cj
        #end ci
        for ci in list(elementBoundaryJacobian_eb.keys()):
            for cj in list(elementBoundaryJacobian_eb[ci].keys()):
                elementBoundaryJacobian_eb[ci][cj].fill(0.0)
            #end cj
        #end ci
    def calculateExteriorElementSpatialBoundaryJacobian(self,elementBoundaryJacobian):
        for ci in list(elementBoundaryJacobian.keys()):
            for cj in list(elementBoundaryJacobian[ci].keys()):
                elementBoundaryJacobian[ci][cj].fill(0.0)
    def initialize_dt(self,t0,tOut,q):
        self.tLast = t0
        self.choose_dt()
    def choose_dt(self):
        """
        Modify self.dt
        """
        #mwf debug
        import pdb
        #pdb.set_trace()
        maxCFL=1.0e-6
        for ci in range(self.nc):
            if ci in self.cfl:
                maxCFL=max(globalMax(self.cfl[ci].max()),maxCFL)
            #end has key
        #mwf running into problems again when want to calculate cfl
        #if running with shock capturing but no stabilization, cfl is zero
        #coming in
        if maxCFL <= 10.0*1.0e-6:
            #mwf debug
            #import pdb
            #pdb.set_trace()
            for ci in range(self.nc):
                if ci in self.cfl:
                    if (('df',ci,ci) in self.transport.q and ('a',ci,ci) in self.transport.q and
                        ('dr',ci,ci) in self.transport.q and ('dmt',ci,ci) in self.transport.q and
                        ('a',ci,ci) in self.transport.q and ('dphi',ci,ci) in self.transport.q):
                        cfemIntegrals.calculateDimensionlessNumbersADR(self.transport.mesh.nElements_global,
                                                                       self.transport.nQuadraturePoints_element,
                                                                       self.transport.nSpace_global,
                                                                       self.transport.elementEffectiveDiametersArray,
                                                                       self.transport.q[('df',ci,ci)],
                                                                       self.transport.q[('a',ci,ci)],
                                                                       self.transport.q[('dphi',ci,ci)],
                                                                       self.transport.q[('dr',ci,ci)],
                                                                       self.transport.q[('dmt',ci,ci)],
                                                                       self.transport.q[('pe',ci)],
                                                                       self.transport.q[('cfl',ci)])
                    elif (('df',ci,ci) in self.transport.q and ('dH',ci,ci) in self.transport.q and
                          ('dm',ci,ci) in self.transport.q):
                        #not likely?
                        cfemIntegrals.calculateCFLADR2speeds(self.transport.elementEffectiveDiametersArray,
                                                             self.transport.q[('dm',ci,ci)],
                                                             self.transport.q[('df',ci,ci)],
                                                            self.transport.q[('dH',ci,ci)],
                                                             self.transport.q[('cfl',ci)])

                    elif (('df',ci,ci) in self.transport.q and ('dm',ci,ci) in self.transport.q):
                        cfemIntegrals.calculateCFLADR(self.transport.elementEffectiveDiametersArray,
                                                      self.transport.q[('dm',ci,ci)],
                                                      self.transport.q[('df',ci,ci)],
                                                      self.transport.q[('cfl',ci)])
                    elif (('dH',ci,ci) in self.transport.q and ('dm',ci,ci) in self.transport.q):
                        cfemIntegrals.calculateCFLADR(self.transport.elementEffectiveDiametersArray,
                                                      self.transport.q[('dm',ci,ci)],
                                                      self.transport.q[('dH',ci,ci)],
                                                      self.transport.q[('cfl',ci)])
                    maxCFL=max(globalMax(self.cfl[ci].max()),maxCFL)
        self.dt = self.runCFL/maxCFL
        if self.dtLast is None:
            self.dtLast = self.dt
        if self.dt/self.dtLast  > self.dtRatioMax:
            self.dt = self.dtLast*self.dtRatioMax
        self.generateSubsteps([self.tLast+self.dt])
        self.t = self.substeps[0]
    def generateSubsteps(self,tList):
        """
        create list of substeps over time values given in tList. These correspond to stages
        so need to have the right number set. For the SSP RK schemes here, t^m = t^{n+1} for
        all the stages except the first, I believe

        The order here is a little confusing. Since the residuals are 'cached' when they
        come into calculateElementResidual in order to be used at the next stage, the
        time levels have to be for the next stage
        This is modulo the number of stages so t^{n+1} at the last stage --> t^n for the
        next time step
        """
        self.substeps = []
        tLast = self.tLast
        for t in tList:
            dttmp = t-tLast
            self.substeps.extend([tLast + self.dcoefs[i]*dttmp for i in range(self.nStages)])
            #self.substeps.extend([tLast + dttmp*(i+1) for i in range(0,self.nStages-1)])#one stage ahead
            #self.substeps.append(t)#always end up at t^{n+1}
            tLast = t
            #mwf debug
            logEvent('Explicit RK dcoefs= %s substeps = %s ' % (self.dcoefs,self.substeps))
            logEvent('Explicit RK t= %s tLast=%s self.tLast= %s ' % (t,tLast,self.tLast))
    def setInitialStageValues(self):
        """
        setup all the stage values if we assume that the initial condition
        has been loaded into lstage+1
        """
        for term in self.eqTerms:
            for ci in list(self.stageValues[term].keys()):
                for i in range(self.nStages+1):
                    if i != self.lstage+1:
                        self.stageValues[term][ci][i].flat[:] = self.stageValues[term][ci][self.lstage+1].flat[:]
                    #end if
                #end for i
            #end ci
        #end term
        self.lstage = 0
    #end setInitialStageValues
    def initializeTimeHistory(self,resetFromDOF=True):
        for term in self.eqTerms:
            for ci in list(self.stageValues[term].keys()):
                if resetFromDOF:
                    assert self.transport is not None, "Must supply transport to reset from DOF"
                    if term == 'res':
                        self.stageValues[term][ci][0].flat[:] = self.stageValues['res'][ci][self.lstage+1].flat[:]
                    else:
                        self.stageValues[term][ci][0].flat[:] = self.transport.q[(term,ci)].flat
                else:
                    self.stageValues[term][ci][0].flat[:] = self.stageValues[term][ci][self.nStages].flat[:]
        self.lstage = 0
    def updateTimeHistory(self,resetFromDOF=False):
        """
        setup for next time step, cycle U^s --> U^0
        """
        self.tLast = self.t
        for term in self.eqTerms:
            for ci in list(self.stageValues[term].keys()):
                if resetFromDOF:
                    assert self.transport is not None, "Must supply transport to reset from DOF"
                    if term == 'res':
                        self.stageValues[term][ci][0].flat[:] = self.stageValues['res'][ci][self.lstage+1].flat[:]
                    else:
                        self.stageValues[term][ci][0].flat[:] = self.transport.q[(term,ci)].flat
                else:
                    self.stageValues[term][ci][0].flat[:] = self.stageValues[term][ci][self.nStages].flat[:]
        self.lstage = 0
        self.dtLast = self.dt
    #end updateTimeHistory
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL
            #need to think of a name for temporal order
        if 'timeOrder' in dir(nOptions):
            if 'nStagesTime' in dir(nOptions):
                self.resetOrder(nOptions.timeOrder,nOptions.nStagesTime)
            else:
                logEvent("WARNING ExplicitRK.setFromOptions assuming nStagesTime=timeOrder")
                self.resetOrder(nOptions.timeOrder,nOptions.timeOrder)


class LinearSSPRKintegration(ExplicitRK_base):
    r"""
    implementation of Explicit RK schemes for ODE that are SSP for
    linear operators (aka) spatial discretizations in

    .. math::

        \od{\vec y}{t} = \mat{L}\vec y

    See Gottlieb, Shu, Tadmor siam review article and notes

    Their formulation for this scheme is

    .. math::

        u^0 = u^n
        u^l = u^{l-1} + \Delta t L(u^{l-1}) l = 1,...,m-1
        u^m = \sum_{k=0}^{l-1}\alpha_{m,k}u^k +
              \alpha_{m,m-1} \Delta t L(u^{m-1})
        u^{n+1} = u^m

        so \alpha_{l,l-1} = 1, \beta_{l,l-1} = 1.0  l < m,
           \beta_{m,m-1}  = \alpha_{m,m-1}

    Apparently,

    .. math::
         d_{l} = l \Delta t l < m,
         d_m = 1.0

    which doesn't make a lot of sense for time dependent problems
    """
    def __init__(self,transport,order=1,runCFL=0.1,usingSSPRKNewton=False):
        ExplicitRK_base.__init__(self,transport,timeOrder=order,nStages=order,runCFL=runCFL,
                                 usingDTinMass = not usingSSPRKNewton)


    def setCoefficients(self):
        r"""
        alpha matrix in Gottlieb, Shu, Tadmor review paper::

          u^{l} = \sum_{k=0}^{l-1}\alpha_{l,k}u^k

        so \alpha_{l,l-1} = 1  for l < m

        beta matrix multiplies::

          \beta_{l,k} \Delta t L(u^{k},t^n + d_{k} \Delta t^n)

        so::

          \beta_{l,l-1} = 1.0 for l < m
          \beta_{m,m-1}   = \alpha_{m,m-1}

        Note that::

          alpha[l,k] -->  alpha_{l+1,k}
          beta[l,k]  -->  beta_{l+1,k}
          dcoefs[k]  -->  d_{k+1} because of the whole caching\ delayed eval deal

        second index (k) is actual level
        """
        order = self.timeOrder; stages = self.nStages
        assert order == stages, "LinearSSPRK problem order (%s) != stages (%s) " % (order,stages)
        self.alpha = numpy.zeros((stages,stages),'d')
        for l in range(stages-1):
            self.alpha[l,l] = 1.0
        assert stages < 6,"%i stage method not yet implemented" % stages
        if stages == 1:
            self.alpha[0,0] = 1.0
        elif stages == 2:
            self.alpha[1,0] = 0.5;     self.alpha[1,1] = 0.5
        elif stages == 3:
            self.alpha[2,0] = 1./3.;   self.alpha[2,1] = 1./2.; self.alpha[2,2] = 1./6.
        elif stages == 4:
            self.alpha[3,0] = 3./8.;   self.alpha[3,1] = 1./3.; self.alpha[3,2] = 1./4.;
            self.alpha[3,3] = 1./24.;
        elif stages == 5:
            self.alpha[4,0] = 11./30.; self.alpha[4,1] = 3./8.; self.alpha[4,2] = 1./6.;
            self.alpha[4,3] = 1./12.;  self.alpha[4,4] = 1./120.
        #
        self.beta = numpy.zeros((stages,stages),'d')
        for l in range(stages-1):
            self.beta[l,l] = 1.0
        self.beta[stages-1,stages-1] = self.alpha[stages-1,stages-1]
        #end if on order
        self.dcoefs = numpy.zeros(stages,'d')
        for l in range(stages-1):
            self.dcoefs[l] = float(l+1) #add 1 because of caching used in elementResidual evals

        self.dcoefs[stages-1] = 1.0
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        ExplicitRK_base.setFromOptions(self,nOptions)


        if 'usingSSPRKNewton' in dir(nOptions):
            self.usingDTinMass = not nOptions.usingSSPRKNewton




class SSPRKPIintegration(ExplicitRK_base):
    r"""So called SSP RK integration with limiting step at end of each
    phase This one holds the original TVD RK schemes up to 3rd order
    1st order is Forward Euler

    2nd order

    .. math::

        u^1 = u^n + \Delta t L(u^n)

        u^2 = 0.5(u^n + u^1) + 0.5\Delta t L(u^1,t^n + \Delta t^n), same stage values etc linear SSPRK

    3rd order

    .. math::

        u^1 =  u^n + \Delta t L(u^n,t^n)

        u^2 = 0.75 u^n + 0.25u^1  + 0.25\Delta t L(u^1,t^n + \Delta t^n),

        u^3 = 1/3 u^n + 2/3 u^2 + 2/3 \Delta t L(u^2,t^n + 0.5\Delta t^n)

    generic formula is

    .. math::

        u^0 = u^n

        u^l = \sum_{k=0}^{l-1}\alpha_{l,k}u^k + \beta_l \Delta t L(u^{l-1},t^n + d_{l-1} \Delta t^n)

        u^{n+1} = u^{m}

    so :math:`beta_l \to beta_{l,k-1}` in our basic framework

    Try to put this into Rothe paradigm with linear implicit mass matrix
    evaluation by saying at stage l

    .. math::

        m_t \approx \frac{u^l - \sum_{k=0}^{l-1}\alpha_{l,k}u^k}{\Delta t}

    spatial residual --> spatial residual * beta_l

    and solve system for :math:`u^l`

    The limiting right now assumes same space for each component
    """
    def __init__(self,transport,order=1,runCFL=0.1,limiterType=None,usingSSPRKNewton=False):
        ExplicitRK_base.__init__(self,transport,timeOrder=order,nStages=order,runCFL=runCFL,
                                 usingDTinMass = not usingSSPRKNewton)

        self.limiter=None; self.uLimDof = None
        self.limiterType = limiterType
        if self.limiterType is not None:
            self.limiter = limiterType(transport.mesh,transport.nSpace_global,
                                       transport.u,
                                       transport=transport)
            self.uLimDof = {}
            for ci in range(self.nc):
                self.uLimDof[ci]=numpy.zeros((transport.u[ci].dof.shape),'d')

    def setCoefficients(self):
        """
        See Cockburn and Hou and Shu

        For TVD schemes beta_{l,k} = 0  k < l-1

        Note that
          
        alpha[l,k] -->  alpha_{l+1,k}
        beta[l,k]  -->  beta_{l+1,k}
        dcoefs[k]  -->  d_{k+1} because of the whole caching\\ delayed eval deal

        """
        order = self.timeOrder; stages = self.nStages
        assert order == stages, "SSPRKPI problem order (%s) != stages (%s) " % (order,stages)
        self.alpha = numpy.zeros((stages,stages),'d')
        assert order <= 3, "Order = %s must be <= 3 " % (order)

        self.alpha = numpy.zeros((stages,stages),'d')
        if order == 1:
            self.alpha[0,0] = 1.0
        elif order == 2:
            self.alpha[0,0] = 1.0;   self.alpha[1,0] = 0.5;  self.alpha[1,1] = 0.5
        elif order == 3:
            self.alpha[0,0] = 1.0;   self.alpha[1,0] = 0.75; self.alpha[1,1] = 0.25;
            self.alpha[2,0] = 1./3.; self.alpha[2,1] = 0.0;  self.alpha[2,2] = 2./3.
        self.beta = numpy.zeros((stages,stages),'d')

        if order == 1:
            self.beta[0,0] = 1.0
        elif order == 2:
            self.beta[0,0] = 1.0; self.beta[1,1] = 0.5
        elif order == 3:
            self.beta[0,0] = 1.0; self.beta[1,1] = 0.25; self.beta[2,2] = 2.0/3.0
        #
        #1st order t = [t^{n+1}]
        #2nd order t = [t^{n+1},t^{n+1}]
        #3rd order t = [t^{n+1},t^{n+1}/2,t^{n+1}]
        #
        self.dcoefs = numpy.zeros((stages),'d')
        if order == 1:
            self.dcoefs[0] = 1.0
        elif order == 2:
            self.dcoefs[0] = 1.0; self.dcoefs[1] = 1.0
        elif order == 3:
            self.dcoefs[0] = 1.0; self.dcoefs[1] = 0.5; self.dcoefs[2] = 1.0

    def updateStage(self):
        """
        increment stage counter by 1.
        lstage here is last stage completed
        """
        if self.limiterType is not None:
            self.limiter.applySlopeLimiting(self.transport.u,
                                            self.uLimDof)
            for ci in range(self.nc):
                self.transport.u[ci].dof.flat[:] = self.uLimDof[ci].flat[:]
                assert self.transport.u[ci].par_dof is not None
                self.transport.u[ci].par_dof.scatter_forward_insert()

                #if par_u is not None:
                #    #mwf debug
                #    for eN in range(self.transport.mesh.nElements_owned,self.transport.mesh.nElements_global):
                #        for i in range(self.transport.u[ci].femSpace.dofMap.l2g.shape[1]):
                #            self.transport.u[ci].dof[self.transport.u[ci].femSpace.dofMap.l2g[eN,i]] = 0.0
                #    u.flat[:] = self.transport.u[ci].dof.flat[:]
                #    logEvent("Min_dt_RKcontroller performing scatter forward t= %s " % self.t)
                #    par_u.scatter_forward_insert()
                #    self.transport.u[ci].dof.flat[:] = u.flat[:]
        #if

        #update coefficients how?
        #stage values get reset here
        self.transport.calculateCoefficients()
        #time integration loads in new residual in this call
        self.transport.calculateElementResidual()
        #limiting
        assert self.lstage >= 0 and self.lstage <= self.nStages,"""stage= %d out of allowed bounds [0,%d]""" % (self.lstage,self.nStages)
        if self.lstage < self.nStages-1:
            self.lstage +=1
        if self.lstage < self.nStages:
            self.t = self.substeps[self.lstage]
        #mwf debug
        #print """SSPRK updateStage finished lstage= %d m[0]= %s""" % (self.lstage,
        #                                                              self.stageValues['m'][0][self.lstage])
        #if self.lstage < self.nStages:
        #    print """SSPRK updateStage lstage+1= %d m[0]= %s """ % (self.lstage+1,
        #                                                            self.stageValues['m'][0][self.lstage+1])

    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        ExplicitRK_base.setFromOptions(self,nOptions)

        if 'limiterType' in dir(nOptions) and nOptions.limiterType is not None:
            self.limiter = None; self.limiterType = None; self.uLimDof = {}
            self.limiterType = nOptions.limiterType
            self.limiter = self.limiterType(self.transport.mesh,
                                            self.transport.nSpace_global,
                                            self.transport.u,
                                            transport=self.transport)

            for ci in range(self.nc):
                #for limiting, in values
                self.uLimDof[ci] = numpy.zeros((self.transport.u[ci].dof.shape),'d')

            #
            self.limiter.setFromOptions(nOptions)
        #

        if 'usingSSPRKNewton' in dir(nOptions):
            self.usingDTinMass = not nOptions.usingSSPRKNewton

class LinearSSPRKPIintegration(LinearSSPRKintegration):
    """
    Linear SSP RK integration with limiting step at end of each phase, this one
    just uses linear SSP RK coefficients so not really correct NL SSP (TVD) RK
    for 3rd order (or higher)

    Way limiter right now assumes same space for everything
    """
    def __init__(self,transport,order=1,runCFL=0.1,limiterType=None,usingSSPRKNewton=False):
        """
        Later on, pass in limiter object I guess
        """
        LinearSSPRKintegration.__init__(self,transport,order=order,runCFL=runCFL,usingSSPRKNewton=usingSSPRKNewton)
        self.limiter=None; self.uLimDof = None
        self.limiterType = limiterType
        if self.limiterType is not None:
            self.limiter = None; self.uLimDof = {}
            self.limiter=limiterType(transport.mesh,transport.nSpace_global,
                                     transport.u,
                                     transport=transport)
            for ci in range(self.nc):
                self.uLimDof[ci]=numpy.zeros((transport.u[ci].dof.shape),'d')
    def updateStage(self):
        """
        increment stage counter by 1.
        lstage here is last stage completed
        """
        if self.limiterType is not None:
            self.limiter.applySlopeLimiting(self.transport.u,
                                            self.uLimDof)
            for ci in range(self.nc):
                self.transport.u[ci].dof.flat[:] = self.uLimDof[ci].flat[:]
                assert self.transport.u[ci].par_dof is not None
                self.transport.u[ci].par_dof.scatter_forward_insert()
            #if
        #for
        #update coefficients how?
        #stage values get reset here
        self.transport.calculateCoefficients()
        #time integration loads in new residual in this call
        self.transport.calculateElementResidual()
        #limiting
        assert (self.lstage >= 0 and self.lstage <= self.nStages),"""stage= %d out of allowed bounds [0,%d]""" % (self.lstage,self.nStages)
        if self.lstage < self.nStages-1:
            self.lstage +=1
        if self.lstage < self.nStages:
            self.t = self.substeps[self.lstage]
        #end if
        #mwf debug
        #print """Linear SSPRKPI updateStage finished lstage= %d m[0]= %s""" % (self.lstage,
        #                                                              self.stageValues['m'][0][self.lstage])
        #if self.lstage < self.nStages:
        #    print """SSPRK updateStage lstage+1= %d m[0]= %s """ % (self.lstage+1,
        #                                                            self.stageValues['m'][0][self.lstage+1])

    #end updateTimeHistory
    def setFromOptions(self,nOptions):
        """
        allow classes to set various numerical parameters
        """
        LinearSSPRKintegration.setFromOptions(self,nOptions)

        if 'limiterType' in dir(nOptions) and nOptions.limiterType is not None:
            self.limiter = None; self.limiterType = None; self.uLimDof = {}
            self.limiterType = nOptions.limiterType
            self.limiter = self.limiterType(self.transport.mesh,
                                            self.transport.nSpace_global,
                                            self.transport.u,
                                            transport=transport)
            for ci in range(self.nc):
                #for limiting, in values
                self.uLimDof[ci] = numpy.zeros((self.transport.u[ci].dof.shape),'d')

            #
            self.limiter.setFromOptions(nOptions)
        #

        if 'usingSSPRKNewton' in dir(nOptions):
            self.usingDTinMass = not nOptions.usingSSPRKNewton



########################################################################
#put limiting procedures here for lack of a better place since
#they go along with SSPRKPI time integration
########################################################################
from . import FemTools
class DGlimiterP1Lagrange1d(object):
    """
    canonical (I hope) 1d DG limiting procedure when original
    local approximation space has P1 Lagrange basis
    assumes all solution fem spaces are identical
    """
    from .ctimeIntegration import applyDGlimitingP1Lagrange1d
    def __init__(self,mesh,nSpace,u,transport=None,limiterFlag=0):
        self.mesh= mesh
        self.nSpace = nSpace
        self.nc = len(u)
        self.femSpaceSolution = dict([(ci,u[ci].femSpace) for ci in range(self.nc)])
        assert self.nSpace == 1, "1d only"
        for ci in range(self.nc):
            assert isinstance(self.femSpaceSolution[ci],FemTools.DG_AffineLinearOnSimplexWithNodalBasis), "DG P1 only"

        self.l2gSolution = dict([(ci,self.femSpaceSolution[ci].dofMap.l2g) for ci in range(self.nc)]) #same for solution and limiting
        self.l2gLimiting = self.l2gSolution
        self.tag         = numpy.zeros((self.mesh.nElements_global),'i')
        self.initializeMeshInfo()
        self.limiterFlag=limiterFlag
    #
    def initializeMeshInfo(self):
        """
        """
        pass
    def setFromOptions(self,nOptions):
        """
        """
        if 'limiterFlag' in dir(nOptions):
            self.limiterFlag = nOptions.limiterFlag
    def applySlopeLimiting(self,uIn,uDofOut):
        """
        Apply limiting procedure directly using dofs
        """
        for ci in range(self.nc):
            self.applyDGlimitingP1Lagrange1d(self.mesh.elementNodesArray,
                                             self.mesh.elementNeighborsArray,
                                             self.mesh.nodeArray,
                                             self.mesh.elementBarycentersArray,
                                             self.l2gLimiting[ci],
                                             self.tag, #ignored
                                             uIn[ci].dof,
                                             uDofOut[ci],
                                             self.limiterFlag)
    #
#DG P1 Lagrange 1d

class DGlimiterP2Lagrange1d(object):
    """
    canonical (I hope) 1d DG limiting procedure when original
    local approximation space has P2 Lagrange basis
    """
    from .ctimeIntegration import applyDGlimitingP1Lagrange1d
    def __init__(self,mesh,nSpace,u,transport=None,limiterFlag=0):
        self.mesh= mesh
        self.nSpace = nSpace
        self.nc = len(u)
        self.femSpaceSolution = dict([(ci,u[ci].femSpace) for ci in range(self.nc)])

        assert self.nSpace == 1, "1d only"
        for ci in range(self.nc):
            assert isinstance(self.femSpaceSolution[ci],FemTools.DG_AffineQuadraticOnSimplexWithNodalBasis), "DG P2 only"

        self.femSpaceLimiting = dict([(ci,FemTools.DG_AffineLinearOnSimplexWithNodalBasis(self.mesh,nd=self.nSpace)) for ci in range(self.nc)])

        self.l2gSolution = dict([(ci,self.femSpaceSolution[ci].dofMap.l2g) for ci in range(self.nc)]) #no longer same
        self.l2gLimiting = dict([(ci,self.femSpaceLimiting[ci].dofMap.l2g) for ci in range(self.nc)])

        self.ulim        = dict([(ci,FemTools.FiniteElementFunction(self.femSpaceLimiting[ci],name="ulim")) for ci in range(self.nc)])
        self.dofout      = dict([(ci,numpy.zeros(self.ulim[ci].dof.shape,'d')) for ci in range(self.nc)])
        self.tag         = numpy.zeros((self.mesh.nElements_global),'i')
        self.initializeMeshInfo()
        self.limiterFlag=limiterFlag
    #
    def initializeMeshInfo(self):
        """

        """
        pass

    def setFromOptions(self,nOptions):
        """
        """
        if 'limiterFlag' in dir(nOptions):
            self.limiterFlag = nOptions.limiterFlag
    def projectToLimitedSpace(self,solndofs,ci=0):
        """
        project fem function from femSpaceSolution with dofs held in solndofs to
        ulim which is DG_AffineQuadraticOnSimplexWithNodalBasis

        nodal interpolant will not preserve mass necessarily
        """
        for eN in range(self.mesh.nElements_global):
            x0   = self.mesh.nodeArray[self.mesh.elementNodesArray[eN,0],0]
            x1   = self.mesh.nodeArray[self.mesh.elementNodesArray[eN,1],0]
            dx   = x1-x0; du = solndofs[self.l2gSolution[ci][eN,1]]-solndofs[self.l2gSolution[ci][eN,0]]
            uBar = (solndofs[self.l2gSolution[ci][eN,0]]+solndofs[self.l2gSolution[ci][eN,1]]+4.0*solndofs[self.l2gSolution[ci][eN,2]])/6.0 #simpson's rule
            xbar = self.mesh.elementBarycentersArray[eN,0]
            self.ulim[ci].dof[self.l2gLimiting[ci][eN,0]] = uBar + (x0-xbar)*du/dx
            self.ulim[ci].dof[self.l2gLimiting[ci][eN,1]] = uBar + (x1-xbar)*du/dx


    def projectFromLimitedSpace(self,solndofs,limdofs,tag,ci=0):
        """
        project to fem function from femSpaceSolution with dofs held in solndofs from
        ulim which is DG_AffineQuadraticOnSimplexWithNodalBasis
        Don't overwrite solution if tag == 1

        start with nodal interpolant, since this will preserve mass
        """
        for eN in range(self.mesh.nElements_global):
            if self.tag[eN] != 1:
                solndofs[self.l2gSolution[ci][eN,0]]=limdofs[self.l2gLimiting[ci][eN,0]]
                solndofs[self.l2gSolution[ci][eN,1]]=limdofs[self.l2gLimiting[ci][eN,1]]
                solndofs[self.l2gSolution[ci][eN,2]]=0.5*(limdofs[self.l2gLimiting[ci][eN,0]]+
                                                      limdofs[self.l2gLimiting[ci][eN,1]])
    def applySlopeLimiting(self,uIn,uDofOut):
        """
        Apply limiting procedure directly using dofs
        """
        #now need a projection step
        for ci in range(self.nc):
            self.projectToLimitedSpace(uIn[ci].dof,ci)

            #should be as before
            self.applyDGlimitingP1Lagrange1d(self.mesh.elementNodesArray,
                                             self.mesh.elementNeighborsArray,
                                             self.mesh.nodeArray,
                                             self.mesh.elementBarycentersArray,
                                             self.l2gLimiting[ci],
                                             self.tag,
                                             self.ulim[ci].dof,
                                             self.dofout[ci],
                                             self.limiterFlag)
        #
            uDofOut[ci].flat[:] = uIn[ci].dof.flat[:] #so that can skip projection if limiting not done
            self.projectFromLimitedSpace(uDofOut[ci],self.dofout[ci],self.tag,ci)
    #
#DG P2 Lagrange 1d

class DGlimiterPkMonomial1d(object):
    """
    canonical (I hope) 1d DG limiting procedure when original
    local approximation space has Pk monomial basis
    """
    from .ctimeIntegration import applyDGlimitingP1Lagrange1d
    def __init__(self,mesh,nSpace,u,transport=None,limiterFlag=0):
        self.mesh   = mesh
        self.nSpace = nSpace
        self.nc = len(u)
        self.limiterFlag=limiterFlag
        self.femSpaceSolution = dict([(ci,u[ci].femSpace) for ci in range(self.nc)])

        assert self.nSpace == 1, "1d only"
        #change this to be variable order next
        for ci in range(self.nc):
            assert isinstance(self.femSpaceSolution[ci],FemTools.DG_AffinePolynomialsOnSimplexWithMonomialBasis), "DG Pk only"

        self.femSpaceLimiting = dict([(ci,FemTools.DG_AffineLinearOnSimplexWithNodalBasis(self.mesh,nd=self.nSpace)) for ci in range(self.nc)])

        self.l2gSolution = dict([(ci,self.femSpaceSolution[ci].dofMap.l2g) for ci in range(self.nc)]) #no longer same
        self.l2gLimiting = dict([(ci,self.femSpaceLimiting[ci].dofMap.l2g) for ci in range(self.nc)])

        self.ulim        = dict([(ci,FemTools.FiniteElementFunction(self.femSpaceLimiting[ci],name="ulim")) for ci in range(self.nc)])
        self.dofout      = dict([(ci,numpy.zeros(self.ulim[ci].dof.shape,'d')) for ci in range(self.nc)])
        self.tag         = numpy.zeros((self.mesh.nElements_global),'i')
        ##build local mass matrix info on reference element
        from . import LinearSolvers
        quadraturePointArray = self.femSpaceSolution[0].referenceFiniteElement.interpolationConditions.quadraturePointArray
        quadratureWeights    = self.femSpaceSolution[0].referenceFiniteElement.interpolationConditions.quadrature.weights
        self.PkToP1 = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,
                                   self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim),'d')
        self.P1mass = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,
                                   self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim),'d')
        self.PkmassLU = self.femSpaceSolution[0].referenceFiniteElement.interpolationConditions.LUV #already factored

        v1 = [[v(p) for p in quadraturePointArray] for v in self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.basis]
        vk = [[v(p) for p in quadraturePointArray] for v in self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.basis]
        for i,vi in enumerate(v1):
            for j,vj in enumerate(vk):
                for vik,vjk,w in zip(vi,vj,quadratureWeights):
                    self.PkToP1[i,j] += vik*vjk*w
            #go ahead and do p1 mass too
            for j,vj in enumerate(v1):
                for vik,vjk,w in zip(vi,vj,quadratureWeights):
                    self.P1mass[i,j] += vik*vjk*w
        #
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.P1ToPk   = numpy.transpose(self.PkToP1)
        self.P1massLU = LinearSolvers.LU(self.P1mass)
        self.P1massLU.norm = l2Norm_local
        self.P1massLU.prepare()
        #hold for right hand sides
        self.uk   = numpy.zeros((self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.bk21 = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.u1   = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.b12k = numpy.zeros((self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim,),'d')

        self.initializeMeshInfo()
    #
    def initializeMeshInfo(self):
        """

        """
        pass

    def setFromOptions(self,nOptions):
        """
        """
        if 'limiterFlag' in dir(nOptions):
            self.limiterFlag = nOptions.limiterFlag
    def projectToLimitedSpace(self,solndofs,ci=0):
        """
        project fem function from femSpaceSolution with dofs held in solndofs to
        ulim which is DG_LinearQuadraticOnSimplexWithNodalBasis

        try more or less generic L2 projection on reference element
        """

        for eN in range(self.mesh.nElements_global):
            for k in range(self.l2gSolution[ci].shape[1]):
                self.uk[k]=solndofs[self.l2gSolution[ci][eN,k]]
            self.bk21 = numpy.dot(self.PkToP1,self.uk)
            self.P1massLU.solve(self.u1,b=self.bk21)
            for k in range(self.l2gLimiting[ci].shape[1]):
                self.ulim[ci].dof[self.l2gLimiting[ci][eN,k]]=self.u1[k]
        #eN
        #mwf debug
        #import pdb
        #pdb.set_trace()
    def projectFromLimitedSpace(self,solndofs,limdofs,tag,ci):
        """
        project to fem function from femSpaceSolution with dofs held in solndofs from
        ulim which is DG_AffineLinearOnSimplexWithNodalBasis
        Don't overwrite solution if tag == 1

        """
        for eN in range(self.mesh.nElements_global):
            if tag[eN] != 1: #assume solndofs set by default
                for k in range(self.l2gLimiting[ci].shape[1]):
                    self.u1[k]=limdofs[self.l2gLimiting[ci][eN,k]]
                self.b12k = numpy.dot(self.P1ToPk,self.u1)
                self.PkmassLU.solve(self.uk,b=self.b12k)
                for k in range(self.l2gSolution[ci].shape[1]):
                    solndofs[self.l2gSolution[ci][eN,k]]=self.uk[k]
        #eN
        #mwf debug
        #import pdb
        #pdb.set_trace()
    def applySlopeLimiting(self,uIn,uDofOut):
        """
        Apply limiting procedure directly using dofs
        """
        for ci in range(self.nc):
            #now need a projection step
            self.projectToLimitedSpace(uIn[ci].dof,ci)

            #should be as before
            self.applyDGlimitingP1Lagrange1d(self.mesh.elementNodesArray,
                                             self.mesh.elementNeighborsArray,
                                             self.mesh.nodeArray,
                                             self.mesh.elementBarycentersArray,
                                             self.l2gLimiting[ci],
                                             self.tag,
                                             self.ulim[ci].dof,
                                             self.dofout[ci],
                                             self.limiterFlag)
            #
            uDofOut[ci].flat[:] = uIn[ci].dof.flat[:] #so that can skip projection if limiting not done
            self.projectFromLimitedSpace(uDofOut[ci],self.dofout[ci],self.tag,ci)
    #
#DG Pk Lagrange 1d

########## 2d limiters ##########
class UnstructuredLimiter_base(object):
    from .ctimeIntegration import computeElementNeighborShapeGradients
    def __init__(self,mesh,nSpace):
        self.mesh  = mesh
        self.nSpace= nSpace
        self.useC  = True
        self.initializeMeshInfo(verbose=0)
    def initializeMeshInfo(self,verbose=0):
        # """elementNeighborShapeGradients stores local gradients for simpleces
        # formed from an element's barycenter and neighboring element
        # baryceners elementNeighborShapeGradients[eN,i] <--- local
        # simplex formed from (\bar{\vec x}_{eN},\bar{\vec
        # x}^i_{eN},\bar{\vec x}^{i+1}_{eN}) where i goes over local
        # numbering of faces, \bar{\vec x}^{i}_{eN} is the barycenter of
        # the neighbor across from face i.  Local numbering for the
        # neighboring simplex is always in that order

        # """
        self.elementNeighborShapeGradients = numpy.zeros((self.mesh.nElements_global,
                                                            self.mesh.nElementBoundaries_element,
                                                            self.mesh.nElementBoundaries_element,
                                                            self.nSpace),'d')

        self.computeElementNeighborShapeGradients(self.mesh.elementBoundariesArray,
                                                           self.mesh.elementNeighborsArray,
                                                           self.mesh.elementBarycentersArray,
                                                           self.mesh.elementBoundaryBarycentersArray,
                                                           self.elementNeighborShapeGradients)

        if verbose > 5:
            #simple test to see if recover a linear function
            uIn = numpy.zeros(self.mesh.nNodes_global,'d')
            for nN in range(self.mesh.nNodes_global):
                uIn[nN] = 3.0*self.mesh.nodeArray[nN][0]+2.0*self.mesh.nodeArray[nN][1]
            #
            #uBar= numpy.zeros(self.mesh.nElements_global,'d') #element barycenter interpolant
            #for eN in range(self.mesh.nElements_global):
            #    for nN in range(self.mesh.nNodes_element):
            #        uBar[eN] += uIn[self.mesh.elementNodesArray[eN,nN]]
            #    uBar[eN] /= float(self.nNodes_element)
            #
            xbar = numpy.zeros((self.mesh.nElementBoundaries_element,3),'d')
            ubar= numpy.zeros((self.mesh.nElementBoundaries_element,),'d')
            gradU=numpy.zeros((self.nSpace,),'d')
            #mwf debug
            #import pdb
            #pdb.set_trace()
            for eN in range(self.mesh.nElements_global):
                #store barycenter for either neigboring element or element boundary
                for ebN in range(self.mesh.nElementBoundaries_element):
                    if self.mesh.elementNeighborsArray[eN,ebN] < 0:
                        xbar[ebN].flat[:] = self.mesh.elementBoundaryBarycentersArray[self.mesh.elementBoundariesArray[eN,ebN]].flat
                    else:
                        xbar[ebN].flat[:] = self.mesh.elementBarycentersArray[self.mesh.elementNeighborsArray[eN,ebN]].flat
                    ubar[ebN] = 3.0*xbar[ebN][0]+2.0*xbar[ebN][1]
                #interpolant at local barycenters for simpleces
                uEn = 3.0*self.mesh.elementBarycentersArray[eN][0] + 2.0*self.mesh.elementBarycentersArray[eN][1]
                #test extrapolation to each node of triangle
                for nN in range(self.mesh.nNodes_element):
                    x = self.mesh.nodeArray[self.mesh.elementNodesArray[eN,nN],:]
                    #try each local interpolant
                    for ebN in range(self.mesh.nElementBoundaries_element):
                        ebN_1 = int(fmod(ebN+1,self.mesh.nElementBoundaries_element))
                        xbar_ebN = (self.mesh.elementBarycentersArray[eN].flat+xbar[ebN].flat+xbar[ebN_1].flat)/3.0
                        gradU.flat[:] = uEn*self.elementNeighborShapeGradients[eN,ebN,0,:]+ubar[ebN]*self.elementNeighborShapeGradients[eN,ebN,1,:]\
                                        +ubar[ebN_1]*self.elementNeighborShapeGradients[eN,ebN,2,:]
                        uout = (uEn+ubar[ebN]+ubar[ebN_1])/3.0 + gradU[0]*(x[0]-xbar_ebN[0])+gradU[1]*(x[1]-xbar_ebN[1])
                        uex  = 3.0*x[0] + 2.0*x[1]
                        assert abs(uout-uex) < 1.0e-4, "mistake eN=%d nN=%d ebN=%d uout=%s uex=%g " % (eN,nN,ebN,uout,uex)
                    #ebN
                #nN
            #eN
        #end verbose
    #initialize mesh info
    def setFromOptions(self,nOptions):
        """
        """
        pass
#

class CockburnNotesLimiter2d_base(UnstructuredLimiter_base):
    from .ctimeIntegration import computeCockburnDGlimiterArrays2d
    def __init__(self,mesh,nSpace):
        UnstructuredLimiter_base.__init__(self,mesh,nSpace)
        self.useC = True
        self.computeAlphaCoefs(verbose=0)
    def computeAlphaCoefs(self,verbose=0):
        """
        loop through each element and element face midpoint, compute barycentric coordinates
        for face midpoint using triangles formed by neighboring barycenters
        accept first nonnegative pair of coordinates

        write \\lambda_i = 1 + (x-x_i).grad \\lambda_i
        """
        self.alphas = numpy.ones((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,
                                  self.nSpace),'d')
        self.alphas.fill(-1.0)
        self.alphaNeighbors = numpy.ones((self.mesh.nElements_global,self.mesh.nElementBoundaries_element,
                                               self.nSpace),'i')
        self.alphaNeighbors.fill(-1)
        if self.useC == True:
            self.computeCockburnDGlimiterArrays2d(self.mesh.elementBoundariesArray,
                                                           self.mesh.elementNeighborsArray,
                                                           self.mesh.elementBarycentersArray,
                                                           self.mesh.elementBoundaryBarycentersArray,
                                                           self.elementNeighborShapeGradients,
                                                           self.alphas,
                                                           self.alphaNeighbors)
        else:
            lam = numpy.array([0.0,0.0]); lamn = numpy.array([-1,-1],'i')
            for eN in range(self.mesh.nElements_global):
                #check to see element has any face
                #on boundary, if so set alpha's to zero to force zero slope
                #for now
                onBoundary = min(self.mesh.elementNeighborsArray[eN,:].flat) < 0
                for ebN in range(self.mesh.nElementBoundaries_element):
                    ebN_global = self.mesh.elementBoundariesArray[eN,ebN]
                    #2d
                    xm = self.mesh.elementBoundaryBarycentersArray[ebN_global]
                    if onBoundary:
                        lam.fill(0.0)
                        lamn.fill(-1)
                        self.alphas[eN,ebN,0]=lam[0]; self.alphaNeighbors[eN,ebN,0]=lamn[0];
                        self.alphas[eN,ebN,1]=lam[1]; self.alphaNeighbors[eN,ebN,1]=lamn[1];

                    else:
                        #positive alpha's
                        posFound = False; i = 0;
                        while i < self.mesh.nElementBoundaries_element and not posFound:
                            lam.fill(-1.0); lamn.fill(-1)
                            #in local neighbor simplex ordering
                            #  0 is always this element
                            #  1 is element across from current face (say i)
                            #  2 is i+1
                            eN_i   = self.mesh.elementNeighborsArray[eN,i]
                            iplus1 = int(fmod(i+1,self.mesh.nElementBoundaries_element))
                            eN_iplus1 = self.mesh.elementNeighborsArray[eN,iplus1]
                            lamn[0]= eN_i; lamn[1]=eN_iplus1
                            lam[0] = 1.0 + numpy.innerproduct(self.elementNeighborShapeGradients[eN,i,1,:],
                                                                xm[0:2]-self.mesh.elementBarycentersArray[eN_i][0:2])
                            lam[1] = 1.0 + numpy.innerproduct(self.elementNeighborShapeGradients[eN,i,2,:],
                                                                xm[0:2]-self.mesh.elementBarycentersArray[eN_iplus1][0:2])
#                             print "eN=%d ebN=%d xm=%s i=%d i+1=%d eN_i=%d eN_i+1=%d lam=%s lamn=%s " % (eN,ebN,xm,
#                                                                                                         i,iplus1,eN_i,
#                                                                                                         eN_iplus1,lam,lamn)
                            #j
                            if min(lam) >= -1.0e-10:
                                posFound = True
                                self.alphas[eN,ebN,0]=max(lam[0],0.0); self.alphaNeighbors[eN,ebN,0]=lamn[0];
                                self.alphas[eN,ebN,1]=max(lam[1],0.0); self.alphaNeighbors[eN,ebN,1]=lamn[1];
#                                 print  "posFound=%s eN=%s ebN=%s \nxm=%s i=%s \nalphas=\n%s \nalphaNeighbors=\n%s " % (posFound,eN,ebN,xm,i,
#                                                                                                                    self.alphas[eN,ebN],
#                                                                                                                    self.alphaNeighbors[eN,ebN])
                            #if
                            i += 1
                        #i
                        assert posFound, "alpha failed eN=%s ebN=%s \nxm=%s \nelemNeigs=\n%s \nelemNeigGrads=\n%s " % (eN,ebN,xm,
                                                                                                                       self.mesh.elementNeighorsArray[eN],
                                                                                                                       self.elementNeighborShapeGradients[eN])

                #ebN
            #eN
        #end construct in python
        #do some checking
        if Profiling.logLevel > 5:
            for eN in range(self.mesh.nElements_global):
                for ebN in range(self.mesh.nElementBoundaries_element):
                    ebN_global = self.mesh.elementBoundariesArray[eN,ebN]
                    eN1= self.alphaNeighbors[eN,ebN,0]; eN2 = self.alphaNeighbors[eN,ebN,1]
                    if eN1 >= self.mesh.nElements_global or eN2 >= self.mesh.nElements_global:
                        import pdb
                        pdb.set_trace()
                    if eN1 >= 0 and eN2 >= 0:
                        xa = self.alphas[eN,ebN,0]*self.mesh.elementBarycentersArray[eN1]+self.alphas[eN,ebN,1]*self.mesh.elementBarycentersArray[eN2] + (1.0-self.alphas[eN,ebN,0]-self.alphas[eN,ebN,1])*self.mesh.elementBarycentersArray[eN]
                        xm = self.mesh.elementBoundaryBarycentersArray[ebN_global,:]
                        assert abs(numpy.sqrt(numpy.inner(xm-xa,xm-xa))) <= 1.0e-4, "alpha prob eN=%s ebN=%s ebn_global=%s xa=%s xm=%s " % (eN,ebN,ebN_global,xa,xm)

        #mwf hack force upwind
        #print "WARNING testStuff DGlimiterP11d setting alphas to zero "
        #self.alphas.flat[:] = 0.0
#mwf debug
from . import Comm
class DGlimiterP1Lagrange2d(CockburnNotesLimiter2d_base):
    """
    apply standard, Cockburn RKDG limiter in 2d (maybe)
    """
    from .ctimeIntegration import applyCockburnDGlimiterP1Lagrange2d
    def __init__(self,mesh,nSpace,u,transport=None,nu=1.5,M=0.0):
        CockburnNotesLimiter2d_base.__init__(self,mesh,nSpace)
        self.nc = len(u)
        self.femSpaceSolution= dict([(ci,u[ci].femSpace) for ci in range(self.nc)])

        assert self.nSpace == 2, "2d only"
        for ci in range(self.nc):
            assert isinstance(self.femSpaceSolution[ci],FemTools.DG_AffineLinearOnSimplexWithNodalBasis), "DG P1 only"
        self.limiter=lambda a,b : mminmod2(a,b,fct=M*mesh.h**2) #minmod
        self.nu=nu
        self.Mfact = M*mesh.h**2
        self.l2gSolution = dict([(ci,self.femSpaceSolution[ci].dofMap.l2g) for ci in range(self.nc)]) #same for solution and limiting
        self.l2gLimiting = self.l2gSolution
        self.tag         = numpy.zeros((self.mesh.nElements_global),'i')
        self.useC = True

    def applySlopeLimiting(self,uIn,uDofOut):
        for ci in range(self.nc):
            if self.useC == True:
                self.applyCockburnDGlimiterP1Lagrange2d(self.nu,self.Mfact,
                                                        self.mesh.elementNeighborsArray,
                                                        self.l2gLimiting[ci],
                                                        self.tag,
                                                        self.alphas,
                                                        self.alphaNeighbors,
                                                        uIn[ci].dof,
                                                        uDofOut[ci])
            else:
                uDofIn = uIn[ci].dof
                l2g = self.l2gLimiting[ci] #same as solution
                deltaU  = numpy.zeros((self.mesh.nElementBoundaries_element),'d')
                tdeltaU = numpy.zeros((self.mesh.nElementBoundaries_element),'d')
                tag_eN    = numpy.zeros((self.mesh.nElementBoundaries_element),'i')
                uMlim   = numpy.zeros((self.mesh.nElementBoundaries_element),'d')
                uM      = numpy.zeros((self.mesh.nElementBoundaries_element),'d')
                #for debugging
                uavIn = 0.0; uavgOut = 0.0
                for eN in range(self.mesh.nElements_global):
                    uBar = (uDofIn[l2g[eN,0]]+uDofIn[l2g[eN,1]]+uDofIn[l2g[eN,2]])/3.0 #P1 2d
                    uavgIn = uBar
                    deltaU.fill(0.0)
                    tdeltaU.fill(0.0)
                    onBoundary = min(self.mesh.elementNeighborsArray[eN].flat) < 0
                    if not onBoundary:
                        for i in range(self.mesh.nElementBoundaries_element): #local faces
                            #nodes on this face
                            ip1= int(fmod(i+1,self.mesh.nElementBoundaries_element))
                            ip2= int(fmod(i+2,self.mesh.nElementBoundaries_element))
                            uM[i] = 0.5*(uDofIn[l2g[eN,ip1]]+uDofIn[l2g[eN,ip2]])
                            eN1   = self.alphaNeighbors[eN,i,0]; eN2 = self.alphaNeighbors[eN,i,1]
                            uBar1 = (uDofIn[l2g[eN1,0]]+uDofIn[l2g[eN1,1]]+uDofIn[l2g[eN1,2]])/3.0 #P1 2d
                            uBar2 = (uDofIn[l2g[eN2,0]]+uDofIn[l2g[eN2,1]]+uDofIn[l2g[eN2,2]])/3.0 #P1 2d
                            dUi = self.alphas[eN,i,0]*(uBar1-uBar)+self.alphas[eN,i,1]*(uBar2-uBar)
                            deltaU[i],tag_eN[i] = self.limiter(uM[i]-uBar,self.nu*dUi)
                            tdeltaU[i]  = deltaU[i]
                        #i
                    #on boundary
                    sumUi = numpy.sum(deltaU)
                    if abs(sumUi) > 1.0e-6:#need nondim tol
                        posi = [max(0.0,deltaU[i]) for i in range(self.mesh.nElementBoundaries_element)]
                        negi = [max(0.0,-deltaU[i]) for i in range(self.mesh.nElementBoundaries_element)]
                        pos = sum(posi); neg = sum(negi)
                        thp= min(1.,neg/(pos+1.0e-8)); thm = min(1.0,pos/(neg+1.0e-8))
                        for i in range(self.mesh.nElementBoundaries_element):
                            tdeltaU[i] = thp*posi[i] - thm*negi[i]
                            tag_eN[i]=1
                            #assert abs(tdeltaU[i]) < 1.0e-4, "found nonzero slope, = %s" % tdeltaU[i]
                    #limiting
                    #limited midpoints of element boundaries
                    for i in range(self.mesh.nElementBoundaries_element):
                        uMlim[i] = uBar + tdeltaU[i] #phi_i(x^m_j) = kronecker delta
                    #recover vertex points, using phi_i = 1 - 2\lambda_i
                    for i in range(self.mesh.nElementBoundaries_element):
                        ip1= int(fmod(i+1,self.mesh.nElementBoundaries_element))
                        ip2= int(fmod(i+2,self.mesh.nElementBoundaries_element))
                        uDofOut[ci][l2g[eN,i]] = uMlim[ip1]+uMlim[ip2]-uMlim[i]
                    #i
                    #for debugging
                    uavgOut = (uDofOut[ci][l2g[eN,0]]+uDofOut[ci][l2g[eN,1]]+uDofOut[ci][l2g[eN,2]])/3.0
                    assert abs(uavgOut-uavgIn) < 1.0e-6, "eN=%d uavgOut=%s uavgIn=%s" % (eN,uavgOut,uavgIn)
                #eN
        #end else
    #
    def setFromOptions(self,nOptions):
        if 'limiter_nu' in dir(nOptions):
            self.nu = nOptions.limiter_nu
        if 'limiter_M' in dir(nOptions):
            self.Mfact = nOptions.limiter_M*self.mesh.h**2
class DGlimiterP2Lagrange2d(CockburnNotesLimiter2d_base):
    """
    canonical (I hope) 2d DG limiting procedure when original
    local approximation space has P2 Lagrange basis
    go ahead and use a generic L2 projection though

    TODO
      move projection steps to c
    """
    from .ctimeIntegration import applyCockburnDGlimiterP1Lagrange2d
    def __init__(self,mesh,nSpace,u,transport=None,nu=1.5,M=0.0):
        CockburnNotesLimiter2d_base.__init__(self,mesh,nSpace)
        self.nc = len(u)
        self.nu=nu
        self.Mfact = M*mesh.h**2
        self.femSpaceSolution = dict([(ci,u[ci].femSpace) for ci in range(self.nc)])


        assert self.nSpace == 2, "2d only"
        #change this to be variable order next
        for ci in range(self.nc):
            assert isinstance(self.femSpaceSolution[ci],FemTools.DG_AffineQuadraticOnSimplexWithNodalBasis), "DG P2 only"

        self.femSpaceLimiting = dict([(ci,FemTools.DG_AffineLinearOnSimplexWithNodalBasis(self.mesh,nd=self.nSpace)) for ci in range(self.nc)])

        self.l2gSolution = dict([(ci,self.femSpaceSolution[ci].dofMap.l2g) for ci in range(self.nc)]) #no longer same
        self.l2gLimiting = dict([(ci,self.femSpaceLimiting[ci].dofMap.l2g) for ci in range(self.nc)])


        self.ulim        = dict([(ci,FemTools.FiniteElementFunction(self.femSpaceLimiting[ci],name="ulim")) for ci in range(self.nc)])
        self.dofout      = dict([(ci,numpy.zeros(self.ulim[ci].dof.shape,'d')) for ci in range(self.nc)])
        self.tag         = numpy.zeros((self.mesh.nElements_global),'i')
        ##build local mass matrix info on reference element
        from . import Quadrature
        quadrature           =Quadrature.SimplexGaussQuadrature(self.nSpace,4)
        quadraturePointArray = numpy.zeros((len(quadrature.weights),3),'d')
        for k,p in enumerate(quadrature.points):
            for I in range(self.nSpace):
                quadraturePointArray[k,I]=p[I]
        quadratureWeights  = quadrature.weights
        from . import LinearSolvers
        self.P2ToP1 = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,
                                     self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim),'d')
        self.P1mass = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,
                                     self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim),'d')
        self.P2mass = numpy.zeros((self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim,
                                     self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim),'d')
        v1 = [[v(p) for p in quadraturePointArray] for v in self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.basis]
        v2 = [[v(p) for p in quadraturePointArray] for v in self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.basis]
        for i,vi in enumerate(v1):
            for j,vj in enumerate(v2):
                for vik,vjk,w in zip(vi,vj,quadratureWeights):
                    self.P2ToP1[i,j] += vik*vjk*w
            #go ahead and do p1 mass too
            for j,vj in enumerate(v1):
                for vik,vjk,w in zip(vi,vj,quadratureWeights):
                    self.P1mass[i,j] += vik*vjk*w
        #now do p2 mass
        for i,vi in enumerate(v2):
            for j,vj in enumerate(v2):
                for vik,vjk,w  in zip(vi,vj,quadratureWeights):
                    self.P2mass[i,j] += vik*vjk*w
        #
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.P1ToP2   = numpy.transpose(self.P2ToP1)
        self.P1massLU = LinearSolvers.LU(self.P1mass)
        self.P1massLU.norm = l2Norm_local
        self.P1massLU.prepare()
        self.P2massLU = LinearSolvers.LU(self.P2mass)
        self.P2massLU.norm = l2Norm_local
        self.P2massLU.prepare()
        #hold for right hand sides
        self.u2   = numpy.zeros((self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.b221 = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.u1   = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.b122 = numpy.zeros((self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim,),'d')

    def projectToLimitedSpace(self,solndofs,ci=0):
        """
        project fem function from femSpaceSolution with dofs held in solndofs to
        ulim which is DG_AffineLinearOnSimplexWithNodalBasis

        try more or less generic L2 projection on reference element
        """
        #mwf debug
        #comm = Comm.get()
        #print "LimiterP2Lagrange2d entering projectToLimitedSpace rank %s " % comm.rank()
        for eN in range(self.mesh.nElements_global):
            for k in range(self.l2gSolution[ci].shape[1]):
                self.u2[k]=solndofs[self.l2gSolution[ci][eN,k]]
            self.b221 = numpy.dot(self.P2ToP1,self.u2)
            self.P1massLU.solve(self.u1,b=self.b221)
            for k in range(self.l2gLimiting[ci].shape[1]):
                self.ulim[ci].dof[self.l2gLimiting[ci][eN,k]]=self.u1[k]
        #eN
        #mwf debug
        #import pdb
        #pdb.set_trace()
        #mwf debug
        #print "LimiterP2Lagrange2d leaving projectToLimitedSpacee rank %s " % comm.rank()
    def projectFromLimitedSpace(self,solndofs,limdofs,tag,ci=0):
        """
        project to fem function from femSpaceSolution with dofs held in solndofs from
        ulim which is DG_AffineLinearOnSimplexWithNodalBasis
        Don't overwrite solution if tag == 1

        """
        #mwf debug
        #comm = Comm.get()
        #print "LimiterP2Lagrange2d entering projectFromLimitedSpace rank %s " % (comm.rank())
        for eN in range(self.mesh.nElements_global):
            #if comm.rank() == 0 and False:
            #    print "rank %s eN= %s/%s tag= %s " % (comm.rank(),eN,self.mesh.nElements_global,tag[eN])
            if tag[eN] != 1: #assume solndofs set by default
                #if comm.rank() == 0 and False:
                #    print "rank %s eN= %s/%s l2gLimiting= %s " % (comm.rank(),eN,self.mesh.nElements_global,self.l2gLimiting[ci][eN])
                for k in range(self.l2gLimiting[ci].shape[1]):
                    self.u1[k]=limdofs[self.l2gLimiting[ci][eN,k]]
                #if comm.rank() == 0 and False:
                #    print "rank %s eN= %s/%s u1= %s " % (comm.rank(),eN,self.mesh.nElements_global,self.u1)
                self.b122 = numpy.dot(self.P1ToP2,self.u1)
                #if comm.rank() == 0 and False:
                #    print "rank %s eN= %s/%s b122= %s \n P2Mass=%s " % (comm.rank(),eN,self.mesh.nElements_global,self.b122,self.P2mass)
                self.P2massLU.solve(self.u2,b=self.b122)
                #if comm.rank() == 0 and False:
                #    print "rank %s eN= %s/%s u1= %s b122= %s u2= %s " % (comm.rank(),eN,self.mesh.nElements_global,self.u1,self.b122,self.u2)
                for k in range(self.l2gSolution[ci].shape[1]):
                    solndofs[self.l2gSolution[ci][eN,k]]=self.u2[k]
        #mwf debug
        #print "LimiterP2Lagrange2d leaving projectFromLimitedSpace %s " % comm.rank()
        #comm.barrier()
        #eN
        #mwf debug
        #import pdb
        #pdb.set_trace()
    def applySlopeLimiting(self,uIn,uDofOut):
        """
        Apply limiting procedure directly using dofs
        """
        #now need a projection step
        for ci in range(self.nc):
            self.projectToLimitedSpace(uIn[ci].dof,ci)


            self.applyCockburnDGlimiterP1Lagrange2d(self.nu,self.Mfact,
                                                    self.mesh.elementNeighborsArray,
                                                    self.l2gLimiting[ci],
                                                    self.tag,
                                                    self.alphas,
                                                    self.alphaNeighbors,
                                                    self.ulim[ci].dof,
                                                    self.dofout[ci])
        #
            uDofOut[ci].flat[:] = uIn[ci].dof.flat[:] #so that can skip projection if limiting not done
            self.projectFromLimitedSpace(uDofOut[ci],self.dofout[ci],self.tag,ci)
    #
    def setFromOptions(self,nOptions):
        if 'limiter_nu' in dir(nOptions):
            self.nu = nOptions.limiter_nu
        if 'limiter_M' in dir(nOptions):
            self.Mfact = nOptions.limiter_M*self.mesh.h**2
#DG Pk Lagrange 2d

class DGlimiterPkMonomial2d(CockburnNotesLimiter2d_base):
    """
    canonical (I hope) 2d DG limiting procedure when original
    local approximation space has Pk monomial basis
    """
    from .ctimeIntegration import applyCockburnDGlimiterP1Lagrange2d
    def __init__(self,mesh,nSpace,u,transport=None,nu=1.5,M=0.0):
        CockburnNotesLimiter2d_base.__init__(self,mesh,nSpace)
        self.nc = len(u)
        self.nu=nu
        self.Mfact = M*mesh.h**2
        self.femSpaceSolution = dict([(ci,u[ci].femSpace) for ci in range(self.nc)])

        assert self.nSpace == 2, "2d only"
        #change this to be variable order next
        for ci in range(self.nc):
            assert isinstance(self.femSpaceSolution[ci],FemTools.DG_AffinePolynomialsOnSimplexWithMonomialBasis), "DG Pk only"

        self.femSpaceLimiting = dict([(ci,FemTools.DG_AffineLinearOnSimplexWithNodalBasis(self.mesh,nd=self.nSpace)) for ci in range(self.nc)])

        self.l2gSolution = dict([(ci,self.femSpaceSolution[ci].dofMap.l2g) for ci in range(self.nc)]) #no longer same
        self.l2gLimiting = dict([(ci,self.femSpaceLimiting[ci].dofMap.l2g) for ci in range(self.nc)])

        self.ulim        = dict([(ci,FemTools.FiniteElementFunction(self.femSpaceLimiting[ci],name="ulim")) for ci in range(self.nc)])
        self.dofout      = dict([(ci,numpy.zeros(self.ulim[ci].dof.shape,'d')) for ci in range(self.nc)])
        self.tag                     = numpy.zeros((self.mesh.nElements_global),'i')
        ##build local mass matrix info on reference element
        from . import LinearSolvers
        quadraturePointArray = self.femSpaceSolution[0].referenceFiniteElement.interpolationConditions.quadraturePointArray
        quadratureWeights    = self.femSpaceSolution[0].referenceFiniteElement.interpolationConditions.quadrature.weights
        self.PkToP1 = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,
                                     self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim),'d')
        self.P1mass = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,
                                     self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim),'d')
        self.PkmassLU = self.femSpaceSolution[0].referenceFiniteElement.interpolationConditions.LUV #already factored

        v1 = [[v(p) for p in quadraturePointArray] for v in self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.basis]
        vk = [[v(p) for p in quadraturePointArray] for v in self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.basis]
        for i,vi in enumerate(v1):
            for j,vj in enumerate(vk):
                for vik,vjk,w in zip(vi,vj,quadratureWeights):
                    self.PkToP1[i,j] += vik*vjk*w
            #go ahead and do p1 mass too
            for j,vj in enumerate(v1):
                for vik,vjk,w in zip(vi,vj,quadratureWeights):
                    self.P1mass[i,j] += vik*vjk*w
        #
        #mwf debug
        #import pdb
        #pdb.set_trace()
        self.P1ToPk   = numpy.transpose(self.PkToP1)
        self.P1massLU = LinearSolvers.LU(self.P1mass)
        self.P1massLU.norm = l2Norm_local
        self.P1massLU.prepare()
        #hold for right hand sides
        self.uk   = numpy.zeros((self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.bk21 = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.u1   = numpy.zeros((self.femSpaceLimiting[0].referenceFiniteElement.localFunctionSpace.dim,),'d')
        self.b12k = numpy.zeros((self.femSpaceSolution[0].referenceFiniteElement.localFunctionSpace.dim,),'d')

    def projectToLimitedSpace(self,solndofs,ci=0):
        """
        project fem function from femSpaceSolution with dofs held in solndofs to
        ulim which is DG_AffineQuadraticOnSimplexWithNodalBasis

        try more or less generic L2 projection on reference element
        """

        for eN in range(self.mesh.nElements_global):
            for k in range(self.l2gSolution[ci].shape[1]):
                self.uk[k]=solndofs[self.l2gSolution[ci][eN,k]]
            self.bk21 = numpy.dot(self.PkToP1,self.uk)
            self.P1massLU.solve(self.u1,b=self.bk21)
            for k in range(self.l2gLimiting[ci].shape[1]):
                self.ulim[ci].dof[self.l2gLimiting[ci][eN,k]]=self.u1[k]
        #eN
        #mwf debug
        #import pdb
        #pdb.set_trace()
    def projectFromLimitedSpace(self,solndofs,limdofs,tag,ci=0):
        """
        project to fem function from femSpaceSolution with dofs held in solndofs from
        ulim which is DG_AffineQuadraticOnSimplexWithNodalBasis
        Don't overwrite solution if tag == 1

        start with nodal interpolant, since this will preserve mass
        """
        for eN in range(self.mesh.nElements_global):
            if tag[eN] != 1: #assume solndofs set by default
                for k in range(self.l2gLimiting[ci].shape[1]):
                    self.u1[k]=limdofs[self.l2gLimiting[ci][eN,k]]
                self.b12k = numpy.dot(self.P1ToPk,self.u1)
                self.PkmassLU.solve(self.uk,b=self.b12k)
                for k in range(self.l2gSolution[ci].shape[1]):
                    solndofs[self.l2gSolution[ci][eN,k]]=self.uk[k]
        #eN
        #mwf debug
        #import pdb
        #pdb.set_trace()
    def applySlopeLimiting(self,uIn,uDofOut):
        """
        Apply limiting procedure directly using dofs
        """
        #now need a projection step
        for ci in range(self.nc):
            self.projectToLimitedSpace(uIn[ci].dof,ci)

            self.applyCockburnDGlimiterP1Lagrange2d(self.nu,self.Mfact,
                                                    self.mesh.elementNeighborsArray,
                                                    self.l2gLimiting[ci],
                                                    self.tag,
                                                    self.alphas,
                                                    self.alphaNeighbors,
                                                    self.ulim[ci].dof,
                                                    self.dofout[ci])
        #
            uDofOut[ci].flat[:] = uIn[ci].dof.flat[:] #so that can skip projection if limiting not done
            self.projectFromLimitedSpace(uDofOut[ci],self.dofout[ci],self.tag,ci)
    #
    def setFromOptions(self,nOptions):
        if 'limiter_nu' in dir(nOptions):
            self.nu = nOptions.limiter_nu
        if 'limiter_M' in dir(nOptions):
            self.Mfact = nOptions.limiter_M*self.mesh.h**2
#DG Pk Lagrange 2d
##\todo put in DurlofskyP2 and Pk limiting
class DGlimiterDurlofskyP1Lagrange2d(UnstructuredLimiter_base):
    from .ctimeIntegration import applyDurlofskyDGlimiterP1Lagrange2d
    def __init__(self,mesh,nSpace,u,transport=None,killExtrema=1,allowMinWithUndershoot=0):
        UnstructuredLimiter_base.__init__(self,mesh,nSpace)
        self.nc = len(u)
        self.femSpaceSolution = dict([(ci,u[ci].femSpace) for ci in range(self.nc)])
        self.femSpaceLimiting = self.femSpaceSolution

        assert self.nSpace == 2, "2d only"
        for ci in range(self.nc):
            assert isinstance(self.femSpaceSolution[ci],FemTools.DG_AffineLinearOnSimplexWithNodalBasis), "DG P1 only"
        self.l2gSolution = dict([(ci,self.femSpaceSolution[ci].dofMap.l2g) for ci in range(self.nc)]) #same for solution and limiting
        self.l2gLimiting = self.l2gSolution
        self.tag         = numpy.zeros((self.mesh.nElements_global,),'i')
        self.elementAverages = numpy.zeros((self.mesh.nElements_global,),'d')
        self.useC = True
        self.killExtrema = killExtrema  #set zero slope for local extrema?
        self.allowMinWithUndershoot = allowMinWithUndershoot #allow use of minimum gradient even when causes under/overshoot?
        #compute shape function gradients so can get gradient easily
        #could do this manually as well
        from . import Quadrature
        quadrature = Quadrature.SimplexGaussQuadrature(self.nSpace,1)#only need constants
        nQuadraturePoints = len(quadrature.weights)
        assert nQuadraturePoints == 1, "want barycenter quadrature"
        quadraturePointsArray = numpy.zeros((nQuadraturePoints,3),'d')
        for k,p in enumerate(quadrature.points):
            quadraturePointsArray[k,:] = p
        #
        #get rid of these after compute grad_v
        J   = numpy.zeros((self.mesh.nElements_global,nQuadraturePoints,nSpace,nSpace),'d')
        invJ= numpy.zeros((self.mesh.nElements_global,nQuadraturePoints,nSpace,nSpace),'d')
        detJ= numpy.zeros((self.mesh.nElements_global,nQuadraturePoints),'d')
        self.femSpaceLimiting[0].elementMaps.getJacobianValues(quadraturePointsArray,
                                                               J,invJ,detJ)
        self.grad_v = numpy.zeros((self.mesh.nElements_global,nQuadraturePoints,
                                   self.femSpaceLimiting[0].max_nDOF_element,self.nSpace),
                                  'd')
        self.femSpaceLimiting[0].getBasisGradientValues(quadraturePointsArray,invJ,self.grad_v)

        #now get rid of storage for fem calculations
        del J; del invJ; del detJ

    def applySlopeLimiting(self,uIn,uDofOut):
        for ci in range(self.nc):
            self.applyDurlofskyDGlimiterP1Lagrange2d(self.killExtrema,
                                                     self.allowMinWithUndershoot,
                                                     self.mesh.elementNeighborsArray,
                                                     self.mesh.elementBoundariesArray,
                                                     self.mesh.elementNodesArray,
                                                     self.mesh.nodeArray,
                                                     self.mesh.elementBarycentersArray,
                                                     self.mesh.elementBoundaryBarycentersArray,
                                                     self.elementNeighborShapeGradients,
                                                     self.l2gLimiting[ci],
                                                     self.grad_v,
                                                     self.elementAverages,
                                                     self.tag,
                                                     uIn[ci].dof,
                                                     uDofOut[ci])
    #end apply

    def setFromOptions(self,nOptions):
        if 'limiter_killExtrema' in dir(nOptions):
            self.killExtrema = nOptions.limiter_killExtrema
        if 'limiter_allowMinWithUndershoot' in dir(nOptions):
            self.allowMinWithUndershoot = nOptions.limiter_allowMinWithUndershoot
#
class DGlimiterDurlofskyP1Lagrange3d(UnstructuredLimiter_base):
    from .ctimeIntegration import applyDurlofskyDGlimiterP1Lagrange3d
    def __init__(self,mesh,nSpace,u,transport=None,killExtrema=1,allowMinWithUndershoot=1):
        UnstructuredLimiter_base.__init__(self,mesh,nSpace)
        self.nc = len(u)
        self.femSpaceSolution = dict([(ci,u[ci].femSpace) for ci in range(self.nc)])
        self.femSpaceLimiting = self.femSpaceSolution

        assert self.nSpace == 3, "3d only"
        for cin in range(self.nc):
            assert isinstance(self.femSpaceSolution[ci],FemTools.DG_AffineLinearOnSimplexWithNodalBasis), "DG P1 only"
        self.l2gSolution = dict([(ci,self.femSpaceSolution[ci].dofMap.l2g) for ci in range(self.nc)]) #same for solution and limiting
        self.l2gLimiting = self.l2gSolution
        self.tag         = numpy.zeros((self.mesh.nElements_global,),'i')
        self.elementAverages = numpy.zeros((self.mesh.nElements_global,),'d')
        self.useC = True
        self.killExtrema = killExtrema  #set zero slope for local extrema?
        self.allowMinWithUndershoot = allowMinWithUndershoot #allow use of minimum gradient even when causes under/overshoot?
        #compute shape function gradients so can get gradient easily
        #could do this manually as well
        from . import Quadrature
        quadrature = Quadrature.SimplexGaussQuadrature(self.nSpace,1)#only need constants
        nQuadraturePoints = len(quadrature.weights)
        assert nQuadraturePoints == 1, "want barycenter quadrature"
        quadraturePointsArray = numpy.zeros((nQuadraturePoints,3),'d')
        for k,p in enumerate(quadrature.points):
            quadraturePointsArray[k,:] = p
        #
        #get rid of these after compute grad_v
        J   = numpy.zeros((self.mesh.nElements_global,nQuadraturePoints,nSpace,nSpace),'d')
        invJ= numpy.zeros((self.mesh.nElements_global,nQuadraturePoints,nSpace,nSpace),'d')
        detJ= numpy.zeros((self.mesh.nElements_global,nQuadraturePoints),'d')
        self.femSpaceLimiting[0].elementMaps.getJacobianValues(quadraturePointsArray,
                                                            J,invJ,detJ)
        self.grad_v = numpy.zeros((self.mesh.nElements_global,nQuadraturePoints,
                                   self.femSpaceLimiting[0].max_nDOF_element,self.nSpace),
                                  'd')
        self.femSpaceLimiting[0].getBasisGradientValues(quadraturePointsArray,invJ,self.grad_v)

        #now get rid of storage for fem calculations
        del J; del invJ; del detJ

    def applySlopeLimiting(self,uIn,uDofOut):
        for ci in range(self.nc):
            self.applyDurlofskyDGlimiterP1Lagrange3d(self.killExtrema,
                                                     self.allowMinWithUndershoot,
                                                     self.mesh.elementNeighborsArray,
                                                     self.mesh.elementBoundariesArray,
                                                     self.mesh.elementNodesArray,
                                                     self.mesh.nodeArray,
                                                     self.mesh.elementBarycentersArray,
                                                     self.mesh.elementBoundaryBarycentersArray,
                                                     self.elementNeighborShapeGradients,
                                                     self.l2gLimiting[ci],
                                                     self.grad_v,
                                                     self.elementAverages,
                                                     self.tag,
                                                     uIn[ci].dof,
                                                     uDofOut[ci])
    #end apply
    def setFromOptions(self,nOptions):
        if 'limiter_killExtrema' in dir(nOptions):
            self.killExtrema = nOptions.limiter_killExtrema
        if 'limiter_allowMinWithUndershoot' in dir(nOptions):
            self.allowMinWithUndershoot = nOptions.limiter_allowMinWithUndershoot

class DGlimiterP1Lagrange1d_Sw(DGlimiterP1Lagrange1d):
    """
    DGP1 version of limiting that applies minmod limiting for h < h_eps
      and more aggressive limiting elsewhere
    """
    from proteus.ctimeIntegration import applyDGlimitingP1Lagrange1d_withVacuumTol
    def __init__(self,mesh,nSpace,u,transport=None,limiterFlag=0,h_eps=1.0e-3):
        DGlimiterP1Lagrange1d.__init__(self,mesh,nSpace,u,transport=transport,limiterFlag=limiterFlag)
        self.h_eps=h_eps
        self.transport=transport
    #
    def applySlopeLimiting(self,uIn,uDofOut):
        """
        Apply limiting procedure directly using dofs using standard approach
        Then go through and eliminate negative values of water height

        for cells that have average (h_bar < h_eps)
           
        if average height is negative, then zero
           
        if both vertices are positive leave alone (could kill slope)
        
        if one of the vertices is negative choose slope so that this value is exactly zero
           
        zero discharge at this vertex

        May need to add additional step that limits discharge where h is much less than h_eps

        """
        #go ahead and limit h and hu
        for ci in range(self.nc):
            enforcePositivity = int(ci == 0)
            self.applyDGlimitingP1Lagrange1d_withVacuumTol(enforcePositivity,
                                                           self.h_eps,
                                                           self.mesh.elementNodesArray,
                                                           self.mesh.elementNeighborsArray,
                                                           self.mesh.nodeArray,
                                                           self.mesh.elementBarycentersArray,
                                                           self.l2gLimiting[ci],
                                                           self.tag, #ignored
                                                           uIn[ci].dof,
                                                           uDofOut[ci])



    def setFromOptions(self,nOptions):
        DGlimiterP1Lagrange1d.setFromOptions(self,nOptions)
        if 'limiter_h_eps' in dir(nOptions):
            self.h_eps = nOptions.limiter_h_eps


#DG P1 Lagrange 1d
class DGlimiterP2Lagrange1d_Sw(DGlimiterP2Lagrange1d):
    """
    DGP1 version of limiting that applies minmod limiting for h < h_eps
      and more aggressive limiting elsewhere

    """
    from .ctimeIntegration import applyDGlimitingP1Lagrange1d_withVacuumTol
    def __init__(self,mesh,nSpace,u,transport=None,limiterFlag=0,h_eps=1.0e-2):
        DGlimiterP2Lagrange1d.__init__(self,mesh,nSpace,u,transport=transport,limiterFlag=limiterFlag)
        self.h_eps=h_eps
        self.transport=transport
    #
    def applySlopeLimiting(self,uIn,uDofOut):
        """Apply limiting procedure directly using dofs using standard approach
        Then go through and eliminate negative values of water height

        for cells that have average (h_bar < h_eps)
           
        if average height is negative, then zero
           
        if both vertices are positive leave alone (could kill slope)
           
        if one of the vertices is negative, choose slope so that this
        value is exactly zero, zero discharge at this vertex

        May need to add additional step that limits discharge where h
        is much less than h_eps

        """
        #go ahead and limit h and hu
        for ci in range(self.nc):
            self.projectToLimitedSpace(uIn[ci].dof,ci)

            enforcePositivity = int(ci == 0)
            self.applyDGlimitingP1Lagrange1d_withVacuumTol(enforcePositivity,
                                                           self.h_eps,
                                                           self.mesh.elementNodesArray,
                                                           self.mesh.elementNeighborsArray,
                                                           self.mesh.nodeArray,
                                                           self.mesh.elementBarycentersArray,
                                                           self.l2gLimiting[ci],
                                                           self.tag, #ignored
                                                           self.ulim[ci].dof,
                                                           self.dofout[ci])


        #
            uDofOut[ci].flat[:] = uIn[ci].dof.flat[:] #so that can skip projection if limiting not done
            self.projectFromLimitedSpace(uDofOut[ci],self.dofout[ci],self.tag,ci)
    #

    def setFromOptions(self,nOptions):
        DGlimiterP2Lagrange1d.setFromOptions(self,nOptions)
        if 'limiter_h_eps' in dir(nOptions):
            self.h_eps = nOptions.limiter_h_eps

#DG P2 Lagrange 1d

class DGlimiterPkMonomial1d_Sw(DGlimiterPkMonomial1d):
    """
    DGPk version of limiting that applies minmod limiting for h < h_eps
      and more aggressive limiting elsewhere
    """
    from .ctimeIntegration import applyDGlimitingP1Lagrange1d_withVacuumTol
    def __init__(self,mesh,nSpace,u,transport=None,limiterFlag=0,h_eps=1.0e-1):
        DGlimiterPkMonomial1d.__init__(self,mesh,nSpace,u,transport=transport,limiterFlag=limiterFlag)
        self.h_eps=h_eps
        self.transport=transport
    #
    def applySlopeLimiting(self,uIn,uDofOut):
        """
        Apply limiting procedure directly using dofs
        """
        #go ahead and limit h and hu
        for ci in range(self.nc):
            self.projectToLimitedSpace(uIn[ci].dof,ci)

            enforcePositivity = int(ci == 0)
            self.applyDGlimitingP1Lagrange1d_withVacuumTol(enforcePositivity,
                                                           self.h_eps,
                                                           self.mesh.elementNodesArray,
                                                           self.mesh.elementNeighborsArray,
                                                           self.mesh.nodeArray,
                                                           self.mesh.elementBarycentersArray,
                                                           self.l2gLimiting[ci],
                                                           self.tag, #ignored
                                                           self.ulim[ci].dof,
                                                           self.dofout[ci])


        #
            uDofOut[ci].flat[:] = uIn[ci].dof.flat[:] #so that can skip projection if limiting not done
            self.projectFromLimitedSpace(uDofOut[ci],self.dofout[ci],self.tag,ci)
    #
    def setFromOptions(self,nOptions):
        DGlimiterPkMonomial1d.setFromOptions(self,nOptions)
        if 'limiter_h_eps' in dir(nOptions):
            self.h_eps = nOptions.limiter_h_eps


class DGlimiterDurlofskyP1Lagrange2d_Sw(DGlimiterDurlofskyP1Lagrange2d):
    from .ctimeIntegration import applyDurlofskyDGlimiterP1Lagrange2d_withVacuumTol
    def __init__(self,mesh,nSpace,u,transport=None,killExtrema=1,allowMinWithUndershoot=0,h_eps=2.0e-2):
        DGlimiterDurlofskyP1Lagrange2d.__init__(self,mesh,nSpace,u,transport=transport,killExtrema=killExtrema,
                                                allowMinWithUndershoot=allowMinWithUndershoot)
        self.h_eps=h_eps


    def applySlopeLimiting(self,uIn,uDofOut):
        """

        """
        for ci in range(self.nc):
            enforcePositivity = int(ci == 0)
            self.applyDurlofskyDGlimiterP1Lagrange2d_withVacuumTol(self.killExtrema,
                                                                   self.allowMinWithUndershoot,
                                                                   enforcePositivity,
                                                                   self.h_eps,
                                                                   self.mesh.elementNeighborsArray,
                                                                   self.mesh.elementBoundariesArray,
                                                                   self.mesh.elementNodesArray,
                                                                   self.mesh.nodeArray,
                                                                   self.mesh.elementBarycentersArray,
                                                                   self.mesh.elementBoundaryBarycentersArray,
                                                                   self.elementNeighborShapeGradients,
                                                                   self.l2gLimiting[ci],
                                                                   self.grad_v,
                                                                   self.elementAverages,
                                                                   self.tag,
                                                                   uIn[ci].dof,
                                                                   uDofOut[ci])
    #end apply

    def setFromOptions(self,nOptions):
        DGlimiterDurlofskyP1Lagrange2d.setFromOptions(self,nOptions)
        if 'limiter_h_eps' in dir(nOptions):
            self.h_eps = nOptions.limiter_h_eps

########################################################################
#end limiting
########################################################################

#
#cek planning on cutting these
#
class ForwardIntegrator(object):
    """
    class that is responsible for basic process of integrating a problem
    forward in time given a VectorTranport Problem, Nonlinear Solver,
    and a Time Integration method
    """
    def __init__(self,mlvtran,mlnl,dtMeth,nOptions,stepExact=False):
        """
        mlvTran  --- multilevel vector transport object for system being integrated
        mlNL     --- multilevel nonlinear solver to solve discrete system
        dtMet    --- time integration method to use
        nOptions --- configuration options
        """
        self.mlvTran   = mlvtran
        self.mlNL      = mlnl
        self.dtMeth    = dtMeth
        self.tLast     = None
        self.stepExact = stepExact
        if self.dtMeth == FLCBDF: #takes care of step exact?
            self.stepExact = False
        self.dtSET     = None
        self.tstring   = None
        self.nOptions  = nOptions
        self.firstStep = True
        #mwf where to set the parameters for the time integration
        for mi in self.mlvTran.modelList:
            mi.timeIntegration.setFromOptions(nOptions)
        self.callUpdateTimeHistory=True
    #end init
    def initialize(self,DTSET=None,t0=0.0,T=1.0):
        """

        """
        self.tLast = t0
        self.dtSET = DTSET
        if self.dtMeth == NoIntegration:
            self.dtSET = 1.0
        self.mlvTran.initializeTimeIntegration(self.dtSET,t0,T)
        self.firstStep=True
    def calculateSolution(self,tIn,tOut):
        """
        Move forward from time tIn to time tOut
        For now doesn't worry about potential mismatch between tIn and
        last time value used by model
        """
        t = tIn
        failedFlag = False
        if self.dtMeth == NoIntegration:
            logEvent("""NoIntegration, fint t=%g tOut=%g DTSET=%s DT=%g """ % (t,tOut,self.dtSET,self.mlvTran.dt))

            failedFlag=self.mlNL.solveMultilevel(uList=self.mlvTran.uList,
                                                 rList=self.mlvTran.rList,
                                                 par_uList=self.mlvTran.par_uList,
                                                 par_rList=self.mlvTran.par_rList)

            self.tLast = tOut
            t = tOut
            self.mlvTran.updateTimeHistory(t)
            logEvent("""fint t=%g tOut=%g DTSET=%s DT=%g """ % (t,tOut,self.dtSET,self.mlvTran.dt))
            self.mlvTran.choose_dt(DTSET=self.dtSET,tOut=tOut)
            #mwf debug
            #print """after solve failedFlag= %s """ % failedFlag

        else:
            if (not self.firstStep):
                self.mlvTran.choose_dt(self.dtSET,tOut)
                self.firstStep=False
            while t < tOut and failedFlag== False:
                #mwf debug
                logEvent("""\nfint t=%g tOut=%g DTSET=%s DT=%g """ % (t,tOut,self.dtSET,self.mlvTran.dt))
                #try not to step past tOut
#                 if self.stepExact and abs(t+self.mlvTran.dt-tOut) < 1.0e-10 or t+self.mlvTran.dt > tOut+1.0e-10:
#                     self.mlvTran.resetDT()
#                     self.mlvTran.choose_dt(tOut-t,tOut)
#                     #mwf debug
#                     print """mwf hack fint t=%s avoiding stepping past tOut=%s  in forward integrator, new dt=%s""" % (t,tOut,self.mlvTran.dt)

                self.writeProgress(t,self.mlvTran.dt,tOut)
                istage = 0
                while istage < self.nOptions.nStagesTime and failedFlag == False:
                    failures=0
                    maxFailures=10
                    while failures < maxFailures:
                        failedFlag= self.mlNL.solveMultilevel(uList=self.mlvTran.uList,
                                                              rList=self.mlvTran.rList,
                                                              par_uList=self.mlvTran.par_uList,
                                                              par_rList=self.mlvTran.par_rList)
                        #mwf debug
                        #what about nonlinear solver failure?
#                         print "fint failures=%s failedFlag=%s " % (failures,failedFlag)
                        if failedFlag:
                            self.mlvTran.retryStep_solverFailure()
                            failures +=1
                        else:
                            if self.mlvTran.lastStepErrorOk():
                                break
                            else:
                                self.mlvTran.retryStep_errorFailure();
                                failures += 1
                    self.mlvTran.updateStage()
                    istage += 1
                    #mwf debug
#                     print """fint t=%g after istage=%d failed=%s """ % (t,istage,failedFlag)
                #end stages
                if self.callUpdateTimeHistory:
                    t += self.mlvTran.dt
                    self.mlvTran.updateTimeHistory(t)
                    if t < tOut:
                        self.mlvTran.choose_dt(self.dtSET,tOut)
                    self.optDT = self.mlvTran.dt
#                     if self.stepExact and abs(tOut - t + self.mlvTran.dt) > 1.0e-8*abs(tOut):
#                        if t + self.mlvTran.dt != tOut:
#                            self.mlvTran.resetDT()
#                            self.mlvTran.choose_dt(tOut-t)
#                            #mwf debug
#                            print """fint adjusting DT to %g """ % self.mlvTran.dt
                else:
                    self.optDT = self.mlvTran.dt
                    #cek hack--just exit assuming this step was what we wanted to take
                    break
            #end while t < tOut
            self.tLast = t
        #end else do time integration

        #need to put in linear solver for profiling
        #Profiling.logEvent(self.mlNL.info(),level=5)

        return failedFlag,t

    #end calculateSolution
    def writeProgress(self,tn,dt,T):
        """
        just echo to screen what new and final time levels are
        """
        from . import Profiling
        if Profiling.logLevel < 2:
            eraseTime='\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'
            if self.tstring is not None:
                sys.stdout.write(eraseTime)
        sys.stdout.write('T= %12.5e, tn = ' % T)
        self.tstring = '%12.5e' % (tn+dt,)
        if Profiling.logLevel >= 2:
            self.tstring+='\n'
        sys.stdout.write(self.tstring)
        sys.stdout.flush()

class SteadyStateIntegrator(ForwardIntegrator):
    """
    apply some form of pseudo-transient continuation or direct solve to
    compute steady state solution regardless of requested time step

    """
    def __init__(self,mlvtran,mlnl,dtMeth,nOptions,ssatol=1.0e-3,ssrtol=1.0e-3,
                 maxsteps=50,stepExact=True,ignoreFailure=False):
        ForwardIntegrator.__init__(self,mlvtran,mlnl,dtMeth,nOptions,stepExact)
        self.steadyStateAtol = ssatol
        self.steadyStateRtol = ssrtol
        #mwf hack
        self.maxPTCsteps     = nOptions.maxNonlinearIts#maxsteps
        self.ignoreFailure   = ignoreFailure
        #cek debug
        #self.ignoreFailure = False
        #self.nestedPsiTC=True
        self.nestedPsiTC=False
        self.u_dof_save=[]
        for u_j in list(mlvtran.modelList[-1].u.values()):
            self.u_dof_save.append(numpy.zeros(u_j.dof.shape,'d'))
    #end init
    def initialize(self,DTSET=None,t0=0.0,T=1.0):
        """

        """
        ForwardIntegrator.initialize(self,DTSET,t0)
        if not self.dtMeth == NoIntegration:
            #force nonlinear solver to take 1 iteration
            self.mlNL.maxIts = 1
            for nl in self.mlNL.solverList:
                nl.convergenceTest = 'its'
                nl.maxIts = 1
                nl.lineSearch = False
            #end for
            #get atol and rtol from nonlinear solver
            self.steadyStateAtol= self.mlNL.solverList[-1].atol_r
            self.steadyStateRtol= self.mlNL.solverList[-1].rtol_r
            #mwf set this to self.nOptions.maxNonlinearIts
            #self.maxPTCsteps    = self.nOptions.maxNonlinearIts
            self.mlNL.tolList = None #need this force use of its test?
        else:
            self.dtSET = None #force PsiTCtte to pick own time step
        #end dtMeth
        #force number of stages to be 1
        if not (self.dtMeth == NoIntegration or self.dtMeth == PsiTCtte):
            logEvent("""WARNING SteadyStateIntegrator type= %s must be NoIntegration or
PsiTCtte!""" % self.dtMeth)
    def calculateSolution(self,tIn,tOut):
        """
        integrate to steady state regardless of what tIn and tOut are
        """
        t = tIn
        failedFlag = False
        converged = False
        for uj_dof_save,uj in zip(self.u_dof_save,list(self.mlvTran.modelList[-1].u.values())):
            uj_dof_save[:]=uj.dof
        if self.dtMeth == NoIntegration:
            self.mlvTran.choose_dt(self.dtSET,tOut)
            failedFlag=self.mlNL.solveMultilevel(uList=self.mlvTran.uList,
                                                 rList=self.mlvTran.rList,
                                                 par_uList=self.mlvTran.par_uList,
                                                 par_rList=self.mlvTran.par_rList)

            self.tLast = tOut
            t = tOut
            self.mlvTran.updateTimeHistory(t)
            converged = not failedFlag
        elif self.dtMeth == PsiTCtte and self.nestedPsiTC:
            nLevels = len(self.mlvTran.modelList)
            for solver,model,u,r,par_u,par_r,l in zip(self.mlNL.solverList,self.mlvTran.modelList,
                                                      self.mlvTran.uList,self.mlvTran.rList,
                                                      self.mlvTran.par_uList,self.mlvTran.par_rList,
                                                      list(range(nLevels))):
                model.initializeTimeIntegration(tIn,tOut,u,r,DTSET=self.dtSET)
                #until get residual figured out
                nssteps = 0; res0 = None; res = None ; ssError= 1.e20
                converged = False
                failedFlag = False
                while nssteps < self.maxPTCsteps and not converged:
                    if self.dtSET is None:
                        model.timeIntegration.choose_dt(tOut)
                    else:
                        model.timeIntegration.dt = self.dtSET
                    istage = 0
                    while istage < self.nOptions.nStagesTime and failedFlag == False:
                        failedFlag= solver.solve(u=u,r=r,par_u=par_u,par_r=par_r)
                        model.timeIntegration.updateStage()
                        istage += 1
                    #end stages
                    t += model.timeIntegration.dt
                    model.updateTimeHistory(t)
                    if nssteps == 0:
                        res0 = solver.norm_r
                    res = solver.norm_r
                    ssError = res/(res0*self.steadyStateRtol+self.steadyStateAtol)
                    converged = ssError < 1.0
                    #mwf hack
                    #converged = False
                    nssteps+= 1
                    self.writeProgress(t,model.timeIntegration.dt,res)
                    #cek debug
                    #self.mlvTran.modelList[-1].viewSolution(t)
                    #mwf debug
#                     print """\nnsteps= %d res0 = %g res= %g ssError= %g atol= %g rtol= %g not conv= %s """ % (nssteps,res0,
#                                                                                                               res,
#                                                                                                               ssError,
#                                                                                                               self.steadyStateAtol,
#                                                                                                               self.steadyStateRtol,
#                                                                                                               ssError > 1.0)
                self.tLast = tOut
                failedFlag = not converged
                if converged:
                    if l < nLevels-1:
                        self.mlvTran.meshTransfers.prolongList[l+1].matvec(u,self.mlvTran.uList[l+1])
                        self.mlvTran.modelList[l+1].setUnknowns(self.mlvTran.uList[l+1])
        else:
            self.mlvTran.initializeTimeIntegration(self.dtSET,tIn,tOut)

            #until get residual figured out
            nssteps = 0; res0 = None; res = None ; ssError= 1.e20
            converged = False
            while nssteps < self.maxPTCsteps and not converged:
                self.mlvTran.choose_dt(self.dtSET,1.0)
                istage = 0
                while istage < self.nOptions.nStagesTime and failedFlag == False:
                    failedFlag= self.mlNL.solveMultilevel(uList=self.mlvTran.uList,
                                                          rList=self.mlvTran.rList,
                                                          par_uList=self.mlvTran.par_uList,
                                                          par_rList=self.mlvTran.par_rList)
                    #print """\nSS int failedFlag = %s """ % failedFlag
                    #failedFlag=False
                    self.mlvTran.updateStage()
                    istage += 1
                #end stages
                t += self.mlvTran.dt
                self.mlvTran.updateTimeHistory(t)
                if nssteps == 0:
                    res0 = self.mlNL.solverList[-1].norm_r
                res = self.mlNL.solverList[-1].norm_r
                ssError = res/(res0*self.steadyStateRtol+self.steadyStateAtol)
                converged = ssError < 1.0
                #mwf hack
                #converged = False
                nssteps+= 1
                self.writeProgress(t,self.mlvTran.dt,res)
                #cek debug
                #self.mlvTran.modelList[-1].viewSolution(t)
                #mwf debug
#                 print """\nnsteps= %d res0 = %g res= %g ssError= %g atol= %g rtol= %g not conv= %s """ % (nssteps,res0,
#                                                                                                         res,
#                                                                                                         ssError,
#                                                                                                         self.steadyStateAtol,
#                                                                                                         self.steadyStateRtol,
#                                                                                                         ssError > 1.0)
            #end while
            self.tLast = tOut
            failedFlag = not converged
        #end else do time integration
        #put this in multilevel VectorTransport? updateTimeHistory?
        #if self.nOptions.conservativeFlux == 'pwc':
        #    self.mlvTran.modelList[-1].getConservationFluxPWC()
        #elif self.nOptions.conservativeFlux == 'pwl':
        #    self.mlvTran.modelList[-1].getConservationFluxPWL()

        #need to put in linear solver for profiling
        #Profiling.logEvent(self.mlNL.info(),level=5)
        #mwf hack
        if self.ignoreFailure and not converged:
            if failedFlag:
                logEvent("""SteadyStateIntegrator ignoredFailure t=%s """ % tOut)
            #undo any damage. this will get propagated to the other grids and u-list in the projection
            for j,uj in enumerate(self.mlvTran.modelList[-1].u.values()):
                uj.dof[:] = self.u_dof_save[j][:]
            return False,tOut
        return failedFlag,tOut

    #end calculateSolution
    def writeProgress(self,tn,dt,res):
        from . import Profiling
        """
        just echo to screen what new and final time levels are as well as residual
        """
        if Profiling.logLevel < 2:
            eraseTime='\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'
            if self.tstring is not None:
                sys.stdout.write(eraseTime)
        sys.stdout.write('ResSS= %12.5e, tn = ' % res)
        self.tstring = '%12.5e' % (tn+dt,)
        sys.stdout.write(self.tstring)
        sys.stdout.flush()

class SignedDistanceIntegrator(ForwardIntegrator):
    """
    apply some form of pseudo-transient continuation or direct solve to
    compute steady state solution regardless of requested time step

    """
    def __init__(self,mlvtran,mlnl,dtMeth,nOptions,stepExact=False,nSteps=3):
        ForwardIntegrator.__init__(self,mlvtran,mlnl,dtMeth,nOptions,stepExact)
        self.nSteps = nSteps
        self.u_dof_save=[]
        for u_j in list(mlvtran.modelList[-1].u.values()):
            self.u_dof_save.append(numpy.zeros(u_j.dof.shape,'d'))
    #end init
    def initialize(self,DTSET=None,t0=0.0,T=1.0):
        ForwardIntegrator.initialize(self,DTSET,t0)
        self.dtSET = None #force PsiTCtte to pick own time step

    def calculateSolution(self,tIn,tOut):
        """
        take a couple of steps toward steady state
        """
        failedFlag = False
        converged = False
        for uj_dof_save,uj in zip(self.u_dof_save,list(self.mlvTran.modelList[-1].u.values())):
            uj_dof_save[:]=uj.dof
        nLevels = len(self.mlvTran.modelList)
        for solver,model,u,r,par_u,par_r,l in zip(self.mlNL.solverList,self.mlvTran.modelList,
                                                  self.mlvTran.uList,self.mlvTran.rList,
                                                  self.mlvTran.par_uList,self.mlvTran.par_rList,
                                                  list(range(nLevels))):
            model.timeIntegration.runCFL = 10.0
            model.initializeTimeIntegration(0,10000,u,r,DTSET=self.dtSET)
            #until get residual figured out
            nssteps = 0; res0 = None; res = None ;
            failedFlag = False
            t=0.0
            for step in range(self.nSteps):
                failedFlag= solver.solve(u=u,r=r,par_u=par_u,par_r=par_r)
                t += model.timeIntegration.dt
                model.updateTimeHistory(t)
                model.timeIntegration.choose_dt(t+model.timeIntegration.dt)
        return failedFlag,tOut

    #end calculateSolution
    def writeProgress(self,tn,dt,res):
        from . import Profiling
        """
        just echo to screen what new and final time levels are as well as residual
        """
        if Profiling.logLevel < 2:
            eraseTime='\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'
            if self.tstring is not None:
                sys.stdout.write(eraseTime)
        sys.stdout.write('ResSS= %12.5e, tn = ' % res)
        self.tstring = '%12.5e' % (tn+dt,)
        sys.stdout.write(self.tstring)
        sys.stdout.flush()
#

#add a special FLCBDF integrator for Two-phase Darcy flow to take care of offdiagonal problems
#when have incompressible flow
#current FLCBDF doesn't handle non-diagonal nonlinearity yet
class FLCBDF_TwophaseDarcy_fc(FLCBDF):
    def __init__(self,transport):
        FLCBDF.__init__(self,transport)
    def calculateElementCoefficients(self,q):
        #w equation (aq mass balance)
        #mwf now what about if have compressibility
        ci = 0;
        for cj in [0]: #calculate S_w as beflre
            self.flcbdf[ci].calculate_yprime(q[('m',ci)],q[('dm',ci,cj)],q[('mt',ci)],q[('dmt',ci,cj)])
        self.m[ci]=q[('m',ci)]
        #current flcbdf ignores arguments in subsequent calls to calculate_yprime
        cj = 1;
        if ('dm',ci,cj) in q:
            alpha = self.flcbdf[ci].getCurrentAlpha()
            q[('dmt',ci,cj)].flat[:] = q[('dm',ci,cj)].flat
            q[('dmt',ci,cj)] *= alpha

        #n equation (napl mass balance)
        ci = 1;
        for cj in [0]: #which diagonal to use, sand getting a lot of failures either way with comp.?
            self.flcbdf[ci].calculate_yprime(q[('m',ci)],q[('dm',ci,cj)],q[('mt',ci)],q[('dmt',ci,cj)])
        self.m[ci]=q[('m',ci)]
        #
        cj = 1;
        if ('dm',ci,cj) in q:
            alpha = self.flcbdf[ci].getCurrentAlpha()
            q[('dmt',ci,cj)].flat[:] = q[('dm',ci,cj)].flat
            q[('dmt',ci,cj)] *= alpha

        for ci in range(self.nc):
            self.flcbdf[('u',ci)].calculate_yprime(self.transport.u[ci].dof,self.dummy[ci],self.Ddof_Dt[ci],self.dummy[ci])


class CentralDifference_2ndD(TI_base):
    def __init__(self,transport,integrateInterpolationPoints=False):
        TI_base.__init__(self,transport)
        self.integrateInterpolationPoints = integrateInterpolationPoints
        self.m_last={}
        self.m_tmp={}
        self.mt1_last={}
        self.mt1_tmp={}
        self.m_ip_last={}
        self.m_ip_tmp={}
        for ci in self.massComponents:
            if ('m_last',ci) in transport.q:
                self.m_last[ci] = transport.q[('m_last',ci)]
            else:
                self.m_last[ci] = numpy.array(transport.q[('m',ci)])
            if ('m_tmp',ci) in transport.q:
                self.m_tmp[ci] = transport.q[('m_tmp',ci)]
            else:
                self.m_tmp[ci] = numpy.array(transport.q[('m',ci)])
            self.mt1_last[ci] = numpy.zeros(transport.q[('m',ci)].shape,"d")
            self.mt1_tmp[ci] = numpy.zeros(transport.q[('m',ci)].shape,"d")
            if self.integrateInterpolationPoints:
                self.m_ip_last[ci] = numpy.array(transport.phi_ip[('m',ci)])
                self.m_ip_tmp[ci]  = numpy.array(transport.phi_ip[('m',ci)])
            self.massIsImplicit[ci]           = True
            self.advectionIsImplicit[ci]      = False
            self.diffusionIsImplicit[ci]      = False
            self.reactionIsImplicit[ci]       = False
            self.hamiltonianIsImplicit[ci]    = False
            self.stabilizationIsImplicit[ci]  = False
            self.shockCapturingIsImplicit[ci] = False
        #moving to generic bdf interface
        self.m_history = []; self.m_history.append({})
        self.alpha_bdf = 1.0
        self.beta_bdf  = {}
        for ci in self.massComponents:
            self.m_history[0][ci] = self.m_last[ci]
            self.beta_bdf[ci] = numpy.copy(self.m_tmp[ci])
    def calculateElementCoefficients(self,q):
        #for bdf interface
        self.calculateCoefs()
        for ci in self.massComponents:
            self.m_tmp[ci][:] = q[('m',ci)]
            self.mt1_tmp[ci][:]   = q[('m',ci)]
            self.mt1_tmp[ci] -= self.m_last[ci]
            self.mt1_tmp[ci] /= self.dt
            q[('mt',ci)][:]   = self.mt1_tmp[ci]
            q[('mt',ci)] -= self.mt1_last[ci]
            q[('mt',ci)] /= self.dt
            for cj in range(self.nc):
                if ('dmt',ci,cj) in q:
                    q[('dmt',ci,cj)][:] = q[('dm',ci,cj)]
                    q[('dmt',ci,cj)] /= self.dt**2
                if ('dm_sge',ci,cj) in q and ('dmt_sge',ci,cj) in q:
                    q[('dmt_sge',ci,cj)][:] = q[('dm_sge',ci,cj)]
                    q[('dmt_sge',ci,cj)] /= self.dt**2
            #print q[('mt',ci)]
    def calculateGeneralizedInterpolationCoefficients(self,cip):
        pass
    def initializeTimeHistory(self,resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        for ci in self.massComponents:
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat
            self.mt1_last[ci].flat[:] = self.mt1_tmp[ci].flat
    def updateTimeHistory(self,resetFromDOF=False):
        self.tLast = self.t
        for ci in self.massComponents:
            self.m_last[ci].flat[:] = self.m_tmp[ci].flat
            self.mt1_last[ci].flat[:] = self.mt1_tmp[ci].flat
    def calculateCoefs(self):
        #for bdf interface
        dtInv = 1.0/self.dt**2
        self.alpha_bdf = dtInv
        for ci in list(self.m_last.keys()):
            self.beta_bdf[ci].flat[:] = self.m_last[ci].flat
            self.beta_bdf[ci] *= -dtInv
            self.beta_bdf[ci].flat[:] -= self.mt1_last[ci].flat


## @}
