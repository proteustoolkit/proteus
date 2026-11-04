"""
Time integration, step control and iterative split-operator classes used by the 2012 gwvd decks.

Ported from erdc/proteus-gwvd@14961d2 `src/Gwvd_consolidated_classes.py`: only the classes the
continuous Galerkin decks select (`Fixed_dt_controller_systemit`, `NonConservativeBackwardEuler`,
`ISO_fixed_MinModelStep`).
"""
import proteus.StepControl
import proteus.TimeIntegration
import proteus.SplitOperator
from proteus.Comm import globalMax
from proteus.Profiling import logEvent


class Fixed_dt_controller_systemit(proteus.StepControl.Min_dt_controller):
    """Fixed model step dt = nOptions.DT, for the iterated (ISO) coupling of flow and transport."""

    def __init__(self, model, nOptions):
        proteus.StepControl.Min_dt_controller.__init__(self, model, nOptions)
        self.dt = nOptions.DT
        self.dt_model_last = None
        model.levelModelList[-1].timeIntegration.isAdaptive = True  # needed by the ISO controller

    def initialize_dt_model(self, t0, tOut):
        self.saveSolution()
        self.dt_model = self.dt
        if self.dt_model_last is None:
            self.dt_model_last = self.dt_model
        self.set_dt_allLevels()
        self.substeps = [self.t_model]
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name, self.dt_model), level=1)

    def choose_dt_model(self):
        self.solverFailures = 0
        self.errorFailures = 0
        self.saveSolution()
        self.dt_model = self.dt
        if self.dt_model_last is None:
            self.dt_model_last = self.dt_model
        self.set_dt_allLevels()
        self.setSubsteps([self.t_model])

    def stepExact_model(self, tOut):
        pass


class NonConservativeBackwardEuler(proteus.TimeIntegration.BackwardEuler):
    """Backward Euler on the primary variable: m_t = dm (u - u_last)/dt, with dm frozen at the new iterate.

    Until the time history is initialized, m_t is reported as zero. Current proteus spins up model 0
    (the flow) before the transport history exists, when u_last is still zero and dt is the default
    1.0; the flow source omega (drho/dc) c_t / rho would then see a fictitious c_t = omega rho c.
    """

    def __init__(self, transport, integrateInterpolationPoints=False):
        proteus.TimeIntegration.BackwardEuler.__init__(self, transport,
                                                       integrateInterpolationPoints=integrateInterpolationPoints)
        self.historyInitialized = False

    def initializeTimeHistory(self, resetFromDOF=True):
        proteus.TimeIntegration.BackwardEuler.initializeTimeHistory(self, resetFromDOF)
        self.historyInitialized = True

    def calculateElementCoefficients(self, q):
        self.calculateCoefs()
        if not self.historyInitialized:
            for ci in self.m_last.keys():
                self.m_tmp[ci][:] = q[('u', ci)]
                q[('mt', ci)].fill(0.0)
            return
        for ci in self.m_last.keys():
            self.m_tmp[ci][:] = q[('u', ci)]
            q[('mt', ci)][:] = q[('u', ci)]
            q[('mt', ci)] -= self.m_last[ci]
            q[('mt', ci)] /= self.dt
            q[('mt', ci)] *= q[('dm', ci, ci)]
            for cj in range(self.nc):
                if ('dmt', ci, cj) in q:
                    q[('dmt', ci, cj)][:] = q[('dm', ci, cj)]
                    q[('dmt', ci, cj)] /= self.dt
                if ('dm_sge', ci, cj) in q and ('dmt_sge', ci, cj) in q:
                    q[('dmt_sge', ci, cj)][:] = q[('dm_sge', ci, cj)]
                    q[('dmt_sge', ci, cj)] /= self.dt

    def calculateGeneralizedInterpolationCoefficients(self, cip):
        if not self.integrateInterpolationPoints:
            return
        for ci in self.m_ip_last.keys():
            self.m_ip_tmp[ci][:] = cip[('u', ci)]
            cip[('mt', ci)][:] = cip[('u', ci)]
            cip[('mt', ci)] -= self.m_ip_last[ci]
            cip[('mt', ci)] /= self.dt
            cip[('mt', ci)] *= cip[('dm', ci, ci)]
            for cj in range(self.nc):
                if ('dmt', ci, cj) in cip:
                    cip[('dmt', ci, cj)][:] = cip[('dm', ci, cj)]
                    cip[('dmt', ci, cj)] /= self.dt
                if ('dmt_sge', ci, cj) in cip:
                    cip[('dmt_sge', ci, cj)][:] = cip[('dm_sge', ci, cj)]
                    cip[('dmt_sge', ci, cj)] /= self.dt


class ISO_fixed_MinModelStep(proteus.SplitOperator.ISO_fixed_MinAdaptiveModelStep):
    """
    Iterate flow and transport within a step (Povich 2013, Sec. 4.5) until the max norm of the owned
    residuals of both models is below atol_iso, then advance.
    """

    def __init__(self, modelList, system=proteus.SplitOperator.defaultSystem, stepExact=False, atol_iso=1.e-8):
        proteus.SplitOperator.ISO_fixed_MinAdaptiveModelStep.__init__(self, modelList, system, stepExact)
        self.atol_iso = atol_iso
        self.convergence_flag = False
        self.velocity_Postprocess = False

    def setFromOptions(self, soOptions):
        self.velocity_Postprocess = False
        if 'atol_iso' in dir(soOptions):
            self.atol_iso = soOptions.atol_iso
        if 'velocity_Postprocess' in dir(soOptions):
            self.velocity_Postprocess = soOptions.velocity_Postprocess

    def converged(self):
        if self.convergence_flag and self.its != 0:
            self.its = 0
            self.convergence_flag = False
            return True
        return False

    def SysNorm(self, rSys):
        # The 2012 code takes the signed maximum of each residual (not max |r|), so negative
        # residuals never block convergence. Kept for fidelity with the published runs.
        return max(globalMax(float(r.max())) for r in rSys)

    def sequenceTaken(self):
        self.its += 1
        rSys = []
        for model in self.modelList:
            lm = model.levelModelList[-1]
            lm.getResidual(model.uList[-1], model.rList[-1])
            rSys.append(model.rList[-1][:lm.mesh.subdomainMesh.nNodes_owned])
        rNorm = self.SysNorm(rSys)
        logEvent("ISO iteration %d: max residual %12.5e (atol_iso %12.5e)" % (self.its, rNorm, self.atol_iso), level=1)
        if rNorm < self.atol_iso and self.its != 0:
            self.convergence_flag = True
        if self.convergence_flag:
            for model in self.modelList:
                model.stepController.updateTimeHistory()
                model.stepController.choose_dt_model()
