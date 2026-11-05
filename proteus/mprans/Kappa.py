import proteus
from proteus.mprans.cKappa import *
from proteus.mprans.cKappa2D import *
from proteus import Profiling as prof
import numpy as np
from proteus.Transport import OneLevelTransport
from proteus import cfemIntegrals
from . import cArgumentsDict

"""
NOTES:

Hardwired Numerics include:

lagging all terms from Navier-Stokes, Epsilon equations same solution
space for velocity from Navier-Stokes and Kappa equations

This can be removed by saving gradient calculations in N-S and lagging
rather than passing degrees of freedom between models

"""


class SubgridError(proteus.SubgridError.SGE_base):
    def __init__(self, coefficients, nd):
        proteus.SubgridError.SGE_base.__init__(self, coefficients, nd, lag=False)

    def initializeElementQuadrature(self, mesh, t, cq):
        pass

    def updateSubgridErrorHistory(self, initializationPhase=False):
        pass

    def calculateSubgridError(self, q):
        pass


class ShockCapturing(proteus.ShockCapturing.ShockCapturing_base):
    def __init__(self, coefficients, nd, shockCapturingFactor=0.25, lag=True, nStepsToDelay=None):
        proteus.ShockCapturing.ShockCapturing_base.__init__(self, coefficients, nd, shockCapturingFactor, lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0
        if self.lag:
            prof.logEvent("Kappa.ShockCapturing: lagging requested but must lag the first step; switching lagging off and delaying")
            self.nStepsToDelay = 1
            self.lag = False

    def initializeElementQuadrature(self, mesh, t, cq):
        self.mesh = mesh
        self.numDiff = []
        self.numDiff_last = []
        for ci in range(self.nc):
            self.numDiff.append(cq[('numDiff', ci, ci)])
            self.numDiff_last.append(cq[('numDiff', ci, ci)])

    def updateShockCapturingHistory(self):
        self.nSteps += 1
        if self.lag:
            for ci in range(self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            prof.logEvent("Kappa.ShockCapturing: switched to lagged shock capturing")
            self.lag = True
            self.numDiff_last = []
            for ci in range(self.nc):
                self.numDiff_last.append(self.numDiff[ci].copy())
        prof.logEvent("Kappa: max numDiff %e" % (proteus.Comm.globalMax(self.numDiff_last[0].max()),))


class NumericalFlux(proteus.NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior):
    def __init__(self, vt, getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions):
        proteus.NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior.__init__(self, vt, getPointwiseBoundaryConditions,
                                                                                        getAdvectiveFluxBoundaryConditions,
                                                                                        getDiffusiveFluxBoundaryConditions)


class Coefficients(proteus.TransportCoefficients.TC_base):
    """Basic k-epsilon model for incompressible flow from Hutter etal
Chaper 11 but solves for just k assuming epsilon computed
independently and lagged in time

    """
# \bar{\vec v} = <\vec v> Reynolds-ave
# raged (mean) velocity
# \vec v^{'}   = turbulent fluctuation
# assume \vec v = <\vec v> + \vec v^{'}, with <\vec v^{'}> = 0

# Reynolds averaged NS equations

# \deld \bar{\vec v} = 0

# \pd{\bar{\vec v}}{t} + \deld \left(\bar{\vec v} \outer \bar{\vec v}\right)
#                -\nu \deld \ten \bar{D} + \frac{1}{\rho}\grad \bar p
#                - \frac{1}{rho}\deld \ten{R} = 0

# Reynolds stress term

# \ten R = -\rho <\vec v^{'}\outer \vec v^{'}>
# \frac{1}{\rho}\ten{R} = 2 \nu_t \bar{D} - \frac{2}{3}k\ten{I}

# D_{ij}(\vec v) = \frac{1}{2} \left( \pd{v_i}{x_j} + \pd{v_j}{x_i})
# \ten D \bar{\ten D} = D(<\vec v>), \ten D^{'} = \ten D(\vec v^{'})


# k-epsilon tranport equations

# \pd{k}{t} + \deld (k\bar{\vec v})
#           - \deld\left[\left(\frac{\nu_t}{\sigma_k} + \nu\right)\grad k \right]
#           - 4\nu_t \Pi_{D} + \epsilon = 0

# \pd{\varepsilon}{t} + \deld (\varepsilon \bar{\vec v})
#           - \deld\left[\left(\frac{\nu_t}{\sigma_\varepsilon} + \nu\right)\grad \varepsilon \right]
#           - 4c_1 k \Pi_{D} + c_2 \frac{\epsilon^2}{k} = 0


# k              -- turbulent kinetic energy = <\vec v^{'}\dot \vec v^{'}>
# \varepsilon    -- turbulent dissipation rate = 4 \nu <\Pi_{D^{'}}>

# \nu            -- kinematic viscosity (\mu/\rho)
# \nu_t          -- turbulent viscosity = c_mu \frac{k^2}{\varepsilon}


# \Pi_{\ten A} = \frac{1}{2}tr(\ten A^2) = 1/2 \ten A\cdot \ten A
# \ten D \cdot \ten D = \frac{1}{4}\left[ (4 u_x^2 + 4 v_y^2 +
#                                         1/2 (u_y + v_x)^2 \right]

# 4 \Pi_{D} = 2 \frac{1}{4}\left[ (4 u_x^2 + 4 v_y^2 +
#                                 1/2 (u_y + v_x)^2 \right]
#           = \left[ (2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2 \right]

# \sigma_k -- Prandtl number \approx 1
# \sigma_e -- c_{\mu}/c_e

# c_{\mu} = 0.09, c_1 = 0.126, c_2 = 1.92, c_{\varepsilon} = 0.07


#     """

    from proteus.ctransportCoefficients import kEpsilon_k_3D_Evaluate_sd
    from proteus.ctransportCoefficients import kEpsilon_k_2D_Evaluate_sd
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    kEpsilon_k_3D_Evaluate_sd = staticmethod(kEpsilon_k_3D_Evaluate_sd)
    kEpsilon_k_2D_Evaluate_sd = staticmethod(kEpsilon_k_2D_Evaluate_sd)

    def __init__(self,
                 VOS_model=None, # Solid model
                 V_model=None, # Fluid model
                 LS_model=None,
                 RD_model=None,
                 dissipation_model=None,
                 ME_model=6,
                 SED_model=None,
                 dissipation_model_flag=1,  # default K-Epsilon, 2 --> K-Omega, 1998, 3 --> K-Omega 1988
                 c_mu=0.09,
                 sigma_k=1.0,  # Prandtl Number
                 rho_0=998.2,
                 nu_0=1.004e-6,
                 rho_1=1.205,
                 nu_1=1.500e-5,
                 g=[0.0, -9.8],
                 nd=3,
                 epsFact=0.01,
                 useMetrics=0.0,
                 sc_uref=1.0,
                 sc_beta=1.0,
                 default_dissipation=1.0e-3,
                 closure=None,
                 nullSpace='NoNullSpace',
                 initialize=True):

        self.useMetrics = useMetrics
        self.variableNames = ['kappa']
        self.nc = 1
        self.nd = nd
        # assert self.nd == 3, "Kappa only implements 3d for now" #assume 3d for now
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.c_mu = c_mu
        self.sigma_k = sigma_k
        self.g = g
        self.epsFact = epsFact
        self.flowModelIndex = V_model
        self.modelIndex = ME_model
        self.RD_modelIndex = RD_model
        self.LS_modelIndex = LS_model
        self.VOS_modelIndex = VOS_model
        self.SED_modelIndex = SED_model
        self.dissipation_modelIndex = dissipation_model
        self.dissipation_model_flag = dissipation_model_flag  # default K-Epsilon, 2 --> K-Omega, 1998, 3 --> K-Omega 1988
        self.sc_uref = sc_uref
        self.sc_beta = sc_beta
        self.nullSpace = nullSpace
        # for debugging model
        self.default_dissipation = default_dissipation
        self.closure = closure
        if initialize:
            self.initialize()

    def initialize(self):
        closure = self.closure
        try:
            self.aDarcy=closure.aDarcy
            self.betaForch=closure.betaForch
            self.grain=closure.grain
            self.packFraction=closure.packFraction
            self.packMargin=closure.packMargin
            self.maxFraction=closure.maxFraction
            self.frFraction=closure.frFraction
            self.sigmaC=closure.sigmaC
            self.C3e=closure.C3e
            self.C4e=closure.C4e
            self.eR=closure.eR
            self.fContact=closure.fContact
            self.mContact=closure.mContact
            self.nContact=closure.nContact
            self.angFriction=closure.angFriction
            self.vos_limiter = closure.vos_limiter
            self.mu_fr_limiter = closure.mu_fr_limiter
            self.sedFlag = 1
            prof.logEvent("INFO: Loading parameters for sediment closure",2)
        except:
            self.aDarcy=-1.
            self.betaForch=-1.
            self.grain=-1.
            self.packFraction=-1.
            self.packMargin=-1.
            self.maxFraction=-1.
            self.frFraction=-1.
            self.sigmaC=-1.
            self.C3e=-1.
            self.C4e=-1.
            self.eR=-1.
            self.fContact=-1.
            self.mContact=-1.
            self.nContact=-1.
            self.angFriction=-1.
            self.vos_limiter = -1.
            self.mu_fr_limiter = -1.
            self.sedFlag=0
            assert self.VOS_modelIndex == None
            assert self.SED_modelIndex == None
            prof.logEvent("Sediment module is off. Loading dummy parameters",2)
        #
        mass = {0: {0: 'linear'}}
        advection = {0: {0: 'linear'}}
        hamiltonian = {}
        potential = {0: {0: 'u'}}
        diffusion = {0: {0: {0: 'nonlinear', }}}
        reaction = {0: {0: 'nonlinear'}}
        if self.nd == 2:
            sdInfo = {(0, 0): (np.array([0, 1, 2], dtype='i'),
                               np.array([0, 1], dtype='i'))}
        else:
            sdInfo = {(0, 0): (np.array([0, 1, 2, 3], dtype='i'),
                               np.array([0, 1, 2], dtype='i'))}
        proteus.TransportCoefficients.TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames,
                         sparseDiffusionTensors=sdInfo)

    def initializeMesh(self, mesh):
        self.eps = self.epsFact * mesh.h

    def attachModels(self, modelList):
        assert self.modelIndex is not None and self.modelIndex < len(modelList), "Kappa: invalid index for self model allowed range: [0,%s]" % len(modelList)
        # self
        self.model = modelList[self.modelIndex]

        #self.u_old_dof = np.zeros(self.model.u[0].dof.shape,'d')
        self.u_old_dof = np.copy(self.model.u[0].dof)

        # redistanced level set
        if self.RD_modelIndex is not None:
            self.rdModel = modelList[self.RD_modelIndex]
        # level set
        if self.LS_modelIndex is not None:
            self.lsModel = modelList[self.LS_modelIndex]
            self.q_phi = modelList[self.LS_modelIndex].q[('u', 0)]
            self.ebqe_phi = modelList[self.LS_modelIndex].ebqe[('u', 0)]
            if ('u', 0) in modelList[self.LS_modelIndex].ebq:
                self.ebq_phi = modelList[self.LS_modelIndex].ebq[('u', 0)]
            else:
                self.ebq_phi = None
        else:
            self.q_phi =-np.ones( modelList[self.dissipation_modelIndex].q[('u', 0)].shape, 'd')
            #self.ebq_phi =-np.ones( modelList[self.dissipation_modelIndex].ebq[('u', 0)].shape, 'd')
            self.ebqe_phi = -np.ones( modelList[self.dissipation_modelIndex].ebqe[('u', 0)].shape, 'd')
        # flow model
        assert self.flowModelIndex is not None, "Kappa: invalid index for flow model allowed range: [0,%s]" % len(modelList)
        # print "flow model index------------",self.flowModelIndex,modelList[self.flowModelIndex].q.has_key(('velocity',0))
        if self.flowModelIndex is not None:  # keep for debugging for now
            self.model.ebqe['n'][:] = modelList[self.flowModelIndex].ebqe['n']
            if ('velocity', 0) in modelList[self.flowModelIndex].q:
                self.q_v = modelList[self.flowModelIndex].q[('velocity', 0)]
                self.ebqe_v = modelList[self.flowModelIndex].ebqe[('velocity', 0)]
            else:
                self.q_v = modelList[self.flowModelIndex].q[('f', 0)]
                self.ebqe_v = modelList[self.flowModelIndex].ebqe[('f', 0)]
            if ('velocity', 0) in modelList[self.flowModelIndex].ebq:
                self.ebq_v = modelList[self.flowModelIndex].ebq[('velocity', 0)]
            else:
                if ('f', 0) in modelList[self.flowModelIndex].ebq:
                    self.ebq_v = modelList[self.flowModelIndex].ebq[('f', 0)]
            #
            import copy
            self.q_grad_u = modelList[self.flowModelIndex].q[('grad(u)', 1)]
            self.q_grad_v = modelList[self.flowModelIndex].q[('grad(u)', 2)]
            #
            self.ebqe_grad_u = modelList[self.flowModelIndex].ebqe[('grad(u)', 1)]
            self.ebqe_grad_v = modelList[self.flowModelIndex].ebqe[('grad(u)', 2)]
            if ('grad(u)', 1) in modelList[self.flowModelIndex].ebq:
                self.ebq_grad_u = modelList[self.flowModelIndex].ebq[('grad(u)', 1)]
            if ('grad(u)', 2) in modelList[self.flowModelIndex].ebq:
                self.ebq_grad_v = modelList[self.flowModelIndex].ebq[('grad(u)', 2)]
            #
            # now allocate the 3D variables
            if self.nd == 2:
                self.q_grad_w = self.q_grad_v.copy()
                self.ebqe_grad_w = self.ebqe_grad_v.copy()
                if ('grad(u)', 2) in modelList[self.flowModelIndex].ebq:
                    self.ebq_grad_w = self.ebq_grad_v.copy()
            else:
                self.q_grad_w = modelList[self.flowModelIndex].q[('grad(u)', 3)]
                self.ebqe_grad_w = modelList[self.flowModelIndex].ebqe[('grad(u)', 3)]
                if ('grad(u)', 3) in modelList[self.flowModelIndex].ebq:
                    self.ebq_grad_w = modelList[self.flowModelIndex].ebq[('grad(u)', 3)]
            #

            self.velocity_dof_u = modelList[self.flowModelIndex].u[1].dof
            self.velocity_dof_v = modelList[self.flowModelIndex].u[2].dof
            if self.nd == 2:
                self.velocity_dof_w = self.velocity_dof_v.copy()
            else:
                self.velocity_dof_w = modelList[self.flowModelIndex].u[3].dof
            if hasattr(modelList[self.flowModelIndex].coefficients, 'q_porosity'):
                self.q_porosity = modelList[self.flowModelIndex].coefficients.q_porosity
            else:
                self.q_porosity = np.ones(self.q[('u', 0)].shape, 'd')
            if hasattr(modelList[self.flowModelIndex].coefficients, 'ebqe_porosity'):
                self.ebqe_porosity = modelList[self.flowModelIndex].coefficients.ebqe_porosity
            else:
                self.ebqe_porosity = np.ones( modelList[self.flowModelIndex].ebqe[('velocity', 0)].shape, 'd')
        else:
            self.velocity_dof_u = np.zeros(self.model.u[0].dof.shape, 'd')
            self.velocity_dof_v = np.zeros(self.model.u[0].dof.shape, 'd')
            if self.nd == 2:
                self.velocity_dof_w = self.velocity_dof_v.copy()
            else:
                self.velocity_dof_w = np.zeros(self.model.u[0].dof.shape, 'd')
            self.q_porosity = np.ones(self.q[('u', 0)].shape, 'd')
            self.ebqe_porosity = np.ones( modelList[self.dissipation_modelIndex].ebqe[('u', 0)].shape, 'd')

        #
        #assert self.dissipation_modelIndex is not None and self.dissipation_modelIndex < len(modelList), "Kappa: invalid index for dissipation model allowed range: [0,%s]" % len(modelList)
        if self.dissipation_modelIndex is not None:  # keep for debugging for now
            # assume have q,ebqe always
            self.q_dissipation = modelList[self.dissipation_modelIndex].q[('u', 0)]
            self.ebqe_dissipation = modelList[self.dissipation_modelIndex].ebqe[('u', 0)]
            self.q_grad_dissipation = modelList[self.dissipation_modelIndex].q[('grad(u)', 0)]
            if ('u', 0) in modelList[self.dissipation_modelIndex].ebq:
                self.ebq_dissipation = modelList[self.dissipation_modelIndex].ebq[('u', 0)]
        else:
            self.q_dissipation = np.zeros(self.model.q[('u', 0)].shape, 'd')
            self.q_dissipation.fill(self.default_dissipation)
            self.ebqe_dissipation = np.zeros(self.model.ebqe[('u', 0)].shape, 'd')
            self.ebqe_dissipation.fill(self.default_dissipation)
            self.q_grad_dissipation = np.zeros(self.model.q[('grad(u)', 0)].shape, 'd')

            if ('u', 0) in self.model.ebq:
                self.ebq_dissipation = np.zeros(self.model.ebq[('u', 0)].shape, 'd')
                self.ebq_dissipation.fill(self.default_dissipation)
            #
        if self.VOS_modelIndex is not None:
            self.vosModel = model[self.VOS_modelIndex ]
            self.q_vos = modelList[self.VOS_modelIndex].q[('u', 0)]
            self.grad_vos = modelList[self.VOS_modelIndex].q[('grad(u)', 0)]
            self.ebqe_vos = modelList[self.VOS_modelIndex].ebqe[('u', 0)]
            self.ebqe_grad_vos = modelList[self.VOS_modelIndex].ebqe[('grad(u)', 0)]
        else:
            self.q_vos = self.model.q[('u', 0)]
            self.grad_vos = self.model.q[('u', 0)]
            self.ebqe_vos = self.model.ebqe[('u', 0)]
            self.ebqe_grad_vos = self.model.ebqe[('u', 0)]
        if self.SED_modelIndex is not None:
            self.rho_s=modelList[self.SED_modelIndex].coefficients.rho_s
            self.vs=modelList[self.SED_modelIndex].q[('u', 0)]
            self.ebqe_vs=modelList[self.SED_modelIndex].ebqe[('u', 0)]
        else:
            self.rho_s=self.rho_0
            self.vs=self.q_v
            self.ebqe_vs=self.ebqe_v
        #

    def initializeElementQuadrature(self, t, cq):
        if self.flowModelIndex is None:
            self.q_v = np.ones(cq[('f', 0)].shape, 'd')
            self.q_grad_u = np.ones(cq[('grad(u)', 0)].shape, 'd')
            self.q_grad_v = np.ones(cq[('grad(u)', 0)].shape, 'd')
            if self.nd == 2:
                self.q_grad_w = self.q_grad_v.copy()
            else:
                self.q_grad_w = np.ones(cq[('grad(u)', 0)].shape, 'd')
        if self.dissipation_modelIndex is None:
            self.q_dissipation = np.ones(cq[('u', 0)].shape, 'd')
            self.q_dissipation.fill(self.default_dissipation)
            self.q_grad_dissipation = np.zeros(cq[('grad(u)', 0)].shape, 'd')

    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        if self.flowModelIndex is None:
            self.ebq_v = np.ones(cebq[('f', 0)].shape, 'd')
            self.ebq_grad_u = np.ones(cebq[('grad(u)', 0)].shape, 'd')
            self.ebq_grad_v = np.ones(cebq[('grad(u)', 0)].shape, 'd')
            if self.nd == 2:
                self.ebq_grad_w = self.ebq_grad_v.copy()
            else:
                self.ebq_grad_w = np.ones(cebq[('grad(u)', 0)].shape, 'd')
        if self.dissipation_modelIndex is None:
            self.ebq_dissipation = np.ones(cebq[('u', 0)].shape, 'd')
            self.ebq_dissipation.fill(self.default_dissipation)

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        if self.flowModelIndex is None:
            self.ebqe_v = np.ones(cebqe[('f', 0)].shape, 'd')
            self.ebqe_grad_u = np.ones(cebqe[('grad(u)', 0)].shape, 'd')
            self.ebqe_grad_v = np.ones(cebqe[('grad(u)', 0)].shape, 'd')
            if self.nd == 2:
                self.ebqe_grad_w = self.ebqe_grad_v.copy()
            else:
                self.ebqe_grad_w = np.ones(cebqe[('grad(u)', 0)].shape, 'd')
        if self.dissipation_modelIndex is None:
            self.ebqe_dissipation = np.ones(cebqe[('u', 0)].shape, 'd')
            self.ebqe_dissipation.fill(self.default_dissipation)

    def preStep(self, t, firstStep=False):
        copyInstructions = {}
        return copyInstructions

    def postStep(self, t, firstStep=False):
        self.u_old_dof = np.copy(self.model.u[0].dof)

        #Limit k (hard limit)
        
        for eN in range(self.model.q[('u',0)].shape[0]):
            for k in range(self.model.q[('u',0)].shape[1]):                
                self.model.q[('u',0)][eN,k] = max(  self.model.q[('u',0)][eN,k], 1e-20)

        if ('u', 0) in self.model.ebq:
            for eN in range(self.model.ebq[('u',0)].shape[0]):
                for k in range(self.model.ebq[('u',0)].shape[1]):
                    for l in range(len(self.model.ebq[('u',0)][eN,k])):
                        self.model.ebq[('u',0)][eN,k,l] = max(  self.model.ebq[('u',0)][eN,k,l], 1e-20)
        for eN in range(self.model.ebqe[('u',0)].shape[0]):
            for k in range(self.model.ebqe[('u',0)].shape[1]):
                self.model.ebqe[('u',0)][eN,k] = max(  self.model.ebqe[('u',0)][eN,k], 1e-20)


        copyInstructions = {}
        return copyInstructions

    def updateToMovingDomain(self, t, c):
        # in a moving domain simulation the velocity coming in is already for the moving domain
        pass

    def evaluate(self, t, c):
        # mwf debug
        # print "Kappacoeficients eval t=%s " % t
        if c[('f', 0)].shape == self.q_v.shape:
            v = self.q_v
            phi = self.q_phi
            grad_u = self.q_grad_u
            grad_v = self.q_grad_v
            grad_w = self.q_grad_w
            dissipation = self.q_dissipation
        elif c[('f', 0)].shape == self.ebqe_v.shape:
            v = self.ebqe_v
            phi = self.ebqe_phi
            grad_u = self.ebqe_grad_u
            grad_v = self.ebqe_grad_v
            grad_w = self.ebqe_grad_w
            dissipation = self.ebqe_dissipation
        elif ((self.ebq_v is not None and self.ebq_phi is not None and self.ebq_grad_u is not None and self.ebq_grad_v is not None and self.ebq_grad_w is not None and self.ebq_dissipation is not None) and c[('f', 0)].shape == self.ebq_v.shape):
            v = self.ebq_v
            phi = self.ebq_phi
            grad_u = self.ebq_grad_u
            grad_v = self.ebq_grad_v
            grad_w = self.ebqe_grad_w
            dissipation = self.ebq_dissipation
        else:
            v = None
            phi = None
            grad_u = None
            grad_v = None
            grad_w = None
        if v is not None and self.dissipation_model_flag < 2:
            if self.nd == 2:
                self.kEpsilon_k_2D_Evaluate_sd(self.sigma_k,
                                               self.c_mu,
                                               self.nu,
                                               velocity,
                                               gradu,
                                               gradv,
                                               c[('u', 0)],
                                               dissipation,
                                               c[('m', 0)],
                                               c[('dm', 0, 0)],
                                               c[('f', 0)],
                                               c[('df', 0, 0)],
                                               c[('a', 0, 0)],
                                               c[('da', 0, 0, 0)],
                                               c[('r', 0)],
                                               c[('dr', 0, 0)])

            else:
                self.kEpsilon_k_3D_Evaluate_sd(self.sigma_k,
                                               self.c_mu,
                                               self.nu,
                                               velocity,
                                               gradu,
                                               gradv,
                                               gradw,
                                               c[('u', 0)],
                                               dissipation,
                                               c[('m', 0)],
                                               c[('dm', 0, 0)],
                                               c[('f', 0)],
                                               c[('df', 0, 0)],
                                               c[('a', 0, 0)],
                                               c[('da', 0, 0, 0)],
                                               c[('r', 0)],
                                               c[('dr', 0, 0)])
        else:
            print("WARNING! dissipation_model_flag != 1 not implemented in Kappa.coefficients")


class LevelModel(proteus.Transport.OneLevelTransport):
    nCalls = 0

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
                 stressTraceBoundaryConditionsSetterDict=None,
                 stabilization=None,
                 shockCapturing=None,
                 conservativeFluxDict=None,
                 numericalFluxType=None,
                 TimeIntegrationClass=None,
                 massLumping=False,
                 reactionLumping=False,
                 options=None,
                 name='defaultName',
                 reuse_trial_and_test_quadrature=True,
                 sd = True,
                 movingDomain=False,
                 bdyNullSpace=False):
        #
        # set the objects describing the method and boundary conditions
        #
        self.bdyNullSpace=bdyNullSpace
        self.movingDomain=movingDomain
        self.tLast_mesh=None
        #
        self.name = name
        self.sd = sd
        self.Hess = False
        self.lowmem = True
        self.timeTerm = True  # allow turning off  the  time derivative
        # self.lowmem=False
        self.testIsTrial = True
        self.phiTrialIsTrial = True
        self.u = uDict
        self.ua = {}  # analytical solutions
        self.phi = phiDict
        self.dphi = {}
        self.matType = matType
        # try to reuse test and trial information across components if spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature  # True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1, coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        # Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh  # assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList = None  # explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict  # no velocity post-processing for now
        self.fluxBoundaryConditions = fluxBoundaryConditionsDict
        self.advectiveFluxBoundaryConditionsSetterDict = advectiveFluxBoundaryConditionsSetterDict
        self.diffusiveFluxBoundaryConditionsSetterDictDict = diffusiveFluxBoundaryConditionsSetterDictDict
        # determine whether  the stabilization term is nonlinear
        self.stabilizationIsNonlinear = False
        # cek come back
        if self.stabilization is not None:
            for ci in range(self.nc):
                if ci in coefficients.mass:
                    for flag in list(coefficients.mass[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.advection:
                    for flag in list(coefficients.advection[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.diffusion:
                    for diffusionDict in list(coefficients.diffusion[ci].values()):
                        for flag in list(diffusionDict.values()):
                            if flag != 'constant':
                                self.stabilizationIsNonlinear = True
                if ci in coefficients.potential:
                    for flag in list(coefficients.potential[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.reaction:
                    for flag in list(coefficients.reaction[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.hamiltonian:
                    for flag in list(coefficients.hamiltonian[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
        # determine if we need element boundary storage
        self.elementBoundaryIntegrals = {}
        for ci in range(self.nc):
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux is not None) or
                                                 (numericalFluxType is not None) or
                                                 (self.fluxBoundaryConditions[ci] == 'outFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'mixedFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        # calculate some dimensions
        #
        self.nSpace_global = self.u[0].femSpace.nSpace_global  # assume same space dim for all variables
        self.nDOF_trial_element = [u_j.femSpace.max_nDOF_element for u_j in list(self.u.values())]
        self.nDOF_phi_trial_element = [phi_k.femSpace.max_nDOF_element for phi_k in list(self.phi.values())]
        self.n_phi_ip_element = [phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for phi_k in list(self.phi.values())]
        self.nDOF_test_element = [femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        self.nFreeDOF_global = [dc.nFreeDOF_global for dc in list(self.dirichletConditions.values())]
        self.nVDOF_element = sum(self.nDOF_trial_element)
        self.nFreeVDOF_global = sum(self.nFreeDOF_global)
        #
        proteus.NonlinearSolvers.NonlinearEquation.__init__(self, self.nFreeVDOF_global)
        #
        # build the quadrature point dictionaries from the input (this
        # is just for convenience so that the input doesn't have to be
        # complete)
        #
        elementQuadratureDict = {}
        elemQuadIsDict = isinstance(elementQuadrature, dict)
        if elemQuadIsDict:  # set terms manually
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
                        elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature
        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff', ci, ci) in elementQuadrature:
                        elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature[('numDiff', ci, ci)]
                    else:
                        elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature
        if massLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('m', ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        if reactionLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('r', ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        elementBoundaryQuadratureDict = {}
        if isinstance(elementBoundaryQuadrature, dict):  # set terms manually
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
        # mwf include tag telling me which indices are which quadrature rule?
        (self.elementQuadraturePoints, self.elementQuadratureWeights,
         self.elementQuadratureRuleIndeces) = proteus.Quadrature.buildUnion(elementQuadratureDict)
        self.nQuadraturePoints_element = self.elementQuadraturePoints.shape[0]
        self.nQuadraturePoints_global = self.nQuadraturePoints_element * self.mesh.nElements_global
        #
        # Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints,
         self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = proteus.Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[0]
        self.nElementBoundaryQuadraturePoints_global = (self.mesh.nElements_global *
                                                        self.mesh.nElementBoundaries_element *
                                                        self.nElementBoundaryQuadraturePoints_elementBoundary)
#        if isinstance(self.u[0].femSpace,C0_AffineLinearOnSimplexWithNodalBasis):
#            print self.nQuadraturePoints_element
#            if self.nSpace_global == 3:
#                assert(self.nQuadraturePoints_element == 5)
#            elif self.nSpace_global == 2:
#                assert(self.nQuadraturePoints_element == 6)
#            elif self.nSpace_global == 1:
#                assert(self.nQuadraturePoints_element == 3)
#
#            print self.nElementBoundaryQuadraturePoints_elementBoundary
#            if self.nSpace_global == 3:
#                assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 4)
#            elif self.nSpace_global == 2:
#                assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 4)
#            elif self.nSpace_global == 1:
#                assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 1)

        #
        # storage dictionaries
        self.scalars_element = set()
        #
        # simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q = {}
        self.ebq = {}
        self.ebq_global = {}
        self.ebqe = {}
        self.phi_ip = {}
        # mesh
        #self.q['x'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,3),'d')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary, 3), 'd')
        self.ebqe['n'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.q[('u', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('grad(u)', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        #diffusion, isotropic
        self.q[('a', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('da', 0, 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        # linear potential
        self.q[('phi', 0)] = self.q[('u', 0)]
        self.q[('grad(phi)', 0)] = self.q[('grad(u)', 0)]
        self.q[('dphi', 0, 0)] = np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        # mass
        self.q[('m', 0)] = self.q[('u', 0)]
        self.q[('m_last', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 0)] = self.q[('u', 0)]
        self.q[('cfl', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe[('u', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('grad(u)', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('advectiveFlux_bc_flag', 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc_flag', 0, 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc', 0, 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('penalty')] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set([('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,), 'i')
            self.inflowBoundaryBC_values[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nDOF_trial_element[cj]), 'd')
            self.inflowFlux[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.internalNodes = set(range(self.mesh.nNodes_global))
        # identify the internal nodes this is ought to be in mesh
        # \todo move this to mesh
        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global = self.mesh.elementBoundaryElementsArray[ebN, 0]
            ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
            for i in range(self.mesh.nNodes_element):
                if i != ebN_element:
                    I = self.mesh.elementNodesArray[eN_global, i]
                    self.internalNodes -= set([I])
        self.nNodes_internal = len(self.internalNodes)
        self.internalNodesArray = np.zeros((self.nNodes_internal,), 'i')
        for nI, n in enumerate(self.internalNodes):
            self.internalNodesArray[nI] = n
        #
        del self.internalNodes
        self.internalNodes = None
        prof.logEvent("Updating local to global mappings", 2)
        self.updateLocal2Global()
        prof.logEvent("Building time integration object", 2)
        prof.logEvent(prof.memory("inflowBC, internalNodes,updateLocal2Global", "OneLevelTransport"), level=4)
        # mwf for interpolating subgrid error for gradients etc
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(self, integrateInterpolationPoints=True)
        else:
            self.timeIntegration = TimeIntegrationClass(self)

        if options is not None:
            self.timeIntegration.setFromOptions(options)
        prof.logEvent(prof.memory("TimeIntegration", "OneLevelTransport"), level=4)
        prof.logEvent("Calculating numerical quadrature formulas", 2)
        self.calculateQuadrature()

        self.setupFieldStrides()

        comm = proteus.Comm.get()
        self.comm = comm
        if comm.size() > 1:
            assert numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions, "You must use a numerical flux to apply weak boundary conditions for parallel runs"

        prof.logEvent(prof.memory("stride+offset", "OneLevelTransport"), level=4)
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
        else:
            self.numericalFlux = None
        # set penalty terms
        # cek todo move into numerical flux initialization
        if 'penalty' in self.ebq_global:
            for ebN in range(self.mesh.nElementBoundaries_global):
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['penalty'][ebN, k] = self.numericalFlux.penalty_constant/(self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power)
        # penalty term
        # cek move  to Numerical flux initialization
        if 'penalty' in self.ebqe:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebqe['penalty'][ebNE, k] = self.numericalFlux.penalty_constant/self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power
        prof.logEvent(prof.memory("numericalFlux", "OneLevelTransport"), level=4)
        self.elementEffectiveDiametersArray = self.mesh.elementInnerDiametersArray
        # use post processing tools to get conservative fluxes, None by default
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)
        prof.logEvent(prof.memory("velocity postprocessor", "OneLevelTransport"), level=4)
        # helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        # TODO get rid of this
        # mwf can I use the numericalFlux's flag information?
        for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag', ci)] = np.zeros(self.ebqe[('advectiveFlux_bc', ci)].shape, 'i')
            for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1

            for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                self.ebqe[('diffusiveFlux_bc_flag', ck, ci)] = np.zeros(self.ebqe[('diffusiveFlux_bc', ck, ci)].shape, 'i')
                for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                    self.ebqe[('diffusiveFlux_bc', ck, ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('diffusiveFlux_bc_flag', ck, ci)][t[0], t[1]] = 1
        if hasattr(self.numericalFlux, 'setDirichletValues'):
            self.numericalFlux.setDirichletValues(self.ebqe)
        if not hasattr(self.numericalFlux, 'isDOFBoundary'):
            self.numericalFlux.isDOFBoundary = {0: np.zeros(self.ebqe[('u', 0)].shape, 'i')}
        if not hasattr(self.numericalFlux, 'ebqe'):
            self.numericalFlux.ebqe = {('u', 0): np.zeros(self.ebqe[('u', 0)].shape, 'd')}
        # TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        compKernelFlag = 0
        if self.nSpace_global == 2:
            self.kappa = cKappa2D_base(self.nSpace_global,
                                       self.nQuadraturePoints_element,
                                       self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                       self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                       self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                       self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                   compKernelFlag,
                                                   self.coefficients.aDarcy,
                                                    self.coefficients.betaForch,
                                                    self.coefficients.grain,
                                                    self.coefficients.packFraction,
                                                    self.coefficients.packMargin,
                                                    self.coefficients.maxFraction,
                                                    self.coefficients.frFraction,
                                                    self.coefficients.sigmaC,
                                                    self.coefficients.C3e,
                                                    self.coefficients.C4e,
                                                    self.coefficients.eR,
                                                    self.coefficients.fContact,
                                                    self.coefficients.mContact,
                                                    self.coefficients.nContact,
                                                    self.coefficients.angFriction,
                                                    self.coefficients.vos_limiter,
                                                    self.coefficients.mu_fr_limiter)

        else:
            self.kappa = cKappa_base(self.nSpace_global,
                                     self.nQuadraturePoints_element,
                                     self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                     self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                     self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                     self.nElementBoundaryQuadraturePoints_elementBoundary,
                                     compKernelFlag,
                                     self.coefficients.aDarcy,
                                     self.coefficients.betaForch,
                                     self.coefficients.grain,
                                     self.coefficients.packFraction,
                                     self.coefficients.packMargin,
                                     self.coefficients.maxFraction,
                                     self.coefficients.frFraction,
                                     self.coefficients.sigmaC,
                                     self.coefficients.C3e,
                                     self.coefficients.C4e,
                                     self.coefficients.eR,
                                     self.coefficients.fContact,
                                     self.coefficients.mContact,
                                     self.coefficients.nContact,
                                     self.coefficients.angFriction,
                                     self.coefficients.vos_limiter,
                                     self.coefficients.mu_fr_limiter)


        self.forceStrongConditions = False
        if self.forceStrongConditions:
            self.dirichletConditionsForceDOF = DOFBoundaryConditions(self.u[0].femSpace, dofBoundaryConditionsSetterDict[0], weakDirichletConditions=False)

        if self.movingDomain:
            self.MOVING_DOMAIN = 1.0
        else:
            self.MOVING_DOMAIN = 0.0
        # cek hack
        self.movingDomain = False
        self.MOVING_DOMAIN = 0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape, 'd')
    # mwf these are getting called by redistancing classes,

    def calculateCoefficients(self):
        pass

    def calculateElementResidual(self):
        if self.globalResidualDummy is not None:
            self.getResidual(self.u[0].dof, self.globalResidualDummy)

    def getResidual(self, u, r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """

        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        # print "***************max/min(m_last)*********************",max(self.timeIntegration.m_last[0].flat[:]),min(self.timeIntegration.m_last[0].flat[:])
        # print "***************max/min(m_last)*********************",max(-self.timeIntegration.dt*self.timeIntegration.beta_bdf[0].flat[:]),min(-self.timeIntegration.dt*self.timeIntegration.beta_bdf[0].flat[:]),
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        # cek can put in logic to skip of BC's don't depend on t or u
        # Dirichlet boundary conditions
        # if hasattr(self.numericalFlux,'setDirichletValues'):
        self.numericalFlux.setDirichletValues(self.ebqe)
        # flux boundary conditions
        for t, g in list(self.fluxBoundaryConditionsObjectsDict[0].advectiveFluxBoundaryConditionsDict.items()):
            self.ebqe[('advectiveFlux_bc', 0)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
            self.ebqe[('advectiveFlux_bc_flag', 0)][t[0], t[1]] = 1
        for ck, diffusiveFluxBoundaryConditionsDict in list(self.fluxBoundaryConditionsObjectsDict[0].diffusiveFluxBoundaryConditionsDictDict.items()):
            for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                self.ebqe[('diffusiveFlux_bc', ck, 0)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                self.ebqe[('diffusiveFlux_bc_flag', ck, 0)][t[0], t[1]] = 1
        # self.shockCapturing.lag=True

        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                self.u[0].dof[dofN] = g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        #
        # mwf debug
        #import pdb
        # pdb.set_trace()
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nu_0"] = self.coefficients.nu_0
        argsDict["nu_1"] = self.coefficients.nu_1
        argsDict["sigma_k"] = self.coefficients.sigma_k
        argsDict["c_mu"] = self.coefficients.c_mu
        argsDict["rho_0"] = self.coefficients.rho_0
        argsDict["rho_1"] = self.coefficients.rho_1
        argsDict["sedFlag"] = self.coefficients.sedFlag
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["q_vos_gradc"] = self.coefficients.grad_vos
        argsDict["ebqe_q_vos"] = self.coefficients.ebqe_vos
        argsDict["ebqe_q_vos_gradc"] = self.coefficients.ebqe_grad_vos
        argsDict["rho_f"] = self.coefficients.rho_0
        argsDict["rho_s"] = self.coefficients.rho_s
        argsDict["vs"] = self.coefficients.vs
        argsDict["ebqe_vs"] = self.coefficients.ebqe_vs
        argsDict["g"] = self.coefficients.g
        argsDict["dissipation_model_flag"] = self.coefficients.dissipation_model_flag
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = self.shockCapturing.shockCapturingFactor
        argsDict["sc_uref"] = self.coefficients.sc_uref
        argsDict["sc_alpha"] = self.coefficients.sc_beta
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.coefficients.u_old_dof
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["phi_ls"] = self.coefficients.q_phi
        argsDict["q_dissipation"] = self.coefficients.q_dissipation
        argsDict["q_grad_dissipation"] = self.coefficients.q_grad_dissipation
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["velocity_dof_u"] = self.coefficients.velocity_dof_u
        argsDict["velocity_dof_v"] = self.coefficients.velocity_dof_v
        argsDict["velocity_dof_w"] = self.coefficients.velocity_dof_w
        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_grad_u"] = self.q[('grad(u)', 0)]
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u"] = self.shockCapturing.numDiff[0]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["ebqe_bc_advectiveFlux_u_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["ebqe_bc_diffusiveFlux_u_ext"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        argsDict["ebqe_phi"] = self.coefficients.ebqe_phi
        argsDict["epsFact"] = self.coefficients.epsFact
        argsDict["ebqe_dissipation"] = self.coefficients.ebqe_dissipation
        argsDict["ebqe_porosity"] = self.coefficients.ebqe_porosity
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux', 0)]
        self.kappa.calculateResidual(argsDict)
        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                r[dofN] = 0

        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        prof.logEvent("Global residual", level=9, data=r)
        # mwf decide if this is reasonable for keeping solver statistics
        self.nonlinear_function_evaluations += 1
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape, 'd')

    def getJacobian(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nu_0"] = self.coefficients.nu_0
        argsDict["nu_1"] = self.coefficients.nu_1
        argsDict["sigma_k"] = self.coefficients.sigma_k
        argsDict["c_mu"] = self.coefficients.c_mu
        argsDict["rho_0"] = self.coefficients.rho_0
        argsDict["rho_1"] = self.coefficients.rho_1
        argsDict["dissipation_model_flag"] = self.coefficients.dissipation_model_flag
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = self.shockCapturing.shockCapturingFactor
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.coefficients.u_old_dof
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["phi_ls"] = self.coefficients.q_phi
        argsDict["q_dissipation"] = self.coefficients.q_dissipation
        argsDict["q_grad_dissipation"] = self.coefficients.q_grad_dissipation
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["sedFlag"] = self.coefficients.sedFlag
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["q_vos_gradc"] = self.coefficients.grad_vos
        argsDict["ebqe_q_vos"] = self.coefficients.ebqe_vos
        argsDict["ebqe_q_vos_gradc"] = self.coefficients.ebqe_grad_vos
        argsDict["rho_f"] = self.coefficients.rho_0
        argsDict["rho_s"] = self.coefficients.rho_s
        argsDict["vs"] = self.coefficients.vs
        argsDict["ebqe_vs"] = self.coefficients.ebqe_vs
        argsDict["g"] = self.coefficients.g
        argsDict["velocity_dof_u"] = self.coefficients.velocity_dof_u
        argsDict["velocity_dof_v"] = self.coefficients.velocity_dof_v
        argsDict["velocity_dof_w"] = self.coefficients.velocity_dof_w
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["ebqe_bc_advectiveFlux_u_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["ebqe_bc_diffusiveFlux_u_ext"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["ebqe_phi"] = self.coefficients.ebqe_phi
        argsDict["epsFact"] = self.coefficients.epsFact
        argsDict["ebqe_dissipation"] = self.coefficients.ebqe_dissipation
        argsDict["ebqe_porosity"] = self.coefficients.ebqe_porosity
        self.kappa.calculateJacobian(argsDict)  # VRANS

        # Load the Dirichlet conditions directly into residual
        if self.forceStrongConditions:
            scaling = 1.0  # probably want to add some scaling to match non-dirichlet diagonals in linear system
            for dofN in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.keys()):
                global_dofN = dofN
                for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                            # print "RBLES forcing residual cj = %s dofN= %s global_dofN= %s was self.nzval[i]= %s now =%s " % (cj,dofN,global_dofN,self.nzval[i],scaling)
                        self.nzval[i] = scaling
                    else:
                        self.nzval[i] = 0.0
                        # print "RBLES zeroing residual cj = %s dofN= %s global_dofN= %s " % (cj,dofN,global_dofN)

        prof.logEvent("Jacobian ", level=10, data=jacobian)
        # mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian

    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        # self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
        #                                         self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t, self.q)
        if self.stabilization is not None:
            self.stabilization.initializeElementQuadrature(self.mesh, self.timeIntegration.t, self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing is not None:
            self.shockCapturing.initializeElementQuadrature(self.mesh, self.timeIntegration.t, self.q)

    def calculateElementBoundaryQuadrature(self):
        pass

    def calculateExteriorElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        #
        # get physical locations of element boundary quadrature points
        #
        # assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        self.fluxBoundaryConditionsObjectsDict = dict([(cj, proteus.FemTools.FluxBoundaryConditions(self.mesh,
                                                                                   self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                   self.ebqe[('x')],
                                                                                   getAdvectiveFluxBoundaryConditions=self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                                                                   getDiffusiveFluxBoundaryConditions=self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                       for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t, self.ebqe)

    def estimate_mt(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        pass
