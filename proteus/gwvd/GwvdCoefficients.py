"""
Variable-density groundwater flow and transport coefficients (continuous Galerkin path).

Ported from erdc/proteus-gwvd@14961d2: `src/GwvdCoefficients.py` (classes) and
`src/gwvdCoefficients.c` (`density_eval`, `knabnerFrolkovicConsistentVelCorrection_Triangle_P1`,
`vdgroundwater_mod2_FlowCoefficientsEvaluate`, `vdgroundwater_mod2_TransportCoefficientsEvaluate`).
The primary variables are the freshwater head h and the normalized mass fraction c (`mf`).
The flow "velocity" handed to transport is the mass flux rho*q from the flow model's velocity
post-processing (`conservativeFlux`).
"""
import numpy as np
from proteus import TransportCoefficients
from proteus import FemTools
from proteus.Profiling import logEvent


def density_eval(mf, epsilon, rho_fw):
    """Linear density relation rho = rho_fw (1 + epsilon c)."""
    return rho_fw * (1.0 + epsilon * mf)


def var_visc(mf, mf_s):
    """Relative viscosity of Lever & Jackson (1985), as in the 2012 code (a ratio, not a viscosity)."""
    mf_a = mf * mf_s
    return 1.0 + 1.85 * mf_a - 4.1 * mf_a**2 + 44.5 * mf_a**3


def knabnerFrolkovicConsistentVelCorrection_Triangle_P1(epsilon, c_l2g, c_dof, J, Jinv, gravity,
                                                        q_rhog_consistent, eps_tol=1.0e-9):
    """Knabner-Frolkovic (1996) consistent rho*g on P1 triangles, Eqs. 3.3-3.16.

    Writes J^{-T} hhat into q_rhog_consistent (nElements x 2), where hhat holds the reference-element
    derivatives of the integrated buoyancy term. For the linear density used here the exact integral
    is the mean of the two vertex values, which is what the 2012 C code evaluates.
    """
    JeT = np.swapaxes(J[:, 0], 1, 2)
    JeTinv = np.swapaxes(Jinv[:, 0], 1, 2)
    ghat = np.einsum('eij,j->ei', JeT, gravity)
    c_e = c_dof[c_l2g[:, :3]]
    h1 = np.where(np.abs(c_e[:, 1] - c_e[:, 0]) <= eps_tol,
                  epsilon * c_e[:, 0], 0.5 * epsilon * (c_e[:, 1] + c_e[:, 0]))
    h2 = np.where(np.abs(c_e[:, 2] - c_e[:, 0]) <= eps_tol,
                  epsilon * c_e[:, 0], 0.5 * epsilon * (c_e[:, 2] + c_e[:, 0]))
    h_hat = np.stack([ghat[:, 0] * h1, ghat[:, 1] * h2], axis=1)
    q_rhog_consistent[:] = np.einsum('eij,ej->ei', JeTinv, h_hat)


class VariableDensity_mod2_FlowEquationCoefficients(TransportCoefficients.TC_base):
    """
    Variable-density subsurface flow in non-divergence (mod2) form for the freshwater head h:

      S rho h_t + div(f - a grad h) + r = 0,
      a = k rho_fw g rho / mu,  f = -a epsilon c e_z  (or -a (rho g)~ with the consistent velocity),
      r = omega (drho/dc) c_t / rho.

    The mass fraction c and its time derivative come from the transport model.
    """

    def __init__(self, aOfX=None, rOfX=None, bOfX=None, S=1.0, omega=.3, epsilon=.025, rho_fw=1000.0,
                 gc=9.8, Kx=1.0e-10, Ky=1.0e-10, Kz=1.0e-10, mu=1.0e-3, mf_s=0.167, drho_dmf=200.0,
                 rho_source=0.0, T_model=None, C_model=None, flowModelIndex=0, transModelIndex=1,
                 cModelIndex=2, nc=1, nd=2, consistentVelocityFlag=0, consistentVelocityFunction=None,
                 variableViscosityFlag=0, rho_eval=None, rhoInt_eval=None, gravity=None,
                 getInterface=None, Height=1.0):
        self.aOfX = aOfX
        self.rOfX = rOfX
        self.bOfX = bOfX
        self.omega = omega
        self.epsilon = epsilon
        self.S = S
        self.gc = gc
        self.rho_fw = rho_fw
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
        self.mu = mu
        self.mf_s = mf_s
        self.rho_source = rho_source
        self.nd = nd
        self.drho_dmf = drho_dmf
        self.T_model = T_model
        self.C_model = C_model
        self.getInterface = getInterface
        self.Height = Height
        self.consistentVelocityFlag = consistentVelocityFlag  # 0 none, 1 Knabner-Frolkovic 96
        self.variableViscosityFlag = variableViscosityFlag
        if C_model is not None:
            raise NotImplementedError("surface-water coupling (C_model) is not ported")
        if consistentVelocityFlag == 1 and nd != 2:
            raise NotImplementedError("only the P1 triangle consistent velocity is ported")
        mass, advection, diffusion, potential, reaction, hamiltonian = {}, {}, {}, {}, {}, {}
        for i in range(nc):
            mass[i] = {i: 'linear'}
            advection[i] = {i: 'constant'}
            diffusion[i] = {i: {i: 'constant'}}
            potential[i] = {i: 'u'}
            reaction[i] = {i: 'constant'}
        TransportCoefficients.TC_base.__init__(self, nc, mass, advection, diffusion, potential,
                                               reaction, hamiltonian, variableNames=['h'])
        self.transModelIndex = transModelIndex
        self.flowModelIndex = flowModelIndex
        self.cModelIndex = cModelIndex
        self.q_mf = None
        self.q_mfdt = None
        self.ebqe_mf = None
        self.ebqe_mfdt = None
        self.ebq_mf = None
        self.ebq_mfdt = None
        self.consistentVelocityFunction = consistentVelocityFunction
        self.q_rhog_consistent = None
        self.rho_eval = rho_eval
        self.rhoInt_eval = rhoInt_eval
        self.gravity = gravity

    def preStep(self, t, firstStep=False):
        """Build the consistent rho*g from the current mass fraction, if requested."""
        if self.consistentVelocityFlag == 1 and self.T_model is not None:
            assert isinstance(self.transModel.u[0].femSpace, FemTools.C0_AffineLinearOnSimplexWithNodalBasis)
            knabnerFrolkovicConsistentVelCorrection_Triangle_P1(self.epsilon,
                                                                self.transModel.u[0].femSpace.dofMap.l2g,
                                                                self.transModel.u[0].dof,
                                                                self.model.q['J'],
                                                                self.model.q['inverse(J)'],
                                                                np.asarray(self.gravity, 'd'),
                                                                self.q_rhog_consistent)

    def initializeMesh(self, mesh):
        self.elementBoundaryElements = mesh.elementBoundaryElementsArray
        self.exteriorElementBoundaries = mesh.exteriorElementBoundariesArray

    def attachModels(self, modelList):
        self.model = modelList[self.flowModelIndex]
        if self.T_model is not None:
            self.transModel = modelList[self.transModelIndex]
            self.q_mf = self.transModel.q[('u', 0)]
            self.q_mfdt = self.transModel.q[('mt', 0)]
            self.ebqe_mf = self.transModel.ebqe[('u', 0)]
            self.ebqe_mfdt = np.zeros(self.ebqe_mf.shape, 'd')
            if ('u', 0) in self.transModel.ebq:
                self.ebq_mf = self.transModel.ebq[('u', 0)]
                self.ebq_mfdt = np.zeros(self.ebq_mf.shape, 'd')
            if self.consistentVelocityFlag != 0:
                self.q_rhog_consistent = np.zeros((self.q_mf.shape[0], self.nd), 'd')

    def evaluate(self, t, c):
        if self.q_mf is not None and c[('u', 0)].shape == self.q_mf.shape:
            mf, mfdt, b_flag = self.q_mf, self.q_mfdt, 0
        elif self.ebqe_mf is not None and c[('u', 0)].shape == self.ebqe_mf.shape:
            mf, mfdt, b_flag = self.ebqe_mf, self.ebqe_mfdt, 1
        elif self.ebq_mf is not None and c[('u', 0)].shape == self.ebq_mf.shape:
            # as in 2012: element-boundary points have no c_t, so the coefficients are left unset
            return
        else:
            raise RuntimeError("no mass fraction for points of shape %s" % (c[('u', 0)].shape,))
        nd = self.nd
        a = c[('a', 0, 0)]
        assert a.shape[-1] == nd * nd, "expected full (non-sparse) diffusion storage"
        density = density_eval(mf, self.epsilon, self.rho_fw)
        logEvent("gwvd flow: t=%g b_flag=%d max|c_t|=%g" % (t, b_flag, np.abs(mfdt).max()), level=5)
        c[('m', 0)][:] = c[('u', 0)] * self.S * density  # replaced by the time integrator
        c[('dm', 0, 0)][:] = self.S * density
        c[('r', 0)][:] = self.drho_dmf * mfdt / density
        K = [self.Kx, self.Ky, self.Kz][:nd]
        denom = self.mu if self.variableViscosityFlag == 0 else var_visc(mf, self.mf_s)
        a.fill(0.0)
        for I in range(nd):
            a[..., I * nd + I] = K[I] * self.gc * self.rho_fw * density / denom
        f, df = c[('f', 0)], c[('df', 0, 0)]
        df.fill(0.0)
        if self.consistentVelocityFlag == 1:
            if b_flag == 0:
                rhog = self.q_rhog_consistent[:, np.newaxis, :]
            else:
                eN = self.elementBoundaryElements[self.exteriorElementBoundaries, 0]
                rhog = self.q_rhog_consistent[eN][:, np.newaxis, :]
            A = a.reshape(a.shape[:-1] + (nd, nd))
            f[:] = -np.einsum('...ij,...j->...i', A, np.broadcast_to(rhog, f.shape))
        else:
            f.fill(0.0)
            f[..., nd - 1] = -a[..., (nd - 1) * nd + nd - 1] * self.epsilon * mf


class VDGroundwater_mod2_TransportCoefficients(TransportCoefficients.TC_base):
    """
    Mass-fraction transport in advective (Hamilton-Jacobi) form driven by the flow model's mass flux v:

      omega rho c_t + v . grad c - div(a grad c) = 0,
      a = rho d omega I + alpha_T |v| I + (alpha_L - alpha_T) v v^T / |v|.
    """

    def __init__(self, nc=1, nd=2, omega=0.3, alpha_L=1.0, alpha_T=0.2, rho_fw=1000, d=1.3e-9,
                 epsilon=0.0, rho_source=0.0, mf_source=0.0, flowModelIndex=0, transModelIndex=1,
                 cModelIndex=2, C_Model=None, F_Model=0, getInterface=None, Height=1.0):
        self.nd = nd
        self.omega = omega
        self.nc = nc
        self.d = d
        self.alpha_L = alpha_L
        self.alpha_T = alpha_T
        self.rho_fw = rho_fw
        self.epsilon = epsilon
        self.rho_source = rho_source
        self.mf_source = mf_source
        self.F_Model = F_Model
        self.C_Model = C_Model
        self.flowModelIndex = flowModelIndex
        self.transModelIndex = transModelIndex
        self.cModelIndex = cModelIndex
        self.getInterface = getInterface
        self.Height = Height
        if C_Model is not None:
            raise NotImplementedError("surface-water coupling (C_Model) is not ported")
        mass, advection, diffusion, potential, reaction, hamiltonian = {}, {}, {}, {}, {}, {}
        for i in range(nc):
            mass[i] = {i: 'linear'}
            advection[i] = {i: 'linear'}
            diffusion[i] = {i: {i: 'constant'}}
            potential[i] = {i: 'u'}
            reaction[i] = {i: 'linear'}
            hamiltonian[i] = {i: 'linear'}
        TransportCoefficients.TC_base.__init__(self, nc, mass, advection, diffusion, potential,
                                               reaction, hamiltonian, variableNames=['mf'])

    def attachModels(self, modelList):
        self.model = modelList[self.transModelIndex]
        if self.F_Model is not None:
            flow = modelList[self.flowModelIndex]
            self.flowModel = flow
            self.q_v = flow.q[('velocity', 0)]
            self.ebq_v = flow.ebq.get(('velocity', 0))
            self.ebqe_v = flow.ebqe[('velocity', 0)]
            self.ebq_global_v = flow.ebq_global.get(('velocity', 0))

    def evaluate(self, t, c):
        shape = c[('df', 0, 0)].shape
        for v in (self.q_v, self.ebq_v, self.ebqe_v, self.ebq_global_v):
            if v is not None and v.shape == shape:
                break
        else:
            raise RuntimeError("no flow velocity for points of shape %s" % (shape,))
        nd = self.nd
        a = c[('a', 0, 0)]
        assert a.shape[-1] == nd * nd, "expected full (non-sparse) diffusion storage"
        u = c[('u', 0)]
        density = density_eval(u, self.epsilon, self.rho_fw)
        c[('m', 0)][:] = u  # replaced by NonConservativeBackwardEuler: m_t = dm (u - u_last)/dt
        c[('dm', 0, 0)][:] = self.omega * density
        c[('r', 0)].fill(0.0)
        c[('dr', 0, 0)].fill(0.0)
        c[('f', 0)].fill(0.0)
        c[('df', 0, 0)].fill(0.0)
        c[('H', 0)][:] = np.einsum('...i,...i->...', c[('grad(u)', 0)], v)
        c[('dH', 0, 0)][:] = v
        norm_v = np.sqrt(np.einsum('...i,...i->...', v, v))
        safe = np.where(norm_v > 0.0, norm_v, 1.0)
        a.fill(0.0)
        for I in range(nd):
            a[..., I * nd + I] = (density * self.d * self.omega
                                  + np.where(norm_v > 0.0,
                                             self.alpha_T * norm_v
                                             + (self.alpha_L - self.alpha_T) * v[..., I]**2 / safe, 0.0))
            for J in range(I + 1, nd):
                off = np.where(norm_v > 0.0, (self.alpha_L - self.alpha_T) * v[..., I] * v[..., J] / safe, 0.0)
                a[..., I * nd + J] = off
                a[..., J * nd + I] = off
