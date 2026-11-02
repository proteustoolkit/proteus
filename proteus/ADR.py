"""
An optimized Advection-Diffusion-Reaction module
"""
import numpy as np
import os
from math import fabs
import proteus
from proteus import  cfemIntegrals, Quadrature
from proteus.NonlinearSolvers import NonlinearEquation
from proteus.FemTools import (DOFBoundaryConditions,
                              FluxBoundaryConditions,
                              FiniteElementFunction,
                              C0_AffineLinearOnSimplexWithNodalBasis)
from proteus.Comm import globalMax, globalSum
from proteus.Profiling import  memory
from proteus.Profiling import  logEvent as log
from proteus.Transport import OneLevelTransport
from proteus.TransportCoefficients import TC_base
from proteus.SubgridError import SGE_base
from proteus.ShockCapturing import  ShockCapturing_base
from proteus.cADR import *
from proteus.mprans import cArgumentsDict

class SubgridError(SGE_base):
    """
    SubgridError approximation for ADR equations

    .. inheritance-diagram:: SubgridError
       :parts: 2
    """

    def __init__(self,coefficients,nd):
        SGE_base.__init__(self,coefficients,nd,lag=False)
    def initializeElementQuadrature(self,mesh,t,cq):
        pass
    def updateSubgridErrorHistory(self,initializationPhase=False):
        pass
    def calculateSubgridError(self,q):
        pass

class ShockCapturing(ShockCapturing_base):
    """
    Residual-based shock capturing for ADR equations

    .. inheritance-diagram:: ShockCapturing
       :parts: 2
    """
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True,nStepsToDelay=None):
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps=0
        if self.lag:
            log("ADR.ShockCapturing: lagging requested but must lag the first step; switching lagging off and delaying")
            self.nStepsToDelay=1
            self.lag=False
    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.numDiff=[]
        self.numDiff_last=[]
        for ci in range(self.nc):
            self.numDiff.append(cq[('numDiff',ci,ci)])
            self.numDiff_last.append(cq[('numDiff',ci,ci)])
    def updateShockCapturingHistory(self):
        self.nSteps += 1
        if self.lag:
            for ci in range(self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            log("ADR.ShockCapturing: switched to lagged shock capturing")
            self.lag = True
            self.numDiff_last=[]
            for ci in range(self.nc):
                self.numDiff_last.append(self.numDiff[ci].copy())
        #log("VOF: max numDiff %e" % (globalMax(self.numDiff_last[0].max()),))

class NumericalFlux_IIPG(proteus.NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior):
    def __init__(self,vt,getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions):
        proteus.NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior.__init__(self,vt,getPointwiseBoundaryConditions,
                                                                                        getAdvectiveFluxBoundaryConditions,
                                                                                        getDiffusiveFluxBoundaryConditions)

class NumericalFlux_SIPG(proteus.NumericalFlux.Advection_DiagonalUpwind_Diffusion_SIPG_exterior):
    def __init__(self,vt,getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions):
        proteus.NumericalFlux.Advection_DiagonalUpwind_Diffusion_SIPG_exterior.__init__(self,vt,getPointwiseBoundaryConditions,
                                                                                        getAdvectiveFluxBoundaryConditions,
                                                                                        getDiffusiveFluxBoundaryConditions)

class NumericalFlux_NIPG(proteus.NumericalFlux.Advection_DiagonalUpwind_Diffusion_NIPG_exterior):
    def __init__(self,vt,getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions):
        proteus.NumericalFlux.Advection_DiagonalUpwind_Diffusion_NIPG_exterior.__init__(self,vt,getPointwiseBoundaryConditions,
                                                                                        getAdvectiveFluxBoundaryConditions,
                                                                                        getDiffusiveFluxBoundaryConditions)

NumericalFlux = NumericalFlux_SIPG

class Coefficients(TC_base):
    """
    Coefficients of linear ADR equations

    .. inheritance-diagram:: Coefficients
       :parts: 2
    """
    from proteus.ctransportCoefficients import L2projectEvaluate
    def __init__(self,aOfX,fOfX,velocity=None,nc=1,nd=2,l2proj=None,
                 timeVaryingCoefficients=False,
                 forceStrongDirichlet=False,
                 useMetrics=0.0,
                 sc_uref=1.0,
                 sc_beta=1.0,
                 mua=1.0,
                 mub=1.0,
                 jf=0.0,
                 embeddedBoundary=False,
                 embeddedBoundary_penalty=100.0,
                 embeddedBoundary_ghost_penalty=0.1,
                 embeddedBoundary_sdf=None,
                 embeddedBoundary_u=None,
                 embeddedBoundary_u_inner=None,
                 embeddedBoundary_u_outer=None,
                 embeddedBoundary_robin=False,
                 embeddedBoundary_robin_h=1.0,
                 embeddedBoundary_robin_h_inner=None,
                 embeddedBoundary_robin_h_outer=None,
                 embeddedBoundary_robin_uInfinity_inner=None,
                 embeddedBoundary_robin_uInfinity_outer=None,
                 immersedBoundary=False,
                 immersedBoundary_penalty=100.0,
                 immersedSCIFEM_switch=0.0,
                 immersedSCIFEM_penalty=0.0,
                 PG=False,
                 immersedBoundary_sdf=None,
                 immersedBoundary_u=None,
                 immersedBoundary_fluxJump=None,
                 immersedBoundary_fluxJumpVector=None,
                 immersedBoundary_solutionJump=None,
                 immersedBoundary_fluxJumpLift=True,
                 analyticalSolution=None,
                 test=1.0):
        self.test = test
        self.analyticalSolution = analyticalSolution
        self.embeddedBoundary=embeddedBoundary
        self.embeddedBoundary_penalty=embeddedBoundary_penalty
        self.embeddedBoundary_ghost_penalty=embeddedBoundary_ghost_penalty
        self.embeddedBoundary_sdf=embeddedBoundary_sdf
        self.embeddedBoundary_u=embeddedBoundary_u
        # Per-phase weak-Dirichlet targets on the solid surface. On a triply-cut element the
        # solid boundary has a water piece and an air piece, and the equivalent-polynomial
        # quadrature is only exact against polynomials over the WHOLE element. A single
        # pointwise u_s switches branch across Gamma_f, so it is not a polynomial and the
        # moment fit does not apply. Each side instead gets its own branch, analytically
        # continued over the whole element, which makes (u_side - u_s_side) identically zero
        # for the exact solution and the Nitsche terms exact.
        self.embeddedBoundary_u_inner=embeddedBoundary_u_inner
        self.embeddedBoundary_u_outer=embeddedBoundary_u_outer
        # Robin (convective) alternative to the Nitsche-Dirichlet boundary above: instead of
        # weakly enforcing u = embeddedBoundary_u, impose the natural law
        #   mu * du/dn = -h * (u - u_infinity)      on Gamma_s,
        # where n is the fluid's own outward normal (into the solid). This is already a
        # natural boundary condition -- nothing is being constrained -- so unlike Dirichlet it
        # needs no consistency/symmetry/penalty construction, only a single reaction-like term
        # (see updateEmbeddedBoundaryRobinTerms in ADR.h).
        #
        # u_infinity gets its OWN fields (embeddedBoundary_robin_uInfinity*), deliberately not
        # reusing embeddedBoundary_u/_u_inner/_u_outer: those are also read directly, by name,
        # in initializeSDF below to pin a masked triply-cut node's raw DOF to the TRUE exact
        # solution there -- a mechanism with nothing to do with Gamma_s's boundary condition
        # type. Reusing them for u_infinity would silently repoint that unrelated pinning at
        # the wrong value whenever Robin is on.
        #
        # h itself is taken as a phase constant (the physically standard treatment of a
        # convective coefficient), not a spatially varying field: embeddedBoundary_robin_h_inner
        # is used on the mua (water) side and _h_outer on the mub (air) side, selected the same
        # way mua/mub themselves are (D_s_a/D_s_b on a triply-cut element, icase_f's sign
        # elsewhere); both default to the single embeddedBoundary_robin_h if not given.
        # embeddedBoundary_robin_uInfinity_inner/_outer are selected the same way, but (unlike
        # h) have no single-field fallback -- see the note just above __init__'s assignment.
        self.embeddedBoundary_robin=embeddedBoundary_robin
        self.embeddedBoundary_robin_h=embeddedBoundary_robin_h
        self.embeddedBoundary_robin_h_inner = (embeddedBoundary_robin_h_inner
            if embeddedBoundary_robin_h_inner is not None else embeddedBoundary_robin_h)
        self.embeddedBoundary_robin_h_outer = (embeddedBoundary_robin_h_outer
            if embeddedBoundary_robin_h_outer is not None else embeddedBoundary_robin_h)
        # No single "generic" uInfinity: unlike embeddedBoundary_u (whose masked-DOF role
        # forces a value everywhere), a Robin target is only ever consumed per-phase -- via
        # D_s_a/D_s_b on a triply-cut element, via icase_f's sign otherwise -- exactly like
        # h_inner/h_outer above, so there is nothing for a combined field to do.
        self.embeddedBoundary_robin_uInfinity_inner = embeddedBoundary_robin_uInfinity_inner
        self.embeddedBoundary_robin_uInfinity_outer = embeddedBoundary_robin_uInfinity_outer
        if self.embeddedBoundary:
            assert(self.embeddedBoundary_sdf is not None)
            assert(self.embeddedBoundary_u is not None)
        self.immersedBoundary=immersedBoundary
        self.immersedBoundary_penalty=immersedBoundary_penalty
        self.immersedSCIFEM_switch=immersedSCIFEM_switch
        self.immersedSCIFEM_penalty=immersedSCIFEM_penalty
        self.PG=PG
        self.immersedBoundary_sdf=immersedBoundary_sdf
        self.immersedBoundary_u=immersedBoundary_u
        # Prescribed interface jump data, supplied by the physics file instead of
        # being hardcoded per test number in ADR.h:
        #   flux jump      [beta du/dn] = immersedBoundary_fluxJump(x,t)
        #                                 + immersedBoundary_fluxJumpVector(x,t) . n
        #   solution jump  [u]          = immersedBoundary_solutionJump(x,t)
        # All default to zero, i.e. a continuous interface with continuous flux.
        self.immersedBoundary_fluxJump=immersedBoundary_fluxJump
        self.immersedBoundary_fluxJumpVector=immersedBoundary_fluxJumpVector
        self.immersedBoundary_solutionJump=immersedBoundary_solutionJump
        # Flux-jump lift: the IFE basis is built for a homogeneous flux jump, and a prescribed
        # [beta du/dn] = J_f is represented by adding J_f*psi once to the trial function (psi: the
        # basis constructor's solution with zero nodal values and unit flux jump), like the
        # solution-jump lift. The load term -int J_f v stays. False reproduces the load-only scheme.
        self.immersedBoundary_fluxJumpLift=immersedBoundary_fluxJumpLift
        if self.immersedBoundary:
            assert(self.immersedBoundary_sdf is not None)
            assert(self.immersedBoundary_u is not None)
        self.useMetrics = useMetrics
        self.forceStrongDirichlet=forceStrongDirichlet
        self.aOfX = aOfX
        self.fOfX = fOfX
        self.velocity=velocity
        self.mua = mua
        self.mub = mub
        self.jf = jf
        self.nd = nd
        self.l2proj = l2proj
        self.timeVaryingCoefficients=timeVaryingCoefficients
        self.sc_uref=sc_uref
        self.sc_beta=sc_beta
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction  = {}
        hamiltonian = {}
        for i in range(nc):
            diffusion[i] = {i : {i:'constant'}}
            reaction[i]  = {i : 'constant'}
            advection[i] = {i : 'linear'} #now include for gravity type terms
            potential[i] = {i : 'u'}
        #end i
        sdInfo = {(0,0):(np.arange(start=0,stop=self.nd**2+1,step=self.nd,dtype='i'),
                         np.array([range(self.nd) for row in range(self.nd)],dtype='i'))}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames=['u'],
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion=True,
                         movingDomain=False)
    def initializeSDF(self,femSpace):
        nodeArray = femSpace.mesh.nodeArray if (femSpace.max_nDOF_element==3) else femSpace.dofMap.lagrangeNodesArray
        self.embeddedBoundary_sdf_nodes = 100*np.ones((nodeArray.shape[0],),'d')
        if self.embeddedBoundary:
            for nN in range(nodeArray.shape[0]):
                self.embeddedBoundary_sdf_nodes[nN], dummy_normal = self.embeddedBoundary_sdf(t=0.0,x=nodeArray[nN])
        # Nudge nodal values off the exact-zero band. equivalent_polynomials classifies any
        # |phi| <= 1e-8 as "node sits on the interface", which sends the element down a
        # degenerate branch where the Dirac is hard-set to zero -- silently dropping the solid
        # Nitsche condition on the one element that most needs it (edge == +1: the fluid element
        # whose edge IS the solid boundary). Pushing such nodes just inside the solid turns the
        # element into an ordinary, if thin, cut cell that the normal machinery handles. The
        # geometric change is ~1e-7, far below discretization error, and it is done once per node
        # so neighbouring elements always agree on where the interface is.
        _EPS_LS = 1.0e-8
        _degenerate = np.abs(self.embeddedBoundary_sdf_nodes) <= _EPS_LS
        if _degenerate.any():
            self.embeddedBoundary_sdf_nodes[_degenerate] = -10.0*_EPS_LS
        self.immersedBoundary_sdf_nodes = -100*np.ones((nodeArray.shape[0],),'d')
        if self.immersedBoundary:
            for nN in range(nodeArray.shape[0]):
                self.immersedBoundary_sdf_nodes[nN], dummy_normal = self.immersedBoundary_sdf(t=0.0,x=nodeArray[nN])
        # [u] at the DOF locations: the kernel builds JA/JB from per-node values, so the
        # jump must be sampled where the DOFs live, not at quadrature points.
        self.immersedBoundary_solutionJump_nodes = np.zeros((nodeArray.shape[0],),'d')
        if self.immersedBoundary and self.immersedBoundary_solutionJump is not None:
            for nN in range(nodeArray.shape[0]):
                self.immersedBoundary_solutionJump_nodes[nN] = self.immersedBoundary_solutionJump(t=0.0,x=nodeArray[nN])
        # Prescribed flux jump of each Gamma_f-cut cell, for the flux-jump lift: J_f at the midpoint
        # of the cell's Gamma_f segment (zero of the linear interpolant of the vertex values), the
        # vector part dotted with the segment normal (from phi<0 to phi>0). Exact for a constant J_f.
        nElements_f = femSpace.mesh.elementNodesArray.shape[0]
        self.immersedBoundary_fluxJump_element = np.zeros((nElements_f,),'d')
        if (self.immersedBoundary and self.immersedBoundary_fluxJumpLift and
                (self.immersedBoundary_fluxJump is not None or self.immersedBoundary_fluxJumpVector is not None)):
            l2g_f = femSpace.dofMap.l2g
            for eN in range(nElements_f):
                g = l2g_f[eN,:3]
                phi = self.immersedBoundary_sdf_nodes[g]
                if not ((phi < 0.0).any() and (phi > 0.0).any()):
                    continue
                X = nodeArray[g]
                cuts = [X[i] + phi[i]/(phi[i] - phi[j])*(X[j] - X[i])
                        for i, j in ((0, 1), (1, 2), (2, 0)) if phi[i]*phi[j] < 0.0]
                if len(cuts) != 2:
                    continue
                xm = 0.5*(cuts[0] + cuts[1])
                J = 0.0
                if self.immersedBoundary_fluxJump is not None:
                    J += self.immersedBoundary_fluxJump(t=0.0,x=xm)
                if self.immersedBoundary_fluxJumpVector is not None:
                    A = np.array([[X[1,0]-X[0,0], X[1,1]-X[0,1]], [X[2,0]-X[0,0], X[2,1]-X[0,1]]])
                    grad = np.linalg.solve(A, np.array([phi[1]-phi[0], phi[2]-phi[0]]))
                    n = np.zeros(3); n[:2] = grad/np.linalg.norm(grad)
                    J += np.dot(self.immersedBoundary_fluxJumpVector(t=0.0,x=xm), n)
                self.immersedBoundary_fluxJump_element[eN] = J
        # Triply-cut (P1 only): a masked node's raw DOF is physically undefined (u_ex isn't
        # defined inside the solid), but gf_f's two-sided basis reads it directly regardless.
        # Fix: substitute the analytic embeddedBoundary_u_inner/_outer value at that node's
        # coordinates. triplyCut_maskedLocalNode/overrideValue record one masked node per
        # element for the historical residual-only patch in ADR.h; the real fix, for one or
        # two masked nodes, is triplyCut_globalDofOverrides, pinned via forceStrongDirichlet.
        nElements = femSpace.mesh.elementNodesArray.shape[0]
        self.triplyCut_maskedLocalNode = -np.ones((nElements,),'i')
        self.triplyCut_overrideValue = np.zeros((nElements,),'d')
        self.triplyCut_globalDofOverrides = {}
        if (self.embeddedBoundary and self.immersedBoundary and femSpace.max_nDOF_element == 3
                and self.embeddedBoundary_u_inner is not None
                and self.embeddedBoundary_u_outer is not None
                and os.environ.get('ADR_DISABLE_TC') != '1'):
            l2g = femSpace.dofMap.l2g
            _watch = int(os.environ['ADR_DEBUG_NODE']) if os.environ.get('ADR_DEBUG_NODE') else None
            for eN in range(nElements):
                local_g = l2g[eN,:3]
                local_phi_s = self.embeddedBoundary_sdf_nodes[local_g]
                masked = np.where(local_phi_s < 0.0)[0]
                if _watch is not None and _watch in local_g:
                    print(f"[DEBUG_NODE] eN={eN} local_g={local_g} phi_s={local_phi_s} "
                          f"nMasked={len(masked)} masked_locals={masked}")
                _allowed = (1,) if os.environ.get('ADR_TC_G1_ONLY') == '1' else (1, 2)
                if len(masked) not in _allowed:
                    continue  # fully solid, fully fluid, or (never for P1) all 3 masked
                self.triplyCut_maskedLocalNode[eN] = int(masked[0])
                for m in masked:
                    m = int(m)
                    gN = int(local_g[m])
                    xN = nodeArray[gN]
                    if self.immersedBoundary_sdf_nodes[gN] < 0.0:
                        override = self.embeddedBoundary_u_inner(t=0.0,x=xN)
                    else:
                        override = self.embeddedBoundary_u_outer(t=0.0,x=xN)
                    if m == self.triplyCut_maskedLocalNode[eN]:
                        self.triplyCut_overrideValue[eN] = override
                    self.triplyCut_globalDofOverrides[gN] = override
        if os.environ.get('ADR_DEBUG_TC'):
            _n = int((self.triplyCut_maskedLocalNode >= 0).sum())
            print(f"[triplyCut] elements flagged: {_n} / {nElements}  "
                  f"embeddedBoundary={self.embeddedBoundary} immersedBoundary={self.immersedBoundary} "
                  f"max_nDOF_element={femSpace.max_nDOF_element}")
            l2g_dbg = femSpace.dofMap.l2g
            _genuine = []
            for eN in range(nElements):
                lg = l2g_dbg[eN,:3]
                ps = self.embeddedBoundary_sdf_nodes[lg]
                pf = self.immersedBoundary_sdf_nodes[lg]
                if (ps < 0).any() and (ps > 0).any() and (pf < 0).any() and (pf > 0).any():
                    _genuine.append((eN, int((ps < 0).sum()), self.triplyCut_maskedLocalNode[eN]))
            print(f"[triplyCut] genuinely triply-cut elements (mixed phi_s AND mixed phi_f): {len(_genuine)}")
            _pinned_global_dofs = set()
            for eN in range(nElements):
                m = self.triplyCut_maskedLocalNode[eN]
                if m >= 0:
                    _pinned_global_dofs.add(int(l2g_dbg[eN, m]))
            for eN, nmasked, flagged in _genuine:
                gN_dbg = l2g_dbg[eN, flagged] if flagged >= 0 else -1
                coords_dbg = nodeArray[gN_dbg] if flagged >= 0 else None
                print(f"  eN={eN} nMaskedNodes={nmasked} maskedLocalNode(flagged)={flagged} "
                      f"globalNode={gN_dbg} coords={coords_dbg}")
                if nmasked == 2:
                    lg = l2g_dbg[eN, :3]
                    ps = self.embeddedBoundary_sdf_nodes[lg]
                    for _li in np.where(ps < 0)[0]:
                        _gN = int(lg[_li])
                        print(f"    -> G3/G4 masked local node {_li} = global dof {_gN}, "
                              f"coords={nodeArray[_gN]}, pinned via a neighbor={_gN in _pinned_global_dofs}")
    def initializeElementQuadrature(self,t,cq):
        nd = self.nd
        for ci in range(self.nc):
            if ('df',ci,ci) in cq:
                if self.velocity is not None:
                    cq[('df',ci,ci)][...,:] = self.velocity
                else:
                    cq[('df',ci,ci)].flat[:] = 0.0
            for i in range(len(cq[('r',ci)].flat)):
                cq[('r',ci)].flat[i] = -self.fOfX[ci](cq['x'].flat[3*i:3*(i+1)])
                cq[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[ci](cq['x'].flat[3*i:3*(i+1)]).flat
        cq['embeddedBoundary_sdf'] = 100*np.ones_like(cq[('u',0)])
        cq['embeddedBoundary_normal'] = np.ones_like(cq['x'])
        cq['embeddedBoundary_u'] = np.ones_like(cq[('u',0)])
        cq['embeddedBoundary_u_inner'] = np.ones_like(cq[('u',0)])
        cq['embeddedBoundary_u_outer'] = np.ones_like(cq[('u',0)])
        cq['embeddedBoundary_robin_uInfinity_inner'] = np.ones_like(cq[('u',0)])
        cq['embeddedBoundary_robin_uInfinity_outer'] = np.ones_like(cq[('u',0)])
        if self.embeddedBoundary:
            for eN in range(cq['embeddedBoundary_sdf'].shape[0]):
                for k in range(cq['embeddedBoundary_sdf'].shape[1]):
                    cq['embeddedBoundary_sdf'][eN,k],cq['embeddedBoundary_normal'][eN,k] = self.embeddedBoundary_sdf(t=0.0,x=cq['x'][eN,k])
                    cq['embeddedBoundary_u'][eN,k] = self.embeddedBoundary_u(t=0.0,x=cq['x'][eN,k])
                    if self.embeddedBoundary_u_inner is not None:
                        cq['embeddedBoundary_u_inner'][eN,k] = self.embeddedBoundary_u_inner(t=0.0,x=cq['x'][eN,k])
                    else:
                        cq['embeddedBoundary_u_inner'][eN,k] = cq['embeddedBoundary_u'][eN,k]
                    if self.embeddedBoundary_u_outer is not None:
                        cq['embeddedBoundary_u_outer'][eN,k] = self.embeddedBoundary_u_outer(t=0.0,x=cq['x'][eN,k])
                    else:
                        cq['embeddedBoundary_u_outer'][eN,k] = cq['embeddedBoundary_u'][eN,k]
                    if self.embeddedBoundary_robin:
                        if self.embeddedBoundary_robin_uInfinity_inner is not None:
                            cq['embeddedBoundary_robin_uInfinity_inner'][eN,k] = \
                                self.embeddedBoundary_robin_uInfinity_inner(t=0.0,x=cq['x'][eN,k])
                        if self.embeddedBoundary_robin_uInfinity_outer is not None:
                            cq['embeddedBoundary_robin_uInfinity_outer'][eN,k] = \
                                self.embeddedBoundary_robin_uInfinity_outer(t=0.0,x=cq['x'][eN,k])
        cq['immersedBoundary_sdf'] = -100*np.ones_like(cq[('u',0)])
        cq['immersedBoundary_normal'] = -np.ones_like(cq['x'])
        cq['immersedBoundary_u'] = np.ones_like(cq[('u',0)])
        if self.immersedBoundary:
            for eN in range(cq['immersedBoundary_sdf'].shape[0]):
                for k in range(cq['immersedBoundary_sdf'].shape[1]):
                    cq['immersedBoundary_sdf'][eN,k],cq['immersedBoundary_normal'][eN,k] = self.immersedBoundary_sdf(t=0.0,x=cq['x'][eN,k])
                    cq['immersedBoundary_u'][eN,k] = self.immersedBoundary_u(t=0.0,x=cq['x'][eN,k])
        # Exact solution at quadrature points, reused from the analyticalSolution
        # supplied by the physics (p) file instead of being redefined in C++.
        # Cut elements need both the "inner"/"outer" branch values at the same
        # physical point, so uOfX_inner/uOfX_outer (raw branch formulas) are
        # queried directly when the analytical solution class provides them.
        cq['immersedBoundary_fluxJump'] = np.zeros_like(cq[('u',0)])
        cq['immersedBoundary_fluxJumpVector'] = np.zeros_like(cq['x'])
        if self.immersedBoundary:
            for eN in range(cq['x'].shape[0]):
                for k in range(cq['x'].shape[1]):
                    xk = cq['x'][eN,k]
                    if self.immersedBoundary_fluxJump is not None:
                        cq['immersedBoundary_fluxJump'][eN,k] = self.immersedBoundary_fluxJump(t=0.0,x=xk)
                    if self.immersedBoundary_fluxJumpVector is not None:
                        cq['immersedBoundary_fluxJumpVector'][eN,k] = self.immersedBoundary_fluxJumpVector(t=0.0,x=xk)
        cq[('u_exact_inner',0)] = np.zeros_like(cq[('u',0)])
        cq[('u_exact_outer',0)] = np.zeros_like(cq[('u',0)])
        if self.analyticalSolution is not None and self.analyticalSolution.get(0) is not None:
            ans0 = self.analyticalSolution[0]
            has_split = hasattr(ans0,'uOfX_inner') and hasattr(ans0,'uOfX_outer')
            for eN in range(cq['x'].shape[0]):
                for k in range(cq['x'].shape[1]):
                    xk = cq['x'][eN,k]
                    if has_split:
                        cq[('u_exact_inner',0)][eN,k] = ans0.uOfX_inner(xk)
                        cq[('u_exact_outer',0)][eN,k] = ans0.uOfX_outer(xk)
                    else:
                        cq[('u_exact_inner',0)][eN,k] = ans0.uOfX(xk)
                        cq[('u_exact_outer',0)][eN,k] = cq[('u_exact_inner',0)][eN,k]

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        nd = self.nd
        for c in [cebq,cebq_global]:
            for ci in range(self.nc):
                if ('df',ci,ci) in c:
                    if self.velocity is not None:
                        c[('df',ci,ci)][...,:] = self.velocity
                    else:
                        c[('df',ci,ci)].flat[:] = 0.0
                if ('r',ci) in c and ('a',ci,ci)in c:
                    for i in range(len(c[('u',ci)].flat)):
                        c[('r',ci)].flat[i] = -self.fOfX[ci](c['x'].flat[3*i:3*(i+1)])
                        c[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[ci](c['x'].flat[3*i:3*(i+1)]).flat

    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        nd = self.nd
        for c in [cebqe]:
            for ci in range(self.nc):
                if ('df',ci,ci)in c:
                    if self.velocity is not None:
                        c[('df',ci,ci)][...,:] = self.velocity
                    else:
                        c[('df',ci,ci)].flat[:] = 0.0
                if ('r',ci) in c and ('a',ci,ci) in c:
                    for i in range(len(c[('u',ci)].flat)):
                        c[('r',ci)].flat[i] = -self.fOfX[ci](c['x'].flat[3*i:3*(i+1)])
                        c[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[ci](c['x'].flat[3*i:3*(i+1)]).flat

    def evaluate(self,t,c):
        if self.timeVaryingCoefficients:
            nd = self.nd
            for ci in range(self.nc):
                if self.velocity is not None:
                    c[('df',ci,ci)][...,:] = self.velocity
                else:
                    c[('df',ci,ci)].flat[:] = 0.0
                for i in range(len(c[('r',ci)].flat)):
                    c[('r',ci)].flat[i] = -self.fOfX[ci](c['x'].flat[3*i:3*(i+1)])
                    c[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[ci](c['x'].flat[3*i:3*(i+1)]).flat

class LevelModel(proteus.Transport.OneLevelTransport):
    """
    Optimized LevelModel for ADR equations

    .. inheritance-diagram:: LevelModel
       :parts: 2
    """
    nCalls=0
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
                 movingDomain=False):
        if coefficients.embeddedBoundary or coefficients.immersedBoundary:
            self.hasCutCells=True
        # Tells the C++ layer to (re)build its per-element cache of the
        # equivalent-polynomial/IFEM reconstruction (cut classification, basis
        # coefficients, H/ImH/D, VA/VB) on the next residual/Jacobian call, then
        # reuse it until this is set True again. Starts True so the cache is
        # built on first use. For a steady problem with a fixed interface this
        # never needs to be set again. For a future moving-interface/unsteady
        # problem, call invalidateIFEMGeometry() whenever the embedded/immersed
        # sdf is re-evaluated (e.g. at the start of each new time step).
        self.recompute_ifem_geometry = True
        from proteus import Comm
        #
        #set the objects describing the method and boundary conditions
        #
        self.movingDomain=movingDomain
        self.tLast_mesh=None
        #
        self.name=name
        self.sd=sd
        self.Hess=False
        self.lowmem=True
        self.timeTerm=True#allow turning off  the  time derivative
        #self.lowmem=False
        self.testIsTrial=True
        self.phiTrialIsTrial=True
        self.u = uDict
        self.ua = {}#analytical solutions
        self.phi  = phiDict
        self.dphi={}
        self.matType = matType
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature#True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1,coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        ## Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh #assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList=None #explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.coefficients = coefficients
        self.coefficients.initializeSDF(self.u[0].femSpace)
        # Level sets as ordinary finite element functions, purely so they land in the
        # h5/xmf archive next to the solution and can be contoured in ParaView. The
        # nodal sdf arrays built by initializeSDF are already indexed the same way as
        # the DOFs of this space, so they can be used as dof arrays directly.
        self.levelSetFunctions = []
        for _name, _dof, _active in [
                ('phi_s', self.coefficients.embeddedBoundary_sdf_nodes,
                 self.coefficients.embeddedBoundary),
                ('phi_f', self.coefficients.immersedBoundary_sdf_nodes,
                 self.coefficients.immersedBoundary)]:
            if _active:
                self.levelSetFunctions.append(
                    FiniteElementFunction(self.u[0].femSpace,
                                                   dof=_dof,
                                                   name=_name))
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict #no velocity post-processing for now
        self.fluxBoundaryConditions=fluxBoundaryConditionsDict
        self.advectiveFluxBoundaryConditionsSetterDict=advectiveFluxBoundaryConditionsSetterDict
        self.diffusiveFluxBoundaryConditionsSetterDictDict = diffusiveFluxBoundaryConditionsSetterDictDict
        #determine whether  the stabilization term is nonlinear
        self.stabilizationIsNonlinear = False
        #cek come back
        if self.stabilization is not None:
            for ci in range(self.nc):
                if ci in coefficients.mass:
                    for flag in coefficients.mass[ci].values():
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if  ci in coefficients.advection:
                    for  flag  in coefficients.advection[ci].values():
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if  ci in coefficients.diffusion:
                    for diffusionDict in coefficients.diffusion[ci].values():
                        for  flag  in diffusionDict.values():
                            if flag != 'constant':
                                self.stabilizationIsNonlinear=True
                if  ci in coefficients.potential:
                     for flag in coefficients.potential[ci].values():
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if ci in coefficients.reaction:
                    for flag in coefficients.reaction[ci].values():
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if ci in coefficients.hamiltonian:
                    for flag in coefficients.hamiltonian[ci].values():
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
        #determine if we need element boundary storage
        self.elementBoundaryIntegrals = {}
        for ci  in range(self.nc):
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux is not None) or
                                                 (numericalFluxType is not None) or
                                                 (self.fluxBoundaryConditions[ci] == 'outFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'mixedFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        #calculate some dimensions
        #
        self.nSpace_global    = self.u[0].femSpace.nSpace_global #assume same space dim for all variables
        self.nDOF_trial_element     = [u_j.femSpace.max_nDOF_element for  u_j in self.u.values()]
        self.nDOF_phi_trial_element     = [phi_k.femSpace.max_nDOF_element for  phi_k in self.phi.values()]
        self.n_phi_ip_element = [phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for  phi_k in self.phi.values()]
        self.nDOF_test_element     = [femSpace.max_nDOF_element for femSpace in self.testSpace.values()]
        self.nFreeDOF_global  = [dc.nFreeDOF_global for dc in self.dirichletConditions.values()]
        self.nVDOF_element    = sum(self.nDOF_trial_element)
        self.nFreeVDOF_global = sum(self.nFreeDOF_global)
        #
        NonlinearEquation.__init__(self,self.nFreeVDOF_global)
        #
        #build the quadrature point dictionaries from the input (this
        #is just for convenience so that the input doesn't have to be
        #complete)
        #
        elementQuadratureDict={}
        elemQuadIsDict = isinstance(elementQuadrature,dict)
        if elemQuadIsDict: #set terms manually
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
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature
        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff',ci,ci) in elementQuadrature:
                        elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature[('numDiff',ci,ci)]
                    else:
                        elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature
        if massLumping:
            for ci in self.coefficients.mass.keys():
                elementQuadratureDict[('m',ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',)+I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
        if reactionLumping:
            for ci in self.coefficients.mass.keys():
                elementQuadratureDict[('r',ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',)+I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
        elementBoundaryQuadratureDict={}
        if isinstance(elementBoundaryQuadrature,dict): #set terms manually
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
        (self.elementQuadraturePoints,self.elementQuadratureWeights,
         self.elementQuadratureRuleIndeces) = Quadrature.buildUnion(elementQuadratureDict)
        self.nQuadraturePoints_element = self.elementQuadraturePoints.shape[0]
        self.nQuadraturePoints_global = self.nQuadraturePoints_element*self.mesh.nElements_global
        #
        #Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints,
         self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[0]
        self.nElementBoundaryQuadraturePoints_global = (self.mesh.nElements_global*
                                                        self.mesh.nElementBoundaries_element*
                                                        self.nElementBoundaryQuadraturePoints_elementBoundary)
        #
        #simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q={}
        self.ebq={}
        self.ebq_global={}
        self.ebqe={}
        self.phi_ip={}
        #mesh
        self.q['x'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,3),'d')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        self.q[('u',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('grad(u)',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('a',0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.coefficients.sdInfo[(0,0)][0][-1]),'d')
        self.q[('df',0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('r',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('cfl',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('numDiff',0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')

        self.ebqe['penalty'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('u',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('diffusiveFlux_bc_flag',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('diffusiveFlux_bc',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('advectiveFlux_bc_flag',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('advectiveFlux_bc',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('grad(u)',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        self.ebqe[('a',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.coefficients.sdInfo[(0,0)][0][-1]),'d')
        self.ebqe[('df',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        self.ebqe[('r',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')

        self.points_elementBoundaryQuadrature= set()
        self.scalars_elementBoundaryQuadrature= set([('u',ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature= set()
        self.tensors_elementBoundaryQuadrature= set()
        log(memory("element and element boundary Jacobians","OneLevelTransport"),level=4)
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
             self.inflowBoundaryBC[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,),'i')
             self.inflowBoundaryBC_values[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nDOF_trial_element[cj]),'d')
             self.inflowFlux[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.internalNodes = set(range(self.mesh.nNodes_global))
        #identify the internal nodes this is ought to be in mesh
        ##\todo move this to mesh
        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            for i in range(self.mesh.nNodes_element):
                if i != ebN_element:
                    I = self.mesh.elementNodesArray[eN_global,i]
                    self.internalNodes -= set([I])
        self.nNodes_internal = len(self.internalNodes)
        self.internalNodesArray=np.zeros((self.nNodes_internal,),'i')
        for nI,n in enumerate(self.internalNodes):
            self.internalNodesArray[nI]=n
        #
        del self.internalNodes
        self.internalNodes = None
        log("Updating local to global mappings",2)
        self.updateLocal2Global()
        log("Building time integration object",2)
        log(memory("inflowBC, internalNodes,updateLocal2Global","OneLevelTransport"),level=4)
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(self,integrateInterpolationPoints=True)
        else:
             self.timeIntegration = TimeIntegrationClass(self)

        if options is not None:
            self.timeIntegration.setFromOptions(options)
        log(memory("TimeIntegration","OneLevelTransport"),level=4)
        log("Calculating numerical quadrature formulas",2)
        self.calculateQuadrature()

        comm = Comm.get()
        self.comm=comm
        if comm.size() > 1:
            assert numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions,"You must use a numerical flux to apply weak boundary conditions for parallel runs"

        self.setupFieldStrides()

        log(memory("stride+offset","OneLevelTransport"),level=4)
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
        #set penalty terms
        #cek todo move into numerical flux initialization
        if 'penalty'in self.ebq_global:
            for ebN in range(self.mesh.nElementBoundaries_global):
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['penalty'][ebN,k] = self.numericalFlux.penalty_constant/(self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power)
        #penalty term
        #cek move  to Numerical flux initialization
        if 'penalty' in self.ebqe:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebqe['penalty'][ebNE,k] = self.numericalFlux.penalty_constant/self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power
        log(memory("numericalFlux","OneLevelTransport"),level=4)
        self.elementEffectiveDiametersArray  = self.mesh.elementInnerDiametersArray
        #use post processing tools to get conservative fluxes, None by default
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)
        log(memory("velocity postprocessor","OneLevelTransport"),level=4)
        #helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        for ci,fbcObject  in self.fluxBoundaryConditionsObjectsDict.items():
            self.ebqe[('advectiveFlux_bc_flag',ci)] = np.zeros(self.ebqe[('advectiveFlux_bc',ci)].shape,'i')
            for t,g in fbcObject.advectiveFluxBoundaryConditionsDict.items():
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc',ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag',ci)][t[0],t[1]] = 1
            for ck,diffusiveFluxBoundaryConditionsDict in fbcObject.diffusiveFluxBoundaryConditionsDictDict.items():
                self.ebqe[('diffusiveFlux_bc_flag',ck,ci)] = np.zeros(self.ebqe[('diffusiveFlux_bc',ck,ci)].shape,'i')
                for t,g in diffusiveFluxBoundaryConditionsDict.items():
                    self.ebqe[('diffusiveFlux_bc',ck,ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                    self.ebqe[('diffusiveFlux_bc_flag',ck,ci)][t[0],t[1]] = 1
        self.numericalFlux.setDirichletValues(self.ebqe)
        if self.movingDomain:
            self.MOVING_DOMAIN=1.0
        else:
            self.MOVING_DOMAIN=0.0
        #cek hack
        self.movingDomain=False
        self.MOVING_DOMAIN=0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape,'d')
        #cek/ido todo replace python loops in modules with optimized code if possible/necessary
        self.forceStrongConditions=coefficients.forceStrongDirichlet
        self.dirichletConditionsForceDOF = DOFBoundaryConditions(self.u[0].femSpace,dofBoundaryConditionsSetterDict[0],weakDirichletConditions=False)
        # Triply-cut: pin masked global DOFs via strong-Dirichlet, not just the residual-side
        # ADR.h patch (which alone left the Jacobian inconsistent and broke Newton). Covers
        # every masked node (one or two per element), deduplicated by global DOF.
        if getattr(coefficients, 'triplyCut_globalDofOverrides', None) and self.forceStrongConditions:
            _nForced = 0
            for dofN, v in coefficients.triplyCut_globalDofOverrides.items():
                if dofN in self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict:
                    continue  # already forced by an exterior boundary condition (Section 7's coincidence)
                self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict[dofN] = (lambda x, t, _v=v: _v)
                self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN] = self.mesh.nodeArray[dofN]
                _nForced += 1
            if os.environ.get('ADR_DEBUG_TC'):
                print(f"[triplyCut] strongly pinned {_nForced} masked DOF(s) via forceStrongDirichlet "
                      f"(of {len(coefficients.triplyCut_globalDofOverrides)} identified)")
        compKernelFlag = 0
        self.adr = cADR_base(self.nSpace_global,
                               self.nQuadraturePoints_element,
                               self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                               self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                               self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                               self.nElementBoundaryQuadraturePoints_elementBoundary,
                               compKernelFlag)
    def calculateCoefficients(self):
        pass
    # 4-point Gauss-Legendre abscissae on [0,1], used to hand the kernel the exterior Dirichlet data of
    # each phase along an exterior edge (ADR_EXT_NITSCHE). Edge parameter t runs from the lower to the
    # higher global node number of the edge.
    _EXT_T = np.array([0.5 - 0.5*0.8611363115940526, 0.5 - 0.5*0.3399810435848563,
                       0.5 + 0.5*0.3399810435848563, 0.5 + 0.5*0.8611363115940526])

    def _exteriorEdgeBranchData(self):
        if getattr(self, '_extEdgeData', None) is not None:
            return self._extEdgeData
        nExt = self.mesh.nExteriorElementBoundaries_global
        ua = np.zeros((nExt, 4), 'd')
        ub = np.zeros((nExt, 4), 'd')
        ans = self.coefficients.analyticalSolution
        ans0 = ans.get(0) if ans is not None else None
        if ans0 is not None and os.environ.get('ADR_EXT_NITSCHE'):
            for ebNE in range(nExt):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                n0, n1 = sorted(self.mesh.elementBoundaryNodesArray[ebN][:2])
                P0, P1 = self.mesh.nodeArray[n0], self.mesh.nodeArray[n1]
                for k, t in enumerate(self._EXT_T):
                    x = P0 + t*(P1 - P0)
                    if hasattr(ans0, 'uOfX_inner') and hasattr(ans0, 'uOfX_outer'):
                        ua[ebNE, k] = ans0.uOfX_inner(x)
                        ub[ebNE, k] = ans0.uOfX_outer(x)
                    else:
                        ua[ebNE, k] = ub[ebNE, k] = ans0.uOfX(x)
        self._extEdgeData = (ua, ub)
        return self._extEdgeData

    def getResidual(self,u,r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        r.fill(0.0)
        try:
            self.isActiveDOF[:] = 0.0
        except AttributeError:
            self.isActiveDOF = np.zeros_like(r)
        #Load the unknowns into the finite element dof
        self.setUnknowns(u)


        #no flux boundary conditions
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        # ToDo: Have lagrange nodes for all DOFmaps. 
        argsDict["mesh_dof"] = self.mesh.nodeArray if (self.u[0].femSpace.max_nDOF_element==3) else self.u[0].femSpace.dofMap.lagrangeNodesArray
        # argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["x_ref"] = self.elementQuadraturePoints
        argsDict["xB_ref"] = self.elementBoundaryQuadraturePoints
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["elementBoundaryDiameter"] = self.mesh.elementBoundaryDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["cfl"] = self.q[('cfl',0)]
        argsDict["Ct_sge"] = self.shockCapturing.shockCapturingFactor
        argsDict["sc_uref"] = self.coefficients.sc_uref
        argsDict["sc_alpha"] = self.coefficients.sc_beta
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nElementBoundaries_owned"] = int(self.mesh.nElementBoundaries_owned)
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.u[0].dof
        argsDict["triplyCut_maskedLocalNode"] = self.coefficients.triplyCut_maskedLocalNode
        argsDict["triplyCut_overrideValue"] = self.coefficients.triplyCut_overrideValue
        argsDict["sd_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        argsDict["sd_colind"] = self.coefficients.sdInfo[(0,0)][1]
        argsDict["q_a"] = self.q[('a',0,0)]
        argsDict["q_v"] = self.q[('df',0,0)]
        argsDict["q_r"] = self.q[('r',0)]
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = self.shockCapturing.shockCapturingFactor
        argsDict["q_numDiff_u"] = self.shockCapturing.numDiff[0]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_a"] = self.ebqe[('a',0,0)]
        argsDict["ebqe_v"] = self.ebqe[('df',0,0)]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        ext_ua, ext_ub = self._exteriorEdgeBranchData()
        argsDict["ext_edge_u_inner"] = ext_ua
        argsDict["ext_edge_u_outer"] = ext_ub
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag',0,0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_u_ext"] = self.ebqe[('diffusiveFlux_bc',0,0)]
        argsDict["ebqe_bc_advectiveFlux_u_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["eb_adjoint_sigma"] = self.numericalFlux.boundaryAdjoint_sigma
        argsDict["embeddedBoundary"] = self.coefficients.embeddedBoundary
        argsDict["embeddedBoundary_penalty"] = self.coefficients.embeddedBoundary_penalty
        argsDict["embeddedBoundary_ghost_penalty"] = self.coefficients.embeddedBoundary_ghost_penalty
        argsDict["embedded_ghost_penalty"] = self.coefficients.embeddedBoundary_ghost_penalty
        argsDict["embeddedBoundary_sdf_nodes"] = self.coefficients.embeddedBoundary_sdf_nodes
        argsDict["embeddedBoundary_sdf_q"] = self.q['embeddedBoundary_sdf']
        argsDict["embeddedBoundary_normal_q"] = self.q['embeddedBoundary_normal']
        argsDict["embeddedBoundary_u_q"] = self.q['embeddedBoundary_u']
        argsDict["embeddedBoundary_u_inner_q"] = self.q['embeddedBoundary_u_inner']
        argsDict["embeddedBoundary_u_outer_q"] = self.q['embeddedBoundary_u_outer']
        argsDict["embeddedBoundary_robin"] = self.coefficients.embeddedBoundary_robin
        argsDict["embeddedBoundary_robin_h_inner"] = self.coefficients.embeddedBoundary_robin_h_inner
        argsDict["embeddedBoundary_robin_h_outer"] = self.coefficients.embeddedBoundary_robin_h_outer
        argsDict["embeddedBoundary_robin_uInfinity_inner_q"] = self.q['embeddedBoundary_robin_uInfinity_inner']
        argsDict["embeddedBoundary_robin_uInfinity_outer_q"] = self.q['embeddedBoundary_robin_uInfinity_outer']
        argsDict["immersedBoundary"] = self.coefficients.immersedBoundary
        argsDict["immersedBoundary_penalty"] = self.coefficients.immersedBoundary_penalty
        argsDict["immersedSCIFEM_switch"] = self.coefficients.immersedSCIFEM_switch
        argsDict["immersedSCIFEM_penalty"] = self.coefficients.immersedSCIFEM_penalty
        argsDict["PG"] = self.coefficients.PG
        argsDict["immersedBoundary_sdf_nodes"] = self.coefficients.immersedBoundary_sdf_nodes
        argsDict["immersedBoundary_sdf_q"] = self.q['immersedBoundary_sdf']
        argsDict["immersedBoundary_normal_q"] = self.q['immersedBoundary_normal']
        argsDict["immersedBoundary_u_q"] = self.q['immersedBoundary_u']
        argsDict["x_ref"] = self.elementQuadraturePoints
        argsDict["elementBoundaryDiameter"] = self.mesh.elementBoundaryDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["nElementBoundaries_owned"] = int(self.mesh.nElementBoundaries_owned)
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
        argsDict["immersedBoundary_fluxJump_q"] = self.q['immersedBoundary_fluxJump']
        argsDict["immersedBoundary_fluxJumpVector_q"] = self.q['immersedBoundary_fluxJumpVector']
        argsDict["immersedBoundary_fluxJump_element"] = self.coefficients.immersedBoundary_fluxJump_element
        argsDict["immersedBoundary_solutionJump_nodes"] = self.coefficients.immersedBoundary_solutionJump_nodes
        argsDict["isActiveDOF"] = self.isActiveDOF
        argsDict["mua"] = self.coefficients.mua
        argsDict["mub"] = self.coefficients.mub
        argsDict["jf"] = self.coefficients.jf
        argsDict["q_u_exact_inner"] = self.q[('u_exact_inner',0)]
        argsDict["q_u_exact_outer"] = self.q[('u_exact_outer',0)]
        argsDict["recomputeIFEMGeometry"] = int(self.recompute_ifem_geometry)
        self.recompute_ifem_geometry = False
        self.L2_error = np.array((0.0,),'d')
        argsDict["L2_error"] = self.L2_error
        self.Linfty_error = np.array((0.0,),'d')
        argsDict["Linfty_error"] = self.Linfty_error
        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                self.u[0].dof[dofN] = g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        self.adr.calculateResidual(argsDict)
        self.L2_error[0] = globalSum(self.L2_error[0])
        self.Linfty_error[0] = globalMax(self.Linfty_error[0])
        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                r[self.offset[0] + self.stride[0] * dofN] = self.u[0].dof[dofN] - g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        # Inactive DOFs (isActiveDOF==0) touch only elements where H_s=D_s=D_f=0
        # everywhere -- every bulk/Nitsche/immersed term in every such element is
        # multiplied by one of those, so an inactive DOF's own value is never read by
        # anything the solve actually uses. Leave it alone instead of zeroing it: a
        # DOF that also carries an exterior Dirichlet condition (or, once masked,
        # a forced continuation value) should keep that value, not get stomped back
        # to 0 after this call already used it correctly to build the residual above.
        r*=self.isActiveDOF
        log("2 error = {0}".format(self.L2_error**0.5),level=3)
        log("infty error = {0}".format(self.Linfty_error),level=3)
        if os.environ.get('ADR_DEBUG_RESID'):
            rows = []
            for i in range(len(self.u[0].dof)):
                ri = r[self.offset[0] + self.stride[0]*i]
                x = self.mesh.nodeArray[i]
                phi_s = self.coefficients.embeddedBoundary_sdf_nodes[i] if self.coefficients.embeddedBoundary else None
                phi_f = self.coefficients.immersedBoundary_sdf_nodes[i] if self.coefficients.immersedBoundary else None
                rows.append((abs(ri), i, tuple(x), ri, phi_s, phi_f, self.isActiveDOF[i]))
            rows.sort(reverse=True)
            print(f"[RESID] top 15 |r[dof]| (nonlinear_function_evaluations="
                  f"{self.nonlinear_function_evaluations}):")
            for a, i, x, ri, phi_s, phi_f, act in rows[:15]:
                print(f"  node={i} x={x} r={ri:+.6e} phi_s={phi_s} phi_f={phi_f} active={act}")
        if os.environ.get('ADR_DEBUG_DOFERR') and self.coefficients.analyticalSolution is not None:
            ans0 = self.coefficients.analyticalSolution.get(0)
            if ans0 is not None:
                errs = []
                for i in range(len(self.u[0].dof)):
                    x = self.mesh.nodeArray[i]
                    exact = ans0.uOfX(x)
                    err = self.u[0].dof[i] - exact
                    phi_s = self.coefficients.embeddedBoundary_sdf_nodes[i] if self.coefficients.embeddedBoundary else None
                    phi_f = self.coefficients.immersedBoundary_sdf_nodes[i] if self.coefficients.immersedBoundary else None
                    errs.append((abs(err), i, tuple(x), err, phi_s, phi_f, self.isActiveDOF[i]))
                errs_active = [e for e in errs if e[6] == 1.0]
                errs_active.sort(reverse=True)
                print(f"[DOFERR] top 12 |u_dof - u_exact| among ACTIVE dofs only "
                      f"(nonlinear_function_evaluations={self.nonlinear_function_evaluations}):")
                for a, i, x, err, phi_s, phi_f, act in errs_active[:12]:
                    print(f"  node={i} x={x} err={err:+.6e} phi_s={phi_s} phi_f={phi_f} active={act}")
                _watchset = set(int(v) for v in os.environ.get('ADR_DEBUG_NODESET','').split(',') if v)
                if _watchset:
                    print("[DOFERR] watched nodes:")
                    for a, i, x, err, phi_s, phi_f, act in errs:
                        if i in _watchset:
                            print(f"  node={i} x={x} err={err:+.6e} phi_s={phi_s} phi_f={phi_f} active={act}")
        log("Global residual",level=9,data=r)
        #self.coefficients.massConservationError = fabs(globalSum(sum(r.flat[:self.mesh.nElements_owned])))
        #log("   Mass Conservation Error",level=3,data=self.coefficients.massConservationError)
        self.nonlinear_function_evaluations += 1
    def getJacobian(self,jacobian):
        #import superluWrappers
        #import numpy
        import pdb
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,jacobian)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray if (self.u[0].femSpace.max_nDOF_element==3) else self.u[0].femSpace.dofMap.lagrangeNodesArray
        # argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["x_ref"] = self.elementQuadraturePoints
        argsDict["xB_ref"] = self.elementBoundaryQuadraturePoints
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["elementBoundaryDiameter"] = self.mesh.elementBoundaryDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["cfl"] = self.q[('cfl',0)]
        argsDict["Ct_sge"] = self.shockCapturing.shockCapturingFactor
        argsDict["sc_uref"] = self.coefficients.sc_uref
        argsDict["sc_alpha"] = self.coefficients.sc_beta
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nElementBoundaries_owned"] = int(self.mesh.nElementBoundaries_owned)
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.u[0].dof
        argsDict["sd_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        argsDict["sd_colind"] = self.coefficients.sdInfo[(0,0)][1]
        argsDict["q_a"] = self.q[('a',0,0)]
        argsDict["q_v"] = self.q[('df',0,0)]
        argsDict["q_r"] = self.q[('r',0)]
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = self.shockCapturing.shockCapturingFactor
        argsDict["q_numDiff_u"] = self.shockCapturing.numDiff[0]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0,0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_a"] = self.ebqe[('a',0,0)]
        argsDict["ebqe_v"] = self.ebqe[('df',0,0)]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        ext_ua, ext_ub = self._exteriorEdgeBranchData()
        argsDict["ext_edge_u_inner"] = ext_ua
        argsDict["ext_edge_u_outer"] = ext_ub
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag',0,0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_u_ext"] = self.ebqe[('diffusiveFlux_bc',0,0)]
        argsDict["ebqe_bc_advectiveFlux_u_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0,0)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["eb_adjoint_sigma"] = self.numericalFlux.boundaryAdjoint_sigma
        argsDict["embeddedBoundary"] = self.coefficients.embeddedBoundary
        argsDict["embeddedBoundary_penalty"] = self.coefficients.embeddedBoundary_penalty
        argsDict["embeddedBoundary_ghost_penalty"] = self.coefficients.embeddedBoundary_ghost_penalty
        argsDict["embedded_ghost_penalty"] = self.coefficients.embeddedBoundary_ghost_penalty
        argsDict["embeddedBoundary_sdf_nodes"] = self.coefficients.embeddedBoundary_sdf_nodes
        argsDict["embeddedBoundary_sdf_q"] = self.q['embeddedBoundary_sdf']
        argsDict["embeddedBoundary_normal_q"] = self.q['embeddedBoundary_normal']
        argsDict["embeddedBoundary_u_q"] = self.q['embeddedBoundary_u']
        argsDict["embeddedBoundary_u_inner_q"] = self.q['embeddedBoundary_u_inner']
        argsDict["embeddedBoundary_u_outer_q"] = self.q['embeddedBoundary_u_outer']
        argsDict["embeddedBoundary_robin"] = self.coefficients.embeddedBoundary_robin
        argsDict["embeddedBoundary_robin_h_inner"] = self.coefficients.embeddedBoundary_robin_h_inner
        argsDict["embeddedBoundary_robin_h_outer"] = self.coefficients.embeddedBoundary_robin_h_outer
        argsDict["embeddedBoundary_robin_uInfinity_inner_q"] = self.q['embeddedBoundary_robin_uInfinity_inner']
        argsDict["embeddedBoundary_robin_uInfinity_outer_q"] = self.q['embeddedBoundary_robin_uInfinity_outer']
        argsDict["immersedBoundary"] = self.coefficients.immersedBoundary
        argsDict["immersedBoundary_penalty"] = self.coefficients.immersedBoundary_penalty
        argsDict["immersedSCIFEM_switch"] = self.coefficients.immersedSCIFEM_switch
        argsDict["immersedSCIFEM_penalty"] = self.coefficients.immersedSCIFEM_penalty
        argsDict["PG"] = self.coefficients.PG
        argsDict["immersedBoundary_sdf_nodes"] = self.coefficients.immersedBoundary_sdf_nodes
        argsDict["immersedBoundary_sdf_q"] = self.q['immersedBoundary_sdf']
        argsDict["immersedBoundary_normal_q"] = self.q['immersedBoundary_normal']
        argsDict["immersedBoundary_u_q"] = self.q['immersedBoundary_u']
        argsDict["x_ref"] = self.elementQuadraturePoints
        argsDict["elementBoundaryDiameter"] = self.mesh.elementBoundaryDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["nElementBoundaries_owned"] = int(self.mesh.nElementBoundaries_owned)
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
        argsDict["immersedBoundary_fluxJump_q"] = self.q['immersedBoundary_fluxJump']
        argsDict["immersedBoundary_fluxJumpVector_q"] = self.q['immersedBoundary_fluxJumpVector']
        argsDict["immersedBoundary_fluxJump_element"] = self.coefficients.immersedBoundary_fluxJump_element
        argsDict["immersedBoundary_solutionJump_nodes"] = self.coefficients.immersedBoundary_solutionJump_nodes
        argsDict["isActiveDOF"] = self.isActiveDOF
        argsDict["mua"] = self.coefficients.mua
        argsDict["mub"] = self.coefficients.mub
        argsDict["jf"] = self.coefficients.jf
        argsDict["recomputeIFEMGeometry"] = int(self.recompute_ifem_geometry)
        self.recompute_ifem_geometry = False
        self.adr.calculateJacobian(argsDict)
        if self.forceStrongConditions:
            for dofN in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.keys()):
                global_dofN = self.offset[0] + self.stride[0] * dofN
                for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                        self.nzval[i] = 1.0
                    else:
                        self.nzval[i] = 0.0
        log("Jacobian ",level=10,data=jacobian)
        self.nonlinear_function_jacobian_evaluations += 1
        for global_dofN_a in np.argwhere(self.isActiveDOF==0.0):
            global_dofN = global_dofN_a[0]
            for i in range(
                    self.rowptr[global_dofN],
                    self.rowptr[global_dofN + 1]):
                if (self.colind[i] == global_dofN):
                    self.nzval[i] = 1.0
                else:
                    self.nzval[i] = 0.0

        # An element is marked active if H_s, D_s or D_f is nonzero anywhere in it. That is an
        # element-level test, so an element straddling the solid boundary claims *all* of its
        # DOFs -- including ones lying strictly inside the solid, where H_s == 0 (no bulk
        # contribution) and D_s == 0 (the Nitsche term lives on the solid boundary, not its
        # interior). Such a DOF is marked active but receives no matrix entry at all, leaving a
        # structurally zero row and a singular Jacobian. It only shows up once the mesh is fine
        # enough to put a node strictly inside the solid on an element the fluid interface also
        # cuts -- i.e. a triply-cut configuration.
        #
        # Catch them the same way genuinely inactive DOFs are handled: unit diagonal, and mark
        # them inactive so the residual is masked out there too. Inert wherever no such row
        # exists, which is every single-level-set case.
        # Tolerance, not exact equality: same rule as the buried-interface area check in
        # ADR.h (equivalent_polynomials -- "never compare a geometric fit to exact zero").
        # A DOF this row-degenerate is fed only by a moment-fit/Dirac weight evaluated at (or
        # extremely close to) a genuine geometric degeneracy -- e.g. a triple point landing
        # almost exactly on a mesh node once the mesh is refined enough -- so its diagonal
        # lands a few ULPs off zero, not bit-exact, the same way a buried sub-region's area
        # does. An exact-equality check misses that case and leaves a near-singular (not
        # exactly singular) row that a direct solve can still return a garbage correction
        # for -- observed as a stalled Newton iteration whose residual never moves, not a
        # clean failure. _diagTol is scaled to this matrix's own entries so it adapts to
        # whatever mua/mub/mesh-size combination is in use, the same way areaTol is scaled to
        # the reference triangle rather than being one fixed absolute number.
        _diagTol = 1.0e-10 * max(1.0, np.max(np.abs(self.nzval))) if len(self.nzval) else 1.0e-10
        _unsupported = []
        for _row in range(len(self.rowptr) - 1):
            if self.isActiveDOF[_row] == 0.0:
                continue
            _diag = 0.0
            for _i in range(self.rowptr[_row], self.rowptr[_row + 1]):
                if self.colind[_i] == _row:
                    _diag = self.nzval[_i]
            if abs(_diag) < _diagTol:
                _unsupported.append(_row)
        for _row in _unsupported:
            self.isActiveDOF[_row] = 0.0
            for _i in range(self.rowptr[_row], self.rowptr[_row + 1]):
                self.nzval[_i] = 1.0 if self.colind[_i] == _row else 0.0
        if _unsupported:
            log("ADR: pinned %d unsupported DOF(s) with no Jacobian contribution "
                "(active element, but inside the embedded solid)" % len(_unsupported), level=4)
        import os
        # ADR_DUMP_JACOBIAN=<prefix>: save the final Jacobian (after strong Dirichlet rows and pinned
        # rows) once, with the strongly imposed Dirichlet DOFs and the active flags, for an offline
        # study of conditioning and linear solvers on the free (active, non-Dirichlet) unknowns.
        if os.environ.get("ADR_DUMP_JACOBIAN") and not getattr(self, '_adr_jac_dumped', False):
            _dir = [self.offset[0] + self.stride[0]*d for d in self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.keys()] \
                if self.forceStrongConditions else []
            np.savez(os.environ["ADR_DUMP_JACOBIAN"] + ".npz", rowptr=np.array(self.rowptr), colind=np.array(self.colind),
                     nzval=np.array(self.nzval), dirichlet=np.array(_dir, dtype='i'), active=np.array(self.isActiveDOF))
            self._adr_jac_dumped = True
        if os.environ.get("ADR_DEBUG_COND"):
            import scipy.sparse as _sp
            import scipy.sparse.linalg as _spla
            _n = len(self.rowptr) - 1
            _A = _sp.csr_matrix((self.nzval, self.colind, self.rowptr), shape=(_n, _n))
            try:
                if _n <= 2000:
                    _dense = _A.toarray()
                    _sv = np.linalg.svd(_dense, compute_uv=False)
                    print("ADR_DEBUG_COND: n=%d smallest_sv=%.6e largest_sv=%.6e cond=%.6e rank=%d"
                          % (_n, _sv[-1], _sv[0], _sv[0] / max(_sv[-1], 1e-300), np.sum(_sv > 1e-10 * _sv[0])))
                else:
                    _sv_min = _spla.svds(_A, k=1, which='SM', return_singular_vectors=False)
                    _sv_max = _spla.svds(_A, k=1, which='LM', return_singular_vectors=False)
                    print("ADR_DEBUG_COND: n=%d smallest_sv~=%.6e largest_sv~=%.6e"
                          % (_n, _sv_min[0], _sv_max[0]))
            except Exception as _e:
                print("ADR_DEBUG_COND: failed:", _e)
            # One-shot finite-difference Jacobian consistency check: does the assembled
            # Jacobian actually predict how getResidual responds to a perturbation, at the
            # SAME u this Jacobian was just built at? A mismatch here -- not a conditioning
            # problem -- would explain a well-conditioned linear solve whose correction has
            # zero effect on the true (residual-defined) objective.
            if not getattr(self, '_adr_fd_done', False):
                self._adr_fd_done = True
                try:
                    _u = self.u[0].dof.copy()
                    _r0 = np.zeros(_n, 'd')
                    self.getResidual(_u, _r0)
                    _eps = 1.0e-6
                    _forced = set(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.keys())
                    _active = np.argwhere(self.isActiveDOF != 0.0).flatten()
                    _sample = [i for i in _active if i not in _forced]
                    print("ADR_DEBUG_FD: n=%d forced=%d testing %d free active dofs"
                          % (_n, len(_forced), len(_sample)))
                    _worst_overall = 0.0
                    _worst_dof = -1
                    _nbad = 0
                    for _i in _sample:
                        _up = _u.copy()
                        _scale = max(1.0, abs(_u[_i]))
                        _h = _eps * _scale
                        _up[_i] += _h
                        _r1 = np.zeros(_n, 'd')
                        self.getResidual(_up, _r1)
                        _fd_col = (_r1 - _r0) / _h
                        _an_col = _A[:, _i].toarray().flatten()
                        _tol = 1.0e-3 * max(1.0, np.max(np.abs(_an_col)))
                        _diff = np.abs(_fd_col - _an_col)
                        _worst = np.argmax(_diff)
                        if _diff[_worst] > _tol:
                            _nbad += 1
                            print("ADR_DEBUG_FD: MISMATCH dof=%d h=%.3e max|fd-an|=%.6e at row=%d "
                                  "fd=%.6e an=%.6e (tol=%.3e)"
                                  % (_i, _h, _diff[_worst], _worst, _fd_col[_worst], _an_col[_worst], _tol))
                        if _diff[_worst] > _worst_overall:
                            _worst_overall = _diff[_worst]
                            _worst_dof = _i
                    print("ADR_DEBUG_FD: SUMMARY %d/%d free dofs mismatched (tol 1e-3 rel), "
                          "worst=%.6e at dof=%d" % (_nbad, len(_sample), _worst_overall, _worst_dof))
                except Exception as _e:
                    print("ADR_DEBUG_FD: failed:", _e)
        return jacobian
    def invalidateIFEMGeometry(self):
        """
        Call whenever the embedded/immersed interface geometry (sdf) changes -
        e.g. mesh adaptation, or a new time step for a moving interface - so
        the C++ layer recomputes and refreshes its per-element equivalent-
        polynomial/IFEM cache on the next residual/Jacobian call instead of
        reusing stale cut-cell data.
        """
        self.recompute_ifem_geometry = True
    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
                                                 self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t,self.q)
        if self.stabilization is not None:
            self.stabilization.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing is not None:
            self.shockCapturing.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q)
        self.invalidateIFEMGeometry()
    def archiveFiniteElementSolutions(self,archive,t,tCount,initialPhase=False,writeVectors=True,
                                      meshChanged=False,femSpaceWritten={},writeVelocityPostProcessor=True):
        OneLevelTransport.archiveFiniteElementSolutions(self,archive,t,tCount,
                                                        initialPhase=initialPhase,
                                                        writeVectors=writeVectors,
                                                        meshChanged=meshChanged,
                                                        femSpaceWritten=femSpaceWritten,
                                                        writeVelocityPostProcessor=writeVelocityPostProcessor)
        # Append the level sets so ParaView sees them as node-centred scalars on the
        # same grid; the zero contour of each is the corresponding interface.
        for lsFunction in getattr(self,'levelSetFunctions',[]):
            self.u[0].femSpace.writeFunctionXdmf(archive,lsFunction,tCount)

    def calculateElementBoundaryQuadrature(self):
        pass
    def calculateExteriorElementBoundaryQuadrature(self):
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        self.fluxBoundaryConditionsObjectsDict = dict([(cj,FluxBoundaryConditions(self.mesh,
                                                                                  self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                  self.ebqe[('x')],
                                                                                  self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                                                                  self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                       for cj in self.advectiveFluxBoundaryConditionsSetterDict.keys()])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t,self.ebqe)
    def estimate_mt(self):
        pass
    def calculateAuxiliaryQuantitiesAfterStep(self):
        log("L2 error = [{0}]".format(" ".join("%.16e" % e for e in np.atleast_1d(self.L2_error**0.5))),level=3)
        log("Linfty error = [{0}]".format(" ".join("%.16e" % e for e in np.atleast_1d(self.Linfty_error))),level=3)
        pass
    def calculateSolutionAtQuadrature(self):
        pass
