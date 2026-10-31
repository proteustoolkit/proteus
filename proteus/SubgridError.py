"""
A class hierarchy for subgrid error estimation methods (multiscale methods)

.. inheritance-diagram:: proteus.SubgridError
   :parts: 1
"""
import numpy
from . import csubgridError
from . import FemTools
from .Profiling import logEvent
class SGE_base(object):
    def __init__(self,coefficients,nd,lag=False,trackSubScales=False):
        self.nc = coefficients.nc
        self.nd = nd
        self.components=list(range(self.nc))
        self.lag=lag
        self.lagRequested=lag
        self.coefficients=coefficients
        self.trackSubScales = trackSubScales
        self.usesGradientStabilization = False
    def state(self):
        """Nothing to carry beyond the solution (the state contract, see TimeIntegration.TI_base.state)."""
        return None

    def set_state(self, state):
        """Restore what ``state()`` returned; see TimeIntegration.TI_base.state."""
        pass

    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        for ci in range(self.nc):
            if self.lag:
                self.tau_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            else:
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            for cj in range(self.nc):
                if ('df',ci,cj) in cq:
                    cq[('df_sge',ci,cj)]=cq[('df',ci,cj)]
                if ('dH',ci,cj) in cq:
                    cq[('dH_sge',ci,cj)]=cq[('dH',ci,cj)]
                if ('dm',ci,cj) in cq:
                    cq[('dm_sge',ci,cj)]=cq[('dm',ci,cj)]
                if ('dmt',ci,cj) in cq:
                    cq[('dmt_sge',ci,cj)]=cq[('dmt',ci,cj)]
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                cq[('grad(phi)_sge',ck)]=cq[('grad(phi)',ck)]
                for cj in list(cjDict.keys()):
                    cq[('dphi_sge',ck,cj)]=cq[('dphi',ck,cj)]
                    cq[('da_sge',ci,ck,cj)]=cq[('da',ci,ck,cj)]
    def initializeTimeIntegration(self,timeIntegration):
        """
        allow for connection with time integration method if tracking subscales
        """
        pass
    def calculateSubgridError(self,q):
        pass
    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in range(self.nc):
                self.tau_last[ci][:] = self.tau[ci]
    def accumulateSubgridMassHistory(self,q):
        """
        incorporate subgrid scale mass accumulation
        \\delta m^{n}/\\delta t^{n+1}
        """
        pass
class Advection_ASGS(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.df_last={}
        self.cq=cq
        for ci in range(self.nc):
            if self.lag:
                self.tau_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                if ('df',ci,ci) in cq:
                    self.df_last = copy.deepcopy(cq[('df',ci,ci)])
                    cq[('df_sge',ci,ci)] = self.df_last
            else:
                if ('df',ci,ci) in cq:
                    cq[('df_sge',ci,ci)] = cq[('df',ci,ci)]
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in range(self.nc):
                self.tau_last[ci][:] = self.tau[ci]
                self.df_last[:] = self.cq[('df',ci,ci)]
    def calculateSubgridError(self,q):
        for ci in range(self.nc):
            csubgridError.calculateSubgridError_A_tau(self.stabilizationFlag,
                                                      self.mesh.elementDiametersArray,
                                                      q[('dmt',ci,ci)],
                                                      q[('df',ci,ci)],
                                                      q[('cfl',ci)],
                                                      self.tau[ci])
            if self.lag:
                tau=self.tau_last[ci]
            else:
                tau=self.tau[ci]
            for cj in range(self.nc):
                if ('dpdeResidual',ci,cj) in q:
                    csubgridError.calculateSubgridError_tauRes(tau,
                                                               q[('pdeResidual',ci)],
                                                               q[('dpdeResidual',ci,cj)],
                                                               q[('subgridError',ci)],
                                                               q[('dsubgridError',ci,cj)])

class AdvectionLag_ASGS(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag

    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.df_last={}
        self.cq=cq
        for ci in range(self.nc):
            if self.lag:
                self.tau_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                if ('df',ci,ci) in cq:
                    self.df_last = copy.deepcopy(cq[('df',ci,ci)])
                    cq[('df_sge',ci,ci)] = self.df_last
            else:
                if ('df',ci,ci) in cq:
                    cq[('df_sge',ci,ci)] = cq[('df',ci,ci)]
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in range(self.nc):
                self.tau_last[ci][:] = self.tau[ci]
                self.df_last[:] = self.cq[('df',ci,ci)]
    def calculateSubgridError(self,q):
        for ci in range(self.nc):
            csubgridError.calculateSubgridError_A_tau(self.stabilizationFlag,
                                                      self.mesh.elementDiametersArray,
                                                      q[('dmt',ci,ci)],
                                                      q[('df_sge',ci,ci)],
                                                      q[('cfl',ci)],
                                                      self.tau[ci])
            tau=self.tau[ci]
            for cj in range(self.nc):
                if ('dpdeResidual',ci,cj) in q:
                    csubgridError.calculateSubgridError_tauRes(tau,
                                                               q[('pdeResidual',ci)],
                                                               q[('dpdeResidual',ci,cj)],
                                                               q[('subgridError',ci)],
                                                               q[('dsubgridError',ci,cj)])

class AdvectionDiffusionReaction_ASGS(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.cq=cq
        for ci in range(self.nc):
            if self.lag:
                self.tau_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                if ('df',ci,ci) in cq:
                    cq[('df_sge',ci,ci)] = copy.deepcopy(cq[('df',ci,ci)])
                if ('dm',ci,ci) in cq:
                    cq[('dm_sge',ci,ci)] = copy.deepcopy(cq[('dm',ci,ci)])
                if ('dmt',ci,ci) in cq:
                    cq[('dmt_sge',ci,ci)] = copy.deepcopy(cq[('dmt',ci,ci)])
            else:
                if ('df',ci,ci) in cq:
                    cq[('df_sge',ci,ci)] = cq[('df',ci,ci)]
                if ('dm',ci,ci) in cq:
                    cq[('dm_sge',ci,ci)] = cq[('dm',ci,ci)]
                if ('dmt',ci,ci) in cq:
                    cq[('dmt_sge',ci,ci)] = cq[('dmt',ci,ci)]
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))

        for ci,ckDict in self.coefficients.diffusion.items():
            if self.lag:#mwf looks like this was missing if lag May 7 09
                for ck,cjDict in ckDict.items():
                    cq[('grad(phi)_sge',ck)]=copy.deepcopy(cq[('grad(phi)',ck)])
                    for cj in list(cjDict.keys()):
                        cq[('dphi_sge',ck,cj)]=copy.deepcopy(cq[('dphi',ck,cj)])
                        cq[('da_sge',ci,ck,cj)]=copy.deepcopy(cq[('da',ci,ck,cj)])
            else:
                for ck,cjDict in ckDict.items():
                    cq[('grad(phi)_sge',ck)]=cq[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        cq[('dphi_sge',ck,cj)]=cq[('dphi',ck,cj)]
                        cq[('da_sge',ci,ck,cj)]=cq[('da',ci,ck,cj)]

    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in range(self.nc):
                self.tau_last[ci][:] = self.tau[ci]
                #mwf should these be deep copies?
                self.cq[('df_sge',ci,ci)][:] = self.cq[('df',ci,ci)]
                self.cq[('dm_sge',ci,ci)][:] = self.cq[('dm',ci,ci)]
            for ci,ckDict in self.coefficients.diffusion.items():
                for ck,cjDict in ckDict.items():
                    self.cq[('grad(phi)_sge',ck)][:]=self.cq[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        self.cq[('dphi_sge',ck,cj)][:]=0.0 #grad(phi) will be a constant when lagged so dphi=0 not 1
                        self.cq[('da_sge',ci,ck,cj)][:]=self.cq[('da',ci,ck,cj)]
    def calculateSubgridError(self,q):
        oldTau=False#True #mwf oldTau not working with sd!
        for ci in range(self.nc):
            if oldTau:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridError_ADR_tau_sd(self.stabilizationFlag,
                                                                   self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                   self.mesh.elementDiametersArray,
                                                                   q[('dmt',ci,ci)],
                                                                   q[('df',ci,ci)],
                                                                   q[('a',ci,ci)],
                                                                   q[('da',ci,ci,ci)],
                                                                   q[('grad(phi)',ci)],
                                                                   q[('dphi',ci,ci)],
                                                                   q[('dr',ci,ci)],
                                                                   q[('pe',ci)],
                                                                   q[('cfl',ci)],
                                                                   self.tau[ci])
                else:
                    csubgridError.calculateSubgridError_ADR_tau(self.stabilizationFlag,
                                                                self.mesh.elementDiametersArray,
                                                                q[('dmt',ci,ci)],
                                                                q[('df',ci,ci)],
                                                                q[('a',ci,ci)],
                                                                q[('da',ci,ci,ci)],
                                                                q[('grad(phi)',ci)],
                                                                q[('dphi',ci,ci)],
                                                                q[('dr',ci,ci)],
                                                                q[('pe',ci)],
                                                                q[('cfl',ci)],
                                                                self.tau[ci])
            else:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridError_ADR_generic_tau_sd(self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                           q['inverse(J)'],
                                                                           q[('dmt',ci,ci)],
                                                                           q[('df',ci,ci)],
                                                                           q[('a',ci,ci)],
                                                                           q[('da',ci,ci,ci)],
                                                                           q[('grad(phi)',ci)],
                                                                           q[('dphi',ci,ci)],
                                                                           q[('dr',ci,ci)],
                                                                           q[('pe',ci)],
                                                                           q[('cfl',ci)],
                                                                           self.tau[ci])
                else:
                    csubgridError.calculateSubgridError_ADR_generic_tau(q['inverse(J)'],
                                                                        q[('dmt',ci,ci)],
                                                                        q[('df',ci,ci)],
                                                                        q[('a',ci,ci)],
                                                                        q[('da',ci,ci,ci)],
                                                                        q[('grad(phi)',ci)],
                                                                        q[('dphi',ci,ci)],
                                                                        q[('dr',ci,ci)],
                                                                        q[('pe',ci)],
                                                                        q[('cfl',ci)],
                                                                        self.tau[ci])
            if self.lag:
                tau=self.tau_last[ci]
            else:
                tau=self.tau[ci]
            for cj in range(self.nc):
                if ('dpdeResidual',ci,cj) in q:
                    csubgridError.calculateSubgridError_tauRes(tau,
                                                               q[('pdeResidual',ci)],
                                                               q[('dpdeResidual',ci,cj)],
                                                               q[('subgridError',ci)],
                                                               q[('dsubgridError',ci,cj)])
            #mwf debug
            #import pdb
            #pdb.set_trace()
#             print "tau",tau
#             print "pdeResidual",q[('pdeResidual',ci)]
#             print "dpdeResidual",q[('dpdeResidual',ci,ci)]
#             print "subgrid error",q[('subgridError',ci)]
#             print "dsubgrid error",q[('dsubgridError',ci,ci)]
class FFDarcyFC_ASGS(SGE_base):
    """
    basic stablization for TwophaseDarcy_fc_ff, only 'mixture' equation has advection term
    'w' phase equation has nonlinear diffusion wrt mixture potential,
    'mixture' equation has two nonlinear diffusion terms
    """
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
        self.dftemp = None
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.df_last={}
        self.cq=cq
        for ci in [0]:
            if self.lag:
                self.tau_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                if ('df',ci,ci) in cq:
                    self.df_last = copy.deepcopy(cq[('df',ci,ci)])
                    cq[('df_sge',ci,ci)] = self.df_last
            else:
                if ('df',ci,ci) in cq:
                    cq[('df_sge',ci,ci)] = cq[('df',ci,ci)]
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
        self.cq=cq
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                cq[('grad(phi)_sge',ck)]=copy.deepcopy(cq[('grad(phi)',ck)])
                for cj in list(cjDict.keys()):
                    cq[('dphi_sge',ck,cj)]=copy.deepcopy(cq[('dphi',ck,cj)])
                    cq[('da_sge',ci,ck,cj)]=copy.deepcopy(cq[('da',ci,ck,cj)])

    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in [0]:
                self.tau_last[ci][:] = self.tau[ci]
                #self.df_last[:] = self.cq[('df',ci,ci)]
            for ci,ckDict in self.coefficients.diffusion.items():
                for ck,cjDict in ckDict.items():
                    self.cq[('grad(phi)_sge',ck)][:]=self.cq[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        self.cq[('dphi_sge',ck,cj)][:]=0.0 #grad(phi) will be a constant when lagged so dphi=0 not 1
                        self.cq[('da_sge',ci,ck,cj)][:]=self.cq[('da',ci,ck,cj)]

    def calculateSubgridError(self,q):
        oldTau = False
        if self.dftemp is None or self.dftemp.shape != q[('grad(phi)',1)].shape:
            self.dftemp = numpy.zeros(q[('grad(phi)',1)].shape,'d')
        ci = 0; cj = 0; ck = 1;
        if oldTau:
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_tau_sd(self.stabilizationFlag,
                                                               self.coefficients.sdInfo[(0,1)][0],self.coefficients.sdInfo[(0,1)][1],
                                                               self.mesh.elementDiametersArray,
                                                               q[('dmt',0,0)],
                                                               self.dftemp,
                                                               q[('a',0,1)],
                                                               q[('da',0,1,0)],
                                                               q[('grad(phi)',1)],
                                                               q[('dphi',1,0)],
                                                               q[('dr',0,0)],
                                                               q[('pe',0)],
                                                               q[('cfl',0)],
                                                               self.tau[0])
            else:
                csubgridError.calculateSubgridError_ADR_tau(self.stabilizationFlag,
                                                            self.mesh.elementDiametersArray,
                                                            q[('dmt',0,0)],
                                                            self.dftemp,
                                                            q[('a',0,1)],
                                                            q[('da',0,1,0)],
                                                            q[('grad(phi)',1)],
                                                            q[('dphi',1,0)],
                                                            q[('dr',0,0)],
                                                            q[('pe',0)],
                                                            q[('cfl',0)],
                                                            self.tau[0])
        else:
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_generic_tau_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                       q['inverse(J)'],
                                                                       q[('dmt',ci,ci)],
                                                                       self.dftemp,
                                                                       q[('a',ci,ck)],
                                                                       q[('da',ci,ck,cj)],
                                                                       q[('grad(phi)',ck)],
                                                                       q[('dphi',ck,cj)],
                                                                       q[('dr',ci,cj)],
                                                                       q[('pe',ci)],
                                                                       q[('cfl',ci)],
                                                                       self.tau[ci])
            else:
                csubgridError.calculateSubgridError_ADR_generic_tau(q['inverse(J)'],
                                                                    q[('dmt',ci,ci)],
                                                                    self.dftemp,
                                                                    q[('a',ci,ck)],
                                                                    q[('da',ci,ck,cj)],
                                                                    q[('grad(phi)',ck)],
                                                                    q[('dphi',ck,cj)],
                                                                    q[('dr',ci,cj)],
                                                                    q[('pe',ci)],
                                                                    q[('cfl',ci)],
                                                                    self.tau[ci])
        if self.lag:
            tau=self.tau_last[0]
        else:
            tau=self.tau[0]
        csubgridError.calculateSubgridError_tauRes(tau,
                                                   q[('pdeResidual',0)],
                                                   q[('dpdeResidual',0,0)],
                                                   q[('subgridError',0)],
                                                   q[('dsubgridError',0,0)])
#             print "tau",tau
#             print "pdeResidual",q[('pdeResidual',ci)]
#             print "dpdeResidual",q[('dpdeResidual',ci,ci)]
#             print "subgrid error",q[('subgridError',ci)]
#             print "dsubgrid error",q[('dsubgridError',ci,ci)]
class DarcyFC_ASGS(SGE_base):
    """
    basic stablization for TwophaseDarcy_fc, no advection term
    'w' phase and 'n' phase have nonlinear diffusion wrt to their own potential
    phi_w = psi_w, phi_n = psi_w + psi_c
    """
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
        self.dftemp = None; self.drtmp = {(0,0):None,(1,0):None}
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.df_last={}
        self.cq=cq
        for ci in [0,1]:
            if self.lag:
                self.tau_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            else:
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
        self.cq=cq
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                cq[('grad(phi)_sge',ck)]=copy.deepcopy(cq[('grad(phi)',ck)])
                for cj in list(cjDict.keys()):
                    cq[('dphi_sge',ck,cj)]=copy.deepcopy(cq[('dphi',ck,cj)])
                    cq[('da_sge',ci,ck,cj)]=copy.deepcopy(cq[('da',ci,ck,cj)])

    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in [0,1]:
                self.tau_last[ci][:] = self.tau[ci]
                #self.df_last[:] = self.cq[('df',ci,ci)]
            for ci,ckDict in self.coefficients.diffusion.items():
                for ck,cjDict in ckDict.items():
                    self.cq[('grad(phi)_sge',ck)][:]=self.cq[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        self.cq[('dphi_sge',ck,cj)][:]=0.0 #grad(phi) will be a constant when lagged so dphi=0 not 1
                        self.cq[('da_sge',ci,ck,cj)][:]=self.cq[('da',ci,ck,cj)]

    def calculateSubgridError(self,q):
        oldTau=False
        if self.dftemp is None or self.dftemp.shape != q[('grad(phi)',1)].shape:
            self.dftemp = numpy.zeros(q[('grad(phi)',1)].shape,'d')

        #'w' phase equation
        ci = 0; cj = 0; ck = 0;
        if ('dr',ci,cj) in q:
            self.drtmp[(ci,cj)] = q[('dr',ci,cj)]
        elif self.drtmp[(ci,cj)] is None:
            self.drtmp[(ci,cj)] = numpy.zeros(q[('r',ci)].shape,'d')
        if self.drtmp[(ci,cj)] is None or self.drtmp[(ci,cj)].shape != q[('r',ci)].shape:
            self.drtmp[(ci,cj)] = numpy.zeros(q[('r',ci)].shape,'d')
        if oldTau:
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_tau_sd(self.stabilizationFlag,
                                                               self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                               self.mesh.elementDiametersArray,
                                                               q[('dmt',ci,cj)],
                                                               self.dftemp,
                                                               q[('a',ci,ck)],
                                                               q[('da',ci,ck,cj)],
                                                               q[('grad(phi)',ck)],
                                                               self.drtmp[(ci,cj)],
                                                               self.drtmp[(ci,cj)],
                                                               q[('pe',ci)],
                                                               q[('cfl',ci)],
                                                               self.tau[ci])
            else:
                csubgridError.calculateSubgridError_ADR_tau(self.stabilizationFlag,
                                                         self.mesh.elementDiametersArray,
                                                         q[('dmt',ci,cj)],
                                                         self.dftemp,
                                                         q[('a',ci,ck)],
                                                         q[('da',ci,ck,cj)],
                                                         q[('grad(phi)',ck)],
                                                         self.drtmp[(ci,cj)],
                                                         self.drtmp[(ci,cj)],
                                                         q[('pe',ci)],
                                                         q[('cfl',ci)],
                                                         self.tau[ci])
        else:
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_generic_tau_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                       q['inverse(J)'],
                                                                       q[('dmt',ci,cj)],
                                                                       self.dftemp,
                                                                       q[('a',ci,ck)],
                                                                       q[('da',ci,ck,cj)],
                                                                       q[('grad(phi)',ck)],
                                                                       self.drtmp[(ci,cj)],
                                                                       self.drtmp[(ci,cj)],
                                                                       q[('pe',ci)],
                                                                       q[('cfl',ci)],
                                                                       self.tau[ci])
            else:
                csubgridError.calculateSubgridError_ADR_generic_tau(q['inverse(J)'],
                                                                    q[('dmt',ci,cj)],
                                                                    self.dftemp,
                                                                    q[('a',ci,ck)],
                                                                    q[('da',ci,ck,cj)],
                                                                    q[('grad(phi)',ck)],
                                                                    self.drtmp[(ci,cj)],
                                                                    self.drtmp[(ci,cj)],
                                                                    q[('pe',ci)],
                                                                    q[('cfl',ci)],
                                                                    self.tau[ci])
        #'n' phase equation
        ci = 1; cj = 0; ck = 1;
        if ('dr',ci,cj) in q:
            self.drtmp[(ci,cj)] = q[('dr',ci,cj)]
        elif self.drtmp[(ci,cj)] is None:
            self.drtmp[(ci,cj)] = numpy.zeros(q[('r',ci)].shape,'d')
        if oldTau:
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_tau_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                               self.stabilizationFlag,
                                                               self.mesh.elementDiametersArray,
                                                               q[('dmt',ci,cj)],
                                                               self.dftemp,
                                                               q[('a',ci,ck)],
                                                               q[('da',ci,ck,cj)],
                                                               q[('grad(phi)',ck)],
                                                               q[('dphi',ck,cj)],
                                                               self.drtmp[(ci,cj)],
                                                               q[('pe',ci)],
                                                               q[('cfl',ci)],
                                                               self.tau[ci])
            else:
                csubgridError.calculateSubgridError_ADR_tau(self.stabilizationFlag,
                                                        self.mesh.elementDiametersArray,
                                                        q[('dmt',ci,cj)],
                                                        self.dftemp,
                                                        q[('a',ci,ck)],
                                                        q[('da',ci,ck,cj)],
                                                        q[('grad(phi)',ck)],
                                                        q[('dphi',ck,cj)],
                                                        self.drtmp[(ci,cj)],
                                                        q[('pe',ci)],
                                                        q[('cfl',ci)],
                                                        self.tau[ci])
        else:
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_generic_tau_sd(self.coefficients.sdInfo[(ci,ck)][0],self.coefficients.sdInfo[(ci,ck)][1],
                                                                       q['inverse(J)'],
                                                                       q[('dmt',ci,cj)],
                                                                       self.dftemp,
                                                                       q[('a',ci,ck)],
                                                                       q[('da',ci,ck,cj)],
                                                                       q[('grad(phi)',ck)],
                                                                       q[('dphi',ck,cj)],
                                                                       self.drtmp[(ci,cj)],
                                                                       q[('pe',ci)],
                                                                       q[('cfl',ci)],
                                                                       self.tau[ci])
            else:
                csubgridError.calculateSubgridError_ADR_generic_tau(q['inverse(J)'],
                                                                    q[('dmt',ci,cj)],
                                                                    self.dftemp,
                                                                    q[('a',ci,ck)],
                                                                    q[('da',ci,ck,cj)],
                                                                    q[('grad(phi)',ck)],
                                                                    q[('dphi',ck,cj)],
                                                                    self.drtmp[(ci,cj)],
                                                                    q[('pe',ci)],
                                                                    q[('cfl',ci)],
                                                                    self.tau[ci])
        for ci in [0,1]:
            if self.lag:
                tau=self.tau_last[ci]
            else:
                tau=self.tau[ci]
            #for now just compute wrt to cj?
#             cj = 0
#             csubgridError.calculateSubgridError_tauRes(tau,
#                                                        q[('pdeResidual',ci)],
#                                                        q[('dpdeResidual',ci,cj)],
#                                                        q[('subgridError',ci)],
#                                                        q[('dsubgridError',ci,cj)])
            csubgridError.calculateSubgridError_tauRes(tau,
                                                       q[('pdeResidual',0)],
                                                       q[('dpdeResidual',0,0)],
                                                       q[('subgridError',ci)],
                                                       q[('dsubgridError',ci,0)])


#             print "tau",tau
#             print "pdeResidual",q[('pdeResidual',ci)]
#             print "dpdeResidual",q[('dpdeResidual',ci,ci)]
#             print "subgrid error",q[('subgridError',ci)]
#             print "dsubgrid error",q[('dsubgridError',ci,ci)]
class HamiltonJacobi_ASGS(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.cq=cq
        self.mesh=mesh
        self.tau={}
        for ci in range(self.nc):
            self.tau[ci]=numpy.zeros(cq[('u',ci)].shape,'d')
            if self.lag:
                cq[('dH_sge',ci,ci)]=copy.deepcopy(cq[('dH',ci,ci)])
            else:
                cq[('dH_sge',ci,ci)]=cq[('dH',ci,ci)]
    def calculateSubgridError(self,q):
        for ci in range(self.nc):
            csubgridError.calculateSubgridError_HJ_tau(self.stabilizationFlag,
                                                       self.mesh.elementDiametersArray,
                                                       q[('dmt',ci,ci)],
                                                       q[('dH_sge',ci,ci)],
                                                       q[('cfl',ci)],
                                                       self.tau[ci])
            csubgridError.calculateSubgridError_tauRes(self.tau[ci],
                                                       q[('pdeResidual',ci)],
                                                       q[('dpdeResidual',ci,ci)],
                                                       q[('subgridError',ci)],
                                                       q[('dsubgridError',ci,ci)])
    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in range(self.nc):
                self.cq[('dH_sge',ci,ci)][:]= self.cq[('dH',ci,ci)]

class HamiltonJacobiDiffusionReaction_ASGS(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.cq=cq
        for ci in range(self.nc):
            if self.lag:
                self.tau_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                if ('dH',ci,ci) in cq:
                    cq[('dH_sge',ci,ci)] = copy.deepcopy(cq[('dH',ci,ci)])
                if ('dm',ci,ci) in cq:
                    cq[('dm_sge',ci,ci)] = copy.deepcopy(cq[('dm',ci,ci)])
                if ('dmt',ci,ci) in cq:
                    cq[('dmt_sge',ci,ci)] = copy.deepcopy(cq[('dmt',ci,ci)])
            else:
                if ('dH',ci,ci) in cq:
                    cq[('dH_sge',ci,ci)] = cq[('dH',ci,ci)]
                if ('dm',ci,ci) in cq:
                    cq[('dm_sge',ci,ci)] = cq[('dm',ci,ci)]
                if ('dmt',ci,ci) in cq:
                    cq[('dmt_sge',ci,ci)] = cq[('dmt',ci,ci)]
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))

        for ci,ckDict in self.coefficients.diffusion.items():
            if self.lag:#mwf looks like this was missing if lag May 7 09
                for ck,cjDict in ckDict.items():
                    cq[('grad(phi)_sge',ck)]=copy.deepcopy(cq[('grad(phi)',ck)])
                    for cj in list(cjDict.keys()):
                        cq[('dphi_sge',ck,cj)]=copy.deepcopy(cq[('dphi',ck,cj)])
                        cq[('da_sge',ci,ck,cj)]=copy.deepcopy(cq[('da',ci,ck,cj)])
            else:
                for ck,cjDict in ckDict.items():
                    cq[('grad(phi)_sge',ck)]=cq[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        cq[('dphi_sge',ck,cj)]=cq[('dphi',ck,cj)]
                        cq[('da_sge',ci,ck,cj)]=cq[('da',ci,ck,cj)]

    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in range(self.nc):
                self.tau_last[ci][:] = self.tau[ci]
                #mwf should these be deep copies?
                self.cq[('dH_sge',ci,ci)][:] = self.cq[('dH',ci,ci)]
                self.cq[('dm_sge',ci,ci)][:] = self.cq[('dm',ci,ci)]
            for ci,ckDict in self.coefficients.diffusion.items():
                for ck,cjDict in ckDict.items():
                    self.cq[('grad(phi)_sge',ck)][:]=self.cq[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        self.cq[('dphi_sge',ck,cj)][:]=0.0 #grad(phi) will be a constant when lagged so dphi=0 not 1
                        self.cq[('da_sge',ci,ck,cj)][:]=self.cq[('da',ci,ck,cj)]

    def calculateSubgridError(self,q):
        oldTau=False#True #mwf oldTau not working with sd!
        for ci in range(self.nc):
            if oldTau:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridError_ADR_tau_sd(self.stabilizationFlag,
                                                                   self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                   self.mesh.elementDiametersArray,
                                                                   q[('dmt',ci,ci)],
                                                                   q[('dH',ci,ci)],
                                                                   q[('a',ci,ci)],
                                                                   q[('da',ci,ci,ci)],
                                                                   q[('grad(phi)',ci)],
                                                                   q[('dphi',ci,ci)],
                                                                   q[('dr',ci,ci)],
                                                                   q[('pe',ci)],
                                                                   q[('cfl',ci)],
                                                                   self.tau[ci])
                else:
                    csubgridError.calculateSubgridError_ADR_tau(self.stabilizationFlag,
                                                                self.mesh.elementDiametersArray,
                                                                q[('dmt',ci,ci)],
                                                                q[('dH',ci,ci)],
                                                                q[('a',ci,ci)],
                                                                q[('da',ci,ci,ci)],
                                                                q[('grad(phi)',ci)],
                                                                q[('dphi',ci,ci)],
                                                                q[('dr',ci,ci)],
                                                                q[('pe',ci)],
                                                                q[('cfl',ci)],
                                                                self.tau[ci])
            else:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridError_ADR_generic_tau_sd(self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                           q['inverse(J)'],
                                                                           q[('dmt',ci,ci)],
                                                                           q[('dH',ci,ci)],
                                                                           q[('a',ci,ci)],
                                                                           q[('da',ci,ci,ci)],
                                                                           q[('grad(phi)',ci)],
                                                                           q[('dphi',ci,ci)],
                                                                           q[('dr',ci,ci)],
                                                                           q[('pe',ci)],
                                                                           q[('cfl',ci)],
                                                                           self.tau[ci])
                else:
                    csubgridError.calculateSubgridError_ADR_generic_tau(q['inverse(J)'],
                                                                        q[('dmt',ci,ci)],
                                                                        q[('dH',ci,ci)],
                                                                        q[('a',ci,ci)],
                                                                        q[('da',ci,ci,ci)],
                                                                        q[('grad(phi)',ci)],
                                                                        q[('dphi',ci,ci)],
                                                                        q[('dr',ci,ci)],
                                                                        q[('pe',ci)],
                                                                        q[('cfl',ci)],
                                                                        self.tau[ci])
            if self.lag:
                tau=self.tau_last[ci]
            else:
                tau=self.tau[ci]
            for cj in range(self.nc):
                if ('dpdeResidual',ci,cj) in q:
                    csubgridError.calculateSubgridError_tauRes(tau,
                                                               q[('pdeResidual',ci)],
                                                               q[('dpdeResidual',ci,cj)],
                                                               q[('subgridError',ci)],
                                                               q[('dsubgridError',ci,cj)])

class HamiltonJacobi_ASGS_opt(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.cq=cq
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        for ci in range(self.nc):
            if self.lag:
                cq[('dH_sge',ci,ci)]=copy.deepcopy(cq[('dH',ci,ci)])
            else:
                cq[('dH_sge',ci,ci)]=cq[('dH',ci,ci)]
    def calculateSubgridError(self,q):
        pass
    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in range(self.nc):
                self.cq[('dH_sge',ci,ci)][:]= self.cq[('dH',ci,ci)]
    def state(self):
        """The lagged dH_sge, the only history this class keeps; None when
        not lagging (dH_sge is then dH itself)."""
        if not self.lag:
            return None
        return {'dH_sge_%d' % ci: numpy.array(self.cq[('dH_sge',ci,ci)], copy=True)
                for ci in range(self.nc)}
    def set_state(self, state):
        for ci in range(self.nc):
            self.cq[('dH_sge',ci,ci)][:] = state['dH_sge_%d' % ci]

class StokesStabilization_1(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        SGE_base.__init__(self,coefficients,nd,lag)
    def calculateSubgridError(self,q):
        if self.coefficients.sd:
            csubgridError.calculateSubgridErrorStokes2D_1_sd(self.mesh.elementDiametersArray,
                                                             q[('u',1)],
                                                             q[('u',2)],
                                                             q[('a',1,1)],
                                                             q[('pdeResidual',0)],
                                                             q[('dpdeResidual',0,1)],
                                                             q[('dpdeResidual',0,2)],
                                                             q[('pdeResidual',1)],
                                                             q[('dpdeResidual',1,0)],
                                                             q[('dpdeResidual',1,1)],
                                                             q[('pdeResidual',2)],
                                                             q[('dpdeResidual',2,0)],
                                                             q[('dpdeResidual',2,2)],
                                                             q[('subgridError',0)],
                                                             q[('dsubgridError',0,0)],
                                                             q[('dsubgridError',0,1)],
                                                             q[('dsubgridError',0,2)],
                                                             q[('subgridError',1)],
                                                             q[('dsubgridError',1,0)],
                                                             q[('dsubgridError',1,1)],
                                                             q[('dsubgridError',1,2)],
                                                             q[('subgridError',2)],
                                                             q[('dsubgridError',2,0)],
                                                             q[('dsubgridError',2,1)],
                                                             q[('dsubgridError',2,2)])
        else:
            csubgridError.calculateSubgridErrorStokes2D_1(self.mesh.elementDiametersArray,
                                                          q[('u',1)],
                                                          q[('u',2)],
                                                          q[('a',1,1)],
                                                          q[('pdeResidual',0)],
                                                          q[('dpdeResidual',0,1)],
                                                          q[('dpdeResidual',0,2)],
                                                          q[('pdeResidual',1)],
                                                          q[('dpdeResidual',1,0)],
                                                          q[('dpdeResidual',1,1)],
                                                          q[('pdeResidual',2)],
                                                          q[('dpdeResidual',2,0)],
                                                          q[('dpdeResidual',2,2)],
                                                          q[('subgridError',0)],
                                                          q[('dsubgridError',0,0)],
                                                          q[('dsubgridError',0,1)],
                                                          q[('dsubgridError',0,2)],
                                                          q[('subgridError',1)],
                                                          q[('dsubgridError',1,0)],
                                                          q[('dsubgridError',1,1)],
                                                          q[('dsubgridError',1,2)],
                                                          q[('subgridError',2)],
                                                          q[('dsubgridError',2,0)],
                                                          q[('dsubgridError',2,1)],
                                                          q[('dsubgridError',2,2)])
    def updateSubgridErrorHistory(self,initializationPhase=False):
        pass

class StokesASGS_velocity(SGE_base):
    def __init__(self,coefficients,nd):
        SGE_base.__init__(self,coefficients,nd,lag=False)
        self.stabilizationFlag = '1'
        coefficients.stencil[0].add(0)
        if nd == 2:
            coefficients.stencil[1].add(2)
            coefficients.stencil[2].add(1)
        elif nd == 3:
            coefficients.stencil[1].add(2)
            coefficients.stencil[1].add(3)
            coefficients.stencil[2].add(1)
            coefficients.stencil[2].add(3)
            coefficients.stencil[3].add(1)
            coefficients.stencil[3].add(2)
    def calculateSubgridError(self,q):
        if self.nd == 2:
            if self.coefficients.sd:
                csubgridError.calculateSubgridErrorStokes2D_GLS_velocity_sd(self.mesh.elementDiametersArray,
                                                                            q[('a',1,1)],
                                                                            q[('pdeResidual',1)],
                                                                            q[('dpdeResidual',1,0)],
                                                                            q[('dpdeResidual',1,1)],
                                                                            q[('pdeResidual',2)],
                                                                            q[('dpdeResidual',2,0)],
                                                                            q[('dpdeResidual',2,2)],
                                                                            q[('subgridError',1)],
                                                                            q[('dsubgridError',1,0)],
                                                                            q[('dsubgridError',1,1)],
                                                                            q[('subgridError',2)],
                                                                            q[('dsubgridError',2,0)],
                                                                            q[('dsubgridError',2,2)])
            else:
                csubgridError.calculateSubgridErrorStokes2D_GLS_velocity(self.mesh.elementDiametersArray,
                                                                         q[('a',1,1)],
                                                                         q[('pdeResidual',1)],
                                                                         q[('dpdeResidual',1,0)],
                                                                         q[('dpdeResidual',1,1)],
                                                                         q[('pdeResidual',2)],
                                                                         q[('dpdeResidual',2,0)],
                                                                         q[('dpdeResidual',2,2)],
                                                                         q[('subgridError',1)],
                                                                         q[('dsubgridError',1,0)],
                                                                         q[('dsubgridError',1,1)],
                                                                         q[('subgridError',2)],
                                                                         q[('dsubgridError',2,0)],
                                                                         q[('dsubgridError',2,2)])
        elif self.nd == 3:
            if self.coefficients.sd:
                csubgridError.calculateSubgridErrorStokes3D_GLS_velocity_sd(self.mesh.elementDiametersArray,
                                                                            q[('a',1,1)],
                                                                            q[('pdeResidual',1)],
                                                                            q[('dpdeResidual',1,0)],
                                                                            q[('dpdeResidual',1,1)],
                                                                            q[('pdeResidual',2)],
                                                                            q[('dpdeResidual',2,0)],
                                                                            q[('dpdeResidual',2,2)],
                                                                            q[('pdeResidual',3)],
                                                                            q[('dpdeResidual',3,0)],
                                                                            q[('dpdeResidual',3,3)],
                                                                            q[('subgridError',1)],
                                                                            q[('dsubgridError',1,0)],
                                                                            q[('dsubgridError',1,1)],
                                                                            q[('subgridError',2)],
                                                                            q[('dsubgridError',2,0)],
                                                                            q[('dsubgridError',2,2)],
                                                                            q[('subgridError',3)],
                                                                            q[('dsubgridError',3,0)],
                                                                            q[('dsubgridError',3,3)])
            else:
                csubgridError.calculateSubgridErrorStokes3D_GLS_velocity(self.mesh.elementDiametersArray,
                                                                         q[('a',1,1)],
                                                                         q[('pdeResidual',1)],
                                                                         q[('dpdeResidual',1,0)],
                                                                         q[('dpdeResidual',1,1)],
                                                                         q[('pdeResidual',2)],
                                                                         q[('dpdeResidual',2,0)],
                                                                         q[('dpdeResidual',2,2)],
                                                                         q[('pdeResidual',3)],
                                                                         q[('dpdeResidual',3,0)],
                                                                         q[('dpdeResidual',3,3)],
                                                                         q[('subgridError',1)],
                                                                         q[('dsubgridError',1,0)],
                                                                         q[('dsubgridError',1,1)],
                                                                         q[('subgridError',2)],
                                                                         q[('dsubgridError',2,0)],
                                                                         q[('dsubgridError',2,2)],
                                                                         q[('subgridError',3)],
                                                                         q[('dsubgridError',3,0)],
                                                                         q[('dsubgridError',3,3)])
    def updateSubgridErrorHistory(self,initializationPhase=False):
        pass

class NavierStokesASGS_velocity_pressure(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False,delayLagSteps=5,hFactor=1.0,noPressureStabilization=False):
        self.noPressureStabilization=noPressureStabilization
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
        coefficients.stencil[0].add(0)
        self.nSteps=0
        self.delayLagSteps=delayLagSteps
        self.hFactor=hFactor
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.df_last={}
        self.cq=cq
        self.v_last = copy.deepcopy(cq[('f',0)])
        for ci in range(self.nc):
            if self.lag:
                self.tau_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                for cj in range(self.nc):
                    if ('df',ci,cj) in cq:
                        if ci ==0:
                            cq[('df_sge',ci,cj)]=cq[('df',ci,cj)]
                        else:
                            #cek for incompressible form weshould just be able to use v_last
                            #cq[('df_sge',ci,cj)] = numpy.zeros(cq[('df',ci,cj)].shape,'d')
                            if ci == cj:
                                cq[('df_sge',ci,cj)] = self.v_last
                            else:
                                cq[('df_sge',ci,cj)] = numpy.zeros(cq[('df',ci,cj)].shape,'d')
            else:
                for cj in range(self.nc):
                    if ('df',ci,cj) in cq:
                        cq[('df_sge',ci,cj)]=cq[('df',ci,cj)]
                self.tau.append(numpy.zeros(cq[('u',ci)].shape,'d'))
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                cq[('grad(phi)_sge',ck)]=cq[('grad(phi)',ck)]
                for cj in list(cjDict.keys()):
                    cq[('dphi_sge',ck,cj)]=cq[('dphi',ck,cj)]
                    cq[('da_sge',ci,ck,cj)]=cq[('da',ci,ck,cj)]
        for ci,cjDict in self.coefficients.hamiltonian.items():
            for cj in cjDict:
                cq[('dH_sge',ci,cj)]=cq[('dH',ci,cj)]
        if self.lag:
            if self.coefficients.sd:
                csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau_sd(self.hFactor,
                                                                             self.mesh.elementDiametersArray,
                                                                             cq[('dmt',1,1)],
                                                                             cq[('dm',1,1)],
                                                                             cq[('f',0)],
                                                                             cq[('a',1,1)],
                                                                             self.tau[0],
                                                                             self.tau[1],
                                                                             cq[('cfl',0)])
            else:
                csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau(self.hFactor,
                                                                          self.mesh.elementDiametersArray,
                                                                          cq[('dmt',1,1)],
                                                                          cq[('dm',1,1)],
                                                                          cq[('f',0)],
                                                                          cq[('a',1,1)],
                                                                          self.tau[0],
                                                                          self.tau[1],
                                                                          cq[('cfl',0)])
            self.v_last[:]=self.cq[('f',0)]
    def updateSubgridErrorHistory(self,initializationPhase=False):
        self.nSteps+=1
        if self.lag:
            for ci in range(self.nc):
                self.tau_last[ci][:] = self.tau[ci]
                self.v_last[:]=self.cq[('f',0)]
#cek for incompressible form we can just use v_last
#                 for cj in range(self.nc):
#                     if self.cq.has_key(('df',ci,cj)):
#                         if ci != 0:
#                             self.cq[('df_sge',ci,cj)][:] = self.cq[('df',ci,cj)]
    def calculateSubgridError(self,q):
        from . import LinearAlgebraTools
        oldTau=True
        if self.nd == 2:
            if self.lag and self.nSteps < self.delayLagSteps:
                v = q[('f',0)]
            elif self.lag:
                v = self.v_last
            else:
                v = q[('f',0)]
            if oldTau:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau_sd(self.hFactor,
                                                                                 self.mesh.elementDiametersArray,
                                                                                 q[('dmt',1,1)],
                                                                                 q[('dm',1,1)],
                                                                                 v,
                                                                                 q[('a',1,1)],
                                                                                 self.tau[0],
                                                                                 self.tau[1],
                                                                                 q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau(self.hFactor,
                                                                              self.mesh.elementDiametersArray,
                                                                              q[('dmt',1,1)],
                                                                              q[('dm',1,1)],
                                                                              v,
                                                                              q[('a',1,1)],
                                                                              self.tau[0],
                                                                              self.tau[1],
                                                                              q[('cfl',0)])
            else:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau_sd(q['inverse(J)'],
                                                                                     q[('dmt',1,1)],
                                                                                     q[('dm',1,1)],
                                                                                     v,
                                                                                     q[('a',1,1)],
                                                                                     self.tau[0],
                                                                                     self.tau[1],
                                                                                     q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau(q['inverse(J)'],
                                                                                  q[('dmt',1,1)],
                                                                                  q[('dm',1,1)],
                                                                                  v,
                                                                                  q[('a',1,1)],
                                                                                  self.tau[0],
                                                                                  self.tau[1],
                                                                                  q[('cfl',0)])
            tau0=self.tau[0]
            tau1=self.tau[1]
            csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tauRes(tau0,
                                                                         tau1,
                                                                         q[('pdeResidual',0)],
                                                                         q[('dpdeResidual',0,1)],
                                                                         q[('dpdeResidual',0,2)],
                                                                         q[('pdeResidual',1)],
                                                                         q[('dpdeResidual',1,0)],
                                                                         q[('dpdeResidual',1,1)],
                                                                         q[('dpdeResidual',1,2)],
                                                                         q[('pdeResidual',2)],
                                                                         q[('dpdeResidual',2,0)],
                                                                         q[('dpdeResidual',2,1)],
                                                                         q[('dpdeResidual',2,2)],
                                                                         q[('subgridError',0)],
                                                                         q[('dsubgridError',0,1)],
                                                                         q[('dsubgridError',0,2)],
                                                                         q[('subgridError',1)],
                                                                         q[('dsubgridError',1,0)],
                                                                         q[('dsubgridError',1,1)],
                                                                         q[('dsubgridError',1,2)],
                                                                         q[('subgridError',2)],
                                                                         q[('dsubgridError',2,0)],
                                                                         q[('dsubgridError',2,1)],
                                                                         q[('dsubgridError',2,2)])
            if self.noPressureStabilization:
                q[('subgridError',0)][:]=0.0
                q[('dsubgridError',0,1)][:]=0.0
                q[('dsubgridError',0,2)][:]=0.0
        elif self.nd == 3:
            if self.lag and self.nSteps < self.delayLagSteps:
                v = q[('f',0)]
            elif self.lag:
                v = self.v_last
            else:
                v = q[('f',0)]
            if oldTau:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau_sd(self.hFactor,
                                                                                 self.mesh.elementDiametersArray,
                                                                                 q[('dmt',1,1)],
                                                                                 q[('dm',1,1)],
                                                                                 v,
                                                                                 q[('a',1,1)],
                                                                                 self.tau[0],
                                                                                 self.tau[1],
                                                                                 q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau(self.hFactor,
                                                                              self.mesh.elementDiametersArray,
                                                                              q[('dmt',1,1)],
                                                                              q[('dm',1,1)],
                                                                              v,
                                                                              q[('a',1,1)],
                                                                              self.tau[0],
                                                                              self.tau[1],
                                                                              q[('cfl',0)])
            else:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau_sd(q['inverse(J)'],
                                                                                     q[('dmt',1,1)],
                                                                                     q[('dm',1,1)],
                                                                                     v,
                                                                                     q[('a',1,1)],
                                                                                     self.tau[0],
                                                                                     self.tau[1],
                                                                                     q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau(q['inverse(J)'],
                                                                                  q[('dmt',1,1)],
                                                                                  q[('dm',1,1)],
                                                                                  v,
                                                                                  q[('a',1,1)],
                                                                                  self.tau[0],
                                                                                  self.tau[1],
                                                                                  q[('cfl',0)])
            tau0=self.tau[0]
            tau1=self.tau[1]
            csubgridError.calculateSubgridErrorNavierStokes3D_GLS_tauRes(tau0,
                                                                         tau1,
                                                                         q[('pdeResidual',0)],
                                                                         q[('dpdeResidual',0,1)],
                                                                         q[('dpdeResidual',0,2)],
                                                                         q[('dpdeResidual',0,3)],
                                                                         q[('pdeResidual',1)],
                                                                         q[('dpdeResidual',1,0)],
                                                                         q[('dpdeResidual',1,1)],
                                                                         q[('dpdeResidual',1,2)],
                                                                         q[('dpdeResidual',1,3)],
                                                                         q[('pdeResidual',2)],
                                                                         q[('dpdeResidual',2,0)],
                                                                         q[('dpdeResidual',2,1)],
                                                                         q[('dpdeResidual',2,2)],
                                                                         q[('dpdeResidual',2,3)],
                                                                         q[('pdeResidual',3)],
                                                                         q[('dpdeResidual',3,0)],
                                                                         q[('dpdeResidual',3,1)],
                                                                         q[('dpdeResidual',3,2)],
                                                                         q[('dpdeResidual',3,3)],
                                                                         q[('subgridError',0)],
                                                                         q[('dsubgridError',0,1)],
                                                                         q[('dsubgridError',0,2)],
                                                                         q[('dsubgridError',0,3)],
                                                                         q[('subgridError',1)],
                                                                         q[('dsubgridError',1,0)],
                                                                         q[('dsubgridError',1,1)],
                                                                         q[('dsubgridError',1,2)],
                                                                         q[('dsubgridError',1,3)],
                                                                         q[('subgridError',2)],
                                                                         q[('dsubgridError',2,0)],
                                                                         q[('dsubgridError',2,1)],
                                                                         q[('dsubgridError',2,2)],
                                                                         q[('dsubgridError',2,3)],
                                                                         q[('subgridError',3)],
                                                                         q[('dsubgridError',3,0)],
                                                                         q[('dsubgridError',3,1)],
                                                                         q[('dsubgridError',3,2)],
                                                                         q[('dsubgridError',3,3)])
            if self.noPressureStabilization:
                q[('subgridError',0)][:]=0.0
                q[('dsubgridError',0,1)][:]=0.0
                q[('dsubgridError',0,2)][:]=0.0
                q[('dsubgridError',0,3)][:]=0.0
        for ci in range(self.nd):
            q[('cfl',ci+1)][:] = q[('cfl',0)]

class NavierStokesASGS_velocity_pressure_opt(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False,delayLagSteps=5,hFactor=1.0,noPressureStabilization=False):
        self.noPressureStabilization=noPressureStabilization
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
        coefficients.stencil[0].add(0)
        self.nSteps=0
        self.delayLagSteps=delayLagSteps
        self.hFactor=hFactor
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.df_last={}
        self.cq=cq
        if self.lag:
            self.v_last = self.cq[('velocity',0)]
        else:
            self.v_last = cq[('f',0)]
        cq[('df_sge',1,1)]=self.v_last
        cq[('df_sge',2,2)]=self.v_last
        cq[('df_sge',3,3)]=self.v_last
    def updateSubgridErrorHistory(self,initializationPhase=False):
        self.nSteps+=1
    def calculateSubgridError(self,q):
        if self.nSteps < self.delayLagSteps:
            self.v_last = q[('f',0)]
            cq[('df_sge',1,1)]=q[('f',0)]
            cq[('df_sge',2,2)]=q[('f',0)]
            cq[('df_sge',3,3)]=q[('f',0)]
        else:
            self.v_last = q[('velocity',0)]
            cq[('df_sge',1,1)]=q[('velocity',0)]
            cq[('df_sge',2,2)]=q[('velocity',0)]
            cq[('df_sge',3,3)]=q[('velocity',0)]

class NavierStokesASGS_velocity_pressure_optV2(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False,delayLagSteps=0,hFactor=1.0,noPressureStabilization=False):
        self.noPressureStabilization=noPressureStabilization
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
        coefficients.stencil[0].add(0)
        self.nSteps=0
        self.delayLagSteps=delayLagSteps
        self.hFactor=hFactor
    def initializeElementQuadrature(self,mesh,t,cq):
        import copy
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        self.df_last={}
        self.cq=cq
        if self.lag:
            self.v_last = copy.deepcopy(self.cq[('velocity',0)])
        else:
            self.v_last = self.cq[('velocity',0)]
    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            self.v_last[:] = self.cq[('velocity',0)]
    def calculateSubgridError(self,q):
        pass

class NavierStokesWithBodyForceASGS_velocity_pressure(NavierStokesASGS_velocity_pressure):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False,delayLagSteps=5,hFactor=1.0,noPressureStabilization=False):
        NavierStokesASGS_velocity_pressure.__init__(self,coefficients,nd,stabFlag=stabFlag,lag=lag,
                                                    delayLagSteps=delayLagSteps,hFactor=hFactor,noPressureStabilization=noPressureStabilization)
    def initializeElementQuadrature(self,mesh,t,cq):
        NavierStokesASGS_velocity_pressure.initializeElementQuadrature(self,mesh,t,cq)
        self.q_dmt_r = numpy.zeros(cq[('dmt',1,1)].shape,'d')

    def calculateSubgridError(self,q):
        from . import LinearAlgebraTools
        oldTau=True
        self.q_dmt_r.flat[:] = q[('dmt',1,1)].flat
        self.q_dmt_r += q[('dr',1,1)]
        if self.nd == 2:
            if self.lag and self.nSteps < self.delayLagSteps:
                v = q[('f',0)]
            elif self.lag:
                v = self.v_last
            else:
                v = q[('f',0)]
            if oldTau:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau_sd(self.hFactor,
                                                                                 self.mesh.elementDiametersArray,
                                                                                 self.q_dmt_r,
                                                                                 q[('dm',1,1)],
                                                                                 v,
                                                                                 q[('a',1,1)],
                                                                                 self.tau[0],
                                                                                 self.tau[1],
                                                                                 q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau(self.hFactor,
                                                                              self.mesh.elementDiametersArray,
                                                                              self.q_dmt_r,
                                                                              q[('dm',1,1)],
                                                                              v,
                                                                              q[('a',1,1)],
                                                                              self.tau[0],
                                                                              self.tau[1],
                                                                              q[('cfl',0)])
            else:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau_sd(q['inverse(J)'],
                                                                                     self.q_dmt_r,
                                                                                     q[('dm',1,1)],
                                                                                     v,
                                                                                     q[('a',1,1)],
                                                                                     self.tau[0],
                                                                                     self.tau[1],
                                                                                     q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau(q['inverse(J)'],
                                                                                  self.q_dmt_r,
                                                                                  q[('dm',1,1)],
                                                                                  v,
                                                                                  q[('a',1,1)],
                                                                                  self.tau[0],
                                                                                  self.tau[1],
                                                                                  q[('cfl',0)])
            tau0=self.tau[0]
            tau1=self.tau[1]
            csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tauRes(tau0,
                                                                         tau1,
                                                                         q[('pdeResidual',0)],
                                                                         q[('dpdeResidual',0,1)],
                                                                         q[('dpdeResidual',0,2)],
                                                                         q[('pdeResidual',1)],
                                                                         q[('dpdeResidual',1,0)],
                                                                         q[('dpdeResidual',1,1)],
                                                                         q[('dpdeResidual',1,2)],
                                                                         q[('pdeResidual',2)],
                                                                         q[('dpdeResidual',2,0)],
                                                                         q[('dpdeResidual',2,1)],
                                                                         q[('dpdeResidual',2,2)],
                                                                         q[('subgridError',0)],
                                                                         q[('dsubgridError',0,1)],
                                                                         q[('dsubgridError',0,2)],
                                                                         q[('subgridError',1)],
                                                                         q[('dsubgridError',1,0)],
                                                                         q[('dsubgridError',1,1)],
                                                                         q[('dsubgridError',1,2)],
                                                                         q[('subgridError',2)],
                                                                         q[('dsubgridError',2,0)],
                                                                         q[('dsubgridError',2,1)],
                                                                         q[('dsubgridError',2,2)])
            if self.noPressureStabilization:
                q[('subgridError',0)][:]=0.0
                q[('dsubgridError',0,1)][:]=0.0
                q[('dsubgridError',0,2)][:]=0.0
        elif self.nd == 3:
            if self.lag and self.nSteps < self.delayLagSteps:
                v = q[('f',0)]
            elif self.lag:
                v = self.v_last
            else:
                v = q[('f',0)]
            if oldTau:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau_sd(self.hFactor,
                                                                                 self.mesh.elementDiametersArray,
                                                                                 self.q_dmt_r,
                                                                                 q[('dm',1,1)],
                                                                                 v,
                                                                                 q[('a',1,1)],
                                                                                 self.tau[0],
                                                                                 self.tau[1],
                                                                                 q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau(self.hFactor,
                                                                              self.mesh.elementDiametersArray,
                                                                              self.q_dmt_r,
                                                                              q[('dm',1,1)],
                                                                              v,
                                                                              q[('a',1,1)],
                                                                              self.tau[0],
                                                                              self.tau[1],
                                                                              q[('cfl',0)])
            else:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau_sd(q['inverse(J)'],
                                                                                     self.q_dmt_r,
                                                                                     q[('dm',1,1)],
                                                                                     v,
                                                                                     q[('a',1,1)],
                                                                                     self.tau[0],
                                                                                     self.tau[1],
                                                                                     q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau(q['inverse(J)'],
                                                                                  self.q_dmt_r,
                                                                                  q[('dm',1,1)],
                                                                                  v,
                                                                                  q[('a',1,1)],
                                                                                  self.tau[0],
                                                                                  self.tau[1],
                                                                                  q[('cfl',0)])
            tau0=self.tau[0]
            tau1=self.tau[1]
            csubgridError.calculateSubgridErrorNavierStokes3D_GLS_tauRes(tau0,
                                                                         tau1,
                                                                         q[('pdeResidual',0)],
                                                                         q[('dpdeResidual',0,1)],
                                                                         q[('dpdeResidual',0,2)],
                                                                         q[('dpdeResidual',0,3)],
                                                                         q[('pdeResidual',1)],
                                                                         q[('dpdeResidual',1,0)],
                                                                         q[('dpdeResidual',1,1)],
                                                                         q[('dpdeResidual',1,2)],
                                                                         q[('dpdeResidual',1,3)],
                                                                         q[('pdeResidual',2)],
                                                                         q[('dpdeResidual',2,0)],
                                                                         q[('dpdeResidual',2,1)],
                                                                         q[('dpdeResidual',2,2)],
                                                                         q[('dpdeResidual',2,3)],
                                                                         q[('pdeResidual',3)],
                                                                         q[('dpdeResidual',3,0)],
                                                                         q[('dpdeResidual',3,1)],
                                                                         q[('dpdeResidual',3,2)],
                                                                         q[('dpdeResidual',3,3)],
                                                                         q[('subgridError',0)],
                                                                         q[('dsubgridError',0,1)],
                                                                         q[('dsubgridError',0,2)],
                                                                         q[('dsubgridError',0,3)],
                                                                         q[('subgridError',1)],
                                                                         q[('dsubgridError',1,0)],
                                                                         q[('dsubgridError',1,1)],
                                                                         q[('dsubgridError',1,2)],
                                                                         q[('dsubgridError',1,3)],
                                                                         q[('subgridError',2)],
                                                                         q[('dsubgridError',2,0)],
                                                                         q[('dsubgridError',2,1)],
                                                                         q[('dsubgridError',2,2)],
                                                                         q[('dsubgridError',2,3)],
                                                                         q[('subgridError',3)],
                                                                         q[('dsubgridError',3,0)],
                                                                         q[('dsubgridError',3,1)],
                                                                         q[('dsubgridError',3,2)],
                                                                         q[('dsubgridError',3,3)])
            if self.noPressureStabilization:
                q[('subgridError',0)][:]=0.0
                q[('dsubgridError',0,1)][:]=0.0
                q[('dsubgridError',0,2)][:]=0.0
                q[('dsubgridError',0,3)][:]=0.0
        for ci in range(self.nd):
            q[('cfl',ci+1)][:] = q[('cfl',0)]
#mwf orig
#         if self.nd == 2:
#             if self.coefficients.sd:
#                 csubgridError.calculateSubgridErrorNavierStokes2D_generic_withBodyForce_tau_sd(q['inverse(J)'],
#                                                                                                q[('dmt',1,1)],
#                                                                                                q[('dm',1,1)],
#                                                                                                q[('df',1,1)],
#                                                                                                q[('a',1,1)],
#                                                                                                q[('dr',1,1)],
#                                                                                                self.tau[0],
#                                                                                                self.tau[1],
#                                                                                                q[('cfl',0)])
#             else:
#                 csubgridError.calculateSubgridErrorNavierStokes2D_generic_withBodyForce_tau(q['inverse(J)'],
#                                                                                             q[('dmt',1,1)],
#                                                                                             q[('dm',1,1)],
#                                                                                             q[('df',1,1)],
#                                                                                             q[('a',1,1)],
#                                                                                             q[('dr',1,1)],
#                                                                                             self.tau[0],
#                                                                                             self.tau[1],
#                                                                                             q[('cfl',0)])
#             if self.lag:#TODO: make sure up to date with delaySteps flag
#                 tau0=self.tau_last[0]
#                 tau1=self.tau_last[1]
#             else:
#                 tau0=self.tau[0]
#                 tau1=self.tau[1]
#             csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tauRes(tau0,
#                                                                          tau1,
#                                                                          q[('pdeResidual',0)],
#                                                                          q[('dpdeResidual',0,1)],
#                                                                          q[('dpdeResidual',0,2)],
#                                                                          q[('pdeResidual',1)],
#                                                                          q[('dpdeResidual',1,0)],
#                                                                          q[('dpdeResidual',1,1)],
#                                                                          q[('pdeResidual',2)],
#                                                                          q[('dpdeResidual',2,0)],
#                                                                          q[('dpdeResidual',2,2)],
#                                                                          q[('subgridError',0)],
#                                                                          q[('dsubgridError',0,1)],
#                                                                          q[('dsubgridError',0,2)],
#                                                                          q[('subgridError',1)],
#                                                                          q[('dsubgridError',1,0)],
#                                                                          q[('dsubgridError',1,1)],
#                                                                          q[('subgridError',2)],
#                                                                          q[('dsubgridError',2,0)],
#                                                                          q[('dsubgridError',2,2)])
#         elif self.nd == 3:
#             return NavierStokesASGS_velocity_pressure.calculateSubgridError(q)

class StokesASGS_velocity_pressure(SGE_base):
    def __init__(self,coefficients,nd):
        SGE_base.__init__(self,coefficients,nd,lag=False)
        coefficients.stencil[0].add(0)
        if nd == 2:
            coefficients.stencil[1].add(2)
            coefficients.stencil[2].add(1)
        elif nd == 3:
            coefficients.stencil[1].add(2)
            coefficients.stencil[1].add(3)
            coefficients.stencil[2].add(1)
            coefficients.stencil[2].add(3)
            coefficients.stencil[3].add(1)
            coefficients.stencil[3].add(3)
    def calculateSubgridError(self,q):
        if self.nd == 2:
            if self.coefficients.sd:
                csubgridError.calculateSubgridErrorStokes_GLS_tau_sd(self.mesh.elementDiametersArray,
                                                                     q[('dH',1,0)],
                                                                     q[('a',1,1)],
                                                                     self.tau[0],
                                                                     self.tau[1])
            else:
                csubgridError.calculateSubgridErrorStokes_GLS_tau(self.mesh.elementDiametersArray,
                                                                  q[('dH',1,0)],
                                                                  q[('a',1,1)],
                                                                  self.tau[0],
                                                                  self.tau[1])
            csubgridError.calculateSubgridErrorStokes2D_GLS_tauRes(self.tau[0],
                                                                   self.tau[1],
                                                                   q[('pdeResidual',0)],
                                                                   q[('dpdeResidual',0,1)],
                                                                   q[('dpdeResidual',0,2)],
                                                                   q[('pdeResidual',1)],
                                                                   q[('dpdeResidual',1,0)],
                                                                   q[('dpdeResidual',1,1)],
                                                                   q[('pdeResidual',2)],
                                                                   q[('dpdeResidual',2,0)],
                                                                   q[('dpdeResidual',2,2)],
                                                                   q[('subgridError',0)],
                                                                   q[('dsubgridError',0,1)],
                                                                   q[('dsubgridError',0,2)],
                                                                   q[('subgridError',1)],
                                                                   q[('dsubgridError',1,0)],
                                                                   q[('dsubgridError',1,1)],
                                                                   q[('subgridError',2)],
                                                                   q[('dsubgridError',2,0)],
                                                                   q[('dsubgridError',2,2)])
        elif self.nd == 3:
            if self.coefficients.sd:
                csubgridError.calculateSubgridErrorStokes_GLS_tau_sd(self.mesh.elementDiametersArray,
                                                                     q[('dH',1,0)],
                                                                     q[('a',1,1)],
                                                                     self.tau[0],
                                                                     self.tau[1])
            else:
                csubgridError.calculateSubgridErrorStokes_GLS_tau(self.mesh.elementDiametersArray,
                                                                  q[('dH',1,0)],
                                                                  q[('a',1,1)],
                                                                  self.tau[0],
                                                                  self.tau[1])
                self.tau[0][:] = 0.0
                csubgridError.calculateSubgridErrorStokes3D_GLS_tauRes(self.tau[0],
                                                                       self.tau[1],
                                                                       q[('pdeResidual',0)],
                                                                       q[('dpdeResidual',0,1)],
                                                                       q[('dpdeResidual',0,2)],
                                                                       q[('dpdeResidual',0,3)],
                                                                       q[('pdeResidual',1)],
                                                                       q[('dpdeResidual',1,0)],
                                                                       q[('dpdeResidual',1,1)],
                                                                       q[('pdeResidual',2)],
                                                                       q[('dpdeResidual',2,0)],
                                                                       q[('dpdeResidual',2,2)],
                                                                       q[('pdeResidual',3)],
                                                                       q[('dpdeResidual',3,0)],
                                                                       q[('dpdeResidual',3,3)],
                                                                       q[('subgridError',0)],
                                                                       q[('dsubgridError',0,1)],
                                                                       q[('dsubgridError',0,2)],
                                                                       q[('dsubgridError',0,3)],
                                                                       q[('subgridError',1)],
                                                                       q[('dsubgridError',1,0)],
                                                                       q[('dsubgridError',1,1)],
                                                                       q[('subgridError',2)],
                                                                       q[('dsubgridError',2,0)],
                                                                       q[('dsubgridError',2,2)],
                                                                       q[('subgridError',3)],
                                                                       q[('dsubgridError',3,0)],
                                                                       q[('dsubgridError',3,3)])

class TwophaseStokes_LS_FC_ASGS(SGE_base):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False):
        self.nc = coefficients.nc
        self.nd = nd
        self.components=list(range(self.nc))
        self.lag=lag
        self.stabilizationFlag = stabFlag
        coefficients.stencil[0].add(0)
    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.tau=[]
        self.tau_last=[]
        if self.lag:
            self.tau_last = numpy.zeros(cq[('u',0)].shape,'d')
            self.tau = numpy.zeros(cq[('u',0)].shape,'d')
        else:
            self.tau = numpy.zeros(cq[('u',0)].shape,'d')
    def calculateSubgridError(self,q):
        csubgridError.calculateSubgridError_A_tau(self.stabilizationFlag,
                                                  self.mesh.elementDiametersArray,
                                                  q[('dmt',0,0)],
                                                  q[('df',0,0)],
                                                  q[('cfl',0)],
                                                  self.tau)
        if self.lag:
            tau = self.tau_last
        else:
            tau = self.tau
        csubgridError.calculateSubgridError_tauRes(tau,
                                                   q[('pdeResidual',0)],
                                                   q[('dpdeResidual',0,0)],
                                                   q[('subgridError',0)],
                                                   q[('dsubgridError',0,0)])
        csubgridError.calculateSubgridErrorStokes2D_GLS_velocity(self.mesh.elementDiametersArray,
                                                                 q[('a',2,2)],
                                                                 q[('pdeResidual',2)],
                                                                 q[('dpdeResidual',2,1)],
                                                                 q[('dpdeResidual',2,2)],
                                                                 q[('pdeResidual',3)],
                                                                 q[('dpdeResidual',3,1)],
                                                                 q[('dpdeResidual',3,3)],
                                                                 q[('subgridError',2)],
                                                                 q[('dsubgridError',2,1)],
                                                                 q[('dsubgridError',2,2)],
                                                                 q[('subgridError',3)],
                                                                 q[('dsubgridError',3,1)],
                                                                 q[('dsubgridError',3,3)])
    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag is not None:
            self.tau_last[:] = self.tau

class ShallowWater_CFL(SGE_base):
    def __init__(self,coefficients,nd,g):
        SGE_base.__init__(self,coefficients,nd,lag=False)
        self.g=g
        self.nc=nd+1
        self.nd=nd
    def calculateSubgridError(self,q):
        if self.nd==1:
            csubgridError.calculateSubgridErrorShallowWater1D(self.g,
                                                              self.mesh.elementDiametersArray,
                                                              q[('u',0)],
                                                              q[('u',1)],
                                                              q[('cfl',0)],
                                                              q[('cfl',1)])
        if self.nd==2:
            csubgridError.calculateSubgridErrorShallowWater2D(self.g,
                                                              self.mesh.elementDiametersArray,
                                                              q[('u',0)],
                                                              q[('u',1)],
                                                              q[('u',2)],
                                                              q[('cfl',0)],
                                                              q[('cfl',1)],
                                                              q[('cfl',2)])

class SkewStabilization_1(object):
    def __init__(self,mesh,nc,nd):
        self.mesh = mesh
        self.nc = nc
        self.nd = nd
    def calculateSubgridError(self,q):
        nc = self.nc
        for ci in range(self.nc):
            vfemIntegrals.calculateSubgridErrorScalarADR_1(self.mesh.elementDiametersArray,
                                                           q[('df',ci,nc-1-ci)],
                                                           q[('a',ci,nc-1-ci)],
                                                           q[('da',ci,nc-1-ci,nc-1-ci)],
                                                           q[('grad(phi)',nc-1-ci)],
                                                           q[('dphi',nc-1-ci,nc-1-ci)],
                                                           q[('dr',ci,nc-1-ci)],
                                                           q[('dmt',ci,nc-1-ci)],
                                                           q[('pe',ci)],
                                                           q[('cfl',ci)],
                                                           q[('pdeResidual',ci)],
                                                           q[('dpdeResidual',ci,nc-1-ci)],
                                                           q[('subgridError',ci)],
                                                           q[('dsubgridError',ci,nc-1-ci)])



class AdvectionDiffusionReactionTransientSubscales_ASGS(AdvectionDiffusionReaction_ASGS):
    r"""
    track subgrid scales in time with Backward Euler

    .. math::

       \delta u^{n+1} &= -\tau_t\tilde{R}_h

       \tilde{R}_h &= R_h - m^{\prime,k}\frac{\delta u^{n}}{\Delta t^{n+1}}

       \tau_t &= \frac{\Delta t^{n+1}\tau_s}{m^{prime,n+1}\tau_s + \Delta t^{n+1}}

       \tau_s &= \text{normal spatial tau, supposed to have } \tau_s \approx \mathcal{L}^{-1}_{s}

    for now m^{prime} evaluated at k=n for subgrid error but not sure if this is right or not

    TODO:
      Check Peclet  number calculation in generic tau and cfl calculation, what's returned in cfl array
        (advective or max of advective,diffusive stab. constraint)

      FLCBDF seems less happy with tracking subgrid scales than without tracking
    """
    def __init__(self,coefficients,nd,stabFlag='1',lag=False,trackSubScales=False,useHarariDirectly=False,
                 limit_tau_t=False,tau_t_limit_min=0.0,tau_t_limit_max=1.0):
        AdvectionDiffusionReaction_ASGS.__init__(self,coefficients,nd,stabFlag=stabFlag,lag=lag)
        self.trackSubScales=trackSubScales
        self.timeIntegration = None
        self.useHarariDirectly = useHarariDirectly
        #apply bounds to tau_t?
        self.limit_tau_t = limit_tau_t
        self.tau_t_limit_min = tau_t_limit_min
        self.tau_t_limit_max = tau_t_limit_max
    def initializeElementQuadrature(self,mesh,t,cq):
        AdvectionDiffusionReaction_ASGS.initializeElementQuadrature(self,mesh,t,cq)
        import copy
        self.subgridError_last=[]
        self.subgridErrorMassCoef_last = []
        self.subgridTmp = []; self.subgridTmp2 = []
        for ci in range(self.nc):
            self.subgridTmp.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            if self.trackSubScales:
                self.subgridError_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.subgridErrorMassCoef_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.subgridTmp2.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            else:
                self.subgridError_last.append(None)
                self.subgridErrorMassCoef_last.append(None)
    def initializeTimeIntegration(self,timeIntegration):
        """
        allow for connection with time integration method if tracking subscales
        """
        self.timeIntegration = timeIntegration
    def updateSubgridErrorHistory(self,initializationPhase=False):
        AdvectionDiffusionReaction_ASGS.updateSubgridErrorHistory(self,initializationPhase=initializationPhase)

        if self.trackSubScales:
            for ci in range(self.nc):
                if not initializationPhase:
                    #we are storing subgridError = tau*Res so need to reverse sign
                    self.subgridError_last[ci].flat[:] = self.cq[('subgridError',ci)].flat
                    self.subgridError_last[ci] *= -1.0
                    #mwf debug
                    logEvent("ADR_ASGS tracksubscales updateSubgridErrorHistory max subgridError = %s " % (self.subgridError_last[ci].max()),10)
                #how are we going to define subgrid mass?
                self.subgridErrorMassCoef_last[ci].flat[:] = self.cq[('dm',ci,ci)].flat
    def calculateSubgridError(self,q):
        for ci in range(self.nc):
            #mwf need to calculate tau_s without dm/dt
            mttmp = q[('dmt',ci,ci)]
            if self.trackSubScales:
                self.subgridTmp[ci].fill(0.0)
                mttmp = self.subgridTmp[ci]
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_generic_tau_sd(self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                       q['inverse(J)'],
                                                                       mttmp,
                                                                       q[('df',ci,ci)],
                                                                       q[('a',ci,ci)],
                                                                       q[('da',ci,ci,ci)],
                                                                       q[('grad(phi)',ci)],
                                                                       q[('dphi',ci,ci)],
                                                                       q[('dr',ci,ci)],
                                                                       q[('pe',ci)],
                                                                       q[('cfl',ci)],
                                                                       self.tau[ci])
            else:
                csubgridError.calculateSubgridError_ADR_generic_tau(q['inverse(J)'],
                                                                    mttmp,
                                                                    q[('df',ci,ci)],
                                                                    q[('a',ci,ci)],
                                                                    q[('da',ci,ci,ci)],
                                                                    q[('grad(phi)',ci)],
                                                                    q[('dphi',ci,ci)],
                                                                    q[('dr',ci,ci)],
                                                                    q[('pe',ci)],
                                                                    q[('cfl',ci)],
                                                                    self.tau[ci])
            if self.lag:
                tau=self.tau_last[ci]
                #dm_subgrid = self.subgridErrorMassCoef_last[ci]
            else:
                tau=self.tau[ci]
                #dm_subgrid = q[('dm',ci,ci)]
            dm_subgrid = self.cq[('dm_sge',ci,ci)]
            if self.trackSubScales:
                #mwf debug
                logEvent("ADR_ASGS trackScales before transient modficication (tau_s) tau[ci].max= %s tau[ci].min=%s  " % (tau[ci].max(),tau[ci].min()),10)
                #tau here should be the same as tau_t in Codina's formalism if dmdt is included?
                #calculate \tilde{R}_h = R_h - \delta m^{n}/dt^{n+1}
                self.subgridTmp[ci][:] = self.subgridError_last[ci]

                dt = self.timeIntegration.dt
                assert dt > 0.0
                dtInv = 1.0/dt

                self.subgridTmp[ci] *= dtInv
                self.subgridTmp[ci] *= self.subgridErrorMassCoef_last[ci]#decide what time level to use
                q[('pdeResidual',ci)] -= self.subgridTmp[ci]  #R_h --> \tilde{R}_h

                if tau.max() > 0.0:
                    self.subgridTmp[ci][:] = tau
                    self.subgridTmp[ci] *= dt
                    self.subgridTmp2[ci][:] = tau
                    self.subgridTmp2[ci] *= dm_subgrid
                    self.subgridTmp2[ci] += dt
                    self.subgridTmp[ci] /= self.subgridTmp2[ci]
                    if self.coefficients.sd and self.useHarariDirectly:
                        csubgridError.calculateSubgridError_Harari_tau_sd(self.nd,dt,
                                                                          self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                          self.mesh.elementDiametersArray,
                                                                          q[('a',ci,ci)],
                                                                          self.subgridTmp[ci])

                    #bound tau_t based on dt size
                    if self.limit_tau_t:
                        numpy.clip(self.subgridTmp[ci],self.tau_t_limit_min*dt,self.tau_t_limit_max*dt,self.subgridTmp[ci])
                    tau = self.subgridTmp[ci]
                    #mwf debug
                    logEvent("ADR_ASGS trackScales after modifying tau[ci].max= %s tau[ci].min= %s " % (tau[ci].max(),tau[ci].min()),10)
                    #mwf should be 1.0/m'
                    assert tau.max() * dm_subgrid.max() /dt <= 1.0, "Subgrid scales, modified tau_t.max() = %s dt = %s dm_subgrid.max() = %s tau.m'/dt = %s must be less than 1 " % (tau.max(),
                                                                                                                                                                                     dt,
                                                                                                                                                                                     dm_subgrid.max(),
                                                                                                                                                                                     tau.max()/dt)
                #
            for cj in range(self.nc):
                if ('dpdeResidual',ci,cj) in q:
                    csubgridError.calculateSubgridError_tauRes(tau,
                                                               q[('pdeResidual',ci)],
                                                               q[('dpdeResidual',ci,cj)],
                                                               q[('subgridError',ci)],
                                                               q[('dsubgridError',ci,cj)])
                    #mwf debug
                    logEvent("ADR_ASGS pdeResidual[ci].max = %s subgridError.max = %s subgridError.min= %s " % (q[('pdeResidual',ci)].max(),
                                                                                                           q[('subgridError',ci)].max(),
                                                                                                           q[('subgridError',ci)].min()),10)
    def accumulateSubgridMassHistory(self,q):
        """
        incorporate subgrid scale mass accumulation
        \\delta m^{n}/\\delta t^{n+1}
        """
        if self.trackSubScales:
            for ci in range(self.nc):
                self.subgridTmp[ci][:] = self.subgridError_last[ci]
                dt = self.timeIntegration.dt
                assert dt > 0.0
                dtInv = 1.0/dt
                self.subgridTmp[ci] *= dtInv
                self.subgridTmp[ci] *= self.subgridErrorMassCoef_last[ci]#decide how to approximate
                logEvent("ADR trackSubScales accumulating delta u^n.abs.max= %s dm.max=%s  " % (max(numpy.absolute(self.subgridTmp[ci].flat)),
                                                                                           max(numpy.absolute(self.subgridErrorMassCoef_last[ci].flat))),10)

                q[('mt',ci)] -= self.subgridTmp[ci]


class AdvectionDiffusionReactionHaukeSangalliInterpolant_ASGS(SGE_base):
    """
    Should be basic Hauke Sangalli approach but computes terms at interpolation points
    and then uses this to compute the gradient for Sangalli type approach
    Adjoint gradient is computed manually
    """
    def __init__(self,coefficients,nd,stabFlag='1',lag=False,interpolationFemSpaceType=None,tau_00_force=None,
                 tau_11_force=None):
        SGE_base.__init__(self,coefficients,nd,lag)
        self.stabilizationFlag = stabFlag
        self.interpolationFemSpaceType = interpolationFemSpaceType
        assert self.interpolationFemSpaceType is not None
        self.usesFEMinterpolant = True
        self.usesGradientStabilization = True
        self.tau_00_force=tau_00_force; self.tau_11_force = tau_11_force
    def initializeElementQuadrature(self,mesh,t,cq,cip=None):
        """

        """
        SGE_base.initializeElementQuadrature(self,mesh,t,cq)
        import copy
        self.cq=cq
        self.cip=cip
        assert self.cip is not None
        self.tau_gradient = []
        self.tau_gradient_last = []
        self.subgridTmp = [];
        self.subgridTmp_ip = [];
        self.grad_u_last = []
        for ci in range(self.nc):
            self.subgridTmp.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            self.subgridTmp_ip.append(numpy.zeros(cip[('u',ci)].shape,'d'))
            self.grad_u_last.append(numpy.zeros(cq[('grad(u)',ci)].shape,'d'))
            if self.lag:
                if ('df',ci,ci) in cip:
                    cip[('df_sge',ci,ci)] = copy.deepcopy(cip[('df',ci,ci)])
                if ('dm',ci,ci) in cip:
                    cip[('dm_sge',ci,ci)] = copy.deepcopy(cip[('dm',ci,ci)])
                if ('dmt',ci,ci) in cip:
                    cip[('dmt_sge',ci,ci)] = copy.deepcopy(cip[('dmt',ci,ci)])
                #
                if ('df',ci,ci) in cq:
                    cq[('df_sge',ci,ci)] = copy.deepcopy(cq[('df',ci,ci)])
                if ('dm',ci,ci) in cq:
                    cq[('dm_sge',ci,ci)] = copy.deepcopy(cq[('dm',ci,ci)])
                if ('dmt',ci,ci) in cq:
                    cq[('dmt_sge',ci,ci)] = copy.deepcopy(cq[('dmt',ci,ci)])
            else:
                if ('df',ci,ci) in cip:
                    cip[('df_sge',ci,ci)] = cip[('df',ci,ci)]
                if ('dm',ci,ci) in cip:
                    cip[('dm_sge',ci,ci)] = cip[('dm',ci,ci)]
                if ('dmt',ci,ci) in cip:
                    cip[('dmt_sge',ci,ci)] = cip[('dmt',ci,ci)]
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                #
                if self.lag:#mwf looks like this was missing if lag May 7 09
                    cip[('grad(phi)_sge',ck)]=copy.deepcopy(cip[('grad(phi)',ck)])
                    for cj in list(cjDict.keys()):
                        cip[('dphi_sge',ck,cj)]=copy.deepcopy(cip[('dphi',ck,cj)])
                        cip[('da_sge',ci,ck,cj)]=copy.deepcopy(cip[('da',ci,ck,cj)])
                    #
                    cq[('grad(phi)_sge',ck)]=copy.deepcopy(cq[('grad(phi)',ck)])
                    for cj in list(cjDict.keys()):
                        cq[('dphi_sge',ck,cj)]=copy.deepcopy(cq[('dphi',ck,cj)])
                        cq[('da_sge',ci,ck,cj)]=copy.deepcopy(cq[('da',ci,ck,cj)])
                else:
                    cip[('grad(phi)_sge',ck)]=cip[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        cip[('dphi_sge',ck,cj)]=cip[('dphi',ck,cj)]
                        cip[('da_sge',ci,ck,cj)]=cip[('da',ci,ck,cj)]

        #
        self.interpolationSpace = {}; self.strongResidualInterpolant = {};
        if self.interpolationFemSpaceType is not None:
            for ci in range(self.nc):
                self.interpolationSpace[ci] = self.interpolationFemSpaceType(self.mesh.subdomainMesh,self.nd)
                self.strongResidualInterpolant[ci] = FemTools.FiniteElementFunction(self.interpolationSpace[ci])
        if self.usesGradientStabilization == True:
            for ci in range(self.nc):
                cq[('grad(pdeResidual)',ci)]= numpy.zeros(cq[('grad(u)',ci)].shape,'d')
                cq[('grad(subgridError)',ci)]= numpy.zeros(cq[('grad(u)',ci)].shape,'d')
                #mwf hack just make a scalar to test Jacobian
                for cj in range(self.nc):
                    cq[('dgrad(subgridError)',ci,cj)]= numpy.zeros(cq[('u',ci)].shape,'d')
                self.tau_gradient.append(numpy.zeros(self.tau[ci].shape,'d'))
                if self.lag:
                    self.tau_gradient_last.append(numpy.zeros(self.tau_last[ci].shape,'d'))

    def initializeTimeIntegration(self,timeIntegration):
        """
        allow for connection with time integration method if tracking subscales
        """
        self.timeIntegration = timeIntegration
    def calculateSubgridErrorInterpolants(self,ci):
        """
        should interpolate strong residual. One problem is that
        strong residual is discontinuous when grad(u) terms are nonzero
        so standard C0 projection
        won't necessarily be what we expect locally on each element
        for C0, P1 and linear problem with constant coefficients
        computing gradient locally should be just the same as
        ignoring gradient terms altogether
        """
        #now project to finite element space
        if self.usesGradientStabilization:
            #mwf hack!
            #self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.cip[('pdeResidual',ci)])
            #self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.cip[('mt',ci)])
            #self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.cip[('m',ci)]/self.timeIntegration.dt)
            #self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.cip[('r',ci)])
            self.subgridTmp_ip[ci].fill(0.0)
            if ('mt',ci) in self.cip:
                self.subgridTmp_ip[ci] += self.cip[('mt',ci)]
            if ('r',ci) in self.cip:
                self.subgridTmp_ip[ci] += self.cip[('r',ci)]
            self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.subgridTmp_ip[ci])
    def calculateSubgridError(self,q):

        #compute basic ASGS stabilization as before
        for ci in range(self.nc):
            self.calculateSubgridErrorInterpolants(ci)
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_Sangalli_tau_sd(self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                        q['inverse(J)'],
                                                                        q[('dmt',ci,ci)],
                                                                        q[('df',ci,ci)],
                                                                        q[('a',ci,ci)],
                                                                        q[('da',ci,ci,ci)],
                                                                        q[('grad(phi)',ci)],
                                                                        q[('dphi',ci,ci)],
                                                                        q[('dr',ci,ci)],
                                                                        q[('pe',ci)],
                                                                        q[('cfl',ci)],
                                                                        self.tau[ci],
                                                                        self.tau_gradient[ci])
            else:
                assert False
            if self.lag:
                tau=self.tau_last[ci]
                tau_gradient=self.tau_gradient_last[ci]
                #have to figure out way to update dmt_sge if lagging

            else:
                tau=self.tau[ci]
                tau_gradient = self.tau_gradient[ci]
            #mwf hack ...
            if self.coefficients.sd and False:
                logEvent("HaukeSangalli Hack switching from tau.max()= %s tau.min()= %s to " % (tau[ci].max(),tau[ci].min()),1)
                csubgridError.calculateSubgridError_ADR_generic_tau_sd(self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                       q['inverse(J)'],
                                                                       q[('dmt',ci,ci)],
                                                                       q[('df',ci,ci)],
                                                                       q[('a',ci,ci)],
                                                                       q[('da',ci,ci,ci)],
                                                                       q[('grad(phi)',ci)],
                                                                       q[('dphi',ci,ci)],
                                                                       q[('dr',ci,ci)],
                                                                       q[('pe',ci)],
                                                                       q[('cfl',ci)],
                                                                       self.tau[ci])
                tau = self.tau[ci]
                logEvent("Generic tau is tau.max() =%s tau.min() = %s to " % (tau[ci].max(),tau[ci].min()),1)
            #mwf hack
            if self.tau_00_force is not None:
                tau.fill(self.tau_00_force)
            if self.tau_11_force is not None:
                tau_gradient.fill(self.tau_11_force)

            for cj in range(self.nc):
                if ('dpdeResidual',ci,cj) in q:
                    csubgridError.calculateSubgridError_tauRes(tau,
                                                               q[('pdeResidual',ci)],
                                                               q[('dpdeResidual',ci,cj)],
                                                               q[('subgridError',ci)],
                                                               q[('dsubgridError',ci,cj)])


            for ci in range(self.nc):
                #this is the general way but right now we're having a problem when we interpolate
                #the actual residual because of the discontinuous gradient terms
                self.strongResidualInterpolant[ci].getGradientValues(q[('grad(v)',ci)],
                                                                     q[('grad(pdeResidual)',ci)])
                #mwf hack to test calculation
                q[('grad(pdeResidual)',ci)].flat[:] = q[('grad(u)',ci)].flat
                q[('grad(pdeResidual)',ci)]        -= self.grad_u_last[ci]
                q[('grad(pdeResidual)',ci)]        /= self.timeIntegration.dt
                for cj in range(self.nc):
                    if ('dpdeResidual',ci,cj) in q:
                        csubgridError.calculateSubgridErrorGradient_tauRes(tau_gradient,
                                                                           q[('grad(pdeResidual)',ci)],
                                                                           q[('grad(subgridError)',ci)])
                        #have got to come up with way to handle jacobian
                        q[('dgrad(subgridError)',ci,cj)].flat[:] = q[('dmt',ci,cj)].flat
                        q[('dgrad(subgridError)',ci,cj)] +=  q[('dr',ci,cj)]
                        self.subgridTmp[ci].flat[:] =  q[('dmt_sge',ci,cj)].flat
                        self.subgridTmp[ci] +=    q[('dr',ci,cj)]
                        q[('dgrad(subgridError)',ci,cj)] *= self.subgridTmp[ci]
                        q[('dgrad(subgridError)',ci,cj)] *= tau_gradient
                        q[('dgrad(subgridError)',ci,cj)] *= -1.0
                        #below works for just dmt approx for residual
                        # q[('dgrad(subgridError)',ci,cj)].flat[:] = tau_gradient.flat
                        # q[('dgrad(subgridError)',ci,cj)] *= -1.0
                        # q[('dgrad(subgridError)',ci,cj)] *= q[('dmt_sge',ci,cj)]
                        # q[('dgrad(subgridError)',ci,cj)] *= q[('dmt',ci,cj)]

            logEvent("HaukeSangalli ADR tau_00.max() = %s tau_11.max() = %s grad(pdeResidual).max= %s grad(subgridError).max= %s dgrad(subgridError).max= %s " % (tau.max(),tau_gradient.max(),
                                                                                                                                                            q[('grad(pdeResidual)',ci)].max(),
                                                                                                                                                            q[('grad(subgridError)',ci)].max(),
                                                                                                                                                            q[('dgrad(subgridError)',ci,ci)].max()),1)
            #mwf debug
            #import pdb
            #pdb.set_trace()
#             print "tau",tau
#             print "pdeResidual",q[('pdeResidual',ci)]
#             print "dpdeResidual",q[('dpdeResidual',ci,ci)]
#             print "subgrid error",q[('subgridError',ci)]
#             print "dsubgrid error",q[('dsubgridError',ci,ci)]
    def updateSubgridErrorHistory(self,initializationPhase=False):
        if self.lag:
            for ci in range(self.nc):
                self.tau_last[ci][:] = self.tau[ci]
                self.tau_gradient_last[ci][:] = self.tau_gradient[ci]
                self.cip[('df_sge',ci,ci)][:] = self.cip[('df',ci,ci)]
                self.cip[('dm_sge',ci,ci)][:] = self.cip[('dm',ci,ci)]
                #
                self.cq[('df_sge',ci,ci)][:] = self.cq[('df',ci,ci)]
                self.cq[('dm_sge',ci,ci)][:] = self.cq[('dm',ci,ci)]
            for ci,ckDict in self.coefficients.diffusion.items():
                for ck,cjDict in ckDict.items():
                    self.cip[('grad(phi)_sge',ck)][:]=self.cip[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        self.cip[('dphi_sge',ck,cj)][:]=0.0 #grad(phi) will be a constant when lagged so dphi=0 not 1
                        self.cip[('da_sge',ci,ck,cj)][:]=self.cip[('da',ci,ck,cj)]
                for ck,cjDict in ckDict.items():
                    self.cq[('grad(phi)_sge',ck)][:]=self.cq[('grad(phi)',ck)]
                    for cj in list(cjDict.keys()):
                        self.cq[('dphi_sge',ck,cj)][:]=0.0 #grad(phi) will be a constant when lagged so dphi=0 not 1
                        self.cq[('da_sge',ci,ck,cj)][:]=self.cq[('da',ci,ck,cj)]
                #
            #
            #mwf hack to test
            self.grad_u_last[ci].flat[:] = self.cq[('grad(u)',ci)].flat
    #

class AdvectionDiffusionReactionHaukeSangalliInterpolantWithTransientSubScales_ASGS(AdvectionDiffusionReactionHaukeSangalliInterpolant_ASGS):
    """
    Should be basic Hauke Sangalli approach but computes terms at interpolation points
    and then uses this to compute the gradient for Sangalli type approach
    Adjoint gradient is computed manually
    And SubScales are tracked in time
    """
    def __init__(self,coefficients,nd,stabFlag='1',lag=False,interpolationFemSpaceType=None,trackSubScales=False,tau_00_force=None,
                 tau_11_force=None,includeSubgridScalesInGradientStabilization=True):
        AdvectionDiffusionReactionHaukeSangalliInterpolant_ASGS.__init__(self,coefficients,nd,lag,
                                                                         interpolationFemSpaceType=interpolationFemSpaceType,tau_00_force=tau_00_force,
                                                                         tau_11_force=tau_11_force)
        self.trackSubScales = trackSubScales
        self.includeSubgridScalesInGradientStabilization = includeSubgridScalesInGradientStabilization
    def initializeElementQuadrature(self,mesh,t,cq,cip=None):
        """

        """
        AdvectionDiffusionReactionHaukeSangalliInterpolant_ASGS.initializeElementQuadrature(self,mesh,t,cq,cip)
        import copy
        self.subgridError_last=[]
        self.subgridErrorMassCoef_last = []
        self.subgridTmp = []; self.subgridTmp2 = []
        self.subgridError_ip_last=[]
        self.subgridErrorMassCoef_ip_last = []
        self.subgridTmp_ip = []; self.subgridTmp2_ip = []
        self.tau_ip = [] ; self.tau_ip_last = []
        self.tau_gradient_ip = [] ; self.tau_gradient_ip_last = []
        for ci in range(self.nc):
            self.subgridTmp.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            self.subgridTmp_ip.append(numpy.zeros(cip[('u',ci)].shape,'d'))
            if self.trackSubScales:
                self.subgridError_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.subgridErrorMassCoef_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.subgridTmp2.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                #
                self.subgridError_ip_last.append(numpy.zeros(cip[('u',ci)].shape,'d'))
                self.subgridErrorMassCoef_ip_last.append(numpy.zeros(cip[('u',ci)].shape,'d'))
                self.subgridTmp2_ip.append(numpy.zeros(cip[('u',ci)].shape,'d'))
                self.tau_ip.append(copy.deepcopy(self.tau[ci]))
                self.tau_gradient_ip.append(copy.deepcopy(self.tau_gradient[ci]))
                if self.lag:
                    self.tau_ip_last.append(copy.deepcopy(self.tau_last[ci]))
                    self.tau_gradient_ip_last.append(copy.deepcopy(self.tau_gradient_last[ci]))

            else:
                self.subgridError_last.append(None)
                self.subgridErrorMassCoef_last.append(None)
                self.subgridError_ip_last.append(None)
                self.subgridErrorMassCoef_ip_last.append(None)
    def calculateSubgridErrorInterpolants(self,ci):
        #now project to finite element space
        hack = False
        if self.usesGradientStabilization:
            if hack:#mwf hack!
                #self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.cip[('pdeResidual',ci)])
                #self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.cip[('mt',ci)])
                #self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.cip[('m',ci)]/self.timeIntegration.dt)
                self.subgridTmp_ip[ci].flat[:] = self.cip[('m',ci)]
                self.subgridTmp_ip[ci] /= self.timeIntegration.dt
            else:
                self.subgridTmp_ip[ci].fill(0.0)
                if ('mt',ci) in self.cip:
                    self.subgridTmp_ip[ci] += self.cip[('mt',ci)]
                if ('r',ci) in self.cip:
                    self.subgridTmp_ip[ci] += self.cip[('r',ci)]
            if self.includeSubgridScalesInGradientStabilization:
                #unless accumulate subgrid term has been callled this will miss old subgrid mass
                dt = self.timeIntegration.dt
                assert dt > 0.0
                dtInv = 1.0/dt
                self.subgridTmp2_ip[ci][:] = self.subgridError_ip_last[ci]
                self.subgridTmp2_ip[ci] *= dtInv
                self.subgridTmp2_ip[ci] *= self.subgridErrorMassCoef_ip_last[ci]#figure this out
                logEvent("HaukeSangalli pdeResidualInterpolant accumulating subgridHistory dt=%s subgridError_ip_last.max=%s subgridError_ip_last.min=%s " % (dt,
                                                                                                                                                         self.subgridError_ip_last[ci].max(),
                                                                                                                                                         self.subgridError_ip_last[ci].min()),1)
                #should be -=
                self.subgridTmp_ip[ci] -= self.subgridTmp2_ip[ci]
                #
            self.strongResidualInterpolant[ci].projectFromInterpolationConditions(self.subgridTmp_ip[ci])
    def calculateSubgridError(self,q):

        #calculate tau's
        for ci in range(self.nc):
            #calculate interpolant here if want gradient stabilization to be for R_h instead of \tilde{R}_h
            if not self.includeSubgridScalesInGradientStabilization:
                self.calculateSubgridErrorInterpolants(ci)
            if self.coefficients.sd:
                csubgridError.calculateSubgridError_ADR_Sangalli_tau_sd(self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                        q['inverse(J)'],
                                                                        q[('dmt',ci,ci)],
                                                                        q[('df',ci,ci)],
                                                                        q[('a',ci,ci)],
                                                                        q[('da',ci,ci,ci)],
                                                                        q[('grad(phi)',ci)],
                                                                        q[('dphi',ci,ci)],
                                                                        q[('dr',ci,ci)],
                                                                        q[('pe',ci)],
                                                                        q[('cfl',ci)],
                                                                        self.tau[ci],
                                                                        self.tau_gradient[ci])
            else:
                assert False
            if self.lag:
                tau=self.tau_last[ci]
                tau_gradient=self.tau_gradient_last[ci]
            else:
                tau=self.tau[ci]
                tau_gradient = self.tau_gradient[ci]

            if self.trackSubScales:
                #Repeat for interpolation points
                if self.coefficients.sd:
                    csubgridError.calculateSubgridError_ADR_Sangalli_tau_sd(self.coefficients.sdInfo[(ci,ci)][0],self.coefficients.sdInfo[(ci,ci)][1],
                                                                            self.cip['inverse(J)'],
                                                                            self.cip[('dmt',ci,ci)],
                                                                            self.cip[('df',ci,ci)],
                                                                            self.cip[('a',ci,ci)],
                                                                            self.cip[('da',ci,ci,ci)],
                                                                            self.cip[('grad(phi)',ci)],
                                                                            self.cip[('dphi',ci,ci)],
                                                                            self.cip[('dr',ci,ci)],
                                                                            self.cip[('pe',ci)],
                                                                            self.cip[('cfl',ci)],
                                                                            self.tau_ip[ci],
                                                                            self.tau_gradient_ip[ci])
                else:
                    assert False
                if self.lag:
                    tau_ip=self.tau_ip_last[ci]
                    tau_gradient_ip=self.tau_gradient_ip_last[ci]
                else:
                    tau_ip=self.tau[ci]
                    tau_gradient_ip = self.tau_gradient_ip[ci]
            #mwf hack
            if self.tau_00_force is not None:
                tau.fill(self.tau_00_force)
                if self.trackSubScales: tau_ip.fill(self.tau_00_force)
            if self.tau_11_force is not None:
                tau_gradient.fill(self.tau_11_force)
                if self.trackSubScales: tau_gradient_ip.fill(self.tau_11_force)

            if self.trackSubScales:
                #mwf debug
                print("HaukeSangalli_ASGS trackScales tau[ci].max= %s " % (tau[ci].max()))
                #
                #would be nice to have dt^{n+1} alone, try to get this from timeIntegration directly?
                dt = self.timeIntegration.dt
                assert dt > 0.0
                dtInv = 1.0/dt
                #calculate \tilde{R}_h = R_h - \delta m^{n}/dt^{n+1}
                self.subgridTmp[ci][:] = self.subgridError_last[ci]
                self.subgridTmp[ci] *= dtInv
                self.subgridTmp[ci] *= self.subgridErrorMassCoef_last[ci]#figure this out
                q[('pdeResidual',ci)] -= self.subgridTmp[ci]  #R_h --> \tilde{R}_h


                #calculate \tilde{R}_h = R_h - \delta m^{n}/dt^{n+1}
                self.subgridTmp_ip[ci][:] = self.subgridError_ip_last[ci]
                self.subgridTmp_ip[ci] *= dtInv
                self.subgridTmp_ip[ci] *= self.subgridErrorMassCoef_ip_last[ci]#figure this out
                self.cip[('pdeResidual',ci)] -= self.subgridTmp_ip[ci]  #R_h --> \tilde{R}_h

            #
            for cj in range(self.nc):
                if ('dpdeResidual',ci,cj) in q:
                    csubgridError.calculateSubgridError_tauRes(tau,
                                                               q[('pdeResidual',ci)],
                                                               q[('dpdeResidual',ci,cj)],
                                                               q[('subgridError',ci)],
                                                               q[('dsubgridError',ci,cj)])


            if self.trackSubScales:
                for cj in range(self.nc):
                    if ('dpdeResidual',ci,cj) in self.cip:
                        csubgridError.calculateSubgridError_tauRes(tau_ip,
                                                                   self.cip[('pdeResidual',ci)],
                                                                   self.cip[('dpdeResidual',ci,cj)],
                                                                   self.cip[('subgridError',ci)],
                                                                   self.cip[('dsubgridError',ci,cj)])

                logEvent("HaukeSangalli cip.subgridError.max=%s cip.subgridError.min=%s cq.subgridError.max=%s cq.subgridError.min=%s " % (self.cip[('subgridError',ci)].max(),
                                                                                                                                      self.cip[('subgridError',ci)].min(),
                                                                                                                                      q[('subgridError',ci)].max(),
                                                                                                                                      q[('subgridError',ci)].min()),1)
            #computing interpolant here will pick up \tilde{R}_h
            if self.includeSubgridScalesInGradientStabilization:
                #have to make sure interpolant has subgrid history update too
                self.calculateSubgridErrorInterpolants(ci)


            self.strongResidualInterpolant[ci].getGradientValues(q[('grad(v)',ci)],
                                                                 q[('grad(pdeResidual)',ci)])
            for cj in range(self.nc):
                if ('dpdeResidual',ci,cj) in q:
                    csubgridError.calculateSubgridErrorGradient_tauRes(tau_gradient,
                                                                       q[('grad(pdeResidual)',ci)],
                                                                       q[('grad(subgridError)',ci)])
                    #have got to come up with way to handle jacobian
                    q[('dgrad(subgridError)',ci,cj)].flat[:] = q[('dmt',ci,cj)].flat
                    q[('dgrad(subgridError)',ci,cj)] +=  q[('dr',ci,cj)]
                    self.subgridTmp[ci].flat[:] =  q[('dmt_sge',ci,cj)].flat
                    self.subgridTmp[ci] +=    q[('dr',ci,cj)]
                    q[('dgrad(subgridError)',ci,cj)] *= self.subgridTmp[ci]
                    q[('dgrad(subgridError)',ci,cj)] *= tau_gradient
                    q[('dgrad(subgridError)',ci,cj)] *= -1.0
                    #below works for just dmt approx for residual
                    #q[('dgrad(subgridError)',ci,cj)].flat[:] = tau_gradient.flat
                    #q[('dgrad(subgridError)',ci,cj)] *= -1.0
                    #q[('dgrad(subgridError)',ci,cj)] *= q[('dmt_sge',ci,cj)]
                    #q[('dgrad(subgridError)',ci,cj)] *= q[('dmt',ci,cj)]
            #
            logEvent("HaukeSangalli pdeResidual[ci].max = %s subgridError.max = %s subgridError.min= %s " % (q[('pdeResidual',ci)].max(),
                                                                                                        q[('subgridError',ci)].max(),
                                                                                                        q[('subgridError',ci)].min()),1)
            logEvent("HaukeSangalliTrackSubScales ADR tau_00.max() = %s tau_11.max() = %s grad(pdeResidual).max= %s grad(subgridError).max= %s dgrad(subgridError).max= %s " % (tau.max(),tau_gradient.max(),
                                                                                                                                                                           q[('grad(pdeResidual)',ci)].max(),
                                                                                                                                                                           q[('grad(subgridError)',ci)].max(),
                                                                                                                                                                           q[('dgrad(subgridError)',ci,ci)].max()),1)
            #mwf debug
            #import pdb
            #pdb.set_trace()
#             print "tau",tau
#             print "pdeResidual",q[('pdeResidual',ci)]
#             print "dpdeResidual",q[('dpdeResidual',ci,ci)]
#             print "subgrid error",q[('subgridError',ci)]
#             print "dsubgrid error",q[('dsubgridError',ci,ci)]
    def updateSubgridErrorHistory(self,initializationPhase=False):
        AdvectionDiffusionReactionHaukeSangalliInterpolant_ASGS.updateSubgridErrorHistory(self,initializationPhase)
        if self.trackSubScales:
            for ci in range(self.nc):
                if not initializationPhase:
                    #mwf I believe we are storing subgridError = tau*Res so need to reverse sign
                    self.subgridError_last[ci].flat[:] = self.cq[('subgridError',ci)].flat
                    self.subgridError_last[ci] *= -1.0
                    #
                    self.subgridError_ip_last[ci].flat[:] = self.cip[('subgridError',ci)].flat
                    self.subgridError_ip_last[ci] *= -1.0

                    #mwf debug
                    logEvent("HaukeSangalliTrackSubScales ADR tracksubscales updateSubgridErrorHistory max subgridError = %s at ip max= %s " % (self.subgridError_last[ci].max(),
                                                                                                                                           self.subgridError_ip_last[ci].max()),1)
                #how are we going to define subgrid mass?
                self.subgridErrorMassCoef_last[ci].flat[:] = self.cq[('dm',ci,ci)].flat
                self.subgridErrorMassCoef_ip_last[ci].flat[:] = self.cip[('dm',ci,ci)].flat
    def accumulateSubgridMassHistory(self,q):
        """
        incorporate subgrid scale mass accumulation
        \\delta m^{n}/\\delta t^{n+1}
        """
        if self.trackSubScales:
            for ci in range(self.nc):
                self.subgridTmp[ci][:] = self.subgridError_last[ci]
                #would be nice to have dt^{n+1} alone
                dt = self.timeIntegration.dt
                assert dt > 0.0
                dtInv = 1.0/dt
                self.subgridTmp[ci] *= dtInv
                self.subgridTmp[ci] *= self.subgridErrorMassCoef_last[ci]#figure this out
                #mwf debug
                logEvent("HaukeSangalliTrackSubScales accumulating delta u^n.abs.max= %s dm.max=%s  " % (max(numpy.absolute(self.subgridTmp[ci].flat)),max(numpy.absolute(self.subgridErrorMassCoef_last[ci].flat))),1)
                #mwf should be
                q[('mt',ci)] -= self.subgridTmp[ci]
                #don't think this matters right  now because called after calculateSubgridError
                self.subgridTmp_ip[ci][:] = self.subgridError_ip_last[ci]
                self.subgridTmp_ip[ci] *= dtInv
                self.subgridTmp_ip[ci] *= self.subgridErrorMassCoef_ip_last[ci]#figure this out
                self.cip[('mt',ci)] -= self.subgridTmp_ip[ci]

    #

class NavierStokesTransientSubScalesASGS_velocity_pressure(NavierStokesASGS_velocity_pressure):
    def __init__(self,coefficients,nd,stabFlag='1',lag=False,delayLagSteps=5,hFactor=1,noPressureStabilization=False,
                 trackSubScales=False,limit_tau_t=False,tau_t_limit_min=0.0,tau_t_limit_max=1.0):
        NavierStokesASGS_velocity_pressure.__init__(self,coefficients,nd,stabFlag=stabFlag,lag=lag,delayLagSteps=delayLagSteps,hFactor=hFactor,
                                                    noPressureStabilization=noPressureStabilization)
        self.trackSubScales = trackSubScales
        self.timeIntegration = None
        #apply bounds to tau_t?
        self.limit_tau_t = limit_tau_t
        self.tau_t_limit_min = tau_t_limit_min
        self.tau_t_limit_max = tau_t_limit_max
        self.trackSubScales_pressure = True
    def initializeElementQuadrature(self,mesh,t,cq):
        NavierStokesASGS_velocity_pressure.initializeElementQuadrature(self,mesh,t,cq)
        import copy
        self.subgridError_last = []
        self.subgridErrorMassCoef_last = []
        self.subgridTmp = []; self.subgridTmp2 = []
        for ci in range(self.nc):
            self.subgridTmp.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            if self.trackSubScales:
                self.subgridError_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.subgridErrorMassCoef_last.append(numpy.zeros(cq[('u',ci)].shape,'d'))
                self.subgridTmp2.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            else:
                self.subgridError_last.append(None)
                self.subgridErrorMassCoef_last.append(None)
            if self.lag:
                if ('dm',ci,ci) in cq:
                    cq[('dm_sge',ci,ci)] = copy.deepcopy(cq[('dm',ci,ci)])
                if ('dmt',ci,ci) in cq:
                    cq[('dmt_sge',ci,ci)] = copy.deepcopy(cq[('dmt',ci,ci)])
            else:
                if ('dm',ci,ci) in cq:
                    cq[('dm_sge',ci,ci)] = cq[('dm',ci,ci)]
                if ('dmt',ci,ci) in cq:
                    cq[('dmt_sge',ci,ci)] = cq[('dmt',ci,ci)]
    def initializeTimeIntegration(self,timeIntegration):
        """
        allow for connection with time integration method if tracking subscales
        """
        self.timeIntegration = timeIntegration
    def updateSubgridErrorHistory(self,initializationPhase=False):
        NavierStokesASGS_velocity_pressure.updateSubgridErrorHistory(self,initializationPhase=initializationPhase)
        if self.lag:
            for ci in range(1,self.nc):
                self.cq[('dm_sge',ci,ci)][:] = self.cq[('dm',ci,ci)]

        if self.trackSubScales:
            #momentum terms
            for ci in range(1,self.nc):
                if not initializationPhase:
                    #we are storing subgridError = tau*Res so need to reverse sign
                    self.subgridError_last[ci].flat[:] = self.cq[('subgridError',ci)].flat
                    self.subgridError_last[ci] *= -1.0
                    #mwf debug
                    logEvent("NS_ASGS tracksubscales updateSubgridErrorHistory subgridError[%s] max = %s min = %s " % (ci,self.subgridError_last[ci].max(),
                                                                                                                  self.subgridError_last[ci].min()),1)
                #how are we going to define subgrid mass?
                self.subgridErrorMassCoef_last[ci].flat[:] = self.cq[('dm',ci,ci)].flat
            #for pressure we have to store  strong residual
            if not initializationPhase:
                self.subgridError_last[0].flat[:] = self.cq[('pdeResidual',0)].flat
                logEvent("NS_ASGS tracksubscales updateSubgridErrorHistory subgridError[0] max = %s min = %s " % (self.subgridError_last[0].max(),
                                                                                                             self.subgridError_last[0].min() ),1)
    def accumulateSubgridMassHistory(self,q):
        """
        incorporate subgrid scale mass accumulation
        \\delta m^{n}/\\delta t^{n+1}
        """
        if self.trackSubScales:
            for ci in range(1,self.nc):
                self.subgridTmp[ci][:] = self.subgridError_last[ci]
                dt = self.timeIntegration.dt
                assert dt > 0.0
                dtInv = 1.0/dt
                self.subgridTmp[ci] *= dtInv
                self.subgridTmp[ci] *= self.subgridErrorMassCoef_last[ci]#decide how to approximate
                logEvent("NS_ASGS trackSubScales accumulating delta u^n ci=%s .abs.max= %s dm.max=%s  " % (ci,max(numpy.absolute(self.subgridTmp[ci].flat)),
                                                                                                 max(numpy.absolute(self.subgridErrorMassCoef_last[ci].flat))),1)

                q[('mt',ci)] -= self.subgridTmp[ci]


    def calculateSubgridError(self,q):
        from . import LinearAlgebraTools
        oldTau=True
        if self.nd == 2:
            if self.lag and self.nSteps < self.delayLagSteps:
                v = q[('f',0)]
            elif self.lag:
                v = self.v_last
            else:
                v = q[('f',0)]
            dmttmp = q[('dmt',1,1)]
            if self.trackSubScales:
                self.subgridTmp[1].fill(0.0)
                dmttmp = self.subgridTmp[1]
            if oldTau:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau_sd(self.hFactor,
                                                                                 self.mesh.elementDiametersArray,
                                                                                 dmttmp,
                                                                                 q[('dm',1,1)],
                                                                                 v,
                                                                                 q[('a',1,1)],
                                                                                 self.tau[0],
                                                                                 self.tau[1],
                                                                                 q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tau(self.hFactor,
                                                                              self.mesh.elementDiametersArray,
                                                                              dmttmp,
                                                                              q[('dm',1,1)],
                                                                              v,
                                                                              q[('a',1,1)],
                                                                              self.tau[0],
                                                                              self.tau[1],
                                                                              q[('cfl',0)])
            else:
                if self.coefficients.sd:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau_sd(q['inverse(J)'],
                                                                                     dmttmp,
                                                                                     q[('dm',1,1)],
                                                                                     v,
                                                                                     q[('a',1,1)],
                                                                                     self.tau[0],
                                                                                     self.tau[1],
                                                                                     q[('cfl',0)])
                else:
                    csubgridError.calculateSubgridErrorNavierStokes2D_generic_tau(q['inverse(J)'],
                                                                                  dmttmp,
                                                                                  q[('dm',1,1)],
                                                                                  v,
                                                                                  q[('a',1,1)],
                                                                                  self.tau[0],
                                                                                  self.tau[1],
                                                                                  q[('cfl',0)])
            tau0=self.tau[0]
            tau1=self.tau[1]
            #mwf seeing some difference in tau0 and self.tau_last
            #need to synchronize the lagging
            #tau0 = self.tau_last[0]
            #tau1 = self.tau_last[1]
            #TODO: make sure dm_sge is set correctly for lagging
            dm_subgrid = q[('dm_sge',1,1)]#density same for both velocity components
            if self.trackSubScales:
                dt = self.timeIntegration.dt
                assert dt > 0.0
                dtInv = 1.0/dt
                #pressure,
                #   \delta p = -tau_1*(1+tau_0/dt)*R^{n+1}_p + tau_1*tau_0/dt*R^n_p
                #recall that code is expecting subgridError to be tau*R instead of -tau*R
                logEvent("NS_ASGS trackScales before transient modficication (tau_s) tau[0].max= %s tau[0].min=%s   " % (tau0.max(),tau0.min()),1)
                logEvent("NS_ASGS trackScales before transient modficication (tau_s) tau[1].max= %s tau[1].min=%s  " % (tau1.max(),tau1.min()),1)
                if self.lag:
                    logEvent("NS_ASGS trackScales before transient modficication (tau_s) nSteps=%d delayLagSteps=%d tau_last[0].max= %s  tau_last[0].min= %s " % (self.nSteps,
                                                                                                                                                             self.delayLagSteps,
                                                                                                                                                             self.tau_last[0].max(),self.tau_last[0].min()),1)
                #tau for current pressure subgridError
                self.subgridTmp[0][:] = tau0
                self.subgridTmp[0] *= dtInv
                self.subgridTmp[0] += 1.0
                self.subgridTmp[0] *= tau1
                #mwf codina has an extra 1/4 in tau?
                self.subgridTmp[0] *= 0.25
                #tau for history term when updating pressure subgrid error
                self.subgridTmp2[0][:] = tau0
                self.subgridTmp2[0] *= dtInv
                self.subgridTmp2[0] *= tau1
                #mwf codina has an extra 1/4 in tau?
                self.subgridTmp2[0] *= 0.25
                #now modify tau0 --> tau_t0
                if tau0.max() > 0.0:
                    self.subgridTmp[1][:] = tau0
                    self.subgridTmp[1] *= dt
                    self.subgridTmp2[1][:] = tau0
                    self.subgridTmp2[1] *= dm_subgrid
                    self.subgridTmp2[1] += dt
                    self.subgridTmp[1] /= self.subgridTmp2[1]
                    #bound tau_t based on dt size
                    if self.limit_tau_t:
                        numpy.clip(self.subgridTmp[1],self.tau_t_limit_min*dt,self.tau_t_limit_max*dt,self.subgridTmp[1])
                #
                #set tau0 --> to point to subgridTmp[1] since this multiplies momentum residual
                #set tau1 --> to point to subgridTmp[0] since this multiplies continuity residual
                tau0 = self.subgridTmp[1]
                tau1 = self.subgridTmp[0]

                #mwf debug
                logEvent("NS_ASGS trackScales after modifying tau[0].max= %s tau[0].min= %s " % (tau0.max(),tau0.min()),1)
                logEvent("NS_ASGS trackScales after modifying tau[1].max= %s tau[1].min= %s " % (tau1.max(),tau1.min()),1)
                #mwf should be 1.0/m'
                assert tau0.max() * dm_subgrid.max() /dt <= 1.0, "Subgrid scales, modified tau_t.max() = %s dt = %s dm_subgrid.max() = %s tau.m'/dt = %s must be less than 1 " % (tau.max(),
                                                                                                                                                                                  dt,
                                                                                                                                                                                  dm_subgrid.max(),
                                                                                                                                                                                  tau.max()/dt)
                #
                #account for old subgrid error in momentum strong residual
                for ci in range(1,self.nc):
                    #tau here should be the same as tau_t in Codina's formalism if dmdt is included?
                    #calculate \tilde{R}_h = R_h - \delta m^{n}/dt^{n+1}
                    self.subgridTmp2[ci][:] = self.subgridError_last[ci]


                    self.subgridTmp2[ci] *= dtInv
                    self.subgridTmp2[ci] *= self.subgridErrorMassCoef_last[ci]#decide what time level to use
                    q[('pdeResidual',ci)] -= self.subgridTmp2[ci]  #R_h --> \tilde{R}_h
                #momentum components

            #end track subgrid scales
            csubgridError.calculateSubgridErrorNavierStokes2D_GLS_tauRes(tau0,
                                                                         tau1,
                                                                         q[('pdeResidual',0)],
                                                                         q[('dpdeResidual',0,1)],
                                                                         q[('dpdeResidual',0,2)],
                                                                         q[('pdeResidual',1)],
                                                                         q[('dpdeResidual',1,0)],
                                                                         q[('dpdeResidual',1,1)],
                                                                         q[('dpdeResidual',1,2)],
                                                                         q[('pdeResidual',2)],
                                                                         q[('dpdeResidual',2,0)],
                                                                         q[('dpdeResidual',2,1)],
                                                                         q[('dpdeResidual',2,2)],
                                                                         q[('subgridError',0)],
                                                                         q[('dsubgridError',0,1)],
                                                                         q[('dsubgridError',0,2)],
                                                                         q[('subgridError',1)],
                                                                         q[('dsubgridError',1,0)],
                                                                         q[('dsubgridError',1,1)],
                                                                         q[('dsubgridError',1,2)],
                                                                         q[('subgridError',2)],
                                                                         q[('dsubgridError',2,0)],
                                                                         q[('dsubgridError',2,1)],
                                                                         q[('dsubgridError',2,2)])
            if self.trackSubScales and self.trackSubScales_pressure:
                #modify subgrid pressure error, tau1*tau0/dt sits in subgridTmp2[0]
                self.subgridTmp2[0] *= self.subgridError_last[0]
                q[('subgridError',0)]  -= self.subgridTmp2[0]
            if self.noPressureStabilization:
                q[('subgridError',0)][:]=0.0
                q[('dsubgridError',0,1)][:]=0.0
                q[('dsubgridError',0,2)][:]=0.0
            #
            for ci in range(self.nc):
                if ('mt',ci) in q:
                    logEvent("NS_ASGS trackSubScales calculateSubgridError mt[%s] max= %s min=%s  " % (ci,q[('mt',ci)].max(),q[('mt',ci)].min()),1)
                logEvent("NS_ASGS trackSubScales calculateSubgridError pdeResidual[%s] max= %s min=%s  " % (ci,q[('pdeResidual',ci)].max(),q[('pdeResidual',ci)].min()),1)

                logEvent("NS_ASGS trackSubScales calculateSubgridError subgridError[%s] max= %s min=%s  " % (ci,q[('subgridError',ci)].max(),q[('subgridError',ci)].min()),1)
                if self.trackSubScales:
                    logEvent("NS_ASGS trackSubScales calculateSubgridError subgridError_last[%s] max= %s min=%s  " % (ci,self.subgridError_last[ci].max(),self.subgridError_last[ci].min()),1)
                for cj in range(self.nc):
                    if ('df_sge',ci,cj) in q:
                        logEvent("NS_ASGS trackSubScales calculateSubgridError df_sge %s %s max= %s min=%s  " % (ci,cj,q[('df_sge',ci,cj)].max(),q[('df_sge',ci,cj)].min()),1)


        elif self.nd == 3:
            assert False
        for ci in range(self.nd):
            q[('cfl',ci+1)][:] = q[('cfl',0)]