"""
A class hierarchy for shock capturing diffusion methods

.. inheritance-diagram:: proteus.ShockCapturing
   :parts: 1
"""
import numpy
from . import cshockCapturing
class DelayedLagState(object):
    """state()/set_state() for a shock capturing that starts unlagged
    (``numDiff_last`` aliased to ``numDiff``) and switches to lagging after
    ``nStepsToDelay`` history updates (mprans NCLS, VOF and RANS3PF).

    The state is the machine, not just the array: ``lag``, ``nSteps``, and
    whether ``numDiff_last`` is still the alias. ``set_state`` rebuilds the
    separate copy that the switch makes, which writing values into an
    aliased array cannot do.
    """
    def _components(self):
        # RANS3PF keeps a dict keyed 0..2 whatever the space dimension; NCLS and VOF a list of nc arrays
        return sorted(self.numDiff_last) if isinstance(self.numDiff_last, dict) else range(len(self.numDiff_last))

    def state(self):
        last = [self.numDiff_last[ci] for ci in self._components()]
        alias = all(l is self.numDiff[ci] for ci, l in zip(self._components(), last))
        st = {'lag': bool(self.lag),
              'nSteps': int(self.nSteps),
              'alias': bool(alias)}
        if not alias:
            for ci, a in zip(self._components(), last):
                st['numDiff_last_%d' % ci] = numpy.array(a, copy=True)
        return st

    def set_state(self, state):
        self.lag = bool(state['lag'])
        self.nSteps = int(state['nSteps'])
        comps = list(self._components())
        if state['alias']:
            last = [self.numDiff[ci] for ci in comps]
        else:
            last = [numpy.array(state['numDiff_last_%d' % ci], copy=True) for ci in comps]
        self.numDiff_last = dict(zip(comps, last)) if isinstance(self.numDiff_last, dict) else last

class ShockCapturing_base(object):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True):
        self.nc = coefficients.nc
        self.nd = nd
        self.components=list(range(self.nc))
        self.shockCapturingFactor=shockCapturingFactor
        self.lag=lag
        #what was asked for: some shock capturing switches lag at run time (it delays it)
        self.lagRequested=lag
        #mwf had to put in da_sge, df_sge
        self.coefficients = coefficients
    def state(self):
        """Nothing to carry beyond the solution (the state contract, see TimeIntegration.TI_base.state)."""
        return None

    def set_state(self, state):
        """Restore what ``state()`` returned; see TimeIntegration.TI_base.state."""
        pass

    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.numDiff=[]
        self.numDiff_last=[]
        for ci in range(self.nc):
            if self.lag:
                self.numDiff_last.append(cq[('numDiff',ci,ci)])
                self.numDiff.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            else:
                self.numDiff_last.append(cq[('numDiff',ci,ci)])
                self.numDiff.append(cq[('numDiff',ci,ci)])
        #mwf what if just want shock capturing and no stabilization?
        #need to put in lag steps for da_sge and df_sge below as well if not
        #done by Subgrid error
        for ci in range(self.nc):
            for cj in range(self.nc):
                if ('df',ci,cj) in cq and ('df_sge',ci,cj) not in cq:
                    cq[('df_sge',ci,cj)]=cq[('df',ci,cj)]
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                if ('grad(phi)_sge',ck) not in cq:
                    cq[('grad(phi)_sge',ck)]=cq[('grad(phi)',ck)]
                for cj in list(cjDict.keys()):
                    if ('dphi_sge',ck,cj) not in cq:
                        cq[('dphi_sge',ck,cj)]=cq[('dphi',ck,cj)]
                    if ('da_sge',ci,ck,cj) not in cq:
                        cq[('da_sge',ci,ck,cj)]=cq[('da',ci,ck,cj)]

    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            self.numDiff[ci].flat[:] = self.shockCapturingFactor
    def updateShockCapturingHistory(self):
        if self.lag:
            for ci in range(self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]

ConstantDiffusion_SC = ShockCapturing_base

class ResGrad_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True):
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            cshockCapturing.calculateNumericalDiffusionResGrad(self.shockCapturingFactor,
                                                               self.mesh.elementDiametersArray,
                                                               q[('pdeResidual',ci)],
                                                               q[('grad(u)',ci)],
                                                               self.numDiff[ci])
class ResGradFFDarcy_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True):
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
    def calculateNumericalDiffusion(self,q):
        cshockCapturing.calculateNumericalDiffusionResGrad(self.shockCapturingFactor,
                                                           self.mesh.elementDiametersArray,
                                                           q[('pdeResidual',0)],
                                                           q[('grad(u)',0)],
                                                           self.numDiff[0])

class ResGradQuad_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True,gradLag=True):
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.debug=False
        self.gradLag = gradLag
    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            if self.debug:
                if numpy.isnan(q[('pdeResidual',ci)]).any(): #
                    import pdb
                    print("NaN's in res")
                    pdb.set_trace()
            cshockCapturing.calculateNumericalDiffusionResGradQuad(self.shockCapturingFactor,
                                                                   self.mesh.elementDiametersArray,
                                                                   q[('pdeResidual',ci)],
                                                                   q[('grad(u)',ci)],
                                                                   self.numDiff[ci])
            if self.debug:
                if numpy.isnan(self.numDiff[ci]).any():
                    import pdb
                    print("NaN's in numDiff")
                    pdb.set_trace()

class Eikonal_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True):
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.debug=False
    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            cshockCapturing.calculateNumericalDiffusionEikonal(self.shockCapturingFactor,
                                                               self.mesh.elementDiametersArray,
                                                               q[('pdeResidual',ci)],
                                                               self.numDiff[ci])

class ScalarAdvection_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.5,lag=True):
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            cshockCapturing.calculateNumericalDiffusion_A_1(self.shockCapturingFactor,
                                                            self.mesh.elementDiametersArray,
                                                            q[('pdeResidual',ci)],
                                                            q[('mt',ci)],
                                                            q[('df',ci,ci)],
                                                            self.numDiff[ci])

class HamiltonJacobi_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFlag='1',shockCapturingFactor=0.5,lag=True):
        self.shockCapturingFlag=shockCapturingFlag=shockCapturingFlag
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            cshockCapturing.calculateNumericalDiffusionHJ(self.shockCapturingFlag,
                                                          self.shockCapturingFactor,
                                                          self.mesh.elementDiametersArray,
                                                          q[('pdeResidual',ci)],
                                                          q[('mt',ci)],
                                                          q[('H',ci)],
                                                          self.numDiff[ci])
class HamiltonJacobiJaffre_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFlag='1',shockCapturingFactor=0.5,lag=True,betaPower=0.1):
        self.shockCapturingFlag=shockCapturingFlag=shockCapturingFlag
        self.beta=betaPower
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            cshockCapturing.calculateNumericalDiffusionJaffre(self.shockCapturingFactor,
                                                              self.beta,
                                                              self.mesh.elementDiametersArray,
                                                              q[('pdeResidual',ci)],
                                                              q[('dH',ci,ci)],
                                                              self.numDiff[ci])
class JaffreGradU_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFlag='1',shockCapturingFactor=0.5,lag=True,betaPower=0.1):
        self.shockCapturingFlag=shockCapturingFlag=shockCapturingFlag
        self.beta=betaPower
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            cshockCapturing.calculateNumericalDiffusionJaffre(self.shockCapturingFactor,
                                                              self.beta,
                                                              self.mesh.elementDiametersArray,
                                                              q[('pdeResidual',ci)],
                                                              q[('grad(u)',ci)],
                                                              self.numDiff[ci])

class ResGradJuanes_SC(ShockCapturing_base):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,uSC=1.0,lag=True):
        ShockCapturing_base.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.uSC = uSC
    def calculateNumericalDiffusion(self,q):
        for ci in range(self.nc):
            cshockCapturing.calculateNumericalDiffusionResGradJuanes(self.shockCapturingFactor,
                                                                     self.uSC,
                                                                     self.mesh.elementDiametersArray,
                                                                     q[('pdeResidual',ci)],
                                                                     q[('grad(u)',ci)],
                                                                     self.numDiff[ci])

#examples of shock capturing with lagging allowed after certain number of steps
class ResGradDelayLag_SC(ResGrad_SC):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True,nStepsToDelay=None):
        ResGrad_SC.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps=0
    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.numDiff=[]
        self.numDiff_last=[]
        self.cq_numDiff=[]
        for ci in range(self.nc):
            if self.lag:
                self.numDiff_last.append(cq[('numDiff',ci,ci)])
                self.numDiff.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            elif self.lag == False and self.nStepsToDelay is not None:
                self.cq_numDiff.append(cq[('numDiff',ci,ci)])
                self.numDiff.append(cq[('numDiff',ci,ci)])
            else:
                self.numDiff.append(cq[('numDiff',ci,ci)])
        #mwf what if just want shock capturing and no stabilization?
        #need to put in lag steps for da_sge and df_sge below as well if not
        #done by Subgrid error
        for ci in range(self.nc):
            for cj in range(self.nc):
                if ('df',ci,cj) in cq and ('df_sge',ci,cj) not in cq:
                    cq[('df_sge',ci,cj)]=cq[('df',ci,cj)]
        for ci,ckDict in self.coefficients.diffusion.items():
            for ck,cjDict in ckDict.items():
                if ('grad(phi)_sge',ck) not in cq:
                    cq[('grad(phi)_sge',ck)]=cq[('grad(phi)',ck)]
                for cj in list(cjDict.keys()):
                    if ('dphi_sge',ck,cj) not in cq:
                        cq[('dphi_sge',ck,cj)]=cq[('dphi',ck,cj)]
                    if ('da_sge',ci,ck,cj) not in cq:
                        cq[('da_sge',ci,ck,cj)]=cq[('da',ci,ck,cj)]
    #
    def updateShockCapturingHistory(self):
        self.nSteps += 1
        if self.lag:
            for ci in range(self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            self.lag = True
            self.numDiff = []
            self.numDiff_last=[]
            for ci in range(self.nc):
                self.numDiff_last.append(self.cq_numDiff[ci])
                self.numDiff.append(numpy.zeros(self.cq_numDiff[ci].shape,'d'))

class ResGradQuadDelayLag_SC(ResGradQuad_SC):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True,nStepsToDelay=None,isotropic=False):
        ResGradQuad_SC.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps=0
        self.isotropic=isotropic
    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.numDiff=[]
        self.numDiff_last=[]
        for ci in range(self.nc):
            if self.lag:
                self.numDiff_last.append(cq[('numDiff',ci,ci)])
                self.numDiff.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            elif self.lag == False and self.nStepsToDelay is not None:
                self.numDiff.append(cq[('numDiff',ci,ci)])
                self.numDiff_last.append(cq[('numDiff',ci,ci)])
            else:
                self.numDiff.append(cq[('numDiff',ci,ci)])
                self.numDiff_last.append(cq[('numDiff',ci,ci)])
    def updateShockCapturingHistory(self):
        self.nSteps += 1

        if self.isotropic:
            for ci in range(1,self.nc):
                self.numDiff[0][:] = numpy.maximum(self.numDiff[0],self.numDiff[ci])
            for ci  in range(1,self.nc):
                self.numDiff[ci][:] = self.numDiff[0]
        if self.lag:
            for ci in range(self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            self.lag = True
            self.numDiff=[]
            for ci in range(self.nc):
                self.numDiff.append(self.numDiff_last[ci].copy())

class NavierStokes_SC(ResGradQuad_SC):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True,nStepsToDelay=None):
        ResGradQuad_SC.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps=0
    def calculateNumericalDiffusion(self,q):
        for ci in range(1,self.nc):
            if numpy.isnan(q[('pdeResidual',ci)]).any():
                import pdb
                print("NaN's in res")
                pdb.set_trace()
            cshockCapturing.calculateNumericalDiffusionResGradQuad(self.shockCapturingFactor,
                                                                   self.mesh.elementDiametersArray,
                                                                   q[('pdeResidual',ci)],
                                                                   q[('grad(u)',ci)],
                                                                   self.numDiff[ci])
            if numpy.isnan(self.numDiff[ci]).any():
                import pdb
                print("NaN's in numDiff")
                pdb.set_trace()
    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.numDiff={}
        self.numDiff_last={}
        self.cq_numDiff={}
        for ci in range(1,self.nc):
            if self.lag:
                self.numDiff_last[ci]=cq[('numDiff',ci,ci)]
                self.numDiff[ci]=numpy.zeros(cq[('u',ci)].shape,'d')
            elif self.lag == False and self.nStepsToDelay is not None:
                self.cq_numDiff[ci]=cq[('numDiff',ci,ci)]
                self.numDiff[ci]=cq[('numDiff',ci,ci)]
            else:
                self.numDiff[ci]=cq[('numDiff',ci,ci)]
    def updateShockCapturingHistory(self):
        self.nSteps += 1
        if self.lag:
            for ci in range(1,self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            self.lag = True
            self.numDiff = []
            self.numDiff_last=[]
            for ci in range(1,self.nc):
                self.numDiff_last[ci]=self.cq_numDiff[ci]
                self.numDiff[ci]=numpy.zeros(self.cq_numDiff[ci].shape,'d')

class NavierStokes_SC_opt(ResGradQuad_SC):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True,nStepsToDelay=None):
        ResGradQuad_SC.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps=0
    def calculateNumericalDiffusion(self,q):
        for ci in range(1,self.nc):
            if numpy.isnan(q[('pdeResidual',ci)]).any():
                import pdb
                print("NaN's in res")
                pdb.set_trace()
            cshockCapturing.calculateNumericalDiffusionResGradQuad(self.shockCapturingFactor,
                                                                   self.mesh.elementDiametersArray,
                                                                   q[('pdeResidual',ci)],
                                                                   q[('grad(u)',ci)],
                                                                   self.numDiff[ci])
            if numpy.isnan(self.numDiff[ci]).any():
                import pdb
                print("NaN's in numDiff")
                pdb.set_trace()
    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.numDiff={}
        self.numDiff_last={}
        for ci in range(1,self.nc):
            if self.lag:
                #note: _opt  does  the  lagging backwards from above
                self.numDiff_last[ci]=numpy.zeros(cq[('u',ci)].shape,'d')
                self.numDiff[ci]=cq[('numDiff',ci,ci)]
            else:
                self.numDiff_last[ci]=cq[('numDiff',ci,ci)]
                self.numDiff[ci]=cq[('numDiff',ci,ci)]
    def updateShockCapturingHistory(self):
        for ci in range(1,self.nc):
            if self.lag:
                self.numDiff_last[ci][:]=self.numDiff[ci]

class ResGradFFDarcyDelayLag_SC(ResGradFFDarcy_SC):
    def __init__(self,coefficients,nd,shockCapturingFactor=0.25,lag=True,nStepsToDelay=None):
        ResGradFFDarcy_SC.__init__(self,coefficients,nd,shockCapturingFactor,lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps=0
    def initializeElementQuadrature(self,mesh,t,cq):
        self.mesh=mesh
        self.numDiff=[]
        self.numDiff_last=[]
        self.cq_numDiff=[]
        for ci in range(self.nc):
            if self.lag:
                self.numDiff_last.append(cq[('numDiff',ci,ci)])
                self.numDiff.append(numpy.zeros(cq[('u',ci)].shape,'d'))
            elif self.lag == False and self.nStepsToDelay is not None:
                self.cq_numDiff.append(cq[('numDiff',ci,ci)])
                self.numDiff.append(cq[('numDiff',ci,ci)])
            else:
                self.numDiff.append(cq[('numDiff',ci,ci)])
    def updateShockCapturingHistory(self):
        self.nSteps += 1
        if self.lag:
            for ci in range(self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            self.lag = True
            self.numDiff = []
            self.numDiff_last=[]
            for ci in range(self.nc):
                self.numDiff_last.append(self.cq_numDiff[ci])
                self.numDiff.append(numpy.zeros(self.cq_numDiff[ci].shape,'d'))
