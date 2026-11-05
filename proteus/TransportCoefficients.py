"""
Classes for implementing the coefficients of transport equations.

TC_base defines the interface. The classes derived from TC_base in
this module define common PDE's.

.. inheritance-diagram:: proteus.TransportCoefficients
   :parts: 1
"""
from math import *
from warnings import warn
import numpy
from . import Norms
from .Profiling import logEvent
from warnings import warn

## \file TransportCoefficients.py
#
#@{

##\brief Base class for transport coefficients classes
#
#The child classes of TC_base define the coefficients of a system of PDE's. At present
#we constrain each tightly coupled system of model equations to have
#the form of a general system of transport equations:
#
#\f[
#\frac{\partial m^i}{\partial t} + \nabla \cdot \left[ \mathbf{f}^i - \sum_j \mathbf{a}^{ij} \nabla \phi^j \right] + r^i + h^i(\nabla u) = 0
#\f]
#
#where \f$i=0,\ldots n_c-1\f$ and \f$u\f$ is the vector of components,
#\f$ (u_0,\ldots,u_{n_c-1})^t \f$.  For the sake of generality we
#label \f$m^i\f the masses, \f$f^i\f$ the advection vectors,
#\f$a^{ij}\f$ the diffusion tensors, \f$r^i\f$ the reactions, and
#\f$h^i\f$ the Hamiltonians. Any of these terms may be ommitted from
#the equations and the diffusion tensor may be provided as a sparse
#matrix.
#
class TC_base(object):
    """
    This is the base class for coefficients of the vector transport
    equation with nc components. The coefficients and their derivative
    are evaluated at a time t over an array of solution values, u, at
    an array of points x.

    methods:

    evaluate(t,c) -- t, time, is a float, c is a dictionary of x,u, and coefficients

    members:

    nc        -- (number  of components) integer
    mass      -- ({{}} logically nc x nc) 'linear' or 'nonlinear'
    advection -- ({{}} logically nc x nc) 'constant', 'linear' or 'nonlinear'
    diffusion -- ({{{}}} logically nc x nc x nc) 'constant' or 'nonlinear'
    potential -- ({{}} logically nc x nc) 'u' or 'nonlinear'
    reaction  -- ({{}} logically nc x nc) 'constant','linear', or 'nonlinear'
    hamiltonian -- ({{}} logically nx x nc) 'linear' or 'nonlinear'
    stencil   -- ([set(),] nc x nc) indicates jacobian block structure
    variableNames -- (number of components) string describing the component

    These dictionaries indicate which coefficients are nonzero and how
    the non-zero coefficietns depend on the solution variables. The
    dictionaries will be used like sparse multi-dimensional
    arrays. These flags are used to deduce the terms in the PDE and
    allow optimizations in special cases. Non-existant values imply
    that the equation for that component has no term corresponding to
    the given coefficient.

    .. math::

    a_{i,j} = 1
    """
    def __init__(self,
                 nc=2,
                 mass      = {},
# {0: {0: 'nonlinear'},
#  1: {1: 'nonlinear'}},
                 advection = {},
#{0: {0: 'nonlinear', 1: 'nonlinear'},
# 1: {0: 'nonlinear', 1: 'nonlinear'}},
                 diffusion = {},
#{0: {0: {0:'constant'}},
# 1: {1: {1:'constant'}}},
                 potential = {},
#{0: {0: 'u'},
# 1: {1: 'u'}},
                 reaction  = {},
#{0: {0: 'nonlinear',1: 'nonlinear'},
# 1: {0: 'nonlinear',1: 'nonlinear'}},
                 hamiltonian  = {},
#{0: {0: 'linear'},
# 1: {1: 'linear'}},
                 variableNames=None,
                 stress={},
#                  stress= {0:{0:'linear',1:'linear',2:'linear'},
#                           1:{0:'linear',1:'linear',2:'linear'},
#                           2:{0:'linear',1:'linear',2:'linear'}}
                 sparseDiffusionTensors = {},
                 useSparseDiffusion = True,
                 movingDomain=False):
        """
        Set the number of components (equations) of the
        PDE and initialize the dicitionaries describing the form of the
        coefficients. Strings naming each component (used for viewing
        and archiving) and a structure defining the sparsity pattern
        of diffusion tensors may also be provided.
        """
        self.nc = nc
        if variableNames is None:
            self.variableNames = ['u'+repr(i) for i in range(nc)]
        else:
            self.variableNames=variableNames
        self.mass=mass
        self.advection=advection
        self.diffusion=diffusion
        self.potential=potential
        self.reaction=reaction
        self.hamiltonian=hamiltonian
        self.stress=stress
        self.sd = useSparseDiffusion
        self.movingDomain=movingDomain
        self.sdInfo = sparseDiffusionTensors#{(0,0):(rowptr,colind),...}}
        self.elementIntegralKeys=[]
        self.elementBoundaryIntegralKeys=[]
        self.stencil=[set() for ci in range(self.nc)]
        self.integrals={}
        for ci,cjDict in self.mass.items():
            self.elementIntegralKeys.append(('m',ci))
            for cj in cjDict:
                self.stencil[ci].add(cj)
                #if moving domain and mass term add advection term
                if self.movingDomain:
                    if ci in self.advection:
                        if cj in self.advection[ci]:
                            if self.mass[ci][cj] == 'nonlinear':
                                self.advection[ci][cj] == 'nonlinear'
                            elif self.mass[ci][cj] == 'linear':
                                if self.advection[ci][cj] == 'constant':
                                    self.advection[ci][cj] == 'linear'
                        else:
                            self.advection[ci][cj] == self.mass[ci][cj]
                    else:
                        self.advection[ci] = {cj:self.mass[ci][cj]}
        for ci,cjDict in self.advection.items():
            self.elementIntegralKeys.append(('f',ci))
            self.elementBoundaryIntegralKeys.append(('f',ci))
            for cj in cjDict:
                self.stencil[ci].add(cj)
        for ci,ckDict in self.diffusion.items():
            for ck,cjDict in ckDict.items():
                self.elementIntegralKeys.append(('a',ci,ck))
                self.elementBoundaryIntegralKeys.append(('a',ci,ck))
                if ck not in self.potential:
                    logEvent("""a[ci=%d][ck=%d] is non-zero but phi[ck=%d] is undefined. Setting
                    phi[ck=%d]=u[ck=%d], the potential definition
                    should be corrected in the future\n""" % (ci,ck,ck,ck,ck))
                    self.potential[ck]='u'
                for cj in list(cjDict.keys()):
                    self.stencil[ci].add(cj)
                    self.stencil[ci].add(ck)
        for ci,cjDict in self.potential.items():
            for cj in list(cjDict.keys()):
                self.stencil[ci].add(cj)
        for ci,cjDict in self.reaction.items():
            self.elementIntegralKeys.append(('r',ci))
            for cj in cjDict:
                self.stencil[ci].add(cj)
        for ci,cjDict in self.hamiltonian.items():
            self.elementIntegralKeys.append(('H',ci))
            self.elementBoundaryIntegralKeys.append(('H',ci))
            for cj in cjDict:
                self.stencil[ci].add(cj)
        for ci,cjDict in self.stress.items():
            self.elementIntegralKeys.append(('sigma',ci))
            self.elementBoundaryIntegralKeys.append(('sigma',ci))
            for cj in cjDict:
                self.stencil[ci].add(cj)
        for ci in range(self.nc):
            if len(self.stencil[ci]) == 0:
                raise RuntimeError("""Equation %d is independent of the solution,
                which means the system defined by the coefficients is singular.""")
        self.vectorComponents=None
        self.vectorName="velocity"
        try:
            self.nullSpace
        except AttributeError:
            self.nullSpace = 'NoNullSpace'
        #mwf append u integral key by default
        for ci in range(self.nc):
            self.elementIntegralKeys.append(('u',ci))
            self.elementBoundaryIntegralKeys.append(('u',ci))

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__setattr__(key, val)

    def evaluate(self,t,c):
        """
        Evaluate the coefficients at a given time, t, using the coefficient storage passed in as the dictionary c.
        """
        pass
    def initializeMesh(self,mesh):
        """
        Give the TC object access to the mesh for any mesh-dependent information.
        """
        pass
    def initializeElementQuadrature(self,t,cq):
        """
        Give the TC object access to the element quadrature storage
        """
        pass
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        pass
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        """
        Give the TC object access to the exterior element boundary quadrature storage
        """
        pass
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        """
        Give the TC object access to the generalized interpolation point storage. These points are used  to project nonlinear potentials (phi).
        """
        pass
    def updateToMovingDomain(self,t,c):
        from . import cfemIntegrals
        for ci in range(self.nc):
            if ('m',ci) in c:
                cfemIntegrals.update_f_movingDomain(c['xt'],c[('m',ci)],c[('f',ci)])
                for cj in range(self.nc):
                    if ('dm',ci,cj) in c:
                        cfemIntegrals.update_df_movingDomain(c['xt'],c[('dm',ci,cj)],c[('df',ci,cj)])
            #else: transport coefficients derived class should add this if a constant mass is assumed
            #    cfemIntegrals.update_f_movingDomain_constantMass(c['xt'],c[('f',ci)])
    def attachModels(self,modelList):
        """
        Give the TC object access to other models in a loosely coupled split operator formulation (e.g. a transport equation for concentration might get velocity from a flow equation)
        """
        pass
    def preStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        return {}
    def postStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself after the time step.
        """
        return {}
    def allocateDummyCoefficients(self,c=None,nPoints=101,uMax=1.0,uMin=0.0,nSpace=1):
        """
        Allocate some coefficient dictionaries to use for viewing the coefficients
        """
        import copy
        if c is None:
            ctemp = {}
            for ci in range(self.nc):
                ctemp[('u',ci)] = numpy.zeros(nPoints,'d')
                delta_u = (uMax-uMin)/(nPoints-1)
                for i in range(nPoints):
                    ctemp[('u',ci)] = i*delta_u + uMin
            for ci,cjDict in self.mass.items():
                for cj in cjDict:
                    ctemp[('m',ci)] = numpy.zeros(nPoints,'d')
                    ctemp[('dm',ci,cj)] = numpy.zeros(nPoints,'d')
            for ci,cjDict in self.advection.items():
                for cj in list(cjDict.keys()):
                    ctemp[('f',ci)] = numpy.zeros((nPoints,nSpace),'d')
                    ctemp[('df',ci,cj)] = numpy.zeros((nPoints,nSpace),'d')
            for ci,ckDict in self.diffusion.items():
                for ck,cjDict in ckDict.items():
                    for cj in list(cjDict.keys()):
                        ctemp[('a',ci,ck)] = numpy.zeros((nPoints,nSpace,nSpace),'d')
                        ctemp[('da',ci,ck,cj)] = numpy.zeros((nPoints,nSpace,nSpace),'d')
            for ci,cjDict in self.potential.items():
                for cj,flag in cjDict.items():
                    ctemp[('phi',ci)] = numpy.zeros(nPoints,'d')
                    ctemp[('dphi',ci,cj)] = numpy.zeros(nPoints,'d')
            for ci,cjDict in self.reaction.items():
                for cj in list(cjDict.keys()):
                    ctemp[('r',ci)] = numpy.zeros(nPoints,'d')
                    ctemp[('dr',ci,cj)] = numpy.zeros(nPoints,'d')
            for ci,cjDict in self.hamiltonian.items():
                for cj in list(cjDict.keys()):
                    ctemp[('H',ci)] = numpy.zeros(nPoints,'d')
                    ctemp[('dH',ci,cj)] = numpy.zeros((nPoints,nSpace),'d')
        else:
            nPoints = len(c[('u',0)].flat)
            ctemp = copy.deepcopy(c)
            for ci in range(self.nc):
                uMin = min(c[('u',ci)].flat)
                uMax = max(c[('u',ci)].flat)
                delta_u = (uMax-uMin)/(nPoints-1)
                for i in range(nPoints):
                    ctemp[('u',ci)].flat[i]=i*delta_u+uMin
        return ctemp

    def plotCoefficientFunctions(self,t,ctemp):
        """
        Plot the coefficients
        """
        from . import Viewers
        nPoints = len(ctemp[('u',0)].flat)
        self.evaluate(t,ctemp)
        if Viewers.viewerType == 'gnuplot':
            def linePlot(x,y,title,start=0,stride=1):
                for i in range(nPoints):
                    Viewers.datFile.write("%12.5e %12.5e \n" % (x.flat[i],y.flat[i*stride+start]))
                Viewers.datFile.write("\n \n")
                cmd = "set term x11 %i; plot \'%s\' index %i with linespoints title \"%s\" \n" % (Viewers.windowNumber,
                                                                                            Viewers.datFilename,
                                                                                            Viewers.plotNumber,
                                                                                            title)
                Viewers.cmdFile.write(cmd)
                Viewers.viewerPipe.write(cmd)
                Viewers.newPlot()
                Viewers.newWindow()
            for ci,cjDict in self.mass.items():
                for cj in cjDict:
                    linePlot(ctemp[('u',cj)],ctemp[('m',ci)],'m_%i vs. u_%i' % (ci,cj) )
                    linePlot(ctemp[('u',cj)],ctemp[('dm',ci,cj)],'dm/du_%i vs. u_%i' % (ci,cj))
            for ci,cjDict in self.advection.items():
                for cj in list(cjDict.keys()):
                    nSpace = ctemp[('f',ci)].shape[-1]
                    for I in range(nSpace):
                        linePlot(ctemp[('u',cj)],ctemp[('f',ci)],'f_%i vs. u_%i' % (ci,cj),start=I,stride=nSpace)
                        linePlot(ctemp[('u',cj)],ctemp[('df',ci,cj)],'df/du_%i vs. u_%i' % (ci,cj),start=I,stride=nSpace)
            for ci,ckDict in self.diffusion.items():
                for ck,cjDict in ckDict.items():
                    for cj in list(cjDict.keys()):
                        nSpace = ctemp[('a',ci,ck)].shape[-1]
                        for I in range(nSpace):
                            for J in range(nSpace):
                                linePlot(ctemp[('u',cj)],ctemp[('a',ci,ck)],'a_%i,%i vs. u_%i' % (ci,ck,cj),start=I*nSpace+J,stride=nSpace*nSpace)
                                linePlot(ctemp[('u',cj)],ctemp[('da',ci,ck,cj)],'da/du_%i,%i vs. u_%i' % (ci,ck,cj),start=I*nSpace+J,stride=nSpace*nSpace)
            for ci,cjDict in self.potential.items():
                for cj,flag in cjDict.items():
                    if flag == 'nonlinear':
                        linePlot(ctemp[('u',cj)],ctemp[('phi',ci)],'phi_%i vs. u_%i' % (ci,cj) )
                        linePlot(ctemp[('u',cj)],ctemp[('dphi',ci,cj)],'dphi/du_%i vs. u_%i' % (ci,cj))
            for ci,cjDict in self.reaction.items():
                for cj in list(cjDict.keys()):
                    linePlot(ctemp[('u',cj)],ctemp[('r',ci)],'r_%i vs. u_%i' % (ci,cj) )
                    linePlot(ctemp[('u',cj)],ctemp[('dr',ci,cj)],'dr/du_%i vs. u_%i' % (ci,cj))
            for ci,cjDict in self.hamiltonian.items():
                for cj in list(cjDict.keys()):
                    linePlot(ctemp[('u',cj)],ctemp[('H',ci)],'H_%i vs. u_%i' % (ci,cj) )
                    linePlot(ctemp[('u',cj)],ctemp[('dH',ci,cj)],'dH/dgrad_u_%i vs. u_%i' % (ci,cj))

##\brief Linear advection-diffusion-reaction (single-component or uncoupled multi-component systems)
#
#The system of equations is formulated as
#
#\f[
# m^i u^i_t + \nabla \cdot \left( \mathbf{b}^i u^i - \mathbf{a}^i \nabla u^i \right) + c^i u^i = 0
#\f]
#
#where \f$i=0,\ldots,nc-1\f$
from .ctransportCoefficients import linearADR_ConstantCoefficientsEvaluate
class LinearVADR_ConstantCoefficients(TC_base):
    """
    This class implements constant coefficients with no cross-component diffusion for
    a system of advection-diffuion-reaction equations.
    """
    def __init__(self,nc=1,M=[0],A=[0],B=[0],C=[0],rFunc=None,useSparseDiffusion = True):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        for i in range(nc):
            mass[i]      = {i:'linear'}
            advection[i] = {i:'linear'}
            diffusion[i] = {i: {i:'constant'}}
            potential[i] = {i: 'u'}
            reaction[i]  = {i: 'linear'}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         useSparseDiffusion = useSparseDiffusion)
        self.M = M
        self.A = A
        self.B = B
        self.C = C
        self.rFunc=rFunc
    def evaluate(self,t,c):
        for  i  in range(self.nc):
            linearADR_ConstantCoefficientsEvaluate(self.M[i],
                                                   self.A[i],
                                                   self.B[i],
                                                   self.C[i],
                                                   t,
                                                   c['x'],
                                                   c[('u',i)],
                                                   c[('m',i)],c[('dm',i,i)],
                                                   c[('f',i)],c[('df',i,i)],
                                                   c[('a',i,i)],
                                                   c[('r',i)],c[('dr',i,i)])
            nSpace=c['x'].shape[-1]
            if self.rFunc is not None:
                for n in range(len(c[('u',i)].flat)):
                    c[('r',i)].flat[n] = self.rFunc[i].rOfUX(c[('u',i)].flat[n],c['x'].flat[n*nSpace:(n+1)*nSpace])
                    c[('dr',i,i)].flat[n] = self.rFunc[i].drOfUX(c[('u',i)].flat[n],c['x'].flat[n*nSpace:(n+1)*nSpace])

##\brief Linear advection-diffusion-reaction skew-symmetric systems)
#
#The equations are formulated as
#
#\f[
# m^i u^{nc-1-i} + \nabla \cdot \left( \mathbf{b}^i u^{nc-1-i} - \mathbf{a}^{i,nc-1-i} \nabla u^{nc-1-i} \right) + c^i u^{nc-1-i} = 0
#\f]
#
#where \f$=0,\ldots,nc-1\f$
from .ctransportCoefficients import linearADR_ConstantCoefficientsEvaluate
class LinearVADR_ConstantCoefficients_skew(TC_base):
    """
    This class implements constant coefficients for a skew symmetric system, mainly for testing and debugging
    """
    def __init__(self,nc=1,M=0,A=0,B=0,C=0):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        for i in range(nc):
            mass[i]      = {nc-1-i:'linear'}
            advection[i] = {nc-1-i:'linear'}
            diffusion[i] = {nc-1-i: {nc-1-i:'constant'}}
            potential[i] = {i: 'u'}
            reaction[i]  = {nc-1-i: 'linear'}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
        self.M = M
        self.A = A
        self.B = B
        self.C = C
    def evaluate(self,t,c):
        nc=self.nc
        for  i  in range(self.nc):
            linearADR_ConstantCoefficientsEvaluate(self.M[i],
                                                   self.A[i],
                                                   self.B[i],
                                                   self.C[i],
                                                   t,
                                                   c['x'],
                                                   c[('u',nc-1-i)],
                                                   c[('m',i)],c[('dm',i,nc-1-i)],
                                                   c[('f',i)],c[('df',i,nc-1-i)],
                                                   c[('a',i,nc-1-i)],
                                                   c[('r',i)],c[('dr',i,nc-1-i)])

##\brief Linear advection-diffusion-reaction (upper triangular multi-component systems)
#
#The equations have the form
#
#\f[
# m^i + \nabla \cdot \left( \mathbf{f}^i + \sum_{j=i}^{nc-1} \mathbf{a}^i \nabla u^j \right) + r^i = 0 \\
#\f]
#
#where \f$i,j=0,\ldots,cn-1\f$ and
#
#\f{eqnarray*}
#m^i &=& M^i u^i + \sum_{j = i+1}^{nc-1} m^j \\
#f^i &=& \mathbf{b}^i u^i + \sum_{j = i+1}^{nc-1} f^j \\
#r^i &=& c^i u^i + \sum_{j = i+1}^{nc-1} r^j
#\f}
from .ctransportCoefficients import linearADR_ConstantCoefficientsEvaluate
class LinearVADR_ConstantCoefficients_upper(TC_base):
    """
    This class implements constant coefficients with upper diagonal coupling
    """
    def __init__(self,nc=1,M=0,A=0,B=0,C=0):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        for i in range(nc):
            mass[i]={}
            advection[i]={}
            diffusion[i]={}
            potential[i]={i:'u'}
            reaction[i]={}
            for j in range(nc-1,i-1,-1):
                mass[i][j]='linear'
                advection[i][j] = 'linear'
                diffusion[i][j] = {j:'constant'}
                reaction[i][j] = 'linear'
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
        self.M = M
        self.A = A
        self.B = B
        self.C = C
    def evaluate(self,t,c):
        for  i  in range(self.nc-1,-1,-1):
            linearADR_ConstantCoefficientsEvaluate(self.M[i],
                                                   self.A[i],
                                                   self.B[i],
                                                   self.C[i],
                                                   t,
                                                   c['x'],
                                                   c[('u',i)],
                                                   c[('m',i)],c[('dm',i,i)],
                                                   c[('f',i)],c[('df',i,i)],
                                                   c[('a',i,i)],
                                                   c[('r',i)],c[('dr',i,i)])
            for j in range(self.nc-1,i,-1):
                c[('m',i)]+=c[('m',j)]
                c[('dm',i,j)].flat[:]=c[('dm',j,j)].flat
                c[('f',i)]+=c[('f',j)]
                c[('df',i,j)].flat[:]=c[('df',j,j)].flat
                c[('a',i,j)].flat[:]=c[('a',j,j)].flat
                c[('r',i)]+=c[('r',j)]
                c[('dr',i,j)].flat[:]=c[('dr',j,j)].flat

##\brief Linear advection-diffusion-reaction (lower triangular multi-component systems)
#
#The equations have the form
#\f[
# m^i + \nabla \cdot \left( \mathbf{f}^i + \sum_{j=i}^{nc-1} \mathbf{a}^i \nabla u^j \right) + r^i = 0 \\
#\f]
#
#where \f$i,j=0,\ldots,cn-1\f$ and
#
#\f{eqnarray*}
#m^i &=& M^i u^i + \sum_{j = 0}^{i-1} m^j \\
#f^i &=& \mathbf{b}^i u^i + \sum_{j = 0}{i-1} f^j \\
#r^i &=& c^i u^i + \sum_{j = 0}{i-1} r^j
#\f}
from .ctransportCoefficients import linearADR_ConstantCoefficientsEvaluate
class LinearVADR_ConstantCoefficients_lower(TC_base):
    """
    This class implements constant coefficients with lower diagonal coupling
    """
    def __init__(self,nc=1,M=0,A=0,B=0,C=0):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        for i in range(nc):
            mass[i]={}
            advection[i]={}
            diffusion[i]={}
            potential[i]={i:'u'}
            reaction[i]={}
            for j in range(i+1):
                mass[i][j]='linear'
                advection[i][j] = 'linear'
                diffusion[i][j] = {j:'constant'}
                reaction[i][j] = 'linear'
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
        self.M = M
        self.A = A
        self.B = B
        self.C = C
    def evaluate(self,t,c):
        for  i  in range(self.nc):
            linearADR_ConstantCoefficientsEvaluate(self.M[i],
                                                   self.A[i],
                                                   self.B[i],
                                                   self.C[i],
                                                   t,
                                                   c['x'],
                                                   c[('u',i)],
                                                   c[('m',i)],c[('dm',i,i)],
                                                   c[('f',i)],c[('df',i,i)],
                                                   c[('a',i,i)],
                                                   c[('r',i)],c[('dr',i,i)])
            for j in range(i):
                c[('m',i)]+=c[('m',j)]
                c[('dm',i,j)].flat[:]=c[('dm',j,j)].flat
                c[('f',i)]+=c[('f',j)]
                c[('df',i,j)].flat[:]=c[('df',j,j)].flat
                c[('a',i,j)].flat[:]=c[('a',j,j)].flat
                c[('r',i)]+=c[('r',j)]
                c[('dr',i,j)].flat[:]=c[('dr',j,j)].flat


##\brief Linear advection-diffusion-reaction (fully-coupled, multi-component systems)
#
#The equations have the form
#\f[
# m^i + \nabla \cdot \left( \mathbf{f}^i + \sum_{j=i}^{nc-1} \mathbf{a}^i \nabla u^j \right) + r^i = 0 \\
#\f]
#
#where \f$i,j=0,\ldots,cn-1\f$ and
#
#\f{eqnarray*}
#m^i &=& M^i u^i + \sum_{j \neq i} \epsilon m^j \\
#f^i &=& \mathbf{b}^i u^i + \sum_{j \neq i} \epsilon f^j \\
#a^{i,j} &=& \epsilon \mathbf{a}^{i,i} \quad j \neq i \\
#r^i &=& c^i u^i + \sum_{j \neq i} \epsilon r^j
#\f}
from .ctransportCoefficients import linearADR_ConstantCoefficientsEvaluate
class LinearVADR_ConstantCoefficients_full(TC_base):
    """
    This class implements constant coefficients with full coupling
    """
    def __init__(self,nc=1,M=0,A=0,B=0,C=0):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        for i in range(nc):
            mass[i]={}
            advection[i]={}
            diffusion[i]={}
            potential[i]={i:'u'}
            reaction[i]={}
            for j in range(nc):
                mass[i][j]='linear'
                advection[i][j] = 'linear'
                diffusion[i][j] = {j:'constant'}
                reaction[i][j] = 'linear'
        TC_base.__init__(self,
                                             nc,
                                             mass,
                                             advection,
                                             diffusion,
                                             potential,
                                             reaction,
                                             hamiltonian)
        self.M = M
        self.A = A
        self.B = B
        self.C = C
        self.eps=0.1
    def evaluate(self,t,c):
        mtmp=numpy.zeros(c[('m',0)].shape,'d')
        ftmp=numpy.zeros(c[('f',0)].shape,'d')
        rtmp=numpy.zeros(c[('r',0)].shape,'d')
        for  i  in range(self.nc):
            linearADR_ConstantCoefficientsEvaluate(self.M[i],
                                                   self.A[i],
                                                   self.B[i],
                                                   self.C[i],
                                                   t,
                                                   c['x'],
                                                   c[('u',i)],
                                                   c[('m',i)],c[('dm',i,i)],
                                                   c[('f',i)],c[('df',i,i)],
                                                   c[('a',i,i)],
                                                   c[('r',i)],c[('dr',i,i)])
            for j in list(range(0,i))+list(range(i+1,self.nc)):
                linearADR_ConstantCoefficientsEvaluate(self.eps*self.M[j],
                                                       self.eps*self.A[j],
                                                       self.eps*self.B[j],
                                                       self.eps*self.C[j],
                                                       t,
                                                       c['x'],
                                                       c[('u',j)],
                                                       mtmp,c[('dm',i,j)],
                                                       ftmp,c[('df',i,j)],
                                                       c[('a',i,j)],
                                                       rtmp,c[('dr',i,j)])
                c[('m',i)]+=mtmp
                c[('f',i)]+=ftmp
                c[('r',i)]+=rtmp


##Nonlinear advection-diffusion-reaction (single-component or uncoupled multi-component systems)
#
#Each equation is written as
#
#\f[
# m u^p + \nabla \cdot \left( \mathbf{b} u^q - \mathbf{a} u^t \nabla u^r \right) + c u^s = 0
#\f]
from .ctransportCoefficients import nonlinearADR_pqrstEvaluate
class NonlinearVADR_pqrst(TC_base):
    """
    This class implements simple monomial nonlinear coefficients with no cross-component diffusion for
    a system of advection-diffuion-reaction equations.
    """
    def __init__(self,nc=1,M={0:0.0},A={0:1.0},B={0:0.0},C={0:0.0},
                 p={0:1.0},q={0:1.0},r={0:1.0},s={0:1.0},t={0:0.0}):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        for i in range(nc):
            if p[i] != 1.0:
                mass[i]      = {i:'nonlinear'}
            else:
                mass[i]      = {i:'linear'}
            if q[i] != 1.0:
                advection[i] = {i:'nonlinear'}
            else:
                advection[i] = {i:'linear'}
            if t[i] != 0:
                diffusion[i] = {i: {i:'nonlinear'}}
            else:
                diffusion[i] = {i: {i:'constant'}}
            if r[i] != 1:
                potential[i] = {i: 'nonlinear'}
            else:
                potential[i] = {i: 'u'}
            if s[i] != 1:
                reaction[i]  = {i: 'nonlinear'}
            else:
                reaction[i]  = {i: 'linear'}
        TC_base.__init__(self,
                                             nc,
                                             mass,
                                             advection,
                                             diffusion,
                                             potential,
                                             reaction,
                                             hamiltonian)
        self.M = M
        self.A = A
        self.B = B
        self.C = C
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.t = t
        self.useC=True
    def evaluate(self,t,c):
        for  i  in range(self.nc):
            self.nonlinearADR_pqrstEvaluate(self.M[i],
                                            self.A[i],
                                            self.B[i],
                                            self.C[i],
                                            self.p[i],
                                            self.q[i],
                                            self.r[i],
                                            self.s[i],
                                            self.t[i],
                                            t,
                                            c['x'],
                                            c[('u',i)],
                                            c[('m',i)],c[('dm',i,i)],
                                            c[('f',i)],c[('df',i,i)],
                                            c[('a',i,i)],c[('da',i,i,i)],
                                            c[('phi',i)],c[('dphi',i,i)],
                                            c[('r',i)],c[('dr',i,i)])

##\brief Nonlinear advection-diffusion reaction (fully-coupled, multi-component systems)
#
#The equations have the form
#
#\f[
# m^i + \nabla \cdot \left( \mathbf{f}^i + \sum_{j=i}^{nc-1} \mathbf{a}^i \nabla u^j \right) + r^i = 0 \\
#\f]
#
#where \f$i,j=0,\ldots,cn-1\f$ and
#
#\f{eqnarray*}
#m^i &=& M^i (u^i)^p + \sum_{j \neq i} \epsilon m^j \\
#f^i &=& \mathbf{b}^i (u^i)^q + \sum_{j \neq i} \epsilon f^j \\
#a^{i,i} &=& A^i (u^i)^t \\
#a^{i,j} &=& \epsilon \mathbf{a}^{i,i} \quad j \neq i \\
#\phi^i &=& (u^i)^r \\
#r^i &=& c^i (u^i)^s + \sum_{j \neq i} \epsilon r^j
#\f}
from .ctransportCoefficients import nonlinearADR_pqrstEvaluate
class NonlinearVADR_pqrst_full(TC_base):
    """
    This class implements the simple nonlinear ADR equation with full coupling
    """
    def __init__(self,nc=1,M={0:0.0},A={0:1.0},B={0:0.0},C={0:0.0},
                 p={0:1.0},q={0:1.0},r={0:1.0},s={0:1.0},t={0:0.0}):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        for i in range(nc):
            mass[i]={}
            advection[i]={}
            diffusion[i]={}
            potential[i] = {i: 'u'}
            reaction[i]={}
            for j in range(nc):
                mass[i][j] = 'nonlinear'
                advection[i][j] = 'nonlinear'
                diffusion[i][j] =  {j:'nonlinear'}
                reaction[i][j] =  'nonlinear'
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
        self.M = M
        self.A = A
        self.B = B
        self.C = C
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.t = t
        self.useC=True
        self.eps=0.01
    def evaluate(self,t,c):
        mtmp=numpy.zeros(c[('m',0)].shape,'d')
        ftmp=numpy.zeros(c[('f',0)].shape,'d')
        rtmp=numpy.zeros(c[('r',0)].shape,'d')
        phitmp=numpy.zeros(c[('phi',0)].shape,'d')
        for  i  in range(self.nc):
            nonlinearADR_pqrstEvaluate(self.M[i],
                                       self.A[i],
                                       self.B[i],
                                       self.C[i],
                                       self.p[i],
                                       self.q[i],
                                       self.r[i],
                                       self.s[i],
                                       self.t[i],
                                       t,
                                       c['x'],
                                       c[('u',i)],
                                       c[('m',i)],c[('dm',i,i)],
                                       c[('f',i)],c[('df',i,i)],
                                       c[('a',i,i)],c[('da',i,i,i)],
                                       phitmp,phitmp,
                                       c[('r',i)],c[('dr',i,i)])
            for j in list(range(0,i))+list(range(i+1,self.nc)):
                nonlinearADR_pqrstEvaluate(self.eps*self.M[j],
                                           self.eps*self.A[j],
                                           self.eps*self.B[j],
                                           self.eps*self.C[j],
                                           self.p[j],
                                           self.q[j],
                                           self.r[j],
                                           self.s[j],
                                           self.t[j],
                                           t,
                                           c['x'],
                                           c[('u',j)],
                                           mtmp,c[('dm',i,j)],
                                           ftmp,c[('df',i,j)],
                                           c[('a',i,j)],c[('da',i,j,j)],
                                           phitmp,phitmp,
                                           rtmp,c[('dr',i,j)])
                c[('m',i)]+=mtmp
                c[('f',i)]+=ftmp
                c[('r',i)]+=rtmp

##\brief Conservative linear advection with a rotating velocity field
#
#The equations are fomulated as
#\f[
# u_t + \nabla \cdot (\mathbf{v} u)
#\f]
#
#where
#
#\f{eqnarray*}
# v_x &=& 2 \pi (y - 1/2) \\
# v_y &=& 2 \pi (1/2-x) \\
#\f}
from .ctransportCoefficients import unitSquareRotationEvaluate
class UnitSquareRotation(TC_base):
    """
    Conservative linear advection with a rotating velocity field
    """
    def __init__(self):
        mass={0:{0:'linear'}}
        advection={0:{0:'linear'}}
        diffusion={0:{0:{0:'constant'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         1,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
    def evaluate(self,t,c):
        unitSquareRotationEvaluate(c['x'],
                                   c[('u',0)],
                                   c[('m',0)],c[('dm',0,0)],
                                   c[('f',0)],c[('df',0,0)])
        
from .ctransportCoefficients import unitCubeRotationEvaluate
class UnitCubeRotation(TC_base):
    """
    Conservative linear advection with a rotating velocity field in 3d
    """
    def __init__(self):
        mass={0:{0:'linear'}}
        advection={0:{0:'linear'}}
        diffusion={0:{0:{0:'constant'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         1,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
    def evaluate(self,t,c):
        unitCubeRotationEvaluate(c['x'],
                                 c[('u',0)],
                                 c[('m',0)],c[('dm',0,0)],
                                 c[('f',0)],c[('df',0,0)])

##\brief Incompressible Navier-Stokes equations
#
#The equations are formulated as
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \left \mathbf{v}\right }{\partial t} + \nabla \cdot \left(\mathbf{v} \otimes \mathbf{v} - \nu \Delta \mathbf{f} \right) + \frac{\nabla p}{\rho}- \mathbf{g} &=& 0
#\f}
#
#where \f$\mathbf{v}\f$ is the velocity, \f$p\f$ is the pressure, \f$\nu\f$ is the kinematic viscosity, \f$\rho\f$ is the density, and \f$\mathbf{g}\f$ is the gravitational acceleration.
from .ctransportCoefficients import NavierStokes_2D_Evaluate
from .ctransportCoefficients import NavierStokes_3D_Evaluate
class NavierStokes(TC_base):
    """
    The coefficients for the incompressible Navier-Stokes equations.
    """
    def __init__(self,rho=998.2,nu=1.004e-6,g=[0.0,9.8],nd=2):
        self.rho = rho
        self.nu = nu
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            mass= {1:{1:'linear'},
                   2:{2:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear'}}
            diffusion  = {1:{1:{1:'constant'}},
                          2:{2:{2:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            mass = {1:{1:'linear'},
                    2:{2:'linear'},
                    3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         3:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}},
                         3:{3:{3:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'},
                        3:{3:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'},
                        3:{3:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
    def evaluate(self,t,c):
        if self.nd==2:
            NavierStokes_2D_Evaluate(self.rho,
                                     self.nu,
                                     self.g,
                                     c[('u',0)],
                                     c[('grad(u)',0)],
                                     c[('u',1)],
                                     c[('u',2)],
                                     c[('m',1)],
                                     c[('dm',1,1)],
                                     c[('m',2)],
                                     c[('dm',2,2)],
                                     c[('f',0)],
                                     c[('df',0,1)],
                                     c[('df',0,2)],
                                     c[('f',1)],
                                     c[('df',1,1)],
                                     c[('df',1,2)],
                                     c[('f',2)],
                                     c[('df',2,1)],
                                     c[('df',2,2)],
                                     c[('a',1,1)],
                                     c[('a',2,2)],
                                     c[('r',1)],
                                     c[('r',2)],
                                     c[('H',1)],
                                     c[('dH',1,0)],
                                     c[('H',2)],
                                     c[('dH',2,0)])
        elif  self.nd==3:
            NavierStokes_3D_Evaluate(self.rho,
                                     self.nu,
                                     self.g,
                                     c[('u',0)],
                                     c[('grad(u)',0)],
                                     c[('u',1)],
                                     c[('u',2)],
                                     c[('u',3)],
                                     c[('m',1)],
                                     c[('dm',1,1)],
                                     c[('m',2)],
                                     c[('dm',2,2)],
                                     c[('m',3)],
                                     c[('dm',3,3)],
                                     c[('f',0)],
                                     c[('df',0,1)],
                                     c[('df',0,2)],
                                     c[('df',0,3)],
                                     c[('f',1)],
                                     c[('df',1,1)],
                                     c[('df',1,2)],
                                     c[('df',1,3)],
                                     c[('f',2)],
                                     c[('df',2,1)],
                                     c[('df',2,2)],
                                     c[('df',2,3)],
                                     c[('f',3)],
                                     c[('df',3,1)],
                                     c[('df',3,2)],
                                     c[('df',3,3)],
                                     c[('a',1,1)],
                                     c[('a',2,2)],
                                     c[('a',3,3)],
                                     c[('r',1)],
                                     c[('r',2)],
                                     c[('r',3)],
                                     c[('H',1)],
                                     c[('dH',1,0)],
                                     c[('H',2)],
                                     c[('dH',2,0)],
                                     c[('H',3)],
                                     c[('dH',3,0)])
from .ctransportCoefficients import shallowWater_1D_Evaluate
from .ctransportCoefficients import shallowWater_2D_Evaluate
class ShallowWater(TC_base):
    r"""The coefficients for the shallow water equations.

    Right hand side for bed friction looks like :math:`-\tau_b/\rho`
    where the bed friction stress is

    .. math::

       \tau_b = \rho C_f \vec u \|\vec u\|
       C_f = g b/h^{a}

    :math:`b = n^2` for Mannings law --> bedFrictionCoefficient
    :math:`a = 1/3` for Mannings law --> bedFrictionPower

    """
#    from ctransportCoefficients import shallowWater_3D_Evaluate
    def __init__(self,g=9.8,nd=1,h_eps=1.0e-8,
                 bedFrictionCoefficient=0.0, #bed friction law coefficient (n^2 for Manning)
                 bedFrictionPower=1.0,#h power in bed frition law (4/3) for Manning
                 bathymetryFunc=None,bathymetryGradientFunc=None,
                 eddyViscosity = 0.0):
        self.g = g
        self.nd=nd
        self.h_eps=1.0e-8
        self.bedFrictionCoefficient=bedFrictionCoefficient
        self.bedFrictionPower = bedFrictionPower
        self.eddyViscosity    = eddyViscosity
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        self.bathymetryFunc=bathymetryFunc
        self.bathymetryGradientFunc=bathymetryGradientFunc
        assert ((self.bathymetryFunc is None and self.bathymetryGradientFunc is None) or
                (self.bathymetryFunc is not None and self.bathymetryGradientFunc is not None))
        #index of bathymetry values in spatial points
        self.bind = 2;
        if self.nd == 1:
            self.bind=1
        if nd==1:
            variableNames=['h','hu']
            mass= {0:{0:'linear'},
                   1:{1:'linear'}}
            advection = {0:{1:'nonlinear'},
                         1:{0:'nonlinear',
                            1:'nonlinear'}}
            reaction  = {1:{0:'nonlinear',1:'nonlinear'}}
            diffusion = {1:{1:{1:'linear'}}}
            potential = {1:{1: 'u'}}
            sdInfo    = {(1,1):(numpy.array([0,1],dtype='i'),
                                numpy.array([0],dtype='i'))}
            TC_base.__init__(self,
                             2,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo)
            self.vectorComponents=None
        if nd==2:
            variableNames=['h','hu','hv']
            mass= {0:{0:'linear'},
                   1:{1:'linear'},
                   2:{2:'linear'}}
            advection = {0:{1:'nonlinear',
                            2:'nonlinear'},
                         1:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear'},
                         2:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear'}}
            reaction  = {1:{0:'nonlinear',1:'nonlinear',2:'nonlinear'},
                         2:{0:'nonlinear',1:'nonlinear',2:'nonlinear'}}
            diffusion = {1:{1:{1:'linear'}},
                         2:{2:{2:'linear'}}}
            potential = {1:{1: 'u'},
                         2:{2: 'u'}}
            sdInfo    = {(1,1):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i')),
                         (2,2):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i'))}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo)
            #self.vectorComponents=[1,2]
        #save bathymetry slopes for different quadrature types
        for term in ['q_grad_b','ebq_grad_b','ebq_global_grad_b','ebqe_grad_b','ip_grad_b']:
            setattr(self,term,None)

    def initializeMesh(self,mesh):
        if self.bathymetryFunc:
            #mesh vertices are held in vector nodeArray which is nNodes_global x 3
            for nN in range(mesh.nNodes_global):
                x = mesh.nodeArray.flat[nN*3:(nN+1)*3]; b = self.bathymetryFunc(x)
                mesh.nodeArray.flat[nN*3+self.bind]=b
    def initializeElementQuadrature(self,t,cq):
        self.q_grad_b = numpy.zeros(cq[('grad(u)',0)].shape,'d')
        if self.bathymetryFunc:
            #cq is elementQuadrature dictionary so x is nElements_global x nQuadraturePoints_element x 3
            nPoints = cq['x'].shape[0]*cq['x'].shape[1]
            for i in range(nPoints):
                x = cq['x'].flat[i*3:(i+1)*3]; b = self.bathymetryFunc(x); grad_b = self.bathymetryGradientFunc(x)
                cq['x'].flat[i*3+self.bind]=b
                self.q_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b
        #
        cq[('H',0)] = numpy.copy(cq[('u',0)])
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.ebq_grad_b = numpy.zeros(cebq[('grad(u)',0)].shape,'d')
        if ('grad(u)',0) in cebq_global:
            self.ebq_global_grad_b = numpy.zeros(cebq_global[('grad(u)',0)].shape,'d')
        else:
            sh = (cebq_global['x'].shape[0],cebq_global['x'].shape[1],self.nd)
            self.ebq_global_grad_b = numpy.zeros(sh,'d')
        if self.bathymetryFunc:
            #cebq is quadrature dictionary for local "faces" on each element so
            #  x is nElements_global x nElementBoundaries_element x nQuadraturePoints_elementBoundary x 3
            nPoints = cebq['x'].shape[0]*cebq['x'].shape[1]*cebq['x'].shape[2]
            for i in range(nPoints):
                x = cebq['x'].flat[i*3:(i+1)*3]; b =  self.bathymetryFunc(x); grad_b = self.bathymetryGradientFunc(x)
                cebq['x'].flat[i*3+self.bind]=b
                self.ebq_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b
            #cebq_global is quadrature dictionary for global "faces" so
            #x is nElementBoundaries_global x nQuadraturePoints_elementBoundary x 3
            nPoints = cebq_global['x'].shape[0]*cebq_global['x'].shape[1]
            for i in range(nPoints):
                x = cebq_global['x'].flat[i*3:(i+1)*3]; b =  self.bathymetryFunc(x); grad_b = self.bathymetryGradientFunc(x)
                cebq_global['x'].flat[i*3+self.bind]=b
                self.ebq_global_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b
        #
        if ('u',0) in cebq_global:
            cebq_global[('H',0)] = numpy.copy(cebq_global[('u',0)]);
        else:
            sh = (cebq_global['x'].shape[0],cebq_global['x'].shape[1])
            cebq_global[('H',0)] = numpy.zeros(sh,'d')
        cebq[('H',0)] = numpy.copy(cebq[('u',0)])

    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.ebqe_grad_b = numpy.zeros(cebqe[('grad(u)',0)].shape,'d')
        if self.bathymetryFunc:
            #cebqe is quadrature dictionary for global exterior "faces"
            #  x is nExteriorElementBoundaries_global x nQuadraturePoints_elementBoundary x 3
            nPoints = cebqe['x'].shape[0]*cebqe['x'].shape[1]
            for i in range(nPoints):
                x = cebqe['x'].flat[i*3:(i+1)*3]; b =  self.bathymetryFunc(x); grad_b = self.bathymetryGradientFunc(x)
                cebqe['x'].flat[i*3+self.bind]=b
                self.ebqe_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b
        #
        cebqe[('H',0)] = numpy.copy(cebqe[('u',0)]);

    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        self.ip_grad_b = numpy.zeros((cip['x'].shape[0],cip['x'].shape[1],self.nd),'d')
        if self.bathymetryFunc:
            #cip is dictionary for interpolation points
            #  so x is nElements x number of interpolation points (usual dof) per element
            nPoints = cip['x'].shape[0]*cip['x'].shape[1]
            for i in range(nPoints):
                x = cip['x'].flat[i*3:(i+1)*3]; b =  self.bathymetryFunc(x);  grad_b = self.bathymetryGradientFunc(x)
                cip['x'].flat[i*3+self.bind]=b
                self.ip_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b
        #
        cip[('H',0)] = numpy.copy(cip[('u',0)])
    def evaluate(self,t,c):
        grad_b = None
        if c['x'].shape[0:-1] == self.q_grad_b.shape[0:-1]:
            grad_b = self.q_grad_b
        elif  c['x'].shape[0:-1] == self.ebqe_grad_b.shape[0:-1]:
            grad_b = self.ebqe_grad_b
        elif  c['x'].shape[0:-1] == self.ip_grad_b.shape[0:-1]:
            grad_b = self.ip_grad_b
        elif  c['x'].shape[0:-1] == self.ebq_grad_b.shape[0:-1]:
            grad_b = self.ebq_grad_b
        elif  c['x'].shape[0:-1] == self.ebq_global_grad_b.shape[0:-1]:
            grad_b = self.ebq_global_grad_b
        assert grad_b is not None
        #total elevation
        if ('H',0) not in c and ('u',0) in c:
            c[('H',0)] = numpy.copy(c[('u',0)])
        if self.nd==1:
            shallowWater_1D_Evaluate(self.h_eps,
                                     self.g,
                                     self.bedFrictionCoefficient,
                                     self.bedFrictionPower,
                                     self.eddyViscosity,
                                     c['x'],
                                     grad_b,
                                     c[('u',0)],
                                     c[('u',1)],
                                     c[('H',0)],
                                     c[('m',0)],
                                     c[('dm',0,0)],
                                     c[('m',1)],
                                     c[('dm',1,1)],
                                     c[('f',0)],
                                     c[('df',0,1)],
                                     c[('f',1)],
                                     c[('df',1,0)],
                                     c[('df',1,1)],
                                     c[('a',1,1)],
                                     c[('r',1)],
                                     c[('dr',1,0)],
                                     c[('dr',1,1)])
        if self.nd==2:
            shallowWater_2D_Evaluate(self.h_eps,
                                     self.g,
                                     self.bedFrictionCoefficient,
                                     self.bedFrictionPower,
                                     self.eddyViscosity,
                                     c['x'],
                                     grad_b,
                                     c[('u',0)],
                                     c[('u',1)],
                                     c[('u',2)],
                                     c[('H',0)],
                                     c[('m',0)],
                                     c[('dm',0,0)],
                                     c[('m',1)],
                                     c[('dm',1,1)],
                                     c[('m',2)],
                                     c[('dm',2,2)],
                                     c[('f',0)],
                                     c[('df',0,1)],
                                     c[('df',0,2)],
                                     c[('f',1)],
                                     c[('df',1,0)],
                                     c[('df',1,1)],
                                     c[('df',1,2)],
                                     c[('f',2)],
                                     c[('df',2,0)],
                                     c[('df',2,1)],
                                     c[('df',2,2)],
                                     c[('a',1,1)],
                                     c[('a',2,2)],
                                     c[('r',1)],
                                     c[('dr',1,0)],
                                     c[('dr',1,1)],
                                     c[('dr',1,2)],
                                     c[('r',2)],
                                     c[('dr',2,0)],
                                     c[('dr',2,1)],
                                     c[('dr',2,2)])
            #for now allow constant eddy viscosity
            #once have diagonal sparse rep
            #c[('a',1,1)].fill(self.eddyViscosity)
            #c[('a',2,2)].fill(self.eddyViscosity)
#         elif  self.nd==3:
#             self.NavierStokes_3D_Evaluate(self.rho,
#                                           self.nu,
#                                           self.g,
#                                           c[('u',0)],
#                                           c[('grad(u)',0)],
#                                           c[('u',1)],
#                                           c[('u',2)],
#                                           c[('u',3)],
#                                           c[('m',1)],
#                                           c[('dm',1,1)],
#                                           c[('m',2)],
#                                           c[('dm',2,2)],
#                                           c[('m',3)],
#                                           c[('dm',3,3)],
#                                           c[('f',0)],
#                                           c[('df',0,1)],
#                                           c[('df',0,2)],
#                                           c[('df',0,3)],
#                                           c[('f',1)],
#                                           c[('df',1,1)],
#                                           c[('df',1,2)],
#                                           c[('df',1,3)],
#                                           c[('f',2)],
#                                           c[('df',2,1)],
#                                           c[('df',2,2)],
#                                           c[('df',2,3)],
#                                           c[('f',3)],
#                                           c[('df',3,1)],
#                                           c[('df',3,2)],
#                                           c[('df',3,3)],
#                                           c[('a',1,1)],
#                                           c[('a',2,2)],
#                                           c[('a',3,3)],
#                                           c[('r',1)],
#                                           c[('r',2)],
#                                           c[('r',3)],
#                                           c[('H',1)],
#                                           c[('dH',1,0)],
#                                           c[('H',2)],
#                                           c[('dH',2,0)],
#                                           c[('H',3)],
#                                           c[('dH',3,0)])
from .ctransportCoefficients import Laplace_2D_Evaluate
from .ctransportCoefficients import Laplace_3D_Evaluate
class DiscreteLaplaceOperator(TC_base):
    r""" A coefficient class to construct the discrete Laplace Operator.
    
    This class defines the coefficients necessary to construct the
    discrete Laplace operator :math:`A` where

    .. math::
    
        a^{c}_{i,j} = \int_{T} \nabla \phi^{c}_{i} \cdot \nabla \phi^{c}_{j} dT

    for all :math:`T \in \Omega`, :math:`c=1,...,nc` and 
    :math:`\phi^{c}_{i}, i=1,...,k` is a basis for component :math:`c`.
    """
    def __init__(self,nd=2,nu=1.0):
        self.nd=nd
        self.nu=nu # ... Detail I need to worry about later ...
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {}
        hamiltonian = {}
        if nd==2:
            variableNames=['p','u','v']
            diffusion = {0:{0:{0:'constant'}},
                         1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}}}
            potential = {0:{0:'u'},
                         1:{1:'u'},
                         2:{2:'u'}}
            sdInfo    = {(0,0):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i')),
                         (1,1):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i')),
                         (2,2):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i'))}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion=True)
            self.vectorComponents=[1,2]
        if nd==3:
            variableNames=['p','u','v','w']
            diffusion ={0:{0:{0:'constant'}},
                        1:{1:{1:'constant'}},
                        2:{2:{2:'constant'}},
                        3:{3:{3:'constant'}}}
            potential = {0:{0:'u'},
                         1:{1:'u'},
                         2:{2:'u'},
                         3:{3:'u'}}
            sdInfo  = {(0,0):(numpy.array([0,1,2,3],dtype='i'),numpy.array([0,1,2],dtype='i')),
                       (1,1):(numpy.array([0,1,2,3],dtype='i'),numpy.array([0,1,2],dtype='i')),
                       (2,2):(numpy.array([0,1,2,3],dtype='i'),numpy.array([0,1,2],dtype='i')),
                       (3,3):(numpy.array([0,1,2,3],dtype='i'),numpy.array([0,1,2],dtype='i'))}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion=True)
            self.vectorComponents=[1,2,3]
    def evaluate(self,t,c):
        if self.nd==2:
            Laplace_2D_Evaluate(c[('u',0)],
                                c[('u',1)],
                                c[('u',2)],
                                c[('a',0,0)],
                                c[('a',1,1)],
                                c[('a',2,2)])
        if self.nd==3:
            Laplace_3D_Evaluate(c[('u',0)],
                                c[('u',1)],
                                c[('u',2)],
                                c[('u',3)],
                                c[('a',0,0)],
                                c[('a',1,1)],
                                c[('a',2,2)],
                                c[('a',3,3)])

##\brief Incompressible Stokes equations
#
#The equations are formulated as
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \mathbf{v}}{\partial t} + \nabla \cdot \left(- \nu \Delta \mathbf{f} \right) + \frac{\nabla p}{\rho}- \mathbf{g} &=& 0
#\f}
#
#where \f$\mathbf{v}\f$ is the velocity, \f$p\f$ is the pressure, \f$\nu\f$ is the kinematic viscosity, \f$\rho\f$ is the density, and \f$\mathbf{g}\f$ is the gravitational acceleration.
from .ctransportCoefficients import Stokes_2D_Evaluate
from .ctransportCoefficients import Stokes_3D_Evaluate
class Stokes(TC_base):
    """
    The coefficients for the Stokes equations.
    """
    def __init__(self,rho=998.2,nu=1.004e-6,g=[0.0,9.8],nd=2,steady=True,weakBoundaryConditions=True):
        self.steady=steady
        self.rho = rho
        self.nu = nu
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            mass={0:{0:'linear'},
                  1:{1:'linear'},
                  2:{2:'linear'}}
            if not weakBoundaryConditions:
                advection = {0:{0:'linear',
                                1:'linear',
                                2:'linear'}}
            else:
                advection = {0:{0:'linear',
                                1:'linear',
                                2:'linear'},
                             1:{0:'linear'},
                             2:{0:'linear'}}
            diffusion = {0:{0:{0:'constant'}},
                         1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}}}
            potential = {0:{0:'u'},
                         1:{1:'u'},
                         2:{2:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             useSparseDiffusion=True)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            mass={0:{0:'linear'},
                  1:{1:'linear'},
                  2:{2:'linear'},
                  3:{3:'linear'}}
            if not weakBoundaryConditions:
                advection = {0:{0:'linear',
                                1:'linear',
                                2:'linear',
                                3:'linear'}}
            else:
                advection = {0:{0:'linear',
                                1:'linear',
                                2:'linear',
                                3:'linear'},
                             1:{0:'linear'},
                             2:{0:'linear'},
                             3:{0:'linear'}}
            diffusion = {0:{0:{0:'constant'}},
                         1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}},
                         3:{3:{3:'constant'}}}
            potential = {0:{0:'u'},
                         1:{1:'u'},
                         2:{2:'u'},
                         3:{3:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'},
                        3:{3:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             useSparseDiffusion=True)
            self.vectorComponents=[1,2,3]

    def attachModels(self,modelList):
        modelList[0].pp_hasConstantNullSpace = False

    def evaluate(self,t,c):
        if self.nd==2:
            Stokes_2D_Evaluate(self.rho,
                               self.nu,
                               self.g,
                               c[('u',0)],
                               c[('grad(u)',0)],
                               c[('u',1)],
                               c[('u',2)],
                               c[('m',1)],
                               c[('dm',1,1)],
                               c[('m',2)],
                               c[('dm',2,2)],
                               c[('f',0)],
                               c[('df',0,1)],
                               c[('df',0,2)],
                               c[('a',1,1)],
                               c[('a',2,2)],
                               c[('r',1)],
                               c[('r',2)],
                               c[('H',1)],
                               c[('dH',1,0)],
                               c[('H',2)],
                               c[('dH',2,0)])
        elif self.nd==3:
            Stokes_3D_Evaluate(self.rho,
                               self.nu,
                               self.g,
                               c[('u',0)],
                               c[('grad(u)',0)],
                               c[('u',1)],
                               c[('u',2)],
                               c[('u',3)],
                               c[('m',1)],
                               c[('dm',1,1)],
                               c[('m',2)],
                               c[('dm',2,2)],
                               c[('m',3)],
                               c[('dm',3,3)],
                               c[('f',0)],
                               c[('df',0,1)],
                               c[('df',0,2)],
                               c[('df',0,3)],
                               c[('a',1,1)],
                               c[('a',2,2)],
                               c[('a',3,3)],
                               c[('r',1)],
                               c[('r',2)],
                               c[('r',3)],
                               c[('H',1)],
                               c[('dH',1,0)],
                               c[('H',2)],
                               c[('dH',2,0)],
                               c[('H',3)],
                               c[('dH',3,0)])
from .ctransportCoefficients import StokesP_2D_Evaluate
from .ctransportCoefficients import StokesP_3D_Evaluate
class StokesP(TC_base):
    """
    The coefficients for the Stokes equations.
    """
    def __init__(self,rho=998.2,nu=1.004e-6,g=[0.0,9.8],nd=2,steady=True):
        self.steady=steady
        self.rho = rho
        self.nu = nu
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            mass={1:{1:'linear'},
                  2:{2:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear'},
                         1:{0:'linear'},
                         2:{0:'linear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}}}
            potential = {1:{1:'u'},
                         2:{2:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            mass={1:{1:'linear'},
                  2:{2:'linear'},
                  3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'},
                         1:{0:'linear'},
                         2:{0:'linear'},
                         3:{0:'linear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}},
                         3:{3:{3:'constant'}}}
            potential = {1:{1:'u'},
                         2:{2:'u'},
                         3:{3:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'},
                        3:{3:'constant'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
    def evaluate(self,t,c):
        if self.nd==2:
            StokesP_2D_Evaluate(self.rho,
                                self.nu,
                                self.g,
                                c[('u',0)],
                                c[('u',1)],
                                c[('u',2)],
                                c[('m',1)],
                                c[('dm',1,1)],
                                c[('m',2)],
                                c[('dm',2,2)],
                                c[('f',0)],
                                c[('df',0,1)],
                                c[('df',0,2)],
                                c[('f',1)],
                                c[('df',1,0)],
                                c[('f',2)],
                                c[('df',2,0)],
                                c[('a',1,1)],
                                c[('a',2,2)],
                                c[('r',1)],
                                c[('r',2)])
        elif self.nd==3:
            StokesP_3D_Evaluate(self.rho,
                                self.nu,
                                self.g,
                                c[('u',0)],
                                c[('u',1)],
                                c[('u',2)],
                                c[('u',3)],
                                c[('m',1)],
                                c[('dm',1,1)],
                                c[('m',2)],
                                c[('dm',2,2)],
                                c[('m',3)],
                                c[('dm',3,3)],
                                c[('f',0)],
                                c[('df',0,1)],
                                c[('df',0,2)],
                                c[('df',0,3)],
                                c[('f',1)],
                                c[('df',1,0)],
                                c[('f',2)],
                                c[('df',2,0)],
                                c[('f',3)],
                                c[('df',3,0)],
                                c[('a',1,1)],
                                c[('a',2,2)],
                                c[('a',3,3)],
                                c[('r',1)],
                                c[('r',2)],
                                c[('r',3)])

##\brief Two-phase, Incompressible Navier-Stokes equations (level-set formulation)
#
#The equations are
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \mathbf{v} }{\partial t} + \nabla \cdot \left(\mathbf{v}\otimes \mathbf{v} - \nu \Delta \mathbf{f} \right) +\frac{\nabla p}{\rho} - \mathbf{g} &=& 0
#\f}
#
#where \f$\mathbf{v}\f$ is the velocity, \f$p\f$ is the pressure, \f$\nu\f$ is the kinematic viscosity, \f$\rho\f$ is the density, and \f$\mathbf{g}\f$ is the gravitational acceleration.
#
#The variable viscosity and density are given by
#
#\f{eqnarray*}
#\rho &=& \rho_0 (1-H) + \rho_1 H \\
#\nu  &=& \nu_0 (1-H) + \nu_1 H \\
#\f}
#
#with
#
#\f[
# H = \left\{ \begin{array}{lr}
# 0 & \phi \leq - \epsilon \\
# \frac{1}{2}\left(1+\frac{\phi}{\epsilon} + \frac{1}{\pi}\sin(\frac{\pi \phi}{\epsilon})\right) & -\epsilon < \phi < \epsilon \\
# 1 & \phi \geq \epsilon
#\end{array} \right.
#\f]
#
#The level set function, \f$\phi\f$, is provided from some other model (e.g. proteus::TransportCoeffcieints::NCLevelSetCoefficients) that is solved seperately (via operator splitting).
#
from .ctransportCoefficients import TwophaseNavierStokes_LS_SO_2D_Evaluate
from .ctransportCoefficients import TwophaseNavierStokes_LS_SO_3D_Evaluate
class TwophaseNavierStokes_LS_SO(TC_base):
    """
    The coefficients for two incompresslble fluids governed by the Navier-Stokes equations and separated by a sharp interface represented by a level set function
    """
    def __init__(self,
                 epsFact=1.5,
                 rho_0=998.2,nu_0=1.004e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=None):
        self.LS_model=LS_model
        self.epsFact=epsFact
        self.eps=None
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            mass= {1:{1:'linear'},
                   2:{2:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear'}}
            diffusion  = {1:{1:{1:'constant'}},
                          2:{2:{2:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            mass = {1:{1:'linear'},
                    2:{2:'linear'},
                    3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         3:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}},
                         3:{3:{3:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'},
                        3:{3:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'},
                        3:{3:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
    def attachModels(self,modelList):
        if self.LS_model is not None:
            self.q_phi = modelList[self.LS_model].q[('u',0)]
            self.ebqe_phi = modelList[self.LS_model].ebqe[('u',0)]
            self.ebq_phi = None
            if ('u',0) in modelList[self.LS_model].ebq:
                self.ebq_phi = modelList[self.LS_model].ebq[('u',0)]

    def initializeMesh(self,mesh):
        self.eps = self.epsFact*mesh.h
    def initializeElementQuadrature(self,t,cq):
        #initialize so it can run without a flow model
        self.q_phi = numpy.ones(cq[('u',0)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.ebq_phi = numpy.ones(cebq[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.ebqe_phi = numpy.ones(cebqe[('u',0)].shape,'d')
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_phi.shape:
            phi = self.q_phi
        elif c[('u',0)].shape == self.ebqe_phi.shape:
            phi = self.ebqe_phi
        else:
            phi = self.ebq_phi
        if self.nd==2:
            self.TwophaseNavierStokes_LS_SO_2D_Evaluate(self.eps,
                                                        self.rho_0,
                                                        self.nu_0,
                                                        self.rho_1,
                                                        self.nu_1,
                                                        self.g,
                                                        phi,
                                                        c[('u',0)],
                                                        c[('grad(u)',0)],
                                                        c[('u',1)],
                                                        c[('u',2)],
                                                        c[('m',1)],
                                                        c[('dm',1,1)],
                                                        c[('m',2)],
                                                        c[('dm',2,2)],
                                                        c[('f',0)],
                                                        c[('df',0,1)],
                                                        c[('df',0,2)],
                                                        c[('f',1)],
                                                        c[('df',1,1)],
                                                        c[('df',1,2)],
                                                        c[('f',2)],
                                                        c[('df',2,1)],
                                                        c[('df',2,2)],
                                                        c[('a',1,1)],
                                                        c[('a',2,2)],
                                                        c[('r',1)],
                                                        c[('r',2)],
                                                        c[('H',1)],
                                                        c[('dH',1,0)],
                                                        c[('H',2)],
                                                        c[('dH',2,0)])
        elif  self.nd==3:
            self.TwophaseNavierStokes_LS_SO_3D_Evaluate(self.eps,
                                                        self.rho_0,
                                                        self.nu_0,
                                                        self.rho_1,
                                                        self.nu_1,
                                                        self.g,
                                                        phi,
                                                        c[('u',0)],
                                                        c[('grad(u)',0)],
                                                        c[('u',1)],
                                                        c[('u',2)],
                                                        c[('u',3)],
                                                        c[('m',1)],
                                                        c[('dm',1,1)],
                                                        c[('m',2)],
                                                        c[('dm',2,2)],
                                                        c[('m',3)],
                                                        c[('dm',3,3)],
                                                        c[('f',0)],
                                                        c[('df',0,1)],
                                                        c[('df',0,2)],
                                                        c[('df',0,3)],
                                                        c[('f',1)],
                                                        c[('df',1,1)],
                                                        c[('df',1,2)],
                                                        c[('df',1,3)],
                                                        c[('f',2)],
                                                        c[('df',2,1)],
                                                        c[('df',2,2)],
                                                        c[('df',2,3)],
                                                        c[('f',3)],
                                                        c[('df',3,1)],
                                                        c[('df',3,2)],
                                                        c[('df',3,3)],
                                                        c[('a',1,1)],
                                                        c[('a',2,2)],
                                                        c[('a',3,3)],
                                                        c[('r',1)],
                                                        c[('r',2)],
                                                        c[('r',3)],
                                                        c[('H',0)],
                                                        c[('dH',1,0)],
                                                        c[('H',2)],
                                                        c[('dH',2,0)],
                                                        c[('H',3)],
                                                        c[('dH',3,0)])

##\brief Two-phase, Incompressible Navier-Stokes equations (level-set formulation)
#
#The equations are
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \mathbf{v} }{\partial t} + \nabla \cdot \left(\mathbf{v}\otimes \mathbf{v} - \nu \Delta \mathbf{f} \right) +\frac{\nabla p}{\rho} - \mathbf{g} &=& 0
#\f}
#
#where \f$\mathbf{v}\f$ is the velocity, \f$p\f$ is the pressure, \f$\nu\f$ is the kinematic viscosity, \f$\rho\f$ is the density, and \f$\mathbf{g}\f$ is the gravitational acceleration.
#
#The variable viscosity and density are given by
#
#\f{eqnarray*}
#\rho &=& \rho_0 (1-H) + \rho_1 H \\
#\nu  &=& \nu_0 (1-H) + \nu_1 H \\
#\f}
#
#with
#
#\f[
# H = \left\{ \begin{array}{lr}
# 0 & \phi \leq - \epsilon \\
# \frac{1}{2}\left(1+\frac{\phi}{\epsilon} + \frac{1}{\pi}\sin(\frac{\pi \phi}{\epsilon})\right) & -\epsilon < \phi < \epsilon \\
# 1 & \phi \geq \epsilon
#\end{array} \right.
#\f]
#
#The level set function, \f$\phi\f$, is provided from some other model (e.g. proteus::TransportCoeffcieints::NCLevelSetCoefficients) that is solved seperately (via operator splitting).
#
from .ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_2D_Evaluate
from .ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_3D_Evaluate
from .ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_2D_Evaluate_sd
from .ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_3D_Evaluate_sd
class TwophaseNavierStokes_ST_LS_SO(TC_base):
    """
    The coefficients for two incompresslble fluids governed by the Navier-Stokes equations and separated by a sharp interface represented by a level set function
    """
    def __init__(self,
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2,nu_0=1.004e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=None,
                 KN_model=None,
                 epsFact_density=None,
                 stokes=False,
                 sd=True,
                 movingDomain=False,
                 useRBLES=0.0):
        self.useRBLES=useRBLES
        self.sd=sd
        if epsFact_density is not None:
            self.epsFact_density = epsFact_density
        else:
            self.epsFact_density = epsFact
        self.stokes=stokes
        self.LS_model=LS_model
        self.KN_model=KN_model
        self.epsFact=epsFact
        self.eps=None
        self.sigma=sigma
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        #cek for debugging using single phase test problems
        self.rho=rho_0
        self.nu=nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.g = numpy.array(g)
        self.nd=nd
        #VRANS
        self.linearDragFactor    = 0.0
        self.nonlinearDragFactor = 0.0
        #end VRANS
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            mass= {1:{1:'linear'},
                   2:{2:'linear'}}
            advection = {0:{0:'linear',
                            1:'linear',
                            2:'linear'},
                         1:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear'},
                         2:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear'}}
            diffusion  = {1:{1:{1:'constant'},2:{2:'constant'}},
                          2:{2:{2:'constant'},1:{1:'constant'}}}
            sdInfo  = {(1,1):(numpy.array([0,1,2],dtype='i'),
                             numpy.array([0,1],dtype='i')),
                       (1,2):(numpy.array([0,0,1],dtype='i'),
                              numpy.array([0],dtype='i')),
                       (2,2):(numpy.array([0,1,2],dtype='i'),
                              numpy.array([0,1],dtype='i')),
                       (2,1):(numpy.array([0,1,1],dtype='i'),
                              numpy.array([1],dtype='i'))}
            potential= {1:{1:'u'},
                        2:{2:'u'}}
            reaction = {0:{0:'constant'},
                        1:{1:'nonlinear',2:'nonlinear'},
                        2:{1:'nonlinear',2:'nonlinear'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion = sd,
                             movingDomain=movingDomain)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            mass = {1:{1:'linear'},
                    2:{2:'linear'},
                    3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'},
                         1:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         2:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         3:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'}}
            diffusion = {1:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}},
                         2:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}},
                         3:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}}}
            sdInfo={}
            sdInfo  = {(1,1):(numpy.array([0,1,2,3],dtype='i'),numpy.array([0,1,2],dtype='i')),
                       (1,2):(numpy.array([0,0,1,1],dtype='i'),numpy.array([0],dtype='i')),
                       (1,3):(numpy.array([0,0,0,1],dtype='i'),numpy.array([0],dtype='i')),
                       (2,1):(numpy.array([0,1,1,1],dtype='i'),numpy.array([1],dtype='i')),
                       (2,2):(numpy.array([0,1,2,3],dtype='i'),numpy.array([0,1,2],dtype='i')),
                       (2,3):(numpy.array([0,0,0,1],dtype='i'),numpy.array([1],dtype='i')),
                       (3,1):(numpy.array([0,1,1,1],dtype='i'),numpy.array([2],dtype='i')),
                       (3,2):(numpy.array([0,0,1,1],dtype='i'),numpy.array([2],dtype='i')),
                       (3,3):(numpy.array([0,1,2,3],dtype='i'),numpy.array([0,1,2],dtype='i'))}
            potential= {1:{1:'u'},
                        2:{2:'u'},
                        3:{3:'u'}}
            reaction = {0:{0:'constant'},#added for Lin, Liu wave forcing,
                        1:{1:'nonlinear',2:'nonlinear',3:'nonlinear'},
                        2:{1:'nonlinear',2:'nonlinear',3:'nonlinear'},
                        3:{1:'nonlinear',2:'nonlinear',3:'nonlinear'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion = sd,
                             movingDomain=movingDomain)
            self.vectorComponents=[1,2,3]

    def attachModels(self,modelList):
        #level set
        self.model = modelList[0]
        if self.LS_model is not None:
            self.q_phi = modelList[self.LS_model].q[('u',0)]
            if ('u',0) in modelList[self.LS_model].ebq:
                self.ebq_phi = modelList[self.LS_model].ebq[('u',0)]
            else:
                self.ebq_phi = None
            self.ebqe_phi = modelList[self.LS_model].ebqe[('u',0)]
            #normal
            self.q_n = modelList[self.LS_model].q[('grad(u)',0)]
            if ('grad(u)',0) in modelList[self.LS_model].ebq:
                self.ebq_n = modelList[self.LS_model].ebq[('grad(u)',0)]
            else:
                self.ebq_n   = None
            self.ebqe_n    = modelList[self.LS_model].ebqe[('grad(u)',0)]
        #curvature
        if self.KN_model is not None:
            self.q_kappa    = modelList[self.KN_model].q[('u',0)]
            self.ebqe_kappa = modelList[self.KN_model].ebqe[('u',0)]
            if ('u',0) in modelList[self.KN_model].ebq:
                self.ebq_kappa = modelList[self.KN_model].ebq[('u',0)]
            else:
                self.ebq_kappa = None
    def initializeMesh(self,mesh):
        #cek we eventually need to use the local element diameter
        self.eps_density = self.epsFact_density*mesh.h
        self.eps_viscosity = self.epsFact*mesh.h
    #initialize so it can run as single phase
    def initializeElementQuadrature(self,t,cq):
        if self.LS_model is None:
            self.q_phi = -numpy.ones(cq[('u',1)].shape,'d')
            self.q_n = -numpy.ones(cq[('velocity',0)].shape,'d')
        if self.KN_model is None:
            self.q_kappa = -numpy.zeros(cq[('u',1)].shape,'d')
        #VRANS
        self.q_porosity = numpy.ones(cq[('u',1)].shape,'d')
        self.q_meanGrain= numpy.ones(cq[('u',1)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        if self.LS_model is None:
            self.ebq_phi = -numpy.ones(cebq[('u',1)].shape,'d')
            self.ebq_n = -numpy.ones(cebq[('velocity',0)].shape,'d')
        if self.KN_model is None:
            self.ebq_kappa = -numpy.zeros(cebq[('u',1)].shape,'d')
        #VRANS
        self.ebq_porosity = numpy.ones(cebq[('u',1)].shape,'d')
        self.ebq_meanGrain= numpy.ones(cebq[('u',1)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        if self.LS_model is None:
            self.ebqe_phi = -numpy.ones(cebqe[('u',1)].shape,'d')
            self.ebqe_n = -numpy.ones(cebqe[('velocity',0)].shape,'d')
        if self.KN_model is None:
            self.ebqe_kappa = -numpy.zeros(cebqe[('u',1)].shape,'d')
        #VRANS
        self.ebqe_porosity = numpy.ones(cebqe[('u',1)].shape,'d')
        self.ebqe_meanGrain = numpy.ones(cebqe[('u',1)].shape,'d')
    def updateToMovingDomain(self,t,c):
        from . import cfemIntegrals
        assert(self.movingDomain)
        if self.movingDomain:
            cfemIntegrals.update_f_movingDomain_constantMass(c['xt'],c[('f',0)])
            cfemIntegrals.update_f_movingDomain(c['xt'],c[('m',1)],c[('f',1)])
            cfemIntegrals.update_df_movingDomain(c['xt'],c[('dm',1,1)],c[('df',1,1)])
            cfemIntegrals.update_f_movingDomain(c['xt'],c[('m',2)],c[('f',2)])
            cfemIntegrals.update_df_movingDomain(c['xt'],c[('dm',2,2)],c[('df',2,2)])
            if self.nd == 3:
                cfemIntegrals.update_f_movingDomain(c['xt'],c[('m',3)],c[('f',3)])
                cfemIntegrals.update_df_movingDomain(c['xt'],c[('dm',3,3)],c[('df',3,3)])
    def evaluate(self,t,c):
        import math
        #self.rho_0 = 1000.0*self.rho_1
        #self.nu_0 = 0.1*self.nu_1
        if c[('u',0)].shape == self.q_phi.shape:
            phi = self.q_phi
            n   = self.q_n
            kappa = self.q_kappa
            #slopeAngle=0.1*math.pi/2.0#math.pi/4.0
            #surfaceNormal = [-sin(slopeAngle),cos(slopeAngle)]
            #waterLevel=0.5
            #for eN in range(phi.shape[0]):
            #   for k in range(phi.shape[1]):
            #       phi[eN,k] = (c['x'][eN,k,0] - 0.5)*surfaceNormal[0]+(c['x'][eN,k,1] - waterLevel)*surfaceNormal[1]
        elif c[('u',0)].shape == self.ebqe_phi.shape:
            phi   = self.ebqe_phi
            n     = self.ebqe_n
            kappa = self.ebqe_kappa
        else:
            phi   = self.ebq_phi
            n     = self.ebq_n
            kappa = self.ebq_kappa
        if self.nd==2:
            if self.sd:
                TwophaseNavierStokes_ST_LS_SO_2D_Evaluate_sd(self.eps_density,
                                                             self.eps_viscosity,
                                                             self.sigma,
                                                             self.rho_0,
                                                             self.nu_0,
                                                             self.rho_1,
                                                             self.nu_1,
                                                             self.g,
                                                             phi,
                                                             n,
                                                             kappa,
                                                             c[('u',0)],
                                                             c[('grad(u)',0)],
                                                             c[('u',1)],
                                                             c[('u',2)],
                                                             c[('m',1)],
                                                             c[('dm',1,1)],
                                                             c[('m',2)],
                                                             c[('dm',2,2)],
                                                             c[('f',0)],
                                                             c[('df',0,1)],
                                                             c[('df',0,2)],
                                                             c[('f',1)],
                                                             c[('df',1,1)],
                                                             c[('df',1,2)],
                                                             c[('f',2)],
                                                             c[('df',2,1)],
                                                             c[('df',2,2)],
                                                             c[('a',1,1)],
                                                             c[('a',2,2)],
                                                             c[('a',1,2)],
                                                             c[('a',2,1)],
                                                             c[('r',1)],
                                                             c[('r',2)],
                                                             c[('H',1)],
                                                             c[('dH',1,0)],
                                                             c[('H',2)],
                                                             c[('dH',2,0)])
            else:
                TwophaseNavierStokes_ST_LS_SO_2D_Evaluate(self.eps_density,
                                                          self.eps_viscosity,
                                                          self.sigma,
                                                          self.rho_0,
                                                          self.nu_0,
                                                          self.rho_1,
                                                          self.nu_1,
                                                          self.g,
                                                          phi,
                                                          n,
                                                          kappa,
                                                          c[('u',0)],
                                                          c[('grad(u)',0)],
                                                          c[('u',1)],
                                                          c[('u',2)],
                                                          c[('m',1)],
                                                          c[('dm',1,1)],
                                                          c[('m',2)],
                                                          c[('dm',2,2)],
                                                          c[('f',0)],
                                                          c[('df',0,1)],
                                                          c[('df',0,2)],
                                                          c[('f',1)],
                                                          c[('df',1,1)],
                                                          c[('df',1,2)],
                                                          c[('f',2)],
                                                          c[('df',2,1)],
                                                          c[('df',2,2)],
                                                          c[('a',1,1)],
                                                          c[('a',2,2)],
                                                          c[('a',1,2)],
                                                          c[('a',2,1)],
                                                          c[('r',1)],
                                                          c[('r',2)],
                                                          c[('H',1)],
                                                          c[('dH',1,0)],
                                                          c[('H',2)],
                                                          c[('dH',2,0)])
            if self.stokes:
                c[('f',1)].flat[:] = 0.0
                c[('df',1,1)].flat[:] = 0.0
                c[('df',1,2)].flat[:] = 0.0
                c[('f',2)].flat[:] = 0.0
                c[('df',2,1)].flat[:] = 0.0
                c[('df',2,2)].flat[:] = 0.0
        elif  self.nd==3:
            if self.sd:
                TwophaseNavierStokes_ST_LS_SO_3D_Evaluate_sd(self.eps_density,
                                                             self.eps_viscosity,
                                                             self.sigma,
                                                             self.rho_0,
                                                             self.nu_0,
                                                             self.rho_1,
                                                             self.nu_1,
                                                             self.g,
                                                             phi,
                                                             n,
                                                             kappa,
                                                             c[('u',0)],
                                                             c[('grad(u)',0)],
                                                             c[('u',1)],
                                                             c[('u',2)],
                                                             c[('u',3)],
                                                             c[('m',1)],
                                                             c[('dm',1,1)],
                                                             c[('m',2)],
                                                             c[('dm',2,2)],
                                                             c[('m',3)],
                                                             c[('dm',3,3)],
                                                             c[('f',0)],
                                                             c[('df',0,1)],
                                                             c[('df',0,2)],
                                                             c[('df',0,3)],
                                                             c[('f',1)],
                                                             c[('df',1,1)],
                                                             c[('df',1,2)],
                                                             c[('df',1,3)],
                                                             c[('f',2)],
                                                             c[('df',2,1)],
                                                             c[('df',2,2)],
                                                             c[('df',2,3)],
                                                             c[('f',3)],
                                                             c[('df',3,1)],
                                                             c[('df',3,2)],
                                                             c[('df',3,3)],
                                                             c[('a',1,1)],
                                                             c[('a',2,2)],
                                                             c[('a',3,3)],
                                                             c[('a',1,2)],
                                                             c[('a',1,3)],
                                                             c[('a',2,1)],
                                                             c[('a',2,3)],
                                                             c[('a',3,1)],
                                                             c[('a',3,2)],
                                                             c[('r',1)],
                                                             c[('r',2)],
                                                             c[('r',3)],
                                                             c[('H',1)],
                                                             c[('dH',1,0)],
                                                             c[('H',2)],
                                                             c[('dH',2,0)],
                                                             c[('H',3)],
                                                             c[('dH',3,0)])
            else:
                TwophaseNavierStokes_ST_LS_SO_3D_Evaluate(self.eps_density,
                                                          self.eps_viscosity,
                                                          self.sigma,
                                                          self.rho_0,
                                                          self.nu_0,
                                                          self.rho_1,
                                                          self.nu_1,
                                                          self.g,
                                                          phi,
                                                          n,
                                                          kappa,
                                                          c[('u',0)],
                                                          c[('grad(u)',0)],
                                                          c[('u',1)],
                                                          c[('u',2)],
                                                          c[('u',3)],
                                                          c[('m',1)],
                                                          c[('dm',1,1)],
                                                          c[('m',2)],
                                                          c[('dm',2,2)],
                                                          c[('m',3)],
                                                          c[('dm',3,3)],
                                                          c[('f',0)],
                                                          c[('df',0,1)],
                                                          c[('df',0,2)],
                                                          c[('df',0,3)],
                                                          c[('f',1)],
                                                          c[('df',1,1)],
                                                          c[('df',1,2)],
                                                          c[('df',1,3)],
                                                          c[('f',2)],
                                                          c[('df',2,1)],
                                                          c[('df',2,2)],
                                                          c[('df',2,3)],
                                                          c[('f',3)],
                                                          c[('df',3,1)],
                                                          c[('df',3,2)],
                                                          c[('df',3,3)],
                                                          c[('a',1,1)],
                                                          c[('a',2,2)],
                                                          c[('a',3,3)],
                                                          c[('a',1,2)],
                                                          c[('a',1,3)],
                                                          c[('a',2,1)],
                                                          c[('a',2,3)],
                                                          c[('a',3,1)],
                                                          c[('a',3,2)],
                                                          c[('r',1)],
                                                          c[('r',2)],
                                                          c[('r',3)],
                                                          c[('H',1)],
                                                          c[('dH',1,0)],
                                                          c[('H',2)],
                                                          c[('dH',2,0)],
                                                          c[('H',3)],
                                                          c[('dH',3,0)])
            if self.stokes:
                c[('f',1)].flat[:] = 0.0
                c[('df',1,1)].flat[:] = 0.0
                c[('df',1,2)].flat[:] = 0.0
                c[('df',1,3)].flat[:] = 0.0
                c[('f',2)].flat[:] = 0.0
                c[('df',2,1)].flat[:] = 0.0
                c[('df',2,2)].flat[:] = 0.0
                c[('df',2,3)].flat[:] = 0.0
                c[('f',3)].flat[:] = 0.0
                c[('df',3,1)].flat[:] = 0.0
                c[('df',3,2)].flat[:] = 0.0
                c[('df',3,3)].flat[:] = 0.0
            
from .ctransportCoefficients import ThreephaseNavierStokes_ST_LS_SO_2D_Evaluate
from .ctransportCoefficients import ThreephaseNavierStokes_ST_LS_SO_3D_Evaluate
class ThreephaseNavierStokes_ST_LS_SO(TC_base):
    """
    The coefficients for two incompresslble fluids governed by the Navier-Stokes equations and separated by a sharp interface represented by a level set function
    """
    def __init__(self,
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2,nu_0=1.004e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 rho_s=998.2,nu_s=1.004e-6,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=None,
                 KN_model=None,
                 epsFact_density=None,
                 defaultSolidProfile=None,
                 defaultFluidProfile=None,
                 stokes=False,
                 ptsFile=None,
                 boundaryPenaltyCoef=1.0,
                 volumePenaltyCoef=1000.0):
        if epsFact_density is not None:
            self.epsFact_density = epsFact_density
        else:
            self.epsFact_density = epsFact
        self.stokes=stokes
        self.LS_model=LS_model
        self.KN_model=KN_model
        self.epsFact=epsFact
        self.eps=None
        self.sigma=sigma
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        self.defaultSolidProfile=defaultSolidProfile #possible analytical representation for solid boundary
        self.defaultFluidProfile=defaultFluidProfile #possible analytical representation for air-water boundary
        #cek for debugging using single phase test problems
        self.rho=rho_0
        self.nu=nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.rho_s = rho_s
        self.nu_s = nu_s
        self.g = numpy.array(g)
        self.nd=nd
        self.boundaryPenaltyCoef=boundaryPenaltyCoef
        self.volumePenaltyCoef=volumePenaltyCoef
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            mass= {1:{1:'linear'},
                   2:{2:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear'},
                         1:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear'},
                         2:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear'}}
            diffusion  = {1:{1:{1:'constant'},2:{2:'constant'}},
                          2:{2:{2:'constant'},1:{1:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'}}
            reaction = {1:{1:'nonlinear',2:'nonlinear'},
                        2:{1:'nonlinear',2:'nonlinear'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            mass = {1:{1:'linear'},
                    2:{2:'linear'},
                    3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'},
                         1:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         2:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         3:{0:'nonlinear',
                            1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'}}
            diffusion = {1:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}},
                         2:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}},
                         3:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'},
                        3:{3:'u'}}
            reaction = {1:{1:'nonlinear',2:'nonlinear',3:'nonlinear'},
                        2:{1:'nonlinear',2:'nonlinear',3:'nonlinear'},
                        3:{1:'nonlinear',2:'nonlinear',3:'nonlinear'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
        if ptsFile is not None:
            self.ptsFile=ptsFile
            self.readPTS(self.ptsFile)
    def readPTS(self,ptsFile):
        f = open(self.ptsFile,'r')
        lines = f.readlines()
        nParticles=int(lines[4].split()[0])
        self.centers = numpy.zeros((nParticles,3),'d')
        self.radii = numpy.zeros((nParticles,),'d')
        words = lines[6].split()
        max_x = min_x = float(words[2])
        max_y = min_y = float(words[2])
        max_z = min_z = float(words[2])
        for pN,line in enumerate(lines[5:5+nParticles]):
            words=line.split()
            x = float(words[2])
            y = float(words[3])
            z = float(words[4])
            self.centers[pN][0] = x
            self.centers[pN][1] = y
            self.centers[pN][2] = z
            max_x = max(x,max_x); min_x = min(x,min_x)
            max_y = max(y,max_y); min_y = min(y,min_y)
            max_z = max(z,max_z); min_z = min(z,min_z)
            self.radii[pN] = float(words[5])
        #print (max_x,max_y,max_z)
        #print (min_x,min_y,min_z)
    def signedDistance(self,x):
        import math
        sd=0.0
        ad=10000000000000.0
        for c,r in zip(self.centers,self.radii):
#             R = math.sqrt((x[0]-c[0])**2 +
#                           (x[1]-c[1])**2 +
#                           (x[2]-c[2])**2)
            R = math.sqrt((3.0-c[0])**2 +
                          (x[0]-c[1])**2 +
                          (x[1]-c[2])**2)
            sdi = R - r
            adi = abs(sdi)
            if adi < ad:
                sd = sdi
                ad = adi
        #print "==================",sd
        return sd
    def attachModels(self,modelList):
        #level set
        if self.LS_model is not None:
            self.q_phi = modelList[self.LS_model].q[('u',0)]
            if ('u',0) in modelList[self.LS_model].ebq:
                self.ebq_phi = modelList[self.LS_model].ebq[('u',0)]
            else:
                self.ebq_phi = None
            self.ebqe_phi = modelList[self.LS_model].ebqe[('u',0)]
            #normal
            self.q_n = modelList[self.LS_model].q[('grad(u)',0)]
            if ('grad(u)',0) in modelList[self.LS_model].ebq:
                self.ebq_n = modelList[self.LS_model].ebq[('grad(u)',0)]
            else:
                self.ebq_n   = None
            self.ebqe_n    = modelList[self.LS_model].ebqe[('grad(u)',0)]
        #curvature
        if self.KN_model is not None:
            self.q_kappa    = modelList[self.KN_model].q[('u',0)]
            self.ebqe_kappa = modelList[self.KN_model].ebqe[('u',0)]
            if ('u',0) in modelList[self.KN_model].ebq:
                self.ebq_kappa = modelList[self.KN_model].ebq[('u',0)]
            else:
                self.ebq_kappa = None
    def initializeMesh(self,mesh):
        self.eps_density = self.epsFact_density*mesh.h
        self.eps_viscosity = self.epsFact*mesh.h
    #initialize so it can run as single phase
    def initializeElementQuadrature(self,t,cq):
        import math
        self.q_phi = numpy.ones(cq[('u',0)].shape,'d')
        self.q_n = numpy.ones(cq[('f',0)].shape,'d')
        self.q_kappa = numpy.zeros(cq[('u',0)].shape,'d')
        self.q_phi_s = numpy.ones(cq[('u',0)].shape,'d')
        self.q_n_s = numpy.ones(cq[('f',0)].shape,'d')
        if self.defaultSolidProfile is None:
            if self.nd==2:
                for i in range(len(cq[('u',0)].flat)):
                    x = cq['x'].flat[i*3+0]
                    y = cq['x'].flat[i*3+1]
                    u =x-0.5
                    v =y-0.5
                    self.q_phi_s.flat[i] = math.sqrt(u**2 + v**2) - 0.25/2.0
                    self.q_n_s.flat[2*i+0] = u/math.sqrt(u**2+v**2)
                    self.q_n_s.flat[2*i+1] = v/math.sqrt(u**2+v**2)
        else:
            try:
                self.defaultSolidProfile(t,cq['x'],self.q_phi_s,self.q_n_s)
            except TypeError:
                for i in range(len(cq[('u',0)].flat)):
                    phi,n = self.defaultSolidProfile(cq['x'].flat[3*i:3*(i+1)])
                    self.q_phi_s.flat[i] = phi
                    self.q_n_s.flat[self.nd*i:self.nd*(i+1)] = n[:]
        if self.defaultFluidProfile is not None:
            try:
                self.defaultFluidProfile(t,cq['x'],self.q_phi,self.q_n)
            except TypeError:
                for i in range(len(cq[('u',0)].flat)):
                    phi,n = self.defaultFluidProfile(cq['x'].flat[3*i:3*(i+1)])
                    self.q_phi.flat[i] = phi
                    self.q_n.flat[self.nd*i:self.nd*(i+1)] = n[:]
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        import math
        self.ebq_phi = numpy.ones(cebq[('u',0)].shape,'d')
        self.ebq_n = numpy.ones(cebq[('f',0)].shape,'d')
        self.ebq_kappa = numpy.zeros(cebq[('u',0)].shape,'d')
        self.ebq_phi_s = numpy.ones(cebq[('u',0)].shape,'d')
        self.ebq_n_s = numpy.ones(cebq[('f',0)].shape,'d')
        if self.defaultSolidProfile is None:
            if self.nd==2:
                for i in range(len(cebq[('u',0)].flat)):
                    x = cebq['x'].flat[i*3+0]
                    y = cebq['x'].flat[i*3+1]
                    u =x-0.5
                    v =y-0.5
                    self.ebq_phi_s.flat[i] = math.sqrt(u**2 + v**2) - 0.25/2.0
                    self.ebq_n_s.flat[2*i+0] = u/math.sqrt(u**2+v**2)
                    self.ebq_n_s.flat[2*i+1] = v/math.sqrt(u**2+v**2)
        else:
            try:
                self.defaultSolidProfile(t,cebq['x'],self.ebq_phi_s,self.ebq_n_s)
            except TypeError:
                for i in range(len(cebq[('u',0)].flat)):
                    phi,n = self.defaultSolidProfile(cebq['x'].flat[3*i:3*(i+1)])
                    self.ebq_phi_s.flat[i] = phi
                    self.ebq_n_s.flat[self.nd*i:self.nd*(i+1)] = n[:]
        if self.defaultFluidProfile is not None:
            try:
                self.defaultFluidProfile(t,cebq['x'],self.ebq_phi,self.ebq_n)
            except TypeError:
                for i in range(len(cebq[('u',0)].flat)):
                    phi,n = self.defaultFluidProfile(cebq['x'].flat[3*i:3*(i+1)])
                    self.ebq_phi.flat[i] = phi
                    self.ebq_n.flat[self.nd*i:self.nd*(i+1)] = n[:]
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        import math
        self.ebqe_phi = numpy.ones(cebqe[('u',0)].shape,'d')
        self.ebqe_n = numpy.ones(cebqe[('f',0)].shape,'d')
        self.ebqe_kappa = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.ebqe_phi_s = numpy.ones(cebqe[('u',0)].shape,'d')
        self.ebqe_n_s = numpy.ones(cebqe[('f',0)].shape,'d')
        if self.defaultSolidProfile is None:
            if self.nd==2:
                for i in range(len(cebqe[('u',0)].flat)):
                    x = cebqe['x'].flat[i*3+0]
                    y = cebqe['x'].flat[i*3+1]
                    u =x-0.5
                    v =y-0.5
                    self.ebqe_phi_s.flat[i] = math.sqrt(u**2 + v**2) - 0.25/2.0
                    self.ebqe_n_s.flat[2*i+0] = u/math.sqrt(u**2+v**2)
                    self.ebqe_n_s.flat[2*i+1] = v/math.sqrt(u**2+v**2)
        else:
            try:
                self.defaultSolidProfile(t,cebqe['x'],self.ebqe_phi_s,self.ebqe_n_s)
            except TypeError:
                for i in range(len(cebqe[('u',0)].flat)):
                    phi,n = self.defaultSolidProfile(cebqe['x'].flat[3*i:3*(i+1)])
                    self.ebqe_phi_s.flat[i] = phi
                    self.ebqe_n_s.flat[self.nd*i:self.nd*(i+1)] = n[:]
        if self.defaultFluidProfile is None:
            try:
                self.defaultFluidProfile(t,cebqe['x'],self.ebqe_phi,self.ebqe_n)
            except TypeError:
                for i in range(len(cebqe[('u',0)].flat)):
                    phi,n = self.defaultFluidProfile(cebqe['x'].flat[3*i:3*(i+1)])
                    self.ebqe_phi.flat[i] = phi
                    self.ebqe_n.flat[self.nd*i:self.nd*(i+1)] = n[:]
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_phi.shape:
            phi = self.q_phi
            n   = self.q_n
            kappa = self.q_kappa
            phi_s = self.q_phi_s
            n_s   = self.q_n_s
        elif c[('u',0)].shape == self.ebqe_phi.shape:
            phi   = self.ebqe_phi
            n     = self.ebqe_n
            kappa = self.ebqe_kappa
            phi_s   = self.ebqe_phi_s
            n_s     = self.ebqe_n_s
        else:
            phi   = self.ebq_phi
            n     = self.ebq_n
            kappa = self.ebq_kappa
            phi_s   = self.ebq_phi_s
            n_s     = self.ebq_n_s
        if self.nd==2:
            ThreephaseNavierStokes_ST_LS_SO_2D_Evaluate(self.boundaryPenaltyCoef,
                                                        self.volumePenaltyCoef,
                                                        self.eps_density,
                                                        self.eps_viscosity,
                                                        self.sigma,
                                                        self.rho_0,
                                                        self.nu_0,
                                                        self.rho_1,
                                                        self.nu_1,
                                                        self.rho_s,
                                                        self.nu_s,
                                                        self.g,
                                                        phi,
                                                        n,
                                                        kappa,
                                                        phi_s,
                                                        n_s,
                                                        c[('u',0)],
                                                        c[('grad(u)',0)],
                                                        c[('u',1)],
                                                        c[('u',2)],
                                                        c[('m',1)],
                                                        c[('dm',1,1)],
                                                        c[('m',2)],
                                                        c[('dm',2,2)],
                                                        c[('f',0)],
                                                        c[('df',0,1)],
                                                        c[('df',0,2)],
                                                        c[('f',1)],
                                                        c[('df',1,1)],
                                                        c[('df',1,2)],
                                                        c[('f',2)],
                                                        c[('df',2,1)],
                                                        c[('df',2,2)],
                                                        c[('a',1,1)],
                                                        c[('a',2,2)],
                                                        c[('a',1,2)],
                                                        c[('a',2,1)],
                                                        c[('r',1)],
                                                        c[('dr',1,1)],
                                                        c[('dr',1,2)],
                                                        c[('r',2)],
                                                        c[('dr',2,1)],
                                                        c[('dr',2,2)],
                                                        c[('H',1)],
                                                        c[('dH',1,0)],
                                                        c[('H',2)],
                                                        c[('dH',2,0)])
            if self.stokes:
                c[('f',1)].flat[:] = 0.0
                c[('df',1,1)].flat[:] = 0.0
                c[('df',1,2)].flat[:] = 0.0
                c[('f',2)].flat[:] = 0.0
                c[('df',2,1)].flat[:] = 0.0
                c[('df',2,2)].flat[:] = 0.0
        elif  self.nd==3:
            ThreephaseNavierStokes_ST_LS_SO_3D_Evaluate(self.boundaryPenaltyCoef,
                                                        self.volumePenaltyCoef,
                                                        self.eps_density,
                                                        self.eps_viscosity,
                                                        self.sigma,
                                                        self.rho_0,
                                                        self.nu_0,
                                                        self.rho_1,
                                                        self.nu_1,
                                                        self.rho_s,
                                                        self.nu_s,
                                                        self.g,
                                                        phi,
                                                        n,
                                                        kappa,
                                                        phi_s,
                                                        n_s,
                                                        c[('u',0)],
                                                        c[('grad(u)',0)],
                                                        c[('u',1)],
                                                        c[('u',2)],
                                                        c[('u',3)],
                                                        c[('m',1)],
                                                        c[('dm',1,1)],
                                                        c[('m',2)],
                                                        c[('dm',2,2)],
                                                        c[('m',3)],
                                                        c[('dm',3,3)],
                                                        c[('f',0)],
                                                        c[('df',0,1)],
                                                        c[('df',0,2)],
                                                        c[('df',0,3)],
                                                        c[('f',1)],
                                                        c[('df',1,1)],
                                                        c[('df',1,2)],
                                                        c[('df',1,3)],
                                                        c[('f',2)],
                                                        c[('df',2,1)],
                                                        c[('df',2,2)],
                                                        c[('df',2,3)],
                                                        c[('f',3)],
                                                        c[('df',3,1)],
                                                        c[('df',3,2)],
                                                        c[('df',3,3)],
                                                        c[('a',1,1)],
                                                        c[('a',2,2)],
                                                        c[('a',3,3)],
                                                        c[('a',1,2)],
                                                        c[('a',1,3)],
                                                        c[('a',2,1)],
                                                        c[('a',2,3)],
                                                        c[('a',3,1)],
                                                        c[('a',3,2)],
                                                        c[('r',1)],
                                                        c[('dr',1,1)],
                                                        c[('dr',1,2)],
                                                        c[('dr',1,3)],
                                                        c[('r',2)],
                                                        c[('dr',2,1)],
                                                        c[('dr',2,2)],
                                                        c[('dr',2,3)],
                                                        c[('r',3)],
                                                        c[('dr',3,1)],
                                                        c[('dr',3,2)],
                                                        c[('dr',3,3)],
                                                        c[('H',1)],
                                                        c[('dH',1,0)],
                                                        c[('H',2)],
                                                        c[('dH',2,0)],
                                                        c[('H',3)],
                                                        c[('dH',3,0)])
            if self.stokes:
                c[('f',1)].flat[:] = 0.0
                c[('df',1,1)].flat[:] = 0.0
                c[('df',1,2)].flat[:] = 0.0
                c[('df',1,3)].flat[:] = 0.0
                c[('f',2)].flat[:] = 0.0
                c[('df',2,1)].flat[:] = 0.0
                c[('df',2,2)].flat[:] = 0.0
                c[('df',2,3)].flat[:] = 0.0
                c[('f',3)].flat[:] = 0.0
                c[('df',3,1)].flat[:] = 0.0
                c[('df',3,2)].flat[:] = 0.0
                c[('df',3,3)].flat[:] = 0.0
        c[('phis_viz')] = phi_s
        c[('phi_viz')]  = phi
##\brief Two-phase, Incompressible Navier-Stokes equations (level-set formulation)
#
#The equations are
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \mathbf{v} }{\partial t} + \nabla \cdot \left(- \nu \Delta \mathbf{f} \right) +\frac{\nabla p}{\rho} - \mathbf{g} &=& 0
#\f}
#
#where \f$\mathbf{v}\f$ is the velocity, \f$p\f$ is the pressure, \f$\nu\f$ is the kinematic viscosity, \f$\rho\f$ is the density, and \f$\mathbf{g}\f$ is the gravitational acceleration.
#
#The variable viscosity and density are given by
#
#\f{eqnarray*}
#\rho &=& \rho_0 (1-H) + \rho_1 H \\
#\nu  &=& \nu_0 (1-H) + \nu_1 H \\
#\f}
#
#with
#
#\f[
# H = \left\{ \begin{array}{lr}
# 0 & \phi \leq - \epsilon \\
# \frac{1}{2}\left(1+\frac{\phi}{\epsilon} + \frac{1}{\pi}\sin(\frac{\pi \phi}{\epsilon})\right) & -\epsilon < \phi < \epsilon \\
# 1 & \phi \geq \epsilon
#\end{array} \right.
#\f]
#
#The level set function, \f$\phi\f$, is provided from some other model that is solved separately (via operator splitting).
#
from .ctransportCoefficients import TwophaseStokes_LS_SO_2D_Evaluate
from .ctransportCoefficients import TwophaseStokes_LS_SO_3D_Evaluate
class TwophaseStokes_LS_SO(TC_base):
    """
    The coefficients for two incompresslble fluids governed by the Stokes equations and seperated by a sharp interface represented by a level set function
    """
    def __init__(self,
                 epsFact=1.5,
                 rho_0=998.2,nu_0=1.005e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=1,
                 steady=False):
        self.epsFact=epsFact
        self.eps=None
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            if not steady:
                mass={1:{1:'linear'},
                      2:{2:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}}}
            potential = {1:{1:'u'},
                         2:{2:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            if not steady:
                mass={1:{1:'linear'},
                      2:{2:'linear'},
                      3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}},
                         3:{3:{3:'constant'}}}
            potential = {1:{1:'u'},
                         2:{2:'u'},
                         3:{3:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'},
                        3:{3:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
        self.q_phi = None
        self.ebq_phi = None
        self.ebqe_phi = None
        self.levelSetModelIndex=LS_model
        self.dummyWaterLevel=0.5
    def attachModels(self,modelList):
        self.q_phi    = modelList[self.levelSetModelIndex].q[('u',0)]
        self.ebqe_phi = modelList[self.levelSetModelIndex].ebqe[('u',0)]
        if ('u',0) in modelList[self.levelSetModelIndex].ebq:
            self.ebq_phi = modelList[self.levelSetModelIndex].ebq[('u',0)]
    def initializeMesh(self,mesh):
        self.eps = self.epsFact*mesh.h
    def initializeElementQuadrature(self,t,cq):
        if self.levelSetModelIndex is None:
            self.q_phi = numpy.ones(cq[('u',0)].shape,'d')
#         for eN in range(cq['x'].shape[0]):
#             for q in range(cq['x'].shape[1]):
#                 if cq['x'][eN,q,1] <= self.dummyWaterLevel:
#                     self.q_phi[eN,q] = cq['x'][eN,q,1] -self.dummyWaterLevel
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        if self.levelSetModelIndex is None:
            self.ebq_phi = numpy.ones(cebq[('u',0)].shape,'d')
#         for eN in range(cebq['x'].shape[0]):
#             for ebN in range(cebq['x'].shape[1]):
#                 for q in range(cebq['x'].shape[2]):
#                     if cebq['x'][eN,ebN,q,1] <= self.dummyWaterLevel:
#                         self.ebq_phi[eN,q] = cebq['x'][eN,ebN,q,1]-self.dummyWaterLevel
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        if self.levelSetModelIndex is None:
            self.ebqe_phi = numpy.ones(cebqe[('u',0)].shape,'d')
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_phi.shape:
            phi = self.q_phi
        elif c[('u',0)].shape == self.ebqe_phi.shape:
            phi = self.ebqe_phi
        else:
            phi = self.ebq_phi
        if self.nd==2:
            TwophaseStokes_LS_SO_2D_Evaluate(self.eps,
                                             self.rho_0,
                                             self.nu_0,
                                             self.rho_1,
                                             self.nu_1,
                                             self.g,
                                             phi,
                                             c[('u',0)],
                                             c[('grad(u)',0)],
                                             c[('u',1)],
                                             c[('u',2)],
                                             c[('m',1)],
                                             c[('dm',1,1)],
                                             c[('m',2)],
                                             c[('dm',2,2)],
                                             c[('f',0)],
                                             c[('df',0,1)],
                                             c[('df',0,2)],
                                             c[('a',1,1)],
                                             c[('a',2,2)],
                                             c[('r',1)],
                                             c[('r',2)],
                                             c[('H',1)],
                                             c[('dH',1,0)],
                                             c[('H',2)],
                                             c[('dH',2,0)])

        elif self.nd==3:
            TwophaseStokes_LS_SO_3D_Evaluate(self.eps,
                                             self.rho_0,
                                             self.nu_0,
                                             self.rho_1,
                                             self.nu_1,
                                             self.g,
                                             phi,
                                             c[('u',0)],
                                             c[('grad(u)',0)],
                                             c[('u',1)],
                                             c[('u',2)],
                                             c[('u',3)],
                                             c[('m',1)],
                                             c[('dm',1,1)],
                                             c[('m',2)],
                                             c[('dm',2,2)],
                                             c[('m',3)],
                                             c[('dm',3,3)],
                                             c[('f',0)],
                                             c[('df',0,1)],
                                             c[('df',0,2)],
                                             c[('df',0,3)],
                                             c[('a',1,1)],
                                             c[('a',2,2)],
                                             c[('a',3,3)],
                                             c[('r',1)],
                                             c[('r',2)],
                                             c[('r',3)],
                                             c[('H',1)],
                                             c[('dH',1,0)],
                                             c[('H',2)],
                                             c[('dH',2,0)],
                                             c[('H',3)],
                                             c[('dH',3,0)])

##\brief Two-phase Navier-Stokes equations (volume-of-fluid formulation)
#
#The equations are formulated as
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \mathbf{v} }{\partial t} + \nabla \cdot \left(\mathbf{v}\otimes \mathbf{v} - \nu \Delta \mathbf{f} \right) +\frac{\nabla p}{\rho} - \mathbf{g} &=& 0
#\f}
#
#where the coefficients are given by
#
#\f{eqnarray*}
#\rho &=& \rho_0 (1-V) + \rho_1 V \\
#\nu  &=& \nu_0 (1-V) + \nu_1 V \\
#\f}
#
#where \f$V\f$ is the fraction of space occupied by the fluid, which
#is provided from some other model that is solved seperately (via operator splitting).
#
from .ctransportCoefficients import TwophaseNavierStokes_VOF_SO_2D_Evaluate
from .ctransportCoefficients import TwophaseNavierStokes_VOF_SO_3D_Evaluate
class TwophaseNavierStokes_VOF_SO(TC_base):
    """
    The coefficients for two incompresslble fluids governed by the Navier-Stokes equations and separated by a sharp interface represented by a volume of  fluid (volume fraction) function
    """
    def __init__(self,
                 epsFact=1.5,
                 rho_0=998.2,nu_0=1.004e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=1):
        self.LS_model=LS_model
        self.epsFact=epsFact
        self.eps=None
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            mass= {1:{1:'linear'},
                   2:{2:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear'}}
            diffusion  = {1:{1:{1:'constant'}},
                          2:{2:{2:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            mass = {1:{1:'linear'},
                    2:{2:'linear'},
                    3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         3:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}},
                         3:{3:{3:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'},
                        3:{3:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'},
                        3:{3:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
    def attachModels(self,modelList):
        self.q_vof = modelList[self.LS_model].q[('u',0)]
        self.ebqe_vof = modelList[self.LS_model].ebqe[('u',0)]
        self.ebq_vof  = None
        if ('u',0) in modelList[self.LS_model].ebq:
            self.ebq_vof = modelList[self.LS_model].ebq[('u',0)]

    def initializeMesh(self,mesh):
        self.eps = self.epsFact*mesh.h
    def initializeElementQuadrature(self,t,cq):
        #initialize so it can run without a flow model
        self.q_vof = numpy.ones(cq[('u',0)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.ebq_vof = numpy.ones(cebq[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.ebqe_vof = numpy.ones(cebqe[('u',0)].shape,'d')
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_vof.shape:
            vof = self.q_vof
        elif c[('u',0)].shape == self.ebqe_vof.shape:
            vof = self.ebqe_vof
        else:
            vof = self.ebq_vof
        if self.nd==2:
            TwophaseNavierStokes_VOF_SO_2D_Evaluate(self.eps,
                                                    self.rho_0,
                                                    self.nu_0,
                                                    self.rho_1,
                                                    self.nu_1,
                                                    self.g,
                                                    vof,
                                                    c[('u',0)],
                                                    c[('grad(u)',0)],
                                                    c[('u',1)],
                                                    c[('u',2)],
                                                    c[('m',1)],
                                                    c[('dm',1,1)],
                                                    c[('m',2)],
                                                    c[('dm',2,2)],
                                                    c[('f',0)],
                                                    c[('df',0,1)],
                                                    c[('df',0,2)],
                                                    c[('f',1)],
                                                    c[('df',1,1)],
                                                    c[('df',1,2)],
                                                    c[('f',2)],
                                                    c[('df',2,1)],
                                                    c[('df',2,2)],
                                                    c[('a',1,1)],
                                                    c[('a',2,2)],
                                                    c[('r',1)],
                                                    c[('r',2)],
                                                    c[('H',1)],
                                                    c[('dH',1,0)],
                                                    c[('H',2)],
                                                    c[('dH',2,0)])
        elif  self.nd==3:
            TwophaseNavierStokes_VOF_SO_3D_Evaluate(self.eps,
                                                    self.rho_0,
                                                    self.nu_0,
                                                    self.rho_1,
                                                    self.nu_1,
                                                    self.g,
                                                    vof,
                                                    c[('u',0)],
                                                    c[('grad(u)',0)],
                                                    c[('u',1)],
                                                    c[('u',2)],
                                                    c[('u',3)],
                                                    c[('m',1)],
                                                    c[('dm',1,1)],
                                                    c[('m',2)],
                                                    c[('dm',2,2)],
                                                    c[('m',3)],
                                                    c[('dm',3,3)],
                                                    c[('f',0)],
                                                    c[('df',0,1)],
                                                    c[('df',0,2)],
                                                    c[('df',0,3)],
                                                    c[('f',1)],
                                                    c[('df',1,1)],
                                                    c[('df',1,2)],
                                                    c[('df',1,3)],
                                                    c[('f',2)],
                                                    c[('df',2,1)],
                                                    c[('df',2,2)],
                                                    c[('df',2,3)],
                                                    c[('f',3)],
                                                    c[('df',3,1)],
                                                    c[('df',3,2)],
                                                    c[('df',3,3)],
                                                    c[('a',1,1)],
                                                    c[('a',2,2)],
                                                    c[('a',3,3)],
                                                    c[('r',1)],
                                                    c[('r',2)],
                                                    c[('r',3)],
                                                    c[('H',0)],
                                                    c[('dH',1,0)],
                                                    c[('H',2)],
                                                    c[('dH',2,0)],
                                                    c[('H',3)],
                                                    c[('dH',3,0)])

##\brief Two-phase Stokes equations (volume-of-fluid formulation)
#
#The equations are formulated as
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \mathbf{v} }{\partial t} + \nabla \cdot \left(- \nu \Delta \mathbf{f} \right) +\frac{\nabla p}{\rho} - \mathbf{g} &=& 0
#\f}
#
#where the coefficients are given by
#
#\f{eqnarray*}
#\rho &=& \rho_0 (1-V) + \rho_1 V \\
#\nu  &=& \nu_0 (1-V) + \nu_1 V \\
#\f}
#
#where \f$V\f$ is the fraction of space occupied by the fluid, which
#is provided from some other model that is solved seperately (via operator splitting).
#
from .ctransportCoefficients import TwophaseStokes_VOF_SO_2D_Evaluate
from .ctransportCoefficients import TwophaseStokes_VOF_SO_3D_Evaluate
class TwophaseStokes_VOF_SO(TC_base):
    """
    The coefficients for two incompresslble fluids governed by the Stokes equations and seperated by a sharp interface represented by a volume of fluid function
    """
    def __init__(self,
                 epsFact=1.5,
                 rho_0=998.2,nu_0=1.005e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,9.8],
                 nd=2,
                 LS_model=1,
                 steady=False):
        self.epsFact=epsFact
        self.eps=None
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            if not steady:
                mass={1:{1:'linear'},
                      2:{2:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}}}
            potential = {1:{1:'u'},
                         2:{2:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            if not steady:
                mass={1:{1:'linear'},
                      2:{2:'linear'},
                      3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'}}
            diffusion = {1:{1:{1:'constant'}},
                         2:{2:{2:'constant'}},
                         3:{3:{3:'constant'}}}
            potential = {1:{1:'u'},
                         2:{2:'u'},
                         3:{3:'u'}}
            reaction = {1:{1:'constant'},
                        2:{2:'constant'},
                        3:{3:'constant'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
        self.q_vof = None
        self.ebq_vof = None
        self.ebqe_vof = None
        self.levelSetModelIndex=1
        self.dummyWaterLevel=0.5
    def attachModels(self,modelList):
        self.q_vof = modelList[self.levelSetModelIndex].q[('u',0)]
        self.ebqe_vof = modelList[self.levelSetModelIndex].ebqe[('u',0)]
        if ('u',0) in modelList[self.levelSetModelIndex].ebq:
            self.ebq_vof = modelList[self.levelSetModelIndex].ebq[('u',0)]
    def initializeMesh(self,mesh):
        self.eps = self.epsFact*mesh.h
    def initializeElementQuadrature(self,t,cq):
        #initialize so it can run without a flow model
        self.q_vof = numpy.ones(cq[('u',0)].shape,'d')
        for eN in range(cq['x'].shape[0]):
            for q in range(cq['x'].shape[1]):
                if cq['x'][eN,q,1] <= self.dummyWaterLevel:
                    self.q_vof[eN,q] = 0.0
                else:
                    self.q_vof[eN,q] = 1.0
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.ebqe_vof = numpy.ones(cebqe[('u',0)].shape,'d')
        for ebNE in range(cebqe['x'].shape[0]):
            for q in range(cebqe['x'].shape[1]):
                if cebqe['x'][ebNE,q,1] <= self.dummyWaterLevel:
                    self.ebqe_vof[ebNE,q] = 0.0
                else:
                    self.ebqe_vof[ebNE,q] = 1.0
    def evaluate(self,t,c):
        if self.q_vof is None:
            vof = numpy.zeros(c[('u',0)].shape,'d')
        else:
            if c[('u',0)].shape == self.q_vof.shape:
                vof = self.q_vof
            elif c[('u',0)].shape == self.ebqe_vof.shape:
                vof = self.ebqe_vof
            else:
                vof = self.ebq_vof
        if self.nd==2:
            TwophaseStokes_VOF_SO_2D_Evaluate(self.eps,
                                              self.rho_0,
                                              self.nu_0,
                                              self.rho_1,
                                              self.nu_1,
                                              self.g,
                                              vof,
                                              c[('u',0)],
                                              c[('grad(u)',0)],
                                              c[('u',1)],
                                              c[('u',2)],
                                              c[('m',1)],
                                              c[('dm',1,1)],
                                              c[('m',2)],
                                              c[('dm',2,2)],
                                              c[('f',0)],
                                              c[('df',0,1)],
                                              c[('df',0,2)],
                                              c[('a',1,1)],
                                              c[('a',2,2)],
                                              c[('r',1)],
                                              c[('r',2)],
                                              c[('H',1)],
                                              c[('dH',1,0)],
                                              c[('H',2)],
                                              c[('dH',2,0)])

        elif self.nd==3:
            TwophaseStokes_VOF_SO_3D_Evaluate(self.eps,
                                              self.rho_0,
                                              self.nu_0,
                                              self.rho_1,
                                              self.nu_1,
                                              self.g,
                                              vof,
                                              c[('u',0)],
                                              c[('grad(u)',0)],
                                              c[('u',1)],
                                              c[('u',2)],
                                              c[('u',3)],
                                              c[('m',1)],
                                              c[('dm',1,1)],
                                              c[('m',2)],
                                              c[('dm',2,2)],
                                              c[('m',3)],
                                              c[('dm',3,3)],
                                              c[('f',0)],
                                              c[('df',0,1)],
                                              c[('df',0,2)],
                                              c[('df',0,3)],
                                              c[('a',1,1)],
                                              c[('a',2,2)],
                                              c[('a',3,3)],
                                              c[('r',1)],
                                              c[('r',2)],
                                              c[('r',3)],
                                              c[('H',1)],
                                              c[('dH',1,0)],
                                              c[('H',2)],
                                              c[('dH',2,0)],
                                              c[('H',3)],
                                              c[('dH',3,0)])
##\brief The non-conservative form of the level set equation for a moving boundary
#
#The equations are formulated as
#
#\f[
#u_t + \mathbf{v} \cdot \nabla u = 0
#\f]
#
from .ctransportCoefficients import ncLevelSetCoefficientsEvaluate
class NCLevelSetCoefficients(TC_base):

    def __init__(self,
                 V_model=0,
                 RD_model=None,
                 ME_model=1,
                 checkMass=True,epsFact=1.5):
        self.epsFact=epsFact
        self.variableNames=['phi']
        nc=1
        mass={0:{0:'linear'}}
        hamiltonian={0:{0:'linear'}}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         ['phi'])
        self.flowModelIndex=V_model
        self.modelIndex=ME_model
        self.RD_modelIndex=RD_model
        self.checkMass = checkMass
    def attachModels(self,modelList):
        #the level set model
        self.model = modelList[self.modelIndex]
        #the velocity
        if self.flowModelIndex >= 0:
            self.flowModel = modelList[self.flowModelIndex]
            self.q_v = modelList[self.flowModelIndex].q[('velocity',0)]
            self.ebqe_v = modelList[self.flowModelIndex].ebqe[('velocity',0)]
            if ('velocity',0) in modelList[self.flowModelIndex].ebq:
                self.ebq_v  = modelList[self.flowModelIndex].ebq[('velocity',0)]
            else:
                self.ebq_v  = None
            if ('u',0) not in self.model.ebq and ('u',0) in self.flowModel.ebq:
                self.model.ebq[('u',0)] = numpy.zeros(self.flowModel.ebq[('u',0)].shape,'d')
                self.model.ebq[('grad(u)',0)] = numpy.zeros(self.flowModel.ebq[('grad(u)',0)].shape,'d')
            if ('v',1) in self.flowModel.ebq:
                self.model.u[0].getValuesTrace(self.flowModel.ebq[('v',1)],self.model.ebq[('u',0)])
                self.model.u[0].getGradientValuesTrace(self.flowModel.ebq[('grad(v)',1)],self.model.ebq[('grad(u)',0)])
        if self.RD_modelIndex is not None:
            #print self.RD_modelIndex,len(modelList)
            self.rdModel = modelList[self.RD_modelIndex]
        if self.checkMass:
            self.m_pre = Norms.scalarSmoothedHeavisideDomainIntegral(self.epsFact,
                                                                     self.model.mesh.elementDiametersArray,
                                                                     self.model.q['dV'],
                                                                     self.model.q[('u',0)],
                                                                     self.model.mesh.nElements_owned)
            logEvent("Attach Models NCLS: Phase  0 mass before NCLS step = %12.5e" % (self.m_pre,),level=2)
            self.totalFluxGlobal=0.0
            self.lsGlobalMassArray = [self.m_pre]
            self.lsGlobalMassErrorArray = [0.0]
            self.fluxArray = [0.0]
            self.timeArray = [self.model.timeIntegration.t]
            self.ebqe_dS = self.model.ebqe['dS']
    def initializeElementQuadrature(self,t,cq):
        if self.flowModelIndex is None:
            self.q_v = numpy.zeros(cq[('grad(u)',0)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        if self.flowModelIndex is None:
            self.ebq_v = numpy.zeros(cebq[('grad(u)',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        if self.flowModelIndex is None:
            self.ebqe_v = numpy.zeros(cebqe[('grad(u)',0)].shape,'d')
    def preStep(self,t,firstStep=False):
        if self.checkMass:
            self.m_pre = Norms.scalarSmoothedHeavisideDomainIntegral(self.epsFact,
                                                                     self.model.mesh.elementDiametersArray,
                                                                     self.model.q['dV'],
                                                                     self.model.q[('u',0)],
                                                                     self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass before NCLS step = %12.5e" % (self.m_pre,),level=2)
            self.m_last = self.m_pre
            # self.m_last = Norms.scalarSmoothedHeavisideDomainIntegral(self.epsFact,
            #                                                           self.model.mesh.elementDiametersArray,
            #                                                           self.model.q['dV'],
            #                                                           self.model.timeIntegration.m_last[0],
            #                                                           self.model.mesh.nElements_owned)
            # logEvent("Phase  0 mass before NCLS step (m_last) = %12.5e" % (self.m_last,),level=2)
        #cek todo why is this here
        if self.flowModelIndex >= 0 and ('v',1) in self.flowModel.ebq:
            self.model.u[0].getValuesTrace(self.flowModel.ebq[('v',1)],self.model.ebq[('u',0)])
            self.model.u[0].getGradientValuesTrace(self.flowModel.ebq[('grad(v)',1)],self.model.ebq[('grad(u)',0)])
        copyInstructions = {}
        return copyInstructions
    def postStep(self,t,firstStep=False):
        if self.checkMass:
            self.m_post = Norms.scalarSmoothedHeavisideDomainIntegral(self.epsFact,
                                                                      self.model.mesh.elementDiametersArray,
                                                                      self.model.q['dV'],
                                                                      self.model.q[('u',0)],
                                                                      self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass after NCLS step = %12.5e" % (self.m_post,),level=2)
            #need a flux here not a velocity
            self.fluxIntegral = Norms.fluxDomainBoundaryIntegralFromVector(self.ebqe_dS,
                                                                           self.ebqe_v,
                                                                           self.model.ebqe['n'],
                                                                           self.model.mesh)
            logEvent("Flux integral = %12.5e" % (self.fluxIntegral,),level=2)
            logEvent("Phase  0 mass conservation after NCLS step = %12.5e" % (self.m_post - self.m_last + self.model.timeIntegration.dt*self.fluxIntegral,),level=2)
            self.lsGlobalMass = self.m_post
            self.fluxGlobal = self.fluxIntegral*self.model.timeIntegration.dt
            self.totalFluxGlobal += self.fluxGlobal
            self.lsGlobalMassArray.append(self.lsGlobalMass)
            self.lsGlobalMassErrorArray.append(self.lsGlobalMass - self.lsGlobalMassArray[0] + self.totalFluxGlobal)
            self.fluxArray.append(self.fluxIntegral)
            self.timeArray.append(self.model.timeIntegration.t)
        if self.flowModelIndex >= 0 and ('v',1) in self.flowModel.ebq:
            self.model.u[0].getValuesTrace(self.flowModel.ebq[('v',1)],self.model.ebq[('u',0)])
            self.model.u[0].getGradientValuesTrace(self.flowModel.ebq[('grad(v)',1)],self.model.ebq[('grad(u)',0)])
        copyInstructions = {}
        return copyInstructions
    def updateToMovingDomain(self,t,c):
        #in a moving domain simulation the velocity coming in is already for the moving domain
        pass
    def evaluate(self,t,c):
        v = None
        if c[('dH',0,0)].shape == self.q_v.shape:
            v = self.q_v
        elif c[('dH',0,0)].shape == self.ebqe_v.shape:
            v = self.ebqe_v
        elif self.ebq_v is not None and c[('dH',0,0)].shape == self.ebq_v.shape:
            v = self.ebq_v
        else:
            raise RuntimeError("don't have v for NC Level set of shape = " +repr(c[('dH',0,0)].shape))
        if v is not None:
            ncLevelSetCoefficientsEvaluate(v,
                                           c[('u',0)],
                                           c[('grad(u)',0)],
                                           c[('m',0)],
                                           c[('dm',0,0)],
                                           c[('H',0)],
                                           c[('dH',0,0)])

from .ctransportCoefficients import cLevelSetCoefficientsEvaluate
class CLevelSetCoefficients(TC_base):
    def __init__(self,V_model=0,RD_model=-1,ME_model=1,checkMass=True):
        self.variableNames=['vof']
        nc=1
        mass={0:{0:'linear'}}
        advection={0:{0:'linear'}}
        hamiltonian={}
        diffusion={}
        potential={}
        reaction={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames)
        self.flowModelIndex=V_model
        self.modelIndex=ME_model
        self.RD_modelIndex=RD_model
        self.checkMass = checkMass
    def attachModels(self,modelList):
        self.model = modelList[self.modelIndex]
        if self.RD_modelIndex >= 0:
            self.rdModel = modelList[self.RD_modelIndex]
        self.q_v = modelList[self.flowModelIndex].q[('velocity',0)]
        self.ebqe_v = modelList[self.flowModelIndex].ebqe[('velocity',0)]
        self.ebq_v = None
        if ('velocity',0) in modelList[self.flowModelIndex].ebq:
            self.ebq_v = modelList[self.flowModelIndex].ebq[('velocity',0)]
    def initializeElementQuadrature(self,t,cq):
        if self.flowModelIndex is None:
            self.q_v = numpy.ones(cq[('f',0)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        if self.flowModelIndex is None:
            self.ebq_v = numpy.ones(cebq[('f',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        if self.flowModelIndex is None:
            self.ebqe_v = numpy.ones(cebqe[('f',0)].shape,'d')
    def preStep(self,t,firstStep=False):
#         if self.RD_modelIndex >= 0:
#             print "updating time history",self.RD_modelIndex
#             nDOF_element=self.model.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim
#             dg_dofMap = self.model.u[0].femSpace.dofMap
#             cg_dofMap = self.rdModel.u[0].femSpace.dofMap
#             for eN in range(self.model.mesh.nElements_global):
#                 for j in range(nDOF_element):
#                     self.model.u[0].dof[dg_dofMap.l2g[eN,j]] = self.rdModel.u[0].dof[cg_dofMap.l2g[eN,j]]
#             self.model.calculateCoefficients()
#             self.model.calculateElementResidual()
#             for eN in range(self.rdModel.q[('u',0)].shape[0]):
#                 for k in range(self.rdModel.q[('u',0)].shape[1]):
#                     if self.rdModel.q[('u',0)][eN,k] > 0:
#                         self.model.q[('m',0)][eN,k] = 1.0
#                     else:
#                         self.model.q[('m',0)][eN,k] = 0.0
#             for eN in range(self.rdModel.ebq[('u',0)].shape[0]):
#                 for ebN in range(self.rdModel.ebq[('u',0)].shape[1]):
#                     for k in range(self.rdModel.ebq[('u',0)].shape[2]):
#                         if self.rdModel.ebq[('u',0)][eN,ebN,k] > 0:
#                             self.model.ebq[('m',0)][eN,ebN,k] = 1.0
#                         else:
#                             self.model.ebq[('m',0)][eN,ebN,k] = 0.0
#             self.model.updateTimeHistory(t,resetFromDOF=True)
#             copyInstructions = {}#'copy_uList':True,'uList_model':self.RD_modelIndex}
#             return copyInstructions
#        print "==============================Air volume-Clevelset-pre", Norms.scalarDomainIntegral(self.model.q['abs(det(J))'],
#                                                                                                   self.model.elementQuadratureWeights[('m',0)],
#                                                                                                   self.model.q[('m',0)])
        pass
        if firstStep: #try to force calculation of cfl
            self.model.calculateCoefficients()
            copyInstructions = {}
            return copyInstructions
    def postStep(self,t,firstStep=False):
 #       print "==============================Air volume-Clevelset-post", Norms.scalarDomainIntegral(self.model.q['abs(det(J))'],
 #                                                                                                   self.model.elementQuadratureWeights[('m',0)],
 #                                                                                                   self.model.q[('m',0)])
        if self.checkMass:
            heavisideMassPre = 0.0;
            for eN in range(self.model.mesh.nElements_global):
                for iq in range(self.model.nQuadraturePoints_element):
                    if self.model.q[('u',0)][eN,iq] >= 0.0:
                        heavisideMassPre += self.model.q[('dV_u',0)][eN,iq]
            #print """CLevel postStep t=%s u massIn =%s """ % (t,heavisideMassPre)
    def evaluate(self,t,c):
        if c[('f',0)].shape == self.q_v.shape:
            v = self.q_v
        elif c[('f',0)].shape == self.ebqe_v.shape:
            v = self.ebqe_v
        else:
            v = self.ebq_v
        cLevelSetCoefficientsEvaluate(v,
                                      c[('u',0)],
                                      c[('m',0)],
                                      c[('dm',0,0)],
                                      c[('f',0)],
                                      c[('df',0,0)])
from .ctransportCoefficients import VOFCoefficientsEvaluate
class VOFCoefficients(TC_base):
    def __init__(self,LS_model=-1,V_model=0,RD_model=-1,ME_model=1,checkMass=True,epsFact=0.0):
        self.variableNames=['vof']
        nc=1
        mass={0:{0:'linear'}}
        advection={0:{0:'linear'}}
        hamiltonian={}
        diffusion={}
        potential={}
        reaction={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames)
        self.epsFact=epsFact
        self.flowModelIndex=V_model
        self.modelIndex=ME_model
        self.RD_modelIndex=RD_model
        self.LS_modelIndex=LS_model
        self.checkMass = checkMass
    def initializeMesh(self,mesh):
        self.eps = self.epsFact*mesh.h
    def attachModels(self,modelList):
        #self
        self.model = modelList[self.modelIndex]
        #redistanced level set
        if self.RD_modelIndex >= 0:
            self.rdModel = modelList[self.RD_modelIndex]
        #level set
        self.lsModel = modelList[self.LS_modelIndex]
        self.q_phi = modelList[self.LS_modelIndex].q[('u',0)]
        self.ebqe_phi = modelList[self.LS_modelIndex].ebqe[('u',0)]
        if ('u',0) in modelList[self.LS_modelIndex].ebq:
            self.ebq_phi = modelList[self.LS_modelIndex].ebq[('u',0)]
        else:
            self.ebq_phi = None
        #flow model
        #print "flow model index------------",self.flowModelIndex,modelList[self.flowModelIndex].q.has_key(('velocity',0))
        if self.flowModelIndex >= 0:
            if ('velocity',0) in modelList[self.flowModelIndex].q:
                self.q_v = modelList[self.flowModelIndex].q[('velocity',0)]
                self.ebqe_v = modelList[self.flowModelIndex].ebqe[('velocity',0)]
            else:
                self.q_v = modelList[self.flowModelIndex].q[('f',0)]
                self.ebqe_v = modelList[self.flowModelIndex].ebqe[('f',0)]
            if ('velocity',0) in modelList[self.flowModelIndex].ebq:
                self.ebq_v = modelList[self.flowModelIndex].ebq[('velocity',0)]
            else:
                if ('f',0) in modelList[self.flowModelIndex].ebq:
                    self.ebq_v = modelList[self.flowModelIndex].ebq[('f',0)]
        if self.checkMass:
            self.m_pre = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                     self.model.q[('m',0)],
                                                     self.model.mesh.nElements_owned)
            logEvent("Attach Models VOF: Phase  0 mass after VOF step = %12.5e" % (self.m_pre,),level=2)
            self.m_post = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                     self.model.q[('m',0)],
                                                     self.model.mesh.nElements_owned)
            logEvent("Attach Models VOF: Phase  0 mass after VOF step = %12.5e" % (self.m_post,),level=2)
            if ('advectiveFlux',0) in self.model.ebqe:
                self.fluxIntegral = Norms.fluxDomainBoundaryIntegral(self.model.ebqe['dS'],
                                                                     self.model.ebqe[('advectiveFlux',0)],
                                                                     self.model.mesh)
                logEvent("Attach Models VOF: Phase  0 mass conservation after VOF step = %12.5e" % (self.m_post - self.m_pre + self.model.timeIntegration.dt*self.fluxIntegral,),level=2)

    def initializeElementQuadrature(self,t,cq):
        if self.flowModelIndex is None:
            self.q_v = numpy.ones(cq[('f',0)].shape,'d')
        #VRANS
        self.q_porosity = numpy.ones(cq[('u',0)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        if self.flowModelIndex is None:
            self.ebq_v = numpy.ones(cebq[('f',0)].shape,'d')
        #VRANS
        self.ebq_porosity = numpy.ones(cebq[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        if self.flowModelIndex is None:
            self.ebqe_v = numpy.ones(cebqe[('f',0)].shape,'d')
        #VRANS
        self.ebqe_porosity = numpy.ones(cebqe[('u',0)].shape,'d')
    def preStep(self,t,firstStep=False):
        if self.checkMass:
            self.m_pre = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                    self.model.q[('m',0)],
                                                    self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass before VOF step = %12.5e" % (self.m_pre,),level=2)
            self.m_last = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                     self.model.timeIntegration.m_last[0],
                                                     self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass before VOF (m_last) step = %12.5e" % (self.m_last,),level=2)
        copyInstructions = {}
        return copyInstructions
    def postStep(self,t,firstStep=False):
        if self.checkMass:
            self.m_post = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                     self.model.q[('m',0)],
                                                     self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass after VOF step = %12.5e" % (self.m_post,),level=2)
            self.fluxIntegral = Norms.fluxDomainBoundaryIntegral(self.model.ebqe['dS'],
                                                                 self.model.ebqe[('advectiveFlux',0)],
                                                                 self.model.mesh)
            logEvent("Phase  0 mass flux boundary integral after VOF step = %12.5e" % (self.fluxIntegral,),level=2)
            logEvent("Phase  0 mass conservation after VOF step = %12.5e" % (self.m_post - self.m_last + self.model.timeIntegration.dt*self.fluxIntegral,),level=2)
            divergence = Norms.fluxDomainBoundaryIntegralFromVector(self.model.ebqe['dS'],
                                                                    self.ebqe_v,
                                                                    self.model.ebqe['n'],
                                                                    self.model.mesh)
            logEvent("Divergence = %12.5e" % (divergence,),level=2)
        copyInstructions = {}
        return copyInstructions
    def updateToMovingDomain(self,t,c):
        #in a moving domain simulation the velocity coming in is already for the moving domain
        pass
    def evaluate(self,t,c):
        if c[('f',0)].shape == self.q_v.shape:
            v = self.q_v
            phi = self.q_phi
        elif c[('f',0)].shape == self.ebqe_v.shape:
            v = self.ebqe_v
            phi = self.ebqe_phi
        elif ((self.ebq_v is not None and self.ebq_phi is not None) and c[('f',0)].shape == self.ebq_v.shape):
            v = self.ebq_v
            phi = self.ebq_phi
        else:
            v=None
            phi=None
        if v is not None:
            VOFCoefficientsEvaluate(self.eps,
                                    v,
                                    phi,
                                    c[('u',0)],
                                    c[('m',0)],
                                    c[('dm',0,0)],
                                    c[('f',0)],
                                    c[('df',0,0)])
        if self.checkMass:
            logEvent("Phase  0 mass in eavl = %12.5e" % (Norms.scalarDomainIntegral(self.model.q['dV'],
                                                                               self.model.q[('m',0)],
                                                                               self.model.mesh.nElements_owned),),level=2)

class LevelSetNormalCoefficients(TC_base):
    def __init__(self,epsFact=1.5,LSModel_index=-1,phi_func=None):
        self.phi_func = phi_func
        self.variableNames=['phi_smooth']
        nc=1
        mass={}
        advection={}
        hamiltonian={}
        diffusion={0:{0:{0:'constant'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames)
        self.levelSetModelIndex=LSModel_index
        self.epsFact=epsFact
    def initializeMesh(self,mesh):
        self.eps = self.epsFact*mesh.h
    def attachModels(self,modelList):
        #print "normal attach models",self.levelSetModelIndex
        if self.levelSetModelIndex > 0:
            self.q_r   = modelList[self.levelSetModelIndex].q[('u',0)]
            self.ebqe_r= modelList[self.levelSetModelIndex].ebqe[('u',0)]
            self.ebq_r = None
            if ('u',0) in modelList[self.levelSetModelIndex].ebq:
                self.ebq_r = modelList[self.levelSetModelIndex].ebq[('u',0)]
    def initializeElementQuadrature(self,t,cq):
        #initialize so it can run without a flow model
        self.q_r = numpy.ones(cq[('u',0)].shape,'d')
        if self.phi_func is not None:
            self.phi_func(cq['x'],self.q_r)
        for eN in range(cq[('a',0,0)].shape[0]):
            for k in range(cq[('a',0,0)].shape[1]):
                for I in range(cq[('a',0,0)].shape[2]):
                    cq[('a',0,0)][eN,k,I,I]=self.eps
        cq[('dr',0,0)].flat[:]=1.0
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.ebq_r = numpy.ones(cebq[('u',0)].shape,'d')
        if self.phi_func is not None:
            self.phi_func(cebq['x'],self.ebq_r)
        for eN in range(cebq[('a',0,0)].shape[0]):
            for ebN in range(cebq[('a',0,0)].shape[1]):
                for k in range(cebq[('a',0,0)].shape[2]):
                    for I in range(cebq[('a',0,0)].shape[3]):
                        cebq[('a',0,0)][eN,ebN,k,I,I]=self.eps
        cebq[('dr',0,0)].flat[:]=1.0
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.ebqe_r = numpy.ones(cebqe[('u',0)].shape,'d')
        if self.phi_func is not None:
            self.phi_func(cebqe['x'],self.ebqe_r)
        for ebNE in range(cebqe[('a',0,0)].shape[0]):
            for k in range(cebqe[('a',0,0)].shape[1]):
                for I in range(cebqe[('a',0,0)].shape[2]):
                    cebq[('a',0,0)][ebNE,k,I,I]=self.eps
        cebqe[('dr',0,0)].flat[:]=1.0
    def evaluate(self,t,c):
        if c[('r',0)].shape == self.q_r.shape:
            r = self.q_r
        elif c[('r',0)].shape == self.ebqe_r.shape:
            r = self.ebqe_r
        else:
            r = self.ebq_r
        c[('r',0)].flat[:]=c[('u',0)].flat
        c[('r',0)] -= r

from .ctransportCoefficients import levelSetCurvatureCoefficientsEvaluate    
class LevelSetCurvatureCoefficients(TC_base):
    def __init__(self,epsFact=0.0,LSModel_index=3,grad_phi_func=None,sd=True,nd=None):
        self.sd=sd
        self.grad_phi_func = grad_phi_func
        self.variableNames=['kappa']
        nc=1
        mass={}
        advection={0:{0:'constant'}}
        hamiltonian={}
        diffusion={}
        potential={}
        diffusion={0:{0:{0:'constant'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        if self.sd:
            assert nd is not None,"You must set the number of dimensions to use sparse diffusion in LevelSetCurvatureCoefficients"
            sdInfo = {(0,0):(numpy.arange(start=0,stop=nd+1,step=1,dtype='i'),
                             numpy.arange(start=0,stop=nd,step=1,dtype='i'))}
        else:
            sdInfo={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames,
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion = sd)
        self.levelSetModelIndex=LSModel_index
        self.epsFact=epsFact
    def initializeMesh(self,mesh):
        self.eps = self.epsFact*mesh.h
    def attachModels(self,modelList):
        logEvent("Attaching \\grad \\phi in curvature model")
        self.q_grad_phi    = modelList[self.levelSetModelIndex].q[('grad(u)',0)]
        self.ebqe_grad_phi = modelList[self.levelSetModelIndex].ebqe[('grad(u)',0)]
        if ('grad(u)',0) in modelList[self.levelSetModelIndex].ebq:
            self.ebq_grad_phi = modelList[self.levelSetModelIndex].ebq[('grad(u)',0)]
        else:
            self.ebq_grad_phi  = None
    def initializeElementQuadrature(self,t,cq):
        if self.levelSetModelIndex is None:
            self.q_grad_phi = numpy.ones(cq[('f',0)].shape,'d')
            if self.grad_phi_func is not None:
                self.grad_phi_func(cq['x'],self.q_grad_phi)
            if self.sd:
                cq[('a',0,0)].fill(self.eps)
            else:
                for eN in range(cq[('a',0,0)].shape[0]):
                    for k in range(cq[('a',0,0)].shape[1]):
                        for I in range(cq[('a',0,0)].shape[2]):
                            cq[('a',0,0)][eN,k,I,I]=self.eps
            cq[('df',0,0)].fill(0.0)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        if self.levelSetModelIndex is None:
            self.ebq_grad_phi = numpy.ones(cebq[('f',0)].shape,'d')
            if self.grad_phi_func is not None:
                self.grad_phi_func(cebq['x'],self.ebq_grad_phi)
            if self.sd:
                cebq[('a',0,0)].fill(self.eps)
            else:
                for eN in range(cebq[('a',0,0)].shape[0]):
                    for ebN in range(cebq[('a',0,0)].shape[1]):
                        for k in range(cebq[('a',0,0)].shape[2]):
                            for I in range(cebq[('a',0,0)].shape[3]):
                                cebq[('a',0,0)][eN,ebN,k,I,I]=self.eps
            cebq[('df',0,0)].fill(0.0)
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        if self.levelSetModelIndex is None:
            self.ebqe_grad_phi = numpy.ones(cebqe[('f',0)].shape,'d')
            if self.grad_phi_func is not None:
                self.grad_phi_func(cebqe['x'],self.ebqe_grad_phi)
            if self.sd:
                cebqe[('a',0,0)].fill(self.eps)
            else:
                for ebNE in range(cebqe[('a',0,0)].shape[0]):
                    for k in range(cebqe[('a',0,0)].shape[1]):
                        for I in range(cebqe[('a',0,0)].shape[2]):
                            cebqe[('a',0,0)][ebNE,k,I,I]=self.eps
            cebqe[('df',0,0)].fill(0.0)
    def evaluate(self,t,c):
        if c[('f',0)].shape == self.q_grad_phi.shape:
            grad_phi = self.q_grad_phi
        elif c[('f',0)].shape == self.ebqe_grad_phi.shape:
            grad_phi = self.ebqe_grad_phi
        else:
            grad_phi = self.ebq_grad_phi
        levelSetCurvatureCoefficientsEvaluate(grad_phi,
                                              c[('u',0)],
                                              c[('f',0)],
                                              c[('r',0)],
                                              c[('dr',0,0)])

from .ctransportCoefficients import levelSetConservationCoefficientsEvaluate
from .ctransportCoefficients import levelSetConservationCoefficientsEvaluate_sd
class LevelSetConservation(TC_base):
    def __init__(self,applyCorrection=True,epsFactHeaviside=0.0,epsFactDirac=1.0,epsFactDiffusion=2.0,LSModel_index=3,V_model=2,me_model=5,VOFModel_index=4,checkMass=True,sd=True,nd=None,applyCorrectionToDOF=True):
        self.sd=sd
        self.checkMass=checkMass
        self.variableNames=['phiCorr']
        nc=1
        mass={}
        advection={}
        hamiltonian={}
        diffusion={0:{0:{0:'constant'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'nonlinear'}}
        #reaction={}
        if self.sd:
            assert nd is not None,"You must set the number of dimensions to use sparse diffusion in LevelSetConservationCoefficients"
            sdInfo = {(0,0):(numpy.arange(start=0,stop=nd+1,step=1,dtype='i'),
                             numpy.arange(start=0,stop=nd,step=1,dtype='i'))}
        else:
            sdInfo={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames,
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion = sd)
        self.levelSetModelIndex=LSModel_index
        self.flowModelIndex=V_model
        self.epsFactHeaviside=epsFactHeaviside
        self.epsFactDirac=epsFactDirac
        self.epsFactDiffusion=epsFactDiffusion
        self.me_model=me_model
        self.VOFModelIndex=VOFModel_index
        self.useC = True
        self.applyCorrection=applyCorrection
        if self.applyCorrection:
            self.applyCorrectionToDOF=applyCorrectionToDOF
        else:
            self.applyCorrection = False
    def initializeMesh(self,mesh):
        self.h=mesh.h
        self.epsHeaviside = self.epsFactHeaviside*mesh.h
        self.epsDirac = self.epsFactDirac*mesh.h
        self.epsDiffusion = self.epsFactDiffusion*mesh.h
    def attachModels(self,modelList):
        import copy
        logEvent("Attaching models in LevelSetConservation")
        #level set
        self.lsModel = modelList[self.levelSetModelIndex]
        self.q_u_ls    = modelList[self.levelSetModelIndex].q[('u',0)]
        self.ebqe_u_ls = modelList[self.levelSetModelIndex].ebqe[('u',0)]
        if ('u',0) in modelList[self.levelSetModelIndex].ebq:
            self.ebq_u_ls = modelList[self.levelSetModelIndex].ebq[('u',0)]
        else:
            self.ebq_u_ls = None
        #volume of fluid
        self.vofModel = modelList[self.VOFModelIndex]
        self.q_H_vof = modelList[self.VOFModelIndex].q[('u',0)]
        self.ebqe_H_vof = modelList[self.VOFModelIndex].ebqe[('u',0)]
        if ('u',0) in modelList[self.VOFModelIndex].ebq:
            self.ebq_H_vof = modelList[self.VOFModelIndex].ebq[('u',0)]
        else:
            self.ebq_H_vof = None
        #correction
        self.massCorrModel = modelList[self.me_model]
        if self.checkMass:
            self.m_tmp = copy.deepcopy(self.massCorrModel.q[('r',0)])
            if self.checkMass:
                self.vofGlobalMass = Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                                                                self.vofModel.q[('u',0)],
                                                                self.massCorrModel.mesh.nElements_owned)
                self.lsGlobalMass = Norms.scalarHeavisideDomainIntegral(self.vofModel.q['dV'],
                                                                        self.lsModel.q[('u',0)],
                                                                        self.massCorrModel.mesh.nElements_owned)
                logEvent("Attach Models MCorr: mass correction %12.5e" % (Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                                                                                                self.massCorrModel.q[('r',0)],
                                                                                                self.massCorrModel.mesh.nElements_owned),),level=2)
                self.fluxGlobal = 0.0
                self.totalFluxGlobal = 0.0
                self.vofGlobalMassArray = [self.vofGlobalMass]
                self.lsGlobalMassArray = [self.lsGlobalMass]
                self.vofGlobalMassErrorArray = [self.vofGlobalMass - self.vofGlobalMassArray[0] + self.vofModel.timeIntegration.dt*self.vofModel.coefficients.fluxIntegral]
                self.lsGlobalMassErrorArray = [self.lsGlobalMass - self.lsGlobalMassArray[0] + self.vofModel.timeIntegration.dt*self.vofModel.coefficients.fluxIntegral]
                self.fluxArray = [self.vofModel.coefficients.fluxIntegral]
                self.timeArray = [self.vofModel.timeIntegration.t]
                logEvent("Attach Models MCorr: Phase 0 mass after mass correction (VOF) %12.5e" % (self.vofGlobalMass,),level=2)
                logEvent("Attach Models MCorr: Phase 0 mass after mass correction (LS) %12.5e" % (self.lsGlobalMass,),level=2)
                logEvent("Attach Models MCorr: Phase  0 mass conservation (VOF) after step = %12.5e" % (self.vofGlobalMass - self.vofModel.coefficients.m_pre + self.vofModel.timeIntegration.dt*self.vofModel.coefficients.fluxIntegral,),level=2)
                logEvent("Attach Models MCorr: Phase  0 mass conservation (LS) after step = %12.5e" % (self.lsGlobalMass - self.lsModel.coefficients.m_pre + self.vofModel.timeIntegration.dt*self.vofModel.coefficients.fluxIntegral,),level=2)
    def initializeElementQuadrature(self,t,cq):
        if self.sd and ('a',0,0) in cq:
            cq[('a',0,0)].fill(self.epsDiffusion)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        if self.sd and ('a',0,0) in cebq:
            cebq[('a',0,0)].fill(self.epsDiffusion)
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        if self.sd and ('a',0,0) in cebqe:
            cebqe[('a',0,0)].fill(self.epsDiffusion)
    def preStep(self,t,firstStep=False):
        if self.checkMass:
            logEvent("Phase 0 mass before mass correction (VOF) %12.5e" % (Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                                                                                                 self.vofModel.q[('m',0)],
                                                                                                 self.massCorrModel.mesh.nElements_owned),),level=2)
            logEvent("Phase 0 mass before mass correction (LS) %12.5e" % (Norms.scalarHeavisideDomainIntegral(self.vofModel.q['dV'],
                                                                                                         self.lsModel.q[('m',0)],
                                                                                                         self.massCorrModel.mesh.nElements_owned),),level=2)
        copyInstructions = {'clear_uList':True}
        return copyInstructions
    def postStep(self,t,firstStep=False):
        if self.applyCorrection:

            self.vofModel.q[('m',0)] += self.massCorrModel.q[('r',0)]
            self.lsModel.q[('m',0)] += self.massCorrModel.q[('u',0)]
            if self.vofModel.q[('u',0)] is not self.vofModel.q[('m',0)]:
                self.vofModel.q[('u',0)][:]=self.vofModel.q[('m',0)]
            if self.lsModel.q[('u',0)] is not self.lsModel.q[('m',0)]:
                self.lsModel.q[('u',0)][:]=self.lsModel.q[('m',0)]
            if ('mt',0) in self.vofModel.q:
                self.vofModel.timeIntegration.calculateElementCoefficients(self.vofModel.q)
                self.vofModel.timeIntegration.lastStepErrorOk()
            if self.applyCorrectionToDOF:
                self.lsModel.u[0].dof += self.massCorrModel.u[0].dof
            if ('mt',0) in self.lsModel.q:
                self.lsModel.timeIntegration.calculateElementCoefficients(self.lsModel.q)
                self.lsModel.timeIntegration.lastStepErrorOk()
            if self.checkMass:
                self.vofGlobalMass = Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                                                                self.vofModel.q[('u',0)],
                                                                self.massCorrModel.mesh.nElements_owned)
                self.lsGlobalMass = Norms.scalarHeavisideDomainIntegral(self.vofModel.q['dV'],
                                                                        self.lsModel.q[('u',0)],
                                                                        self.massCorrModel.mesh.nElements_owned)
                logEvent("mass correction %12.5e" % (Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                                                                           self.massCorrModel.q[('r',0)],
                                                                           self.massCorrModel.mesh.nElements_owned),),level=2)
                self.fluxGlobal = self.vofModel.coefficients.fluxIntegral*self.vofModel.timeIntegration.dt
                self.totalFluxGlobal += self.vofModel.coefficients.fluxIntegral*self.vofModel.timeIntegration.dt
                self.vofGlobalMassArray.append(self.vofGlobalMass)
                self.lsGlobalMassArray.append(self.lsGlobalMass)
                self.vofGlobalMassErrorArray.append(self.vofGlobalMass - self.vofGlobalMassArray[0] + self.totalFluxGlobal)
                self.lsGlobalMassErrorArray.append(self.lsGlobalMass - self.lsGlobalMassArray[0] + self.totalFluxGlobal)
                self.fluxArray.append(self.vofModel.coefficients.fluxIntegral)
                self.timeArray.append(self.vofModel.timeIntegration.t)
                logEvent("Phase 0 mass after mass correction (VOF) %12.5e" % (self.vofGlobalMass,),level=2)
                logEvent("Phase 0 mass after mass correction (LS) %12.5e" % (self.lsGlobalMass,),level=2)
                logEvent("Phase  0 mass conservation (VOF) after step = %12.5e" % (self.vofGlobalMass - self.vofModel.coefficients.m_last + self.fluxGlobal,),level=2)
                logEvent("Phase  0 mass conservation (LS) after step = %12.5e" % (self.lsGlobalMass - self.lsModel.coefficients.m_last + self.fluxGlobal,),level=2)
        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        import math
        if c[('u',0)].shape == self.q_u_ls.shape:
            u_ls = self.q_u_ls
            H_vof = self.q_H_vof
        elif c[('u',0)].shape == self.ebqe_u_ls.shape:
            u_ls = self.ebqe_u_ls
            H_vof = self.ebqe_H_vof
        elif self.ebq_u_ls is not None and c[('u',0)].shape == self.ebq_u_ls.shape:
            u_ls = self.ebq_u_ls
            H_vof = self.ebq_H_vof
        else:
            #\todo trap errors in TransportCoefficients.py
            u_ls = None
            H_vof = None
        if u_ls is not None and H_vof is not None:
            if self.useC:
                if self.sd:
                    levelSetConservationCoefficientsEvaluate_sd(self.epsHeaviside,
                                                                self.epsDirac,
                                                                u_ls,
                                                                H_vof,
                                                                c[('u',0)],
                                                                c[('r',0)],
                                                                c[('dr',0,0)])
                else:
                    levelSetConservationCoefficientsEvaluate(self.epsHeaviside,
                                                             self.epsDirac,
                                                             self.epsDiffusion,
                                                             u_ls,
                                                             H_vof,
                                                             c[('u',0)],
                                                             c[('r',0)],
                                                             c[('dr',0,0)],
                                                             c[('a',0,0)])
        if (self.checkMass and c[('u',0)].shape == self.q_u_ls.shape):
            self.m_tmp[:] = H_vof
            self.m_tmp += self.massCorrModel.q[('r',0)]
            logEvent("mass correction during Newton %12.5e" % (Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                                                                                     self.massCorrModel.q[('r',0)],
                                                                                     self.massCorrModel.mesh.nElements_owned),),level=2)
            logEvent("Phase 0 mass during Newton %12.5e" % (Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                                                                                 self.m_tmp,
                                                                                  self.massCorrModel.mesh.nElements_owned),),level=2)

from .ctransportCoefficients import  conservativeHeadRichardsL2projMualemVanGenuchtenHomEvaluate
from .ctransportCoefficients import  conservativeHeadRichardsL2projBndMualemVanGenuchtenHomEvaluate
class ConservativeHeadRichardsL2projMualemVanGenuchten(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 thetaS,
                 thetaR,
                 alpha,
                 n,
                 m):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.Ks = hydraulicConductivity
        self.gravity=gravity
        self.rho = density
        self.thetaS = thetaS
        self.thetaR = thetaR
        self.thetaSR = thetaS - thetaR
        self.alpha = alpha
        self.n = n
        self.m = m
    def evaluate(self,t,c):
        if ('dV_u',0) in c:
            volFact = 1.0;
            if c[('f',0)].shape[2] == 2:
                volFact =0.5
            elif c[('f',0)].shape[2] == 3:
                volFact = 1./6.
            for eN in range(c[('dV_u',0)].shape[0]):
                vol = 0.0;
                for k in range(c[('dV_u',0)].shape[1]):
                    vol += c[('dV_u',0)][eN,k]
                #end k
                detJ_eN = c['abs(det(J))'][eN,0]
                assert abs(vol-detJ_eN*volFact) < 1.0e-8,  "L2proj coefs eN=%d vol=%g detJ_eN=%g volFact= %g" % (eN,vol,detJ_eN,
                                                                                                      volFact)
            conservativeHeadRichardsL2projMualemVanGenuchtenHomEvaluate(self.rho,
                                                                        self.gravity,
                                                                        self.alpha,
                                                                        self.n,
                                                                        self.m,
                                                                        self.thetaR,
                                                                        self.thetaSR,
                                                                        self.Ks,
                                                                        c[('dV_u',0)],
                                                                        c[('u',0)],
                                                                        c[('m',0)],
                                                                        c[('dm',0,0)],
                                                                        c[('f',0)],
                                                                        c[('df',0,0)],
                                                                        c[('a',0,0)],
                                                                        c[('da',0,0,0)])

        elif ('dS_u',0) in c:
            volFact = 1.0;
            if c[('f',0)].shape[-1] == 3:
                volFact =0.5
            for eN in range(c[('dS_u',0)].shape[0]):
                for ebN in range(c[('dS_u',0)].shape[1]):
                    area = 0.0;
                    for k in range(c[('dS_u',0)].shape[2]):
                        area += c[('dS_u',0)][eN,ebN,k]
                    #end k
                    detg_ebN = c['sqrt(det(g))'][eN,ebN,0]
                    assert abs(area-detg_ebN*volFact) < 1.0e-8,  "L2Bndproj coefs eN=%d ebN=%d area=%g detg_ebN=%g" % (eN,
                                                                                                                       ebN,
                                                                                                                       area,
                                                                                                                       detg_ebN)

            conservativeHeadRichardsL2projBndMualemVanGenuchtenHomEvaluate(self.rho,
                                                                           self.gravity,
                                                                           self.alpha,
                                                                           self.n,
                                                                           self.m,
                                                                           self.thetaR,
                                                                           self.thetaSR,
                                                                           self.Ks,
                                                                           c[('dS_u',0)],
                                                                           c[('u',0)],
                                                                           c[('m',0)],
                                                                           c[('dm',0,0)],
                                                                           c[('f',0)],
                                                                           c[('df',0,0)],
                                                                           c[('a',0,0)],
                                                                           c[('da',0,0,0)])
        else:
            assert False, "dS_u or dV_u keys not found!"

from .ctransportCoefficients import  conservativeHeadRichardsL2projMualemVanGenuchtenHetEvaluate        
class ConservativeHeadRichardsL2projMualemVanGenuchtenBlockHet(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 setParamsFunc):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.gravity=gravity
        self.rho = density
        self.setParamsFunc = setParamsFunc
    def initializeElementQuadrature(self,t,cq):
        self.vgm_n_q = numpy.zeros(cq[('u',0)].shape,'d')
        self.vgm_alpha_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.Ks_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.thetaR_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.thetaSR_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.setParamsFunc(cq['x'],
                           self.vgm_n_q,
                           self.vgm_alpha_q,
                           self.Ks_q,
                           self.thetaR_q,
                           self.thetaSR_q)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.vgm_n_ebq = numpy.zeros(cebq[('u',0)].shape,'d')
        self.vgm_alpha_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.Ks_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.thetaR_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.thetaSR_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.setParamsFunc(cebq['x'],
                           self.vgm_n_ebq,
                           self.vgm_alpha_ebq,
                           self.Ks_ebq,
                           self.thetaR_ebq,
                           self.thetaSR_ebq)
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.vgm_n_ebqe = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.vgm_alpha_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.Ks_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.thetaR_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.thetaSR_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.setParamsFunc(cebqe['x'],
                           self.vgm_n_ebqe,
                           self.vgm_alpha_ebqe,
                           self.Ks_ebqe,
                           self.thetaR_ebqe,
                           self.thetaSR_ebqe)
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.Ks_q.shape:
            vgm_n     = self.vgm_n_q
            vgm_alpha = self.vgm_alpha_q
            Ks = self.Ks_q
            thetaR = self.thetaR_q
            thetaSR = self.thetaSR_q
            dV = c[('dV_u',0)]
        elif c[('u',0)].shape == self.Ks_ebqe.shape:
            vgm_n     = self.vgm_n_ebqe
            vgm_alpha = self.vgm_alpha_ebqe
            Ks = self.Ks_ebqe
            thetaR = self.thetaR_ebqe
            thetaSR = self.thetaSR_ebqe
            dV = c[('dS_u',0)]
        else:
            vgm_n     = self.vgm_n_ebq
            vgm_alpha = self.vgm_alpha_ebq
            Ks = self.Ks_ebq
            thetaR = self.thetaR_ebq
            thetaSR = self.thetaSR_ebq
            dV = c[('dS_u',0)]



        conservativeHeadRichardsL2projMualemVanGenuchtenHetEvaluate(self.rho,
                                                                    self.gravity,
                                                                    vgm_alpha,
                                                                    vgm_n,
                                                                    thetaR,
                                                                    thetaSR,
                                                                    Ks,
                                                                    dV,
                                                                    c[('u',0)],
                                                                    c[('m',0)],
                                                                    c[('dm',0,0)],
                                                                    c[('f',0)],
                                                                    c[('df',0,0)],
                                                                    c[('a',0,0)],
                                                                    c[('da',0,0,0)])


from .ctransportCoefficients import  conservativeHeadRichardsMualemVanGenuchtenHomEvaluate
class ConservativeHeadRichardsMualemVanGenuchten(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 thetaS,
                 thetaR,
                 alpha,
                 n,
                 m,
                 beta=0.0):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        #potential={0:{0:'nonlinear'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.Ks = hydraulicConductivity
        self.gravity=gravity
        self.rho = density
        self.thetaS = thetaS
        self.thetaR = thetaR
        self.thetaSR = thetaS - thetaR
        self.alpha = alpha
        self.n = n
        self.m = m
        self.beta=beta
    def evaluate(self,t,c):
        conservativeHeadRichardsMualemVanGenuchtenHomEvaluate(self.rho,
                                                              self.beta,
                                                              self.gravity,
                                                              c[('x')],
                                                              self.alpha,
                                                              self.n,
                                                              self.m,
                                                              self.thetaR,
                                                              self.thetaSR,
                                                              self.Ks,
                                                              c[('u',0)],
                                                              c[('m',0)],
                                                              c[('dm',0,0)],
                                                              c[('f',0)],
                                                              c[('df',0,0)],
                                                              c[('a',0,0)],
                                                              c[('da',0,0,0)],
                                                              c[('phi',0)],
                                                              c[('dphi',0,0)])

from .ctransportCoefficients import  conservativeSatRichardsMualemVanGenuchtenHomEvaluate
class ConservativeSatRichardsMualemVanGenuchten(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 thetaS,
                 thetaR,
                 alpha,
                 n,
                 m):
        variableNames=['s']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'nonlinear'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.Ks = hydraulicConductivity
        self.gravity=gravity
        self.rho = density
        self.thetaS = thetaS
        self.thetaR = thetaR
        self.thetaSR = thetaS - thetaR
        self.alpha = alpha
        self.n = n
        self.m = m
    def evaluate(self,t,c):
        conservativeSatRichardsMualemVanGenuchtenHomEvaluate(self.rho,
                                                             self.gravity,
                                                             c[('x')],
                                                             self.alpha,
                                                             self.n,
                                                             self.m,
                                                             self.thetaR,
                                                             self.thetaSR,
                                                             self.Ks,
                                                             c[('u',0)],
                                                             c[('m',0)],
                                                             c[('dm',0,0)],
                                                             c[('f',0)],
                                                             c[('df',0,0)],
                                                             c[('a',0,0)],
                                                             c[('da',0,0,0)],
                                                             c[('phi',0)],
                                                             c[('dphi',0,0)])

from .ctransportCoefficients import  conservativeTotalHeadRichardsMualemVanGenuchtenHomEvaluate
class ConservativeTotalHeadRichardsMualemVanGenuchten(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 thetaS,
                 thetaR,
                 alpha,
                 n,
                 m):
        variableNames=['Pressure_Head'] #depends on how formulate?['h'],['Pressure Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'constant'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        #unknown is h
        #potential={0:{0:'u'}}
        #unknown is psi
        potential={0:{0:'nonlinear'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.Ks = hydraulicConductivity
        self.gravity=gravity
        self.rho = density
        self.thetaS = thetaS
        self.thetaR = thetaR
        self.thetaSR = thetaS - thetaR
        self.alpha = alpha
        self.n = n
        self.m = m
    def evaluate(self,t,c):
        conservativeTotalHeadRichardsMualemVanGenuchtenHomEvaluate(self.rho,
                                                                   self.gravity,
                                                                   c[('x')],
                                                                   self.alpha,
                                                                   self.n,
                                                                   self.m,
                                                                   self.thetaR,
                                                                   self.thetaSR,
                                                                   self.Ks,
                                                                   c[('u',0)],
                                                                   c[('m',0)],
                                                                   c[('dm',0,0)],
                                                                   c[('f',0)],
                                                                   c[('df',0,0)],
                                                                   c[('a',0,0)],
                                                                   c[('da',0,0,0)],
                                                                   c[('phi',0)],
                                                                   c[('dphi',0,0)])
        
def VGM_to_BCB_Simple(vgm_alpha,vgm_n):
    bcb_lambda = vgm_n-1
    bcb_pd = 1.0/vgm_alpha
    return (bcb_lambda,bcb_pd)
def BCB_to_VGM_Simple(bcb_pd,bcb_lambda):
    vgm_n = bcb_lambda + 1
    vgm_alpha = 1.0/bcb_pd
    return (vgm_alpha,vgm_n)

def VGM_to_BCB_Johns(vgm_alpha,vgm_n):
    vgm_m = 1.0 - 1.0/vgm_n
    bcb_lambda = vgm_m/(1.0-vgm_m)*(1.0-(0.5)**(1.0/vgm_m))
    thetaStar=0.72-0.35*exp(-vgm_n**4);
    bcb_pd = (thetaStar**(1.0/bcb_lambda))/vgm_alpha*(thetaStar**(-1.0/vgm_m)-1.0)**( 1.0-vgm_m)
    return (bcb_lambda,bcb_pd)

def VGM_to_BCB_MorelSeytoux(vgm_alpha,vgm_n):
    vgm_m = 1.0 - 1.0/vgm_n
    p = 1.0 + 2.0/vgm_m
    bcb_pd = ((1.0/vgm_alpha)*
              ((p+3.0)/((2.0*p)*(p-1.0)))*
              ((147.8+8.1*p+0.092*p**2)/(55.6+7.4*p+p**2)))
    bcb_lambda = 2.0/(p-3.0)
    return (bcb_lambda,bcb_pd)

from .ctransportCoefficients import  conservativeHeadRichardsBrooksCoreyBurdineHomEvaluate
class ConservativeHeadRichardsBrooksCoreyBurdine(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 thetaS,
                 thetaR,
                 lambdab,
                 pd,
                 beta=0.0):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.Ks = hydraulicConductivity
        self.gravity=gravity
        self.rho = density
        self.thetaS = thetaS
        self.thetaR = thetaR
        self.thetaSR = thetaS - thetaR
        self.lambdab = lambdab
        self.pd = pd
        self.beta=beta
    def evaluate(self,t,c):
        conservativeHeadRichardsBrooksCoreyBurdineHomEvaluate(self.rho,
                                                              self.beta,
                                                              self.gravity,
                                                              self.lambdab,
                                                              self.pd,
                                                              self.thetaR,
                                                              self.thetaSR,
                                                              self.Ks,
                                                              c[('u',0)],
                                                              c[('m',0)],
                                                              c[('dm',0,0)],
                                                              c[('f',0)],
                                                              c[('df',0,0)],
                                                              c[('a',0,0)],
                                                              c[('da',0,0,0)])

from .ctransportCoefficients import  conservativeHeadRichardsMualemVanGenuchtenHetEvaluate
class ConservativeHeadRichardsMualemVanGenuchtenHet(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 thetaS,
                 thetaR,
                 alpha,
                 n,
                 m):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.Ks_mean = hydraulicConductivity
        self.gravity=gravity
        self.rho = density
        self.thetaSMean = thetaS
        self.thetaRMean = thetaR
        self.thetaSRMean = thetaS - thetaR
        self.alphaMean = alpha
        self.nMean = n
        self.hetLeft=0.0
        self.hetRight=0.35
        self.Ks_sigma=0.65
    def initializeElementQuadrature(self,t,cq):
        import random
        rng = random.Random()
        self.Ks_q        = numpy.zeros(cq[('u',0)].shape,'d')
        self.vgm_alpha_q = numpy.zeros(cq[('u',0)].shape,'d')
        self.vgm_n_q     = numpy.zeros(cq[('u',0)].shape,'d')
        self.thetaR_q    = numpy.zeros(cq[('u',0)].shape,'d')
        self.thetaSR_q   = numpy.zeros(cq[('u',0)].shape,'d')

        nPoints=1
        for d in cq[('u',0)].shape:
            nPoints*=d
        for k in range(nPoints):
            if cq['x'].flat[k*3+0] >= self.hetLeft and cq['x'].flat[k*3+0] <= self.hetRight:
                #self.Ksq.flat[k] = rng.lognormvariate(mu=self.Ks_mean,sigma=self.Ks_sigma)
                self.Ks_q.flat[k] = self.Ks_mean*10.0
            else:
                self.Ks_q.flat[k] = self.Ks_mean
            #self.Ksq.flat[k] = self.Ks_mean
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        import random
        rng = random.Random()
        self.Ks_ebq      = numpy.zeros(cebq[('u',0)].shape,'d')
        self.vgm_alpha_ebq = numpy.zeros(cebq[('u',0)].shape,'d')
        self.vgm_n_ebq     = numpy.zeros(cebq[('u',0)].shape,'d')
        self.thetaR_ebq    = numpy.zeros(cebq[('u',0)].shape,'d')
        self.thetaSR_ebq   = numpy.zeros(cebq[('u',0)].shape,'d')
        nPoints=1
        for d in cebq[('u',0)].shape:
            nPoints*=d
        for k in range(nPoints):
            if cebq['x'].flat[k*3+0] >= self.hetLeft and cebq['x'].flat[k*3+0] <= self.hetRight:
                #self.Ks_ebq.flat[k] = rng.lognormvariate(mu=self.Ks_mean,sigma=self.Ks_sigma)
                self.Ks_ebq.flat[k] = self.Ks_mean*10.0
            else:
                self.Ks_ebq.flat[k] = self.Ks_mean
            #self.Ks_ebq.flat[k] = self.Ks_mean
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        import random
        rng = random.Random()
        self.Ks_ebqe        = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.vgm_alpha_ebqe = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.vgm_n_ebqe     = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.thetaR_ebqe    = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.thetaSR_ebqe   = numpy.zeros(cebqe[('u',0)].shape,'d')
        nPoints=1
        for d in cebqe[('u',0)].shape:
            nPoints*=d
        for k in range(nPoints):
            if cebqe['x'].flat[k*3+0] >= self.hetLeft and cebqe['x'].flat[k*3+0] <= self.hetRight:
                #self.Ks_ebqe.flat[k] = rng.lognormvariate(mu=self.Ks_mean,sigma=self.Ks_sigma)
                self.Ks_ebqe.flat[k] = self.Ks_mean*10.0
            else:
                self.Ks_ebqe.flat[k] = self.Ks_mean
            #self.Ks_ebqe.flat[k] = self.Ks_mean
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.Ks_q.shape:
            Ks        = self.Ks_q
            vgm_alpha = self.vgm_alpha_q
            vgm_n     = self.vgm_n_q
            thetaR    = self.thetaR_q
            thetaSR   = self.thetaSR_q
        elif c[('u',0)].shape == self.Ks_ebqe.shape:
            Ks        = self.Ks_ebqe
            vgm_alpha = self.vgm_alpha_ebqe
            vgm_n     = self.vgm_n_ebqe
            thetaR    = self.thetaR_ebqe
            thetaSR   = self.thetaSR_ebqe
        else:
            Ks        = self.Ks_ebq
            vgm_alpha = self.vgm_alpha_ebq
            vgm_n     = self.vgm_n_ebq
            thetaR    = self.thetaR_ebq
            thetaSR   = self.thetaSR_ebq

        conservativeHeadRichardsMualemVanGenuchtenHetEvaluate(self.rho,
                                                              self.gravity,
                                                              vgm_alpha,
                                                              vgm_n,
                                                              thetaR,
                                                              thetaSR,
                                                              Ks,
                                                              c[('u',0)],
                                                              c[('m',0)],
                                                              c[('dm',0,0)],
                                                              c[('f',0)],
                                                              c[('df',0,0)],
                                                              c[('a',0,0)],
                                                              c[('da',0,0,0)])

from .ctransportCoefficients import  conservativeHeadRichardsBrooksCoreyBurdineHetEvaluate
class ConservativeHeadRichardsBrooksCoreyBurdineHet(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 setParamsFunc):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.gravity=gravity
        self.rho = density
        self.setParamsFunc = setParamsFunc
    def initializeElementQuadrature(self,t,cq):
        self.bcb_lambda_q = numpy.zeros(cq[('u',0)].shape,'d')
        self.bcb_pd_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.Ks_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.thetaR_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.thetaS_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.setParamsFunc(cq['x'],
                           self.bcb_lambda_q,
                           self.bcb_pd_q,
                           self.Ks_q,
                           self.thetaR_q,
                           self.thetaS_q)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.bcb_lambda_ebq = numpy.zeros(cebq[('u',0)].shape,'d')
        self.bcb_pd_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.Ks_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.thetaR_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.thetaS_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.setParamsFunc(cebq['x'],
                           self.bcb_lambda_ebq,
                           self.bcb_pd_ebq,
                           self.Ks_ebq,
                           self.thetaR_ebq,
                           self.thetaS_ebq)
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.bcb_lambda_ebqe = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.bcb_pd_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.Ks_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.thetaR_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.thetaS_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.setParamsFunc(cebqe['x'],
                           self.bcb_lambda_ebqe,
                           self.bcb_pd_ebqe,
                           self.Ks_ebqe,
                           self.thetaR_ebqe,
                           self.thetaS_ebqe)
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.Ks_q.shape:
            bcb_lambda = self.bcb_lambda_q
            bcb_pd = self.bcb_pd_q
            Ks = self.Ks_q
            thetaR = self.thetaR_q
            thetaS = self.thetaS_q
        elif c[('u',0)].shape == self.Ks_ebqe.shape:
            bcb_lambda = self.bcb_lambda_ebqe
            bcb_pd = self.bcb_pd_ebqe
            Ks = self.Ks_ebqe
            thetaR = self.thetaR_ebqe
            thetaS = self.thetaS_ebqe
        elif c[('u',0)].shape == self.Ks_ebq.shape:
            bcb_lambda = self.bcb_lambda_ebq
            bcb_pd = self.bcb_pd_ebq
            Ks = self.Ks_ebq
            thetaR = self.thetaR_ebq
            thetaS = self.thetaS_ebq
        else:
            raise RuntimeError("nothing of this size in transportCoefficients " +repr(c[('u',0)].shape))
        conservativeHeadRichardsBrooksCoreyBurdineHetEvaluate(self.rho,
                                                              self.gravity,
                                                              bcb_lambda,
                                                              bcb_pd,
                                                              thetaR,
                                                              thetaS,
                                                              Ks,
                                                              c[('u',0)],
                                                              c[('m',0)],
                                                              c[('dm',0,0)],
                                                              c[('f',0)],
                                                              c[('df',0,0)],
                                                              c[('a',0,0)],
                                                              c[('da',0,0,0)])
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('df',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()


from .ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluate                
class ConservativeHeadRichardsMualemVanGenuchtenBlockHet(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 setParamsFunc):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.gravity=gravity
        self.rho = density
        self.setParamsFunc = setParamsFunc
    def initializeElementQuadrature(self,t,cq):
        self.vgm_n_q = numpy.zeros(cq[('u',0)].shape,'d')
        self.vgm_alpha_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.Ks_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.thetaR_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.thetaSR_q= numpy.zeros(cq[('u',0)].shape,'d')
        self.setParamsFunc(cq['x'],
                           self.vgm_n_q,
                           self.vgm_alpha_q,
                           self.Ks_q,
                           self.thetaR_q,
                           self.thetaSR_q)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.vgm_n_ebq = numpy.zeros(cebq[('u',0)].shape,'d')
        self.vgm_alpha_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.Ks_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.thetaR_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.thetaSR_ebq= numpy.zeros(cebq[('u',0)].shape,'d')
        self.setParamsFunc(cebq['x'],
                           self.vgm_n_ebq,
                           self.vgm_alpha_ebq,
                           self.Ks_ebq,
                           self.thetaR_ebq,
                           self.thetaSR_ebq)
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.vgm_n_ebqe = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.vgm_alpha_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.Ks_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.thetaR_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.thetaSR_ebqe= numpy.zeros(cebqe[('u',0)].shape,'d')
        self.setParamsFunc(cebqe['x'],
                           self.vgm_n_ebqe,
                           self.vgm_alpha_ebqe,
                           self.Ks_ebqe,
                           self.thetaR_ebqe,
                           self.thetaSR_ebqe)
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.Ks_q.shape:
            vgm_n     = self.vgm_n_q
            vgm_alpha = self.vgm_alpha_q
            Ks = self.Ks_q
            thetaR = self.thetaR_q
            thetaSR = self.thetaSR_q
        elif c[('u',0)].shape == self.Ks_ebqe.shape:
            vgm_n     = self.vgm_n_ebqe
            vgm_alpha = self.vgm_alpha_ebqe
            Ks = self.Ks_ebqe
            thetaR = self.thetaR_ebqe
            thetaSR = self.thetaSR_ebqe
        else:
            vgm_n     = self.vgm_n_ebq
            vgm_alpha = self.vgm_alpha_ebq
            Ks = self.Ks_ebq
            thetaR = self.thetaR_ebq
            thetaSR = self.thetaSR_ebq
        conservativeHeadRichardsMualemVanGenuchtenHetEvaluate(self.rho,
                                                              self.gravity,
                                                              vgm_alpha,
                                                              vgm_n,
                                                              thetaR,
                                                              thetaSR,
                                                              Ks,
                                                              c[('u',0)],
                                                              c[('m',0)],
                                                              c[('dm',0,0)],
                                                              c[('f',0)],
                                                              c[('df',0,0)],
                                                              c[('a',0,0)],
                                                              c[('da',0,0,0)])

from .ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2
class ConservativeHeadRichardsMualemVanGenuchtenBlockHetV2(TC_base):
    """
    version of Re where element material type id's used in evals
    """
    def __init__(self,
                 Ks_block,
                 vgm_n_block,
                 vgm_alpha_block,
                 thetaR_block,
                 thetaSR_block,
                 gravity,
                 density,
                 beta):
        variableNames=['pressure_head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.gravity=gravity
        self.rho = density
        self.beta=beta
        self.Ks_block = Ks_block
        self.vgm_n_block = vgm_n_block
        self.vgm_alpha_block = vgm_alpha_block
        self.thetaR_block    = thetaR_block
        self.thetaSR_block   = thetaSR_block
        self.elementMaterialTypes = None
        self.exteriorElementBoundaryTypes  = None
        self.materialTypes_q    = None
        self.materialTypes_ebq  = None
        self.materialTypes_ebqe  = None
        self.getSeepageFace = None
    def initializeMesh(self,mesh):
        self.elementMaterialTypes = mesh.elementMaterialTypes
        #want element boundary material types for evaluating heterogeneity
        #not boundary conditions
        self.exteriorElementBoundaryTypes = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
        self.isSeepageFace = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            eN  = mesh.elementBoundaryElementsArray[ebN,0]
            self.exteriorElementBoundaryTypes[ebNE] = self.elementMaterialTypes[eN]
        if self.getSeepageFace is not None:
            for ebNE in range(mesh.nExteriorElementBoundaries_global):
                ebN = mesh.exteriorElementBoundariesArray[ebNE]
                eN  = mesh.elementBoundaryElementsArray[ebN,0]
                self.isSeepageFace[ebNE] = self.getSeepageFace(mesh.elementBoundaryMaterialTypes[ebN])
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
        if cq[('u',0)].shape == self.q_shape:
            cq['Ks'] = numpy.zeros(self.q_shape)
            for eN in range(self.q_shape[0]):
                for k in range(self.q_shape[1]):
                    cq['Ks'][eN,k] = self.Ks_block[self.materialTypes_q[eN]]
                    #cq['Ks'][eN,k] = self.materialTypes_q[eN]
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for eN in range(self.elementMaterialTypes.shape[0]):
            self.materialTypes_ebq[eN,:] = self.elementMaterialTypes[eN]
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = numpy.zeros(cebqe[('u',0)].shape[0],'i')
        self.ebqe_shape = cebqe[('u',0)].shape
        for ebNE in range(self.exteriorElementBoundaryTypes.shape[0]):
            self.materialTypes_ebqe[ebNE] = self.exteriorElementBoundaryTypes[ebNE]
        #
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
#        for em in materialTypes:
#            print em
        conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2(materialTypes,
                                                                self.rho,
                                                                self.beta,
                                                                self.gravity,
                                                                self.vgm_alpha_block,
                                                                self.vgm_n_block,
                                                                self.thetaR_block,
                                                                self.thetaSR_block,
                                                                self.Ks_block,
                                                                c[('u',0)],
                                                                c[('m',0)],
                                                                c[('dm',0,0)],
                                                                c[('f',0)],
                                                                c[('df',0,0)],
                                                                c[('a',0,0)],
                                                                c[('da',0,0,0)])
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('df',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()

from .ctransportCoefficients import seepageBrezis
class SeepageBrezis(TC_base):
    """
    version of Re where element material type id's used in evals
    """
    def __init__(self,
                 Ks_block,
                 vgm_n_block,
                 vgm_alpha_block,
                 thetaR_block,
                 thetaSR_block,
                 gravity,
                 density,
                 beta,
                 epsFact=1.5):
        self.epsFact=epsFact
        variableNames=['pressure_head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.gravity=gravity
        self.rho = density
        self.beta=beta
        self.Ks_block = Ks_block
        self.vgm_n_block = vgm_n_block
        self.vgm_alpha_block = vgm_alpha_block
        self.thetaR_block    = thetaR_block
        self.thetaSR_block   = thetaSR_block
        self.elementMaterialTypes = None
        self.exteriorElementBoundaryTypes  = None
        self.materialTypes_q    = None
        self.materialTypes_ebq  = None
        self.materialTypes_ebqe  = None
        self.getSeepageFace = None
    def initializeMesh(self,mesh):
        self.elementDiameters = mesh.elementDiametersArray
        self.elementBoundaryDiameters = mesh.elementBoundaryDiametersArray
        self.exteriorElementBoundaryDiameters = numpy.zeros((mesh.nExteriorElementBoundaries_global,),'d')
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            self.exteriorElementBoundaryDiameters[ebNE] = mesh.elementDiametersArray[mesh.elementBoundaryElementsArray[mesh.exteriorElementBoundariesArray[ebNE],0]]
        self.elementMaterialTypes = mesh.elementMaterialTypes
        #want element boundary material types for evaluating heterogeneity
        #not boundary conditions
        self.exteriorElementBoundaryTypes = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
        self.isSeepageFace = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            eN  = mesh.elementBoundaryElementsArray[ebN,0]
            self.exteriorElementBoundaryTypes[ebNE] = self.elementMaterialTypes[eN]
        if self.getSeepageFace is not None:
            for ebNE in range(mesh.nExteriorElementBoundaries_global):
                ebN = mesh.exteriorElementBoundariesArray[ebNE]
                eN  = mesh.elementBoundaryElementsArray[ebN,0]
                self.isSeepageFace[ebNE] = self.getSeepageFace(mesh.elementBoundaryMaterialTypes[ebN])
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
        if cq[('u',0)].shape == self.q_shape:
            cq['Ks'] = numpy.zeros(self.q_shape)
            for eN in range(self.q_shape[0]):
                for k in range(self.q_shape[1]):
                    cq['Ks'][eN,k] = self.Ks_block[self.materialTypes_q[eN]]
                    #cq['Ks'][eN,k] = self.materialTypes_q[eN]
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for eN in range(self.elementMaterialTypes.shape[0]):
            self.materialTypes_ebq[eN,:] = self.elementMaterialTypes[eN]
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = numpy.zeros(cebqe[('u',0)].shape[0],'i')
        self.ebqe_shape = cebqe[('u',0)].shape
        for ebNE in range(self.exteriorElementBoundaryTypes.shape[0]):
            self.materialTypes_ebqe[ebNE] = self.exteriorElementBoundaryTypes[ebNE]
        #
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
            elementDiameters = self.elementDiameters
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
            elementDiameters = self.exteriorElementBoundaryDiameters
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
            elementDiameters = self.elementBoundaryDiameters
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        seepageBrezis(materialTypes,
                      self.epsFact,
                      self.rho,
                      self.beta,
                      elementDiameters,
                      self.gravity,
                      self.vgm_alpha_block,
                      self.vgm_n_block,
                      self.thetaR_block,
                      self.thetaSR_block,
                      self.Ks_block,
                      c[('u',0)],
                      c[('m',0)],
                      c[('dm',0,0)],
                      c[('f',0)],
                      c[('df',0,0)],
                      c[('a',0,0)],
                      c[('da',0,0,0)])

from .ctransportCoefficients import conservativeHeadRichardsJLeverettEvaluate
class ConservativeHeadRichardsJLeverett(TC_base):
    """
    version of Re where element material type id's used in evals
    """
    def __init__(self,
                 phi_block,
                 psiD_block,
                 ns_block,
                 nk_block,
                 S_wirr_block,
                 S_nwr_block,
                 kr0_block,
                 gravity,
                 density,
                 beta):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.gravity=gravity
        self.rho = density
        self.beta=beta
        self.phi_block = phi_block
        self.psiD_block = psiD_block
        self.ns_block = ns_block
        self.nk_block    = nk_block
        self.S_wirr_block   = S_wirr_block
        self.S_nwr_block   = S_nwr_block
        self.kr0_block   = kr0_block
        self.elementMaterialTypes = None
        self.exteriorElementBoundaryTypes  = None
        self.materialTypes_q    = None
        self.materialTypes_ebq  = None
        self.materialTypes_ebqe  = None
        self.debug=True
    def initializeMesh(self,mesh):
        self.elementMaterialTypes = mesh.elementMaterialTypes
        #want element boundary material types for evaluating heterogeneity
        #not boundary conditions
        self.exteriorElementBoundaryTypes = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            eN  = mesh.elementBoundaryElementsArray[ebN,0]
            self.exteriorElementBoundaryTypes[ebNE] = self.elementMaterialTypes[eN]
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for eN in range(self.elementMaterialTypes.shape[0]):
            self.materialTypes_ebq[eN,:] = self.elementMaterialTypes[eN]
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = numpy.zeros(cebqe[('u',0)].shape[0],'i')
        self.ebqe_shape = cebqe[('u',0)].shape
        for ebNE in range(self.exteriorElementBoundaryTypes.shape[0]):
            self.materialTypes_ebqe[ebNE] = self.exteriorElementBoundaryTypes[ebNE]
        #
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        if self.debug:
            if numpy.isnan(c[('u',0)]).any():
                import pdb
                print("NaN's on input")
                pdb.set_trace()
        conservativeHeadRichardsJLeverettEvaluate(materialTypes,
                                                  self.rho,
                                                  self.beta,
                                                  self.gravity,
                                                  self.phi_block,
                                                  self.psiD_block,
                                                  self.ns_block,
                                                  self.nk_block,
                                                  self.S_wirr_block,
                                                  self.S_nwr_block,
                                                  self.kr0_block,
                                                  c[('u',0)],
                                                  c[('m',0)],
                                                  c[('dm',0,0)],
                                                  c[('f',0)],
                                                  c[('df',0,0)],
                                                  c[('a',0,0)],
                                                  c[('da',0,0,0)])
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('df',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()

        if c[('u',0)].shape == self.q_shape:
            c[('visPerm',0)]=c[('a',0,0)][:,:,0,0]

from .ctransportCoefficients import conservativeHeadRichardsJLeverettAniEvaluate
class ConservativeHeadRichardsJLeverettAni(TC_base):
    """
    version of Re where element material type id's used in evals
    """
    def __init__(self,
                 phi_block,
                 psiD_block,
                 ns_block,
                 nk_block,
                 S_wirr_block,
                 S_nwr_block,
                 kr0x_block,
                 kr0y_block,
                 kr0z_block,
                 gravity,
                 density,
                 beta):
        variableNames=['Pressure_Head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.gravity=gravity
        self.rho = density
        self.beta=beta
        self.phi_block = phi_block
        self.psiD_block = psiD_block
        self.ns_block = ns_block
        self.nk_block    = nk_block
        self.S_wirr_block   = S_wirr_block
        self.S_nwr_block   = S_nwr_block
        self.kr0x_block   = kr0x_block
        self.kr0y_block   = kr0y_block
        self.kr0z_block   = kr0z_block
        self.elementMaterialTypes = None
        self.exteriorElementBoundaryTypes  = None
        self.materialTypes_q    = None
        self.materialTypes_ebq  = None
        self.materialTypes_ebqe  = None
        self.debug=True
    def initializeMesh(self,mesh):
        self.elementMaterialTypes = mesh.elementMaterialTypes
        #want element boundary material types for evaluating heterogeneity
        #not boundary conditions
        self.exteriorElementBoundaryTypes = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            eN  = mesh.elementBoundaryElementsArray[ebN,0]
            self.exteriorElementBoundaryTypes[ebNE] = self.elementMaterialTypes[eN]
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for eN in range(self.elementMaterialTypes.shape[0]):
            self.materialTypes_ebq[eN,:] = self.elementMaterialTypes[eN]
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = numpy.zeros(cebqe[('u',0)].shape[0],'i')
        self.ebqe_shape = cebqe[('u',0)].shape
        for ebNE in range(self.exteriorElementBoundaryTypes.shape[0]):
            self.materialTypes_ebqe[ebNE] = self.exteriorElementBoundaryTypes[ebNE]
        #
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        if self.debug:
            if numpy.isnan(c[('u',0)]).any():
                import pdb
                print("NaN's on input")
                pdb.set_trace()
        conservativeHeadRichardsJLeverettAniEvaluate(materialTypes,
                                                     self.rho,
                                                     self.beta,
                                                     self.gravity,
                                                     self.phi_block,
                                                     self.psiD_block,
                                                     self.ns_block,
                                                     self.nk_block,
                                                     self.S_wirr_block,
                                                     self.S_nwr_block,
                                                     self.kr0x_block,
                                                     self.kr0y_block,
                                                     self.kr0z_block,
                                                     c[('u',0)],
                                                     c[('m',0)],
                                                     c[('dm',0,0)],
                                                     c[('f',0)],
                                                     c[('df',0,0)],
                                                     c[('a',0,0)],
                                                     c[('da',0,0,0)])
        if (numpy.isnan(c[('da',0,0,0)]).any() or
            numpy.isnan(c[('a',0,0)]).any() or
            numpy.isnan(c[('df',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()

        if c[('u',0)].shape == self.q_shape:
            c[('visPerm',0)]=c[('a',0,0)][:,:,0,0]

from .ctransportCoefficients import constantVelocityLevelSetEvaluate
class ConstantVelocityLevelSet(TC_base):
    def __init__(self,b=[1.0,0.0],lsModelId=0):
        self.b=b
        mass={0:{0:'linear'}}
        advection={0:{0:'linear'}}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={0:{0:'linear'}}
        TC_base.__init__(self,
                         1,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
        self.lsModelId = lsModelId
        self.lsModel= None  #the level set model itself
        self.tLast = -12345.0  #the last time step seen
    def attachModels(self,modelList):
        if len(modelList) > 1:
            self.lsModel = modelList[self.lsModelId]
    def evaluate(self,t,c):
        constantVelocityLevelSetEvaluate(self.b,
                                         c['x'],
                                         c[('u',0)],
                                         c[('grad(u)',0)],
                                         c[('m',0)],
                                         c[('dm',0,0)],
                                         c[('f',0)],
                                         c[('df',0,0)],
                                         c[('H',0)],
                                         c[('dH',0,0)]);

    #end def

from .ctransportCoefficients import unitSquareVortexLevelSetEvaluate
class UnitSquareVortexLevelSet(TC_base):
    def __init__(self):
        mass={0:{0:'linear'}}
        advection={0:{0:'linear'}}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={0:{0:'linear'}}
        TC_base.__init__(self,
                         1,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
    def evaluate(self,t,c):
        unitSquareVortexLevelSetEvaluate(t,
                                         c['x'],
                                         c[('u',0)],
                                         c[('grad(u)',0)],
                                         c[('m',0)],
                                         c[('dm',0,0)],
                                         c[('f',0)],
                                         c[('df',0,0)],
                                         c[('H',0)],
                                         c[('dH',0,0)]);

    #end def

from .ctransportCoefficients import unitSquareRotationLevelSetEvaluate
class RotatingVelocityLevelSet(TC_base):
    def __init__(self):
        mass={0:{0:'linear'}}
        advection={0:{0:'linear'}}
        diffusion={}
        potential={}
        reaction={0:{0:'constant'}}
        hamiltonian={0:{0:'linear'}}
        TC_base.__init__(self,
                                             1,
                                             mass,
                                             advection,
                                             diffusion,
                                             potential,
                                             reaction,
                                             hamiltonian)
    def evaluate(self,t,c):
        unitSquareRotationLevelSetEvaluate(t,
                                           c['x'],
                                           c[('u',0)],
                                           c[('grad(u)',0)],
                                           c[('m',0)],
                                           c[('dm',0,0)],
                                           c[('f',0)],
                                           c[('df',0,0)],
                                           c[('H',0)],
                                           c[('dH',0,0)]);

        #make sure r gets set to zero if using general HJ form
        if ('r',0) in list(c.keys()):
            c[('r',0)].flat[:] = 0.0
    #end def

from .ctransportCoefficients import eikonalEquationEvaluate
class EikonalEquationCoefficients(TC_base):
    def __init__(self,rhsval=1.0):
        mass={0:{0:'linear'}}
        advection={}
        diffusion={}
        potential={}
        reaction={0:{0:'constant'}}
        hamiltonian={0:{0:'nonlinear'}}
        TC_base.__init__(self,
                         1,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
        self.rhsval = rhsval
    def evaluate(self,t,c):
        eikonalEquationEvaluate(self.rhsval,
                                c[('u',0)],
                                c[('grad(u)',0)],
                                c[('m',0)],
                                c[('dm',0,0)],
                                c[('H',0)],
                                c[('dH',0,0)],
                                c[('r',0)]);

    #end def


from proteus.ctransportCoefficients import redistanceLevelSetCoefficientsEvaluate
class RedistanceLevelSet(TC_base):
    def __init__(self,applyRedistancing=True,epsFact=2.0,nModelId=None,u0=None,rdModelId=0,penaltyParameter=0.0):
        variableNames=['phid']
        nc=1
        mass={0:{0:'linear'}}
        hamiltonian={0:{0:'nonlinear'}}
        advection={}
        diffusion={}
        potential={}
        reaction={0:{0:'constant'}}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.nModelId = nModelId
        self.rdModelId= rdModelId
        self.epsFact=epsFact
        self.q_u0   = None
        self.ebq_u0 = None
        self.ebqe_u0= None
        self.dof_u0 = None
        self.u0 = u0
        self.applyRedistancing = applyRedistancing
        self.weakBC_on=True#False
        self.penaltyParameter=penaltyParameter
    def attachModels(self,modelList):
        if self.nModelId is not None:
            self.nModel = modelList[self.nModelId]
            self.q_u0 =   self.nModel.q[('u',0)]
            if ('u',0) in self.nModel.ebq:
                self.ebq_u0 = self.nModel.ebq[('u',0)]
            self.ebqe_u0 =   self.nModel.ebqe[('u',0)]
            self.dof_u0 = self.nModel.u[0].dof
        else:
            self.nModel = None
        self.rdModel = modelList[self.rdModelId]
    def initializeMesh(self,mesh):
        self.h=mesh.h
        self.eps = self.epsFact*mesh.h
    def initializeElementQuadrature(self,t,cq):
        if self.nModelId is None:
            if self.q_u0 is None:
                self.q_u0 = numpy.zeros(cq[('u',0)].shape,'d')
            if self.u0 is not None:
                for i in range(len(cq[('u',0)].flat)):
                    self.q_u0.flat[i]=self.u0.uOfXT(cq['x'].flat[3*i:3*(i+1)],0.)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        if self.nModelId is None:
            if self.ebq_u0 is None:
                self.ebq_u0 = numpy.zeros(cebq[('u',0)].shape,'d')
            if self.u0 is not None:
                for i in range(len(cebq[('u',0)].flat)):
                    self.ebq_u0.flat[i]=self.u0.uOfXT(cebq['x'].flat[3*i:3*(i+1)],0.)
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        if self.nModelId is None:
            if self.ebqe_u0 is None:
                self.ebqe_u0 = numpy.zeros(cebqe[('u',0)].shape,'d')
            if self.u0 is not None:
                for i in range(len(cebqe[('u',0)].flat)):
                    self.ebqe_u0.flat[i]=self.u0.uOfXT(cebqe['x'].flat[3*i:3*(i+1)],0.)
    def preStep(self,t,firstStep=False):
        if self.nModel is not None:
            logEvent("resetting signed distance level set to current level set",level=2)
            self.rdModel.u[0].dof[:] = self.nModel.u[0].dof[:]
            self.rdModel.calculateCoefficients()
            self.rdModel.calculateElementResidual()
            self.rdModel.timeIntegration.updateTimeHistory(resetFromDOF=True)
            self.rdModel.timeIntegration.resetTimeHistory(resetFromDOF=True)
            self.rdModel.updateTimeHistory(t,resetFromDOF=True)
            #now do again because of subgrid error lagging
            #\todo modify subgrid error lagging so this won't be necessary
            self.rdModel.calculateCoefficients()
            self.rdModel.calculateElementResidual()
            self.rdModel.timeIntegration.updateTimeHistory(resetFromDOF=True)
            self.rdModel.timeIntegration.resetTimeHistory(resetFromDOF=True)
            self.rdModel.updateTimeHistory(t,resetFromDOF=True)
            copyInstructions = {'copy_uList':True,
                                'uList_model':self.nModelId}
            copyInstructions = {'reset_uList':True}
            return copyInstructions
        else:
            return {}
    def postStep(self,t,firstStep=False):
        if self.nModel is not None:
            if self.applyRedistancing == True:
                logEvent("resetting level set to signed distance")
                self.nModel.u[0].dof.flat[:]  = self.rdModel.u[0].dof.flat[:]
                self.nModel.calculateCoefficients()
                self.nModel.calculateElementResidual()
            copyInstructions = {}
            return copyInstructions
        else:
            return {}
    def getICDofs(self,cj):
        return self.dof_u0
    def updateToMovingDomain(self,t,c):
        #the redistancing equations is not physical so it just needs the updated mesh
        pass
    def evaluate(self,t,c):
        if c[('H',0)].shape == self.q_u0.shape:
            u0 = self.q_u0
            #print "numdiff",c[('numDiff',0,0)].min(),c[('numDiff',0,0)].max()
            #print "tau",self.rdModel.stabilization.tau[0].min(),self.rdModel.stabilization.tau[0].max(),
        elif c[('H',0)].shape == self.ebqe_u0.shape:
            u0 = self.ebqe_u0
        else:
            u0 = self.ebq_u0
        assert u0 is not None
        ##\todo make redistancing epsilon depend on local element diamater instead of global max
        redistanceLevelSetCoefficientsEvaluate(self.eps,
                                               u0,
                                               c[('u',0)],
                                               c[('grad(u)',0)],
                                               c[('m',0)],
                                               c[('dm',0,0)],
                                               c[('H',0)],
                                               c[('dH',0,0)],
                                               c[('r',0)])
    #weak Dirichlet conditions on level set (boundary conditions of Eikonal equation)
    ##\todo clean up weak Dirichlet conditions for Eikonal equation in transport coefficents
    def setZeroLSweakDirichletBCs(vt):
        if vt.coefficients.weakBC_on:
            #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!=========================Setting new weak BC's=======================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            vt.dirichletNodeSetList[0] = []
            vt.dirichletGlobalNodeSet[0]=set()
            vt.dirichletValues[0]={}
            for eN in range(vt.mesh.nElements_global):
                vt.dirichletNodeSetList[0].append(set())
                signU = 0
                j0=0
                eps = vt.u[0].femSpace.mesh.elementDiametersArray[eN]
                #loop over nodes looking for a node not within eps of zero
                while ((signU == 0) and
                       (j0 < vt.nDOF_trial_element[0])):
                    J0 = vt.u[0].femSpace.dofMap.l2g[eN,j0]
                    if vt.u[0].dof[J0] < -eps:
                        signU = -1
                    elif  vt.u[0].dof[J0] > eps:
                        signU = 1
                    else: #freeze this node within eps of zero
                        vt.dirichletNodeSetList[0][eN].add(j0)
                        vt.dirichletValues[0][(eN,j0)]=vt.u[0].dof[J0]
                        vt.dirichletGlobalNodeSet[0].add(J0)
                    j0 += 1
                #loop over remaining nodes to see if the zero level set cuts element
                for j in range(j0,vt.nDOF_trial_element[0]):
                    J = vt.u[0].femSpace.dofMap.l2g[eN,j]
                    if (((vt.u[0].dof[J] < -eps) and
                         (signU == 1)) or
                        ((vt.u[0].dof[J] > eps) and
                         (signU == -1))): #level set cuts element, freeze whole element
                        for jj in range(vt.nDOF_trial_element[0]):
                            JJ = vt.u[0].femSpace.dofMap.l2g[eN,jj]
                            vt.dirichletNodeSetList[0][eN].add(jj)
                            vt.dirichletValues[0][(eN,jj)]=float(vt.u[0].dof[JJ])
                            vt.dirichletGlobalNodeSet[0].add(JJ)
                        break
                    elif (fabs(vt.u[0].dof[J]) < eps):#freeze this node within eps of zero
                        vt.dirichletNodeSetList[0][eN].add(j)
                        vt.dirichletValues[0][(eN,j)]=float(vt.u[0].dof[J])
                        vt.dirichletGlobalNodeSet[0].add(J)
            #get all frozen dof and make sure they're frozen on each element
            for eN in range(vt.mesh.nElements_global):
                for j in range(vt.nDOF_trial_element[0]):
                    J = vt.u[0].femSpace.dofMap.l2g[eN,j]
                    if J in vt.dirichletGlobalNodeSet[0]:
                        vt.dirichletNodeSetList[0][eN].add(j)
                        vt.dirichletValues[0][(eN,j)]=float(vt.u[0].dof[J])
        else:
            #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!=========================Unsetting weak BC's=======================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            vt.dirichletNodeSetList[0] = []
            vt.dirichletGlobalNodeSet[0]=set()
            vt.dirichletValues[0]={}
            for eN in range(vt.mesh.nElements_global):
                vt.dirichletNodeSetList[0].append(set())
    def setZeroLSweakDirichletBCs2(vt):
        #just look for cut edges and nodes
        vt.dirichletNodeSetList[0] = []
        vt.dirichletGlobalNodeSet[0]=set()
        vt.dirichletValues[0]={}
        for eN in range(vt.mesh.nElements_global):
            vt.dirichletNodeSetList[0].append(set())
            signU = 0
            j0=0
            eps = .1*vt.u[0].femSpace.mesh.elementDiametersArray[eN]
            eps = 0.0
            for j0 in range(vt.nDOF_trial_element[0]):
                J0 = vt.u[0].femSpace.dofMap.l2g[eN,j0]
                if vt.u[0].dof[J0] < -eps:
                    signU = -1
                elif  vt.u[0].dof[J0] > eps:
                    signU = 1
                else:
                    vt.dirichletNodeSetList[0][eN].add(j0)
                    vt.dirichletValues[0][(eN,j0)]=float(vt.u[0].dof[J0])
                    vt.dirichletGlobalNodeSet[0].add(J0)
                if signU != 0:
                    for j in (list(range(0,j0))+list(range(j0+1,vt.nDOF_trial_element[0]))):
                        J = vt.u[0].femSpace.dofMap.l2g[eN,j]
                        if (((vt.u[0].dof[J] < -eps) and
                             (signU == 1)) or
                            ((vt.u[0].dof[J] > eps) and
                             (signU == -1))):
                            vt.dirichletNodeSetList[0][eN].add(j)
                            vt.dirichletValues[0][(eN,j)]=float(vt.u[0].dof[J])
                            vt.dirichletGlobalNodeSet[0].add(J)
                            vt.dirichletNodeSetList[0][eN].add(j0)
                            vt.dirichletValues[0][(eN,j0)]=float(vt.u[0].dof[j0])
                            vt.dirichletGlobalNodeSet[0].add(j0)
        for eN in range(vt.mesh.nElements_global):
            for j in range(vt.nDOF_trial_element[0]):
                J = vt.u[0].femSpace.dofMap.l2g[eN,j]
                if J in vt.dirichletGlobalNodeSet[0]:
                    vt.dirichletNodeSetList[0][eN].add(j)
                    vt.dirichletValues[0][(eN,j)]=float(vt.u[0].dof[J])
    def setZeroLSweakDirichletBCs3(vt):
        if vt.coefficients.weakBC_on:
            #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!=========================Setting new weak BC's=======================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            vt.dirichletNodeSetList[0] = []
            vt.dirichletGlobalNodeSet[0]=set()
            vt.dirichletValues[0]={}
            for eN in range(vt.mesh.nElements_global):
                vt.dirichletNodeSetList[0].append(set())
                eps = vt.coefficients.epsFact*vt.u[0].femSpace.mesh.elementDiametersArray[eN]
                #eps = 1.5*vt.u[0].femSpace.mesh.elementDiametersArray[eN]
                #loop over nodes looking for a node not within eps of zero
                for j in range(vt.nDOF_trial_element[0]):
                    J = vt.u[0].femSpace.dofMap.l2g[eN,j]
                    if (fabs(vt.u[0].dof[J]) < eps):#freeze this node within eps of zero
                        vt.dirichletNodeSetList[0][eN].add(j)
                        vt.dirichletValues[0][(eN,j)]=float(vt.u[0].dof[J])
                        vt.dirichletGlobalNodeSet[0].add(J)
        else:
            #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!=========================Unsetting weak BC's=======================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            vt.dirichletNodeSetList[0] = []
            vt.dirichletGlobalNodeSet[0]=set()
            vt.dirichletValues[0]={}
            for eN in range(vt.mesh.nElements_global):
                vt.dirichletNodeSetList[0].append(set())

    #def
    setZeroLSweakDirichletBCs = staticmethod(setZeroLSweakDirichletBCs)
    setZeroLSweakDirichletBCs2 = staticmethod(setZeroLSweakDirichletBCs2)
    setZeroLSweakDirichletBCs3 = staticmethod(setZeroLSweakDirichletBCs3)
#end class
class RedistanceLevelSetWithWeakPenalty(RedistanceLevelSet):
    from proteus.ctransportCoefficients import redistanceLevelSetCoefficientsWithWeakPenaltyEvaluate
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    redistanceLevelSetCoefficientsWithWeakPenaltyEvaluate = staticmethod(redistanceLevelSetCoefficientsWithWeakPenaltyEvaluate)
    def __init__(self,applyRedistancing=True,epsFact=2.0,penaltyParameter=1.0,nModelId=None,u0=None,rdModelId=0):
        variableNames=['phid']
        nc=1
        mass={0:{0:'linear'}}
        hamiltonian={0:{0:'nonlinear'}}
        advection={}
        diffusion={}
        potential={}
        reaction={0:{0:'nonlinear'}}#now have quadratic penalty
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames)
        self.nModelId = nModelId
        self.rdModelId= rdModelId
        self.epsFact=epsFact
        self.penaltyParameter=penaltyParameter
        self.q_u0   = None
        self.ebq_u0 = None
        self.ebqe_u0= None
        self.dof_u0 = None
        self.u0 = u0
        self.applyRedistancing = applyRedistancing
        self.weakBC_on=False
    def evaluate(self,t,c):
        if c[('H',0)].shape == self.q_u0.shape:
            u0 = self.q_u0
            #print "numdiff",c[('numDiff',0,0)].min(),c[('numDiff',0,0)].max()
            #print "tau",self.rdModel.stabilization.tau[0].min(),self.rdModel.stabilization.tau[0].max(),
        elif c[('H',0)].shape == self.ebqe_u0.shape:
            u0 = self.ebqe_u0
        else:
            u0 = self.ebq_u0
        assert u0 is not None
        ##\todo make redistancing epsilon depend on local element diamater instead of global max
        redistanceLevelSetCoefficientsWithWeakPenaltyEvaluate(self.eps,
                                                              self.penaltyParameter,
                                                              u0,
                                                              c[('u',0)],
                                                              c[('grad(u)',0)],
                                                              c[('m',0)],
                                                              c[('dm',0,0)],
                                                              c[('H',0)],
                                                              c[('dH',0,0)],
                                                              c[('r',0)],
                                                              c[('dr',0,0)])

from proteus.ctransportCoefficients import redistanceLevelSetSandFCoefficientsEvaluate
class RedistanceLevelSetSandF(RedistanceLevelSet):
    def __init__(self,applyRedistancing=True,epsFact=2.0,nModelId=1,u0=None,rdModelId=-1,vofModelId=4,massCorrModelId=5,
                 checkMass=True):
        RedistanceLevelSet.__init__(self,applyRedistancing=applyRedistancing,epsFact=epsFact,
                                    nModelId=nModelId,u0=u0,rdModelId=rdModelId,
                                    vofModelId=vofModelId,massCorrModelId=massCorrModelId,
                                    checkMass=checkMass)
    #init
    def evaluate(self,t,c):
        quadWeights= None
        if c[('H',0)].shape == self.q_u0.shape:
            u0 = self.q_u0
            quadWeights = c[('dV_u',0)]
        elif c[('H',0)].shape == self.ebqe_u0.shape:
            u0 = self.ebqe_u0
            quadWeights = c[('dS_u',0)]
        else:
            u0 = self.ebq_u0
            quadWeights = c[('dS_u',0)]
        redistanceLevelSetSandFCoefficientsEvaluate(self.eps,
                                                    u0,
                                                    quadWeights,
                                                    c[('u',0)],
                                                    c[('grad(u)',0)],
                                                    c[('m',0)],
                                                    c[('dm',0,0)],
                                                    c[('H',0)],
                                                    c[('dH',0,0)],
                                                    c[('r',0)])


# class RedistanceLevelSetWithWeakPenalty(RedistanceLevelSet):
#     def __init__(self,applyRedistancing=True,epsFact=2.0,nModelId=None,u0=None,rdModelId=0):
#         variableNames=['phid']
#         nc=1
#         mass={0:{0:'linear'}}
#         hamiltonian={0:{0:'nonlinear'}}
#         advection={}
#         diffusion={}
#         potential={}
#         reaction={0:{0:'linear'}}#only difference with RedistanceLevelSet
#         TC_base.__init__(self,
#                          nc,
#                          mass,
#                          advection,
#                          diffusion,
#                          potential,
#                          reaction,
#                          hamiltonian,
#                          variableNames)
#         self.nModelId = nModelId
#         self.rdModelId= rdModelId
#         self.epsFact=epsFact
#         self.q_u0   = None
#         self.ebq_u0 = None
#         self.ebqe_u0= None
#         self.dof_u0 = None
#         self.u0 = u0
#         self.applyRedistancing = applyRedistancing
#         self.weakBC_on=False#off by default

#         def evaluate(self,t,c):
#             self.redistanceLevelSetCoefficientsWithWeakPenaltyEvaluate(self.eps,
#                                                                        u0,
#                                                                        c[('u',0)],
#                                                                        c[('grad(u)',0)],
#                                                                        c[('m',0)],
#                                                                        c[('dm',0,0)],
#                                                                        c[('H',0)],
#                                                                        c[('dH',0,0)],
#                                                                        c[('r',0)],
#                                                                        c[('dr',0)])

from proteus.ctransportCoefficients import  conservativeHeadRichardsMualemVanGenuchtenHomEvaluate
class ConservativeHead2PMualemVanGenuchten(TC_base):
    def __init__(self,
                 hydraulicConductivity,
                 gravity,
                 density,
                 thetaS,
                 thetaR,
                 alpha,
                 n,
                 m):
        variableNames=['psi_w','psi_n']
        nc=2
        mass={0:{0:'nonlinear'},1:{1:'nonlinear'}}
        advection={0:{0:'nonlinear'},1:{1:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}},1:{1:{1:'nonlinear'}}}
        potential={0:{0:'u'},1:{1:'u'}}
        reaction={0:{0:'linear'},1:{1:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                                             nc,
                                             mass,
                                             advection,
                                             diffusion,
                                             potential,
                                             reaction,
                                             hamiltonian,
                                             variableNames)
        self.Ks = hydraulicConductivity
        self.gravity=gravity
        self.rho = density
        self.thetaS = thetaS
        self.thetaR = thetaR
        self.thetaSR = thetaS - thetaR
        self.alpha = alpha
        self.n = n
        self.m = m
    def evaluate(self,t,c):
        conservativeHeadRichardsMualemVanGenuchtenHomEvaluate(self.rho,
                                                              self.gravity,
                                                              self.alpha,
                                                              self.n,
                                                              self.m,
                                                              self.thetaR,
                                                              self.thetaSR,
                                                              self.Ks,
                                                              c[('u',0)],
                                                              c[('m',0)],
                                                              c[('dm',0,0)],
                                                              c[('f',0)],
                                                              c[('df',0,0)],
                                                              c[('a',0,0)],
                                                              c[('da',0,0,0)])
        conservativeHeadRichardsMualemVanGenuchtenHomEvaluate(self.rho,
                                                              self.gravity,
                                                              self.alpha,
                                                              self.n,
                                                              self.m,
                                                              self.thetaR,
                                                              self.thetaSR,
                                                              self.Ks,
                                                              c[('u',1)],
                                                              c[('m',1)],
                                                              c[('dm',1,1)],
                                                              c[('f',1)],
                                                              c[('df',1,1)],
                                                              c[('a',1,1)],
                                                              c[('da',1,1,1)])

from proteus.ctransportCoefficients import L2projectEvaluate

class PoissonEquationCoefficients(TC_base):
    def __init__(self,aOfX,fOfX,nc=1,nd=2,l2proj=None,
                 timeVaryingCoefficients=False):
        self.aOfX = aOfX
        self.fOfX = fOfX
        self.nd = nd
        self.l2proj = l2proj
        self.timeVaryingCoefficients=timeVaryingCoefficients
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction  = {}
        hamiltonian = {}
        for i in range(nc):
            diffusion[i] = {i : {i:'constant'}}
            reaction[i]  = {i : 'constant'}
            advection[i] = {i : 'constant'} #now include for gravity type terms
            potential[i] = {i : 'u'}
        #end i
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         sparseDiffusionTensors={})
    def initializeElementQuadrature(self,t,cq):
        nd = self.nd
        for ci in range(self.nc):
            cq[('f',ci)].flat[:] = 0.0
            for i in range(len(cq[('r',ci)].flat)):
                cq[('r',ci)].flat[i] = -self.fOfX[ci](cq['x'].flat[3*i:3*(i+1)])
                cq[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[ci](cq['x'].flat[3*i:3*(i+1)]).flat

            if self.l2proj is not None and self.l2proj[ci] == True:
                if ('dV_u',ci) in cq:
                    assert cq[('r',ci)].shape == cq[('dV_u',ci)].shape, "wrong scalar shape"
                    L2projectEvaluate(0,cq[('dV_u',ci)],cq[('r',ci)])
                    L2projectEvaluate(1,cq[('dV_u',ci)],cq[('f',ci)])
                    L2projectEvaluate(2,cq[('dV_u',ci)],cq[('a',ci,ci)])
            #ci
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        nd = self.nd
        for c in [cebq,cebq_global]:
            for ci in range(self.nc):
                if ('f',ci) in c: c[('f',ci)].flat[:] = 0.0
                if ('r',ci) in c and ('a',ci,ci) in c:
                    for i in range(len(c[('u',ci)].flat)):
                        c[('r',ci)].flat[i] = -self.fOfX[ci](c['x'].flat[3*i:3*(i+1)])
                        c[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[ci](c['x'].flat[3*i:3*(i+1)]).flat
                    if self.l2proj is not None and self.l2proj[ci] == True:
                        assert c[('r',ci)].shape == c[('dS_u',ci)].shape, "wrong scalar shape"
                        L2projectEvaluate(0,c[('dS_u',ci)],c[('r',ci)])
                        L2projectEvaluate(1,c[('dS_u',ci)],c[('f',ci)])
                        L2projectEvaluate(2,c[('dS_u',ci)],c[('a',ci,ci)])
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        nd = self.nd
        for c in [cebqe]:
            for ci in range(self.nc):
                if ('f',ci) in c: c[('f',ci)].flat[:] = 0.0
                if ('r',ci) in c and ('a',ci,ci) in c:
                    for i in range(len(c[('u',ci)].flat)):
                        c[('r',ci)].flat[i] = -self.fOfX[ci](c['x'].flat[3*i:3*(i+1)])
                        c[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[ci](c['x'].flat[3*i:3*(i+1)]).flat
                    if self.l2proj is not None and self.l2proj[ci] == True:
                        assert c[('r',ci)].shape == c[('dS_u',ci)].shape, "wrong scalar shape"
                        L2projectEvaluate(0,c[('dS_u',ci)],c[('r',ci)])
                        L2projectEvaluate(1,c[('dS_u',ci)],c[('f',ci)])
                        L2projectEvaluate(2,c[('dS_u',ci)],c[('a',ci,ci)])

    def evaluate(self,t,c):
        if self.timeVaryingCoefficients:
            nd = self.nd
            for ci in range(self.nc):
                c[('f',ci)].flat[:] = 0.0
                for i in range(len(c[('r',ci)].flat)):
                    c[('r',ci)].flat[i] = -self.fOfX[ci](c['x'].flat[3*i:3*(i+1)])
                    c[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[ci](c['x'].flat[3*i:3*(i+1)]).flat
                #end i
                if self.l2proj is not None and self.l2proj[ci] == True:
                    if ('dV_u',ci) in c:
                        assert c[('r',ci)].shape == c[('dV_u',ci)].shape, "wrong scalar shape"
                        L2projectEvaluate(0,c[('dV_u',ci)],c[('r',ci)])
                        L2projectEvaluate(1,c[('dV_u',ci)],c[('f',ci)])
                        L2projectEvaluate(2,c[('dV_u',ci)],c[('a',ci,ci)])
                    else:
                        assert c[('r',ci)].shape == c[('dS_u',ci)].shape, "wrong scalar shape"
                        L2projectEvaluate(0,c[('dS_u',ci)],c[('r',ci)])
                        L2projectEvaluate(1,c[('dS_u',ci)],c[('f',ci)])
                        L2projectEvaluate(2,c[('dS_u',ci)],c[('a',ci,ci)])
                #ci
            #end ci
    #end def

##\brief Linear Elasticity
#
from .ctransportCoefficients import LinearElasticity_1D_Evaluate
from .ctransportCoefficients import LinearElasticity_2D_Evaluate
from .ctransportCoefficients import LinearElasticity_3D_Evaluate
class LinearElasticity(TC_base):
    def __init__(self,E=1.0,nu=0.75,g=[0.0,9.8],nd=2):
        self.E = E
        self.nu = nu
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==1:
            variableNames=['hx']
            diffusion = {0:{0:{0:'constant'}}}
            potential = {0:{0:'hx'}}
            reaction = {0:{0:'constant'}}
            TC_base.__init__(self,
                             1,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
        elif nd==2:
            variableNames=['hx','hy']
            diffusion = {0:{0:{0:'constant'},
                            1:{1:'constant'}},
                         1:{0:{0:'constant'},
                            1:{1:'constant'}}}
            potential = {0:{0:'u'},
                         1:{1:'u'}}
            reaction = {0:{0:'constant'},
                        1:{1:'constant'}}
            TC_base.__init__(self,
                             2,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[0,1]
        elif nd==3:
            variableNames=['hx','hy','hz']
            diffusion = {0:{0:{0:'constant'},
                            1:{1:'constant'},
                            2:{2:'constant'}},
                         1:{0:{0:'constant'},
                            1:{1:'constant'},
                            2:{2:'constant'}},
                         2:{0:{0:'constant'},
                            1:{1:'constant'},
                            2:{2:'constant'}}}
            potential = {0:{0:'u'},
                         1:{1:'u'},
                         2:{2:'u'}}
            reaction = {0:{0:'constant'},
                        1:{1:'constant'},
                        2:{2:'constant'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[0,1,2]
        self.vectorName="displacement"
    def evaluate(self,t,c):
        if self.nd==1:
            LinearElasticity_1D_Evaluate(self.E,
                                         self.nu,
                                         self.g,
                                         c[('u',0)],
                                         c[('a',0,0)],
                                         c[('r',0)])
        elif self.nd==2:
            LinearElasticity_2D_Evaluate(self.E,
                                         self.nu,
                                         self.g,
                                         c[('u',0)],
                                         c[('u',1)],
                                         c[('a',0,0)],c[('a',0,1)],
                                         c[('a',1,0)],c[('a',1,1)],
                                         c[('r',0)],
                                         c[('r',1)])
        elif self.nd==3:
            LinearElasticity_3D_Evaluate(self.E,
                                         self.nu,
                                         self.g,
                                         c[('u',0)],
                                         c[('u',1)],
                                         c[('u',2)],
                                         c[('a',0,0)],c[('a',0,1)],c[('a',0,2)],
                                         c[('a',1,0)],c[('a',1,1)],c[('a',1,2)],
                                         c[('a',2,0)],c[('a',2,1)],c[('a',2,2)],
                                         c[('r',0)],
                                         c[('r',1)],
                                         c[('r',2)])
from .ctransportCoefficients import MovingMesh_1D_Evaluate
from .ctransportCoefficients import MovingMesh_2D_Evaluate
from .ctransportCoefficients import MovingMesh_3D_Evaluate
class MovingMesh(TC_base):
    def __init__(self,E=1.0,nu=0.3,g=[0.0,0.0],nd=2,moveMesh=True):
        self.moveMesh=moveMesh
        self.E = E
        self.nu = nu
        self.g = numpy.array(g)
        self.nd=nd
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==1:
            variableNames=['hx']
            diffusion = {0:{0:{0:'constant'}}}
            potential = {0:{0:'hx'}}
            reaction = {0:{0:'constant'}}
            TC_base.__init__(self,
                             1,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
        elif nd==2:
            variableNames=['hx','hy']
            diffusion = {0:{0:{0:'constant'},
                            1:{1:'constant'}},
                         1:{0:{0:'constant'},
                            1:{1:'constant'}}}
            potential = {0:{0:'u'},
                         1:{1:'u'}}
            reaction = {0:{0:'constant'},
                        1:{1:'constant'}}
            TC_base.__init__(self,
                             2,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
        elif nd==3:
            variableNames=['hx','hy','hz']
            diffusion = {0:{0:{0:'constant'},
                            1:{1:'constant'},
                            2:{2:'constant'}},
                         1:{0:{0:'constant'},
                            1:{1:'constant'},
                            2:{2:'constant'}},
                         2:{0:{0:'constant'},
                            1:{1:'constant'},
                            2:{2:'constant'}}}
            potential = {0:{0:'u'},
                         1:{1:'u'},
                         2:{2:'u'}}
            reaction = {0:{0:'constant'},
                        1:{1:'constant'},
                        2:{2:'constant'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
    def initializeMesh(self,mesh):
        self.mesh = mesh
    def attachModels(self,modelList):
        self.modelList = modelList
    def postStep(self,t,firstStep=False):
        if self.moveMesh:
#             for nN in range(self.mesh.nNodes_global):
#                 self.mesh.nodeArray[nN,0]+=self.modelList[-1].u[0].dof[nN]
#                 self.mesh.nodeArray[nN,1]+=self.modelList[-1].u[1].dof[nN]
            self.mesh.nodeArray[:,0]+=self.modelList[-1].u[0].dof
            self.mesh.nodeArray[:,1]+=self.modelList[-1].u[1].dof
            self.mesh.computeGeometricInfo()
        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        if 'abs(det(J))' in c:
            det_J = c['abs(det(J))']
        else:
            det_J = c['sqrt(det(g))']
        if self.nd==1:
            MovingMesh_1D_Evaluate(self.E,
                                   self.nu,
                                   self.g,
                                   det_J,
                                   c[('u',0)],
                                   c[('a',0,0)],
                                   c[('r',0)])
        elif self.nd==2:
            MovingMesh_2D_Evaluate(self.E,
                                   self.nu,
                                   self.g,
                                   det_J,
                                   c[('u',0)],
                                   c[('u',1)],
                                   c[('a',0,0)],c[('a',0,1)],
                                   c[('a',1,0)],c[('a',1,1)],
                                   c[('r',0)],
                                   c[('r',1)])
        elif self.nd==3:
            MovingMesh_3D_Evaluate(self.E,
                                   self.nu,
                                   self.g,
                                   det_J,
                                   c[('u',0)],
                                   c[('u',1)],
                                   c[('u',2)],
                                   c[('a',0,0)],c[('a',0,1)],c[('a',0,2)],
                                   c[('a',1,0)],c[('a',1,1)],c[('a',1,2)],
                                   c[('a',2,0)],c[('a',2,1)],c[('a',2,2)],
                                   c[('r',0)],
                                   c[('r',1)],
                                   c[('r',2)])

            
from proteus.ctransportCoefficients import kEpsilon_2D_Evaluate
from proteus.ctransportCoefficients import kEpsilon_2D_Evaluate_sd
from proteus.ctransportCoefficients import kEpsilon_3D_Evaluate_sd
class kEpsilon(TC_base):
    r"""Basic k-epsilon model for incompressible flow from Hutter etal Chaper 11

    :math:`\bar{\vec v} = <\vec v>` Reynolds-averaged (mean) velocity
    :math:`\vec v^{'}   =` turbulent fluctuation

    assume :math:`\vec v = <\vec v> + \vec v^{'}`, with :math:`<\vec v^{'}> = 0`

    Reynolds averaged NS equations

    .. math::

       \deld \bar{\vec v} = 0
    
    .. math::

       \pd{\bar{\vec v}}{t} + \deld \left(\bar{\vec v} \outer \bar{\vec v}\right)
               -\nu \deld \ten{\bar{D}} + \frac{1}{\rho}\grad \bar p
               - \frac{1}{\rho}\deld \ten{R} = 0

    Reynolds stress term

    .. math::

       \ten R = -\rho <\vec v^{'}\outer \vec v^{'}>
       \frac{1}{\rho}\ten{R} = 2 \nu_t \bar{D} - \frac{2}{3}k\ten{I}
       D_{ij}(\vec v) = \frac{1}{2} \left( \pd{v_i}{x_j} + \pd{v_j}{x_i}\right)
       \ten D \bar{\ten D} = D(<\vec v>), \ten D^{'} = \ten D(\vec v^{'})

    k-epsilon tranport equations

    .. math::

       \pd{k}{t} + \deld (k\bar{\vec v})
       - \deld\left[\left(\frac{\nu_t}{\sigma_k} + \nu\right)\grad k \right]
       - 4\nu_t \Pi_{D} + \epsilon = 0

    .. math::

       \pd{\varepsilon}{t} + \deld (\varepsilon \bar{\vec v})
       - \deld\left[\left(\frac{\nu_t}{\sigma_\varepsilon} + \nu\right)\grad \varepsilon \right]
       - 4c_1 k \Pi_{D} + c_2 \frac{\epsilon^2}{k} = 0
    
    """

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
    def __init__(self,
                 flowModelID=0,
                 nd     =2,
                 c_mu   =0.09,
                 c_1    =0.126,
                 c_2    =1.92,
                 c_epsilon = 0.07, #Default values from ...
                 sigma_k=1.0,#Prandtl Number
                 sigma_e=1.29,#essentially c_mu/c_epsilon
                 g=[0.0,-9.8],
                 nu=1.004e-6,
                 rho=998.0):
        self.c_mu= c_mu
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_e = c_epsilon
        self.sigma_k = sigma_k
        self.sigma_e = sigma_e
        self.g = numpy.array(g)
        self.nu= nu
        self.rho=rho
        self.nc=2
        self.nd=nd
        variableNames=['k','eps']
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        mass = {0:{0:'linear'},
                1:{1:'linear'}}
        advection = {0:{0:'linear'},
                     1:{1:'linear'}}
        potential = {0:{0:'u'},
                     1:{1:'u'}}
        diffusion = {0:{0:{0:'nonlinear',
                           1:'nonlinear'}},
                     1:{1:{0:'nonlinear',
                           1:'nonlinear'}}}
        reaction = {0:{0:'nonlinear',
                       1:'nonlinear'},
                    1:{0:'nonlinear',
                       1:'nonlinear'}}
        if self.nd == 2:
            sdInfo    = {(0,0):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i')),
                         (1,1):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i'))}
        else:
            sdInfo    = {(0,0):(numpy.array([0,1,2,3],dtype='i'),
                                numpy.array([0,1,2],dtype='i')),
                         (1,1):(numpy.array([0,1,2,3],dtype='i'),
                                numpy.array([0,1,2],dtype='i'))}

        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors=sdInfo)
        self.flowModelID = flowModelID
        for term in ['q_velocity','ebq_velocity','ebqe_velocity','ebq_global_velocity']:
            setattr(self,term,None)
        for term in ['q_gradu','q_gradv','q_gradw',
                     'ebq_gradu','ebq_gradv','ebq_gradw',
                     'ebq_global_gradu','ebq_global_gradv','ebq_global_gradw',
                     'ebqe_gradu','ebqe_gradv','ebqe_gradw']:
            setattr(self,term,None)
        #assumed entries in flow model for velocity, grad(u) and grad(v)
        self.nameMap = {'velocity':('velocity',0),
                        'gradu':('grad(u)',1),
                        'gradv':('grad(u)',2),
                        'gradw':('grad(u)',3)}

    def attachModels(self,modelList):
        if self.flowModelID is not None:
            terms = ['velocity','gradu','gradv']
            if self.nd == 3: terms.append('gradw')
            for quad in ['q','ebq','ebqe','ebq_global']:
                for term in terms:
                    d = getattr(modelList[self.flowModelID],quad)
                    key = self.nameMap[term]
                    if key in d:
                        name = quad+'_'+term
                        setattr(self,name,d[key])
            self.flowModel=modelList[self.flowModelID]#debug
    def initializeMesh(self,mesh):
        pass
    def initializeElementQuadrature(self,t,cq):
        #initialize so it can run without a flow model
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'q'+'_'+term
            #in case don't have grad(u)?
            dims = [i for i in cq['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'ebq'+'_'+term
            dims = [i for i in cebq['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebq[('grad(u)',0)].shape,'d'))
            name = 'ebq_global'+'_'+term
            dims = [i for i in cebq_global['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebq_global[('grad(u)',0)].shape,'d'))
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'ebqe'+'_'+term
            dims = [i for i in cebqe['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebqe[('grad(u)',0)].shape,'d'))
    def evaluate(self,t,c):
        velocity = None; gradu = None; gradv = None; gradw = None
        if c['x'].shape[:-1] == self.q_gradu.shape[:-1]:
            gradu = self.q_gradu; gradv = self.q_gradv; velocity = self.q_velocity; gradw = self.q_gradw
        elif c['x'].shape[:-1] == self.ebqe_gradu.shape[:-1]:
            gradu = self.ebqe_gradu; gradv = self.ebqe_gradv; velocity = self.ebqe_velocity; gradw = self.ebqe_gradw
        elif c['x'].shape[:-1] == self.ebq_global_gradu.shape[:-1]:
            gradu = self.ebq_global_gradu; gradv = self.ebq_global_gradv; velocity = self.ebq_global_velocity; gradw = self.ebq_global_gradw
        elif c['x'].shape[:-1] == self.ebq_gradu.shape[:-1]:
            gradu = self.ebq_gradu; gradv = self.ebq_gradv; velocity = self.ebq_velocity; gradw = self.ebq_gradw
        else:
            raise TypeError("c['x'].shape= not recognized ")
        hackSourceTerm = False#True
        if self.nd == 2:
            if self.sd == True:
                kEpsilon_2D_Evaluate_sd(self.sigma_k,
                                        self.sigma_e,
                                        self.c_1,
                                        self.c_2,
                                        self.c_mu,
                                        self.c_e,
                                        self.nu,
                                        velocity,
                                        gradu,
                                        gradv,
                                        c[('u',0)],
                                        c[('u',1)],
                                        c[('m',0)],
                                        c[('dm',0,0)],
                                        c[('m',1)],
                                        c[('dm',1,1)],
                                        c[('phi',0)],   #get rid of nonlinear potential
                                        c[('dphi',0,0)],
                                        c[('phi',1)],
                                        c[('dphi',1,1)],
                                        c[('f',0)],
                                        c[('df',0,0)],
                                        c[('f',1)],
                                        c[('df',1,1)],
                                        c[('a',0,0)],
                                        c[('da',0,0,0)],
                                        c[('da',0,0,1)],
                                        c[('a',1,1)],
                                        c[('da',1,1,0)],
                                        c[('da',1,1,1)],
                                        c[('r',0)],
                                        c[('dr',0,0)],
                                        c[('dr',0,1)],
                                        c[('r',1)],
                                        c[('dr',1,0)],
                                        c[('dr',1,1)])
            else:
                kEpsilon_2D_Evaluate(self.sigma_k,
                                     self.sigma_e,
                                     self.c_1,
                                     self.c_2,
                                     self.c_mu,
                                     self.c_e,
                                     self.nu,
                                     velocity,
                                     gradu,
                                     gradv,
                                     c[('u',0)],
                                     c[('u',1)],
                                     c[('m',0)],
                                     c[('dm',0,0)],
                                     c[('m',1)],
                                     c[('dm',1,1)],
                                     c[('phi',0)],   #get rid of nonlinear potential
                                     c[('dphi',0,0)],
                                     c[('phi',1)],
                                     c[('dphi',1,1)],
                                     c[('f',0)],
                                     c[('df',0,0)],
                                     c[('f',1)],
                                     c[('df',1,1)],
                                     c[('a',0,0)],
                                     c[('da',0,0,0)],
                                     c[('da',0,0,1)],
                                     c[('a',1,1)],
                                     c[('da',1,1,0)],
                                     c[('da',1,1,1)],
                                     c[('r',0)],
                                     c[('dr',0,0)],
                                     c[('dr',0,1)],
                                     c[('r',1)],
                                     c[('dr',1,0)],
                                     c[('dr',1,1)])
        elif self.nd == 3:
            if self.sd == True:
                kEpsilon_3D_Evaluate_sd(self.sigma_k,
                                        self.sigma_e,
                                        self.c_1,
                                        self.c_2,
                                        self.c_mu,
                                        self.c_e,
                                        self.nu,
                                        velocity,
                                        gradu,
                                        gradv,
                                        gradw,
                                        c[('u',0)],
                                        c[('u',1)],
                                        c[('m',0)],
                                        c[('dm',0,0)],
                                        c[('m',1)],
                                        c[('dm',1,1)],
                                        c[('phi',0)],   #get rid of nonlinear potential
                                        c[('dphi',0,0)],
                                        c[('phi',1)],
                                        c[('dphi',1,1)],
                                        c[('f',0)],
                                        c[('df',0,0)],
                                        c[('f',1)],
                                        c[('df',1,1)],
                                        c[('a',0,0)],
                                        c[('da',0,0,0)],
                                        c[('da',0,0,1)],
                                        c[('a',1,1)],
                                        c[('da',1,1,0)],
                                        c[('da',1,1,1)],
                                        c[('r',0)],
                                        c[('dr',0,0)],
                                        c[('dr',0,1)],
                                        c[('r',1)],
                                        c[('dr',1,0)],
                                        c[('dr',1,1)])
            else:
                assert False, "k-epsilon 3d non sd not implemented"

            if hackSourceTerm:
                c[('r',0)].flat[:] = 0.0
                c[('dr',0,0)].flat[:] = 0.0
                c[('dr',0,1)].flat[:] = 0.0
                c[('r',1)].flat[:] = 0.0
                c[('dr',1,0)].flat[:] = 0.0
                c[('dr',1,1)].flat[:] = 0.0

from proteus.ctransportCoefficients import kEpsilon_k_2D_Evaluate_sd
from proteus.ctransportCoefficients import kEpsilon_k_3D_Evaluate_sd
class kEpsilon_k(TC_base):
    r"""Basic k-epsilon model for incompressible flow from Hutter etal
    Chaper 11 but solves for just k assuming epsilon computed
    independently and lagged in time

    
    :math:`\bar{\vec v} = <\vec v>` Reynolds-averaged (mean) velocity

    :math:`\vec v^{'}   =` turbulent fluctuation

    assume :math:`\vec v = <\vec v> + \vec v^{'}`, with :math:`<\vec v^{'}> = 0`

    Reynolds averaged NS equations

    .. math::
    
       \deld \bar{\vec v} = 0
       \pd{\bar{\vec v}}{t} + \deld \left(\bar{\vec v} \outer \bar{\vec v}\right)
               -\nu \deld \ten{\bar{D}} + \frac{1}{\rho}\grad \bar p
               - \frac{1}{\rho}\deld \ten{R} = 0

    Reynolds stress term

    .. math::

       \ten R = -\rho <\vec v^{'}\outer \vec v^{'}>
       \frac{1}{\rho}\ten{R} = 2 \nu_t \bar{D} - \frac{2}{3}k\ten{I}
       D_{ij}(\vec v) = \frac{1}{2} \left( \pd{v_i}{x_j} + \pd{v_j}{x_i}\right)
       \ten D \bar{\ten D} = D(<\vec v>), \ten D^{'} = \ten D(\vec v^{'})

    k-epsilon tranport equations

    .. math::

        \pd{k}{t} + \deld (k\bar{\vec v})
          - \deld\left[\left(\frac{\nu_t}{\sigma_k} + \nu\right)\grad k \right]
          - 4\nu_t \Pi_{D} + \epsilon = 0
        \pd{\varepsilon}{t} + \deld (\varepsilon \bar{\vec v})
          - \deld\left[\left(\frac{\nu_t}{\sigma_\varepsilon} + \nu\right)\grad \varepsilon \right]
          - 4c_1 k \Pi_{D} + c_2 \frac{\epsilon^2}{k} = 0



    :math:`k`              -- turbulent kinetic energy = <\vec v^{'}\dot \vec v^{'}>
    :math:`\varepsilon`   -- turbulent dissipation rate = 4 \nu <\Pi_{D^{'}}>
    :math:`\nu`            -- kinematic viscosity (\mu/\rho)
    :math:`\nu_t`          -- turbulent viscosity = c_mu \frac{k^2}{\varepsilon}

    .. math::

       \Pi_{\ten A} = \frac{1}{2}tr(\ten A^2) = 1/2 \ten A\cdot \ten A
       \ten D \cdot \ten D = \frac{1}{4}\left[ (4 u_x^2 + 4 v_y^2 +
                                        1/2 (u_y + v_x)^2 \right]
       4 \Pi_{D} = 2 \frac{1}{4}\left[ (4 u_x^2 + 4 v_y^2 +
                                1/2 (u_y + v_x)^2 \right]
          = \left[ (2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2 \right]
   
    :math:`\sigma_k` -- Prandtl number \approx 1
    :math:`\sigma_e` -- c_{\mu}/c_e

    .. math::

       c_{\mu} = 0.09, c_1 = 0.126, c_2 = 1.92, c_{\varepsilon} = 0.07

    """
    def __init__(self,
                 flowModelID=0,
                 epsilonModelID=2,
                 nd     =2,
                 c_mu   =0.09,
                 sigma_k=1.0,#Prandtl Number
                 g=[0.0,-9.8],
                 nu=1.004e-6,
                 rho=998.0):
        self.c_mu= c_mu
        self.sigma_k = sigma_k
        self.g = numpy.array(g)
        self.nu= nu
        self.rho=rho
        self.nc=1
        self.nd=nd
        variableNames=['k']
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        mass = {0:{0:'linear'}}
        advection = {0:{0:'linear'}}
        potential = {0:{0:'u'}}
        diffusion = {0:{0:{0:'nonlinear',}}}
        reaction = {0:{0:'nonlinear'}}
        if self.nd == 2:
            sdInfo    = {(0,0):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i'))}
        else:
            sdInfo    = {(0,0):(numpy.array([0,1,2,3],dtype='i'),
                                numpy.array([0,1,2],dtype='i'))}
        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors=sdInfo)
        self.flowModelID = flowModelID
        self.epsilonModelID = epsilonModelID
        for term in ['q_velocity','ebq_velocity','ebqe_velocity','ebq_global_velocity']:
            setattr(self,term,None)
        for term in ['q_gradu','q_gradv','q_gradw',
                     'ebq_gradu','ebq_gradv','ebq_gradw',
                     'ebq_global_gradu','ebq_global_gradv','ebq_global_gradw',
                     'ebqe_gradu','ebqe_gradv','ebqe_gradw',
                     'q_epsilon','ebq_epsilon','ebq_global_epsilon','ebqe_epsilon']:
            setattr(self,term,None)
        #assumed entries in flow model for velocity, grad(u) and grad(v)
        self.nameMap = {'velocity':('velocity',0),
                        'gradu':('grad(u)',1),
                        'gradv':('grad(u)',2),
                        'gradw':('grad(u)',3),
                        'epsilon':('u',0)}

    def attachModels(self,modelList):
        if self.flowModelID is not None:
            terms = ['velocity','gradu','gradv']
            if self.nd == 3: terms.append('gradw')
            for quad in ['q','ebq','ebqe','ebq_global']:
                for term in terms:
                    d = getattr(modelList[self.flowModelID],quad)
                    key = self.nameMap[term]
                    if key in d:
                        name = quad+'_'+term
                        setattr(self,name,d[key])
            self.flowModel=modelList[self.flowModelID]#debug
        if self.epsilonModelID is not None:
            terms = ['epsilon']
            for quad in ['q','ebq','ebqe','ebq_global']:
                for term in terms:
                    d = getattr(modelList[self.epsilonModelID],quad)
                    key = self.nameMap[term]
                    if key in d:
                        name = quad+'_'+term
                        setattr(self,name,d[key])

    def initializeMesh(self,mesh):
        pass
    def initializeElementQuadrature(self,t,cq):
        #initialize so it can run without a flow model
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'q'+'_'+term
            #in case don't have grad(u)?
            dims = [i for i in cq['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
        terms = ['epsilon']
        for term in terms:
            name = 'q'+'_'+term
            dims = [i for i in cq['x'].shape[:-1]]
            setattr(self,name,numpy.zeros(tuple(dims),'d'))

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'ebq'+'_'+term
            dims = [i for i in cebq['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebq[('grad(u)',0)].shape,'d'))
            name = 'ebq_global'+'_'+term
            dims = [i for i in cebq_global['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebq_global[('grad(u)',0)].shape,'d'))
        terms = ['epsilon']
        for term in terms:
            name = 'ebq'+'_'+term
            dims = [i for i in cebq['x'].shape[:-1]]
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            name = 'ebq_global'+'_'+term
            dims = [i for i in cebq_global['x'].shape[:-1]]
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'ebqe'+'_'+term
            dims = [i for i in cebqe['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebqe[('grad(u)',0)].shape,'d'))
        terms = ['epsilon']
        for term in terms:
            name = 'ebqe'+'_'+term
            dims = [i for i in cebqe['x'].shape[:-1]]
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
    def evaluate(self,t,c):
        velocity = None; gradu = None; gradv = None; gradw = None; epsilon = None
        if c['x'].shape[:-1] == self.q_gradu.shape[:-1]:
            gradu = self.q_gradu; gradv = self.q_gradv; velocity = self.q_velocity; gradw = self.q_gradw
            epsilon = self.q_epsilon
        elif c['x'].shape[:-1] == self.ebqe_gradu.shape[:-1]:
            gradu = self.ebqe_gradu; gradv = self.ebqe_gradv; velocity = self.ebqe_velocity; gradw = self.ebqe_gradw
            epsilon = self.ebqe_epsilon
        elif c['x'].shape[:-1] == self.ebq_global_gradu.shape[:-1]:
            gradu = self.ebq_global_gradu; gradv = self.ebq_global_gradv; velocity = self.ebq_global_velocity; gradw = self.ebq_global_gradw
            epsilon = self.ebq_global_epsilon
        elif c['x'].shape[:-1] == self.ebq_gradu.shape[:-1]:
            gradu = self.ebq_gradu; gradv = self.ebq_gradv; velocity = self.ebq_velocity; gradw = self.ebq_gradw
            epsilon = self.ebq_epsilon
        else:
            raise TypeError("c['x'].shape= not recognized ")
        hackSourceTerm = False#True
        if self.nd == 2:
            if self.sd == True:
                kEpsilon_k_2D_Evaluate_sd(self.sigma_k,
                                          self.c_mu,
                                          self.nu,
                                          velocity,
                                          gradu,
                                          gradv,
                                          c[('u',0)],
                                          epsilon,
                                          c[('m',0)],
                                          c[('dm',0,0)],
                                          c[('phi',0)],   #get rid of nonlinear potential
                                          c[('dphi',0,0)],
                                          c[('f',0)],
                                          c[('df',0,0)],
                                          c[('a',0,0)],
                                          c[('da',0,0,0)],
                                          c[('r',0)],
                                          c[('dr',0,0)])

            else:
                raise NotImplementedError
        else:
            if self.sd == True:
                kEpsilon_k_3D_Evaluate_sd(self.sigma_k,
                                          self.c_mu,
                                          self.nu,
                                          velocity,
                                          gradu,
                                          gradv,
                                          gradw,
                                          c[('u',0)],
                                          epsilon,
                                          c[('m',0)],
                                          c[('dm',0,0)],
                                          c[('phi',0)],   #get rid of nonlinear potential
                                          c[('dphi',0,0)],
                                          c[('f',0)],
                                          c[('df',0,0)],
                                          c[('a',0,0)],
                                          c[('da',0,0,0)],
                                          c[('r',0)],
                                          c[('dr',0,0)])

            else:
                raise NotImplementedError

        if hackSourceTerm:
            c[('r',0)].flat[:] = 0.0
            c[('dr',0,0)].flat[:] = 0.0

from proteus.ctransportCoefficients import kEpsilon_epsilon_2D_Evaluate_sd
from proteus.ctransportCoefficients import kEpsilon_epsilon_3D_Evaluate_sd
class kEpsilon_epsilon(TC_base):
    r"""Basic k-epsilon model for incompressible flow from Hutter etal
    Chaper 11 but solves for just epsilon assuming k lagged

    :math:`\bar{\vec v} = <\vec v>` Reynolds-averaged (mean) velocity

    :math:`\vec v^{'}`   = turbulent fluctuation

    assume :math:`\vec v = <\vec v> + \vec v^{'}`, with :math:`<\vec v^{'}> = 0`

    Reynolds averaged NS equations

    .. math::

       \deld \bar{\vec v} = 0
       \pd{\bar{\vec v}}{t} + \deld \left(\bar{\vec v} \outer \bar{\vec v}\right)
               -\nu \deld \ten{\bar{D}} + \frac{1}{\rho}\grad \bar p
               - \frac{1}{\rho}\deld \ten{R} = 0
    
    Reynolds stress term

    .. math::

       \ten R = -\rho <\vec v^{'}\outer \vec v^{'}>
       \frac{1}{\rho}\ten{R} = 2 \nu_t \bar{D} - \frac{2}{3}k\ten{I}
       D_{ij}(\vec v) = \frac{1}{2} \left( \pd{v_i}{x_j} + \pd{v_j}{x_i}\right)
       \ten D \bar{\ten D} = D(<\vec v>), \ten D^{'} = \ten D(\vec v^{'})

    k-epsilon tranport equations

    .. math::
       \pd{k}{t} + \deld (k\bar{\vec v})
          - \deld\left[\left(\frac{\nu_t}{\sigma_k} + \nu\right)\grad k \right]
          - 4\nu_t \Pi_{D} + \epsilon = 0

    .. math::

       \pd{\varepsilon}{t} + \deld (\varepsilon \bar{\vec v})
          - \deld\left[\left(\frac{\nu_t}{\sigma_\varepsilon} + \nu\right)\grad \varepsilon \right]
          - 4c_1 k \Pi_{D} + c_2 \frac{\epsilon^2}{k} = 0


    :math:`k`              -- turbulent kinetic energy = <\vec v^{'}\dot \vec v^{'}>
    :math:`\varepsilon`    -- turbulent dissipation rate = 4 \nu <\Pi_{D^{'}}>
    :math:`\nu`            -- kinematic viscosity (\mu/\rho)
    :math:`\nu_t`          -- turbulent viscosity = c_mu \frac{k^2}{\varepsilon}

    .. math::
       \Pi_{\ten A} = \frac{1}{2}tr(\ten A^2) = 1/2 \ten A\cdot \ten A
       \ten D \cdot \ten D = \frac{1}{4}\left[ (4 u_x^2 + 4 v_y^2 +
                                        1/2 (u_y + v_x)^2 \right]

        4 \Pi_{D} = 2 \frac{1}{4}\left[ (4 u_x^2 + 4 v_y^2 +
                                1/2 (u_y + v_x)^2 \right]
          = \left[ (2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2 \right]

    :math:`\sigma_k` -- Prandtl number \approx 1
    :math:`\sigma_e` -- :math:`c_{\mu}/c_e`

    :math:c_{\mu} = 0.09, c_1 = 0.126, c_2 = 1.92, c_{\varepsilon} = 0.07`

    """
    def __init__(self,
                 flowModelID=0,
                 kModelID=1,
                 nd     =2,
                 c_mu   =0.09,
                 c_1    =0.126,
                 c_2    =1.92,
                 c_epsilon = 0.07, #Default values from ...
                 sigma_e=1.29,#essentially c_mu/c_epsilon
                 g=[0.0,-9.8],
                 nu=1.004e-6,
                 rho=998.0):
        self.c_mu= c_mu
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_e = c_epsilon
        self.sigma_e = sigma_e
        self.g = numpy.array(g)
        self.nu= nu
        self.rho=rho
        self.nc=1
        self.nd=nd
        variableNames=['epsilon']
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        mass = {0:{0:'linear'}}
        advection = {0:{0:'linear'}}
        potential = {0:{0:'u'}}
        diffusion = {0:{0:{0:'nonlinear',}}}
        reaction = {0:{0:'nonlinear'}}
        if self.nd == 2:
            sdInfo    = {(0,0):(numpy.array([0,1,2],dtype='i'),
                                numpy.array([0,1],dtype='i'))}
        else:
            sdInfo    = {(0,0):(numpy.array([0,1,2,3],dtype='i'),
                                numpy.array([0,1,2],dtype='i'))}
        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors=sdInfo)
        self.flowModelID = flowModelID
        self.kModelID = kModelID
        for term in ['q_velocity','ebq_velocity','ebqe_velocity','ebq_global_velocity']:
            setattr(self,term,None)
        for term in ['q_gradu','q_gradv','q_gradw',
                     'ebq_gradu','ebq_gradv','ebq_gradw',
                     'ebq_global_gradu','ebq_global_gradv','ebq_global_gradw',
                     'ebqe_gradu','ebqe_gradv','ebqe_gradw',
                     'q_k','ebq_k','ebq_global_k','ebqe_k']:
            setattr(self,term,None)
        #assumed entries in flow model for velocity, grad(u) and grad(v)
        self.nameMap = {'velocity':('velocity',0),
                        'gradu':('grad(u)',1),
                        'gradv':('grad(u)',2),
                        'gradw':('grad(u)',3),
                        'k':('u',0)}

    def attachModels(self,modelList):
        if self.flowModelID is not None:
            terms = ['velocity','gradu','gradv']
            if self.nd == 3: terms.append('gradw')
            for quad in ['q','ebq','ebqe','ebq_global']:
                for term in terms:
                    d = getattr(modelList[self.flowModelID],quad)
                    key = self.nameMap[term]
                    if key in d:
                        name = quad+'_'+term
                        setattr(self,name,d[key])
            self.flowModel=modelList[self.flowModelID]#debug
        if self.kModelID is not None:
            terms = ['k']
            for quad in ['q','ebq','ebqe','ebq_global']:
                for term in terms:
                    d = getattr(modelList[self.kModelID],quad)
                    key = self.nameMap[term]
                    if key in d:
                        name = quad+'_'+term
                        setattr(self,name,d[key])

    def initializeMesh(self,mesh):
        pass
    def initializeElementQuadrature(self,t,cq):
        #initialize so it can run without a flow model
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'q'+'_'+term
            #in case don't have grad(u)?
            dims = [i for i in cq['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
        terms = ['k']
        for term in terms:
            name = 'q'+'_'+term
            dims = [i for i in cq['x'].shape[:-1]]
            setattr(self,name,numpy.zeros(tuple(dims),'d'))

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'ebq'+'_'+term
            dims = [i for i in cebq['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebq[('grad(u)',0)].shape,'d'))
            name = 'ebq_global'+'_'+term
            dims = [i for i in cebq_global['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebq_global[('grad(u)',0)].shape,'d'))
        terms = ['k']
        for term in terms:
            name = 'ebq'+'_'+term
            dims = [i for i in cebq['x'].shape[:-1]]
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            name = 'ebq_global'+'_'+term
            dims = [i for i in cebq_global['x'].shape[:-1]]
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        terms = ['velocity','gradu','gradv']
        if self.nd == 3: terms.append('gradw')
        for term in terms:
            name = 'ebqe'+'_'+term
            dims = [i for i in cebqe['x'].shape[:-1]]
            dims.append(self.nd)
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
            #setattr(self,name,numpy.zeros(cebqe[('grad(u)',0)].shape,'d'))
        terms = ['k']
        for term in terms:
            name = 'ebqe'+'_'+term
            dims = [i for i in cebqe['x'].shape[:-1]]
            setattr(self,name,numpy.zeros(tuple(dims),'d'))
    def evaluate(self,t,c):
        velocity = None; gradu = None; gradv = None; gradw = None; k = None
        if c['x'].shape[:-1] == self.q_gradu.shape[:-1]:
            gradu = self.q_gradu; gradv = self.q_gradv; velocity = self.q_velocity; gradw = self.q_gradw
            k = self.q_k
        elif c['x'].shape[:-1] == self.ebqe_gradu.shape[:-1]:
            gradu = self.ebqe_gradu; gradv = self.ebqe_gradv; velocity = self.ebqe_velocity; gradw = self.ebqe_gradw
            k = self.ebqe_k
        elif c['x'].shape[:-1] == self.ebq_global_gradu.shape[:-1]:
            gradu = self.ebq_global_gradu; gradv = self.ebq_global_gradv; velocity = self.ebq_global_velocity; gradw = self.ebq_global_gradw
            k = self.ebq_global_k
        elif c['x'].shape[:-1] == self.ebq_gradu.shape[:-1]:
            gradu = self.ebq_gradu; gradv = self.ebq_gradv; velocity = self.ebq_velocity; gradw = self.ebq_gradw
            k = self.ebq_k
        else:
            raise TypeError("c['x'].shape= not recognized ")
        hackSourceTerm = False#True
        if self.nd == 2:
            if self.sd == True:
                kEpsilon_epsilon_2D_Evaluate_sd(self.sigma_e,
                                                self.c_1,
                                                self.c_2,
                                                self.c_mu,
                                                self.c_e,
                                                self.nu,
                                                velocity,
                                                gradu,
                                                gradv,
                                                k,
                                                c[('u',0)],
                                                c[('m',0)],
                                                c[('dm',0,0)],
                                                c[('phi',0)],   #get rid of nonlinear potential
                                                c[('dphi',0,0)],
                                                c[('f',0)],
                                                c[('df',0,0)],
                                                c[('a',0,0)],
                                                c[('da',0,0,0)],
                                                c[('r',0)],
                                                c[('dr',0,0)])

            else:
                raise NotImplementedError
        else:
            if self.sd == True:
                kEpsilon_epsilon_3D_Evaluate_sd(self.sigma_e,
                                                self.c_1,
                                                self.c_2,
                                                self.c_mu,
                                                self.c_e,
                                                self.nu,
                                                velocity,
                                                gradu,
                                                gradv,
                                                gradw,
                                                k,
                                                c[('u',0)],
                                                c[('m',0)],
                                                c[('dm',0,0)],
                                                c[('phi',0)],   #get rid of nonlinear potential
                                                c[('dphi',0,0)],
                                                c[('f',0)],
                                                c[('df',0,0)],
                                                c[('a',0,0)],
                                                c[('da',0,0,0)],
                                                c[('r',0)],
                                                c[('dr',0,0)])

            else:
                raise NotImplementedError

        if hackSourceTerm:
            c[('r',0)].flat[:] = 0.0
            c[('dr',0,0)].flat[:] = 0.0


from .ctransportCoefficients import ReynoldsAveragedNavierStokes_kEpsilon_2D_Update
from .ctransportCoefficients import ReynoldsAveragedNavierStokes_kEpsilon_2D_Update_sd
class ReynoldsAveragedNavierStokes_kEpsilon(TwophaseNavierStokes_ST_LS_SO):
    """
    The coefficients for incompresslble fluid governed by the Navier-Stokes equations
    assuming k-epsilon model for turbulence

    KEmodelID = [m1,m2] if using two separate models for k and epsilon
    otherwise should be just an integer id
    """
    def __init__(self,
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2,nu_0=1.004e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=None,
                 KN_model=None,
                 epsFact_density=None,
                 stokes=False,
                 sd=True,
                 movingDomain=False,
                 c_mu=0.09,
                 KEmodelID=None):
        self.KEmodelID=KEmodelID
        if self.KEmodelID is not None and len(self.KEmodelID)==2:
            self.KEmodelSplit=True
        else:
            self.KEmodelSplit=False
        self.c_mu=c_mu
        TwophaseNavierStokes_ST_LS_SO.__init__(self,
                                               epsFact=epsFact,
                                               sigma=sigma,
                                               rho_0=rho_0,nu_0=nu_0,
                                               rho_1=rho_1,nu_1=nu_1,
                                               g=g,
                                               nd=nd,
                                               LS_model=LS_model,
                                               KN_model=KN_model,
                                               epsFact_density=epsFact_density,
                                               stokes=stokes,
                                               sd=sd,
                                               movingDomain=movingDomain)

        #name of terms taken from k-epsilon model
        self.kEpsilonTerms = ['k','epsilon','grad_k']
        #assumed entries in k-epsilon model for k,epsilon and grad(k)
        if not self.KEmodelSplit:
            self.nameMap = {'k':('u',0),
                            'epsilon':('u',1),
                            'grad_k':('grad(u)',0)}
        else:
            self.nameMap = {'k':('u',0),
                            'epsilon':('u',0),
                            'grad_k':('grad(u)',0)}

        for quad in ['q','ebq','ebqe','ebq_global']:
            for term in self.kEpsilonTerms:
                name = quad + '_' + term
                setattr(self,term,None)

    def attachModels(self,modelList):
        if self.KEmodelID is not None:
            for quad in ['q','ebq','ebqe','ebq_global']:
                for term in self.kEpsilonTerms:
                    if not self.KEmodelSplit:
                        d = getattr(modelList[self.KEmodelID],quad)
                        key = self.nameMap[term]
                        if key in d:
                            name = quad+'_'+term
                            setattr(self,name,d[key])
                    else:
                        if term == 'epsilon':
                            ID = self.KEmodelID[1]
                        else:
                            ID = self.KEmodelID[0]
                        d = getattr(modelList[ID],quad)
                        key = self.nameMap[term]
                        if key in d:
                            name = quad+'_'+term
                            setattr(self,name,d[key])

            if self.KEmodelSplit:
                self.Kmodel=modelList[self.KEmodelID[0]]
                self.EpsilonModel=modelList[self.KEmodelID[1]]
            else:
                self.KEmodel=modelList[self.KEmodelID]#debug
                self.Kmodel=self.KEmodel
                self.EpsilonModel=self.KEmodel
    def initializeMesh(self,mesh):
        TwophaseNavierStokes_ST_LS_SO.initializeMesh(self,mesh)
    #initialize so it can run as single phase
    def initializeElementQuadrature(self,t,cq):
        TwophaseNavierStokes_ST_LS_SO.initializeElementQuadrature(self,t,cq)
        for term in self.kEpsilonTerms:
            name = 'q'+'_'+term
            #in case don't have grad(u)?
            dims = [i for i in cq['x'].shape[:-1]]
            if term == 'grad_k':
                dims.append(self.nd)
            if term == 'epsilon':
                setattr(self,name,numpy.ones(tuple(dims),'d'))
            else:
                setattr(self,name,numpy.zeros(tuple(dims),'d'))
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseNavierStokes_ST_LS_SO.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        for term in self.kEpsilonTerms:
            name = 'ebq'+'_'+term
            dims = [i for i in cebq['x'].shape[:-1]]
            if term == 'grad_k':
                dims.append(self.nd)
            if term == 'epsilon':
                setattr(self,name,numpy.ones(tuple(dims),'d'))
            else:
                setattr(self,name,numpy.zeros(tuple(dims),'d'))
            name = 'ebq_global'+'_'+term
            dims = [i for i in cebq_global['x'].shape[:-1]]
            if term == 'grad_k':
                dims.append(self.nd)
            if term == 'epsilon':
                setattr(self,name,numpy.ones(tuple(dims),'d'))
            else:
                setattr(self,name,numpy.zeros(tuple(dims),'d'))
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseNavierStokes_ST_LS_SO.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        for term in self.kEpsilonTerms:
            name = 'ebqe'+'_'+term
            dims = [i for i in cebqe['x'].shape[:-1]]
            if term == 'grad_k':
                dims.append(self.nd)
            if term == 'epsilon':
                setattr(self,name,numpy.ones(tuple(dims),'d'))
            else:
                setattr(self,name,numpy.zeros(tuple(dims),'d'))
    def evaluate(self,t,c):
        TwophaseNavierStokes_ST_LS_SO.evaluate(self,t,c)
        if self.KEmodelID is None:
            return
        k = None; epsilon = None; grad_k = None
        if c['x'].shape[:-1] == self.q_grad_k.shape[:-1]:
            k = self.q_k; epsilon = self.q_epsilon; grad_k = self.q_grad_k;
        elif c['x'].shape[:-1] == self.ebqe_grad_k.shape[:-1]:
            k = self.ebqe_k; epsilon = self.ebqe_epsilon; grad_k = self.ebqe_grad_k;
        elif c['x'].shape[:-1] == self.ebq_global_grad_k.shape[:-1]:
            k = self.ebq_global_k; epsilon = self.ebq_global_epsilon; grad_k = self.ebq_global_grad_k;
        elif c['x'].shape[:-1] == self.ebq_grad_k.shape[:-1]:
            k = self.ebq_k; epsilon = self.ebq_epsilon; grad_k = self.ebq_grad_k;
        else:
            raise TypeError("c['x'].shape= not recognized ")
        if self.nd == 2:
            if self.sd == True:
                ReynoldsAveragedNavierStokes_kEpsilon_2D_Update_sd(self.rho,
                                                                   self.nu,
                                                                   self.c_mu,
                                                                   k,
                                                                   grad_k,
                                                                   epsilon,
                                                                   c[('a',1,1)],
                                                                   c[('a',2,2)],
                                                                   c[('a',1,2)],
                                                                   c[('a',2,1)],
                                                                   c[('r',1)],
                                                                   c[('r',2)])
            else:
                ReynoldsAveragedNavierStokes_kEpsilon_2D_Update(self.rho,
                                                                self.nu,
                                                                self.c_mu,
                                                                k,
                                                                grad_k,
                                                                epsilon,
                                                                c[('a',1,1)],
                                                                c[('a',2,2)],
                                                                c[('a',1,2)],
                                                                c[('a',2,1)],
                                                                c[('r',1)],
                                                                c[('r',2)])

        else:
            if self.sd == True:
                ReynoldsAveragedNavierStokes_kEpsilon_3D_Update_sd(self.rho,
                                                                   self.nu,
                                                                   self.c_mu,
                                                                   k,
                                                                   grad_k,
                                                                   epsilon,
                                                                   c[('a',1,1)],
                                                                   c[('a',2,2)],
                                                                   c[('a',3,3)],
                                                                   c[('a',1,2)],
                                                                   c[('a',1,3)],
                                                                   c[('a',2,1)],
                                                                   c[('a',2,3)],
                                                                   c[('a',3,1)],
                                                                   c[('a',3,2)],
                                                                   c[('r',1)],
                                                                   c[('r',2)],
                                                                   c[('r',3)])
            else:
                ReynoldsAveragedNavierStokes_kEpsilon_3D_Update(self.rho,
                                                                self.nu,
                                                                self.c_mu,
                                                                k,
                                                                grad_k,
                                                                epsilon,
                                                                c[('a',1,1)],
                                                                c[('a',2,2)],
                                                                c[('a',3,3)],
                                                                c[('a',1,2)],
                                                                c[('a',1,3)],
                                                                c[('a',2,1)],
                                                                c[('a',2,3)],
                                                                c[('a',3,1)],
                                                                c[('a',3,2)],
                                                                c[('r',1)],
                                                                c[('r',2)],
                                                                c[('r',3)])

class ReynoldsAveragedNavierStokes_AlgebraicClosure(TwophaseNavierStokes_ST_LS_SO):
    """
    The coefficients for incompresslble fluid governed by the Navier-Stokes equations
    assuming k-epsilon model for turbulence
    """
    def __init__(self,
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2,nu_0=1.004e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=None,
                 KN_model=None,
                 epsFact_density=None,
                 stokes=False,
                 sd=True,
                 movingDomain=False,
                 turbulenceClosureFlag=None):
        #allow algebraic models for turbulence
        self.turbulenceClosureFlag = turbulenceClosureFlag
        for term in ['q_nu_t','ebq_nu_t','ebqe_nu_t','ebq_global_nu_t']:
            setattr(self,term,None)
        #easier to just hold on to quadrature dictionaries for calculations
        for term in ['q','ebq','ebqe','ebq_global']:
            setattr(self,term,None)

from .ctransportCoefficients import eddyViscosity_2D_Update_sd,eddyViscosity_2D_Update
from .ctransportCoefficients import calculateEddyViscosity_Smagorinsky_2D
from .ctransportCoefficients import calculateEddyViscosity_Smagorinsky2P_2D
from .ctransportCoefficients import eddyViscosity_3D_Update_sd,eddyViscosity_3D_Update
from .ctransportCoefficients import calculateEddyViscosity_Smagorinsky_3D
from .ctransportCoefficients import calculateEddyViscosity_Smagorinsky2P_3D
class TwophaseReynoldsAveragedNavierStokes_AlgebraicClosure(TwophaseNavierStokes_ST_LS_SO):
    """
    The coefficients for incompresslble fluid governed by the Navier-Stokes equations
    assuming k-epsilon model for turbulence
    """
    def __init__(self,
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2,nu_0=1.004e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=None,
                 KN_model=None,
                 epsFact_density=None,
                 stokes=False,
                 sd=True,
                 movingDomain=False,
                 turbulenceClosureFlag=None,
                 smagorinskyConstant_0=0.1,
                 smagorinskyConstant_1=0.1,
                 epsFact_smagorinsky=0.33):
        #allow algebraic models for turbulence
        self.turbulenceClosureFlag = turbulenceClosureFlag
        self.smagorinskyConstant_0 = smagorinskyConstant_0
        self.smagorinskyConstant_1 = smagorinskyConstant_1
        self.epsFact_smagorinsky=epsFact_smagorinsky

        for term in ['q_nu_t']:
            setattr(self,term,None)
        #easier to just hold on to quadrature dictionaries for calculations
        for term in ['q']:
            setattr(self,term,None)

        TwophaseNavierStokes_ST_LS_SO.__init__(self,
                                               epsFact=epsFact,
                                               sigma=sigma,
                                               rho_0=rho_0,nu_0=nu_0,
                                               rho_1=rho_1,nu_1=nu_1,
                                               g=g,
                                               nd=nd,
                                               LS_model=LS_model,
                                               KN_model=KN_model,
                                               epsFact_density=epsFact_density,
                                               stokes=stokes,
                                               sd=sd,
                                               movingDomain=movingDomain)


    def initializeMesh(self,mesh):
        TwophaseNavierStokes_ST_LS_SO.initializeMesh(self,mesh)
        self.mesh = mesh
        self.eps_smagorinsky= self.mesh.h*self.epsFact_smagorinsky
    #initialize so it can run as single phase
    def initializeElementQuadrature(self,t,cq):
        TwophaseNavierStokes_ST_LS_SO.initializeElementQuadrature(self,t,cq)
        if self.turbulenceClosureFlag is not None:
            self.q_nu_t = numpy.zeros(cq[('u',1)].shape,'d')
            self.q      = cq

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseNavierStokes_ST_LS_SO.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        #if self.turbulenceClosureFlag is not None:
        #    self.ebq_nu_t = numpy.zeros(cebq[('u',0)].shape,'d')
        #    self.ebq_global_nu_t = numpy.zeros(cebq_global[('u',0)].shape,'d')
        #    self.ebq = cebq
        #    self.ebq_global = cebq_global
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseNavierStokes_ST_LS_SO.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        #if self.turbulenceClosureFlag is not None:
        #    self.ebqe_nu_t = numpy.zeros(cebqe[('u',0)].shape,'d')
        #    self.ebqe = cebqe
    def preStep(self,t,firstStep=False):
        if self.turbulenceClosureFlag == 0: #Smagorinsky
            if self.nd == 2:
                #q
                self.calculateEddyViscosity_Smagorinsky_2D(self.smagorinskyConstant_0,
                                                           self.mesh.elementDiametersArray,
                                                           self.q[('grad(u)',1)],
                                                           self.q[('grad(u)',2)],
                                                           self.q_nu_t)
                #print "Smagorinsky preStep t=%s min=%s max=%s " % (t,self.q_nu_t.min(),self.q_nu_t.max())
                #ebqe
            elif self.nd == 3:
                #q
                self.calculateEddyViscosity_Smagorinsky_3D(self.smagorinskyConstant_0,
                                                           self.mesh.elementDiametersArray,
                                                           self.q[('grad(u)',1)],
                                                           self.q[('grad(u)',2)],
                                                           self.q[('grad(u)',3)],
                                                           self.q_nu_t)
                #print "Smagorinsky preStep t=%s min=%s max=%s " % (t,self.q_nu_t.min(),self.q_nu_t.max())
                #ebqe
            else:
                assert False, "nd= %s not implemented " % self.nd
                #ebq
                #ebq_global
        elif self.turbulenceClosureFlag == 1: #Smagorinsky with 2p level set variation
            if self.nd == 2:
                #q
                self.calculateEddyViscosity_Smagorinsky2P_2D(self.smagorinskyConstant_0,
                                                             self.smagorinskyConstant_1,
                                                             self.eps_smagorinsky,
                                                             self.q_phi,
                                                             self.mesh.elementDiametersArray,
                                                             self.q[('grad(u)',1)],
                                                             self.q[('grad(u)',2)],
                                                             self.q_nu_t)
                #print "Smagorinsky preStep t=%s min=%s max=%s " % (t,self.q_nu_t.min(),self.q_nu_t.max())
                #ebqe
            elif self.nd == 3:
                #q
                self.calculateEddyViscosity_Smagorinsky2P_3D(self.smagorinskyConstant_0,
                                                           self.smagorinskyConstant_1,
                                                           self.eps_smagorinsky,
                                                           self.q_phi,
                                                           self.mesh.elementDiametersArray,
                                                           self.q[('grad(u)',1)],
                                                           self.q[('grad(u)',2)],
                                                           self.q[('grad(u)',3)],
                                                           self.q_nu_t)
                #print "Smagorinsky preStep t=%s min=%s max=%s " % (t,self.q_nu_t.min(),self.q_nu_t.max())
                #ebqe
            else:
                assert False, "nd= %s not implemented " % self.nd
                #ebq
                #ebq_global
    def evaluate(self,t,c):
        TwophaseNavierStokes_ST_LS_SO.evaluate(self,t,c)
        #update viscosity with eddy viscosity
        if self.turbulenceClosureFlag is not None:
            nu_t = None
            if c['x'].shape[:-1] == self.q_nu_t.shape[:]:
                nu_t = self.q_nu_t;
            else:
                return
            if self.nd == 2:
                if self.sd == True:
                    #print "before Smagorinsky eval t=%s nu_t: min=%s max=%s " % (t,nu_t.min(),nu_t.max())
                    #print "                           a(1,1): min=%s max=%s " % (c[('a',1,1)].min(),c[('a',1,1)].max())

                    eddyViscosity_2D_Update_sd(nu_t,
                                               c[('a',1,1)],
                                               c[('a',2,2)],
                                               c[('a',1,2)],
                                               c[('a',2,1)])
                    #print "after Smagorinsky eval t=%s a(1,1): min=%s max=%s " % (t,c[('a',1,1)].min(),c[('a',1,1)].max())
                else:
                    eddyViscosity_2D_Update(nu_t,
                                            c[('a',1,1)],
                                            c[('a',2,2)],
                                            c[('a',1,2)],
                                            c[('a',2,1)])
            elif self.nd == 3:
                if self.sd == True:
                    #print "before Smagorinsky eval t=%s nu_t: min=%s max=%s " % (t,nu_t.min(),nu_t.max())
                    #print "                           a(1,1): min=%s max=%s " % (c[('a',1,1)].min(),c[('a',1,1)].max())

                    eddyViscosity_3D_Update_sd(nu_t,
                                               c[('a',1,1)],
                                               c[('a',2,2)],
                                               c[('a',3,3)],
                                               c[('a',1,2)],
                                               c[('a',1,3)],
                                               c[('a',2,1)],
                                               c[('a',2,3)],
                                               c[('a',3,1)],
                                               c[('a',3,2)])
                    #print "after Smagorinsky eval t=%s a(1,1): min=%s max=%s " % (t,c[('a',1,1)].min(),c[('a',1,1)].max())
                else:
                    eddyViscosity_2D_Update(nu_t,
                                            c[('a',1,1)],
                                            c[('a',2,2)],
                                            c[('a',3,3)],
                                            c[('a',1,2)],
                                            c[('a',1,3)],
                                            c[('a',2,1)],
                                            c[('a',2,3)],
                                            c[('a',3,1)],
                                            c[('a',3,2)])
            else:
                assert False, "nd != 2,3 not done yet"

from proteus.ctransportCoefficients import burgersDiagonalVelEvaluate,burgersDiagonalVelHJEvaluate
class ViscousBurgersEqn(TC_base):
    def __init__(self,v=[1.0,0.0,0.0],nu=0.0,nc=1,nd=3,useHJ=False):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction ={}
        hamiltonian={}
        self.useHJ = useHJ #nonconservative form
        for i in range(nc):
            mass[i]     = {i:'linear'}
            advection[i]= {i:'nonlinear'}
            diffusion[i]= {i: {i:'constant'}}
            potential[i]= {i: {i:'u'}}
            reaction[i] = {i:'constant'}
            if self.useHJ:
                hamiltonian[i] = {i:{i:'nonlinear'}}

        #nc
        TC_base.__init__(self,nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
        self.nd= nd
        self.nu= numpy.zeros((nd,nd),'d')
        self.v = numpy.zeros((nd,),'d')
        for i in range(self.nd):
            self.v[i] = v[i]
            self.nu[i,i] = nu
        self.useC = True
    def evaluate(self,t,c):
        if self.useC:
            if not self.useHJ:
                for ci in range(self.nc):
                    burgersDiagonalVelEvaluate(self.nu[0,0],
                                               self.v,
                                               c[('u',ci)],
                                               c[('m',ci)],
                                               c[('dm',ci,ci)],
                                               c[('f',ci)],
                                               c[('df',ci,ci)],
                                               c[('a',ci,ci)],
                                               c[('phi',ci)],
                                               c[('dphi',ci,ci)])
            else:
                for ci in range(self.nc):
                    burgersDiagonalVelHJEvaluate(self.nu[0,0],
                                                 self.v,
                                                 c[('u',ci)],
                                                 c[('grad(u)',ci)],
                                                 c[('m',ci)],
                                                 c[('dm',ci,ci)],
                                                 c[('H',ci)],
                                                 c[('dH',ci,ci)],
                                                 c[('a',ci,ci)],
                                                 c[('phi',ci)],
                                                 c[('dphi',ci,ci)])

        else:
            nd = self.nd
            for ci in range(self.nc):
                c[('m',ci)].flat[:]    =c[('u',ci)].flat
                c[('dm',ci,ci)].flat[:]=1.0
                if self.useHJ:
                    for i in range(len(c[('u',ci)].flat)):
                        c[('dH',ci,ci)].flat[nd*i:nd*(i+1)]      = self.v.flat*c[('u',ci)].flat[i]
                        c[('H',ci)].flat[i]                      = c[('u',ci)].flat[i]*numpy.dot(self.v.flat[:],
                                                                                                 c[('grad(u)',ci)].flat[nd*i:nd*(i+1)])
                        c[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.nu.flat
                else:
                    for i in range(len(c[('u',ci)].flat)):
                        c[('f',ci)].flat[nd*i:nd*(i+1)]          = self.v.flat*0.5*(c[('u',ci)].flat[i]**2)
                        c[('df',ci,ci)].flat[nd*i:nd*(i+1)]      = self.v.flat*c[('u',ci)].flat[i]
                        c[('a',ci,ci)].flat[nd*nd*i:nd*nd*(i+1)] = self.nu.flat

            #ci
        #useC
    #ci
from proteus.ctransportCoefficients import evaluateBuckleyLeverettLiuExample
class BuckleyLeverettLiuExample(TC_base):
    r"""
    5 spot example from Liu 93 Siam paper.

    S_t + \deld (\vec a f(S)) = 0

    f(S) = S^2 / (0.2 - 0.4 S + 1.2 S^2)

    \vec a = (\pd{\phi}{x},\pd{phi}{y})

    \phi = 0.01 \log(\sqrt{x^2 + y^2})

    However, note that his example is for
    S=0 injecting into S=1 background saturation using his definition of f

    """
    def __init__(self,nu=0.0,nc=1,nd=2):
        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction ={}
        hamiltonian={}

        for i in range(nc):
            mass[i]     = {i:'linear'}
            advection[i]= {i:'nonlinear'}
            diffusion[i]= {i: {i:'constant'}}
            potential[i]= {i: {i:'u'}}
            reaction[i] = {i:'constant'}
        #nc
        TC_base.__init__(self,nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian)
        self.nd= nd
        self.nu= nu
        self.useC = True
        self.variableNames[0]='S_w'#check if really S_w or S_n
    def evaluate(self,t,c):
        if self.useC:
            for ci in range(self.nc):
                evaluateBuckleyLeverettLiuExample(c['x'],
                                                  c[('u',ci)],
                                                  c[('m',ci)],
                                                  c[('dm',ci,ci)],
                                                  c[('f',ci)],
                                                  c[('df',ci,ci)],
                                                  c[('a',ci,ci)]);
        #useC
    #ci
##\brief Incompressible Navier-Stokes equations for porous domain
#
#The equations are (needs to be updated)
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \mathbf{v} }{\partial t} + \nabla \cdot \left(\mathbf{v}\otimes \mathbf{v} - \nu \Delta \mathbf{f} \right) +\frac{\nabla p}{\rho} - \mathbf{g} &=& 0
#\f}
#
#where \f$\mathbf{v}\f$ is the velocity, \f$p\f$ is the pressure, \f$\nu\f$ is the kinematic viscosity, \f$\rho\f$ is the density, and \f$\mathbf{g}\f$ is the gravitational acceleration.
#
#The variable viscosity and density are given by
#
#\f{eqnarray*}
#\rho &=& \rho_0 (1-H) + \rho_1 H \\
#\nu  &=& \nu_0 (1-H) + \nu_1 H \\
#\f}
#
#with
#
#\f[
# H = \left\{ \begin{array}{lr}
# 0 & \phi \leq - \epsilon \\
# \frac{1}{2}\left(1+\frac{\phi}{\epsilon} + \frac{1}{\pi}\sin(\frac{\pi \phi}{\epsilon})\right) & -\epsilon < \phi < \epsilon \\
# 1 & \phi \geq \epsilon
#\end{array} \right.
#\f]
#
#The level set function, \f$\phi\f$, is provided from some other model (e.g. proteus::TransportCoeffcieints::NCLevelSetCoefficients) that is solved seperately (via operator splitting).
#
from .ctransportCoefficients import VolumeAveragedNavierStokesFullDevStress_2D_Evaluate
from .ctransportCoefficients import VolumeAveragedNavierStokesFullDevStress_3D_Evaluate
class VolumeAveragedNavierStokesFullDevStress(TC_base):
    """
    The coefficients for incompressible fluid in a porous domain governed by the Navier-Stokes equations

    """
    def __init__(self,
                 rho=998.2,mu=1.0e-3,
                 g=[0.0,-9.8],
                 nd=2,
                 meanGrainSize=0.01,
                 setParamsFunc=None, #uses setParamsFunc if given
                 stokesOnly=False,
                 meanGrainSizeTypes=None, #otherwise can use element constant values
                 porosityTypes=None):
        self.rho = rho
        self.mu  = mu
        self.g = numpy.array(g)
        self.nd=nd
        #
        self.meanGrainSize     = meanGrainSize
        self.setParamsFunc=setParamsFunc
        self.stokesOnly = stokesOnly
        self.meanGrainSizeTypes = meanGrainSizeTypes
        self.porosityTypes      = porosityTypes
        #
        self.ebq_porosity = None
        self.ebq_meanGrain = None
        self.ebq_global_porosity = None
        self.ebq_global_meanGrain = None


        mass={}
        advection={}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        if nd==2:
            variableNames=['p','u','v']
            mass= {1:{1:'linear'},
                   2:{2:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear'}}
            diffusion  = {1:{1:{1:'constant'},2:{2:'constant'}},
                          2:{2:{2:'constant'},1:{1:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'}}
            reaction = {1:{1:'nonlinear',2:'nonlinear'},
                        2:{1:'nonlinear',2:'nonlinear'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2]
        elif nd==3:
            variableNames=['p','u','v','w']
            mass = {1:{1:'linear'},
                    2:{2:'linear'},
                    3:{3:'linear'}}
            advection = {0:{1:'linear',
                            2:'linear',
                            3:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'},
                         3:{1:'nonlinear',
                            2:'nonlinear',
                            3:'nonlinear'}}
            diffusion = {1:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}},
                         2:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}},
                         3:{1:{1:'constant'},2:{2:'constant'},3:{3:'constant'}}}
            potential= {1:{1:'u'},
                        2:{2:'u'},
                        3:{3:'u'}}
            reaction = {1:{1:'nonlinear',2:'nonlinear',3:'nonlinear'},
                        2:{1:'nonlinear',2:'nonlinear',3:'nonlinear'},
                        3:{1:'nonlinear',2:'nonlinear',3:'nonlinear'}}
            hamiltonian = {1:{0:'linear'},
                           2:{0:'linear'},
                           3:{0:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
    def initializeMesh(self,mesh):
        self.elementMaterialTypes = mesh.elementMaterialTypes
        self.mesh = mesh
    def initializeElementQuadrature(self,t,cq):
        self.q_porosity = numpy.ones(cq[('u',0)].shape,'d')
        self.q_meanGrain= numpy.ones(cq[('u',0)].shape,'d')
        self.q_meanGrain.flat[:] = self.meanGrainSize
        if self.setParamsFunc is not None:
            self.setParamsFunc(cq['x'],self.q_porosity,self.q_meanGrain)
        else:
            if self.porosityTypes is not None:
                for eN in range(self.q_porosity.shape[0]):
                    self.q_porosity[eN,:] = self.porosityTypes[self.elementMaterialTypes[eN]]
            if self.meanGrainSizeTypes is not None:
                for eN in range(self.q_meanGrain.shape[0]):
                    self.q_meanGrain[eN,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN]]
        #
        #mwf hack
        cq['porosity_vis'] = self.q_porosity
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.ebq_porosity = numpy.ones(cebq[('u',0)].shape,'d')
        self.ebq_meanGrain = numpy.ones(cebq[('u',0)].shape,'d')
        self.ebq_meanGrain.flat[:] = self.meanGrainSize
        self.ebq_global_porosity = numpy.ones(cebq_global[('u',0)].shape,'d')
        self.ebq_global_meanGrain = numpy.ones(cebq_global[('u',0)].shape,'d')
        self.ebq_global_meanGrain.flat[:] = self.meanGrainSize
        if self.setParamsFunc is not None:
            self.setParamsFunc(cebq['x'],self.ebq_porosity,self.ebq_meanGrain)
            self.setParamsFunc(cebq_global['x'],self.ebq_global_porosity,self.ebq_global_meanGrain)
        else:
            #\todo which mean to use or leave discontinuous, add ebq_global
            if self.porosityTypes is not None:
                for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                    ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                    eN_left  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    eN_right = self.mesh.elementBoundaryElementsArray[ebN,1]
                    ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                    ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
                    avg = 0.5*(self.porosityTypes[self.elementMaterialTypes[eN_left]]+
                               self.porosityTypes[self.elementMaterialTypes[eN_right]])
                    self.ebq_porosity[eN_left,ebN_element_left,:]  = self.porosityTypes[self.elementMaterialTypes[eN_left]]
                    self.ebq_porosity[eN_right,ebN_element_right,:]= self.porosityTypes[self.elementMaterialTypes[eN_right]]
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                    self.ebq_porosity[eN,ebN_element,:] = self.porosityTypes[self.elementMaterialTypes[eN]]
            if self.meanGrainSizeTypes is not None:
                for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                    ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                    eN_left  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    eN_right = self.mesh.elementBoundaryElementsArray[ebN,1]
                    ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                    ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
                    avg = 0.5*(self.meanGrainSizeTypes[self.elementMaterialTypes[eN_left]]+
                               self.meanGrainSizeTypes[self.elementMaterialTypes[eN_right]])
                    self.ebq_meanGrain[eN_left,ebN_element_left,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN_left]]
                    self.ebq_meanGrain[eN_right,ebN_element_right,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN_right]]
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                    self.ebq_meanGrain[eN,ebN_element,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN]]
         #
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.ebqe_porosity = numpy.ones(cebqe[('u',0)].shape,'d')
        self.ebqe_meanGrain= numpy.ones(cebqe[('u',0)].shape,'d')
        if self.setParamsFunc is not None:
            self.setParamsFunc(cebqe['x'],self.ebqe_porosity,self.ebqe_meanGrain)
        else:
            if self.porosityTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    self.ebqe_porosity[ebNE,:] = self.porosityTypes[self.elementMaterialTypes[eN]]
            if self.meanGrainSizeTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    self.ebqe_meanGrain[ebNE,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN]]
        #
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        self.ip_porosity = numpy.ones(cip[('u',0)].shape,'d')
        self.ip_meanGrain= numpy.ones(cip[('u',0)].shape,'d')
        self.ip_meanGrain.flat[:] = self.meanGrainSize
        if self.setParamsFunc is not None:
            self.setParamsFunc(cip['x'],self.ip_porosity,self.ip_meanGrain)
        else:
            if self.porosityTypes is not None:
                for eN in range(self.ip_porosity.shape[0]):
                    self.ip_porosity[eN,:] = self.porosityTypes[self.elementMaterialTypes[eN]]
            if self.meanGrainSizeTypes is not None:
                for eN in range(self.ip_meanGrain.shape[0]):
                    self.ip_meanGrain[eN,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN]]
        #
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_porosity.shape:
            porosity = self.q_porosity
            meanGrain= self.q_meanGrain
        elif c[('u',0)].shape == self.ebqe_porosity.shape:
            porosity = self.ebqe_porosity
            meanGrain= self.ebqe_meanGrain
        elif c[('u',0)].shape == self.ebq_global_porosity.shape:
            porosity = self.ebq_global_porosity
            meanGrain= self.ebq_global_meanGrain
        elif c[('u',0)].shape == self.ip_porosity.shape:
            porosity = self.ip_porosity
            meanGrain= self.ip_meanGrain
        else:
            porosity = self.ebq_porosity
            meanGrain= self.ebq_meanGrain

        if self.nd==2:
            VolumeAveragedNavierStokesFullDevStress_2D_Evaluate(self.rho,
                                                                self.mu,
                                                                meanGrain,
                                                                self.g,
                                                                c[('u',0)],
                                                                c[('grad(u)',0)],
                                                                c[('u',1)],
                                                                c[('u',2)],
                                                                porosity,
                                                                c[('m',1)],
                                                                c[('dm',1,1)],
                                                                c[('m',2)],
                                                                c[('dm',2,2)],
                                                                c[('f',0)],
                                                                c[('df',0,1)],
                                                                c[('df',0,2)],
                                                                c[('f',1)],
                                                                c[('df',1,1)],
                                                                c[('df',1,2)],
                                                                c[('f',2)],
                                                                c[('df',2,1)],
                                                                c[('df',2,2)],
                                                                c[('a',1,1)],
                                                                c[('a',2,2)],
                                                                c[('a',1,2)],
                                                                c[('a',2,1)],
                                                                c[('r',1)],
                                                                c[('r',2)],
                                                                c[('dr',1,1)],
                                                                c[('dr',1,2)],
                                                                c[('dr',2,1)],
                                                                c[('dr',2,2)],
                                                                c[('H',1)],
                                                                c[('dH',1,0)],
                                                                c[('H',2)],
                                                                c[('dH',2,0)])

            if self.stokesOnly:
                c[('f',1)].flat[:]=0.0; c[('f',2)].flat[:]=0.0;
                c[('df',1,1)].flat[:]=0.0;c[('df',1,2)].flat[:]=0.0;
                c[('df',2,2)].flat[:]=0.0;c[('df',2,1)].flat[:]=0.0;
        elif  self.nd==3:
            VolumeAveragedNavierStokesFullDevStress_3D_Evaluate(self.rho,
                                                                self.mu,
                                                                meanGrain,
                                                                self.g,
                                                                c[('u',0)],
                                                                c[('grad(u)',0)],
                                                                c[('u',1)],
                                                                c[('u',2)],
                                                                c[('u',3)],
                                                                porosity,
                                                                c[('m',1)],
                                                                c[('dm',1,1)],
                                                                c[('m',2)],
                                                                c[('dm',2,2)],
                                                                c[('m',3)],
                                                                c[('dm',3,3)],
                                                                c[('f',0)],
                                                                c[('df',0,1)],
                                                                c[('df',0,2)],
                                                                c[('df',0,3)],
                                                                c[('f',1)],
                                                                c[('df',1,1)],
                                                                c[('df',1,2)],
                                                                c[('df',1,3)],
                                                                c[('f',2)],
                                                                c[('df',2,1)],
                                                                c[('df',2,2)],
                                                                c[('df',2,3)],
                                                                c[('f',3)],
                                                                c[('df',3,1)],
                                                                c[('df',3,2)],
                                                                c[('df',3,3)],
                                                                c[('a',1,1)],
                                                                c[('a',2,2)],
                                                                c[('a',3,3)],
                                                                c[('a',1,2)],
                                                                c[('a',1,3)],
                                                                c[('a',2,1)],
                                                                c[('a',2,3)],
                                                                c[('a',3,1)],
                                                                c[('a',3,2)],
                                                                c[('r',1)],
                                                                c[('r',2)],
                                                                c[('r',3)],
                                                                c[('dr',1,1)],
                                                                c[('dr',1,2)],
                                                                c[('dr',1,3)],
                                                                c[('dr',2,1)],
                                                                c[('dr',2,2)],
                                                                c[('dr',2,3)],
                                                                c[('dr',3,1)],
                                                                c[('dr',3,2)],
                                                                c[('dr',3,3)],
                                                                c[('H',1)],
                                                                c[('dH',1,0)],
                                                                c[('H',2)],
                                                                c[('dH',2,0)],
                                                                c[('H',3)],
                                                                c[('dH',3,0)])

            if self.stokesOnly:
                c[('f',1)].flat[:]=0.0; c[('f',2)].flat[:]=0.0; c[('f',3)].flat[:]=0.0;
                c[('df',1,1)].flat[:]=0.0;c[('df',1,2)].flat[:]=0.0; c[('df',1,3)].flat[:]=0.0;
                c[('df',2,2)].flat[:]=0.0;c[('df',2,1)].flat[:]=0.0; c[('df',2,3)].flat[:]=0.0;
                c[('df',3,3)].flat[:]=0.0;c[('df',3,2)].flat[:]=0.0; c[('df',3,1)].flat[:]=0.0;
#         nPoints=1
#         for d in c['x'].shape[:-1]:
#             nPoints*=d
#         for k in range(nPoints):
#             if c[('r',1)].flat[k] != 0.0 and c[('r',2)].flat[k] != 0.0:
#                 print "x %12.5e y %12.5e \n" % (c['x'].flat[k*3+0]-0.5,c['x'].flat[k*3+1]-0.5)
#                 print "sf_x %12.5e sf_y %12.5e \n" % (c[('r',1)].flat[k],c[('r',2)].flat[k])


##\brief Two-phase, Volume-Averaged Incompressible Navier-Stokes equations (level-set formulation)
#
#The equations are (need to be updated)
#
#\f{eqnarray*}
#\nabla \cdot \mathbf{v} &=& 0 \\
#\frac{\partial \mathbf{v} }{\partial t} + \nabla \cdot \left(\mathbf{v}\otimes \mathbf{v} - \nu \Delta \mathbf{f} \right) +\frac{\nabla p}{\rho} - \mathbf{g} &=& 0
#\f}
#
#where \f$\mathbf{v}\f$ is the velocity, \f$p\f$ is the pressure, \f$\nu\f$ is the kinematic viscosity, \f$\rho\f$ is the density, and \f$\mathbf{g}\f$ is the gravitational acceleration.
#
#The variable viscosity and density are given by
#
#\f{eqnarray*}
#\rho &=& \rho_0 (1-H) + \rho_1 H \\
#\nu  &=& \nu_0 (1-H) + \nu_1 H \\
#\f}
#
#with
#
#\f[
# H = \left\{ \begin{array}{lr}
# 0 & \phi \leq - \epsilon \\
# \frac{1}{2}\left(1+\frac{\phi}{\epsilon} + \frac{1}{\pi}\sin(\frac{\pi \phi}{\epsilon})\right) & -\epsilon < \phi < \epsilon \\
# 1 & \phi \geq \epsilon
#\end{array} \right.
#\f]
#
#The level set function, \f$\phi\f$, is provided from some other model (e.g. proteus::TransportCoeffcieints::NCLevelSetCoefficients) that is solved seperately (via operator splitting).
#
from .ctransportCoefficients import VolumeAveragedTwophaseNavierStokes_ST_LS_SO_2D_Evaluate
from .ctransportCoefficients import VolumeAveragedTwophaseNavierStokes_ST_LS_SO_2D_Evaluate_sd
from .ctransportCoefficients import VolumeAveragedTwophaseNavierStokes_ST_LS_SO_3D_Evaluate
from .ctransportCoefficients import VolumeAveragedTwophaseNavierStokes_ST_LS_SO_3D_Evaluate_sd
class VolumeAveragedTwophaseNavierStokes(TwophaseReynoldsAveragedNavierStokes_AlgebraicClosure):
    """
    The coefficients for two incompresslble fluids governed by the Navier-Stokes equations and separated by a sharp interface represented by a level set function
    """
    def __init__(self,
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2,nu_0=1.004e-6,
                 rho_1=1.205,nu_1=1.500e-5,
                 g=[0.0,-9.8],
                 nd=2,
                 LS_model=3,
                 KN_model=None,
                 epsFact_density=None,
                 meanGrainSize=0.01,
                 setParamsFunc=None,      #uses setParamsFunc if given
                 stokes=False,
                 sd=True,
                 movingDomain=False,
                 turbulenceClosureFlag=None,
                 smagorinskyConstant_0=0.1,
                 smagorinskyConstant_1=0.1,
                 epsFact_smagorinsky=0.33,
                 meanGrainSizeTypes=None, #otherwise can use element constant values
                 porosityTypes=None,
                 killNonlinearDrag=False,
                 epsFact_source=1.):

        TwophaseReynoldsAveragedNavierStokes_AlgebraicClosure.__init__(self,
                                                                       epsFact=epsFact,
                                                                       sigma=sigma,
                                                                       rho_0=rho_0,nu_0=nu_0,
                                                                       rho_1=rho_1,nu_1=nu_1,
                                                                       g=g,
                                                                       nd=nd,
                                                                       LS_model=LS_model,
                                                                       KN_model=KN_model,
                                                                       epsFact_density=epsFact_density,
                                                                       stokes=stokes,
                                                                       sd=sd,
                                                                       movingDomain=movingDomain,
                                                                       turbulenceClosureFlag=turbulenceClosureFlag,
                                                                       smagorinskyConstant_0=smagorinskyConstant_0,
                                                                       smagorinskyConstant_1=smagorinskyConstant_1,
                                                                       epsFact_smagorinsky=epsFact_smagorinsky)
        #
        self.meanGrainSize     = meanGrainSize
        self.setParamsFunc=setParamsFunc
        self.meanGrainSizeTypes = meanGrainSizeTypes
        self.porosityTypes      = porosityTypes
        self.killNonlinearDrag  = int(killNonlinearDrag)
        #just modify reaction dictionary keys to be nonlinear
        if nd==2:
            self.reaction = {0:{0:'constant'},
                             1:{1:'nonlinear',2:'nonlinear'},
                             2:{1:'nonlinear',2:'nonlinear'}}
        elif nd==3:
            self.reaction = {0:{0:'constant'},
                             1:{1:'nonlinear',2:'nonlinear',3:'nonlinear'},
                             2:{1:'nonlinear',2:'nonlinear',3:'nonlinear'},
                             3:{1:'nonlinear',2:'nonlinear',3:'nonlinear'}}

        for ci,cjDict in self.reaction.items():
            self.elementIntegralKeys.append(('r',ci))
            for cj in cjDict:
                self.stencil[ci].add(cj)
        self.epsFact_source=epsFact_source
        self.linearDragFactor = 1.0; self.nonlinearDragFactor = 1.0
        if self.killNonlinearDrag:
            self.nonlinearDragFactor = 0.0
    def initializeMesh(self,mesh):
        TwophaseReynoldsAveragedNavierStokes_AlgebraicClosure.initializeMesh(self,mesh)
        self.elementMaterialTypes = mesh.elementMaterialTypes
        self.eps_source=self.epsFact_source*mesh.h
    def initializeElementQuadrature(self,t,cq):
        TwophaseReynoldsAveragedNavierStokes_AlgebraicClosure.initializeElementQuadrature(self,t,cq)
        self.q_porosity = numpy.ones(cq[('u',1)].shape,'d')
        self.q_meanGrain= numpy.ones(cq[('u',1)].shape,'d')
        self.q_meanGrain.fill(self.meanGrainSize)
        if self.setParamsFunc is not None:
            self.setParamsFunc(cq['x'],self.q_porosity,self.q_meanGrain)
        else:
            #TODO make loops faster
            if self.porosityTypes is not None:
                for eN in range(self.q_porosity.shape[0]):
                    self.q_porosity[eN,:] = self.porosityTypes[self.elementMaterialTypes[eN]]
            if self.meanGrainSizeTypes is not None:
                for eN in range(self.q_meanGrain.shape[0]):
                    self.q_meanGrain[eN,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN]]
        #
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseReynoldsAveragedNavierStokes_AlgebraicClosure.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        self.ebq_porosity = numpy.ones(cebq[('u',1)].shape,'d')
        self.ebq_meanGrain = numpy.ones(cebq[('u',1)].shape,'d')
        self.ebq_meanGrain.fill(self.meanGrainSize)
        if self.setParamsFunc is not None:
            self.setParamsFunc(cebq['x'],self.ebq_porosity,self.ebq_meanGrain)
        #TODO which mean to use or leave discontinuous
        #TODO make loops faster
        if self.porosityTypes is not None:
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                eN_left  = self.mesh.elementBoundaryElementsArray[ebN,0]
                eN_right = self.mesh.elementBoundaryElementsArray[ebN,1]
                ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
                avg = 0.5*(self.porosityTypes[self.elementMaterialTypes[eN_left]]+
                           self.porosityTypes[self.elementMaterialTypes[eN_right]])
                self.ebq_porosity[eN_left,ebN_element_left,:]  = self.porosityTypes[self.elementMaterialTypes[eN_left]]
                self.ebq_porosity[eN_right,ebN_element_right,:]= self.porosityTypes[self.elementMaterialTypes[eN_right]]
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                self.ebq_porosity[eN,ebN_element,:] = self.porosityTypes[self.elementMaterialTypes[eN]]
        if self.meanGrainSizeTypes is not None:
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                eN_left  = self.mesh.elementBoundaryElementsArray[ebN,0]
                eN_right = self.mesh.elementBoundaryElementsArray[ebN,1]
            ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
            avg = 0.5*(self.meanGrainSizeTypes[self.elementMaterialTypes[eN_left]]+
                       self.meanGrainSizeTypes[self.elementMaterialTypes[eN_right]])
            self.ebq_meanGrain[eN_left,ebN_element_left,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN_left]]
            self.ebq_meanGrain[eN_right,ebN_element_right,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN_right]]
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                self.ebq_meanGrain[eN,ebN_element,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN]]
         #
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseReynoldsAveragedNavierStokes_AlgebraicClosure.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        self.ebqe_porosity = numpy.ones(cebqe[('u',1)].shape,'d')
        self.ebqe_meanGrain = numpy.ones(cebqe[('u',1)].shape,'d')
        self.ebqe_meanGrain.fill(self.meanGrainSize)
        #TODO make loops faster
        if self.setParamsFunc is not None:
            self.setParamsFunc(cebqe['x'],self.ebqe_porosity,self.ebqe_meanGrain)
        else:
            if self.porosityTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    self.ebqe_porosity[ebNE,:] = self.porosityTypes[self.elementMaterialTypes[eN]]
            if self.meanGrainSizeTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    self.ebqe_meanGrain[ebNE,:] = self.meanGrainSizeTypes[self.elementMaterialTypes[eN]]
        #
    def preStep(self,t,firstStep=False):
        TwophaseReynoldsAveragedNavierStokes_AlgebraicClosure.preStep(self,t,firstStep=firstStep)
        #TODO really need to modify turbulence model in porous region, account for mean grain size etc
        if self.turbulenceClosureFlag is not None:
            self.q_nu_t *= self.q_porosity
    def evaluate(self,t,c):
        import pdb
        phi = None; n = None; kappa = None; porosity = None; meanGrain = None

        if c[('u',0)].shape == self.q_phi.shape:
            phi = self.q_phi
            n   = self.q_n
            kappa = self.q_kappa
            porosity = self.q_porosity
            meanGrain= self.q_meanGrain
        elif c[('u',0)].shape == self.ebqe_phi.shape:
            phi = self.ebqe_phi
            n = self.ebqe_n
            kappa = self.ebqe_kappa
            porosity = self.ebqe_porosity
            meanGrain= self.ebqe_meanGrain
        else:
            phi = self.ebq_phi
            n = self.ebq_n
            kappa = self.ebq_kappa
            porosity = self.ebq_porosity
            meanGrain= self.ebq_meanGrain
        #
        if phi is None or n is None or kappa is None or porosity is None or meanGrain is None:
            pdb.set_trace()
        if self.nd==2:
            if self.sd:
                VolumeAveragedTwophaseNavierStokes_ST_LS_SO_2D_Evaluate_sd(self.killNonlinearDrag,
                                                                           self.eps_density,
                                                                           self.eps_viscosity,
                                                                           self.sigma,
                                                                           self.rho_0,
                                                                           self.nu_0,
                                                                           self.rho_1,
                                                                           self.nu_1,
                                                                           meanGrain,
                                                                           self.g,
                                                                           phi,
                                                                           n,
                                                                           kappa,
                                                                           c[('u',0)],
                                                                           c[('grad(u)',0)],
                                                                           c[('u',1)],
                                                                           c[('u',2)],
                                                                           porosity,
                                                                           c[('m',1)],
                                                                           c[('dm',1,1)],
                                                                           c[('m',2)],
                                                                           c[('dm',2,2)],
                                                                           c[('f',0)],
                                                                           c[('df',0,1)],
                                                                           c[('df',0,2)],
                                                                           c[('f',1)],
                                                                           c[('df',1,1)],
                                                                           c[('df',1,2)],
                                                                           c[('f',2)],
                                                                           c[('df',2,1)],
                                                                           c[('df',2,2)],
                                                                           c[('a',1,1)],
                                                                           c[('a',2,2)],
                                                                           c[('a',1,2)],
                                                                           c[('a',2,1)],
                                                                           c[('r',1)],
                                                                           c[('r',2)],
                                                                           c[('dr',1,1)],
                                                                           c[('dr',1,2)],
                                                                           c[('dr',2,1)],
                                                                           c[('dr',2,2)],
                                                                           c[('H',1)],
                                                                           c[('dH',1,0)],
                                                                           c[('H',2)],
                                                                           c[('dH',2,0)])
            else:
                VolumeAveragedTwophaseNavierStokes_ST_LS_SO_2D_Evaluate(self.killNonlinearDrag,
                                                                        self.eps_density,
                                                                        self.eps_viscosity,
                                                                        self.sigma,
                                                                        self.rho_0,
                                                                        self.nu_0,
                                                                        self.rho_1,
                                                                        self.nu_1,
                                                                        meanGrain,
                                                                        self.g,
                                                                        phi,
                                                                        n,
                                                                        kappa,
                                                                        c[('u',0)],
                                                                        c[('grad(u)',0)],
                                                                        c[('u',1)],
                                                                        c[('u',2)],
                                                                        porosity,
                                                                        c[('m',1)],
                                                                        c[('dm',1,1)],
                                                                        c[('m',2)],
                                                                        c[('dm',2,2)],
                                                                        c[('f',0)],
                                                                        c[('df',0,1)],
                                                                        c[('df',0,2)],
                                                                        c[('f',1)],
                                                                        c[('df',1,1)],
                                                                        c[('df',1,2)],
                                                                        c[('f',2)],
                                                                        c[('df',2,1)],
                                                                        c[('df',2,2)],
                                                                        c[('a',1,1)],
                                                                        c[('a',2,2)],
                                                                        c[('a',1,2)],
                                                                        c[('a',2,1)],
                                                                        c[('r',1)],
                                                                        c[('r',2)],
                                                                        c[('dr',1,1)],
                                                                        c[('dr',1,2)],
                                                                        c[('dr',2,1)],
                                                                        c[('dr',2,2)],
                                                                        c[('H',1)],
                                                                        c[('dH',1,0)],
                                                                        c[('H',2)],
                                                                        c[('dH',2,0)])

            #
            if self.stokes:
                c[('f',1)].flat[:]=0.0; c[('f',2)].flat[:]=0.0;
                c[('df',1,1)].flat[:]=0.0;c[('df',1,2)].flat[:]=0.0;
                c[('df',2,2)].flat[:]=0.0;c[('df',2,1)].flat[:]=0.0;
            #
        elif  self.nd==3:
            if self.sd:
                VolumeAveragedTwophaseNavierStokes_ST_LS_SO_3D_Evaluate_sd(self.killNonlinearDrag,
                                                                           self.eps_density,
                                                                           self.eps_viscosity,
                                                                           self.sigma,
                                                                           self.rho_0,
                                                                           self.nu_0,
                                                                           self.rho_1,
                                                                           self.nu_1,
                                                                           meanGrain,
                                                                           self.g,
                                                                           phi,
                                                                           n,
                                                                           kappa,
                                                                           c[('u',0)],
                                                                           c[('grad(u)',0)],
                                                                           c[('u',1)],
                                                                           c[('u',2)],
                                                                           c[('u',3)],
                                                                           porosity,
                                                                           c[('m',1)],
                                                                           c[('dm',1,1)],
                                                                           c[('m',2)],
                                                                           c[('dm',2,2)],
                                                                           c[('m',3)],
                                                                           c[('dm',3,3)],
                                                                           c[('f',0)],
                                                                           c[('df',0,1)],
                                                                           c[('df',0,2)],
                                                                           c[('df',0,3)],
                                                                           c[('f',1)],
                                                                           c[('df',1,1)],
                                                                           c[('df',1,2)],
                                                                           c[('df',1,3)],
                                                                           c[('f',2)],
                                                                           c[('df',2,1)],
                                                                           c[('df',2,2)],
                                                                           c[('df',2,3)],
                                                                           c[('f',3)],
                                                                           c[('df',3,1)],
                                                                           c[('df',3,2)],
                                                                           c[('df',3,3)],
                                                                           c[('a',1,1)],
                                                                           c[('a',2,2)],
                                                                           c[('a',3,3)],
                                                                           c[('a',1,2)],
                                                                           c[('a',1,3)],
                                                                           c[('a',2,1)],
                                                                           c[('a',2,3)],
                                                                           c[('a',3,1)],
                                                                           c[('a',3,2)],
                                                                           c[('r',1)],
                                                                           c[('r',2)],
                                                                           c[('r',3)],
                                                                           c[('dr',1,1)],
                                                                           c[('dr',1,2)],
                                                                           c[('dr',1,3)],
                                                                           c[('dr',2,1)],
                                                                           c[('dr',2,2)],
                                                                           c[('dr',2,3)],
                                                                           c[('dr',3,1)],
                                                                           c[('dr',3,2)],
                                                                           c[('dr',3,3)],
                                                                           c[('H',1)],
                                                                           c[('dH',1,0)],
                                                                           c[('H',2)],
                                                                           c[('dH',2,0)],
                                                                           c[('H',3)],
                                                                           c[('dH',3,0)])
            else:
                VolumeAveragedTwophaseNavierStokes_ST_LS_SO_3D_Evaluate(self.killNonlinearDrag,
                                                                        self.eps_density,
                                                                        self.eps_viscosity,
                                                                        self.sigma,
                                                                        self.rho_0,
                                                                        self.nu_0,
                                                                        self.rho_1,
                                                                        self.nu_1,
                                                                        meanGrain,
                                                                        self.g,
                                                                        phi,
                                                                        n,
                                                                        kappa,
                                                                        c[('u',0)],
                                                                        c[('grad(u)',0)],
                                                                        c[('u',1)],
                                                                        c[('u',2)],
                                                                        c[('u',3)],
                                                                        porosity,
                                                                        c[('m',1)],
                                                                        c[('dm',1,1)],
                                                                        c[('m',2)],
                                                                        c[('dm',2,2)],
                                                                        c[('m',3)],
                                                                        c[('dm',3,3)],
                                                                        c[('f',0)],
                                                                        c[('df',0,1)],
                                                                        c[('df',0,2)],
                                                                        c[('df',0,3)],
                                                                        c[('f',1)],
                                                                        c[('df',1,1)],
                                                                        c[('df',1,2)],
                                                                        c[('df',1,3)],
                                                                        c[('f',2)],
                                                                        c[('df',2,1)],
                                                                        c[('df',2,2)],
                                                                        c[('df',2,3)],
                                                                        c[('f',3)],
                                                                        c[('df',3,1)],
                                                                        c[('df',3,2)],
                                                                        c[('df',3,3)],
                                                                        c[('a',1,1)],
                                                                        c[('a',2,2)],
                                                                        c[('a',3,3)],
                                                                        c[('a',1,2)],
                                                                        c[('a',1,3)],
                                                                        c[('a',2,1)],
                                                                        c[('a',2,3)],
                                                                        c[('a',3,1)],
                                                                        c[('a',3,2)],
                                                                        c[('r',1)],
                                                                        c[('r',2)],
                                                                        c[('r',3)],
                                                                        c[('dr',1,1)],
                                                                        c[('dr',1,2)],
                                                                        c[('dr',1,3)],
                                                                        c[('dr',2,1)],
                                                                        c[('dr',2,2)],
                                                                        c[('dr',2,3)],
                                                                        c[('dr',3,1)],
                                                                        c[('dr',3,2)],
                                                                        c[('dr',3,3)],
                                                                        c[('H',1)],
                                                                        c[('dH',1,0)],
                                                                        c[('H',2)],
                                                                        c[('dH',2,0)],
                                                                        c[('H',3)],
                                                                        c[('dH',3,0)])

        #update viscosity with eddy viscosity
        if self.turbulenceClosureFlag is not None:
            nu_t = None
            if c['x'].shape[:-1] == self.q_nu_t.shape[:]:
                nu_t = self.q_nu_t;
                if self.nd == 2:
                    if self.sd == True:
                        #print "before Smagorinsky eval t=%s nu_t: min=%s max=%s " % (t,nu_t.min(),nu_t.max())
                        #print "                           a(1,1): min=%s max=%s " % (c[('a',1,1)].min(),c[('a',1,1)].max())

                        self.eddyViscosity_2D_Update_sd(nu_t,
                                                        c[('a',1,1)],
                                                        c[('a',2,2)],
                                                        c[('a',1,2)],
                                                        c[('a',2,1)])
                        #print "after Smagorinsky eval t=%s a(1,1): min=%s max=%s " % (t,c[('a',1,1)].min(),c[('a',1,1)].max())
                    else:
                        self.eddyViscosity_2D_Update(nu_t,
                                                     c[('a',1,1)],
                                                     c[('a',2,2)],
                                                     c[('a',1,2)],
                                                     c[('a',2,1)])
                elif self.nd == 3:
                    if self.sd == True:
                        #print "before Smagorinsky eval t=%s nu_t: min=%s max=%s " % (t,nu_t.min(),nu_t.max())
                        #print "                           a(1,1): min=%s max=%s " % (c[('a',1,1)].min(),c[('a',1,1)].max())

                        self.eddyViscosity_3D_Update_sd(nu_t,
                                                        c[('a',1,1)],
                                                        c[('a',2,2)],
                                                        c[('a',3,3)],
                                                        c[('a',1,2)],
                                                        c[('a',1,3)],
                                                        c[('a',2,1)],
                                                        c[('a',2,3)],
                                                        c[('a',3,1)],
                                                        c[('a',3,2)])
                        #print "after Smagorinsky eval t=%s a(1,1): min=%s max=%s " % (t,c[('a',1,1)].min(),c[('a',1,1)].max())
                    else:
                        self.eddyViscosity_3D_Update(nu_t,
                                                     c[('a',1,1)],
                                                     c[('a',2,2)],
                                                     c[('a',3,3)],
                                                     c[('a',1,2)],
                                                     c[('a',1,3)],
                                                     c[('a',2,1)],
                                                     c[('a',2,3)],
                                                     c[('a',3,1)],
                                                     c[('a',3,2)])
                else:
                    assert False, "nd != 2,3 not done yet"
        if 'x' in c and len(c['x'].shape) == 3:
            if self.nd == 2:
                c[('r',0)].fill(0.0)
                eps_source=self.eps_source
                if numpy.isnan(c[('r',0)].any()):
                    import pdb
                    pdb.set_trace()
            else:
                c[('r',0)].fill(0.0)
                eps_source=self.eps_source

########################################################################
#VOF coefficients when have variable porosity term
########################################################################
from .ctransportCoefficients import VolumeAveragedVOFCoefficientsEvaluate
from .cfemIntegrals import copyExteriorElementBoundaryValuesFromElementBoundaryValues
class VolumeAveragedVOFCoefficients(VOFCoefficients):
    def __init__(self,LS_model=-1,V_model=0,RD_model=-1,ME_model=1,checkMass=True,epsFact=0.0,
                 setParamsFunc=None):
        VOFCoefficients.__init__(self,
                                 LS_model=LS_model,
                                 V_model=V_model,
                                 RD_model=RD_model,
                                 ME_model=ME_model,
                                 checkMass=checkMass,
                                 epsFact=epsFact)
        self.setParamsFunc   = setParamsFunc
        self.flowCoefficients=None
        self.q_porosity = None; self.ebq_porosity = None; self.ebqe_porosity = None
    #
    def attachModels(self,modelList):
        VOFCoefficients.attachModels(self,modelList)
        self.flowCoefficients = modelList[self.flowModelIndex].coefficients
        if hasattr(self.flowCoefficients,'q_porosity'):
            self.q_porosity = self.flowCoefficients.q_porosity
        else:
            self.q_porosity = numpy.ones(modelList[self.modelIndex].q[('u',0)].shape,
                                           'd')
            if self.setParamsFunc is not None:
                self.setParamsFunc(modelList[self.modelIndex].q['x'],self.q_porosity)
            #
        #
        if hasattr(self.flowCoefficients,'ebq_porosity'):
            self.ebq_porosity = self.flowCoefficients.ebq_porosity
        elif ('u',0) in modelList[self.modelIndex].ebq:
            self.ebq_porosity = numpy.ones(modelList[self.modelIndex].ebq[('u',0)].shape,
                                           'd')
            if self.setParamsFunc is not None:
                self.setParamsFunc(modelList[self.modelIndex].ebq['x'],self.ebq_porosity)
            #
        #
        if hasattr(self.flowCoefficients,'ebqe_porosity'):
            self.ebqe_porosity = self.flowCoefficients.ebqe_porosity
        else:
            self.ebqe_porosity = numpy.ones(modelList[self.LS_modelIndex].ebqe[('u',0)].shape,
                                            'd')
            if self.setParamsFunc is not None:
                self.setParamsFunc(modelList[self.LS_modelIndex].ebqe['x'],self.ebqe_porosity)
            #
        #


    #
    def evaluate(self,t,c):

        if c[('f',0)].shape == self.q_v.shape:
            v = self.q_v
            phi = self.q_phi
            porosity  = self.q_porosity
        elif c[('f',0)].shape == self.ebqe_v.shape:
            v = self.ebqe_v
            phi = self.ebqe_phi
            porosity  = self.ebq_porosity
        elif ((self.ebq_v is not None and self.ebq_phi is not None) and c[('f',0)].shape == self.ebq_v.shape):
            v = self.ebq_v
            phi = self.ebq_phi
            porosity  = self.ebq_porosity
        else:
            v=None
            phi=None
            porosity=None
        if v is not None:
            VolumeAveragedVOFCoefficientsEvaluate(self.eps,
                                                  v,
                                                  phi,
                                                  porosity,
                                                  c[('u',0)],
                                                  c[('m',0)],
                                                  c[('dm',0,0)],
                                                  c[('f',0)],
                                                  c[('df',0,0)])
        else:
            import pdb
            pdb.set_trace()
        if (numpy.isnan(c[('u',0)]).any() or
            numpy.isnan(c[('m',0)]).any() or
            numpy.isnan(c[('dm',0,0)]).any() or
            numpy.isnan(c[('f',0)]).any() or
            numpy.isnan(c[('df',0,0)]).any()):
            import pdb
            pdb.set_trace()

    #
#
from proteus.ctransportCoefficients import groundwaterTransportCoefficientsEvaluate
class GroundwaterTransportCoefficients(TC_base):
    """
    groundwater advection-dispersion equation with constant coefficients but variable
    velocity
    """
    def __init__(self,nc=1,
                 omega=0.3,
                 alpha_L=1.0,
                 alpha_T=0.2,
                 d=1.3e-9):
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction  = {}
        hamiltonian = {}
        for i in range(nc):
            diffusion[i] = {i : {i:'constant'}}
            advection[i] = {i : {i:'linear'}}
            mass[i] = {i : {i:'linear'}}
            reaction[i] = {i : {i:'constant'}}
            potential[i] = {i : 'u'}
        #end i
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         ['C'])
        self.omega = omega#0.3 #porosity [-]
        self.d = d#1.3e-9 #m^2/s
        self.alpha_L = alpha_L#1.0 #m
        self.alpha_T = alpha_T#self.alpha_L/5.0 #m
        self.useC = True
    def attachModels(self,modelList):
        self.q_v = modelList[0].q[('velocity',0)]
        self.ebq_v = modelList[0].ebq[('velocity',0)]
        self.ebqe_v= modelList[0].ebqe[('velocity',0)]
        self.ebq_global_v= modelList[0].ebq_global[('velocity',0)]
    def evaluate(self,t,c):
        nSpace = c[('df',0,0)].shape[-1]
        if self.q_v.shape == c[('df',0,0)].shape:
            v = self.q_v
        elif self.ebq_v.shape == c[('df',0,0)].shape:
            v = self.ebq_v
        elif self.ebqe_v.shape == c[('df',0,0)].shape:
            v = self.ebqe_v
        elif self.ebq_global_v.shape == c[('df',0,0)].shape:
            v = self.ebq_global_v
        else:
            print(c[('df',0,0)].shape)
            print(self.ebq_v.shape)
            print(self.q_v.shape)
            print(self.ebqe_v.shape)
            print(self.ebq_global_v.shape)
            print("no v---------------------")
            raise RuntimeError
        if self.useC:
            groundwaterTransportCoefficientsEvaluate(self.omega,
                                                     self.d,
                                                     self.alpha_L,
                                                     self.alpha_T,
                                                     v,
                                                     c[('u',0)],
                                                     c[('m',0)],
                                                     c[('dm',0,0)],
                                                     c[('f',0)],
                                                     c[('df',0,0)],
                                                     c[('a',0,0)])
            if v.shape == self.q_v.shape:
                for i in range(len(c[('u',0)].flat)):
                    if c['x'].flat[3*i+0] > 100.0-2.5 and c['x'].flat[3*i+1] < 2.5 and c[('u',0)].flat[i] < -1.0e-1:
                        import pdb
                        pdb.set_trace()


from proteus.ctransportCoefficients import groundwaterBiodegradation01EvaluateFC
class GroundwaterBiodegradation01Coefficients(TC_base):
    """
    groundwater advection-dispersion equation with constant coefficients but variable
    velocity and simple 3 component biodegradation system
    """
    def __init__(self,
                 omega=0.3,
                 alpha_L=1.0,
                 alpha_T=0.2,
                 d_c=1.3e-9,
                 d_e=1.3e-9,
                 Kox_max=100.0,
                 Kox_C=0.1,
                 Kox_E=0.1,
                 Kox_X=3.5e-3,
                 Yield=0.05,
                 k_d  =0.0):
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction  = {}
        hamiltonian = {}
        nc = 3
        for i in range(3):
            diffusion[i] = {i : {i: {i:'constant'}}}
            advection[i] = {i : {i:'linear'}}
            mass[i] = {i : {i:'linear'}}
            potential[i] = {i : 'u'}
        #end i
        #mass[2] = {2 : {2:'linear'}}
        reaction = {0 : {0:'nonlinear',1:'nonlinear',2:'nonlinear'},
                    1 : {0:'nonlinear',1:'nonlinear',2:'nonlinear'},
                    2 : {0:'nonlinear',1:'nonlinear',2:'nonlinear'}}


        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         ['C','E','X'])
        self.omega = omega#0.3 #porosity [-]
        self.d_c = d_c#1.3e-9 #m^2/s
        self.d_e = d_e#1.3e-9 #m^2/s
        self.alpha_L = alpha_L#1.0 #m
        self.alpha_T = alpha_T#self.alpha_L/5.0 #m
        self.Kox_max=Kox_max
        self.Kox_C=Kox_C
        self.Kox_E=Kox_E
        self.Kox_X=Kox_X
        self.Yield=Yield
        self.k_d  =k_d

        self.useC = True
    def attachModels(self,modelList):
        self.q_v = modelList[0].q[('velocity',0)]
        self.ebq_v = modelList[0].ebq[('velocity',0)]
        self.ebqe_v= modelList[0].ebqe[('velocity',0)]
        self.ebq_global_v = modelList[0].ebq_global[('velocity',0)]
    def evaluate(self,t,c):
        nSpace = c[('df',0,0)].shape[-1]
        if self.q_v.shape == c[('df',0,0)].shape:
            v = self.q_v
        elif self.ebq_v.shape == c[('df',0,0)].shape:
            v = self.ebq_v
        elif self.ebqe_v.shape == c[('df',0,0)].shape:
            v = self.ebqe_v
        else:
            print(c[('df',0,0)].shape)
            print(self.ebq_v.shape)
            print(self.q_v.shape)
            print(self.ebqe_v.shape)
            print(self.ebq_global_v.shape)
            raise RuntimeError("no v---------------------")
        if self.useC:
            groundwaterBiodegradation01EvaluateFC(self.omega,
                                                  self.d_c,
                                                  self.d_e,
                                                  self.alpha_L,
                                                  self.alpha_T,
                                                  self.Kox_max,
                                                  self.Kox_C,
                                                  self.Kox_E,
                                                  self.Kox_X,
                                                  self.Yield,
                                                  self.k_d,
                                                  v,
                                                  c[('u',0)],
                                                  c[('u',1)],
                                                  c[('u',2)],
                                                  c[('m',0)],
                                                  c[('dm',0,0)],
                                                  c[('m',1)],
                                                  c[('dm',1,1)],
                                                  c[('m',2)],
                                                  c[('dm',2,2)],
                                                  c[('f',0)],
                                                  c[('df',0,0)],
                                                  c[('f',1)],
                                                  c[('df',1,1)],
                                                  c[('a',0,0)],
                                                  c[('a',1,1)],
                                                  c[('r',0)],
                                                  c[('dr',0,0)],
                                                  c[('dr',0,1)],
                                                  c[('dr',0,2)],
                                                  c[('r',1)],
                                                  c[('dr',1,0)],
                                                  c[('dr',1,1)],
                                                  c[('dr',1,2)],
                                                  c[('r',2)],
                                                  c[('dr',2,0)],
                                                  c[('dr',2,1)],
                                                  c[('dr',2,2)])

from proteus.ctransportCoefficients import groundwaterBryantDawsonIonExEvaluateFC
class GroundwaterBryantDawsonIonExCoefficients(TC_base):
    """
    groundwater advection-dispersion equation with constant coefficients but variable
    velocity and competitive ion exchange problem from Bryant, Dawson etal 00
    """
    def __init__(self,
                 omega=0.3,
                 alpha_L=1.0,
                 alpha_T=0.2,
                 d_m=1.3e-9,
                 d_h=1.3e-9,
                 K_m=1.0,
                 K_h=1.0,
                 K_w=1.0,
                 Z_tot=100.0):
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction  = {}
        hamiltonian = {}
        nc = 2
        for i in range(nc):
            diffusion[i] = {i : {i: {i:'constant'}}}
        #end i
        #mass[2] = {2 : {2:'linear'}}
        advection = {0 : {0:'linear'},
                     1 : {1:'nonlinear'}}
        mass     = {0 : {0:'nonlinear',1:'nonlinear'},
                    1 : {0:'nonlinear',1:'nonlinear'}}
        potential= {0 : {0: 'u'},
                    1 : {1:'nonlinear'}}
        reaction = {0 : {0:'nonlinear',1:'nonlinear'},
                    1 : {0:'nonlinear',1:'nonlinear'}}


        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         ['M','H'])
        self.omega = omega#0.3 #porosity [-]
        self.d_m = d_m#1.3e-9 #m^2/s
        self.d_h = d_h#1.3e-9 #m^2/s
        self.alpha_L = alpha_L#1.0 #m
        self.alpha_T = alpha_T#self.alpha_L/5.0 #m
        self.K_m=K_m
        self.K_h=K_h
        self.K_w=K_w
        self.Z_tot = Z_tot
        self.useC = True
    def attachModels(self,modelList):
        self.q_v = modelList[0].q[('velocity',0)]
        self.ebq_v = modelList[0].ebq[('velocity',0)]
        self.ebqe_v= modelList[0].ebqe[('velocity',0)]
    def evaluate(self,t,c):
        nSpace = c[('df',0,0)].shape[-1]
        if self.q_v.shape == c[('df',0,0)].shape:
            v = self.q_v
        elif self.ebq_v.shape == c[('df',0,0)].shape:
            v = self.ebq_v
        elif self.ebqe_v.shape == c[('df',0,0)].shape:
            v = self.ebqe_v
        else:
            print(c[('df',0,0)].shape)
            print(self.ebq_v.shape)
            print(self.q_v.shape)
            print(self.ebqe_v.shape)
            print(self.ebq_global_v.shape)
            raise RuntimeError("no v---------------------")
        if self.useC:
            groundwaterBryantDawsonIonExEvaluateFC(self.omega,
                                                   self.d_m,
                                                   self.d_h,
                                                   self.alpha_L,
                                                   self.alpha_T,
                                                   self.K_m,
                                                   self.K_h,
                                                   self.K_w,
                                                   self.Z_tot,
                                                   v,
                                                   c[('u',0)],
                                                   c[('u',1)],
                                                   c[('m',0)],
                                                   c[('dm',0,0)],
                                                   c[('dm',0,1)],
                                                   c[('m',1)],
                                                   c[('dm',1,0)],
                                                   c[('dm',1,1)],
                                                   c[('f',0)],
                                                   c[('df',0,0)],
                                                   c[('f',1)],
                                                   c[('df',1,1)],
                                                   c[('a',0,0)],
                                                   c[('a',1,1)],
                                                   c[('phi',1)],
                                                   c[('dphi',1,1)],
                                                   c[('r',0)],
                                                   c[('dr',0,0)],
                                                   c[('dr',0,1)],
                                                   c[('r',1)],
                                                   c[('dr',1,0)],
                                                   c[('dr',1,1)])

from .ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2
from .ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwind
from .ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwindAndHarm
from .ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwindAndHarm_sd
class ConservativeHeadRichardsMualemVanGenuchtenBlockHetV2withUpwind(TC_base):
    """
    version of Re where element material type id's used in evals
    """

    def __init__(self,
                 nd,
                 Ks_block,
                 vgm_n_block,
                 vgm_alpha_block,
                 thetaR_block,
                 thetaSR_block,
                 gravity,
                 density,
                 beta,
                 upwindFlag=0,
                 sd = False):
        variableNames=['pressure_head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         useSparseDiffusion = sd)
        self.gravity=gravity
        self.rho = density
        self.beta=beta
        self.Ks_block = Ks_block
        self.vgm_n_block = vgm_n_block
        self.vgm_alpha_block = vgm_alpha_block
        self.thetaR_block    = thetaR_block
        self.thetaSR_block   = thetaSR_block
        self.elementMaterialTypes = None
        self.exteriorElementBoundaryTypes  = None
        self.materialTypes_q    = None
        self.materialTypes_ebq  = None
        self.materialTypes_ebqe  = None
        #
        self.nd = nd
        self.upwindFlag = upwindFlag
        self.q_avg = {}; self.ebq_avg = {}; self.ebq_global_avg = {}; self.ebqe_avg = {}
        self.ebq_global_n = None
        self.quadraturePointToElementBoundary = {}
        self.useOrigUpwind = False
    def initializeMesh(self,mesh):
        self.elementMaterialTypes = mesh.elementMaterialTypes
        #want element boundary material types for evaluating heterogeneity
        #not boundary conditions
        self.exteriorElementBoundaryTypes = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            eN  = mesh.elementBoundaryElementsArray[ebN,0]
            self.exteriorElementBoundaryTypes[ebNE] = self.elementMaterialTypes[eN]
        #need to keep this around to build quadrature mappings
        self.mesh = mesh
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
        assert self.q_shape[1] == self.nd+1 #have to have unique mapping for q points to upwind
        #for holding element averages of coefficients
        #
        avgs = {('f',0):[self.nd],('df',0,0):[self.nd],('a',0,0):[self.nd,self.nd],('da',0,0,0):[self.nd,self.nd]}
        for key in list(avgs.keys()):
            dims = [self.q_shape[0]]; dims.extend(avgs[key])
            self.q_avg[key] = numpy.zeros(tuple(dims),'d')
        #map from quadrature points to element boundary
        if self.nd == 1:
            self.quadraturePointToElementBoundary['q'] = numpy.zeros(self.q_shape,'i')#could index by shape
            #quad points (0),(1)  --> 0.0, 1.0, reverse of node to element boundary (across) convention on element
            for eN in range(self.q_shape[0]):
                self.quadraturePointToElementBoundary['q'][eN,0] = self.mesh.elementBoundariesArray[eN,1]
                self.quadraturePointToElementBoundary['q'][eN,1] = self.mesh.elementBoundariesArray[eN,0]
        elif self.nd == 2:
            #quad points (1/2,1/2),(0,1/2),(1/2,0) --> local faces across from nodes (0),(1),(2)
            self.quadraturePointToElementBoundary['q'] = self.mesh.elementBoundariesArray
        else:
            assert False, "Need 3d face quadrature for Upwinding!"

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        self.ebq_global_shape = cebq['x'].shape[:-2]
        for eN in range(self.elementMaterialTypes.shape[0]):
            self.materialTypes_ebq[eN,:] = self.elementMaterialTypes[eN]
        #for holding element averages of coefficients
        avgs = {('f',0):[self.nd],('df',0,0):[self.nd],('a',0,0):[self.nd,self.nd],('da',0,0,0):[self.nd,self.nd]}
        for key in list(avgs.keys()):
            dims = [self.ebq_shape[0],self.ebq_shape[1]]; dims.extend(avgs[key])
            self.ebq_avg[key] = numpy.zeros(tuple(dims),'d')
            dims = [self.ebq_global_shape[0]]; dims.extend(avgs[key])
            self.ebq_global_avg[key] = numpy.zeros(tuple(dims),'d')
        #needed for upwinding
        self.ebq_global_n = cebq_global['n']
        #map from quadrature points to element boundary
        self.quadraturePointToElementBoundary['ebq'] = numpy.zeros(self.ebq_shape,'i')
        for eN in range(self.ebq_shape[0]):
            for ebN_local in range(self.ebq_shape[1]):
                self.quadraturePointToElementBoundary['ebq'][eN,ebN_local,:] = self.mesh.elementBoundariesArray[eN,ebN_local]
        self.quadraturePointToElementBoundary['ebq_global'] = numpy.zeros((cebq_global['x'].shape[0],cebq_global['x'].shape[1]),'i')
        for ebN_global in range(cebq_global['x'].shape[0]):
            self.quadraturePointToElementBoundary['ebq_global'][ebN_global,:] = ebN_global
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = numpy.zeros(cebqe[('u',0)].shape[0],'i')
        self.ebqe_shape = cebqe[('u',0)].shape
        for ebNE in range(self.exteriorElementBoundaryTypes.shape[0]):
            self.materialTypes_ebqe[ebNE] = self.exteriorElementBoundaryTypes[ebNE]
        #for holding element averages of coefficients
        avgs = {('f',0):[self.nd],('df',0,0):[self.nd],('a',0,0):[self.nd,self.nd],('da',0,0,0):[self.nd,self.nd]}
        for key in list(avgs.keys()):
            dims = [self.ebqe_shape[0]]; dims.extend(avgs[key])
            self.ebqe_avg[key] = numpy.zeros(tuple(dims),'d')
        #
        self.quadraturePointToElementBoundary['ebqe'] = numpy.zeros(self.ebqe_shape,'i')
        for ebNE in range(self.ebqe_shape[0]):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            self.quadraturePointToElementBoundary['ebqe'][ebNE,:] = ebN

    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        if self.upwindFlag == 0:
            conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2(materialTypes,
                                                                    self.rho,
                                                                    self.beta,
                                                                    self.gravity,
                                                                    self.vgm_alpha_block,
                                                                    self.vgm_n_block,
                                                                    self.thetaR_block,
                                                                    self.thetaSR_block,
                                                                    self.Ks_block,
                                                                    c[('u',0)],
                                                                    c[('m',0)],
                                                                    c[('dm',0,0)],
                                                                    c[('f',0)],
                                                                    c[('df',0,0)],
                                                                    c[('a',0,0)],
                                                                    c[('da',0,0,0)])
        elif self.upwindFlag >= 1:
            quad_avg = None; quad2bnd = None; dV = None
            computeAverages = 0;
            useUpwindApprox = 0; #second style upwinding only works for elements quad
            if c[('u',0)].shape == self.q_shape:
                quad_avg = self.q_avg
                quad2bnd = self.quadraturePointToElementBoundary['q']
                dV       = c[('dV_u',0)]
                computeAverages = 1; useUpwindApprox = 1
            elif c[('u',0)].shape == self.ebqe_shape:
                quad_avg = self.ebqe_avg
                quad2bnd = self.quadraturePointToElementBoundary['ebqe']
                dV       = c[('dS_u',0)]
            elif c[('u',0)].shape == self.ebq_shape:
                quad_avg = self.ebq_avg
                quad2bnd = self.quadraturePointToElementBoundary['ebq']
                dV       = c[('dS_u',0)]
            else:
                assert False, "no quad maps found to match c[('u',0)].shape= %s " % c[('u',0)].shape
            if self.useOrigUpwind:
                conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwind(self.upwindFlag,
                                                                                  computeAverages,
                                                                                  self.mesh.elementBoundaryElementsArray,
                                                                                  quad2bnd,
                                                                                  materialTypes,
                                                                                  self.rho,
                                                                                  self.beta,
                                                                                  self.gravity,
                                                                                  self.vgm_alpha_block,
                                                                                  self.vgm_n_block,
                                                                                  self.thetaR_block,
                                                                                  self.thetaSR_block,
                                                                                  self.Ks_block,
                                                                                  c[('u',0)],
                                                                                  c[('grad(u)',0)],
                                                                                  self.ebq_global_n,
                                                                                  dV,
                                                                                  c[('m',0)],
                                                                                  c[('dm',0,0)],
                                                                                  quad_avg[('f',0)],
                                                                                  quad_avg[('df',0,0)],
                                                                                  quad_avg[('a',0,0)],
                                                                                  quad_avg[('da',0,0,0)],
                                                                                  c[('f',0)],
                                                                                  c[('df',0,0)],
                                                                                  c[('a',0,0)],
                                                                                  c[('da',0,0,0)])
            else:
                if self.sd:
                    conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwindAndHarm_sd(useUpwindApprox,
                                                                                                computeAverages,
                                                                                                self.sdInfo[(0,0)][0],
                                                                                                self.sdInfo[(0,0)][1],
                                                                                                self.mesh.elementBoundaryElementsArray,
                                                                                                quad2bnd,
                                                                                                materialTypes,
                                                                                                self.rho,
                                                                                                self.beta,
                                                                                                self.gravity,
                                                                                                self.vgm_alpha_block,
                                                                                                self.vgm_n_block,
                                                                                                self.thetaR_block,
                                                                                                self.thetaSR_block,
                                                                                                self.Ks_block,
                                                                                                c[('u',0)],
                                                                                                c[('grad(u)',0)],
                                                                                                self.ebq_global_n,
                                                                                                dV,
                                                                                                c[('m',0)],
                                                                                                c[('dm',0,0)],
                                                                                                quad_avg[('f',0)],
                                                                                                quad_avg[('df',0,0)],
                                                                                                quad_avg[('a',0,0)],
                                                                                                quad_avg[('da',0,0,0)],
                                                                                                c[('f',0)],
                                                                                                c[('df',0,0)],
                                                                                                c[('a',0,0)],
                                                                                                c[('da',0,0,0)])
                else:
                    conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwindAndHarm(useUpwindApprox,
                                                                                             computeAverages,
                                                                                             self.mesh.elementBoundaryElementsArray,
                                                                                             quad2bnd,
                                                                                             materialTypes,
                                                                                             self.rho,
                                                                                             self.beta,
                                                                                             self.gravity,
                                                                                             self.vgm_alpha_block,
                                                                                             self.vgm_n_block,
                                                                                             self.thetaR_block,
                                                                                             self.thetaSR_block,
                                                                                             self.Ks_block,
                                                                                             c[('u',0)],
                                                                                             c[('grad(u)',0)],
                                                                                             self.ebq_global_n,
                                                                                             dV,
                                                                                             c[('m',0)],
                                                                                             c[('dm',0,0)],
                                                                                             quad_avg[('f',0)],
                                                                                             quad_avg[('df',0,0)],
                                                                                             quad_avg[('a',0,0)],
                                                                                             quad_avg[('da',0,0,0)],
                                                                                             c[('f',0)],
                                                                                             c[('df',0,0)],
                                                                                             c[('a',0,0)],
                                                                                             c[('da',0,0,0)])

from .ctransportCoefficients import diffusiveWave1DCoefficientsEvaluate
class DiffusiveWave_1D(TC_base):

    """
    A class implementing the coefficients of the diffusive wave equation in 1D.

    This implements the regularized formulation in M. Santillana's thesis (ref).
    """
    def __init__(self,alpha=5.0/3.0,gamma=0.5,epsilon=1.0e-6,bathymetryFunc=None, reactionFunc=None, analyticalSoln=None):
        """
        Construct a coefficients object given the parameters of the Manning/Chezy formula and the regularization parameter.

        Optionally provide a function for bathymetry, right hand side (source), and an analytical solution. The
        """
        self.alpha=alpha; self.gamma=gamma; self.epsilon=epsilon
        TC_base.__init__(self, nc=1,
                         variableNames=['H'],
                         mass={0:{0:'linear'}},
                         advection={},
                         diffusion={0:{0:{0:'nonlinear'}}},
                         potential={0:{0:'u'}},
                         reaction={0:{0:'constant'}},
                         hamiltonian={})
        self.bathymetryFunc=bathymetryFunc
        self.reactionFunc=reactionFunc
        self.analyticalSoln=analyticalSoln
    def initializeMesh(self,mesh):
        """
        Set the y component of the 1D mesh using the bathymetry function.
        """
        if self.bathymetryFunc:
            #mesh vertices are held in vector nodeArray which is nNodes_global x 3
            for nN in range(mesh.nNodes_global):
                x = mesh.nodeArray.flat[nN*3:(nN+1)*3]; z = self.bathymetryFunc(x)
                mesh.nodeArray.flat[nN*3+1]= z
    def initializeElementQuadrature(self,t,cq):
        """
        Set the y component of the element quadrature points using the bathymetry function.
        """
        if self.bathymetryFunc:
            #cq is elementQuadrature dictionary so x is nElements_global x nQuadraturePoints_element x 3
            nPoints = cq['x'].shape[0]*cq['x'].shape[1]
            for i in range(nPoints):
                x = cq['x'].flat[i*3:(i+1)*3]; z =  self.bathymetryFunc(x)
                cq['x'].flat[i*3+1]=z
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Set the y component of the element boundary quadrature points using the bathymetry function.
        """
        if self.bathymetryFunc:
            #cebq is quadrature dictionary for local "faces" on each element so
            #  x is nElements_global x nElementBoundaries_element x nQuadraturePoints_elementBoundary x 3
            nPoints = cebq['x'].shape[0]*cebq['x'].shape[1]*cebq['x'].shape[2]
            for i in range(nPoints):
                x = cebq['x'].flat[i*3:(i+1)*3]; z =  self.bathymetryFunc(x)
                cebq['x'].flat[i*3+1]=z
            #cebq_global is quadrature dictionary for global "faces" so
            #x is nElementBoundaries_global x nQuadraturePoints_elementBoundary x 3
            nPoints = cebq_global['x'].shape[0]*cebq_global['x'].shape[1]
            for i in range(nPoints):
                x = cebq_global['x'].flat[i*3:(i+1)*3]; z =  self.bathymetryFunc(x)
                cebq_global['x'].flat[i*3+1]=z

    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        """
        Set the y component of the exterior element boundary quadrature points using the bathymetry function.
        """
        if self.bathymetryFunc:
            #cebqe is quadrature dictionary for global exterior "faces"
            #  x is nExteriorElementBoundaries_global x nQuadraturePoints_elementBoundary x 3
            nPoints = cebqe['x'].shape[0]*cebqe['x'].shape[1]
            for i in range(nPoints):
                x = cebqe['x'].flat[i*3:(i+1)*3]; z =  self.bathymetryFunc(x)
                cebqe['x'].flat[i*3+1]=z

    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        """
        Set the y component of the generlatized interpolation points using the bathymetry function.
        """
        if self.bathymetryFunc:
            #cip is dictionary for interpolation points
            #  so x is nElements x number of interpolation points (usual dof) per element
            nPoints = cip['x'].shape[0]*cip['x'].shape[1]
            for i in range(nPoints):
                x = cip['x'].flat[i*3:(i+1)*3]; z =  self.bathymetryFunc(x)
                cip['x'].flat[i*3+1]=z

    def evaluate(self,t,c):

        """
        Evaluated the coefficients of the 1D diffusive wave model.
        """


        diffusiveWave1DCoefficientsEvaluate(self.alpha,
                                            self.gamma,
                                            self.epsilon,
                                            c['x'],
                                            c[('u',0)],
                                            c[('grad(u)',0)],
                                            c[('m',0)],
                                            c[('dm',0,0)],
                                            c[('a',0,0)],
                                            c[('da',0,0,0)])



from .ctransportCoefficients import diffusiveWave2DCoefficientsEvaluate
class DiffusiveWave_2D(TC_base):

    """
    A class implementing the coefficients of the diffusive wave equation in 2D.

    This implements the regularized formulation in M. Santillana's thesis (ref).
    """
    def __init__(self,nd=2,alpha=5.0/3.0,gamma=0.5,epsilon=1.0e-6,bathymetryFunc=None, bathymetryGradientFunc=None):
        """
        Construct a coefficients object given the parameters of the Manning/Chezy formula and the regularization parameter.

        Optionally provide a function for bathymetry, right hand side (source), and an analytical solution. The
        """
        self.alpha=alpha; self.gamma=gamma; self.epsilon=epsilon; self.nd=nd
        TC_base.__init__(self, nc=1,
                         variableNames=['H'],
                         mass={0:{0:'linear'}},
                         advection={0:{0:'constant'}},#mwf added for LDG right now
                         diffusion={0:{0:{0:'nonlinear'}}},
                         potential={0:{0:'u'}},
                         reaction={0:{0:'constant'}},
                         hamiltonian={})
        self.bathymetryFunc=bathymetryFunc
        self.bathymetryGradientFunc=bathymetryGradientFunc
        assert ((self.bathymetryFunc is None and self.bathymetryGradientFunc is None) or
                (self.bathymetryFunc is not None and self.bathymetryGradientFunc is not None))
        self.bind=2

    def initializeMesh(self,mesh):
        if self.bathymetryFunc:
            #mesh vertices are held in vector nodeArray which is nNodes_global x 3
            for nN in range(mesh.nNodes_global):
                x = mesh.nodeArray.flat[nN*3:(nN+1)*3]; b = self.bathymetryFunc(x)
                mesh.nodeArray.flat[nN*3+self.bind]=b
    def initializeElementQuadrature(self,t,cq):
        self.q_grad_b = numpy.zeros(cq[('grad(u)',0)].shape,'d')
        if self.bathymetryFunc:
            #cq is elementQuadrature dictionary so x is nElements_global x nQuadraturePoints_element x 3
            nPoints = cq['x'].shape[0]*cq['x'].shape[1]
            for i in range(nPoints):
                x = cq['x'].flat[i*3:(i+1)*3]; b = self.bathymetryFunc(x); grad_b = self.bathymetryGradientFunc(x)
                cq['x'].flat[i*3+self.bind]=b
                self.q_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.ebq_grad_b = numpy.zeros(cebq[('grad(u)',0)].shape,'d')
        if ('grad(u)',0) in cebq_global:
            self.ebq_global_grad_b = numpy.zeros(cebq_global[('grad(u)',0)].shape,'d')
        else:
            sh = (cebq_global['x'].shape[0],cebq_global['x'].shape[1],self.nd)
            self.ebq_global_grad_b = numpy.zeros(sh,'d')
        if self.bathymetryFunc:
            #cebq is quadrature dictionary for local "faces" on each element so
            #  x is nElements_global x nElementBoundaries_element x nQuadraturePoints_elementBoundary x 3
            nPoints = cebq['x'].shape[0]*cebq['x'].shape[1]*cebq['x'].shape[2]
            for i in range(nPoints):
                x = cebq['x'].flat[i*3:(i+1)*3]; b =  self.bathymetryFunc(x); grad_b = self.bathymetryGradientFunc(x)
                cebq['x'].flat[i*3+self.bind]=b
                self.ebq_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b
            #cebq_global is quadrature dictionary for global "faces" so
            #x is nElementBoundaries_global x nQuadraturePoints_elementBoundary x 3
            nPoints = cebq_global['x'].shape[0]*cebq_global['x'].shape[1]
            for i in range(nPoints):
                x = cebq_global['x'].flat[i*3:(i+1)*3]; b =  self.bathymetryFunc(x); grad_b = self.bathymetryGradientFunc(x)
                cebq_global['x'].flat[i*3+self.bind]=b
                self.ebq_global_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b

    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.ebqe_grad_b = numpy.zeros(cebqe[('grad(u)',0)].shape,'d')
        if self.bathymetryFunc:
            #cebqe is quadrature dictionary for global exterior "faces"
            #  x is nExteriorElementBoundaries_global x nQuadraturePoints_elementBoundary x 3
            nPoints = cebqe['x'].shape[0]*cebqe['x'].shape[1]
            for i in range(nPoints):
                x = cebqe['x'].flat[i*3:(i+1)*3]; b =  self.bathymetryFunc(x); grad_b = self.bathymetryGradientFunc(x)
                cebqe['x'].flat[i*3+self.bind]=b
                self.ebqe_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b

    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        self.ip_grad_b = numpy.zeros((cip['x'].shape[0],cip['x'].shape[1],self.nd),'d')
        if self.bathymetryFunc:
            #cip is dictionary for interpolation points
            #  so x is nElements x number of interpolation points (usual dof) per element
            nPoints = cip['x'].shape[0]*cip['x'].shape[1]
            for i in range(nPoints):
                x = cip['x'].flat[i*3:(i+1)*3]; b =  self.bathymetryFunc(x);  grad_b = self.bathymetryGradientFunc(x)
                cip['x'].flat[i*3+self.bind]=b
                self.ip_grad_b.flat[i*self.nd:(i+1)*self.nd] = grad_b

    def evaluate(self,t,c):
        """
        Evaluated the coefficients of the 2D diffusive wave model.
        """

        diffusiveWave2DCoefficientsEvaluate(self.nd,
                                            self.alpha,
                                            self.gamma,
                                            self.epsilon,
                                            c['x'],
                                            c[('u',0)],
                                            c[('grad(u)',0)],
                                            c[('m',0)],
                                            c[('dm',0,0)],
                                            c[('a',0,0)],
                                            c[('da',0,0,0)])

from .ctransportCoefficients import Mass_2D_Evaluate
from .ctransportCoefficients import Mass_3D_Evaluate
class DiscreteMassMatrix(TC_base):
    r"""Coefficients class for the discrete Mass Operator.
    
    This class defines the coefficients necessary to construct the
    discrete mass operator :math:`A` where

    .. math::
    
        a^{c}_{i,j} = \int_{T} \phi^{c}_{i} \phi^{c}_{j} dT

    for all :math:`T \in \Omega`, :math:`c=1,...,nc` and 
    :math:`\phi^{c}_{i}, i=1,...,k` is a basis for component :math:`c`.
    """
    def __init__(self,rho=1.0,nd=2):
        self.rho = rho
        self.nd = nd
        mass = {}
        advection= {}
        diffusion = {}
        potential = {}
        reaction = {}
        hamiltonian = {}
        if nd==2:
            variableNames = ['p','u','v']
            mass = {0:{0:'linear'},
                    1:{1:'linear'},
                    2:{2:'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors={})
            self.vectorComponents = [1,2]
        elif nd==3:
            variableNames = ['p','u','v','w']
            mass = {0:{0:'linear'},
                    1:{1:'linear'},
                    2:{2:'linear'},
                    3:{3:'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames)
            self.vectorComponents=[1,2,3]
    def evaluate(self,t,c):
        if self.nd==2:
            Mass_2D_Evaluate(self.rho,
                             c[('u',0)],
                             c[('u',1)],
                             c[('u',2)],
                             c[('m',0)],
                             c[('m',1)],
                             c[('m',2)],
                             c[('dm',0,0)],
                             c[('dm',1,1)],
                             c[('dm',2,2)])
        elif self.nd==3:
            Mass_3D_Evaluate(self.rho,
                             c[('u',0)],
                             c[('u',1)],
                             c[('u',2)],
                             c[('u',3)],
                             c[('m',0)],
                             c[('m',1)],
                             c[('m',2)],
                             c[('m',3)],
                             c[('dm',0,0)],
                             c[('dm',1,1)],
                             c[('dm',2,2)],
                             c[('dm',3,3)])

from .ctransportCoefficients import TwoPhaseAdvection_2D_Evaluate
class DiscreteTwoPhaseAdvectionOperator(TC_base):
    r""" A coefficient class to build the discrete advection operator.

    This class defines the coefficients necessary to construct the
    discrete advection operator :math:`N` where
    
    .. math::

       n^{c}_{i,j} = \int_{T} (\mathbf{w}_{h} \phi_{j}) \cdot
       \nabla \phi_{i} d T

    for all :math:`T \in \Omega`, :math:`c = 0,...nc-1` and
    :math:`phi^{c}_{i}`, `i=0,\cdot k-1` is a basis component for
    :math:`c`.  Also note, :math:`\mathbf{w}_{h}` is a vector field 
    (often the solution from the last non-linear iteration).
    
    Parameters
    ----------
    nd : int
        The dimension of the physical domain
    u : numpy array
        An array of arrays with the advective field evaluated at the 
        quadrature points.
    """
#    from ctransportCoefficients import TwoPhaseAdvection_3D_Evaluate
    def __init__(self,
                 u,
                 nd=2,
                 rho_0 = 1.0,
                 nu_0 = 1.0,
                 rho_1 = 1.0,
                 nu_1 = 1.0,
                 eps = 0.00000001,
                 LS_model = None,
                 phase_function = None):
        self.nd=nd
        self.advection_field_u = numpy.copy(u[0])
        self.advection_field_v = numpy.copy(u[1])
        if self.nd==3:
            self.advection_field_w = numpy.copy(u[2])
        self.eps = eps
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.LS_model = LS_model
        self.phase_function = phase_function
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {}
        hamiltonian = {}
        if self.nd==2:
            variableNames=['p','u','v']
            advection = {0:{0:'linear',
                            1:'linear',
                            2:'linear'},
                         1:{1:'nonlinear',
                            2:'nonlinear'},
                         2:{1:'nonlinear',
                            2:'nonlinear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors={})
            self.vectorComponents = [1,2]
        # if self.nd==3:
        #     variableNames=['p','u','v','w']
        #     advection = {0:{0:'linear',
        #                     1:'linear',
        #                     2:'linear',
        #                     3:'linear'},
        #                  1:{1:'nonlinear',
        #                     2:'nonlinear',
        #                     3:'nonlinear'},
        #                  2:{1:'nonlinear',
        #                     2:'nonlinear',
        #                     3:'nonlinear'},
        #                   3:{1:'nonlinear',
        #                      2:'nonlinear',
        #                      3:'nonlinear'}}
        #     TC_base.__init__(self,
        #                      3,
        #                      mass,
        #                      advection,
        #                      diffusion,
        #                      potential,
        #                      reaction,
        #                      hamiltonian,
        #                      variableNames)
        #     self.vectorComponents = [1,2,3]
    def attachModels(self,modelList):
        if self.LS_model != None:
            self.q_phi = modelList[self.LS_model].q[('u',0)]
            self.ebqe_phi = modelList[self.LS_model].ebqe[('u',0)]
            self.ebq_phi = None

    def initializeQuadratureWithPhaseFunction(self,c):
        self.q_phi = c[('u',0)].copy()
        for i, element in enumerate(c['x']):
            for j,pt in enumerate(c['x'][i]):
                self.q_phi[i][j] = self.phase_function(pt)

    def evaluate(self,t,c):
        if self.phase_function != None:
            self.initializeQuadratureWithPhaseFunction(c)

        if c[('u',0)].shape == self.q_phi.shape:
            phi = self.q_phi
        else:
            phi = self.ebq_phi
            
        if self.nd==2:
            self.TwoPhaseAdvection_2D_Evaluate(self.eps,
                                               self.rho_0,
                                               self.nu_0,
                                               self.rho_1,
                                               self.nu_1,
                                               phi,
                                               c[('u',0)],
                                               self.advection_field_u,
                                               self.advection_field_v,
                                               c[('f',0)],
                                               c[('f',1)],
                                               c[('f',2)],
                                               c[('df',0,0)],
                                               c[('df',0,1)],
                                               c[('df',0,2)],
                                               c[('df',1,1)],
                                               c[('df',1,2)],
                                               c[('df',2,1)],
                                               c[('df',2,2)])
        elif self.nd==3:
            self.TwoPhaseAdvection_3D_Evaluate(self.eps,
                                               self.rho_0,
                                               self.nu_0,
                                               self.rho_1,
                                               self.nu_1,
                                               phi,
                                               c[('u',0)],
                                               self.advection_field_u,
                                               self.advection_field_v,
                                               self.advection_field_w,
                                               c[('f',0)],
                                               c[('f',1)],
                                               c[('f',2)],
                                               c[('f',3)],
                                               c[('df',0,0)],
                                               c[('df',0,1)],
                                               c[('df',0,2)],
                                               c[('df',0,3)],
                                               c[('df',1,1)],
                                               c[('df',1,2)],
                                               c[('df',1,3)],
                                               c[('df',2,1)],
                                               c[('df',2,2)],
                                               c[('df',2,3)],
                                               c[('df',3,1)],
                                               c[('df',3,2)],
                                               c[('df',3,3)])