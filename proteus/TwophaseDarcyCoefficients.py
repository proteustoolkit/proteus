"""
An extension of the TransportCoefficients module for two-phase flow in porous media

.. inheritance-diagram:: proteus.TwophaseDarcyCoefficients
   :parts: 1
"""
from math import *
from .TransportCoefficients import TC_base
import numpy
from .Profiling import logEvent

#base class for setting up fluid and material properties
class TwophaseDarcyFlow_base(TC_base):
    def __init__(self,
                 g=9.8,
                 rhon=1.0,
                 rhow=0.0,
                 mun    = 1.0,
                 muw    = 1.0,
                 Ksw=1.0,
                 psk_model='VGM',
                 vg_alpha = 5.0,
                 vg_m  = 0.75,
                 bc_pd  = 1.0/5.47,
                 bc_lambda = 0.5,
                 omega  = 1.0,
                 Sw_max = 1.0,
                 Sw_min = 0.0,
                 density_w_model = 'Constant',
                 density_n_model = 'Constant'):
        #set psk model and parameters
        self.psk_model = psk_model
        self.psk_types={'simp':0,
                        'VGM':1,
                        'VGB':2,
                        'BCM':3,
                        'BCB':4}
        assert(self.psk_model in list(self.psk_types.keys()))
        self.vg_alpha=vg_alpha
        self.vg_m = vg_m
        self.bc_pd = bc_pd
        self.bc_lambda=bc_lambda
        #normalize gravity
        self.g = numpy.array(g,dtype='d')
        gMag= sqrt(numpy.dot(self.g,self.g))
        if gMag <= 0.0:
            gMag = 1.0
        self.g /= gMag
        #set media properties
        self.hasMaterialTypes=False
        self.setParams=None
        self.Ksw=Ksw
        self.omega=omega
        self.Sw_min = Sw_min
        self.Sw_max = Sw_max
        #set fluid properies
        self.b = rhon/rhow         #normalize density
        self.rhon=rhon/rhon
        self.rhow=rhow/rhow
        self.muw=muw/muw
        self.mun=mun/muw
        #setup rwork arrays for homogeneous media, make heterogeneity later
        if self.psk_model == 'simp':
            self.len_rwork_psk=2
            self.rwork_psk = numpy.array([self.Sw_min,self.Sw_max],dtype='d')
        elif self.psk_model in ['VGM','VGB']:
            self.len_rwork_psk=4
            self.rwork_psk = numpy.array([self.Sw_min,self.Sw_max,self.vg_alpha,self.vg_m],dtype='d')
        elif self.psk_model in ['BCM','BCB']:
            self.len_rwork_psk=4
            self.rwork_psk = numpy.array([self.Sw_min,self.Sw_max,bc_pd,bc_lambda],dtype='d')
        #density eos options
        self.density_w_model = density_w_model
        self.density_n_model = density_n_model
        self.density_types={'Constant':0,
                            'Exponential':1,
                            'IdealGas':2}
        assert self.density_w_model in self.density_types
        assert self.density_n_model in self.density_types
        #default is Constant density, put in more general init later
        self.rwork_density_w=numpy.array([self.rhow],dtype='d')
        self.rwork_density_n=numpy.array([self.rhon],dtype='d')
        #self.sd = sd
    def setMaterialTypes(self,
                         Ksw_types=[1.0],
                         omega_types  = [0.4],
                         Sw_max_types = [1.0],
                         Sw_min_types = [0.0],
                         bc_lambda_types = None,
                         bc_pd_types = None,
                         vg_alpha_types = None,
                         vg_m_types = None):
        self.nTypesAvailable=len(Ksw_types)
        self.hasMaterialTypes=True
        self.Ksw_types  = Ksw_types
        self.omega_types= omega_types
        self.setParams=None
        if self.psk_model == 'simp':
            self.rwork_psk = numpy.zeros((self.nTypesAvailable,2),'d')
            for Sw_min,Sw_max,i in zip(Sw_min_types,Sw_max_types,list(range(self.nTypesAvailable))):
                self.rwork_psk[i,0] = Sw_min
                self.rwork_psk[i,1] = Sw_max
        elif self.psk_model in ['VGM','VGB']:
            assert(vg_alpha_types is not None and vg_m_types  is not None)
            self.rwork_psk = numpy.zeros((self.nTypesAvailable,4),'d')
            for Sw_min,Sw_max,vg_alpha,vg_m,i in zip(Sw_min_types,Sw_max_types,vg_alpha_types,vg_m_types,list(range(self.nTypesAvailable))):
                self.rwork_psk[i,0] = Sw_min
                self.rwork_psk[i,1] = Sw_max
                self.rwork_psk[i,2] = vg_alpha
                self.rwork_psk[i,3] = vg_m
        elif self.psk_model in ['BCM','BCB']:
            assert(bc_lambda_types is not None and bc_lambda_types  is not None)
            self.rwork_psk = numpy.zeros((self.nTypesAvailable,4),'d')
            for Sw_min,Sw_max,bc_pd,bc_lambda,i in zip(Sw_min_types,Sw_max_types,bc_pd_types,bc_lambda_types,list(range(self.nTypesAvailable))):
                self.rwork_psk[i,0] = Sw_min
                self.rwork_psk[i,1] = Sw_max
                self.rwork_psk[i,2] = bc_pd
                self.rwork_psk[i,3] = bc_lambda
    def setMaterialFunction(self,setParams):
        self.hasMaterialTypes=False
        self.setParams=setParams
    def initializeMesh(self,mesh):
        if self.hasMaterialTypes:
            self.elementMaterialTypes = mesh.elementMaterialTypes
            #want element boundary material types for evaluating heterogeneity
            #not boundary conditions
            self.exteriorElementBoundaryTypes = numpy.zeros((mesh.nExteriorElementBoundaries_global),'i')
            for ebNE in range(mesh.nExteriorElementBoundaries_global):
                ebN = mesh.exteriorElementBoundariesArray[ebNE]
                eN  = mesh.elementBoundaryElementsArray[ebN,0]
                self.exteriorElementBoundaryTypes[ebNE] = self.elementMaterialTypes[eN]
    def initializeElementQuadrature(self,t,cq):
        #mwf not sure if this is ok
        cq['psi_n'] = numpy.zeros(cq[('u',0)].shape,'d')
        cq[('dpsi_n',0)] = numpy.zeros(cq[('u',0)].shape,'d')
        cq[('dpsi_n',1)] = numpy.zeros(cq[('u',0)].shape,'d')

        if self.hasMaterialTypes:
            self.materialTypes_q = self.elementMaterialTypes
            self.q_shape = cq[('u',0)].shape
        elif self.setParams is not None:
            self.rwork_psk_q = numpy.zeros(cq[('u',0)].shape+(self.len_rwork_psk,),'d')
            self.Ks_q = numpy.zeros(cq[('u',0)].shape,'d')
            self.omega_q = numpy.zeros(cq[('u',0)].shape,'d')
            self.setParamsFunc(cq['x'],
                               self.rwork_psk_q,
                               self.Ks_q,
                               self.omega_q)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        #mwf not sure if this is ok
        if ('u',0) in cebq:
            cebq['psi_n'] = numpy.zeros(cebq[('u',0)].shape,'d')
            cebq[('dpsi_n',0)] = numpy.zeros(cebq[('u',0)].shape,'d')
            cebq[('dpsi_n',1)] = numpy.zeros(cebq[('u',0)].shape,'d')
        if ('u',0) in cebq_global:
            cebq_global['psi_n'] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            cebq_global[('dpsi_n',0)] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            cebq_global[('dpsi_n',1)] = numpy.zeros(cebq_global[('u',0)].shape,'d')
        if self.hasMaterialTypes:
            self.materialTypes_ebq = numpy.zeros(cebq[('u',0)].shape[0:2],'i')
            self.ebq_shape = cebq[('u',0)].shape
            for eN in range(self.elementMaterialTypes.shape[0]):
                self.materialTypes_ebq[eN,:] = self.elementMaterialTypes[eN]
        elif self.setParams is not None:
            self.rwork_psk_ebq = numpy.zeros(cebq[('u',0)].shape+(self.len_rwork_psk,),'d')
            self.Ks_ebq = numpy.zeros(cebq[('u',0)].shape,'d')
            self.omega_ebq = numpy.zeros(cebq[('u',0)].shape,'d')
            self.setParamsFunc(cebq['x'],
                               self.rwork_psk_ebq,
                               self.Ks_ebq,
                               self.omega_ebq)
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        cebqe['psi_n'] = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe[('dpsi_n',0)] = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe[('dpsi_n',1)] = numpy.zeros(cebqe[('u',0)].shape,'d')
        if self.hasMaterialTypes:
            self.materialTypes_ebqe = numpy.zeros(cebqe[('u',0)].shape[0],'i')
            self.ebqe_shape = cebqe[('u',0)].shape
            for ebNE in range(self.exteriorElementBoundaryTypes.shape[0]):
                self.materialTypes_ebqe[ebNE] = self.exteriorElementBoundaryTypes[ebNE]
        elif self.setParams is not None:
            self.rwork_psk_ebqe = numpy.zeros(cebqe[('u',0)].shape+(self.len_rwork_psk,),'d')
            self.Ks_ebqe = numpy.zeros(cebqe[('u',0)].shape,'d')
            self.omega_ebqe = numpy.zeros(cebqe[('u',0)].shape,'d')
            self.setParamsFunc(cebqe['x'],
                               self.rwork_psk_ebqe,
                               self.Ks_ebqe,
                               self.omega_ebqe)
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        cip['psi_n'] = numpy.zeros(cip[('u',0)].shape,'d')
        cip[('dpsi_n',0)] = numpy.zeros(cip[('u',0)].shape,'d')
        cip[('dpsi_n',1)] = numpy.zeros(cip[('u',0)].shape,'d')
        if self.hasMaterialTypes:
            self.ip_shape = cip[('u',0)].shape
            #should be element based so can use elementMaterialTypes
            self.materialTypes_ip = self.elementMaterialTypes
        elif self.setParams is not None:
            self.rwork_psk_ip = numpy.zeros(cip[('u',0)].shape+(self.len_rwork_psk,),'d')
            self.Ks_ip = numpy.zeros(cip[('u',0)].shape,'d')
            self.omega_ip = numpy.zeros(cip[('u',0)].shape,'d')
            self.setParamsFunc(cip['x'],
                               self.rwork_psk_ip,
                               self.Ks_ip,
                               self.omega_ip)

#primitive fully coupled formulation
class TwophaseDarcy_fc(TwophaseDarcyFlow_base):
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_fc_sd_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_fc_sd_het_matType = staticmethod(twophaseDarcy_fc_sd_het_matType)
    def __init__(self,
                 g=9.8,
                 rhon=1.0,
                 rhow=0.0,
                 mun    = 1.0,
                 muw    = 1.0,
                 Ksw=1.0,
                 psk_model='VGM',
                 vg_alpha = 5.0,
                 vg_m  = 0.75,
                 bc_pd  = 1.0/5.47,
                 bc_lambda = 0.5,
                 omega  = 1.0,
                 Sw_max = 1.0,
                 Sw_min = 0.0,
                 #mwf come up with a more succinct way of handling density variation
                 density_w_parameters = None, #{'model',...}
                 density_n_parameters = None,
                 diagonalHet = False,
                 sparseDiffusionTensors={},
                 sd = True):
        self.nc=2
        variableNames=['s_w','psi_w']
        #just assume mass is a function of psi w now?
        mass = {0:{0:'linear',1:'nonlinear'},
                1:{0:'nonlinear',1:'nonlinear'}}
        advection = {0:{0:'nonlinear'},
                     1:{0:'nonlinear'}}
        hamiltonian={}
        potential = {0:{1:'nonlinear',
                        0:'nonlinear'},
                     1:{0:'nonlinear',
                        1:'nonlinear'}}
        diffusion = {0:{0:{0:'nonlinear',
                           1:'nonlinear'}},
                     1:{1:{0:'nonlinear',
                           1:'nonlinear'}}}
        reaction = {0:{0:'linear'},
                    1:{1:'linear'}}
        #now account for density parameterization
        self.density_w_parameters = density_w_parameters
        self.density_n_parameters = density_n_parameters
        density_w_model = 'Constant'
        density_n_model = 'Constant'
        if self.density_w_parameters is not None:
            density_w_model = self.density_w_parameters['model']
        if self.density_n_parameters is not None:
            density_n_model = self.density_n_parameters['model']
        #for handling sparse diffusion options
        assert not diagonalHet
        self.nd = len(g) #need to check
        assert len(sparseDiffusionTensors) > 0
        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = sd)
        TwophaseDarcyFlow_base.__init__(self,
                                        g,
                                        rhon,
                                        rhow,
                                        mun,
                                        muw,
                                        Ksw,
                                        psk_model,
                                        vg_alpha,
                                        vg_m,
                                        bc_pd,
                                        bc_lambda,
                                        omega,
                                        Sw_max,
                                        Sw_min,
                                        density_w_model,
                                        density_n_model)
        #set up density relationships
        for params,rwork in zip([self.density_w_parameters,self.density_n_parameters],
                                ['rwork_density_w','rwork_density_n']):
            if params is not None:
                if params['model'] == 'Exponential':
                    setattr(self,rwork,numpy.array([params['rho_0']/params['rho_0'],#normalize by phase density
                                                    params['psi_0'],
                                                    params['beta']],dtype='d'))
                elif params['model'] == 'IdealGas':
                    setattr(self,rwork,numpy.array([params['T'],
                                                    params['W']/params['rho_0'],#normalize by phase density
                                                    params['R'],
                                                    params['headToPressure'],
                                                    params['rho_0']/params['rho_0'],#normalize by phase density
                                                    params['psi_0']],dtype='d'))
                else:
                    assert False, 'TwophaseDarcy_fc density params= %s not found ' % params
        #

    def evaluate(self,t,c):
        assert self.hasMaterialTypes
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
        elif c[('u',0)].shape == self.ip_shape:
            materialTypes = self.materialTypes_ip
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        self.twophaseDarcy_fc_sd_het_matType(self.psk_types[self.psk_model],
                                             self.density_types[self.density_w_model],
                                             self.density_types[self.density_n_model],
                                             materialTypes,
                                             self.muw,
                                             self.mun,
                                             self.omega_types,
                                             self.Ksw_types,
                                             self.b,
                                             self.rwork_psk,
                                             self.rwork_density_w,
                                             self.rwork_density_n,
                                             self.g,
                                             c['x'],
                                             c[('u',0)],
                                             c[('u',1)],
                                             c[('m',0)],
                                             c[('dm',0,0)],
                                             c[('dm',0,1)],
                                             c[('m',1)],
                                             c[('dm',1,0)],
                                             c[('dm',1,1)],
                                             c['psi_n'],
                                             c[('dpsi_n',0)],
                                             c[('dpsi_n',1)],
                                             c[('phi',0)],
                                             c[('dphi',0,1)],
                                             c[('phi',1)],
                                             c[('dphi',1,1)],
                                             c[('dphi',1,0)],
                                             c[('a',0,0)],
                                             c[('da',0,0,0)],
                                             c[('da',0,0,1)],
                                             c[('a',1,1)],
                                             c[('da',1,1,0)],
                                             c[('da',1,1,1)])

#split fractional flow formulation--pressure equation
class TwophaseDarcy_split_pressure(TwophaseDarcyFlow_base):
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_incompressible_split_sd_pressure_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_incompressible_split_sd_pressure_het_matType = staticmethod(twophaseDarcy_incompressible_split_sd_pressure_het_matType)
    def __init__(self,
                 g=9.8,
                 rhon=1.0,
                 rhow=0.0,
                 mun    = 1.0,
                 muw    = 1.0,
                 Ksw=1.0,
                 psk_model='VGM',
                 vg_alpha = 5.0,
                 vg_m  = 0.75,
                 bc_pd  = 1.0/5.47,
                 bc_lambda = 0.5,
                 omega  = 1.0,
                 Sw_max = 1.0,
                 Sw_min = 0.0,
                 swConstant=0.5,
                 capillaryDiffusionScaling=1.0,
                 nModel = 1,
                 diagonalHet = False,
                 sparseDiffusionTensors={},
                 sd = True):
        self.swConstant=swConstant
        self.nc=1
        variableNames=['psi_w']
        #these are only nonlinear for compressible flow
        mass      = {0:{0:'nonlinear'}}
        advection = {0:{0:'nonlinear'}}
        hamiltonian={}
        diffusion = {0:{0: {0:'nonlinear'}}}
        potential = {0:{0: 'u'}}         # if phi is nonlinear
        reaction  = {0:{0:'linear'}}
        #for handling sparse diffusion options
        assert not diagonalHet
        self.nd = len(g) #need to check
        assert len(sparseDiffusionTensors) > 0
        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = sd)
        TwophaseDarcyFlow_base.__init__(self,
                                       g,
                                       rhon,
                                       rhow,
                                       mun,
                                       muw,
                                       Ksw,
                                       psk_model,
                                       vg_alpha,
                                       vg_m,
                                       bc_pd,
                                       bc_lambda,
                                       omega,
                                       Sw_max,
                                       Sw_min)
        self.nModel=nModel
        #capillary term combined with gravity in 'f'
        #so can't turn off easily
        self.capillaryDiffusionScaling = capillaryDiffusionScaling
    def attachModels(self,modelList):
        if self.nModel is None:
            print('Warning Twophase_split_pressure nModel is None returning in attachModels')
            return
        self.q_s_w   = modelList[self.nModel].q[('u',0)]
        self.ebqe_s_w = modelList[self.nModel].ebqe[('u',0)]
        if ('u',0) in modelList[self.nModel].ebq:
            self.ebq_s_w = modelList[self.nModel].ebq[('u',0)]
        assert ('u',0) in modelList[self.nModel].phi_ip
        assert self.ip_s_w.shape ==  modelList[self.nModel].phi_ip[('u',0)].shape
        self.ip_s_w = modelList[self.nModel].phi_ip[('u',0)]
        self.ip_grad_psic = None#modelList[self.nModel].phi_ip[('grad(phi)',0)]
        self.q_grad_psic   = modelList[self.nModel].q[('grad(phi)',0)]
        self.ebqe_grad_psic = modelList[self.nModel].ebqe[('grad(phi)',0)]
        if ('grad(phi)',0) in modelList[self.nModel].ebq:
            self.ebq_grad_psic = modelList[self.nModel].ebq[('grad(phi)',0)]
        self.q_psic   = modelList[self.nModel].q[('phi',0)]
        self.ebqe_psic= modelList[self.nModel].ebqe[('phi',0)]
        if ('phi',0) in modelList[self.nModel].ebq:
            self.ebq_psic = modelList[self.nModel].ebq[('phi',0)]
        assert ('phi',0) in modelList[self.nModel].phi_ip
        assert self.ip_psic.shape ==  modelList[self.nModel].phi_ip[('phi',0)].shape
        self.ip_psic = modelList[self.nModel].phi_ip[('phi',0)]

    def initializeElementQuadrature(self,t,cq):
        TwophaseDarcyFlow_base.initializeElementQuadrature(self,t,cq)
        #set up dummy values in case we're not running the other model
        self.q_s_w   = numpy.zeros(cq[('u',0)].shape,'d')
        self.q_s_w[:] = self.swConstant
        for i in range(len(self.q_s_w.flat)//2,len(self.q_s_w.flat)):
            self.q_s_w.flat[i] = 1.0e-4
        self.q_grad_psic   = numpy.zeros(cq[('f',0)].shape,'d')
        self.q_psic        = numpy.zeros(cq[('u',0)].shape,'d')
        cq['psi_n'] = numpy.zeros(cq[('u',0)].shape,'d')
        cq[('dpsi_n',0)] = numpy.ones(cq[('u',0)].shape,'d')

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseDarcyFlow_base.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        #set up dummy values in case we're not running the other model
        self.ebq_s_w = numpy.zeros(cebq[('u',0)].shape,'d')
        self.ebq_s_w[:]=self.swConstant
        for i in range(len(self.ebq_s_w.flat)//2,len(self.ebq_s_w.flat)):
            self.ebq_s_w.flat[i] = 1.0e-4
        self.ebq_grad_psic = numpy.zeros(cebq[('f',0)].shape,'d')
        self.ebq_psic = numpy.zeros(cebq[('u',0)].shape,'d')
        if ('u',0) in cebq:
            cebq['psi_n'] = numpy.zeros(cebq[('u',0)].shape,'d')
            cebq[('dpsi_n',0)] = numpy.ones(cebq[('u',0)].shape,'d')
        if ('u',0) in cebq_global:
            cebq_global['psi_n'] = numpy.zeros(cebq_global[('u',0)].shape,'d')
            cebq_global[('dpsi_n',0)] = numpy.ones(cebq_global[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseDarcyFlow_base.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        #set up dummy values in case we're not running the other model
        self.ebqe_s_w = numpy.zeros(cebqe[('u',0)].shape,'d')
        self.ebqe_s_w[:]=self.swConstant
        for i in range(len(self.ebqe_s_w.flat)//2,len(self.ebqe_s_w.flat)):
            self.ebqe_s_w.flat[i] = 1.0e-4
        self.ebqe_grad_psic = numpy.zeros(cebqe[('f',0)].shape,'d')
        self.ebqe_psic = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe['psi_n'] = numpy.zeros(cebqe[('u',0)].shape,'d')
        cebqe[('dpsi_n',0)] = numpy.ones(cebqe[('u',0)].shape,'d')
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        TwophaseDarcyFlow_base.initializeGeneralizedInterpolationPointQuadrature(self,t,cip)
        #set up dummy values in case we're not running the other model
        self.ip_s_w = numpy.zeros(cip[('u',0)].shape,'d')
        self.ip_s_w[:]=self.swConstant
        for i in range(len(self.ip_s_w.flat)//2,len(self.ip_s_w.flat)):
            self.ip_s_w.flat[i] = 1.0e-4
        self.ip_grad_psic = numpy.zeros(cip[('f',0)].shape,'d')
        self.ip_psic = numpy.zeros(cip[('u',0)].shape,'d')
        cip['psi_n'] = numpy.zeros(cip[('u',0)].shape,'d')
        cip[('dpsi_n',0)] = numpy.ones(cip[('u',0)].shape,'d')
    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_s_w.shape:
            s_w = self.q_s_w
            grad_psic = self.q_grad_psic
            c['psi_n']= numpy.copy(self.q_psic)
            c['psi_n'] += c[('u',0)]
        elif c[('u',0)].shape == self.ebqe_s_w.shape:
            s_w = self.ebqe_s_w
            grad_psic = self.ebqe_grad_psic
            c['psi_n']= numpy.copy(self.ebqe_psic)
            c['psi_n'] += c[('u',0)]
        elif c[('u',0)].shape == self.ip_s_w.shape:
            c['psi_n']= numpy.copy(self.ip_psic)
            c['psi_n'] += c[('u',0)]
            s_w = self.ip_s_w
            grad_psic = self.ip_grad_psic
        else:
            assert c[('u',0)].shape == self.ebq_s_w.shape
            s_w = self.ebq_s_w
            grad_psic = self.ebq_grad_psic
            c['psi_n']= numpy.copy(self.ebq_psic)
            c['psi_n'] += c[('u',0)]
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
        elif c[('u',0)].shape == self.ip_shape:
            materialTypes = self.materialTypes_ip
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        self.twophaseDarcy_incompressible_split_sd_pressure_het_matType(self.psk_types[self.psk_model],
                                                      materialTypes,
                                                      self.muw,
                                                      self.mun,
                                                      self.omega_types,
                                                      self.Ksw_types,
                                                      self.b,
                                                      self.capillaryDiffusionScaling,
                                                      self.rwork_psk,
                                                      self.rwork_density_w,
                                                      self.rwork_density_n,
                                                      self.g,
                                                      s_w,
                                                      grad_psic,
                                                      c[('f',0)],
                                                      c[('a',0,0)])

#split fractional flow formulation--saturation equation
class TwophaseDarcy_split_saturation(TwophaseDarcyFlow_base):
    from proteus.cTwophaseDarcyCoefficients import twophaseDarcy_incompressible_split_sd_saturation_het_matType
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    twophaseDarcy_incompressible_split_sd_saturation_het_matType = staticmethod(twophaseDarcy_incompressible_split_sd_saturation_het_matType)
    def __init__(self,
                 g=[9.8],
                 rhon=1.0,
                 rhow=1.0,
                 mun    = 1.0,
                 muw    = 1.0,
                 Ksw=1.0,
                 psk_model='VGM',
                 vg_alpha = 5.0,
                 vg_m  = 0.75,
                 bc_pd  = 1.0/5.47,
                 bc_lambda = 0.5,
                 omega  = 1.0,
                 Sw_max = 1.0,
                 Sw_min = 0.0,
                 qScalarConstant=1.0,
                 capillaryDiffusionScaling=1.0,
                 nModel = 0,
                 diagonalHet = False,
                 sparseDiffusionTensors={},
                 sd = True):
        self.qScalarConstant=qScalarConstant
        self.nc=1
        variableNames=['s_w']
        mass      = {0:{0:'nonlinear'}}
        advection = {0:{0:'nonlinear'}}
        hamiltonian={}
        diffusion = {0:{0:{0:'nonlinear'}}}
        potential = {0:{0: 'nonlinear'}}
        reaction  = {0:{0:'linear'}}
        #for handling sparse diffusion options
        assert not diagonalHet
        self.nd = len(g) #need to check
        assert len(sparseDiffusionTensors) > 0
        TC_base.__init__(self,
                         self.nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion= sd)
        TwophaseDarcyFlow_base.__init__(self,
                                       g,
                                       rhon,
                                       rhow,
                                       mun,
                                       muw,
                                       Ksw,
                                       psk_model,
                                       vg_alpha,
                                       vg_m,
                                       bc_pd,
                                       bc_lambda,
                                       omega,
                                       Sw_max,
                                       Sw_min)
        self.nModel=nModel
        #could just turn off diffusion to test hyperbolic problem but
        #capillary diffusion combined with gravity in pressure 'f' so use
        #scaling factor in evals instead
        self.capillaryDiffusionScaling = capillaryDiffusionScaling
    def attachModels(self,modelList):
        if self.nModel is None:
            print('Warning Twophase_split_saturation nModel is None returning in attachModels')
            return
        self.flowModel = modelList[self.nModel]
        self.q_q_t    = modelList[self.nModel].q[('velocity',0)]
        self.ebqe_q_t  = modelList[self.nModel].ebqe[('velocity',0)]
        if ('velocity',0) in modelList[self.nModel].ebq:
            self.ebq_q_t  = modelList[self.nModel].ebq[('velocity',0)]
        #do we really need other model values for q_t in potential calculation?
        assert self.ip_psiw.shape == modelList[self.nModel].phi_ip[('u',0)].shape
        self.ip_psiw = modelList[self.nModel].phi_ip[('u',0)]
        self.q_psiw    = modelList[self.nModel].q[('u',0)]
        self.ebqe_psiw = modelList[self.nModel].ebqe[('u',0)]
        if ('u',0) in modelList[self.nModel].ebq:
            self.ebq_psiw = modelList[self.nModel].ebq[('u',0)]
    def initializeElementQuadrature(self,t,cq):
        TwophaseDarcyFlow_base.initializeElementQuadrature(self,t,cq)
        #set up dummy values in case we're not running the other model
        self.q_q_t   = numpy.zeros(cq[('f',0)].shape,'d')
        self.q_q_t[:] = self.qScalarConstant
        self.q_psiw   = numpy.ones(cq[('u',0)].shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        TwophaseDarcyFlow_base.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        #set up dummy values in case we're not running the other model
        self.ebq_q_t = numpy.zeros(cebq[('f',0)].shape,'d')
        self.ebq_q_t[:] = self.qScalarConstant
        self.ebq_psiw = numpy.ones(cebq[('u',0)].shape,'d')
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        TwophaseDarcyFlow_base.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        #set up dummy values in case we're not running the other model
        self.ebqe_q_t = numpy.zeros(cebqe[('f',0)].shape,'d')
        self.ebqe_q_t[:] = self.qScalarConstant
        self.ebqe_psiw = numpy.ones(cebqe[('u',0)].shape,'d')
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        TwophaseDarcyFlow_base.initializeGeneralizedInterpolationPointQuadrature(self,t,cip)
        #set up dummy values in case we're not running the other model
        self.ip_q_t = numpy.zeros(cip[('f',0)].shape,'d')
        self.ip_q_t[:] = self.qScalarConstant
        self.ip_psiw = numpy.ones(cip[('u',0)].shape,'d')
    def evaluate(self,t,c):
        if c[('f',0)].shape == self.q_q_t.shape:
            q_t = self.q_q_t
            psiw = self.q_psiw
        elif c[('f',0)].shape == self.ebqe_q_t.shape:
            q_t = self.ebqe_q_t
            psiw = self.ebqe_psiw
        elif c[('f',0)].shape == self.ip_q_t.shape:
            q_t = self.ip_q_t
            psiw = self.ip_psiw
        else:
            assert c[('f',0)].shape == self.ebq_q_t.shape
            q_t = self.ebq_q_t
            psiw = self.ebq_psiw
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
        elif c[('u',0)].shape == self.ip_shape:
            materialTypes = self.materialTypes_ip
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        self.twophaseDarcy_incompressible_split_sd_saturation_het_matType(self.psk_types[self.psk_model],
                                                           materialTypes,
                                                           self.muw,
                                                           self.mun,
                                                           self.omega_types,
                                                           self.Ksw_types,
                                                           self.b,self.capillaryDiffusionScaling,
                                                           self.rwork_psk,
                                                           self.rwork_density_w,
                                                           self.rwork_density_n,
                                                           self.g,
                                                           q_t,
                                                           c[('u',0)],
                                                           c[('m',0)],
                                                           c[('dm',0,0)],
                                                           c[('phi',0)],
                                                           c[('dphi',0,0)],
                                                           c[('f',0)],
                                                           c[('df',0,0)],
                                                           c[('a',0,0)],
                                                           c[('da',0,0,0)])