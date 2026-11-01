from proteus import Comm, Profiling
import numpy as np
import numpy.testing as npt
import unittest
import pytest

comm = Comm.init()
Profiling.procID = comm.rank()

Profiling.logEvent("Testing SedClosure")
class GlobalVariables(object):
    def __init__(self):
        from proteus.mprans.SedClosure import HsuSedStress
        self.C4e = 1.
        self.C3e = 1.2
        self.eR = 0.8
        self.aDarcy = 1.
        self.bForch = 1.
        self.grain = 0.1
        self.packFraction = 0.2
        self.packMargin = 0.01
        self.sigmaC = 1.1
        self.maxFraction = 0.635
        self.frFraction = 0.57
        self.fContact = 0.02
        self.mContact = 2.
        self.nContact = 5.
        self.angFriction = np.pi/6.
        self.vos_limiter = 0.6
        self.mu_fr_limiter  = 0.1
        self.sedSt = HsuSedStress( self.aDarcy, self.bForch, self.grain, self.packFraction,  self.packMargin,self.maxFraction, self.frFraction, self.sigmaC, self.C3e, self.C4e, self.eR ,self.fContact, self.mContact, self.nContact, self.angFriction, self.vos_limiter, self.mu_fr_limiter)
        self.sedSt_nl = HsuSedStress( self.aDarcy, self.bForch, self.grain, self.packFraction,  self.packMargin,self.maxFraction, self.frFraction, self.sigmaC, self.C3e, self.C4e, self.eR ,self.fContact, self.mContact, self.nContact, self.angFriction, self.vos_limiter, 1e100)
    

class TestHsu(unittest.TestCase):
    def testGranularDrag1(self):
        gl = GlobalVariables()
        import random
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        rhoFluid = 1. + random.random()
        umag_temp = (uf - us)*(uf - us)
        umag = np.sqrt(sum(umag_temp))
        # Testing for sedF >> gl.packFraction
        nu = 1.
        sedF = 0.5
        drag = (gl.aDarcy * nu* sedF /((1.-sedF)*gl.grain**2) +  gl.bForch * umag / gl.grain)*rhoFluid
        # For sedF > pacFraction - > drag = a * nu* sedF /((1-sedF)*gl.grain^2) + beta * umag / gl.grain
        drag2 = gl.sedSt.betaCoeff(sedF, rhoFluid, uf, us, nu)
        if(drag2 != 0):
            drag /=drag2
            drag2/=drag2
        npt.assert_almost_equal(drag,drag2)
#    @pytest.mark.skip(reason="in development")
    def testGranularDrag2(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1. + random.random()
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        umag_temp = (uf - us)*(uf - us)
        umag = np.sqrt(sum(umag_temp))
        # Testing for sedF << gl.packFraction and Rep = (1. - sed) * umag*nu/gl.grain = 0.9 * 5. * 0.1 / 1. = 0.45
        nu = 1.
        sedF = 0.1
        Rep = (1.- sedF)*umag*gl.grain / nu    
        drag = rhoFluid * ( 24. * (1.+0.15*Rep**(0.687))/Rep) * 0.75 * umag * (1. -sedF)**(-1.65) / gl.grain # Chen and Hsu 2014
        drag2 = gl.sedSt.betaCoeff(sedF, rhoFluid, uf, us, nu)
        if(drag2 != 0):
            drag /=drag2
            drag2/=drag2
        #if you use npt.assert_almost_equal you get more info on failure...
        #self.assertTrue(round(drag,10) == round(drag2,10))
        npt.assert_almost_equal(drag,drag2)
#    @pytest.mark.skip(reason="in development")
    def testGranularDrag3(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1. + random.random()
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        umag_temp = (uf - us)*(uf - us)
        umag = np.sqrt(sum(umag_temp))
        # Testing for sedF << gl.packFraction and Rep > 1000
        sedF = 0.1
        nu = 1e-4
        Rep = (1.- sedF) * umag * gl.grain / nu    
        drag =  rhoFluid * ( 0.44 * 0.75 * umag * (1. -sedF)**(-1.65) )/ gl.grain # Chen and Hsu 2014
        drag2 = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)
        if(drag2 != 0):
            drag /=drag2
            drag2/=drag2
        #self.assertTrue(round(drag,10) == round(drag2,10))
        npt.assert_almost_equal(drag, drag2)
#    @pytest.mark.skip(reason="in development")
    def testGranularDrag4(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1. + random.random()
        # Constant params
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        umag_temp = (uf - us)*(uf - us)
        umag = np.sqrt(sum(umag_temp))
        # Testing for sedF =  gl.packFraction +0.5 packmargin and Rep > 1000
        sedF = 0.205
        nu = 1e-4
        Rep = (1.- sedF) * umag * gl.grain / nu
        draga = gl.aDarcy * nu* sedF /((1.-sedF)*gl.grain**2) +  gl.bForch * umag / gl.grain
        dragb =  (0.44*0.75*umag*(1.-sedF)**(-1.65))/gl.grain # Chen and Hsu 2014
        w = 0.5 + (sedF-gl.packFraction)/(2.*gl.packMargin)
        drag =rhoFluid* (w*draga + (1.-w) * dragb)
        drag2 = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)
        if(drag2 != 0):
            drag /=drag2
            drag2/=drag2
        #self.assertTrue(round(drag,10) == round(drag2,10))
        npt.assert_almost_equal(drag, drag2)
    def testgs0(self):
        gl=GlobalVariables()
        f = 8
        sedF = 0.2
        gs0 = gl.sedSt.gs0(sedF)
        self.assertTrue(gs0 == 0.5*(2-sedF)/(1-sedF)**3)
        sedF = 0.55
        gs0 = gl.sedSt.gs0(sedF)
        npt.assert_almost_equal(gs0, 0.5*(2-0.49)/(1-0.49)**3 * (0.64-0.49)/(0.64-sedF), decimal=f)
        sedF = 0.65
        gs0 = gl.sedSt.gs0(sedF)
        npt.assert_almost_equal(gs0, 0.5*(2-0.49)/(1-0.49)**3 * (0.64-0.49)/(0.64-0.635), decimal=f)



        
    def testTkeSed(self):
        g = np.array([0.,9.81])
        gl=GlobalVariables()
        import random
        rhoFluid = 10. + random.random()
        f = 9
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        gradC=np.array([0.1,0.1])
        rhoS = 2000
        nu = 1
        nuT = 1
        sedF = 0.3
# Setting 0 t_c
        theta_n = 0.25 + 0.25*random.random() + 1e-30
        kappa_n = 0.1 + 0.1*random.random() + 1e-30
        epsilon_n = 0.1  + 0.1 * random.random() + 1e-30


        beta = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)
        t_p =rhoS/ beta
        l_c = np.sqrt(np.pi)*gl.grain / (24.*sedF * gl.sedSt.gs0(sedF))
        t_cl = min(l_c/np.sqrt(theta_n) , 0.165*kappa_n/epsilon_n)
        aa = 1/( 1. + t_p/t_cl)
        gc = g[:]*gradC[:]
        gc = sum(gc)
        ss = (rhoS/rhoFluid-1)
        es = 2.*beta * (1-aa)*sedF/((1-sedF)*rhoFluid)*kappa_n+ss*gc*nuT/gl.sigmaC/(1.-sedF)
        kappa_sed = gl.sedSt.kappa_sed1(sedF,rhoFluid,rhoS,uf,us,gradC,nu,theta_n,kappa_n,epsilon_n,nuT,g)
        npt.assert_almost_equal(kappa_sed, es, decimal=f)

        
    def test_dTkeSed_dk(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1000. + random.random()
        f = 8
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        gradC=np.array([0.1,0.1])
        rhoS = 2000
        nu = 1e-2
        nuT = 1e-2
        sedF = 0.3
# Setting 0 t_c
        theta_n = 0.25 + 0.25*random.random() + 1e-30
        kappa_n = 0.1 + 0.1*random.random() + 1e-30
        kappa_np1 = 0.1  + 0.1 * random.random() + 1e-30
        epsilon_n = 0.1  + 0.1 * random.random() + 1e-30


        beta = gl.sedSt.betaCoeff(sedF,rhoFluid, uf, us, nu)      
        t_p =rhoS/beta
        l_c = np.sqrt(np.pi)*gl.grain / (24.*sedF * gl.sedSt.gs0(sedF))
        t_cl = min(l_c/np.sqrt(theta_n),0.165*kappa_n/epsilon_n)
        aa = 1/( 1. + t_p/t_cl)
      
        es1 = 2.*beta * rhoS*(1-aa)*sedF/((1-sedF)*rhoFluid)


        kappa_sed = gl.sedSt.dkappa_sed1_dk(sedF,rhoFluid,rhoS,uf,us,gradC,nu,theta_n,kappa_n,epsilon_n,nuT)

        npt.assert_almost_equal(kappa_sed, es1, decimal=f)

    def test_dEpsSed_dE(self):
        gl=GlobalVariables()
        g = np.array([0.,9.81])
        import random
        rhoFluid = 1000. + random.random()
        f = 8
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        gradC=np.array([0.1,0.1])
        rhoS = 2000
        nu = 1e-4
        nuT = 1e-2
        sedF = 0.3
# Setting 0 t_c
        theta_n = random.random() + 1e-30
        kappa_n = random.random() + 1e-30
        epsilon_n = random.random() + 1e-30


        beta = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)      
        t_p =rhoS/beta
        l_c = np.sqrt(np.pi)*gl.grain / (24.*sedF * gl.sedSt.gs0(sedF))
        t_cl = min(l_c/np.sqrt(theta_n),0.165*kappa_n/epsilon_n)
        aa = 1/( 1. + t_p/t_cl)

       
        es1 = 2.*beta *(1-aa)*sedF*kappa_n/((1-sedF)*rhoFluid)

        UgradC = np.dot(g,gradC)

        es2  =  nuT * UgradC*(rhoS/rhoFluid-1.) / ((1-sedF)*gl.sigmaC)

        eps_sed = gl.sedSt.deps_sed_deps(sedF,rhoFluid,rhoS,uf,us,gradC,nu,theta_n,kappa_n,epsilon_n,nuT,g)
        valid = gl.C3e*es1/kappa_n+gl.C4e*es2/kappa_n
        npt.assert_almost_equal(eps_sed, valid, decimal=f)

    def testPsc(self):
        gl=GlobalVariables()
        import random
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        rhoS = 2000
        psc = gl.sedSt.psc(sedF,rhoS,theta)
        npt.assert_almost_equal(psc, rhoS*sedF*(1. + 2*(1.+gl.eR)*sedF*gl.sedSt.gs0(sedF))*theta, decimal=f)

    def testPscTerm(self):
        gl=GlobalVariables()
        import random
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        dudx = random.random() + 1e-30
        dvdy = random.random() + 1e-30
        dwdz = random.random() + 1e-30
        divU = dudx + dvdy + dwdz
        rhoS = 2000
        test = gl.sedSt.psc_term(sedF,rhoS,theta,dudx,dvdy,dwdz)
        npt.assert_almost_equal(test, -2.*gl.sedSt.psc(sedF,rhoS,theta)*divU/(3.*rhoS*sedF), decimal=f)


    def testdpsc_term_dtheta(self):
        gl=GlobalVariables()
        import random
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        dudx = random.random() + 1e-30
        dvdy = random.random() + 1e-30
        dwdz = random.random() + 1e-30
        divU = dudx + dvdy + dwdz
        rhoS = 2000
        test = gl.sedSt.dpsc_term_dtheta(sedF,rhoS,dudx,dvdy,dwdz)
        npt.assert_almost_equal(test, -2.*gl.sedSt.psc(sedF,rhoS,theta)*divU/(3.*rhoS*sedF)/theta, decimal=f)
    def testMu_sc(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        rhoS = 2000
        test = gl.sedSt.mu_sc(sedF,rhoS,theta)
        g0 = gl.sedSt.gs0(sedF)
        valid = rhoS * gl.grain * sqrt(theta) * ( 0.8 *sedF**2 * g0 * (1. + gl.eR) / sqrt(np.pi) + (1./15) *sedF**2 * g0 * (1. + gl.eR) * sqrt(np.pi) +  (1./6.) *sedF * sqrt(np.pi) +  (5./48.) * sqrt(np.pi)/((1+gl.eR)*g0))        
        npt.assert_almost_equal(test, valid, decimal=f)

    def testL_sc(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        rhoS = 2000
        test = gl.sedSt.l_sc(sedF,rhoS,theta)
        g0 = gl.sedSt.gs0(sedF)
        valid = (4./3.)*sedF**2 * rhoS * gl.grain *g0*(1+gl.eR)* (sqrt(theta)/sqrt(np.pi) )
        
        npt.assert_almost_equal(test, valid, decimal=f)

    def test_tsc_term(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        dudx = random.random() + 1e-30
        dudy = random.random() + 1e-30
        dudz = random.random() + 1e-30
        dvdx = random.random() + 1e-30
        dvdy = random.random() + 1e-30
        dvdz = random.random() + 1e-30
        dwdx = random.random() + 1e-30
        dwdy = random.random() + 1e-30
        dwdz = random.random() + 1e-30
        rhoS = 2000
        test = gl.sedSt.tausc_term_theta(sedF,rhoS,theta,dudx,dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
        divU = dudx + dvdy + dwdz
        mu = gl.sedSt.mu_sc(sedF, rhoS, theta)
        l = gl.sedSt.l_sc(sedF, rhoS, theta)
        s_tensor =  np.array([ [ 2.*dudx ,  dudy+dvdx,  dudz+dwdx],
                                [ dudy+dvdx, 2.*dvdy,    dvdz+dwdy],
                                [ dudz+dwdx, dvdz+dwdy,  2.* dwdz]])
        t_tensor = mu * s_tensor  + (l - (2./3.) * mu) * divU * np.array([ [ 1 , 0 , 0],
                                                   [ 0,  1,  0],
                                                   [ 0,  0,  1] ])
        product = s_tensor * t_tensor
        valid = 0.
        for i in product:
            for j in i:
                valid+=j/(3.*rhoS*sedF)
                
        npt.assert_almost_equal(test, valid, decimal=f)


    def testgamma_s(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        theta_np1 = random.random() + 1e-30
        rhoS = 2000
        g0 = gl.sedSt.gs0(sedF)
        dudx = random.random() + 1e-30
        dvdy = random.random() + 1e-30
        dwdz = random.random() + 1e-30
        test = gl.sedSt.gamma_s(sedF,rhoS,theta , theta_np1, dudx, dvdy, dwdz)
        divU = dudx + dvdy + dwdz
        valid =  - 3. *  (1. - gl.eR**2) * sedF**2 * rhoS * g0 * theta_np1 * ( (sqrt(theta)/sqrt(np.pi)) * (4./gl.grain) - divU) * (2./(3.*rhoS*sedF))
        npt.assert_almost_equal(test, valid, decimal=f)

    def testdgamma_s_dtheta(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        rhoS = 2000
        g0 = gl.sedSt.gs0(sedF)
        dudx = random.random() + 1e-30
        dvdy = random.random() + 1e-30
        dwdz = random.random() + 1e-30

        test = gl.sedSt.dgamma_s_dtheta(sedF,rhoS,theta , dudx, dvdy, dwdz)
        divU = dudx + dvdy + dwdz
        
        valid =  - 3. *  (1. - gl.eR**2) * sedF**2 * rhoS * g0  * ( (sqrt(theta)/sqrt(np.pi)) * (4./gl.grain) - divU) * (2./(3.*rhoS*sedF))
        npt.assert_almost_equal(test, valid, decimal=f)

    def testJint1(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1. + random.random()
        f = 8
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        gradC=np.array([0.1,0.1])
        rhoS = 2000
        nu = 1e-4
        nuT = 1e-2
        sedF = 0.3
# Setting 0 t_c
        theta_n = random.random() + 1e-30
        kappa_n = random.random() + 1e-30
        epsilon_n = random.random() + 1e-30


        beta = gl.sedSt.betaCoeff(sedF, rhoFluid, uf, us, nu)      
        t_p =rhoS/beta
        l_c = np.sqrt(np.pi)*gl.grain / (24.*sedF * gl.sedSt.gs0(sedF))
        t_cl = min(l_c/np.sqrt(theta_n),0.165*kappa_n/epsilon_n)
        aa = 1/( 1. + t_p/t_cl)

       
        es1 = 2.*beta * rhoS*(1-aa)*sedF*kappa_n/((1-sedF)*rhoFluid)

        UgradC = np.dot((uf - us),gradC)

        es2  = beta * rhoFluid * nuT * UgradC / ((1-sedF)*rhoFluid)

        test = gl.sedSt.jint1(sedF,rhoFluid, rhoS, uf,us, kappa_n,epsilon_n, theta_n, nu)

        npt.assert_almost_equal(test, 2*aa*beta*sedF*kappa_n*(2./(3.*sedF*rhoS)), decimal=f)


    def testJint2(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1. + random.random()
        f = 8
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        gradC=np.array([0.1,0.1])
        rhoS = 2000
        nu = 1e-4
        nuT = 1e-2
        sedF = 0.3
# Setting 0 t_c
        theta_n = random.random() + 1e-30

        beta = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)      

       
        test = gl.sedSt.jint2(sedF,rhoFluid, rhoS,uf,us, theta_n, nu)

        npt.assert_almost_equal(test, -3*beta*sedF*theta_n*(2./(3.*sedF*rhoS)), decimal=f)

    def testJint2dTheta(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1. + random.random()
        f = 8
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        gradC=np.array([0.1,0.1])
        rhoS = 2000
        nu = 1e-4
        nuT = 1e-2
        sedF = 0.3
# Setting 0 t_c
        theta_n = random.random() + 1e-30

        beta = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)      

       
        test = gl.sedSt.djint2_dtheta(sedF,rhoFluid, rhoS,uf,us, nu)

        npt.assert_almost_equal(test, -3*beta*sedF*(2./(3.*sedF*rhoS)), decimal=f)


    def testK_diff(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        rhoS = 2000
        test = gl.sedSt.k_diff(sedF,rhoS,theta)
        g0 = gl.sedSt.gs0(sedF)
        valid = rhoS * gl.grain * sqrt(theta) * ( 2. *sedF**2 * g0 * (1. + gl.eR) / sqrt(np.pi) + (9./16) *sedF**2 * g0 * (1. + gl.eR) * sqrt(np.pi) +  (15./16.) *sedF * sqrt(np.pi) +  (25./64.) * sqrt(np.pi)/((1+gl.eR)*g0))
        
        npt.assert_almost_equal(test, valid, decimal=f)


    def test_p_fr_limiter(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        # No stress
        sedF = 0.1
        p_friction = 0. 
        p_ftest = gl.sedSt.p_friction(sedF)
        self.assertAlmostEqual(p_friction,p_ftest)
        
        # No limiter
        sedF = 0.58
        p_friction = gl.fContact*(sedF-gl.frFraction)**gl.mContact/(gl.maxFraction-sedF)**gl.nContact
        p_ftest = gl.sedSt.p_friction(sedF)
        self.assertAlmostEqual(p_friction,p_ftest)
        
        
        # Exactly at the limiter
        sedF = 0.6
        p_friction = gl.fContact*(sedF-gl.frFraction)**gl.mContact/(gl.maxFraction-sedF)**gl.nContact
        p_ftest = gl.sedSt.p_friction(sedF)
        self.assertAlmostEqual(p_friction,p_ftest)
        # Over the limiter
        sedF=0.62
        sedFm = min(gl.vos_limiter, sedF)
        p_friction2 = gl.fContact*(sedFm-gl.frFraction)**gl.mContact/(gl.maxFraction-sedFm)**gl.nContact
        p_ftest2 = gl.sedSt.p_friction(sedF)
        self.assertAlmostEqual(p_friction2,p_ftest2)
        self.assertAlmostEqual(p_friction2,p_ftest)
        
    def test_gradp_fr_limiter(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        # No stress
        # No stress
        sedF = 0.1
        gradp = 0. 
        gradp_test = gl.sedSt.gradp_friction(sedF)
        self.assertAlmostEqual(gradp,gradp_test)
        
        # No limiter
        sedF = 0.58
        p_friction = gl.fContact*(sedF-gl.frFraction)**gl.mContact/(gl.maxFraction-sedF)**gl.nContact
        gradp = p_friction*(gl.mContact/(sedF-gl.frFraction)+gl.nContact/(gl.maxFraction-sedF))
        gradp_test = gl.sedSt.gradp_friction(sedF)
        self.assertAlmostEqual(gradp,gradp_test)

        
        
        # Exactly at the limiter
        sedF = 0.6
        p_friction = gl.fContact*(sedF-gl.frFraction)**gl.mContact/(gl.maxFraction-sedF)**gl.nContact
        gradp = p_friction*(gl.mContact/(sedF-gl.frFraction)+gl.nContact/(gl.maxFraction-sedF))
        gradp_test = gl.sedSt.gradp_friction(sedF)
        self.assertAlmostEqual(gradp,gradp_test)
        # Over the limiter
        sedF = 0.62
        sedFm = min(gl.vos_limiter,sedF)
        p_friction = gl.fContact*(sedFm-gl.frFraction)**gl.mContact/(gl.maxFraction-sedFm)**gl.nContact
        gradp = p_friction*(gl.mContact/(sedFm-gl.frFraction)+gl.nContact/(gl.maxFraction-sedFm))
        gradp_test = gl.sedSt.gradp_friction(sedF)
        self.assertAlmostEqual(gradp,gradp_test)
        
        
        
    def test_mu_fr2D(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        dudx = random.random() + 1e-30
        dudy = random.random() + 1e-30
        dudz = random.random() + 1e-30
        dvdx = random.random() + 1e-30
        dvdy = random.random() + 1e-30
        dvdz = random.random() + 1e-30
        dwdx = random.random() + 1e-30
        dwdy = random.random() + 1e-30
        dwdz = random.random() + 1e-30
        rhoS = 2000
        divU = dudx + dvdy + dwdz
        
        s_tensor = 0.5* np.array([ [ 2.*dudx  -divU ,  dudy+dvdx,               dudz+dwdx],
                                [ dudy+dvdx,                   2.*dvdy-divU,    dvdz+dwdy],
                                [ dudz+dwdx,                   dvdz+dwdy,               2.*dwdz-divU]])

        product = s_tensor * s_tensor
        magn = sum(product)
        magn = sum(magn)

        test = gl.sedSt.mu_fr(sedF,dudx,dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
        valid = sqrt(2)*gl.sedSt.p_friction(sedF)*np.sin(gl.angFriction)/2./sqrt(magn)
        self.assertAlmostEqual(test,valid)

        
        sedF = 0.58
        valid = sqrt(2)*gl.sedSt.p_friction(sedF)*np.sin(gl.angFriction)/2./sqrt(magn)
        test = gl.sedSt_nl.mu_fr(sedF,dudx,dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
        self.assertAlmostEqual(test,valid)


        sedF = 0.6
        valid = sqrt(2)*gl.sedSt.p_friction(sedF)*np.sin(gl.angFriction)/2./sqrt(magn)
        test = gl.sedSt_nl.mu_fr(sedF,dudx,dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
        self.assertAlmostEqual(test,valid)

        sedF = 0.62
        valid = sqrt(2)*gl.sedSt.p_friction(sedF)*np.sin(gl.angFriction)/2./sqrt(magn)
        test = gl.sedSt_nl.mu_fr(sedF,dudx,dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
        self.assertAlmostEqual(test,valid)

        sedF = 0.62
        valid = sqrt(2)*gl.sedSt.p_friction(sedF)*np.sin(gl.angFriction)/2./sqrt(magn)
        test = gl.sedSt.mu_fr(sedF,dudx,dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
        self.assertAlmostEqual(test,min(valid,0.1))
      


    def test_p_s(self):
        gl=GlobalVariables()
        import random
        sqrt = np.sqrt
        f = 10        
        # Setting 0 t_c
        sedF = 0.3
        theta = random.random() + 1e-30
        dudx = random.random() + 1e-30
        dudy = random.random() + 1e-30
        dudz = random.random() + 1e-30
        dvdx = random.random() + 1e-30
        dvdy = random.random() + 1e-30
        dvdz = random.random() + 1e-30
        dwdx = random.random() + 1e-30
        dwdy = random.random() + 1e-30
        dwdz = random.random() + 1e-30
        rhoS = 2000
        test = gl.sedSt.tausc_term_theta(sedF,rhoS,theta,dudx,dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
        divU = dudx + dvdy + dwdz
        mu = gl.sedSt.mu_sc(sedF, rhoS, theta)
        muf = gl.sedSt.mu_fr(sedF,dudx,dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)

        l = gl.sedSt.l_sc(sedF, rhoS, theta)

        test = gl.sedSt.p_s( sedF, rhoS, theta, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
        valid =gl.sedSt.p_friction(sedF) + gl.sedSt.psc(sedF, rhoS, theta) + (2./3.*(mu+muf) - l)*divU 


        npt.assert_almost_equal(test, valid, decimal=f)        

        
        





    def testMintFluid(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1. + random.random()
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        ufp1 = np.array([5.,4.],"d")
        usp1 = np.array([1.,1.],"d") 
        gradc = np.array([0.1,0.1],"d") 
        sedF = 0.205
        nu = 1e-4
        nuT = 1e-2
        beta = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)      
        mint = gl.sedSt.mIntFluid(sedF, rhoFluid, uf, us, ufp1, nu, nuT, gradc)
        assert np.allclose(mint,-sedF*beta*(ufp1)/(1.-sedF)/rhoFluid,atol=1e-10)
    def testMintSolid(self):
        gl=GlobalVariables()
        import random
        rhoFluid = 1. + random.random()
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        ufp1 = np.array([5.,4.],"d")
        usp1 = np.array([1.,1.],"d") 
        gradc = np.array([0.1,0.1],"d") 
        sedF = 0.205
        nu = 1e-4
        nuT = 1e-2
        beta = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)      
        mint = gl.sedSt.mIntSolid(sedF, rhoFluid, uf, us, usp1 , nu, nuT, gradc)
        assert np.allclose(mint, -sedF*beta*( - usp1) / (1.-sedF)/rhoFluid,atol=1.0e-10, rtol=1.0e-10)
    def testMintgradC(self):
        import random
        gl=GlobalVariables()
        rhoFluid = 1. + random.random()
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        ufp1 = np.array([5.,4.],"d")
        usp1 = np.array([1.,1.],"d") 
        gradc = np.array([0.1,0.1],"d") 
        sedF = 0.205
        nu = 1e-4
        nuT = 1e-2
        beta = gl.sedSt.betaCoeff(sedF,rhoFluid, uf, us, nu)      
        mint = gl.sedSt.mIntgradC(sedF,rhoFluid, uf, us , nu, nuT, gradc)
        assert np.allclose(mint,- sedF * beta * gradc * nuT / gl.sigmaC / rhoFluid / (1.-sedF), 1.0e-10)

    def testdMintdUf(self):
        import random
        rhoFluid = 1. + random.random()
        gl=GlobalVariables()
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        gradc = np.array([0.1,0.1],"d") 
        sedF = 0.205
        nu = 1e-4
        nuT = 1e-2
        beta = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)      
        mint = gl.sedSt.dmInt_duFluid(sedF, rhoFluid, uf, us , nu)
        npt.assert_almost_equal(mint, - sedF*beta/(1.-sedF)/rhoFluid, decimal=10)
    def testdMintdUs(self):
        import random
        gl=GlobalVariables()
        rhoFluid = 1. + random.random()
        gl=GlobalVariables()
        uf = np.array([5.,4.],"d")
        us = np.array([1.,1.],"d") 
        gradc = np.array([0.1,0.1],"d") 
        sedF = 0.205
        nu = 1e-4
        nuT = 1e-2
        beta = gl.sedSt.betaCoeff(sedF, rhoFluid,uf, us, nu)      
        mint = gl.sedSt.dmInt_duSolid(sedF, rhoFluid, uf, us , nu)
        npt.assert_almost_equal(mint, sedF*beta/(1.-sedF)/rhoFluid, decimal=10)



if __name__ == '__main__':
    unittest.main(verbosity=2)
