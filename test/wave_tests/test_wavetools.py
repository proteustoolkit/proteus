from proteus import Comm, Profiling
import numpy as np
import numpy.testing as npt
import unittest
import random
from math import cos,sin,cosh,sinh,pi,tanh,log
import sys,os
import logging
import pytest

comm = Comm.init()
Profiling.procID = comm.rank()
def getpath():
    path = os.path.dirname(os.path.abspath(__file__))
    return path

def remove_files(filenames):
    ''' delete files in filenames list '''
    for f in filenames:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                print ("Error: %s - %s" %(e.filename,e.strerror))
        else:
            pass

Profiling.logEvent("Testing WaveTools")
class TestAuxFunctions(unittest.TestCase):
    def testFastCos(self):
        from proteus.WaveTools import fastcos_test
        RMS = 0.
        MaxErr = 0.
        for ii in range(1,1001):
            phase = 2*pi/float(ii)
            RMS += (cos(phase) - fastcos_test(phase))**2
            MaxErr = max(MaxErr, abs(cos(phase) - fastcos_test(phase)))
        RMS /=1000.
        RMS = np.sqrt(RMS)
        self.assertTrue(RMS<1e-04)
        self.assertTrue(MaxErr<1e-3)

    def testFastCosh(self):
        from proteus.WaveTools import fastcosh_test
        RMS = 0.
        maxErr = 0.
        ll = 5.
        k = 2*pi/ll
        d = 5.
        mwl = 0.
        for ii in range(0,1001):
            Z  = -float(ii)/1000.*d
            err = (cosh(k*Z)-fastcosh_test(k,Z))/cosh(k*Z)
            RMS += err**2
            maxErr = max(maxErr, abs(err))
        RMS /=1000.
        RMS = np.sqrt(RMS)
        self.assertTrue(RMS<3e-02)
        self.assertTrue(maxErr<4e-2)

    def testFastSinh(self):
        from proteus.WaveTools import fastsinh_test
        RMS = 0.
        maxErr = 0.
        ll = 5.
        k = 2*pi/ll
        d = 5.
        mwl = 0.
        for ii in range(0,1001):
            Z  = -float(ii)/1000.*d
            err = (sinh(k*Z)-fastsinh_test(k,Z))/(cosh(k*Z))
            RMS += err**2
            maxErr = max(maxErr, abs(err))
        RMS /=1000.
        RMS = np.sqrt(RMS)
        self.assertTrue(RMS<3e-2)
        self.assertTrue(maxErr<8e-2)

    def testFastProfileCosh(self):
        from proteus.WaveTools import coshkzd_test
        RMS = 0.
        maxErr = 0.
        ll = 5.
        k = 2*pi/ll
        d = 5.
        mwl = 0.
        for ii in range(0,1001):
            Z  = -float(ii)/1000.*d
            err = (cosh(k*(Z+d))/sinh(k*d))
            fcos = coshkzd_test(k,Z,d)
            err = (err - fcos)#/sinh(k*d)
            RMS += err**2
            maxErr = max(maxErr, abs(err))
        RMS /=1000.
        RMS = np.sqrt(RMS)
        self.assertTrue(RMS<3e-02)
        self.assertTrue(maxErr<4.5e-2)

    def testFastProfileSinh(self):
        from proteus.WaveTools import sinhkzd_test
        depth = 1.
        RMS = 0.
        maxErr = 0.
        ll = 5.
        k = 2*pi/ll
        d = 5.
        mwl = 0.
        for ii in range(0,1001):
            Z  = -float(ii)/1000.*d
            err = (sinh(k*(Z+d))/sinh(k*d))
            err = err- sinhkzd_test(k,Z,d)
            RMS += err**2
            maxErr = max(maxErr, abs(err))
        RMS /=1000.
        RMS = np.sqrt(RMS)
        self.assertTrue(RMS<3e-02)
        self.assertTrue(maxErr<4.5e-2)


    def testVDir(self):
        from proteus.WaveTools import setVertDir
        self.assertTrue(np.array_equal(setVertDir(np.array([0,-9.81,0])), np.array([0,1,0])))

    def testDirVector(self):
        from proteus.WaveTools import setDirVector
        self.assertTrue(all(setDirVector(np.array([2.,2.,1.]))== np.array([2.,2.,1])/3.))

    def testDirCheck(self):
        from proteus.WaveTools import dirCheck
        dirCheck(np.array([1.,2.,3.]),np.array([7.,4.,-5.]) )# Just loading the function with two vertical vectors
        with self.assertRaises(SystemExit) as cm:
            dirCheck(np.array([9,9,9]),np.array([4,5,6]))
        self.assertEqual(cm.exception.code, 1)

    def testReduceToIntervals(self):
        from proteus.WaveTools import reduceToIntervals
        fi = np.linspace(0, 100, 101)
        df = 1
        fr = reduceToIntervals(fi,df)
        fi[:-1]+=0.5
        fi_te = np.zeros(len(fi)+1,)
        fi_te[1:] = fi[:]
        self.assertTrue((fr- fi_te == 0).all())

    def testIntegrateRectangles(self): # Testing the integration fynction for y = 2*x at [0,1]. The area should be 1

        from proteus.WaveTools import reduceToIntervals,returnRectangles
        x = np.linspace(0,1,101)
        dx = 0.01
        xim = reduceToIntervals(x,dx)
        y = 2*xim
        A = sum(returnRectangles(y,xim))
        self.assertTrue(round(A,10) == 1.0)

    def testIntegrateRectangles3D(self): # Testing the integration fynction for y = 2*x at [0,1]. The area should be 1
        from proteus.WaveTools import reduceToIntervals,returnRectangles3D
        x = np.linspace(0,1,101)
        dx = 0.01
        z = np.linspace(0,1,201)
        dz = 0.005
        xim = reduceToIntervals(x,dx)
        zim = reduceToIntervals(z,dz)
        y1 = np.zeros((len(xim),len(zim)),)

        for j in range(len(zim)):
            for i in range(len(xim)):
                y1[i,j] = 2.*xim[i]*zim[j]

        A = sum(sum(returnRectangles3D(y1,xim,zim)))
        # Integrate function z*(2*x) over x[0,1], z[0,1] result == 0.5
        self.assertTrue(round(A,10)== 0.5)

    def testNormInt(self): # Testing the integration fynction for y = 2*x at [0,1]. The area should be 1
        from proteus.WaveTools import normIntegral, reduceToIntervals, returnRectangles
        #pickin


        x = np.linspace(0,1,101)
        dx = 0.01
        xim = reduceToIntervals(x,dx)
        y = 5*xim
        A  = normIntegral(y,xim)
        A = sum(returnRectangles(A,xim))
        self.assertTrue(round(A,10)== 1)



    def testEtaMode(self):
        from proteus.WaveTools import eta_mode
        x = 10.
        y =15.
        z = 2.
        t =50.
        kDir = [0.05,0.02,0.0]
        omega = 0.5
        phi = 3.14/5.
        amplitude =0.2
        eta = amplitude*cos(kDir[0]*x+kDir[1]*y+kDir[2]*z - omega*t +phi)
        self.assertTrue((eta - eta_mode([x,y,z],t,kDir,omega,phi,amplitude)==0.))# check eta

    def testUdrift(self):
        from proteus.WaveTools import Udrift
        amp = 0.1
        gAbs = 9.81
        c = 1.
        height = 2.*amp
        depth = 1
        self.assertTrue(0.125*gAbs*height**2/c/depth == Udrift(amp,gAbs,c,depth))
    def testVelMode(self): # Checking particle velocities
        from proteus.WaveTools import vel_mode, Udrift

        kDir = np.array([2*pi,0.0,0.0])# Wavelength == 1
        omega = 2*pi
        phi =0.
        amplitude = 1.
        g = np.array( [0,0.,-9.81])
        gAbs = 9.81
        depth = 2.
        mwl =3.
        x=  pi/4./kDir[0]
        y = 0.
        z= 2.
        vDir = np.array([0,0,1])
        t= 0.
        kAbs = 2*pi
        Ud = Udrift(amplitude,abs(g[-1]),omega/kAbs,depth)
        for i in range(4):
            U_x, U_y, U_z = vel_mode([x,y,z],t,kDir,kAbs,omega,phi,amplitude,mwl,depth,vDir,gAbs)
            x+= 0.25
            U_x = U_x+Ud
            # Checking velocity signs with quadrants
            if i ==0:
                #1st Quadrant
               self.assertTrue((U_x > 0.) and (U_y == 0.) and (U_z >0.))
            if i ==1:
                #2nd Quadrant
               self.assertTrue((U_x < 0.) and (U_y == 0.) and (U_z > 0.))
            if i ==2:
                #3rd Quadrant
               self.assertTrue((U_x < 0.) and (U_y == 0.) and (U_z < 0.))
            if i ==3:
                #4th Quadrant
               self.assertTrue((U_x > 0.) and (U_y == 0.) and (U_z < 0.))
        #Checking that the code does not allow z to be outside (-d,0)
#Checking vertical coherency
# U_z = 0 at z = mwl-d
        self.assertTrue(vel_mode([x,y,1.],t,kDir,kAbs,omega,phi,amplitude,mwl,depth,vDir,gAbs)[2]==0.)

    def testTophat(self):
        from proteus.WaveTools import tophat
        a  = np.random.rand(100)
        filt = tophat(100,0.1)
        af = a*filt
        a[:10] = 0.
        a[-10:] =0.
        self.assertTrue( a.all() == af.all())

    def testcosTap(self):
        from proteus.WaveTools import costap
        a  = np.random.rand(100)
        filt = costap(100,0.1)
        af = a*filt
        a[:10] = 0.5*(1.-np.cos(pi*np.linspace(0,9,10)/10.))
        a[-10:] =0.5*(1.-np.cos(pi*np.linspace(9,0,10)/10.))
        self.assertTrue( a.all() == af.all())

    def testDecomposeFFT(self):
        from proteus.WaveTools import decompose_tseries
        dt = 0.01
        time = np.linspace(0,199,200 )
        eta = np.cos(0.2*pi*time + 0.2)
        nfft = len(time)
        # testing full decompistion
        dec = decompose_tseries(time,eta,1)
        time = np.linspace(0,199,len(dec[1]))
        eta = np.cos(0.2*pi*time + 0.2)
        rec = np.zeros(len(time),)

        for ii in range(len(dec[0])):
            rec[:]+=dec[1][ii]*np.cos(dec[0][ii]*time[:]+dec[2][ii])
        rec[:]+=dec[3]
        self.assertTrue( rec.all() == eta.all())


class TestWaveParameters(unittest.TestCase):
#Checking dispersion calculation for a predicted wavelenght of 5.00m
    def test_dispersion(self):
        from proteus.WaveTools import dispersion
        length = 2*pi/dispersion(2*pi/5.,4.)
        lTheor = 27.958
        self.assertTrue(abs(length-lTheor)/lTheor<0.001)
        length = 2*pi/dispersion([2*pi/1.94,2*pi/1.94,],1.)
        length-=5.
        length/=5
        self.assertTrue( (all(length) <0.001) or  (all(length) > -0.001))

#Check  sigma
    def test_sigma(self):
        from proteus.WaveTools import sigma,JONSWAP
        omega0=0.01
        sigma0 = 0.07
        sigma1 = 0.09
        x = np.ones((3),'d')
        x[0] = 0.5*omega0
        x[1] = omega0
        x[2] = 2.*omega0
        sigma = sigma(x,omega0)
        self.assertTrue((sigma[0] == sigma0).all())
        self.assertTrue((sigma[1] == sigma0).all())
        self.assertTrue((sigma[2] == sigma1).all())

    def test_Jonswap(self): #JONSWAP tests
# Test Jonswap spectrum without TMA modification
        from proteus.WaveTools import sigma, JONSWAP, dispersion
        import random
        f0 = random.random() + 1.
        f = np.linspace(f0/2.,2.*f0,100)
        sig = sigma(f,f0)
        gamma = 6.*random.random() + 1.
        Hs = random.random()
        bj = 0.0624*(1.094 - 0.01915*log(gamma))/(0.23+0.0336*gamma-0.185/(1.9+gamma))
        r_exp = np.exp(-(f/f0 -1 )**2/(2.*sig**2))
        JON = (bj*(Hs**2)*(f0**4)/f**5)*np.exp(-1.25*(f0/f)**4)*(gamma**r_exp)
        JON2 = JONSWAP(f,f0,Hs,gamma,TMA=False, depth = None)

        JCOMP = JON2/JON
        self.assertTrue((np.around(JCOMP,10)==1).all())
        h = random.random()
# Checking failure mode
        with self.assertRaises(SystemExit) as cm:
            JON2 = JONSWAP(f,f0,Hs,gamma,TMA=True)
        self.assertEqual(cm.exception.code, 1)
# Check TMA modification
        k = dispersion(2*pi*f,h)
        TMA = np.tanh(k*h)*np.tanh(k*h)/(1.+2.*k*h/np.sinh(2*k*h))
        JON2 = JONSWAP(f,f0,Hs,gamma,TMA=True, depth=h)
        JCOMP = JON2/(TMA*JON)
        self.assertTrue((np.around(JCOMP,10)==1).all())

    def test_PM(self): #PM tests
        from proteus.WaveTools import PM_mod
        f0 = random.random() + 1.
        f = np.linspace(f0/2.,2.*f0,10)
        Hs = random.random()
        g = 9.81
        S_PM = (5./16.) * Hs**2 * f0**4 / f**5 * np.exp(-5./4. * (f0/f)**4)
        S_PM2 =  PM_mod(f,f0,Hs)
        SCOMP = S_PM2/S_PM
        self.assertTrue((np.around(SCOMP,10)==1).all())

    def testCos2s(self):
        from proteus.WaveTools import cos2s
        from math import cos
        f0 = random.random() + 1.
        f = np.linspace(f0/2.,2.*f0,10)
        thetas = np.linspace(-pi/2.,pi/2.,11)
        s = 10. + 10. * random.random()
        S_PM = np.zeros((len(thetas),len(f)),'d')
        for ii in range(len(thetas)):
            for jj in range(len(f)):
                S_PM[ii,jj]= cos(thetas[ii]/2.0)**(2*s)
        S_PM2 =  cos2s(thetas,f,s)
        SCOMP = S_PM2/S_PM
        npt.assert_array_almost_equal(S_PM,S_PM2)

    def testMitsuyasu(self):
        from proteus.WaveTools import mitsuyasu
        from math import cos
        f0 = random.random() + 1.
        f = np.linspace(f0/2.,2.*f0,10)
        thetas = np.linspace(-pi/2.,pi/2.,11)
        smax = 10 + 10. * random.random()
        ss = np.zeros(len(f),)
        ss = smax*(f/f0)**5
        i = np.where(f>f0)[0][0]
        ss[i:] = smax*(f[i:]/f0)**(-2.5)
        S_PM = np.zeros((len(thetas),len(f)),'d')
        for ii in range(len(thetas)):
            for jj in range(len(f)):
                S_PM[ii,jj]= cos(thetas[ii]/2.)**(2*ss[jj])
        S_PM2 =  mitsuyasu(thetas,f,f0,smax)
        npt.assert_array_almost_equal(S_PM,S_PM2)

class VerifySteadyCurrent(unittest.TestCase):
    def testCurrent(self):
        from proteus.WaveTools import SteadyCurrent
        U = np.array([2.5,2.,1.])
        mwl = 0.5
        xx = np.array([2.,0.,0.,])
        t = 0.1

        # no ramptime
        WW = SteadyCurrent(U,mwl)
        self.assertAlmostEqual(U.all(), WW.u(xx,t).all())
        self.assertAlmostEqual(0., WW.eta(xx,t))

        # with ramp
        Up = 0.5*U
        WW = SteadyCurrent(U,mwl,0.2)
        self.assertAlmostEqual(Up.all(), WW.u(xx,t).all())
        self.assertAlmostEqual(0., WW.eta(xx,t))
    def testCurrentFailure(self):
        from proteus.WaveTools import SteadyCurrent
        U = 1.
        mwl = 0.5

        with self.assertRaises(SystemExit) as cm1:
            SteadyCurrent(U,mwl,0.2)
        self.assertEqual(cm1.exception.code, 1)



class VerifySolitaryWave(unittest.TestCase):
    def testSolitary(self):
        from proteus.WaveTools import SolitaryWave
        HH = 0.1
        mwl = 0.
        dd = 1.5
        g = np.array([0,-9.81,0])
        waveDir = np.array([5.,0.,0.])
        trans = np.array([1. ,0., 0.])

        #No translation
        aa = SolitaryWave(HH,mwl,dd,g,waveDir)

        x = 2.5
        t = 5.

        cc = np.sqrt(9.81*(dd+HH))
        eta_Ref = HH/np.cosh(np.sqrt(3.*HH/4./dd**3)*(x-cc*t))**2
        xx = x*waveDir/5.
        self.assertAlmostEqual(eta_Ref, aa.eta(xx,t))

        def pow(a,b):
            return a**b
        h_ = dd
        G_ = 9.81
        H_ = HH
        Z = -0.1
        d2 = dd**2
        d3 = dd**3

        c =  np.sqrt(G_ * (dd+HH))
        K =  np.sqrt(3. *HH/ (4. * dd))/dd

        xx[1] = -0.2
        Z = xx[1] - mwl
        sqrt = np.sqrt
        cosh = np.cosh
        tanh = np.tanh
        phase = sum( xx[:]*waveDir[:]/5.)  - c * t
        a1 =  cosh(K*phase*2.)
        a2 =  cosh(K*phase)

        Uhorz =  1.0 /(4.0 * dd**4 ) * np.sqrt(G_ * dd) *  HH *(
            2.0 * d3 + d2 * HH  + 12.0 * dd * HH * Z + 6.0 *  HH * Z**2.0 +
            (2.0 * d3 - d2 * HH - 6.0 * dd * HH * Z - 3.0 * HH * Z**2 ) * a1)/(a2)**4

        Uvert =   1.0 / ( 4.0 * np.sqrt(G_* dd) ) * np.sqrt(3.0) * G_ * (HH/dd**3.0)** 1.5  * (dd + Z)*(
                2.0 * dd**3 - 7.0 * dd**2.0 * HH + 10.0 * dd * HH * Z + 5.0 * HH * Z**2.0 +
                (2.0 * dd**3.0 + dd**2.0 * HH - 2.0 * dd * HH * Z - HH * Z**2.0)*
                cosh(np.sqrt( 3.0 * HH / dd**3.0) * phase ))/(
                cosh(np.sqrt( 3.0 * HH / ( 4.0 * dd**3.0))*
                phase )   )** 4.0*( tanh( np.sqrt( 3.0 * HH / ( 4.0 * dd**3.0))*phase ))

        self.assertAlmostEqual(Uhorz, aa.u(xx,t)[0])
        self.assertAlmostEqual(Uvert, aa.u(xx,t)[1])


class CheckMonochromaticWavesFailures(unittest.TestCase):
    def testFailureModes(self):
        from proteus.WaveTools import MonochromaticWaves
#Failure 1: Give non existent waveType
        with self.assertRaises(SystemExit) as cm1:
            MonochromaticWaves(1.,1.,0.,10.,np.array([0,0,-9.81]),np.array([1,0,0]),wavelength=None,waveType="Error",autoFenton=False,Ycoeff = np.array([1.]), Bcoeff = np.array([1.]),Nf = 1, meanVelocity = np.array([0.,0.,0.]),phi0 = 0.)
        self.assertEqual(cm1.exception.code, 1)
#Failure 2: Give gravity direction not vertical to wave direction
        with self.assertRaises(SystemExit) as cm2:
            MonochromaticWaves(1.,1.,0.,10.,np.array([0,0,1]),np.array([0,0,-9.81]),wavelength=None,waveType="Linear",autoFenton=False,Ycoeff = np.array([1.]), Bcoeff = np.array([1.]),Nf = 1, meanVelocity = np.array([0.,0.,0.]),phi0 = 0.)
        self.assertEqual(cm2.exception.code, 1)
# Failure 3: Give Fenton type without wavelength
        with self.assertRaises(SystemExit) as cm3:
            MonochromaticWaves(1.,1.,0.,10.,np.array([0,0,-9.81]),np.array([1,0,0]),wavelength=None,waveType="Fenton",autoFenton=False,Ycoeff = np.array([1.]), Bcoeff = np.array([1.]),Nf = 1, meanVelocity = np.array([0.,0.,0.]),phi0 = 0.)
        self.assertEqual(cm3.exception.code, 1)
# Failure 4: Give Fenton type without YCoeff
        with self.assertRaises(SystemExit) as cm4:
            MonochromaticWaves(1.,1.,0.,10.,np.array([0,0,-9.81]),np.array([0,1,0]),wavelength=5.,waveType="Fenton",autoFenton=False,Ycoeff = np.array([0.]), Bcoeff = np.array([1.]), Nf = 1, meanVelocity = np.array([0.,0.,0.]),phi0 = 0.)
        self.assertEqual(cm4.exception.code, 1)
# Failure 5: Give Fenton type without BCoeff
        with self.assertRaises(SystemExit) as cm5:
            MonochromaticWaves(1.,1.,0.,10.,np.array([0,0,-9.81]),np.array([0,1.,0]),wavelength=5.,waveType="Fenton",autoFenton=False,Ycoeff = np.array([1.]), Bcoeff =np.array([0.]),Nf = 1, meanVelocity = np.array([0.,0.,0.]),phi0 = 0.)
        self.assertEqual(cm5.exception.code, 1)
  # Failure 6: Give meanVelocity a vector value but not with 3 components
        with self.assertRaises(SystemExit) as cm6:
            MonochromaticWaves(1.,1.,0.,10.,np.array([0,0,-9.81]),np.array([0,1,0]),wavelength=5.,waveType="Fenton",autoFenton=False,Ycoeff = np.array([1.,1,1.]), Bcoeff =np.array([1.,1,1.]), Nf = 3, meanVelocity =np.array([0.,0.,0.,0.]) ,phi0 = 0.)
        self.assertEqual(cm6.exception.code, 1)
  # Failure 7: Not giving the correct Nf
        with self.assertRaises(SystemExit) as cm7:
            MonochromaticWaves(1.,1.,0.,10.,np.array([0,0,-9.81]),np.array([0,1,0]),wavelength=5.,waveType="Fenton",autoFenton=False,Ycoeff = np.array([1.,1.,1.]), Bcoeff =np.array([1.,1.,1.]), meanVelocity =np.array([0.,0.,0.]) ,phi0 = 0.)
        self.assertEqual(cm7.exception.code, 1)
  # Success!: Give all parameters in correct form!
        a = MonochromaticWaves(1.,1.,0.,10.,np.array([0,0,-9.81]),np.array([0,1,0]),wavelength=5.,waveType="Fenton",autoFenton=False,Ycoeff = np.array([1.,1.,1.]), Bcoeff =np.array([1.,1.,1.]), Nf = 3, meanVelocity =np.array([0.,0.,0.]) ,phi0 = 0.)
        self.assertTrue(None is None)

class VerifyMonoChromaticLinearWaves(unittest.TestCase):
    def testLinear(self):
        from proteus.WaveTools import MonochromaticWaves
        from proteus.WaveTools import fastcos_test as fcos
        from proteus.WaveTools import coshkzd_test as fcosh
        from proteus.WaveTools import sinhkzd_test as fsinh
        from proteus.WaveTools import Udrift as Ud

# Wave direction, random in x,y plane
        period = 2.
        waveHeight = 1.
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.6
        dir2 = 0.5
        waveDir = np.array([dir1,dir2, 0])
        phi0 = 0.
        a = MonochromaticWaves(period,waveHeight,mwl,depth,g,waveDir,wavelength=None,waveType="Linear",Ycoeff = np.array([0.]), Bcoeff  = np.array([0.]), meanVelocity = np.array([0.,0,0.]),phi0 = phi0, fast = False)
        x = 150.
        y = 130.
        z = mwl
        t =  125.
        eta = a.eta([x, y, z], t)
        ux, uy, uz = a.u([x, y, z], t)

        omega = 2.*pi/period
        Uo = waveHeight*omega / 2.
# dispersion and setDirVector are tested above
        from proteus.WaveTools import dispersion,setDirVector
        kw = dispersion(omega,depth,gAbs)
        normDir = setDirVector(waveDir)
        amp = 0.5 * waveHeight
# Flow equation from Wikipedia, Airy wave theory https://en.wikipedia.org/wiki/Airy_wave_theoryhttps://en.wikipedia.org/wiki/Airy_wave_theory
        etaRef = amp*cos(kw*(normDir[0]*x+normDir[1]*y+normDir[2]*z) - omega * t +phi0)
        z0 = z - mwl

        uxRef = normDir[0]*(amp*omega*fcosh(kw,z0,depth,fast=False)*cos(kw*(normDir[0]*x+normDir[1]*y+normDir[2]*z) - omega * t +phi0)-Ud(amp,gAbs,omega/kw,depth))
        uyRef = normDir[1]*(amp*omega*fcosh(kw,z0,depth,fast=False)*cos(kw*(normDir[0]*x+normDir[1]*y+normDir[2]*z) - omega * t +phi0)-Ud(amp,gAbs,omega/kw,depth))
        uzRef = amp*omega*fsinh(kw,z0,depth,fast=False)*sin(kw*(normDir[0]*x+normDir[1]*y+normDir[2]*z) - omega * t +phi0)

       # print ux,uxRef
        err = abs(eta/etaRef - 1.)
        err_x = abs(ux/uxRef - 1.)
        err_y = abs(uy/uyRef - 1.)
        err_z = abs(uz/uzRef - 1.)
        self.assertTrue((err <= 1e-8))
        self.assertTrue((err_x <= 1e-8))
        self.assertTrue((err_y <= 1e-8))
        self.assertTrue((err_z <= 1e-8))

class VerifyMonoChromaticFentonWaves(unittest.TestCase):
#Fenton methodology equations at http://johndfenton.com/Papers/Fenton88-The-numerical-solution-of-steady-water-wave-problems.pdf
#http://johndfenton.com/Steady-waves/Fourier.html

    def testFenton(self):
        from proteus.WaveTools import MonochromaticWaves
        from proteus.WaveTools import fastcos_test as fcos
        from proteus.WaveTools import coshkzd_test as fcosh
        from proteus.WaveTools import sinhkzd_test as fsinh
        from proteus.WaveTools import Udrift as Ud
        period = 1.
        waveHeight = 0.15
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.5
        dir2 = 0.45
        waveDir = np.array([dir1,dir2, 0])
        phi0 = 0.3
        wl = 10.
        YC =  np.array([.05,.001,.0001,.0001])
        BC =  np.array([.05,.001,.001,.001])
        mv =  np.array([1.,1.,1.])
# Set-up of Y and B coeffs does not correspond to physical properties
        a = MonochromaticWaves(period,waveHeight,mwl,depth,g,waveDir,wavelength=wl,waveType="Fenton",autoFenton=False,Ycoeff = YC, Bcoeff = BC, Nf = len(YC), meanVelocity = mv,phi0 = phi0)
        x = 145.
        y = 155.
        z =  mwl
        t =  125.
        eta = a.eta([x, y, z], t)
        ux, uy, uz = a.u([x, y, z], t)
        omega = 2.*pi/period
# setDirVector are tested above
        from proteus.WaveTools import setDirVector
        kw = 2*pi/wl
        z0 = z - mwl
        normDir = setDirVector(waveDir)
        amp = 0.5 * waveHeight
        etaRef = 0.
        uxRef = 0.
        uyRef = 0.
        uzRef = 0.
        jj = 0
        uxRef= mv[0]
        uyRef= mv[1]
        uzRef= mv[2]

        for ii in range(len(YC)):
            jj+=1
            etaRef+=YC[ii]*fcos(jj*kw*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-jj*omega*t + jj*phi0)/kw
            amp = tanh(kw*jj*depth)*np.sqrt(gAbs/kw)*BC[ii]/omega
            c = omega/kw
            uxRef += normDir[0]*( np.sqrt(gAbs/kw)*jj*BC[ii]*fcosh(jj*kw,z0,depth) *fcos(jj*kw*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-jj*omega*t +jj*phi0)*tanh(jj*kw*depth)-Ud(amp,gAbs,c,depth))
            uyRef += normDir[1]* (np.sqrt(gAbs/kw)*jj*BC[ii]*fcosh(jj*kw,z0,depth) *fcos(jj*kw*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-jj*omega*t +jj*phi0)*tanh(jj*kw*depth)-Ud(amp,gAbs,c,depth))
            uzRef +=  np.sqrt(gAbs/kw)*jj*BC[ii]*fsinh(jj*kw,z0,depth) *fcos(0.5*pi -( jj*kw*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-jj*omega*t +jj*phi0))*tanh(jj*kw*depth)

        err = abs(eta/etaRef - 1.)
        err_x = abs(ux/uxRef - 1.)
        err_y = abs(uy/uyRef - 1.)
        err_z = abs(uz/uzRef - 1.)
        Uo = waveHeight * omega * tanh(kw*depth)
        Uoz = waveHeight * omega


        self.assertTrue((err <= 1e-8))
        self.assertTrue((err_x <= 1e-08))
        self.assertTrue((err_y <= 1e-08))
        self.assertTrue((err_z <= 1e-08))

    def testAutoFenton(self):
        waveHeight = 0.05
        depth = 0.45
        period = 2.5
        Nf = 8
        g = np.array([0,-9.81,0])
        #Values calculated from the FFT tool, as we would if autoFenton=False
        wavelength = 5.032
        Bc = [ 0.04267051, 0.00339916, 0.00025773, 0.00001592, 0.00000056, -0.00000003, -0.00000001, 0.00000000]
        Yc = [ 0.03057459, 0.00477966, 0.00063110, 0.00008325, 0.00001146, 0.00000166, 0.00000026, 0.00000008]
        from proteus.fenton import Fenton
        autoFentonOpts = {'mode': 'Period',
                          'current_criterion': 1,
                          'height_steps': 1,
                          'niter': 40,
                          'conv_crit': 1e-05,
                          'points_freesurface': 50,
                          'points_velocity': 16,
                          'points_vertical': 20}
        Fenton.writeInput(waveheight=waveHeight,
                          depth=depth,
                          period=period,
                          mode=autoFentonOpts['mode'],
                          current_criterion=autoFentonOpts['current_criterion'],
                          current_magnitude=0,
                          ncoeffs=Nf,
                          height_steps=autoFentonOpts['height_steps'],
                          g=np.linalg.norm(g),
                          niter=autoFentonOpts['niter'],
                          conv_crit=autoFentonOpts['conv_crit'],
                          points_freesurface=autoFentonOpts['points_freesurface'],
                          points_velocity=autoFentonOpts['points_velocity'],
                          points_vertical=autoFentonOpts['points_vertical'])
        Fenton.runFourier()
        Fenton.copyFiles()
        Bc_test, Yc_test = Fenton.getBYCoeffs()
        wl_test = Fenton.getWavelength()*depth
        err =  (wl_test-wavelength)/wavelength
        self.assertTrue((err <= 1e-3))
        self.assertEqual(np.round(Bc_test,7).all(), np.round(Bc,7).all())
        self.assertEqual(np.round(Yc_test,7).all(), np.round(Yc,7).all())



#========================================= RANDOM WAVES ======================================



class CheckRandomWavesFailures(unittest.TestCase):
    def testFailureModes(self):
        from proteus.WaveTools import RandomWaves
#Failure 1: Give a wrong spectrum name
        with self.assertRaises(SystemExit) as cm1:
            RandomWaves(1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),100,2.,"blahblah", spectral_params= None )
        self.assertEqual(cm1.exception.code, 1 )

#Failure 2: Give gravity direction not vertical to wave direction
        with self.assertRaises(SystemExit) as cm2:
            RandomWaves(1.,1.,0.,10.,np.array([0,0,1]),np.array([0,0,-9.81]),100,2.,"JONSWAP", spectral_params=None )
        self.assertEqual(cm2.exception.code, 1 )
#Failure 3: Give random spectral parameters
        with self.assertRaises(SystemExit) as cm3:
            RandomWaves(1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),100,2.,"JONSWAP", spectral_params= {"random1": 3.3, "random2":True,"random3" : 10.}  )
        self.assertEqual(cm3.exception.code, 1)
#Failure 4: Give wrong type of phases
        with self.assertRaises(SystemExit) as cm4:
            RandomWaves(1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),100,2.,"JONSWAP", spectral_params= {"random1": 3.3, "random2":True,"random3" : 10.}, phi = 0.  )
        self.assertEqual(cm4.exception.code, 1)
#Failure 5: Give wrong number of phase
        with self.assertRaises(SystemExit) as cm5:
            RandomWaves(1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),100,2.,"JONSWAP", spectral_params= {"random1": 3.3, "random2":True,"random3" : 10.}, phi = np.zeros(1,)  )
        self.assertEqual(cm5.exception.code, 1)
#Failure 6: Give too many frequencies
        with self.assertRaises(SystemExit) as cm6:
            RandomWaves(1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),100001,2.,"JONSWAP", spectral_params= {"random1": 3.3, "random2":True,"random3" : 10.}  )
        self.assertEqual(cm6.exception.code, 1)

  # Success!: Give all parameters in correct form!
        RandomWaves(2.,1.,0.,1.,np.array([0,0,1]),np.array([0,1,0]),100,2.,"JONSWAP", spectral_params=None )
        RandomWaves(2.,1.,0.,1.,np.array([0,0,1]),np.array([0,1,0]),100,2.,"JONSWAP", spectral_params={"gamma": 3.3, "TMA":True,"depth": 10.} )
        RandomWaves(2.,1.,0.,1.,np.array([0,0,1]),np.array([0,1,0]),100,2.,"JONSWAP", spectral_params={"gamma": 3.3, "TMA":True,"depth": 10.}, phi = np.zeros(100, float) )
        self.assertTrue(None is None)


class VerifyNewWaves(unittest.TestCase):
    def testNewWave(self):
        from proteus.WaveTools import NewWave,dispersion
        import random
        from proteus.WaveTools import fastcos_test as fcos
        from proteus.WaveTools import coshkzd_test as fcosh
        from proteus.WaveTools import sinhkzd_test as fsinh
        # Assinging a random value at a field and getting the expected output
        Tp = 2.
        Hs = 0.1
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 1.
        dir2 = 0.
        waveDir = np.array([dir1,dir2, 0])
        N = 32
        crestFocus = True
        xfocus = np.array([0.,0.,0.])
        tfocus = 10.
        gamma = 3.3
        TMA = False
        spectName = "JONSWAP"
        bandFactor = 2.0
        Nmax = 10000
        a= NewWave(Tp,
                  Hs,
                  mwl,#m significant wave height
                  depth ,           #m depth
                  waveDir,
                  g,      #peak  frequency
                  N,
                  bandFactor,         #accelerationof gravity
                  spectName ,# random words will result in error and return the available spectra
                  spectral_params =  None, #JONPARAMS = {"gamma": 3.3, "TMA":True,"depth": depth}
                  crestFocus=True,
                   xfocus=xfocus,
                   tfocus=tfocus,
                   Nmax = Nmax,
                   fast = True)

        eta = np.zeros(3000,)
        tt = np.linspace(0,30,3000)
        ii=-1
        for t in tt:
            ii+=1
            eta[ii] = a.eta(xfocus,t)

        np.savetxt(X=eta,fname="eta.txt")
        imax=np.where(tt>=tfocus)[0][0]
        self.assertAlmostEqual(eta[imax],max(eta))
        self.assertAlmostEqual(max(eta),0.107290525)

class VerifyRandomWaves(unittest.TestCase):
    def testRandom(self):
        from proteus.WaveTools import RandomWaves,dispersion
        import random
        from proteus.WaveTools import fastcos_test as fcos
        from proteus.WaveTools import coshkzd_test as fcosh
        from proteus.WaveTools import sinhkzd_test as fsinh
        # Assinging a random value at a field and getting the expected output
        Tp = 2.
        Hs = 0.15
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.1
        dir2 = 2
        waveDir = np.array([dir1,dir2, 0])
        N = 100
        phi = np.linspace(1,N,N)
        gamma = 1.2
        TMA = True
        spectName = "JONSWAP"
        bandFactor = 2.0

        a= RandomWaves(Tp,
                     Hs,
                     mwl,#m significant wave height
                     depth ,           #m depth
                     waveDir,
                     g,      #peak  frequency
                     N,
                     bandFactor,         #accelerationof gravity
                     spectName# random words will result in error and return the available spectra
                   )
        wl = 2*np.pi/dispersion(2*np.pi/Tp,depth)
        self.assertAlmostEqual(wl,a.wavelength)
        x = 103.
        y = 112.
        z =  mwl
        t =  145.
        # Just loading functions
        eta = a.eta([x, y, z], t)
        ux, uy, uz = a.u([x, y, z], t)

        # Testing with a specific phi array
        a= RandomWaves(
            Tp,
            Hs,
            mwl,#m significant wave height
            depth ,           #m depth
            waveDir,
            g,      #peak  frequency
            N,
            bandFactor,         #accelerationof gravity
            spectName,
            spectral_params =  {"gamma": gamma, "TMA": TMA,"depth": depth},
            phi = phi# random words will result in error and return the available spectra
    )
        eta = a.eta([x, y, z], t)
        ux, uy, uz = a.u([x, y, z], t)


        # setDirVector are tested above
        from proteus.WaveTools import setDirVector, dispersion, reduceToIntervals, returnRectangles, JONSWAP,Udrift
        fmin = 1./(Tp*bandFactor)
        fmax = bandFactor/(Tp)
        fi = np.linspace(fmin,fmax,N)
        df = (fmax-fmin)/(N-1)
        ki = dispersion(2*pi*fi,depth)
        kp = dispersion(2*pi / Tp,depth)
        z0 = z - mwl
        normDir = setDirVector(waveDir)
        fim = reduceToIntervals(fi,df)
        Si_Jm = JONSWAP(fim,1./Tp,Hs,gamma,TMA, depth)
        ai = np.sqrt(2.*returnRectangles(Si_Jm,fim))
        omega = 2*pi*fi
        etaRef = 0.
        uxRef = 0.
        uyRef = 0.
        uzRef = 0.

        for ii in range(N):
            etaRef+=ai[ii]*fcos(ki[ii]*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-omega[ii]*t +phi[ii])
            uxRef += normDir[0]*ai[ii]*omega[ii] *fcosh(ki[ii],z0,depth) *fcos(ki[ii]*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-omega[ii]*t +phi[ii])-normDir[0]*Udrift(ai[ii],gAbs,omega[ii]/ki[ii],depth)
            uyRef += normDir[1]*ai[ii]*omega[ii] *fcosh(ki[ii],z0,depth) * fcos(ki[ii]*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-omega[ii]*t +phi[ii])-normDir[1]*Udrift(ai[ii],gAbs,omega[ii]/ki[ii],depth)
            uzRef +=  ai[ii]*omega[ii] *fsinh(ki[ii],z0,depth) * fcos(ki[ii]*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-omega[ii]*t +phi[ii],True)

        err = abs(eta/etaRef - 1.)
        err_x =abs(ux/uxRef - 1.)
        err_y =abs(uy/uyRef - 1.)
        err_z =abs(uz/uzRef - 1.)


        self.assertTrue(err <= 1e-8)
        self.assertTrue(err_x <= 1e-8)
        self.assertTrue(err_y <= 1e-8)
        self.assertTrue(err_z <= 1e-8)




        # Asserting write function from Random waves
        x0 = np.array([0,0,0])
        Lgen = np.array([5,0,0])
        Tstart = 0
        Tend = 2.
        Tlag = sum(Lgen[:]*normDir[:])/min(omega[:]/ki[:])
        Tstart2 = Tstart -  Tlag
        dt = Tp/50.
        Nf = int((Tend-Tstart2)/dt)
        tlist = np.linspace(Tstart2,Tend,Nf)
        etaWrite = np.zeros(len(tlist),)
        for ii in range(len(tlist)):
            etaWrite[ii] = a.eta(x0,tlist[ii])
        path =getpath()
        fname = "randomSeries.txt"      # the working directory, where remove_files looks
        filenames = ['randomSeries.txt']
        if Tlag < 0.:
            with self.assertRaises(SystemExit) as cm1:
                a.writeEtaSeries(Tstart,Tend,x0,fname, Lgen)
            self.assertEqual(cm1.exception.code, 1 )
        else:
            a.writeEtaSeries(Tstart,Tend,x0,fname, Lgen)
            series = np.loadtxt(open(fname,"r"))
            remove_files(filenames)
            self.assertTrue((abs(series[:,0])- abs(tlist) <= 1e-10  ).all())
            self.assertTrue((abs(series[:,1])- abs(etaWrite) <= 1e-10).all())





# Test contours
"""
        xi = np.linspace(0,20,101)
        yi = np.linspace(0,20,101)

        eta_t = np.zeros((101,101),)
        for i in range(len(xi)):
            for j in range(len(yi)):
                eta_t[i,j] = aa.eta(xi[i],yi[j],0.,0.)



        fig = plt.figure(2)
        X,Y = np.meshgrid(xi,yi)
        CS = plt.contourf(X,Y,eta_t)
#        fig.legend((line2,line1),("Field data from FRF", "Reconstructed time series"),"upper right",bbox_to_anchor=(0.6,0.9))
#        plt.xlabel("x (m)",size=20)
#        plt.ylabel("y (m)",size=20)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.savefig("Contour.png")
"""

class CheckMultiSpectraRandomWavesFailures(unittest.TestCase):
    def testFailureModes(self):

        from proteus.WaveTools import MultiSpectraRandomWaves,Udrift
#Failure 1: Give parameters as float rather than list
        Tp = 1.
        Hs = 0.15
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.8
        dir2 = 0.6
        waveDir = np.array([dir1,dir2, 0])
        N = 100
        phi = np.linspace(1,N,N)
        gamma = 1.2
        TMA = True
        spectName = "JONSWAP"
        bandFactor = 2.0
# Give wrong length of list
        with self.assertRaises(SystemExit) as cm1:
            MultiSpectraRandomWaves(
            2,
            Tp,
            [Hs,Hs],
            mwl,#m significant wave height
            depth ,           #m depth
            [waveDir,waveDir],
            g,      #peak  frequency
            np.array([N,N]),
            [bandFactor,bandFactor],         #accelerationof gravity
            [spectName, spectName],
            spectral_params =[  {"gamma": gamma, "TMA": TMA,"depth": depth},  {"gamma": gamma, "TMA": TMA,"depth": depth} ],
            phi = [phi,phi]
        )
        self.assertEqual(cm1.exception.code, 1 )

        with self.assertRaises(SystemExit) as cm2:
            MultiSpectraRandomWaves(
            2,
                [Tp,Tp,Tp],
            [Hs,Hs],
            mwl,#m significant wave height
            depth ,           #m depth
            [waveDir,waveDir],
            g,      #peak  frequency
            np.array([N,N]),
            [bandFactor,bandFactor],         #accelerationof gravity
            [spectName, spectName],
            spectral_params =[  {"gamma": gamma, "TMA": TMA,"depth": depth},  {"gamma": gamma, "TMA": TMA,"depth": depth} ],
            phi = [phi,phi]
    )

        self.assertEqual(cm2.exception.code, 1 )
#Failure 6: Give too many frequencies
        with self.assertRaises(SystemExit) as cm3:
            MultiSpectraRandomWaves(
            2,
            [Tp,Tp],
            [Hs,Hs],
            mwl,#m significant wave height
            depth ,           #m depth
            [waveDir,waveDir],
            g,      #peak  frequency
            np.array([5000,5001]),
            [bandFactor,bandFactor],         #accelerationof gravity
            [spectName, spectName],
            spectral_params =[  {"gamma": gamma, "TMA": TMA,"depth": depth},  {"gamma": gamma, "TMA": TMA,"depth": depth} ],
            phi = [phi,phi])
        self.assertEqual(cm3.exception.code, 1)

        # Success!: Give all parameters in correct form!
        MultiSpectraRandomWaves(
            2,
            [Tp,Tp],
            [Hs,Hs],
            mwl,#m significant wave height
            depth ,           #m depth
            [waveDir,waveDir],
            g,      #peak  frequency
            np.array([N,N]),
            [bandFactor,bandFactor],         #accelerationof gravity
            [spectName, spectName],
            spectral_params =[  {"gamma": gamma, "TMA": TMA,"depth": depth},  {"gamma": gamma, "TMA": TMA,"depth": depth} ],
            phi = [phi,phi]
    )

        self.assertTrue(None is None)



class VerifyMultiSpectraRandomWaves(unittest.TestCase):
    def testMultiSpectraDoubleExact(self):
        from proteus.WaveTools import MultiSpectraRandomWaves, RandomWaves,Udrift
        Tp = 1.
        Hs = 0.15
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.15
        dir2 = 2.
        waveDir = np.array([dir1,dir2, 0])
        waveDir2 = np.array([2.*dir1,dir2, 0])
        N = 100
        phi = np.linspace(1,5,N)
        gamma = 1.2
        TMA = True
        spectName = "JONSWAP"
        bandFactor = 2.0
        x = 135.
        y = 123.
        z =  mwl
        t =  112.
        # Testing with a specific phi array
        a= RandomWaves(
            Tp,
            Hs,
            mwl,#m significant wave height
            depth ,           #m depth
            waveDir,
            g,      #peak  frequency
            N,
            bandFactor,         #accelerationof gravity
            spectName,
            spectral_params =  {"gamma": gamma, "TMA": TMA,"depth": depth},
            phi = phi
    )
        a2= RandomWaves(
            Tp,
            Hs,
            mwl,#m significant wave height
            depth ,           #m depth
            waveDir2,
            g,      #peak  frequency
            N,
            bandFactor,         #accelerationof gravity
            spectName,
            spectral_params =  {"gamma": gamma, "TMA": TMA,"depth": depth},
            phi = phi
    )
        eta = a.eta([x, y, z], t)
        ux, uy, uz = a.u([x, y, z], t)
        etaa = a2.eta([x, y, z], t)
        uxa, uya, uza = a2.u([x, y, z], t)

#Doubling the spectral properties
        aa= MultiSpectraRandomWaves(
            2,
            [Tp,Tp],
            [Hs,Hs],
            mwl,#m significant wave height
            depth ,           #m depth
            [waveDir,waveDir2],
            g,      #peak  frequency
            np.array([N,N]),
            [bandFactor,bandFactor],         #accelerationof gravity
            [spectName, spectName],
            spectral_params =[  {"gamma": gamma, "TMA": TMA,"depth": depth},  {"gamma": gamma, "TMA": TMA,"depth": depth} ],
            phi = [phi,phi]# random words will result in error and return the available spectra
    )
        eta2 = aa.eta([x, y, z], t)
        ux2, uy2, uz2 = aa.u([x, y, z], t)
#        print 2.*eta, eta2
        self.assertTrue(round(etaa + eta,8) == round(eta2,8))
        self.assertTrue(round(uxa + ux,8) == round(ux2,8))
        self.assertTrue(round(uya + uy,8) == round(uy2,8))
        self.assertTrue(round(uza + uz,8) == round(uz2,8))
# Testing with 5 spectra
        aa= MultiSpectraRandomWaves(
            5,
            [Tp,Tp,Tp,Tp,Tp],
            [Hs,Hs,Hs,Hs,Hs],
            mwl,#m significant wave height
            depth ,           #m depth
            [waveDir,waveDir,waveDir,waveDir,waveDir],
            g,      #peak  frequency
            np.array([N,N,N,N,N]),
            [bandFactor,bandFactor,bandFactor,bandFactor,bandFactor],         #accelerationof gravity
            [spectName, spectName,spectName,spectName,spectName],
            spectral_params =[  {"gamma": gamma, "TMA": TMA,"depth": depth},  {"gamma": gamma, "TMA": TMA,"depth": depth}, {"gamma": gamma, "TMA": TMA,"depth": depth},  {"gamma": gamma, "TMA": TMA,"depth": depth}, {"gamma": gamma, "TMA": TMA,"depth": depth}  ],
            phi = [phi,phi,phi,phi,phi]# random words will result in error and return the available spectra
    )
        eta2 = aa.eta([x, y, z], t)
        ux2, uy2, uz2 = aa.u([x, y, z], t)

        self.assertTrue(round(5.*eta,8) == round(eta2,8))
        self.assertTrue(round(5.*ux,8) == round(ux2,8))
        self.assertTrue(round(5.*uy,8) == round(uy2,8))
        self.assertTrue(round(5.*uz,8) == round(uz2,8))


class CheckDirectionalWaveFailures(unittest.TestCase):
    def testFailureModes(self):
        from proteus.WaveTools import DirectionalWaves
        DirectionalWaves(20,1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),10,2.,"JONSWAP", "cos2s", spectral_params= None, spread_params = {"s":10}, phi = None, phiSymm = False  )
       # Putting a silly name
        with self.assertRaises(SystemExit) as cm1:
            DirectionalWaves(20,1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),10,2.,"JONSWAP", "aaaargh", spectral_params= None, spread_params = {"s":10}, phi = None, phiSymm = False  )
        self.assertEqual(cm1.exception.code, 1 )
       # Putting incorrect phi
        with self.assertRaises(SystemExit) as cm2:
            DirectionalWaves(20,1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),10,2.,"JONSWAP", "cos2s", spectral_params= None, spread_params = {"s":10}, phi = np.zeros(15,), phiSymm = False  )
        self.assertEqual(cm2.exception.code, 1 )
        #putting non existent parameters
        with self.assertRaises(SystemExit) as cm3:
            DirectionalWaves(20,1.,1.,0.,10.,np.array([0,0,1]),np.array([0,-9.81,0]),10,2.,"JONSWAP", "cos2s", spectral_params= None, spread_params = {"blah":10}, phi = None, phiSymm = False  )
        self.assertEqual(cm3.exception.code, 1 )



class VerifyDirectionals(unittest.TestCase):
    def testDirectional(self):
        from proteus.WaveTools import fastcos_test as fcos
        from proteus.WaveTools import coshkzd_test as fcosh
        from proteus.WaveTools import sinhkzd_test as fsinh
        from proteus.WaveTools import DirectionalWaves,Udrift
        import random
        # Assinging a random value at a field and getting the expected output
        Tp = 2.
        Hs = 0.15
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        # Setting the angle to the first quadrant otherwise the acos function further below is confused
        theta0 = 2.
        dir1 = cos(theta0)
        dir2 = sin(theta0)

        waveDir = np.array([dir1,dir2, 0])
        N = 10
        M = 10
        phi =  np.zeros((2*M+1,N),"d")
        gamma = 1.2
        TMA = True
        spectName = "JONSWAP"
        spectral_params =  {"gamma": gamma, "TMA": TMA,"depth": depth}
        bandFactor = 2.0
        spreadName = "mitsuyasu"
        spread_params = {"f0": 1./Tp, "smax": 15.}
        phiSymm = False

        aa= DirectionalWaves(
            M,
            Tp,
            Hs,
            mwl,#m significant wave height
            depth ,           #m depth
            waveDir,
            g,      #peak  frequency
            N,
            bandFactor,         #accelerationof gravity
            spectName,
            spreadName,
            spectral_params,
            spread_params,
            phi,
            phiSymm
            )
        x = 188.
        y = 134.
        z =  mwl
        t =  167.
        eta = aa.eta([x, y, z], t)
        ux, uy, uz = aa.u([x, y, z], t)

        # Testing with a specific phi array

        # setDirVector are tested above
        from proteus.WaveTools import setDirVector, dispersion, reduceToIntervals, returnRectangles3D, JONSWAP,mitsuyasu, normIntegral

        fmin = 1./(Tp*bandFactor)
        fmax = bandFactor/(Tp)
        fi = np.linspace(fmin,fmax,N)
        thetas = np.linspace(theta0 - pi/2,theta0+pi/2,2*M+1)
        dth = pi/(2*M)
        df = (fmax-fmin)/(N-1)
        ki = dispersion(2*pi*fi,depth)
        kp = dispersion(2*pi/Tp,depth)
        waveDirs = np.zeros((2*M+1,3),)
        for jj in range(2*M+1):
            waveDirs[jj,:] = np.array([cos(thetas[jj]),sin(thetas[jj]),0])
        z0 = z - mwl
        fim = reduceToIntervals(fi,df)
        thetas-=theta0
        thetas_m = reduceToIntervals(thetas,dth)
        Si_Jm = JONSWAP(fim,1./Tp,Hs,gamma,TMA, depth)
        Si_dir = mitsuyasu(thetas_m,fim,1./Tp, 15.)
        for ii in range(0,N):
            Si_dir[:,ii] = normIntegral(Si_dir[:,ii],thetas_m)
            Si_dir[:,ii]*= Si_Jm[ii]
        ai = np.sqrt(2.*returnRectangles3D(Si_dir,thetas_m,fim))

        omega = 2*pi*fi
        etaRef = 0.
        uxRef = 0.
        uyRef = 0.
        uzRef = 0.

        for ii in range(N):
            for jj in range(2*M+1):
                normDir = waveDirs[jj,:]
                etaRef+=ai[jj,ii]*fcos(ki[ii]*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-omega[ii]*t +phi[jj,ii])
                uxRef += normDir[0]*ai[jj,ii]*omega[ii] *fcosh(ki[ii],z0,depth) *fcos(ki[ii]*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-omega[ii]*t +phi[jj,ii])-normDir[0]*Udrift(ai[jj,ii],gAbs,omega[ii]/ki[ii],depth)
                uyRef += normDir[1]*ai[jj,ii]*omega[ii] *fcosh(ki[ii],z0,depth) * fcos(ki[ii]*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-omega[ii]*t +phi[jj,ii])-normDir[1]*Udrift(ai[jj,ii],gAbs,omega[ii]/ki[ii],depth)
                uzRef +=  ai[jj,ii]*omega[ii] *fsinh(ki[ii],z0,depth) * fcos(ki[ii]*(normDir[0]*x+normDir[1]*y+normDir[2]*z)-omega[ii]*t +phi[jj,ii],True)

        err = abs(eta/etaRef - 1.)
        err_x =abs(ux/uxRef - 1.)
        err_y =abs(uy/uyRef - 1.)
        err_z =abs(uz/uzRef - 1.)


        self.assertTrue(err <= 1e-8)
        self.assertTrue(err_x <= 1e-8)
        self.assertTrue(err_y <= 1e-8)
        self.assertTrue(err_z <= 1e-8)



        ab= DirectionalWaves(
            50,
            2.,
            0.1,
            mwl,
            1. ,
            np.array([1,1,0]),
            g,
            51,
            1.1,         #accelerationof gravity
            "JONSWAP",
            "cos2s",
            spectral_params=None,
            spread_params={"s":15},
            phi=None,
            phiSymm=False
            )
# For plotting a contour showing directional waves
"""
        xi = np.linspace(0,10,101)
        yi = np.linspace(0,10,101)

        eta_t = np.zeros((101,101),)
        for i in range(len(xi)):
            for j in range(len(yi)):
                eta_t[i,j] = ab.eta(xi[i],yi[j],0.,0.)


        from matplotlib import pyplot as plt
        fig = plt.figure(2)
        X,Y = np.meshgrid(xi,yi)
        CS = plt.contourf(X,Y,eta_t)
        plt.xlabel("x (m)",size=20)
        plt.ylabel("y (m)",size=20)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        CB.set_label("$\eta$ (m)",size = 20)
        plt.savefig("Contour.png")
"""



class CheckTimeSeriesFailureModes(unittest.TestCase):
    def testTimeSeriesFailureModes(self):
        from proteus.WaveTools import TimeSeries
        #load successfully - direct decomposition
        path = getpath()
        aa= TimeSeries(
            os.path.join(path, "data_timeSeries.txt"),
            0,
            np.array([1.,0,0]),
            1.  ,
            64 ,          #number of frequency bins
            1. ,
            np.array([1.,0,0]),
            np.array([0,0,-9.81])
            )

        with self.assertRaises(SystemExit) as cm1:
            aa= TimeSeries(
                os.path.join(path,"data_timeSeries.dat"),
                0,
                np.array([1.,0,0]),
                1.  ,
               64 ,          #number of frequency bins
                1. ,
                np.array([1,0,0]),
                np.array([0,0,-9.81])
            )
        with self.assertRaises(SystemExit) as cm2:
            aa= TimeSeries(
                os.path.join(path, "data_timeSeries_err1.csv"),
                0,
                np.array([1.,0,0]),
                1.  ,
                64 ,          #number of frequency bins
                1. ,
                np.array([1,0,0]),
                np.array([0,0,-9.81])
            )
        self.assertEqual(cm2.exception.code, 1 )
        with self.assertRaises(SystemExit) as cm3:
            aa= TimeSeries(
                os.path.join(path, "data_timeSeries_err2.txt"),
                0,
                np.array([1.,0,0]),
                1.  ,
                64 ,          #number of frequency bins
                1. ,
                np.array([1,0,0]),
                np.array([0,0,-9.81])
            )
        self.assertEqual(cm3.exception.code, 1 )

    def testWindTimeSeriesFailureModes(self):
        from proteus.WaveTools import TimeSeries
        #load successfully - window decomposition
        path = getpath()
        aa= TimeSeries(
            os.path.join(path, "data_timeSeries.txt"),
            0,
            np.array([1.,0,0]),
            1.  ,
            64 ,          #number of frequency bins
            1. ,
            np.array([1.,0,0]),
            np.array([0,0,-9.81]),
            0.025,
            False,
            {"Nwaves" : 5,"Tm":1, "Window":"costap"}

            )
        #load successfully - import array

        numarray  =np.zeros((10001,2),float)
        numarray[:,0] = np.linspace(0,100,10001)
        numarray[:,1] = np.cos(pi*numarray[:,0])
        aa= TimeSeries(
            os.path.join(path, "data_timeSeries.txt"),
            0,
            np.array([1.,0,0]),
            1.  ,
            64 ,          #number of frequency bins
            1. ,
            np.array([1.,0,0]),
            np.array([0,0,-9.81]),
            0.025,
            False,
            {"Nwaves" : 5,"Tm":1, "Window":"costap"},
            True,
            numarray
            )

        # Putting too many waves
        with self.assertRaises(SystemExit) as cm1:
            aa= TimeSeries(
            os.path.join(path, "data_timeSeries.txt"),
            0,
            np.array([1.,0,0]),
            1.  ,
            64 ,          #number of frequency bins
            1. ,
            np.array([1.,0,0]),
            np.array([0,0,-9.81]),
            0.025,
            False,
            {"Nwaves" : 500,"Tm":1, "Window":"costap"}

            )
        self.assertEqual(cm1.exception.code, 1 )


        #No Nwaves
        with self.assertRaises(SystemExit) as cm2:
            aa= TimeSeries(
                os.path.join(path, "data_timeSeries.txt"),
                0,
                np.array([1.,0,0]),
                1.  ,
                64 ,          #number of frequency bins
                1. ,
                np.array([1.,0,0]),
                np.array([0,0,-9.81]),
                0.025,
                False,
                {"Tm":1, "Window":"costap"}

            )

        self.assertEqual(cm2.exception.code, 1 )




        #No tm
        with self.assertRaises(SystemExit) as cm3:
            aa= TimeSeries(
                os.path.join(path, "data_timeSeries.txt"),
                0,
                np.array([1.,0,0]),
                1.  ,
                64 ,          #number of frequency bins
                1. ,
                np.array([1.,0,0]),
                np.array([0,0,-9.81]),
                0.025,
                False,
                {"Nwaves":5, "Window":"costap"}

            )

        self.assertEqual(cm3.exception.code, 1 )

        #No window
        with self.assertRaises(SystemExit) as cm3a:
            aa= TimeSeries(
                os.path.join(path,"data_timeSeries.txt"),
                0,
                np.array([1.,0,0]),
                1.  ,
                64 ,          #number of frequency bins
                1. ,
                np.array([1.,0,0]),
                np.array([0,0,-9.81]),
                0.025,
                False,
                {"Nwaves":5, "Tm":1}

            )
        self.assertEqual(cm3a.exception.code, 1 )


        #Wrong window name
        with self.assertRaises(SystemExit) as cm4:
            aa= TimeSeries(
                os.path.join(path,"data_timeSeries.txt"),
                0,
                np.array([1.,0,0]),
                1.  ,
                64 ,          #number of frequency bins
                1. ,
                np.array([1.,0,0]),
                np.array([0,0,-9.81]),
                0.025,
                False,
                {"Nwaves":5, "Tm":1, "Window":"aargh"}

            )

        self.assertEqual(cm4.exception.code, 1 )

#Wrong cut-off
        with self.assertRaises(SystemExit) as cm5:

            aa= TimeSeries(
                os.path.join(path,"data_timeSeries.txt"),
                0,
                np.array([1.,0,0]),
                1.  ,
                64 ,          #number of frequency bins
                1. ,
                np.array([1.,0,0]),
                np.array([0,0,-9.81]),
                0.025,
                False,
                {"Nwaves":5, "Tm":1, "Window":"costap", "Cutoff" : 0.8}

            )
        self.assertEqual(cm5.exception.code, 1 )


#Using not enough overlap for absorption zone
        with self.assertRaises(SystemExit) as cm6:

            aa= TimeSeries(
                os.path.join(path,"data_timeSeries.txt"),
                0,
                np.array([1.,0,0]),
                1.  ,
                4 ,          #number of frequency bins
                1. ,
                np.array([1.,0,0]),
                np.array([0,0,-9.81]),
                0.025,
                False,
                {"Nwaves":5, "Tm":1, "Window":"costap"},
                Lgen = np.array([5.,0.,0.])

            )
        self.assertEqual(cm6.exception.code, 1 )

class VerifyTimeSeries(unittest.TestCase):
    def testDirect(self):
# Testing class while reading from file
        path =getpath()
        from proteus.WaveTools import TimeSeries, costap
        import random
        aa= TimeSeries(
            os.path.join(path,"data_timeSeries.txt"),
            0,
             np.array([0.,0.,0]),
            1.  ,
            256,          #number of frequency bins
            1. ,
            np.array([1,0,0]),
            np.array([0,0,-9.81]),
            cutoffTotal=0.025
            )
        fid = open(os.path.join(path, "data_timeSeries.txt"),"r")
        data = np.loadtxt(fid)
        timeRef = data[:,0]
        etaRef = data[:,1]

        timeInt = np.linspace(timeRef[0],timeRef[-1],len(timeRef))
        etaInt = np.interp(timeInt, timeRef, etaRef)
        etaTest = np.zeros(len(timeRef),"d")
        x = 0.
        y = 0.
        z = 0.
        ii = -1
        for tt in timeInt:
            ii+=1
            etaTest[ii] = aa.eta([x, y, z], tt)

        etaInt-=np.mean(etaInt)
        etaInt*=costap(len(data),0.025)
        norm = max(etaRef)
        err = (etaInt - etaTest)**2
        err = np.sqrt(sum(err))/len(etaInt)/np.mean(abs(etaInt))
        self.assertTrue(err<1e-2 )
# Testing class while getting a timeseries from an array
        series = np.zeros( (len(timeInt),2),)
        series[:,0] = timeInt
        series[:,1] = etaInt
        aa2= TimeSeries(
            os.path.join(path, "data_timeSeries.txt"),
            0,
             np.array([0.,0.,0]),
            1.  ,
            256,          #number of frequency bins
            1. ,
            np.array([1,0,0]),
            np.array([0,0,-9.81]),
            0.025,
            True,
            None,
            True,
            series
            )
        ii = -1
        for tt in timeInt:
            ii+=1
            etaTest[ii] = aa2.eta([x, y, z], tt)
        etaInt-=np.mean(etaInt)
        etaInt*=costap(len(data),0.025)
        norm = max(etaRef)
        err = (etaInt - etaTest)**2
        err = np.sqrt(sum(err))/len(etaInt)/np.mean(abs(etaInt))
        self.assertTrue(err<1e-2 )



    def testWindow(self):
        path =getpath()
        from proteus.WaveTools import TimeSeries, costap
        import random
        aa= TimeSeries(
            os.path.join(path, "data_timeSeries.txt"),
            0,
             np.array([0.,0.,0]),
            1.  ,
            48,          #number of frequency bins
            1. ,
            np.array([1,0,0]),
            np.array([0,0,-9.81]),
            0.025,
            False,
            {"Nwaves":3, "Tm":8, "Window":"costap"}
            )
        fid = open(os.path.join(path,"data_timeSeries.txt"),"r")
        data = np.loadtxt(fid)
        timeRef = data[:,0]
        etaRef = data[:,1]

        timeInt = np.linspace(timeRef[0],timeRef[-1],len(timeRef))
        etaInt = np.interp(timeInt, timeRef, etaRef)
        etaTest = np.zeros(len(timeRef),"d")
        x = 0.
        y = 0.
        z = 0.
        ii = -1
        for tt in timeInt:
            ii+=1
            etaTest[ii] = aa.eta([x, y, z], tt)

        etaInt-=np.mean(etaInt)
        etaInt*=costap(len(data),0.025)
        norm = max(etaRef)
        err = (etaInt - etaTest)**2
        err = np.sqrt(sum(err))/len(etaInt)/np.mean(abs(etaInt))
        self.assertTrue(err<1e-2 )

# Testing class while getting a timeseries from an array
        series = np.zeros( (len(timeInt),2),)
        series[:,0] = timeInt
        series[:,1] = etaInt
        aa2= TimeSeries(
            os.path.join(path, "data_timeSeries.txt"),
            0,
             np.array([0.,0.,0]),
            1.  ,
            32,          #number of frequency bins
            1. ,
            np.array([1,0,0]),
            np.array([0,0,-9.81]),
            0.025,
            False,
            {"Nwaves":3, "Tm":8, "Window":"costap"},
            True,
            series
            )
        ii = -1
        for tt in timeInt:
            ii+=1
            etaTest[ii] = aa2.eta([x, y, z], tt)
        etaInt-=np.mean(etaInt)
        etaInt*=costap(len(data),0.025)
        norm = max(etaRef)
        err = (etaInt - etaTest)**2
        err = np.sqrt(sum(err))/len(etaInt)/np.mean(abs(etaInt))
        self.assertTrue(err<1e-2 )


class CheckRandomWavesFastFailureModes(unittest.TestCase):
    def testRandomWavesFastFailure(self):
        from proteus.WaveTools import RandomWavesFast
        with self.assertRaises(SystemExit) as cm1:
            aRF = RandomWavesFast(0,
                         100,
                         np.array([0.,0.,0.]),
                         2.,
                         0.1,
                         0.,
                         1.,
                         np.array([1.,0.,0.]),
                         np.array([0.,9.81,0.]),
                         100,
                         2.,
                         "JONSWAP",
                         None,
                         None,
                         Nfreq = 4
                              )
        self.assertEqual(cm1.exception.code, 1 )

        with self.assertRaises(SystemExit) as cm2:
            aRF = RandomWavesFast(0,
                         5,
                         np.array([0.,0.,0.]),
                         2.,
                         0.1,
                         0.,
                         1.,
                         np.array([1.,0.,0.]),
                         np.array([0.,9.81,0.]),
                         100,
                         2.,
                         "JONSWAP",
                         None,
                         None,
                         Nfreq = 32
                              )
        self.assertEqual(cm2.exception.code, 1 )
        filenames = ['RandomSeries_Hs_0.1_Tp_2.0_depth_1.0']
        remove_files(filenames)


class VerifyRandomWavesFast(unittest.TestCase):
# RandomWavesFast will be tested to the point that it gives the same answer as TimeSeriesClass
    def testRandomFast(self):
        from proteus.WaveTools import RandomWaves,TimeSeries,RandomWavesFast
        import random
        path =getpath()
        fname = "randomFastSeries.txt"      # the working directory, where remove_files looks
        # Assinging a random value at a field and getting the expected output
        Tp = 1.
        Hs = 0.15
        mwl = 0.
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.6
        dir2 = 0.8
        waveDir = np.array([dir1,dir2, 0.])
        N = 100
        Nf =32
        phi = np.linspace(1,N,N)
        spectName = "JONSWAP"
        bandFactor = 2.0
        Lgen = 1.5 * waveDir
        x0 =  np.array([2.,0.,-0.2 ])
        Tstart = 0.
        Nwaves = 15
        duration = 3.*Nwaves*Tp + 70.*Tp
        Tend = Tstart + duration
        aR= RandomWaves(Tp,
                     Hs,
                     mwl,
                     depth ,
                     waveDir,
                     g,
                     N,
                     bandFactor,
                     spectName,
                     None,
                     phi
                   )

        series = aR.writeEtaSeries(Tstart,Tend,x0,fname, 4.*Lgen)
        duration-=series[0,0]
        cutoff = max(0.2*Tp/duration , 0.1*Nwaves*Tp / duration)
        rec_d =False



        aT= TimeSeries(
            fname,
            0,
            x0,
            depth,
            Nf,          #number of frequency bins
            mwl ,
            waveDir,
            g,
            cutoff,
            rec_d,
            {"Nwaves":Nwaves, "Tm":Tp/1.1, "Window":"costap","Overlap":0.7, "Cutoff":0.1},
            True,
            series
            )


        aRF = RandomWavesFast(Tstart,
                         Tend,
                         x0,
                         Tp,
                         Hs,
                         mwl,
                         depth,
                         waveDir,
                         g,
                         N,
                         bandFactor,
                         spectName,
                         None,
                         phi,
                         Lgen,
                         Nfreq=Nf,
                         Nwaves = 15,
                              checkAcc = True)

        x = x0 + Lgen * 0.5


        eta0 = np.zeros(len(series),)
        eta1 =  np.zeros(len(series),)
        eta2 =  np.zeros(len(series),)

        t = series[0,0] + 2.*duration*cutoff +0.4*duration*(1.-4.*cutoff)

        #print "\n"
        #print aRF.printOut()
        """
        print "Number of windows = ",Nwindows
        print "Start time = ",series[0,0]
        print "End time = ",series[-1,0]
        print "Cutoff = ",cutoff
        """
        #print "\n Max error in fast approximation=%s%%" %round(100*aRF.er1,3)

        filenames = ['RandomSeries_Hs_0.15_Tp_1.0_depth_0.9',
                     'randomFastSeries.txt',]
        remove_files(filenames)

        self.assertTrue(round(abs(aRF.eta(x,t)/aT.eta(x,t)),8) == 1.)
        self.assertTrue(round(abs(aRF.u(x,t)[0]/aT.u(x,t)[0]),8) == 1.)
        self.assertTrue(round(abs(aRF.u(x,t)[1]/aT.u(x,t)[1]),8) == 1.)
        self.assertTrue(round(abs(aRF.u(x,t)[2]/aT.u(x,t)[2]),8) == 1.)

        """
        from matplotlib import pyplot as plt
        for ii in range(len(series)):
            tt = series[ii,0]
            eta0[ii] = aR.eta(x,tt)
            eta1[ii] = aT.eta(x,tt)
            eta2[ii] = aRF.eta(x,tt)
        import pylab as plt
        plt.plot(series[:,0],series[:,1],"ko")
        plt.plot(series[:,0],eta0,"k-")
#        plt.plot(series[:,0],eta1,"b--")
        plt.plot(series[:,0],eta2,"r-.")

        plt.xlim(t-5.,t+5)
        plt.grid()
        plt.savefig("t.pdf")
        """



class CheckFailureRandomNLWaves(unittest.TestCase):
    def testFailures(self):
        waveDir = np.array([0.,0.,1.])
        Lgen = np.array([0.,0.,-1])
        from proteus.WaveTools import RandomNLWaves
        RR = RandomNLWaves(0,100,1.,1.,0.,10.,waveDir,np.array([0,-9.81,0]),100,2.,"JONSWAP", spectral_params= None )
        xi = np.array([0.,0.,0.])
        t = 0.
#        print RR.writeEtaSeries(0.,100,1,xi,"aa.txt","blah")
#Failure 1:  call eta
        with self.assertRaises(SystemExit) as cm1:
            f = RR.eta(xi,t)
        self.assertEqual(cm1.exception.code, 1 )
#Failure 2:  call u
        with self.assertRaises(SystemExit) as cm2:
            f = RR.eta(xi,t)
        self.assertEqual(cm2.exception.code, 1 )
#Failure 3:  call writeEtaSeries with a wrong mode
        with self.assertRaises(SystemExit) as cm3:
            f = RR.writeEtaSeries(0.,100,1,xi,"aa.txt","blah")
        self.assertEqual(cm3.exception.code, 1 )
        with self.assertRaises(SystemExit) as cm4:
            f = RR.writeEtaSeries(0.,100,1,xi,"aa.txt","long",False,Lgen)
        self.assertEqual(cm4.exception.code, 1 )




class VerifyRandomNLWaves(unittest.TestCase):
    def testFunctions(self):
        from proteus.WaveTools import RandomWaves,TimeSeries,RandomNLWaves,eta_mode
        path =getpath()
        fname = "randomSeries.txt"      # the working directory, where remove_files looks
        # Assinging a random value at a field and getting the expected output
        Tp = 1.
        Hs = 0.1
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.7
        dir2 = 0.5
        waveDir = np.array([dir1,dir2, 0])
        N = 20
        phi = np.linspace(0,N,N)
        spectName = "JONSWAP"
        bandFactor = 2.0
        Lgen = 5 * waveDir
        x0 =  np.array([2.,0.,0 ])
        Tstart = 0.
        Tend = 150.


        aR = RandomWaves(
            Tp,                      #wave period
            Hs,                      #significant wave height
            mwl,                     #mean water level
            depth,                   #water depth
            waveDir,                 #wave direction vector with three components
            g,                       #gravitational accelaration vector with three components
            N,                       #number of frequency bins
            bandFactor,              #width factor for band around peak frequency fp
            spectName,               #random words will result in error and return the available spectra
            spectral_params=None,    #JONPARAMS = {"gamma": 3.3, "TMA":True,"depth": depth}
            phi=phi                 #array of component phases
            )


        aNL = RandomNLWaves(
            Tstart,
            Tend,
            Tp,                      #wave period
            Hs,                      #significant wave height
            mwl,                     #mean water level
            depth,                   #water depth
            waveDir,                 #wave direction vector with three components
            g,                       #gravitational accelaration vector with three components
            N,                       #number of frequency bins
            bandFactor,              #width factor for band around peak frequency fp
            spectName,               #random words will result in error and return the available spectra
            spectral_params=None,    #JONPARAMS = {"gamma": 3.3, "TMA":True,"depth": depth}
            phi = aR.phi               #array of component phases
            )
        from proteus.WaveTools import fastcos_test as fcos

        x = 150.
        y = 135.
        z =  mwl
        t =  120.
        xi = np.array([x, y, z])
#        print aR.eta(xi,t),aNL.eta(xi,t)
        self.assertTrue(round(aR.eta(xi,t),8) == round(aNL.eta_linear(xi,t),8))

        etaT = 0.
        for ii in range(N):
            kh = aR.ki[ii]*aR.depth
            ai = 0.25 * aR.ai[ii]**2 * aR.ki[ii] / tanh(kh) * (2. + 3./(sinh(kh)**2))
            etaT += eta_mode(xi,t,2.*aR.kDir[ii],2.*aR.omega[ii],2.*aR.phi[ii],ai)
        # 2nd order testing
#        print etaT,aNL.eta_2ndOrder(xi,t)
        self.assertTrue(round(etaT/aNL.eta_2ndOrder(xi,t),2)==1)

        ww = aR.omega
        ki = aR.ki

# Testing higher harmonics
        etaT = 0.
        N = aR.N
        for ii in range(0,N-1):
            for jj in range(ii+1,N):
                w1p2 = ww[ii] + ww[jj]
                w1p2_sq = ww[ii]**2 + ww[jj]**2
                k1p2 = abs(ki[ii] + ki[jj])
                w1b2 = ww[ii]* ww[jj]
                kh12 = k1p2 * aR.depth
                k1h = ki[ii] * aR.depth
                k2h = ki[jj] * aR.depth
                Dp = (w1p2)**2  - aR.gAbs*k1p2*tanh(kh12)
                Bp =  w1p2_sq
                Bp = Bp - w1b2*( 1. - 1./(tanh(k1h)*tanh(k2h))) * (w1p2**2 + aR.gAbs * k1p2  *tanh(kh12)) / Dp
                Bp += w1p2*( ww[ii]**3/sinh(k1h)**2 + ww[jj]**3/sinh(k2h)**2)/Dp
                Bp =0.5* Bp / aR.gAbs



                ai = aR.ai[ii]*aR.ai[jj]*Bp
                etaT += eta_mode(xi,t,aR.kDir[ii] + aR.kDir[jj],w1p2,aR.phi[ii] + aR.phi[jj],ai)
#        print etaT,aNL.eta_short(xi,t)
        self.assertTrue(round(etaT/aNL.eta_short(xi,t),2)==1 )
# Testing lower harmonics
        etaT = 0.
        N = aR.N
        for ii in range(0,N-1):
            for jj in range(ii+1,N):
                w1p2 = ww[ii] - ww[jj]
                w1p2_sq = ww[ii]**2 + ww[jj]**2
                k1p2 = abs(ki[ii] - ki[jj])
                w1b2 = ww[ii]* ww[jj]
                kh12 = k1p2 * aR.depth
                k1h = ki[ii] * aR.depth
                k2h = ki[jj] * aR.depth
                Dp = (w1p2)**2  - aR.gAbs*k1p2*tanh(kh12)
                Bp =  w1p2_sq
                Bp = Bp + w1b2*( 1. + 1./(tanh(k1h)*tanh(k2h))) * (w1p2**2 + aR.gAbs * k1p2  *tanh(kh12)) / Dp
                Bp += w1p2*( ww[ii]**3/sinh(k1h)**2 - ww[jj]**3/sinh(k2h)**2)/Dp
                Bp =0.5* Bp / aR.gAbs



                ai = aR.ai[ii]*aR.ai[jj]*Bp
                etaT += eta_mode(xi,t,aR.kDir[ii] - aR.kDir[jj],w1p2,aR.phi[ii] - aR.phi[jj],ai)
#        print etaT,aNL.eta_long(xi,t)
        self.assertTrue(round(etaT/aNL.eta_long(xi,t),2)==1 )

# Testing setup
        etaT = 0.
        N = aR.N
        for ii in range(0,N-1):
            setup =  0.5*aR.ai[ii]*aR.ai[ii]*aR.ki[ii]/sinh(2.*ki[ii]*aR.depth)
            etaT += setup

        self.assertTrue(round(etaT,8) == round(aNL.eta_setUp(xi,t),8))
        etaT =aNL.eta_linear(xi,t)+aNL.eta_2ndOrder(xi,t)+aNL.eta_short(xi,t)+aNL.eta_long(xi,t)
        self.assertTrue(round(etaT,8) == round(aNL.eta_overall(xi,t),8))
        etaT= etaT - aNL.eta_setUp(xi,t)
        self.assertTrue(round(etaT,8) == round(aNL.eta_overall(xi,t,True),8))

# Test writing series for different modes
        Tstart = 0
        Tend = 2.
        dt = 1.
        fname = "2ndorderseries.txt"

        series = aNL.writeEtaSeries(Tstart,Tend,dt,xi,fname,"all",False)
        fid = open(fname,"r")
        seriesFile = np.loadtxt(fid)
        fid.close()
        filenames = ['2ndorderseries.txt']
        remove_files(filenames)


        for ii in range(3):
            self.assertTrue(round(series[ii,1],8) ==     round(aNL.eta_overall(xi,float(ii)),8) )
            self.assertTrue( round(seriesFile[ii,1],8) == round(aNL.eta_overall(xi,float(ii)),8) )



        series = aNL.writeEtaSeries(Tstart,Tend,dt,xi,fname,"all",True)
        fid = open(fname,"r")
        seriesFile = np.loadtxt(fid)
        fid.close()
        filenames = ['2ndorderseries.txt']
        remove_files(filenames)

        for ii in range(3):
            self.assertTrue(round(series[ii,1],8) ==     round(aNL.eta_overall(xi,float(ii),True),8) )
            self.assertTrue( round(seriesFile[ii,1],8) == round(aNL.eta_overall(xi,float(ii),True),8) )

        series = aNL.writeEtaSeries(Tstart,Tend,dt,xi,fname,"linear")
        fid = open(fname,"r")
        seriesFile = np.loadtxt(fid)
        fid.close()
        filenames = ['2ndorderseries.txt']
        remove_files(filenames)

        for ii in range(3):
            self.assertTrue(round(series[ii,1],8) ==     round(aNL.eta_linear(xi,float(ii)),8) )
            self.assertTrue( round(seriesFile[ii,1],8) == round(aNL.eta_linear(xi,float(ii)),8) )

        series = aNL.writeEtaSeries(Tstart,Tend,dt,xi,fname,"short")
        fid = open(fname,"r")
        seriesFile = np.loadtxt(fid)
        fid.close()
        filenames = ['2ndorderseries.txt']
        remove_files(filenames)

        for ii in range(3):
            self.assertTrue(round(series[ii,1],8) ==     round(aNL.eta_short(xi,float(ii))+aNL.eta_2ndOrder(xi,float(ii)),8) )
            self.assertTrue( round(seriesFile[ii,1],8) == round(aNL.eta_short(xi,float(ii))+aNL.eta_2ndOrder(xi,float(ii)),8) )



        series = aNL.writeEtaSeries(Tstart,Tend,dt,xi,fname,"long")
        fid = open(fname,"r")
        seriesFile = np.loadtxt(fid)
        fid.close()
        filenames = ['2ndorderseries.txt']
        remove_files(filenames)

        for ii in range(3):
            self.assertTrue(round(series[ii,1],8) ==     round(aNL.eta_long(xi,float(ii)),8) )
            self.assertTrue( round(seriesFile[ii,1],8) == round(aNL.eta_long(xi,float(ii)),8) )

        series = aNL.writeEtaSeries(Tstart,Tend,dt,xi,fname,"setup")
        fid = open(fname,"r")
        seriesFile = np.loadtxt(fid)
        fid.close()
        filenames = ['2ndorderseries.txt']
        remove_files(filenames)

        for ii in range(3):
            self.assertTrue(round(series[ii,1],8) ==     round(aNL.eta_setUp(xi,float(ii)),8) )
            self.assertTrue( round(seriesFile[ii,1],8) == round(aNL.eta_setUp(xi,float(ii)),8) )

class VerifyRandomNLWavesFast(unittest.TestCase):
# RandomWavesFast will be tested to the point that it gives the same answer as TimeSeriesClass
    def testRandomNLFast(self):
        from proteus.WaveTools import RandomNLWaves,RandomNLWavesFast,TimeSeries
        import random
        Tp = 1.
        Hs = 0.1
        mwl = 0.47

        depth = 0.5
        g = np.array([0., -9.81, 0])
        N = 2 # Let's see how it copes with a bimodal sea state
        bandFactor = 1.1
        spectName = "JONSWAP"
        spectral_params = None
        phi = np.linspace(1,N,N)
        waveDir = np.array([1., 0., 0.])


        nperiod = 50
        npoints = 25
        n = npoints * nperiod
        tnlist=np.linspace(0,nperiod*Tp,n)
        x0 = np.array([2., 0., 0.])
        Lgen = np.array([5., 0., 0.])
        Tstart = tnlist[0]
        Tend = tnlist[-1]
        NLongW = 10.
        fname ="RNLWaves.txt"

        aR = RandomNLWaves(tnlist[0],
                            tnlist[-1],
                              Tp,
                              Hs,
                              mwl,#m significant wave height
                              depth , #m depth
                              waveDir,
                              g, #peak frequency
                              N,
                              bandFactor, #accelerationof gravity
                              spectName ,# random words will result in error and return the available spectra
                              spectral_params, #JONPARAMS = {"gamma": 3.3, "TMA":True,"depth": depth}
                              phi)
        aRF= RandomNLWavesFast(
            Tstart,
            Tend,
            x0,
            Tp,
            Hs,
            mwl,
            depth,
            waveDir,
            g,
            N,
            bandFactor,
            spectName,
            None,
            phi,
            Lgen,
            NLongW=NLongW)

        Tm = Tp/1.1
        Ts = Tm/2.
        Tmax = NLongW*Tm

        dt_s = Ts/50.
        dt =  Tm/50.
        dt_l = Tmax/50.

        series = aR.writeEtaSeries(Tstart,Tend,dt,x0,fname,"linear",False,Lgen)
        series_l = aR.writeEtaSeries(Tstart,Tend,dt_l,x0,fname,"long",False,Lgen)
        series_s = aR.writeEtaSeries(Tstart,Tend,dt_s,x0,fname,"short",False ,Lgen)

        filenames = ['RNLWaves.txt']
        append = ['_linear.csv','_long.csv','_short.csv']
        filenames.extend(['randomNLWaves'+end for end in append])
        remove_files(filenames)

        Tstart = series_s[0,0]
        Tend = series_s[-1,0]
        cutoff = 0.2*Tp/(Tend-Tstart)




        Nw = int((Tend-Tstart)/Ts)
        Nw1 = min(15,Nw)
        Nw = int(Nw/Nw1)

        if Nw < 3:
            rec_d = True
        else:
            rec_d = False

        aT_s= TimeSeries(
            fname,
            0,
            x0,
            depth,
            32,          #number of frequency bins
            mwl ,
            waveDir,
            g,
            cutoff,
            rec_d,
            {"Nwaves":15, "Tm":Ts, "Window":"costap"},
            True,
            series_s
            )
        Tstart = series[0,0]
        Tend = series[-1,0]
        cutoff = 0.2*Ts/(Tend-Tstart)

        Nw = int((Tend-Tstart)/Tm)
        Nw1 = min(15,Nw)
        Nw = int(Nw/Nw1)

        if Nw < 3:
            rec_d = True
        else:
            rec_d = False

        aT= TimeSeries(
            fname,
            0,
            x0,
            depth,
            32,          #number of frequency bins
            mwl ,
            waveDir,
            g,
            cutoff,
            rec_d,
            {"Nwaves":15, "Tm":Tm, "Window":"costap"},
            True,
            series
            )
        Tstart = series_l[0,0]
        Tend = series_l[-1,0]
        cutoff = 0.2*Tmax/(Tend-Tstart)

        Nw = int((Tend-Tstart)/Tmax)
        Nw1 = min(15,Nw)
        Nw = int(Nw/Nw1)
        if Nw < 3:
            rec_d = True
        else:
            rec_d = False

        aT_l= TimeSeries(
            fname,
            0,
            x0,
            depth,
            32,          #number of frequency bins
            mwl ,
            waveDir,
            g,
            cutoff,
            rec_d,
            {"Nwaves":15, "Tm":Tmax, "Window":"costap"},
            True,
            series_l
            )
        #print cutoff,aRF.eta(x0,50.)[8]#, aT_s.eta(x,t)+aT.eta(x,t)#+aT_l.eta(x,t)

#Checking consistency with RandomNLWaves class
        sumerr = 0
        sumabs = 0

        for aa in range(len(series)):
            Tcut =  0.2*Tp
            if (series[aa,0] > Tcut) and (series[aa,0] < series[-1,0] - Tcut):
                sumerr += (aR.eta_linear(x0,series[aa,0]) - aT.eta(x0,series[aa,0]))**2
                sumabs += abs(aR.eta_linear(x0,series[aa,0]))

        err = np.sqrt(sumerr)/len(series)
        err = err/(sumabs/len(series))
        self.assertTrue(err < 0.005)
#        print err



        for aa in range(len(series_s)):
            Tcut =  0.2*Tp
            if (series_s[aa,0] > Tcut) and (series_s[aa,0] < series_s[-1,0] - Tcut):
                sumerr += (aR.eta_short(x0,series_s[aa,0])+aR.eta_2ndOrder(x0,series_s[aa,0]) - aT_s.eta(x0,series_s[aa,0]))**2
                sumabs += abs( aR.eta_short(x0,series_s[aa,0])+ aR.eta_2ndOrder(x0,series_s[aa,0]) )
        err = np.sqrt(sumerr)/len(series_s)
        err = err/(sumabs/len(series_s))
        self.assertTrue(err < 0.005)
#        print err
        for aa in range(len(series_l)):
            Tcut =  0.2*Tp
            if (series_l[aa,0] > Tcut) and (series_l[aa,0] < series_l[-1,0] - Tcut):
                sumerr += (aR.eta_long(x0,series_l[aa,0]) - aT_l.eta(x0,series_l[aa,0]))**2
                sumabs += abs(aR.eta_linear(x0,series_l[aa,0]))
        err = np.sqrt(sumerr)/len(series_l)
        err = err/(sumabs/len(series_l))

        self.assertTrue(err < 0.005)
#        print err


#Cjecking consistency of the timeSeriesClass
        x = x0 + Lgen * 0.3
        t = Tend/2.


        self.assertTrue( round(aRF.eta(x,t) == aT_s.eta(x,t)+aT.eta(x,t)+aT_l.eta(x,t),8) )
        self.assertTrue( aRF.u(x,t).all() == (aT_s.u(x,t)+aT.u(x,t)+aT_l.u(x,t) ).all())

class VerifyCombinedWaves(unittest.TestCase):
    def testFailureCombinedWaves(self):
        from proteus.WaveTools import MonochromaticWaves
        from proteus.WaveTools import CombineWaves
        period = 2.
        waveHeight = 1.
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.6
        dir2 = 0.5
        waveDir = np.array([dir1,dir2, 0])
        phi0 = 0.
        a = MonochromaticWaves(period,
                               waveHeight,
                               mwl,
                               depth,
                               g,
                               waveDir)

        with self.assertRaises(SystemExit) as cm:
            waveList = [period,a]
            CombineWaves(waveList)
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm1:
            waveList = [a,period]
            CombineWaves(waveList)
        self.assertEqual(cm1.exception.code, 1)
    def testClassCombinedWaves(self):
        from proteus.WaveTools import MonochromaticWaves
        from proteus.WaveTools import RandomWaves
        from proteus.WaveTools import CombineWaves
        period = 2.
        waveHeight = 1.
        mwl = 4.5
        depth = 0.9
        g = np.array([0,0,-9.81])
        gAbs = 9.81
        dir1 = 0.6
        dir2 = 0.5
        waveDir = np.array([dir1,dir2, 0])
        phi0 = 0.
        MW = MonochromaticWaves(period,
                                waveHeight,
                                mwl,
                                depth,
                                g,
                                waveDir)
        RW = RandomWaves(period,
                         waveHeight,
                         mwl,
                         depth,
                         waveDir,
                         g,
                         50,
                         2,
                         "JONSWAP")

        waveList = [MW,RW]
        CW = CombineWaves(waveList)
        x = np.zeros(3,)
        t = 2.1
        CW_test = RW.eta(x,t)+MW.eta(x,t)
        npt.assert_equal(CW.eta(x,t),CW_test)
        CW_test = RW.u(x,t)+MW.u(x,t)
        npt.assert_equal(CW.u(x,t),CW_test)


if __name__ == '__main__':
    unittest.main(verbosity=2)
