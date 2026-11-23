import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
import math
import numpy.linalg as linalg


class FEMTools(object):
    def __init__(self,
                 L=1.0,
                 nElements=10,
                 quadOrder=3,
                 EI=1.0e3,
                 GJ=1.0e3,
                 nlTol=1.0e-6,
                 useSparse=False,
                 beamLocation=(0.5, 0.5)):
        self.L = L
        self.nElements = nElements
        self.quadOrder = quadOrder
        self.EI = EI
        self.GJ = GJ
        self.nlTol = nlTol
        self.useSparse = useSparse
        self.beamLocation = beamLocation

    def structuredMesh(self):
        self.nNodes = 2 * self.nElements + 1
        self.nDOF = 3 * self.nNodes
        self.mesh = np.linspace(0.0, self.L, self.nElements + 1)
        self.h = self.L / float(self.nElements) * np.ones(self.nElements)
        return self.mesh, self.h

    def initializePhi(self):
        self.Phi = np.zeros(self.nDOF)
        # self.Phi[3::3]=-math.pi/3.0
        # for i in range(self.nNodes):
        #    self.Phi[3*i+2]=math.pi/4.0*float(i)/float(self.nNodes)
        # self.Phi[4::3]=0.0
        # self.Phi[5::3]=-math.pi/3.0
        self.g = np.zeros(self.nDOF)

    def GaussQuad(self):
        if self.quadOrder == 2:
            self.w = (1.0, 1.0)
            self.zeta = (-1.0/3.0**.5, 1.0/3.0**0.5)
            #self.quadSpacing = (self.zeta[0]+1.0, self.zeta[1]-self.zeta[0], 1.0-self.zeta[1])
        elif self.quadOrder == 3:
            self.w = (5.0/9.0, 8.0/9.0, 5.0/9.0)
            self.zeta = (-(3.0/5.0)**.5, 0.0, (3.0/5.0)**0.5)
            #self.quadSpacing = (self.zeta[0]+1.0, self.zeta[1]-self.zeta[0], self.zeta[2]-self.zeta[1],1.0-self.zeta[2])

    def initializeCoords(self):
        self.x = np.zeros(self.nElements + 1)
        self.y = np.zeros(self.nElements + 1)
        self.z = np.linspace(0.0, self.L, self.nElements + 1)
        # return self.x, self.y, self.z

    def basisFunctions(self):
        if self.quadOrder == 2:
            v1 = 1.0 / 6.0 * np.array([1.0 + 3.0**.5, 4.0, 1.0 - 3.0**.5])
            v2 = 1.0 / 6.0 * np.array([1.0 - 3.0**.5, 4.0, 1.0 + 3.0**.5])
            dv1 = 1.0 / 6.0 * np.array([-2.0 * 3.0**.5 - 3.0, 4.0 * 3.0**.5, -2.0 * 3.0**0.5 + 3.0])
            dv2 = 1.0 / 6.0 * np.array([2.0 * 3.0**.5 - 3.0, -4.0 * 3.0**.5, 2.0 * 3.0**0.5 + 3.0])
            self.v = [v1, v2]
            self.dv = [dv1, dv2]

            self.vv = [np.outer(self.v[0], self.v[0]), np.outer(self.v[1], self.v[1])]
            self.dvdv = [np.outer(self.dv[0], self.dv[0]), np.outer(self.dv[1], self.dv[1])]
            self.vdv = [np.outer(self.v[0], self.dv[0]), np.outer(self.v[1], self.dv[1])]
        elif self.quadOrder == 3:
            self.v = [np.array([0.6872983345e0, 0.4000000001e0, -0.8729833465e-1]), np.array([0.0, 0.10e1, 0.0]),
                      np.array([-0.8729833465e-1, 0.4000000001e0, 0.6872983345e0])]
            self.dv = [np.array([-0.1274596669e1, 0.1549193338e1, -0.2745966692e0]), np.array([-0.50e0, 0.0, 0.50e0]),
                       np.array([0.2745966692e0, -0.1549193338e1, 0.1274596669e1])]
            self.vv = [np.outer(self.v[0], self.v[0]), np.outer(self.v[1], self.v[1]), np.outer(self.v[2], self.v[2])]
            self.dvdv = [np.outer(self.dv[0], self.dv[0]), np.outer(self.dv[1], self.dv[1]), np.outer(self.dv[2], self.dv[2])]
            self.vdv = [np.outer(self.v[0], self.dv[0]), np.outer(self.v[1], self.dv[1]), np.outer(self.v[2], self.dv[2])]

    def updateCoords(self):
        #import pdb
        # pdb.set_trace()
        arcLength = 0.0
        self.x[0] = 0.0
        self.y[0] = 0.0
        for i in range(self.nElements):
            theta_el = np.array([self.Phi[6 * i], self.Phi[6 * i + 3], self.Phi[6 * (i + 1)]])
            psi_el = np.array([self.Phi[6 * i + 1], self.Phi[6 * i + 4], self.Phi[6 * i + 7]])
            phi_el = np.array([self.Phi[6 * i + 2], self.Phi[6 * i + 5], self.Phi[6 * i + 8]])
            # self.x[i+1] = self.x[i]
            # self.y[i+1] = self.y[i]
            # self.z[i+1] = self.z[i]
            xval = 0.0
            yval = 0.0
            zval = 0.0
            for j in range(self.quadOrder):
                theta = np.dot(self.v[j], theta_el)
                phi = np.dot(self.v[j], phi_el)
                xval += self.w[j] * math.cos(theta) * math.sin(phi)
                yval += -self.w[j] * math.sin(theta)  # math.sin(theta)*math.sin(psi)
                zval += self.w[j] * math.cos(theta) * math.cos(phi)
            self.x[i + 1] = self.x[i] + 0.5 * self.h[i] * xval
            self.y[i + 1] = self.y[i] + 0.5 * self.h[i] * yval
            self.z[i + 1] = self.z[i] + 0.5 * self.h[i] * zval
            arcLength += ((self.x[i + 1] - self.x[i])**2 + (self.y[i + 1] - self.y[i])**2 + (self.z[i + 1] - self.z[i])**2)**0.5

        # print arcLength, self.x[-1]
        for i in range(self.nElements + 1):
            self.x[i] += self.beamLocation[0]
            self.y[i] += self.beamLocation[1]
        return self.x[:], self.y[:], self.z[:]

    def updateLoads(self, q1, q2, q3):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def updateQs(self, endLoad, scale):
        self.Q1 = np.zeros(self.nNodes)
        self.Q2 = np.zeros(self.nNodes)
        self.Q3 = np.zeros(self.nNodes)

        count1 = endLoad[0]
        count2 = endLoad[1]
        count3 = endLoad[2]
        for i in range(self.nElements, 0, -1):
            self.Q1[i * 2] = count1
            self.Q2[i * 2] = count2
            self.Q3[i * 2] = count3
            for j in range(self.quadOrder):
                count1 += 0.5 * self.h[i - 1] * self.w[j] * self.q1[i - 1, j]
                count2 += 0.5 * self.h[i - 1] * self.w[j] * self.q2[i - 1, j]
                count3 += 0.5 * self.h[i - 1] * self.w[j] * self.q3[i - 1, j]
            self.Q1[i * 2 - 1] = 0.5 * (self.Q1[i * 2] + count1)
            self.Q2[i * 2 - 1] = 0.5 * (self.Q2[i * 2] + count2)
            self.Q3[i * 2 - 1] = 0.5 * (self.Q3[i * 2] + count3)
        self.Q1[0] = count1
        self.Q2[0] = count2
        self.Q3[0] = count3
        # print self.Q1
        # print self.Q2
        self.Q1 *= -1.0 * scale
        self.Q2 *= -1.0 * scale
        self.Q3 *= -1.0 * scale

    def calculateGradient_Hessian(self):
        self.g = np.zeros(self.nDOF)
        self.K = np.zeros((self.nDOF, self.nDOF))
        for i in range(self.nElements):
            theta_el = np.array([self.Phi[6 * i], self.Phi[6 * i + 3], self.Phi[6 * (i + 1)]])
            psi_el = np.array([self.Phi[6 * i + 1], self.Phi[6 * i + 4], self.Phi[6 * i + 7]])
            phi_el = np.array([self.Phi[6 * i + 2], self.Phi[6 * i + 5], self.Phi[6 * i + 8]])
            Q1_el = np.array([self.Q1[2 * i], self.Q1[2 * i + 1], self.Q1[2 * i + 2]])
            Q2_el = np.array([self.Q2[2 * i], self.Q2[2 * i + 1], self.Q2[2 * i + 2]])
            Q3_el = np.array([self.Q3[2 * i], self.Q3[2 * i + 1], self.Q3[2 * i + 2]])

            gstheta = np.zeros(3)
            gspsi = np.zeros(3)
            gsphi = np.zeros(3)
            gltheta = np.zeros(3)
            glpsi = np.zeros(3)
            glphi = np.zeros(3)
            Ksthetatheta = np.zeros((3, 3))
            Ksthetapsi = np.zeros((3, 3))
            Ksthetaphi = np.zeros((3, 3))
            Kspsipsi = np.zeros((3, 3))
            Kspsiphi = np.zeros((3, 3))
            Ksphiphi = np.zeros((3, 3))
            Klthetatheta = np.zeros((3, 3))
            Klthetapsi = np.zeros((3, 3))
            Klthetaphi = np.zeros((3, 3))
            Klpsipsi = np.zeros((3, 3))
            Klpsiphi = np.zeros((3, 3))
            Klphiphi = np.zeros((3, 3))

            for j in range(self.quadOrder):
                theta = np.dot(self.v[j], theta_el)
                psi = np.dot(self.v[j], psi_el)
                phi = np.dot(self.v[j], phi_el)
                thetad = np.dot(self.dv[j], theta_el)
                psid = np.dot(self.dv[j], psi_el)
                phid = np.dot(self.dv[j], phi_el)
                Q1 = np.dot(self.v[j], Q1_el)
                Q2 = np.dot(self.v[j], Q2_el)
                Q3 = np.dot(self.v[j], Q3_el)

                st = math.sin(theta)
                ct = math.cos(theta)
                if abs(ct) < 1.0e-6:
                    ct = math.copysign(1.0e-6, ct)

                # self.w[j]*(self.EI*thetad*self.dv[j] + ((self.EI-self.GJ)*psid*psid*st*ct -self.GJ*psid*phid*st)*self.v[j])
                gstheta += self.w[j] * (self.EI * thetad * self.dv[j] + ((self.GJ - self.EI) * phid**2 * st * ct - self.GJ * psid * phid * ct) * self.v[j])
                # self.w[j]*(((self.EI*st*st+self.GJ*(1.0+ct*ct))*psid +self.GJ*phid*ct)*self.dv[j])
                gspsi += self.GJ * self.w[j] * (psid - phid * st) * self.dv[j]
                gsphi += self.w[j] * (self.EI * phid * ct * ct + self.GJ * (phid * st * st - psid * st)) * self.dv[j]  # self.w[j]*(self.GJ*psid*ct*self.dv[j])

                # self.w[j]*((Q1*ct*math.cos(psi) + Q2*ct*math.sin(psi) - Q3*st)*self.v[j])
                gltheta += self.w[j] * (-Q1 * st * math.sin(phi) - Q2 * ct - Q3 * st * math.cos(phi)) * self.v[j]
                #glpsi += self.w[j]*((-Q1*st*math.sin(psi) + Q2*st*math.cos(psi))*self.v[j])
                glphi += self.w[j] * (Q1 * ct * math.cos(phi) - Q3 * ct * math.sin(phi)) * self.v[j]

                # self.w[j]*(self.EI*self.dvdv[j] +((self.EI-self.GJ)*(2.0*ct*ct-1.0)-self.GJ*psid*phid*ct)*self.vv[j])
                Ksthetatheta += self.w[j] * (self.EI * self.dvdv[j] + (self.GJ - self.EI) * (2.0 * ct * ct - 1.0)
                                             * phid * phid * self.vv[j] + self.GJ * psid * phid * st * self.vv[j])
                Ksthetapsi += -self.GJ * self.w[j] * phid * ct * self.vdv[j]  # self.w[j]*((2.0*(self.EI-self.GJ)*phid*st*ct -self.GJ*phid*st)*self.vdv[j])
                Ksthetaphi += self.w[j] * (2.0 * (self.GJ - self.EI) * phid * st * ct - self.GJ * psid * ct) * \
                    self.vdv[j]  # self.w[j]*(-self.GJ*phid*st*self.vdv[j])
                Kspsipsi += self.GJ * self.w[j] * self.dvdv[j]  # self.w[j]*((self.EI*st*st + self.GJ*(1.0+ct*ct))*self.dvdv[j])
                Kspsiphi += -self.GJ * self.w[j] * st * self.dvdv[j]  # self.w[j]*(self.GJ*ct*self.dvdv[j])
                Ksphiphi += self.w[j] * (self.EI * ct * ct + self.GJ * st * st) * self.dvdv[j]

                Klthetatheta += self.w[j] * (-Q1 * ct * math.sin(phi) + Q2 * st - Q3 * ct * math.cos(phi)) * \
                    self.vv[j]  # self.w[j]*((-Q1*st*math.cos(psi) -Q2*st*math.sin(psi) -Q3*ct)*self.vv[j])
                #Klthetapsi += self.w[j]*((-Q1*ct*math.sin(psi) + Q2*ct*math.cos(psi))*self.vv[j])
                #Klpsipsi += self.w[j]*((-Q1*st*math.cos(psi) - Q2*st*math.sin(psi))*self.vv[j])
                Klthetaphi += self.w[j] * (-Q1 * st * math.cos(phi) + Q3 * st * math.sin(phi)) * self.vv[j]
                Klphiphi += self.w[j] * (-Q1 * ct * math.sin(phi) - Q3 * ct * math.cos(phi)) * self.vv[j]

            # import pdb
            # pdb.set_trace()

            self.K[i * 6:i * 6 + 9:3, i * 6:i * 6 + 9:3] += 2.0 / self.h[i] * Ksthetatheta + 0.5 * self.h[i] * Klthetatheta
            self.K[i * 6:i * 6 + 9:3, i * 6 + 1:i * 6 + 9:3] += 2.0 / self.h[i] * Ksthetapsi + 0.5 * self.h[i] * Klthetapsi
            self.K[i * 6 + 1:i * 6 + 9:3, i * 6:i * 6 + 9:3] += np.transpose(2.0 / self.h[i] * Ksthetapsi + 0.5 * self.h[i] * Klthetapsi)
            self.K[i * 6:i * 6 + 9:3, i * 6 + 2:i * 6 + 9:3] += 2.0 / self.h[i] * Ksthetaphi + 0.5 * self.h[i] * Klthetaphi
            self.K[i * 6 + 2:i * 6 + 9:3, i * 6:i * 6 + 9:3] += np.transpose(2.0 / self.h[i] * Ksthetaphi + 0.5 * self.h[i] * Klthetaphi)
            self.K[i * 6 + 1:i * 6 + 9:3, i * 6 + 1:i * 6 + 9:3] += 2.0 / self.h[i] * Kspsipsi + 0.5 * self.h[i] * Klpsipsi
            self.K[i * 6 + 1:i * 6 + 9:3, i * 6 + 2:i * 6 + 9:3] += 2.0 / self.h[i] * Kspsiphi + 0.5 * self.h[i] * Klpsiphi
            self.K[i * 6 + 2:i * 6 + 9:3, i * 6 + 1:i * 6 + 9:3] += np.transpose(2.0 / self.h[i] * Kspsiphi + 0.5 * self.h[i] * Klpsiphi)
            self.K[i * 6 + 2:i * 6 + 9:3, i * 6 + 2:i * 6 + 9:3] += 2.0 / self.h[i] * Ksphiphi + 0.5 * self.h[i] * Klphiphi

            self.g[i * 6:i * 6 + 9:3] += 2.0 / self.h[i] * gstheta + 0.5 * self.h[i] * gltheta
            self.g[i * 6 + 1:i * 6 + 9:3] += 2.0 / self.h[i] * gspsi + 0.5 * self.h[i] * glpsi
            self.g[i * 6 + 2:i * 6 + 9:3] += 2.0 / self.h[i] * gsphi + 0.5 * self.h[i] * glphi
        #import pdb
        # pdb.set_trace()

    def calculateResidual(self):
        # import numpy.linalg
        # ut, nt, vt = np.linalg.svd(self.K)
        # rank = np.sum(nt > 1e-15)
        # Kondition = np.linalg.cond(self.K)
        #import pdb
        # pdb.set_trace()

        # if self.useSparse:
        #     self.K=self.K.tocsr()
        #     self.Residual = spsolve(self.K,self.g)
        #     #self.Residual = cg(self.K,self.g,tol=1.0e-5)[0]
        # else:
        self.Residual = linalg.solve(self.K, self.g)
        #import pdb
        # pdb.set_trace()
        self.error = linalg.norm(self.Residual, np.inf)
        return self.error

    def updateSolution(self):
        count = 0
        for i in range(3, self.nDOF):
            if (i - 3) in self.deleteList:
                count += 1
            else:
                self.Phi[i] -= self.Residual[i - 3 - count]
        #self.Phi[3::] -= self.Residual

    def checkConvergence(self):
        if self.error < self.nlTol:
            return False
        else:
            return True

    def setBCs(self):
        self.K = self.K[3::, 3::]
        self.g = self.g[3::]

    def reduceOrder(self):
        newg = self.g[:]
        # newK=self.K[:,:]
        self.deleteList = []

    def getCoords_Qs_at_Quad(self):
        x_quad = np.zeros(self.quadOrder * self.nElements)
        y_quad = np.zeros(self.quadOrder * self.nElements)
        z_quad = np.zeros(self.quadOrder * self.nElements)
        Q1_quad = np.zeros(self.quadOrder * self.nElements)
        Q2_quad = np.zeros(self.quadOrder * self.nElements)
        Q3_quad = np.zeros(self.quadOrder * self.nElements)

        for i in range(self.nElements):
            Q1_el = np.array([self.F1[2 * i], self.F1[2 * i + 1], self.F1[2 * i + 2]])
            Q2_el = np.array([self.F2[2 * i], self.F2[2 * i + 1], self.F2[2 * i + 2]])
            Q3_el = np.array([self.F3[2 * i], self.F3[2 * i + 1], self.F3[2 * i + 2]])
            x_el = np.array([self.x[i], 0.5 * (self.x[i + 1] + self.x[i]), self.x[i + 1]])
            y_el = np.array([self.y[i], 0.5 * (self.y[i + 1] + self.y[i]), self.y[i + 1]])
            z_el = np.array([self.z[i], 0.5 * (self.z[i + 1] + self.z[i]), self.z[i + 1]])
            for j in range(self.quadOrder):
                Q1_quad[self.quadOrder * i + j] += np.dot(self.v[j], Q1_el)
                Q2_quad[self.quadOrder * i + j] += np.dot(self.v[j], Q2_el)
                Q3_quad[self.quadOrder * i + j] += np.dot(self.v[j], Q3_el)
                x_quad[self.quadOrder * i + j] += np.dot(self.v[j], x_el)
                y_quad[self.quadOrder * i + j] += np.dot(self.v[j], y_el)
                z_quad[self.quadOrder * i + j] += np.dot(self.v[j], z_el)
        weights = np.array(self.w)
        return x_quad[:], y_quad[:], z_quad[:], Q1_quad[:], Q2_quad[:], Q3_quad[:], weights

    def getCoords_at_Quad(self):
        x_quad = np.zeros(self.quadOrder * self.nElements)
        y_quad = np.zeros(self.quadOrder * self.nElements)
        z_quad = np.zeros(self.quadOrder * self.nElements)

        for i in range(self.nElements):
            x_el = np.array([self.x[i], 0.5 * (self.x[i + 1] + self.x[i]), self.x[i + 1]])
            y_el = np.array([self.y[i], 0.5 * (self.y[i + 1] + self.y[i]), self.y[i + 1]])
            z_el = np.array([self.z[i], 0.5 * (self.z[i + 1] + self.z[i]), self.z[i + 1]])
            for j in range(self.quadOrder):
                x_quad[self.quadOrder * i + j] += np.dot(self.v[j], x_el)
                y_quad[self.quadOrder * i + j] += np.dot(self.v[j], y_el)
                z_quad[self.quadOrder * i + j] += np.dot(self.v[j], z_el)
        weights = np.array(self.w)
        return x_quad[:], y_quad[:], z_quad[:]