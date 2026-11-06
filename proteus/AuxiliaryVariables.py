"""
Classes for calculating auxiliary variables based on the numerical solution.

.. inheritance-diagram:: proteus.AuxiliaryVariables
   :parts: 1
"""
import numpy
from . import Viewers
from . import Archiver
from xml.etree.ElementTree import *

from . import Profiling
from .Profiling import logEvent

try:
    from proteusGraphical import vtkViewers
except:
    pass
import copy
from . import cfemIntegrals
import math
import os

class AV_base(object):
    def __init__(self):
        pass
    def attachModel(self,model,ar):
        self.model=model
        self.ar=ar
        return self
    def attachAuxiliaryVariables(self,avDict):
        pass
    def writeScalarXdmf(self,u,name):
        #get the current spatial grid
        for child in self.ar.domain:
            if child.tag == "Grid" and child.attrib["Name"] == "Mesh Spatial_Domain":
                self.arGridCollection = child
        assert self.arGridCollection is not None
        self.arGrid = self.arGridCollection[-1]
        #write the attribute
        attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                        "AttributeType":"Scalar",
                                                        "Precision":"8",
                                                        "Center":"Grid"})
        values    = SubElement(attribute,"DataItem",
                               {"Format":"XML",
                                "DataType":"Float",
                                "Precision":"8",
                                "Dimensions":"1"})
        values.text = str(u)
    def calculate_init(self):
        self.calculate()
    def calculate(self):
        pass

class GatherDOF(AV_base):
    def __init__(self,filename):
        self.filename=filename
        self.firstCall = True
    def calculate(self):
        from . import Comm
        comm = Comm.get()
        if self.firstCall:
            self.firstCall = False
            self.filename = os.path.join(Profiling.logDir,self.filename)

        self.fineGridModel=self.model.levelModelList[-1]
        comm.beginSequential()
        print("writing for dof processsor ",comm.rank())
        print("opening dof and node files for processsor ",comm.rank())
        if comm.isMaster():
            doffile=open(self.filename+"_dof.txt","w")
            nodefile=open(self.filename+"_node.txt","w")
        else:
            doffile=open(self.filename+"_dof.txt","a")
            nodefile=open(self.filename+"_node.txt","a")
        for j in range(self.fineGridModel.nc):
            print("writing dof for component ",j)
            self.fineGridModel.u[j].dof.tofile(doffile,sep='\n',format='%21.16e')
            doffile.write('\n')
        print("writing nodes for processor ",comm.rank())
        self.fineGridModel.mesh.nodeArray.tofile(nodefile,sep='\n',format='%21.16e')
        nodefile.write('\n')
        print("closing dof and node files for processsor ",comm.rank())
        doffile.close()
        nodefile.close()
        comm.endSequential()
class BoundaryForce(AV_base):
    def __init__(self,D=1.0,Ubar=1.0,rho=1.0):
        self.C_fact = 2.0/(rho*D*Ubar**2)
    def attachModel(self,model,ar):
        self.model=model
        self.ar=ar
        self.writer = Archiver.XdmfWriter()
        self.nd = model.levelModelList[-1].nSpace_global
        m = self.model.levelModelList[-1]
        flagMax = max(m.mesh.elementBoundaryMaterialTypes)
        flagMin = min(m.mesh.elementBoundaryMaterialTypes)
        assert(flagMin == 0)
        assert(flagMax >= 0)
        self.nForces=flagMax+1
        self.levelFlist=[]
        for m in self.model.levelModelList:
            if self.nd == 2:
                F = numpy.zeros((self.nForces,2),'d')
            elif self.nd == 3:
                F = numpy.zeros((self.nForces,3),'d')
            else:
                logEvent("Can't use stress computation for nd = "+repr(self.nd))
                F=None
            self.levelFlist.append(F)
        self.historyF=[]
        self.historyF.append(copy.deepcopy(self.levelFlist))
#         try:
#             self.dragWindow=Viewers.windowNumber
#             self.viewForce()
#         except:
#             pass
        return self
    def calculate(self):
        for m,F in zip(self.model.levelModelList,self.levelFlist):
            F.flat[:]=0.0
            if self.nd ==3:
                cfemIntegrals.calculateExteriorElementBoundaryStress3D(m.mesh.elementBoundaryMaterialTypes,
                                                                       m.mesh.exteriorElementBoundariesArray,
                                                                       m.mesh.elementBoundaryElementsArray,
                                                                       m.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                       m.ebqe[('u',0)],#pressure
                                                                       m.ebqe[('velocity',1)],#mom_flux_vec_u
                                                                       m.ebqe[('velocity',2)],#mom_flux_vec_v
                                                                       m.ebqe[('velocity',3)],#mom_flux_vec_w
                                                                       m.ebqe[('dS_u',0)],#dS
                                                                       m.ebqe[('n')],
                                                                       F)
            if self.nd == 2:
                cfemIntegrals.calculateExteriorElementBoundaryStress2D(m.mesh.elementBoundaryMaterialTypes,
                                                                       m.mesh.exteriorElementBoundariesArray,
                                                                       m.mesh.elementBoundaryElementsArray,
                                                                       m.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                       m.ebqe[('u',0)],#pressure
                                                                       m.ebqe[('velocity',1)],#mom_flux_vec_u
                                                                       m.ebqe[('velocity',2)],#mom_flux_vec_v
                                                                       m.ebqe[('dS_u',0)],#dS
                                                                       m.ebqe[('n')],
                                                                       F)
            logEvent("Force")
            logEvent(repr(F))
            Ftot=F[0,:]
            for ib in range(1,self.nForces):
                Ftot+=F[ib,:]
            logEvent("Total force on all boundaries")
            logEvent(repr(Ftot))
        logEvent("Drag Force " +repr(self.model.stepController.t_model)+" "+repr(F[-1,0]))
        logEvent("Lift Force " +repr(self.model.stepController.t_model)+" "+repr(F[-1,1]))
        logEvent("Drag Coefficient " +repr(self.model.stepController.t_model)+" "+repr(self.C_fact*F[-1,0]))
        logEvent("Lift Coefficient " +repr(self.model.stepController.t_model)+" "+repr(self.C_fact*F[-1,1]))
#        for ib in range(self.nForces):
#             self.writeScalarXdmf(self.C_fact*F[ib,0],"Drag Coefficient %i" % (ib,))
#             self.writeScalarXdmf(self.C_fact*F[ib,1],"Lift Coefficient %i" % (ib,))
        self.historyF.append(copy.deepcopy(self.levelFlist))
#         try:
#             self.viewForce()
#         except:
#             pass
#     def viewForce(self):
#         tList=[]
#         FxList=[]
#         FyList=[]
#         for ti,levelFlist in enumerate(self.historyF):
#             tList.append(ti)
#             FxList.append(self.C_fact*levelFlist[-1][-1,0])
#             FyList.append(self.C_fact*levelFlist[-1][-1,1])
#         print "In view force",FxList
#         Viewers.windowNumber=self.dragWindow
#         vtkViewers.viewScalar_1D(numpy.array(tList),numpy.array(FxList),"t","Fx","Drag Coefficients",Viewers.windowNumber,
#                                  Pause=False,
#                                  sortPoints=False,newMesh=True)
#         Viewers.newPlot()
#         Viewers.newWindow()
#         vtkViewers.viewScalar_1D(numpy.array(tList),numpy.array(FyList),"t","Fy","Lift Coefficients",Viewers.windowNumber,
#                                  Pause=False,
#                                  sortPoints=False,newMesh=True)
#         Viewers.newPlot()
#         Viewers.newWindow()

class PressureProfile(AV_base):
    def __init__(self,flag=0,center=(0.0,0.0),radius=1.0):
        self.flag=flag
        self.center=center
        self.r=radius
    def attachModel(self,model,ar):
        self.tCount=0
        self.model=model
        self.ar=ar
        self.nd = model.levelModelList[-1].nSpace_global
        self.levelPlist=[]
        self.levelThetalist=[]
        for m in self.model.levelModelList:
            p=[]
            theta=[]
            if self.nd == 2:
                for nN in range(m.mesh.nNodes_global):
                    if m.mesh.nodeMaterialTypes[nN] == self.flag:
                        p.append(m.u[0].dof[nN])
                        theta.append((180.0/math.pi)*math.atan2(m.mesh.nodeArray[nN][1]-self.center[1],
                                                                m.mesh.nodeArray[nN][0]-self.center[0]))
            elif self.nd == 3:
                pass
            else:
                logEvent("Can't use stress computation for nd = "+repr(self.nd))
                pass
            self.levelPlist.append(p)
            self.levelThetalist.append(theta)
        if self.ar.hdfFile is not None:
            self.ar.hdfFile.createArray("/",'theta',theta)
        #self.historyP=[]
        #self.historyP.append(copy.deepcopy(self.levelPlist))
#         try:
#             self.pWindow=Viewers.windowNumber
#             self.viewP()
#         except:
#             pass
        #self.dataStorage['PressureTheta'] = self.levelThetalist
        #self.dataStorage['PressureHistory'] = [self.levelPlist]
        return self
    def calculate(self):
        self.levelPlist=[]
        self.maxX=0.0
        self.maxY=0.0
        for m in self.model.levelModelList:
            p=[]
            if self.nd ==3:
                pass
            if self.nd == 2:
                for nN in range(m.mesh.nNodes_global):
                    if m.mesh.nodeMaterialTypes[nN] == self.flag:
                        p.append(m.u[0].dof[nN])
            self.levelPlist.append(p)
        if self.ar.hdfFile is not None:
            self.ar.hdfFile.createArray("/","pressure"+str(self.tCount),p)
        self.tCount+=1
#         if self.dataStorage is not None:
#             tmp = self.dataStorage['PressureHistory']
#             tmp.append(self.levelPlist)
#             self.dataStorage['PressureHistory']=tmp
#         try:
#             self.viewP()
#         except:
#             pass
#     def viewP(self):
#         Viewers.windowNumber=self.pWindow
#         vtkViewers.vtkDisplayXYPlot(self.levelThetalist[-1],
#                                     self.levelPlist[-1],
#                                     "theta",
#                                     "P",
#                                     "Pressure Profile",
#                                     Viewers.windowNumber,
#                                     Pause=False,
#                                     sortPoints=True)
#         Viewers.newPlot()
#         Viewers.newWindow()

class RecirculationLength(AV_base):
    def __init__(self,rcStartX=None):
        self.rcStartX=rcStartX
    def attachModel(self,model,ar):
        self.model=model
        self.ar=ar
        #self.dataStorage=model.simTools.dataStorage
        self.levelLlist=[]
        for m in self.model.levelModelList:
            self.levelLlist.append(0.0)
        #self.historyL=[]
        #self.historyL.append(copy.deepcopy(self.levelLlist))
#         try:
#             self.LWindow=Viewers.windowNumber
#             self.viewL()
#         except:
#             pass
        #self.dataStorage['RecirculationLengthHistory'] = [self.levelLlist]
        return self
    def calculate(self):
        #cek hack for min
        self.minX_neg=1.0e10
        self.maxX_neg=-1.0e10
        self.minX=1.0e10
        self.maxX=-1.0e10
        self.maxX_domain=-1.0e10
        self.minX_domain=1.0e10
        for l,m in enumerate(self.model.levelModelList):
            for eN in range(m.mesh.nElements_global):
                for nN in range(m.mesh.nNodes_element):
                    x = m.mesh.nodeArray[m.mesh.elementNodesArray[eN,nN]][0]
                    u = m.u[1].dof[m.mesh.elementNodesArray[eN,nN]]
                    if x > self.maxX_domain:
                        self.maxX_domain = x
                    if x < self.minX_domain:
                        self.minX_domain = x
                    if u < 0.0:
                        if x > self.maxX_neg:
                            self.maxX_neg = x
                            if x > self.maxX:
                                self.maxX = x
                            #check to see if 0 contour is further to right
                            for nN_neigh in list(range(0,nN))+list(range(nN+1,m.mesh.nNodes_element)):
                                u2 = m.u[1].dof[m.mesh.elementNodesArray[eN,nN_neigh]]
                                x2 = m.mesh.nodeArray[m.mesh.elementNodesArray[eN,nN_neigh]][0]
                                if x2 > x:
                                    if u2 >=0.0:
                                        x0 = x - u*(x2-x)/(u2-u)
                                        if x0 > self.maxX:
                                            self.maxX=x0
                        if x < self.minX_neg:
                            self.minX_neg = x
                            if x < self.minX:
                                self.minX = x
                            #check to see if 0 contour is further to right
                            for nN_neigh in list(range(0,nN))+list(range(nN+1,m.mesh.nNodes_element)):
                                u2 = m.u[1].dof[m.mesh.elementNodesArray[eN,nN_neigh]]
                                x2 = m.mesh.nodeArray[m.mesh.elementNodesArray[eN,nN_neigh]][0]
                                if x2 < x:
                                    if u2 >=0.0:
                                        x0 = x - u*(x2-x)/(u2-u)
                                        if x0 < self.minX:
                                            self.minX=x0
            #catch case with zero recirculation
            if self.minX > self.maxX_domain:
                self.minX = self.minX_domain
            if self.maxX < self.minX_domain:
                self.maxX = self.minX_domain
            if self.rcStartX is not None:
                if self.maxX > self.rcStartX:
                    self.levelLlist.append(self.maxX-self.rcStartX)
                else:
                    self.levelLlist.append(0.0)
            else:
                self.levelLlist.append(self.maxX-self.minX)
        #self.historyL.append(copy.deepcopy(self.levelLlist))
        #self.writeScalarXdmf(self.levelLlist,"Recirculation Length")
#         if self.dataStorage is not None:
#             tmp = self.dataStorage['RecirculationLengthHistory']
#             tmp.append(self.levelLlist)
#             self.dataStorage['RecirculationLengthHistory']=tmp
#         try:
#             self.viewL()
#         except:
#             pass
        logEvent("Recirculation Length "+repr(self.levelLlist[-1]))
#     def viewL(self):
#         tList=[]
#         LList=[]
#         for ti,levelLlist in enumerate(self.historyL):
#             tList.append(ti)
#             LList.append(levelLlist[-1])
#         Viewers.windowNumber=self.LWindow
#         vtkViewers.vtkDisplayXYPlot(tList,
#                                     LList,
#                                     "t",
#                                     "L",
#                                     "Recirculation Length",
#                                     Viewers.windowNumber,
#                                     Pause=False,
#                                     sortPoints=False)
#         Viewers.newPlot()
#         Viewers.newWindow()
class VelocityAverage(AV_base):
    def __init__(self):
        pass
    def attachModel(self,model,ar):
        filename = os.path.join(Profiling.logDir,'velocity_average.txt')
        self.Vfile = open(filename,'w')
        self.model=model
        self.ar=ar
        self.writer = Archiver.XdmfWriter()
        self.nd = model.levelModelList[-1].nSpace_global
        m = self.model.levelModelList[-1]
        #get the volume of the domain
        self.volume=0.0
        for eN in range(m.mesh.nElements_global):
            for k in range(m.nQuadraturePoints_element):
                self.volume+=m.q[('dV_u',0)][eN,k]
        #flagMax = max(m.mesh.elementBoundaryMaterialTypes)
        #flagMin = min(m.mesh.elementBoundaryMaterialTypes)
        #assert(flagMin == 0)
        #assert(flagMax >= 0)
        self.levelVlist=[]
        for m in self.model.levelModelList:
            if self.nd == 2:
                V = numpy.zeros((2,),'d')
            elif self.nd == 3:
                V = numpy.zeros((3,),'d')
            else:
                logEvent("Can't use velocity average for nd = "+repr(self.nd))
                V=None
            self.levelVlist.append(V)
#         self.historyV=[]
#         self.historyV.append(copy.deepcopy(self.levelVlist))
#         try:
#             self.velocityWindow=Viewers.windowNumber
#             self.viewVelocity()
#         except:
#             pass
        return self
    def calculate(self):
        for m,V in zip(self.model.levelModelList,self.levelVlist):
            V.flat[:]=0.0
            for eN in range(m.mesh.nElements_global):
                for k in range(m.nQuadraturePoints_element):
                    V[0] += m.q[('u',1)][eN,k]*m.q[('dV_u',0)][eN,k]
                    V[1] += m.q[('u',2)][eN,k]*m.q[('dV_u',0)][eN,k]
                    if self.nd == 3:
                        V[2] += m.q[('u',3)][eN,k]*m.q[('dV_u',0)][eN,k]
            V/=self.volume
#             if self.nd ==3:
#                 cfemIntegrals.calculateVelocityVolumeAverage3D(m.mesh.elementBoundaryMaterialTypes,
#                                                                m.q[('u',1)],#u
#                                                                m.q[('u',2)],#v
#                                                                m.q[('u',3)],#w
#                                                                m.q[('dV_u',0)],#dV
#                                                                V)
#             if self.nd == 2:
#                 cfemIntegrals.calculateVelocityVolumeAverage2D(m.mesh.elementBoundaryMaterialTypes,
#                                                                m.q[('u',1)],#u
#                                                                m.q[('u',2)],#v
#                                                                m.q[('dV_u',0)],#dV
#                                                                V)
            logEvent("Average Velocity")
            logEvent(repr(V))
            if self.nd == 2:
                self.Vfile.write('%21.15e %21.15e \n' % tuple(V))
            else:
                self.Vfile.write('%21.15e %21.15e %21.15e\n' % tuple(V))

class BoundaryPressure(AV_base):
    def __init__(self):
        pass
    def attachModel(self,model,ar):
        #MultilevelTransport model
        self.model=model
        #data archive and writer
        self.ar=ar
        self.writer = Archiver.XdmfWriter()
        #number of space dimensions
        self.nd = model.levelModelList[-1].nSpace_global
        #Transport object (numerical approximation) on finest level in mesh hierarchy
        m = self.model.levelModelList[-1]
        #figure out the number of pressures might compute based on different boundary types
        #listed in mesh
        flagMax = max(m.mesh.elementBoundaryMaterialTypes)
        flagMin = min(m.mesh.elementBoundaryMaterialTypes)
        assert(flagMin == 0)
        assert(flagMax >= 0)
        self.nPressures=flagMax+1
        #create a list containing pressure for each boundary face on each mesh in
        #multilevel hierarchy
        self.levelPlist=[]
        #also area/length of boundaries
        self.levelAlist=[]
        for m in self.model.levelModelList:
            P = numpy.zeros((self.nPressures,),'d')
            A = numpy.zeros((self.nPressures,),'d')
            self.levelPlist.append(P)
            self.levelAlist.append(A)
        #time history of pressures
        self.historyP=[]
        self.historyP.append(copy.deepcopy(self.levelPlist))
        return self
    def calculate(self):
        #loop through meshes in mesh hiearchy and compute average pressure for each face
        for m,P,A in zip(self.model.levelModelList,self.levelPlist,self.levelAlist):
            P.flat[:]=0.0
            A.flat[:]=0.0
            cfemIntegrals.accumulateExteriorElementPressureIntegrals(m.mesh.elementBoundaryMaterialTypes,
                                                                     m.mesh.exteriorElementBoundariesArray,
                                                                     m.ebqe[('u',0)],#pressure
                                                                     m.ebqe[('dS_u',0)],#dS
                                                                     P,
                                                                     A)
            for i in range(len(A)):#could do this in a fancier way with numpy
                if abs(A[i]) > 0.0:
                    P[i] /= A[i]
            logEvent("Pressure")
            logEvent(repr(P))
        self.historyP.append(copy.deepcopy(self.levelPlist))

class ConservationHistoryMC(AV_base):
    """A simple class for storing the time history of things related conservation in conservative level set methods"""
    def __init__(self,filename):
        from . import Comm
        self.comm = Comm.get()
        self.filename=filename
        AV_base.__init__(self)
    def attachModel(self,model,ar):
        AV_base.attachModel(self,model,ar)
        self.filename = os.path.join(Profiling.logDir,self.filename)
        self.massVOFFile = open(self.filename+"massVOF",'w')
        self.massLSFile = open(self.filename+"massLS",'w')
        self.massErrorVOFFile = open(self.filename+"massErrorVOF",'w')
        self.massErrorLSFile = open(self.filename+"massErrorLS",'w')
        self.fluxFile = open(self.filename+"flux",'w')
        self.timeFile = open(self.filename+"time",'w')
    def calculate(self):
        if self.comm.isMaster():
            c = self.model.levelModelList[-1].coefficients
            self.massVOFFile.write("%21.16e\n" % (c.vofGlobalMassArray[-1],))
            self.massLSFile.write("%21.16e\n" % (c.lsGlobalMassArray[-1],))
            self.massErrorVOFFile.write("%21.16e\n" % (c.vofGlobalMassErrorArray[-1],))
            self.massErrorLSFile.write("%21.16e\n" % (c.lsGlobalMassErrorArray[-1],))
            self.fluxFile.write("%21.16e\n" % (c.fluxArray[-1],))
            self.timeFile.write("%21.16e\n" % (c.timeArray[-1],))

class ConservationHistoryLS(AV_base):
    """A simple class for storing the time history of things related conservation in non-conservative level set methods"""
    def __init__(self,filename):
        from . import Comm
        self.comm = Comm.get()
        self.filename=filename
        AV_base.__init__(self)
    def attachModel(self,model,ar):
        AV_base.attachModel(self,model,ar)
        self.filename = os.path.join(Profiling.logDir,self.filename)
        self.massFile = open(self.filename+"massLS",'w')
        self.massErrorFile = open(self.filename+"massErrorLS",'w')
        self.fluxFile = open(self.filename+"flux",'w')
        self.timeFile = open(self.filename+"time",'w')
    def calculate(self):
        if self.comm.isMaster():
            c = self.model.levelModelList[-1].coefficients
            self.massFile.write("%21.16e\n" % c.lsGlobalMassArray[-1])
            self.massErrorFile.write("%21.16e\n" % c.lsGlobalMassErrorArray[-1])
            self.fluxFile.write("%21.16e\n" % c.fluxArray[-1])
            self.timeFile.write("%21.16e\n" % c.timeArray[-1])


class VelocityNormOverRegion(AV_base):
    def __init__(self,regionIdList=[1]):
        self.regionIdList=  regionIdList #which elements to select for norms
    def attachModel(self,model,ar):
        filename=os.path.join(Profiling.logDir,'velocity_norm_%s.txt')
        self.Vfile = open(filename % model.name,'w')
        self.model=model
        self.ar=ar
        self.writer = Archiver.XdmfWriter()
        self.nd = model.levelModelList[-1].nSpace_global
        #element ids for the region sampled
        self.elementsInRegion = []
        for m in self.model.levelModelList:
            self.elementsInRegion.append([])
            for id in self.regionIdList:
                self.elementsInRegion[-1].append(numpy.where(m.mesh.elementMaterialTypes == id))
        #get the volume of the region
        m = self.model.levelModelList[0]
        self.volume=0.0
        for region in self.elementsInRegion[0]:#coarse level only needed
            for eN in region[0]: #where returns a tuple
                for k in range(m.nQuadraturePoints_element):
                    self.volume+=m.q[('dV_u',0)][eN,k]

        self.levelNormList=[]
        for m in self.model.levelModelList:
            self.levelNormList.append({'L2':-12345.0,'L1':-12345.0,'LI':-12345.0})
        return self
    def calculate(self):
        for m,regions,results in zip(self.model.levelModelList,self.elementsInRegion,self.levelNormList):
            for key in ['L2','L1','LI']:
                results[key] = 0.0
            for region in regions:#go through material types
                for eN in region[0]: #where returns a tuple
                    for k in range(m.nQuadraturePoints_element):
                        e1 = numpy.sum(numpy.absolute(m.q[('velocity',0)][eN,k,:]))
                        e2 = numpy.inner(m.q[('velocity',0)][eN,k,:],m.q[('velocity',0)][eN,k,:])
                        ei = numpy.max(numpy.absolute(m.q[('velocity',0)][eN,k,:]))
                        results['L2'] += e2*m.q[('dV_u',0)][eN,k]
                        results['L1'] += e1*m.q[('dV_u',0)][eN,k]
                        results['LI'] = max(results['LI'],ei)
            logEvent("Velocity Norms in Region %s L2= %s L1= %s LI= %s " % (self.regionIdList,results['L2'],results['L1'],results['LI']))
            self.Vfile.write('%21.15e %21.15e %21.15e\n' % (results['L2'],results['L1'],results['LI']))

class MassOverRegion(AV_base):
    def __init__(self,regionIdList=None,ci=0):
        self.regionIdList=  regionIdList #which elements to select for mass integral, None means all
        self.ci=ci
    def attachModel(self,model,ar):
        filename=os.path.join(Profiling.logDir,'total_mass_comp_%s_%s.txt' )
        self.ofile = open(filename % (self.ci,model.name),'w')
        self.model=model
        self.ar=ar
        self.writer = Archiver.XdmfWriter()
        self.nd = model.levelModelList[-1].nSpace_global
        #element ids for the region sampled
        self.elementsInRegion = []
        for m in self.model.levelModelList:
            self.elementsInRegion.append([])
            if self.regionIdList is None:
                #where returns a tuple
                self.elementsInRegion[-1].append((numpy.arange(m.mesh.nElements_global),))
            else:
                for id in self.regionIdList:
                    self.elementsInRegion[-1].append(numpy.where(m.mesh.elementMaterialTypes == id))
        #get the volume of the region
        m = self.model.levelModelList[0]
        self.volume=0.0
        for region in self.elementsInRegion[0]:#coarse level only needed
            for eN in region[0]: #where returns a tuple
                for k in range(m.nQuadraturePoints_element):
                    self.volume+=m.q[('dV_u',0)][eN,k]

        self.levelNormList=[]
        for m in self.model.levelModelList:
            self.levelNormList.append({'total':-12345.0,'L2':-12345.0,'L1':-12345.0,'LI':-12345.0})
        return self
    def calculate(self):
        ci = self.ci
        for m,regions,results in zip(self.model.levelModelList,self.elementsInRegion,self.levelNormList):
            for key in ['total','L2','L1','LI']:
                results[key] = 0.0
            for region in regions:#go through material types
                for eN in region[0]: #where returns a tuple
                    for k in range(m.nQuadraturePoints_element):
                        e1 = abs(m.q[('m',ci)][eN,k])
                        e2 = e1*e1
                        ei = e1
                        mtot= m.q[('m',self.ci)][eN,k]
                        results['total']+= mtot*m.q[('dV_u',ci)][eN,k]
                        results['L2'] += e2*m.q[('dV_u',ci)][eN,k]
                        results['L1'] += e1*m.q[('dV_u',ci)][eN,k]
                        results['LI'] = max(results['LI'],ei)
            if self.regionIdList is None:
                logEvent("Mass Norms in Domain Total= %s L2= %s L1= %s LI= %s " % (results['total'],results['L2'],results['L1'],results['LI']))
            else:
                logEvent("Mass Norms in Domain %s Total= %s L2= %s L1= %s LI= %s " % (self.regionIdList,results['total'],results['L2'],results['L1'],results['LI']))
            self.ofile.write('%21.15e %21.15e %21.15e %21.15e\n' % (results['total'],results['L2'],results['L1'],results['LI']))

class PT123velocityGenerator(AV_base):
    """
    write out the velocity field from a model for PT123 to use in tracking as a
    standalone application
    """
    from .cpostprocessing import getElementRT0velocityValues
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    getElementRT0velocityValues = staticmethod(getElementRT0velocityValues)
    def __init__(self,filebase,tnList,ci=0):
        self.filebase = filebase
        self.tnList = tnList
        self.ci=ci
        self.nodal_velocity_file = None
        self.element_velocity_file=None
        self.element_vf_file=None
        self.nodal_vf_file=None
    def attachModel(self,model,ar):
        AV_base.attachModel(self,model,ar)
        self.filebase=os.path.join(Profiling.logDir,self.filebase)

        #write mesh out first and don't rewrite for now
        self.nd  = model.levelModelList[-1].nSpace_global
        self.mesh= model.levelModelList[-1].mesh
        self.writePT123inputMesh(self.nd,self.mesh)
        self.velocityPostProcessor = self.model.levelModelList[-1].velocityPostProcessor
        if self.velocityPostProcessor is not None:
            self.PT123_RT0_interpolation_points = self.mesh.nodeArray[self.mesh.elementNodesArray]
            self.PT123_interpolation_values = numpy.zeros((self.mesh.nElements_global,self.mesh.nNodes_element,self.nd),'d')

        return self
    def checkFileClose(self,filetoclose):
        if self.model.levelModelList[-1].timeIntegration.t >= self.tnList[-1]:
            filetoclose.write("ENDR\n")
        #let file close itself?
    def writePT123inputMesh(self,nd,mesh):
        base = 1 #base 1 numbering
        mesh_prefix = '1dm'; ele_label   = 'GE2'; node_label = 'GN'
        if nd == 2:
            mesh_prefix = '2dm'; ele_label   = 'GE3';
        elif nd == 3:
            mesh_prefix = '3dm'; ele_label   = 'GE4';
        #
        fmesh = open(self.filebase+'.'+mesh_prefix,'w')
        fmesh.write('MESH \n')
        for eN in range(mesh.nElements_global):
            fmesh.write("%s   %d " % (ele_label,eN+base))
            for nN in range(mesh.nNodes_element):
                fmesh.write("   %d " % (mesh.elementNodesArray[eN,nN]+base))
            fmesh.write("\n")
        #
        for nN in range(mesh.nNodes_global):
            fmesh.write("%s   %d " % (node_label,nN+base))
            for I in range(nd):
                fmesh.write("   %g " % (mesh.nodeArray[nN,I]))
            fmesh.write("\n")
        #
        fmesh.write("ENDR\n")
        fmesh.close()
    #
    def writePT123nodalVelocity(self,components):
        #write out velocity file for current time step
        if self.nodal_velocity_file is None:
            vel_prefix = 'vn1';
            if self.nd == 2:
                vel_prefix = 'vn2';
            elif self.nd == 3:
                vel_prefix = 'vn3';

            self.nodal_velocity_file = open(self.filebase+'.'+vel_prefix,'w')
            self.nodal_velocity_file.write("     %d     %d    %d \n" % (self.mesh.nNodes_global,self.nd,len(self.tnList)-1))
        #
        self.nodal_velocity_file.write("TS    %f \n" % self.model.levelModelList[-1].timeIntegration.t)

        for I in range(self.mesh.nNodes_global):
            for j in range(self.nd):
                self.nodal_velocity_file.write("  %f  " % self.model.levelModelList[-1].u[components[j]][I])
            #mwf PT123 doc incorrect, only reads up to neq
            #for j in range(p.nd,3):
            #    fvel.write("  %g  " % 0.0)
            self.nodal_velocity_file.write("\n")
        #
        self.checkFileClose(self.nodal_velocity_file)
    #
    def writePT123elementVelocity(self,components):
        #write out velocity file for current time step
        if self.element_velocity_file is None:
            vel_prefix = 've1';
            if self.nd == 2:
                vel_prefix = 've2';
            elif self.nd == 3:
                vel_prefix = 've3';

            self.element_velocity_file = open(self.filebase+'.'+vel_prefix,'w')
            self.element_velocity_file.write("     %d     %d    %d \n" % (self.mesh.nElements_global,self.nd,len(self.tnList)-1))
        #
        self.element_velocity_file.write("TS    %f \n" % self.model.levelModelList[-1].timeIntegration.t)

        for eN in range(self.mesh.nElements_global):
            for nN in range(self.mesh.nNodes_element):
                for j in range(self.nd):
                    I = self.model.levelModelList[-1].u[components[j]].femSpace.dofMap.l2g[eN,nN]
                    self.element_velocity_file.write("  %f  " % self.model.levelModelList[-1].u[components[j]][I])
            #mwf PT123 doc incorrect, only reads up to neq
            #for j in range(p.nd,3):
            #    fvel.write("  %g  " % 0.0)
            self.element_velocity_file.write("\n")
        #
        self.checkFileClose(self.element_velocity_file)
    #
    def writePT123MixedVelocityAsElementVelocity(self,ci):
        #write out velocity file for current time step
        if self.element_velocity_file is None:
            vel_prefix = 've1';
            if self.nd == 2:
                vel_prefix = 've2';
            elif self.nd == 3:
                vel_prefix = 've3';

            self.element_velocity_file = open(self.filebase+'.'+vel_prefix,'w')
            self.element_velocity_file.write("     %d     %d    %d \n" % (self.mesh.nElements_global,self.nd,len(self.tnList)-1))
        #
        self.element_velocity_file.write("TS    %f \n" % self.model.levelModelList[-1].timeIntegration.t)
        self.PT123_interpolation_values = self.velocityPostProcessor.evaluateElementVelocityField(self.PT123_RT0_interpolation_points,ci)

        for i in range(self.PT123_interpolation_values.shape[0]):
            for j in range(self.PT123_interpolation_values.shape[1]):
                for k in range(self.PT123_interpolation_values.shape[2]):
                    self.element_velocity_file.write("  %f  " % self.PT123_interpolation_values[i,j,k])
            #mwf PT123 doc incorrect, only reads up to neq
            #for j in range(p.nd,3):
            #    fvel.write("  %g  " % 0.0)
            self.element_velocity_file.write("\n")
        #
        self.checkFileClose(self.element_velocity_file)

    #
    def writePT123elementPorosity(self,value=1.0):
        #write out constant value for volume fraction for now
        if self.element_vf_file is None:
            vf_prefix = 'eemc1';
            if self.nd == 2:
                vf_prefix = 'eemc2';
            elif self.nd == 3:
                vf_prefix = 'eemc3';

            self.element_vf_file = open(self.filebase+'.'+vf_prefix,'w')
            self.element_vf_file.write("     %d    %d \n" % (self.mesh.nElements_global,len(self.tnList)-1))
        #
        self.element_vf_file.write("TS    %f \n" % self.model.levelModelList[-1].timeIntegration.t)

        for eN in range(self.mesh.nElements_global):
            self.element_vf_file.write("  %f  \n" % value)
        #
        self.checkFileClose(self.element_vf_file)

    #
    def writePT123nodalPorosity(self,value=1.0):
        #write out constant value for volume fraction for now
        if self.nodal_vf_file is None:
            vf_prefix = 'nemc1';
            if self.nd == 2:
                vf_prefix = 'nemc2';
            elif self.nd == 3:
                vf_prefix = 'nemc3';

            self.nodal_vf_file = open(self.filebase+'.'+vf_prefix,'w')
            self.nodal_vf_file.write("     %d    %d \n" % (self.mesh.nNodes_global,len(self.tnList)-1))
        #
        self.nodal_vf_file.write("TS    %f \n" % self.model.levelModelList[-1].timeIntegration.t)

        for nN in range(self.mesh.nNodes_global):
            self.nodal_vf_file.write("  %f  \n" % value)
        #
        self.checkFileClose(self.nodal_vf_file)
    #
    def calculate(self):
        #mwf todo when to write what type of file?
        timeToPrint = False
        for t in self.tnList[1:]:
            if abs(t-self.model.levelModelList[-1].timeIntegration.t) < 1.0e-6:
                timeToPrint = True
                break
        if timeToPrint:
            if self.velocityPostProcessor is not None:
                self.writePT123MixedVelocityAsElementVelocity(self.ci)
                self.writePT123elementPorosity(value=1.)
        #mwf stopped here