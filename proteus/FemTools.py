"""
Class hierarchies for constructing and working with finite element spaces

.. inheritance-diagram:: proteus.FemTools
   :parts: 1
"""
from .EGeometry import *
from .MeshTools import *
from .LinearAlgebraTools import *
from .Quadrature import *
from . import cfemIntegrals
from . import cpartitioning
from .Profiling import logEvent
import numpy as np


class ReferenceElement(object):
    """
    A base class for simple domains on which
    local function spaces will be built
    """
    def __init__(self,dim=0,nNodes=0,nElementBoundaries=0):
        self.dim=dim
        self.range_dim = list(range(dim))
        self.nNodes = nNodes
        self.range_nNodes = list(range(nNodes))
        self.nElementBoundaries = nElementBoundaries
        self.range_nElementBoundaries = list(range(nElementBoundaries))
        self.nodeList=[]
        self.boundaryMapList=[]
        self.boundaryMapInverseList=[]
        self.boundaryJacobianList=[]
        self.boundaryUnitNormalList=[]
    def onElement(self,xi):
        raise NotImplementedError

class ReferenceSimplex(ReferenceElement):
    """
    The unit simplex in :math:`R^n, n<=3`
    """
    def __init__(self,nd=3):
        ReferenceElement.__init__(self,nd,nd+1,nd+1)
        if nd == 1:
            self.nodeList=[numpy.array([0.0]),
                           numpy.array([1.0])]
            #build mapping from reference simplex of lower dimension to the boundaries of the simplex
            #for 1D the 0D space only contains 0 so the map is really just translation of 0 to the endpoint
            #The boundaries are numbered according to the node opposite the boundary
            #0
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0] + 1.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0] - 1.0]))
            self.boundaryJacobianList.append(numpy.array([1.0]))
            self.boundaryUnitNormalList.append(numpy.array([1.0]))
            #1
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0] + 0.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0]]))
            self.boundaryJacobianList.append(numpy.array([1.0]))
            self.boundaryUnitNormalList.append(numpy.array([-1.0]))
        elif nd == 2:
            self.nodeList=[numpy.array([0.0,0.0]),
                           numpy.array([1.0,0.0]),
                           numpy.array([0.0,1.0])]
            #0
            self.boundaryMapList.append(lambda xBar: numpy.array([    xBar[0],
                                                                    1.0-xBar[0]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0]]))
            self.boundaryJacobianList.append(numpy.array([[ 1.0],
                                                            [-1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([1.0/sqrt(2.0),
                                                              1.0/sqrt(2.0)]))
            #1
            self.boundaryMapList.append(lambda xBar: numpy.array([0.0,
                                                                    xBar[0]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[1]]))
            self.boundaryJacobianList.append(numpy.array([[0.0],
                                                            [1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([-1.0,
                                                              0.0]))
            #2
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],
                                                                    0.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0]]))
            self.boundaryJacobianList.append(numpy.array([[1.0],
                                                            [0.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,
                                                              -1.0]))
        elif nd == 3:
            self.nodeList=[numpy.array([0.0,
                                         0.0,
                                         0.0]),
                           numpy.array([1.0,
                                         0.0,
                                         0.0]),
                           numpy.array([0.0,
                                         1.0,
                                         0.0]),
                           numpy.array([0.0,
                                         0.0,
                                         1.0])]
            #0
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],
                                                                    xBar[1],
                                                                    1.0-xBar[0] - xBar[1]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0],
                                                                        x[1],
                                                                        0.0]))
            self.boundaryJacobianList.append(numpy.array([[1.0 , 0.0],
                                                            [0.0 , 1.0],
                                                            [-1.0,-1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([1.0/sqrt(3.0),
                                                              1.0/sqrt(3.0),
                                                              1.0/sqrt(3.0)]))
            #1
            self.boundaryMapList.append(lambda xBar: numpy.array([0.0,
                                                                    xBar[1],
                                                                    xBar[0]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[2],
                                                                        x[1],
                                                                        0.0]))
            self.boundaryJacobianList.append(numpy.array([[0.0,0.0],
                                                            [0.0,1.0],
                                                            [1.0,0.0]]))
            self.boundaryUnitNormalList.append(numpy.array([-1.0,
                                                              0.0,
                                                              0.0]))
            #2
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],
                                                                    0.0,
                                                                    xBar[1]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0],
                                                                        x[2],
                                                                        0.0]))
            self.boundaryJacobianList.append(numpy.array([[1.0,0.0],
                                                            [0.0,0.0],
                                                            [0.0,1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,
                                                              -1.0,
                                                              0.0]))
            #3
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],
                                                                    xBar[1],
                                                                    0.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0],
                                                                        x[1],
                                                                        0.0]))
            self.boundaryJacobianList.append(numpy.array([[1.0,0.0],
                                                            [0.0,1.0],
                                                            [0.0,0.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,
                                                              0.0,
                                                              -1.0]))
    def onElement(self,xi):
        return (xi >= 0).all() and sum(xi) <= 1

class ReferenceCube(ReferenceElement):
    """
    The unit cube in :math:`R^n, n<=3`
    """
    def __init__(self,nd=3):
        ReferenceElement.__init__(self,nd,2**nd,2*nd)
        if nd == 1:
            self.nodeList=[numpy.array([0.0]),
                           numpy.array([1.0])]
            #build mapping from reference simplex of lower dimension to the boundaries of the simplex
            #for 1D the 0D space only contains 0 so the map is really just translation of 0 to the endpoint
            #The boundaries are numbered according to the node opposite the boundary
            #0
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0] + 1.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0] - 1.0]))
            self.boundaryJacobianList.append(numpy.array([1.0]))
            self.boundaryUnitNormalList.append(numpy.array([1.0]))
            #1
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0] - 1.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0]]))
            self.boundaryJacobianList.append(numpy.array([1.0]))
            self.boundaryUnitNormalList.append(numpy.array([-1.0]))
        elif nd == 2:
            self.nodeList=[numpy.array([-1.0,-1.0]),
                           numpy.array([ 1.0,-1.0]),
                           numpy.array([ 1.0, 1.0]),
                           numpy.array([-1.0, 1.0])]
            #remember  boundary reference geometry is  [0,1], not [-1,1].
            #0: 0-1
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],-1.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0]]))
            self.boundaryJacobianList.append(numpy.array([[ 1.0],[0.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,-1.0]))

            #1:  1-2
            self.boundaryMapList.append(lambda xBar: numpy.array([1.0,xBar[0]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[1]]))
            self.boundaryJacobianList.append(numpy.array([[0.0],[1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([1.0,0.0]))

            #2: 2-3
            self.boundaryMapList.append(lambda xBar: numpy.array([-xBar[0],1.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([-x[0]]))
            self.boundaryJacobianList.append(numpy.array([[-1.0],[0.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,1.0]))

            #3: 3-0
            self.boundaryMapList.append(lambda xBar: numpy.array([-1.0,-xBar[0]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([-x[1]]))
            self.boundaryJacobianList.append(numpy.array([[0.0],[-1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([-1.0,0.0]))
        elif nd == 3:
            self.nodeList=[numpy.array([-1.0,-1.0,-1.0]),
                           numpy.array([ 1.0,-1.0,-1.0]),
                           numpy.array([ 1.0, 1.0,-1.0]),
                           numpy.array([-1.0, 1.0,-1.0]),
                           numpy.array([-1.0,-1.0, 1.0]),
                           numpy.array([ 1.0,-1.0, 1.0]),
                           numpy.array([ 1.0, 1.0, 1.0]),
                           numpy.array([-1.0, 1.0, 1.0])]
            #0: 0-1-2-3
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],
                                                                  xBar[1],
                                                                  -1.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0],x[1]]))
            self.boundaryJacobianList.append(numpy.array([[1.0, 0.0],
                                                          [0.0, 1.0],
                                                          [0.0, 0.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,0.0,-1.0]))

            #1: 0-1-5-4
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],
                                                                 -1.0,
                                                                 xBar[1]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0],x[2]]))
            self.boundaryJacobianList.append(numpy.array([[1.0,0.0],
                                                          [0.0,0.0],
                                                          [0.0,1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,-1.0,0.0]))

            #2: 1-2-6-5
            self.boundaryMapList.append(lambda xBar: numpy.array([1.0,
                                                                  xBar[0],
                                                                  xBar[1]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[1],x[2]]))
            self.boundaryJacobianList.append(numpy.array([[0.0,0.0],
                                                          [1.0,0.0],
                                                          [0.0,1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([1.0,0.0,0.0]))

            #3: 2-3-7-6
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],
                                                                  1.0,
                                                                  xBar[1]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0],x[2]]))
            self.boundaryJacobianList.append(numpy.array([[1.0,0.0],
                                                          [0.0,0.0],
                                                          [0.0,1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,1.0,0.0]))

            #4: 3-0-4-7
            self.boundaryMapList.append(lambda xBar: numpy.array([-1.0,
                                                                  xBar[0],
                                                                  xBar[1]]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[1],x[2]]))
            self.boundaryJacobianList.append(numpy.array([[0.0,0.0],
                                                          [1.0,0.0],
                                                          [0.0,1.0]]))
            self.boundaryUnitNormalList.append(numpy.array([-1.0,0.0,0.0]))

            #5: 4-5-6-7
            self.boundaryMapList.append(lambda xBar: numpy.array([xBar[0],
                                                                  xBar[1],
                                                                  1.0]))
            self.boundaryMapInverseList.append(lambda x: numpy.array([x[0],x[1]]))
            self.boundaryJacobianList.append(numpy.array([[1.0,0.0],
                                                          [0.0,1.0],
                                                          [0.0,0.0]]))
            self.boundaryUnitNormalList.append(numpy.array([0.0,0.0,1.0]))
    def onElement(self,xi):
        return (xi >= -1.0).all() and (xi <= 1.0).all()

class LocalFunctionSpace(object):
    """
    Base class for low-dimensional spaces of functions
    on a reference element.

    For example, linear functions on a reference triangle
    """
    def __init__(self,dim=0,referenceElement=None):
        self.dim=dim
        self.range_dim = list(range(dim))
        self.referenceElement = referenceElement
        #the following are lists of functions
        self.basis=[]
        self.basisGradients=[]
        self.basisHessians=[(lambda xi: numpy.zeros((self.referenceElement.dim,self.referenceElement.dim),'d')) for i in range(dim)]
        self.nonzeroHessians=False
        self.basisTrace=[]
        self.basisGradientsTrace=[]

    def defineTraceFunctions(self):
        for ebN in self.referenceElement.range_nElementBoundaries:
            self.basisTrace.append([])
            self.basisGradientsTrace.append([])
        for fi in self.referenceElement.range_nElementBoundaries:
            for si in  range(self.dim):
                self.basisTrace[fi].append(lambda xBar,fiIn=fi,siIn=si:
                                           self.basis[siIn](self.referenceElement.boundaryMapList[fiIn](xBar)))
                self.basisGradientsTrace[fi].append(lambda xBar,fiIn=fi,siIn=si:
                                                    self.basisGradients[siIn](self.referenceElement.boundaryMapList[fiIn](xBar)))

class LinearOnSimplexWithNodalBasis(LocalFunctionSpace):
    """
    First order polynomials on the unit nd-simplex with the nodal basis.

    Nodal basis functions on the reference nd-simplex  (nd <= 3) with
    coordinates xi[0],xi[1],and xi[2]. The basis functions are numbered according to
    the nodes.
    """
    def __init__(self,nd=3):
        self.referenceElement = ReferenceSimplex(nd)
        LocalFunctionSpace.__init__(self,nd+1,self.referenceElement)
        self.gradientList=[]
        for ebN in self.referenceElement.range_nElementBoundaries:
            self.basisTrace.append([])
            self.basisGradientsTrace.append([])
        if nd == 1:
            #0
            self.basis.append(lambda xi: 1. - xi[0])
            self.basisTrace[0].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[1](xBar)))
            self.gradientList.append(numpy.array([-1.0]))
            self.basisGradients.append(lambda xi: self.gradientList[0])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[0])
            #1
            self.basis.append(lambda xi: xi[0])
            self.basisTrace[0].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[1](xBar)))
            self.gradientList.append(numpy.array([1.0]))
            self.basisGradients.append(lambda xi: self.gradientList[1])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[1])
        elif nd == 2:
            #0
            self.basis.append(lambda xi:1 - xi[0] - xi[1])
            self.basisTrace[0].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[2](xBar)))
            self.gradientList.append(numpy.array([-1.0,-1.0]))
            self.basisGradients.append(lambda xi: self.gradientList[0])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[0])
            #1
            self.basis.append(lambda xi: xi[0])
            self.basisTrace[0].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[2](xBar)))
            self.gradientList.append(numpy.array([1.0,0.0]))
            self.basisGradients.append(lambda xi: self.gradientList[1])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[1])
            #2
            self.basis.append(lambda xi: xi[1])
            self.basisTrace[0].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[2](xBar)))
            self.gradientList.append(numpy.array([0.0,1.0]))
            self.basisGradients.append(lambda xi: self.gradientList[2])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[2])
        elif nd == 3:
            #0
            self.basis.append(lambda xi:1 - xi[0] - xi[1] - xi[2])
            self.basisTrace[0].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(numpy.array([-1.0,
                                                    -1.0,
                                                    -1.0]))
            self.basisGradients.append(lambda xi: self.gradientList[0])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[3].append(lambda xBar: self.gradientList[0])
            #1
            self.basis.append(lambda xi: xi[0])
            self.basisTrace[0].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(numpy.array([1.0,
                                                    0.0,
                                                    0.0]))
            self.basisGradients.append(lambda xi: self.gradientList[1])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[3].append(lambda xBar: self.gradientList[1])
            #2
            self.basis.append(lambda xi: xi[1])
            self.basisTrace[0].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(numpy.array([0.0,
                                                    1.0,
                                                    0.0]))
            self.basisGradients.append(lambda xi: self.gradientList[2])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[3].append(lambda xBar: self.gradientList[2])
            #3
            self.basis.append(lambda xi: xi[2])
            self.basisTrace[0].append(lambda xBar: self.basis[3](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[3](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[3](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar: self.basis[3](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(numpy.array([0.0,
                                                    0.0,
                                                    1.0]))
            self.basisGradients.append(lambda xi: self.gradientList[3])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[3])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[3])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[3])
            self.basisGradientsTrace[3].append(lambda xBar: self.gradientList[3])


class LinearOnCubeWithNodalBasis(LocalFunctionSpace):
    """
    First order polynomials on the unit nd-cube with the nodal basis.

    Nodal basis functions on the reference nd-cube  (nd <=3) with
    coordinates xi[0],xi[1],and xi[2]. The basis functions are numbered according to
    the nodes.
    """
    def __init__(self,nd=3):
        self.referenceElement = ReferenceCube(nd)
        LocalFunctionSpace.__init__(self,2**nd,self.referenceElement)
        self.gradientList=[]

        if nd == 1:
            #0
            self.basis.append(lambda xi: 0.5*(1.0 - xi[0]))
            self.gradientList.append(numpy.array([-0.5]))
            self.basisGradients.append(lambda xi: self.gradientList[0])
            #1
            self.basis.append(lambda xi: 0.5*(1.0 + xi[0]))
            self.gradientList.append(numpy.array([0.5]))
            self.basisGradients.append(lambda xi: self.gradientList[1])
        elif nd == 2:
            for node in self.referenceElement.nodeList:
                self.basis.append(lambda xi, n0=node[0],n1=node[1]:0.25*(1.0+n0*xi[0])*(1.0+n1*xi[1]))
                self.basisGradients.append(lambda xi, n0=node[0],n1=node[1]: numpy.array([0.25*     n0       *(1.0+n1*xi[1]),
                                                                                          0.25*(1.0+n0*xi[0])*     n1       ]))
        elif nd == 3:
            for node in self.referenceElement.nodeList:
                self.basis.append         (lambda xi, n0=node[0],n1=node[1],n2=node[2]:              0.125*(1.0+n0*xi[0])*(1.0+n1*xi[1])*(1.0+n2*xi[2]))
                self.basisGradients.append(lambda xi, n0=node[0],n1=node[1],n2=node[2]: numpy.array([0.125*     n0       *(1.0+n1*xi[1])*(1.0+n2*xi[2]),
                                                                                                     0.125*(1.0+n0*xi[0])*     n1       *(1.0+n2*xi[2]),
                                                                                                     0.125*(1.0+n0*xi[0])*(1.0+n1*xi[1])*     n2       ]))

        self.defineTraceFunctions()

class LagrangeOnCubeWithNodalBasis(LocalFunctionSpace):
    """
    Lagrange polynomials on the unit nd-cube with the nodal basis.

    Nodal basis functions on the reference nd-cube  (nd <=3) with
    coordinates xi[0],xi[1],and xi[2]. The basis functions are numbered according to
    the nodes.
    """
    def __init__(self,nd=3, order=2):
        self.referenceElement = ReferenceCube(nd)
        LocalFunctionSpace.__init__(self,(order+1)**nd,self.referenceElement)
        self.gradientList=[]
        self.order = order

        # Generate equi distance nodes for generation of lagrange basis
        # Should use Gauss Labatto points

        self.nodes=[]

        self.quadrature = LobattoEdgeAlt(order=order)
        for i in range(order+1):
            self.nodes.append(self.quadrature.points[i][0] )

        # Define 1D functions using recursion formulas
        self.fun=[]
        self.dfun=[]
        fun  = []
        dfun = []
        dfun2= []
        fc=-1
        fc2=-1
        for a in range(order+1):
            fun .append(lambda xi: 1.0)
            dfun.append(lambda xi: 0.0)
            fc=fc+1
            den  = 1.0
            for b in range(order+1):
                if a!=b:
                    dfun2.append(lambda xi: 1.0)
                    fc2=fc2+1
                    for c in range(order+1):
                        if  a!=c and b!=c:
                            dfun2.append(lambda xi, xb=self.nodes[c],fc2=fc2: dfun2[fc2](xi)*(xi - xb))
                            fc2=fc2+1

                    dfun.append(lambda xi,fc=fc,fc2=fc2: dfun[fc](xi) + dfun2[fc2](xi))

                    fun.append(lambda xi, xb=self.nodes[b],fc=fc:  fun[fc](xi)*(xi - xb))
                    den   = den*(self.nodes[a]-self.nodes[b])
                    fc=fc+1
            self. fun.append(lambda xi,fc=fc, den=den:   fun[fc](xi)/den)
            self.dfun.append(lambda xi,fc=fc, den=den:  dfun[fc](xi)/den)

        # Define multi-dimensional stuff
        basis= []
        basisGradients = []
        if nd == 1:
            basis = self.fun
            basisGradients = self.dfun
            funMap=[0,2,1]
        elif nd == 2:
            for j in range(order+1):
                for i in range(order+1):
                    basis.append(lambda xi,i=i,j=j:self.fun[i](xi[0])*self.fun[j](xi[1]))
                    basisGradients.append(lambda xi,i=i,j=j:numpy.array([self.dfun[i](xi[0])*self. fun[j](xi[1]),
                                                                              self. fun[i](xi[0])*self.dfun[j](xi[1])]))
            funMap=[0,7,3,  4,8,6,   1,5,2]
        elif nd == 3:
            for k in range(order+1):
                for j in range(order+1):
                    for i in range(order+1):
                        basis.append(
                            lambda xi,i=i,j=j,k=k: self.fun[i](xi[0])*self.fun[j](xi[1])*self.fun[k](xi[2]))
                        basisGradients.append(
                            lambda xi,i=i,j=j,k=k: numpy.array([self.dfun[i](xi[0])*self. fun[j](xi[1])*self. fun[k](xi[2]),
                                                                self. fun[i](xi[0])*self.dfun[j](xi[1])*self. fun[k](xi[2]),
                                                                self. fun[i](xi[0])*self. fun[j](xi[1])*self.dfun[k](xi[2])]))

            #funMap=numpy.array([0,3,6,8,18,19,24,26])
            funMap = [ 0, 8, 1,
                      11,20, 9,
                       3,10, 2,
                      12,21,13,
                      24,26,22,
                      15,23,14,
                       4,16, 5,
                      19,25,17,
                       7,18, 6]

        # Reorder local functions
        invMap=numpy.zeros((self.dim),'i')
        for i in range(self.dim):
            invMap[funMap[i]] = i

        for i in range(self.dim):
            self.basis.append(basis[invMap[i]])
            self.basisGradients.append(basisGradients[invMap[i]])
        # Get boundary data
        self.defineTraceFunctions()

class BernsteinOnCube(LocalFunctionSpace):
    """
    Bernstein polynomials on the unit nd-cube

    Bernstein basis functions on the reference nd-cube (nd <=3) with
    coordinates xi[0],xi[1],and xi[2].
    The basis functions are numbered as follows::

        ##############
        # *** 2D *** #
        ##############
        ###########
        # Order=1 #
        ###########
        3--2
        |  |
        0--1
        ###########
        # Order=2 #
        ###########
        3-6-2
        | | |
        7-8-5
        | | |
        0-4-1
        ###########
        # Order=3 #
        ###########
        3--9--8--2
        |  |  |  |
        10-14-15-7
        |  |  |  |
        11-12-13-6
        |  |  |  |
        0--4--5--1
        ###########
        # Order=4 #
        ###########
        3--12-11-10-2
        |  |  |  |  |
        13-22-23-24-9
        |  |  |  |  |
        14-19-20-21-8
        |  |  |  |  |
        15-16-17-18-7
        |  |  |  |  |
        0--4--5--6--1

        NOTE (mql): *** The polynomials are defined for arbitrary order in 1D.
                        This definition is carried to multi-D via tensor products.
                    *** The numbering is defined for arbitrary order (via funMap) in 2D.
                    *** TODO: Generalize numbering for 1D and 3D.
                    *** TODO: The quad rule Lobatto edge alt must be generalized.
                    *** TODO: define hessians and higher order deriavtives
    """
    from math import factorial 

    def nChooseK(self,n,k):
        return factorial(n)/(factorial(k)*factorial(n-k))
    
    def __init__(self,nd=3, order=2):
        self.referenceElement = ReferenceCube(nd)
        LocalFunctionSpace.__init__(self,(order+1)**nd,self.referenceElement)
        self.gradientList=[]
        self.order = order

        # Generate equi distance nodes for generation of basis
        # Should use Gauss Labatto points

        self.nodes=[]

        self.quadrature = LobattoEdgeAlt(order=order)
        for i in range(order+1):
            self.nodes.append(self.quadrature.points[i][0])
        # Define 1D functions using definition via binomial coefficients
        self.fun=[]
        self.dfun=[]
        #self.dfun2=[] 

        for k in range(order+1):
            self.fun.append(lambda x,n=order,k=k:
                            self.nChooseK(n,k)*((x+1)/2.)**k*((1-x)/2.)**(n-k))
            self.dfun.append(lambda x,n=order,k=k:
                             #-2.**(-n)*(1-x)**(-1-k+n)*(1+x)**(k-1)*(-2*k+n+n*x)*self.nChooseK(n,k))
                             # Rule out the cases when 1-x or 1+x = 0. This is to avoid warnings due to division by zero
                             0. if 1-x == 0. or 1+x == 0. else -2.**(-n)*(1-x)**(-1-k+n)*(1+x)**(k-1)*(-2*k+n+n*x)*self.nChooseK(n,k))
            #self.dfun2.append(lambda x,n=order,k=k: ...)

        # Define multi-dimensional stuff
        basis= []
        basisGradients = []
        if nd == 1:
            basis = self.fun
            basisGradients = self.dfun
            funMap=[0,2,1]
        elif nd == 2:
            #Define the basis and its gradient via tensor products
            for j in range(order+1):
                for i in range(order+1):
                    basis.append(lambda xi,i=i,j=j:self.fun[i](xi[0])*self.fun[j](xi[1]))
                    basisGradients.append(lambda xi,i=i,j=j:
                                          numpy.array([self.dfun[i](xi[0])*self. fun[j](xi[1]),
                                                       self. fun[i](xi[0])*self.dfun[j](xi[1])]))
            #Define numbering for arbitrary order (see above)
            funMap=np.zeros((order+1)**2,'i')
            for i in range(order+1):
                #auxiliary vector to construct ith column of funMap
                vec=np.zeros(order+1)
                # COMPUTE BOTTOM AND UPPER ROW OF funMap #
                if (i==0): #left column
                    vec[0]=0
                    vec[order]=3
                elif (i==order): #right column
                    vec[0]=1
                    vec[order]=2
                else: #middle columns                    
                    vec[0]=nd+1+i #bottom row
                    vec[order]=nd+2*order+(order-1)-i #upper row
                # COMPUTE MIDDLE ROWS OF funMap #
                for k in range(1,order):
                    if (i==0): #left column
                        vec[order-k] = nd+3*order-2+k
                    elif (i==order): #right column
                        vec[k] = nd+order+k
                    else: #middle columns
                        vec[k] = (2*nd)*(order-1)+2*nd-1+i+(k-1)*(order-1)
                #print vec
                funMap[i*(order+1):(i+1)*(order+1)]=vec[:]
                #print funMap
        elif nd == 3:
            for k in range(order+1):
                for j in range(order+1):
                    for i in range(order+1):
                        basis.append(lambda xi,i=i,j=j,k=k:
                                     self.fun[i](xi[0])*self.fun[j](xi[1])*self.fun[k](xi[2]))
                        basisGradients.append(lambda xi,i=i,j=j,k=k:
                         numpy.array([self.dfun[i](xi[0])*self. fun[j](xi[1])*self. fun[k](xi[2]),
                                      self. fun[i](xi[0])*self.dfun[j](xi[1])*self. fun[k](xi[2]),
                                      self. fun[i](xi[0])*self. fun[j](xi[1])*self.dfun[k](xi[2])]))
            funMap = [ 0, 8, 1,
                       11,20, 9,
                       3,10, 2,
                       12,21,13,
                       24,26,22,
                       15,23,14,
                       4,16, 5,
                       19,25,17,
                       7,18, 6]

        # Reorder local functions
        invMap=numpy.zeros((self.dim),'i')
        for i in range(self.dim):
            invMap[funMap[i]] = i

        for i in range(self.dim):
            self.basis.append(basis[invMap[i]])
            self.basisGradients.append(basisGradients[invMap[i]])
        # Get boundary data
        self.defineTraceFunctions()
        
class QuadraticOnSimplexWithNodalBasis(LocalFunctionSpace):
    """
    Quadratic polynomials on the unit nd-simplex with the nodal basis.

    Nodal basis functions on the reference nd-simplex (nd <=3) with
    coordinates xi[0],xi[1],and xi[2]. The basis functions are
    numbered according to

    .. math::

    \\psi &= \\lambda_i(2\\lambda_i-1)  0<= i<= d
    \\psi &= 4\\lambda_j\\lambda_k       0<= j < k <= d

    where :math:`\\lambda_i` is the barycentric coordinate associated
    with node i (i.e., it's 1 at node i and zero elsewhere)

    Gradients of shape functions are

    .. math::

     \nabla \\psi_i &= (4\\lambda_i-1)\nabla\\lambda_i   0<= i <= d
     \nabla \\psi_i &= 4\\lambda_k\nabla\\lambda_j + 4\\lambda_j\nabla\\lambda_k \\mbox{for} 0 <= j < k <= d

    In 2d we have

    .. math::

    \\psi_i &= \\lambda_i(2\\lambda_i-1)  0<= i<= 2
    \\psi_3 &= 4\\lambda_0\\lambda_1
    \\psi_4 &= 4\\lambda_1\\lambda_2
    \\psi_5 &= 4\\lambda_0\\lambda_2


    2d numberings for :math:`\\psi`

      2
      |\
      | \
      |  \
      5   4
      |    \
      |     \
      0---3--1

    Note that mesh numbers edges according to the node they are across
    from, so that local dof 3 corresponds to edge 2, local dof 4
    corresponds to edge 0, local dof 5 corresponds to edge 1,

    3d should be

    .. math::

    \\psi_i &= \\lambda_i(2\\lambda_i-1)  0<= i<= 3
    \\psi_4 &= 4\\lambda_0\\lambda_1
    \\psi_5 &= 4\\lambda_1\\lambda_2
    \\psi_6 &= 4\\lambda_2\\lambda_3
    \\psi_7 &= 4\\lambda_0\\lambda_2
    \\psi_8 &= 4\\lambda_1\\lambda_3
    \\psi_9 &= 4\\lambda_0\\lambda_3
    """
    def __init__(self,nd=3):
        from .RefUtils import baryCoords
        from .RefUtils import fact
        from .RefUtils import baryGrads
        from .RefUtils import p2refNodes

        self.referenceElement = ReferenceSimplex(nd)
        LocalFunctionSpace.__init__(self, fact(nd+2)//(2*fact(nd)),
                                    self.referenceElement)
        self.gradientList=[]
        self.basisHessians=[]
        self.nonzeroHessians=True
        for ebN in self.referenceElement.range_nElementBoundaries:
            self.basisTrace.append([])
            self.basisGradientsTrace.append([])
        if nd == 1:
            for i in range(nd+1): #0,1
                self.basis.append(lambda  xi, i=i:
                                  baryCoords['1d'][i](xi)*(2.0*baryCoords['1d'][i](xi)-1.0))
                self.basisTrace[0].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.gradientList.append(lambda xi, i=i:
                                         (4.0*baryCoords['1d'][i](xi)-1.0)*baryGrads['1d'][i])
                self.basisGradients.append(lambda xi, i=i: self.gradientList[i](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i:
                                     self.gradientList[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i:
                                     self.gradientList[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisHessians.append(lambda xi, i=i:
                                              4.0*numpy.outer(baryGrads['1d'][i],baryGrads['1d'][i]))
            #end 0,1
            #2
            self.basis.append(lambda xi: 4.0*baryCoords['1d'][0](xi)*baryCoords['1d'][1](xi))
            self.basisTrace[0].append(lambda xBar:
                                      self.basis[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar:
                                      self.basis[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.gradientList.append(lambda xi:
                                     4.0*baryCoords['1d'][1](xi)*baryGrads['1d'][0]+
                                     4.0*baryCoords['1d'][0](xi)*baryGrads['1d'][1])
            self.basisGradients.append(lambda xi: self.gradientList[2](xi))
            self.basisGradientsTrace[0].append(lambda xBar:
                              self.gradientList[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisGradientsTrace[1].append(lambda xBar:
                              self.gradientList[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisHessians.append(lambda xi:
                                          (4.0*numpy.outer(baryGrads['1d'][0],baryGrads['1d'][1])+
                                           4.0*numpy.outer(baryGrads['1d'][1],baryGrads['1d'][0])))
        elif nd == 2:
            for i in range(nd+1): #0,1,2
                self.basis.append(lambda xi, i=i:
                                  baryCoords['2d'][i](xi)*(2.0*baryCoords['2d'][i](xi)-1.0))
                self.basisTrace[0].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.gradientList.append(lambda xi, i=i:
                                         (4.0*baryCoords['2d'][i](xi)-1.0)*baryGrads['2d'][i])

                self.basisGradients.append(lambda xi, i=i: self.gradientList[i](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisGradientsTrace[2].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisHessians.append(lambda xi, i=i:
                                          4.0*numpy.outer(baryGrads['2d'][i],baryGrads['2d'][i]))

            #end for on 0,1,2
            nsofar=nd+1
            #increment size one
            r=1
            for i in range(2): #(0,1) and (1,2)
                self.basis.append(lambda xi, i=i, r=r:
                                  4.0*baryCoords['2d'][i](xi)*baryCoords['2d'][i+r](xi))
                self.basisTrace[0].append(lambda xBar, i=i, nsofar=nsofar:
                                  self.basis[i+nsofar](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i, nsofar=nsofar:
                                  self.basis[i+nsofar](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, i=i, nsofar=nsofar:
                                  self.basis[i+nsofar](self.referenceElement.boundaryMapList[2](xBar)))
                self.gradientList.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                         4.0*baryCoords['2d'][i+r](xi)*baryGrads['2d'][i]+
                                         4.0*baryCoords['2d'][i](xi)*baryGrads['2d'][i+r])

                self.basisGradients.append(lambda xi, i=i, nsofar=nsofar:
                                           self.gradientList[i+nsofar](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i, nsofar=nsofar:
                             self.gradientList[i+nsofar](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i, nsofar=nsofar:
                             self.gradientList[i+nsofar](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisGradientsTrace[2].append(lambda xBar, i=i, nsofar=nsofar:
                             self.gradientList[i+nsofar](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisHessians.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                          4.0*numpy.outer(baryGrads['2d'][i],baryGrads['2d'][i+r])+
                                          4.0*numpy.outer(baryGrads['2d'][i+r],baryGrads['2d'][i]))

            #end for increment size 1
            nsofar += 2
            #increment size 2
            r=2
            for i in range(1): #(0,2)
                self.basis.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                  4.0*baryCoords['2d'][i](xi)*baryCoords['2d'][i+r](xi))
                self.basisTrace[0].append(lambda xBar, i=i, nsofar=nsofar:
                                  self.basis[i+nsofar](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i, nsofar=nsofar:
                                  self.basis[i+nsofar](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, i=i, nsofar=nsofar:
                                  self.basis[i+nsofar](self.referenceElement.boundaryMapList[2](xBar)))
                self.gradientList.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                         4.0*baryCoords['2d'][i+r](xi)*baryGrads['2d'][i]+
                                         4.0*baryCoords['2d'][i](xi)*baryGrads['2d'][i+r])

                self.basisGradients.append(lambda xi, i=i, nsofar=nsofar:
                                           self.gradientList[i+nsofar](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i, nsofar=nsofar:
                             self.gradientList[i+nsofar](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i, nsofar=nsofar:
                             self.gradientList[i+nsofar](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisGradientsTrace[2].append(lambda xBar, i=i, nsofar=nsofar:
                             self.gradientList[i+nsofar](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisHessians.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                          4.0*numpy.outer(baryGrads['2d'][i],baryGrads['2d'][i+r])+
                                          4.0*numpy.outer(baryGrads['2d'][i+r],baryGrads['2d'][i]))

            #end for increment size 2
            nsofar+=1

        elif nd == 3:
            #mwf TODO check hessian formulas!!
            nsofar=0
            for i in range(nd+1):
                self.basis.append(lambda xi, i=i:
                                  baryCoords['3d'][i](xi)*(2.0*baryCoords['3d'][i](xi)-1.0))

                self.basisTrace[0].append(lambda xBar, nsofar=nsofar:
                                 self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, nsofar=nsofar:
                                 self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, nsofar=nsofar:
                                 self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisTrace[3].append(lambda xBar, nsofar=nsofar:
                                 self.basis[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
                self.gradientList.append(lambda xi, i=i, nsofar=nsofar:
                                         (4.0*baryCoords['3d'][i](xi)-1.0)*baryGrads['3d'][i])

                self.basisGradients.append(lambda xi, nsofar=nsofar:
                                           self.gradientList[nsofar](xi))
                for ib in range(nd+1):
                    self.basisGradientsTrace[ib].append(lambda xBar, nsofar=nsofar, ib=ib:
                         self.gradientList[nsofar](self.referenceElement.boundaryMapList[ib](xBar)))
                #end ib
                self.basisHessians.append(lambda xi, i=i:
                                          4.0*numpy.outer(baryGrads['3d'][i],baryGrads['3d'][i]))
                nsofar += 1
                #end ib
            #end for nd+1

            #no go through increments of size 1,2,3
            #number of combos per increment are dim-incr+1
            for r in range(1,nd+1): #1,2,3
                for i in range(nd-r+1):
                    self.basis.append(lambda xi, i=i, r=r:
                                      4.0*baryCoords['3d'][i](xi)*baryCoords['3d'][i+r](xi))
                    self.basisTrace[0].append(lambda xBar, nsofar=nsofar:
                         self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
                    self.basisTrace[1].append(lambda xBar, nsofar=nsofar:
                         self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
                    self.basisTrace[2].append(lambda xBar, nsofar=nsofar:
                         self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
                    self.basisTrace[3].append(lambda xBar, nsofar=nsofar:
                         self.basis[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
                    self.gradientList.append(lambda xi, i=i, r=r:
                                             4.0*baryCoords['3d'][i+r](xi)*baryGrads['3d'][i]+
                                             4.0*baryCoords['3d'][i](xi)*baryGrads['3d'][i+r])
                    self.basisGradients.append(lambda xi, nsofar=nsofar:
                                               self.gradientList[nsofar](xi))
                    for ib in range(nd+1):
                        self.basisGradientsTrace[ib].append(lambda xBar, nsofar=nsofar, ib=ib:
                             self.gradientList[nsofar](self.referenceElement.boundaryMapList[ib](xBar)))
                    #end ib
                    self.basisHessians.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                              4.0*numpy.outer(baryGrads['3d'][i],baryGrads['3d'][i+r])+
                                              4.0*numpy.outer(baryGrads['3d'][i+r],baryGrads['3d'][i]))
                    nsofar += 1
#             for j in range(1,nd+1): #1,2,3
#                 for i in range(j):
#                     self.basis.append(lambda xi, i=i, j=j:
#                                       4.0*baryCoords['3d'][i](xi)*baryCoords['3d'][j](xi))

#                     self.basisTrace[0].append(lambda xBar, nsofar=nsofar:
#                          self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
#                     self.basisTrace[1].append(lambda xBar, nsofar=nsofar:
#                          self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
#                     self.basisTrace[2].append(lambda xBar, nsofar=nsofar:
#                          self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
#                     self.basisTrace[3].append(lambda xBar, nsofar=nsofar:
#                          self.basis[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
#                     self.gradientList.append(lambda xi, i=i, j=j:
#                                              4.0*baryCoords['3d'][j](xi)*baryGrads['3d'][i]+
#                                              4.0*baryCoords['3d'][i](xi)*baryGrads['3d'][j])

#                     self.basisGradients.append(lambda xi, nsofar=nsofar:
#                                                self.gradientList[nsofar](xi))
#                     for ib in range(nd+1):
#                         self.basisGradientsTrace[ib].append(lambda xBar, nsofar=nsofar, ib=ib:
#                              self.gradientList[nsofar](self.referenceElement.boundaryMapList[ib](xBar)))
#                     #end ib
#                     self.basisHessians.append(lambda xi, i=i, nsofar=nsofar, j=j:
#                                               4.0*numpy.outer(baryGrads['3d'][i],baryGrads['3d'][j])+
#                                               4.0*numpy.outer(baryGrads['3d'][j],baryGrads['3d'][i]))
#                     nsofar += 1
                #end for i
            #end for r
        #end 3d
    #end init
#end QuadraticOnSimplex

class BernsteinOnSimplex(LocalFunctionSpace):
    """
    Quadratic Bernstein polynomials on the unit nd-simplex.

    The basis functions are numbered according to

    .. math::

    \\psi &= \\lambda_i^2               0<= i<= d
    \\psi &= 2\\lambda_j\\lambda_k       0<= j < k <= d

    where :math:`\\lambda_i` is the barycentric coordinate associated
    with node i (i.e., it's 1 at node i and zero elsewhere). 
    These can be generalized to:

    \\psi &= p!/(i!j!k!)\\lambda_0^i\\lambda_1^j\\lambda_2^k, 0<=i,j,k<=order, i+j+k=order

    Gradients of shape functions are

    .. math::

     \nabla \\psi_i &= 2\\lambda_i\nabla\\lambda_i   0<= i <= d
     \nabla \\psi_i &= 2\\lambda_k\nabla\\lambda_j + 2\\lambda_j\nabla\\lambda_k \\mbox{for} 0 <= j < k <= d

    which again can be generalized via the general formula above.

    In 2d we have

    .. math::

    \\psi_i &= \\lambda_i^2  0<= i<= 2
    \\psi_3 &= 2\\lambda_0\\lambda_1
    \\psi_4 &= 2\\lambda_1\\lambda_2
    \\psi_5 &= 2\\lambda_0\\lambda_2

    2d numberings for :math:`\\psi`

      2
      |\
      | \
      |  \
      5   4
      |    \
      |     \
      0---3--1

    Note that mesh numbers edges according to the node they are across
    from, so that local dof 3 corresponds to edge 2, local dof 4
    corresponds to edge 0, local dof 5 corresponds to edge 1,

    3d should be

    .. math::

    \\psi_i &= \\lambda_i^2  0<= i<= 3
    \\psi_4 &= 2\\lambda_0\\lambda_1
    \\psi_5 &= 2\\lambda_1\\lambda_2
    \\psi_6 &= 2\\lambda_2\\lambda_3
    \\psi_7 &= 2\\lambda_0\\lambda_2
    \\psi_8 &= 2\\lambda_1\\lambda_3
    \\psi_9 &= 2\\lambda_0\\lambda_3
    """
    def __init__(self,nd=3):
        from .RefUtils import baryCoords
        from .RefUtils import fact
        from .RefUtils import baryGrads
        from .RefUtils import p2refNodes

        self.referenceElement = ReferenceSimplex(nd)
        LocalFunctionSpace.__init__(self, fact(nd+2)//(2*fact(nd)),
                                    self.referenceElement)
        self.gradientList=[]
        self.basisHessians=[]
        self.nonzeroHessians=True
        for ebN in self.referenceElement.range_nElementBoundaries:
            self.basisTrace.append([])
            self.basisGradientsTrace.append([])
        if nd == 1:
            for i in range(nd+1): #0,1
                self.basis.append(lambda  xi, i=i:
                                  baryCoords['1d'][i](xi)**2.0)
                self.basisTrace[0].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.gradientList.append(lambda xi, i=i:
                                         2.0*baryCoords['1d'][i](xi)*baryGrads['1d'][i])
                self.basisGradients.append(lambda xi, i=i: self.gradientList[i](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i:
                                     self.gradientList[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i:
                                     self.gradientList[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisHessians.append(lambda xi, i=i:
                                          2.0*numpy.outer(baryGrads['1d'][i],baryGrads['1d'][i]))
            #end 0,1
            #2
            self.basis.append(lambda xi: 2.0*baryCoords['1d'][0](xi)*baryCoords['1d'][1](xi))
            self.basisTrace[0].append(lambda xBar:
                                      self.basis[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar:
                                      self.basis[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.gradientList.append(lambda xi:
                                     2.0*baryCoords['1d'][1](xi)*baryGrads['1d'][0]+
                                     2.0*baryCoords['1d'][0](xi)*baryGrads['1d'][1])
            self.basisGradients.append(lambda xi: self.gradientList[2](xi))
            self.basisGradientsTrace[0].append(lambda xBar:
                              self.gradientList[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisGradientsTrace[1].append(lambda xBar:
                              self.gradientList[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisHessians.append(lambda xi:
                                          (2.0*numpy.outer(baryGrads['1d'][0],baryGrads['1d'][1])+
                                           2.0*numpy.outer(baryGrads['1d'][1],baryGrads['1d'][0])))
        elif nd == 2:
            #NOTE (mql): the formulas for the basis functions and its first derivatives are grl
            # with respect to the polynomial order. The Hessian is computed assuming order=2
            ijk = np.array([[2,0,0],                     
                            [0,2,0],
                            [0,0,2],
                            [1,1,0],
                            [0,1,1],
                            [1,0,1]])            
            for i in range(len(ijk)):
                self.basis.append(lambda xi, i=i:
                                  math.factorial(2)/math.factorial(ijk[i][0])/math.factorial(ijk[i][1])/math.factorial(ijk[i][2])*
                                  (1. if ijk[i][0]==0 else baryCoords['2d'][0](xi)**ijk[i][0])*
                                  (1. if ijk[i][1]==0 else baryCoords['2d'][1](xi)**ijk[i][1])*
                                  (1. if ijk[i][2]==0 else baryCoords['2d'][2](xi)**ijk[i][2]))
                self.basisTrace[0].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.gradientList.append(
                    lambda xi, i=i:
                    math.factorial(2)/math.factorial(ijk[i][0])/math.factorial(ijk[i][1])/math.factorial(ijk[i][2])*
                    (
                    baryCoords['2d'][0](xi)**ijk[i][0]*
                    baryCoords['2d'][1](xi)**ijk[i][1]*
                    (0. if baryCoords['2d'][2](xi) == 0. else ijk[i][2]*baryCoords['2d'][2](xi)**(ijk[i][2]-1))
                        *baryGrads['2d'][2]+
                    baryCoords['2d'][0](xi)**ijk[i][0]*
                    (0. if baryCoords['2d'][1](xi) == 0. else ijk[i][1]*baryCoords['2d'][1](xi)**(ijk[i][1]-1))*
                    baryCoords['2d'][2](xi)**ijk[i][2]
                        *baryGrads['2d'][1]+
                    (0. if baryCoords['2d'][0](xi) == 0. else ijk[i][0]*baryCoords['2d'][0](xi)**(ijk[i][0]-1))*
                    baryCoords['2d'][1](xi)**ijk[i][1]*
                    baryCoords['2d'][2](xi)**ijk[i][2]
                        *baryGrads['2d'][0]))
                self.basisGradients.append(lambda xi, i=i: self.gradientList[i](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i:
                                                   self.gradientList[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i:
                                                   self.gradientList[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisGradientsTrace[2].append(lambda xBar, i=i:
                                                   self.gradientList[i](self.referenceElement.boundaryMapList[2](xBar)))
            # COMPUTE HESSIANS. These formulas assume that order=2
            for i in range(nd+1): #0, 1, 2
                self.basisHessians.append(lambda xi, i=i:
                                          2.0*numpy.outer(baryGrads['2d'][i],baryGrads['2d'][i]))
            r=1
            nsofar=nd+1
            for i in range(2): #(0,1) and (1,2)
                self.basisHessians.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                          2.0*numpy.outer(baryGrads['2d'][i],baryGrads['2d'][i+r])+
                                          2.0*numpy.outer(baryGrads['2d'][i+r],baryGrads['2d'][i]))
            nsofar+=2
            r=2
            for i in range(1): #(0,2)
                self.basisHessians.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                          2.0*numpy.outer(baryGrads['2d'][i],baryGrads['2d'][i+r])+
                                          2.0*numpy.outer(baryGrads['2d'][i+r],baryGrads['2d'][i]))

        elif nd == 3:
            # NOTE (mql): the formulas are not general; i.e., I assume order=2
            nsofar=0
            for i in range(nd+1): #0,1,2,3
                self.basis.append(lambda xi, i=i:
                                  baryCoords['3d'][i](xi)**2.0)
                self.basisTrace[0].append(lambda xBar, nsofar=nsofar:
                                 self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, nsofar=nsofar:
                                 self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, nsofar=nsofar:
                                 self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisTrace[3].append(lambda xBar, nsofar=nsofar:
                                 self.basis[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
                self.gradientList.append(lambda xi, i=i, nsofar=nsofar:
                                         2.0*baryCoords['3d'][i](xi)*baryGrads['3d'][i])
                self.basisGradients.append(lambda xi, nsofar=nsofar:
                                           self.gradientList[nsofar](xi))
                for ib in range(nd+1):
                    self.basisGradientsTrace[ib].append(lambda xBar, nsofar=nsofar, ib=ib:
                         self.gradientList[nsofar](self.referenceElement.boundaryMapList[ib](xBar)))
                #end ib
                self.basisHessians.append(lambda xi, i=i:
                                          2.0*numpy.outer(baryGrads['3d'][i],baryGrads['3d'][i]))
                nsofar += 1
                #end ib
            #end for nd+1

            #no go through increments of size 1,2,3
            #number of combos per increment are dim-incr+1
            for r in range(1,nd+1): #1,2,3
                for i in range(nd-r+1):
                    self.basis.append(lambda xi, i=i, r=r:
                                      2.0*baryCoords['3d'][i](xi)*baryCoords['3d'][i+r](xi))
                    self.basisTrace[0].append(lambda xBar, nsofar=nsofar:
                         self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
                    self.basisTrace[1].append(lambda xBar, nsofar=nsofar:
                         self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
                    self.basisTrace[2].append(lambda xBar, nsofar=nsofar:
                         self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
                    self.basisTrace[3].append(lambda xBar, nsofar=nsofar:
                         self.basis[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
                    self.gradientList.append(lambda xi, i=i, r=r:
                                             2.0*baryCoords['3d'][i+r](xi)*baryGrads['3d'][i]+
                                             2.0*baryCoords['3d'][i](xi)*baryGrads['3d'][i+r])
                    self.basisGradients.append(lambda xi, nsofar=nsofar:
                                               self.gradientList[nsofar](xi))
                    for ib in range(nd+1):
                        self.basisGradientsTrace[ib].append(lambda xBar, nsofar=nsofar, ib=ib:
                             self.gradientList[nsofar](self.referenceElement.boundaryMapList[ib](xBar)))
                    #end ib
                    self.basisHessians.append(lambda xi, i=i, nsofar=nsofar, r=r:
                                              2.0*numpy.outer(baryGrads['3d'][i],baryGrads['3d'][i+r])+
                                              2.0*numpy.outer(baryGrads['3d'][i+r],baryGrads['3d'][i]))
                    nsofar += 1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#finite element spaces for P^1 non-conforming approximation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class CrouzeixRaviartWithNodalBasis(LocalFunctionSpace):
    """
    Crouzeix-Raviart element implemented as a Lagrange type element. That is, the
    degrees of freedom are the values at barycenters of faces. Defined for
    the reference nd-simplex  (nd <=3)
    """

    # In 1d, it's identical to the standard P^1 Lagrange element

    # The basis functions are numbered according to the faces, which are
    # in turn numbered according to the node that they are opposite.

    # If :math:`\lambda_i` is the barycentric coordinate that's 1 at node p_i, then
    # the CrR shape functions are

    # .. math::

    # \psi_i( x) = d(1/d - \lambda_i)

    # and is one along :math:`E_i`, the face opposite :math:`p_i`.

    # These relationships hold in physical space or reference space as long as the
    # barycentric coordinates are defined appropriately.

    # In physical space, the barycentric coordinates are defined by

    # .. math::

    #   \lambda_i = 1 - \frac{( x -  p_i)\cdot  n_i}
    #                        {( p_j -  p_i)\cdot  n_i}

    # for :math:`j \neq i,  p_j` is a vertex in :math:`E_i`

    #                                                            |  t
    # On the reference element, in 2d we have                   u| /
    #                                                            |/___
    # .. math::

    # \lambda_0 = 1-s-t, \lambda_1 = s, \lambda_2 = t

    # For 3d the reference coordinates are

    # .. math::

    #   \lambda_0 = 1-s-t-u, \lambda_1 = s, \lambda_2 = t, \lambda_3 = u


    # In 2d, the CrR shape functions on the reference element are

    # .. math::

    # \hat{\psi}_0 =  2(s+t)-1, \hat{\psi}_1 = 1-2s, \hat{\psi}_2 = 1-2t

    # In 3d, the CrR shape functions on the reference element are

    # .. math::

    # \hat{\psi}_0 =  3(s+t+u)-2, \hat{\psi}_1 = 1-3s, \hat{\psi}_2 = 1-3t,
    #      \hat{\psi}_3 = 1-3u

    # The gradients are obviously constant on an element
    def __init__(self,nd=3):
        self.referenceElement = ReferenceSimplex(nd)
        LocalFunctionSpace.__init__(self,nd+1,self.referenceElement)
        self.gradientList=[]
        for ebN in self.referenceElement.range_nElementBoundaries:
            self.basisTrace.append([])
            self.basisGradientsTrace.append([])
        if nd == 1:
            #note these are the reverse ordering for standard C0 basis
            #0
            self.basis.append(lambda xi:xi[0])
            self.basisTrace[0].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[1](xBar)))
            self.gradientList.append(numpy.array([1.0]))
            self.basisGradients.append(lambda xi: self.gradientList[0])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[0])
            #1
            self.basis.append(lambda xi:1 - xi[0])
            self.basisTrace[0].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[1](xBar)))
            self.gradientList.append(numpy.array([-1.0]))
            self.basisGradients.append(lambda xi: self.gradientList[1])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[1])
        elif nd == 2:
            #0
            self.basis.append(lambda xi:2.*(xi[0] + xi[1])-1.)
            self.basisTrace[0].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[2](xBar)))
            self.gradientList.append(numpy.array([2.0,2.0]))
            self.basisGradients.append(lambda xi: self.gradientList[0])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[0])
            #1
            self.basis.append(lambda xi: 1.0-2.0*xi[0])
            self.basisTrace[0].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[2](xBar)))
            self.gradientList.append(numpy.array([-2.0,0.0]))
            self.basisGradients.append(lambda xi: self.gradientList[1])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[1])
            #2
            self.basis.append(lambda xi: 1.0-2*xi[1])
            self.basisTrace[0].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[2](xBar)))
            self.gradientList.append(numpy.array([0.0,-2.0]))
            self.basisGradients.append(lambda xi: self.gradientList[2])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[2])
        elif nd == 3:
            #0
            self.basis.append(lambda xi:3.0*(xi[0] + xi[1] + xi[2])-2.0)
            self.basisTrace[0].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar: self.basis[0](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(numpy.array([3.0,
                                                    3.0,
                                                    3.0]))
            self.basisGradients.append(lambda xi: self.gradientList[0])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[0])
            self.basisGradientsTrace[3].append(lambda xBar: self.gradientList[0])
            #1
            self.basis.append(lambda xi: 1.0-3.0*xi[0])
            self.basisTrace[0].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar: self.basis[1](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(numpy.array([-3.0,
                                                     0.0,
                                                     0.0]))
            self.basisGradients.append(lambda xi: self.gradientList[1])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[1])
            self.basisGradientsTrace[3].append(lambda xBar: self.gradientList[1])
            #2
            self.basis.append(lambda xi: 1.0-3.0*xi[1])
            self.basisTrace[0].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar: self.basis[2](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(numpy.array([ 0.0,
                                                    -3.0,
                                                     0.0]))
            self.basisGradients.append(lambda xi: self.gradientList[2])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[2])
            self.basisGradientsTrace[3].append(lambda xBar: self.gradientList[2])
            #3
            self.basis.append(lambda xi:1.0-3.0*xi[2])
            self.basisTrace[0].append(lambda xBar: self.basis[3](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar: self.basis[3](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar: self.basis[3](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar: self.basis[3](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(numpy.array([ 0.0,
                                                     0.0,
                                                    -3.0]))
            self.basisGradients.append(lambda xi: self.gradientList[3])
            self.basisGradientsTrace[0].append(lambda xBar: self.gradientList[3])
            self.basisGradientsTrace[1].append(lambda xBar: self.gradientList[3])
            self.basisGradientsTrace[2].append(lambda xBar: self.gradientList[3])
            self.basisGradientsTrace[3].append(lambda xBar: self.gradientList[3])

class p0(LocalFunctionSpace):
    def __init__(self,nd=3):
        self.referenceElement = ReferenceSimplex(nd)
        LocalFunctionSpace.__init__(self,1,self.referenceElement)
        self.basis.append(lambda xi: 1.0)
        self.basisGradients.append(lambda xi,nd=nd: numpy.zeros((nd),'d'))

class Monomials(LocalFunctionSpace):
    def __init__(self,nd=3,k=0):
        self.kOrder = k
        basis=[]
        basisGradients=[]
        for a in range(1+k):
            alpha = max(0.0,a-1.0)
            if nd > 1:
                for b in range(1+k-a):
                    beta = max(0.0,b-1.0)
                    if nd > 2:
                        for c in range(1+k-a-b):
                            gamma = max(0.0,c-1.0)
                            basis.append(lambda xi,a=a,b=b,c=c: xi[0]**a * xi[1]**b * xi[2]**c)
                            basisGradients.append(lambda xi,a=a,b=b,c=c,alpha=alpha,beta=beta,gamma=gamma:
                                                  numpy.array([a*xi[0]**alpha* xi[1]**b * xi[2]**c,
                                                                 xi[0]**a * b*xi[1]**beta * xi[2]**c,
                                                                 xi[0]**a * xi[1]**b * c*xi[2]**gamma]))
                    else:
                        basis.append(lambda xi,a=a,b=b: xi[0]**a * xi[1]**b)
                        basisGradients.append(lambda xi,a=a,b=b,alpha=alpha,beta=beta:
                                              numpy.array([a*xi[0]**alpha* xi[1]**b,
                                                             xi[0]**a * b*xi[1]**beta]))
            else:
                basis.append(lambda xi,a=a: xi[0]**a)
                basisGradients.append(lambda xi,a=a,alpha=alpha: numpy.array([a*xi[0]**alpha]))
        self.referenceElement = ReferenceSimplex(nd)
        LocalFunctionSpace.__init__(self,len(basis),self.referenceElement)
        self.basis=basis
        self.basisGradients=basisGradients


class P1BubblesWithNodalBasis(LocalFunctionSpace):
    """
    First order polynomials on the unit nd-simplex with the nodal basis + bubble

    .. math::

    b = (n_d+1)^{n_d+1} \\Pi_{i=0}^{n_d} \\lambda_i

    First nd+1 basis functions are nodal ones on the reference nd-simplex  (nd <=3) with
    coordinates xi[0],xi[1],and xi[2]. The basis functions are numbered according to
    the nodes. The last shape function is the bubble, b
    """
    def __init__(self,nd=3):
        from .RefUtils import baryCoords
        from .RefUtils import baryGrads

        self.referenceElement = ReferenceSimplex(nd)
        LocalFunctionSpace.__init__(self,nd+2,self.referenceElement)
        self.gradientList=[]
        for ebN in self.referenceElement.range_nElementBoundaries:
            self.basisTrace.append([])
            self.basisGradientsTrace.append([])
        self.basisHessians = []
        if nd == 1:
            for i in range(nd+1): #0,1
                self.basis.append(lambda  xi, i=i: baryCoords['1d'][i](xi))
                self.basisTrace[0].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.gradientList.append(lambda xi, i=i: baryGrads['1d'][i])
                self.basisGradients.append(lambda xi, i=i: self.gradientList[i](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i:
                                     self.gradientList[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i:
                                     self.gradientList[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisHessians.append(lambda xi, i=i:numpy.zeros((nd,nd),'d'))
            #end 0,1
            #2
            self.basis.append(lambda xi: 4.0*baryCoords['1d'][0](xi)*baryCoords['1d'][1](xi))
            self.basisTrace[0].append(lambda xBar:
                                      self.basis[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar:
                                      self.basis[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.gradientList.append(lambda xi:
                                     4.0*baryCoords['1d'][1](xi)*baryGrads['1d'][0]+
                                     4.0*baryCoords['1d'][0](xi)*baryGrads['1d'][1])
            self.basisGradients.append(lambda xi: self.gradientList[2](xi))
            self.basisGradientsTrace[0].append(lambda xBar:
                                               self.gradientList[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisGradientsTrace[1].append(lambda xBar:
                                               self.gradientList[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisHessians.append(lambda xi:
                                      (4.0*numpy.outer(baryGrads['1d'][0],baryGrads['1d'][1])+
                                       4.0*numpy.outer(baryGrads['1d'][1],baryGrads['1d'][0])))
        elif nd == 2:
            for i in range(nd+1): #0,1,2
                self.basis.append(lambda xi, i=i: baryCoords['2d'][i](xi))
                self.basisTrace[0].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.gradientList.append(lambda xi, i=i: baryGrads['2d'][i])

                self.basisGradients.append(lambda xi, i=i: self.gradientList[i](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisGradientsTrace[2].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisHessians.append(lambda xi, i=i:numpy.zeros((nd,nd),'d'))

            #end for on 0,1,2
            #3
            nsofar = 3
            self.basis.append(lambda xi: 9.0*baryCoords['2d'][0](xi)*baryCoords['2d'][1](xi)*baryCoords['2d'][2](xi))
            self.basisTrace[0].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
            self.gradientList.append(lambda xi:
                                     9.0*baryCoords['2d'][1](xi)*baryCoords['2d'][2](xi)*baryGrads['2d'][0]+
                                     9.0*baryCoords['2d'][0](xi)*baryCoords['2d'][2](xi)*baryGrads['2d'][1]+
                                     9.0*baryCoords['2d'][0](xi)*baryCoords['2d'][1](xi)*baryGrads['2d'][2])

            self.basisGradients.append(lambda xi:
                                       self.gradientList[nsofar](xi))
            self.basisGradientsTrace[0].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisGradientsTrace[1].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisGradientsTrace[2].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisHessians.append(lambda xi:
                                      9.0*numpy.outer(baryGrads['2d'][0],baryGrads['2d'][1])*baryCoords['2d'][2](xi)+
                                      9.0*numpy.outer(baryGrads['2d'][0],baryGrads['2d'][2])*baryCoords['2d'][1](xi)+
                                      9.0*numpy.outer(baryGrads['2d'][1],baryGrads['2d'][0])*baryCoords['2d'][2](xi)+
                                      9.0*numpy.outer(baryGrads['2d'][1],baryGrads['2d'][2])*baryCoords['2d'][0](xi)+
                                      9.0*numpy.outer(baryGrads['2d'][2],baryGrads['2d'][0])*baryCoords['2d'][1](xi)+
                                      9.0*numpy.outer(baryGrads['2d'][2],baryGrads['2d'][1])*baryCoords['2d'][0](xi))


        elif nd == 3:
            #mwf TODO check hessian formulas!!
            for i in range(nd+1):
                self.basis.append(lambda xi, i=i: baryCoords['3d'][i](xi))
                self.basisTrace[0].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisTrace[3].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[3](xBar)))
                self.gradientList.append(lambda xi, i=i: baryGrads['3d'][i])

                self.basisGradients.append(lambda xi, i=i:
                                           self.gradientList[i](xi))
                for ib in range(nd+1):
                    self.basisGradientsTrace[ib].append(lambda xBar, i=i, ib=ib:
                                                        self.gradientList[i](self.referenceElement.boundaryMapList[ib](xBar)))
                #end ib
                self.basisHessians.append(lambda xi: numpy.zeros((nd,nd),'d'))
            #end for nd+1
            nsofar = 4
            self.basis.append(lambda xi:
                              256.0*baryCoords['3d'][0](xi)*baryCoords['3d'][1](xi)*baryCoords['3d'][2](xi)*baryCoords['3d'][3](xi))
            self.basisTrace[0].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(lambda xi:
                                     256.0*baryCoords['3d'][1](xi)*baryCoords['3d'][2](xi)*baryCoords['3d'][3](xi)*baryGrads['3d'][0]+
                                     256.0*baryCoords['3d'][0](xi)*baryCoords['3d'][2](xi)*baryCoords['3d'][3](xi)*baryGrads['3d'][1]+
                                     256.0*baryCoords['3d'][0](xi)*baryCoords['3d'][1](xi)*baryCoords['3d'][3](xi)*baryGrads['3d'][2]+
                                     256.0*baryCoords['3d'][0](xi)*baryCoords['3d'][1](xi)*baryCoords['3d'][2](xi)*baryGrads['3d'][3])

            self.basisGradients.append(lambda xi:
                                       self.gradientList[nsofar](xi))
            self.basisGradientsTrace[0].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisGradientsTrace[1].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisGradientsTrace[2].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisGradientsTrace[3].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
            self.basisHessians.append(lambda xi:
                                      256.0*numpy.outer(baryGrads['3d'][0],baryGrads['3d'][1])*baryCoords['3d'][2](xi)*baryCoords['3d'][3](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][0],baryGrads['3d'][2])*baryCoords['3d'][1](xi)*baryCoords['3d'][3](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][0],baryGrads['3d'][3])*baryCoords['3d'][1](xi)*baryCoords['3d'][2](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][1],baryGrads['3d'][0])*baryCoords['3d'][2](xi)*baryCoords['3d'][3](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][1],baryGrads['3d'][2])*baryCoords['3d'][0](xi)*baryCoords['3d'][3](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][1],baryGrads['3d'][3])*baryCoords['3d'][0](xi)*baryCoords['3d'][2](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][2],baryGrads['3d'][0])*baryCoords['3d'][1](xi)*baryCoords['3d'][3](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][2],baryGrads['3d'][1])*baryCoords['3d'][0](xi)*baryCoords['3d'][3](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][2],baryGrads['3d'][3])*baryCoords['3d'][0](xi)*baryCoords['3d'][1](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][3],baryGrads['3d'][0])*baryCoords['3d'][1](xi)*baryCoords['3d'][2](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][3],baryGrads['3d'][1])*baryCoords['3d'][0](xi)*baryCoords['3d'][2](xi)+
                                      256.0*numpy.outer(baryGrads['3d'][3],baryGrads['3d'][2])*baryCoords['3d'][0](xi)*baryCoords['3d'][1](xi))


class P1P0BubblesWithNodalBasis(LocalFunctionSpace):
    """
    First order polynomials on the unit nd-simplex with the nodal basis + dg bubble

    .. math::

    b_e = 1 for x in \\Omega_e

    First nd+1 basis functions are nodal ones on the reference nd-simplex  (nd <=3) with
    coordinates xi[0],xi[1],and xi[2]. The basis functions are numbered according to
    the nodes. The last shape function is the bubble, b
    """
    def __init__(self,nd=3):
        from .RefUtils import baryCoords
        from .RefUtils import baryGrads

        self.referenceElement = ReferenceSimplex(nd)
        LocalFunctionSpace.__init__(self,nd+2,self.referenceElement)
        self.gradientList=[]
        for ebN in self.referenceElement.range_nElementBoundaries:
            self.basisTrace.append([])
            self.basisGradientsTrace.append([])
        self.basisHessians = []
        if nd == 1:
            for i in range(nd+1): #0,1
                self.basis.append(lambda  xi, i=i: baryCoords['1d'][i](xi))
                self.basisTrace[0].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.gradientList.append(lambda xi, i=i: baryGrads['1d'][i])
                self.basisGradients.append(lambda xi, i=i: self.gradientList[i](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i:
                                     self.gradientList[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i:
                                     self.gradientList[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisHessians.append(lambda xi, i=i:numpy.zeros((nd,nd),'d'))
            #end 0,1
            #2
            self.basis.append(lambda xi: 1.0)
            self.basisTrace[0].append(lambda xBar:
                                      self.basis[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar:
                                      self.basis[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.gradientList.append(lambda xi: 0.0)
            self.basisGradients.append(lambda xi: self.gradientList[2](xi))
            self.basisGradientsTrace[0].append(lambda xBar:
                                               self.gradientList[2](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisGradientsTrace[1].append(lambda xBar:
                                               self.gradientList[2](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisHessians.append(lambda xi: 0.0)
        elif nd == 2:
            for i in range(nd+1): #0,1,2
                self.basis.append(lambda xi, i=i: baryCoords['2d'][i](xi))
                self.basisTrace[0].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, i=i:
                                        self.basis[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.gradientList.append(lambda xi, i=i: baryGrads['2d'][i])

                self.basisGradients.append(lambda xi, i=i: self.gradientList[i](xi))
                self.basisGradientsTrace[0].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisGradientsTrace[1].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisGradientsTrace[2].append(lambda xBar, i=i:
                                        self.gradientList[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisHessians.append(lambda xi, i=i:numpy.zeros((nd,nd),'d'))

            #end for on 0,1,2
            #bubble
            nsofar = 3
            self.basis.append(lambda xi: 1.0)
            self.basisTrace[0].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
            self.gradientList.append(lambda xi:numpy.array([0.0,0.0]))

            self.basisGradients.append(lambda xi:
                                       self.gradientList[nsofar](xi))
            self.basisGradientsTrace[0].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisGradientsTrace[1].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisGradientsTrace[2].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisHessians.append(lambda xi:numpy.zeros((nd,nd),'d'))


        elif nd == 3:
            #mwf TODO check hessian formulas!!
            for i in range(nd+1):
                self.basis.append(lambda xi, i=i: baryCoords['3d'][i](xi))
                self.basisTrace[0].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[0](xBar)))
                self.basisTrace[1].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[1](xBar)))
                self.basisTrace[2].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[2](xBar)))
                self.basisTrace[3].append(lambda xBar, i=i:
                                          self.basis[i](self.referenceElement.boundaryMapList[3](xBar)))
                self.gradientList.append(lambda xi, i=i: baryGrads['3d'][i])

                self.basisGradients.append(lambda xi, i=i:
                                           self.gradientList[i](xi))
                for ib in range(nd+1):
                    self.basisGradientsTrace[ib].append(lambda xBar, i=i, ib=ib:
                                                        self.gradientList[i](self.referenceElement.boundaryMapList[ib](xBar)))
                #end ib
                self.basisHessians.append(lambda xi: numpy.zeros((nd,nd),'d'))
            #end for nd+1
            nsofar = 4
            self.basis.append(lambda xi: 1.0)
            self.basisTrace[0].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisTrace[1].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisTrace[2].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisTrace[3].append(lambda xBar:
                                      self.basis[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
            self.gradientList.append(lambda xi: numpy.array([0.0,0.0,0.0]))
            self.basisGradients.append(lambda xi:
                                       self.gradientList[nsofar](xi))
            self.basisGradientsTrace[0].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[0](xBar)))
            self.basisGradientsTrace[1].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[1](xBar)))
            self.basisGradientsTrace[2].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[2](xBar)))
            self.basisGradientsTrace[3].append(lambda xBar:
                                               self.gradientList[nsofar](self.referenceElement.boundaryMapList[3](xBar)))
            self.basisHessians.append(lambda xi: numpy.zeros((nd,nd),'d'))

"""
Interpolation conditions.
"""

class InterpolationConditions(object):
    """
    Base class for generalized interpolation conditions
    for function spaces.

    For example, a function's values at a set of points
    that is large enough to uniquely specify an element
    of the local function space that will be paired with
    the interpolation conditions.
    """
    def __init__(self,dim=0,referenceElement=None):
        self.dim=dim
        self.range_dim = list(range(dim))
        self.referenceElement=referenceElement
        self.functionals=[]
        self.functionalsQuadrature=[]
        self.quadraturePointArray=[]
        self.nQuadraturePoints=0
    def quadrature2DOF_element(self,k):
        return k
    #mwf find out if local element boundary interpolation point
    #is "on" local face ebN_local, if this idea makes sense
    #None means not defined
    def definedOnlocalElementBoundary(self,k,ebN_local):
        return None #doesn't have one
    #mwf map interpolation conditions to local enumeration of mesh nodes (vertices)
    #returns None if no correspondence
    def quadrature2Node_element(self,k):
        return None
    #optimized projection routine
    projectFiniteElementFunctionFromInterpolationConditions_opt = None
class NodalInterpolationConditions(InterpolationConditions):
    """
    Obtains the DOF from the function values at the nodes
    """
    def __init__(self,referenceElement):
        InterpolationConditions.__init__(self,referenceElement.nNodes,referenceElement)
        self.quadraturePointArray = numpy.zeros((len(referenceElement.nodeList),3),'d')
        for k,n in enumerate(referenceElement.nodeList):
            for I in range(referenceElement.dim):
                self.quadraturePointArray[k,I]=n[I]
        self.nQuadraturePoints = self.quadraturePointArray.shape[0]
        #self.functionals.append(lambda f: f(referenceElement.nodeList[0]))
        #self.functionalsQuadrature.append(lambda fList: fList[0])
        #assert referenceElement.nNodes < 6,"Haven't implemented %d nodes for nodal interpolation conditions" % (referenceElement.nNodes,)
        for ni in referenceElement.range_nNodes:
            self.functionals.append(lambda f,n=ni: f(referenceElement.nodeList[n]))
            self.functionalsQuadrature.append(lambda fList,n=ni: fList[n])
        #for c based projection from interpolation conditions
        self.functionals_quadrature_map = numpy.arange(len(self.functionalsQuadrature),dtype='i')

    def definedOnLocalElementBoundary(self,k,ebN_local):
        #interpolation points indexed like nodeList so nodes are not on
        #the local element across from them
        return k < self.nQuadraturePoints and k != ebN_local
    def quadrature2Node_element(self,k):
        return k

    def projectFiniteElementFunctionFromInterpolationConditions_opt(self,finiteElementFunction,interpolationValues):
        """
        Allow the interpolation conditions to control projection of a (global) finite element function from
        an array of interpolation values in order to take advantage of specific structure, otherwise
        can just use functionals interface
        """
        cfemIntegrals.projectFromNodalInterpolationConditions(finiteElementFunction.dim_dof,
                                                              finiteElementFunction.femSpace.dofMap.l2g,
                                                              self.functionals_quadrature_map,
                                                              interpolationValues,
                                                              finiteElementFunction.dof)

class CubeNodalInterpolationConditions(NodalInterpolationConditions):
    from .RefUtils import quadrilateralLocalBoundaryLookup
    from .RefUtils import hexahedronLocalBoundaryLookup
    def __init__(self,referenceElement):
        NodalInterpolationConditions.__init__(self,referenceElement)
    def definedOnLocalElementBoundary(self,k,ebN_local):
        #overide this  one function from nodal interpolation conditions,
        #which only holds for simplex
        #return k < self.nQuadraturePoints and k != ebN_local
        if self.referenceElement.dim == 1:
            return k != ebN_local
        elif self.referenceElement.dim == 2:
            return ebN_local in self.quadrilateralLocalBoundaryLookup[k]
        elif self.referenceElement.dim == 3:
            return ebN_local in self.hexahedronLocalBoundaryLookup[k]

class QuadraticLagrangeNodalInterpolationConditions(InterpolationConditions):
    """
    Obtains the DOF from the function values at vertices and
    midpoints of edges (whole element is considered an edge in 1d)
    """
    from .RefUtils import p2tetrahedronLocalBoundaryLookup
    from math import fmod
    def __init__(self,referenceElement):
        from .RefUtils import fact
        from .RefUtils import p2refNodes
        sdim  = referenceElement.dim
        self.nInterpNodes= fact(2+sdim)//(2*fact(sdim))
        InterpolationConditions.__init__(self,self.nInterpNodes,referenceElement)
        self.quadraturePointArray = numpy.zeros((self.nInterpNodes,3),'d')
        for k in range(self.nInterpNodes):
            for I in range(sdim):
                self.quadraturePointArray[k,I] = p2refNodes[sdim-1][k,I]
        #self.nQuadraturePoints = len(self.quadraturePointArray)
        self.nQuadraturePoints = self.quadraturePointArray.shape[0]
        for i in range(self.nQuadraturePoints):
            self.functionals.append(lambda f,i=i: f(self.quadraturePointArray[i,:]))
            self.functionalsQuadrature.append(lambda fList, i=i: fList[i])
        #end for
        #for c based projection from interpolation conditions
        self.functionals_quadrature_map = numpy.arange(len(self.functionalsQuadrature),dtype='i')
   #end init
    def definedOnLocalElementBoundary(self,k,ebN_local):
        if k <= self.referenceElement.dim:
            return k != ebN_local
        if self.referenceElement.dim == 2:
            i = int(fmod(k-3 + 2,3))
            return i == ebN_local
        if self.referenceElement.dim == 3:
            return ebN_local in self.p2tetrahedronLocalBoundaryLookup[k]
        return False
    def quadrature2Node_element(self,k):
        if k <= self.referenceElement.dim:
            return k
        else:
            return None
    def projectFiniteElementFunctionFromInterpolationConditions_opt(self,finiteElementFunction,interpolationValues):
        """
        Allow the interpolation conditions to control projection of a (global) finite element function from
        an array of interpolation values in order to take advantage of specific structure, otherwise
        can just use functionals interface
        """
        cfemIntegrals.projectFromNodalInterpolationConditions(finiteElementFunction.dim_dof,
                                                              finiteElementFunction.femSpace.dofMap.l2g,
                                                              self.functionals_quadrature_map,
                                                              interpolationValues,
                                                              finiteElementFunction.dof)

class QuadraticLagrangeCubeNodalInterpolationConditions(InterpolationConditions):
    """
    Obtains the DOF from the function values at vertices and
    midpoints of edges (whole element is considered an edge in 1d)
    """
    from .RefUtils import q2quadrilateralLocalBoundaryLookup
    from .RefUtils import q2hexahedronLocalBoundaryLookup
    from math import fmod
    def __init__(self,referenceElement):
        from .RefUtils import fact
        from .RefUtils import q2refNodes
        sdim  = referenceElement.dim
        if sdim==2:
            self.nInterpNodes = 9
        elif sdim==3:
            self.nInterpNodes = 27
        InterpolationConditions.__init__(self,self.nInterpNodes,referenceElement)
        self.quadraturePointArray = numpy.zeros((self.nInterpNodes,3),'d')
        #self.nQuadraturePoints = len(self.quadraturePointArray)
        if sdim==2:
            for k in range(self.nInterpNodes):
                for I in range(sdim):
                    self.quadraturePointArray[k,I] = q2refNodes[1][k,I]
        elif sdim==3:
            for k in range(self.nInterpNodes):
                for I in range(sdim):
                    self.quadraturePointArray[k,I] = q2refNodes[2][k,I]
        self.nQuadraturePoints = len(self.quadraturePointArray)
        self.nQuadraturePoints = self.quadraturePointArray.shape[0]
        for i in range(self.nQuadraturePoints):
            self.functionals.append(lambda f,i=i: f(self.quadraturePointArray[i,:]))
            self.functionalsQuadrature.append(lambda fList, i=i: fList[i])
        #end for
        #for c based projection from interpolation conditions
        self.functionals_quadrature_map = numpy.arange(len(self.functionalsQuadrature),dtype='i')
   #end init
    def definedOnLocalElementBoundary(self,k,ebN_local):
        if self.referenceElement.dim == 1:
            if k <= self.referenceElement.dim:
                return k != ebN_local
        elif self.referenceElement.dim == 2:
            return ebN_local in self.q2quadrilateralLocalBoundaryLookup[k]
        elif self.referenceElement.dim == 3:
            return ebN_local in self.q2hexahedronLocalBoundaryLookup[k]
        else:
            return False
    def quadrature2Node_element(self,k):
        if k <= self.referenceElement.dim**2:
                return k
        else:
            return None
    def projectFiniteElementFunctionFromInterpolationConditions_opt(self,finiteElementFunction,interpolationValues):
        """
        Allow the interpolation conditions to control projection of a (global) finite element function from
        an array of interpolation values in order to take advantage of specific structure, otherwise
        can just use functionals interface
        """
        cfemIntegrals.projectFromNodalInterpolationConditions(finiteElementFunction.dim_dof,
                                                              finiteElementFunction.femSpace.dofMap.l2g,
                                                              self.functionals_quadrature_map,
                                                              interpolationValues,
                                                              finiteElementFunction.dof)
        interpolationValues = finiteElementFunction.dof

#end interp conditions

class FaceBarycenterInterpolationConditions(InterpolationConditions):
    """
    Obtains the DOF from the function values at the barycenter of faces
    """
    def __init__(self,referenceElement):
        InterpolationConditions.__init__(self,referenceElement.nElementBoundaries,referenceElement)
        self.quadraturePointArray = numpy.zeros((referenceElement.nElementBoundaries,3),'d')
        #put in crude switch on number of dimensions for now?
        self.ebaryList = []
        ebary = EVec(0.0,0.0,0.0)
        if referenceElement.dim == 2: #2d, interpolation points are edge barycenters
            ebary = EVec(0.5,0.0,0.0)
        elif referenceElement.dim == 3: #3d, interpolation points are face barycenters
            ebary = EVec(1.0/3.0,1.0/3.0,0.0)
        #end
        for k in referenceElement.range_nElementBoundaries:
            for I in range(referenceElement.dim):
                self.quadraturePointArray[k,I]=referenceElement.boundaryMapList[k](ebary)[I]
            ntmp   = Node()
            ntmp.p = self.quadraturePointArray[k,:]
            self.ebaryList.append(ntmp)
        #end k loop
        #end loop through boundary maps
        self.nQuadraturePoints = self.quadraturePointArray.shape[0]
        self.functionals.append(lambda f: f(self.ebaryList[0]))
        self.functionalsQuadrature.append(lambda fList: fList[0])
        assert referenceElement.nElementBoundaries < 6,"Haven't implemented %d faces with nodal interpolation conditions" % (referenceElement.nElementBoundaries,)
        if referenceElement.nElementBoundaries > 1:
            self.functionals.append(lambda f: f(self.ebaryList[1]))
            self.functionalsQuadrature.append(lambda fList: fList[1])
        if referenceElement.nElementBoundaries > 2:
            self.functionals.append(lambda f: f(self.ebaryList[2]))
            self.functionalsQuadrature.append(lambda fList: fList[2])
        if referenceElement.nElementBoundaries > 3:
            self.functionals.append(lambda f: f(self.ebaryList[3]))
            self.functionalsQuadrature.append(lambda fList: fList[3])
        if referenceElement.nElementBoundaries > 4:
            self.functionals.append(lambda f: f(self.ebaryList[4]))
            self.functionalsQuadrature.append(lambda fList: fList[4])
        #for c based projection from interpolation conditions
        self.functionals_quadrature_map = numpy.arange(len(self.functionalsQuadrature),dtype='i')


    #end init
    def definedOnLocalElementBoundary(self,k,ebN_local):
        #interpolation conditions indexed like locak boundaries
        return k == ebN_local
    def projectFiniteElementFunctionFromInterpolationConditions_opt(self,finiteElementFunction,interpolationValues):
        """
        Allow the interpolation conditions to control projection of a (global) finite element function from
        an array of interpolation values in order to take advantage of specific structure, otherwise
        can just use functionals interface
        """
        cfemIntegrals.projectFromNodalInterpolationConditions(finiteElementFunction.dim_dof,
                                                              finiteElementFunction.femSpace.dofMap.l2g,
                                                              self.functionals_quadrature_map,
                                                              interpolationValues,
                                                              finiteElementFunction.dof)

#end interp conditions

class p0InterpolationConditions(InterpolationConditions):
    from . import Quadrature
    """
    Obtains the DOF from the function values at the nodes
    """
    def __init__(self,referenceElement):
        self.quadrature=self.Quadrature.SimplexLobattoQuadrature(referenceElement.dim,1)
        InterpolationConditions.__init__(self,1,referenceElement)
        self.quadraturePointArray = numpy.zeros((len(self.quadrature.weights),3),'d')
        for k,p in enumerate(self.quadrature.points):
            for I in range(referenceElement.dim):
                self.quadraturePointArray[k,I]=p[I]
        self.nQuadraturePoints = self.quadraturePointArray.shape[0]
        self.vol=sum([w for  w in  self.quadrature.weights])
        self.functionals.append(lambda f: sum([w*f(p) for  w,p in zip(self.quadrature.weights,self.quadrature.points)])/self.vol)
        self.functionalsQuadrature.append(lambda fList: sum([w*f for w,f in zip(self.quadrature.weights,fList)])/self.vol)
    def quadrature2DOF_element(self,k):
        return 0
    def definedOnLocalElementBoundary(self,k,ebN_local):
        #have choice are interior nodal conditions for dg on all boundaries or none?
        return True #if matching quadrature2DOF_element behavior,

class MonomialInterpolationConditions(InterpolationConditions):
    from . import Quadrature
    """
    Obtains the DOF from the function values at the nodes
    """
    def __init__(self,referenceElement,monomialSpace):
        from . import LinearSolvers
        self.quadrature=self.Quadrature.SimplexGaussQuadrature(referenceElement.dim,max(monomialSpace.kOrder*2,1))
        InterpolationConditions.__init__(self,monomialSpace.dim,referenceElement)
        self.quadraturePointArray = numpy.zeros((len(self.quadrature.weights),3),'d')
        for k,p in enumerate(self.quadrature.points):
            for I in range(referenceElement.dim):
                self.quadraturePointArray[k,I]=p[I]
        self.nQuadraturePoints = self.quadraturePointArray.shape[0]
        self.vol=sum([w for  w in  self.quadrature.weights])
        vk = [[v(p) for p in self.quadrature.points] for v in monomialSpace.basis]
        self.V = Mat(monomialSpace.dim,monomialSpace.dim)
        for i,vi in enumerate(vk):
            for j,vj in enumerate(vk):
                for vik,vjk,w in zip(vi,vj,self.quadrature.weights):
                    self.V[i,j] += vjk*vik*w
        self.LUV = LinearSolvers.LU(self.V)
        #set for local (on processor) solves
        self.LUV.norm = l2Norm_local
        self.LUV.prepare()
        self.w=numpy.zeros((monomialSpace.dim,self.nQuadraturePoints),'d')
        for k,w in enumerate(self.quadrature.weights):
            vjk = numpy.zeros((monomialSpace.dim,),'d')
            wk = numpy.zeros((monomialSpace.dim,),'d')
            for j in range(monomialSpace.dim):
                vjk[j] = vk[j][k]
            self.LUV.solve(wk,b=vjk)
            for i in range(monomialSpace.dim):
                self.w[i,k] += wk[i]*w
        self.functionals = [(lambda f: sum([f(p)*wik for  wik,p in zip(self.w[i,:],self.quadrature.points)]))
                            for i in range(monomialSpace.dim)]
        self.functionalsQuadrature = [(lambda fList, i=i: sum([f*wik for wik,f in zip(self.w[i,:],fList)]))
                                      for i in range(monomialSpace.dim)]

    def quadrature2DOF_element(self,k):
        return 0
    def definedOnLocalElementBoundary(self,k,ebN_local):
        #have choice are interior nodal conditions for dg on all boundaries or none?
        return True #if matching quadrature2DOF_element behavior,


class P1BubbleInterpolationConditions(InterpolationConditions):
    """
    Interpolation conditions for space P1 enriched with bubbles
    """
    # .. math::

    # P^1(\hat{\Omega_e}) \oplus \mbox{span}{\hat{b}} b = (n_d+1)^{n_d+1} \Pi_{i=0}^{n_d} \lambda_i

    # Note :math:`b(\bar{x_{e}}) = 1`

    # Interpolation conditions are
    # .. math::

    # v(x_j), j = 0, n_d \mbox{for vertices} x_j \mbox{and} v(\bar{x}_{e}) - \frac{1}{d+1}\sum_{j=0}^{n_d}(v( x_j)) \mbox{for} b
    def __init__(self,referenceElement):
        InterpolationConditions.__init__(self,referenceElement.nNodes+1,referenceElement)
        self.quadraturePointArray = numpy.zeros((len(referenceElement.nodeList)+1,3),'d')
        #vertices
        for k,n in enumerate(referenceElement.nodeList):
            for I in range(referenceElement.dim):
                self.quadraturePointArray[k,I]=n[I]
        #bubble
        for k,n in enumerate(referenceElement.nodeList):
            for I in range(referenceElement.dim):
                self.quadraturePointArray[self.referenceElement.nNodes,I] += n[I]
        self.quadraturePointArray[self.referenceElement.nNodes,:] /= float(self.referenceElement.nNodes)

        self.nQuadraturePoints = self.quadraturePointArray.shape[0]
        #vertices
        self.functionals.append(lambda f: f(referenceElement.nodeList[0]))
        self.functionalsQuadrature.append(lambda fList: fList[0])
        if referenceElement.nNodes > 1:
            self.functionals.append(lambda f: f(referenceElement.nodeList[1]))
            self.functionalsQuadrature.append(lambda fList: fList[1])
        if referenceElement.nNodes > 2:
            self.functionals.append(lambda f: f(referenceElement.nodeList[2]))
            self.functionalsQuadrature.append(lambda fList: fList[2])
        if referenceElement.nNodes > 3:
            self.functionals.append(lambda f: f(referenceElement.nodeList[3]))
            self.functionalsQuadrature.append(lambda fList: fList[3])
        if referenceElement.nNodes > 4:
            self.functionals.append(lambda f: f(referenceElement.nodeList[4]))
            self.functionalsQuadrature.append(lambda fList: fList[4])
        if referenceElement.nNodes > 5:
            logEvent("Haven't implemented this many nodes for nodal interpolation conditions",level=1)
        #bubble
        dp1inv = 1.0/float(self.referenceElement.nNodes)
        self.functionals.append(lambda f: f(self.quadraturePointArray[-1,:]) - \
                                    dp1inv * sum([self.functionals[i](self.quadraturePointArray[-1,:]) for i in range(self.referenceElement.nNodes)]))
        self.functionalsQuadrature.append(lambda fList: fList[self.referenceElement.nNodes] - \
                                              dp1inv * sum(fList[:self.referenceElement.nNodes]))

        #mwf debug
        #import pdb
        #pdb.set_trace()
    def definedOnLocalElementBoundary(self,k,ebN_local):
        #interpolation points indexed like nodeList so nodes are not on
        #the local element across from them
        return k < self.nQuadraturePoints-1 and k != ebN_local
    def quadrature2Node_element(self,k):
        if k < self.nQuadraturePoints-1:
            return k
        return None


class P1P0BubbleInterpolationConditions(InterpolationConditions):
    from . import Quadrature
    """
    Obtains the DOF from the function values at the nodes
    """
    def __init__(self,referenceElement,monomialSpace):
        from . import LinearSolvers
        self.quadrature=self.Quadrature.SimplexGaussQuadrature(referenceElement.dim,2)
        InterpolationConditions.__init__(self,monomialSpace.dim,referenceElement)
        self.quadraturePointArray = numpy.zeros((len(self.quadrature.weights),3),'d')
        for k,p in enumerate(self.quadrature.points):
            for I in range(referenceElement.dim):
                self.quadraturePointArray[k,I]=p[I]
        self.nQuadraturePoints = self.quadraturePointArray.shape[0]
        self.vol=sum([w for  w in  self.quadrature.weights])
        vk = [[v(p) for p in self.quadrature.points] for v in monomialSpace.basis]
        self.V = Mat(monomialSpace.dim,monomialSpace.dim)
        for i,vi in enumerate(vk):
            for j,vj in enumerate(vk):
                for vik,vjk,w in zip(vi,vj,self.quadrature.weights):
                    self.V[i,j] += vjk*vik*w
        self.LUV = LinearSolvers.LU(self.V)
        #set for local (on processor) solves
        self.LUV.norm = l2Norm_local
        self.LUV.prepare()
        self.w=numpy.zeros((monomialSpace.dim,self.nQuadraturePoints),'d')
        for k,w in enumerate(self.quadrature.weights):
            vjk = numpy.zeros((monomialSpace.dim,),'d')
            wk = numpy.zeros((monomialSpace.dim,),'d')
            for j in range(monomialSpace.dim):
                vjk[j] = vk[j][k]
            self.LUV.solve(wk,b=vjk)
            for i in range(monomialSpace.dim):
                self.w[i,k] += wk[i]*w
        self.functionals = [(lambda f: sum([f(p)*wik for  wik,p in zip(self.w[i,:],self.quadrature.points)]))
                            for i in range(monomialSpace.dim)]
        self.functionalsQuadrature = [(lambda fList, i=i: sum([f*wik for wik,f in zip(self.w[i,:],fList)]))
                                      for i in range(monomialSpace.dim)]

    def quadrature2DOF_element(self,k):
        return 0
    def definedOnLocalElementBoundary(self,k,ebN_local):
        #have choice are interior nodal conditions for dg on all boundaries or none?
        return True #if matching quadrature2DOF_element behavior,


"""
Finite Elements
"""

class ReferenceFiniteElement(object):
    """
    The geometric element, local function space, and interpolation
    conditions.

    I distinguish between the "elements" (patches) of the mesh and
    the "finite elements" which also include information about the
    function space. This is sort of consistent with the mathematical
    theory where a finite element is a triple: (T,PI,SIGMA)
    """
    def __init__(self,
                 localFunctionSpace,
                 interpolationConditions):
        self.referenceElement = localFunctionSpace.referenceElement
        self.localFunctionSpace = localFunctionSpace
        self.interpolationConditions = interpolationConditions

"""
Degrees of freedom mappings
"""

class DOFMap(object):
    """
    Base class for integer mappings between local degrees of freedom
    and global degrees of freedom.
    """
    def __init__(self,nDOF=0):
        self.nDOF = nDOF
        self.range_nDOF = range(nDOF)
        self.l2g=None
        ###for parallel subdomain local dof to global dof mappings
        #array of 'offsets' where owned dof numbers start on each subdomain
        self.dof_offsets_subdomain_owned = None
        #total number of dofs in whole domain
        self.nDOF_all_processes = None
        #total number of dofs including ghosts on subdomain
        self.nDOF_subdomain = None
        #mapping of dof numbers from processor to global numbering
        self.subdomain2global = None
        #maximum number of neighbors for connectivity of dofs
        self.max_dof_neighbors = None
    def updateAfterParallelPartitioning(self,mesh):
        pass
class NodalDOFMap(DOFMap):
    """
    The mapping that associates a local degree of freedom number
    with the global node numbers of the element.
    """
    def __init__(self,mesh):
        DOFMap.__init__(self,mesh.nNodes_global)
        self.l2g=mesh.elementNodesArray
        #save for parallel now
        if mesh == mesh.subdomainMesh:
            self.updateAfterParallelPartitioning(mesh.globalMesh)
        else:
            self.dof_offsets_subdomain_owned = mesh.nodeOffsets_subdomain_owned
            self.nDOF_all_processes = mesh.nNodes_global
            self.nDOF_subdomain = mesh.nNodes_global
            self.subdomain2global = mesh.nodeNumbering_subdomain2global
            self.max_dof_neighbors = mesh.max_nNodeNeighbors_node
    def updateAfterParallelPartitioning(self,mesh):
        #array of 'offsets' where owned dof numbers start on each subdomain
        self.dof_offsets_subdomain_owned = mesh.nodeOffsets_subdomain_owned
        #total number of dofs in whole domain
        self.nDOF_all_processes = mesh.nNodes_global
        #total number of dofs including ghosts on subdomain
        self.nDOF_subdomain = mesh.subdomainMesh.nNodes_global
        #mapping of dof numbers from processor to global numbering
        self.subdomain2global = mesh.nodeNumbering_subdomain2global
        #maximum number of neighbors for connectivity of dofs
        self.max_dof_neighbors = mesh.max_nNodeNeighbors_node

class DiscontinuousGalerkinDOFMap(DOFMap):
    """
    A DOF map to use with discontinuous Galerkin spaces, which have
    unique degrees of freedom on each element.
    """
    def __init__(self,mesh,localFunctionSpace):
        DOFMap.__init__(self,mesh.nElements_global*localFunctionSpace.dim)
        self.l2g = numpy.zeros((mesh.nElements_global,
                                localFunctionSpace.dim),
                               'i')
        self.local_dim = localFunctionSpace.dim
        self.updateAfterParallelPartitioning(mesh.globalMesh)
        #for eN in range(mesh.nElements_global):
        #    for i in localFunctionSpace.range_dim:
        #        self.l2g[eN,i] = eN*localFunctionSpace.dim + i
        #    #end i
        #end eN
    #end init
    def updateAfterParallelPartitioning(self,globalMesh):
        """
        Fix nDOF_all_processes, nDOF_subdomain, and max_dof_neighbors
        """
        from . import Comm
        mpi_comm = Comm.get().comm.tompi4py()
        self.dof_offsets_subdomain_owned = numpy.zeros(globalMesh.nodeOffsets_subdomain_owned.shape,'i')
        self.nDOF_all_processes = 0; self.nDOF_subdomain = 0; self.max_dof_neighbors = 0
        self.subdomain2global = numpy.zeros((self.nDOF),'i')
        (self.nDOF_all_processes,
         self.nDOF_subdomain,
         self.max_dof_neighbors) = cpartitioning.buildDiscontinuousGalerkinLocal2GlobalMappings(mpi_comm,
                                                                                                self.local_dim,
                                                                                                globalMesh.cmesh,
                                                                                                globalMesh.subdomainMesh.cmesh,
                                                                                                globalMesh.elementOffsets_subdomain_owned,
                                                                                                globalMesh.elementNumbering_subdomain2global,
                                                                                                self.dof_offsets_subdomain_owned,
                                                                                                self.l2g,
                                                                                                self.subdomain2global)
#end disc.map
class ElementBoundaryDOFMap(DOFMap):
    """
    The mapping that associates a local degree of freedom number
    with the global edge numbers of the element.
    """
    def __init__(self,mesh):
        DOFMap.__init__(self,mesh.nElementBoundaries_global)
        self.l2g = mesh.elementBoundariesArray
        #save for parallel now
        self.updateAfterParallelPartitioning(mesh.globalMesh)
    def updateAfterParallelPartitioning(self,mesh):
        #array of 'offsets' where owned dof numbers start on each subdomain
        self.dof_offsets_subdomain_owned = mesh.elementBoundaryOffsets_subdomain_owned
        #total number of dofs in whole domain
        self.nDOF_all_processes = mesh.nElementBoundaries_global
        #total number of dofs including ghosts on subdomain
        self.nDOF_subdomain = mesh.subdomainMesh.nElementBoundaries_global
        #mapping of dof numbers from processor to global numbering
        self.subdomain2global = mesh.elementBoundaryNumbering_subdomain2global
        #maximum number of neighbors for connectivity of dofs
        self.max_dof_neighbors = 2*(mesh.nElementBoundaries_element-1)+1
    #end init
#end ElementBoundaryDOFMap
class QuadraticLagrangeCubeDOFMap(DOFMap):
    """
    DOF mapping for quadratic lagrange finite element functions on
    unit cubes

    The mapping associates local degree of freedom with global vertex
    number for iloc 0<= iloc<= space dim global edge number for
    spacedim < iloc

    total dimension is number of vertices + number of edges
    """
    # TODO fix
    # lagrangeNodesArray to hold all the nodes for parallel in 3d
    # determine if really need to call updateAfterParallelPartitioning
    # after __init__ or not
    def __init__(self,mesh,localFunctionSpace,nd):
        if nd == 1:
            print("QuadraticLagrangeCubeDOFMap not supported for nd = 1")
            #ndof += mesh.nElements_global
        elif nd == 2:
            ndof = mesh.nNodes_global
            ndof += mesh.nElementBoundaries_global
            ndof += mesh.nElements_global
        else:
            ndof = mesh.nNodes_global
            ndof += mesh.nEdges_global
            ndof += mesh.nElementBoundaries_global
            ndof += mesh.nElements_global

        DOFMap.__init__(self,ndof)
        #holds lagrange nodes for all points
        self.nd = nd
        self.lagrangeNodesArray = numpy.zeros((ndof,3),'d')
        self.l2g = numpy.zeros((mesh.nElements_global,
                                localFunctionSpace.dim),
                               'i')
        self.nd = nd
        #do simplest numbering first, which is to assign first d+1
        #unknowns the corresponding global node number.
        #
        #In 1d, extra unknown can be associated with global element number
        #In 2d, extra unknowns can be associated with element boundaries array (edges)
        #In 3d, extra unknowns have to be associated with edge
        self.updateAfterParallelPartitioning(mesh.globalMesh)

        maxSeen = max(self.l2g.flat)
        assert maxSeen < self.nDOF,('QuadDOF max(l2g)= %d ndof= %d' % (maxSeen,self.nDOF))
        #save for parallel mappings
    #end init
    def updateAfterParallelPartitioning(self,globalMesh):
        """
        Fix self.nDOF_all_processes,self.nDOF_subdomain, self.max_dof_neighbors
        """
        from . import Comm
        mpi_comm = Comm.get().comm.tompi4py()
        if self.nd==2:
            # start with the vertex DoFs
            for i,node in enumerate(globalMesh.nodeArray):
                self.lagrangeNodesArray[i] = node
            # next fill up the edge DoF
            edge_indicator = numpy.zeros(globalMesh.nElementBoundaries_global,'i')
            for i,edge_list in enumerate(globalMesh.elementBoundariesArray):
                for j,edge in enumerate(edge_list):
                    if edge_indicator[edge]==0:
                        edge_indicator[edge]==1
                        node1 = globalMesh.elementNodesArray[i][j]
                        node2 = globalMesh.elementNodesArray[i][(j+1)%4]
                        edge_coordinate = [0.,0.,0.]
                        edge_coordinate[0] = 0.5*(globalMesh.nodeArray[node1][0]+globalMesh.nodeArray[node2][0])
                        edge_coordinate[1] = 0.5*(globalMesh.nodeArray[node1][1]+globalMesh.nodeArray[node2][1])
                        self.lagrangeNodesArray[len(globalMesh.nodeArray)+edge] = edge_coordinate
                    else:
                        pass
            # fill up the DoF for the center of the elements
            for i,element in enumerate(globalMesh.elementNodesArray):
                element_coordinate = [0.,0.,0.]
                element_coordinate[0] = 0.5*(globalMesh.nodeArray[element[0]][0] + globalMesh.nodeArray[element[-1]][0])
                element_coordinate[1] = 0.5*(globalMesh.nodeArray[element[0]][1] + globalMesh.nodeArray[element[1]][1])
                self.lagrangeNodesArray[globalMesh.nNodes_global + globalMesh.nElementBoundaries_global + i] = element_coordinate
            # populate the l2g vector
            for i in range(globalMesh.nElements_global):
                # start by looping over element's vertices
                self.l2g[i][0] = globalMesh.elementNodesArray[i][0]
                self.l2g[i][1] = globalMesh.elementNodesArray[i][3]
                self.l2g[i][2] = globalMesh.elementNodesArray[i][2]
                self.l2g[i][3] = globalMesh.elementNodesArray[i][1]

                self.l2g[i][4] = globalMesh.elementBoundariesArray[i][3]+globalMesh.nNodes_global
                self.l2g[i][5] = globalMesh.elementBoundariesArray[i][2]+globalMesh.nNodes_global
                self.l2g[i][6] = globalMesh.elementBoundariesArray[i][1]+globalMesh.nNodes_global
                self.l2g[i][7] = globalMesh.elementBoundariesArray[i][0]+globalMesh.nNodes_global
                self.l2g[i][len(globalMesh.elementNodesArray[i]) + len(globalMesh.elementBoundariesArray[0]) ] = globalMesh.nNodes_global + globalMesh.nElementBoundaries_global + i
                # subdomain2global is just the identity mapping in the serial case
            self.subdomain2global = np.arange(self.nDOF,dtype='i')
            # dof_offsets_subdomain_owned
            # ARB - the next argument should use shape not len...something is being fed in wrong for 2D-Quads...Fix before
            # final merge
            self.dof_offsets_subdomain_owned = numpy.zeros(len(globalMesh.nodeOffsets_subdomain_owned),'i')
            self.dof_offsets_subdomain_owned[1] = self.nDOF
            self.nDOF_all_processes = self.nDOF
            self.nDOF_subdomain = self.nDOF
            self.max_dof_neighbors = 4
        elif self.nd==3:
            self.dof_offsets_subdomain_owned = numpy.zeros(globalMesh.nodeOffsets_subdomain_owned.shape,'i')
            self.nDOF_all_processes = 0; self.nDOF_subdomain = 0; self.max_dof_neighbors = 0
            self.subdomain2global = numpy.zeros((self.nDOF),'i')
            (self.nDOF_all_processes,
             self.nDOF_subdomain,
             self.max_dof_neighbors) = cpartitioning.buildQuadraticCubeLocal2GlobalMappings(mpi_comm,
                                                                                            self.nd,
                                                                                            globalMesh.cmesh,
                                                                                            globalMesh.subdomainMesh.cmesh,
                                                                                            globalMesh.elementOffsets_subdomain_owned,
                                                                                            globalMesh.nodeOffsets_subdomain_owned,
                                                                                            globalMesh.elementBoundaryOffsets_subdomain_owned,
                                                                                            globalMesh.edgeOffsets_subdomain_owned,
                                                                                            globalMesh.elementNumbering_subdomain2global,
                                                                                            globalMesh.nodeNumbering_subdomain2global,
                                                                                            globalMesh.elementBoundaryNumbering_subdomain2global,
                                                                                            globalMesh.edgeNumbering_subdomain2global,
                                                                                            self.dof_offsets_subdomain_owned,
                                                                                            self.l2g,
                                                                                            self.subdomain2global,
                                                                                            self.lagrangeNodesArray)
            assert self.nDOF == self.nDOF_subdomain
#QuadraticDOFMap

class QuadraticLagrangeDOFMap(DOFMap):
    """
    DOF mapping for quadratic lagrange finite element functions on
    unit simplexes

    The mapping associates local degree of freedom with global vertex
    number for iloc 0<= iloc<= space dim global edge number for
    spacedim < iloc

    total dimension is number of vertices + number of edges
    """
    #    TODO fix lagrangeNodesArray to hold all the nodes for
    #    parallel in 3d determine if really need to call
    #    updateAfterParallelPartitioning after __init__ or not
    def __init__(self,mesh,localFunctionSpace,nd):
        ndof = mesh.nNodes_global
        if nd == 1:
            ndof += mesh.nElements_global
        elif nd == 2:
            ndof += mesh.nElementBoundaries_global
        else:
            ndof += mesh.nEdges_global

        DOFMap.__init__(self,ndof)
        #holds lagrange nodes for all points
        self.lagrangeNodesArray = numpy.zeros((ndof,3),'d')
        self.l2g = numpy.zeros((mesh.nElements_global,
                                localFunctionSpace.dim),
                               'i')
        self.nd = nd
        #do simplest numbering first, which is to assign first d+1
        #unknowns the corresponding global node number.
        #
        #In 1d, extra unknown can be associated with global element number
        #In 2d, extra unknowns can be associated with element boundaries array (edges)
        #In 3d, extra unknowns have to be associated with edge
        self.updateAfterParallelPartitioning(mesh.globalMesh)

        maxSeen = max(self.l2g.flat)
        assert maxSeen < self.nDOF,('QuadDOF max(l2g)= %d ndof= %d' % (maxSeen,self.nDOF))
        #save for parallel mappings
        self.nd = nd
    #end init
    def updateAfterParallelPartitioning(self,globalMesh):
        """
        Fix self.nDOF_all_processes,self.nDOF_subdomain, self.max_dof_neighbors
        """
        from . import Comm
        mpi_comm = Comm.get().comm.tompi4py()
        self.dof_offsets_subdomain_owned = numpy.zeros(globalMesh.nodeOffsets_subdomain_owned.shape,'i')
        self.nDOF_all_processes = 0; self.nDOF_subdomain = 0; self.max_dof_neighbors = 0
        self.subdomain2global = numpy.zeros((self.nDOF),'i')
        (self.nDOF_all_processes,self.nDOF_subdomain,
         self.max_dof_neighbors) = cpartitioning.buildQuadraticLocal2GlobalMappings(mpi_comm,
                                                                                    self.nd,
                                                                                    globalMesh.cmesh,
                                                                                    globalMesh.subdomainMesh.cmesh,
                                                                                    globalMesh.elementOffsets_subdomain_owned,
                                                                                    globalMesh.nodeOffsets_subdomain_owned,
                                                                                    globalMesh.elementBoundaryOffsets_subdomain_owned,
                                                                                    globalMesh.edgeOffsets_subdomain_owned,
                                                                                    globalMesh.elementNumbering_subdomain2global,
                                                                                    globalMesh.nodeNumbering_subdomain2global,
                                                                                    globalMesh.elementBoundaryNumbering_subdomain2global,
                                                                                    globalMesh.edgeNumbering_subdomain2global,
                                                                                    self.dof_offsets_subdomain_owned,
                                                                                    self.l2g,
                                                                                    self.subdomain2global,
                                                                                    self.lagrangeNodesArray)
        assert self.nDOF == self.nDOF_subdomain
 #QuadraticDOFMap

class p0DOFMap(DOFMap):
    def __init__(self,mesh):
        DOFMap.__init__(self,mesh.nElements_global)
        self.l2g=numpy.zeros((mesh.nElements_global,1),'i')
        self.updateAfterParallelPartitioning(mesh.globalMesh)
        #for i in range(mesh.nElements_global):
        #    self.l2g[i,0]=i
    def updateAfterParallelPartitioning(self,globalMesh):
        """
        Fix self.nDOF_all_processes,self.nDOF_subdomain, self.max_dof_neighbors
        """
        from . import Comm
        mpi_comm = Comm.get().comm.tompi4py()
        local_dim = 1
        self.dof_offsets_subdomain_owned = numpy.zeros(globalMesh.nodeOffsets_subdomain_owned.shape,'i')
        self.nDOF_all_processes = 0; self.nDOF_subdomain = 0; self.max_dof_neighbors = 0
        self.subdomain2global = numpy.zeros((self.nDOF),'i')
        (self.nDOF_all_processes,self.nDOF_subdomain,
         self.max_dof_neighbors) = cpartitioning.buildDiscontinuousGalerkinLocal2GlobalMappings(mpi_comm,
                                                                                                local_dim,
                                                                                                globalMesh.cmesh,
                                                                                                globalMesh.subdomainMesh.cmesh,
                                                                                                globalMesh.elementOffsets_subdomain_owned,
                                                                                                globalMesh.elementNumbering_subdomain2global,
                                                                                                self.dof_offsets_subdomain_owned,
                                                                                                self.l2g,
                                                                                                self.subdomain2global)

class P1BubbleDOFMap(DOFMap):
    """
    DOF mapping for Lagrange P1 + bubble finite element functions on
    unit simplexes

    The mapping associates local degree of freedom with
       global vertex number for iloc 0<= iloc<= space dim
       global element number for  iloc = space dim + 1

    total dimension is number of vertices + number of elements
    TODO: implement for parallel
    """
    def __init__(self,mesh,localFunctionSpace,nd):
        ndof = mesh.nNodes_global + mesh.nElements_global
        assert localFunctionSpace.dim == nd+2, "P1 Bubble space only"
        DOFMap.__init__(self,ndof)

        #cek adding array for lagrange nodes of quadratics
        self.lagrangeNodesArray = numpy.zeros((ndof-mesh.nNodes_global,3),'d')
        self.l2g = numpy.zeros((mesh.nElements_global,
                                localFunctionSpace.dim),
                               'i')
        #do simplest numbering first, which is to assign first d+1
        #unknowns the corresponding global node number.
        #rest of unknowns associated with global element number

        for eN in range(mesh.nElements_global):
            self.l2g[eN,:-1] = mesh.elementNodesArray[eN,:]
            self.l2g[eN,-1]  = mesh.nNodes_global + eN

            self.lagrangeNodesArray[self.l2g[eN,-1]-mesh.nNodes_global,:]=mesh.elementBarycentersArray[eN]
    #end init
#end P1 bubble map

"""
Mappings.
"""

class ElementMaps(object):
    """
    Base class for a set of real number vector fields that map a reference
    element into the physical domain.
    """
    def __init__(self,mesh):
        self.mesh = mesh
        pass
    def getBasisValuesRef(self,
                          xiArray):
        """
        Evaluate the basis of the map at a set of points, xiArray,
        given on the reference element. Store in member array self.psi
        """
        pass
    def getValues(self,
                  xiArray,
                  xArray):
        """
        Evaluate the set of nElements maps at a set of points, xiArray,
        given on the reference element.
        """
        pass
    def getBasisGradientValuesRef(self,
                                  xiArray):
        """
        Evaluate the basis gradients of the map at a set of points, xiArray.
        Store in member self.grad_psi
        """
        pass
    def getJacobianValues(self,
                          xiArray,
                          jacobianArray,
                          jacobianDeterminantArray,
                          inverseJacobianArray):
        """
        Evaluate the jacobian, jacobian determinant, and inverse jacobian
        of the set of nElements maps at a set of points, xiArray.
        """
        pass
    def getInverseValues(self,
                         inverseJacobianArray,
                         xArray,
                         xiArray):
        """
        Evaluate the set of nElements inverse maps at a set of points, xArray,
        given on the physical elements. Return the value of the inverse map.
        """
        pass
    def getInverseValue(self,
                        eN,
                        x):
        """
        Evaluate the element inverse map at a point, x,
        given in physical coorindates. Return the value of the inverse map, xi, in reference coordinates.

        Note: The mapped point may not lie on the reference  element
        """
        pass
    def getBasisValuesTraceRef(self,
                               xiArray):
        """
        Evaluate the basis of the maps at a set of points, xiArray, on
        the boundaries of the reference element. Store in member
        self.psi_trace
        """
        pass
    def getValuesTrace(self,
                       xiArray,
                       xArray):
        """
        Evaluate the nElements x nElementBoundaries_element maps
        at a set of points, xiArray, on the element boundary
        reference element.
        """
        pass
    def getJacobianValuesTrace(self,
                               xiArray,
                               jacobianInverseArray,
                               metricTensorArray,
                               metricTensorDeterminantSqrtArray,
                               unitNormalArray):
        """
        Evaluate the metric tensor, square root of the determinant of
        the metric tensor, and the unit normal at the
        nElements x nElementBoundaries_element element boundaries.
        """
        pass
    def getBasisGradientValuesTraceRef(self,
                                       xiArray):
        """
        Evaluate the basis gradients of the map at a set of points,
        xiArray, on the reference element boundaries.
        """
        pass

    def getJacobianValuesTrace_movingDomain(self,
                                            xiArray,
                                            xt,
                                            jacobianInverseArray,
                                            metricTensorArray,
                                            metricTensorDeterminantSqrtArray,
                                            unitNormalArray):
        """
        Evaluate the metric tensor, square root of the determinant of
        the metric tensor, and the unit normal at the
        nElements x nElementBoundaries_element element boundaries.
        """
        pass

    def getValuesGlobalExteriorTrace(self,
                                     xiArray,
                                     xArray):
        """
        Evaluate the nExteriorElementBoundaries_global maps
        at a set of points, xiArray, on the element boundary
        reference element.
        """
        pass

    def getJacobianValuesGlobalExteriorTrace(self,
                                             xiArray,
                                             jacobianInverseArray,
                                             metricTensorArray,
                                             metricTensorDeterminantSqrtArray,
                                             unitNormalArray):
        """
        Evaluate the metric tensor, square root of the determinant of
        the metric tensor, and the unit normal at the
        nExteriorElementBoundaries_global element boundaries.
        """
        pass
    def getJacobianValuesGlobalExteriorTrace_movingDomain(self,
                                                          xiArray,
                                                          xt,
                                                          jacobianInverseArray,
                                                          metricTensorArray,
                                                          metricTensorDeterminantSqrtArray,
                                                          unitNormalArray):
        """
        Evaluate the metric tensor, square root of the determinant of
        the metric tensor, and the unit normal at the
        nExteriorElementBoundaries_global element boundaries.
        """
        pass
class ParametricMaps(ElementMaps):
    """
    A class that calculates the element maps using the
    nodes of a mesh as the degrees of freedom and a local function space
    for which the nodes are the local DOF.
    """
    def __init__(self,
                 mesh,
                 referenceElement,
                 localFunctionSpace):
        ElementMaps.__init__(self,mesh)
        self.mesh = mesh
        self.referenceElement = referenceElement
        self.localFunctionSpace = localFunctionSpace
        self.meshDOFMap=NodalDOFMap(mesh)
        self.useC=True
    def getBasisValuesIP(self, interpolationPoints):
        n_xi = interpolationPoints.shape[0]
        range_n_xi = list(range(n_xi))
        self.psi_ip = numpy.zeros((n_xi,
                                self.localFunctionSpace.dim),
                               'd')
        for k in range_n_xi:
            for j in self.localFunctionSpace.range_dim:
                self.psi_ip[k,j] = self.localFunctionSpace.basis[j](interpolationPoints[k])
        return self.psi_ip
    def getBasisValuesRef(self,xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.psi = numpy.zeros((n_xi,
                                self.localFunctionSpace.dim),
                               'd')
        for k in range_n_xi:
            for j in self.localFunctionSpace.range_dim:
                self.psi[k,j] = self.localFunctionSpace.basis[j](xiArray[k])
        return self.psi
    def getValues(self,xiArray,
                  xArray):
        xArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        psi = numpy.zeros((n_xi,
                             self.localFunctionSpace.dim),
                            'd')
        for k in range_n_xi:
            for j in self.localFunctionSpace.range_dim:
                psi[k,j] = self.localFunctionSpace.basis[j](xiArray[k])
        if self.useC==True:
            cfemIntegrals.parametricMaps_getValues(psi,
                                                   self.meshDOFMap.l2g,
                                                   self.mesh.nodeArray,
                                                   xArray)
        else:
            for eN in range(self.mesh.nElements_global):
                for k in range_n_xi:
                    for j in self.localFunctionSpace.range_dim:
                        J = self.meshDOFMap.l2g[eN,j]
                        for m in self.referenceElement.range_dim:
                            xArray[eN,k,m] += self.mesh.nodeArray[J,m]*psi[k,j]
    def getBasisGradientValuesRef(self,xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.grad_psi = numpy.zeros((n_xi,
                                     self.localFunctionSpace.dim,
                                     self.referenceElement.dim),
                                    'd')
        for k in range_n_xi:
            for j in self.localFunctionSpace.range_dim:
                self.grad_psi[k,j,:] = self.localFunctionSpace.basisGradients[j](xiArray[k])
        return self.grad_psi
    def getJacobianValues(self,xiArray,
                          jacobianArray,
                          jacobianInverseArray,
                          jacobianDeterminantArray):
        jacobianArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        grad_psi = numpy.zeros((n_xi,
                                  self.localFunctionSpace.dim,
                                  self.referenceElement.dim),
                                 'd')
        for k in range_n_xi:
            for j in self.localFunctionSpace.range_dim:
                grad_psi[k,j,:] = self.localFunctionSpace.basisGradients[j](xiArray[k])
        if self.useC==True:
            cfemIntegrals.parametricMaps_getJacobianValues(grad_psi,
                                                           self.meshDOFMap.l2g,
                                                           self.mesh.nodeArray,
                                                           jacobianArray,
                                                           jacobianDeterminantArray,
                                                           jacobianInverseArray)
        else:
            for eN in range(self.mesh.nElements_global):
                for k in range_n_xi:
                    for j in self.localFunctionSpace.range_dim:
                        J = self.meshDOFMap.l2g[eN,j]
                        for m in self.referenceElement.range_dim:
                            for n in self.referenceElement.range_dim:
                                jacobianArray[eN,k,m,n] += self.mesh.nodeArray[J,m]*grad_psi[k,j,n]
                    jacobianDeterminantArray[eN,k] = det(jacobianArray[eN,k])
                    jacobianInverseArray[eN,k,:,:] = adj(jacobianArray[eN,k])/jacobianDeterminantArray[eN,k]
    def getBasisValuesTraceRef(self,
                               xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.psi_trace = numpy.zeros((self.referenceElement.nElementBoundaries,
                                      n_xi,
                                      self.localFunctionSpace.dim),
                                     'd')
        for ebN in self.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.localFunctionSpace.range_dim:
                    #mwf now manually map from \bar{x} (reference element boundary quadrature point to reference element space
                    #and then evaluate using basis, since basisTrace will be deprecated
                    #psi[ebN,k,j]        = self.localFunctionSpace.basisTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceElement.boundaryMapList[ebN](xiArray[k])
                    self.psi_trace[ebN,k,j] = self.localFunctionSpace.basis[j](xiHat_k)
        return self.psi_trace
    def getValuesTrace(self,
                       xiArray,
                       xArray):
        xArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        psi = numpy.zeros((self.referenceElement.nElementBoundaries,
                             n_xi,
                             self.localFunctionSpace.dim),
                            'd')
        for ebN in self.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.localFunctionSpace.range_dim:
                    #mwf now manually map from \bar{x} (reference element boundary quadrature point to reference element space
                    #and then evaluate using basis, since basisTrace will be deprecated
                    #psi[ebN,k,j]        = self.localFunctionSpace.basisTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceElement.boundaryMapList[ebN](xiArray[k])
                    psi[ebN,k,j] = self.localFunctionSpace.basis[j](xiHat_k)
        if self.useC==True:
            cfemIntegrals.parametricMaps_getValuesTrace(psi,self.meshDOFMap.l2g,self.mesh.nodeArray,xArray)
        else:
            for eN in range(self.mesh.nElements_global):
                for ebN in self.referenceElement.range_nElementBoundaries:
                    for k in range_n_xi:
                        for j in self.localFunctionSpace.range_dim:
                            J = self.meshDOFMap.l2g[eN,j]
                            for m in self.referenceElement.range_dim:
                                xArray[eN,ebN,k,m] += self.mesh.nodeArray[J,m]*psi[ebN,k,j]
    def getBasisGradientValuesTraceRef(self,
                                       xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.grad_psi_trace = numpy.zeros((self.referenceElement.nElementBoundaries,
                                           n_xi,
                                           self.localFunctionSpace.dim,
                                           self.referenceElement.dim),
                                          'd')
        self.boundaryNormals = numpy.zeros((self.referenceElement.nElementBoundaries,
                                            n_xi,
                                            self.referenceElement.dim),
                                           'd')
        self.boundaryJacobians = numpy.zeros((self.referenceElement.nElementBoundaries,
                                              n_xi,
                                              self.referenceElement.dim,
                                              self.referenceElement.dim-1),
                                             'd')
        for ebN in self.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.localFunctionSpace.range_dim:
                    #mwf move away from using basisGradientsTrace directly since will be deprecated
                    #switch to using boundaryMapList directly
                    #grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradientsTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceElement.boundaryMapList[ebN](xiArray[k])
                    self.grad_psi_trace[ebN,k,j,:] = self.localFunctionSpace.basisGradients[j](xiHat_k)
                self.boundaryNormals[ebN,k,:] = self.referenceElement.boundaryUnitNormalList[ebN]#planar faces in physical space
                self.boundaryJacobians[ebN,k,:,:] = self.referenceElement.boundaryJacobianList[ebN]
        return self.grad_psi_trace
    def getJacobianValuesTrace(self,
                               xiArray,
                               jacobianInverseArray,
                               metricTensorArray,
                               metricTensorDeterminantSqrtArray,
                               unitNormalArray):
        jacobianInverseArray.flat[:]=0.0
        unitNormalArray.flat[:]=0.0
        metricTensorArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        grad_psi = numpy.zeros((self.referenceElement.nElementBoundaries,
                                  n_xi,
                                  self.localFunctionSpace.dim,
                                  self.referenceElement.dim),
                                 'd')
        elementMappingJacobian = numpy.zeros((self.referenceElement.dim,
                                                self.referenceElement.dim),
                                               'd')
        elementBoundaryMappingJacobian = numpy.zeros((self.referenceElement.dim,
                                                        self.referenceElement.dim-1),
                                                       'd')
        boundaryNormals = numpy.array(self.referenceElement.boundaryUnitNormalList)
        boundaryJacobians = numpy.array(self.referenceElement.boundaryJacobianList)
        for ebN in self.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.localFunctionSpace.range_dim:
                    #mwf move away from using basisGradientsTrace directly since will be deprecated
                    #switch to using boundaryMapList directly
                    #grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradientsTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceElement.boundaryMapList[ebN](xiArray[k])
                    grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradients[j](xiHat_k)
        if self.useC == True:
            cfemIntegrals.parametricMaps_getJacobianValuesTrace(grad_psi,
                                                                boundaryNormals,
                                                                boundaryJacobians,
                                                                self.meshDOFMap.l2g,
                                                                self.mesh.nodeArray,
                                                                jacobianInverseArray,
                                                                metricTensorArray,
                                                                metricTensorDeterminantSqrtArray,
                                                                unitNormalArray)
        else:
            if self.referenceElement.dim == 1:
                for eN in range(self.mesh.nElements_global):
                    for ebN in self.referenceElement.range_nElementBoundaries:
                        for k in range_n_xi:
                            elementMappingJacobian.flat[:]=0.0
                            elementBoundaryMappingJacobian[:]=0.0
                            for j in self.localFunctionSpace.range_dim:
                                J = self.meshDOFMap.l2g[eN,j]
                                for m in self.referenceElement.range_dim:
                                    for n in self.referenceElement.range_dim:
                                        elementMappingJacobian[m,n] += self.mesh.nodeArray[J,m]*grad_psi[ebN,k,j,n]
                            jacobianInverseArray[eN,ebN,k,:,:] = inv(elementMappingJacobian)
                            unitNormalArray[eN,ebN,k,:] = self.referenceElement.boundaryUnitNormalList[ebN]
                            metricTensorArray[eN,ebN,k,0,0] = 1.0
                            metricTensorDeterminantSqrtArray[eN,ebN,k] = 1.0
            else:
                for eN in range(self.mesh.nElements_global):
                    for ebN in self.referenceElement.range_nElementBoundaries:
                        for k in range_n_xi:
                            elementMappingJacobian.flat[:]=0.0
                            elementBoundaryMappingJacobian[:]=0.0
                            for j in self.localFunctionSpace.range_dim:
                                J = self.meshDOFMap.l2g[eN,j]
                                for m in self.referenceElement.range_dim:
                                    for n in self.referenceElement.range_dim:
                                        elementMappingJacobian[m,n] += self.mesh.nodeArray[J,m]*grad_psi[ebN,k,j,n]
                            jacobianInverseArray[eN,ebN,k,:,:] = inv(elementMappingJacobian)
                            #unit normal = J^{-t} \hat{n}
                            for m in self.referenceElement.range_dim:
                                for n in self.referenceElement.range_dim:
                                    unitNormalArray[eN,ebN,k,m] += jacobianInverseArray[eN,ebN,k,n,m]*self.referenceElement.boundaryUnitNormalList[ebN][n]
                            unitNormalArray[eN,ebN,k]/=norm(unitNormalArray[eN,ebN,k])
                            #metric tensor and determinant
                            for m in self.referenceElement.range_dim:
                                for p in range(self.referenceElement.dim-1):
                                    for n in self.referenceElement.range_dim:
                                        elementBoundaryMappingJacobian[m,p] += elementMappingJacobian[m,n]*self.referenceElement.boundaryJacobianList[ebN][n,p]
                            for p in range(self.referenceElement.dim-1):
                                for q in range(self.referenceElement.dim-1):
                                    for m in self.referenceElement.range_dim:
                                        metricTensorArray[eN,ebN,k,p,q] += elementBoundaryMappingJacobian[m,p]*elementBoundaryMappingJacobian[m,q]
                            metricTensorDeterminantSqrtArray[eN,ebN,k] = sqrt(det(metricTensorArray[eN,ebN,k]))
    def getJacobianValuesTrace_movingDomain(self,
                                            xiArray,
                                            xt,
                                            jacobianInverseArray,
                                            metricTensorArray,
                                            metricTensorDeterminantSqrtArray,
                                            unitNormalArray):
        jacobianInverseArray.flat[:]=0.0
        unitNormalArray.flat[:]=0.0
        metricTensorArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        grad_psi = numpy.zeros((self.referenceElement.nElementBoundaries,
                                  n_xi,
                                  self.localFunctionSpace.dim,
                                  self.referenceElement.dim),
                                 'd')
        elementMappingJacobian = numpy.zeros((self.referenceElement.dim,
                                                self.referenceElement.dim),
                                               'd')
        elementBoundaryMappingJacobian = numpy.zeros((self.referenceElement.dim,
                                                        self.referenceElement.dim-1),
                                                       'd')
        boundaryNormals = numpy.array(self.referenceElement.boundaryUnitNormalList)
        boundaryJacobians = numpy.array(self.referenceElement.boundaryJacobianList)
        for ebN in self.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.localFunctionSpace.range_dim:
                    #mwf move away from using basisGradientsTrace directly since will be deprecated
                    #switch to using boundaryMapList directly
                    #grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradientsTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceElement.boundaryMapList[ebN](xiArray[k])
                    grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradients[j](xiHat_k)
        cfemIntegrals.parametricMaps_getJacobianValuesTrace_movingDomain(xt,
                                                                         grad_psi,
                                                                         boundaryNormals,
                                                                         boundaryJacobians,
                                                                         self.meshDOFMap.l2g,
                                                                         self.mesh.nodeArray,
                                                                         jacobianInverseArray,
                                                                         metricTensorArray,
                                                                         metricTensorDeterminantSqrtArray,
                                                                         unitNormalArray)
    def getPermutations(self,xiArray):
        #cek todo, figure out how to get the permutations without staring any points
        self.permutations = numpy.zeros(xiArray.shape[:-1],'i')
        cfemIntegrals.parametricMaps_getPermutations(xiArray,self.permutations)
    def getValuesGlobalExteriorTrace(self,
                                     xiArray,
                                     xArray):
        """
        treat this like element boundary version but only store values on exterior elements
        """
        xArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        psi = numpy.zeros((self.referenceElement.nElementBoundaries,
                           n_xi,
                           self.localFunctionSpace.dim),
                          'd')
        for ebN in self.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.localFunctionSpace.range_dim:
                    #mwf now manually map from \bar{x} (reference element boundary quadrature point to reference element space
                    #and then evaluate using basis, since basisTrace will be deprecated
                    #psi[ebN,k,j]        = self.localFunctionSpace.basisTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceElement.boundaryMapList[ebN](xiArray[k])
                    psi[ebN,k,j]        = self.localFunctionSpace.basis[j](xiHat_k)
        if self.useC==True:
            cfemIntegrals.parametricMaps_getValuesGlobalExteriorTrace(self.mesh.exteriorElementBoundariesArray,
                                                                      self.mesh.elementBoundaryElementsArray,
                                                                      self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                      psi,
                                                                      self.meshDOFMap.l2g,self.mesh.nodeArray,
                                                                      xArray)
        else:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                ebN_local = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                for k in range_n_xi:
                    for j in self.localFunctionSpace.range_dim:
                        J = self.meshDOFMap.l2g[eN,j]
                        for m in self.referenceElement.range_dim:
                            xArray[ebNE,k,m] += self.mesh.nodeArray[J,m]*psi[ebN_local,k,j]
    def getJacobianValuesGlobalExteriorTrace(self,
                                             xiArray,
                                             jacobianInverseArray,
                                             metricTensorArray,
                                             metricTensorDeterminantSqrtArray,
                                             unitNormalArray):
        """
        treat this like element boundary version but only store values on exterior elements
        """

        jacobianInverseArray.flat[:]=0.0
        unitNormalArray.flat[:]=0.0
        metricTensorArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        grad_psi = numpy.zeros((self.referenceElement.nElementBoundaries,
                                  n_xi,
                                  self.localFunctionSpace.dim,
                                  self.referenceElement.dim),
                                 'd')
        elementMappingJacobian = numpy.zeros((self.referenceElement.dim,
                                                self.referenceElement.dim),
                                               'd')
        elementBoundaryMappingJacobian = numpy.zeros((self.referenceElement.dim,
                                                        self.referenceElement.dim-1),
                                                       'd')
        boundaryNormals = numpy.array(self.referenceElement.boundaryUnitNormalList)
        boundaryJacobians = numpy.array(self.referenceElement.boundaryJacobianList)
        for ebN in self.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.localFunctionSpace.range_dim:
                    #mwf move away from using basisGradientsTrace directly since will be deprecated
                    #switch to using boundaryMapList directly
                    #grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradientsTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceElement.boundaryMapList[ebN](xiArray[k])
                    grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradients[j](xiHat_k)
        if self.useC == True:
            cfemIntegrals.parametricMaps_getJacobianValuesGlobalExteriorTrace(self.mesh.exteriorElementBoundariesArray,
                                                                              self.mesh.elementBoundaryElementsArray,
                                                                              self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                              grad_psi,
                                                                              boundaryNormals,
                                                                              boundaryJacobians,
                                                                              self.meshDOFMap.l2g,
                                                                              self.mesh.nodeArray,
                                                                              jacobianInverseArray,
                                                                              metricTensorArray,
                                                                              metricTensorDeterminantSqrtArray,
                                                                              unitNormalArray)
        else:
            if self.referenceElement.dim == 1:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    ebN_local = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                    for k in range_n_xi:
                        elementMappingJacobian.flat[:]=0.0
                        elementBoundaryMappingJacobian[:]=0.0
                        for j in self.localFunctionSpace.range_dim:
                            J = self.meshDOFMap.l2g[eN,j]
                            for m in self.referenceElement.range_dim:
                                for n in self.referenceElement.range_dim:
                                    elementMappingJacobian[m,n] += self.mesh.nodeArray[J,m]*grad_psi[ebN_local,k,j,n]
                        jacobianInverseArray[ebNE,k,:,:] = inv(elementMappingJacobian)
                        unitNormalArray[ebNE,k,:] = self.referenceElement.boundaryUnitNormalList[ebN_local]
                        metricTensorArray[ebNE,k,0,0] = 1.0
                        metricTensorDeterminantSqrtArray[ebN,k] = 1.0
            else:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                    ebN_local = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                    for k in range_n_xi:
                        elementMappingJacobian.flat[:]=0.0
                        elementBoundaryMappingJacobian[:]=0.0
                        for j in self.localFunctionSpace.range_dim:
                            J = self.meshDOFMap.l2g[eN,j]
                            for m in self.referenceElement.range_dim:
                                for n in self.referenceElement.range_dim:
                                    elementMappingJacobian[m,n] += self.mesh.nodeArray[J,m]*grad_psi[ebN_local,k,j,n]
                        jacobianInverseArray[ebNE,k,:,:] = inv(elementMappingJacobian)
                        #unit normal = J^{-t} \hat{n}
                        for m in self.referenceElement.range_dim:
                            for n in self.referenceElement.range_dim:
                                unitNormalArray[ebNE,k,m] += jacobianInverseArray[ebN,k,n,m]*self.referenceElement.boundaryUnitNormalList[ebN_local][n]
                        unitNormalArray[ebNE,k]/=norm(unitNormalArray[ebN,k])
                        #metric tensor and determinant
                        for m in self.referenceElement.range_dim:
                            for p in range(self.referenceElement.dim-1):
                                for n in self.referenceElement.range_dim:
                                    elementBoundaryMappingJacobian[m,p] += elementMappingJacobian[m,n]*self.referenceElement.boundaryJacobianList[ebN_local][n,p]
                        for p in range(self.referenceElement.dim-1):
                            for q in range(self.referenceElement.dim-1):
                                for m in self.referenceElement.range_dim:
                                    metricTensorArray[ebNE,k,p,q] += elementBoundaryMappingJacobian[m,p]*elementBoundaryMappingJacobian[m,q]
                        metricTensorDeterminantSqrtArray[ebNE,k] = sqrt(det(metricTensorArray[ebNE,k]))
                    #k
                #ebNE
            #else dim
        #else C
    #def
    def getJacobianValuesGlobalExteriorTrace_movingDomain(self,
                                                          xiArray,
                                                          xt,
                                                          jacobianInverseArray,
                                                          metricTensorArray,
                                                          metricTensorDeterminantSqrtArray,
                                                          unitNormalArray):
        """
        treat this like element boundary version but only store values on exterior elements
        """

        jacobianInverseArray.flat[:]=0.0
        unitNormalArray.flat[:]=0.0
        metricTensorArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        grad_psi = numpy.zeros((self.referenceElement.nElementBoundaries,
                                  n_xi,
                                  self.localFunctionSpace.dim,
                                  self.referenceElement.dim),
                                 'd')
        elementMappingJacobian = numpy.zeros((self.referenceElement.dim,
                                                self.referenceElement.dim),
                                               'd')
        elementBoundaryMappingJacobian = numpy.zeros((self.referenceElement.dim,
                                                        self.referenceElement.dim-1),
                                                       'd')
        boundaryNormals = numpy.array(self.referenceElement.boundaryUnitNormalList)
        boundaryJacobians = numpy.array(self.referenceElement.boundaryJacobianList)
        for ebN in self.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.localFunctionSpace.range_dim:
                    #mwf move away from using basisGradientsTrace directly since will be deprecated
                    #switch to using boundaryMapList directly
                    #grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradientsTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceElement.boundaryMapList[ebN](xiArray[k])
                    grad_psi[ebN,k,j,:] = self.localFunctionSpace.basisGradients[j](xiHat_k)
        cfemIntegrals.parametricMaps_getJacobianValuesGlobalExteriorTrace_movingDomain(self.mesh.exteriorElementBoundariesArray,
                                                                                       self.mesh.elementBoundaryElementsArray,
                                                                                       self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                       xt,
                                                                                       grad_psi,
                                                                                       boundaryNormals,
                                                                                       boundaryJacobians,
                                                                                       self.meshDOFMap.l2g,
                                                                                       self.mesh.nodeArray,
                                                                                       jacobianInverseArray,
                                                                                       metricTensorArray,
                                                                                       metricTensorDeterminantSqrtArray,
                                                                                       unitNormalArray)
class AffineMaps(ParametricMaps):
    def __init__(self,mesh,referenceElement,localFunctionSpace):
        ParametricMaps.__init__(self,mesh,referenceElement,localFunctionSpace)
        self.useC=True
    def getInverseValues(self,
                         inverseJacobian,
                         xArray,
                         xiArray):
        if self.useC == True:
            cfemIntegrals.parametricMaps_getInverseValues(inverseJacobian,self.meshDOFMap.l2g,self.mesh.nodeArray,xArray,xiArray)
        else:
            # I don't think this code works correctly when self.useC == False...
            xiArray.flat[:]=0.0
            n_x = xArray.shape[1]
            range_nx = list(range(n_x))
            grad_psi = numpy.zeros((self.localFunctionSpace.dim,
                                    self.referenceElement.dim),
                                   'd')
            dx = numpy.zeros((self.referenceElement.dim),
                             'd')
            jacobian = numpy.zeros((self.referenceElement.dim,
                                    self.referenceElement.dim),
                                   'd')
            inverseJacobian = numpy.zeros((self.referenceElement.dim,
                                           self.referenceElement.dim),
                                          'd')
            for j in self.localFunctionSpace.range_dim:
                grad_psi[j,:] = self.localFunctionSpace.basisGradients[j](xiArray[0])
            for eN in range(self.mesh.nElements_global):
                jacobian.flat[:]=0.0
                inverseJacobian.flat[:]=0.0
                for j in self.localFunctionSpace.range_dim:
                    J = self.meshDOFMap.l2g[eN,j]
                    for m in self.referenceElement.range_dim:
                        for n in self.referenceElement.range_dim:
                            jacobian[m,n] += self.mesh.nodeArray[J,m]*grad_psi[j,n]
                J = self.meshDOFMap.l2g[eN,0]
                inverseJacobian = inv(jacobian)
                for k in range_nx:
                    dx[:]=xArray[eN,k]
                    for m in self.referenceElement.range_dim:
                        dx[m]-=self.mesh.nodeArray[J,m]
                    for m in self.referenceElement.range_dim:
                        for n in self.referenceElement.range_dim:
                            xiArray[eN,k,m] += inverseJacobian[m,n]*dx[n]
    def getInverseValue(self,
                        eN,
                        x):
        xi=numpy.zeros((self.referenceElement.dim,),'d')
        grad_psi = numpy.zeros((self.localFunctionSpace.dim,
                                self.referenceElement.dim),
                               'd')
        dx = numpy.zeros((self.referenceElement.dim),
                         'd')
        jacobian = numpy.zeros((self.referenceElement.dim,
                                self.referenceElement.dim),
                               'd')
        inverseJacobian = numpy.zeros((self.referenceElement.dim,
                                       self.referenceElement.dim),
                                      'd')
        for j in self.localFunctionSpace.range_dim:
            grad_psi[j,:] = self.localFunctionSpace.basisGradients[j](xi)
        jacobian.flat[:]=0.0
        inverseJacobian.flat[:]=0.0
        for j in self.localFunctionSpace.range_dim:
            J = self.meshDOFMap.l2g[eN,j]
            for m in self.referenceElement.range_dim:
                for n in self.referenceElement.range_dim:
                    jacobian[m,n] += self.mesh.nodeArray[J,m]*grad_psi[j,n]
        J = self.meshDOFMap.l2g[eN,0]
        inverseJacobian = inv(jacobian)
        for m in self.referenceElement.range_dim:
            dx[m]=x[m]
        for m in self.referenceElement.range_dim:
            dx[m]-=self.mesh.nodeArray[J,m]
        for m in self.referenceElement.range_dim:
            for n in self.referenceElement.range_dim:
                xi[m] += inverseJacobian[m,n]*dx[n]
        return xi
    def getInverseValuesTrace(self,
                              inverseJacobian,
                              xArray,
                              xiArray):
        if self.useC:
            cfemIntegrals.parametricMaps_getInverseValuesTrace(inverseJacobian,self.meshDOFMap.l2g,self.mesh.nodeArray,xArray,xiArray)
        else:
            xiArray.flat[:]=0.0
            n_x = xArray.shape[2]
            range_nx = list(range(n_x))
            grad_psi = numpy.zeros((self.localFunctionSpace.dim,
                                      self.referenceElement.dim),
                                     'd')
            dx = numpy.zeros((self.referenceElement.dim),
                               'd')
            jacobian = numpy.zeros((self.referenceElement.dim,
                                      self.referenceElement.dim),
                                     'd')
            inverseJacobian = numpy.zeros((self.referenceElement.dim,
                                             self.referenceElement.dim),
                                            'd')
            for j in self.localFunctionSpace.range_dim:
                grad_psi[j,:] = self.localFunctionSpace.basisGradients[j](xiArray[0])
            for eN in range(self.mesh.nElements_global):
                jacobian.flat[:]=0.0
                inverseJacobian.flat[:]=0.0
                for j in self.localFunctionSpace.range_dim:
                    J = self.meshDOFMap.l2g[eN,j]
                    for m in self.referenceElement.range_dim:
                        for n in self.referenceElement.range_dim:
                            jacobian[m,n] += self.mesh.nodeArray[J,m]*grad_psi[j,n]
                J = self.meshDOFMap.l2g[eN,0]
                inverseJacobian = inv(jacobian)
                for ebN in range(self.mesh.nElementBoundaries_element):
                    for k in range_nx:
                        dx[:]=xArray[eN,ebN,k,:self.referenceElement.dim]
                        for m in self.referenceElement.range_dim:
                            dx[m]-=self.mesh.nodeArray[J,m]
                        for m in self.referenceElement.range_dim:
                            for n in self.referenceElement.range_dim:
                                xiArray[eN,ebN,k,m] += inverseJacobian[m,n]*dx[n]
    def getInverseValuesGlobalExteriorTrace(self,
                                            inverseJacobian,
                                            xArray,
                                            xiArray):
        if self.useC:
            cfemIntegrals.parametricMaps_getInverseValuesGlobalExteriorTrace(self.mesh.exteriorElementBoundariesArray,
                                                                             self.mesh.elementBoundaryElementsArray,
                                                                             self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                             inverseJacobian,self.meshDOFMap.l2g,
                                                                             self.mesh.nodeArray,xArray,xiArray)
        else:
            xiArray.flat[:]=0.0
            n_x = xArray.shape[1]
            range_nx = list(range(n_x))
            grad_psi = numpy.zeros((self.localFunctionSpace.dim,
                                    self.referenceElement.dim),
                                   'd')
            dx = numpy.zeros((self.referenceElement.dim),
                             'd')
            jacobian = numpy.zeros((self.referenceElement.dim,
                                    self.referenceElement.dim),
                                   'd')
            inverseJacobian = numpy.zeros((self.referenceElement.dim,
                                           self.referenceElement.dim),
                                          'd')
            for j in self.localFunctionSpace.range_dim:
                grad_psi[j,:] = self.localFunctionSpace.basisGradients[j](xiArray[0])
            for ebNE in range(self.mesh.nExteriorElements_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN  = self.mesh.elementBoundaryElementsArray[ebN,0]
                ebN_local = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                jacobian.flat[:]=0.0
                inverseJacobian.flat[:]=0.0
                for j in self.localFunctionSpace.range_dim:
                    J = self.meshDOFMap.l2g[eN,j]
                    for m in self.referenceElement.range_dim:
                        for n in self.referenceElement.range_dim:
                            jacobian[m,n] += self.mesh.nodeArray[J,m]*grad_psi[j,n]
                J = self.meshDOFMap.l2g[eN,0]
                inverseJacobian = inv(jacobian)
                for k in range_nx:
                    dx[:]=xArray[ebNE,k,:self.referenceElement.dim]
                    for m in self.referenceElement.range_dim:
                        dx[m]-=self.mesh.nodeArray[J,m]
                    for m in self.referenceElement.range_dim:
                        for n in self.referenceElement.range_dim:
                            xiArray[ebNE,k,m] += inverseJacobian[m,n]*dx[n]
#  """
#  Function Spaces iso-(diffeo-)morphic to reference spaces

#  Arguments will still be in reference space (xi), but
#  gradients will be in physical space (grad = grad_x).
#  """

#  class MappedSpace(LocalFunctionSpace):
#      """
#      psi(x) = psi_r ( xi(x)) = psi_r(xi)
#      grad_x psi(xi) = J^{-t}(xi) grad_{xi} psi(xi)
#      """
#      def __init__(self,referenceSpace,map,nd=3):
#          LocalFunctionSpace.__init__(self)
#          self.referenceSpace = referenceSpace
#          self.map = map
#          self.dim=nd+1
#          self.basis = self.referenceSpace.basis
#          self.basisGradients.append(
#              lambda xi: etenvec(self.map.getInverseJacobianTranspose(xi),
#                                 self.referenceSpace.basisGradients[0](xi)))
#          self.basisGradients.append(
#              lambda xi: etenvec(self.map.getInverseJacobianTranspose(xi),
#                                 self.referenceSpace.basisGradients[1](xi)))
#          if nd > 1:
#              self.basisGradients.append(
#                  lambda xi: etenvec(self.map.getInverseJacobianTranspose(xi),
#                                     self.referenceSpace.basisGradients[2](xi)))
#          if nd > 2:
#              self.basisGradients.append(
#                  lambda xi: etenvec(self.map.getInverseJacobianTranpose(xi),
#                                     self.referenceSpace.basisGradients[3](xi)))

#  class AffineMappedSpace(LocalFunctionSpace):
#      """
#      x = M xi + b
#      psi(x) = psi_r(xi)
#      grad_x psi = M^{-t} grad_{xi} (constant w.r.t. xi or x
#      so we provide updateGradients to do this whenever map changes)
#      """
#      def __init__(self,referenceSpace,affineMap,nd=3):
#          LocalFunctionSpace.__init__(self)
#          self.referenceSpace = referenceSpace
#          self.map = affineMap
#          self.dim=nd+1
#          self.basis = self.referenceSpace.basis
#          self.basisOnMappedDomain=[]
#          self.basisOnMappedDomain.append(
#              lambda x: self.basis[0](self.map.getInverseValue(x)))
#          self.gradientList=[]
#          self.gradientList.append(
#              etenvec(self.map.getInverseJacobianTranspose(),
#                      self.referenceSpace.gradientList[0]))
#          self.basisGradients.append(lambda xi: self.gradientList[0])
#          self.basisOnMappedDomain.append(
#              lambda x: self.basis[1](self.map.getInverseValue(x)))
#          self.gradientList.append(
#              etenvec(self.map.getInverseJacobianTranspose(),
#                      self.referenceSpace.gradientList[1]))
#          self.basisGradients.append(lambda xi: self.gradientList[1])
#          if nd > 1:
#              self.basisOnMappedDomain.append(
#                  lambda x: self.basis[2](self.map.getInverseValue(x)))
#              self.gradientList.append(
#                  etenvec(self.map.getInverseJacobianTranspose(),
#                          self.referenceSpace.gradientList[2]))
#              self.basisGradients.append(lambda xi: self.gradientList[2])
#          if nd > 2:
#              self.basisOnMappedDomain.append(
#                  lambda x: self.basis[3](self.map.getInverseValue(x)))
#              self.gradientList.append(
#                  etenvec(self.map.getInverseJacobianTranspose(),
#                          self.referenceSpace.gradientList[3]))
#              self.basisGradients.append(lambda xi: self.gradientList[3])
#      def updateGradients():
#          for i in range(self.dim):
#              self.gradientList[i]=etenvec(
#                  self.map.getInverseJacobianTranspose(),
#                  self.referenceSpace.gradientList[i])

"""
Finite Element Spaces
"""

class ParametricFiniteElementSpace(object):
    """
    Base class for spaces of functions defined by a set of finite
    elements that are each related to the same reference finite element.

    dim -- global dimension of the space
    mesh -- a Mesh object, which is a partition of the domain
    finiteElements -- a dictionary of the FiniteElement objects
    indexed by element number
    dofMap -- a DOF
    """
    def __init__(self, referenceFiniteElement, elementMaps, dofMap):
        self.strongDirichletConditions = True
        self.dim=dofMap.nDOF
        self.range_dim = list(range(dofMap.nDOF))
        self.referenceFiniteElement=referenceFiniteElement
        self.elementMaps = elementMaps
        self.mesh = elementMaps.mesh
        self.nSpace_global = referenceFiniteElement.referenceElement.dim
        self.dofMap=dofMap
        self.max_nDOF_element = referenceFiniteElement.localFunctionSpace.dim
        self.interpolationPoints = numpy.zeros((self.elementMaps.mesh.nElements_global,
                                                  self.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                                                  3),
                                                 'd')
        self.updateInterpolationPoints()
        self.nOutput=0
        self.viewer=None
        self.useC=True
    def getBasisValuesRef(self,xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.psi = numpy.zeros((n_xi,
                                self.referenceFiniteElement.localFunctionSpace.dim),
                               'd')
        for k in range_n_xi:
            for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                self.psi[k,j] = self.referenceFiniteElement.localFunctionSpace.basis[j](xiArray[k])
        return self.psi
    def getBasisValues(self,
                       xiArray,
                       vArray):
        #\todo it isn't really necessarry to load values into array for uniform quadrature and basis functions
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        psi = numpy.zeros((n_xi,
                             self.referenceFiniteElement.localFunctionSpace.dim),
                            'd')
        for k in range_n_xi:
            for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                psi[k,j] = self.referenceFiniteElement.localFunctionSpace.basis[j](xiArray[k])
        if self.useC == True:
            cfemIntegrals.parametricFiniteElementSpace_getValues(psi,vArray)
        else:
            for eN in range(self.elementMaps.mesh.nElements_global):
                for k in range_n_xi:
                    for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                        vArray[eN,k,j] = psi[k,j]
    def getBasisValuesAtArray(self,
                              xiArrayArray,
                              vArray):
        n_xi = xiArrayArray.shape[1]
        range_n_xi = list(range(n_xi))
        for eN in range(self.elementMaps.mesh.nElements_global):
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    vArray[eN,k,j] = self.referenceFiniteElement.localFunctionSpace.basis[j](xiArrayArray[eN,k,:self.referenceFiniteElement.referenceElement.dim])
    def getBasisGradientValuesRef(self,
                                  xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.grad_psi = numpy.zeros((n_xi,
                                     self.referenceFiniteElement.localFunctionSpace.dim,
                                     self.referenceFiniteElement.referenceElement.dim),
                                    'd')
        for k in range_n_xi:
            for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                self.grad_psi[k,j,:] = self.referenceFiniteElement.localFunctionSpace.basisGradients[j](xiArray[k])
        return self.grad_psi
    def getBasisGradientValues(self,
                               xiArray,
                               inverseJacobianArray,
                               grad_vArray):
        '''
        This function calculates the BasisGradientValues for
        calculations on the reference element.

        Parameters
        ----------
        xiArray : input, :obj:`list` of :obj:`tuple`
            A list of quadrature points (x,y,z) in 2D case, z = 0.
        inverseJacobianArray : input, :obj:`numpy.array`
            Values of the inverseJacobian matrix used in the affine
            transformation from the physical domain to the reference
            element.
        grad_vArray : output, :obj:`numpy.array`
            Gradient values of basis functions on reference triangle,
            adjusted for transformation from physical domain.
        '''
        grad_vArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        grad_psi = numpy.zeros((n_xi,
                                  self.referenceFiniteElement.localFunctionSpace.dim,
                                  self.referenceFiniteElement.referenceElement.dim),
                                 'd')
        for k in range_n_xi:
            for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                grad_psi[k,j,:] = self.referenceFiniteElement.localFunctionSpace.basisGradients[j](xiArray[k])
        if self.useC == True:
            cfemIntegrals.parametricFiniteElementSpace_getGradientValues(grad_psi,
                                                                         inverseJacobianArray,
                                                                         grad_vArray)
        else:
            for eN in range(self.elementMaps.mesh.nElements_global):
                for k in range_n_xi:
                    for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                        for m in self.referenceFiniteElement.referenceElement.range_dim:
                            for n in self.referenceFiniteElement.referenceElement.range_dim:
                                grad_vArray[eN,k,j,m] += grad_psi[k,j,n]*inverseJacobianArray[eN,k,n,m]
    def getBasisHessianValuesRef(self,
                                 xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.Hessian_psi = numpy.zeros((n_xi,
                                        self.referenceFiniteElement.localFunctionSpace.dim,
                                        self.referenceFiniteElement.referenceElement.dim,
                                        self.referenceFiniteElement.referenceElement.dim),
                                       'd')
        for k in range_n_xi:
            for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                self.Hessian_psi[k,j,:] = self.referenceFiniteElement.localFunctionSpace.basisHessians[j](xiArray[k])
        return self.Hessian_psi
    def getBasisHessianValues(self,
                              xiArray,
                              inverseJacobianArray,
                              Hessian_vArray):
        Hessian_vArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        Hessian_psi = numpy.zeros((n_xi,
                                   self.referenceFiniteElement.localFunctionSpace.dim,
                                   self.referenceFiniteElement.referenceElement.dim,
                                   self.referenceFiniteElement.referenceElement.dim),
                                  'd')
        for k in range_n_xi:
            for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                Hessian_psi[k,j,:] = self.referenceFiniteElement.localFunctionSpace.basisHessians[j](xiArray[k])
        if self.useC == True:
            cfemIntegrals.parametricFiniteElementSpace_getHessianValues(Hessian_psi,
                                                                        inverseJacobianArray,
                                                                        Hessian_vArray)
    def getBasisGradientValuesAtArray(self,
                                      xiArrayArray,
                                      inverseJacobianArray,
                                      grad_vArray):
        grad_vArray.flat[:]=0.0
        n_xi = xiArrayArray.shape[1]
        range_n_xi = list(range(n_xi))
        for eN in range(self.elementMaps.mesh.nElements_global):
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    grad_psi = self.referenceFiniteElement.localFunctionSpace.basisGradients[j](xiArrayArray[eN,k,:self.referenceFiniteElement.referenceElement.dim])
                    for m in self.referenceFiniteElement.referenceElement.range_dim:
                        for n in self.referenceFiniteElement.referenceElement.range_dim:
                            grad_vArray[eN,k,j,m] += grad_psi[n]*inverseJacobianArray[eN,k,n,m]
    def getBasisValuesTrace(self,
                            permutations,
                            xiArray,
                            vArray):
        '''
        This function calculates the basis function values on the trace of the element boundaries.
        permutations (input)    - 
        xiArray (input)         - a list of the element boundary quarature points mapped from the physical domain to the reference triangle.
        vArray (output)         - the vector to store the basis function trace values on the reference triangle
        '''
        n_xi = xiArray.shape[2]
        range_n_xi = list(range(n_xi))
        psi = numpy.zeros((self.referenceFiniteElement.referenceElement.nElementBoundaries*n_xi,
                             self.referenceFiniteElement.localFunctionSpace.dim),
                            'd')
        for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    psi[ebN*n_xi+k,j] = self.referenceFiniteElement.localFunctionSpace.basis[j](xiArray[0,ebN,k])
        if self.useC == True:
            cfemIntegrals.parametricFiniteElementSpace_getValuesTrace(psi,
                                                                      permutations,
                                                                      vArray)
        else:
            for eN in range(self.elementMaps.mesh.nElements_global):
                for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
                    for k in range_n_xi:
                        for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                            vArray[eN,ebN,k,j] = psi[permutations[eN,ebN,k],j]
    def getBasisValuesTraceAtArray(self,
                                   xiArrayArray,
                                   vArray):
        n_xi = xiArrayArray.shape[2]
        range_n_xi = list(range(n_xi))
        for eN in range(self.elementMaps.mesh.nElements_global):
            for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
                for k in range_n_xi:
                    for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                        #mwf now manually map from \bar{x} (reference element boundary quadrature point to reference element space
                        #and then evaluate using basis, since basisTrace will be deprecated
                        #vArray[eN,ebN,k,j] = self.referenceFiniteElement.localFunctionSpace.basisTrace[ebN][j](xiArrayArray[eN,ebN,k,:self.referenceFiniteElement.referenceElement.dim])
                        xiHat_k = self.referenceFiniteElement.referenceElement.boundaryMapList[ebN](xiArrayArray[eN,ebN,k,:self.referenceFiniteElement.referenceElement.dim])
                        vArray[eN,ebN,k,j] = self.referenceFiniteElement.localFunctionSpace.basis[j](xiHat_k)
    def getBasisGradientValuesTrace(self,
                                    permutations,
                                    xiArray,
                                    inverseJacobianTraceArray,
                                    grad_vArray):
        grad_vArray.flat[:]=0.0
        n_xi = xiArray.shape[2]
        range_n_xi = list(range(n_xi))
        grad_psi = numpy.zeros((self.referenceFiniteElement.referenceElement.nElementBoundaries*n_xi,
                                  self.referenceFiniteElement.localFunctionSpace.dim,
                                  self.referenceFiniteElement.referenceElement.dim),
                                 'd')
        for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    grad_psi[ebN*n_xi+k,j,:] = self.referenceFiniteElement.localFunctionSpace.basisGradients[j](xiArray[0,ebN,k])
        if self.useC == True:
            cfemIntegrals.parametricFiniteElementSpace_getGradientValuesTrace(grad_psi,
                                                                              permutations,
                                                                              inverseJacobianTraceArray,
                                                                              grad_vArray)
        else:
            for eN in range(self.elementMaps.mesh.nElements_global):
                for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
                    for k in range_n_xi:
                        for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                            for m in self.referenceFiniteElement.referenceElement.range_dim:
                                for n in self.referenceFiniteElement.referenceElement.range_dim:
                                    grad_vArray[eN,ebN,k,j,m] += grad_psi[permutations[eN,ebN,k],j,n]*inverseJacobianTraceArray[eN,ebN,k,n,m]
    def getBasisGradientValuesTraceAtArray(self,
                                           xiArrayArray,
                                           inverseJacobianTraceArray,
                                           grad_vArray):
        grad_vArray.flat[:]=0.0
        n_xi = xiArrayArray.shape[2]
        range_n_xi = list(range(n_xi))
        for eN in range(self.elementMaps.mesh.nElements_global):
            for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
                for k in range_n_xi:
                    for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                        #mwf move away from using basisGradientsTrace directly since will be deprecated
                        #switch to using boundaryMapList directly
                        #grad_psi = self.referenceFiniteElement.localFunctionSpace.basisGradientsTrace[ebN][j](xiArrayArray[eN,ebN,k,:self.referenceFiniteElement.referenceElement.dim])
                        xiHat_k = self.referenceFiniteElement.referenceElement.boundaryMapList[ebN](xiArrayArray[eN,ebN,k,:self.referenceFiniteElement.referenceElement.dim])
                        grad_psi = self.referenceFiniteElement.localFunctionSpace.basisGradients[j](xiHat_k)
                        for m in self.referenceFiniteElement.referenceElement.range_dim:
                            for n in self.referenceFiniteElement.referenceElement.range_dim:
                                grad_vArray[eN,ebN,k,j,m] += grad_psi[n]*inverseJacobianTraceArray[eN,ebN,k,n,m]

    def getBasisValuesTraceRef(self,xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.psi_trace = numpy.zeros((self.referenceFiniteElement.referenceElement.nElementBoundaries,
                           n_xi,
                           self.referenceFiniteElement.localFunctionSpace.dim),
                          'd')
        for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    #mwf now manually map from \bar{x} (reference element boundary quadrature point to reference element space
                    #and then evaluate using basis, since basisTrace will be deprecated
                    #psi[ebN,k,j] = self.referenceFiniteElement.localFunctionSpace.basisTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceFiniteElement.referenceElement.boundaryMapList[ebN](xiArray[k])
                    self.psi_trace[ebN,k,j] = self.referenceFiniteElement.localFunctionSpace.basis[j](xiHat_k)
        return self.psi_trace
    def getBasisValuesGlobalExteriorTrace(self,
                                          xiArray,
                                          vArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        psi = numpy.zeros((self.referenceFiniteElement.referenceElement.nElementBoundaries,
                           n_xi,
                           self.referenceFiniteElement.localFunctionSpace.dim),
                          'd')
        for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    #mwf now manually map from \bar{x} (reference element boundary quadrature point to reference element space
                    #and then evaluate using basis, since basisTrace will be deprecated
                    #psi[ebN,k,j] = self.referenceFiniteElement.localFunctionSpace.basisTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceFiniteElement.referenceElement.boundaryMapList[ebN](xiArray[k])
                    psi[ebN,k,j] = self.referenceFiniteElement.localFunctionSpace.basis[j](xiHat_k)
        if self.useC == True:
            cfemIntegrals.parametricFiniteElementSpace_getValuesGlobalExteriorTrace(self.elementMaps.mesh.exteriorElementBoundariesArray,
                                                                                    self.elementMaps.mesh.elementBoundaryElementsArray,
                                                                                    self.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                    psi,
                                                                                    vArray)
        else:
            for ebNE in range(self.elementMaps.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.elementMaps.mesh.exteriorElementBoundariesArray[ebNE]
                eN  = self.mesh.elementMaps.mesh.elementBoundaryElementsArray[ebN,0]
                ebN_local = self.mesh.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                for k in range_n_xi:
                    for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                        vArray[ebNE,k,j] = psi[ebN_local,k,j]


    def getBasisValuesGlobalExteriorTraceAtArray(self,
                                                 xiArrayArray,
                                                 vArray):
        n_xi = xiArrayArray.shape[1]
        range_n_xi = list(range(n_xi))
        for ebNE in range(self.elementMaps.mesh.nExteriorElementBoundaries_global):
            ebN = self.elementMaps.mesh.exteriorElementBoundariesArray[ebNE]
            ebN_local = self.elementMaps.elementBoundayLocalElementBoundariesArray[ebN,0]
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    #mwf now manually map from \bar{x} (reference element boundary quadrature point to reference element space
                    #and then evaluate using basis, since basisTrace will be deprecated
                    #vArray[ebNE,k,j] = self.referenceFiniteElement.localFunctionSpace.basisTrace[ebN_local][j](xiArrayArray[ebNE,k,:self.referenceFiniteElement.referenceElement.dim])
                    xiHat_k = self.referenceFiniteElement.referenceElement.boundaryMapList[ebN_local](xiArrayArray[ebNE,k,:self.referenceFiniteElement.referenceElement.dim])
                    vArray[ebNE,k,j] = self.referenceFiniteElement.localFunctionSpace.basis[j](xiHat_k)
    def getBasisGradientValuesTraceRef(self,
                                       xiArray):
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        self.grad_psi_trace = numpy.zeros((self.referenceFiniteElement.referenceElement.nElementBoundaries,
                                           n_xi,
                                           self.referenceFiniteElement.localFunctionSpace.dim,
                                           self.referenceFiniteElement.referenceElement.dim),
                                          'd')
        for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                # Map the boundary point onto face ebN first, as getBasisValuesTraceRef does. Without
                # this every face's gradients are taken on reference face 2; that is invisible for P1
                # (constant gradients) but wrong for P2 and higher.
                xiHat_k = self.referenceFiniteElement.referenceElement.boundaryMapList[ebN](xiArray[k])
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    self.grad_psi_trace[ebN,k,j,:] = self.referenceFiniteElement.localFunctionSpace.basisGradients[j](xiHat_k)
        return self.grad_psi_trace
    def getBasisGradientValuesGlobalExteriorTrace(self,
                                                  xiArray,
                                                  inverseJacobianTraceArray,
                                                  grad_vArray):
        grad_vArray.flat[:]=0.0
        n_xi = xiArray.shape[0]
        range_n_xi = list(range(n_xi))
        grad_psi = numpy.zeros((self.referenceFiniteElement.referenceElement.nElementBoundaries,
                                n_xi,
                                self.referenceFiniteElement.localFunctionSpace.dim,
                                self.referenceFiniteElement.referenceElement.dim),
                               'd')

        for ebN in self.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    #mwf move away from using basisGradientsTrace directly since will be deprecated
                    #switch to using boundaryMapList directly
                    #grad_psi[ebN,k,j,:] = self.referenceFiniteElement.localFunctionSpace.basisGradientsTrace[ebN][j](xiArray[k])
                    xiHat_k = self.referenceFiniteElement.referenceElement.boundaryMapList[ebN](xiArray[k])
                    grad_psi[ebN,k,j,:] = self.referenceFiniteElement.localFunctionSpace.basisGradients[j](xiHat_k)
        if self.useC == True:
            cfemIntegrals.parametricFiniteElementSpace_getGradientValuesGlobalExteriorTrace(self.elementMaps.mesh.exteriorElementBoundariesArray,
                                                                                            self.elementMaps.mesh.elementBoundaryElementsArray,
                                                                                            self.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                            grad_psi,
                                                                                            inverseJacobianTraceArray,
                                                                                            grad_vArray)
        else:
            for ebNE in range(self.elementMaps.mesh.nExteriorElementBoundaries_global):
                ebN = self.elementMaps.mesh.exteriorElementBoundariesArray[ebNE]
                eN  = self.elementMaps.mesh.elementBoundaryElementsArray[ebN,0]
                ebN_local = self.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                for k in range_n_xi:
                    for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                        for m in self.referenceFiniteElement.referenceElement.range_dim:
                            for n in self.referenceFiniteElement.referenceElement.range_dim:
                                grad_vArray[ebNE,k,j,m] += grad_psi[ebN_local,k,j,n]*inverseJacobianArray[ebNE,k,n,m]
    def getBasisGradientValuesGlobalExteriorTraceAtArray(self,
                                                         xiArrayArray,
                                                         inverseJacobianTraceArray,
                                                         grad_vArray):
        grad_vArray.flat[:]=0.0
        n_xi = xiArrayArray.shape[1]
        range_n_xi = list(range(n_xi))
        for ebNE in range(self.elementMaps.mesh.nExteriorElementBoundaries_global):
            ebN = self.elementMaps.mesh.exteriorElementBoundariesArray[ebNE]
            eN  = self.elementMaps.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_local = self.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            for k in range_n_xi:
                for j in self.referenceFiniteElement.localFunctionSpace.range_dim:
                    #mwf move away from using basisGradientsTrace directly since will be deprecated
                    #switch to using boundaryMapList directly
                    #grad_psi = self.referenceFiniteElement.localFunctionSpace.basisGradientsTrace[ebN_local][j](xiArrayArray[ebNE,k,:self.referenceFiniteElement.referenceElement.dim])
                    xiHat_k = self.referenceFiniteElement.referenceElement.boundaryMapList[ebN_local](xiArrayArray[ebNE,k,:self.referenceFiniteElement.referenceElement.dim])
                    grad_psi = self.referenceFiniteElement.localFunctionSpace.basisGradients[j](xiHat_k)
                    for m in self.referenceFiniteElement.referenceElement.range_dim:
                        for n in self.referenceFiniteElement.referenceElement.range_dim:
                            grad_vArray[ebNE,k,j,m] += grad_psi[n]*inverseJacobianTraceArray[ebNE,k,n,m]

    def updateInterpolationPoints(self):
        import pdb
       # pdb.set_trace()
        self.elementMaps.getValues(self.referenceFiniteElement.interpolationConditions.quadraturePointArray,self.interpolationPoints)
        return self.interpolationPoints
    def endTimeSeriesEnsight(self,timeValues,filename,description,ts=1):
        #cek this could break something, but it should be true:
        self.nOutput = len(timeValues)
        lines = ('TIME\n'+'time set: '+repr(ts)+' '+description+'\n'+
                 'number of steps: '+ repr(self.nOutput)+'\n'+
                 'filename start number: 0\n'+
                 'filename increment: 1\n'+
                 'time values:')
        caseOut=open(filename+'.case','a')
        caseOut.write(lines)
        for tn in timeValues:
            caseOut.write(' %12.5E\n' %tn)
        caseOut.close()

    def getValuesAtMeshNodes(self,dof,nodalValues,isVector,dim_dof):
        """
        we would like to be able to get references to existing values for values at nodes for some calculations
        like vtkVisualization
        """
        raise NotImplementedError


class C0_AffineLinearOnSimplexWithNodalBasis(ParametricFiniteElementSpace):
    """
    The standard linear CG space.

    Globally C0
    Each geometric element is the image of the reference simplex under
    a linear affine mapping. The nodal basis is used on the reference simplex.
    """
    def __init__(self,mesh,nd=3):
        self.order = 1
        localFunctionSpace = LinearOnSimplexWithNodalBasis(nd)
        interpolationConditions = NodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              NodalDOFMap(mesh))
    def writeMeshEnsight(self,filename,description=None):
        self.mesh.writeMeshEnsight(filename,description)
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        self.mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount)
        return self.mesh.arGrid
    def writeFunctionGnuplot(self,u,filename):
        import Gnuplot
        if self.referenceFiniteElement.referenceElement.dim == 1:
            if self.viewer is None:
                self.viewer = Gnuplot.Gnuplot()
                #self.viewer("set terminal x11")
            self.viewer.plot(Gnuplot.Data(self.elementMaps.mesh.nodeArray[:,0],
                                    u.dof,

                                    title=filename))
        elif self.referenceFiniteElement.referenceElement.dim == 2:
            if self.viewer is None:
                self.viewer = Gnuplot.Gnuplot()
                #self.viewer("set terminal x11")
            nx = sqrt(self.elementMaps.mesh.nNodes_global)
            ny = nx
            x = numpy.arange(nx,dtype='i')/float(nx-1)
            y = numpy.arange(nx,dtype='i')/float(nx-1)
            nSol = numpy.reshape(u.dof,(nx,ny))
            self.viewer('set parametric')
            self.viewer('set data style lines')
            self.viewer('set hidden')
            self.viewer('set contour base')
            self.viewer.xlabel('x')
            self.viewer.ylabel('y')
            self.viewer.splot(Gnuplot.GridData(nSol,
                                               x,
                                               y,
                                               binary=0,
                                               inline=0,
                                               title=filename))
    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        if ar.global_sync:
            attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":u.name,
                                                     "AttributeType":"Scalar",
                                                     "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i" % (self.mesh.globalMesh.nNodes_global,)})
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+u.name+"_t{0:d}".format(tCount)
                comm = Comm.get()
                ar.create_dataset_sync(u.name+"_t{0:d}".format(tCount),
                                       #need to use the mesh node offsets due to parallel periodic case
                                       #offsets = self.dofMap.dof_offsets_subdomain_owned,
                                       #data = u.dof[:(self.dofMap.dof_offsets_subdomain_owned[comm.rank()+1] -self.dofMap.dof_offsets_subdomain_owned[comm.rank()])])
                                       offsets=self.mesh.globalMesh.nodeOffsets_subdomain_owned,
                                       data = u.dof[:(self.mesh.globalMesh.nodeOffsets_subdomain_owned[comm.rank()+1] -self.mesh.globalMesh.nodeOffsets_subdomain_owned[comm.rank()])])
            else:
                assert False, "global_sync not supported with text heavy data"
        else:
            attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":u.name,
                                                                 "AttributeType":"Scalar",
                                                                 "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i" % (self.mesh.nNodes_global,)})
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+u.name+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount)
                ar.create_dataset_async(u.name+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount), data = u.dof)
            else:
                numpy.savetxt(ar.textDataDir+"/"+"{0:s}{1:d}".format(u.name, tCount)+".txt",u.dof)
                SubElement(values,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/"+"{0:s}{1:d}".format(u.name, tCount)+".txt"})
    def readFunctionXdmf(self,ar,u,tCount=0):
        if ar.hdfFile is not None:
            if ar.hdfFileGlb is not None:
                map = self.mesh.globalMesh.nodeNumbering_subdomain2global
                array=ar.hdfFileGlb.get_node("/","{0:s}{1:d}".format(u.name, tCount))
                for i in range(len(map)):
                    u.dof[i] = array[map[i]]
                del array
            else:
                if ar.global_sync:
                    #this is known to be slow but it scales with
                    #respect to memory (as opposed to the faster
                    #approach of pulling in the entire array)
                    permute = np.argsort(self.mesh.globalMesh.nodeNumbering_subdomain2global)
                    u.dof[permute] = ar.hdfFile["/"+u.name+"_t{0:d}".format(tCount)][self.mesh.globalMesh.nodeNumbering_subdomain2global[permute].tolist()]
                    #faster way
                    #u.dof[:] = ar.hdfFile["/"+u.name+"_t{0:d}".format(tCount)].value[self.mesh.globalMesh.nodeNumbering_subdomain2global]                        
                else:
                    u.dof[:]=ar.hdfFile["/"+u.name+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount)]
        else:
            assert(False)
            #numpy.savetxt(ar.textDataDir+"/"+"{0:s}{0:d}".format(u.name, tCount)+".txt",u.dof)
            #SubElement(values,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/"+"{0:s}{0:d}".format(u.name, tCount)+".txt"})
    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        concatNow=True
        if concatNow:
            if ar.global_sync:
                attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":vectorName,
                                                                     "AttributeType":"Vector",
                                                                     "Center":"Node"})
                values    = SubElement(attribute,"DataItem",
                                       {"Format":ar.dataItemFormat,
                                        "DataType":"Float",
                                        "Precision":"8",
                                        "Dimensions":"%i %i" % (self.mesh.globalMesh.nNodes_global,3)})
                u_dof = uList[components[0]].dof
                if len(components) < 2:
                    v_dof = numpy.zeros(u_dof.shape,dtype='d')
                else:
                    v_dof = uList[components[1]].dof
                if len(components) < 3:
                    w_dof = numpy.zeros(u_dof.shape,dtype='d')
                else:
                    w_dof = uList[components[2]].dof
                velocity = numpy.column_stack((u_dof,v_dof,w_dof))
                if ar.hdfFile is not None:
                    values.text = ar.hdfFilename+":/"+vectorName+"_t{0:d}".format(tCount)
                    ar.create_dataset_sync(vectorName+"_t{0:d}".format(tCount),
                                           offsets=self.mesh.globalMesh.nodeOffsets_subdomain_owned,
                                           data = velocity[:self.mesh.nNodes_owned,:])
            else:
                attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":vectorName,
                                                                     "AttributeType":"Vector",
                                                                     "Center":"Node"})
                values    = SubElement(attribute,"DataItem",
                                       {"Format":ar.dataItemFormat,
                                        "DataType":"Float",
                                        "Precision":"8",
                                        "Dimensions":"%i %i" % (self.mesh.nNodes_global,3)})
                u_dof = uList[components[0]].dof
                if len(components) < 2:
                    v_dof = numpy.zeros(u_dof.shape,dtype='d')
                else:
                    v_dof = uList[components[1]].dof
                if len(components) < 3:
                    w_dof = numpy.zeros(u_dof.shape,dtype='d')
                else:
                    w_dof = uList[components[2]].dof
                velocity = numpy.column_stack((u_dof,v_dof,w_dof))
                if ar.hdfFile is not None:
                    values.text = ar.hdfFilename+":/"+vectorName+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount)
                    ar.create_dataset_async(vectorName+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount), data = velocity)
        else:
            attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":vectorName,
                                                        "AttributeType":"Vector",
                                                        "Center":"Node"})
            if len(components) == 2:
                values    = SubElement(attribute,"DataItem",
                                       {"ItemType":"Function",
                                        "Function":"JOIN($0 , $1 , (0.0 * $1 ))",
                                        "Dimensions":"%i %i" % (self.mesh.nNodes_global,3)})
            elif len(components) == 3:
                values    = SubElement(attribute,"DataItem",
                                       {"ItemType":"Function",
                                        "Function":"JOIN($0 , $1 , $2)",
                                        "Dimensions":"%i %i" % (self.mesh.nNodes_global,3)})
            for ci in components:
                ReferenceString="/Xdmf/Domain/Grid/Grid[%i]/Attribute[%i]/DataItem" % (tCount+1,ci+1)
                component = SubElement(values,"DataItem",{"Reference":ReferenceString})
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'.vec****\n')
            uOut=open(filename+u.name+'.vec%4.4i' % self.nOutput,'w')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+' '+filename+u.name+'.scl****\n')
            uOut=open(filename+u.name+'.scl%4.4i' % self.nOutput,'w')
        uOut.write(u.name+'\n')
        uOut.write('part\n'+'%10i\n' % 1)
        uOut.write('coordinates\n')
        for t in u.range_dim_dof:
            cfemIntegrals.writeDOF(u.femSpace.dim,u.dim_dof,t,'%12.5e\n',u.dof,uOut)
            #for n in u.femSpace.range_dim:
            #    uOut.write('%12.5e\n' % u.dof[n*u.dim_dof + t])
        if u.isVector:
            for t in range(3 - u.dim_dof):
                for n in u.femSpace.range_dim:
                    uOut.write('%12.5e\n' % 0.0)
        uOut.close()
        self.nOutput+=1
    def writeE2VectorFunctionEnsight(self,u,v,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if not append:
            caseOut=open(case_filename+'.case','a')
            if firstVariable==True:
                caseOut.write('VARIABLE\n')
            caseOut.write('vector per node: '+
                          u.name+'2 '+filename+u.name+'2.vec****\n')
        uOut=open(filename+u.name+'2.vec%4.4i' % nOutput,'w')
        uOut.write(u.name+'\n')
        uOut.write('part\n'+'%10i\n' % 1)
        uOut.write('coordinates\n')
        for t in u.range_dim_dof:
            cfemIntegrals.writeDOF(u.femSpace.dim,u.dim_dof,t,'%12.5e\n',u.dof,uOut)
            cfemIntegrals.writeDOF(v.femSpace.dim,v.dim_dof,t,'%12.5e\n',v.dof,uOut)
            cfemIntegrals.writeDOF_ZEROS(v.femSpace.dim,v.dim_dof,t,'%12.5e\n',uOut)
        uOut.close()
    def writeE2VectorFunctionHeaderEnsight(self,u,v,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if not append:
            caseOut=open(case_filename+'.case','a')
            if firstVariable==True:
                caseOut.write('VARIABLE\n')
            caseOut.write('vector per node: '+
                          u.name+'2 '+filename+u.name+'2.vec****\n')
        caseOut.close()
    def writeE3VectorFunctionEnsight(self,u,v,w,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if not append:
            caseOut=open(case_filename+'.case','a')
            if firstVariable==True:
                caseOut.write('VARIABLE\n')
            caseOut.write('vector per node: '+
                          u.name+'2 '+filename+u.name+'2.vec****\n')
        uOut=open(filename+u.name+'2.vec%4.4i' % nOutput,'w')
        uOut.write(u.name+'\n')
        uOut.write('part\n'+'%10i\n' % 1)
        uOut.write('coordinates\n')
        for t in u.range_dim_dof:
            cfemIntegrals.writeDOF(u.femSpace.dim,u.dim_dof,t,'%12.5e\n',u.dof,uOut)
            cfemIntegrals.writeDOF(v.femSpace.dim,v.dim_dof,t,'%12.5e\n',v.dof,uOut)
            cfemIntegrals.writeDOF(w.femSpace.dim,w.dim_dof,t,'%12.5e\n',w.dof,uOut)
        uOut.close()
    def writeE3VectorFunctionHeaderEnsight(self,u,v,w,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if not append:
            caseOut=open(case_filename+'.case','a')
            if firstVariable==True:
                caseOut.write('VARIABLE\n')
            caseOut.write('vector per node: '+
                          u.name+'2 '+filename+u.name+'2.vec****\n')
            caseOut.close()
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'.vec****\n')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+' '+filename+u.name+'.scl****\n')
    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_LagrangeC0P1(fout,
                                                  u.femSpace.nSpace_global,
                                                  u.femSpace.elementMaps.mesh.nodeArray,
                                                  u.femSpace.elementMaps.mesh.elementNodesArray,
                                                  u.dof,
                                                  name=u.name,
                                                  storeMeshData=storeMeshData,
                                                  figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0
    def getValuesAtMeshNodes(self,dof,nodalValues,isVector,dim_dof):
        """
        """
        nodalValues.flat[:] = dof.flat[:]

class C0_LinearOnCubeWithNodalBasis(C0_AffineLinearOnSimplexWithNodalBasis):
    """
    The standard linear CG space.

    Globally C0
    Each geometric element is the image of the reference cube under
    a n-linear(non-affine) mapping. The nodal basis is used on the reference cube.
    """
    def __init__(self,mesh,nd=3):
        localFunctionSpace = LinearOnCubeWithNodalBasis(nd)
        interpolationConditions = CubeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              ParametricMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnCubeWithNodalBasis(nd)),
                                              NodalDOFMap(mesh))
class C0_AffineLinearOnCubeWithNodalBasis(ParametricFiniteElementSpace):
    """
    The standard linear CG space.

    Globally C0
    Each geometric element is the image of the reference simplex under
    a linear affine mapping. The nodal basis is used on the reference simplex.
    """
    def __init__(self,mesh,nd=3):
        localFunctionSpace = LinearOnCubeWithNodalBasis(nd)
        interpolationConditions = CubeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnCubeWithNodalBasis(nd)),
                                              NodalDOFMap(mesh))

    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        self.mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount)
        return self.mesh.arGrid

    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        comm = Comm.get()
        if ar.global_sync:
            attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":u.name,
                                                                 "AttributeType":"Scalar",
                                                                 "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i" % (self.mesh.globalMesh.nNodes_global,)})
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+u.name+"_t{0:d}".format(tCount)
                ar.create_dataset_sync(u.name+"_t{0:d}".format(tCount),
                                       #need to use the mesh node offsets due to parallel periodic case
                                       #offsets = self.dofMap.dof_offsets_subdomain_owned,
                                       #data = u.dof[:(self.dofMap.dof_offsets_subdomain_owned[comm.rank()+1] - self.dofMap.dof_offsets_subdomain_owned[comm.rank()])],
                                       offsets=self.mesh.globalMesh.nodeOffsets_subdomain_owned,
                                       data = u.dof[:(self.mesh.globalMesh.nodeOffsets_subdomain_owned[comm.rank()+1] -self.mesh.globalMesh.nodeOffsets_subdomain_owned[comm.rank()])])
            else:
                assert False, "global_sync not supported with text heavy data"
        else:
            attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":u.name,
                                                                 "AttributeType":"Scalar",
                                                                 "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i" % (self.mesh.nNodes_global,)})
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+u.name+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount)
                ar.create_dataset_async(u.name+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount), data = u.dof)
            else:
                numpy.savetxt(ar.textDataDir+"/"+"{0:s}{1:d}".format(u.name, tCount)+".txt",u.dof)
                SubElement(values,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/"+"{0:s}{1:d}".format(u.name, tCount)+".txt"})

    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        concatNow=True
        if concatNow:
            if ar.global_sync:
                attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":vectorName,
                                                                     "AttributeType":"Vector",
                                                                     "Center":"Node"})
                values    = SubElement(attribute,"DataItem",
                                       {"Format":ar.dataItemFormat,
                                        "DataType":"Float",
                                        "Precision":"8",
                                        "Dimensions":"%i %i" % (self.mesh.globalMesh.nNodes_global,3)})
                u_dof = uList[components[0]].dof
                if len(components) < 2:
                    v_dof = numpy.zeros(u_dof.shape,dtype='d')
                else:
                    v_dof = uList[components[1]].dof
                if len(components) < 3:
                    w_dof = numpy.zeros(u_dof.shape,dtype='d')
                else:
                    w_dof = uList[components[2]].dof
                velocity = numpy.column_stack((u_dof,v_dof,w_dof))
                if ar.hdfFile is not None:
                    values.text = ar.hdfFilename+":/"+vectorName+"_t{0:d}".format(tCount)
                    ar.create_dataset_sync(vectorName+"_t{0:d}".format(tCount),
                                           offsets =self.mesh.globalMesh.nodeOffsets_subdomain_owned,
                                           data = velocity[:self.mesh.nNodes_owned,:])
            else:
                attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":vectorName,
                                                                     "AttributeType":"Vector",
                                                                     "Center":"Node"})
                values    = SubElement(attribute,"DataItem",
                                       {"Format":ar.dataItemFormat,
                                        "DataType":"Float",
                                        "Precision":"8",
                                        "Dimensions":"%i %i" % (self.mesh.nNodes_global,3)})
                u_dof = uList[components[0]].dof
                if len(components) < 2:
                    v_dof = numpy.zeros(u_dof.shape,dtype='d')
                else:
                    v_dof = uList[components[1]].dof
                if len(components) < 3:
                    w_dof = numpy.zeros(u_dof.shape,dtype='d')
                else:
                    w_dof = uList[components[2]].dof
                velocity = numpy.column_stack((u_dof,v_dof,w_dof))
                if ar.hdfFile is not None:
                    values.text = ar.hdfFilename+":/"+vectorName+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount)
                    ar.create_dataset_async(vectorName+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount), data = velocity)

        else:
            if ar.global_sync:
                attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":vectorName,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
                if len(components) == 2:
                    values    = SubElement(attribute,"DataItem",
                                           {"ItemType":"Function",
                                            "Function":"JOIN($0 , $1 , (0.0 * $1 ))",
                                            "Dimensions":"%i %i" % (self.mesh.globalMesh.nNodes_global,3)})
                elif len(components) == 3:
                    values    = SubElement(attribute,"DataItem",
                                           {"ItemType":"Function",
                                            "Function":"JOIN($0 , $1 , $2)",
                                            "Dimensions":"%i %i" % (self.mesh.globalMesh.nNodes_global,3)})
                for ci in components:
                    ReferenceString="/Xdmf/Domain/Grid/Grid[%i]/Attribute[%i]/DataItem" % (tCount+1,ci+1)
                    component = SubElement(values,"DataItem",{"Reference":ReferenceString})
            else:
                attribute = SubElement(self.mesh.arGrid,"Attribute",{"Name":vectorName,
                                                                     "AttributeType":"Vector",
                                                                     "Center":"Node"})
                if len(components) == 2:
                    values    = SubElement(attribute,"DataItem",
                                           {"ItemType":"Function",
                                            "Function":"JOIN($0 , $1 , (0.0 * $1 ))",
                                            "Dimensions":"%i %i" % (self.mesh.nNodes_global,3)})
                elif len(components) == 3:
                    values    = SubElement(attribute,"DataItem",
                                           {"ItemType":"Function",
                                            "Function":"JOIN($0 , $1 , $2)",
                                            "Dimensions":"%i %i" % (self.mesh.nNodes_global,3)})
                for ci in components:
                    ReferenceString="/Xdmf/Domain/Grid/Grid[%i]/Attribute[%i]/DataItem" % (tCount+1,ci+1)
                    component = SubElement(values,"DataItem",{"Reference":ReferenceString})

P1 = C0_AffineLinearOnSimplexWithNodalBasis

class C0_LagrangeOnCubeWithNodalBasis(C0_AffineLinearOnSimplexWithNodalBasis):
    """
    The standard linear CG space.

    Globally C0
    Each geometric element is the image of the reference cube under
    a n-linear(non-affine) mapping. The nodal basis is used on the reference cube.
    """
    def __init__(self,mesh,nd=3,order=2):
        localFunctionSpace = LagrangeOnCubeWithNodalBasis(nd,order=2)
        #todo fix these interpolation conditions to work on Cube
        interpolationConditions = QuadraticLagrangeCubeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              ParametricMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LagrangeOnCubeWithNodalBasis(nd,order=mesh.px)),
                                              QuadraticLagrangeCubeDOFMap(mesh))

        print("C0_LagrangeOnCubeWithNodalBasis")
        print(mesh.px)

class C0_AffineLagrangeOnCubeWithNodalBasis(ParametricFiniteElementSpace):
    """
    The standard linear CG space.

    Globally C0
    Each geometric element is the image of the reference simplex under
    a linear affine mapping. The nodal basis is used on the reference simplex.
    """
    def __init__(self,mesh,nd=3,order=2):
        self.order = order
        localGeometricSpace= LinearOnCubeWithNodalBasis(nd)
        #todo fix these interpolation conditions to work on Cube
        if self.order==2:
            localFunctionSpace = LagrangeOnCubeWithNodalBasis(nd,order=2)
            interpolationConditions = QuadraticLagrangeCubeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        # elif self.order==1:
        #     localFunctionSpace = LagrangeOnCubeWithNodalBasis(nd,order=1)
        #     interpolationConditions = CubeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        else:
            raise NotImplementedError ("Lagrange factory only implemented for Q2"
                    "elements so far. For Q1 use C0_AffineLinearOnCubeWithNodalBasis.")
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localGeometricSpace.referenceElement,
                                                         LinearOnCubeWithNodalBasis(nd)),
                                              QuadraticLagrangeCubeDOFMap(mesh,localFunctionSpace,nd))

        for i in range(localFunctionSpace.dim):
            for j in range(localFunctionSpace.dim):
                x_j = interpolationConditions.quadraturePointArray[j]
                psi_ij = localFunctionSpace.basis[i](x_j)
#                print i,j,x_j,psi_ij
                if i==j:
                    assert(abs(1.0-psi_ij) < 1.0e-8)
                else:
                    assert(abs(psi_ij) < 1.0e-8)
        #for archiving
        from . import Archiver
        self.XdmfWriter=Archiver.XdmfWriter()

    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        if self.order == 2:
            return self.XdmfWriter.writeMeshXdmf_C0Q2Lagrange(ar,name,mesh=self.mesh,spaceDim=self.nSpace_global,
                                                              dofMap=self.dofMap,t=t,init=init,meshChanged=meshChanged,
                                                              arGrid=arGrid,tCount=tCount)
        else:
            raise NotImplementedError ("Lagrange factory only implemented for Q2"
                    "elements so far. For Q1 use C0_AffineLinearOnCubeWithNodalBasis.")

    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        self.XdmfWriter.writeFunctionXdmf_C0P2Lagrange(ar,u,tCount=tCount,init=init)
    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        self.XdmfWriter.writeVectorFunctionXdmf_nodal(ar,uList,components,vectorName,"c0p2_Lagrange",tCount=tCount,init=init)


# Lagrange Factory On Cube
def LagrangeCubeFactory(OrderIn):
    class LagrangeCubeOrderN(C0_AffineLagrangeOnCubeWithNodalBasis):
        def __init__(self,mesh,nd):
            C0_AffineLagrangeOnCubeWithNodalBasis.__init__(self,mesh,nd,order=OrderIn)
    return LagrangeCubeOrderN

# TODO - migrate Q1 to an instance of LagrangeCubeFactor
Q1 = C0_AffineLinearOnCubeWithNodalBasis
Q2 = LagrangeCubeFactory(2)
C0_AffineQuadraticOnCubeWithNodalBasis = Q2

class C0_BernsteinOnCube(C0_AffineLinearOnSimplexWithNodalBasis):
    """
    The standard linear CG space.
    Globally C0
    Each geometric element is the image of the reference cube under
    a n-linear(non-affine) mapping. The Bernstein basis is used on the reference cube.
    """
    def __init__(self,mesh,nd=3,order=2):
        assert False, "Bernstein polynomials are not interpolatory. Remove this assert to use p2 Lagrange interpolatory conditions"
        localFunctionSpace = BernsteinOnCube(nd,order=2)
        # NOTE: Bernstein polynomials ARE NOT INTERPOLATORY but we use them as control points to create a bezier curve. 
        interpolationConditions = QuadraticLagrangeCubeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              ParametricMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LagrangeOnCubeWithNodalBasis(nd,order=mesh.px)),
                                              QuadraticLagrangeCubeDOFMap(mesh))

        print("C0_BernsteinOnCubeWithNodalBasis")
        print(mesh.px)

class C0_AffineBernsteinOnCube(ParametricFiniteElementSpace):
    """
    Bernstein CG space.
    Globally C0
    Each geometric element is the image of the reference simplex under
    a linear affine mapping. The Bernstein basis is used on the reference simplex.
    """
    def __init__(self,mesh,nd=3,order=2):
        self.order = order
        localGeometricSpace= LinearOnCubeWithNodalBasis(nd)
        #todo fix these interpolation conditions to work on Cube
        
        if self.order==2:
            localFunctionSpace = BernsteinOnCube(nd,order=2)
            # NOTE: Bernstein polynomials ARE NOT INTERPOLATORY but we use them as control points to create a bezier curve.
            interpolationConditions = QuadraticLagrangeCubeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        #elif self.order==1:
            #localFunctionSpace = LagrangeOnCubeWithNodalBasis(nd,order=1)
            #interpolationConditions = CubeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        else:
            raise NotImplementedError ("Lagrange factory only implemented for Q2"
                                       "elements so far. For Q1 use C0_AffineLinearOnCubeWithNodalBasis.")
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localGeometricSpace.referenceElement,
                                                         LinearOnCubeWithNodalBasis(nd)),
                                              QuadraticLagrangeCubeDOFMap(mesh,localFunctionSpace,nd))

        for i in range(localFunctionSpace.dim):
            for j in range(localFunctionSpace.dim):
                x_j = interpolationConditions.quadraturePointArray[j]
                psi_ij = localFunctionSpace.basis[i](x_j)
                #print i,j,x_j,psi_ij
        #for archiving
        from . import Archiver
        self.XdmfWriter=Archiver.XdmfWriter()

    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        if self.order == 2:
            return self.XdmfWriter.writeMeshXdmf_C0Q2Lagrange(ar,name,mesh=self.mesh,spaceDim=self.nSpace_global,
                                                              dofMap=self.dofMap,t=t,init=init,meshChanged=meshChanged,
                                                              arGrid=arGrid,tCount=tCount)
        else:
            raise NotImplementedError ("Lagrange factory only implemented for Q2"
                                       "elements so far. For Q1 use C0_AffineLinearOnCubeWithNodalBasis.")

    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        self.XdmfWriter.writeFunctionXdmf_C0P2Lagrange(ar,u,tCount=tCount,init=init)
    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        self.XdmfWriter.writeVectorFunctionXdmf_nodal(ar,uList,components,vectorName,"c0p2_Lagrange",tCount=tCount,init=init)

# Bernstein Factory On Cube
#def BernsteinCubeFactory(OrderIn):
#    class BernsteinCubeOrderN(C0_AffineBernsteinOnCube):
#        def __init__(self,mesh,nd):
#            C0_AffineBernsteinOnCube.__init__(self,mesh,nd,order=OrderIn)
#    return BernsteinCubeOrderN

# TODO - migrate Q1 to an instance of BernsteinCubeFactor
#Q1 = C0_AffineLinearOnCubeWithNodalBasis
#Q2 = BernsteinCubeFactory(2)

class DG_AffinePolynomialsOnSimplexWithMonomialBasis(ParametricFiniteElementSpace):
    def __init__(self,mesh,nd=3,k=0):
        localFunctionSpace = Monomials(nd,k)
        interpolationConditions = MonomialInterpolationConditions(localFunctionSpace.referenceElement,localFunctionSpace)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              DiscontinuousGalerkinDOFMap(mesh,localFunctionSpace))
        self.strongDirichletConditions = False
        #for archiving
        from . import Archiver
        self.XdmfWriter = Archiver.XdmfWriter()

    def writeMeshEnsight(self,filename,description=None):
        self.mesh.writeMeshEnsight(filename,description)#need to allow paraview to even read in quadpoint data
    def writeFunctionGnuplot(self,u,filename):
        pass
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        #mwf hack, to allow for output from quadrature arrays even if not plotting solution directly
        self.nOutput+= 1
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        pass
    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written

        Warning, currently has to compute values at interpolation points!
        tries to use basis values at interpolation points if in u
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            basisValuesAtInterpolationPoints = None
            if 'basisValuesAtInterpolationPoints' in dir(u):
                basisValuesAtInterpolationPoints = u.basisValuesAtInterpolationPoints
            else:
                u.basisValuesAtInterpolationPoints = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                                  u.femSpace.interpolationPoints.shape[1],
                                                                  u.femSpace.referenceFiniteElement.localFunctionSpace.dim),
                                                                 'd')
                u.femSpace.getBasisValues(u.femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                          u.basisValuesAtInterpolationPoints)

            interpolationValuesArray        = None
            if 'interpolationValuesArray' not in dir(u):
                u.interpolationValuesArray = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                          u.femSpace.interpolationPoints.shape[1]),'d')
            u.getValues(u.basisValuesAtInterpolationPoints,u.interpolationValuesArray)
            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_MonomialDGPK(fout,
                                                  u.femSpace.nSpace_global,
                                                  u.femSpace.elementMaps.mesh.nodeArray,
                                                  u.femSpace.elementMaps.mesh.elementNodesArray,
                                                  u.femSpace.interpolationPoints,
                                                  u.interpolationValuesArray,
                                                  name=u.name,
                                                  storeMeshData=storeMeshData,
                                                  figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_MonomialDGPK(ar,self.elementMaps.mesh,self.nSpace_global,
                                                          self.interpolationPoints,
                                                          t=t,init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)
    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        """
        not much choice except to get u values at interpolation points?
        """
        basisValuesAtInterpolationPoints = None
        if 'basisValuesAtInterpolationPoints' in dir(u):
            basisValuesAtInterpolationPoints = u.basisValuesAtInterpolationPoints
        else:
            u.basisValuesAtInterpolationPoints = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                              u.femSpace.interpolationPoints.shape[1],
                                                              u.femSpace.referenceFiniteElement.localFunctionSpace.dim),
                                                             'd')
            u.femSpace.getBasisValues(u.femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                      u.basisValuesAtInterpolationPoints)

        interpolationValuesArray        = None
        if 'interpolationValuesArray' not in dir(u):
            u.interpolationValuesArray = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                      u.femSpace.interpolationPoints.shape[1]),'d')
        u.getValues(u.basisValuesAtInterpolationPoints,u.interpolationValuesArray)

        return self.XdmfWriter.writeFunctionXdmf_MonomialDGPK(ar,u.interpolationValuesArray,u.name,tCount=tCount,init=init, mesh=u.femSpace.mesh)
    #
    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        """
        """
        logEvent("Monomial DGPK writeVectorFunctionXdmf not implemented yet",level=1)

class DG_AffineP0_OnSimplexWithMonomialBasis(DG_AffinePolynomialsOnSimplexWithMonomialBasis):
    def __init__(self,mesh,nd=3):
        DG_AffinePolynomialsOnSimplexWithMonomialBasis.__init__(self,mesh,nd,0)
    def writeMeshEnsight(self,filename,description=None):
        self.mesh.writeMeshEnsight(filename,description)
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'_average.vec****\n')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+' '+filename+u.name+'_average.scl****\n')
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'_average.vec****\n')
                caseOut.write('vector per node: '+
                        u.name+' '+filename+u.name+'_jump_max.vec****\n')
            AverageOut=open(filename+u.name+'_average.vec%4.4i' % self.nOutput,'w')
            JumpOut=open(filename+u.name+'_jump_max.vec%4.4i' % self.nOutput,'w')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+"_average"+' '+filename+u.name+'_average.scl****\n')
                    caseOut.write('scalar per node: '+
                                  u.name+"_jump"+' '+filename+u.name+'_jump_max.scl****\n')

            AverageOut=open(filename+u.name+'_average.scl%4.4i' % self.nOutput,'w')
            JumpOut=open(filename+u.name+'_jump_max.scl%4.4i' % self.nOutput,'w')
        AverageOut.write(u.name+'\n')
        AverageOut.write('part\n'+'%10i\n' % 1)
        AverageOut.write('coordinates\n')
        JumpOut.write(u.name+'\n')
        JumpOut.write('part\n'+'%10i\n' % 1)
        JumpOut.write('coordinates\n')
        nodal_average = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_min= numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_jump_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_nDOF = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                   'd')
        for t in u.range_dim_dof:
            nodal_average[:]=0.0
            nodal_max[:]=0.0
            nodal_min[:]=0.0
            nodal_jump_max[:]=0.0
            nodal_nDOF[:]=0.0
            for eN in range(self.elementMaps.mesh.nElements_global):
                for i in range(self.elementMaps.mesh.nNodes_element):
                    n = self.elementMaps.mesh.elementNodesArray[eN,i]
                    u_node = u.dof[eN]
                    nodal_average[n] += u_node
                    nodal_nDOF[n]+=1
                    if nodal_nDOF[n] >=2:
                        nodal_max[n] = max(nodal_max[n],u_node)
                        nodal_min[n] = min(nodal_min[n],u_node)
                        nodal_jump_max[n] = nodal_max[n]-nodal_min[n]
            for n in range(self.elementMaps.mesh.nNodes_global):
                nodal_average[n] /= nodal_nDOF[n]
                AverageOut.write('%12.5e\n' % nodal_average[n])
                JumpOut.write('%12.5e\n' % nodal_jump_max[n])
        if u.isVector:
            for t in range(3 - u.dim_dof):
                for n in range(self.elementMaps.mesh.nNodes_global):
                    AverageOut.write('%12.5e\n' % 0.0)
                    JumpOut.write('%12.5e\n' % 0.0)
        AverageOut.close()
        JumpOut.close()
        self.nOutput+=1
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_DGP0(ar,self.elementMaps.mesh,
                                                  self.nSpace_global,
                                                  t=t,init=init,meshChanged=meshChanged,
                                                  arGrid=arGrid,tCount=tCount)

    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        return self.XdmfWriter.writeFunctionXdmf_DGP0(ar,u,tCount=tCount,init=init)

    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        return self.XdmfWriter.writeVectorFunctionXdmf_DGP0(ar,uList,components,vectorName,tCount=tCount,init=init)

    def getValuesAtMeshNodes(self,dof,nodalValues,isVector,dim_dof):
        """
        """
        cfemIntegrals.computeC0P1InterpolantDGP0(self.elementMaps.mesh.elementNodesArray,
                                                 self.elementMaps.mesh.nodeElementOffsets,
                                                 self.elementMaps.mesh.nodeElementsArray,
                                                 self.dofMap.l2g,
                                                 dof,
                                                 nodalValues,
                                                 dim_dof)

class DG_AffineP1_OnSimplexWithMonomialBasis(DG_AffinePolynomialsOnSimplexWithMonomialBasis):
    def __init__(self,mesh,nd=3):
        DG_AffinePolynomialsOnSimplexWithMonomialBasis.__init__(self,mesh,nd,1)
class DG_AffineP2_OnSimplexWithMonomialBasis(DG_AffinePolynomialsOnSimplexWithMonomialBasis):
    def __init__(self,mesh,nd=3):
        DG_AffinePolynomialsOnSimplexWithMonomialBasis.__init__(self,mesh,nd,2)
class DG_AffineP3_OnSimplexWithMonomialBasis(DG_AffinePolynomialsOnSimplexWithMonomialBasis):
    def __init__(self,mesh,nd=3):
        DG_AffinePolynomialsOnSimplexWithMonomialBasis.__init__(self,mesh,nd,3)
class DG_AffineP4_OnSimplexWithMonomialBasis(DG_AffinePolynomialsOnSimplexWithMonomialBasis):
    def __init__(self,mesh,nd=3):
        DG_AffinePolynomialsOnSimplexWithMonomialBasis.__init__(self,mesh,nd,4)
class DG_AffineP5_OnSimplexWithMonomialBasis(DG_AffinePolynomialsOnSimplexWithMonomialBasis):
    def __init__(self,mesh,nd=3):
        DG_AffinePolynomialsOnSimplexWithMonomialBasis.__init__(self,mesh,nd,5)
class DG_AffineP6_OnSimplexWithMonomialBasis(DG_AffinePolynomialsOnSimplexWithMonomialBasis):
    def __init__(self,mesh,nd=3):
        DG_AffinePolynomialsOnSimplexWithMonomialBasis.__init__(self,mesh,nd,6)

class DG_AffineLinearOnSimplexWithNodalBasis(ParametricFiniteElementSpace):
    """
    A linear DG space with the nodal basis.

    Globally piecewise continuous.
    Each geometric element is the image of the reference simplex under
    a piecewise linear, continuous, affine mapping. The nodal basis is used on the reference simplex.
    mwf/cek
    """

    def __init__(self,mesh,nd=3):
        localFunctionSpace = LinearOnSimplexWithNodalBasis(nd)
        interpolationConditions = NodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              DiscontinuousGalerkinDOFMap(mesh,localFunctionSpace))
        self.strongDirichletConditions = False
        #for archiving
        from . import Archiver
        self.XdmfWriter = Archiver.XdmfWriter()

    def writeFunctionGnuplot(self,u,filename):
        import Gnuplot
        nodal_average = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_min= numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_jump_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_nDOF = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                   'd')
        if self.viewer is None:
            self.viewer = Gnuplot.Gnuplot()
            #self.viewer("set terminal x11")
        for t in u.range_dim_dof:
            nodal_average[:]=0.0
            nodal_max[:]=0.0
            nodal_min[:]=0.0
            nodal_jump_max[:]=0.0
            nodal_nDOF[:]=0.0
            for eN in range(self.elementMaps.mesh.nElements_global):
                for i in range(self.referenceFiniteElement.localFunctionSpace.dim):
                    I = self.dofMap.l2g[eN,i]*u.dim_dof+t
                    n = self.elementMaps.mesh.elementNodesArray[eN,i]
                    u_node = u.dof[I]
                    nodal_average[n] += u_node
                    nodal_nDOF[n]+=1
                    if nodal_nDOF[n] >=2:
                        nodal_max[n] = max(nodal_max[n],u_node)
                        nodal_min[n] = min(nodal_min[n],u_node)
                        nodal_jump_max[n] = nodal_max[n]-nodal_min[n]
            for n in range(self.elementMaps.mesh.nNodes_global):
                nodal_average[n] /= nodal_nDOF[n]
            if self.referenceFiniteElement.referenceElement.dim == 1:
                self.viewer.plot(Gnuplot.Data(self.elementMaps.mesh.nodeArray[:,0],
                                        nodal_average,

                                        title=u.name))
            elif self.referenceFiniteElement.referenceElement.dim == 2:
                nx = sqrt(self.elementMaps.mesh.nNodes_global)
                ny = nx
                x = numpy.arange(nx,dtype='i')/float(nx-1)
                y = numpy.arange(nx,dtype='i')/float(nx-1)
                nSol = numpy.reshape(nodal_average,(nx,ny))
                self.viewer('set parametric')
                self.viewer('set data style lines')
                self.viewer('set hidden')
                self.viewer('set contour base')
                self.viewer.xlabel('x')
                self.viewer.ylabel('y')
                self.viewer.splot(Gnuplot.GridData(nSol,
                                                   x,
                                                   y,
                                                   binary=0,
                                                   inline=0,
                                                   title=filename))
    def writeMeshEnsight(self,filename,description=None):
        self.mesh.writeMeshEnsight(filename,description)
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'_average.vec****\n')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+' '+filename+u.name+'_average.scl****\n')
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'_average.vec****\n')
                caseOut.write('vector per node: '+
                        u.name+' '+filename+u.name+'_jump_max.vec****\n')
            AverageOut=open(filename+u.name+'_average.vec%4.4i' % self.nOutput,'w')
            JumpOut=open(filename+u.name+'_jump_max.vec%4.4i' % self.nOutput,'w')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+"_average"+' '+filename+u.name+'_average.scl****\n')
                    caseOut.write('scalar per node: '+
                                  u.name+"_jump"+' '+filename+u.name+'_jump_max.scl****\n')

            AverageOut=open(filename+u.name+'_average.scl%4.4i' % self.nOutput,'w')
            JumpOut=open(filename+u.name+'_jump_max.scl%4.4i' % self.nOutput,'w')
        AverageOut.write(u.name+'\n')
        AverageOut.write('part\n'+'%10i\n' % 1)
        AverageOut.write('coordinates\n')
        JumpOut.write(u.name+'\n')
        JumpOut.write('part\n'+'%10i\n' % 1)
        JumpOut.write('coordinates\n')
        nodal_average = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_min= numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_jump_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_nDOF = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                   'd')
        for t in u.range_dim_dof:
            nodal_average[:]=0.0
            nodal_max[:]=0.0
            nodal_min[:]=0.0
            nodal_jump_max[:]=0.0
            nodal_nDOF[:]=0.0
            for eN in range(self.elementMaps.mesh.nElements_global):
                for i in range(self.referenceFiniteElement.localFunctionSpace.dim):
                    I = self.dofMap.l2g[eN,i]*u.dim_dof+t
                    n = self.elementMaps.mesh.elementNodesArray[eN,i]
                    u_node = u.dof[I]
                    nodal_average[n] += u_node
                    nodal_nDOF[n]+=1
                    if nodal_nDOF[n] >=2:
                        nodal_max[n] = max(nodal_max[n],u_node)
                        nodal_min[n] = min(nodal_min[n],u_node)
                        nodal_jump_max[n] = nodal_max[n]-nodal_min[n]
            for n in range(self.elementMaps.mesh.nNodes_global):
                nodal_average[n] /= nodal_nDOF[n]
                AverageOut.write('%12.5e\n' % nodal_average[n])
                JumpOut.write('%12.5e\n' % nodal_jump_max[n])
        if u.isVector:
            for t in range(3 - u.dim_dof):
                for n in range(self.elementMaps.mesh.nNodes_global):
                    AverageOut.write('%12.5e\n' % 0.0)
                    JumpOut.write('%12.5e\n' % 0.0)
        AverageOut.close()
        JumpOut.close()
        self.nOutput+=1
    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_LagrangeDGP1(fout,
                                                  u.femSpace.nSpace_global,
                                                  u.femSpace.elementMaps.mesh.nodeArray,
                                                  u.femSpace.elementMaps.mesh.elementNodesArray,
                                                  u.femSpace.dofMap.l2g,
                                                  u.dof,
                                                  name=u.name,
                                                  storeMeshData=storeMeshData,
                                                  figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_DGP1Lagrange(ar,name,self.mesh,self.nSpace_global,self.dofMap,
                                                          self.mesh.elementNodesArray,
                                                          t=t,init=init,meshChanged=meshChanged,
                                                          arGrid=arGrid,tCount=tCount)
    #def
    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        self.XdmfWriter.writeFunctionXdmf_DGP1Lagrange(ar,u,tCount=tCount,init=init, dofMap = self.dofMap)
    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        self.XdmfWriter.writeVectorFunctionXdmf_nodal(ar,uList,components,vectorName,"dgp1_Lagrange",tCount=tCount,init=init, dofMap=self.dofMap)

    def getValuesAtMeshNodes(self,dof,nodalValues,isVector,dim_dof):
        """
        Calculate function at mesh nodes from degrees of freedom defined elsewhere
        """
        cfemIntegrals.computeC0P1InterpolantDGP12(self.elementMaps.mesh.elementNodesArray,
                                                  self.elementMaps.mesh.nodeElementOffsets,
                                                  self.elementMaps.mesh.nodeElementsArray,
                                                  self.dofMap.l2g,
                                                  dof,
                                                  nodalValues,
                                                  dim_dof)
        
class C0_AffineQuadraticOnSimplexWithNodalBasis(ParametricFiniteElementSpace):
    """
    A quadratic C0 space with the nodal basis.

    Globally piecewise continuous.
    Each geometric element is the image of the reference simplex under
    a piecewise linear, continuous, affine mapping.
    The nodal basis is used on the reference simplex.
    """
    def __init__(self,mesh,nd=3):
        self.order = 2
        localFunctionSpace = QuadraticOnSimplexWithNodalBasis(nd)
        localGeometricSpace= LinearOnSimplexWithNodalBasis(nd)
        interpolationConditions = QuadraticLagrangeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localGeometricSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              QuadraticLagrangeDOFMap(mesh,localFunctionSpace,nd))

        #for archiving
        from . import Archiver
        self.XdmfWriter=Archiver.XdmfWriter()
    def writeFunctionGnuplot(self,u,filename):
        """
        for now, just print out nodal dofs for vertices
        """

        import Gnuplot
        nodal_average = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_min = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_jump_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_nDOF = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                   'd')
        if self.viewer is None:
            self.viewer = Gnuplot.Gnuplot()
            #self.viewer("set terminal x11")
        #mwf for now just loop over vertices
        dim4plot = self.referenceFiniteElement.referenceElement.dim+1

        for t in u.range_dim_dof:
            nodal_average[:]=0.0
            nodal_max[:]=0.0
            nodal_min[:]=0.0
            nodal_jump_max[:]=0.0
            nodal_nDOF[:]=0.0
            for eN in range(self.elementMaps.mesh.nElements_global):
                #mwf for now just loop over vertices
                #for i in range(self.referenceFiniteElement.localFunctionSpace.dim):
                for i in range(dim4plot):
                    I = self.dofMap.l2g[eN,i]*u.dim_dof+t
                    n = self.elementMaps.mesh.elementNodesArray[eN,i]
                    u_node = u.dof[I]
                    nodal_average[n] += u_node
                    nodal_nDOF[n]+=1
                    if nodal_nDOF[n] >=2:
                        nodal_max[n] = max(nodal_max[n],u_node)
                        nodal_min[n] = min(nodal_min[n],u_node)
                        nodal_jump_max[n] = nodal_max[n]-nodal_min[n]
            for n in range(self.elementMaps.mesh.nNodes_global):
                nodal_average[n] /= nodal_nDOF[n]
            if self.referenceFiniteElement.referenceElement.dim == 1:
                self.viewer.plot(Gnuplot.Data(self.elementMaps.mesh.nodeArray[:,0],
                                        nodal_average,

                                        title=filename))
            elif self.referenceFiniteElement.referenceElement.dim == 2:
                nx = sqrt(self.elementMaps.mesh.nNodes_global)
                ny = nx
                x = numpy.arange(nx,dtype='i')/float(nx-1)
                y = numpy.arange(nx,dtype='i')/float(nx-1)
                nSol = numpy.reshape(nodal_average,(nx,ny))
                self.viewer('set parametric')
                self.viewer('set data style lines')
                self.viewer('set hidden')
                self.viewer('set contour base')
                self.viewer.xlabel('x')
                self.viewer.ylabel('y')
                self.viewer.splot(Gnuplot.GridData(nSol,
                                                   x,
                                                   y,
                                                   binary=0,
                                                   inline=0,
                                                   title=filename))
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_C0P2Lagrange(ar,name,mesh=self.mesh,spaceDim=self.nSpace_global,
                                                          dofMap=self.dofMap,t=t,init=init,meshChanged=meshChanged,
                                                          arGrid=arGrid,tCount=tCount)
    #def
    def writeMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','w')
        caseOut.write('FORMAT\n'+'type: ensight gold\n')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        if self.referenceFiniteElement.referenceElement.dim == 1:
            meshOut.write('Unstructured 3 Node Bar Mesh\n\n')
            meshOut.write('node id given\n')
            meshOut.write('element id given\n')
            meshOut.write('part \n'+'%10i\n' % 1)
            if description:
                meshOut.write(description+'\n')
            else:
                meshOut.write('A Mesh\n')
            meshOut.write('coordinates\n'+'%10i\n' % self.dim)
            mesh = self.elementMaps.mesh
            lagrangeNodesArray = self.dofMap.lagrangeNodesArray
            #mwf fix lagrange nodes format
            #node numbers
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%10i\n' % (i + base))
            #x-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,0])
            #y-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,1])
            #z-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,2])

#             #node numbers
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%10i\n' % (nN + base))
#             for eN in range(mesh.nElements_global):
#                 meshOut.write('%10i\n' % (mesh.nNodes_global + eN + base))
#             #x-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,0])
#             for eN in range(mesh.nElements_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[eN,0])
#             #y-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,1])
#             for eN in range(mesh.nElements_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[eN,1])
#             #z-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,2])
#             for eN in range(mesh.nElements_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[eN,2])
            meshOut.write('bar3\n'+'%10i\n' % mesh.nElements_global)
            for eN in range(mesh.nElements_global):
                meshOut.write('%10i\n' % (eN+base))
            for eN in range(mesh.nElements_global):
                meshOut.write('%10i%10i%10i\n' % tuple(base+self.dofMap.l2g[eN,j] for j in self.referenceFiniteElement.localFunctionSpace.range_dim))
        elif self.referenceFiniteElement.referenceElement.dim == 2:
            meshOut.write('Unstructured 6 Node Triangular Mesh\n\n')
            meshOut.write('node id given\n')
            meshOut.write('element id given\n')
            meshOut.write('part \n'+'%10i\n' % 1)
            if description:
                meshOut.write(description+'\n')
            else:
                meshOut.write('A Mesh\n')
            meshOut.write('coordinates\n'+'%10i\n' % self.dim)
            mesh = self.elementMaps.mesh
            lagrangeNodesArray = self.dofMap.lagrangeNodesArray
            #mwf fix lagrange nodes format
            #node numbers
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%10i\n' % (i + base))
            #x-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,0])
            #y-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,1])
            #z-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,2])
#             #node numbers
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%10i\n' % (nN + base))
#             for ebN in range(mesh.nElementBoundaries_global):
#                 meshOut.write('%10i\n' % (mesh.nNodes_global + ebN + base))
#             #x-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,0])
#             for ebN in range(mesh.nElementBoundaries_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[ebN,0])
#             #y-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,1])
#             for ebN in range(mesh.nElementBoundaries_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[ebN,1])
#             #z-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,2])
#             for ebN in range(mesh.nElementBoundaries_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[ebN,2])
            meshOut.write('tria6\n'+'%10i\n' % mesh.nElements_global)
            for eN in range(mesh.nElements_global):
                meshOut.write('%10i\n' % (eN+base))
            for eN in range(mesh.nElements_global):
                meshOut.write('%10i%10i%10i%10i%10i%10i\n' % tuple(base+self.dofMap.l2g[eN,j] for j in self.referenceFiniteElement.localFunctionSpace.range_dim))
        elif self.referenceFiniteElement.referenceElement.dim == 3:
            meshOut.write('Unstructured 10 Node Tetrahedral Mesh\n\n')
            meshOut.write('node id given\n')
            meshOut.write('element id given\n')
            meshOut.write('part \n'+'%10i\n' % 1)
            if description:
                meshOut.write(description+'\n')
            else:
                meshOut.write('A Mesh\n')
            meshOut.write('coordinates\n'+'%10i\n' % self.dim)
            mesh = self.elementMaps.mesh
            lagrangeNodesArray = self.dofMap.lagrangeNodesArray
            #mwf fix lagrange nodes format
            #node numbers
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%10i\n' % (i + base))
            #x-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,0])
            #y-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,1])
            #z-coordinates
            for i in range(lagrangeNodesArray.shape[0]):
                meshOut.write('%12.5E\n' % lagrangeNodesArray[i,2])
#             #node numbers
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%10i\n' % (nN + base))
#             for eN in range(mesh.nEdges_global):
#                 meshOut.write('%10i\n' % (mesh.nNodes_global + eN + base))
#             #x-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,0])
#             for eN in range(mesh.nEdges_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[eN,0])
#             #y-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,1])
#             for eN in range(mesh.nEdges_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[eN,1])
#             #z-coordinates
#             for nN in range(mesh.nNodes_global):
#                 meshOut.write('%12.5E\n' % mesh.nodeArray[nN,2])
#             for eN in range(mesh.nEdges_global):
#                 meshOut.write('%12.5E\n' % lagrangeNodesArray[eN,2])
            meshOut.write('tetra10\n'+'%10i\n' % mesh.nElements_global)
            #mwf todo fix for new parallel numbering
            for eN in range(mesh.nElements_global):
                meshOut.write('%10i\n' % (eN+base))
            for eN in range(mesh.nElements_global):
                #
                nodes=dict([(j,base+self.dofMap.l2g[eN,j]) for j in range(4)])
                #brute force way to make sure the remaining nodes are in the right order to describe the element
                for edgeN_element in range(6):
                    edgeN = self.dofMap.l2g[eN,4+edgeN_element] - mesh.nNodes_global
                    n0 = mesh.edgeNodesArray[edgeN,0]
                    n1 = mesh.edgeNodesArray[edgeN,1]
                    if n0 == self.dofMap.l2g[eN,0]:
                        if n1 == self.dofMap.l2g[eN,1]:
                            nodes[4+0]=base+self.dofMap.l2g[eN,edgeN_element+4]
                        if n1 == self.dofMap.l2g[eN,2]:
                            nodes[4+2]=base+self.dofMap.l2g[eN,edgeN_element+4]
                        if n1 == self.dofMap.l2g[eN,3]:
                            nodes[4+3]=base+self.dofMap.l2g[eN,edgeN_element+4]
                    elif n1 == self.dofMap.l2g[eN,0]:
                        if n0 == self.dofMap.l2g[eN,1]:
                            nodes[4+0]=base+self.dofMap.l2g[eN,edgeN_element+4]
                        if n0 == self.dofMap.l2g[eN,2]:
                            nodes[4+2]=base+self.dofMap.l2g[eN,edgeN_element+4]
                        if n0 == self.dofMap.l2g[eN,3]:
                            nodes[4+3]=base+self.dofMap.l2g[eN,edgeN_element+4]
                    elif n0 == self.dofMap.l2g[eN,1]:
                        if n1 == self.dofMap.l2g[eN,2]:
                            nodes[4+1]=base+self.dofMap.l2g[eN,edgeN_element+4]
                        if n1 == self.dofMap.l2g[eN,3]:
                            nodes[4+4]=base+self.dofMap.l2g[eN,edgeN_element+4]
                    elif n1 == self.dofMap.l2g[eN,1]:
                        if n0 == self.dofMap.l2g[eN,2]:
                            nodes[4+1]=base+self.dofMap.l2g[eN,edgeN_element+4]
                        if n0 == self.dofMap.l2g[eN,3]:
                            nodes[4+4]=base+self.dofMap.l2g[eN,edgeN_element+4]
                    elif n0 == self.dofMap.l2g[eN,2]:
                        if n1 == self.dofMap.l2g[eN,3]:
                            nodes[4+5]=base+self.dofMap.l2g[eN,edgeN_element+4]
                    elif n1 == self.dofMap.l2g[eN,2]:
                        if n0 == self.dofMap.l2g[eN,3]:
                            nodes[4+5]=base+self.dofMap.l2g[eN,edgeN_element+4]
                    else:
                        print("fell through",n0,n1)
                meshOut.write('%10i%10i%10i%10i%10i%10i%10i%10i%10i%10i\n' % tuple(nodes.values()))
        meshOut.close()
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'.vec****\n')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+' '+filename+u.name+'.scl****\n')
    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        self.XdmfWriter.writeFunctionXdmf_C0P2Lagrange(ar,u,tCount=tCount,init=init)
    def readFunctionXdmf(self,ar,u,tCount=0):
        if ar.global_sync:
            permute = np.argsort(u.femSpace.dofMap.subdomain2global)
            u.dof[permute] = ar.hdfFile["/"+u.name+"_t{0:d}".format(tCount)][u.femSpace.dofMap.subdomain2global[permute].tolist()]
        else:
            u.dof[:]=ar.hdfFile["/"+u.name+"_p"+repr(ar.comm.rank())+"_t{0:d}".format(tCount)].value

    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        self.XdmfWriter.writeVectorFunctionXdmf_nodal(ar,uList,components,vectorName,"c0p2_Lagrange",tCount=tCount,init=init)

    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        """
        For now only works for triangles
        """
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'.vec****\n')
                #caseOut.write('VARIABLE\n'+'vector per node: '+
                #        u.name+' '+filename+u.name+'_average.vec\n')
                #caseOut.write('VARIABLE\n'+'vector per node: '+
                #        u.name+' '+filename+u.name+'_jump_max.vec\n')
            #AverageOut=open(filename+u.name+'_average.vec'+`self.nOutput`,'w')
            #JumpOut=open(filename+u.name+'_jump_max.vec'+`self.nOutput`,'w')
            uOut=open(filename+u.name+'.vec%4.4i' % self.nOutput,'w')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+' '+filename+u.name+'.scl****\n')
                    #caseOut.write('scalar per node: '+
                    #              u.name+"_average_"+`t`+' '+filename+u.name+'_average.scl'+`self.nOutput`+'\n')
                    #caseOut.write('scalar per node: '+
                    #              u.name+"_jump_"+`t`+' '+filename+u.name+'_jump_max.scl'+`self.nOutput`+'\n')
                    #
            #AverageOut=open(filename+u.name+'_average.scl'+`self.nOutput`,'w')
            #JumpOut=open(filename+u.name+'_jump_max.scl'+`self.nOutput`,'w')
            uOut=open(filename+u.name+'.scl%4.4i' % self.nOutput,'w')
        #AverageOut.write(u.name+'\n')
        #AverageOut.write('part\n'+'%10i\n' % 1)
        #AverageOut.write('coordinates\n')
        #JumpOut.write(u.name+'\n')
        #JumpOut.write('part\n'+'%10i\n' % 1)
        #JumpOut.write('coordinates\n')
        #nodal_average = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
        #                              'd')
        #nodal_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
        #                              'd')
        #nodal_min= numpy.zeros((self.elementMaps.mesh.nNodes_global,),
        #                              'd')
        #nodal_jump_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
        #                              'd')
        #nodal_nDOF = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
        #                           'd')
        #for t in u.range_dim_dof:
        #    nodal_average[:]=0.0
        #    nodal_max[:]=0.0
        #    nodal_min[:]=0.0
        #    nodal_jump_max[:]=0.0
        #    nodal_nDOF[:]=0.0
        #    for eN in range(self.elementMaps.mesh.nElements_global):
        #        #mwf for now just loop over vertices
        #        #for i in range(self.referenceFiniteElement.localFunctionSpace.dim):
        #        for i in range(self.referenceFiniteElement.referenceElement.dim+1):
        #            I = self.dofMap.l2g[eN,i]*u.dim_dof+t
        #            n = self.elementMaps.mesh.elementNodesArray[eN,i]
        #            u_node = u.dof[I]
        #            nodal_average[n] += u_node
        #            nodal_nDOF[n]+=1
        #            if nodal_nDOF[n] >=2:
        #                nodal_max[n] = max(nodal_max[n],u_node)
        #                nodal_min[n] = min(nodal_min[n],u_node)
        #                nodal_jump_max[n] = nodal_max[n]-nodal_min[n]
        #    for n in range(self.elementMaps.mesh.nNodes_global):
        #        nodal_average[n] /= nodal_nDOF[n]
        #        AverageOut.write('%12.5e\n' % nodal_average[n])
        #        JumpOut.write('%12.5e\n' % nodal_jump_max[n])
        #if u.isVector:
        #    for t in range(3 - u.dim_dof):
        #        for n in range(self.elementMaps.mesh.nNodes_global):
        #            AverageOut.write('%12.5e\n' % 0.0)
        #            JumpOut.write('%12.5e\n' % 0.0)
        #AverageOut.close()
        #JumpOut.close()
        uOut.write(u.name+'\n')
        uOut.write('part\n'+'%10i\n' % 1)
        uOut.write('coordinates\n')
        for t in u.range_dim_dof:
            cfemIntegrals.writeDOF(u.femSpace.dim,u.dim_dof,t,'%12.5e\n',u.dof,uOut)
            #for n in u.femSpace.range_dim:
            #    uOut.write('%12.5e\n' % u.dof[n*u.dim_dof + t])
        if u.isVector:
            for t in range(3 - u.dim_dof):
                for n in u.femSpace.range_dim:
                    uOut.write('%12.5e\n' % 0.0)
        uOut.close()
        self.nOutput+=1
    def writeE2VectorFunctionEnsight(self,u,v,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        """
        old ensight printing
        """
        if case_filename is None:
            case_filename = filename
        if not append:
            caseOut=open(case_filename+'.case','a')
            if firstVariable==True:
                caseOut.write('VARIABLE\n')
            caseOut.write('vector per node: '+
                          u.name+'2 '+filename+u.name+'2.vec****\n')
        uOut=open(filename+u.name+'2.vec%4.4i' % nOutput,'w')
        uOut.write(u.name+'\n')
        uOut.write('part\n'+'%10i\n' % 1)
        uOut.write('coordinates\n')
        for t in u.range_dim_dof:
            cfemIntegrals.writeDOF(u.femSpace.dim,u.dim_dof,t,'%12.5e\n',u.dof,uOut)
            cfemIntegrals.writeDOF(v.femSpace.dim,v.dim_dof,t,'%12.5e\n',v.dof,uOut)
            cfemIntegrals.writeDOF_ZEROS(v.femSpace.dim,v.dim_dof,t,'%12.5e\n',uOut)
        uOut.close()
    def writeE2VectorFunctionHeaderEnsight(self,u,v,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if not append:
            caseOut=open(case_filename+'.case','a')
            if firstVariable==True:
                caseOut.write('VARIABLE\n')
            caseOut.write('vector per node: '+
                          u.name+'2 '+filename+u.name+'2.vec****\n')
        caseOut.close()
    def writeE3VectorFunctionEnsight(self,u,v,w,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if not append:
            caseOut=open(case_filename+'.case','a')
            if firstVariable==True:
                caseOut.write('VARIABLE\n')
            caseOut.write('vector per node: '+
                          u.name+'2 '+filename+u.name+'2.vec****\n')
        uOut=open(filename+u.name+'2.vec%4.4i' % nOutput,'w')
        uOut.write(u.name+'\n')
        uOut.write('part\n'+'%10i\n' % 1)
        uOut.write('coordinates\n')
        for t in u.range_dim_dof:
            cfemIntegrals.writeDOF(u.femSpace.dim,u.dim_dof,t,'%12.5e\n',u.dof,uOut)
            cfemIntegrals.writeDOF(v.femSpace.dim,v.dim_dof,t,'%12.5e\n',v.dof,uOut)
            cfemIntegrals.writeDOF(w.femSpace.dim,w.dim_dof,t,'%12.5e\n',w.dof,uOut)
        uOut.close()
    def writeE3VectorFunctionHeaderEnsight(self,u,v,w,filename,nOutput,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if not append:
            caseOut=open(case_filename+'.case','a')
            if firstVariable==True:
                caseOut.write('VARIABLE\n')
            caseOut.write('vector per node: '+
                          u.name+'2 '+filename+u.name+'2.vec****\n')
        caseOut.close()
    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_LagrangeC0P2(fout,
                                                  u.femSpace.nSpace_global,
                                                  u.femSpace.dofMap.lagrangeNodesArray,u.femSpace.dofMap.l2g,
                                                  u.femSpace.elementMaps.mesh.elementNodesArray,
                                                  u.dof,
                                                  name=u.name,
                                                  storeMeshData=storeMeshData,
                                                  figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0

P2 = C0_AffineQuadraticOnSimplexWithNodalBasis

class C0_AffineBernsteinOnSimplex(C0_AffineQuadraticOnSimplexWithNodalBasis):
    """
    A quadratic C0 space with Bernstein basis.

    Globally piecewise continuous.
    Each geometric element is the image of the reference simplex under
    a piecewise linear, continuous, affine mapping.
    The nodal basis is used on the reference simplex.
    """    
    def __init__(self,mesh,nd=3):
        self.order = 2
        localFunctionSpace = BernsteinOnSimplex(nd)
        localGeometricSpace= LinearOnSimplexWithNodalBasis(nd)
        # NOTE: Bernstein polynomials ARE NOT INTERPOLATORY but we use them as control points to create a bezier curve.
        interpolationConditions = QuadraticLagrangeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localGeometricSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              QuadraticLagrangeDOFMap(mesh,localFunctionSpace,nd))

        #for archiving
        from . import Archiver
        self.XdmfWriter=Archiver.XdmfWriter()

class DG_AffineQuadraticOnSimplexWithNodalBasis(ParametricFiniteElementSpace):
    """
    A quadratic DG space with the nodal basis.

    Globally piecewise continuous.  Each geometric element is the
    image of the reference simplex under a piecewise linear,
    continuous, affine mapping. The nodal basis is used on the
    reference simplex.
    """
    def __init__(self,mesh,nd=3):
        localFunctionSpace = QuadraticOnSimplexWithNodalBasis(nd)
        localGeometricSpace= LinearOnSimplexWithNodalBasis(nd)
        interpolationConditions = QuadraticLagrangeNodalInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localGeometricSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              DiscontinuousGalerkinDOFMap(mesh,localFunctionSpace))
        self.strongDirichletConditions = False
        self.CGDOFMap = QuadraticLagrangeDOFMap(mesh,localFunctionSpace,nd)
        #for archiving
        from . import Archiver
        self.XdmfWriter = Archiver.XdmfWriter()
    def writeMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','w')
        caseOut.write('FORMAT\n'+'type: ensight gold\n')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        if self.nSpace_global == 2:
            meshOut.write('Unstructured 6 Node Triangular Mesh\n\n')
        elif self.nSpace_global == 3:
            meshOut.write('Unstructured 10 Node Tetrahedral Mesh\n\n')
        elif self.nSpace_global == 1:
            meshOut.write('Unstructured 3 Node edge Mesh\n\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write('A Mesh\n')
        meshOut.write('coordinates\n'+'%10i\n' % self.CGDOFMap.nDOF)
        mesh = self.elementMaps.mesh
        lagrangeNodesArray = self.CGDOFMap.lagrangeNodesArray
        #mwf fix lagrange nodes format
        #node numbers
        for i in range(lagrangeNodesArray.shape[0]):
            meshOut.write('%10i\n' % (i + base))
        #x-coordinates
        for i in range(lagrangeNodesArray.shape[0]):
            meshOut.write('%12.5E\n' % lagrangeNodesArray[i,0])
        #y-coordinates
        for i in range(lagrangeNodesArray.shape[0]):
            meshOut.write('%12.5E\n' % lagrangeNodesArray[i,1])
        #z-coordinates
        for i in range(lagrangeNodesArray.shape[0]):
            meshOut.write('%12.5E\n' % lagrangeNodesArray[i,2])
#       #node numbers
#         for nN in range(mesh.nNodes_global):
#             meshOut.write('%10i\n' % (nN + base))
#       for ebN in range(mesh.nElementBoundaries_global):
#           meshOut.write('%10i\n' % (mesh.nNodes_global + ebN + base))
#       #x-coordinates
#         for nN in range(mesh.nNodes_global):
#             meshOut.write('%12.5E\n' % mesh.nodeArray[nN,0])
#       for ebN in range(mesh.nElementBoundaries_global):
#             meshOut.write('%12.5E\n' % lagrangeNodesArray[ebN,0])
#       #y-coordinates
#       for nN in range(mesh.nNodes_global):
#             meshOut.write('%12.5E\n' % mesh.nodeArray[nN,1])
#       for ebN in range(mesh.nElementBoundaries_global):
#             meshOut.write('%12.5E\n' % lagrangeNodesArray[ebN,1])
#       #z-coordinates
#         for nN in range(mesh.nNodes_global):
#             meshOut.write('%12.5E\n' % mesh.nodeArray[nN,2])
#       for ebN in range(mesh.nElementBoundaries_global):
#             meshOut.write('%12.5E\n' % lagrangeNodesArray[ebN,2])
        #
        if self.nSpace_global == 2:
            meshOut.write('tria6\n'+'%10i\n' % mesh.nElements_global)
        elif self.nSpace_global == 3:
            meshOut.write('tetra10\n'+'%10i\n' % mesh.nElements_global)
        elif self.nSpace_global == 1:
            meshOut.write('bar3\n'+'%10i\n' % mesh.nElements_global)
        for eN in range(mesh.nElements_global):
            meshOut.write('%10i\n' % (eN+base))
        elemString = '%10i'
        for j in range(self.referenceFiniteElement.localFunctionSpace.dim-1):
            elemString += '%10i'
        elemString += '\n'
        for eN in range(mesh.nElements_global):
            #meshOut.write('%10i%10i%10i%10i%10i%10i\n' % tuple(base+self.CGDOFMap.l2g[eN,j] for j in self.referenceFiniteElement.localFunctionSpace.range_dim))
            meshOut.write(elemString % tuple(base+self.CGDOFMap.l2g[eN,j] for j in self.referenceFiniteElement.localFunctionSpace.range_dim))
        meshOut.close()

    def writeFunctionGnuplot(self,u,filename):
        import Gnuplot
        nodal_average = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_min= numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_jump_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_nDOF = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                   'd')
        if self.viewer is None:
            self.viewer = Gnuplot.Gnuplot()
            #mwf added terminal cmd
            self.viewer("set terminal x11")
        for t in u.range_dim_dof:
            nodal_average[:]=0.0
            nodal_max[:]=0.0
            nodal_min[:]=0.0
            nodal_jump_max[:]=0.0
            nodal_nDOF[:]=0.0
            for eN in range(self.elementMaps.mesh.nElements_global):
                #for i in range(self.referenceFiniteElement.localFunctionSpace.dim):
                for i in range(self.referenceFiniteElement.referenceElement.dim+1):
                    I = self.dofMap.l2g[eN,i]*u.dim_dof+t
                    n = self.elementMaps.mesh.elementNodesArray[eN,i]
                    u_node = u.dof[I]
                    nodal_average[n] += u_node
                    nodal_nDOF[n]+=1
                    if nodal_nDOF[n] >=2:
                        nodal_max[n] = max(nodal_max[n],u_node)
                        nodal_min[n] = min(nodal_min[n],u_node)
                        nodal_jump_max[n] = nodal_max[n]-nodal_min[n]
            for n in range(self.elementMaps.mesh.nNodes_global):
                nodal_average[n] /= nodal_nDOF[n]
            if self.referenceFiniteElement.referenceElement.dim == 1:
                self.viewer.plot(Gnuplot.Data(self.elementMaps.mesh.nodeArray[:,0],
                                        nodal_average,

                                        title=u.name))
            elif self.referenceFiniteElement.referenceElement.dim == 2:
                nx = sqrt(self.elementMaps.mesh.nNodes_global)
                ny = nx
                x = numpy.arange(nx,dtype='i')/float(nx-1)
                y = numpy.arange(nx,dtype='i')/float(nx-1)
                nSol = numpy.reshape(nodal_average,(nx,ny))
                self.viewer('set parametric')
                self.viewer('set data style lines')
                self.viewer('set hidden')
                self.viewer('set contour base')
                self.viewer.xlabel('x')
                self.viewer.ylabel('y')
                self.viewer.splot(Gnuplot.GridData(nSol,
                                             x,
                                             y,
                                             binary=0,
                                             inline=0))
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'.vec****\n')
                #caseOut.write('VARIABLE\n'+'vector per node: '+
                #        u.name+' '+filename+u.name+'_average.vec\n')
                #caseOut.write('VARIABLE\n'+'vector per node: '+
                #        u.name+' '+filename+u.name+'_jump_max.vec\n')
            #AverageOut=open(filename+u.name+'_average.vec'+`self.nOutput`,'w')
            #JumpOut=open(filename+u.name+'_jump_max.vec'+`self.nOutput`,'w')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+' '+filename+u.name+'.scl****\n')
                    #caseOut.write('scalar per node: '+
                    #              u.name+"_average_"+`t`+' '+filename+u.name+'_average.scl'+`self.nOutput`+'\n')
                    #caseOut.write('scalar per node: '+
                    #              u.name+"_jump_"+`t`+' '+filename+u.name+'_jump_max.scl'+`self.nOutput`+'\n')
                    #
            #AverageOut=open(filename+u.name+'_average.scl'+`self.nOutput`,'w')
            #JumpOut=open(filename+u.name+'_jump_max.scl'+`self.nOutput`,'w')
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        """
        For now only works for triangles
        """
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            #if not append:
            #    caseOut=open(case_filename+'.case','a')
            #    if firstVariable==True:
            #        caseOut.write('VARIABLE\n')
            #    caseOut.write('vector per node: '+
            #                  u.name+' '+filename+u.name+'.vec****\n')
                #caseOut.write('VARIABLE\n'+'vector per node: '+
                #        u.name+' '+filename+u.name+'_average.vec\n')
                #caseOut.write('VARIABLE\n'+'vector per node: '+
                #        u.name+' '+filename+u.name+'_jump_max.vec\n')
            #AverageOut=open(filename+u.name+'_average.vec'+`self.nOutput`,'w')
            #JumpOut=open(filename+u.name+'_jump_max.vec'+`self.nOutput`,'w')
            uOut=open(filename+u.name+'.vec%4.4i' % self.nOutput,'w')
        else:
            #if not append:
            #    caseOut=open(case_filename+'.case','a')
            #    caseOut.write('VARIABLE\n')
            #    for t in u.range_dim_dof:
            #        caseOut.write('scalar per node: '+
            #                      u.name+' '+filename+u.name+'.scl****\n')
                    #caseOut.write('scalar per node: '+
                    #              u.name+"_average_"+`t`+' '+filename+u.name+'_average.scl'+`self.nOutput`+'\n')
                    #caseOut.write('scalar per node: '+
                    #              u.name+"_jump_"+`t`+' '+filename+u.name+'_jump_max.scl'+`self.nOutput`+'\n')
                    #
            #AverageOut=open(filename+u.name+'_average.scl'+`self.nOutput`,'w')
            #JumpOut=open(filename+u.name+'_jump_max.scl'+`self.nOutput`,'w')
            uOut=open(filename+u.name+'.scl%4.4i' % self.nOutput,'w')
        nodal_average = numpy.zeros((self.CGDOFMap.nDOF*u.dim_dof,),'d')
        #mwf mesh.nElements_node deprecated
        nElements_node = numpy.zeros((self.mesh.nNodes_global,),'i')
        for eN in range(self.elementMaps.mesh.nElements_global):
            #first do triangle  node
            for nN_element in range(self.elementMaps.mesh.nNodes_element):
                j = nN_element
                J_cg = self.CGDOFMap.l2g[eN,j]
                nN = J_cg
                J_dg = self.dofMap.l2g[eN,j]
                #nDOF_node = float(self.elementMaps.mesh.nElements_node[nN])
                nElements_node[nN] += 1
                for t in u.range_dim_dof:
                    nodal_average[J_cg] += u.dof[J_dg]#/nDOF_node
        #now do edge nodes
        for nN in range(self.mesh.nNodes_global):
            nodal_average[nN] /= float(nElements_node[nN])
        mesh = self.elementMaps.mesh
        for ebNI in range(mesh.nInteriorElementBoundaries_global):
            ebN = mesh.interiorElementBoundariesArray[ebNI]
            left_eN_global = mesh.elementBoundaryElementsArray[ebN,0]
            right_eN_global = mesh.elementBoundaryElementsArray[ebN,1]
            left_ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            right_ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
            left_j_element = (left_ebN_element+1)%3 + mesh.nNodes_element #needs to be fixed for 1d, 3d
            right_j_element =(right_ebN_element+1)%3 + mesh.nNodes_element#needs to be fixed for 1d, 3d
            left_J_cg = self.CGDOFMap.l2g[left_eN_global,left_j_element]
            left_J_dg = self.dofMap.l2g[left_eN_global,left_j_element]
            right_J_cg = self.CGDOFMap.l2g[right_eN_global,right_j_element]
            right_J_dg = self.dofMap.l2g[right_eN_global,right_j_element]
            if left_J_cg != right_J_cg:
                print("problem in DGAffineQuadratic writeFunctionEnsight")
            for t in u.range_dim_dof:
                nodal_average[left_J_cg*u.dim_dof + t] = 0.5*(u.dof[left_J_dg*u.dim_dof + t]
                                                              +
                                                              u.dof[right_J_dg*u.dim_dof + t])

        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            eN_global = mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            j_element = (ebN_element+1)%3 + mesh.nNodes_element
            J_cg = self.CGDOFMap.l2g[eN_global,j_element]
            J_dg = self.dofMap.l2g[eN_global,j_element]
            for t in u.range_dim_dof:
                nodal_average[J_cg*u.dim_dof + t] = u.dof[J_dg*u.dim_dof + t]
#           for eN in range(mesh.nElements_global):
#              for j in range(mesh.nNodes_element,self.referenceFiniteElement.localFunctionSpace.dim):
#                  J_cg = self.CGDOFMap.l2g[eN,j]
#                  J_dg = self.dofMap.l2g[eN,j]
#                  #find element  boundary number  using Farthing's rule
#                  ebN_element  = (j-1)%3
#                  for t in u.range_dim_dof:
#                      nodal_average[J_cg*u.dim_dof + t] += u.dof[J_dg*u.dim_dof + t]*0.5
        uOut.write(u.name+'\n')
        uOut.write('part\n'+'%10i\n' % 1)
        uOut.write('coordinates\n')
        for t in u.range_dim_dof:
            cfemIntegrals.writeDOF(self.CGDOFMap.nDOF,u.dim_dof,t,'%12.5e\n',nodal_average,uOut)
            #for J in range(self.CGDOFMap.nDOF):
            #    uOut.write('%12.5e\n' % nodal_average[J*u.dim_dof + t])
        if u.isVector:
            for t in range(3 - u.dim_dof):
                for J in range(self.CGDOFMap.nDOF):
                    uOut.write('%12.5e\n' % 0.0)
        uOut.close()
        self.nOutput+=1
    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_LagrangeDGP2(fout,
                                                  u.femSpace.nSpace_global,
                                                  u.femSpace.elementMaps.mesh.nodeArray,
                                                  u.femSpace.elementMaps.mesh.elementNodesArray,
                                                  u.femSpace.CGDOFMap.lagrangeNodesArray,
                                                  u.femSpace.dofMap.l2g,
                                                  u.femSpace.CGDOFMap.l2g,
                                                  u.dof,
                                                  name=u.name,
                                                  storeMeshData=storeMeshData,
                                                  figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0

    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_DGP2Lagrange(ar,name,self.mesh,self.nSpace_global,
                                                          self.dofMap,self.CGDOFMap,t=t,
                                                          init=init,meshChanged=meshChanged,arGrid=arGrid,
                                                          tCount=tCount)
    #def
    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        self.XdmfWriter.writeFunctionXdmf_DGP2Lagrange(ar,u,tCount=tCount,init=init, dofMap=self.dofMap)
    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        self.XdmfWriter.writeVectorFunctionXdmf_nodal(ar,uList,components,vectorName,"dgp2_Lagrange",tCount=tCount,init=init)

    def getValuesAtMeshNodes(self,dof,nodalValues,isVector,dim_dof):
        """
        Calculate solution at mesh nodes from degrees of freedom defind elsewhere.
        """
        cfemIntegrals.computeC0P1InterpolantDGP12(self.elementMaps.mesh.elementNodesArray,
                                                  self.elementMaps.mesh.nodeElementOffsets,
                                                  self.elementMaps.mesh.nodeElementsArray,
                                                  self.dofMap.l2g,
                                                  dof,
                                                  nodalValues,
                                                  dim_dof)
class NC_AffineLinearOnSimplexWithNodalBasis(ParametricFiniteElementSpace):
    """
    Space of functions that are P^1 on elements and continuous at
    element boundary barycenters.

    Each geometric element is the image of the reference simplex under
    a linear affine mapping. The nodal basis is used on the reference simplex.
    """
    def __init__(self,mesh,nd=3):
        localFunctionSpace = CrouzeixRaviartWithNodalBasis(nd)
        interpolationConditions = FaceBarycenterInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              ElementBoundaryDOFMap(mesh))
        #for archiving
        from . import Archiver
        self.XdmfWriter = Archiver.XdmfWriter()
    def writeFunctionGnuplot(self,u,filename):
        import Gnuplot
        if self.referenceFiniteElement.referenceElement.dim == 1:
            self.uPlot = Gnuplot.Gnuplot()
            self.uPlot("set terminal x11")
            self.uPlot.plot(Gnuplot.Data(self.elementMaps.mesh.nodeArray[:,0],
                                    u.dof,

                                    title=u.name))
    def writeMeshEnsight(self,filename,description=None):
        self.mesh.writeMeshEnsight(filename,description)
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        #for now plotting average and jump at nodes
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'_average.vec****\n')
                caseOut.write('vector per node: '+
                        u.name+' '+filename+u.name+'_jump_max.vec****\n')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable == True:
                    caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+"_average"+' '+filename+u.name+'_average.scl****\n')
                    caseOut.write('scalar per node: '+
                                  u.name+"_jump"+' '+filename+u.name+'_jump_max.scl****\n')
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        if case_filename is None:
            case_filename = filename
        if u.isVector:
            if not append:
                caseOut=open(case_filename+'.case','a')
                if firstVariable==True:
                    caseOut.write('VARIABLE\n')
                caseOut.write('vector per node: '+
                              u.name+' '+filename+u.name+'_average.vec****\n')
                caseOut.write('vector per node: '+
                        u.name+' '+filename+u.name+'_jump_max.vec****\n')
            AverageOut=open(filename+u.name+'_average.vec%4.4i' % self.nOutput,'w')
            JumpOut=open(filename+u.name+'_jump_max.vec%4.4i' % self.nOutput,'w')
        else:
            if not append:
                caseOut=open(case_filename+'.case','a')
                caseOut.write('VARIABLE\n')
                for t in u.range_dim_dof:
                    caseOut.write('scalar per node: '+
                                  u.name+"_average"+' '+filename+u.name+'_average.scl****\n')
                    caseOut.write('scalar per node: '+
                                  u.name+"_jump"+' '+filename+u.name+'_jump_max.scl****\n')

            AverageOut=open(filename+u.name+'_average.scl%4.4i' % self.nOutput,'w')
            JumpOut=open(filename+u.name+'_jump_max.scl%4.4i' % self.nOutput,'w')
        AverageOut.write(u.name+'\n')
        AverageOut.write('part\n'+'%10i\n' % 1)
        AverageOut.write('coordinates\n')
        JumpOut.write(u.name+'\n')
        JumpOut.write('part\n'+'%10i\n' % 1)
        JumpOut.write('coordinates\n')
        nodal_average = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_min= numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_jump_max = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                      'd')
        nodal_nDOF = numpy.zeros((self.elementMaps.mesh.nNodes_global,),
                                   'd')
        n_xi    = u.femSpace.referenceFiniteElement.localFunctionSpace.referenceElement.nNodes
        xiArray = u.femSpace.referenceFiniteElement.localFunctionSpace.referenceElement.nodeList
        psi = numpy.zeros((n_xi,u.femSpace.referenceFiniteElement.localFunctionSpace.dim),
                            'd')
        udofs = numpy.zeros(u.femSpace.referenceFiniteElement.localFunctionSpace.dim,
                              'd')

        for t in u.range_dim_dof:
            nodal_average[:]=0.0
            nodal_max[:]=0.0
            nodal_min[:]=0.0
            nodal_jump_max[:]=0.0
            nodal_nDOF[:]=0.0
            for eN in range(self.elementMaps.mesh.nElements_global):
                psi.flat[:] = 0.0; udofs.flat[:] = 0.0
                for j in range(self.referenceFiniteElement.localFunctionSpace.dim):
                    J = self.dofMap.l2g[eN,j]*u.dim_dof+t
                    udofs[j] = u.dof[J]
                    for k in range(n_xi):
                        psi[k,j] = u.femSpace.referenceFiniteElement.localFunctionSpace.basis[j](xiArray[k])
                    #end k
                #j
                uatNodes = numpy.dot(psi,udofs) #should be nNodes long
                for i in range(self.elementMaps.mesh.nNodes_element):
                    n = self.elementMaps.mesh.elementNodesArray[eN,i]
                    u_node = uatNodes[i]
                    nodal_average[n] += u_node
                    nodal_nDOF[n]+=1
                    if nodal_nDOF[n] >=2:
                        nodal_max[n] = max(nodal_max[n],u_node)
                        nodal_min[n] = min(nodal_min[n],u_node)
                        nodal_jump_max[n] = nodal_max[n]-nodal_min[n]
                    #
                #i
            #eN
            for n in range(self.elementMaps.mesh.nNodes_global):
                nodal_average[n] /= nodal_nDOF[n]
                AverageOut.write('%12.5e\n' % nodal_average[n])
                JumpOut.write('%12.5e\n' % nodal_jump_max[n])
        if u.isVector:
            for t in range(3 - u.dim_dof):
                for n in range(self.elementMaps.mesh.nNodes_global):
                    AverageOut.write('%12.5e\n' % 0.0)
                    JumpOut.write('%12.5e\n' % 0.0)
        AverageOut.close()
        JumpOut.close()
        self.nOutput+=1
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_CrouzeixRaviartP1(ar,self.elementMaps.mesh,
                                                               self.nSpace_global,
                                                               self.dofMap,
                                                               t=t,init=init,meshChanged=meshChanged,
                                                               arGrid=arGrid,tCount=tCount)

    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        return self.XdmfWriter.writeFunctionXdmf_CrouzeixRaviartP1(ar,u,tCount=tCount,init=init, dofMap=self.dofMap)

    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        return self.XdmfWriter.writeVectorFunctionXdmf_CrouzeixRaviartP1(ar,uList,components,vectorName,tCount=tCount,init=init, dofMap=self.dofMap)

    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_CrouzeixRaviartP1(fout,
                                                       u.femSpace.nSpace_global,
                                                       u.femSpace.elementMaps.mesh.nodeArray,
                                                       u.femSpace.elementMaps.mesh.elementNodesArray,
                                                       u.femSpace.dofMap.l2g,
                                                       u.dof,
                                                       name=u.name,
                                                       storeMeshData=storeMeshData,
                                                       figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0
    def getValuesAtMeshNodes(self,dof,nodalValues,isVector,dim_dof):
        """
        Calculate function at mesh nodes from degrees of freedom defined elsewhere
        """
        cfemIntegrals.computeC0P1InterpolantNCP1(self.elementMaps.mesh.elementNodesArray,
                                                 self.elementMaps.mesh.nodeElementOffsets,
                                                 self.elementMaps.mesh.nodeElementsArray,
                                                 self.dofMap.l2g,
                                                 dof,
                                                 nodalValues,
                                                 dim_dof)
#end NC

class DG_Constants(ParametricFiniteElementSpace):
    """
    The constant DG space
    """
    def __init__(self,mesh,nd=3):
        localFunctionSpace = p0(nd)
        interpolationConditions = p0InterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              p0DOFMap(mesh))
        self.strongDirichletConditions = False
        from . import Archiver
        self.XdmfWriter = Archiver.XdmfWriter()
    def writeMeshEnsight(self,filename,description=None):
        self.mesh.writeMeshEnsight(filename,description)#need to allow paraview to even read in quadpoint data
    def writeFunctionGnuplot(self,u,filename):
        pass
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        #mwf hack, to allow for output from quadrature arrays even if not plotting solution directly
        self.nOutput+= 1
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        pass
    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_DGP0(fout,
                                          u.femSpace.nSpace_global,
                                          u.femSpace.elementMaps.mesh.nodeArray,
                                          u.femSpace.elementMaps.mesh.elementNodesArray,
                                          u.femSpace.dofMap.l2g,
                                          u.dof,
                                          name=u.name,
                                          storeMeshData=storeMeshData,
                                          figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_DGP0(ar,self.elementMaps.mesh,
                                                  self.nSpace_global,
                                                  t=t,init=init,meshChanged=meshChanged,
                                                  arGrid=arGrid,tCount=tCount)

    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        return self.XdmfWriter.writeFunctionXdmf_DGP0(ar,u,tCount=tCount,init=init)

    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        return self.XdmfWriter.writeVectorFunctionXdmf_DGP0(ar,uList,components,vectorName,tCount=tCount,init=init)

    def getValuesAtMeshNodes(self,dof,nodalValues,isVector,dim_dof):
        """
        Compute function at mesh nodes from degrees of freedom defined elsewhere
        """
        cfemIntegrals.computeC0P1InterpolantDGP0(self.elementMaps.mesh.elementNodesArray,
                                                 self.elementMaps.mesh.nodeElementOffsets,
                                                 self.elementMaps.mesh.nodeElementsArray,
                                                 self.dofMap.l2g,
                                                 dof,
                                                 nodalValues,
                                                 dim_dof)

class C0_AffineP1BubbleOnSimplexWithNodalBasis(ParametricFiniteElementSpace):
    """
    C0P1 enriched with bubble functions
    """
    def __init__(self,mesh,nd=3):
        localFunctionSpace = P1BubblesWithNodalBasis(nd)
        interpolationConditions = P1BubbleInterpolationConditions(localFunctionSpace.referenceElement)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              P1BubbleDOFMap(mesh,localFunctionSpace,nd))
        self.strongDirichletConditions = True
        #for archiving
        from . import Archiver
        self.XdmfWriter = Archiver.XdmfWriter()

    def writeMeshEnsight(self,filename,description=None):
        self.mesh.writeMeshEnsight(filename,description)#need to allow paraview to even read in quadpoint data
    def writeFunctionGnuplot(self,u,filename):
        pass
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        #mwf hack, to allow for output from quadrature arrays even if not plotting solution directly
        self.nOutput+= 1
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        pass
    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written

        Warning, currently has to compute values at interpolation points!
        tries to use basis values at interpolation points if in u
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            basisValuesAtInterpolationPoints = None
            if 'basisValuesAtInterpolationPoints' in dir(u):
                basisValuesAtInterpolationPoints = u.basisValuesAtInterpolationPoints
            else:
                u.basisValuesAtInterpolationPoints = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                                  u.femSpace.interpolationPoints.shape[1],
                                                                  u.femSpace.referenceFiniteElement.localFunctionSpace.dim),
                                                                 'd')
                u.femSpace.getBasisValues(u.femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                          u.basisValuesAtInterpolationPoints)

            interpolationValuesArray        = None
            if 'interpolationValuesArray' not in dir(u):
                u.interpolationValuesArray = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                          u.femSpace.interpolationPoints.shape[1]),'d')
            u.getValues(u.basisValuesAtInterpolationPoints,u.interpolationValuesArray)
            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_MonomialDGPK(fout,
                                                  u.femSpace.nSpace_global,
                                                  u.femSpace.elementMaps.mesh.nodeArray,
                                                  u.femSpace.elementMaps.mesh.elementNodesArray,
                                                  u.femSpace.interpolationPoints,
                                                  u.interpolationValuesArray,
                                                  name=u.name,
                                                  storeMeshData=storeMeshData,
                                                  figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_P1Bubble(ar,self.elementMaps.mesh,self.nSpace_global,
                                                      self.dofMap.l2g,
                                                      t=t,init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)

    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        """
        not much choice except to get u values at interpolation points?
        """
        return self.XdmfWriter.writeFunctionXdmf_P1Bubble(ar,u,tCount=tCount,init=init)
    #
    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        """
        Write function to XDFM rep
        """
        return self.XdmfWriter.writeVectorFunctionXdmf_P1Bubble(ar,uList,components,vectorName,"_c0p1_Bubble",tCount=tCount,init=init)
    def getValuesAtMeshNodes(self,dof,nodalValues,isVector,dim_dof):
        """
        Calculate function values at mesh nodes from degrees of freedom defined elsewhere
        """
        ntot = len(nodalValues.flat)
        nodalValues.flat[:] = dof.flat[:ntot]



class C0_AffineP1P0BubbleOnSimplexWithNodalBasis(ParametricFiniteElementSpace):
    """
    TODO set output to take advantage of nodal information
    """
    def __init__(self,mesh,nd=3):
        localFunctionSpace = P1P0BubblesWithNodalBasis(nd)
        interpolationConditions = P1P0BubbleInterpolationConditions(localFunctionSpace.referenceElement,localFunctionSpace)
        ParametricFiniteElementSpace.__init__(self,
                                              ReferenceFiniteElement(localFunctionSpace,
                                                                     interpolationConditions),
                                              AffineMaps(mesh,
                                                         localFunctionSpace.referenceElement,
                                                         LinearOnSimplexWithNodalBasis(nd)),
                                              P1BubbleDOFMap(mesh,localFunctionSpace,nd))
        self.strongDirichletConditions = True
        #for archiving
        from . import Archiver
        self.XdmfWriter = Archiver.XdmfWriter()

    def writeMeshEnsight(self,filename,description=None):
        self.mesh.writeMeshEnsight(filename,description)#need to allow paraview to even read in quadpoint data
    def writeFunctionGnuplot(self,u,filename):
        pass
    def writeFunctionEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        #mwf hack, to allow for output from quadrature arrays even if not plotting solution directly
        self.nOutput+= 1
    def writeFunctionHeaderEnsight(self,u,filename,append=False,firstVariable=True,case_filename=None):
        pass
    def writeFunctionMatlab(self,u,output,append=True,storeMeshData=True,figureOffset=1):
        """
        save a scalar finite element function to matlab format for viewing
        returns number of function representations written

        Warning, currently has to compute values at interpolation points!
        tries to use basis values at interpolation points if in u
        """
        if not u.isVector:
            if isinstance(output,file):
                fout = output
            elif isinstance(output,str):
                if append:
                    fout = open(output,'a')
                else:
                    fout = open(output,'w')
            else:
                raise IOError("output = %s should be file or filename")

            basisValuesAtInterpolationPoints = None
            if 'basisValuesAtInterpolationPoints' in dir(u):
                basisValuesAtInterpolationPoints = u.basisValuesAtInterpolationPoints
            else:
                u.basisValuesAtInterpolationPoints = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                                  u.femSpace.interpolationPoints.shape[1],
                                                                  u.femSpace.referenceFiniteElement.localFunctionSpace.dim),
                                                                 'd')
                u.femSpace.getBasisValues(u.femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                          u.basisValuesAtInterpolationPoints)

            interpolationValuesArray        = None
            if 'interpolationValuesArray' not in dir(u):
                u.interpolationValuesArray = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                          u.femSpace.interpolationPoints.shape[1]),'d')
            u.getValues(u.basisValuesAtInterpolationPoints,u.interpolationValuesArray)
            from . import Viewers
            writer = Viewers.MatlabWriter(nxgrid=50,nygrid=50,nzgrid=10)
            nout = writer.viewScalar_MonomialDGPK(fout,
                                                  u.femSpace.nSpace_global,
                                                  u.femSpace.elementMaps.mesh.nodeArray,
                                                  u.femSpace.elementMaps.mesh.elementNodesArray,
                                                  u.femSpace.interpolationPoints,
                                                  u.interpolationValuesArray,
                                                  name=u.name,
                                                  storeMeshData=storeMeshData,
                                                  figureNumber=figureOffset)
            if isinstance(output,str):
                fout.close()

            return nout
        #scalar
        return 0
    def writeMeshXdmf(self,ar,name,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.XdmfWriter.writeMeshXdmf_MonomialDGPK(ar,self.elementMaps.mesh,self.nSpace_global,
                                                          self.interpolationPoints,
                                                          t=t,init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)

    def writeFunctionXdmf(self,ar,u,tCount=0,init=True):
        """
        not much choice except to get u values at interpolation points?
        """
        basisValuesAtInterpolationPoints = None
        if 'basisValuesAtInterpolationPoints' in dir(u):
            basisValuesAtInterpolationPoints = u.basisValuesAtInterpolationPoints
        else:
            u.basisValuesAtInterpolationPoints = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                              u.femSpace.interpolationPoints.shape[1],
                                                              u.femSpace.referenceFiniteElement.localFunctionSpace.dim),
                                                             'd')
            u.femSpace.getBasisValues(u.femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray,
                                      u.basisValuesAtInterpolationPoints)

        interpolationValuesArray        = None
        if 'interpolationValuesArray' not in dir(u):
            u.interpolationValuesArray = numpy.zeros((u.femSpace.interpolationPoints.shape[0],
                                                      u.femSpace.interpolationPoints.shape[1]),'d')
        u.getValues(u.basisValuesAtInterpolationPoints,u.interpolationValuesArray)

        return self.XdmfWriter.writeFunctionXdmf_MonomialDGPK(ar,u.interpolationValuesArray,u.name,tCount=tCount,init=init,mesh=u.femSpace.mesh)
    #
    def writeVectorFunctionXdmf(self,ar,uList,components,vectorName,tCount=0,init=True):
        """
        Write function to XDMF representation
        """
        logEvent("Monomial DGPK writeVectorFunctionXdmf not implemented yet",level=1)


"""
Members of the finite element space.
"""
from . import Comm

class FiniteElementFunction(object):
    """A member of a finite element space of scalar functions.

    This class manages the calculation of function, gradient, tensor,
    and trace values associated with a finite element function.
    Typically, this function will be a scalar function  (For example,
    in a 2D stokes problem with a pressure, and two velocity
    components (say u or v), this class will manage the calculation of
    one of the scalar functions for p, u or v.

    Arguments
    ---------
    finiteElementSpace : :class:`proteus.FemTools.ParametricFiniteElementSpace`

    Notes
    -----
    `femSpace` and `dof` are the two main attributes of this class.
    The `dof` attributes provide the scalar multiples for the
    underlying basis functions that comes from `femSpace`'s
    ParametricFiniteElementSpace object
    """
    def __init__(self,finiteElementSpace,dof=None,dof_last=None,dof_last_last=None,dim_dof=1,name="no_name",isVector=False):
        #: Name assigned to the FiniteElementFunction instance
        self.name=name
        #: boolean variable identifying whether the
        #: finiteElementFunction is vector quantitied.
        self.isVector=isVector
        #: :class:`proteus.FemTools.ParametricFiniteElementSpace`
        #: provides information on the underlying finite element
        #: space.
        self.femSpace = finiteElementSpace
        #: integer variable with the global number of degrees of
        #: freedom
        self.nDOF_global = self.femSpace.dim
        #: dimension of the degree of freedom
        self.dim_dof = dim_dof
        self.range_dim_dof = list(range(self.dim_dof))
        if dof is not None:
            #: numpy array of degree-of-freedom values used to
            #: calculate the finite element function.
            self.dof=dof
            self.dof_last=dof_last
            self.dof_last_last=dof_last_last
        else:
            self.dof = numpy.zeros((self.femSpace.dim*dim_dof),
                                     'd')
            self.dof_last = numpy.zeros((self.femSpace.dim*dim_dof),
                                     'd')
            self.dof_last_last = numpy.zeros((self.femSpace.dim*dim_dof),
                                     'd')

        #: boolean variable indentifying whether calculations should
        #: use C-routines.
        self.useC=True
        #we need to be able to get references to existing values for values at nodes for some calculations
        #like vtkVisualization
        self.meshNodeValues = None
        #: for parallel FiniteElementFunciton capability
        self.par_dof = None

    def copy(self):
        return FiniteElementFunction(finiteElementSpace=self.femSpace,
                                     dof=self.dof.copy(),
                                     dim_dof=self.dim_dof,
                                     name=self.name,
                                     isVector=self.isVector)

    def projectFromInterpolationConditions(self,interpolationValues):
        if self.useC and self.femSpace.referenceFiniteElement.interpolationConditions.projectFiniteElementFunctionFromInterpolationConditions_opt is not None:
            self.femSpace.referenceFiniteElement.interpolationConditions.projectFiniteElementFunctionFromInterpolationConditions_opt(self,interpolationValues)
        else:
            functionals = self.femSpace.referenceFiniteElement.interpolationConditions.functionalsQuadrature
            for eN in range(self.femSpace.elementMaps.mesh.nElements_global):
                for i in self.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                    dof_eN_i = functionals[i](interpolationValues[eN])
                    self.dof[self.femSpace.dofMap.l2g[eN,i]*self.dim_dof:self.femSpace.dofMap.l2g[eN,i]*self.dim_dof+self.dim_dof] = dof_eN_i
    def getValue(self,eN,xi):
        """ Calculate the function value at a point on an element.

        Arguments
        ---------
        eN : int
            Global element number.
        xi : point
            Evaluation coordinate.

        Returns
        -------
        value: float
           The finite element functions value at xi.
        """
        value = 0.0
        for i,psi in zip(
            self.femSpace.dofMap.l2g[eN],
            self.femSpace.elementMaps.localFunctionSpace.basis):
            value+=self.dof[i]*psi(xi)
        return value
    def getValues(self,
                  v,
                  u):
        n_xi = v.shape[1]
        if self.useC==True:
            cfemIntegrals.calculateFiniteElementFunctionValues(self.femSpace.dofMap.l2g,
                                                               self.dof,
                                                               v,
                                                               u)
        else:
            u.flat[:]=0.0
            range_n_xi = list(range(n_xi))
            for eN in range(self.femSpace.elementMaps.mesh.nElements_global):
                for k in range_n_xi:
                    for j in self.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                        J = self.femSpace.dofMap.l2g[eN,j]
                        for t in self.range_dim_dof:
                            u[eN,k*self.dim_dof + t]+=self.dof[J*self.dim_dof + t]*v[eN,k,j]
    def getGradientValues(self,
                          grad_v,
                          grad_u):
        n_xi = grad_v.shape[1]
        nSpace = grad_v.shape[-1]
        if self.useC==True:
            cfemIntegrals.calculateFiniteElementFunctionGradientValues(self.femSpace.dofMap.l2g,
                                                                       self.dof,
                                                                       grad_v,
                                                                       grad_u)
        else:
            range_n_xi = list(range(n_xi))
            grad_u.flat[:]=0.0
            for eN in range(self.femSpace.elementMaps.mesh.nElements_global):
                for k in range_n_xi:
                    for j in self.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                        J = self.femSpace.dofMap.l2g[eN,j]
                        for t in self.range_dim_dof:
                            for m in self.femSpace.referenceFiniteElement.referenceElement.range_dim:
                                grad_u[eN,k*self.dim_dof + t,m]+=self.dof[J*self.dim_dof+t]*grad_v[eN,k,j,m]
    def getHessianValues(self,
                         Hessian_v,
                         Hessian_u):
        n_xi = Hessian_v.shape[1]
        nSpace = Hessian_v.shape[-1]
        cfemIntegrals.calculateFiniteElementFunctionHessianValues(self.femSpace.dofMap.l2g,
                                                                  self.dof,
                                                                  Hessian_v,
                                                                  Hessian_u)
    def getGradientTensorValues(self,
                                grad_v_x_grad_w,
                                grad_u_x_grad_w):
        n_xi = grad_v_x_grad_w.shape[1]
        nSpace = grad_v_x_grad_w.shape[-1]
        cfemIntegrals.calculateFiniteElementFunctionGradientTensorValues(self.femSpace.dofMap.l2g,
                                                                         self.dof,
                                                                         grad_v_x_grad_w,
                                                                         grad_u_x_grad_w)
    def getValuesTrace(self,
                       v,
                       u):
        n_xi = v.shape[2]
        if self.useC==True:
            cfemIntegrals.calculateFiniteElementFunctionValuesTrace(self.femSpace.dofMap.l2g,
                                                                   self.dof,
                                                                   v,
                                                                   u)
        else:
            u.flat[:]=0.0
            range_n_xi = list(range(n_xi))
            for eN in range(self.femSpace.elementMaps.mesh.nElements_global):
                for ebN in self.femSpace.referenceFiniteElement.referenceElement.range_nElementBoundaries:
                    for k in range_n_xi:
                        for j in self.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                            J = self.femSpace.dofMap.l2g[eN,j]
                            for t in self.range_dim_dof:
                                u[eN,ebN,k*self.dim_dof + t]+=self.dof[J*self.dim_dof + t]*v[eN,ebN,k,j]
    def getGradientValuesTrace(self,
                               grad_v,
                               grad_u):
        n_xi = grad_v.shape[2]
        nSpace = grad_v.shape[-1]
        if self.useC==True:
            cfemIntegrals.calculateFiniteElementFunctionGradientValuesTrace(self.femSpace.dofMap.l2g,
                                                                           self.dof,
                                                                           grad_v,
                                                                           grad_u)
        else:
            grad_u.flat[:]=0.0
            range_n_xi = list(range(n_xi))
            for eN in range(self.femSpace.elementMaps.mesh.nElements_global):
                for ebN in self.femSpace.referenceFiniteElement.referenceElement.range_nElementBoundaries:
                    for k in range_n_xi:
                        for j in self.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                            J = self.femSpace.dofMap.l2g[eN,j]
                            for t in self.range_dim_dof:
                                for m in self.femSpace.referenceFiniteElement.referenceElement.range_dim:
                                    grad_u[eN,ebN,k*self.dim_dof + t,m]+=self.dof[J*self.dim_dof + t]*grad_v[eN,ebN,k,j,m]
    def getValuesGlobalExteriorTrace(self,
                                     v,
                                     u):
        n_xi = v.shape[2]
        if self.useC==True:
            cfemIntegrals.calculateFiniteElementFunctionValuesGlobalExteriorTrace(self.femSpace.elementMaps.mesh.exteriorElementBoundariesArray,
                                                                                  self.femSpace.elementMaps.mesh.elementBoundaryElementsArray,
                                                                                  self.femSpace.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                  self.femSpace.dofMap.l2g,
                                                                                  self.dof,
                                                                                  v,
                                                                                  u)
        else:
            u.flat[:]=0.0
            range_n_xi = list(range(n_xi))
            for ebNE in range(self.femSpace.elementMaps.mesh.nExteriorElementBoundaries_global):
                ebN = self.femSpace.elementMaps.mesh.exteriorElementBoundariesArray[ebNE]
                eN  = self.femSpace.elementMaps.mesh.elementBoundaryElementsArray[ebN,0]
                ebN_local = self.femSpace.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                for k in range_n_xi:
                    for j in self.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                        J = self.femSpace.dofMap.l2g[eN,j]
                        for t in self.range_dim_dof:
                            u[ebNE,k*self.dim_dof + t]+=self.dof[J*self.dim_dof + t]*v[ebNE,k,j]
    def getGradientValuesGlobalExteriorTrace(self,
                                             grad_v,
                                             grad_u):
        n_xi = grad_v.shape[1]
        nSpace = grad_v.shape[-1]
        if self.useC==True:
            cfemIntegrals.calculateFiniteElementFunctionGradientValuesGlobalExteriorTrace(self.femSpace.elementMaps.mesh.exteriorElementBoundariesArray,
                                                                                          self.femSpace.elementMaps.mesh.elementBoundaryElementsArray,
                                                                                          self.femSpace.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                          self.femSpace.dofMap.l2g,
                                                                                          self.dof,
                                                                                          grad_v,
                                                                                          grad_u)
        else:
            grad_u.flat[:]=0.0
            range_n_xi = list(range(n_xi))
            for ebNE in range(self.femSpace.elementMaps.mesh.nExteriorElementBoundaries_global):
                ebN = self.femSpace.elementMaps.mesh.exteriorElementBoundariesArray[ebNE]
                eN  = self.femSpace.elementMaps.mesh.elementBoundariesArray[ebN,0]
                ebN_local = self.femSpace.elementMaps.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                for k in range_n_xi:
                    for j in self.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                        J = self.femSpace.dofMap.l2g[eN,j]
                        for t in self.range_dim_dof:
                            for m in self.femSpace.referenceFiniteElement.referenceElement.range_dim:
                                grad_u[ebNE,k*self.dim_dof + t,m]+=self.dof[J*self.dim_dof + t]*grad_v[ebNE,k,j,m]

    def writeFunctionGnuplot(self,filename,append=False):
        self.femSpace.writeFunctionGnuplot(self,filename)
    def writeFunctionEnsight(self,filename,append=False,case_filename=None):
        self.femSpace.writeFunctionEnsight(self,filename,append,case_filename)
    def writeFunctionMatlab(self,output,append=True,storeMeshData=True,figureOffset=1):
        return self.femSpace.writeFunctionMatlab(self,output,append=append,storeMeshData=storeMeshData,figureOffset=figureOffset)

    #we need to be able to get references to existing values for values at nodes for some calculations
    #like vtkVisualization, call structure is different that getValues because realy want internal storage to be modified
    def calculateValuesAtMeshNodes(self):
        if self.meshNodeValues is None:
            self.meshNodeValues = numpy.zeros((self.femSpace.mesh.nNodes_global*self.dim_dof),'d')
        self.femSpace.getValuesAtMeshNodes(self.dof,self.meshNodeValues,self.isVector,self.dim_dof)

    #
    def setupParallelCommunication(self):
        """
        build data structure for parallel communication
        """
        if self.femSpace.dofMap.dof_offsets_subdomain_owned is None:
            logEvent("WARNING setupParallelCommunication not valid for %s must have parallel information for dofMap" % self,level=-1)
            return
        comm = Comm.get()
        par_n = self.femSpace.dofMap.dof_offsets_subdomain_owned[comm.rank()+1] - self.femSpace.dofMap.dof_offsets_subdomain_owned[comm.rank()]
        par_N = self.femSpace.dofMap.nDOF_all_processes
        par_nghost = self.femSpace.dofMap.nDOF_subdomain - par_n
        subdomain2global = self.femSpace.dofMap.subdomain2global
        max_dof_neighbors= self.femSpace.dofMap.max_dof_neighbors
        par_bs = self.dim_dof
        self.par_dof = ParVec_petsc4py(self.dof,par_bs,par_n,par_N,par_nghost,subdomain2global)

"""
Boundary Conditions
"""

class DOFBoundaryConditions(object):
    """
    A class for generating the set of DOF that are replaced by
    Dirichlet conditions and the values that are to be assigned to
    those DOF. For now I will ignore the ability to specify
    different boundary conditions for interpolatory DOF that correspond
    to the same physical point (i.e. discontinuous approximations where
    left and right limits can be specified)

    DOFBoundaryConditionDict -- a dictionary of boundary condition functions
    accessed by global DOF number
    DOFBoundryConditionPointDict -- a dictionary of physical points associated
    with the boundary condition
    freeDOFSet -- the DOF not specified by boundary conditions
    global2freeGlobal -- an integer mapping from global DOF numbers
    to the free DOF numbers.

    The constructor requires a function that takes a point as input
    and, if the point is on a part of the boundary where values are
    specified, then it returns the function of t,x, and the unknown
    that computes the boundary condition.

    TODO
     For now allow for flag to specify type of dirichlet condition to allow
     say nonlinear function of solution to be specified
     initially, only weak bc's will allow this functionality though
    """
    def __init__(self,femSpace,getPointwiseBoundaryConditions=None,weakDirichletConditions=False,
                 getPeriodicBoundaryConditions=None,allowNodalMaterialBoundaryTypes=True):
        self.DOFBoundaryConditionsDict={}
        self.DOFBoundaryPointDict={}
        self.DOFBoundaryMaterialFlag={}
        self.freeDOFSet=set()
        self.periodicDOFDict={}
        class ptuple(object):
            """
            define a dictionary key that defines points as equal if they're "close"
            """
            h=femSpace.mesh.hMin
            def __init__(self,p):
                self.p=p
            def __hash__(self):
                return hash(tuple(self.p))
            def __eq__(self,other):
                return  enorm(self.p - other.p) <= self.h
        if getPointwiseBoundaryConditions is not None and femSpace.strongDirichletConditions and not weakDirichletConditions:
            for eN in range(femSpace.elementMaps.mesh.nElements_global):
            # mesh = femSpace.elementMaps.mesh
            # for ebNE in range(mesh.nExteriorElementBoundaries_global):
            #     ebN = mesh.exteriorElementBoundariesArray[ebNE]
            #     eN = mesh.elementBoundaryElementsArray[ebN,0]
                for k in range(femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints):
                    i = femSpace.referenceFiniteElement.interpolationConditions.quadrature2DOF_element(k)
                    dofN = femSpace.dofMap.l2g[eN,i]
                    x = femSpace.interpolationPoints[eN,k]
                    try:
                        #in case interpolation is wholly in element interior
                        interiorInterpolationPoint = True
                        for ebN_element in range(femSpace.elementMaps.mesh.nElementBoundaries_element):
                            if femSpace.referenceFiniteElement.interpolationConditions.definedOnLocalElementBoundary(k,ebN_element) == True:
                                interiorInterpolationPoint = False
                                ebN = femSpace.elementMaps.mesh.elementBoundariesArray[eN,ebN_element]
                                materialFlag = femSpace.elementMaps.mesh.elementBoundaryMaterialTypes[ebN]
                                #mwf now allow for flag to specify type of dirichlet condition to allow
                                #say nonlinear function of solution to be specified
                                #initially, only weak bc's will allow this functionality though
                                gReturn = getPointwiseBoundaryConditions(x,materialFlag)
                                try:
                                    g = gReturn[0]
                                    gFlag = gReturn[1]
                                except TypeError:
                                    g = gReturn
                                    gFlag = 1
                                if gFlag != 1:
                                    logEvent("WARNING strong Dirichlet conditions do not enforce nonlinear bcs")
                                p = None
                                if getPeriodicBoundaryConditions is not None:
                                    p = getPeriodicBoundaryConditions(x,materialFlag)
                                if p is not None:
                                    if ptuple(p) in self.periodicDOFDict:
                                        self.periodicDOFDict[ptuple(p)].add(dofN)
                                    else:
                                        self.periodicDOFDict[ptuple(p)] = set([dofN])
                                elif g is not None:
                                    self.DOFBoundaryConditionsDict[dofN] = g
                                    self.DOFBoundaryPointDict[dofN]=x
                                    self.DOFBoundaryMaterialFlag[dofN] = materialFlag
                                    self.freeDOFSet.discard(dofN)
                                else:
                                    if dofN not in list(self.DOFBoundaryConditionsDict.keys()):
                                        self.freeDOFSet.add(dofN)
                                #has Dirichlet bc set or not
                            #on ebN_element
                        #local faces
                        if interiorInterpolationPoint and dofN not in list(self.DOFBoundaryConditionsDict.keys()):
                            self.freeDOFSet.add(dofN)
                    except TypeError:
                        logEvent("""WARNING DOFBoundaryCondition Pointwise conditions should take arguments (x,flag) now trying without flag""")
                        #mwf now allow for flag to specify type of dirichlet condition to allow
                        #say nonlinear function of solution to be specified
                        #initially, only weak bc's will allow this functionality though
                        gReturn = getPointwiseBoundaryConditions(x)
                        try:
                            g = gReturn[0]
                            gFlag = gReturn[1]
                        except TypeError:
                            g = gReturn
                            gFlag = 1
                        if gFlag != 1:
                            logEvent("WARNING strong Dirichlet conditions do not enforce nonlinear bcs")
                        p = None
                        if getPeriodicBoundaryConditions is not None:
                            p = getPeriodicBoundaryConditions(x)
                        if p is not None:
                            #print "periodic DOF bc ",tuple(p)
                            if ptuple(p) in self.periodicDOFDict:
                                self.periodicDOFDict[ptuple(p)].add(dofN)
                                self.freeDOFSet.discard(dofN)
                            else:
                                #print "inserting dof in periodic and free",dofN
                                self.periodicDOFDict[ptuple(p)] = set([dofN])
                                self.freeDOFSet.discard(dofN)
                        elif g:
                            self.DOFBoundaryConditionsDict[dofN] = g
                            self.DOFBoundaryPointDict[dofN]=x
                            self.freeDOFSet.discard(dofN)
                        else:
                            if dofN not in list(self.DOFBoundaryConditionsDict.keys()):
                                self.freeDOFSet.add(dofN)
                    #exception on argument list for getPointWiseBoundaryConditions
                    #now also try setting Dirichlet boundary conditions using nodal id tags
                    nN_element = femSpace.referenceFiniteElement.interpolationConditions.quadrature2Node_element(k)
                    ##todo work out use cases where this matters
                    if allowNodalMaterialBoundaryTypes and (nN_element is not None and nN_element < femSpace.elementMaps.mesh.nNodes_element):
                        try:
                            nN_global = femSpace.elementMaps.mesh.elementNodesArray[eN,nN_element]
                            materialFlag = femSpace.elementMaps.mesh.nodeMaterialTypes[nN_global]
                            gReturn = getPointwiseBoundaryConditions(x,materialFlag)
                            try:
                                g = gReturn[0]
                                gFlag = gReturn[1]
                            except TypeError:
                                g = gReturn
                                gFlag = 1
                            if gFlag != 1:
                                logEvent("WARNING strong Dirichlet conditions do not enforce nonlinear bcs")
                            p = None
                            if getPeriodicBoundaryConditions is not None:
                                p = getPeriodicBoundaryConditions(x,materialFlag)
                            if p is not None: #skip dof setting here
                                pass#cek changed from break to pass/elif
                            elif g: #override elementBoundary condition
                                self.DOFBoundaryConditionsDict[dofN] = g
                                self.DOFBoundaryPointDict[dofN]=x
                                self.DOFBoundaryMaterialFlag[dofN] = materialFlag
                                self.freeDOFSet.discard(dofN)
                            else:
                                if dofN not in list(self.DOFBoundaryConditionsDict.keys()):
                                    self.freeDOFSet.add(dofN)
                            #has Dirichlet bc set or not
                        except TypeError:
                            logEvent("""WARNING DOFBoundaryCondition Pointwise conditions should take arguments (x,flag) skipping nodal flag test""")
                        #
                    #has correspondence between interpolation conditions and mesh nodes
                #k
            #eN
        else:
            self.freeDOFSet = set(range(femSpace.dim))
        #
        for nodeSet in list(self.periodicDOFDict.values()):
            nodeList = list(nodeSet)
            nodeList.sort()
            self.freeDOFSet.add(nodeList[0])
            for dofN in nodeList[1:]:
                self.freeDOFSet.discard(dofN)
        self.nFreeDOF_global = len(self.freeDOFSet)
        self.global2freeGlobal={}
        self.myFreeDOF={}
        for free_dofN, dofN in enumerate(self.freeDOFSet):
            self.global2freeGlobal[dofN] = free_dofN
            self.myFreeDOF[dofN] = dofN
        for nodeSet in list(self.periodicDOFDict.values()):
            nodeList = list(nodeSet)
            nodeList.sort()
            free_dofN = self.global2freeGlobal[nodeList[0]]
            print("node list",nodeList)
            for dofN in nodeSet:
                self.global2freeGlobal[dofN] = free_dofN
                self.myFreeDOF[dofN] = nodeList[0]
        #create arrays for iterating over dofs in c
        #not necessarily a 1-1 correspondence between free_dofN and dofN because of
        #periodic bcs, so have to have 2 arrays
        nfree = len(self.global2freeGlobal)
        self.global2freeGlobal_global_dofs = numpy.zeros((nfree,),'i')
        self.global2freeGlobal_free_dofs = numpy.zeros((nfree,),'i')
        test = numpy.array(list(range(nfree)),dtype='i')
        for i,dofN in enumerate(self.global2freeGlobal.keys()):
            self.global2freeGlobal_global_dofs[i] = dofN#map each of the unknown DOF's to the original node number
            self.global2freeGlobal_free_dofs[i] = self.global2freeGlobal[dofN]#map each of the unknown DOF's to the free unknown number


class DOFBoundaryConditions_alt(object):
    """
    A class for generating the set of DOF that are replaced by
    Dirichlet conditions and the values that are to be assigned to
    those DOF. For now I will ignore the ability to specify
    different boundary conditions for interpolatory DOF that correspond
    to the same physical point (i.e. discontinuous approximations where
    left and right limits can be specified)

    DOFBoundaryConditionDict -- a dictionary of boundary condition functions
    accessed by global DOF number
    DOFBoundryConditionPointDict -- a dictionary of physical points associated
    with the boundary condition
    freeDOFSet -- the DOF not specified by boundary conditions
    global2freeGlobal -- an integer mapping from global DOF numbers
    to the free DOF numbers.

    The constructor requires a function that takes a point as input
    and, if the point is on a part of the boundary where values are
    specified, then it returns the function of t,x, and the unknown
    that computes the boundary condition.

    TODO
     For now allow for flag to specify type of dirichlet condition to allow
     say nonlinear function of solution to be specified
     initially, only weak bc's will allow this functionality though

    """
    def __init__(self,femSpace,getPointwiseBoundaryConditions=None,weakDirichletConditions=False,
                 getPeriodicBoundaryConditions=None,allowNodalMaterialBoundaryTypes=True):
        self.DOFBoundaryConditionsDict={}
        self.DOFBoundaryPointDict={}
        self.DOFBoundaryMaterialFlag={}
        self.periodicDOFDict={}
        class ptuple(object):
            """
            define a dictionary key that defines points as equal if they're "close"
            """
            h=femSpace.mesh.hMin
            def __init__(self,p):
                self.p=p
            def __hash__(self):
                return hash(tuple(self.p))
            def __eq__(self,other):
                return  enorm(self.p - other.p) < self.h
        if getPointwiseBoundaryConditions is not None and femSpace.strongDirichletConditions and not weakDirichletConditions:
            for eN in range(femSpace.elementMaps.mesh.nElements_global):
            # mesh = femSpace.elementMaps.mesh
            # for ebNE in range(mesh.nExteriorElementBoundaries_global):
            #     ebN = mesh.exteriorElementBoundariesArray[ebNE]
            #     eN = mesh.elementBoundaryElementsArray[ebN,0]
                for k in range(femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints):
                    i = femSpace.referenceFiniteElement.interpolationConditions.quadrature2DOF_element(k)
                    dofN = femSpace.dofMap.l2g[eN,i]
                    x = femSpace.interpolationPoints[eN,k]

                    #in case interpolation is wholly in element interior
                    interiorInterpolationPoint = True
                    for ebN_element in range(femSpace.elementMaps.mesh.nElementBoundaries_element):
                        if femSpace.referenceFiniteElement.interpolationConditions.definedOnLocalElementBoundary(k,ebN_element) == True:
                            interiorInterpolationPoint = False
                            ebN = femSpace.elementMaps.mesh.elementBoundariesArray[eN,ebN_element]
                            materialFlag = femSpace.elementMaps.mesh.elementBoundaryMaterialTypes[ebN]
                            #mwf now allow for flag to specify type of dirichlet condition to allow
                            #say nonlinear function of solution to be specified
                            #initially, only weak bc's will allow this functionality though
                            gReturn = getPointwiseBoundaryConditions(x,materialFlag)
                            try:
                                g = gReturn[0]
                                gFlag = gReturn[1]
                            except TypeError:
                                g = gReturn
                                gFlag = 1
                            if gFlag != 1:
                                logEvent("WARNING strong Dirichlet conditions do not enforce nonlinear bcs")
                            p = None
                            if getPeriodicBoundaryConditions is not None:
                                p = getPeriodicBoundaryConditions(x,materialFlag)
                            if p is not None:
                                if ptuple(p) in self.periodicDOFDict:
                                    self.periodicDOFDict[ptuple(p)].add(dofN)
                                else:
                                    self.periodicDOFDict[ptuple(p)] = set([dofN])
                            elif g is not None:
                                self.DOFBoundaryConditionsDict[dofN] = g
                                self.DOFBoundaryPointDict[dofN]=x
                                self.DOFBoundaryMaterialFlag[dofN] = materialFlag

                            #has Dirichlet bc set or not
                        #on ebN_element
                    #local faces
                #k
            #eN
        self.freeDOFSet = set(range(femSpace.dim))
        for nodeSet in list(self.periodicDOFDict.values()):
            nodeList = list(nodeSet)
            nodeList.sort()
            self.freeDOFSet.add(nodeList[0])
            for dofN in nodeList[1:]:
                self.freeDOFSet.discard(dofN)
        self.nFreeDOF_global = len(self.freeDOFSet)
        self.global2freeGlobal={}
        self.myFreeDOF={}
        for free_dofN, dofN in enumerate(self.freeDOFSet):
            self.global2freeGlobal[dofN] = free_dofN
            self.myFreeDOF[dofN] = dofN
        for nodeSet in list(self.periodicDOFDict.values()):
            nodeList = list(nodeSet)
            nodeList.sort()
            free_dofN = self.global2freeGlobal[nodeList[0]]
            print("node list",nodeList)
            for dofN in nodeSet:
                self.global2freeGlobal[dofN] = free_dofN
                self.myFreeDOF[dofN] = nodeList[0]
        #create arrays for iterating over dofs in c
        #not necessarily a 1-1 correspondence between free_dofN and dofN because of
        #periodic bcs, so have to have 2 arrays
        nfree = len(self.global2freeGlobal)
        self.global2freeGlobal_global_dofs = numpy.zeros((nfree,),'i')
        self.global2freeGlobal_free_dofs = numpy.zeros((nfree,),'i')
        test = numpy.array(list(range(nfree)),dtype='i')
        for i,dofN in enumerate(self.global2freeGlobal.keys()):
            self.global2freeGlobal_global_dofs[i] = dofN#map each of the unknown DOF's to the original node number
            self.global2freeGlobal_free_dofs[i] = self.global2freeGlobal[dofN]#map each of the unknown DOF's to the free unknown number



class FluxBoundaryConditions(object):
    """Generating class for the fluxBoundaryConditions dictionaries.

    This class manages the generation of the advective, stress and
    diffusive fluxBoundaryCondition dictionaries.  These dictionaries
    store lists of functions that describe the flux conditions along
    the finite element boundaries.

    Attributes
    ----------
    advectiveFluxBoundaryConditionsDict : :obj:`dict`
        Stores advective flux boundary conditions functions.
    stressFluxBoundaryConditionsDict : :obj:`dict`
        Stores stress flux boundary conditions functions.
    diffusiveFluxBoundaryConditionsDictDict : :obj:`dict`
        Stores diffusive flux boundary conditions functions.

    Notes
    -----
    The boundary condition dictionaries all use the key convention
    (ebNE,k) where ebNE is the global edge number and k is the local
    quadrature point number.
    """
    def __init__(self,mesh,nElementBoundaryQuadraturePoints_elementBoundary,x,
                 getAdvectiveFluxBoundaryConditions=None,
                 getDiffusiveFluxBoundaryConditions={},
                 getStressFluxBoundaryConditions=None):
        self.advectiveFluxBoundaryConditionsDict={}
        self.stressFluxBoundaryConditionsDict={}
        self.diffusiveFluxBoundaryConditionsDictDict=dict([(ck,{}) for ck in list(getDiffusiveFluxBoundaryConditions.keys())])
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            materialFlag = mesh.elementBoundaryMaterialTypes[ebN]
            for k in range(nElementBoundaryQuadraturePoints_elementBoundary):
                try:
                    if getAdvectiveFluxBoundaryConditions is not None:
                        g = getAdvectiveFluxBoundaryConditions(x[ebNE,k],materialFlag)
                        if g:
                            self.advectiveFluxBoundaryConditionsDict[(ebNE,k)] = g
                    if getStressFluxBoundaryConditions is not None:
                        g = getStressFluxBoundaryConditions(x[ebNE,k],materialFlag)
                        if g:
                            self.stressFluxBoundaryConditionsDict[(ebNE,k)] = g
                    for ck in list(getDiffusiveFluxBoundaryConditions.keys()):
                        g = getDiffusiveFluxBoundaryConditions[ck](x[ebNE,k],materialFlag)
                        if g:
                            self.diffusiveFluxBoundaryConditionsDictDict[ck][(ebNE,k)] = g
                except TypeError:
                    logEvent("""WARNING FluxBoundaryCondition should take arguments (x,flag) now trying without flag""")
                    g = getAdvectiveFluxBoundaryConditions(x[ebNE,k])
                    if g:
                        self.advectiveFluxBoundaryConditionsDict[(ebNE,k)] = g
                    for ck in list(getDiffusiveFluxBoundaryConditions.keys()):
                        g = getDiffusiveFluxBoundaryConditions[ck](x[ebNE,k])
                        if g:
                            self.diffusiveFluxBoundaryConditionsDictDict[ck][(ebNE,k)] = g
            #k
        #ebNE
class FluxBoundaryConditionsGlobalElementBoundaries(object):
    """
    mwf original version that sets indeces based on all element boundaries
    A class for generating the list of element boundaries
    where values are specified.
    """
    def __init__(self,mesh,nElementBoundaryQuadraturePoints_elementBoundary,x,getAdvectiveFluxBoundaryConditions=None,getDiffusiveFluxBoundaryConditions={}):
        self.advectiveFluxBoundaryConditionsDict={}
        self.diffusiveFluxBoundaryConditionsDictDict=dict([(ck,{}) for ck in list(getDiffusiveFluxBoundaryConditions.keys())])
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            eN_global   = mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            materialFlag = mesh.elementBoundaryMaterialTypes[ebN]
            for k in range(nElementBoundaryQuadraturePoints_elementBoundary):
                try:
                    g = None
                    if getAdvectiveFluxBoundaryConditions:
                        g = getAdvectiveFluxBoundaryConditions(x[ebN,k],materialFlag)
                    if g:
                        self.advectiveFluxBoundaryConditionsDict[(ebN,k)] = g
                    for ck in list(getDiffusiveFluxBoundaryConditions.keys()):
                        g = getDiffusiveFluxBoundaryConditions[ck](x[ebN,k],materialFlag)
                        if g:
                            self.diffusiveFluxBoundaryConditionsDictDict[ck][(ebN,k)] = g
                except TypeError:
                    logEvent("""WARNING FluxBoundaryCondition GlobalElement should take arguments (x,flag) now trying without flag""")
                    g = None
                    if getAdvectiveFluxBoundaryConditions:
                        g = getAdvectiveFluxBoundaryConditions(x[ebN,k])
                    if g:
                        self.advectiveFluxBoundaryConditionsDict[(ebN,k)] = g
                    for ck in list(getDiffusiveFluxBoundaryConditions.keys()):
                        g = getDiffusiveFluxBoundaryConditions[ck](x[ebN,k])
                        if g:
                            self.diffusiveFluxBoundaryConditionsDictDict[ck][(ebN,k)] = g
class StressBoundaryConditions(object):
    """
    A class for generating the list of element boundaries
    where values are specified.
    """
    def __init__(self,mesh,nElementBoundaryQuadraturePoints_elementBoundary,x,getStressTraceBoundaryConditions=None):
        self.stressTraceBoundaryConditionsDict={}
        for ebNE in range(mesh.nExteriorElementBoundaries_global):
            ebN = mesh.exteriorElementBoundariesArray[ebNE]
            materialFlag = mesh.elementBoundaryMaterialTypes[ebN]
            for k in range(nElementBoundaryQuadraturePoints_elementBoundary):
                try:
                    g = getStressTraceBoundaryConditions(x[ebNE,k],materialFlag)
                    if g:
                        self.stressTraceBoundaryConditionsDict[(ebNE,k)] = g
                except TypeError:
                    logEvent("""WARNING FluxBoundaryCondition should take arguments (x,flag) now trying without flag""")
                    g = getStressTraceBoundaryConditions(x[ebNE,k])
                    if g:
                        self.stressTraceBoundaryConditionsDict[(ebNE,k)] = g
            #k
        #ebNE
class MultilevelProjectionOperators(object):
    """
    A class that takes a hierarchical (multiLevel) mesh and generates
    the interpolation and restriction operators.

    restrictList/prolongList -- projection and restriction at only the free nodes
    restrict_bcList/prolong_bcList -- includes dirichlet boundaries as well

    By default this is set up for conforming spaces. Since the spaces
    are conforming the coarse basis functions are in the fine space so we
    need only find the coefficients of the fine space basis functions that
    yield the coarse space basis functions. This is the matrix of the
    trivial injection from coarse to fine and it is used as the projection
    operator. Restriction is taken as the matrix of the adjoint of the
    injection, which is simply the transpose of the projection matrix. These
    operators fall out if you try to solve for the error on the coarse grid:
    Starting with u_f we have a(u_f+e_f,w_f) = <f,w_f>, and we want to
    solve instead  a(e_c,w_c) = <f - a(u_f,w_f),w_c> on the coarse grid
    Using the injection i we can express this in the fine space as
    a(i e_c, i w_c) = <f - a(u_f,w_f),i w_c> writing
    this in matrix form yields p^t A_f p E = p^t R_f
    --- P1 nonconforming space ----
    Try to set up now for nonconforming P1 approximation following Chen_96b.
    Setup prolongation by evaluating coarse grid basis functions at
    fine grid interpolation condition points (face barycenters).

    Then, just need to know if fine grid interpolationCondition point falls on
    interface of coarse grid elements or not. If so, use average value of coarse
    grid quantity on fine grid. Otherwise just evaluate it

    Use this simple interpolation from coarse to fine as the projection operator.
    Restriction is taken as the matrix of the adjoint of the
    injection, which is simply the transpose of the projection matrix.

    I don't think these fall out as nicely since they're nonconforming."""
    ## \todo put the MultilevelProjectionOperators constructor  partially  into C
    def __init__(self,
                 multilevelMesh,
                 femSpaceDictList,
                 offsetListList,
                 strideListList,
                 dofBoundaryConditionsDictList):
        self.nc = len(offsetListList[0])
        self.restrictList = [[]]
        self.rzvalList = [[]]
        self.restrictSumList=[[]]
        self.prolongList = [[]]
        self.pzvalList = [[]]
        self.restrict_bcListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.rbczvalListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.restrict_bcSumListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.prolong_bcListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.pbczvalListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.scaled_restrict_bcListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.scaled_rbczvalListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.interp_bcListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.interp_bczvalListDict = dict([(cj,[[]]) for cj in range(self.nc)])
        self.femSpaceDictList=femSpaceDictList
        self.dof_bc_DictList = dofBoundaryConditionsDictList
        usingAtLeastOneNCproj = False
        for l in range(len(multilevelMesh.meshList)-1):
            r = {}
            p = {}
            coarse_nFreeDOF_global = 0
            fine_nFreeDOF_global=0
            for cj in range(self.nc):
                coarse_nFreeDOF_global += self.dof_bc_DictList[l][cj].nFreeDOF_global
                fine_nFreeDOF_global   += self.dof_bc_DictList[l+1][cj].nFreeDOF_global
            rSum = Vec(coarse_nFreeDOF_global)
            rColumnIndeces=[set() for row in range(coarse_nFreeDOF_global)]
            for cj in range(self.nc):
                coarseSpace = self.femSpaceDictList[l][cj]
                coarseMesh  = multilevelMesh.meshList[l]
                coarseDOFBoundaryConditions = self.dof_bc_DictList[l][cj]
                #cek old mesh interface
                #children = multilevelMesh.elementChildren[l]
                #new mesh interface uses packed arrays, basically a sparse matrix
                children = multilevelMesh.elementChildrenArrayList[l]
                childrenOffsets = multilevelMesh.elementChildrenOffsetsList[l]
                #cek old mesh interface
#                 for coarse_eN in range(coarseMesh.nElements_global):
#                     nChildrenMax_element = max(nChildrenMax_element,len(children[coarse_eN]))
                nChildrenMax_element = max(childrenOffsets[1:] - childrenOffsets[:-1])
                fineSpace = self.femSpaceDictList[l+1][cj]
                fineMesh  = multilevelMesh.meshList[l+1]
                fineDOFBoundaryConditions = self.dof_bc_DictList[l+1][cj]
                fineSpaceInterpolationFunctionals = fineSpace.referenceFiniteElement.interpolationConditions.functionalsQuadrature
                #interpolationPointsOnFineElement_reference = fineSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray
                nInterpolationPoints = fineSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints
                range_nInterpolationPoints = list(range(nInterpolationPoints))
                referenceElement = fineSpace.referenceFiniteElement.referenceElement
                rbcSum = Vec(coarseSpace.dim)
                rbcColumnIndeces=[set() for row in range(coarseSpace.dim)]
                rbc = {}
                pbc = {}
                scaled_rbc = {}
                interp_bc = {}
                #mwf should tell us if interpolation condition is at coarse element interface or not
                # aka (nonconforming on coarse grid?)
                #check typecode if go to c
                isNonConformingOnCoarseGrid = numpy.zeros((fineMesh.nElements_global,nInterpolationPoints),'i')
                if referenceElement.dim  > 1 and isinstance(fineSpace,NC_AffineLinearOnSimplexWithNodalBasis):
                    #mwf old mesh interface multilevelMesh.calculateElementParents()
                    parentsArray = multilevelMesh.elementParentsArrayList[l+1]
                    usingAtLeastOneNCproj = True
                    #need to care about interpolation conditions being conforming on coarse grid or not
                    for ebNI in range(fineMesh.nInteriorElementBoundaries_global):
                        ebN = fineMesh.interiorElementBoundariesArray[ebNI]
                        eN_left  = fineMesh.elementBoundaryElementsArray[ebN,0]
                        eN_right = fineMesh.elementBoundaryElementsArray[ebN,1]
                        ebn_left = fineMesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                        ebn_right= fineMesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
                        #mwf old interface
                        #PeN_left = multilevelMesh.elementParents[l+1][eN_left]
                        #PeN_right= multilevelMesh.elementParents[l+1][eN_right]
                        PeN_left = parentsArray[eN_left]
                        PeN_right= parentsArray[eN_right]

                        if PeN_left != PeN_right:
                            #assumes a unique correspondence interpCondition <--> local element boundary
                            isNonConformingOnCoarseGrid[eN_left,ebn_left]  = 1
                            isNonConformingOnCoarseGrid[eN_right,ebn_right]= 1
                            #mwf debug
                            #print """MultilevelProjNC nc IP ebN= %d eN_left=%d PeN_left=%d eN_right=%d PeN_right=%d """ % (ebN,
                            #                                                                                               eN_left,
                            #                                                                                               PeN_left,
                            #                                                                                               eN_right,
                            #                                                                                               PeN_right)
                        #different parents
                    #interior element boundaries
                    #what about exterior ones?
                    for ebNE in range(fineMesh.nExteriorElementBoundaries_global):
                        ebN = fineMesh.exteriorElementBoundariesArray[ebNE]
                        eN_left = fineMesh.elementBoundaryElementsArray[ebN,0]
                        ebn_left = fineMesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                        isNonConformingOnCoarseGrid[eN_left,ebn_left]  = -1
                    #end exterior
                #need to care about nonconforming points
                if (not isinstance(fineSpace,NC_AffineLinearOnSimplexWithNodalBasis) and
                    referenceElement.dim > 1 and usingAtLeastOneNCproj):
                    logEvent("""WARNING testStuff.MultiLevelProjectionOperatorsNC mixing NC and C0 projection operators!!!""")
                #map reference interpolation points of fine elements to physical space
                interpolationPointsOnFineElement_physical =  fineSpace.updateInterpolationPoints()
                #copy physical space reference points on fine elements to an array for their parents
                interpolationPointsOnCoarseElement_physical = numpy.zeros((coarseMesh.nElements_global,
                                                                             nChildrenMax_element*nInterpolationPoints,
                                                                             3),
                                                                            'd')
                child_N_global=0 #cek new mesh interface, keep track of position in children array
                for coarse_eN in range(coarseMesh.nElements_global):
                    #cek old mesh interface
                    #for child_N,fine_e in enumerate(children[coarse_eN]):
                    #new mesh interface
                    for child_N,offset in enumerate(range(childrenOffsets[coarse_eN],childrenOffsets[coarse_eN+1])):
                        fine_eN = children[offset] #new
                        for pN in range_nInterpolationPoints:
                            #cek old mesh interface
                            #interpolationPointsOnCoarseElement_physical[coarse_eN,child_N*nInterpolationPoints + pN,:] = interpolationPointsOnFineElement_physical[fine_e.N,pN]
                            interpolationPointsOnCoarseElement_physical[coarse_eN,child_N*nInterpolationPoints + pN,:] = interpolationPointsOnFineElement_physical[fine_eN,pN]
                #map physical interpolation points on coarse elements to coarse reference coordinates
                interpolationPointsOnCoarseElement_reference = numpy.zeros((coarseMesh.nElements_global,
                                                                              nChildrenMax_element*nInterpolationPoints,
                                                                              3),
                                                                             'd')
                J = numpy.zeros((coarseMesh.nElements_global,
                                   nChildrenMax_element*nInterpolationPoints,
                                   referenceElement.dim,
                                   referenceElement.dim),
                                  'd')
                invJ = numpy.zeros((coarseMesh.nElements_global,
                                      nChildrenMax_element*nInterpolationPoints,
                                      referenceElement.dim,
                                      referenceElement.dim),
                                     'd')
                detJ = numpy.zeros((coarseMesh.nElements_global,
                                      nChildrenMax_element*nInterpolationPoints),
                                     'd')
                interpolationPointsOnCoarseElement_reference_dummy = numpy.zeros((nChildrenMax_element*nInterpolationPoints,
                                                                                    3),
                                                                                  'd')
                coarseSpace.elementMaps.getJacobianValues(interpolationPointsOnCoarseElement_reference_dummy,
                                                          J,
                                                          invJ,
                                                          detJ)
                coarseSpace.elementMaps.getInverseValues(invJ,
                                                         interpolationPointsOnCoarseElement_physical,
                                                         interpolationPointsOnCoarseElement_reference)
                #get coarse scale basis function values at these reference points
                psi = numpy.zeros((coarseMesh.nElements_global,
                                     nChildrenMax_element*nInterpolationPoints,
                                     coarseSpace.referenceFiniteElement.localFunctionSpace.dim),
                                    'd')
                coarseSpace.getBasisValuesAtArray(interpolationPointsOnCoarseElement_reference,
                                                  psi)
                #cek new mesh interface
                child_N_global=0
                for coarse_eN in range(coarseMesh.nElements_global):
                    #cek old mesh interface
                    #for fine_eN,fine_e in enumerate(children[coarse_eN]):
                    #cek new mesh interface
                    for fine_eN,offset in enumerate(range(childrenOffsets[coarse_eN],childrenOffsets[coarse_eN+1])):
                        fine_eN_global = children[offset] #new
                        for i in fineSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                            #I = fineSpace.dofMap.l2g[fine_e.N,i]
                            I = fineSpace.dofMap.l2g[fine_eN_global,i]
                            for j in coarseSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                                J = coarseSpace.dofMap.l2g[coarse_eN,j]
                                psi_j = psi[coarse_eN,fine_eN*nInterpolationPoints:(fine_eN+1)*nInterpolationPoints,j]
                                F_ij = fineSpaceInterpolationFunctionals[i](psi_j)
                                if abs(F_ij ) > 1.0e-16:
                                    #mwf orig F_ij > 1.e-16 changed to abs(F_ij)
                                    #mwf now account for nonconforming points?
                                    #cek old mesh interface
                                    #if usingAtLeastOneNCproj and isNonConformingOnCoarseGrid[fine_e.N,i] > 0:
                                    #cek new mesh interface
                                    if usingAtLeastOneNCproj and isNonConformingOnCoarseGrid[fine_eN_global,i] > 0:
                                        F_ij *= 0.5
                                        #have to allow for summing up values hit multiple times
                                        #only here
                                        if (J,I) in rbc:
                                            rbc[(J,I)] += F_ij
                                            pbc[(I,J)] += F_ij
                                        else:
                                            rbc[(J,I)] = F_ij
                                            pbc[(I,J)] = F_ij
                                    else:
                                        rbc[(J,I)] = F_ij
                                        pbc[(I,J)] = F_ij
                                    rbcColumnIndeces[J].add(I)
                                    #check why this is being called, todo
                                    if I in fineDOFBoundaryConditions.global2freeGlobal:
                                        II = fineDOFBoundaryConditions.global2freeGlobal[I]*strideListList[l+1][cj]+offsetListList[l+1][cj]
                                        if J in coarseDOFBoundaryConditions.global2freeGlobal:
                                            JJ = coarseDOFBoundaryConditions.global2freeGlobal[J]*strideListList[l][cj]+offsetListList[l][cj]
                                            rColumnIndeces[JJ].add(II)
                                            r[(JJ,II)] = rbc[(J,I)]
                                            p[(II,JJ)] = pbc[(I,J)]
                #end coarse_eN
                for I in range(coarseSpace.dim):
                    for J in rbcColumnIndeces[I]:
                        rbcSum[I] += abs(rbc[(I,J)])
                        if rbc[(I,J)] > 1.0-1.0e-8 and rbc[(I,J)] < 1.0 + 1.0e-8:
                            interp_bc[(I,J)] = 1.0
                for I in range(offsetListList[l][cj],offsetListList[l][cj]+coarseDOFBoundaryConditions.nFreeDOF_global,strideListList[l][cj]):
                    for J in rColumnIndeces[I]:
                        rSum[I] += r[(I,J)]
                for I in range(coarseSpace.dim):
                    for J in rbcColumnIndeces[I]:
                        if rbc[(I,J)] > 0.0 - 1.0e-8 and rbc[(I,J)] < 0.0 + 1.0e-8:
                            scaled_rbc[(I,J)] = 0.0
                        else:
                            scaled_rbc[(I,J)] = rbc[(I,J)]/rbcSum[I]
                #now make real sparse matrices
                (rbc,rbczval) = SparseMatFromDict(coarseSpace.dim,fineSpace.dim,rbc)
                (scaled_rbc,scaled_rbczval) = SparseMatFromDict(coarseSpace.dim,fineSpace.dim,scaled_rbc)
                (pbc,pbczval) = SparseMatFromDict(fineSpace.dim,coarseSpace.dim,pbc)
                (interp_bc,interp_bczval) =SparseMatFromDict(coarseSpace.dim,fineSpace.dim,interp_bc)
                self.restrict_bcSumListDict[cj].append(rbcSum)
                self.restrict_bcListDict[cj].append(rbc)
                self.scaled_restrict_bcListDict[cj].append(scaled_rbc)
                self.interp_bcListDict[cj].append(interp_bc)
                self.rbczvalListDict[cj].append(rbczval)
                self.scaled_rbczvalListDict[cj].append(scaled_rbczval)
                self.interp_bczvalListDict[cj].append(interp_bczval)
                self.prolong_bcListDict[cj].append(pbc)
                self.pbczvalListDict[cj].append(pbczval)
            (r,rzval) = SparseMatFromDict(coarse_nFreeDOF_global,fine_nFreeDOF_global,r)
            (p,pzval) = SparseMatFromDict(fine_nFreeDOF_global,coarse_nFreeDOF_global,p)
            self.restrictSumList.append(rSum)
            self.restrictList.append(r)
            self.rzvalList.append(rzval)
            self.prolongList.append(p)
            self.pzvalList.append(pzval)

## @} */

if __name__ == '__main__':
    nq1Db=1
    xiArray0D = numpy.zeros((nq1Db,1), 'd')
    xiArray0D[0] = 0.0
    nq1D=3
    xiArray1D = numpy.zeros((nq1D,1), 'd')
    xiArray1D[0] = 0.0
    xiArray1D[1] = 1.0
    xiArray1D[2] = 0.5
    nq2D = 4
    nq2Db = nq1D
    xiArray2D = numpy.zeros((nq2D,2), 'd')
    xiArray2D[0,:] = [0.0,0.0]
    xiArray2D[1,:] = [1.0,0.0]
    xiArray2D[2,:] = [0.0,1.0]
    xiArray2D[3,:] = [0.33,0.33]
    nq3Db = nq2D
    nq3D = 5
    xiArray3D = numpy.zeros((nq3D,3), 'd')
    xiArray3D[0,:] = [0.0,0.0,0.0]
    xiArray3D[1,:] = [1.0,0.0,0.0]
    xiArray3D[2,:] = [0.0,1.0,0.0]
    xiArray3D[3,:] = [0.0,0.0,1.0]
    xiArray3D[4,:] = [0.25,0.25,0.25]
    print("****************************************ReferenceSimplex(1)****************************************")
    unitInterval = ReferenceSimplex(1)
    print("dim = "+repr(unitInterval.dim))
    print("nNodes = "+repr(unitInterval.nNodes))
    print("nElementBoundaries = "+repr(unitInterval.nElementBoundaries))
    for nN in unitInterval.range_nNodes:
        print("nN = "+repr(nN)+" : "+repr(unitInterval.nodeList[nN]))
    for ebN in unitInterval.range_nElementBoundaries:
        print("ebN = "+repr(ebN))
        print("JHat = "+repr(unitInterval.boundaryJacobianList[ebN]))
        print("nHat = "+repr(unitInterval.boundaryUnitNormalList[ebN]))
        for k in range(nq1Db):
            print("xBar = "+repr(xiArray0D[k]))
            print("xHat(xBar) = "+repr(unitInterval.boundaryMapList[ebN](xiArray0D[k])))
    print("****************************************LinearOnSimplexWithNodalBasis(1)****************************************")
    localFunctionSpace1D = LinearOnSimplexWithNodalBasis(1)
    print("dim = "+repr(localFunctionSpace1D.dim))
    print("v and grad(v)")
    for j in localFunctionSpace1D.range_dim:
        print("j = "+repr(j))
        for k in range(nq1D):
            print("xHat = "+repr(xiArray1D[k]))
            print("v = "+repr(localFunctionSpace1D.basis[j](xiArray1D[k])))
            print("grad(v) = "+repr(localFunctionSpace1D.basisGradients[j](xiArray1D[k])))
    print("trace(v) and trace(grad(v))")
    for j in localFunctionSpace1D.range_dim:
        print("j = "+repr(j))
        for ebN in localFunctionSpace1D.referenceElement.range_nElementBoundaries:
            print("ebN = "+repr(ebN))
            for k in range(nq1Db):
                print("xBar = "+repr(xiArray0D[k]))
                print("trace(v) = "+repr(localFunctionSpace1D.basisTrace[ebN][j](xiArray0D[k])))
                print("trace(grad(v)) = "+repr(localFunctionSpace1D.basisGradientsTrace[ebN][j](xiArray0D[k])))
    print("****************************************NodalInterpolationConditions(unitInterval)****************************************")
    interpolationConditions = NodalInterpolationConditions(unitInterval)
    for j in localFunctionSpace1D.range_dim:
        for i in localFunctionSpace1D.range_dim:
            print("F(i="+repr(i)+",v(j="+repr(j)+"))="+repr(interpolationConditions.functionals[i](localFunctionSpace1D.basis[j])))
    fListList = [[1.0,0.0],[0.0,1.0]]
    print("fListList = "+repr(fListList))
    for i in localFunctionSpace1D.range_dim:
        for fList in fListList:
            print("F(i="+repr(i)+",fList="+repr(fList)+")="+repr(interpolationConditions.functionalsQuadrature[i](fList)))
    print("****************************************ReferenceFiniteElement*********************************************")
    referenceFiniteElement = ReferenceFiniteElement(localFunctionSpace1D,interpolationConditions)
    print("****************************************ReferenceSimplex(2)****************************************")
    unitTriangle = ReferenceSimplex(2)
    print("dim = "+repr(unitTriangle.dim))
    print("nNodes = "+repr(unitTriangle.nNodes))
    print("nElementBoundaries = "+repr(unitTriangle.nElementBoundaries))
    for nN in unitTriangle.range_nNodes:
        print("nN = "+repr(nN)+" : "+repr(unitTriangle.nodeList[nN]))
    for ebN in unitTriangle.range_nElementBoundaries:
        print("ebN = "+repr(ebN))
        print("JHat = "+repr(unitTriangle.boundaryJacobianList[ebN]))
        print("nHat = "+repr(unitTriangle.boundaryUnitNormalList[ebN]))
        for k in range(nq2Db):
            print("xBar = "+repr(xiArray1D[k]))
            print("xHat(xBar) = "+repr(unitTriangle.boundaryMapList[ebN](xiArray1D[k])))
    print("****************************************LinearOnSimplexWithNodalBasis(2)****************************************")
    localFunctionSpace2D = LinearOnSimplexWithNodalBasis(2)
    print("dim = "+repr(localFunctionSpace2D.dim))
    print("v and grad(v)")
    for j in localFunctionSpace2D.range_dim:
        print("j = "+repr(j))
        for k in range(nq2D):
            print("xHat = "+repr(xiArray2D[k]))
            print("v = "+repr(localFunctionSpace2D.basis[j](xiArray2D[k])))
            print("grad(v) = "+repr(localFunctionSpace2D.basisGradients[j](xiArray2D[k])))
    print("trace(v) and trace(grad(v))")
    for j in localFunctionSpace2D.range_dim:
        print("j = "+repr(j))
        for ebN in localFunctionSpace2D.referenceElement.range_nElementBoundaries:
            print("ebN = "+repr(ebN))
            for k in range(nq2Db):
                print("xBar = "+repr(xiArray1D[k]))
                print("trace(v) = "+repr(localFunctionSpace2D.basisTrace[ebN][j](xiArray1D[k])))
                print("trace(grad(v)) = "+repr(localFunctionSpace2D.basisGradientsTrace[ebN][j](xiArray1D[k])))
    print("****************************************NodalInterpolationConditions(unitTriangle)****************************************")
    interpolationConditions = NodalInterpolationConditions(unitTriangle)
    for i in localFunctionSpace2D.range_dim:
        for j in localFunctionSpace2D.range_dim:
            print("F(i="+repr(i)+",j="+repr(j)+")="+repr(interpolationConditions.functionals[i](localFunctionSpace2D.basis[j])))
    fListList = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    print(fListList)
    for i in localFunctionSpace2D.range_dim:
        for fList in fListList:
            print("F(i="+repr(i)+",fList="+repr(fList)+")="+repr(interpolationConditions.functionalsQuadrature[i](fList)))
    print("****************************************ReferenceFiniteElement*********************************************")
    referenceFiniteElement = ReferenceFiniteElement(localFunctionSpace2D,interpolationConditions)
    print("****************************************ReferenceSimplex(3)****************************************")
    unitTetrahedron = ReferenceSimplex(3)
    print("dim = "+repr(unitTetrahedron.dim))
    print("nNodes = "+repr(unitTetrahedron.nNodes))
    print("nElementBoundaries = "+repr(unitTetrahedron.nElementBoundaries))
    for nN in unitTetrahedron.range_nNodes:
        print("nN = "+repr(nN)+" : "+repr(unitTetrahedron.nodeList[nN]))
    print("boundary maps")
    for ebN in unitTetrahedron.range_nElementBoundaries:
        print("ebN = "+repr(ebN))
        print("JHat = "+repr(unitTetrahedron.boundaryJacobianList[ebN]))
        print("nHat = "+repr(unitTetrahedron.boundaryUnitNormalList[ebN]))
        for k in range(nq3Db):
            print("xBar = "+repr(xiArray2D[k]))
            print("xHat(xBar) = "+repr(unitTetrahedron.boundaryMapList[ebN](xiArray2D[k])))
    print("****************************************LinearOnSimplexWithNodalBasis(3)****************************************")
    localFunctionSpace3D = LinearOnSimplexWithNodalBasis(3)
    print("dim = "+repr(localFunctionSpace3D.dim))
    print("v and grad(v)")
    for j in localFunctionSpace3D.range_dim:
        print("j = "+repr(j))
        for k in range(nq3D):
            print("xHat = "+repr(xiArray3D[k]))
            print("v = "+repr(localFunctionSpace3D.basis[j](xiArray3D[k])))
            print("grad(v) = "+repr(localFunctionSpace3D.basisGradients[j](xiArray3D[k])))
    print("trace(v) and trace(grad(v))")
    for j in localFunctionSpace3D.range_dim:
        print("j = "+repr(j))
        for ebN in localFunctionSpace3D.referenceElement.range_nElementBoundaries:
            print("ebN = "+repr(ebN))
            for k in range(nq3Db):
                print("xBar = "+repr(xiArray2D[k]))
                print("trace(v) = "+repr(localFunctionSpace3D.basisTrace[ebN][j](xiArray2D[k])))
                print("trace(grad(v))"+repr(localFunctionSpace3D.basisGradientsTrace[ebN][j](xiArray2D[k])))
    print("****************************************NodalInterpolationConditions(unitTetrahedron)****************************************")
    interpolationConditions = NodalInterpolationConditions(unitTetrahedron)
    for i in localFunctionSpace3D.range_dim:
        for j in localFunctionSpace3D.range_dim:
            print("F(i = "+repr(i)+", j = "+repr(j)+") = "+repr(interpolationConditions.functionals[i](localFunctionSpace3D.basis[j])))
    fListList = [[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]]
    for i in localFunctionSpace3D.range_dim:
        for fList in fListList:
            print(fList)
            print("F_i(fList) = "+repr(interpolationConditions.functionalsQuadrature[i](fList)))
    print("****************************************ReferenceFiniteElement*********************************************")
    referenceFiniteElement = ReferenceFiniteElement(localFunctionSpace3D,interpolationConditions)
    print("****************************************NodalDOFMap-1D************************************************************")
    nnDim = 3
    grid1d = RectangularGrid(nnDim,1,1,1.0,1.0,1.0)
    mesh1d = EdgeMesh()
    mesh1d.rectangularToEdge(grid1d)
    mesh1d.writeMeshEnsight("mesh1d","1D test mesh")
    dofMap = NodalDOFMap(mesh1d)
    print("nDOF")
    print(dofMap.nDOF)
    for eN in range(mesh1d.nElements_global):
        for nN in range(mesh1d.nNodes_element):
            print("l2g(eN="+repr(eN)+",nN_element="+repr(nN)+")="+repr(dofMap.l2g[eN,nN]))
    print("****************************************NodalDOFMap-2D************************************************************")
    grid2d = RectangularGrid(nnDim,nnDim,1,1.0,1.0,1.0)
    mesh2d = TriangularMesh()
    mesh2d.rectangularToTriangular(grid2d)
    mesh2d.writeMeshEnsight("mesh2d","2D test mesh")
    dofMap = NodalDOFMap(mesh2d)
    print("nDOF")
    print(dofMap.nDOF)
    for eN in range(mesh2d.nElements_global):
        for nN in range(mesh2d.nNodes_element):
            print("l2g(eN="+repr(eN)+",nN_element="+repr(nN)+")="+repr(dofMap.l2g[eN,nN]))
    print("****************************************NodalDOFMap-3D************************************************************")
    grid3d = RectangularGrid(nnDim,nnDim,nnDim,1.0,1.0,1.0)
    mesh3d = TetrahedralMesh()
    mesh3d.rectangularToTetrahedral(grid3d)
    mesh3d.writeMeshEnsight("mesh3d","3D test mesh")
    dofMap = NodalDOFMap(mesh3d)
    print("nDOF")
    print(dofMap.nDOF)
    for eN in range(mesh3d.nElements_global):
        for nN in range(mesh3d.nNodes_element):
            print("l2g(eN="+repr(eN)+",nN_element="+repr(nN)+")="+repr(dofMap.l2g[eN,nN]))
    print("**************************************AffineMaps-1D**************************************************************")
    elementMaps1D = AffineMaps(mesh1d,unitInterval,localFunctionSpace1D)
    xArray1D = numpy.zeros((mesh1d.nElements_global,nq1D,1),'d')
    xiArray1DNew = numpy.zeros((mesh1d.nElements_global,nq1D,1),'d')
    jacobianArray1D = numpy.zeros((mesh1d.nElements_global,nq1D,1,1),
                                  'd')
    jacobianInverseArray1D = numpy.zeros((mesh1d.nElements_global,nq1D,1,1),
                                         'd')
    jacobianDeterminantArray1D = numpy.zeros((mesh1d.nElements_global,nq1D),
                                             'd')
    elementMaps1D.getValues(xiArray1D,
                            xArray1D)
    elementMaps1D.getJacobianValues(xiArray1D,
                                    jacobianArray1D,
                                    jacobianInverseArray1D,
                                    jacobianDeterminantArray1D)
    for eN in range(mesh1d.nElements_global):
        for k in range(nq1D):
            print("hat(x)")
            print(xiArray1D[k])
            print("x")
            print(xArray1D[eN,k])
            print("jacobian")
            print(jacobianArray1D[eN,k])
            print("jacobianInverse")
            print(jacobianInverseArray1D[eN,k])
            print("determinant")
            print(jacobianDeterminantArray1D[eN,k])
    print("inverse map")
    elementMaps1D.getInverseValues(jacobianInverseArray1D,xArray1D,xiArray1DNew)
    for eN in range(mesh1d.nElements_global):
        for k in range(nq1D):
            print("hat(x)")
            print(xiArray1D[k])
            print("x")
            print(xArray1D[eN,k])
            print("hat(x)(x)")
            print(xiArray1DNew[eN,k])
    xArray1DTrace = numpy.zeros((mesh1d.nElements_global,elementMaps1D.referenceElement.nElementBoundaries,nq1Db,elementMaps1D.referenceElement.dim),'d')
    #the metric tensor only makes sense in dim > 1 so we use dim here but dim-1 for 2 and 3d
    jacobianInverseArray1DTrace = numpy.zeros((mesh1d.nElements_global,
                                                 elementMaps1D.referenceElement.nElementBoundaries,
                                                 nq1Db,
                                                 elementMaps1D.referenceElement.dim,
                                                 elementMaps1D.referenceElement.dim),
                                           'd')
    metricTensorArray1D = numpy.zeros((mesh1d.nElements_global,elementMaps1D.referenceElement.nElementBoundaries,nq1Db,elementMaps1D.referenceElement.dim,elementMaps1D.referenceElement.dim),
                                  'd')
    metricTensorDeterminantSqrtArray1D = numpy.zeros((mesh1d.nElements_global,elementMaps1D.referenceElement.nElementBoundaries,nq1Db),
                                             'd')
    unitNormalArray1D = numpy.zeros((mesh1d.nElements_global,elementMaps1D.referenceElement.nElementBoundaries,nq1Db,elementMaps1D.referenceElement.dim),
                                    'd')
    elementMaps1D.getValuesTrace(xiArray0D,
                                 xArray1DTrace)
    elementMaps1D.getJacobianValuesTrace(xiArray0D,
                                         jacobianInverseArray1DTrace,
                                         metricTensorArray1D,
                                         metricTensorDeterminantSqrtArray1D,
                                         unitNormalArray1D)
    for eN in range(mesh1d.nElements_global):
        print("eN")
        print(eN)
        for ebN in elementMaps1D.referenceElement.range_nElementBoundaries:
            print("ebN")
            print(ebN)
            for k in range(nq1Db):
                print("x")
                print(xArray1DTrace[eN,ebN,k])
                print("jacobian inverse")
                print(jacobianInverseArray1DTrace[eN,ebN,k])
                print("metric tensor")
                print(metricTensorArray1D[eN,ebN,k])
                print("metric tensor determinant sqrt")
                print(metricTensorDeterminantSqrtArray1D[eN,ebN,k])
                print("unit normal")
                print(unitNormalArray1D[eN,ebN,k])
                print("metric tensor")
                print(metricTensorArray1D[eN,ebN,k])
                print("metric tensor determinant sqrt")
                print(metricTensorDeterminantSqrtArray1D[eN,ebN,k])
                print("unit normal")
                print(unitNormalArray1D[eN,ebN,k])
    caseOut=open('mesh1d.case','a')
    caseOut.write('measured: mesh1dFaceQuadrature.geo\n')
    caseOut.write('VARIABLE\n')
    caseOut.write('vector per measured node: unitNormal unitNormal1d.vec\n')
    caseOut.close()
    quadratureOut=open('mesh1dFaceQuadrature.geo','w')
    quadratureOut.write('quadrature points\n'+'particle coordinates\n')
    quadratureOut.write('%8i\n' % (mesh1d.nElements_global*elementMaps1D.referenceElement.nElementBoundaries*nq1Db,))
    normalOut = open('unitNormal1d.vec','w')
    normalOut.write('element boundary unit normal\n')
    nColumns = 0
    for eN in range(mesh1d.nElements_global):
        for ebN in elementMaps1D.referenceElement.range_nElementBoundaries:
            for k in range(nq1Db):
                qN = eN*elementMaps1D.referenceElement.nElementBoundaries*nq1Db+ebN*nq1Db+k +1
                quadratureOut.write('%8i%12.5e%12.5e%12.5e\n' % (qN,xArray1DTrace[eN,ebN,k,0],0.0,0.0))
                normalOut.write('%12.5e%12.5e%12.5e' %(unitNormalArray1D[eN,ebN,k,0],0.0,0.0))
                nColumns+=3
                if nColumns == 6:
                    nColumns = 0
                    normalOut.write('\n')
    normalOut.write('\n')
    normalOut.close()
    quadratureOut.close()
    print("**************************************AffineMaps-2D**************************************************************")
    elementMaps2D = AffineMaps(mesh2d,unitTriangle,localFunctionSpace2D)
    xArray2D = numpy.zeros((mesh2d.nElements_global,
                            nq2D,
                            elementMaps2D.referenceElement.dim),
                           'd')
    xiArray2DNew = numpy.zeros((mesh2d.nElements_global,
                            nq2D,
                            elementMaps2D.referenceElement.dim),
                           'd')
    jacobianArray2D = numpy.zeros((mesh2d.nElements_global,
                                   nq2D,
                                   elementMaps2D.referenceElement.dim,
                                   elementMaps2D.referenceElement.dim),
                                  'd')
    jacobianInverseArray2D = numpy.zeros((mesh2d.nElements_global,
                                            nq2D,
                                            elementMaps2D.referenceElement.dim,
                                            elementMaps2D.referenceElement.dim),
                                           'd')
    jacobianDeterminantArray2D = numpy.zeros((mesh2d.nElements_global,nq2D),
                                             'd')
    elementMaps2D.getValues(xiArray2D,
                            xArray2D)
    elementMaps2D.getJacobianValues(xiArray2D,
                                    jacobianArray2D,
                                    jacobianInverseArray2D,
                                    jacobianDeterminantArray2D)
    for eN in range(mesh2d.nElements_global):
        for k in range(nq2D):
            print("x")
            print(xArray2D[eN,k])
            print("jacobian")
            print(jacobianArray2D[eN,k])
            print("jacobianInverse")
            print(jacobianInverseArray2D[eN,k])
            print("determinant")
            print(jacobianDeterminantArray2D[eN,k])
    print("inverse map")
    elementMaps2D.getInverseValues(jacobianInverseArray2D,xArray2D,xiArray2DNew)
    for eN in range(mesh2d.nElements_global):
        for k in range(nq2D):
            print("hat(x)")
            print(xiArray2D[k])
            print("x")
            print(xArray2D[eN,k])
            print("hat(x)(x)")
            print(xiArray2DNew[eN,k])
    xArray2DTrace = numpy.zeros((mesh2d.nElements_global,elementMaps2D.referenceElement.nElementBoundaries,nq2Db,elementMaps2D.referenceElement.dim),'d')
    #the metric tensor only makes sense in dim > 1 so we use dim here but dim-1 for 2 and 3d
    jacobianInverseArray2DTrace = numpy.zeros((mesh2d.nElements_global,
                                                 elementMaps2D.referenceElement.nElementBoundaries,
                                                 nq2Db,
                                                 elementMaps2D.referenceElement.dim,
                                                 elementMaps2D.referenceElement.dim),
                                                'd')
    metricTensorArray2D = numpy.zeros((mesh2d.nElements_global,elementMaps2D.referenceElement.nElementBoundaries,nq2Db,elementMaps2D.referenceElement.dim-1,elementMaps2D.referenceElement.dim-1),
                                  'd')
    metricTensorDeterminantSqrtArray2D = numpy.zeros((mesh2d.nElements_global,elementMaps2D.referenceElement.nElementBoundaries,nq2Db),
                                             'd')
    unitNormalArray2D = numpy.zeros((mesh2d.nElements_global,elementMaps2D.referenceElement.nElementBoundaries,nq2Db,elementMaps2D.referenceElement.dim),
                                    'd')
    elementMaps2D.getValuesTrace(xiArray1D,
                                 xArray2DTrace)
    elementMaps2D.getJacobianValuesTrace(xiArray1D,
                                         jacobianInverseArray2DTrace,
                                         metricTensorArray2D,
                                         metricTensorDeterminantSqrtArray2D,
                                         unitNormalArray2D)
    for eN in range(mesh2d.nElements_global):
        print("eN")
        print(eN)
        for ebN in elementMaps2D.referenceElement.range_nElementBoundaries:
            print("ebN")
            print(ebN)
            for k in range(nq2Db):
                print("x")
                print(xArray2DTrace[eN,ebN,k])
                print("jacobian inverse")
                print(jacobianInverseArray2DTrace[eN,ebN,k])
                print("metric tensor")
                print(metricTensorArray2D[eN,ebN,k])
                print("metric tensor determinant sqrt")
                print(metricTensorDeterminantSqrtArray2D[eN,ebN,k])
                print("unit normal")
                print(unitNormalArray2D[eN,ebN,k])
                print("metric tensor")
                print(metricTensorArray2D[eN,ebN,k])
                print("metric tensor determinant sqrt")
                print(metricTensorDeterminantSqrtArray2D[eN,ebN,k])
                print("unit normal")
                print(unitNormalArray2D[eN,ebN,k])
    caseOut=open('mesh2d.case','a')
    caseOut.write('measured: mesh2dFaceQuadrature.geo\n')
    caseOut.write('VARIABLE\n')
    caseOut.write('vector per measured node: unitNormal unitNormal2d.vec\n')
    caseOut.close()
    quadratureOut=open('mesh2dFaceQuadrature.geo','w')
    quadratureOut.write('quadrature points\n'+'particle coordinates\n')
    quadratureOut.write('%8i\n' % (mesh2d.nElements_global*elementMaps2D.referenceElement.nElementBoundaries*nq2Db,))
    normalOut = open('unitNormal2d.vec','w')
    normalOut.write('element boundary unit normal\n')
    nColumns = 0
    for eN in range(mesh2d.nElements_global):
        for ebN in elementMaps2D.referenceElement.range_nElementBoundaries:
            for k in range(nq2Db):
                qN = eN*elementMaps2D.referenceElement.nElementBoundaries*nq2Db+ebN*nq2Db+k +1
                quadratureOut.write('%8i%12.5e%12.5e%12.5e\n' % (qN,xArray2DTrace[eN,ebN,k,0],xArray2DTrace[eN,ebN,k,1],0.0))
                normalOut.write('%12.5e%12.5e%12.5e' %(unitNormalArray2D[eN,ebN,k,0],unitNormalArray2D[eN,ebN,k,1],0.0))
                nColumns+=3
                if nColumns == 6:
                    nColumns = 0
                    normalOut.write('\n')
    normalOut.write('\n')
    normalOut.close()
    quadratureOut.close()
    print("**************************************AffineMaps-3D**************************************************************")
    elementMaps3D = AffineMaps(mesh3d,unitTetrahedron,localFunctionSpace3D)
    xArray3D = numpy.zeros((mesh3d.nElements_global,
                            nq3D,
                            elementMaps3D.referenceElement.dim),
                           'd')
    xiArray3DNew = numpy.zeros((mesh3d.nElements_global,
                                nq3D,
                                elementMaps3D.referenceElement.dim),
                           'd')
    jacobianArray3D = numpy.zeros((mesh3d.nElements_global,
                                   nq3D,
                                   elementMaps3D.referenceElement.dim,
                                   elementMaps3D.referenceElement.dim),
                                  'd')
    jacobianInverseArray3D = numpy.zeros((mesh3d.nElements_global,
                                          nq3D,
                                          elementMaps3D.referenceElement.dim,
                                          elementMaps3D.referenceElement.dim),
                                         'd')
    jacobianDeterminantArray3D = numpy.zeros((mesh3d.nElements_global,nq3D),
                                             'd')
    elementMaps3D.getValues(xiArray3D,
                          xArray3D)
    elementMaps3D.getJacobianValues(xiArray3D,
                                    jacobianArray3D,
                                    jacobianInverseArray3D,
                                    jacobianDeterminantArray3D)
    for eN in range(mesh3d.nElements_global):
        for k in range(nq3D):
            print("x")
            print(xArray3D[eN,k])
            print("jacobian")
            print(jacobianArray3D[eN,k])
            print("jacobianInverse")
            print(jacobianInverseArray3D[eN,k])
            print("determinant")
            print(jacobianDeterminantArray3D[eN,k])
    print("inverse map")
    elementMaps3D.getInverseValues(jacobianInverseArray3D,xArray3D,xiArray3DNew)
    for eN in range(mesh2d.nElements_global):
        for k in range(nq3D):
            print("hat(x)")
            print(xiArray3D[k])
            print("x")
            print(xArray3D[eN,k])
            print("hat(x)(x)")
            print(xiArray3DNew[eN,k])
    xArray3DTrace = numpy.zeros((mesh3d.nElements_global,elementMaps3D.referenceElement.nElementBoundaries,nq3Db,elementMaps3D.referenceElement.dim),'d')
    #the metric tensor only makes sense in dim > 1 so we use dim here but dim-1 for 2 and 3d
    jacobianInverseArray3DTrace = numpy.zeros((mesh3d.nElements_global,
                                                 elementMaps3D.referenceElement.nElementBoundaries,
                                                 nq3Db,
                                                 elementMaps3D.referenceElement.dim,
                                                 elementMaps3D.referenceElement.dim),
                                                'd')
    metricTensorArray3D = numpy.zeros((mesh3d.nElements_global,elementMaps3D.referenceElement.nElementBoundaries,nq3Db,elementMaps3D.referenceElement.dim-1,elementMaps3D.referenceElement.dim-1),
                                  'd')
    metricTensorDeterminantSqrtArray3D = numpy.zeros((mesh3d.nElements_global,elementMaps3D.referenceElement.nElementBoundaries,nq3Db),
                                             'd')
    unitNormalArray3D = numpy.zeros((mesh3d.nElements_global,elementMaps3D.referenceElement.nElementBoundaries,nq3Db,elementMaps3D.referenceElement.dim),
                                    'd')
    elementMaps3D.getValuesTrace(xiArray2D,
                                 xArray3DTrace)
    elementMaps3D.getJacobianValuesTrace(xiArray2D,
                                         jacobianInverseArray3DTrace,
                                         metricTensorArray3D,
                                         metricTensorDeterminantSqrtArray3D,
                                         unitNormalArray3D)
    for eN in range(mesh3d.nElements_global):
        print("eN")
        print(eN)
        for ebN in elementMaps3D.referenceElement.range_nElementBoundaries:
            print("ebN")
            print(ebN)
            for k in range(nq3Db):
                print("x")
                print(xArray3DTrace[eN,ebN,k])
                print("metric tensor")
                print(metricTensorArray3D[eN,ebN,k])
                print("metric tensor determinant sqrt")
                print(metricTensorDeterminantSqrtArray3D[eN,ebN,k])
                print("unit normal")
                print(unitNormalArray3D[eN,ebN,k])
                print("metric tensor")
                print(metricTensorArray3D[eN,ebN,k])
                print("metric tensor determinant sqrt")
                print(metricTensorDeterminantSqrtArray3D[eN,ebN,k])
                print("unit normal")
                print(unitNormalArray3D[eN,ebN,k])
    caseOut=open('mesh3d.case','a')
    caseOut.write('measured: mesh3dFaceQuadrature.geo\n')
    caseOut.write('VARIABLE\n')
    caseOut.write('vector per measured node: unitNormal unitNormal3d.vec\n')
    caseOut.close()
    quadratureOut=open('mesh3dFaceQuadrature.geo','w')
    quadratureOut.write('quadrature points\n'+'particle coordinates\n')
    quadratureOut.write('%8i\n' % (mesh3d.nElements_global*elementMaps3D.referenceElement.nElementBoundaries*nq3Db,))
    normalOut = open('unitNormal3d.vec','w')
    normalOut.write('element boundary unit normal\n')
    nColumns = 0
    for eN in range(mesh3d.nElements_global):
        for ebN in elementMaps3D.referenceElement.range_nElementBoundaries:
            for k in range(nq3Db):
                qN = eN*elementMaps3D.referenceElement.nElementBoundaries*nq3Db+ebN*nq3Db+k +1
                quadratureOut.write('%8i%12.5e%12.5e%12.5e\n' % (qN,xArray3DTrace[eN,ebN,k,0],xArray3DTrace[eN,ebN,k,1],xArray3DTrace[eN,ebN,k,2]))
                normalOut.write('%12.5e%12.5e%12.5e' %(unitNormalArray3D[eN,ebN,k,0],unitNormalArray3D[eN,ebN,k,1],unitNormalArray3D[eN,ebN,k,2]))
                nColumns+=3
                if nColumns == 6:
                    nColumns = 0
                    normalOut.write('\n')
    normalOut.write('\n')
    normalOut.close()
    quadratureOut.close()
    print("*********************************C0_AffineLinearOnSimplexWithNodalBasis-1D******************************************************")
    femSpace1D = C0_AffineLinearOnSimplexWithNodalBasis(mesh1d,1)
    vArray1D = numpy.zeros((mesh1d.nElements_global,nq1D,femSpace1D.referenceFiniteElement.localFunctionSpace.dim),
                             'd')
    grad_vArray1D = numpy.zeros((mesh1d.nElements_global,
                                   nq1D,
                                   femSpace1D.referenceFiniteElement.localFunctionSpace.dim,
                                   femSpace1D.referenceFiniteElement.referenceElement.dim),
                                  'd')
    femSpace1D.getBasisValues(xiArray1D,vArray1D)
    femSpace1D.getBasisGradientValues(xiArray1D,jacobianInverseArray1D,grad_vArray1D)
    for eN in range(mesh1d.nElements_global):
        print("eN")
        print(eN)
        for k in range(nq1D):
            print("k")
            print(k)
            print("xHat")
            print(xiArray1D[k])
            print("x")
            print(xArray1D[eN,k])
            for j in femSpace1D.referenceFiniteElement.localFunctionSpace.range_dim:
                print("j")
                print(j)
                print("v")
                print(vArray1D[eN,k,j])
                print("grad(v)")
                print(grad_vArray1D[eN,k,j])
    nq1Db = 1
    vArray1DTrace = numpy.zeros((mesh1d.nElements_global,
                                   femSpace1D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                   nq1Db,
                                   femSpace1D.referenceFiniteElement.localFunctionSpace.dim),
                                  'd')

    grad_vArray1DTrace = numpy.zeros((mesh1d.nElements_global,
                                        femSpace1D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                        nq1Db,
                                        femSpace1D.referenceFiniteElement.localFunctionSpace.dim,
                                        femSpace1D.referenceFiniteElement.referenceElement.dim),
                                       'd')
    femSpace1D.getBasisValuesTrace(xiArray0D,vArray1DTrace)
    femSpace1D.getBasisGradientValuesTrace(xiArray0D,jacobianInverseArray1DTrace,grad_vArray1DTrace)
    for eN in range(mesh1d.nElements_global):
        print("eN")
        print(eN)
        for ebN in femSpace1D.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            print("ebN")
            print(ebN)
            for k in range(nq1Db):
                print("k")
                print(k)
                print("xBar")
                print(xiArray0D[k])
                print("x")
                print(xArray1DTrace[eN,ebN,k])
                for j in femSpace1D.referenceFiniteElement.localFunctionSpace.range_dim:
                    print("j")
                    print(j)
                    print("v")
                    print(vArray1DTrace[eN,ebN,k,j])
                    print("grad(v)")
                    print(grad_vArray1DTrace[eN,ebN,k,j])
    print("*********************************C0_AffineLinearOnSimplexWithNodalBasis-2D******************************************************")
    femSpace2D = C0_AffineLinearOnSimplexWithNodalBasis(mesh2d,2)
    vArray2D = numpy.zeros((mesh2d.nElements_global,nq2D,femSpace2D.referenceFiniteElement.localFunctionSpace.dim),'d')
    grad_vArray2D = numpy.zeros((mesh2d.nElements_global,
                                   nq2D,
                                   femSpace2D.referenceFiniteElement.localFunctionSpace.dim,
                                   femSpace2D.referenceFiniteElement.referenceElement.dim),
                                  'd')
    femSpace2D.getBasisValues(xiArray2D,vArray2D)
    femSpace2D.getBasisGradientValues(xiArray2D,jacobianInverseArray2D,grad_vArray2D)
    for eN in range(mesh2d.nElements_global):
        print("eN")
        print(eN)
        for k in range(nq2D):
            print("k")
            print(k)
            print("xHat")
            print(xiArray2D[k])
            print("x")
            print(xArray2D[eN,k])
            for j in femSpace2D.referenceFiniteElement.localFunctionSpace.range_dim:
                print("j")
                print(j)
                print("v")
                print(vArray2D[eN,k,j])
                print("grad(v)")
                print(grad_vArray2D[eN,k,j])
    nq2Db = nq1D
    vArray2DTrace = numpy.zeros((mesh2d.nElements_global,
                                   femSpace2D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                   nq2Db,
                                   femSpace2D.referenceFiniteElement.localFunctionSpace.dim),
                                  'd')

    grad_vArray2DTrace = numpy.zeros((mesh2d.nElements_global,
                                        femSpace2D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                        nq2Db,
                                        femSpace2D.referenceFiniteElement.localFunctionSpace.dim,
                                        femSpace2D.referenceFiniteElement.referenceElement.dim),
                                       'd')
    femSpace2D.getBasisValuesTrace(xiArray1D,vArray2DTrace)
    femSpace2D.getBasisGradientValuesTrace(xiArray1D,jacobianInverseArray2DTrace,grad_vArray2DTrace)
    for eN in range(mesh2d.nElements_global):
        print("eN")
        print(eN)
        for ebN in femSpace2D.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            print("ebN")
            print(ebN)
            for k in range(nq2Db):
                print("k")
                print(k)
                print("xBar")
                print(xiArray1D[k])
                print("x")
                print(xArray2DTrace[eN,ebN,k])
                for j in femSpace2D.referenceFiniteElement.localFunctionSpace.range_dim:
                    print("j")
                    print(j)
                    print("v")
                    print(vArray2DTrace[eN,ebN,k,j])
                    print("grad(v)")
                    print(grad_vArray2DTrace[eN,ebN,k,j])
    print("*********************************C0_AffineLinearOnSimplexWithNodalBasis-3D******************************************************")
    femSpace3D = C0_AffineLinearOnSimplexWithNodalBasis(mesh3d,3)
    vArray3D = numpy.zeros((mesh3d.nElements_global,nq3D,femSpace3D.referenceFiniteElement.localFunctionSpace.dim),'d')
    grad_vArray3D = numpy.zeros((mesh3d.nElements_global,
                                   nq3D,
                                   femSpace3D.referenceFiniteElement.localFunctionSpace.dim,
                                   femSpace3D.referenceFiniteElement.referenceElement.dim),
                                  'd')
    femSpace3D.getBasisValues(xiArray3D,vArray3D)
    femSpace3D.getBasisGradientValues(xiArray3D,jacobianInverseArray3D,grad_vArray3D)
    for eN in range(mesh3d.nElements_global):
        print("eN")
        print(eN)
        for k in range(nq3D):
            print("k")
            print(k)
            print("xHat")
            print(xiArray3D[k])
            print("x")
            print(xArray3D[eN,k])
            for j in femSpace3D.referenceFiniteElement.localFunctionSpace.range_dim:
                print("j")
                print(j)
                print("v")
                print(vArray3D[eN,k,j])
                print("grad(v)")
                print(grad_vArray3D[eN,k,j])
    nq3Db = nq2D
    vArray3DTrace = numpy.zeros((mesh3d.nElements_global,
                                   femSpace3D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                   nq3Db,
                                   femSpace3D.referenceFiniteElement.localFunctionSpace.dim),'d')

    grad_vArray3DTrace = numpy.zeros((mesh3d.nElements_global,
                                        femSpace3D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                        nq3Db,
                                        femSpace3D.referenceFiniteElement.localFunctionSpace.dim,
                                        femSpace3D.referenceFiniteElement.referenceElement.dim),
                                       'd')
    femSpace3D.getBasisValuesTrace(xiArray2D,vArray3DTrace)
    femSpace3D.getBasisGradientValuesTrace(xiArray2D,jacobianInverseArray3DTrace,grad_vArray3DTrace)
    for eN in range(mesh3d.nElements_global):
        print("eN")
        print(eN)
        for ebN in femSpace3D.referenceFiniteElement.referenceElement.range_nElementBoundaries:
            print("ebN")
            print(ebN)
            for k in range(nq3Db):
                print("k")
                print(k)
                print("xBar")
                print(xiArray2D[k])
                print("x")
                print(xArray3DTrace[eN,ebN,k])
                for j in femSpace3D.referenceFiniteElement.localFunctionSpace.range_dim:
                    print("j")
                    print(j)
                    print("v")
                    print(vArray3DTrace[eN,ebN,k,j])
                    print("grad(v)")
                    print(grad_vArray3DTrace[eN,ebN,k,j])
    print("*************************************************FiniteElementFunction-1D-Scalar************************************************")
    u1ds = FiniteElementFunction(femSpace1D,name="dist0.5")
    for nN in range(mesh1d.nNodes_global):
        u1ds.dof[nN] = sqrt((mesh1d.nodeArray[nN,0]-0.5)**2)
    u1ds.writeFunctionEnsight("mesh1d",append=True)
    uArray1D = numpy.zeros((mesh1d.nElements_global,
                              nq1D),
                             'd')
    grad_uArray1D = numpy.zeros((mesh1d.nElements_global,
                                   nq1D,
                                   1,
                                   femSpace1D.referenceFiniteElement.referenceElement.dim),
                                  'd')
    u1ds.getValues(vArray1D,uArray1D)
    u1ds.getGradientValues(grad_vArray1D,grad_uArray1D)
    #read the values and gradients back into degrees of freedom and verify that they match the original
    vDOF1D = numpy.zeros((u1ds.femSpace.dim*u1ds.dim_dof),
                         'd')
    grad_vDOF1D = numpy.zeros((u1ds.femSpace.dim*u1ds.femSpace.referenceFiniteElement.referenceElement.dim),
                              'd')
    for eN in range(u1ds.femSpace.elementMaps.mesh.nElements_global):
        for nN in u1ds.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
            vDOF1D[u1ds.femSpace.dofMap.l2g[eN,nN]] = uArray1D[eN,nN]
            grad_vDOF1D[u1ds.femSpace.dofMap.l2g[eN,nN]] = grad_uArray1D[eN,nN,0]
    u1dsNew = FiniteElementFunction(femSpace1D,dof = vDOF1D,name="newdist0.5")
    u1dsNew.writeFunctionEnsight("mesh1d",append=True)
    grad_u1ds = FiniteElementFunction(femSpace1D,dof = grad_vDOF1D,dim_dof=u1ds.femSpace.referenceFiniteElement.referenceElement.dim,name="graddist0.5",isVector=True)
    grad_u1ds.writeFunctionEnsight("mesh1d",append=True)
    uArray1DTrace = numpy.zeros((mesh1d.nElements_global,
                              femSpace1D.referenceFiniteElement.referenceElement.nElementBoundaries,
                              nq1Db),
                             'd')
    grad_uArray1DTrace = numpy.zeros((mesh1d.nElements_global,
                                   femSpace1D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                   nq1Db,
                                   femSpace1D.referenceFiniteElement.referenceElement.dim),
                                  'd')
    u1ds.getValuesTrace(vArray1DTrace,uArray1DTrace)
    u1ds.getGradientValuesTrace(grad_vArray1DTrace,grad_uArray1DTrace)
    caseOut=open('mesh1d.case','a')
    caseOut.write('scalar per measured node: distBoundary distBoundary1d.scl\n')
    caseOut.write('vector per measured node: gradDistBoundary gradDistBoundary1d.vec\n')
    caseOut.close()
    distOut = open('distBoundary1D.scl','w')
    distOut.write('distance function on element boundary\n')
    gradDistOut = open('gradDistBoundary1D.vec','w')
    gradDistOut.write('gradient of distance functionon element boundary\n')
    nColumnsVec = 0
    nColumnsScl = 0
    for eN in range(mesh1d.nElements_global):
        for ebN in elementMaps1D.referenceElement.range_nElementBoundaries:
            for k in range(nq1Db):
                distOut.write('%12.5e' %(uArray1DTrace[eN,ebN,k],))
                nColumnsScl+=1
                if nColumnsScl == 6:
                    nColumnsScl = 0
                    distOut.write('\n')
                gradDistOut.write('%12.5e%12.5e%12.5e' %(grad_uArray1DTrace[eN,ebN,k,0],0.0,0.0))
                nColumnsVec+=3
                if nColumnsVec == 6:
                    nColumnsVec = 0
                    gradDistOut.write('\n')
    distOut.write('\n')
    distOut.close()
    gradDistOut.write('\n')
    gradDistOut.close()
    print("*************************************************FiniteElementFunction-2D-Scalar************************************************")
    u2ds = FiniteElementFunction(femSpace2D,name="dist(0.5,x)")
    for nN in range(mesh2d.nNodes_global):
        u2ds.dof[nN] = sqrt((mesh2d.nodeArray[nN,0] - 0.5)**2+(mesh2d.nodeArray[nN,1]-0.5)**2)
    u2ds.writeFunctionEnsight("mesh2d",append=True)
    uArray2D = numpy.zeros((mesh2d.nElements_global,nq2D),'d')
    grad_uArray2D = numpy.zeros((mesh2d.nElements_global,
                                   nq2D,
                                   femSpace2D.referenceFiniteElement.referenceElement.dim),
                                  'd')
    u2ds.getValues(vArray2D,uArray2D)
    u2ds.getGradientValues(grad_vArray2D,grad_uArray2D)
    #read the values and gradients back into degrees of freedom and verify that they match the original
    vDOF2D = numpy.zeros((u2ds.femSpace.dim*u2ds.dim_dof),
                         'd')
    grad_vDOF2D = numpy.zeros((u2ds.femSpace.dim*u2ds.femSpace.referenceFiniteElement.referenceElement.dim),
                                'd')
    for eN in range(u2ds.femSpace.elementMaps.mesh.nElements_global):
        for nN in u2ds.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
            vDOF2D[u2ds.femSpace.dofMap.l2g[eN,nN]] = uArray2D[eN,nN]
            grad_vDOF2D[u2ds.femSpace.dofMap.l2g[eN,nN]*2+0] = grad_uArray2D[eN,nN,0]
            grad_vDOF2D[u2ds.femSpace.dofMap.l2g[eN,nN]*2+1] = grad_uArray2D[eN,nN,1]
    u2dsNew = FiniteElementFunction(femSpace2D,dof = vDOF2D,name="newdist0.5")
    u2dsNew.writeFunctionEnsight("mesh2d",append=True)
    grad_u2ds = FiniteElementFunction(femSpace2D,dof = grad_vDOF2D,dim_dof=u2ds.femSpace.referenceFiniteElement.referenceElement.dim,name="graddist0.5",isVector=True)
    grad_u2ds.writeFunctionEnsight("mesh2d",append=True)
    uArray2DTrace = numpy.zeros((mesh2d.nElements_global,
                                   femSpace2D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                   nq2Db),
                                  'd')
    grad_uArray2DTrace = numpy.zeros((mesh2d.nElements_global,
                                        femSpace2D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                        nq2Db,
                                        femSpace2D.referenceFiniteElement.referenceElement.dim),
                                       'd')
    u2ds.getValuesTrace(vArray2DTrace,uArray2DTrace)
    u2ds.getGradientValuesTrace(grad_vArray2DTrace,grad_uArray2DTrace)
    caseOut=open('mesh2d.case','a')
    caseOut.write('scalar per measured node: distBoundary distBoundary2d.scl\n')
    caseOut.write('vector per measured node: gradDistBoundary gradDistBoundary2d.vec\n')
    caseOut.close()
    distOut = open('distBoundary2D.scl','w')
    distOut.write('distance function on element boundary\n')
    gradDistOut = open('gradDistBoundary2D.vec','w')
    gradDistOut.write('gradient of distance functionon element boundary\n')
    nColumnsVec = 0
    nColumnsScl = 0
    for eN in range(mesh2d.nElements_global):
        for ebN in elementMaps2D.referenceElement.range_nElementBoundaries:
            for k in range(nq2Db):
                distOut.write('%12.5e' %(uArray2DTrace[eN,ebN,k],))
                nColumnsScl+=1
                if nColumnsScl == 6:
                    nColumnsScl = 0
                    distOut.write('\n')
                gradDistOut.write('%12.5e%12.5e%12.5e' %(grad_uArray2DTrace[eN,ebN,k,0],grad_uArray2DTrace[eN,ebN,k,1],0.0))
                nColumnsVec+=3
                if nColumnsVec == 6:
                    nColumnsVec = 0
                    gradDistOut.write('\n')
    distOut.write('\n')
    distOut.close()
    gradDistOut.write('\n')
    gradDistOut.close()
    print("*************************************************FiniteElementFunction-3D-Scalar************************************************")
    u3ds = FiniteElementFunction(femSpace3D,name="dist(0.5,x)")
    for nN in range(mesh3d.nNodes_global):
        u3ds.dof[nN] = sqrt((mesh3d.nodeArray[nN,0]-0.5)**2+(mesh3d.nodeArray[nN,1]-0.5)**2+(mesh3d.nodeArray[nN,2]-0.5)**2)
        print(u3ds.dof[nN])
    u3ds.writeFunctionEnsight("mesh3d",append=True)
    uArray3D = numpy.zeros((mesh3d.nElements_global,nq3D),'d')
    grad_uArray3D = numpy.zeros((mesh3d.nElements_global,
                                   nq3D,
                                   femSpace3D.referenceFiniteElement.referenceElement.dim),
                                  'd')
    u3ds.getValues(vArray3D,uArray3D)
    u3ds.getGradientValues(grad_vArray3D,grad_uArray3D)
    #read the values and gradients back into degrees of freedom and verify that they match the original
    vDOF3D = numpy.zeros((u3ds.femSpace.dim*u3ds.dim_dof),
                         'd')
    grad_vDOF3D = numpy.zeros((u3ds.femSpace.dim*u3ds.femSpace.referenceFiniteElement.referenceElement.dim),
                                'd')
    for eN in range(u3ds.femSpace.elementMaps.mesh.nElements_global):
        for nN in u3ds.femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
            vDOF3D[u3ds.femSpace.dofMap.l2g[eN,nN]] = uArray3D[eN,nN]
            grad_vDOF3D[u3ds.femSpace.dofMap.l2g[eN,nN]*3+0] = grad_uArray3D[eN,nN,0]
            grad_vDOF3D[u3ds.femSpace.dofMap.l2g[eN,nN]*3+1] = grad_uArray3D[eN,nN,1]
            grad_vDOF3D[u3ds.femSpace.dofMap.l2g[eN,nN]*3+2] = grad_uArray3D[eN,nN,2]
    u3dsNew = FiniteElementFunction(femSpace3D,dof = vDOF3D,name="newdist0.5")
    u3dsNew.writeFunctionEnsight("mesh3d",append=True)
    grad_u3ds = FiniteElementFunction(femSpace3D,dof = grad_vDOF3D,dim_dof=u3ds.femSpace.referenceFiniteElement.referenceElement.dim,name="graddist0.5",isVector=True)
    grad_u3ds.writeFunctionEnsight("mesh3d",append=True)
    uArray3DTrace = numpy.zeros((mesh3d.nElements_global,
                              femSpace3D.referenceFiniteElement.referenceElement.nElementBoundaries,
                              nq3Db),
                             'd')
    grad_uArray3DTrace = numpy.zeros((mesh3d.nElements_global,
                                   femSpace3D.referenceFiniteElement.referenceElement.nElementBoundaries,
                                   nq3Db,
                                   femSpace3D.referenceFiniteElement.referenceElement.dim),
                                  'd')
    u3ds.getValuesTrace(vArray3DTrace,uArray3DTrace)
    u3ds.getGradientValuesTrace(grad_vArray3DTrace,grad_uArray3DTrace)
    caseOut=open('mesh3d.case','a')
    caseOut.write('scalar per measured node: distBoundary distBoundary3d.scl\n')
    caseOut.write('vector per measured node: gradDistBoundary gradDistBoundary3d.vec\n')
    caseOut.close()
    distOut = open('distBoundary3D.scl','w')
    distOut.write('distance function on element boundary\n')
    gradDistOut = open('gradDistBoundary3D.vec','w')
    gradDistOut.write('gradient of distance functionon element boundary\n')
    nColumnsVec = 0
    nColumnsScl = 0
    for eN in range(mesh3d.nElements_global):
        for ebN in elementMaps3D.referenceElement.range_nElementBoundaries:
            for k in range(nq3Db):
                distOut.write('%12.5e' %(uArray3DTrace[eN,ebN,k],))
                nColumnsScl+=1
                if nColumnsScl == 6:
                    nColumnsScl = 0
                    distOut.write('\n')
                gradDistOut.write('%12.5e%12.5e%12.5e' %(grad_uArray3DTrace[eN,ebN,k,0],grad_uArray3DTrace[eN,ebN,k,1],grad_uArray3DTrace[eN,ebN,k,2]))
                nColumnsVec+=3
                if nColumnsVec == 6:
                    nColumnsVec = 0
                    gradDistOut.write('\n')
    distOut.write('\n')
    distOut.close()
    gradDistOut.write('\n')
    gradDistOut.close()
    mlMesh1D = MultilevelEdgeMesh(3,1,1,refinementLevels=3)
    meshTransfers1D = MultilevelProjectionOperators(mlMesh1D,C0_AffineLinearOnSimplexWithNodalBasis,1)
    print(meshTransfers1D.prolongList)
    print(meshTransfers1D.restrictList)
    print(meshTransfers1D.prolong_bcList)
    print(meshTransfers1D.restrict_bcList)
    mlMesh2D = MultilevelTriangularMesh(3,3,1,refinementLevels=3)
    meshTransfers2D = MultilevelProjectionOperators(mlMesh2D,C0_AffineLinearOnSimplexWithNodalBasis,2)
    print(meshTransfers2D.prolongList)
    print(meshTransfers2D.restrictList)
    print(meshTransfers2D.prolong_bcList)
    print(meshTransfers2D.restrict_bcList)
    mlMesh3D = MultilevelTetrahedralMesh(3,3,3,refinementLevels=3)
    meshTransfers3D = MultilevelProjectionOperators(mlMesh3D,C0_AffineLinearOnSimplexWithNodalBasis,3)
    print(meshTransfers3D.prolongList)
    print(meshTransfers3D.restrictList)
    print(meshTransfers3D.prolong_bcList)
    print(meshTransfers3D.restrict_bcList)
    #
    # mwf tests
    #
    verboseLevel = 1
    #first just create a simple triangular mesh and look at it in a
    #couple of different ways
    Lx = 1.0   #domain length in x and y
    Ly = 1.0

    #number of nodes for rectangular grid upon which triangular mesh
    #will be built should get 2 triangles for each rectangle
    #(nx-1)(ny-1) in the original grid
    nx = 21
    ny = 21

    #flag for viewing mesh in construction
    #0 -- do nothing (default)
    #1 -- gnuplot
    #2 -- matlab
    viewMesh = 2
    mesh = TriangularMesh()
    mesh.constructTriangularMeshOnRectangle(Lx,Ly,nx,ny,viewMesh)

    print('mesh Info says \n',mesh.meshInfo())

    #now create a DG1 finite element space
    globalDGspace = DG_AffineLinearOnSimplexWithNodalBasis(mesh,2)
    globalCGspace = C0_AffineLinearOnSimplexWithNodalBasis(mesh,2)
    dgDofMap = globalDGspace.dofMap
    if verboseLevel > 0:
        print("DG1 dofMap l2g is ")
        for eN in range(mesh.nElements_global):
            sn = '\t %i %i %i' % tuple(dgDofMap.l2g[eN,:])
            print(sn)
    ngdim = globalDGspace.dim
    nelem = mesh.nElements_global
    polydim = 3
    #what about a member of the DG1 space
    u = FiniteElementFunction(globalDGspace)
    ucg = FiniteElementFunction(globalCGspace)
    #try to just calculate u as an analytical function of x
    #nodalRefPoints = numpy.zeros((polydim,3),'d')
    #nodalRefPoints[0,:] = (0.0,0.0)
    #nodalRefPoints[1,:] = (1.0,0.0)
    #nodalRefPoints[2,:] = (0.0,1.0)
    nodalRefPoints = u.femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray
    if verboseLevel > 0:
        print('nodalRefPoints = \n',nodalRefPoints)
    #end verbose check

    #physNodes = numpy.zeros((nelem,polydim,2),'d')
    physNodes = numpy.zeros((mesh.nElements_global,
                               u.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints,
                               u.femSpace.referenceFiniteElement.referenceElement.dim),
                              'd')

    u.femSpace.elementMaps.getValues(nodalRefPoints,physNodes)

    if verboseLevel > 0:
        print('physical Nodes = \n',physNodes)
    #end verbose check

    #set value of u manually
    for ie in range(mesh.nElements_global):
        for iv in range(u.femSpace.referenceFiniteElement.localFunctionSpace.dim):
            px = physNodes[ie,iv,0]
            py = physNodes[ie,iv,1]
            ig = u.femSpace.dofMap.l2g[ie,iv]
            u.dof[ig] = math.sin(0.5*math.pi*px)*math.cos(0.5*math.pi*py)
            ucg.dof[ucg.femSpace.dofMap.l2g[ie,iv]] = u.dof[ig]
        #end local shape loop
    #end element loop

    if verboseLevel > 0:
        print('u dof values = \n',u.dof)
    #end verbose check

    u.writeFunctionMatlab('sol','mesh')

    #moved  to MeshTools.py cek
    #if verboseLevel > 0:
    #    #now look at edge --> element loops for dg algorithms
    #    testEdgeToElementMapping(mesh)
    #end verbose
    #try to build trace operators in crude way
    #
    #traceOp = buildTraceOpArray(mesh,globalDGspace,verboseLevel)
    #def checkFunctionTrace(u,mesh,dgspace,traceArray,jumpTol=1.0e-8):
    #cek
    #Changing this to the way one would do it with FemTools
    #
    jumpTol=1.0e-8
#      """
#      take a dg finite element function and see if values
#      differ from left and right neighbors on an edge
#      Basically, it's computing a jump term.
#      For a continuous solution, should get no difference.
#      """

#      #global number of edges and elements
#      ngEdges = mesh.nElementBoundaries_global
#      ngElems = mesh.nElements_global
#      #
#      polyOrder  = dgspace.polyOrder
#      edgeDim    = polyOrder+1
#      elemDim    = dgspace.localDim
#      #hold local element dofs
#      locDofValues = numpy.zeros((elemDim),'d')
#      for globedge in range(ngEdges):
#          jumpValues   = numpy.zeros((edgeDim),'d')
#          for neig in range(2):
#              elid = mesh.elementBoundaryElementsArray[globedge,neig]
#              if elid > -1: #element neighbor exists
#                  for k in range(elemDim):
#                      ig = u.femSpace.dofMap.l2g[elid,k]
#                      locDofValues[k] = u.dof[ig]
#                  #end k
#                  for j in range(edgeDim):
#                      for k in range(elemDim):
#                          jumpValues[j] += (1.0-2.*neig)*traceArray[globedge,neig,j,k]*locDofValues[k]
#                      #end k
#                  #end j
#              else: #what to do about edges on bounary
#                  jumpValues *= 0.0
#              #end if
#          #end neig loop
    nElementBoundaryPoints = 2
    elementBoundaryPoints = numpy.array([[0.0],[1.0]])
    jumps = numpy.zeros((mesh.nInteriorElementBoundaries_global,nElementBoundaryPoints),'d')
    physTraceArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints,2),'d')
    refTraceArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints,2),'d')
    invjArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints,2,2),'d')
    gArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints,2,1),'d')
    sqrt_det_gArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints),'d')
    nArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints,2),'d')
    elementBoundaryPointsArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints,1),'d')
    vTraceArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints,u.femSpace.max_nDOF_element),'d')
    vcgTraceArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints,u.femSpace.max_nDOF_element),'d')
    uTraceArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints),'d')
    ucgTraceArray = numpy.zeros((mesh.nElements_global,mesh.nElementBoundaries_element,nElementBoundaryPoints),'d')
    u.femSpace.elementMaps.getValuesTrace(elementBoundaryPoints,physTraceArray)
    for ebNI in range(mesh.nInteriorElementBoundaries_global):
        ebN = mesh.interiorElementBoundariesArray[ebNI]
        left_eN_global   = mesh.elementBoundaryElementsArray[ebN][0]
        right_eN_global  = mesh.elementBoundaryElementsArray[ebN][1]
        left_ebN_element  = mesh.elementBoundaryLocalElementBoundariesArray[ebN][0]
        right_ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN][1]
        for n in range(nElementBoundaryPoints):
            physTraceArray[right_eN_global,right_ebN_element,n,:] = physTraceArray[left_eN_global,left_ebN_element,n]
    u.femSpace.elementMaps.getJacobianValuesTrace(refTraceArray,invjArray,gArray,sqrt_det_gArray,nArray)
    u.femSpace.elementMaps.getInverseValuesTrace(physTraceArray,refTraceArray)
    for ebNI in range(mesh.nInteriorElementBoundaries_global):
        ebN = mesh.interiorElementBoundariesArray[ebNI]
        left_eN_global   = mesh.elementBoundaryElementsArray[ebN][0]
        right_eN_global  = mesh.elementBoundaryElementsArray[ebN][1]
        left_ebN_element  = mesh.elementBoundaryLocalElementBoundariesArray[ebN][0]
        right_ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN][1]
        for n in range(nElementBoundaryPoints):
            elementBoundaryPointsArray[left_eN_global,left_ebN_element,n,:]=elementBoundaryPoints[n]
            elementBoundaryPointsArray[right_eN_global,right_ebN_element,n,:]= u.femSpace.referenceFiniteElement.referenceElement.boundaryMapInverseList[right_ebN_element](refTraceArray[right_eN_global,right_ebN_element,n])
    for ebNE in range(mesh.nExteriorElementBoundaries_global):
        ebN = mesh.exteriorElementBoundariesArray[ebNE]
        eN_global = mesh.elementBoundaryElementsArray[ebN][0]
        ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN][0]
        for n in range(nElementBoundaryPoints):
            elementBoundaryPointsArray[eN_global,ebN_element,n,:] = elementBoundaryPoints[n]
    u.femSpace.getBasisValuesTraceAtArray(elementBoundaryPointsArray,vTraceArray)
    #u.femSpace.getBasisValuesTrace(elementBoundaryPoints,vTraceArray)
    u.getValuesTrace(vTraceArray,uTraceArray)
    ucg.femSpace.getBasisValuesTrace(elementBoundaryPoints,vcgTraceArray)
    ucg.getValuesTrace(vcgTraceArray,ucgTraceArray)
    for ebNI in range(mesh.nInteriorElementBoundaries_global):
        ebN = mesh.interiorElementBoundariesArray[ebNI]
        left_eN_global   = mesh.elementBoundaryElementsArray[ebN][0]
        right_eN_global  = mesh.elementBoundaryElementsArray[ebN][1]
        left_ebN_element  = mesh.elementBoundaryLocalElementBoundariesArray[ebN][0]
        right_ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN][1]
        for n in range(nElementBoundaryPoints):
            jumps[ebNI,n] = uTraceArray[left_eN_global,left_ebN_element,n] - uTraceArray[right_eN_global,right_ebN_element,n]
            if abs(jumps[ebNI,n]) > jumpTol:
                print('Warning edge ',ebN,' jumpValues big= \n',jumps[ebNI,n])
                print('left element ',left_eN_global,'left edge ',left_ebN_element,' value = ',uTraceArray[left_eN_global,left_ebN_element,n])
                print('right element ',right_eN_global,'right edge ', right_eN_global,' value = ',uTraceArray[right_eN_global,right_ebN_element,n])
                print('left element ',left_eN_global,'left edge ',left_ebN_element,' value = ',ucgTraceArray[left_eN_global,left_ebN_element,n])
                print('right element ',right_eN_global,'right edge ', right_eN_global,' value = ',ucgTraceArray[right_eN_global,right_ebN_element,n])
                print(physTraceArray[left_eN_global,left_ebN_element,n])
                print(physTraceArray[right_eN_global,right_ebN_element,n])
                print(math.sin(0.5*math.pi*physTraceArray[left_eN_global,left_ebN_element,n,0])*math.cos(0.5*math.pi*physTraceArray[left_eN_global,left_ebN_element,n,1]))
                print(math.sin(0.5*math.pi*physTraceArray[right_eN_global,right_ebN_element,n,0])*math.cos(0.5*math.pi*physTraceArray[right_eN_global,right_ebN_element,n,1]))
#          #end if big jump
#      #end for loop on edges
#  #end checkFunctionTrace
    #check if contiuous solution has no jumps?
    #checkFunctionTrace(u,mesh,globalDGspace,traceOp)
