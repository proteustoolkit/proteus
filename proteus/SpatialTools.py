"""Module creating predifined or custom shapes. Each shape needs a
Domain as argument (from proteus.Domain). A Domain can contain any
number of shapes.  Boundary conditions objects are automatically
created for each facet (3D) or segment (2D) defining the shape.

Classes:

  * Shape: super class, regroups functions common to all shapes
  * Cuboid: creates a 3D cuboid
  * Rectangle: creates a 2D rectangle
  * Custom: creates a custom shape from a given set vertices, facets, etc.


Example::

  from proteus import Domain
  from proteus import SpatialTools as st
  import numpy as np
  domain = Domain.PlanarStraightLineGraphDomain()
  shape1 = st.Rectangle(domain, dim=[0.5, 0.5], coords=[1., 1.])
  shape2 = st.Rectangle(domain. dim=[0.3, 0.2], coords=[3., 3.])
  shape2.rotate(np.pi/3.)
  shape2.BC_dict["left"].uOfXT = lambda x, t: 0.

  st.assembleDomain(domain)

.. inheritance-diagram:: proteus.SpatialTools
   :parts: 1

"""
from math import cos, sin, sqrt
import math
import sys
import copy
import numpy as np
from proteus import BoundaryConditions as bc
from .Profiling import logEvent
from subprocess import check_call
from proteus import Comm
from itertools import compress
import csv
import numbers
import os
from math import atan2, acos, asin
from proteus import AuxiliaryVariables, Archiver, Profiling, Gauges
from proteus.mbd import BodyDynamics as bd
from copy import deepcopy
from functools import reduce


class Shape(object):
    """
    Base/super class of all shapes.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    nd: Optional[int]
        Number of dimensions of the shape. If not set, will take the number of
        dimensions of the domain.
    BC_class: Optional[proteus.BoundaryConditions.BC_Base]
        Class to use for boundary conditions (e.g.
        proteus.BoundaryConditions.BC_Base or
        proteus.BoundaryConditions.mprans.BC_RANS).
    """
    count_all = 0

    def __init__(self, domain, nd=None, BC_class=None):
        if nd != domain.nd:
            logEvent('Shape ('+repr(nd)+'D) and Domain ('+repr(domain.nd)+'D)' \
                ' have different dimensions!')
            sys.exit()
        self.Domain = domain
        domain.shape_list.append(self)
        self.__class__.count_all += 1
        self.name = 'shape'+str(self.__class__.count_all)
        self.nd = nd
        self.BC_class = BC_class or bc.BC_Base
        self.vertices = None
        self.vertexFlags = None
        self.vertexFlags_global = None
        self.segments = None
        self.segments_global = None
        self.segmentFlags = None
        self.segmentFlags_global = None
        self.facets = None
        self.facets_global = None
        self.facetFlags = None
        self.facetFlags_global = None
        self.regions = None
        self.volumes = None
        self.regionFlags = None
        self.regionFlags_global = None
        self.holes = None
        self.holes_ind = None
        self.barycenter = np.zeros(3)
        self.coords = None  # Only used for predefined shapes
                            # (can be different from barycenter)
        self.coords_system = np.eye(nd)
        self.boundaryTags = {}
        self.boundaryTags_global = {}
        self.b_or = None  # boundary orientation
        self.volume = None
        self.children = {}
        self.BC_list = []

    def _checkFlags(self, flagSet):
        """
        Checks if flags are set correctly

        Parameters
        ----------
        flagSet: list
            List of flags.
        """
        flagSet = set(flagSet)
        checkFlag = min(flagSet)
        assert checkFlag == 1, 'Minimum boundary/region tag/flag must be 1'
        for flag in flagSet:
            # hack for SWFlow BCs
            if (flag==99):
                checkFlag += 1
                return
            assert flag == checkFlag, 'Boundary/region tags/flags must be'
            'defined as a suite of numbers with no gap!'
            checkFlag += 1

    def _checkListOfLists(self, list_of_lists):
        """
        Checks if the list of lists has the right dimension

        Parameters
        ----------
        list_of_lists: list
        """
        assert len(list_of_lists[0]) == self.nd, 'must have be a list of: ' \
            'lists of length ' + self.nd

    def _checkNd(self, array):
        """
        Checks if an array is of the same dimension of the Shape instance or
        is of dimension 3

        Parameters
        ----------
        array: array_like
        """
        assert len(array) == self.nd or len(array) == 3, 'wrong dimension'

    def setPosition(self, coords):
        """
        Set position with coords of the barycenter

        Parameters
        ----------
        coords: array_like
            New set of coordinates for barycenter (list/array).
        """
        old_coords = np.array(self.barycenter)
        if self.Domain.nd == 2 and len(old_coords) == 3:
            trans = coords - old_coords[:2]
        else:
            trans = coords - old_coords
        self.translate(trans)

    def setBarycenter(self, barycenter):
        """
        Set barycenter (center of mass) of the shape
        (!) this function does not move the shape

        Parameters
        ----------
        barycenter: array_like
            Global coordinates of barycenter (list/array).
        """
        if self.Domain.nd == 2 and len(barycenter) == 2:
            self.barycenter[:2] = barycenter
        else:
            self.barycenter[:] = barycenter

    def setRegions(self, regions, regionFlags=None):
        """
        Sets new regions for the Shape

        Parameters
        ----------
        regions: array_like
            Array of coordinates of regions.
        regionFlags: Optional[array_like]
            Array of flags.
        """
        self._checkListOfLists(regions)
        if regionFlags is not None:
            self._checkFlags(regionFlags)
        else:
            regionFLags = self.regionFlags
        assert len(regions) == len(regionFlags), 'regions and regionFLags'\
            'must have the same length'
        self.regions = np.array(regions)
        self.regionFlags = np.array(regionFlags)

    def setChildShape(self, shape, ind=0):
        """
        Indicates that a shape is contained in this shape
        (this function or setParentShape is necessary for gmsh)

        Parameters
        ----------
        shape: proteus.SpatialTools.Shape
            child of this shape
        ind: int
            parent (current) shape's local index of volume (3D) or facet (2D)
            containing the child shape
        """
        if ind in self.children:
            self.children[ind] += [shape]
        else:
            self.children[ind] = [shape]

    def setParentShape(self, shape, ind=0):
        """
        Indicates that this shape is within another shape
        (this function or setChildShape is necessary for gmsh)
        Sets the shape within

        Parameters
        ----------
        shape: proteus.SpatialTools.Shape
            the parent of this shape
        ind: int
            parent shape's local index of volume (3D) or facet (2D) containing
            the child (current) shape
        """
        if ind in shape.children:
            shape.children[ind] += [self]
        else:
            shape.children[ind] = [self]

    def setHoles(self, holes, indice=None):
        """
        Sets a 'hole' in the mesh. The region where the hole is defined will
        not be meshed.

        Parameters
        ----------
        holes: array_like
            Array of coordinates of holes (list/array).
        indice: array_like
            Array of index of region where hole is (list/array). Only for gmsh
        """
        self._checkListOfLists(holes)
        self.holes = np.array(holes)
        if indice is not None:
            self.holes_ind = indice

    def rotate(self, rot, axis=(0, 0, 1), pivot=None):
        """
        Function to rotate Shape

        Parameters
        ----------
        rot: float
            Angle of rotation (in radians).
        axis: Optional[array_like]
            Vector used for rotation. Not necessary for rotation in 2D.
        pivot: Optional[array_list]
            Point around which the shape will rotate. If not set, the
            barycenter will be the center of rotation.

        Notes
        -----
        Rotated attributes:
        - vertices
        - holes
        - regions
        - local coordinate system
        - boundary orientations
        - coords (if not None)
        - barycenter
        """
        # This function and rotate2D/rotate3D could be optimized
        rot = float(rot)
        nd = self.nd
        if pivot is None:
            pivot = self.barycenter
        if nd == 2:
            pivot = pivot[:2]
            self.vertices[:] = rotation2D(self.vertices, rot, pivot)
            if self.holes is not None:
                self.holes[:] = rotation2D(self.holes, rot, pivot)
            if self.regions is not None:
                self.regions[:] = rotation2D(self.regions, rot, pivot)
            self.barycenter[:2] = rotation2D(self.barycenter[:nd], rot, pivot)
            self.coords_system[:] = rotation2D(self.coords_system, rot,
                                               (0., 0.))
            if self.b_or is not None:
                self.b_or[:] = rotation2D(self.b_or, rot, (0., 0.))
            if self.coords is not None:
                self.coords[:] = rotation2D(self.coords, rot, pivot)
        elif nd == 3:
            self.vertices[:] = rotation3D(self.vertices, rot, axis, pivot)
            if self.holes is not None:
                self.holes[:] = rotation3D(self.holes, rot, axis, pivot)
            if self.regions is not None:
                self.regions[:] = rotation3D(self.regions, rot, axis, pivot)
            self.barycenter[:] = rotation3D(self.barycenter, rot, axis, pivot)
            self.coords_system[:] = rotation3D(self.coords_system, rot, axis,
                                               (0., 0., 0.))
            if self.b_or is not None:
                self.b_or[:] = rotation3D(self.b_or, rot, axis, (0., 0., 0.))
            if self.coords is not None:
                self.coords[:] = rotation3D(self.coords, rot, axis, pivot)

    def translate(self, trans):
        """
        Function to translate Shape

        Parameters
        ----------
        trans: array_like
            Translation values.

        Notes
        -----
        Translated attributes:
        - vertices
        - regions
        - coords (if not None)
        - barycenters
        - holes
        """
        self.vertices += trans
        if self.regions is not None:
            self.regions += trans
        if self.coords is not None:
            self.coords += trans
        if self.Domain.nd == 2:
            trans2 = (trans[0], trans[1], 0.)
            self.barycenter += trans2
        else:
            self.barycenter += trans
        if self.holes is not None:
            self.holes += trans

    def getPosition(self):
        """
        Returns
        -------
        barycenter: array_like
            Current position of barycenter.
        """
        return self.barycenter

    def getRotation(self):
        """
        Returns
        -------
        coords_system: array_like
            Local coordinate system relative to global coordinate system.
        """
        return self.coords_system


class Cuboid(Shape):
    """
    Class to create a 3D cuboid

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    dim: Optional[array_like]
        Dimensions of the cuboid.
    coords: Optional[array_like]
        Coordinates of the centroid of the shape.
    barycenter: Optional[array_like]
        Coordinates of the barycenter.
    """
    count = 0

    def __init__(self, domain, dim=(0., 0., 0.), coords=(0., 0., 0.),
                 barycenter=None):
        super(Cuboid, self).__init__(domain, nd=3)
        self.__class__.count += 1
        self.name = "cuboid" + str(self.__class__.count)
        self.dim = L, W, H = dim  # length, width height
        self.volume = L*W*H
        self.coords = x, y, z = np.array(coords)
        self.vertices = np.array([[x-0.5*L, y-0.5*W, z-0.5*H],
                                  [x-0.5*L, y+0.5*W, z-0.5*H],
                                  [x+0.5*L, y+0.5*W, z-0.5*H],
                                  [x+0.5*L, y-0.5*W, z-0.5*H],
                                  [x-0.5*L, y-0.5*W, z+0.5*H],
                                  [x-0.5*L, y+0.5*W, z+0.5*H],
                                  [x+0.5*L, y+0.5*W, z+0.5*H],
                                  [x+0.5*L, y-0.5*W, z+0.5*H]])
        if self.Domain.nd == 2:
            self.vertices = np.array([[x-0.5*L, y-0.5*H],
                                      [x+0.5*L, y-0.5*H],
                                      [x+0.5*L, y+0.5*H],
                                      [x-0.5*L, y+0.5*H]])
        self.facets = np.array([[[0, 1, 2, 3]],  # z-
                                [[1, 2, 6, 5]],  # y+
                                [[2, 3, 7, 6]],  # x+
                                [[3, 0, 4, 7]],  # y-
                                [[0, 1, 5, 4]],  # x-
                                [[4, 5, 6, 7]]])  # z+
        self.b_or = np.array([[0.,  0., -1.],
                              [0.,  1.,  0.],
                              [1.,  0.,  0.],
                              [0., -1.,  0.],
                              [-1., 0.,  0.],
                              [0.,  0.,  1.]])
        self.regions = np.array([[x, y, z]])
        self.volumes = [[[0, 1, 2, 3, 4, 5]]]
        # defining flags for boundary conditions
        self.boundaryTags = bt = {'z-': 1,
                                  'y+': 2,
                                  'x+': 3,
                                  'y-': 4,
                                  'x-': 5,
                                  'z+': 6}
        self.facetFlags = np.array([bt['z-'], bt['y+'], bt['x+'],
                                    bt['y-'], bt['x-'], bt['z+']])
        self.vertexFlags = np.array([bt['z-'], bt['z-'], bt['z-'],
                                     bt['z-'], bt['z+'], bt['z+'],
                                     bt['z+'], bt['z+']])
        self.regionFlags = np.array([1])
        # Initialize (empty) boundary conditions
        self.BC = {'z-': self.BC_class(shape=self, name='z-',
                                       b_or=self.b_or, b_i=0),
                   'y+': self.BC_class(shape=self, name='y+',
                                       b_or=self.b_or, b_i=1),
                   'x+': self.BC_class(shape=self, name='x+',
                                       b_or=self.b_or, b_i=2),
                   'y-': self.BC_class(shape=self, name='y-',
                                       b_or=self.b_or, b_i=3),
                   'x-': self.BC_class(shape=self, name='x-',
                                       b_or=self.b_or, b_i=4),
                   'z+': self.BC_class(shape=self, name='z+',
                                       b_or=self.b_or, b_i=5)}
        self.BC_list = [self.BC['z-'],
                        self.BC['y+'],
                        self.BC['x+'],
                        self.BC['y-'],
                        self.BC['x-'],
                        self.BC['z+']]
        # self.BC = BCContainer(self.BC_dict)
        if barycenter is None:
            self.barycenter = np.array(coords)
        else:
            self.barycenter = np.array(barycenter)
        self.It = np.array([[(W**2.+H**2.)/12., 0, 0],
                            [0, (L**2.+H**2.)/12., 0],
                            [0, 0, (W**2.+L**2.)/12.]])

    def setDimensions(self, dim):
        """
        Sets dimensions of the shape.

        Parameters
        ----------

        dim: array_like
            New dimensions of the shape.
        """
        self.dim = dim
        L, W, H = dim
        x, y, z = self.coords
        self.vertices[:] = [[x-0.5*L, y-0.5*W, z-0.5*H],
                            [x-0.5*L, y+0.5*W, z-0.5*H],
                            [x+0.5*L, y+0.5*W, z-0.5*H],
                            [x+0.5*L, y-0.5*W, z-0.5*H],
                            [x-0.5*L, y-0.5*W, z+0.5*H],
                            [x-0.5*L, y+0.5*W, z+0.5*H],
                            [x+0.5*L, y+0.5*W, z+0.5*H],
                            [x+0.5*L, y-0.5*W, z+0.5*H]]
        self.volume = L*W*H




class Sphere(Shape):
    """
    Class to create a 3D cuboid

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    dim: Optional[array_like]
        Dimensions of the cuboid.
    coords: Optional[array_like]
        Coordinates of the centroid of the shape.
    barycenter: Optional[array_like]
        Coordinates of the barycenter.
    """
    count = 0

    def __init__(self, domain, radius, coords=(0.,0.,0.), barycenter=None,
                 nSectors=10):
        super(Sphere, self).__init__(domain, nd=3)
        self.__class__.count += 1
        self.name = "sphere" + str(self.__class__.count)
        self.radius = radius
        self.coords = np.array(coords)
        self.nSectors = nSectors
        self.constructShape()
        self.volumes = [[[i for i in range(len(self.facets))]]]
        # defining flags for boundary conditions
        self.boundaryTags = bt = {'sphere': 1}
        self.BC = {'sphere': self.BC_class(shape=self, name='sphere')}
        self.BC_list = [self.BC['sphere']]
        # self.BC = BCContainer(self.BC_dict)
        if barycenter is None:
            self.barycenter = np.array(coords)
        else:
            self.barycenter = np.array(barycenter)

    def constructShape(self):
        """
        regualar nx,ny,nz cube packing with padding to boundary
        returns domain size and boundary flags
        """
        coords=(0.,0.,0.)
        n_domain_vertices = 8
        radius = self.radius
        nSectors = self.nSectors
        diameter=2.0*radius
        grain_centers  = {}
        north_poles = {}
        right_poles = {}
        back_poles = {}
        south_poles = {}
        left_poles = {}
        front_poles = {}

        pad= -radius
        noff= 0.
        #cube boundary vertices
        nN=0
        nF=0
        vertices=[]
        facets=[]


        nN = 0
        #cube boundary facets
        # sphereTag = 10
        # boundaries = ['bottom','front','left','right','back','top']
        # boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])
        # boundaryTags['sphere']=sphereTag

        nF = 0
        #adding poles of grains to make sure those nodes are unique

        #build sphere
        grain_center = (noff*diameter, noff*diameter, noff*diameter)
        north_pole = nN
        vertices.append([coords[0]+noff*diameter,
                        coords[1]+noff*diameter,
                        coords[2]+noff*diameter+radius])
        nN+=1
        back_pole = nN
        vertices.append([coords[0]+noff*diameter,
                        coords[1]+noff*diameter+radius,
                        coords[2]+noff*diameter])
        nN+=1
        right_pole = nN
        vertices.append([coords[0]+noff*diameter+radius,
                        coords[1]+noff*diameter,
                        coords[2]+noff*diameter])
        nN+=1
        south_pole = nN
        vertices.append([coords[0]+noff*diameter,
                        coords[1]+noff*diameter,
                        coords[2]+noff*diameter-radius])
        nN+=1
        front_pole = nN
        vertices.append([coords[0]+noff*diameter,
                        coords[1]+noff*diameter-radius,
                        coords[2]+noff*diameter])
        nN+=1

        left_pole = nN
        vertices.append([coords[0]+noff*diameter-radius,
                        coords[1]+noff*diameter,
                        coords[2]+noff*diameter])
        nN+=1


        hxi = radius/(math.sqrt(2.0)*float(nSectors));
        heta = radius/(math.sqrt(2.0)*float(nSectors));
        #now loop over grains
        #top half  sphere nodes
        top_nodes = {}
        for ii in range(2*nSectors+1):
            for jj in range(2*nSectors+1):
                if (ii,jj) == (0,0):
                    top_nodes[(ii,jj)] = left_pole
                elif (ii,jj) == (2*nSectors,0):
                    top_nodes[(ii,jj)] = back_pole
                elif (ii,jj) == (2*nSectors,2*nSectors):
                    top_nodes[(ii,jj)] = right_pole
                elif (ii,jj) == (0,2*nSectors):
                    top_nodes[(ii,jj)] = front_pole
                elif (ii,jj) == (nSectors,nSectors):
                    top_nodes[(ii,jj)] = north_pole
                else:
                    top_nodes[(ii,jj)] = nN
                    #rotate  corners of ref square to line up with poles
                    x0s = (jj-nSectors)*hxi
                    y0s = (ii-nSectors)*heta
                    r0s = math.sqrt(x0s**2 + y0s**2)
                    theta0s = math.atan2(y0s,x0s)
                    theta1s = theta0s - math.pi/4.0
                    r1s = r0s
                    x1s = r1s*math.cos(theta1s)
                    y1s = r1s*math.sin(theta1s)
                    #do each quadrant
                    if x1s >= 0.0  and y1s >=0.0:
                        rc = x1s + y1s
                        thetac = theta1s
                    elif x1s < 0.0 and y1s >=0.0:
                        rc = y1s - x1s
                        thetac = theta1s
                    elif x1s <= 0.0 and y1s < 0.0:
                        rc = -(x1s+y1s)
                        thetac = theta1s
                    else:
                        rc = x1s - y1s
                        thetac = theta1s
                    eta = 0.5*math.pi*(radius-rc)/radius
    #                         xc = rc*math.cos(thetac)
    #                         yc = rc*math.sin(thetac)
    #                         zc = math.sqrt(math.fabs(radius**2 - xc**2 - yc**2))
                    xc = radius*math.cos(thetac)*math.cos(eta)
                    yc = radius*math.sin(thetac)*math.cos(eta)
                    zc = radius*math.sin(eta)
                    #zc = math.sqrt(math.fabs(radius**2 - xc**2 - yc**2))
                    #print xc,yc,zc,rc,radius**2 - xc**2 - yc**2
                    #physical coordinates
                    vertices.append([xc+grain_center[0],
                                    yc+grain_center[1],
                                    zc+grain_center[2]])
                    nN+=1
        #bottom half sphere nodes
        bottom_nodes = {}
        for ii in range(2*nSectors+1):
            for jj in range(2*nSectors+1):
                if (ii,jj) == (0,0):
                    bottom_nodes[(ii,jj)] = left_pole
                elif (ii,jj) == (2*nSectors,0):
                    bottom_nodes[(ii,jj)] = back_pole
                elif (ii,jj) == (2*nSectors,2*nSectors):
                    bottom_nodes[(ii,jj)] = right_pole
                elif (ii,jj) == (0,2*nSectors):
                    bottom_nodes[(ii,jj)] = front_pole
                elif (ii,jj) == (nSectors,nSectors):
                    bottom_nodes[(ii,jj)] = south_pole
                elif (ii in [0,2*nSectors] or
                    jj in [0,2*nSectors]):#equator
                    bottom_nodes[(ii,jj)] = top_nodes[(ii,jj)]
                else:
                    bottom_nodes[(ii,jj)] = nN
                    #rotate  corners of ref square to line up with poles
                    x0s = (jj-nSectors)*hxi
                    y0s = (ii-nSectors)*heta
                    r0s = math.sqrt(x0s**2 + y0s**2)
                    theta0s = math.atan2(y0s,x0s)
                    theta1s = theta0s - math.pi/4.0
                    r1s = r0s
                    x1s = r1s*math.cos(theta1s)
                    y1s = r1s*math.sin(theta1s)
                    #do each quadrant
                    if x1s >= 0.0  and y1s >=0.0:
                        rc = x1s + y1s
                        thetac = theta1s
                    elif x1s < 0.0 and y1s >=0.0:
                        rc = y1s - x1s
                        thetac = theta1s
                    elif x1s <= 0.0 and y1s < 0.0:
                        rc = -(x1s+y1s)
                        thetac = theta1s
                    else:
                        rc = x1s - y1s
                        thetac = theta1s
                    eta = 0.5*math.pi*(radius-rc)/radius
    #                         xc = rc*math.cos(thetac)
    #                         yc = rc*math.sin(thetac)
    #                         zc = -math.sqrt(math.fabs(radius**2 - xc**2 - yc**2))
                    xc = radius*math.cos(thetac)*math.cos(eta)
                    yc = radius*math.sin(thetac)*math.cos(eta)
                    zc = -radius*math.sin(eta)
                    #print xc,yc,zc
                    #physical coordinates
                    vertices.append([xc+grain_center[0],
                                    yc+grain_center[1],
                                    zc+grain_center[2]])
                    nN+=1
        for ii in range(2*nSectors):
            for jj in range(2*nSectors):
                if ((ii < nSectors and jj < nSectors) or
                    (ii>=nSectors and  jj>=nSectors)):
                    #top
                    facets.append([[ top_nodes[(ii,jj)],top_nodes[(ii+1,jj)],top_nodes[(ii+1,jj+1)]] ])
                    facets.append([[top_nodes[(ii,jj)],top_nodes[(ii+1,jj+1)],top_nodes[(ii,jj+1)]]])
                    nF+=2
                    #bottom
                    facets.append([[ bottom_nodes[(ii,jj)],bottom_nodes[(ii+1,jj)],bottom_nodes[(ii+1,jj+1)]] ])
                    facets.append([[ bottom_nodes[(ii,jj)],bottom_nodes[(ii+1,jj+1)],bottom_nodes[(ii,jj+1)] ]])
                    nF+=2
                else:
                    #top
                    facets.append([[ top_nodes[(ii,jj)],top_nodes[(ii+1,jj)],top_nodes[(ii,jj+1)] ]])
                    facets.append([[ top_nodes[(ii+1,jj)],top_nodes[(ii+1,jj+1)],top_nodes[(ii,jj+1)] ]])
                    nF+=2
                    #bottom
                    facets.append([[ bottom_nodes[(ii,jj)],bottom_nodes[(ii+1,jj)],bottom_nodes[(ii,jj+1)] ]])
                    facets.append([[ bottom_nodes[(ii+1,jj)],bottom_nodes[(ii+1,jj+1)],bottom_nodes[(ii,jj+1)] ]])
                    nF+=2
        self.vertices = np.array(vertices)+self.coords

        self.vertexFlags = np.array([1]*len(vertices))
        self.facets = np.array(facets)
        self.facetFlags = np.array([1]*len(facets))
        self.regions = np.array([self.coords])
        self.regionFlags = np.array([1])


class Rectangle(Shape):
    """
    Class to create a rectangle

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    dim: Optional[array_like]
        Dimensions of the cuboid.
    coords: Optional[array_like]
        Coordinates of the centroid of the shape.
    barycenter: Optional[array_like]
        Coordinates of the barycenter.
    """
    count = 0

    def __init__(self, domain, dim=(0., 0.), coords=(0., 0.), barycenter=None):
        super(Rectangle, self).__init__(domain, nd=2)
        self.__class__.count += 1
        self.name = "rectangle" + str(self.__class__.count)
        self.dim = L, H = dim  # length, height
        self.coords = x, y = np.array(coords)
        self.vertices = np.array([[x-0.5*L, y-0.5*H],
                                  [x+0.5*L, y-0.5*H],
                                  [x+0.5*L, y+0.5*H],
                                  [x-0.5*L, y+0.5*H]])
        self.segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        self.barycenter = np.zeros(3)
        if barycenter is not None:
            self.barycenter[0:2] = barycenter[0:2]
        else:
            self.barycenter[0:2] = coords[0:2]
        self.b_or = np.array([[0., -1.],
                              [1., 0.],
                              [0., 1.],
                              [-1., 0.]])
        self.regions = np.array([[x, y]])
        self.facets = np.array([[[0, 1, 2, 3]]])
        self.boundaryTags = bt = {'y-': 1,
                                  'x+': 2,
                                  'y+': 3,
                                  'x-': 4}
        self.segmentFlags = np.array([bt['y-'], bt['x+'], bt['y+'],
                                      bt['x-']])  # y-, x+, y+, x-
        self.vertexFlags = np.array([bt['y-'], bt['y-'], bt['y+'],
                                     bt['y+']])  # y-, y-, y+, y+
        self.regionFlags = np.array([1])
        self.facetFlags = self.regionFlags
        self.BC = {'y-': self.BC_class(shape=self, name='y-',
                                       b_or=self.b_or, b_i=0),
                   'x+': self.BC_class(shape=self, name='x+',
                                       b_or=self.b_or, b_i=1),
                   'y+': self.BC_class(shape=self, name='y+',
                                       b_or=self.b_or, b_i=2),
                   'x-': self.BC_class(shape=self, name='x-',
                                       b_or=self.b_or, b_i=3)}
        self.BC_list = [self.BC['y-'],
                        self.BC['x+'],
                        self.BC['y+'],
                        self.BC['x-']]
        # self.BC = BCContainer(self.BC_dict)
        self.It = (L**2+H**2)/12

    def setDimensions(self, dim):
        """
        Set dimensions of the shape

        :param dim: new dimensions (list/array)
        """
        self.dim = dim
        L, H = dim
        x, y = self.coords
        self.vertices[:] = [[x-0.5*L, y-0.5*H],
                            [x+0.5*L, y-0.5*H],
                            [x+0.5*L, y+0.5*H],
                            [x-0.5*L, y+0.5*H]]
        self.volume = L*H

class Cylinder(Shape):
    """
    Class to create a cylinder.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    radius: float
        radius of cylinder.
    height: float
        height of cylinder.
    nPoints: int
        number of points to discretize circles of cylinder.
    coords: Optional[array_like]
        Coordinates of the centroid of the shape.
    barycenter: Optional[array_like]
        Coordinates of the barycenter.
    """
    count = 0
    def __init__(self, domain, radius, height, nPoints, coords=(0.,0.,0.), barycenter=None):
        super(Cylinder, self).__init__(domain, nd=3)
        self.__class__.count += 1
        self.name = "Cylinder" + str(self.__class__.count)
        self.radius = radius
        self.height = height
        self.coords = np.array(coords)
        self.barycenter = np.array(barycenter)
        self.nPoints = nPoints
        self.constructShape()

    def constructShape(self):
        h_offset = np.array([0., 0., self.height])
        arc = 2.*np.pi*self.radius/self.nPoints
        ang = arc/self.radius
        vert = []
        facets = []
        segs = []
        for i in range(0, self.nPoints):
            vert += [[self.radius*cos(ang*i),
                      self.radius*sin(ang*i),
                      0]]
            if i > 0:
                segs += [[i-1, i]]
        segs += [[i, 0]]
        segs_bottom = np.array(segs)
        vert_bottom = np.array(vert)
        facets += [[[i for i in range(0, len(vert_bottom))]]]
        vert_top = np.array(vert)+h_offset
        segs_top = np.array(segs)+len(vert)
        nvb = len(vert_bottom)
        facets += [[[i+nvb for i in range(0, len(vert_top))]]]
        for i in range(len(vert_bottom)-1):
            facets += [[[i, i+1, i+nvb+1, i+nvb]]]
        facets += [[[i+1, 0, nvb, i+1+nvb]]]  # last facet
        self.vertices = np.vstack((vert_bottom, vert_top))-h_offset/2.+np.array(self.coords)
        self.segments = np.vstack((segs_bottom, segs_top))
        self.segmentFlags = np.array([1 for i in range(len(segs_bottom))]+[2 for i in range(len(segs_top))])
        self.facets = facets
        self.vertexFlags = np.array([1 for i in range(len(vert_bottom))]+[2 for i in range(len(vert_top))])
        self.facetFlags = np.array([1, 2]+[3 for i in range(len(self.facets)-2)])
        self.boundaryTags = {'z-': 1, 'z+': 2, 'cylinder': 3}
        self.regions = np.array([self.coords])
        self.regionFlags = np.array([1])
        self.volumes = [[[i for i in range(len(self.facets))]]]
        self.BC = {'z-': self.BC_class(shape=self, name='z-'),
                   'z+': self.BC_class(shape=self, name='z+'),
                   'cylinder': self.BC_class(shape=self, name='cylinder')}
        self.BC_list = [self.BC['z-'], self.BC['z+'], self.BC['cylinder']]


class Circle(Shape):
    """
    Class to create a circular shape.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    radius:
        Radius of the circular shape.
    coords: Optional[array_like]
        Coordinates of the centroid of the shape.
    """

    count = 0

    def __init__(self, domain, radius, coords, barycenter, nPoints, boundaryTag=None):
        super(Circle, self).__init__(domain, nd=2)
        self.__class__.count += 1
        self.name = "Circle" + str(self.__class__.count)

        pi=np.pi
        self.radius = radius
        self.coords = xc, yc = np.array(coords)
        self.nPoints = nPoints
        self.arc=2.0*pi*self.radius/self.nPoints
        self.ang=self.arc/self.radius
        verti=[]
        segm=[]

        # points
        for iii in range(0,nPoints):
            vert=np.array([xc+self.radius*cos(self.ang*iii),
                           yc+self.radius*sin(self.ang*iii),
                           ])
            verti.append(vert,)
        self.vertices=np.array(verti)
        nVertices=len(self.vertices)

        # segments
        for jjj in range(1,nVertices):
            seg=[jjj-1, jjj]
            segm.append(seg,)
        seg_last=[nVertices-1,0]
        segm.append(seg_last,)
        self.segments=np.array(segm)

        # facets
        facets=[]
        for kkk in range(nVertices):
            facets.append(kkk)
        self.facets=np.array([[facets]])

        # barycenter, orientations, region
        if barycenter is not None:
            self.barycenter[0:2] = barycenter[0:2]
        else:
            self.barycenter[0:2] = coords[0:2]

        self.b_or = np.array([[1., 0.],
                              ])
        self.regions = np.array([[xc, yc]])

        # boundary tags
        if boundaryTag is None:
            self.boundaryTags = bt = {'circle':1}
            vertFlag=[]
            segmFlag=[]
            for kkk in range(nVertices):
                    vertFlag.append(bt['circle'],)
                    segmFlag.append(bt['circle'],)
            self.vertexFlags = np.array(vertFlag)
            self.segmentFlags = np.array(segmFlag)
            self.facetFlags=np.array([1])
            self.regionFlags = np.array([1])

            # boundary list
            self.BC = {'circle': self.BC_class(shape=self, name='circle',
                                           b_or=self.b_or, b_i=0),
                      }
            self.BC_list = [self.BC['circle'],
                           ]
        else:
            self.boundaryTags = boundaryTag
            bt = boundaryTag
            # should only be one key
            for key in boundaryTag:
                boundaryKey = key
                boundaryFlag = boundaryTag[key]
            vertFlag=[]
            segmFlag=[]
            for kkk in range(nVertices):
                    vertFlag.append(boundaryFlag,)
                    segmFlag.append(boundaryFlag,)
            self.vertexFlags = np.array(vertFlag)
            self.segmentFlags = np.array(segmFlag)
            self.facetFlags=np.array([boundaryFlag])
            self.regionFlags = np.array([boundaryFlag])

            # boundary list
            self.BC = {boundaryKey: self.BC_class(shape=self, name=boundaryKey,
                                           b_or=self.b_or, b_i=0),
                      }
            self.BC_list = [self.BC[boundaryKey],
                           ]

        # set the inertia
        self.It = pi*(radius**4)/2.



class CustomShape(Shape):
    """
    Class to create a custom 2D or 3D shape

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    barycenter: Optional[array_like]
        Coordinates of the barycenter.
    vertices: array_like
        Array of vertex coordinates.
    vertexFlags: array_like
        Array of vertex flags (used for boundary conditions)
    segments: array_like
        Array of segments (each defined by indice of 2 vertex).
    segmentFlags: array_like
        Array of segment flags (used for boundary conditions)
    facetss: array_like
        Array of facets (defined by clockwise or counterclockwise loop of
        vertices).
    facetFlags: array_like
        Array of facet flags (used for boundary conditions)
    vertices: array_like
        Array of region coordinates.
    regionFlags: array_like
        Array of region flags (used for boundary conditions)
    volumes: array_like
        Surface loops describing volumes (used for gmsh)
    holes: array_like
        Array of holes coordinates (unmeshed regions; for triangle/tetgen only)
    holes: array_like
        Array of holes index (volume or facet; unmeshed regions; for gmsh only)
    boundaryTags: dict
        Dictionary of flags (int) as keys, and tags (e.g. string) for BC.
    boundaryOrientations: Optional[array_like]
        Array of orientation of boundaries. Can be used for BC.
    """
    count = 0

    def __init__(self, domain, barycenter=None, vertices=None,
                 vertexFlags=None, segments=None, segmentFlags=None,
                 facets=None, facetFlags=None, holes=None, holes_ind=None,
                 regions=None, regionFlags=None, volumes=None,
                 boundaryTags=None, boundaryOrientations=None):
        super(CustomShape, self).__init__(domain, nd=len(vertices[0]))
        self.__class__.count += 1
        self.name = "custom" + str(self.__class__.count)
        self._checkFlags(list(boundaryTags.values()))
        self.boundaryTags = boundaryTags
        self.vertices = np.array(vertices)
        self.vertexFlags = np.array(vertexFlags)
        if segments is not None:
            self.segments = np.array(segments)
            self.segmentFlags = np.array(segmentFlags)
        if facets is not None:
            self.facets = facets
            self.facetFlags = np.array(facetFlags)
        if holes is not None:
            self.holes = np.array(holes)
        if holes_ind is not None:
            self.holes_ind = np.array(holes_ind)
        if volumes is not None:
            self.volumes = volumes
        if regions is not None:
            self._checkFlags(regionFlags)
            self.regions = np.array(regions)
            self.regionFlags = np.array(regionFlags)
        self.BC = {}
        self.BC_list = [None]*len(boundaryTags)
        b_or = [None]*len(boundaryTags)
        self.b_or = b_or
        for tag, flag in boundaryTags.items():
            b_i = flag-1  # start at index 0
            # hack for SWFlow BCs
            if (b_i==98):
                b_i=len(boundaryTags)-1
            if boundaryOrientations is not None:
                b_or[b_i] = boundaryOrientations[tag]
            self.BC[tag] = self.BC_class(shape=self, name=tag, b_or=b_or, b_i=b_i)
            self.BC_list[b_i] = self.BC[tag]
        # self.BC = BCContainer(self.BC_dict)
        if barycenter is not None:
            self.barycenter = np.array(barycenter)


class ShapeSTL(Shape):
    """
    Class to extract geometrical information from STL file

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    filename: string
        Name of the stl file.
    """
    count = 0

    def __init__(self, domain, filename):
        super(ShapeSTL, self).__init__(domain, nd=3)
        self.__class__.count += 1
        self.name = 'STL'+str(self.__class__.count)
        self.filename = filename
        self.vertices, self.facets, self.facetnormals,self.vertexFlags,self.facetFlags,self.boundaryTags = getInfoFromSTL(self.filename)

#        self.facetFlags = np.ones(len(self.facets))
#        self.vertexFlags = np.ones(len(self.vertices))
        self.volumes = [[[i for i in range(len(self.facets))]]]
        #        self.boundaryTags = {'stl': 1}
        self.BC={}
        self.BC_list=[]
        for key,value in self.boundaryTags.items():
            self.BC[key] = self.BC_class(shape=self, name=key)
            self.BC_list += [self.BC[key]]
        # self.BC = BCContainer(self.BC_dict)

def getInfoFromSTL(filename):
    """
    Extracts information from STL file and converts it to a Proteus friendly
    format. Duplicate vertices and segments are removed during the process,
    so the shape is ready for meshing.

    Parameters
    ----------
    filename: name of STL file

    Returns
    -------
    vertices: array_like
        Array of vertices that define STL shape (duplicates removed)
    facets: array_like
        Array of facets (loops of 3 vertices)
    facetnormals: array_like
        normal vertors of each facet
    vertexFlags: array_like
        vertex flags according to boundary ownership
    facetFlags: array_like
        facet flags according to boundary ownership
    boundaryTags: dictionary
        Dictionary of boundaries. Has the same number and name as the stl blocks contained
        in the stl file
    """
    file = open(filename, 'r')
    facetnormals = []
    facet = []
    facets = []
    vertices = []
    vFlag = 0
    vertexFlags=[]
    facetFlags=[]
    bFlag = 0
    bTags = {}
    for line in file:
        if ("solid" in line) and (not "endsolid" in line) :
            word_list = line.split()
            name = word_list[1]
            bFlag+=1
            bTags[name]=int(bFlag)
        elif "vertex" in line:
            word_list = line.split()
            vertex = (word_list[1], word_list[2], word_list[3])
            vertices += [vertex]
            facet += [vFlag]
            vFlag += 1
            vertexFlags += [bFlag]
        if "facet normal" in line:
            word_list = line.split()
            facetnormals += [[word_list[2], word_list[3], word_list[4]]]
        elif "endfacet" in line:
            facets += [[facet]]
            facetFlags+=[bFlag]
            facet = []

    file.close()
    # vertices_u, inverse = np.unique(vertices, return_inverse=True)
    vertices = np.array(vertices).astype(float)
    facets = np.array(facets).astype(int)
    vertexFlags = np.array(vertexFlags).astype(int)
    facetFlags =np.array(facetFlags).astype(int)

    vertices, inverse, indices = unique_rows(vertices)
    facets = inverse[facets]
    facetnormals = np.array(facetnormals).astype(float)

    indices = np.array(indices).astype(int)
    vertexFlags = vertexFlags[indices]
    return vertices, facets, facetnormals,vertexFlags,facetFlags,bTags

def unique_rows(arr):
    arr = np.array(arr)
    ca = np.ascontiguousarray(arr).view([('', arr.dtype)] * arr.shape[1])
    unique, indices, inverse = np.unique(ca, return_index=True, return_inverse=True)
    # NumPy >=2.0 shapes `inverse` like the (N, 1) structured-dtype view `ca`
    # instead of the flat (N,) array older NumPy returned; flatten it back so
    # downstream fancy-indexing (e.g. inverse[facets]) keeps its expected rank.
    inverse = inverse.reshape(-1)
    # counts = np.bincount(inverse)
    # sort_indices = np.argsort(counts)[::-1]
    # sorted_arr = arr[indices[sort_indices]]
    # sorted_count = counts[sort_indices]
    return (arr[indices], inverse ,indices)

class BCContainer(object):
    """
    Creates a class from a dictionary (keys become class variable names)
    """
    def __init__(self, BC_dict):
        self.__dict__ = BC_dict


# --------------------------------------------------------------------------- #
# -------------------------SPATIAL TOOLS FOR SHAPES-------------------------- #
# --------------------------------------------------------------------------- #

def rotation2D(points, rot, pivot=(0., 0.)):
    """
    Rotates a set of points/vertices/vectors around a pivotal point in 2D.

    Parameters
    ----------
    points: array_like
        Array of point coordinates to rotate.
    rot: float
        Angle of rotation.
    pivot: array_like
        Pivotal point around which the set of points will be rotated.

    Returns
    -------
    points_rot: array_like
        Rotated set of points.
    """
    # function could be optimized
    points = np.array(points)
    rot = float(rot)
    # get coordinates for translation
    x, y = pivot
    # translation matrix
    T = np.array([[1,   0,    0],
                  [0,   1,    0],
                  [-x,  -y,   1]])
    # rotation matrices
    R = np.array([[cos(rot),  sin(rot),  0],
                  [-sin(rot), cos(rot),  0],
                  [0,         0,         1]])
    # full transformation matrix
    M = reduce(np.dot, [T, R, np.linalg.inv(T)])
    # transform points (check also if it is only a 1D array or 2D)
    if points.ndim > 1:
        points_rot = np.ones((len(points), 3))
        points_rot[:, :-1] = points
        points_rot = np.dot(points_rot, M)  # matrix dot product on each vector
        points_rot = points_rot[:, :-1]
    else:
        points_rot = np.ones(3)
        points_rot[:-1] = points
        points_rot = np.dot(points_rot, M)  # matrix dot product on each vector
        points_rot = points_rot[:-1]
    return points_rot


def rotation3D(points, rot, axis=(0., 0., 1.), pivot=(0., 0., 0.)):
    """
    Rotates a set of points/vertices/vectors around a pivotal point in 3D.

    Parameters
    ----------
    points: array_like
        Array of point coordinates to rotate.
    rot: float
        Angle of rotation.
    axis: array_like
        Axis of rotation.
    pivot: array_like
        Pivotal point around which the set of points will be rotated.

    Returns
    -------
    points_rot: array_like
        Rotated set of points.
    """
    # function could be optimized
    points = np.array(points)
    rot = float(rot)
    # get coordinates for translation
    x, y, z = pivot
    # make axis a unity vector
    axis = np.array(axis)
    r = np.linalg.norm(axis)
    axis = axis/r
    # get values for rotation matrix
    cx, cy, cz = axis
    d = sqrt(cy**2+cz**2)
    # rotation matrices
    if d != 0:
        Rx = np.array([[1,         0,        0,    0],
                       [0,         cz/d,     cy/d, 0],
                       [0,         -cy/d,    cz/d, 0],
                       [0,         0,        0,    1]])
    else:  # special case: rotation axis aligned with x axis
        Rx = np.array([[1,         0,        0,    0],
                       [0,         1,        0,    0],
                       [0,         0,        1,    0],
                       [0,         0,        0,    1]])
    Ry = np.array([[d,         0,        cx, 0],
                   [0,         1,        0,   0],
                   [-cx,       0,        d,   0],
                   [0,         0,        0,   1]])
    Rz = np.array([[cos(rot),  sin(rot), 0,   0],
                   [-sin(rot), cos(rot), 0,   0],
                   [0,         0,        1,   0],
                   [0,         0,        0,   1]])
    # translation matrix
    T = np.array([[1,  0,  0,  0],
                  [0,  1,  0,  0],
                  [0,  0,  1,  0],
                  [-x, -y, -z, 1]])
    # full transformation matrix
    inv = np.linalg.inv
    M = reduce(np.dot, [T, Rx, Ry, Rz, inv(Ry), inv(Rx), inv(T)])
    if points.ndim > 1:
        points_rot = np.ones((len(points), 4))
        points_rot[:, :-1] = points
        points_rot = np.dot(points_rot, M)  # matrix dot product on each vector
        points_rot = points_rot[:, :-1]
    else:
        points_rot = np.ones(4)
        points_rot[:-1] = points
        points_rot = np.dot(points_rot, M)  # matrix dot product on each vector
        points_rot = points_rot[:-1]
    return points_rot


def _assembleGeometry(domain, BC_class):
    """
    Assembles all the geometrical informations of the shapes attached to a
    domain.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    """
    # reinitialize geometry of domain
    domain.useSpatialTools = True
    domain.vertices = []
    domain.vertexFlags = []
    domain.segments = []
    domain.segmentFlags = []
    domain.facets = []
    domain.facetFlags = []
    domain.holes = []
    domain.regions = []
    domain.regionFlags = []
    domain.volumes = []
    domain.boundaryTags = {}
    domain.reversed_boundaryTags = {}
    # BC at flag 0
    bc0 = BC_class(nd=domain.nd)
    domain.BCbyFlag[0] = bc0
    domain.bc = [bc0]
    # domain.bc[0].setNonMaterial()
    # barycenter at flag 0
    domain.barycenters = np.array([[0., 0., 0.]])
    start_flag = 0
    start_vertex = 0
    for shape in domain.shape_list:
        # --------------------------- #
        # ----- DOMAIN GEOMETRY ----- #
        # --------------------------- #
        start_flag = shape.start_flag = len(domain.bc)-1
        start_vertex = shape.start_vertex = len(domain.vertices)
        start_segment = shape.start_segment = len(domain.segments)
        start_facet = shape.start_facet = len(domain.facets)
        if shape.boundaryTags:
            shape.boundaryTags_global = {}
            for tag, value in shape.boundaryTags.items():
                new_tag = shape.name+'_'+str(tag)
                new_flag = value+start_flag
                domain.boundaryTags[new_tag] = new_flag
                shape.boundaryTags_global[tag] = new_flag
                domain.reversed_boundaryTags[new_flag] = new_tag
                domain.BCbyFlag[new_flag] = shape.BC[tag]
                domain.BC[new_tag] = shape.BC[tag]
        if domain.regionFlags:
            start_rflag = max(domain.regionFlags)
        else:
            start_rflag = 0
        domain.bc += shape.BC_list
        # making copies of shape properties before operations/modifications
        vertices = copy.deepcopy(shape.vertices)
        vertexFlags = copy.deepcopy(shape.vertexFlags)
        if shape.segments is not None:
            segments = copy.deepcopy(shape.segments)
        if shape.facets is not None:
            facets = copy.deepcopy(shape.facets)
        # deleting duplicate vertices and updating segment/facets accordingly
        del_v = 0
        for i_s, vertex in enumerate(shape.vertices):
            if vertex.tolist() in domain.vertices:
                vertices = np.delete(vertices, i_s-del_v, axis=0)
                verticesFlags = np.delete(vertexFlags, i_s-del_v)
                i_s -= del_v
                del_v += 1
                i_d = domain.vertices.index(vertex.tolist())
                if shape.segments is not None:
                    for i in np.nditer(segments, op_flags=['readwrite']):
                        if i > i_s:
                            i[...] -= 1
                        elif i == i_s:
                            i[...] = i_d-start_vertex
                if shape.facets is not None:
                    for facet in shape.facets:
                        for subfacet in facet:
                            for ind, ver in enumerate(subfacet):
                                if ver > i_s:
                                    subfacet[ind] -= 1
                                elif ver == i_s:
                                    subfacet[ind] = i_d-start_vertex
        # adding shape geometry to domain
        domain.vertices += vertices.tolist()
        domain.vertexFlags += (vertexFlags+start_flag).tolist()
        shape.vertexFlags_global = vertexFlags+start_flag
        barycenters = np.array([shape.barycenter for bco in shape.BC_list])
        domain.barycenters = np.append(domain.barycenters, barycenters, axis=0)
        if shape.segments is not None:
            domain.segments += (segments+start_vertex).tolist()
            shape.segments_global = segments+start_vertex
            domain.segmentFlags += (shape.segmentFlags+start_flag).tolist()
            shape.segmentFlags_global = segments+start_flag
        if shape.regions is not None:
            domain.regions += (shape.regions).tolist()
            domain.regionFlags += (shape.regionFlags+start_rflag).tolist()
            shape.regionFlags_global = shape.regionFlags+start_flag
        if shape.facets is not None:
            if type(facets) is np.ndarray:
                facets = facets.tolist()
            else:
                facets = copy.deepcopy(shape.facets)
            for i, facet in enumerate(facets):
                for j, subf in enumerate(facet):
                    for k, v_nb in enumerate(subf):
                        facets[i][j][k] = v_nb+shape.start_vertex
            domain.facets += facets
            shape.facets_global = facets
            if shape.nd == 2:  # facet flags are actually region flags if 2D
                domain.facetFlags += (shape.regionFlags+start_rflag).tolist()
                shape.facetFlags_global = shape.regionFlags+start_rflag
            elif shape.nd == 3:
                domain.facetFlags += (shape.facetFlags+start_flag).tolist()
                shape.facetFlags_global = shape.facetFlags+start_flag
        if shape.holes is not None:
            domain.holes += (shape.holes).tolist()

    # 2D holes (only for gmsh)
    domain.holes_ind = []
    if domain.nd == 2 and shape.facets is not None:
        domain.facets = []
        for shape in domain.shape_list:
            if shape.holes_ind is not None:
                    domain.holes_ind += (np.array(shape.holes_ind)+shape.start_facet).tolist()
            if shape.facets is not None:
                facets = deepcopy(shape.facets)
                if type(facets) is np.ndarray:
                    facets = facets.tolist()
                for i, facet in enumerate(facets):
                    for j, subf in enumerate(facet):
                        for k, v_nb in enumerate(subf):
                            facets[i][j][k] = v_nb+shape.start_vertex
                # facets = (shape.facets+shape.start_vertex).tolist()
                f_to_remove = []
                f_to_add = []
                facets_again = deepcopy(shape.facets)
                if type(facets_again) is np.ndarray:
                    facets_again = facets_again.tolist()
                for i, facet in enumerate(facets_again):
                    if i in list(shape.children.keys()):
                        for child in shape.children[i]:
                            child_seg = {}
                            child_facs = deepcopy(child.facets)
                            if type(child_facs) is np.ndarray:
                                child_facs = child_facs.tolist()
                            for child_fac in child_facs:
                                fac = child_fac[0]
                                for j, v in enumerate(fac):
                                    if (fac[j-1], fac[j]) in child_seg:
                                        fn = child_seg[(fac[j-1], fac[j])]
                                        f_to_merge = child.facets[fn][0]
                                        # reverse list
                                        f_to_merge = f_to_merge[::-1]
                                        # shift lists
                                        f_to_merge = list(deque(f_to_merge).rotate(-f_to_merge.index(fac[j-1])))
                                        fac = list(deque(fac).rotate(-fac.index(fac[j])))
                                        new_f = fac[:-1]+f_to_merge[:-1]
                                        new_f = fac[:fac.index(fac[j-1])]+f_to_merge
                                        f_to_remove += [fn]
                                        f_to_add += [new_f]
                                    elif (fac[j], fac[j-1]) in child_seg:
                                        fn = child_seg[(fac[j], fac[i-1])]
                                        f_to_merge = child.facets[fn][0]
                                        # shift lists
                                        f_to_merge = list(deque(f_to_merge).rotate(-f_to_merge.index(fac[j-1])))
                                        fac = list(deque(fac).rotate(-fac.index(fac[j])))
                                        new_f = fac[:-1]+f_to_merge[:-1]
                                        new_f = fac[:fac.index(fac[j-1])]+f_to_merge
                                        f_to_remove += [fn, j]
                                        f_to_add += [new_f]
                                    else:
                                        child_seg[(fac[j-1], fac[j])] = j
                            child_facets = deepcopy(child.facets)
                            if type(child_facets) is np.ndarray:
                                child_facets = child_facets.tolist()
                            facets2 = [f[0] for k, f in enumerate(child_facets) if k not in f_to_remove]
                            if f_to_add:
                                facets2 += [f_to_add]
                        new_facets = facets2
                        for facet_i in range(len(new_facets)):
                            facet = new_facets[facet_i]
                            for v_i in range(len(facet)):
                                print(new_facets[facet_i][v_i])
                                new_facets[facet_i][v_i] += child.start_vertex
                        facets[i] += new_facets
                domain.facets += facets

    # 3D holes (only for gmsh)
    elif domain.nd == 3 and shape.volumes is not None:
        for shape in domain.shape_list:
            shape.start_volume = len(domain.volumes)
            if shape.holes_ind is not None:
                domain.holes_ind += (np.array(shape.holes_ind)+shape.start_volume).tolist()
            if shape.volumes is not None:
                volumes = copy.deepcopy(shape.volumes)
                for volume in volumes:
                    for subvolume in volume:
                        for i in range(len(subvolume)):
                            subvolume[i] += shape.start_facet
                volumes_to_remove = []
                for i, volume in enumerate(volumes):
                    # add holes to volumes
                    if i in shape.children:
                        for child in shape.children[i]:
                            child_vols = []
                            child_facets = []  # facets in volumes
                            for child_vol in child.volumes:
                                # don't need holes of child volumes, only outer shells:
                                child_vol_copy = copy.deepcopy(child_vol[0])
                                for i in range(len(child_vol_copy)):
                                    child_vol_copy[i] += child.start_facet
                                child_vols += [child_vol_copy]
                            if len(child_vols) > 1:
                                # merge volumes that share a facet
                                inter = np.intersect1d(*child_vols)
                                new_child_vols = list(child_vols)
                                for f in inter:
                                    ind_to_remove = set()
                                    merged = []
                                    for ind, vol in enumerate(child_vols):
                                        if f in vol:
                                            ind_to_remove.add(ind)
                                            merged += vol
                                    for ind in sorted(ind_to_remove, reverse=True):
                                        child_vols.remove(ind)
                                    merged.remove(f)
                                    child_vols += [merged]
                            volume += child_vols
                    if shape.holes_ind is not None and i in shape.holes_ind:
                        volumes_to_remove += [i]
                volumes2 = [vol for i, vol in enumerate(volumes) if i not in volumes_to_remove]
                domain.volumes += volumes2

    domain.getBoundingBox()


def _generateMesh(domain):
    """
    Generates tetgen mesh of domain

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    """
    # --------------------------- #
    # ----- MESH GENERATION ----- #
    # --------------------------- #
    from proteus import Comm
    from proteus import MeshTools
    comm = Comm.get()
    mesh = MeshTools.getMeshOptions(domain)
    if comm.isMaster():
        if mesh.outputFiles['poly'] is True:
            domain.writePoly(mesh.outputFiles_name)
        if mesh.outputFiles['ply'] is True:
            domain.writePLY(mesh.outputFiles_name)
        if mesh.outputFiles['asymptote'] is True:
            domain.writeAsymptote(mesh.outputFiles_name)
        if mesh.outputFiles['geo'] is True or mesh.use_gmsh is True:
            domain.writeGeo(mesh.outputFiles_name)
    else:
        domain.polyfile=mesh.outputFiles_name
    mesh.setTriangleOptions()


def getGmshPhysicalGroups(geofile):
    boundaryTags = {}
    import re
    # gmsh_cmd = "gmsh {0:s} -0".format(geofile)
    # check_call(gmsh_cmd, shell=True)
    with open(geofile, 'r') as f:
        for line in f:
            if line.startswith("Physical "):
                words = line.lstrip().split('=', 1)
                tagflag = re.search(r'\((.*?)\)', words[0]).group(1).split(',')
                if len(tagflag) == 2:
                    tag = str(tagflag[0][2:-2])
                    boundaryTags[tag] = None  # add empty BC holder
                    flag = int(tagflag[1])
                    print(tagflag)
                else:
                    try:
                        flag = tag = int(tagflag[0])
                    except:
                        flag = tag = tagflag[0]
                boundaryTags[tag] = flag  # add empty BC holder
    return boundaryTags

def extrude2Dto3D(extrusion, vertices, segments, facets, regions=None):
    """
    Extrude 2D geometry attributes in 3D

    Parameters
    ----------
    extrusion: array_like
        value of the extrusion (e.g. [1.4, 0., 0.])
    vertices: array_like
        list of vertices of the 2D shape
    segments: array_like
        list of segments of the 2D shape
    facets: array_like
        list of facets of the 2D shape
    regions: array_like
        list of regions of the 2D shape
    """
    nv = len(vertices)
    ns = len(segments)
    vertices = np.array(vertices)
    vertices = np.vstack((vertices, vertices+extrusion))
    segments_side = [[i, i+nv] for i in range(nv)]
    segments_extruded = copy.deepcopy(segments)
    for segment in segments_extruded:
        segment[0] += ns
        segment[1] += ns
    segments = np.vstack((segments, segments_side, segments_extruded))
    facets_side = [[[i, i+1, i+nv+1, i+nv]] for i in range(nv)]
    facets_side[-1][0][1] = 0
    facets_side[-1][0][2] = nv
    facets_extruded = copy.deepcopy(facets)
    for facet in facets_extruded:
        for subfacet in facet:
            for i in range(len(subfacet)):
                subfacet[i] += nv
    facets = facets+facets_side+facets_extruded
    regions_extruded = copy.deepcopy(regions)
    if regions is not None:
        for region in regions_extruded:
            region += extrusion/2.
    return vertices, segments, facets, regions_extruded


# -----------------------------------------------------------------------------
# TWO-PHASE FLOW (RANS) SHAPES, formerly proteus.SpatialTools (merged here by C49 R1)
#
# Importing this module makes every predefined shape a ShapeRANS (the bases are reassigned below), as importing
# proteus.SpatialTools did, so a shape of either kind has the rigid-body, relaxation-zone, gauge and
# auxiliary-variable methods; the boundary conditions of a shape are BC_RANS. assembleDomain (below) takes the
# two-phase path when any shape of the domain is a ShapeRANS and the generic one when none is.
# -----------------------------------------------------------------------------

class ShapeRANS(Shape):
    """
    Base/super class of all shapes. Sets the boundary condition class to
    proteus.BoundaryConditions.BC_RANS.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    nd: Optional[int]
        Number of dimensions of the shape. If not set, will take the number of
        dimensions of the domain.
    """

    def __init__(self, domain, nd):
        super(ShapeRANS, self).__init__(domain, nd, BC_class=bc.BC_RANS)
        self.mass = None
        self.density = None
        self.free_x = (1, 1, 1)
        self.free_r = (1, 1, 1)
        self.record_values = False
        self.zones = {}  # for absorption/generation zones
        self.auxiliaryVariables = {}  # list of auxvar attached to shape
        self.It = None  # inertia tensor

    def _attachAuxiliaryVariable(self, key, auxvar=None, gauge=None):
        """
        Attaches an auxiliary variable to the auxiliaryVariables dictionary of
        the shape (used in buildDomain function)

        Parameters
        ----------
        key: string
            Dictionary key defining the auxiliaryVariable to attach
        auxvar:
            auxiliaryVariable to associate with key

        gauge: Gauges

        Notes
        -----
        This function is called automatically when using other functions to set
        auxiliaryVariables and should not be used manually.
        """
        if key not in self.auxiliaryVariables:
            if key == 'RigidBody':
                self.auxiliaryVariables[key] = auxvar
                if self.holes is None:
                    self.holes = np.array([self.barycenter[:self.nd]])
            elif key == 'WallFunction':
                self.auxiliaryVariables[key] = auxvar
            elif key == 'kWallFunction':
                self.auxiliaryVariables[key] = auxvar
            elif key == 'RelaxZones':
                self.auxiliaryVariables[key] = self.zones
            elif str(key).startswith('Gauge_'):
                self.auxiliaryVariables[key] = [gauge]
            else:
                logEvent("auxiliaryVariable key: "
                         "{key} not recognized.".format(key=str(key)), level=1)
        elif str(key).startswith('Gauge_'):
            if gauge not in self.auxiliaryVariables[key]:
                self.auxiliaryVariables[key] += [gauge]
            else:
                logEvent(
                    "Attempted to put identical "
                    "gauge at key: {key}".format(key=str(key)), level=1)
        else:
            logEvent("Key {key} is already attached.".format(key=str(key)),
                     level=1)

    def attachPointGauges(self, model_key, gauges, activeTime=None,
                          sampleRate=0,
                          fileName='point_gauges.csv'):
        """Attaches Point Gauges (in the Proteus/Gauges.py style) to the shape.

        Parameters
        ----------
        model_key: string
            Label of the model to use as a key for selecting particular gauges.
        See proteus Gauges.py PointGauges class for the remaining parameters.
        """
        new_gauges = Gauges.PointGauges(gauges, activeTime, sampleRate,
                                        fileName)
        self._attachAuxiliaryVariable('Gauge_' + model_key,
                                      gauge=new_gauges)

    def attachLineGauges(self, model_key, gauges, activeTime=None,
                         sampleRate=0,
                         fileName='line_gauges.csv'):
        """Attaches Line Gauges (in the Proteus/Gauges.py style) to the shape.

        Parameters
        ----------
        model_key: string
            Label of the model to use as a key for selecting particular gauges.
        See proteus Gauges.py LineGauges class for the remaining parameters.
        """
        new_gauges = Gauges.LineGauges(gauges, activeTime, sampleRate,
                                       fileName)
        self._attachAuxiliaryVariable('Gauge_' + model_key,
                                      gauge=new_gauges)

    def attachLineIntegralGauges(self, model_key, gauges, activeTime=None,
                                 sampleRate=0,
                                 fileName='line_integral_gauges.csv'):
        """Attaches Line Integral Gauges (in the Proteus/Gauges.py style).

        Parameters
        ----------
        model_key: string
            Label of the model to use as a key for selecting particular gauges.
        See proteus Gauges.py LineIntegralGauges class for the remaining parameters.
        """
        new_gauges = Gauges.LineIntegralGauges(gauges, activeTime,
                                               sampleRate, fileName)
        self._attachAuxiliaryVariable('Gauge_' + model_key,
                                      gauge=new_gauges)

    def setTank(self):
        """
        Sets tank boundary conditions (for moving domain).
        """
        for boundcond in self.BC_list:
            boundcond.setTank()

    def setTurbulentWall(self, wall):
        """
        Sets a turbulent wall as an object to be attacched to auxiliaryVariable.
        The objects has to be defined with WallFunction class.

        Parameters
        ----------
        wall: list of WallFunction class object
        """

        auxvar = wall
        self._attachAuxiliaryVariable('WallFunction', auxvar)

    def setTurbulentKWall(self, kWall):
        """
        Sets a turbulent wall as an object to be attacched to auxiliaryVariable.
        The objects has to be defined with WallFunction class.

        Parameters
        ----------
        kWall: list of WallFunction class object for kappa
        """

        auxvar = kWall
        self._attachAuxiliaryVariable('kWallFunction', auxvar)

    def setAbsorptionZones(self, flags, epsFact_porous, center, orientation,
                           dragAlpha, vert_axis=None,dragBeta=0.,
                           porosity=1.):
        """
        Sets a region (given the local flag) to an absorption zone

        Parameters
        ----------
        dragAlpha: Optional
            Relaxation zone coefficient.
        flags: array_like, int
            Local flags of the region. Can be an integer or a list.
        epsFact_porous: float
            Half of absorption zone (region) length (used for blending func).
        center: array_like
            Coordinates of the center of the absorption zone.
        orientation: array_like
            Orientation vector pointing TOWARDS incoming waves.
        vert_axis: Optional[int]
            index of vertical position vector (x:0, y:1, z:2), must always be
            aligned with gravity. If not set, will be 1 in 2D (y), 2 in 3D (z).
        dragBeta: Optional[float]
            Relaxation zone coefficient.
        porosity: Optional[float]
            Relaxation zone coefficient.
        """
        self._attachAuxiliaryVariable('RelaxZones')
        waves = None
        wind_speed = np.array([0., 0., 0.])
        if isinstance(flags, numbers.Integral):
            flags = [int(flags)]
            epsFact_porous = [epsFact_porous]
            center = np.array([center])
            orientation = np.array([orientation])
            dragAlpha = [dragAlpha]
            dragBeta = [dragBeta]
            porosity = [porosity]
        for i, flag in enumerate(flags):
            self._checkNd(center[i])
            self._checkNd(orientation[i])
            ori = get_unit_vector(orientation[i])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='absorption',
                                                 orientation=ori,
                                                 center=center[i],
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous[i],
                                                 dragAlpha=dragAlpha[i],
                                                 dragBeta=dragBeta[i],
                                                 porosity=porosity[i],
                                                 vert_axis=vert_axis)

    def setGenerationZones(self, flags, epsFact_porous, center, orientation,
                           waves, dragAlpha,vert_axis=None,
                           wind_speed=(0., 0., 0.), dragBeta=0.,
                           porosity=1., smoothing=0.):
        """
        Sets a region (given the local flag) to a generation zone

        Parameters
        ----------
        flags: array_like, int
            Local flags of the region. Can be an integer or a list.
        epsFact_porous: float
            Half of absorption zone (region) length (used for blending func).
        center: array_like
            Coordinates of the center of the absorption zone.
        orientation: array_like
            Orientation vector pointing ACCORDING WITH the direction of generated waves.
        waves: proteus.WaveTools
            Class instance of wave generated from proteus.WaveTools.
        dragAlpha: Optional[float]
            Relaxation zone coefficient.
        vert_axis: Optional[int]
            index of vertical position vector (x:0, y:1, z:2), must always be
            aligned with gravity. If not set, will be 1 in 2D (y), 2 in 3D (z).
        wind_speed: Optional[array_like]
            Speed of wind in generation zone (default is (0., 0., 0.))
        dragBeta: Optional[float]
            Relaxation zone coefficient.
        porosity: Optional[float]
            Relaxation zone coefficient.
        """
        self._attachAuxiliaryVariable('RelaxZones')
        if isinstance(flags, numbers.Integral):
            flags = [int(flags)]
            epsFact_porous = [epsFact_porous]
            center = np.array([center])
            orientation = np.array([orientation])
            waves = [waves]
            wind_speed = np.array([wind_speed])
            dragAlpha = [dragAlpha]
            dragBeta = [dragBeta]
            porosity = [porosity]
            smoothing = [smoothing]
        for i, flag in enumerate(flags):
            self._checkNd(center[i])
            self._checkNd(orientation[i])
            ori = get_unit_vector(orientation[i])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='generation',
                                                 orientation=ori,
                                                 center=center[i],
                                                 waves=waves[i],
                                                 wind_speed=wind_speed[i],
                                                 epsFact_porous=epsFact_porous[i],
                                                 dragAlpha=dragAlpha[i],
                                                 dragBeta=dragBeta[i],
                                                 porosity=porosity[i],
                                                 smoothing=smoothing[i],
                                                 vert_axis=vert_axis)

    def setPorousZones(self, flags, dragAlpha, dragBeta,
                       porosity):
        """
        Sets a region (given the local flag) to a porous zone

        Parameters
        ----------
        flags: array_like, int
            Local flags of the region. Can be an integer or a list.
        dragAlpha: float
            Darcy-type coefficient
        dragBeta: float
            Forchheimer-type coefficient
        porosity: float
            Porosity
        """
        self._attachAuxiliaryVariable('RelaxZones')
        if isinstance(flags, numbers.Integral):
            flags = [int(flags)]
            dragAlpha = [dragAlpha]
            dragBeta = [dragBeta]
            porosity = [porosity]
        for i, flag in enumerate(flags):
            # note for porous zone:
            # epsFact_porous = q_phi_sponge, --> Hs always equal to 1.
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='porous',
                                                 orientation=None,
                                                 center=None,
                                                 waves=None,
                                                 wind_speed=None,
                                                 epsFact_porous=1.,
                                                 dragAlpha=dragAlpha[i],
                                                 dragBeta=dragBeta[i],
                                                 porosity=porosity[i])

# -----------------------------------------------------------------------------
# ADDING FUNCTIONALITY TO SHAPE FROM proteus.SpatialTools
# -----------------------------------------------------------------------------


# reassigning base/super class to access all functions from ShapeRANS and Shape
Rectangle.__bases__ = (ShapeRANS,)
Cuboid.__bases__ = (ShapeRANS,)
Sphere.__bases__ = (ShapeRANS,)
Cylinder.__bases__ = (ShapeRANS,)
CustomShape.__bases__ = (ShapeRANS,)
ShapeSTL.__bases__ = (ShapeRANS,)
Circle.__bases__ = (ShapeRANS,)

# adding extra functionality to predefined shapes


def _CuboidsetInertiaTensor(self):
    """
    Sets the inertia tensor of the cuboid
    (!) should not be used manually
    """
    L, W, H = self.dim
    self.It = [[(W**2.+H**2.)/12., 0, 0],
               [0, (L**2.+H**2.)/12., 0],
               [0, 0, (W**2.+L**2.)/12.]]


Cuboid._setInertiaTensor = _CuboidsetInertiaTensor


def _RectanglesetInertiaTensor(self):
    """
    Sets the inertia tensor of the rectangle
    (!) should not be used manually
    """
    L, H = self.dim
    self.It = (L**2+H**2)/12


Rectangle._setInertiaTensor = _RectanglesetInertiaTensor


# -----------------------------------------------------------------------------
# DEFINING NEW SHAPES TYPES
# -----------------------------------------------------------------------------

class Tank3D(ShapeRANS):
    """
    Class to create a 3D tank (cuboidal shape).

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    dim: Optional[array_like]
        Dimensions of the cuboid.
    coords: Optional[array_like]
        Coordinates of the centroid of the shape.
    """
    count = 0

    def __init__(self, domain, dim=(0., 0., 0.), coords=None):
        super(Tank3D, self).__init__(domain, nd=3)
        self.__class__.count += 1
        self.name = "tank3d" + repr(self.__class__.count)
        if coords is None:
            self.coords = np.array(dim)/2.
            self.from_0 = True
        else:
            self.coords = coords
            self.from_0 = False
        self.holes = None
        self.boundaryTags = {'z-': 1,
                             'x-': 2,
                             'y+': 3,
                             'x+': 4,
                             'y-': 5,
                             'z+': 6,
                             'sponge': 7,
                             'wall': 8}
        self.b_or = np.array([[0.,  0., -1.],
                              [-1., 0.,  0.],
                              [0.,  1.,  0.],
                              [1.,  0.,  0.],
                              [0., -1.,  0.],
                              [0.,  0.,  1.]])
        self.BC = {'z-': self.BC_class(shape=self, name='z-',
                                       b_or=self.b_or, b_i=0),
                   'x-': self.BC_class(shape=self, name='x-',
                                       b_or=self.b_or, b_i=1),
                   'y+': self.BC_class(shape=self, name='y+',
                                       b_or=self.b_or, b_i=2),
                   'x+': self.BC_class(shape=self, name='x+',
                                       b_or=self.b_or, b_i=3),
                   'y-': self.BC_class(shape=self, name='y+',
                                       b_or=self.b_or, b_i=4),
                   'z+': self.BC_class(shape=self, name='z+',
                                       b_or=self.b_or, b_i=5),
                   'sponge': self.BC_class(shape=self, name='sponge'),
                   'wall': self.BC_class(shape=self, name='wall')}
        self.BC_list = [self.BC['z-'],
                        self.BC['x-'],
                        self.BC['y+'],
                        self.BC['x+'],
                        self.BC['y+'],
                        self.BC['z+'],
                        self.BC['sponge'],
                        self.BC['wall']]
        # self.BC = BCContainer(self.BC_dict)
        for i in range(6):
            self.BC_list[i].setTank()
        self.barycenter = np.array([0., 0., 0.])
        self.spongeLayers = {'y+': None, 'y-': None, 'x+': None, 'x-': None}
        self.setDimensions(dim)

    def setSponge(self, x_p=None, x_n=None, y_p=None, y_n=None):
        """
        Set length of sponge layers of the tank (used for wave absorption or
        generation zones).
        (!) Sponge layers expand outwards.

        Parameters
        ----------
        x_p: Optional[float]
            length of sponge layer in +x direction.
        x_n: Optional[float]
            length of sponge layer in -x direction.
        y_p: Optional[float]
            length of sponge layer in +y direction.
        y_n: Optional[float]
            length of sponge layer in -y direction.
        """
        self.spongeLayers['x+'] = x_p
        self.spongeLayers['x-'] = x_n
        self.spongeLayers['y+'] = y_p
        self.spongeLayers['y-'] = y_n
        self.setDimensions(self.dim)

    def setDimensions(self, dim):
        """
        Set dimension of the tank
        Parameters
        ----------
        dim: array_like
            dimensions of the tank (excluding sponge layers), array of length 3.
        """
        L, W, H = dim
        self.dim = dim
        if self.from_0 is True:
            x, y, z = L/2., W/2., H/2.
        else:
            x, y, z = self.coords
        self.coords = [x, y, z]
        x0, x1 = x - 0.5 * L, x + 0.5 * L
        y0, y1 = y - 0.5 * W, y + 0.5 * W
        z0, z1 = z - 0.5 * H, z + 0.5 * H
        # ---------------------------------------------
        # first add all vecors, facets, regions at the bottom
        # ---------------------------------------------
        bt = self.boundaryTags
        x_p = self.spongeLayers['x+'] or 0.
        x_n = self.spongeLayers['x-'] or 0.
        y_p = self.spongeLayers['y+'] or 0.
        y_n = self.spongeLayers['y-'] or 0.
        vertices = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
        vertexFlags = [bt['z-'], bt['z-'], bt['z-'], bt['z-'],
                       bt['z+'], bt['z+'], bt['z+'], bt['z+']]
        facets = [[[0, 1, 2, 3]], [[4, 5, 6, 7]]]
        segments = []
        segmentFlags = []
        volumes = [[[0, 1]]]
        facetFlags = [bt['z-'], bt['z+']]
        regions = [[(x0+x1)/2., (y0+y1)/2., (z0+z1)/2.]]
        regionFlags = [1]
        self.regionIndice = {'tank': 0}
        v_i = 8  # index of next vector to add
        r_i = 1  # index of next region to add
        f_i = len(facets)
        nb_sponge = 0  # number of sponge layers defined

        ymin = y0 - y_n
        ymax = y1 + y_p
        xmin = x0 - x_n
        xmax = x1 + x_p
        corners = False
        # y-
        if y_n:
            v0 = v_i
            v1 = v_i + 2
            v2 = v_i + 1
            v3 = v_i + 3
            if corners:
                vertices += [[x0, ymin, z0], [x0, ymin, z1],
                             [x1, ymin, z0], [x1, ymin, z1]]
            else:
                vertices += [[xmin, ymin, z0], [xmin, ymin, z1],
                             [xmax, ymin, z0], [xmax, ymin, z1]]
            vertexFlags += [bt['z-'], bt['z+'],
                            bt['z-'], bt['z+']]
            facets += [[[v0, v1, v3, v2]]]
            facetFlags += [bt['y-']]
            volumes[0][0] += [f_i]
            regions += [[(x0+x1)/2., (y0+(ymin))/2., (z0+z1)/2.]]
            self.regionIndice['y-'] = r_i
            regionFlags += [r_i + 1]
            facets += [[[0, 1, v1, v0]],
                       [[4, 5, v3, v2]],
                       [[0, 1, 5, 4]],
                       [[0, v0, v2, 4]],
                       [[1, v1, v3, 5]]]
            facetFlags += [bt['z-'],
                           bt['z+'],
                           bt['sponge']]
            if x_n > 0:
                facetFlags += [bt['wall']]
            else:
                facetFlags += [bt['x-']]
            if x_p > 0:
                facetFlags += [bt['wall']]
            else:
                facetFlags += [bt['x+']]
            volumes[0][0][-1] = f_i + 3  # add to volume
            r_i += 1
            nb_sponge += 1
            volumes += [[[len(facets) - i - 1 for i in range(6)]]]
            x_n_facet_i = len(facets) - 2
        else:
            facets += [[[0, 1, 5, 4]]]
            facetFlags += [bt['y-']]
            volumes[0][0] += [f_i]
        v_i = len(vertices)
        f_i = len(facets)
        # x+
        if x_p:
            if corners:
                vertices += [[xmax, y0, z0], [xmax, y0, z1],
                             [xmax, y1, z0], [xmax, y1, z1]]
                v0 = v_i
                v1 = v_i + 2
                v2 = v_i + 1
                v3 = v_i + 3
                vertexFlags += [bt['z-'], bt['z+'],
                                bt['z-'], bt['z+']]
            else:
                v0 = v_i - 2
                v1 = v_i
                v2 = v_i - 1
                v3 = v_i + 1
                if not y_n:
                    vertices += [[xmax, ymin, z0], [xmax, ymin, z1]]
                    vertexFlags += [bt['z-'], bt['z+']]
                    v0 = v_i
                    v1 = v_i + 2
                    v2 = v_i + 1
                    v3 = v_i + 3
                vertices += [[xmax, ymax, z0], [xmax, ymax, z1]]
                vertexFlags += [bt['z-'], bt['z+']]
            facets += [[[v0, v1, v3, v2]]]
            facetFlags += [bt['x+']]
            volumes[0][0] += [f_i]
            regions += [[(x1+(xmax))/2., (y0+y1)/2., (z0+z1)/2.]]
            self.regionIndice['x+'] = r_i
            regionFlags += [r_i + 1]
            facets += [[[1, 2, v1, v0]],
                       [[5, 6, v3, v2]],
                       [[1, 2, 6, 5]]]
            facetFlags += [bt['z-'],
                           bt['z+'],
                           bt['sponge']]
            if corners or not y_n:
                facets += [[[1, v0, v2, 5]]]
                if y_n > 0:
                    facetFlags += [bt['wall']]
                else:
                    facetFlags += [bt['y-']]
            facets += [[[2, v1, v3, 6]]]
            if y_p > 0:
                facetFlags += [bt['wall']]
            else:
                facetFlags += [bt['y+']]
            volumes[0][0][-1] = f_i + 3  # add to volume
            r_i += 1
            nb_sponge += 1
            volumes += [[[len(facets) - i - 1 for i in range(6)]]]
        else:
            facets += [[[1, 2, 6, 5]]]
            facetFlags += [bt['x+']]
            volumes[0][0] += [f_i]
        v_i = len(vertices)
        f_i = len(facets)
        # y+
        if y_p:
            if corners:
                vertices += [[x1, ymax, z0], [x1, ymax, z1],
                             [x0, ymax, z0], [x0, ymax, z1]]
                v0 = v_i
                v1 = v_i + 2
                v2 = v_i + 1
                v3 = v_i + 3
                vertexFlags += [bt['z-'], bt['z+'],
                                bt['z-'], bt['z+']]
            else:
                v0 = v_i - 2
                v1 = v_i
                v2 = v_i - 1
                v3 = v_i + 1
                if not x_p:
                    vertices += [[xmax, ymax, z0], [xmax, ymax, z1]]
                    vertexFlags += [bt['z-'], bt['z+']]
                    v0 = v_i
                    v1 = v_i + 2
                    v2 = v_i + 1
                    v3 = v_i + 3
                vertices += [[xmin, ymax, z0], [xmin, ymax, z1]]
                vertexFlags += [bt['z-'], bt['z+']]
            facets += [[[v0, v1, v3, v2]]]
            volumes[0][0] += [f_i]
            facetFlags += [bt['y+']]
            regions += [[(x0+x1)/2., (y1+(ymax))/2., (z0+z1)/2.]]
            self.regionIndice['y+'] = r_i
            regionFlags += [r_i + 1]
            facets += [[[2, 3, v1, v0]],
                       [[6, 7, v3, v2]],
                       [[2, 3, 7, 6]]]
            facetFlags += [bt['z-'],
                           bt['z+'],
                           bt['sponge']]
            if corners or not x_p:
                facets += [[[2, v0, v2, 6]]]
                if x_p > 0:
                    facetFlags += [bt['wall']]
                else:
                    facetFlags += [bt['x+']]
            facets += [[[3, v1, v3, 7]]]
            if x_n > 0:
                facetFlags += [bt['wall']]
            else:
                facetFlags += [bt['x-']]
            volumes[0][0][-1] = f_i + 3
            r_i += 1
            nb_sponge += 1
            volumes += [[[len(facets) - i - 1 for i in range(6)]]]
        else:
            facets += [[[3, 2, 6, 7]]]
            facetFlags += [bt['y+']]
            volumes[0][0] += [f_i]
        v_i = len(vertices)
        f_i = len(facets)
        # x-
        if x_n:
            volumes[0][0] += [f_i]
            if corners:
                vertices += [[xmin, y0, z0], [xmin, y0, z1],
                             [xmin, y1, z0], [xmin, y1, z1]]
                v0 = v_i
                v1 = v_i + 2
                v2 = v_i + 1
                v3 = v_i + 3
                vertexFlags += [bt['z-'], bt['z+'],
                                bt['z-'], bt['z+']]
            else:
                if not y_p:
                    vertices += [[xmin, ymax, z0], [xmin, ymax, z1]]
                    v0 = v_i - 2
                    v1 = v_i
                    v2 = v_i - 1
                    v3 = v_i + 1
                    vertexFlags += [bt['z-'], bt['z+']]
                if not y_n:
                    vertices += [[xmin, ymin, z0], [xmin, ymin, z1]]
                    if not y_p:
                        v0 = v_i + 2
                        v1 = v_i
                        v2 = v_i + 3
                        v3 = v_i + 1
                    else:
                        v0 = v_i
                        v1 = v_i - 2
                        v2 = v_i + 1
                        v3 = v_i - 1
                    vertexFlags += [bt['z-'], bt['z+']]
                else:
                    v1 = v_i - 2
                    v0 = 8
                    v3 = v_i - 1
                    v2 = 9
                    if not y_p:
                        v1 = v_i
                        v3 = v_i + 1
            facets += [[[v0, v1, v3, v2]]]
            facetFlags += [bt['x-']]
            regions += [[(x0+(xmin))/2., (y0+y1)/2., (z0+z1)/2.]]
            self.regionIndice['x-'] = r_i
            regionFlags += [r_i + 1]
            facets += [[[0, 3, v1, v0]],
                       [[4, 7, v3, v2]],
                       [[0, 3, 7, 4]]]
            facetFlags += [bt['z-'],
                           bt['z+'],
                           bt['sponge']]
            volumes[0][0][-1] = f_i + 3
            if corners or not y_n or not y_p:
                if corners or not y_n:
                    facets += [[[0, v0, v2, 4]]]
                    if y_n > 0:
                        facetFlags += [bt['wall']]
                    else:
                        facetFlags += [bt['y-']]
                if corners or not y_p:
                    facets += [[[3, v1, v3, 7]]]
                    if y_p > 0:
                        facetFlags += [bt['wall']]
                    else:
                        facetFlags += [bt['y+']]
            r_i += 1
            nb_sponge += 1
            if corners or not y_n:
                volumes += [[[len(facets) - i - 1 for i in range(6)]]]
            else:
                volumes += [[[len(facets) - i - 1 for i in range(5)] + [x_n_facet_i]]]
        else:
            facets += [[[0, 3, 7, 4]]]
            facetFlags += [bt['x-']]
            volumes[0][0] += [f_i]
        v_i = len(vertices)
        f_i = len(facets)
        self.vertices = np.array(vertices)
        self.vertices = np.dot(self.vertices, self.coords_system)
        self.vertexFlags = np.array(vertexFlags)
        self.segments = np.array(segments)
        self.segmentFlags = np.array(segmentFlags)
        self.facets = np.array(facets)
        self.facetFlags = np.array(facetFlags)
        self.regions = np.array(regions)
        self.regionFlags = np.array(regionFlags)
        self.volumes = volumes

    def setAbsorptionZones(self, dragAlpha, allSponge=False,
                           y_n=False, y_p=False,
                           x_n=False, x_p=False,
                           dragBeta=0., porosity=1.):
        """
        Sets regions (x+, x-, y+, y-) to absorption zones

        Parameters
        ----------
        dragAlpha: float
            Relaxation zone coefficient.
        allSponge: bool
            If True, all sponge layers are converted to absorption zones.
        x_p: bool
            If True, x+ region is converted to absorption zone.
        x_n: bool
            If True, x- region is converted to absorption zone.
        y_p: bool
            If True, y+ region is converted to absorption zone.
        y_n: bool
            If True, y- region is converted to absorption zone.
        dragBeta: Optional[float]
            Relaxation zone coefficient.
        porosity: Optional[float]
            Relaxation zone coefficient.
        """
        self.abs_zones = {'y-': y_n, 'y+': y_p, 'x-': x_n, 'x+': x_p}
        if allSponge is True:
            for key in self.abs_zones:
                self.abs_zones[key] = True
        waves = None
        wind_speed = np.array([0., 0., 0.])
        sl = self.spongeLayers
        for key, value in list(self.abs_zones.items()):
            if value is True:
                self._attachAuxiliaryVariable('RelaxZones')
                ind = self.regionIndice[key]
                flag = self.regionFlags[ind]
                epsFact_porous = self.spongeLayers[key]/2.
                center = np.array(self.coords)
                zeros_to_append = 3 - len(center)
                if zeros_to_append:
                    for i in range(zeros_to_append):
                        center = np.append(center, [0])
                if key == 'x-':
                    center[0] += -0.5 * self.dim[0] - 0.5 * sl['x-']
                    orientation = np.array([1., 0., 0.])
                elif key == 'x+':
                    center[0] += +0.5 * self.dim[0] + 0.5 * sl['x+']
                    orientation = np.array([-1., 0., 0.])
                elif key == 'y-':
                    center[1] += -0.5 * self.dim[1] - 0.5 * sl['y-']
                    orientation = np.array([0., 1., 0.])
                elif key == 'y+':
                    center[1] += +0.5 * self.dim[1] + 0.5 * sl['y+']
                    orientation = np.array([0., -1., 0.])
                self.zones[flag] = bc.RelaxationZone(shape=self,
                                                     zone_type='absorption',
                                                     orientation=orientation,
                                                     center=center,
                                                     waves=waves,
                                                     wind_speed=wind_speed,
                                                     epsFact_porous=epsFact_porous,
                                                     dragAlpha=dragAlpha,
                                                     dragBeta=dragBeta,
                                                     porosity=porosity)

    def setGenerationZones(self,  dragAlpha, smoothing, waves=None,
                           wind_speed=(0., 0., 0.), allSponge=False, y_n=False,
                           y_p=False, x_n=False, x_p=False, dragBeta=0.,
                           porosity=1.):
        """
        Sets regions (x+, x-, y+, y-) to generation zones

        Parameters
        ----------
        dragAlpha: float
            Relaxation zone coefficient.
        smoothing: float
            Smoothing distance (typically 3.*he)
        waves: proteus.WaveTools
            Class instance of wave generated from proteus.WaveTools.
        wind_speed: Optional[array_like]
            Speed of wind in generation zone (default is (0., 0., 0.))
        allSponge: bool
            If True, all sponge layers are converted to generation zones.
        x_p: bool
            If True, x+ region is converted to generation zone.
        x_n: bool
            If True, x- region is converted to generation zone.
        y_p: bool
            If True, y+ region is converted to generation zone.
        y_n: bool
            If True, y- region is converted to generation zone.
        dragBeta: Optional[float]
            Relaxation zone coefficient.
        porosity: Optional[float]
            Relaxation zone coefficient.
        """
        self.abs_zones = {'y-': y_n, 'y+': y_p, 'x-': x_n, 'x+': x_p}
        if allSponge is True:
            for key in self.abs_zones:
                self.abs_zones[key] = True
        waves = waves
        wind_speed = np.array(wind_speed)
        sl = self.spongeLayers
        for key, value in list(self.abs_zones.items()):
            if value is True:
                self._attachAuxiliaryVariable('RelaxZones')
                ind = self.regionIndice[key]
                flag = self.regionFlags[ind]
                epsFact_porous = self.spongeLayers[key]/2.
                center = np.array(self.coords)
                zeros_to_append = 3 - len(center)
                if zeros_to_append:
                    for i in range(zeros_to_append):
                        center = np.append(center, [0])
                if key == 'x-':
                    center[0] += -0.5 * self.dim[0] - sl['x-']/2.
                    orientation = np.array([1., 0., 0.])
                    self.BC['x-'].setUnsteadyTwoPhaseVelocityInlet(wave=waves,
                                                                   wind_speed=wind_speed,
                                                                   smoothing=smoothing)
                elif key == 'x+':
                    center[0] += +0.5 * self.dim[0] + sl['x+']/2.
                    orientation = np.array([-1., 0., 0.])
                    self.BC['x+'].setUnsteadyTwoPhaseVelocityInlet(wave=waves,
                                                                   wind_speed=wind_speed,
                                                                   smoothing=smoothing)
                elif key == 'y-':
                    center[1] += -0.5 * self.dim[1] - sl['y-']/2.
                    orientation = np.array([0., 1., 0.])
                    self.BC['y-'].setUnsteadyTwoPhaseVelocityInlet(wave=waves,
                                                                   wind_speed=wind_speed,
                                                                   smoothing=smoothing)
                elif key == 'y+':
                    center[1] += +0.5 * self.dim[1] + sl['y+']/2.
                    orientation = np.array([0., -1., 0.])
                    self.BC['y+'].setUnsteadyTwoPhaseVelocityInlet(wave=waves,
                                                                   wind_speed=wind_speed,
                                                                   smoothing=smoothing)
                self.zones[flag] = bc.RelaxationZone(shape=self,
                                                     zone_type='generation',
                                                     orientation=orientation,
                                                     center=center,
                                                     waves=waves,
                                                     wind_speed=wind_speed,
                                                     epsFact_porous=epsFact_porous,
                                                     dragAlpha=dragAlpha,
                                                     dragBeta=dragBeta,
                                                     porosity=porosity,
                                                     smoothing=smoothing)


class Tank2D(ShapeRANS):
    """
    Class to create a 2D tank (rectangular shape).

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    dim: array_like
        Dimensions of the tank (excluding sponge layers).
    coords: Optional[array_like]
        Coordinates of the centroid of the shape.
    """
    count = 0

    def __init__(self, domain, dim, coords=None):
        super(Tank2D, self).__init__(domain, nd=2)
        if coords is None:
            self.coords = np.array(dim)/2.
            self.from_0 = True
        else:
            self.coords = coords
            self.from_0 = False
        self.dim = dim
        self._nameSelf()
        self._setupBCs()
        self.spongeLayers = {'x-': None,
                             'x+': None}
        self.constructShape()

    def _nameSelf(self):
        self.__class__.count += 1
        self.name = "tank2D" + repr(self.__class__.count)

    def _setupBCs(self):
        self.boundaryTags = {'y-': 1, 'x+': 2, 'y+': 3, 'x-': 4, 'sponge': 5}
        self.b_or = np.array([[0., -1., 0.],
                              [1., 0., 0.],
                              [0., 1., 0.],
                              [-1., 0., 0.]])
        self.BC = {'y-': self.BC_class(shape=self, name='y-',
                                       b_or=self.b_or, b_i=0),
                   'x+': self.BC_class(shape=self, name='x+',
                                       b_or=self.b_or, b_i=1),
                   'y+': self.BC_class(shape=self, name='y+',
                                       b_or=self.b_or, b_i=2),
                   'x-': self.BC_class(shape=self, name='x-',
                                       b_or=self.b_or, b_i=3),
                   'sponge': self.BC_class(shape=self, name='sponge')}
        self.BC_list = [self.BC['y-'],
                        self.BC['x+'],
                        self.BC['y+'],
                        self.BC['x-'],
                        self.BC['sponge']]
        # self.BC = BCContainer(self.BC_dict)
        for i in range(4):
            self.BC_list[i].setTank()

    def constructShape(self):
        """
        Construct the geometry of the tank: segments, regions, etc.

        Parameters
        ----------
        frame: array_like
            An array of (x,y) coordinates in counterclockwise order to define
            the boundaries of the main (that is, excluding extensions such as
            sponge zones) shape.  This can be generated with tank2DFrame or
            subclass specific methods.
        frame_flags: array_like
            A corresponding array of boundary tags associated with each point
            in the frame.  This can be generated with tank2DFrame or subclass
            specific methods.
        """
        self._findEdges(self.dim)
        vertices, vertexFlags = self._constructVertices()
        segments, segmentFlags = self._constructSegments(vertices, vertexFlags)
        regions, regionFlags = self._constructRegions(vertices, vertexFlags,
                                                      segments, segmentFlags)
        facets, facetFlags = self._constructFacets()

        self.vertices = np.array(vertices)
        self.vertexFlags = np.array(vertexFlags)
        self.segments = np.array(segments)
        self.segmentFlags = np.array(segmentFlags)
        self.regions = np.array(regions)
        self.regionFlags = np.array(regionFlags)
        self.facets = np.array(facets)
        self.facetFlags = np.array(facetFlags)

    def _findEdges(self, dim):

        coords = self.coords
        from_0 = self.from_0

        if from_0:
            self.x0 = 0
            self.x1 = dim[0]
            self.y0 = 0
            self.y1 = dim[1]
        else:  # not from_0 and coords is not None
            self.x0 = coords[0] - 0.5 * dim[0]
            self.x1 = coords[0] + 0.5 * dim[0]
            self.y0 = coords[1] - 0.5 * dim[1]
            self.y1 = coords[1] + 0.5 * dim[1]

    def _constructVertices(self):
        vertices = [[self.x0, self.y0],
                    [self.x1, self.y0],
                    [self.x1, self.y1],
                    [self.x0, self.y1]]
        vertexFlags = [self.boundaryTags['y-'],
                       self.boundaryTags['y-'],
                       self.boundaryTags['y+'],
                       self.boundaryTags['y+']]
        if self.spongeLayers['x-']:
            vertices += [[self.x0 - self.spongeLayers['x-'], self.y0],
                         [self.x0 - self.spongeLayers['x-'], self.y1]]
            vertexFlags += [self.boundaryTags['y-'],
                            self.boundaryTags['y+']]
        if self.spongeLayers['x+']:
            vertices += [[self.x1 + self.spongeLayers['x+'], self.y0],
                         [self.x1 + self.spongeLayers['x+'], self.y1]]
            vertexFlags += [self.boundaryTags['y-'],
                            self.boundaryTags['y+']]
        return vertices, vertexFlags

    def _constructSegments(self, vertices, vertexFlags):
        segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
        segmentFlags = [self.boundaryTags['y-'],
                        self.boundaryTags['x+'],
                        self.boundaryTags['y+'],
                        self.boundaryTags['x-']]
        added_vertices = 0
        if self.spongeLayers['x-']:
            segments += [[0, 4 + added_vertices],
                         [4 + added_vertices, 5 + added_vertices],
                         [5 + added_vertices, 3]]
            segmentFlags += [self.boundaryTags['y-'],
                             self.boundaryTags['x-'],
                             self.boundaryTags['y+']]
            segmentFlags[3] = self.boundaryTags['sponge']
            added_vertices += 2
        if self.spongeLayers['x+']:
            segments += [[1, 4 + added_vertices],
                         [4 + added_vertices, 5 + added_vertices],
                         [5 + added_vertices, 2]]
            segmentFlags += [self.boundaryTags['y-'],
                             self.boundaryTags['x+'],
                             self.boundaryTags['y+']]
            segmentFlags[1] = self.boundaryTags['sponge']
            added_vertices += 2
        return segments, segmentFlags

    def _constructFacets(self):
        facets = [[[0, 1, 2, 3]]]
        facetFlags = [1]
        added_vertices = 0
        added_facets = 0
        if self.spongeLayers['x-']:
            facets += [[[3, 0, 4, 5]]]
            facetFlags += [2 + added_facets]
            added_vertices += 2
            added_facets += 1
        if self.spongeLayers['x+']:
            facets += [[[2, 1, added_vertices + 4, added_vertices + 5]]]
            facetFlags += [2 + added_facets]
        return facets, facetFlags

    def _constructRegions(self, vertices, vertexFlags, segments, segmentFlags):
        regions = [[self.x0 + 0.01 * (self.x1 - self.x0), 0.5 * (self.y0 + self.y1)], ]
        ind_region = 1
        regionFlags = [ind_region, ]
        self.regionIndice = {'tank': ind_region - 1}
        if self.spongeLayers['x-']:
            regions += [[self.x0 - 0.5 * self.spongeLayers['x-'],
                         0.5 * (self.y0 + self.y1)]]
            ind_region += 1
            regionFlags += [ind_region]
            self.regionIndice['x-'] = ind_region - 1
        if self.spongeLayers['x+']:
            regions += [[self.x1 + 0.5 * self.spongeLayers['x+'],
                         0.5 * (self.y0 + self.y1)]]
            ind_region += 1
            regionFlags += [ind_region]
            self.regionIndice['x+'] = ind_region - 1
        return regions, regionFlags

    def setSponge(self, x_n=None, x_p=None):
        """
        Set length of sponge layers of the tank (used for wave absorption or
        generation zones).
        (!) Sponge layers expand outwards.

        Parameters
        ----------
        x_p: Optional[float]
            length of sponge layer in +x direction.
        x_n: Optional[float]
            length of sponge layer in -x direction.
        """
        self.spongeLayers['x-'] = x_n
        self.spongeLayers['x+'] = x_p
        self.constructShape()

    def setAbsorptionZones(self, dragAlpha, x_n=False, x_p=False,
                           dragBeta=0., porosity=1.):
        """
        Sets regions (x+, x-) to absorption zones

        Parameters
        ----------
        dragAlpha: float
            Relaxation zone coefficient.
        allSponge: bool
            If True, all sponge layers are converted to absorption zones.
        x_p: bool
            If True, x+ region is converted to absorption zone.
        x_n: bool
            If True, x- region is converted to absorption zone.
        dragBeta: Optional[float]
            Relaxation zone coefficient.
        porosity: Optional[float]
            Relaxation zone coefficient.
        """
        waves = None
        wind_speed = np.array([0., 0., 0.])
        if x_n or x_p:
            self._attachAuxiliaryVariable('RelaxZones')
        if x_n is True:
            center = np.array([self.coords[0]-0.5*self.dim[0]-0.5*self.spongeLayers['x-'],
                               self.coords[1],
                               0.])
            ind = self.regionIndice['x-']
            flag = self.regionFlags[ind]
            epsFact_porous = self.spongeLayers['x-']/2.
            orientation = np.array([1., 0.])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='absorption',
                                                 orientation=orientation,
                                                 center=center,
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous,
                                                 dragAlpha=dragAlpha,
                                                 dragBeta=dragBeta,
                                                 porosity=porosity)
        if x_p is True:
            center = np.array([self.coords[0]+0.5*self.dim[0]+0.5*self.spongeLayers['x+'],
                               self.coords[1],
                               0.])
            ind = self.regionIndice['x+']
            flag = self.regionFlags[ind]
            epsFact_porous = self.spongeLayers['x+']/2.
            orientation = np.array([-1., 0.])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='absorption',
                                                 orientation=orientation,
                                                 center=center,
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous,
                                                 dragAlpha=dragAlpha,
                                                 dragBeta=dragBeta,
                                                 porosity=porosity)

    def setGenerationZones(self,  dragAlpha,  smoothing,
                           waves=None, wind_speed=(0., 0., 0.),
                           x_n=False, x_p=False,
                           dragBeta=0., porosity=1.):
        """
        Sets regions (x+, x-) to generation zones

        Parameters
        ----------
        dragAlpha: float
            Relaxation zone coefficient
        smoothing:
            Smoothing distance
        waves: proteus.WaveTools
            Class instance of wave generated from proteus.WaveTools.
        wind_speed: Optional[array_like]
            Speed of wind in generation zone (default is (0., 0., 0.))
        allSponge: bool
            If True, all sponge layers are converted to generation zones.
        x_p: bool
            If True, x+ region is converted to generation zone.
        x_n: bool
            If True, x- region is converted to generation zone.
        dragBeta: Optional[float]
            Relaxation zone coefficient.
        porosity: Optional[float]
            Relaxation zone coefficient.
        """
        waves = waves
        wind_speed = np.array(wind_speed)
        if x_n or x_p:
            self._attachAuxiliaryVariable('RelaxZones')
        if x_n is True:
            center = np.array([self.coords[0]-0.5*self.dim[0]-0.5*self.spongeLayers['x-'],
                               self.coords[1],
                               0.])
            ind = self.regionIndice['x-']
            flag = self.regionFlags[ind]
            epsFact_porous = self.spongeLayers['x-']/2.
            orientation = np.array([1., 0.])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='generation',
                                                 orientation=orientation,
                                                 center=center,
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous,
                                                 dragAlpha=dragAlpha,
                                                 dragBeta=dragBeta,
                                                 porosity=porosity,
                                                 smoothing=smoothing)
            self.BC['x-'].setUnsteadyTwoPhaseVelocityInlet(wave=waves,
                                                           wind_speed=wind_speed,
                                                           smoothing=smoothing)
        if x_p is True:
            center = np.array([self.coords[0]+0.5*self.dim[0]+0.5*self.spongeLayers['x+'],
                               self.coords[1],
                               0.])
            ind = self.regionIndice['x+']
            flag = self.regionFlags[ind]
            epsFact_porous = self.spongeLayers['x+']/2.
            orientation = np.array([-1., 0.])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='generation',
                                                 orientation=orientation,
                                                 center=center,
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous,
                                                 dragAlpha=dragAlpha,
                                                 dragBeta=dragBeta,
                                                 porosity=porosity,
                                                 smoothing=smoothing)
            self.BC['x+'].setUnsteadyTwoPhaseVelocityInlet(wave=waves,
                                                           wind_speed=wind_speed,
                                                           smoothing=smoothing)

#[temp] no tests yet!


class TankWithObstacles2D(Tank2D):
    """
    Class to create a 2D rectangular tank with obstacles built out of any wall.

    An obstacle is defined by a contiguous list of points which start and end
    on the walls or corners of the tank.

    This also covers special boundary conditions.  To tag a segment with a
    unique boundary tag, add the starting vertex (in the counterclockwise
    format the shape is generated in) of the segment as a value in a dictionary
    element keyed to the name of the boundary tag.

    (!) Warning: If each of the four corners of the rectangular tank is inside
    an obstacle or a vertex for an obstacle, then the tank's region is defined
    in a pseudorandom manner, which may make it unreliable when dealing with
    holes caused by other shapes.
    (!) Warning: Obstacle boundary tags are keyed to whichever edge they started
    from.  If this is undesirable, it may be manually overriden by applying
    special boundary conditions to affected segments.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    dim: Optional[array_like]
        Dimensions of the cuboid.
    obstacles: Optional[array_like]
        A list of lists of (x,y) coordinates.  Each (x,y) coordinate is a length
        and height relative to the x-,y- corner of the tank.  Each list of
        coordinates is an obstacle defined by points connected in the order of
        their index.  The list of lists gives all such obstacles in a
        counterclockwise manner of which they should be appended, starting from
        the (x-,y-) corner.
    special_boundaries: Optional[mapping]
        A dictionary of lists of vertices keyed to boundary names. The vertices
        listed will be given the boundary name they are keyed to, overriding
        any other designation they would be given.
        If this is a distinct boundary name, the segment starting from the vertex
        will be assigned the same boundary tag.
    full_circle: Optional[bool]
        A boolean tag to check if the final obstacle ends on the same edge that
        the first obstacle starts on.  Default is False.
    coords: Optional[array_like]
        Coordinates of the centroid of the shape.
    hole: Optional[bool]
        If True (default), the obstacle of the tank is just an open hole at the
        bottom of the tank. If False, a segment at the bottom of the obstacle is
        created to close the hole.
    obstacle_regions: Optional[array_like]
        To use only if hole=False.(x,y) coordinates of a point inside the
        obstacle in order to fill the obstacle with what should be inside
        (for example a porous material).

    """

    def __init__(self, domain, dim=(0., 0.),
                 obstacles=None, special_boundaries=None,
                 full_circle=False,
                 coords=None, hole=True, obstacle_regions=None):
        if obstacles:
            self.obstacles = obstacles
        else:
            self.obstacles = []

        self.special_boundaries = []
        self.special_BC_vertices = []
        self.full_circle = full_circle

        self.spongeLayers = {'x-': None,
                             'x+': None}

        if special_boundaries:
            for key in list(special_boundaries.keys()):
                self.special_boundaries += [key for v in special_boundaries[key]]
                self.special_BC_vertices += special_boundaries[key]

        self.corners = {'x-y-': False, 'x+y-': False,
                        'x+y+': False, 'x-y+': False}

        self.hole = hole
        self.obstacle_regions = obstacle_regions
        super(TankWithObstacles2D, self).__init__(domain, dim, coords)

    def _setupBCs(self):
        super(TankWithObstacles2D, self)._setupBCs()
        for boundary in self.special_boundaries:
            if boundary not in list(self.boundaryTags.keys()):
                self.boundaryTags[boundary] = len(self.boundaryTags) + 1
                self.BC[boundary] = self.BC_class(shape=self, name=boundary)
                self.BC_list += [self.BC[boundary]]
        # add boundaryTags
        self.obstacle_flags = []
        max_flag = 0
        for tag, flag in list(self.boundaryTags.items()):
            if flag > max_flag:
                max_flag = flag
        flag = max_flag + 1
        for i in range(len(self.obstacles)):
            tag = 'obstacle' + repr(i + 1)
            self.boundaryTags[tag] = flag
            self.obstacle_flags += [flag]
            self.BC[tag] = self.BC_class(shape=self, name=tag)
            self.BC_list += [self.BC[tag]]
            flag += 1
        if self.hole is False:
            assert len(self.obstacles) == len(self.obstacle_regions), 'must have same number of regions as obstacles'

    def _resetEdgesFromVertices(self, vertices):
        """
        Resets self.x0, self.x1, self.y0, self.y1 based on the actual shape.

        In particular, they will form a bounding box form around the shape -
        the furthest points in x and y dimensions, both high and low.

        Parameters
        ----------
        vertices: array_like
        """
        sorted_vertices = sorted(vertices, key=lambda vertex: vertex[1])
        self.y0 = sorted_vertices[0][1]
        self.y1 = sorted_vertices[-1][1]
        sorted_vertices = sorted(vertices, key=lambda vertex: vertex[0])
        self.x0 = sorted_vertices[0][0]
        self.x1 = sorted_vertices[-1][0]

    def _findSpongeLayerCorners(self, vertices):
        """
        Finds the corners for horizontal (x-, x+) sponge layers.

        Parameters
        ----------
        vertices: array_like
        """
        self._resetEdgesFromVertices(vertices)

        potential_x_n_corners = [vertex for vertex in vertices
                                 if np.isclose(vertex[0], self.x0)]
        potential_x_p_corners = [vertex for vertex in vertices
                                 if np.isclose(vertex[0], self.x1)]

        potential_x_n_corners.sort(key=lambda vertex: vertex[1])
        potential_x_p_corners.sort(key=lambda vertex: vertex[1])

        self.x0y0 = potential_x_n_corners[0]
        self.x0y1 = potential_x_n_corners[-1]
        self.x1y0 = potential_x_p_corners[0]
        self.x1y1 = potential_x_p_corners[-1]

    def _constructVertices(self):

        def getClockwiseOrder(first_point):
            clockwise_ordering = ('x-y-', 'y-', 'x+y-', 'x+',
                                  'x+y+', 'y+', 'x-y+', 'x-')
            index = clockwise_ordering.index(first_point)
            return clockwise_ordering[index:] + clockwise_ordering[:index]

        def findLocation(vertex):
            """
            Given an (x,y) coordinate gives a label associated to corner or edge
            """
            dim = [self.x1 - self.x0, self.y1 - self.y0]
            if np.isclose(vertex[0], 0) and np.isclose(vertex[1], 0):
                return 'x-y-'
            elif np.isclose(vertex[0], dim[0]) and np.isclose(vertex[1], dim[1]):
                return 'x+y+'
            elif np.isclose(vertex[0], 0) and np.isclose(vertex[1], dim[1]):
                return 'x-y+'
            elif np.isclose(vertex[0], dim[0]) and np.isclose(vertex[1], 0):
                return 'x+y-'
            elif np.isclose(vertex[0], 0):
                return 'x-'
            elif np.isclose(vertex[0], dim[0]):
                return 'x+'
            elif np.isclose(vertex[1], 0):
                return 'y-'
            elif np.isclose(vertex[1], dim[1]):
                return 'y+'
            else:
                raise ValueError("Point " + str(vertex) + " does not seem to"
                                 "be on a tank wall.")

        def checkClosure(start_point, end_point):
            if start_point == end_point:
                return True

        def addCorner(corner_flag):
            if corner_flag == 'x-y-':
                corner = [[self.x0, self.y0]]
            elif corner_flag == 'x+y-':
                corner = [[self.x1, self.y0]]
            elif corner_flag == 'x+y+':
                corner = [[self.x1, self.y1]]
            elif corner_flag == 'x-y+':
                corner = [[self.x0, self.y1]]
            # vertex flags
            if corner_flag in ['x-y-', 'x+y-']:
                corner_tag = [self.boundaryTags['y-']]
            else:
                corner_tag = [self.boundaryTags['y+']]

            return corner, corner_tag

        def addIntermediateCorners(first, last):
            """
            Returns corner vertices (and flags) in between two segments
            """
            ordering = getClockwiseOrder(first)
            corners = [x for x in ordering
                       if x in list(self.corners.keys())
                       and ordering.index(x) < ordering.index(last)
                       ]
            corner_vertices = []
            corner_flags = []

            for corner in corners:
                self.corners[corner] = True
                vertex, flag = addCorner(corner)
                corner_vertices += vertex
                corner_flags += flag

            return corner_vertices, corner_flags

        def addRemainingCorners(first, last):
            if first == last:
                if self.full_circle:
                    return []
                else:
                    return addAllCorners(first)
            else:
                return addIntermediateCorners(first, last)

        def addAllCorners(starting_point):
            """
            Returns all corners and flags.
            """
            corner_vertices = []
            corner_flags = []

            ordering = getClockwiseOrder(starting_point)

            for potential_corner in ordering:
                if potential_corner in list(self.corners.keys()):
                    self.corners[potential_corner] = True
                    vertex, flag = addCorner(potential_corner)
                    corner_vertices += vertex
                    corner_flags += flag

            return corner_vertices, corner_flags

        def addSpongeVertices():
            sponge_vertices = []
            sponge_vertexFlags = []
            if self.spongeLayers['x-']:
                sponge_vertices += [[v[0] - self.spongeLayers['x-'], v[1]]
                                    for v in [self.x0y0, self.x0y1]]
                sponge_vertexFlags += [self.boundaryTags['y-'],
                                       self.boundaryTags['y+']]
            if self.spongeLayers['x+']:
                sponge_vertices += [[v[0] + self.spongeLayers['x+'], v[1]]
                                    for v in [self.x1y0, self.x1y1]]
                sponge_vertexFlags += [self.boundaryTags['y-'],
                                       self.boundaryTags['y+']]
            return sponge_vertices, sponge_vertexFlags

        #--------------------------------------------------------#
        vertices = []
        vertexFlags = []
        former_end = None
        first_start = None

        for nb, obstacle in enumerate(self.obstacles):
            start = findLocation(obstacle[0])
            end = findLocation(obstacle[-1])

            if start == end and checkClosure(obstacle[0], obstacle[-1]):
                raise ValueError("Obstacles must be open (start and end"
                                 " vertices must be distinct)")
            if start == former_end and checkClosure(obstacle[0], vertices[-1]):
                vertices.pop()
                vertexFlags.pop()

            # ---- In-Between Corner Vertices ---- #
            if former_end is not None:
                new_vertices, new_flags = addIntermediateCorners(former_end, start)
                vertices += new_vertices
                vertexFlags += new_flags

            # ---- Obstacle ---- #
            if self.hole is True:
                vertices += obstacle
                vertexFlags += [self.boundaryTags[start]
                                for i in range(len(obstacle))]
            elif self.hole is False:
                vertices += obstacle
                vertexFlags += [self.boundaryTags['obstacle' + repr(nb + 1)]
                                for i in range(len(obstacle))]

            # ---- Paperwork ---- #
            former_end = end
            if first_start is None:
                first_start = start

        # ---- Remaining Corner Vertices ---- #
        if first_start is not None:
            new_vertices, new_flags = addRemainingCorners(former_end,
                                                          first_start)
        else:
            new_vertices, new_flags = addAllCorners('x-')

        vertices += new_vertices
        vertexFlags += new_flags

        # ---- Check for Special Conditions ---- #
        for vertex in self.special_BC_vertices:
            flag_index = vertices.index(vertex)
            boundary_index = self.special_BC_vertices.index(vertex)
            boundary_name = self.special_boundaries[boundary_index]
            vertexFlags[flag_index] = self.boundaryTags[boundary_name]

        # ---- Adjustments for Sponge Zones ---- #
        self._findSpongeLayerCorners(vertices=vertices)

        # ---- Add Sponge Zone Vertices ---- #
        new_vertices, new_flags = addSpongeVertices()
        vertices += new_vertices
        vertexFlags += new_flags

        return vertices, vertexFlags

    def _constructSegments(self, vertices, vertexFlags):
        # VertexFlag --> SegmentFlag logic:
        #
        # if EITHER are x+  --> segment is x+
        #                       UNLESS the other is x-  --> y+
        # if EITHER are x-  --> segment is x-
        #                       UNLESS the other is x+  --> y-
        # if it STARTS y-   --> segment is y-
        #                       UNLESS they are vertical --> x+
        # if it STARTS y+   --> segment is y+
        #                       UNLESS they are vertical --> x-
        # if BOTH are ***   --> segment is ***
        # (if two different *** are around, it takes the first)
        segments = []
        segmentFlags = []

        on_sponge_edge = {'x-': False, 'x+': False}
        sponge_edges_covered = {'x-': False, 'x+': False}

        def checkSpongeStatus(start_index, end_index):
            start_vertex = vertices[start_index]
            if self.spongeLayers['x-']:
                if not on_sponge_edge['x-']:
                    if start_vertex in (self.x0y0, self.x0y1):
                        on_sponge_edge['x-'] = True
                elif not sponge_edges_covered['x-']:
                    if start_vertex in (self.x0y0, self.x0y1):
                        on_sponge_edge['x-'] = False
                        sponge_edges_covered['x-'] = True
                    else:
                        vertexFlags[start_index] = self.boundaryTags['sponge']
                else:
                    pass

            if self.spongeLayers['x+']:
                if not on_sponge_edge['x+']:
                    if start_vertex in (self.x1y0, self.x1y1):
                        on_sponge_edge['x+'] = True
                elif not sponge_edges_covered['x+']:
                    if start_vertex in (self.x1y0, self.x1y1):
                        on_sponge_edge['x+'] = False
                        sponge_edges_covered['x+'] = True
                    else:
                        vertexFlags[start_index] = self.boundaryTags['sponge']
                else:
                    pass

            end_vertex = vertices[end_index]
            if on_sponge_edge['x-']:
                if end_vertex not in (self.x0y0, self.x0y1):
                    vertexFlags[end_index] = self.boundaryTags['sponge']
            if on_sponge_edge['x+']:
                if end_vertex not in (self.x1y0, self.x1y1):
                    vertexFlags[end_index] = self.boundaryTags['sponge']

        def getSegmentFlag(start, end):
            if ((self.spongeLayers['x-'] and not sponge_edges_covered['x-']) or
                    (self.spongeLayers['x+'] and not sponge_edges_covered['x+'])):
                checkSpongeStatus(start, end)

            if on_sponge_edge['x-'] or on_sponge_edge['x+']:
                return [self.boundaryTags['sponge'], ]

            else:
                if vertexFlags[start] in self.obstacle_flags:
                    if vertexFlags[end] == vertexFlags[start]:
                        return [vertexFlags[start], ]
                    else:
                        return [self.boundaryTags['y-'], ]
                elif vertexFlags[start] == self.boundaryTags['x+']:
                    if vertexFlags[end] == self.boundaryTags['x-']:
                        return [self.boundaryTags['y+'], ]
                    else:
                        return [self.boundaryTags['x+'], ]

                elif vertexFlags[start] == self.boundaryTags['x-']:
                    if vertexFlags[end] == self.boundaryTags['x+']:
                        return [self.boundaryTags['y-'], ]
                    else:
                        return [self.boundaryTags['x-'], ]

                elif vertexFlags[end] == self.boundaryTags['x+']:
                    if vertexFlags[start] in [self.boundaryTags['y-'],
                                              self.boundaryTags['y+']]:
                        return [self.boundaryTags['x+'], ]

                elif vertexFlags[end] == self.boundaryTags['x-']:
                    if vertexFlags[start] in [self.boundaryTags['y-'],
                                              self.boundaryTags['y+']]:
                        return [self.boundaryTags['x-'], ]

                elif vertexFlags[start] == self.boundaryTags['y-']:
                    if (vertexFlags[end] == self.boundaryTags['y+']
                        and np.isclose(vertices[start][0], vertices[end][0])
                        ):
                        return [self.boundaryTags['x+'], ]
                    else:
                        return [self.boundaryTags['y-'], ]

                elif vertexFlags[start] == self.boundaryTags['y+']:
                    if (vertexFlags[end] == self.boundaryTags['y-']
                        and np.isclose(vertices[start][0], vertices[end][0])
                        ):
                        return [self.boundaryTags['x-'], ]
                    else:
                        return [self.boundaryTags['y+'], ]

                else:
                    return [vertexFlags[start], ]

        # ---- Initial Sponge Logic ---- #
        sponge_vertex_count = 0

        if self.spongeLayers['x-']:
            sponge_vertex_count += 2
        if self.spongeLayers['x+']:
            sponge_vertex_count += 2

        # ---- Build Main Segments ---- #
        for i in range(len(vertices) - 1 - sponge_vertex_count):
            segments += [[i, i + 1], ]
            segmentFlags += getSegmentFlag(i, i + 1)
        segments += [[len(vertices) - 1 - sponge_vertex_count, 0], ]
        segmentFlags += getSegmentFlag(len(vertices) - 1 - sponge_vertex_count,
                                       0)
        if self.hole is False:
            start_vertex_obstacle = 0
            for obstacle in self.obstacles:
                segments += [[start_vertex_obstacle, start_vertex_obstacle + (len(obstacle) - 1)]]
                segmentFlags += [self.boundaryTags['y-']]
                start_vertex_obstacle += len(obstacle)

        # ---- Build Sponge Segments ---- #
        if self.spongeLayers['x-']:
            segments += [[vertices.index(self.x0y0),
                          len(vertices) - sponge_vertex_count],
                         [len(vertices) - sponge_vertex_count,
                          len(vertices) - sponge_vertex_count + 1],
                         [len(vertices) - sponge_vertex_count + 1,
                          vertices.index(self.x0y1)]
                         ]
            segmentFlags += [self.boundaryTags['y-'],
                             self.boundaryTags['x-'],
                             self.boundaryTags['y+']]
        if self.spongeLayers['x+']:
            segments += [[vertices.index(self.x1y0), len(vertices) - 2],
                         [len(vertices) - 2, len(vertices) - 1],
                         [len(vertices) - 1, vertices.index(self.x1y1)]
                         ]
            segmentFlags += [self.boundaryTags['y-'],
                             self.boundaryTags['x+'],
                             self.boundaryTags['y+']]

        return segments, segmentFlags

    def _constructRegions(self, vertices, vertexFlags, segments, segmentFlags):

        ind_region = 0
        self.regionIndice = {}
        regions = []
        regionFlags = []

        if self.hole is False:
            for i, region in enumerate(self.obstacle_regions):
                regions = [[region[0], region[1]]]
                ind_region += 1
                regionFlags = [ind_region, ]
                self.regionIndice['obstacle' + repr(i + 1)] = ind_region - 1

        if True in list(self.corners.values()):
            regions += self._getCornerRegion()
        else:
            regions += self._getRandomRegion(vertices, segments)

        ind_region += 1
        regionFlags += [ind_region]
        self.regionIndice['tank'] = ind_region - 1

        sponge_half_height_x0 = 0.5 * (self.x0y0[1] + self.x0y1[1])
        sponge_half_height_x1 = 0.5 * (self.x1y0[1] + self.x1y1[1])
        sponge_x0 = self.x0y0[0]
        sponge_x1 = self.x1y0[0]

        if self.spongeLayers['x-']:
            regions += [[sponge_x0 - 0.5 * self.spongeLayers['x-'],
                         sponge_half_height_x0]]
            ind_region += 1
            regionFlags += [ind_region]
            self.regionIndice['x-'] = ind_region - 1
        if self.spongeLayers['x+']:
            regions += [[sponge_x1 + 0.5 * self.spongeLayers['x+'],
                         sponge_half_height_x1]]
            ind_region += 1
            regionFlags += [ind_region]
            self.regionIndice['x+'] = ind_region - 1

        return regions, regionFlags

    def _findExtrema(self, points):
        """
        Return the extrema of a series of points in n dimensions in the form:
        max(x1), max(x2), ... , max(xn), min(x1), ... , min(xn)
        """
        points = np.array(points)
        return np.max(points, 0).tolist() + np.min(points, 0).tolist()

    def _getCornerRegion(self):
        eps = np.finfo(float).eps
        if self.corners['x-y-']:
            return [[self.x0 + eps, self.y0 + eps], ]
        elif self.corners['x+y-']:
            return [[self.x1 - eps, self.y0 + eps], ]
        elif self.corners['x+y+']:
            return [[self.x1 - eps, self.y1 - eps], ]
        elif self.corners['x-y+']:
            return [[self.x0 + eps, self.y1 - eps], ]

    def _getRandomRegion(self, vertices, segments):
        x_p, y_p, x_n, y_n = self._findExtrema(vertices)
        if self.spongeLayers['x-']:
            x_n += self.spongeLayers['x-']
        if self.spongeLayers['x+']:
            x_p -= self.spongeLayers['x+']

        count = 0
        allowed_tries = 100

        while True:
            count += 1
            vertical_line = np.random.uniform(x_n, x_p)
            if True in [np.isclose(vertical_line, vertex[0]) for vertex in
                        vertices]:
                continue

            lowest_intersect = second_intersect = y_p

            for segment in segments:
                line_x0 = vertices[segment[0]][0]
                line_y0 = vertices[segment[0]][1]
                line_x1 = vertices[segment[1]][0]
                line_y1 = vertices[segment[1]][1]
                if (line_x0 < vertical_line < line_x1
                        or line_x0 > vertical_line > line_x1):
                    # (due to the strict inequality check and
                    # our selection of vertical_line - x1 > x0 should be sure)
                    intersection_height = line_y0 + (
                        (line_y1 - line_y0)
                        * (vertical_line - line_x0)
                        / (line_x1 - line_x0)
                    )
                    if intersection_height < lowest_intersect:
                        second_intersect = lowest_intersect
                        lowest_intersect = intersection_height
                    elif intersection_height < second_intersect:
                        second_intersect = intersection_height

            interior_point = 0.5 * (lowest_intersect + second_intersect)

            if lowest_intersect < interior_point < second_intersect:
                break
            if count > allowed_tries:
                ValueError(
                    "Cannot find a proper interior point of the defined "
                    "shape after " + str(count) + " tries.")

        return [[vertical_line, interior_point], ]

    def setAbsorptionZones(self, dragAlpha, x_n=False, x_p=False,
                           dragBeta=0., porosity=1.):
        """
        Sets regions (x+, x-) to absorption zones

        Parameters
        ----------
        dragAlpha: float
            Relaxation zone coefficient
        allSponge: bool
            If True, all sponge layers are converted to absorption zones.
        x_p: bool
            If True, x+ region is converted to absorption zone.
        x_n: bool
            If True, x- region is converted to absorption zone.
        dragBeta: Optional[float]
            Relaxation zone coefficient.
        porosity: Optional[float]
            Relaxation zone coefficient.
        """
        sponge_half_height_x0 = 0.5 * (self.x0y0[1] + self.x0y1[1])
        sponge_half_height_x1 = 0.5 * (self.x1y0[1] + self.x1y1[1])
        sponge_x0 = self.x0y0[0]
        sponge_x1 = self.x1y0[0]

        waves = None
        wind_speed = np.array([0., 0., 0.])
        if x_n or x_p:
            self._attachAuxiliaryVariable('RelaxZones')
        if x_n is True:
            center = np.array([sponge_x0 - 0.5 * self.spongeLayers['x-'],
                               sponge_half_height_x0, 0.])
            ind = self.regionIndice['x-']
            flag = self.regionFlags[ind]
            epsFact_porous = self.spongeLayers['x-']/2.
            orientation = np.array([1., 0.])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='absorption',
                                                 orientation=orientation,
                                                 center=center,
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous,
                                                 dragAlpha=dragAlpha,
                                                 dragBeta=dragBeta,
                                                 porosity=porosity)
        if x_p is True:
            center = np.array([sponge_x1 + 0.5 * self.spongeLayers['x+'],
                               sponge_half_height_x1, 0.])
            ind = self.regionIndice['x+']
            flag = self.regionFlags[ind]
            epsFact_porous = self.spongeLayers['x+']/2.
            orientation = np.array([-1., 0.])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='absorption',
                                                 orientation=orientation,
                                                 center=center,
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous,
                                                 dragAlpha=dragAlpha,
                                                 dragBeta=dragBeta,
                                                 porosity=porosity)

    def setGenerationZones(self,  dragAlpha, smoothing, waves=None,
                           wind_speed=(0., 0., 0.), x_n=False, x_p=False,
                           dragBeta=0., porosity=1.):
        """
        Sets regions (x+, x-) to generation zones

        Parameters
        ----------
        dragAlpha: float
            Relaxation zone coefficient.
        smoothing: float
            Smoothing distance (typically 3*he)
        waves: proteus.WaveTools
            Class instance of wave generated from proteus.WaveTools.
        wind_speed: Optional[array_like]
            Speed of wind in generation zone (default is (0., 0., 0.))
        allSponge: bool
            If True, all sponge layers are converted to generation zones.
        x_p: bool
            If True, x+ region is converted to generation zone.
        x_n: bool
            If True, x- region is converted to generation zone.
        dragBeta: Optional[float]
            Relaxation zone coefficient.
        porosity: Optional[float]
            Relaxation zone coefficient.
        """
        sponge_half_height_x0 = 0.5 * (self.x0y0[1] + self.x0y1[1])
        sponge_half_height_x1 = 0.5 * (self.x1y0[1] + self.x1y1[1])
        sponge_x0 = self.x0y0[0]
        sponge_x1 = self.x1y0[0]

        waves = waves
        wind_speed = np.array(wind_speed)
        if x_n or x_p:
            self._attachAuxiliaryVariable('RelaxZones')
        if x_n is True:

            center = np.array([sponge_x0 - 0.5 * self.spongeLayers['x-'],
                               sponge_half_height_x0, 0.])
            ind = self.regionIndice['x-']
            flag = self.regionFlags[ind]
            epsFact_porous = self.spongeLayers['x-']/2.
            orientation = np.array([1., 0.])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='generation',
                                                 orientation=orientation,
                                                 center=center,
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous,
                                                 dragAlpha=dragAlpha,
                                                 dragBeta=dragBeta,
                                                 porosity=porosity,
                                                 smoothing=smoothing)
            self.BC['x-'].setUnsteadyTwoPhaseVelocityInlet(wave=waves,
                                                           wind_speed=wind_speed,
                                                           smoothing=smoothing)
        if x_p is True:

            center = np.array([sponge_x1 + 0.5 * self.spongeLayers['x+'],
                               sponge_half_height_x1, 0.])
            ind = self.regionIndice['x+']
            flag = self.regionFlags[ind]
            epsFact_porous = self.spongeLayers['x+']/2.
            orientation = np.array([-1., 0.])
            self.zones[flag] = bc.RelaxationZone(shape=self,
                                                 zone_type='generation',
                                                 orientation=orientation,
                                                 center=center,
                                                 waves=waves,
                                                 wind_speed=wind_speed,
                                                 epsFact_porous=epsFact_porous,
                                                 dragAlpha=dragAlpha,
                                                 dragBeta=dragBeta,
                                                 porosity=porosity,
                                                 smoothing=smoothing)
            self.BC['x+'].setUnsteadyTwoPhaseVelocityInlet(wave=waves,
                                                           wind_speed=wind_speed,
                                                           smoothing=smoothing)



def assembleDomain(domain):
    """
    This function sets up everything needed for the domain, meshing, and
    AuxiliaryVariables calculations (if any).
    It should always be called after defining and manipulating all the shapes
    to be attached to the domain.

    If any shape of the domain is a ShapeRANS (the predefined shapes are, once this module is imported; so are
    Tank2D, Tank3D and TankWithObstacles2D) the two-phase flow boundary conditions (BC_RANS) are used, the
    boundary between processors is set non-material, the auxiliary variables of the shapes are assembled and the
    mesh is generated unless the domain is a PUMIDomain. A domain without such a shape gets the generic boundary
    conditions (BC_Base) and a mesh.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.
    """
    if any(isinstance(shape, ShapeRANS) for shape in domain.shape_list):
        _assembleGeometry(domain, BC_class=bc.BC_RANS)
        domain.BCbyFlag[0].setNonMaterial()  # set BC for boundary between processors
        assembleAuxiliaryVariables(domain)
        if(domain.name != "PUMIDomain"):
            _generateMesh(domain)
    else:
        # reinitialize geometry of domain
        _assembleGeometry(domain, BC_class=bc.BC_Base)
        _generateMesh(domain)


def assembleAuxiliaryVariables(domain):
    """
    Adds the auxiliary variables to the domain.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        Domain class instance that hold all the geometrical informations and
        boundary conditions of the shape.

    Notes
    -----
    Should be called after assembleGeometry
    """

    domain.auxiliaryVariables = {
        'dissipation': [],
        'kappa': [],
        'ls': [],
        'ls_consrv': [],
        'moveMesh': [],
        'redist': [],
        'twp': [],
        'vof': [],
        'thp': [],
        'vos': [],
        'SWFlow' :[]
    }

    zones_global = {}
    start_region = 0
    start_rflag = 0
    start_flag = 0
    for shape in domain.shape_list:
        aux = domain.auxiliaryVariables
        # ----------------------------
        # RIGID BODIES
        if 'RigidBody' in list(shape.auxiliaryVariables.keys()):
            body = shape.auxiliaryVariables['RigidBody']
            aux['twp'] += [body]
            # fixing mesh on rigid body
            for boundcond in shape.BC_list:
                boundcond.setRigidBodyMoveMesh(body)
            # update the indice for force/moment calculations
            body.i_start = start_flag + 1
            body.i_end = start_flag + 1 + len(shape.BC_list)
        if 'ChRigidBody' in list(shape.auxiliaryVariables.keys()):
            body = shape.auxiliaryVariables['ChRigidBody']
            for boundcond in shape.BC_list:
                boundcond.setChMoveMesh(body)
            if body.boundaryFlags.size == 0:
                flags = []
                for tag, flag in shape.boundaryTags.items():
                    flags += [start_flag+flag]
                body.setBoundaryFlags(flags)
        if 'WallFunction' in list(shape.auxiliaryVariables.keys()):
            wall = shape.auxiliaryVariables['WallFunction']
            for ii in range(len(wall)):
                aux['twp'] += [wall[ii]]
                logEvent('WALL ATTACHED TO AUXVAR --> %s' % wall[ii])
        if 'kWallFunction' in list(shape.auxiliaryVariables.keys()):
            kWall = shape.auxiliaryVariables['kWallFunction']
            for ii in range(len(kWall)):
                aux['kappa'] += [kWall[ii]]
                logEvent('kWALL ATTACHED TO AUXVAR --> %s' % kWall[ii])
        # ----------------------------
        # ABSORPTION/GENERATION ZONES

        if 'RelaxZones' in list(shape.auxiliaryVariables.keys()):
            if not zones_global:
                aux['twp'] += [bc.RelaxationZoneWaveGenerator(zones_global,
                                                              domain.nd)]
                # add relaxation zones to SWFlow AuxiliaryVariables
                aux['SWFlow'] += [bc.RelaxationZoneWaveGenerator(zones_global,
                                                              domain.nd)]                                              
            if not hasattr(domain, 'porosityTypes'):
                # create arrays of default values
                domain.porosityTypes = np.ones(len(domain.regionFlags) + 1)
                domain.dragAlphaTypes = np.zeros(len(domain.regionFlags) + 1)
                domain.dragBetaTypes = np.zeros(len(domain.regionFlags) + 1)
                domain.epsFact_porous = np.zeros(len(domain.regionFlags) + 1)
            i0 = start_region + 1
            for flag, zone in list(shape.zones.items()):
                ind = [i for i, f in enumerate(shape.regionFlags) if f == flag]
                for i1 in ind:
                    domain.porosityTypes[i0 + i1] = zone.porosity
                    domain.dragAlphaTypes[i0 + i1] = zone.dragAlpha
                    domain.dragBetaTypes[i0 + i1] = zone.dragBeta
                    domain.epsFact_porous[i0 + i1] = zone.epsFact_porous
                # update dict with global key instead of local key
                key = flag + start_rflag
                zones_global[key] = zone
        start_flag += len(shape.BC_list)
        # ----------------------------
        # GAUGES
        gauge_dict = {key: shape.auxiliaryVariables.get(key, [])
                      for key in list(shape.auxiliaryVariables.keys())
                      if str(key).startswith('Gauge_')}
        for key in list(gauge_dict.keys()):
            key_name = key.split('_', 1)[1]  # Cutting off "Gauge_" prefix
            if key_name not in aux:
                # It is probably too dangerous to simply put "aux[key_name] = []"
                # as this system is fragile to typos. Instead, we throw an error.
                raise ValueError('ERROR: Gauge key ',
                                 key_name,
                                 ' is not a recognized model by SpatialTools.',
                                 ' The known models in our dictionary are ',
                                 str(list(aux.keys()))
                                 )
            else:
                aux[key_name] += gauge_dict[key]
        if shape.regions is not None:
            start_region += len(shape.regions)
            start_rflag = max(domain.regionFlags[0:start_region])


def get_unit_vector(vector):
    return np.array(vector)/np.linalg.norm(vector)
