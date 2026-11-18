"""
Testing module for Domain.py, Shape.py, BC.py
Work in progress
TO DO:
test inertia
test rigid body calculations
"""
import unittest
import numpy.testing as npt
import numpy as np
import os
from proteus import Comm, Profiling, Gauges
from proteus.Profiling import logEvent as log
from proteus.Domain import (PiecewiseLinearComplexDomain,
                            PlanarStraightLineGraphDomain)
from proteus.SpatialTools import (Rectangle,
                                  Cuboid,
                                  CustomShape,
                                  assembleDomain)
from proteus.mprans.SpatialTools import (Rectangle as RectangleRANS,
                                         Cuboid as CuboidRANS,
                                         CustomShape as CustomShapeRANS,
                                         assembleDomain as assembleDomainRANS,
                                         Tank2D,
                                         Tank3D,
                                         ShapeSTL)
from proteus.mprans.BodyDynamics import RigidBody
import pytest

comm = Comm.init()
Profiling.procID = comm.rank()

log("Testing SpatialTools")

def create_domain2D():
    return PlanarStraightLineGraphDomain()

def create_domain3D():
    return PiecewiseLinearComplexDomain()

def create_rectangle(domain, dim=(0., 0.), coords=(0., 0.), folder=None):
    if folder is None:
        return Rectangle(domain, dim, coords)
    elif folder == 'mprans':
        return RectangleRANS(domain, dim, coords)

def create_cuboid(domain, dim=(0., 0., 0.), coords=(0., 0., 0.), folder=None):
    if folder is None:
        return Cuboid(domain, dim, coords)
    elif folder == 'mprans':
        return CuboidRANS(domain, dim, coords)

def create_custom2D(domain, folder=None):
    domain2D = domain
    bt2D = {'bottom': 1, 'right': 2, 'left': 3, 'top': 4}
    vertices2D = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    vertexFlags2D = [bt2D['bottom'], bt2D['bottom'], bt2D['top'], bt2D['top']]
    segments2D = [[0, 1], [1, 2], [2, 3], [3, 0]]
    segmentFlags2D = [bt2D['bottom'], bt2D['right'], bt2D['top'], bt2D['left']]
    if folder is None:
        custom = CustomShape
    elif folder == 'mprans':
        custom = CustomShapeRANS
    custom2D = custom(domain=domain2D, vertices=vertices2D,
                      vertexFlags=vertexFlags2D, segments=segments2D,
                      segmentFlags=segmentFlags2D, boundaryTags=bt2D)
    return custom2D

def create_custom3D(domain, folder=None):
    domain3D = domain
    bt3D = {'bottom': 1, 'front':2, 'right':3, 'back': 4, 'left':5,
            'top':6}
    vertices3D = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                  [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]]
    vertexFlags3D = [bt3D['left'], bt3D['right'], bt3D['right'], bt3D['left'],
                     bt3D['left'], bt3D['right'], bt3D['right'], bt3D['left']]
    facets3D = [[[0, 1, 2, 3]], [[0, 1, 5, 4]], [[1, 2, 6, 5]], [[2, 3, 7, 6]],
                [[3, 0, 4, 7]], [[4, 5, 6, 7]]]
    facetFlags3D = [bt3D['bottom'], bt3D['front'], bt3D['right'], bt3D['back'],
                    bt3D['left'], bt3D['top']]
    if folder is None:
        custom = CustomShape
    elif folder == 'mprans':
        custom = CustomShapeRANS
    custom3D = custom(domain=domain3D, vertices=vertices3D,
                      vertexFlags=vertexFlags3D, facets=facets3D,
                      facetFlags=facetFlags3D, boundaryTags=bt3D)
    return custom3D

def create_tank2D(domain, dim=(0., 0.), coords=None):
    return Tank2D(domain, dim, coords)

def create_tank3D(domain, dim=(0., 0., 0.), coords=None):
    return Tank3D(domain, dim, coords)
def create_stl3D(domain,stlfile):
    return ShapeSTL(domain,stlfile)


class TestShapeDomainBuilding(unittest.TestCase):

    # NOTE: need thorough evaluation -- unskipped this session after finding
    # test/STLBlocks.stl (an input fixture ShapeSTL reads, not a comparison
    # baseline) was simply missing from this checkout; real fixture was
    # supplied and no further code changes were needed for this test.
    def test_create_shapes(self):
        """
        Testing if shapes can be created
        """
        domain2D = create_domain2D()
        domain3D = create_domain3D()
        rectangle = create_rectangle(domain2D)
        rectangleRANS = create_rectangle(domain2D, folder='mprans')
        cuboid = create_cuboid(domain3D)
        cuboidRANS = create_cuboid(domain3D, folder='mprans')
        tand2D = create_tank2D(domain2D)
        tank3D = create_tank3D(domain3D)
        custom2D = create_custom2D(domain2D)
        custom2DRANS = create_custom2D(domain2D, folder='mprans')
        custom3D = create_custom3D(domain3D)
        custom3DRANS = create_custom3D(domain3D, folder='mprans')
        path1 = getpath()
        stl = create_stl3D(domain3D,os.path.join(path1,"STLBlocks.stl"))
        
    # NOTE: need thorough evaluation -- unskipped this session after finding
    # test/STLBlocks.stl was simply missing from this checkout (supplied
    # real fixture), and fixing a real NumPy>=2.0 incompatibility in
    # proteus/SpatialTools.py's unique_rows(): np.unique(return_inverse=True)
    # now shapes `inverse` like the (N,1) structured-dtype view instead of
    # flat (N,), which broke inverse[facets] indexing for any ShapeSTL-based
    # domain, not just this test's data.
    def test_assemble_domain(self):
        """
        Testing the assembleDomain() for different domains with multiple shapes
        """
        nb_shapes = 10
        domain2D = create_domain2D()
        domain2DRANS = create_domain2D()
        domain3D = create_domain3D()
        domain3DRANS = create_domain3D()
        domainSTL = create_domain3D()
        dim2D = np.array([1., 1.])
        dim3D = np.array([1., 1., 1.])
        coords2D = np.array([0.5, 0.5])
        coords3D = np.array([0.5, 0.5, 0.5])
        nb_bc2D = 0
        nb_bc2DRANS = 0
        nb_bc3D = 0
        nb_bc3DRANS = 0
        for shape in range(nb_shapes):
            coords2D += 1.5
            coords3D += 1.5
            a = create_rectangle(domain2D, dim=dim2D, coords=coords2D)
            nb_bc2D += len(a.BC_list)
            a = create_cuboid(domain3D, dim=dim3D, coords=coords3D)
            nb_bc3D += len(a.BC_list)
            a = create_rectangle(domain2DRANS, dim=dim2D, coords=coords2D,
                                folder='mprans')
            nb_bc2DRANS += len(a.BC_list)
            a = create_cuboid(domain3DRANS, dim=dim3D, coords=coords3D,
                            folder='mprans')
            nb_bc3DRANS += len(a.BC_list)
        a = create_tank2D(domain2DRANS, dim=[50., 50.])
        nb_bc2DRANS += len(a.BC_list)
        a = create_tank3D(domain3DRANS, dim=[50., 50., 50.])
        nb_bc3DRANS += len(a.BC_list)
        assembleDomain(domain2D)
        assembleDomain(domain3D)
        assembleDomainRANS(domain2DRANS)
        assembleDomainRANS(domain3DRANS)

        x2D = domain2D.x
        x3D = domain3D.x
        x2DRANS = domain2DRANS.x
        x3DRANS = domain3DRANS.x
        L2D = domain2D.L
        L3D = domain3D.L
        L2DRANS = domain2DRANS.L
        L3DRANS = domain3DRANS.L

        # check that each domain has the right number of shapes
        npt.assert_equal(len(domain2D.shape_list), nb_shapes)
        npt.assert_equal(len(domain3D.shape_list), nb_shapes)
        npt.assert_equal(len(domain2DRANS.shape_list), nb_shapes+1)
        npt.assert_equal(len(domain3DRANS.shape_list), nb_shapes+1)

        # check that the number of boundary conditions is right
        npt.assert_equal(len(domain2D.bc), nb_bc2D+1)
        npt.assert_equal(len(domain3D.bc), nb_bc3D+1)
        npt.assert_equal(len(domain2DRANS.bc), nb_bc2DRANS+1)
        npt.assert_equal(len(domain3DRANS.bc), nb_bc3DRANS+1)

        # check that bounding boxes are rightly calculated
        npt.assert_equal(L2D, [14.5, 14.5])
        npt.assert_equal(L3D, [14.5, 14.5, 14.5])
        npt.assert_equal(L2DRANS, [50., 50.])
        npt.assert_equal(L3DRANS, [50., 50., 50.])
        npt.assert_equal(x2D, [1.5, 1.5])
        npt.assert_equal(x3D, [1.5, 1.5, 1.5])
        npt.assert_equal(x2DRANS, [0., 0.])
        npt.assert_equal(x3DRANS, [0., 0., 0.])

        #stl domain checks
        path1 = getpath()
        stl = create_stl3D(domainSTL,os.path.join(path1,"STLBlocks.stl"))
        assembleDomainRANS(domainSTL)
        STLnames = ["Bed0","Concrete0","Inlet0","Outlet0","Top0","Wall0"]
        nSTLs = len(STLnames)
        npt.assert_equal(nSTLs,max(stl.vertexFlags))
        npt.assert_equal(nSTLs,max(stl.facetFlags))
        npt.assert_equal(nSTLs,max(domainSTL.vertexFlags))
        npt.assert_equal(nSTLs,max(domainSTL.facetFlags))

    # NOTE: need thorough evaluation -- unskipped this session after finding
    # test/STLBlocks.stl was simply missing from this checkout (supplied
    # real fixture), and fixing a real NumPy>=2.0 incompatibility in
    # proteus/SpatialTools.py's unique_rows() (see test_assemble_domain
    # above for details).
    def test_BC_flags(self):
        """
        Testing the flags of shapes and their in their domain
        """
        nb_shapes = 3
        domain2D = create_domain2D()
        domain3D = create_domain3D()
        domain2DRANS = create_domain2D()
        domain3DRANS = create_domain3D()
        domainSTL = create_domain3D()
        flags_v2D = []
        flags_s2D = []
        flags_v3D = []
        flags_f3D = []
        flags_v2DRANS = []
        flags_s2DRANS = []
        flags_v3DRANS = []
        flags_f3DRANS = []
        maxf = 0
        for i in range(nb_shapes):
            # 2D
            a = create_custom2D(domain2D)
            if flags_v2D:
                maxf = np.max([np.max(flags_v2D), np.max(flags_s2D)])
            flags_v2D += (a.vertexFlags+maxf).tolist()
            flags_s2D += (a.segmentFlags+maxf).tolist()
            # 3D
            a = create_custom3D(domain3D)
            if flags_v3D:
                maxf = np.max([np.max(flags_v3D), np.max(flags_f3D)])
            flags_v3D += (a.vertexFlags+maxf).tolist()
            flags_f3D += (a.facetFlags+maxf).tolist()
            # 2D RANS
            a = create_custom2D(domain2DRANS, folder='mprans')
            if flags_v2DRANS:
                maxf = np.max([np.max(flags_v2DRANS), np.max(flags_s2DRANS)])
            flags_v2DRANS += (a.vertexFlags+maxf).tolist()
            flags_s2DRANS += (a.segmentFlags+maxf).tolist()
            # 3D RANS
            a = create_custom3D(domain3DRANS, folder='mprans')
            if flags_v3DRANS:
                maxf = np.max([np.max(flags_v3DRANS), np.max(flags_f3DRANS)])
            flags_v3DRANS += (a.vertexFlags+maxf).tolist()
            flags_f3DRANS += (a.facetFlags+maxf).tolist()
        assembleDomain(domain2D)
        assembleDomain(domain3D)
        assembleDomainRANS(domain2DRANS)
        assembleDomainRANS(domain3DRANS)

        # test flags
        npt.assert_equal(domain2D.vertexFlags, flags_v2D)
        npt.assert_equal(domain2D.segmentFlags, flags_s2D)
        npt.assert_equal(domain3D.vertexFlags, flags_v3D)
        npt.assert_equal(domain3D.facetFlags, flags_f3D)
        npt.assert_equal(domain2DRANS.vertexFlags, flags_v2DRANS)
        npt.assert_equal(domain2DRANS.segmentFlags, flags_s2DRANS)
        npt.assert_equal(domain3DRANS.vertexFlags, flags_v3DRANS)
        npt.assert_equal(domain3DRANS.facetFlags, flags_f3DRANS)
        #stl flags
        path1 = getpath()
        stl = create_stl3D(domainSTL,os.path.join(path1,"STLBlocks.stl"))
        assembleDomainRANS(domainSTL)
        STLnames = ["Bed0","Concrete0","Inlet0","Outlet0","Top0","Wall0"]
        nSTLs = len(STLnames)
        j = 0
        for key,value in stl.boundaryTags.items():
            self.assertTrue(STLnames[value-1]==key)
        for key,value in domainSTL.boundaryTags.items():
            self.assertTrue(stl.name+'_'+STLnames[value-1]==key)

        npt.assert_equal(nSTLs,len(stl.boundaryTags))
        

class TestShapeRANS(unittest.TestCase):

    def test_gauges(self):
        domain = create_domain2D()
        tank = create_tank2D(domain)
        tank.attachPointGauges('kappa',
                               gauges = ((('k', ),((0.,0.,0.),)),),
                               activeTime = (0.,1.),
                               sampleRate = 0.5,
                               fileName = 'point1.csv')
        tank.attachLineGauges('kappa',
                               gauges = ((('k', ),(((0.,0.,0.),(0.,0.,1.)),)),),
                               activeTime = (0.,2.),
                               sampleRate = 0.5,
                               fileName = 'line1.csv')
        tank.attachLineIntegralGauges('vof',
                                      gauges=(
                                      (('vof',), (((0., 0., 0.), (0., 0., 1.)),)),),
                                      activeTime=(0., 2.),
                                      sampleRate=0.5,
                                      fileName='line_int1.csv')
        assert tank.auxiliaryVariables.get('Gauge_kappa', None) is not None
        assert tank.auxiliaryVariables.get('Gauge_vof', None) is not None
        assert len(tank.auxiliaryVariables['Gauge_kappa']) is 2
        assert len(tank.auxiliaryVariables['Gauge_vof']) is 1
        self.assertIsInstance(tank.auxiliaryVariables['Gauge_kappa'][0],
                              Gauges.Gauges)
        assert tank.auxiliaryVariables['Gauge_kappa'][0].activeTime == (0.,1.)
        assert tank.auxiliaryVariables['Gauge_kappa'][1].activeTime == (0.,2.)
        assert tank.auxiliaryVariables['Gauge_vof'][0].sampleRate == 0.5

        assembleDomainRANS(domain)

        assert domain.auxiliaryVariables.get('kappa', None) is not None
        assert domain.auxiliaryVariables.get('vof', None) is not None
        self.assertIsInstance(domain.auxiliaryVariables['vof'][0],
                              Gauges.Gauges)
        assert len(domain.auxiliaryVariables['kappa']) is 2
        assert len(domain.auxiliaryVariables['vof']) is 1
        assert domain.auxiliaryVariables['kappa'][0].activeTime == (0., 1.)
        assert domain.auxiliaryVariables['kappa'][1].activeTime == (0., 2.)
        assert domain.auxiliaryVariables['vof'][0].sampleRate == 0.5

        tank.attachPointGauges('voff',
                                gauges=((('vof',), ((0., 0., 0.),)),),
                                activeTime=(0., 1.),
                                sampleRate=0.5,
                                fileName='point2.csv')
        self.assertRaises(ValueError,assembleDomainRANS,domain)


    def test_zones_accept_numpy_integer_flags(self):
        """A flag taken from shape.regionFlags is a numpy integer: the zone
        setters tested isinstance(flags, int) and iterated over it
        (TypeError: 'numpy.int64' object is not iterable)."""
        for flag in (np.int32(1), np.int64(1), np.array([1, 2])[0]):
            domain = create_domain2D()
            custom = create_custom2D(domain, 'mprans')
            custom.setAbsorptionZones(flags=flag, epsFact_porous=0.5,
                                      center=[0.5, 0., 0.], orientation=[1., 0., 0.],
                                      vert_axis=0, dragAlpha=1., dragBeta=0.,
                                      porosity=1.)
            self.assertIn(1, custom.zones)
            self.assertEqual(custom.zones[1].zone_type, 'absorption')
            domain = create_domain2D()
            custom = create_custom2D(domain, 'mprans')
            custom.setPorousZones(flags=flag, dragAlpha=1., dragBeta=0., porosity=0.4)
            self.assertIn(1, custom.zones)
            self.assertEqual(custom.zones[1].zone_type, 'porous')

    def test_absorption_zones(self):
        flag = 1
        epsFact_porous = 0.5
        center = [0.5, 0., 0.]
        orientation = [1., 0., 0.]
        dragAlpha = 0.5/1.005e-6
        dragBeta = 0.
        porosity = 1.
        domain = create_domain2D()
        # for custom (same principle for rectangle or cuboid or STL)
        custom = create_custom2D(domain, 'mprans')
        custom.setAbsorptionZones(flags=flag, epsFact_porous=epsFact_porous,
                                  center=center, orientation=orientation,vert_axis=0,
                                  dragAlpha=dragAlpha, dragBeta=dragBeta,
                                  porosity=porosity)
        npt.assert_equal(custom.auxiliaryVariables['RelaxZones'], custom.zones)
        zone = custom.zones[flag]
        npt.assert_equal(zone.zone_type, 'absorption')
        npt.assert_equal(zone.center[0], center[0])
        npt.assert_equal(zone.center[1], center[1])
        npt.assert_equal(zone.center[2], center[2])
        npt.assert_equal(zone.orientation[0], orientation[0])
        npt.assert_equal(zone.orientation[1], orientation[1])
        npt.assert_equal(zone.orientation[2], orientation[2])
        npt.assert_equal(zone.dragAlpha, dragAlpha)
        npt.assert_equal(zone.dragBeta, dragBeta)
        npt.assert_equal(zone.porosity, porosity)
        npt.assert_equal(zone.Shape, custom)
        npt.assert_equal(zone.vert_axis, 0)
        # for tanks in 2D
        tank = create_tank2D(domain, dim=[4., 4.])
        tank.setSponge(x_n=1.5, x_p=2.)
        npt.assert_equal(tank.spongeLayers['x-'], 1.5)
        npt.assert_equal(tank.spongeLayers['x+'], 2.)
        tank.setAbsorptionZones(dragAlpha ,x_n=True, x_p=True)
        leftzone = tank.zones[tank.regionFlags[tank.regionIndice['x-']]]
        rightzone = tank.zones[tank.regionFlags[tank.regionIndice['x+']]]
        npt.assert_equal(leftzone.zone_type, 'absorption')
        npt.assert_equal(leftzone.center[0], -0.75)
        npt.assert_equal(leftzone.center[1], 2.)
        npt.assert_equal(leftzone.orientation[0], 1.)
        npt.assert_equal(leftzone.orientation[1], 0.)
        npt.assert_equal(leftzone.dragAlpha, dragAlpha)
        npt.assert_equal(leftzone.dragBeta, dragBeta)
        npt.assert_equal(leftzone.porosity, porosity)
        npt.assert_equal(leftzone.Shape, tank)
        npt.assert_equal(rightzone.zone_type, 'absorption')
        npt.assert_equal(rightzone.center[0], 5.)
        npt.assert_equal(rightzone.center[1], 2.)
        npt.assert_equal(rightzone.orientation[0], -1.)
        npt.assert_equal(rightzone.orientation[1], 0.)
        npt.assert_equal(rightzone.dragAlpha, dragAlpha)
        npt.assert_equal(rightzone.dragBeta, dragBeta)
        npt.assert_equal(rightzone.porosity, porosity)
        npt.assert_equal(rightzone.Shape, tank)
        # for tanks in 3D
        domain = create_domain3D()
        tank = create_tank3D(domain, dim=[4., 4., 4.])
        tank.setSponge(x_n=1.5, x_p=2., y_n=3., y_p=0.2)
        npt.assert_equal(tank.spongeLayers['x-'], 1.5)
        npt.assert_equal(tank.spongeLayers['x+'], 2.)
        npt.assert_equal(tank.spongeLayers['y-'], 3)
        npt.assert_equal(tank.spongeLayers['y+'], 0.2)
        tank.setAbsorptionZones(dragAlpha, x_n=True, x_p=True, y_n=True, y_p=True)
        leftzone = tank.zones[tank.regionFlags[tank.regionIndice['x-']]]
        rightzone = tank.zones[tank.regionFlags[tank.regionIndice['x+']]]
        frontzone = tank.zones[tank.regionFlags[tank.regionIndice['y-']]]
        backzone = tank.zones[tank.regionFlags[tank.regionIndice['y+']]]
        npt.assert_equal(leftzone.zone_type, 'absorption')
        npt.assert_equal(leftzone.center[0], -0.75)
        npt.assert_equal(leftzone.center[1], 2.)
        npt.assert_equal(leftzone.center[2], 2.)
        npt.assert_equal(leftzone.orientation[0], 1.)
        npt.assert_equal(leftzone.orientation[1], 0.)
        npt.assert_equal(leftzone.orientation[2], 0.)
        npt.assert_equal(leftzone.dragAlpha, dragAlpha)
        npt.assert_equal(leftzone.dragBeta, dragBeta)
        npt.assert_equal(leftzone.porosity, porosity)
        npt.assert_equal(leftzone.Shape, tank)
        npt.assert_equal(rightzone.zone_type, 'absorption')
        npt.assert_equal(rightzone.center[0], 5.)
        npt.assert_equal(rightzone.center[1], 2.)
        npt.assert_equal(rightzone.center[2], 2.)
        npt.assert_equal(rightzone.orientation[0], -1.)
        npt.assert_equal(rightzone.orientation[1], 0.)
        npt.assert_equal(rightzone.orientation[2], 0.)
        npt.assert_equal(rightzone.dragAlpha, dragAlpha)
        npt.assert_equal(rightzone.dragBeta, dragBeta)
        npt.assert_equal(rightzone.porosity, porosity)
        npt.assert_equal(rightzone.Shape, tank)
        npt.assert_equal(frontzone.zone_type, 'absorption')
        npt.assert_equal(frontzone.center[0], 2.)
        npt.assert_equal(frontzone.center[1], -1.5)
        npt.assert_equal(frontzone.center[2], 2.)
        npt.assert_equal(frontzone.orientation[0], 0.)
        npt.assert_equal(frontzone.orientation[1], 1.)
        npt.assert_equal(frontzone.orientation[2], 0.)
        npt.assert_equal(frontzone.dragAlpha, dragAlpha)
        npt.assert_equal(frontzone.dragBeta, dragBeta)
        npt.assert_equal(frontzone.porosity, porosity)
        npt.assert_equal(frontzone.Shape, tank)
        npt.assert_equal(backzone.zone_type, 'absorption')
        npt.assert_equal(backzone.center[0], 2.)
        npt.assert_equal(backzone.center[1], 4.1)
        npt.assert_equal(backzone.center[2], 2.)
        npt.assert_equal(backzone.orientation[0], 0.)
        npt.assert_equal(backzone.orientation[1], -1.)
        npt.assert_equal(backzone.orientation[2], 0.)
        npt.assert_equal(backzone.dragAlpha, dragAlpha)
        npt.assert_equal(backzone.dragBeta, dragBeta)
        npt.assert_equal(backzone.porosity, porosity)
        npt.assert_equal(backzone.Shape, tank)

    def test_generation_zones(self):
        flag = 1
        epsFact_porous = 0.5
        center = [0.5, 0., 0.]
        orientation = [1., 0., 0.]
        dragAlpha = 0.5/1.005e-6
        dragBeta = 0.
        porosity = 1.
        from proteus import WaveTools as wt
        waves = wt.MonochromaticWaves(period=0.8, waveHeight=0.029, mwl=0.9,
                                      depth=0.9, g=np.array([0., -9.81, 0.]),
                                      waveDir=np.array([1., 0., 0.]))
        domain = create_domain2D()
        # for custom (same principle for rectangle or cuboid or STL)
        custom = create_custom2D(domain, 'mprans')
        custom.setGenerationZones(flags=flag, epsFact_porous=epsFact_porous,
                                  center=center, orientation=orientation,
                                  vert_axis=0,waves=waves, dragAlpha=dragAlpha,
                                  dragBeta=dragBeta, porosity=porosity)
        npt.assert_equal(custom.auxiliaryVariables['RelaxZones'], custom.zones)
        zone = custom.zones[flag]
        npt.assert_equal(zone.zone_type, 'generation')
        npt.assert_equal(zone.center[0], center[0])
        npt.assert_equal(zone.center[1], center[1])
        npt.assert_equal(zone.center[2], center[2])
        npt.assert_equal(zone.orientation[0], orientation[0])
        npt.assert_equal(zone.orientation[1], orientation[1])
        npt.assert_equal(zone.orientation[2], orientation[2])
        npt.assert_equal(zone.dragAlpha, dragAlpha)
        npt.assert_equal(zone.dragBeta, dragBeta)
        npt.assert_equal(zone.porosity, porosity)
        npt.assert_equal(zone.Shape, custom)
        npt.assert_equal(zone.vert_axis, 0)
        # for tanks in 2D
        tank = create_tank2D(domain, dim=[4., 4.])
        tank.setSponge(x_n=1.5, x_p=2.)
        npt.assert_equal(tank.spongeLayers['x-'], 1.5)
        npt.assert_equal(tank.spongeLayers['x+'], 2.)
        tank.setGenerationZones(dragAlpha, smoothing = 0.,waves=waves, x_n=True, x_p=True)
        leftzone = tank.zones[tank.regionFlags[tank.regionIndice['x-']]]
        rightzone = tank.zones[tank.regionFlags[tank.regionIndice['x+']]]
        npt.assert_equal(leftzone.zone_type, 'generation')
        npt.assert_equal(leftzone.center[0], -0.75)
        npt.assert_equal(leftzone.center[1], 2.)
        npt.assert_equal(leftzone.orientation[0], 1.)
        npt.assert_equal(leftzone.orientation[1], 0.)
        npt.assert_equal(leftzone.dragAlpha, dragAlpha)
        npt.assert_equal(leftzone.dragBeta, dragBeta)
        npt.assert_equal(leftzone.porosity, porosity)
        npt.assert_equal(leftzone.Shape, tank)
        npt.assert_equal(rightzone.zone_type, 'generation')
        npt.assert_equal(rightzone.center[0], 5.)
        npt.assert_equal(rightzone.center[1], 2.)
        npt.assert_equal(rightzone.orientation[0], -1.)
        npt.assert_equal(rightzone.orientation[1], 0.)
        npt.assert_equal(rightzone.dragAlpha, dragAlpha)
        npt.assert_equal(rightzone.dragBeta, dragBeta)
        npt.assert_equal(rightzone.porosity, porosity)
        npt.assert_equal(rightzone.Shape, tank)
        # test translation
        tank = create_tank2D(domain, dim=[4., 4.])
        tank.setSponge(x_n=1.5, x_p=2.)
        tr_val = np.array([1.1, 1.2])
        tank.translate(tr_val)
        npt.assert_equal(tank.spongeLayers['x-'], 1.5)
        npt.assert_equal(tank.spongeLayers['x+'], 2.)
        tank.setGenerationZones(dragAlpha, smoothing = 0.,waves=waves, x_n=True, x_p=True)
        leftzone = tank.zones[tank.regionFlags[tank.regionIndice['x-']]]
        rightzone = tank.zones[tank.regionFlags[tank.regionIndice['x+']]]
        npt.assert_equal(leftzone.zone_type, 'generation')
        npt.assert_equal(leftzone.center[0], -0.75+tr_val[0])
        npt.assert_equal(leftzone.center[1], 2.+tr_val[1])
        npt.assert_equal(leftzone.orientation[0], 1.)
        npt.assert_equal(leftzone.orientation[1], 0.)
        npt.assert_equal(leftzone.dragAlpha, dragAlpha)
        npt.assert_equal(leftzone.dragBeta, dragBeta)
        npt.assert_equal(leftzone.porosity, porosity)
        npt.assert_equal(leftzone.Shape, tank)
        npt.assert_equal(rightzone.zone_type, 'generation')
        npt.assert_equal(rightzone.center[0], 5.+tr_val[0])
        npt.assert_equal(rightzone.center[1], 2.+tr_val[1])
        npt.assert_equal(rightzone.orientation[0], -1.)
        npt.assert_equal(rightzone.orientation[1], 0.)
        npt.assert_equal(rightzone.dragAlpha, dragAlpha)
        npt.assert_equal(rightzone.dragBeta, dragBeta)
        npt.assert_equal(rightzone.porosity, porosity)
        npt.assert_equal(rightzone.Shape, tank)
        # for tanks in 3D
        domain = create_domain3D()
        tank = create_tank3D(domain, dim=[4., 4., 4.])
        tank.setSponge(x_n=1.5, x_p=2., y_n=3., y_p=0.2)
        npt.assert_equal(tank.spongeLayers['x-'], 1.5)
        npt.assert_equal(tank.spongeLayers['x+'], 2.)
        npt.assert_equal(tank.spongeLayers['y-'], 3)
        npt.assert_equal(tank.spongeLayers['y+'], 0.2)
        tank.setGenerationZones(dragAlpha, smoothing=0., waves=waves, x_n=True, x_p=True, y_n=True, y_p=True)
        leftzone = tank.zones[tank.regionFlags[tank.regionIndice['x-']]]
        rightzone = tank.zones[tank.regionFlags[tank.regionIndice['x+']]]
        frontzone = tank.zones[tank.regionFlags[tank.regionIndice['y-']]]
        backzone = tank.zones[tank.regionFlags[tank.regionIndice['y+']]]
        npt.assert_equal(leftzone.zone_type, 'generation')
        npt.assert_equal(leftzone.center[0], -0.75)
        npt.assert_equal(leftzone.center[1], 2.)
        npt.assert_equal(leftzone.center[2], 2.)
        npt.assert_equal(leftzone.orientation[0], 1.)
        npt.assert_equal(leftzone.orientation[1], 0.)
        npt.assert_equal(leftzone.orientation[2], 0.)
        npt.assert_equal(leftzone.dragAlpha, dragAlpha)
        npt.assert_equal(leftzone.dragBeta, dragBeta)
        npt.assert_equal(leftzone.porosity, porosity)
        npt.assert_equal(leftzone.Shape, tank)
        npt.assert_equal(rightzone.zone_type, 'generation')
        npt.assert_equal(rightzone.center[0], 5.)
        npt.assert_equal(rightzone.center[1], 2.)
        npt.assert_equal(rightzone.center[2], 2.)
        npt.assert_equal(rightzone.orientation[0], -1.)
        npt.assert_equal(rightzone.orientation[1], 0.)
        npt.assert_equal(rightzone.orientation[2], 0.)
        npt.assert_equal(rightzone.dragAlpha, dragAlpha)
        npt.assert_equal(rightzone.dragBeta, dragBeta)
        npt.assert_equal(rightzone.porosity, porosity)
        npt.assert_equal(rightzone.Shape, tank)
        npt.assert_equal(frontzone.zone_type, 'generation')
        npt.assert_equal(frontzone.center[0], 2.)
        npt.assert_equal(frontzone.center[1], -1.5)
        npt.assert_equal(frontzone.center[2], 2.)
        npt.assert_equal(frontzone.orientation[0], 0.)
        npt.assert_equal(frontzone.orientation[1], 1.)
        npt.assert_equal(frontzone.orientation[2], 0.)
        npt.assert_equal(frontzone.dragAlpha, dragAlpha)
        npt.assert_equal(frontzone.dragBeta, dragBeta)
        npt.assert_equal(frontzone.porosity, porosity)
        npt.assert_equal(frontzone.Shape, tank)
        npt.assert_equal(backzone.zone_type, 'generation')
        npt.assert_equal(backzone.center[0], 2.)
        npt.assert_equal(backzone.center[1], 4.1)
        npt.assert_equal(backzone.center[2], 2.)
        npt.assert_equal(backzone.orientation[0], 0.)
        npt.assert_equal(backzone.orientation[1], -1.)
        npt.assert_equal(backzone.orientation[2], 0.)
        npt.assert_equal(backzone.dragAlpha, dragAlpha)
        npt.assert_equal(backzone.dragBeta, dragBeta)
        npt.assert_equal(backzone.porosity, porosity)
        npt.assert_equal(backzone.Shape, tank)

    def test_porous_zones(self):
        flag = 1
        dragAlpha = 0.5/1.005e-6
        dragBeta = 0.
        porosity = 1.
        domain = create_domain2D()
        # for custom (same principle for rectangle or cuboid)
        custom = create_custom2D(domain, 'mprans')
        custom.setPorousZones(flags=flag, dragAlpha=dragAlpha, dragBeta=dragBeta,
                              porosity=porosity)
        npt.assert_equal(custom.auxiliaryVariables['RelaxZones'], custom.zones)
        zone = custom.zones[flag]
        npt.assert_equal(zone.zone_type, 'porous')
        npt.assert_equal(zone.center, None)
        npt.assert_equal(zone.orientation, None)
        npt.assert_equal(zone.dragAlpha, dragAlpha)
        npt.assert_equal(zone.dragBeta, dragBeta)
        npt.assert_equal(zone.porosity, porosity)
        npt.assert_equal(zone.Shape, custom)
def getpath():
    path = os.path.dirname(os.path.abspath(__file__))
    return path    
if __name__ == '__main__':

    unittest.main(verbosity=2)
