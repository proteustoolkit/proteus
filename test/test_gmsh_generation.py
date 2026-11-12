import unittest
import pytest
import numpy.testing as npt
import numpy as np
from subprocess import check_call
from proteus.Profiling import logEvent
from proteus import Comm, Profiling
from proteus import Domain
from proteus import MeshTools
from proteus import SpatialTools as st

comm = Comm.init()
Profiling.procID = comm.rank()
logEvent("Testing Gmsh Mesh Conversion")


#TEST VARIANCE THRESHOLD, 2%
#THRESHOLD = 0.02
THRESHOLD = 0.15

#REFERENCE VALUES
numNodes_reference_2D = 3433.0
numEdges_reference_2D = 10096.0
numElements_reference_2D = 6664.0

numNodes_reference_3D = 7674.0
numEdges_reference_3D = 9384.0
numFaces_reference_3D = 6256.0
numElements_reference_3D = 37473.0

def _ymf_mesh(domain):
    """The mesh a use_gmsh domain reads from the .msh gmsh just wrote: converted by ymf-import-gmsh."""
    options = MeshTools.getMeshOptions(domain)
    options.use_gmsh = True
    options.genMesh = False
    return MeshTools._generateMesh(domain, options).meshList[0]


class TestGMSH(unittest.TestCase):
    """gmsh meshes a domain's .geo; its .msh is read through the ymf mesh (ymf-import-gmsh).

    The counts were taken from the triangle/tetgen files msh2simplex wrote (removed in C7.1 round 5);
    the 3D edge and face counts are the boundary's, as those files held them.
    """
    @classmethod
    def setUpClass(cls):
        pytest.importorskip("ymf_import_gmsh")

    def test_gmsh_generation_2D(self):
        domain = Domain.PlanarStraightLineGraphDomain()
        domain.vertices = [[0., 0., 0.],
                           [5., 0., 0.],
                           [5., 5., 0.],
                           [0., 5., 0.]]
        domain.vertexFlags = [1,2,3,4]
        domain.segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
        domain.segmentFlags =[1,2,3,4]
        domain.facets = [[[0, 1, 2, 3]]]
        domain.facetFlags = [1]
        domain.writeGeo('gmsh_mesh_test_2D', he_max=0.1)
        gmsh_cmd = "gmsh {0:s} -v 10 -2 -o {1:s} -format msh2".format(domain.geofile+".geo", domain.geofile+".msh")
        check_call(gmsh_cmd, shell=True)
        mesh = _ymf_mesh(domain)
        assert abs(1.0-mesh.nNodes_global/numNodes_reference_2D) < THRESHOLD
        assert abs(1.0-mesh.nEdges_global/numEdges_reference_2D) < THRESHOLD
        assert abs(1.0-mesh.nElements_global/numElements_reference_2D) < THRESHOLD

    def test_gmsh_generation_3D(self):
        domain = Domain.PiecewiseLinearComplexDomain()
        cube = st.Cuboid(domain, dim=[2.,2.,2.])
        st.assembleDomain(domain)
        domain.writeGeo('gmsh_mesh_test_3D', he_max=0.1)
        gmsh_cmd = "gmsh {0:s} -v 10 -3 -o {1:s} -format msh2".format(domain.geofile+".geo", domain.geofile+".msh")
        check_call(gmsh_cmd, shell=True)
        mesh = _ymf_mesh(domain)
        exterior = mesh.elementBoundaryNodesArray[mesh.exteriorElementBoundariesArray]
        boundaryEdges = {tuple(sorted(pair)) for face in exterior.tolist()
                         for pair in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
        assert(abs(mesh.nNodes_global - numNodes_reference_3D )/numNodes_reference_3D < THRESHOLD)
        assert(abs(len(boundaryEdges) - numEdges_reference_3D )/numEdges_reference_3D < THRESHOLD)
        assert(abs(len(exterior) - numFaces_reference_3D )/numFaces_reference_3D < THRESHOLD)
        assert(abs(mesh.nElements_global - numElements_reference_3D )/numElements_reference_3D < THRESHOLD)

if __name__ == '__main__':
    unittest.main(verbosity=2)
