#! /usr/bin/env python
from numpy import *
import proteus
from proteus.MeshTools import *
from proteus.FemTools import *

## \ingroup scripts
#
# \file readFun.py
#
# \brief A script for reading a tetrahedral ymf mesh (an ADH .3dm converted by proteustoolkit/ymf-import-sms)
# and a .data file to build a finite element function.

def readFun():
    from optparse import OptionParser
    usage = "usage: %readFun [options] meshFile funFile funOutFile"
    parser = OptionParser(usage=usage)
    parser.add_option("-m", "--matlab",
                      help="print edges to files for plotting in matlab",
                      action="store_true",
                      dest="matlab",
                      default=False)
    parser.add_option("-M", "--view-matlab",
                      help="plot edges in matlab",
                      action="store_true",
                      dest="viewMatlab",
                      default=False)
    parser.add_option("-x", "--nnodes-x",
                      help="use NX nodes in the x direction",
                      action="store",
                      type="int",
                      dest="nx",
                      default=2)
    parser.add_option("-y", "--nnodes-y",
                      help="use NY nodes in the y direction",
                      action="store",
                      type="int",
                      dest="ny",
                      default=2)
    parser.add_option("-z", "--nnodes-z",
                      help="use NZ nodes in the z direction",
                      action="store",
                      type="int",
                      dest="nz",
                      default=2)
    (opts, args) = parser.parse_args()
    if len(args) == 3:
        meshFilename = args[0]
        funFilename = args[1]
        funOutFilename = args[2]
    else:
        print(usage)
        exit(1)
    mesh=TetrahedralMesh()
    # Proteus no longer reads .3dm itself: convert it to a ymf mesh first (meshFile is its prefix)
    mesh.generateFromYmfFile(meshFilename)
    spaceMap = SpaceMapping(mesh)
    funIn = open(funFilename,'r')
    nNodes=0
    fun = []
    data = []
    for i in range(6):
        line = funIn.readline()
        print(line.strip())
        words = line.split()
        if words[0] == 'ND':
            nNodes = int(words[1])
    line = funIn.readline()
    while line.strip() != 'ENDDS':
        print("Reading "+line.strip())
        words = line.split()
        u = zeros(nNodes,Float)
        for i in range(nNodes):
            u[mesh.oldToNewNode[i]] = float(funIn.readline())
        data.append(u)
        fun.append(ScalarFiniteElementFunction(u,
                                            mesh,
                                            LinearNodalBasisTet(),
                                            NodalInterpolationConditions(mesh)))
        line = funIn.readline()
    print("Read %i timesteps" % len(fun))
    #find maximum coorindates (assume they are positive)
    nodesX = mesh.nodeArray[:][0]
    nodesY = mesh.nodeArray[:][1]
    nodesZ = mesh.nodeArray[:][2]
    maxx = max(nodesX)
    maxy = max(nodesY)
    maxz = max(nodesZ)
    minx = min(nodesX)
    miny = min(nodesY)
    minz = min(nodesZ)
    offSet = EVec(minx,miny,minz)
    Lx = maxx - minx
    Ly = maxy - miny
    Lz = maxz - minz
    print("Building rectangular output grid ["+repr(minx)+","+repr(maxx)+"] (x), [" + \
          repr(miny)+","+repr(maxy)+"] (y), [" + \
          repr(minz)+","+repr(maxz)+"] (z)")
    outputGrid = RectangularGrid(opts.nx,opts.ny,opts.nz,Lx,Ly,Lz)
    #now compute the projection of the mesh and solution onto this grid
    eList={}
    xiList={}
    xOut = open('x.grf','w')
    yOut = open('y.grf','w')
    zOut = open('z.grf','w')
    for k in range(outputGrid.nz):
        for j in range(outputGrid.ny):
            for i in range(outputGrid.nx):
                # RectangularGrid's method is getNode, not get_node --
                # unrelated to the PyTables get_node elsewhere, but
                # equally an AttributeError on every call
                X = outputGrid.getNode(i,j,k)+offSet
                e = spaceMap.findElement(X)
                eList[(i,j,k)]=e
                if e!='OFFMESH':
                    xiList[(i,j,k)]=spaceMap.inverseAffineMap(e,X)
                xOut.write('%15.8e '% X.x())
                yOut.write('%15.8e '% X.y())
                zOut.write('%15.8e '% X.z())
            xOut.write('\n')
            yOut.write('\n')
            zOut.write('\n')
    xOut.close()
    yOut.close()
    zOut.close()
    U = zeros((opts.nx,opts.ny,opts.nz),Float)
    for ts,f in enumerate(fun):
        funOut = open(funOutFilename+repr(ts)+'.grf','w')
        for k in range(outputGrid.nz):
            for j in range(outputGrid.ny):
                for i in range(outputGrid.nx):
                    if eList[(i,j,k)] != 'OFFMESH':
                        U[i,j,k] = f.u_of_xi(eList[(i,j,k)],xiList[(i,j,k)])
                    else:
                        U[i,j,k] = float('nan')
                    funOut.write('%15.8e '% U[i,j,k])
                funOut.write('\n')
        funOut.close()
if __name__ == '__main__':
#      import sys
#      import profile
#      profile.run('readFun(sys.argv[1],sys.argv[2],sys.argv[3])','readFunProf')
#      import pstats
#      p = pstats.Stats('readFunProf')
#      p.sort_stats('cumulative').print_stats(20)
#      p.sort_stats('time').print_stats(20)
    readFun()
