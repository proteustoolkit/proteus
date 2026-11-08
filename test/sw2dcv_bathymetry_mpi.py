#!/usr/bin/env python
"""SW2DCV / GN_SW2DCV obstacle_flow with the bathymetry given as an array, for test_sw2dcv_bathymetry_partition.py.

The stored Triangle mesh (SWFlow/for_testMeshes/obstacle, he=4.0, 149 nodes) and a bathymetry array in the .node
file's order, b = 0.01 x + 0.02 y + 0.001 i (i the file node), so a node that took another node's entry shows.
FILES=1 runs ``parun -F`` (the partitioner reads the files); otherwise the mesh is read into memory and
partitionNodes cuts it. GN=1 runs the dispersive model (GN_SW2DCV). After the coefficients have permuted the array,
every rank writes b_rank<r>.csv: x, y, file node, b.dof, the expected value, for every node of its subdomain, ghosts
included, and calls_rank<r>.txt, the partitioner entry points that ran. Run under mpiexec in an empty directory.
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm
import rank_staging

HERE = os.path.dirname(os.path.abspath(__file__))
GN = bool(os.environ.get("GN"))
# every rank stages the same files: atomically (rank_staging). The deck copies the stored mesh into the run
# directory when he=4.0; the appended line replaces its bathymetry function by an array in the file's node order
# (one statement: the deck's module-level names become Context fields, which may not start with an underscore).
_deck = open(os.path.join(HERE, "SWFlow", "obstacle_flow.py")).read()
_here = "os.path.dirname(os.path.abspath(__file__))"
assert _deck.count(_here) == 1          # where the deck finds its stored mesh: the checkout, not the staged copy
rank_staging.write("obstacle_flow.py", _deck.replace(_here, repr(os.path.join(HERE, "SWFlow"))) + """
mySWFlowProblem.bathymetry = (lambda xy: 0.01 * xy[:, 0] + 0.02 * xy[:, 1] + 0.001 * np.arange(len(xy)))(np.array(
    [[float(r[1]), float(r[2])] for r in sorted([l.split() for l in open('obstacle.node').read().splitlines()
                                                 if l.strip() and not l.lstrip().startswith('#')][1:],
                                                key=lambda r: int(r[0]))]))
""")
rank_staging.write("__init__.py", "")

# which partitioner ran (the proof that the case is the path it is named for), into calls_rank<r>.txt
from proteus import cpartitioning
calls = []
for name in ('partitionNodes', 'partitionElements', 'partitionNodesFromTriangleFiles'):
    def wrap(f, name=name):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return f(*args, **kwargs)
        return wrapped
    setattr(cpartitioning, name, wrap(getattr(cpartitioning, name)))

from proteus.mprans import SW2DCV, GN_SW2DCV
module = GN_SW2DCV if GN else SW2DCV
_initializeMesh = module.Coefficients.initializeMesh
def initializeMesh(self, mesh):
    _initializeMesh(self, mesh)
    if not isinstance(self.bathymetry, np.ndarray):
        return
    rows = [l.split() for l in open('obstacle.node').read().splitlines()
            if l.strip() and not l.lstrip().startswith('#')][1:]
    xy = {(float(r[1]), float(r[2])): int(r[0]) - 1 for r in rows}
    file_node = np.array([xy[(x, y)] for x, y in mesh.nodeArray[:, :2]])
    expected = self.bathymetry[file_node]
    np.savetxt("b_rank%d.csv" % Comm.get().rank(),
               np.column_stack([mesh.nodeArray[:, :2], file_node, self.b.dof, expected]),
               delimiter=",", header="x,y,file_node,b,expected")
    with open("calls_rank%d.txt" % Comm.get().rank(), "w") as f:
        f.write(",".join(calls))
module.Coefficients.initializeMesh = initializeMesh

sys.argv = (['parun', '--SWEs', 'obstacle_flow.py', '-l1'] + (['-F'] if os.environ.get("FILES") else [])
            + ['-C', 'he=4.0 final_time=0.01 dt_output=0.01 sw_model=%d' % (1 if GN else 0)])
runpy.run_path(shutil.which('parun'), run_name='__main__')
