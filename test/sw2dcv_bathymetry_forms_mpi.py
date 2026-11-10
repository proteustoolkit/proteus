#!/usr/bin/env python
"""SW2DCV / GN_SW2DCV obstacle_flow with the bathymetry in each accepted form, for test_sw2dcv_bathymetry_forms.py.

The stored Triangle mesh (SWFlow/for_testMeshes/obstacle, he=4.0, 149 nodes). FORM picks how the bathymetry reaches
the coefficients:

- ``callable``: b(X) = 0.01 x + 0.02 y, a function of the coordinates;
- ``1d``: an array of shape (N,), one value per node of the input mesh, in the .node file's order;
- ``col1``, ``col2``, ``col3``: the same values as an (N, k) array whose last column is the bathymetry, the columns
  before it the node's coordinates (x, then y), as a .node/xyz file reads.

The array forms carry b = 0.01 x + 0.02 y + 0.001 i (i the file node), so a node that took another node's entry shows.
GN=1 runs the dispersive model (GN_SW2DCV). After the coefficients have read the bathymetry, every rank writes
b_rank<r>.csv: x, y, file node, b.dof, mesh.nodeArray[:, 2], the expected value, for every node of its subdomain,
ghosts included. Run under mpiexec (1 rank or more) in an empty directory.
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm
import rank_staging

HERE = os.path.dirname(os.path.abspath(__file__))
GN = bool(os.environ.get("GN"))
FORM = os.environ["FORM"]
assert FORM in ("callable", "1d", "col1", "col2", "col3"), FORM
_deck = open(os.path.join(HERE, "SWFlow", "obstacle_flow.py")).read()
_here = "os.path.dirname(os.path.abspath(__file__))"
assert _deck.count(_here) == 1          # where the deck finds its stored mesh: the checkout, not the staged copy
# stage the stored mesh once per rank atomically (rank_staging) and drop the deck's in-place copy
_copy = "copyfile(saved_file, copied_to)"
assert _deck.count(_copy) == 1
_deck = _deck.replace(_copy, "pass")
for _name in ["obstacle.1.neigh", "obstacle.1.poly", "obstacle.edge", "obstacle.ele", "obstacle.node", "obstacle.poly"]:
    rank_staging.write(_name, open(os.path.join(HERE, "SWFlow", "for_testMeshes", _name)).read())
NODE_FILE = os.path.join(HERE, "SWFlow", "for_testMeshes", "obstacle.node")


def node_file_xy():
    """The .node file's coordinates, in its order."""
    rows = [l.split() for l in open(NODE_FILE).read().splitlines() if l.strip() and not l.lstrip().startswith('#')][1:]
    return np.array([[float(r[1]), float(r[2])] for r in sorted(rows, key=lambda r: int(r[0]))])


# the deck's module-level names become Context fields, which may not start with an underscore: one statement
if FORM == "callable":
    _bathymetry = "lambda X: 0.01 * X[0] + 0.02 * X[1]"
else:
    _bathymetry = """(lambda xy, b: {"1d": b, "col1": b[:, None], "col2": np.column_stack([xy[:, 0], b]),
                         "col3": np.column_stack([xy, b])}[%r])(
    np.array([[float(r[1]), float(r[2])] for r in sorted([l.split() for l in open(%r).read().splitlines()
                                                          if l.strip() and not l.lstrip().startswith('#')][1:],
                                                         key=lambda r: int(r[0]))]),
    (lambda xy: 0.01 * xy[:, 0] + 0.02 * xy[:, 1] + 0.001 * np.arange(len(xy)))(
        np.array([[float(r[1]), float(r[2])] for r in sorted([l.split() for l in open(%r).read().splitlines()
                                                              if l.strip() and not l.lstrip().startswith('#')][1:],
                                                             key=lambda r: int(r[0]))])))""" % (FORM, NODE_FILE,
                                                                                                 NODE_FILE)
rank_staging.write("obstacle_flow.py", _deck.replace(_here, repr(os.path.join(HERE, "SWFlow")))
                   + "\nmySWFlowProblem.bathymetry = " + _bathymetry + "\n")
rank_staging.write("__init__.py", "")

from proteus.mprans import SW2DCV, GN_SW2DCV
module = GN_SW2DCV if GN else SW2DCV
_initializeMesh = module.Coefficients.initializeMesh
def initializeMesh(self, mesh):
    given = self.bathymetry.copy() if isinstance(self.bathymetry, np.ndarray) else None
    _initializeMesh(self, mesh)
    if given is not None:
        assert np.array_equal(given, self.bathymetry), "the coefficients changed the caller's bathymetry array"
    xy_file = node_file_xy()
    index = {(x, y): i for i, (x, y) in enumerate(xy_file)}
    file_node = np.array([index[(x, y)] for x, y in mesh.nodeArray[:, :2]])
    x, y = mesh.nodeArray[:, 0], mesh.nodeArray[:, 1]
    expected = 0.01 * x + 0.02 * y + (0.0 if FORM == "callable" else 0.001 * file_node)
    np.savetxt("b_rank%d.csv" % Comm.get().rank(),
               np.column_stack([x, y, file_node, np.asarray(self.b.dof, 'd').reshape(len(x), -1)[:, -1],
                                mesh.nodeArray[:, 2], expected]),
               delimiter=",", header="x,y,file_node,b,z,expected")
    with open("bshape_rank%d.txt" % Comm.get().rank(), "w") as f:
        f.write(repr(np.shape(self.b.dof)))
module.Coefficients.initializeMesh = initializeMesh

sys.argv = (['parun', '--SWEs', 'obstacle_flow.py', '-l1']
            + ['-C', 'he=4.0 final_time=0.01 dt_output=0.01 sw_model=%d' % (1 if GN else 0)])
runpy.run_path(shutil.which('parun'), run_name='__main__')
