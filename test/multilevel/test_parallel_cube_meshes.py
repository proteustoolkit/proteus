#!/usr/bin/env python
"""Quadrilateral and hexahedral meshes, partitioned by nodes.

A node-based partition gives each rank the elements in the stars of the nodes it
owns plus an overlap, and the nodes of those elements. The overlap *nodes* were
taken from the node star, which is built from edges, while the *elements* came
from node-to-element membership. For a simplex every node of an element is an
edge neighbour of every other, so the two agree. For a quadrilateral or
hexahedron they do not: the face and body diagonal nodes of an element are not
edge neighbours. An element in the star of an owned node could therefore
reference nodes the subdomain never received -- and because the subdomain
numbering is a std::map read with operator[], a missing node did not fail, it
silently became subdomain node 0.

On a 3x3x3 hex mesh split over two ranks one rank held all eight elements but
24 of the 27 nodes, one element had a centroid at x=0.188 on a grid whose cells
are 0.5 wide, and the Q1 Poisson solution differed from serial by 2.4%. On three
and four ranks the process crashed. This is the same on origin/main; quads share
the construction and only escaped on the grids they had been tried on.

Separately, the overlap's edge collection looked every pair of an element's nodes
up in the edge map. That is right for a simplex and wrong for a quad or hex, whose
diagonals are not edges; std::map::operator[] turned each into a bogus "edge 0".
"""
import os
import shutil
import subprocess
import sys

import pytest


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _launch(helper, args, n_ranks):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), helper)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, path] + [str(a) for a in args],
                            capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize('kind,n', [('hex', 3), ('hex', 5), ('hex', 7), ('quad', 3), ('quad', 9)])
@pytest.mark.parametrize('n_ranks', [2, 3, 4])
@pytest.mark.parametrize('how', ['generated', 'global'])
def test_every_subdomain_element_is_a_cell_of_the_grid(kind, n, n_ranks, how):
    # also checks every owned element boundary has all its elements on its owner;
    # the quadrilateral partitioner used to hand edges to ranks without them
    out = _launch('cube_mesh_mpi.py', [kind, n, how], n_ranks)
    assert 'CUBE_MESH_OK' in out, out


@pytest.mark.parametrize('kind,n', [('hex', 3), ('hex', 5), ('quad', 5)])
@pytest.mark.parametrize('n_ranks', [2, 3, 4])
def test_q1_solution_matches_serial(kind, n, n_ranks):
    def checksum(ranks):
        out = _launch('cube_q1_mpi.py', [kind, n], ranks)
        line = [l for l in out.splitlines() if l.startswith('RESULT')][0]
        return float(line.split('checksum=')[1].split()[0])
    serial = checksum(1)
    parallel = checksum(n_ranks)
    assert abs(parallel - serial) < 1.0e-9 * abs(serial), (
        "%s Q1 (n=%i) on %i ranks gave %.12e against %.12e in serial"
        % (kind, n, n_ranks, parallel, serial))
