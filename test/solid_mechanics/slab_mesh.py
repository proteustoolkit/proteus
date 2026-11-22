"""Structured one-element-thick tetrahedral slab meshes in tetgen format, for quasi-2D (plane strain) solid mechanics tests.

The meshes are generated from formulas (no mesh generator), so refinement is nested and reproducible.
"""
import numpy as np

# boundary flags
HOLE, RIGHT, TOP, BOTTOM, LEFT, BACK, FRONT = 1, 2, 3, 4, 5, 6, 7
boundaryFlags = {'hole': HOLE, 'right': RIGHT, 'top': TOP, 'bottom': BOTTOM, 'left': LEFT, 'back': BACK, 'front': FRONT}

# Kuhn split of a hexahedron in 6 tetrahedra around the diagonal 0-6; face diagonals match between neighbours
_KUHN = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def _write(fileprefix, nodes, tets, faces, faceFlags):
    nodeFlags = np.zeros(len(nodes), 'i')
    for f, flag in zip(faces, faceFlags):
        nodeFlags[f] = flag
    with open(fileprefix + '.node', 'w') as fh:
        fh.write('%d 3 0 1\n' % len(nodes))
        for i, (x, flag) in enumerate(zip(nodes, nodeFlags)):
            fh.write('%d %.17g %.17g %.17g %d\n' % (i + 1, x[0], x[1], x[2], flag))
    with open(fileprefix + '.ele', 'w') as fh:
        fh.write('%d 4 1\n' % len(tets))
        for i, t in enumerate(tets):
            fh.write('%d %d %d %d %d 0\n' % (i + 1, t[0] + 1, t[1] + 1, t[2] + 1, t[3] + 1))
    with open(fileprefix + '.face', 'w') as fh:
        fh.write('%d 1\n' % len(faces))
        for i, (f, flag) in enumerate(zip(faces, faceFlags)):
            fh.write('%d %d %d %d %d\n' % (i + 1, f[0] + 1, f[1] + 1, f[2] + 1, flag))


def _slab(xy, flag_i0, flag_i1, flag_j0, flag_j1, thickness):
    """xy: (ni+1, nj+1, 2) node coordinates of a logically rectangular 2D grid; flag_*: face flag (or callable of the face
    centroid) on the four sides. Returns nodes, tets, boundary faces and their flags for a one-element-thick slab."""
    ni, nj = xy.shape[0] - 1, xy.shape[1] - 1
    nid = lambda i, j, k: (i*(nj + 1) + j)*2 + k
    nodes = np.zeros(((ni + 1)*(nj + 1)*2, 3))
    for i in range(ni + 1):
        for j in range(nj + 1):
            for k in range(2):
                nodes[nid(i, j, k)] = (xy[i, j, 0], xy[i, j, 1], k*thickness)
    tets, faces, faceFlags = [], [], []
    sides = {(0, 3, 7, 4): ('i', 0), (1, 2, 6, 5): ('i', 1), (0, 1, 5, 4): ('j', 0), (3, 2, 6, 7): ('j', 1),
             (0, 1, 2, 3): ('k', 0), (4, 5, 6, 7): ('k', 1)}
    for i in range(ni):
        for j in range(nj):
            h = [nid(i, j, 0), nid(i + 1, j, 0), nid(i + 1, j + 1, 0), nid(i, j + 1, 0),
                 nid(i, j, 1), nid(i + 1, j, 1), nid(i + 1, j + 1, 1), nid(i, j + 1, 1)]
            for t in _KUHN:
                tet = [h[t[0]], h[t[1]], h[t[2]], h[t[3]]]
                p = nodes[tet]
                if np.linalg.det(p[1:] - p[0]) < 0.0:
                    tet[2], tet[3] = tet[3], tet[2]
                tets.append(tet)
                for tri in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                    loc = set(t[m] for m in tri)
                    for quad, (axis, side) in sides.items():
                        if loc <= set(quad):
                            onBoundary = ((axis == 'i' and ((side == 0 and i == 0) or (side == 1 and i == ni - 1))) or
                                          (axis == 'j' and ((side == 0 and j == 0) or (side == 1 and j == nj - 1))) or
                                          axis == 'k')
                            if onBoundary:
                                f = [h[t[m]] for m in tri]
                                flag = {('i', 0): flag_i0, ('i', 1): flag_i1, ('j', 0): flag_j0, ('j', 1): flag_j1,
                                        ('k', 0): BACK, ('k', 1): FRONT}[(axis, side)]
                                if callable(flag):
                                    flag = flag(nodes[f].mean(axis=0))
                                faces.append(f)
                                faceFlags.append(flag)
    return nodes, np.array(tets), np.array(faces), np.array(faceFlags)


def quarter_plate_with_hole(fileprefix, n, a=1.0, L=4.0, grading=2.0):
    """Quarter plate [0,L]^2 minus a hole of radius a at the origin: n radial by 2n angular cells, graded toward the hole.

    Returns the slab thickness."""
    nr, nt = n, 2*n
    xy = np.zeros((nr + 1, nt + 1, 2))
    for j in range(nt + 1):
        th = 0.5*np.pi*j/nt
        c, s = np.cos(th), np.sin(th)
        R = L/max(c, s)
        for i in range(nr + 1):
            xi = float(i)/nr
            r = a + (R - a)*xi**grading
            xy[i, j] = (r*c, r*s)
    xy[:, 0, 1] = 0.0
    xy[:, nt, 0] = 0.0
    thickness = 0.5*np.pi*a/nt
    outer = lambda x: RIGHT if x[0] > x[1] else TOP
    nodes, tets, faces, flags = _slab(xy, HOLE, outer, BOTTOM, LEFT, thickness)
    _write(fileprefix, nodes, tets, faces, flags)
    return thickness


def rectangle(fileprefix, nx, ny, Lx=1.0, Ly=1.0):
    """Rectangle [0,Lx] x [0,Ly] with nx by ny cells. Returns the slab thickness."""
    xy = np.zeros((nx + 1, ny + 1, 2))
    xy[:, :, 0] = np.linspace(0.0, Lx, nx + 1)[:, None]
    xy[:, :, 1] = np.linspace(0.0, Ly, ny + 1)[None, :]
    thickness = Lx/nx
    nodes, tets, faces, flags = _slab(xy, LEFT, RIGHT, BOTTOM, TOP, thickness)
    _write(fileprefix, nodes, tets, faces, flags)
    return thickness
