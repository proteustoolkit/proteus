"""Kirsch plate with a hole in plane strain, solved with proteus.elastoplastic on a one-element-thick tetrahedral slab.

Linear elasticity is obtained from the elastoplastic model with a cohesion far above any stress and P_a = 0.
Exact displacements are imposed on the outer boundary, symmetry on x = 0 and y = 0, w = 0 on the slab faces, and the hole
is traction free.
"""
import numpy as np
try:
    from . import kirsch_exact, slab_mesh, solid_deck
except ImportError:
    import kirsch_exact, slab_mesh, solid_deck

S, A, L, E, NU = 100.0, 1.0, 4.0, 2.0e5, 0.3
bf = slab_mesh.boundaryFlags


def make_pn(fileprefix, order=1):
    p = solid_deck.physics('kirsch_p', fileprefix, bf, solid_deck.material(E, NU, cohesion=1.0e12))

    def ux(x, t):
        return kirsch_exact.displacement(x[0], x[1], S, A, E, NU)[0]

    def uy(x, t):
        return kirsch_exact.displacement(x[0], x[1], S, A, E, NU)[1]

    zero = solid_deck.zero
    solid_deck.boundary_conditions(p, {0: {bf['right']: ux, bf['top']: ux, bf['left']: zero},
                                       1: {bf['right']: uy, bf['top']: uy, bf['bottom']: zero},
                                       2: {bf['back']: zero, bf['front']: zero}})
    return p, solid_deck.numerics('kirsch_n', order)


def solve(fileprefix, nRefine, order, opts):
    """Generate the mesh, solve, and return the level model."""
    slab_mesh.quarter_plate_with_hole(fileprefix, nRefine, a=A, L=L)
    p, n = make_pn(fileprefix, order)
    return solid_deck.run(p, n, 'kirsch_n%d_p%d' % (nRefine, order), opts)


def errors(model):
    """Displacement and stress errors of the solution against the exact one, from the vertex values (P1 interpolant).

    Returns h, relative L2 displacement error, relative energy-type (stress L2) error, max nodal displacement error."""
    mesh = model.mesh
    X = mesh.nodeArray
    nN = X.shape[0]
    u = np.array([model.u[0].dof[:nN], model.u[1].dof[:nN], model.u[2].dof[:nN]]).T
    ue = np.zeros_like(u)
    ue[:, 0], ue[:, 1] = kirsch_exact.displacement(X[:, 0], X[:, 1], S, A, E, NU)
    tets = mesh.elementNodesArray
    P = X[tets]                                  # (nE, 4, 3)
    J = P[:, 1:, :] - P[:, :1, :]                # rows: edge vectors
    vol = np.abs(np.linalg.det(J))/6.0
    Jinv = np.linalg.inv(J)
    U = u[tets]
    dU = U[:, 1:, :] - U[:, :1, :]               # (nE, 3 edges, 3 comps)
    grad = np.einsum('eij,ejk->eki', Jinv, dU)   # grad[e, comp, dir]
    eps = 0.5*(grad + np.transpose(grad, (0, 2, 1)))
    lam = E*NU/((1 + NU)*(1 - 2*NU))
    G = E/(2*(1 + NU))
    tr = np.trace(eps, axis1=1, axis2=2)
    sig = 2*G*eps + lam*tr[:, None, None]*np.eye(3)[None]
    xc = P.mean(axis=1)
    # 4-point degree-2 quadrature for the exact stress and displacement
    aq, bq = 0.5854101966249685, 0.1381966011250105
    W = np.full((4, 4), bq) + (aq - bq)*np.eye(4)
    errS = np.zeros(len(tets))
    normS = np.zeros(len(tets))
    errU = np.zeros(len(tets))
    normU = np.zeros(len(tets))
    for q in range(4):
        xq = np.einsum('n,enk->ek', W[q], P)
        uq = np.einsum('n,enk->ek', W[q], U)
        sxx, syy, szz, sxy = kirsch_exact.stress(xq[:, 0], xq[:, 1], S, A, NU)
        errS += 0.25*vol*((sig[:, 0, 0] - sxx)**2 + (sig[:, 1, 1] - syy)**2 + (sig[:, 2, 2] - szz)**2
                          + 2*(sig[:, 0, 1] - sxy)**2 + 2*sig[:, 0, 2]**2 + 2*sig[:, 1, 2]**2)
        normS += 0.25*vol*(sxx**2 + syy**2 + szz**2 + 2*sxy**2)
        uex, uey = kirsch_exact.displacement(xq[:, 0], xq[:, 1], S, A, E, NU)
        errU += 0.25*vol*((uq[:, 0] - uex)**2 + (uq[:, 1] - uey)**2 + uq[:, 2]**2)
        normU += 0.25*vol*(uex**2 + uey**2)
    h = (vol.sum()/len(tets))**(1.0/3.0)
    return (h, np.sqrt(errU.sum()/normU.sum()), np.sqrt(errS.sum()/normS.sum()), np.abs(u - ue).max(),
            np.abs(sig[:, 0, 0]).max())
