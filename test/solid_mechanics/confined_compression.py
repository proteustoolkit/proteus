"""Confined (uniaxial strain) compression of a block past yield: a homogeneous elastic-perfectly-plastic state.

Rollers on the sides and the slab faces, fixed base in y, and the top pushed down by delta. The strain is
eps_yy = -delta/Ly everywhere, so the stress and the plastic strain follow from the constitutive model alone and the
mesh does not matter. One load step is exact for von Mises here because the deviatoric strain direction is fixed.
"""
import numpy as np
try:
    from . import slab_mesh, solid_deck
except ImportError:
    import slab_mesh, solid_deck

E, NU, SY = 2.0e5, 0.3, 400.0
LY = 1.0
bf = slab_mesh.boundaryFlags


def exact_von_mises(eps_yy_path):
    """Stress (xx, yy, zz) and plastic strain (xx, yy, zz) after following a path of uniaxial strains eps_yy from a
    virgin state: von Mises perfect plasticity with yield stress SY, radial return at every point of the path."""
    G = E/(2.0*(1.0 + NU))
    K = E/(3.0*(1.0 - 2.0*NU))
    epp = np.zeros(3)
    for eps_yy in np.atleast_1d(eps_yy_path):
        ee = np.array([0.0, eps_yy, 0.0]) - epp
        tr = ee.sum()
        dev = 2.0*G*(ee - tr/3.0)
        q = np.sqrt(1.5*np.dot(dev, dev))
        if q > SY:
            epp = epp + (dev - dev*SY/q)/(2.0*G)
            dev = dev*SY/q
        sig = K*tr + dev
    return sig, epp


def solve(fileprefix, eps_yy_path, opts, model=solid_deck.VON_MISES, nx=2, ny=2):
    """Apply the uniaxial strains eps_yy_path[k] at times t = k + 1 (piecewise linear in between)."""
    path = np.concatenate(([0.0], np.atleast_1d(eps_yy_path)))
    times = np.arange(len(path), dtype='d')
    slab_mesh.rectangle(fileprefix, nx, ny, 1.0, LY)
    p = solid_deck.physics('cc_p', fileprefix, bf, solid_deck.material(E, NU, cohesion=0.5*SY, model=model),
                           loadStepping=True)
    zero = solid_deck.zero
    solid_deck.boundary_conditions(p, {0: {bf['left']: zero, bf['right']: zero},
                                       1: {bf['bottom']: zero, bf['top']: lambda x, t: LY*np.interp(t, times, path)},
                                       2: {bf['back']: zero, bf['front']: zero}})
    n = solid_deck.numerics('cc_n', 1, tnList=times)
    return solid_deck.run(p, n, 'confined_compression', opts)


def state(model):
    """Element-quadrature strain and plastic strain (xx, yy, zz and the largest shear magnitude) and the stress from them."""
    eps = model.q['strain']
    epp = model.q['plasticStrain']
    ee = eps[..., :3] - epp[..., :3]
    G = E/(2.0*(1.0 + NU))
    lam = E*NU/((1.0 + NU)*(1.0 - 2.0*NU))
    sig = 2.0*G*ee + lam*ee.sum(axis=-1)[..., None]
    return eps, epp, sig
