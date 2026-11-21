"""Verification of the constitutive model selection and of load stepping in proteus.elastoplastic: confined compression
of a block past yield, unloading and reverse yield, against a radial-return reference."""
import os
import numpy as np
import pytest
from proteus.elastoplastic import ElastoPlastic
from . import confined_compression as cc
from . import solid_deck

M = cc.E*(1.0 - cc.NU)/((1.0 + cc.NU)*(1.0 - 2.0*cc.NU))
EPS_Y = cc.SY*(1.0 - cc.NU)/(1.0 - 2.0*cc.NU)/M  # uniaxial strain at first yield

# strain paths in units of the yield strain (compression positive): loading, two load steps, unloading inside the
# elastic range, and unloading into reverse yield
PATHS = ([0.5], [1.5], [1.5, 3.0], [3.0, 1.0], [3.0, -0.5])


def _run(path, model, opts):
    eps_path = -EPS_Y*np.array(path)
    ns, m = cc.solve('cc_mesh', eps_path, opts, model=model)
    eps, epp, sig = cc.state(m)
    del ns
    for name in ('cc_mesh', 'confined_compression'):
        for ext in ('h5', 'xmf', 'ymf', 'node', 'ele', 'face'):
            if os.path.exists(name + '.' + ext):
                os.remove(name + '.' + ext)
    sig_exact, epp_exact = cc.exact_von_mises(eps_path)
    return (np.abs(sig.reshape(-1, 3) - sig_exact).max(), np.abs(epp[..., :3].reshape(-1, 3) - epp_exact).max(),
            np.abs(epp[..., 3:]).max())


@pytest.mark.parametrize('path', PATHS)
def test_von_mises(opts, path):
    """Measured 2026-11-14: stress error <= 4.3e-12 MPa (stresses up to 1567 MPa), plastic strain error <= 2.8e-17."""
    err_sig, err_epp, shear = _run(path, solid_deck.VON_MISES, opts)
    assert err_sig < 1.0e-8
    assert err_epp < 1.0e-13
    assert shear < 1.0e-13


@pytest.mark.parametrize('path', ([1.5], [3.0, 1.0]))
def test_drucker_prager_without_friction_is_von_mises(opts, path):
    """Measured 2026-11-14: stress error 5.9e-8 and 7.0e-7 MPa, plastic strain error 3.9e-13 and 6.1e-12."""
    err_sig, err_epp, shear = _run(path, solid_deck.DRUCKER_PRAGER, opts)
    assert err_sig < 1.0e-4
    assert err_epp < 1.0e-9


def test_default_model_is_unchanged(opts):
    """Model types 0 and 1 run mcdp, as before the selection existed. Measured: stress error 1.6e-5 MPa."""
    err_sig, err_epp, shear = _run([1.5], solid_deck.MCDP, opts)
    assert err_sig < 1.0e-3
    assert err_epp < 1.0e-8


def test_unknown_model_type_is_an_error():
    types, params = solid_deck.material(cc.E, cc.NU, cohesion=1.0, model=7)
    with pytest.raises(ValueError):
        ElastoPlastic.Coefficients(modelType_block=types, modelParams_block=params, nd=3)
