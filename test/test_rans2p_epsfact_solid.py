"""RANS2P.Coefficients(epsFact_solid=...) changed meaning between 1.x and 2.0 without a word.

In 1.x epsFact_solid was the smoothing width of the porous and relaxation zones; in 2.0 that is epsFact_porous and the
RANS2P kernels never read epsFact_solid. An old _p file that still passes epsFact_solid=... ran, its zones got the default
smoothing (ones), and the width it asked for went nowhere, with no message (air-water-vv 3d/linear_waves_flat_3D, 3d/Penny_Price_Diffraction,
3d/Directional_Wave_Current_interaction, sharp_crested_weir_AV_V1). It now warns, with the keyword to use.
"""
import warnings

import numpy as np
import pytest

from proteus.kernels import RANS2P, RANS2P_IB


def test_epsFact_solid_warns_in_rans2p():
    with pytest.warns(UserWarning, match="epsFact_porous"):
        RANS2P.Coefficients(nd=2, epsFact_solid=np.array([0., 5.]))


def test_epsFact_porous_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RANS2P.Coefficients(nd=2, epsFact_porous=np.array([0., 5.]))
        RANS2P.Coefficients(nd=2)


def test_immersed_boundary_variant_uses_epsFact_solid_and_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RANS2P_IB.Coefficients(nd=2, epsFact_solid=np.array([0., 5.]))
