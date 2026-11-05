"""One scatterDOFArray, on OneLevelTransport, for every model that refreshes the ghost entries of a dof array (there
were 11 copies in the mprans model files). The parallel behaviour is covered by the partition tests of the models
that call it (NCLS, NCLS3P, RDLS, VOF, VOF3P, TADR, MCorr, VOS3P, RANS3PF, SW2DCV, GN_SW2DCV)."""
import importlib

import pytest

from proteus.Transport import OneLevelTransport

MODELS = ["GN_SW2DCV", "MCorr", "NCLS", "NCLS3P", "RANS3PF", "RDLS", "SW2DCV", "TADR", "VOF", "VOF3P", "VOS3P"]


@pytest.mark.parametrize("name", MODELS)
def test_models_use_the_shared_scatter(name):
    model = importlib.import_module("proteus.mprans." + name).LevelModel
    assert issubclass(model, OneLevelTransport)
    assert model.scatterDOFArray is OneLevelTransport.scatterDOFArray


def test_serial_scatter_leaves_the_array_alone():
    import numpy

    class Serial(object):
        class comm(object):
            @staticmethod
            def size():
                return 1

    a = numpy.arange(5.0)
    OneLevelTransport.scatterDOFArray(Serial(), a)
    assert (a == numpy.arange(5.0)).all()
