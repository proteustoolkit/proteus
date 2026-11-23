"""Flux classes nothing uses refuse to be built on a partitioned mesh."""
import pytest

from proteus import Comm, NumericalFlux


class _Comm:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _build(cls, monkeypatch, ranks):
    monkeypatch.setattr(Comm, "get", lambda: _Comm(ranks))
    obj = cls.__new__(cls)
    # the refusal comes before anything touches the model
    NumericalFlux.NF_base.__init__(obj, None, None, None, None)


@pytest.mark.parametrize("name", sorted(NumericalFlux.UNUSED_FLUX_CLASSES_REFUSED_IN_PARALLEL))
def test_unused_class_is_refused_on_two_ranks(name, monkeypatch):
    cls = getattr(NumericalFlux, name)
    with pytest.raises(NotImplementedError, match=name):
        _build(cls, monkeypatch, 2)


def test_unused_class_is_not_refused_on_one_rank(monkeypatch):
    cls = NumericalFlux.Curvature_exterior
    with pytest.raises(AttributeError):         # got past the guard, then needed a model
        _build(cls, monkeypatch, 1)


def test_a_class_in_use_is_not_refused(monkeypatch):
    with pytest.raises(AttributeError):
        _build(NumericalFlux.StrongDirichlet, monkeypatch, 2)


def test_the_mprans_ncls_flux_subclass_is_not_refused(monkeypatch):
    """It subclasses HamiltonJacobi_DiagonalLesaintRaviart, which is refused, but by
    exact name: NCLS computes its exterior flux in its own kernel and runs in parallel."""
    from proteus.kernels import NCLS
    assert issubclass(NCLS.NumericalFlux, NumericalFlux.HamiltonJacobi_DiagonalLesaintRaviart)
    assert "NumericalFlux" not in NumericalFlux.UNUSED_FLUX_CLASSES_REFUSED_IN_PARALLEL
    monkeypatch.setattr(Comm, "get", lambda: _Comm(2))
    obj = NCLS.NumericalFlux.__new__(NCLS.NumericalFlux)
    with pytest.raises(AttributeError):         # past the guard, then needed a model
        NumericalFlux.NF_base.__init__(obj, None, None, None, None)
