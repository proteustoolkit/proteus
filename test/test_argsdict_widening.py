"""ArgumentsDict hands a kernel that asks for a double any number it was given.

A Python int is a valid value for a double argument (a deck may say
``useRBLES = 1`` or ``shockCapturingFactor = 0``). Until C0 round 12 the int was
stored only in the int dicts, so a double lookup raised a key error and every
call site wrapped the value in ``float(...)``. The int is now also stored,
widened, in the double dict; the integer dicts are unchanged.

Scalars are classified by kind, not by pybind11's implicit conversions: a numpy
float32 is a double, never an int (the int64 overload used to truncate it).
"""
import numpy as np
import pytest

from proteus import cArgumentsDict


def make(**kw):
    ad = cArgumentsDict.ArgumentsDict()
    for k, v in kw.items():
        ad[k] = v
    return ad


def test_a_python_int_is_also_a_double():
    ad = make(n=3, big=2 ** 40, neg=-7)
    assert ad.dscalar["n"] == 3.0 and isinstance(ad.dscalar["n"], float)
    assert ad.dscalar["neg"] == -7.0
    assert ad.dscalar["big"] == float(2 ** 40)
    # the integer dicts are what they were
    assert ad.iscalar["n"] == 3 and ad.lscalar["n"] == 3
    assert ad.lscalar["big"] == 2 ** 40
    with pytest.raises(RuntimeError):
        ad.iscalar["big"]


@pytest.mark.parametrize("value", [True, np.int32(5), np.int64(5), np.uint8(5), np.bool_(True)])
def test_integer_kinds_land_in_the_integer_dicts(value):
    ad = make(v=value)
    assert ad.iscalar["v"] == int(value) and ad.lscalar["v"] == int(value)
    assert ad.dscalar["v"] == float(int(value))


@pytest.mark.parametrize("value", [1.5, np.float64(1.5), np.float32(1.5), np.float16(1.5)])
def test_floating_kinds_are_doubles_not_truncated(value):
    ad = make(v=value)
    assert ad.dscalar["v"] == 1.5
    for d in (ad.iscalar, ad.lscalar):
        with pytest.raises(RuntimeError):
            d["v"]


def test_the_last_assignment_wins_in_the_double_dict():
    ad = make(v=2)
    assert ad.dscalar["v"] == 2.0
    ad["v"] = 0.25
    assert ad.dscalar["v"] == 0.25
    ad["v"] = 9
    assert ad.dscalar["v"] == 9.0


def test_arrays_still_go_to_their_own_dicts_and_not_to_scalars():
    ad = make(one=np.array([4.0]), ione=np.array([4], dtype=np.int32))
    assert list(ad.darray["one"]) == [4.0] and list(ad.iarray["ione"]) == [4]
    for name in ("one", "ione"):
        with pytest.raises(RuntimeError):
            ad.dscalar[name]


@pytest.mark.parametrize("bad", [(1.0,), [1.0], None, "x", {}])
def test_a_non_number_is_a_type_error(bad):
    ad = cArgumentsDict.ArgumentsDict()
    with pytest.raises(TypeError):
        ad["k"] = bad
