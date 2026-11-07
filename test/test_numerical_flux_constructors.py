"""Every numerical flux in proteus/NumericalFlux.py can be built the two ways OneLevelTransport builds one.

Transport.py (and the mprans models) call ``numericalFluxType(vt, getPointwiseBoundaryConditions,
getAdvectiveFluxBoundaryConditions, getDiffusiveFluxBoundaryConditions)``, and append
``options.periodicDirichletConditions`` when the p module sets periodic conditions. Until 2026-11-05
RusanovNumericalFlux_Diagonal and ConvexOneSonicPointNumericalFlux required the periodic argument, so the ordinary
call raised TypeError, and RusanovNumericalFlux called RusanovNumericalFlux_Diagonal.__init__ without it, so every
deck using RusanovNumericalFlux (test/shallow_water/dambreak_1d) failed in its constructor (since b8912a02, 2014).
"""
import ast
import inspect

from proteus import NumericalFlux

_FLUXES = {name: cls for name, cls in vars(NumericalFlux).items()
           if inspect.isclass(cls) and cls.__module__ == NumericalFlux.__name__ and issubclass(cls, NumericalFlux.NF_base)}

# These take no periodic argument, so a periodic deck using them stops with TypeError in the constructor. That is the
# loud refusal, kept: most of them have no periodic face pairing in their exterior fluxes, so accepting the argument
# would only hide it. A class leaving this set must also pair periodic faces (NF_base.periodicBoundaryConditionsDictList).
NO_PERIODIC_ARGUMENT = {
    "Advection_DiagonalUpwind_Diffusion_IIPG", "Advection_DiagonalUpwind_Diffusion_LDG",
    "Advection_DiagonalUpwind_Diffusion_NIPG", "Advection_DiagonalUpwind_Diffusion_NIPG_exterior",
    "Advection_DiagonalUpwind_Diffusion_SIPG", "Advection_DiagonalUpwind_Diffusion_SIPG_exterior",
    "Advection_Diagonal_average", "ConstantAdvection_Diffusion_IIPG_exterior",
    "ConstantAdvection_Diffusion_NIPG_exterior", "ConstantAdvection_Diffusion_SIPG_exterior",
    "ConstantAdvection_exterior", "Curvature_exterior", "DarcyFCFF_IIPG_exterior", "DarcyFCPP_IIPG_exterior",
    "DarcyFC_IIPG_exterior", "DarcySplitPressure_IIPG_exterior", "Diffusion_LDG", "MixedDarcy_exterior",
    "Richards_IIPG_exterior", "Richards_SIPG_exterior", "RusanovLDG", "RusanovNumericalFlux_Diagonal_Diffusion_IIPG",
    "StokesP_Advection_DiagonalUpwind_Diffusion_IIPG_exterior",
}


def _binds(cls, nargs):
    try:
        inspect.signature(cls.__init__).bind(*range(1 + nargs))  # self + arguments
        return True
    except TypeError:
        return False


def test_every_flux_takes_the_ordinary_call():
    refused = sorted(name for name, cls in _FLUXES.items() if not _binds(cls, 4))
    assert refused == [], refused


def test_the_fluxes_that_take_no_periodic_argument_are_the_known_set():
    refused = {name for name, cls in _FLUXES.items() if not _binds(cls, 5)}
    assert refused == NO_PERIODIC_ARGUMENT, (sorted(refused - NO_PERIODIC_ARGUMENT),
                                             sorted(NO_PERIODIC_ARGUMENT - refused))


def _base_init_calls():
    """(class, line, base, parameters of the calling __init__, call) for every Base.__init__(self, ...) in a flux's
    __init__."""
    tree = ast.parse(inspect.getsource(NumericalFlux))
    for cdef in (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in _FLUXES):
        for fdef in cdef.body:
            if not (isinstance(fdef, ast.FunctionDef) and fdef.name == "__init__"):
                continue
            params = [a.arg for a in fdef.args.args]
            for call in ast.walk(fdef):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "__init__" and isinstance(call.func.value, ast.Name)
                        and call.func.value.id in _FLUXES):
                    yield cdef.name, call.lineno, _FLUXES[call.func.value.id], params, call


def test_every_base_constructor_call_binds_and_forwards_the_periodic_conditions():
    wrong = []
    for name, line, base, params, call in _base_init_calls():
        names = [arg.id if isinstance(arg, ast.Name) else ast.unparse(arg) for arg in call.args]
        try:
            inspect.signature(base.__init__).bind(*names, **{kw.arg: kw.arg for kw in call.keywords})
        except TypeError as e:
            wrong.append("%s:%d %s.__init__: %s" % (name, line, base.__name__, e))
            continue
        periodic = [p for p in params if p.startswith("getPeriodic")]
        if periodic and periodic[0] not in names + [kw.arg for kw in call.keywords]:
            wrong.append("%s:%d drops %s" % (name, line, periodic[0]))
    assert wrong == [], "\n".join(wrong)


# Seven fluxes accept the periodic argument and pair no periodic faces. Measured on a C0 P1 periodic Poisson problem
# (NumericalFlux.PERIODIC_SEAM_REFUSED says how): the four diffusion ones are wrong with periodic conditions and refuse
# them; the other three are not changed by them and keep accepting them:
#   StrongDirichlet: 4.6e-2 without the periodic pair, 4.7e-2 with it, 1 Newton iteration (ADRProblem's periodic
#     problems use it through StrongDirichletFactory: test_adr_problem's periodic Poisson, channel, several ranks);
#   DoNothing: no boundary term at all, the right one for a model without boundary conditions on a periodic C0 space
#     (the MCorr and RDLS models of test/periodic and test/hotstart_measure use it);
#   NoFlux: the same answer with and without the periodic pair (it imposes no Dirichlet condition either way).
PERIODIC_ACCEPTED_WITHOUT_PAIRING = {"StrongDirichlet", "DoNothing", "NoFlux"}


def _periodic():
    return {0: lambda x, flag: None}


def test_the_diffusion_exterior_fluxes_refuse_periodic_conditions():
    import pytest
    for name in sorted(NumericalFlux.PERIODIC_SEAM_REFUSED):
        with pytest.raises(ValueError, match="periodic"):
            _FLUXES[name](None, {}, {}, {}, _periodic())


def test_the_other_unpaired_fluxes_still_take_periodic_conditions():
    # they get past the refusal and fail later on the missing transport (None), or not at all (DoNothing)
    for name in sorted(PERIODIC_ACCEPTED_WITHOUT_PAIRING):
        try:
            _FLUXES[name](None, {}, {}, {}, _periodic())
        except ValueError as e:
            raise AssertionError("%s refused: %s" % (name, e))
        except (AttributeError, TypeError):
            pass
