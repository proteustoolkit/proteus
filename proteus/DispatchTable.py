"""The dispatch table: (formulation x discretization options) -> (Proteus module, flags).

Data only, so that it can be reviewed on its own; :mod:`proteus.Dispatch` reads it. A ymf spec states the physics
(the model), the formulation (a ``weak_forms`` entry, named by a discretization's ``from_weak_form``) and the
discretization (typed keys: spaces, time integration, stabilization, the split into sub-problems, bound symbols,
regularization). One mprans module covers several cells of this table (RANS2P serves both inertia forms through
``NONCONSERVATIVE_FORM``), and a cell no module covers is refused by name (YMF_MODEL_FAMILIES_C42 section 6, D1).
A generated kernel (C19) will be one more cell.

How a sub-problem finds its cell:

1. **Family**: the model's ``physical_model`` (``ymf.vocabulary.family_of``).
2. **Role**: what the sub-problem does, read from the closed spec's structure: which unknowns its equations are for,
   what it enforces, what it introduces (``ymf.vocabulary.role_of``).

The words (families, roles, time integrations, stabilizations, scheme options) are ymf's (``ymf.vocabulary``, C42
round 3): the cells below are keyed on them, and :mod:`proteus.Dispatch` reads a spec's words through
``ymf.vocabulary.canonical``, so a spec that still says ``vbdf`` or ``SSP_RK3`` dispatches as it did.
3. **Cell**: the cells of that family and role, matched on the formulation and on each typed option the sub-problem
   states. An option the spec leaves out takes the cell's ``default`` and is recorded as defaulted.

Each cell gives the module, the attributes of the physics module (``physics``: the coefficients class and its
flags, the level model) and of the numerics module (``numerics``), each with where its value comes from:

* ``fixed``: the cell itself (what makes this cell this cell);
* ``default``: TwoPhaseFlow's or SWFlow's default, used when neither the spec nor a hint says otherwise;
* rules (``from_*`` keys): the dispatcher computes the value from a typed key of the spec.

``hints`` lists the flags an interim ``proteus:`` block may set (D8). A hint on a flag the dispatcher fixed or
computed from the spec is checked and refused on disagreement; a hint on a defaulted flag replaces the default (it
is an option ymf does not type yet); a hint the cell does not list is refused, so a misspelled option cannot be
dropped silently.

Dotted names are resolved by the dispatcher when asked (``Dispatch.resolve``); ``{module}`` is the cell's module.
"""

#: (family, order) of a ymf finite element -> the Proteus basis
SPACES = {
    ("CG", 1): "proteus.FemTools.C0_AffineLinearOnSimplexWithNodalBasis",
    ("CG", 2): "proteus.FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis",
}

#: element quadrature order from the highest polynomial order of the discretization's spaces (TwoPhaseFlow's
#: FESpace.setFESpace and SWFlow's FESpace); cut-cell integration (``useExact``) uses 6 whatever the order
QUADRATURE_ORDER = {1: 3, 2: 5}
QUADRATURE_ORDER_EXACT = 6

#: element size scaling of the stabilization by velocity order (FESpace.hFactor)
H_FACTOR = {1: 1.0, 2: 0.5}

#: time integration by name, per kind of cell. ``none`` is a sub-problem with no time derivative.
_IMPLICIT_TIME = {
    "backward_euler": {"timeIntegration": "proteus.TimeIntegration.BackwardEuler_cfl", "timeOrder": 1},
    "bdf2": {"timeIntegration": "proteus.TimeIntegration.VBDF", "timeOrder": 2},
}
_NO_TIME = {"none": {"timeIntegration": "proteus.TimeIntegration.NoIntegration"}}
_SSP_RK3 = {"ssp_rk3": {"timeIntegration": "{module}.RKEV", "timeOrder": 3, "nStagesTime": 3}}

#: the inertia forms of the momentum equation the table knows, by weak-form label (D1: a formulation is a
#: weak_forms entry). Both are formulations of the same strong form when div v = 0.
FORMULATIONS = {
    "conservative_inertia": "d(rho v)/dt + div(rho v (x) v)",
    "nonconservative_inertia": "rho (dv/dt + v . grad v)",
}

#: a sub-problem whose equation is the model's own, or introduced by the discretization: no formulation choice
AS_WRITTEN = "as_written"

_RANS2P_COEFFICIENTS = {
    "default": {"useMetrics": 1.0, "eb_penalty_constant": 100.0, "particle_epsFact": 3.0, "useExact": False,
                "MOMENTUM_SGE": 1.0, "PRESSURE_SGE": 1.0, "VELOCITY_SGE": 1.0,
                "PRESSURE_PROJECTION_STABILIZATION": 0.0},
    # epsFact: the width of the regularized heaviside, in element sizes; useVF: rho and mu from the VOF carrier
    # (1) or from heaviside_eps(level set) (0), read from what the sub-problem lags
    "from_heaviside_width": ["epsFact"],
    "from_carrier_lag": "useVF",
}


def _rans2p(cell, form, flag):
    return {
        "cell": cell,
        "family": "two_phase_level_set",
        "role": "velocity_pressure",
        "module": "proteus.mprans.RANS2P",
        "name": "rans2p",
        "formulation": form,
        "spaces": {"velocity": {"options": [("CG", 1)], "default": ("CG", 1)},
                   "pressure": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["pressure", "velocity"],
        "time_integration": {"options": _IMPLICIT_TIME, "default": "backward_euler"},
        "stabilization": {"options": ["ASGS"], "default": "ASGS",
                          "subgrid_lag": True, "shock_capturing": {"factor": 0.5, "lag": True}},
        "physics": {
            "coefficients": "{module}.Coefficients",
            "LevelModelType": "{module}.LevelModel",
            "fixed": {"NONCONSERVATIVE_FORM": flag},
            **_RANS2P_COEFFICIENTS,
        },
        "numerics": {
            "fixed": {"numericalFluxType": "{module}.NumericalFlux",
                      "subgridError": "{module}.SubgridError",
                      "shockCapturing": "{module}.ShockCapturing",
                      "stepController": "proteus.StepControl.Min_dt_cfl_controller",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton"},
            "default": {"conservativeFlux": {0: "pwl-bdm-opt"}, "maxNonlinearIts": 100},
        },
        "hints": ["NONCONSERVATIVE_FORM", "useVF", "epsFact", "useMetrics", "eb_penalty_constant",
                  "particle_epsFact", "useExact", "MOMENTUM_SGE", "PRESSURE_SGE", "VELOCITY_SGE",
                  "PRESSURE_PROJECTION_STABILIZATION", "conservativeFlux", "maxNonlinearIts"],
    }


def _scalar_transport(cell, role, module, name, coefficients, stabilization, extra_hints=()):
    """VOF and NCLS: P1, backward Euler, SUPG with lagged shock capturing, the same numerics."""
    return {
        "cell": cell,
        "family": "two_phase_level_set",
        "role": role,
        "module": module,
        "name": name,
        "formulation": AS_WRITTEN,
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields"],
        "time_integration": {"options": {"backward_euler": _IMPLICIT_TIME["backward_euler"]},
                             "default": "backward_euler"},
        "stabilization": stabilization,
        "physics": {
            "coefficients": "{module}.Coefficients",
            "LevelModelType": "{module}.LevelModel",
            "default": coefficients,
            "from_heaviside_width": ["epsFact"],
            # shock-capturing parameters the coefficients hold (sc_uref, sc_beta)
            "from_shock_capturing": ["sc_uref", "sc_beta"],
        },
        "numerics": {
            "fixed": {"numericalFluxType": "{module}.NumericalFlux",
                      "subgridError": "{module}.SubgridError",
                      "shockCapturing": "{module}.ShockCapturing",
                      "stepController": "proteus.StepControl.Min_dt_cfl_controller",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton"},
            "default": {"maxNonlinearIts": 50},
        },
        "hints": ["epsFact", "useMetrics", "sc_uref", "sc_beta", "maxNonlinearIts"] + list(extra_hints),
    }


_SUPG = {"options": ["SUPG"], "default": "SUPG",
         "shock_capturing": {"factor": 0.5, "lag": True, "sc_uref": 1.0, "sc_beta": 1.5}}

CELLS = [
    # ---- two-phase level-set flow: the flow solve -------------------------------------------------------------
    # RANS2P, equal-order P1/P1 with ASGS (MOMENTUM/PRESSURE/VELOCITY_SGE): both inertia forms, one module.
    # TwoPhaseFlow's default (and damBreakLevelSet's) is the non-conservative form: RANS2P.Coefficients'
    # NONCONSERVATIVE_FORM defaults to 1.0.
    _rans2p("rans2p_nonconservative", "nonconservative_inertia", 1.0),
    _rans2p("rans2p_conservative", "conservative_inertia", 0.0),
    # RANS3PF: the velocity step of a projection scheme, P2 velocity, followed by PresInc, Pres (P1) and a
    # PresInit spin-up. Its kernel hardwires the non-conservative form (RANS3PF.h: "mom_u_acc=u; trick for
    # non-conservative form"), so (conservative_inertia, projection) is a cell no module covers.
    {
        "cell": "rans3pf_projection",
        "family": "two_phase_level_set",
        "role": "velocity",
        "module": "proteus.mprans.RANS3PF",
        "name": "rans3p",
        "formulation": "nonconservative_inertia",
        "spaces": {"velocity": {"options": [("CG", 2)], "default": ("CG", 2)}},
        "components": ["velocity"],
        "time_integration": {"options": _IMPLICIT_TIME, "default": "bdf2"},
        # ARTIFICIAL_VISCOSITY: 0 none, 1 residual shock capturing, 2 entropy viscosity, 3 graph viscosity with a
        # Hessian smoothness indicator (TwoPhaseFlow's), 4 graph viscosity with an entropy residual (ymf.vocabulary
        # STABILIZATION)
        "stabilization": {"options": ["graph_viscosity_smoothness", "graph_viscosity_entropy_residual",
                                      "entropy_viscosity", "residual_shock_capturing", "SUPG"],
                          "default": "graph_viscosity_smoothness",
                          "flags": {"graph_viscosity_smoothness": {"ARTIFICIAL_VISCOSITY": 3, "USE_SUPG": 0},
                                    "graph_viscosity_entropy_residual": {"ARTIFICIAL_VISCOSITY": 4, "USE_SUPG": 0},
                                    "entropy_viscosity": {"ARTIFICIAL_VISCOSITY": 2, "USE_SUPG": 0},
                                    "residual_shock_capturing": {"ARTIFICIAL_VISCOSITY": 1, "USE_SUPG": 0},
                                    "SUPG": {"ARTIFICIAL_VISCOSITY": 0, "USE_SUPG": 1}},
                          "subgrid_lag": True, "shock_capturing": {"factor": 0.5, "lag": True}},
        "requires": ["pressure_increment", "pressure", "pressure_spin_up"],
        "physics": {
            "coefficients": "{module}.Coefficients",
            "LevelModelType": "{module}.LevelModel",
            "fixed": {"INT_BY_PARTS_PRESSURE": 1},
            "default": {"useMetrics": 1.0, "particle_epsFact": 3.0, "eb_penalty_constant": 100.0},
            "from_heaviside_width": ["epsFact_density"],
            "from_carrier_lag": "useVF",
        },
        "numerics": {
            "fixed": {"numericalFluxType": "{module}.NumericalFlux",
                      "subgridError": "{module}.SubgridError",
                      "shockCapturing": "{module}.ShockCapturing",
                      "stepController": "proteus.StepControl.Min_dt_cfl_controller",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton"},
            # one Newton iteration: the velocity step is linear in the new velocity
            "default": {"maxNonlinearIts": 1},
        },
        "hints": ["useVF", "epsFact_density", "useMetrics", "particle_epsFact", "eb_penalty_constant",
                  "ARTIFICIAL_VISCOSITY", "USE_SUPG", "INT_BY_PARTS_PRESSURE", "maxNonlinearIts"],
    },
    {
        "cell": "presinc",
        "family": "two_phase_level_set",
        "role": "pressure_increment",
        "module": "proteus.mprans.PresInc",
        "name": "pressureIncrement",
        "formulation": AS_WRITTEN,
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields"],
        "time_integration": {"options": _NO_TIME, "default": "none"},
        "physics": {"coefficients": "{module}.Coefficients", "LevelModelType": "{module}.LevelModel"},
        "numerics": {
            "fixed": {"numericalFluxType": "{module}.NumericalFlux",
                      "subgridError": None, "shockCapturing": None,
                      "stepController": "proteus.StepControl.FixedStep",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton"},
            "default": {"maxNonlinearIts": 50},
        },
        "hints": ["maxNonlinearIts"],
    },
    {
        "cell": "pres",
        "family": "two_phase_level_set",
        "role": "pressure",
        "module": "proteus.mprans.Pres",
        "name": "pressure",
        "formulation": AS_WRITTEN,
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields"],
        "time_integration": {"options": _NO_TIME, "default": "none"},
        "physics": {"coefficients": "{module}.Coefficients", "LevelModelType": "{module}.LevelModel"},
        "numerics": {
            "fixed": {"numericalFluxType": "proteus.NumericalFlux.ConstantAdvection_exterior",
                      "subgridError": None, "shockCapturing": None,
                      "stepController": "proteus.StepControl.FixedStep",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton"},
        },
        "hints": [],
    },
    {
        "cell": "presinit",
        "family": "two_phase_level_set",
        "role": "pressure_spin_up",
        "module": "proteus.mprans.PresInit",
        "name": "pressureInitial",
        "formulation": AS_WRITTEN,
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields"],
        "time_integration": {"options": _NO_TIME, "default": "none"},
        "physics": {"coefficients": "{module}.Coefficients"},
        "numerics": {
            "fixed": {"numericalFluxType": "proteus.NumericalFlux.ConstantAdvection_exterior",
                      "subgridError": None, "shockCapturing": None,
                      "stepController": "proteus.StepControl.FixedStep",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton"},
        },
        "hints": [],
    },
    # ---- two-phase level-set flow: the interface -------------------------------------------------------------
    _scalar_transport("vof", "volume_fraction", "proteus.mprans.VOF", "vof",
                      {"useMetrics": 1.0}, _SUPG),
    _scalar_transport("ncls", "level_set", "proteus.mprans.NCLS", "ncls",
                      {"useMetrics": 1.0, "checkMass": False}, _SUPG, extra_hints=["checkMass"]),
    {
        "cell": "rdls",
        "family": "two_phase_level_set",
        "role": "redistancing",
        "module": "proteus.mprans.RDLS",
        "name": "rdls",
        "formulation": AS_WRITTEN,
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields"],
        # pseudo-time to steady state: no physical time integration, a Newton step controller
        "time_integration": {"options": _NO_TIME, "default": "none"},
        "stabilization": {"options": ["SUPG"], "default": "SUPG",
                          "shock_capturing": {"factor": 0.9, "lag": False}},
        "physics": {
            "coefficients": "{module}.Coefficients",
            "LevelModelType": "{module}.LevelModel",
            "default": {"useMetrics": 1.0},
            # the regularized sign's width, in element sizes
            "from_sign_width": ["epsFact"],
            # weak Dirichlet condition phi = phi* near the zero level set of what the transport left
            "attributes": {"weakDirichletConditions": {0: "{module}.setZeroLSweakDirichletBCsSimple"}},
        },
        "numerics": {
            "fixed": {"numericalFluxType": "proteus.NumericalFlux.DoNothing",
                      "subgridError": "{module}.SubgridError",
                      "shockCapturing": "{module}.ShockCapturing",
                      "stepController": "proteus.StepControl.Newton_controller",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton",
                      "nonlinearSmoother": "proteus.NonlinearSolvers.NLGaussSeidel"},
            "default": {"maxNonlinearIts": 50},
        },
        "hints": ["epsFact", "useMetrics", "maxNonlinearIts"],
    },
    {
        "cell": "mcorr",
        "family": "two_phase_level_set",
        "role": "mass_correction",
        "module": "proteus.mprans.MCorr",
        "name": "mcorr",
        "formulation": AS_WRITTEN,
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields"],
        # MCorr's equation has no time derivative; Proteus steps it with BackwardEuler_cfl, which carries dt only
        "time_integration": {"options": {"none": _IMPLICIT_TIME["backward_euler"]}, "default": "none"},
        "physics": {
            "coefficients": "{module}.Coefficients",
            "LevelModelType": "{module}.LevelModel",
            "default": {"useMetrics": 1.0, "checkMass": False},
            "from_heaviside_width": ["epsFactHeaviside", "epsFactDirac"],
            # the correction's numerical diffusion width (binds: {εd: {formula: "10 h"}})
            "from_binding": {"εd": "epsFactDiffusion"},
        },
        "numerics": {
            "fixed": {"numericalFluxType": "proteus.NumericalFlux.DoNothing",
                      "subgridError": None, "shockCapturing": None,
                      "stepController": "proteus.StepControl.Min_dt_cfl_controller",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton"},
            "default": {"maxNonlinearIts": 50, "useEisenstatWalker": True},
        },
        "hints": ["epsFactHeaviside", "epsFactDirac", "epsFactDiffusion", "useMetrics", "checkMass",
                  "maxNonlinearIts", "useEisenstatWalker"],
    },
    # CLSVOF: transport and redistancing of the level set in one solve, with the volume of heaviside(phi)
    # conserved (Quezada de Luna et al.). Replaces VOF, NCLS, RDLS and MCorr.
    {
        "cell": "clsvof",
        "family": "two_phase_level_set",
        "role": "level_set_redistanced",
        "module": "proteus.mprans.CLSVOF",
        "name": "clsvof",
        "formulation": AS_WRITTEN,
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields"],
        "time_integration": {"options": {"backward_euler": _IMPLICIT_TIME["backward_euler"]},
                             "default": "backward_euler"},
        "physics": {
            "coefficients": "{module}.Coefficients",
            "LevelModelType": "{module}.LevelModel",
            "default": {"useMetrics": 1.0, "computeMetrics": 1, "epsFactRedist": 0.33, "lambdaFact": 10.0,
                        "disc_ICs": False},
            "from_heaviside_width": ["epsFactHeaviside", "epsFactDirac"],
        },
        "numerics": {
            "fixed": {"numericalFluxType": "{module}.NumericalFlux",
                      "subgridError": None, "shockCapturing": None,
                      "stepController": "proteus.StepControl.Min_dt_controller",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton",
                      "levelNonlinearSolver": "proteus.NonlinearSolvers.CLSVOFNewton"},
            "default": {"maxNonlinearIts": 50, "updateJacobian": True},
        },
        "hints": ["epsFactHeaviside", "epsFactDirac", "epsFactRedist", "lambdaFact", "useMetrics",
                  "computeMetrics", "disc_ICs", "maxNonlinearIts", "updateJacobian"],
    },
    # ---- shallow water ----------------------------------------------------------------------------------------
    # SW2DCV: convex-limited graph viscosity with SSP RK3 (an algebraic scheme, no weak form). The scheme's
    # options are free-form in ymf (the discretization's `discretization:` block) until a second solver needs them
    # typed; ``from_scheme`` maps those keys onto the coefficients' flags.
    {
        "cell": "sw2dcv",
        "family": "shallow_water",
        "role": "shallow_water",
        "module": "proteus.mprans.SW2DCV",
        "name": "sw",
        "formulation": AS_WRITTEN,
        "scheme": {"options": ["graph_viscosity_convex_limiting"], "default": "graph_viscosity_convex_limiting"},
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields", "fields", "fields"],
        "time_integration": {"options": _SSP_RK3, "default": "ssp_rk3"},
        "physics": {
            "coefficients": "{module}.Coefficients",
            "LevelModelType": "{module}.LevelModel",
            "default": {"cE": 1.0, "LUMPED_MASS_MATRIX": 0, "INTEGRATE_BY_PARTS_WALLS": 1, "BOUNDS_RELAXATION": 1,
                        "BOUNDS_EXCLUDE_SELF": 0, "WALL_RIEMANN": 0, "LOW_ORDER_ONLY": 0, "NODAL_H_BOUNDS": 0,
                        "NO_STAR_STATES": 0},
            "from_scheme": {"entropy_viscosity_coefficient": ("cE", float), "lumped_mass": ("LUMPED_MASS_MATRIX", int),
                            "bounds_relaxation": ("BOUNDS_RELAXATION", int)},
        },
        "numerics": {
            "fixed": {"numericalFluxType": "{module}.NumericalFlux",
                      "subgridError": None, "shockCapturing": None,
                      "stepController": "proteus.StepControl.Min_dt_controller",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton",
                      "fullNewtonFlag": False},
            # the explicit solve: the consistent mass matrix, or the lumped one
            "by_flag": {"LUMPED_MASS_MATRIX": {
                0: {"levelNonlinearSolver":
                    "proteus.NonlinearSolvers.ExplicitConsistentMassMatrixShallowWaterEquationsSolver"},
                1: {"levelNonlinearSolver":
                    "proteus.NonlinearSolvers.ExplicitLumpedMassMatrixShallowWaterEquationsSolver"}}},
        },
        "hints": ["cE", "LUMPED_MASS_MATRIX", "INTEGRATE_BY_PARTS_WALLS", "BOUNDS_RELAXATION", "BOUNDS_EXCLUDE_SELF",
                  "WALL_RIEMANN", "LOW_ORDER_ONLY", "NODAL_H_BOUNDS", "NO_STAR_STATES"],
    },
    # GN_SW2DCV: the dispersive model by hyperbolic relaxation (unknowns h, hu, hv, h eta, h w, h beta).
    {
        "cell": "gn_sw2dcv",
        "family": "dispersive_shallow_water",
        "role": "shallow_water",
        "module": "proteus.mprans.GN_SW2DCV",
        "name": "GN_sw",
        "formulation": AS_WRITTEN,
        "scheme": {"options": ["graph_viscosity_convex_limiting"], "default": "graph_viscosity_convex_limiting"},
        "spaces": {"fields": {"options": [("CG", 1)], "default": ("CG", 1)}},
        "components": ["fields"] * 6,
        "time_integration": {"options": _SSP_RK3, "default": "ssp_rk3"},
        "physics": {
            "coefficients": "{module}.Coefficients",
            "LevelModelType": "{module}.LevelModel",
            "default": {"LUMPED_MASS_MATRIX": 0, "INTEGRATE_BY_PARTS_WALLS": 1},
            "from_scheme": {"lumped_mass": ("LUMPED_MASS_MATRIX", int)},
        },
        "numerics": {
            "fixed": {"numericalFluxType": "{module}.NumericalFlux",
                      "subgridError": None, "shockCapturing": None,
                      "stepController": "proteus.StepControl.Min_dt_controller",
                      "multilevelNonlinearSolver": "proteus.NonlinearSolvers.Newton",
                      "fullNewtonFlag": False,
                      # GN_sw_n uses the lumped solve whatever LUMPED_MASS_MATRIX says
                      "levelNonlinearSolver":
                          "proteus.NonlinearSolvers.ExplicitLumpedMassMatrixShallowWaterEquationsSolver"},
        },
        "hints": ["LUMPED_MASS_MATRIX", "INTEGRATE_BY_PARTS_WALLS"],
    },
]

#: the system's step controller by the discretization's step_control type
STEP_CONTROL = {
    # every model proposes a dt from its CFL, the system takes the minimum (TwoPhaseFlow)
    ("two_phase_level_set", "min_cfl"): "proteus.SplitOperator.Sequential_MinAdaptiveModelStep",
    # SWFlow's SWEs_so leaves default_so's controller
    ("shallow_water", "min_cfl"): "proteus.SplitOperator.Sequential_MinModelStep",
    ("dispersive_shallow_water", "min_cfl"): "proteus.SplitOperator.Sequential_MinModelStep",
}


# -----------------------------------------------------------------------------------------------------------------
# Realization (C5 step 2): what proteus.DispatchRun needs to build a dispatched discretization through the decks'
# own machinery (TwoPhaseFlowProblem, SWFlowProblem) and run it.
# -----------------------------------------------------------------------------------------------------------------

#: the deck machinery each family is realized through; a family without an entry dispatches but does not run yet
REALIZERS = {
    "two_phase_level_set": "proteus.DispatchRun.two_phase",
    "shallow_water": "proteus.DispatchRun.shallow_water",
}

#: TwoPhaseFlow's Parameters class for each module: the deck machinery builds each model's p and n from it
TWO_PHASE_PARAMETERS = {
    "proteus.mprans.RANS2P": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelRANS2P",
    "proteus.mprans.RANS3PF": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelRANS3PF",
    "proteus.mprans.PresInc": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelPressureIncrement",
    "proteus.mprans.Pres": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelPressure",
    "proteus.mprans.PresInit": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelPressureInitial",
    "proteus.mprans.VOF": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelVOF",
    "proteus.mprans.NCLS": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelNCLS",
    "proteus.mprans.RDLS": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelRDLS",
    "proteus.mprans.MCorr": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelMCorr",
    "proteus.mprans.CLSVOF": "proteus.TwoPhaseFlow.utils.Parameters.ParametersModelCLSVOF",
}

#: a dispatched numerics attribute -> where TwoPhaseFlow's Parameters takes it before it builds the n module.
#: ``n.<a>.<b>`` is an options object of the n module; ``timeDiscretization`` is the Parameters' own switch, by the
#: dispatched time integration. Anything else is built by the Parameters class itself and checked afterwards.
TWO_PHASE_NUMERICS_KNOBS = {
    "shockCapturing.shockCapturingFactor": "n.ShockCapturingOptions.shockCapturingFactor",
    "shockCapturing.lag": "n.ShockCapturingOptions.lag",
    "subgridError.lag": "n.SubgridErrorOptions.lag",
    "maxNonlinearIts": "n.maxNonlinearIts",
    "useEisenstatWalker": "n.useEisenstatWalker",
    "conservativeFlux": "n.conservativeFlux",
}
TWO_PHASE_TIME_DISCRETIZATION = {
    "proteus.TimeIntegration.BackwardEuler_cfl": "be",
    "proteus.TimeIntegration.VBDF": "vbdf",
}

#: the roles (ymf.vocabulary.COEFFICIENT_ROLES) of the model's coefficients TwoPhaseFlow takes as SystemPhysics values.
#: A model states a coefficient's role (``ρ: {formula: ..., role: density}``) and the realizer finds the coefficient by
#: it, whatever the model calls it. Phase 0 is where the phase indicator (the coefficient that is heaviside of the
#: level set) is 0, phase 1 where it is 1.
TWO_PHASE_COEFFICIENT_ROLES = ["density", "dynamic_viscosity", "gravity"]

#: a boundary region's closed conditions (the expansion ymf writes, never the alias) -> the BC_RANS setter that
#: imposes them. A signature is the set of (variable, type, component, only, value); values are numbers, a vector's
#: per component, and for a signed condition without a value ``sign:<±1>``. ``kwargs`` takes a setter argument from one
#: condition, by (variable, type) and optionally the path of the field (default ``value``).
#: A region whose conditions match no signature is refused.
TWO_PHASE_BOUNDARY_SETTERS = [
    {"name": "free_slip", "setter": "setFreeSlip",
     "signature": {("v", "dirichlet", "normal", None, 0.0), ("v", "neumann", "tangential", None, 0.0)}},
    # setAtmosphere also sets the velocity's Dirichlet data to 0 (which RANS2P applies weakly), beyond the zero
    # Neumann condition the expansion of `open` states: a question for C12, kept as the deck has it
    {"name": "atmosphere", "setter": "setAtmosphere",
     "signature": {("p", "dirichlet", None, None, 0.0), ("v", "neumann", None, None, "zero_vector"),
                   ("χ", "dirichlet", None, "inflow", "any")},
     "kwargs": {"vof_air": ("χ", "dirichlet")}},
    # With CLSVOF as the interface solver no sub-problem carries χ, and ymf states the atmosphere's inflow on the
    # level set itself: φ is signed (+1, air) where flow enters, with the phase indicator's value on the condition
    # (`indicator`). setAtmosphere sets no CLSVOF data (clsvof_dirichlet stays unset, as in damBreakLevelSet's
    # interface=1), so this realization is weaker than the spec on one point, and says so (``note``, which the run
    # records): where flow enters, CLSVOF's level set is not held at the air side. vof_air is the indicator's value.
    {"name": "atmosphere_clsvof", "setter": "setAtmosphere",
     "signature": {("p", "dirichlet", None, None, 0.0), ("v", "neumann", None, None, "zero_vector"),
                   ("φ", "dirichlet", None, "inflow", "sign:1")},
     "kwargs": {"vof_air": ("φ", "dirichlet", "indicator.value")},
     "note": "setAtmosphere sets no CLSVOF boundary data: the level set's inflow condition (sign 1, indicator "
             "value %(vof_air)s) is not imposed"},
]

#: the roles of the projection scheme's own variables (the pressure increment, the pressure spin-up): the spec states no
#: unknown for them, and a run starts them at zero as the decks do
SCHEME_AUXILIARY_ROLES = ("pressure_increment", "pressure_spin_up")

#: a regularization by name -> Proteus's own implementation, so that an initial condition such as heaviside(φ)
#: under smoothed_heaviside of width ε is the value the decks compute (smoothedHeaviside(ε, φ)) bit for bit
REGULARIZED_FUNCTIONS = {
    "smoothed_heaviside": "proteus.ctransportCoefficients.smoothedHeaviside",
}

#: SWFlow's p and n modules for each module (proteus/SWFlow/models, read through Context's mySWFlowProblem, as
#: parun --SWEs loads them) and its sw_model switch
SHALLOW_WATER_MODELS = {
    "proteus.mprans.SW2DCV": {"sw_model": 0, "p": "sw_p", "n": "sw_n"},
}

#: the coefficient role (ymf.vocabulary.COEFFICIENT_ROLES) -> SWFlow's physical_parameters key. The bed
#: (``bed_elevation``, the model's Z) is SWFlowProblem's bathymetry; the Manning coefficient is n in the friction
#: coefficient τb, the linear and quadratic drag are the porous medium's α and β.
SHALLOW_WATER_COEFFICIENT_ROLES = {"gravity": "gravity", "manning_coefficient": "mannings", "porosity": "porosity",
                                   "linear_drag": "drag_alpha", "quadratic_drag": "drag_beta"}
SHALLOW_WATER_BED_ROLE = "bed_elevation"

#: SWFlow's swe_parameters a dispatched coefficient flag sets (the same names), and the time order
SHALLOW_WATER_FLAGS = ["cE", "LUMPED_MASS_MATRIX", "INTEGRATE_BY_PARTS_WALLS", "BOUNDS_RELAXATION",
                       "BOUNDS_EXCLUDE_SELF", "WALL_RIEMANN", "LOW_ORDER_ONLY", "NODAL_H_BOUNDS", "NO_STAR_STATES"]

#: closed conditions on the whole boundary -> SWFlow's boundary treatment. Walls everywhere is its reflectingBCs
#: (no Dirichlet data); anything else is refused until SWFlow's open boundaries are wired up
SHALLOW_WATER_BOUNDARY = [
    {"name": "wall", "reflectingBCs": True,
     "signature": {("q", "dirichlet", "normal", None, 0.0), ("q", "neumann", "tangential", None, 0.0)}},
]
