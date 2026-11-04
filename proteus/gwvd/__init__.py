"""
Variable-density groundwater flow and transport (Povich et al. 2013, Comput. Geosci. 17:529).

Port of the continuous Galerkin path of erdc/proteus-gwvd at commit 14961d2 (2012-07-31): the
mod2 flow and transport coefficients, the Knabner-Frolkovic consistent-velocity correction and the
time-integration, step-control and iterative split-operator classes the 2012 decks use. The C
coefficient evaluators are reimplemented with numpy.
"""
__all__ = ["GwvdCoefficients", "Gwvd_consolidated_classes"]
