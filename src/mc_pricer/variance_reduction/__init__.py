"""Variance-reduction constructions with explicit independent sampling units."""

from mc_pricer.variance_reduction.antithetic import pair_means, simulate_antithetic_payoffs
from mc_pricer.variance_reduction.control_variates import ControlVariateFit, fit_control_variate, apply_control_variate

__all__ = ["pair_means", "simulate_antithetic_payoffs", "ControlVariateFit", "fit_control_variate", "apply_control_variate"]
