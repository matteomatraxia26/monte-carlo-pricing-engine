"""Analytical benchmarks, GBM Greeks and GBM/Heston Monte Carlo pricing."""

from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.pricing.analytical import (
    BlackScholesGreeks,
    ArithmeticAverageMoments,
    arithmetic_average_moments,
    black_scholes_greeks,
    black_scholes_price,
    terminal_monitored_up_and_out_price,
    GeometricAverageLogMoments, geometric_average_log_moments, geometric_asian_price,
)
from mc_pricer.simulation.paths import paths_from_normals, simulate_paths, simulate_terminal, terminal_from_normals
from mc_pricer.simulation.random import make_rng, standard_normal_draws, standard_normal_matrix
from mc_pricer.products.european import EuropeanOption
from mc_pricer.products.digital import DigitalOption
from mc_pricer.products.asian import ArithmeticAsianOption, GeometricAsianOption
from mc_pricer.products.barrier import UpAndOutCall
from mc_pricer.pricing.monte_carlo import (
    MonteCarloResult, price_from_payoff_samples, price_from_path_samples,
    price_from_terminal_samples, price_monte_carlo,
    price_from_antithetic_samples, price_from_control_samples,
)
from mc_pricer.variance_reduction.antithetic import pair_means, simulate_antithetic_payoffs

from mc_pricer.variance_reduction.control_variates import ControlVariateFit, fit_control_variate, apply_control_variate

from mc_pricer.greeks import (
    FiniteDifferenceResult, central_first_difference, central_second_difference,
    finite_difference_from_normals, finite_difference_greek,
)

from mc_pricer.greeks.pathwise import (
    PathwiseDeltaResult, european_pathwise_delta_samples,
    pathwise_delta_from_terminal_samples, pathwise_delta,
)

from mc_pricer.pricing.digital import digital_price, digital_delta

from mc_pricer.greeks.likelihood_ratio import (
    LikelihoodRatioDeltaResult, gbm_spot_score, likelihood_ratio_delta_samples,
    likelihood_ratio_delta_from_normals, likelihood_ratio_delta,
)

from mc_pricer.models.heston import HestonModel
from mc_pricer.simulation.heston import HestonPaths, correlate_normals, heston_paths_from_normals, simulate_heston_paths
from mc_pricer.pricing.heston_monte_carlo import HestonMonteCarloResult, price_heston_monte_carlo

from mc_pricer.pricing.heston_analytical import (
    HestonReferenceResult, deterministic_integrated_variance,
    heston_centered_characteristic, heston_european_reference,
)

__all__ = [
    "HestonReferenceResult", "deterministic_integrated_variance",
    "heston_centered_characteristic", "heston_european_reference",
    "HestonModel",
    "HestonPaths",
    "correlate_normals",
    "heston_paths_from_normals",
    "simulate_heston_paths",
    "HestonMonteCarloResult",
    "price_heston_monte_carlo",

    "LikelihoodRatioDeltaResult", "gbm_spot_score", "likelihood_ratio_delta_samples",
    "likelihood_ratio_delta_from_normals", "likelihood_ratio_delta",
    "digital_price", "digital_delta",
    "DigitalOption",
    "PathwiseDeltaResult",
    "european_pathwise_delta_samples",
    "pathwise_delta_from_terminal_samples",
    "pathwise_delta",
    "FiniteDifferenceResult",
    "central_first_difference",
    "central_second_difference",
    "finite_difference_from_normals",
    "finite_difference_greek",
    "GeometricAsianOption",
    "ControlVariateFit",
    "fit_control_variate",
    "apply_control_variate",
    "GeometricAverageLogMoments",
    "geometric_average_log_moments",
    "geometric_asian_price",
    "price_from_control_samples",
    "BlackScholesModel",
    "BlackScholesGreeks",
    "ArithmeticAverageMoments",
    "arithmetic_average_moments",
    "black_scholes_price",
    "terminal_monitored_up_and_out_price",
    "black_scholes_greeks",
    "make_rng",
    "standard_normal_draws",
    "standard_normal_matrix",
    "terminal_from_normals",
    "simulate_terminal",
    "paths_from_normals",
    "simulate_paths",
    "EuropeanOption",
    "ArithmeticAsianOption",
    "UpAndOutCall",
    "MonteCarloResult",
    "price_from_payoff_samples",
    "price_from_path_samples",
    "price_from_terminal_samples",
    "price_monte_carlo",
    "pair_means",
    "simulate_antithetic_payoffs",
    "price_from_antithetic_samples",
]
