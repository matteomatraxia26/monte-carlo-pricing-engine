"""Plain Monte Carlo pricing on a specified Heston integration/monitoring grid."""
from dataclasses import dataclass
from time import perf_counter
from numpy.random import Generator
from mc_pricer.models.heston import HestonModel
from mc_pricer.products.european import EuropeanOption
from mc_pricer.products.digital import DigitalOption
from mc_pricer.products.asian import ArithmeticAsianOption, GeometricAsianOption
from mc_pricer.products.barrier import UpAndOutCall
from mc_pricer.pricing.monte_carlo import MonteCarloResult, price_from_payoff_samples, _pricing_inputs
from mc_pricer.simulation.heston import simulate_heston_paths, _configuration

HestonProduct = EuropeanOption | DigitalOption | ArithmeticAsianOption | GeometricAsianOption | UpAndOutCall


@dataclass(frozen=True)
class HestonMonteCarloResult(MonteCarloResult):
    """Sampling CI for a discretized Heston price, excluding timestep bias.

    n_steps is the integration count, n_observations the contractual future
    monitoring count (one for terminal products). No control or antithetic method
    is enabled here; n_paths equals n_independent_samples.
    """
    n_observations: int = 1
    negative_variance_fraction: float = 0.
    scheme: str = 'full_truncation_log_euler'


def price_heston_monte_carlo(model: HestonModel, product: HestonProduct,
                             maturity: float, n_paths: int, n_steps: int, *,
                             rng: Generator, n_observations: int | None = None,
                             confidence_level: float = .95) -> HestonMonteCarloResult:
    """Price with full truncation, keeping integration and monitoring separate.

    Terminal products omit n_observations; path products default to all steps,
    or specify a divisor of n_steps. Asians exclude S0; barriers include it.
    Runtime includes draws/simulation/payoff/statistics. Plain IID sampling only.
    This separate API prevents applying GBM-specific controls/Greeks to Heston.
    """
    start = perf_counter()
    if not isinstance(product, (EuropeanOption, DigitalOption, ArithmeticAsianOption, GeometricAsianOption, UpAndOutCall)):
        raise TypeError('unsupported Heston product')
    terminal = isinstance(product, (EuropeanOption, DigitalOption))
    if terminal and n_observations is not None:
        raise ValueError('terminal product: omit n_observations')
    observations = 1 if terminal else n_observations
    time, count, steps, observations = _configuration(model, maturity, n_paths, n_steps, observations)
    if count < 2:
        raise ValueError('at least two independent paths are needed')
    _, level, _ = _pricing_inputs(model.rate, time, confidence_level)
    paths = simulate_heston_paths(model, time, count, steps, rng=rng, n_observations=observations)
    payoff = product.payoff(paths.stock_paths[:,-1] if terminal else paths.stock_paths)
    stats = price_from_payoff_samples(payoff, rate=model.rate, maturity=time, confidence_level=level)
    return HestonMonteCarloResult(price=stats.price, standard_error=stats.standard_error,
        sample_variance=stats.sample_variance, confidence_interval=stats.confidence_interval,
        n_paths=count, n_independent_samples=count, runtime_seconds=perf_counter()-start,
        confidence_level=level, n_steps=steps, n_observations=observations,
        negative_variance_fraction=paths.negative_variance_fraction)
