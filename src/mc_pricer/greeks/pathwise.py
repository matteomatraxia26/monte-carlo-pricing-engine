"""Pathwise European Delta under exact non-dividend-paying GBM.

The derivative is evaluated along each path before averaging. This method
requires regularity beyond almost-sure payoff differentiability; do not apply
it unchanged to digital or barrier payoffs, or differentiate it for Gamma.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from mc_pricer._validation import finite_real, finite_vector, positive_integer
from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.products.european import EuropeanOption
from mc_pricer.pricing.monte_carlo import price_from_payoff_samples
from mc_pricer.simulation.paths import simulate_terminal
from mc_pricer.statistics.confidence_intervals import validate_confidence_level


@dataclass(frozen=True)
class PathwiseDeltaResult:
    """Delta and approximate IID sampling uncertainty; no finite-difference bump.

    sample_variance is the variance of one discounted derivative sample.
    n_paths is the independent terminal count, each valued once. Runtime scope
    depends on whether simulation or supplied terminal samples were requested.
    A zero empirical SE does not prove absence of rare-event uncertainty.
    """

    estimate: float
    standard_error: float
    sample_variance: float
    confidence_interval: tuple[float, float]
    n_paths: int
    runtime_seconds: float
    confidence_level: float = .95

    @property
    def estimator_variance(self) -> float:
        """Sampling variance of the estimated Delta mean."""
        return self.sample_variance / self.n_paths

    @property
    def n_terminal_evaluations(self) -> int:
        """One terminal-state and derivative evaluation per independent row."""
        return self.n_paths


def european_pathwise_delta_samples(
    product: EuropeanOption, terminal_prices: ArrayLike, *, spot: float,
) -> NDArray[np.float64]:
    """Return UNDISCOUNTED GBM derivative samples of shape (N,), N>=1.

    Call: (ST/S0)*1{ST>K}; put: -(ST/S0)*1{ST<K}. This uses the GBM scaling
    derivative dST/dS0=ST/S0 with all other parameters held fixed. It is not a
    formula for arbitrary dynamics. Supplied prices must be strictly positive.
    Equality at K returns zero by convention for both products; under positive
    GBM variance this event has probability zero. Inputs are not modified.
    """
    if not isinstance(product, EuropeanOption):
        raise TypeError('pathwise Delta supports EuropeanOption only')
    spot = finite_real('spot', spot)
    if spot <= 0:
        raise ValueError('spot must be positive')
    terminal = finite_vector('terminal_prices', terminal_prices)
    if np.any(terminal <= 0):
        raise ValueError('GBM terminal prices must be strictly positive')
    samples = np.zeros(terminal.shape, dtype=np.float64)
    active = terminal > product.strike if product.option_type == 'call' else terminal < product.strike
    # where= avoids an unnecessary or overflowing ratio on inactive paths.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        np.divide(terminal, spot, out=samples, where=active)
    if product.option_type == 'put':
        samples *= -1.
    return samples


def _validate_inputs(model: BlackScholesModel, product: EuropeanOption, maturity: float,
                     confidence_level: float) -> tuple[float, float]:
    if not isinstance(model, BlackScholesModel):
        raise TypeError('pathwise Delta requires a BlackScholesModel')
    if not isinstance(product, EuropeanOption):
        raise TypeError('pathwise Delta supports EuropeanOption only')
    time = finite_real('maturity', maturity)
    if time <= 0 or model.volatility <= 0:
        raise ValueError('pathwise Delta requires positive maturity and volatility')
    level = validate_confidence_level(confidence_level)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        discount = float(np.exp(-model.rate*time))
    if not np.isfinite(discount) or discount <= 0:
        raise FloatingPointError('discount factor exceeds positive finite float64 range')
    return time, level


def pathwise_delta_from_terminal_samples(
    model: BlackScholesModel, product: EuropeanOption, maturity: float,
    terminal_prices: ArrayLike, *, confidence_level: float = .95,
) -> PathwiseDeltaResult:
    """Summarize pathwise Delta from IID terminal values (N,), N>=2.

    Caller must supply IID draws from this model/maturity. No distribution test
    is performed. SE uses ddof=1 on discounted derivative samples. Antithetic
    legs, QMC and dependent samples need different uncertainty treatment.
    Runtime excludes terminal simulation. T=0 and sigma=0 are outside this API.
    """
    start = perf_counter()
    time, level = _validate_inputs(model, product, maturity, confidence_level)
    terminal = finite_vector('terminal_prices', terminal_prices, min_size=2)
    samples = european_pathwise_delta_samples(product, terminal, spot=model.spot)
    stats = price_from_payoff_samples(samples, rate=model.rate, maturity=time, confidence_level=level)
    return PathwiseDeltaResult(stats.price, stats.standard_error, stats.sample_variance,
                               stats.confidence_interval, stats.n_paths, perf_counter()-start, level)


def pathwise_delta(
    model: BlackScholesModel, product: EuropeanOption, maturity: float, n_paths: int, *,
    rng: Generator, confidence_level: float = .95,
) -> PathwiseDeltaResult:
    """Simulate terminal GBM and estimate European call/put Delta pathwise.

    O(N) time/storage; N standard-normal draws and N terminal evaluations.
    All normal input errors fail before RNG consumption. Numerical range
    failures during simulation may advance the RNG. No bump size is required.
    """
    start = perf_counter()
    time, level = _validate_inputs(model, product, maturity, confidence_level)
    count = positive_integer('n_paths', n_paths)
    if count < 2:
        raise ValueError('at least two independent terminal samples are required')
    terminal = simulate_terminal(model, time, count, rng=rng)
    result = pathwise_delta_from_terminal_samples(model, product, time, terminal, confidence_level=level)
    return replace(result, runtime_seconds=perf_counter()-start)
