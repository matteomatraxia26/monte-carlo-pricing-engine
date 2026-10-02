"""Likelihood-ratio spot Delta for terminal European GBM payoffs.

Differentiate the lognormal density at a FIXED terminal value, then evaluate
its score using the original Gaussian draw. Retaining Z avoids reconstructing
it with a potentially ill-conditioned logarithmic subtraction at small sigma√T.
This terminal-density score is not a path-dependent Greek formula.
"""

from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from mc_pricer._validation import finite_real, finite_vector, positive_integer
from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.products.european import EuropeanOption
from mc_pricer.products.digital import DigitalOption
from mc_pricer.pricing.monte_carlo import price_from_payoff_samples
from mc_pricer.simulation.paths import terminal_from_normals
from mc_pricer.simulation.random import standard_normal_draws
from mc_pricer.statistics.confidence_intervals import validate_confidence_level

TerminalProduct = EuropeanOption | DigitalOption


@dataclass(frozen=True)
class LikelihoodRatioDeltaResult:
    """Delta with normal IID sampling CI; no finite-difference bump bias.

    sample_variance: one discounted payoff-times-score sample, not its mean.
    n_paths counts independent draws, each with one terminal/payoff evaluation.
    A zero empirical SE does not establish absence of rare-event uncertainty.
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
        return self.sample_variance / self.n_paths

    @property
    def n_terminal_evaluations(self) -> int:
        return self.n_paths


def _score_inputs(model: BlackScholesModel, maturity: float) -> tuple[float, float]:
    if not isinstance(model, BlackScholesModel):
        raise TypeError('likelihood ratio requires a BlackScholesModel')
    time = finite_real('maturity', maturity)
    if time <= 0 or model.volatility <= 0:
        raise ValueError('likelihood ratio requires positive maturity and volatility')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        denominator = float(model.spot * model.volatility * np.sqrt(time))
    if not np.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError('score denominator exceeds positive finite float64 range')
    return time, denominator


def gbm_spot_score(model: BlackScholesModel, maturity: float,
                   normals: ArrayLike) -> NDArray[np.float64]:
    """Return d(log p(ST;S0))/dS0 = Z/(S0*sigma*sqrt(T)), shape (N,).

    Input is the Gaussian draw generating ST under this model and maturity.
    It is NOT d(log p(Z))/dS0. The normal density itself has no spot dependence.
    Accept finite real vectors (N,), N>=1, without testing normality or mutating.
    """
    _, denominator = _score_inputs(model, maturity)
    z = finite_vector('normals', normals)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        return z / denominator


def likelihood_ratio_delta_samples(model: BlackScholesModel, product: TerminalProduct,
                                   maturity: float, normals: ArrayLike) -> NDArray[np.float64]:
    """Return UNDISCOUNTED payoff times spot score, shape (N,), N>=1.

    Supports vanilla/digital calls and puts with fixed strike and payout.
    Payoffs have no explicit S0 dependence when the terminal argument is held
    fixed. For general parameter-dependent payoffs a derivative term is needed.
    No path, antithetic or control-variate uncertainty conventions are inferred.
    """
    if not isinstance(product, (EuropeanOption, DigitalOption)):
        raise TypeError('likelihood ratio supports EuropeanOption or DigitalOption only')
    score = gbm_spot_score(model, maturity, normals)
    terminal = terminal_from_normals(model, maturity, normals)
    with np.errstate(over='raise', invalid='raise'):
        return product.payoff(terminal) * score


def _validate_inputs(model: BlackScholesModel, product: TerminalProduct,
                     maturity: float, confidence_level: float) -> tuple[float, float]:
    if not isinstance(product, (EuropeanOption, DigitalOption)):
        raise TypeError('likelihood ratio supports EuropeanOption or DigitalOption only')
    time, _ = _score_inputs(model, maturity)
    level = validate_confidence_level(confidence_level)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        discount = float(np.exp(-model.rate*time))
    if not np.isfinite(discount) or discount <= 0:
        raise FloatingPointError('discount factor exceeds positive finite float64 range')
    return time, level


def likelihood_ratio_delta_from_normals(
    model: BlackScholesModel, product: TerminalProduct, maturity: float,
    normals: ArrayLike, *, confidence_level: float = .95,
) -> LikelihoodRatioDeltaResult:
    """Summarize IID supplied normal draws (N,), N>=2, without mutation.

    SE uses ddof=1 on discounted payoff-score samples. Caller owns the IID
    standard-normal assumption. Runtime includes transformation and payoff/score,
    excludes RNG draws. Reuse shocks explicitly for paired estimator diagnostics.
    """
    start = perf_counter()
    time, level = _validate_inputs(model, product, maturity, confidence_level)
    z = finite_vector('normals', normals, min_size=2)
    samples = likelihood_ratio_delta_samples(model, product, time, z)
    stats = price_from_payoff_samples(samples, rate=model.rate, maturity=time, confidence_level=level)
    return LikelihoodRatioDeltaResult(stats.price, stats.standard_error, stats.sample_variance,
                                      stats.confidence_interval, stats.n_paths, perf_counter()-start, level)


def likelihood_ratio_delta(
    model: BlackScholesModel, product: TerminalProduct, maturity: float, n_paths: int,
    *, rng: Generator, confidence_level: float = .95,
) -> LikelihoodRatioDeltaResult:
    """Simulate terminal GBM and estimate spot Delta using likelihood ratios.

    O(N) time/storage, N normal draws and N terminal evaluations. Validate ordinary
    configuration errors before drawing; simulation range failures may advance RNG.
    Only spot Delta is implemented; no likelihood-ratio Gamma, Vega or path score.
    """
    start = perf_counter()
    time, level = _validate_inputs(model, product, maturity, confidence_level)
    count = positive_integer('n_paths', n_paths)
    if count < 2:
        raise ValueError('at least two independent normal draws are required')
    normals = standard_normal_draws(count, rng=rng)
    result = likelihood_ratio_delta_from_normals(model, product, time, normals, confidence_level=level)
    return replace(result, runtime_seconds=perf_counter()-start)
