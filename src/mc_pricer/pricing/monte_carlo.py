"""Plain, antithetic and independent-pilot control-variate GBM pricing."""

from dataclasses import dataclass, replace
from math import sqrt
from numbers import Integral
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike

from mc_pricer._validation import finite_matrix, finite_real, finite_vector, positive_integer
from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.products.european import EuropeanOption
from mc_pricer.products.digital import DigitalOption
from mc_pricer.products.asian import ArithmeticAsianOption, GeometricAsianOption
from mc_pricer.products.barrier import UpAndOutCall
from mc_pricer.simulation.paths import simulate_paths, simulate_terminal
from mc_pricer.variance_reduction.antithetic import pair_means, simulate_antithetic_payoffs
from mc_pricer.variance_reduction.control_variates import fit_control_variate, apply_control_variate
from mc_pricer.pricing.analytical import geometric_asian_price
from mc_pricer.statistics.confidence_intervals import (
    normal_confidence_interval,
    validate_confidence_level,
)


@dataclass(frozen=True)
class MonteCarloResult:
    """Price and uncertainty from independent discounted sampling units.

    ``sample_variance`` estimates variance of one discounted sampling unit:
    an individual payoff for plain MC, a pair mean for antithetic MC, or a
    production residual conditional on the pilot for control variates. Its units
    are currency squared. ``estimator_variance`` estimates variance of the
    mean. ``standard_error`` and CI endpoints have price/currency units.
    Plain MC has n_independent_samples == n_paths; antithetic MC has n_paths/2.
    Control variates have n_paths-n_pilot_paths production sampling units;
    n_paths includes the pilot. control_expectation is an UNDISCOUNTED payoff mean.
    Runtime scope is documented by the function constructing the result.
    """

    price: float
    standard_error: float
    sample_variance: float
    confidence_interval: tuple[float, float]
    n_paths: int
    n_independent_samples: int
    runtime_seconds: float
    confidence_level: float = 0.95
    n_steps: int | None = None
    variance_reduction: Literal["none", "antithetic", "control_variate"] = "none"
    n_pilot_paths: int = 0
    control_beta: float | None = None
    control_expectation: float | None = None
    pilot_correlation: float | None = None
    control_fit_status: str | None = None

    @property
    def estimator_variance(self) -> float:
        """Estimated variance of the mean, not of an individual payoff."""
        return self.sample_variance / self.n_independent_samples


def _pricing_inputs(rate: float, maturity: float, confidence_level: float) -> tuple[float, float, float]:
    rate = finite_real("rate", rate)
    maturity = finite_real("maturity", maturity)
    if maturity < 0:
        raise ValueError("maturity must be nonnegative")
    level = validate_confidence_level(confidence_level)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        discount = float(np.exp(-rate * maturity))
    if not np.isfinite(discount) or discount <= 0:
        raise FloatingPointError("discount factor exceeds positive finite float64 range")
    return maturity, level, discount


def price_from_payoff_samples(
    payoffs: ArrayLike,
    *,
    rate: float,
    maturity: float,
    confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Summarize IID undiscounted payoff samples of shape (N,), N >= 2.

    This is the single discount/mean/variance/CI implementation used by all
    pricing routes. Signed samples are allowed, for example paired payoff
    differences. Samples must share a payment date and be independent across
    rows. Antithetic/QMC outputs require appropriate independent sampling units.
    Input is unchanged. Runtime excludes simulation and payoff construction.
    """
    start = perf_counter()
    _, level, discount = _pricing_inputs(rate, maturity, confidence_level)
    values = finite_vector("payoffs", payoffs, min_size=2)
    with np.errstate(over="raise", invalid="raise"):
        discounted_payoffs = discount * values
        # Identical values have exactly zero empirical variance. Avoid spurious
        # roundoff variance when their floating-point mean rounds differently.
        if np.all(discounted_payoffs == discounted_payoffs[0]):
            price = float(discounted_payoffs[0])
            sample_variance = 0.0
        else:
            price = float(np.mean(discounted_payoffs))
            sample_variance = float(np.var(discounted_payoffs, ddof=1))
    count = values.size
    standard_error = sqrt(sample_variance / count)
    interval = normal_confidence_interval(price, standard_error, level)
    return MonteCarloResult(
        price=price, standard_error=standard_error, sample_variance=sample_variance,
        confidence_interval=interval, n_paths=count, n_independent_samples=count,
        runtime_seconds=perf_counter() - start, confidence_level=level,
    )


def price_from_antithetic_samples(
    positive_payoffs: ArrayLike, negative_payoffs: ArrayLike, *, rate: float,
    maturity: float, confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Price from aligned undiscounted payoff vectors (M,), M>=2.

    Different pairs must be IID; members within a pair can be dependent.
    n_paths=2*M, n_independent_samples=M. Sample variance refers to discounted
    pair means, not individual payoffs. Runtime excludes simulation/payoff work.
    """
    start = perf_counter()
    averages = pair_means(positive_payoffs, negative_payoffs)
    result = price_from_payoff_samples(averages, rate=rate, maturity=maturity,
                                       confidence_level=confidence_level)
    return replace(result, n_paths=2 * averages.size, variance_reduction="antithetic",
                   runtime_seconds=perf_counter() - start)


def price_from_control_samples(
    target_payoffs: ArrayLike, control_payoffs: ArrayLike, *, control_expectation: float,
    beta: float, rate: float, maturity: float, confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Summarize independent production pairs using fixed/independently fitted beta.

    Target/control vectors (N,) and the known control mean are UNDISCOUNTED.
    Beta is fixed across production rows. Runtime and n_paths exclude the cost
    of obtaining beta; the full engine adds its pilot budget. Conditional on
    an independent pilot, residuals are IID. Caller owns independence/alignment.
    """
    start = perf_counter()
    beta = finite_real("beta", beta)
    expected = finite_real("control_expectation", control_expectation)
    adjusted = apply_control_variate(target_payoffs, control_payoffs,
                                     control_expectation=expected, beta=beta)
    result = price_from_payoff_samples(adjusted, rate=rate, maturity=maturity,
                                       confidence_level=confidence_level)
    return replace(result, variance_reduction="control_variate", control_beta=beta,
                   control_expectation=expected, runtime_seconds=perf_counter() - start)


def price_from_terminal_samples(
    product: EuropeanOption | DigitalOption, terminal_prices: ArrayLike, *, rate: float,
    maturity: float, confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Price a European contract from IID terminal prices of shape (N,), N>=2.

    Inputs are unchanged. Runtime includes validation, payoff and statistics,
    but excludes simulation. Discounting occurs once in the common estimator.
    """
    start = perf_counter()
    if not isinstance(product, (EuropeanOption, DigitalOption)):
        raise TypeError("price_from_terminal_samples requires a EuropeanOption or DigitalOption")
    _pricing_inputs(rate, maturity, confidence_level)
    terminal = finite_vector("terminal_prices", terminal_prices, min_size=2)
    result = price_from_payoff_samples(product.payoff(terminal), rate=rate,
                                       maturity=maturity, confidence_level=confidence_level)
    return replace(result, runtime_seconds=perf_counter() - start)


def price_from_path_samples(
    product: ArithmeticAsianOption | GeometricAsianOption | UpAndOutCall, paths: ArrayLike, *, rate: float,
    maturity: float, confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Price from IID path rows of shape (N, m+1), N>=2, m>=1.

    Column zero is initial spot. Asians average the remaining columns; barriers
    monitor every column, including inception. The caller supplies the schedule;
    dates are not inferred from prices. Correlation within a path is expected.
    Rows must be independent. Runtime excludes simulation and inputs are unchanged.
    """
    start = perf_counter()
    if not isinstance(product, (ArithmeticAsianOption, GeometricAsianOption, UpAndOutCall)):
        raise TypeError("price_from_path_samples requires an Asian or UpAndOutCall product")
    _pricing_inputs(rate, maturity, confidence_level)
    values = finite_matrix("paths", paths)
    if values.shape[0] < 2:
        raise ValueError("at least two independent paths are required")
    result = price_from_payoff_samples(product.payoff(values), rate=rate,
                                       maturity=maturity, confidence_level=confidence_level)
    return replace(result, n_steps=values.shape[1] - 1, runtime_seconds=perf_counter() - start)


def _asian_control_payoffs(
    model: BlackScholesModel, product: ArithmeticAsianOption, maturity: float,
    count: int, n_steps: int, rng: Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned arithmetic/geometric payoffs (count,) from one grid.

    The path matrix is released on return, so pilot and production grids do
    not coexist. Both contracts exclude S0 and share weights, strike and type.
    """
    paths = simulate_paths(model, maturity, count, n_steps, rng=rng)
    control = GeometricAsianOption(product.strike, product.option_type)
    return product.payoff(paths), control.payoff(paths)


def price_monte_carlo(
    model: BlackScholesModel,
    product: EuropeanOption | DigitalOption | ArithmeticAsianOption | GeometricAsianOption | UpAndOutCall,
    maturity: float,
    n_paths: int,
    *,
    rng: Generator,
    confidence_level: float = 0.95,
    n_steps: int | None = None,
    variance_reduction: Literal["none", "antithetic", "control_variate"] = "none",
    n_pilot_paths: int | None = None,
) -> MonteCarloResult:
    """Price a vanilla/digital European, discrete Asian or discrete up-and-out under exact GBM.

    n_paths is the TOTAL simulated path budget for every method. Plain MC
    requires at least two paths. Antithetic MC requires an even n_paths>=4;
    n_paths/2 pair means are independent units. Runtime
    includes simulation, payoff evaluation and statistics, excluding caller RNG
    construction. Path products require n_steps and uniform dates j*T/m.
    Asians exclude S0; barriers monitor S0 too. Even an initially knocked-out
    barrier follows the usual simulation and RNG consumption convention.
    European pricing requires n_steps=None and simulates
    terminal prices only. Control variates support arithmetic Asians only and
    require explicit n_pilot_paths>=2 plus at least two production paths. Pilot
    then production consume disjoint IID blocks from the supplied stream. Beta
    uses ONLY the pilot; production rows alone enter the final estimate/SE.
    Both payoffs are evaluated on each control-variate path. No combination
    with antithetics or general batching is implemented.
    """
    start = perf_counter()
    if not isinstance(model, BlackScholesModel):
        raise TypeError("price_monte_carlo requires a BlackScholesModel")
    if not isinstance(product, (EuropeanOption, DigitalOption, ArithmeticAsianOption, GeometricAsianOption, UpAndOutCall)):
        raise TypeError("unsupported product")
    maturity, level, _ = _pricing_inputs(model.rate, maturity, confidence_level)
    if isinstance(n_paths, bool) or not isinstance(n_paths, Integral):
        raise TypeError("n_paths must be an integer")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 to estimate sample variance")
    if variance_reduction not in ("none", "antithetic", "control_variate"):
        raise ValueError("variance_reduction must be 'none', 'antithetic' or 'control_variate'")
    if variance_reduction != "control_variate" and n_pilot_paths is not None:
        raise ValueError("n_pilot_paths is only valid with control_variate")
    if variance_reduction == "control_variate":
        if not isinstance(product, ArithmeticAsianOption):
            raise TypeError("geometric control requires an ArithmeticAsianOption target")
        if n_steps is None or n_pilot_paths is None:
            raise ValueError("control_variate requires n_steps and n_pilot_paths")
        n_steps = positive_integer("n_steps", n_steps)
        pilot_count = positive_integer("n_pilot_paths", n_pilot_paths)
        if pilot_count < 2 or n_paths - pilot_count < 2:
            raise ValueError("require at least two pilot and two production paths")
        _, _, discount = _pricing_inputs(model.rate, maturity, level)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            expected = float(np.divide(geometric_asian_price(model, product.strike, maturity,
                                                            n_steps, product.option_type), discount))
        pilot_x, pilot_y = _asian_control_payoffs(model, product, maturity, pilot_count, n_steps, rng)
        fit = fit_control_variate(pilot_x, pilot_y)
        del pilot_x, pilot_y
        target, control = _asian_control_payoffs(model, product, maturity, n_paths - pilot_count, n_steps, rng)
        result = price_from_control_samples(target, control, beta=fit.beta, control_expectation=expected,
                                            rate=model.rate, maturity=maturity, confidence_level=level)
        return replace(result, n_paths=n_paths, n_steps=n_steps, n_pilot_paths=pilot_count,
                       pilot_correlation=fit.correlation, control_fit_status=fit.status,
                       runtime_seconds=perf_counter() - start)
    if variance_reduction == "antithetic":
        if n_paths < 4 or n_paths % 2:
            raise ValueError("antithetic pricing requires an even n_paths >= 4")
        positive, negative = simulate_antithetic_payoffs(
            model, product, maturity, n_paths // 2, rng=rng, n_steps=n_steps,
        )
        result = price_from_antithetic_samples(positive, negative, rate=model.rate,
                                               maturity=maturity, confidence_level=level)
        return replace(result, n_steps=n_steps, runtime_seconds=perf_counter() - start)
    if isinstance(product, (ArithmeticAsianOption, GeometricAsianOption, UpAndOutCall)):
        if n_steps is None:
            raise ValueError("path-dependent pricing requires n_steps for its monitoring grid")
        n_steps = positive_integer("n_steps", n_steps)
        paths = simulate_paths(model, maturity, n_paths, n_steps, rng=rng)
        result = price_from_path_samples(product, paths, rate=model.rate,
                                          maturity=maturity, confidence_level=level)
    else:
        if n_steps is not None:
            raise ValueError("European pricing uses terminal draws; omit n_steps")
        terminal = simulate_terminal(model, maturity, n_paths, rng=rng)
        result = price_from_terminal_samples(product, terminal, rate=model.rate,
                                             maturity=maturity, confidence_level=level)
    return replace(result, runtime_seconds=perf_counter() - start)
