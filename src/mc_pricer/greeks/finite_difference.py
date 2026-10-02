"""Central finite-difference Greeks from paired, path-level payoff differences.

Common random numbers are the default. An independent-leg mode supplies a
comparison baseline. No antithetic/control corrections or pathwise Greeks here.
"""

from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from mc_pricer._validation import finite_matrix, finite_real, finite_vector, positive_integer
from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.products.european import EuropeanOption
from mc_pricer.products.digital import DigitalOption
from mc_pricer.products.asian import ArithmeticAsianOption, GeometricAsianOption
from mc_pricer.products.barrier import UpAndOutCall
from mc_pricer.pricing.monte_carlo import price_from_payoff_samples
from mc_pricer.simulation.paths import paths_from_normals, terminal_from_normals
from mc_pricer.simulation.random import standard_normal_draws, standard_normal_matrix
from mc_pricer.statistics.confidence_intervals import validate_confidence_level

Greek = Literal["delta", "gamma", "vega"]
Product = EuropeanOption | DigitalOption | ArithmeticAsianOption | GeometricAsianOption | UpAndOutCall


@dataclass(frozen=True)
class FiniteDifferenceResult:
    """Estimate and sampling uncertainty for a *finite-bump* sensitivity.

    CI targets the central price difference, not necessarily the exact Greek:
    truncation bias, monitoring error and floating-point errors are excluded.
    n_paths counts independent coupled rows, not all bumped path valuations.
    sample_variance is the discounted difference-sample variance (Greek units²).
    Vega is per unit absolute volatility, not per percentage point.
    """

    greek: Greek
    estimate: float
    standard_error: float
    sample_variance: float
    confidence_interval: tuple[float, float]
    bump: float
    n_paths: int
    n_payoff_evaluations: int
    common_random_numbers: bool
    runtime_seconds: float
    n_steps: int | None = None
    confidence_level: float = .95

    @property
    def estimator_variance(self) -> float:
        """Estimated sampling variance of the finite-difference mean."""
        return self.sample_variance / self.n_paths


def _bump_denominator(bump: float, order: int) -> tuple[float, float]:
    h = finite_real("bump", bump)
    if h <= 0:
        raise ValueError("bump must be positive")
    denominator = 2*h if order == 1 else h*h
    if not isfinite(denominator) or denominator == 0:
        raise FloatingPointError("finite-difference denominator is not positive finite float64")
    return h, denominator


def central_first_difference(up: ArrayLike, down: ArrayLike, bump: float) -> NDArray[np.float64]:
    """Return (up-down)/(2h) for aligned finite payoff vectors (N,).

    No discounting or independence assumptions. Inputs are not modified.
    """
    _, denominator = _bump_denominator(bump, 1)
    upper, lower = finite_vector("up", up), finite_vector("down", down)
    if upper.shape != lower.shape:
        raise ValueError("payoff vectors must have matching shapes")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return (upper-lower)/denominator


def central_second_difference(up: ArrayLike, base: ArrayLike, down: ArrayLike,
                              bump: float) -> NDArray[np.float64]:
    """Return [(up-base)+(down-base)]/h² for aligned payoff vectors (N,).

    Subtract each nearby value before summing, avoiding the intermediate 2*base.
    Floating-point cancellation remains possible for tiny bumps.
    """
    _, denominator = _bump_denominator(bump, 2)
    upper, middle, lower = (finite_vector(name, values) for name, values in
                            (("up", up), ("base", base), ("down", down)))
    if upper.shape != middle.shape or upper.shape != lower.shape:
        raise ValueError("payoff vectors must have matching shapes")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return ((upper-middle)+(lower-middle))/denominator


def _validate_inputs(model: BlackScholesModel, product: Product, maturity: float,
                     greek: Greek, bump: float, confidence_level: float
                     ) -> tuple[float, float, float, BlackScholesModel, BlackScholesModel]:
    if not isinstance(model, BlackScholesModel):
        raise TypeError("finite differences require a BlackScholesModel")
    if not isinstance(product, (EuropeanOption, DigitalOption, ArithmeticAsianOption, GeometricAsianOption, UpAndOutCall)):
        raise TypeError("unsupported product")
    if greek not in ("delta", "gamma", "vega"):
        raise ValueError("greek must be delta, gamma or vega")
    time = finite_real("maturity", maturity)
    if time <= 0 or model.volatility <= 0:
        raise ValueError("Greeks require positive maturity and volatility")
    level = validate_confidence_level(confidence_level)
    h, _ = _bump_denominator(bump, 2 if greek == "gamma" else 1)
    parameter = "volatility" if greek == "vega" else "spot"
    center = getattr(model, parameter)
    if h >= center:
        raise ValueError(f"central bump must be smaller than {parameter}")
    if center+h == center or center-h == center:
        raise ValueError("bump is too small to change the float64 model parameter")
    upper = replace(model, **{parameter: center+h})
    lower = replace(model, **{parameter: center-h})
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        discount = float(np.exp(-model.rate*time))
    if not np.isfinite(discount) or discount <= 0:
        raise FloatingPointError("discount factor exceeds positive finite float64 range")
    return time, h, level, upper, lower


def _payoffs(model: BlackScholesModel, product: Product, maturity: float,
             normals: NDArray[np.float64]) -> NDArray[np.float64]:
    # The full path matrix is temporary: only the payoff vector survives.
    states = (terminal_from_normals(model, maturity, normals) if isinstance(product, (EuropeanOption, DigitalOption))
              else paths_from_normals(model, maturity, normals))
    return product.payoff(states)


def finite_difference_from_normals(
    model: BlackScholesModel, product: Product, maturity: float, normals: ArrayLike, *,
    greek: Greek, bump: float, down_normals: ArrayLike | None = None,
    base_normals: ArrayLike | None = None, confidence_level: float = .95,
) -> FiniteDifferenceResult:
    """Use supplied shocks for one central Greek; runtime excludes RNG draws.

    Vanilla/digital European shocks: (N,); Asian/barrier shocks: (N,m). Require N>=2.
    Default: every leg reuses `normals`, giving common random numbers (CRN).
    Independent-leg mode: supply `down_normals`, and `base_normals` for Gamma.
    All shapes must match. Supplying extra arrays labels independent-leg mode;
    their actual independence and IID rows are the caller's responsibility.
    Antithetic rows require a different summary and must not be passed as IID.
    """
    start = perf_counter()
    time, h, level, upper, lower = _validate_inputs(model, product, maturity, greek, bump, confidence_level)
    shared = down_normals is None
    if (shared and base_normals is not None) or (greek != "gamma" and base_normals is not None):
        raise ValueError("base shocks are used only with explicit independent Gamma legs")
    if not shared and greek == "gamma" and base_normals is None:
        raise ValueError("independent Gamma requires both down_normals and base_normals")
    validate = finite_vector if isinstance(product, (EuropeanOption, DigitalOption)) else finite_matrix
    z = validate("normals", normals)
    if z.shape[0] < 2:
        raise ValueError("at least two independent shock rows are required")
    zd = z if shared else validate("down_normals", down_normals)
    zb = z if shared or greek != "gamma" else validate("base_normals", base_normals)
    if z.shape != zd.shape or z.shape != zb.shape:
        raise ValueError("all shock arrays must have matching shapes")
    up_payoffs = _payoffs(upper, product, time, z)
    down_payoffs = _payoffs(lower, product, time, zd)
    if greek == "gamma":
        base_payoffs = _payoffs(model, product, time, zb)
        differences = central_second_difference(up_payoffs, base_payoffs, down_payoffs, h)
    else:
        differences = central_first_difference(up_payoffs, down_payoffs, h)
    # Reuse the single discount/variance/CI implementation; differences can be signed.
    stats = price_from_payoff_samples(differences, rate=model.rate, maturity=time, confidence_level=level)
    return FiniteDifferenceResult(
        greek=greek, estimate=stats.price, standard_error=stats.standard_error,
        sample_variance=stats.sample_variance, confidence_interval=stats.confidence_interval,
        bump=h, n_paths=z.shape[0], n_payoff_evaluations=z.shape[0]*(3 if greek == "gamma" else 2),
        common_random_numbers=shared, runtime_seconds=perf_counter()-start,
        n_steps=None if z.ndim == 1 else z.shape[1], confidence_level=level,
    )


def finite_difference_greek(
    model: BlackScholesModel, product: Product, maturity: float, n_paths: int, *,
    greek: Greek, bump: float, rng: Generator, n_steps: int | None = None,
    common_random_numbers: bool = True, confidence_level: float = .95,
) -> FiniteDifferenceResult:
    """Simulate and estimate a central Greek, with CRN by default.

    `n_paths` is N independent shock tuples; each needs 2 payoff evaluations
    for Delta/Vega or 3 for Gamma. CRN uses N normals (N*m for paths);
    independent legs use 2N or 3N normals (times m). Draw order: up, down,
    then base for Gamma. With CRN draw once and share all shocks unchanged.
    Validate ordinary input errors before RNG consumption. State-transform
    overflow discovered only during simulation can still advance the RNG.
    """
    start = perf_counter()
    _validate_inputs(model, product, maturity, greek, bump, confidence_level)
    count = positive_integer("n_paths", n_paths)
    if count < 2:
        raise ValueError("at least two independent shock rows are required")
    if not isinstance(common_random_numbers, bool):
        raise TypeError("common_random_numbers must be a bool")
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if isinstance(product, (EuropeanOption, DigitalOption)):
        if n_steps is not None:
            raise ValueError("European Greeks use terminal shocks; omit n_steps")
        def draw() -> NDArray[np.float64]:
            return standard_normal_draws(count, rng=rng)
    else:
        if n_steps is None:
            raise ValueError("path-dependent Greeks require n_steps")
        steps = positive_integer("n_steps", n_steps)
        def draw() -> NDArray[np.float64]:
            return standard_normal_matrix(count, steps, rng=rng)
    z = draw()
    zd = None if common_random_numbers else draw()
    zb = draw() if not common_random_numbers and greek == "gamma" else None
    result = finite_difference_from_normals(model, product, maturity, z, greek=greek, bump=bump,
                                           down_normals=zd, base_normals=zb, confidence_level=confidence_level)
    return replace(result, runtime_seconds=perf_counter()-start)
