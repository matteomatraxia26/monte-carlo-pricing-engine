"""European formulas, terminal-monitored barrier and exact average moments.

All inputs are scalars. Calls and puts have European exercise and no dividends.
These formulas are research benchmarks, not a specialized extreme-tail engine.
"""

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Literal

import numpy as np
from scipy.stats import norm

from mc_pricer._validation import finite_real, positive_integer
from mc_pricer.models.black_scholes import BlackScholesModel

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BlackScholesGreeks:
    """Spot Delta, spot Gamma, and Vega per unit absolute volatility.

    Vega / 100 is sensitivity per one percentage point of volatility.
    Rates, strike and time to maturity are held fixed.
    """

    delta: float
    gamma: float
    vega: float


def _validate_contract(
    strike: float, maturity: float, option_type: OptionType
) -> tuple[float, float]:
    strike = finite_real("strike", strike)
    maturity = finite_real("maturity", maturity)
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be nonnegative")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    return strike, maturity


def _d1_d2(
    model: BlackScholesModel, strike: float, maturity: float
) -> tuple[float, float]:
    """Compute d1 and d2 after validating T > 0 and sigma > 0."""
    total_volatility = model.volatility * sqrt(maturity)
    # Separate logarithms avoid overflow in the ratio spot / strike.
    log_moneyness = log(model.spot) - log(strike)
    d1 = (log_moneyness + model.rate * maturity) / total_volatility
    d1 += 0.5 * total_volatility
    return d1, d1 - total_volatility


def black_scholes_price(
    model: BlackScholesModel,
    strike: float,
    maturity: float,
    option_type: OptionType = "call",
) -> float:
    """Return a European option's present value in spot currency units.

    Expiry returns intrinsic value. At zero volatility, the discounted
    deterministic payoff is used explicitly to avoid division by zero.
    Extreme parameters can overflow float64; see docs/methodology.md.
    """
    strike, maturity = _validate_contract(strike, maturity, option_type)
    sign = 1.0 if option_type == "call" else -1.0
    if maturity == 0:
        return max(sign * (model.spot - strike), 0.0)

    discounted_strike = strike * exp(-model.rate * maturity)
    if model.volatility == 0:
        return max(sign * (model.spot - discounted_strike), 0.0)

    d1, d2 = _d1_d2(model, strike, maturity)
    if option_type == "call":
        return float(model.spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    # Evaluate Phi(-d) directly; do not subtract a near-one CDF from one.
    return float(discounted_strike * norm.cdf(-d2) - model.spot * norm.cdf(-d1))


def black_scholes_greeks(
    model: BlackScholesModel,
    strike: float,
    maturity: float,
    option_type: OptionType = "call",
) -> BlackScholesGreeks:
    """Return analytical Delta, Gamma and Vega for T > 0 and sigma > 0.

    Boundary Greeks are deliberately outside this API: at expiry or zero
    volatility there are kinks and/or one-sided derivative conventions.
    The price function still supports both boundaries.
    """
    strike, maturity = _validate_contract(strike, maturity, option_type)
    if maturity == 0 or model.volatility == 0:
        raise ValueError("Greeks require positive maturity and volatility")
    d1, _ = _d1_d2(model, strike, maturity)
    root_time = sqrt(maturity)
    density = float(norm.pdf(d1))
    delta = float(norm.cdf(d1)) if option_type == "call" else -float(norm.cdf(-d1))
    return BlackScholesGreeks(
        delta=delta,
        gamma=density / (model.spot * model.volatility * root_time),
        vega=model.spot * density * root_time,
    )


@dataclass(frozen=True)
class ArithmeticAverageMoments:
    """Exact mean and variance of the undiscounted discrete arithmetic average."""

    mean: float
    variance: float


def arithmetic_average_moments(
    model: BlackScholesModel, maturity: float, n_observations: int,
) -> ArithmeticAverageMoments:
    """GBM moments on uniform dates j*T/m, j=1,...,m; S0 is excluded.

    This is not an Asian option pricing formula. Uses stock covariance
    S0^2 exp(r*(s+t)) * expm1(sigma^2 * min(s,t)) and prefix sums to evaluate
    the covariance double sum in O(m) time/storage rather than O(m^2).
    """
    if not isinstance(model, BlackScholesModel):
        raise TypeError("arithmetic_average_moments requires a BlackScholesModel")
    maturity = finite_real("maturity", maturity)
    if maturity < 0:
        raise ValueError("maturity must be nonnegative")
    count = positive_integer("n_observations", n_observations)
    if maturity == 0:
        return ArithmeticAverageMoments(model.spot, 0.0)
    times = maturity * (np.arange(1, count + 1, dtype=np.float64) / count)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        growth = np.exp(model.rate * times)
        mean = float(model.spot * growth.mean())
        if model.volatility == 0:
            variance = 0.0
        else:
            variance_factors = np.expm1(model.volatility * model.volatility * times)
            prefix = np.cumsum(growth * variance_factors)
            earlier = np.concatenate(([0.0], prefix[:-1]))
            covariance_sum = np.sum(growth * growth * variance_factors) + 2.0 * (growth @ earlier)
            variance = float((model.spot / count) ** 2 * covariance_sum)
    if not isfinite(mean) or not isfinite(variance):
        raise FloatingPointError("arithmetic-average moments exceed finite float64 range")
    return ArithmeticAverageMoments(mean, variance)


def terminal_monitored_up_and_out_price(
    model: BlackScholesModel, strike: float, barrier: float, maturity: float,
) -> float:
    """Exact zero-rebate call monitored ONLY at inception and maturity.

    This benchmarks a one-step discrete barrier, NOT a continuous barrier.
    For spot < barrier and strike < barrier, integrate (ST-K) over K<ST<B.
    Equality knocks out. Handles expiry/zero volatility explicitly. Ordinary
    float64 CDF differences can lose relative accuracy for very narrow bands
    or extreme tails; no extreme-parameter stability guarantee or clipping.
    """
    if not isinstance(model, BlackScholesModel):
        raise TypeError("requires a BlackScholesModel")
    strike, maturity = _validate_contract(strike, maturity, "call")
    barrier = finite_real("barrier", barrier)
    if barrier <= 0:
        raise ValueError("barrier must be positive")
    if model.spot >= barrier or strike >= barrier:
        return 0.0
    if maturity == 0:
        return max(model.spot - strike, 0.0)
    if model.volatility == 0:
        terminal = model.spot * exp(model.rate * maturity)
        return black_scholes_price(model, strike, maturity) if terminal < barrier else 0.0
    d1_k, d2_k = _d1_d2(model, strike, maturity)
    d1_b, d2_b = _d1_d2(model, barrier, maturity)
    return float(model.spot * (norm.cdf(d1_k) - norm.cdf(d1_b))
                 - strike * exp(-model.rate * maturity) * (norm.cdf(d2_k) - norm.cdf(d2_b)))


@dataclass(frozen=True)
class GeometricAverageLogMoments:
    """Mean and variance of log(G) on uniform future monitoring dates."""

    log_mean: float
    log_variance: float


def geometric_average_log_moments(
    model: BlackScholesModel, maturity: float, n_observations: int,
) -> GeometricAverageLogMoments:
    """Exact log(G) moments for j*T/m, j=1,...,m; S0 excluded.

    Mean log(G)=log(S0)+(r-sigma²/2)*T*(m+1)/(2m).
    Variance log(G)=sigma²*T*(m+1)*(2m+1)/(6m²).
    This discrete formula must not be replaced by a continuous-average limit.
    """
    if not isinstance(model, BlackScholesModel):
        raise TypeError("requires a BlackScholesModel")
    maturity = finite_real("maturity", maturity)
    if maturity < 0:
        raise ValueError("maturity must be nonnegative")
    count = positive_integer("n_observations", n_observations)
    average_time = maturity * (1 + 1 / count) / 2
    mean = log(model.spot) + (model.rate - .5 * model.volatility**2) * average_time
    variance = model.volatility**2 * maturity * (1 + 1 / count) * (2 + 1 / count) / 6
    if not isfinite(mean) or not isfinite(variance):
        raise FloatingPointError("geometric log moments exceed finite float64 range")
    return GeometricAverageLogMoments(mean, variance)


def geometric_asian_price(
    model: BlackScholesModel, strike: float, maturity: float,
    n_observations: int, option_type: OptionType = "call",
) -> float:
    """Exact discrete geometric Asian price; equal weights, exclude S0, pay at T.

    Supports call/put, expiry and zero volatility. Ordinary float64 CDF formulas
    have the same extreme-tail limitations as the European benchmark.
    """
    strike, maturity = _validate_contract(strike, maturity, option_type)
    moments = geometric_average_log_moments(model, maturity, n_observations)
    if n_observations == 1 or maturity == 0:
        return black_scholes_price(model, strike, maturity, option_type)
    sign = 1.0 if option_type == "call" else -1.0
    discount = exp(-model.rate * maturity)
    if moments.log_variance == 0:
        return discount * max(sign * (exp(moments.log_mean) - strike), 0.0)
    root_variance = sqrt(moments.log_variance)
    d2 = (moments.log_mean - log(strike)) / root_variance
    d1 = d2 + root_variance
    mean_average = exp(moments.log_mean + .5 * moments.log_variance)
    return float(discount * sign * (mean_average * norm.cdf(sign * d1) - strike * norm.cdf(sign * d2)))
