"""Transparent convergence fits and summaries of independent replications."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mc_pricer._validation import finite_real, finite_vector
from mc_pricer.statistics.confidence_intervals import normal_confidence_interval


@dataclass(frozen=True)
class PowerLawFit:
    """Descriptive OLS fit log(value) = intercept + slope * log(size).

    r_squared is None for constant log values. No classical OLS standard errors
    are returned: noisy SE estimates need not have homoskedastic residuals.
    """

    intercept: float
    slope: float
    r_squared: float | None
    n_points: int


def fit_power_law(sample_sizes: ArrayLike, values: ArrayLike) -> PowerLawFit:
    """Fit positive 1-D arrays of identical shape (n_points,), n_points >= 3.

    Sizes must be strictly increasing. Arrays are not modified. This computes
    a descriptive exponent; it does not establish a convergence theorem.
    """
    sizes = finite_vector("sample_sizes", sample_sizes, min_size=3)
    measurements = finite_vector("values", values, min_size=3)
    if sizes.shape != measurements.shape:
        raise ValueError("sample_sizes and values must have the same shape")
    if np.any(sizes <= 0) or np.any(np.diff(sizes) <= 0) or np.any(measurements <= 0):
        raise ValueError("sizes must be positive/increasing and values positive")
    x, y = np.log(sizes), np.log(measurements)
    centered_x, centered_y = x - x.mean(), y - y.mean()
    denominator = float(centered_x @ centered_x)
    if denominator == 0:
        raise ValueError("sample sizes are indistinguishable after taking logs")
    if np.all(measurements == measurements[0]):
        return PowerLawFit(float(y[0]), 0.0, None, sizes.size)
    slope = float(centered_x @ centered_y) / denominator
    intercept = float(y.mean() - slope * x.mean())
    residual = centered_y - slope * centered_x
    total = float(centered_y @ centered_y)
    r_squared = None if total == 0 else 1.0 - float(residual @ residual) / total
    return PowerLawFit(intercept, slope, r_squared, sizes.size)


@dataclass(frozen=True)
class ReplicationSummary:
    """Statistics across IID scalar estimates, not across underlying payoffs.

    empirical_variance uses R-1; rmse uses R. The normal bias interval is an
    approximate repeated-run interval for E[estimate] - target. At least two
    replications are required, but reliable normal coverage needs more.
    """

    n_replications: int
    mean_estimate: float
    empirical_bias: float
    empirical_variance: float
    rmse: float
    standard_error_of_mean: float
    bias_confidence_interval: tuple[float, float]


def summarize_replications(
    estimates: ArrayLike, target: float, confidence_level: float = 0.95,
) -> ReplicationSummary:
    """Summarize a real finite vector of independent estimates, shape (R,).

    The caller must ensure independence and a common target/configuration.
    Neither a vector of nested estimates nor different sample sizes form IID
    replications. Input arrays are not modified.
    """
    values = finite_vector("estimates", estimates, min_size=2)
    target = finite_real("target", target)
    with np.errstate(over="raise", invalid="raise"):
        constant = np.all(values == values[0])
        mean = float(values[0] if constant else values.mean())
        variance = 0.0 if constant else float(values.var(ddof=1))
        bias = mean - target
        rmse = float(np.sqrt(np.mean((values - target) ** 2)))
        mean_se = float(np.sqrt(variance / values.size))
    interval = normal_confidence_interval(bias, mean_se, confidence_level)
    return ReplicationSummary(values.size, mean, bias, variance, rmse, mean_se, interval)
