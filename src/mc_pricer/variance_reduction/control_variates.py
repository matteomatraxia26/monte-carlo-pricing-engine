"""Pilot coefficient estimation and fixed-coefficient control adjustments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mc_pricer._validation import finite_real, finite_vector


@dataclass(frozen=True)
class ControlVariateFit:
    """Pilot diagnostics. Beta=0 safely falls back when the control is constant."""

    beta: float
    n_pilot_samples: int
    correlation: float | None
    control_sample_variance: float
    status: str


def fit_control_variate(target: ArrayLike, control: ArrayLike) -> ControlVariateFit:
    """Fit covariance(X,Y)/variance(Y) on aligned pilot vectors (P,), P>=2.

    Use the same discount convention for X and Y. Common discounting leaves
    beta/correlation unchanged. Apply this fit ONLY to an independent production
    sample for the exact conditional-unbiasedness argument. Arrays are unchanged.
    A zero pilot control variance gives beta=0, not a division by zero or a
    guarantee that the population control variance is zero.
    """
    x = finite_vector("pilot_target", target, min_size=2)
    y = finite_vector("pilot_control", control, min_size=2)
    if x.shape != y.shape:
        raise ValueError("pilot target and control must have matching shapes")
    if np.all(y == y[0]):
        return ControlVariateFit(0.0, y.size, None, 0.0, "zero_control_variance")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        yc = y - y.mean()
        yy = float(yc @ yc)
        if yy == 0:
            return ControlVariateFit(0.0, y.size, None, 0.0, "zero_control_variance")
        xc = np.zeros_like(x) if np.all(x == x[0]) else x - x.mean()
        xx, xy = float(xc @ xc), float(xc @ yc)
        beta = xy / yy
        correlation = float(xy / np.sqrt(xx) / np.sqrt(yy)) if xx > 0 else None
    if not np.isfinite(beta):
        raise FloatingPointError("control coefficient exceeds finite float64 range")
    return ControlVariateFit(beta, y.size, correlation, yy / (y.size - 1), "fitted")


def apply_control_variate(
    target: ArrayLike, control: ArrayLike, *, control_expectation: float, beta: float,
) -> NDArray[np.float64]:
    """Return X-beta*(Y-EY) for aligned vectors (N,), using one unit convention.

    Samples may be signed. This helper neither discounts nor fits beta. Beta
    must be fixed or estimated independently of these production samples for
    exact unbiasedness. A known expectation must match the control's contract.
    """
    x, y = finite_vector("target", target), finite_vector("control", control)
    if x.shape != y.shape:
        raise ValueError("target and control must have matching shapes")
    expected = finite_real("control_expectation", control_expectation)
    beta = finite_real("beta", beta)
    with np.errstate(over="raise", invalid="raise"):
        return x - beta * (y - expected)
