"""Exact Black-Scholes simulation at terminal or uniformly spaced grid dates."""

from math import isfinite, sqrt

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from mc_pricer._validation import finite_matrix, finite_real, finite_vector
from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.simulation.random import standard_normal_draws, standard_normal_matrix


def _validate_maturity(maturity: float) -> float:
    maturity = finite_real("maturity", maturity)
    if maturity < 0:
        raise ValueError("maturity must be nonnegative")
    return maturity


def _log_increment_coefficients(model: BlackScholesModel, time_interval: float) -> tuple[float, float]:
    """Shared mean and standard deviation of an exact GBM log increment."""
    drift = (model.rate - 0.5 * model.volatility * model.volatility) * time_interval
    diffusion = model.volatility * sqrt(time_interval)
    if not isfinite(drift) or not isfinite(diffusion):
        raise FloatingPointError("GBM coefficients exceed float64 range")
    return drift, diffusion


def terminal_from_normals(
    model: BlackScholesModel, maturity: float, normals: ArrayLike
) -> NDArray[np.float64]:
    """Map supplied Gaussian shocks to terminal prices without randomness.

    Parameters
    ----------
    normals
        Nonempty, finite real array-like of shape ``(n_paths,)``. Numeric lists
        and real NumPy arrays are accepted; no empirical normality test is
        performed. IID standard-normal sampling is the caller's responsibility.

    Returns
    -------
    numpy.ndarray
        Newly allocated float64 terminal prices of shape ``(n_paths,)``.
        Inputs are never modified, including noncontiguous/read-only views.

    Notes
    -----
    Formula: S_T = S_0 exp((r - sigma^2/2) T + sigma sqrt(T) Z).
    At T=0 return spot exactly. At sigma=0 the outcome is deterministic.
    Floating-point overflow or underflow to a zero price raises an error
    instead of silently violating positivity/finite-price assumptions.
    """
    maturity = _validate_maturity(maturity)
    values = finite_vector("normals", normals)
    if maturity == 0:
        return np.full(values.shape, model.spot, dtype=np.float64)

    drift, diffusion = _log_increment_coefficients(model, maturity)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        terminal = model.spot * np.exp(drift + diffusion * values)
    if not np.all(np.isfinite(terminal)) or np.any(terminal <= 0):
        raise FloatingPointError("GBM terminal prices exceed positive finite float64 range")
    return terminal


def simulate_terminal(
    model: BlackScholesModel,
    maturity: float,
    n_paths: int,
    *,
    rng: Generator,
) -> NDArray[np.float64]:
    """Draw normals and return exact GBM terminal prices of shape ``(n_paths,)``.

    No time grid is needed. All paths are transformed with vectorized NumPy
    operations. Time and storage complexity are O(n_paths).
    An existing Generator is mandatory; use make_rng(seed) to create one.

    Consume one normal draw per path even at T=0 or sigma=0. This explicit
    convention makes stream advancement independent of deterministic limits.
    Invalid maturity/count/RNG arguments fail before any draws are consumed.
    """
    maturity = _validate_maturity(maturity)
    normals = standard_normal_draws(n_paths, rng=rng)
    return terminal_from_normals(model, maturity, normals)


def paths_from_normals(
    model: BlackScholesModel, maturity: float, normals: ArrayLike,
) -> NDArray[np.float64]:
    """Map shocks of shape (n_paths, n_steps) to exact uniform-grid GBM paths.

    Return a new C-contiguous float64 array of shape (n_paths, n_steps + 1).
    Column j represents time j * maturity / n_steps; column 0 equals spot.
    Both shock dimensions must be positive. Supplying IID N(0,1) draws is the
    caller's responsibility. Input arrays are never modified or aliased by output.

    Uses cumulative sums along axis=1, with no Python loop over paths or times.
    The newly allocated output is also working storage for log increments.
    Complexity: O(n_paths * n_steps) arithmetic and storage. The input shocks
    and output coexist; validation and non-float64 conversion can allocate more.

    At zero maturity every column equals spot, after validating supplied shocks.
    At zero volatility each path follows the deterministic risk-neutral growth.
    Grid transitions are exact under GBM, up to floating-point arithmetic. They
    do not reveal continuous inter-date barrier crossings or a continuous average.
    Overflow, underflow to zero prices, or a positive dt rounding to zero raises.
    """
    if not isinstance(model, BlackScholesModel):
        raise TypeError("paths_from_normals requires a BlackScholesModel")
    maturity = _validate_maturity(maturity)
    values = finite_matrix("normals", normals)
    n_paths, n_steps = values.shape
    if maturity == 0:
        return np.full((n_paths, n_steps + 1), model.spot, dtype=np.float64)
    dt = maturity / n_steps
    if dt == 0:
        raise FloatingPointError("positive GBM time step underflowed to zero")
    drift, diffusion = _log_increment_coefficients(model, dt)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = model.spot
    # This slice is a view of our new output, never a view of caller shocks.
    log_paths = paths[:, 1:]
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        np.multiply(values, diffusion, out=log_paths)
        log_paths += drift
        np.cumsum(log_paths, axis=1, out=log_paths)
        np.exp(log_paths, out=log_paths)
        log_paths *= model.spot
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise FloatingPointError("GBM path prices exceed positive finite float64 range")
    return paths


def simulate_paths(
    model: BlackScholesModel, maturity: float, n_paths: int, n_steps: int, *, rng: Generator,
) -> NDArray[np.float64]:
    """Draw shocks and return exact uniform-grid paths of shape (N, m + 1).

    Requires n_paths >= 1 and n_steps >= 1. Consume N*m normals even at zero
    maturity or volatility. Invalid model/maturity/count/RNG arguments fail before
    sampling. Numerical range failures during transformation can advance the RNG.
    For European payoffs, simulate_terminal avoids the O(N*m) grid allocation.
    """
    if not isinstance(model, BlackScholesModel):
        raise TypeError("simulate_paths requires a BlackScholesModel")
    maturity = _validate_maturity(maturity)
    normals = standard_normal_matrix(n_paths, n_steps, rng=rng)
    return paths_from_normals(model, maturity, normals)
