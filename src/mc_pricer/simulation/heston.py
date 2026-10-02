"""Full truncation variance Euler with left-point log-stock Euler.

One vectorized operation per time step; no loop over paths. The raw variance
state is carried through negative values. Clipping the state itself would be
an absorption scheme, not the implemented full truncation scheme.
"""
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from mc_pricer._validation import finite_matrix, finite_real, finite_vector, positive_integer
from mc_pricer.models.heston import HestonModel


@dataclass(frozen=True)
class HestonPaths:
    """Observed states with shape (N, n_observations+1), including time zero.

    raw_variance_paths may be negative; effective variance is its positive part.
    integrated_variance contains each path's left-grid sum of effective variance,
    shape (N,), including unobserved integration steps. Array values are mutable
    even though dataclass attributes cannot be rebound. Negative fraction counts
    all N*n_steps raw updates, not just the saved dates.
    """
    stock_paths: NDArray[np.float64]
    raw_variance_paths: NDArray[np.float64]
    times: NDArray[np.float64]
    integrated_variance: NDArray[np.float64]
    n_steps: int
    negative_variance_fraction: float

    @property
    def variance_paths(self) -> NDArray[np.float64]:
        """New array of nonnegative effective variance at the stored dates."""
        return np.maximum(self.raw_variance_paths, 0.)


def correlate_normals(stock_normals: ArrayLike, independent_normals: ArrayLike,
                      rho: float) -> NDArray[np.float64]:
    """Return rho*Zs+sqrt(1-rho²)*Zi for equal finite vectors or matrices.

    Input normals must be independent standard Gaussians for this to have the
    claimed covariance; array validation cannot establish distributional facts.
    Endpoints rho=±1 are supported. Inputs are never modified.
    """
    rho = finite_real('rho', rho)
    if not -1 <= rho <= 1:
        raise ValueError('rho must lie in [-1,1]')
    validate = finite_vector if np.asarray(stock_normals).ndim == 1 else finite_matrix
    zs, zi = validate('stock_normals', stock_normals), validate('independent_normals', independent_normals)
    if zs.shape != zi.shape:
        raise ValueError('normal arrays must have identical shapes')
    with np.errstate(over='raise', invalid='raise'):
        return rho*zs+np.sqrt((1-rho)*(1+rho))*zi


def _configuration(model: HestonModel, maturity: float, n_paths: int,
                   n_steps: int, n_observations: int | None) -> tuple[float, int, int, int]:
    if not isinstance(model, HestonModel):
        raise TypeError('requires a HestonModel')
    time = finite_real('maturity', maturity)
    if time < 0:
        raise ValueError('maturity must be nonnegative')
    count = positive_integer('n_paths', n_paths)
    steps = positive_integer('n_steps', n_steps)
    observations = steps if n_observations is None else positive_integer('n_observations', n_observations)
    if steps % observations:
        raise ValueError('n_steps must be divisible by n_observations')
    return time, count, steps, observations


def _evolve(model: HestonModel, time: float, count: int, steps: int, observations: int,
            draw_step: Callable[[int], tuple[np.ndarray, np.ndarray]]) -> HestonPaths:
    """Shared kernel for supplied shocks and memory-aware generated shocks."""
    stocks = np.empty((count, observations+1))
    variances = np.empty_like(stocks)
    stock = np.full(count, model.spot)
    raw = np.full(count, model.v0)
    integrated = np.zeros(count)
    stocks[:,0], variances[:,0] = stock, raw
    dt = time/steps
    stride = steps//observations
    orthogonal = np.sqrt((1-model.rho)*(1+model.rho))
    negative = 0
    for j in range(steps):
        zs, zi = draw_step(j)
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            positive = np.maximum(raw, 0.)
            variance_dt = positive*dt
            diffusion = np.sqrt(variance_dt)
            # Both updates use the OLD variance and the same correlated shocks.
            stock = stock*np.exp(model.rate*dt-.5*variance_dt+diffusion*zs)
            raw = raw+model.kappa*(model.theta-positive)*dt+model.xi*diffusion*(model.rho*zs+orthogonal*zi)
            integrated += variance_dt
        if not np.isfinite(stock).all() or np.any(stock<=0) or not np.isfinite(raw).all() or not np.isfinite(integrated).all():
            raise FloatingPointError('Heston state exceeds finite float64 range')
        negative += int(np.count_nonzero(raw<0))
        if (j+1)%stride == 0:
            column = (j+1)//stride
            stocks[:,column], variances[:,column] = stock, raw
    return HestonPaths(stocks, variances, np.linspace(0.,time,observations+1),
                       integrated, steps, negative/(count*steps))


def heston_paths_from_normals(model: HestonModel, maturity: float,
                              stock_normals: ArrayLike, independent_normals: ArrayLike,
                              *, n_observations: int | None = None) -> HestonPaths:
    """Use two IID independent normal matrices (N, n_steps), without mutation.

    The second matrix is INDEPENDENT of the stock shocks, not already correlated.
    Fine-grid shocks may be summed and divided by sqrt(block size) to couple
    coarser integration grids. n_observations evenly spaced dates exclude t=0;
    t=0 is then prepended in output. Default stores the full integration grid.
    """
    zs = finite_matrix('stock_normals', stock_normals)
    zi = finite_matrix('independent_normals', independent_normals)
    if zs.shape != zi.shape:
        raise ValueError('normal matrices must have matching shapes')
    config = _configuration(model, maturity, *zs.shape, n_observations)
    return _evolve(model, *config, lambda j: (zs[:,j], zi[:,j]))


def simulate_heston_paths(model: HestonModel, maturity: float, n_paths: int,
                           n_steps: int, *, rng: Generator,
                           n_observations: int | None = None) -> HestonPaths:
    """Generate shocks one time step at a time, storing only selected dates.

    Draw order: for each time step, rng.standard_normal((N,2)); first column
    stock, second independent variance driver. Reproducibility depends on N and
    draw order. Consume 2*N*n_steps draws even at T=0 or xi=0. Output O(N*m_obs),
    working state/shocks O(N). Set n_observations=1 for O(N) terminal storage.
    Invalid configurations fail before draws; numerical failures may advance RNG.
    """
    config = _configuration(model, maturity, n_paths, n_steps, n_observations)
    if not isinstance(rng, Generator):
        raise TypeError('rng must be a numpy.random.Generator')
    def draw_step(_: int) -> tuple[np.ndarray, np.ndarray]:
        shocks = rng.standard_normal((config[1],2))
        return shocks[:,0], shocks[:,1]
    return _evolve(model, *config, draw_step)
