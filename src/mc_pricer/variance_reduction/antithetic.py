"""Antithetic GBM payoffs: negate Gaussian shocks, never asset prices."""

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from mc_pricer._validation import finite_real, finite_vector, positive_integer
from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.products.asian import ArithmeticAsianOption, GeometricAsianOption
from mc_pricer.products.barrier import UpAndOutCall
from mc_pricer.products.european import EuropeanOption
from mc_pricer.products.digital import DigitalOption
from mc_pricer.simulation.paths import paths_from_normals, terminal_from_normals
from mc_pricer.simulation.random import standard_normal_draws, standard_normal_matrix


def pair_means(positive: ArrayLike, negative: ArrayLike) -> NDArray[np.float64]:
    """Average aligned payoff vectors (M,) without mutating either input.

    Row i must pair the SAME Gaussian vector with its negative. Rows, rather
    than the 2*M individual cashflows, must be independent. No distribution or
    independence can be inferred just by inspecting two arrays. Signed values
    are allowed. This helper neither discounts nor estimates uncertainty.
    """
    first = finite_vector("positive", positive)
    second = finite_vector("negative", negative)
    if first.shape != second.shape:
        raise ValueError("paired payoffs must have matching shapes")
    with np.errstate(over="raise", invalid="raise"):
        return 0.5 * first + 0.5 * second


def simulate_antithetic_payoffs(
    model: BlackScholesModel,
    product: EuropeanOption | DigitalOption | ArithmeticAsianOption | GeometricAsianOption | UpAndOutCall,
    maturity: float, n_pairs: int, *, rng: Generator, n_steps: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return aligned undiscounted payoffs (positive, negative), each (M,).

    Draw M IID shocks for Europeans, or a matrix (M,m) for path products.
    Negate EVERY time-step shock to construct the paired path. Drift is unchanged.
    Compute/pay off one branch before the other, so full plus/minus path grids
    do not coexist. The owned normal buffer is negated in place; no supplied
    arrays are modified. RNG advances by M (or M*m) normals even in deterministic
    limits. Grid conventions are inherited from the existing product/simulator.
    At least one pair may be simulated; estimating a sample SE requires two.
    """
    if not isinstance(model, BlackScholesModel):
        raise TypeError("requires a BlackScholesModel")
    if not isinstance(product, (EuropeanOption, DigitalOption, ArithmeticAsianOption, GeometricAsianOption, UpAndOutCall)):
        raise TypeError("unsupported antithetic product")
    maturity = finite_real("maturity", maturity)
    if maturity < 0:
        raise ValueError("maturity must be nonnegative")
    n_pairs = positive_integer("n_pairs", n_pairs)
    if isinstance(product, (EuropeanOption, DigitalOption)):
        if n_steps is not None:
            raise ValueError("European pricing uses terminal draws; omit n_steps")
        normals = standard_normal_draws(n_pairs, rng=rng)
        transform = terminal_from_normals
    else:
        if n_steps is None:
            raise ValueError("path-dependent pricing requires n_steps")
        n_steps = positive_integer("n_steps", n_steps)
        normals = standard_normal_matrix(n_pairs, n_steps, rng=rng)
        transform = paths_from_normals
    positive = product.payoff(transform(model, maturity, normals))
    normals *= -1.0
    negative = product.payoff(transform(model, maturity, normals))
    return positive, negative
