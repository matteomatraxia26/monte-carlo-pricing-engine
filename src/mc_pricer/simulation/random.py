"""Small explicit RNG helpers; no global random state is used."""

from numbers import Integral

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mc_pricer._validation import positive_integer


def make_rng(seed: int | None = None) -> Generator:
    """Construct a Generator with NumPy's default bit generator.

    Use a nonnegative integer for reproducible experiments. ``None`` requests
    OS entropy and is intentionally non-reproducible. A fresh generator with
    the same seed restarts a stream; an existing generator advances its state.
    Record NumPy version, bit generator and draw order as well as the seed.
    """
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer or None")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        seed = int(seed)
    return np.random.default_rng(seed)


def standard_normal_draws(n_paths: int, *, rng: Generator) -> NDArray[np.float64]:
    """Return float64 N(0,1) draws of shape ``(n_paths,)``.

    ``n_paths`` must be a positive integer. ``rng`` is required explicitly and
    its state is advanced. Invalid counts/RNG types fail before consuming draws.
    """
    n_paths = positive_integer("n_paths", n_paths)
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng.standard_normal(int(n_paths), dtype=np.float64)


def standard_normal_matrix(n_paths: int, n_steps: int, *, rng: Generator) -> NDArray[np.float64]:
    """Draw IID N(0,1) shocks with shape (n_paths, n_steps), float64.

    Rows identify independent paths; columns identify consecutive time steps.
    These are standardized shocks, not Brownian increments: multiply by sqrt(dt)
    to obtain Brownian increments. Draw order follows the C-order matrix layout
    in the recorded NumPy environment. Counts and RNG are checked before draws.
    """
    n_paths = positive_integer("n_paths", n_paths)
    n_steps = positive_integer("n_steps", n_steps)
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng.standard_normal((n_paths, n_steps), dtype=np.float64)
