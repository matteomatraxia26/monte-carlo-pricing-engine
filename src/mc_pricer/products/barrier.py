"""Discrete zero-rebate up-and-out call payoffs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mc_pricer._validation import finite_matrix, finite_real, finite_vector


@dataclass(frozen=True)
class UpAndOutCall:
    """Call paid at maturity only if every observed price is below barrier.

    Monitor every supplied column, including initial spot and maturity.
    Equality knocks out. No rebate or between-date crossing correction.
    The GBM pricer supplies uniform dates j*T/m, j=0,...,m.
    """

    strike: float
    barrier: float

    def __post_init__(self) -> None:
        for name in ("strike", "barrier"):
            value = finite_real(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def payoff_from_extrema(
        self, terminal_prices: ArrayLike, path_maxima: ArrayLike,
    ) -> NDArray[np.float64]:
        """Undiscounted payoffs from two vectors of shape (N,), N>=1.

        Maxima must include inception and maturity on the intended monitoring
        schedule. Require finite nonnegative prices and maximum >= terminal.
        This compact representation supports replay without retaining paths.
        Arrays are not mutated. This function cannot verify omitted dates.
        """
        terminal = finite_vector("terminal_prices", terminal_prices)
        maximum = finite_vector("path_maxima", path_maxima)
        if terminal.shape != maximum.shape:
            raise ValueError("terminal_prices and path_maxima must have the same shape")
        if np.any(terminal < 0) or np.any(maximum < terminal):
            raise ValueError("require 0 <= terminal_prices <= path_maxima")
        survives = maximum < self.barrier
        return np.maximum(terminal - self.strike, 0.0) * survives

    def payoff(self, paths: ArrayLike) -> NDArray[np.float64]:
        """Return payoffs (N,) from paths (N,m+1), N>=1, m>=1.

        Column zero is monitored, unlike the arithmetic Asian average.
        Initial prices can differ across supplied rows. Input is unchanged.
        """
        values = finite_matrix("paths", paths)
        if values.shape[1] < 2:
            raise ValueError("paths require inception and at least one future observation")
        if np.any(values < 0):
            raise ValueError("paths must be nonnegative")
        return self.payoff_from_extrema(values[:, -1], values.max(axis=1))
