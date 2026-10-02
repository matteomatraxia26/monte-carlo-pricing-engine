"""Discrete arithmetic/geometric Asian payoffs, independent of pricing."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mc_pricer._validation import finite_matrix, finite_real, finite_vector


@dataclass(frozen=True)
class ArithmeticAsianOption:
    """Fixed-strike payoff on equally weighted future observations.

    Path column zero is the initial price and is excluded. Every other column
    is a monitoring observation. The GBM pricer supplies uniform dates j*T/m,
    j=1,...,m. The payoff itself does not simulate, discount or infer dates.
    """

    strike: float
    option_type: Literal["call", "put"] = "call"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strike", finite_real("strike", self.strike))
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    def average(self, paths: ArrayLike) -> NDArray[np.float64]:
        """Return arithmetic means of shape (N,) from paths of shape (N, m+1).

        Require N>=1, m>=1, real finite nonnegative prices. Initial prices may
        differ across rows in supplied samples; they are ignored by the average.
        Input is not modified. Output is newly allocated float64 storage.
        """
        values = finite_matrix("paths", paths)
        if values.shape[1] < 2:
            raise ValueError("paths require an initial column and at least one observation")
        if np.any(values < 0):
            raise ValueError("paths must be nonnegative")
        with np.errstate(over="raise", invalid="raise"):
            return np.mean(values[:, 1:], axis=1)

    def payoff_from_average(self, averages: ArrayLike) -> NDArray[np.float64]:
        """Return undiscounted payoffs from supplied nonnegative means, shape (N,).

        The caller must have computed the intended contract average. This entry
        point supports compact saved experiment samples without retaining paths.
        """
        values = finite_vector("averages", averages)
        if np.any(values < 0):
            raise ValueError("averages must be nonnegative")
        if self.option_type == "call":
            return np.maximum(values - self.strike, 0.0)
        return np.maximum(self.strike - values, 0.0)

    def payoff(self, paths: ArrayLike) -> NDArray[np.float64]:
        """Return undiscounted payoffs of shape (N,), excluding column zero."""
        return self.payoff_from_average(self.average(paths))


@dataclass(frozen=True)
class GeometricAsianOption:
    """Fixed-strike call/put on the equal-weight geometric future average.

    Exclude column zero, include every future column; cashflow paid at maturity.
    Strictly positive supplied prices are required for the logarithm.
    """

    strike: float
    option_type: Literal["call", "put"] = "call"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strike", finite_real("strike", self.strike))
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    def average(self, paths: ArrayLike) -> NDArray[np.float64]:
        """Map positive paths (N,m+1) to geometric averages (N,), unchanged input.

        Logs avoid overflow from multiplying all observations. A single future
        date returns a copy of that column without a log/exp round trip.
        """
        values = finite_matrix("paths", paths)
        if values.shape[1] < 2 or np.any(values <= 0):
            raise ValueError("require positive paths with inception and a future observation")
        if values.shape[1] == 2:
            return values[:, 1].copy()
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.exp(np.log(values[:, 1:]).mean(axis=1))
        if np.any(result <= 0) or not np.all(np.isfinite(result)):
            raise FloatingPointError("geometric averages exceed positive finite float64 range")
        return result

    def payoff_from_average(self, averages: ArrayLike) -> NDArray[np.float64]:
        """Return undiscounted payoffs (N,) from positive geometric means (N,)."""
        values = finite_vector("averages", averages)
        if np.any(values <= 0):
            raise ValueError("geometric averages must be positive")
        sign = 1.0 if self.option_type == "call" else -1.0
        return np.maximum(sign * (values - self.strike), 0.0)

    def payoff(self, paths: ArrayLike) -> NDArray[np.float64]:
        """Return undiscounted payoff vector (N,), excluding initial spot."""
        return self.payoff_from_average(self.average(paths))
