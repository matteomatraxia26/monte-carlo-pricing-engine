"""Pure European call/put payoffs; no discounting or random sampling."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mc_pricer._validation import finite_real, finite_vector


@dataclass(frozen=True)
class EuropeanOption:
    """European payoff with a positive strike and call/put direction.

    Rate, maturity and model dynamics are inputs to the pricing method, not
    this payoff definition. Payoffs and terminal inputs use the same currency.
    """

    strike: float
    option_type: Literal["call", "put"] = "call"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strike", finite_real("strike", self.strike))
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    def payoff(self, terminal_prices: ArrayLike) -> NDArray[np.float64]:
        """Return undiscounted payoffs of shape ``(n_paths,)``.

        Input must be a nonempty finite 1-D real array with nonnegative prices.
        Zero is valid for a payoff even though exact GBM stays strictly positive.
        The input is never modified and the result is a new float64 array.
        """
        terminal = finite_vector("terminal_prices", terminal_prices)
        if np.any(terminal < 0):
            raise ValueError("terminal_prices must be nonnegative")
        if self.option_type == "call":
            return np.maximum(terminal - self.strike, 0.0)
        return np.maximum(self.strike - terminal, 0.0)
