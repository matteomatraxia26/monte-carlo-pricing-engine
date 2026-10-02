"""Cash-or-nothing European payoffs, independent of model and estimator."""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray
from mc_pricer._validation import finite_real, finite_vector


@dataclass(frozen=True)
class DigitalOption:
    """Pay fixed cash at maturity: call if ST>K, put if ST<=K.

    The complementary equality convention matters in deterministic cases.
    Under positive-variance GBM the equality event has probability zero.
    Payout is positive and denominated in the same currency as the option price.
    """

    strike: float
    option_type: Literal['call', 'put'] = 'call'
    payout: float = 1.

    def __post_init__(self) -> None:
        for name in ('strike', 'payout'):
            value = finite_real(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f'{name} must be positive')
            object.__setattr__(self, name, value)
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")

    def payoff(self, terminal_prices: ArrayLike) -> NDArray[np.float64]:
        """Return a new undiscounted float64 payoff vector (N,), N>=1.

        Prices must be finite and nonnegative; inputs, including views, are
        never modified. This function does not differentiate the indicator.
        """
        terminal = finite_vector('terminal_prices', terminal_prices)
        if np.any(terminal < 0):
            raise ValueError('terminal_prices must be nonnegative')
        active = terminal > self.strike if self.option_type == 'call' else terminal <= self.strike
        return active.astype(np.float64) * self.payout
