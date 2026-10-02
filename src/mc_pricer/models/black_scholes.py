"""Parameters of a non-dividend-paying Black-Scholes model."""

from dataclasses import dataclass

from mc_pricer._validation import finite_real


@dataclass(frozen=True)
class BlackScholesModel:
    """Scalar market parameters; time is measured in years.

    ``spot`` must be positive. ``rate`` is continuously compounded and may
    be negative. ``volatility`` is annualized, nonnegative and expressed as
    a decimal, e.g. 0.20 for 20%. Terminal and grid simulation are implemented separately
    in ``mc_pricer.simulation.paths`` so parameters contain no RNG state.
    """

    spot: float
    rate: float
    volatility: float

    def __post_init__(self) -> None:
        for name in ("spot", "rate", "volatility"):
            object.__setattr__(self, name, finite_real(name, getattr(self, name)))
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be nonnegative")
