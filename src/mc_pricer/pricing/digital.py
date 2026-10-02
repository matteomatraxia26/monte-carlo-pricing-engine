"""Direct cash-or-nothing Black-Scholes price and spot Delta benchmarks."""

import numpy as np
from scipy.stats import norm
from mc_pricer._validation import finite_real
from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.products.digital import DigitalOption
from mc_pricer.pricing.analytical import _d1_d2, _validate_contract, OptionType


def _inputs(model: BlackScholesModel, strike: float, maturity: float,
            option_type: OptionType, payout: float) -> tuple[float, float, float, float]:
    if not isinstance(model, BlackScholesModel):
        raise TypeError('digital benchmark requires a BlackScholesModel')
    strike, time = _validate_contract(strike, maturity, option_type)
    cash = finite_real('payout', payout)
    if cash <= 0:
        raise ValueError('payout must be positive')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        discount = float(np.exp(-model.rate*time))
    if not np.isfinite(discount) or discount <= 0:
        raise FloatingPointError('discount factor exceeds positive finite float64 range')
    return strike, time, cash, discount


def digital_price(model: BlackScholesModel, strike: float, maturity: float,
                  option_type: OptionType = 'call', payout: float = 1.) -> float:
    """Return discounted cash times Phi(±d2); handle deterministic boundaries.

    Call exercises strictly above strike; put exercises at or below it.
    Positive-variance tails use Phi(-d2) directly, avoiding subtraction from one.
    """
    strike, time, cash, discount = _inputs(model, strike, maturity, option_type, payout)
    if time == 0 or model.volatility == 0:
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            terminal = model.spot * np.exp(model.rate*time)
        if not np.isfinite(terminal) or terminal <= 0:
            raise FloatingPointError('terminal stock exceeds positive finite float64 range')
        return float(discount*DigitalOption(strike, option_type, cash).payoff([terminal])[0])
    _, d2 = _d1_d2(model, strike, time)
    with np.errstate(over='raise', invalid='raise'):
        return float(discount * cash * norm.cdf(d2 if option_type == 'call' else -d2))


def digital_delta(model: BlackScholesModel, strike: float, maturity: float,
                  option_type: OptionType = 'call', payout: float = 1.) -> float:
    """Return ±cash*exp(-rT)*phi(d2)/(S0*sigma*sqrt(T)).

    T and sigma must be positive; deterministic-boundary Greeks are unsupported.
    Delta units are payout currency per spot currency.
    """
    strike, time, cash, discount = _inputs(model, strike, maturity, option_type, payout)
    if time <= 0 or model.volatility <= 0:
        raise ValueError('digital Delta requires positive maturity and volatility')
    _, d2 = _d1_d2(model, strike, time)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = discount * cash * norm.pdf(d2) / (model.spot*model.volatility*np.sqrt(time))
    return float(result if option_type == 'call' else -result)
