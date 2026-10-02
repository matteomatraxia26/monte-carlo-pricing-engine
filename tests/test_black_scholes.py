"""Plain pytest test functions, also runnable with unittest.FunctionTestCase.

No pytest-specific helpers are required. The normal command is python -m pytest.
"""
from dataclasses import FrozenInstanceError, replace
from math import exp, log, sqrt

import numpy as np
from numpy.testing import assert_allclose, assert_raises
from scipy.integrate import quad
from scipy.stats import norm

from mc_pricer import BlackScholesModel, black_scholes_greeks, black_scholes_price


def test_known_benchmark():
    model = BlackScholesModel(100, 0.05, 0.20)
    for kind, price, delta in (("call", 10.450583572185565, 0.6368306511756191),
                               ("put", 5.573526022256971, -0.3631693488243809)):
        assert_allclose(black_scholes_price(model, 100, 1, kind), price, rtol=0, atol=1e-11)
        greeks = black_scholes_greeks(model, 100, 1, kind)
        assert_allclose(greeks.delta, delta, rtol=0, atol=1e-12)
        assert_allclose(greeks.gamma, 0.018762017345846895, rtol=0, atol=1e-12)
        assert_allclose(greeks.vega, 37.52403469169379, rtol=0, atol=1e-10)


def test_put_call_parity_and_no_arbitrage_bounds():
    cases = [(60, 100, -0.02, 0.35, 0.25), (100, 100, 0, 0.20, 1),
             (150, 100, 0.07, 0.50, 3), (100, 120, 0.03, 0, 2), (90, 100, 0.05, 0.20, 0)]
    for spot, strike, rate, volatility, maturity in cases:
        model = BlackScholesModel(spot, rate, volatility)
        call = black_scholes_price(model, strike, maturity, "call")
        put = black_scholes_price(model, strike, maturity, "put")
        discounted_strike = strike * exp(-rate * maturity)
        assert_allclose(call - put, spot - discounted_strike, rtol=0, atol=2e-12)
        assert max(spot - discounted_strike, 0) - 1e-12 <= call <= spot + 1e-12
        assert max(discounted_strike - spot, 0) - 1e-12 <= put <= discounted_strike + 1e-12


def test_price_against_direct_risk_neutral_integral():
    """Independent quadrature split at the kink; negligible tails beyond 12 SD."""
    for spot, rate, volatility, maturity in ((80, -0.02, 0.30, 0.5),
                                            (100, 0.05, 0.20, 1), (140, 0.01, 0.40, 2)):
        model = BlackScholesModel(spot, rate, volatility)
        strike = 100
        drift = (rate - 0.5 * volatility**2) * maturity
        scale = volatility * sqrt(maturity)
        cutoff = (log(strike / spot) - drift) / scale
        for kind in ("call", "put"):
            sign = 1 if kind == "call" else -1

            def integrand(z):
                terminal = spot * exp(drift + scale * z)
                return exp(-rate * maturity) * max(sign * (terminal - strike), 0) * norm.pdf(z)

            lower, upper = (cutoff, 12) if kind == "call" else (-12, cutoff)
            value, error = quad(integrand, lower, upper, epsabs=1e-10, epsrel=1e-11)
            assert error < 1e-7
            assert_allclose(black_scholes_price(model, strike, maturity, kind),
                            value, rtol=0, atol=1e-8)


def test_greeks_against_deterministic_price_derivatives():
    """Check analytical Greeks; this is not a Monte Carlo Greek engine."""
    for spot, rate, volatility, maturity in ((85, -0.01, 0.30, 0.4),
                                            (100, 0.05, 0.20, 1), (130, 0.03, 0.40, 2)):
        model = BlackScholesModel(spot, rate, volatility)
        h_spot, h_vol = 0.01, 1e-5
        for kind in ("call", "put"):
            def price(updated):
                return black_scholes_price(updated, 100, maturity, kind)

            up = price(replace(model, spot=spot + h_spot))
            base = price(model)
            down = price(replace(model, spot=spot - h_spot))
            delta = (up - down) / (2 * h_spot)
            gamma = (up - 2 * base + down) / h_spot**2
            vega = (price(replace(model, volatility=volatility + h_vol))
                    - price(replace(model, volatility=volatility - h_vol))) / (2 * h_vol)
            greeks = black_scholes_greeks(model, 100, maturity, kind)
            assert_allclose(greeks.delta, delta, rtol=0, atol=5e-8)
            assert_allclose(greeks.gamma, gamma, rtol=0, atol=5e-8)
            assert_allclose(greeks.vega, vega, rtol=1e-7, atol=0)


def test_greek_parity():
    model = BlackScholesModel(100, 0.05, 0.20)
    call = black_scholes_greeks(model, 100, 1, "call")
    put = black_scholes_greeks(model, 100, 1, "put")
    assert_allclose(call.delta - put.delta, 1, rtol=0, atol=1e-15)
    assert call.gamma == put.gamma
    assert call.vega == put.vega


def test_expiry():
    cases = [(110, 100, "call", 10), (90, 100, "call", 0), (90, 100, "put", 10),
             (110, 100, "put", 0), (100, 100, "call", 0)]
    for spot, strike, kind, expected in cases:
        assert black_scholes_price(BlackScholesModel(spot, 0.05, 0.2), strike, 0, kind) == expected


def test_zero_volatility():
    for rate in (-0.05, 0, 0.05):
        for kind in ("call", "put"):
            sign = 1 if kind == "call" else -1
            expected = exp(-rate) * max(sign * (100 * exp(rate) - 100), 0)
            assert_allclose(black_scholes_price(BlackScholesModel(100, rate, 0), 100, 1, kind),
                            expected, rtol=0, atol=1e-12)


def test_invalid_model_parameters():
    model = BlackScholesModel(100, 0.05, 0.2)
    cases = [("spot", 0), ("spot", -1), ("spot", np.nan), ("spot", np.inf),
             ("rate", np.nan), ("rate", -np.inf), ("volatility", -0.1),
             ("volatility", np.inf), ("volatility", np.nan)]
    for field, value in cases:
        with assert_raises(ValueError):
            replace(model, **{field: value})


def test_non_scalar_or_non_numeric_model_parameters():
    for value in (True, "100", [100], 1 + 2j):
        with assert_raises(TypeError):
            BlackScholesModel(value, 0.05, 0.2)


def test_invalid_contract_parameters():
    model = BlackScholesModel(100, 0.05, 0.2)
    cases = [(0, 1, "call"), (-100, 1, "put"), (np.nan, 1, "call"),
             (np.inf, 1, "call"), (100, -1, "call"), (100, np.inf, "call"),
             (100, np.nan, "put"), (100, 1, "CALL")]
    for function in (black_scholes_price, black_scholes_greeks):
        for strike, maturity, kind in cases:
            with assert_raises(ValueError):
                function(model, strike, maturity, kind)
        for strike, maturity in ((True, 1), (100, "1"), ([100], 1)):
            with assert_raises(TypeError):
                function(model, strike, maturity)


def test_boundary_greeks_are_explicitly_unsupported():
    for volatility, maturity in ((0, 1), (0.2, 0), (0, 0)):
        with assert_raises(ValueError):
            black_scholes_greeks(BlackScholesModel(100, 0.05, volatility), 100, maturity)


def test_model_is_immutable():
    model = BlackScholesModel(100, 0.05, 0.2)
    with assert_raises(FrozenInstanceError):
        model.spot = 110


def test_numpy_real_scalars_are_accepted():
    model = BlackScholesModel(np.float64(100), np.float64(0.05), np.float64(0.2))
    assert isinstance(model.spot, float)
    assert black_scholes_price(model, np.float64(100), np.float64(1)) > 0
