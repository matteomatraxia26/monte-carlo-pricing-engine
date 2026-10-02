"""Deterministic checks of undiscounted European payoff definitions."""

from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

from mc_pricer import EuropeanOption


def test_call_and_put_payoffs_at_below_and_above_strike():
    terminal = [0, 80, 100, 110, 130]
    assert_array_equal(EuropeanOption(100, "call").payoff(terminal), [0, 0, 0, 10, 30])
    assert_array_equal(EuropeanOption(100, "put").payoff(terminal), [100, 20, 0, 0, 0])


def test_payoffs_have_pathwise_put_call_parity():
    terminal = np.array([0, 50, 99.5, 100, 101.5, 150])
    call = EuropeanOption(100, "call").payoff(terminal)
    put = EuropeanOption(100, "put").payoff(terminal)
    assert_allclose(call - put, terminal - 100, rtol=0, atol=1e-14)


def test_payoff_output_shape_dtype_and_input_immutability():
    source = np.array([50, 80, 100, 110, 130, 150], dtype=np.float64)
    snapshot = source.copy()
    view = source[::2]
    view.setflags(write=False)
    for kind in ("call", "put"):
        result = EuropeanOption(100, kind).payoff(view)
        assert result.shape == view.shape
        assert result.dtype == np.float64
        assert not np.shares_memory(result, source)
        assert_array_equal(source, snapshot)


def test_product_validation_and_immutability():
    for strike in (0, -1, np.nan, np.inf):
        with assert_raises(ValueError):
            EuropeanOption(strike)
    for strike in (True, "100", [100]):
        with assert_raises(TypeError):
            EuropeanOption(strike)
    for kind in ("CALL", "digital", None):
        with assert_raises(ValueError):
            EuropeanOption(100, kind)
    product = EuropeanOption(100)
    with assert_raises(FrozenInstanceError):
        product.strike = 110


def test_payoff_rejects_invalid_terminal_inputs():
    product = EuropeanOption(100)
    for values in ([], 100, [[100, 110]], [-1, 100], [np.inf], [np.nan]):
        with assert_raises(ValueError):
            product.payoff(values)
    for values in ([True], ["100"], [100 + 1j], np.array([100], dtype=object)):
        with assert_raises(TypeError):
            product.payoff(values)
