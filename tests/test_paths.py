"""Exact grid-path checks: identities, stream contracts and joint moments."""

from dataclasses import replace
from math import exp, sqrt

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

from mc_pricer import (
    BlackScholesModel, make_rng, paths_from_normals, simulate_paths,
    standard_normal_matrix, terminal_from_normals,
)


def test_path_shape_dtype_start_and_positivity():
    model = BlackScholesModel(80, -.03, .3)
    for count, steps in ((1, 1), (3, 7), (101, 12)):
        paths = simulate_paths(model, 2, count, steps, rng=make_rng(12))
        assert paths.shape == (count, steps + 1)
        assert paths.dtype == np.float64 and paths.flags.c_contiguous
        assert_array_equal(paths[:, 0], np.full(count, model.spot))
        assert np.all(np.isfinite(paths)) and np.all(paths > 0)


def test_vectorized_paths_match_small_recursive_reference():
    model = BlackScholesModel(80, -.03, .3)
    shocks = [[-1.0, 0.0, 2.0], [.5, -2.0, 1.0]]
    maturity, dt = 1.5, .5
    expected = []
    # A tiny scalar reference is intentionally independent of the cumsum code.
    for row in shocks:
        prices = [model.spot]
        for shock in row:
            prices.append(prices[-1] * exp((model.rate - model.volatility**2 / 2) * dt
                                          + model.volatility * sqrt(dt) * shock))
        expected.append(prices)
    assert_allclose(paths_from_normals(model, maturity, shocks), expected, rtol=4e-15, atol=0)


def test_path_endpoint_matches_aggregated_terminal_shock():
    model = BlackScholesModel(100, .05, .2)
    shocks = standard_normal_matrix(2000, 252, rng=make_rng(66))
    paths = paths_from_normals(model, 1, shocks)
    terminal = terminal_from_normals(model, 1, shocks.sum(axis=1) / sqrt(252))
    assert_allclose(paths[:, -1], terminal, rtol=5e-14, atol=0)


def test_fine_and_coarse_paths_agree_at_shared_dates():
    model = BlackScholesModel(100, -.02, .4)
    shocks = standard_normal_matrix(1000, 12, rng=make_rng(431))
    coarse_shocks = shocks.reshape(1000, 3, 4).sum(axis=2) / sqrt(4)
    fine = paths_from_normals(model, 2, shocks)
    coarse = paths_from_normals(model, 2, coarse_shocks)
    assert_allclose(fine[:, ::4], coarse, rtol=5e-15, atol=0)


def test_one_step_paths_agree_with_terminal_mapper():
    model = BlackScholesModel(100, .05, .2)
    shocks = np.array([[-2.0], [0.0], [1.0]])
    assert_array_equal(paths_from_normals(model, 1, shocks)[:, -1],
                       terminal_from_normals(model, 1, shocks[:, 0]))


def test_matrix_draw_order_and_wrapper_reproducibility():
    model = BlackScholesModel(100, .05, .2)
    rng_a, rng_b = make_rng(29), make_rng(29)
    shocks = standard_normal_matrix(17, 9, rng=rng_a)
    assert_array_equal(shocks.ravel(), make_rng(29).standard_normal(17 * 9))
    actual = simulate_paths(model, 1, 17, 9, rng=rng_b)
    assert_array_equal(actual, paths_from_normals(model, 1, shocks))
    assert_array_equal(rng_a.standard_normal(11), rng_b.standard_normal(11))
    assert_array_equal(actual, simulate_paths(model, 1, 17, 9, rng=make_rng(29)))


def test_path_transform_does_not_mutate_read_only_noncontiguous_shocks():
    original = np.arange(60, dtype=float).reshape(5, 12) / 20 - 1.5
    before = original.copy()
    view = original[::2, ::3]
    view.setflags(write=False)
    model = BlackScholesModel(100, .05, .2)
    actual = paths_from_normals(model, 1, view)
    assert_array_equal(original, before)
    assert not np.shares_memory(actual, original)
    assert_array_equal(actual, paths_from_normals(model, 1, view.tolist()))


def test_grid_deterministic_limits_and_rng_consumption():
    for volatility, maturity in ((.2, 0), (0, 1), (0, 0)):
        model = BlackScholesModel(100, -.02, volatility)
        rng_a, rng_b = make_rng(78), make_rng(78)
        actual = simulate_paths(model, maturity, 7, 12, rng=rng_a)
        rng_b.standard_normal((7, 12))
        assert_array_equal(rng_a.standard_normal(10), rng_b.standard_normal(10))
        expected = 100 * np.exp(model.rate * np.linspace(0, maturity, 13))
        assert_allclose(actual, np.broadcast_to(expected, (7, 13)), rtol=3e-15, atol=0)
        if maturity == 0:
            assert_array_equal(actual, np.full((7, 13), 100.0))


def test_path_counts_and_maturity_fail_before_sampling():
    model = BlackScholesModel(100, .05, .2)
    cases = [(0, 3, 1, ValueError), (3, 0, 1, ValueError), (3, -1, 1, ValueError),
             (True, 3, 1, TypeError), (3, 1.0, 1, TypeError), (3, "2", 1, TypeError),
             (3, 2, -1, ValueError), (3, 2, np.nan, ValueError), (3, 2, True, TypeError)]
    for count, steps, maturity, error in cases:
        rng_a, rng_b = make_rng(4), make_rng(4)
        with assert_raises(error):
            simulate_paths(model, maturity, count, steps, rng=rng_a)
        assert_array_equal(rng_a.standard_normal(5), rng_b.standard_normal(5))
    assert standard_normal_matrix(np.int64(2), np.int64(3), rng=make_rng(1)).shape == (2, 3)


def test_path_model_and_rng_validation():
    rng_a, rng_b = make_rng(4), make_rng(4)
    with assert_raises(TypeError):
        simulate_paths(None, 1, 3, 2, rng=rng_a)
    assert_array_equal(rng_a.standard_normal(5), rng_b.standard_normal(5))
    with assert_raises(TypeError):
        paths_from_normals(None, 1, [[0]])
    with assert_raises(TypeError):
        standard_normal_matrix(3, 2, rng=4)


def test_path_shocks_require_a_finite_real_nonempty_matrix():
    model = BlackScholesModel(100, .05, .2)
    for shocks in (1.0, [1, 2], [], [[]], np.empty((0, 3)), [[[0]]], [[np.nan]], [[np.inf]]):
        with assert_raises(ValueError):
            paths_from_normals(model, 1, shocks)
    for shocks in ([[True]], [[1j]], [["0"]], np.array([[0]], dtype=object)):
        with assert_raises(TypeError):
            paths_from_normals(model, 1, shocks)
    # Zero maturity still validates the shock contract.
    with assert_raises(ValueError):
        paths_from_normals(model, 0, [[np.nan]])


def test_unrepresentable_intermediate_states_and_time_step_raise():
    model = BlackScholesModel(100, .05, .2)
    for shocks in ([[1e6, -1e6]], [[-1e6, 1e6]]):
        with assert_raises(FloatingPointError):
            paths_from_normals(model, 1, shocks)
    with assert_raises(FloatingPointError):
        paths_from_normals(replace(model, volatility=1e308), 1, [[0]])
    with assert_raises(FloatingPointError):
        paths_from_normals(model, np.nextafter(0.0, 1.0), [[0, 0]])


def test_shared_path_shocks_preserve_spot_scaling():
    model = BlackScholesModel(100, .05, .2)
    shocks = standard_normal_matrix(100, 12, rng=make_rng(17))
    baseline = paths_from_normals(model, 1, shocks)
    bumped = paths_from_normals(replace(model, spot=101), 1, shocks)
    assert_allclose(bumped, 1.01 * baseline, rtol=4e-15, atol=0)


def test_path_log_increment_moments_and_joint_covariance():
    """Joint-normal moment diagnostics with fixed seed and six-SE tolerances."""
    model = BlackScholesModel(100, .05, .2)
    count, steps, maturity = 100_000, 4, 1.0
    paths = simulate_paths(model, maturity, count, steps, rng=make_rng(20260926))
    increments = np.log(paths[:, 1:] / paths[:, :-1])
    dt = maturity / steps
    mean, variance = (model.rate - .5 * model.volatility**2) * dt, model.volatility**2 * dt
    assert np.all(np.abs(increments.mean(axis=0) - mean) < 6 * sqrt(variance / count))
    assert np.all(np.abs(increments.var(axis=0, ddof=1) - variance) < 6 * variance * sqrt(2 / (count - 1)))
    increment_cov = np.cov(increments, rowvar=False, ddof=1)
    off_diagonal = increment_cov - np.diag(np.diag(increment_cov))
    assert np.max(np.abs(off_diagonal)) < 6 * variance / sqrt(count - 1)
    times = np.linspace(0, maturity, steps + 1)[1:]
    target_cov = model.volatility**2 * np.minimum.outer(times, times)
    sample_cov = np.cov(np.log(paths[:, 1:] / model.spot), rowvar=False, ddof=1)
    covariance_se = np.sqrt((np.outer(np.diag(target_cov), np.diag(target_cov)) + target_cov**2) / (count - 1))
    assert np.all(np.abs(sample_cov - target_cov) < 6 * covariance_se)
