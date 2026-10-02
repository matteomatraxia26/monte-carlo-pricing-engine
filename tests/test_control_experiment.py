"""Independent pilot study: budgets, stream replay and summary denominators."""

from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from mc_pricer import ArithmeticAsianOption, price_monte_carlo, fit_control_variate


def _experiment():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/"experiments/control_variates.py"))


def test_control_study_replays_distinct_streams_and_includes_pilot_cost():
    module = _experiment()
    raw, checks = module["collect_comparison"](128, 24, 3, 994)
    assert len(raw) == 27 and raw.spawn_key.nunique() == 27 and len(checks) == 9
    for row in raw.itertuples():
        _, kind, steps = module["CASES"][row.case_index]
        method_index = module["METHODS"].index(row.method)
        result = price_monte_carlo(module["MODEL"], ArithmeticAsianOption(100., kind), 1., 128,
                                  n_steps=steps, variance_reduction=row.method,
                                  n_pilot_paths=24 if method_index == 2 else None,
                                  rng=module["stream"](994, row.case_index, row.replication, method_index))
        assert_allclose([row.price, row.standard_error, row.estimator_variance],
                        [result.price, result.standard_error, result.estimator_variance], rtol=0, atol=0)
        assert row.n_independent_samples == (128, 64, 104)[method_index]
        assert row.payoff_evaluations == (128, 128, 256)[method_index]
        if method_index == 2:
            samples = module["replay_control_samples"](994, row.case_index, row.replication, 128, 24)
            fit = fit_control_variate(samples["pilot_target"], samples["pilot_control"])
            assert row.control_beta == fit.beta and samples["production_target"].size == 104
            residual = samples["production_target"]-fit.beta*(samples["production_control"]-row.control_expectation)
            assert_allclose(row.price, np.exp(-.05)*residual.mean(), rtol=1e-14)


def test_control_study_summaries_use_run_variance_and_known_geometric_target_only():
    module = _experiment()
    raw, checks = module["collect_comparison"](128, 24, 4, 995)
    summary, comparison = module["summarize_comparison"](raw)
    assert "empirical_bias" not in summary and "rmse" not in summary
    for row in comparison.itertuples():
        case = raw.loc[raw.case == row.case]
        v0 = case.loc[case.method == "none", "price"].var(ddof=1)
        v1 = case.loc[case.method == row.method, "price"].var(ddof=1)
        assert_allclose(row.empirical_vrf, v0/v1)
        assert_allclose(row.difference_standard_error, np.sqrt((v0+v1)/4))
        assert row.vrf_ci_lower < row.empirical_vrf < row.vrf_ci_upper
    geo = module["summarize_geometric"](checks)
    for row in geo.itertuples():
        errors = checks.loc[checks.case == row.case, "price"]-row.analytical_price
        assert_allclose(row.rmse, np.sqrt(np.mean(errors**2)))
    bad = raw.copy()
    bad.loc[bad.method == "control_variate", "n_paths"] = 256
    with assert_raises(ValueError):
        module["summarize_comparison"](bad)
    degenerate = raw.copy()
    degenerate[["price", "estimator_variance"]] = 0.
    _, undefined = module["summarize_comparison"](degenerate)
    assert undefined.empirical_vrf.isna().all()


def test_control_study_rejects_invalid_predeclared_design():
    collect = _experiment()["collect_comparison"]
    for paths, pilot, runs, seed in ((3,2,4,1),(8,1,4,1),(8,7,4,1),(8,2,1,1),(8,2,4,-1)):
        with assert_raises(ValueError):
            collect(paths, pilot, runs, seed)
    with assert_raises(TypeError):
        collect(8., 2, 4, 1)
