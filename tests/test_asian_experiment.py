"""Verify calendar-date coupling and uncertainty units in the Asian study."""

from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mc_pricer import BlackScholesModel, simulate_paths


def _experiment():
    source = Path(__file__).resolve().parents[1] / "experiments" / "asian_pricing.py"
    return runpy.run_path(str(source))


def test_asian_study_replays_calendar_observations_from_each_stream():
    collect_averages = _experiment()["collect_averages"]
    model = BlackScholesModel(100.0, 0.05, 0.2)
    counts = (1, 2, 4)
    averages, raw = collect_averages(model, 1.0, 100.0, counts,
                                     n_paths=32, n_replications=3, seed=207)
    assert averages.shape == (3, 32, 3)
    assert len(raw) == 18
    for replication in range(3):
        stream = np.random.SeedSequence(207, spawn_key=(replication,))
        paths = simulate_paths(model, 1.0, 32, 4, rng=np.random.default_rng(stream))
        # Explicit dates on the four-step calendar, without the initial price.
        for index, columns in enumerate(([4], [2, 4], [1, 2, 3, 4])):
            expected = np.array([sum(path[j] for j in columns) / len(columns) for path in paths])
            assert_allclose(averages[replication, :, index], expected, rtol=2e-15)
        for row in raw.loc[raw.replication == replication].itertuples():
            sample = averages[replication, :, counts.index(row.n_observations)]
            sign = 1 if row.product == "call" else -1
            payoffs = np.exp(-0.05) * np.maximum(sign * (sample - 100), 0)
            assert_allclose(row.price, payoffs.mean(), rtol=2e-15)
            assert_allclose(row.standard_error, payoffs.std(ddof=1) / np.sqrt(32), rtol=2e-15)
    assert not np.array_equal(averages[0], averages[1])


def test_asian_study_uses_paired_covariance_and_pooled_path_count():
    summarize_averages = _experiment()["summarize_averages"]
    # Two independent runs of two paths. Cross-grid values are correlated;
    # call payoff differences are [10, 0, 5, 0], with nonzero sampling variance.
    averages = np.array([[[120., 110.], [80., 90.]], [[110., 105.], [90., 95.]]])
    saved = averages.copy()
    table, validation = summarize_averages(averages, BlackScholesModel(100., 0., .2),
                                          1., 100., (1, 2))
    row = table.loc[(table.option_type == "call") & (table.n_steps == 1)].iloc[0]
    assert row.n_paths == 4  # Neither 2 replications nor 8 grid entries.
    assert row.price == 7.5
    assert row.difference_to_reference == 3.75
    assert_allclose(row.difference_standard_error, np.sqrt(275 / 48), rtol=1e-15)
    fine = table.loc[(table.option_type == "call") & table.is_reference].iloc[0]
    naive_independent_se = np.hypot(row.standard_error, fine.standard_error)
    assert row.difference_standard_error < naive_independent_se
    assert_allclose(table.loc[table.is_reference, "difference_standard_error"], 0, atol=0)
    assert_allclose(validation.sample_parity_residual, 0, atol=0)
    assert_array_equal(averages, saved)
