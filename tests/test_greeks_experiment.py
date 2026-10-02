"""Validate Greek study targets, budgets, streams and two sources of error."""

from pathlib import Path
import json
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from mc_pricer import finite_difference_greek


def _experiment():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/'experiments/greeks_comparison.py'))


def test_greek_study_replays_streams_and_equal_payoff_budgets():
    module=_experiment()
    raw=module['collect_bump_study'](64,3,319)
    assert len(raw)==108 and raw.spawn_key.nunique()==108
    for row in raw.loc[raw.replication==0].itertuples():
        result=finite_difference_greek(module['MODEL'],module['PRODUCT'],1.,64,
                                      greek=row.greek,bump=row.bump,
                                      rng=module['stream'](319,tuple(json.loads(row.spawn_key))),
                                      common_random_numbers=row.method=='crn')
        assert result.estimate==row.estimate and result.standard_error==row.standard_error
        assert row.n_payoff_evaluations==64*(3 if row.greek=='gamma' else 2)
        assert_allclose(row.finite_bump_target,module['analytical_fd'](row.greek,row.bump),rtol=0,atol=0)


def test_greek_summary_separates_fd_bias_mc_error_and_empirical_mse():
    module=_experiment()
    raw=module['collect_bump_study'](128,4,320)
    summary=module['summarize_greeks'](raw)
    comparison=module['compare_methods'](summary)
    for row in summary.itertuples():
        assert_allclose(row.empirical_bias,row.mc_error_of_mean+row.deterministic_fd_bias,atol=1e-12)
        assert_allclose(row.rmse**2,row.empirical_bias**2+3/4*row.empirical_variance,rtol=1e-12)
        assert_allclose(row.empirical_standard_error_of_mean**2,row.empirical_variance/4,rtol=1e-12)
        assert 0<=row.greek_coverage<=1 and 0<=row.finite_bump_coverage<=1
    for row in comparison.itertuples():
        values=raw.loc[(raw.greek==row.greek)&(raw.bump==row.bump)]
        expected=values.loc[values.method=='independent','estimate'].var(ddof=1)/values.loc[values.method=='crn','estimate'].var(ddof=1)
        assert_allclose(row.empirical_vrf,expected,rtol=1e-12)
    broken=summary.copy()
    broken.loc[broken.method=='crn','n_payoff_evaluations']+=1
    with assert_raises(ValueError):
        module['compare_methods'](broken)


def test_delta_study_has_distinct_counts_and_rejects_invalid_design():
    module=_experiment()
    raw=module['collect_delta_convergence'](3,321,(16,32,64))
    assert len(raw)==18 and raw.spawn_key.nunique()==18
    assert raw.greek.eq('delta').all() and raw.bump.eq(1.).all()
    summary=module['summarize_greeks'](raw)
    assert len(summary)==6
    for paths,runs,seed in ((1,3,1),(8,1,1),(8,3,-1)):
        with assert_raises(ValueError):
            module['collect_bump_study'](paths,runs,seed)
    with assert_raises(TypeError):
        module['collect_bump_study'](8.,3,1)
    with assert_raises(ValueError):
        module['collect_delta_convergence'](3,321,(64,32,16))
    # Large-to-moderate bumps show O(h²) truncation before roundoff dominates.
    diagnostic=module['cancellation_diagnostic']()
    ratio=diagnostic.absolute_error.iloc[1]/diagnostic.absolute_error.iloc[2]
    assert 90<ratio<110
