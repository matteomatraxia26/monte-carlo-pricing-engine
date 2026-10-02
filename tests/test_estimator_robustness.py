"""M13 study design: units, independent streams, rare outcomes and bootstrap."""
import json
from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from mc_pricer import ArithmeticAsianOption, price_monte_carlo


def _experiment():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/'experiments/estimator_robustness.py'))


def test_robustness_greek_design_replays_targets_streams_and_budgets():
    m=_experiment()
    raw=m['collect_greeks'](64,3,513)
    assert len(raw)==192 and raw.spawn_key.nunique()==192
    for row in raw.loc[raw.replication==0].itertuples():
        case=m['GREEK_CASES'][row.case_index]
        result=m['run_greek'](case,row.method_index,64,m['stream'](513,tuple(json.loads(row.spawn_key))))
        assert row.estimate==result.estimate and row.standard_error==result.standard_error
        assert row.n_paths==(128 if row.method=='pw_delta' else 64)
        assert row.terminal_evaluations==(192 if row.greek=='gamma' else 128)
        true,target,h=m['greek_targets'](case,row.method_index)
        assert row.analytical_value==true and row.estimator_target==target
    assert raw.loc[raw.method=='fd_delta_1pct','bump'].nunique()>1  # Relative spot bumps.


def test_robustness_asian_counts_pilot_and_does_not_invent_exact_targets():
    m=_experiment()
    raw=m['collect_asian'](128,16,3,514)
    assert len(raw)==36 and raw.spawn_key.nunique()==36
    for row in raw.loc[raw.replication==0].itertuples():
        case=m['ASIAN_CASES'][row.case_index]
        result=price_monte_carlo(case.model,ArithmeticAsianOption(case.strike),case.maturity,128,n_steps=12,
                                n_pilot_paths=16 if row.method=='control_variate' else None,
                                variance_reduction=row.method,rng=m['stream'](514,tuple(json.loads(row.spawn_key))))
        assert result.price==row.estimate and result.standard_error==row.standard_error
        assert row.n_independent_samples=={'none':128,'antithetic':64,'control_variate':112}[row.method]
    summary=m['summarize_asian'](raw)
    assert 'rmse' not in summary and 'empirical_bias' not in summary and 'coverage' not in summary
    for row in summary.itertuples():
        group=raw.loc[(raw.scenario==row.scenario)&(raw.method==row.method)]
        assert_allclose(row.upper_ci_below_lower_bound_rate,(group.ci_upper<group.geometric_lower_bound).mean())


def test_robustness_zero_se_runs_remain_in_coverage_and_rmse():
    m=_experiment()
    raw=m['collect_greeks'](64,3,515)
    rare=raw.loc[(raw.scenario=='rare')&(raw.method=='pw_delta')].copy()
    rare[['estimate','standard_error','estimator_variance','ci_lower','ci_upper']]=0.
    row=m['summarize_greeks'](rare).iloc[0]
    assert row.zero_se_rate==1. and row.greek_coverage==0. and row.estimator_coverage==0.
    assert row.empirical_variance==0.
    assert_allclose(row.rmse,abs(row.analytical_value))
    assert row.zero_se_ci_lower<1. and row.zero_se_ci_upper==1.
    diagnosis=m['rare_event_diagnostic'](64,raw)
    r=diagnosis.loc[diagnosis.scenario=='rare'].iloc[0]
    assert_allclose(r.theoretical_no_active_probability,(1-r.exercise_probability)**128,rtol=1e-12)


def test_bootstrap_ratio_matches_explicit_resampling_and_handles_degeneracy():
    m=_experiment()
    x=np.array([0.,1.,2.,3.])
    y=np.array([-2.,0.,2.,4.])
    result=m['bootstrap_variance_ratio'](x,y,rng=np.random.default_rng(516),resamples=400)
    rng=np.random.default_rng(516)
    vx=x[rng.integers(0,4,size=(400,4))].var(axis=1,ddof=1)
    vy=y[rng.integers(0,4,size=(400,4))].var(axis=1,ddof=1)
    valid=(vx>0)&(vy>0)
    assert_allclose(result['empirical_vrf'],.25)
    assert_allclose([result['bootstrap_lower'],result['bootstrap_upper']],np.quantile(vx[valid]/vy[valid],[.025,.975]))
    assert result['bootstrap_valid_fraction']==valid.mean()
    degenerate=m['bootstrap_variance_ratio'](np.zeros(4),y,rng=np.random.default_rng(516))
    assert np.isnan(degenerate['empirical_vrf']) and degenerate['status']=='zero_empirical_variance'
    with assert_raises(ValueError):
        m['bootstrap_variance_ratio']([0.,np.nan],y,rng=np.random.default_rng(1))
    with assert_raises(ValueError):
        m['bootstrap_variance_ratio'](x,y,rng=np.random.default_rng(1),resamples=1)


def test_robustness_comparison_enforces_costs_and_mse_denominator():
    m=_experiment()
    greek=m['collect_greeks'](128,4,517)
    asian=m['collect_asian'](128,16,4,517)
    summary=m['summarize_greeks'](greek)
    assert_allclose(summary.rmse**2,summary.empirical_bias**2+.75*summary.empirical_variance,rtol=1e-11,atol=1e-16)
    comparison=m['comparisons'](greek,asian,517,100)
    assert len(comparison)==48
    assert comparison.loc[comparison.study=='asian','empirical_mse_ratio'].isna().all()
    bad=greek.copy()
    bad.loc[bad.method=='pw_delta','terminal_evaluations']+=1
    with assert_raises(ValueError):
        m['comparisons'](bad,asian,517,100)
    for args in ((3,4,1,2),(8,1,1,2),(8,4,-1,2),(8,4,1,7)):
        with assert_raises(ValueError):
            m['validate_design'](*args)
    with assert_raises(TypeError):
        m['validate_design'](8.,4,1,2)
