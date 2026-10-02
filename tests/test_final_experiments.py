"""M16: budget accounting, independent replay, statistics and evidence integrity."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_raises
from experiments import final_experiments as e
from mc_pricer import price_monte_carlo, finite_difference_greek


def test_final_design_rejects_invalid_counts_and_methods():
    for args in ((11,3,0,2),(13,3,0,2),(120,1,0,12),(120,3,-1,12),(120,3,0,1),(120,3,0,120),(True,3,0,2)):
        with assert_raises((TypeError,ValueError)): e.validate_design(*args)
    with assert_raises(ValueError):e.run_cell(8,'none',120,12,e.stream(0,(0,)))
    with assert_raises(ValueError):e.run_cell(0,'control_variate',120,12,e.stream(0,(0,)))


def test_final_pricing_replay_counts_pilot_and_antithetic_units():
    raw=e.collect(120,3,161,12)
    assert len(raw)==48 and raw.spawn_key.nunique()==48
    for row in raw.loc[(raw.case_index<5)&(raw.replication==0)].itertuples():
        case=e.study_cases()[row.case_index]
        expected=price_monte_carlo(case.model,case.product,case.maturity,120,
            n_steps=case.n_steps,variance_reduction=row.method,
            n_pilot_paths=12 if row.method=='control_variate' else None,
            rng=e.stream(161,tuple(json.loads(row.spawn_key))))
        assert row.estimate==expected.price and row.standard_error==expected.standard_error
        assert row.budget==row.n_paths==120
        assert row.n_independent_samples=={'none':120,'antithetic':60,'control_variate':108}[row.method]
        assert row.n_payoff_evaluations==(240 if row.method=='control_variate' else 120)
    assert raw.loc[raw.case.isin(['asian_call_12','barrier_call_96']),'analytical_value'].isna().all()


def test_final_greeks_use_equal_terminal_evaluations_and_correct_targets():
    raw=e.collect(120,2,162,12)
    greek=raw.loc[raw.case_index>=5]
    assert (greek.n_payoff_evaluations==120).all()
    for row in greek.loc[(greek.method=='fd')&(greek.replication==0)].itertuples():
        expected=finite_difference_greek(e.BASE,e.EuropeanOption(100.),1.,120//(3 if row.quantity=='gamma' else 2),
            greek=row.quantity,bump=row.bump,rng=e.stream(162,tuple(json.loads(row.spawn_key))))
        assert row.estimate==expected.estimate
        assert row.n_paths==(40 if row.quantity=='gamma' else 60)
        # Analytical finite differences are independent from the MC samples.
        assert np.isfinite(row.estimator_target) and row.estimator_target!=row.analytical_value
    assert (greek.loc[greek.method!='fd','n_paths']==120).all()


def test_final_summary_has_correct_rmse_identity_unknown_targets_and_zero_runs():
    raw=e.collect(120,3,163,12)
    # Deliberately retain a zero run rather than deleting inconvenient evidence.
    mask=(raw.case_index==0)&(raw.method=='none')
    raw.loc[mask,['estimate','standard_error','reported_estimator_variance','ci_lower','ci_upper']]=0.
    summary=e.summarize(raw)
    row=summary.loc[(summary.case=='european_call')&(summary.method=='none')].iloc[0]
    assert row.zero_se_fraction==1 and row.exact_target_coverage==0
    assert_allclose(row.rmse,row.analytical_value,rtol=2e-15)
    assert row.empirical_variance==0
    for row in summary.loc[summary.analytical_value.notna()].itertuples():
        assert_allclose(row.rmse**2,row.empirical_bias**2+(row.n_replications-1)/row.n_replications*row.empirical_variance,rtol=1e-12)
    unknown=summary.loc[summary.case.isin(['asian_call_12','barrier_call_96'])]
    assert unknown[['rmse','empirical_bias','exact_target_coverage','deterministic_fd_bias']].isna().all().all()
    comparison=e.compare(raw,163,resamples=100)
    zero=comparison.loc[comparison.case=='european_call'].iloc[0]
    assert np.isnan(zero.empirical_vrf) and zero.status=='zero_empirical_variance'


def test_final_summary_rejects_missing_rows_mixed_contracts_and_budget_units():
    raw=e.collect(120,3,164,12)
    for column,value in (('budget',240),('budget_unit','unrelated'),('analytical_value',-5.),('n_steps',2)):
        broken=raw.copy();broken.loc[0,column]=value
        with assert_raises(ValueError):e.summarize(broken)
    for broken in (raw.iloc[1:],pd.concat([raw,raw.iloc[:1]],ignore_index=True),raw.loc[raw.case_index!=7]):
        with assert_raises(ValueError):e.summarize(broken)


def test_final_comparison_replays_independent_bootstrap_and_preserves_costs():
    raw=e.collect(120,4,165,12);comparison=e.compare(raw,165,resamples=100)
    assert len(comparison)==8
    row=comparison.loc[(comparison.case=='european_call_delta')&(comparison.candidate=='pw')].iloc[0]
    group=raw.loc[raw.case=='european_call_delta']
    ref=group.loc[group.method=='fd'];cand=group.loc[group.method=='pw']
    expected=e.bootstrap_variance_ratio(ref.estimate.to_numpy(),cand.estimate.to_numpy(),
        rng=e.stream(165,tuple(json.loads(row.bootstrap_spawn_key))),resamples=100)
    assert row.empirical_vrf==expected['empirical_vrf'] and row.bootstrap_lower==expected['bootstrap_lower']
    assert_allclose(row.time_adjusted_efficiency,row.empirical_vrf*ref.runtime_seconds.mean()/cand.runtime_seconds.mean())
    assert row.budget==120 and row.budget_unit=='terminal_evaluations'


def test_final_evidence_index_detects_missing_or_invalid_artifacts():
    with TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'tables').mkdir();(root/'figures').mkdir()
        with assert_raises(FileNotFoundError):e.evidence_index(root)
        for _,figure,table,_ in e.EVIDENCE:
            (root/'figures'/f'{figure}.png').write_bytes(b'\x89PNG\r\n\x1a\nfixture')
            (root/'tables'/f'{table}.csv').write_text('value\n1\n')
        for name in e.EXTRA_SOURCES:(root/name).write_text('value\n1\n')
        index=e.evidence_index(root)
        assert len(index)==12 and index.figure.nunique()==12
        path=root/index.figure.iloc[0];path.write_bytes(b'invalid')
        with assert_raises(ValueError):e.evidence_index(root)
