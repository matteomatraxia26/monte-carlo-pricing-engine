"""M15 research protocol: coupled grids, fixed dates, and pooled moments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_raises
from mc_pricer import HestonModel, heston_paths_from_normals, price_from_payoff_samples
from experiments import heston_analysis as e


def test_heston_study_replays_brownian_coupling_and_contractual_dates():
    reference=e.references()
    raw,paired=e.collect_grids(32,3,159,reference,grids=(12,24))
    assert len(raw)==len(paired)==120
    for ci,(name,model) in enumerate(e.CASES):
        rng=e.stream(159,(0,ci,0))
        zs=rng.standard_normal((32,24));zi=rng.standard_normal((32,24))
        fine=heston_paths_from_normals(model,1.,zs,zi,n_observations=12)
        coarse=heston_paths_from_normals(model,1.,e.aggregate_normals(zs,12),e.aggregate_normals(zi,12),n_observations=12)
        assert np.array_equal(fine.times,coarse.times)
        for pi,(product_name,product) in enumerate(e.PRODUCTS):
            x=product.payoff(coarse.stock_paths[:,-1] if pi<2 else coarse.stock_paths)
            y=product.payoff(fine.stock_paths[:,-1] if pi<2 else fine.stock_paths)
            expected=price_from_payoff_samples(x,rate=model.rate,maturity=1.)
            row=raw.loc[(raw.scenario==name)&(raw.replication==0)&(raw.n_steps==12)&(raw['product']==product_name)].iloc[0]
            assert row.price==expected.price and row.standard_error==expected.standard_error
            difference=price_from_payoff_samples(x-y,rate=model.rate,maturity=1.)
            pair=paired.loc[(paired.scenario==name)&(paired.replication==0)&(paired.n_steps==12)&(paired['product']==product_name)].iloc[0]
            assert pair.difference==difference.price and pair.standard_error==difference.standard_error
    assert (paired.loc[paired.n_steps==24,'difference']==0).all()
    summary=e.summarize_prices(raw)
    assert summary.loc[summary['product']=='asian_call','reference_price'].isna().all()
    assert summary.loc[summary['product']=='asian_call','run_rmse'].isna().all()
    assert (summary.total_paths==96).all()


def test_heston_pooled_variance_matches_concatenated_independent_samples():
    samples=np.array([[1.,2.,4.],[-1.,8.,3.],[2.,9.,6.]])
    frame=pd.DataFrame(dict(price=samples.mean(axis=1),sample_variance=samples.var(axis=1,ddof=1),n_paths=3))
    out=e.pooled_summary(frame,'price')
    assert_allclose(out['mean_estimate'],samples.mean())
    assert_allclose(out['pooled_sample_variance'],samples.ravel().var(ddof=1))
    assert_allclose(out['pooled_standard_error'],samples.ravel().std(ddof=1)/3)
    broken=frame.copy();broken.loc[0,'n_paths']=2
    with assert_raises(ValueError):e.pooled_summary(broken,'price')


def test_heston_correlation_study_replays_and_reports_units():
    raw,summary=e.correlation_study(160,replications=3,n_pairs=2000)
    assert len(raw)==15 and len(summary)==5
    for row in raw.loc[raw.replication==0].itertuples():
        z=e.stream(160,tuple(json.loads(row.spawn_key))).standard_normal((2000,2))
        zv=e.correlate_normals(z[:,0],z[:,1],row.rho)
        assert np.corrcoef(z[:,0],zv)[0,1]==row.sample_correlation
        assert row.ci_lower<row.sample_correlation<row.ci_upper


def test_heston_study_design_and_calibration_example():
    for args in ((1,3,0,(12,24)),(32,1,0,(12,24)),(32,3,-1,(12,24)),(32,3,0,(12,25)),(32,3,0,(24,12)),(32,3,0,())):
        with assert_raises((TypeError,ValueError)):e.validate_design(*args)
    fine=np.arange(48.).reshape(2,24)
    coarse=e.aggregate_normals(fine,12)
    assert_allclose(coarse,fine.reshape(2,12,2).sum(axis=2)/np.sqrt(2))
    with assert_raises(ValueError):e.aggregate_normals(fine,5)
    path=Path(__file__).resolve().parents[1]/'examples/heston_calibration.json'
    assert HestonModel.from_json(path.read_text(),spot=100,rate=.05)==e.BASE
