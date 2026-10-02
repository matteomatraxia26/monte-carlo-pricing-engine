"""Heston parameter, full-truncation, stream and deterministic-limit checks."""
from dataclasses import asdict, FrozenInstanceError, replace
import json
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from mc_pricer import (
    HestonModel, BlackScholesModel, EuropeanOption, DigitalOption,
    ArithmeticAsianOption, UpAndOutCall, correlate_normals, heston_paths_from_normals,
    simulate_heston_paths, price_heston_monte_carlo, terminal_from_normals,
    price_from_payoff_samples, deterministic_integrated_variance,
)
MODEL=HestonModel(100.,.05,.04,2.,.04,.3,-.7)


def test_heston_model_validation_feller_and_immutability():
    assert_allclose(MODEL.feller_margin,.07)
    assert replace(MODEL,xi=1.).feller_margin<0  # Allowed, not repaired.
    for key,value in (('spot',0),('kappa',0),('theta',-1),('v0',-1),('xi',-1),('rho',1.1),('rate',np.inf),('rho',True)):
        with assert_raises((ValueError,TypeError)):
            replace(MODEL,**{key:value})
    with assert_raises(FrozenInstanceError):
        MODEL.rho=0.


def test_heston_calibration_dict_json_roundtrip_and_strict_keys():
    full=asdict(MODEL); copy=full.copy()
    assert HestonModel.from_dict(full)==MODEL and full==copy
    partial={k:v for k,v in full.items() if k not in ('spot','rate')}
    assert HestonModel.from_dict(partial,spot=100,rate=.05)==MODEL
    assert HestonModel.from_json(json.dumps(partial),spot=100,rate=.05)==MODEL
    for data in ({**full,'rhoo':0},partial,[],{'parameters':full}):
        with assert_raises((ValueError,TypeError)):
            HestonModel.from_dict(data)
    with assert_raises(ValueError):
        HestonModel.from_dict(full,spot=100)
    for text in ('{"rho":0,"rho":1}', '[1,2]', 'invalid', json.dumps({**full,'v0':True})):
        with assert_raises((ValueError,TypeError)):
            HestonModel.from_json(text)


def test_correlated_shocks_formula_endpoints_and_empirical_correlation():
    z=np.random.default_rng(151).standard_normal((200000,2))
    for rho in (-1.,-.7,0.,.6,1.):
        zv=correlate_normals(z[:,0],z[:,1],rho)
        assert_allclose(zv,rho*z[:,0]+np.sqrt(1-rho*rho)*z[:,1],atol=1e-14)
        assert abs(np.corrcoef(z[:,0],zv)[0,1]-rho)<.012
        assert abs(zv.var(ddof=1)-1)<.02
    assert np.array_equal(correlate_normals(z[:,0],z[:,1],1),z[:,0])
    for x,y,rho in (([1],[1,2],0),([1],[1],2),([True],[1],0),([[1]],[[np.nan]],0)):
        with assert_raises((ValueError,TypeError)):
            correlate_normals(x,y,rho)


def test_full_truncation_carries_negative_auxiliary_state():
    model=HestonModel(100.,0.,.01,1.,.04,1.,1.)
    paths=heston_paths_from_normals(model,3.,[[-1.,0.,0.]],[[0.,0.,0.]])
    assert_allclose(paths.raw_variance_paths,[[.01,-.06,-.02,.02]],atol=1e-15)
    assert_allclose(paths.variance_paths,[[.01,0.,0.,.02]],atol=1e-15)
    assert_allclose(paths.stock_paths,[[100.,100*np.exp(-.105),100*np.exp(-.105),100*np.exp(-.105)]])
    assert_allclose(paths.integrated_variance,[.01])
    assert paths.negative_variance_fraction==2/3


def test_heston_first_step_uses_old_variance_and_correlation_orientation():
    zs=np.array([[1.],[-1.]])
    zi=np.array([[.5],[2.]])
    out=heston_paths_from_normals(MODEL,.5,zs,zi)
    expected_stock=100*np.exp((.05-.04/2)*.5+np.sqrt(.04*.5)*zs[:,0])
    expected_raw=.04+.3*np.sqrt(.04*.5)*(-.7*zs[:,0]+np.sqrt(1-.7**2)*zi[:,0])
    assert_allclose(out.stock_paths[:,-1],expected_stock)
    assert_allclose(out.raw_variance_paths[:,-1],expected_raw)


def test_heston_generated_shocks_replay_and_monitoring_storage():
    rng=np.random.default_rng(152)
    full=simulate_heston_paths(MODEL,1.,64,24,rng=rng)
    replay=np.random.default_rng(152)
    shocks=replay.standard_normal((24,64,2))
    supplied=heston_paths_from_normals(MODEL,1.,shocks[:,:,0].T,shocks[:,:,1].T)
    assert np.array_equal(full.stock_paths,supplied.stock_paths)
    assert np.array_equal(full.raw_variance_paths,supplied.raw_variance_paths)
    assert rng.standard_normal()==replay.standard_normal()
    sparse=simulate_heston_paths(MODEL,1.,64,24,rng=np.random.default_rng(152),n_observations=12)
    terminal=simulate_heston_paths(MODEL,1.,64,24,rng=np.random.default_rng(152),n_observations=1)
    assert np.array_equal(sparse.stock_paths,full.stock_paths[:,::2])
    assert np.array_equal(terminal.stock_paths,full.stock_paths[:,[0,-1]])
    assert sparse.negative_variance_fraction==full.negative_variance_fraction
    assert np.array_equal(terminal.integrated_variance,full.integrated_variance)


def test_heston_xi_zero_constant_variance_matches_exact_gbm_pathwise():
    model=replace(MODEL,xi=0.)
    z=np.random.default_rng(153).standard_normal((1000,12))
    zi=np.zeros_like(z); z.flags.writeable=False
    out=heston_paths_from_normals(model,1.,z,zi)
    normal=z.sum(axis=1)/np.sqrt(12)
    expected=terminal_from_normals(BlackScholesModel(100,.05,.2),1.,normal)
    assert_allclose(out.stock_paths[:,-1],expected,rtol=3e-15)
    assert_allclose(out.raw_variance_paths,.04)
    assert_allclose(out.integrated_variance,.04)


def test_heston_deterministic_variance_recursion_and_integral_converge():
    model=replace(MODEL,xi=0.,v0=.09)
    errors=[]
    exact=deterministic_integrated_variance(model,1.)
    for steps in (12,48,192):
        z=np.zeros((2,steps))
        out=heston_paths_from_normals(model,1.,z,z)
        grid=np.arange(steps+1)
        raw=model.theta+(model.v0-model.theta)*(1-model.kappa/steps)**grid
        assert_allclose(out.raw_variance_paths[0],raw)
        errors.append(abs(out.integrated_variance[0]-exact))
    assert errors[2]<errors[1]<errors[0]


def test_heston_zero_time_zero_variance_and_rng_consumption():
    for model,time in ((MODEL,0.),(replace(MODEL,v0=0.,theta=0.),1.)):
        rng=np.random.default_rng(154)
        out=simulate_heston_paths(model,time,8,12,rng=rng)
        expected=model.spot*np.exp(model.rate*out.times)
        assert_allclose(out.stock_paths,np.broadcast_to(expected,out.stock_paths.shape))
        replay=np.random.default_rng(154);replay.standard_normal((12,8,2))
        assert rng.standard_normal()==replay.standard_normal()


def test_heston_invalid_configurations_do_not_consume_rng():
    for changes in (dict(model=object()),dict(maturity=-1),dict(n_paths=True),dict(n_steps=0),dict(n_observations=5)):
        rng=np.random.default_rng(155)
        args=dict(model=MODEL,maturity=1.,n_paths=8,n_steps=12,rng=rng);args.update(changes)
        with assert_raises((ValueError,TypeError)):
            simulate_heston_paths(**args)
        assert rng.standard_normal()==np.random.default_rng(155).standard_normal()
    for zs,zi in (([],[]),([[1]],[[1,2]]),([[np.inf]],[[0]]),([[True]],[[0]])):
        with assert_raises((ValueError,TypeError)):
            heston_paths_from_normals(MODEL,1.,zs,zi)


def test_heston_pricing_replays_pure_payoffs_and_sampling_statistics():
    for product in (EuropeanOption(100),DigitalOption(100),ArithmeticAsianOption(100),UpAndOutCall(100,120)):
        path_product=isinstance(product,(ArithmeticAsianOption,UpAndOutCall))
        out=price_heston_monte_carlo(MODEL,product,1.,1000,24,rng=np.random.default_rng(156),n_observations=12 if path_product else None)
        paths=simulate_heston_paths(MODEL,1.,1000,24,rng=np.random.default_rng(156),n_observations=12 if path_product else 1)
        samples=product.payoff(paths.stock_paths if path_product else paths.stock_paths[:,-1])
        reference=price_from_payoff_samples(samples,rate=.05,maturity=1.)
        assert out.price==reference.price and out.standard_error==reference.standard_error
        assert out.n_steps==24 and out.n_observations==(12 if path_product else 1)
        assert out.n_paths==out.n_independent_samples==1000
    for kwargs in (dict(n_paths=1),dict(confidence_level=1),dict(n_observations=12)):
        rng=np.random.default_rng(157)
        args=dict(model=MODEL,product=EuropeanOption(100),maturity=1,n_paths=8,n_steps=12,rng=rng);args.update(kwargs)
        with assert_raises(ValueError):price_heston_monte_carlo(**args)
        assert rng.standard_normal()==np.random.default_rng(157).standard_normal()


def test_heston_stock_martingale_and_variance_mean_at_fine_grid():
    paths=simulate_heston_paths(MODEL,1.,40000,192,rng=np.random.default_rng(158),n_observations=1)
    discounted=np.exp(-.05)*paths.stock_paths[:,-1]
    assert abs(discounted.mean()-100)<6*discounted.std(ddof=1)/np.sqrt(len(discounted))
    effective=paths.variance_paths[:,-1]
    assert abs(effective.mean()-.04)<6*effective.std(ddof=1)/np.sqrt(len(effective))+.0002
