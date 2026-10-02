"""Finite differences: formula, covariance, stream and analytical-target checks."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from mc_pricer import (
    ArithmeticAsianOption, BlackScholesModel, EuropeanOption, GeometricAsianOption,
    UpAndOutCall, black_scholes_price, central_first_difference, central_second_difference,
    finite_difference_from_normals, finite_difference_greek, paths_from_normals,
    terminal_from_normals, price_from_payoff_samples,
)

MODEL = BlackScholesModel(100., .05, .2)


def test_central_differences_recover_polynomial_derivatives():
    x = np.array([-3., 0., 2., 7.])
    h = .5
    assert_allclose(central_first_difference((x+h)**2, (x-h)**2, h), 2*x, atol=0)
    assert_allclose(central_second_difference((x+h)**2, x**2, (x-h)**2, h), 2., atol=0)
    assert_allclose(central_first_difference((x+h)**3, (x-h)**3, h), 3*x*x+h*h)
    assert_allclose(central_second_difference((x+h)**4, x**4, (x-h)**4, h), 12*x*x+2*h*h)


def test_difference_helpers_validate_shapes_types_and_bumps():
    for h in (0., -1., np.nan, np.inf):
        with assert_raises(ValueError):
            central_first_difference([1.,2.], [0.,1.], h)
    with assert_raises(FloatingPointError):
        central_second_difference([1.], [1.], [1.], 1e-200)
    for values in ([True,False], [1+2j,2], ['1','2']):
        with assert_raises(TypeError):
            central_first_difference(values,[1.,2.],1.)
    with assert_raises(ValueError):
        central_second_difference([1.,2.],[1.],[1.,2.],1.)
    with assert_raises(ValueError):
        central_first_difference([1.,np.inf],[1.,2.],1.)


def test_crn_results_equal_hand_computed_discounted_difference_statistics():
    z = np.array([-2., -1., -.2, 0., .8, 2.])
    product = EuropeanOption(100.)
    for greek, h in (('delta',1.), ('gamma',2.), ('vega',.01)):
        field = 'volatility' if greek == 'vega' else 'spot'
        center = getattr(MODEL,field)
        up = product.payoff(terminal_from_normals(replace(MODEL,**{field:center+h}),1.,z))
        down = product.payoff(terminal_from_normals(replace(MODEL,**{field:center-h}),1.,z))
        base = product.payoff(terminal_from_normals(MODEL,1.,z))
        diff = ((up-base)+(down-base))/h**2 if greek=='gamma' else (up-down)/(2*h)
        stats = price_from_payoff_samples(diff,rate=.05,maturity=1.)
        result = finite_difference_from_normals(MODEL,product,1.,z,greek=greek,bump=h)
        assert_allclose([result.estimate,result.standard_error,result.sample_variance],
                        [stats.price,stats.standard_error,stats.sample_variance],rtol=0,atol=0)
        assert result.confidence_interval == stats.confidence_interval
        assert result.n_paths == 6 and result.common_random_numbers
        assert result.n_payoff_evaluations == 6*(3 if greek=='gamma' else 2)
        assert_allclose(result.estimator_variance,result.standard_error**2)
        with assert_raises(FrozenInstanceError):
            result.bump = 4.


def test_crn_standard_error_includes_leg_covariance():
    z = np.random.default_rng(311).standard_normal(10000)
    product = EuropeanOption(100.)
    up = np.exp(-.05)*product.payoff(terminal_from_normals(replace(MODEL,spot=101.),1.,z))
    down = np.exp(-.05)*product.payoff(terminal_from_normals(replace(MODEL,spot=99.),1.,z))
    result = finite_difference_from_normals(MODEL,product,1.,z,greek='delta',bump=1.)
    cov = np.cov(up,down,ddof=1)
    correct = (cov[0,0]+cov[1,1]-2*cov[0,1])/(4*len(z))
    naive = (cov[0,0]+cov[1,1])/(4*len(z))
    assert_allclose(result.estimator_variance,correct,rtol=1e-11)
    assert naive > 100*correct


def test_full_greek_engine_replays_shocks_and_rng_continuation():
    for greek,h in (('delta',1.),('gamma',1.),('vega',.01)):
        for crn in (True,False):
            rng = np.random.default_rng(312)
            result = finite_difference_greek(MODEL,EuropeanOption(100.),1.,128,greek=greek,bump=h,
                                             common_random_numbers=crn,rng=rng)
            replay = np.random.default_rng(312)
            z = replay.standard_normal(128)
            zd = None if crn else replay.standard_normal(128)
            zb = replay.standard_normal(128) if not crn and greek=='gamma' else None
            expected = finite_difference_from_normals(MODEL,EuropeanOption(100.),1.,z,greek=greek,
                                                      bump=h,down_normals=zd,base_normals=zb)
            assert result.estimate == expected.estimate and result.standard_error == expected.standard_error
            assert result.common_random_numbers == crn
            assert rng.standard_normal() == replay.standard_normal()


def test_path_greek_routes_use_same_whole_shock_matrix():
    z = np.random.default_rng(313).standard_normal((64,12))
    for product in (ArithmeticAsianOption(100.),ArithmeticAsianOption(100.,'put'),
                    GeometricAsianOption(100.),UpAndOutCall(100.,120.)):
        for greek,h in (('delta',1.),('gamma',2.),('vega',.01)):
            field = 'volatility' if greek=='vega' else 'spot'
            center = getattr(MODEL,field)
            up = product.payoff(paths_from_normals(replace(MODEL,**{field:center+h}),1.,z))
            down = product.payoff(paths_from_normals(replace(MODEL,**{field:center-h}),1.,z))
            base = product.payoff(paths_from_normals(MODEL,1.,z))
            samples = ((up-base)+(down-base))/h**2 if greek=='gamma' else (up-down)/(2*h)
            result = finite_difference_from_normals(MODEL,product,1.,z,greek=greek,bump=h)
            assert_allclose(result.estimate,np.exp(-.05)*samples.mean(),rtol=1e-13,atol=1e-13)
            assert result.n_steps == 12


def test_independent_path_gamma_replays_three_full_matrices():
    replay = np.random.default_rng(314)
    z,zd,zb = [replay.standard_normal((32,4)) for _ in range(3)]
    product=ArithmeticAsianOption(100.)
    expected=finite_difference_from_normals(MODEL,product,1.,z,greek='gamma',bump=1.,
                                            down_normals=zd,base_normals=zb)
    rng=np.random.default_rng(314)
    result=finite_difference_greek(MODEL,product,1.,32,n_steps=4,greek='gamma',bump=1.,
                                  common_random_numbers=False,rng=rng)
    assert result.estimate == expected.estimate and result.standard_error == expected.standard_error
    assert rng.standard_normal()==replay.standard_normal()


def test_greek_shocks_are_readonly_noncontiguous_and_not_mutated():
    z=np.arange(60,dtype=float).reshape(10,6)[::2,::2]/40-1
    before=z.copy()
    z.flags.writeable=False
    result=finite_difference_from_normals(MODEL,ArithmeticAsianOption(100.),1.,z,greek='vega',bump=.01)
    assert np.array_equal(z,before) and result.n_paths==5 and result.n_steps==3


def test_one_observation_asian_greeks_equal_european_greeks():
    z=np.random.default_rng(315).standard_normal(1000)
    for greek,h in (('delta',1.),('gamma',1.),('vega',.01)):
        european=finite_difference_from_normals(MODEL,EuropeanOption(100.),1.,z,greek=greek,bump=h)
        for product in (ArithmeticAsianOption(100.),GeometricAsianOption(100.)):
            asian=finite_difference_from_normals(MODEL,product,1.,z[:,None],greek=greek,bump=h)
            assert_allclose([asian.estimate,asian.standard_error],[european.estimate,european.standard_error],
                            rtol=1e-10,atol=1e-12)


def test_fd_call_put_sample_parity_and_vega_units():
    z=np.random.default_rng(316).standard_normal(10000)
    call,put=EuropeanOption(100.),EuropeanOption(100.,'put')
    deltas=[finite_difference_from_normals(MODEL,p,1.,z,greek='delta',bump=1.).estimate for p in (call,put)]
    stock=np.exp(-.05)*terminal_from_normals(MODEL,1.,z)/MODEL.spot
    assert_allclose(deltas[0]-deltas[1],stock.mean(),rtol=1e-13)
    gammas=[finite_difference_from_normals(MODEL,p,1.,z,greek='gamma',bump=1.).estimate for p in (call,put)]
    assert_allclose(gammas[0],gammas[1],atol=3e-14)
    vega=finite_difference_from_normals(MODEL,call,1.,z,greek='vega',bump=.001).estimate
    assert 30 < vega < 45 and .3 < vega/100 < .45


def test_fd_mc_matches_analytic_finite_bump_expectations_in_se_units():
    # Compare to the exact central price difference, separating MC error from O(h²) bias.
    for kind in ('call','put'):
        for greek,h in (('delta',5.),('gamma',2.),('vega',.05)):
            field='volatility' if greek=='vega' else 'spot'
            center=getattr(MODEL,field)
            up=black_scholes_price(replace(MODEL,**{field:center+h}),100.,1.,kind)
            down=black_scholes_price(replace(MODEL,**{field:center-h}),100.,1.,kind)
            base=black_scholes_price(MODEL,100.,1.,kind)
            target=((up-base)+(down-base))/h**2 if greek=='gamma' else (up-down)/(2*h)
            result=finite_difference_greek(MODEL,EuropeanOption(100.,kind),1.,150000,greek=greek,bump=h,
                                          rng=np.random.default_rng(317))
            assert abs(result.estimate-target) < 6*result.standard_error


def test_invalid_greek_configuration_fails_before_rng_consumption():
    cases=[{'greek':'theta'},{'bump':0.},{'bump':100.},{'bump':1e-18},
           {'greek':'vega','bump':.2},{'maturity':0.},{'n_paths':1},{'n_steps':12},
           {'common_random_numbers':1},{'confidence_level':1.},
           {'product':object()},{'model':object()}, {'model':BlackScholesModel(100.,.05,0.)}]
    for changes in cases:
        rng=np.random.default_rng(318)
        kwargs=dict(model=MODEL,product=EuropeanOption(100.),maturity=1.,n_paths=10,
                    greek='delta',bump=1.,rng=rng)
        kwargs.update(changes)
        with assert_raises((ValueError,TypeError)):
            finite_difference_greek(**kwargs)
        assert rng.standard_normal()==np.random.default_rng(318).standard_normal()
    with assert_raises(ValueError):
        finite_difference_greek(MODEL,ArithmeticAsianOption(100.),1.,10,greek='delta',bump=1.,rng=np.random.default_rng(1))


def test_supplied_greek_shocks_validate_full_leg_configuration():
    z=np.zeros(4)
    bad=[dict(normals=np.zeros((4,1))),dict(normals=np.zeros(1)),
         dict(normals=z,down_normals=np.zeros(3)),dict(normals=z,base_normals=z),
         dict(normals=z,greek='gamma',down_normals=z)]
    for changes in bad:
        kwargs=dict(model=MODEL,product=EuropeanOption(100.),maturity=1.,normals=z,greek='delta',bump=1.)
        kwargs.update(changes)
        with assert_raises(ValueError):
            finite_difference_from_normals(**kwargs)


def test_zero_sample_gamma_is_not_proof_of_zero_true_gamma():
    # All sample paths are far ITM and no path switches at the strike.
    result=finite_difference_from_normals(BlackScholesModel(128.,0.,.25),EuropeanOption(1.),1.,
                                        np.full(4,.125),greek='gamma',bump=1.)
    assert abs(result.estimate)<1e-12 and result.standard_error==0.
