"""Independent density, quadrature, boundary and stream validation for M14."""
from dataclasses import FrozenInstanceError, replace
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from scipy.integrate import quad
from scipy.stats import norm
from mc_pricer import (
    ArithmeticAsianOption, BlackScholesModel, DigitalOption, EuropeanOption,
    black_scholes_greeks, digital_price, digital_delta, gbm_spot_score,
    likelihood_ratio_delta_samples, likelihood_ratio_delta_from_normals,
    likelihood_ratio_delta, pathwise_delta, price_monte_carlo,
    price_from_terminal_samples, terminal_from_normals, finite_difference_greek,
    finite_difference_from_normals,
)

MODEL = BlackScholesModel(100., .05, .2)


def test_digital_hand_payoffs_complement_and_immutability():
    terminal = np.array([0., 80., 100., 120., 150.])[::2]
    terminal.flags.writeable = False
    before = terminal.copy()
    call = DigitalOption(100., payout=7.)
    put = DigitalOption(100., 'put', 7.)
    assert_allclose(call.payoff(terminal), [0., 0., 7.])
    assert_allclose(put.payoff(terminal), [7., 7., 0.])
    assert_allclose(call.payoff(terminal)+put.payoff(terminal), 7.)
    assert np.array_equal(before, terminal)
    assert call.payoff(terminal).dtype == np.float64
    with assert_raises(FrozenInstanceError):
        call.payout = 2.


def test_digital_input_validation():
    for kwargs in (dict(strike=0), dict(strike=True), dict(strike=100, payout=-1),
                   dict(strike=100, payout=np.nan), dict(strike=100, option_type='x')):
        with assert_raises((TypeError, ValueError)):
            DigitalOption(**kwargs)
    for x in ([], [-1., 100.], [np.inf], [[100.]], [True], ['100']):
        with assert_raises((TypeError, ValueError)):
            DigitalOption(100).payoff(x)


def test_digital_price_boundaries_parity_and_delta_by_differentiation():
    for model, t in ((MODEL, 1.), (BlackScholesModel(80., -.02, .4), .3)):
        for kind in ('call', 'put'):
            h = .001
            derivative = (digital_price(replace(model, spot=model.spot+h), 100., t, kind, 3.)-
                          digital_price(replace(model, spot=model.spot-h), 100., t, kind, 3.))/(2*h)
            assert_allclose(derivative, digital_delta(model, 100., t, kind, 3.), rtol=2e-8)
        assert_allclose(digital_price(model, 100, t)+digital_price(model, 100, t, 'put'), np.exp(-model.rate*t))
        assert digital_delta(model, 100, t) == -digital_delta(model, 100, t, 'put')
    for t in (0., 1.):
        zero = BlackScholesModel(100., 0., 0.)
        assert digital_price(zero, 100., t) == 0.
        assert digital_price(zero, 100., t, 'put') == 1.
        with assert_raises(ValueError):
            digital_delta(zero, 100., t)
    assert digital_price(MODEL, 90., 0., payout=2.) == 2.
    for kwargs in (dict(maturity=-1), dict(payout=0), dict(strike=0), dict(model=object())):
        inputs=dict(model=MODEL, strike=100., maturity=1.); inputs.update(kwargs)
        with assert_raises((TypeError, ValueError)):
            digital_price(**inputs)


def test_spot_score_is_fixed_terminal_log_density_derivative():
    z = np.array([-3., -1., 0., .5, 2.])
    terminal = terminal_from_normals(MODEL, .7, z)
    h = .001
    def log_density(spot):
        mean = np.log(spot)+(MODEL.rate-.5*MODEL.volatility**2)*.7
        scale = MODEL.volatility*np.sqrt(.7)
        return norm.logpdf((np.log(terminal)-mean)/scale)-np.log(scale)-np.log(terminal)
    numerical = (log_density(100+h)-log_density(100-h))/(2*h)
    assert_allclose(gbm_spot_score(MODEL, .7, z), numerical, rtol=1e-8, atol=1e-10)
    assert_allclose(quad(lambda u: u/(100*.2)*norm.pdf(u), -10, 10)[0], 0, atol=1e-15)
    assert_allclose(quad(lambda u: (u/(100*.2))**2*norm.pdf(u), -10, 10)[0], 1/(100*.2)**2)


def test_lr_samples_and_discounted_statistics_by_hand():
    z = np.array([-2., -.1, 0., 1., 2.])
    terminal = terminal_from_normals(MODEL, 1., z)
    for product in (EuropeanOption(100), EuropeanOption(100, 'put'), DigitalOption(100, payout=4.), DigitalOption(100, 'put')):
        samples = product.payoff(terminal)*z/20
        result = likelihood_ratio_delta_from_normals(MODEL, product, 1., z, confidence_level=.9)
        discounted = np.exp(-.05)*samples
        assert_allclose(likelihood_ratio_delta_samples(MODEL, product, 1., z), samples)
        assert_allclose(result.estimate, discounted.mean())
        assert_allclose(result.sample_variance, discounted.var(ddof=1))
        assert_allclose(result.standard_error, discounted.std(ddof=1)/np.sqrt(5))
        assert_allclose(result.confidence_interval, result.estimate+np.array([-1,1])*norm.ppf(.95)*result.standard_error)
        assert_allclose(result.estimator_variance, result.standard_error**2)
        assert result.n_terminal_evaluations == 5
        with assert_raises(FrozenInstanceError):
            result.estimate = 0.


def test_lr_quadrature_matches_call_put_and_digital_deltas():
    for s, r, vol, t in ((80., -.02, .35, .4), (100., .05, .2, 1.), (140., .01, .4, 2.)):
        model = BlackScholesModel(s, r, vol)
        a = vol*np.sqrt(t); drift = (r-.5*vol**2)*t
        c = (np.log(100/s)-drift)/a
        for cls in (EuropeanOption, DigitalOption):
            for kind in ('call', 'put'):
                product = cls(100, kind)
                lo, hi = (c, 12.) if kind == 'call' else (-12., c)
                integral = quad(lambda z: float(likelihood_ratio_delta_samples(model, product, t, [z])[0])*np.exp(-r*t)*norm.pdf(z), lo, hi, epsabs=1e-11)[0]
                exact = black_scholes_greeks(model, 100, t, kind).delta if cls is EuropeanOption else digital_delta(model, 100, t, kind)
                assert_allclose(integral, exact, rtol=2e-9, atol=1e-11)
                if cls is DigitalOption:
                    price = quad(lambda z: np.exp(-r*t)*norm.pdf(z), lo, hi, epsabs=1e-12)[0]
                    assert_allclose(price, digital_price(model, 100, t, kind), atol=1e-11)


def test_lr_stream_replay_readonly_views_and_continuation():
    rng = np.random.default_rng(141)
    result = likelihood_ratio_delta(MODEL, EuropeanOption(100), 1, 4096, rng=rng)
    replay = np.random.default_rng(141)
    z = replay.standard_normal(4096)
    expected = likelihood_ratio_delta_from_normals(MODEL, EuropeanOption(100), 1, z)
    assert result.estimate == expected.estimate and result.standard_error == expected.standard_error
    assert rng.standard_normal() == replay.standard_normal()
    view = z[::3]; before = view.copy(); view.flags.writeable = False
    samples = likelihood_ratio_delta_samples(MODEL, DigitalOption(100), 1, view)
    assert np.array_equal(before, view) and not np.shares_memory(samples, view)


def test_lr_configuration_rejected_before_rng_consumption():
    for changes in (dict(model=object()), dict(model=replace(MODEL, volatility=0)),
                    dict(maturity=0), dict(maturity=np.inf), dict(n_paths=1),
                    dict(n_paths=True), dict(product=ArithmeticAsianOption(100)),
                    dict(confidence_level=1)):
        rng = np.random.default_rng(142)
        args = dict(model=MODEL, product=EuropeanOption(100), maturity=1, n_paths=8, rng=rng)
        args.update(changes)
        with assert_raises((ValueError, TypeError)):
            likelihood_ratio_delta(**args)
        assert rng.standard_normal() == np.random.default_rng(142).standard_normal()
    with assert_raises(TypeError):
        likelihood_ratio_delta(MODEL, EuropeanOption(100), 1, 8, rng=42)
    for z in ([], [1.], [[1., 2.]], [np.nan, 1], [True, False], ['1', '2']):
        with assert_raises((ValueError, TypeError)):
            likelihood_ratio_delta_from_normals(MODEL, EuropeanOption(100), 1, z)
    with assert_raises(FloatingPointError):
        gbm_spot_score(BlackScholesModel(1e-300, 0., 1e-300), 1, [1.])


def test_lr_zero_observed_events_does_not_establish_zero_delta():
    result = likelihood_ratio_delta_from_normals(MODEL, DigitalOption(100), 1, [-2., -1.])
    assert result.estimate == result.standard_error == 0.
    assert digital_delta(MODEL, 100, 1) > 0
    # Differentiating the digital indicator along a path would be zero a.s.
    with assert_raises(TypeError):
        pathwise_delta(MODEL, DigitalOption(100), 1, 8, rng=np.random.default_rng(1))


def test_digital_engine_plain_antithetic_and_fd_routes():
    product = DigitalOption(100, payout=2.)
    n = 1000
    for method in ('none', 'antithetic'):
        rng = np.random.default_rng(143)
        result = price_monte_carlo(MODEL, product, 1, n, rng=rng, variance_reduction=method)
        replay = np.random.default_rng(143)
        z = replay.standard_normal(n if method == 'none' else n//2)
        x = product.payoff(terminal_from_normals(MODEL, 1, z))
        if method == 'antithetic':
            x = (x+product.payoff(terminal_from_normals(MODEL, 1, -z)))/2
        assert_allclose(result.price, np.exp(-.05)*x.mean())
        assert_allclose(result.estimator_variance, np.exp(-.1)*x.var(ddof=1)/len(x))
        assert rng.standard_normal() == replay.standard_normal()
    terminal = np.array([80, 100, 120.])
    assert_allclose(price_from_terminal_samples(product, terminal, rate=0., maturity=0.).price, 2/3)
    for greek in ('delta', 'gamma', 'vega'):
        h = .01 if greek == 'vega' else 1.
        result = finite_difference_greek(MODEL, product, 1, n, greek=greek, bump=h, rng=np.random.default_rng(144))
        z = np.random.default_rng(144).standard_normal(n)
        replay = finite_difference_from_normals(MODEL, product, 1, z, greek=greek, bump=h)
        assert result.estimate == replay.estimate and result.n_steps is None
    with assert_raises(ValueError):
        price_monte_carlo(MODEL, product, 1, n, rng=np.random.default_rng(1), n_steps=12)
    with assert_raises(ValueError):
        finite_difference_greek(MODEL, product, 1, n, greek='delta', bump=1, rng=np.random.default_rng(1), n_steps=12)
