"""Independent European Heston reference by affine transform and quadrature.

Numerical Fourier inversion is a deterministic benchmark, not an exact float64
oracle. Cutoff sensitivity and integration error must be inspected. No external
quant-finance library is used. See docs for the Riccati derivation and sources.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from mc_pricer._validation import finite_real
from mc_pricer.models.heston import HestonModel
from mc_pricer.models.black_scholes import BlackScholesModel
from mc_pricer.pricing.analytical import black_scholes_price, _validate_contract, OptionType


@dataclass(frozen=True)
class HestonReferenceResult:
    price: float
    quadrature_error_estimate: float
    cutoff: float
    method: str


def deterministic_integrated_variance(model: HestonModel, maturity: float) -> float:
    """Integral of v(t)=theta+(v0-theta)exp(-kappa*t); requires xi=0."""
    if not isinstance(model, HestonModel):
        raise TypeError('requires a HestonModel')
    time = finite_real('maturity', maturity)
    if time < 0 or model.xi != 0:
        raise ValueError('requires nonnegative maturity and xi=0')
    return float(model.theta*time+(model.v0-model.theta)*(-np.expm1(-model.kappa*time))/model.kappa)


def heston_centered_characteristic(model: HestonModel, maturity: float, argument: complex) -> complex:
    """E[exp(i*u*log(ST/(S0*exp(rT))))], scalar u in strip Im(u) in [-.5,0].

    Restricted strip suffices for real-axis tests and Lewis inversion; arbitrary
    complex arguments can require moment/explosion analysis and are rejected.
    The xi=0 limit is exact deterministic integrated variance. For xi>0 the
    decaying-exponential affine form uses the principal root with Re(d)>=0.
    """
    if not isinstance(model, HestonModel):
        raise TypeError('requires a HestonModel')
    time = finite_real('maturity', maturity)
    if time < 0:
        raise ValueError('maturity must be nonnegative')
    if isinstance(argument, (bool, str)) or not np.isscalar(argument):
        raise TypeError('argument must be a real or complex scalar')
    u = complex(argument)
    if not np.isfinite(u.real) or not np.isfinite(u.imag) or not -.5 <= u.imag <= 0:
        raise ValueError('argument must be finite with imaginary part in [-.5,0]')
    if time == 0 or u == 0:
        return 1.+0j
    if model.xi == 0:
        integrated = deterministic_integrated_variance(model,time)
        return complex(np.exp(-.5*(u*u+1j*u)*integrated))
    if abs(model.rho) == 1:
        raise ValueError('Fourier reference requires abs(rho)<1; simulator supports endpoints')
    k, theta, xi, rho = model.kappa, model.theta, model.xi, model.rho
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        b = k-rho*xi*1j*u
        q = u*u+1j*u
        d = np.sqrt(b*b+xi*xi*q)
        # Stable b-d via rationalization; avoids small-xi cancellation.
        minus = -xi*xi*q/(b+d)
        g = minus/(b+d)
        decay = np.exp(-d*time)
        one_minus_decay = -np.expm1(-d*time)
        # log((1-g*decay)/(1-g)) = log(1+w). Explicit real/imaginary
        # parts preserve tiny real terms that some complex log1p routines lose.
        w = g*one_minus_decay/(1-g)
        logarithm = .5*np.log1p(2*w.real+w.real*w.real+w.imag*w.imag)+1j*np.arctan2(w.imag,1+w.real)
        c = k*theta/(xi*xi)*(minus*time-2*logarithm)
        coefficient = minus/(xi*xi)*one_minus_decay/(1-g*decay)
        value = np.exp(c+coefficient*model.v0)
    if not np.isfinite(value):
        raise FloatingPointError('characteristic function exceeds finite float64 range')
    return complex(value)


def heston_european_reference(model: HestonModel, strike: float, maturity: float,
                              option_type: OptionType = 'call', *, cutoff: float = 200.,
                              tolerance: float = 1e-9) -> HestonReferenceResult:
    """Lewis inversion over [0,cutoff], put via parity; no MC/time stepping.

    quadrature_error_estimate is QUADPACK's scaled finite-interval estimate,
    NOT a tail bound or a confidence interval. Compare larger cutoffs explicitly.
    Raises on integration warnings or materially violated vanilla price bounds;
    never clips an inaccurate reference into its no-arbitrage bounds.
    """
    if not isinstance(model, HestonModel):
        raise TypeError('requires a HestonModel')
    strike, time = _validate_contract(strike,maturity,option_type)
    limit = finite_real('cutoff',cutoff)
    tol = finite_real('tolerance',tolerance)
    if limit <= 0 or tol <= 0:
        raise ValueError('cutoff and tolerance must be positive')
    if time == 0 or model.xi == 0 or (model.v0==0 and model.theta==0):
        integrated = 0. if time==0 else deterministic_integrated_variance(HestonModel(model.spot,model.rate,model.v0,model.kappa,model.theta,0.,model.rho),time)
        vol = np.sqrt(integrated/time) if time>0 else 0.
        value = black_scholes_price(BlackScholesModel(model.spot,model.rate,vol),strike,time,option_type)
        return HestonReferenceResult(value,0.,limit,'deterministic_variance')
    if abs(model.rho)==1:
        raise ValueError('Fourier reference requires abs(rho)<1')
    log_forward_moneyness = np.log(model.spot)-np.log(strike)+model.rate*time
    factor = np.sqrt(model.spot)*np.sqrt(strike)*np.exp(-.5*model.rate*time)/np.pi
    def integrand(u: float) -> float:
        phi = heston_centered_characteristic(model,time,u-.5j)
        return float(np.real(np.exp(1j*u*log_forward_moneyness)*phi)/(u*u+.25))
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        integral,error = quad(integrand,0.,limit,epsabs=tol,epsrel=tol,limit=400)
    call = model.spot-factor*integral
    discounted_strike = strike*np.exp(-model.rate*time)
    price = call if option_type=='call' else call-model.spot+discounted_strike
    lower = max(model.spot-discounted_strike,0.) if option_type=='call' else max(discounted_strike-model.spot,0.)
    upper = model.spot if option_type=='call' else discounted_strike
    allowance = max(1e-8,10*factor*error)
    if not np.isfinite(price) or price < lower-allowance or price > upper+allowance:
        raise FloatingPointError('Fourier reference violates price bounds; inspect cutoff/parameters')
    return HestonReferenceResult(float(price),float(factor*error),limit,'lewis_affine_quadrature')
