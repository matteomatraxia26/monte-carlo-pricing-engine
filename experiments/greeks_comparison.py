"""M11: central FD Greeks, CRN versus independent legs, bumps and Delta convergence.

The benchmark is a European call under GBM. Later milestones add pathwise/LR
estimators. Full-pricer timings include RNG draws but exclude RNG construction.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import f

from mc_pricer import (BlackScholesModel, EuropeanOption, black_scholes_greeks,
                       black_scholes_price, finite_difference_greek)
from mc_pricer.statistics.confidence_intervals import binomial_wilson_interval
from mc_pricer.statistics.diagnostics import fit_power_law, summarize_replications

MODEL = BlackScholesModel(100., .05, .2)
PRODUCT = EuropeanOption(100.)
GREEKS = ('delta', 'gamma', 'vega')
SPOT_BUMPS = (10., 5., 1., .5, .1, .01)
VOL_BUMPS = (.1, .05, .01, .005, .001, .0001)
PATH_COUNTS = (1000, 3000, 10000, 30000, 100000)
METHODS = ('crn', 'independent')


def stream(seed: int, key: tuple[int, ...]) -> np.random.Generator:
    """Each study/Greek/bump/run/method cell has a distinct recorded stream."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def analytical_fd(greek: str, bump: float) -> float:
    """Exact-price central difference; float64 cancellation matters for tiny h."""
    field = 'volatility' if greek == 'vega' else 'spot'
    center = getattr(MODEL, field)
    up = black_scholes_price(replace(MODEL, **{field: center+bump}), 100., 1.)
    down = black_scholes_price(replace(MODEL, **{field: center-bump}), 100., 1.)
    if greek == 'gamma':
        base = black_scholes_price(MODEL, 100., 1.)
        return ((up-base)+(down-base))/bump**2
    return (up-down)/(2*bump)


def _validate_design(n_paths: int, replications: int, seed: int) -> None:
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (n_paths, replications, seed)):
        raise TypeError('counts and seed must be integers')
    if n_paths < 2 or replications < 2 or seed < 0:
        raise ValueError('require paths>=2, replications>=2, seed>=0')


def _collect_cell(greek: str, bump: float, n_paths: int, replications: int,
                  seed: int, prefix: tuple[int, ...]) -> list[dict]:
    target = getattr(black_scholes_greeks(MODEL, 100., 1.), greek)
    fd_target = analytical_fd(greek, bump)
    rows = []
    # Separate warmup namespace. At fixed Greek, both methods value equal numbers of payoffs.
    for method_index in range(2):
        finite_difference_greek(MODEL, PRODUCT, 1., 128, greek=greek, bump=bump,
                                rng=stream(seed, (99,)+prefix+(method_index,)),
                                common_random_numbers=method_index == 0)
    for replication in range(replications):
        order = (0,1) if replication % 2 == 0 else (1,0)
        for method_index in order:
            key = prefix+(replication,method_index)
            result = finite_difference_greek(MODEL, PRODUCT, 1., n_paths, greek=greek, bump=bump,
                                            rng=stream(seed, key), common_random_numbers=method_index == 0)
            row = asdict(result)
            low, high = row.pop('confidence_interval')
            row.update(method=METHODS[method_index], replication=replication,
                       spawn_key=json.dumps(key), analytical_value=target, finite_bump_target=fd_target,
                       finite_difference_bias=fd_target-target, absolute_error=abs(result.estimate-target),
                       ci_lower=low, ci_upper=high, estimator_variance=result.estimator_variance)
            rows.append(row)
    return rows


def collect_bump_study(n_paths: int, replications: int, seed: int) -> pd.DataFrame:
    """Independent cells across bumps as well as methods and replications."""
    _validate_design(n_paths, replications, seed)
    rows = []
    for gi, greek in enumerate(GREEKS):
        bumps = VOL_BUMPS if greek == 'vega' else SPOT_BUMPS
        for hi, bump in enumerate(bumps):
            rows.extend(_collect_cell(greek, bump, n_paths, replications, seed, (0,gi,hi)))
    return pd.DataFrame(rows)


def collect_delta_convergence(replications: int, seed: int,
                              path_counts: tuple[int, ...] = PATH_COUNTS) -> pd.DataFrame:
    if len(path_counts) < 3 or any(a>=b for a,b in zip(path_counts,path_counts[1:])):
        raise ValueError('require at least three increasing path counts')
    rows = []
    for index, count in enumerate(path_counts):
        _validate_design(count, replications, seed)
        rows.extend(_collect_cell('delta', 1., count, replications, seed, (1,index)))
    return pd.DataFrame(rows)


def summarize_greeks(raw: pd.DataFrame) -> pd.DataFrame:
    """Independent-run bias/variance/RMSE and coverage of two distinct targets."""
    rows = []
    for (greek,bump,n_paths,method), group in raw.groupby(['greek','bump','n_paths','method'],sort=False):
        if len(group)<2 or group.analytical_value.nunique()!=1 or group.finite_bump_target.nunique()!=1:
            raise ValueError('each cell needs repeated estimates with identical targets')
        target = group.analytical_value.iloc[0]
        fd_target = group.finite_bump_target.iloc[0]
        stats = summarize_replications(group.estimate.to_numpy(), target)
        row = dict(greek=greek,bump=bump,n_paths=n_paths,method=method,
                   n_replications=len(group),mean_estimate=stats.mean_estimate,
                   analytical_value=target,finite_bump_target=fd_target,
                   absolute_error_of_mean=abs(stats.empirical_bias),empirical_bias=stats.empirical_bias,
                   deterministic_fd_bias=fd_target-target,mc_error_of_mean=stats.mean_estimate-fd_target,
                   empirical_variance=stats.empirical_variance,rmse=stats.rmse,
                   empirical_standard_error_of_mean=stats.standard_error_of_mean,
                   bias_ci_lower=stats.bias_confidence_interval[0],bias_ci_upper=stats.bias_confidence_interval[1],
                   mean_reported_estimator_variance=group.estimator_variance.mean(),
                   rms_reported_standard_error=np.sqrt(group.estimator_variance.mean()),
                   mean_runtime_seconds=group.runtime_seconds.mean(),
                   n_payoff_evaluations=int(group.n_payoff_evaluations.iloc[0]),
                   zero_se_runs=int((group.standard_error==0).sum()))
        # Empirical MSE uses R denominator; unbiased variance uses R-1.
        assert np.isclose(stats.rmse**2,stats.empirical_bias**2+(len(group)-1)/len(group)*stats.empirical_variance)
        for name,value in (('greek',target),('finite_bump',fd_target)):
            hits=int(((group.ci_lower<=value)&(value<=group.ci_upper)).sum())
            low,high=binomial_wilson_interval(hits,len(group))
            row.update({f'{name}_coverage':hits/len(group),f'{name}_coverage_ci_lower':low,
                        f'{name}_coverage_ci_upper':high})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_methods(summary: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (greek,bump,n_paths),group in summary.groupby(['greek','bump','n_paths'],sort=False):
        if set(group.method)!=set(METHODS) or group.n_payoff_evaluations.nunique()!=1:
            raise ValueError('need both methods at equal payoff-evaluation budgets')
        crn=group.loc[group.method=='crn'].iloc[0]
        independent=group.loc[group.method=='independent'].iloc[0]
        ratio=(independent.empirical_variance/crn.empirical_variance
               if min(independent.empirical_variance,crn.empirical_variance)>0 else np.nan)
        rows.append(dict(greek=greek,bump=bump,n_paths=n_paths,empirical_vrf=ratio,
                         vrf_ci_lower=ratio/f.ppf(.975,independent.n_replications-1,crn.n_replications-1),
                         vrf_ci_upper=ratio/f.ppf(.025,independent.n_replications-1,crn.n_replications-1),
                         reported_variance_vrf=independent.mean_reported_estimator_variance/crn.mean_reported_estimator_variance
                         if crn.mean_reported_estimator_variance>0 else np.nan,
                         time_adjusted_efficiency=ratio*independent.mean_runtime_seconds/crn.mean_runtime_seconds))
    return pd.DataFrame(rows)


def cancellation_diagnostic() -> pd.DataFrame:
    """Deterministic Gamma difference of analytical prices, isolating roundoff.

    This is not Monte Carlo data and tiny-h values are not reference targets.
    """
    true=black_scholes_greeks(MODEL,100.,1.).gamma
    return pd.DataFrame([dict(bump=h,finite_difference=analytical_fd('gamma',h),
                              analytical_gamma=true,absolute_error=abs(analytical_fd('gamma',h)-true))
                         for h in (10.,1.,.1,.01,.001,.0001,.00001,.000001,.0000001)])


def make_figures(summary: pd.DataFrame, convergence: pd.DataFrame, cancellation: pd.DataFrame,
                 folder: Path) -> None:
    colors={'crn':'#008B7D','independent':'#C97A35'}
    labels={'crn':'Common random numbers','independent':'Independent legs'}
    units={'delta':'price / spot','gamma':'price / spot²','vega':'price / unit volatility'}
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,greek in zip(axes,GREEKS):
        for method in METHODS:
            data=summary.loc[(summary.greek==greek)&(summary.method==method)].sort_values('bump')
            ax.loglog(data.bump,data.rmse,'o-',label=labels[method],color=colors[method])
        data=summary.loc[(summary.greek==greek)&(summary.method=='crn')].sort_values('bump')
        ax.loglog(data.bump,np.abs(data.deterministic_fd_bias),'k--',label='Exact-price FD bias')
        ax.set(title=greek.capitalize(),xlabel='Volatility bump (decimal)' if greek=='vega' else 'Spot bump (currency)',
               ylabel=f'RMSE or absolute bias ({units[greek]})')
        ax.grid(alpha=.2,which='both')
    axes[0].legend(fontsize=8)
    row=summary.iloc[0]
    fig.suptitle(f"Bump-size trade-off • {int(row.n_replications)} independent runs • N={int(row.n_paths):,} rows/run")
    fig.savefig(folder/'greek_error_vs_bump.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for method in METHODS:
        data=convergence.loc[convergence.method==method].sort_values('n_paths')
        axes[0].errorbar(data.n_paths,data.mean_estimate,yerr=1.96*data.empirical_standard_error_of_mean,
                         fmt='o-',capsize=3,label=labels[method],color=colors[method])
        axes[1].loglog(data.n_paths,data.rms_reported_standard_error,'o-',label=labels[method],color=colors[method])
    target=convergence.analytical_value.iloc[0]
    axes[0].axhline(target,color='black',linestyle='--',label='Analytical Delta')
    axes[0].set_xscale('log')
    axes[0].set(title='Delta mean ± 95% interval across replications',ylabel='Delta (price / spot)')
    axes[1].set(title='Sampling SE of one run; fixed spot bump h=1',ylabel='RMS reported Delta SE (price / spot)')
    for ax in axes:
        ax.set_xlabel('Independent shock rows N')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(folder/'delta_fd_convergence.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4),constrained_layout=True)
    ax.loglog(cancellation.bump,cancellation.absolute_error,'o-',color='#6656A5',label='Float64 central Gamma from exact prices')
    ax.set(title='Tiny bumps expose floating-point cancellation',xlabel='Spot bump (currency)',
           ylabel='Absolute Gamma error (price / spot²)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2,which='both')
    fig.savefig(folder/'gamma_float64_cancellation.png',dpi=160)
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-paths',type=int,default=20000)
    parser.add_argument('--replications',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261001)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    args=parser.parse_args()
    for name in ('raw','tables','figures'):
        (args.output_dir/name).mkdir(parents=True,exist_ok=True)
    raw=collect_bump_study(args.n_paths,args.replications,args.seed)
    convergence_raw=collect_delta_convergence(args.replications,args.seed)
    summary=summarize_greeks(raw)
    convergence=summarize_greeks(convergence_raw)
    comparisons=compare_methods(summary)
    cancellation=cancellation_diagnostic()
    baseline=summary.loc[((summary.greek.isin(('delta','gamma')))&(summary.bump==1.))|
                         ((summary.greek=='vega')&(summary.bump==.01))].copy()
    fits=[]
    for method in METHODS:
        data=convergence.loc[convergence.method==method].sort_values('n_paths')
        fit=fit_power_law(data.n_paths.to_numpy(),data.rms_reported_standard_error.to_numpy())
        fits.append(dict(method=method,**asdict(fit)))
    artifacts=[]
    for folder,name,frame in (('raw','fd_greek_replications',raw),('raw','fd_delta_convergence_replications',convergence_raw),
                              ('tables','fd_greek_bumps',summary),('tables','fd_greek_comparison',comparisons),
                              ('tables','fd_greek_baseline',baseline),('tables','fd_delta_convergence',convergence),
                              ('tables','fd_delta_se_slopes',pd.DataFrame(fits)),('tables','fd_gamma_cancellation',cancellation)):
        path=args.output_dir/folder/f'{name}.csv'
        frame.to_csv(path,index=False)
        artifacts.append(path)
    make_figures(summary,convergence,cancellation,args.output_dir/'figures')
    metadata=dict(milestone=11,model=asdict(MODEL),product=asdict(PRODUCT),maturity=1.,
                  n_paths=args.n_paths,replications=args.replications,seed=args.seed,
                  bit_generator='PCG64',spot_bumps=SPOT_BUMPS,volatility_bumps=VOL_BUMPS,
                  delta_convergence_counts=PATH_COUNTS,delta_convergence_bump=1.,
                  stream_design='bump study (0,Greek,bump,replication,method); convergence (1,count,replication,method); warmup prefix 99',
                  methods=METHODS,budget='N independent coupled rows; 2N payoff evaluations for Delta/Vega, 3N for Gamma; equal between methods',
                  timing='full Greek call including shock draws and all valuations; excludes Generator construction, warmups, analytical benchmarks and output',
                  coverage='finite-bump target and true Greek reported separately; normal CI is only a sampling interval',
                  limits='no pathwise/LR estimator; no optimization of bump choice; tiny-h analytical Gamma diagnostic is not a reliable target',
                  versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
                  sha256={str(p.relative_to(args.output_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    (args.output_dir/'raw/fd_greek_study.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(baseline[['greek','method','bump','mean_estimate','analytical_value','empirical_bias','rms_reported_standard_error','rmse','mean_runtime_seconds']].to_string(index=False))
    print(comparisons.to_string(index=False))
    print(pd.DataFrame(fits).to_string(index=False))
    print(cancellation.to_string(index=False))


if __name__=='__main__':
    main()
