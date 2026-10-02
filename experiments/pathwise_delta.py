"""M12: pathwise versus CRN finite-difference Delta, explicit budgets and pairing."""

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
from scipy.stats import f, norm

from mc_pricer import (
    BlackScholesModel, EuropeanOption, black_scholes_greeks, black_scholes_price,
    central_first_difference, european_pathwise_delta_samples, finite_difference_greek,
    pathwise_delta, price_from_payoff_samples, terminal_from_normals,
)
from mc_pricer.statistics.confidence_intervals import binomial_wilson_interval
from mc_pricer.statistics.diagnostics import fit_power_law, summarize_replications

MODEL=BlackScholesModel(100.,.05,.2)
PRODUCT=EuropeanOption(100.)
METHODS=(('pathwise',None),('fd_h10',10.),('fd_h1',1.),('fd_h0.1',.1))
BUDGETS=('equal_rows','equal_terminal_evaluations')
COUNTS=(1000,3000,10000,30000,100000)
PAIRED_BUMPS=(10.,5.,1.,.5,.1,.01)


def stream(seed: int, key: tuple[int,...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed,spawn_key=key))


def validate_design(n_paths: int, replications: int, seed: int) -> None:
    if any(isinstance(x,bool) or not isinstance(x,Integral) for x in (n_paths,replications,seed)):
        raise TypeError('counts and seed must be integers')
    if n_paths<2 or replications<2 or seed<0:
        raise ValueError('require paths>=2, replications>=2 and seed>=0')


def exact_targets(bump: float | None) -> tuple[float,float]:
    delta=black_scholes_greeks(MODEL,PRODUCT.strike,1.).delta
    if bump is None:
        return delta,delta
    up=black_scholes_price(replace(MODEL,spot=MODEL.spot+bump),100.,1.)
    down=black_scholes_price(replace(MODEL,spot=MODEL.spot-bump),100.,1.)
    return delta,(up-down)/(2*bump)


def population_pathwise_variance() -> float:
    """Exact discounted call-Delta sample variance, from truncated lognormal moments."""
    delta=black_scholes_greeks(MODEL,100.,1.).delta
    d1=(np.log(MODEL.spot/100.)+MODEL.rate+.5*MODEL.volatility**2)/MODEL.volatility
    return float(np.exp(MODEL.volatility**2)*norm.cdf(d1+MODEL.volatility)-delta**2)


def _estimate(method_index: int, count: int, rng: np.random.Generator):
    _,bump=METHODS[method_index]
    if bump is None:
        return pathwise_delta(MODEL,PRODUCT,1.,count,rng=rng)
    return finite_difference_greek(MODEL,PRODUCT,1.,count,greek='delta',bump=bump,rng=rng)


def _record(method_index: int, count: int, replication: int, seed: int,
            key: tuple[int,...], budget: str) -> dict:
    method,bump=METHODS[method_index]
    delta,target=exact_targets(bump)
    result=_estimate(method_index,count,stream(seed,key))
    return dict(budget=budget,method=method,bump=bump,replication=replication,spawn_key=json.dumps(key),
                n_paths=count,n_terminal_evaluations=count*(1 if bump is None else 2),
                estimate=result.estimate,standard_error=result.standard_error,
                sample_variance=result.sample_variance,estimator_variance=result.estimator_variance,
                ci_lower=result.confidence_interval[0],ci_upper=result.confidence_interval[1],
                runtime_seconds=result.runtime_seconds,analytical_delta=delta,estimator_target=target,
                deterministic_bias=target-delta,absolute_error=abs(result.estimate-delta))


def collect_comparison(n_paths: int, replications: int, seed: int) -> pd.DataFrame:
    """Independent method cells; compare equal rows and equal terminal evaluations.

    Equal rows: PW N, FD N coupled rows, so costs N vs 2N terminal states.
    Equal terminal evaluations: PW 2N, FD N coupled rows, each costing 2N states.
    State count is an explicit proxy, not an assertion of equal wall-clock cost.
    """
    validate_design(n_paths,replications,seed)
    rows=[]
    for bi,budget in enumerate(BUDGETS):
        for mi in range(len(METHODS)):
            _estimate(mi,128,stream(seed,(99,0,bi,mi)))
        for replication in range(replications):
            for method_index in np.roll(np.arange(len(METHODS)),replication%len(METHODS)):
                mi=int(method_index)
                count=2*n_paths if bi==1 and mi==0 else n_paths
                rows.append(_record(mi,count,replication,seed,(0,bi,replication,mi),budget))
    return pd.DataFrame(rows)


def collect_convergence(replications: int, seed: int, counts: tuple[int,...]=COUNTS) -> pd.DataFrame:
    if len(counts)<3 or any(a>=b for a,b in zip(counts,counts[1:])):
        raise ValueError('require at least three increasing counts')
    for count in counts:
        validate_design(count,replications,seed)
    rows=[]
    for ci,count in enumerate(counts):
        for replication in range(replications):
            for mi in ((0,2) if replication%2==0 else (2,0)):
                rows.append(_record(mi,count,replication,seed,(1,ci,replication,mi),'equal_rows'))
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (budget,method,count),group in raw.groupby(['budget','method','n_paths'],sort=False):
        if len(group)<2 or group.estimator_target.nunique()!=1 or group.n_terminal_evaluations.nunique()!=1:
            raise ValueError('need repeated identical configurations')
        target=group.analytical_delta.iloc[0]
        stats=summarize_replications(group.estimate.to_numpy(),target)
        row=dict(budget=budget,method=method,n_paths=count,n_replications=len(group),bump=group.bump.iloc[0],
                 n_terminal_evaluations=int(group.n_terminal_evaluations.iloc[0]),
                 mean_estimate=stats.mean_estimate,analytical_delta=target,
                 estimator_target=group.estimator_target.iloc[0],empirical_bias=stats.empirical_bias,
                 deterministic_bias=group.deterministic_bias.iloc[0],absolute_error_of_mean=abs(stats.empirical_bias),
                 empirical_variance=stats.empirical_variance,rmse=stats.rmse,
                 mean_standard_error=group.standard_error.mean(),
                 rms_reported_standard_error=np.sqrt(group.estimator_variance.mean()),
                 mean_reported_estimator_variance=group.estimator_variance.mean(),
                 empirical_standard_error_of_mean=stats.standard_error_of_mean,
                 bias_ci_lower=stats.bias_confidence_interval[0],bias_ci_upper=stats.bias_confidence_interval[1],
                 mean_runtime_seconds=group.runtime_seconds.mean())
        for name,value in (('delta',target),('estimator',row['estimator_target'])):
            hits=int(((group.ci_lower<=value)&(value<=group.ci_upper)).sum())
            low,high=binomial_wilson_interval(hits,len(group))
            row.update({f'{name}_coverage':hits/len(group),f'{name}_coverage_ci_lower':low,f'{name}_coverage_ci_upper':high})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_variance(summary: pd.DataFrame) -> pd.DataFrame:
    """Independent-method F ratio intervals: FD variance / PW variance."""
    rows=[]
    for budget,group in summary.groupby('budget',sort=False):
        pw=group.loc[group.method=='pathwise']
        if len(pw)!=1 or set(group.method)!={x[0] for x in METHODS}:
            raise ValueError('one full four-method comparison required per budget')
        if budget=='equal_rows' and group.n_paths.nunique()!=1:
            raise ValueError('equal-row comparison requires common N')
        if budget=='equal_terminal_evaluations' and group.n_terminal_evaluations.nunique()!=1:
            raise ValueError('equal-state comparison requires common terminal-evaluation budget')
        pw=pw.iloc[0]
        for fd in group.loc[group.method!='pathwise'].itertuples():
            ratio=fd.empirical_variance/pw.empirical_variance if min(fd.empirical_variance,pw.empirical_variance)>0 else np.nan
            rows.append(dict(budget=budget,method=fd.method,fd_over_pw_variance=ratio,
                             ratio_ci_lower=ratio/f.ppf(.975,fd.n_replications-1,pw.n_replications-1),
                             ratio_ci_upper=ratio/f.ppf(.025,fd.n_replications-1,pw.n_replications-1),
                             reported_variance_ratio=(fd.mean_reported_estimator_variance/pw.mean_reported_estimator_variance
                                                      if pw.mean_reported_estimator_variance>0 else np.nan),
                             time_adjusted_efficiency=ratio*fd.mean_runtime_seconds/pw.mean_runtime_seconds))
    return pd.DataFrame(rows)


def paired_samples(normals: np.ndarray, bump: float) -> tuple[np.ndarray,np.ndarray]:
    """Undiscounted PW and FD Delta samples from the same shock rows; no RNG."""
    terminal=terminal_from_normals(MODEL,1.,normals)
    pw=european_pathwise_delta_samples(PRODUCT,terminal,spot=MODEL.spot)
    up=PRODUCT.payoff(terminal_from_normals(replace(MODEL,spot=MODEL.spot+bump),1.,normals))
    down=PRODUCT.payoff(terminal_from_normals(replace(MODEL,spot=MODEL.spot-bump),1.,normals))
    return pw,central_first_difference(up,down,bump)


def collect_paired(n_paths: int, replications: int, seed: int) -> pd.DataFrame:
    """Same draws within each replication across methods/bumps; do not pool bumps.

    This diagnoses FD-minus-PW differences with covariance-aware uncertainty.
    It is separate from independent-method variance-ratio experiments.
    """
    validate_design(n_paths,replications,seed)
    rows=[]
    for replication in range(replications):
        key=(2,replication)
        z=stream(seed,key).standard_normal(n_paths)
        for h in PAIRED_BUMPS:
            pw,fd=paired_samples(z,h)
            delta,target=exact_targets(h)
            stats=price_from_payoff_samples(fd-pw,rate=MODEL.rate,maturity=1.)
            rows.append(dict(replication=replication,bump=h,n_paths=n_paths,spawn_key=json.dumps(key),
                             fd_minus_pw=stats.price,paired_standard_error=stats.standard_error,
                             analytical_difference=target-delta,
                             ci_lower=stats.confidence_interval[0],ci_upper=stats.confidence_interval[1],
                             sample_correlation=np.corrcoef(pw,fd)[0,1]))
    return pd.DataFrame(rows)


def summarize_paired(raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for h,group in raw.groupby('bump',sort=False):
        target=group.analytical_difference.iloc[0]
        stats=summarize_replications(group.fd_minus_pw.to_numpy(),target)
        rows.append(dict(bump=h,mean_difference=stats.mean_estimate,analytical_difference=target,
                         empirical_standard_error_of_mean=stats.standard_error_of_mean,
                         difference_ci_lower=stats.mean_estimate-1.96*stats.standard_error_of_mean,
                         difference_ci_upper=stats.mean_estimate+1.96*stats.standard_error_of_mean,
                         rms_paired_standard_error=np.sqrt(np.mean(group.paired_standard_error**2)),
                         mean_correlation=group.sample_correlation.mean()))
    return pd.DataFrame(rows)


def figures(summary: pd.DataFrame, convergence: pd.DataFrame, paired: pd.DataFrame, folder: Path) -> None:
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    names=[x[0] for x in METHODS]
    x=np.arange(4)
    for bi,budget in enumerate(BUDGETS):
        data=summary.loc[summary.budget==budget].set_index('method').loc[names]
        offset=(bi-.5)*.2
        label='Equal sample rows' if bi==0 else 'Equal terminal evaluations'
        axes[0].errorbar(x+offset,data.mean_estimate,yerr=1.96*data.empirical_standard_error_of_mean,
                         fmt='o',capsize=4,label=label)
        axes[1].bar(x+offset,data.empirical_variance,width=.19,label=label)
    axes[0].axhline(exact_targets(None)[0],color='black',linestyle='--',label='Analytical Delta')
    axes[0].set(title='Mean Delta ± 95% interval across replications',ylabel='Delta (price / spot)')
    axes[1].set(title='Variance across independent run estimates',ylabel='Estimator variance (Delta²)')
    for ax in axes:
        ax.set_xticks(x,['Pathwise','FD h=10','FD h=1','FD h=0.1'])
        ax.set_xlabel('Estimator (FD uses common random numbers)')
        ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=8)
    base=summary.loc[summary.budget=='equal_rows'].iloc[0]
    fig.suptitle(f"{int(base.n_replications)} runs • base N={int(base.n_paths):,} • equal-evaluation budget {2*int(base.n_paths):,} terminal states")
    fig.savefig(folder/'finite_difference_vs_pathwise_delta.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    for method in ('pathwise','fd_h1'):
        data=convergence.loc[convergence.method==method].sort_values('n_paths')
        axes[0].errorbar(data.n_paths,data.mean_estimate,yerr=1.96*data.empirical_standard_error_of_mean,
                         fmt='o-',capsize=3,label=method)
        axes[1].loglog(data.n_paths,data.rms_reported_standard_error,'o-',label=method)
    data=convergence.loc[convergence.method=='pathwise'].sort_values('n_paths')
    axes[1].loglog(data.n_paths,np.sqrt(population_pathwise_variance()/data.n_paths),'k--',label='Exact pathwise SE')
    axes[0].set_xscale('log')
    axes[0].axhline(exact_targets(None)[0],color='black',linestyle='--',label='Analytical Delta')
    axes[0].set(title='Mean Delta ± 95% interval across replications',ylabel='Delta (price / spot)')
    axes[1].set(title='Single-run sampling SE versus N',ylabel='RMS reported SE (Delta units)')
    for ax in axes:
        ax.set_xlabel('Independent sample rows N')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(folder/'pathwise_delta_convergence.png',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.2),constrained_layout=True)
    data=paired.sort_values('bump')
    ax.errorbar(data.bump,data.mean_difference,yerr=1.96*data.empirical_standard_error_of_mean,
                 fmt='o-',capsize=4,label='Paired mean difference ± 95% interval')
    ax.plot(data.bump,data.analytical_difference,'k--',label='Exact FD bias')
    ax.set_xscale('log')
    ax.set_yscale('symlog',linthresh=1e-5)
    ax.set_ylim(min(data.difference_ci_lower.min(),data.analytical_difference.min())*1.35,
                max(data.difference_ci_upper.max(),1e-5)*1.7)
    ax.axhline(0,color='grey',linewidth=.8)
    ax.set(title='Common-path FD minus pathwise Delta',xlabel='Spot bump (currency)',ylabel='Delta difference (symlog scale)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(folder/'pathwise_fd_paired_difference.png',dpi=170)
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-paths',type=int,default=20000)
    parser.add_argument('--replications',type=int,default=200)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    args=parser.parse_args()
    for name in ('raw','tables','figures'):
        (args.output_dir/name).mkdir(parents=True,exist_ok=True)
    raw=collect_comparison(args.n_paths,args.replications,args.seed)
    conv=collect_convergence(args.replications,args.seed)
    pairs=collect_paired(args.n_paths,args.replications,args.seed)
    summary=summarize(raw)
    convergence=summarize(conv)
    comparison=compare_variance(summary)
    paired=summarize_paired(pairs)
    fits=[]
    for method in ('pathwise','fd_h1'):
        data=convergence.loc[convergence.method==method].sort_values('n_paths')
        fits.append(dict(method=method,**asdict(fit_power_law(data.n_paths.to_numpy(),data.rms_reported_standard_error.to_numpy()))))
    artifacts=[]
    for directory,name,frame in (('raw','pathwise_replications',raw),('raw','pathwise_convergence_replications',conv),
                                 ('raw','pathwise_paired_replications',pairs),('tables','pathwise_estimators',summary),
                                 ('tables','pathwise_variance_comparison',comparison),('tables','pathwise_convergence',convergence),
                                 ('tables','pathwise_paired_differences',paired),('tables','pathwise_se_slopes',pd.DataFrame(fits))):
        path=args.output_dir/directory/f'{name}.csv'
        frame.to_csv(path,index=False)
        artifacts.append(path)
    figures(summary,convergence,paired,args.output_dir/'figures')
    meta=dict(milestone=12,model=asdict(MODEL),product=asdict(PRODUCT),maturity=1.,n_paths=args.n_paths,
              replications=args.replications,seed=args.seed,bit_generator='PCG64',methods=METHODS,budgets=BUDGETS,
              convergence_counts=COUNTS,paired_bumps=PAIRED_BUMPS,
              exact_pathwise_sample_variance=population_pathwise_variance(),
              streams='independent comparison (0,budget,replication,method); convergence (1,count,replication,method); paired (2,replication); warmup prefix 99',
              dependence='method streams independent in comparisons/convergence; same paths across methods/bumps ONLY in paired diagnostic',
              timing='full simulator/estimator including RNG draws; excludes Generator construction, warmups, reference formulas and output',
              scope='pathwise European Delta only; main study call, put covered by formula and tests; no pathwise Gamma/Vega or exotic Greeks',
              versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
              sha256={str(p.relative_to(args.output_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    (args.output_dir/'raw/pathwise_study.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(summary[['budget','method','n_paths','mean_estimate','empirical_bias','rms_reported_standard_error','empirical_variance','rmse','mean_runtime_seconds']].to_string(index=False))
    print(comparison.to_string(index=False))
    print(paired.to_string(index=False))
    print(pd.DataFrame(fits).to_string(index=False))


if __name__=='__main__':
    main()
