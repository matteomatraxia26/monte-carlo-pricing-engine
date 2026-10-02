"""M14: likelihood-ratio Delta, vanilla comparisons and digital discontinuities."""
import argparse
from dataclasses import asdict, dataclass, replace
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
from scipy.stats import norm
from mc_pricer import (
    BlackScholesModel, DigitalOption, EuropeanOption, black_scholes_price,
    black_scholes_greeks, digital_price, digital_delta, likelihood_ratio_delta,
    pathwise_delta, finite_difference_greek,
)
from mc_pricer.statistics.confidence_intervals import binomial_wilson_interval
from mc_pricer.statistics.diagnostics import summarize_replications, fit_power_law
from experiments.estimator_robustness import bootstrap_variance_ratio


@dataclass(frozen=True)
class Case:
    name: str
    digital: bool = False
    kind: str = 'call'
    spot: float = 100.
    volatility: float = .2
    maturity: float = 1.
    rate: float = .05
    strike: float = 100.

    @property
    def model(self) -> BlackScholesModel:
        return BlackScholesModel(self.spot, self.rate, self.volatility)

    @property
    def product(self) -> EuropeanOption | DigitalOption:
        cls = DigitalOption if self.digital else EuropeanOption
        return cls(self.strike, self.kind)


CASES = (Case('vanilla_call'), Case('vanilla_put', kind='put'),
         Case('vanilla_short', maturity=.04), Case('digital_call', digital=True),
         Case('digital_short', digital=True, maturity=.04),
         Case('digital_rare', digital=True, spot=80., volatility=.1, maturity=.25))
COUNTS = (1000, 3000, 10000, 30000, 100000)
CONVERGENCE = ((0, 'lr'), (0, 'pw'), (3, 'lr'))
BUMPS = {'fd_h1': 1., 'fd_h0.1': .1, 'fd_h0.01': .01}


def methods(case: Case) -> tuple[str, ...]:
    return ('lr', 'fd_h1', 'fd_h0.1', 'fd_h0.01') if case.digital else ('lr', 'pw', 'fd_h1', 'fd_h0.1')


def stream(seed: int, key: tuple[int, ...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def validate_design(n_paths: int, replications: int, seed: int) -> None:
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (n_paths, replications, seed)):
        raise TypeError('counts and seed must be integers')
    if n_paths < 2 or replications < 2 or seed < 0:
        raise ValueError('require n_paths>=2, replications>=2, seed>=0')


def tilted_normal_moments(b: float, c: float, upper: bool) -> tuple[float, float]:
    """E[Z exp(bZ) 1{Z>c}] and E[Z² exp(bZ) 1{Z>c}], or lower tail.

    Differentiate exp(b²/2)*Phi(b-c) once/twice in b. These are deterministic
    moment benchmarks for moderate parameters, not an extreme-tail routine.
    """
    p = norm.cdf(b-c) if upper else norm.cdf(c-b)
    density = norm.pdf(c-b)
    sign = 1 if upper else -1
    factor = np.exp(.5*b*b)
    return (float(factor*(b*p+sign*density)),
            float(factor*((1+b*b)*p+sign*(b+c)*density)))


def lr_population_moments(case: Case) -> tuple[float, float]:
    """Exact-formula discounted LR sample mean and variance; float64 evaluation."""
    s, k, t, v, r = case.spot, case.strike, case.maturity, case.volatility, case.rate
    a = v*np.sqrt(t); drift = (r-.5*v*v)*t; d = np.exp(-r*t)
    c = (np.log(k/s)-drift)/a
    m10, m20 = tilted_normal_moments(0., c, case.kind == 'call')
    if case.digital:
        mean = d/(s*a)*m10
        second = (d/(s*a))**2*m20
    else:
        m11, m21 = tilted_normal_moments(a, c, case.kind == 'call')
        _, m22 = tilted_normal_moments(2*a, c, case.kind == 'call')
        sign = 1 if case.kind == 'call' else -1
        forward_scale = s*np.exp(drift)
        mean = sign*d/(s*a)*(forward_scale*m11-k*m10)
        second = (d/(s*a))**2*(forward_scale**2*m22-2*k*forward_scale*m21+k*k*m20)
    variance = second-mean*mean
    if variance < 0:
        raise FloatingPointError('moment cancellation produced negative variance')
    return float(mean), float(variance)


def targets(case: Case, method: str) -> tuple[float, float, float]:
    """Return exact Delta, finite-bump target, and known sample variance or NaN."""
    exact = (digital_delta(case.model, case.strike, case.maturity, case.kind) if case.digital
             else black_scholes_greeks(case.model, case.strike, case.maturity, case.kind).delta)
    if method == 'lr':
        return exact, exact, lr_population_moments(case)[1]
    if method == 'pw':
        if case.digital:
            raise ValueError('pathwise indicator differentiation is not a valid digital Greek')
        a = case.volatility*np.sqrt(case.maturity)
        d1 = (np.log(case.spot/case.strike)+case.rate*case.maturity)/a+.5*a
        sign = 1 if case.kind == 'call' else -1
        variance = np.exp(a*a)*norm.cdf(sign*(d1+a))-exact**2
        return exact, exact, float(variance)
    h = BUMPS[method]
    formula = digital_price if case.digital else black_scholes_price
    up = formula(replace(case.model, spot=case.spot+h), case.strike, case.maturity, case.kind)
    down = formula(replace(case.model, spot=case.spot-h), case.strike, case.maturity, case.kind)
    variance = np.nan
    if case.digital:
        a = case.volatility*np.sqrt(case.maturity)
        drift = (case.rate-.5*case.volatility**2)*case.maturity
        lo = (np.log(case.strike/(case.spot+h))-drift)/a
        hi = (np.log(case.strike/(case.spot-h))-drift)/a
        p = norm.sf(lo)-norm.sf(hi) if lo >= 0 else norm.cdf(hi)-norm.cdf(lo)
        variance = np.exp(-2*case.rate*case.maturity)*p*(1-p)/(4*h*h)
    return exact, (up-down)/(2*h), float(variance)


def estimate(case: Case, method: str, count: int, rng: np.random.Generator):
    if method == 'lr':
        return likelihood_ratio_delta(case.model, case.product, case.maturity, count, rng=rng)
    if method == 'pw':
        return pathwise_delta(case.model, case.product, case.maturity, count, rng=rng)
    return finite_difference_greek(case.model, case.product, case.maturity, count,
                                   greek='delta', bump=BUMPS[method], rng=rng)


def record(case_index: int, method: str, count: int, replication: int,
           seed: int, key: tuple[int, ...], budget: str) -> dict:
    case = CASES[case_index]
    exact, target, population_variance = targets(case, method)
    # Build Generator and targets before timing the full library call.
    result = estimate(case, method, count, stream(seed, key))
    return dict(scenario=case.name, case_index=case_index, digital=case.digital, method=method,
                replication=replication, spawn_key=json.dumps(key), budget=budget, n_paths=count,
                terminal_evaluations=count*(2 if method in BUMPS else 1), bump=BUMPS.get(method),
                estimate=result.estimate, standard_error=result.standard_error,
                sample_variance=result.sample_variance, estimator_variance=result.estimator_variance,
                ci_lower=result.confidence_interval[0], ci_upper=result.confidence_interval[1],
                runtime_seconds=result.runtime_seconds, analytical_delta=exact, estimator_target=target,
                deterministic_bias=target-exact, population_sample_variance=population_variance,
                population_estimator_variance=population_variance/count)


def collect_comparison(n_paths: int, replications: int, seed: int) -> pd.DataFrame:
    validate_design(n_paths, replications, seed)
    rows = []
    for ci, case in enumerate(CASES):
        names = methods(case)
        for mi, method in enumerate(names):
            estimate(case, method, 128, stream(seed, (99, 0, ci, mi)))
        for rep in range(replications):
            for index in np.roll(np.arange(len(names)), rep % len(names)):
                mi = int(index); method = names[mi]
                count = n_paths if method in BUMPS else 2*n_paths
                rows.append(record(ci, method, count, rep, seed, (0, ci, rep, mi), 'equal_terminal_evaluations'))
    return pd.DataFrame(rows)


def collect_convergence(replications: int, seed: int, counts: tuple[int, ...] = COUNTS) -> pd.DataFrame:
    rows = []
    if not counts:
        raise ValueError('need at least one count')
    for count in counts:
        validate_design(count, replications, seed)
    for si, (ci, method) in enumerate(CONVERGENCE):
        for ni, count in enumerate(counts):
            for rep in range(replications):
                rows.append(record(ci, method, count, rep, seed, (1, si, ni, rep), 'convergence'))
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, method, count), group in raw.groupby(['scenario', 'method', 'n_paths'], sort=False):
        exact = group.analytical_delta.iloc[0]
        stats = summarize_replications(group.estimate.to_numpy(), exact)
        row = dict(scenario=scenario, method=method, n_paths=count, n_replications=len(group),
                   terminal_evaluations=int(group.terminal_evaluations.iloc[0]),
                   analytical_delta=exact, estimator_target=group.estimator_target.iloc[0],
                   deterministic_bias=group.deterministic_bias.iloc[0],
                   mean_estimate=group.estimate.mean(), empirical_bias=stats.empirical_bias,
                   bias_ci_lower=stats.bias_confidence_interval[0], bias_ci_upper=stats.bias_confidence_interval[1],
                   empirical_variance=group.estimate.var(ddof=1), rmse=stats.rmse,
                   mean_reported_se=group.standard_error.mean(),
                   rms_reported_se=np.sqrt(group.estimator_variance.mean()),
                   mean_sample_variance=group.sample_variance.mean(),
                   population_sample_variance=group.population_sample_variance.iloc[0],
                   population_estimator_variance=group.population_estimator_variance.iloc[0],
                   mean_runtime_seconds=group.runtime_seconds.mean(), bump=group.bump.iloc[0])
        for label, mask in (('greek_coverage', (group.ci_lower<=exact)&(exact<=group.ci_upper)),
                            ('target_coverage', (group.ci_lower<=group.estimator_target)&(group.estimator_target<=group.ci_upper)),
                            ('zero_se_rate', group.standard_error==0)):
            hits = int(mask.sum()); lo, hi = binomial_wilson_interval(hits, len(group))
            row.update({label: hits/len(group), label+'_lower': lo, label+'_upper': hi})
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(raw: pd.DataFrame, seed: int, resamples: int = 2000) -> pd.DataFrame:
    """Reference LR / candidate variance and MSE at equal terminal budgets."""
    rows = []
    for ci, case in enumerate(CASES):
        group = raw.loc[raw.scenario==case.name]
        if group.terminal_evaluations.nunique() != 1:
            raise ValueError('comparison requires equal terminal-evaluation budgets')
        reference = group.loc[group.method=='lr']
        for mi, method in enumerate(methods(case)[1:]):
            candidate = group.loc[group.method==method]
            ratio = bootstrap_variance_ratio(reference.estimate.to_numpy(), candidate.estimate.to_numpy(),
                                             rng=stream(seed, (2, ci, mi)), resamples=resamples)
            truth = reference.analytical_delta.iloc[0]
            mse_ref = np.mean((reference.estimate-truth)**2)
            mse_candidate = np.mean((candidate.estimate-truth)**2)
            vr = reference.population_estimator_variance.iloc[0]
            vc = candidate.population_estimator_variance.iloc[0]
            rows.append(dict(scenario=case.name, reference='lr', candidate=method, **ratio,
                             empirical_mse_ratio=mse_ref/mse_candidate if mse_candidate>0 else np.nan,
                             population_variance_ratio=vr/vc if np.isfinite(vc) and vc>0 else np.nan,
                             time_adjusted_efficiency=ratio['empirical_vrf']*reference.runtime_seconds.mean()/candidate.runtime_seconds.mean()))
    return pd.DataFrame(rows)


def slopes(convergence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, method), group in convergence.groupby(['scenario', 'method'], sort=False):
        fit = fit_power_law(group.n_paths.to_numpy(), group.mean_reported_se.to_numpy())
        rows.append(dict(scenario=scenario, method=method, **asdict(fit)))
    return pd.DataFrame(rows)


def figures(summary: pd.DataFrame, convergence: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    names = [c.name for c in CASES[:3]]; x = np.arange(3)
    for mi, method in enumerate(methods(CASES[0])):
        data = summary.loc[summary.method==method].set_index('scenario').loc[names]
        axes[0].bar(x+(mi-1.5)*.2, data.rmse, width=.19, label=method)
        axes[1].bar(x+(mi-1.5)*.2, data.mean_runtime_seconds*1000, width=.19, label=method)
    for ax, ylabel, title in zip(axes, ('Delta RMSE (price / spot)', 'Mean runtime (ms)'),
                                 ('Vanilla Delta error at equal terminal budgets', 'Full library-call runtime')):
        ax.set_xticks(x, names); ax.set(xlabel='Scenario', ylabel=ylabel, title=title)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.savefig(output/'lr_vanilla_comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for col, case in enumerate(CASES[3:]):
        data = summary.loc[summary.scenario==case.name].set_index('method')
        fd = data.loc[list(BUMPS)]
        axes[0,col].loglog(fd.bump, fd.rmse, 'o-', label='FD empirical RMSE')
        theoretical = np.sqrt(fd.population_estimator_variance+fd.deterministic_bias**2)
        axes[0,col].loglog(fd.bump, theoretical, ':', color='C0', label='FD population RMSE')
        axes[0,col].axhline(data.loc['lr','rmse'], color='C1', label='LR empirical RMSE')
        axes[0,col].axhline(np.sqrt(data.loc['lr','population_estimator_variance']), color='C1', ls='--', label='LR population RMSE')
        axes[1,col].semilogx(fd.bump, fd.greek_coverage, 'o-', label='CRN finite difference')
        axes[1,col].axhline(data.loc['lr','greek_coverage'], color='C1', label='Likelihood ratio')
        axes[1,col].axhline(.95, color='k', ls=':', label='Nominal 95%')
        axes[0,col].set(title=case.name, ylabel='Delta RMSE (price / spot)')
        axes[1,col].set(ylabel='Coverage of true Delta', ylim=(-.04,1.04))
        for ax in axes[:,col]:
            ax.set_xlabel('Spot bump h (currency)'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(output/'lr_digital_bumps.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.5), constrained_layout=True)
    for (scenario, method), group in convergence.groupby(['scenario','method'], sort=False):
        ax.loglog(group.n_paths, group.mean_reported_se, 'o-', label=f'{scenario}: {method}')
    ax.set(xlabel='Independent paths N', ylabel='Mean reported Delta SE (price / spot)', title='Delta sampling-error convergence')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output/'lr_delta_convergence.png', dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-paths', type=int, default=10000)
    parser.add_argument('--replications', type=int, default=200)
    parser.add_argument('--convergence-replications', type=int, default=100)
    parser.add_argument('--bootstrap-resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1]/'results')
    args = parser.parse_args()
    validate_design(args.n_paths, args.replications, args.seed)
    validate_design(2, args.convergence_replications, args.seed)
    if args.bootstrap_resamples<2:
        parser.error('bootstrap-resamples must be >=2')
    for folder in ('raw','tables','figures'):
        (args.output_dir/folder).mkdir(parents=True, exist_ok=True)
    raw = collect_comparison(args.n_paths, args.replications, args.seed)
    conv_raw = collect_convergence(args.convergence_replications, args.seed)
    summary, convergence = summarize(raw), summarize(conv_raw)
    ratios = comparisons(raw, args.seed, args.bootstrap_resamples)
    fits = slopes(convergence)
    paths = []
    for folder, name, frame in (('raw','lr_replications',raw), ('raw','lr_convergence_replications',conv_raw),
                                ('tables','lr_estimators',summary), ('tables','lr_comparisons',ratios),
                                ('tables','lr_convergence',convergence), ('tables','lr_se_slopes',fits)):
        path = args.output_dir/folder/f'{name}.csv'; frame.to_csv(path,index=False); paths.append(path)
    figures(summary, convergence, args.output_dir/'figures')
    metadata = dict(milestone=14, seed=args.seed, n_paths=args.n_paths, replications=args.replications,
                    convergence_replications=args.convergence_replications, counts=COUNTS,
                    bootstrap_resamples=args.bootstrap_resamples, cases=[asdict(c) for c in CASES],
                    methods={c.name:methods(c) for c in CASES}, bumps=BUMPS,
                    streams='PCG64 SeedSequence: comparison (0,case,rep,method); convergence (1,series,count,rep); bootstrap (2,case,candidate); warmup prefix99',
                    budget='Comparison: LR/PW 2N draws vs FD N coupled rows, all 2N terminal evaluations. Convergence: N independent paths. Digital payout=1.',
                    timing='Full library call; excludes Generator construction, analytical targets, bootstrap, warmups and output; no equal-CPU claim',
                    uncertainty='Normal per-run intervals, Wilson intervals for coverage, independent run-level bootstrap ratios; pointwise; no rare-event guarantee',
                    limitations='Digital PW is invalid and excluded. No LR path-dependent Greeks, Gamma, Vega, score controls or Heston. Moment formulas use float64, not extreme-tail arithmetic.',
                    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
                    sha256={str(p.relative_to(args.output_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output_dir/'raw/lr_study.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(summary[['scenario','method','analytical_delta','mean_estimate','rmse','empirical_variance','greek_coverage','zero_se_rate']].to_string(index=False))
    print(ratios.to_string(index=False)); print(fits.to_string(index=False))


if __name__ == '__main__':
    main()
