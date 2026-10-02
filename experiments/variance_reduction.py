"""Milestone 9: equal-path-budget plain versus antithetic MC, independent runs.

Includes a pre-specified terminal-barrier counterexample with a symmetric
survival window in Gaussian space. The separate M10 control_variates experiment
extends the comparison to geometric Asian controls.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import f
from scipy.integrate import quad
from scipy.stats import norm

from mc_pricer import (
    ArithmeticAsianOption, BlackScholesModel, EuropeanOption, UpAndOutCall,
    black_scholes_price, price_monte_carlo, simulate_antithetic_payoffs,
    terminal_monitored_up_and_out_price,
)
from mc_pricer.statistics.confidence_intervals import binomial_wilson_interval
from mc_pricer.statistics.diagnostics import summarize_replications


@dataclass(frozen=True)
class StudyCase:
    name: str
    model: BlackScholesModel
    product: EuropeanOption | ArithmeticAsianOption | UpAndOutCall
    n_steps: int | None
    analytical_target: float | None
    maturity: float = 1.0


def study_cases() -> tuple[StudyCase, ...]:
    """Fixed scenarios chosen before comparing realized estimator variances."""
    model = BlackScholesModel(100, .05, .2)
    # r=sigma^2/2 gives zero log drift. Reciprocal K/S and B/S make the
    # terminal survival/positive-payoff window symmetric around Z=0.
    window_model = BlackScholesModel(100, .02, .2)
    window = UpAndOutCall(100 / 1.1, 110)
    return (
        StudyCase("european_call", model, EuropeanOption(100), None,
                  black_scholes_price(model, 100, 1, "call")),
        StudyCase("european_put", model, EuropeanOption(100, "put"), None,
                  black_scholes_price(model, 100, 1, "put")),
        StudyCase("asian_call_12", model, ArithmeticAsianOption(100), 12, None),
        StudyCase("barrier_call_96", model, UpAndOutCall(100, 120), 96, None),
        StudyCase("barrier_symmetric_window", window_model, window, 1,
                  terminal_monitored_up_and_out_price(window_model, window.strike, window.barrier, 1)),
    )


def stream(seed: int, case: int, replication: int, method: int) -> np.random.Generator:
    """Independent case/run/method cell; recorded spawn keys allow exact replay."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(case, replication, method)))


def symmetric_window_benchmark(n_paths: int) -> pd.DataFrame:
    """Independent Gaussian quadrature for the fixed counterexample's moments.

    This deterministic validation establishes the covariance sign without
    selecting more Monte Carlo runs until an empirical ratio excludes one.
    Integrate only the finite positive-payoff interval and its reflection.
    """
    case = study_cases()[-1]
    spot, rate, volatility = case.model.spot, case.model.rate, case.model.volatility
    strike, barrier, time = case.product.strike, case.product.barrier, case.maturity
    drift = (rate - 0.5 * volatility**2) * time
    diffusion = volatility * np.sqrt(time)
    lower = (np.log(strike / spot) - drift) / diffusion
    upper = (np.log(barrier / spot) - drift) / diffusion
    discount = np.exp(-rate * time)

    def cashflow(z: float) -> float:
        return discount * (spot * np.exp(drift + diffusion * z) - strike)

    mean, mean_error = quad(lambda z: cashflow(z) * norm.pdf(z), lower, upper, epsabs=1e-11)
    second, second_error = quad(lambda z: cashflow(z)**2 * norm.pdf(z), lower, upper, epsabs=1e-11)
    cross, cross_error = quad(lambda z: cashflow(z) * cashflow(-z) * norm.pdf(z),
                              max(lower, -upper), min(upper, -lower), epsabs=1e-11)
    marginal_variance = second - mean**2
    covariance = cross - mean**2
    return pd.DataFrame([{
        "case": case.name, "n_paths": n_paths, "quadrature_price": mean,
        "analytical_price": case.analytical_target, "discounted_payoff_variance": marginal_variance,
        "pair_covariance": covariance, "pair_correlation": covariance / marginal_variance,
        "plain_estimator_variance": marginal_variance / n_paths,
        "antithetic_estimator_variance": (marginal_variance + covariance) / n_paths,
        "population_vrf_quadrature": marginal_variance / (marginal_variance + covariance),
        "max_quadrature_reported_absolute_error": max(mean_error, second_error, cross_error),
    }])


def collect_comparison(n_paths: int, replications: int, seed: int) -> pd.DataFrame:
    """Time full pricer calls, alternating method order; replay pair diagnostics.

    Methods have independent streams, NOT shared first halves. Both evaluate
    n_paths payoffs. Diagnostics regenerate antithetic samples outside measured
    pricing time. Raw rows retain every estimate, variance, CI and runtime.
    """
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (n_paths, replications, seed)):
        raise TypeError("path count, replication count and seed must be integers")
    if n_paths < 4 or n_paths % 2 or replications < 2 or seed < 0:
        raise ValueError("require even n_paths>=4, replications>=2 and seed>=0")
    cases = study_cases()
    rows = []
    for case_index, case in enumerate(cases):
        for method_index, method in enumerate(("none", "antithetic")):
            price_monte_carlo(case.model, case.product, case.maturity, 256,
                              n_steps=case.n_steps, variance_reduction=method,
                              rng=stream(seed, len(cases), case_index, method_index))
        for replication in range(replications):
            order = (0, 1) if replication % 2 == 0 else (1, 0)
            for method_index in order:
                method = ("none", "antithetic")[method_index]
                result = price_monte_carlo(
                    case.model, case.product, case.maturity, n_paths,
                    n_steps=case.n_steps, variance_reduction=method,
                    rng=stream(seed, case_index, replication, method_index),
                )
                covariance = correlation = pair_implied_vrf = None
                if method == "antithetic":
                    positive, negative = simulate_antithetic_payoffs(
                        case.model, case.product, case.maturity, n_paths // 2,
                        n_steps=case.n_steps, rng=stream(seed, case_index, replication, method_index),
                    )
                    discount = np.exp(-case.model.rate * case.maturity)
                    positive, negative = discount * positive, discount * negative
                    matrix = np.cov(positive, negative, ddof=1)
                    covariance = float(matrix[0, 1])
                    denominator = np.sqrt(matrix[0, 0] * matrix[1, 1])
                    correlation = float(covariance / denominator) if denominator > 0 else None
                    if result.sample_variance > 0:
                        pair_implied_vrf = float((matrix[0, 0] + matrix[1, 1]) / (4 * result.sample_variance))
                rows.append({
                    "case": case.name, "case_index": case_index, "replication": replication,
                    "method": method, "spawn_key": json.dumps([case_index, replication, method_index]),
                    "n_paths": n_paths, "n_steps": case.n_steps,
                    "n_independent_samples": result.n_independent_samples,
                    "price": result.price, "standard_error": result.standard_error,
                    "sample_variance": result.sample_variance, "estimator_variance": result.estimator_variance,
                    "ci_lower": result.confidence_interval[0], "ci_upper": result.confidence_interval[1],
                    "runtime_seconds": result.runtime_seconds, "analytical_target": case.analytical_target,
                    "pair_covariance": covariance, "pair_correlation": correlation,
                    "pair_implied_vrf": pair_implied_vrf,
                })
    return pd.DataFrame(rows)


def summarize_comparison(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirical estimator variance and approximate F intervals for its ratio.

    F intervals assume independent, approximately normal run-level estimates;
    normality here is a large-path-count approximation, not exact payoff normality.
    Bias/RMSE/coverage are absent when no independent analytical target exists.
    """
    summaries = []
    for (name, method), group in raw.groupby(["case", "method"], sort=False):
        estimates = group.price.to_numpy()
        count = len(group)
        if count < 2 or group.n_paths.nunique() != 1 or group.n_independent_samples.nunique() != 1:
            raise ValueError("need at least two independent runs with a common budget")
        target = group.analytical_target.iloc[0]
        empirical_variance = estimates.var(ddof=1)
        row = {
            "case": name, "method": method, "n_replications": count,
            "n_paths_per_run": int(group.n_paths.iloc[0]),
            "n_independent_samples_per_run": int(group.n_independent_samples.iloc[0]),
            "mean_price": estimates.mean(), "empirical_estimator_variance": empirical_variance,
            "mean_reported_estimator_variance": group.estimator_variance.mean(),
            "rms_reported_standard_error": np.sqrt(group.estimator_variance.mean()),
            "empirical_standard_error_of_mean": np.sqrt(empirical_variance / count),
            "mean_runtime_seconds": group.runtime_seconds.mean(),
            "median_runtime_seconds": group.runtime_seconds.median(),
            "mean_pair_correlation": group.pair_correlation.mean(),
            "analytical_target": target, "empirical_bias": None, "rmse": None,
            "bias_ci_lower": None, "bias_ci_upper": None,
            "coverage": None, "coverage_ci_lower": None, "coverage_ci_upper": None,
        }
        if pd.notna(target):
            stats = summarize_replications(estimates, target)
            hits = int(((group.ci_lower <= target) & (target <= group.ci_upper)).sum())
            coverage_ci = binomial_wilson_interval(hits, count)
            row.update(empirical_bias=stats.empirical_bias, rmse=stats.rmse,
                       bias_ci_lower=stats.bias_confidence_interval[0], bias_ci_upper=stats.bias_confidence_interval[1],
                       coverage=hits/count, coverage_ci_lower=coverage_ci[0], coverage_ci_upper=coverage_ci[1])
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    comparisons = []
    for name, group in summary.groupby("case", sort=False):
        if set(group.method) != {"none", "antithetic"} or group.n_paths_per_run.nunique() != 1:
            raise ValueError("each case requires both methods at the same total path budget")
        plain = group.loc[group.method == "none"].iloc[0]
        anti = group.loc[group.method == "antithetic"].iloc[0]
        ratio = (plain.empirical_estimator_variance / anti.empirical_estimator_variance
                 if min(plain.empirical_estimator_variance, anti.empirical_estimator_variance) > 0 else np.nan)
        reported_ratio = (plain.mean_reported_estimator_variance / anti.mean_reported_estimator_variance
                          if min(plain.mean_reported_estimator_variance, anti.mean_reported_estimator_variance) > 0 else np.nan)
        df_plain, df_anti = plain.n_replications-1, anti.n_replications-1
        gap = anti.mean_price-plain.mean_price
        gap_se = np.sqrt(plain.empirical_standard_error_of_mean**2 + anti.empirical_standard_error_of_mean**2)
        comparisons.append({
            "case": name, "n_paths_per_run": plain.n_paths_per_run,
            "empirical_vrf": ratio, "vrf_ci_lower": ratio/f.ppf(.975,df_plain,df_anti),
            "vrf_ci_upper": ratio/f.ppf(.025,df_plain,df_anti),
            "reported_variance_vrf": reported_ratio,
            "empirical_variance_reduction_percent": 100*(1-1/ratio),
            "mean_pair_correlation": anti.mean_pair_correlation,
            "time_adjusted_efficiency": ratio*plain.mean_runtime_seconds/anti.mean_runtime_seconds,
            "anti_minus_plain_mean": gap, "difference_standard_error": gap_se,
            "difference_ci_lower": gap-1.959963984540054*gap_se,
            "difference_ci_upper": gap+1.959963984540054*gap_se,
        })
    return summary, pd.DataFrame(comparisons)


def plot_comparison(summary: pd.DataFrame, comparison: pd.DataFrame, destination: Path) -> None:
    names = [case.name for case in study_cases()]
    labels = ["European\ncall", "European\nput", "Asian call\n12 dates", "Barrier call\n96 dates", "Barrier\nsymmetric window"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        x = np.arange(len(names))
        for offset, method, color in ((-.18,"none","#708090"),(.18,"antithetic","#126F8A")):
            rows = summary.loc[summary.method == method].set_index("case").loc[names]
            axes[0].bar(x+offset, rows.empirical_estimator_variance, width=.36, color=color,
                        label="Plain MC" if method == "none" else "Antithetic MC")
        axes[0].set(yscale="log", ylabel="Estimator variance (currency², log scale)",
                    title="Empirical variance across independent runs")
        axes[0].legend(frameon=False)
        ratios = comparison.set_index("case").loc[names]
        axes[1].errorbar(x, ratios.empirical_vrf,
                        yerr=[ratios.empirical_vrf-ratios.vrf_ci_lower,ratios.vrf_ci_upper-ratios.empirical_vrf],
                        fmt="o", capsize=5, color="#126F8A")
        axes[1].axhline(1,color="#9B3B32",ls="--",label="Equal variance; below 1 is worse")
        axes[1].scatter(x, ratios.reported_variance_vrf, marker="x", color="#BE8127",
                        label="Ratio of mean reported estimator variances", zorder=3)
        axes[1].set(ylabel="VRF = variance plain / variance antithetic",
                    title="Approximate 95% F intervals (normal run-estimate approximation)", xlabel="Contract scenario")
        axes[1].legend(frameon=False, fontsize=9)
        for axis in axes:
            axis.set_xticks(x, labels)
            axis.grid(axis="y",alpha=.2)
        first = summary.iloc[0]
        fig.suptitle(f"Equal payoff budget: {int(first.n_paths_per_run):,} paths per run and method\n"
                     f"{int(first.n_replications)} independent runs; antithetic uncertainty uses pair means", fontsize=11)
        fig.savefig(destination/"antithetic_variance_reduction.png",dpi=170)
        plt.close(fig)


def run(output_dir: Path, n_paths: int, replications: int, seed: int) -> None:
    raw = collect_comparison(n_paths, replications, seed)
    summary, comparison = summarize_comparison(raw)
    for folder in ("raw","tables","figures"):
        (output_dir/folder).mkdir(parents=True,exist_ok=True)
    raw.to_csv(output_dir/"raw/antithetic_replications.csv",index=False)
    summary.to_csv(output_dir/"tables/antithetic_estimators.csv",index=False)
    comparison.to_csv(output_dir/"tables/antithetic_comparison.csv",index=False)
    symmetric_window_benchmark(n_paths).to_csv(output_dir/"tables/antithetic_window_benchmark.csv",index=False)
    metadata = {
        "milestone":9,"root_seed":seed,"n_paths_per_run_and_method":n_paths,"n_replications":replications,
        "methods":["none","antithetic"],"cases":[asdict(case) for case in study_cases()],
        "rng":"PCG64; SeedSequence(seed, spawn_key=(case_index, replication, method_index)); methods independent",
        "warmup":"256-path calls per case/method, keys (5,case_index,method_index); excluded from tables",
        "execution_order":"Alternate method order each replication. Case order is fixed.",
        "draw_shapes":"Plain N or (N,m); antithetic N/2 or (N/2,m). Negate all owned shocks for second branch.",
        "budget":"Equal total payoff evaluations per contract, not equal Gaussian draws or wall-clock time.",
        "variance_units":"Raw sample_variance is payoff variance for plain, pair-mean variance for antithetic; compare estimator_variance instead.",
        "timing_scope":"Full price_monte_carlo call; excludes caller RNG construction, warmup, diagnostic pair replay, tables and plots. Runtime efficiency is environment-specific.",
        "ratio_intervals":"F intervals use R-1 degrees per independent method, assuming approximately normal run estimates. They are pointwise, not simultaneous.",
        "degenerate_ratios":"A zero estimated variance makes VRF comparisons undefined here; report missing values, not an infinite efficiency claim.",
        "counterexample":"Pre-specified terminal-only barrier: S0=100, r=sigma²/2=.02, sigma=.2, K=100/1.1, B=110. Both legs share the same central Gaussian payoff-support window.",
        "counterexample_validation":"Deterministic Gaussian quadrature checks price, marginal variance, paired covariance and population VRF independently of repeated MC.",
        "storage":"Save estimator-level rows and spawn keys; path/payoff samples are regenerated, not all retained. No general batching.",
        "raw_replications_sha256":hashlib.sha256((output_dir/"raw/antithetic_replications.csv").read_bytes()).hexdigest(),
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__},
        "limitations":"Five scenarios, finite R, approximate ratio intervals. No exact Asian or 96-date barrier targets, no bias/RMSE claims for these. Antithetics do not remove monitoring/model bias.",
    }
    (output_dir/"raw/antithetic_study.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plot_comparison(summary,comparison,output_dir/"figures")
    print(summary[["case","method","mean_price","rms_reported_standard_error","empirical_estimator_variance","mean_runtime_seconds"]].to_string(index=False,float_format=lambda x:f"{x:.8g}"))
    print(comparison.to_string(index=False,float_format=lambda x:f"{x:.8g}"))
    print(f"Saved {len(raw)} estimator rows to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("results"))
    parser.add_argument("--n-paths",type=int,default=10000)
    parser.add_argument("--replications",type=int,default=200)
    parser.add_argument("--seed",type=int,default=20260929)
    args = parser.parse_args()
    run(args.output_dir,args.n_paths,args.replications,args.seed)


if __name__ == "__main__":
    main()
