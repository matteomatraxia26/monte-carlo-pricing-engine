"""M10: independent-pilot geometric controls at equal total stock-path budgets."""

import argparse
from dataclasses import asdict
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

from mc_pricer import (
    ArithmeticAsianOption, BlackScholesModel, GeometricAsianOption,
    geometric_asian_price, price_from_payoff_samples, price_monte_carlo, simulate_paths,
)
from mc_pricer.statistics.confidence_intervals import binomial_wilson_interval

METHODS = ("none", "antithetic", "control_variate")
CASES = (("asian_call_12", "call", 12), ("asian_put_12", "put", 12),
         ("asian_call_48", "call", 48))
MODEL = BlackScholesModel(100., .05, .2)


def stream(seed: int, case: int, replication: int, method: int) -> np.random.Generator:
    """Distinct method/run/case streams; pilot and production use successive draws."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(case, replication, method)))


def replay_control_samples(seed: int, case: int, replication: int,
                           n_paths: int, n_pilot: int) -> dict[str, np.ndarray]:
    """Regenerate paired undiscounted payoffs, outside the full-pricer timer."""
    _, kind, steps = CASES[case]
    arithmetic = ArithmeticAsianOption(100., kind)
    geometric = GeometricAsianOption(100., kind)
    rng = stream(seed, case, replication, 2)
    arrays = {}
    for name, count in (("pilot", n_pilot), ("production", n_paths-n_pilot)):
        paths = simulate_paths(MODEL, 1., count, steps, rng=rng)
        arrays[f"{name}_target"] = arithmetic.payoff(paths)
        arrays[f"{name}_control"] = geometric.payoff(paths)
        del paths
    return arrays


def collect_comparison(n_paths: int, n_pilot: int, replications: int,
                       seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-call timings and independent estimates; no exact arithmetic target.

    Equal stock-path counts include pilot paths, but CV computes two payoffs on
    every path. Method order rotates to reduce timing-order effects. Replay of
    the geometric production samples is diagnostic and excluded from timings.
    """
    counts = (n_paths, n_pilot, replications, seed)
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in counts):
        raise TypeError("budgets, replications and seed must be integers")
    if n_paths < 4 or n_paths % 2 or not 2 <= n_pilot <= n_paths-2 or replications < 2 or seed < 0:
        raise ValueError("require even total>=4, 2<=pilot<=total-2, replications>=2, seed>=0")
    rows, checks = [], []
    for case_index, (name, kind, steps) in enumerate(CASES):
        product = ArithmeticAsianOption(100., kind)
        exact_geometric = geometric_asian_price(MODEL, 100., 1., steps, kind)
        for method_index, method in enumerate(METHODS):
            price_monte_carlo(MODEL, product, 1., 256, n_steps=steps,
                              rng=stream(seed, len(CASES), case_index, method_index),
                              variance_reduction=method,
                              n_pilot_paths=32 if method == "control_variate" else None)
        for replication in range(replications):
            for method_index in np.roll(np.arange(3), replication % 3):
                method_index = int(method_index)
                method = METHODS[method_index]
                result = price_monte_carlo(
                    MODEL, product, 1., n_paths, n_steps=steps,
                    rng=stream(seed, case_index, replication, method_index),
                    variance_reduction=method,
                    n_pilot_paths=n_pilot if method == "control_variate" else None,
                )
                row = asdict(result)
                lower, upper = row.pop("confidence_interval")
                row.pop("variance_reduction")
                row.update(case=name, case_index=case_index, replication=replication,
                           method=method, spawn_key=json.dumps([case_index, replication, method_index]),
                           ci_lower=lower, ci_upper=upper, estimator_variance=result.estimator_variance,
                           payoff_evaluations=n_paths*(2 if method == "control_variate" else 1))
                rows.append(row)
            samples = replay_control_samples(seed, case_index, replication, n_paths, n_pilot)
            control = price_from_payoff_samples(samples["production_control"], rate=.05, maturity=1.)
            checks.append({"case": name, "case_index": case_index, "replication": replication,
                           "price": control.price, "analytical_price": exact_geometric,
                           "standard_error": control.standard_error,
                           "ci_lower": control.confidence_interval[0], "ci_upper": control.confidence_interval[1],
                           "production_correlation": np.corrcoef(samples["production_target"],
                                                                  samples["production_control"])[0, 1]})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def summarize_comparison(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variance of estimates across runs, never pooled payoff variance.

    F intervals are approximate: independent near-normal run-level estimates.
    Production normality is asymptotic; random pilot slopes produce a mixture.
    No arithmetic bias/RMSE is reported without a known reference value.
    """
    rows = []
    for (case, method), group in raw.groupby(["case", "method"], sort=False):
        if len(group) < 2 or any(group[x].nunique() != 1 for x in
                                ("n_paths", "n_independent_samples", "n_pilot_paths")):
            raise ValueError("need repeated runs with identical method budgets")
        variance = group.price.var(ddof=1)
        rows.append({"case": case, "method": method, "n_replications": len(group),
                     "n_paths": int(group.n_paths.iloc[0]),
                     "n_pilot_paths": int(group.n_pilot_paths.iloc[0]),
                     "n_independent_samples": int(group.n_independent_samples.iloc[0]),
                     "mean_price": group.price.mean(), "empirical_estimator_variance": variance,
                     "mean_reported_estimator_variance": group.estimator_variance.mean(),
                     "rms_reported_standard_error": np.sqrt(group.estimator_variance.mean()),
                     "empirical_standard_error_of_mean": np.sqrt(variance/len(group)),
                     "mean_runtime_seconds": group.runtime_seconds.mean(),
                     "median_runtime_seconds": group.runtime_seconds.median(),
                     "mean_beta": group.control_beta.mean(),
                     "mean_pilot_correlation": group.pilot_correlation.mean(),
                     "fallback_count": int((group.control_fit_status == "zero_control_variance").sum())})
    summary = pd.DataFrame(rows)
    comparisons = []
    for case, group in summary.groupby("case", sort=False):
        if set(group.method) != set(METHODS) or group.n_paths.nunique() != 1:
            raise ValueError("all three methods must share a total path budget")
        plain = group.loc[group.method == "none"].iloc[0]
        for other in group.loc[group.method != "none"].itertuples():
            v0, v1 = plain.empirical_estimator_variance, other.empirical_estimator_variance
            vrf = v0/v1 if min(v0, v1) > 0 else np.nan
            r0, r1 = plain.mean_reported_estimator_variance, other.mean_reported_estimator_variance
            reported = r0/r1 if min(r0, r1) > 0 else np.nan
            gap = other.mean_price-plain.mean_price
            se = np.hypot(plain.empirical_standard_error_of_mean, other.empirical_standard_error_of_mean)
            comparisons.append({"case": case, "method": other.method, "empirical_vrf": vrf,
                                "vrf_ci_lower": vrf/f.ppf(.975, plain.n_replications-1, other.n_replications-1),
                                "vrf_ci_upper": vrf/f.ppf(.025, plain.n_replications-1, other.n_replications-1),
                                "reported_variance_vrf": reported,
                                "variance_reduction_percent": 100*(1-1/vrf),
                                "time_adjusted_efficiency": vrf*plain.mean_runtime_seconds/other.mean_runtime_seconds,
                                "method_minus_plain_mean": gap, "difference_standard_error": se,
                                "difference_ci_lower": gap-1.96*se, "difference_ci_upper": gap+1.96*se})
    return summary, pd.DataFrame(comparisons)


def summarize_geometric(checks: pd.DataFrame) -> pd.DataFrame:
    """Known-target validation is for the geometric control, not the arithmetic target."""
    rows = []
    for case, group in checks.groupby("case", sort=False):
        errors = group.price-group.analytical_price
        hits = int(((group.ci_lower <= group.analytical_price) &
                    (group.analytical_price <= group.ci_upper)).sum())
        low, high = binomial_wilson_interval(hits, len(group))
        rows.append({"case": case, "analytical_price": group.analytical_price.iloc[0],
                     "mean_price": group.price.mean(), "empirical_bias": errors.mean(),
                     "empirical_bias_standard_error": errors.std(ddof=1)/np.sqrt(len(group)),
                     "rmse": np.sqrt(np.mean(errors**2)), "coverage": hits/len(group),
                     "coverage_ci_lower": low, "coverage_ci_upper": high,
                     "mean_production_correlation": group.production_correlation.mean()})
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, comparison: pd.DataFrame, path: Path) -> None:
    labels = [x[0].replace("asian_", "").replace("_", " / ") for x in CASES]
    colors = ("#526D82", "#E39435", "#008B7D")
    names = ("Plain MC", "Antithetic", "Geometric control")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    x = np.arange(len(CASES))
    for index, method in enumerate(METHODS):
        data = summary.loc[summary.method == method].set_index("case").loc[[x[0] for x in CASES]]
        axes[0].bar(x+(index-1)*.24, data.empirical_estimator_variance, width=.23,
                    color=colors[index], label=names[index])
        if method != "none":
            ratios = comparison.loc[comparison.method == method].set_index("case").loc[[x[0] for x in CASES]]
            y = ratios.empirical_vrf.to_numpy()
            err = np.array([y-ratios.vrf_ci_lower.to_numpy(), ratios.vrf_ci_upper.to_numpy()-y])
            axes[1].errorbar(x+(index-1.5)*.12, y, yerr=err, fmt="o", capsize=4,
                             color=colors[index], label=names[index])
    axes[0].set(title="Variance across independent price estimates", ylabel="Estimator variance (currency²)")
    axes[1].set(title="Variance reduction with approximate 95% intervals", ylabel="Plain variance / method variance")
    axes[1].axhline(1, color="grey", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Arithmetic Asian contract / monitoring dates")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    row = summary.iloc[0]
    pilot = int(summary.loc[summary.method == "control_variate", "n_pilot_paths"].iloc[0])
    fig.suptitle(f"{int(row.n_replications)} runs per method • {int(row.n_paths):,} total paths/run • CV pilot {pilot:,}")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=10000)
    parser.add_argument("--n-pilot", type=int, default=1000)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]/"results")
    args = parser.parse_args()
    for directory in ("raw", "tables", "figures"):
        (args.output_dir/directory).mkdir(parents=True, exist_ok=True)
    raw, checks = collect_comparison(args.n_paths, args.n_pilot, args.replications, args.seed)
    summary, comparison = summarize_comparison(raw)
    geo = summarize_geometric(checks)
    artifacts = []
    for directory, name, frame in (("raw", "control_replications", raw),
                                    ("raw", "geometric_control_replications", checks),
                                    ("tables", "control_estimators", summary),
                                    ("tables", "control_comparison", comparison),
                                    ("tables", "geometric_control_validation", geo)):
        path = args.output_dir/directory/f"{name}.csv"
        frame.to_csv(path, index=False)
        artifacts.append(path)
    sample_path = args.output_dir/"raw/control_samples_case0_rep0.npz"
    np.savez_compressed(sample_path, **replay_control_samples(args.seed, 0, 0, args.n_paths, args.n_pilot))
    artifacts.append(sample_path)
    plot_comparison(summary, comparison, args.output_dir/"figures/control_variate_variance_reduction.png")
    metadata = {"milestone": 10, "model": asdict(MODEL), "strike": 100., "maturity": 1.,
                "monitoring": "equal future dates jT/m, excludes S0", "cases": CASES,
                "n_paths": args.n_paths, "n_pilot": args.n_pilot, "replications": args.replications,
                "seed": args.seed, "bit_generator": "PCG64",
                "stream_design": "SeedSequence(seed, spawn_key=(case, replication, method)); CV successive pilot/production blocks",
                "budget": "equal total stock paths including pilot; CV evaluates two payoffs per path",
                "timing": "full pricer including pilot, analytical control and fit; excludes RNG construction, warmup, diagnostic replay, I/O",
                "uncertainty": "normal per-run CIs; approximate F VRF intervals, independent near-normal run estimates",
                "reference": "geometric exact only; no arithmetic bias/RMSE claim",
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "sha256": {str(p.relative_to(args.output_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}
    (args.output_dir/"raw/control_study.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(summary[["case", "method", "mean_price", "rms_reported_standard_error", "mean_runtime_seconds"]].to_string(index=False))
    print(comparison[["case", "method", "empirical_vrf", "vrf_ci_lower", "vrf_ci_upper", "variance_reduction_percent"]].to_string(index=False))
    print(geo.to_string(index=False))


if __name__ == "__main__":
    main()
