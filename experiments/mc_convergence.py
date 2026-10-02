"""Milestone 5: independent-replication convergence and confidence-interval study.

Run from the repository root: python -m experiments.mc_convergence
Stores estimator-level raw records, not hundreds of millions of terminal states.
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from mc_pricer import (
    BlackScholesModel, EuropeanOption, black_scholes_price,
    price_from_terminal_samples, simulate_terminal,
)
from mc_pricer.statistics import (
    binomial_wilson_interval, fit_power_law, summarize_replications,
)
from mc_pricer.statistics.confidence_intervals import validate_confidence_level

DEFAULT_COUNTS = (1_000, 2_000, 5_000, 10_000, 20_000,
                  50_000, 100_000, 200_000, 500_000, 1_000_000)


def collect_replications(
    model: BlackScholesModel, strike: float, maturity: float,
    path_counts: Sequence[int], *, n_replications: int, seed: int,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Collect independent runs for each N; call/put share each run's terminals.

    The SeedSequence tree has spawn_key=(count_index, replication). Distinct
    cells use distinct child streams. At each N, R prices for a given product
    are independent replications. Call/put rows in a cell are paired.
    Memory is O(max(path_counts)), plus O(R * len(path_counts)) scalar records.
    """
    counts = tuple(path_counts)
    if len(counts) < 3:
        raise ValueError("at least three path counts are required for a slope")
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in counts):
        raise TypeError("path_counts must contain integers")
    if any(n < 2 for n in counts) or any(a >= b for a, b in zip(counts, counts[1:])):
        raise ValueError("path counts must be strictly increasing and >= 2")
    for name, value in (("n_replications", n_replications), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if n_replications < 2 or seed < 0:
        raise ValueError("require n_replications >= 2 and seed >= 0")
    level = validate_confidence_level(confidence_level)
    products = [EuropeanOption(strike, kind) for kind in ("call", "put")]
    targets = {p.option_type: black_scholes_price(model, strike, maturity, p.option_type)
               for p in products}
    root = np.random.SeedSequence(int(seed))
    rows = []
    for count_index, (count, count_stream) in enumerate(zip(counts, root.spawn(len(counts)))):
        for replication, child in enumerate(count_stream.spawn(int(n_replications))):
            rng = np.random.default_rng(child)
            start = perf_counter()
            terminal = simulate_terminal(model, maturity, int(count), rng=rng)
            simulation_seconds = perf_counter() - start
            for product in products:
                result = price_from_terminal_samples(
                    product, terminal, rate=model.rate, maturity=maturity,
                    confidence_level=level,
                )
                target = targets[product.option_type]
                error = result.price - target
                rows.append({
                    "product": product.option_type, "n_paths": int(count),
                    "count_index": count_index, "replication": replication,
                    "spawn_key": json.dumps(list(child.spawn_key)),
                    "estimate": result.price, "analytical_value": target,
                    "signed_error": error, "absolute_error": abs(error),
                    "standard_error": result.standard_error,
                    "estimator_variance": result.estimator_variance,
                    "confidence_level": level,
                    "ci_lower": result.confidence_interval[0],
                    "ci_upper": result.confidence_interval[1],
                    "covers_target": result.confidence_interval[0] <= target <= result.confidence_interval[1],
                    "zero_standard_error": result.standard_error == 0,
                    "signed_error_in_se": error / result.standard_error if result.standard_error else None,
                    "simulation_runtime_seconds": simulation_seconds,
                    "pricing_runtime_seconds": result.runtime_seconds,
                    "runtime_seconds": simulation_seconds + result.runtime_seconds,
                })
            # Do not retain the previous terminal vector while drawing the next.
            del terminal
    return pd.DataFrame(rows)


def summarize_study(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-N statistics, aggregate slopes and per-replication slopes.

    Input is the complete output of collect_replications. Fits are descriptive;
    uncertainty of the mean per-replication slope comes from independent curves,
    not the homoskedastic OLS residual formula. All diagnostic intervals are 95%.
    """
    summaries = []
    for (product, count), group in raw.groupby(["product", "n_paths"], sort=True):
        target = float(group["analytical_value"].iloc[0])
        stats = summarize_replications(group["estimate"].to_numpy(), target)
        covered = int(group["covers_target"].sum())
        coverage_interval = binomial_wilson_interval(covered, len(group))
        summaries.append({
            "product": product, "n_paths": int(count), "n_replications": len(group),
            "analytical_value": target, "mean_estimate": stats.mean_estimate,
            "empirical_bias": stats.empirical_bias,
            "bias_standard_error": stats.standard_error_of_mean,
            "bias_ci_lower": stats.bias_confidence_interval[0],
            "bias_ci_upper": stats.bias_confidence_interval[1],
            "empirical_variance": stats.empirical_variance,
            "empirical_sd": stats.empirical_variance ** 0.5, "rmse": stats.rmse,
            "mean_reported_se": float(group["standard_error"].mean()),
            "rms_reported_se": float(np.sqrt(group["estimator_variance"].mean())),
            "mean_reported_variance": float(group["estimator_variance"].mean()),
            "covered_runs": covered, "empirical_coverage": covered / len(group),
            "coverage_wilson_lower": coverage_interval[0],
            "coverage_wilson_upper": coverage_interval[1],
            "nominal_price_coverage": float(group["confidence_level"].iloc[0]),
            "coverage_interval_confidence_level": 0.95,
            "zero_se_runs": int(group["zero_standard_error"].sum()),
            "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
        })
    summary = pd.DataFrame(summaries)
    curve_rows = []
    for (product, replication), group in raw.groupby(["product", "replication"], sort=True):
        ordered = group.sort_values("n_paths")
        if (ordered["standard_error"] <= 0).any():
            raise ValueError("positive SE is required at every N for log-log fitting")
        fit = fit_power_law(ordered["n_paths"].to_numpy(), ordered["standard_error"].to_numpy())
        curve_rows.append({"product": product, "replication": int(replication), **asdict(fit)})
    curves = pd.DataFrame(curve_rows)
    fits = []
    for product, group in summary.groupby("product", sort=True):
        fit = fit_power_law(group["n_paths"].to_numpy(), group["mean_reported_se"].to_numpy())
        rmse_fit = fit_power_law(group["n_paths"].to_numpy(), group["rmse"].to_numpy())
        slopes = curves.loc[curves["product"] == product, "slope"].to_numpy()
        slope_stats = summarize_replications(slopes, target=-0.5)
        fits.append({
            "product": product, "n_points": fit.n_points, "n_replications": len(slopes),
            "log_mean_se_intercept": fit.intercept, "log_mean_se_slope": fit.slope,
            "log_mean_se_r_squared": fit.r_squared, "log_rmse_slope": rmse_fit.slope,
            "mean_replication_slope": slope_stats.mean_estimate,
            "replication_slope_sd": slope_stats.empirical_variance ** 0.5,
            "mean_slope_standard_error": slope_stats.standard_error_of_mean,
            "mean_slope_ci_lower": -0.5 + slope_stats.bias_confidence_interval[0],
            "mean_slope_ci_upper": -0.5 + slope_stats.bias_confidence_interval[1],
        })
    return summary, pd.DataFrame(fits), curves


def plot_study(raw: pd.DataFrame, summary: pd.DataFrame, fits: pd.DataFrame, destination: Path) -> None:
    """Save five scientific figures with explicit units and sampling design."""
    colors = {"call": "#136F8A", "put": "#CB6632"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for chart in ("price_vs_n", "absolute_error_vs_n", "standard_error_vs_n", "loglog_convergence", "ci_coverage"):
            fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
            for axis, product in zip(axes, ("call", "put")):
                group = summary.loc[summary["product"] == product].sort_values("n_paths")
                first = raw.loc[(raw["product"] == product) & (raw["replication"] == 0)].sort_values("n_paths")
                sizes = group["n_paths"].to_numpy()
                color = colors[product]
                if chart == "price_vs_n":
                    axis.errorbar(sizes, first["estimate"], yerr=first["ci_upper"] - first["estimate"],
                                  fmt="o-", capsize=3, color=color, label="Replication 0 ± price CI")
                    axis.axhline(group["analytical_value"].iloc[0], ls="--", color="#40464D", label="Black–Scholes")
                    axis.set(ylabel="Option price (currency units)")
                elif chart == "absolute_error_vs_n":
                    axis.plot(sizes, first["absolute_error"], "o--", color=color, alpha=.65,
                              label="Replication 0 absolute error")
                    axis.plot(sizes, group["rmse"], "s-", color="#40464D", label="RMSE across replications")
                    axis.set(yscale="log", ylabel="Pricing error (currency units; log scale)")
                elif chart == "standard_error_vs_n":
                    axis.plot(sizes, group["mean_reported_se"], "o-", color=color, label="Mean reported SE")
                    axis.plot(sizes, group["empirical_sd"], "x--", color="#40464D", label="SD of replicated prices")
                    axis.set(ylabel="Price uncertainty (currency units)")
                elif chart == "loglog_convergence":
                    fit = fits.loc[fits["product"] == product].iloc[0]
                    axis.plot(sizes, group["mean_reported_se"], "o", color=color, label="Mean reported SE")
                    axis.plot(sizes, np.exp(fit.log_mean_se_intercept) * sizes ** fit.log_mean_se_slope,
                              color=color, label=f"Fitted slope {fit.log_mean_se_slope:.5f}")
                    reference = group["mean_reported_se"].iloc[0] * (sizes / sizes[0]) ** (-.5)
                    axis.plot(sizes, reference, "--", color="#40464D", label=r"$N^{-1/2}$ reference (anchored at first N)")
                    axis.set(yscale="log", ylabel="Standard error (currency units; log scale)")
                else:
                    coverage = group["empirical_coverage"].to_numpy()
                    errors = np.vstack((coverage - group["coverage_wilson_lower"].to_numpy(),
                                        group["coverage_wilson_upper"].to_numpy() - coverage))
                    axis.errorbar(sizes, 100 * coverage, yerr=100 * errors, fmt="o", capsize=4,
                                  color=color, label="Observed coverage ± 95% Wilson CI")
                    nominal = group["nominal_price_coverage"].iloc[0]
                    axis.axhline(100 * nominal, ls="--", color="#40464D", label=f"Nominal {100 * nominal:g}%")
                    axis.set(ylabel="Price-interval coverage (%)")
                axis.set(xscale="log", xlabel="Number of paths N (log scale)", title=f"European {product}")
                axis.grid(alpha=.18)
                axis.legend(frameon=False, fontsize=8)
            titles = {
                "price_vs_n": "One prespecified replication: independent samples at each N",
                "absolute_error_vs_n": "Realized error fluctuates; RMSE summarizes repeated runs",
                "standard_error_vs_n": "Reported uncertainty versus observed dispersion",
                "loglog_convergence": "Empirical standard-error scaling under exact GBM simulation",
                "ci_coverage": "Coverage uncertainty: pointwise intervals, not simultaneous bands",
            }
            count = int(summary["n_replications"].iloc[0])
            fig.suptitle(f"{titles[chart]}\n{count} independent replications per N; call/put paired within each run", fontsize=11)
            fig.savefig(destination / f"mc_{chart}.png", dpi=170)
            plt.close(fig)


def run(output_dir: Path, path_counts: Sequence[int], n_replications: int, seed: int) -> None:
    """Execute the baseline study and persist raw estimates, summaries and plots."""
    model = BlackScholesModel(100.0, .05, .20)
    start = perf_counter()
    raw = collect_replications(model, 100.0, 1.0, path_counts,
                               n_replications=n_replications, seed=seed)
    simulation_and_collection_seconds = perf_counter() - start
    summary, fits, curves = summarize_study(raw)
    for folder in ("raw", "tables", "figures"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw" / "mc_convergence_replications.csv"
    raw.to_csv(raw_path, index=False)
    curves.to_csv(output_dir / "raw" / "mc_convergence_slopes_by_replication.csv", index=False)
    summary.to_csv(output_dir / "tables" / "mc_convergence.csv", index=False)
    fits.to_csv(output_dir / "tables" / "mc_convergence_slopes.csv", index=False)
    metadata = {
        "milestone": 5, "model": asdict(model), "strike": 100.0, "maturity_years": 1.0,
        "path_counts": [int(n) for n in path_counts], "n_replications": n_replications,
        "root_seed": seed, "bit_generator": type(np.random.default_rng(seed).bit_generator).__name__,
        "stream_design": "SeedSequence(seed).spawn(len(path_counts))[count_index].spawn(R)[replication]. "
                         "Distinct streams per N/replication; paired call/put terminals. "
                         "No nested sample counts. Spawn keys recorded per row.",
        "draw_call": "Generator.standard_normal(n_paths, dtype=float64), once per cell",
        "raw_records": len(raw), "total_terminal_draws": int(sum(path_counts) * n_replications),
        "largest_terminal_vector_bytes": int(max(path_counts) * 8),
        "storage_scope": "All estimator-level outputs retained. Terminal draws discarded after "
                         "each cell; reconstruct from root entropy and spawn_key in recorded environment.",
        "confidence_levels": {"price": .95, "coverage_wilson": .95, "bias_normal": .95, "mean_slope_normal": .95},
        "fit_scope": "OLS of log mean reported SE against log N; no OLS residual CI. "
                     "Separate fits per independent replication quantify mean-slope uncertainty. "
                     "This finite-grid mean slope is not the asymptotic exponent itself.",
        "runtime_scope": "Each raw row sums shared terminal simulation time and that product's "
                         "supplied-sample pricing time. RNG construction and bookkeeping excluded. "
                         "Do not sum row runtimes for study wall time: simulation is shared.",
        "simulation_and_collection_seconds": simulation_and_collection_seconds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "raw_csv_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "limitations": "Pointwise coverage intervals; multiple rows are not a simultaneous test. "
                       "Finite replications, finite N, one baseline, normal price approximation, "
                       "no model-risk quantification or estimator-efficiency comparison.",
    }
    (output_dir / "raw" / "mc_convergence.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_study(raw, summary, fits, output_dir / "figures")
    print(fits.to_string(index=False, float_format=lambda x: f"{x:.7g}"))
    print(summary[["product", "n_paths", "mean_reported_se", "empirical_sd", "empirical_bias", "rmse",
                   "empirical_coverage", "coverage_wilson_lower", "coverage_wilson_upper"]]
          .to_string(index=False, float_format=lambda x: f"{x:.7g}"))
    print(f"Collected {len(raw)} price records in {simulation_and_collection_seconds:.2f}s; saved to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--path-counts", type=int, nargs="+", default=DEFAULT_COUNTS)
    args = parser.parse_args()
    if args.replications < 2 or args.seed < 0 or len(args.path_counts) < 3:
        parser.error("require replications >= 2, seed >= 0, and at least three path counts")
    if any(n < 2 for n in args.path_counts) or any(a >= b for a, b in zip(args.path_counts, args.path_counts[1:])):
        parser.error("path counts must be strictly increasing integers >= 2")
    run(args.output_dir, args.path_counts, args.replications, args.seed)


if __name__ == "__main__":
    main()
