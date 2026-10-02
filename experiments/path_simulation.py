"""Milestone 6: exact GBM grid paths, joint moments and coupled grid checks.

Run: python -m experiments.path_simulation --n-paths 50000 --n-steps 252
Only a small path preview, terminal states and aggregate shocks are saved.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from math import sqrt
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

from mc_pricer import (
    BlackScholesModel, EuropeanOption, black_scholes_price, make_rng,
    paths_from_normals, price_from_terminal_samples, standard_normal_matrix,
    terminal_from_normals,
)


def marginal_table(paths: np.ndarray, model: BlackScholesModel, maturity: float) -> pd.DataFrame:
    """Per-date stock moments; N paths are the independent units at each date."""
    count, columns = paths.shape
    times = np.linspace(0, maturity, columns)
    discount = np.exp(-model.rate * times)
    mean = paths.mean(axis=0)
    variance = paths.var(axis=0, ddof=1)
    theoretical_mean = model.spot * np.exp(model.rate * times)
    theoretical_variance = theoretical_mean**2 * np.expm1(model.volatility**2 * times)
    discounted_mean = discount * mean
    sample_se = discount * np.sqrt(variance / count)
    theoretical_se = discount * np.sqrt(theoretical_variance / count)
    standardized_error = np.divide(discounted_mean - model.spot, theoretical_se,
                                    out=np.zeros(columns), where=theoretical_se > 0)
    critical = norm.ppf(.975)
    return pd.DataFrame({
        "step": np.arange(columns), "time_years": times, "n_paths": count,
        "mean_spot": mean, "theoretical_mean_spot": theoretical_mean,
        "sample_spot_variance": variance, "theoretical_spot_variance": theoretical_variance,
        "discounted_mean": discounted_mean, "discounted_mean_target": model.spot,
        "discounted_mean_se": sample_se, "theoretical_discounted_mean_se": theoretical_se,
        "discounted_ci_lower": discounted_mean - critical * sample_se,
        "discounted_ci_upper": discounted_mean + critical * sample_se,
        "discounted_error_in_theoretical_se": standardized_error,
    })


def increment_table(paths: np.ndarray, model: BlackScholesModel, maturity: float) -> pd.DataFrame:
    """Normal-increment moments at prespecified steps and adjacent covariance."""
    count, columns = paths.shape
    steps = columns - 1
    dt = maturity / steps
    target_mean = (model.rate - .5 * model.volatility**2) * dt
    target_variance = model.volatility**2 * dt
    rows = []
    for step in sorted({1, steps // 2, steps}):
        increments = np.log(paths[:, step] / paths[:, step - 1])
        metrics = [
            ("log_increment_mean", float(increments.mean()), target_mean, sqrt(target_variance / count)),
            ("log_increment_variance", float(increments.var(ddof=1)), target_variance,
             target_variance * sqrt(2 / (count - 1))),
        ]
        for name, estimate, target, se in metrics:
            rows.append({"metric": name, "step": step, "estimate": estimate, "theoretical": target,
                         "theoretical_standard_error": se, "error_in_standard_errors": (estimate - target) / se})
    first = np.log(paths[:, 1] / paths[:, 0])
    second = np.log(paths[:, 2] / paths[:, 1])
    covariance = float(np.cov(first, second, ddof=1)[0, 1])
    se = target_variance / sqrt(count - 1)
    rows.append({"metric": "covariance_steps_1_2", "step": 2, "estimate": covariance,
                 "theoretical": 0.0, "theoretical_standard_error": se,
                 "error_in_standard_errors": covariance / se})
    return pd.DataFrame(rows)


def covariance_table(paths: np.ndarray, model: BlackScholesModel, maturity: float) -> pd.DataFrame:
    """Joint log-level covariance at roughly quarter, half and final maturity."""
    count, columns = paths.shape
    steps = columns - 1
    indices = np.array(sorted({steps // 4, steps // 2, steps}))
    times = indices * (maturity / steps)
    log_levels = np.log(paths[:, indices] / model.spot)
    sample = np.cov(log_levels, rowvar=False, ddof=1)
    target = model.volatility**2 * np.minimum.outer(times, times)
    # Exact variance of an unbiased sample covariance for joint-normal data.
    se = np.sqrt((np.outer(np.diag(target), np.diag(target)) + target**2) / (count - 1))
    rows = []
    for i, step_i in enumerate(indices):
        for j, step_j in enumerate(indices):
            rows.append({"step_i": int(step_i), "step_j": int(step_j),
                         "time_i_years": times[i], "time_j_years": times[j],
                         "sample_log_covariance": sample[i, j], "theoretical_log_covariance": target[i, j],
                         "theoretical_standard_error": se[i, j],
                         "error_in_standard_errors": (sample[i, j] - target[i, j]) / se[i, j]})
    return pd.DataFrame(rows)


def coupled_grid_table(
    paths: np.ndarray, normals: np.ndarray, model: BlackScholesModel,
    maturity: float, fine_transform_seconds: float,
) -> pd.DataFrame:
    """Couple coarser grids by summing Brownian increments over contiguous blocks.

    Each row reuses the same underlying fine noise. This is an algebraic grid
    consistency check, not independent statistical replications or a bias study.
    Timings exclude normal generation and block aggregation.
    """
    count, fine_steps = normals.shape
    counts = sorted({n for n in (1, 4, 12, 63, fine_steps) if fine_steps % n == 0})
    rows = []
    for steps in counts:
        block = fine_steps // steps
        if steps == fine_steps:
            # This last row is an explicitly labeled identity reference.
            max_error, max_relative, elapsed = 0.0, 0.0, fine_transform_seconds
        else:
            coarse_normals = normals.reshape(count, steps, block).sum(axis=2) / sqrt(block)
            start = perf_counter()
            coarse_paths = paths_from_normals(model, maturity, coarse_normals)
            elapsed = perf_counter() - start
            fine_at_shared_dates = paths[:, ::block]
            differences = np.abs(coarse_paths - fine_at_shared_dates)
            max_error = float(differences.max())
            max_relative = float(np.max(differences / fine_at_shared_dates))
            del coarse_normals, coarse_paths, differences
        rows.append({"n_paths": count, "n_steps": steps, "fine_n_steps": fine_steps,
                     "block_size": block, "identity_reference": steps == fine_steps,
                     "max_abs_common_date_error": max_error, "max_relative_common_date_error": max_relative,
                     "path_output_bytes": count * (steps + 1) * 8,
                     "normal_input_bytes": count * steps * 8,
                     "transform_runtime_seconds": elapsed})
    return pd.DataFrame(rows)


def plot_diagnostics(
    preview: np.ndarray, first_standardized_increment: np.ndarray,
    marginals: pd.DataFrame, coupling: pd.DataFrame, model: BlackScholesModel,
    destination: Path,
) -> None:
    """Three figures: paths/martingale, increment distribution, coupling/memory."""
    times = marginals["time_years"].to_numpy()
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
        axes[0].plot(times, preview.T, alpha=.35, lw=.8)
        axes[0].plot(times, marginals["mean_spot"], color="#136F8A", lw=2, label="Full-sample mean")
        axes[0].plot(times, marginals["theoretical_mean_spot"], color="#303840", ls="--", label="Theoretical mean")
        axes[0].set(title=f"First {len(preview)} paths; lines join simulated dates", xlabel="Time (years)", ylabel="Asset price (currency units)")
        axes[1].plot(times, marginals["discounted_mean"], color="#136F8A", label="Mean discounted stock")
        axes[1].fill_between(times, marginals["discounted_ci_lower"], marginals["discounted_ci_upper"],
                             color="#136F8A", alpha=.18, label="Pointwise 95% normal CI")
        axes[1].axhline(model.spot, color="#303840", ls="--", label="Theoretical expectation S₀")
        axes[1].set(title="Unconditional discounted-mean diagnostic", xlabel="Time (years)", ylabel="Discounted stock (currency units)")
        for axis in axes:
            axis.grid(alpha=.18)
            axis.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Exact GBM grid simulation | N = {int(marginals.n_paths.iloc[0]):,}, steps = {len(times)-1}", fontsize=12)
        fig.savefig(destination / "gbm_paths.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
        axes[0].hist(first_standardized_increment, bins=70, density=True, color="#136F8A", alpha=.65, label="Recovered first-step shocks")
        support = np.linspace(-4, 4, 400)
        axes[0].plot(support, norm.pdf(support), color="#303840", label="Standard normal density")
        axes[0].set(title="Distribution recovered from simulated prices", xlabel="Standardized log increment (dimensionless)", ylabel="Probability density")
        probabilities = np.linspace(.01, .99, 99)
        theoretical = norm.ppf(probabilities)
        axes[1].plot(theoretical, np.quantile(first_standardized_increment, probabilities), "o", ms=3, color="#136F8A", label="1%–99% sample quantiles")
        axes[1].plot(theoretical, theoretical, ls="--", color="#303840", label="Normal reference")
        axes[1].set(title="Normal quantile comparison", xlabel="Theoretical normal quantile", ylabel="Recovered-shock sample quantile")
        for axis in axes:
            axis.grid(alpha=.18)
            axis.legend(frameon=False, fontsize=8)
        fig.savefig(destination / "gbm_increment_diagnostics.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
        comparisons = coupling.loc[~coupling["identity_reference"]]
        axes[0].semilogx(comparisons["n_steps"], comparisons["max_abs_common_date_error"], "o", color="#136F8A")
        axes[0].set(title="Agreement at shared grid dates", xlabel="Coarse-grid steps (log scale)", ylabel="Maximum shared-date price difference (currency)", ylim=(0, None))
        axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[1].semilogx(coupling["n_steps"], coupling["path_output_bytes"] / 2**20, "o-", label="Path output only", color="#136F8A")
        axes[1].semilogx(coupling["n_steps"], (coupling["path_output_bytes"] + coupling["normal_input_bytes"]) / 2**20,
                        "s--", label="Output plus supplied shocks", color="#CB6632")
        axes[1].set(title="Storage of explicit arrays; excludes temporaries", xlabel="Number of time steps (log scale)", ylabel="Storage (MiB; 1 MiB = 2²⁰ bytes)")
        axes[1].legend(frameon=False, fontsize=8)
        for axis in axes:
            axis.grid(alpha=.18)
        fig.suptitle("Coupled-grid consistency and memory requirements", fontsize=12)
        fig.savefig(destination / "gbm_grid_coupling.png", dpi=170)
        plt.close(fig)


def run(output_dir: Path, n_paths: int, n_steps: int, seed: int) -> None:
    """Run one fixed-seed validation, retain compact evidence and print results."""
    if n_paths < 2 or n_steps < 4:
        raise ValueError("this diagnostic experiment requires n_paths >= 2 and n_steps >= 4")
    model = BlackScholesModel(100.0, .05, .20)
    maturity, strike = 1.0, 100.0
    rng = make_rng(seed)
    start = perf_counter()
    normals = standard_normal_matrix(n_paths, n_steps, rng=rng)
    normal_seconds = perf_counter() - start
    start = perf_counter()
    paths = paths_from_normals(model, maturity, normals)
    transform_seconds = perf_counter() - start
    aggregated_normals = normals.sum(axis=1) / sqrt(n_steps)
    terminal_reference = terminal_from_normals(model, maturity, aggregated_normals)
    endpoint_max_abs = float(np.max(np.abs(paths[:, -1] - terminal_reference)))
    coupling = coupled_grid_table(paths, normals, model, maturity, transform_seconds)
    preview_normals = normals[:min(32, n_paths)].copy()
    del normals
    marginals = marginal_table(paths, model, maturity)
    increments = increment_table(paths, model, maturity)
    covariance = covariance_table(paths, model, maturity)
    pricing_rows = []
    for kind in ("call", "put"):
        product = EuropeanOption(strike, kind)
        grid_result = price_from_terminal_samples(product, paths[:, -1], rate=model.rate, maturity=maturity)
        terminal_result = price_from_terminal_samples(product, terminal_reference, rate=model.rate, maturity=maturity)
        target = black_scholes_price(model, strike, maturity, kind)
        pricing_rows.append({"product": kind, "n_paths": n_paths, "n_steps": n_steps,
                             "price_from_paths": grid_result.price, "price_from_aggregated_normal": terminal_result.price,
                             "bs_price": target, "absolute_error": abs(grid_result.price - target),
                             "standard_error": grid_result.standard_error,
                             "signed_error_in_se": (grid_result.price - target) / grid_result.standard_error if grid_result.standard_error > 0 else None,
                             "ci_lower": grid_result.confidence_interval[0], "ci_upper": grid_result.confidence_interval[1],
                             "pricing_only_runtime_seconds": grid_result.runtime_seconds})
    prices = pd.DataFrame(pricing_rows)
    preview = paths[:min(32, n_paths)].copy()
    dt = maturity / n_steps
    first_standardized = (np.log(paths[:, 1] / model.spot) - (model.rate - .5 * model.volatility**2) * dt) / (model.volatility * sqrt(dt))
    for directory in ("raw", "tables", "figures"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    for name, table in (("gbm_path_marginals", marginals), ("gbm_path_increments", increments),
                        ("gbm_path_covariance", covariance), ("gbm_grid_coupling", coupling),
                        ("gbm_path_european_validation", prices)):
        table.to_csv(output_dir / "tables" / f"{name}.csv", index=False)
    raw_path = output_dir / "raw" / "gbm_path_samples.npz"
    np.savez_compressed(raw_path, preview_normals=preview_normals, preview_paths=preview,
                        terminal=paths[:, -1], aggregated_normals=aggregated_normals,
                        times=marginals["time_years"].to_numpy())
    metadata = {
        "milestone": 6, "model": asdict(model), "maturity_years": maturity, "strike": strike,
        "n_paths": n_paths, "n_steps": n_steps, "seed": seed,
        "bit_generator": type(rng.bit_generator).__name__,
        "draw_call": "Generator.standard_normal((n_paths, n_steps), dtype=float64)",
        "path_shape": list(paths.shape), "path_dtype": str(paths.dtype),
        "path_output_bytes": paths.nbytes, "normal_input_bytes": n_paths * n_steps * 8,
        "memory_scope": "Explicit ndarray storage, not measured peak RSS. Input shocks and path output coexist. "
                        "Coarse-grid checks, variance calculations along array axes and boolean validation allocate temporaries.",
        "normal_generation_seconds": normal_seconds, "fine_transform_seconds": transform_seconds,
        "runtime_scope": "Single wall-clock observations. Draw and path transformation timed separately. "
                         "Diagnostics, aggregation, serialization and plots excluded from these two timings.",
        "endpoint_max_abs_difference": endpoint_max_abs,
        "coupling_design": "Coarse Z = sum of each block of fine Z / sqrt(block size). "
                           "Common-date differences should be rounding scale; fine-grid self-row is an identity reference.",
        "raw_scope": "First min(32,N) shock/path rows, all terminals, aggregate normals, time grid. "
                     "Full path matrix is regenerated with the recorded seed, shape and environment.",
        "raw_npz_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "limitations": "One fixed-seed validation, not coverage estimation or an efficiency ranking. "
                       "Per-date means/CIs are dependent and intervals are pointwise, not simultaneous. "
                       "Grid transitions are exact; between-date crossings and continuous averages are not observed.",
    }
    (output_dir / "raw" / "path_simulation.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_diagnostics(preview, first_standardized, marginals, coupling, model, output_dir / "figures")
    print(prices.to_string(index=False, float_format=lambda x: f"{x:.9g}"))
    print(increments.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(coupling.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"Endpoint aggregation max difference: {endpoint_max_abs:.3g}")
    print(f"Largest absolute joint-covariance discrepancy / theoretical SE: {covariance.error_in_standard_errors.abs().max():.4f}")
    print(f"Path output: {paths.nbytes:,} bytes; shocks: {n_paths * n_steps * 8:,} bytes; excludes temporaries.")
    print(f"Normal draws: {normal_seconds:.3f}s; fine path transform: {transform_seconds:.3f}s.")
    print(f"Saved results to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-paths", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=252)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if args.n_paths < 2 or args.n_steps < 4 or args.seed < 0:
        parser.error("require n_paths >= 2, n_steps >= 4, seed >= 0")
    run(args.output_dir, args.n_paths, args.n_steps, args.seed)


if __name__ == "__main__":
    main()
