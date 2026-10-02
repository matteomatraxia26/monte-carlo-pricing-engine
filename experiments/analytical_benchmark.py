"""Milestone 1: deterministic benchmarks, identities and analytical curves.

Run from the repository root: python -m experiments.analytical_benchmark
This is not a Monte Carlo performance or convergence experiment.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from mc_pricer import BlackScholesModel, black_scholes_greeks, black_scholes_price


def run(output_dir: Path) -> None:
    """Write baseline CSV, curve CSV, raw metadata and a four-panel PNG."""
    for subdirectory in ("tables", "raw", "figures"):
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    model = BlackScholesModel(spot=100.0, rate=0.05, volatility=0.20)
    strike, maturity = 100.0, 1.0
    # These rounded reference constants are fixed independently of this run.
    expected = {"call": 10.450583572185565, "put": 5.573526022256971}
    rows = []
    for option_type in ("call", "put"):
        start = perf_counter()
        price = black_scholes_price(model, strike, maturity, option_type)
        greeks = black_scholes_greeks(model, strike, maturity, option_type)
        elapsed = perf_counter() - start
        rows.append({
            "option_type": option_type,
            "price": price,
            "reference_price": expected[option_type],
            "absolute_error": abs(price - expected[option_type]),
            **asdict(greeks),
            "vega_per_vol_point": greeks.vega / 100,
            "runtime_seconds": elapsed,
        })
    baseline = pd.DataFrame(rows)
    baseline.to_csv(output_dir / "tables" / "analytical_benchmark.csv", index=False)

    curve_rows = []
    for spot in np.linspace(50.0, 150.0, 201):
        current = replace(model, spot=float(spot))
        call_greeks = black_scholes_greeks(current, strike, maturity, "call")
        put_greeks = black_scholes_greeks(current, strike, maturity, "put")
        curve_rows.append({
            "spot": spot,
            "call": black_scholes_price(current, strike, maturity, "call"),
            "put": black_scholes_price(current, strike, maturity, "put"),
            "call_delta": call_greeks.delta,
            "put_delta": put_greeks.delta,
            "gamma": call_greeks.gamma,
            "vega_per_vol_point": call_greeks.vega / 100,
        })
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(output_dir / "raw" / "analytical_curves.csv", index=False)
    parity_residual = float(
        baseline.iloc[0]["price"] - baseline.iloc[1]["price"]
        - (model.spot - strike * np.exp(-model.rate * maturity))
    )
    metadata = {
        "milestone": 1,
        "model": asdict(model),
        "strike": strike,
        "maturity_years": maturity,
        "seed": None,
        "reason_no_seed": "Deterministic analytical experiment; no simulation.",
        "put_call_parity_residual": parity_residual,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "timing_note": "Single calls including first-call overhead; not a speed comparison.",
        "baseline": rows,
    }
    (output_dir / "raw" / "analytical_benchmark.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        panels = [
            (axes[0, 0], [("call", "Call"), ("put", "Put")],
             "European option prices", "Price (currency units)"),
            (axes[0, 1], [("call_delta", "Call"), ("put_delta", "Put")],
             "Spot Delta", "Delta (price change / spot change)"),
            (axes[1, 0], [("gamma", "Call and put")],
             "Spot Gamma", "Gamma (1 / currency unit)"),
            (axes[1, 1], [("vega_per_vol_point", "Call and put")],
             "Vega per volatility point", "Price change per +1 percentage point vol"),
        ]
        for ax, series, title, ylabel in panels:
            for (column, label), color in zip(series, ("#136F8A", "#CB6632")):
                ax.plot(curves["spot"], curves[column], label=label, color=color, lw=2)
            ax.axvline(strike, color="#9AA4AD", ls="--", lw=1, label="Strike")
            ax.set(title=title, xlabel="Spot (currency units)", ylabel=ylabel)
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle("Black–Scholes analytical benchmarks | K = 100, r = 5%, σ = 20%, T = 1 year",
                     fontsize=13)
        fig.savefig(output_dir / "figures" / "analytical_benchmark.png", dpi=170)
        plt.close(fig)
    print(baseline.to_string(index=False, float_format=lambda value: f"{value:.10g}"))
    print(f"Put–call parity residual: {parity_residual:.3e}")
    print(f"Saved results to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
