"""Run the staffing experiments and print aggregate results as Markdown tables.

    python scripts/run_experiment.py                      # synthetic data
    python scripts/run_experiment.py --data real          # CSVs in data/raw/

Prints three tables (reproduction of the original notebooks, the corrected train/test
experiment, and an ablation of each fix) and writes CSVs and PNG figures to ``--out``.
Every printed number is an aggregate, so the output is safe to share.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from staffing import plots  # noqa: E402
from staffing.config import load_config  # noqa: E402
from staffing.data import DEFAULT_DATA_DIR, load_dataset  # noqa: E402
from staffing.evaluate import CORRECTED, LEGACY, reproduce_original, run_ablation, run_experiment  # noqa: E402
from staffing.forecast import FORECASTERS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/)")
    ap.add_argument("--country", default=None, help="country to model (default: from config)")
    ap.add_argument("--seed", type=int, default=None, help="synthetic data seed")
    ap.add_argument("--model", choices=["corrected", "legacy"], default="corrected",
                    help="model specification for the main experiment")
    ap.add_argument("--policy", nargs="+", choices=["baseline", "lookahead"],
                    default=["baseline", "lookahead"])
    ap.add_argument("--forecaster", nargs="+", choices=sorted(FORECASTERS), default=None,
                    help="forecasters to test (default: all in config)")
    ap.add_argument("--no-reproduce", action="store_true", help="skip the original reproduction")
    ap.add_argument("--no-ablation", action="store_true", help="skip the ablation")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    country = args.country or cfg.country
    seed = args.seed if args.seed is not None else cfg.seed
    dataset = load_dataset(args.data, args.data_dir, seed)
    out = args.out / args.data
    out.mkdir(parents=True, exist_ok=True)
    print(f"# Results: {args.data} data, {country}\n")

    if not args.no_reproduce:
        with _timed("reproducing the original notebooks"):
            repro = reproduce_original(dataset, cfg, country)
        repro.to_csv(out / "reproduction.csv", index=False)
        _print("Original notebooks, reproduced (legacy model)",
               _money(repro, ["original_real_data", "reproduced"]))

    spec = CORRECTED if args.model == "corrected" else LEGACY
    with _timed(f"running the {spec.name} experiment"):
        exp = run_experiment(dataset, cfg, spec, country, args.policy, args.forecaster)
    exp.results.to_csv(out / "results.csv", index=False)
    exp.forecast_errors.to_csv(out / "forecast_errors.csv", index=False)
    _print(f"{spec.name.capitalize()} model: tuned on {cfg.train_year}, tested on {cfg.test_year}",
           _money(exp.results, ["profit", "revenue", "cost"]))
    _print(f"Forecast error on {cfg.test_year}", exp.forecast_errors.round(2))

    if not args.no_ablation:
        with _timed("running the ablation"):
            abl = run_ablation(dataset, cfg, country=country)
        abl.to_csv(out / "ablation.csv", index=False)
        _print(f"Ablation: profit on {cfg.test_year} with last year as forecast",
               _money(abl, ["profit"]))

    if not args.no_plots:
        _save_figures(exp, dataset, cfg, country, out / "figures")
    print(f"CSVs and figures written to {out}/", file=sys.stderr)
    return 0


def _save_figures(exp, dataset, cfg, country, fig_dir: Path) -> None:
    for policy in exp.tuning:
        daily = {fc: df for (p, fc), df in exp.daily.items() if p == policy and fc != "sarima"}
        name = plots.label(policy)
        plots.plot_daily(daily, "agents", f"{name} policy: agents employed, {cfg.test_year}",
                         "Agents", fig_dir / f"{policy}_agents.png")
        plots.plot_daily(daily, "idle_agents", f"{name} policy: idle agents, {cfg.test_year}",
                         "Agents with no advertisers", fig_dir / f"{policy}_idle.png")
        plots.plot_daily(daily, "waiting", f"{name} policy: advertisers waiting, {cfg.test_year}",
                         "Advertisers in the pool", fig_dir / f"{policy}_waiting.png")
    test_dates = pd.date_range(f"{cfg.test_year}-01-01", f"{cfg.test_year}-12-31", freq="D")
    actual = dataset.daily_demand(country, exp.spec.eligible_demand)[test_dates]
    plots.plot_forecasts(actual, exp.forecasts, fig_dir / "forecasts.png")


def _money(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].map(lambda v: f"${v / 1e6:,.1f}M")
    return df


def _print(title: str, df: pd.DataFrame) -> None:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    print(f"## {title}\n\n" + "\n".join(lines) + "\n")


class _timed:
    def __init__(self, what: str) -> None:
        self.what = what

    def __enter__(self) -> None:
        print(f"{self.what}...", file=sys.stderr, flush=True)
        self.start = time.perf_counter()

    def __exit__(self, *exc) -> None:
        print(f"  done in {time.perf_counter() - self.start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
