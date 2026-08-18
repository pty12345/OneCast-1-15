#!/usr/bin/env python3
"""Summarize predictor-level holdout logs into a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "predictor_holdout"
DATASETS = ["ETTh2", "ETTm2", "weather", "traffic", "CzeLan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--targets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--pred-lens", nargs="+", type=int, default=[48])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--few-shot-ratios", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def ratio_tag(ratio: float) -> str:
    return f"{ratio * 100:g}pct".replace(".", "p")


def parse_metrics(log_path: Path, target: str) -> dict[str, float] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    marker = re.compile(rf"Testing\s+{re.escape(target)}\.\.\.", re.I)
    sections = marker.split(text)
    if len(sections) < 2:
        return None
    matches = re.findall(
        r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)", sections[-1]
    )
    if not matches:
        return None
    mse, mae = matches[-1]
    return {"mse": float(mse), "mae": float(mae)}


def collect_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    log_root = args.output_root / "logs"
    for seed in args.seeds:
        for pred_len in args.pred_lens:
            for target in args.targets:
                configurations = [
                    (
                        "zero_shot",
                        0.0,
                        log_root / f"seed{seed}" / f"zs_test_leave_{target}_h{pred_len}.log",
                    )
                ]
                configurations.extend(
                    (
                        "few_shot",
                        ratio,
                        log_root
                        / f"seed{seed}"
                        / f"fs_{ratio_tag(ratio)}_leave_{target}_h{pred_len}.log",
                    )
                    for ratio in args.few_shot_ratios
                )
                for mode, ratio, log_path in configurations:
                    metrics = parse_metrics(log_path, target)
                    rows.append(
                        {
                            "seed": seed,
                            "target": target,
                            "pred_len": pred_len,
                            "mode": mode,
                            "few_shot_ratio": ratio,
                            "mse": "" if metrics is None else metrics["mse"],
                            "mae": "" if metrics is None else metrics["mae"],
                            "status": "missing" if metrics is None else "complete",
                            "log_path": str(log_path),
                        }
                    )
    return rows


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    output = args.output or args.output_root / "predictor_holdout_summary.csv"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"Wrote {output} ({complete}/{len(rows)} completed configurations)")


if __name__ == "__main__":
    main()
