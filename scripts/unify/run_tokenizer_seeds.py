#!/usr/bin/env python3
"""Train the grouped time-series tokenizers for the reported random seeds."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_ROOT = ROOT / "TSTokenizer" / "checkpoints"
DEFAULT_LOG_ROOT = ROOT / "TSTokenizer" / "logs" / "Tokenizer"

SETUPS = {
    "large": {
        "pred_lens": (24, 48, 96, 192),
        "scripts": (
            "Cross_ETTh2_ETTm2_weather.sh",
            "Cross_Traffic.sh",
            "Cross_CzeLan.sh",
        ),
    },
    "small": {
        "pred_lens": (24, 36, 48, 60),
        "scripts": (
            "Cross_FRED_Covid_NYSE.sh",
            "Cross_Wike2000.sh",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setups", nargs="+", choices=SETUPS, default=["large", "small"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument(
        "--pred-lens",
        nargs="+",
        type=int,
        help="optional horizons; each must belong to every selected setup",
    )
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("ONECAST_DATA_ROOT", ROOT / "dataset")),
    )
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        default=Path(os.environ.get("ONECAST_TOKENIZER_ROOT", DEFAULT_TOKENIZER_ROOT)),
        help="checkpoint root; each seed is written to a seed<value> subdirectory",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path(os.environ.get("ONECAST_TOKENIZER_LOG_ROOT", DEFAULT_LOG_ROOT)),
        help="tokenizer log root; each seed is written to a seed<value> subdirectory",
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
        help="CUDA_VISIBLE_DEVICES value",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def pred_lens_for(setup: str, args: argparse.Namespace) -> tuple[int, ...]:
    available = SETUPS[setup]["pred_lens"]
    chosen = tuple(args.pred_lens) if args.pred_lens else available
    invalid = set(chosen) - set(available)
    if invalid:
        raise ValueError(
            f"Invalid {setup} horizons: {sorted(invalid)}; available: {available}"
        )
    return chosen


def validate_inputs(args: argparse.Namespace) -> None:
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")
    if len(set(args.setups)) != len(args.setups):
        raise ValueError("Setups must be distinct")
    if args.pred_lens and len(set(args.pred_lens)) != len(args.pred_lens):
        raise ValueError("Horizons must be distinct")
    for setup in args.setups:
        pred_lens_for(setup, args)
    if not args.dry_run and not args.data_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {args.data_root}")


def command_text(script: Path, env_values: dict[str, str]) -> str:
    assignments = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in env_values.items()
    )
    return f"{assignments} {shlex.join(['bash', str(script)])}"


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.tokenizer_root = args.tokenizer_root.expanduser().resolve()
    args.log_root = args.log_root.expanduser().resolve()
    validate_inputs(args)

    for seed in args.seeds:
        for setup in args.setups:
            horizons = " ".join(map(str, pred_lens_for(setup, args)))
            for script_name in SETUPS[setup]["scripts"]:
                script = ROOT / "TSTokenizer" / "scripts" / script_name
                env_values = {
                    "PYTHON_BIN": args.python,
                    "GPU_IDS": args.gpu,
                    "ONECAST_DATA_ROOT": str(args.data_root),
                    "ONECAST_RANDOM_SEED": str(seed),
                    "ONECAST_TOKENIZER_ROOT": str(args.tokenizer_root),
                    "ONECAST_TOKENIZER_LOG_ROOT": str(args.log_root),
                    "ONECAST_PRED_LENS": horizons,
                    "ONECAST_SKIP_EXISTING": "1" if args.skip_existing else "0",
                }
                display = command_text(script, env_values)
                if args.dry_run:
                    print(display)
                    continue

                print(
                    f"Running tokenizer group {script.stem}, setup {setup}, seed {seed}",
                    flush=True,
                )
                env = os.environ.copy()
                env.update(env_values)
                subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
