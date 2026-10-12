#!/usr/bin/env python3
"""Reproduce main-table predictor runs with one shared set of tokenizers."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "main_predictor_seeds"
VQ_TYPE = "W_SimVQ_decompose"
TOKENIZER_EPOCHS = 25

SETUPS = {
    "large": {
        "name": "Cross_large",
        "data": "Cross_large",
        "datasets": "ETTh2,ETTm2,weather,traffic,CzeLan",
        "seq_len": 96,
        "pred_lens": (24, 48, 96, 192),
        "token_len": 16,
        "wave_length": 8,
        "d_model": 64,
        "n_embed": 128,
        "infer_batch_size": 64,
        "groups": (
            ("Cross_ETTh2_ETTm2_weather", 256),
            ("Cross_traffic", 128),
            ("Cross_CzeLan", 128),
        ),
    },
    "small": {
        "name": "Cross_small",
        "data": "TFB",
        # run_longExp.py normalizes Cross_small to this order in the original script.
        "datasets": "FRED-MD,Covid-19,NYSE,Wike2000",
        "seq_len": 36,
        "pred_lens": (24, 36, 48, 60),
        "token_len": 12,
        "wave_length": 6,
        "d_model": 32,
        "n_embed": 32,
        "infer_batch_size": 32,
        "groups": (
            ("Cross_FRED_Covid_NYSE", 32),
            ("Cross_Wike2000", 32),
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
        "--pred-lens", nargs="+", type=int,
        help="optional horizons; each must belong to every selected setup",
    )
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("ONECAST_DATA_ROOT", ROOT / "dataset")),
    )
    parser.add_argument(
        "--text-tokenizer-root", type=Path,
        default=Path(os.environ.get("ONECAST_GPT2_ROOT", ROOT / "pretrained" / "GPT2")),
    )
    parser.add_argument(
        "--tokenizer-root", type=Path, default=ROOT / "TSTokenizer" / "checkpoints",
        help="directory containing the five pretrained tokenizer groups",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gpu", default=os.environ.get("GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
        help="CUDA_VISIBLE_DEVICES value; one visible GPU reproduces the single-GPU setup",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def pred_lens_for(setup: str, args: argparse.Namespace) -> tuple[int, ...]:
    available = SETUPS[setup]["pred_lens"]
    chosen = tuple(args.pred_lens) if args.pred_lens else available
    invalid = set(chosen) - set(available)
    if invalid:
        raise ValueError(f"Invalid {setup} horizons: {sorted(invalid)}; available: {available}")
    return chosen


def tokenizer_template(setup: str, pred_len: int, args: argparse.Namespace) -> Path:
    cfg = SETUPS[setup]
    name = (
        f"{cfg['name']}_{cfg['token_len']}_sl{cfg['seq_len']}_pl{pred_len}"
        f"_dm{cfg['d_model']}_dr0.2_emb{cfg['n_embed']}"
        f"_wl{cfg['wave_length']}_bl3_{VQ_TYPE}_epoch{TOKENIZER_EPOCHS}"
    )
    return args.tokenizer_root / name


def tokenizer_artifacts(setup: str, pred_len: int, args: argparse.Namespace) -> list[Path]:
    cfg = SETUPS[setup]
    template = str(tokenizer_template(setup, pred_len, args))
    artifacts = []
    for group, size in cfg["groups"]:
        group_dir = Path(
            template.replace(cfg["name"], group).replace(
                f"emb{cfg['n_embed']}", f"emb{size}"
            )
        )
        artifacts.extend((group_dir / "model.pkl", group_dir / "weight.pkl"))
    return artifacts


def model_id(setup: str, pred_len: int, seed: int) -> str:
    cfg = SETUPS[setup]
    return f"{cfg['name']}_seed{seed}_{cfg['seq_len']}_{pred_len}"


def predictor_setting(setup: str, pred_len: int, seed: int) -> str:
    cfg = SETUPS[setup]
    return (
        f"{model_id(setup, pred_len, seed)}_trm_er0.0"
        f"_wl{cfg['wave_length']}_tl{cfg['token_len']}_ne{cfg['n_embed']}"
        f"_nl2_{VQ_TYPE}_Ve{TOKENIZER_EPOCHS}_Exp0"
    )


def train_command(
    setup: str, pred_len: int, seed: int, args: argparse.Namespace
) -> list[str]:
    cfg = SETUPS[setup]
    return [
        args.python, "-u", str(ROOT / "run_longExp.py"),
        "--random_seed", str(seed),
        "--use_multi_gpu",
        "--cross_dataset", cfg["datasets"],
        "--vqvae_path", str(tokenizer_template(setup, pred_len, args)),
        "--checkpoints", str(args.output_root / "checkpoints"),
        "--text_tokenizer_root", str(args.text_tokenizer_root),
        "--mask_schedule", "cosine",
        "--is_training", "1",
        "--is_unify",
        "--backbone", "trm",
        "--backbone_dim", "128",
        "--root_path", str(args.data_root),
        "--model_id", model_id(setup, pred_len, seed),
        "--model", "TokenTime",
        "--data", cfg["data"],
        "--features", "M",
        "--seq_len", str(cfg["seq_len"]),
        "--pred_len", str(pred_len),
        "--VQ_type", VQ_TYPE,
        "--VQ_epoch", str(TOKENIZER_EPOCHS),
        "--elect_rate", "0",
        "--token_len", str(cfg["token_len"]),
        "--wave_length", str(cfg["wave_length"]),
        "--wave_stride", str(cfg["wave_length"]),
        "--block_num", "3",
        "--n_embed", str(cfg["n_embed"]),
        "--d_model", str(cfg["d_model"]),
        "--d_ff", "256",
        "--dropout", "0.2",
        "--des", "Exp",
        "--train_epochs", "20",
        "--itr", "1",
        "--batch_size", "32",
        "--n_classifier_layer", "2",
        "--infer_batch_size", str(cfg["infer_batch_size"]),
        "--infer_step", "4",
        "--unify_lr", "0.0001",
        "--adaptive_lr", "0.0001",
    ]


def validate_inputs(args: argparse.Namespace) -> None:
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")
    if len(set(args.setups)) != len(args.setups):
        raise ValueError("Setups must be distinct")
    if args.pred_lens and len(set(args.pred_lens)) != len(args.pred_lens):
        raise ValueError("Horizons must be distinct")
    for setup in args.setups:
        pred_lens_for(setup, args)
    if args.dry_run:
        return
    for label, path in (
        ("dataset root", args.data_root),
        ("GPT-2 tokenizer root", args.text_tokenizer_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    missing = [
        path
        for setup in args.setups
        for pred_len in pred_lens_for(setup, args)
        for path in tokenizer_artifacts(setup, pred_len, args)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing pretrained tokenizer artifacts:\n" + "\n".join(map(str, missing))
        )


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.text_tokenizer_root = args.text_tokenizer_root.expanduser().resolve()
    args.tokenizer_root = args.tokenizer_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate_inputs(args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"

    for setup in args.setups:
        for pred_len in pred_lens_for(setup, args):
            for seed in args.seeds:
                cmd = train_command(setup, pred_len, seed, args)
                command_text = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {shlex.join(cmd)}"
                setting = predictor_setting(setup, pred_len, seed)
                checkpoint = args.output_root / "checkpoints" / "unify" / setting / "checkpoint.pth"
                log_path = args.output_root / "logs" / setup / f"seed{seed}" / f"h{pred_len}.log"
                done_path = log_path.with_suffix(".done")
                if args.skip_existing and checkpoint.is_file() and done_path.is_file():
                    print(f"Skipping existing predictor: {checkpoint}")
                    continue
                if args.dry_run:
                    print(command_text)
                    continue

                log_path.parent.mkdir(parents=True, exist_ok=True)
                done_path.unlink(missing_ok=True)
                log_path.with_suffix(".cmd").write_text(command_text + "\n", encoding="utf-8")
                print(f"Running {setup}, horizon {pred_len}, seed {seed}; log: {log_path}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        check=True,
                    )
                done_path.write_text("training and test completed\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
