#!/usr/bin/env python3
"""Run predictor-level domain holdout and prefix few-shot experiments."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "predictor_holdout"
DATASETS = ["ETTh2", "ETTm2", "weather", "traffic", "CzeLan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["zs_train", "zs_test", "fewshot", "all"],
        default="all",
    )
    parser.add_argument("--targets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--pred-lens", nargs="+", type=int, default=[48])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--few-shot-ratios", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    parser.add_argument(
        "--text-tokenizer-root",
        type=Path,
        default=Path(
            os.environ.get("ONECAST_GPT2_ROOT", ROOT / "pretrained" / "GPT2")
        ),
        help="local GPT-2 tokenizer/model directory used by the transformer backbone",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        default=Path(
            os.environ.get("ONECAST_TOKENIZER_ROOT", ROOT / "TSTokenizer" / "checkpoints")
        ),
        help="root containing seed<value> directories for grouped tokenizers",
    )
    parser.add_argument(
        "--tokenizer-seed",
        type=int,
        default=2024,
        help="seed of the frozen grouped tokenizers used by the holdout runs",
    )
    parser.add_argument(
        "--tokenizer-path-template",
        default=None,
        help="grouped-tokenizer path template; may contain {pred_len} and {seed}",
    )
    parser.add_argument("--tokenizer-epochs", type=int, default=25)
    parser.add_argument("--predictor-epochs", type=int, default=20)
    parser.add_argument("--few-shot-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--infer-batch-size", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dataset_arg(datasets: list[str]) -> str:
    return ",".join(datasets)


def ratio_tag(ratio: float) -> str:
    return f"{ratio * 100:g}pct".replace(".", "p")


def tokenizer_path(pred_len: int, args: argparse.Namespace) -> Path:
    if args.tokenizer_path_template:
        return Path(
            args.tokenizer_path_template.format(
                pred_len=pred_len,
                seed=args.tokenizer_seed,
            )
        ).expanduser().resolve()
    return (
        args.tokenizer_root
        / f"seed{args.tokenizer_seed}"
        / (
            f"Cross_large_16_sl96_pl{pred_len}_dm64_dr0.2_emb128_"
            f"wl8_bl3_W_SimVQ_decompose_epoch{args.tokenizer_epochs}"
        )
    )


def grouped_tokenizer_artifacts(template: Path) -> list[Path]:
    base = str(template)
    if "Cross_large" not in base or "emb128" not in base:
        raise ValueError(
            "tokenizer path template must contain Cross_large and emb128 placeholders"
        )
    groups = [
        ("Cross_ETTh2_ETTm2_weather", 256),
        ("Cross_traffic", 128),
        ("Cross_CzeLan", 128),
    ]
    artifacts = []
    for group, n_embed in groups:
        group_path = base.replace("Cross_large", group).replace("emb128", f"emb{n_embed}")
        artifacts.extend([Path(group_path) / "model.pkl", Path(group_path) / "weight.pkl"])
    return artifacts


def source_model_id(target: str, pred_len: int, seed: int) -> str:
    # The Cross_large tag keeps all standard grouped tokenizers loaded and frozen.
    return f"Cross_large_holdout_{target}_seed{seed}_96_{pred_len}"


def fewshot_model_id(target: str, pred_len: int, seed: int, ratio: float) -> str:
    return f"Cross_large_holdout_{target}_seed{seed}_{ratio_tag(ratio)}_96_{pred_len}"


def predictor_setting(model_id: str, tokenizer_epochs: int) -> str:
    return (
        f"{model_id}_trm_er0.0_wl8_tl16_ne128_nl2_"
        f"W_SimVQ_decompose_Ve{tokenizer_epochs}_Exp0"
    )


def source_checkpoint(target: str, pred_len: int, seed: int, args: argparse.Namespace) -> Path:
    setting = predictor_setting(
        source_model_id(target, pred_len, seed), args.tokenizer_epochs
    )
    return args.output_root / "checkpoints" / "unify" / setting / "checkpoint.pth"


def fewshot_checkpoint(
    target: str,
    pred_len: int,
    seed: int,
    ratio: float,
    args: argparse.Namespace,
) -> Path:
    setting = predictor_setting(
        fewshot_model_id(target, pred_len, seed, ratio), args.tokenizer_epochs
    )
    return (
        args.output_root
        / "checkpoints"
        / "adaptive"
        / target
        / setting
        / "checkpoint.pth"
    )


def predictor_base_cmd(
    pred_len: int,
    seed: int,
    train_epochs: int,
    model_id: str,
    datasets: list[str],
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        args.python,
        "-u",
        str(ROOT / "run_longExp.py"),
        "--random_seed",
        str(seed),
        "--cross_dataset",
        dataset_arg(datasets),
        "--vqvae_path",
        str(tokenizer_path(pred_len, args)),
        "--checkpoints",
        str(args.output_root / "checkpoints"),
        "--mask_schedule",
        "cosine",
        "--backbone",
        "trm",
        "--backbone_dim",
        "128",
        "--root_path",
        str(args.data_root),
        "--text_tokenizer_root",
        str(args.text_tokenizer_root),
        "--model_id",
        model_id,
        "--model",
        "TokenTime",
        "--data",
        "predictor_holdout",
        "--features",
        "M",
        "--seq_len",
        "96",
        "--pred_len",
        str(pred_len),
        "--VQ_type",
        "W_SimVQ_decompose",
        "--VQ_epoch",
        str(args.tokenizer_epochs),
        "--elect_rate",
        "0",
        "--token_len",
        "16",
        "--wave_length",
        "8",
        "--wave_stride",
        "8",
        "--block_num",
        "3",
        "--n_embed",
        "128",
        "--d_model",
        "64",
        "--d_ff",
        "256",
        "--dropout",
        "0.2",
        "--des",
        "Exp",
        "--train_epochs",
        str(train_epochs),
        "--itr",
        "1",
        "--batch_size",
        str(args.batch_size),
        "--n_classifier_layer",
        "2",
        "--infer_batch_size",
        str(args.infer_batch_size),
        "--infer_step",
        "4",
        "--unify_lr",
        "0.0001",
        "--adaptive_lr",
        "0.0001",
        "--num_workers",
        "0",
        "--no_train_shuffle",
    ]
    if args.multi_gpu:
        cmd.insert(3, "--use_multi_gpu")
    return cmd


def zero_shot_train_cmd(
    target: str, pred_len: int, seed: int, args: argparse.Namespace
) -> list[str]:
    sources = [dataset for dataset in DATASETS if dataset != target]
    cmd = predictor_base_cmd(
        pred_len,
        seed,
        args.predictor_epochs,
        source_model_id(target, pred_len, seed),
        sources,
        args,
    )
    cmd += ["--is_training", "1", "--is_unify"]
    return cmd


def zero_shot_test_cmd(
    target: str, pred_len: int, seed: int, args: argparse.Namespace
) -> list[str]:
    cmd = predictor_base_cmd(
        pred_len,
        seed,
        args.predictor_epochs,
        source_model_id(target, pred_len, seed),
        [target],
        args,
    )
    cmd += ["--is_training", "0", "--is_unify"]
    return cmd


def fewshot_cmd(
    target: str,
    pred_len: int,
    seed: int,
    ratio: float,
    args: argparse.Namespace,
) -> list[str]:
    cmd = predictor_base_cmd(
        pred_len,
        seed,
        args.few_shot_epochs,
        fewshot_model_id(target, pred_len, seed, ratio),
        [target],
        args,
    )
    source_setting = predictor_setting(
        source_model_id(target, pred_len, seed), args.tokenizer_epochs
    )
    cmd += [
        "--is_training",
        "1",
        "--load_unify",
        "1",
        "--load_unify_setting",
        source_setting,
        "--adaptive_dataset",
        target,
        "--few_shot_ratio",
        str(ratio),
        "--few_shot_strategy",
        "prefix",
    ]
    return cmd


def run_command(
    cmd: list[str],
    log_path: Path,
    args: argparse.Namespace,
    skip_path: Path | None = None,
) -> None:
    if args.dry_run:
        print("DRY RUN:", shlex.join(cmd))
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.with_suffix(log_path.suffix + ".cmd").write_text(
        shlex.join(cmd) + "\n", encoding="utf-8"
    )
    done_path = log_path.with_suffix(log_path.suffix + ".done")

    if (
        args.skip_existing
        and skip_path is not None
        and skip_path.exists()
        and done_path.exists()
    ):
        print(f"Skipping existing artifact: {skip_path}")
        return
    if args.skip_existing and skip_path is None and done_path.exists():
        print(f"Skipping completed log: {log_path}")
        return

    done_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp")

    print(f"\n=== Running {log_path.relative_to(args.output_root)} ===")
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="")
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    done_path.write_text("training or evaluation completed\n", encoding="utf-8")


def validate_inputs(args: argparse.Namespace) -> None:
    for ratio in args.few_shot_ratios:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"few-shot ratios must be in (0, 1], got {ratio}")
    artifacts_by_horizon = {
        pred_len: grouped_tokenizer_artifacts(tokenizer_path(pred_len, args))
        for pred_len in args.pred_lens
    }
    if args.dry_run:
        return
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")
    if not args.text_tokenizer_root.exists():
        raise FileNotFoundError(
            f"GPT-2 tokenizer/model directory not found: {args.text_tokenizer_root}"
        )
    for pred_len, artifacts in artifacts_by_horizon.items():
        missing = [
            path for path in artifacts if not path.exists()
        ]
        if missing:
            formatted = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing grouped-tokenizer artifacts:\n{formatted}")


def run_zs_train(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        for pred_len in args.pred_lens:
            for target in args.targets:
                log_path = (
                    args.output_root
                    / "logs"
                    / f"seed{seed}"
                    / f"zs_train_leave_{target}_h{pred_len}.log"
                )
                run_command(
                    zero_shot_train_cmd(target, pred_len, seed, args),
                    log_path,
                    args,
                    skip_path=source_checkpoint(target, pred_len, seed, args),
                )


def run_zs_test(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        for pred_len in args.pred_lens:
            for target in args.targets:
                checkpoint = source_checkpoint(target, pred_len, seed, args)
                if not args.dry_run and not checkpoint.exists():
                    raise FileNotFoundError(f"Zero-shot predictor checkpoint not found: {checkpoint}")
                log_path = (
                    args.output_root
                    / "logs"
                    / f"seed{seed}"
                    / f"zs_test_leave_{target}_h{pred_len}.log"
                )
                run_command(
                    zero_shot_test_cmd(target, pred_len, seed, args),
                    log_path,
                    args,
                )


def run_fewshot(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        for pred_len in args.pred_lens:
            for target in args.targets:
                checkpoint = source_checkpoint(target, pred_len, seed, args)
                if not args.dry_run and not checkpoint.exists():
                    raise FileNotFoundError(f"Source predictor checkpoint not found: {checkpoint}")
                for ratio in args.few_shot_ratios:
                    log_path = (
                        args.output_root
                        / "logs"
                        / f"seed{seed}"
                        / f"fs_{ratio_tag(ratio)}_leave_{target}_h{pred_len}.log"
                    )
                    run_command(
                        fewshot_cmd(target, pred_len, seed, ratio, args),
                        log_path,
                        args,
                        skip_path=fewshot_checkpoint(target, pred_len, seed, ratio, args),
                    )


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.text_tokenizer_root = args.text_tokenizer_root.expanduser().resolve()
    args.tokenizer_root = args.tokenizer_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate_inputs(args)

    if args.phase in {"zs_train", "all"}:
        run_zs_train(args)
    if args.phase in {"zs_test", "all"}:
        run_zs_test(args)
    if args.phase in {"fewshot", "all"}:
        run_fewshot(args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(
            f"Command failed with exit code {exc.returncode}: {shlex.join(exc.cmd)}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)
