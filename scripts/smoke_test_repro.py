#!/usr/bin/env python3
"""Dependency-light smoke tests for the OneCast reproducibility runners.

The default test validates script syntax, generated command matrices, CLI
option-name compatibility, seed/checkpoint pairing, and orchestration by using
a temporary fake training process. It does not construct or train a neural
network, deserialize a real checkpoint, or validate numerical results.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TOKENIZER_RUNNER = ROOT / "scripts" / "unify" / "run_tokenizer_seeds.py"
PREDICTOR_RUNNER = ROOT / "scripts" / "unify" / "run_predictor_seeds.py"
HOLDOUT_RUNNER = ROOT / "scripts" / "holdout" / "run_predictor_holdout.py"
SUMMARIZER = ROOT / "scripts" / "holdout" / "summarize_predictor_holdout.py"
RUN_LONG_EXP = ROOT / "run_longExp.py"
TOKENIZER_ARGS = ROOT / "TSTokenizer" / "args.py"


class SmokeFailure(RuntimeError):
    """Raised when a smoke-test invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        cwd=cwd,
        env=merged_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def command_tokens(line: str) -> list[str]:
    if line.startswith("DRY RUN: "):
        line = line[len("DRY RUN: ") :]
    tokens = shlex.split(line)
    if tokens and tokens[0].startswith("CUDA_VISIBLE_DEVICES="):
        tokens = tokens[1:]
    return tokens


def option_value(tokens: list[str], option: str) -> str:
    try:
        index = tokens.index(option)
    except ValueError as exc:
        raise SmokeFailure(f"Missing option {option}: {shlex.join(tokens)}") from exc
    require(index + 1 < len(tokens), f"Missing value for {option}")
    return tokens[index + 1]


def declared_options(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    options.add(argument.value)
    return options


def generated_options(lines: list[str]) -> set[str]:
    return {
        token
        for line in lines
        for token in command_tokens(line)
        if token.startswith("--")
    }


def check_syntax() -> None:
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        compile(path.read_bytes(), str(path), "exec")

    shell_files = sorted(ROOT.rglob("*.sh"))
    for path in shell_files:
        run(["bash", "-n", str(path)])
    print(f"[pass] syntax: {len(python_files)} Python and {len(shell_files)} shell files")


def check_help() -> None:
    for path in (TOKENIZER_RUNNER, PREDICTOR_RUNNER, HOLDOUT_RUNNER, SUMMARIZER):
        run([PYTHON, str(path), "--help"])
    run([PYTHON, "args.py", "--help"], cwd=ROOT / "TSTokenizer")
    print("[pass] lightweight CLI help")


def check_dry_runs() -> None:
    tokenizer_lines = nonempty_lines(
        run([PYTHON, str(TOKENIZER_RUNNER), "--dry-run"]).stdout
    )
    require(len(tokenizer_lines) == 15, "Tokenizer dry run must emit 15 group commands")
    tokenizer_specs = {
        "Cross_ETTh2_ETTm2_weather.sh": "24 48 96 192",
        "Cross_Traffic.sh": "24 48 96 192",
        "Cross_CzeLan.sh": "24 48 96 192",
        "Cross_FRED_Covid_NYSE.sh": "24 36 48 60",
        "Cross_Wike2000.sh": "24 36 48 60",
    }
    tokenizer_observed: set[tuple[str, str, str]] = set()
    tokenizer_job_count = 0
    for line in tokenizer_lines:
        tokens = shlex.split(line)
        bash_index = tokens.index("bash")
        env_values = dict(token.split("=", 1) for token in tokens[:bash_index])
        script_name = Path(tokens[bash_index + 1]).name
        seed = env_values["ONECAST_RANDOM_SEED"]
        horizons = env_values["ONECAST_PRED_LENS"]
        tokenizer_observed.add((seed, script_name, horizons))
        tokenizer_job_count += len(horizons.split())
        require(
            env_values["ONECAST_SKIP_EXISTING"] == "0",
            "Default tokenizer dry run unexpectedly enables resume mode",
        )
    tokenizer_expected = {
        (seed, script_name, horizons)
        for seed in ("2023", "2024", "2025")
        for script_name, horizons in tokenizer_specs.items()
    }
    require(
        tokenizer_observed == tokenizer_expected,
        "Tokenizer dry run does not cover the exact seed/group/horizon matrix",
    )
    require(tokenizer_job_count == 60, "Tokenizer shell loops must expand to 60 jobs")

    predictor_lines = nonempty_lines(
        run([PYTHON, str(PREDICTOR_RUNNER), "--dry-run"]).stdout
    )
    require(len(predictor_lines) == 24, "Predictor dry run must emit 24 commands")
    predictor_observed: set[tuple[str, int, int]] = set()
    setup_by_data = {"Cross_large": "large", "TFB": "small"}
    for line in predictor_lines:
        tokens = command_tokens(line)
        seed = option_value(tokens, "--random_seed")
        data = option_value(tokens, "--data")
        require(data in setup_by_data, f"Unexpected predictor data setting: {data}")
        setup = setup_by_data[data]
        pred_len = int(option_value(tokens, "--pred_len"))
        predictor_observed.add((setup, pred_len, int(seed)))
        model_id = option_value(tokens, "--model_id")
        tokenizer_path = option_value(tokens, "--vqvae_path")
        require(f"_seed{seed}_" in model_id, f"Seed missing from model ID: {model_id}")
        require(
            f"{os.sep}seed{seed}{os.sep}" in tokenizer_path,
            f"Predictor seed {seed} is not paired with its tokenizer: {tokenizer_path}",
        )
        require(option_value(tokens, "--itr") == "1", "Each seed command must use --itr 1")
    predictor_expected = {
        (setup, pred_len, seed)
        for setup, pred_lens in (
            ("large", (24, 48, 96, 192)),
            ("small", (24, 36, 48, 60)),
        )
        for pred_len in pred_lens
        for seed in (2023, 2024, 2025)
    }
    require(
        predictor_observed == predictor_expected,
        "Predictor dry run does not cover the exact setup/horizon/seed matrix",
    )

    holdout_lines = nonempty_lines(
        run([PYTHON, str(HOLDOUT_RUNNER), "--phase", "all", "--dry-run"]).stdout
    )
    require(len(holdout_lines) == 20, "Holdout dry run must emit 20 commands")
    holdout_datasets = ["ETTh2", "ETTm2", "weather", "traffic", "CzeLan"]
    seen: dict[str, dict[str, object]] = {}
    for line in holdout_lines:
        tokens = command_tokens(line)
        model_id = option_value(tokens, "--model_id")
        match = re.search(r"Cross_large_holdout_(.+?)_seed\d+", model_id)
        require(match is not None, f"Cannot identify holdout target from {model_id}")
        target = match.group(1)
        record = seen.setdefault(target, {"train": 0, "test": 0, "ratios": []})
        datasets = option_value(tokens, "--cross_dataset").split(",")
        seed = option_value(tokens, "--random_seed")
        pred_len = option_value(tokens, "--pred_len")
        source_setting = (
            f"Cross_large_holdout_{target}_seed{seed}_96_{pred_len}"
            "_trm_er0.0_wl8_tl16_ne128_nl2_"
            "W_SimVQ_decompose_Ve25_Exp0"
        )
        require(seed == "2024", "Default holdout seed must be 2024")
        require(option_value(tokens, "--itr") == "1", "Holdout commands must use --itr 1")
        require(
            f"{os.sep}seed2024{os.sep}" in option_value(tokens, "--vqvae_path"),
            "Holdout predictor is not paired with the seed-2024 tokenizer",
        )
        if "--few_shot_ratio" in tokens:
            require(datasets == [target], f"Few-shot data must contain only {target}")
            require("--is_unify" not in tokens, "Few-shot adaptation must use adaptive mode")
            require(option_value(tokens, "--load_unify") == "1", "Few-shot must load LODO weights")
            require(
                option_value(tokens, "--load_unify_setting") == source_setting,
                f"Few-shot command loads the wrong LODO checkpoint for {target}",
            )
            require(
                option_value(tokens, "--adaptive_dataset") == target,
                f"Few-shot adaptive dataset must be {target}",
            )
            require(
                option_value(tokens, "--few_shot_strategy") == "prefix",
                "Few-shot adaptation must select a chronological prefix",
            )
            record["ratios"].append(float(option_value(tokens, "--few_shot_ratio")))
        elif option_value(tokens, "--is_training") == "0":
            require(datasets == [target], f"Zero-shot test data must contain only {target}")
            require("--is_unify" in tokens, "Zero-shot test must use the unified predictor")
            require("--load_unify" not in tokens, "Zero-shot test must not enter adaptive mode")
            record["test"] += 1
        else:
            expected_sources = [name for name in holdout_datasets if name != target]
            require(
                datasets == expected_sources,
                f"LODO sources for {target} must be exactly {expected_sources}",
            )
            require("--is_unify" in tokens, "LODO training must train a unified predictor")
            require("--load_unify" not in tokens, "LODO training must start without adaptive loading")
            record["train"] += 1
    require(len(seen) == 5, "Holdout commands must cover five targets")
    for target, record in seen.items():
        require(record["train"] == 1, f"Expected one LODO train command for {target}")
        require(record["test"] == 1, f"Expected one zero-shot test command for {target}")
        require(
            sorted(record["ratios"]) == [0.05, 0.10],
            f"Expected 5% and 10% adaptation for {target}",
        )

    run_long_options = declared_options(RUN_LONG_EXP)
    unknown_predictor = generated_options(predictor_lines) - run_long_options
    unknown_holdout = generated_options(holdout_lines) - run_long_options
    require(not unknown_predictor, f"Unknown predictor CLI options: {sorted(unknown_predictor)}")
    require(not unknown_holdout, f"Unknown holdout CLI options: {sorted(unknown_holdout)}")

    tokenizer_options = declared_options(TOKENIZER_ARGS)
    tokenizer_shells = [
        ROOT / "TSTokenizer" / "scripts" / name
        for name in (
            "Cross_ETTh2_ETTm2_weather.sh",
            "Cross_Traffic.sh",
            "Cross_CzeLan.sh",
            "Cross_FRED_Covid_NYSE.sh",
            "Cross_Wike2000.sh",
        )
    ]
    shell_options = {
        option
        for path in tokenizer_shells
        for option in re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", path.read_text())
    }
    require(
        not shell_options - tokenizer_options,
        f"Unknown tokenizer CLI options: {sorted(shell_options - tokenizer_options)}",
    )
    print(
        "[pass] dry-run matrices, seed pairing, holdout protocol, "
        "and CLI option-name compatibility"
    )


FAKE_TRAINER = r'''#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

argv = sys.argv[1:]
if argv and argv[0] == "-u":
    argv = argv[1:]
script = argv[0]
args = argv[1:]

def value(option, default=None):
    if option not in args:
        return default
    index = args.index(option)
    return args[index + 1]

call_log = Path(os.environ["ONECAST_SMOKE_CALL_LOG"])
with call_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"script": script, "args": args}) + "\n")

if script.endswith("main.py"):
    save_path = Path(value("--save_path"))
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "model.pkl").write_text("smoke\n", encoding="utf-8")
    (save_path / "weight.pkl").write_text("smoke\n", encoding="utf-8")
    print("fake tokenizer completed")
    raise SystemExit(0)

if not script.endswith("run_longExp.py"):
    raise SystemExit(f"Unexpected training entry point: {script}")

model_id = value("--model_id")
setting = (
    f"{model_id}_{value('--backbone')}_er{float(value('--elect_rate'))}"
    f"_wl{value('--wave_length')}_tl{value('--token_len')}"
    f"_ne{value('--n_embed')}_nl{value('--n_classifier_layer')}"
    f"_{value('--VQ_type')}_Ve{value('--VQ_epoch')}_Exp0"
)
if value("--is_training") == "1":
    checkpoint_root = Path(value("--checkpoints"))
    if "--is_unify" in args:
        checkpoint = checkpoint_root / "unify" / setting / "checkpoint.pth"
    else:
        checkpoint = (
            checkpoint_root
            / "adaptive"
            / value("--adaptive_dataset")
            / setting
            / "checkpoint.pth"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("smoke\n", encoding="utf-8")

target = value("--adaptive_dataset")
if target is None:
    datasets = value("--cross_dataset", "").split(",")
    if len(datasets) == 1:
        target = datasets[0]
if target is None:
    match = re.search(r"Cross_large_holdout_(.+?)_seed\d+", model_id)
    target = match.group(1) if match else "smoke"
print(f"Testing {target}...")
print("mse:0.1, mae:0.2")
'''


def read_calls(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check_orchestration() -> None:
    with tempfile.TemporaryDirectory(prefix="onecast-smoke-") as directory:
        workspace = Path(directory)
        data_root = workspace / "data root"
        text_root = workspace / "GPT2 root"
        tokenizer_root = workspace / "tokenizer checkpoints"
        tokenizer_logs = workspace / "tokenizer logs"
        predictor_output = workspace / "predictor output"
        holdout_output = workspace / "holdout output"
        call_log = workspace / "calls.jsonl"
        for path in (data_root, text_root, tokenizer_root):
            path.mkdir(parents=True)

        fake_python = workspace / "fake python"
        fake_python.write_text(textwrap.dedent(FAKE_TRAINER), encoding="utf-8")
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
        env = {"ONECAST_SMOKE_CALL_LOG": str(call_log)}

        tokenizer_command = [
            PYTHON,
            str(TOKENIZER_RUNNER),
            "--setups",
            "large",
            "small",
            "--seeds",
            "2023",
            "--pred-lens",
            "24",
            "--python",
            str(fake_python),
            "--data-root",
            str(data_root),
            "--tokenizer-root",
            str(tokenizer_root),
            "--log-root",
            str(tokenizer_logs),
        ]
        run(tokenizer_command, env=env)
        tokenizer_calls = read_calls(call_log)
        require(len(tokenizer_calls) == 5, "Expected five tokenizer subprocesses")
        expected_groups = {
            "Cross_ETTh2_ETTm2_weather",
            "Cross_traffic",
            "Cross_CzeLan",
            "Cross_FRED_Covid_NYSE",
            "Cross_Wike2000",
        }
        observed_groups: set[str] = set()
        for call in tokenizer_calls:
            require(
                Path(str(call["script"])).name == "main.py",
                "Tokenizer shell called the wrong entry point",
            )
            tokens = [str(token) for token in call["args"]]
            group = option_value(tokens, "--model_id")
            observed_groups.add(group)
            require(
                option_value(tokens, "--random_seed") == "2023",
                "Tokenizer shell changed the requested seed",
            )
            require(
                option_value(tokens, "--pred_len") == "24",
                "Tokenizer shell changed the requested horizon",
            )
            save_path = Path(option_value(tokens, "--save_path"))
            require(
                save_path.parent == tokenizer_root.resolve() / "seed2023",
                f"Tokenizer shell wrote outside the matching seed directory: {save_path}",
            )
            require(
                f"{group}_" in save_path.name,
                f"Tokenizer path does not match group {group}",
            )
        require(
            observed_groups == expected_groups,
            "Tokenizer shells did not execute all five groups",
        )
        require(len(list(tokenizer_root.rglob("model.pkl"))) == 5, "Missing tokenizer models")
        require(len(list(tokenizer_root.rglob("weight.pkl"))) == 5, "Missing tokenizer weights")
        tokenizer_calls_before_resume = len(read_calls(call_log))
        run(tokenizer_command + ["--skip-existing"], env=env)
        require(
            len(read_calls(call_log)) == tokenizer_calls_before_resume,
            "--skip-existing unexpectedly reran completed tokenizer commands",
        )

        predictor_command = [
            PYTHON,
            str(PREDICTOR_RUNNER),
            "--setups",
            "large",
            "small",
            "--seeds",
            "2023",
            "--pred-lens",
            "24",
            "--python",
            str(fake_python),
            "--data-root",
            str(data_root),
            "--text-tokenizer-root",
            str(text_root),
            "--tokenizer-root",
            str(tokenizer_root),
            "--output-root",
            str(predictor_output),
        ]
        run(predictor_command, env=env)
        require(len(read_calls(call_log)) == 7, "Expected two predictor subprocesses")
        predictor_calls = read_calls(call_log)[5:]
        require(
            {
                option_value(
                    [str(token) for token in call["args"]], "--model_id"
                )
                for call in predictor_calls
            }
            == {"Cross_large_seed2023_96_24", "Cross_small_seed2023_36_24"},
            "Predictor runner did not execute both requested setups",
        )
        for call in predictor_calls:
            tokens = [str(token) for token in call["args"]]
            require(
                option_value(tokens, "--random_seed") == "2023",
                "Predictor changed the requested seed",
            )
            require(
                f"{os.sep}seed2023{os.sep}" in option_value(tokens, "--vqvae_path"),
                "Predictor did not receive the tokenizer from the matching seed",
            )
        require(len(list(predictor_output.rglob("*.cmd"))) == 2, "Missing predictor commands")
        require(len(list(predictor_output.rglob("*.done"))) == 2, "Missing predictor done markers")
        require(
            len(list((predictor_output / "checkpoints" / "unify").rglob("checkpoint.pth")))
            == 2,
            "Missing predictor checkpoints",
        )
        predictor_calls_before_resume = len(read_calls(call_log))
        run(predictor_command + ["--skip-existing"], env=env)
        require(
            len(read_calls(call_log)) == predictor_calls_before_resume,
            "--skip-existing unexpectedly reran completed predictor commands",
        )

        holdout_common = [
            "--targets",
            "ETTh2",
            "--pred-lens",
            "24",
            "--seeds",
            "2023",
            "--few-shot-ratios",
            "0.05",
            "0.10",
            "--python",
            str(fake_python),
            "--data-root",
            str(data_root),
            "--text-tokenizer-root",
            str(text_root),
            "--tokenizer-root",
            str(tokenizer_root),
            "--tokenizer-seed",
            "2023",
            "--output-root",
            str(holdout_output),
        ]
        wrapper_env = dict(env)
        wrapper_env["PYTHON_BIN"] = PYTHON
        lodo_command = [
            "bash",
            str(ROOT / "scripts" / "holdout" / "run_lodo.sh"),
            *holdout_common,
        ]
        fewshot_command = [
            "bash",
            str(ROOT / "scripts" / "holdout" / "run_fewshot.sh"),
            *holdout_common,
        ]
        run(lodo_command, env=wrapper_env)
        run(fewshot_command, env=wrapper_env)
        require(len(read_calls(call_log)) == 11, "Expected four holdout subprocesses")
        holdout_calls = read_calls(call_log)[7:]
        source_setting = (
            "Cross_large_holdout_ETTh2_seed2023_96_24_"
            "trm_er0.0_wl8_tl16_ne128_nl2_W_SimVQ_decompose_Ve25_Exp0"
        )
        require(
            option_value(
                [str(token) for token in holdout_calls[0]["args"]], "--is_training"
            )
            == "1",
            "LODO wrapper did not train first",
        )
        require(
            option_value(
                [str(token) for token in holdout_calls[1]["args"]], "--is_training"
            )
            == "0",
            "LODO wrapper did not test second",
        )
        for call, ratio in zip(holdout_calls[2:], ("0.05", "0.1")):
            tokens = [str(token) for token in call["args"]]
            require(
                option_value(tokens, "--few_shot_ratio") == ratio,
                "Few-shot wrapper emitted the wrong ratio",
            )
            require(
                option_value(tokens, "--load_unify") == "1",
                "Few-shot wrapper did not load the LODO predictor",
            )
            require(
                option_value(tokens, "--load_unify_setting") == source_setting,
                "Few-shot wrapper loaded the wrong LODO predictor",
            )
            require(
                option_value(tokens, "--adaptive_dataset") == "ETTh2",
                "Few-shot wrapper adapted on the wrong target",
            )
            require(
                option_value(tokens, "--few_shot_strategy") == "prefix",
                "Few-shot wrapper did not use prefix sampling",
            )
        require(
            len(list((holdout_output / "checkpoints").rglob("checkpoint.pth"))) == 3,
            "Missing LODO or few-shot checkpoints",
        )

        calls_before_resume = len(read_calls(call_log))
        run(lodo_command + ["--skip-existing"], env=wrapper_env)
        run(fewshot_command + ["--skip-existing"], env=wrapper_env)
        require(
            len(read_calls(call_log)) == calls_before_resume,
            "--skip-existing unexpectedly reran completed holdout commands",
        )

        summary_path = holdout_output / "summary.csv"
        run(
            [
                PYTHON,
                str(SUMMARIZER),
                "--output-root",
                str(holdout_output),
                "--targets",
                "ETTh2",
                "--pred-lens",
                "24",
                "--seeds",
                "2023",
                "--few-shot-ratios",
                "0.05",
                "0.10",
                "--output",
                str(summary_path),
            ]
        )
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 3, "Summary must contain zero-shot, 5%, and 10% rows")
        require(all(row["status"] == "complete" for row in rows), "Summary is incomplete")
        require(
            {(row["mode"], float(row["few_shot_ratio"])) for row in rows}
            == {("zero_shot", 0.0), ("few_shot", 0.05), ("few_shot", 0.10)},
            "Summary contains the wrong evaluation modes or ratios",
        )
        require(
            all(float(row["mse"]) == 0.1 and float(row["mae"]) == 0.2 for row in rows),
            "Summary did not parse the fake metrics correctly",
        )
        require(
            all(
                row["seed"] == "2023"
                and row["target"] == "ETTh2"
                and row["pred_len"] == "24"
                and Path(row["log_path"]).is_file()
                for row in rows
            ),
            "Summary row identity or log path is incorrect",
        )
    print("[pass] fake-process wrappers, artifacts, resume paths, and summary plumbing")


def check_training_imports() -> None:
    with tempfile.TemporaryDirectory(prefix="onecast-import-smoke-") as directory:
        cache_root = Path(directory)
        env = {
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "HF_HOME": str(cache_root / "huggingface"),
        }
        run([PYTHON, str(RUN_LONG_EXP), "--help"], env=env)
        run([PYTHON, "main.py", "--help"], cwd=ROOT / "TSTokenizer", env=env)
    print("[pass] training-entry imports in the active Python environment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-training-imports",
        action="store_true",
        help="also import the real training entry points; requires the ML environment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_syntax()
    check_help()
    check_dry_runs()
    check_orchestration()
    if args.check_training_imports:
        check_training_imports()
    print("All dependency-light reproducibility smoke tests passed.")
    if not args.check_training_imports:
        print(
            "Neural-network construction/training was not run; use "
            "--check-training-imports in the prepared ML environment, then run a "
            "small real-data job for end-to-end validation."
        )


if __name__ == "__main__":
    try:
        main()
    except (SmokeFailure, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            details = exc.stderr.strip() or exc.stdout.strip()
            print(f"Smoke command failed: {shlex.join(exc.cmd)}", file=sys.stderr)
            if details:
                print(details, file=sys.stderr)
        else:
            print(f"Smoke assertion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
