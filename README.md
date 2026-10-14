# OneCast
OneCast: Structured Decomposition and Modular Generation for Cross-Domain Time Series Forecasting

### Prepare environments
```bash
pip install -r requirements.txt
```

Activate the desired environment before running the scripts. Repository scripts
do not assume a machine-specific Conda installation. Their paths and devices can
be configured without editing the files:

```bash
export PYTHON_BIN=/path/to/python
export GPU_IDS=0
export ONECAST_DATA_ROOT=/path/to/forecasting-datasets
export ONECAST_GPT2_ROOT=/path/to/GPT2
export ONECAST_TOKENIZER_ROOT=/path/to/tokenizer-checkpoints
# Optional when using the Qwen backbone:
export ONECAST_QWEN_ROOT=/path/to/Qwen2.5-0.5B-Instruct
```

`PYTHON_BIN` defaults to `python3`, `GPU_IDS` defaults to `0`, and
`ONECAST_DATA_ROOT` defaults to `./dataset` under the repository root.
`ONECAST_GPT2_ROOT` defaults to `./pretrained/GPT2`, and
`ONECAST_TOKENIZER_ROOT` defaults to `./TSTokenizer/checkpoints`.

### Prepare datasets

```bash
### Download CzeLan, FRED-MD, NYSE, Covid-19, Wike2000 from Time Series Benchmark
# TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

### Download ETTh2, ETTm2, electricity, weather, traffic from Autoformer
# https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
```

### Train Time Series Tokenizer

```bash
# large datasets' tokenizer
bash TSTokenizer/scripts/Cross_CzeLan.sh
bash TSTokenizer/scripts/Cross_Traffic.sh
bash TSTokenizer/scripts/Cross_ETTh2_ETTm2_weather.sh

# small datasets' tokenizer
bash TSTokenizer/scripts/Cross_Wike2000.sh
bash TSTokenizer/scripts/Cross_FRED_Covid_NYSE.sh
```

The individual tokenizer scripts default to seed 2024 and save under
`TSTokenizer/checkpoints/seed2024/`. Set `ONECAST_RANDOM_SEED` to run another
seed. Each tokenizer checkpoint directory contains `model.pkl` and
`weight.pkl`.

### Reproduce Main-Table Runs

For each reported seed, first train all grouped tokenizers with that seed:

```bash
python3 scripts/unify/run_tokenizer_seeds.py --dry-run
python3 scripts/unify/run_tokenizer_seeds.py \
  --setups large small \
  --seeds 2023 2024 2025 \
  --gpu 0
```

Then train each predictor with the tokenizer produced by the same seed:

```bash
python3 scripts/unify/run_predictor_seeds.py --dry-run
python3 scripts/unify/run_predictor_seeds.py \
  --setups large small \
  --seeds 2023 2024 2025 \
  --gpu 0
```

The predictor commands intentionally use `--itr 1`: each command is one
independent run, and the runner enumerates seeds 2023, 2024, and 2025.

Before launching the full experiments, run the dependency-light smoke test:

```bash
python3 scripts/smoke_test_repro.py
```

It checks Python and shell syntax, the complete dry-run command matrices,
generated CLI option names, and seed/checkpoint pairing. A temporary fake
training process then exercises the tokenizer and predictor runners, the LODO
and few-shot wrapper scripts, artifact naming, resume paths, and summary CSV
plumbing. The default test does not import or construct the real neural
networks, load datasets or checkpoints, run forward/backward passes, verify
checkpoint deserialization, or validate numerical results. In a prepared
training environment, add `--check-training-imports` to verify that the real
tokenizer and predictor entry points can import successfully, then run a small
real-data job for end-to-end validation.

Both runners use `TSTokenizer/checkpoints/seed<seed>/` by default. Tokenizer
logs are stored under `TSTokenizer/logs/Tokenizer/seed<seed>/`; predictor logs,
exact `.cmd` snapshots, and checkpoints are stored under
`outputs/main_predictor_seeds/`. Both runners accept `--tokenizer-root` and
`--data-root`; the tokenizer runner also accepts `--log-root`, while the
predictor runner accepts `--text-tokenizer-root` and `--output-root`.
`--pred-lens 48` runs one horizon, and `--skip-existing` resumes completed
runs.

### Single-Seed Unify Model

The original shell commands below run seed 2024. Use the three-seed runner above
for the reported main-table protocol.

```bash
bash scripts/unify/Cross_large_unify.sh
# or 
bash scripts/unify/Cross_small_unify.sh
```

### Train Adaptive Model
```bash
bash scripts/adaptive/Cross_large_adaptive.sh
# or 
bash scripts/adaptive/Cross_small_adaptive.sh
```

### Predictor-Level Domain Holdout and Few-Shot Adaptation

The reproducible zero-shot and 5%/10% prefix few-shot protocol is documented in
[`scripts/holdout/README.md`](scripts/holdout/README.md). This protocol keeps the
standard grouped tokenizers frozen and excludes the target domain only from
token-predictor training. Use `scripts/holdout/run_lodo.sh` for predictor-level
LODO and `scripts/holdout/run_fewshot.sh` for 5%/10% target adaptation. These
produce the zero-shot and few-shot columns in Appendix F.2; its full-data
reference column reuses the main-table results from the three-seed runner above.
The general adaptive scripts above are separate.
