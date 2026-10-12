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
# Optional when using the Qwen backbone:
export ONECAST_QWEN_ROOT=/path/to/Qwen2.5-0.5B-Instruct
```

`PYTHON_BIN` defaults to `python3`, `GPU_IDS` defaults to `0`, and
`ONECAST_DATA_ROOT` defaults to `./dataset` under the repository root.
`ONECAST_GPT2_ROOT` defaults to `./pretrained/GPT2`.

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

The tokenizer scripts use seed 2024. Train each tokenizer group once for each
forecast horizon; the three main-table predictor runs below reuse those same
group checkpoints. Each tokenizer checkpoint directory must contain `model.pkl`
and `weight.pkl`.

### Reproduce Main-Table Predictor Runs

Run the large and small cross-domain predictors once under each of seeds
2023, 2024, and 2025. The runner keeps the pretrained tokenizers fixed and
gives each predictor run a distinct model ID, log, and checkpoint path:

```bash
python3 scripts/unify/run_predictor_seeds.py --dry-run
python3 scripts/unify/run_predictor_seeds.py \
  --setups large small \
  --seeds 2023 2024 2025 \
  --gpu 0
```

By default, it reads grouped tokenizer checkpoints from
`TSTokenizer/checkpoints/` and writes logs, exact `.cmd` snapshots, and
predictor checkpoints under `outputs/main_predictor_seeds/`. Use
`--tokenizer-root`, `--data-root`, `--text-tokenizer-root`, and `--output-root`
when those files are elsewhere. `--pred-lens 48` runs a single shared horizon;
`--skip-existing` resumes after runs that completed training and testing.

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
