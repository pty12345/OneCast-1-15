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

### Train Unify Model
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
LODO and `scripts/holdout/run_fewshot.sh` for 5%/10% target adaptation.
