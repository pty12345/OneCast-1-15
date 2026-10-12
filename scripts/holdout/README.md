# Predictor-Level Domain Holdout

This runner evaluates transfer at the token-predictor level. It is not an
end-to-end unseen-domain protocol.

## Protocol

- The standard `Cross_large` grouped tokenizers are trained on their usual
  datasets, including the target domain, and remain frozen in every phase.
- For each target, the unified token predictor is trained on the other four
  source domains only.
- Zero-shot evaluation applies that source-domain predictor directly to the
  predictor-held-out target.
- Few-shot adaptation initializes from the matching zero-shot checkpoint and
  uses only the first 5% or 10% of the target-domain training windows.
- Validation and test windows are never reduced by the few-shot ratio.
- The full-data reference column in Appendix F.2 directly reuses the main-table
  all-domain results. Reproduce it with `scripts/unify/run_predictor_seeds.py`;
  it is not a separate target-only adaptation run.

## Prerequisites

Train the three standard large-domain tokenizer groups first:

```bash
cd TSTokenizer
bash scripts/Cross_ETTh2_ETTm2_weather.sh
bash scripts/Cross_Traffic.sh
bash scripts/Cross_CzeLan.sh
cd ..
```

By default, the runner expects their checkpoints under
`TSTokenizer/checkpoints/`. Use `--tokenizer-path-template` when the checkpoint
root or naming convention differs. The template may contain `{pred_len}` and
must use `Cross_large` and `emb128` as the group and vocabulary placeholders.

## LODO Command

Train one predictor per held-out target on the other four source domains, then
evaluate each predictor directly on its held-out target:

```bash
bash scripts/holdout/run_lodo.sh \
  --pred-lens 48 \
  --seeds 2024 \
  --data-root /path/to/dataset \
  --text-tokenizer-root /path/to/GPT2 \
  --gpu 0 \
  --skip-existing
```

## Few-Shot Command

After the matching LODO checkpoints are available, initialize from them and
adapt on the first 5% and 10% of each target's training windows:

```bash
bash scripts/holdout/run_fewshot.sh \
  --pred-lens 48 \
  --seeds 2024 \
  --few-shot-ratios 0.05 0.10 \
  --data-root /path/to/dataset \
  --text-tokenizer-root /path/to/GPT2 \
  --gpu 0 \
  --skip-existing
```

Add `--dry-run --targets ETTh2` to either command to inspect its exact commands
without starting training. To repeat a run with another seed, change the value
passed to `--seeds`.

Use `--multi-gpu` when multiple GPUs are exposed through `--gpu`, for example
`--gpu 0,1 --multi-gpu`. The Python executable can be selected with
`PYTHON_BIN=/path/to/python` or `--python /path/to/python`. The GPT-2 location
can alternatively be supplied through `ONECAST_GPT2_ROOT`.

Summarize the completed logs:

```bash
python3 scripts/holdout/summarize_predictor_holdout.py \
  --pred-lens 48 \
  --seeds 2024 \
  --few-shot-ratios 0.05 0.10
```

Logs, exact `.cmd` snapshots, checkpoints, and the summary CSV are stored under
`outputs/predictor_holdout/` by default. Use `--output-root` to change this.
