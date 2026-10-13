#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENIZER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TOKENIZER_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
root_path_name="${ONECAST_DATA_ROOT:-${ROOT_DIR}/dataset}"
random_seed="${ONECAST_RANDOM_SEED:-2024}"
tokenizer_root="${ONECAST_TOKENIZER_ROOT:-${TOKENIZER_DIR}/checkpoints}"
tokenizer_log_root="${ONECAST_TOKENIZER_LOG_ROOT:-${TOKENIZER_DIR}/logs/Tokenizer}"
skip_existing="${ONECAST_SKIP_EXISTING:-0}"
read -r -a pred_lens <<< "${ONECAST_PRED_LENS:-24 48 96 192}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
cd "${TOKENIZER_DIR}"
model_id_name=Cross_traffic
data_name=Cross

n_embed=128
wave_length=8
wave_stride=8
token_len=16
vq_model='W_SimVQ_decompose' # VanillaVQ, SimVQ, SimVQ_CNN, SimVQ_CNN_double_token
d_model=64
block_num=3

seq_len=96
pred_len=96

num_epoch=25
lr_decay_rate=0.99

log_dir="${tokenizer_log_root}/seed${random_seed}/${model_id_name}"
mkdir -p "${log_dir}"

for pred_len in "${pred_lens[@]}"; do
    save_path="${tokenizer_root}/seed${random_seed}/${model_id_name}_${token_len}_sl${seq_len}_pl${pred_len}_dm${d_model}_dr0.2_emb${n_embed}_wl${wave_length}_bl${block_num}_${vq_model}_epoch${num_epoch}"
    if [[ "${skip_existing}" == "1" && -f "${save_path}/model.pkl" && -f "${save_path}/weight.pkl" ]]; then
        echo "Skipping existing tokenizer checkpoint: ${save_path}"
        continue
    fi

    "${PYTHON_BIN}" -u main.py \
    --is_training 1 \
    --random_seed "${random_seed}" \
    --save_path "${save_path}" \
    --cross_dataset "traffic" \
    --seq_len $seq_len \
    --label_len 0 \
    --model_id $model_id_name \
    --pred_len $pred_len \
    --lr_decay_rate $lr_decay_rate \
    --vq_model $vq_model \
    --train_batch_size 16 \
    --test_batch_size 16 \
    --root_path "$root_path_name" \
    --data $data_name \
    --wave_length $wave_length \
    --wave_stride $wave_stride \
    --features M \
    --token_len $token_len \
    --n_embed $n_embed \
    --eval_per_epoch \
    --d_model $d_model \
    --block_num $block_num \
    --dropout 0.2 \
    --num_epoch $num_epoch \
    --lr 0.0005 >"${log_dir}/${model_id_name}_${token_len}_sl${seq_len}_pl${pred_len}_emb${n_embed}_wl${wave_length}_bl${block_num}_dm${d_model}_${vq_model}_epoch${num_epoch}.log"
done
