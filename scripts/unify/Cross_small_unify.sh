#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
root_path_name="${ONECAST_DATA_ROOT:-${ROOT_DIR}/dataset}"
text_tokenizer_root="${ONECAST_GPT2_ROOT:-${ROOT_DIR}/pretrained/GPT2}"
tokenizer_root="${ONECAST_TOKENIZER_ROOT:-${ROOT_DIR}/TSTokenizer/checkpoints}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
cd "${ROOT_DIR}"
mkdir -p logs/unify/Cross_small

model_name=TokenTime

model_id_name=Cross_small
data_name=TFB

# VQVAE
token_len=12
wave_length=6
wave_stride=6
block_num=3
d_model=32
n_embed=32
VQ_type='W_SimVQ_decompose'
VQ_epoch=25
# 

# Backbone
backbone='trm' # 'trm' 'gpt2' 'qwen'
backbone_dim=128 # not for gpt2

random_seed=2024

n_classifier_layer=2

infer_step=4

mask_schedule='cosine' # 'cosine' 'linear' 'pow' 'sigmoid'

seq_len=36

elect_rate=0

for pred_len in 24 36 48 60 #  720 # 192      # 336 720
do
    vqvae_path="${tokenizer_root}/seed${random_seed}/${model_id_name}_${token_len}_sl${seq_len}_pl${pred_len}_dm${d_model}_dr0.2_emb${n_embed}_wl${wave_length}_bl${block_num}_${VQ_type}_epoch${VQ_epoch}"
    "${PYTHON_BIN}" -u run_longExp.py \
    --random_seed $random_seed \
    --use_multi_gpu \
    --cross_dataset "FRED-MD,NYSE,Covid-19,Wike2000" \
    --vqvae_path "$vqvae_path" \
    --text_tokenizer_root "$text_tokenizer_root" \
    --mask_schedule $mask_schedule \
    --is_training 1 \
    --is_unify \
    --backbone $backbone \
    --backbone_dim $backbone_dim \
    --root_path "$root_path_name" \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --VQ_type $VQ_type \
    --VQ_epoch $VQ_epoch \
    --elect_rate $elect_rate \
    --token_len $token_len \
    --wave_length $wave_length \
    --wave_stride $wave_stride \
    --block_num $block_num \
    --n_embed $n_embed \
    --d_model $d_model \
    --d_ff 256 \
    --dropout 0.2 \
    --des 'Exp' \
    --train_epochs 20 \
    --itr 1 \
    --batch_size 32 \
    --n_classifier_layer $n_classifier_layer \
    --infer_batch_size 32 \
    --infer_step $infer_step \
    --unify_lr 0.0001 \
    --adaptive_lr 0.0001 >logs/unify/Cross_small/$model_name'_'$backbone'_'$model_id_name'_'$seq_len'_'$pred_len'_'$VQ_type'_bl'$block_num'_wl'$wave_length'_tl'$token_len'_nl'$n_classifier_layer'_nb'$n_embed'_VQepoch'$VQ_epoch'_'$mask_schedule.log
done
