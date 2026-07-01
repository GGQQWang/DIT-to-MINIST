#!/usr/bin/env bash
set -euo pipefail

# 扫描辅助原型去噪模块的两个关键超参：
# - noise-rho: 特征噪声强度，sigma = rho * Std(z_q)
# - lambda-denoise: 辅助去噪损失权重
#
# 用法：
#   bash protonet/run_aux_grid.sh
#
# 可选覆盖：
#   GPU_ID=1 TRAIN_EPISODES=20000 EVAL_EPISODES=600 bash protonet/run_aux_grid.sh

GPU_ID="${GPU_ID:-1}"
TRAIN_EPISODES="${TRAIN_EPISODES:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-600}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

RHO_LIST=(0.1 0.2 0.3 0.5)
LAMBDA_LIST=(0.1 0.05 0.3)

export HF_ENDPOINT

for rho in "${RHO_LIST[@]}"; do
  for lambda in "${LAMBDA_LIST[@]}"; do
    rho_tag="${rho/./p}"
    lambda_tag="${lambda/./p}"
    output_dir="./protonet/outputs/aux_grid_rho${rho_tag}_lambda${lambda_tag}_miniimagenet_20way_train_5way_eval"

    echo "============================================================"
    echo "running rho=${rho} lambda=${lambda}"
    echo "output_dir=${output_dir}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python protonet/noisy_protonet.py \
      --device cuda \
      --dataset miniimagenet-hf \
      --hf-dataset-id GATE-engine/mini_imagenet \
      --eval-split test \
      --image-size 84 \
      --preload-data cuda \
      --preload-batch-size 1024 \
      --num-workers 4 \
      --output-dir "${output_dir}" \
      --train-noise-timesteps 0 \
      --noise-space feature \
      --train-noise-target none \
      --eval-noise-target none \
      --loss-type ce \
      --aux-denoise \
      --noise-rho "${rho}" \
      --lambda-denoise "${lambda}" \
      --denoiser-hidden-multiplier 2 \
      --feature-normalization none \
      --distance-reduction mean \
      --optimizer adam \
      --lr 1e-3 \
      --lr-step-size 0 \
      --weight-decay 0 \
      --way 5 \
      --train-way 20 \
      --eval-way 5 \
      --shot 5 \
      --query 15 \
      --train-episodes "${TRAIN_EPISODES}" \
      --eval-episodes "${EVAL_EPISODES}" \
      --eval-interval "${EVAL_INTERVAL}" \
      --hidden-channels 64 \
      --encoder-head none \
      --embedding-dim 64
  done
done
