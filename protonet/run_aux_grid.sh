#!/usr/bin/env bash
set -euo pipefail

# 扫描辅助原型去噪模块的损失权重。
# 当前版本固定使用 DDPM feature-space t=200：
# z_t = sqrt(alpha_bar_t) * z_q + sqrt(1-alpha_bar_t) * Std(z_q) * eps
#
# 用法：
#   bash protonet/run_aux_grid.sh
#
# 可选覆盖：
#   GPU_ID=1 TRAIN_EPISODES=20000 EVAL_EPISODES=600 bash protonet/run_aux_grid.sh

GPU_ID="${GPU_ID:-2}"
TRAIN_EPISODES="${TRAIN_EPISODES:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-600}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

AUX_NOISE_TIMESTEP="${AUX_NOISE_TIMESTEP:-200}"
LAMBDA_LIST=(0.1 0.3 1.0)

export HF_ENDPOINT

for lambda in "${LAMBDA_LIST[@]}"; do
    lambda_tag="${lambda/./p}"
    output_dir="./protonet/outputs/aux_ddpm_t${AUX_NOISE_TIMESTEP}_lambda${lambda_tag}_miniimagenet_20way_train_5way_eval"

    echo "============================================================"
    echo "running aux_noise=ddpm aux_t=${AUX_NOISE_TIMESTEP} lambda=${lambda}"
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
      --train-schedule alternate \
      --main-steps 5 \
      --aux-steps 1 \
      --freeze-bn-in-aux \
      --aux-noise-mode ddpm \
      --aux-noise-timestep "${AUX_NOISE_TIMESTEP}" \
      --lambda-denoise "${lambda}" \
      --normalize-denoise-loss \
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
