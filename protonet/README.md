# Noise-Regularized ProtoNet Experiments

This folder contains the current ProtoNet paper experiments.

## 1. Prototype-centered spectrum analysis

Purpose: verify how DDPM forward noise changes the class-wise covariance spectrum around class-average prototypes.

Recommended full run:

```bash
python protonet/prototype_centered_noise_spectrum.py \
  --device cuda \
  --data-dir ./data \
  --output-dir ./protonet/outputs/prototype_centered_noise_spectrum_full \
  --sample-split train \
  --prototype-per-class 0 \
  --samples-per-class 0 \
  --timesteps 0,100,200,250,300,400,500 \
  --diffusion-steps 1000 \
  --noise-repeats 1 \
  --center-mode scaled-prototype \
  --batch-size 2048 \
  --num-workers 8 \
  --eigen-top 120
```

Key outputs:

- `aggregate_spectrum_metrics.csv`
- `class_spectrum_metrics.csv`
- `eigenvalues_by_class.csv`
- `summary.md`

## 2. Noisy ProtoNet

Purpose: test whether moderate DDPM forward noise in embedding space improves prototype-based few-shot learning.

Default method:

- support images are encoded cleanly;
- support embeddings form clean prototypes;
- query embeddings receive DDPM forward noise;
- prototypes are synchronously scaled by `sqrt(alpha_bar_t)`;
- training uses a geometric distance loss with margin contrast and prototype separation.

Recommended Fashion-MNIST run:

```bash
python protonet/noisy_protonet.py \
  --device cuda \
  --dataset fashion-mnist \
  --data-dir ./data \
  --output-dir ./protonet/outputs/noisy_protonet_fashion_mnist \
  --train-noise-timesteps 0,100,200,250,300,400,500 \
  --noise-space feature \
  --train-noise-target query \
  --eval-noise-target query \
  --loss-type distance \
  --feature-normalization layernorm \
  --way 5 \
  --train-way 20 \
  --eval-way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 5000 \
  --eval-episodes 300 \
  --eval-interval 500 \
  --hidden-channels 64 \
  --embedding-dim 64 \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 5e-4
```

Recommended Omniglot run:

```bash
python protonet/noisy_protonet.py \
  --device cuda \
  --dataset omniglot \
  --data-dir ./data \
  --output-dir ./protonet/outputs/noisy_protonet_omniglot \
  --train-noise-timesteps 0,100,200,250,300,400,500 \
  --noise-space feature \
  --train-noise-target query \
  --eval-noise-target query \
  --loss-type distance \
  --feature-normalization layernorm \
  --way 5 \
  --train-way 20 \
  --eval-way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 5000 \
  --eval-episodes 600 \
  --eval-interval 500 \
  --hidden-channels 64 \
  --embedding-dim 64 \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 5e-4
```

Recommended miniImageNet run:

Option A: Hugging Face dataset, no manual conversion:

```bash
python protonet/noisy_protonet.py \
  --device cuda \
  --dataset miniimagenet-hf \
  --hf-dataset-id GATE-engine/mini_imagenet \
  --data-dir ./data \
  --eval-split test \
  --image-size 84 \
  --output-dir ./protonet/outputs/noisy_protonet_miniimagenet_hf_5way5shot \
  --train-noise-timesteps 0,100,200,250,300,400,500 \
  --noise-space feature \
  --train-noise-target query \
  --eval-noise-target query \
  --loss-type distance \
  --feature-normalization layernorm \
  --way 5 \
  --train-way 20 \
  --eval-way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 10000 \
  --eval-episodes 600 \
  --eval-interval 1000 \
  --hidden-channels 64 \
  --embedding-dim 64 \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 5e-4
```

Option B: local ImageFolder layout:

```txt
data/miniImageNet/
  train/
    class_001/*.jpg
    class_002/*.jpg
  val/
    class_064/*.jpg
  test/
    class_084/*.jpg
```

Then run:

```bash
python protonet/noisy_protonet.py \
  --device cuda \
  --dataset miniimagenet \
  --data-dir ./data \
  --eval-split test \
  --image-size 84 \
  --output-dir ./protonet/outputs/noisy_protonet_miniimagenet_5way5shot \
  --train-noise-timesteps 0,100,200,250,300,400,500 \
  --noise-space feature \
  --train-noise-target query \
  --eval-noise-target query \
  --loss-type distance \
  --feature-normalization layernorm \
  --way 5 \
  --train-way 20 \
  --eval-way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 10000 \
  --eval-episodes 600 \
  --eval-interval 1000 \
  --hidden-channels 64 \
  --embedding-dim 64 \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 5e-4
```

Key outputs:

- `train_eval_log.csv`
- `final_results.csv`

## Current miniImageNet run

This is the current diagnostic run: feature-space query noise at `t=250`, original ProtoNet CE loss, no feature LayerNorm, SGD optimizer, 20-way training and 5-way evaluation.

```bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=1 python protonet/noisy_protonet.py \
  --device cuda \
  --dataset miniimagenet-hf \
  --hf-dataset-id GATE-engine/mini_imagenet \
  --eval-split test \
  --image-size 84 \
  --output-dir ./protonet/outputs/ce_feature_noise_t250_sgd_20way_train_5way_eval \
  --train-noise-timesteps 250 \
  --noise-space feature \
  --train-noise-target query \
  --eval-noise-target query \
  --loss-type ce \
  --feature-normalization none \
  --distance-reduction mean \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --way 5 \
  --train-way 20 \
  --eval-way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 20000 \
  --eval-episodes 600 \
  --eval-interval 1000 \
  --hidden-channels 64 \
  --embedding-dim 64
```

Adam schedule variant:

```bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=1 python protonet/noisy_protonet.py \
  --device cuda \
  --dataset miniimagenet-hf \
  --hf-dataset-id GATE-engine/mini_imagenet \
  --eval-split test \
  --image-size 84 \
  --output-dir ./protonet/outputs/ce_feature_noise_t250_adam_step_20way_train_5way_eval \
  --train-noise-timesteps 250 \
  --noise-space feature \
  --train-noise-target query \
  --eval-noise-target query \
  --loss-type ce \
  --feature-normalization none \
  --distance-reduction mean \
  --optimizer adam \
  --lr 1e-3 \
  --lr-step-size 0 \
  --lr-gamma 0.5 \
  --weight-decay 0 \
  --way 5 \
  --train-way 20 \
  --eval-way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 20000 \
  --eval-episodes 600 \
  --eval-interval 1000 \
  --hidden-channels 64 \
  --embedding-dim 64
```

## Auxiliary Prototype Denoising Module

This command follows `protonet/新模块.md`: clean ProtoNet CE is the main loss, and an auxiliary residual MLP denoises noisy query embeddings back to stop-gradient class prototypes. Noise is controlled by relative feature scale `rho`, not by DDPM timestep.

```bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=1 python protonet/noisy_protonet.py \
  --device cuda \
  --dataset miniimagenet-hf \
  --hf-dataset-id GATE-engine/mini_imagenet \
  --eval-split test \
  --image-size 84 \
  --preload-data cuda \
  --preload-batch-size 1024 \
  --num-workers 4 \
  --output-dir ./protonet/outputs/aux_ddpm_t200_lambda0p3_miniimagenet_20way_train_5way_eval \
  --train-noise-timesteps 0 \
  --noise-space feature \
  --train-noise-target none \
  --eval-noise-target none \
  --loss-type ce \
  --aux-denoise \
  --train-schedule alternate \
  --main-steps 5 \
  --aux-steps 1 \
  --aux-noise-mode ddpm \
  --aux-noise-timestep 200 \
  --lambda-denoise 0.3 \
  --normalize-denoise-loss \
  --denoiser-hidden-multiplier 2 \
  --feature-normalization none \
  --distance-reduction mean \
  --optimizer adam \
  --lr 1e-3 \
  --lr-step-size 0 \
  --lr-gamma 0.5 \
  --weight-decay 0 \
  --way 5 \
  --train-way 20 \
  --eval-way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 20000 \
  --eval-episodes 600 \
  --eval-interval 1000 \
  --hidden-channels 64 \
  --encoder-head none \
  --embedding-dim 64
```

The default ProtoNet uses squared Euclidean distance without embedding normalization. `--encoder-head none` means Conv4 outputs the 64-channel pooled feature directly; `--encoder-head linear` restores the older extra projection head.

To scan fixed DDPM auxiliary noise `t=200` and `lambda-denoise in {0.1,0.3,1.0}`:

```bash
bash protonet/run_aux_grid.sh
```

To change GPU or shorten a dry run:

```bash
GPU_ID=1 TRAIN_EPISODES=1000 EVAL_EPISODES=100 EVAL_INTERVAL=500 AUX_NOISE_TIMESTEP=200 bash protonet/run_aux_grid.sh
```

To reproduce the older image-noise ProtoNet baseline, add:

```bash
--noise-space image \
--loss-type ce \
--feature-normalization none \
--train-noise-target both \
--eval-noise-target both
```
