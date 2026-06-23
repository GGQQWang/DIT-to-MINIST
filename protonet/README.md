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

Purpose: test whether moderate DDPM forward noise during episodic ProtoNet training improves prototype-based few-shot learning.

Recommended Fashion-MNIST run:

```bash
python protonet/noisy_protonet.py \
  --device cuda \
  --dataset fashion-mnist \
  --data-dir ./data \
  --output-dir ./protonet/outputs/noisy_protonet_fashion_mnist \
  --train-noise-timesteps 0,100,200,250,300,400,500 \
  --train-noise-target both \
  --eval-noise-target both \
  --way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 5000 \
  --eval-episodes 300 \
  --eval-interval 500 \
  --hidden-channels 64 \
  --embedding-dim 64 \
  --lr 1e-3 \
  --weight-decay 1e-4
```

Recommended Omniglot run:

```bash
python protonet/noisy_protonet.py \
  --device cuda \
  --dataset omniglot \
  --data-dir ./data \
  --output-dir ./protonet/outputs/noisy_protonet_omniglot \
  --train-noise-timesteps 0,100,200,250,300,400,500 \
  --train-noise-target both \
  --eval-noise-target both \
  --way 5 \
  --shot 5 \
  --query 15 \
  --train-episodes 5000 \
  --eval-episodes 600 \
  --eval-interval 500 \
  --hidden-channels 64 \
  --embedding-dim 64 \
  --lr 1e-3 \
  --weight-decay 1e-4
```

Key outputs:

- `train_eval_log.csv`
- `final_results.csv`

The default ProtoNet uses squared Euclidean distance without embedding normalization.
