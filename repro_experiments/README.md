# Wine KDE 与 MNIST DiT 原型分类复现实验

本目录包含两个独立实验：

1. `wine_kde_classifier.py`：Wine 数据集 KDE 非参数分类器。
2. `mnist_dit_prototype.py`：MNIST 扩散加噪 + DiT 原型收缩分类实验。
3. `mnist_attraction_field_models.py`：MNIST 多模型原型吸引场对比实验，支持 Linear、MLP、CNN、CNN-Strong、DiT-XS、DiT-S、DiT-B。

## 1. 环境准备

建议在项目根目录或本目录中创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r repro_experiments/requirements.txt
```

如果已经在 `repro_experiments` 目录内，则执行：

```bash
pip install -r requirements.txt
```

## 2. Wine KDE 分类实验

### 实验思想

Wine 数据集包含 13 维化学特征和 3 个类别。KDE 分类器不训练神经网络参数，而是根据训练样本在特征空间中的核密度估计进行分类。实验中先对特征进行标准化，再分别估计测试样本在各类别条件分布下的密度，最终选择密度最高的类别作为预测结果。

### 默认参数

```txt
数据集：sklearn Wine
类别数：3
训练测试划分：70% / 30%
划分方式：分层随机划分
随机种子：42
特征标准化：StandardScaler
核函数：gaussian, epanechnikov
带宽：0.2,0.4,0.6,0.8,1.0,1.5,2.0
输出目录：./outputs_wine_kernel
```

### 复现命令

在项目根目录执行：

```bash
python repro_experiments/wine_kde_classifier.py --device cpu
```

只运行 Gaussian 核：

```bash
python repro_experiments/wine_kde_classifier.py \
  --device cpu \
  --kernels gaussian
```

自定义带宽搜索：

```bash
python repro_experiments/wine_kde_classifier.py \
  --device cpu \
  --kernels gaussian \
  --bandwidths 0.3,0.5,0.7,1.0,1.5
```

输出文件：

```txt
outputs_wine_kernel/kernel_kde_results.csv
outputs_wine_kernel/confusion_<kernel>_h<bandwidth>.csv
```

## 3. MNIST DiT 原型收缩分类实验

### 实验思想

原始 MNIST 图像受到书写风格、笔画粗细、倾斜角度等个体差异影响，直接进行类别平均原型匹配容易被样本独有细节干扰。本实验引入扩散加噪机制削弱高频个体细节，并训练 DiT 学习从带噪样本 `x_t` 到对应类别平均原型 `mu_y` 的非线性收缩映射。测试时，模型输出图像与 10 个类别原型比较距离，距离最近的原型类别即为预测类别。

这里提供两个 MNIST 版本：

```txt
mnist_dit_prototype.py：
  只比较 DiT-XS / DiT-S / DiT-B。
  模型输入为 x_t 和 t，先预测噪声，再反推 x0_pred。
  训练损失为 MSE(x0_pred, prototype_y)。

mnist_attraction_field_models.py：
  比较 Linear / MLP / CNN / CNN-Strong / DiT-XS / DiT-S / DiT-B。
  模型输入为 x_t 和 t，直接输出原型化图像 z。
  训练损失为 MSE(z, prototype_y)。
```

### 默认参数

```txt
数据集：MNIST
图像大小：1 x 28 x 28
类别数：10
模型：DiT-B
扩散步数：1000
训练目标：prototype
训练时间步范围：250-350
评估时间步：50,100,150
优化器：AdamW
学习率：1e-4
batch size：128
epoch：20
输出目录：./outputs
checkpoint：./checkpoints/dit_mnist.pt
```

### 快速小规模复现

如果只想确认流程可运行，可以使用较小模型和较少 epoch：

```bash
python repro_experiments/mnist_dit_prototype.py \
  --mode train \
  --dit-size XS \
  --epochs 1 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --target-mode prototype \
  --train-timestep-min 250 \
  --train-timestep-max 350 \
  --eval-timesteps 50,100,150 \
  --output-dir ./outputs_mnist_xs \
  --checkpoint ./checkpoints/dit_mnist_xs.pt \
  --device cpu
```

### 正式训练命令

有 GPU 时建议使用：

```bash
python repro_experiments/mnist_dit_prototype.py \
  --mode train \
  --dit-size B \
  --epochs 20 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --target-mode prototype \
  --train-timestep-min 250 \
  --train-timestep-max 350 \
  --eval-timesteps 50,100,150 \
  --output-dir ./outputs_mnist_dit \
  --checkpoint ./checkpoints/dit_mnist.pt \
  --device auto
```

### 只评估已有 checkpoint

```bash
python repro_experiments/mnist_dit_prototype.py \
  --mode eval \
  --checkpoint ./checkpoints/dit_mnist.pt \
  --output-dir ./outputs_mnist_dit \
  --eval-timesteps 50,100,150 \
  --device auto
```

### 扫描不同评估噪声时间步

```bash
python repro_experiments/mnist_dit_prototype.py \
  --mode sweep \
  --checkpoint ./checkpoints/dit_mnist.pt \
  --output-dir ./outputs_mnist_dit_sweep \
  --timestep-sweep "0;10;20;50;100;150;200;300;500;20,50,100;50,100,150;100,150,200" \
  --device auto
```

输出文件包括：

```txt
outputs_mnist_dit/noise_grid.png
outputs_mnist_dit/image_prototypes.png
outputs_mnist_dit/denoising_grid.png
outputs_mnist_dit/train_metrics.csv
outputs_mnist_dit/confusion_image_denoised.csv
outputs_mnist_dit_sweep/sweep_metrics.csv
```

## 4. 指标说明

Wine 实验输出：

```txt
accuracy：KDE 分类准确率
confusion matrix：真实类别与预测类别的混淆矩阵
```

MNIST 实验输出：

```txt
image_clean_acc：原始图像直接匹配类别原型的准确率
image_noisy_acc：加噪图像直接匹配类别原型的准确率
image_denoised_acc：DiT 输出图像匹配类别原型的准确率
```

其中 `image_denoised_acc` 是核心指标，用于衡量扩散加噪与 DiT 原型收缩映射是否提升了原型匹配分类效果。

## 5. MNIST 多模型原型吸引场对比

如果需要验证“不同模型是否都能学习从加噪样本到类别原型的映射”，使用：

```bash
python repro_experiments/mnist_attraction_field_models.py \
  --model dit-xs \
  --epochs 5 \
  --train-timestep 200 \
  --eval-timesteps 0,50,100,150,200,300 \
  --output-dir ./outputs_attraction_dit_xs \
  --device auto
```

可选模型包括：

```txt
linear
mlp
cnn
cnn-strong
unet
dit-xs
dit-s
dit-b
```

例如运行 CNN 对比：

```bash
python repro_experiments/mnist_attraction_field_models.py \
  --model cnn \
  --epochs 5 \
  --train-timestep 200 \
  --eval-timesteps 0,50,100,150,200,300 \
  --output-dir ./outputs_attraction_cnn \
  --device auto
```

该脚本的训练目标更直接：

```txt
输入：x_t
输出：model(x_t, t)
目标：prototype_y
损失：MSE(model(x_t, t), prototype_y)
分类：argmin_c ||model(x_t, t) - prototype_c||^2
```

因此，如果报告重点强调“加噪样本向类别原型吸引域收缩”，这个脚本是最直接的多模型验证版本。
