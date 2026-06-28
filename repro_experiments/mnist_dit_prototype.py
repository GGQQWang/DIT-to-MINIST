import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# MNIST DiT 原型收缩分类实验。
# 该版本保留扩散模型的“预测噪声”形式：
# 输入带噪图像 x_t 和时间步 t，模型预测噪声 eps_pred，
# 再通过扩散反推公式得到 x0_pred，并让 x0_pred 靠近真实类别原型。


@dataclass
class DiTConfig:
    """DiT 模型和扩散过程的基础配置。"""
    image_size: int = 28
    patch_size: int = 4
    in_channels: int = 1
    num_classes: int = 10
    depth: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_diffusion_steps: int = 1000


DIT_CONFIGS: Dict[str, Dict[str, int]] = {
    "B": {"depth": 12, "hidden_size": 768, "num_heads": 12},
    "S": {"depth": 8, "hidden_size": 384, "num_heads": 6},
    "XS": {"depth": 4, "hidden_size": 192, "num_heads": 3},
}


class PatchEmbed(nn.Module):
    """将图像切成 patch，并投影为 Transformer token。"""
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    """将扩散时间步 t 编码为隐藏向量，用于调制 DiT 模块。"""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN 调制：根据时间步嵌入对归一化后的特征做平移和缩放。"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT 基本块：自注意力 + MLP，并由时间条件控制 AdaLN。"""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # 时间条件 c 生成 attention 和 MLP 两个分支的 shift、scale、gate。
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class FinalLayer(nn.Module):
    """将 token 特征映射回每个 patch 对应的噪声预测。"""
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class ImageDenoisingDiT(nn.Module):
    """图像去噪 DiT：输入 x_t 和 t，输出与图像同形状的噪声预测。"""
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, cfg.hidden_size), requires_grad=False)
        self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, self.patch_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """初始化 DiT 权重；最后层置零使模型初始输出更稳定。"""
        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(init_linear)
        self.pos_embed.data.copy_(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size))
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.shape[0], -1))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """图像编码为 patch token，并加入固定位置编码。"""
        return self.patch_embed(x) + self.pos_embed

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """将 patch 级输出重新拼回 1 x 28 x 28 图像。"""
        batch_size = x.shape[0]
        patch_size = self.cfg.patch_size
        channels = self.cfg.in_channels
        grid_size = self.patch_embed.grid_size
        x = x.reshape(batch_size, grid_size, grid_size, patch_size, patch_size, channels)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(batch_size, channels, grid_size * patch_size, grid_size * patch_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """单次前向：根据带噪图像和时间步预测噪声 eps_pred。"""
        c = self.t_embedder(t)
        x = self.encode(x_t)
        for block in self.blocks:
            x = block(x, c)
        patch_noise = self.final_layer(x, c)
        return self.unpatchify(patch_noise)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """生成二维正弦余弦位置编码。"""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return torch.from_numpy(pos_embed).float().unsqueeze(0)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid) -> "numpy.ndarray":
    import numpy as np

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos) -> "numpy.ndarray":
    import numpy as np

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class DiffusionSchedule:
    """DDPM 前向加噪日程，提供 q_sample 和一步反推 x0 的公式。"""
    def __init__(self, timesteps: int, device: torch.device):
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """前向扩散：把原图 x0 加噪成 x_t。"""
        shape = (t.shape[0],) + (1,) * (x0.dim() - 1)
        sqrt_ab = self.sqrt_alpha_bars[t].view(shape)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(shape)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """根据模型预测的噪声 eps_pred，一步反推出 x0_pred。"""
        shape = (t.shape[0],) + (1,) * (x_t.dim() - 1)
        sqrt_ab = self.sqrt_alpha_bars[t].view(shape)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(shape)
        return (x_t - sqrt_omab * eps_pred) / sqrt_ab.clamp_min(1e-8)


def build_model(args: argparse.Namespace) -> ImageDenoisingDiT:
    """根据命令行选择 DiT-B、DiT-S 或 DiT-XS。"""
    size_cfg = DIT_CONFIGS[args.dit_size]
    cfg = DiTConfig(
        depth=size_cfg["depth"],
        hidden_size=size_cfg["hidden_size"],
        num_heads=size_cfg["num_heads"],
        dropout=args.dropout,
        num_diffusion_steps=args.diffusion_steps,
    )
    return ImageDenoisingDiT(cfg)


def get_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """加载 MNIST，并归一化到 [-1, 1]。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(
    model: ImageDenoisingDiT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    epoch: int,
    log_interval: int,
    grad_clip: float,
    image_prototypes: torch.Tensor,
    target_mode: str,
    train_timestep_min: int,
    train_timestep_max: int,
) -> float:
    """训练一个 epoch。

    target_mode=noise：普通扩散去噪，直接拟合真实噪声。
    target_mode=prototype：先预测噪声并一步反推 x0_pred，再让 x0_pred 靠近真实类别原型。
    """
    model.train()
    total_loss = 0.0
    total_count = 0
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        noise = torch.randn_like(images)
        t = torch.randint(train_timestep_min, train_timestep_max + 1, (images.shape[0],), device=device)
        x_t = schedule.q_sample(images, t, noise)
        eps_pred = model(x_t, t)
        if target_mode == "noise":
            loss = F.mse_loss(eps_pred, noise)
        elif target_mode == "prototype":
            # 这里不是还原原始个体图像，而是让反推出的图像靠近类别平均原型。
            x0_pred = schedule.predict_x0(x_t, t, eps_pred).clamp(-1, 1)
            loss = F.mse_loss(x0_pred, image_prototypes[labels])
        else:
            raise ValueError(f"Unsupported target_mode: {target_mode}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size
        if step % log_interval == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} denoise_loss={total_loss / total_count:.6f}")
    return total_loss / max(total_count, 1)


@torch.no_grad()
def build_prototypes(
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> torch.Tensor:
    """计算每个数字类别的平均图像原型 mu_c。"""
    image_sums = torch.zeros(num_classes, 1, 28, 28, device=device)
    counts = torch.zeros(num_classes, device=device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                image_sums[cls] += images[mask].sum(dim=0)
                counts[cls] += mask.sum()

    counts = counts.clamp_min(1).view(-1, 1, 1, 1)
    return image_sums / counts


def nearest_image_proto(images: torch.Tensor, image_prototypes: torch.Tensor) -> torch.Tensor:
    """最近原型分类：选择 MSE 距离最小的类别。"""
    errors = (images[:, None] - image_prototypes[None]).pow(2).mean(dim=(2, 3, 4))
    return errors.argmin(dim=1)


@torch.no_grad()
def denoise_images(
    model: ImageDenoisingDiT,
    images: torch.Tensor,
    schedule: DiffusionSchedule,
    device: torch.device,
    eval_timesteps: Iterable[int],
    noise_repeats: int,
) -> torch.Tensor:
    """对图像做加噪和单次 DiT 去噪，并可对多个时间步/噪声重复取平均。

    注意：每个时间步内部都是一次模型前向预测噪声，不是多步扩散采样。
    """
    batch_size = images.shape[0]
    x0_pred_sum = torch.zeros_like(images)
    count = 0

    for timestep in eval_timesteps:
        t = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
        for _ in range(noise_repeats):
            noise = torch.randn_like(images)
            x_t = schedule.q_sample(images, t, noise)
            eps_pred = model(x_t, t)
            # 单次预测 eps_pred 后，用闭式公式一步反推 x0_pred。
            x0_pred_sum += schedule.predict_x0(x_t, t, eps_pred)
            count += 1

    return (x0_pred_sum / max(count, 1)).clamp(-1, 1)


@torch.no_grad()
def evaluate_prototypes(
    model: ImageDenoisingDiT,
    loader: DataLoader,
    image_prototypes: torch.Tensor,
    schedule: DiffusionSchedule,
    device: torch.device,
    eval_timesteps: Iterable[int],
    noise_repeats: int,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """比较 clean/noisy/denoised 三种图像的最近原型分类准确率。"""
    model.eval()
    correct = {
        "image_clean": 0,
        "image_noisy": 0,
        "image_denoised": 0,
    }
    confusion = torch.zeros(model.cfg.num_classes, model.cfg.num_classes, dtype=torch.long)
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        noisy_images = denoise_image_baseline(images, schedule, device, eval_timesteps, noise_repeats)
        denoised_images = denoise_images(model, images, schedule, device, eval_timesteps, noise_repeats)

        # image_denoised 是核心指标：模型原型化输出是否比原图/加噪图更适合原型匹配。
        preds = {
            "image_clean": nearest_image_proto(images, image_prototypes),
            "image_noisy": nearest_image_proto(noisy_images, image_prototypes),
            "image_denoised": nearest_image_proto(denoised_images, image_prototypes),
        }
        for name, pred in preds.items():
            correct[name] += (pred == labels).sum().item()
        for true_label, pred_label in zip(labels.cpu(), preds["image_denoised"].cpu()):
            confusion[true_label.long(), pred_label.long()] += 1
        total += labels.numel()

    metrics = {name: value / max(total, 1) for name, value in correct.items()}
    return metrics, confusion


@torch.no_grad()
def denoise_image_baseline(
    images: torch.Tensor,
    schedule: DiffusionSchedule,
    device: torch.device,
    eval_timesteps: Iterable[int],
    noise_repeats: int,
) -> torch.Tensor:
    """不经过模型，仅对加噪图像求平均，作为 noisy baseline。"""
    image_sum = torch.zeros_like(images)
    count = 0
    for timestep in eval_timesteps:
        t = torch.full((images.shape[0],), int(timestep), device=device, dtype=torch.long)
        for _ in range(noise_repeats):
            noise = torch.randn_like(images)
            image_sum += schedule.q_sample(images, t, noise)
            count += 1
    return image_sum / max(count, 1)


def save_checkpoint(
    path: str,
    model: ImageDenoisingDiT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    """保存训练 checkpoint。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str, model: ImageDenoisingDiT, device: torch.device, optimizer=None) -> int:
    """加载 checkpoint，并返回保存时的 epoch。"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


def parse_eval_timesteps(value: str, diffusion_steps: int) -> Tuple[int, ...]:
    """解析评估使用的扩散时间步。"""
    steps = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not steps:
        raise ValueError("--eval-timesteps cannot be empty")
    for step in steps:
        if step < 0 or step >= diffusion_steps:
            raise ValueError(f"eval timestep {step} is outside [0, {diffusion_steps - 1}]")
    return steps


def parse_timestep_sweep(value: str, diffusion_steps: int) -> Tuple[Tuple[int, ...], ...]:
    """解析噪声时间步扫描配置，分号分隔不同评估组。"""
    groups = []
    for group in value.split(";"):
        group = group.strip()
        if group:
            groups.append(parse_eval_timesteps(group, diffusion_steps))
    if not groups:
        raise ValueError("--timestep-sweep cannot be empty")
    return tuple(groups)


def parse_args() -> argparse.Namespace:
    """定义 MNIST DiT 原型收缩实验参数。"""
    parser = argparse.ArgumentParser(description="Unconditional DiT image denoising with prototype matching for MNIST.")
    parser.add_argument("--mode", choices=["train", "eval", "sweep"], default="train")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/dit_mnist.pt")
    parser.add_argument("--dit-size", choices=sorted(DIT_CONFIGS), default="B")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--target-mode", choices=["noise", "prototype"], default="prototype")
    parser.add_argument("--train-timestep-min", type=int, default=250)
    parser.add_argument("--train-timestep-max", type=int, default=350)
    parser.add_argument("--eval-timesteps", default="50,100,150")
    parser.add_argument(
        "--timestep-sweep",
        default="0;10;20;50;100;150;200;300;500;20,50,100;50,100,150;100,150,200",
    )
    parser.add_argument("--viz-timesteps", default="0,20,50,100,150,200,300,500")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """自动选择可用设备。"""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def format_metrics(metrics: Dict[str, float]) -> str:
    return " ".join(f"{name}_acc={value:.4f}" for name, value in metrics.items())


def format_timesteps(timesteps: Tuple[int, ...]) -> str:
    return ",".join(str(step) for step in timesteps)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_training_log(path: str, epoch: int, train_loss: float, metrics: Dict[str, float]) -> None:
    """追加保存每个 epoch 的训练损失和评估准确率。"""
    ensure_dir(os.path.dirname(path) or ".")
    fieldnames = ["epoch", "train_loss", *metrics.keys()]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {"epoch": epoch, "train_loss": train_loss}
        row.update(metrics)
        writer.writerow(row)


def write_sweep_log(path: str, rows: Iterable[Dict[str, float]]) -> None:
    """保存不同 eval_timesteps 组合下的扫描结果。"""
    rows = list(rows)
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: str, confusion: torch.Tensor) -> None:
    """保存 image_denoised 预测结果的混淆矩阵。"""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *range(confusion.shape[1])])
        for cls, row in enumerate(confusion.tolist()):
            writer.writerow([cls, *row])


def save_image_prototypes(path: str, image_prototypes: torch.Tensor) -> None:
    """保存 0-9 类平均原型图像。"""
    ensure_dir(os.path.dirname(path) or ".")
    images = (image_prototypes.detach().cpu().clamp(-1, 1) + 1) / 2
    save_image(images, path, nrow=image_prototypes.shape[0], padding=2)


@torch.no_grad()
def save_noise_grid(
    path: str,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    timesteps: Iterable[int],
    max_images: int = 8,
) -> None:
    """保存不同时间步的加噪可视化。"""
    ensure_dir(os.path.dirname(path) or ".")
    images, _ = next(iter(loader))
    images = images[:max_images].to(device)
    rows = [images]
    for timestep in timesteps:
        t = torch.full((images.shape[0],), int(timestep), device=device, dtype=torch.long)
        rows.append(schedule.q_sample(images, t, torch.randn_like(images)))
    grid_images = torch.cat(rows, dim=0)
    grid_images = (grid_images.detach().cpu().clamp(-1, 1) + 1) / 2
    save_image(grid_images, path, nrow=max_images, padding=2)


@torch.no_grad()
def save_denoising_grid(
    path: str,
    model: ImageDenoisingDiT,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    timesteps: Iterable[int],
    max_images: int = 8,
) -> None:
    """保存带噪图像和一步反推结果的可视化对比。"""
    ensure_dir(os.path.dirname(path) or ".")
    model.eval()
    images, _ = next(iter(loader))
    images = images[:max_images].to(device)
    rows = [images]
    for timestep in timesteps:
        t = torch.full((images.shape[0],), int(timestep), device=device, dtype=torch.long)
        noise = torch.randn_like(images)
        x_t = schedule.q_sample(images, t, noise)
        eps_pred = model(x_t, t)
        rows.append(x_t)
        rows.append(schedule.predict_x0(x_t, t, eps_pred).clamp(-1, 1))
    grid_images = torch.cat(rows, dim=0)
    grid_images = (grid_images.detach().cpu().clamp(-1, 1) + 1) / 2
    save_image(grid_images, path, nrow=max_images, padding=2)


def main() -> None:
    """主入口：构建原型、训练/评估 DiT，并保存指标和可视化结果。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_loader, test_loader = get_loaders(args)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    if args.train_timestep_min < 0 or args.train_timestep_max >= args.diffusion_steps:
        raise ValueError("--train-timestep-min/max must be inside the diffusion step range")
    if args.train_timestep_min > args.train_timestep_max:
        raise ValueError("--train-timestep-min cannot be larger than --train-timestep-max")
    eval_timesteps = parse_eval_timesteps(args.eval_timesteps, args.diffusion_steps)
    viz_timesteps = parse_eval_timesteps(args.viz_timesteps, args.diffusion_steps)
    ensure_dir(args.output_dir)
    save_noise_grid(os.path.join(args.output_dir, "noise_grid.png"), train_loader, schedule, device, viz_timesteps)
    # 类别原型只由训练集构建，避免使用测试集信息。
    image_prototypes = build_prototypes(train_loader, device, model.cfg.num_classes)
    save_image_prototypes(os.path.join(args.output_dir, "image_prototypes.png"), image_prototypes)

    print(
        f"device={device} dit={args.dit_size} "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
        f"target_mode={args.target_mode} train_t={args.train_timestep_min}-{args.train_timestep_max}"
    )

    if args.mode == "eval":
        # eval 模式只加载已有 checkpoint，不继续训练。
        load_checkpoint(args.checkpoint, model, device)
        save_denoising_grid(os.path.join(args.output_dir, "denoising_grid.png"), model, test_loader, schedule, device, eval_timesteps)
        metrics, confusion = evaluate_prototypes(
            model,
            test_loader,
            image_prototypes,
            schedule,
            device,
            eval_timesteps,
            args.noise_repeats,
        )
        write_confusion_csv(os.path.join(args.output_dir, "confusion_image_denoised.csv"), confusion)
        print(format_metrics(metrics))
        return

    if args.mode == "sweep":
        # sweep 模式用于分析不同评估噪声强度对原型分类的影响。
        load_checkpoint(args.checkpoint, model, device)
        sweep_rows = []
        for timestep_group in parse_timestep_sweep(args.timestep_sweep, args.diffusion_steps):
            metrics, confusion = evaluate_prototypes(
                model,
                test_loader,
                image_prototypes,
                schedule,
                device,
                timestep_group,
                args.noise_repeats,
            )
            row = {"eval_timesteps": format_timesteps(timestep_group)}
            row.update(metrics)
            sweep_rows.append(row)
            write_confusion_csv(
                os.path.join(args.output_dir, f"confusion_image_denoised_t{format_timesteps(timestep_group).replace(',', '_')}.csv"),
                confusion,
            )
            print(f"eval_timesteps={format_timesteps(timestep_group)} {format_metrics(metrics)}")
        write_sweep_log(os.path.join(args.output_dir, "sweep_metrics.csv"), sweep_rows)
        return

    start_epoch = 0
    if os.path.exists(args.checkpoint):
        start_epoch = load_checkpoint(args.checkpoint, model, device, optimizer)
        print(f"resumed_from={args.checkpoint} epoch={start_epoch}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            schedule,
            device,
            epoch,
            args.log_interval,
            args.grad_clip,
            image_prototypes,
            args.target_mode,
            args.train_timestep_min,
            args.train_timestep_max,
        )
        save_denoising_grid(os.path.join(args.output_dir, "denoising_grid.png"), model, test_loader, schedule, device, eval_timesteps)
        metrics, confusion = evaluate_prototypes(
            model,
            test_loader,
            image_prototypes,
            schedule,
            device,
            eval_timesteps,
            args.noise_repeats,
        )
        append_training_log(os.path.join(args.output_dir, "train_metrics.csv"), epoch, loss, metrics)
        write_confusion_csv(os.path.join(args.output_dir, "confusion_image_denoised.csv"), confusion)
        save_checkpoint(args.checkpoint, model, optimizer, epoch, args)
        print(f"epoch={epoch} train_loss={loss:.6f} {format_metrics(metrics)} saved={args.checkpoint}")


if __name__ == "__main__":
    main()
