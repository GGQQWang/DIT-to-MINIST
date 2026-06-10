import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class DiTConfig:
    image_size: int = 28
    patch_size: int = 4
    in_channels: int = 1
    depth: int = 4
    hidden_size: int = 192
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0


DIT_CONFIGS: Dict[str, Dict[str, int]] = {
    "dit-b": {"depth": 12, "hidden_size": 768, "num_heads": 12},
    "dit-s": {"depth": 8, "hidden_size": 384, "num_heads": 6},
    "dit-xs": {"depth": 4, "hidden_size": 192, "num_heads": 3},
}


def parse_int_list(value: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("integer list cannot be empty")
    return values


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def balanced_subset_indices_from_pool(pool: Sequence[int], labels: torch.Tensor, per_class: int, num_classes: int, seed: int) -> List[int]:
    generator = torch.Generator().manual_seed(seed)
    pool_tensor = torch.as_tensor(list(pool), dtype=torch.long)
    indices: List[int] = []
    for cls in range(num_classes):
        cls_indices = pool_tensor[labels[pool_tensor] == cls]
        perm = torch.randperm(cls_indices.numel(), generator=generator)
        take = cls_indices.numel() if per_class <= 0 else min(per_class, cls_indices.numel())
        indices.extend(cls_indices[perm[:take]].tolist())
    return indices


def split_train_val_indices(dataset, args: argparse.Namespace) -> Tuple[List[int], List[int]]:
    labels = torch.as_tensor(dataset.targets)
    generator = torch.Generator().manual_seed(args.seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for cls in range(args.num_classes):
        cls_indices = torch.where(labels == cls)[0]
        permuted = cls_indices[torch.randperm(cls_indices.numel(), generator=generator)]
        val_take = min(args.val_per_class, cls_indices.numel())
        val_cls = permuted[:val_take]
        train_pool = permuted[val_take:]
        train_take = train_pool.numel() if args.train_per_class <= 0 else min(args.train_per_class, train_pool.numel())
        train_cls = train_pool[:train_take]
        val_indices.extend(val_cls.tolist())
        train_indices.extend(train_cls.tolist())

    return train_indices, val_indices


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_indices, val_indices = split_train_val_indices(train_set, args)
    test_indices = balanced_subset_indices_from_pool(
        range(len(test_set)),
        torch.as_tensor(test_set.targets),
        args.test_per_class,
        args.num_classes,
        args.seed + 1000,
    )
    train_loader = DataLoader(
        Subset(train_set, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(train_set, val_indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(test_set, test_indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    prototype_loader = DataLoader(
        Subset(train_set, train_indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, prototype_loader


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device):
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (x0.dim() - 1)
        sqrt_ab = self.sqrt_alpha_bars[t].view(shape)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(shape)
        return sqrt_ab * x0 + sqrt_omab * noise

    def scaled_centers(self, centers: torch.Tensor, timestep: int) -> torch.Tensor:
        return self.sqrt_alpha_bars[timestep] * centers


class TimestepEmbedder(nn.Module):
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


class LinearAttractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(28 * 28, 28 * 28)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(x.flatten(start_dim=1)).view_as(x)


class MLPAttractor(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.in_proj = nn.Linear(28 * 28, hidden_size)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 28 * 28),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.flatten(start_dim=1)) + self.t_embedder(t)
        return self.net(h).view_as(x)


class CNNAttractor(nn.Module):
    def __init__(self, hidden_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.t_embedder = TimestepEmbedder(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_channels)
        self.in_conv = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.out_conv = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        time = self.time_proj(self.t_embedder(t)).view(x.shape[0], -1, 1, 1)
        h = h + time
        return self.out_conv(self.blocks(h)).clamp(-1, 1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_embedding).view(x.shape[0], -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class StrongCNNAttractor(nn.Module):
    def __init__(self, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.t_embedder = TimestepEmbedder(time_dim)
        self.in_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.enc1 = ResidualConvBlock(base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ResidualConvBlock(base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.mid1 = ResidualConvBlock(base_channels * 4, time_dim)
        self.mid2 = ResidualConvBlock(base_channels * 4, time_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualConvBlock(base_channels * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualConvBlock(base_channels, time_dim)
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embedding = self.t_embedder(t)
        h1 = self.enc1(self.in_conv(x), time_embedding)
        h2 = self.enc2(self.down1(h1), time_embedding)
        h = self.down2(h2)
        h = self.mid2(self.mid1(h, time_embedding), time_embedding)
        h = self.dec1(self.up1(h) + h2, time_embedding)
        h = self.dec2(self.up2(h) + h1, time_embedding)
        return self.out_conv(F.silu(self.out_norm(h))).clamp(-1, 1)


class PatchEmbed(nn.Module):
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiTAttractor(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, cfg.hidden_size), requires_grad=False)
        self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, self.patch_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
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

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patch_size = self.cfg.patch_size
        channels = self.cfg.in_channels
        grid_size = self.patch_embed.grid_size
        x = x.reshape(batch_size, grid_size, grid_size, patch_size, patch_size, channels)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(batch_size, channels, grid_size * patch_size, grid_size * patch_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t)
        h = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            h = block(h, c)
        return self.unpatchify(self.final_layer(h, c)).clamp(-1, 1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
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


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "linear":
        return LinearAttractor()
    if args.model == "mlp":
        return MLPAttractor(args.mlp_hidden_size)
    if args.model == "cnn":
        return CNNAttractor(args.cnn_channels)
    if args.model == "cnn-strong":
        return StrongCNNAttractor(args.cnn_channels)
    if args.model in DIT_CONFIGS:
        cfg_values = DIT_CONFIGS[args.model]
        return DiTAttractor(
            DiTConfig(
                depth=cfg_values["depth"],
                hidden_size=cfg_values["hidden_size"],
                num_heads=cfg_values["num_heads"],
                dropout=args.dropout,
            )
        )
    raise ValueError(f"unsupported model: {args.model}")


@torch.no_grad()
def build_prototypes(loader: DataLoader, device: torch.device, num_classes: int) -> torch.Tensor:
    sums = torch.zeros(num_classes, 1, 28, 28, device=device)
    counts = torch.zeros(num_classes, device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                sums[cls] += images[mask].sum(dim=0)
                counts[cls] += mask.sum()
    return sums / counts.clamp_min(1).view(-1, 1, 1, 1)


def nearest_proto_metrics(outputs: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> Dict[str, float]:
    residuals = (outputs[:, None] - prototypes[None]).pow(2).mean(dim=(2, 3, 4))
    preds = residuals.argmin(dim=1)
    true_residual = residuals[torch.arange(labels.shape[0], device=labels.device), labels]
    wrong = residuals.clone()
    wrong[torch.arange(labels.shape[0], device=labels.device), labels] = float("inf")
    nearest_wrong = wrong.min(dim=1).values
    margin = nearest_wrong - true_residual
    return {
        "correct": int((preds == labels).sum().item()),
        "total": int(labels.numel()),
        "margin_sum": float(margin.sum().item()),
        "true_distance_sum": float(true_residual.sum().item()),
        "wrong_distance_sum": float(nearest_wrong.sum().item()),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    prototypes: torch.Tensor,
    device: torch.device,
    train_timestep: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t = torch.full((images.shape[0],), train_timestep, device=device, dtype=torch.long)
        x_t = schedule.q_sample(images, t, torch.randn_like(images))
        outputs = model(x_t, t)
        loss = F.mse_loss(outputs, prototypes[labels])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.shape[0]
        total_count += images.shape[0]
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    prototypes: torch.Tensor,
    device: torch.device,
    eval_timestep: int,
    noise_repeats: int,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "model_correct": 0,
        "noisy_correct": 0,
        "clean_correct": 0,
        "total": 0,
        "model_margin_sum": 0.0,
        "noisy_margin_sum": 0.0,
        "clean_margin_sum": 0.0,
        "model_true_distance_sum": 0.0,
        "noisy_true_distance_sum": 0.0,
        "clean_true_distance_sum": 0.0,
        "model_wrong_distance_sum": 0.0,
        "noisy_wrong_distance_sum": 0.0,
        "clean_wrong_distance_sum": 0.0,
        "contraction_sum": 0.0,
        "contraction_count": 0,
    }
    scaled_prototypes = schedule.scaled_centers(prototypes, eval_timestep)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_metrics = nearest_proto_metrics(images, labels, prototypes)
        totals["clean_correct"] += clean_metrics["correct"]
        totals["clean_margin_sum"] += clean_metrics["margin_sum"]
        totals["clean_true_distance_sum"] += clean_metrics["true_distance_sum"]
        totals["clean_wrong_distance_sum"] += clean_metrics["wrong_distance_sum"]
        totals["total"] += labels.numel()

        for _ in range(noise_repeats):
            t = torch.full((images.shape[0],), eval_timestep, device=device, dtype=torch.long)
            x_t = schedule.q_sample(images, t, torch.randn_like(images))
            outputs = model(x_t, t)
            noisy_metrics = nearest_proto_metrics(x_t, labels, scaled_prototypes)
            model_metrics = nearest_proto_metrics(outputs, labels, prototypes)

            totals["noisy_correct"] += noisy_metrics["correct"]
            totals["model_correct"] += model_metrics["correct"]
            totals["noisy_margin_sum"] += noisy_metrics["margin_sum"]
            totals["model_margin_sum"] += model_metrics["margin_sum"]
            totals["noisy_true_distance_sum"] += noisy_metrics["true_distance_sum"]
            totals["model_true_distance_sum"] += model_metrics["true_distance_sum"]
            totals["noisy_wrong_distance_sum"] += noisy_metrics["wrong_distance_sum"]
            totals["model_wrong_distance_sum"] += model_metrics["wrong_distance_sum"]

            numerator = (outputs - prototypes[labels]).flatten(start_dim=1).norm(dim=1)
            denominator = (x_t - scaled_prototypes[labels]).flatten(start_dim=1).norm(dim=1).clamp_min(1e-8)
            totals["contraction_sum"] += float((numerator / denominator).sum().item())
            totals["contraction_count"] += labels.numel()

    repeated_total = max(totals["total"] * noise_repeats, 1)
    total = max(totals["total"], 1)
    return {
        "eval_timestep": float(eval_timestep),
        "clean_acc": totals["clean_correct"] / total,
        "noisy_acc": totals["noisy_correct"] / repeated_total,
        "model_acc": totals["model_correct"] / repeated_total,
        "clean_margin": totals["clean_margin_sum"] / total,
        "noisy_margin": totals["noisy_margin_sum"] / repeated_total,
        "model_margin": totals["model_margin_sum"] / repeated_total,
        "clean_true_distance": totals["clean_true_distance_sum"] / total,
        "noisy_true_distance": totals["noisy_true_distance_sum"] / repeated_total,
        "model_true_distance": totals["model_true_distance_sum"] / repeated_total,
        "clean_wrong_distance": totals["clean_wrong_distance_sum"] / total,
        "noisy_wrong_distance": totals["noisy_wrong_distance_sum"] / repeated_total,
        "model_wrong_distance": totals["model_wrong_distance_sum"] / repeated_total,
        "contraction": totals["contraction_sum"] / max(totals["contraction_count"], 1),
    }


def write_csv(path: str, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_model_checkpoint(path: str, model: nn.Module, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return int(checkpoint.get("epoch", 0))


def write_summary(
    path: str,
    args: argparse.Namespace,
    train_rows: List[Dict[str, float]],
    val_rows: List[Dict[str, float]],
    test_rows: List[Dict[str, float]],
    best_epoch: int,
    best_val_acc: float,
) -> None:
    best = max(test_rows, key=lambda row: row["model_acc"])
    params = train_rows[-1]["params_m"] if train_rows else 0.0
    with open(path, "w") as f:
        f.write("# Attraction Field Model Summary\n\n")
        f.write(f"- model: {args.model}\n")
        f.write(f"- params_m: {params:.4f}\n")
        f.write(f"- train_timestep: {args.train_timestep}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- best_epoch: {best_epoch}\n")
        f.write(f"- best_val_acc: {best_val_acc:.6f}\n")
        f.write(f"- train_per_class: {args.train_per_class}\n")
        f.write(f"- val_per_class: {args.val_per_class}\n")
        f.write(f"- test_per_class: {args.test_per_class}\n")
        f.write(f"- noise_repeats: {args.noise_repeats}\n\n")
        f.write("## Best Test Eval\n\n")
        f.write(f"- eval_timestep: {int(best['eval_timestep'])}\n")
        f.write(f"- model_acc: {best['model_acc']:.6f}\n")
        f.write(f"- noisy_acc: {best['noisy_acc']:.6f}\n")
        f.write(f"- clean_acc: {best['clean_acc']:.6f}\n")
        f.write(f"- model_margin: {best['model_margin']:.6f}\n")
        f.write(f"- contraction: {best['contraction']:.6f}\n\n")
        f.write("## Test Rows\n\n")
        f.write("| eval_timestep | clean_acc | noisy_acc | model_acc | model_margin | contraction |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in test_rows:
            f.write(
                f"| {int(row['eval_timestep'])} | {row['clean_acc']:.6f} | "
                f"{row['noisy_acc']:.6f} | {row['model_acc']:.6f} | "
                f"{row['model_margin']:.6f} | {row['contraction']:.6f} |\n"
            )
        if val_rows:
            f.write("\n## Validation Selection\n\n")
            f.write("| epoch | val_acc | val_margin | val_contraction |\n")
            f.write("| ---: | ---: | ---: | ---: |\n")
            for row in val_rows:
                f.write(
                    f"| {int(row['epoch'])} | {row['model_acc']:.6f} | "
                    f"{row['model_margin']:.6f} | {row['contraction']:.6f} |\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare models on noisy prototype attraction field learning.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./吸引场验证/outputs")
    parser.add_argument("--model", choices=["linear", "mlp", "cnn", "cnn-strong", "dit-xs", "dit-s", "dit-b"], default="dit-xs")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--train-per-class", type=int, default=0)
    parser.add_argument("--val-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=0)
    parser.add_argument("--train-timestep", type=int, default=200)
    parser.add_argument("--eval-timesteps", default="200")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--val-noise-repeats", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping but still saves best validation checkpoint")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    ensure_dir(args.output_dir)
    eval_timesteps = parse_int_list(args.eval_timesteps)
    if args.train_timestep < 0 or args.train_timestep >= args.diffusion_steps:
        raise ValueError("--train-timestep is outside diffusion range")
    for timestep in eval_timesteps:
        if timestep < 0 or timestep >= args.diffusion_steps:
            raise ValueError(f"eval timestep {timestep} is outside diffusion range")

    train_loader, val_loader, test_loader, prototype_loader = build_loaders(args)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    prototypes = build_prototypes(prototype_loader, device, args.num_classes)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"device={device} model={args.model} params={params_m:.4f}M train_t={args.train_timestep} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}"
    )

    train_rows: List[Dict[str, float]] = []
    val_rows: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = os.path.join(args.output_dir, "best_checkpoint.pt")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            schedule,
            prototypes,
            device,
            args.train_timestep,
            args.grad_clip,
        )
        row = {"epoch": epoch, "train_loss": loss, "params_m": params_m}
        train_rows.append(row)
        write_csv(os.path.join(args.output_dir, "train_log.csv"), train_rows)
        val_metrics = evaluate(
            model,
            val_loader,
            schedule,
            prototypes,
            device,
            args.train_timestep,
            args.val_noise_repeats,
        )
        val_row = {"epoch": epoch}
        val_row.update(val_metrics)
        val_rows.append(val_row)
        write_csv(os.path.join(args.output_dir, "val_metrics.csv"), val_rows)

        improved = val_metrics["model_acc"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["model_acc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, epoch, args)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch} train_loss={loss:.6f} "
            f"val_acc={val_metrics['model_acc']:.4f} best_val={best_val_acc:.4f}@{best_epoch}"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch} patience={args.patience}")
            break

    load_model_checkpoint(best_path, model, device)
    test_rows = [
        evaluate(model, test_loader, schedule, prototypes, device, timestep, args.noise_repeats)
        for timestep in eval_timesteps
    ]
    write_csv(os.path.join(args.output_dir, "test_metrics.csv"), test_rows)
    save_checkpoint(os.path.join(args.output_dir, "final_checkpoint.pt"), model, optimizer, best_epoch, args)
    write_summary(os.path.join(args.output_dir, "summary.md"), args, train_rows, val_rows, test_rows, best_epoch, best_val_acc)
    best = max(test_rows, key=lambda row: row["model_acc"])
    print(
        f"wrote={args.output_dir} best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} "
        f"best_test_t={int(best['eval_timestep'])} test_acc={best['model_acc']:.4f} contraction={best['contraction']:.4f}"
    )


if __name__ == "__main__":
    main()
