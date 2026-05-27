import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DiTConfig:
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
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class FeatureDenoisingDiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, cfg.hidden_size), requires_grad=False)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.final_layer = FinalLayer(cfg.hidden_size)
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(x) + self.pos_embed

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t)
        x = x_t
        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)


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


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device):
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (x_t - sqrt_omab * eps_pred) / sqrt_ab.clamp_min(1e-8)


def build_model(args: argparse.Namespace) -> FeatureDenoisingDiT:
    size_cfg = DIT_CONFIGS[args.dit_size]
    cfg = DiTConfig(
        depth=size_cfg["depth"],
        hidden_size=size_cfg["hidden_size"],
        num_heads=size_cfg["num_heads"],
        dropout=args.dropout,
        num_diffusion_steps=args.diffusion_steps,
    )
    return FeatureDenoisingDiT(cfg)


def get_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
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
    model: FeatureDenoisingDiT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    epoch: int,
    log_interval: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for step, (images, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        x0 = model.encode(images)
        noise = torch.randn_like(x0)
        t = torch.randint(0, schedule.timesteps, (images.shape[0],), device=device)
        x_t = schedule.q_sample(x0, t, noise)
        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)

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
    model: FeatureDenoisingDiT,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    image_sums = torch.zeros(model.cfg.num_classes, model.cfg.in_channels, model.cfg.image_size, model.cfg.image_size, device=device)
    feature_sums = torch.zeros(
        model.cfg.num_classes,
        model.patch_embed.num_patches,
        model.cfg.hidden_size,
        device=device,
    )
    counts = torch.zeros(model.cfg.num_classes, device=device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = model.encode(images)
        for cls in range(model.cfg.num_classes):
            mask = labels == cls
            if mask.any():
                image_sums[cls] += images[mask].sum(dim=0)
                feature_sums[cls] += features[mask].sum(dim=0)
                counts[cls] += mask.sum()

    counts = counts.clamp_min(1).view(-1, 1, 1, 1)
    image_prototypes = image_sums / counts
    feature_prototypes = feature_sums / counts.view(-1, 1, 1)
    return image_prototypes, feature_prototypes


def nearest_image_proto(images: torch.Tensor, image_prototypes: torch.Tensor) -> torch.Tensor:
    errors = F.mse_loss(
        images[:, None],
        image_prototypes[None],
        reduction="none",
    ).mean(dim=(2, 3, 4))
    return errors.argmin(dim=1)


def nearest_feature_proto(features: torch.Tensor, feature_prototypes: torch.Tensor) -> torch.Tensor:
    errors = F.mse_loss(
        features[:, None],
        feature_prototypes[None],
        reduction="none",
    ).mean(dim=(2, 3))
    return errors.argmin(dim=1)


@torch.no_grad()
def denoise_features(
    model: FeatureDenoisingDiT,
    x0: torch.Tensor,
    schedule: DiffusionSchedule,
    device: torch.device,
    eval_timesteps: Iterable[int],
    noise_repeats: int,
) -> torch.Tensor:
    batch_size = x0.shape[0]
    x0_pred_sum = torch.zeros_like(x0)
    count = 0

    for timestep in eval_timesteps:
        t = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
        for _ in range(noise_repeats):
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)
            eps_pred = model(x_t, t)
            x0_pred_sum += schedule.predict_x0(x_t, t, eps_pred)
            count += 1

    return x0_pred_sum / max(count, 1)


@torch.no_grad()
def evaluate_prototypes(
    model: FeatureDenoisingDiT,
    loader: DataLoader,
    image_prototypes: torch.Tensor,
    feature_prototypes: torch.Tensor,
    schedule: DiffusionSchedule,
    device: torch.device,
    eval_timesteps: Iterable[int],
    noise_repeats: int,
) -> Dict[str, float]:
    model.eval()
    correct = {
        "image_clean": 0,
        "image_noisy": 0,
        "feature_clean": 0,
        "feature_denoised": 0,
    }
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        x0 = model.encode(images)
        noisy_images = denoise_image_baseline(images, schedule, device, eval_timesteps, noise_repeats)
        x0_pred = denoise_features(model, x0, schedule, device, eval_timesteps, noise_repeats)

        preds = {
            "image_clean": nearest_image_proto(images, image_prototypes),
            "image_noisy": nearest_image_proto(noisy_images, image_prototypes),
            "feature_clean": nearest_feature_proto(x0, feature_prototypes),
            "feature_denoised": nearest_feature_proto(x0_pred, feature_prototypes),
        }
        for name, pred in preds.items():
            correct[name] += (pred == labels).sum().item()
        total += labels.numel()

    return {name: value / max(total, 1) for name, value in correct.items()}


@torch.no_grad()
def denoise_image_baseline(
    images: torch.Tensor,
    schedule: DiffusionSchedule,
    device: torch.device,
    eval_timesteps: Iterable[int],
    noise_repeats: int,
) -> torch.Tensor:
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
    model: FeatureDenoisingDiT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
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


def load_checkpoint(path: str, model: FeatureDenoisingDiT, device: torch.device, optimizer=None) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


def parse_eval_timesteps(value: str, diffusion_steps: int) -> Tuple[int, ...]:
    steps = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not steps:
        raise ValueError("--eval-timesteps cannot be empty")
    for step in steps:
        if step < 0 or step >= diffusion_steps:
            raise ValueError(f"eval timestep {step} is outside [0, {diffusion_steps - 1}]")
    return steps


def parse_timestep_sweep(value: str, diffusion_steps: int) -> Tuple[Tuple[int, ...], ...]:
    groups = []
    for group in value.split(";"):
        group = group.strip()
        if group:
            groups.append(parse_eval_timesteps(group, diffusion_steps))
    if not groups:
        raise ValueError("--timestep-sweep cannot be empty")
    return tuple(groups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unconditional DiT feature denoising with prototype matching for MNIST.")
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
    parser.add_argument("--eval-timesteps", default="50,100,150")
    parser.add_argument(
        "--timestep-sweep",
        default="0;10;20;50;100;150;200;300;500;20,50,100;50,100,150;100,150,200",
    )
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_loader, test_loader = get_loaders(args)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    eval_timesteps = parse_eval_timesteps(args.eval_timesteps, args.diffusion_steps)

    print(f"device={device} dit={args.dit_size} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if args.mode == "eval":
        load_checkpoint(args.checkpoint, model, device)
        image_prototypes, feature_prototypes = build_prototypes(model, train_loader, device)
        metrics = evaluate_prototypes(
            model,
            test_loader,
            image_prototypes,
            feature_prototypes,
            schedule,
            device,
            eval_timesteps,
            args.noise_repeats,
        )
        print(format_metrics(metrics))
        return

    if args.mode == "sweep":
        load_checkpoint(args.checkpoint, model, device)
        image_prototypes, feature_prototypes = build_prototypes(model, train_loader, device)
        for timestep_group in parse_timestep_sweep(args.timestep_sweep, args.diffusion_steps):
            metrics = evaluate_prototypes(
                model,
                test_loader,
                image_prototypes,
                feature_prototypes,
                schedule,
                device,
                timestep_group,
                args.noise_repeats,
            )
            print(f"eval_timesteps={format_timesteps(timestep_group)} {format_metrics(metrics)}")
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
        )
        image_prototypes, feature_prototypes = build_prototypes(model, train_loader, device)
        metrics = evaluate_prototypes(
            model,
            test_loader,
            image_prototypes,
            feature_prototypes,
            schedule,
            device,
            eval_timesteps,
            args.noise_repeats,
        )
        save_checkpoint(args.checkpoint, model, optimizer, epoch, args)
        print(f"epoch={epoch} train_loss={loss:.6f} {format_metrics(metrics)} saved={args.checkpoint}")


if __name__ == "__main__":
    main()
