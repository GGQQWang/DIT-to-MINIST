import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


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


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device):
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (x0.dim() - 1)
        sqrt_ab = self.sqrt_alpha_bars[t].view(shape)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(shape)
        return sqrt_ab * x0 + sqrt_omab * noise


class IndexedDataset:
    def __init__(self, dataset, num_classes: int):
        self.dataset = dataset
        self.num_classes = num_classes
        self.class_to_indices: Dict[int, List[int]] = {cls: [] for cls in range(num_classes)}
        for idx in range(len(dataset)):
            label = self._label_at(idx)
            if 0 <= label < num_classes:
                self.class_to_indices[label].append(idx)
        missing = [cls for cls, indices in self.class_to_indices.items() if not indices]
        if missing:
            raise ValueError(f"dataset has no samples for classes: {missing}")

    def _label_at(self, idx: int) -> int:
        if hasattr(self.dataset, "targets"):
            target = self.dataset.targets[idx]
            return int(target.item() if hasattr(target, "item") else target)
        if hasattr(self.dataset, "_flat_character_images"):
            return int(self.dataset._flat_character_images[idx][1])
        _, label = self.dataset[idx]
        return int(label)

    def sample_episode(
        self,
        way: int,
        shot: int,
        query: int,
        device: torch.device,
        rng: random.Random,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        classes = rng.sample(range(self.num_classes), way)
        support_images = []
        query_images = []
        support_labels = []
        query_labels = []

        for episode_label, cls in enumerate(classes):
            indices = self.class_to_indices[cls]
            needed = shot + query
            if len(indices) >= needed:
                chosen = rng.sample(indices, needed)
            else:
                chosen = [rng.choice(indices) for _ in range(needed)]
            support_idx = chosen[:shot]
            query_idx = chosen[shot:]
            for idx in support_idx:
                image, _ = self.dataset[idx]
                support_images.append(image)
                support_labels.append(episode_label)
            for idx in query_idx:
                image, _ = self.dataset[idx]
                query_images.append(image)
                query_labels.append(episode_label)

        return (
            torch.stack(support_images).to(device),
            torch.tensor(support_labels, device=device, dtype=torch.long),
            torch.stack(query_images).to(device),
            torch.tensor(query_labels, device=device, dtype=torch.long),
        )


class HFMiniImageNetDataset:
    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform
        raw_labels = [int(label) for label in hf_split["label"]]
        unique_labels = sorted(set(raw_labels))
        self.label_to_contiguous = {label: idx for idx, label in enumerate(unique_labels)}
        self.targets = [self.label_to_contiguous[label] for label in raw_labels]
        self.classes = [str(label) for label in unique_labels]

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int):
        item = self.hf_split[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv4Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
        )
        self.proj = nn.Linear(hidden_channels, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.mean(dim=(2, 3))
        return self.proj(h)


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor, reduction: str) -> torch.Tensor:
    distances = (x[:, None, :] - y[None]).pow(2)
    if reduction == "sum":
        return distances.sum(dim=2)
    if reduction == "mean":
        return distances.mean(dim=2)
    raise ValueError(f"unsupported distance reduction: {reduction}")


def normalize_features(z: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return z
    if mode == "layernorm":
        return F.layer_norm(z, z.shape[1:])
    if mode == "l2":
        return F.normalize(z, dim=1)
    raise ValueError(f"unsupported feature normalization: {mode}")


def build_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, way: int) -> torch.Tensor:
    prototypes = []
    for cls in range(way):
        mask = labels == cls
        if not mask.any():
            raise ValueError(f"support set missing episode class {cls}")
        prototypes.append(embeddings[mask].mean(dim=0))
    return torch.stack(prototypes, dim=0)


def apply_training_noise(
    support_x: torch.Tensor,
    query_x: torch.Tensor,
    schedule: DiffusionSchedule,
    noise_timestep: int,
    noise_target: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_timestep == 0 or noise_target == "none":
        return support_x, query_x
    if noise_target not in {"support", "query", "both"}:
        raise ValueError(f"unsupported noise_target: {noise_target}")
    if noise_target in {"support", "both"}:
        t = torch.full((support_x.shape[0],), noise_timestep, device=support_x.device, dtype=torch.long)
        support_x = schedule.q_sample(support_x, t, torch.randn_like(support_x))
    if noise_target in {"query", "both"}:
        t = torch.full((query_x.shape[0],), noise_timestep, device=query_x.device, dtype=torch.long)
        query_x = schedule.q_sample(query_x, t, torch.randn_like(query_x))
    return support_x.clamp(-1, 1), query_x.clamp(-1, 1)


def apply_feature_noise(
    query_z: torch.Tensor,
    prototypes: torch.Tensor,
    schedule: DiffusionSchedule,
    noise_timestep: int,
    noise_target: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_timestep == 0 or noise_target == "none":
        return query_z, prototypes
    if noise_target != "query":
        raise ValueError("feature-space DDPM currently supports --*-noise-target query or none only")

    t = torch.full((query_z.shape[0],), noise_timestep, device=query_z.device, dtype=torch.long)
    query_z_t = schedule.q_sample(query_z, t, torch.randn_like(query_z))
    prototypes_t = schedule.sqrt_alpha_bars[noise_timestep] * prototypes
    return query_z_t, prototypes_t


def distance_geometry_loss(
    distances: torch.Tensor,
    query_y: torch.Tensor,
    prototypes: torch.Tensor,
    margin: float,
    proto_sep_margin: float,
    lambda_margin: float,
    lambda_positive: float,
    lambda_proto_sep: float,
    lambda_var: float,
    clean_embeddings: torch.Tensor,
    var_gamma: float,
    contrast_temperature: float,
    distance_reduction: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    true_dist = distances[torch.arange(query_y.numel(), device=query_y.device), query_y]
    wrong = distances.clone()
    wrong[torch.arange(query_y.numel(), device=query_y.device), query_y] = float("inf")
    nearest_wrong = wrong.min(dim=1).values

    positive_loss = true_dist.mean()
    negative_distances = distances.masked_fill(
        F.one_hot(query_y, num_classes=distances.shape[1]).bool(),
        float("inf"),
    )
    contrast_logits = (margin + true_dist[:, None] - negative_distances) / contrast_temperature
    margin_loss = F.softplus(contrast_logits[torch.isfinite(contrast_logits)]).mean()

    proto_pair_dist = pairwise_squared_distance(prototypes, prototypes, distance_reduction)
    pair_mask = torch.triu(torch.ones_like(proto_pair_dist, dtype=torch.bool), diagonal=1)
    proto_sep_loss = F.softplus((proto_sep_margin - proto_pair_dist[pair_mask]) / contrast_temperature).mean()

    feature_std = clean_embeddings.std(dim=0, unbiased=False)
    variance_loss = F.relu(var_gamma - feature_std).mean()

    loss = (
        lambda_positive * positive_loss
        + lambda_margin * margin_loss
        + lambda_proto_sep * proto_sep_loss
        + lambda_var * variance_loss
    )
    parts = {
        "positive_loss": float(positive_loss.item()),
        "contrast_loss": float(margin_loss.item()),
        "proto_sep_loss": float(proto_sep_loss.item()),
        "variance_loss": float(variance_loss.item()),
    }
    return loss, parts


def protonet_episode(
    encoder: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    way: int,
    args: argparse.Namespace,
    schedule: DiffusionSchedule,
    noise_timestep: int,
    noise_target: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if args.noise_space == "image":
        support_x, query_x = apply_training_noise(support_x, query_x, schedule, noise_timestep, noise_target)

    support_z = normalize_features(encoder(support_x), args.feature_normalization)
    query_z = normalize_features(encoder(query_x), args.feature_normalization)
    if args.normalize_embeddings:
        support_z = F.normalize(support_z, dim=1)
        query_z = F.normalize(query_z, dim=1)
    prototypes = build_prototypes(support_z, support_y, way)

    clean_embeddings = torch.cat([support_z, query_z], dim=0)
    if args.noise_space == "feature":
        query_z, prototypes_for_query = apply_feature_noise(query_z, prototypes, schedule, noise_timestep, noise_target)
    else:
        prototypes_for_query = prototypes

    distances = pairwise_squared_distance(query_z, prototypes_for_query, args.distance_reduction)

    if args.loss_type == "ce":
        logits = -distances
        loss = F.cross_entropy(logits, query_y)
        loss_parts = {
            "positive_loss": 0.0,
            "contrast_loss": 0.0,
            "proto_sep_loss": 0.0,
            "variance_loss": 0.0,
        }
    elif args.loss_type == "distance":
        loss, loss_parts = distance_geometry_loss(
            distances=distances,
            query_y=query_y,
            prototypes=prototypes,
            margin=args.distance_margin,
            proto_sep_margin=args.proto_sep_margin,
            lambda_margin=args.lambda_margin,
            lambda_positive=args.lambda_positive,
            lambda_proto_sep=args.lambda_proto_sep,
            lambda_var=args.lambda_var,
            clean_embeddings=clean_embeddings,
            var_gamma=args.var_gamma,
            contrast_temperature=args.contrast_temperature,
            distance_reduction=args.distance_reduction,
        )
    else:
        raise ValueError(f"unsupported loss type: {args.loss_type}")

    preds = distances.argmin(dim=1)
    true_dist = distances[torch.arange(query_y.numel(), device=query_y.device), query_y]
    wrong = distances.clone()
    wrong[torch.arange(query_y.numel(), device=query_y.device), query_y] = float("inf")
    nearest_wrong = wrong.min(dim=1).values
    metrics = {
        "acc": float((preds == query_y).float().mean().item()),
        "margin": float((nearest_wrong - true_dist).mean().item()),
        "true_distance": float(true_dist.mean().item()),
        "nearest_wrong_distance": float(nearest_wrong.mean().item()),
    }
    metrics.update(loss_parts)
    return loss, metrics


def mean_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def write_csv(path: str, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args: argparse.Namespace, train: bool):
    if args.dataset in {"mnist", "fashion-mnist"}:
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        cls = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
        return cls(args.data_dir, train=train, download=True, transform=transform), 1, 10

    if args.dataset == "omniglot":
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        background = train
        dataset = datasets.Omniglot(args.data_dir, background=background, download=True, transform=transform)
        return dataset, 1, len(dataset._characters)

    if args.dataset == "miniimagenet":
        transform = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        split = "train" if train else args.eval_split
        root_candidates = [
            os.path.join(args.data_dir, "miniImageNet", split),
            os.path.join(args.data_dir, "miniimagenet", split),
            os.path.join(args.data_dir, split),
        ]
        split_root = next((path for path in root_candidates if os.path.isdir(path)), None)
        if split_root is None:
            candidates = "\n".join(f"  - {path}" for path in root_candidates)
            raise FileNotFoundError(
                "miniImageNet split directory was not found. Expected one of:\n"
                f"{candidates}\n"
                "Use ImageFolder layout: split/class_name/image.jpg"
            )
        print(f"loading dataset=miniimagenet split={split} root={split_root}", flush=True)
        dataset = datasets.ImageFolder(split_root, transform=transform)
        print(
            f"loaded dataset=miniimagenet split={split} samples={len(dataset)} "
            f"classes={len(dataset.classes)}",
            flush=True,
        )
        return dataset, 3, len(dataset.classes)

    if args.dataset == "miniimagenet-hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install Hugging Face datasets first: pip install datasets") from exc

        transform = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        split = "train" if train else args.eval_split
        if split == "val":
            split = "validation"
        print(f"loading dataset=miniimagenet-hf id={args.hf_dataset_id} split={split}", flush=True)
        hf_split = load_dataset(args.hf_dataset_id, split=split)
        dataset = HFMiniImageNetDataset(hf_split, transform)
        print(
            f"loaded dataset=miniimagenet-hf split={split} samples={len(dataset)} "
            f"classes={len(dataset.classes)}",
            flush=True,
        )
        return dataset, 3, len(dataset.classes)

    raise ValueError(f"unsupported dataset: {args.dataset}")


def evaluate(
    encoder: nn.Module,
    indexed_dataset: IndexedDataset,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    eval_noise_timestep: int,
    eval_way: int,
) -> Dict[str, float]:
    encoder.eval()
    rng = random.Random(seed)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    rows = []
    with torch.no_grad():
        for _ in range(args.eval_episodes):
            support_x, support_y, query_x, query_y = indexed_dataset.sample_episode(
                eval_way,
                args.shot,
                args.query,
                device,
                rng,
            )
            _, metrics = protonet_episode(
                encoder,
                support_x,
                support_y,
                query_x,
                query_y,
                eval_way,
                args,
                schedule,
                eval_noise_timestep,
                args.eval_noise_target,
            )
            rows.append(metrics)
    return mean_metrics(rows)


def train_one_setting(args: argparse.Namespace, train_noise_timestep: int) -> List[Dict]:
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_dataset, in_channels, num_classes = build_dataset(args, train=True)
    test_dataset, _, test_num_classes = build_dataset(args, train=False)
    train_indexed = IndexedDataset(train_dataset, num_classes)
    test_indexed = IndexedDataset(test_dataset, test_num_classes)
    train_way = args.train_way if args.train_way > 0 else args.way
    eval_way = args.eval_way if args.eval_way > 0 else args.way
    if train_way > num_classes:
        raise ValueError("--train-way cannot exceed the number of training classes")
    if eval_way > test_num_classes:
        raise ValueError("--eval-way cannot exceed the number of evaluation classes")

    encoder = Conv4Encoder(in_channels, args.hidden_channels, args.embedding_dim).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            encoder.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"unsupported optimizer: {args.optimizer}")
    scheduler = None
    if args.lr_step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    train_rng = random.Random(args.seed + 10_000 + train_noise_timestep)

    rows: List[Dict] = []
    for episode in range(1, args.train_episodes + 1):
        encoder.train()
        support_x, support_y, query_x, query_y = train_indexed.sample_episode(
            train_way,
            args.shot,
            args.query,
            device,
            train_rng,
        )
        loss, train_metrics = protonet_episode(
            encoder,
            support_x,
            support_y,
            query_x,
            query_y,
            train_way,
            args,
            schedule,
            train_noise_timestep,
            args.train_noise_target,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if episode % args.eval_interval == 0 or episode == args.train_episodes:
            eval_clean = evaluate(
                encoder,
                test_indexed,
                args,
                device,
                seed=args.seed + 50_000 + episode,
                eval_noise_timestep=0,
                eval_way=eval_way,
            )
            eval_same = evaluate(
                encoder,
                test_indexed,
                args,
                device,
                seed=args.seed + 60_000 + episode,
                eval_noise_timestep=train_noise_timestep,
                eval_way=eval_way,
            )
            row = {
                "train_noise_timestep": train_noise_timestep,
                "episode": episode,
                "train_way": train_way,
                "eval_way": eval_way,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": float(loss.item()),
                "train_acc": train_metrics["acc"],
                "eval_clean_acc": eval_clean["acc"],
                "eval_clean_margin": eval_clean["margin"],
                "eval_same_noise_acc": eval_same["acc"],
                "eval_same_noise_margin": eval_same["margin"],
                "positive_loss": train_metrics["positive_loss"],
                "contrast_loss": train_metrics["contrast_loss"],
                "proto_sep_loss": train_metrics["proto_sep_loss"],
                "variance_loss": train_metrics["variance_loss"],
            }
            rows.append(row)
            print(
                f"t={train_noise_timestep} episode={episode} loss={loss.item():.4f} "
                f"train_acc={train_metrics['acc']:.4f} clean_acc={eval_clean['acc']:.4f} "
                f"same_noise_acc={eval_same['acc']:.4f}"
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoNet with feature-space DDPM noise and geometric distance loss.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./protonet/outputs/noisy_protonet")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion-mnist", "omniglot", "miniimagenet", "miniimagenet-hf"],
        default="fashion-mnist",
    )
    parser.add_argument("--hf-dataset-id", default="GATE-engine/mini_imagenet")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test", help="Evaluation split for ImageFolder datasets.")
    parser.add_argument("--image-size", type=int, default=84, help="Image size for miniImageNet.")
    parser.add_argument("--train-noise-timesteps", default="0,100,200,250,300,400,500")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-space", choices=["feature", "image"], default="feature")
    parser.add_argument("--train-noise-target", choices=["none", "support", "query", "both"], default="query")
    parser.add_argument("--eval-noise-target", choices=["none", "support", "query", "both"], default="query")
    parser.add_argument("--loss-type", choices=["distance", "ce"], default="distance")
    parser.add_argument("--feature-normalization", choices=["none", "layernorm", "l2"], default="layernorm")
    parser.add_argument("--distance-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--distance-margin", type=float, default=1.0)
    parser.add_argument("--lambda-margin", type=float, default=1.0)
    parser.add_argument("--lambda-positive", type=float, default=0.0)
    parser.add_argument("--proto-sep-margin", type=float, default=1.0)
    parser.add_argument("--lambda-proto-sep", type=float, default=0.1)
    parser.add_argument("--lambda-var", type=float, default=0.0)
    parser.add_argument("--var-gamma", type=float, default=1.0)
    parser.add_argument("--contrast-temperature", type=float, default=0.2)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--train-way", type=int, default=0, help="0 means use --way.")
    parser.add_argument("--eval-way", type=int, default=0, help="0 means use --way.")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-step-size", type=int, default=0, help="0 disables StepLR. Unit: episodes.")
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    all_rows: List[Dict] = []
    for train_noise_timestep in parse_int_list(args.train_noise_timesteps):
        rows = train_one_setting(args, train_noise_timestep)
        all_rows.extend(rows)
        write_csv(os.path.join(args.output_dir, "train_eval_log.csv"), all_rows)

    final_rows = []
    for timestep in parse_int_list(args.train_noise_timesteps):
        matches = [row for row in all_rows if int(row["train_noise_timestep"]) == timestep]
        if matches:
            final_rows.append(matches[-1])
    write_csv(os.path.join(args.output_dir, "final_results.csv"), final_rows)
    print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
