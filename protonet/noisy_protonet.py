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


def squared_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x[:, None, :] - y[None]).pow(2).sum(dim=2)


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


def protonet_episode(
    encoder: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    way: int,
    normalize_embeddings: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    support_z = encoder(support_x)
    query_z = encoder(query_x)
    if normalize_embeddings:
        support_z = F.normalize(support_z, dim=1)
        query_z = F.normalize(query_z, dim=1)
    prototypes = build_prototypes(support_z, support_y, way)
    distances = squared_euclidean(query_z, prototypes)
    logits = -distances
    loss = F.cross_entropy(logits, query_y)
    preds = logits.argmax(dim=1)
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
) -> Dict[str, float]:
    encoder.eval()
    rng = random.Random(seed)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    rows = []
    with torch.no_grad():
        for _ in range(args.eval_episodes):
            support_x, support_y, query_x, query_y = indexed_dataset.sample_episode(
                args.way,
                args.shot,
                args.query,
                device,
                rng,
            )
            support_x, query_x = apply_training_noise(
                support_x,
                query_x,
                schedule,
                eval_noise_timestep,
                args.eval_noise_target,
            )
            _, metrics = protonet_episode(
                encoder,
                support_x,
                support_y,
                query_x,
                query_y,
                args.way,
                args.normalize_embeddings,
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
    if args.way > num_classes or args.way > test_num_classes:
        raise ValueError("--way cannot exceed the number of classes in train/test split")

    encoder = Conv4Encoder(in_channels, args.hidden_channels, args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    train_rng = random.Random(args.seed + 10_000 + train_noise_timestep)

    rows: List[Dict] = []
    for episode in range(1, args.train_episodes + 1):
        encoder.train()
        support_x, support_y, query_x, query_y = train_indexed.sample_episode(
            args.way,
            args.shot,
            args.query,
            device,
            train_rng,
        )
        support_x, query_x = apply_training_noise(
            support_x,
            query_x,
            schedule,
            train_noise_timestep,
            args.train_noise_target,
        )
        loss, train_metrics = protonet_episode(
            encoder,
            support_x,
            support_y,
            query_x,
            query_y,
            args.way,
            args.normalize_embeddings,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
        optimizer.step()

        if episode % args.eval_interval == 0 or episode == args.train_episodes:
            eval_clean = evaluate(
                encoder,
                test_indexed,
                args,
                device,
                seed=args.seed + 50_000 + episode,
                eval_noise_timestep=0,
            )
            eval_same = evaluate(
                encoder,
                test_indexed,
                args,
                device,
                seed=args.seed + 60_000 + episode,
                eval_noise_timestep=train_noise_timestep,
            )
            row = {
                "train_noise_timestep": train_noise_timestep,
                "episode": episode,
                "train_loss": float(loss.item()),
                "train_acc": train_metrics["acc"],
                "eval_clean_acc": eval_clean["acc"],
                "eval_clean_margin": eval_clean["margin"],
                "eval_same_noise_acc": eval_same["acc"],
                "eval_same_noise_margin": eval_same["margin"],
            }
            rows.append(row)
            print(
                f"t={train_noise_timestep} episode={episode} loss={loss.item():.4f} "
                f"train_acc={train_metrics['acc']:.4f} clean_acc={eval_clean['acc']:.4f} "
                f"same_noise_acc={eval_same['acc']:.4f}"
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoNet baseline with DDPM forward-noise training.")
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
    parser.add_argument("--train-noise-target", choices=["none", "support", "query", "both"], default="both")
    parser.add_argument("--eval-noise-target", choices=["none", "support", "query", "both"], default="both")
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
