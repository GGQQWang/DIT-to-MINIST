import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def parse_float_list(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("The bandwidth list cannot be empty.")
    return values


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def balanced_subset_indices(dataset, per_class: int, num_classes: int, seed: int) -> List[int]:
    if per_class <= 0:
        return list(range(len(dataset)))

    generator = torch.Generator().manual_seed(seed)
    labels = torch.as_tensor(dataset.targets)
    selected: List[int] = []
    for cls in range(num_classes):
        cls_indices = torch.where(labels == cls)[0]
        perm = torch.randperm(len(cls_indices), generator=generator)
        take = min(per_class, len(cls_indices))
        selected.extend(cls_indices[perm[:take]].tolist())
    return selected


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_indices = balanced_subset_indices(train_set, args.train_per_class, args.num_classes, args.seed)
    if args.test_limit > 0:
        test_indices = list(range(min(args.test_limit, len(test_set))))
        test_set = Subset(test_set, test_indices)

    train_loader = DataLoader(
        Subset(train_set, train_indices),
        batch_size=args.load_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    return images.flatten(start_dim=1).float()


@torch.no_grad()
def collect_train_by_class(
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    features_by_class: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    counts = torch.zeros(num_classes, dtype=torch.long)

    for images, labels in loader:
        features = flatten_images(images).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                features_by_class[cls].append(features[mask])
                counts[cls] += int(mask.sum().item())

    class_tensors = []
    for cls in range(num_classes):
        if not features_by_class[cls]:
            raise ValueError(f"No training samples found for class {cls}.")
        class_tensors.append(torch.cat(features_by_class[cls], dim=0).contiguous())
    return class_tensors, counts


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x.pow(2).sum(dim=1, keepdim=True)
    y_norm = y.pow(2).sum(dim=1).unsqueeze(0)
    distances = x_norm + y_norm - 2.0 * x @ y.t()
    return distances.clamp_min_(0.0)


def logaddexp_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(a, b)


@torch.no_grad()
def gaussian_log_scores(
    x: torch.Tensor,
    class_tensors: List[torch.Tensor],
    bandwidth: float,
    train_chunk_size: int,
    use_priors: bool,
    class_counts: torch.Tensor,
) -> torch.Tensor:
    scores = []
    h2 = bandwidth * bandwidth
    total_count = class_counts.sum().item()

    for cls, train_features in enumerate(class_tensors):
        log_sum = torch.full((x.shape[0],), -float("inf"), device=x.device)
        for start in range(0, train_features.shape[0], train_chunk_size):
            chunk = train_features[start : start + train_chunk_size]
            d2 = pairwise_squared_distances(x, chunk)
            chunk_log_sum = torch.logsumexp(-d2 / (2.0 * h2), dim=1)
            log_sum = logaddexp_pair(log_sum, chunk_log_sum)

        log_score = log_sum - torch.log(torch.tensor(train_features.shape[0], device=x.device, dtype=x.dtype))
        if use_priors:
            prior = class_counts[cls].item() / total_count
            log_score = log_score + torch.log(torch.tensor(prior, device=x.device, dtype=x.dtype))
        scores.append(log_score)

    return torch.stack(scores, dim=1)


@torch.no_grad()
def epanechnikov_log_scores(
    x: torch.Tensor,
    class_tensors: List[torch.Tensor],
    bandwidth: float,
    train_chunk_size: int,
    use_priors: bool,
    class_counts: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    scores = []
    h2 = bandwidth * bandwidth
    total_count = class_counts.sum().item()

    for cls, train_features in enumerate(class_tensors):
        kernel_sum = torch.zeros((x.shape[0],), device=x.device)
        for start in range(0, train_features.shape[0], train_chunk_size):
            chunk = train_features[start : start + train_chunk_size]
            d2 = pairwise_squared_distances(x, chunk)
            kernel_values = (1.0 - d2 / h2).clamp_min_(0.0)
            kernel_sum += kernel_values.sum(dim=1)

        score = torch.log(kernel_sum / train_features.shape[0] + eps)
        if use_priors:
            prior = class_counts[cls].item() / total_count
            score = score + torch.log(torch.tensor(prior, device=x.device, dtype=x.dtype))
        scores.append(score)

    return torch.stack(scores, dim=1)


@torch.no_grad()
def predict_batch(
    x: torch.Tensor,
    class_tensors: List[torch.Tensor],
    kernel: str,
    bandwidth: float,
    train_chunk_size: int,
    use_priors: bool,
    class_counts: torch.Tensor,
) -> torch.Tensor:
    if kernel == "gaussian":
        scores = gaussian_log_scores(x, class_tensors, bandwidth, train_chunk_size, use_priors, class_counts)
    elif kernel == "epanechnikov":
        scores = epanechnikov_log_scores(x, class_tensors, bandwidth, train_chunk_size, use_priors, class_counts)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    return scores.argmax(dim=1)


@torch.no_grad()
def evaluate_kernel(
    test_loader: DataLoader,
    class_tensors: List[torch.Tensor],
    kernel: str,
    bandwidth: float,
    device: torch.device,
    train_chunk_size: int,
    use_priors: bool,
    class_counts: torch.Tensor,
    num_classes: int,
) -> Tuple[float, torch.Tensor]:
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in test_loader:
        x = flatten_images(images).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = predict_batch(x, class_tensors, kernel, bandwidth, train_chunk_size, use_priors, class_counts)
        correct += int((preds == labels).sum().item())
        total += labels.numel()

        for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
            confusion[true_label.long(), pred_label.long()] += 1

    return correct / max(total, 1), confusion


def write_rows(path: str, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: str, confusion: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *range(confusion.shape[1])])
        for cls, row in enumerate(confusion.tolist()):
            writer.writerow([cls, *row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-parametric KDE kernel classifier for MNIST.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs_kernel")
    parser.add_argument("--kernels", default="gaussian,epanechnikov")
    parser.add_argument("--bandwidths", default="2,4,6,8,10,12")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-limit", type=int, default=10000)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--load-batch-size", type=int, default=1024)
    parser.add_argument("--train-chunk-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-priors", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    kernels = [item.strip() for item in args.kernels.split(",") if item.strip()]
    bandwidths = parse_float_list(args.bandwidths)

    train_loader, test_loader = build_loaders(args)
    class_tensors, class_counts = collect_train_by_class(train_loader, device, args.num_classes)
    class_counts = class_counts.to(device)

    print(
        f"device={device} train_per_class={args.train_per_class} "
        f"test_limit={args.test_limit} train_total={int(class_counts.sum().item())}"
    )

    rows = []
    best = {"acc": -1.0, "kernel": None, "bandwidth": None, "confusion": None}
    for kernel in kernels:
        for bandwidth in bandwidths:
            acc, confusion = evaluate_kernel(
                test_loader,
                class_tensors,
                kernel,
                bandwidth,
                device,
                args.train_chunk_size,
                args.use_priors,
                class_counts,
                args.num_classes,
            )
            row = {
                "kernel": kernel,
                "bandwidth": bandwidth,
                "accuracy": acc,
                "train_per_class": args.train_per_class,
                "test_limit": args.test_limit,
                "use_priors": int(args.use_priors),
            }
            rows.append(row)
            print(f"kernel={kernel} bandwidth={bandwidth:g} acc={acc:.4f}")

            if acc > best["acc"]:
                best = {"acc": acc, "kernel": kernel, "bandwidth": bandwidth, "confusion": confusion}

    write_rows(os.path.join(args.output_dir, "kernel_kde_results.csv"), rows)
    if best["confusion"] is not None:
        name = f"confusion_{best['kernel']}_h{best['bandwidth']:g}.csv"
        write_confusion(os.path.join(args.output_dir, name), best["confusion"])
    print(f"best_kernel={best['kernel']} best_bandwidth={best['bandwidth']:g} best_acc={best['acc']:.4f}")


if __name__ == "__main__":
    main()
