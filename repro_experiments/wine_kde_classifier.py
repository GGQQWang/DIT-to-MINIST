import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple

import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    wine = load_wine()
    features = wine.data.astype("float32")
    labels = wine.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    if args.test_limit > 0:
        x_test = x_test[: args.test_limit]
        y_test = y_test[: args.test_limit]

    train_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_set = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(
        train_set,
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


def as_features(samples: torch.Tensor) -> torch.Tensor:
    return samples.float()


@torch.no_grad()
def collect_train_by_class(
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    features_by_class: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    counts = torch.zeros(num_classes, dtype=torch.long)

    for samples, labels in loader:
        features = as_features(samples).to(device, non_blocking=True)
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

    for samples, labels in test_loader:
        x = as_features(samples).to(device, non_blocking=True)
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
    parser = argparse.ArgumentParser(description="Non-parametric KDE kernel classifier for the Wine dataset.")
    parser.add_argument("--output-dir", default="./outputs_wine_kernel")
    parser.add_argument("--kernels", default="gaussian,epanechnikov")
    parser.add_argument("--bandwidths", default="0.2,0.4,0.6,0.8,1.0,1.5,2.0")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=3)
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
        f"device={device} test_size={args.test_size} "
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
                "test_size": args.test_size,
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
