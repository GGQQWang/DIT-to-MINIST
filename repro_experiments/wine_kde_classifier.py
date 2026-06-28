import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple

import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Wine 数据集 KDE 分类实验。
# 核心思想：不训练神经网络，而是保存每一类训练样本，
# 测试时估计样本在各类别条件分布下的核密度，选择密度最高的类别。


def parse_float_list(value: str) -> List[float]:
    """将命令行中的带宽字符串解析为浮点数列表。"""
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("The bandwidth list cannot be empty.")
    return values


def resolve_device(name: str) -> torch.device:
    """根据命令行参数选择 CPU、CUDA 或 Apple MPS。"""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """加载 Wine 数据集，并完成分层划分和特征标准化。"""
    wine = load_wine()
    features = wine.data.astype("float32")
    labels = wine.target.astype("int64")

    # Wine 样本量较小，使用 stratify 保证训练集和测试集类别比例一致。
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    # KDE 使用欧氏距离，必须先标准化特征，避免大尺度特征主导距离。
    # 注意：标准化器只在训练集上 fit，避免测试集信息泄漏。
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
    """Wine 样本本身就是 13 维特征向量，不需要图像展平。"""
    return samples.float()


@torch.no_grad()
def collect_train_by_class(
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """按类别收集训练特征，供 KDE 逐类估计条件密度。"""
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
    """计算两个样本矩阵之间的两两欧氏距离平方。"""
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
    """使用 Gaussian 核计算每个测试样本在每个类别下的对数密度分数。"""
    scores = []
    h2 = bandwidth * bandwidth
    total_count = class_counts.sum().item()

    for cls, train_features in enumerate(class_tensors):
        log_sum = torch.full((x.shape[0],), -float("inf"), device=x.device)
        for start in range(0, train_features.shape[0], train_chunk_size):
            chunk = train_features[start : start + train_chunk_size]
            d2 = pairwise_squared_distances(x, chunk)
            # Gaussian 核：距离越近贡献越大，距离越远贡献按指数衰减。
            chunk_log_sum = torch.logsumexp(-d2 / (2.0 * h2), dim=1)
            log_sum = logaddexp_pair(log_sum, chunk_log_sum)

        # 对同一类别的训练样本求平均，得到该类别的 KDE 密度估计。
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
    """使用 Epanechnikov 核计算每个测试样本在每个类别下的对数密度分数。"""
    scores = []
    h2 = bandwidth * bandwidth
    total_count = class_counts.sum().item()

    for cls, train_features in enumerate(class_tensors):
        kernel_sum = torch.zeros((x.shape[0],), device=x.device)
        for start in range(0, train_features.shape[0], train_chunk_size):
            chunk = train_features[start : start + train_chunk_size]
            d2 = pairwise_squared_distances(x, chunk)
            # Epanechnikov 核：带宽范围内有贡献，范围外贡献为 0。
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
    """对一个 batch 的测试样本预测类别。"""
    if kernel == "gaussian":
        scores = gaussian_log_scores(x, class_tensors, bandwidth, train_chunk_size, use_priors, class_counts)
    elif kernel == "epanechnikov":
        scores = epanechnikov_log_scores(x, class_tensors, bandwidth, train_chunk_size, use_priors, class_counts)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    # 选择条件核密度估计值最大的类别作为预测结果。
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
    """评估指定核函数和带宽下的分类准确率与混淆矩阵。"""
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
    """保存不同核函数和带宽的实验结果。"""
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: str, confusion: torch.Tensor) -> None:
    """保存最佳配置对应的混淆矩阵。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *range(confusion.shape[1])])
        for cls, row in enumerate(confusion.tolist()):
            writer.writerow([cls, *row])


def parse_args() -> argparse.Namespace:
    """定义 Wine KDE 实验的可调参数。"""
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
    """主入口：加载数据、按类别收集训练样本、搜索核函数和带宽。"""
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
