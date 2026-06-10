import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def parse_int_list(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("list cannot be empty")
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


def balanced_subset_indices(dataset, per_class: int, num_classes: int, seed: int) -> List[int]:
    if per_class <= 0:
        return list(range(len(dataset)))

    generator = torch.Generator().manual_seed(seed)
    labels = torch.as_tensor(dataset.targets)
    indices: List[int] = []
    for cls in range(num_classes):
        cls_indices = torch.where(labels == cls)[0]
        perm = torch.randperm(cls_indices.numel(), generator=generator)
        indices.extend(cls_indices[perm[: min(per_class, cls_indices.numel())]].tolist())
    return indices


def build_loader(args: argparse.Namespace, train: bool) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(args.data_dir, train=train, download=True, transform=transform)
    per_class = args.train_per_class if train else args.eval_per_class
    indices = balanced_subset_indices(dataset, per_class, args.num_classes, args.seed + (0 if train else 1000))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def collect_features(loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for images, batch_labels in loader:
        features.append(images.flatten(start_dim=1).to(device, non_blocking=True).float())
        labels.append(batch_labels.to(device, non_blocking=True))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def diffusion_coefficients(timestep: int, diffusion_steps: int, device: torch.device) -> Tuple[float, float]:
    betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar = alpha_bars[timestep].item()
    return math.sqrt(alpha_bar), math.sqrt(1.0 - alpha_bar)


def noisy_features(
    x: torch.Tensor,
    timestep: int,
    diffusion_steps: int,
    seed: int,
) -> torch.Tensor:
    sqrt_ab, sqrt_omab = diffusion_coefficients(timestep, diffusion_steps, x.device)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return sqrt_ab * x + sqrt_omab * noise


def build_clean_centers(x: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    centers = torch.zeros(num_classes, x.shape[1], device=x.device)
    for cls in range(num_classes):
        mask = labels == cls
        if not mask.any():
            raise ValueError(f"class {cls} has no samples")
        centers[cls] = x[mask].mean(dim=0)
    return centers


def class_pca_from_noisy_samples(
    clean_x: torch.Tensor,
    labels: torch.Tensor,
    clean_centers: torch.Tensor,
    timestep: int,
    diffusion_steps: int,
    max_dim: int,
    train_noise_repeats: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sqrt_ab, _ = diffusion_coefficients(timestep, diffusion_steps, clean_x.device)
    centers_t = sqrt_ab * clean_centers
    num_classes = clean_centers.shape[0]
    feature_dim = clean_x.shape[1]
    bases = torch.zeros(num_classes, max_dim, feature_dim, device=clean_x.device)
    explained = torch.zeros(num_classes, max_dim, device=clean_x.device)

    for cls in range(num_classes):
        cls_clean = clean_x[labels == cls]
        centered_parts = []
        for repeat in range(train_noise_repeats):
            cls_noisy = noisy_features(
                cls_clean,
                timestep=timestep,
                diffusion_steps=diffusion_steps,
                seed=seed + 100_003 * repeat + 1009 * cls + timestep,
            )
            centered_parts.append(cls_noisy - centers_t[cls])
        centered = torch.cat(centered_parts, dim=0)
        cov = centered.t().matmul(centered) / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.flip(0).clamp_min(0)
        eigvecs = eigvecs.flip(1).t().contiguous()
        take = min(max_dim, eigvecs.shape[0])
        bases[cls, :take] = eigvecs[:take]
        total = eigvals.sum().clamp_min(1e-12)
        explained[cls, :take] = eigvals[:take] / total
    return centers_t, bases, explained


def projection_residuals(x: torch.Tensor, centers: torch.Tensor, bases: torch.Tensor, dim: int) -> torch.Tensor:
    diff = x[:, None, :] - centers[None, :, :]
    total = diff.pow(2).sum(dim=2)
    if dim <= 0:
        return total
    selected = bases[:, :dim, :]
    coeff = torch.einsum("bcd,ckd->bck", diff, selected)
    projected = coeff.pow(2).sum(dim=2)
    return (total - projected).clamp_min(0)


def evaluate_residual_classifier(
    clean_eval_x: torch.Tensor,
    eval_labels: torch.Tensor,
    centers_t: torch.Tensor,
    bases: torch.Tensor,
    dim: int,
    timestep: int,
    diffusion_steps: int,
    seed: int,
    batch_size: int,
) -> Dict[str, float]:
    eval_x_t = noisy_features(clean_eval_x, timestep, diffusion_steps, seed)
    correct = 0
    total = 0
    margin_sum = 0.0
    true_residual_sum = 0.0
    wrong_residual_sum = 0.0

    for start in range(0, eval_x_t.shape[0], batch_size):
        batch = eval_x_t[start : start + batch_size]
        labels = eval_labels[start : start + batch_size]
        residuals = projection_residuals(batch, centers_t, bases, dim)
        preds = residuals.argmin(dim=1)
        true_residual = residuals[torch.arange(labels.shape[0], device=labels.device), labels]
        wrong_residuals = residuals.clone()
        wrong_residuals[torch.arange(labels.shape[0], device=labels.device), labels] = float("inf")
        nearest_wrong = wrong_residuals.min(dim=1).values
        margin = nearest_wrong - true_residual

        correct += int((preds == labels).sum().item())
        total += labels.numel()
        margin_sum += float(margin.sum().item())
        true_residual_sum += float(true_residual.sum().item())
        wrong_residual_sum += float(nearest_wrong.sum().item())

    return {
        "acc": correct / max(total, 1),
        "margin_mean": margin_sum / max(total, 1),
        "true_residual_mean": true_residual_sum / max(total, 1),
        "nearest_wrong_residual_mean": wrong_residual_sum / max(total, 1),
    }


def subspace_similarity(bases_a: torch.Tensor, bases_b: torch.Tensor, dim: int) -> float:
    if dim <= 0:
        return 1.0
    sims = []
    for cls in range(bases_a.shape[0]):
        gram = bases_a[cls, :dim].matmul(bases_b[cls, :dim].t())
        sims.append(gram.pow(2).sum() / dim)
    return float(torch.stack(sims).mean().item())


def write_csv(path: str, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["timestep"]), int(row["dim"]))].append(row)

    summary_rows = []
    for (timestep, dim), group in sorted(groups.items()):
        acc = torch.tensor([float(row["acc"]) for row in group])
        margin = torch.tensor([float(row["margin_mean"]) for row in group])
        true_residual = torch.tensor([float(row["true_residual_mean"]) for row in group])
        wrong_residual = torch.tensor([float(row["nearest_wrong_residual_mean"]) for row in group])
        summary_rows.append(
            {
                "timestep": timestep,
                "dim": dim,
                "acc_mean": float(acc.mean().item()),
                "acc_std": float(acc.std(unbiased=False).item()),
                "margin_mean": float(margin.mean().item()),
                "margin_std": float(margin.std(unbiased=False).item()),
                "true_residual_mean": float(true_residual.mean().item()),
                "nearest_wrong_residual_mean": float(wrong_residual.mean().item()),
                "runs": len(group),
            }
        )
    return summary_rows


def write_markdown_summary(path: str, args: argparse.Namespace, summary_rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    best = max(summary_rows, key=lambda row: row["acc_mean"])
    with open(path, "w") as f:
        f.write("# Supervised Noisy Subspace Stage 1 Summary\n\n")
        f.write("This experiment does not train a model. It verifies whether supervised class centers plus noisy samples form useful class subspaces.\n\n")
        f.write("## Config\n\n")
        f.write(f"- timesteps: {args.timesteps}\n")
        f.write(f"- dims: {args.dims}\n")
        f.write(f"- train_per_class: {args.train_per_class}\n")
        f.write(f"- eval_per_class: {args.eval_per_class}\n")
        f.write(f"- train_noise_repeats: {args.train_noise_repeats}\n")
        f.write(f"- eval_noise_repeats: {args.eval_noise_repeats}\n\n")
        f.write("## Best Result\n\n")
        f.write(f"- timestep: {best['timestep']}\n")
        f.write(f"- dim: {best['dim']}\n")
        f.write(f"- acc_mean: {best['acc_mean']:.6f}\n")
        f.write(f"- acc_std: {best['acc_std']:.6f}\n")
        f.write(f"- margin_mean: {best['margin_mean']:.6f}\n\n")
        f.write("## Metrics Table\n\n")
        f.write("| timestep | dim | acc_mean | acc_std | margin_mean | true_residual | nearest_wrong_residual |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            f.write(
                f"| {row['timestep']} | {row['dim']} | "
                f"{row['acc_mean']:.6f} | {row['acc_std']:.6f} | "
                f"{row['margin_mean']:.6f} | {row['true_residual_mean']:.6f} | "
                f"{row['nearest_wrong_residual_mean']:.6f} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 supervised noisy class-subspace validation for MNIST.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./几何假设验证/outputs_stage1")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-per-class", type=int, default=0, help="0 means full training set")
    parser.add_argument("--eval-per-class", type=int, default=0, help="0 means full test set")
    parser.add_argument("--timesteps", default="0,50,100,150,200,250,300,400,500")
    parser.add_argument("--dims", default="0,1,2,5,10,20,50,100")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--train-noise-repeats", type=int, default=1)
    parser.add_argument("--eval-noise-repeats", type=int, default=5)
    parser.add_argument("--stability", action="store_true", help="compare subspace directions from two train noise seeds")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    ensure_dir(args.output_dir)

    timesteps = parse_int_list(args.timesteps)
    dims = parse_int_list(args.dims)
    max_dim = max(dims)

    train_loader = build_loader(args, train=True)
    eval_loader = build_loader(args, train=False)
    train_x, train_y = collect_features(train_loader, device)
    eval_x, eval_y = collect_features(eval_loader, device)
    clean_centers = build_clean_centers(train_x, train_y, args.num_classes)

    print(
        f"device={device} train={train_x.shape[0]} eval={eval_x.shape[0]} "
        f"timesteps={timesteps} dims={dims}"
    )

    run_rows: List[Dict] = []
    evr_rows: List[Dict] = []
    stability_rows: List[Dict] = []

    for timestep in timesteps:
        centers_t, bases, explained = class_pca_from_noisy_samples(
            train_x,
            train_y,
            clean_centers,
            timestep=timestep,
            diffusion_steps=args.diffusion_steps,
            max_dim=max_dim,
            train_noise_repeats=args.train_noise_repeats,
            seed=args.seed + 17,
        )

        alt_bases = None
        if args.stability:
            _, alt_bases, _ = class_pca_from_noisy_samples(
                train_x,
                train_y,
                clean_centers,
                timestep=timestep,
                diffusion_steps=args.diffusion_steps,
                max_dim=max_dim,
                train_noise_repeats=args.train_noise_repeats,
                seed=args.seed + 991,
            )

        for dim in dims:
            evr_rows.append(
                {
                    "timestep": timestep,
                    "dim": dim,
                    "explained_variance_mean": float(explained[:, :dim].sum(dim=1).mean().item()) if dim > 0 else 0.0,
                    "explained_variance_min": float(explained[:, :dim].sum(dim=1).min().item()) if dim > 0 else 0.0,
                    "explained_variance_max": float(explained[:, :dim].sum(dim=1).max().item()) if dim > 0 else 0.0,
                }
            )
            if alt_bases is not None:
                stability_rows.append(
                    {
                        "timestep": timestep,
                        "dim": dim,
                        "subspace_similarity": subspace_similarity(bases, alt_bases, dim),
                    }
                )

            for repeat in range(args.eval_noise_repeats):
                metrics = evaluate_residual_classifier(
                    eval_x,
                    eval_y,
                    centers_t,
                    bases,
                    dim=dim,
                    timestep=timestep,
                    diffusion_steps=args.diffusion_steps,
                    seed=args.seed + 10_000 * repeat + timestep,
                    batch_size=args.eval_batch_size,
                )
                row = {
                    "timestep": timestep,
                    "dim": dim,
                    "eval_noise_repeat": repeat,
                }
                row.update(metrics)
                run_rows.append(row)

            latest = run_rows[-args.eval_noise_repeats :]
            acc_mean = sum(row["acc"] for row in latest) / len(latest)
            print(f"t={timestep} d={dim} acc_mean={acc_mean:.4f}")

    summary_rows = summarize_rows(run_rows)
    write_csv(os.path.join(args.output_dir, "metrics_by_run.csv"), run_rows)
    write_csv(os.path.join(args.output_dir, "metrics_summary.csv"), summary_rows)
    write_csv(os.path.join(args.output_dir, "explained_variance.csv"), evr_rows)
    write_csv(os.path.join(args.output_dir, "subspace_stability.csv"), stability_rows)
    write_markdown_summary(os.path.join(args.output_dir, "summary.md"), args, summary_rows)
    best = max(summary_rows, key=lambda row: row["acc_mean"])
    print(
        f"wrote={args.output_dir} best_t={best['timestep']} "
        f"best_d={best['dim']} best_acc={best['acc_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
