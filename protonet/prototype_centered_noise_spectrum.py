import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


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


def balanced_subset_indices(dataset, per_class: int, num_classes: int, seed: int) -> List[int]:
    if per_class <= 0:
        return list(range(len(dataset)))

    generator = torch.Generator().manual_seed(seed)
    labels = torch.as_tensor(dataset.targets)
    selected: List[int] = []
    for cls in range(num_classes):
        cls_indices = torch.where(labels == cls)[0]
        perm = torch.randperm(cls_indices.numel(), generator=generator)
        selected.extend(cls_indices[perm[: min(per_class, cls_indices.numel())]].tolist())
    return selected


def build_loader(args: argparse.Namespace, train: bool) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(args.data_dir, train=train, download=True, transform=transform)
    per_class = args.prototype_per_class if train else args.samples_per_class
    seed = args.seed if train else args.seed + 1000
    indices = balanced_subset_indices(dataset, per_class, args.num_classes, seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"missing prototype samples for classes: {missing}")
    return sums / counts.view(-1, 1, 1, 1)


@torch.no_grad()
def collect_noisy_samples(
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    timestep: int,
    noise_repeats: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    repeats = max(noise_repeats, 1)
    for repeat in range(repeats):
        generator = torch.Generator(device=device).manual_seed(seed + 100_003 * repeat + timestep)
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            t = torch.full((images.shape[0],), timestep, device=device, dtype=torch.long)
            if timestep == 0:
                x_t = images
            else:
                noise = torch.randn(images.shape, generator=generator, device=device, dtype=images.dtype)
                x_t = schedule.q_sample(images, t, noise)
            features.append(x_t.flatten(start_dim=1).detach().cpu())
            labels.append(batch_labels.detach().cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def center_for_timestep(
    prototypes: torch.Tensor,
    schedule: DiffusionSchedule,
    timestep: int,
    center_mode: str,
) -> torch.Tensor:
    if center_mode == "scaled-prototype":
        return schedule.sqrt_alpha_bars[timestep].detach().cpu() * prototypes.detach().cpu().flatten(start_dim=1)
    if center_mode == "prototype":
        return prototypes.detach().cpu().flatten(start_dim=1)
    raise ValueError(f"unsupported center_mode: {center_mode}")


def spectrum_around_center(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    residual = x - center.view(1, -1)
    singular_values = torch.linalg.svdvals(residual)
    eigenvalues = singular_values.pow(2) / max(x.shape[0] - 1, 1)
    return eigenvalues.clamp_min(0)


def spectral_metrics(eigenvalues: torch.Tensor, topks: Iterable[int], variance_levels: Iterable[float]) -> Dict[str, float]:
    total = eigenvalues.sum().clamp_min(1e-12)
    probs = eigenvalues / total
    cumulative = torch.cumsum(probs, dim=0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    participation = total.pow(2) / eigenvalues.pow(2).sum().clamp_min(1e-12)

    row: Dict[str, float] = {
        "trace": float(total.item()),
        "top1_eigenvalue": float(eigenvalues[0].item()) if eigenvalues.numel() else 0.0,
        "spectral_entropy": float(entropy.item()),
        "effective_rank": float(torch.exp(entropy).item()),
        "participation_dim": float(participation.item()),
    }
    for topk in topks:
        take = min(topk, eigenvalues.numel())
        row[f"top{topk}_ratio"] = float(probs[:take].sum().item()) if take else 0.0
    for level in variance_levels:
        row[f"dim_{int(level * 100)}var"] = int((cumulative < level).sum().item() + 1)
    return row


def class_distance_metrics(x: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor) -> Dict[str, float]:
    d2 = (x[:, None, :] - centers[None]).pow(2).mean(dim=2)
    preds = d2.argmin(dim=1)
    true_d2 = d2[torch.arange(x.shape[0]), labels]
    wrong = d2.clone()
    wrong[torch.arange(x.shape[0]), labels] = float("inf")
    nearest_wrong = wrong.min(dim=1).values
    margin = nearest_wrong - true_d2
    return {
        "nearest_center_acc": float((preds == labels).float().mean().item()),
        "margin_mean": float(margin.mean().item()),
        "true_mse_mean": float(true_d2.mean().item()),
        "nearest_wrong_mse_mean": float(nearest_wrong.mean().item()),
    }


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: str, args: argparse.Namespace, aggregate_rows: List[Dict]) -> None:
    best_sep = max(aggregate_rows, key=lambda row: row["between_within_ratio"])
    flattest = max(aggregate_rows, key=lambda row: row["effective_rank_mean"])
    with open(path, "w") as f:
        f.write("# Prototype-Centered Noise Spectrum Summary\n\n")
        f.write("This experiment analyzes MNIST noisy samples in pixel space, centered at class-average prototypes.\n\n")
        f.write("## Config\n\n")
        f.write(f"- sample_split: {args.sample_split}\n")
        f.write(f"- timesteps: {args.timesteps}\n")
        f.write(f"- center_mode: {args.center_mode}\n")
        f.write(f"- prototype_per_class: {args.prototype_per_class}\n")
        f.write(f"- samples_per_class: {args.samples_per_class}\n")
        f.write(f"- noise_repeats: {args.noise_repeats}\n\n")
        f.write("## Notable Timesteps\n\n")
        f.write(
            f"- highest between/within ratio: t={int(best_sep['timestep'])}, "
            f"ratio={best_sep['between_within_ratio']:.6f}\n"
        )
        f.write(
            f"- flattest mean spectrum: t={int(flattest['timestep'])}, "
            f"effective_rank={flattest['effective_rank_mean']:.2f}\n\n"
        )
        f.write("## Aggregate Metrics\n\n")
        f.write("| timestep | trace_mean | effective_rank_mean | top10_ratio_mean | between_within_ratio | nearest_center_acc |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in aggregate_rows:
            f.write(
                f"| {int(row['timestep'])} | {row['trace_mean']:.6f} | "
                f"{row['effective_rank_mean']:.2f} | {row['top10_ratio_mean']:.4f} | "
                f"{row['between_within_ratio']:.6f} | {row['nearest_center_acc']:.4f} |\n"
            )


def maybe_write_plots(output_dir: str, aggregate_rows: List[Dict], eigen_rows: List[Dict], eigen_top: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip_plots reason={exc}")
        return

    timesteps = [int(row["timestep"]) for row in aggregate_rows]
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, [row["trace_mean"] for row in aggregate_rows], "o-", label="within trace")
    plt.plot(timesteps, [row["effective_rank_mean"] for row in aggregate_rows], "s-", label="effective rank")
    plt.plot(timesteps, [row["between_within_ratio"] for row in aggregate_rows], "^-", label="between/within")
    plt.plot(timesteps, [row["nearest_center_acc"] for row in aggregate_rows], "d-", label="nearest center acc")
    plt.xlabel("timestep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregate_noise_spectrum_curves.png"), dpi=180)
    plt.close()

    by_timestep: Dict[int, List[List[float]]] = {}
    for row in eigen_rows:
        timestep = int(row["timestep"])
        values = [float(row[f"lambda_{idx}"]) for idx in range(1, eigen_top + 1) if f"lambda_{idx}" in row]
        by_timestep.setdefault(timestep, []).append(values)

    plt.figure(figsize=(8, 5))
    for timestep in sorted(by_timestep):
        values = torch.tensor(by_timestep[timestep], dtype=torch.float32)
        mean_values = values.mean(dim=0).clamp_min(1e-12)
        plt.plot(range(1, mean_values.numel() + 1), mean_values.tolist(), label=f"t={timestep}")
    plt.yscale("log")
    plt.xlabel("eigenvalue index")
    plt.ylabel("mean class eigenvalue")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_class_eigenvalue_spectra.png"), dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Class prototype-centered spectrum of noisy MNIST samples.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./protonet/outputs/prototype_centered_noise_spectrum")
    parser.add_argument("--sample-split", choices=["train", "test"], default="train")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--prototype-per-class", type=int, default=0, help="0 means all train samples.")
    parser.add_argument("--samples-per-class", type=int, default=0, help="0 means all samples in sample split.")
    parser.add_argument("--timesteps", default="0,100,200,250,300,400,500")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--center-mode", choices=["scaled-prototype", "prototype"], default="scaled-prototype")
    parser.add_argument("--topks", default="1,5,10,20,50,100")
    parser.add_argument("--eigen-top", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    timesteps = parse_int_list(args.timesteps)
    topks = parse_int_list(args.topks)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    prototype_loader = build_loader(args, train=True)
    sample_loader = build_loader(args, train=args.sample_split == "train")
    prototypes = build_prototypes(prototype_loader, device, args.num_classes)

    print(
        f"device={device} sample_split={args.sample_split} batches={len(sample_loader)} "
        f"timesteps={timesteps} center_mode={args.center_mode}"
    )

    class_rows: List[Dict] = []
    eigen_rows: List[Dict] = []
    aggregate_rows: List[Dict] = []

    for timestep in timesteps:
        samples, labels = collect_noisy_samples(
            sample_loader,
            schedule,
            device,
            timestep=timestep,
            noise_repeats=args.noise_repeats,
            seed=args.seed + 17,
        )
        centers = center_for_timestep(prototypes, schedule, timestep, args.center_mode)
        distance_metrics = class_distance_metrics(samples, labels, centers)

        traces = []
        effective_ranks = []
        top10_ratios = []
        for cls in range(args.num_classes):
            cls_samples = samples[labels == cls]
            if cls_samples.shape[0] < 2:
                raise ValueError(f"class {cls} has fewer than 2 samples at timestep {timestep}")
            eigenvalues = spectrum_around_center(cls_samples, centers[cls])
            metrics = spectral_metrics(eigenvalues, topks=topks, variance_levels=(0.80, 0.90, 0.95))
            row = {
                "timestep": timestep,
                "class": cls,
                "num_samples": int(cls_samples.shape[0]),
                "feature_dim": int(cls_samples.shape[1]),
            }
            row.update(metrics)
            class_rows.append(row)

            eigen_row = {"timestep": timestep, "class": cls}
            for idx, value in enumerate(eigenvalues[: args.eigen_top].tolist(), start=1):
                eigen_row[f"lambda_{idx}"] = float(value)
            eigen_rows.append(eigen_row)

            traces.append(metrics["trace"])
            effective_ranks.append(metrics["effective_rank"])
            top10_ratios.append(metrics.get("top10_ratio", 0.0))

        center_global = centers.mean(dim=0, keepdim=True)
        between_trace = (centers - center_global).pow(2).sum(dim=1).mean().item()
        within_trace = float(sum(traces) / len(traces))
        aggregate_row = {
            "timestep": timestep,
            "trace_mean": within_trace,
            "trace_std": float(torch.tensor(traces).std(unbiased=False).item()),
            "effective_rank_mean": float(torch.tensor(effective_ranks).mean().item()),
            "effective_rank_std": float(torch.tensor(effective_ranks).std(unbiased=False).item()),
            "top10_ratio_mean": float(torch.tensor(top10_ratios).mean().item()),
            "between_trace": float(between_trace),
            "within_trace": within_trace,
            "between_within_ratio": float(between_trace / max(within_trace, 1e-12)),
        }
        aggregate_row.update(distance_metrics)
        aggregate_rows.append(aggregate_row)
        print(
            f"t={timestep} trace={aggregate_row['trace_mean']:.6f} "
            f"erank={aggregate_row['effective_rank_mean']:.2f} "
            f"top10={aggregate_row['top10_ratio_mean']:.4f} "
            f"between/within={aggregate_row['between_within_ratio']:.6f} "
            f"acc={aggregate_row['nearest_center_acc']:.4f}"
        )

    write_csv(os.path.join(args.output_dir, "class_spectrum_metrics.csv"), class_rows)
    write_csv(os.path.join(args.output_dir, "aggregate_spectrum_metrics.csv"), aggregate_rows)
    write_csv(os.path.join(args.output_dir, "eigenvalues_by_class.csv"), eigen_rows)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    write_summary(os.path.join(args.output_dir, "summary.md"), args, aggregate_rows)
    if not args.no_plots:
        maybe_write_plots(args.output_dir, aggregate_rows, eigen_rows, args.eigen_top)
    print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
