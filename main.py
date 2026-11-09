"""训练并评估基于配对含噪 MNIST 图像的条件 IR-SDE 去噪模型。

脚本负责参数校验、训练/验证/测试数据划分、模型训练、经典方法对比，
以及保存指标、图像、训练记录和模型检查点。
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm.auto import tqdm

from ir_sde.classical import (
    CLASSICAL_CONFIG,
    CLASSICAL_METHODS,
    denoise_batch,
    edge_magnitude_mae,
)
from ir_sde.data import NoisyMNISTDataset
from ir_sde.model import IRSdePipeline, IRSdeScheduler, make_mnist_unet


def train_epoch(
    unet,
    scheduler,
    loader,
    optimizer,
    device,
    description="Training",
    batch_history=None,
    epoch=1,
    lr_scheduler=None,
):
    """训练一个 epoch，并返回按样本数加权的损失和噪声预测误差。

    每个批次随机抽取 IR-SDE 时间步和标准高斯噪声，根据干净图像 ``x0``
    与含噪观测 ``y`` 构造 ``xt``。模型输入为 ``[xt,y]``，目标由
    ``prediction_type`` 决定；可选的 ``batch_history`` 记录逐批训练信息。
    """
    unet.train()
    total, total_noise, count = 0.0, 0.0, 0
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)
    for y, x0 in progress:
        # 将成对的含噪观测与干净目标放到训练设备上。
        y, x0 = y.to(device), x0.to(device)
        # 排除 t=0，因为该时刻噪声方差为零，不适合作为训练扰动状态。
        t = torch.randint(1, scheduler.T + 1, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        xt = scheduler.add_noise(x0, noise, t, y)
        # 在通道维拼接状态和观测，形成条件 U-Net 输入 [B,2,H,W]。
        prediction = unet(torch.cat([xt, y], dim=1), t).sample
        # 两种训练目标共享同一网络：sample 预测 x0，epsilon 预测添加的噪声。
        target = x0 if scheduler.config.prediction_type == "sample" else noise
        loss = F.mse_loss(prediction, target)
        with torch.no_grad():
            # 统一换算成噪声空间的 MSE，便于不同预测目标的过程监控。
            _, predicted_noise = scheduler.prediction_to_x0_noise(prediction, xt, t, y)
            noise_mse = F.mse_loss(predicted_noise, noise).item()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪限制单次更新幅度，降低训练不稳定的风险。
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if lr_scheduler is not None:
            # 学习率调度器按优化器更新步数推进，而不是每个 epoch 推进一次。
            lr_scheduler.step()
        batch_loss = loss.item()
        if batch_history is not None:
            batch_history.append(
                {
                    "global_batch": len(batch_history) + 1,
                    "epoch": epoch,
                    "train_loss": batch_loss,
                    "train_noise_mse": noise_mse,
                    "lr": current_lr,
                }
            )
        total += batch_loss * x0.size(0)
        total_noise += noise_mse * x0.size(0)
        count += x0.size(0)
        progress.set_postfix(
            loss=f"{batch_loss:.6f}",
            noise_mse=f"{noise_mse:.6f}",
            avg_loss=f"{total / count:.6f}",
            lr=f"{current_lr:.3e}",
            refresh=True,
        )
    if count == 0:
        raise ValueError("Training loader must not be empty")
    # 按每个批次实际包含的样本数加权，避免末批较小时平均值失真。
    return total / count, total_noise / count


@torch.no_grad()
def evaluate(unet, scheduler, loader, device, steps, seed, image_path):
    """用 IR-SDE 管线评估数据，并保存首批样本的目标/输入/恢复结果图。

    MSE 和 PSNR 按全体像素汇总；边缘 MAE 则先按图像计算再按样本数平均。
    固定生成器种子使同一配置下的随机采样过程可复现。
    """
    unet.eval()
    pipeline = IRSdePipeline(unet=unet, scheduler=scheduler).to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    noisy_error, restored_error = 0.0, 0.0
    noisy_edge_error, restored_edge_error, pixels = 0.0, 0.0, 0
    progress = tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True)
    for index, (y, x0) in enumerate(progress):
        # 每个样本使用同样的观测条件，比较输入噪声与模型恢复误差。
        y, x0 = y.to(device), x0.to(device)
        restored = pipeline(y, num_inference_steps=steps, generator=generator).images
        if not torch.isfinite(restored).all():
            raise RuntimeError("Non-finite restoration; check SDE parameters")
        # 累加平方误差而非批次均值，最后除以总像素数得到全数据集 MSE。
        noisy_error += F.mse_loss(y, x0, reduction="sum").item()
        restored_error += F.mse_loss(restored, x0, reduction="sum").item()
        noisy_edge_error += edge_magnitude_mae(y, x0) * x0.size(0)
        restored_edge_error += edge_magnitude_mae(restored, x0) * x0.size(0)
        pixels += x0.numel()
        progress.set_postfix(restored_mse=f"{restored_error / pixels:.6f}")
        if index == 0:
            n = min(8, x0.size(0))
            # 保存对比图：第一行干净目标、第二行含噪输入、第三行模型输出。
            save_image(
                torch.cat([x0[:n], y[:n], restored[:n]]).cpu(), image_path, nrow=n
            )
    if pixels == 0:
        raise ValueError("Evaluation loader must not be empty")
    # PSNR 的像素动态范围为 1，因此 PSNR = -10 log10(MSE)。
    noisy_mse, restored_mse = noisy_error / pixels, restored_error / pixels
    return {
        "noisy_mse": noisy_mse,
        "restored_mse": restored_mse,
        "noisy_edge_mae": noisy_edge_error / (pixels / x0[0].numel()),
        "noisy_psnr": -10 * torch.tensor(noisy_mse).log10().item(),
        "restored_psnr": -10 * torch.tensor(restored_mse).log10().item(),
        "restored_edge_mae": restored_edge_error / (pixels / x0[0].numel()),
    }


@torch.no_grad()
def evaluate_classical(loader, image_path):
    """在加载器给定的同一批观测上评估全部传统去噪基线。

    返回输入噪声指标以及每种经典方法的 MSE、PSNR 和边缘 MAE；首批结果按
    README 中的方法顺序生成可视化，确保比较对象使用一致的测试样本。
    """
    noisy_error = 0.0
    noisy_edge_error = 0.0
    pixels = 0
    method_errors = {name: 0.0 for name in CLASSICAL_METHODS}
    method_edge_errors = {name: 0.0 for name in CLASSICAL_METHODS}
    progress = tqdm(loader, desc="Evaluating classical baselines", unit="batch")
    for index, (y, x0) in enumerate(progress):
        # denoise_batch 返回以方法名为键的图像批次字典。
        denoised = denoise_batch(y)
        noisy_error += F.mse_loss(y, x0, reduction="sum").item()
        noisy_edge_error += edge_magnitude_mae(y, x0) * x0.size(0)
        pixels += x0.numel()
        for name, result in denoised.items():
            method_errors[name] += F.mse_loss(result, x0, reduction="sum").item()
            method_edge_errors[name] += (
                edge_magnitude_mae(result, x0) * x0.size(0)
            )
        if index == 0:
            n = min(8, x0.size(0))
            # 行顺序与 CLASSICAL_METHODS 一致，便于对照图例/结果表。
            rows = [x0[:n], y[:n]]
            rows.extend(denoised[name][:n] for name in CLASSICAL_METHODS)
            save_image(torch.cat(rows).cpu(), image_path, nrow=n)
    if pixels == 0:
        raise ValueError("Classical evaluation loader must not be empty")
    noisy_mse = noisy_error / pixels
    samples = pixels / x0[0].numel()
    return {
        "noisy_mse": noisy_mse,
        "noisy_psnr": -10 * torch.tensor(noisy_mse).log10().item(),
        "noisy_edge_mae": noisy_edge_error / samples,
        "methods": {
            name: {
                "restored_mse": error / pixels,
                "restored_psnr": -10 * torch.tensor(error / pixels).log10().item(),
                "restored_edge_mae": method_edge_errors[name] / samples,
            }
            for name, error in method_errors.items()
        },
    }


def save_comparison_csv(ir_sde_metrics, classical_metrics, output):
    """将输入、IR-SDE 与经典方法指标写入便于外部分析的 CSV 表格。"""
    rows = [
        {
            "method": "noisy",
            "mse": ir_sde_metrics["noisy_mse"],
            "psnr": ir_sde_metrics["noisy_psnr"],
            "edge_mae": ir_sde_metrics["noisy_edge_mae"],
        },
        {
            "method": "IR-SDE",
            "mse": ir_sde_metrics["restored_mse"],
            "psnr": ir_sde_metrics["restored_psnr"],
            "edge_mae": ir_sde_metrics["restored_edge_mae"],
        },
    ]
    rows.extend(
        {
            "method": name,
            "mse": metrics["restored_mse"],
            "psnr": metrics["restored_psnr"],
            "edge_mae": metrics["restored_edge_mae"],
        }
        for name, metrics in classical_metrics["methods"].items()
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "mse", "psnr", "edge_mae"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """声明命令行选项并校验数值范围、噪声配置和推理步数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # 路径和优化器/模型参数。
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./outputs/ir_sde")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    # 噪声配方概率顺序固定为数据集定义的四种噪声顺序。
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Rate per integer step; default: 5.0 / timesteps",
    )
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument(
        "--noise-probs", type=float, nargs=4, default=None,
        metavar=("GAUSSIAN", "SALT_PEPPER", "SPECKLE", "MIXED"),
        help="Noise probabilities in this order: gaussian salt_pepper speckle mixed; sum must be 1",
    )
    parser.add_argument("--prediction-type", choices=("sample", "epsilon"), default="sample",
                        help="Predict clean image (stable default) or diffusion noise")
    parser.add_argument("--gaussian-sigma", type=float, nargs=2, default=(0.05, 0.20),
                        metavar=("MIN", "MAX"), help="Gaussian corruption standard deviation range")
    parser.add_argument("--salt-pepper-amount", type=float, nargs=2, default=(0.01, 0.08),
                        metavar=("MIN", "MAX"), help="Fraction of pixels selected for salt/pepper replacement")
    # 数据规模、评估频率、每张图像生成的独立扰动数和运行设备。
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate fixed test samples every N epochs",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=256,
        help="Number of fixed test samples for intermediate evaluation",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=4,
        help="Independent noisy samples per training image (default: 4)",
    )
    parser.add_argument(
        "--test-samples-per-image",
        type=int,
        default=1,
        help="Fixed noisy samples per test image (default: 1)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    # 将位置参数转换为具名概率表，并复用数据集侧统一的校验逻辑。
    try:
        args.noise_probs = NoisyMNISTDataset.validate_noise_probs(
            None if args.noise_probs is None else
            dict(zip(NoisyMNISTDataset.NOISE_TYPES, args.noise_probs))
        )
    except ValueError as error:
        parser.error(str(error))
    # argparse 负责类型转换；此处集中拒绝非有限值、不合法范围和空数据配置。
    if not all(math.isfinite(v) for v in (args.lr, args.lam)) or (
        args.theta is not None and not math.isfinite(args.theta)
    ):
        parser.error("lr, lam and theta must be finite")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    if args.timesteps < 1 or not 1 <= args.inference_steps <= args.timesteps:
        parser.error("require 1 <= inference-steps <= timesteps")
    if args.lam <= 0 or (args.theta is not None and args.theta <= 0):
        parser.error("theta and lam must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if args.eval_every < 1 or args.eval_samples < 1:
        parser.error("eval-every and eval-samples must be positive")
    for name, bounds in (("gaussian-sigma", args.gaussian_sigma),
                         ("salt-pepper-amount", args.salt_pepper_amount)):
        if not all(math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]:
            parser.error(f"{name} must be a finite nonnegative increasing range")
    if args.salt_pepper_amount[1] > 1:
        parser.error("salt-pepper-amount must be <= 1")
    if args.samples_per_image < 1 or args.test_samples_per_image < 1:
        parser.error("samples-per-image and test-samples-per-image must be positive")
    for limit in (args.train_samples, args.test_samples):
        if limit is not None and limit < 1:
            parser.error("sample limits must be positive")
    return args


def save_history(history, batch_history, output):
    """保存 epoch/批次训练记录，并重绘训练损失与固定验证指标曲线。

    JSON 保留完整的记录结构；CSV 方便表格工具读取。曲线在每个 epoch 后更新，
    因此中断运行时也能查看已经完成部分的训练过程。
    """
    # epoch 级记录包含训练指标，以及评估发生时才存在的验证指标。
    fields = [
        "epoch",
        "global_batch",
        "train_noise_mse",
        "train_loss",
        "validation_noisy_mse",
        "validation_restored_mse",
        "validation_noisy_psnr",
        "validation_restored_psnr",
        "validation_noisy_edge_mae",
        "validation_restored_edge_mae",
    ]
    # 同时输出 JSON 和 CSV，批次文件额外保留学习率变化。
    with (output / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    with (output / "batch_history.json").open("w", encoding="utf-8") as file:
        json.dump(batch_history, file, indent=2)
    with (output / "batch_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["global_batch", "epoch", "train_loss", "train_noise_mse", "lr"]
        )
        writer.writeheader()
        writer.writerows(batch_history)
    # 左轴绘制训练损失与验证 MSE，右轴绘制验证 PSNR。
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        [r["global_batch"] for r in batch_history],
        [r["train_loss"] for r in batch_history],
        linewidth=0.8,
        alpha=0.65,
        label="Train loss (each batch)",
    )
    for index, row in enumerate(history):
        step = row["global_batch"]
        ax.axvline(step, color="gray", linestyle=":", alpha=0.4)
        ax.scatter(
            step,
            batch_history[step - 1]["train_loss"],
            color="tab:blue",
            s=25,
            zorder=4,
            label="Epoch end" if index == 0 else None,
        )
        ax.annotate(
            f"E{row['epoch']}",
            (step, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    # 仅连接已完成验证的 epoch，避免把未评估记录误当成零值。
    evaluated = [r for r in history if "validation_restored_mse" in r]
    steps = [r["global_batch"] for r in evaluated]
    psnr_ax = ax.twinx()
    for kind, color in (("noisy", "tab:gray"), ("restored", "tab:orange")):
        ax.plot(
            steps,
            [r[f"validation_{kind}_mse"] for r in evaluated],
            marker="o",
            linestyle="--",
            color=color,
            label=f"Fixed validation {kind} MSE",
        )
        psnr_ax.plot(
            steps,
            [r[f"validation_{kind}_psnr"] for r in evaluated],
            marker="s",
            linestyle=":",
            color=color,
            label=f"Fixed validation {kind} PSNR",
        )
    ax.set_xlabel("Global training batch (continuous across epochs)")
    ax.set_ylabel("MSE / training loss")
    psnr_ax.set_ylabel("PSNR (dB)")
    ax.set_title("Training and fixed validation history", pad=30)
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    right_lines, right_labels = psnr_ax.get_legend_handles_labels()
    ax.legend(lines + right_lines, labels + right_labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=160)
    plt.close(fig)


def main():
    """执行完整实验：准备数据、训练模型、评估并持久化所有结果。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # 默认令 theta*T=5，使终端信号系数约为 exp(-5)，接近观测中心。
    theta = 5.0 / args.timesteps if args.theta is None else args.theta
    scheduler = IRSdeScheduler(T=args.timesteps, theta=theta, lam=args.lam,
                              prediction_type=args.prediction_type)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    noise_ranges = {"gaussian": args.gaussian_sigma,
                    "salt_pepper": args.salt_pepper_amount}
    # 训练集采用在线随机噪声；验证集使用同一训练划分但固定噪声，保证跨 epoch 可比。
    train_source = NoisyMNISTDataset(
        root=args.data_root,
        train=True,
        seed=args.seed,
        samples_per_image=args.samples_per_image,
        noise_probs=args.noise_probs,
        noise_ranges=noise_ranges,
    )
    validation_source = NoisyMNISTDataset(
        root=args.data_root,
        train=True,
        seed=args.seed,
        samples_per_image=args.samples_per_image,
        noise_probs=args.noise_probs,
        noise_ranges=noise_ranges,
        deterministic_noise=True,
    )
    # 官方 MNIST 测试划分仅用于训练结束后的最终评估。
    test_data = NoisyMNISTDataset(
        root=args.data_root,
        train=False,
        seed=args.seed,
        samples_per_image=args.test_samples_per_image,
        noise_probs=args.noise_probs,
        noise_ranges=noise_ranges,
    )
    image_count = len(train_source.data)
    if image_count < 2:
        raise ValueError("At least two MNIST training images are required")
    # 按固定随机种子打乱原始训练图像，再划出 10% 作为验证图像，避免图像级泄漏。
    image_order = torch.randperm(
        image_count, generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    validation_image_count = max(1, round(image_count * 0.1))
    validation_images = image_order[:validation_image_count]
    training_images = image_order[validation_image_count:]

    def virtual_indices(image_ids, samples_per_image):
        """把原始图像编号展开成数据集内部的多个虚拟噪声样本索引。"""
        return [
            image_id * samples_per_image + slot
            for image_id in image_ids
            for slot in range(samples_per_image)
        ]

    # 训练/验证子集基于原图划分，而非虚拟噪声样本划分。
    train_data = Subset(
        train_source, virtual_indices(training_images, args.samples_per_image)
    )
    validation_data = Subset(
        validation_source, virtual_indices(validation_images, args.samples_per_image)
    )
    if args.train_samples is not None:
        train_data = Subset(train_data, range(min(args.train_samples, len(train_data))))
    if args.test_samples is not None:
        test_data = Subset(test_data, range(min(args.test_samples, len(test_data))))
    # 训练批次打乱；验证和测试加载器保持顺序稳定，所有加载器均使用指定 worker 数。
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    # 每轮验证固定使用相同子集；独立测试集保留到训练完成后再参与最终报告。
    eval_indices = torch.randperm(
        len(validation_data), generator=torch.Generator().manual_seed(args.seed)
    )[: min(args.eval_samples, len(validation_data))].tolist()
    eval_loader = DataLoader(
        Subset(validation_data, eval_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    # 另取固定测试子集，供不同噪声类型的分项分析使用。
    test_eval_indices = torch.randperm(
        len(test_data), generator=torch.Generator().manual_seed(args.seed + 1)
    )[: min(args.eval_samples, len(test_data))].tolist()
    # 记录固定样本编号和关键随机设置，便于复现实验评估。
    with (output / "fixed_eval.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "validation_indices": eval_indices,
                "test_indices": test_eval_indices,
                "validation_image_ids": validation_images,
                "seed": args.seed,
                "noise_probs": args.noise_probs,
                "inference_steps": args.inference_steps,
                "test_samples_per_image": args.test_samples_per_image,
                "validation_samples_per_image": args.samples_per_image,
            },
            file,
            indent=2,
        )
    # 保存命令行参数、实际采用的 SDE 参数、噪声范围及经典基线设置。
    with (output / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                **vars(args),
                "theta": theta,
                "noise_probs": args.noise_probs,
                "noise_ranges": {**NoisyMNISTDataset.DEFAULT_NOISE_RANGES, **noise_ranges},
                "lr_schedule": "cosine",
                "classical_baselines": CLASSICAL_CONFIG,
                "validation_fraction": 0.1,
                "training_image_count": len(training_images),
                "validation_image_count": len(validation_images),
            },
            file,
            indent=2,
        )
    unet = make_mnist_unet().to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    # 余弦学习率按整个训练期间的优化器更新总步数连续衰减。
    total_steps = args.epochs * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.0
    )
    print(
        f"Device: {device}; train: {len(train_data)}; "
        f"validation: {len(validation_data)}; test: {len(test_data)}",
        flush=True,
    )
    print(f"Noise probabilities: {args.noise_probs}", flush=True)
    history = []
    batch_history = []
    for epoch in range(1, args.epochs + 1):
        # 完成一个训练轮次，并将批次级损失、噪声误差和学习率追加到记录中。
        loss, noise_mse = train_epoch(
            unet,
            scheduler,
            train_loader,
            optimizer,
            device,
            description=f"Epoch {epoch}/{args.epochs}",
            batch_history=batch_history,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
        )
        print(f"Epoch {epoch}/{args.epochs}: loss = {loss:.6f}, noise MSE = {noise_mse:.6f}", flush=True)
        history.append(
            {
                "epoch": epoch,
                "global_batch": len(batch_history),
                "train_loss": loss,
                "train_noise_mse": noise_mse,
            }
        )
        # 定期评估固定验证集，且无论间隔如何都保证最后一轮评估。
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            fixed_metrics = evaluate(
                unet,
                scheduler,
                eval_loader,
                device,
                args.inference_steps,
                args.seed,
                output / f"comparison_epoch_{epoch:04d}.png",
            )
            history[-1].update(
                {f"validation_{key}": value for key, value in fixed_metrics.items()}
            )
            print(
                f"Fixed validation: MSE = {fixed_metrics['restored_mse']:.6f}, "
                f"PSNR = {fixed_metrics['restored_psnr']:.2f} dB",
                flush=True,
            )
        # 每轮都更新记录并保存可恢复训练状态的检查点。
        save_history(history, batch_history, output)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "scheduler_config": dict(scheduler.config),
                "args": vars(args),
                "history": history,
                "batch_history": batch_history,
            },
            output / "checkpoint.pt",
        )
    # 训练结束后仅评估一次官方测试集，并在完全相同的观测上运行经典基线。
    metrics = evaluate(
        unet,
        scheduler,
        test_loader,
        device,
        args.inference_steps,
        args.seed,
        output / "comparison.png",
    )
    classical_metrics = evaluate_classical(test_loader, output / "comparison_classical.png")
    save_comparison_csv(metrics, classical_metrics, output)
    unet.save_pretrained(output / "unet")
    scheduler.save_pretrained(output / "scheduler")
    # 对训练配置中概率非零的噪声类型分别评估，便于分析各类扰动的表现。
    per_noise = {}
    for kind, probability in args.noise_probs.items():
        if probability == 0:
            continue
        dataset = NoisyMNISTDataset(
            root=args.data_root, train=False, noise_probs={kind: 1.0}, seed=args.seed,
            samples_per_image=args.test_samples_per_image, noise_ranges=noise_ranges,
        )
        loader = DataLoader(Subset(dataset, test_eval_indices), batch_size=args.batch_size,
                            generator=torch.Generator().manual_seed(args.seed))
        per_noise_ir_sde = evaluate(
            unet, scheduler, loader, device, args.inference_steps,
            args.seed, output / f"comparison_{kind}.png",
        )
        per_noise_classical = evaluate_classical(
            loader, output / f"comparison_classical_{kind}.png"
        )
        per_noise[kind] = {
            "ir_sde": per_noise_ir_sde,
            "classical": per_noise_classical["methods"],
        }
    # 聚合总体验证/测试记录、经典方法指标以及按噪声拆分的结果。
    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "history": history,
                **metrics,
                "classical": classical_metrics["methods"],
                "per_noise": per_noise,
            },
            file,
            indent=2,
        )
    print(json.dumps(metrics, indent=2), flush=True)
    print(json.dumps(classical_metrics["methods"], indent=2), flush=True)
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
