"""传统图像去噪基线及其评估指标。

本模块实现不需要训练的高斯、中值、双边和全变分（TV）去噪方法，
用于与 IR-SDE 模型在同一批含噪图像上进行对照。
"""

import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter, sobel


CLASSICAL_METHODS = ("gaussian", "median", "bilateral", "tv")
# 固定的经典方法参数；同时写入实验配置，便于复现实验和解释比较结果。
CLASSICAL_CONFIG = {
    "gaussian_sigma": 0.7,
    "median_kernel": 3,
    "bilateral_sigma_color": 0.1,
    "bilateral_radius": 2,
    "tv_weight": 0.08,
    "tv_iterations": 30,
}


def _bilateral(images, sigma_color, radius):
    """对形状为 [B,H,W] 的 NumPy 图像批次执行双边滤波。

    双边权重同时考虑像素空间距离与灰度差：空间上较远或灰度差较大的邻点
    权重更低，因此在平滑局部噪声的同时尽量保留图像边缘。
    """
    # 空间高斯尺度由滤波半径决定，避免半径较小时尺度退化为零。
    spatial_sigma = max(radius / 2, 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    spatial_weights = np.exp(
        -(xx**2 + yy**2) / (2 * spatial_sigma**2)
    ).astype(np.float32)
    # 使用镜像边界延拓，避免边缘邻域因补零而产生明显暗边。
    padded = np.pad(
        images, ((0, 0), (radius, radius), (radius, radius)), mode="reflect"
    )
    result = np.zeros_like(images)
    total_weight = np.zeros_like(images)
    for row in range(2 * radius + 1):
        for column in range(2 * radius + 1):
            # 当前偏移位置的邻居像素；批次维保持不变。
            neighbor = padded[
                :, row : row + images.shape[1], column : column + images.shape[2]
            ]
            difference = neighbor - images
            weights = spatial_weights[row, column] * np.exp(
                -(difference**2) / (2 * sigma_color**2)
            )
            # 累加加权像素值和权重，最后逐像素归一化。
            result += weights * neighbor
            total_weight += weights
    return result / np.maximum(total_weight, np.finfo(np.float32).eps)


def _tv_chambolle(images, weight, iterations):
    """用 Chambolle 对偶投影迭代执行 TV 去噪。

    将去噪写作 ``min_x 0.5*||x-images||^2 + weight*TV(x)``。
    ``px``、``py`` 是对偶向量场，每轮根据当前估计的水平/垂直梯度更新，
    最终通过其散度得到近端解。
    """
    # 对偶变量分别对应水平和垂直方向的梯度。
    px = np.zeros_like(images)
    py = np.zeros_like(images)
    # Chambolle 对偶投影使用的固定步长。
    tau = 0.25
    for _ in range(iterations):
        # 由对偶变量计算散度，并得到当前图像估计。
        divergence = px.copy()
        divergence[:, :, 1:] -= px[:, :, :-1]
        divergence += py
        divergence[:, 1:, :] -= py[:, :-1, :]
        estimate = images - weight * divergence

        # 在图像右侧/底部之外不再计算向前差分，边界梯度置为零。
        grad_x = np.zeros_like(images)
        grad_y = np.zeros_like(images)
        grad_x[:, :, :-1] = estimate[:, :, 1:] - estimate[:, :, :-1]
        grad_y[:, :-1, :] = estimate[:, 1:, :] - estimate[:, :-1, :]
        norm = np.sqrt(grad_x**2 + grad_y**2)
        denominator = 1 + (tau / weight) * norm
        # 对偶场投影更新，抑制其范数超过 TV 对偶约束。
        px = (px - (tau / weight) * grad_x) / denominator
        py = (py - (tau / weight) * grad_y) / denominator
    # 用收敛后的对偶变量恢复去噪结果。
    divergence = px.copy()
    divergence[:, :, 1:] -= px[:, :, :-1]
    divergence += py
    divergence[:, 1:, :] -= py[:, :-1, :]
    return images - weight * divergence


def denoise_batch(
    noisy: torch.Tensor,
    *,
    gaussian_sigma=CLASSICAL_CONFIG["gaussian_sigma"],
    bilateral_sigma_color=CLASSICAL_CONFIG["bilateral_sigma_color"],
    tv_weight=CLASSICAL_CONFIG["tv_weight"],
    tv_iterations=CLASSICAL_CONFIG["tv_iterations"],
):
    """在含噪图像批次上运行四种传统去噪方法。

    输入和返回张量均为 ``[B,1,H,W]``；计算期间转到 CPU/float32 以便使用
    SciPy 与 NumPy，返回前恢复输入张量的设备和数据类型。各方法的输出裁剪
    至 MNIST 像素范围 ``[0,1]``。
    """
    if noisy.ndim != 4 or noisy.shape[1] != 1:
        raise ValueError("noisy must have shape [B, 1, H, W]")
    if not torch.isfinite(noisy).all():
        raise ValueError("noisy images must contain finite values")
    # 先校验数值，避免非法参数导致滤波过程产生 NaN 或无意义结果。
    strengths = (gaussian_sigma, bilateral_sigma_color, tv_weight)
    if not all(math.isfinite(value) for value in strengths):
        raise ValueError("Denoising strengths must be finite")
    if gaussian_sigma < 0 or bilateral_sigma_color <= 0 or tv_weight <= 0:
        raise ValueError("Denoising strengths must be positive (Gaussian may be zero)")
    if (
        isinstance(tv_iterations, bool)
        or not isinstance(tv_iterations, int)
        or tv_iterations < 1
    ):
        raise ValueError("tv_iterations must be a positive integer")

    # 传统滤波实现使用 NumPy；移除通道维后统一得到 [B,H,W]。
    images = noisy.detach().to(device="cpu", dtype=torch.float32).numpy()[:, 0]
    filtered = {
        "gaussian": gaussian_filter(
            images, sigma=(0, gaussian_sigma, gaussian_sigma), mode="reflect"
        ),
        "median": median_filter(
            images,
            size=(
                1,
                CLASSICAL_CONFIG["median_kernel"],
                CLASSICAL_CONFIG["median_kernel"],
            ),
            mode="reflect",
        ),
        "bilateral": _bilateral(
            images,
            sigma_color=bilateral_sigma_color,
            radius=CLASSICAL_CONFIG["bilateral_radius"],
        ),
        "tv": _tv_chambolle(images, weight=tv_weight, iterations=tv_iterations),
    }
    # 恢复通道维、限制像素范围，并将结果放回原设备/类型。
    return {
        name: torch.from_numpy(np.clip(result[:, None], 0, 1)).to(
            device=noisy.device, dtype=noisy.dtype
        )
        for name, result in filtered.items()
    }


def edge_magnitude_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """计算预测与目标的 Sobel 边缘幅值平均绝对误差。

    该指标补充像素 MSE/PSNR，用于观察边缘响应差异；它本身不能证明纹理
    或视觉质量得到保留。输入采用 ``[B,1,H,W]`` 张量。
    """
    # SciPy 在 CPU NumPy 数组上计算 Sobel；两个空间方向的梯度合成为幅值。
    pred = prediction.detach().to(device="cpu", dtype=torch.float32).numpy()[:, 0]
    truth = target.detach().to(device="cpu", dtype=torch.float32).numpy()[:, 0]
    pred_edges = np.hypot(sobel(pred, axis=1), sobel(pred, axis=2))
    truth_edges = np.hypot(sobel(truth, axis=1), sobel(truth, axis=2))
    return float(np.abs(pred_edges - truth_edges).mean())
