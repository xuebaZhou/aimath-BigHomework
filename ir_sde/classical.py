"""Classical image denoisers used as non-learned comparison baselines."""

import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter, sobel


CLASSICAL_METHODS = ("gaussian", "median", "bilateral", "tv")
CLASSICAL_CONFIG = {
    "gaussian_sigma": 0.7,
    "median_kernel": 3,
    "bilateral_sigma_color": 0.1,
    "bilateral_radius": 2,
    "tv_weight": 0.08,
    "tv_iterations": 30,
}


def _bilateral(images, sigma_color, radius):
    spatial_sigma = max(radius / 2, 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    spatial_weights = np.exp(
        -(xx**2 + yy**2) / (2 * spatial_sigma**2)
    ).astype(np.float32)
    padded = np.pad(
        images, ((0, 0), (radius, radius), (radius, radius)), mode="reflect"
    )
    result = np.zeros_like(images)
    total_weight = np.zeros_like(images)
    for row in range(2 * radius + 1):
        for column in range(2 * radius + 1):
            neighbor = padded[
                :, row : row + images.shape[1], column : column + images.shape[2]
            ]
            difference = neighbor - images
            weights = spatial_weights[row, column] * np.exp(
                -(difference**2) / (2 * sigma_color**2)
            )
            result += weights * neighbor
            total_weight += weights
    return result / np.maximum(total_weight, np.finfo(np.float32).eps)


def _tv_chambolle(images, weight, iterations):
    """Proximal TV denoising via Chambolle's dual-projection iterations."""
    px = np.zeros_like(images)
    py = np.zeros_like(images)
    tau = 0.25
    for _ in range(iterations):
        divergence = px.copy()
        divergence[:, :, 1:] -= px[:, :, :-1]
        divergence += py
        divergence[:, 1:, :] -= py[:, :-1, :]
        estimate = images - weight * divergence

        grad_x = np.zeros_like(images)
        grad_y = np.zeros_like(images)
        grad_x[:, :, :-1] = estimate[:, :, 1:] - estimate[:, :, :-1]
        grad_y[:, :-1, :] = estimate[:, 1:, :] - estimate[:, :-1, :]
        norm = np.sqrt(grad_x**2 + grad_y**2)
        denominator = 1 + (tau / weight) * norm
        px = (px - (tau / weight) * grad_x) / denominator
        py = (py - (tau / weight) * grad_y) / denominator
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
    """Run fixed traditional denoisers on a [B, 1, H, W] image batch."""
    if noisy.ndim != 4 or noisy.shape[1] != 1:
        raise ValueError("noisy must have shape [B, 1, H, W]")
    if not torch.isfinite(noisy).all():
        raise ValueError("noisy images must contain finite values")
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
    return {
        name: torch.from_numpy(np.clip(result[:, None], 0, 1)).to(
            device=noisy.device, dtype=noisy.dtype
        )
        for name, result in filtered.items()
    }


def edge_magnitude_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error between Sobel gradient magnitudes."""
    pred = prediction.detach().to(device="cpu", dtype=torch.float32).numpy()[:, 0]
    truth = target.detach().to(device="cpu", dtype=torch.float32).numpy()[:, 0]
    pred_edges = np.hypot(sobel(pred, axis=1), sobel(pred, axis=2))
    truth_edges = np.hypot(sobel(truth, axis=1), sobel(truth, axis=2))
    return float(np.abs(pred_edges - truth_edges).mean())
