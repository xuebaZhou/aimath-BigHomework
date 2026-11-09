import unittest
from unittest.mock import patch

import torch

from ir_sde.classical import CLASSICAL_METHODS, denoise_batch, edge_magnitude_mae
from ir_sde.data import NoisyMNISTDataset


class ClassicalDenoiserTests(unittest.TestCase):
    """检查传统去噪输出契约、边缘指标和可复现数据扰动。"""

    def test_methods_preserve_shape_finite_range_and_device(self):
        """四种方法都应保留批次形状/设备，且结果有限并位于合法像素范围。"""
        torch.manual_seed(7)
        clean = torch.zeros(2, 1, 28, 28)
        clean[:, :, 8:20, 8:20] = 1
        noisy = (clean + 0.15 * torch.randn_like(clean)).clamp(0, 1)

        results = denoise_batch(noisy)

        self.assertEqual(set(results), set(CLASSICAL_METHODS))
        for result in results.values():
            self.assertEqual(result.shape, noisy.shape)
            self.assertEqual(result.device, noisy.device)
            self.assertTrue(torch.isfinite(result).all())
            self.assertGreaterEqual(result.min().item(), 0)
            self.assertLessEqual(result.max().item(), 1)

    def test_edge_metric_is_zero_for_identical_images(self):
        """相同图像的边缘幅值误差应严格为零。"""
        image = torch.rand(2, 1, 28, 28)

        self.assertEqual(edge_magnitude_mae(image, image), 0.0)

    def test_tv_optimization_reduces_synthetic_gaussian_noise(self):
        """简单方块图像上，TV 去噪应降低合成高斯噪声造成的均方误差。"""
        torch.manual_seed(7)
        clean = torch.zeros(2, 1, 28, 28)
        clean[:, :, 8:20, 8:20] = 1
        noisy = (clean + 0.12 * torch.randn_like(clean)).clamp(0, 1)

        restored = denoise_batch(noisy)["tv"]

        self.assertLess(
            torch.mean((restored - clean) ** 2).item(),
            torch.mean((noisy - clean) ** 2).item(),
        )

    def test_invalid_input_shape_is_rejected(self):
        """缺少通道维的输入必须在滤波前被拒绝。"""
        with self.assertRaisesRegex(ValueError, "shape"):
            denoise_batch(torch.rand(1, 28, 28))

    @patch("ir_sde.data.MNIST")
    def test_validation_corruption_is_reproducible_on_training_split(self, mnist):
        """显式开启确定性扰动时，即使读取训练划分也应复现相同噪声。"""
        mnist.return_value = [(torch.full((1, 28, 28), 0.5), 0)]
        dataset = NoisyMNISTDataset(
            root="unused",
            train=True,
            noise_probs={"gaussian": 1.0},
            seed=17,
            noise_ranges={"gaussian": (0.2, 0.2)},
            deterministic_noise=True,
        )

        noisy_a, clean_a = dataset[0]
        noisy_b, clean_b = dataset[0]

        self.assertTrue(torch.equal(noisy_a, noisy_b))
        self.assertTrue(torch.equal(clean_a, clean_b))
        self.assertTrue(mnist.call_args.kwargs["train"])


if __name__ == "__main__":
    unittest.main()
