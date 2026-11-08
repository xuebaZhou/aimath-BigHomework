import unittest
from unittest.mock import patch

import torch

from ir_sde.classical import CLASSICAL_METHODS, denoise_batch, edge_magnitude_mae
from ir_sde.data import NoisyMNISTDataset


class ClassicalDenoiserTests(unittest.TestCase):
    def test_methods_preserve_shape_finite_range_and_device(self):
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
        image = torch.rand(2, 1, 28, 28)

        self.assertEqual(edge_magnitude_mae(image, image), 0.0)

    def test_tv_optimization_reduces_synthetic_gaussian_noise(self):
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
        with self.assertRaisesRegex(ValueError, "shape"):
            denoise_batch(torch.rand(1, 28, 28))

    @patch("ir_sde.data.MNIST")
    def test_validation_corruption_is_reproducible_on_training_split(self, mnist):
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
