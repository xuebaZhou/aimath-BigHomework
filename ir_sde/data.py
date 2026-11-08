"""On-demand MNIST corruption with a configurable noise mixture."""

import math

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class NoisyMNISTDataset(Dataset):
    """Return (corrupted, clean) images in [0, 1].

    Training redraws noise on every access. Deterministic noise uses seed +
    image index so comparisons use identical corruptions without caching images.
    Each image has samples_per_image virtual slots with independent corruptions.
    Probabilities select one recipe per slot, rather than exact dataset counts.
    """

    NOISE_TYPES = ("gaussian", "salt_pepper", "speckle", "mixed")
    DEFAULT_NOISE_PROBS = {
        "gaussian": 0.25,
        "salt_pepper": 0.25,
        "speckle": 0.25,
        "mixed": 0.25,
    }
    DEFAULT_NOISE_RANGES = {
        "gaussian": (0.05, 0.20),
        "salt_pepper": (0.01, 0.08),
        "speckle": (0.10, 0.30),
        "mixed_gaussian": (0.03, 0.12),
        "mixed_salt_pepper": (0.01, 0.04),
    }

    @classmethod
    def validate_noise_probs(cls, noise_probs=None):
        """Validate probabilities and fill omitted noise types with zero."""
        values = cls.DEFAULT_NOISE_PROBS if noise_probs is None else noise_probs
        if not values or set(values) - set(cls.NOISE_TYPES):
            raise ValueError(f"noise_probs must use these names: {', '.join(cls.NOISE_TYPES)}")
        try:
            probabilities = {name: float(values.get(name, 0.0)) for name in cls.NOISE_TYPES}
        except (TypeError, ValueError) as error:
            raise ValueError("Noise probabilities must be numeric") from error
        total = sum(probabilities.values())
        if any(not math.isfinite(p) or p < 0 for p in probabilities.values()) or not math.isclose(
            total, 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("Noise probabilities must be finite, nonnegative and sum to 1")
        return {name: p / total for name, p in probabilities.items()}

    def __init__(
        self,
        root: str,
        train: bool = True,
        noise_probs=None,
        seed: int = 42,
        samples_per_image: int = 1,
        noise_ranges=None,
        deterministic_noise=None,
    ):
        if deterministic_noise is not None and not isinstance(deterministic_noise, bool):
            raise ValueError("deterministic_noise must be a boolean")
        if (
            isinstance(samples_per_image, bool)
            or not isinstance(samples_per_image, int)
            or samples_per_image < 1
        ):
            raise ValueError("samples_per_image must be a positive integer")
        self.samples_per_image = samples_per_image
        self.train = train
        self.deterministic_noise = not train if deterministic_noise is None else deterministic_noise
        self.seed = seed
        self.noise_ranges = dict(self.DEFAULT_NOISE_RANGES)
        if noise_ranges is not None:
            if set(noise_ranges) - self.noise_ranges.keys():
                raise ValueError("Unknown noise strength range")
            self.noise_ranges.update(noise_ranges)
        for name, bounds in self.noise_ranges.items():
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds):
                raise ValueError(f"Invalid noise range: {name}")
            low, high = bounds
            if not 0 <= low <= high or ("salt_pepper" in name and high > 1):
                raise ValueError(f"Invalid noise range: {name}")
        self.noise_probs = self.validate_noise_probs(noise_probs)
        weights = list(self.noise_probs.values())
        self.noise_names = list(self.noise_probs)
        self.noise_weights = torch.tensor(weights, dtype=torch.float64)
        self.data = MNIST(root=root, train=train, download=True, transform=ToTensor())

    def __len__(self):
        return len(self.data) * self.samples_per_image

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError("Dataset index out of range")
        # Each source image occupies samples_per_image virtual sample slots.
        img, _ = self.data[idx // self.samples_per_image]
        generator = None
        if self.deterministic_noise:
            generator = torch.Generator().manual_seed(self.seed + idx)
        choice = torch.multinomial(self.noise_weights, 1, generator=generator).item()
        noisy_img = self._corrupt(img, self.noise_names[choice], generator)
        return noisy_img.clamp(0, 1), img

    @staticmethod
    def _uniform(low, high, generator):
        return low + (high - low) * torch.rand((), generator=generator).item()

    def _corrupt(self, img, kind, generator):
        if kind == "gaussian":
            sigma = self._uniform(*self.noise_ranges["gaussian"], generator)
            return img + sigma * torch.randn(img.shape, generator=generator)
        if kind == "salt_pepper":
            return self._salt_pepper(img, *self.noise_ranges["salt_pepper"], generator)
        if kind == "speckle":
            sigma = self._uniform(*self.noise_ranges["speckle"], generator)
            return img + img * sigma * torch.randn(img.shape, generator=generator)
        if kind == "mixed":
            sigma = self._uniform(*self.noise_ranges["mixed_gaussian"], generator)
            noisy = img + sigma * torch.randn(img.shape, generator=generator)
            return self._salt_pepper(
                noisy, *self.noise_ranges["mixed_salt_pepper"], generator
            )
        raise ValueError(f"Unsupported noise type: {kind}")

    def _salt_pepper(self, img, low, high, generator):
        probability = self._uniform(low, high, generator)
        mask = torch.rand(img.shape, generator=generator)
        noisy = img.clone()
        noisy[mask < probability / 2] = 0
        noisy[mask > 1 - probability / 2] = 1
        return noisy
