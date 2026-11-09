"""按需生成带噪 MNIST 样本，并支持配置多种噪声的混合比例。"""

import math

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class NoisyMNISTDataset(Dataset):
    """返回 ``(含噪图像, 干净图像)``，两者像素值均在 ``[0,1]``。

    训练数据默认每次读取时重新采样噪声；确定性模式使用 ``seed + 索引`` 作为
    随机种子，因此同一虚拟索引可复现相同扰动，且无需缓存全部噪声图像。
    每张原图可通过 ``samples_per_image`` 扩展为多个独立扰动的虚拟样本。
    噪声概率用于逐个虚拟样本随机选择噪声配方，并不保证整套数据中的精确数量。
    """

    # 可选的扰动配方；mixed 表示叠加高斯噪声和椒盐噪声。
    NOISE_TYPES = ("gaussian", "salt_pepper", "speckle", "mixed")
    # 未指定时四种噪声等概率抽样。
    DEFAULT_NOISE_PROBS = {
        "gaussian": 0.25,
        "salt_pepper": 0.25,
        "speckle": 0.25,
        "mixed": 0.25,
    }
    # 各噪声强度的默认闭区间；椒盐噪声范围表示被替换像素的总比例。
    DEFAULT_NOISE_RANGES = {
        "gaussian": (0.05, 0.20),
        "salt_pepper": (0.01, 0.08),
        "speckle": (0.10, 0.30),
        "mixed_gaussian": (0.03, 0.12),
        "mixed_salt_pepper": (0.01, 0.04),
    }

    @classmethod
    def validate_noise_probs(cls, noise_probs=None):
        """校验混合噪声概率并补齐未提供类型的零概率。

        概率之和允许 ``1e-6`` 的浮点误差，最终会归一化为严格和为 1 的分布。
        """
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
            """初始化 MNIST 数据集和按需扰动配置。

            ``train`` 决定使用 MNIST 训练划分还是官方测试划分。默认情况下训练集
            每次随机生成噪声，测试集则使用可复现噪声；调用者可以显式覆盖该行为。
            """
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
        # 同一原图在虚拟数据集中的连续 samples_per_image 个索引对应多个噪声样本。
        self.samples_per_image = samples_per_image
        self.train = train
        self.deterministic_noise = not train if deterministic_noise is None else deterministic_noise
        self.seed = seed
        # 复制默认范围后再覆盖，避免修改类级默认配置。
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
        # torch.multinomial 使用的类别顺序与名称列表保持一致。
        self.noise_probs = self.validate_noise_probs(noise_probs)
        weights = list(self.noise_probs.values())
        self.noise_names = list(self.noise_probs)
        self.noise_weights = torch.tensor(weights, dtype=torch.float64)
        self.data = MNIST(root=root, train=train, download=True, transform=ToTensor())

    def __len__(self):
        """返回展开虚拟样本后的数据集长度。"""
        return len(self.data) * self.samples_per_image

    def __getitem__(self, idx: int):
        """读取一个干净样本并按所选配方生成噪声观测。

        负索引遵循 Python 序列语义；越界索引显式报错。确定性模式下随机生成器
        与虚拟索引绑定，从而既可复现，也可让同一原图的多个虚拟槽拥有独立噪声。
        """
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
        """在给定闭区间内抽取一个标量，允许区间两端相同以固定强度。"""
        return low + (high - low) * torch.rand((), generator=generator).item()

    def _corrupt(self, img, kind, generator):
        """根据 ``kind`` 选择扰动公式，返回尚未裁剪的含噪图像。"""
        if kind == "gaussian":
            # 加性高斯噪声：每个像素独立采样 N(0, sigma^2)。
            sigma = self._uniform(*self.noise_ranges["gaussian"], generator)
            return img + sigma * torch.randn(img.shape, generator=generator)
        if kind == "salt_pepper":
            return self._salt_pepper(img, *self.noise_ranges["salt_pepper"], generator)
        if kind == "speckle":
            # 乘性斑点噪声：噪声幅度随原始像素值变化。
            sigma = self._uniform(*self.noise_ranges["speckle"], generator)
            return img + img * sigma * torch.randn(img.shape, generator=generator)
        if kind == "mixed":
            # 混合配方先添加高斯噪声，再按独立概率替换为椒盐值。
            sigma = self._uniform(*self.noise_ranges["mixed_gaussian"], generator)
            noisy = img + sigma * torch.randn(img.shape, generator=generator)
            return self._salt_pepper(
                noisy, *self.noise_ranges["mixed_salt_pepper"], generator
            )
        raise ValueError(f"Unsupported noise type: {kind}")

    def _salt_pepper(self, img, low, high, generator):
        """以随机总比例将像素替换为黑色或白色。

        每个像素落入盐/椒区域的概率各为总替换比例的一半；其余像素保持原值。
        """
        probability = self._uniform(low, high, generator)
        mask = torch.rand(img.shape, generator=generator)
        noisy = img.clone()
        noisy[mask < probability / 2] = 0
        noisy[mask > 1 - probability / 2] = 1
        return noisy
