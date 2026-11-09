"""条件图像预测网络与确定性 IR-SDE 恢复采样器。

观测图像 ``y`` 可以来自 ``NoisyMNISTDataset`` 配置的任意噪声混合。
混合概率只决定训练/评估样本的生成，不改变高斯 SDE 的系数。
"""

import math
from typing import List

import torch
from diffusers import DiffusionPipeline, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput


class IRSdeScheduler(SchedulerMixin, ConfigMixin):
    """实现 IR-SDE 的离散前向扰动与确定性反向跳步。

    时间步为 ``0,1,...,T``，前向条件分布满足
    ``x_t = y + a_t (x_0-y) + sqrt(v_t) epsilon``，
    其中 ``a_t=exp(-theta*t)``、``v_t=lam^2(1-a_t^2)``。
    网络可预测干净图像 ``x_0``（sample）或标准高斯噪声 ``epsilon``。
    """

    @register_to_config
    def __init__(self, T: int = 100, theta: float = 0.05, lam: float = 0.1,
                 prediction_type: str = "epsilon", clip_sample: bool = True):
        if isinstance(T, bool) or not isinstance(T, int) or T < 1:
            raise ValueError("T must be a positive integer.")
        if not all(math.isfinite(value) and value > 0 for value in (theta, lam)):
            raise ValueError("theta and lam must be positive.")
        if theta * T > 10:
            raise ValueError("theta*T must be <= 10 to avoid severe signal loss; use theta=5/T.")
        if prediction_type not in ("epsilon", "sample"):
            raise ValueError("prediction_type must be 'epsilon' or 'sample'")

        # 用 float64 预计算系数，后续按输入张量的数据类型/设备转换。
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float64)
        self.a = torch.exp(-theta * t)  # a_t = exp(-theta * t)
        self.v = lam**2 * (-torch.expm1(-2 * theta * t))  # v_t = lam^2 (1-a_t^2)
        self.train_timesteps = torch.arange(1, T + 1)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: int, y: torch.Tensor):
        """按给定整数时间步构造前向状态 ``x_t``。

        ``t`` 可为单个时间步（广播到整个批次）或长度为 batch_size 的向量；
        时间步 0 留给最终重建，因此训练采样范围是 ``[1,T]``。
        """
        if x0.shape != noise.shape or x0.shape != y.shape:
            raise ValueError("x0, noise and y must have the same shape")
        t = torch.as_tensor(t, device=x0.device)
        if not torch.isfinite(t).all() or (t != t.round()).any():
            raise ValueError("t must contain finite integer timesteps")
        t = t.long().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x0.size(0))
        if t.numel() != x0.size(0) or (t < 1).any() or (t > self.T).any():
            raise ValueError("t 必须为 1,...,T，且长度等于 batch_size（或为标量）")
        # 将每个样本的标量系数扩展为可广播到图像通道/空间维的形状。
        shape = (-1,) + (1,) * (x0.ndim - 1)
        a = self.a[t.cpu()].to(x0).reshape(shape)
        v = self.v[t.cpu()].to(x0).reshape(shape)
        return y + a * (x0 - y) + v.sqrt() * noise

    def prediction_to_x0_noise(self, model_output, sample, t, y, clip=False):
        """将网络输出统一还原为 ``(x0, epsilon)``。

        两种预测目标经此函数转换后可共用同一反向更新公式。若裁剪了 ``x0``，
        会基于裁剪后的结果重新计算噪声，以保持二者与当前 ``sample`` 一致。
        """
        t = torch.as_tensor(t, device=sample.device, dtype=torch.long).reshape(-1)
        shape = (-1,) + (1,) * (sample.ndim - 1)
        a = self.a.to(sample)[t].reshape(shape)
        sigma = self.v.to(sample)[t].sqrt().reshape(shape)
        if self.config.prediction_type == "sample":
            # 网络直接输出干净图像时无需反演。
            x0 = model_output
        else:
            # 根据前向公式解出干净图像估计。
            x0 = y + (sample - y - sigma * model_output) / a
        if clip:
            x0 = x0.clamp(0, 1)
        # Recompute epsilon after clipping, so the DDIM path remains consistent.
        noise = (sample - y - a * (x0 - y)) / sigma
        return x0, noise

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        timesteps: List[int] = None,
        device: torch.device = None,
    ):
        """设置采样时间节点，并为每个节点确定下一跳。

        可在均匀选取的推理步数与自定义严格递减节点列表中二选一；
        自定义列表必须以 ``T`` 开始，所有路径最后都会从最小节点更新到 0。
        """
        if timesteps is not None and num_inference_steps is not None:
            raise ValueError("Specify num_inference_steps or timesteps, not both")
        if timesteps is None:
            steps = self.T if num_inference_steps is None else num_inference_steps
            if not isinstance(steps, int) or not 1 <= steps <= self.T:
                raise ValueError("num_inference_steps 必须是 [1,T] 内的整数")
            # 将 [T,1] 均匀取样并四舍五入为整数离散时间步。
            nodes = torch.linspace(self.T, 1, steps).round().long()
        else:
            nodes = torch.as_tensor(timesteps)
            if not torch.isfinite(nodes).all() or (nodes != nodes.round()).any():
                raise ValueError("timesteps must contain finite integers")
            nodes = nodes.long()
            if nodes.ndim != 1 or nodes.numel() == 0 or nodes[0] != self.T:
                raise ValueError("timesteps 必须是以 T 开始的非空降序整数节点列表")
            if (
                (nodes < 1).any()
                or (nodes > self.T).any()
                or (nodes[:-1] <= nodes[1:]).any()
            ):
                raise ValueError("timesteps 必须严格递减，且每个节点在 [1,T] 内")
        # prev_timesteps 的末项为 0，确保采样最终回到观测中心附近的干净状态。
        self.timesteps = nodes.to(device)
        self.prev_timesteps = torch.cat(
            (self.timesteps[1:], self.timesteps.new_zeros(1))
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        y: torch.Tensor,
        prev_timestep: int,
    ):
        """依据网络预测将状态从时间 ``i`` 确定性地跳转到 ``j``。

        允许跳过中间时间步（``j < i-1``）：先恢复一致的干净图像与噪声估计，
        再使用时间 ``j`` 对应的 ``a_j``、``v_j`` 重新组合条件状态。
        """
        if any(not math.isfinite(float(t)) or float(t) != int(t)
               for t in (timestep, prev_timestep)):
            raise ValueError("Sampling timesteps must be finite integers")
        i, j = int(timestep), int(prev_timestep)
        if not 0 <= j < i <= self.T:
            raise ValueError("Require 0 <= prev_timestep < timestep <= T")
        if sample.shape != y.shape or sample.shape != model_output.shape:
            raise ValueError("sample, y and model_output must have the same shape")
        x0, noise = self.prediction_to_x0_noise(
            model_output, sample, i, y, clip=self.config.clip_sample
        )
        return SchedulerOutput(
            prev_sample=y + self.a[j].item() * (x0 - y)
            + self.v[j].sqrt().item() * noise
        )


def make_mnist_unet():
    """构造适用于 28×28 MNIST 的条件 U-Net。

    输入为沿通道拼接的 ``[x_t,y]``（2 个通道），输出为 1 个通道；
    输出究竟表示干净图像还是噪声由 scheduler 的 ``prediction_type`` 决定。
    """
    return UNet2DModel(
        sample_size=28,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64),
        norm_num_groups=8,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


class IRSdePipeline(DiffusionPipeline):
    """用 IR-SDE 条件初始化和跳步公式执行图像恢复。"""

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: IRSdeScheduler):
        """注册网络和调度器，使 Diffusers 管线可统一管理设备及保存流程。"""
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        y,
        num_inference_steps: int = None,
        timesteps: List[int] = None,
        generator: torch.Generator = None,
        output_type: str = "pt",
    ):
        """从含噪观测 ``y`` 生成恢复图像。

        ``y`` 的形状为 ``[B,1,H,W]``。采样由终端状态 ``x_T`` 初始化，
        随后在 scheduler 给出的时间节点上反复预测并更新；输出支持张量、
        NumPy 数组或 PIL 图像。
        """
        if output_type not in ("pt", "np", "pil"):
            raise ValueError("output_type 只能为 pt、np 或 pil")
        if y.ndim != 4 or y.shape[0] < 1 or y.shape[1] != 1:
            raise ValueError("y must have shape [B, 1, H, W] with B > 0")
        if not torch.isfinite(y).all():
            raise ValueError("y must contain finite pixels")
        # 统一网络所需设备和精度；模型处于推理模式并关闭梯度计算。
        self.unet.eval()
        y = y.to(device=self._execution_device, dtype=self.unet.dtype)
        self.scheduler.set_timesteps(num_inference_steps, timesteps, device=y.device)

        # x_T 近似服从 N(y, v_T I)，只有 a_T 很小时近似才好。
        noise = torch.randn(
            y.shape, device=y.device, dtype=y.dtype, generator=generator
        )
        x = y + self.scheduler.v[-1].sqrt().item() * noise

        # 每次迭代都把当前状态与固定观测拼接，作为条件 U-Net 的输入。
        for t, prev_t in self.progress_bar(
            zip(self.scheduler.timesteps, self.scheduler.prev_timesteps),
            total=len(self.scheduler.timesteps),
        ):
            prediction = self.unet(torch.cat([x, y], dim=1), t).sample
            x = self.scheduler.step(prediction, t, x, y, prev_t).prev_sample

        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite sampling result before clamping")
        x = x.clamp(0, 1)  # MNIST 数据范围 [0,1]
        if output_type != "pt":
            x = x.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                x = self.numpy_to_pil(x)
        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=x)
