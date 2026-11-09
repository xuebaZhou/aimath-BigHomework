# 2026 秋季人工智能基础数学理论与算法课程作业

本项目使用 MNIST 手写数字数据集，训练条件 IR-SDE 图像去噪模型，并与传统图像处理方法进行比较。

## 方法与比较

实验使用成对的干净图像和含噪观测。MNIST 官方训练集中的 10% 图像被固定划作验证集；官方测试集保留到训练结束后用于最终评估。训练时，模型接收当前 SDE 状态与含噪观测作为条件，预测干净图像（默认设置）或扩散噪声。

给定干净图像 \(x_0\)、含噪观测 \(y\) 和离散时间 \(t\)，前向扰动为：

\[
x_t = y + a_t(x_0-y) + \sqrt{v_t}\,\epsilon,\qquad
a_t=\exp(-\theta t),\qquad
v_t=\lambda^2(1-a_t^2),
\]

其中 \(\epsilon\) 为标准高斯噪声。默认条件 U-Net 直接预测干净图像，并使用均方误差（MSE）作为训练目标，优化器为 AdamW。代码也支持预测噪声的模式。IR-SDE 是概率模型并具有明确的优化目标；高斯基线是局部线性滤波（可视为卷积算子），TV 基线则通过经典变分优化实现。

所有传统方法均在与 IR-SDE 相同的固定测试观测上运行：

| 方法 | 原理与默认参数 |
|---|---|
| Gaussian（高斯滤波） | 高斯平滑，标准差为 0.7 像素，作为简单平滑基线。 |
| Median（中值滤波） | 3×3 次序统计滤波，常用于脉冲噪声。 |
| Bilateral（双边滤波） | 半径 2、灰度标准差 0.1；同时按空间距离和像素灰度差加权，旨在保留边缘。 |
| TV（全变分去噪） | 权重 0.08，Chambolle 迭代 30 次；近似求解 \(0.5\|x-y\|_2^2+\text{weight}\cdot TV(x)\)。 |

报告包括像素保真度指标 MSE、PSNR，以及 Sobel 边缘幅值平均绝对误差（Edge MAE），三项指标均为越低越好。`comparison.png` 和 `comparison_classical.png` 展示干净目标、含噪输入及恢复结果；经典方法对比图的行顺序与上表一致。指标和可视化可辅助比较，但不能单独证明纹理得到保留。

MNIST 是小型灰度数字数据集，并非自然图像基准。因此本项目应视为资源受限的原型，结论仅适用于当前数据集和合成噪声设置。经典基线参数固定，目的是提供透明的参考设置，并不代表经过充分调参的最优比较对象。

## 环境与安装

示例使用启用了 CUDA 的 `pytorch` Conda 环境。先安装项目列出的非 PyTorch 依赖，再单独安装并选择与本机 CUDA 匹配的 PyTorch 和 torchvision：

```bash
conda activate pytorch
pip install -r requirements.txt
```

MNIST 数据集由 torchvision 在首次运行时下载到 `--data-root`（默认为 `./data`）。

## 运行

以下示例将训练数据和测试数据分别限制为 12,000 和 2,000 个样本，以控制首次运行的资源消耗；移除 `--train-samples` 和 `--test-samples` 参数即可使用完整划分。

```bash
python main.py --output-dir "./outputs/ir_sde" --epochs 5 --batch-size 32 --lr 1e-3 --timesteps 100 --lam 0.1 --noise-probs 0.5 0.5 0 0 --inference-steps 50 --samples-per-image 1 --train-samples 12000 --test-samples 2000 --seed 42 --device cuda
```

第二个示例让四种噪声配方等概率出现：

```bash
python main.py --output-dir "./outputs/ir_sde_all" --epochs 5 --batch-size 32 --lr 1e-3 --timesteps 100 --lam 0.1 --noise-probs 0.25 0.25 0.25 0.25 --inference-steps 50 --samples-per-image 1 --train-samples 12000 --test-samples 2000 --seed 42 --device cuda
```

### 常用参数

| 参数 | 说明 |
|---|---|
| `--data-root`、`--output-dir` | 数据集目录（默认 `./data`）和结果目录（默认 `./outputs/ir_sde`）。 |
| `--epochs`、`--batch-size`、`--lr` | 训练轮数、批大小和初始学习率。学习率在所有优化器更新步中按余弦曲线衰减。 |
| `--timesteps`、`--theta`、`--lam` | SDE 离散时间步数、回复速率和噪声尺度。未指定 `--theta` 时取 `5 / timesteps`。 |
| `--prediction-type` | 网络预测目标：`sample`（干净图像，默认）或 `epsilon`（高斯噪声）。 |
| `--noise-probs` | 按 `gaussian salt_pepper speckle mixed` 顺序给出四种噪声概率，必须非负且总和为 1。 |
| `--gaussian-sigma` | 加性高斯噪声标准差范围，默认 `0.05 0.20`。 |
| `--salt-pepper-amount` | 椒盐噪声像素替换比例范围，默认 `0.01 0.08`。 |
| `--inference-steps` | 恢复时使用的采样步数，必须在 1 到 `--timesteps` 之间。 |
| `--train-samples`、`--test-samples` | 可选的数据样本上限；不设置时使用完整数据划分。 |
| `--samples-per-image`、`--test-samples-per-image` | 每张原图生成的训练噪声样本数和测试样本数。 |
| `--eval-every`、`--eval-samples` | 每隔多少轮进行一次固定验证，以及中间评估样本数。 |
| `--seed`、`--device`、`--num-workers` | 随机种子、计算设备和数据加载进程数。设备默认在 CUDA 可用时选择 `cuda`。 |

训练噪声类型按给定概率逐个虚拟样本随机抽取，因此概率定义的是采样分布，不保证整个数据集中的各类样本数量恰好等于该比例。训练噪声默认按访问重新采样；验证和测试噪声固定，以便公平比较。

## 输出文件

结果写入 `--output-dir` 指定的目录，主要包括：

- `metrics.json`：最终 IR-SDE、经典方法及分噪声类别的评估指标。
- `comparison.csv`：各方法的 MSE、PSNR 和 Edge MAE 汇总表。
- `comparison.png`、`comparison_classical.png`：整体测试集恢复结果图。
- `comparison_<噪声类型>.png`、`comparison_classical_<噪声类型>.png`：按噪声类型划分的结果图。
- `comparison_epoch_XXXX.png`：训练期间固定验证子集的阶段性结果图。
- `history.json`、`history.csv`：按 epoch 记录的训练和验证历史。
- `batch_history.json`、`batch_history.csv`：逐训练批次记录的损失、噪声 MSE 和学习率。
- `curves.png`：训练与固定验证指标曲线。
- `checkpoint.pt`：当前检查点，包含模型、优化器、学习率调度器状态及训练历史。
- `config.json`、`fixed_eval.json`：实验配置和固定评估样本信息。
- `unet/`、`scheduler/`：Diffusers 格式保存的网络和调度器配置/权重。

## 测试

在已安装项目依赖的 Python 环境中运行：

```bash
python -m unittest discover -s tests
```
