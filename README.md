# 2026_autumn_aimath_coursework
Course work for Foundational Mathematics and Algorithms for Artificial Intelligence, autumn, 2026.

## Method and comparison

The current experiment uses paired clean/noisy MNIST images. Ten percent of the original MNIST training images are reserved for a fixed validation subset; the official MNIST test split is kept for final evaluation. IR-SDE perturbs a clean image toward its noisy observation as $x_t = y + a_t (x_0-y) + \sqrt{v_t}\,\epsilon$, where $a_t = \exp(-\theta t)$ and $v_t = \lambda^2 (1-a_t^2)$. The conditional UNet predicts the clean image by default, minimizing mean-squared error with AdamW. This provides a probabilistic model and an explicit optimization objective. The Gaussian baseline is a local linear operator ($\hat{x} = Gy$, with $G$ representing convolution); the TV baseline is a classical variational optimization method. The same fixed test observations are also restored with four non-learned computer-vision baselines:

| Method | Principle |
|---|---|
| Gaussian | Gaussian filtering with sigma 0.7 pixels; simple smoothing baseline. |
| Median | 3x3 order-statistic filter; commonly used for impulse noise. |
| Bilateral | Radius 2, intensity sigma 0.1; spatial and intensity-weighted averaging intended to retain edges. |
| TV | Weight 0.08 and 30 Chambolle iterations; approximately minimizes \(0.5||x-y||_2^2 + \text{weight} \cdot \mathrm{TV}(x)\). |

The report includes MSE and PSNR for pixel fidelity, plus Sobel edge-magnitude MAE as a limited edge-preservation indicator. Lower is better for all three metrics. `comparison.png` and `comparison_classical.png` show clean targets, noisy inputs, and restored outputs; rows in the latter follow the method order above. These metrics and visualizations support comparison but do not by themselves prove texture preservation.

MNIST is a small grayscale-digit benchmark, not a natural-image benchmark. Treat it as a resource-conscious prototype; conclusions should be limited to this dataset and the configured synthetic noise. The fixed default baseline parameters are intended as transparent reference settings, not as optimally tuned competitors.

## Run

Use the CUDA-enabled `pytorch` Conda environment. The example limits the data to keep a first run manageable on a 4 GB GPU; remove `--train-samples` and `--test-samples` to use the full MNIST splits. Install the non-PyTorch dependencies with `pip install -r requirements.txt`; install/select a CUDA-enabled PyTorch and matching torchvision separately.

```bash
conda activate pytorch
pip install -r requirements.txt
```

```bash
python main.py --output-dir "./outputs/ir_sde" --epochs 5 --batch-size 32 --lr 1e-3 --timesteps 100 --lam 0.1 --noise-probs 0.5 0.5 0 0 --inference-steps 50 --samples-per-image 1 --train-samples 12000 --test-samples 2000 --seed 42 --device cuda
```

```bash
python main.py --output-dir "./outputs/ir_sde_all" --epochs 5 --batch-size 32 --lr 1e-3 --timesteps 100 --lam 0.1 --noise-probs 0.25 0.25 0.25 0.25 --inference-steps 50 --samples-per-image 1 --train-samples 12000 --test-samples 2000 --seed 42 --device cuda
```

`--eval-samples` controls the fixed subset used for intermediate and per-noise analysis. Final aggregate metrics compare IR-SDE and all classical methods on the same held-out test split. Results include `metrics.json`, `comparison.csv`, `comparison.png`, `comparison_classical.png`, per-noise comparison images, training histories, and a model checkpoint.
