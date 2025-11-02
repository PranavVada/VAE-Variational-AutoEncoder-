[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/REPO_USER/REPO_NAME/blob/main/VAE_Colab_Annotated.ipynb)

# Variational Autoencoder (VAE) in PyTorch

A clean, Colab-ready notebook implementing a Variational Autoencoder (VAE) with a standard ELBO objective (reconstruction + KL). This repository is structured for GitHub with clear sections, comments, and instructions.

## Highlights
- **Dataset**: CIFAR10  
- **Framework**: PyTorch + torchvision
- **Notebook**: `VAE_Colab_Annotated.ipynb` (non-executed refactor — results remain unchanged)
- **Loss**: ELBO = reconstruction loss + KL divergence
- **Reconstruction Loss**: MSE (set to `reduction="sum"` inside ELBO)
- **KL**: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`

## Quick Start (Colab)
1. Open the notebook in Google Colab.
2. (Optional) Mount Drive if you want to save models/outputs.
3. Run all cells.

## Local Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## File Structure
```
.
├── VAE_Colab_Annotated.ipynb   # Cleaned notebook with headings
├── requirements.txt
└── README.md
```

## ELBO (Objective)
Let \(x\) be the input, \(\hat{x}\) the reconstruction, and \(q_\phi(z|x)\sim \mathcal{N}(\mu, \sigma^2)\).
The Evidence Lower Bound (ELBO) is:

\[
\mathcal{L}(x) = \underbrace{\|x-\hat{x}\|_2^2}_\text{reconstruction} + \beta\,\underbrace{D_\mathrm{KL}(\mathcal{N}(\mu,\sigma^2)\;\|\;\mathcal{N}(0, I))}_\text{regularization}
\]

- In the notebook, \(\beta=1\) by default. You can try **β-VAE** by multiplying the KL term by \(\beta>1\).

## Tips
- Keep loss terms on the **same scale** (we average per batch; reconstruction uses `reduction="sum"` then divided by `B`).
- Monitor both train and validation ELBO.
- Visualize reconstructions every few epochs.
- Try different latent dimensions (e.g., 2, 8, 16, 32).

## License
MIT

## Scripts
- `train_vae.py` — CLI training script for VAE with ELBO, supports MNIST/FashionMNIST/CIFAR10 via torchvision.
  - Example:
    ```bash
    python train_vae.py --dataset MNIST --epochs 20 --batch-size 128 --latent-dim 16 --beta 1.0
    ```

## Suggested Repo Structure
```
.
├── notebooks/
│   └── VAE_Colab_Annotated.ipynb
├── src/
│   └── train_vae.py
├── results/
│   ├── samples/
│   └── checkpoints/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```
> You can keep files at the repo root if you prefer — the script defaults will create `results/` when needed.
