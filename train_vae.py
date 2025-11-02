#!/usr/bin/env python
import argparse, os, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

# --------------------
# Utils
# --------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Data
# --------------------
def get_dataset(name="MNIST", root="./data", train=True, download=True):
    name = name.lower()
    tfm = transforms.Compose([
        transforms.ToTensor()
    ])
    if name == "mnist":
        return datasets.MNIST(root, train=train, download=download, transform=tfm)
    if name == "fashionmnist":
        return datasets.FashionMNIST(root, train=train, download=download, transform=tfm)
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, download=download, transform=tfm)
    raise ValueError(f"Unsupported dataset: {name}")

# --------------------
# Model: simple MLP VAE on flattened input
# --------------------
class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim=16, hidden_dim=512):
        super().__init__()
        self.in_dim = in_dim
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),  # for images in [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

# --------------------
# ELBO
# --------------------
def elbo(x, xhat, mu, logvar, beta=1.0, reduction="sum"):
    # Reconstruction (MSE) with sum reduction to match classic VAE scaling,
    # divided by batch size outside (to get per-sample average).
    if reduction == "sum":
        recon = F.mse_loss(xhat, x, reduction="sum")
    else:
        recon = F.mse_loss(xhat, x, reduction="mean") * x.numel() / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

# --------------------
# Train / Eval
# --------------------
@torch.no_grad()
def save_reconstructions(model, data, outdir, step, num=16):
    model.eval()
    x, _ = next(iter(data))
    x = x.to(next(model.parameters()).device)
    b = min(num, x.size(0))
    flat = x.view(b, -1)
    xhat, _, _ = model(flat)
    # reshape
    c = x.size(1)
    h = x.size(2)
    w = x.size(3)
    xhat = xhat.view(b, c, h, w)
    grid_real = vutils.make_grid(x[:b], nrow=int(math.sqrt(b)))
    grid_fake = vutils.make_grid(xhat[:b], nrow=int(math.sqrt(b)))
    os.makedirs(outdir, exist_ok=True)
    vutils.save_image(grid_real, os.path.join(outdir, f"real_{step:06d}.png"))
    vutils.save_image(grid_fake, os.path.join(outdir, f"recon_{step:06d}.png"))

def run(args):
    set_seed(args.seed)
    device = get_device()

    train_ds = get_dataset(args.dataset, root=args.data_root, train=True, download=True)
    val_ds = get_dataset(args.dataset, root=args.data_root, train=False, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # infer dims
    sample, _ = train_ds[0]
    c, h, w = sample.shape
    in_dim = c * h * w

    model = VAE(in_dim=in_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            b = x.size(0)
            flat = x.view(b, -1)

            xhat, mu, logvar = model(flat)
            loss, recon, kl = elbo(flat, xhat, mu, logvar, beta=args.beta, reduction="sum")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_loss += loss.item() / b
            train_recon += recon.item() / b
            train_kl += kl.item() / b
            global_step += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                b = x.size(0)
                flat = x.view(b, -1)
                xhat, mu, logvar = model(flat)
                loss, recon, kl = elbo(flat, xhat, mu, logvar, beta=args.beta, reduction="sum")
                val_loss += loss.item() / b
                val_recon += recon.item() / b
                val_kl += kl.item() / b

        print(f"Epoch {epoch:03d} | train: ELBO {train_loss/len(train_loader):.2f}, recon {train_recon/len(train_loader):.2f}, KL {train_kl/len(train_loader):.2f} | "
              f"val: ELBO {val_loss/len(val_loader):.2f}, recon {val_recon/len(val_loader):.2f}, KL {val_kl/len(val_loader):.2f}")

        if epoch % args.save_every == 0:
            # Save reconstructions
            save_reconstructions(model, val_loader, args.samples_dir, global_step, num=16)
            # Save checkpoint
            ckpt_path = os.path.join(args.ckpt_dir, f"vae_epoch{epoch:03d}.pt")
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "in_dim": in_dim,
                "shape": (c,h,w),
            }, ckpt_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a simple VAE (MLP) with ELBO.")
    p.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "CIFAR10"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0, help="beta-VAE scaling on KL term")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", type=str, default="./results/checkpoints")
    p.add_argument("--samples-dir", type=str, default="./results/samples")
    p.add_argument("--save-every", type=int, default=1)
    args = p.parse_args()
    run(args)
