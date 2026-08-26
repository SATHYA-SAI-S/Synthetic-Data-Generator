"""
Adaptive single-config DP-SGD training for the Kaggle kernel route.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def run_adaptive_training(
    clean_csv_path: str,
    output_dir: str,
    epsilon: float = 1.0,
    delta: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 256,
    clip_norm: float = 1.0,
    num_samples: int = -1,
    clean_columns: list = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_progress(**kw):
        payload = {
            "ts": time.time(),
            "stage": kw.get("stage", "running"),
            "pct": float(kw.get("pct", 0.0)),
            "epoch": int(kw.get("epoch", 0)),
            "total_epochs": int(kw.get("total_epochs", epochs)),
            "loss": float(kw.get("loss", 0.0)),
            "epsilon_spent": float(kw.get("epsilon_spent", 0.0)),
        }
        tmp = output_dir / "progress.json.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(output_dir / "progress.json")

    raw_df = pd.read_csv(clean_csv_path)
    if clean_columns:
        df = raw_df[[c for c in clean_columns if c in raw_df.columns]]
    else:
        df = raw_df

    n_rows = len(df)
    if num_samples and num_samples > 0:
        n_rows = int(num_samples)
    if n_rows <= 0:
        raise ValueError("num_samples must be positive")

    from src.diffusion.schema_adapter import TabularEncoder
    from src.diffusion.denoiser import MLPDenoiser

    enc = TabularEncoder(df)
    X = enc.encode(df)
    dim = enc.dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    denoiser = MLPDenoiser(input_dim=dim, hidden_dims=[256, 256, 256],
                           num_timesteps=50).to(device)

    from src.privacy.accountant import CentralPrivacyAccountant
    
    noise_multiplier = max(1.0, 5.0 / max(float(epsilon), 0.1))
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=2e-4)
    
    sample_rate = float(batch_size) / max(float(n_rows), 1.0)
    accountant = CentralPrivacyAccountant()

    ds = torch.from_numpy(X)
    loader = torch.utils.data.DataLoader(
        TensorDataset(ds), batch_size=int(batch_size), shuffle=True)

    write_progress(stage="DP-SGD Training", pct=5, epoch=0, loss=0.0,
                   epsilon_spent=0.0)

    losses = []
    eps_spent = 0.0
    total_epochs = max(1, int(epochs))
    for ep in range(total_epochs):
        denoiser.train()
        ep_loss, nb = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            t = torch.rand(len(xb), device=device) * 0.999 + 0.001
            noise = torch.randn_like(xb) * t.sqrt().view(-1, 1)
            target = noise
            x_noisy = xb + noise
            pred = denoiser(x_noisy, t)
            loss = ((pred - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            total_norm = 0.0
            for p in denoiser.parameters():
                if p.grad is None:
                    continue
                total_norm += float(p.grad.detach().norm() ** 2)
            norm = float(np.sqrt(total_norm))
            scale = min(1.0, float(clip_norm) / max(norm, 1e-12))
            for p in denoiser.parameters():
                if p.grad is None:
                    continue
                noise_t = torch.randn_like(p.grad) * noise_multiplier * clip_norm
                p.grad.mul_(scale).add_(noise_t)
            optimizer.step()
            ep_loss += float(loss.item())
            nb += 1

        avg = ep_loss / max(nb, 1)
        losses.append(avg)
        
        for _ in range(nb):
            accountant.record_step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        eps_spent = accountant.get_epsilon(target_delta=float(delta))
        
        print(f"Epoch {ep + 1}/{total_epochs} - Loss: {avg:.4f} - Epsilon Spent: {eps_spent:.4f}", flush=True)
        
        write_progress(stage="DP-SGD Training",
                       pct=5 + int(88 * (ep + 1) / total_epochs),
                       epoch=ep + 1, total_epochs=total_epochs, loss=avg,
                       epsilon_spent=eps_spent)

    write_progress(stage="Synthetic Generation", pct=95, loss=float(losses[-1]),
                   epsilon_spent=eps_spent)
    denoiser.eval()
    with torch.no_grad():
        x = torch.randn(n_rows, dim, device=device)
        T = 50
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, 0)
        for step in reversed(range(T)):
            t = torch.full((n_rows,), (step + 1) / T, device=device)
            pred_noise = denoiser(x, t)
            alpha, abar = alphas[step], acp[step]
            mean = (x - betas[step] / torch.sqrt(1 - abar) * pred_noise) / torch.sqrt(alpha)
            if step > 0:
                mean += torch.sqrt(betas[step]) * torch.randn_like(mean)
            x = mean
        synthetic_tensor = x.cpu().numpy()

    synth_df = enc.decode(synthetic_tensor)

    for c in enc.numeric_cols:
        lo, hi = float(df[c].min()), float(df[c].max())
        synth_df[c] = synth_df[c].clip(lo, hi)
        if (df[c] % 1 == 0).all():
            synth_df[c] = synth_df[c].round().astype(int)

    out_csv = output_dir / "synthetic_clean.csv"
    synth_df.to_csv(out_csv, index=False)

    report = {
        "target_epsilon": float(epsilon),
        "target_delta": float(delta),
        "actual_epsilon_spent": float(eps_spent),
        "final_loss": float(losses[-1]) if losses else None,
        "rows_generated": int(len(synth_df)),
        "columns": [str(c) for c in synth_df.columns],
        "batch_size": int(batch_size),
        "epochs": int(total_epochs),
        "noise_multiplier": float(noise_multiplier),
        "clip_norm": float(clip_norm),
    }
    (output_dir / "output_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    write_progress(stage="complete", pct=100, loss=float(losses[-1]),
                   epsilon_spent=eps_spent)
    print(f"Wrote {out_csv} ({len(synth_df):,} rows)")
    return synth_df