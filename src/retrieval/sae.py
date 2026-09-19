"""Top-K SAE reconstruction training with resumable optimizer-step checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def fit_sae(train, validation, spec, seed, checkpoint_dir, stop, device="cpu"):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    d, m, k = train.shape[1], spec["features"], spec["active"]
    if not 0 < k <= m:
        raise ValueError("SAE active features must be in [1, features]")

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(d, m)
            self.decoder = nn.Linear(m, d, bias=False)
            self.center = nn.Parameter(torch.from_numpy(np.mean(train, axis=0).copy()))
            nn.init.kaiming_uniform_(self.decoder.weight)
            with torch.no_grad():
                self.decoder.weight.copy_(nn.functional.normalize(self.decoder.weight, dim=0))
                self.encoder.weight.copy_(self.decoder.weight.T)
                self.encoder.bias.zero_()

        def forward(self, x):
            act = torch.relu(self.encoder(x - self.center))
            ids = torch.argsort(act, descending=True, stable=True, dim=-1)[:, :k]
            codes = torch.zeros_like(act).scatter(1, ids, act.gather(1, ids))
            return self.decoder(codes) + self.center, codes

    model = Autoencoder().to(device)
    lr = float(spec.get("learning_rate", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epochs = int(spec.get("epochs", 3))
    batch = int(spec.get("effective_batch_size", spec.get("batch_size", 4096)))
    micro = int(spec.get("micro_batch_size", 256))
    if epochs <= 0 or batch <= 0 or micro <= 0:
        raise ValueError("SAE epochs and batch sizes must be positive")
    total_steps = epochs * math.ceil(len(train) / batch)
    warmup = max(1, int(total_steps * spec.get("warmup_fraction", 0.05)))
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    checkpoint = directory / "training.pt" if directory else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    epoch, cursor, step, best_loss = 0, 0, 0, float("inf")
    best = None
    history = []
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, cursor, step = saved["epoch"], saved["cursor"], saved["step"]
        best_loss, best, history = saved["best_loss"], saved["best"], saved["history"]

    def save():
        if checkpoint is not None:
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "cursor": cursor, "step": step,
                        "best_loss": best_loss, "best": best, "history": history}, temporary)
            temporary.replace(checkpoint)

    while epoch < epochs:
        permutation = np.random.default_rng(seed + epoch).permutation(len(train))
        model.train()
        while cursor < len(train):
            end = min(cursor + batch, len(train))
            ids = permutation[cursor:end]
            if step < warmup:
                factor = (step + 1) / warmup
            else:
                factor = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(ids), micro):
                x = torch.from_numpy(np.asarray(train[ids[start:start + micro]]).copy()).to(device)
                reconstructed, _ = model(x)
                loss = (reconstructed.float() - x.float()).square().mean()
                (loss * len(x) / len(ids)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                model.decoder.weight.copy_(nn.functional.normalize(model.decoder.weight, dim=0))
            cursor = end
            step += 1
            if step % spec.get("checkpoint_steps", 50) == 0 or (stop is not None and stop.requested):
                save()
            if stop is not None:
                stop.check()
        model.eval()
        error = 0.0
        live = torch.zeros(m, dtype=torch.bool, device=device)
        with torch.inference_mode():
            for start in range(0, len(validation), micro):
                x = torch.from_numpy(np.asarray(validation[start:start + micro]).copy()).to(device)
                reconstructed, codes = model(x)
                error += (reconstructed.float() - x.float()).square().sum().item()
                live |= (codes > 0).any(dim=0)
        error /= len(validation) * d
        history.append({"epoch": epoch + 1, "validation_mse": error,
                        "dead_feature_fraction": float(1 - live.float().mean().item())})
        if error < best_loss:
            best_loss = error
            best = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        epoch += 1
        cursor = 0
        save()
    if best is None:
        raise ValueError("SAE did not produce a validation checkpoint")
    arrays = {"encoder": best["encoder.weight"].numpy(), "bias": best["encoder.bias"].numpy(),
              "decoder": best["decoder.weight"].numpy(), "center": best["center"].numpy()}
    return arrays, {"best_validation_mse": best_loss, "history": history,
                    "optimizer_steps": step, "precision": "float32", "device": device}
