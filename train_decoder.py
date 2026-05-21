"""Train a VisualDecoder on extracted CLS tokens from a trained LeWM checkpoint.

Usage:
    PYTHONPATH=.. uv run python train_decoder.py \\
        data=waam \\
        data.dataset.path=../output/datagen/.../stage5_full.h5 \\
        ckpt=~/.stable_worldmodel/checkpoints/lewm_epoch_100_object.ckpt \\
        decoder.epochs=50

Extracts CLS token embeddings from the training split and trains
the decoder with MSE reconstruction loss. The decoder is never
trained jointly with the world model — see LeWorldModel App. D.
"""
from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from probes import VisualDecoder


def extract_cls_pixel_pairs(model, dataset, device="cpu", batch_size=128):
    """Extract (cls_token, pixels) pairs from the training split.

    Args:
        model: LeWM model with ``encode(batch)`` method.
        dataset: Training split dataset returning ``pixels`` key.
        device: Torch device.
        batch_size: Extraction batch size.

    Returns:
        Tuple of ``(cls_tokens, pixel_images)`` tensors.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_cls = []
    all_pixels = []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixels"].to(device)
            B, T = pixels.shape[:2]
            batch["pixels"] = pixels
            info = model.encode(batch)
            emb = info["emb"]  # [B, T, D]
            # Use CLS token from the first timestep per sample
            cls = emb[:, 0, :].cpu()  # [B, D]
            all_cls.append(cls)
            # Store the corresponding pixel image (first timestep, [3, H, W])
            img = pixels[:, 0, ...].cpu()  # [B, 3, H, W]
            all_pixels.append(img)

    return torch.cat(all_cls, dim=0), torch.cat(all_pixels, dim=0)


def train_decoder(
    decoder,
    cls_tokens,
    pixel_images,
    *,
    epochs=50,
    lr=1e-3,
    batch_size=64,
    val_split=0.1,
):
    """Train the decoder on extracted CLS-pixel pairs.

    Args:
        decoder: VisualDecoder instance.
        cls_tokens: Extracted CLS embeddings [N, D].
        pixel_images: Target images [N, 3, H, W].
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        val_split: Fraction of data for validation.

    Returns:
        Dict with ``train_losses`` and ``val_losses`` per epoch.
    """
    N = cls_tokens.size(0)
    n_val = max(int(N * val_split), 4)
    n_train = N - n_val
    perm = torch.randperm(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_cls, train_pix = cls_tokens[train_idx], pixel_images[train_idx]
    val_cls, val_pix = cls_tokens[val_idx], pixel_images[val_idx]

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        decoder.train()
        epoch_loss = 0.0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            preds = decoder(train_cls[idx])
            loss = F.mse_loss(preds, train_pix[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()
        train_losses.append(epoch_loss / n_train)

        decoder.eval()
        with torch.inference_mode():
            val_preds = decoder(val_cls)
            val_loss = float(F.mse_loss(val_preds, val_pix))
            val_losses.append(val_loss)

    return {"train_losses": train_losses, "val_losses": val_losses}


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    """Load checkpoint, extract CLS tokens, train decoder, save weights."""
    import stable_worldmodel as swm

    print("Loading checkpoint...")
    ckpt_path = Path(cfg.get("ckpt", "")).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load the model: .ckpt files are full torch.save(model) checkpoints;
    # folders are load_pretrained-compatible (weights.pt + config.json).
    if ckpt_path.is_file():
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # _object.ckpt is a spt.Module wrapping JEPA; extract the .model
        model = loaded.model if hasattr(loaded, "model") else loaded
    else:
        if not ckpt_path.suffix:
            ckpt_path = ckpt_path / "weights.pt"
        model = swm.utils.load_pretrained(
            str(ckpt_path.parent) if ckpt_path.suffix == ".pt" else str(ckpt_path)
        )
    model = model.to(cfg.get("device", "cuda"))
    model.eval()

    if cfg.data.dataset.get("type") == "waam":
        from src.dataloader.waam_dataset import WaamFlatDataset

        data_path = Path(cfg.data.dataset.path)
        if not data_path.is_absolute():
            data_path = Path(os.getcwd()) / data_path
        dataset = WaamFlatDataset(
            path=str(data_path.expanduser().resolve()),
            frameskip=cfg.data.dataset.get("frameskip", 1),
            num_steps=cfg.data.dataset.get("num_steps", 1),
            keys_to_load=["pixels"],
        )
    else:
        dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, _ = swm.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    print(f"Extracting CLS tokens from {len(train_set)} training samples...")
    cls_tokens, pixel_images = extract_cls_pixel_pairs(model, train_set, device="cuda")
    print(f"Extracted {cls_tokens.size(0)} pairs. CLS shape: {cls_tokens.shape}, pixels: {pixel_images.shape}")

    decoder = VisualDecoder(
        embed_dim=cfg.wm.embed_dim,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        depth=cfg.get("decoder", {}).get("depth", 2),
        heads=cfg.get("decoder", {}).get("heads", 8),
        dim_head=cfg.get("decoder", {}).get("dim_head", 64),
    ).to(cls_tokens.device)

    dc_cfg = cfg.get("decoder", {})
    results = train_decoder(
        decoder,
        cls_tokens,
        pixel_images,
        epochs=dc_cfg.get("epochs", 50),
        lr=dc_cfg.get("lr", 1e-3),
        batch_size=dc_cfg.get("batch_size", 64),
        val_split=dc_cfg.get("val_split", 0.1),
    )

    print(f"Final train MSE: {results['train_losses'][-1]:.6f}")
    print(f"Final val MSE: {results['val_losses'][-1]:.6f}")

    save_path = Path(cfg.get("subdir", ".")) / "decoder_weights.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), save_path)
    print(f"Saved decoder weights to {save_path}")


if __name__ == "__main__":
    run()
