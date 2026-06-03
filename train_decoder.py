"""Train a VisualDecoder on extracted CLS tokens from a trained LeWM checkpoint.

Usage:
    PYTHONPATH=.. uv run python train_decoder.py \\
        --dataset ../output/datagen/.../stage5_full.h5 \\
        --ckpt ~/.stable_worldmodel/lewm_epoch_47_object.ckpt \\
        --epochs 50

Extracts CLS token embeddings from the training split and trains
the decoder with MSE reconstruction loss.  Logs training curves
and sample reconstructions to W&B.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from probes import VisualDecoder


def extract_cls_pixel_pairs(model, dataset, device="cpu", batch_size=128, max_frames=None):
    """Extract (cls_token, pixels) pairs from a dataset.

    Args:
        max_frames: If set, stop after extracting this many frames
            (prevents OOM on datasets with millions of frames).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_cls = []
    all_pixels = []
    total = len(loader)
    frame_count = 0

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            info = {"pixels": batch["pixels"].to(device)}
            info = model.encode(info)
            emb = info["emb"]
            cls = emb[:, 0, :].cpu()
            all_cls.append(cls)
            img = batch["pixels"][:, 0, ...].cpu()
            all_pixels.append(img)
            frame_count += img.shape[0]

            if i % 10 == 0 or i == total - 1:
                print(f"  CLS extraction  batch {i+1}/{total}  ({100*(i+1)//total}%)  "
                      f"frames={frame_count}")

            if max_frames is not None and frame_count >= max_frames:
                print(f"  (stopped at max_frames={max_frames})")
                break

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

    Logs train/val MSE and sample reconstructions to W&B if a run is active.
    """
    device = next(decoder.parameters()).device

    N = cls_tokens.size(0)
    n_val = max(int(N * val_split), 4)
    n_train = N - n_val
    perm = torch.randperm(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_cls = cls_tokens[train_idx].to(device)
    train_pix = pixel_images[train_idx]
    val_cls = cls_tokens[val_idx].to(device)
    val_pix = pixel_images[val_idx]

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []

    try:
        import wandb
        _has_wandb = wandb.run is not None
    except ImportError:
        _has_wandb = False

    for epoch in range(epochs):
        decoder.train()
        epoch_loss = 0.0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            preds = decoder(train_cls[idx])
            loss = F.mse_loss(preds, train_pix[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()
        avg_train = epoch_loss / n_train
        train_losses.append(avg_train)

        decoder.eval()
        with torch.inference_mode():
            # Batched validation — avoid OOM from processing all val samples at once.
            val_preds = []
            for i in range(0, n_val, batch_size):
                c = val_cls[i : i + batch_size]
                val_preds.append(decoder(c).cpu())
            val_preds = torch.cat(val_preds, dim=0)
            val_loss = float(F.mse_loss(val_preds, val_pix))
            val_losses.append(val_loss)

        if _has_wandb:
            wandb.log({"decoder/train_mse": avg_train, "decoder/val_mse": val_loss,
                        "decoder/epoch": epoch, "decoder/lr": scheduler.get_last_lr()[0]})

        if epoch % 10 == 0 or epoch == epochs - 1:
            msg = f"epoch {epoch+1:3d}/{epochs}  train_mse={avg_train:.6f}  val_mse={val_loss:.6f}"
            if _has_wandb and epoch % 10 == 0:
                with torch.inference_mode():
                    sample_recon = decoder(val_cls[:4])
                    wandb.log({"decoder/reconstructions": [
                        wandb.Image(img, caption=f"sample_{i}")
                        for i, img in enumerate(sample_recon)
                    ]}, commit=False)
            print(msg)

    return {"train_losses": train_losses, "val_losses": val_losses}


def main():
    parser = argparse.ArgumentParser(description="Train VisualDecoder on LeWM CLS tokens")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to WAAM HDF5 dataset")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to LeWM checkpoint (*_object.ckpt or training output dir)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--decoder-dim-head", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="lwm-waam")
    parser.add_argument("--wandb-entity", type=str, default="fsandco")
    parser.add_argument("--wandb-name", type=str, default="decoder")
    parser.add_argument("--max-frames", type=int, default=20000,
                        help="Max frames for CLS extraction (default 20K, ~3.8 GB pixels). "
                             "Set higher if you have more RAM.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for decoder_weights.pt (default: decoder_weights.pt in cwd)")
    args = parser.parse_args()

    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt_path = args.ckpt.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    if ckpt_path.is_file():
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = loaded.model if hasattr(loaded, "model") else loaded
    else:
        import stable_worldmodel as swm
        model = swm.utils.load_pretrained(str(ckpt_path))
    model = model.to(args.device)
    model.eval()

    # ── Dataset ──────────────────────────────────────────────────────
    from src.dataloader.waam_dataset import WaamFlatDataset

    data_path = args.dataset.expanduser().resolve()
    dataset = WaamFlatDataset(
        path=str(data_path),
        frameskip=1,
        num_steps=1,
        keys_to_load=["pixels"],
    )
    import stable_pretraining as spt

    rnd_gen = torch.Generator().manual_seed(42)
    train_set, _ = spt.data.random_split(
        dataset, lengths=[0.9, 0.1], generator=rnd_gen,
    )
    print(f"Extracting CLS tokens from {len(train_set)} training samples...")
    cls_tokens, pixel_images = extract_cls_pixel_pairs(
        model, train_set, device=args.device, max_frames=args.max_frames)
    print(f"Extracted {cls_tokens.size(0)} pairs. CLS: {cls_tokens.shape}, Pixels: {pixel_images.shape}")

    # ── Free JEPA model to make room for decoder on GPU ────────────────
    model.cpu()
    del model
    torch.cuda.empty_cache()

    # ── Build decoder ────────────────────────────────────────────────
    decoder = VisualDecoder(
        embed_dim=args.embed_dim,
        img_size=args.img_size,
        patch_size=args.patch_size,
        depth=args.decoder_depth,
        heads=args.decoder_heads,
        dim_head=args.decoder_dim_head,
    ).to(args.device)

    # ── W&B init ─────────────────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )

    # ── Train ────────────────────────────────────────────────────────
    results = train_decoder(
        decoder, cls_tokens, pixel_images,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_split=args.val_split,
    )

    print(f"Final train MSE: {results['train_losses'][-1]:.6f}")
    print(f"Final val MSE: {results['val_losses'][-1]:.6f}")

    # ── Save ─────────────────────────────────────────────────────────
    save_path = (args.output or Path("decoder_weights.pt")).resolve()
    torch.save(decoder.state_dict(), save_path)
    print(f"Saved decoder weights to {save_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
