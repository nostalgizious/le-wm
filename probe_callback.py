"""ProbeValidationCallback: Lightning Callback for epoch-end probe fitting.

Fits linear and MLP probes on accumulated validation-epoch embeddings and
logs MSE, Pearson r, and degenerate flags to W&B.

Optionally loads a pre-trained VisualDecoder for qualitative image
reconstruction during validation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lightning import pytorch as pl

from probes import (
    LinearProbe,
    MLPProbe,
    fit_linear_lstsq,
    fit_mlp_sgd,
    pearson_r,
)


class ProbeValidationCallback(pl.Callback):
    """Accumulates embeddings/targets during validation, fits probes at
    epoch end, and logs metrics.

    Probes are re-fit from scratch every validation epoch.  Probe state is
    NOT persisted or checkpointed (old weights are stale after every world-
    model update).

    **Single-device only.**  Multi-GPU requires ``all_gather`` of
    accumulated embeddings before fitting.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of ``output["emb"]`` from the forward pass.
    target_specs : dict[str, int]
        Mapping from batch key to target dimensionality.
    mlp_fit_epochs : int
        Number of SGD epochs for MLP fitting (default 20).
    mlp_fit_lr : float
        Learning rate for MLP fitting (default 1e-3).
    mlp_fit_batch_size : int
        Batch size for MLP fitting (default 256).
    h5_attrs : dict, optional
        HDF5 root attributes for on-the-fly derived target computation.
    max_probe_samples : int, optional
        If set, subsample validation frames before probe fitting
        (default None = use all).  Keep constant across dataset sizes
        for comparable probe runtime.
    decoder_ckpt_path : str, optional
        Path to a pre-trained VisualDecoder checkpoint (state_dict or
        full module).  If provided, the decoder reconstructs images
        during validation.
    n_decode_images : int
        Number of images to reconstruct per validation epoch (default 4).
    """

    def __init__(
        self,
        embed_dim: int,
        target_specs: dict[str, int],
        mlp_fit_epochs: int = 20,
        mlp_fit_lr: float = 1e-3,
        mlp_fit_batch_size: int = 256,
        h5_attrs: dict | None = None,
        decoder_ckpt_path: str | None = None,
        n_decode_images: int = 4,
        max_probe_samples: int | None = None,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim
        self._target_specs = dict(target_specs)
        self._mlp_fit_cfg = {
            "epochs": mlp_fit_epochs,
            "lr": mlp_fit_lr,
            "batch_size": mlp_fit_batch_size,
        }
        self._h5_attrs = h5_attrs
        self._max_probe_samples = max_probe_samples

        # Probe modules — owned by this callback, not the LightningModule
        self.linear_probes = nn.ModuleDict({
            name: LinearProbe(embed_dim, dim)
            for name, dim in target_specs.items()
        })
        self.mlp_probes = nn.ModuleDict({
            name: MLPProbe(embed_dim, dim)
            for name, dim in target_specs.items()
        })

        # Accumulation buffers (CPU)
        self._accum_emb: list[torch.Tensor] = []
        self._accum_targets: dict[str, list[torch.Tensor]] = {
            name: [] for name in target_specs
        }

        # ── Decoder (optional) ────────────────────────────────────────
        self._decoder: torch.nn.Module | None = None
        self._n_decode = n_decode_images
        self._accum_decode_emb: list[torch.Tensor] = []
        self._accum_decode_pixels: list[torch.Tensor] = []

        if decoder_ckpt_path is not None:
            self._load_decoder(decoder_ckpt_path, embed_dim)

    def _load_decoder(self, ckpt_path: str, embed_dim: int) -> None:
        """Load a pre-trained VisualDecoder from checkpoint."""
        from pathlib import Path
        from probes import VisualDecoder

        ckpt_path_p = Path(ckpt_path)
        if not ckpt_path_p.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path_p), map_location="cpu", weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            self._decoder = ckpt
            return
        if isinstance(ckpt, dict):
            # Infer architecture from checkpoint state dict shapes
            query_shape = next((v.shape for k, v in ckpt.items() if "query_tokens" in k), None)
            if query_shape is not None:
                num_patches = query_shape[1]
                hidden_dim = query_shape[2]
            else:
                num_patches, hidden_dim = 64, 512  # WAAM defaults

            grid = int(num_patches ** 0.5)
            img_size = grid * 16  # default patch_size=16
            # Infer heads: dim_head defaults to 64, so heads = hidden / 64
            # But if dim_head differs, try to detect from attn weight shape
            heads = max(1, hidden_dim // 64)
            decoder = VisualDecoder(
                embed_dim=embed_dim, img_size=img_size, patch_size=16,
                depth=2, heads=heads, dim_head=64,
            )
            decoder.load_state_dict(ckpt, strict=False)
            self._decoder = decoder
            return
        raise RuntimeError(f"Unrecognized decoder checkpoint format: {type(ckpt)}")

    # ── Lightning hooks ────────────────────────────────────────────────

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate detached embeddings, targets, and (optionally) decode images."""
        if "emb" not in outputs:
            return

        emb = outputs["emb"].detach()  # [B, T, D]
        emb_flat = emb.reshape(-1, emb.size(-1)).cpu()  # [B*T, D]
        self._accum_emb.append(emb_flat)

        # Compute derived targets on-the-fly if raw slabs present
        batch = self._maybe_compute_derived_targets(batch)

        for name in self._target_specs:
            if name in batch:
                target = batch[name]
                if target.ndim == 2:
                    target = target.unsqueeze(-1)
                target_flat = target.reshape(-1, target.size(-1)).cpu()
                self._accum_targets[name].append(target_flat)

        # ── Decoder accumulation ──────────────────────────────────────
        if self._decoder is not None and "pixels" in batch:
            # Accumulate first-timestep CLS tokens + pixel images for decoding
            B, T = emb.shape[:2]
            cls_t0 = emb[:, 0, :].cpu()  # [B, D] — first timestep only
            pixels = batch["pixels"]
            if isinstance(pixels, torch.Tensor):
                px_t0 = pixels[:, 0, ...].cpu()  # [B, 3, H, W] — first timestep
                self._accum_decode_emb.append(cls_t0)
                self._accum_decode_pixels.append(px_t0)

    def _maybe_compute_derived_targets(self, batch: dict) -> dict:
        """Compute peak_temp_k and pct_geometry from raw slabs if needed."""
        if self._h5_attrs is None:
            return batch
        needs_peak = "peak_temp_k" in self._target_specs and "peak_temp_k" not in batch
        needs_pct = "pct_geometry" in self._target_specs and "pct_geometry" not in batch
        if not (needs_peak or needs_pct):
            return batch
        if "material" not in batch or "temperature" not in batch:
            return batch

        from src.dataloader.derived_targets import WaamDerivedTargets

        material = batch["material"].detach().cpu().numpy()
        temperature = batch["temperature"].detach().cpu().numpy()
        goal_geom = batch.get("goal_geometry")
        if goal_geom is not None:
            goal_geom = goal_geom.detach().cpu().numpy()
        else:
            import logging
            logging.getLogger("ProbeCallback").warning(
                "goal_geometry missing from batch — pct_geometry will be degenerate. "
                f"batch keys: {list(batch.keys())}"
            )
            goal_geom = np.zeros((material.shape[0], 1, 1), dtype=np.float32)

        t = WaamDerivedTargets(h5_attrs=self._h5_attrs)
        sub_batch = {"material": material, "temperature": temperature, "goal_geometry": goal_geom}
        result = t(sub_batch)

        batch = dict(batch)
        if needs_peak and "peak_temp_k" in result:
            batch["peak_temp_k"] = torch.from_numpy(result["peak_temp_k"])
        if needs_pct and "pct_geometry" in result:
            batch["pct_geometry"] = torch.from_numpy(result["pct_geometry"])
        return batch

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Concatenate accumulated data, fit probes, log metrics."""
        if not self._accum_emb:
            return

        device = pl_module.device
        all_emb = torch.cat(self._accum_emb, dim=0).to(device=device, dtype=torch.float32)
        all_targets: dict[str, torch.Tensor] = {}
        for name, lst in self._accum_targets.items():
            if lst:
                all_targets[name] = torch.cat(lst, dim=0).to(device=device, dtype=torch.float32)

        # ── Subsample if capped (constant across dataset sizes) ────────
        n_samples = all_emb.shape[0]
        if self._max_probe_samples is not None and n_samples > self._max_probe_samples:
            idx = torch.randperm(n_samples)[:self._max_probe_samples]
            all_emb = all_emb[idx]
            all_targets = {k: v[idx] for k, v in all_targets.items()}

        # ── Move probes to device and fit ──────────────────────────────
        self.linear_probes.to(device)
        self.mlp_probes.to(device)

        for name, target_dim in self._target_specs.items():
            if name not in all_targets:
                continue
            y = all_targets[name]
            fit_linear_lstsq(self.linear_probes[name], all_emb, y)
            self.mlp_probes[name].train()
            fit_mlp_sgd(self.mlp_probes[name], all_emb, y, **self._mlp_fit_cfg)

        # ── Compute and log probe metrics ──────────────────────────────
        with torch.inference_mode():
            for name in self._target_specs:
                if name not in all_targets:
                    continue
                y = all_targets[name]
                for probe_type, probes in [("linear", self.linear_probes), ("mlp", self.mlp_probes)]:
                    probe = probes[name]
                    probe.eval()
                    preds = probe(all_emb)
                    mse = F.mse_loss(preds, y)
                    r_vals, deg_vals = [], []
                    for d in range(y.size(1)):
                        r_val, deg = pearson_r(preds[:, d], y[:, d])
                        r_vals.append(r_val)
                        deg_vals.append(deg)
                    r_mean = torch.stack(r_vals).mean()
                    deg_mean = torch.stack(deg_vals).mean()
                    pred_var = preds.var(dim=0).mean()
                    targ_var = y.var(dim=0).mean()

                    for logger in trainer.loggers:
                        logger.log_metrics({
                            f"val/{probe_type}_{name}_mse": float(mse.detach().cpu()),
                            f"val/{probe_type}_{name}_r": float(r_mean.detach().cpu()),
                            f"val/{probe_type}_{name}_r_degenerate": float(deg_mean.detach().cpu()),
                            f"val/{probe_type}_{name}_pred_var": float(pred_var.detach().cpu()),
                            f"val/{probe_type}_{name}_targ_var": float(targ_var.detach().cpu()),
                        }, step=trainer.global_step)

        # ── Decoder: reconstruct first N images ─────────────────────────
        if self._decoder is not None and self._accum_decode_emb:
            self._decoder.to(device)
            self._decoder.eval()
            emb_cat = torch.cat(self._accum_decode_emb, dim=0).to(device)
            pix_cat = torch.cat(self._accum_decode_pixels, dim=0)
            n = min(self._n_decode, emb_cat.size(0))

            with torch.inference_mode():
                recons = self._decoder(emb_cat[:n].float()).cpu()

            for logger in trainer.loggers:
                import wandb
                if hasattr(logger, "experiment") and isinstance(logger.experiment, type(wandb)):
                    images = []
                    for i in range(n):
                        orig = pix_cat[i]
                        recon = recons[i]
                        images.append(wandb.Image(orig, caption=f"original_{i}"))
                        images.append(wandb.Image(recon, caption=f"reconstructed_{i}"))
                    logger.experiment.log({"val/decoded_images": images}, step=trainer.global_step)

        # Clear accumulators
        self._accum_emb.clear()
        for lst in self._accum_targets.values():
            lst.clear()
        self._accum_decode_emb.clear()
        self._accum_decode_pixels.clear()

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """No-op — fitting moved to on_validation_epoch_end for correct step alignment."""
        pass

    # ── State management ──────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "linear_probes": self.linear_probes.state_dict(),
            "mlp_probes": self.mlp_probes.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.linear_probes.load_state_dict(state_dict["linear_probes"])
        self.mlp_probes.load_state_dict(state_dict["mlp_probes"])
