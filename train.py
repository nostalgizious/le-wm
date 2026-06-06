import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

torch.set_float32_matmul_precision("high")

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from probe_callback import ProbeValidationCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    if cfg.data.dataset.get("type") == "waam":
        from src.dataloader.waam_dataset import WaamFlatDataset

        render_overrides = (
            OmegaConf.to_container(cfg.data.render, resolve=True)
            if "render" in cfg.data else None
        )

        dataset = WaamFlatDataset(
            path=cfg.data.dataset.path,
            frameskip=cfg.data.dataset.frameskip,
            num_steps=cfg.data.dataset.num_steps,
            keys_to_load=cfg.data.dataset.keys_to_load,
            render_overrides=render_overrides,
            observation_source=cfg.data.dataset.get("observation_source", "auto"),
            return_raw=OmegaConf.select(cfg, "probe.enabled", default=False),
            max_episodes=cfg.data.dataset.get("max_episodes"),
        )
    else:
        dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            # Raw slabs are consumed by WaamDerivedTargets before normalization;
            # they are 4D [N, Z, Y, X] and cannot be normalization-collapsed.
            if col in ("material", "temperature"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=True, drop_last=False)

    # ── Probe precondition check ─────────────────────────────────────
    if OmegaConf.select(cfg, "probe.enabled"):
        available = set(dataset.column_names)
        missing_raw = {"material", "temperature"} - available
        if missing_raw:
            msg = (
                f"probe.enabled=true but the HDF5 dataset does not contain raw "
                f"material/temperature slabs ({sorted(missing_raw)} not in {sorted(available)}).\n"
                f"The dataset was generated with store_raw=False.\n"
                f"Re-run datagen with store_raw=True, or set probe.enabled=false."
            )
            raise RuntimeError(msg)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = dataset.get_dim("action")

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                **OmegaConf.to_container(cfg.get("scheduler", {}), resolve=True),
            },
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    # ── Probe validation callback (opt-in) ──────────────────────────────
    extra_callbacks = [object_dump_callback]

    # ── Within-epoch checkpointing (for large datasets) ──
    ckpt_every = cfg.get("checkpoint_every_n_steps", None)
    if ckpt_every is not None and ckpt_every > 0:
        extra_callbacks.append(ModelCheckpoint(
            dirpath=str(run_dir),
            filename="lewm_{step:06d}",
            every_n_train_steps=ckpt_every,
            save_top_k=3,
            save_last=False,
        ))
    if OmegaConf.select(cfg, "probe.enabled"):
        # Build h5_attrs for on-the-fly derived target computation
        h5_attrs = getattr(dataset, "probe_h5_attrs", None)
        # Auto-discover decoder weights if decoder was trained
        decoder_ckpt = None
        decoder_dir = OmegaConf.select(cfg, "probe.decoder_ckpt_path")
        if decoder_dir is not None:
            decoder_ckpt = str(Path(decoder_dir).expanduser())
        else:
            # Check subdir (Hydra output) and current working dir
            subdir = cfg.get("subdir") or "."
            for search in [Path(subdir), Path(".")]:
                candidate = search / "decoder_weights.pt"
                if candidate.exists():
                    decoder_ckpt = str(candidate.resolve())
                    break
        probe_cb = ProbeValidationCallback(
            embed_dim=cfg.wm.embed_dim,
            target_specs={
                "position_xy_mm": 2,
                "peak_temp_k": 1,
                "pct_geometry": 1,
            },
            mlp_fit_epochs=cfg.probe.mlp_fit.epochs,
            mlp_fit_lr=cfg.probe.mlp_fit.lr,
            mlp_fit_batch_size=cfg.probe.mlp_fit.batch_size,
            h5_attrs=h5_attrs,
            decoder_ckpt_path=decoder_ckpt,
            max_probe_samples=cfg.probe.get("max_probe_samples", None),
        )
        extra_callbacks.append(probe_cb)

    trainer = pl.Trainer(
        **cfg.trainer,
        limit_val_batches=cfg.probe.get("limit_val_batches", 1.0) if cfg.probe.enabled else 1.0,
        callbacks=extra_callbacks,
        num_sanity_val_steps=0 if OmegaConf.select(cfg, "probe.enabled") else 1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
