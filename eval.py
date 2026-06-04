import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())

    # If a direct .h5 path is given, use the WAAM dataset loader which
    # handles the datagen HDF5 format (episodes/<id>/<column>).
    name_str = str(dataset_name)
    if name_str.endswith(".h5"):
        h5_path = Path(name_str)
        if not h5_path.is_absolute():
            h5_path = dataset_path / h5_path
        from src.dataloader.waam_dataset import WaamFlatDataset
        dataset = WaamFlatDataset(
            name=h5_path.stem,
            cache_dir=str(h5_path.parent),
            keys_to_cache=cfg.dataset.keys_to_cache,
        )
    else:
        dataset = swm.data.HDF5Dataset(
            dataset_name,
            keys_to_cache=cfg.dataset.keys_to_cache,
            cache_dir=dataset_path,
        )
    return dataset


@torch.inference_mode()
def run_multistep_prediction_eval(
    model,      # JEPA / LeWM model (loaded via AutoCostModel)
    dataset,    # WaamFlatDataset
    cfg,        # Hydra DictConfig
    device: str = "cuda",
) -> dict:
    """Offline autoregressive latent prediction eval.

    Measures whether autoregressive latent prediction degrades over the
    same frame-skipped horizon used by CEM planning.

    Returns dict of metrics suitable for W&B logging.  When no valid
    chunks exist, returns only metadata and accounting keys.
    """
    frameskip = getattr(dataset, 'frameskip', 1)
    ctx_frames = cfg.wm.history_size
    pred_frames = cfg.eval.multistep_pred_frames
    total_loaded_frames = ctx_frames + pred_frames
    raw_horizon_steps = pred_frames * frameskip

    num_valid = 0
    num_skipped = 0

    # ── Collect valid episode starting points ──────────────────────────
    ep_lengths = getattr(dataset, 'lengths', None)
    if ep_lengths is None:
        ep_lengths = getattr(dataset, 'ep_len', None)
    ep_offsets = getattr(dataset, 'offsets', None)
    if ep_offsets is None:
        ep_offsets = getattr(dataset, 'ep_offset', None)

    n_episodes = len(ep_lengths)
    valid_starts: list[tuple[int, int, int]] = []  # (ep, start, length)

    for ep in range(n_episodes):
        ep_len = int(ep_lengths[ep])
        if ep_len >= total_loaded_frames:
            valid_starts.append((ep, 0, ep_len))

    if not valid_starts:
        return {
            "eval/multistep_frameskip": frameskip,
            "eval/multistep_pred_frames": pred_frames,
            "eval/multistep_raw_steps": raw_horizon_steps,
            "eval/multistep_num_valid_chunks": 0,
            "eval/multistep_num_skipped_chunks": n_episodes,
        }

    # ── Sample chunks ──────────────────────────────────────────────────
    num_chunks = min(cfg.eval.multistep_num_chunks, len(valid_starts))
    rng = np.random.default_rng(cfg.seed + 1)  # independent seed
    chosen = rng.choice(len(valid_starts), size=num_chunks, replace=True)

    all_per_step_mse: list[torch.Tensor] = []

    batch_size = cfg.eval.multistep_batch_size
    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        batch_indices = chosen[batch_start:batch_end]

        pixels_batch = []
        actions_batch = []

        for idx in batch_indices:
            ep, _, ep_len = valid_starts[idx]
            max_start = ep_len - total_loaded_frames
            start = rng.integers(0, max_start + 1) if max_start > 0 else 0

            chunk = dataset.load_chunk(
                [ep], [start], [start + total_loaded_frames * frameskip]
            )[0]

            px = chunk["pixels"]  # [T, 3, H, W]
            act = chunk.get("action")

            if px.shape[0] >= total_loaded_frames:
                px = px[:total_loaded_frames]
                pixels_batch.append(px)
                if act is not None:
                    act = act[:total_loaded_frames]
                    actions_batch.append(act)
                num_valid += 1
            else:
                num_skipped += 1

        if not pixels_batch:
            continue

        # Stack batch
        pixels = torch.stack([torch.as_tensor(p, device=device) for p in pixels_batch])
        # pixels: [B, total_loaded_frames, 3, H, W]

        # ── Encode frames ──────────────────────────────────────────
        encode_input = {"pixels": pixels}
        if actions_batch:
            raw_actions = np.stack(actions_batch, axis=0)
            raw_actions = torch.nan_to_num(
                torch.as_tensor(raw_actions, device=device, dtype=torch.float32),
                nan=0.0,
            )
            encode_input["action"] = raw_actions

        info = model.encode(encode_input)
        emb_gt = info["emb"]  # [B, total_loaded_frames, D]

        # ── Encode actions like training ───────────────────────────
        if "act_emb" in info:
            act_emb = info["act_emb"]  # [B, total_loaded_frames, D_a]
        elif actions_batch and hasattr(model, 'action_encoder'):
            act_emb = model.action_encoder(raw_actions)
        else:
            # Fallback: zero action embeddings (shouldn't happen with
            # a properly loaded model, but keeps eval robust).
            D = emb_gt.shape[-1]
            act_emb = torch.zeros(
                len(pixels_batch), total_loaded_frames, D,
                device=device, dtype=emb_gt.dtype,
            )

        # ── Autoregressive rollout ─────────────────────────────────
        pred = emb_gt[:, :ctx_frames].clone()

        for k in range(pred_frames):
            emb_window = pred[:, -ctx_frames:]
            act_window = act_emb[:, k : k + ctx_frames]

            out = model.predict(emb_window, act_window)
            next_emb = out[:, -1:]
            pred = torch.cat([pred, next_emb], dim=1)

        # ── Per-step MSE ───────────────────────────────────────────
        per_step_mse = (
            (pred[:, ctx_frames:] - emb_gt[:, ctx_frames:]) ** 2
        ).mean(dim=-1)  # [B, pred_frames]
        all_per_step_mse.append(per_step_mse.cpu())

    # ── Aggregate ──────────────────────────────────────────────────────
    if not all_per_step_mse:
        return {
            "eval/multistep_frameskip": frameskip,
            "eval/multistep_pred_frames": pred_frames,
            "eval/multistep_raw_steps": raw_horizon_steps,
            "eval/multistep_num_valid_chunks": 0,
            "eval/multistep_num_skipped_chunks": num_skipped,
        }

    all_mse = torch.cat(all_per_step_mse, dim=0)  # [total_chunks, pred_frames]
    mean_per_step = all_mse.mean(dim=0)  # [pred_frames]

    metrics = {
        "eval/multistep_num_valid_chunks": num_valid,
        "eval/multistep_num_skipped_chunks": num_skipped,
    }

    for h in cfg.eval.multistep_horizons:
        if h <= pred_frames:
            metrics[f"eval/latent_mse_predframe@{h}"] = float(mean_per_step[h - 1])
        else:
            print(f"  Warning: requested horizon {h} exceeds pred_frames={pred_frames}, skipping.")

    metrics["eval/latent_mse_mean"] = float(all_mse.mean())

    if pred_frames > 1:
        metrics["eval/latent_mse_auc"] = float(
            torch.trapz(mean_per_step, dx=1.0) / (pred_frames - 1)
        )
    else:
        metrics["eval/latent_mse_auc"] = float(mean_per_step[0])

    metrics["eval/multistep_frameskip"] = int(frameskip)
    metrics["eval/multistep_pred_frames"] = int(pred_frames)
    metrics["eval/multistep_raw_steps"] = int(raw_horizon_steps)

    return metrics


@hydra.main(version_base=None, config_path="./config/eval", config_name="waam")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # ── WAAM integration: resolve env config + render overrides ──
    import src.environment.register  # noqa: F401  — triggers gym registration

    waam_cfg_dict = OmegaConf.to_container(cfg.waam, resolve=True)
    overrides_dict = OmegaConf.to_container(cfg.planning_render_overrides, resolve=True)

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world,
        image_shape=(cfg.eval.img_size, cfg.eval.img_size),
        waam_cfg=waam_cfg_dict,
        render_overrides=overrides_dict,
    )

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        col_data = stats_dataset.get_col_data(col)
        # Skip multi-dimensional columns (geom_map, goal_geometry, etc.)
        if col_data.ndim > 2:
            continue
        processor = preprocessing.StandardScaler()
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    # ── Multi-step latent prediction eval (offline, no simulator) ──
    metrics: dict = {}
    if cfg.eval.get("run_multistep_eval", False) and policy != "random":
        print("\n── Multi-step latent prediction eval ──")
        multistep_metrics = run_multistep_prediction_eval(
            model, dataset, cfg, device="cuda"
        )
        metrics.update(multistep_metrics)
        print(f"  MSE@1={multistep_metrics.get('eval/latent_mse_predframe@1', 'N/A')}  "
              f"MSE@25={multistep_metrics.get('eval/latent_mse_predframe@25', 'N/A')}  "
              f"AUC={multistep_metrics.get('eval/latent_mse_auc', 'N/A')}")

    start_time = time.time()
    mpc_metrics = world._evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=None,
        mode="auto",
    )
    end_time = time.time()
    
    metrics.update(mpc_metrics)
    print(metrics)

    # ── Log CEM metrics to W&B if enabled ──
    if OmegaConf.select(cfg, "wandb.enabled"):
        try:
            import wandb
            wandb.init(
                project=OmegaConf.select(cfg, "wandb.config.project") or "lwm-waam",
                entity=OmegaConf.select(cfg, "wandb.config.entity") or "fsandco",
                name=(OmegaConf.select(cfg, "wandb.config.name") or "waam_eval"),
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
            )
            wandb.log({
                "eval/success_rate": float(mpc_metrics.get("success_rate", 0)),
                "eval/evaluation_time_s": end_time - start_time,
                "eval/num_eval": int(cfg.eval.num_eval),
                "eval/goal_offset": int(cfg.eval.goal_offset_steps),
            })
            # Also log multistep eval metrics if available
            if cfg.eval.get("run_multistep_eval", False):
                wandb.log({k: v for k, v in multistep_metrics.items()
                          if isinstance(v, (int, float))})
            wandb.finish()
        except Exception:
            pass  # W&B logging is best-effort; don't fail eval if it errors

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"mpc_metrics: {mpc_metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
