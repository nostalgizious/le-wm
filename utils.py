import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """Image preprocessor that handles numpy (T, C, H, W) correctly.

    ``ToImage`` assumes numpy arrays are in (…, H, W, C) layout and does
    ``transpose(-3, -1)``.  WAAM datasets return (T, C, H, W) numpy
    (channels-first) so the transpose corrupts the channel dimension.
    We convert numpy → torch first so the tensor fast-path is used.
    """
    def _ensure_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    to_tensor = dt.transforms.WrapTorchTransform(
        _ensure_tensor, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_tensor, resize)


def _compute_norm_code_hash() -> str:
    """SHA256 of this file — invalidates cached stats when normalization code changes."""
    h = hashlib.sha256()
    with open(__file__, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:16]


def _read_cached_norm_stats(h5_path: str) -> dict | None:
    """Return cached per-column normalization stats, or None if invalid/missing."""
    code_hash = _compute_norm_code_hash()
    try:
        with h5py.File(h5_path, "r", libver="latest") as f:
            if f.attrs.get("norm_stats_code_hash") != code_hash:
                return None
            raw = f.attrs.get("norm_stats_json")
            if raw is None:
                return None
        return json.loads(raw)
    except Exception:
        return None


def _write_cached_norm_stats(h5_path: str, stats: dict) -> None:
    """Write per-column normalization stats as HDF5 root attributes."""
    code_hash = _compute_norm_code_hash()
    with h5py.File(h5_path, "r+", libver="latest") as f:
        f.attrs["norm_stats_code_hash"] = code_hash
        f.attrs["norm_stats_json"] = json.dumps(stats)


def _compute_all_norm_stats(dataset) -> dict[str, dict[str, list[float]]]:
    """Compute mean/std for all normalizable columns in one streaming pass.

    Returns ``{col: {"mean": [D], "std": [D]}}``.
    """
    if len(dataset.lengths) == 0:
        return {}

    # Collect all normalizable columns (skip pixels — handled by img preprocessor).
    columns = []
    for ep_len in range(len(dataset.lengths)):
        if ep_len == 0:
            g = dataset._episode_group(0)
            for col in dataset.column_names:
                if col in ("pixels", "material", "temperature", "goal", "ir", "depth",
                           "episode_idx", "step_idx", "geom_map", "goal_geometry",
                           "goal_temperature"):
                    continue
                if col in g:
                    columns.append(col)
            break

    if not columns:
        return {}

    print(f"  Computing normalization stats for: {columns}", flush=True)

    n_workers = 0  # single-threaded: h5py handles + fork are unsafe
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=n_workers,
    )

    accum: dict[str, dict] = {col: {"total": None, "total_sq": None, "count": 0}
                              for col in columns}
    total_batches = len(loader)

    for i, batch in enumerate(loader):
        if i % max(1, total_batches // 10) == 0:
            print(f"    batch {i+1:4d}/{total_batches}  "
                  f"({100*(i+1)//total_batches:3d}%)", flush=True)

        for col in columns:
            data = batch.get(col)
            if data is None:
                continue
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            D = data.shape[-1] if data.dim() > 0 else 1
            data = data.reshape(-1, D).to(dtype=torch.float64)
            valid = ~torch.isnan(data).any(dim=1)
            data = data[valid]
            n = data.shape[0]
            if n == 0:
                continue

            a = accum[col]
            if a["total"] is None:
                a["total"] = data.sum(dim=0)
                a["total_sq"] = (data * data).sum(dim=0)
            else:
                a["total"] += data.sum(dim=0)
                a["total_sq"] += (data * data).sum(dim=0)
            a["count"] += n

    result = {}
    for col in columns:
        a = accum[col]
        if a["total"] is None or a["count"] < 2:
            mean = np.array([0.0])
            std = np.array([1.0])
        else:
            count = a["count"]
            mean_f64 = a["total"] / count
            pop_var = (a["total_sq"] / count) - mean_f64 ** 2
            pop_var = pop_var.clamp(min=0.0)
            sample_var = pop_var * count / (count - 1)
            std_f64 = sample_var.sqrt()
            mean = mean_f64.to(dtype=torch.float32).cpu().numpy().tolist()
            std = std_f64.to(dtype=torch.float32).cpu().numpy().tolist()

        result[col] = {"mean": mean, "std": std}

    return result


def _make_normalizer(mean: list, std: list, source: str, target: str):
    """Build a WrapTorchTransform normalizer from cached mean/std lists."""
    mean_t = torch.tensor(mean, dtype=torch.float32).reshape(1, -1)
    std_t = torch.tensor(std, dtype=torch.float32).reshape(1, -1)

    def norm_fn(x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        m = mean_t.to(device=x.device, dtype=x.dtype)
        s = std_t.to(device=x.device, dtype=x.dtype)
        return ((x - m) / s).float()

    return dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)


def get_column_normalizer(dataset, source: str, target: str):
    """Get a per-channel normalizer, cached on the HDF5 file.

    On first call for a dataset, computes stats for all normalizable
    columns in one streaming pass and stores them as HDF5 root attributes
    (``norm_stats_json`` + ``norm_stats_code_hash``).  Subsequent calls
    are instant reads.

    Invalidation: stats are recomputed when ``utils.py`` changes (SHA256
    hash of this file).
    """
    if source in ("pixels", "material", "temperature"):
        def _id(x):
            if not torch.is_tensor(x):
                x = torch.from_numpy(np.asarray(x))
            return x.float()
        return dt.transforms.WrapTorchTransform(_id, source=source, target=target)

    if len(dataset.lengths) == 0:
        return _make_normalizer([0.0], [1.0], source, target)

    h5_path = getattr(dataset, "path", None)
    if h5_path is None:
        # Fallback: compute without caching
        stats = _compute_all_norm_stats(dataset)
        if source in stats:
            return _make_normalizer(stats[source]["mean"], stats[source]["std"],
                                    source, target)
        return _make_normalizer([0.0], [1.0], source, target)

    # ── Try cached path ──
    cached = _read_cached_norm_stats(h5_path)
    if cached is not None and source in cached:
        s = cached[source]
        return _make_normalizer(s["mean"], s["std"], source, target)

    # ── Compute all columns at once, cache, then return ──
    print(f"  ⚡ Computing & caching normalization stats → {h5_path}")
    stats = _compute_all_norm_stats(dataset)
    try:
        _write_cached_norm_stats(h5_path, stats)
        print(f"  ✓ Cached normalization stats to HDF5 root attrs")
    except Exception as exc:
        print(f"  ⚠  Could not write norm cache ({exc})")

    if source in stats:
        return _make_normalizer(stats[source]["mean"], stats[source]["std"],
                                source, target)
    return _make_normalizer([0.0], [1.0], source, target)

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")