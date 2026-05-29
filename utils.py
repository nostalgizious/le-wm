import numpy as np
import torch
import os
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """Image preprocessor that handles numpy (T, C, H, W) correctly.

    ``ToImage`` assumes numpy arrays are in (…, H, W, C) layout and does
    ``transpose(-3, -1)``.  WAAM datasets return (T, C, H, W) numpy
    (channels-first) so the transpose corrupts the channel dimension.
    We convert numpy → torch first so the tensor fast-path is used.

    No ImageNet normalization is applied — the ViT is randomly initialised
    and pixels stay in [0, 1].
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


def get_column_normalizer(dataset, source: str, target: str):
    """Compute per-channel mean/std in a single streaming pass.

    Uses the DataLoader to read batches (O(1) memory, no
    ``get_col_data`` materialization).  Accumulates sum and sum-of-squares
    in ``float64`` for numerical stability, then derives mean/std.
    """
    if len(dataset.lengths) == 0:
        def _id(x):
            if not torch.is_tensor(x):
                x = torch.from_numpy(np.asarray(x))
            return x.float()
        return dt.transforms.WrapTorchTransform(_id, source=source, target=target)

    n_workers = min(8, max(1, os.cpu_count() or 1))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=n_workers,
        # Avoid re-importing heavy modules in worker processes
        persistent_workers=False,
    )

    count = 0
    total = None
    total_sq = None
    total_batches = len(loader)
    n_zeros = 0

    for i, batch in enumerate(loader):
        if i == 0 or i % max(1, total_batches // 10) == 0 or i == total_batches - 1:
            print(f"  Normalizing {source:20s}  batch {i+1:4d}/{total_batches}  "
                  f"({100*(i+1)//total_batches:3d}%)  "
                  f"samples={count:8d}  workers={n_workers}")
        data = batch.get(source)
        if data is None:
            n_zeros += 1
            continue
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Flatten spatial dims, keep channel dim → [N_flat, D]
        D = data.shape[-1] if data.dim() > 0 else 1
        data = data.reshape(-1, D).to(dtype=torch.float64)

        valid = ~torch.isnan(data).any(dim=1)
        data = data[valid]
        n = data.shape[0]
        if n == 0:
            continue

        if total is None:
            total = data.sum(dim=0)
            total_sq = (data * data).sum(dim=0)
        else:
            total += data.sum(dim=0)
            total_sq += (data * data).sum(dim=0)
        count += n

    if total is None or count < 2:
        mean = torch.zeros(1)
        std = torch.ones(1)
    else:
        mean = (total / count).to(dtype=torch.float32).reshape(1, -1)
        # E[X^2] - E[X]^2  → population variance, Bessel-corrected to sample variance
        pop_var = (total_sq / count) - (mean.to(dtype=torch.float64).reshape(-1)) ** 2
        pop_var = pop_var.clamp(min=0.0)
        sample_var = pop_var * count / (count - 1)
        std = sample_var.sqrt().to(dtype=torch.float32).reshape(1, -1)

    def norm_fn(x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        m = mean.to(device=x.device, dtype=x.dtype)
        s = std.to(device=x.device, dtype=x.dtype)
        return ((x - m) / s).float()

    return dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)

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