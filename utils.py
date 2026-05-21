import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224, normalize: bool = True):
    """Image preprocessor that handles numpy (T, C, H, W) correctly.

    ``ToImage`` assumes numpy arrays are in (…, H, W, C) layout and does
    ``transpose(-3, -1)``.  WAAM datasets return (T, C, H, W) numpy
    (channels-first) so the transpose corrupts the channel dimension.
    We convert numpy → torch first so the tensor fast-path is used.
    """
    stats = dt.dataset_stats.ImageNet if normalize else {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

    def _ensure_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    to_tensor = dt.transforms.WrapTorchTransform(
        _ensure_tensor, source=source, target=target
    )
    to_image = dt.transforms.ToImage(
        **stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_tensor, to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        m = mean.to(device=x.device, dtype=x.dtype)
        s = std.to(device=x.device, dtype=x.dtype)
        return ((x - m) / s).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

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