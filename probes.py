"""Probe modules and fitting utilities for LeWM latent-space validation.

Provides:
- LinearProbe: single linear layer with bias for linear decoding
- MLPProbe: 2-layer MLP with LayerNorm for non-linear decoding
- VisualDecoder: transformer decoder (paper App. D) — architecture scaffold
- fit_linear_lstsq: closed-form least-squares fitting
- fit_mlp_sgd: mini-batch SGD fitting
- pearson_r: Pearson correlation with degenerate-flag output
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Probe modules
# ═══════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Single linear layer probe with bias.

    Args:
        embed_dim: Dimensionality of the input embedding.
        target_dim: Dimensionality of the target quantity.
    """

    def __init__(self, embed_dim: int, target_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, target_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """*x*: ``[B, embed_dim]`` → returns ``[B, target_dim]``."""
        return self.linear(x)


class MLPProbe(nn.Module):
    """2-layer MLP probe with LayerNorm and GELU activation.

    Uses LayerNorm (not BatchNorm1d) because probe fitting happens on
    accumulated data with variable batch sizes.

    Args:
        embed_dim: Input embedding dimensionality.
        target_dim: Target quantity dimensionality.
        hidden_dim: Hidden layer width (default 512).
    """

    def __init__(
        self, embed_dim: int, target_dim: int, hidden_dim: int = 512
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """*x*: ``[B, embed_dim]`` → returns ``[B, target_dim]``."""
        return self.net(x)


class VisualDecoder(nn.Module):
    """Transformer decoder that reconstructs images from CLS embeddings.

    Architecture from LeWorldModel App. D: project CLS token → use as K/V
    in cross-attention → learnable query tokens (one per output patch) →
    N cross-attention + MLP blocks → linear project to pixels → rearrange.

    **Not integrated into the training loop.**  Must be trained separately
    with MSE reconstruction loss on extracted CLS tokens.

    Args:
        embed_dim: CLS embedding dimensionality.
        img_size: Output image size (square).
        patch_size: Patch size for output image.
        depth: Number of cross-attention blocks.
        heads: Attention heads.
        dim_head: Dimension per attention head.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        img_size: int = 224,
        patch_size: int = 16,
        depth: int = 2,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        num_patches = (img_size // patch_size) ** 2
        hidden_dim = heads * dim_head
        patch_pixels = patch_size * patch_size * 3

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        self.cls_to_hidden = nn.Linear(embed_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)

        self.blocks = nn.ModuleList([
            _CrossAttentionBlock(hidden_dim, heads, dim_head) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, patch_pixels)
        self.output_activation = nn.Sigmoid()

    def forward(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        """*cls_tokens*: ``[B, embed_dim]`` → returns ``[B, 3, H, W]``."""
        B = cls_tokens.size(0)

        # Project CLS to hidden and prepare as K/V
        kv = self.cls_to_hidden(cls_tokens).unsqueeze(1)  # [B, 1, hidden_dim]

        # Expand query tokens for batch
        q = self.query_tokens.expand(B, -1, -1)  # [B, num_patches, hidden_dim]

        # Cross-attention blocks
        x = q
        for block in self.blocks:
            x = block(x, kv)

        x = self.norm(x)
        x = self.output_proj(x)  # [B, num_patches, patch_pixels]
        x = self.output_activation(x)

        # Unpatchify: [B, num_patches, patch_pixels] → [B, 3, H, W]
        grid = int(self.num_patches ** 0.5)
        x = x.view(B, grid, grid, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 3, self.img_size, self.img_size)
        return x


class _CrossAttentionBlock(nn.Module):
    """Cross-attention block with residual MLP."""

    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)
        x = q + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Fitting utilities
# ═══════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def fit_linear_lstsq(
    probe: LinearProbe, X: torch.Tensor, y: torch.Tensor
) -> None:
    """Fit a LinearProbe via closed-form least squares.

    Sets ``probe.linear.weight`` and ``probe.linear.bias`` in-place.
    Falls back to pseudo-inverse for rank-deficient *X*.

    Args:
        probe: LinearProbe instance to fit.
        X: Input embeddings ``[N, embed_dim]``.
        y: Targets ``[N, target_dim]``.
    """
    device = X.device
    X = X.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

    # Add bias column
    X_aug = torch.cat([X, torch.ones(X.size(0), 1, device=device)], dim=1)

    try:
        solution = torch.linalg.lstsq(X_aug, y).solution  # [embed_dim+1, target_dim]
    except RuntimeError:
        # Rank-deficient fallback
        solution = torch.linalg.pinv(X_aug) @ y

    if not torch.isfinite(solution).all():
        solution = torch.linalg.pinv(X_aug) @ y

    weight = solution[:-1, :].T  # [target_dim, embed_dim]
    bias = solution[-1, :]       # [target_dim]

    probe.linear.weight.copy_(weight.to(probe.linear.weight.dtype))
    probe.linear.bias.copy_(bias.to(probe.linear.bias.dtype))


def fit_mlp_sgd(
    probe: MLPProbe,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> None:
    """Fit an MLPProbe via mini-batch SGD.

    Trains the probe in-place on the full dataset for *epochs* passes.

    Args:
        probe: MLPProbe instance to fit.
        X: Input embeddings ``[N, embed_dim]``.
        y: Targets ``[N, target_dim]``.
        epochs: Number of passes over the data.
        lr: Learning rate.
        batch_size: Mini-batch size.
    """
    probe.train()
    # Under torch.inference_mode(), parameter requires_grad flags can be
    # suppressed.  Explicitly re-enable them so the optimizer can compute
    # gradients.
    for p in probe.parameters():
        p.requires_grad_(True)
    X = X.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    N = X.size(0)

    for _ in range(epochs):
        perm = torch.randperm(N, device=X.device)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            with torch.enable_grad():
                preds = probe(X[idx])
                loss = F.mse_loss(preds, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


# ═══════════════════════════════════════════════════════════════════════════
# Metric utilities
# ═══════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def pearson_r(
    preds: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Pearson correlation coefficient.

    Returns ``(r_value, degenerate_flag)`` where *degenerate_flag* is 1.0
    when either side has zero variance (correlation is undefined), and
    *r_value* is set to 0.0 in that case.

    Args:
        preds: Predicted values ``[N]`` 1-D tensor.
        targets: Ground-truth values ``[N]`` 1-D tensor.

    Returns:
        Tuple of ``(r_value, degenerate_flag)``, both scalar tensors.
    """
    preds = preds.to(dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)

    p_std = preds.std()
    t_std = targets.std()
    degenerate = (p_std == 0.0) | (t_std == 0.0)

    if degenerate:
        return torch.tensor(0.0, device=preds.device), torch.tensor(1.0, device=preds.device)

    p_centered = preds - preds.mean()
    t_centered = targets - targets.mean()
    r = (p_centered * t_centered).sum() / (
        (p_centered * p_centered).sum().sqrt() * (t_centered * t_centered).sum().sqrt()
    )
    return r, torch.tensor(0.0, device=preds.device)


__all__ = [
    "LinearProbe",
    "MLPProbe",
    "VisualDecoder",
    "fit_linear_lstsq",
    "fit_mlp_sgd",
    "pearson_r",
]
