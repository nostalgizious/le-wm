"""Physical prior for CEM planning — irreversible deposition penalties.

This module provides geometric welding priors that penalise material
deposited outside the goal region.  It operates in **physical** action
space (vx/vy in mm/s, wfs in m/min) and is independent of any learned
world model.  Use ``PhysicalCEMCost`` to wrap a latent cost model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Distance helpers
# ═══════════════════════════════════════════════════════════════════════════


def precompute_goal_distance_map(
    goal_geometry_2d: torch.Tensor,  # [H, W] float
    dx_mm: float,
) -> torch.Tensor:
    """Compute a 2D distance map (mm) from a binary goal mask.

    Args:
        goal_geometry_2d: Normalised goal height/occupancy field ``[H, W]``.
            Values > 1e-6 are considered goal pixels.
        dx_mm: Physical size of one pixel in mm.

    Returns:
        ``[H, W]`` float32 tensor of per-pixel distances to the nearest
        goal pixel, in millimetres.

    Note:
        Prefer ``scipy.ndimage.distance_transform_edt`` if available;
        otherwise fall back to a vectorised torch implementation.
        Do **not** use nested Python loops over pixels.
    """
    try:
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        mask = (goal_geometry_2d > 1e-6).cpu().numpy().astype(np.uint8)
        dist_px = distance_transform_edt(1 - mask)  # distance in pixels
        return torch.from_numpy(dist_px.astype(np.float32) * dx_mm)
    except ImportError:
        # Vectorised torch fallback (brute-force min over goal pixels).
        H, W = goal_geometry_2d.shape
        mask = goal_geometry_2d > 1e-6
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        goal_px = torch.stack(torch.where(mask), dim=-1).float()  # [K, 2]
        if goal_px.numel() == 0:
            return torch.full((H, W), float("inf"), dtype=torch.float32)

        # Chunked min-distance to keep memory bounded for large grids.
        chunk = 256
        dist_map = torch.full((H, W), float("inf"))
        for r0 in range(0, H, chunk):
            rows = yy[r0 : r0 + chunk].flatten()
            cols = xx[r0 : r0 + chunk].flatten()
            pts = torch.stack([rows, cols], dim=-1)  # [N, 2]
            d2 = ((pts[:, None, :] - goal_px[None, :, :]) ** 2).sum(-1)
            d_min = d2.min(dim=-1).values.sqrt()
            dist_map[r0 : r0 + chunk] = d_min.reshape(-1, W)
        return dist_map * dx_mm


def dist_to_segment(
    p: torch.Tensor,  # [..., 2]
    a: torch.Tensor,  # [2]
    b: torch.Tensor,  # [2]
) -> torch.Tensor:
    """Euclidean distance from points to line segment a→b in mm.

    Returns:
        ``[...]`` scalar distances.
    """
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum().clamp_min(1e-8)
    t = t.clamp(0.0, 1.0)
    proj = a + t.unsqueeze(-1) * ab
    return ((p - proj) ** 2).sum(-1).sqrt()


# ═══════════════════════════════════════════════════════════════════════════
# Position / distance-map sampling
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_position_xy(
    pos: torch.Tensor,
    num_candidates: int,
) -> torch.Tensor:
    """Normalise position tensor to ``[B, N, 2]``.

    Supported input shapes:
    - ``[B, 2]``        → expand to ``[B, N, 2]``
    - ``[B, 1, 2]``     → expand to ``[B, N, 2]``
    - ``[B, N, 2]``     → return as-is
    - ``[B, N, 1, 2]``  → squeeze to ``[B, N, 2]``

    Raises:
        ValueError: Unsupported shape.
    """
    ndim = pos.ndim
    shape = pos.shape

    if ndim == 2 and shape[-1] == 2:  # [B, 2]
        return pos.unsqueeze(1).expand(-1, num_candidates, -1)
    if ndim == 3 and shape[-1] == 2 and shape[-2] == 1:  # [B, 1, 2]
        return pos.expand(-1, num_candidates, -1)
    if ndim == 3 and shape[-1] == 2:  # [B, N, 2] — could be [B, 1, 2] or [B, N, 2]
        if shape[-2] == num_candidates:
            return pos
        # [B, 1, 2] with N != 1
        if shape[-2] == 1:
            return pos.expand(-1, num_candidates, -1)
        raise ValueError(
            f"Expected position shape [B, 2], [B, 1, 2], [B, N, 2], or [B, N, 1, 2], "
            f"got {tuple(pos.shape)} with num_candidates={num_candidates}"
        )
    if ndim == 4 and shape[-1] == 2 and shape[-2] == 1:  # [B, N, 1, 2]
        return pos.squeeze(-2)

    raise ValueError(
        f"Unsupported position_xy_mm shape: {tuple(pos.shape)}. "
        f"Expected [B, 2], [B, 1, 2], [B, N, 2], or [B, N, 1, 2]."
    )


def sample_distance_map(
    distance_map: torch.Tensor,  # [H, W]
    xy_mm: torch.Tensor,          # [..., 2]
    workspace_bounds: dict[str, float],
) -> torch.Tensor:
    """Sample the distance map at XY positions in mm.

    Args:
        distance_map: Precomputed goal distance map ``[H, W]`` in mm.
        xy_mm: Positions ``[..., 2]`` where ``xy_mm[..., 0] = x_mm``
            and ``xy_mm[..., 1] = y_mm``.
        workspace_bounds: Dict with keys ``x_min_mm``, ``x_max_mm``,
            ``y_min_mm``, ``y_max_mm``.

    Convention:
        - ``x_mm`` → image column / W dimension
        - ``y_mm`` → image row / H dimension
        - ``grid_sample`` grid: ``grid[..., 0] = x_norm``,
          ``grid[..., 1] = y_norm``.

    Returns:
        Sampled distances with same leading shape as ``xy_mm`` (minus the
        final ``2`` dimension).
    """
    x_min = workspace_bounds["x_min_mm"]
    x_max = workspace_bounds["x_max_mm"]
    y_min = workspace_bounds["y_min_mm"]
    y_max = workspace_bounds["y_max_mm"]

    x_norm = 2.0 * (xy_mm[..., 0] - x_min) / max(x_max - x_min, 1e-6) - 1.0
    y_norm = 2.0 * (xy_mm[..., 1] - y_min) / max(y_max - y_min, 1e-6) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)  # [..., 2]

    # grid_sample expects [N, H_out, W_out, 2]
    orig_shape = grid.shape
    flat = grid.reshape(-1, 1, 1, 2)
    dm = distance_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    sampled = F.grid_sample(dm, flat, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.reshape(orig_shape[:-1])  # [...] — drop the final dims


# ═══════════════════════════════════════════════════════════════════════════
# Geometric penalty functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_off_goal_penalty(
    positions: torch.Tensor,       # [B, N, T, 2]
    mass: torch.Tensor,            # [B, N, T]
    distance_map: torch.Tensor,    # [H, W]
    workspace_bounds: dict[str, float],
    allowed_radius_mm: float,
) -> torch.Tensor:
    """Penalise mass deposited outside the allowed radius of the goal.

    Returns:
        ``[B, N]`` per-candidate penalty.
    """
    D_goal = sample_distance_map(distance_map, positions, workspace_bounds)  # [B, N, T]
    excess = F.relu(D_goal - allowed_radius_mm)
    return (mass * excess.pow(2)).sum(dim=-1)  # [B, N]


def compute_wrong_direction_penalty(
    positions: torch.Tensor,       # [B, N, T, 2]
    vxy: torch.Tensor,             # [B, N, H, action_block, 2]
    mass: torch.Tensor,            # [B, N, T]  (T = H * action_block)
    distance_map: torch.Tensor,    # [H, W]
    workspace_bounds: dict[str, float],
    current_xy: torch.Tensor,      # [B, N, 2]
) -> torch.Tensor:
    """Penalise depositing while moving away from the nearest goal region.

    Direction to goal is approximated as the negative gradient of the
    distance map: ``d_goal = -∇D_goal`` (gradient points *away* from goal,
    so we negate it).

    This is goal-region aware, NOT missing-goal aware — it does not
    know the current deposited geometry.

    Returns:
        ``[B, N]`` per-candidate penalty.
    """
    B, N, H, AB = vxy.shape
    T = positions.shape[2]

    # Flatten vxy to match positions: [B, N, T, 2]
    vxy_flat = vxy.reshape(B, N, T, 2)

    # Approximate d_goal from distance-map gradient.
    # Use finite differences around each position.
    eps_mm = 0.5  # small offset for gradient estimation
    pos_dx = positions.clone()
    pos_dx[..., 0] += eps_mm
    pos_dy = positions.clone()
    pos_dy[..., 1] += eps_mm

    d0 = sample_distance_map(distance_map, positions, workspace_bounds)
    dx = sample_distance_map(distance_map, pos_dx, workspace_bounds)
    dy = sample_distance_map(distance_map, pos_dy, workspace_bounds)

    grad = torch.stack([dx - d0, dy - d0], dim=-1) / eps_mm  # [B, N, T, 2]
    # -grad points toward goal (distance decreases)
    d_goal = -F.normalize(grad, dim=-1, eps=1e-8)

    d_action = F.normalize(vxy_flat, dim=-1, eps=1e-8)
    dot = (d_action * d_goal).sum(dim=-1)  # [B, N, T]
    # Penalise when dot < 0 (moving away from goal)
    wrong = F.relu(-dot)

    return (mass * wrong.pow(2)).sum(dim=-1)  # [B, N]


def compute_overdeposit_penalty(
    speed: torch.Tensor,          # [B, N, H, action_block]
    wfs: torch.Tensor,            # [B, N, H, action_block]
    mass: torch.Tensor,           # [B, N, H * action_block]
    nominal_ratio: float,
    ratio_max: float = 2.0,
) -> torch.Tensor:
    """Penalise excessive deposition rate (WFS / speed).

    ``wfs/speed`` is an empirical proxy with mixed units (m/min ÷ mm/s).
    Ratios are normalised by the dataset-median ``nominal_ratio``.

    Idle WFS (zero deposition) is gated to zero.

    Returns:
        ``[B, N]`` per-candidate penalty.
    """
    B, N, H, AB = speed.shape
    eps = 1e-6
    ratio = wfs / (speed + eps)
    norm_ratio = ratio / max(nominal_ratio, eps)

    # Gate by deposition: no mass → no overdeposit penalty.
    mass_aligned = mass.reshape(B, N, H, AB)
    overdep_per_step = F.relu(norm_ratio - ratio_max).pow(2)
    overdep_per_step = overdep_per_step * (mass_aligned > 0).float()

    return overdep_per_step.sum(dim=(-2, -1))  # [B, N]


def compute_action_smoothness(
    action_blocks: torch.Tensor,        # [B, N, H, action_block * 3]
    action_std: torch.Tensor | None = None,  # [3] or [action_block*3] or None
) -> torch.Tensor:
    """Smoothness penalty across horizon steps.

    If ``action_std`` is provided, action differences are divided by the
    physical action standard deviation so that vx/vy (mm/s) and wfs
    (m/min) are comparable.  ``action_std`` of shape ``[3]`` is repeated
    by ``action_block``.

    Returns:
        ``[B, N]`` per-candidate penalty.
    """
    diffs = action_blocks[:, :, 1:, :] - action_blocks[:, :, :-1, :]  # [B, N, H-1, AD]
    if action_std is not None:
        if action_std.ndim == 1 and action_std.shape[0] == 3:
            action_block_dim = action_blocks.shape[-1] // 3
            action_std = action_std.repeat(action_block_dim)
        diffs = diffs / action_std.clamp_min(1e-6)
    return (diffs**2).sum(dim=(-2, -1))  # [B, N]


def compute_on_goal_forward_progress(
    positions: torch.Tensor,       # [B, N, T, 2]
    mass: torch.Tensor,            # [B, N, T]
    start_xy: torch.Tensor,        # [B, N, 2]
    distance_map: torch.Tensor,    # [H, W]
    workspace_bounds: dict[str, float],
    on_goal_radius_mm: float,
) -> torch.Tensor:
    """Reward depositing near the goal while making forward progress.

    Uses **incremental** progress (delta between consecutive positions)
    to avoid over-rewarding later points.

    Returns:
        ``[B, N]`` positive reward (should be **subtracted** from cost).
    """
    B, N, T = positions.shape[:3]

    D = sample_distance_map(distance_map, positions, workspace_bounds)  # [B, N, T]
    # Pad with start_xy distance for delta computation.
    D_start = sample_distance_map(distance_map, start_xy, workspace_bounds)  # [B, N]
    D_prev = torch.cat([D_start.unsqueeze(-1), D[..., :-1]], dim=-1)
    delta_D = F.relu(D_prev - D)  # reduction in distance = progress [B, N, T]

    near_goal = (D < on_goal_radius_mm).float()
    reward = (mass * delta_D * near_goal).sum(dim=-1)  # [B, N]
    return reward


# ═══════════════════════════════════════════════════════════════════════════
# Cost wrapper
# ═══════════════════════════════════════════════════════════════════════════


class PhysicalCEMCost:
    """Wraps a NormalizedCostModel and adds irreversible-deposition priors.

    Reads current torch position from ``info_dict["position_xy_mm"]`` on
    every ``get_cost()`` call — position changes across MPC cycles.
    """

    def __init__(
        self,
        model: Any,                    # NormalizedCostModel (physical→normalized→latent)
        distance_map: torch.Tensor,    # [H, W] goal distance in mm
        workspace_bounds: dict[str, float],
        *,
        action_block: int = 4,
        phys_dt: float = 0.25,
        min_wfs_m_min: float = 1.5,
        allowed_radius_mm: float = 6.0,
        on_goal_radius_mm: float = 8.0,
        nominal_ratio: float = 0.5,
        ratio_max: float = 2.0,
        action_std: torch.Tensor | None = None,
        lambda_latent: float = 1.0,
        lambda_off: float = 100.0,
        lambda_dir: float = 20.0,
        lambda_over: float = 10.0,
        lambda_smooth: float = 0.05,
        lambda_prog: float = 5.0,
    ):
        self.model = model
        self.distance_map = distance_map
        self.workspace_bounds = workspace_bounds
        self.action_block = action_block
        self.phys_dt = phys_dt
        self.min_wfs_m_min = min_wfs_m_min
        self.allowed_radius_mm = allowed_radius_mm
        self.on_goal_radius_mm = on_goal_radius_mm
        self.nominal_ratio = nominal_ratio
        self.ratio_max = ratio_max
        self.action_std = action_std
        self.lambda_latent = lambda_latent
        self.lambda_off = lambda_off
        self.lambda_dir = lambda_dir
        self.lambda_over = lambda_over
        self.lambda_smooth = lambda_smooth
        self.lambda_prog = lambda_prog

    def parameters(self):
        """Forward to wrapped model — required by ICEMSolver for dtype inference."""
        return self.model.parameters()

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Compute total cost [B, N] for action candidates."""
        comps = self.compute_components(info_dict, action_candidates)
        return comps["total"]

    def compute_components(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return per-term cost breakdown.

        Returns dict with keys: ``latent``, ``off_goal``, ``direction``,
        ``overdep``, ``smooth``, ``progress_reward``, ``progress_cost``,
        ``total``.  Each value is ``[B, N]``.
        """
        # action_candidates: [B, N, H, action_block * 3] — physical units
        device = action_candidates.device
        dtype = action_candidates.dtype

        # Cast stored tensors to candidate device/dtype.
        distance_map = self.distance_map.to(device=device, dtype=dtype)
        action_std = (
            self.action_std.to(device=device, dtype=dtype)
            if self.action_std is not None
            else None
        )

        # Latent cost via normalised model.
        latent_cost = self.model.get_cost(info_dict, action_candidates)

        # Read current XY — handles solver-expanded shapes.
        current_xy = _normalize_position_xy(
            info_dict["position_xy_mm"],
            num_candidates=action_candidates.shape[1],
        )  # [B, N, 2]

        # Reshape action blocks to raw sub-actions [B, N, H, 4, 3].
        B, N, H, AD = action_candidates.shape
        raw = action_candidates.reshape(B, N, H, self.action_block, 3)
        vxy = raw[..., :2]  # [B, N, H, action_block, 2]
        wfs = raw[..., 2]   # [B, N, H, action_block]

        # Integrate XY path and mass.
        T = H * self.action_block
        positions = torch.empty(B, N, T, 2, device=device, dtype=dtype)
        mass_all = torch.empty(B, N, T, device=device, dtype=dtype)
        curr = current_xy  # [B, N, 2]
        idx = 0
        for h in range(H):
            for s in range(self.action_block):
                curr = curr + vxy[:, :, h, s, :] * self.phys_dt
                positions[:, :, idx, :] = curr
                m = F.relu(wfs[:, :, h, s] - self.min_wfs_m_min) * self.phys_dt
                mass_all[:, :, idx] = m
                idx += 1

        speed = torch.linalg.vector_norm(vxy, dim=-1)  # [B, N, H, action_block]

        # Geometric penalties.
        off_goal = compute_off_goal_penalty(
            positions, mass_all, distance_map,
            self.workspace_bounds, self.allowed_radius_mm,
        )
        wrong_dir = compute_wrong_direction_penalty(
            positions, vxy, mass_all, distance_map,
            self.workspace_bounds, current_xy,
        )
        overdep = compute_overdeposit_penalty(
            speed, wfs, mass_all, self.nominal_ratio, self.ratio_max,
        )
        smooth = compute_action_smoothness(action_candidates, action_std)
        progress_reward = compute_on_goal_forward_progress(
            positions, mass_all, current_xy, distance_map,
            self.workspace_bounds, self.on_goal_radius_mm,
        )

        total = (
            self.lambda_latent * latent_cost
            + self.lambda_off * off_goal
            + self.lambda_dir * wrong_dir
            + self.lambda_over * overdep
            + self.lambda_smooth * smooth
            - self.lambda_prog * progress_reward
        )
        return {
            "latent": latent_cost,
            "off_goal": off_goal,
            "direction": wrong_dir,
            "overdep": overdep,
            "smooth": smooth,
            "progress_reward": progress_reward,
            "progress_cost": -self.lambda_prog * progress_reward,
            "total": total,
        }
