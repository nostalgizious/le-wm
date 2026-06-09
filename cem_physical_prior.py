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

    # grid_sample expects [N, H_out, W_out, 2] with matching batch size.
    orig_shape = grid.shape
    flat = grid.reshape(-1, 1, 1, 2)  # [N, 1, 1, 2]
    N = flat.shape[0]
    dm = distance_map.unsqueeze(0).unsqueeze(0).expand(N, -1, -1, -1)  # [N, 1, H, W]
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
    B, N, H, AB, _ = vxy.shape
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


# ═══════════════════════════════════════════════════════════════════════════
# Raw-step dynamics penalties (WAAM-specific, 0.25 s resolution)
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_prev_action_raw(
    prev_action: torch.Tensor,
    num_candidates: int,
) -> torch.Tensor:
    """Normalise a previous raw action tensor to ``[B, N, 3]``.

    Supported input shapes:
    - ``[B, 3]``        → expand to ``[B, N, 3]``
    - ``[B, 1, 3]``     → expand to ``[B, N, 3]``
    - ``[B, N, 3]``     → return as-is
    - ``[B, N, 1, 3]``  → squeeze to ``[B, N, 3]``

    Raises:
        ValueError: Unsupported shape.
    """
    ndim = prev_action.ndim
    shape = prev_action.shape

    if ndim == 2 and shape[-1] == 3:  # [B, 3]
        return prev_action.unsqueeze(1).expand(-1, num_candidates, -1)
    if ndim == 3 and shape[-1] == 3 and shape[-2] == 1:  # [B, 1, 3]
        return prev_action.expand(-1, num_candidates, -1)
    if ndim == 3 and shape[-1] == 3:  # [B, N, 3] or [B, 1, 3]
        if shape[-2] == num_candidates:
            return prev_action
        if shape[-2] == 1:
            return prev_action.expand(-1, num_candidates, -1)
        raise ValueError(
            f"Expected prev_action shape [B, 3], [B, 1, 3], [B, N, 3], or [B, N, 1, 3], "
            f"got {tuple(prev_action.shape)} with num_candidates={num_candidates}"
        )
    if ndim == 4 and shape[-1] == 3 and shape[-2] == 1:  # [B, N, 1, 3]
        return prev_action.squeeze(-2)

    raise ValueError(
        f"Unsupported prev_action shape: {tuple(prev_action.shape)}. "
        f"Expected [B, 3], [B, 1, 3], [B, N, 3], or [B, N, 1, 3]."
    )


def compute_action_rate_penalty(
    action_candidates: torch.Tensor,          # [B, N, H, action_block * 3]
    *,
    action_block: int,
    prev_action_raw: torch.Tensor | None = None,  # [B, 3], [B, 1, 3], or [B, N, 3]
    delta_scale: torch.Tensor | None = None,       # [3] — allowed per-step change
    accel_scale: torch.Tensor | None = None,       # [3] — allowed per-step accel
    delta_limit: torch.Tensor | None = None,       # [3] — hard slew limit
    include_accel: bool = False,
) -> dict[str, torch.Tensor]:
    """Raw-substep WAAM dynamics penalties at 0.25 s resolution.

    Smoothness is enforced at the raw control timestep, not only at
    horizon-step / block level.  The first planned raw action is anchored
    to ``prev_action_raw`` (the last executed real action) to penalise
    discontinuities at MPC cycle boundaries.

    Args:
        action_candidates: Planned action blocks ``[B, N, H, AD]`` where
            ``AD = action_block * 3``.  Layout: substep0(vx,vy,wfs),
            substep1(vx,vy,wfs), ...
        action_block: Number of raw substeps per horizon step (typ. 4).
        prev_action_raw: Previous executed raw action ``[vx, vy, wfs]``.
            Used to anchor the first planned substep.  If ``None``, no
            anchoring — the first planned step is unconstrained.
        delta_scale: Per-channel allowed change per raw step ``[3]``.
            Used to normalise first-differences so vx/vy (mm/s) and wfs
            (m/min) are comparable.  Initial heuristic defaults:
            ``[2.0, 2.0, 0.5]`` (mm/s, mm/s, m/min per 0.25 s).
        accel_scale: Per-channel allowed acceleration ``[3]`` for optional
            second-difference (jerk) penalty.  Only used if
            ``include_accel=True``.
        delta_limit: Per-channel hard slew-rate limit ``[3]``.  Excess
            beyond ``|Δ| > limit`` is penalised separately.
        include_accel: If ``True``, also compute second-difference penalty.

    Returns:
        Dict with keys ``rate``, ``accel`` (zeros if disabled), ``slew``
        (zeros if no limit), ``total``.  Each ``[B, N]``.
    """
    B, N, H, AD = action_candidates.shape
    # Reshape to raw sub-actions: [B, N, H, action_block, 3] → [B, N, T, 3]
    raw = action_candidates.reshape(B, N, H, action_block, 3)
    T = H * action_block

    # Build raw sequence, optionally anchoring to prev_action.
    if prev_action_raw is not None:
        # Normalise to [B, N, 3].
        pa = _normalize_prev_action_raw(prev_action_raw, N)
        T_extra = 1
        raw_flat = raw.reshape(B, N, T, 3)
        # Prepend prev_action: [B, N, T_extra + T, 3]
        seq = torch.cat([pa.unsqueeze(2), raw_flat], dim=2)
    else:
        T_extra = 0
        seq = raw.reshape(B, N, T, 3)

    T_all = T + T_extra

    # ── First-difference (rate) penalty ──
    # Δ_t = seq[..., t, :] - seq[..., t-1, :]  for t in [1, T_all)
    diffs = seq[:, :, 1:, :] - seq[:, :, :-1, :]  # [B, N, T_all-1, 3]

    if delta_scale is not None:
        ds = delta_scale.to(device=diffs.device, dtype=diffs.dtype).clamp_min(1e-6)
        diffs_norm = diffs / ds  # [B, N, T_all-1, 3]
    else:
        diffs_norm = diffs

    # Separate velocity (channels 0-1) and WFS (channel 2) for independent weighting.
    rate_v = (diffs_norm[:, :, :, :2] ** 2).sum(dim=-1).sum(dim=-1)  # [B, N]
    rate_wfs = (diffs_norm[:, :, :, 2] ** 2).sum(dim=-1)  # [B, N]
    rate_total = rate_v + rate_wfs

    # ── Optional second-difference (accel / jerk) penalty ──
    accel = torch.zeros(B, N, device=diffs.device, dtype=diffs.dtype)
    if include_accel and T_all >= 3:
        # Δ²_t = seq[..., t, :] - 2·seq[..., t-1, :] + seq[..., t-2, :]
        accel_diffs = (
            seq[:, :, 2:, :] - 2.0 * seq[:, :, 1:-1, :] + seq[:, :, :-2, :]
        )  # [B, N, T_all-2, 3]
        if accel_scale is not None:
            a_s = accel_scale.to(device=diffs.device, dtype=diffs.dtype).clamp_min(1e-6)
            accel_diffs = accel_diffs / a_s
        accel = (accel_diffs ** 2).sum(dim=(-2, -1))  # [B, N]

    # ── Optional hard slew-rate penalty ──
    slew = torch.zeros(B, N, device=diffs.device, dtype=diffs.dtype)
    if delta_limit is not None:
        dl = delta_limit.to(device=diffs.device, dtype=diffs.dtype)
        excess = F.relu(diffs.abs() - dl)  # [B, N, T_all-1, 3]
        slew = (excess ** 2).sum(dim=(-2, -1))  # [B, N]

    total = rate_total + accel + slew
    return {"rate": rate_total, "rate_v": rate_v, "rate_wfs": rate_wfs,
            "accel": accel, "slew": slew, "total": total}


def compute_on_goal_forward_progress(
    positions: torch.Tensor,       # [B, N, T, 2]
    mass: torch.Tensor,            # [B, N, T]
    start_xy: torch.Tensor,        # [B, N, 2]
    distance_map: torch.Tensor,    # [H, W]
    workspace_bounds: dict[str, float],
) -> torch.Tensor:
    """Reward depositing while making forward progress toward the goal.

    Uses **incremental** progress (delta between consecutive positions)
    to avoid over-rewarding later points.  Rewards ANY reduction in
    distance-to-goal so CEM can navigate from far-away starting positions.

    The ``mass`` gate ensures no reward for idle / non-depositing steps.

    Returns:
        ``[B, N]`` positive reward (should be **subtracted** from cost).
    """
    B, N, T = positions.shape[:3]

    D = sample_distance_map(distance_map, positions, workspace_bounds)  # [B, N, T]
    # Pad with start_xy distance for delta computation.
    D_start = sample_distance_map(distance_map, start_xy, workspace_bounds)  # [B, N]
    D_prev = torch.cat([D_start.unsqueeze(-1), D[..., :-1]], dim=-1)
    delta_D = F.relu(D_prev - D)  # reduction in distance = progress [B, N, T]

    # Reward ANY reduction in distance-to-goal, weighted by deposition mass.
    # The mass gate already ensures no reward for idle steps.
    # No near_goal gate — we need a gradient even when far from the goal,
    # otherwise CEM cannot navigate toward it.
    reward = (mass * delta_D).sum(dim=-1)  # [B, N]
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
        allowed_radius_mm: float = 3.0,
        nominal_ratio: float = 0.5,
        ratio_max: float = 2.0,
        action_std: torch.Tensor | None = None,
        lambda_latent: float = 200.0,
        lambda_off: float = 10.0,
        lambda_dir: float = 20.0,
        lambda_over: float = 10.0,
        lambda_smooth: float = 0.5,     # weak block-level regularizer only
        lambda_prog: float = 100.0,
        # ── Raw-step dynamics (WAAM-specific, 0.25 s resolution) ──
        lambda_rate_v: float = 2.0,
        lambda_rate_wfs: float = 0.5,
        lambda_accel: float = 0.0,
        lambda_slew: float = 10.0,
        delta_scale: torch.Tensor | None = None,   # [3] per-step allowed change
        accel_scale: torch.Tensor | None = None,   # [3] per-step allowed accel
        delta_limit: torch.Tensor | None = None,   # [3] hard slew limit
        include_accel: bool = False,
    ):
        self.model = model
        self.distance_map = distance_map
        self.workspace_bounds = workspace_bounds
        self.action_block = action_block
        self.phys_dt = phys_dt
        self.min_wfs_m_min = min_wfs_m_min
        self.allowed_radius_mm = allowed_radius_mm
        self.nominal_ratio = nominal_ratio
        self.ratio_max = ratio_max
        self.action_std = action_std
        self.lambda_latent = lambda_latent
        self.lambda_off = lambda_off
        self.lambda_dir = lambda_dir
        self.lambda_over = lambda_over
        self.lambda_smooth = lambda_smooth
        self.lambda_prog = lambda_prog
        self.lambda_rate_v = lambda_rate_v
        self.lambda_rate_wfs = lambda_rate_wfs
        self.lambda_accel = lambda_accel
        self.lambda_slew = lambda_slew
        self.delta_scale = delta_scale
        self.accel_scale = accel_scale
        self.delta_limit = delta_limit
        self.include_accel = include_accel

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

        # ── Raw-step dynamics penalties (WAAM-specific, 0.25 s resolution) ──
        # Cast stored tensors for rate penalty.
        delta_scale = (
            self.delta_scale.to(device=device, dtype=dtype)
            if self.delta_scale is not None else None
        )
        accel_scale = (
            self.accel_scale.to(device=device, dtype=dtype)
            if self.accel_scale is not None else None
        )
        delta_limit = (
            self.delta_limit.to(device=device, dtype=dtype)
            if self.delta_limit is not None else None
        )
        # Read previous raw action for MPC cycle-boundary anchoring.
        prev_action_raw = info_dict.get("prev_action_raw", None)

        rate_components = compute_action_rate_penalty(
            action_candidates,
            action_block=self.action_block,
            prev_action_raw=prev_action_raw,
            delta_scale=delta_scale,
            accel_scale=accel_scale,
            delta_limit=delta_limit,
            include_accel=self.include_accel,
        )

        progress_reward = compute_on_goal_forward_progress(
            positions, mass_all, current_xy, distance_map,
            self.workspace_bounds,
        )

        total = (
            self.lambda_latent * latent_cost
            + self.lambda_off * off_goal
            + self.lambda_dir * wrong_dir
            + self.lambda_over * overdep
            + self.lambda_smooth * smooth
            + self.lambda_rate_v * rate_components["rate_v"]
            + self.lambda_rate_wfs * rate_components["rate_wfs"]
            + self.lambda_accel * rate_components["accel"]
            + self.lambda_slew * rate_components["slew"]
            - self.lambda_prog * progress_reward
        )
        return {
            "latent": latent_cost,
            "off_goal": off_goal,
            "direction": wrong_dir,
            "overdep": overdep,
            "smooth": smooth,
            "rate_v": rate_components["rate_v"],
            "rate_wfs": rate_components["rate_wfs"],
            "accel": rate_components["accel"],
            "slew": rate_components["slew"],
            "progress_reward": progress_reward,
            "progress_cost": -self.lambda_prog * progress_reward,
            "total": total,
        }


class PolarActionWrapper:
    """Wraps a cost model so iCEM samples in (speed, dx, dy, wfs) space.

    The ``(dx, dy)`` pair represents a direction vector — it gets normalised
    to unit length, then multiplied by speed to get ``(vx, vy)``.  This
    replaces ``(speed, angle)`` polar sampling and **avoids angle wrap-around**
    at ±π, which destroys iCEM's Gaussian-distribution assumptions.

    Noise on ``(dx, dy)`` with ``noise_beta=2`` produces smooth direction
    changes because the raw (dx,dy) components drift continuously — no
    circular-topology discontinuity.

    Usage::

        cartesian_model = PhysicalCEMCost(...)
        polarmodel = PolarActionWrapper(cartesian_model, action_block=4)
        solver = ICEMSolver(model=polarmodel, ...)

        # Convert Cartesian warm-start to (speed, dx, dy, wfs) before solving:
        init = PolarActionWrapper.to_polar(warm_start_cartesian, 4)
        outputs = solver.solve(info, init_action=init)

        # Convert best from (speed, dx, dy, wfs) back to Cartesian:
        best_cartesian = PolarActionWrapper.to_cartesian(outputs["actions"], 4)
    """

    def __init__(self, model, action_block: int = 4):
        self._model = model
        self.action_block = action_block

    def parameters(self):
        return self._model.parameters()

    # ── Core conversion: (speed, dx, dy, wfs) → (vx, vy, wfs) ──────────

    def _to_cartesian(self, dd: "torch.Tensor") -> "torch.Tensor":
        """dd in (speed, dx, dy, wfs) per substep → (vx, vy, wfs)."""
        import torch as _torch
        if dd.shape[-1] != self.action_block * 4:
            raise RuntimeError(
                f"PolarActionWrapper: expected last dim {self.action_block * 4} "
                f"(action_block={self.action_block} × 4), got shape {list(dd.shape)}"
            )
        B, N, H, AD = dd.shape
        block = self.action_block
        raw = dd.reshape(B, N, H, block, 4)  # (speed, dx, dy, wfs)
        speed = raw[..., 0].clamp(min=0.0)
        dx = raw[..., 1]
        dy = raw[..., 2]
        wfs = raw[..., 3]
        norm = _torch.sqrt(dx**2 + dy**2).clamp(min=1e-8)
        vx = speed * dx / norm
        vy = speed * dy / norm
        out_dim = block * 3  # (vx, vy, wfs) per substep
        return _torch.stack([vx, vy, wfs], dim=-1).reshape(B, N, H, out_dim)

    # ── CostModel protocol ─────────────────────────────────────────────

    def get_cost(self, info_dict, dd):
        return self._model.get_cost(info_dict, self._to_cartesian(dd))

    def compute_components(self, info_dict, dd):
        return self._model.compute_components(info_dict, self._to_cartesian(dd))

    # ── Static helpers ─────────────────────────────────────────────────

    @staticmethod
    def to_polar(
        cartesian: "torch.Tensor | None", action_block: int = 4,
    ) -> "torch.Tensor | None":
        """Cartesian [..., H, AD] → (speed, dx, dy, wfs) [... H, AD+?]."""
        import torch as _torch
        if cartesian is None:
            return None
        *prefix, H, AD = cartesian.shape
        raw = cartesian.reshape(*prefix, H, action_block, 3)
        vx, vy, wfs = raw[..., 0], raw[..., 1], raw[..., 2]
        speed = _torch.sqrt(vx**2 + vy**2).clamp(min=1e-8)
        dx = vx / speed
        dy = vy / speed
        return _torch.stack([speed, dx, dy, wfs], dim=-1).reshape(*prefix, H, action_block * 4)

    @staticmethod
    def to_cartesian(
        dd: "torch.Tensor | None", action_block: int = 4,
    ) -> "torch.Tensor | None":
        """(speed, dx, dy, wfs) [..., H, AD'] → Cartesian [..., H, AD]."""
        import torch as _torch
        if dd is None:
            return None
        *prefix, H, AD4 = dd.shape
        raw = dd.reshape(*prefix, H, action_block, 4)
        speed, dx, dy, wfs = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
        norm = _torch.sqrt(dx**2 + dy**2).clamp(min=1e-8)
        vx = speed * dx / norm
        vy = speed * dy / norm
        out_dim = action_block * 3
        return _torch.stack([vx, vy, wfs], dim=-1).reshape(*prefix, H, out_dim)
