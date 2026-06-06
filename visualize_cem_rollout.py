#!/usr/bin/env -S uv run python
"""Visualize CEM internal rollout with decoder predictions side-by-side.

Shows the model's imagined rollout (decoder reconstructions), the actual
simulator execution, and the goal — all in one MP4 video.

Usage:
    uv run python visualize_cem_rollout.py \\
        --dataset=output/datagen/foo.h5 \\
        --ckpt=output/training/.../lewm_epoch_30_object.ckpt \\
        --decoder=output/training/.../decoder_weights.pt

The CEM solver's ``callbacks`` mechanism (line 236 of cem.py) is used to
capture the best action sequence at each planning iteration without modifying
any solver or model code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))          # for "from src.dataloader..."
sys.path.insert(0, str(REPO_ROOT / "le-wm"))  # for "from probes...", "from jepa..."



# ═══════════════════════════════════════════════════════════════════════════
# Action-normalizing cost model wrapper
# ═══════════════════════════════════════════════════════════════════════════


class NormalizedCostModel:
    """Wraps a world model so it can be called with physical actions.

    The model was trained on StandardScaler-normalized actions (zero mean,
    unit variance).  This wrapper applies that normalization inside
    ``get_cost`` so that CEM can sample in **physical** action space.
    Best actions are already in physical units — no denormalization needed
    before ``env.step()``.
    """

    def __init__(self, model: torch.nn.Module, action_mean: torch.Tensor, action_std: torch.Tensor):
        self._model = model
        self.action_mean = action_mean
        self.action_std = action_std

    def parameters(self):
        return self._model.parameters()

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Normalize actions, then delegate to the real model."""
        mean = self.action_mean.to(device=action_candidates.device, dtype=action_candidates.dtype)
        std = self.action_std.to(device=action_candidates.device, dtype=action_candidates.dtype)
        normalized = (action_candidates - mean) / std
        return self._model.get_cost(info_dict, normalized)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions back to physical units."""
        mean = self.action_mean.to(device=actions.device, dtype=actions.dtype)
        std = self.action_std.to(device=actions.device, dtype=actions.dtype)
        return actions * std + mean


# ═══════════════════════════════════════════════════════════════════════════
# CEM callback
# ═══════════════════════════════════════════════════════════════════════════


class CaptureBestRollout:
    """iCEM callback that records the best action sequence per iteration."""

    def __init__(self):
        self.best_mean: torch.Tensor | None = None  # [B, horizon, action_dim]
        self.cost_history: list[float] = []
        self.mean_history: list[torch.Tensor] = []
        self.history: list = []  # required by CEM solver
        self.output_key: str = "CaptureBestRollout"  # required by CEM solver

    def reset(self):
        self.best_mean = None
        self.cost_history.clear()
        self.mean_history.clear()
        self.history.clear()

    def start_batch(self):
        pass

    def end_solve(self):
        pass

    def __call__(self, **kwargs):
        step = kwargs.get("step", 0)
        mean = kwargs["mean"]  # [B, horizon, action_dim] — current best mean
        topk_vals = kwargs["topk_vals"]  # [B, K]
        self.best_mean = mean.clone()
        best = float(topk_vals[:, 0].mean())  # mean of best elite cost across batch
        topk_avg = float(topk_vals.mean())  # mean of all top-k costs
        self.cost_history.append(best)
        self.mean_history.append(mean.clone())
        if step % 5 == 0 or step == 0:
            print(f"    iCEM iter {step:2d}: best={best:.2f}  topk_avg={topk_avg:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════════════════



def _to_uint8(t: torch.Tensor, img_size: int = 128) -> np.ndarray:
    """[3, H, W] in [0, 1] → [H, W, 3] uint8 numpy."""
    if t.shape[1] != img_size or t.shape[2] != img_size:
        t = F.interpolate(
            t.unsqueeze(0), size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return t.clamp(0.0, 1.0).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Decoder frame decoding
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Decoder frame decoding
# ═══════════════════════════════════════════════════════════════════════════


@torch.inference_mode()
def encode_decode(
    model,
    decoder: torch.nn.Module,
    pixels: torch.Tensor,  # [1, 1, 3, H, W] — raw [0,1] pixels
) -> torch.Tensor:
    """Encode raw pixels → CLS token → decode → [3, H, W] in [0, 1].

    Identical code path to ``ablation_study.run_decoder_video``.
    The predictor outputs of this model are in a collapsed subspace and
    cannot be decoded — this encoder→decoder path is the only way to
    produce readable frames for this checkpoint.

    NOTE: This produces a **reconstruction** of the actual simulator state,
    NOT a model-imagined rollout.  The video layout label reflects this.
    """
    info = model.encode({"pixels": pixels})
    cls = info["emb"][:, 0, :]  # [1, D]
    img = decoder(cls.float()).clamp(0.0, 1.0)  # [1, 3, H, W]
    return img[0].cpu()  # [3, H, W]


# ═══════════════════════════════════════════════════════════════════════════
# Main visualization
# ═══════════════════════════════════════════════════════════════════════════


def visualize_cem_rollout(
    dataset_path: Path,
    ckpt_path: Path,
    decoder_weights: Path,
    output_path: Path | None = None,
    *,
    episode_idx: int = 0,
    start_step: int = 0,
    device: str = "cuda",
    img_size: int = 128,
    fps: int = 4,
    horizon: int = 25,
    receding_horizon: int = 15,
    total_steps: int = 60,
    debug_costs: bool = False,
    lambda_latent: float | None = None,
    lambda_off: float | None = None,
    lambda_dir: float | None = None,
    lambda_over: float | None = None,
    lambda_smooth: float | None = None,
    lambda_prog: float | None = None,
) -> Path | None:
    """Generate a 3-up MP4 with closed-loop MPC.

    Each cycle: iCEM plans ``horizon`` steps, the first ``receding_horizon`` are
    executed in the simulator and decoded from the model rollout.  The env state
    is then used as the starting point for the next cycle, up to ``total_steps``
    physical environment steps.
    """
    import imageio
    import gymnasium as gym
    import stable_worldmodel as swm
    from stable_worldmodel.solver.icem import ICEMSolver

    from probes import VisualDecoder
    from cem_physical_prior import PhysicalCEMCost, precompute_goal_distance_map
    from src.dataloader.waam_dataset import WaamFlatDataset
    from src.environment.env import WaamEnv

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = loaded.model if hasattr(loaded, "model") else loaded
    model = model.to(device)
    model.eval()

    # ── Load decoder ──────────────────────────────────────────────────
    print(f"Loading decoder: {decoder_weights}")
    state_dict = torch.load(decoder_weights, map_location="cpu", weights_only=False)
    query_shape = next(
        (v.shape for k, v in state_dict.items() if "query_tokens" in k), None
    )
    if query_shape is not None:
        num_patches, hidden_dim = query_shape[1], query_shape[2]
    else:
        num_patches, hidden_dim = 64, 512
    grid = int(num_patches**0.5)
    decoder_img_size = grid * 16
    heads = max(1, hidden_dim // 64)
    decoder = VisualDecoder(
        embed_dim=192,
        img_size=decoder_img_size,
        patch_size=16,
        depth=2,
        heads=heads,
        dim_head=64,
    )
    decoder.load_state_dict(state_dict, strict=False)
    decoder = decoder.to(device)
    decoder.eval()
    img_size = decoder_img_size

    # ── Load dataset ──────────────────────────────────────────────────
    ds = WaamFlatDataset(
        path=str(dataset_path.expanduser().resolve()),
        frameskip=4,
        num_steps=1,
        keys_to_load=["pixels", "goal", "action", "position_xy_mm"],
    )
    ep_len = int(ds.lengths[episode_idx])
    if start_step >= ep_len:
        raise ValueError(f"start_step={start_step} >= episode length={ep_len}")

    # ── Compute action normalizer stats (model was trained on normalized actions) ──
    # Use cached normalization stats if available (matches training normalizer).
    import json, os

    cached = None
    if isinstance(ds.path, str):
        cache_path = ds.path + ".norm_stats.json"
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                cached = json.load(fh).get("stats", {})

    if cached and "action" in cached:
        action_mean = torch.tensor(cached["action"]["mean"], dtype=torch.float32).to(device)
        action_std = torch.tensor(cached["action"]["std"], dtype=torch.float32).to(device)
        print(f"  Loaded action normalizer from cache ({cache_path})")
    else:
        print("Computing action normalizer stats from dataset...")
        action_data = torch.from_numpy(np.asarray(ds.get_col_data("action"), dtype=np.float32))
        action_data = action_data[~torch.isnan(action_data).any(dim=1)]
        action_mean = action_data.mean(0).to(device)
        action_std = action_data.std(0).to(device)
    action_std = action_std.clamp(min=1e-6)  # avoid division by zero
    action_dim_per_step = action_mean.numel()
    print(f"  action dim: {action_dim_per_step}, mean range: [{action_mean.min():.2f}, {action_mean.max():.2f}], "
          f"std range: [{action_std.min():.4f}, {action_std.max():.4f}]")

    # Compute physical action std (3D) and nominal_ratio for geometric priors.
    # action_data is 12D (frameskip=4); reshape to raw [*, 3] sub-actions.
    raw_actions = action_data.reshape(-1, 4, 3).reshape(-1, 3)
    physical_std = raw_actions.std(0).clamp(min=1e-6).to(device)  # [3]
    wfs_col = raw_actions[:, 2]
    speed = torch.sqrt(raw_actions[:, 0]**2 + raw_actions[:, 1]**2)
    active = (wfs_col >= 1.5) & (speed > 1e-3)
    if active.sum() > 0:
        nominal_ratio = (wfs_col[active] / speed[active]).median().item()
    else:
        nominal_ratio = 0.5  # fallback
    print(f"  physical_std (vx,vy,wfs): {physical_std.tolist()}, nominal_ratio: {nominal_ratio:.4f}")

    # ── Wrap model with action normalization ────────────────────────────
    wrapped_model = NormalizedCostModel(model, action_mean, action_std)
    print(f"Model wrapped with action normalizer ({action_dim_per_step} dims).")

    # ── Read HDF5 rendering parameters (matches dataloader pipeline) ───
    import h5py as _h5py
    with _h5py.File(str(dataset_path.expanduser().resolve()), "r") as _f:
        _attrs = dict(_f.attrs)

    def _render_state_slab(state, *, xy_mm: np.ndarray | None = None) -> np.ndarray:
        """Render env state → [H, W, 3] uint8, matching dataloader pipeline.

        Channels: [depth, ir, position_gaussian] in mm workspace coords.
        """
        from src.environment.camera import render_ir_and_depth_from_state

        ir, depth = render_ir_and_depth_from_state(
            state,
            dx=float(_attrs["dx"]),
            workspace_y0=int(_attrs["workspace_y0"]),
            workspace_y1=int(_attrs["workspace_y1"]),
            workspace_x0=int(_attrs["workspace_x0"]),
            workspace_x1=int(_attrs["workspace_x1"]),
            substrate_height_vox=int(_attrs["substrate_height_vox"]),
            max_bead_height_vox=int(_attrs["max_bead_height_vox"]),
            thermal_min_k=float(_attrs["thermal_min_k"]),
            thermal_max_k=float(_attrs["thermal_max_k"]),
            projection_mode=int(_attrs["projection_mode"]),
            mask_empty_space=bool(_attrs.get("mask_empty_space", False)),
            hot_threshold_k=float(_attrs.get("hot_threshold_k", -1.0)),
            blur_kernel_size=int(_attrs.get("blur_kernel_size", 0)),
            blur_sigma_px=float(_attrs.get("blur_sigma_px", 0.0)),
            noise_std_norm=float(_attrs.get("noise_std_norm", 0.0)),
            normalize_depth=bool(_attrs.get("normalize_depth", True)),
            return_absolute_height=bool(_attrs.get("return_absolute_height", False)),
        )
        # Position channel: Gaussian blob at current tool position
        if xy_mm is not None:
            sigma_mm = max(
                float(_attrs.get("position_sigma_px", 5.0)) * float(_attrs["dx"]),
                1e-6,
            )
            _H, _W = depth.shape[1:]
            _xs = np.linspace(
                float(_attrs.get("x_min_mm", 0.0)),
                float(_attrs.get("x_max_mm", 128.0)),
                _W,
                dtype=np.float32,
            )
            _ys = np.linspace(
                float(_attrs.get("y_min_mm", 0.0)),
                float(_attrs.get("y_max_mm", 128.0)),
                _H,
                dtype=np.float32,
            )
            _X, _Y = np.meshgrid(_xs, _ys)
            _xy = np.asarray(xy_mm, dtype=np.float32).reshape(-1, 2)
            _pos = np.empty((_xy.shape[0], _H, _W), dtype=np.float32)
            for b in range(_xy.shape[0]):
                _dx2 = (_X - _xy[b, 0]) ** 2
                _dy2 = (_Y - _xy[b, 1]) ** 2
                _pos[b] = np.exp(-(_dx2 + _dy2) / (2.0 * sigma_mm * sigma_mm))
            _pos = np.clip(_pos, 0.0, 1.0)
        else:
            _pos = np.zeros_like(depth.cpu().numpy())
        # Stack as [depth, ir, position] to match dataloader channel order
        rgb = torch.stack([depth, ir, torch.as_tensor(_pos, device=depth.device)], dim=-1)
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)
        return rgb.cpu().numpy()[0]  # [H, W, 3] uint8

    # ── Get initial state + goal from dataset ─────────────────────────
    initial = ds.load_chunk([episode_idx], [start_step], [start_step + 1])
    goal_offset = min(25, ep_len - start_step - 1)
    goal_chunk = ds.load_chunk(
        [episode_idx], [start_step + goal_offset], [start_step + goal_offset + 1]
    )

    # ── Set up WaamEnv ────────────────────────────────────────────────
    import src.environment.register  # noqa: F401
    env: WaamEnv = gym.make("swm/Waam-v1", waam_cfg={
        "env": {
            "files": {"sim_toml": "sim_cfg.toml"},
            "runtime": {"device": device, "precision": "float32", "max_steps": 600,
                        "action_mode": "direct"},
            "writer": {"write": False, "write_file": None},
            "workspace": {"x_min_mm": 0.0, "x_max_mm": 128.0,
                          "y_min_mm": 0.0, "y_max_mm": 128.0},
            "actions": {"min_speed_mm_s": 3.0, "max_speed_mm_s": 30.0,
                        "min_wfs_m_min": 1.5, "max_wfs_m_min": 12.0,
                        "max_wait_time_s": 30.0},
            "observation": {"mode": "dict", "style": "real",
                            "goal_image_shape": [3, 128, 128]},
            "camera": {"thermal_min_k": 473.15, "thermal_max_k": 3000.0,
                       "thermal_projection_mode": 0, "ir_mask_empty_space": True,
                       "ir_hot_threshold_k": 400.0,
                       "ir_blur_kernel_size": 0, "ir_blur_sigma_px": 0.0,
                       "ir_noise_std_norm": 0.0},
            "goal": {"goal_temp_delta_": 300.0, "position_sigma_px": 3.0},
            "deposition": {"deposition_substeps": 5},
            "redistribution": {"enable_redistribution": False},
            "termination": {"terminate_on_nonfinite": True,
                            "terminate_on_workspace_exit": True,
                            "workspace_exit_is_truncation": True},
        },
        "sim": {
            "grid": {"dx": 1.0, "substrate_height_mm": 8.0},
            "solver": {"phys_dt": 0.25, "goldak_radius": 20, "max_bead_radius_vox": 10},
            "material": {"density_s": 7855.0, "density_l": 7360.0,
                         "specific_heat_s": 600.0, "specific_heat_l": 800.0,
                         "thermal_conductivity": 50.0, "T_solidus": 1770.0,
                         "T_liquidus": 1790.0, "L_fusion": 2.56e5,
                         "T_ambient": 293.15, "wire_superheat": 50.0,
                         "wire_radius": 0.0006},
            "bead": {"bead_shape_toml": "bead_shapes.toml"},
            "redistribution": {
                "redist_dep_excess_max_thresh": 0.05,
                "redist_dep_excess_sum_thresh": 0.01,
                "redist_dt_limit_s": 1.0, "redist_max_iters": 3,
                "redist_excess_eps": 1.0e-2,
                "redist_hot_threshold_density": 0.0,
                "redist_hot_full_density": 0.0,
                "redist_hot_min_mobility": 0.1,
                "redist_move_fraction": 0.7,
                "redist_source_hot_power": 1.5,
                "redist_surface_power": 0.5,
                "redist_upward_bias": 2.0,
                "redist_downward_bias": 1.0,
                "redist_same_z_bias": 0.1,
                "redist_lateral_decay_power": 2.0,
                "redist_max_sources_per_batch": 3000,
                "redist_fallback_enthalpy_density": 0.0,
                "redist_allow_excess_destinations": True,
                "redist_max_excess_per_voxel": 0.7,
                "redist_excess_capacity_factor": 0.5,
            },
        },
    }).unwrapped

    env.reset()
    # ── Set goal from dataset ───────────────────────────────────────────
    from src.config.config_datagen import GoalSpec
    start_xy = np.asarray(initial[0]["position_xy_mm"], dtype=np.float32)  # [2]
    goal_image = np.asarray(goal_chunk[0]["goal"][0], dtype=np.float32)  # [3, H, W]
    # Get per-episode goal_geometry from the HDF5 episode group
    g = ds._episode_group(int(episode_idx))
    goal_geom = np.asarray(g["goal_geometry"], dtype=np.float32)  # [H, W]
    goal_spec = GoalSpec(
        start_xy_mm=start_xy[np.newaxis, :],  # [1, 2]
        goal_image=goal_image[np.newaxis, :],  # [1, 3, H, W]
        goal_geometry=goal_geom[np.newaxis, :, :],  # [1, H, W]
    )
    env.reset(seed=None, options={"_goal_spec": goal_spec})

    # ── Precompute goal distance map for geometric priors ──────────────
    goal_geom_tensor = torch.from_numpy(goal_geom).float().to(device)  # [H, W]
    distance_map = precompute_goal_distance_map(goal_geom_tensor, dx_mm=float(_attrs["dx"]))
    workspace_bounds = {
        "x_min_mm": float(_attrs.get("x_min_mm", 0.0)),
        "x_max_mm": float(_attrs.get("x_max_mm", 128.0)),
        "y_min_mm": float(_attrs.get("y_min_mm", 0.0)),
        "y_max_mm": float(_attrs.get("y_max_mm", 128.0)),
    }
    print(f"  Goal distance map precomputed ({distance_map.shape[0]}×{distance_map.shape[1]}).")

    # ── Wrap model with physical CEM cost ──────────────────────────────
    lambda_kwargs = {}
    if lambda_latent is not None:
        lambda_kwargs["lambda_latent"] = lambda_latent
    if lambda_off is not None:
        lambda_kwargs["lambda_off"] = lambda_off
    if lambda_dir is not None:
        lambda_kwargs["lambda_dir"] = lambda_dir
    if lambda_over is not None:
        lambda_kwargs["lambda_over"] = lambda_over
    if lambda_smooth is not None:
        lambda_kwargs["lambda_smooth"] = lambda_smooth
    if lambda_prog is not None:
        lambda_kwargs["lambda_prog"] = lambda_prog

    physical_cost = PhysicalCEMCost(
        wrapped_model,
        distance_map=distance_map,
        workspace_bounds=workspace_bounds,
        action_block=4,
        phys_dt=float(_attrs.get("phys_dt", 0.25)),
        min_wfs_m_min=float(_attrs.get("min_wfs_m_min", 1.5)),
        nominal_ratio=nominal_ratio,
        action_std=physical_std,
        **lambda_kwargs,
    )

    # ── Build iCEM solver with callback ────────────────────────────────
    plan_cfg = swm.PlanConfig(horizon=horizon, receding_horizon=receding_horizon,
                               action_block=4, history_len=3, warm_start=True)
    callback = CaptureBestRollout()
    solver = ICEMSolver(
        model=physical_cost, callbacks=[callback], device=device,
        noise_beta=0.0,        # white noise (standard CEM) — avoids correlated "swirling"
        num_samples=500,       # more samples for better exploration coverage
        n_steps=30,            # default iterations
        topk=50,               # slightly larger elite set
        alpha=0.1,             # momentum (default)
        n_elite_keep=5,        # elite injection (default)
    )
    # iCEM clamps candidates to action bounds.  Use physical-space bounds
    # so the solver explores in the units the geometric priors expect.
    # Normalization to model space happens inside NormalizedCostModel.get_cost().
    # CRITICAL: WFS lower bound is 0.0 (not min_wfs_m_min) so CEM can choose idle.
    from gymnasium.spaces import Box
    raw_as = env.action_space  # Box(shape=(3,), dtype=float32)
    physical_low = raw_as.low.copy()
    physical_low[2] = 0.0  # allow idle / no deposition
    batched_as = Box(
        low=physical_low[np.newaxis, :],
        high=raw_as.high[np.newaxis, :],
        shape=(1,) + raw_as.shape,
        dtype=raw_as.dtype,
    )
    solver.configure(
        action_space=batched_as, n_envs=1, config=plan_cfg,
    )

    # ── Helpers ────────────────────────────────────────────────────────
    def _to_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.float().unsqueeze(0).to(device)
        return torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device)

    def _env_pixels_raw(env) -> torch.Tensor:
        """Render env state → raw [0,1] pixels [1, 1, 3, H, W].
        For encoder→decoder reconstruction (matching decoder extraction)."""
        render_out = _render_state_slab(env.state, xy_mm=env.curr_xy_mm)  # [H, W, 3] uint8
        t = torch.from_numpy(render_out).float().permute(2, 0, 1).to(device) / 255.0
        return t.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]

    # ── Goal (constant across cycles) ───────────────────────────────────
    goal_data = _to_tensor(goal_chunk[0]["pixels"])  # [1, 1, 3, H, W]
    goal_frame = _to_uint8(goal_data[0, 0].cpu().clamp(0, 1), img_size)

    # ── Debug-costs mode: sample 3 hand-coded plans and exit ──────────
    if debug_costs:
        print("\n=== DEBUG COSTS ===")
        current_xy_np = np.asarray(initial[0]["position_xy_mm"], dtype=np.float32)  # [2]
        goal_end_xy_np = np.asarray(goal_chunk[0]["position_xy_mm"], dtype=np.float32) if "position_xy_mm" in goal_chunk[0] else current_xy_np + np.array([20.0, 0.0])
        goal_dir = goal_end_xy_np - current_xy_np
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
        perp_dir = np.array([-goal_dir[1], goal_dir[0]], dtype=np.float32)

        speed_phys = 10.0  # mm/s
        wfs_phys = 5.0     # m/min

        def _make_plan(vx, vy, wfs):
            step = np.array([vx, vy, wfs], dtype=np.float32)
            block = np.tile(step, 4)  # [12]
            plan = np.tile(block[np.newaxis, :], (horizon, 1))  # [H, 12]
            return torch.from_numpy(plan).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, 12]

        plans = {
            "toward":   _make_plan(goal_dir[0] * speed_phys, goal_dir[1] * speed_phys, wfs_phys),
            "sideways": _make_plan(perp_dir[0] * speed_phys, perp_dir[1] * speed_phys, wfs_phys),
            "backward": _make_plan(-goal_dir[0] * speed_phys, -goal_dir[1] * speed_phys, wfs_phys),
        }

        info_debug = {
            "pixels": _to_tensor(initial[0]["pixels"]),
            "position_xy_mm": _to_tensor(current_xy_np),
            "action": _to_tensor(initial[0]["action"]),
            "goal": goal_data,
        }

        header = f"{'Plan':<12} {'speed':>7} {'wfs':>6} {'dist':>6} | {'latent':>8} {'off_goal':>9} {'dir':>9} {'overdep':>8} {'smooth':>8} {'prog_rwd':>9} {'prog_cost':>10} | {'total':>10}"
        print(header)
        print("-" * len(header))
        for name, plan in plans.items():
            comps = physical_cost.compute_components(info_debug, plan)
            latent = comps["latent"].item()
            off = comps["off_goal"].item()
            direction = comps["direction"].item()
            overdep = comps["overdep"].item()
            smooth = comps["smooth"].item()
            prog_rwd = comps["progress_reward"].item()
            prog_cost = comps["progress_cost"].item()
            total = comps["total"].item()

            # Approximate mean distance to goal
            B, N, H, AD = plan.shape
            raw = plan.reshape(B, N, H, 4, 3)
            vxy_dbg = raw[..., :2]
            curr = torch.from_numpy(current_xy_np).float().to(device).view(1, 1, 2)
            pts = [curr]
            for h in range(H):
                for s in range(4):
                    curr = curr + vxy_dbg[:, :, h, s, :] * float(_attrs.get("phys_dt", 0.25))
                    pts.append(curr)
            all_pos = torch.cat(pts, dim=1)  # [1, T+1, 2]
            from cem_physical_prior import sample_distance_map
            dists = sample_distance_map(distance_map, all_pos, workspace_bounds)
            avg_dist = dists.mean().item()
            print(f"{name:<12} {speed_phys:>7.1f} {wfs_phys:>6.1f} {avg_dist:>6.1f} | {latent:>8.1f} {off:>9.1f} {direction:>9.1f} {overdep:>8.1f} {smooth:>8.1f} {prog_rwd:>9.1f} {prog_cost:>10.1f} | {total:>10.1f}")

        print("\nExit (debug-costs mode).")
        env.close()
        return None

    # ── Closed-loop MPC ─────────────────────────────────────────────────
    all_pred_frames: list = []
    all_sim_frames: list = []
    total_env_steps = 0
    prev_mean: torch.Tensor | None = None  # warm-start for CEM
    cycle = 0

    while total_env_steps < total_steps:
        cycle += 1
        # ── Build info_dict for this cycle ─────────────────────────
        if total_env_steps == 0:
            # Solver expects raw [0,1] pixels — matches decoder extraction
            # and the eval pipeline's transform order.
            raw_px = _to_tensor(initial[0]["pixels"])  # [1, 1, 3, H, W] raw [0,1]
            info_dict = {
                "pixels": raw_px,
                "position_xy_mm": _to_tensor(initial[0]["position_xy_mm"]),
                "action": _to_tensor(initial[0]["action"]),
                "goal": goal_data,
            }
        else:
            px = _env_pixels_raw(env)  # [1, 1, 3, H, W] raw [0,1]
            pos = torch.from_numpy(
                np.asarray(env.curr_xy_mm, dtype=np.float32)
            ).unsqueeze(0).to(device)  # [1, 1, 2]
            act = torch.zeros(1, 1, 12, device=device, dtype=torch.float32)
            info_dict = {
                "pixels": px,
                "position_xy_mm": pos,
                "action": act,
                "goal": goal_data,
            }

        orig_info = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in info_dict.items()}

        # Warm-start CEM with shifted previous mean
        init_action = None
        if prev_mean is not None:
            # prev_mean: [1, horizon, 12]; shift by receding_horizon, pad zeros
            shifted = prev_mean[:, receding_horizon:, :]  # [1, horizon-RH, 12]
            pad = torch.zeros(1, receding_horizon, prev_mean.shape[-1],
                              device=prev_mean.device, dtype=prev_mean.dtype)
            init_action = torch.cat([shifted, pad], dim=1)  # [1, horizon, 12]

        outputs = solver.solve(info_dict, init_action=init_action)
        best_mean = outputs["mean"][0]  # [1, horizon, action_dim]
        if isinstance(best_mean, list):
            best_mean = callback.best_mean[0] if callback.best_mean is not None else outputs["mean"][-1]
        if best_mean.ndim == 3:
            best_mean = best_mean[0]  # [horizon, action_dim]
        # Cap how many steps we take this cycle
        take = min(receding_horizon, (total_steps - total_env_steps + 3) // 4)
        best_actions = best_mean[:take].to(device)

        # Best actions are already in physical units (CEM sampled in physical space).
        # No denormalization needed — NormalizedCostModel normalizes internally.
        physical_actions = best_actions

        # Save for next warm-start
        if best_mean.ndim == 2:
            prev_mean = best_mean.unsqueeze(0)  # [1, horizon, action_dim]
        else:
            prev_mean = best_mean

        cost_info = (
            f"cost trend: {callback.cost_history[0]:.1f} → {callback.cost_history[-1]:.1f}"
            if len(callback.cost_history) >= 2 else ""
        )
        print(f"  Cycle {cycle}: {best_actions.shape[0]} planning actions "
              f"({best_actions.shape[0] * 4} env steps)  {cost_info}")

        # ── Simulator execution + per-block prediction frames ──────
        for t in range(physical_actions.shape[0]):
            block = physical_actions[t].cpu().numpy().reshape(-1, 3)  # [4, 3]
            for i in range(block.shape[0]):
                obs, _, _, _, _ = env.step(
                    np.array([block[i]], dtype=np.float32)
                )
                total_env_steps += 1
            # Render simulator frame after this CEM action block
            sim_img = _render_state_slab(env.state, xy_mm=env.curr_xy_mm)
            sim_t = torch.from_numpy(sim_img).float().permute(2, 0, 1) / 255.0
            all_sim_frames.append(_to_uint8(sim_t, img_size))
            # Encode+decode current env state → model's internal perception
            raw_px = _env_pixels_raw(env)  # [1, 1, 3, H, W] raw [0,1]
            with torch.inference_mode():
                pred_frame = encode_decode(model, decoder, raw_px)  # [3, H, W]
            all_pred_frames.append(_to_uint8(pred_frame, img_size))

        if total_env_steps >= total_steps:
            break

    env.close()

    # ── Assemble 3-up frames ──────────────────────────────────────────
    gap = np.full((img_size, 8, 3), 0, dtype=np.uint8)
    side_by_side = []
    for t in range(min(len(all_pred_frames), len(all_sim_frames))):
        pred = all_pred_frames[t]
        sim = all_sim_frames[t]
        row = np.concatenate([pred, gap, sim, gap, goal_frame], axis=1)
        side_by_side.append(row)

    if not side_by_side:
        print("\n  ✗ No frames generated")
        return None

    # ── Write MP4 ─────────────────────────────────────────────────────
    out_path = output_path or (ckpt_path.parent / f"cem_rollout_{dataset_path.stem}.mp4")
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(out_path), fps=fps, format="FFMPEG", codec="libx264")
    for frame in side_by_side:
        writer.append_data(frame)
    writer.close()

    print(f"✓ Video written: {out_path}  ({len(side_by_side)} frames)")
    print("  Layout: [reconstruction | simulator | goal]")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CEM internal rollout with decoder predictions"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--receding-horizon", type=int, default=15)
    parser.add_argument("--total-steps", type=int, default=60,
                        help="Total physical env steps across all cycles (default: 60 = ~4 cycles)")
    parser.add_argument("--debug-costs", action="store_true",
                        help="Print cost breakdown for 3 hand-coded plans and exit (no video)")
    parser.add_argument("--lambda-latent", type=float, default=None)
    parser.add_argument("--lambda-off", type=float, default=None)
    parser.add_argument("--lambda-dir", type=float, default=None)
    parser.add_argument("--lambda-over", type=float, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument("--lambda-prog", type=float, default=None)
    parser.add_argument("--latent-cost-weight", type=float, default=None,
                        help="Shorthand for --lambda-latent (e.g. 0 to isolate geometric prior)")
    args = parser.parse_args()

    # Resolve --latent-cost-weight shorthand
    if args.latent_cost_weight is not None:
        args.lambda_latent = args.latent_cost_weight

    visualize_cem_rollout(
        dataset_path=args.dataset,
        ckpt_path=args.ckpt,
        decoder_weights=args.decoder,
        output_path=args.output,
        episode_idx=args.episode,
        start_step=args.start_step,
        device=args.device,
        img_size=args.img_size,
        fps=args.fps,
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        total_steps=args.total_steps,
        debug_costs=args.debug_costs,
        lambda_latent=args.lambda_latent,
        lambda_off=args.lambda_off,
        lambda_dir=args.lambda_dir,
        lambda_over=args.lambda_over,
        lambda_smooth=args.lambda_smooth,
        lambda_prog=args.lambda_prog,
    )


if __name__ == "__main__":
    main()
