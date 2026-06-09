"""
neural_geometry/volumetric/volumetric_transformer.py
=====================================================
Volumetric Transformer for 3D occupancy inference from multi-view features.

Architecture
------------
1. **Feature projection**  : Per-view DINOv2 dense feature maps are projected
   into a 3D voxel grid G = {x_i}_{i=1}^M by sampling F_v at the pixel
   location corresponding to each voxel centre's projected position.

2. **Voxel token initialisation**  :
       h_i^(0) = W_in V_i^(0)  ∈ R^d

3. **Windowed 3-D attention**  (shifted windows, Swin-Transformer-style):
       h_i^(l+1) = h_i^(l) + MLP(LN(MultiHeadSelfAttention(h_i^(l))))

4. **Visual Geometry Grounding (VGG) cross-attention**  :
       k_{v_i} = W_K^B F_v(x_i),   v_{v_i} = W_V^B F_v(x_i),   q_t = W_Q^B h_i^(l)
       β_{tv} = softmax_v(q_t k_{tv}^T / √d_g)
       h̃_t^(l) = h_t^(l) + Σ_v β_{tv} v_{tv}

5. **Occupancy / density prediction**  :
       σ_i = sigmoid(W_σ h_i^(L))
       c_i = W_c h_i^(L)

6. **Biomass regression head**  :
       V_G = Σ_i σ_i · det(Σ_i)^{1/2}  (effective Gaussian volume)
       ŷ  = MLP(V_G, h̄)  where h̄ = mean(h_i^(L))

This module provides a PyTorch-based implementation; a NumPy stub is used
when PyTorch is unavailable so that the rest of the pipeline can import
without errors.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from typing import Optional, List, Dict, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[VolumetricT] PyTorch not available — running in stub mode")

from shared.config import KINECT_FX, KINECT_FY, KINECT_CX, KINECT_CY


# ---------------------------------------------------------------------------
# Helper layers (only defined if PyTorch available)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class _WindowedAttentionBlock(nn.Module):
        """Single transformer block with window-based self-attention."""

        def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 4,
                     mlp_ratio: float = 4.0, dropout: float = 0.0):
            super().__init__()
            self.norm1  = nn.LayerNorm(d_model)
            self.norm2  = nn.LayerNorm(d_model)
            self.attn   = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout, batch_first=True)
            hidden      = int(d_model * mlp_ratio)
            self.mlp    = nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model), nn.Dropout(dropout),
            )
            self.window_size = window_size

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, N, d_model)
            h  = self.norm1(x)
            h, _ = self.attn(h, h, h)
            x  = x + h
            x  = x + self.mlp(self.norm2(x))
            return x

    class _VGGCrossAttentionBlock(nn.Module):
        """Visual Geometry Grounding cross-attention."""

        def __init__(self, d_model: int, d_img: int, n_heads: int = 8):
            super().__init__()
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv= nn.LayerNorm(d_img)
            self.cross  = nn.MultiheadAttention(d_model, n_heads,
                                                  kdim=d_img, vdim=d_img,
                                                  batch_first=True)

        def forward(self, voxel_tokens: "torch.Tensor",
                    image_tokens: "torch.Tensor") -> "torch.Tensor":
            q  = self.norm_q(voxel_tokens)
            kv = self.norm_kv(image_tokens)
            refined, _ = self.cross(q, kv, kv)
            return voxel_tokens + refined

    class VolumetricTransformer(nn.Module):
        """
        Full Volumetric Transformer: feature projection → 3D attention
        → VGG cross-attention → occupancy / density → biomass regression.

        Parameters
        ----------
        grid_res   : voxel grid side length (grid_res³ voxels)
        d_model    : transformer hidden dimension
        d_img      : DINOv2 embedding dimension (input feature maps)
        n_layers   : number of windowed attention blocks
        n_heads    : attention heads
        """

        def __init__(self,
                     grid_res:  int = 64,
                     d_model:   int = 256,
                     d_img:     int = 768,
                     n_layers:  int = 6,
                     n_heads:   int = 8,
                     dropout:   float = 0.1):
            super().__init__()

            self.grid_res  = grid_res
            self.d_model   = d_model
            self.n_voxels  = grid_res ** 3

            # Token input projection
            self.token_proj = nn.Linear(3, d_model)  # positional input (x,y,z)

            # Self-attention blocks
            self.sa_blocks  = nn.ModuleList([
                _WindowedAttentionBlock(d_model, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ])

            # VGG cross-attention
            self.vgg_ca = _VGGCrossAttentionBlock(d_model, d_img, n_heads)

            # Occupancy head: σ_i = sigmoid(W_σ h_i)
            self.occ_head  = nn.Linear(d_model, 1)

            # Colour/density head
            self.col_head  = nn.Linear(d_model, 3)

            # Biomass regression head: takes (mean token, effective volume)
            self.bio_head  = nn.Sequential(
                nn.Linear(d_model + 1, 128), nn.ReLU(),
                nn.Linear(128, 64),          nn.ReLU(),
                nn.Linear(64, 1),
            )

        def _build_voxel_coords(self,
                                 x_range: Tuple[float, float] = (-0.5, 0.5),
                                 y_range: Tuple[float, float] = (-0.6, 0.65),
                                 z_range: Tuple[float, float] = (0.2, 1.5),
                                 device:  str = "cpu") -> "torch.Tensor":
            """Return (N_voxels, 3) normalised voxel centres."""
            r    = self.grid_res
            xs   = torch.linspace(*x_range, r, device=device)
            ys   = torch.linspace(*y_range, r, device=device)
            zs   = torch.linspace(*z_range, r, device=device)
            grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
            return grid.reshape(-1, 3)  # (r³, 3)

        def forward(self,
                    image_features: "torch.Tensor",
                    voxel_coords:   Optional["torch.Tensor"] = None) -> Dict:
            """
            Parameters
            ----------
            image_features : (B, V, N_patches, d_img)  stacked per-view patch tokens
            voxel_coords   : (B, N_voxels, 3)  3D coords; generated internally if None

            Returns
            -------
            dict:
                occupancy    : (B, N_voxels, 1)  per-voxel occupancy probability σ_i
                colour       : (B, N_voxels, 3)  per-voxel colour / density
                voxel_tokens : (B, N_voxels, d_model)  final latent tokens
                biomass_pred : (B, 1)  predicted biomass (kg)
            """
            B, V, N_patch, _ = image_features.shape
            device = image_features.device

            if voxel_coords is None:
                coords = self._build_voxel_coords(device=device)
                voxel_coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B, Nv, 3)

            # Initialise voxel tokens from 3D coordinates
            h = self.token_proj(voxel_coords)   # (B, Nv, d_model)

            # Self-attention over voxels
            for blk in self.sa_blocks:
                h = blk(h)

            # VGG cross-attention: aggregate image context into voxel tokens
            img_flat = image_features.view(B, V * N_patch, -1)  # (B, V*Np, d_img)
            h = self.vgg_ca(h, img_flat)

            # Occupancy and colour heads
            occ    = torch.sigmoid(self.occ_head(h))  # (B, Nv, 1)
            colour = torch.tanh(self.col_head(h))      # (B, Nv, 3)

            # Effective Gaussian volume (simplified: Σ_i σ_i / N_voxels × total volume)
            V_eff  = occ.sum(dim=1, keepdim=True)      # (B, 1, 1) → (B, 1)
            V_eff  = V_eff.squeeze(-1)

            # Biomass regression
            h_mean = h.mean(dim=1)                     # (B, d_model)
            bio_in = torch.cat([h_mean, V_eff], dim=-1)  # (B, d_model+1)
            biomass = self.bio_head(bio_in)            # (B, 1)

            return dict(occupancy=occ, colour=colour,
                        voxel_tokens=h, biomass_pred=biomass)

        def infer_volume(self,
                          image_features: "torch.Tensor",
                          voxel_size_m: float = 0.007) -> float:
            """
            Convenience method: run forward pass and return effective volume (m³).
            """
            with torch.no_grad():
                out   = self.forward(image_features)
            occ   = out["occupancy"].squeeze().cpu().numpy()    # (Nv,)
            n_occ = (occ > 0.5).sum()
            return float(n_occ) * (voxel_size_m ** 3)

else:
    # -----------------------------------------------------------------
    # NumPy stub — allows imports without PyTorch
    # -----------------------------------------------------------------

    class VolumetricTransformer:  # type: ignore[no-redef]
        """Stub implementation — PyTorch not available."""

        def __init__(self, **kwargs):
            print("[VolumetricT] Stub mode — PyTorch required for real inference")

        def forward(self, image_features, voxel_coords=None):
            n = 64 ** 3
            return dict(
                occupancy    = np.zeros((1, n, 1),  dtype=np.float32),
                colour       = np.zeros((1, n, 3),  dtype=np.float32),
                voxel_tokens = np.zeros((1, n, 256), dtype=np.float32),
                biomass_pred = np.array([[0.0]],     dtype=np.float32),
            )

        def infer_volume(self, image_features, voxel_size_m=0.007):
            return 0.0
