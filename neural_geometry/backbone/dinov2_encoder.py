"""
neural_geometry/backbone/dinov2_encoder.py
============================================
DINOv2 Vision Transformer backbone for multi-view plant feature extraction.

Architecture Overview
---------------------
Each RGB (or RGB-D) view is tokenised into patches and processed by a
pre-trained DINOv2 ViT.  The resulting CLS token and patch tokens are
used for two purposes:

1. **Representation head**  : CLS token → species recognition / trait encoding
2. **Dense feature map**    : patch tokens → bilinearly up-sampled to a dense
                               feature map F_v : Ω_v → R^d used downstream by
                               the Volumetric Transformer.

Self-supervised pre-training (DINO)
------------------------------------
The teacher encoder  f'_θ  produces target representations.
The student encoder  f_θ   learns by minimising a cross-entropy loss over
multi-crop augmented views.  See pretraining/dino_pretraining.py for the
training loop.

Usage
-----
    from neural_geometry.backbone.dinov2_encoder import DINOv2Encoder
    enc = DINOv2Encoder(model_name="dinov2_vitb14")
    enc.load_pretrained()
    features = enc.encode_multiview(views)   # list of (H,W,3) RGB arrays
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import List, Dict, Optional, Tuple
import numpy as np

# DINOv2 requires torch; gracefully degrade if not installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[DINOv2] PyTorch not available — DINOv2Encoder will run in stub mode")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DINOV2_MODELS = {
    "dinov2_vits14":  {"embed_dim": 384,  "patch_size": 14},
    "dinov2_vitb14":  {"embed_dim": 768,  "patch_size": 14},
    "dinov2_vitl14":  {"embed_dim": 1024, "patch_size": 14},
    "dinov2_vitg14":  {"embed_dim": 1536, "patch_size": 14},
}


# ---------------------------------------------------------------------------
# DINOv2 Encoder
# ---------------------------------------------------------------------------

class DINOv2Encoder:
    """
    Wraps a DINOv2 ViT for multi-view plant image encoding.

    Parameters
    ----------
    model_name : one of DINOV2_MODELS keys (default: dinov2_vitb14)
    device     : 'cuda', 'cpu', or None (auto-detect)
    freeze     : if True, freeze all backbone weights (feature extractor mode)
    """

    def __init__(self,
                 model_name: str = "dinov2_vitb14",
                 device: Optional[str] = None,
                 freeze: bool = True):

        self.model_name = model_name
        self.cfg        = DINOV2_MODELS.get(model_name, DINOV2_MODELS["dinov2_vitb14"])
        self.embed_dim  = self.cfg["embed_dim"]
        self.patch_size = self.cfg["patch_size"]
        self.model      = None
        self.freeze     = freeze

        if device is None and TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device or "cpu"

        print(f"[DINOv2] model={model_name}  embed_dim={self.embed_dim}  "
              f"device={self.device}")

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def load_pretrained(self, local_path: Optional[str] = None) -> bool:
        """
        Load DINOv2 weights from torch.hub or a local checkpoint.

        Parameters
        ----------
        local_path : path to a locally saved checkpoint (.pth); if None,
                     downloads from torch.hub (requires internet)
        """
        if not TORCH_AVAILABLE:
            print("[DINOv2] Cannot load — PyTorch not available")
            return False

        try:
            if local_path and Path(local_path).exists():
                self.model = torch.load(local_path, map_location=self.device)
                print(f"[DINOv2] Loaded from {local_path}")
            else:
                # Official DINOv2 hub models
                self.model = torch.hub.load(
                    "facebookresearch/dinov2",
                    self.model_name,
                    pretrained=True
                )
                print(f"[DINOv2] Loaded {self.model_name} from torch.hub")

            self.model = self.model.to(self.device)

            if self.freeze:
                for p in self.model.parameters():
                    p.requires_grad = False
                print("[DINOv2] Backbone frozen")

            self.model.eval()
            return True

        except Exception as exc:
            print(f"[DINOv2] Load failed: {exc}")
            return False

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """
        Normalise and tensorise a single RGB image.
        Input: (H, W, 3)  uint8 or float32
        Output: (1, 3, H', W')  float32 tensor, ImageNet-normalised
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = image.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        img = (img - mean) / std                          # (H, W, 3)
        img = torch.from_numpy(img).permute(2, 0, 1)     # (3, H, W)
        return img.unsqueeze(0).to(self.device)           # (1, 3, H, W)

    # -----------------------------------------------------------------------
    # Single-view encoding
    # -----------------------------------------------------------------------

    def encode(self, image: np.ndarray) -> Dict[str, "torch.Tensor"]:
        """
        Encode a single view.

        Returns
        -------
        dict with:
            cls_token     : (1, embed_dim)   global image representation
            patch_tokens  : (1, N_patches, embed_dim)
            feature_map   : (1, embed_dim, H/patch, W/patch)  dense map F_v
        """
        if not TORCH_AVAILABLE or self.model is None:
            return self._stub_encode(image)

        with torch.no_grad():
            x = self._preprocess(image)
            out = self.model.forward_features(x)

        cls   = out["x_norm_clstoken"]               # (1, D)
        patch = out["x_norm_patchtokens"]             # (1, N, D)
        H     = image.shape[0] // self.patch_size
        W     = image.shape[1] // self.patch_size
        fmap  = patch.permute(0, 2, 1).reshape(1, self.embed_dim, H, W)

        return {"cls_token": cls, "patch_tokens": patch, "feature_map": fmap}

    # -----------------------------------------------------------------------
    # Multi-view encoding
    # -----------------------------------------------------------------------

    def encode_multiview(self,
                          views: List[np.ndarray],
                          angles_deg: Optional[List[int]] = None) -> Dict:
        """
        Encode all views and aggregate into a single representation.

        Parameters
        ----------
        views      : list of (H, W, 3) RGB arrays — one per captured angle
        angles_deg : capture angles for each view (for positional encoding)

        Returns
        -------
        dict with:
            per_view   : list of single-view encode() dicts
            aggregated : (1, embed_dim)  mean-pooled CLS across views
            feature_volume : (N_views, embed_dim, H/p, W/p)  stacked dense maps
        """
        per_view = [self.encode(v) for v in views]
        cls_stack = np.stack(
            [pv["cls_token"].cpu().numpy() if TORCH_AVAILABLE else pv["cls_token"]
             for pv in per_view], axis=0)   # (V, 1, D)
        aggregated = cls_stack.mean(axis=0)  # (1, D)

        return dict(per_view=per_view, aggregated=aggregated)

    # -----------------------------------------------------------------------
    # Stub (no PyTorch)
    # -----------------------------------------------------------------------

    def _stub_encode(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Return zero-filled arrays with correct shapes when PyTorch is absent."""
        H = image.shape[0] // self.patch_size
        W = image.shape[1] // self.patch_size
        N = H * W
        return {
            "cls_token":    np.zeros((1, self.embed_dim), dtype=np.float32),
            "patch_tokens": np.zeros((1, N, self.embed_dim), dtype=np.float32),
            "feature_map":  np.zeros((1, self.embed_dim, H, W), dtype=np.float32),
        }
