"""
neural_geometry/pretraining/dino_pretraining.py
================================================
DINO self-supervised pre-training loop for plant imagery.

Implements the teacher-student distillation objective:
  L_DINO = -Σ_v' P_teacher(v)  log P_student(v')

where v ∈ global crops, v' ∈ {global, local crops}

The teacher parameters f'_θ are updated as an exponential moving average
of the student f_θ, with a cosine schedule for the momentum coefficient.

Usage
-----
    python dino_pretraining.py --data_dir acquisition/dataset/specimens \\
                               --epochs 100 --batch_size 8
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import math
import json
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[DINOPretrain] PyTorch not available")

from shared.config import SPECIMENS_DIR, CAPTURE_ANGLES_DEG, NEURAL_GEOMETRY_DIR


# ---------------------------------------------------------------------------
# Multi-crop augmentation
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class MultiCropAugmentation:
        """
        Standard DINO multi-crop: 2 global crops (224×224) + N local crops (96×96).
        """

        def __init__(self,
                     n_global: int = 2,
                     n_local:  int = 6,
                     global_scale: tuple = (0.4, 1.0),
                     local_scale:  tuple = (0.05, 0.4)):
            colour_jitter = T.ColorJitter(0.4, 0.4, 0.2, 0.1)
            self.global_tf = T.Compose([
                T.RandomResizedCrop(224, scale=global_scale, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([colour_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            self.local_tf = T.Compose([
                T.RandomResizedCrop(96, scale=local_scale, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([colour_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            self.n_global = n_global
            self.n_local  = n_local

        def __call__(self, pil_image) -> list:
            global_views = [self.global_tf(pil_image) for _ in range(self.n_global)]
            local_views  = [self.local_tf(pil_image)  for _ in range(self.n_local)]
            return global_views + local_views


    # -----------------------------------------------------------------------
    # Plant image dataset
    # -----------------------------------------------------------------------

    class PlantImageDataset(Dataset):
        """
        Yields PIL images from acquisition/dataset/specimens/{id}/rgb/.
        Suitable for self-supervised pre-training (no labels required).
        """

        def __init__(self,
                     specimens_dir: Path = SPECIMENS_DIR,
                     cam_label:     str  = "A",
                     transform               = None):
            self.paths     = []
            self.transform = transform
            for spec_dir in sorted(specimens_dir.iterdir()):
                if not spec_dir.is_dir():
                    continue
                for angle in CAPTURE_ANGLES_DEG:
                    p = spec_dir / "rgb" / f"view_{angle:03d}deg_cam{cam_label}_rgb.jpg"
                    if p.exists():
                        self.paths.append(p)
            print(f"[Dataset] {len(self.paths)} images across all specimens")

        def __len__(self): return len(self.paths)

        def __getitem__(self, idx):
            from PIL import Image
            img = Image.open(str(self.paths[idx])).convert("RGB")
            if self.transform:
                return self.transform(img)
            return img


    # -----------------------------------------------------------------------
    # DINO loss
    # -----------------------------------------------------------------------

    class DINOLoss(nn.Module):
        """Centering + sharpening loss as in Caron et al. (2021)."""

        def __init__(self, out_dim: int, n_crops: int,
                     warmup_teacher_temp: float = 0.04,
                     teacher_temp: float = 0.07,
                     warmup_epochs: int = 30,
                     n_epochs: int = 100,
                     center_momentum: float = 0.9):
            super().__init__()
            self.n_crops    = n_crops
            self.t_temp     = teacher_temp
            self.s_temp     = 0.1
            self.cm         = center_momentum
            self.register_buffer("center", torch.zeros(1, out_dim))

            self.teacher_temp_schedule = np.concatenate([
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
                np.full(n_epochs - warmup_epochs, teacher_temp),
            ])

        def forward(self, student_out: list, teacher_out: list, epoch: int):
            s_out = [F.log_softmax(s / self.s_temp, dim=-1) for s in student_out]
            t_tmp = self.teacher_temp_schedule[epoch]
            t_out = [F.softmax((t - self.center) / t_tmp, dim=-1) for t in teacher_out]

            total_loss = 0.0
            n_terms    = 0
            for t in t_out:
                for i, s in enumerate(s_out):
                    if i == t_out.index(t): continue   # skip same view
                    total_loss -= (t * s).sum(dim=-1).mean()
                    n_terms += 1

            total_loss /= max(n_terms, 1)

            # Update center
            batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
            self.center  = self.center * self.cm + batch_center * (1 - self.cm)

            return total_loss


    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def run_pretraining(data_dir: Path    = SPECIMENS_DIR,
                        epochs:   int     = 100,
                        batch_size: int   = 8,
                        lr:       float   = 5e-4,
                        out_dim:  int     = 65536,
                        model_name: str   = "dinov2_vitb14",
                        save_dir: Path    = NEURAL_GEOMETRY_DIR / "pretraining" / "checkpoints"):
        """
        Run DINO self-supervised pre-training on the plant dataset.

        Checkpoints are saved every 10 epochs to save_dir/.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DINOPretrain] device={device}  epochs={epochs}  batch={batch_size}")

        augment = MultiCropAugmentation(n_global=2, n_local=6)
        dataset = PlantImageDataset(data_dir, transform=augment)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, drop_last=True)

        # Build student & teacher (same architecture, different weights)
        from neural_geometry.backbone.dinov2_encoder import DINOv2Encoder
        student_enc = DINOv2Encoder(model_name, device=device, freeze=False)
        teacher_enc = DINOv2Encoder(model_name, device=device, freeze=True)
        student_enc.load_pretrained()
        teacher_enc.load_pretrained()

        # DINO projection heads (not part of the backbone)
        proj_head = nn.Sequential(
            nn.Linear(student_enc.embed_dim, 2048), nn.GELU(),
            nn.Linear(2048, out_dim)
        ).to(device)

        optimiser = torch.optim.AdamW(
            list(student_enc.model.parameters()) + list(proj_head.parameters()),
            lr=lr, weight_decay=0.04
        )
        criterion = DINOLoss(out_dim=out_dim, n_crops=8,
                             n_epochs=epochs).to(device)

        momentum_schedule = np.linspace(0.996, 1.0, epochs)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                # batch is a list of views (global+local); each (B, C, H, W)
                all_views = [v.to(device) for v in batch]

                # Student forward on all views
                s_out = [proj_head(student_enc.model(v)) for v in all_views]
                # Teacher forward on global views only
                with torch.no_grad():
                    t_out = [proj_head(teacher_enc.model(v)) for v in all_views[:2]]

                loss = criterion(s_out, t_out, epoch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # EMA teacher update
                m = momentum_schedule[epoch]
                for ps, pt in zip(student_enc.model.parameters(),
                                  teacher_enc.model.parameters()):
                    pt.data = m * pt.data + (1 - m) * ps.data

                epoch_loss += loss.item()

            avg = epoch_loss / max(len(loader), 1)
            print(f"[DINOPretrain] Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

            if (epoch + 1) % 10 == 0:
                ckpt_path = save_dir / f"dino_epoch_{epoch+1:03d}.pth"
                torch.save(student_enc.model.state_dict(), str(ckpt_path))
                print(f"[DINOPretrain] Checkpoint → {ckpt_path}")

        print("[DINOPretrain] Pre-training complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch required for DINO pre-training.  "
              "Install with: pip install torch torchvision")
        sys.exit(1)

    p = argparse.ArgumentParser(description="DINO self-supervised pre-training")
    p.add_argument("--data_dir",   default=str(SPECIMENS_DIR))
    p.add_argument("--epochs",     default=100, type=int)
    p.add_argument("--batch_size", default=8,   type=int)
    p.add_argument("--lr",         default=5e-4, type=float)
    p.add_argument("--model",      default="dinov2_vitb14")
    args = p.parse_args()

    run_pretraining(
        data_dir   = Path(args.data_dir),
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        model_name = args.model,
    )
