# model/entry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils: robust cfg access
# -----------------------------
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """
    Supports:
      - dict-like: cfg["a"]["b"]
      - attribute-like: cfg.a.b
      - OmegaConf-like (has get / attribute)
    key: dot-separated, e.g. "model.base_ch"
    """
    parts = key.split(".")
    cur = cfg
    for p in parts:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            # OmegaConf / SimpleNamespace / argparse namespace / dataclass
            if hasattr(cur, p):
                cur = getattr(cur, p)
            elif hasattr(cur, "get"):
                try:
                    cur = cur.get(p)
                except Exception:
                    return default
            else:
                return default
    return default if cur is None else cur


def _num_modalities_from_cfg(cfg: Any, fallback: int = 1) -> int:
    use_mods = _cfg_get(cfg, "data.use_modalities", None)
    if isinstance(use_mods, (list, tuple)) and len(use_mods) > 0:
        return int(len(use_mods))
    in_ch = _cfg_get(cfg, "model.in_ch", None)
    if isinstance(in_ch, int) and in_ch > 0:
        return int(in_ch)
    return int(fallback)


# -----------------------------
# Building blocks
# -----------------------------
def _norm3d(norm: str, ch: int) -> nn.Module:
    norm = (norm or "instance").lower()
    if norm in ("in", "instance", "instancenorm"):
        return nn.InstanceNorm3d(ch, affine=True)
    if norm in ("bn", "batch", "batchnorm"):
        return nn.BatchNorm3d(ch)
    if norm in ("gn", "group", "groupnorm"):
        # 8 groups is a safe default
        g = 8 if ch % 8 == 0 else 4 if ch % 4 == 0 else 1
        return nn.GroupNorm(g, ch)
    raise ValueError(f"Unknown norm: {norm}")


class ConvNormAct3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str = "instance",
        act: str = "leaky_relu",
        k: int = 3,
        s: int = 1,
        p: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.norm = _norm3d(norm, out_ch)
        if (act or "leaky_relu").lower() in ("relu",):
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout3d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class ResidualBlock3d(nn.Module):
    def __init__(self, ch: int, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        self.c1 = ConvNormAct3d(ch, ch, norm=norm, dropout=dropout)
        self.c2 = nn.Conv3d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = _norm3d(norm, ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.n2(self.c2(y))
        return self.act(x + y)


# -----------------------------
# Style conditioning (AdaIN-like)
# -----------------------------
class AdaIN3d(nn.Module):
    """
    Adaptive InstanceNorm for 3D features using a style vector.
    x: [B, C, H, W, D]
    style: [B, style_dim]
    """
    def __init__(self, ch: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm3d(ch, affine=False)
        self.fc = nn.Linear(style_dim, 2 * ch)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma_beta = self.fc(style)  # [B, 2C]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * h + beta


# -----------------------------
# Encoders / Decoder
# -----------------------------
class StructureEncoder3D(nn.Module):
    """
    Produces a multi-scale structure feature map z_s (keeps spatial detail).
    """
    def __init__(self, in_ch: int, base_ch: int = 32, depth: int = 4, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2 ** i) for i in range(self.depth)]

        self.stem = ConvNormAct3d(in_ch, chs[0], norm=norm, dropout=dropout)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.depth):
            self.down_blocks.append(nn.Sequential(
                ResidualBlock3d(chs[i], norm=norm, dropout=dropout),
                ResidualBlock3d(chs[i], norm=norm, dropout=dropout),
            ))
            if i < self.depth - 1:
                self.downsamples.append(ConvNormAct3d(chs[i], chs[i+1], norm=norm, k=3, s=2, p=1, dropout=dropout))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = []
        h = self.stem(x)
        for i in range(self.depth):
            h = self.down_blocks[i](h)
            feats.append(h)
            if i < self.depth - 1:
                h = self.downsamples[i](h)
        return {"z_s": feats[-1], "skips": feats}


class StyleEncoder3D(nn.Module):
    """
    Produces a global style vector z_t (no spatial detail).
    """
    def __init__(self, in_ch: int, base_ch: int = 32, depth: int = 4, style_dim: int = 128, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2 ** i) for i in range(self.depth)]

        self.stem = ConvNormAct3d(in_ch, chs[0], norm=norm, dropout=dropout)
        blocks = []
        for i in range(self.depth):
            blocks.append(ResidualBlock3d(chs[i], norm=norm, dropout=dropout))
            if i < self.depth - 1:
                blocks.append(ConvNormAct3d(chs[i], chs[i+1], norm=norm, k=3, s=2, p=1, dropout=dropout))
        self.body = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool3d(1)  # [B, C, 1,1,1]
        self.fc = nn.Sequential(
            nn.Linear(chs[-1], style_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.body(h)
        h = self.pool(h).flatten(1)
        z_t = self.fc(h)
        return z_t


class SegDecoder3D(nn.Module):
    """
    U-Net-like decoder to produce logits from structure features.
    Note: it only consumes structure path.
    """
    def __init__(self, base_ch: int = 32, depth: int = 4, out_ch: int = 1, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2 ** i) for i in range(self.depth)]

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(self.depth - 1, 0, -1):
            self.upconvs.append(nn.ConvTranspose3d(chs[i], chs[i-1], kernel_size=2, stride=2))
            self.dec_blocks.append(nn.Sequential(
                ConvNormAct3d(chs[i-1] + chs[i-1], chs[i-1], norm=norm, dropout=dropout),
                ResidualBlock3d(chs[i-1], norm=norm, dropout=dropout),
            ))

        self.head = nn.Conv3d(chs[0], out_ch, kernel_size=1, bias=True)

    def forward(self, z_s: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        h = z_s
        # skips: [level0, level1, ..., level_last] where last corresponds to z_s
        for idx, i in enumerate(range(self.depth - 1, 0, -1)):
            h = self.upconvs[idx](h)
            # Handle odd shapes due to resize/spacing differences
            s = skips[i-1]
            if h.shape[-3:] != s.shape[-3:]:
                h = F.interpolate(h, size=s.shape[-3:], mode="trilinear", align_corners=False)
            h = torch.cat([h, s], dim=1)
            h = self.dec_blocks[idx](h)
        logits = self.head(h)
        return logits


class ReconGenerator3D(nn.Module):
    """
    Reconstructs X from (structure feature map z_s) + (style vector z_t).
    Uses AdaIN modulation so style has an explicit role.
    """
    def __init__(self, out_ch: int, base_ch: int = 32, depth: int = 4, style_dim: int = 128, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2 ** i) for i in range(self.depth)]

        self.adains = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Start from deepest structure map channel = chs[-1]
        for i in range(self.depth - 1, 0, -1):
            self.adains.append(AdaIN3d(chs[i], style_dim))
            self.blocks.append(nn.Sequential(
                ConvNormAct3d(chs[i], chs[i], norm=norm, dropout=dropout),
                ResidualBlock3d(chs[i], norm=norm, dropout=dropout),
            ))
            self.upconvs.append(nn.ConvTranspose3d(chs[i], chs[i-1], kernel_size=2, stride=2))

        self.adain0 = AdaIN3d(chs[0], style_dim)
        self.block0 = nn.Sequential(
            ConvNormAct3d(chs[0], chs[0], norm=norm, dropout=dropout),
            ResidualBlock3d(chs[0], norm=norm, dropout=dropout),
        )
        self.out = nn.Conv3d(chs[0], out_ch, kernel_size=1, bias=True)

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, target_spatial: Optional[tuple[int, int, int]] = None) -> torch.Tensor:
        h = z_s
        # Up path
        for i in range(self.depth - 1, 0, -1):
            idx = (self.depth - 1) - i
            h = self.adains[idx](h, z_t)
            h = self.blocks[idx](h)
            h = self.upconvs[idx](h)

        # Final refine
        h = self.adain0(h, z_t)
        h = self.block0(h)

        if target_spatial is not None and h.shape[-3:] != target_spatial:
            h = F.interpolate(h, size=target_spatial, mode="trilinear", align_corners=False)

        x_hat = self.out(h)
        return x_hat


# -----------------------------
# Model output container
# -----------------------------
@dataclass
class DGSegOutput:
    logits: torch.Tensor         # [B, out_ch, H, W, D]
    recon: torch.Tensor          # [B, in_ch,  H, W, D] (reconstruct input)
    z_s: torch.Tensor            # deep structure feature map
    z_t: torch.Tensor            # style vector


# -----------------------------
# Full paradigm model
# -----------------------------
class CausalDGSegModel(nn.Module):
    """
    Paradigm-level DG segmentation:
      - Structure encoder Es -> z_s (spatial)
      - Style encoder Et     -> z_t (global)
      - Seg decoder uses ONLY z_s -> logits
      - Generator uses (z_s, z_t) -> recon, ensuring z_t participates and disentanglement is meaningful

    IMPORTANT for DDP:
      - During training, use compute_dg_losses() so recon loss backprops into Et + G.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        style_dim: int = 128,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)

        self.Es = StructureEncoder3D(in_ch=in_ch, base_ch=base_ch, depth=depth, norm=norm, dropout=dropout)
        self.Et = StyleEncoder3D(in_ch=in_ch, base_ch=base_ch, depth=depth, style_dim=style_dim, norm=norm, dropout=dropout)

        self.Dseg = SegDecoder3D(base_ch=base_ch, depth=depth, out_ch=out_ch, norm=norm, dropout=dropout)
        self.G = ReconGenerator3D(out_ch=in_ch, base_ch=base_ch, depth=depth, style_dim=style_dim, norm=norm, dropout=dropout)

    def forward(self, x: torch.Tensor) -> DGSegOutput:
        # x: [B, C, *, *, *]
        s_pack = self.Es(x)
        z_s = s_pack["z_s"]
        skips = s_pack["skips"]

        z_t = self.Et(x)  # [B, style_dim]

        logits = self.Dseg(z_s=z_s, skips=skips)
        recon = self.G(z_s=z_s, z_t=z_t, target_spatial=x.shape[-3:])

        return DGSegOutput(logits=logits, recon=recon, z_s=z_s, z_t=z_t)


# -----------------------------
# Loss pack for "all params participate"
# -----------------------------
def compute_dg_losses(
    out: Union[DGSegOutput, Dict[str, torch.Tensor]],
    x: torch.Tensor,
    y: torch.Tensor,
    seg_loss_fn: nn.Module,
    recon_weight: float = 0.1,
    inv_weight: float = 0.0,
    out2: Optional[Union[DGSegOutput, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict with:
      - loss_total: used for backward()
      - loss_seg
      - loss_recon
      - loss_inv (optional; requires out2)

    Parameters:
      out: forward output for view-1
      x: original input tensor [B, C, H, W, D]
      y: label tensor; recommended shapes:
           - binary: [B, 1, H, W, D] with {0,1}
           - multi-class: [B, 1, ...] with class ids, depending on your seg_loss_fn
      seg_loss_fn: e.g. MONAI DiceCELoss
      recon_weight: MUST be > 0 in DDP if you want to guarantee Et/G participate.
      inv_weight: optional invariance term on z_s between two views (out and out2)
      out2: forward output for view-2 (another augmentation of same x) if you use invariance

    Notes:
      - If recon_weight == 0 and inv_weight == 0, Et/G may become unused params in DDP.
    """
    if isinstance(out, dict):
        logits = out["logits"]
        recon = out["recon"]
        z_s = out["z_s"]
    else:
        logits = out.logits
        recon = out.recon
        z_s = out.z_s

    loss_seg = seg_loss_fn(logits, y)

    # Reconstruction: encourages explicit factorization and ensures Et/G get gradients
    loss_recon = F.l1_loss(recon, x)

    loss_total = loss_seg + float(recon_weight) * loss_recon

    loss_inv = torch.tensor(0.0, device=loss_total.device)
    if inv_weight and inv_weight > 0:
        if out2 is None:
            raise ValueError("inv_weight>0 requires out2 (a second forward on another view).")
        if isinstance(out2, dict):
            z_s2 = out2["z_s"]
        else:
            z_s2 = out2.z_s
        # Invariance on deep structure map (encourage structure to be stable under style perturbations)
        loss_inv = F.mse_loss(z_s, z_s2)
        loss_total = loss_total + float(inv_weight) * loss_inv

    return {
        "loss_total": loss_total,
        "loss_seg": loss_seg.detach(),
        "loss_recon": loss_recon.detach(),
        "loss_inv": loss_inv.detach(),
    }


# -----------------------------
# Single entry for main.py
# -----------------------------
def build_model(cfg: Any) -> nn.Module:
    """
    main.py usage:
        from model import build_model
        model = build_model(cfg)

    Reads:
      - data.use_modalities -> in_ch
      - model.out_ch (default 1)
      - model.base_ch (default 32)
      - model.depth (default 4)
      - model.style_dim (default 128)
      - model.norm (default 'instance')
      - model.dropout (default 0.0)
    """
    in_ch = _num_modalities_from_cfg(cfg, fallback=1)
    out_ch = int(_cfg_get(cfg, "model.out_ch", 1))
    base_ch = int(_cfg_get(cfg, "model.base_ch", 32))
    depth = int(_cfg_get(cfg, "model.depth", 4))
    style_dim = int(_cfg_get(cfg, "model.style_dim", 128))
    norm = str(_cfg_get(cfg, "model.norm", "instance"))
    dropout = float(_cfg_get(cfg, "model.dropout", 0.0))

    model = CausalDGSegModel(
        in_ch=in_ch,
        out_ch=out_ch,
        base_ch=base_ch,
        depth=depth,
        style_dim=style_dim,
        norm=norm,
        dropout=dropout,
    )
    return model
