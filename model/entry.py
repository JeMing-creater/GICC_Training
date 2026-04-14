from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception as e:  # pragma: no cover
    Mamba = None
    _MAMBA_IMPORT_ERROR = e
else:
    _MAMBA_IMPORT_ERROR = None


# -----------------------------
# Utils: robust cfg access
# -----------------------------
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    parts = key.split(".")
    cur = cfg
    for p in parts:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
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
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
            groups=groups,
        )
        self.norm = _norm3d(norm, out_ch)
        if (act or "leaky_relu").lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif (act or "leaky_relu").lower() == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop = (
            nn.Dropout3d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        )

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


class AdaIN3d(nn.Module):
    def __init__(self, ch: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm3d(ch, affine=False)
        self.fc = nn.Linear(style_dim, 2 * ch)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma_beta = self.fc(style)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * h + beta


# -----------------------------
# Encoders
# -----------------------------
class StructureEncoder3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 32,
        depth: int = 4,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2**i) for i in range(self.depth)]

        self.stem = ConvNormAct3d(in_ch, chs[0], norm=norm, dropout=dropout)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.depth):
            self.down_blocks.append(
                nn.Sequential(
                    ResidualBlock3d(chs[i], norm=norm, dropout=dropout),
                    ResidualBlock3d(chs[i], norm=norm, dropout=dropout),
                )
            )
            if i < self.depth - 1:
                self.downsamples.append(
                    ConvNormAct3d(
                        chs[i], chs[i + 1], norm=norm, k=3, s=2, p=1, dropout=dropout
                    )
                )

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        h = self.stem(x)
        for i in range(self.depth):
            h = self.down_blocks[i](h)
            feats.append(h)
            if i < self.depth - 1:
                h = self.downsamples[i](h)
        return {"z_s": feats[-1], "skips": feats}


class StyleEncoder3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 32,
        depth: int = 4,
        style_dim: int = 128,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2**i) for i in range(self.depth)]

        self.stem = ConvNormAct3d(in_ch, chs[0], norm=norm, dropout=dropout)
        blocks: List[nn.Module] = []
        for i in range(self.depth):
            blocks.append(ResidualBlock3d(chs[i], norm=norm, dropout=dropout))
            if i < self.depth - 1:
                blocks.append(
                    ConvNormAct3d(
                        chs[i], chs[i + 1], norm=norm, k=3, s=2, p=1, dropout=dropout
                    )
                )
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(chs[-1], style_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.body(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)


# -----------------------------
# Innovation 1: deformation-aware canonicalization
# -----------------------------
class SkipAdapter3D(nn.Module):
    def __init__(self, skip_ch: int, canon_ch: int, norm: str = "instance"):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(
                canon_ch * 2, skip_ch, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Sigmoid(),
        )
        self.adapter = nn.Sequential(
            ConvNormAct3d(skip_ch, skip_ch, norm=norm, act="gelu", k=3, s=1, p=1),
            nn.Conv3d(skip_ch, skip_ch, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(
        self, skip: torch.Tensor, z_up: torch.Tensor, deform_up: torch.Tensor
    ) -> torch.Tensor:
        gate = self.gate(torch.cat([z_up, deform_up], dim=1))
        return self.adapter(skip + skip * gate)


class TriMambaCanonicalizer(nn.Module):
    def __init__(
        self,
        dim: int,
        skip_channels: Sequence[int],
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_slices: int = 8,
        norm: str = "instance",
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                f"mamba_ssm is required but unavailable: {_MAMBA_IMPORT_ERROR}"
            )

        self.dim = int(dim)
        self.skip_channels = list(skip_channels)
        self.norm_tokens = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )
        self.dir_inner_dim = self.mamba.d_inner
        self.dir_proj = nn.Conv3d(
            self.dir_inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.longitudinal_refine = nn.Sequential(
            ConvNormAct3d(dim, dim, norm=norm, act="gelu", k=3, s=1, p=1),
            ResidualBlock3d(dim, norm=norm, dropout=0.0),
        )
        self.slice_anchor = nn.Sequential(
            ConvNormAct3d(dim, dim, norm=norm, act="gelu", k=3, s=1, p=1),
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.deform_sense = nn.Sequential(
            ConvNormAct3d(dim, dim, norm=norm, act="gelu", k=3, s=1, p=1),
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.anchor_gate = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.align = nn.Sequential(
            ConvNormAct3d(dim * 3, dim, norm=norm, act="gelu", k=3, s=1, p=1),
            ResidualBlock3d(dim, norm=norm, dropout=0.0),
        )
        self.skip_adapters = nn.ModuleList(
            [
                SkipAdapter3D(skip_ch=ch, canon_ch=dim, norm=norm)
                for ch in self.skip_channels
            ]
        )

    def _to_3d(
        self, x_seq: torch.Tensor, img_dims: Tuple[int, int, int], project: bool = False
    ) -> torch.Tensor:
        b = x_seq.shape[0]
        num_voxels = img_dims[0] * img_dims[1] * img_dims[2]
        if x_seq.shape[1] == num_voxels:
            feat = x_seq.transpose(1, 2).reshape(b, x_seq.shape[2], *img_dims)
        elif x_seq.shape[2] == num_voxels:
            feat = x_seq.reshape(b, x_seq.shape[1], *img_dims)
        else:
            raise ValueError(
                f"Cannot reshape tensor with shape {tuple(x_seq.shape)} to 3D dims {img_dims}; "
                f"expected one dimension to equal {num_voxels}."
            )
        if project:
            feat = self.dir_proj(feat)
        return feat

    def forward(
        self, z_s: torch.Tensor, skips: Sequence[torch.Tensor]
    ) -> Dict[str, Any]:
        if len(skips) != len(self.skip_channels) + 1:
            raise ValueError(
                f"Expected {len(self.skip_channels) + 1} skip tensors, got {len(skips)}."
            )

        b, c, h, w, z = z_s.shape
        assert c == self.dim
        img_dims = (h, w, z)

        x_flat = z_s.reshape(b, c, -1).transpose(-1, -2)
        x_norm = self.norm_tokens(x_flat)

        out, fwd, bwd, slc = self.mamba(x_norm)

        out_m = self._to_3d(out, img_dims)
        fwd_m = self._to_3d(fwd, img_dims, project=True)
        bwd_m = self._to_3d(bwd, img_dims, project=True)
        slc_m = self._to_3d(slc, img_dims, project=True)

        long_consensus = self.longitudinal_refine(0.5 * (fwd_m + bwd_m))
        deform_residual = torch.abs(fwd_m - bwd_m)
        deform_gate = self.deform_sense(deform_residual)
        slice_anchor = self.slice_anchor(slc_m)
        anchor_gate = self.anchor_gate(torch.cat([long_consensus, slice_anchor], dim=1))
        canon_anchor = anchor_gate * long_consensus + (1.0 - anchor_gate) * slice_anchor

        z_s_canon = (
            self.align(
                torch.cat([out_m, canon_anchor, deform_gate * deform_residual], dim=1)
            )
            + z_s
        )

        canon_skips: List[torch.Tensor] = []
        for idx, s in enumerate(skips[:-1]):
            z_up = F.interpolate(
                z_s_canon, size=s.shape[-3:], mode="trilinear", align_corners=False
            )
            d_up = F.interpolate(
                deform_gate, size=s.shape[-3:], mode="trilinear", align_corners=False
            )
            canon_skips.append(self.skip_adapters[idx](s, z_up, d_up))
        canon_skips.append(z_s_canon)

        deform_score = deform_residual.mean(dim=(1, 2, 3, 4), keepdim=False)

        return {
            "z_s_canon": z_s_canon,
            "canon_skips": canon_skips,
            "route": torch.stack(
                [
                    long_consensus.mean(dim=(1, 2, 3, 4)),
                    slice_anchor.mean(dim=(1, 2, 3, 4)),
                    deform_residual.mean(dim=(1, 2, 3, 4)),
                ],
                dim=1,
            ),
            "direction_features": {
                "fwd": fwd_m,
                "bwd": bwd_m,
                "slc": slc_m,
                "long_consensus": long_consensus,
                "slice_anchor": slice_anchor,
                "deform_residual": deform_residual,
            },
            "deform_gate": deform_gate,
            "deform_score": deform_score,
        }


# -----------------------------
# Segmentation decoder
# -----------------------------
class MultiModalSegDecoder3D(nn.Module):
    def __init__(
        self,
        base_ch: int = 32,
        depth: int = 4,
        out_ch: int = 1,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2**i) for i in range(self.depth)]

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(chs[i], chs[i - 1], kernel_size=2, stride=2)
            )
            self.dec_blocks.append(
                nn.Sequential(
                    ConvNormAct3d(
                        chs[i - 1] + chs[i - 1], chs[i - 1], norm=norm, dropout=dropout
                    ),
                    ResidualBlock3d(chs[i - 1], norm=norm, dropout=dropout),
                )
            )
        self.head = nn.Conv3d(chs[0], out_ch, kernel_size=1, bias=True)

    def forward(
        self, z_s_canon: torch.Tensor, canon_skips: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = z_s_canon
        for idx, i in enumerate(range(self.depth - 1, 0, -1)):
            h = self.upconvs[idx](h)
            s = canon_skips[i - 1]
            if h.shape[-3:] != s.shape[-3:]:
                h = F.interpolate(
                    h, size=s.shape[-3:], mode="trilinear", align_corners=False
                )
            h = torch.cat([h, s], dim=1)
            h = self.dec_blocks[idx](h)
        return self.head(h), h


# -----------------------------
# Innovation 2: classification-oriented invasive reasoning
# -----------------------------
class BoundaryRelationAttention(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 128,
        num_heads: int = 4,
        norm: str = "instance",
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_ch = max(hidden_ch, num_heads)
        if hidden_ch % num_heads != 0:
            hidden_ch = (hidden_ch // num_heads + 1) * num_heads

        self.core_proj = nn.Sequential(
            ConvNormAct3d(in_ch, hidden_ch, norm=norm, act="gelu", k=1, s=1, p=0),
            ResidualBlock3d(hidden_ch, norm=norm, dropout=dropout),
        )
        self.front_proj = nn.Sequential(
            ConvNormAct3d(in_ch, hidden_ch, norm=norm, act="gelu", k=3, s=1, p=1),
            ConvNormAct3d(
                hidden_ch,
                hidden_ch,
                norm=norm,
                act="gelu",
                k=3,
                s=1,
                p=1,
                groups=hidden_ch,
            ),
        )
        self.context_conv1 = ConvNormAct3d(
            in_ch, hidden_ch, norm=norm, act="gelu", k=3, s=1, p=2
        )
        self.context_conv1.conv = nn.Conv3d(
            in_ch, hidden_ch, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )
        self.context_conv2 = ConvNormAct3d(
            hidden_ch, hidden_ch, norm=norm, act="gelu", k=3, s=1, p=2
        )
        self.context_conv2.conv = nn.Conv3d(
            hidden_ch,
            hidden_ch,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            bias=False,
        )

        self.token_attn = nn.MultiheadAttention(
            hidden_ch, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.token_norm = nn.LayerNorm(hidden_ch)
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ch * 2, hidden_ch),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(
                hidden_ch * 3, hidden_ch, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.InstanceNorm3d(hidden_ch),
            nn.GELU(),
            nn.Conv3d(
                hidden_ch, hidden_ch, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvNormAct3d(
                hidden_ch * 3, hidden_ch, norm=norm, act="gelu", k=3, s=1, p=1
            ),
            ResidualBlock3d(hidden_ch, norm=norm, dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_ch * 2, hidden_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ch, 1),
        )

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(x, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_core = self.core_proj(x)
        f_front = self.front_proj(x)
        f_context = self.context_conv2(self.context_conv1(x))

        tokens = torch.stack(
            [self._gap(f_core), self._gap(f_front), self._gap(f_context)], dim=1
        )
        q = tokens[:, 1:2, :]
        attn_out, attn_weights = self.token_attn(q, tokens, tokens, need_weights=True)
        front_token = self.token_norm(q + attn_out)
        front_token = front_token + self.token_mlp(front_token)
        front_token = front_token.squeeze(1)

        gate = self.spatial_gate(torch.cat([f_core, f_front, f_context], dim=1))
        fused_spatial = self.fuse(
            torch.cat([f_core, gate * f_front, (1.0 - gate) * f_context], dim=1)
        )
        fused_vec = self._gap(fused_spatial)
        class_logit = self.classifier(torch.cat([front_token, fused_vec], dim=1))

        return {
            "class_logit": class_logit,
            "f_core": f_core,
            "f_front": f_front,
            "f_context": f_context,
            "f_fused": fused_spatial,
            "attn_weights": attn_weights,
        }


# -----------------------------
# Reconstruction generator
# -----------------------------
class ReconGenerator3D(nn.Module):
    def __init__(
        self,
        out_ch: int,
        base_ch: int = 32,
        depth: int = 4,
        style_dim: int = 128,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2**i) for i in range(self.depth)]

        self.adains = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            self.adains.append(AdaIN3d(chs[i], style_dim))
            self.blocks.append(
                nn.Sequential(
                    ConvNormAct3d(chs[i], chs[i], norm=norm, dropout=dropout),
                    ResidualBlock3d(chs[i], norm=norm, dropout=dropout),
                )
            )
            self.upconvs.append(
                nn.ConvTranspose3d(chs[i], chs[i - 1], kernel_size=2, stride=2)
            )

        self.adain0 = AdaIN3d(chs[0], style_dim)
        self.block0 = nn.Sequential(
            ConvNormAct3d(chs[0], chs[0], norm=norm, dropout=dropout),
            ResidualBlock3d(chs[0], norm=norm, dropout=dropout),
        )
        self.out = nn.Conv3d(chs[0], out_ch, kernel_size=1, bias=True)

    def forward(
        self,
        z_s_canon: torch.Tensor,
        z_t: torch.Tensor,
        target_spatial: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        h = z_s_canon
        for i in range(self.depth - 1, 0, -1):
            idx = (self.depth - 1) - i
            h = self.adains[idx](h, z_t)
            h = self.blocks[idx](h)
            h = self.upconvs[idx](h)

        h = self.adain0(h, z_t)
        h = self.block0(h)
        if target_spatial is not None and h.shape[-3:] != target_spatial:
            h = F.interpolate(
                h, size=target_spatial, mode="trilinear", align_corners=False
            )
        return self.out(h)


# -----------------------------
# Output container
# -----------------------------
@dataclass
class MultiTaskDGOutput:
    seg: torch.Tensor
    class_logit: torch.Tensor
    recon: torch.Tensor
    z_s: torch.Tensor
    z_s_canon: torch.Tensor
    z_t: torch.Tensor
    canon_skips: List[torch.Tensor]
    f_core: torch.Tensor
    f_front: torch.Tensor
    f_context: torch.Tensor
    f_fused: torch.Tensor
    route: torch.Tensor

    @property
    def logits(self) -> torch.Tensor:
        return self.seg


# -----------------------------
# Full multi-task model
# -----------------------------
class CausalDGMultiTaskModel(nn.Module):
    def __init__(
        self,
        in_ch: int,
        seg_out_ch: int,
        base_ch: int = 32,
        depth: int = 4,
        style_dim: int = 128,
        norm: str = "instance",
        dropout: float = 0.0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_num_slices: int = 8,
        cls_hidden_ch: int = 128,
        cls_num_heads: int = 4,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.seg_out_ch = int(seg_out_ch)
        self.depth = int(depth)
        self.base_ch = int(base_ch)

        self.Es = StructureEncoder3D(
            in_ch=in_ch, base_ch=base_ch, depth=depth, norm=norm, dropout=dropout
        )
        self.Et = StyleEncoder3D(
            in_ch=in_ch,
            base_ch=base_ch,
            depth=depth,
            style_dim=style_dim,
            norm=norm,
            dropout=dropout,
        )

        chs = [base_ch * (2**i) for i in range(depth)]
        deep_ch = chs[-1]
        self.canonicalizer = TriMambaCanonicalizer(
            dim=deep_ch,
            skip_channels=chs[:-1],
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            num_slices=mamba_num_slices,
            norm=norm,
        )
        self.seg_decoder = MultiModalSegDecoder3D(
            base_ch=base_ch,
            depth=depth,
            out_ch=seg_out_ch,
            norm=norm,
            dropout=dropout,
        )
        self.cls_branch = BoundaryRelationAttention(
            in_ch=base_ch,
            hidden_ch=cls_hidden_ch,
            num_heads=cls_num_heads,
            norm=norm,
            dropout=dropout,
        )
        self.recon = ReconGenerator3D(
            out_ch=in_ch,
            base_ch=base_ch,
            depth=depth,
            style_dim=style_dim,
            norm=norm,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, enable_cls_branch: bool = True
    ) -> MultiTaskDGOutput:
        s_pack = self.Es(x)
        z_s = s_pack["z_s"]
        skips = s_pack["skips"]

        z_t = self.Et(x)
        canon_pack = self.canonicalizer(z_s, skips)
        z_s_canon = canon_pack["z_s_canon"]
        canon_skips = canon_pack["canon_skips"]

        seg_logits, seg_feat = self.seg_decoder(z_s_canon, canon_skips)

        if enable_cls_branch:
            cls_pack = self.cls_branch(seg_feat)
            class_logit = cls_pack["class_logit"]
            f_core = cls_pack["f_core"]
            f_front = cls_pack["f_front"]
            f_context = cls_pack["f_context"]
            f_fused = cls_pack["f_fused"]
        else:
            b = seg_feat.shape[0]
            class_logit = seg_feat.new_zeros((b, 1))
            f_core = seg_feat.new_zeros(seg_feat.shape)
            f_front = seg_feat.new_zeros(seg_feat.shape)
            f_context = seg_feat.new_zeros(seg_feat.shape)
            f_fused = seg_feat.new_zeros(seg_feat.shape)

        x_hat = self.recon(z_s_canon, z_t, target_spatial=x.shape[-3:])

        return MultiTaskDGOutput(
            seg=seg_logits,
            class_logit=class_logit,
            recon=x_hat,
            z_s=z_s,
            z_s_canon=z_s_canon,
            z_t=z_t,
            canon_skips=list(canon_skips),
            f_core=f_core,
            f_front=f_front,
            f_context=f_context,
            f_fused=f_fused,
            route=canon_pack["route"],
        )


# -----------------------------
# Loss pack for multi-task training
# -----------------------------
def compute_dg_losses(
    out: Union[MultiTaskDGOutput, Dict[str, torch.Tensor]],
    x: torch.Tensor,
    y: torch.Tensor,
    seg_loss_fn: nn.Module,
    class_label: Optional[torch.Tensor] = None,
    cls_loss_fn: Optional[nn.Module] = None,
    recon_weight: float = 0.1,
    cls_weight: float = 1.0,
    inv_weight: float = 0.0,
    out2: Optional[Union[MultiTaskDGOutput, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    if isinstance(out, dict):
        seg_logits = out["seg"] if "seg" in out else out["logits"]
        class_logit = out.get("class_logit", None)
        recon = out["recon"]
        z_s_canon = out.get("z_s_canon", out.get("z_s", None))
    else:
        seg_logits = out.seg
        class_logit = out.class_logit
        recon = out.recon
        z_s_canon = out.z_s_canon

    loss_seg = seg_loss_fn(seg_logits, y)
    loss_recon = F.l1_loss(recon, x)
    loss_total = loss_seg + float(recon_weight) * loss_recon

    loss_cls = torch.tensor(0.0, device=loss_total.device)
    if class_label is not None and class_logit is not None and cls_weight > 0:
        if cls_loss_fn is None:
            cls_loss_fn = nn.BCEWithLogitsLoss()
        class_target = class_label.float().view(class_logit.shape)
        loss_cls = cls_loss_fn(class_logit, class_target)
        loss_total = loss_total + float(cls_weight) * loss_cls

    loss_inv = torch.tensor(0.0, device=loss_total.device)
    if inv_weight and inv_weight > 0:
        if out2 is None:
            raise ValueError("inv_weight>0 requires out2.")
        if isinstance(out2, dict):
            z_s2 = out2.get("z_s_canon", out2.get("z_s", None))
        else:
            z_s2 = out2.z_s_canon
        loss_inv = F.mse_loss(z_s_canon, z_s2)
        loss_total = loss_total + float(inv_weight) * loss_inv

    return {
        "loss_total": loss_total,
        "loss_seg": loss_seg.detach(),
        "loss_cls": loss_cls.detach(),
        "loss_recon": loss_recon.detach(),
        "loss_inv": loss_inv.detach(),
    }


# -----------------------------
# Single entry for main.py
# -----------------------------
def build_model(cfg: Any) -> nn.Module:
    in_ch = _num_modalities_from_cfg(cfg, fallback=1)
    seg_out_equals_modalities = bool(
        _cfg_get(cfg, "model.seg_out_equals_modalities", True)
    )
    seg_out_ch = (
        in_ch
        if seg_out_equals_modalities
        else int(_cfg_get(cfg, "model.out_ch", in_ch))
    )

    base_ch = int(_cfg_get(cfg, "model.base_ch", 32))
    depth = int(_cfg_get(cfg, "model.depth", 4))
    style_dim = int(_cfg_get(cfg, "model.style_dim", 128))
    norm = str(_cfg_get(cfg, "model.norm", "instance"))
    dropout = float(_cfg_get(cfg, "model.dropout", 0.0))

    mamba_d_state = int(_cfg_get(cfg, "model.mamba_d_state", 16))
    mamba_d_conv = int(_cfg_get(cfg, "model.mamba_d_conv", 4))
    mamba_expand = int(_cfg_get(cfg, "model.mamba_expand", 2))
    mamba_num_slices = int(_cfg_get(cfg, "model.mamba_num_slices", 8))

    cls_hidden_ch = int(_cfg_get(cfg, "model.cls_hidden_ch", 128))
    cls_num_heads = int(_cfg_get(cfg, "model.cls_num_heads", 4))

    return CausalDGMultiTaskModel(
        in_ch=in_ch,
        seg_out_ch=seg_out_ch,
        base_ch=base_ch,
        depth=depth,
        style_dim=style_dim,
        norm=norm,
        dropout=dropout,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_num_slices=mamba_num_slices,
        cls_hidden_ch=cls_hidden_ch,
        cls_num_heads=cls_num_heads,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch = 2
    modal = 3
    width = 128
    height = 128
    depth = 64

    x = torch.randn(batch, modal, width, height, depth, device=device)

    model = CausalDGMultiTaskModel(
        in_ch=modal,
        seg_out_ch=modal,
        base_ch=16,
        depth=4,
        style_dim=64,
        norm="instance",
        dropout=0.0,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_num_slices=8,
        cls_hidden_ch=64,
        cls_num_heads=4,
    ).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x, enable_cls_branch=True)

    print("=" * 80)
    print("Debug forward finished")
    print("=" * 80)
    print(f"Input shape           : {tuple(x.shape)}")
    print(f"Seg output shape      : {tuple(out.seg.shape)}")
    print(f"Class output shape    : {tuple(out.class_logit.shape)}")
    print(f"Recon output shape    : {tuple(out.recon.shape)}")
    print(f"Structure z_s shape   : {tuple(out.z_s.shape)}")
    print(f"Canon z_s shape       : {tuple(out.z_s_canon.shape)}")
    print(f"Style z_t shape       : {tuple(out.z_t.shape)}")
    print(f"Core feature shape    : {tuple(out.f_core.shape)}")
    print(f"Front feature shape   : {tuple(out.f_front.shape)}")
    print(f"Context feature shape : {tuple(out.f_context.shape)}")
    print(f"Fused feature shape   : {tuple(out.f_fused.shape)}")
    print(f"Route weight shape    : {tuple(out.route.shape)}")
    print(f"Route weights sample  : {out.route[0].detach().cpu().tolist()}")
    print(f"Class logits sample   : {out.class_logit[:, 0].detach().cpu().tolist()}")
