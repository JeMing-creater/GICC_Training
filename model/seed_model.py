from __future__ import annotations
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


# =========================
# Output container
# =========================
@dataclass
class SeedDetectorOutput:
    seed_region_modal: torch.Tensor  # (B, Modal, w, h, z)
    seed_region_fused: torch.Tensor  # (B, 1, w, h, z)
    seed_coord_modal: torch.Tensor  # (B, Modal, 3) in detector grid coords
    seed_coord_fused: torch.Tensor  # (B, 3) in detector grid coords
    agreement_map: torch.Tensor  # (B, 1, w, h, z)
    modal_reliability: torch.Tensor  # (B, Modal, w, h, z)


# =========================
# Basic blocks
# =========================
class ConvNormAct3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: Tuple[int, int, int] = (3, 3, 3),
        s: Tuple[int, int, int] = (1, 1, 1),
        p: Tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                bias=False,
                groups=groups,
            ),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock3d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = ConvNormAct3d(ch, ch)
        self.c2 = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.c2(y)
        return self.act(x + y)


# =========================
# Z-only Mamba contextualizer
# =========================
class ZOnlyMamba(nn.Module):
    """
    沿 Z 方向建模，不 flatten 整个 3D 体，显存更稳。
    输入输出 shape 相同: (B, C, W, H, Z)
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        nslices: int = 8,
    ):
        super().__init__()

        self.dim = int(dim)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.nslices = int(nslices)

        self.norm = nn.LayerNorm(self.dim)

        self.mamba = Mamba(
            d_model=self.dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            bimamba_type="v3",
            nslices=self.nslices,
        )

        # Mamba 内部通道，一般等于 expand * dim
        self.inner_dim = int(getattr(self.mamba, "d_inner", self.dim * self.expand))

        # 静态投影，避免 forward 动态创建参数
        self.proj_from_dim_fwd = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.proj_from_dim_bwd = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.proj_from_dim_slc = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)

        self.proj_from_inner_fwd = nn.Conv3d(
            self.inner_dim, self.dim, kernel_size=1, bias=True
        )
        self.proj_from_inner_bwd = nn.Conv3d(
            self.inner_dim, self.dim, kernel_size=1, bias=True
        )
        self.proj_from_inner_slc = nn.Conv3d(
            self.inner_dim, self.dim, kernel_size=1, bias=True
        )

    def _reshape_back(
        self, t: torch.Tensor, B: int, W: int, H: int, Z: int
    ) -> torch.Tensor:
        """
        兼容:
        - (N, Z, dim)
        - (N, dim, Z)
        - (N, Z, inner_dim)
        - (N, inner_dim, Z)
        其中 N = B*W*H
        返回:
        - y: (B, C_in, W, H, Z)
        - c_in: 输入通道数（dim 或 inner_dim）
        """
        if t.dim() != 3:
            raise RuntimeError(f"Unexpected Mamba output shape: {tuple(t.shape)}")

        n = B * W * H
        if t.shape[0] != n:
            raise RuntimeError(
                f"Unexpected batch tokens in Mamba output: got {t.shape[0]}, expected {n}"
            )

        # 统一变成 (N, Z, C_out)
        if t.shape[1] == Z:
            y = t
        elif t.shape[2] == Z:
            y = t.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(
                f"Cannot infer valid Mamba output layout from {tuple(t.shape)} with expected Z={Z}"
            )

        c_in = int(y.shape[-1])
        if c_in not in (self.dim, self.inner_dim):
            raise RuntimeError(
                f"Unexpected channel dim after reshape: got {c_in}, expected one of ({self.dim}, {self.inner_dim})"
            )

        y = y.reshape(B, W, H, Z, c_in).permute(0, 4, 1, 2, 3).contiguous()
        return y, c_in

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, C, W, H, Z)

        返回:
            {
                "core": (B, C, W, H, Z),
                "rim":  (B, C, W, H, Z),
                "unc":  (B, C, W, H, Z),
            }
        """
        B, C, W, H, Z = x.shape
        if C != self.dim:
            raise RuntimeError(f"Expected channel dim {self.dim}, but got {C}")

        x_seq = x.permute(0, 2, 3, 4, 1).reshape(-1, Z, C).contiguous()
        x_seq = self.norm(x_seq)

        out, fwd, bwd, slc = self.mamba(x_seq)

        fwd_3d, fwd_c = self._reshape_back(fwd, B, W, H, Z)
        bwd_3d, bwd_c = self._reshape_back(bwd, B, W, H, Z)
        slc_3d, slc_c = self._reshape_back(slc, B, W, H, Z)

        if fwd_c == self.inner_dim:
            fwd_3d = self.proj_from_inner_fwd(fwd_3d)
        else:
            fwd_3d = self.proj_from_dim_fwd(fwd_3d)

        if bwd_c == self.inner_dim:
            bwd_3d = self.proj_from_inner_bwd(bwd_3d)
        else:
            bwd_3d = self.proj_from_dim_bwd(bwd_3d)

        if slc_c == self.inner_dim:
            slc_3d = self.proj_from_inner_slc(slc_3d)
        else:
            slc_3d = self.proj_from_dim_slc(slc_3d)

        return {
            "core": 0.5 * (fwd_3d + bwd_3d),
            "rim": slc_3d,
            "unc": torch.abs(fwd_3d - bwd_3d),
        }


# =========================
# Soft argmax utilities
# =========================
def soft_argmax_3d(logits: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """
    logits: (B, C, W, H, Z)
    return: (B, C, 3) with coords in [0, W-1], [0, H-1], [0, Z-1]
    """
    B, C, W, H, Z = logits.shape
    flat = logits.reshape(B, C, -1)
    prob = F.softmax(beta * flat, dim=-1).reshape(B, C, W, H, Z)

    xs = torch.linspace(0, W - 1, W, device=logits.device, dtype=logits.dtype).view(
        1, 1, W, 1, 1
    )
    ys = torch.linspace(0, H - 1, H, device=logits.device, dtype=logits.dtype).view(
        1, 1, 1, H, 1
    )
    zs = torch.linspace(0, Z - 1, Z, device=logits.device, dtype=logits.dtype).view(
        1, 1, 1, 1, Z
    )

    x_exp = (prob * xs).sum(dim=(2, 3, 4))
    y_exp = (prob * ys).sum(dim=(2, 3, 4))
    z_exp = (prob * zs).sum(dim=(2, 3, 4))

    return torch.stack([x_exp, y_exp, z_exp], dim=-1)


# =========================
# Target generation utilities
# =========================
def binary_erode_3d(
    mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1
) -> torch.Tensor:
    """
    mask: (B,1,W,H,Z) in {0,1}
    min-pooling based erosion
    """
    x = mask
    pad = kernel_size // 2
    for _ in range(iterations):
        x = -F.max_pool3d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    return (x > 0.999).float()


def compute_center_of_mass(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    mask: (B, 1, W, H, Z)
    return: (B, 3) coords in detector grid
    """
    if mask.dim() != 5 or mask.shape[1] != 1:
        raise RuntimeError(
            f"compute_center_of_mass expects mask shape (B,1,W,H,Z), got {tuple(mask.shape)}"
        )

    B, _, W, H, Z = mask.shape
    device = mask.device
    dtype = mask.dtype

    # 分母保持为 (B,)
    mass = mask.sum(dim=(1, 2, 3, 4)).clamp_min(eps)  # (B,)

    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype).view(1, 1, W, 1, 1)
    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype).view(1, 1, 1, H, 1)
    zs = torch.linspace(0, Z - 1, Z, device=device, dtype=dtype).view(1, 1, 1, 1, Z)

    # 分子也强制为 (B,)
    x_num = (mask * xs).sum(dim=(1, 2, 3, 4))  # (B,)
    y_num = (mask * ys).sum(dim=(1, 2, 3, 4))  # (B,)
    z_num = (mask * zs).sum(dim=(1, 2, 3, 4))  # (B,)

    x = x_num / mass
    y = y_num / mass
    z = z_num / mass

    return torch.stack([x, y, z], dim=-1)  # (B, 3)


# =========================
# Detector
# =========================
class SeedDetector(nn.Module):
    """
    输入:
        x: (B, Modal, W, H, Z)

    输出:
        seed_region_modal: (B, Modal, w, h, z)
        seed_region_fused: (B, 1, w, h, z)
        seed_coord_modal:  (B, Modal, 3)
        seed_coord_fused:  (B, 3)

    设计思想:
    - 共享编码器获取稳定解剖语义
    - 模态专属分支保留每模态“我认为的病灶核心”
    - Z-only Mamba 建模 slice 方向一致性
    - 跨模态一致性融合形成 fused seed prior
    """

    def __init__(
        self,
        in_modal: int,
        base_ch: int = 32,
        feat_dim: int = 96,
        detector_stride_xy: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_nslices: int = 8,
        softargmax_beta: float = 10.0,
        roi_size_fullres: Tuple[int, int, int] = (64, 64, 32),
        seed_region_radius_fullres: Tuple[int, int, int] = (16, 16, 8),
        in_channel: Optional[int] = None,
        out_channel: int = 1,
    ):
        super().__init__()
        assert detector_stride_xy in (
            2,
            4,
        ), "Current implementation expects xy stride 2 or 4."
        self.in_modal = int(in_modal)
        self.detector_stride_xy = int(detector_stride_xy)
        self.softargmax_beta = float(softargmax_beta)

        # ===== 新增字段 =====
        self.roi_size_fullres = tuple(int(v) for v in roi_size_fullres)
        self.seed_region_radius_fullres = tuple(
            int(v) for v in seed_region_radius_fullres
        )

        # 输入/输出通道超参数
        # in_channel 默认跟 in_modal 一致
        self.in_channel = int(in_modal if in_channel is None else in_channel)
        self.out_channel = int(out_channel)

        # ---------- shared encoder ----------
        self.stem = nn.Sequential(
            ConvNormAct3d(in_modal, base_ch, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1)),
            ResidualBlock3d(base_ch),
        )

        self.down1 = nn.Sequential(
            ConvNormAct3d(base_ch, base_ch * 2, k=(3, 3, 3), s=(2, 2, 1), p=(1, 1, 1)),
            ResidualBlock3d(base_ch * 2),
        )

        if detector_stride_xy == 4:
            self.down2 = nn.Sequential(
                ConvNormAct3d(
                    base_ch * 2, feat_dim, k=(3, 3, 3), s=(2, 2, 1), p=(1, 1, 1)
                ),
                ResidualBlock3d(feat_dim),
            )
        else:
            self.down2 = nn.Sequential(
                ConvNormAct3d(
                    base_ch * 2, feat_dim, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1)
                ),
                ResidualBlock3d(feat_dim),
            )

        # ---------- modal-specific refinement ----------
        self.modal_shared_proj = nn.Sequential(
            nn.Conv3d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feat_dim),
            nn.GELU(),
        )

        self.modal_refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(feat_dim),
                    nn.GELU(),
                    nn.Conv3d(feat_dim, 1, kernel_size=1, bias=True),
                )
                for _ in range(self.in_modal)
            ]
        )

        # ---------- Z-only Mamba contextualizer ----------
        self.z_mamba = ZOnlyMamba(
            dim=feat_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            nslices=mamba_nslices,
        )

        # ---------- fused seed prior ----------
        self.center_head = nn.Conv3d(feat_dim, 1, kernel_size=1, bias=True)
        self.core_head = nn.Conv3d(feat_dim, 1, kernel_size=1, bias=True)
        self.excl_head = nn.Conv3d(feat_dim, 1, kernel_size=1, bias=True)

        self.reliability_head = nn.Sequential(
            nn.Conv3d(in_modal + 1, in_modal, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.agreement_head = nn.Sequential(
            nn.Conv3d(in_modal, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def _reduce_modal_seed_centers(
        self,
        seed_coord_modal_full: torch.Tensor,
        mode: str = "mean",
    ) -> torch.Tensor:
        """
        将多个输入模态的 full-resolution seed 坐标融合为一个公共中心。

        Args:
            seed_coord_modal_full:
                (B, M, 3)
            mode:
                当前支持:
                - "mean": 各模态中心点求平均，作为统一 crop center

        Returns:
            reduced_center:
                (B, 3)
        """
        if seed_coord_modal_full.dim() != 3 or seed_coord_modal_full.shape[-1] != 3:
            raise RuntimeError(
                f"_reduce_modal_seed_centers expects shape (B,M,3), got {tuple(seed_coord_modal_full.shape)}"
            )

        if mode.lower() == "mean":
            return seed_coord_modal_full.mean(dim=1)

        raise ValueError(f"Unsupported reduction mode: {mode}")

    @torch.no_grad()
    def map_seed_coord_to_fullres(
        self,
        seed_coord_det: torch.Tensor,
        fullres_size: Tuple[int, int, int],
        det_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        将 detector grid 坐标映射回原图(full-resolution)坐标。

        Args:
            seed_coord_det:
                (B, 3), detector grid 坐标
            fullres_size:
                原图尺寸 (W, H, Z)
            det_size:
                detector grid 尺寸 (w, h, z)

        Returns:
            seed_coord_full:
                (B, 3), 原图坐标
        """
        if seed_coord_det.dim() != 2 or seed_coord_det.shape[1] != 3:
            raise RuntimeError(
                f"map_seed_coord_to_fullres expects seed_coord_det of shape (B,3), got {tuple(seed_coord_det.shape)}"
            )

        W, H, Z = [int(v) for v in fullres_size]
        w, h, z = [int(v) for v in det_size]

        if w <= 0 or h <= 0 or z <= 0:
            raise RuntimeError(f"Invalid det_size: {det_size}")
        if W <= 0 or H <= 0 or Z <= 0:
            raise RuntimeError(f"Invalid fullres_size: {fullres_size}")

        sx = float(W) / float(w)
        sy = float(H) / float(h)
        sz = float(Z) / float(z)

        seed_coord_full = seed_coord_det.clone().float()
        seed_coord_full[:, 0] = seed_coord_full[:, 0] * sx
        seed_coord_full[:, 1] = seed_coord_full[:, 1] * sy
        seed_coord_full[:, 2] = seed_coord_full[:, 2] * sz

        seed_coord_full[:, 0] = seed_coord_full[:, 0].clamp(0.0, float(W - 1))
        seed_coord_full[:, 1] = seed_coord_full[:, 1].clamp(0.0, float(H - 1))
        seed_coord_full[:, 2] = seed_coord_full[:, 2].clamp(0.0, float(Z - 1))

        return seed_coord_full

    @torch.no_grad()
    def build_fullres_seed_region(
        self,
        seed_coord_full: torch.Tensor,
        fullres_size: Tuple[int, int, int],
        radius: Optional[Tuple[int, int, int]] = None,
        region_type: str = "box",
    ) -> torch.Tensor:
        """
        在原图空间中，围绕 seed 坐标显式构建 seed region。

        Args:
            seed_coord_full:
                (B, 3), 原图坐标
            fullres_size:
                (W, H, Z)
            radius:
                半径 (rx, ry, rz)
                若为 None，则使用 self.seed_region_radius_fullres
            region_type:
                "box" 或 "ellipsoid"

        Returns:
            seed_region:
                (B, 1, W, H, Z), float tensor in {0,1}
        """
        if seed_coord_full.dim() != 2 or seed_coord_full.shape[1] != 3:
            raise RuntimeError(
                f"build_fullres_seed_region expects seed_coord_full of shape (B,3), got {tuple(seed_coord_full.shape)}"
            )

        W, H, Z = [int(v) for v in fullres_size]
        B = seed_coord_full.shape[0]
        device = seed_coord_full.device
        dtype = seed_coord_full.dtype

        if radius is None:
            radius = self.seed_region_radius_fullres
        rx, ry, rz = [max(1, int(v)) for v in radius]

        xs = torch.arange(W, device=device, dtype=dtype).view(1, W, 1, 1)
        ys = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
        zs = torch.arange(Z, device=device, dtype=dtype).view(1, 1, 1, Z)

        cx = seed_coord_full[:, 0].view(B, 1, 1, 1)
        cy = seed_coord_full[:, 1].view(B, 1, 1, 1)
        cz = seed_coord_full[:, 2].view(B, 1, 1, 1)

        if region_type.lower() == "box":
            region = (
                (torch.abs(xs - cx) <= float(rx))
                & (torch.abs(ys - cy) <= float(ry))
                & (torch.abs(zs - cz) <= float(rz))
            ).float()

        elif region_type.lower() == "ellipsoid":
            region = (
                ((xs - cx) / float(rx)) ** 2
                + ((ys - cy) / float(ry)) ** 2
                + ((zs - cz) / float(rz)) ** 2
                <= 1.0
            ).float()
        else:
            raise ValueError(
                f"Unsupported region_type: {region_type}. Use 'box' or 'ellipsoid'."
            )

        return region.unsqueeze(1)  # (B, 1, W, H, Z)

    @torch.no_grad()
    def crop_roi_around_seed(
        self,
        volume: torch.Tensor,
        center_coord_full: torch.Tensor,
        roi_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        对单个样本，在原图空间中围绕 seed 中心裁剪 ROI。

        Args:
            volume:
                (C, W, H, Z)
            center_coord_full:
                (3,), 原图坐标
            roi_size:
                (rw, rh, rz)
                若为 None，则使用 self.roi_size_fullres

        Returns:
            crop:
                (C, rw, rh, rz)
            crop_info:
                {
                    "x1","x2","y1","y2","z1","z2",
                    "cx","cy","cz","rw","rh","rz"
                }
        """
        if volume.dim() != 4:
            raise RuntimeError(
                f"crop_roi_around_seed expects volume shape (C,W,H,Z), got {tuple(volume.shape)}"
            )

        if center_coord_full.dim() != 1 or center_coord_full.shape[0] != 3:
            raise RuntimeError(
                f"crop_roi_around_seed expects center_coord_full shape (3,), got {tuple(center_coord_full.shape)}"
            )

        if roi_size is None:
            roi_size = self.roi_size_fullres

        C, W, H, Z = volume.shape
        rw, rh, rz = [int(v) for v in roi_size]

        rw = min(rw, W)
        rh = min(rh, H)
        rz = min(rz, Z)

        cx = int(round(float(center_coord_full[0].item())))
        cy = int(round(float(center_coord_full[1].item())))
        cz = int(round(float(center_coord_full[2].item())))

        cx = max(0, min(cx, W - 1))
        cy = max(0, min(cy, H - 1))
        cz = max(0, min(cz, Z - 1))

        x1 = cx - rw // 2
        y1 = cy - rh // 2
        z1 = cz - rz // 2

        x2 = x1 + rw
        y2 = y1 + rh
        z2 = z1 + rz

        if x1 < 0:
            x2 += -x1
            x1 = 0
        if y1 < 0:
            y2 += -y1
            y1 = 0
        if z1 < 0:
            z2 += -z1
            z1 = 0

        if x2 > W:
            shift = x2 - W
            x1 -= shift
            x2 = W
        if y2 > H:
            shift = y2 - H
            y1 -= shift
            y2 = H
        if z2 > Z:
            shift = z2 - Z
            z1 -= shift
            z2 = Z

        x1 = max(0, x1)
        y1 = max(0, y1)
        z1 = max(0, z1)

        crop = volume[:, x1:x2, y1:y2, z1:z2].contiguous()

        # 双保险：若由于极端边界导致尺寸仍不一致，则补齐
        pad_x = rw - crop.shape[1]
        pad_y = rh - crop.shape[2]
        pad_z = rz - crop.shape[3]

        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            crop = F.pad(
                crop,
                pad=(0, max(0, pad_z), 0, max(0, pad_y), 0, max(0, pad_x)),
                mode="constant",
                value=0.0,
            )

        crop_info = {
            "x1": int(x1),
            "x2": int(x2),
            "y1": int(y1),
            "y2": int(y2),
            "z1": int(z1),
            "z2": int(z2),
            "cx": int(cx),
            "cy": int(cy),
            "cz": int(cz),
            "rw": int(rw),
            "rh": int(rh),
            "rz": int(rz),
        }

        return crop, crop_info

    @torch.no_grad()
    def crop_roi_batch(
        self,
        image_full: torch.Tensor,
        seed_coord_full: torch.Tensor,
        seg_full: Optional[torch.Tensor] = None,
        roi_size: Optional[Tuple[int, int, int]] = None,
        crop_mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        对 batch 数据，在原图空间中围绕 seed 批量裁剪 ROI。

        裁剪规则：
        1) 若 in_channel == out_channel:
        - 认为 image 与 seg 模态一一对应
        - 每个模态用自己的 seed 单独裁 image / seg
        - 因而要求 seed_coord_full shape = (B, C, 3)
        2) 若 in_channel != out_channel:
        - 用多个输入模态 seed 的中心点(平均点)定义统一 ROI
        - image 与 seg 都按这个统一 ROI 裁
        - 允许 seed_coord_full shape = (B, C, 3) 或 (B, 3)

        Args:
            image_full:
                (B, Cin, W, H, Z)
            seed_coord_full:
                - (B, Cin, 3)
                - or (B, 3)
            seg_full:
                (B, Cout, W, H, Z), optional
            roi_size:
                optional
            crop_mode:
                "auto" | "paired" | "shared"

        Returns:
            {
                "image_roi": (B, Cin, rw, rh, rz),
                "seg_roi":   (B, Cout, rw, rh, rz) or None,
                "crop_infos": List[Dict[str, Any]],
                "used_center": (B, 3) or (B, Cin, 3),
                "effective_crop_mode": str
            }
        """
        if image_full.dim() != 5:
            raise RuntimeError(
                f"crop_roi_batch expects image_full shape (B,C,W,H,Z), got {tuple(image_full.shape)}"
            )

        if seg_full is not None:
            if seg_full.dim() != 5:
                raise RuntimeError(
                    f"crop_roi_batch expects seg_full shape (B,C,W,H,Z), got {tuple(seg_full.shape)}"
                )

        if roi_size is None:
            roi_size = self.roi_size_fullres

        B, Cin, W, H, Z = image_full.shape
        Cout = 0 if seg_full is None else int(seg_full.shape[1])

        if Cin != self.in_channel:
            raise RuntimeError(
                f"image_full channel mismatch: got {Cin}, expected self.in_channel={self.in_channel}"
            )

        if seg_full is not None and Cout != self.out_channel:
            raise RuntimeError(
                f"seg_full channel mismatch: got {Cout}, expected self.out_channel={self.out_channel}"
            )

        if crop_mode.lower() == "auto":
            effective_crop_mode = (
                "paired" if (self.in_channel == self.out_channel) else "shared"
            )
        else:
            effective_crop_mode = crop_mode.lower()

        if effective_crop_mode not in ("paired", "shared"):
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")

        if effective_crop_mode == "paired":
            if (
                seed_coord_full.dim() != 3
                or seed_coord_full.shape[0] != B
                or seed_coord_full.shape[1] != Cin
                or seed_coord_full.shape[2] != 3
            ):
                raise RuntimeError(
                    f"paired mode requires seed_coord_full shape (B,Cin,3), got {tuple(seed_coord_full.shape)}"
                )

            if seg_full is not None and Cout != Cin:
                raise RuntimeError(
                    f"paired mode requires Cout == Cin, got Cin={Cin}, Cout={Cout}"
                )

            image_crops_b = []
            seg_crops_b = [] if seg_full is not None else None
            crop_infos = []

            for b in range(B):
                per_img_crops = []
                per_seg_crops = [] if seg_full is not None else None
                per_infos = []

                for c in range(Cin):
                    img_crop_c, crop_info_c = self.crop_roi_around_seed(
                        volume=image_full[b, c : c + 1],
                        center_coord_full=seed_coord_full[b, c],
                        roi_size=roi_size,
                    )
                    per_img_crops.append(img_crop_c)
                    per_infos.append(
                        {
                            "channel": int(c),
                            "center_coord": [
                                float(v)
                                for v in seed_coord_full[b, c].detach().cpu().tolist()
                            ],
                            **crop_info_c,
                        }
                    )

                    if seg_full is not None:
                        seg_crop_c, _ = self.crop_roi_around_seed(
                            volume=seg_full[b, c : c + 1],
                            center_coord_full=seed_coord_full[b, c],
                            roi_size=roi_size,
                        )
                        per_seg_crops.append(seg_crop_c)

                image_crops_b.append(torch.cat(per_img_crops, dim=0))
                if seg_full is not None:
                    seg_crops_b.append(torch.cat(per_seg_crops, dim=0))
                crop_infos.append(per_infos)

            image_roi = torch.stack(image_crops_b, dim=0)
            seg_roi = None if seg_full is None else torch.stack(seg_crops_b, dim=0)

            return {
                "image_roi": image_roi,
                "seg_roi": seg_roi,
                "crop_infos": crop_infos,
                "used_center": seed_coord_full,
                "effective_crop_mode": effective_crop_mode,
            }

        else:
            if seed_coord_full.dim() == 3:
                if (
                    seed_coord_full.shape[0] != B
                    or seed_coord_full.shape[1] != Cin
                    or seed_coord_full.shape[2] != 3
                ):
                    raise RuntimeError(
                        f"shared mode with modal seeds requires shape (B,Cin,3), got {tuple(seed_coord_full.shape)}"
                    )
                shared_center = self._reduce_modal_seed_centers(
                    seed_coord_full, mode="mean"
                )

            elif seed_coord_full.dim() == 2:
                if seed_coord_full.shape[0] != B or seed_coord_full.shape[1] != 3:
                    raise RuntimeError(
                        f"shared mode with common seed requires shape (B,3), got {tuple(seed_coord_full.shape)}"
                    )
                shared_center = seed_coord_full

            else:
                raise RuntimeError(
                    f"shared mode expects seed_coord_full shape (B,Cin,3) or (B,3), got {tuple(seed_coord_full.shape)}"
                )

            image_crops = []
            seg_crops = []
            crop_infos = []

            for b in range(B):
                img_crop, crop_info = self.crop_roi_around_seed(
                    volume=image_full[b],
                    center_coord_full=shared_center[b],
                    roi_size=roi_size,
                )
                image_crops.append(img_crop)

                if seg_full is not None:
                    seg_crop, _ = self.crop_roi_around_seed(
                        volume=seg_full[b],
                        center_coord_full=shared_center[b],
                        roi_size=roi_size,
                    )
                    seg_crops.append(seg_crop)

                crop_infos.append(
                    {
                        "shared_center": [
                            float(v) for v in shared_center[b].detach().cpu().tolist()
                        ],
                        **crop_info,
                    }
                )

            image_roi = torch.stack(image_crops, dim=0)
            seg_roi = None if seg_full is None else torch.stack(seg_crops, dim=0)

            return {
                "image_roi": image_roi,
                "seg_roi": seg_roi,
                "crop_infos": crop_infos,
                "used_center": shared_center,
                "effective_crop_mode": effective_crop_mode,
            }

    # =========================
    # Forward
    # =========================
    def forward(self, x: torch.Tensor) -> SeedDetectorOutput:
        B, M, W, H, Z = x.shape
        assert M == self.in_modal, f"Expected in_modal={self.in_modal}, got {M}"

        # -------------------------
        # shared encoder
        # -------------------------
        f = self.stem(x)
        f = self.down1(f)
        f = self.down2(f)  # (B, feat_dim, w, h, z)

        # -------------------------
        # modal-specific evidence
        # -------------------------
        modal_feat = self.modal_shared_proj(f)
        modal_logits = []
        for m in range(self.in_modal):
            modal_logits.append(self.modal_refine[m](modal_feat))
        seed_region_modal_logits = torch.cat(modal_logits, dim=1)  # (B, Modal, w, h, z)

        # -------------------------
        # Z-only Mamba contextualization
        # -------------------------
        mamba_pack = self.z_mamba(f)

        ctr_logits = self.center_head(mamba_pack["rim"])
        core_logits = self.core_head(mamba_pack["core"])
        excl_logits = self.excl_head(mamba_pack["unc"])

        # fused prior = center + core - exclusion
        seed_region_fused_logits = ctr_logits + core_logits - excl_logits

        # -------------------------
        # cross-modal reliability
        # -------------------------
        fused_prob = torch.sigmoid(seed_region_fused_logits)
        modal_prob = torch.sigmoid(seed_region_modal_logits)

        reliab_in = torch.cat([modal_prob, fused_prob], dim=1)
        modal_reliability = self.reliability_head(reliab_in)  # (B, Modal, w, h, z)

        agreement_map = self.agreement_head(
            modal_prob * modal_reliability
        )  # (B,1,w,h,z)

        # final fused seed region
        seed_region_fused_logits = seed_region_fused_logits + torch.logit(
            agreement_map.clamp(1e-4, 1.0 - 1e-4)
        )

        # -------------------------
        # coordinates via soft-argmax
        # -------------------------
        seed_coord_modal = soft_argmax_3d(
            seed_region_modal_logits,
            beta=self.softargmax_beta,
        )  # (B, Modal, 3)

        seed_coord_fused = soft_argmax_3d(
            seed_region_fused_logits,
            beta=self.softargmax_beta,
        ).squeeze(
            1
        )  # (B, 3)

        return SeedDetectorOutput(
            seed_region_modal=seed_region_modal_logits,
            seed_region_fused=seed_region_fused_logits,
            seed_coord_modal=seed_coord_modal,
            seed_coord_fused=seed_coord_fused,
            agreement_map=agreement_map,
            modal_reliability=modal_reliability,
        )

    @torch.no_grad()
    def _find_best_roi_center_from_mask(
        self,
        mask: torch.Tensor,
        roi_size_fullres: Tuple[int, int, int] = (64, 64, 32),
    ) -> torch.Tensor:
        """
        在 detector grid 上，为每个样本寻找“最优 ROI 中心”：
        目标是在固定 ROI 尺寸下，使裁出的 ROI 对病灶覆盖最大。

        Args:
            mask:
                (B, 1, w, h, z), detector grid 上的二值病灶 mask
            roi_size_fullres:
                原图空间中的 ROI 尺寸，例如 (64, 64, 32)

        Returns:
            best_centers:
                (B, 3), detector grid 坐标系下的最优中心 [x, y, z]
        """
        if mask.dim() != 5 or mask.shape[1] != 1:
            raise RuntimeError(
                f"_find_best_roi_center_from_mask expects mask shape (B,1,w,h,z), got {tuple(mask.shape)}"
            )

        B, _, w, h, z = mask.shape
        device = mask.device
        dtype = mask.dtype

        # fullres -> detector grid 尺寸映射
        sx = float(self.detector_stride_xy)
        sy = float(self.detector_stride_xy)
        sz = 1.0

        roi_w = max(1, int(round(float(roi_size_fullres[0]) / sx)))
        roi_h = max(1, int(round(float(roi_size_fullres[1]) / sy)))
        roi_z = max(1, int(round(float(roi_size_fullres[2]) / sz)))

        # 防止 ROI 超出 detector grid
        roi_w = min(roi_w, w)
        roi_h = min(roi_h, h)
        roi_z = min(roi_z, z)

        # 用 avg_pool3d 计算“以每个位置为左上前角时，窗口内 lesion 体素数”
        lesion = mask.float()
        coverage = F.avg_pool3d(
            lesion,
            kernel_size=(roi_w, roi_h, roi_z),
            stride=1,
            padding=0,
        ) * float(
            roi_w * roi_h * roi_z
        )  # -> lesion voxel count inside ROI

        # coverage shape: (B,1,w-roi_w+1,h-roi_h+1,z-roi_z+1)
        _, _, cw, ch, cz = coverage.shape

        best_centers = []

        for b in range(B):
            lesion_sum = float(lesion[b, 0].sum().item())

            # 若空 mask，则退化为图像中心
            if lesion_sum <= 0.0:
                cx = float((w - 1) / 2.0)
                cy = float((h - 1) / 2.0)
                cz_ = float((z - 1) / 2.0)
                best_centers.append([cx, cy, cz_])
                continue

            cov_b = coverage[b, 0]  # (cw, ch, cz)

            # 找所有最大 coverage 候选
            max_cov = cov_b.max()
            candidate_mask = cov_b >= (max_cov - 1e-6)
            candidate_idx = torch.nonzero(
                candidate_mask, as_tuple=False
            )  # (N, 3), 左上前角坐标

            # lesion 几何中心（只用于在 coverage 并列时做 tie-break）
            coords = torch.nonzero(lesion[b, 0] > 0.5, as_tuple=False).float()  # (K, 3)
            geom_center = coords.mean(dim=0)  # [x, y, z]

            # 将候选左上前角转换为候选中心
            cand_centers = candidate_idx.float().to(device=device)
            cand_centers[:, 0] = cand_centers[:, 0] + (roi_w - 1) / 2.0
            cand_centers[:, 1] = cand_centers[:, 1] + (roi_h - 1) / 2.0
            cand_centers[:, 2] = cand_centers[:, 2] + (roi_z - 1) / 2.0

            # tie-break: 选离 lesion 几何中心最近的候选
            dist = ((cand_centers - geom_center.unsqueeze(0)) ** 2).sum(dim=1)
            best_idx = int(torch.argmin(dist).item())
            best_center = cand_centers[best_idx]

            # 最后再 clamp 一次，确保合法
            best_center[0] = best_center[0].clamp(0, w - 1)
            best_center[1] = best_center[1].clamp(0, h - 1)
            best_center[2] = best_center[2].clamp(0, z - 1)

            best_centers.append(best_center.tolist())

        return torch.tensor(best_centers, device=device, dtype=dtype)

    # =========================
    # Target generation
    # =========================
    @torch.no_grad()
    @torch.no_grad()
    def generate_seed_targets(
        self,
        seg_gt: torch.Tensor,
        erosion_iters: int = 1,
        roi_size_fullres: Tuple[int, int, int] = (64, 64, 32),
        center_sigma: float = 1.5,
    ) -> Dict[str, torch.Tensor]:
        """
        根据你当前的真实语义，统一支持两种情况：

        1) in_channel == out_channel:
        seg_gt 语义为按 modal 维度堆叠的模态对应标签
        -> 每个 modal 都有自己的 modal target / modal center target
        -> fused center 取各 modal center 的均值
        -> fused region 取各 modal core target 的并集(max)

        2) in_channel != out_channel:
        seg_gt 不再与输入 modal 一一对应
        -> 先将 seg_gt 在 channel 维做并集，得到 shared mask
        -> 所有 modal branch 共用 shared target
        -> fused center 由 shared mask 计算
        -> modal center target 复制 shared center

        Returns:
            {
                "seed_modal_target":         (B, in_modal, w, h, z)
                "seed_fused_target":         (B, 1, w, h, z)
                "center_coord_modal_target": (B, in_modal, 3)
                "center_coord_fused_target": (B, 3)
                "center_blob_target":        (B, 1, w, h, z)
            }
        """
        if seg_gt.dim() != 5:
            raise RuntimeError(
                f"generate_seed_targets expects seg_gt shape (B,C,W,H,Z), got {tuple(seg_gt.shape)}"
            )

        B, C, W, H, Z = seg_gt.shape

        if C != self.out_channel:
            raise RuntimeError(
                f"generate_seed_targets got seg_gt with C={C}, but self.out_channel={self.out_channel}"
            )

        # detector grid size
        if self.detector_stride_xy == 4:
            size = (W // 4, H // 4, Z)
        else:
            size = (W // 2, H // 2, Z)

        seg_ds = F.interpolate(seg_gt.float(), size=size, mode="nearest")

        # --------------------------------------------------
        # Case A: modal-aligned supervision
        # in_channel == out_channel 且 label 按 modal 堆叠
        # --------------------------------------------------
        if self.in_channel == self.out_channel:
            modal_targets = []
            modal_centers = []

            for c in range(self.in_channel):
                seg_c = seg_ds[:, c : c + 1]  # (B,1,w,h,z)

                if erosion_iters > 0:
                    core_c = binary_erode_3d(
                        seg_c, kernel_size=3, iterations=erosion_iters
                    )
                    empty = core_c.sum(dim=(2, 3, 4), keepdim=True) == 0
                    core_c = torch.where(empty, seg_c, core_c)
                else:
                    core_c = seg_c

                center_c = self._find_best_roi_center_from_mask(
                    mask=seg_c,
                    roi_size_fullres=roi_size_fullres,
                )  # (B,3)

                modal_targets.append(core_c)
                modal_centers.append(center_c.unsqueeze(1))

            seed_modal_target = torch.cat(modal_targets, dim=1)  # (B,in_modal,w,h,z)
            center_coord_modal_target = torch.cat(
                modal_centers, dim=1
            )  # (B,in_modal,3)

            # fused target: 多个模态 core 的并集
            seed_fused_target = seed_modal_target.amax(
                dim=1, keepdim=True
            )  # (B,1,w,h,z)

            # fused center: 多个 modal center 的中心点
            center_coord_fused_target = center_coord_modal_target.mean(dim=1)  # (B,3)

            center_blob_target = self._render_gaussian_targets(
                center_coord_fused_target,
                size=size,
                sigma=center_sigma,
            )

            return {
                "seed_modal_target": seed_modal_target,
                "seed_fused_target": seed_fused_target,
                "center_coord_modal_target": center_coord_modal_target,
                "center_coord_fused_target": center_coord_fused_target,
                "center_blob_target": center_blob_target,
            }

        # --------------------------------------------------
        # Case B: shared supervision
        # in_channel != out_channel
        # --------------------------------------------------
        else:
            # 输出通道不与输入 modal 一一对应时，
            # 用所有输出通道的并集定义 shared lesion mask
            seg_shared = seg_ds.amax(dim=1, keepdim=True)  # (B,1,w,h,z)

            if erosion_iters > 0:
                core_shared = binary_erode_3d(
                    seg_shared, kernel_size=3, iterations=erosion_iters
                )
                empty = core_shared.sum(dim=(2, 3, 4), keepdim=True) == 0
                core_shared = torch.where(empty, seg_shared, core_shared)
            else:
                core_shared = seg_shared

            center_shared = self._find_best_roi_center_from_mask(
                mask=seg_shared,
                roi_size_fullres=roi_size_fullres,
            )  # (B,3)

            center_blob_target = self._render_gaussian_targets(
                center_shared,
                size=size,
                sigma=center_sigma,
            )

            seed_modal_target = core_shared.repeat(
                1, self.in_channel, 1, 1, 1
            )  # (B,in_modal,w,h,z)
            center_coord_modal_target = center_shared.unsqueeze(1).repeat(
                1, self.in_channel, 1
            )  # (B,in_modal,3)

            return {
                "seed_modal_target": seed_modal_target,
                "seed_fused_target": core_shared,
                "center_coord_modal_target": center_coord_modal_target,
                "center_coord_fused_target": center_shared,
                "center_blob_target": center_blob_target,
            }

    def _render_gaussian_targets(
        self,
        coords: torch.Tensor,
        size: Tuple[int, int, int],
        sigma: float = 1.5,
    ) -> torch.Tensor:
        """
        coords:
            - (B, 3)
            - or (B, 1, 3)
        return:
            (B, 1, W, H, Z)
        """
        if coords.dim() == 3 and coords.shape[1] == 1 and coords.shape[2] == 3:
            coords = coords.squeeze(1)
        elif coords.dim() != 2 or coords.shape[1] != 3:
            raise RuntimeError(
                f"_render_gaussian_targets expects coords of shape (B,3) or (B,1,3), got {tuple(coords.shape)}"
            )

        B = coords.shape[0]
        W, H, Z = size
        device = coords.device
        dtype = coords.dtype

        xs = torch.arange(W, device=device, dtype=dtype).view(1, W, 1, 1)
        ys = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
        zs = torch.arange(Z, device=device, dtype=dtype).view(1, 1, 1, Z)

        cx = coords[:, 0].reshape(B, 1, 1, 1)
        cy = coords[:, 1].reshape(B, 1, 1, 1)
        cz = coords[:, 2].reshape(B, 1, 1, 1)

        g = torch.exp(
            -((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) / (2.0 * sigma * sigma)
        )

        return g.unsqueeze(1)

    # =========================
    # Loss
    # =========================
    def compute_loss(
        self,
        out: SeedDetectorOutput,
        seg_gt: torch.Tensor,
        modal_consistency_weight: float = 0.10,
        coord_weight: float = 0.25,
        sharpness_weight: float = 0.05,
        center_blob_weight: float = 0.10,
    ) -> Dict[str, torch.Tensor]:
        """
        detector warmup loss

        现在支持：
        - in_channel == out_channel: modal-wise supervision
        - in_channel != out_channel: shared supervision
        """
        tgt = self.generate_seed_targets(seg_gt)

        seed_modal_target = tgt["seed_modal_target"]  # (B,in_modal,w,h,z)
        seed_fused_target = tgt["seed_fused_target"]  # (B,1,w,h,z)
        center_coord_modal_target = tgt["center_coord_modal_target"]  # (B,in_modal,3)
        center_coord_fused_target = tgt["center_coord_fused_target"]  # (B,3)
        center_blob_target = tgt["center_blob_target"]  # (B,1,w,h,z)

        # -------------------------
        # region losses
        # -------------------------
        loss_seed_modal = F.binary_cross_entropy_with_logits(
            out.seed_region_modal, seed_modal_target
        ) + self._soft_dice_from_logits(out.seed_region_modal, seed_modal_target)

        loss_seed_fused = F.binary_cross_entropy_with_logits(
            out.seed_region_fused, seed_fused_target
        ) + self._soft_dice_from_logits(out.seed_region_fused, seed_fused_target)

        # -------------------------
        # coordinate losses
        # -------------------------
        loss_coord_modal = 0.0 * out.seed_coord_modal.sum()
        for m in range(self.in_modal):
            loss_coord_modal = loss_coord_modal + F.smooth_l1_loss(
                out.seed_coord_modal[:, m, :],
                center_coord_modal_target[:, m, :],
            )
        loss_coord_modal = loss_coord_modal / float(self.in_modal)

        loss_coord_fused = F.smooth_l1_loss(
            out.seed_coord_fused,
            center_coord_fused_target,
        )

        # -------------------------
        # center blob alignment
        # -------------------------
        loss_center_blob = F.binary_cross_entropy_with_logits(
            out.seed_region_fused,
            center_blob_target,
        )

        # -------------------------
        # modal consistency
        # fused prob 对齐到各 modal branch
        # -------------------------
        fused_prob = torch.sigmoid(out.seed_region_fused).repeat(
            1, self.in_modal, 1, 1, 1
        )
        modal_prob = torch.sigmoid(out.seed_region_modal)
        loss_consistency = F.l1_loss(modal_prob, fused_prob)

        # -------------------------
        # sharpness / compactness
        # -------------------------
        loss_sharp = modal_prob.mean()

        loss_total = (
            loss_seed_modal
            + loss_seed_fused
            + coord_weight * (loss_coord_modal + loss_coord_fused)
            + modal_consistency_weight * loss_consistency
            + sharpness_weight * loss_sharp
            + center_blob_weight * loss_center_blob
        )

        return {
            "loss_total": loss_total,
            "loss_seed_modal": loss_seed_modal,
            "loss_seed_fused": loss_seed_fused,
            "loss_coord_modal": loss_coord_modal,
            "loss_coord_fused": loss_coord_fused,
            "loss_consistency": loss_consistency,
            "loss_sharp": loss_sharp,
            "loss_center_blob": loss_center_blob,
        }

    def _soft_dice_from_logits(
        self, logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum()
        union = prob.sum() + target.sum()
        return 1.0 - (2.0 * inter + eps) / (union + eps)

    # =========================
    # Metrics
    # =========================
    @torch.no_grad()
    def compute_metrics(
        self,
        out: SeedDetectorOutput,
        seg_gt: torch.Tensor,
        roi_size: Tuple[int, int, int] = (64, 64, 32),
    ) -> Dict[str, float]:
        """
        评价 detector 是否真正可用于后续 ROI segmentation

        统一按 fused/shared 目标做主评估：
        - 若 in_channel == out_channel:
        fused target = 各 modal target 的并集
        fused center = 各 modal center 的均值
        - 若 in_channel != out_channel:
        fused target = shared target
        fused center = shared center
        """
        tgt = self.generate_seed_targets(seg_gt)

        center_coord_fused_target = tgt["center_coord_fused_target"]  # (B,3)
        seed_fused_target = tgt["seed_fused_target"]  # (B,1,w,h,z)

        B, _, W_full, H_full, Z_full = seg_gt.shape
        w_det = out.seed_region_fused.shape[2]
        h_det = out.seed_region_fused.shape[3]
        z_det = out.seed_region_fused.shape[4]

        # -------------------------
        # center error in detector grid
        # -------------------------
        center_l1 = F.l1_loss(
            out.seed_coord_fused,
            center_coord_fused_target,
            reduction="mean",
        ).item()

        # -------------------------
        # seed in lesion/core
        # -------------------------
        seed_fused_round = out.seed_coord_fused.round().long()
        seed_in_lesion = 0.0

        for b in range(B):
            x = int(seed_fused_round[b, 0].clamp(0, w_det - 1).item())
            y = int(seed_fused_round[b, 1].clamp(0, h_det - 1).item())
            z = int(seed_fused_round[b, 2].clamp(0, z_det - 1).item())

            if float(seed_fused_target[b, 0, x, y, z].item()) > 0.5:
                seed_in_lesion += 1.0

        seed_in_lesion_rate = seed_in_lesion / float(B)

        # -------------------------
        # ROI coverage in original volume
        # 使用 full-resolution shared mask 做 coverage 评估
        # -------------------------
        seg_shared_full = seg_gt.amax(dim=1, keepdim=True)  # (B,1,W,H,Z)

        sx = float(W_full) / float(w_det)
        sy = float(H_full) / float(h_det)
        sz = float(Z_full) / float(z_det)

        roi_cov = 0.0
        for b in range(B):
            cx = int(round(float(out.seed_coord_fused[b, 0].item() * sx)))
            cy = int(round(float(out.seed_coord_fused[b, 1].item() * sy)))
            cz = int(round(float(out.seed_coord_fused[b, 2].item() * sz)))

            rw, rh, rz = roi_size
            x1 = max(0, cx - rw // 2)
            y1 = max(0, cy - rh // 2)
            z1 = max(0, cz - rz // 2)
            x2 = min(W_full, x1 + rw)
            y2 = min(H_full, y1 + rh)
            z2 = min(Z_full, z1 + rz)

            gt = seg_shared_full[b, 0]
            gt_sum = float(gt.sum().item())

            if gt_sum <= 0:
                roi_cov += 1.0
            else:
                covered = float(gt[x1:x2, y1:y2, z1:z2].sum().item())
                roi_cov += covered / gt_sum

        roi_coverage = roi_cov / float(B)

        # -------------------------
        # peakness
        # -------------------------
        prob = torch.sigmoid(out.seed_region_fused)
        flat = prob.flatten(2)
        top1 = flat.max(dim=-1).values.mean().item()
        meanv = flat.mean(dim=-1).mean().item()
        peakness = top1 / max(meanv, 1e-6)

        return {
            "center_l1": float(center_l1),
            "seed_in_lesion_rate": float(seed_in_lesion_rate),
            "roi_coverage": float(roi_coverage),
            "peakness": float(peakness),
        }


# =========================
# Model
# =========================
# =========================
# Seed-conditioned Multimodal Backbone Blocks
# =========================
class SeedConditionedModalStem(nn.Module):
    """
    单模态浅层编码器
    输入:  (B, 1, W, H, Z)
    输出:  (B, C, W, H, Z)
    """

    def __init__(self, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3d(1, out_ch, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1)),
            ResidualBlock3d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SeedConditionedModulation(nn.Module):
    """
    用 seed prior 对模态特征进行空间调制
    输入:
        feat:       (B, C, W, H, Z)
        seed_prior: (B, 1, W, H, Z)
    输出:
        feat_cond:  (B, C, W, H, Z)
    """

    def __init__(self, feat_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(feat_ch + 1, feat_ch, kernel_size=1, bias=True),
            nn.InstanceNorm3d(feat_ch),
            nn.GELU(),
            nn.Conv3d(feat_ch, feat_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.bias = nn.Sequential(
            nn.Conv3d(feat_ch + 1, feat_ch, kernel_size=1, bias=True),
            nn.InstanceNorm3d(feat_ch),
            nn.GELU(),
            nn.Conv3d(feat_ch, feat_ch, kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor, seed_prior: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, seed_prior], dim=1)
        gamma = self.gate(x)
        beta = self.bias(x)
        return feat * (1.0 + gamma) + beta


class SeedAwareFusionBlock(nn.Module):
    """
    seed-aware multimodal fusion
    输入:
        modal_feats: list[(B, C, W, H, Z), ...]
        seed_prior:  (B, 1, W, H, Z)
    输出:
        {
            "fused": (B, C, W, H, Z),
            "modal_weights": (B, M, 1, 1, 1)
        }
    """

    def __init__(self, modal_num: int, feat_ch: int):
        super().__init__()
        self.modal_num = int(modal_num)
        self.feat_ch = int(feat_ch)

        self.weight_head = nn.Sequential(
            nn.Conv3d(modal_num * feat_ch + 1, feat_ch, kernel_size=1, bias=True),
            nn.InstanceNorm3d(feat_ch),
            nn.GELU(),
            nn.Conv3d(feat_ch, modal_num, kernel_size=1, bias=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(modal_num * feat_ch, feat_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feat_ch),
            nn.GELU(),
            ResidualBlock3d(feat_ch),
        )

    def forward(
        self,
        modal_feats: list[torch.Tensor],
        seed_prior: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if len(modal_feats) != self.modal_num:
            raise RuntimeError(
                f"SeedAwareFusionBlock expects {self.modal_num} modal feats, got {len(modal_feats)}"
            )

        x_cat = torch.cat(modal_feats, dim=1)  # (B, M*C, W, H, Z)
        w_logits = self.weight_head(
            torch.cat([x_cat, seed_prior], dim=1)
        )  # (B, M, W, H, Z)
        w = torch.softmax(w_logits, dim=1)

        weighted_feats = []
        for m in range(self.modal_num):
            weighted_feats.append(modal_feats[m] * w[:, m : m + 1])

        fused_weighted = torch.cat(weighted_feats, dim=1)
        fused = self.fuse(fused_weighted)
        modal_weights = w.mean(dim=(2, 3, 4), keepdim=True)  # (B, M, 1, 1, 1)

        return {
            "fused": fused,
            "modal_weights": modal_weights,
        }


# =========================
# Seed-conditioned Multimodal Backbone
# =========================
class SeedConditionedMultiModalBackbone(nn.Module):
    """
    ROI 主干本体

    输入:
        image_roi:      (B, Cin, Wr, Hr, Zr)
        seed_prior_roi: (B, 1, Wr, Hr, Zr)

    输出:
        {
            "seg_logits":    (B, Cout, Wr, Hr, Zr),
            "cls_logits":    (B, 1),
            "modal_weights": (B, Cin, 1, 1, 1),
            "seg_prob":      (B, 1, Wr, Hr, Zr)
        }
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        base_ch: int = 32,
        use_bottleneck_context: bool = False,
    ):
        super().__init__()
        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        self.base_ch = int(base_ch)
        self.use_bottleneck_context = bool(use_bottleneck_context)

        self.modal_stems = nn.ModuleList(
            [SeedConditionedModalStem(out_ch=base_ch) for _ in range(self.in_channel)]
        )

        self.modal_modulators = nn.ModuleList(
            [SeedConditionedModulation(feat_ch=base_ch) for _ in range(self.in_channel)]
        )

        self.fusion = SeedAwareFusionBlock(
            modal_num=self.in_channel,
            feat_ch=base_ch,
        )

        self.enc1 = nn.Sequential(
            ResidualBlock3d(base_ch),
        )

        self.down1 = nn.Sequential(
            ConvNormAct3d(base_ch, base_ch * 2, s=(2, 2, 2)),
            ResidualBlock3d(base_ch * 2),
        )

        self.down2 = nn.Sequential(
            ConvNormAct3d(base_ch * 2, base_ch * 4, s=(2, 2, 2)),
            ResidualBlock3d(base_ch * 4),
        )

        self.down3 = nn.Sequential(
            ConvNormAct3d(base_ch * 4, base_ch * 8, s=(2, 2, 2)),
            ResidualBlock3d(base_ch * 8),
        )

        self.bottleneck_context = (
            ResidualBlock3d(base_ch * 8)
            if self.use_bottleneck_context
            else nn.Identity()
        )

        self.up2 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvNormAct3d(base_ch * 4, base_ch * 4),
            ResidualBlock3d(base_ch * 4),
        )

        self.up1 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvNormAct3d(base_ch * 2, base_ch * 2),
            ResidualBlock3d(base_ch * 2),
        )

        self.up0 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(
            ConvNormAct3d(base_ch, base_ch),
            ResidualBlock3d(base_ch),
        )

        self.seg_head = nn.Conv3d(base_ch, out_channel, kernel_size=1, bias=True)

        cls_in_dim = base_ch * 2 + base_ch * 8
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in_dim, base_ch * 4),
            nn.GELU(),
            nn.Linear(base_ch * 4, base_ch * 2),
            nn.GELU(),
            nn.Linear(base_ch * 2, 1),
        )

    def forward(
        self,
        image_roi: torch.Tensor,
        seed_prior_roi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if image_roi.dim() != 5:
            raise RuntimeError(
                f"SeedConditionedMultiModalBackbone expects image_roi shape (B,C,W,H,Z), got {tuple(image_roi.shape)}"
            )
        if seed_prior_roi.dim() != 5 or seed_prior_roi.shape[1] != 1:
            raise RuntimeError(
                f"SeedConditionedMultiModalBackbone expects seed_prior_roi shape (B,1,W,H,Z), got {tuple(seed_prior_roi.shape)}"
            )

        B, Cin, Wr, Hr, Zr = image_roi.shape
        if Cin != self.in_channel:
            raise RuntimeError(
                f"SeedConditionedMultiModalBackbone got Cin={Cin}, expected {self.in_channel}"
            )

        modal_feats = []
        for m in range(self.in_channel):
            x_m = image_roi[:, m : m + 1]
            f_m = self.modal_stems[m](x_m)
            f_m = self.modal_modulators[m](f_m, seed_prior_roi)
            modal_feats.append(f_m)

        fuse_pack = self.fusion(modal_feats, seed_prior_roi)
        f0 = fuse_pack["fused"]
        modal_weights = fuse_pack["modal_weights"]

        e1 = self.enc1(f0)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        b = self.down3(e3)
        b = self.bottleneck_context(b)

        d2 = self.up2(b)
        if d2.shape[2:] != e3.shape[2:]:
            d2 = F.interpolate(
                d2, size=e3.shape[2:], mode="trilinear", align_corners=False
            )
        d2 = self.dec2(d2 + e3)

        d1 = self.up1(d2)
        if d1.shape[2:] != e2.shape[2:]:
            d1 = F.interpolate(
                d1, size=e2.shape[2:], mode="trilinear", align_corners=False
            )
        d1 = self.dec1(d1 + e2)

        d0 = self.up0(d1)
        if d0.shape[2:] != e1.shape[2:]:
            d0 = F.interpolate(
                d0, size=e1.shape[2:], mode="trilinear", align_corners=False
            )
        d0 = self.dec0(d0 + e1)

        seg_logits = self.seg_head(d0)
        seg_prob = torch.sigmoid(seg_logits).mean(dim=1, keepdim=True)

        eps = 1e-6
        feat_core = (d0 * seg_prob).sum(dim=(2, 3, 4)) / (
            seg_prob.sum(dim=(2, 3, 4)) + eps
        )

        peri = 1.0 - seg_prob
        feat_peri = (d0 * peri).sum(dim=(2, 3, 4)) / (peri.sum(dim=(2, 3, 4)) + eps)

        feat_bottleneck = b.mean(dim=(2, 3, 4))
        feat_cls = torch.cat([feat_core, feat_peri, feat_bottleneck], dim=1)
        cls_logits = self.cls_head(feat_cls)

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "modal_weights": modal_weights,
            "seg_prob": seg_prob,
        }


# =========================
# Final End-to-End Model
# =========================
class SeedGuidedMultiTaskModel(nn.Module):
    """
    最终单入口模型：
    外部只输入 MRI

    forward(image_full, seg_gt=None, cls_gt=None) 内部自动完成：
    1) detector
    2) ROI crop
    3) seed prior construction
    4) ROI backbone
    5) restore seg to fullres

    输出:
        {
            "seg_logits_full": ...,
            "seg_logits_roi": ...,
            "cls_logits": ...,
            "detector_out": ...,
            "crop_infos": ...,
            "seed_coord_full": ...,
            "seed_prior_roi": ...,
            "modal_weights": ...
        }
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        detector_base_ch: int = 32,
        detector_feat_dim: int = 96,
        detector_stride_xy: int = 4,
        detector_mamba_d_state: int = 16,
        detector_mamba_d_conv: int = 4,
        detector_mamba_expand: int = 2,
        detector_mamba_nslices: int = 8,
        detector_softargmax_beta: float = 10.0,
        roi_size_fullres: Tuple[int, int, int] = (64, 64, 32),
        seed_region_radius_fullres: Tuple[int, int, int] = (16, 16, 8),
        backbone_base_ch: int = 32,
        seed_prior_sigma_scale: float = 0.20,
        seg_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
        detector_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        self.roi_size_fullres = tuple(int(v) for v in roi_size_fullres)
        self.seed_prior_sigma_scale = float(seed_prior_sigma_scale)

        self.seg_loss_weight = float(seg_loss_weight)
        self.cls_loss_weight = float(cls_loss_weight)
        self.detector_loss_weight = float(detector_loss_weight)

        # detector 内嵌
        self.detector = SeedDetector(
            in_modal=self.in_channel,
            in_channel=self.in_channel,
            out_channel=self.out_channel,
            base_ch=detector_base_ch,
            feat_dim=detector_feat_dim,
            detector_stride_xy=detector_stride_xy,
            mamba_d_state=detector_mamba_d_state,
            mamba_d_conv=detector_mamba_d_conv,
            mamba_expand=detector_mamba_expand,
            mamba_nslices=detector_mamba_nslices,
            softargmax_beta=detector_softargmax_beta,
            roi_size_fullres=roi_size_fullres,
            seed_region_radius_fullres=seed_region_radius_fullres,
        )

        self.backbone = SeedConditionedMultiModalBackbone(
            in_channel=self.in_channel,
            out_channel=self.out_channel,
            base_ch=backbone_base_ch,
            use_bottleneck_context=False,
        )

    def _build_seed_prior_roi(
        self,
        roi_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        Wr, Hr, Zr = [int(v) for v in roi_shape]

        xs = torch.arange(Wr, device=device, dtype=dtype).view(1, Wr, 1, 1)
        ys = torch.arange(Hr, device=device, dtype=dtype).view(1, 1, Hr, 1)
        zs = torch.arange(Zr, device=device, dtype=dtype).view(1, 1, 1, Zr)

        cx = torch.tensor((Wr - 1) / 2.0, device=device, dtype=dtype).view(1, 1, 1, 1)
        cy = torch.tensor((Hr - 1) / 2.0, device=device, dtype=dtype).view(1, 1, 1, 1)
        cz = torch.tensor((Zr - 1) / 2.0, device=device, dtype=dtype).view(1, 1, 1, 1)

        sigma_x = max(1.0, Wr * self.seed_prior_sigma_scale)
        sigma_y = max(1.0, Hr * self.seed_prior_sigma_scale)
        sigma_z = max(1.0, Zr * self.seed_prior_sigma_scale)

        g = torch.exp(
            -(
                ((xs - cx) ** 2) / (2.0 * sigma_x * sigma_x)
                + ((ys - cy) ** 2) / (2.0 * sigma_y * sigma_y)
                + ((zs - cz) ** 2) / (2.0 * sigma_z * sigma_z)
            )
        ).unsqueeze(
            1
        )  # (1,1,Wr,Hr,Zr)

        return g.repeat(batch_size, 1, 1, 1, 1)

    def _restore_seg_to_fullres(
        self,
        seg_logits_roi: torch.Tensor,
        crop_infos: Any,
        fullres_shape: Tuple[int, int, int],
        paired_mode: bool,
        outside_logit: float = -20.0,
    ) -> torch.Tensor:
        """
        将 ROI segmentation logits 映射回 full-resolution 空间。

        关键修正：
        - ROI 外区域不能初始化为 0
        - 因为 sigmoid(0)=0.5，后续 threshold>=0.5 会被误判为前景
        - 因此这里统一初始化为一个很小的负值 outside_logit，使 ROI 外显式表示为背景
        """
        if seg_logits_roi.dim() != 5:
            raise RuntimeError(
                f"_restore_seg_to_fullres expects seg_logits_roi shape (B,C,Wr,Hr,Zr), got {tuple(seg_logits_roi.shape)}"
            )

        B, Cout, Wr, Hr, Zr = seg_logits_roi.shape
        W, H, Z = [int(v) for v in fullres_shape]

        seg_logits_full = torch.full(
            (B, Cout, W, H, Z),
            fill_value=float(outside_logit),
            device=seg_logits_roi.device,
            dtype=seg_logits_roi.dtype,
        )

        if paired_mode:
            for b in range(B):
                for c in range(Cout):
                    info = crop_infos[b][c]
                    x1, x2 = info["x1"], info["x2"]
                    y1, y2 = info["y1"], info["y2"]
                    z1, z2 = info["z1"], info["z2"]

                    seg_logits_full[b, c : c + 1, x1:x2, y1:y2, z1:z2] = seg_logits_roi[
                        b, c : c + 1, : x2 - x1, : y2 - y1, : z2 - z1
                    ]
        else:
            for b in range(B):
                info = crop_infos[b]
                x1, x2 = info["x1"], info["x2"]
                y1, y2 = info["y1"], info["y2"]
                z1, z2 = info["z1"], info["z2"]

                seg_logits_full[b, :, x1:x2, y1:y2, z1:z2] = seg_logits_roi[
                    b, :, : x2 - x1, : y2 - y1, : z2 - z1
                ]

        return seg_logits_full

    def _build_seg_roi_target(
        self,
        seg_gt: torch.Tensor,
        crop_infos: Any,
        paired_mode: bool,
    ) -> torch.Tensor:
        """
        从 fullres seg_gt 构造 ROI supervision target
        """
        if seg_gt.dim() != 5:
            raise RuntimeError(
                f"_build_seg_roi_target expects seg_gt shape (B,C,W,H,Z), got {tuple(seg_gt.shape)}"
            )

        B, Cout, W, H, Z = seg_gt.shape

        if paired_mode:
            roi_targets = []
            for b in range(B):
                per_ch = []
                for c in range(Cout):
                    info = crop_infos[b][c]
                    x1, x2 = info["x1"], info["x2"]
                    y1, y2 = info["y1"], info["y2"]
                    z1, z2 = info["z1"], info["z2"]
                    per_ch.append(seg_gt[b : b + 1, c : c + 1, x1:x2, y1:y2, z1:z2])
                roi_targets.append(torch.cat(per_ch, dim=1))
            return torch.cat(roi_targets, dim=0)

        roi_targets = []
        for b in range(B):
            info = crop_infos[b]
            x1, x2 = info["x1"], info["x2"]
            y1, y2 = info["y1"], info["y2"]
            z1, z2 = info["z1"], info["z2"]
            roi_targets.append(seg_gt[b : b + 1, :, x1:x2, y1:y2, z1:z2])
        return torch.cat(roi_targets, dim=0)

    def _seg_loss(
        self,
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        eps: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        seg_target = seg_target.float()

        loss_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target)

        prob = torch.sigmoid(seg_logits)
        inter = (prob * seg_target).sum()
        union = prob.sum() + seg_target.sum()
        loss_dice = 1.0 - (2.0 * inter + eps) / (union + eps)

        loss_total = loss_bce + loss_dice
        return {
            "loss_seg_total": loss_total,
            "loss_seg_bce": loss_bce,
            "loss_seg_dice": loss_dice,
        }

    def _cls_loss(
        self,
        cls_logits: torch.Tensor,
        cls_gt: torch.Tensor,
    ) -> torch.Tensor:
        if cls_gt.dim() == 1:
            cls_gt = cls_gt.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(cls_logits, cls_gt.float())

    def forward(
        self,
        image_full: torch.Tensor,
        seg_gt: Optional[torch.Tensor] = None,
        cls_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if image_full.dim() != 5:
            raise RuntimeError(
                f"SeedGuidedMultiTaskModel expects image_full shape (B,C,W,H,Z), got {tuple(image_full.shape)}"
            )

        B, Cin, W, H, Z = image_full.shape
        if Cin != self.in_channel:
            raise RuntimeError(
                f"SeedGuidedMultiTaskModel got Cin={Cin}, expected {self.in_channel}"
            )

        detector_out = self.detector(image_full)

        det_size = (
            int(detector_out.seed_region_fused.shape[2]),
            int(detector_out.seed_region_fused.shape[3]),
            int(detector_out.seed_region_fused.shape[4]),
        )
        fullres_size = (W, H, Z)

        paired_mode = self.in_channel == self.out_channel

        if paired_mode:
            seed_coord_modal_full = []
            for m in range(self.in_channel):
                coord_m_full = self.detector.map_seed_coord_to_fullres(
                    seed_coord_det=detector_out.seed_coord_modal[:, m, :],
                    fullres_size=fullres_size,
                    det_size=det_size,
                )
                seed_coord_modal_full.append(coord_m_full.unsqueeze(1))
            seed_coord_full = torch.cat(seed_coord_modal_full, dim=1)  # (B,Cin,3)

            roi_pack = self.detector.crop_roi_batch(
                image_full=image_full,
                seed_coord_full=seed_coord_full,
                seg_full=seg_gt if seg_gt is not None else None,
                roi_size=self.roi_size_fullres,
                crop_mode="paired",
            )

        else:
            seed_coord_modal_full = []
            for m in range(self.in_channel):
                coord_m_full = self.detector.map_seed_coord_to_fullres(
                    seed_coord_det=detector_out.seed_coord_modal[:, m, :],
                    fullres_size=fullres_size,
                    det_size=det_size,
                )
                seed_coord_modal_full.append(coord_m_full.unsqueeze(1))
            seed_coord_modal_full = torch.cat(seed_coord_modal_full, dim=1)

            seed_coord_shared_full = self.detector._reduce_modal_seed_centers(
                seed_coord_modal_full,
                mode="mean",
            )

            seed_coord_full = seed_coord_shared_full

            roi_pack = self.detector.crop_roi_batch(
                image_full=image_full,
                seed_coord_full=seed_coord_shared_full,
                seg_full=seg_gt if seg_gt is not None else None,
                roi_size=self.roi_size_fullres,
                crop_mode="shared",
            )

        image_roi = roi_pack["image_roi"]
        crop_infos = roi_pack["crop_infos"]

        roi_shape = (
            int(image_roi.shape[2]),
            int(image_roi.shape[3]),
            int(image_roi.shape[4]),
        )
        seed_prior_roi = self._build_seed_prior_roi(
            roi_shape=roi_shape,
            device=image_roi.device,
            dtype=image_roi.dtype,
            batch_size=image_roi.shape[0],
        )

        backbone_out = self.backbone(
            image_roi=image_roi,
            seed_prior_roi=seed_prior_roi,
        )

        seg_logits_roi = backbone_out["seg_logits"]
        cls_logits = backbone_out["cls_logits"]
        modal_weights = backbone_out["modal_weights"]

        seg_logits_full = self._restore_seg_to_fullres(
            seg_logits_roi=seg_logits_roi,
            crop_infos=crop_infos,
            fullres_shape=fullres_size,
            paired_mode=paired_mode,
        )

        out = {
            "seg_logits_full": seg_logits_full,
            "seg_logits_roi": seg_logits_roi,
            "cls_logits": cls_logits,
            "detector_out": detector_out,
            "crop_infos": crop_infos,
            "seed_coord_full": seed_coord_full,
            "seed_prior_roi": seed_prior_roi,
            "modal_weights": modal_weights,
            "image_roi": image_roi,
        }

        if seg_gt is not None:
            seg_roi_target = (
                roi_pack["seg_roi"]
                if roi_pack.get("seg_roi", None) is not None
                else self._build_seg_roi_target(
                    seg_gt=seg_gt,
                    crop_infos=crop_infos,
                    paired_mode=paired_mode,
                )
            )
            out["seg_roi_target"] = seg_roi_target

        if cls_gt is not None:
            out["cls_gt"] = cls_gt

        return out

    def compute_total_loss(
        self,
        model_out: Dict[str, Any],
        seg_gt: Optional[torch.Tensor] = None,
        cls_gt: Optional[torch.Tensor] = None,
        detector_modal_consistency_weight: float = 0.10,
        detector_coord_weight: float = 0.25,
        detector_sharpness_weight: float = 0.05,
        detector_center_blob_weight: float = 0.10,
    ) -> Dict[str, torch.Tensor]:
        """
        内嵌总损失：
        total = detector_loss + roi_seg_loss + cls_loss
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss = None

        # 1) detector loss
        if seg_gt is not None:
            det_loss_dict = self.detector.compute_loss(
                out=model_out["detector_out"],
                seg_gt=seg_gt,
                modal_consistency_weight=detector_modal_consistency_weight,
                coord_weight=detector_coord_weight,
                sharpness_weight=detector_sharpness_weight,
                center_blob_weight=detector_center_blob_weight,
            )
            loss_detector = det_loss_dict["loss_total"] * self.detector_loss_weight
            loss_dict["loss_detector"] = loss_detector
            for k, v in det_loss_dict.items():
                loss_dict[f"detector_{k}"] = v
            total_loss = (
                loss_detector if total_loss is None else total_loss + loss_detector
            )

        # 2) roi segmentation loss
        if seg_gt is not None:
            seg_roi_target = model_out["seg_roi_target"]
            seg_loss_dict = self._seg_loss(
                seg_logits=model_out["seg_logits_roi"],
                seg_target=seg_roi_target,
            )
            loss_seg = seg_loss_dict["loss_seg_total"] * self.seg_loss_weight
            loss_dict["loss_seg"] = loss_seg
            for k, v in seg_loss_dict.items():
                loss_dict[k] = v
            total_loss = loss_seg if total_loss is None else total_loss + loss_seg

        # 3) classification loss
        if cls_gt is not None:
            loss_cls_raw = self._cls_loss(
                cls_logits=model_out["cls_logits"],
                cls_gt=cls_gt,
            )
            loss_cls = loss_cls_raw * self.cls_loss_weight
            loss_dict["loss_cls"] = loss_cls
            loss_dict["loss_cls_raw"] = loss_cls_raw
            total_loss = loss_cls if total_loss is None else total_loss + loss_cls

        if total_loss is None:
            raise RuntimeError(
                "compute_total_loss requires at least one of seg_gt or cls_gt."
            )

        loss_dict["loss_total"] = total_loss
        return loss_dict


# =========================
# Debug
# =========================
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 100)
    print("SeedGuidedMultiTaskModel DEBUG START")
    print("=" * 100)

    # -------------------------
    # 1. 构造模型
    # -------------------------
    B = 2
    Cin = 2  # 输入模态数
    Cout = 2  # 输出通道（可改成1/3测试另一分支）
    W, H, Z = 128, 128, 64

    model = SeedGuidedMultiTaskModel(
        in_channel=Cin,
        out_channel=Cout,
        detector_base_ch=32,
        detector_feat_dim=96,
        detector_stride_xy=4,
        roi_size_fullres=(64, 64, 32),
        backbone_base_ch=32,
    )

    model = model.cuda()
    model.train()

    # -------------------------
    # 2. 构造输入数据
    # -------------------------
    image_full = torch.randn(B, Cin, W, H, Z).cuda()

    # segmentation GT（注意这里支持 multi-channel）
    seg_gt = torch.zeros(B, Cout, W, H, Z).cuda()

    # 构造一个假 lesion（中心 blob）
    for b in range(B):
        cx, cy, cz = 60, 60, 30
        seg_gt[b, :, cx - 5 : cx + 5, cy - 5 : cy + 5, cz - 3 : cz + 3] = 1.0

    # 分类标签
    cls_gt = torch.randint(0, 2, (B, 1)).float().cuda()

    print(f"Input image shape     : {tuple(image_full.shape)}")
    print(f"GT seg shape          : {tuple(seg_gt.shape)}")
    print(f"GT cls shape          : {tuple(cls_gt.shape)}")

    # -------------------------
    # 3. 前向传播
    # -------------------------
    out = model(
        image_full=image_full,
        seg_gt=seg_gt,
        cls_gt=cls_gt,
    )

    print("-" * 100)
    print("Forward Outputs:")

    print(f"seg_logits_full       : {tuple(out['seg_logits_full'].shape)}")
    print(f"seg_logits_roi        : {tuple(out['seg_logits_roi'].shape)}")
    print(f"cls_logits            : {tuple(out['cls_logits'].shape)}")
    print(f"image_roi             : {tuple(out['image_roi'].shape)}")
    print(f"seed_prior_roi        : {tuple(out['seed_prior_roi'].shape)}")

    if "seg_roi_target" in out:
        print(f"seg_roi_target        : {tuple(out['seg_roi_target'].shape)}")

    print("-" * 100)

    # -------------------------
    # 4. Detector 信息
    # -------------------------
    det = out["detector_out"]

    print("Detector Outputs:")
    print(f"seed_coord_modal      : {tuple(det.seed_coord_modal.shape)}")
    print(f"seed_coord_fused      : {tuple(det.seed_coord_fused.shape)}")
    print(f"seed_region_modal     : {tuple(det.seed_region_modal.shape)}")
    print(f"seed_region_fused     : {tuple(det.seed_region_fused.shape)}")

    print("-" * 100)

    # -------------------------
    # 5. Crop 信息
    # -------------------------
    crop_infos = out["crop_infos"]

    print("Crop Infos (per sample):")
    for b in range(B):
        if isinstance(crop_infos[b], list):
            # paired mode
            info = crop_infos[b][0]
        else:
            info = crop_infos[b]

        print(
            f"Sample {b}: "
            f"x:[{info['x1']},{info['x2']}) "
            f"y:[{info['y1']},{info['y2']}) "
            f"z:[{info['z1']},{info['z2']})"
        )

    print("-" * 100)

    # -------------------------
    # 6. 模态权重
    # -------------------------
    if "modal_weights" in out:
        print("Modal Weights (mean per modal):")
        print(out["modal_weights"].view(B, -1).detach().cpu())

    print("-" * 100)

    # -------------------------
    # 7. Loss 计算
    # -------------------------
    loss_dict = model.compute_total_loss(
        model_out=out,
        seg_gt=seg_gt,
        cls_gt=cls_gt,
    )

    print("Loss Breakdown:")
    for k, v in loss_dict.items():
        print(f"{k:25s}: {float(v.item()):.6f}")

    print("-" * 100)

    # -------------------------
    # 8. 反向传播测试
    # -------------------------
    loss = loss_dict["loss_total"]
    loss.backward()

    print("Backward PASS ✔ (no error)")

    print("=" * 100)
    print("DEBUG FINISHED")
    print("=" * 100)
