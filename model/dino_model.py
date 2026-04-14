import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import Dinov2Model

from mamba_ssm import Mamba
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# =========================
# Output
# =========================
@dataclass
class Output:
    seg: torch.Tensor
    cls: torch.Tensor


# =========================
# DINOv2 slice encoder
# =========================
class DinoEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.proj = nn.Conv2d(in_ch, 3, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, W, H, Z)
        return:
            feat: (B, C_feat, w, h, Z)
        """
        B, C, W, H, Z = x.shape

        # slice-wise
        x = x.permute(0, 4, 1, 2, 3).reshape(B * Z, C, W, H)
        x = self.proj(x)

        outputs = self.model(pixel_values=x)
        out = outputs.last_hidden_state  # (B*Z, N+1, C_feat), includes CLS token

        # 去掉 CLS token
        patch_tokens = out[:, 1:, :]  # (B*Z, N, C_feat)

        BZ, N, C_feat = patch_tokens.shape
        s = int(N ** 0.5)

        if s * s != N:
            raise ValueError(
                f"Patch token number {N} is not a perfect square after removing CLS token. "
                f"Input spatial size may be incompatible with DINOv2 patch embedding."
            )

        # (B*Z, N, C) -> (B, C, w, h, Z)
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, Z, C_feat, s, s)
        patch_tokens = patch_tokens.permute(0, 2, 3, 4, 1).contiguous()

        return patch_tokens

class LocalLesionStem(nn.Module):
    """
    高分辨率局部病灶锚定分支

    输入:
        x: (B, in_ch, W, H, Z)

    输出:
        local_feat:   (B, local_ch, W/4, H/4, Z)
        anchor_logits:(B, 1,       W/4, H/4, Z)

    设计目的：
    - 保留高于 DINOv2 9x9 的空间分辨率
    - 专门负责小病灶/边界/局部异常区域的定位
    - 为全局语义主干提供更可靠的 lesion anchor
    """

    def __init__(self, in_ch: int, base_ch: int = 32, local_ch: int = 128):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_ch),
            nn.GELU(),
            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_ch),
            nn.GELU(),
        )

        # 128 -> 64 (仅降 XY，不降 Z)
        self.down1 = nn.Sequential(
            nn.Conv3d(
                base_ch,
                base_ch * 2,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(base_ch * 2),
            nn.GELU(),
        )

        # 64 -> 32
        self.down2 = nn.Sequential(
            nn.Conv3d(
                base_ch * 2,
                local_ch,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(local_ch),
            nn.GELU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(local_ch, local_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(local_ch),
            nn.GELU(),
            nn.Conv3d(local_ch, local_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(local_ch),
            nn.GELU(),
        )

        self.anchor_head = nn.Conv3d(local_ch, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        local_feat = self.refine(h)
        anchor_logits = self.anchor_head(local_feat)
        return local_feat, anchor_logits

class LocalGlobalLesionFusion(nn.Module):
    """
    将高分辨率 local lesion anchor 融合回 DINOv2 + SliceMamba 的全局语义特征

    输入:
        z_global:       (B, global_dim, w, h, Z)   # 通常是 (B,768,9,9,Z)
        local_feat:     (B, local_ch,   Wl,Hl,Z)
        local_logits:   (B, 1,          Wl,Hl,Z)

    输出:
        z_fused:        (B, global_dim, w, h, Z)
    """

    def __init__(self, global_dim: int = 768, local_ch: int = 128):
        super().__init__()
        self.global_dim = global_dim

        self.local_proj = nn.Sequential(
            nn.Conv3d(local_ch, global_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(global_dim),
            nn.GELU(),
        )

        self.anchor_embed = nn.Sequential(
            nn.Conv3d(1, global_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(global_dim),
            nn.GELU(),
            nn.Conv3d(global_dim, global_dim, kernel_size=1, bias=True),
        )

        self.fuse_gate = nn.Sequential(
            nn.Conv3d(global_dim * 3, global_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(global_dim),
            nn.GELU(),
            nn.Conv3d(global_dim, global_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(global_dim, global_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(global_dim),
            nn.GELU(),
            nn.Conv3d(global_dim, global_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(global_dim),
            nn.GELU(),
        )

    def forward(
        self,
        z_global: torch.Tensor,
        local_feat: torch.Tensor,
        local_logits: torch.Tensor,
    ) -> torch.Tensor:
        B, C, w, h, Z = z_global.shape

        # 对齐到全局低分辨率空间
        local_feat_ds = F.interpolate(
            local_feat, size=(w, h, Z), mode="trilinear", align_corners=False
        )
        local_logits_ds = F.interpolate(
            local_logits, size=(w, h, Z), mode="trilinear", align_corners=False
        )

        local_feat_proj = self.local_proj(local_feat_ds)
        local_prior = torch.sigmoid(local_logits_ds)
        local_anchor = self.anchor_embed(local_prior)

        gate = self.fuse_gate(
            torch.cat([z_global, local_feat_proj, local_anchor], dim=1)
        )

        z_fused = z_global + gate * (local_feat_proj + local_anchor)
        z_fused = self.refine(z_fused)
        return z_fused
    
# =========================
# Slice-direction Mamba
# =========================
class SliceMamba(nn.Module):
    """
    参考原始 TriMambaCanonicalizer 的稳定调用方式：
        x_flat = x.reshape(B, C, -1).transpose(-1, -2)
    然后只取三向 Mamba 的 slc 分支，作为 slice-direction 表征。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
            bimamba_type="v3",
            nslices=8,
        )

        # 原始实现里三向分支输出维度不是 dim，而是 d_inner，需要投影回 dim
        self.dir_proj = nn.Conv3d(self.mamba.d_inner, dim, kernel_size=1, bias=True)

        self.slice_gate = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=True),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        x: (B, C, W, H, Z)
        对当前 DINOv2 输出，例如 (B, 768, 9, 9, 64)，
        展平后长度为 9*9*64=5184，可被 nslices=8 整除，符合原始调用逻辑。
        """
        B, C, W, H, Z = x.shape
        assert C == self.dim, f"Expected channel dim {self.dim}, got {C}"

        # -----------------------------------------
        # 1) 完全沿用原始稳定调用方式
        #    (B, C, W, H, Z) -> (B, W*H*Z, C)
        # -----------------------------------------
        x_flat = x.reshape(B, C, -1).transpose(-1, -2).contiguous()
        x_flat = self.norm(x_flat)

        # -----------------------------------------
        # 2) 三向 Mamba 输出
        #    原始成功方案：out, fwd, bwd, slc = self.mamba(x_flat)
        # -----------------------------------------
        out, fwd, bwd, slc = self.mamba(x_flat)

        # -----------------------------------------
        # 3) 只取 slc 分支，作为 slice-direction 表征
        #    slc shape: (B, d_inner, W*H*Z) 或 (B, W*H*Z, d_inner)
        #    参考原始 _to_3d 逻辑做安全变换
        # -----------------------------------------
        num_voxels = W * H * Z

        if slc.shape[1] == num_voxels:
            # (B, W*H*Z, d_inner) -> (B, d_inner, W, H, Z)
            slc_3d = slc.transpose(1, 2).reshape(B, slc.shape[2], W, H, Z)
        elif slc.shape[2] == num_voxels:
            # (B, d_inner, W*H*Z) -> (B, d_inner, W, H, Z)
            slc_3d = slc.reshape(B, slc.shape[1], W, H, Z)
        else:
            raise ValueError(
                f"Cannot reshape slc with shape {tuple(slc.shape)} back to "
                f"(B, *, {W}, {H}, {Z}); expected one dimension to equal {num_voxels}."
            )

        # 投影回 dim
        slc_3d = self.dir_proj(slc_3d)

        # -----------------------------------------
        # 4) slice-aware residual recomposition
        # -----------------------------------------
        gate = self.slice_gate(slc_3d)
        z = x + gate * slc_3d
        z = self.refine(z)

        return z

# =========================
# Segmentation Decoder
# =========================
class OrderedCompartmentFieldSegDecoder(nn.Module):
    """
    Ordered Compartment Field Decoder (upgraded)

    升级点：
    1) 不再从同一base特征直接并行预测4个field
    2) 改为 progressive refinement:
       core -> mural -> extra -> meso
    3) 让 ordered structure 同时存在于 feature generation 和 field generation 中
    """

    def __init__(self, dim, out_ch):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
        )

        self.core_refine = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
        )
        self.mural_refine = nn.Sequential(
            nn.Conv3d(dim + 1, dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
        )
        self.extra_refine = nn.Sequential(
            nn.Conv3d(dim + 1, dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
        )
        self.meso_refine = nn.Sequential(
            nn.Conv3d(dim + 1, dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
        )

        self.field_core = nn.Conv3d(dim, 1, 1)
        self.field_mural = nn.Conv3d(dim, 1, 1)
        self.field_extra = nn.Conv3d(dim, 1, 1)
        self.field_meso = nn.Conv3d(dim, 1, 1)

        self.out_proj = nn.Conv3d(4, out_ch, 1)

    def forward(self, x, shape):
        f0 = self.stem(x)

        # core
        f_core = self.core_refine(f0)
        phi_core = self.field_core(f_core)

        # mural: conditioned on core field
        f_mural = self.mural_refine(torch.cat([f0, phi_core], dim=1))
        phi_mural = phi_core + F.softplus(self.field_mural(f_mural))

        # extra: conditioned on mural field
        f_extra = self.extra_refine(torch.cat([f0, phi_mural], dim=1))
        phi_extra = phi_mural + F.softplus(self.field_extra(f_extra))

        # meso: conditioned on extra field
        f_meso = self.meso_refine(torch.cat([f0, phi_extra], dim=1))
        phi_meso = phi_extra + F.softplus(self.field_meso(f_meso))

        fields = torch.cat([phi_core, phi_mural, phi_extra, phi_meso], dim=1)

        fields = F.interpolate(
            fields, size=shape, mode="trilinear", align_corners=False
        )

        seg = self.out_proj(fields)
        return seg


# =========================
# Classification Decoder
# =========================
class MesorectalClinicalTokenReasoner(nn.Module):
    """
    Mesorectal Clinical Token Reasoner (upgraded)

    升级点：
    1) clinical tokens 不再纯由全局平均生成
    2) 可显式使用 segmentation evidence 作为 lesion prior
    3) 保持 v3 Mamba + slc + clinical queries 的稳定结构
    """

    def __init__(self, dim: int, num_heads: int = 4, num_slices: int = 8):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                f"mamba_ssm is required but unavailable: {_MAMBA_IMPORT_ERROR}"
            )

        self.dim = dim

        self.proj = nn.Linear(dim, dim)
        self.token_norm = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
            bimamba_type="v3",
            nslices=num_slices,
        )

        self.dir_proj = nn.Linear(self.mamba.d_inner, dim)

        self.query_embed = nn.Parameter(torch.randn(4, dim))

        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def _extract_slc(self, mamba_out):
        if isinstance(mamba_out, tuple):
            if len(mamba_out) == 4:
                _, _, _, slc = mamba_out
                return slc
            return mamba_out[0]
        return mamba_out

    def _align(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got shape {tuple(t.shape)}")

        if t.shape[-1] == self.dim:
            return t
        if t.shape[-1] == self.mamba.d_inner:
            return self.dir_proj(t)
        if t.shape[1] == self.mamba.d_inner:
            t = t.transpose(1, 2).contiguous()
            return self.dir_proj(t)

        raise ValueError(
            f"Cannot align token shape {tuple(t.shape)} to dim={self.dim} "
            f"with d_inner={self.mamba.d_inner}."
        )

    def forward(self, x: torch.Tensor, seg_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, W, H, Z = x.shape
        assert C == self.dim, f"Expected channel dim {self.dim}, got {C}"

        # lesion prior from segmentation evidence
        if seg_logits is not None:
            # seg_logits: (B, out_ch, W_full, H_full, Z)
            seg_prior = torch.sigmoid(seg_logits).mean(dim=1, keepdim=True)  # (B,1,W,H,Z) after resize below
            seg_prior = F.interpolate(
                seg_prior, size=(W, H, Z), mode="trilinear", align_corners=False
            )
        else:
            seg_prior = torch.ones(B, 1, W, H, Z, device=x.device, dtype=x.dtype)

        # weighted lesion-centric tokens
        x_weighted = x * seg_prior

        t_core = x_weighted.mean(dim=(2, 3))  # (B,C,Z)

        diff = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        first = diff[:, :, :, :, :1]
        diff = torch.cat([first, diff], dim=-1)
        t_front = (diff.abs() * seg_prior).mean(dim=(2, 3))  # (B,C,Z)

        context_map = F.avg_pool3d(
            x, kernel_size=(3, 3, 3), stride=1, padding=1
        )
        t_context = (context_map * seg_prior).mean(dim=(2, 3))  # (B,C,Z)

        t = (t_core + t_front + t_context) / 3.0
        t = t.permute(0, 2, 1).contiguous()  # (B,Z,C)

        t = self.proj(t)
        t = self.token_norm(t)

        mamba_out = self.mamba(t)
        t = self._extract_slc(mamba_out)
        t = self._align(t)

        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        q_out, _ = self.cross_attn(queries, t, t)
        q_out = self.attn_norm(queries + q_out)
        q_out = q_out + self.mlp(q_out)

        v = q_out.mean(dim=1)
        return self.cls_head(v)


# =========================
# Model
# =========================
class Model(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.enc = DinoEncoder(in_ch)
        self.mamba = SliceMamba(768)

        # -----------------------------
        # 新增：高分辨率局部病灶分支
        # -----------------------------
        self.local_stem = LocalLesionStem(
            in_ch=in_ch,
            base_ch=32,
            local_ch=128,
        )

        self.local_global_fuse = LocalGlobalLesionFusion(
            global_dim=768,
            local_ch=128,
        )

        # -----------------------------
        # segmentation / classification heads
        # -----------------------------
        self.seg_coarse = OrderedCompartmentFieldSegDecoder(768, out_ch)
        self.seg = OrderedCompartmentFieldSegDecoder(768, out_ch)
        self.cls = MesorectalClinicalTokenReasoner(768)

        # DDP-safe branch
        self.dummy = nn.Conv3d(768, 768, 1)

    def forward(self, x):
        B, C, W, H, Z = x.shape

        # -----------------------------
        # 1) global foundation branch
        # -----------------------------
        z_global = self.enc(x)       # (B,768,9,9,Z)
        z_global = self.mamba(z_global)

        # -----------------------------
        # 2) local lesion branch
        # -----------------------------
        local_feat, local_anchor_logits = self.local_stem(x)   # high-res anchor

        # -----------------------------
        # 3) fuse local lesion anchor into global semantic feature
        # -----------------------------
        z_fused = self.local_global_fuse(
            z_global,
            local_feat,
            local_anchor_logits,
        )

        # -----------------------------
        # 4) coarse segmentation
        # -----------------------------
        seg_coarse = self.seg_coarse(z_fused, (W, H, Z))

        # -----------------------------
        # 5) final segmentation
        #    这里不再使用旧的 LesionAwareSemanticReweighting
        #    因为 lesion anchoring 已经由 local branch 提供
        # -----------------------------
        seg = self.seg(z_fused, (W, H, Z))

        # -----------------------------
        # 6) classification
        #    仍然使用 final seg 作为 lesion prior
        # -----------------------------
        cls = self.cls(z_fused, seg_logits=seg)

        # -----------------------------
        # 7) DDP-safe
        # -----------------------------
        dummy_scalar = 0.0 * self.dummy(z_fused).mean()
        seg = seg + dummy_scalar
        cls = cls + dummy_scalar

        return Output(seg=seg, cls=cls)
    
# =========================
# Debug
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(2, 3, 128, 128, 64).to(device)

    model = Model(3, 3).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)

    print("=" * 80)
    print("Debug OK")
    print("=" * 80)
    print("Input:", x.shape)
    print("Seg:", out.seg.shape)
    print("Cls:", out.cls.shape)
