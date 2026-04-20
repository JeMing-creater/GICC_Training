import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from monai.networks.nets import SwinUNETR
from mamba_ssm import Mamba


# =========================
# Output
# =========================
@dataclass
class Output:
    seg: torch.Tensor
    cls: torch.Tensor
    core: torch.Tensor
    rim: torch.Tensor
    unc: torch.Tensor


# =========================
# Z-only Mamba（显存安全）
# =========================
class ZMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
            bimamba_type="v3",
            nslices=8,
        )

        self.proj = nn.Conv3d(self.mamba.d_inner, dim, 1)

    def forward(self, x):
        B, C, W, H, Z = x.shape

        # (B,C,W,H,Z) → (B*W*H, Z, C)
        x_seq = x.permute(0,2,3,4,1).reshape(-1, Z, C)
        x_seq = self.norm(x_seq)

        out, fwd, bwd, slc = self.mamba(x_seq)

        def reshape_back(t):
            if t.shape[-1] == C:
                y = t
            else:
                y = t.transpose(1,2)
            return y.reshape(B, W, H, Z, -1).permute(0,4,1,2,3)

        fwd = self.proj(reshape_back(fwd))
        bwd = self.proj(reshape_back(bwd))
        slc = self.proj(reshape_back(slc))

        return {
            "core": 0.5 * (fwd + bwd),
            "rim": slc,
            "unc": torch.abs(fwd - bwd),
        }


# =========================
# Uncertainty Head
# =========================
class UncertaintyHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.core = nn.Conv3d(dim, 1, 1)
        self.rim  = nn.Conv3d(dim, 1, 1)
        self.unc  = nn.Conv3d(dim, 1, 1)

    def forward(self, tri_feat, out_size):
        core = self.core(tri_feat["core"])
        rim  = self.rim(tri_feat["rim"])
        unc  = self.unc(tri_feat["unc"])

        core = F.interpolate(core, size=out_size, mode="trilinear", align_corners=False)
        rim  = F.interpolate(rim, size=out_size, mode="trilinear", align_corners=False)
        unc  = F.interpolate(unc, size=out_size, mode="trilinear", align_corners=False)

        return core, rim, unc


# =========================
# Classification Head
# =========================
class ClassificationHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, feat, core, rim, unc):
        core_feat = (feat * torch.sigmoid(core)).mean(dim=(2,3,4))
        rim_feat  = (feat * torch.sigmoid(rim)).mean(dim=(2,3,4))
        unc_feat  = (feat * torch.sigmoid(unc)).mean(dim=(2,3,4))

        x = torch.cat([core_feat, rim_feat, unc_feat], dim=1)
        return self.fc(x)


# =========================
# Model
# =========================
class Model(nn.Module):
    def __init__(self, in_ch, out_ch, img_size=(128,128,64)):
        super().__init__()

        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=48,
        )

        # 降维（关键）
        self.feat_proj = nn.Conv3d(out_ch, 128, 1)

        self.z_mamba = ZMamba(128)
        self.uncertainty = UncertaintyHead(128)
        self.cls_head = ClassificationHead(128)

    # =========================
    # Forward（DDP安全版）
    # =========================
    def forward(self, x):
        B, C, W, H, Z = x.shape

        seg = self.backbone(x)

        feat = self.feat_proj(seg)

        # ⭐ 下采样（降低显存）
        feat_ds = F.interpolate(feat, scale_factor=(0.5,0.5,1), mode="trilinear")

        tri = self.z_mamba(feat_ds)

        # 上采样回原尺寸
        for k in tri:
            tri[k] = F.interpolate(tri[k], size=(W,H,Z), mode="trilinear")

        core, rim, unc = self.uncertainty(tri, (W,H,Z))

        cls = self.cls_head(feat, core, rim, unc)

        # =========================
        # ⭐ DDP安全（关键）
        # =========================
        dummy = 0.0
        for p in self.parameters():
            if p.requires_grad:
                dummy = dummy + 0.0 * p.sum()

        seg = seg + dummy
        cls = cls + dummy

        return Output(seg, cls, core, rim, unc)

    # =========================
    # Dice Loss
    # =========================
    def dice_loss(self, pred, target, eps=1e-5):
        inter = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        return 1 - (2 * inter + eps) / (union + eps)

    # =========================
    # Compute Loss（DDP安全）
    # =========================
    def compute_loss(self, out, seg_gt, cls_gt):

        seg, cls, core, rim, unc = out.seg, out.cls, out.core, out.rim, out.unc

        seg_prob = torch.sigmoid(seg)

        # Seg
        loss_seg = self.dice_loss(seg_prob, seg_gt) + \
                   F.binary_cross_entropy_with_logits(seg, seg_gt)

        # Cls（安全）
        if cls_gt is not None:
            loss_cls = F.binary_cross_entropy_with_logits(cls, cls_gt)
        else:
            loss_cls = cls.mean() * 0.0

        # Boundary
        grad = torch.abs(seg_prob[:, :, 1:] - seg_prob[:, :, :-1])
        boundary = F.pad(grad, (0,0,0,0,0,1))

        unc_prob = torch.sigmoid(unc)
        core_prob = torch.sigmoid(core)
        rim_prob  = torch.sigmoid(rim)

        loss_unc = F.l1_loss(unc_prob, boundary.detach())
        loss_core = torch.mean(core_prob * unc_prob)
        loss_rim  = F.l1_loss(rim_prob, unc_prob)

        loss_total = (
            loss_seg
            + loss_cls
            + 0.5 * loss_unc
            + 0.2 * loss_core
            + 0.3 * loss_rim
        )

        return {
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_cls": loss_cls,
            "loss_unc": loss_unc,
            "loss_core": loss_core,
            "loss_rim": loss_rim,
        }


# =========================
# Debug
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(1,2,128,128,64).to(device)
    seg_gt = torch.randint(0,2,(1,1,128,128,64)).float().to(device)
    cls_gt = torch.tensor([[1.0]]).to(device)

    model = Model(2,1).to(device)

    out = model(x)
    loss_dict = model.compute_loss(out, seg_gt, cls_gt)

    print("Seg:", out.seg.shape)
    print("Cls:", out.cls.shape)

    print("\nLoss:")
    for k,v in loss_dict.items():
        print(k, v.item())