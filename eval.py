from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from accelerate import Accelerator

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete

from src.utils import load_cfg, select_label_channel, load_weights
from src.loader import get_loaders, get_transforms, build_data_list
from model import build_model


@torch.no_grad()
def eval_val_one_epoch(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    val_loader,
    cfg: Any,
) -> Dict[str, float]:
    """
    重新执行一次验证集评估，输出 Dice / HD95
    - include_background=False
    - HD95 对 pred/gt 空前景样本跳过（避免 nan/inf）
    """
    model.eval()

    thr = float(getattr(cfg.eval, "threshold", 0.5))
    act = Activations(sigmoid=True)
    to_bin = AsDiscrete(threshold=thr)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    out_ch = int(getattr(cfg.model, "out_ch", 1))
    take_first = bool(getattr(cfg.data, "label_take_first_channel", True))
    device = accelerator.device

    # 0-dim 标量累计，避免 shape 广播问题
    dice_sum = torch.tensor(0.0, device=device)
    dice_cnt = torch.tensor(0.0, device=device)
    hd95_sum = torch.tensor(0.0, device=device)
    hd95_cnt = torch.tensor(0.0, device=device)

    for batch in val_loader:
        if batch is None or (not isinstance(batch, dict)) or ("image" not in batch) or ("seg_label" not in batch):
            continue

        x = batch["image"].to(device, non_blocking=True)
        y = batch["seg_label"].to(device, non_blocking=True)
        y = select_label_channel(y, out_ch=out_ch, take_first=take_first)

        with accelerator.autocast():
            out = model(x)

        prob = act(out.logits)
        pred = to_bin(prob)

        # Dice
        dice_metric(pred, y)
        d = dice_metric.aggregate().detach()
        dice_metric.reset()
        if torch.isfinite(d):
            dice_sum = dice_sum + d.float().view(())
            dice_cnt = dice_cnt + torch.tensor(1.0, device=device)

        # HD95（过滤空前景）
        pred_has = (pred.sum(dim=(1, 2, 3, 4)) > 0)
        gt_has = (y.sum(dim=(1, 2, 3, 4)) > 0)
        valid = pred_has & gt_has
        if valid.any():
            hd95_metric(pred[valid], y[valid])
            h = hd95_metric.aggregate().detach()
            hd95_metric.reset()
            if torch.isfinite(h):
                hd95_sum = hd95_sum + h.float().view(())
                hd95_cnt = hd95_cnt + torch.tensor(1.0, device=device)

    # 多卡/单卡都可：reduce 一下
    dice_sum_g = accelerator.reduce(dice_sum, reduction="sum")
    dice_cnt_g = accelerator.reduce(dice_cnt, reduction="sum")
    hd95_sum_g = accelerator.reduce(hd95_sum, reduction="sum")
    hd95_cnt_g = accelerator.reduce(hd95_cnt, reduction="sum")

    dice_mean = (dice_sum_g / dice_cnt_g).item() if dice_cnt_g.item() > 0 else 0.0
    hd95_mean = (hd95_sum_g / hd95_cnt_g).item() if hd95_cnt_g.item() > 0 else 0.0

    return {"dice": float(dice_mean), "hd95": float(hd95_mean)}


def _find_case_entry_in_val(cfg: Any, case_id: str) -> Dict[str, str]:
    """
    在验证集（DG）列表中找到指定 case_id 对应的 entry（支持 id 或 excel_id）。
    entry 结构来自 build_data_list。
    """
    root_dir = cfg.data.root_dir
    leapfrog_list = cfg.data.get("leapfrog", [])
    req_mods = cfg.data.use_modalities

    dg_valid, _dg_fail = build_data_list(
        cfg.data.excel_configs.dg,
        root_dir,
        leapfrog_list,
        data_folder_name="All",
        required_modalities=req_mods,
        tag="DG",
    )
    
    mei_valid, mei_fail = build_data_list(cfg.data.excel_configs.mei, root_dir, leapfrog_list, "All", req_mods, "Mei")
    gz_valid, gz_fail = build_data_list(cfg.data.excel_configs.gz, root_dir, leapfrog_list, "All", req_mods, "GZ")
    
    valid = dg_valid + mei_valid + gz_valid

    cid = str(case_id).strip()
    for e in valid:
        if cid in str(e.get("id", "")).strip():
            return e
    for e in valid:
        if cid in str(e.get("excel_id", "")).strip():
            return e

    raise ValueError(f"[eval.py] case_id='{case_id}' not found in DG(val) list. Try id or excel_id.")


def _resize_3d_volume_to_shape(
    vol: "np.ndarray",
    target_shape: tuple[int, int, int],
    *,
    is_label: bool,
) -> "np.ndarray":
    """
    将 3D 体数据 vol resize 到 target_shape。

    - vol: (X, Y, Z) 或 (H, W, D) 的 3D numpy
    - target_shape: (X, Y, Z)
    - is_label=True -> 最近邻 (避免标签插值产生非整数/破碎)
      is_label=False -> 三线性 (适合概率图)
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")

    # Torch 的 interpolate 认为输入是 [N, C, D, H, W]
    x = torch.from_numpy(vol).float()[None, None]  # [1,1,X,Y,Z] 但我们将其视作 [D,H,W] 的排列
    # 为稳妥起见：显式转成 [1,1,Z,Y,X] 再插值，再转回 [X,Y,Z]
    x = x.permute(0, 1, 4, 3, 2)  # [1,1,Z,Y,X]

    mode = "nearest" if is_label else "trilinear"
    y = F.interpolate(
        x,
        size=(target_shape[2], target_shape[1], target_shape[0]),  # (Z,Y,X)
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )

    y = y.permute(0, 1, 4, 3, 2)[0, 0]  # 回到 [X,Y,Z]
    out = y.detach().cpu().numpy()

    if is_label:
        out = (out > 0.5).astype(np.uint8)
    return out


def _resample_pred_to_rawspace(
    pred_np: np.ndarray,
    raw_shape: tuple[int, int, int],
    *,
    is_label: bool,
) -> np.ndarray:
    """
    将网络空间预测 resize 回 raw 体素网格大小。
    仅改变 voxel grid，不改变物理坐标。

    pred_np: (Xn,Yn,Zn)
    raw_shape: (Xr,Yr,Zr)
    """

    if pred_np.ndim != 3:
        raise ValueError(f"Expected 3D pred, got {pred_np.shape}")

    x = torch.from_numpy(pred_np).float()[None, None]  # [1,1,X,Y,Z]
    x = x.permute(0, 1, 4, 3, 2)  # -> [1,1,Z,Y,X]

    mode = "nearest" if is_label else "trilinear"

    y = F.interpolate(
        x,
        size=(raw_shape[2], raw_shape[1], raw_shape[0]),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )

    y = y.permute(0, 1, 4, 3, 2)[0, 0].cpu().numpy()

    if is_label:
        y = (y > 0.5).astype(np.uint8)

    return y


def _best_align_mask_by_axis_flips(
    mask_raw: np.ndarray,
    label_raw: np.ndarray,
    *,
    min_foreground: int = 5,
) -> tuple[np.ndarray, dict]:
    """
    对 mask_raw 尝试 3 个轴的所有 flip 组合（共 8 种），选择与 label_raw Dice 最高的版本。

    返回：
      - best_mask
      - info: {"best_dice": float, "flip_axes": tuple[int,...], "tested": list[...]}
    """
    if mask_raw.shape != label_raw.shape:
        raise ValueError(f"mask_raw.shape {mask_raw.shape} != label_raw.shape {label_raw.shape}")

    # 二值化（label 可能是 0/255 或多值）
    gt = (label_raw > 0).astype(np.uint8)
    if gt.sum() < min_foreground:
        # GT 太小/为空，不做自动校正，直接返回原 mask
        return mask_raw, {"best_dice": -1.0, "flip_axes": (), "tested": []}

    def dice(a: np.ndarray, b: np.ndarray) -> float:
        a = (a > 0).astype(np.uint8)
        b = (b > 0).astype(np.uint8)
        inter = int((a & b).sum())
        sa = int(a.sum())
        sb = int(b.sum())
        if sa + sb == 0:
            return 1.0
        return (2.0 * inter) / (sa + sb + 1e-8)

    best_d = -1.0
    best_axes: tuple[int, ...] = ()
    best_mask = mask_raw
    tested = []

    # 8 种 flip 组合
    flip_sets = [
        (), (0,), (1,), (2,),
        (0, 1), (0, 2), (1, 2),
        (0, 1, 2),
    ]

    for axes in flip_sets:
        cand = mask_raw
        for ax in axes:
            cand = np.flip(cand, axis=ax)
        d = dice(cand, gt)
        tested.append({"flip_axes": axes, "dice": float(d)})
        if d > best_d:
            best_d = d
            best_axes = axes
            best_mask = cand

    return best_mask.astype(np.uint8), {"best_dice": float(best_d), "flip_axes": best_axes, "tested": tested}


@torch.no_grad()
def export_case_prediction(
    *,
    accelerator: "Accelerator",
    model: "torch.nn.Module",
    cfg: Any,
) -> Dict[str, str]:
    """
    所有模态 × 三文件（精简版 + 自动镜像纠正）：
      - <id>_<m>_image_raw.nii.gz
      - <id>_<m>_label_raw.nii.gz
      - <id>_<m>_pred_mask_rawspace_thrX.nii.gz

    关键改进：
      - 在保存 pred mask 前，读取该模态 label_raw
      - 对 pred mask 尝试 8 种 axis flip 组合，选择 Dice 最高者保存
      - 解决典型的“左右镜像相反 / 上下镜像相反”错位问题
    """
    import shutil
    from pathlib import Path
    import numpy as np
    import torch
    import nibabel as nib
    from monai.data import Dataset, DataLoader

    from src.loader import get_transforms, collate_fn_ignore_none

    case_id = str(cfg.eval.case_id).strip()
    if not case_id:
        raise ValueError("[eval.py] cfg.eval.mode='export' requires cfg.eval.case_id")

    thr = float(getattr(cfg.eval, "threshold", 0.5))
    out_root = Path(getattr(cfg.eval, "out_dir", "exports"))
    out_dir = out_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    entry = _find_case_entry_in_val(cfg, case_id)
    mods = list(cfg.data.use_modalities)
    if len(mods) == 0:
        raise ValueError("[export_case_prediction] cfg.data.use_modalities is empty")

    # 1) 单病例 loader 走 transforms（与训练一致）
    transform = get_transforms(cfg.data, stage="val")
    ds = Dataset(data=[entry], transform=transform)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_ignore_none,
    )
    batch = next(iter(loader))
    x = batch["image"].to(accelerator.device, non_blocking=True)

    # 2) 推理（netspace）
    model.eval()
    with accelerator.autocast():
        out = model(x)

    prob = torch.sigmoid(out.logits)     # [1,out_ch,Xn,Yn,Zn]
    prob0_t = prob[:, 0:1]              # [1,1,Xn,Yn,Zn]

    saved: Dict[str, str] = {}

    # 3) 对每个模态：copy raw + 回写 raw + 自动 flip 校正 + 保存
    for m in mods:
        img_src = entry[f"image_{m}"]
        lab_src = entry[f"label_{m}"]

        img_dst = out_dir / f"{entry['id']}_{m}_image_raw.nii.gz"
        lab_dst = out_dir / f"{entry['id']}_{m}_label_raw.nii.gz"
        pred_dst = out_dir / f"{entry['id']}_{m}_pred_mask_rawspace_thr{thr:.2f}.nii.gz"

        shutil.copy2(img_src, img_dst)
        shutil.copy2(lab_src, lab_dst)

        ref_img = nib.load(str(img_dst))
        lab_img = nib.load(str(lab_dst))

        # 3.1 回写 raw（你现有逻辑：pred_affine=None 时是 shape 插值）
        pred_affine = None
        prob_raw = _resample_pred_to_ref_physical(
            prob0_t, ref_img, pred_affine=pred_affine, mode="trilinear"
        ).astype(np.float32)

        mask_raw = (prob_raw > thr).astype(np.uint8)

        # 3.2 ✅ 自动镜像纠正：尝试 8 种 flip，选 Dice 最高
        label_raw = lab_img.get_fdata()
        label_raw = (label_raw > 0).astype(np.uint8)

        best_mask, info = _best_align_mask_by_axis_flips(mask_raw, label_raw)

        # 3.3 保存（继承 header/qform/sform）
        _save_nifti_like(best_mask, ref_img, str(pred_dst), dtype="uint8")

        saved[f"{m}/raw_image"] = str(img_dst)
        saved[f"{m}/raw_label"] = str(lab_dst)
        saved[f"{m}/pred_mask"] = str(pred_dst)
        saved[f"{m}/flip_axes"] = str(info.get("flip_axes", ()))
        saved[f"{m}/best_dice_vs_raw_label"] = f"{info.get('best_dice', -1.0):.4f}"

    return saved

def _resample_pred_to_ref_physical(
    pred: "torch.Tensor",
    ref_img: "nib.Nifti1Image",
    *,
    pred_affine: "np.ndarray | None" = None,
    mode: str = "nearest",
) -> "np.ndarray":
    """
    将 pred（netspace, torch, shape [1,1,X,Y,Z] 或 [X,Y,Z]）按物理空间重采样到 ref_img 的网格。

    pred_affine:
      - pred 的 affine（若 None，则退化为“只按 shape 插值”的情况）
      - 强烈建议后续在 loader 中保存 resized 后 affine 再传入
    """
    import numpy as np
    import torch
    from monai.transforms import SpatialResample

    # ---- pred -> [1,1,X,Y,Z] ----
    if pred.ndim == 3:
        pred_t = pred[None, None]
    elif pred.ndim == 5:
        pred_t = pred
    else:
        raise ValueError(f"pred must be 3D or 5D, got {pred.shape}")

    # target 网格信息（来自 ref）
    target_affine = np.asarray(ref_img.affine, dtype=np.float64)
    target_shape = tuple(int(x) for x in ref_img.shape)  # (X,Y,Z)

    # 如果没有 pred_affine，就只能退化为 shape 插值
    if pred_affine is None:
        # fallback：纯 shape 插值（仍然会对齐 header，至少能打开）
        pred_np = pred_t[0, 0].detach().cpu().numpy()
        import torch.nn.functional as F
        x = torch.from_numpy(pred_np).float()[None, None].permute(0, 1, 4, 3, 2)
        y = F.interpolate(
            x,
            size=(target_shape[2], target_shape[1], target_shape[0]),
            mode="nearest" if mode == "nearest" else "trilinear",
            align_corners=False if mode != "nearest" else None,
        )
        y = y.permute(0, 1, 4, 3, 2)[0, 0].cpu().numpy()
        return y

    src_affine = np.asarray(pred_affine, dtype=np.float64)

    # SpatialResample 要求输入是 channel-first，且 affine 给它
    resampler = SpatialResample(mode=mode, align_corners=False if mode != "nearest" else None)

    # 输出 torch Tensor [1,1,Xr,Yr,Zr]
    out_t = resampler(pred_t, src_affine=src_affine, dst_affine=target_affine, dst_spatial_size=target_shape)
    out_np = out_t[0, 0].detach().cpu().numpy()
    return out_np

def _save_nifti_like(
    data_3d,
    ref_img,
    out_path: str,
    *,
    dtype: str = "uint8",
) -> None:
    """
    将 data_3d 保存为 NIfTI，并尽可能与 ref_img 的空间信息一致：
    - shape 必须与 ref_img.shape 一致
    - 复制 ref header（包含 qform/sform/zooms 等）
    """
    import numpy as np
    import nibabel as nib

    if data_3d.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape={data_3d.shape}")

    if tuple(data_3d.shape) != tuple(ref_img.shape):
        raise ValueError(
            f"Shape mismatch when saving nifti-like: data={data_3d.shape}, ref={ref_img.shape}"
        )

    hdr = ref_img.header.copy()
    data = data_3d.astype(np.dtype(dtype), copy=False)

    out = nib.Nifti1Image(data, affine=ref_img.affine, header=hdr)

    # 强制同步 qform/sform（ITK-SNAP/ITK 更依赖这些）
    try:
        out.set_qform(ref_img.get_qform(), code=int(ref_img.header.get("qform_code", 1)))
    except Exception:
        pass
    try:
        out.set_sform(ref_img.get_sform(), code=int(ref_img.header.get("sform_code", 1)))
    except Exception:
        pass

    nib.save(out, out_path)

def main():
    cfg = load_cfg("config.yml")

    # 基本检查
    mode = str(getattr(cfg.eval, "mode", "metrics")).lower().strip()
    weights_path = str(getattr(cfg.eval, "weights_path", "")).strip()
    if not weights_path:
        raise ValueError("[eval.py] cfg.eval.weights_path is required")

    accelerator = Accelerator()

    # 构建模型并加载权重
    model = build_model(cfg)
    info = load_weights(model=model, weights_path=weights_path, strict=False, map_location="cpu")
    model = accelerator.prepare(model)

    if accelerator.is_main_process:
        accelerator.print(
            f"[eval.py] Loaded weights: {weights_path} | missing={len(info['missing_keys'])} unexpected={len(info['unexpected_keys'])}"
        )

    if mode == "metrics":
        # 复用 loader.py：get_loaders(cfg.data) 返回 train/val loader
        _train_loader, val_loader = get_loaders(cfg.data)
        if cfg.eval.train == True:
            val_loader = _train_loader

        val_loader = accelerator.prepare(val_loader)
        stats = eval_val_one_epoch(accelerator=accelerator, model=model, val_loader=val_loader, cfg=cfg)
        if accelerator.is_main_process:
            accelerator.print(f"[eval.py] VAL metrics | Dice={stats['dice']:.4f} | HD95={stats['hd95']:.3f}")

    elif mode == "export":
        saved = export_case_prediction(accelerator=accelerator, model=model, cfg=cfg)
        if accelerator.is_main_process:
            accelerator.print("[eval.py] Export done. Saved files:")
            for k, v in saved.items():
                accelerator.print(f"  - {k}: {v}")

    else:
        raise ValueError(f"[eval.py] Unknown cfg.eval.mode: {mode!r}")


if __name__ == "__main__":
    main()