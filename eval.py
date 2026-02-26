from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch
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

    cid = str(case_id).strip()
    for e in dg_valid:
        if cid in str(e.get("id", "")).strip():
            return e
    for e in dg_valid:
        if cid in str(e.get("excel_id", "")).strip():
            return e

    raise ValueError(f"[eval.py] case_id='{case_id}' not found in DG(val) list. Try id or excel_id.")


@torch.no_grad()
def export_case_prediction(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    cfg: Any,
) -> Dict[str, str]:
    """
    cfg.eval.case_id 指定病例，导出：
    - 原始图像（各模态，直接拷贝原 nii.gz）
    - 原始标签（各模态，直接拷贝原 nii.gz）
    - 模型预测 mask（各模态分别保存一份 + 额外保存一个统一文件名）
    注意：预测 mask 是在 val transforms 之后的空间（target_size）。
    """
    case_id = str(cfg.eval.case_id).strip()
    if not case_id:
        raise ValueError("[eval.py] cfg.eval.mode='export' requires cfg.eval.case_id")

    out_dir = Path(getattr(cfg.eval, "out_dir", "exports")) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    entry = _find_case_entry_in_val(cfg, case_id)
    mods: List[str] = list(cfg.data.use_modalities)

    saved: Dict[str, str] = {}

    # 1) 转存原始图像/标签（直接拷贝）
    for m in mods:
        img_p = entry[f"image_{m}"]
        lab_p = entry[f"label_{m}"]

        dst_img = out_dir / f"{entry['id']}_{m}_image_raw.nii.gz"
        dst_lab = out_dir / f"{entry['id']}_{m}_label_raw.nii.gz"

        shutil.copy2(img_p, dst_img)
        shutil.copy2(lab_p, dst_lab)

        saved[f"raw_image_{m}"] = str(dst_img)
        saved[f"raw_label_{m}"] = str(dst_lab)

    # 2) 复用 loader.py 的 val transforms，保证与训练一致
    transform = get_transforms(cfg.data, stage="val")
    sample = {f"image_{m}": entry[f"image_{m}"] for m in mods}
    sample.update({f"label_{m}": entry[f"label_{m}"] for m in mods})
    data = transform(sample)

    x = data["image"].unsqueeze(0).to(accelerator.device)  # [1, C, W, H, Z]
    thr = float(getattr(cfg.eval, "threshold", 0.5))

    with accelerator.autocast():
        out = model(x)

    prob = torch.sigmoid(out.logits)
    pred = (prob > thr).to(torch.uint8)  # [1, 1(or K), W, H, Z]

    # 3) 保存预测 mask 为 nii.gz
    # 为保证“可打开”：使用 nibabel 写 NIfTI，affine 取原始第一个模态
    import nibabel as nib

    ref_img = nib.load(entry[f"image_{mods[0]}"])
    affine = ref_img.affine

    pred_np = pred.squeeze(0).detach().cpu().numpy()  # [C_or_1, W, H, Z] 或 [W,H,Z]
    if pred_np.ndim == 4:
        pred_np = pred_np[0]  # 默认导出第0通道（前景）

    pred_nii = nib.Nifti1Image(pred_np.astype("uint8"), affine=affine)

    # 各模态分别保存一份（mask 相同，便于你按模态对照查看）
    for m in mods:
        dst_pred = out_dir / f"{entry['id']}_{m}_pred_mask_thr{thr:.2f}.nii.gz"
        nib.save(pred_nii, str(dst_pred))
        saved[f"pred_mask_{m}"] = str(dst_pred)

    # 额外保存统一命名
    dst_pred_uni = out_dir / f"{entry['id']}_pred_mask_thr{thr:.2f}.nii.gz"
    nib.save(pred_nii, str(dst_pred_uni))
    saved["pred_mask"] = str(dst_pred_uni)

    return saved


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