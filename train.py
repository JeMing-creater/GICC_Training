from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.metrics import DiceMetric, MeanIoU, compute_hausdorff_distance

from src.loader import get_loaders
from src.utils import (
    cfg_to_plain_dict,
    load_cfg,
    maybe_resume_from_latest,
    prepare_run_dir,
    save_best_weights_if_improved,
    save_latest_checkpoint,
    set_seed,
    start_txt_logger,
    select_label_channel,
)

# 按你的工程实际路径修改。如果 seed_model.py 位于 model/seed_model.py，这里保持不变。
from model.seed_model import SeedGuidedMultiTaskModel


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    cur = cfg
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            elif hasattr(cur, "get"):
                try:
                    cur = cur.get(part)
                except Exception:
                    return default
            else:
                return default
    return default if cur is None else cur


def _flatten_tb_config(cfg: Any) -> Dict[str, Any]:
    def _scalarize(v: Any) -> Any:
        if isinstance(v, (int, float, str, bool)):
            return v
        if torch.is_tensor(v):
            return str(tuple(v.shape))
        return str(v)

    def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
        if isinstance(obj, dict):
            for k, vv in obj.items():
                kk = f"{prefix}.{k}" if prefix else str(k)
                _flatten(kk, vv, out)
        elif isinstance(obj, (list, tuple)):
            out[prefix] = _scalarize(obj)
        else:
            out[prefix] = _scalarize(obj)

    plain = cfg_to_plain_dict(cfg)
    out: Dict[str, Any] = {}
    _flatten("", plain, out)
    return out


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if float(denom) > 0 else 0.0


def _classification_stats_from_counts(
    tp: float, tn: float, fp: float, fn: float
) -> Dict[str, float]:
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    miou = _safe_div(tp, tp + fp + fn)
    return {
        "acc": acc,
        "f1": f1,
        "specificity": specificity,
        "recall": recall,
        "miou": miou,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def init_accelerator(cfg: Any, run_dir: Path) -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(_cfg_get(cfg, "train.grad_accum_steps", 1)),
        mixed_precision=str(_cfg_get(cfg, "accelerate.mixed_precision", "no")),
        log_with="tensorboard",
        project_dir=str(run_dir),
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        import yaml

        (run_dir / "config_resolved.yml").write_text(
            yaml.safe_dump(cfg_to_plain_dict(cfg), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    accelerator.wait_for_everyone()
    accelerator.init_trackers(
        project_name=str(
            _cfg_get(cfg, "logging.project_name", "seed_guided_multitask")
        ),
        config=_flatten_tb_config(cfg),
    )
    return accelerator


def build_model_from_cfg(cfg: Any) -> SeedGuidedMultiTaskModel:
    in_ch = int(len(_cfg_get(cfg, "data.use_modalities", [0])))
    out_ch = int(_cfg_get(cfg, "model.out_ch", 1))

    return SeedGuidedMultiTaskModel(
        in_channel=in_ch,
        out_channel=out_ch,
        detector_base_ch=int(_cfg_get(cfg, "model.detector_base_ch", 32)),
        detector_feat_dim=int(_cfg_get(cfg, "model.detector_feat_dim", 96)),
        detector_stride_xy=int(_cfg_get(cfg, "model.detector_stride_xy", 4)),
        detector_mamba_d_state=int(_cfg_get(cfg, "model.detector_mamba_d_state", 16)),
        detector_mamba_d_conv=int(_cfg_get(cfg, "model.detector_mamba_d_conv", 4)),
        detector_mamba_expand=int(_cfg_get(cfg, "model.detector_mamba_expand", 2)),
        detector_mamba_nslices=int(_cfg_get(cfg, "model.detector_mamba_nslices", 8)),
        detector_softargmax_beta=float(
            _cfg_get(cfg, "model.detector_softargmax_beta", 10.0)
        ),
        roi_size_fullres=tuple(
            int(v) for v in _cfg_get(cfg, "model.roi_size_fullres", [64, 64, 32])
        ),
        seed_region_radius_fullres=tuple(
            int(v)
            for v in _cfg_get(cfg, "model.seed_region_radius_fullres", [16, 16, 8])
        ),
        backbone_base_ch=int(_cfg_get(cfg, "model.backbone_base_ch", 32)),
        seed_prior_sigma_scale=float(
            _cfg_get(cfg, "model.seed_prior_sigma_scale", 0.20)
        ),
        seg_loss_weight=float(_cfg_get(cfg, "loss.seg_weight_stage2", 1.0)),
        cls_loss_weight=float(_cfg_get(cfg, "loss.cls_weight_stage3", 1.0)),
        detector_loss_weight=float(_cfg_get(cfg, "loss.detector_weight_stage1", 1.0)),
    )


def build_optimizer(cfg: Any, model: torch.nn.Module) -> AdamW:
    lr_detector = float(
        _cfg_get(cfg, "train.optimizer.lr_detector", _cfg_get(cfg, "train.lr", 1e-4))
    )
    lr_backbone = float(_cfg_get(cfg, "train.optimizer.lr_backbone", lr_detector))
    lr_cls = float(_cfg_get(cfg, "train.optimizer.lr_cls", lr_detector))
    wd = float(_cfg_get(cfg, "train.weight_decay", 1e-4))

    detector_params = []
    backbone_params = []
    cls_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("detector"):
            detector_params.append(p)
        elif name.startswith("backbone.cls_head"):
            cls_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if detector_params:
        param_groups.append(
            {
                "params": detector_params,
                "lr": lr_detector,
                "weight_decay": wd,
                "group_name": "detector",
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": wd,
                "group_name": "backbone",
            }
        )
    if cls_params:
        param_groups.append(
            {
                "params": cls_params,
                "lr": lr_cls,
                "weight_decay": wd,
                "group_name": "cls",
            }
        )
    return AdamW(param_groups)


def build_scheduler(cfg: Any, optimizer: AdamW):
    sched_cfg = _cfg_get(cfg, "train.scheduler", None)

    if isinstance(sched_cfg, str):
        sched_name = sched_cfg.lower()
        min_lr = float(_cfg_get(cfg, "train.min_lr", 1e-6))
    else:
        sched_name = str(_cfg_get(cfg, "train.scheduler.name", "cosine")).lower()
        min_lr = float(_cfg_get(cfg, "train.scheduler.min_lr", 1e-6))

    if sched_name != "cosine":
        return None

    return CosineAnnealingLR(
        optimizer,
        T_max=int(_cfg_get(cfg, "train.epochs", 100)),
        eta_min=min_lr,
    )


def _set_requires_grad_for_module(mod: Optional[torch.nn.Module], flag: bool) -> None:
    if mod is None:
        return
    for p in mod.parameters():
        p.requires_grad = flag


def _compute_stage(epoch: int, cfg: Any) -> str:
    stage1_epochs = int(_cfg_get(cfg, "train.stage1_epochs", 20))
    stage2_epochs = int(_cfg_get(cfg, "train.stage2_epochs", 60))
    if epoch < stage1_epochs:
        return "stage1"
    if epoch < stage1_epochs + stage2_epochs:
        return "stage2"
    return "stage3"


def apply_stage_policy(
    model: torch.nn.Module, accelerator: Accelerator, stage: str
) -> Dict[str, Any]:
    raw = accelerator.unwrap_model(model)

    for p in raw.parameters():
        p.requires_grad = False

    if stage == "stage1":
        _set_requires_grad_for_module(raw.detector, True)
        _set_requires_grad_for_module(raw.backbone, False)
    elif stage == "stage2":
        _set_requires_grad_for_module(raw.detector, True)
        _set_requires_grad_for_module(raw.backbone, True)
        _set_requires_grad_for_module(getattr(raw.backbone, "cls_head", None), False)
    else:
        _set_requires_grad_for_module(raw.detector, False)
        _set_requires_grad_for_module(raw.backbone, False)
        _set_requires_grad_for_module(getattr(raw.backbone, "cls_head", None), True)

    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in raw.parameters() if not p.requires_grad)
    return {"stage": stage, "trainable_params": trainable, "frozen_params": frozen}


def _update_optimizer_lrs(optimizer: AdamW, stage: str, cfg: Any) -> None:
    base_detector = float(
        _cfg_get(cfg, "train.optimizer.lr_detector", _cfg_get(cfg, "train.lr", 1e-4))
    )
    base_backbone = float(_cfg_get(cfg, "train.optimizer.lr_backbone", base_detector))
    base_cls = float(_cfg_get(cfg, "train.optimizer.lr_cls", base_detector))

    if stage == "stage1":
        mult = {"detector": 1.0, "backbone": 0.0, "cls": 0.0}
    elif stage == "stage2":
        mult = {
            "detector": float(
                _cfg_get(cfg, "train.optimizer.stage2_detector_mult", 1.0)
            ),
            "backbone": float(
                _cfg_get(cfg, "train.optimizer.stage2_backbone_mult", 1.0)
            ),
            "cls": 0.0,
        }
    else:
        mult = {"detector": 0.0, "backbone": 0.0, "cls": 1.0}

    for group in optimizer.param_groups:
        name = group.get("group_name", "")
        if name == "detector":
            group["lr"] = base_detector * mult["detector"]
        elif name == "backbone":
            group["lr"] = base_backbone * mult["backbone"]
        elif name == "cls":
            group["lr"] = base_cls * mult["cls"]


def _reduce_scalar(
    accelerator: Accelerator, value: float, device: torch.device
) -> float:
    t = torch.tensor(value, device=device, dtype=torch.float32)
    return float(accelerator.reduce(t, reduction="sum").item())


def _prepare_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "image" not in batch:
        raise KeyError("Batch is missing key 'image'")
    if "seg_label" not in batch:
        raise KeyError("Batch is missing key 'seg_label'")
    if "class_label" not in batch:
        raise KeyError("Batch is missing key 'class_label'")

    x = batch["image"].to(device, non_blocking=True)
    y_seg = batch["seg_label"].to(device, non_blocking=True)
    y_cls = batch["class_label"].to(device, non_blocking=True).float()

    out_ch = int(_cfg_get(cfg, "model.out_ch", 1))
    take_first = bool(_cfg_get(cfg, "data.label_take_first_channel", True))
    num_mod = int(x.shape[1])

    # 新增显式开关：是否要求模态一一对应分割
    expect_modal_paired = bool(
        _cfg_get(cfg, "data.expect_modal_paired_seg", out_ch == num_mod)
    )

    if expect_modal_paired:
        if y_seg.dim() != 5:
            raise ValueError(
                f"Expected seg_label to be 5D [B,C,W,H,Z], got shape {tuple(y_seg.shape)}"
            )
        if y_seg.shape[1] != num_mod:
            raise ValueError(
                f"Modal-paired segmentation requires seg_label channels == num_mod ({num_mod}), "
                f"but got y_seg.shape[1]={y_seg.shape[1]}"
            )
        if out_ch != num_mod:
            raise ValueError(
                f"Modal-paired segmentation requires model.out_ch == num_mod ({num_mod}), "
                f"but got out_ch={out_ch}. Current config/training would collapse labels."
            )
        y_seg = y_seg.float()
    else:
        y_seg = select_label_channel(
            y_seg,
            out_ch=out_ch,
            take_first=take_first,
        ).float()

    if y_cls.dim() == 0:
        y_cls = y_cls.unsqueeze(0).unsqueeze(1)
    elif y_cls.dim() == 1:
        y_cls = y_cls.unsqueeze(1)

    return x, y_seg, y_cls

def _samplewise_bce_dice(
    seg_logits: torch.Tensor, seg_target: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    返回每个样本的 segmentation loss，shape: (B,)
    """
    bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target, reduction="none")
    bce = bce.mean(dim=(1, 2, 3, 4))

    prob = torch.sigmoid(seg_logits)
    inter = (prob * seg_target).sum(dim=(1, 2, 3, 4))
    union = prob.sum(dim=(1, 2, 3, 4)) + seg_target.sum(dim=(1, 2, 3, 4))
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return bce + dice


def _compute_roi_coverage_from_targets(
    seg_roi_target: torch.Tensor, seg_gt_full: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    seg_roi_target: 由当前 detector crop 得到的 ROI GT, shape (B,C,Wr,Hr,Zr)
    seg_gt_full:    full GT, shape (B,C,W,H,Z)
    返回: (B,) coverage
    """
    roi_sum = seg_roi_target.sum(dim=(1, 2, 3, 4))
    gt_sum = seg_gt_full.sum(dim=(1, 2, 3, 4))
    coverage = roi_sum / (gt_sum + eps)
    empty_mask = gt_sum <= eps
    coverage = torch.where(empty_mask, torch.ones_like(coverage), coverage)
    return coverage.clamp(0.0, 1.0)


def _coverage_seg_weights(coverage: torch.Tensor, cfg: Any) -> torch.Tensor:
    tau = float(_cfg_get(cfg, "coverage.threshold", 0.9))
    hard_reject_below = float(_cfg_get(cfg, "coverage.hard_reject_below", 0.5))
    soft_weighting = bool(_cfg_get(cfg, "coverage.soft_weighting", True))

    if not soft_weighting:
        return (coverage >= tau).float()

    weights = torch.clamp(coverage / max(tau, 1e-6), 0.0, 1.0)
    weights = torch.where(
        coverage < hard_reject_below, torch.zeros_like(weights), weights
    )
    return weights


def _coverage_penalty(coverage: torch.Tensor, cfg: Any) -> torch.Tensor:
    tau = float(_cfg_get(cfg, "coverage.threshold", 0.9))
    penalty = torch.clamp(tau - coverage, min=0.0)
    return penalty.mean()


def _compute_seg_metrics(
    seg_logits: torch.Tensor,
    y_seg: torch.Tensor,
    threshold_or_cfg: Any,
) -> Dict[str, float]:
    if isinstance(threshold_or_cfg, (int, float)):
        threshold = float(threshold_or_cfg)
    else:
        threshold = float(
            _cfg_get(threshold_or_cfg, "train.segmentation.threshold", 0.5)
        )

    seg_prob = torch.sigmoid(seg_logits)
    seg_pred = (seg_prob >= threshold).float()

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")

    dice_metric(y_pred=seg_pred, y=y_seg)
    iou_metric(y_pred=seg_pred, y=y_seg)
    dice_val = float(dice_metric.aggregate().detach().item())
    iou_val = float(iou_metric.aggregate().detach().item())
    dice_metric.reset()
    iou_metric.reset()

    pred_has = seg_pred.sum(dim=(1, 2, 3, 4)) > 0
    gt_has = y_seg.sum(dim=(1, 2, 3, 4)) > 0
    valid = pred_has & gt_has

    if valid.any():
        hd = compute_hausdorff_distance(
            y_pred=seg_pred[valid],
            y=y_seg[valid],
            include_background=True,
            percentile=95,
        )
        hd = hd[torch.isfinite(hd)]
        if hd.numel() > 0:
            hd95 = float(hd.mean().item())
            hd95_cnt = 1.0
        else:
            hd95 = 0.0
            hd95_cnt = 0.0
    else:
        hd95 = 0.0
        hd95_cnt = 0.0

    return {"dice": dice_val, "miou": iou_val, "hd95": hd95, "hd95_cnt": hd95_cnt}


def _compute_cls_counts(
    cls_logits: torch.Tensor,
    y_cls: torch.Tensor,
    threshold_or_cfg: Any,
) -> Dict[str, float]:
    if isinstance(threshold_or_cfg, (int, float)):
        threshold = float(threshold_or_cfg)
    else:
        threshold = float(
            _cfg_get(threshold_or_cfg, "train.classification.threshold", 0.5)
        )

    pred = (torch.sigmoid(cls_logits) >= threshold).long().view(-1)
    target = y_cls.long().view(-1)

    tp = float(((pred == 1) & (target == 1)).sum().item())
    tn = float(((pred == 0) & (target == 0)).sum().item())
    fp = float(((pred == 1) & (target == 0)).sum().item())
    fn = float(((pred == 0) & (target == 1)).sum().item())

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _choose_vis_slice(
    y_seg: np.ndarray,
    seg_pred: Optional[np.ndarray] = None,
    roi_info: Optional[Dict[str, Any]] = None,
) -> int:
    """
    可视化切片选择策略：
    1. 若提供 roi_info，则优先选择 ROI 中心切片
    2. 否则退回到 GT 最大切片
    3. 若 GT 为空，再看 pred
    4. 最后取中间切片
    """
    if roi_info is not None:
        z1 = int(roi_info["z1"])
        z2 = int(roi_info["z2"])
        if z2 > z1:
            return int((z1 + z2 - 1) // 2)

    gt_sum = y_seg.sum(axis=(0, 1)) if y_seg.ndim == 3 else y_seg.sum(axis=0)
    if float(gt_sum.max()) > 0:
        return int(np.argmax(gt_sum))

    if seg_pred is not None:
        pred_sum = (
            seg_pred.sum(axis=(0, 1)) if seg_pred.ndim == 3 else seg_pred.sum(axis=0)
        )
        if float(pred_sum.max()) > 0:
            return int(np.argmax(pred_sum))

    return int(y_seg.shape[-1] // 2)


def _norm01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _overlay_mask(
    gray: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float]
) -> np.ndarray:
    gray = _norm01(gray)
    rgb = np.stack([gray, gray, gray], axis=-1)
    alpha = 0.45 * (mask > 0).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = rgb * (1.0 - alpha[..., None]) + color_arr * alpha[..., None]
    return np.clip(out, 0.0, 1.0)


def _draw_roi_box(
    ax, x1: int, x2: int, y1: int, y2: int, color: str = "yellow"
) -> None:
    import matplotlib.patches as patches

    rect = patches.Rectangle(
        (y1, x1), y2 - y1, x2 - x1, linewidth=2.0, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)


def _draw_seed_point(ax, x: float, y: float, color: str = "red") -> None:
    ax.scatter([y], [x], c=color, s=35, marker="o")


def _build_fullres_roi_mask_from_crop_infos(
    crop_infos: Any,
    fullres_shape: Tuple[int, int, int],
    batch_size: int,
    out_channels: int,
    device: torch.device,
    dtype: torch.dtype,
    paired_mode: bool,
) -> torch.Tensor:
    """
    构建 full-resolution ROI mask，供可视化阶段做额外保险。
    返回:
        (B, Cout, W, H, Z)
    """
    W, H, Z = [int(v) for v in fullres_shape]
    roi_mask = torch.zeros(
        (batch_size, out_channels, W, H, Z),
        device=device,
        dtype=dtype,
    )

    if paired_mode:
        for b in range(batch_size):
            for c in range(out_channels):
                info = crop_infos[b][c]
                x1, x2 = info["x1"], info["x2"]
                y1, y2 = info["y1"], info["y2"]
                z1, z2 = info["z1"], info["z2"]
                roi_mask[b, c : c + 1, x1:x2, y1:y2, z1:z2] = 1.0
    else:
        for b in range(batch_size):
            info = crop_infos[b]
            x1, x2 = info["x1"], info["x2"]
            y1, y2 = info["y1"], info["y2"]
            z1, z2 = info["z1"], info["z2"]
            roi_mask[b, :, x1:x2, y1:y2, z1:z2] = 1.0

    return roi_mask


def _compute_stage1_score(stats: Dict[str, float], cfg: Any) -> float:
    a = float(_cfg_get(cfg, "monitor.stage1_weights.coverage", 1.0))
    b = float(_cfg_get(cfg, "monitor.stage1_weights.seed_in", 0.5))
    c = float(_cfg_get(cfg, "monitor.stage1_weights.center_l1", 0.1))
    return (
        a * stats["roi_coverage"]
        + b * stats["seed_in_lesion_rate"]
        - c * stats["center_l1"]
    )


def _compute_stage2_score(stats: Dict[str, float], cfg: Any) -> float:
    a = float(_cfg_get(cfg, "monitor.stage2_weights.seg_dice", 0.8))
    b = float(_cfg_get(cfg, "monitor.stage2_weights.coverage", 0.2))
    return a * stats["seg_dice"] + b * stats["roi_coverage"]


def _compute_stage3_score(stats: Dict[str, float], cfg: Any) -> float:
    monitor = str(_cfg_get(cfg, "monitor.stage3", "cls_f1")).lower()
    if monitor == "cls_acc":
        return float(stats["cls_acc"])
    return float(stats["cls_f1"])


def _select_monitor_score(stage: str, val_stats: Dict[str, float], cfg: Any) -> float:
    if stage == "stage1":
        return _compute_stage1_score(val_stats, cfg)
    if stage == "stage2":
        return _compute_stage2_score(val_stats, cfg)
    return _compute_stage3_score(val_stats, cfg)


def _cleanup_old_visualizations(
    run_dir: Path,
    keep_last_n: int = 5,
) -> None:
    """
    只保留最新 keep_last_n 个 epoch 的可视化目录。
    目录结构假设为:
        run_dir / "visuals" / "epoch_0000" / ...
    """
    vis_root = run_dir / "visuals"
    if not vis_root.exists():
        return

    epoch_dirs = []
    for p in vis_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("epoch_"):
            continue
        try:
            epoch_id = int(name.split("_")[1])
        except Exception:
            continue
        epoch_dirs.append((epoch_id, p))

    epoch_dirs.sort(key=lambda x: x[0])

    if len(epoch_dirs) <= keep_last_n:
        return

    to_delete = epoch_dirs[:-keep_last_n]
    for _, p in to_delete:
        try:
            import shutil

            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def _default_epoch_stats() -> Dict[str, float]:
    return {
        "loss_total": 0.0,
        "loss_detector": 0.0,
        "loss_seg": 0.0,
        "loss_cls": 0.0,
        "loss_cov_penalty": 0.0,
        "seg_dice": 0.0,
        "seg_hd95": 0.0,
        "seg_miou": 0.0,
        "seg_hd95_cnt": 0.0,
        "roi_coverage": 0.0,
        "valid_ratio": 0.0,
        "center_l1": 0.0,
        "seed_in_lesion_rate": 0.0,
        "peakness": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "n_steps": 0.0,
    }


def _merge_epoch_stats(
    accelerator: Accelerator, running: Dict[str, float], device: torch.device
) -> Dict[str, float]:
    reduced = {k: _reduce_scalar(accelerator, v, device) for k, v in running.items()}
    n_steps = max(reduced["n_steps"], 1.0)
    cls_stats = _classification_stats_from_counts(
        reduced["tp"], reduced["tn"], reduced["fp"], reduced["fn"]
    )
    return {
        "loss_total": reduced["loss_total"] / n_steps,
        "loss_detector": reduced["loss_detector"] / n_steps,
        "loss_seg": reduced["loss_seg"] / n_steps,
        "loss_cls": reduced["loss_cls"] / n_steps,
        "loss_cov_penalty": reduced["loss_cov_penalty"] / n_steps,
        "seg_dice": reduced["seg_dice"] / n_steps,
        "seg_hd95": reduced["seg_hd95"] / max(reduced["seg_hd95_cnt"], 1.0),
        "seg_miou": reduced["seg_miou"] / n_steps,
        "roi_coverage": reduced["roi_coverage"] / n_steps,
        "valid_ratio": reduced["valid_ratio"] / n_steps,
        "center_l1": reduced["center_l1"] / n_steps,
        "seed_in_lesion_rate": reduced["seed_in_lesion_rate"] / n_steps,
        "peakness": reduced["peakness"] / n_steps,
        "cls_acc": cls_stats["acc"],
        "cls_f1": cls_stats["f1"],
        "cls_specificity": cls_stats["specificity"],
        "cls_recall": cls_stats["recall"],
        "cls_miou": cls_stats["miou"],
    }


def save_epoch_visualizations(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Any,
    epoch: int,
    split_name: str,
    run_dir: Path,
    stage: str,
) -> None:
    if not accelerator.is_main_process:
        return
    if not bool(_cfg_get(cfg, "visualization.enable", True)):
        return
    if epoch % int(_cfg_get(cfg, "visualization.every_n_epochs", 1)) != 0:
        return

    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()
    device = accelerator.device

    x, y_seg, y_cls = _prepare_batch(batch, device, cfg)
    max_samples = int(_cfg_get(cfg, "visualization.max_samples_per_split", 5))
    keep_last_n = int(_cfg_get(cfg, "visualization.keep_last_n_epochs", 5))

    x = x[:max_samples]
    y_seg = y_seg[:max_samples]
    y_cls = y_cls[:max_samples]

    with torch.no_grad():
        out = raw_model(image_full=x, seg_gt=y_seg, cls_gt=y_cls)

    crop_infos = out["crop_infos"]
    image_roi = out["image_roi"]
    seed_coord_full = out["seed_coord_full"]

    paired_mode = isinstance(crop_infos[0], list)
    B, Cin, W, H, Z = x.shape
    Cout = int(out["seg_logits_full"].shape[1])

    seg_prob_full = torch.sigmoid(out["seg_logits_full"])
    seg_pred_full = (
        seg_prob_full >= float(_cfg_get(cfg, "train.segmentation.threshold", 0.5))
    ).float()

    roi_mask_full = _build_fullres_roi_mask_from_crop_infos(
        crop_infos=crop_infos,
        fullres_shape=(W, H, Z),
        batch_size=B,
        out_channels=Cout,
        device=seg_pred_full.device,
        dtype=seg_pred_full.dtype,
        paired_mode=paired_mode,
    )
    seg_pred_full = seg_pred_full * roi_mask_full

    modalities = list(_cfg_get(cfg, "data.use_modalities", []))
    num_mod = int(x.shape[1])
    if len(modalities) != num_mod:
        modalities = [f"mod_{i}" for i in range(num_mod)]

    # paired mode only in your current project contract
    if int(y_seg.shape[1]) != num_mod:
        raise RuntimeError(
            f"Expected modal-paired GT channels == num_mod ({num_mod}), got {int(y_seg.shape[1])}"
        )
    if Cout != num_mod:
        raise RuntimeError(
            f"Expected model output channels == num_mod ({num_mod}), got Cout={Cout}"
        )

    vis_dir = run_dir / "visuals" / f"epoch_{epoch:04d}" / split_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    for b in range(x.shape[0]):
        x_np = x[b].detach().float().cpu().numpy()               # [M,W,H,Z]
        y_np = y_seg[b].detach().float().cpu().numpy()           # [M,W,H,Z]
        pred_np = seg_pred_full[b].detach().float().cpu().numpy()  # [M,W,H,Z]
        sample_seed = seed_coord_full[b].detach().float().cpu().numpy()

        n_rows = num_mod
        n_cols = 5
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 4.0 * n_rows),
            dpi=150,
            squeeze=False,
        )

        for m in range(num_mod):
            gt_vol_m = y_np[m]         # [W,H,Z]
            pred_vol_m = pred_np[m]    # [W,H,Z]

            if paired_mode:
                info = crop_infos[b][m]
                sc = sample_seed[m] if sample_seed.ndim == 2 else sample_seed
            else:
                info = crop_infos[b]
                sc = sample_seed

            # -----------------------------------------
            # 关键修正：每个模态单独选择自己的 z_idx
            # -----------------------------------------
            z_idx_m = _choose_vis_slice(
                gt_vol_m,
                pred_vol_m,
                roi_info=info,
            )

            raw_slice = x_np[m, :, :, z_idx_m]
            gt_slice = gt_vol_m[:, :, z_idx_m]
            pred_slice = pred_vol_m[:, :, z_idx_m]

            row = axes[m]

            row[0].imshow(_overlay_mask(raw_slice, gt_slice, (0.0, 1.0, 0.0)))
            _draw_seed_point(row[0], float(sc[0]), float(sc[1]), color="red")
            row[0].set_title(f"{modalities[m]} | seed + GT\nz={z_idx_m}")
            row[0].axis("off")

            row[1].imshow(_overlay_mask(raw_slice, gt_slice, (0.0, 1.0, 0.0)))
            _draw_roi_box(
                row[1],
                info["x1"],
                info["x2"],
                info["y1"],
                info["y2"],
                color="yellow",
            )
            row[1].set_title(
                f"{modalities[m]} | ROI box + GT\n"
                f"z={z_idx_m}, roi_z=[{info['z1']},{info['z2']})"
            )
            row[1].axis("off")

            roi_mod = image_roi[b, m].detach().float().cpu().numpy()
            if info["z2"] > info["z1"]:
                roi_z = int(np.clip(z_idx_m - int(info["z1"]), 0, roi_mod.shape[-1] - 1))
            else:
                roi_z = roi_mod.shape[-1] // 2

            row[2].imshow(_norm01(roi_mod[:, :, roi_z]), cmap="gray")
            row[2].set_title(f"{modalities[m]} | ROI input\nroi_z={roi_z}")
            row[2].axis("off")

            row[3].imshow(_overlay_mask(raw_slice, pred_slice, (1.0, 0.0, 0.0)))
            row[3].set_title(f"{modalities[m]} | pred + raw\nz={z_idx_m}")
            row[3].axis("off")

            row[4].imshow(_overlay_mask(raw_slice, gt_slice, (0.0, 1.0, 0.0)))
            row[4].set_title(f"{modalities[m]} | GT + raw\nz={z_idx_m}")
            row[4].axis("off")

        fig.tight_layout()
        fig.savefig(vis_dir / f"sample_{b:03d}_{stage}.png", bbox_inches="tight")
        plt.close(fig)

    _cleanup_old_visualizations(run_dir=run_dir, keep_last_n=keep_last_n)

    del out, seg_prob_full, seg_pred_full, roi_mask_full
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def run_one_epoch(
    *,
    split_name: str,
    is_train: bool,
    accelerator: Accelerator,
    model: nn.Module,
    data_loader,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    stage_cfg: Optional[Dict[str, Any]] = None,
    stage: Optional[Any] = None,
    cfg: Any,
    run_dir: Path,
) -> Dict[str, float]:
    from tqdm import tqdm

    if stage_cfg is None and stage is not None:
        if isinstance(stage, dict):
            stage_cfg = stage
        else:
            stage_cfg = {"stage": str(stage)}

    if stage_cfg is None:
        raise ValueError("run_one_epoch requires either stage_cfg=... or stage=...")

    stage_name = str(stage_cfg.get("stage", "unknown"))
    device = accelerator.device
    grad_clip = float(_cfg_get(cfg, "train.grad_clip", 0.0))
    log_interval = int(_cfg_get(cfg, "logging.log_interval", 20))

    raw_model = accelerator.unwrap_model(model)

    if is_train:
        model.train()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    running = _default_epoch_stats()
    vis_batch_cpu = None

    pbar = tqdm(
        enumerate(data_loader),
        total=len(data_loader) if hasattr(data_loader, "__len__") else None,
        disable=not accelerator.is_main_process,
        desc=f"{split_name.capitalize()} Epoch {epoch} [{stage_name}]",
        dynamic_ncols=True,
    )

    def _detach_batch_to_cpu(batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu()
            else:
                out[k] = v
        return out

    for step, batch in pbar:
        if batch is None:
            continue
        

        if vis_batch_cpu is None:
            vis_batch_cpu = _detach_batch_to_cpu(batch)

        x, y_seg, y_cls = _prepare_batch(batch, device, cfg)

        seg_threshold = float(_cfg_get(cfg, "train.segmentation.threshold", 0.5))
        cls_threshold = float(_cfg_get(cfg, "train.classification.threshold", 0.5))

        if stage_name == "stage1":
            # detector warmup
            def _forward_stage1():
                out = model(image_full=x, seg_gt=y_seg, cls_gt=None)
                det_loss_dict = raw_model.detector.compute_loss(
                    out=out["detector_out"],
                    seg_gt=y_seg,
                    modal_consistency_weight=float(
                        _cfg_get(cfg, "loss.detector_modal_consistency_weight", 0.10)
                    ),
                    coord_weight=float(
                        _cfg_get(cfg, "loss.detector_coord_weight", 0.25)
                    ),
                    sharpness_weight=float(
                        _cfg_get(cfg, "loss.detector_sharpness_weight", 0.05)
                    ),
                    center_blob_weight=float(
                        _cfg_get(cfg, "loss.detector_center_blob_weight", 0.10)
                    ),
                )
                loss_detector = det_loss_dict["loss_total"] * float(
                    _cfg_get(cfg, "loss.detector_weight_stage1", 1.0)
                )
                det_metrics = raw_model.detector.compute_metrics(
                    out["detector_out"],
                    y_seg,
                    roi_size=tuple(
                        int(v)
                        for v in _cfg_get(cfg, "model.roi_size_fullres", [64, 64, 32])
                    ),
                )
                return out, loss_detector, det_loss_dict, det_metrics

            if is_train:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        out, loss, det_loss_dict, det_metrics = _forward_stage1()
                    accelerator.backward(loss)

                    if accelerator.sync_gradients and grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                    if optimizer is not None:
                        optimizer.step()
                    if optimizer is not None and accelerator.sync_gradients:
                        optimizer.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    with accelerator.autocast():
                        out, loss, det_loss_dict, det_metrics = _forward_stage1()

            running["loss_total"] += float(loss.detach().item())
            running["loss_detector"] += float(
                det_loss_dict["loss_total"].detach().item()
            )
            running["roi_coverage"] += float(det_metrics["roi_coverage"])
            running["center_l1"] += float(det_metrics["center_l1"])
            running["seed_in_lesion_rate"] += float(det_metrics["seed_in_lesion_rate"])
            running["peakness"] += float(det_metrics["peakness"])

        elif stage_name == "stage2":
            # coverage-aware joint segmentation
            def _forward_stage2():
                out = model(image_full=x, seg_gt=y_seg, cls_gt=None)

                det_loss_dict = raw_model.detector.compute_loss(
                    out=out["detector_out"],
                    seg_gt=y_seg,
                    modal_consistency_weight=float(
                        _cfg_get(cfg, "loss.detector_modal_consistency_weight", 0.10)
                    ),
                    coord_weight=float(
                        _cfg_get(cfg, "loss.detector_coord_weight", 0.25)
                    ),
                    sharpness_weight=float(
                        _cfg_get(cfg, "loss.detector_sharpness_weight", 0.05)
                    ),
                    center_blob_weight=float(
                        _cfg_get(cfg, "loss.detector_center_blob_weight", 0.10)
                    ),
                )
                det_metrics = raw_model.detector.compute_metrics(
                    out["detector_out"],
                    y_seg,
                    roi_size=tuple(
                        int(v)
                        for v in _cfg_get(cfg, "model.roi_size_fullres", [64, 64, 32])
                    ),
                )

                seg_roi_target = out["seg_roi_target"]
                seg_sample_loss = _samplewise_bce_dice(
                    out["seg_logits_roi"], seg_roi_target
                )
                coverage = _compute_roi_coverage_from_targets(seg_roi_target, y_seg)
                seg_weights = _coverage_seg_weights(coverage, cfg)
                valid_ratio = float((seg_weights > 0).float().mean().item())

                weighted_seg_loss = (seg_sample_loss * seg_weights).sum() / (
                    seg_weights.sum().clamp_min(1e-6)
                )
                cov_penalty = _coverage_penalty(coverage, cfg)

                total_loss = (
                    float(_cfg_get(cfg, "loss.detector_weight_stage2", 0.5))
                    * det_loss_dict["loss_total"]
                    + float(_cfg_get(cfg, "loss.seg_weight_stage2", 1.0))
                    * weighted_seg_loss
                    + float(_cfg_get(cfg, "loss.coverage_penalty_weight_stage2", 0.5))
                    * cov_penalty
                )

                seg_metrics = _compute_seg_metrics(
                    out["seg_logits_full"].detach(),
                    y_seg.detach(),
                    seg_threshold,
                )
                cls_counts = _compute_cls_counts(
                    out["cls_logits"].detach(),
                    y_cls.detach(),
                    cls_threshold,
                )

                return (
                    out,
                    total_loss,
                    det_loss_dict,
                    det_metrics,
                    weighted_seg_loss,
                    cov_penalty,
                    coverage,
                    valid_ratio,
                    seg_metrics,
                    cls_counts,
                )

            if is_train:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        (
                            out,
                            loss,
                            det_loss_dict,
                            det_metrics,
                            weighted_seg_loss,
                            cov_penalty,
                            coverage,
                            valid_ratio,
                            seg_metrics,
                            cls_counts,
                        ) = _forward_stage2()

                    accelerator.backward(loss)

                    if accelerator.sync_gradients and grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                    if optimizer is not None:
                        optimizer.step()
                    if optimizer is not None and accelerator.sync_gradients:
                        optimizer.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    with accelerator.autocast():
                        (
                            out,
                            loss,
                            det_loss_dict,
                            det_metrics,
                            weighted_seg_loss,
                            cov_penalty,
                            coverage,
                            valid_ratio,
                            seg_metrics,
                            cls_counts,
                        ) = _forward_stage2()

            running["loss_total"] += float(loss.detach().item())
            running["loss_detector"] += float(
                det_loss_dict["loss_total"].detach().item()
            )
            running["loss_seg"] += float(weighted_seg_loss.detach().item())
            running["loss_cov_penalty"] += float(cov_penalty.detach().item())
            running["roi_coverage"] += float(coverage.mean().detach().item())
            running["valid_ratio"] += float(valid_ratio)
            running["center_l1"] += float(det_metrics["center_l1"])
            running["seed_in_lesion_rate"] += float(det_metrics["seed_in_lesion_rate"])
            running["peakness"] += float(det_metrics["peakness"])
            running["seg_dice"] += float(seg_metrics["dice"])
            running["seg_hd95"] += float(seg_metrics["hd95"])
            running["seg_hd95_cnt"] += float(seg_metrics["hd95_cnt"])
            running["seg_miou"] += float(seg_metrics["miou"])
            running["tp"] += cls_counts["tp"]
            running["tn"] += cls_counts["tn"]
            running["fp"] += cls_counts["fp"]
            running["fn"] += cls_counts["fn"]

        else:
            # stage3: classifier fine-tuning
            def _forward_stage3():
                out = model(image_full=x, seg_gt=None, cls_gt=y_cls)
                loss_cls = F.binary_cross_entropy_with_logits(
                    out["cls_logits"],
                    y_cls.float(),
                ) * float(_cfg_get(cfg, "loss.cls_weight_stage3", 1.0))

                seg_metrics = _compute_seg_metrics(
                    out["seg_logits_full"].detach(),
                    y_seg.detach(),
                    seg_threshold,
                )
                cls_counts = _compute_cls_counts(
                    out["cls_logits"].detach(),
                    y_cls.detach(),
                    cls_threshold,
                )
                return out, loss_cls, seg_metrics, cls_counts

            if is_train:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        out, loss, seg_metrics, cls_counts = _forward_stage3()
                    accelerator.backward(loss)

                    if accelerator.sync_gradients and grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                    if optimizer is not None:
                        optimizer.step()
                    if optimizer is not None and accelerator.sync_gradients:
                        optimizer.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    with accelerator.autocast():
                        out, loss, seg_metrics, cls_counts = _forward_stage3()

            running["loss_total"] += float(loss.detach().item())
            running["loss_cls"] += float(loss.detach().item())
            running["seg_dice"] += float(seg_metrics["dice"])
            running["seg_hd95"] += float(seg_metrics["hd95"])
            running["seg_hd95_cnt"] += float(seg_metrics["hd95_cnt"])
            running["seg_miou"] += float(seg_metrics["miou"])
            running["tp"] += cls_counts["tp"]
            running["tn"] += cls_counts["tn"]
            running["fp"] += cls_counts["fp"]
            running["fn"] += cls_counts["fn"]

        running["n_steps"] += 1.0

        if accelerator.is_main_process and step % log_interval == 0:
            cls_now = _classification_stats_from_counts(
                running["tp"], running["tn"], running["fp"], running["fn"]
            )

            if stage_name == "stage1":
                pbar.set_postfix(
                    loss=f"{running['loss_total'] / max(running['n_steps'], 1.0):.4f}",
                    dice="Warmup",
                    acc="Warmup",
                )
            elif stage_name == "stage2":
                pbar.set_postfix(
                    loss=f"{running['loss_total'] / max(running['n_steps'], 1.0):.4f}",
                    dice=f"{running['seg_dice'] / max(running['n_steps'], 1.0):.4f}",
                    acc="Step2",
                )
            else:
                pbar.set_postfix(
                    loss=f"{running['loss_total'] / max(running['n_steps'], 1.0):.4f}",
                    dice=f"{running['seg_dice'] / max(running['n_steps'], 1.0):.4f}",
                    acc=f"{cls_now['acc']:.4f}",
                )

        del x, y_seg, y_cls
        if torch.cuda.is_available() and step % 10 == 0:
            torch.cuda.empty_cache()

    stats = _merge_epoch_stats(accelerator, running, device)

    accelerator.wait_for_everyone()

    if vis_batch_cpu is not None:
        save_epoch_visualizations(
            accelerator=accelerator,
            model=model,
            batch=vis_batch_cpu,
            cfg=cfg,
            epoch=epoch,
            split_name=split_name,
            run_dir=run_dir,
            stage=stage_name,
        )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.log(
            {
                f"{split_name}/loss_total": stats["loss_total"],
                f"{split_name}/loss_detector": stats["loss_detector"],
                f"{split_name}/loss_seg": stats["loss_seg"],
                f"{split_name}/loss_cls": stats["loss_cls"],
                f"{split_name}/loss_cov_penalty": stats["loss_cov_penalty"],
                f"{split_name}/seg_dice": stats["seg_dice"],
                f"{split_name}/seg_hd95": stats["seg_hd95"],
                f"{split_name}/seg_miou": stats["seg_miou"],
                f"{split_name}/roi_coverage": stats["roi_coverage"],
                f"{split_name}/valid_ratio": stats["valid_ratio"],
                f"{split_name}/center_l1": stats["center_l1"],
                f"{split_name}/seed_in_lesion_rate": stats["seed_in_lesion_rate"],
                f"{split_name}/peakness": stats["peakness"],
                f"{split_name}/cls_acc": stats["cls_acc"],
                f"{split_name}/cls_f1": stats["cls_f1"],
                f"{split_name}/cls_specificity": stats["cls_specificity"],
                f"{split_name}/cls_recall": stats["cls_recall"],
                f"{split_name}/cls_miou": stats["cls_miou"],
                "epoch": epoch,
            },
            step=epoch,
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def _print_epoch_summary(
    accelerator: Accelerator,
    epoch: int,
    total_epochs: int,
    stage: str,
    train_stats: Dict[str, float],
    val_stats: Dict[str, float],
    test_stats: Optional[Dict[str, float]],
    best_score: float,
    policy_info: Dict[str, Any],
    dt: float,
) -> None:
    if not accelerator.is_main_process:
        return

    if stage == "stage1":
        msg = (
            f"Epoch {epoch}/{total_epochs - 1} [{stage}] | trainable={policy_info['trainable_params']:,} "
            f"| train(loss={train_stats['loss_total']:.4f}, Dice=Warmup, Acc=Warmup) "
            f"| val(loss={val_stats['loss_total']:.4f}, Dice=Warmup, Acc=Warmup)"
        )
        if test_stats is not None:
            msg += (
                f" | test(loss={test_stats['loss_total']:.4f}, Dice=Warmup, Acc=Warmup)"
            )

    elif stage == "stage2":
        msg = (
            f"Epoch {epoch}/{total_epochs - 1} [{stage}] | trainable={policy_info['trainable_params']:,} "
            f"| train(loss={train_stats['loss_total']:.4f}, Dice={train_stats['seg_dice']:.4f}, Acc=Step2) "
            f"| val(loss={val_stats['loss_total']:.4f}, Dice={val_stats['seg_dice']:.4f}, Acc=Step2)"
        )
        if test_stats is not None:
            msg += f" | test(loss={test_stats['loss_total']:.4f}, Dice={test_stats['seg_dice']:.4f}, Acc=Step2)"

    else:
        msg = (
            f"Epoch {epoch}/{total_epochs - 1} [{stage}] | trainable={policy_info['trainable_params']:,} "
            f"| train(loss={train_stats['loss_total']:.4f}, Dice={train_stats['seg_dice']:.4f}, Acc={train_stats['cls_acc']:.4f}) "
            f"| val(loss={val_stats['loss_total']:.4f}, Dice={val_stats['seg_dice']:.4f}, Acc={val_stats['cls_acc']:.4f})"
        )
        if test_stats is not None:
            msg += f" | test(loss={test_stats['loss_total']:.4f}, Dice={test_stats['seg_dice']:.4f}, Acc={test_stats['cls_acc']:.4f})"

    msg += f" | best={best_score:.4f} | {dt:.1f}s"
    accelerator.print(msg)


def debug_compare_modal_labels(batch: Dict[str, torch.Tensor]) -> None:
    """
    检查同一 batch 中，不同模态的 GT label 是否实际相同。
    用于判断“GT 可视化看起来一样”到底是源标签一致，还是程序错贴。
    """
    if "seg_label" not in batch:
        raise KeyError("Batch is missing key 'seg_label'")

    y = batch["seg_label"]
    if not torch.is_tensor(y):
        raise TypeError("batch['seg_label'] must be a torch.Tensor")

    if y.dim() != 5:
        raise ValueError(f"Expected seg_label shape [B,C,W,H,Z], got {tuple(y.shape)}")

    B, C, W, H, Z = y.shape
    print(f"[debug_compare_modal_labels] seg_label shape = {tuple(y.shape)}")

    if C < 2:
        print("[debug_compare_modal_labels] Only one channel, nothing to compare.")
        return

    y_cpu = y.detach().cpu()

    for b in range(B):
        print(f"\n[Sample {b}]")
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                a = y_cpu[b, c1]
                b_ = y_cpu[b, c2]

                same_all = torch.equal(a, b_)
                max_abs_diff = (a - b_).abs().max().item()
                sum_a = a.sum().item()
                sum_b = b_.sum().item()
                xor_sum = ((a > 0.5) ^ (b_ > 0.5)).float().sum().item()

                print(
                    f"  ch{c1} vs ch{c2} | "
                    f"same_all={same_all} | "
                    f"max_abs_diff={max_abs_diff:.6f} | "
                    f"sum(ch{c1})={sum_a:.1f} | sum(ch{c2})={sum_b:.1f} | "
                    f"xor_voxels={xor_sum:.1f}"
                )

def train_loop(cfg: Any) -> None:
    run_dir, run_name = prepare_run_dir(cfg)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        txt_log_path = start_txt_logger(run_dir, filename="console.txt")
        print(f"Console log -> {txt_log_path}")

    accelerator = init_accelerator(cfg, run_dir)
    seed = int(_cfg_get(cfg, "train.seed", 42))
    set_seed(seed + accelerator.process_index)

    train_loader, val_loader, test_loader = get_loaders(cfg)
    if val_loader is None:
        raise ValueError("val_loader is None; please ensure validation set exists.")
    if test_loader is None:
        accelerator.print("Warning: test_loader is None; test metrics will be skipped.")

    model = build_model_from_cfg(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    model, optimizer, train_loader, val_loader, test_loader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader, scheduler
        )
    )

    start_epoch, best_score = maybe_resume_from_latest(
        accelerator=accelerator,
        cfg=cfg,
        run_dir=run_dir,
    )

    if accelerator.is_main_process:
        total_params = sum(
            p.numel() for p in accelerator.unwrap_model(model).parameters()
        )
        trainable_params = sum(
            p.numel()
            for p in accelerator.unwrap_model(model).parameters()
            if p.requires_grad
        )
        accelerator.print(
            f"Run: {run_name} | start_epoch={start_epoch} | best_score={best_score} | total={total_params:,} trainable={trainable_params:,}"
        )

    epochs = int(_cfg_get(cfg, "train.epochs", 100))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        stage = _compute_stage(epoch, cfg)
        policy_info = apply_stage_policy(model, accelerator, stage)
        _update_optimizer_lrs(optimizer, stage, cfg)
        accelerator.wait_for_everyone()

        train_stats = run_one_epoch(
            split_name="train",
            is_train=True,
            accelerator=accelerator,
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            stage=stage,
            cfg=cfg,
            run_dir=run_dir,
        )

        if scheduler is not None:
            scheduler.step()

        val_stats = run_one_epoch(
            split_name="val",
            is_train=False,
            accelerator=accelerator,
            model=model,
            data_loader=val_loader,
            optimizer=None,
            epoch=epoch,
            stage=stage,
            cfg=cfg,
            run_dir=run_dir,
        )

        test_stats = None
        if test_loader is not None:
            test_stats = run_one_epoch(
                split_name="test",
                is_train=False,
                accelerator=accelerator,
                model=model,
                data_loader=test_loader,
                optimizer=None,
                epoch=epoch,
                stage=stage,
                cfg=cfg,
                run_dir=run_dir,
            )

        score = _select_monitor_score(stage, val_stats, cfg)

        save_latest_checkpoint(
            accelerator=accelerator,
            cfg=cfg,
            run_dir=run_dir,
            epoch=epoch,
            best_score=best_score,
            current_score=score,
        )

        best_score = save_best_weights_if_improved(
            accelerator=accelerator,
            model=model,
            cfg=cfg,
            run_dir=run_dir,
            epoch=epoch,
            best_score=best_score,
            current_score=score,
        )

        _print_epoch_summary(
            accelerator=accelerator,
            epoch=epoch,
            total_epochs=epochs,
            stage=stage,
            train_stats=train_stats,
            val_stats=val_stats,
            test_stats=test_stats,
            best_score=best_score,
            policy_info=policy_info,
            dt=time.time() - t0,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.end_training()


def export_best_weight_visualizations(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    data_loader,
    cfg: Any,
    run_dir: Path,
    split_name: str,
    best_weights_path: Optional[str] = None,
) -> None:
    """
    加载 best 权重后，对 data_loader 中所有样本逐一导出可视化。

    输出:
        run_dir / "best_visuals" / split_name / "{sample_id}.png"

    文件名优先级:
        1) batch["id"]        -> 本地真实文件夹名，例如 99657
        2) batch["excel_id"]  -> Excel 中原始 ID
        3) fallback -> sample_{global_idx:05d}.png

    注意:
        - 若用于 train_loader，而 train_loader 使用了 drop_last=True，
          则可能漏掉最后不满 batch 的样本。
        - 若要真正导出“训练集全部样本”，请单独构建一个:
              batch_size=1, shuffle=False, drop_last=False
          的 export loader。
    """
    raw_model = accelerator.unwrap_model(model)
    device = accelerator.device

    # -----------------------------
    # 1) load best weights
    # -----------------------------
    if best_weights_path is None or str(best_weights_path).strip() == "":
        best_filename = str(_cfg_get(cfg, "checkpoint.best_filename", "best.pth"))
        best_weights_path = str(run_dir / "checkpoints" / best_filename)

    best_weights_path = str(best_weights_path)
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Best weights not found: {best_weights_path}")

    ckpt = torch.load(best_weights_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)

    if accelerator.is_main_process:
        accelerator.print(
            f"[Export] Loaded best weights from: {best_weights_path} | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    accelerator.wait_for_everyone()
    raw_model.eval()

    # -----------------------------
    # 2) helper: extract sample ids
    # -----------------------------
    def _extract_sample_ids(batch: Dict[str, Any], batch_size: int) -> List[str]:
        def _normalize_to_list(v, target_len: int) -> List[str]:
            if v is None:
                return [f"sample_{i}" for i in range(target_len)]
            if isinstance(v, (list, tuple)):
                out = [str(x) for x in v]
                if len(out) < target_len:
                    out = out + [f"sample_{i}" for i in range(len(out), target_len)]
                return out[:target_len]
            return [str(v)] * target_len

        if "id" in batch:
            ids = _normalize_to_list(batch["id"], batch_size)
            return ids
        if "excel_id" in batch:
            ids = _normalize_to_list(batch["excel_id"], batch_size)
            return ids
        return [f"sample_{i}" for i in range(batch_size)]

    # -----------------------------
    # 3) output dir
    # -----------------------------
    vis_dir = run_dir / "best_visuals" / split_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    modalities = list(_cfg_get(cfg, "data.use_modalities", []))
    global_counter = 0

    # -----------------------------
    # 4) iterate loader
    # -----------------------------
    for batch in data_loader:
        if batch is None:
            continue

        x, y_seg, y_cls = _prepare_batch(batch, device, cfg)

        with torch.no_grad():
            out = raw_model(image_full=x, seg_gt=y_seg, cls_gt=y_cls)

        crop_infos = out["crop_infos"]
        image_roi = out["image_roi"]
        seed_coord_full = out["seed_coord_full"]

        paired_mode = isinstance(crop_infos[0], list)
        B, Cin, W, H, Z = x.shape
        Cout = int(out["seg_logits_full"].shape[1])

        seg_prob_full = torch.sigmoid(out["seg_logits_full"])
        seg_pred_full = (
            seg_prob_full >= float(_cfg_get(cfg, "train.segmentation.threshold", 0.5))
        ).float()

        # 显示侧保险：ROI 外不显示为前景
        roi_mask_full = _build_fullres_roi_mask_from_crop_infos(
            crop_infos=crop_infos,
            fullres_shape=(W, H, Z),
            batch_size=B,
            out_channels=Cout,
            device=seg_pred_full.device,
            dtype=seg_pred_full.dtype,
            paired_mode=paired_mode,
        )
        seg_pred_full = seg_pred_full * roi_mask_full

        num_mod = int(x.shape[1])
        if len(modalities) != num_mod:
            mod_names = [f"mod_{i}" for i in range(num_mod)]
        else:
            mod_names = modalities

        # 当前是否是“模态对应分割”
        modal_paired_vis = (int(y_seg.shape[1]) == num_mod) and (Cout == num_mod)

        batch_ids = _extract_sample_ids(batch, B)

        for b in range(B):
            case_id = str(batch_ids[b]).strip()
            if case_id == "":
                case_id = f"sample_{global_counter:05d}"

            x_np = x[b].detach().float().cpu().numpy()               # [M,W,H,Z]
            y_np = y_seg[b].detach().float().cpu().numpy()           # [Cy,W,H,Z]
            pred_np = seg_pred_full[b].detach().float().cpu().numpy()  # [Cp,W,H,Z]

            gt_union = y_np.max(axis=0)
            pred_union = pred_np.max(axis=0)

            roi_info_for_slice = crop_infos[b][0] if paired_mode else crop_infos[b]
            z_idx = _choose_vis_slice(gt_union, pred_union, roi_info=roi_info_for_slice)

            sample_seed = seed_coord_full[b].detach().float().cpu().numpy()

            n_rows = num_mod
            n_cols = 5
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.0 * n_cols, 4.0 * n_rows),
                dpi=150,
                squeeze=False,
            )

            for m in range(num_mod):
                raw_slice = x_np[m, :, :, z_idx]

                if modal_paired_vis:
                    gt_slice = y_np[m, :, :, z_idx]
                    pred_slice = pred_np[m, :, :, z_idx]
                    gt_tag = f"GT[ch={m}]"
                    pred_tag = f"Pred[ch={m}]"
                else:
                    gt_slice = gt_union[:, :, z_idx]
                    pred_slice = pred_union[:, :, z_idx]
                    gt_tag = "GT[union]"
                    pred_tag = "Pred[union]"

                if paired_mode:
                    info = crop_infos[b][m] if modal_paired_vis else crop_infos[b][0]
                    sc = sample_seed[m] if sample_seed.ndim == 2 else sample_seed
                else:
                    info = crop_infos[b]
                    sc = sample_seed

                row = axes[m]

                row[0].imshow(_overlay_mask(raw_slice, gt_slice, (0.0, 1.0, 0.0)))
                _draw_seed_point(row[0], float(sc[0]), float(sc[1]), color="red")
                row[0].set_title(f"{mod_names[m]} | seed + {gt_tag}")
                row[0].axis("off")

                row[1].imshow(_overlay_mask(raw_slice, gt_slice, (0.0, 1.0, 0.0)))
                _draw_roi_box(
                    row[1],
                    info["x1"],
                    info["x2"],
                    info["y1"],
                    info["y2"],
                    color="yellow",
                )
                row[1].set_title(
                    f"{mod_names[m]} | ROI box + {gt_tag}\n"
                    f"z={z_idx}, roi_z=[{info['z1']},{info['z2']})"
                )
                row[1].axis("off")

                roi_mod = image_roi[b, m].detach().float().cpu().numpy()
                if info["z2"] > info["z1"]:
                    roi_z = int(np.clip(z_idx - int(info["z1"]), 0, roi_mod.shape[-1] - 1))
                else:
                    roi_z = roi_mod.shape[-1] // 2

                row[2].imshow(_norm01(roi_mod[:, :, roi_z]), cmap="gray")
                row[2].set_title(f"{mod_names[m]} | ROI input\nroi_z={roi_z}")
                row[2].axis("off")

                row[3].imshow(_overlay_mask(raw_slice, pred_slice, (1.0, 0.0, 0.0)))
                row[3].set_title(f"{mod_names[m]} | {pred_tag} + raw")
                row[3].axis("off")

                row[4].imshow(_overlay_mask(raw_slice, gt_slice, (0.0, 1.0, 0.0)))
                row[4].set_title(f"{mod_names[m]} | {gt_tag} + raw")
                row[4].axis("off")

            fig.tight_layout()
            fig.savefig(vis_dir / f"{case_id}.png", bbox_inches="tight")
            plt.close(fig)

            global_counter += 1

        del out, seg_prob_full, seg_pred_full, roi_mask_full, x, y_seg, y_cls
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"[Export] Finished exporting visualizations to: {vis_dir}")


if __name__ == "__main__":
    cfg = load_cfg("config.yml")
    train_loop(cfg)
