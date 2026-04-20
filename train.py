from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.metrics import DiceMetric, MeanIoU, compute_hausdorff_distance

from src.loader import get_loaders
from src.utils import (
    cfg_to_plain_dict,
    count_parameters,
    load_cfg,
    maybe_resume_from_latest,
    prepare_run_dir,
    save_best_weights_if_improved,
    save_latest_checkpoint,
    select_label_channel,
    set_seed,
    start_txt_logger,
)

# 按你的当前模型文件路径修改这里
from model.structured_uncertainty_model import Model


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
            yaml.safe_dump(
                cfg_to_plain_dict(cfg),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    accelerator.wait_for_everyone()
    accelerator.init_trackers(
        project_name=str(
            _cfg_get(cfg, "logging.project_name", "structured_uncertainty")
        ),
        config=_flatten_tb_config(cfg),
    )
    return accelerator


def build_model_from_cfg(cfg: Any) -> nn.Module:
    in_ch = int(len(_cfg_get(cfg, "data.use_modalities", [0])))
    out_ch = int(_cfg_get(cfg, "model.out_ch", 1))
    img_size = tuple(int(v) for v in _cfg_get(cfg, "data.target_size", [128, 128, 64]))
    return Model(in_ch=in_ch, out_ch=out_ch, img_size=img_size)


def build_optimizer(cfg: Any, model: nn.Module) -> AdamW:
    lr_backbone = float(
        _cfg_get(cfg, "train.optimizer.lr_backbone", _cfg_get(cfg, "train.lr", 1e-4))
    )
    lr_structure = float(_cfg_get(cfg, "train.optimizer.lr_structure", lr_backbone))
    lr_cls = float(_cfg_get(cfg, "train.optimizer.lr_cls", lr_backbone))
    wd = float(_cfg_get(cfg, "train.weight_decay", 1e-4))

    backbone_params = []
    structure_params = []
    cls_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(p)
        elif name.startswith("cls_head"):
            cls_params.append(p)
        else:
            structure_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": wd,
                "group_name": "backbone",
            }
        )
    if structure_params:
        param_groups.append(
            {
                "params": structure_params,
                "lr": lr_structure,
                "weight_decay": wd,
                "group_name": "structure",
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


def _set_requires_grad_for_module(mod: Optional[nn.Module], flag: bool) -> None:
    if mod is None:
        return
    for p in mod.parameters():
        p.requires_grad = flag


def _compute_curriculum_stage(epoch: int, cfg: Any) -> Dict[str, Any]:
    epochs = int(_cfg_get(cfg, "train.epochs", 100))
    warmup_epochs = int(_cfg_get(cfg, "train.curriculum.warmup_epochs", 15))
    cls_only_epochs = int(_cfg_get(cfg, "train.curriculum.cls_only_epochs", 10))
    finetune_epochs = int(_cfg_get(cfg, "train.curriculum.finetune_epochs", 10))
    unc_ramp_epochs = int(_cfg_get(cfg, "train.curriculum.uncertainty_ramp_epochs", 20))

    cls_only_start = max(warmup_epochs, epochs - cls_only_epochs - finetune_epochs)
    finetune_start = max(cls_only_start, epochs - finetune_epochs)

    if epoch < warmup_epochs:
        stage = "warmup"
    elif epoch < cls_only_start:
        stage = "uncertainty"
    elif epoch < finetune_start:
        stage = "cls_only"
    else:
        stage = "finetune"

    progress_unc = 0.0
    if stage in ("uncertainty", "finetune"):
        raw = (epoch - warmup_epochs + 1) / max(unc_ramp_epochs, 1)
        progress_unc = float(min(1.0, max(0.0, raw)))

    w_seg = float(_cfg_get(cfg, "train.loss_weights.seg", 1.0))
    w_unc_final = float(_cfg_get(cfg, "train.loss_weights.unc_final", 0.5))
    w_core_final = float(_cfg_get(cfg, "train.loss_weights.core_final", 0.2))
    w_rim_final = float(_cfg_get(cfg, "train.loss_weights.rim_final", 0.3))
    w_cls_final = float(_cfg_get(cfg, "train.loss_weights.cls_final", 1.0))
    w_cls_uncertainty = float(
        _cfg_get(cfg, "train.loss_weights.cls_in_uncertainty_stage", 0.0)
    )
    w_cls_finetune = float(
        _cfg_get(cfg, "train.loss_weights.cls_in_finetune_stage", w_cls_final)
    )

    if stage == "warmup":
        weights = {
            "seg": w_seg,
            "cls": 0.0,
            "unc": 0.0,
            "core": 0.0,
            "rim": 0.0,
        }
    elif stage == "uncertainty":
        weights = {
            "seg": w_seg,
            "cls": w_cls_uncertainty,
            "unc": w_unc_final * progress_unc,
            "core": w_core_final * progress_unc,
            "rim": w_rim_final * progress_unc,
        }
    elif stage == "cls_only":
        weights = {
            "seg": 0.0,
            "cls": w_cls_final,
            "unc": 0.0,
            "core": 0.0,
            "rim": 0.0,
        }
    else:
        weights = {
            "seg": w_seg,
            "cls": w_cls_finetune,
            "unc": w_unc_final,
            "core": w_core_final,
            "rim": w_rim_final,
        }

    return {
        "stage": stage,
        "weights": weights,
        "warmup_epochs": warmup_epochs,
        "cls_only_start": cls_only_start,
        "finetune_start": finetune_start,
        "use_uncertainty": stage in ("uncertainty", "finetune"),
    }


def apply_stage_policy(
    model: nn.Module, accelerator: Accelerator, stage_cfg: Dict[str, Any], cfg: Any
) -> Dict[str, Any]:
    raw = accelerator.unwrap_model(model)
    stage = stage_cfg["stage"]

    # 默认全开
    for p in raw.parameters():
        p.requires_grad = True

    if stage == "warmup":
        _set_requires_grad_for_module(getattr(raw, "cls_head", None), False)
    elif stage == "uncertainty":
        _set_requires_grad_for_module(getattr(raw, "cls_head", None), False)
    elif stage == "cls_only":
        _set_requires_grad_for_module(getattr(raw, "backbone", None), False)
        _set_requires_grad_for_module(getattr(raw, "feat_proj", None), False)
        _set_requires_grad_for_module(getattr(raw, "z_mamba", None), False)
        _set_requires_grad_for_module(getattr(raw, "uncertainty", None), False)
        _set_requires_grad_for_module(getattr(raw, "cls_head", None), True)
    elif stage == "finetune":
        # 全部打开，但 backbone 低学习率由 optimizer group 控制
        pass

    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in raw.parameters() if not p.requires_grad)
    return {"trainable_params": trainable, "frozen_params": frozen, "stage": stage}


def _update_optimizer_lrs(optimizer: AdamW, stage: str, cfg: Any) -> None:
    base_backbone = float(
        _cfg_get(cfg, "train.optimizer.lr_backbone", _cfg_get(cfg, "train.lr", 1e-4))
    )
    base_structure = float(_cfg_get(cfg, "train.optimizer.lr_structure", base_backbone))
    base_cls = float(_cfg_get(cfg, "train.optimizer.lr_cls", base_backbone))

    if stage == "warmup":
        mult = {"backbone": 1.0, "structure": 1.0, "cls": 0.0}
    elif stage == "uncertainty":
        mult = {"backbone": 1.0, "structure": 1.0, "cls": 0.0}
    elif stage == "cls_only":
        mult = {"backbone": 0.0, "structure": 0.0, "cls": 1.0}
    else:
        mult = {
            "backbone": float(
                _cfg_get(cfg, "train.optimizer.finetune_backbone_mult", 0.2)
            ),
            "structure": float(
                _cfg_get(cfg, "train.optimizer.finetune_structure_mult", 0.5)
            ),
            "cls": float(_cfg_get(cfg, "train.optimizer.finetune_cls_mult", 1.0)),
        }

    for group in optimizer.param_groups:
        name = group.get("group_name", "")
        if name == "backbone":
            group["lr"] = base_backbone * mult["backbone"]
        elif name == "structure":
            group["lr"] = base_structure * mult["structure"]
        elif name == "cls":
            group["lr"] = base_cls * mult["cls"]


def _reduce_scalar(
    accelerator: Accelerator, value: float, device: torch.device
) -> float:
    t = torch.tensor(value, device=device, dtype=torch.float32)
    return float(accelerator.reduce(t, reduction="sum").item())


def _prepare_batch(
    batch: Dict[str, torch.Tensor], device: torch.device, cfg: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["image"].to(device, non_blocking=True)
    y_seg = batch["seg_label"].to(device, non_blocking=True)
    y_cls = batch["class_label"].to(device, non_blocking=True).float()

    out_ch = int(_cfg_get(cfg, "model.out_ch", 1))
    take_first = bool(_cfg_get(cfg, "data.label_take_first_channel", True))
    y_seg = select_label_channel(y_seg, out_ch=out_ch, take_first=take_first).float()
    return x, y_seg, y_cls


def _compose_weighted_loss(
    loss_dict: Dict[str, torch.Tensor], stage_cfg: Dict[str, Any]
) -> torch.Tensor:
    w = stage_cfg["weights"]

    def _get(name: str) -> torch.Tensor:
        if name in loss_dict:
            return loss_dict[name]
        ref = next(iter(loss_dict.values()))
        return ref * 0.0

    loss = (
        w["seg"] * _get("loss_seg")
        + w["cls"] * _get("loss_cls")
        + w["unc"] * _get("loss_unc")
        + w["core"] * _get("loss_core")
        + w["rim"] * _get("loss_rim")
    )
    return loss


def _compute_seg_step_metrics(
    seg_logits: torch.Tensor, y_seg: torch.Tensor, cfg: Any
) -> Dict[str, float]:
    threshold = float(_cfg_get(cfg, "train.segmentation.threshold", 0.5))
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
            hd95_val = float(hd.mean().item())
            hd95_cnt = 1.0
        else:
            hd95_val = 0.0
            hd95_cnt = 0.0
    else:
        hd95_val = 0.0
        hd95_cnt = 0.0

    return {"dice": dice_val, "miou": iou_val, "hd95": hd95_val, "hd95_cnt": hd95_cnt}


def _compute_cls_counts(
    class_logit: torch.Tensor, y_cls: torch.Tensor, cfg: Any
) -> Dict[str, float]:
    threshold = float(_cfg_get(cfg, "train.classification.threshold", 0.5))
    pred = (torch.sigmoid(class_logit) >= threshold).long().view(-1)
    target = y_cls.long().view(-1)
    tp = float(((pred == 1) & (target == 1)).sum().item())
    tn = float(((pred == 0) & (target == 0)).sum().item())
    fp = float(((pred == 1) & (target == 0)).sum().item())
    fn = float(((pred == 0) & (target == 1)).sum().item())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _choose_vis_slice(
    y_seg: torch.Tensor, seg_prob: Optional[torch.Tensor] = None
) -> int:
    vol = y_seg[0, 0]
    gt_sum = vol.sum(dim=(0, 1))
    if float(gt_sum.max().item()) > 0:
        return int(torch.argmax(gt_sum).item())
    if seg_prob is not None:
        pred_sum = seg_prob[0, 0].sum(dim=(0, 1))
        return int(torch.argmax(pred_sum).item())
    return int(vol.shape[-1] // 2)


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


def _overlay_heatmap(gray: np.ndarray, heat: np.ndarray) -> np.ndarray:
    gray = _norm01(gray)
    heat = _norm01(heat)
    rgb = np.stack([gray, gray, gray], axis=-1)
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[..., :3].astype(np.float32)
    alpha = 0.45
    out = rgb * (1.0 - alpha) + heat_rgb * alpha
    return np.clip(out, 0.0, 1.0)


def save_visualization(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Any,
    epoch: int,
    split_name: str,
    run_dir: Path,
    stage_cfg: Dict[str, Any],
) -> None:
    if not accelerator.is_main_process:
        return
    if not bool(_cfg_get(cfg, "logging.visualization.enable", True)):
        return
    if not stage_cfg["use_uncertainty"]:
        return

    vis_splits = list(_cfg_get(cfg, "logging.visualization.splits", ["val"]))
    if split_name not in vis_splits:
        return

    raw_model = accelerator.unwrap_model(model)
    device = accelerator.device
    x, y_seg, _ = _prepare_batch(batch, device, cfg)
    x_vis = x[:1]
    y_vis = y_seg[:1]

    raw_model.eval()
    with torch.no_grad():
        out = raw_model(x_vis)

    seg_prob = torch.sigmoid(out.seg[:1])
    seg_pred = (
        seg_prob >= float(_cfg_get(cfg, "train.segmentation.threshold", 0.5))
    ).float()
    core_prob = torch.sigmoid(out.core[:1])
    rim_prob = torch.sigmoid(out.rim[:1])
    unc_prob = torch.sigmoid(out.unc[:1])

    modalities = list(_cfg_get(cfg, "data.use_modalities", []))
    num_mod = int(x_vis.shape[1])
    if len(modalities) != num_mod:
        modalities = [f"mod_{i}" for i in range(num_mod)]

    z_idx = _choose_vis_slice(y_vis, seg_prob=seg_prob)

    gt_mask = y_vis[0, 0].detach().float().cpu().numpy()
    pred_mask = seg_pred[0, 0].detach().float().cpu().numpy()
    core_map = core_prob[0, 0].detach().float().cpu().numpy()
    rim_map = rim_prob[0, 0].detach().float().cpu().numpy()
    unc_map = unc_prob[0, 0].detach().float().cpu().numpy()

    n_rows = num_mod
    n_cols = 6
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        dpi=150,
        squeeze=False,
    )

    for m in range(num_mod):
        raw_vol = x_vis[0, m].detach().float().cpu().numpy()
        raw_slice = raw_vol[:, :, z_idx]
        gt_slice = gt_mask[:, :, z_idx]
        pred_slice = pred_mask[:, :, z_idx]
        core_slice = core_map[:, :, z_idx]
        rim_slice = rim_map[:, :, z_idx]
        unc_slice = unc_map[:, :, z_idx]

        row_axes = axes[m]
        row_axes[0].imshow(_norm01(raw_slice), cmap="gray")
        row_axes[0].set_title(f"{modalities[m]} | Raw")
        row_axes[0].axis("off")

        row_axes[1].imshow(_overlay_mask(raw_slice, gt_slice, (0.0, 1.0, 0.0)))
        row_axes[1].set_title(f"{modalities[m]} | GT")
        row_axes[1].axis("off")

        row_axes[2].imshow(_overlay_mask(raw_slice, pred_slice, (1.0, 0.0, 0.0)))
        row_axes[2].set_title(f"{modalities[m]} | Pred")
        row_axes[2].axis("off")

        row_axes[3].imshow(_overlay_heatmap(raw_slice, core_slice))
        row_axes[3].set_title(f"{modalities[m]} | Core")
        row_axes[3].axis("off")

        row_axes[4].imshow(_overlay_heatmap(raw_slice, rim_slice))
        row_axes[4].set_title(f"{modalities[m]} | Rim")
        row_axes[4].axis("off")

        row_axes[5].imshow(_overlay_heatmap(raw_slice, unc_slice))
        row_axes[5].set_title(f"{modalities[m]} | Unc")
        row_axes[5].axis("off")

    vis_dir = run_dir / "visuals" / split_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / f"epoch_{epoch:04d}_{stage_cfg['stage']}.png"
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    del out, seg_prob, seg_pred, core_prob, rim_prob, unc_prob
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
    stage_cfg: Dict[str, Any],
    cfg: Any,
    run_dir: Path,
) -> Dict[str, float]:
    from tqdm import tqdm

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

    running = {
        "loss": 0.0,
        "loss_seg": 0.0,
        "loss_cls": 0.0,
        "loss_unc": 0.0,
        "loss_core": 0.0,
        "loss_rim": 0.0,
        "seg_dice": 0.0,
        "seg_hd95": 0.0,
        "seg_miou": 0.0,
        "seg_hd95_cnt": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
    }
    n_steps = 0
    vis_batch = None

    pbar = tqdm(
        enumerate(data_loader),
        total=len(data_loader) if hasattr(data_loader, "__len__") else None,
        disable=not accelerator.is_main_process,
        desc=f"{split_name.capitalize()} Epoch {epoch} [{stage_cfg['stage']}]",
        dynamic_ncols=True,
    )

    for step, batch in pbar:
        if batch is None:
            continue
        if vis_batch is None:
            vis_batch = batch

        x, y_seg, y_cls = _prepare_batch(batch, device, cfg)

        if is_train:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    out = model(x)
                    loss_dict = raw_model.compute_loss(out, y_seg, y_cls)
                    loss = _compose_weighted_loss(loss_dict, stage_cfg)

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
                    out = model(x)
                    loss_dict = raw_model.compute_loss(out, y_seg, y_cls)
                    loss = _compose_weighted_loss(loss_dict, stage_cfg)

        seg_metrics = _compute_seg_step_metrics(out.seg.detach(), y_seg.detach(), cfg)
        cls_counts = _compute_cls_counts(out.cls.detach(), y_cls.detach(), cfg)

        running["loss"] += float(loss.detach().item())
        running["loss_seg"] += float(loss_dict["loss_seg"].detach().item())
        running["loss_cls"] += float(loss_dict["loss_cls"].detach().item())
        running["loss_unc"] += float(loss_dict["loss_unc"].detach().item())
        running["loss_core"] += float(loss_dict["loss_core"].detach().item())
        running["loss_rim"] += float(loss_dict["loss_rim"].detach().item())
        running["seg_dice"] += float(seg_metrics["dice"])
        running["seg_hd95"] += float(seg_metrics["hd95"])
        running["seg_hd95_cnt"] += float(seg_metrics["hd95_cnt"])
        running["seg_miou"] += float(seg_metrics["miou"])
        running["tp"] += cls_counts["tp"]
        running["tn"] += cls_counts["tn"]
        running["fp"] += cls_counts["fp"]
        running["fn"] += cls_counts["fn"]
        n_steps += 1

        if accelerator.is_main_process and step % log_interval == 0:
            cls_now = _classification_stats_from_counts(
                running["tp"], running["tn"], running["fp"], running["fn"]
            )
            pbar.set_postfix(
                loss=f"{running['loss'] / max(n_steps, 1):.4f}",
                dice=f"{running['seg_dice'] / max(n_steps, 1):.4f}",
                f1=f"{cls_now['f1']:.4f}",
            )

        del x, y_seg, y_cls, out, loss
        if torch.cuda.is_available() and step % 10 == 0:
            torch.cuda.empty_cache()

    if vis_batch is not None:
        save_visualization(
            accelerator=accelerator,
            model=model,
            batch=vis_batch,
            cfg=cfg,
            epoch=epoch,
            split_name=split_name,
            run_dir=run_dir,
            stage_cfg=stage_cfg,
        )

    if n_steps == 0:
        return {
            "loss": 0.0,
            "loss_seg": 0.0,
            "loss_cls": 0.0,
            "loss_unc": 0.0,
            "loss_core": 0.0,
            "loss_rim": 0.0,
            "seg_dice": 0.0,
            "seg_hd95": 0.0,
            "seg_miou": 0.0,
            "cls_acc": 0.0,
            "cls_f1": 0.0,
            "cls_specificity": 0.0,
            "cls_recall": 0.0,
            "cls_miou": 0.0,
        }

    loss = _reduce_scalar(accelerator, running["loss"], device)
    loss_seg = _reduce_scalar(accelerator, running["loss_seg"], device)
    loss_cls = _reduce_scalar(accelerator, running["loss_cls"], device)
    loss_unc = _reduce_scalar(accelerator, running["loss_unc"], device)
    loss_core = _reduce_scalar(accelerator, running["loss_core"], device)
    loss_rim = _reduce_scalar(accelerator, running["loss_rim"], device)
    seg_dice = _reduce_scalar(accelerator, running["seg_dice"], device)
    seg_hd95 = _reduce_scalar(accelerator, running["seg_hd95"], device)
    seg_hd95_cnt = _reduce_scalar(accelerator, running["seg_hd95_cnt"], device)
    seg_miou = _reduce_scalar(accelerator, running["seg_miou"], device)
    n_steps_g = _reduce_scalar(accelerator, float(n_steps), device)
    tp = _reduce_scalar(accelerator, running["tp"], device)
    tn = _reduce_scalar(accelerator, running["tn"], device)
    fp = _reduce_scalar(accelerator, running["fp"], device)
    fn = _reduce_scalar(accelerator, running["fn"], device)
    cls_stats = _classification_stats_from_counts(tp, tn, fp, fn)

    stats = {
        "loss": loss / max(n_steps_g, 1.0),
        "loss_seg": loss_seg / max(n_steps_g, 1.0),
        "loss_cls": loss_cls / max(n_steps_g, 1.0),
        "loss_unc": loss_unc / max(n_steps_g, 1.0),
        "loss_core": loss_core / max(n_steps_g, 1.0),
        "loss_rim": loss_rim / max(n_steps_g, 1.0),
        "seg_dice": seg_dice / max(n_steps_g, 1.0),
        "seg_hd95": seg_hd95 / max(seg_hd95_cnt, 1.0),
        "seg_miou": seg_miou / max(n_steps_g, 1.0),
        "cls_acc": cls_stats["acc"],
        "cls_f1": cls_stats["f1"],
        "cls_specificity": cls_stats["specificity"],
        "cls_recall": cls_stats["recall"],
        "cls_miou": cls_stats["miou"],
    }

    if accelerator.is_main_process:
        accelerator.log(
            {
                f"{split_name}/loss": stats["loss"],
                f"{split_name}/loss_seg": stats["loss_seg"],
                f"{split_name}/loss_cls": stats["loss_cls"],
                f"{split_name}/loss_unc": stats["loss_unc"],
                f"{split_name}/loss_core": stats["loss_core"],
                f"{split_name}/loss_rim": stats["loss_rim"],
                f"{split_name}/seg_dice": stats["seg_dice"],
                f"{split_name}/seg_hd95": stats["seg_hd95"],
                f"{split_name}/seg_miou": stats["seg_miou"],
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


def _compute_monitor_score(val_stats: Dict[str, float], cfg: Any) -> float:
    monitor = str(_cfg_get(cfg, "train.model_selection.monitor", "hybrid")).lower()
    if monitor == "seg_dice":
        return float(val_stats["seg_dice"])
    if monitor == "cls_f1":
        return float(val_stats["cls_f1"])
    if monitor == "cls_acc":
        return float(val_stats["cls_acc"])

    dice_w = float(_cfg_get(cfg, "train.model_selection.hybrid_weights.dice", 0.7))
    f1_w = float(_cfg_get(cfg, "train.model_selection.hybrid_weights.f1", 0.3))
    return dice_w * float(val_stats["seg_dice"]) + f1_w * float(val_stats["cls_f1"])


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
        raise ValueError(
            "val_loader is None; please ensure val_split_ratio produces a validation set."
        )
    if test_loader is None:
        accelerator.print("Warning: test_loader is None; test metrics will be skipped.")

    model = build_model_from_cfg(cfg)
    optimizer = build_optimizer(cfg, model)

    scheduler = None
    if str(_cfg_get(cfg, "train.scheduler", "cosine")).lower() == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(_cfg_get(cfg, "train.epochs", 100)),
            eta_min=float(_cfg_get(cfg, "train.min_lr", 1e-6)),
        )

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
        pinfo = count_parameters(accelerator.unwrap_model(model))
        accelerator.print(
            f"Run: {run_name} | start_epoch={start_epoch} | best_score={best_score}"
        )
        accelerator.print(
            f"Model params: total={pinfo['total']:,} trainable={pinfo['trainable']:,}"
        )

    epochs = int(_cfg_get(cfg, "train.epochs", 100))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        stage_cfg = _compute_curriculum_stage(epoch, cfg)
        policy_info = apply_stage_policy(model, accelerator, stage_cfg, cfg)
        _update_optimizer_lrs(optimizer, stage_cfg["stage"], cfg)
        accelerator.wait_for_everyone()

        train_stats = run_one_epoch(
            split_name="train",
            is_train=True,
            accelerator=accelerator,
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            stage_cfg=stage_cfg,
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
            stage_cfg=stage_cfg,
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
                stage_cfg=stage_cfg,
                cfg=cfg,
                run_dir=run_dir,
            )

        score = _compute_monitor_score(val_stats, cfg)

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

        if accelerator.is_main_process:
            dt = time.time() - t0
            w = stage_cfg["weights"]
            msg = (
                f"Epoch {epoch}/{epochs - 1} [{stage_cfg['stage']}] "
                f"| trainable={policy_info['trainable_params']:,} "
                f"| w(seg={w['seg']:.3f}, cls={w['cls']:.3f}, unc={w['unc']:.3f}, core={w['core']:.3f}, rim={w['rim']:.3f}) "
                f"| train: loss={train_stats['loss']:.4f} seg(dice={train_stats['seg_dice']:.4f}, hd95={train_stats['seg_hd95']:.3f}, miou={train_stats['seg_miou']:.4f}) "
                f"cls(acc={train_stats['cls_acc']:.4f}, f1={train_stats['cls_f1']:.4f}, spec={train_stats['cls_specificity']:.4f}, rec={train_stats['cls_recall']:.4f}) "
                f"| val: loss={val_stats['loss']:.4f} seg(dice={val_stats['seg_dice']:.4f}, hd95={val_stats['seg_hd95']:.3f}, miou={val_stats['seg_miou']:.4f}) "
                f"cls(acc={val_stats['cls_acc']:.4f}, f1={val_stats['cls_f1']:.4f}, spec={val_stats['cls_specificity']:.4f}, rec={val_stats['cls_recall']:.4f})"
            )
            if test_stats is not None:
                msg += (
                    f" | test: loss={test_stats['loss']:.4f} seg(dice={test_stats['seg_dice']:.4f}, hd95={test_stats['seg_hd95']:.3f}, miou={test_stats['seg_miou']:.4f}) "
                    f"cls(acc={test_stats['cls_acc']:.4f}, f1={test_stats['cls_f1']:.4f}, spec={test_stats['cls_specificity']:.4f}, rec={test_stats['cls_recall']:.4f})"
                )
            msg += f" | best={best_score:.4f} | {dt:.1f}s"
            accelerator.print(msg)

    accelerator.end_training()


if __name__ == "__main__":
    cfg = load_cfg("config.yml")
    train_loop(cfg)
