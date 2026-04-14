from __future__ import annotations

import gc
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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

from monai.losses import DiceCELoss
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
from model.dino_model import Model


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
            return v
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
        project_name=str(_cfg_get(cfg, "logging.project_name", "colon_mri_dg_seg")),
        config=_flatten_tb_config(cfg),
    )
    return accelerator


def build_model_from_cfg(cfg: Any) -> nn.Module:
    in_ch = int(len(_cfg_get(cfg, "data.use_modalities", [0])))
    out_ch = int(_cfg_get(cfg, "model.out_ch", 1))
    return Model(in_ch=in_ch, out_ch=out_ch)


def build_optimizer(cfg: Any, model: nn.Module) -> AdamW:
    lr = float(_cfg_get(cfg, "train.lr", 1e-3))
    wd = float(_cfg_get(cfg, "train.weight_decay", 1e-4))

    raw = model
    enc_model = getattr(getattr(raw, "enc", None), "model", None)
    dino_param_ids = (
        set(id(p) for p in enc_model.parameters()) if enc_model is not None else set()
    )

    backbone_params = []
    head_params = []
    for p in raw.parameters():
        if not p.requires_grad:
            continue
        if id(p) in dino_param_ids:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr, "weight_decay": wd})
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": lr * 0.1, "weight_decay": wd}
        )
    return AdamW(param_groups)


def _get_stage(epoch: int, cfg: Any) -> str:
    a_epochs = int(
        _cfg_get(
            cfg,
            "train.three_stage.stage_a_epochs",
            _cfg_get(cfg, "train.multitask.stage_a_epochs", 20),
        )
    )
    b_epochs = int(_cfg_get(cfg, "train.three_stage.stage_b_epochs", 50))
    if epoch < a_epochs:
        return "stage_a"
    if epoch < a_epochs + b_epochs:
        return "stage_b"
    return "stage_c"


def _dino_blocks(raw_model: nn.Module) -> list[nn.Module]:
    enc_model = getattr(getattr(raw_model, "enc", None), "model", None)
    if enc_model is None:
        return []
    for attr in ["encoder", "dinov2", "vit", "backbone"]:
        obj = getattr(enc_model, attr, None)
        if obj is not None:
            if hasattr(obj, "layer"):
                return list(obj.layer)
            if hasattr(obj, "layers"):
                return list(obj.layers)
    if hasattr(enc_model, "encoder") and hasattr(enc_model.encoder, "layer"):
        return list(enc_model.encoder.layer)
    return []


def apply_stage_policy(
    model: nn.Module, accelerator: Accelerator, stage: str, cfg: Any
) -> Dict[str, Any]:
    raw = accelerator.unwrap_model(model)

    # default: everything trainable except DINO body controlled below
    for p in raw.parameters():
        p.requires_grad = True

    # always keep DINO input projection trainable
    if hasattr(raw, "enc") and hasattr(raw.enc, "proj"):
        for p in raw.enc.proj.parameters():
            p.requires_grad = True

    enc_model = getattr(getattr(raw, "enc", None), "model", None)
    blocks = _dino_blocks(raw)
    n_blocks = len(blocks)

    # freeze all DINO backbone first
    if enc_model is not None:
        for p in enc_model.parameters():
            p.requires_grad = False

    # cls defaults
    cls_weight = 0.0
    train_cls = False

    if stage == "stage_a":
        train_cls = False
        cls_weight = 0.0
    elif stage == "stage_b":
        unfreeze_last = int(
            _cfg_get(cfg, "train.three_stage.unfreeze_last_n_blocks_stage_b", 2)
        )
        if n_blocks > 0:
            for blk in blocks[max(0, n_blocks - unfreeze_last) :]:
                for p in blk.parameters():
                    p.requires_grad = True
        train_cls = False
        cls_weight = 0.0
    else:
        unfreeze_last = int(
            _cfg_get(cfg, "train.three_stage.unfreeze_last_n_blocks_stage_c", 4)
        )
        if n_blocks > 0:
            for blk in blocks[max(0, n_blocks - unfreeze_last) :]:
                for p in blk.parameters():
                    p.requires_grad = True
        train_cls = True
        cls_weight = float(
            _cfg_get(
                cfg,
                "train.three_stage.cls_weight_stage_c",
                _cfg_get(cfg, "train.cls_weight", 0.1),
            )
        )

    # freeze/unfreeze cls head hard to avoid unused grads in early stages
    if hasattr(raw, "cls"):
        for p in raw.cls.parameters():
            p.requires_grad = train_cls

    # head modules always trainable
    for name in ["mamba", "seg", "dummy"]:
        mod = getattr(raw, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    return {
        "stage": stage,
        "train_cls": train_cls,
        "cls_weight": cls_weight,
        "n_dino_blocks": n_blocks,
        "trainable_params": trainable,
    }


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


def compute_losses(
    out, y_seg, y_cls, seg_loss_fn, cls_loss_fn, cls_weight: float
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_seg = seg_loss_fn(out.seg, y_seg)
    loss_cls = cls_loss_fn(out.cls, y_cls) if cls_weight > 0 else out.cls.new_zeros(())
    loss = loss_seg + cls_weight * loss_cls
    return loss, {
        "loss_seg": float(loss_seg.detach().item()),
        "loss_cls": (
            float(loss_cls.detach().item())
            if torch.is_tensor(loss_cls)
            else float(loss_cls)
        ),
    }


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


def _choose_vis_slice(y_seg: torch.Tensor) -> int:
    # y_seg [1,1,W,H,Z] preferred
    vol = y_seg[0, 0]
    flat = vol.sum(dim=(0, 1))
    idx = int(torch.argmax(flat).item())
    return idx


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



def save_visualization(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Any,
    epoch: int,
    split_name: str,
    run_dir: Path,
) -> None:
    if not accelerator.is_main_process:
        return

    raw_model = accelerator.unwrap_model(model)
    device = accelerator.device
    x, y_seg, _ = _prepare_batch(batch, device, cfg)
    x_vis = x[:1]
    y_vis = y_seg[:1]

    raw_model.eval()
    with torch.no_grad():
        # -----------------------------
        # current model forward pipeline
        # -----------------------------
        z_global = raw_model.enc(x_vis)
        z_global = raw_model.mamba(z_global)

        local_feat, local_anchor_logits = raw_model.local_stem(x_vis)
        z_fused = raw_model.local_global_fuse(
            z_global, local_feat, local_anchor_logits
        )

        out = raw_model(x_vis)

    # -----------------------------------------
    # basic data
    # -----------------------------------------
    modalities = list(_cfg_get(cfg, "data.use_modalities", []))
    num_mod = int(x_vis.shape[1])
    if len(modalities) != num_mod:
        modalities = [f"mod_{i}" for i in range(num_mod)]

    pred = (
        torch.sigmoid(out.seg[:1])
        >= float(_cfg_get(cfg, "train.segmentation.threshold", 0.5))
    ).float()

    # GT / pred -> visualization mask
    gt_vol = y_vis[0].detach().float().cpu().numpy()      # [C,W,H,Z]
    pred_vol = pred[0].detach().float().cpu().numpy()     # [C,W,H,Z]

    if gt_vol.shape[0] == 1:
        gt_mask = gt_vol[0]
    else:
        gt_mask = gt_vol.mean(axis=0)

    if pred_vol.shape[0] == 1:
        pred_mask = pred_vol[0]
    else:
        pred_mask = pred_vol.mean(axis=0)

    # -----------------------------------------
    # shared semantic focus from fused feature
    # -----------------------------------------
    shared_focus = z_fused.abs().mean(dim=1, keepdim=True)  # [1,1,w,h,Z]
    shared_focus = F.interpolate(
        shared_focus, size=x_vis.shape[-3:], mode="trilinear", align_corners=False
    )
    shared_focus = shared_focus[0, 0].detach().float().cpu().numpy()  # [W,H,Z]

    # -----------------------------------------
    # local anchor heatmap from local branch
    # -----------------------------------------
    local_anchor = torch.sigmoid(local_anchor_logits)  # [1,1,Wl,Hl,Z]
    local_anchor = F.interpolate(
        local_anchor, size=x_vis.shape[-3:], mode="trilinear", align_corners=False
    )
    local_anchor = local_anchor[0, 0].detach().float().cpu().numpy()  # [W,H,Z]

    z_idx = _choose_vis_slice(y_vis)

    # -----------------------------------------
    # plot: each modality is one row, each row has 5 columns
    # -----------------------------------------
    n_rows = num_mod
    n_cols = 5
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        dpi=150,
        squeeze=False,
    )

    shared_focus_slice = _norm01(shared_focus[:, :, z_idx])
    local_anchor_slice = _norm01(local_anchor[:, :, z_idx])

    for m in range(num_mod):
        raw_vol = x_vis[0, m].detach().float().cpu().numpy()   # [W,H,Z]
        raw_slice = raw_vol[:, :, z_idx]
        raw_norm = _norm01(raw_slice)

        gt_slice = gt_mask[:, :, z_idx]
        pred_slice = pred_mask[:, :, z_idx]

        row_axes = axes[m]

        # 1) Shared semantic focus
        row_axes[0].imshow(raw_norm, cmap="gray")
        row_axes[0].imshow(shared_focus_slice, cmap="jet", alpha=0.45)
        row_axes[0].set_title(f"{modalities[m]} | Shared semantic focus")
        row_axes[0].axis("off")

        # 2) Local anchor heatmap
        row_axes[1].imshow(raw_norm, cmap="gray")
        row_axes[1].imshow(local_anchor_slice, cmap="jet", alpha=0.45)
        row_axes[1].set_title(f"{modalities[m]} | Local anchor heatmap")
        row_axes[1].axis("off")

        # 3) image heatmap
        row_axes[2].imshow(raw_slice, cmap="inferno")
        row_axes[2].set_title(f"{modalities[m]} | Image heatmap")
        row_axes[2].axis("off")

        # 4) GT overlay
        row_axes[3].imshow(_overlay_mask(raw_slice, gt_slice, (0.0, 1.0, 0.0)))
        row_axes[3].set_title(f"{modalities[m]} | GT overlay")
        row_axes[3].axis("off")

        # 5) Pred overlay
        row_axes[4].imshow(_overlay_mask(raw_slice, pred_slice, (1.0, 0.0, 0.0)))
        row_axes[4].set_title(f"{modalities[m]} | Pred overlay")
        row_axes[4].axis("off")

    vis_dir = run_dir / "visuals" / split_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / f"epoch_{epoch:04d}.png"
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    del z_global, local_feat, local_anchor_logits, z_fused, out, pred
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
    seg_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    epoch: int,
    stage_cfg: Dict[str, Any],
    cfg: Any,
    run_dir: Path,
) -> Dict[str, float]:
    from tqdm import tqdm

    device = accelerator.device
    grad_clip = float(_cfg_get(cfg, "train.grad_clip", 0.0))
    log_interval = int(_cfg_get(cfg, "logging.log_interval", 20))
    cls_weight = float(stage_cfg["cls_weight"])

    if is_train:
        model.train()
    else:
        model.eval()

    running = {
        "loss": 0.0,
        "loss_seg": 0.0,
        "loss_cls": 0.0,
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
            optimizer.zero_grad(set_to_none=True)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    out = model(x)
                    loss, loss_parts = compute_losses(
                        out, y_seg, y_cls, seg_loss_fn, cls_loss_fn, cls_weight
                    )
                accelerator.backward(loss)
                if grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            with torch.no_grad():
                with accelerator.autocast():
                    out = model(x)
                    loss, loss_parts = compute_losses(
                        out, y_seg, y_cls, seg_loss_fn, cls_loss_fn, cls_weight
                    )

        seg_metrics = _compute_seg_step_metrics(out.seg.detach(), y_seg.detach(), cfg)
        cls_counts = _compute_cls_counts(out.cls.detach(), y_cls.detach(), cfg)

        running["loss"] += float(loss.detach().item())
        running["loss_seg"] += float(loss_parts["loss_seg"])
        running["loss_cls"] += float(loss_parts["loss_cls"])
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
        )

    if n_steps == 0:
        return {
            "loss": 0.0,
            "loss_seg": 0.0,
            "loss_cls": 0.0,
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


def _build_cls_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
    pos_weight = float(_cfg_get(cfg, "train.classification.pos_weight", 1.0))
    if pos_weight > 0 and abs(pos_weight - 1.0) > 1e-8:
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    return nn.BCEWithLogitsLoss()


def _compute_monitor_score(val_stats: Dict[str, float], stage: str, cfg: Any) -> float:
    # stage A/B emphasize segmentation; stage C hybrid
    if stage in ("stage_a", "stage_b"):
        return float(val_stats["seg_dice"])
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

    seg_loss_fn = DiceCELoss(sigmoid=True, squared_pred=False, reduction="mean")
    cls_loss_fn = _build_cls_loss_fn(cfg, accelerator.device)

    epochs = int(_cfg_get(cfg, "train.epochs", 100))
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        stage = _get_stage(epoch, cfg)
        stage_cfg = apply_stage_policy(model, accelerator, stage, cfg)
        accelerator.wait_for_everyone()

        # refresh optimizer param groups lr after stage changes
        base_lr = float(_cfg_get(cfg, "train.lr", 1e-3))
        for group in optimizer.param_groups:
            group["lr"] = base_lr if group["lr"] >= base_lr * 0.5 else base_lr * 0.1

        train_stats = run_one_epoch(
            split_name="train",
            is_train=True,
            accelerator=accelerator,
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            seg_loss_fn=seg_loss_fn,
            cls_loss_fn=cls_loss_fn,
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
            seg_loss_fn=seg_loss_fn,
            cls_loss_fn=cls_loss_fn,
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
                seg_loss_fn=seg_loss_fn,
                cls_loss_fn=cls_loss_fn,
                epoch=epoch,
                stage_cfg=stage_cfg,
                cfg=cfg,
                run_dir=run_dir,
            )

        score = _compute_monitor_score(val_stats, stage, cfg)
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
            msg = (
                f"Epoch {epoch}/{epochs - 1} [{stage}] | trainable={stage_cfg['trainable_params']:,} | "
                f"train: loss={train_stats['loss']:.4f} seg(dice={train_stats['seg_dice']:.4f}, hd95={train_stats['seg_hd95']:.3f}, miou={train_stats['seg_miou']:.4f}) "
                f"cls(acc={train_stats['cls_acc']:.4f}, f1={train_stats['cls_f1']:.4f}, spec={train_stats['cls_specificity']:.4f}, rec={train_stats['cls_recall']:.4f}, miou={train_stats['cls_miou']:.4f}) | "
                f"val: loss={val_stats['loss']:.4f} seg(dice={val_stats['seg_dice']:.4f}, hd95={val_stats['seg_hd95']:.3f}, miou={val_stats['seg_miou']:.4f}) "
                f"cls(acc={val_stats['cls_acc']:.4f}, f1={val_stats['cls_f1']:.4f}, spec={val_stats['cls_specificity']:.4f}, rec={val_stats['cls_recall']:.4f}, miou={val_stats['cls_miou']:.4f})"
            )
            if test_stats is not None:
                msg += (
                    f" | test: loss={test_stats['loss']:.4f} seg(dice={test_stats['seg_dice']:.4f}, hd95={test_stats['seg_hd95']:.3f}, miou={test_stats['seg_miou']:.4f}) "
                    f"cls(acc={test_stats['cls_acc']:.4f}, f1={test_stats['cls_f1']:.4f}, spec={test_stats['cls_specificity']:.4f}, rec={test_stats['cls_recall']:.4f}, miou={test_stats['cls_miou']:.4f})"
                )
            msg += f" | best={best_score:.4f} | {dt:.1f}s"
            accelerator.print(msg)

    accelerator.end_training()


if __name__ == "__main__":
    cfg = load_cfg("config.yml")
    train_loop(cfg)
