from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU

from src.loader import get_loaders
from src.utils import (
    cfg_to_plain_dict,
    count_parameters,
    load_cfg,
    load_weights,
    maybe_resume_from_latest,
    prepare_run_dir,
    save_best_weights_if_improved,
    save_latest_checkpoint,
    select_label_channel,
    set_seed,
    start_txt_logger,
)

from model import build_model
from model.entry import compute_dg_losses


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


def _seg_out_channels_from_cfg(cfg: Any) -> int:
    use_mods = _cfg_get(cfg, "data.use_modalities", None)
    if bool(_cfg_get(cfg, "model.seg_out_equals_modalities", True)) and isinstance(
        use_mods, (list, tuple)
    ):
        return int(len(use_mods))
    return int(_cfg_get(cfg, "model.out_ch", 1))


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


def init_accelerator_and_trackers_ddp_safe(cfg: Any, run_dir: Path) -> Accelerator:
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


def _build_cls_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
    loss_name = str(_cfg_get(cfg, "train.classification.loss", "bce")).lower()
    if loss_name != "bce":
        raise ValueError(f"Unsupported classification loss: {loss_name}")

    pos_weight = float(_cfg_get(cfg, "train.classification.pos_weight", 1.0))
    if pos_weight > 0 and abs(pos_weight - 1.0) > 1e-8:
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    return nn.BCEWithLogitsLoss()


def _get_stage(epoch: int, cfg: Any) -> str:
    stage_a_epochs = int(_cfg_get(cfg, "train.multitask.stage_a_epochs", 0))
    if bool(_cfg_get(cfg, "train.multitask.enable", True)) and epoch < stage_a_epochs:
        return "stage_a"
    return "stage_b"


def _apply_training_stage(
    model: nn.Module, accelerator: Accelerator, stage: str, cfg: Any
) -> None:
    raw_model = accelerator.unwrap_model(model)
    for p in raw_model.parameters():
        p.requires_grad = True

    freeze_cls = (
        bool(_cfg_get(cfg, "train.multitask.freeze_cls_in_stage_a", True))
        and stage == "stage_a"
    )
    if freeze_cls and hasattr(raw_model, "cls_branch"):
        for p in raw_model.cls_branch.parameters():
            p.requires_grad = False


def _compute_monitor_score(val_stats: Dict[str, float], stage: str, cfg: Any) -> float:
    key = (
        "train.model_selection.stage_a_monitor"
        if stage == "stage_a"
        else "train.model_selection.monitor"
    )
    monitor = str(_cfg_get(cfg, key, "hybrid")).lower()
    if monitor == "dice":
        return float(val_stats["seg_dice"])
    if monitor == "f1":
        return float(val_stats["cls_f1"])
    if monitor == "hybrid":
        dice_w = float(_cfg_get(cfg, "train.model_selection.hybrid_weights.dice", 0.7))
        f1_w = float(_cfg_get(cfg, "train.model_selection.hybrid_weights.f1", 0.3))
        return dice_w * float(val_stats["seg_dice"]) + f1_w * float(val_stats["cls_f1"])
    raise ValueError(f"Unknown monitor mode: {monitor}")


def _prepare_batch(
    batch: Dict[str, torch.Tensor], device: torch.device, cfg: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["image"].to(device, non_blocking=True)
    y_seg = batch["seg_label"].to(device, non_blocking=True)
    y_cls = batch["class_label"].to(device, non_blocking=True).float()

    out_ch = _seg_out_channels_from_cfg(cfg)
    take_first = bool(_cfg_get(cfg, "data.label_take_first_channel", True))
    y_seg = select_label_channel(y_seg, out_ch=out_ch, take_first=take_first).float()
    return x, y_seg, y_cls


def _compute_seg_step_metrics(
    seg_logits: torch.Tensor, y_seg: torch.Tensor, cfg: Any
) -> Dict[str, float]:
    from monai.metrics import compute_hausdorff_distance

    threshold = float(_cfg_get(cfg, "train.segmentation.threshold", 0.5))
    seg_prob = torch.sigmoid(seg_logits)
    seg_pred = (seg_prob >= threshold).float()

    # Dice / mIoU 仍保留当前写法
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")

    dice_metric(y_pred=seg_pred, y=y_seg)
    iou_metric(y_pred=seg_pred, y=y_seg)
    dice_val = float(dice_metric.aggregate().detach().item())
    iou_val = float(iou_metric.aggregate().detach().item())
    dice_metric.reset()
    iou_metric.reset()

    # HD95 改为函数式计算，避免 DDP 下 aggregate()->_sync() 的 buffer 对齐问题
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

        # 兼容可能出现的 nan / inf
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

    return {
        "dice": dice_val,
        "miou": iou_val,
        "hd95": hd95_val,
        "hd95_cnt": hd95_cnt,
    }

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


def _reduce_scalar(
    accelerator: Accelerator, value: float, device: torch.device
) -> float:
    t = torch.tensor(value, device=device, dtype=torch.float32)
    return float(accelerator.reduce(t, reduction="sum").item())


def train_one_epoch_multitask(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    seg_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    epoch: int,
    stage: str,
    cfg: Any,
) -> Dict[str, float]:
    from tqdm import tqdm

    model.train()
    device = accelerator.device
    grad_clip = float(_cfg_get(cfg, "train.grad_clip", 0.0))
    log_interval = int(_cfg_get(cfg, "logging.log_interval", 20))
    step_timeout = float(_cfg_get(cfg, "train.step_timeout_sec", 0.0))
    recon_weight = float(_cfg_get(cfg, "train.recon_weight", 0.02))
    cls_weight = (
        float(_cfg_get(cfg, "train.cls_weight", 1.0)) if stage == "stage_b" else 0.0
    )
    inv_weight = float(_cfg_get(cfg, "train.inv_weight", 0.0))
    enable_cls_branch = stage == "stage_b"

    running = {
        "loss": 0.0,
        "loss_seg": 0.0,
        "loss_cls": 0.0,
        "loss_recon": 0.0,
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

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader) if hasattr(train_loader, "__len__") else None,
        disable=not accelerator.is_main_process,
        desc=f"Train Epoch {epoch} [{stage}]",
        dynamic_ncols=True,
    )

    for step, batch in pbar:
        step_t0 = time.time()
        local_bad = 0
        if (
            batch is None
            or (not isinstance(batch, dict))
            or ("image" not in batch)
            or ("seg_label" not in batch)
            or ("class_label" not in batch)
        ):
            local_bad = 1
        bad = torch.tensor([local_bad], device=device, dtype=torch.int32)
        if int(accelerator.reduce(bad, reduction="sum").item()) > 0:
            optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.set_postfix(skip="bad_batch")
            continue

        try:
            x, y_seg, y_cls = _prepare_batch(batch, device, cfg)
            local_bad2 = 0
        except Exception:
            local_bad2 = 1
        bad2 = torch.tensor([local_bad2], device=device, dtype=torch.int32)
        if int(accelerator.reduce(bad2, reduction="sum").item()) > 0:
            optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.set_postfix(skip="to(device)_fail")
            continue

        t_fw0 = time.time()
        with accelerator.accumulate(model):
            with accelerator.autocast():
                out = model(x, enable_cls_branch=enable_cls_branch)
                loss_pack = compute_dg_losses(
                    out=out,
                    x=x,
                    y=y_seg,
                    seg_loss_fn=seg_loss_fn,
                    class_label=(y_cls if cls_weight > 0 else None),
                    cls_loss_fn=cls_loss_fn,
                    recon_weight=recon_weight,
                    cls_weight=cls_weight,
                    inv_weight=inv_weight,
                    out2=None,
                )
                loss = loss_pack["loss_total"]
            accelerator.backward(loss)
            if grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        t_fw1 = time.time()

        seg_metrics = _compute_seg_step_metrics(out.seg.detach(), y_seg.detach(), cfg)
        cls_counts = _compute_cls_counts(out.class_logit.detach(), y_cls.detach(), cfg)

        step_dt = time.time() - step_t0
        timeout = torch.tensor(
            [1 if (step_timeout > 0 and step_dt > step_timeout) else 0],
            device=device,
            dtype=torch.int32,
        )
        if int(accelerator.reduce(timeout, reduction="sum").item()) > 0:
            optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.set_postfix(skip=f"timeout({step_dt:.1f}s)")
            continue

        running["loss"] += float(loss.detach().item())
        running["loss_seg"] += float(loss_pack["loss_seg"].item())
        running["loss_cls"] += float(loss_pack["loss_cls"].item())
        running["loss_recon"] += float(loss_pack["loss_recon"].item())
        running["seg_dice"] += float(seg_metrics["dice"])
        running["seg_miou"] += float(seg_metrics["miou"])
        running["seg_hd95"] += float(seg_metrics["hd95"])
        running["seg_hd95_cnt"] += float(seg_metrics["hd95_cnt"])
        running["tp"] += cls_counts["tp"]
        running["tn"] += cls_counts["tn"]
        running["fp"] += cls_counts["fp"]
        running["fn"] += cls_counts["fn"]
        n_steps += 1

        if accelerator.is_main_process:
            cls_now = _classification_stats_from_counts(
                running["tp"], running["tn"], running["fp"], running["fn"]
            )
            pbar.set_postfix(
                loss=f"{running['loss'] / max(n_steps, 1):.4f}",
                dice=f"{running['seg_dice'] / max(n_steps, 1):.4f}",
                f1=f"{cls_now['f1']:.4f}",
                fw=f"{t_fw1 - t_fw0:.1f}s",
            )

        if accelerator.is_main_process and (step % log_interval == 0):
            cls_now = _classification_stats_from_counts(
                running["tp"], running["tn"], running["fp"], running["fn"]
            )
            accelerator.log(
                {
                    "train/loss": running["loss"] / max(n_steps, 1),
                    "train/loss_seg": running["loss_seg"] / max(n_steps, 1),
                    "train/loss_cls": running["loss_cls"] / max(n_steps, 1),
                    "train/loss_recon": running["loss_recon"] / max(n_steps, 1),
                    "train/seg_dice": running["seg_dice"] / max(n_steps, 1),
                    "train/seg_hd95": running["seg_hd95"]
                    / max(running["seg_hd95_cnt"], 1.0),
                    "train/seg_miou": running["seg_miou"] / max(n_steps, 1),
                    "train/cls_acc": cls_now["acc"],
                    "train/cls_f1": cls_now["f1"],
                    "train/cls_specificity": cls_now["specificity"],
                    "train/cls_recall": cls_now["recall"],
                    "train/cls_miou": cls_now["miou"],
                    "train/stage": 0 if stage == "stage_a" else 1,
                    "epoch": epoch,
                },
                step=epoch * 100000 + step,
            )

    if n_steps == 0:
        zero = 0.0
        return {
            "loss": zero,
            "loss_seg": zero,
            "loss_cls": zero,
            "loss_recon": zero,
            "seg_dice": zero,
            "seg_hd95": zero,
            "seg_miou": zero,
            "cls_acc": zero,
            "cls_f1": zero,
            "cls_specificity": zero,
            "cls_recall": zero,
            "cls_miou": zero,
        }

    loss = _reduce_scalar(accelerator, running["loss"], device)
    loss_seg = _reduce_scalar(accelerator, running["loss_seg"], device)
    loss_cls = _reduce_scalar(accelerator, running["loss_cls"], device)
    loss_recon = _reduce_scalar(accelerator, running["loss_recon"], device)
    seg_dice = _reduce_scalar(accelerator, running["seg_dice"], device)
    seg_hd95 = _reduce_scalar(accelerator, running["seg_hd95"], device)
    seg_miou = _reduce_scalar(accelerator, running["seg_miou"], device)
    seg_hd95_cnt = _reduce_scalar(accelerator, running["seg_hd95_cnt"], device)
    n_steps_g = _reduce_scalar(accelerator, float(n_steps), device)
    tp = _reduce_scalar(accelerator, running["tp"], device)
    tn = _reduce_scalar(accelerator, running["tn"], device)
    fp = _reduce_scalar(accelerator, running["fp"], device)
    fn = _reduce_scalar(accelerator, running["fn"], device)
    cls_stats = _classification_stats_from_counts(tp, tn, fp, fn)
    return {
        "loss": loss / max(n_steps_g, 1.0),
        "loss_seg": loss_seg / max(n_steps_g, 1.0),
        "loss_cls": loss_cls / max(n_steps_g, 1.0),
        "loss_recon": loss_recon / max(n_steps_g, 1.0),
        "seg_dice": seg_dice / max(n_steps_g, 1.0),
        "seg_hd95": seg_hd95 / max(seg_hd95_cnt, 1.0),
        "seg_miou": seg_miou / max(n_steps_g, 1.0),
        "cls_acc": cls_stats["acc"],
        "cls_f1": cls_stats["f1"],
        "cls_specificity": cls_stats["specificity"],
        "cls_recall": cls_stats["recall"],
        "cls_miou": cls_stats["miou"],
    }


@torch.no_grad()
def eval_one_epoch_multitask(
    *,
    split_name: str,
    accelerator: Accelerator,
    model: nn.Module,
    data_loader,
    seg_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    epoch: int,
    stage: str,
    cfg: Any,
    log_to_tracker: bool = False,
) -> Dict[str, float]:
    from tqdm import tqdm

    model.eval()
    device = accelerator.device
    step_timeout = float(_cfg_get(cfg, "train.step_timeout_sec", 0.0))
    recon_weight = float(_cfg_get(cfg, "train.recon_weight", 0.02))
    cls_weight = (
        float(_cfg_get(cfg, "train.cls_weight", 1.0)) if stage == "stage_b" else 0.0
    )
    inv_weight = float(_cfg_get(cfg, "train.inv_weight", 0.0))
    enable_cls_branch = stage == "stage_b"

    running = {
        "loss": 0.0,
        "loss_seg": 0.0,
        "loss_cls": 0.0,
        "loss_recon": 0.0,
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

    pbar = tqdm(
        enumerate(data_loader),
        total=len(data_loader) if hasattr(data_loader, "__len__") else None,
        disable=not accelerator.is_main_process,
        desc=f"{split_name.capitalize()} Epoch {epoch} [{stage}]",
        dynamic_ncols=True,
    )

    for _, batch in pbar:
        step_t0 = time.time()
        if (
            batch is None
            or (not isinstance(batch, dict))
            or ("image" not in batch)
            or ("seg_label" not in batch)
            or ("class_label" not in batch)
        ):
            continue
        x, y_seg, y_cls = _prepare_batch(batch, device, cfg)
        with accelerator.autocast():
            out = model(x, enable_cls_branch=enable_cls_branch)
            loss_pack = compute_dg_losses(
                out=out,
                x=x,
                y=y_seg,
                seg_loss_fn=seg_loss_fn,
                class_label=(y_cls if cls_weight > 0 else None),
                cls_loss_fn=cls_loss_fn,
                recon_weight=recon_weight,
                cls_weight=cls_weight,
                inv_weight=inv_weight,
                out2=None,
            )
            loss = loss_pack["loss_total"]

        seg_metrics = _compute_seg_step_metrics(out.seg.detach(), y_seg.detach(), cfg)
        cls_counts = _compute_cls_counts(out.class_logit.detach(), y_cls.detach(), cfg)

        step_dt = time.time() - step_t0
        timeout = torch.tensor(
            [1 if (step_timeout > 0 and step_dt > step_timeout) else 0],
            device=device,
            dtype=torch.int32,
        )
        if int(accelerator.reduce(timeout, reduction="sum").item()) > 0:
            if accelerator.is_main_process:
                pbar.set_postfix(skip=f"timeout({step_dt:.1f}s)")
            continue

        running["loss"] += float(loss.detach().item())
        running["loss_seg"] += float(loss_pack["loss_seg"].item())
        running["loss_cls"] += float(loss_pack["loss_cls"].item())
        running["loss_recon"] += float(loss_pack["loss_recon"].item())
        running["seg_dice"] += float(seg_metrics["dice"])
        running["seg_miou"] += float(seg_metrics["miou"])
        running["seg_hd95"] += float(seg_metrics["hd95"])
        running["seg_hd95_cnt"] += float(seg_metrics["hd95_cnt"])
        running["tp"] += cls_counts["tp"]
        running["tn"] += cls_counts["tn"]
        running["fp"] += cls_counts["fp"]
        running["fn"] += cls_counts["fn"]
        n_steps += 1

        if accelerator.is_main_process:
            cls_now = _classification_stats_from_counts(
                running["tp"], running["tn"], running["fp"], running["fn"]
            )
            pbar.set_postfix(
                dice=f"{running['seg_dice'] / max(n_steps, 1):.4f}",
                f1=f"{cls_now['f1']:.4f}",
            )

    if n_steps == 0:
        zero = 0.0
        stats = {
            "loss": zero,
            "loss_seg": zero,
            "loss_cls": zero,
            "loss_recon": zero,
            "seg_dice": zero,
            "seg_hd95": zero,
            "seg_miou": zero,
            "cls_acc": zero,
            "cls_f1": zero,
            "cls_specificity": zero,
            "cls_recall": zero,
            "cls_miou": zero,
        }
    else:
        loss = _reduce_scalar(accelerator, running["loss"], device)
        loss_seg = _reduce_scalar(accelerator, running["loss_seg"], device)
        loss_cls = _reduce_scalar(accelerator, running["loss_cls"], device)
        loss_recon = _reduce_scalar(accelerator, running["loss_recon"], device)
        seg_dice = _reduce_scalar(accelerator, running["seg_dice"], device)
        seg_hd95 = _reduce_scalar(accelerator, running["seg_hd95"], device)
        seg_miou = _reduce_scalar(accelerator, running["seg_miou"], device)
        seg_hd95_cnt = _reduce_scalar(accelerator, running["seg_hd95_cnt"], device)
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
            "loss_recon": loss_recon / max(n_steps_g, 1.0),
            "seg_dice": seg_dice / max(n_steps_g, 1.0),
            "seg_hd95": seg_hd95 / max(seg_hd95_cnt, 1.0),
            "seg_miou": seg_miou / max(n_steps_g, 1.0),
            "cls_acc": cls_stats["acc"],
            "cls_f1": cls_stats["f1"],
            "cls_specificity": cls_stats["specificity"],
            "cls_recall": cls_stats["recall"],
            "cls_miou": cls_stats["miou"],
        }

    if accelerator.is_main_process and log_to_tracker:
        accelerator.log(
            {
                f"{split_name}/loss": stats["loss"],
                f"{split_name}/loss_seg": stats["loss_seg"],
                f"{split_name}/loss_cls": stats["loss_cls"],
                f"{split_name}/loss_recon": stats["loss_recon"],
                f"{split_name}/seg_dice": stats["seg_dice"],
                f"{split_name}/seg_hd95": stats["seg_hd95"],
                f"{split_name}/seg_miou": stats["seg_miou"],
                f"{split_name}/cls_acc": stats["cls_acc"],
                f"{split_name}/cls_f1": stats["cls_f1"],
                f"{split_name}/cls_specificity": stats["cls_specificity"],
                f"{split_name}/cls_recall": stats["cls_recall"],
                f"{split_name}/cls_miou": stats["cls_miou"],
                f"{split_name}/stage": 0 if stage == "stage_a" else 1,
                "epoch": epoch,
            },
            step=epoch,
        )
    return stats


if __name__ == "__main__":
    cfg = load_cfg("config.yml")
    run_dir, run_name = prepare_run_dir(cfg)

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        txt_log_path = start_txt_logger(run_dir, filename="console.txt")
        print(f"Console log -> {txt_log_path}")

    accelerator = init_accelerator_and_trackers_ddp_safe(cfg, run_dir)

    seed = int(_cfg_get(cfg, "train.seed", 42))
    set_seed(seed + accelerator.process_index)

    train_loader, val_loader, test_loader = get_loaders(cfg)
    if val_loader is None:
        raise ValueError(
            "val_loader is None; please ensure val_split_ratio produces a validation set."
        )
    if test_loader is None:
        accelerator.print("Warning: test_loader is None; test metrics will be skipped.")

    model = build_model(cfg)

    init_weights_path = str(_cfg_get(cfg, "checkpoint.init_weights_path", "")).strip()
    if init_weights_path:
        info = load_weights(
            model=model,
            weights_path=init_weights_path,
            strict=False,
            map_location="cpu",
        )
        if accelerator.is_main_process:
            accelerator.print(
                f"Loaded init weights from {init_weights_path} | missing={len(info['missing_keys'])} unexpected={len(info['unexpected_keys'])}"
            )
        

    if accelerator.is_main_process:
        pinfo = count_parameters(model)
        accelerator.print(
            f"Model params: total={pinfo['total']:,} trainable={pinfo['trainable']:,}"
        )

    seg_loss_fn = DiceCELoss(sigmoid=True, squared_pred=False, reduction="mean")
    cls_loss_fn = _build_cls_loss_fn(cfg, device=accelerator.device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(_cfg_get(cfg, "train.lr", 5e-5)),
        weight_decay=float(_cfg_get(cfg, "train.weight_decay", 1e-4)),
    )
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
        accelerator=accelerator, cfg=cfg, run_dir=run_dir
    )
    if accelerator.is_main_process:
        accelerator.print(
            f"Run: {run_name} | start_epoch={start_epoch} | best_score={best_score}"
        )

    epochs = int(_cfg_get(cfg, "train.epochs", 100))
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        stage = _get_stage(epoch, cfg)
        _apply_training_stage(model, accelerator, stage, cfg)
        accelerator.wait_for_everyone()

        train_stats = train_one_epoch_multitask(
            accelerator=accelerator,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            seg_loss_fn=seg_loss_fn,
            cls_loss_fn=cls_loss_fn,
            epoch=epoch,
            stage=stage,
            cfg=cfg,
        )
        if scheduler is not None:
            scheduler.step()

        val_stats = eval_one_epoch_multitask(
            split_name="val",
            accelerator=accelerator,
            model=model,
            data_loader=val_loader,
            seg_loss_fn=seg_loss_fn,
            cls_loss_fn=cls_loss_fn,
            epoch=epoch,
            stage=stage,
            cfg=cfg,
            log_to_tracker=True,
        )
        test_stats = None
        if test_loader is not None:
            test_stats = eval_one_epoch_multitask(
                split_name="test",
                accelerator=accelerator,
                model=model,
                data_loader=test_loader,
                seg_loss_fn=seg_loss_fn,
                cls_loss_fn=cls_loss_fn,
                epoch=epoch,
                stage=stage,
                cfg=cfg,
                log_to_tracker=True,
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
                f"Epoch {epoch}/{epochs - 1} [{stage}] | "
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
