from __future__ import annotations

import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from accelerate import Accelerator

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete

from src.loader import get_loaders
from src.utils import (
    load_cfg,
    set_seed,
    count_parameters,
    prepare_run_dir,
    init_accelerator_and_trackers,
    select_label_channel,
    save_latest_checkpoint,
    save_best_weights_if_improved,
    maybe_resume_from_latest,
    load_weights,
)

from model import build_model
from model.entry import compute_dg_losses


def train_one_epoch(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    seg_loss_fn: nn.Module,
    epoch: int,
    cfg: Any,
) -> Dict[str, float]:
    from tqdm import tqdm

    model.train()

    recon_weight = float(cfg.train.recon_weight)
    inv_weight = float(getattr(cfg.train, "inv_weight", 0.0))
    grad_clip = float(getattr(cfg.train, "grad_clip", 0.0))
    log_interval = int(getattr(cfg.logging, "log_interval", 20))

    act = Activations(sigmoid=True)
    to_bin = AsDiscrete(threshold=0.5)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    running = {"loss": 0.0, "loss_seg": 0.0, "loss_recon": 0.0, "dice": 0.0, "hd95": 0.0}
    n_steps = 0
    hd_steps = 0  # 记录 hd95 有效步数

    out_ch = int(getattr(cfg.model, "out_ch", 1))
    take_first = bool(getattr(cfg.data, "label_take_first_channel", True))
    device = accelerator.device

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader) if hasattr(train_loader, "__len__") else None,
        disable=not accelerator.is_main_process,
        desc=f"Train Epoch {epoch}",
        dynamic_ncols=True,
    )

    for step, batch in pbar:
        # -------------------------
        # 1) 同步判定本 step 是否有效（任何 rank 无效 -> 全体跳过）
        # -------------------------
        local_bad = 0
        if batch is None:
            local_bad = 1
        else:
            # 有些情况下 batch 不是 None，但缺 key / 类型不对
            if not isinstance(batch, dict) or ("image" not in batch) or ("seg_label" not in batch):
                local_bad = 1

        bad = torch.tensor([local_bad], device=device, dtype=torch.int32)
        bad_sum = accelerator.reduce(bad, reduction="sum")  # 所有 rank 都会走到这里
        if int(bad_sum.item()) > 0:
            # ✅ 全体同步跳过这个 step，避免有的 backward 有的没 backward
            optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.set_postfix(skip="bad_batch")
            continue

        # -------------------------
        # 2) 安全搬运到 device（如果搬运失败也要同步跳过）
        # -------------------------
        try:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["seg_label"].to(device, non_blocking=True)
            y = select_label_channel(y, out_ch=out_ch, take_first=take_first)
            local_bad2 = 0
        except Exception:
            local_bad2 = 1

        bad2 = torch.tensor([local_bad2], device=device, dtype=torch.int32)
        bad2_sum = accelerator.reduce(bad2, reduction="sum")
        if int(bad2_sum.item()) > 0:
            optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.set_postfix(skip="to(device)_fail")
            continue

        # -------------------------
        # 3) 正常训练 step（所有 rank 一致执行）
        # -------------------------
        with accelerator.accumulate(model):
            with accelerator.autocast():
                out = model(x)
                loss_pack = compute_dg_losses(
                    out=out,
                    x=x,
                    y=y,
                    seg_loss_fn=seg_loss_fn,
                    recon_weight=recon_weight,
                    inv_weight=inv_weight,
                    out2=None,
                )
                loss = loss_pack["loss_total"]

            accelerator.backward(loss)

            if grad_clip and grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # -------------------------
        # 4) Metrics（本地计算，不做 gather；tqdm 显示心跳）
        # -------------------------
        with torch.no_grad():
            prob = act(out.logits)
            pred = to_bin(prob)

            dice_metric(pred, y)
            dice_val = dice_metric.aggregate().detach()
            dice_metric.reset()

            pred_has = (pred.sum(dim=(1, 2, 3, 4)) > 0)
            gt_has = (y.sum(dim=(1, 2, 3, 4)) > 0)
            valid = pred_has & gt_has

            if valid.any():
                hd95_metric(pred[valid], y[valid])
                hd95_val = hd95_metric.aggregate().detach()
                hd95_metric.reset()
            else:
                hd95_val = torch.tensor(float("nan"), device=device)

        running["loss"] += float(loss.detach().item())
        running["loss_seg"] += float(loss_pack["loss_seg"].item())
        running["loss_recon"] += float(loss_pack["loss_recon"].item())
        running["dice"] += float(dice_val.item())
        if torch.isfinite(hd95_val):
            running["hd95"] += float(hd95_val.item())
            hd_steps += 1

        n_steps += 1

        if accelerator.is_main_process:
            avg_loss = running["loss"] / max(n_steps, 1)
            avg_dice = running["dice"] / max(n_steps, 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", dice=f"{avg_dice:.4f}")

        if accelerator.is_main_process and (step % log_interval == 0):
            accelerator.log(
                {
                    "train/loss": running["loss"] / max(n_steps, 1),
                    "train/loss_seg": running["loss_seg"] / max(n_steps, 1),
                    "train/loss_recon": running["loss_recon"] / max(n_steps, 1),
                    "train/dice": running["dice"] / max(n_steps, 1),
                    "train/hd95": running["hd95"] / max(hd_steps, 1),
                    "epoch": epoch,
                },
                step=epoch * 100000 + step,
            )

    if n_steps == 0:
        return {"loss": 0.0, "loss_seg": 0.0, "loss_recon": 0.0, "dice": 0.0, "hd95": 0.0}

    return {
        "loss": running["loss"] / n_steps,
        "loss_seg": running["loss_seg"] / n_steps,
        "loss_recon": running["loss_recon"] / n_steps,
        "dice": running["dice"] / n_steps,
        "hd95": running["hd95"] / max(hd_steps, 1),
    }


@torch.no_grad()
def val_one_epoch(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    val_loader,
    epoch: int,
    cfg: Any,
) -> Dict[str, float]:
    from tqdm import tqdm

    model.eval()
    act = Activations(sigmoid=True)
    to_bin = AsDiscrete(threshold=0.5)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    out_ch = int(getattr(cfg.model, "out_ch", 1))
    take_first = bool(getattr(cfg.data, "label_take_first_channel", True))
    device = accelerator.device

    dice_sum = torch.tensor(0.0, device=device)
    dice_cnt = torch.tensor(0.0, device=device)
    hd95_sum = torch.tensor(0.0, device=device)
    hd95_cnt = torch.tensor(0.0, device=device)

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader) if hasattr(val_loader, "__len__") else None,
        disable=not accelerator.is_main_process,
        desc=f"Val   Epoch {epoch}",
        dynamic_ncols=True,
    )

    for step, batch in pbar:
        local_bad = 0
        if batch is None:
            local_bad = 1
        else:
            if not isinstance(batch, dict) or ("image" not in batch) or ("seg_label" not in batch):
                local_bad = 1

        bad = torch.tensor([local_bad], device=device, dtype=torch.int32)
        bad_sum = accelerator.reduce(bad, reduction="sum")
        if int(bad_sum.item()) > 0:
            if accelerator.is_main_process:
                pbar.set_postfix(skip="bad_batch")
            continue

        try:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["seg_label"].to(device, non_blocking=True)
            y = select_label_channel(y, out_ch=out_ch, take_first=take_first)
            local_bad2 = 0
        except Exception:
            local_bad2 = 1

        bad2 = torch.tensor([local_bad2], device=device, dtype=torch.int32)
        bad2_sum = accelerator.reduce(bad2, reduction="sum")
        if int(bad2_sum.item()) > 0:
            if accelerator.is_main_process:
                pbar.set_postfix(skip="to(device)_fail")
            continue

        with accelerator.autocast():
            out = model(x)

        prob = act(out.logits)
        pred = to_bin(prob)

        dice_metric(pred, y)
        d = dice_metric.aggregate().detach()
        dice_metric.reset()
        if torch.isfinite(d):
            dice_sum += d
            dice_cnt += 1.0

        pred_has = (pred.sum(dim=(1, 2, 3, 4)) > 0)
        gt_has = (y.sum(dim=(1, 2, 3, 4)) > 0)
        valid = pred_has & gt_has
        if valid.any():
            hd95_metric(pred[valid], y[valid])
            h = hd95_metric.aggregate().detach()
            hd95_metric.reset()
            if torch.isfinite(h):
                hd95_sum += h
                hd95_cnt += 1.0

        if accelerator.is_main_process:
            dice_now = (dice_sum / dice_cnt).item() if dice_cnt.item() > 0 else 0.0
            hd95_now = (hd95_sum / hd95_cnt).item() if hd95_cnt.item() > 0 else 0.0
            pbar.set_postfix(dice=f"{dice_now:.4f}", hd95=f"{hd95_now:.3f}")

    # 全局 reduce（一次）
    dice_sum_g = accelerator.reduce(dice_sum, reduction="sum")
    dice_cnt_g = accelerator.reduce(dice_cnt, reduction="sum")
    hd95_sum_g = accelerator.reduce(hd95_sum, reduction="sum")
    hd95_cnt_g = accelerator.reduce(hd95_cnt, reduction="sum")

    dice_mean = (dice_sum_g / dice_cnt_g).item() if dice_cnt_g.item() > 0 else 0.0
    hd95_mean = (hd95_sum_g / hd95_cnt_g).item() if hd95_cnt_g.item() > 0 else 0.0

    stats = {"dice": float(dice_mean), "hd95": float(hd95_mean)}
    if accelerator.is_main_process:
        accelerator.log({"val/dice": stats["dice"], "val/hd95": stats["hd95"], "epoch": epoch}, step=epoch)
    return stats


if __name__ == "__main__":
    cfg = load_cfg("config.yml")

    run_dir, run_name = prepare_run_dir(cfg)
    accelerator = init_accelerator_and_trackers(cfg, run_dir)

    seed = int(getattr(cfg.train, "seed", 42))
    set_seed(seed + accelerator.process_index)

    train_loader, val_loader = get_loaders(cfg.data)

    model = build_model(cfg)

    init_weights_path = str(getattr(cfg.checkpoint, "init_weights_path", "")).strip()
    if init_weights_path:
        info = load_weights(model=model, weights_path=init_weights_path, strict=False, map_location="cpu")
        if accelerator.is_main_process:
            accelerator.print(
                f"🧩 Loaded init weights from {init_weights_path} | "
                f"missing={len(info['missing_keys'])} unexpected={len(info['unexpected_keys'])}"
            )

    if accelerator.is_main_process:
        pinfo = count_parameters(model)
        accelerator.print(f"🧠 Model params: total={pinfo['total']:,} trainable={pinfo['trainable']:,}")

    seg_loss_fn = DiceCELoss(sigmoid=True, squared_pred=False, reduction="mean")

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(getattr(cfg.train, "weight_decay", 1e-4)),
    )

    scheduler = None
    if str(getattr(cfg.train, "scheduler", "cosine")).lower() == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.epochs),
            eta_min=float(getattr(cfg.train, "min_lr", 1e-6)),
        )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    start_epoch, best_score = maybe_resume_from_latest(accelerator=accelerator, cfg=cfg, run_dir=run_dir)
    if accelerator.is_main_process:
        accelerator.print(f"🏁 Run: {run_name} | start_epoch={start_epoch} | best_score={best_score}")

    epochs = int(cfg.train.epochs)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_stats = train_one_epoch(
            accelerator=accelerator,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            seg_loss_fn=seg_loss_fn,
            epoch=epoch,
            cfg=cfg,
        )

        if scheduler is not None:
            scheduler.step()

        val_stats = val_one_epoch(
            accelerator=accelerator,
            model=model,
            val_loader=val_loader,
            epoch=epoch,
            cfg=cfg,
        )

        score = float(val_stats["dice"])  # best criterion: Dice

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
            accelerator.print(
                f"Epoch {epoch}/{epochs-1} | "
                f"train loss={train_stats['loss']:.4f} dice={train_stats['dice']:.4f} hd95={train_stats['hd95']:.3f} | "
                f"val dice={val_stats['dice']:.4f} hd95={val_stats['hd95']:.3f} | "
                f"best dice={best_score:.4f} | {dt:.1f}s"
            )

    accelerator.end_training()
