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
    model.train()

    recon_weight = float(cfg.train.recon_weight)  # 必须 >0，确保 Et/G 有梯度（多卡不炸）
    inv_weight = float(getattr(cfg.train, "inv_weight", 0.0))
    grad_clip = float(getattr(cfg.train, "grad_clip", 0.0))
    log_interval = int(getattr(cfg.logging, "log_interval", 20))

    act = Activations(sigmoid=True)
    to_bin = AsDiscrete(threshold=0.5)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")

    running = {"loss": 0.0, "loss_seg": 0.0, "loss_recon": 0.0, "dice": 0.0, "hd95": 0.0}
    n_steps = 0

    out_ch = int(getattr(cfg.model, "out_ch", 1))
    take_first = bool(getattr(cfg.data, "label_take_first_channel", True))
    device = accelerator.device

    for step, batch in enumerate(train_loader):
        if batch is None:
            continue

        # ✅ 显式搬运到 GPU（解决你当前报错的关键）
        x = batch["image"].to(device, non_blocking=True)
        y = batch["seg_label"].to(device, non_blocking=True)
        y = select_label_channel(y, out_ch=out_ch, take_first=take_first)

        with accelerator.accumulate(model):
            # ✅ 让 autocast 生效（bf16/fp16）
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

        # Metrics
        with torch.no_grad():
            prob = act(out.logits)
            pred = to_bin(prob)
            dice_metric(pred, y)
            hd95_metric(pred, y)
            dice_val = dice_metric.aggregate().detach()
            hd95_val = hd95_metric.aggregate().detach()
            dice_metric.reset()
            hd95_metric.reset()

        running["loss"] += float(loss.detach().item())
        running["loss_seg"] += float(loss_pack["loss_seg"].item())
        running["loss_recon"] += float(loss_pack["loss_recon"].item())
        running["dice"] += float(dice_val.item())
        running["hd95"] += float(hd95_val.item())
        n_steps += 1

        if accelerator.is_main_process and (step % log_interval == 0):
            accelerator.log(
                {
                    "train/loss": running["loss"] / max(n_steps, 1),
                    "train/loss_seg": running["loss_seg"] / max(n_steps, 1),
                    "train/loss_recon": running["loss_recon"] / max(n_steps, 1),
                    "train/dice": running["dice"] / max(n_steps, 1),
                    "train/hd95": running["hd95"] / max(n_steps, 1),
                    "epoch": epoch,
                },
                step=epoch * 100000 + step,
            )

    if n_steps == 0:
        return {"loss": 0.0, "loss_seg": 0.0, "loss_recon": 0.0, "dice": 0.0, "hd95": 0.0}
    return {k: v / n_steps for k, v in running.items()}


@torch.no_grad()
def val_one_epoch(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    val_loader,
    epoch: int,
    cfg: Any,
) -> Dict[str, float]:
    model.eval()

    act = Activations(sigmoid=True)
    to_bin = AsDiscrete(threshold=0.5)

    dice_metric = DiceMetric(include_background=True, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")

    dice_list = []
    hd95_list = []

    out_ch = int(getattr(cfg.model, "out_ch", 1))
    take_first = bool(getattr(cfg.data, "label_take_first_channel", True))
    device = accelerator.device

    for batch in val_loader:
        if batch is None:
            continue

        # ✅ 显式搬运到 GPU（验证同样要）
        x = batch["image"].to(device, non_blocking=True)
        y = batch["seg_label"].to(device, non_blocking=True)
        y = select_label_channel(y, out_ch=out_ch, take_first=take_first)

        with accelerator.autocast():
            out = model(x)

        prob = act(out.logits)
        pred = to_bin(prob)

        dice_metric(pred, y)
        hd95_metric(pred, y)

        d = dice_metric.aggregate()
        h = hd95_metric.aggregate()
        dice_metric.reset()
        hd95_metric.reset()

        # 多卡汇总
        d_g = accelerator.gather_for_metrics(d)
        h_g = accelerator.gather_for_metrics(h)

        dice_list.append(d_g.flatten())
        hd95_list.append(h_g.flatten())

    if len(dice_list) == 0:
        stats = {"dice": 0.0, "hd95": 0.0}
    else:
        dice_all = torch.cat(dice_list, dim=0)
        hd95_all = torch.cat(hd95_list, dim=0)
        stats = {
            "dice": float(torch.nanmean(dice_all).item()),
            "hd95": float(torch.nanmean(hd95_all).item()),
        }

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
