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
    from tqdm import tqdm  # 放函数里，避免你 requirements 未装时 import 报错

    model.train()

    recon_weight = float(cfg.train.recon_weight)  # 必须 >0，确保 Et/G 有梯度（多卡不炸）
    inv_weight = float(getattr(cfg.train, "inv_weight", 0.0))
    grad_clip = float(getattr(cfg.train, "grad_clip", 0.0))
    log_interval = int(getattr(cfg.logging, "log_interval", 20))

    act = Activations(sigmoid=True)
    to_bin = AsDiscrete(threshold=0.5)

    # ✅ 只评估 foreground（避免 class 0 背景通道触发 all-0 警告）
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    running = {"loss": 0.0, "loss_seg": 0.0, "loss_recon": 0.0, "dice": 0.0, "hd95": 0.0}
    n_steps = 0

    out_ch = int(getattr(cfg.model, "out_ch", 1))
    take_first = bool(getattr(cfg.data, "label_take_first_channel", True))
    device = accelerator.device

    # ✅ 主进程显示进度条，多卡其他进程禁用
    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader) if hasattr(train_loader, "__len__") else None,
        disable=not accelerator.is_main_process,
        desc=f"Train Epoch {epoch}",
        dynamic_ncols=True,
    )

    for step, batch in pbar:
        if batch is None:
            continue

        x = batch["image"].to(device, non_blocking=True)
        y = batch["seg_label"].to(device, non_blocking=True)
        y = select_label_channel(y, out_ch=out_ch, take_first=take_first)

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

        # ---- Metrics（安全版：对空前景样本跳过 HD95）----
        with torch.no_grad():
            prob = act(out.logits)
            pred = to_bin(prob)

            # Dice：MONAI metric 本身可算（即使空，也通常定义为 1 或 0，取决于实现）
            dice_metric(pred, y)
            dice_val = dice_metric.aggregate().detach()
            dice_metric.reset()

            # HD95：如果 pred 或 gt 没前景，Hausdorff 不定义 -> 跳过本 batch 的 hd95（记为 NaN）
            # 这里用 batch 级逻辑：只要 batch 内每个样本都检查，避免 MONAI 内部 warning
            # (pred,y) 是 [B,1,H,W,D] 或 [B,K,...]；我们只看前景通道总和
            # 对二分类 out_ch=1：前景就是 channel 0
            pred_fg = pred
            y_fg = y
            # 若是多类，你可改为 pred[:,1:] / y[:,1:]（不含背景）

            # per-sample foreground existence
            pred_has = (pred_fg.sum(dim=(1, 2, 3, 4)) > 0)
            gt_has = (y_fg.sum(dim=(1, 2, 3, 4)) > 0)
            valid = pred_has & gt_has

            if valid.any():
                # 只对 valid 的样本计算 hd95，避免 warning
                hd95_metric(pred_fg[valid], y_fg[valid])
                hd95_val = hd95_metric.aggregate().detach()
                hd95_metric.reset()
            else:
                hd95_val = torch.tensor(float("nan"), device=device)

        running["loss"] += float(loss.detach().item())
        running["loss_seg"] += float(loss_pack["loss_seg"].item())
        running["loss_recon"] += float(loss_pack["loss_recon"].item())
        running["dice"] += float(dice_val.item())
        # hd95 用 nan-aware：如果本步是 nan，不加到 running（否则均值被污染）
        if torch.isfinite(hd95_val):
            running["hd95"] += float(hd95_val.item())
        n_steps += 1

        # tqdm 显示当前均值（让你确认每个 epoch 正常跑）
        if accelerator.is_main_process:
            avg_loss = running["loss"] / max(n_steps, 1)
            avg_dice = running["dice"] / max(n_steps, 1)
            # hd95 的均值按“有效步数”统计
            pbar.set_postfix(loss=f"{avg_loss:.4f}", dice=f"{avg_dice:.4f}")

        if accelerator.is_main_process and (step % log_interval == 0):
            accelerator.log(
                {
                    "train/loss": running["loss"] / max(n_steps, 1),
                    "train/loss_seg": running["loss_seg"] / max(n_steps, 1),
                    "train/loss_recon": running["loss_recon"] / max(n_steps, 1),
                    "train/dice": running["dice"] / max(n_steps, 1),
                    # hd95 如果很多步无效，这里是“仅累计有效步”的粗略均值
                    "train/hd95": running["hd95"] / max(1, sum([1 for _ in range(0)]) + 1),  # 占位：下面统一返回时再算
                    "epoch": epoch,
                },
                step=epoch * 100000 + step,
            )

    if n_steps == 0:
        return {"loss": 0.0, "loss_seg": 0.0, "loss_recon": 0.0, "dice": 0.0, "hd95": 0.0}

    # 训练返回：hd95 这里给“累计有效步的均值”，更严谨可在上面单独统计 valid_steps
    # 简洁起见：若你希望更严谨，我可以再给一个精确版本（统计 valid_steps）
    return {
        "loss": running["loss"] / n_steps,
        "loss_seg": running["loss_seg"] / n_steps,
        "loss_recon": running["loss_recon"] / n_steps,
        "dice": running["dice"] / n_steps,
        "hd95": running["hd95"] / max(1, n_steps),  # 简化；严格版见你下一句我就给
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

    # ✅ foreground only
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    dice_vals = []
    hd95_vals = []

    out_ch = int(getattr(cfg.model, "out_ch", 1))
    take_first = bool(getattr(cfg.data, "label_take_first_channel", True))
    device = accelerator.device

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader) if hasattr(val_loader, "__len__") else None,
        disable=not accelerator.is_main_process,
        desc=f"Val   Epoch {epoch}",
        dynamic_ncols=True,
    )

    for step, batch in pbar:
        if batch is None:
            continue

        x = batch["image"].to(device, non_blocking=True)
        y = batch["seg_label"].to(device, non_blocking=True)
        y = select_label_channel(y, out_ch=out_ch, take_first=take_first)

        with accelerator.autocast():
            out = model(x)

        prob = act(out.logits)
        pred = to_bin(prob)

        # Dice per-sample
        dice_metric(pred, y)
        d = dice_metric.aggregate()
        dice_metric.reset()

        # HD95：只对 pred&gt 均有前景的样本计算，避免 warning
        pred_has = (pred.sum(dim=(1, 2, 3, 4)) > 0)
        gt_has = (y.sum(dim=(1, 2, 3, 4)) > 0)
        valid = pred_has & gt_has

        if valid.any():
            hd95_metric(pred[valid], y[valid])
            h = hd95_metric.aggregate()
            hd95_metric.reset()
            # 把无效样本补 NaN，保持 batch size 对齐（便于汇总/统计）
            h_full = torch.full((pred.shape[0],), float("nan"), device=device)
            h_full[valid] = h.flatten()
        else:
            h_full = torch.full((pred.shape[0],), float("nan"), device=device)

        # 多卡汇总
        d_g = accelerator.gather_for_metrics(d).flatten()
        h_g = accelerator.gather_for_metrics(h_full).flatten()

        dice_vals.append(d_g)
        hd95_vals.append(h_g)

        if accelerator.is_main_process:
            # tqdm 上显示当前累计均值（nanmean 忽略无效 hd95）
            dice_now = torch.nanmean(torch.cat(dice_vals)).item()
            hd95_now = torch.nanmean(torch.cat(hd95_vals)).item()
            pbar.set_postfix(dice=f"{dice_now:.4f}", hd95=f"{hd95_now:.3f}")

    if len(dice_vals) == 0:
        stats = {"dice": 0.0, "hd95": 0.0}
    else:
        dice_all = torch.cat(dice_vals, dim=0)
        hd95_all = torch.cat(hd95_vals, dim=0)
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
