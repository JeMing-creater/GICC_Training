from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from accelerate import Accelerator

# 尽量使用 easydict；若环境没有，提供一个最小替代，保证不炸
try:
    from easydict import EasyDict  # type: ignore
except Exception:
    class EasyDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as e:
                raise AttributeError(name) from e
        def __setattr__(self, name, value):
            self[name] = value


# -------------------------
# Config
# -------------------------
def load_cfg(path: str | Path = "config.yml") -> EasyDict:
    cfg = EasyDict(
        yaml.load(open(str(path), "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    return cfg


def cfg_to_plain_dict(obj: Any) -> Any:
    """
    把 EasyDict/嵌套 dict/list 递归转换成纯 Python 类型，方便：
      - accelerator.init_trackers(config=...)
      - json/yaml dump
    """
    if isinstance(obj, dict):
        return {k: cfg_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [cfg_to_plain_dict(v) for v in obj]
    return obj


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Filesystem helpers
# -------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_rmtree(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)


def now_run_name(tz_name: str = "Asia/Seoul") -> str:
    try:
        import datetime as _dt
        from zoneinfo import ZoneInfo
        dt = _dt.datetime.now(ZoneInfo(tz_name))
    except Exception:
        import datetime as _dt
        dt = _dt.datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


# -------------------------
# Label adapter
# -------------------------
def select_label_channel(y: torch.Tensor, out_ch: int, take_first: bool = True) -> torch.Tensor:
    """
    适配你的 label：
    - 你当前 loader 可能把 label 也按模态 concat => y: [B,C,H,W,D]
    - 若 out_ch==1，则默认取 y[:,0:1] 作为最终标签
    """
    if y.ndim != 5:
        raise ValueError(f"Expected label tensor 5D [B,C,H,W,D], got {tuple(y.shape)}")

    if y.shape[1] == out_ch:
        return y

    if out_ch == 1 and y.shape[1] > 1:
        if take_first:
            return y[:, 0:1]
        return (y.sum(dim=1, keepdim=True) > 0).to(y.dtype)

    raise ValueError(f"Label channel mismatch: out_ch={out_ch}, y.shape[1]={y.shape[1]}")


# -------------------------
# Logging / TensorBoard (Accelerate trackers)
# -------------------------
def prepare_run_dir(cfg: EasyDict) -> Tuple[Path, str]:
    run_name = str(getattr(cfg.logging, "run_name", "")).strip()
    if not run_name:
        run_name = now_run_name(str(getattr(cfg.logging, "timezone", "Asia/Seoul")))

    log_root = ensure_dir(Path(str(getattr(cfg.logging, "log_root", "log"))))
    run_dir = ensure_dir(log_root / run_name)
    return run_dir, run_name


def init_accelerator_and_trackers(cfg: EasyDict, run_dir: Path) -> Accelerator:
    """
    初始化 Accelerator + TensorBoard tracker

    关键修复：
      - accelerate 会调用 tensorboard.add_hparams(config)
      - add_hparams 的 value 必须是 int/float/str/bool/torch.Tensor
      - 因此要把 cfg 递归清洗成“全标量 dict”（list/dict 转成 str）
    """

    accelerator = Accelerator(
        gradient_accumulation_steps=int(getattr(cfg.train, "grad_accum_steps", 1)),
        mixed_precision=str(getattr(cfg.accelerate, "mixed_precision", "no")),
        log_with="tensorboard",
        project_dir=str(run_dir),
    )

    # 保存完整 config（原始结构）
    if accelerator.is_main_process:
        (run_dir / "config_resolved.yml").write_text(
            yaml.safe_dump(cfg_to_plain_dict(cfg), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()

    # ---- TensorBoard hparams 清洗 ----
    def _tb_scalarize(v):
        # TensorBoard add_hparams only accepts: int/float/str/bool/torch.Tensor
        if isinstance(v, (int, float, str, bool)):
            return v
        if torch.is_tensor(v):
            return v
        # list/tuple/dict/None/其他类型 -> str
        return str(v)

    def _flatten(prefix: str, obj: Any, out: Dict[str, Any]):
        if isinstance(obj, dict):
            for k, vv in obj.items():
                kk = f"{prefix}.{k}" if prefix else str(k)
                _flatten(kk, vv, out)
        elif isinstance(obj, (list, tuple)):
            # list/tuple 直接 stringify（也可展开，但容易太长）
            out[prefix] = _tb_scalarize(obj)
        else:
            out[prefix] = _tb_scalarize(obj)

    cfg_plain = cfg_to_plain_dict(cfg)
    tb_cfg: Dict[str, Any] = {}
    _flatten("", cfg_plain, tb_cfg)

    # ✅ 不传 log_dir（否则会重复）
    accelerator.init_trackers(
        project_name=str(getattr(cfg.logging, "project_name", "colon_mri_dg_seg")),
        config=tb_cfg,  # ✅ 已保证全部 value 合法
    )

    return accelerator

# -------------------------
# Checkpoints (latest overwrite) + best weights
# -------------------------
def _ckpt_paths(cfg: EasyDict, run_dir: Path) -> Dict[str, Path]:
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    return {
        "ckpt_dir": ckpt_dir,
        "latest_dir": ckpt_dir / str(cfg.checkpoint.latest_dirname),
        "best_path": ckpt_dir / str(cfg.checkpoint.best_filename),
        "latest_meta": ckpt_dir / "latest_meta.json",
    }


def save_latest_checkpoint(
    *,
    accelerator: Accelerator,
    cfg: EasyDict,
    run_dir: Path,
    epoch: int,
    best_score: float,
    current_score: float,
) -> None:
    p = _ckpt_paths(cfg, run_dir)
    latest_dir = p["latest_dir"]
    latest_meta = p["latest_meta"]

    if accelerator.is_main_process:
        safe_rmtree(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(str(latest_dir))

    if accelerator.is_main_process:
        meta = {
            "epoch": int(epoch),
            "best_score": float(best_score),
            "current_score": float(current_score),
            "time": float(time.time()),
        }
        latest_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def save_best_weights_if_improved(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    cfg: EasyDict,
    run_dir: Path,
    epoch: int,
    best_score: float,
    current_score: float,
) -> float:
    p = _ckpt_paths(cfg, run_dir)
    best_path = p["best_path"]

    if current_score > best_score:
        best_score = float(current_score)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {"epoch": int(epoch), "model": unwrapped.state_dict(), "best_score": float(best_score)},
                str(best_path),
            )
    return best_score


def maybe_resume_from_latest(
    *,
    accelerator: Accelerator,
    cfg: EasyDict,
    run_dir: Path,
) -> Tuple[int, float]:
    if not bool(getattr(cfg.checkpoint, "resume", True)):
        return 0, float("-inf")

    p = _ckpt_paths(cfg, run_dir)
    latest_dir = p["latest_dir"]
    latest_meta = p["latest_meta"]

    if latest_dir.exists():
        accelerator.print(f"🔁 Resuming from: {latest_dir}")
        accelerator.load_state(str(latest_dir))

        if latest_meta.exists():
            meta = json.loads(latest_meta.read_text(encoding="utf-8"))
            start_epoch = int(meta.get("epoch", -1)) + 1
            best_score = float(meta.get("best_score", float("-inf")))
            return start_epoch, best_score

    return 0, float("-inf")


# -------------------------
# Weights loading (for init)
# -------------------------
def load_weights(
    *,
    model: torch.nn.Module,
    weights_path: str | Path,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    weights_path = Path(weights_path)
    ckpt = torch.load(str(weights_path), map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing_keys": missing, "unexpected_keys": unexpected, "raw": ckpt}
