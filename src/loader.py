import os
import yaml
import pandas as pd
import numpy as np
import torch
from easydict import EasyDict
from monai.data import DataLoader, Dataset as MonaiDataset
from monai.transforms import (
    MapTransform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Resized,
    ScaleIntensityRangePercentilesd,
    ConcatItemsd,
    DeleteItemsd,
    EnsureTyped,
    RandRotated,
    RandFlipd,
    RandShiftIntensityd,
    RandScaleIntensityd
)

# ==========================================
# 1. 核心封装：NonEmptyDataLoader
# ==========================================

class NonEmptyDataLoader:
    """
    【Loader 包装器】
    自动过滤掉 DataLoader 产生的 None (空Batch)，
    确保外部循环永远只接收到有效数据。
    """
    def __init__(self, dataloader):
        self.loader = dataloader

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            try:
                batch = next(iterator)
                if batch is not None:
                    yield batch
                else:
                    # 遇到 None，静默跳过，自动获取下一个
                    # print("⚠️ [Loader] 自动跳过一个空 Batch...")
                    continue
            except StopIteration:
                break

    def __len__(self):
        # 注意：实际产出的 batch 数量可能少于 len(loader)，因为部分被跳过了
        return len(self.loader)

# ==========================================
# 2. 自定义 Transforms & Dataset
# ==========================================

class CheckAndFixDimensionsd(MapTransform):
    """
    【修复 4D 错误】
    检查图像和标签维度。
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key not in d: continue
            img = d[key]
            # 形状可能是 (H,W,D) 或 (H,W,D,C)
            if len(img.shape) == 4:
                # 情况 1: 伪4D (H, W, D, 1) -> 降维
                if img.shape[-1] == 1:
                    if hasattr(img, "squeeze"): 
                        d[key] = img.squeeze(-1)
                    else:
                        d[key] = np.squeeze(img, axis=-1)
                # 情况 2: 多通道4D -> 强制取第0通道
                else:
                    print(f"  ⚠️ [Fix4D] 多通道数据 {key} {img.shape}，强制取第0帧")
                    d[key] = img[..., 0]
        return d


class CollectMetaInfod(MapTransform):
    """
    收集 LoadImaged 产生的 *_meta_dict，压缩为 d["meta"]（单样本 dict）。
    注意：collate 由 collate_fn_ignore_none 保持为 List[dict]，避免被 MONAI 拼坏。
    """
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        meta_out = {
            "id": d.get("id", None),
            "excel_id": d.get("excel_id", None),
            "items": {}
        }

        def _to_list(x):
            try:
                import numpy as _np
                if isinstance(x, _np.ndarray):
                    return x.tolist()
            except Exception:
                pass
            try:
                import torch as _torch
                if _torch.is_tensor(x):
                    return x.detach().cpu().numpy().tolist()
            except Exception:
                pass
            return x

        for key in self.key_iterator(d):
            md_key = f"{key}_meta_dict"
            md = d.get(md_key, None)
            if md is None:
                continue

            # 有些 reader 可能返回 list[meta]（极少见），这里做兼容：取第一个
            if isinstance(md, (list, tuple)) and len(md) > 0:
                md = md[0]

            if not isinstance(md, dict):
                continue

            item = {}

            if "filename_or_obj" in md:
                item["filename"] = str(md["filename_or_obj"])

            if "spatial_shape" in md:
                try:
                    item["spatial_shape"] = tuple(int(x) for x in md["spatial_shape"])
                except Exception:
                    item["spatial_shape"] = _to_list(md["spatial_shape"])

            if "spacing" in md:
                item["spacing"] = _to_list(md["spacing"])
            if "origin" in md:
                item["origin"] = _to_list(md["origin"])
            if "direction" in md:
                item["direction"] = _to_list(md["direction"])

            if "affine" in md:
                item["affine"] = _to_list(md["affine"])
            if "original_affine" in md:
                item["original_affine"] = _to_list(md["original_affine"])

            meta_out["items"][key] = item

        d["meta"] = meta_out
        return d
    

class SafeDataset(MonaiDataset):
    """
    【DDP 友好】绝不返回 None。
    遇到 transform/IO 错误时，自动尝试后续样本（最多 max_retry 次）。
    连续失败则抛异常：宁可早失败，也不要训练中 NCCL hang。
    """
    def __init__(self, *args, max_retry: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retry = int(max_retry)

    def __getitem__(self, index):
        last_e = None
        n = len(self.data)

        for k in range(self.max_retry):
            idx = (index + k) % n
            try:
                return super().__getitem__(idx)
            except Exception as e:
                last_e = e
                info = self.data[idx] if isinstance(self.data, list) and idx < len(self.data) else {}
                sid = info.get("id", "Unknown")
                if k == 0:
                    print(f"\n❌ [Read Error] idx={idx} ID={sid} | {str(e)}")
                continue

        raise RuntimeError(
            f"[SafeDataset] Failed after {self.max_retry} retries. Last error: {str(last_e)}"
        )

def collate_fn_ignore_none(batch):
    """
    【DDP 友好 + 保留 meta】
    - 过滤 None
    - 过滤后为空则 raise（避免 DDP 不同步）
    - 对 image/seg_label 等用 monai.list_data_collate
    - 对 meta：保持为 List[dict]，避免被 list_data_collate 把 dict 深度拼坏
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        raise RuntimeError(
            "[collate_fn_ignore_none] Empty batch after filtering None. "
            "This will break DDP. Please check corrupt files / transform errors."
        )

    # 取出 meta（每个样本一个 dict），从 collate 主体里剥离
    metas = [b.get("meta", None) for b in batch]

    # 构造一个不含 meta 的 batch，交给 MONAI collate
    batch_wo_meta = []
    for b in batch:
        bb = dict(b)
        if "meta" in bb:
            bb.pop("meta")
        batch_wo_meta.append(bb)

    from monai.data import list_data_collate
    out = list_data_collate(batch_wo_meta)

    # 重新挂回 meta（保持 list）
    out["meta"] = metas
    return out

# ==========================================
# 3. 路径查找与 ID 匹配逻辑
# ==========================================
FOLDER_ALIASES = {
    "T2_FS": ["T2_FS", "T2", "t2", "T2FS"],
    "ADC":   ["ADC", "adc", "Adc"],
    "V":     ["V", "v", "Venous", "venous"]
}

def find_modality_path(patient_folder, modality_name):
    candidates = FOLDER_ALIASES.get(modality_name, [modality_name])
    for alias in candidates:
        target_path = os.path.join(patient_folder, alias)
        if os.path.exists(target_path):
            return target_path, alias
    return None, None

def build_folder_index(data_folder):
    if not os.path.exists(data_folder): return {}
    index = {}
    try:
        subfolders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    except Exception: return {}
    
    for real_name in subfolders:
        clean_key = real_name.lstrip('0')
        if clean_key == "": clean_key = "0"
        index[clean_key] = real_name
    print(f"✅ 已索引 {len(index)} 个病人文件夹")
    return index

def validate_patient_data(data_folder, real_folder_name, required_modalities, excel_id):
    p_path = os.path.join(data_folder, real_folder_name)
    data_entry = {"id": real_folder_name, "excel_id": excel_id}
    
    for mod in required_modalities:
        mod_folder, _ = find_modality_path(p_path, mod)
        if mod_folder is None: return False, {}, f"缺失模态文件夹: {mod}"
        
        img_file = os.path.join(mod_folder, f"{real_folder_name}.nii.gz")
        if not os.path.exists(img_file): return False, {}, f"缺失图像: {mod}"
        
        seg_file = os.path.join(mod_folder, f"{real_folder_name}seg.nii.gz")
        if not os.path.exists(seg_file): 
            return False, {}, f"缺失 Label: {mod}"
        
        data_entry[f"image_{mod}"] = img_file
        data_entry[f"label_{mod}"] = seg_file

    return True, data_entry, None

def build_data_list(config_item, root_dir, leapfrog_list, data_folder_name="All", required_modalities=[], tag=""):
    excel_path = os.path.join(root_dir, config_item.filename)
    data_folder = os.path.join(root_dir, data_folder_name)
    col_idx = config_item.id_col_index 
    
    if not os.path.exists(excel_path):
        return [], [{"id": "File Missing", "reason": f"Excel不存在"}]
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        return [], [{"id": "Read Error", "reason": str(e)}]
    
    folder_index = build_folder_index(data_folder)
    raw_ids_series = df.iloc[:, col_idx].astype(str).str.strip()
    ids = [x for x in raw_ids_series.unique() if x.lower() != 'nan' and x != '']

    valid_list, failed_list = [], []
    leapfrog_set = set(str(x).strip() for x in leapfrog_list)
    skipped_count = 0
    
    print(f"[{tag}] 扫描 {len(ids)} 个ID...")

    for raw_id in ids:
        if raw_id in leapfrog_set:
            skipped_count += 1; continue

        clean_key = raw_id.lstrip('0')
        if clean_key == "": clean_key = "0"
        
        real_folder_name = folder_index.get(clean_key)
        
        if real_folder_name:
            if real_folder_name in leapfrog_set:
                skipped_count += 1; continue
            
            is_valid, entry, msg = validate_patient_data(data_folder, real_folder_name, required_modalities, raw_id)
            if is_valid:
                valid_list.append(entry)
            else:
                failed_list.append({"id": f"{raw_id}->{real_folder_name}", "reason": msg})
        else:
            failed_list.append({"id": raw_id, "reason": "未找到匹配文件夹"})
            
    if skipped_count > 0: print(f"[{tag}] 跳过 {skipped_count} 个黑名单样本")
    return valid_list, failed_list

# ==========================================
# 4. 数据处理流水线 (Transforms)
# ==========================================

def get_transforms(cfg, stage="train"):
    req_mods = cfg.use_modalities
    image_keys = [f"image_{m}" for m in req_mods]
    label_keys = [f"label_{m}" for m in req_mods]
    all_load_keys = image_keys + label_keys

    transforms = []

    # 1) ✅ 显式 image_only=False，确保生成 *_meta_dict
    transforms.append(
        LoadImaged(keys=all_load_keys, image_only=False)
    )

    # 2) ✅ 立刻收集 raw meta（此时 *_meta_dict 一定还在）
    transforms.append(
        CollectMetaInfod(keys=all_load_keys)
    )

    # 3) 修复维度 + 通道 + 方向
    transforms.extend([
        CheckAndFixDimensionsd(keys=all_load_keys),
        EnsureChannelFirstd(keys=all_load_keys),
        Orientationd(keys=all_load_keys, axcodes="RAS"),
    ])

    # 4) 统一尺寸（网络空间）
    interp_modes = ['trilinear'] * len(image_keys) + ['nearest'] * len(label_keys)
    transforms.append(
        Resized(
            keys=all_load_keys,
            spatial_size=cfg.target_size,
            mode=interp_modes
        )
    )

    # 5) 拼接模态
    transforms.extend([
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        ConcatItemsd(keys=label_keys, name="seg_label", dim=0),
    ])

    # 6) ✅ 清理原始 keys + *_meta_dict（保留我们整理的 d["meta"]）
    meta_dict_keys = [f"{k}_meta_dict" for k in all_load_keys]
    transforms.append(
        DeleteItemsd(keys=all_load_keys + meta_dict_keys)
    )

    # 7) 归一化
    transforms.append(
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        )
    )

    # 8) 数据增强（训练集）
    if stage == "train":
        transforms.extend([
            RandRotated(
                keys=["image", "seg_label"],
                range_x=0.5, range_y=0.5, range_z=0.5,
                prob=0.5,
                mode=["bilinear", "nearest"],
                padding_mode="border",
            ),
            RandFlipd(
                keys=["image", "seg_label"],
                prob=0.5, spatial_axis=[0, 1, 2]
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5)
        ])

    # 9) ✅ image/seg_label 纯 Tensor（不引入 MetaTensor）
    transforms.append(
        EnsureTyped(keys=["image", "seg_label"], track_meta=False)
    )

    return Compose(transforms)

# ==========================================
# 5. Loader 构建主函数
# ==========================================

def print_report(name, valid_list, failed_list):
    print(f"\n{'='*20} {name} 数据报告 {'='*20}")
    print(f"✅ 成功加载: {len(valid_list)} 例")
    print(f"❌ 读取失败: {len(failed_list)} 例")
    if len(failed_list) > 0:
        print("-" * 60)
        for fail in failed_list: 
            print(f"{str(fail['id']):<25} | {fail['reason']}")
        print("-" * 60)
    print("\n")

def get_loaders(cfg):
    root_dir = cfg.root_dir
    req_mods = cfg.use_modalities
    leapfrog_list = cfg.get("leapfrog", [])

    print(f"🚀 初始化 Loader | 目标尺寸: {cfg.target_size}")

    mei_valid, mei_fail = build_data_list(cfg.excel_configs.mei, root_dir, leapfrog_list, "All", req_mods, "Mei")
    gz_valid, gz_fail = build_data_list(cfg.excel_configs.gz, root_dir, leapfrog_list, "All", req_mods, "GZ")
    train_list = mei_valid + gz_valid
    train_fail = mei_fail + gz_fail
    print_report("Train Set", train_list, train_fail)

    dg_valid, dg_fail = build_data_list(cfg.excel_configs.dg, root_dir, leapfrog_list, "All", req_mods, "DG")
    print_report("Val Set", dg_valid, dg_fail)

    if len(train_list) == 0:
        raise ValueError("训练集为空")
    if len(dg_valid) == 0:
        raise ValueError("验证集为空")

    # ✅ Dataset 绝不返回 None（DDP 必需）
    train_ds = SafeDataset(data=train_list, transform=get_transforms(cfg, "train"), max_retry=50)
    val_ds = SafeDataset(data=dg_valid, transform=get_transforms(cfg, "val"), max_retry=50)

    pin_memory = bool(getattr(cfg, "pin_memory", True))
    num_workers = int(getattr(cfg, "num_workers", 4))

    # ✅ 关键：timeout 用来“抓卡死样本”（比如 SimpleITK 读到坏文件卡住）
    # timeout>0 只在 num_workers>0 时有效
    timeout = int(getattr(cfg, "loader_timeout", 120))  # 秒，建议 60~180

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_ignore_none,
        drop_last=True,  # ✅ 多卡强烈建议
        persistent_workers=(num_workers > 0),
        timeout=timeout if num_workers > 0 else 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_ignore_none,
        drop_last=True,  # ✅ 多卡也建议 True
        persistent_workers=(num_workers > 0),
        timeout=timeout if num_workers > 0 else 0,
    )

    # ❌ 不要再包 NonEmptyDataLoader（它会“跳过 batch”，DDP 必挂）
    return train_loader, val_loader

# ==========================================
# 6. 调试/诊断工具
# ==========================================
def check_batch_statistics(batch):
    # 现在 batch 永远不可能是 None，除非 Loader 真的没东西了
    images = batch["image"]
    labels = batch["seg_label"]
    ids = batch.get("id", ["Unknown"])
    
    img_np = images.detach().cpu().numpy()
    lbl_np = labels.detach().cpu().numpy()
    
    print(f"\n{'>'*5} Batch Diagnosis (ID: {ids[0]}...) {'>'*5}")
    print(f"Shape: Img {img_np.shape}, Lbl {lbl_np.shape}")
    
    if img_np.shape[1] != lbl_np.shape[1]:
        print(f"❌ 警告: 通道数不匹配! Image={img_np.shape[1]}, Label={lbl_np.shape[1]}")
    else:
        print(f"✅ 通道数对齐: {img_np.shape[1]} 模态")


if __name__ == "__main__":
    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader)).data
        
    train_loader, val_loader = get_loaders(cfg)
    
    for batch_data in train_loader:
        print(batch_data["image"].shape)
        print(batch_data["seg_label"].shape)
        
    for batch_data in val_loader:
        print(batch_data["image"].shape)
        print(batch_data["seg_label"].shape)