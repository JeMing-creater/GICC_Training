"""
GICC 结肠癌 MRI 数据集 Loader
===============================
支持从 T.xlsx 读取标签，按 train/test 标记划分数据集，
并输出包含图像、病灶勾画和二分类标签的 batch。

Author: GICC Team
"""

import os
import yaml
import pandas as pd
import numpy as np
import torch
from easydict import EasyDict
from sklearn.model_selection import train_test_split
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
    RandScaleIntensityd,
    ToTensord,
)
import random


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
                    continue
            except StopIteration:
                break

    def __len__(self):
        return len(self.loader)


# ==========================================
# 2. 自定义 Transforms & Dataset
# ==========================================


class CheckAndFixDimensionsd(MapTransform):
    """
    【修复 4D 错误】
    检查图像和标签维度，将伪4D数据降维。
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key not in d:
                continue
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


class SafeDataset(MonaiDataset):
    """
    【安全 Dataset - DDP 友好版】
    绝不返回 None（DDP 下任何 rank 少一步都会 NCCL timeout）。
    遇到异常时，自动尝试其它 index，最多 retry 次。
    如果连续失败，抛出异常（宁可早失败，也不要训练中 NCCL hang）。
    """

    def __init__(self, *args, max_retry: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retry = int(max_retry)

    def __getitem__(self, index):
        last_e = None
        n = len(self.data)

        # 以 index 为起点，线性探测后续样本，最多 max_retry 次
        for k in range(self.max_retry):
            idx = (index + k) % n
            try:
                return super().__getitem__(idx)
            except Exception as e:
                last_e = e
                sample_info = self.data[idx]
                sample_id = sample_info.get("id", "Unknown")
                if k == 0:
                    print(f"\n❌ [Read Error] idx={idx} ID={sample_id} | {str(e)}")
                continue

        raise RuntimeError(
            f"[SafeDataset] Failed to fetch a valid sample after {self.max_retry} retries. "
            f"Last error: {str(last_e)}"
        )


def collate_fn_ignore_none(batch):
    """
    【DDP 友好 Collate】
    训练时不允许返回 None（否则不同 rank step 数不一致 -> NCCL hang）。
    如果 batch 里出现 None，直接过滤；若过滤后为空，抛异常快速暴露问题。
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        raise RuntimeError(
            "[collate_fn_ignore_none] Empty batch after filtering None. "
            "This will break DDP. Please check dataset/transform errors."
        )
    from monai.data import list_data_collate

    return list_data_collate(batch)


def collate_fn_with_class_label(batch):
    """
    【带分类标签的 Collate 函数】
    将 class_label 转换为 (batch_size, 1) 形状的 Tensor。
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        raise RuntimeError(
            "[collate_fn_with_class_label] Empty batch after filtering None."
        )

    from monai.data import list_data_collate

    result = list_data_collate(batch)

    # 处理 class_label: 从 batch 中收集并转为 (B, 1) 形状
    if "class_label" in result:
        class_labels = [item["class_label"] for item in batch]
        # 处理可能的嵌套列表或单个值
        processed_labels = []
        for label in class_labels:
            if isinstance(label, (list, np.ndarray, torch.Tensor)):
                if isinstance(label, torch.Tensor):
                    label = label.item() if label.numel() == 1 else label.tolist()
                elif isinstance(label, np.ndarray):
                    label = label.item() if label.size == 1 else label.tolist()
                elif isinstance(label, list):
                    label = label[0] if len(label) == 1 else label
            label = int(label)
            processed_labels.append(label)

        result["class_label"] = torch.tensor(
            processed_labels, dtype=torch.long
        ).unsqueeze(-1)

    return result


# ==========================================
# 3. 路径查找与 ID 匹配逻辑
# ==========================================

FOLDER_ALIASES = {
    "T2_FS": ["T2_FS", "T2", "t2", "T2FS"],
    "ADC": ["ADC", "adc", "Adc"],
    "V": ["V", "v", "Venous", "venous"],
}


def find_modality_path(patient_folder, modality_name):
    """
    查找模态文件夹路径，支持别名匹配。
    """
    candidates = FOLDER_ALIASES.get(modality_name, [modality_name])
    for alias in candidates:
        target_path = os.path.join(patient_folder, alias)
        if os.path.exists(target_path):
            return target_path, alias
    return None, None


def build_folder_index(data_folder):
    """
    【建立文件夹索引】
    扫描 data_folder 下的子文件夹，建立 ID -> 文件夹名 的映射。
    文件夹名可能带前导零，ID 需要去掉前导零进行匹配。

    Args:
        data_folder: 数据根目录 (如 /mnt/liangjm/GICC/All)

    Returns:
        dict: {clean_id: real_folder_name}
    """
    if not os.path.exists(data_folder):
        print(f"⚠️ 数据文件夹不存在: {data_folder}")
        return {}

    index = {}
    try:
        subfolders = [
            f
            for f in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, f))
        ]
    except Exception as e:
        print(f"⚠️ 扫描文件夹失败: {e}")
        return {}

    for real_name in subfolders:
        # 去掉前导零得到 clean_id
        clean_key = real_name.lstrip("0")
        if clean_key == "":
            clean_key = "0"
        index[clean_key] = real_name

    print(f"✅ 已索引 {len(index)} 个病人文件夹")
    return index


def validate_patient_data(data_folder, real_folder_name, required_modalities, excel_id):
    """
    【验证患者数据完整性】
    检查每个模态的图像和标签文件是否存在。

    Args:
        data_folder: 数据根目录
        real_folder_name: 实际文件夹名 (可能带前导零)
        required_modalities: 需要的模态列表
        excel_id: Excel 中的原始 ID

    Returns:
        (is_valid, data_entry, error_msg)
    """
    p_path = os.path.join(data_folder, real_folder_name)
    data_entry = {
        "id": real_folder_name,  # 使用文件夹名作为主键
        "excel_id": excel_id,  # 保存原始 Excel ID
    }

    for mod in required_modalities:
        mod_folder, matched_alias = find_modality_path(p_path, mod)
        if mod_folder is None:
            return False, {}, f"缺失模态文件夹: {mod}"

        # 图像文件: {ID}.nii.gz
        img_file = os.path.join(mod_folder, f"{real_folder_name}.nii.gz")
        if not os.path.exists(img_file):
            return False, {}, f"缺失图像 {mod}: {img_file}"

        # 标签文件: {ID}seg.nii.gz
        seg_file = os.path.join(mod_folder, f"{real_folder_name}seg.nii.gz")
        if not os.path.exists(seg_file):
            return False, {}, f"缺失标签 {mod}: {seg_file}"

        data_entry[f"image_{mod}"] = img_file
        data_entry[f"label_{mod}"] = seg_file

    return True, data_entry, None


def read_excel_and_split(cfg):
    """
    【读取 Excel 并划分数据集】

    读取 T.xlsx，按 D 列 (train_tag_col) 的 "train" 标记划分：
    - 有 "train" 标记 → 训练候选集
    - 无标记 (NaN) → 测试集

    训练候选集按 val_split_ratio 划分训练/验证集。

    Returns:
        (train_list, val_list, test_list, failed_list)
        每个元素为 dict: {"id": str, "excel_id": str, "class_label": int, ...modal_paths}
    """
    t_cfg = cfg.data.t_dataset
    root_dir = cfg.data.root_dir

    excel_path = os.path.join(root_dir, t_cfg.excel_filename)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")

    # 读取 Excel
    df = pd.read_excel(excel_path)
    print(f"\n📊 T.xlsx 读取完成: {len(df)} 条记录")

    # 提取列
    id_col = int(t_cfg.id_col)
    label_col = int(t_cfg.label_col)
    train_tag_col = int(t_cfg.train_tag_col)
    train_tag_value = str(t_cfg.train_tag_value)
    val_split_ratio = float(t_cfg.val_split_ratio)

    print(f"   ID列: {id_col}, 标签列: {label_col}, 训练标记列: {train_tag_col}")
    print(f"   训练标记值: '{train_tag_value}'")
    print(f"   验证集划分比例: {val_split_ratio}")

    # 构建文件夹索引
    all_folder = os.path.join(root_dir, "All")
    folder_index = build_folder_index(all_folder)

    if not folder_index:
        raise ValueError("文件夹索引为空，请检查数据路径")

    # 遍历 Excel，建立样本列表
    train_candidates = []  # 有 train 标记的
    test_list = []  # 无 train 标记的
    failed_list = []  # 读取失败的

    req_mods = cfg.data.use_modalities

    for idx, row in df.iterrows():
        excel_id = str(row.iloc[id_col])
        class_label = int(row.iloc[label_col])
        train_tag = (
            str(row.iloc[train_tag_col]) if pd.notna(row.iloc[train_tag_col]) else ""
        )

        # 匹配文件夹：ID 补零到 10 位
        padded_id = excel_id.zfill(10)
        real_folder = folder_index.get(excel_id.lstrip("0"), padded_id)

        # 验证数据完整性
        is_valid, data_entry, error_msg = validate_patient_data(
            all_folder, real_folder, req_mods, excel_id
        )

        if not is_valid:
            failed_list.append(
                {"id": real_folder, "excel_id": excel_id, "reason": error_msg}
            )
            continue

        # 添加分类标签
        data_entry["class_label"] = class_label

        # 按 train 标记分类
        if train_tag.strip().lower() == train_tag_value.strip().lower():
            train_candidates.append(data_entry)
        else:
            test_list.append(data_entry)

    print(f"\n📋 数据划分统计:")
    print(f"   训练候选: {len(train_candidates)}")
    print(f"   测试集: {len(test_list)}")
    print(f"   读取失败: {len(failed_list)}")

    # 划分训练/验证集（分层抽样）
    if len(train_candidates) > 0:
        train_ids = [x["excel_id"] for x in train_candidates]
        train_labels = [x["class_label"] for x in train_candidates]

        train_indices, val_indices = train_test_split(
            range(len(train_candidates)),
            test_size=val_split_ratio,
            stratify=train_labels,
            random_state=42,
        )

        train_list = [train_candidates[i] for i in train_indices]
        val_list = [train_candidates[i] for i in val_indices]

        print(f"   训练集: {len(train_list)} (验证集划分后)")
        print(f"   验证集: {len(val_list)}")

        # 打印类别分布
        train_pos = sum(x["class_label"] for x in train_list)
        train_neg = len(train_list) - train_pos
        val_pos = sum(x["class_label"] for x in val_list)
        val_neg = len(val_list) - val_pos

        print(f"   训练集类别分布: 阳性={train_pos}, 阴性={train_neg}")
        print(f"   验证集类别分布: 阳性={val_pos}, 阴性={val_neg}")
    else:
        train_list = []
        val_list = []

    return train_list, val_list, test_list, failed_list


# ==========================================
# 4. Transforms
# ==========================================


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
            "items": {},
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


def get_transforms(cfg, stage="train"):
    req_mods = cfg.use_modalities
    image_keys = [f"image_{m}" for m in req_mods]
    label_keys = [f"label_{m}" for m in req_mods]
    all_load_keys = image_keys + label_keys

    transforms = []

    # 1) ✅ 显式 image_only=False，确保生成 *_meta_dict
    transforms.append(LoadImaged(keys=all_load_keys, image_only=False))

    # 2) ✅ 立刻收集 raw meta（此时 *_meta_dict 一定还在）
    transforms.append(CollectMetaInfod(keys=all_load_keys))

    # 3) 修复维度 + 通道 + 方向
    transforms.extend(
        [
            CheckAndFixDimensionsd(keys=all_load_keys),
            EnsureChannelFirstd(keys=all_load_keys),
            Orientationd(keys=all_load_keys, axcodes="RAS"),
        ]
    )

    # 4) 统一尺寸（网络空间）
    interp_modes = ["trilinear"] * len(image_keys) + ["nearest"] * len(label_keys)
    transforms.append(
        Resized(keys=all_load_keys, spatial_size=cfg.target_size, mode=interp_modes)
    )

    # 5) 拼接模态
    transforms.extend(
        [
            ConcatItemsd(keys=image_keys, name="image", dim=0),
            ConcatItemsd(keys=label_keys, name="seg_label", dim=0),
        ]
    )

    # 6) ✅ 清理原始 keys + *_meta_dict（保留我们整理的 d["meta"]）
    meta_dict_keys = [f"{k}_meta_dict" for k in all_load_keys]
    transforms.append(DeleteItemsd(keys=all_load_keys + meta_dict_keys))

    # 7) 归一化
    transforms.append(
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        )
    )

    # 8) 数据增强（训练集）
    if stage == "train":
        transforms.extend(
            [
                RandRotated(
                    keys=["image", "seg_label"],
                    range_x=0.5,
                    range_y=0.5,
                    range_z=0.5,
                    prob=0.5,
                    mode=["bilinear", "nearest"],
                    padding_mode="border",
                ),
                RandFlipd(
                    keys=["image", "seg_label"], prob=0.5, spatial_axis=[0, 1, 2]
                ),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            ]
        )

    # 9) ✅ image/seg_label 纯 Tensor（不引入 MetaTensor）
    transforms.append(EnsureTyped(keys=["image", "seg_label"], track_meta=False))

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
    """
    【主函数：从 T.xlsx 读取并构建 Loader】

    返回:
        train_loader, val_loader, test_loader
    """
    # 检查是否有 t_dataset 配置
    if not hasattr(cfg.data, "t_dataset"):
        raise ValueError(
            "config.yml 中缺少 data.t_dataset 配置，请添加 T.xlsx 相关配置"
        )

    print(f"\n🚀 初始化 Loader (T.xlsx 模式) | 目标尺寸: {cfg.data.target_size}")

    # 读取 Excel 并划分数据集
    train_list, val_list, test_list, failed_list = read_excel_and_split(cfg)

    # 打印报告
    print_report("Train Set", train_list, failed_list)
    if val_list:
        print_report("Val Set", val_list, [])
    if test_list:
        print_report("Test Set", test_list, [])

    # 检查数据集是否为空
    if len(train_list) == 0:
        raise ValueError("训练集为空")

    # 构建 Dataset
    train_ds = SafeDataset(
        data=train_list, transform=get_transforms(cfg.data, "train"), max_retry=50
    )

    val_ds = None
    if len(val_list) > 0:
        val_ds = SafeDataset(
            data=val_list, transform=get_transforms(cfg.data, "val"), max_retry=50
        )

    test_ds = None
    if len(test_list) > 0:
        test_ds = SafeDataset(
            data=test_list, transform=get_transforms(cfg.data, "val"), max_retry=50
        )

    # DataLoader 参数
    pin_memory = bool(getattr(cfg.data, "pin_memory", True))
    num_workers = int(getattr(cfg.data, "num_workers", 4))
    batch_size = int(getattr(cfg.data, "batch_size", 2))
    timeout = int(getattr(cfg.data, "loader_timeout", 120))

    # 构建 DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_with_class_label,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        timeout=timeout if num_workers > 0 else 0,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn_with_class_label,
            drop_last=True,
            persistent_workers=(num_workers > 0),
            timeout=timeout if num_workers > 0 else 0,
        )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn_with_class_label,
            drop_last=False,
            persistent_workers=(num_workers > 0),
            timeout=timeout if num_workers > 0 else 0,
        )

    return train_loader, val_loader, test_loader


# ==========================================
# 6. 调试/诊断工具
# ==========================================


def check_batch_statistics(batch, batch_idx=0):
    """打印 batch 详细信息"""
    images = batch["image"]
    labels = batch["seg_label"]
    class_labels = batch["class_label"]
    ids = batch.get("id", ["Unknown"])

    img_np = images.detach().cpu().numpy()
    lbl_np = labels.detach().cpu().numpy()
    cls_np = class_labels.detach().cpu().numpy()

    print(f"\n{'='*20} Batch #{batch_idx} 统计 {'='*20}")
    print(f"IDs: {ids[:3]}...")
    print(f"image shape: {img_np.shape} (B, modal, W, H, Z)")
    print(f"seg_label shape: {lbl_np.shape} (B, modal, W, H, Z)")
    print(f"class_label shape: {cls_np.shape} (B, 1)")
    print(f"class_label values: {cls_np.flatten()[:5].tolist()}...")

    # 数值范围
    print(f"image range: [{img_np.min():.3f}, {img_np.max():.3f}]")
    print(f"seg_label range: [{lbl_np.min():.3f}, {lbl_np.max():.3f}]")

    # 通道数检查
    if img_np.shape[1] != lbl_np.shape[1]:
        print(
            f"❌ 警告: 通道数不匹配! Image={img_np.shape[1]}, Label={lbl_np.shape[1]}"
        )
    else:
        print(f"✅ 通道数对齐: {img_np.shape[1]} 模态")

    # 类别分布
    unique, counts = np.unique(cls_np, return_counts=True)
    print(f"类别分布: {dict(zip(unique.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GICC Loader 调试模式")
    print("=" * 60)

    # 加载配置
    with open("/workspace/GICC_training/config.yml", "r", encoding="utf-8") as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # 确保 t_dataset 配置存在
    if not hasattr(cfg.data, "t_dataset"):
        print("\n⚠️ config.yml 中缺少 data.t_dataset 配置，自动添加默认值...")
        cfg.data.t_dataset = EasyDict(
            {
                "excel_filename": "T.xlsx",
                "id_col": 0,
                "label_col": 2,
                "train_tag_col": 3,
                "train_tag_value": "train",
                "val_split_ratio": 0.1,
            }
        )

    print(f"\n📋 T-Dataset 配置:")
    for k, v in cfg.data.t_dataset.items():
        print(f"  {k}: {v}")

    # 初始化 Loader
    print("\n" + "=" * 60)
    print("初始化数据加载器...")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_loaders(cfg)

    # 遍历训练集 1-2 个 batch
    print("\n" + "=" * 60)
    print("遍历训练集 (1-2 个 batch)...")
    print("=" * 60)

    batch_count = 0
    for batch in train_loader:
        check_batch_statistics(batch, batch_count)
        # batch_count += 1
        # if batch_count >= 2:
        #     break

    # 如果有验证集，遍历 1 个 batch
    if val_loader is not None:
        print("\n" + "=" * 60)
        print("遍历验证集 (1 个 batch)...")
        print("=" * 60)
        # for batch in val_loader:
        #     check_batch_statistics(batch, 0)
        #     break

    # 如果有测试集，遍历 1 个 batch
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("遍历测试集 (1 个 batch)...")
        print("=" * 60)
        # for batch in test_loader:
        #     check_batch_statistics(batch, 0)
        #     break

    print("\n✅ 调试完成!")
