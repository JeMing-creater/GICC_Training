import os
from sympy import im
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

class SafeDataset(MonaiDataset):
    """
    【安全 Dataset】
    捕获所有 Transform 中的错误，返回 None。
    """
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            sample_info = self.data[index]
            sample_id = sample_info.get('id', 'Unknown')
            # 打印简短的错误日志
            print(f"\n❌ [Read Error] 跳过样本 ID: {sample_id}")
            print(f"   原因: {str(e)}")
            return None

def collate_fn_ignore_none(batch):
    """
    【安全 Collate】过滤 None
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    from monai.data import list_data_collate
    return list_data_collate(batch)

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

    # 1. 加载 & 维度修复 (必须在最前面)
    transforms.extend([
        LoadImaged(keys=all_load_keys),
        CheckAndFixDimensionsd(keys=all_load_keys), # 修复4D数据
        EnsureChannelFirstd(keys=all_load_keys),
        Orientationd(keys=all_load_keys, axcodes="RAS"),
    ])

    # 2. 统一尺寸
    interp_modes = ['trilinear'] * len(image_keys) + ['nearest'] * len(label_keys)
    transforms.append(
        Resized(
            keys=all_load_keys, 
            spatial_size=cfg.target_size, 
            mode=interp_modes 
        )
    )

    # 3. 拼接模态
    transforms.extend([
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        ConcatItemsd(keys=label_keys, name="seg_label", dim=0),
        DeleteItemsd(keys=all_load_keys) 
    ])

    # 4. 强度归一化
    transforms.append(
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        )
    )

    # 5. 数据增强 (仅训练集)
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

    transforms.append(EnsureTyped(keys=["image", "seg_label"]))
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
    
    if len(train_list) == 0: raise ValueError("训练集为空")

    train_ds = SafeDataset(data=train_list, transform=get_transforms(cfg, "train"))
    
    val_ds = SafeDataset(data=dg_valid, transform=get_transforms(cfg, "val")) 
    
    train_ds = SafeDataset(data=train_list, transform=get_transforms(cfg, "train"))
    val_ds = SafeDataset(data=dg_valid, transform=get_transforms(cfg, "val")) 

    # 创建原始 Loader
    _train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, collate_fn=collate_fn_ignore_none
    )
    _val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, 
        num_workers=cfg.num_workers, collate_fn=collate_fn_ignore_none
    )

    # 【核心】包裹一层 NonEmptyDataLoader
    train_loader = NonEmptyDataLoader(_train_loader)
    val_loader = NonEmptyDataLoader(_val_loader)

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