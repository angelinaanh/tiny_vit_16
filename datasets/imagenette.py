"""Dataset -> Tiền xử lý -> DataLoader cho Imagenette.

Module này CHỈ lo dữ liệu: tải ảnh, biến đổi ảnh (augmentation/chuẩn hoá) và
đóng gói thành DataLoader. Không biết gì về model hay quá trình huấn luyện.

Hợp đồng shape ở đầu ra mỗi batch:
    images : (B, 3, 224, 224)  float32, đã chuẩn hoá theo mean/std
    labels : (B,)              int64, giá trị trong [0, num_classes)
"""
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import Imagenette

from configs import DataConfig


def build_transforms(cfg: DataConfig, train: bool) -> T.Compose:
    """Tiền xử lý: đưa mọi ảnh (cạnh ngắn 320px, tỉ lệ bất kỳ) về 3 x 224 x 224.

    train: RandomResizedCrop(224) + lật ngang  -> augmentation, chống overfit
    val  : Resize(256) + CenterCrop(224)        -> tất định, đánh giá công bằng
    """
    if train:
        geometric = [
            T.RandomResizedCrop(cfg.image_size, scale=tuple(cfg.rrc_scale)),
            T.RandomHorizontalFlip(p=cfg.hflip_prob),
        ]
    else:
        geometric = [
            T.Resize(cfg.resize_size),
            T.CenterCrop(cfg.image_size),
        ]
    return T.Compose(geometric + [
        T.ToTensor(),                          # PIL (H, W, 3) uint8 -> (3, H, W) float [0, 1]
        T.Normalize(cfg.mean, cfg.std),        # (3, H, W) -> (3, H, W), mỗi kênh ~ N(0, 1)
    ])


def _load_split(cfg: DataConfig, split: str, transform: T.Compose) -> Imagenette:
    try:
        return Imagenette(cfg.root, split=split, size=cfg.size,
                          download=cfg.download, transform=transform)
    except RuntimeError:
        # torchvision báo lỗi nếu download=True mà thư mục đã tồn tại -> dùng bản có sẵn
        return Imagenette(cfg.root, split=split, size=cfg.size,
                          download=False, transform=transform)


def build_datasets(cfg: DataConfig) -> Tuple[Dataset, Dataset]:
    train_ds = _load_split(cfg, "train", build_transforms(cfg, train=True))
    val_ds = _load_split(cfg, "val", build_transforms(cfg, train=False))
    n_cls = len(train_ds.classes)
    if n_cls != cfg.num_classes:
        raise ValueError(f"Imagenette có {n_cls} lớp nhưng config data.num_classes={cfg.num_classes}")
    return train_ds, val_ds


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(cfg)
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def class_names(ds: Dataset) -> List[str]:
    """Tên ngắn của từng lớp, ví dụ ('tench', 'Tinca tinca') -> 'tench'."""
    base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    return [c[0] if isinstance(c, (tuple, list)) else str(c) for c in base.classes]
