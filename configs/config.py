"""Đọc configs/config.yaml thành các dataclass có kiểu rõ ràng.

Các dataclass ở đây CHỈ khai báo schema (tên + kiểu), KHÔNG có giá trị mặc
định -> giá trị thật chỉ tồn tại ở một nơi duy nhất là file YAML. Thiếu khoá
hoặc thừa khoá (gõ sai tên) đều báo lỗi ngay khi nạp config.
"""
import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, get_type_hints

import yaml


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str
    device: str
    num_threads: int


@dataclass
class DataConfig:
    root: str
    size: str
    download: bool
    image_size: int
    resize_size: int
    rrc_scale: List[float]
    hflip_prob: float
    mean: List[float]
    std: List[float]
    num_classes: int
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class ModelConfig:
    arch: str
    stem: str
    in_channels: int
    patch_size: int
    embed_dim: int
    num_heads: int
    depth: int
    mlp_hidden: int
    mlp_type: str
    leff_kernel_size: int
    qkv_bias: bool
    attn_dropout: float
    dropout: float
    head_hidden: int
    conv_stem_channels: List[int]
    init_std: float


@dataclass
class LossConfig:
    name: str
    label_smoothing: float


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    betas: List[float]


@dataclass
class SchedulerConfig:
    name: str
    warmup_epochs: int
    min_lr: float


@dataclass
class TrainConfig:
    epochs: int
    grad_clip: float
    log_every: int
    eval_before_training: bool


@dataclass
class VisualizeConfig:
    num_images: int
    head_fusion: str


@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    train: TrainConfig
    visualize: VisualizeConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build(cls, raw: Dict[str, Any], where: str):
    """Dựng dataclass `cls` từ dict `raw`, kiểm tra thiếu/thừa khoá."""
    if not isinstance(raw, dict):
        raise TypeError(f"[config] '{where}' phải là một mapping, nhận {type(raw).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    missing = names - raw.keys()
    unknown = raw.keys() - names
    if missing:
        raise KeyError(f"[config] '{where}' thiếu khoá: {sorted(missing)}")
    if unknown:
        raise KeyError(f"[config] '{where}' có khoá lạ (gõ sai?): {sorted(unknown)}")
    kwargs = {}
    for name in names:
        typ, val = hints[name], raw[name]
        sub = f"{where}.{name}" if where else name
        if is_dataclass(typ):
            kwargs[name] = _build(typ, val, sub)
        elif typ is float and isinstance(val, int):
            kwargs[name] = float(val)          # YAML "1" -> 1.0 cho trường float
        elif typ is float and isinstance(val, str):
            kwargs[name] = float(val)          # YAML "3e-4" (không có dấu chấm) bị đọc thành str
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _apply_override(raw: Dict[str, Any], override: str) -> None:
    """Áp một override dạng 'nhom.khoa=gia_tri' (giá trị được parse bằng YAML)."""
    if "=" not in override:
        raise ValueError(f"[config] override phải có dạng key=value, nhận '{override}'")
    key, value = override.split("=", 1)
    node = raw
    parts = key.strip().split(".")
    for p in parts[:-1]:
        if p not in node:
            raise KeyError(f"[config] override không hợp lệ, không có nhóm '{p}' trong '{key}'")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"[config] override không hợp lệ, không có khoá '{key}'")
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path: str = "configs/config.yaml",
                overrides: Optional[Sequence[str]] = None) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)
    for ov in overrides or []:
        _apply_override(raw, ov)
    return _build(Config, raw, "")


def save_config(cfg: Config, path: Path) -> None:
    """Lưu snapshot config đã resolve (kể cả override) cạnh checkpoint."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
