"""Dựng Loss -> Optimizer -> LR scheduler từ config."""
import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from configs import LossConfig, OptimizerConfig, SchedulerConfig

# Tham số không nên áp weight decay: bias, LayerNorm/BatchNorm (đều 1 chiều),
# token [CLS] và bảng position embedding.
_NO_DECAY_KEYWORDS = ("cls_token", "pos_embed")


def build_loss(cfg: LossConfig) -> nn.Module:
    if cfg.name != "cross_entropy":
        raise ValueError(f"loss.name không hỗ trợ: '{cfg.name}'")
    # logits (B, K), labels (B,) -> loss vô hướng
    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


def build_optimizer(cfg: OptimizerConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.name != "adamw":
        raise ValueError(f"optimizer.name không hỗ trợ: '{cfg.name}'")
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name for k in _NO_DECAY_KEYWORDS):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas))


def build_scheduler(cfg: SchedulerConfig, optimizer: torch.optim.Optimizer, base_lr: float,
                    epochs: int, steps_per_epoch: int) -> LambdaLR:
    """Scheduler được gọi `.step()` sau MỖI iteration.

    cosine: tăng tuyến tính trong `warmup_epochs` rồi giảm theo cosine về `min_lr`.
    none  : giữ nguyên lr.
    """
    total = epochs * steps_per_epoch
    warmup = cfg.warmup_epochs * steps_per_epoch
    min_ratio = cfg.min_lr / base_lr

    if cfg.name == "none":
        return LambdaLR(optimizer, lambda step: 1.0)
    if cfg.name != "cosine":
        raise ValueError(f"scheduler.name không hỗ trợ: '{cfg.name}'")

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, factor)
