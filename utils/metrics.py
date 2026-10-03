"""Tính accuracy và các thống kê đánh giá."""
from typing import List, Sequence

import torch


class AverageMeter:
    """Trung bình có trọng số (theo số mẫu) của một đại lượng qua nhiều batch."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Sequence[int] = (1,)) -> List[float]:
    """Top-k accuracy (%) của một batch.

    logits : (B, K)
    targets: (B,)
    """
    maxk = max(topk)
    pred = logits.topk(maxk, dim=1).indices                 # (B, maxk)
    correct = pred.eq(targets.unsqueeze(1))                 # (B, maxk) bool
    return [correct[:, :k].any(dim=1).float().mean().item() * 100.0 for k in topk]


@torch.no_grad()
def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """preds, targets: (M,) -> ma trận (K, K), hàng = nhãn thật, cột = dự đoán."""
    idx = targets * num_classes + preds                     # (M,)
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
