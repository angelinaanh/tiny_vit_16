"""Biến ma trận attention của các layer thành bản đồ chú ý của token [CLS]."""
from typing import List

import torch
import torch.nn.functional as F


def _fuse_heads(attn: torch.Tensor, head_fusion: str) -> torch.Tensor:
    """(B, h, T, T) -> (B, T, T)"""
    if head_fusion == "mean":
        return attn.mean(dim=1)
    if head_fusion == "max":
        return attn.max(dim=1).values
    raise ValueError(f"head_fusion không hỗ trợ: '{head_fusion}'")


def cls_attention(maps: List[torch.Tensor], method: str, head_fusion: str) -> torch.Tensor:
    """Mức chú ý của [CLS] lên từng patch.

    maps: list `depth` phần tử, mỗi cái (B, h, T, T)
    trả về: (B, N) với N = T - 1 patch

    last_layer: hàng [CLS] của attention ở layer cuối.
    rollout   : Attention Rollout (Abnar & Zuidema, 2020) - nhân dồn attention
                qua mọi layer, cộng thêm ma trận đơn vị để mô phỏng residual.
    """
    if method == "last_layer":
        attn = _fuse_heads(maps[-1], head_fusion)              # (B, T, T)
        return attn[:, 0, 1:]                                  # (B, N)
    if method == "rollout":
        B, _, T, _ = maps[0].shape
        eye = torch.eye(T, device=maps[0].device).expand(B, T, T)
        rollout = eye.clone()                                  # (B, T, T)
        for attn in maps:
            a = _fuse_heads(attn, head_fusion)                 # (B, T, T)
            a = 0.5 * a + 0.5 * eye                            # residual
            a = a / a.sum(dim=-1, keepdim=True)                # chuẩn hoá lại theo hàng
            rollout = a @ rollout                              # (B, T, T)
        return rollout[:, 0, 1:]                               # (B, N)
    raise ValueError(f"method không hỗ trợ: '{method}'")


def to_heatmap(cls_attn: torch.Tensor, grid_size: int, image_size: int) -> torch.Tensor:
    """(B, N) -> (B, image_size, image_size), chuẩn hoá về [0, 1] cho từng ảnh."""
    B = cls_attn.shape[0]
    grid = cls_attn.reshape(B, 1, grid_size, grid_size)        # (B, 1, 14, 14)
    up = F.interpolate(grid, size=(image_size, image_size), mode="bilinear", align_corners=False)
    up = up[:, 0]                                              # (B, 224, 224)
    lo = up.flatten(1).min(dim=1).values.view(B, 1, 1)
    hi = up.flatten(1).max(dim=1).values.view(B, 1, 1)
    return (up - lo) / (hi - lo + 1e-8)
