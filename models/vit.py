"""Ghép các block trong components.py thành ViT hoàn chỉnh.

Class ViT ở đây không tự tính toán gì phức tạp: nó chỉ sắp xếp thứ tự

    ảnh --stem--> patch tokens --[CLS]+position--> tokens
        --TransformerBlock x depth--> LayerNorm --lấy [CLS]--> ClassifierHead --> logits

    (B, 3, 224, 224) -> (B, 196, 64) -> (B, 197, 64) -> (B, 197, 64) -> (B, 64) -> (B, 10)
"""
from contextlib import contextmanager
from typing import Iterator, List

import torch
import torch.nn as nn

from configs import ModelConfig
from .components import (MLP, ClassifierHead, ClsTokenPositionEmbedding, ConvStem,
                         LocalityFeedForward, MultiHeadSelfAttention, PatchEmbedding,
                         TransformerBlock)


def build_stem(cfg: ModelConfig, image_size: int) -> nn.Module:
    if cfg.stem == "linear":
        return PatchEmbedding(image_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim)
    if cfg.stem == "conv":
        return ConvStem(image_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim,
                        cfg.conv_stem_channels)
    raise ValueError(f"model.stem không hợp lệ: '{cfg.stem}' (chọn 'linear' hoặc 'conv')")


def build_mlp(cfg: ModelConfig, grid_size: int) -> nn.Module:
    if cfg.mlp_type == "standard":
        return MLP(cfg.embed_dim, cfg.mlp_hidden, cfg.dropout)
    if cfg.mlp_type == "leff":
        return LocalityFeedForward(cfg.embed_dim, cfg.mlp_hidden, grid_size,
                                   cfg.leff_kernel_size, cfg.dropout)
    raise ValueError(f"model.mlp_type không hợp lệ: '{cfg.mlp_type}' (chọn 'standard' hoặc 'leff')")


def build_block(cfg: ModelConfig, grid_size: int) -> TransformerBlock:
    attn = MultiHeadSelfAttention(cfg.embed_dim, cfg.num_heads, cfg.qkv_bias,
                                  cfg.attn_dropout, cfg.dropout)
    return TransformerBlock(cfg.embed_dim, attn, build_mlp(cfg, grid_size))


class ViT(nn.Module):
    def __init__(self, cfg: ModelConfig, image_size: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.stem = build_stem(cfg, image_size)
        self.grid_size = self.stem.grid_size
        self.tokens = ClsTokenPositionEmbedding(self.stem.num_patches, cfg.embed_dim, cfg.dropout)
        self.blocks = nn.ModuleList([build_block(cfg, self.grid_size) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = ClassifierHead(cfg.embed_dim, num_classes, cfg.head_hidden, cfg.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo ngẫu nhiên kiểu ViT (không có bất kỳ trọng số pretrained nào)."""
        std = self.cfg.init_std
        nn.init.trunc_normal_(self.tokens.cls_token, std=std)
        nn.init.trunc_normal_(self.tokens.pos_embed.weight, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)                   # (B, 3, 224, 224) -> (B, N, D)
        x = self.tokens(x)                      # (B, N, D) -> (B, T, D)
        for block in self.blocks:
            x = block(x)                        # (B, T, D) -> (B, T, D)
        return self.norm(x)                     # (B, T, D)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(images)       # (B, T, D)
        cls_vector = x[:, 0]                    # (B, D)   token [CLS] ở vị trí 0
        return self.head(cls_vector)            # (B, K)   logits

    @contextmanager
    def capture_attention(self) -> Iterator[List[torch.Tensor]]:
        """Ghi lại ma trận attention của từng layer trong lần forward cuối cùng.

            with model.capture_attention() as maps:
                model(images)
            # sau khi ra khỏi khối with:
            # maps: list dài `depth`, mỗi phần tử (B, h, T, T)
        """
        attns = [blk.attn for blk in self.blocks]
        maps: List[torch.Tensor] = []
        for a in attns:
            a.save_attention, a.attention_map = True, None
        try:
            yield maps
        finally:
            maps.extend(a.attention_map for a in attns if a.attention_map is not None)
            for a in attns:
                a.save_attention, a.attention_map = False, None


def build_model(cfg: ModelConfig, image_size: int, num_classes: int) -> ViT:
    return ViT(cfg, image_size, num_classes)
