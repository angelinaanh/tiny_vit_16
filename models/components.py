"""Các "viên Lego" của ViT. Mỗi class làm đúng một việc.

Ký hiệu shape dùng xuyên suốt:
    B  = batch size                         (32)
    C  = số kênh ảnh vào                    (3)
    H,W= chiều cao/rộng ảnh                 (224, 224)
    P  = kích thước patch                   (16)
    N  = số patch = (H/P) * (W/P)           (14 * 14 = 196)
    T  = số token = N + 1 (thêm [CLS])      (197)
    D  = embed_dim                          (64)
    h  = num_heads                          (4)
    Dh = head_dim = D / h                   (16)
    F  = mlp_hidden                         (128)
    K  = num_classes                        (10)
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Tokenization: ảnh -> chuỗi patch embedding
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Bản gốc "An Image is Worth 16x16 Words": cắt ảnh thành patch PxP,
    duỗi phẳng mỗi patch rồi chiếu tuyến tính bằng nn.Linear.

        (B, C, H, W) --cắt patch--> (B, N, C*P*P) --nn.Linear--> (B, N, D)
        (B, 3, 224, 224)            (B, 196, 768)                (B, 196, 64)
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} không chia hết cho patch_size={patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size               # 14
        self.num_patches = self.grid_size ** 2                  # 196
        self.patch_dim = in_channels * patch_size * patch_size  # 768
        self.proj = nn.Linear(self.patch_dim, embed_dim)        # 768 -> 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"PatchEmbedding cần ảnh {self.image_size}x{self.image_size}, nhận {H}x{W}")
        P, G = self.patch_size, self.grid_size
        # (B, C, H, W) -> (B, C, G, P, G, P): tách mỗi trục không gian thành (lưới, trong-patch)
        x = x.reshape(B, C, G, P, G, P)
        # -> (B, G, G, C, P, P): đưa 2 trục lưới lên trước, nội dung patch ra sau
        x = x.permute(0, 2, 4, 1, 3, 5)
        # -> (B, N, C*P*P): mỗi patch thành một vector 768 chiều, patch đánh số theo hàng
        x = x.reshape(B, self.num_patches, self.patch_dim)
        return self.proj(x)                                     # (B, N, D)


class ConvStem(nn.Module):
    """Convolutional Stem (Xiao et al., 2021 - "Early Convolutions Help
    Transformers See Better"): thay bước cắt patch + nn.Linear bằng một chồng
    conv 3x3 stride 2. Vẫn ra đúng lưới 14x14 = 196 token, nên phần còn lại
    của ViT (position embedding, các block) giữ nguyên.

        (B, 3, 224, 224) -conv s2-> (B, 24, 112, 112) -conv s2-> (B, 48, 56, 56)
        -conv s2-> (B, 64, 28, 28) -conv s2-> (B, 64, 14, 14) -conv 1x1-> (B, D, 14, 14)
        -flatten-> (B, 196, D)
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int,
                 embed_dim: int, channels: List[int]):
        super().__init__()
        downsample = 2 ** len(channels)
        if downsample != patch_size:
            raise ValueError(f"ConvStem có {len(channels)} conv stride 2 -> giảm {downsample} lần, "
                             f"phải bằng patch_size={patch_size} để giữ lưới token")
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        layers: List[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Conv2d(c_in, embed_dim, kernel_size=1)   # trộn kênh -> D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)                       # (B, C_last, 14, 14)
        x = self.proj(x)                        # (B, D, 14, 14)
        B, D, Gh, Gw = x.shape
        if Gh * Gw != self.num_patches:
            raise ValueError(f"ConvStem tạo lưới {Gh}x{Gw}, kỳ vọng {self.grid_size}x{self.grid_size}")
        return x.flatten(2).transpose(1, 2)     # (B, D, N) -> (B, N, D)


# ---------------------------------------------------------------------------
# 2. Position: thêm [CLS] và position embedding học được
# ---------------------------------------------------------------------------
class ClsTokenPositionEmbedding(nn.Module):
    """Nối token [CLS] vào đầu chuỗi rồi cộng position embedding.

    Position embedding là một bảng tham số học được nn.Embedding(T, D),
    tra theo chỉ số vị trí 0..T-1 (0 = [CLS], 1..196 = patch theo thứ tự hàng).

        (B, N, D) --nối [CLS]--> (B, T, D) --+ pos_embed(0..T-1)--> (B, T, D)
        (B, 196, 64)             (B, 197, 64)                       (B, 197, 64)
    """

    def __init__(self, num_patches: int, embed_dim: int, dropout: float):
        super().__init__()
        self.num_tokens = num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))     # (1, 1, D)
        self.pos_embed = nn.Embedding(self.num_tokens, embed_dim)      # bảng (T, D)
        self.register_buffer("positions", torch.arange(self.num_tokens), persistent=False)  # (T,)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if N + 1 != self.num_tokens:
            raise ValueError(f"Nhận {N} patch nhưng bảng position có {self.num_tokens} vị trí")
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                    # (B, T, D)
        x = x + self.pos_embed(self.positions)            # (B, T, D) + (T, D) broadcast
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3. Attention
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention tự cài đặt.

        x (B, T, D) --qkv--> (B, T, 3D) --tách head--> q, k, v: (B, h, T, Dh)
        attn = softmax(q k^T / sqrt(Dh))                       : (B, h, T, T)
        out  = attn v                                          : (B, h, T, Dh)
        --gộp head--> (B, T, D) --proj--> (B, T, D)

    Đặt `save_attention = True` để giữ lại ma trận attn (B, h, T, T) ở
    `attention_map` - dùng cho trực quan hoá attention của [CLS].
    """

    def __init__(self, embed_dim: int, num_heads: int, qkv_bias: bool,
                 attn_dropout: float, proj_dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} không chia hết cho num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.attn_dropout = attn_dropout
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.save_attention = False
        self.attention_map: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h, Dh = self.num_heads, self.head_dim
        qkv = self.qkv(x)                                   # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, h, Dh)                   # (B, T, 3, h, Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B, h, T, Dh)
        q, k, v = qkv.unbind(0)                             # mỗi cái (B, h, T, Dh)
        drop_p = self.attn_dropout if self.training else 0.0

        if self.save_attention:
            # Đường tường minh: tính và giữ lại ma trận attention
            attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, T, T)
            attn = attn.softmax(dim=-1)                     # mỗi hàng cộng lại = 1
            self.attention_map = attn.detach()
            out = F.dropout(attn, drop_p, self.training) @ v  # (B, h, T, Dh)
        else:
            # Cùng phép toán nhưng dùng kernel tối ưu của PyTorch (nhanh hơn khi train)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)  # (B, h, T, Dh)

        out = out.transpose(1, 2).reshape(B, T, D)          # (B, T, h, Dh) -> (B, T, D)
        return self.proj_drop(self.proj(out))               # (B, T, D)


class MLP(nn.Module):
    """Feed-forward theo từng token: (B, T, D) -> (B, T, F) -> GELU -> (B, T, D)."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))    # (B, T, F)
        return self.drop(self.fc2(x))           # (B, T, D)


class LocalityFeedForward(nn.Module):
    """LeFF - Locality-enhanced Feed-Forward (CeiT, LocalViT): MLP có thêm một
    depthwise conv kxk giữa fc1 và fc2. Các patch token được xếp lại thành lưới
    14x14 để mỗi token trộn thông tin với 8 patch láng giềng ở MỌI layer, thay
    vì chỉ dựa vào attention toàn cục. Token [CLS] không có vị trí trên lưới nên
    đi thẳng qua (bỏ qua conv).

        x (B, T, D) --fc1--> (B, T, F)
          tách:  [CLS] (B, 1, F)  |  patch (B, N, F)
          patch --xếp lưới--> (B, F, G, G) --DWConv kxk--> (B, F, G, G) --duỗi--> (B, N, F)
          nối lại [CLS] --> (B, T, F) --GELU--> --fc2--> (B, T, D)

    Shape vào = shape ra = (B, T, D), nên thay được trực tiếp cho MLP.
    Tham số thêm so với MLP: F*k*k + F (= 128*9 + 128 = 1,280 với k = 3).
    """

    def __init__(self, embed_dim: int, hidden_dim: int, grid_size: int, kernel_size: int,
                 dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"leff_kernel_size phải lẻ để giữ nguyên lưới, nhận {kernel_size}")
        self.grid_size = grid_size
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2,
                                groups=hidden_dim)                    # mỗi kênh một kernel kxk riêng
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        G = self.grid_size
        if T != G * G + 1:
            raise ValueError(f"LeFF cần T = 1 + {G}x{G} token, nhận T = {T}")
        x = self.fc1(x)                                     # (B, T, F)
        hidden = x.shape[-1]
        cls, patches = x[:, :1], x[:, 1:]                   # (B, 1, F), (B, N, F)
        # patch k nằm ở hàng k // G, cột k % G (cùng thứ tự với PatchEmbedding / ConvStem)
        grid = patches.transpose(1, 2).reshape(B, hidden, G, G)       # (B, F, G, G)
        patches = self.dwconv(grid).flatten(2).transpose(1, 2)        # (B, N, F)
        x = torch.cat([cls, patches], dim=1)                # (B, T, F)
        x = self.drop(self.act(x))                          # (B, T, F)
        return self.drop(self.fc2(x))                       # (B, T, D)


class TransformerBlock(nn.Module):
    """Một encoder block kiểu pre-norm (như ViT):

        x = x + attn(LayerNorm(x))     (B, T, D)
        x = x + mlp(LayerNorm(x))      (B, T, D)

    Block chỉ lo cấu trúc residual + norm; khối attention và khối feed-forward
    được truyền vào (MLP hoặc LocalityFeedForward), nên đổi loại FFN không cần
    sửa class này. Shape vào = shape ra = (B, T, D) -> có thể xếp chồng tuỳ ý.
    """

    def __init__(self, embed_dim: int, attn: nn.Module, mlp: nn.Module):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = attn
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))        # (B, T, D)
        x = x + self.mlp(self.norm2(x))         # (B, T, D)
        return x


# ---------------------------------------------------------------------------
# 4. Classifier
# ---------------------------------------------------------------------------
class ClassifierHead(nn.Module):
    """MLP head ánh xạ vector của [CLS] sang K lớp.

        hidden_dim = 0 : (B, D) --Linear--> (B, K)                      (như ViT gốc)
        hidden_dim > 0 : (B, D) --Linear--> (B, hidden) --GELU--> --Linear--> (B, K)
    """

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_vector: torch.Tensor) -> torch.Tensor:
        return self.net(cls_vector)             # (B, D) -> (B, K)
