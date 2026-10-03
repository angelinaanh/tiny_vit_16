"""Vẽ sơ đồ so sánh kiến trúc. Sơ đồ được SINH TỪ CHÍNH MODEL: shape lấy từ một
lần forward thật (trace_shapes), số tham số đếm từ module, thông số (kênh, kernel,
số head...) đọc từ thuộc tính module - nên sơ đồ luôn khớp với code."""
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.patches import FancyBboxPatch

from models.components import ConvStem, LocalityFeedForward, MLP, PatchEmbedding
from .shapes import trace_shapes

_SAME = dict(face="#e6eef8", edge="#4a6fa5")      # khối giống bản gốc
_CHANGED = dict(face="#fde5cc", edge="#d9822b")   # khối được thay đổi
_IO = dict(face="#f2f2f2", edge="#8a8a8a")        # đầu vào / đầu ra


def _n(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _b(shape: str) -> str:
    """'(2, 196, 64)' -> '(B, 196, 64)'"""
    return re.sub(r"^\(\d+", "(B", shape)


def describe_stages(model: nn.Module, x: torch.Tensor) -> List[dict]:
    """Danh sách các tầng của một ViT: tiêu đề, mô tả, shape đầu ra, số tham số."""
    rows = {name: (inp, out) for name, _, inp, out in trace_shapes(model, x, max_depth=2)}
    D = model.cfg.embed_dim
    stages = [dict(key="input", title="Ảnh vào", lines=["RGB, đã Normalize"],
                   shape=_b(str(tuple(x.shape))), params=None, style=_IO)]

    stem = model.stem
    if isinstance(stem, PatchEmbedding):
        lines = [f"cắt {stem.num_patches} patch {stem.patch_size}×{stem.patch_size} → vector {stem.patch_dim}",
                 f"nn.Linear({stem.proj.in_features} → {stem.proj.out_features})"]
    elif isinstance(stem, ConvStem):
        convs = [m for m in stem.convs if isinstance(m, nn.Conv2d)]
        chans = "→".join([str(convs[0].in_channels)] + [str(c.out_channels) for c in convs])
        lines = [f"{len(convs)}× Conv3×3 stride 2 + BN + ReLU", f"kênh {chans}",
                 f"Conv1×1 → {stem.proj.out_channels}, duỗi lưới {stem.grid_size}×{stem.grid_size}"]
    else:
        lines = [type(stem).__name__]
    stages.append(dict(key=type(stem).__name__, title=type(stem).__name__, lines=lines,
                       shape=_b(rows["stem"][1]), params=_n(stem)))

    tok = model.tokens
    stages.append(dict(key="tokens", title="[CLS] + Position embedding",
                       lines=[f"nối [CLS] → {tok.num_tokens} token",
                              f"+ nn.Embedding({tok.num_tokens}, {D})"],
                       shape=_b(rows["tokens"][1]), params=_n(tok)))

    blk = model.blocks[0]
    mlp = blk.mlp
    if isinstance(mlp, LocalityFeedForward):
        k = mlp.dwconv.kernel_size[0]
        ffn = f"LN → LeFF {mlp.fc1.in_features}→{mlp.fc1.out_features}→DWConv{k}×{k}→{mlp.fc2.out_features}"
    elif isinstance(mlp, MLP):
        ffn = f"LN → MLP {mlp.fc1.in_features}→{mlp.fc1.out_features}→{mlp.fc2.out_features}"
    else:
        ffn = f"LN → {type(mlp).__name__}"
    depth = len(model.blocks)
    stages.append(dict(key=f"block:{type(mlp).__name__}", title=f"TransformerBlock × {depth}",
                       lines=[f"LN → MHSA ({blk.attn.num_heads} head × {blk.attn.head_dim})", ffn,
                              "pre-norm, residual sau mỗi nhánh"],
                       shape=_b(rows[f"blocks.{depth - 1}"][1]), params=_n(model.blocks)))

    stages.append(dict(key="norm", title="LayerNorm → lấy [CLS]", lines=["vector ở vị trí 0"],
                       shape=_b(rows["head"][0]), params=_n(model.norm)))

    lin = [m for m in model.head.modules() if isinstance(m, nn.Linear)]
    head_desc = " → ".join([str(lin[0].in_features)] + [str(m.out_features) for m in lin])
    stages.append(dict(key="head", title="ClassifierHead", lines=[f"Linear {head_desc}"],
                       shape=_b(rows["head"][1]), params=_n(model.head), style=_IO))
    return stages


def _box(ax, x0, y_top, w, title, lines, style, params=None, fs=9):
    """Vẽ một khối, trả về toạ độ y của đáy khối."""
    h = 0.032 + 0.026 * len(lines)
    ax.add_patch(FancyBboxPatch((x0, y_top - h), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=style["face"], edgecolor=style["edge"], lw=1.4))
    ax.text(x0 + 0.02, y_top - 0.022, title, fontsize=fs + 0.5, fontweight="bold", va="center")
    for i, line in enumerate(lines):
        ax.text(x0 + 0.03, y_top - 0.048 - 0.026 * i, line, fontsize=fs - 0.5, va="center")
    if params is not None:
        ax.text(x0 + w - 0.015, y_top - 0.022, f"{params:,} tham số", fontsize=fs - 1.5,
                va="center", ha="right", color="#555555")
    return y_top - h


def _arrow(ax, x, y_from, y_to, label=None):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.1))
    if label:
        ax.text(x + 0.02, (y_from + y_to) / 2, label, fontsize=7.5, va="center",
                family="monospace", color="#333333")


def _draw_column(ax, title, stages, ref_keys, gap=0.052):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    total = sum(s["params"] or 0 for s in stages)
    ax.text(0.5, 0.985, title, ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.95, f"tổng {total:,} tham số", ha="center", va="top", fontsize=9, color="#555555")
    y = 0.91
    for i, s in enumerate(stages):
        style = s.get("style") or (_CHANGED if s["key"] not in ref_keys else _SAME)
        bottom = _box(ax, 0.06, y, 0.88, s["title"], s["lines"], style, s["params"])
        if i < len(stages) - 1:
            _arrow(ax, 0.5, bottom - 0.004, bottom - gap + 0.006, s["shape"])
        else:
            ax.text(0.5, bottom - 0.03, f"logits {s['shape']}", ha="center", fontsize=8, family="monospace")
        y = bottom - gap


def _draw_ffn_detail(ax, mlp: MLP, leff: LocalityFeedForward, T: int):
    """So sánh chi tiết khối feed-forward: MLP (trái) và LeFF (phải)."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.985, "Chi tiết khối feed-forward", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.95, "trong mỗi TransformerBlock", ha="center", va="top", fontsize=9, color="#555555")
    D, Fh = mlp.fc1.in_features, mlp.fc1.out_features
    G, k = leff.grid_size, leff.dwconv.kernel_size[0]
    N = G * G
    flows = {
        (0.03, f"MLP · {_n(mlp):,}"): [
            ("fc1", [f"Linear {D}→{Fh}"], _SAME, f"(B, {T}, {Fh})"),
            ("GELU", [], _SAME, f"(B, {T}, {Fh})"),
            ("fc2", [f"Linear {Fh}→{D}"], _SAME, f"(B, {T}, {D})"),
        ],
        (0.52, f"LeFF · {_n(leff):,}"): [
            ("fc1", [f"Linear {D}→{Fh}"], _SAME, f"(B, {T}, {Fh})"),
            ("tách [CLS]", [f"patch → lưới {G}×{G}", "[CLS] đi thẳng"], _CHANGED, f"(B, {Fh}, {G}, {G})"),
            (f"DWConv {k}×{k}", [f"groups={Fh}", f"+{_n(leff.dwconv):,} tham số"], _CHANGED, f"(B, {Fh}, {G}, {G})"),
            ("duỗi + nối [CLS]", [f"{N} patch + 1"], _CHANGED, f"(B, {T}, {Fh})"),
            ("GELU", [], _SAME, f"(B, {T}, {Fh})"),
            ("fc2", [f"Linear {Fh}→{D}"], _SAME, f"(B, {T}, {D})"),
        ],
    }
    for (x0, title), steps in flows.items():
        w = 0.45
        ax.text(x0 + w / 2, 0.905, title + " tham số", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(x0 + w / 2, 0.875, f"vào (B, {T}, {D})", ha="center", fontsize=7.5, family="monospace")
        y = 0.85
        for i, (name, lines, style, shape) in enumerate(steps):
            bottom = _box(ax, x0, y, w, name, lines, style, fs=8.5)
            _arrow(ax, x0 + 0.06, bottom - 0.004, bottom - 0.046, None)
            ax.text(x0 + 0.09, bottom - 0.025, shape, fontsize=7, va="center", family="monospace")
            y = bottom - 0.05
    ax.text(0.5, 0.06, "Cùng shape vào/ra (B, T, D): LeFF thay thẳng cho MLP,\n"
                       "mỗi patch được trộn với 8 patch láng giềng ở mọi layer.",
            ha="center", fontsize=8.5, style="italic", color="#444444")


def plot_architecture_comparison(models: Dict[str, nn.Module], x: torch.Tensor,
                                 save_path: Optional[Path] = None):
    """models: {tên: ViT}; model đầu tiên là bản tham chiếu. Khối khác bản tham chiếu
    được tô cam. Nếu có cả MLP và LeFF, thêm một cột so sánh chi tiết hai loại FFN."""
    descs = {name: describe_stages(m, x) for name, m in models.items()}
    ref_keys = {s["key"] for s in next(iter(descs.values()))}
    mlps = [m.blocks[0].mlp for m in models.values()]
    mlp = next((f for f in mlps if isinstance(f, MLP)), None)
    leff = next((f for f in mlps if isinstance(f, LocalityFeedForward)), None)
    detail = mlp is not None and leff is not None

    cols = len(models) + int(detail)
    fig, axes = plt.subplots(1, cols, figsize=(5.4 * cols, 9.8), squeeze=False)
    for ax, (name, stages) in zip(axes[0], descs.items()):
        _draw_column(ax, name, stages, ref_keys)
    if detail:
        T = next(iter(models.values())).tokens.num_tokens
        _draw_ffn_detail(axes[0, -1], mlp, leff, T)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=_SAME["face"], ec=_SAME["edge"]),
                        plt.Rectangle((0, 0), 1, 1, fc=_CHANGED["face"], ec=_CHANGED["edge"])],
               labels=["giống bản gốc", "thay đổi so với bản gốc"], loc="lower center", ncol=2,
               fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig
