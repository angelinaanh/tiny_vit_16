"""Vẽ đường cong huấn luyện và attention map chồng lên ảnh."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """(B, 3, H, W) đã chuẩn hoá -> (B, 3, H, W) trong [0, 1] để hiển thị."""
    m = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return (images * s + m).clamp(0, 1)


def plot_history(histories: Dict[str, List[Dict[str, Any]]], save_path: Optional[Path] = None,
                 chance: float = 10.0, show_train: bool = True):
    """histories: {tên run: list các dòng history.json}. Epoch 0 = model khởi tạo ngẫu nhiên.
    show_train=False chỉ vẽ đường val (gọn hơn khi so sánh nhiều run)."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, rows in histories.items():
        ep = [r["epoch"] for r in rows]
        tr = [r for r in rows if r.get("train_loss") is not None]
        line, = ax_loss.plot(ep, [r["val_loss"] for r in rows], marker="o", ms=3, label=f"{name} · val")
        ax_acc.plot(ep, [r["val_acc"] for r in rows], marker="o", ms=3, color=line.get_color(),
                    label=f"{name} · val")
        if show_train:
            ax_loss.plot([r["epoch"] for r in tr], [r["train_loss"] for r in tr], ls="--",
                         color=line.get_color(), label=f"{name} · train")
            ax_acc.plot([r["epoch"] for r in tr], [r["train_acc"] for r in tr], ls="--",
                        color=line.get_color(), label=f"{name} · train")
    ax_acc.axhline(chance, color="gray", ls=":", lw=1, label=f"ngẫu nhiên ({chance:.0f}%)")
    ax_acc.axhline(45, color="tab:red", ls=":", lw=1, label="mục tiêu 45%")
    ax_loss.set(title="Loss", xlabel="epoch (0 = khởi tạo ngẫu nhiên)", ylabel="cross-entropy")
    ax_acc.set(title="Top-1 accuracy", xlabel="epoch (0 = khởi tạo ngẫu nhiên)", ylabel="%")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def plot_attention_overlay(images: torch.Tensor, heatmaps: Dict[str, torch.Tensor],
                           titles: List[str], save_path: Optional[Path] = None, alpha: float = 0.5):
    """images: (B, 3, H, W) trong [0, 1]; heatmaps: {tên phương pháp: (B, H, W) trong [0, 1]}.

    Mỗi hàng là một ảnh: ảnh gốc | heatmap chồng lên ảnh cho từng phương pháp.
    """
    B = images.shape[0]
    cols = 1 + len(heatmaps)
    fig, axes = plt.subplots(B, cols, figsize=(3.4 * cols, 3.4 * B), squeeze=False)
    for i in range(B):
        img = images[i].permute(1, 2, 0).cpu().numpy()          # (H, W, 3)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(titles[i], fontsize=9)
        for j, (name, maps) in enumerate(heatmaps.items(), start=1):
            axes[i, j].imshow(img)
            axes[i, j].imshow(maps[i].cpu().numpy(), cmap="jet", alpha=alpha)
            axes[i, j].set_title(f"[CLS] attention · {name}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def plot_attention_comparison(images: torch.Tensor, per_model: Dict[str, Tuple[torch.Tensor, List[str]]],
                              row_titles: List[str], method: str, save_path: Optional[Path] = None,
                              alpha: float = 0.5):
    """So sánh attention của [CLS] giữa nhiều model trên cùng các ảnh.

    images   : (B, 3, H, W) trong [0, 1]
    per_model: {tên model: (heatmap (B, H, W), tiêu đề cho từng ảnh)}
    Mỗi hàng là một ảnh: ảnh gốc | heatmap của model 1 | model 2 | ...
    """
    B = images.shape[0]
    cols = 1 + len(per_model)
    fig, axes = plt.subplots(B, cols, figsize=(3.3 * cols, 3.8 * B), squeeze=False, layout="constrained")
    for i in range(B):
        img = images[i].permute(1, 2, 0).cpu().numpy()          # (H, W, 3)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(row_titles[i], fontsize=9)
        for j, (maps, titles) in enumerate(per_model.values(), start=1):
            axes[i, j].imshow(img)
            axes[i, j].imshow(maps[i].cpu().numpy(), cmap="jet", alpha=alpha)
            axes[i, j].set_title(titles[i], fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Attention của [CLS] ({method}) trên cùng ảnh val", fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def plot_confusion_matrix(cm: torch.Tensor, class_names: List[str], save_path: Optional[Path] = None):
    cm = cm.cpu().numpy().astype(np.float64)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    for r in range(len(class_names)):
        for c in range(len(class_names)):
            ax.text(c, r, f"{int(cm[r, c])}", ha="center", va="center", fontsize=7,
                    color="white" if norm[r, c] > 0.5 else "black")
    ax.set(xlabel="dự đoán", ylabel="nhãn thật", title="Confusion matrix (tập val)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def plot_position_similarity(pos_weight: torch.Tensor, grid_size: int, title: str = "",
                             save_path: Optional[Path] = None):
    """Cosine similarity giữa position embedding của mỗi patch với mọi patch khác.

    pos_weight: (T, D) - bảng nn.Embedding, hàng 0 là [CLS] nên bị bỏ qua.
    Ô (r, c) của lưới lớn là bản đồ 14x14: patch (r, c) giống patch nào nhất.
    Nếu model học được cấu trúc 2D, mỗi bản đồ sáng quanh đúng vị trí (r, c).
    """
    pos = torch.nn.functional.normalize(pos_weight[1:].detach().cpu(), dim=-1)   # (N, D)
    sim = (pos @ pos.T).reshape(grid_size, grid_size, grid_size, grid_size)      # (G, G, G, G)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(7, 7))
    for r in range(grid_size):
        for c in range(grid_size):
            axes[r, c].imshow(sim[r, c].numpy(), cmap="viridis", vmin=-1, vmax=1)
            axes[r, c].axis("off")
    fig.suptitle(title or "Cosine similarity của position embedding (hàng, cột của patch)", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig
