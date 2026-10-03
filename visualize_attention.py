"""Trực quan hoá attention map của token [CLS], chồng lên vài ảnh tập val.

Ví dụ:
    # một model: ảnh gốc | rollout | last_layer
    python visualize_attention.py --run outputs/vit_tiny_16
    python visualize_attention.py --run outputs/vit_tiny_16_convstem --indices 1450 3200

    # nhiều model trên cùng ảnh (mỗi cột một model)
    python visualize_attention.py --run outputs/vit_tiny_16 outputs/vit_tiny_16_convstem_leff \
        --out outputs/attention_compare.png
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import torch

from configs import load_config
from datasets import build_datasets, class_names
from models import build_model
from utils import cls_attention, load_checkpoint, setup_runtime, to_heatmap
from utils.visualize import denormalize, plot_attention_comparison, plot_attention_overlay

METHODS = ("rollout", "last_layer")


@torch.no_grad()
def compute_attention(run_dir: Path, checkpoint: str = "best.pt",
                      indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """Nạp model của một run, chạy forward trên vài ảnh val và trả về attention của [CLS].

    images  : (B, 3, 224, 224) trong [0, 1] để hiển thị
    heatmaps: {method: (B, 224, 224)} trong [0, 1]
    """
    cfg = load_config(str(run_dir / "config.yaml"))
    device = setup_runtime(cfg.experiment)
    _, val_ds = build_datasets(cfg.data)

    model = build_model(cfg.model, cfg.data.image_size, cfg.data.num_classes).to(device)
    load_checkpoint(run_dir / checkpoint, model, map_location=str(device))
    model.eval()

    if indices is None:   # mặc định: lấy ảnh rải đều trên tập val (mỗi ảnh một lớp khác nhau)
        step = len(val_ds) // cfg.visualize.num_images
        indices = [i * step + step // 2 for i in range(cfg.visualize.num_images)]
    images = torch.stack([val_ds[i][0] for i in indices]).to(device)       # (B, 3, 224, 224)

    with model.capture_attention() as maps:
        logits = model(images)                                             # (B, K)
    # maps: list `depth` x (B, h, T, T)

    heatmaps = {}
    for method in METHODS:
        attn = cls_attention(maps, method, cfg.visualize.head_fusion)      # (B, N)
        heatmaps[method] = to_heatmap(attn, model.grid_size, cfg.data.image_size)  # (B, 224, 224)

    return {
        "names": class_names(val_ds),
        "indices": indices,
        "images": denormalize(images, cfg.data.mean, cfg.data.std),
        "labels": [val_ds[i][1] for i in indices],
        "preds": logits.argmax(dim=1).tolist(),
        "probs": logits.softmax(dim=1).max(dim=1).values.tolist(),
        "heatmaps": heatmaps,
    }


def run_visualize(run_dir: Path, checkpoint: str = "best.pt", indices: Optional[List[int]] = None,
                  save_name: str = "attention_cls.png"):
    """Một model: mỗi hàng một ảnh -> ảnh gốc | rollout | last_layer."""
    a = compute_attention(run_dir, checkpoint, indices)
    names = a["names"]
    titles = [f"thật: {names[y]}\ndự đoán: {names[p]} ({c:.0%})"
              for y, p, c in zip(a["labels"], a["preds"], a["probs"])]
    return plot_attention_overlay(a["images"], a["heatmaps"], titles, run_dir / save_name)


def run_compare(runs: Dict[str, Path], indices: Optional[List[int]] = None, method: str = "rollout",
                checkpoint: str = "best.pt", save_path: Optional[Path] = None):
    """Nhiều model trên CÙNG các ảnh: mỗi hàng một ảnh, mỗi cột một model."""
    results: Dict[str, Dict[str, Any]] = {}
    for name, run_dir in runs.items():
        results[name] = compute_attention(run_dir, checkpoint, indices)
        indices = results[name]["indices"]          # các model sau dùng đúng ảnh của model đầu
    first = next(iter(results.values()))
    names = first["names"]
    row_titles = [f"ảnh val #{i}\nthật: {names[y]}" for i, y in zip(first["indices"], first["labels"])]
    per_model = {
        name: (r["heatmaps"][method],
               [f"{name}\ndự đoán: {names[p]} ({c:.0%})" + ("" if p == y else "  ✗")
                for y, p, c in zip(r["labels"], r["preds"], r["probs"])])
        for name, r in results.items()
    }
    return plot_attention_comparison(first["images"], per_model, row_titles, method, save_path)


if __name__ == "__main__":
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", required=True, help="một hoặc nhiều thư mục run")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--indices", type=int, nargs="*", default=None, help="chỉ số ảnh trong tập val")
    p.add_argument("--method", choices=METHODS, default="rollout", help="dùng khi so sánh nhiều run")
    p.add_argument("--out", default="outputs/attention_compare.png", help="file ảnh khi so sánh nhiều run")
    args = p.parse_args()

    run_dirs = [Path(r) for r in args.run]
    if len(run_dirs) == 1:
        run_visualize(run_dirs[0], args.checkpoint, args.indices)
        print(f"Đã lưu {run_dirs[0] / 'attention_cls.png'}")
    else:
        run_compare({d.name: d for d in run_dirs}, args.indices, args.method, args.checkpoint, Path(args.out))
        print(f"Đã lưu {args.out}")
