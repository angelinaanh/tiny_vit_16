"""Chỉ đánh giá một checkpoint đã huấn luyện trên tập kiểm tra.

Imagenette chỉ có 2 split (train / val) nên split "val" đóng vai trò tập test;
nó không bao giờ được dùng để cập nhật trọng số.

Ví dụ:
    python eval.py --run outputs/vit_tiny_16
    python eval.py --run outputs/vit_tiny_16_convstem --checkpoint last.pt
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from configs import load_config
from datasets import build_dataloaders, class_names
from models import build_model
from utils import build_loss, confusion_matrix, evaluate, load_checkpoint, setup_runtime
from utils.visualize import plot_confusion_matrix


def run_eval(run_dir: Path, checkpoint: str = "best.pt") -> dict:
    cfg = load_config(str(run_dir / "config.yaml"))      # dùng đúng config lúc train
    device = setup_runtime(cfg.experiment)
    _, test_loader = build_dataloaders(cfg.data)
    names = class_names(test_loader.dataset)

    model = build_model(cfg.model, cfg.data.image_size, cfg.data.num_classes).to(device)
    ckpt = load_checkpoint(run_dir / checkpoint, model, map_location=str(device))
    metrics, (preds, targets) = evaluate(model, test_loader, build_loss(cfg.loss), device,
                                         return_predictions=True)

    cm = confusion_matrix(preds, targets, cfg.data.num_classes)            # (K, K)
    per_class = (cm.diag().float() / cm.sum(dim=1).clamp(min=1).float() * 100).tolist()
    result = {
        "run": cfg.experiment.name, "checkpoint": checkpoint, "epoch": ckpt["epoch"],
        "test_loss": metrics["loss"], "test_acc": metrics["acc"], "test_acc5": metrics["acc5"],
        "per_class_acc": dict(zip(names, per_class)),
    }

    print(f"[{cfg.experiment.name}] {checkpoint} (epoch {ckpt['epoch']}) | "
          f"loss {metrics['loss']:.4f} | top-1 {metrics['acc']:.2f}% | top-5 {metrics['acc5']:.2f}%")
    for n, a in zip(names, per_class):
        print(f"   {n:<18} {a:6.2f}%")

    with open(run_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    plt.close(plot_confusion_matrix(cm, names, run_dir / "confusion_matrix.png"))   # chỉ lưu file
    return result


if __name__ == "__main__":
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="thư mục run, ví dụ outputs/vit_tiny_16")
    p.add_argument("--checkpoint", default="best.pt")
    args = p.parse_args()
    run_eval(Path(args.run), args.checkpoint)
