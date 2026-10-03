"""Huấn luyện ViT-Tiny_16 from scratch trên Imagenette.

File này chỉ KHỞI TẠO và NỐI các thành phần theo đúng luồng dữ liệu:

    [1] Dataset  ->  [2] Tiền xử lý  ->  [3] DataLoader
        ->  [4] Model (Patch -> Attention -> Classifier)
        ->  [5] Loss  ->  [6] Optimizer (+ LR scheduler)

rồi chạy vòng lặp huấn luyện. Mọi hyperparameter đến từ configs/config.yaml.

Ví dụ:
    python train.py                                       # bản gốc (patch + nn.Linear)
    python train.py --set model.stem=conv experiment.name=vit_tiny_16_convstem
"""
import argparse
import time
from pathlib import Path

from configs import Config, load_config, save_config
from datasets import build_dataloaders
from models import build_model
from utils import (HistoryWriter, build_loss, build_optimizer, build_scheduler, count_parameters,
                   evaluate, format_shape_table, save_checkpoint, setup_logger, setup_runtime,
                   trace_shapes, train_one_epoch)


def run_training(cfg: Config) -> Path:
    out_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    logger = setup_logger(out_dir / "train_log.txt")
    device = setup_runtime(cfg.experiment)
    logger.info(f"Run '{cfg.experiment.name}' | {cfg.model.arch} | stem={cfg.model.stem} | device={device}")

    # [1] Dataset + [2] Tiền xử lý + [3] DataLoader -> batch (B, 3, 224, 224), (B,)
    train_loader, val_loader = build_dataloaders(cfg.data)
    logger.info(f"Dữ liệu: {len(train_loader.dataset)} ảnh train, {len(val_loader.dataset)} ảnh val, "
                f"{len(train_loader)} iter/epoch, batch_size={cfg.data.batch_size}")

    # [4] Model: (B, 3, 224, 224) -> (B, 10), trọng số khởi tạo ngẫu nhiên
    model = build_model(cfg.model, cfg.data.image_size, cfg.data.num_classes).to(device)
    params = count_parameters(model)
    logger.info("Số tham số: " + ", ".join(f"{k}={v:,}" for k, v in params.items()))
    sample, _ = next(iter(val_loader))
    logger.info("Luồng shape qua model:\n" + format_shape_table(trace_shapes(model, sample[:2].to(device))))

    # [5] Loss + [6] Optimizer / scheduler
    criterion = build_loss(cfg.loss)
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = build_scheduler(cfg.scheduler, optimizer, cfg.optimizer.lr,
                                cfg.train.epochs, len(train_loader))

    history = HistoryWriter(out_dir)
    best_acc = -1.0

    # Epoch 0: đánh giá model vừa khởi tạo ngẫu nhiên -> loss ~ ln(10) = 2.30, acc ~ 10%
    if cfg.train.eval_before_training:
        val, _ = evaluate(model, val_loader, criterion, device)
        logger.info(f"epoch  0 (khởi tạo ngẫu nhiên, chưa train) | val loss {val['loss']:.4f} | "
                    f"val acc {val['acc']:.2f}% | val acc5 {val['acc5']:.2f}%")
        history.append({"epoch": 0, "lr": 0.0, "train_loss": None, "train_acc": None,
                        "val_loss": val["loss"], "val_acc": val["acc"], "val_acc5": val["acc5"],
                        "epoch_time_s": 0.0})

    total_start = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                                epoch, cfg.train.epochs, cfg.train, logger)
        val, _ = evaluate(model, val_loader, criterion, device)

        is_best = val["acc"] > best_acc
        best_acc = max(best_acc, val["acc"])
        logger.info(f"==> epoch {epoch:>2}/{cfg.train.epochs} | train loss {train['loss']:.4f} "
                    f"acc {train['acc']:.2f}% | val loss {val['loss']:.4f} acc {val['acc']:.2f}% "
                    f"acc5 {val['acc5']:.2f}% | best {best_acc:.2f}% | {train['time']:.0f}s"
                    + ("  *best*" if is_best else ""))
        history.append({"epoch": epoch, "lr": lr, "train_loss": train["loss"], "train_acc": train["acc"],
                        "val_loss": val["loss"], "val_acc": val["acc"], "val_acc5": val["acc5"],
                        "epoch_time_s": train["time"]})

        metrics = {"val_acc": val["acc"], "val_loss": val["loss"]}
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, metrics, cfg.to_dict())
        if is_best:
            save_checkpoint(out_dir / "best.pt", model, None, epoch, metrics, cfg.to_dict())

    logger.info(f"Xong {cfg.train.epochs} epoch trong {(time.time() - total_start) / 60:.1f} phút | "
                f"best val acc {best_acc:.2f}% | kết quả ở {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="ghi đè config, ví dụ: model.stem=conv train.epochs=15")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(load_config(args.config, args.set))
