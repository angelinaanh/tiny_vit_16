"""Vòng lặp huấn luyện 1 epoch và đánh giá. Không tự dựng model/data/optimizer -
mọi thứ được truyền vào từ train.py / eval.py."""
import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import TrainConfig
from .metrics import AverageMeter, accuracy


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, device: torch.device,
                    epoch: int, epochs: int, cfg: TrainConfig,
                    logger: logging.Logger) -> Dict[str, float]:
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    n_iters = len(loader)
    start = time.time()

    for it, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)   # (B, 3, 224, 224)
        labels = labels.to(device, non_blocking=True)   # (B,)

        logits = model(images)                          # (B, K)
        loss = criterion(logits, labels)                # ()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        bs = labels.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy(logits, labels)[0], bs)

        if it == 1 or it % cfg.log_every == 0 or it == n_iters:
            elapsed = time.time() - start
            logger.info(f"epoch {epoch:>2}/{epochs} | iter {it:>4}/{n_iters} | "
                        f"loss {loss.item():.4f} (avg {loss_meter.avg:.4f}) | "
                        f"acc avg {acc_meter.avg:5.2f}% | lr {optimizer.param_groups[0]['lr']:.2e} | "
                        f"{elapsed / it:.2f}s/iter")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg, "time": time.time() - start}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
             return_predictions: bool = False
             ) -> Tuple[Dict[str, float], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    model.eval()
    loss_meter, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    all_preds, all_targets = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)   # (B, 3, 224, 224)
        labels = labels.to(device, non_blocking=True)   # (B,)
        logits = model(images)                          # (B, K)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        a1, a5 = accuracy(logits, labels, topk=(1, 5))
        loss_meter.update(loss.item(), bs)
        top1.update(a1, bs)
        top5.update(a5, bs)
        if return_predictions:
            all_preds.append(logits.argmax(dim=1).cpu())
            all_targets.append(labels.cpu())

    metrics = {"loss": loss_meter.avg, "acc": top1.avg, "acc5": top5.avg}
    preds = (torch.cat(all_preds), torch.cat(all_targets)) if return_predictions else None
    return metrics, preds
