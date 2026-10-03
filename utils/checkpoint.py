"""Lưu / nạp checkpoint. Chỉ nạp checkpoint do chính dự án này huấn luyện ra."""
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    epoch: int, metrics: Dict[str, float], config: Dict[str, Any]) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "config": config,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt
