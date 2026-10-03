"""Tiện ích chung: seed, thiết bị, đếm tham số."""
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from configs import ExperimentConfig


def ensure_utf8_console() -> None:
    """Console Windows mặc định là cp1252, không in được tiếng Việt -> chuyển sang UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_runtime(cfg: ExperimentConfig) -> torch.device:
    """Đặt seed, số luồng CPU và chọn thiết bị theo config."""
    ensure_utf8_console()
    seed_everything(cfg.seed)
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Số tham số học được của từng khối con cấp 1 và tổng."""
    counts = {name: sum(p.numel() for p in child.parameters() if p.requires_grad)
              for name, child in model.named_children()}
    counts["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts
