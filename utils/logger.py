"""Logging: in ra màn hình + ghi file log + lưu lịch sử từng epoch (CSV/JSON)."""
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from .misc import ensure_utf8_console


def setup_logger(log_file: Path, name: str = "vit") -> logging.Logger:
    ensure_utf8_console()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class HistoryWriter:
    """Mỗi epoch một dòng: epoch, lr, train_loss, train_acc, val_loss, val_acc, ..."""

    def __init__(self, out_dir: Path) -> None:
        self.csv_path = out_dir / "history.csv"
        self.json_path = out_dir / "history.json"
        self.rows: List[Dict[str, Any]] = []

    def append(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        keys = list(self.rows[0].keys())
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2)


def load_history(out_dir: Path) -> List[Dict[str, Any]]:
    with open(Path(out_dir) / "history.json", encoding="utf-8") as f:
        return json.load(f)
