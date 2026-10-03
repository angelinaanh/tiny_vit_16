"""In bảng shape tensor đi qua từng khối của model - để kiểm chứng hợp đồng shape."""
from typing import List, Tuple

import torch
import torch.nn as nn


def _shape(t) -> str:
    if isinstance(t, torch.Tensor):
        return str(tuple(t.shape))
    if isinstance(t, (tuple, list)) and t and isinstance(t[0], torch.Tensor):
        return str(tuple(t[0].shape))
    return "-"


@torch.no_grad()
def trace_shapes(model: nn.Module, x: torch.Tensor, max_depth: int = 2) -> List[Tuple[str, str, str, str]]:
    """Chạy một forward và ghi lại (tên module, loại, shape vào, shape ra)
    cho mọi module có độ sâu <= max_depth (tính theo số dấu chấm trong tên)."""
    rows: List[Tuple[str, str, str, str]] = []
    hooks = []
    for name, module in model.named_modules():
        if not name or name.count(".") >= max_depth:
            continue
        hooks.append(module.register_forward_hook(
            lambda m, inp, out, name=name: rows.append(
                (name, type(m).__name__, _shape(inp), _shape(out)))))
    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return rows


def format_shape_table(rows: List[Tuple[str, str, str, str]]) -> str:
    header = ("module", "loại", "shape vào", "shape ra")
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(4)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
    return "\n".join([line(header), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])
