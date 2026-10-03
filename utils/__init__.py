"""Hàm hỗ trợ: accuracy, logging, checkpoint, optimizer/scheduler, vòng lặp train/eval,
attention map và vẽ hình. Matplotlib chỉ được import khi cần (utils.visualize)."""
from .attention import cls_attention, to_heatmap
from .checkpoint import load_checkpoint, save_checkpoint
from .engine import evaluate, train_one_epoch
from .logger import HistoryWriter, load_history, setup_logger
from .metrics import AverageMeter, accuracy, confusion_matrix
from .misc import count_parameters, seed_everything, setup_runtime
from .optim import build_loss, build_optimizer, build_scheduler
from .shapes import format_shape_table, trace_shapes
