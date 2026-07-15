"""通用工具：随机种子、绘图格式化、IO 辅助。"""
from __future__ import annotations
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子，保证可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def format_axes(ax, *, title: str = "", xlabel: str = "", ylabel: str = "",
                unit_x: str = "", unit_y: str = "", legend: bool = True) -> None:
    """统一图表四要素：标题、轴标签、单位、图例。任务书硬要求。"""
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    xlab = f"{xlabel} ({unit_x})" if xlabel and unit_x else (xlabel or unit_x)
    ylab = f"{ylabel} ({unit_y})" if ylabel and unit_y else (ylabel or unit_y)
    if xlab:
        ax.set_xlabel(xlab, fontsize=10)
    if ylab:
        ax.set_ylabel(ylab, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def ensure_dir(path) -> None:
    """递归创建目录（若不存在）。"""
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
