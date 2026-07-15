"""WESAD 数据加载、标签重采样与窗口提取。

关键工程细节（N2/N5 决策）：
  1. WESAD 标签存储于 chest 设备（700Hz），wrist 的 BVP=64Hz / EDA=4Hz。
     必须按时间对齐把 700Hz 标签重采样到各 wrist 采样率，否则标签错位。
  2. 10s 窗口内主标签占比 < 80% 直接丢弃（N2 纯净度过滤），
     剔除条件切换过渡段（label=0）造成的标签噪声。
  3. 仅保留 baseline(1)/stress(2)/amusement(3) 三类，其余标签的窗口丢弃。
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

VALID_SUBJECTS = [f"S{i}" for i in range(2, 18)]  # S2..S17，共 15 名被试

# wrist 采样率
FS_CHEST = 700
FS_BVP = 64
FS_EDA = 4

LABEL_BASELINE = 1
LABEL_STRESS = 2
LABEL_AMUSEMENT = 3
STATE_NAMES = ["baseline", "stress", "amusement"]
STATE_TO_IDX = {n: i for i, n in enumerate(STATE_NAMES)}
LABEL_TO_STATE = {1: "baseline", 2: "stress", 3: "amusement"}

# 窗口参数
WIN_SEC = 10.0
STEP_SEC = 5.0
PURITY_THRESH = 0.8  # N2：窗口内主标签占比阈值


# ─────────────────────────────────────────────
# 原始加载
# ─────────────────────────────────────────────
def load_subject(subject_id: str, raw_dir: Optional[Path] = None) -> dict:
    """加载单个被试 .pkl（原始字典结构）。"""
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    pkl_path = raw_dir / f"{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"未找到 {pkl_path}，请先按 data/raw/README.md 放入 WESAD 原始数据。"
        )
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def list_available_subjects(raw_dir: Optional[Path] = None) -> list[str]:
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    return sorted(
        p.stem for p in raw_dir.glob("S*.pkl") if p.stem in VALID_SUBJECTS
    )


def extract_wrist_signals(data: dict) -> dict:
    """从原始 dict 取出 wrist BVP/EDA 与 chest label，展平为一维 ndarray。"""
    wrist = data["signal"]["wrist"]
    bvp = np.asarray(wrist["BVP"]).reshape(-1)
    eda = np.asarray(wrist["EDA"]).reshape(-1)
    label = np.asarray(data["label"]).reshape(-1)
    return {"bvp": bvp, "eda": eda, "label": label}


# ─────────────────────────────────────────────
# 标签重采样：700Hz → 目标采样率（最近邻，按时间对齐）
# ─────────────────────────────────────────────
def resample_label(label_700: np.ndarray, n_target: int,
                   fs_target: int) -> np.ndarray:
    """把 700Hz 标签按时间映射到 fs_target。

    label_700 的第 j 个样本对应时间 j/700 秒；
    目标序列第 i 个样本对应时间 i/fs_target 秒。
    故 target[i] = label_700[round(i/fs_target * 700)]。
    """
    idx = np.round(np.arange(n_target) / fs_target * FS_CHEST).astype(np.int64)
    idx = np.clip(idx, 0, len(label_700) - 1)
    return label_700[idx]


# ─────────────────────────────────────────────
# 窗口提取
# ─────────────────────────────────────────────
@dataclass
class WindowSet:
    """单被试窗口集合。"""
    subject: str
    ppg: np.ndarray       # (N, 640) float32
    eda: np.ndarray       # (N, 40)  float32
    labels: np.ndarray    # (N,)     int64  ∈ {0,1,2}
    raw_state_labels: np.ndarray  # (N,) 原始标签 {1,2,3}，便于调试

    def __len__(self) -> int:
        return len(self.labels)


def _majority_label(window_labels: np.ndarray) -> tuple[int, float]:
    """返回窗口内的多数票标签及其占比（仅考虑 {1,2,3}）。"""
    valid = window_labels[np.isin(window_labels, (1, 2, 3))]
    if len(valid) == 0:
        return -1, 0.0
    counts = np.bincount(valid, minlength=4)
    lab = int(np.argmax(counts[1:4]) + 1)  # 仅在 1,2,3 中取
    purity = counts[lab] / len(window_labels)
    return lab, purity


def extract_windows(bvp: np.ndarray, eda: np.ndarray, label_700: np.ndarray,
                    subject: str, *, win_sec: float = WIN_SEC,
                    step_sec: float = STEP_SEC, purity: float = PURITY_THRESH
                    ) -> WindowSet:
    """按 PPG(64Hz) 节拍切窗，同步切 EDA(4Hz)，标签用 64Hz 重采样后的多数票。

    Returns
    -------
    WindowSet
        ppg(N,640) / eda(N,40) / labels(N,) ∈ {0,1,2} 状态索引
    """
    n_bvp = len(bvp)
    n_eda = len(eda)
    # 把 700Hz 标签到 64Hz（与 BVP 对齐）
    label_64 = resample_label(label_700, n_bvp, FS_BVP)

    win_ppg = int(round(win_sec * FS_BVP))     # 640
    step_ppg = int(round(step_sec * FS_BVP))   # 320
    win_eda = int(round(win_sec * FS_EDA))     # 40
    # EDA 与 PPG 的时间换算：1 个 PPG 样本 = FS_EDA/FS_BVP 个 EDA 样本
    eda_per_bvp = FS_EDA / FS_BVP

    ppg_w, eda_w, lab_idx, raw_lab = [], [], [], []
    start = 0
    while start + win_ppg <= n_bvp:
        end = start + win_ppg
        # EDA 对应区间
        e_start = int(round(start * eda_per_bvp))
        e_end = e_start + win_eda
        if e_end > n_eda:
            break
        # 标签多数票（64Hz 标签窗口）
        lab, p = _majority_label(label_64[start:end])
        if lab == -1 or p < purity:
            start += step_ppg
            continue
        ppg_w.append(bvp[start:end])
        eda_w.append(eda[e_start:e_end])
        raw_lab.append(lab)
        lab_idx.append(LABEL_TO_STATE[lab])
        start += step_ppg

    if not ppg_w:
        return WindowSet(subject, np.empty((0, win_ppg), np.float32),
                         np.empty((0, win_eda), np.float32),
                         np.empty(0, np.int64), np.empty(0, np.int64))
    return WindowSet(
        subject=subject,
        ppg=np.asarray(ppg_w, dtype=np.float32),
        eda=np.asarray(eda_w, dtype=np.float32),
        labels=np.asarray([STATE_TO_IDX[s] for s in lab_idx], dtype=np.int64),
        raw_state_labels=np.asarray(raw_lab, dtype=np.int64),
    )


def window_class_counts(ws: WindowSet) -> dict[str, int]:
    """返回三状态窗口数，用于校验类别比例（N3）。"""
    counts = {n: 0 for n in STATE_NAMES}
    for s, n in STATE_TO_IDX.items():
        counts[s] = int(np.sum(ws.labels == n))
    return counts


# ─────────────────────────────────────────────
# PyTorch Dataset 包装
# ─────────────────────────────────────────────
def to_torch_tensors(ws: WindowSet):
    """把 WindowSet 转为 torch Tensor（不切分，供外部 DataLoader 划分）。"""
    import torch
    return (
        torch.from_numpy(ws.ppg).float().unsqueeze(1),   # (N,1,640)
        torch.from_numpy(ws.eda).float().unsqueeze(1),   # (N,1,40)
        torch.from_numpy(ws.labels).long(),              # (N,)
    )
