"""信号预处理：滤波 / SQI / 标准化 / 分段。

完整实现见计划书第六章 6.2 节，本文件为骨架。
"""
from __future__ import annotations
import numpy as np
from scipy.signal import cheby2, filtfilt, iirnotch, resample_poly


# PPG / EDA 采样率
FS_PPG = 64
FS_EDA = 4


def bandpass_ppg(sig: np.ndarray, fs: int = FS_PPG) -> np.ndarray:
    """Chebyshev II 型带通 0.5–8 Hz。"""
    nyq = fs / 2.0
    b, a = cheby2(N=4, rs=20, Wn=[0.5 / nyq, 8.0 / nyq], btype="bandpass")
    return filtfilt(b, a, sig)


def lowpass_eda(sig: np.ndarray, fs: int = FS_EDA) -> np.ndarray:
    """EDA 1 Hz 低通。"""
    nyq = fs / 2.0
    b, a = cheby2(N=4, rs=20, Wn=1.0 / nyq, btype="lowpass")
    return filtfilt(b, a, sig)


def notch_50(sig: np.ndarray, fs: int, Q: float = 35.0) -> np.ndarray:
    """50 Hz 工频陷波（仅在采样率允许时生效）。"""
    if fs < 100:
        return sig
    b, a = iirnotch(50.0, Q, fs=fs)
    return filtfilt(b, a, sig)


def zscore(sig: np.ndarray) -> np.ndarray:
    mu, sd = sig.mean(), sig.std()
    return (sig - mu) / (sd + 1e-8)


def preprocess_signals(ppg: np.ndarray, eda: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PPG+EDA 全流程预处理（骨架）。"""
    ppg = bandpass_ppg(ppg, FS_PPG)
    ppg = notch_50(ppg, FS_PPG)
    ppg = zscore(ppg)
    eda = lowpass_eda(eda, FS_EDA)
    eda = zscore(eda)
    return ppg, eda
