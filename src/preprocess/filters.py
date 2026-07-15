"""信号预处理：滤波 / per-subject z-score / SQI 加权融合。

设计决策（13 条锁定）：
  - N4：per-subject z-score。每被试用自己全记录的无标签 μ/σ 标准化，
        既保留条件间的相对偏离（stress 抬升 EDA → 正向 z），又去除被试间基线噪声。
  - 第4条 SQI：模板匹配（与理想 PPG 归一化互相关）+ 幅度合理性 + 峰值计数合理性，
        加权融合 SQI = 0.4×correlation + 0.3×amplitude + 0.3×peak_count。
        权重为启发式设定，论文中需注明未做敏感性搜索。
"""
from __future__ import annotations
import numpy as np
from scipy.signal import cheby2, filtfilt, iirnotch, find_peaks

# 采样率（与 loader 一致）
FS_PPG = 64
FS_EDA = 4

# SQI 权重（启发式，论文需注明）
SQI_W_CORR = 0.4
SQI_W_AMP = 0.3
SQI_W_PEAK = 0.3
SQI_REJECT = 0.5  # 窗口 SQI 低于此值丢弃


# ─────────────────────────────────────────────
# 滤波器
# ─────────────────────────────────────────────
def bandpass_ppg(sig: np.ndarray, fs: int = FS_PPG) -> np.ndarray:
    """Chebyshev II 型带通 0.5–8 Hz（PPG 有生理意义的频段）。"""
    nyq = fs / 2.0
    b, a = cheby2(N=4, rs=20, Wn=[0.5 / nyq, 8.0 / nyq], btype="bandpass")
    return filtfilt(b, a, sig)


def lowpass_eda(sig: np.ndarray, fs: int = FS_EDA) -> np.ndarray:
    """EDA 1 Hz 低通（保留慢变 SCL，抑制高频噪声）。"""
    nyq = fs / 2.0
    b, a = cheby2(N=4, rs=20, Wn=1.0 / nyq, btype="lowpass")
    return filtfilt(b, a, sig)


def notch_50(sig: np.ndarray, fs: int, Q: float = 35.0) -> np.ndarray:
    """50 Hz 工频陷波（仅在采样率 >100Hz 时生效，PPG 64Hz 实际跳过）。"""
    if fs < 100:
        return sig
    b, a = iirnotch(50.0, Q, fs=fs)
    return filtfilt(b, a, sig)


# ─────────────────────────────────────────────
# per-subject z-score（N4）
# ─────────────────────────────────────────────
def per_subject_zscore(sig: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """用被试自身全记录的 μ/σ 做标准化。

    Returns
    -------
    z, mu, sigma : 标准化后的信号及所用统计量（测试被试复用训练统计量时可直接传入）。
    """
    mu = float(np.mean(sig))
    sigma = float(np.std(sig)) + eps
    return (sig - mu) / sigma, mu, sigma


def apply_zscore(sig: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """用已有的 μ/σ 标准化（测试被试专用，杜绝泄露）。"""
    return (sig - mu) / sigma


# ─────────────────────────────────────────────
# 完整预处理（单被试）
# ─────────────────────────────────────────────
def preprocess_subject(bvp: np.ndarray, eda: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """单被试：滤波 → per-subject z-score。返回处理后的 BVP/EDA 与统计量。

    统计量 dict 供测试被试复用（虽然 per-subject 用自身统计量，
    但保留接口便于消融实验切换为 global 方案）。
    """
    bvp_f = bandpass_ppg(bvp, FS_PPG)
    bvp_f = notch_50(bvp_f, FS_PPG)
    bvp_z, bvp_mu, bvp_sd = per_subject_zscore(bvp_f)

    eda_f = lowpass_eda(eda, FS_EDA)
    eda_z, eda_mu, eda_sd = per_subject_zscore(eda_f)

    stats = {"bvp_mu": bvp_mu, "bvp_sd": bvp_sd,
             "eda_mu": eda_mu, "eda_sd": eda_sd}
    return bvp_z.astype(np.float32), eda_z.astype(np.float32), stats


def preprocess_signals(ppg: np.ndarray, eda: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """兼容旧接口：保留以避免 import 报错。"""
    return preprocess_subject(ppg, eda)[:2]


# ─────────────────────────────────────────────
# SQI：加权融合（第4条裁定）
# ─────────────────────────────────────────────
def _ideal_ppg_template(n: int = 640) -> np.ndarray:
    """构造一个理想 PPG 脉搏模板（约 1.2Hz，72bpm），用于互相关模板匹配。"""
    t = np.arange(n) / FS_PPG
    # 主峰 + 重搏切迹的简化形态（用 clip 避免 sqrt 负数警告）
    pos = np.clip(beats := np.sin(2 * np.pi * 1.2 * t), 0, None) ** 0.5
    neg = np.clip(-beats, 0, None) ** 0.5
    beats = np.where(beats > 0, pos, -0.3 * neg)
    return beats / (np.linalg.norm(beats) + 1e-8)


_TEMPLATE = None


def _sqi_correlation(window: np.ndarray) -> float:
    """与理想模板的归一化互相关最大值。"""
    global _TEMPLATE
    if _TEMPLATE is None or len(_TEMPLATE) != len(window):
        _TEMPLATE = _ideal_ppg_template(len(window))
    w = (window - window.mean()) / (window.std() + 1e-8)
    corr = np.correlate(w, _TEMPLATE, "full")
    return float(np.max(np.abs(corr)) / len(window))


def _sqi_amplitude(window: np.ndarray) -> float:
    """幅度合理性：标准差落在合理区间得高分。"""
    sd = float(np.std(window))
    # 经验区间：太小说明近平直，太大说明有大幅伪影
    if sd < 0.05:
        return 0.0
    if sd > 5.0:
        return 0.2
    return 1.0


def _sqi_peak_count(window: np.ndarray) -> float:
    """峰值计数合理性：推算心率应落在 30–180 bpm。"""
    peaks, _ = find_peaks(window, distance=int(0.4 * FS_PPG))
    n_beats = len(peaks)
    dur = len(window) / FS_PPG
    bpm = n_beats / dur * 60.0
    if 30 <= bpm <= 180:
        return 1.0
    if 20 <= bpm < 30 or 180 < bpm <= 200:
        return 0.5
    return 0.0


def compute_sqi(ppg_window: np.ndarray) -> float:
    """单窗口 SQI = 0.4×corr + 0.3×amp + 0.3×peak。"""
    return (SQI_W_CORR * _sqi_correlation(ppg_window)
            + SQI_W_AMP * _sqi_amplitude(ppg_window)
            + SQI_W_PEAK * _sqi_peak_count(ppg_window))


def sqi_mask(ppg_windows: np.ndarray, thresh: float = SQI_REJECT) -> np.ndarray:
    """批量计算窗口 SQI，返回保留掩码（bool 数组）。"""
    return np.array([compute_sqi(w) >= thresh for w in ppg_windows], dtype=bool)
