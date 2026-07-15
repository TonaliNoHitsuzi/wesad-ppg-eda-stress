"""信号分析：FFT 功率谱、STFT 时频、HRV 特征提取。

用于可视化与可解释性分析（论文 fig01-05）。HRV 频域分析采用 60 秒长窗
以保证频带估计的可靠性（10 秒窗对 LF/HF 太短）。
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, stft as scipy_stft, butter, filtfilt, welch
from scipy.interpolate import interp1d

from src.data.loader import (load_subject, extract_wrist_signals,
                             resample_label, LABEL_TO_STATE, FS_BVP, FS_CHEST)
from src.preprocess.filters import bandpass_ppg, lowpass_eda, per_subject_zscore

# HRV 频带（Hz）
VLF_BAND = (0.003, 0.04)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.4)


# ─────────────────────────────────────────────
# 频域
# ─────────────────────────────────────────────
def fft_psd(sig: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """单边功率谱密度（dB）。返回 (frequencies, power_db)。"""
    sig = sig - sig.mean()
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(sig * np.hanning(n))) ** 2
    mag[0] = 0  # 去 DC
    psd_db = 10 * np.log10(mag + 1e-12)
    return freqs, psd_db


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """指定频带的积分功率。"""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return 0.0
    integ = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x 用 trapezoid
    return float(integ(psd[mask], freqs[mask]))


# ─────────────────────────────────────────────
# 时频
# ─────────────────────────────────────────────
def stft_magnitude(sig: np.ndarray, fs: int, nperseg: int = 256,
                   noverlap: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT 幅度谱。返回 (times, frequencies, magnitude)。Hanning 窗。"""
    f, t, Z = scipy_stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                         window="hann", boundary=None)
    return t, f, np.abs(Z)


# ─────────────────────────────────────────────
# HRV（时域 + 频域），输入为 PPG 片段
# ─────────────────────────────────────────────
def hrv_from_ppg(ppg: np.ndarray, fs: int = FS_BVP) -> dict | None:
    """从 PPG 片段提取 HRV。返回 None 表示峰值检测失败。

    峰检测采用归一化 + prominence 阈值，剔除重搏切迹与噪声峰。
    """
    # 归一化后检测，prominence 抑制重搏切迹（PPG 每周期有 systolic + dicrotic 两峰）
    sn = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
    peaks, _ = find_peaks(sn, distance=int(0.4 * fs), prominence=0.5)
    if len(peaks) < 5:
        return None
    rr = np.diff(peaks) / fs  # RR 间期（秒）
    # 滤除异常 RR（<0.3s 或 >2s 多为伪检）
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) < 4 or rr.std() == 0:
        return None
    rr_ms = rr * 1000.0
    sdnn = float(np.std(rr_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    hr_mean = float(60.0 / np.mean(rr))

    # 频域：RR 重采样到 4Hz，Welch 估功率谱，积分 LF/HF
    try:
        from scipy.signal import welch
        t_rr = np.cumsum(rr)
        f_interp = interp1d(t_rr, rr, kind="linear", fill_value="extrapolate")
        fs_r = 4.0
        t_even = np.arange(0, t_rr[-1], 1.0 / fs_r)
        rr_even = f_interp(t_even)
        rr_even -= rr_even.mean()
        nperseg = min(256, len(rr_even))
        freqs, psd = welch(rr_even, fs=fs_r, nperseg=nperseg, scaling="density")
        lf = band_power(freqs, psd, LF_BAND)
        hf = band_power(freqs, psd, HF_BAND)
        lf_hf = float(lf / (hf + 1e-12))
    except Exception:
        lf_hf = float("nan")
    return {"SDNN": sdnn, "RMSSD": rmssd, "HR_mean": hr_mean, "LF_HF": lf_hf}


# ─────────────────────────────────────────────
# 按状态提取长窗（60s）用于 HRV 分析
# ─────────────────────────────────────────────
def extract_state_long_segments(subject_id: str, win_sec: float = 60.0,
                                step_sec: float = 30.0) -> dict[str, list[np.ndarray]]:
    """返回 {state: [ppg_segment_60s, ...]}（已滤波，未 z-score，保留 SCL 信息）。"""
    raw = load_subject(subject_id)
    sig = extract_wrist_signals(raw)
    bvp_f = bandpass_ppg(sig["bvp"])
    label_64 = resample_label(sig["label"], len(bvp_f), FS_BVP)
    win = int(win_sec * FS_BVP)
    step = int(step_sec * FS_BVP)
    out: dict[str, list[np.ndarray]] = {s: [] for s in ("baseline", "stress", "amusement")}
    start = 0
    while start + win <= len(bvp_f):
        seg_lab = label_64[start:start + win]
        # 窗内主标签
        valid = seg_lab[np.isin(seg_lab, (1, 2, 3))]
        if len(valid) > 0:
            lab = np.bincount(valid, minlength=4)[1:4].argmax() + 1
            purity = (seg_lab == lab).mean()
            if purity >= 0.9 and lab in LABEL_TO_STATE:
                out[LABEL_TO_STATE[lab]].append(bvp_f[start:start + win])
        start += step
    return out


def collect_hrv_all_subjects(subjects: list[str]) -> dict[str, dict]:
    """汇总所有被试的 HRV 特征（基于胸部 ECG R 峰，金标准），按状态分组。"""
    feats = ["SDNN", "RMSSD", "LF_HF", "HR_mean"]
    agg = {f: {s: [] for s in ("baseline", "stress", "amusement")} for f in feats}
    for sid in subjects:
        try:
            segs = extract_state_ecg_segments(sid)
        except Exception:
            continue
        for state, arrs in segs.items():
            for ecg in arrs:
                h = hrv_from_ecg(ecg)
                if h is None:
                    continue
                for f in feats:
                    val = h[f]
                    if val is not None and not np.isnan(val):
                        agg[f][state].append(val)
    return agg


# ─────────────────────────────────────────────
# ECG R 峰检测与 HRV（金标准，700Hz）
# ─────────────────────────────────────────────
def ecg_bandpass(sig: np.ndarray, fs: int = FS_CHEST,
                 low: float = 5.0, high: float = 15.0) -> np.ndarray:
    """ECG 带通 5–15 Hz（QRS 频段）。"""
    nyq = fs / 2.0
    b, a = butter(N=3, Wn=[low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def detect_r_peaks(ecg: np.ndarray, fs: int = FS_CHEST) -> np.ndarray:
    """Pan-Tompkins 简化版 R 峰检测：带通→微分→平方→滑动积分→峰值。"""
    f = ecg_bandpass(ecg, fs)
    diff = np.append(np.diff(f), 0)  # 一阶微分增强 QRS 斜率
    sq = diff ** 2
    win = max(1, int(0.08 * fs))  # 80ms 积分窗
    ma = np.convolve(sq, np.ones(win) / win, mode="same")
    if ma.max() <= 0:
        return np.array([], dtype=int)
    thr = 0.35 * ma.max()
    peaks, _ = find_peaks(ma, distance=int(0.33 * fs), height=thr)
    # 精化：在每个积分峰附近找原始滤波信号的真实极大
    refined = []
    half = int(0.05 * fs)
    for p in peaks:
        lo, hi = max(0, p - half), min(len(f), p + half)
        if hi > lo:
            refined.append(lo + int(np.argmax(f[lo:hi])))
    return np.unique(np.asarray(refined, dtype=int))


def hrv_from_ecg(ecg: np.ndarray, fs: int = FS_CHEST) -> dict | None:
    """从 ECG 片段提取 HRV。"""
    peaks = detect_r_peaks(ecg, fs)
    if len(peaks) < 6:
        return None
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]  # 滤除伪检
    # 异常 RR 剔除（偏离局部中位数 >20%）
    med = np.median(rr)
    rr = rr[np.abs(rr - med) <= 0.20 * med]
    if len(rr) < 5 or rr.std() == 0:
        return None
    rr_ms = rr * 1000.0
    sdnn = float(np.std(rr_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    hr_mean = float(60.0 / np.mean(rr))
    try:
        t_rr = np.cumsum(rr)
        f_interp = interp1d(t_rr, rr, kind="linear", fill_value="extrapolate")
        fs_r = 4.0
        t_even = np.arange(0, t_rr[-1], 1.0 / fs_r)
        rr_even = f_interp(t_even) - f_interp(t_even).mean()
        nperseg = min(256, len(rr_even))
        freqs, psd = welch(rr_even, fs=fs_r, nperseg=nperseg, scaling="density")
        lf = band_power(freqs, psd, LF_BAND)
        hf = band_power(freqs, psd, HF_BAND)
        lf_hf = float(lf / (hf + 1e-12))
    except Exception:
        lf_hf = float("nan")
    return {"SDNN": sdnn, "RMSSD": rmssd, "HR_mean": hr_mean, "LF_HF": lf_hf}


def extract_state_ecg_segments(subject_id: str, win_sec: float = 60.0,
                               step_sec: float = 30.0) -> dict[str, list[np.ndarray]]:
    """按状态提取 ECG 长窗（700Hz，与标签同源，无需重采样）。"""
    raw = load_subject(subject_id)
    ecg = np.asarray(raw["signal"]["chest"]["ECG"]).reshape(-1)
    label = np.asarray(raw["label"]).reshape(-1)
    win = int(win_sec * FS_CHEST)
    step = int(step_sec * FS_CHEST)
    out = {s: [] for s in ("baseline", "stress", "amusement")}
    start = 0
    while start + win <= len(ecg):
        seg = label[start:start + win]
        valid = seg[np.isin(seg, (1, 2, 3))]
        if len(valid) > 0:
            lab = np.bincount(valid, minlength=4)[1:4].argmax() + 1
            if (seg == lab).mean() >= 0.9 and lab in LABEL_TO_STATE:
                out[LABEL_TO_STATE[lab]].append(ecg[start:start + win])
        start += step
    return out
