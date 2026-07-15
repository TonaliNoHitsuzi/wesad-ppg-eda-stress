"""为 15 个被试各生成一个静态 JSON，供前端被试切换。

每被试文件 frontend/public/data/subjects/S{X}.json 含：
  - 代表性 10s 窗口 PPG/EDA（取干净压力段，生理特征最显著）
  - FFT 频谱 + STFT 时频（压力段）
  - 预测分布：从该被试 LOSO 混淆矩阵派生（真值 vs 预测占比）
  - 该被试作为测试折的个人准确率/macro_F1

切换 = fetch 一个文件，瞬时。
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import (load_subject, extract_wrist_signals, resample_label,
                             FS_BVP, FS_EDA, VALID_SUBJECTS, list_available_subjects)
from src.preprocess.filters import bandpass_ppg, lowpass_eda, per_subject_zscore
from src.features.analysis import fft_psd, band_power, stft_magnitude, VLF_BAND, LF_BAND, HF_BAND

OUT = ROOT / "frontend" / "public" / "data" / "subjects"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["baseline", "stress", "amusement"]

# 读 LOSO 结果，取每被试混淆矩阵 + 个人指标
with open(ROOT / "results" / "json" / "loso_dual.json", encoding="utf-8") as f:
    DUAL = json.load(f)


def find_clean_segment(label_64, lab_val, length_ppg):
    """在 label_64 中找一段 length_ppg 长、主标签为 lab_val 且纯净度 100% 的区间。"""
    for s in range(0, len(label_64) - length_ppg, FS_BVP):
        if (label_64[s:s + length_ppg] == lab_val).mean() == 1.0:
            return s, s + length_ppg
    return None, None


def prediction_from_confusion(cm_sub):
    """从被试 3x3 混淆矩阵派生预测分布与主要预测状态。"""
    cm = np.array(cm_sub, dtype=float)
    total = cm.sum()
    if total == 0:
        return None
    col = cm.sum(axis=0)  # 各预测类样本数
    pred_dist = col / total
    maj = int(np.argmax(col))
    return {
        "true_label": LABELS[int(np.argmax(cm.sum(axis=1)))],
        "predicted_label": LABELS[maj],
        "probabilities": {LABELS[i]: round(float(pred_dist[i]), 3) for i in range(3)},
        "confidence": round(float(pred_dist[maj]), 3),
    }


def process_subject(sid):
    raw = load_subject(sid)
    sig = extract_wrist_signals(raw)
    bvp_z, _, _ = per_subject_zscore(bandpass_ppg(sig["bvp"]))
    eda_z, _, _ = per_subject_zscore(lowpass_eda(sig["eda"]))
    bvp_f = bandpass_ppg(sig["bvp"])
    label_64 = resample_label(sig["label"], len(sig["bvp"]), FS_BVP)
    N_PPG, N_EDA = 10 * FS_BVP, 10 * FS_EDA

    # 代表性窗口：优先压力段(label=2)，否则基线(1)
    win_s = None
    for lab in (2, 1, 3):
        s, _ = find_clean_segment(label_64, lab, N_PPG)
        if s is not None:
            win_s = s
            break
    if win_s is None:
        win_s = 0
    e0 = win_s // (FS_BVP // FS_EDA)

    # FFT 30s 段（同状态）
    seg_s = None
    for lab in (2, 1, 3):
        s, e = find_clean_segment(label_64, lab, 30 * FS_BVP)
        if s is not None:
            seg_s = (s, e); break
    seg_s = seg_s or (0, 30 * FS_BVP)
    f, p = fft_psd(bvp_f[seg_s[0]:seg_s[1]], FS_BVP)
    fmask = f <= 8

    # STFT 60s 段
    seg60 = None
    for lab in (2, 1, 3):
        s, e = find_clean_segment(label_64, lab, 60 * FS_BVP)
        if s is not None:
            seg60 = (s, e); break
    seg60 = seg60 or (0, 60 * FS_BVP)
    t, freqs, mag = stft_magnitude(bvp_f[seg60[0]:seg60[1]], FS_BVP, nperseg=256, noverlap=192)
    smask = freqs <= 8

    per = DUAL["per_subject"].get(sid, {})
    pred = prediction_from_confusion(per.get("confusion"))

    return {
        "subject_id": sid,
        "recording_duration_min": round(len(sig["bvp"]) / FS_BVP / 60),
        "current_state": pred["predicted_label"] if pred else "stress",
        "confidence": pred["confidence"] if pred else 0,
        "signal_ppg": {
            "sampling_rate": FS_BVP, "duration_seconds": 10,
            "data": bvp_z[win_s:win_s + N_PPG].round(4).tolist(),
            "timestamps": (np.arange(N_PPG) / FS_BVP).round(4).tolist(),
        },
        "signal_eda": {
            "sampling_rate": FS_EDA, "duration_seconds": 10,
            "data": eda_z[e0:e0 + N_EDA].round(4).tolist(),
            "timestamps": (np.arange(N_EDA) / FS_EDA).round(4).tolist(),
        },
        "spectrum": {
            "frequencies": f[fmask].round(4).tolist(),
            "power_db": p[fmask].round(3).tolist(),
            "bands": {
                "vlf": {"range": list(VLF_BAND), "power": band_power(f, p, VLF_BAND)},
                "lf": {"range": list(LF_BAND), "power": band_power(f, p, LF_BAND)},
                "hf": {"range": list(HF_BAND), "power": band_power(f, p, HF_BAND)},
            },
            "lf_hf_ratio": band_power(f, p, LF_BAND) / (band_power(f, p, HF_BAND) + 1e-12),
        },
        "spectrogram": {
            "times": t[::2].round(2).tolist(),
            "frequencies": freqs[smask].round(2).tolist(),
            "magnitude": mag[smask][:, ::2].round(3).tolist(),
        },
        "prediction": pred,
        "loso": {
            "accuracy": round(per.get("accuracy", 0), 3),
            "macro_f1": round(per.get("macro_f1", 0), 3),
        },
    }


def main():
    subs = list_available_subjects()
    print(f"导出 {len(subs)} 个被试 -> {OUT}")
    for sid in subs:
        rec = process_subject(sid)
        with open(OUT / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        pr = rec["prediction"]["probabilities"] if rec["prediction"] else {}
        print(f"  {sid}: state={rec['current_state']} conf={rec['confidence']:.2f} "
              f"loso_acc={rec['loso']['accuracy']:.3f} pred={pr}")
    print("DONE")


if __name__ == "__main__":
    main()
