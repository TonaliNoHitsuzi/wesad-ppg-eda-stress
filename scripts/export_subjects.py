"""为 15 个被试各生成一个静态 JSON，含三状态（基线/压力/娱乐）的可视图。

每被试文件 frontend/public/data/subjects/S{X}.json：
  states.{baseline|stress|amusement}: 该状态干净段的 PPG/EDA/FFT/STFT
  recall: 逐类检出率（从该被试 LOSO 混淆矩阵派生）
  prediction.probabilities: 全窗口预测分布
  loso: 该被试个人 LOSO 指标
前端可切换状态查看对应波形/频谱/时频。
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import (load_subject, extract_wrist_signals, resample_label,
                             FS_BVP, FS_EDA, list_available_subjects)
from src.preprocess.filters import bandpass_ppg, lowpass_eda, per_subject_zscore
from src.features.analysis import fft_psd, band_power, stft_magnitude, VLF_BAND, LF_BAND, HF_BAND

OUT = ROOT / "frontend" / "public" / "data" / "subjects"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["baseline", "stress", "amusement"]
LAB_VAL = {"baseline": 1, "stress": 2, "amusement": 3}

with open(ROOT / "results" / "json" / "loso_dual.json", encoding="utf-8") as f:
    DUAL = json.load(f)


def find_clean_segment(label_64, lab_val, length_ppg):
    for s in range(0, len(label_64) - length_ppg, FS_BVP):
        if (label_64[s:s + length_ppg] == lab_val).mean() == 1.0:
            return s, s + length_ppg
    return None, None


def state_view(bvp_f, bvp_z, eda_z, label_64, lab_val):
    """返回某状态的 {signal_ppg, signal_eda, spectrum, spectrogram}，无干净段则 None。"""
    N_PPG, N_EDA = 10 * FS_BVP, 10 * FS_EDA
    ws, _ = find_clean_segment(label_64, lab_val, N_PPG)
    if ws is None:
        return None
    e0 = ws // (FS_BVP // FS_EDA)
    # FFT 30s
    s30, e30 = find_clean_segment(label_64, lab_val, 30 * FS_BVP)
    if s30 is None:
        s30, e30 = ws, ws + 30 * FS_BVP
    f, p = fft_psd(bvp_f[s30:e30], FS_BVP)
    fm = f <= 8
    # STFT 60s
    s60, e60 = find_clean_segment(label_64, lab_val, 60 * FS_BVP)
    if s60 is None:
        s60, e60 = ws, min(ws + 60 * FS_BVP, len(bvp_f))
    t, freqs, mag = stft_magnitude(bvp_f[s60:e60], FS_BVP, nperseg=256, noverlap=192)
    sm = freqs <= 8
    return {
        "signal_ppg": {
            "sampling_rate": FS_BVP, "duration_seconds": 10,
            "data": bvp_z[ws:ws + N_PPG].round(4).tolist(),
            "timestamps": (np.arange(N_PPG) / FS_BVP).round(4).tolist(),
        },
        "signal_eda": {
            "sampling_rate": FS_EDA, "duration_seconds": 10,
            "data": eda_z[e0:e0 + N_EDA].round(4).tolist(),
            "timestamps": (np.arange(N_EDA) / FS_EDA).round(4).tolist(),
        },
        "spectrum": {
            "frequencies": f[fm].round(4).tolist(),
            "power_db": p[fm].round(3).tolist(),
            "bands": {
                "vlf": {"range": list(VLF_BAND), "power": band_power(f, p, VLF_BAND)},
                "lf": {"range": list(LF_BAND), "power": band_power(f, p, LF_BAND)},
                "hf": {"range": list(HF_BAND), "power": band_power(f, p, HF_BAND)},
            },
            "lf_hf_ratio": band_power(f, p, LF_BAND) / (band_power(f, p, HF_BAND) + 1e-12),
        },
        "spectrogram": {
            "times": t[::2].round(2).tolist(),
            "frequencies": freqs[sm].round(2).tolist(),
            "magnitude": mag[sm][:, ::2].round(3).tolist(),
        },
    }


def metrics_from_confusion(cm_sub):
    cm = np.array(cm_sub, dtype=float)
    total = cm.sum()
    if total == 0:
        return None
    col = cm.sum(axis=0); row = cm.sum(axis=1)
    pred_dist = col / total
    recall = np.diag(cm) / np.clip(row, 1, None)
    maj = int(np.argmax(col))
    return {
        "predicted_label": LABELS[maj],
        "probabilities": {LABELS[i]: round(float(pred_dist[i]), 3) for i in range(3)},
        "recall": {LABELS[i]: round(float(recall[i]), 3) for i in range(3)},
        "confidence": round(float(pred_dist[maj]), 3),
    }


def process_subject(sid):
    raw = load_subject(sid)
    sig = extract_wrist_signals(raw)
    bvp_z, _, _ = per_subject_zscore(bandpass_ppg(sig["bvp"]))
    eda_z, _, _ = per_subject_zscore(lowpass_eda(sig["eda"]))
    bvp_f = bandpass_ppg(sig["bvp"])
    label_64 = resample_label(sig["label"], len(sig["bvp"]), FS_BVP)

    states = {}
    for name, lv in LAB_VAL.items():
        v = state_view(bvp_f, bvp_z, eda_z, label_64, lv)
        if v is not None:
            states[name] = v

    per = DUAL["per_subject"].get(sid, {})
    m = metrics_from_confusion(per.get("confusion"))
    return {
        "subject_id": sid,
        "recording_duration_min": round(len(sig["bvp"]) / FS_BVP / 60),
        "states": states,
        "recall": m["recall"] if m else {},
        "prediction": {
            "predicted_label": m["predicted_label"] if m else "stress",
            "probabilities": m["probabilities"] if m else {},
        },
        "confidence": m["confidence"] if m else 0,
        "loso": {"accuracy": round(per.get("accuracy", 0), 3),
                 "macro_f1": round(per.get("macro_f1", 0), 3)},
    }


def main():
    subs = list_available_subjects()
    print(f"导出 {len(subs)} 个被试 -> {OUT}")
    for sid in subs:
        rec = process_subject(sid)
        with open(OUT / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        r = rec["recall"]
        print(f"  {sid}: states={list(rec['states'].keys())} "
              f"recall 压力={r.get('stress',0)} 基线={r.get('baseline',0)} 娱乐={r.get('amusement',0)}")
    print("DONE")


if __name__ == "__main__":
    main()
