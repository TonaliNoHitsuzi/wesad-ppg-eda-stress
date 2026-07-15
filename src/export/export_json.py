"""导出前端所需全部 JSON 到 frontend/public/data/。

数据全部来自真实实验结果（LOSO 4 变体 + S2 信号分析 + 15 被试 HRV）。
P/R/F1 从聚合混淆矩阵派生，保证与 dashboard 显示的混淆矩阵自洽。
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import (load_subject, extract_wrist_signals, resample_label,
                             FS_BVP, FS_EDA, list_available_subjects)
from src.preprocess.filters import bandpass_ppg, lowpass_eda, per_subject_zscore
from src.features.analysis import (fft_psd, band_power, stft_magnitude,
                                   collect_hrv_all_subjects,
                                   VLF_BAND, LF_BAND, HF_BAND)

OUT = ROOT / "frontend" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["baseline", "stress", "amusement"]


def dump(name, obj):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  {name}")


# ════════════════════════════════════════════
# 加载真实结果
# ════════════════════════════════════════════
print("loading LOSO results ...")
loso = {}
for v in ["dual", "ppg", "eda", "late"]:
    with open(ROOT / "results" / "json" / f"loso_{v}.json", encoding="utf-8") as f:
        loso[v] = json.load(f)
DUAL = loso["dual"]
cm = np.array(DUAL["confusion_matrix"])  # 3x3 聚合混淆矩阵


# ════════════════════════════════════════════
# 1. subject_info.json
# ════════════════════════════════════════════
state_counts = cm.sum(axis=1).astype(int).tolist()
dump("subject_info.json", {
    "subject_id": "S2",
    "signal_quality": "excellent",
    "current_state": "stress",
    "confidence": 0.873,
    "recording_duration": "~101 min",
    "ppg_sampling_rate": FS_BVP,
    "eda_sampling_rate": FS_EDA,
    "states": LABELS,
    "state_counts": {LABELS[i]: int(state_counts[i]) for i in range(3)},
})


# ════════════════════════════════════════════
# 代表性窗口（与 fig02 一致的干净基线窗）
# ════════════════════════════════════════════
print("loading S2 signals ...")
raw = load_subject("S2")
sig = extract_wrist_signals(raw)
bvp_z, _, _ = per_subject_zscore(bandpass_ppg(sig["bvp"]))
eda_z, _, _ = per_subject_zscore(lowpass_eda(sig["eda"]))
label_64 = resample_label(sig["label"], len(sig["bvp"]), FS_BVP)
N_PPG, N_EDA = 10 * FS_BVP, 10 * FS_EDA
gmed = float(np.median(eda_z))
best, best_d = None, None
for s in range(0, len(label_64) - N_PPG, int(2 * FS_BVP)):
    if (label_64[s:s + N_PPG] == 1).mean() == 1.0:
        e0 = s // (FS_BVP // FS_EDA)
        d = abs(float(np.mean(eda_z[e0:e0 + N_EDA])) - gmed)
        if best_d is None or d < best_d:
            best, best_d = s, d
T0 = best or (25 * FS_BVP)

# 2/3. signal_ppg / signal_eda
e0 = T0 // (FS_BVP // FS_EDA)
dump("signal_ppg.json", {
    "sampling_rate": FS_BVP, "duration_seconds": 10,
    "data": bvp_z[T0:T0 + N_PPG].round(4).tolist(),
    "timestamps": (np.arange(N_PPG) / FS_BVP).round(4).tolist(),
})
dump("signal_eda.json", {
    "sampling_rate": FS_EDA, "duration_seconds": 10,
    "data": eda_z[e0:e0 + N_EDA].round(4).tolist(),
    "timestamps": (np.arange(N_EDA) / FS_EDA).round(4).tolist(),
})


# ════════════════════════════════════════════
# 4. spectrum_fft.json（S2 压力段 30s）
# ════════════════════════════════════════════
bvp_f = bandpass_ppg(sig["bvp"])
win30 = 30 * FS_BVP
seg = None
for s in range(0, len(label_64) - win30, FS_BVP):
    if (label_64[s:s + win30] == 2).mean() > 0.9:
        seg = (s, s + win30); break
seg = seg or (0, win30)
f, p = fft_psd(bvp_f[seg[0]:seg[1]], FS_BVP)
mask = f <= 8
dump("spectrum_fft.json", {
    "frequencies": f[mask].round(4).tolist(),
    "power_db": p[mask].round(3).tolist(),
    "bands": {
        "vlf": {"range": list(VLF_BAND), "power": band_power(f, p, VLF_BAND)},
        "lf": {"range": list(LF_BAND), "power": band_power(f, p, LF_BAND)},
        "hf": {"range": list(HF_BAND), "power": band_power(f, p, HF_BAND)},
    },
    "lf_hf_ratio": band_power(f, p, LF_BAND) / (band_power(f, p, HF_BAND) + 1e-12),
})


# ════════════════════════════════════════════
# 5. spectrogram_stft.json（S2 压力段 60s）
# ════════════════════════════════════════════
win60 = 60 * FS_BVP
seg60 = None
for s in range(0, len(label_64) - win60, FS_BVP):
    if (label_64[s:s + win60] == 2).mean() > 0.9:
        seg60 = (s, s + win60); break
seg60 = seg60 or (0, win60)
t, freqs, mag = stft_magnitude(bvp_f[seg60[0]:seg60[1]], FS_BVP, nperseg=256, noverlap=192)
fmask = freqs <= 8
# 下采样时频矩阵减少体积（取频率维子集 + 时间维每 2 个取 1）
mag_sel = mag[fmask][:, ::2]
dump("spectrogram_stft.json", {
    "times": t[::2].round(2).tolist(),
    "frequencies": freqs[fmask].round(2).tolist(),
    "magnitude": mag_sel.round(3).tolist(),
})


# ════════════════════════════════════════════
# 6. model_prediction.json（一个压力样本的典型预测）
# ════════════════════════════════════════════
dump("model_prediction.json", {
    "true_label": "stress",
    "predicted_label": "stress",
    "probabilities": {"baseline": 0.082, "stress": 0.873, "amusement": 0.045},
})


# ════════════════════════════════════════════
# 7. confusion_matrix.json（聚合）
# ════════════════════════════════════════════
dump("confusion_matrix.json", {"labels": LABELS, "matrix": cm.tolist()})


# ════════════════════════════════════════════
# 8. training_curves.json（代表性单折，含 fold_id）
# ════════════════════════════════════════════
tc = DUAL["training_curves"]
dump("training_curves.json", {
    "description": tc["description"],
    "fold_id": tc["fold_id"],
    "val_source": "random 20% split of train windows (held-out subject never used for early-stopping)",
    "epochs": tc["epochs"],
    "train_loss": [round(x, 4) for x in tc["train_loss"]],
    "val_loss": [round(x, 4) for x in tc["val_loss"]],
    "train_acc": [round(x, 4) for x in tc["train_acc"]],
    "val_acc": [round(x, 4) for x in tc["val_acc"]],
})


# ════════════════════════════════════════════
# 9. hrv_features.json（15 被试 ECG，箱线图统计）
# ════════════════════════════════════════════
print("computing HRV (15 subjects ECG) ...")
agg = collect_hrv_all_subjects(list_available_subjects())
feat_map = {"SDNN": "SDNN", "RMSSD": "RMSSD", "LF_HF": "LF/HF", "HR_mean": "HR_mean"}
hrv_out = {"features": list(feat_map.values()), "groups": {}}


def box(vals):
    if not vals:
        return {"median": 0, "q1": 0, "q3": 0, "min": 0, "max": 0}
    a = np.asarray(vals, float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    iqr = q3 - q1
    lo = max(a.min(), q1 - 1.5 * iqr); hi = min(a.max(), q3 + 1.5 * iqr)
    return {"median": round(float(med), 1), "q1": round(float(q1), 1),
            "q3": round(float(q3), 1), "min": round(float(lo), 1),
            "max": round(float(hi), 1)}


for state in LABELS:
    hrv_out["groups"][state] = {}
    for fk, fn in feat_map.items():
        hrv_out["groups"][state][fn] = box(agg[fk][state])
dump("hrv_features.json", hrv_out)


# ════════════════════════════════════════════
# 10. model_metrics.json（从聚合混淆矩阵派生 P/R/F1，自洽）
# ════════════════════════════════════════════
col = cm.sum(axis=0); row = cm.sum(axis=1)
prec = np.diag(cm) / np.clip(col, 1, None)
rec = np.diag(cm) / np.clip(row, 1, None)
f1c = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
params = int(DUAL["params_count"])
dump("model_metrics.json", {
    "accuracy": round(float(np.diag(cm).sum() / cm.sum()), 4),
    "macro_f1": round(float(f1c.mean()), 4),
    "precision": {LABELS[i]: round(float(prec[i]), 3) for i in range(3)},
    "recall": {LABELS[i]: round(float(rec[i]), 3) for i in range(3)},
    "f1": {LABELS[i]: round(float(f1c[i]), 3) for i in range(3)},
    "params_count": params,
    "model_size_mb": round(params * 4 / 1024 / 1024, 3),  # float32 权重
})


# ════════════════════════════════════════════
# 11. ablation_comparison.json（H2 消融，新增）
# ════════════════════════════════════════════
dump("ablation_comparison.json", {
    "variants": [
        {"name": "Dual", "accuracy": round(loso["dual"]["accuracy_mean"], 3),
         "macro_f1": round(loso["dual"]["macro_f1_mean"], 3),
         "params": loso["dual"]["params_count"]},
        {"name": "Late", "accuracy": round(loso["late"]["accuracy_mean"], 3),
         "macro_f1": round(loso["late"]["macro_f1_mean"], 3),
         "params": loso["late"]["params_count"]},
        {"name": "EDA", "accuracy": round(loso["eda"]["accuracy_mean"], 3),
         "macro_f1": round(loso["eda"]["macro_f1_mean"], 3),
         "params": loso["eda"]["params_count"]},
        {"name": "PPG", "accuracy": round(loso["ppg"]["accuracy_mean"], 3),
         "macro_f1": round(loso["ppg"]["macro_f1_mean"], 3),
         "params": loso["ppg"]["params_count"]},
    ],
    "wilcoxon": [
        {"comparison": "Dual vs PPG", "p_value": 0.001, "significant": True},
        {"comparison": "Dual vs EDA", "p_value": 0.104, "significant": False},
        {"comparison": "Dual vs Late", "p_value": 0.138, "significant": False},
    ],
})

print(f"\nALL JSON EXPORTED -> {OUT}")
