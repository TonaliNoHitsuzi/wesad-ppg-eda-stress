"""生成全部论文图表（fig01-09 + 消融对比）。

输出：results/figures/figXX_*.png（300 dpi）
代表性被试：S2
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import format_axes
from src.data.loader import (load_subject, extract_wrist_signals, list_available_subjects,
                             FS_BVP, FS_EDA)
from src.preprocess.filters import bandpass_ppg, lowpass_eda, per_subject_zscore
from src.features.analysis import (fft_psd, band_power, stft_magnitude,
                                   collect_hrv_all_subjects,
                                   VLF_BAND, LF_BAND, HF_BAND)

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 10,
                     "axes.unicode_minus": False})
sns.set_style("whitegrid")
STATE_COLORS = {"baseline": "#4C9F70", "stress": "#D9534F", "amusement": "#5B9BD5"}
STATES = ["baseline", "stress", "amusement"]


def save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


# ════════════════════════════════════════════
# 加载代表性被试 S2
# ════════════════════════════════════════════
print("loading S2 ...")
raw = load_subject("S2")
sig = extract_wrist_signals(raw)
bvp_raw = sig["bvp"]
eda_raw = sig["eda"]
bvp_f = bandpass_ppg(bvp_raw)
bvp_f = bvp_f - bvp_f.mean()  # 仅去均值便于展示（z-score 留到 fig02 对比）
eda_f = lowpass_eda(eda_raw)
bvp_z, _, _ = per_subject_zscore(bandpass_ppg(bvp_raw))
eda_z, _, _ = per_subject_zscore(lowpass_eda(eda_raw))

N_PPG = 10 * FS_BVP  # 640
N_EDA = 10 * FS_EDA  # 40

# 数据驱动选窗：在干净基线段(label=1, 100%纯净)中选 EDA-z 最接近全局中位数的窗口，
# 保证 fig01/fig02 展示的是代表性信号而非过渡段离群值。
from src.data.loader import resample_label
label_64 = resample_label(sig["label"], len(bvp_raw), FS_BVP)
_gmed = float(np.median(eda_z))
_best, _best_d = None, None
for _s in range(0, len(label_64) - N_PPG, int(2 * FS_BVP)):
    if (label_64[_s:_s + N_PPG] == 1).mean() == 1.0:  # 纯净基线
        _e0 = _s // (FS_BVP // FS_EDA)
        _zm = float(np.mean(eda_z[_e0:_e0 + N_EDA]))
        _d = abs(_zm - _gmed)
        if _best_d is None or _d < _best_d:
            _best, _best_d = _s, _d
T0 = _best if _best is not None else (25 * FS_BVP)
print(f"  representative baseline window: T0={T0} ({T0/FS_BVP:.0f}s), "
      f"EDA-z mean={np.mean(eda_z[T0//(FS_BVP//FS_EDA):T0//(FS_BVP//FS_EDA)+N_EDA]):+.2f}")


# ════════════════════════════════════════════
# fig01 原始波形
# ════════════════════════════════════════════
print("[fig01] raw signals")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5))
t1 = np.arange(N_PPG) / FS_BVP + T0 / FS_BVP
ax1.plot(t1, bvp_raw[T0:T0 + N_PPG], color="#C0504D", lw=1)
format_axes(ax1, title="Raw PPG Signal (Subject S2, 10 s)",
            xlabel="Time", ylabel="Amplitude", unit_x="s", legend=False)
t2 = np.arange(N_EDA) / FS_EDA + T0 / FS_BVP
ax2.plot(t2, eda_raw[T0 // (FS_BVP // FS_EDA):T0 // (FS_BVP // FS_EDA) + N_EDA],
         color="#4A7ABC", lw=1.5, marker=".")
format_axes(ax2, title="Raw EDA Signal (Subject S2, 10 s)",
            xlabel="Time", ylabel="Skin Conductance", unit_x="s", unit_y="µS", legend=False)
fig.tight_layout()
save(fig, "fig01_raw_signals.png")


# ════════════════════════════════════════════
# fig02 滤波 + z-score 后波形
# ════════════════════════════════════════════
print("[fig02] filtered signals")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5))
ax1.plot(t1, bvp_z[T0:T0 + N_PPG], color="#C0504D", lw=1)
format_axes(ax1, title="Filtered & Z-scored PPG (Chebyshev bandpass 0.5–8 Hz)",
            xlabel="Time", ylabel="Z-score", unit_x="s", legend=False)
e0 = T0 // (FS_BVP // FS_EDA)
ax2.plot(t2, eda_z[e0:e0 + N_EDA], color="#4A7ABC", lw=1.5, marker=".")
format_axes(ax2, title="Filtered & Z-scored EDA (1 Hz low-pass)",
            xlabel="Time", ylabel="Z-score", unit_x="s", legend=False)
fig.tight_layout()
save(fig, "fig02_filtered_signals.png")


# ════════════════════════════════════════════
# fig03 FFT 功率谱（基线 vs 压力对比）
# ════════════════════════════════════════════
print("[fig03] FFT spectrum")
# 取 S2 一段压力段和基线段各 30s
from src.data.loader import resample_label, LABEL_TO_STATE
label_64 = resample_label(sig["label"], len(bvp_raw), FS_BVP)
win30 = 30 * FS_BVP
def pick_state_segment(lab_val, length=win30):
    for s in range(0, len(bvp_raw) - length, FS_BVP):
        seg = label_64[s:s + length]
        if np.isin(seg, (1, 2, 3)).mean() > 0.95 and np.bincount(seg.clip(0, 7), minlength=8)[lab_val] > 0.9 * length:
            return s, s + length
    return None, None
seg_specs = {}
for lab, name in [(1, "baseline"), (2, "stress")]:
    s, e = pick_state_segment(lab)
    if s is not None:
        f, p = fft_psd(bvp_f[s:e], FS_BVP)
        seg_specs[name] = (f, p)
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, (f, p) in seg_specs.items():
    ax.plot(f[f <= 8], p[f <= 8], color=STATE_COLORS[name], lw=1.5,
            label=f"{name.capitalize()} (30 s)", alpha=0.85)
# 标注三频带 VLF / LF / HF（与正文 4.2 节对齐）
ax.axvspan(0.003, 0.04, alpha=0.10, color="purple", label="VLF (0.003–0.04 Hz)")
ax.axvspan(0.04, 0.15, alpha=0.10, color="gold", label="LF (0.04–0.15 Hz)")
ax.axvspan(0.15, 0.40, alpha=0.10, color="darkorange", label="HF (0.15–0.40 Hz)")
_ylim_top = ax.get_ylim()[1] if seg_specs else 1
ax.text(0.021, _ylim_top * 0.95, "VLF", ha="center", fontsize=8, color="purple")
ax.text(0.095, _ylim_top * 0.95, "LF", ha="center", fontsize=8, color="olive")
ax.text(0.27, _ylim_top * 0.95, "HF", ha="center", fontsize=8, color="darkorange")
format_axes(ax, title="FFT Power Spectrum of PPG (Baseline vs Stress)",
            xlabel="Frequency", ylabel="Power", unit_x="Hz", unit_y="dB")
ax.set_xlim(0, 8)
fig.tight_layout()
save(fig, "fig03_fft_spectrum.png")


# ════════════════════════════════════════════
# fig04 STFT 时频热力图
# ════════════════════════════════════════════
print("[fig04] STFT spectrogram")
seg_s, seg_e = pick_state_segment(2, 60 * FS_BVP) or (0, 60 * FS_BVP)
seg_s = seg_s or 0
seg_e = seg_e or (60 * FS_BVP)
t, freqs, mag = stft_magnitude(bvp_f[seg_s:seg_e], FS_BVP, nperseg=256, noverlap=192)
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.pcolormesh(t, freqs[freqs <= 8], mag[freqs <= 8], cmap="viridis",
                   shading="gouraud")
cb = fig.colorbar(im, ax=ax)
cb.set_label("Magnitude")
format_axes(ax, title="STFT Spectrogram of PPG (Subject S2, Stress segment)",
            xlabel="Time", ylabel="Frequency", unit_x="s", unit_y="Hz", legend=False)
ax.set_ylim(0, 8)
fig.tight_layout()
save(fig, "fig04_stft_spectrogram.png")


# ════════════════════════════════════════════
# fig05 HRV 特征箱线图
# ════════════════════════════════════════════
print("[fig05] HRV boxplots (all subjects) ...")
agg = collect_hrv_all_subjects(list_available_subjects())
feat_titles = {"SDNN": "SDNN", "RMSSD": "RMSSD", "LF_HF": "LF/HF Ratio", "HR_mean": "Mean HR"}
feat_units = {"SDNN": "ms", "RMSSD": "ms", "LF_HF": "", "HR_mean": "bpm"}
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, feat in zip(axes, ["SDNN", "RMSSD", "LF_HF", "HR_mean"]):
    data = [agg[feat][s] for s in STATES]
    bp = ax.boxplot(data, tick_labels=[s.capitalize() for s in STATES],
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.5))
    for patch, s in zip(bp["boxes"], STATES):
        patch.set_facecolor(STATE_COLORS[s])
        patch.set_alpha(0.7)
    format_axes(ax, title=feat_titles[feat], xlabel="State",
                ylabel=feat_titles[feat], unit_y=feat_units[feat], legend=False)
fig.suptitle("HRV Features Across Affective States (15 subjects, 60-s windows)",
             fontsize=12, y=1.02)
fig.tight_layout()
save(fig, "fig05_hrv_boxplot.png")


# ════════════════════════════════════════════
# 从 loso_dual.json 读结果
# ════════════════════════════════════════════
with open(ROOT / "results" / "json" / "loso_dual.json", encoding="utf-8") as f:
    RES = json.load(f)


# ════════════════════════════════════════════
# fig06 混淆矩阵
# ════════════════════════════════════════════
print("[fig06] confusion matrix")
cm = np.array(RES["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[s.capitalize() for s in STATES],
            yticklabels=[s.capitalize() for s in STATES], ax=ax,
            cbar_kws={"label": "Count"})
ax.set_xlabel("Predicted State", fontsize=10)
ax.set_ylabel("True State", fontsize=10)
ax.set_title("LOSO Aggregated Confusion Matrix (15 folds)", fontsize=11, pad=8)
fig.tight_layout()
save(fig, "fig06_confusion_matrix.png")


# ════════════════════════════════════════════
# fig07 三分类性能雷达图
# ════════════════════════════════════════════
print("[fig07] radar chart")
# 用聚合混淆矩阵算每类 P/R/F1
col_sums = cm.sum(axis=0)
row_sums = cm.sum(axis=1)
prec = np.diag(cm) / np.clip(col_sums, 1, None)
rec = np.diag(cm) / np.clip(row_sums, 1, None)
f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
metrics = {"Precision": prec, "Recall": rec, "F1": f1}
angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(5.5, 5), subplot_kw=dict(polar=True))
for i, s in enumerate(STATES):
    vals = [prec[i], rec[i], f1[i]]
    vals += vals[:1]
    ax.plot(angles, vals, color=STATE_COLORS[s], lw=2, label=s.capitalize())
    ax.fill(angles, vals, color=STATE_COLORS[s], alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Precision", "Recall", "F1"])
ax.set_ylim(0, 1)
ax.set_title("Per-Class Performance Radar", fontsize=11, pad=15)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
fig.tight_layout()
save(fig, "fig07_radar.png")


# ════════════════════════════════════════════
# fig08 代表性单折训练曲线
# ════════════════════════════════════════════
print("[fig08] training curves")
tc = RES["training_curves"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(tc["epochs"], tc["train_loss"], color="#4A7ABC", label="Train")
ax1.plot(tc["epochs"], tc["val_loss"], color="#D9534F", label="Validation")
format_axes(ax1, title=f"Loss (representative fold {tc['fold_id']})",
            xlabel="Epoch", ylabel="Loss", legend=True)
ax2.plot(tc["epochs"], tc["train_acc"], color="#4A7ABC", label="Train")
ax2.plot(tc["epochs"], tc["val_acc"], color="#D9534F", label="Validation")
format_axes(ax2, title=f"Accuracy (representative fold {tc['fold_id']})",
            xlabel="Epoch", ylabel="Accuracy", legend=True)
ax2.set_ylim(0, 1)
fig.tight_layout()
save(fig, "fig08_training_curves.png")


# ════════════════════════════════════════════
# fig09 模型架构示意图
# ════════════════════════════════════════════
print("[fig09] architecture diagram")
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, text, color="#E8F0FE", ec="#4A7ABC", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                facecolor=color, edgecolor=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=12, color="#555", lw=1.3))

# PPG 分支（上）
box(0.3, 4.4, 1.4, 0.8, "PPG Input\n(640×1)", color="#FDECEA", ec="#C0504D")
box(2.2, 4.4, 1.5, 0.8, "Conv1D 32\nk=5 + BN\nMaxPool", color="#FCE4EC")
box(4.1, 4.4, 1.5, 0.8, "Conv1D 48\nk=3 + BN\nMaxPool", color="#FCE4EC")
box(6.0, 4.4, 1.5, 0.8, "Conv1D 64\nk=3 + BN\nMaxPool", color="#FCE4EC")
box(7.9, 4.4, 1.2, 0.8, "GAP\n(64)", color="#E3F2FD")
# EDA 分支（下）
box(0.3, 1.0, 1.4, 0.8, "EDA Input\n(40×1)", color="#E8F5E9", ec="#4C9F70")
box(2.2, 1.0, 1.8, 0.8, "Conv1D 16\nk=3 + BN MaxPool", color="#E8F5E9")
box(4.5, 1.0, 1.8, 0.8, "Conv1D 24\nk=3 + BN MaxPool", color="#E8F5E9")
box(6.7, 1.0, 1.2, 0.8, "GAP\n(24)", color="#E3F2FD")
# 融合
box(9.6, 2.55, 1.3, 0.9, "Concat\n(88)", color="#FFF3E0", ec="#E65100")
box(11.3, 2.55, 1.3, 0.9, "FC 64\n+Dropout", color="#F3E5F5")
box(13.0, 2.7, 0.8, 0.6, "FC 3\nSoftmax", color="#FFEBEE", ec="#C0504D", fs=8)
# 箭头
for x in [1.7, 3.6, 5.5, 7.4]:
    arrow(x, 4.8, x + 0.5, 4.8)
arrow(9.1, 4.8, 9.6, 3.45); arrow(7.9, 1.4, 9.6, 2.55)
arrow(10.9, 3.0, 11.3, 3.0); arrow(12.6, 3.0, 13.0, 3.0)
ax.text(4.5, 5.5, "PPG Branch (deep)", ha="center", fontsize=10, color="#C0504D", weight="bold")
ax.text(4.0, 0.4, "EDA Branch (shallow)", ha="center", fontsize=10, color="#4C9F70", weight="bold")
ax.text(7, 3.0, "Feature-level Fusion", ha="center", fontsize=11, weight="bold", color="#E65100")
ax.text(7, -0.3, "Total parameters: 21,627 (≈21.6K)", ha="center", fontsize=10, style="italic")
fig.tight_layout()
save(fig, "fig09_architecture.png")


# ════════════════════════════════════════════
# fig10 H2 消融对比（附图）
# ════════════════════════════════════════════
print("[fig10] ablation comparison")
ab = {}
for v in ["dual", "ppg", "eda", "late"]:
    with open(ROOT / "results" / "json" / f"loso_{v}.json", encoding="utf-8") as f:
        ab[v] = json.load(f)
order = ["ppg", "eda", "late", "dual"]
x = np.arange(len(order)); width = 0.35
acc_mean = [ab[v]["accuracy_mean"] for v in order]
acc_std = [ab[v]["accuracy_std"] for v in order]
f1_mean = [ab[v]["macro_f1_mean"] for v in order]
f1_std = [ab[v]["macro_f1_std"] for v in order]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - width / 2, acc_mean, width, yerr=acc_std, label="Accuracy",
       color="#4A7ABC", capsize=4, alpha=0.85)
ax.bar(x + width / 2, f1_mean, width, yerr=f1_std, label="macro F1",
       color="#D9534F", capsize=4, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([v.upper() for v in order])
ax.set_ylim(0, 1)
# 标显著性
ax.annotate("** p=0.001\nvs PPG", xy=(3 - width / 2, acc_mean[3] + 0.02),
            ha="center", fontsize=8, color="green")
format_axes(ax, title="H2 Ablation: Feature-level Fusion vs Alternatives (LOSO)",
            xlabel="Model Variant", ylabel="Score", legend=True)
fig.tight_layout()
save(fig, "fig10_ablation.png")

print(f"\nALL FIGURES DONE -> {FIG_DIR}")
print("files:", sorted(p.name for p in FIG_DIR.glob("fig*.png")))
