# 前端 JSON 数据目录

由 `src/export/export_json.py` 与 `scripts/export_subjects.py` 生成，供 React Dashboard 读取。
缺失时前端自动回退到 `src/lib/mockData.ts` 占位数据。

## 全局聚合 JSON（研究级，不随被试切换）

| 文件 | 内容 | 类型 |
|------|------|------|
| `subject_info.json` | 整体信息（采样率、聚合 state_counts） | `SubjectInfo` |
| `signal_ppg.json` / `signal_eda.json` | S2 代表窗波形（初始展示用） | `SignalData` |
| `spectrum_fft.json` | FFT 功率谱 + VLF/LF/HF 频带 | `SpectrumData` |
| `spectrogram_stft.json` | STFT 时频热力图 | `SpectrogramData` |
| `model_prediction.json` | 默认预测概率 | `PredictionResult` |
| `confusion_matrix.json` | LOSO 聚合混淆矩阵 | `ConfusionMatrix` |
| `training_curves.json` | 代表性折训练曲线 | `TrainingCurves` |
| `hrv_features.json` | HRV 四特征箱线图（15 被试，ECG） | `HRVFeatureData` |
| `model_metrics.json` | 整体性能（acc/macro_F1/逐类 P/R/F1/参数量） | `ModelMetrics` |
| `ablation_comparison.json` | H2 四变体消融 + Wilcoxon | `AblationComparison` |

## 被试 JSON（`subjects/S{X}.json`，被试级，切换时刷新）

15 个文件（S2–S17，无 S12），每被试含三状态（baseline/stress/amusement）的波形/FFT/STFT + 逐类检出率 + 个人 LOSO 指标。类型见 `SubjectBundle`。

接口契约详见 `src/types/dashboard.ts`。
