# 前端 JSON 数据目录

`src/export/export_json.py` 运行后，会在此目录生成以下 10 个 JSON 文件，
供 React Dashboard 读取。当目录为空时，前端自动回退到 `src/lib/mockData.ts` 中的占位数据。

| 文件 | 内容 | 对应类型 (`src/types/dashboard.ts`) |
|------|------|-------------------------------------|
| `subject_info.json` | 被试基本信息与当前状态 | `SubjectInfo` |
| `signal_ppg.json` | PPG 波形（10s, 64Hz, 640点） | `SignalData` |
| `signal_eda.json` | EDA 波形（10s, 4Hz, 40点） | `SignalData` |
| `spectrum_fft.json` | FFT 功率谱 + VLF/LF/HF 频带 | `SpectrumData` |
| `spectrogram_stft.json` | STFT 时频热力图 | `SpectrogramData` |
| `model_prediction.json` | 单样本三分类概率 | `PredictionResult` |
| `confusion_matrix.json` | LOSO 聚合混淆矩阵 | `ConfusionMatrix` |
| `training_curves.json` | 训练 loss / acc 曲线 | `TrainingCurves` |
| `hrv_features.json` | HRV 特征箱线图（3 组 × 4 特征） | `HRVFeatureData` |
| `model_metrics.json` | 模型整体性能指标 | `ModelMetrics` |
