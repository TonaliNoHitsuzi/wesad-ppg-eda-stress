/**
 * 占位数据（Mock Data）
 * 
 * 当 public/data/ 目录下没有对应JSON文件时，系统使用这些占位数据。
 * Python处理完成后，将真实数据导出为JSON放入 public/data/ 即可替换。
 */

import type {
  SubjectInfo,
  SignalData,
  SpectrumData,
  SpectrogramData,
  PredictionResult,
  ConfusionMatrix,
  TrainingCurves,
  HRVFeatureData,
  ModelMetrics,
} from "@/types/dashboard";

/* ── 文件1: subject_info.json ── */
export const mockSubjectInfo: SubjectInfo = {
  subject_id: "S2",
  signal_quality: "excellent",
  current_state: "stress",
  confidence: 0.873,
  recording_duration: "60 min",
  ppg_sampling_rate: 64,
  eda_sampling_rate: 4,
  states: ["baseline", "stress", "amusement"],
  state_counts: { baseline: 120, stress: 240, amusement: 120 },
};

/* ── 文件2: signal_ppg.json ── */
function genPPG(): SignalData {
  const fs = 64;
  const dur = 10;
  const n = fs * dur;
  const data: number[] = [];
  const timestamps: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    timestamps.push(t);
    // 模拟PPG: 心跳频率约1.2Hz (72bpm) + 谐波 + 噪声
    const pulse = Math.sin(2 * Math.PI * 1.2 * t) * 0.5
                + Math.sin(2 * Math.PI * 2.4 * t) * 0.15
                + Math.sin(2 * Math.PI * 3.6 * t) * 0.05;
    const baseline = Math.sin(2 * Math.PI * 0.05 * t) * 0.1; // 慢漂移
    const noise = (Math.random() - 0.5) * 0.03;
    data.push(pulse + baseline + noise);
  }
  return { sampling_rate: fs, duration_seconds: dur, data, timestamps };
}
export const mockPPGSignal = genPPG();

/* ── 文件3: signal_eda.json ── */
function genEDA(): SignalData {
  const fs = 4;
  const dur = 10;
  const n = fs * dur;
  const data: number[] = [];
  const timestamps: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    timestamps.push(t);
    // 模拟EDA: 基础水平 + 慢变化 + 小波动
    const tonic = 2.0 + Math.sin(2 * Math.PI * 0.02 * t) * 0.5;
    const phasic = (t > 3 && t < 6) ? Math.exp(-((t - 4.5) ** 2)) * 0.8 : 0;
    const noise = (Math.random() - 0.5) * 0.05;
    data.push(tonic + phasic + noise);
  }
  return { sampling_rate: fs, duration_seconds: dur, data, timestamps };
}
export const mockEDASignal = genEDA();

/* ── 文件4: spectrum_fft.json ── */
export const mockSpectrum: SpectrumData = {
  frequencies: Array.from({ length: 80 }, (_, i) => 0.5 + i * 0.1),
  power_db: Array.from({ length: 80 }, (_, i) => {
    const f = 0.5 + i * 0.1;
    // 峰值在1.2Hz(心率)和LF/HF频带
    let p = -40 + Math.random() * 3;
    p += 15 * Math.exp(-(((f - 1.2) ** 2)) / 0.05); // 心率峰
    p += 8 * Math.exp(-(((f - 0.08) ** 2)) / 0.003); // LF
    p += 6 * Math.exp(-(((f - 0.25) ** 2)) / 0.008); // HF
    return p;
  }),
  bands: {
    vlf: { range: [0.003, 0.04], power: 12.5 },
    lf:  { range: [0.04, 0.15], power: 45.3 },
    hf:  { range: [0.15, 0.4], power: 38.7 },
  },
  lf_hf_ratio: 1.17,
};

/* ── 文件5: spectrogram_stft.json ── */
function genSpectrogram(): SpectrogramData {
  const times = Array.from({ length: 40 }, (_, i) => i * 0.25);
  const frequencies = Array.from({ length: 32 }, (_, i) => i * 0.25);
  const magnitude: number[][] = [];
  for (let i = 0; i < frequencies.length; i++) {
    const row: number[] = [];
    for (let j = 0; j < times.length; j++) {
      const f = frequencies[i];
      const t = times[j];
      // 时频能量集中在1.2Hz附近，随时间有微小漂移
      const hrFreq = 1.2 + 0.1 * Math.sin(2 * Math.PI * t / 10);
      const val = 5 * Math.exp(-(((f - hrFreq) ** 2)) / 0.08) + Math.random() * 0.5;
      row.push(Math.max(0, val));
    }
    magnitude.push(row);
  }
  return { times, frequencies, magnitude };
}
export const mockSpectrogram = genSpectrogram();

/* ── 文件6: model_prediction.json ── */
export const mockPrediction: PredictionResult = {
  true_label: "stress",
  predicted_label: "stress",
  probabilities: {
    baseline: 0.08,
    stress: 0.873,
    amusement: 0.047,
  },
};

/* ── 文件7: confusion_matrix.json ── */
export const mockConfusionMatrix: ConfusionMatrix = {
  labels: ["baseline", "stress", "amusement"],
  matrix: [
    [108, 8, 4],
    [12, 218, 10],
    [5, 7, 108],
  ],
};

/* ── 文件8: training_curves.json ── */
export const mockTrainingCurves: TrainingCurves = {
  epochs: Array.from({ length: 100 }, (_, i) => i + 1),
  train_loss: Array.from({ length: 100 }, (_, i) =>
    1.2 * Math.exp(-i / 25) + 0.08 + Math.random() * 0.03
  ),
  val_loss: Array.from({ length: 100 }, (_, i) =>
    1.3 * Math.exp(-i / 30) + 0.15 + Math.random() * 0.04
  ),
  train_acc: Array.from({ length: 100 }, (_, i) =>
    Math.min(0.92, 0.45 + 0.47 * (1 - Math.exp(-i / 20)) + Math.random() * 0.015)
  ),
  val_acc: Array.from({ length: 100 }, (_, i) =>
    Math.min(0.88, 0.42 + 0.46 * (1 - Math.exp(-i / 22)) + Math.random() * 0.02)
  ),
};

/* ── 文件9: hrv_features.json ── */
export const mockHRVFeatures: HRVFeatureData = {
  features: ["SDNN", "RMSSD", "LF/HF", "HR_mean"],
  groups: {
    baseline: {
      SDNN:   { median: 65, q1: 52, q3: 80, min: 35, max: 115 },
      RMSSD:  { median: 45, q1: 35, q3: 58, min: 22, max: 82 },
      "LF/HF": { median: 1.8, q1: 1.3, q3: 2.5, min: 0.8, max: 4.2 },
      HR_mean: { median: 68, q1: 62, q3: 74, min: 55, max: 88 },
    },
    stress: {
      SDNN:   { median: 38, q1: 28, q3: 48, min: 18, max: 72 },
      RMSSD:  { median: 22, q1: 16, q3: 30, min: 10, max: 48 },
      "LF/HF": { median: 3.5, q1: 2.4, q3: 5.1, min: 1.5, max: 8.5 },
      HR_mean: { median: 88, q1: 78, q3: 98, min: 65, max: 115 },
    },
    amusement: {
      SDNN:   { median: 58, q1: 46, q3: 72, min: 30, max: 105 },
      RMSSD:  { median: 38, q1: 28, q3: 50, min: 18, max: 75 },
      "LF/HF": { median: 2.0, q1: 1.4, q3: 2.8, min: 0.9, max: 4.5 },
      HR_mean: { median: 75, q1: 68, q3: 82, min: 58, max: 95 },
    },
  },
};

/* ── 文件10: model_metrics.json ── */
export const mockModelMetrics: ModelMetrics = {
  accuracy: 0.876,
  macro_f1: 0.868,
  precision: { baseline: 0.89, stress: 0.92, amusement: 0.84 },
  recall:    { baseline: 0.86, stress: 0.91, amusement: 0.85 },
  f1:        { baseline: 0.87, stress: 0.91, amusement: 0.84 },
  params_count: 45230,
  model_size_mb: 0.18,
};
