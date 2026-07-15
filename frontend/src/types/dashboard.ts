/**
 * ============================================
 * 数据接口定义文件
 * ============================================
 * 
 * 本文件定义了前端展示系统所需的全部数据类型。
 * Python数据处理完成后，请将结果导出为以下JSON格式，
 * 放置在 public/data/ 目录下。
 * 
 * 需要生成的数据文件：
 * 1. subject_info.json      -- 被试基本信息与当前状态
 * 2. signal_ppg.json        -- PPG信号波形数据（带时间戳）
 * 3. signal_eda.json        -- EDA信号波形数据（带时间戳）
 * 4. spectrum_fft.json      -- FFT频谱分析结果
 * 5. spectrogram_stft.json  -- STFT时频分析结果
 * 6. model_prediction.json  -- 模型预测结果
 * 7. confusion_matrix.json  -- 混淆矩阵数据
 * 8. training_curves.json   -- 模型训练曲线
 * 9. hrv_features.json      -- HRV特征统计（箱线图数据）
 * 10. model_metrics.json    -- 模型性能指标汇总
 * 
 * 每个接口的详细JSON Schema见下方类型定义。
 */

/** 文件1: subject_info.json
 * {
 *   "subject_id": "S2",
 *   "signal_quality": "excellent",
 *   "current_state": "stress",
 *   "confidence": 0.873,
 *   "recording_duration": "60 min",
 *   "ppg_sampling_rate": 64,
 *   "eda_sampling_rate": 4,
 *   "states": ["baseline", "stress", "amusement"],
 *   "state_counts": { "baseline": 120, "stress": 240, "amusement": 120 }
 * }
 */
export interface SubjectInfo {
  subject_id: string;
  signal_quality: "excellent" | "good" | "fair" | "poor";
  current_state: "baseline" | "stress" | "amusement";
  confidence: number; // 0-1
  recording_duration: string;
  ppg_sampling_rate: number;
  eda_sampling_rate: number;
  states: string[];
  state_counts: Record<string, number>;
}

/** 文件2: signal_ppg.json
 * {
 *   "sampling_rate": 64,
 *   "duration_seconds": 10,
 *   "data": [0.12, 0.15, 0.18, ...],  // 长度 = 640 (64Hz * 10s)
 *   "timestamps": [0, 0.0156, 0.03125, ...]  // 单位：秒，长度与data一致
 * }
 */
export interface SignalData {
  sampling_rate: number;
  duration_seconds: number;
  data: number[];
  timestamps: number[];
}

/** 文件4: spectrum_fft.json
 * {
 *   "frequencies": [0.5, 1.0, 1.5, ...],  // Hz
 *   "power_db": [-20, -15, -10, ...],      // dB
 *   "bands": {
 *     "vlf": { "range": [0.003, 0.04], "power": 12.5 },
 *     "lf":  { "range": [0.04, 0.15],  "power": 45.3 },
 *     "hf":  { "range": [0.15, 0.4],   "power": 38.7 }
 *   },
 *   "lf_hf_ratio": 1.17
 * }
 */
export interface SpectrumData {
  frequencies: number[];
  power_db: number[];
  bands: {
    vlf: { range: [number, number]; power: number };
    lf:  { range: [number, number]; power: number };
    hf:  { range: [number, number]; power: number };
  };
  lf_hf_ratio: number;
}

/** 文件5: spectrogram_stft.json
 * {
 *   "times": [0, 0.5, 1.0, ...],       // 时间轴（秒）
 *   "frequencies": [0, 1, 2, ...],     // 频率轴（Hz）
 *   "magnitude": [[0.1, 0.2, ...], ...] // 2D数组：times x frequencies
 * }
 */
export interface SpectrogramData {
  times: number[];
  frequencies: number[];
  magnitude: number[][];
}

/** 文件6: model_prediction.json
 * {
 *   "true_label": "stress",
 *   "predicted_label": "stress",
 *   "probabilities": {
 *     "baseline": 0.08,
 *     "stress": 0.873,
 *     "amusement": 0.047
 *   }
 * }
 */
export interface PredictionResult {
  true_label: string;
  predicted_label: string;
  probabilities: Record<string, number>;
}

/** 文件7: confusion_matrix.json
 * {
 *   "labels": ["baseline", "stress", "amusement"],
 *   "matrix": [
 *     [108, 8, 4],
 *     [12, 218, 10],
 *     [5, 7, 108]
 *   ]
 * }
 */
export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][]; // 3x3
}

/** 文件8: training_curves.json
 * {
 *   "epochs": [1, 2, 3, ..., 100],
 *   "train_loss": [1.2, 0.9, 0.7, ..., 0.15],
 *   "val_loss": [1.3, 1.0, 0.8, ..., 0.25],
 *   "train_acc": [0.45, 0.55, 0.65, ..., 0.92],
 *   "val_acc": [0.42, 0.52, 0.62, ..., 0.88]
 * }
 */
export interface TrainingCurves {
  epochs: number[];
  train_loss: number[];
  val_loss: number[];
  train_acc: number[];
  val_acc: number[];
}

/** 文件9: hrv_features.json (箱线图)
 * {
 *   "features": ["SDNN", "RMSSD", "LF/HF", "HR_mean"],
 *   "groups": {
 *     "baseline": {
 *       "SDNN": { "median": 65, "q1": 50, "q3": 80, "min": 30, "max": 120 },
 *       ...
 *     },
 *     "stress": { ... },
 *     "amusement": { ... }
 *   }
 * }
 */
export interface HRVFeatureData {
  features: string[];
  groups: Record<string, Record<string, {
    median: number;
    q1: number;
    q3: number;
    min: number;
    max: number;
  }>>;
}

/** 文件10: model_metrics.json
 * {
 *   "accuracy": 0.876,
 *   "macro_f1": 0.868,
 *   "precision": { "baseline": 0.89, "stress": 0.92, "amusement": 0.84 },
 *   "recall":    { "baseline": 0.86, "stress": 0.91, "amusement": 0.85 },
 *   "f1":        { "baseline": 0.87, "stress": 0.91, "amusement": 0.84 },
 *   "params_count": 45230,
 *   "model_size_mb": 0.18
 * }
 */
export interface ModelMetrics {
  accuracy: number;
  macro_f1: number;
  precision: Record<string, number>;
  recall: Record<string, number>;
  f1: Record<string, number>;
  params_count: number;
  model_size_mb: number;
}

/** 文件11: ablation_comparison.json（H2 消融实验，新增）
 * {
 *   "variants": [
 *     { "name": "Dual", "accuracy": 0.749, "macro_f1": 0.71, "params": 21627 }, ...
 *   ],
 *   "wilcoxon": [
 *     { "comparison": "Dual vs PPG", "p_value": 0.001, "significant": true }, ...
 *   ]
 * }
 */
export interface AblationVariant {
  name: "Dual" | "Late" | "EDA" | "PPG" | string;
  accuracy: number;
  macro_f1: number;
  params: number;
}

export interface WilcoxonTest {
  comparison: string;
  p_value: number;
  significant: boolean;
}

export interface AblationComparison {
  variants: AblationVariant[];
  wilcoxon: WilcoxonTest[];
}

/** 被试切换：frontend/public/data/subjects/S{X}.json（每被试一个打包文件） */
export interface StateView {
  signal_ppg: SignalData;
  signal_eda: SignalData;
  spectrum: SpectrumData;
  spectrogram: SpectrogramData;
}

export interface SubjectBundle {
  subject_id: string;
  recording_duration_min: number;
  states: Record<string, StateView>;  // baseline / stress / amusement
  recall: Record<string, number>;
  prediction: { predicted_label: string; probabilities: Record<string, number> };
  confidence: number;
  loso: { accuracy: number; macro_f1: number };
}
