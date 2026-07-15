/**
 * ============================================
 * 可穿戴情感状态监测系统 - 前端框架雏形
 * ============================================
 * 
 * 现代医学Dashboard明亮风格
 * 
 * 数据加载说明：
 * 1. 将Python处理后的JSON数据文件放入 public/data/ 目录
 * 2. 数据文件命名和格式需严格遵循 src/types/dashboard.ts 中的定义
 * 3. 如果数据文件不存在，将自动使用占位数据（mock data）
 * 
 * 需要的数据文件清单（共10个）：
 *   - public/data/subject_info.json
 *   - public/data/signal_ppg.json
 *   - public/data/signal_eda.json
 *   - public/data/spectrum_fft.json
 *   - public/data/spectrogram_stft.json
 *   - public/data/model_prediction.json
 *   - public/data/confusion_matrix.json
 *   - public/data/training_curves.json
 *   - public/data/hrv_features.json
 *   - public/data/model_metrics.json
 */

import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Activity,
  Heart,
  Zap,
  Brain,
  TrendingUp,
  Timer,
  User,
  BarChart3,
} from "lucide-react";
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
  AblationComparison,
  SubjectBundle,
} from "@/types/dashboard";
import {
  mockSubjectInfo,
  mockPPGSignal,
  mockEDASignal,
  mockSpectrum,
  mockSpectrogram,
  mockPrediction,
  mockConfusionMatrix,
  mockTrainingCurves,
  mockHRVFeatures,
  mockModelMetrics,
  mockAblation,
} from "@/lib/mockData";

/* ────────── 通用数据加载工具 ────────── */

async function loadJson<T>(path: string, fallback: T): Promise<T> {
  try {
    // BASE_URL 前缀适配 GitHub Pages 子路径部署（vite base: './'）
    const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
    if (!res.ok) throw new Error("Not found");
    return (await res.json()) as T;
  } catch {
    console.warn(`[Data] Using mock data for ${path}`);
    return fallback;
  }
}

/* ────────── 图表配置工厂函数 ────────── */

/** PPG/EDA 波形图配置 */
function buildWaveOption(
  signal: SignalData,
  title: string,
  color: string,
  yLabel: string
) {
  return {
    animation: false,
    grid: { top: 30, right: 20, bottom: 30, left: 50 },
    title: { text: title, left: "center", textStyle: { fontSize: 13, color } },
    xAxis: {
      type: "category",
      data: signal.timestamps.map((t) => t.toFixed(2)),
      name: "时间 (s)",
      nameLocation: "middle",
      nameGap: 22,
      axisLabel: { fontSize: 10, interval: 63 },
    },
    yAxis: {
      type: "value",
      name: yLabel,
      nameTextStyle: { fontSize: 10 },
      axisLabel: { fontSize: 10 },
    },
    series: [{
      type: "line",
      data: signal.data,
      smooth: false,
      lineStyle: { color, width: 1.2 },
      showSymbol: false,
      areaStyle: { color, opacity: 0.08 },
    }],
    tooltip: { trigger: "axis" },
  };
}

/** FFT频谱图配置 */
function buildSpectrumOption(spec: SpectrumData) {
  const bandColors: Record<string, string> = {
    vlf: "#94a3b8",
    lf: "#3b82f6",
    hf: "#10b981",
  };
  const markAreas = Object.entries(spec.bands).map(([key, band]) => ({
    name: key.toUpperCase(),
    xAxis: band.range[0],
    yAxis: band.range[1],
    itemStyle: { color: bandColors[key], opacity: 0.12 },
    label: {
      show: true,
      position: "insideTop",
      formatter: `${key.toUpperCase()}\n${band.power.toFixed(1)}`,
      fontSize: 9,
      color: bandColors[key],
    },
  }));

  return {
    animation: false,
    grid: { top: 30, right: 20, bottom: 35, left: 50 },
    title: { text: "FFT功率谱密度", left: "center", textStyle: { fontSize: 13 } },
    xAxis: {
      type: "category",
      data: spec.frequencies.map((f) => f.toFixed(1)),
      name: "频率 (Hz)",
      nameLocation: "middle",
      nameGap: 22,
      axisLabel: { fontSize: 10, interval: 9 },
    },
    yAxis: {
      type: "value",
      name: "功率 (dB)",
      nameTextStyle: { fontSize: 10 },
      axisLabel: { fontSize: 10 },
    },
    series: [{
      type: "line",
      data: spec.power_db,
      lineStyle: { color: "#6366f1", width: 1.5 },
      areaStyle: { color: "#6366f1", opacity: 0.15 },
      showSymbol: false,
      markArea: { data: [markAreas] },
    }],
    tooltip: { trigger: "axis" },
  };
}

/** STFT时频热力图配置 */
function buildSpectrogramOption(spec: SpectrogramData) {
  return {
    animation: false,
    grid: { top: 30, right: 60, bottom: 35, left: 50 },
    title: { text: "STFT时频热力图", left: "center", textStyle: { fontSize: 13 } },
    xAxis: {
      type: "category",
      data: spec.times.map((t) => t.toFixed(1)),
      name: "时间 (s)",
      nameLocation: "middle",
      nameGap: 22,
      axisLabel: { fontSize: 10, interval: Math.floor(spec.times.length / 8) },
    },
    yAxis: {
      type: "category",
      data: spec.frequencies.map((f) => f.toFixed(1)),
      name: "频率 (Hz)",
      nameTextStyle: { fontSize: 10 },
      axisLabel: { fontSize: 10, interval: Math.floor(spec.frequencies.length / 6) },
    },
    visualMap: {
      min: 0,
      max: Math.max(...spec.magnitude.flat()),
      calculable: true,
      orient: "vertical",
      right: 5,
      top: "center",
      inRange: { color: ["#f0f9ff", "#7dd3fc", "#0284c7", "#0c4a6e"] },
      textStyle: { fontSize: 9 },
    },
    series: [{
      type: "heatmap",
      data: spec.magnitude.flatMap((row, i) =>
        row.map((val, j) => [j, spec.frequencies.length - 1 - i, val])
      ),
      emphasis: { itemStyle: { borderColor: "#333", borderWidth: 1 } },
    }],
  };
}

/** 混淆矩阵热力图配置 */
function buildConfusionOption(cm: ConfusionMatrix) {
  const maxVal = Math.max(...cm.matrix.flat());
  const data = cm.matrix.flatMap((row, i) =>
    row.map((val, j) => ({
      value: [j, cm.labels.length - 1 - i, val],
      label: { show: true, fontSize: 14, fontWeight: "bold" },
    }))
  );
  return {
    animation: false,
    grid: { top: 30, right: 60, bottom: 35, left: 80 },
    title: { text: "混淆矩阵 (LOSO交叉验证)", left: "center", textStyle: { fontSize: 13 } },
    xAxis: {
      type: "category",
      data: cm.labels,
      name: "预测标签",
      nameLocation: "middle",
      nameGap: 25,
      axisLabel: { fontSize: 11 },
    },
    yAxis: {
      type: "category",
      data: [...cm.labels].reverse(),
      name: "真实标签",
      nameTextStyle: { fontSize: 11 },
      axisLabel: { fontSize: 11 },
    },
    visualMap: {
      min: 0,
      max: maxVal,
      calculable: false,
      orient: "vertical",
      right: 5,
      top: "center",
      inRange: { color: ["#f1f5f9", "#bfdbfe", "#3b82f6", "#1e3a5f"] },
      textStyle: { fontSize: 9 },
    },
    series: [{ type: "heatmap", data, label: { show: true, fontSize: 12 } }],
  };
}

/** 训练曲线配置 */
function buildTrainingOption(tc: TrainingCurves) {
  return {
    animation: false,
    grid: { top: 30, right: 60, bottom: 30, left: 50 },
    title: { text: "模型训练曲线", left: "center", textStyle: { fontSize: 13 } },
    legend: { data: ["训练损失", "验证损失", "训练准确率", "验证准确率"], top: 0, textStyle: { fontSize: 10 } },
    xAxis: { type: "category", data: tc.epochs, name: "Epoch", axisLabel: { fontSize: 10, interval: 19 } },
    yAxis: [
      { type: "value", name: "损失", nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10 } },
      { type: "value", name: "准确率", min: 0, max: 1, axisLabel: { fontSize: 10, formatter: "{value}" } },
    ],
    series: [
      { name: "训练损失", type: "line", data: tc.train_loss, lineStyle: { color: "#94a3b8", width: 1 }, showSymbol: false },
      { name: "验证损失", type: "line", data: tc.val_loss, lineStyle: { color: "#64748b", width: 1 }, showSymbol: false },
      { name: "训练准确率", type: "line", data: tc.train_acc, yAxisIndex: 1, lineStyle: { color: "#3b82f6", width: 1.5 }, showSymbol: false },
      { name: "验证准确率", type: "line", data: tc.val_acc, yAxisIndex: 1, lineStyle: { color: "#10b981", width: 1.5 }, showSymbol: false },
    ],
    tooltip: { trigger: "axis" },
  };
}

/** HRV箱线图配置 */
function buildHRVBoxOption(hrv: HRVFeatureData) {
  const categories = hrv.features;
  const groups = Object.keys(hrv.groups);
  const colors = ["#3b82f6", "#ef4444", "#10b981"];

  const series = groups.map((group, gi) => ({
    name: group,
    type: "boxplot",
    data: categories.map((feat) => {
      const d = hrv.groups[group][feat];
      return [d.min, d.q1, d.median, d.q3, d.max];
    }),
    itemStyle: { color: colors[gi], borderColor: colors[gi] },
  }));

  return {
    animation: false,
    grid: { top: 35, right: 20, bottom: 30, left: 60 },
    title: { text: "HRV特征分布（按情感状态分组）", left: "center", textStyle: { fontSize: 13 } },
    legend: { data: groups, top: 0, textStyle: { fontSize: 10 } },
    xAxis: { type: "category", data: categories, axisLabel: { fontSize: 11 } },
    yAxis: { type: "value", name: "数值", nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10 } },
    series,
    tooltip: { trigger: "item" },
  };
}

/** 概率柱状图 */
function buildProbOption(pred: PredictionResult) {
  const entries = Object.entries(pred.probabilities);
  const colors: Record<string, string> = {
    baseline: "#3b82f6",
    stress: "#ef4444",
    amusement: "#10b981",
  };
  return {
    animation: false,
    grid: { top: 25, right: 20, bottom: 25, left: 50 },
    xAxis: { type: "category", data: entries.map(([k]) => k), axisLabel: { fontSize: 11 } },
    yAxis: { type: "value", max: 1, axisLabel: { fontSize: 10, formatter: "{value}" } },
    series: [{
      type: "bar",
      data: entries.map(([k, v]) => ({
        value: v,
        itemStyle: { color: colors[k] || "#6366f1" },
        label: { show: true, position: "top", formatter: "{c}", fontSize: 11, fontWeight: "bold" },
      })),
      barWidth: "40%",
    }],
  };
}

/** H2 消融对比柱状图（准确率 + macro F1，含显著性标注） */
function buildAblationOption(ab: AblationComparison) {
  const order = ["PPG", "EDA", "Late", "Dual"] as const;
  const vmap = Object.fromEntries(ab.variants.map((v) => [v.name, v]));
  const rows = order.map((n) => vmap[n]).filter(Boolean);
  return {
    animation: false,
    grid: { top: 45, right: 20, bottom: 30, left: 50 },
    legend: { data: ["Accuracy", "macro F1"], top: 4, textStyle: { fontSize: 11 } },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: {
      type: "category",
      data: rows.map((v) => `${v.name}\n(${v.params.toLocaleString()} params)`),
      axisLabel: { fontSize: 10 },
    },
    yAxis: { type: "value", min: 0.4, max: 0.85, axisLabel: { fontSize: 10 } },
    series: [
      {
        name: "Accuracy",
        type: "bar",
        data: rows.map((v) => v.accuracy),
        itemStyle: { color: "#6366f1" },
        barGap: "10%",
      },
      {
        name: "macro F1",
        type: "bar",
        data: rows.map((v) => v.macro_f1),
        itemStyle: {
          color: (p: { dataIndex: number }) =>
            rows[p.dataIndex]?.name === "Dual" ? "#10b981" : "#f59e0b",
        },
      },
      {
        // 显著性标注（仅 Dual 柱顶部）
        type: "custom",
        renderItem: (params: { dataIndex: number }, api: { value: (i: number) => number; coord: (v: number[]) => number[] }) => {
          const dualIdx = rows.findIndex((v) => v.name === "Dual");
          if (params.dataIndex !== dualIdx) return null;
          const point = api.coord([dualIdx, api.value(0)]);
          return {
            type: "text",
            style: { text: "★ p=0.001\nvs PPG", x: point[0] - 26, y: point[1] - 34,
              fill: "#16a34a", fontSize: 10, fontWeight: "bold", textAlign: "center" },
          };
        },
        data: rows.map((v) => (v.name === "Dual" ? v.accuracy : 0)),
      },
    ],
  };
}

/* ────────── 主组件 ────────── */

export default function App() {
  const [subjectInfo, setSubjectInfo] = useState<SubjectInfo>(mockSubjectInfo);
  const [ppgSignal, setPPGSignal] = useState<SignalData>(mockPPGSignal);
  const [edaSignal, setEDASignal] = useState<SignalData>(mockEDASignal);
  const [spectrum, setSpectrum] = useState<SpectrumData>(mockSpectrum);
  const [spectrogram, setSpectrogram] = useState<SpectrogramData>(mockSpectrogram);
  const [prediction, setPrediction] = useState<PredictionResult>(mockPrediction);
  const [confMatrix, setConfMatrix] = useState<ConfusionMatrix>(mockConfusionMatrix);
  const [trainCurves, setTrainCurves] = useState<TrainingCurves>(mockTrainingCurves);
  const [hrvFeatures, setHRVFeatures] = useState<HRVFeatureData>(mockHRVFeatures);
  const [metrics, setMetrics] = useState<ModelMetrics>(mockModelMetrics);
  const [ablation, setAblation] = useState<AblationComparison>(mockAblation);
  const [subjectLosos, setSubjectLosos] = useState({ accuracy: 0, macro_f1: 0 });
  const [recall, setRecall] = useState<Record<string, number>>({ baseline: 0, stress: 0, amusement: 0 });
  const [subjectBundle, setSubjectBundle] = useState<SubjectBundle | null>(null);
  const [selectedState, setSelectedState] = useState<"baseline" | "stress" | "amusement">("stress");
  const [currentTime, setCurrentTime] = useState(new Date());

  // 15 名被试（无 S12）
  const SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"];
  const [currentSubject, setCurrentSubject] = useState("S2");

  // 把某状态的信号套到展示面板
  function applyState(bundle: SubjectBundle | null, stateName: "baseline" | "stress" | "amusement") {
    setSelectedState(stateName);
    const view = bundle?.states[stateName];
    if (view) {
      setPPGSignal(view.signal_ppg);
      setEDASignal(view.signal_eda);
      setSpectrum(view.spectrum);
      setSpectrogram(view.spectrogram);
    }
  }

  // 按被试切换：fetch 该被试的打包 JSON
  async function loadSubjectData(sid: string) {
    const data = await loadJson<SubjectBundle | null>(`/data/subjects/${sid}.json`, null);
    if (!data) return;
    setSubjectBundle(data);
    setSubjectInfo((prev) => ({
      ...prev,
      subject_id: data.subject_id,
      recording_duration: `~${data.recording_duration_min} min`,
    }));
    setRecall(data.recall);
    setPrediction({ true_label: selectedState, predicted_label: data.prediction.predicted_label,
                   probabilities: data.prediction.probabilities });
    setSubjectLosos(data.loso);
    // 若当前选中状态在该被试存在则套用，否则回退到首个可用状态
    const want = data.states[selectedState] ? selectedState
      : (data.states.stress ? "stress" : Object.keys(data.states)[0] as any);
    applyState(data, want);
  }

  // 加载外部数据（如果存在）
  useEffect(() => {
    const loadAll = async () => {
      const [si, ppg, eda, sp, spt, pr, cm, tc, hrv, mm, ab] = await Promise.all([
        loadJson<SubjectInfo>("data/subject_info.json", mockSubjectInfo),
        loadJson<SignalData>("data/signal_ppg.json", mockPPGSignal),
        loadJson<SignalData>("data/signal_eda.json", mockEDASignal),
        loadJson<SpectrumData>("data/spectrum_fft.json", mockSpectrum),
        loadJson<SpectrogramData>("data/spectrogram_stft.json", mockSpectrogram),
        loadJson<PredictionResult>("data/model_prediction.json", mockPrediction),
        loadJson<ConfusionMatrix>("data/confusion_matrix.json", mockConfusionMatrix),
        loadJson<TrainingCurves>("data/training_curves.json", mockTrainingCurves),
        loadJson<HRVFeatureData>("data/hrv_features.json", mockHRVFeatures),
        loadJson<ModelMetrics>("data/model_metrics.json", mockModelMetrics),
        loadJson<AblationComparison>("data/ablation_comparison.json", mockAblation),
      ]);
      setSubjectInfo(si);
      setPPGSignal(ppg);
      setEDASignal(eda);
      setSpectrum(sp);
      setSpectrogram(spt);
      setPrediction(pr);
      setConfMatrix(cm);
      setTrainCurves(tc);
      setHRVFeatures(hrv);
      setMetrics(mm);
      setAblation(ab);
    };
    loadAll();
    loadSubjectData("S2");
  }, []);

  // 时钟
  useEffect(() => {
    const t = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const stateLabels: Record<string, string> = {
    baseline: "基线",
    stress: "压力",
    amusement: "娱乐",
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* ====== 顶部导航栏 ====== */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <h1 className="text-base font-semibold text-slate-800 tracking-tight">
              可穿戴情感状态监测系统
            </h1>
            <Badge variant="outline" className="text-[10px] ml-2 bg-slate-50">WESAD Dataset</Badge>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-500">
            <div className="flex items-center gap-1.5">
              <Timer className="w-3.5 h-3.5" />
              <span className="font-mono text-xs">
                {currentTime.toLocaleTimeString("zh-CN")}
              </span>
            </div>
            <Separator orientation="vertical" className="h-4" />
            <div className="flex items-center gap-1.5">
              <User className="w-3.5 h-3.5" />
              <select
                value={currentSubject}
                onChange={(e) => {
                  setCurrentSubject(e.target.value);
                  loadSubjectData(e.target.value);
                }}
                className="text-xs border border-slate-200 rounded px-1.5 py-0.5 bg-white text-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-400"
                title="切换被试"
              >
                {SUBJECTS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
              <div className="flex items-center bg-slate-100 rounded p-0.5">
                {(["baseline", "stress", "amusement"] as const).map((st) => {
                  const ok = !!subjectBundle?.states[st];
                  const active = selectedState === st;
                  return (
                    <button
                      key={st}
                      disabled={!ok}
                      onClick={() => applyState(subjectBundle, st)}
                      className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${
                        active ? "bg-indigo-600 text-white" : ok ? "text-slate-600 hover:bg-slate-200" : "text-slate-300 cursor-not-allowed"
                      }`}
                      title={ok ? `查看${stateLabels[st]}段` : "该被试无此状态干净段"}
                    >
                      {stateLabels[st]}
                    </button>
                  );
                })}
              </div>
              <span className="text-[10px] text-slate-400">LOSO acc {(subjectLosos.accuracy * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-4 py-4 space-y-4">
        {/* ====== 第一行：状态概览 ====== */}
        <div className="grid grid-cols-4 gap-4">
          {/* 展示状态 + 逐类检出率 */}
          <Card className="col-span-1 border-l-4 border-l-indigo-500">
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-slate-500 flex items-center gap-1.5">
                <Brain className="w-3.5 h-3.5" />
                当前查看状态（真实标签）
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl font-bold text-slate-800">
                  {stateLabels[selectedState] || selectedState}
                </span>
                <span className="text-[10px] text-slate-400">10s 代表性窗口</span>
              </div>
              <div className="space-y-1">
                {(["stress", "baseline", "amusement"] as const).map((s) => (
                  <div key={s} className="flex items-center gap-2 text-[11px]">
                    <span className="text-slate-500 w-8">{stateLabels[s]}</span>
                    <div className="flex-1 bg-slate-100 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${s === "stress" ? "bg-rose-500" : s === "baseline" ? "bg-blue-500" : "bg-emerald-500"}`}
                        style={{ width: `${(recall[s] || 0) * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-slate-700 w-9 text-right">
                      {((recall[s] || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-slate-400 mt-1.5">该被试 LOSO 逐类检出率（高亮=当前查看）</p>
            </CardContent>
          </Card>

          {/* 信号质量 */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-slate-500 flex items-center gap-1.5">
                <Heart className="w-3.5 h-3.5" />
                信号质量 (SQI)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <span className="text-lg font-semibold text-slate-700">
                {subjectInfo.signal_quality === "excellent" ? "优" :
                 subjectInfo.signal_quality === "good" ? "良" :
                 subjectInfo.signal_quality === "fair" ? "中" : "差"}
              </span>
              <div className="w-full bg-slate-100 rounded-full h-1.5 mt-2">
                <div className="bg-emerald-500 h-1.5 rounded-full" style={{ width: "92%" }} />
              </div>
            </CardContent>
          </Card>

          {/* 模型精度 */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-slate-500 flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5" />
                模型准确率
              </CardTitle>
            </CardHeader>
            <CardContent>
              <span className="text-2xl font-bold text-indigo-600">
                {(metrics.accuracy * 100).toFixed(1)}%
              </span>
              <p className="text-[11px] text-slate-400 mt-1">
                LOSO交叉验证 / Macro-F1: {(metrics.macro_f1 * 100).toFixed(1)}%
              </p>
            </CardContent>
          </Card>

          {/* 模型规模 */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-slate-500 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5" />
                模型规模
              </CardTitle>
            </CardHeader>
            <CardContent>
              <span className="text-lg font-semibold text-slate-700">
                {metrics.params_count.toLocaleString()}
              </span>
              <p className="text-[11px] text-slate-400 mt-1">
                参数 / {metrics.model_size_mb} MB
              </p>
            </CardContent>
          </Card>
        </div>

        {/* ====== 第二行：波形图 ====== */}
        <div className="grid grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-0">
              <CardTitle className="text-xs font-medium text-slate-500 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-rose-500" />
                PPG 光电容积脉搏波 ({subjectInfo.ppg_sampling_rate} Hz)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <ReactECharts
                option={buildWaveOption(ppgSignal, "PPG波形", "#f43f5e", "幅度 (mV)")}
                style={{ height: 220 }}
                notMerge
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-0">
              <CardTitle className="text-xs font-medium text-slate-500 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                EDA 皮肤电活动 ({subjectInfo.eda_sampling_rate} Hz)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <ReactECharts
                option={buildWaveOption(edaSignal, "EDA波形", "#f59e0b", "电导 (\u03bcS)")}
                style={{ height: 220 }}
                notMerge
              />
            </CardContent>
          </Card>
        </div>

        {/* ====== 第三行：频谱 + 时频 ====== */}
        <div className="grid grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-0">
              <CardTitle className="text-xs font-medium text-slate-500 flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5 text-indigo-500" />
                频域分析 (FFT)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <ReactECharts
                option={buildSpectrumOption(spectrum)}
                style={{ height: 240 }}
                notMerge
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-0">
              <CardTitle className="text-xs font-medium text-slate-500 flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5 text-cyan-500" />
                时频分析 (STFT)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <ReactECharts
                option={buildSpectrogramOption(spectrogram)}
                style={{ height: 240 }}
                notMerge
              />
            </CardContent>
          </Card>
        </div>

        {/* ====== 第四行：混淆矩阵 + HRV ====== */}
        <div className="grid grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-0">
              <CardTitle className="text-xs font-medium text-slate-500">
                混淆矩阵 (LOSO交叉验证)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <ReactECharts
                option={buildConfusionOption(confMatrix)}
                style={{ height: 260 }}
                notMerge
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-0">
              <CardTitle className="text-xs font-medium text-slate-500">
                HRV特征分布 (按情感状态分组)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <ReactECharts
                option={buildHRVBoxOption(hrvFeatures)}
                style={{ height: 260 }}
                notMerge
              />
            </CardContent>
          </Card>
        </div>

        {/* ====== 第五行：训练曲线 + 概率 + 指标 ====== */}
        <div className="grid grid-cols-3 gap-4">
          <Card className="col-span-2">
            <CardHeader className="pb-0">
              <CardTitle className="text-xs font-medium text-slate-500">
                模型训练曲线 (LOSO Fold 汇总)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <ReactECharts
                option={buildTrainingOption(trainCurves)}
                style={{ height: 250 }}
                notMerge
              />
            </CardContent>
          </Card>

          <div className="space-y-4">
            {/* 概率柱状图 */}
            <Card>
              <CardHeader className="pb-0">
                <CardTitle className="text-xs font-medium text-slate-500">
                  预测分布（全窗口聚合）
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-2">
                <ReactECharts
                  option={buildProbOption(prediction)}
                  style={{ height: 120 }}
                  notMerge
                />
              </CardContent>
            </Card>

            {/* 指标卡片 */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs font-medium text-slate-500">
                  分类性能指标
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {Object.entries(metrics.f1).map(([label, val]) => (
                  <div key={label} className="flex justify-between items-center text-xs">
                    <span className="text-slate-500 capitalize">{stateLabels[label] || label}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-slate-100 rounded-full h-1.5">
                        <div
                          className="bg-indigo-500 h-1.5 rounded-full"
                          style={{ width: `${(val as number) * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-slate-700 w-10 text-right">
                        {((val as number) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* ====== 第六行：H2 消融实验（新增） ====== */}
        <div className="grid grid-cols-3 gap-4">
          <Card className="col-span-2">
            <CardHeader className="pb-0">
              <CardTitle className="text-xs font-medium text-slate-500">
                H2 融合策略消融对比（15 折 LOSO）
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <ReactECharts
                option={buildAblationOption(ablation)}
                style={{ height: 250 }}
                notMerge
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-slate-500">
                Wilcoxon 显著性检验（vs Dual）
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2.5 pt-1">
              {ablation.wilcoxon.map((w) => (
                <div key={w.comparison} className="flex items-center justify-between text-xs">
                  <span className="text-slate-600 font-mono">{w.comparison}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400">p={w.p_value.toFixed(3)}</span>
                    <Badge
                      variant="outline"
                      className={
                        w.significant
                          ? "bg-emerald-50 text-emerald-700 border-emerald-200 text-[10px]"
                          : "bg-slate-50 text-slate-500 border-slate-200 text-[10px]"
                      }
                    >
                      {w.significant ? "显著 **" : "不显著"}
                    </Badge>
                  </div>
                </div>
              ))}
              <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-400 leading-relaxed">
                特征级融合（Dual）显著优于 PPG 单模态；EDA-only 仅 3.1K 参数即达
                66.8%，参数效率最高。
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ====== 底部信息栏 ====== */}
        <div className="bg-white rounded-lg border border-slate-200 px-4 py-3 flex items-center justify-between text-[11px] text-slate-400">
          <div className="flex items-center gap-4">
            <span>WESAD Dataset | Schmidt et al., ICMI 2018</span>
            <Separator orientation="vertical" className="h-3" />
            <span>1D-CNN + PPG/EDA Fusion</span>
            <Separator orientation="vertical" className="h-3" />
            <span>LOSO Cross-Validation</span>
          </div>
          <div className="flex items-center gap-1">
            <span>数据接口定义见</span>
            <code className="bg-slate-100 px-1 rounded text-[10px]">src/types/dashboard.ts</code>
          </div>
        </div>
      </main>
    </div>
  );
}
