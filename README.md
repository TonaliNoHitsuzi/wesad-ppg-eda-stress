基于PPG+EDA多模态信号的轻量级可穿戴情感状态检测系统

课程论文项目 | 使用WESAD公开数据集，采用轻量级1D-CNN实现基线/压力/娱乐三分类，
LOSO留一被试交叉验证，并配套医学Dashboard风格的前端可视化展示。

技术栈：Python (PyTorch/NumPy/SciPy) + React (ECharts)

---

## 简介

本项目面向可穿戴设备的资源约束场景，设计了一种参数量仅约 3–5 万的双分支 1D-CNN，
联合 PPG（光电容积脉搏波，64 Hz）与 EDA（皮肤电活动，4 Hz）两种生理信号，
实现 **基线 / 压力 / 娱乐** 三类情感状态的自动检测。

研究严格采用 **LOSO 留一被试交叉验证**（Leave-One-Subject-Out）以评估被试独立泛化能力，
避免传统随机划分导致的被试信息泄露。配套一个 **医学监护仪风格的 React Dashboard**，
将信号处理全流程与模型推理结果以可交互方式呈现，体现可穿戴健康监测的落地应用形态。

## 核心特性

- **多模态特征级融合**：PPG / EDA 双分支 1D-CNN，全局平均池化后拼接为 256 维特征
- **完整预处理链路**：Chebyshev II 带通 + 工频陷波 + SQI 信号质量筛选 + Z-score 标准化
- **多维度信号分析**：时域（形态学+HRV）、频域（FFT+LF/HF）、时频域（STFT）
- **严格的被试独立评估**：LOSO 交叉验证，15 折
- **轻量级模型**：约 4.5 万参数，可部署到 ARM Cortex-M 级可穿戴设备
- **交互式前端**：ECharts 实现 PPG/EDA 波形、频谱、时频热力图、混淆矩阵、雷达图

## 目录结构

```
wesad-ppg-eda-stress/
├── data/                   # 数据集（不纳入版本控制）
│   ├── raw/                # WESAD 原始 .pkl（S2~S17，需手动放入）
│   ├── interim/            # 中间处理结果
│   └── processed/          # 分段后的训练窗口（.npz）
├── src/                    # Python 源码
│   ├── data/               #   数据加载
│   ├── preprocess/         #   滤波 / SQI / 标准化 / 分段
│   ├── features/           #   时域 / 频域 / 时频 / HRV 特征
│   ├── model/              #   双分支 1D-CNN
│   ├── train/              #   LOSO 训练循环
│   ├── evaluate/           #   指标与混淆矩阵
│   └── export/             #   导出 10 个 JSON 供前端使用
├── scripts/                # 端到端运行脚本
├── notebooks/              # 探索性分析
├── results/                # 实验产出
│   ├── figures/            #   论文图表（8+ 幅）
│   ├── json/               #   导出的前端 JSON
│   ├── models/             #   训练权重（.pth）
│   └── logs/               #   训练日志
├── frontend/               # React + TS + Vite + ECharts 前端
│   └── public/data/        #   前端读取的 10 个 JSON
├── docs/                   # 任务书与项目计划书
├── paper/                  # 课程论文 PDF 与参考文献
├── requirements.txt        # Python 依赖
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# Python 端（推荐 Python 3.10+）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 数据准备

从 WESAD 官方渠道获取数据集，将 15 个被试文件放入 `data/raw/`：

```
data/raw/
├── S2.pkl
├── S3.pkl
├── ...
└── S17.pkl
```

详见 [`data/raw/README.md`](data/raw/README.md)。

### 3. 运行处理流水线

```bash
python scripts/run_pipeline.py            # 完整流水线：预处理 → 分析 → 训练 → 导出
# 或分步运行：
python -m src.preprocess.pipeline
python -m src.train.train_loso
python -m src.export.export_json
```

### 4. 启动前端 Dashboard

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 http://localhost:5173 ，当 `frontend/public/data/` 下无真实 JSON 时自动加载内置 mock 数据。

## 数据集

**WESAD (Wearable Stress and Affect Detection)** —— Schmidt et al., ICMI 2018

| 属性 | 说明 |
|------|------|
| 被试 | 15 人（12 男 3 女），25.8 ± 4.0 岁 |
| 信号 | PPG(64Hz)、EDA(4Hz)、ECG(700Hz)、TEMP、RESP、EMG、ACC |
| 状态 | 基线 / 压力(TSST) / 娱乐 |
| 设备 | RespiBAN Pro（胸带）+ Empatica E4（手腕） |
| 格式 | Python pickle (.pkl) |

## 评估方案

采用 **LOSO 留一被试交叉验证**：每次用 14 个被试训练、1 个被试测试，循环 15 次，
报告聚合后的准确率、宏 F1、每类 Precision/Recall/F1 及混淆矩阵。
该方案保证训练集与测试集无同一被试数据，反映真实的跨被试泛化性能。

## 参考文献

详见 [`paper/`](paper/)。核心参考：
- Schmidt P, et al. *Introducing WESAD, a multimodal dataset for wearable stress and affect detection.* ICMI 2018.
- Cai J, et al. *A lightweight multi-feature fusion CNN-MLP for wearable stress detection.* Sci. Rep. 2024.
- Ali M S, et al. *A Shallow Deep Learning Model for Stress Monitoring from PPG Signals.* IEEE 2025.

## 提交规范

- 课程：劳动教育实践4——专业劳动实践
- 提交截止：2026-07-25
- 仓库用于保留完整开发记录与版本管理

## License

代码部分采用 MIT 协议；WESAD 数据集遵循其原始 EULA，本项目不重新分发原始数据。
