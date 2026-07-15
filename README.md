# 基于 PPG+EDA 多模态信号的轻量级可穿戴情感状态检测系统

> 课程论文项目 | WESAD 公开数据集 | 轻量级 1D-CNN 三分类（基线/压力/娱乐）| LOSO 留一被试交叉验证 | 医学 Dashboard 风格前端

技术栈：**Python**（PyTorch / NumPy / SciPy）+ **React**（TypeScript / Vite / ECharts）

[![Python](https://img.shields.io/badge/Python-3.12-blue)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange)]() [![License](https://img.shields.io/badge/License-MIT-green)]()

## 成果一览

| 指标 | 数值 |
|------|------|
| 模型 | 不对称双分支 1D-CNN |
| **参数量** | **21,627（约 2.16 万）** |
| 评估 | LOSO 留一被试交叉验证（15 折） |
| **准确率** | **74.9% ± 12.5%** |
| **macro F1** | **71.0% ± 12.5%** |
| H2 消融 | 特征级融合 **显著优于** PPG 单模态（Wilcoxon p = 0.001） |
| 独立发现 | EDA 单模态判别力反强于 PPG（3,115 参数达 66.8%） |

完整论文见 [`paper/论文.md`](paper/论文.md)，图表见 [`results/figures/`](results/figures/)（含 [`图片说明.md`](results/figures/图片说明.md)），前端在线 Demo 见文末 GitHub Pages。

## 目录结构

```
wesad-ppg-eda-stress/
├── data/raw/               # WESAD 原始 .pkl（不入库，自行下载）
├── src/
│   ├── data/               #   数据加载、标签 700Hz→64Hz 重采样、窗口提取
│   ├── preprocess/         #   Chebyshev 滤波、per-subject z-score、SQI
│   ├── features/           #   FFT / STFT / HRV（ECG R 峰金标准）
│   ├── model/              #   不对称双分支 1D-CNN（4 变体可切换）
│   ├── train/              #   LOSO 训练循环
│   └── export/             #   导出前端 JSON
├── scripts/                # 出图、消融、被试导出脚本
├── results/{figures,json}/ # 论文图表 + 实验结果 JSON
├── frontend/               # React + ECharts 验证前端（可被试/状态切换）
├── docs/                   # 任务书 + 项目计划书
├── paper/                  # 课程论文
├── archive/                # 过程中已被取代的历史文档
├── requirements.txt
└── README.md
```

## 快速复现

### 1. 环境准备（Python 3.10+）

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows；macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> 国内网络加速：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

### 2. 数据集下载（WESAD）

将 15 个被试文件 `S2.pkl ~ S17.pkl`（无 S12）放入 `data/raw/`。

**下载渠道（任选其一）**：
- **推荐**：官方 sciebo 直链（zip 包，约 2 GB）：
  `https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download`
  > ⚠️ **避坑提示**：浏览器直接下载可能因 SSL/断点问题失败，**强烈建议用迅雷/IDM 等多线程下载工具**，稳定且可断点续传。下载后解压，把 `WESAD/S*/S*.pkl` 取出放入 `data/raw/`。
- 备选：Kaggle 搜索 "WESAD" 镜像。

数据集遵循 WESAD 原始 EULA：仅用于科学研究、非商业用途，发表论文须引用 Schmidt et al., ICMI 2018。详见 [`data/raw/README.md`](data/raw/README.md)。

### 3. 运行

```bash
# 主模型（特征级融合，15 折 LOSO）
python -m src.train.train_loso --variant dual

# H2 消融（4 变体）
python scripts/run_ablation.py

# 生成全部论文图表（fig01-10）
python scripts/make_figures.py

# 导出前端数据 JSON
python src/export/export_json.py
python scripts/export_subjects.py
```

### 4. 前端

```bash
cd frontend
npm install
npm run dev          # 开发服务器（默认 http://localhost:5173）
npm run build        # 生产构建到 frontend/dist/
```

支持 **15 名被试切换** + **基线/压力/娱乐三状态段查看** + 逐类检出率。详见 [`frontend/网页说明.md`](frontend/网页说明.md)。

## 在线 Demo（GitHub Pages）

推送到 main 分支后，`.github/workflows/deploy.yml` 会自动构建并部署前端到 GitHub Pages。

**首次启用**：仓库 Settings → Pages → Source 选 **GitHub Actions**。部署地址形如 `https://tonalynohitsuzi.github.io/wesad-ppg-eda-stress/`。

## 数据集与引用

**WESAD**：Schmidt P, Reiss A, Duerichen R, et al. *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018: 400-408.

## License

代码采用 MIT 协议；WESAD 数据集遵循其原始 EULA，本项目不重新分发原始数据。
