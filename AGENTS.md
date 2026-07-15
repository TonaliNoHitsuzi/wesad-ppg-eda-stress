# AGENTS.md

本文件用于指导 AI 编程助手（如 opencode）在本仓库中工作。

## 项目概述

基于 WESAD 数据集的 PPG+EDA 多模态情感状态三分类（基线/压力/娱乐）。
后端 Python 流水线产出的 JSON 由前端 React Dashboard 读取展示。
主模型为**不对称双分支 1D-CNN**（21,627 参数），严格 LOSO 评估。
详见 `README.md` 与 `docs/项目计划书.md`。

## 关键约定

- 原始数据放在 `data/raw/`（S2.pkl ~ S17.pkl，无 S12），**不纳入 git**，不要提交。
- 前端数据由 `src/export/export_json.py`（全局 5 个 JSON）与
  `scripts/export_subjects.py`（15 个被试 JSON）生成到 `frontend/public/data/`；
  缺失时前端回退到 `frontend/src/lib/mockData.ts`。
- 评估必须使用 **LOSO 留一被试交叉验证**，禁止随机划分。
- HRV 分析使用**胸部 ECG R 峰**（金标准），分类模型仍用**手腕 PPG+EDA**。
- per-subject z-score（每被试自身 μ/σ），非 per-window 或 global。

## Python 端（Python 3.10+）

```bash
# 激活虚拟环境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python -m src.train.train_loso --variant dual      # 主模型 15 折 LOSO
python scripts/run_ablation.py                      # H2 四变体消融
python scripts/make_figures.py                      # 生成 fig01-10
python src/export/export_json.py                    # 全局 JSON
python scripts/export_subjects.py                   # 被试 JSON
```

## 前端（frontend/）

```bash
cd frontend
npm install
npm run dev       # 开发服务器 http://localhost:5173
npm run build     # 类型检查 + 构建
npm run lint      # ESLint
```

`base: './'`（vite.config.ts）+ `import.meta.env.BASE_URL` 前缀适配 GitHub Pages 子路径部署。

## 提交前自查

- 新增 Python 文件须能被 `python -c "import ast; ast.parse(open('文件').read())"` 解析。
- 修改前端后运行 `cd frontend && npm run build` 确认无类型错误。
- 不要提交：`data/raw/*.pkl`、`data/raw/WESAD.zip`、`results/models/*.pth`、`node_modules/`、`.venv/`、`frontend/dist/`。
- 论文图表统一输出到 `results/figures/`，命名与论文图号一致（如 `fig01_raw_signals.png`）。
- 历史文档放 `archive/`，正式交付物不放版本后缀（如"初稿""完整版""改进版"）。
