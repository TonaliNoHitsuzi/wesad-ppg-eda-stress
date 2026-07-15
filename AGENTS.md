# AGENTS.md

本文件用于指导 AI 编程助手（如 opencode）在本仓库中工作。

## 项目概述

基于 WESAD 数据集的 PPG+EDA 多模态情感状态三分类（基线/压力/娱乐）。
后端 Python 流水线产出的 JSON 由前端 React Dashboard 读取展示。
详见 `README.md` 与 `docs/项目计划书.md`。

## 关键约定

- 原始数据放在 `data/raw/`（S2.pkl ~ S17.pkl），**不纳入 git**，不要提交。
- 前端期望的 10 个 JSON 由 `src/export/export_json.py` 生成到 `frontend/public/data/`；
  当目录为空时前端回退到 `frontend/src/lib/mockData.ts`。
- 评估必须使用 **LOSO 留一被试交叉验证**，禁止使用随机划分。
- 模型为双分支 1D-CNN，参数量控制在 3–5 万。

## Python 端

```bash
# 激活虚拟环境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt

# 运行完整流水线
python scripts/run_pipeline.py

# 分步
python -m src.preprocess.pipeline
python -m src.train.train_loso
python -m src.evaluate.metrics
python -m src.export.export_json
```

## 前端（frontend/）

```bash
cd frontend
npm install
npm run dev       # 开发服务器 http://localhost:5173
npm run build     # 类型检查 + 构建
npm run lint      # ESLint
```

## 提交前自查

- 新增 Python 文件须能被 `python -c "import ast; ast.parse(open('文件').read())"` 解析。
- 修改前端后运行 `cd frontend && npm run build` 确认无类型错误。
- 不要提交：`data/raw/*.pkl`、`results/models/*.pth`、`node_modules/`、`.venv/`。
- 论文图表统一输出到 `results/figures/`，命名与论文图号一致（如 `fig01_raw_ppg.png`）。
