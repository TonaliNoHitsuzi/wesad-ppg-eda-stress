# 前端可视化验证系统

WESAD 三分类情感状态检测的研究成果验证看板。医学监护仪风格，支持 15 名被试切换 + 基线/压力/娱乐三状态段查看。

详细面板说明见 [`网页说明.md`](网页说明.md)。

## 技术栈

React 19 + TypeScript + Vite 7 + ECharts 6 + Tailwind CSS 3 + shadcn/ui

## 运行

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # 输出到 dist/
npm run lint     # ESLint
```

## 架构

**纯静态预计算**：Python 流水线一次性导出 JSON 到 `public/data/`，前端 fetch 读取，无后端服务。

- `public/data/*.json`：全局聚合数据（混淆矩阵、HRV、训练曲线、消融等 5 个）
- `public/data/subjects/S*.json`：15 个被试各自的打包数据（三状态波形 + 逐类检出率）
- 缺失时回退到 `src/lib/mockData.ts`

`src/types/dashboard.ts` 定义全部数据接口契约。`base: './'`（vite.config.ts）使构建产物使用相对路径，便于 GitHub Pages 部署。

## 数据接口

11 个 JSON 文件的结构详见 [`public/data/README.md`](public/data/README.md) 与 `src/types/dashboard.ts`。重新生成：

```bash
python ../src/export/export_json.py       # 全局 JSON
python ../scripts/export_subjects.py      # 被试 JSON
```
