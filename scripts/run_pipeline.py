"""端到端流水线入口。

阶段（数据到位后依次接入）：
  1) 数据加载与校验        src.data.loader
  2) 预处理（滤波/per-subject z-score/SQI）  src.preprocess.filters
  3) 时域/频域/时频分析    src.features        （下一阶段）
  4) LOSO 训练             src.train.train_loso
  5) 评估 + 导出 JSON      src.export          （下一阶段）
  6) 前端 Dashboard        frontend/public/data

数据未就绪时打印阶段清单并安全退出。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import list_available_subjects  # noqa: E402

STAGES = [
    ("1 数据加载与校验", "src.data.loader"),
    ("2 预处理 滤波/per-subject z-score/SQI", "src.preprocess.filters"),
    ("3 时域/频域/时频分析与特征提取", "src.features"),
    ("4 LOSO 训练（class-weighted + 8:2 内部 val）", "src.train.train_loso"),
    ("5 评估（acc/macro-F1/混淆矩阵/Wilcoxon）", "src.evaluate"),
    ("6 导出 10 个 JSON 到 frontend/public/data", "src.export"),
]


def main() -> int:
    print("=" * 64)
    print("WESAD PPG+EDA 情感状态检测 —— 处理流水线")
    print("=" * 64)

    subjects = list_available_subjects()
    if not subjects:
        print("\n[!] data/raw/ 未发现任何 S*.pkl 文件。")
        print("    请先按 data/raw/README.md 放入 WESAD 原始数据后再运行。\n")
        print("待执行阶段：")
        for desc, _ in STAGES:
            print(f"  - {desc}")
        print("\n骨架与目录结构已就绪，数据到位后即可启动。")
        return 0

    print(f"发现 {len(subjects)} 名被试: {subjects}\n")
    print("阶段：")
    for desc, _ in STAGES:
        print(f"  ==> {desc}")
    print("\n提示：LOSO 训练请运行  python -m src.train.train_loso --variant dual")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
