"""端到端流水线入口。

按计划书执行：数据加载 → 预处理 → 信号分析 → LOSO 训练 → 评估 → 导出 JSON。
当 data/raw/ 为空时打印提示并安全退出。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import list_available_subjects  # noqa: E402


STEPS = [
    ("1/6 数据加载与校验", "src.data.loader"),
    ("2/6 预处理（滤波/SQI/标准化/分段）", "src.preprocess.filters"),
    ("3/6 时域/频域/时频分析与特征提取", "src.features"),
    ("4/6 双分支 1D-CNN · LOSO 训练", "src.train"),
    ("5/6 评估（准确率/F1/混淆矩阵）", "src.evaluate"),
    ("6/6 导出 10 个 JSON 到 frontend/public/data", "src.export"),
]


def main() -> int:
    print("=" * 64)
    print("WESAD PPG+EDA 情感状态检测 —— 处理流水线")
    print("=" * 64)

    subjects = list_available_subjects()
    if not subjects:
        print("\n[!] data/raw/ 未发现任何 S*.pkl 文件。")
        print("    请先按 data/raw/README.md 放入 WESAD 原始数据后再运行。")
        print("    流水线骨架与目录结构已就绪，数据到位后即可启动。\n")
        print("待执行步骤：")
        for desc, _ in STEPS:
            print(f"  - {desc}")
        return 0

    print(f"发现 {len(subjects)} 名被试: {subjects}\n")
    for desc, _ in STEPS:
        print(f"==> {desc}")
    # TODO: 数据到位后逐模块接入
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
