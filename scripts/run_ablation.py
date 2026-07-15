"""H2 消融：共享缓存跑 ppg/eda/late 三个变体（dual 已有结果）。

用法： python scripts/run_ablation.py
输出： results/json/loso_{variant}.json
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import set_seed
from src.data.loader import list_available_subjects
from src.train.train_loso import prepare_all_subjects, run_loso

set_seed(42)
subs = list_available_subjects()
print(f"=== H2 消融：{len(subs)} 被试 ===\n[1/2] 构建共享预处理缓存（仅一次）...")
t0 = time.time()
cache = prepare_all_subjects(subs)
print(f"缓存构建完成: {time.time()-t0:.1f}s\n")

results = {}
for variant in ["ppg", "eda", "late"]:
    print(f"\n[2/2] 训练变体: {variant}  {'='*40}")
    set_seed(42)
    t1 = time.time()
    res = run_loso(subs, variant=variant, epochs=60, cache=cache)
    res["train_time_sec"] = round(time.time() - t1, 1)
    out = ROOT / "results" / "json" / f"loso_{variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    results[variant] = res
    print(f"  >> {variant}: acc={res['accuracy_mean']:.3f}±{res['accuracy_std']:.3f}  "
          f"macro_f1={res['macro_f1_mean']:.3f}±{res['macro_f1_std']:.3f}  "
          f"({res['train_time_sec']}s)")

# 汇总对比表（含已有 dual）
print("\n" + "=" * 60)
print("H2 消融汇总")
print("=" * 60)
try:
    with open(ROOT / "results" / "json" / "loso_dual.json", encoding="utf-8") as f:
        results["dual"] = json.load(f)
except FileNotFoundError:
    pass
print(f"{'variant':<8} {'acc':>14} {'macro_f1':>14} {'params':>8}")
for v in ["dual", "ppg", "eda", "late"]:
    if v in results:
        r = results[v]
        print(f"{v:<8} {r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}     "
              f"{r['macro_f1_mean']:.3f}±{r['macro_f1_std']:.3f}     "
              f"{r.get('params_count', '?'):>8}")
