"""LOSO 留一被试交叉验证训练循环。

落地决策：
  - N1：测试被试绝不参与 early stopping；训练窗口内 8:2 随机分 train/val，val 仅用于 early stop。
  - N3：class-weighted CrossEntropy（权重=训练集各类频率倒数），主指标 macro-F1。
  - N6：set_seed(42)，每折记录 per-subject macro-F1 供 Wilcoxon 检验。
  - N7：选 val 准确率中位数对应 fold 作为"代表性单折"，保存其逐 epoch 训练曲线。
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import set_seed
from src.data.loader import (list_available_subjects, load_subject,
                             extract_wrist_signals, extract_windows,
                             window_class_counts, WindowSet, STATE_NAMES)
from src.preprocess.filters import preprocess_subject, sqi_mask
from src.model.cnn import build_model

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 数据准备：全被试一次性预处理 + 分窗 + SQI（缓存复用）
# ─────────────────────────────────────────────
@dataclass
class SubjectCache:
    ppg: np.ndarray   # (N, 640)
    eda: np.ndarray   # (N, 40)
    labels: np.ndarray  # (N,)


def prepare_subject(subject_id: str, apply_sqi: bool = True) -> SubjectCache:
    """单被试：加载→滤波→per-subject z-score→分窗→(SQI 过滤)。"""
    raw = load_subject(subject_id)
    sig = extract_wrist_signals(raw)
    bvp_z, eda_z, _ = preprocess_subject(sig["bvp"], sig["eda"])
    ws = extract_windows(bvp_z, eda_z, sig["label"], subject_id)
    if apply_sqi:
        keep = sqi_mask(ws.ppg)
        ws = WindowSet(subject_id, ws.ppg[keep], ws.eda[keep],
                       ws.labels[keep], ws.raw_state_labels[keep])
    return SubjectCache(ws.ppg, ws.eda, ws.labels)


def prepare_all_subjects(subjects: list[str], apply_sqi: bool = True
                         ) -> dict[str, SubjectCache]:
    cache: dict[str, SubjectCache] = {}
    for sid in subjects:
        cache[sid] = prepare_subject(sid, apply_sqi=apply_sqi)
        c = window_class_counts(WindowSet(sid, cache[sid].ppg, cache[sid].eda,
                                          cache[sid].labels, cache[sid].labels))
        total = sum(c.values())
        print(f"  [{sid}] windows={total:>4d}  "
              f"baseline={c['baseline']:>3d} stress={c['stress']:>3d} "
              f"amusement={c['amusement']:>3d}")
    return cache


# ─────────────────────────────────────────────
# 训练单折
# ─────────────────────────────────────────────
def _split_train_val(n: int, val_ratio: float = 0.2, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]   # train_idx, val_idx


def _class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = len(labels) / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


@dataclass
class FoldHistory:
    epoch: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)


def train_one_fold(train_cache: dict[str, SubjectCache], test_subj: str,
                   *, variant: str = "dual", epochs: int = 60,
                   lr: float = 1e-3, batch_size: int = 64,
                   patience: int = 10) -> tuple[dict, FoldHistory]:
    """训练一个 LOSO 折。返回 (该折测试指标, 训练历史)。"""
    set_seed(SEED)

    # 合并训练被试窗口
    tr_ppg = np.concatenate([c.ppg for s, c in train_cache.items() if s != test_subj])
    tr_eda = np.concatenate([c.eda for s, c in train_cache.items() if s != test_subj])
    tr_lab = np.concatenate([c.labels for s, c in train_cache.items() if s != test_subj])
    te = train_cache[test_subj]

    # 8:2 随机分 train/val（N1）
    tr_idx, va_idx = _split_train_val(len(tr_lab))
    ppg_t = torch.from_numpy(tr_ppg[tr_idx]).float().unsqueeze(1)
    eda_t = torch.from_numpy(tr_eda[tr_idx]).float().unsqueeze(1)
    lab_t = torch.from_numpy(tr_lab[tr_idx]).long()
    ppg_v = torch.from_numpy(tr_ppg[va_idx]).float().unsqueeze(1)
    eda_v = torch.from_numpy(tr_eda[va_idx]).float().unsqueeze(1)
    lab_v = torch.from_numpy(tr_lab[va_idx]).long()

    train_dl = DataLoader(TensorDataset(ppg_t, eda_t, lab_t),
                          batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(ppg_v, eda_v, lab_v),
                        batch_size=batch_size, shuffle=False)

    model = build_model(variant).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(lab_t.numpy()))

    hist = FoldHistory()
    best_val_loss, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tl, tc, nb = 0.0, 0, 0
        for ppg_b, eda_b, lab_b in train_dl:
            ppg_b, eda_b, lab_b = ppg_b.to(DEVICE), eda_b.to(DEVICE), lab_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ppg_b, eda_b)
            loss = criterion(logits, lab_b)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(lab_b)
            tc += (logits.argmax(1) == lab_b).sum().item()
            nb += len(lab_b)
        tr_loss, tr_acc = tl / nb, tc / nb

        # ---- val（仅 early stopping）----
        model.eval()
        vl, vc, nv = 0.0, 0, 0
        with torch.no_grad():
            for ppg_b, eda_b, lab_b in val_dl:
                ppg_b, eda_b, lab_b = ppg_b.to(DEVICE), eda_b.to(DEVICE), lab_b.to(DEVICE)
                logits = model(ppg_b, eda_b)
                vl += criterion(logits, lab_b).item() * len(lab_b)
                vc += (logits.argmax(1) == lab_b).sum().item()
                nv += len(lab_b)
        v_loss, v_acc = vl / nv, vc / nv

        hist.epoch.append(ep)
        hist.train_loss.append(tr_loss)
        hist.val_loss.append(v_loss)
        hist.train_acc.append(tr_acc)
        hist.val_acc.append(v_acc)

        if v_loss < best_val_loss - 1e-4:
            best_val_loss, bad = v_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- 测试被试评估 ----
    model.eval()
    with torch.no_grad():
        ppg_te = torch.from_numpy(te.ppg).float().unsqueeze(1).to(DEVICE)
        eda_te = torch.from_numpy(te.eda).float().unsqueeze(1).to(DEVICE)
        logits = model(ppg_te, eda_te)
        pred = logits.argmax(1).cpu().numpy()

    metrics = _eval_metrics(te.labels, pred)
    metrics["final_val_acc"] = hist.val_acc[-1] if hist.val_acc else 0.0
    return metrics, hist


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {STATE_NAMES[i]: {"precision": float(prec[i]),
                                       "recall": float(rec[i]),
                                       "f1": float(f1[i])} for i in range(3)},
        "confusion": cm,
    }


# ─────────────────────────────────────────────
# LOSO 主循环
# ─────────────────────────────────────────────
def run_loso(subjects: list[str] | None = None, *, variant: str = "dual",
             epochs: int = 60, cache: dict | None = None) -> dict:
    subjects = subjects or list_available_subjects()
    if not subjects:
        raise RuntimeError("data/raw 无可用被试，请先放入 WESAD 数据。")
    print(f"[LOSO] 被试 {len(subjects)} 人，variant={variant}")
    if cache is None:
        print("[LOSO] 预处理 + 分窗 + SQI（仅执行一次，缓存复用）...")
        cache = prepare_all_subjects(subjects)
    else:
        print("[LOSO] 复用已构建的预处理缓存")

    per_subject = {}
    histories: dict[str, FoldHistory] = {}
    agg_cm = np.zeros((3, 3), dtype=np.int64)
    for test_subj in subjects:
        set_seed(SEED)
        print(f"\n[LOSO] fold test={test_subj} ...", flush=True)
        m, hist = train_one_fold(cache, test_subj, variant=variant, epochs=epochs)
        per_subject[test_subj] = m
        histories[test_subj] = hist
        agg_cm += np.array(m["confusion"])
        print(f"  acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")

    # ── 聚合 ───────────────────────────────
    accs = [per_subject[s]["accuracy"] for s in subjects]
    f1s = [per_subject[s]["macro_f1"] for s in subjects]

    # 代表性单折：val_acc 中位数对应的 fold（N7）
    rep = min(subjects, key=lambda s: abs(per_subject[s]["final_val_acc"] - np.median(accs)))
    rep_hist = histories[rep]

    summary = {
        "variant": variant,
        "n_subjects": len(subjects),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "per_subject": per_subject,
        "confusion_matrix": agg_cm.tolist(),
        "labels": STATE_NAMES,
        "representative_fold": rep,
        "params_count": build_model(variant).count_parameters(),
        "training_curves": {
            "description": "代表性单折（验证 acc 最接近全体中位数）",
            "fold_id": rep,
            "epochs": rep_hist.epoch,
            "train_loss": rep_hist.train_loss, "val_loss": rep_hist.val_loss,
            "train_acc": rep_hist.train_acc, "val_acc": rep_hist.val_acc,
        },
    }
    print(f"\n[LOSO] 汇总: acc={summary['accuracy_mean']:.3f}±{summary['accuracy_std']:.3f}  "
          f"macro_f1={summary['macro_f1_mean']:.3f}±{summary['macro_f1_std']:.3f}  "
          f"params={summary['params_count']}")
    return summary


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dual")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "results" / "json" / "loso_result.json"))
    a = ap.parse_args()
    res = run_loso(variant=a.variant, epochs=a.epochs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[LOSO] 结果已保存: {a.out}")
