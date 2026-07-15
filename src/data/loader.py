"""WESAD 数据加载器。

读取 data/raw/S{X}.pkl，按 wrist 子集提取 PPG(64Hz) / EDA(4Hz)，
并将 700Hz 的 label 重采样到各信号采样率。

实现见 src/data/loader.py（待实现）。
"""
from __future__ import annotations
from pathlib import Path
import pickle

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

VALID_SUBJECTS = [f"S{i}" for i in range(2, 18)]  # S2..S17，共 15 名被试

LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement"}


def load_subject(subject_id: str, raw_dir: Path | None = None) -> dict:
    """加载单个被试的 .pkl。

    Parameters
    ----------
    subject_id : "S2" .. "S17"
    raw_dir : 数据根目录，默认 data/raw

    Returns
    -------
    dict
        {
          "subject": "S2",
          "bvp":  ndarray (n,1)  @ 64Hz,
          "eda":  ndarray (n,1)  @ 4Hz,
          "label_bvp": ndarray (n,) @ 64Hz,
          "label_eda": ndarray (m,) @ 4Hz,
        }
    """
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    pkl_path = raw_dir / f"{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"未找到 {pkl_path}，请先按 data/raw/README.md 放入 WESAD 原始数据。"
        )
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def list_available_subjects(raw_dir: Path | None = None) -> list[str]:
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    return sorted(
        p.stem for p in raw_dir.glob("S*.pkl") if p.stem in VALID_SUBJECTS
    )
