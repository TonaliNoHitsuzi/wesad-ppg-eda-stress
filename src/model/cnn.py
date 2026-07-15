"""双分支 1D-CNN（不对称），参数量 ≈ 2.2 万。

架构（13 条决策落地）：
  - PPG 分支：3×Conv(k=5/3/3) + 3×MaxPool，输入 (B,1,640) → GAP → (B,64)
  - EDA 分支：2×Conv(k=3/3)   + 2×MaxPool，输入 (B,1,40)  → GAP → (B,24)
  - 特征级融合：concat(88) → FC(88→64) + ELU + Dropout → FC(64→3)

H2 消融通过 variant 参数切换：
  - "dual"   特征级融合（主模型）
  - "ppg"    仅 PPG 单模态（FC 64→64→3）
  - "eda"    仅 EDA 单模态（FC 24→64→3）
  - "late"   晚期融合（两分支各自出 logits 后概率平均）
"""
from __future__ import annotations
import torch
import torch.nn as nn

Variant = str  # "dual" | "ppg" | "eda" | "late"


class ConvBlock(nn.Module):
    """Conv1d + ELU + BatchNorm1d + MaxPool1d(2)。"""

    def __init__(self, in_ch: int, out_ch: int, k: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(out_ch),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PPGBranch(nn.Module):
    """PPG 编码器：640 → 80 → GAP → 64 维特征。通道 32→48→64。"""

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 32, k=5),   # 640→320
            ConvBlock(32, 48, k=3),  # 320→160
            ConvBlock(48, 64, k=3),  # 160→80
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gap(self.net(x)).squeeze(-1)  # (B, 64)
        return h


class EDABranch(nn.Module):
    """EDA 编码器：40 → 10 → GAP → 24 维特征。通道 16→24。"""

    def __init__(self, feat_dim: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 16, k=3),   # 40→20
            ConvBlock(16, 24, k=3),  # 20→10
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gap(self.net(x)).squeeze(-1)  # (B, 24)
        return h


class WESADClassifier(nn.Module):
    """不对称双分支 1D-CNN，支持 H2 消融的 4 种变体。"""

    def __init__(self, num_classes: int = 3, variant: Variant = "dual",
                 dropout: float = 0.3):
        super().__init__()
        assert variant in ("dual", "ppg", "eda", "late"), f"未知 variant: {variant}"
        self.variant = variant
        self.num_classes = num_classes

        self.use_ppg = variant in ("dual", "ppg", "late")
        self.use_eda = variant in ("dual", "eda", "late")

        if self.use_ppg:
            self.ppg = PPGBranch(feat_dim=64)
        if self.use_eda:
            self.eda = EDABranch(feat_dim=24)

        if variant == "dual":
            in_dim = 64 + 24  # 88
            self.head = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ELU(inplace=True),
                nn.Dropout(dropout), nn.Linear(64, num_classes),
            )
        elif variant == "ppg":
            self.head = nn.Sequential(
                nn.Linear(64, 64), nn.ELU(inplace=True),
                nn.Dropout(dropout), nn.Linear(64, num_classes),
            )
        elif variant == "eda":
            self.head = nn.Sequential(
                nn.Linear(24, 64), nn.ELU(inplace=True),
                nn.Dropout(dropout), nn.Linear(64, num_classes),
            )
        else:  # late：两分支各带独立分类头，输出概率平均
            self.ppg_head = nn.Linear(64, num_classes)
            self.eda_head = nn.Linear(24, num_classes)
            self.head = None

    # ── 参数量 ──────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── 前向 ────────────────────────────────
    def forward(self, ppg: torch.Tensor | None, eda: torch.Tensor | None) -> torch.Tensor:
        if self.variant == "late":
            probs = 0.0
            if self.use_ppg:
                probs = probs + torch.softmax(self.ppg_head(self.ppg(ppg)), dim=1)
            if self.use_eda:
                probs = probs + torch.softmax(self.eda_head(self.eda(eda)), dim=1)
            n = int(self.use_ppg) + int(self.use_eda)
            return torch.log(probs / n + 1e-8)  # 返回 log-prob，可与 NLLLoss 配合

        feats = []
        if self.use_ppg:
            feats.append(self.ppg(ppg))
        if self.use_eda:
            feats.append(self.eda(eda))
        h = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        return self.head(h)  # logits


def build_model(variant: Variant = "dual") -> WESADClassifier:
    """工厂函数，方便训练脚本调用。"""
    return WESADClassifier(num_classes=3, variant=variant)
