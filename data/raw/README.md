# WESAD 原始数据目录

请将从 WESAD 官方渠道获取的 15 个被试文件放入此目录。

## 期望的文件结构

```
data/raw/
├── S2.pkl
├── S3.pkl
├── S4.pkl
├── S5.pkl
├── S6.pkl
├── S7.pkl
├── S8.pkl
├── S9.pkl
├── S10.pkl
├── S11.pkl
├── S12.pkl
├── S13.pkl
├── S14.pkl
├── S15.pkl
├── S16.pkl
└── S17.pkl
```

> 注意：WESAD 原始数据集没有 S1（编号从 S2 起），共 15 个被试。

## 单个 .pkl 文件结构（参考）

每个 `S{X}.pkl` 是一个 Python pickle，反序列化后为 dict，关键字段：

```python
{
  'subject': 'S2',
  'signal': {
    'chest': {...},          # 胸带 RespiBAN：ECG/EDA/EMG/RESP/TEMP/ACC（700Hz）
    'wrist': {
      'BVP':  array (n,1),   # PPG 信号，64 Hz
      'EDA':  array (n,1),   # 皮肤电活动，4 Hz
      'TEMP': array (n,1),   # 4 Hz
      'ACC':  array (n,3),   # 32 Hz
    },
  },
  'label': array (n,1),      # 与 chest 同长（700Hz），状态标签
}
```

标签含义：`0=undefined / 1=baseline / 2=stress / 3=amusement / 4=meditation / 5/6/7=ignore`。

## 获取方式

WESAD 官方页面（需签署 EULA）：https://ubicomp.eti.uni-bamberg.de/datasets/sneha/wesad/

本项目不重新分发原始数据，请自行下载并遵守 WESAD 原始 EULA。
