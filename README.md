# TinyAD2 — VisualAD with TinyCLIP-ViT-61M/32

> 跨数据集异常检测框架，默认骨干已切换为 **TinyCLIP-ViT-61M/32**（61M 参数，patch=32，224px），显存占用仅为 ViT-L/14@336px 的约 30%。

---

## 骨干模型对比

| 骨干 | 参数量 | 图像尺寸 | embed_dim | 层数 | 特征层索引 |
|---|---|---|---|---|---|
| **TinyCLIP-ViT-61M/32**（默认） | 61M | 224 | 512 | 12 | [3, 6, 9, 12] |
| ViT-L/14@336px（原始） | 427M | 336 | 1024 | 24 | [6, 12, 18, 24] |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用默认 TinyCLIP-ViT-61M/32 骨干训练

权重会在首次运行时自动下载（~300MB）：

```bash
python train.py \
  --train_data_path /path/to/mvtec \
  --save_path ./checkpoints \
  --train_dataset mvtec \
  --epoch 15 \
  --batch_size 16 \
  --device cuda:0
```

### 3. 测试

```bash
python test.py \
  --test_data_path /path/to/visa \
  --checkpoint_path ./checkpoints/epoch_15.pth \
  --test_dataset visa \
  --save_path ./results \
  --device cuda:0
```

### 4. 使用脚本运行完整跨数据集实验

```bash
bash scripts/TinyCLIP.sh
```

---

## 配置说明

### backbone_settings.yaml

骨干配置位于 `configs/backbone_settings.yaml`，包含各骨干的默认参数：

```yaml
TinyCLIP-ViT-61M/32:
  weights: https://github.com/microsoft/TinyCLIP/releases/download/v1.0/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt
  image_size: 224
  embed_dim: 512
  transformer_layers: 12
  layers: [3, 6, 9, 12]
  drop_text_encoder: true
```

### CLI 参数覆盖

所有关键参数均可通过命令行覆盖：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--backbone` | 骨干名称或检查点路径 | `TinyCLIP-ViT-61M/32` |
| `--backbone_weights` | 覆盖权重路径/URL | 从 YAML 读取 |
| `--image_size` | 输入分辨率 | 从 YAML 读取（224） |
| `--embed_dim` | 嵌入维度 | 从 YAML 读取（512） |
| `--features_list` | 特征提取层索引 | 从 YAML 读取 |
| `--transformer_layers` | Transformer 层数 | 从 YAML 读取 |
| `--drop_text_encoder` | 丢弃文本编码器节省显存 | TinyCLIP 自动开启 |

---

## 切换到其他骨干

### 切换到 ViT-L/14@336px（原始高精度骨干）

```bash
python train.py \
  --backbone "ViT-L/14@336px" \
  --train_data_path /path/to/mvtec \
  --save_path ./checkpoints_vitl \
  --train_dataset mvtec \
  --batch_size 8 \
  --device cuda:0
```

```bash
bash scripts/CLIP.sh
```

### 使用本地权重文件

```bash
python train.py \
  --backbone "TinyCLIP-ViT-61M/32" \
  --backbone_weights /path/to/TinyCLIP-ViT-61M-32.pt \
  --train_data_path /path/to/mvtec \
  --save_path ./checkpoints \
  --train_dataset mvtec
```

---

## 解冻微调策略

解冻最后 N 个 Transformer block，配合小学习率进行骨干微调：

```bash
python train.py \
  --backbone "TinyCLIP-ViT-61M/32" \
  --unfreeze_encoder_layers 2 \
  --encoder_learning_rate 1e-5 \
  --train_data_path /path/to/mvtec \
  --save_path ./checkpoints \
  --train_dataset mvtec
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--unfreeze_encoder_layers` | 解冻最后 N 个 block（0=全冻结） | `0` |
| `--encoder_learning_rate` | 解冻 block 的学习率 | `1e-5` |

---

## 多尺度层索引自动适配

框架会根据骨干实际层数自动生成合理的特征层索引：

- **12 层骨干**（TinyCLIP-ViT-61M/32 等）→ `[3, 6, 9, 12]`  
- **24 层骨干**（ViT-L/14@336px 等）→ `[6, 12, 18, 24]`

也可手动指定：

```bash
python train.py --features_list 2 4 8 12 ...
```

---

## Dry-run 单元测试

运行最小前向/loss 形状校验：

```bash
python -m unittest discover -s tests
```

---

## 注意事项

- `utils/feature_transform.py` 中的 `create_feature_transform` 需要显式传入 `input_dim`（使用 `model.visual.embed_dim`），不再有默认值。
- TinyCLIP 骨干不带 `weights: null` 时需通过 `--backbone_weights` 指定权重路径。
- 使用 TinyCLIP-ViT-61M/32 时文本编码器会自动丢弃以节省显存；若需保留请传入 `--no-drop_text_encoder`。
