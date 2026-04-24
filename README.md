## tinyad2 操作指南（README）

本项目包含两条主要流程：

1. **VisualAD 主训练/测试流程**（`train.py` + `test.py`）
2. **VMamba 学生蒸馏与零样本评估流程**（`train_distill.py` + `eval_zero_shot.py`）

> 关键要求：教师模型必须是 **VisualAD 框架的 ViT-L/14@336px**（包含两个可学习 token 与相关机制），当前代码已按此要求接入。

---

## 1. 环境准备

### 1.1 创建环境（示例）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.2 依赖说明

- 核心依赖：`torch`, `torchvision`, `timm`, `scikit-learn`, `scipy`
- 蒸馏相关：`mamba-ssm`, `causal-conv1d`
- 可选日志：`wandb`

---

## 2. 数据准备

项目支持的数据组织方式由 `dataset.py` 读取（如 `mvtec` / `visa`）。  
常见做法是保证训练与测试路径分别指向对应数据集根目录。

示例变量（按实际路径修改）：

```bash
export MVTEC_ROOT="/path/to/mvtec"
export VISA_ROOT="/path/to/visa"
```

---

## 3. VisualAD 主流程

### 3.1 训练（train.py）

```bash
python train.py \
  --train_data_path "$MVTEC_ROOT" \
  --train_dataset mvtec \
  --save_path ./checkpoints/mvtec_visualad \
  --backbone "ViT-L/14@336px" \
  --epoch 15 \
  --batch_size 8 \
  --device cuda:0
```

常用参数：

- `--features_list`：特征层（默认 `[6,12,18,24]`）
- `--image_size`：输入分辨率（默认 `518`）
- `--save_freq` / `--print_freq`：保存与日志频率

### 3.2 测试（test.py）

```bash
python test.py \
  --test_data_path "$MVTEC_ROOT" \
  --test_dataset mvtec \
  --checkpoint_path ./checkpoints/mvtec_visualad/final_model.pth \
  --save_path ./results/mvtec_visualad \
  --device cuda:0
```

可选：

- `--enable_analysis`：输出额外可视化与分析结果
- `--sigma`：高斯平滑参数（默认 `4`）

---

## 4. VMamba 蒸馏流程

### 4.1 训练蒸馏（train_distill.py）

```bash
python train_distill.py \
  --train_root /path/to/unlabeled_images \
  --save_dir ./checkpoints_distill \
  --epochs 10 \
  --batch_size 8 \
  --teacher_layers 8 16 24 \
  --pairings "3+4:16,4:24" \
  --device cuda
```

说明：

- 教师默认即为 **VisualADTeacher(ViT-L/14@336px)**。
- `--pairings` 用于指定学生阶段与教师层的对齐关系。

### 4.2 零样本评估（eval_zero_shot.py）

```bash
python eval_zero_shot.py \
  --test_data_path "$MVTEC_ROOT" \
  --test_dataset mvtec \
  --checkpoint_path ./checkpoints_distill/epoch_10.pth \
  --prompt_path /path/to/prompt_embeddings.pt \
  --image_size 512 \
  --fusion_stages 3 4 \
  --fusion_weights 0.5 0.5 \
  --device cuda
```

`prompt_embeddings.pt` 支持两种格式：

- `{"normal": tensor, "anomaly": tensor}`
- `{"embeddings": tensor}`（前两行为 normal / anomaly）

---

## 5. 脚本化运行

仓库提供示例脚本：`scripts/CLIP.sh`  
执行前先修改脚本中的数据路径与设备号。

```bash
bash scripts/CLIP.sh
```

---

## 6. 常见问题

1. **教师模型不符要求**  
   请确认蒸馏入口使用的是 `VisualADTeacher`，而非外部 OpenCLIP 教师实现。

2. **torch 缺失导致测试报错**  
   先完成依赖安装：`pip install -r requirements.txt`。

3. **显存不足**  
   减小 `--batch_size`，必要时降低 `--image_size`。

4. **数据路径错误**  
   优先检查 `--train_data_path / --test_data_path / --train_root` 是否指向真实目录。
