# VisualAD 项目说明

本仓库包含两部分：
1. **VisualAD 原始训练/测试流程**（`train.py`、`test.py`、`VisualAD_lib/`）
2. **Phase 1 timm student 启动流程**（`train_timm_student.py`、`student/`、`scripts/run_timm_*.sh`）

> 约束说明：开发过程中尽量不改动 VisualAD 原有文件，尤其是模型文件（`VisualAD_lib/*`）。

---

## 1. 环境安装

```bash
cd <PROJECT_ROOT>
pip install -r requirements.txt
```

---

## 2. 数据准备

项目使用与 AnomalyCLIP 一致的数据组织方式，需要在数据根目录下准备 `meta.json`。

可使用仓库内脚本生成数据集 JSON（示例）：

```bash
python generate_dataset_json/mvtec.py
python generate_dataset_json/visa.py
```

---

## 3. VisualAD 原始流程使用方法

### 3.1 一键跨数据集训练+测试

```bash
bash scripts/CLIP.sh
```

运行前请先修改脚本内数据路径：
- `MVTEC_PATH`
- `VISA_PATH`

### 3.2 单独训练

```bash
python train.py \
  --train_data_path <YOUR_DATASET_ROOT> \
  --save_path ./checkpoints \
  --train_dataset mvtec \
  --backbone "ViT-L/14@336px" \
  --epoch 15 \
  --batch_size 8 \
  --device cuda:0
```

### 3.3 单独测试

```bash
python test.py \
  --test_data_path <YOUR_DATASET_ROOT> \
  --checkpoint_path ./checkpoints/final_model.pth \
  --test_dataset mvtec \
  --save_path ./test_results \
  --device cuda:0
```

---

## 4. Phase 1 (timm student) 正式蒸馏训练使用方法

### 4.1 配置方式启动（正式训练）

```bash
python train_timm_student.py \
  --config configs/default.yaml
```

默认会执行 teacher-student feature distillation，并启用可插拔模块：
- 异常合成模块（`anomaly_synthesizer`）：`none` / `cutpaste` / `perlin`
- 损失调整模块（`loss_adjuster`）：`fixed` / `warmup`

### 4.2 数据集脚本启动

```bash
bash scripts/run_timm_mvtec.sh
bash scripts/run_timm_visa.sh
```

运行前需要替换占位路径：
- `"<REPLACE_WITH_YOUR_MVTEC_PATH>"`
- `"<REPLACE_WITH_YOUR_VISA_PATH>"`

脚本已内置约束：
- `fasternet_t0` 固定使用 `feature_out_indices=[0,1,2,3]`
- 其他学生骨干默认使用 `feature_out_indices=[1,2,3,4]`

### 4.3 消融实验常用参数示例

```bash
# 使用 CutPaste + 固定权重
python train_timm_student.py \
  --config configs/default.yaml \
  --anomaly_synthesizer cutpaste \
  --loss_adjuster fixed

# 使用 Perlin Noise + warmup 权重
python train_timm_student.py \
  --config configs/default.yaml \
  --anomaly_synthesizer perlin \
  --loss_adjuster warmup
```

---

## 5. requirements 兼容性检查结论

已对 **VisualAD 现有依赖** 与 **新增 timm student 流程依赖** 做合并校验，结论如下：

1. 主干深度学习栈兼容：
   - `torch==2.0.0`
   - `torchvision==0.15.1`
   - `timm==0.6.13`
2. 数值与评估栈兼容：
   - `numpy==1.24.4` 与 `scipy==1.9.1`、`scikit-learn==1.2.2`、`scikit-image==0.20.0` 可配套使用
3. 已补齐代码中实际使用但原文件缺失的依赖：
   - `pillow`、`pandas`、`matplotlib`、`opencv-python`、`tabulate`、`ftfy`、`regex`
4. 保留了原有依赖项：
   - `seaborn`、`tqdm`、`PyYAML`、`torchsummary`、`dash-table`

统一依赖清单见：
- `requirements.txt`

---

## 6. 目录重点

- VisualAD 原模型实现：`VisualAD_lib/`
- 原始训练测试入口：`train.py`、`test.py`
- Phase 1 入口：`train_timm_student.py`
- 统一依赖：`requirements.txt`
