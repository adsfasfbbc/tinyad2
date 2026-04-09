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

---

## 7. 统一测试评估与论文表格产出（MVTec + VisA）

### 7.1 先导出学生 ONNX（可选）

```bash
bash scripts/export_students_onnx.sh
```

或按单模型导出：

```bash
python export_timm_student_onnx.py \
  --checkpoint_path <STUDENT_CHECKPOINT_PTH> \
  --student_backbone mobilenetv3_small_100 \
  --output_path ./experiments/onnx_exports/mobilenetv3_small_100.onnx
```

### 7.2 配置统一评测

编辑：
- `configs/eval_benchmark.yaml`

需要填写：
- MVTec/VisA 数据集路径
- 教师模型（VisualAD）checkpoint 路径
- 5 个学生模型 checkpoint 路径
- 可选 ONNX 路径（用于同时展示 pth/onnx 参数量）

### 7.3 一键评测并产出核心对比表

```bash
bash scripts/run_unified_eval.sh
```

输出目录（默认）：
- `experiments/eval_benchmark/core_table.md`
- `experiments/eval_benchmark/core_table.csv`
- `experiments/eval_benchmark/per_dataset_metrics.csv`

核心表格列为：

`Model | Params (M) | FLOPs (G) | FPS | Image-AUROC | Pixel-AUROC | MAP | F1-SCORE`

说明：
- `Image-AUROC / Pixel-AUROC`：跨 MVTec 与 VisA 的均值
- `MAP / F1-SCORE`：默认使用 **Image-level** mAP / F1
- `per_dataset_metrics.csv` 同时保留每个数据集的 image/pixel 明细指标

---

## 8. ViT→MobileViT 异构蒸馏（架构升级版）

新增训练入口：
- `train_vit_mobilevit_distill.py`

新增模块：
- `student/mobilevit_token_student.py`：MobileViT Token 提取 + MLP Projector 对齐
- `utils/vit_teacher_adapter.py`：VisualAD ViT Teacher 冻结适配
- `distill/losses.py`：
  - `TokenContrastiveLoss`（正常拉近、异常排斥）
  - `AttentionMimicryLoss`（Attention Proxy KL 对齐）
  - `CLSTokenAlignmentLoss`（全局语义对齐）
- `utils/anomaly_generator.py`：Perlin/CutPaste 在线异常生成

配置与脚本：
- `configs/vit_mobilevit_distill.yaml`
- `scripts/run_vit_mobilevit_distill.sh`

运行：

```bash
bash scripts/run_vit_mobilevit_distill.sh
```

说明：
- `Dataset.__getitem__` 在训练模式下支持在线异常合成，返回增强图像、mask 与标签。
- 教师为 VisualAD ViT（冻结），学生为 MobileViT（可训练），通过 MLP 将学生 token 映射到教师 token 维度后执行三重蒸馏损失。
