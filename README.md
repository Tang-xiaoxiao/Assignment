```markdown
# Medthink_Code

基于 **T5 多模态生成模型** 的医学视觉问答（Medical VQA）项目，支持图像特征提取、闭集问答训练、推理生成，以及多种解释/推理任务设置。

---

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [数据格式](#数据格式)
- [核心脚本说明](#核心脚本说明)
- [图像特征提取](#图像特征提取)
- [模型训练](#模型训练)
- [模型推理](#模型推理)
- [任务模式说明](#任务模式说明)
- [输出结果说明](#输出结果说明)
- [典型运行流程](#典型运行流程)

---

## 项目简介

本项目面向 **闭集医学视觉问答（Closed-ended Medical VQA）** 场景，将：

- 医学图像特征
- 问题文本
- 候选答案
- 推理过程 / 解释文本

联合输入扩展后的 **T5ForMultimodalGeneration** 模型中，实现多种生成任务，包括：

- **Explanation**：生成答案与解释
- **Reasoning**：生成推理过程与答案
- **First-Stage_Reasoning**：第一阶段，仅生成推理过程
- **Second-Stage_Reasoning**：第二阶段，基于推理过程生成答案
- **without_R**：不使用推理，直接生成答案

项目整体流程包括：

1. 提取图像特征  
2. 构建多模态输入数据  
3. 训练多模态 T5 模型  
4. 使用训练好的模型进行生成与推理  

---

## 项目结构

根据当前代码目录，项目结构如下：

```bash
Medthink_Code/
├── ours_short_closed_end_experiments/   # 实验输出目录（名称可能被截断）
│   ├── Explanation/                 # 对应任务的模型权重及生成结果
│   ├── First-Stage_Reasoning/       # 对应任务的模型权重及生成结果
│   ├── Reasoning/                   # 对应任务的模型权重及生成结果
│   └── without_R/                   # 对应任务的模型权重及生成结果
├── closed_end_generate.py           # 闭集任务生成/推理脚本
├── closed_end_train.py              # 闭集任务训练脚本
├── dataset.py                       # 数据集定义
├── experiments.sh                   # 批量实验脚本
├── extract_img_feature.py           # 图像特征提取脚本
├── extract_img_feature.sh           # 图像特征提取 shell 脚本
├── gemini.py                        # 其他辅助脚本（如外部生成/数据处理）
├── metric.py                        # 评价指标脚本
├── model.py                         # 多模态模型定义
└── README.md
```

---

## 环境依赖

建议环境：

- Python >= 3.8
- PyTorch >= 1.10
- Transformers 4.x
- CUDA（如需使用 GPU）

安装依赖：

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install evaluate
pip install nltk
pip install pillow
pip install timm
pip install numpy
```

若首次使用 `nltk.sent_tokenize`，请先执行：

```python
import nltk
nltk.download('punkt')
```

---

## 数据格式

项目运行依赖三类数据：

1. 文本问答数据（JSON）
2. 图像特征文件（`.pth`）
3. 图像索引映射文件（`name_map.json`）

### 1. 文本问答数据格式

闭集问答数据示例：

```json
{
  "1": {
    "question": "Does the image show clear pathological features indicating an underlying disease risk?",
    "choices": [
      "yes",
      "no"
    ],
    "answer": 0,
    "image": "1.png",
    "answer_type": "CLOSED",
    "image_organ": "",
    "solution": "The image shows clear pathological features, including intraretinal hemorrhages and hard exudates, which indicate an underlying disease process."
  },
  "2": {
    "question": "Does the image show clear pathological features indicating an underlying disease risk?",
    "choices": [
      "yes",
      "no"
    ],
    "answer": 0,
    "image": "2.png",
    "answer_type": "CLOSED",
    "image_organ": "",
    "solution": "The presence of numerous retinal abnormalities, including hemorrhages and hard exudates, indicates that the patient has a pathological eye condition."
  }
}
```

字段说明：

| 字段 | 说明 |
|------|------|
| `question` | 问题文本 |
| `choices` | 候选答案列表 |
| `answer` | 正确答案索引 |
| `solution` | 推理/解释文本 |
| `image` | 图像标识 |

---

### 2. 图像特征文件格式

通过 `extract_img_feature.py` 提取后生成：

```bash
detr.pth
```

通常形状类似：

```python
[num_images, num_visual_tokens, hidden_dim]
```

---

### 3. 图像索引映射文件格式

示例：

```json
{
    "img_0001": "0",
    "img_0002": "1",
    "img_0003": "2"
}
```

表示图像 ID 与特征张量行索引的对应关系。

---

## 核心脚本说明

### `extract_img_feature.py`
提取医学图像特征，输出 `.pth` 特征文件和 `name_map.json` 映射文件。

### `dataset.py`
定义数据集类，包括：

- `ClosedMedVQADataset`
- `OpenMedVQADataset`
- `ClosedInputAndTargetAndImg`
- `OpenInputAndTargetAndImg`

负责：

- 读取文本数据
- 读取图像特征
- 构造输入/输出文本
- 返回训练所需字段

### `model.py`
定义多模态模型 `T5ForMultimodalGeneration`，用于融合文本与图像特征。

### `closed_end_train.py`
训练闭集医学视觉问答模型。

### `closed_end_generate.py`
使用训练后的模型进行推理并保存结果。

### `metric.py`
用于计算评价指标。

### `experiments.sh`
批量执行实验脚本。

### `extract_img_feature.sh`
特征提取 shell 启动脚本。

### `gemini.py`
其他辅助脚本，具体作用可根据实际用途补充。

---

## 图像特征提取

### 功能说明

`extract_img_feature.py` 使用预训练 DETR 模型：

- 从 GitHub 加载 `cooelf/detr:main`
- 使用 `detr_resnet101_dc5`
- 对图片提取视觉特征
- 保存特征与图像索引映射

### 参数说明

| 参数 | 说明 |
|------|------|
| `--device` | 运行设备，如 `cuda:0` / `cpu` |
| `--image_dir` | 图像目录 |
| `--output_dir` | 输出目录 |
| `--dataset` | 数据集类型：`rad` / `slake` / `ours` |
| `--img_type` | 输出特征文件名 |

### 输出结果

```bash
features/
├── detr.pth
└── name_map.json
```

---

## 模型训练

### 功能说明

`closed_end_train.py` 负责：

- 设置随机种子
- 加载预训练多模态 T5
- 对新增多模态层进行稳定初始化
- 构建训练数据集
- 使用 `Seq2SeqTrainer` 进行训练
- 保存训练完成的模型


### 参数说明

| 参数 | 说明 |
|------|------|
| `--train_text_file_path` | 训练文本 JSON 路径 |
| `--img_file_path` | 图像特征 `.pth` 路径 |
| `--img_name_map` | 图像索引映射 JSON 路径 |
| `--pretrained_model_path` | 预训练模型路径 |
| `--output_dir` | 输出目录 |
| `--method` | 训练模式 |
| `--source_len` | 输入最大长度 |
| `--target_len` | 输出最大长度 |
| `--lr` | 学习率 |
| `--epoch` | 训练轮数 |
| `--bs` | Batch Size |
| `--wd` | Weight Decay |
| `--seed` | 随机种子 |
| `--dataset` | `rad` / `slake` |
| `--rational` | 是否使用 ROUGE-L 评价逻辑 |

---

## 模型推理

### 功能说明

`closed_end_generate.py` 负责：

- 加载训练完成的模型
- 构建测试集
- 执行批量生成
- 将结果保存为 JSON 文件

### 参数说明

| 参数 | 说明 |
|------|------|
| `--text_file_path` | 测试文本数据路径 |
| `--img_file_path` | 图像特征路径 |
| `--img_name_map` | 图像映射路径 |
| `--model_path` | 训练好模型的路径 |
| `--output_dir` | 输出目录 |
| `--source_len` | 输入最大长度 |
| `--target_len` | 生成最大长度 |
| `--eval_bs` | 推理 Batch Size |
| `--seed` | 随机种子 |
| `--dataset` | `rad` / `slake` |
| `--method` | 推理模式 |

---

## 任务模式说明

### 1. `Explanation`
输入问题和选项，输出答案与解释。

目标格式示例：

```text
The answer is (A).
Solution: The X-ray image shows a visible fracture line in the bone.
```

### 2. `Reasoning`
输入问题和选项，输出推理过程与答案。

目标格式示例：

```text
The X-ray image shows a visible fracture line in the bone.
Answer: The answer is (A).
```

### 3. `First-Stage_Reasoning`
仅生成推理过程 `solution`。

输入格式：

```text
Question: ...
Options: ...
Solution:
```

输出格式：

```text
The X-ray image shows a visible fracture line in the bone.
```

### 4. `Second-Stage_Reasoning`
给定推理过程，仅生成最终答案。

输入格式：

```text
Question: ...
Options: ...
Solution: ...
Answer:
```

输出格式：

```text
The answer is (A).
```

### 5. `without_R`
不使用推理过程，直接输出答案。

输入格式：

```text
Question: ...
Options: ...
Answer:
```

输出格式：

```text
The answer is (A).
```

---

## 输出结果说明

### 1. 图像特征提取输出

```bash
output_dir/
├── detr.pth
└── name_map.json
```

### 2. 模型训练输出

训练结果通常保存在：

```bash
output_dir/<method>/
```

例如：

```bash
outputs/
├── Explanation/
├── First-Stage_Reasoning/
├── Reasoning/
└── without_R/
```

每个目录中通常包含：

- 模型权重
- tokenizer 文件
- 配置文件

### 3. 模型推理输出

当 `method == "First-Stage_Reasoning"` 时：

- 结果会回写到原始 JSON 中的 `solution` 字段
- 保存为：
  - `output_dir/First-Stage_Reasoning/test.json`
  - 或 `output_dir/First-Stage_Reasoning/train.json`

当 `method` 为其他类型时：

```bash
output_dir/<method>/test_response.json
```

示例：

```json
{
    "question_1": "The answer is (A).",
    "question_2": "The answer is (B)."
}
```

---

## 典型运行流程

### 数据准备：利用 gemini 生成训练数据集

```bash
python gemini.py \
```

### 第一步：提取图像特征

```bash
python extract_img_feature.py \
```

### 第二步：训练推理模型

```bash
python closed_end_train.py \
```

### 第三步：生成推理结果

```bash
python closed_end_generate.py \
```

### 第四步：计算指标

```bash
python metric.py \
```

---
```