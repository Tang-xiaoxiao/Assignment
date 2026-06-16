# -*- coding: utf-8 -*-
"""
本脚本用于：
1. 评估二分类 VQA 预测结果；
2. 绘制必要评估图，包括：
   - Accuracy / Precision / Recall / F1 柱状图
   - 混淆矩阵图
3. 读取训练日志并绘制必要训练曲线，包括：
   - Loss 曲线
   - Learning Rate 曲线
   - Grad Norm 曲线（若存在）

"""

import json
import re
import os
import ast
import matplotlib.pyplot as plt
import numpy as np

# pred_path = "/mnt/workspace/workgroup/xt/Medthink-main/Medthink/Medthink_Code/ours_short_closed_end_experiments/without_R/test_response.json"
pred_path = "/mnt/workspace/workgroup/xt/Medthink-main/Medthink/Medthink_Code/ours_short_closed_end_experiments/Explanation/test_response.json"
# pred_path = "/mnt/workspace/workgroup/xt/Medthink-main/Medthink/Medthink_Code/ours_short_closed_end_experiments/Reasoning/test_response.json"
# pred_path = "/mnt/workspace/workgroup/xt/Medthink-main/Medthink/Medthink_Code/ours_short_closed_end_experiments/Two-Stage_Reasoning/test_response.json"
gt_path = "/mnt/workspace/workgroup/xt/Test_Set/Test_Set/test_vqa_formatted.json"

pred_parent_dir = os.path.dirname(pred_path)
train_log_path = os.path.join(pred_parent_dir, "last_output.log")

save_dir = os.path.join(pred_parent_dir, "pictures")
os.makedirs(save_dir, exist_ok=True)

with open(pred_path, "r", encoding="utf-8") as f:
    preds = json.load(f)

with open(gt_path, "r", encoding="utf-8") as f:
    gt = json.load(f)


def parse_pred(text):
    if not isinstance(text, str):
        return None
    text = text.strip()

    m = re.search(r'answer\s+is\s*\(([A-Za-z])\)', text, flags=re.I)
    if m:
        ch = m.group(1).upper()
        return 0 if ch == 'A' else 1 if ch == 'B' else None

    matches = re.findall(r'\(([A-Za-z])\)', text)
    if matches:
        ch = matches[-1].upper()
        return 0 if ch == 'A' else 1 if ch == 'B' else None

    return None


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def load_training_logs(log_path):
    """
    支持三种情况：
    1. JSON list
    2. 单个 dict
    3. 文本文件，每行一个 python dict
    """
    if log_path is None or not os.path.exists(log_path):
        return []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            return [data]
    except Exception:
        pass

    logs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
                if isinstance(obj, dict):
                    logs.append(obj)
            except Exception:
                continue
    return logs


def plot_curve(x, y, xlabel, ylabel, title, save_path):
    if len(x) == 0 or len(y) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linewidth=1.8, markersize=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 一、评估预测结果
# =========================
y_true = []
y_pred = []

invalid = 0
missing = 0

for qid, ans_text in preds.items():
    m = re.match(r'question_(\d+)$', qid)
    if not m:
        continue

    img_id = m.group(1)
    if img_id not in gt:
        missing += 1
        continue

    pred_label = parse_pred(ans_text)
    if pred_label is None:
        invalid += 1
        continue

    gt_label = gt[img_id]["answer"]
    y_true.append(gt_label)
    y_pred.append(pred_label)

tp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
fn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
fp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
tn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)

total = len(y_true)
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

accuracy = safe_div(correct, total)

precision_0 = safe_div(tp, tp + fp)
recall_0 = safe_div(tp, tp + fn)
f1_0 = safe_div(2 * precision_0 * recall_0, precision_0 + recall_0)

precision_1 = safe_div(tn, tn + fn)
recall_1 = safe_div(tn, tn + fp)
f1_1 = safe_div(2 * precision_1 * recall_1, precision_1 + recall_1)

precision = (precision_0 + precision_1) / 2
recall = (recall_0 + recall_1) / 2
f1 = (f1_0 + f1_1) / 2

print("===== 必要指标 =====")
print(f"total: {total}")
print(f"correct: {correct}")
print(f"accuracy: {accuracy:.6f}")
print(f"precision: {precision:.6f}")
print(f"recall: {recall:.6f}")
print(f"f1: {f1:.6f}")
print(f"invalid_pred: {invalid}")
print(f"missing_gt: {missing}")

metrics_dict = {
    "total": total,
    "correct": correct,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "invalid_pred": invalid,
    "missing_gt": missing,
    "confusion_matrix": {
        "true_0_pred_0": tp,
        "true_0_pred_1": fn,
        "true_1_pred_0": fp,
        "true_1_pred_1": tn
    }
}

with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

metrics = ["Accuracy", "Precision", "Recall", "F1"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Evaluation Metrics")

for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}",
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
metrics_fig_path = os.path.join(save_dir, "metrics_bar.png")
plt.savefig(metrics_fig_path, dpi=300, bbox_inches="tight")
plt.close()

cm = np.array([[tp, fn],
               [fp, tn]])

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
plt.yticks(tick_marks, ["True 0", "True 1"])

thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
cm_fig_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 二、绘制训练日志图
# =========================
train_logs = load_training_logs(train_log_path)

if len(train_logs) > 0:
    if all(("epoch" in item and isinstance(item["epoch"], (int, float))) for item in train_logs):
        x = [item["epoch"] for item in train_logs]
        xlabel = "Epoch"
    else:
        x = list(range(1, len(train_logs) + 1))
        xlabel = "Step"

    losses = [item["loss"] for item in train_logs if "loss" in item and isinstance(item["loss"], (int, float))]
    loss_x = [x[i] for i, item in enumerate(train_logs) if "loss" in item and isinstance(item["loss"], (int, float))]

    lrs = [item["learning_rate"] for item in train_logs if "learning_rate" in item and isinstance(item["learning_rate"], (int, float))]
    lr_x = [x[i] for i, item in enumerate(train_logs) if "learning_rate" in item and isinstance(item["learning_rate"], (int, float))]

    grad_norms = [item["grad_norm"] for item in train_logs if "grad_norm" in item and isinstance(item["grad_norm"], (int, float))]
    grad_x = [x[i] for i, item in enumerate(train_logs) if "grad_norm" in item and isinstance(item["grad_norm"], (int, float))]

    if len(losses) > 0:
        plot_curve(
            loss_x, losses, xlabel, "Loss", "Training Loss Curve",
            os.path.join(save_dir, "train_loss_curve.png")
        )

    if len(lrs) > 0:
        plot_curve(
            lr_x, lrs, xlabel, "Learning Rate", "Learning Rate Curve",
            os.path.join(save_dir, "learning_rate_curve.png")
        )

    if len(grad_norms) > 0:
        plot_curve(
            grad_x, grad_norms, xlabel, "Grad Norm", "Gradient Norm Curve",
            os.path.join(save_dir, "grad_norm_curve.png")
        )

    print("\n===== 训练日志图已保存 =====")
    if len(losses) > 0:
        print(os.path.join(save_dir, "train_loss_curve.png"))
    if len(lrs) > 0:
        print(os.path.join(save_dir, "learning_rate_curve.png"))
    if len(grad_norms) > 0:
        print(os.path.join(save_dir, "grad_norm_curve.png"))
else:
    print(f"\n未找到或无法解析训练日志：{train_log_path}")

print("\n===== 评估图已保存 =====")
print(metrics_fig_path)
print(cm_fig_path)
print(os.path.join(save_dir, "metrics.json"))
