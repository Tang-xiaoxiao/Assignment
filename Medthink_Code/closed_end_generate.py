# ============================================================
# 功能说明：
# 本脚本用于对已经训练完成的多模态 T5 模型进行推理评估（inference），
# 根据输入的文本数据和图像特征生成对应的答案、解释或推理结果，
# 并将生成内容保存为 JSON 文件。
#
# 主要流程：
# 1. 设置随机种子，保证推理结果尽可能可复现；
# 2. 从指定路径加载训练好的 T5ForMultimodalGeneration 模型；
# 3. 加载对应的 tokenizer 和 DataCollatorForSeq2Seq；
# 4. 构建 Seq2SeqTrainer，用于批量执行生成式预测；
# 5. 加载 ClosedMedVQADataset 数据集，将文本输入与图像特征对齐；
# 6. 调用 trainer.predict() 对整个数据集进行生成预测；
# 7. 将模型输出的 token 序列解码为文本；
# 8. 按题目 ID 组织预测结果，并保存为 JSON 文件。
#
# 支持任务类型：
# - Explanation              : 生成解释
# - Reasoning                : 生成推理结果
# - First-Stage_Reasoning    : 第一阶段推理生成
# - Second-Stage_Reasoning   : 第二阶段推理生成
# - without_R                : 不使用推理模块的生成方式
#
# 输入参数说明：
# --text_file_path : 待预测文本数据路径，通常为 JSON 文件
# --img_file_path  : 图像特征文件路径（通常为 .pth）
# --img_name_map   : 图像名到特征索引的映射文件路径（通常为 .json）
# --model_path     : 已训练模型所在路径
# --output_dir     : 预测结果输出目录
# --source_len     : 输入文本最大长度
# --target_len     : 生成文本最大长度
# --eval_bs        : 推理时的 batch size
# --seed           : 随机种子
# --dataset        : 数据集名称，可选 rad 或 slake
# --method         : 当前推理任务类型
#
# 输出结果说明：
# 1. 当 method == "First-Stage_Reasoning" 时：
#    - 会读取原始输入 JSON；
#    - 将生成结果写入对应题目的 "solution" 字段；
#    - 若输入文件名中包含 "test"，则保存为 output_dir/method/test.json；
#    - 否则保存为 output_dir/method/train.json。
#
# 2. 当 method 为其他任务类型时：
#    - 直接将每道题的预测文本保存为字典；
#    - 字典格式为 {"question_题号": "生成结果"}；
#    - 保存到 output_dir/method/test_response.json。
#
#
# ============================================================
import os
from dataset import ClosedMedVQADataset
from model import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import argparse
import torch
import json

def eval_loop(_args):
    torch.manual_seed(_args.seed)  # pytorch random seed
    np.random.seed(_args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    model = T5ForMultimodalGeneration.from_pretrained(_args.model_path, (100, 256))
    tokenizer = AutoTokenizer.from_pretrained(_args.model_path)
    datacollator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    config = Seq2SeqTrainingArguments(
        output_dir="./",
        per_device_eval_batch_size=_args.eval_bs,
        predict_with_generate=True,
        generation_max_length=_args.target_len,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        data_collator=datacollator
    )

    data_set = ClosedMedVQADataset(
        _tokenizer=tokenizer,
        _text_file_path=_args.text_file_path,
        _img_file_path=_args.img_file_path,
        _img_name_map=_args.img_name_map,
        _method=_args.method,
        _source_len=_args.source_len,
        _target_len=_args.target_len,
        _dataset=_args.dataset
    )

    predictions = trainer.predict(test_dataset=data_set, max_length=256)
    preds, targets = predictions.predictions, predictions.label_ids

    # Replace -100 in the Preds/Targets as we can't decode them.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds_text = [test_pred_text.strip() for test_pred_text in preds_text]

    problem_ids = data_set.problem_id
    questions_dict = {f"question_{problem_id}": preds_text[index] for index, problem_id in enumerate(problem_ids)}
    if _args.method == "First-Stage_Reasoning":
        with open(_args.text_file_path, "r", encoding="utf-8") as RawFile:
            raw_data = json.load(RawFile)

        for key, value in questions_dict.items():
            question_number = key.split('_')[1]
            if question_number in raw_data:
                raw_data[question_number]['solution']=value
            else:
                raise ValueError
        if "test" in _args.text_file_path:
            save_path = os.path.join(_args.output_dir, _args.method, "test.json")
        else:
            save_path = os.path.join(_args.output_dir, _args.method, "train.json")
        with open(save_path, "w", encoding="utf-8") as OutputFile:
            json.dump(raw_data, OutputFile, ensure_ascii=False, indent=4)
    else:
        save_path = os.path.join(_args.output_dir, _args.method, "test_response.json")
        with open(save_path, "w", encoding="utf-8") as OutputFile:
            json.dump(questions_dict, OutputFile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file_path', type=str, default='None')
    parser.add_argument('--img_file_path', type=str, default='None')
    parser.add_argument('--img_name_map', type=str, default='None')
    parser.add_argument('--model_path', type=str, default='None')
    parser.add_argument('--output_dir', type=str, default='None')
    parser.add_argument('--source_len', type=int, default=512)
    parser.add_argument('--target_len', type=int, default=256)
    parser.add_argument('--eval_bs', type=int, default=8, help='Evaluation Batch Size')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--dataset', type=str, choices=['rad', 'slake'])
    parser.add_argument('--method', type=str, choices=["Explanation", "Reasoning", "First-Stage_Reasoning", "Second-Stage_Reasoning", "without_R"])
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    eval_loop(args)
