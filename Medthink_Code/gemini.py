"""
脚本说明：
本脚本用于调用 Google Gemini 多模态模型完成医学 VQA（视觉问答）二分类任务推理，并将结果增量保存为 JSON 文件。

主要功能：
1. 读取指定数据集中的问题文件（JSON 格式）；
2. 根据数据集类型定位对应图片；
3. 构造提示词，将“图像 + 问题 + 可选 solution + 固定选项 yes/no”一起输入 Gemini；
4. 从模型输出中提取答案：
   - (A) -> yes
   - (B) -> no
5. 将结果以如下格式保存到输出文件中：
   {
       "question_1": "The answer is (A).",
       "question_2": "The answer is (B)."
   }
6. 支持断点续跑：
   - 如果 output_path 已存在，则自动跳过已完成的问题；
7. 支持多 API Key 随机切换，以缓解单 Key 限流问题；
8. 当请求失败时自动等待并重试。

参数说明：
--solution
    是否在提示词中加入数据中的 solution 字段；
--file_path
    问题 JSON 文件路径；
--image_dir
    图片根目录；
--output_path
    输出文件名，实际会保存在 file_path 同级目录下；
--dataset_type
    数据集类型

注意事项：
1. 模型输出必须包含形如 (A) 或 (B) 的答案标记，否则会记为 FAILED；
2. 当前使用的是 gemini-pro-vision 接口，若 API 版本变动，可能需要同步修改调用方式；
3. 请将 gemini_api_key 中替换为你自己的有效 API Key；
4. 该脚本默认每次请求单张图片和单条问题，不做并发处理。
"""
import google.generativeai as genai
import json
from PIL import Image
import os
import re
import time
import random
import argparse
import base64
import requests


gemini_api_key = [
    '***************************************',  # KEY_1
    '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',  # KEY_1
]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_ans(_ans):
    pattern = re.compile(r'\(([A-B])\)')
    res = pattern.findall(_ans)

    if len(res) == 1:
        _answer = res[0]  # 'A', 'B', ...
    else:
        _answer = "FAILED"
    return _answer

def load_existing_results(_file_path):
    if os.path.exists(_file_path):
        with open(_file_path, 'r', encoding='utf-8') as File:
            return json.load(File)
    else:
        return None

def genimi_to_answer(_solution, _file_path, _image_dir, _output_path, _dataset_type):
    def get_headers():
        return random.randint(0, len(gemini_api_key)-1)

    existing_results = load_existing_results(_output_path)
    if existing_results:
        result = existing_results
    else:
        result = {}
    with open(_file_path, 'r', encoding='utf-8') as File:
        data = json.load(File)
    print(f"There are {len(data)-len(result)} questions.\n")
    instruction = "Let's think step-by-step, and please answer the following VQA question:\n"
    for num, item in enumerate(data):
        if existing_results and f"question_{item}" in existing_results:
            continue
        question = data[item]['question']
        choices = f"(A) yes (B) no"

        context = f"Question: {question}\n"
        if _solution:
            context = context + f"Solution: {data[item]['solution']}\n"
        context = context + f"Options: {choices}\nAnswer:"

        if _dataset_type == 'rad':
            image_path = os.path.join(_image_dir, data[item]['image'])
        elif _dataset_type == 'slake':
            image_path = os.path.join(_image_dir, data[item]['img_name'])
        else:
            raise ValueError
        print(f"Image: {image_path}\n{context}")
        image = Image.open(image_path).convert('RGB')
        while True:
            try:
                genai.configure(api_key=gemini_api_key[get_headers()], transport='rest')
                model = genai.GenerativeModel('gemini-pro-vision')
                response = model.generate_content(
                    [image, instruction+context], stream=False,
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                         },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
                )
                response.resolve()
                print(f"Output: {response.parts[0].text}")
                answer = extract_ans(response.parts[0].text)
                result[f"question_{item}"] = f"The answer is ({answer})."
                with open(_output_path, 'w', encoding='utf-8') as OutFile:
                    json.dump(result, OutFile, indent=4, ensure_ascii=False)
                print(f"{num}: question_{item}: {answer}.\n")
                break

            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                time.sleep(30)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution', action='store_true', default=False)
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--dataset_type', type=str)
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    genimi_to_answer(
        _solution=args.solution,
        _file_path=args.file_path,
        _image_dir=args.image_dir,
        _output_path=os.path.join(os.path.dirname(args.file_path), args.output_path),
        _dataset_type=args.dataset_type
    )
