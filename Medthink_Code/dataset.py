# ============================================================
# 功能说明：
# 本脚本定义了医学视觉问答任务中使用的数据集类，主要负责：
# 1. 读取文本问答数据（JSON 格式）；
# 2. 读取预先提取好的图像特征（.pth 文件）；
# 3. 读取图像名称到特征索引的映射文件（.json）；
# 4. 根据不同训练/推理方法，构造模型输入文本（input）和目标输出文本（target）；
# 5. 将文本通过 tokenizer 编码，并与图像特征一起组织成模型可直接使用的数据格式。
#
# 本脚本包含两类数据集：
# ------------------------------------------------------------
# 1. ClosedMedVQADataset
#    用于闭集医学视觉问答任务（Closed-ended VQA）。
#    其特点是每道题给定有限选项（如 A / B），模型需要从候选答案中选择正确项，
#    并可进一步生成解释或推理过程。
#
# 2. OpenMedVQADataset
#    用于开放式医学视觉问答任务（Open-ended VQA）。
#    其特点是问题没有固定候选选项，模型直接生成答案文本，
#    同样支持解释、推理等多种训练模式。
#
# 每类数据集都配套一个辅助类：
# ------------------------------------------------------------
# - ClosedInputAndTargetAndImg
#   负责将闭集问答样本中的 question / choices / answer / solution / image 等字段
#   按不同 method 组织为输入文本、目标文本和图像 ID。
#
# - OpenInputAndTargetAndImg
#   负责将开放式问答样本中的 question / answer / solution / image 等字段
#   按不同 method 组织为输入文本、目标文本和图像 ID。
#
# 支持的任务模式（method）：
# ------------------------------------------------------------
# - Explanation
#   输入问题（及选项），目标输出为“答案 + 解释”。
#
# - Reasoning
#   输入问题（及选项），目标输出为“推理过程 + 最终答案”。
#
# - First-Stage_Reasoning
#   输入问题（及选项），目标只生成推理过程（solution）。
#
# - Second-Stage_Reasoning
#   输入问题（及选项）以及已知 solution，目标只生成最终答案。
#
# - without_R
#   不使用推理信息，仅直接生成最终答案。
#
# 数据读取与处理流程：
# ------------------------------------------------------------
# 1. 从 _text_file_path 读取 JSON 格式问答数据；
# 2. 从 _img_name_map 读取图像名到索引的映射字典；
# 3. 从 _img_file_path 加载图像特征张量；
# 4. 对每条样本：
#    - 生成 source_text（模型输入）
#    - 生成 target_text（监督目标）
#    - 获取图像对应的特征索引 img_index
#    - 保存 problem_id 便于后续结果对应原题号
# 5. 在 __getitem__ 中：
#    - 对文本进行 tokenizer 编码
#    - 获取对应图像特征
#    - 返回 input_ids / attention_mask / image_ids / labels
#
# 返回字段说明：
# ------------------------------------------------------------
# - input_ids      : 编码后的输入文本 token id
# - attention_mask : 输入文本的注意力掩码
# - image_ids      : 当前样本对应的图像特征向量/矩阵
# - labels         : 编码后的目标文本 token id
#
# 数据格式要求：
# ------------------------------------------------------------
# 文本 JSON 文件中的每条样本通常包含：
# - question : 问题文本
# - choices  : 候选答案列表（闭集任务）
# - answer   : 正确答案索引
# - solution : 解释或推理文本
# - image / img_id : 图像标识
#
# 图像映射文件 name_map.json 中：
# - key   : 图像 ID（通常是不带后缀的文件名）
# - value : 图像特征在 .pth 张量中的行索引
#
# ============================================================
import json
import torch
from torch.utils.data import Dataset

class ClosedMedVQADataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _method, _source_len, _target_len, _dataset):
        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        if torch.isnan(self.pretrained_feature).any():
            print("⚠️ 警告：图像特征中包含 NaN！")
        if torch.isinf(self.pretrained_feature).any():
            print("⚠️ 警告：图像特征中包含 Inf！")
        self.source_len = _source_len
        self.target_len = _target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)

        for problem in data:
            pair = ClosedInputAndTargetAndImg(data[problem])
            prompt = pair.get_input(method=_method)
            target = pair.get_target(method=_method)
            img = pair.get_img(dataset=_dataset)
            self.source_text.append(prompt)
            self.target_text.append(target)
            self.img_index.append(int(name_map[img]))
            self.problem_id.append(problem)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])
        img_index = self.img_index[item]

        # Normalize whitespace: remove extra spaces, tabs, and newlines.
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        image_ids = self.pretrained_feature[img_index].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids
        }

class ClosedInputAndTargetAndImg:
    def __init__(self, problem):
        self.problem = problem
        self.options = ['A', 'B']
        self.question_text = self.get_question_text()
        self.answer_text = self.get_answer()
        self.choice_text = self.get_choice_text()
        self.solution_text = self.get_solution_text()

    def get_choice_text(self):
        choices = self.problem['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(self.options[i], c))
        choice_txt = " ".join(choice_list)
        return choice_txt

    def get_question_text(self):
        return self.problem['question']

    def get_answer(self):
        return "(" + self.options[self.problem['answer']] + ")"

    def get_solution_text(self):
        return self.problem['solution']

    def get_target(self, method):
        if method == "Explanation":
            return f"The answer is {self.answer_text}.\nSolution: {self.problem['solution']}"
        elif method == "Reasoning":
            return f"{self.problem['solution']}\nAnswer: The answer is {self.answer_text}."
        elif method == "First-Stage_Reasoning":
            return f"{self.problem['solution']}"
        elif method == "Second-Stage_Reasoning":
            return f"The answer is {self.answer_text}."
        elif method == "without_R":
            return f"The answer is {self.answer_text}."
        else:
            raise ValueError("Please Check the Value of Method.")

    def get_input(self, method):
        if method == "Explanation":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nAnswer:"
        elif method == "Reasoning":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nSolution:"
        elif method == "First-Stage_Reasoning":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nSolution:"
        elif method == "Second-Stage_Reasoning":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nSolution: {self.solution_text}\nAnswer:"
        elif method == "without_R":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nAnswer:"
        else:
            raise ValueError("Please Check the Value of Method.")

    def get_img(self, dataset):
        if dataset == "rad":
            return self.problem['image'][:-4]
        elif dataset == "slake":
            return self.problem["img_id"]
        else:
            raise ValueError(f"Invalid _dataset value: {dataset}. The value must be 'rad' or 'slake'.")

class OpenMedVQADataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _method, _dataset, _source_len, _target_len):
        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        self.source_len = source_len
        self.target_len = target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)
        for problem in data:
            pair = OpenInputAndTargetAndImg(data[problem])
            prompt = pair.get_input(method=_method)
            target = pair.get_target(method=_method)
            img = pair.get_img(_dataset=_dataset)
            self.source_text.append(prompt)
            self.target_text.append(target)
            self.img_index.append(int(name_map[img]))
            self.problem_id.append(problem)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])
        img_index = self.img_index[item]

        # Normalize whitespace: remove extra spaces, tabs, and newlines.
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        image_ids = self.pretrained_feature[img_index].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids
        }

class OpenInputAndTargetAndImg:
    def __init__(self, problem):
        self.problem = problem
        self.question_text = self.get_question_text()
        self.answer_text = self.get_answer()
        self.solution_text = self.get_solution_text()

    def get_question_text(self):
        return self.problem['question']

    def get_answer(self):
        return self.problem['choices'][0]

    def get_solution_text(self):
        return self.problem['solution']

    def get_target(self, method):
        if method == "Explanation":
            return f"The answer is {self.answer_text}.\nSolution: {self.problem['solution']}"
        elif method == "Reasoning":
            return f"{self.problem['solution']}\nAnswer: The answer is {self.answer_text}."
        elif method == "First-Stage_Reasoning":
            return f"{self.problem['solution']}"
        elif method == "Second-Stage_Reasoning":
            return f"The answer is {self.answer_text}."
        elif method == "without_R":
            return f"The answer is {self.answer_text}."
        else:
            raise ValueError("Please Check the Value of Method.")

    def get_input(self, method):
        if method == "Explanation":
            return f"Question: {self.question_text}\nAnswer:"
        elif method == "Reasoning":
            return f"Question: {self.question_text}\nSolution:"
        elif method == "First-Stage_Reasoning":
            return f"Question: {self.question_text}\nSolution:"
        elif method == "Second-Stage_Reasoning":
            return f"Question: {self.question_text}\nSolution: {self.solution_text}\nAnswer:"
        else:
            raise ValueError("Please Check the Value of Method.")

    def get_img(self, _dataset):
        if _dataset == "rad":
            return self.problem['image'][:-4]
        elif _dataset == "slake":
            return str(self.problem["img_id"])
        else:
            raise ValueError(f"Invalid _dataset value: {_dataset}. The value must be 'rad' or 'slake'.")
