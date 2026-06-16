# ============================================================
# 功能说明：
# 本脚本用于使用预训练的 DETR（detr_resnet101_dc5）模型对图像进行视觉特征提取，
# 并将提取后的特征保存为 .pth 文件，同时生成图片 ID 到特征索引的映射文件 name_map.json。
#
# 主要流程：
# 1. 加载预训练 DETR 模型；
# 2. 使用 torchvision.transforms 对输入图像进行预处理：
#    - Resize(224)
#    - 转为 Tensor
#    - 按 ImageNet 均值和方差归一化
# 3. 遍历指定目录下的图像文件；
# 4. 将每张图像输入模型，提取输出中的最后一层特征；
# 5. 将所有图像特征拼接后保存到输出目录；
# 6. 同时保存图片文件名（去掉后缀）与特征矩阵行索引之间的映射关系。
#
# 支持的数据集格式：
# - ours  : 直接遍历 image_dir 下的图片文件，支持 .jpg / .jpeg / .png
# - rad   : 遍历 image_dir 下的 .jpg 图片
# - slake : 遍历 image_dir 下的子目录，并读取每个子目录中的 source.jpg
#
# 输入参数说明：
# --device     : 指定运行设备，例如 cuda:0 或 cpu
# --image_dir  : 输入图片目录
# --output_dir : 输出特征保存目录
# --dataset    : 数据集类型，可选 rad / slake / ours
# --img_type   : 输出特征文件名，例如 detr，则保存为 detr.pth
#
# 输出文件：
# - {img_type}.pth   : 提取后的视觉特征张量
# - name_map.json    : 图片 ID 到特征索引的映射字典
#
# ============================================================

import torch
from PIL import Image
import torchvision.transforms as T
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import argparse
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def get_model(_img_type):
    print(f"Loading Model from GitHub...")
    # 直接从 github 加载 cooelf 的 detr 仓库，会自动下载并缓存到当前系统的 ~/.cache 中
    _model = torch.hub.load('cooelf/detr:main', 'detr_resnet101_dc5', pretrained=True)
    _transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return _model, _transform


def extract_features(_model, _transform, _input_image, _img_type, _device):
    print(f"Loading {_input_image}...")
    img = Image.open(_input_image).convert("RGB")
    input = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        return _model(input)[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # 将 ours 设为默认 dataset，并加入 choices
    parser.add_argument("--dataset", type=str, default="ours", choices=["rad", "slake", "ours"])
    # 修复原代码缺少的 img_type 参数
    parser.add_argument("--img_type", type=str, default="detr", help="特征保存的文件名(后缀前部分)")
    
    args = vars(parser.parse_args())
    for arg, value in args.items():
        print(f"{arg}: {value}")

    device = args['device'] if torch.cuda.is_available() else "cpu"
    img_type = args['img_type']
    images_dir = args['image_dir']
    output_file_path = args['output_dir']

    # 创建输出目录
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok=True)

    if args["dataset"] == "ours":
        images_path = os.listdir(images_dir)
        print(f"There are {len(images_path)} total files/folders in {images_dir}.")

        model, transform = get_model(img_type)
        model.to(device)
        model.eval()

        name_map = {}   # "KEY" 是图片ID(不带后缀)，"VALUE" 是特征矩阵中的行索引
        vision_features = []
        cnt = 0
        
        # 排序以保证每次提取特征的顺序一致
        images_path.sort() 

        for idx, image_path in enumerate(images_path):
            # 支持常见图片格式
            if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = os.path.join(images_dir, image_path)
                feature = extract_features(model, transform, image, img_type, device)
                
                # 获取不带后缀的文件名作为 KEY
                img_id = os.path.splitext(image_path)[0]
                name_map[img_id] = str(cnt)
                
                vision_features.append(feature.detach().cpu())
                cnt += 1

        if len(vision_features) > 0:
            vision_features = torch.cat(vision_features)
            print(f"Features extraction complete. Shape: {vision_features.shape}")
            
            # 安全保存路径
            torch.save(vision_features, os.path.join(output_file_path, f"{img_type}.pth"))
            with open(os.path.join(output_file_path, "name_map.json"), 'w') as outfile:
                json.dump(name_map, outfile, indent=4, ensure_ascii=False)
            print("Files (features and name_map.json) have been written successfully.")
        else:
            print("No matching images (.jpg, .jpeg, .png) found in the directory.")

    elif args["dataset"] == "rad":
        images_path = os.listdir(images_dir)
        print(f"There are {len(images_path)} images in {images_dir}.")

        model, transform = get_model(img_type)
        model.to(device)
        model.eval()

        name_map = {}
        vision_features = []
        cnt = 0
        for idx, image_path in enumerate(images_path):
            if image_path.endswith('.jpg'):
                image = os.path.join(images_dir, image_path)
                feature = extract_features(model, transform, image, img_type, device)
                name_map[image_path[:-4]] = str(cnt)
                vision_features.append(feature.detach().cpu())
                cnt += 1

        vision_features = torch.cat(vision_features)
        print(vision_features.shape)
        torch.save(vision_features, os.path.join(output_file_path, f"{img_type}.pth"))
        with open(os.path.join(output_file_path, "name_map.json"), 'w') as outfile:
            json.dump(name_map, outfile, indent=4, ensure_ascii=False)
        print("Files have been written...")

    elif args["dataset"] == "slake":
        images_path = os.listdir(images_dir)
        print(images_path)
        print(f"There are {len(images_path)} images in {images_dir}.")

        model, transform = get_model(img_type)
        model.to(device)
        model.eval()

        name_map = {}
        vision_features = []
        cnt = 0
        for idx, image_path in enumerate(images_path):
            if image_path.endswith('.jpg'):
                image = os.path.join(images_dir, image_path, 'source.jpg')
                feature = extract_features(model, transform, image, img_type, device)
                name_map[image_path[5:]] = str(cnt)
                vision_features.append(feature.detach().cpu())
                cnt += 1

        vision_features = torch.cat(vision_features)
        print(vision_features.shape)
        torch.save(vision_features, os.path.join(output_file_path, f"{img_type}.pth"))
        with open(os.path.join(output_file_path, "name_map.json"), 'w') as outfile:
            json.dump(name_map, outfile, indent=4, ensure_ascii=False)
        print("Files have been written...")
    else:
        raise ValueError(f"Invalid dataset value: {args['dataset']}.")
