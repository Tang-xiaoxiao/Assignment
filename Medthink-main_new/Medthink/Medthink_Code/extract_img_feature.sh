#!/bin/bash

# 课程作业图像处理指令
# ours
# 训练数据
python Medthink-main/Medthink/Medthink_Code/extract_img_feature.py --device cuda:0 --image_dir Training_Set/Training_Set/Training/ --output_dir  Training_Set/Training_Set/Training_feature/
# 测试数据
python Medthink-main/Medthink/Medthink_Code/extract_img_feature.py --device cuda:0 --image_dir Test_Set/Test_Set/Test --output_dir  Test_Set/Test_Set/Test_feature/


python extract_img_feature.py --device cuda:0 --image_dir data/R-RAD/images/ --output_dir  data/R-RAD/
python extract_img_feature.py --device cuda:0 --image_dir data/R-SLAKE/img/ --output_dir  data/R-SLAKE/
