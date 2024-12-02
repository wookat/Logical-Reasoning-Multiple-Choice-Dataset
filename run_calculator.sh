#!/bin/bash


# 进入镜像，首先激活环境
source ~/.bashrc  # 确保conda命令可用
eval "$(conda shell.bash hook)"  # 初始化conda
conda activate openmind_finetune

# 配置NPU运行环境
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/

# 配置CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 运行Python脚本
python prompt_token_calculator.py