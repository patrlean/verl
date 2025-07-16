#!/bin/bash
set -e  # 遇到错误就退出

# 配置 Git
git config --global user.name "patrlean"
git config --global user.email "tianyou.2001@outlook.com"

# 克隆仓库并安装
git clone https://github.com/patrlean/verl.git
cd verl
pip install -e .

# 进入数据预处理目录
cd examples/data_preprocess
chmod +x ./process_data_rl.bash
./process_data_rl.bash

# 测试 transformers
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen3-14B')"

python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen3-8B')"