#!/bin/bash

# 设置错误时退出
set -e

echo "Starting data preprocessing..."

# 处理训练集
echo "Processing Training Dataset"
echo "Processing simplerl dataset..."
python simplerl.py --local_dir ~/data/simplerl
if [ $? -eq 0 ]; then
    echo "simplerl processing completed successfully"
else
    echo "simplerl processing failed"
    exit 1
fi

# 处理测试集
echo "Processing Test Dataset"
echo "Processing AIME24 dataset..."
python aime24.py --local_dir ~/data/aime24 
if [ $? -eq 0 ]; then
    echo "AIME24 processing completed successfully"
else
    echo "AIME24 processing failed"
    exit 1
fi

# 处理GSM8K数据集
echo "Processing GSM8K dataset..."
python gsm8k.py --local_dir ~/data/gsm8k 
if [ $? -eq 0 ]; then
    echo "GSM8K processing completed successfully"
else
    echo "GSM8K processing failed"
    exit 1
fi

# 处理AIME25数据集
echo "Processing AIME25 dataset..."
python aime25.py --local_dir ~/data/aime25 
if [ $? -eq 0 ]; then
    echo "AIME25 processing completed successfully"
else
    echo "AIME25 processing failed"
    exit 1
fi

# 处理math500数据集
echo "Processing math500 dataset..."
python math500.py --local_dir ~/data/math500 
if [ $? -eq 0 ]; then
    echo "math500 processing completed successfully"
else
    echo "math500 processing failed"
    exit 1
fi

# 处理livebench_language数据集
echo "Processing livebench_language dataset..."
python livebench_language.py --local_dir ~/data/livebench_language 
if [ $? -eq 0 ]; then
    echo "livebench_language processing completed successfully"
else
    echo "livebench_language processing failed"
    exit 1
fi

# 处理QwQ-LongCoT
echo "Processing QwQ-LongCoT dataest ..."
python QwQLongCoT.py 
if [ $? -eq 0 ]; then
    echo "QwQ-LongCoT processing completed successfully"
else
    echo "QwQ-LongCoT processing failed"
    exit 1
fi

echo "All datasets processed successfully!"
