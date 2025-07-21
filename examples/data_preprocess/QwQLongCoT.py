# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    # 查找 \boxed{ 的位置
    boxed_start = solution_str.find(r'\boxed{')
    if boxed_start == -1:
        raise ValueError("未找到 \\boxed{…} 结构")
    
    # 从 \boxed{ 后开始查找匹配的花括号
    content_start = boxed_start + len(r'\boxed{')
    brace_count = 1
    pos = content_start
    
    # 确保不会超出字符串范围
    while pos < len(solution_str) and brace_count > 0:
        char = solution_str[pos]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count > 0:
        # 提供更详细的错误信息以便调试
        preview = solution_str[max(0, boxed_start):min(len(solution_str), boxed_start + 100)]
        raise ValueError(f"未找到匹配的右花括号，预览内容: {preview}")
    
    # 提取花括号内的内容
    final_answer = solution_str[content_start:pos-1].strip()
    # 移除可能的逗号（如数字中的千位分隔符）
    final_answer = final_answer.replace(",", "")
    return final_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/QwQ-LongCoT-Verified-130K")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "qingy2024/QwQ-LongCoT-Verified-130K"

    dataset = datasets.load_dataset(data_source, "verified")

    # filter solution length
    train_dataset = dataset["train"].filter(lambda x: len(x["solution"]) < 4000)
    
    # filter problem length
    train_dataset = train_dataset.filter(lambda x: len(x["problem"]) < 4000)

    # filter solution with \boxed inside
    def has_complete_boxed(example):
        solution = example["solution"]
        boxed_start = solution.find(r'\boxed{')
        if boxed_start == -1:
            return False
        
        # 检查是否有匹配的右花括号
        content_start = boxed_start + len(r'\boxed{')
        brace_count = 1
        pos = content_start
        
        while pos < len(solution) and brace_count > 0:
            if solution[pos] == '{':
                brace_count += 1
            elif solution[pos] == '}':
                brace_count -= 1
            pos += 1
        
        return brace_count == 0
    
    train_dataset = train_dataset.filter(has_complete_boxed)

    # only save first 2k data
    train_dataset = train_dataset.select(range(2048))

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("solution")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
