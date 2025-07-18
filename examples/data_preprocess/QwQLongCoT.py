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
    pattern = re.compile(r"\$?\s*\\boxed\{\s*([^}]+?)\s*\}\s*\$?")
    m = pattern.search(solution_str)
    if m is None:
        raise ValueError("未找到 \\boxed{…} 结构")
    
    # m.group(1) 就是花括号里的内容
    final_answer = m.group(1).replace(",", "").strip()
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
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
