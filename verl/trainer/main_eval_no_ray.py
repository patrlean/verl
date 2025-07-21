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
Offline evaluate - Debug version with detailed error reporting
"""

from collections import defaultdict
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


def inspect_first_sample(dataset):
    """详细检查第一个样本的数据"""
    print("=== 第一个样本详细数据 ===")
    first_row = dataset.iloc[0]
    
    for column in dataset.columns:
        value = first_row[column]
        print(f"\n{column}:")
        print(f"  类型: {type(value)}")
        print(f"  内容: {repr(value)}")
        
        # 特别处理复杂数据类型
        if column == 'responses':
            if isinstance(value, np.ndarray):
                print(f"  数组长度: {len(value)}")
                for i, item in enumerate(value):
                    print(f"  responses[{i}]: {repr(item)[:200]}...")
            elif isinstance(value, list):
                print(f"  列表长度: {len(value)}")
                for i, item in enumerate(value):
                    print(f"  responses[{i}]: {repr(item)[:200]}...")
        
        elif column == 'reward_model':
            if isinstance(value, dict):
                print(f"  字典键: {list(value.keys())}")
                for k, v in value.items():
                    print(f"  {k}: {repr(v)}")


def process_item(reward_fn, data_source, response_lst, reward_data):
    """Process a single item with detailed error reporting"""
    try:
        ground_truth = reward_data["ground_truth"]
        print(f"DEBUG: data_source={data_source}")
        print(f"DEBUG: ground_truth={ground_truth} (type: {type(ground_truth)})")
        print(f"DEBUG: response_lst type={type(response_lst)}")
        
        if isinstance(response_lst, np.ndarray):
            response_lst = response_lst.tolist()
        elif isinstance(response_lst, str):
            response_lst = [response_lst]
            
        print(f"DEBUG: response_lst after conversion: {len(response_lst)} items")
        
        # 检查reward_fn的签名
        import inspect
        sig = inspect.signature(reward_fn)
        print(f"DEBUG: reward_fn signature: {sig}")
        
        score_lst = []
        for i, response in enumerate(response_lst):
            print(f"DEBUG: Processing response {i}, length: {len(str(response))}")
            print(f"DEBUG: Response preview: {str(response)[:100]}...")
            
            # 尝试不同的调用方式
            try:
                # 方式1：只传递必要参数
                score = reward_fn(response, ground_truth)
                print(f"DEBUG: Score with 2 args: {score}")
            except Exception as e1:
                print(f"DEBUG: 2-arg call failed: {e1}")
                try:
                    # 方式2：传递更多参数
                    score = reward_fn(response, ground_truth, method="strict")
                    print(f"DEBUG: Score with method: {score}")
                except Exception as e2:
                    print(f"DEBUG: 3-arg call failed: {e2}")
                    try:
                        # 方式3：传递data_source
                        score = reward_fn(data_source, response, ground_truth)
                        print(f"DEBUG: Score with data_source: {score}")
                    except Exception as e3:
                        print(f"DEBUG: All calls failed: {e1}, {e2}, {e3}")
                        raise e3
            
            score_lst.append(score)
        
        print(f"DEBUG: All scores: {score_lst}")
        return data_source, np.mean(score_lst)
        
    except Exception as e:
        print(f"ERROR in process_item: {str(e)}")
        print(f"ERROR traceback: {traceback.format_exc()}")
        raise


# 修改main_eval_no_ray.py的main函数，将调试版本改为完整版本

@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Dataset columns: {dataset.columns.tolist()}")
    
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Get custom reward function
    compute_score = get_custom_reward_fn(config)
    if compute_score is None:
        print("ERROR: No reward function loaded!")
        return {}

    print(f"Reward function loaded successfully!")

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)

    # Process all items
    print(f"Processing {total} items...")
    successful_count = 0
    failed_count = 0
    
    for i in tqdm(range(total), desc="Evaluating"):
        try:
            # Convert numpy array to list if needed
            response_lst = responses[i]
            if isinstance(response_lst, np.ndarray):
                response_lst = response_lst.tolist()
            elif isinstance(response_lst, str):
                response_lst = [response_lst]
            
            ground_truth = reward_model_data[i]["ground_truth"]
            
            # Calculate scores for all responses
            score_lst = []
            for response in response_lst:
                score = compute_score(response, ground_truth)
                score_lst.append(score)
            
            # Store average score
            avg_score = np.mean(score_lst)
            data_source_reward[data_sources[i]].append(avg_score)
            successful_count += 1
            
        except Exception as e:
            if failed_count < 5:  # Only print first 5 errors
                print(f"Error processing item {i}: {str(e)}")
            failed_count += 1
            continue

    print(f"\nProcessing completed:")
    print(f"  Successful: {successful_count}")
    print(f"  Failed: {failed_count}")

    # Calculate final metrics
    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        if len(rewards) > 0:
            avg_score = np.mean(rewards)
            metric_dict[f"test_score/{data_source}"] = avg_score
            print(f"\nData source: {data_source}")
            print(f"  Samples: {len(rewards)}")
            print(f"  Average Score: {avg_score:.4f}")
            print(f"  Accuracy: {avg_score*100:.2f}%")

    print(f"\nFinal Results:")
    print(metric_dict)
    return metric_dict


if __name__ == "__main__":
    main()