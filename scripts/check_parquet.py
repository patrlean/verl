#!/usr/bin/env python3
"""
检查parquet文件结构的脚本
"""

import pandas as pd
import sys

def inspect_parquet(file_path):
    """检查parquet文件的结构"""
    print(f"🔍 检查文件: {file_path}")
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        
        print(f"\n📊 基本信息:")
        print(f"   行数: {len(df)}")
        print(f"   列数: {len(df.columns)}")
        
        print(f"\n📋 列名列表:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        print(f"\n🔍 前几行数据预览:")
        print(df.head())
        
        print(f"\n📈 数据类型:")
        print(df.dtypes)
        
        if 'data_source' in df.columns:
            print(f"\n🏷️  data_source 列的唯一值:")
            print(df['data_source'].value_counts())
        else:
            print(f"\n⚠️  没有找到 'data_source' 列")
            
        # 检查其他可能的数据源列
        possible_data_source_cols = ['data_source_key', 'source', 'dataset', 'task']
        for col in possible_data_source_cols:
            if col in df.columns:
                print(f"\n🔍 找到可能的数据源列 '{col}':")
                print(df[col].value_counts())
        
        return df.columns.tolist()
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return None

if __name__ == "__main__":
    # 检查生成的parquet文件
    file_path = "/root/data/gsm8k/qwen3_8b_base_gen_gsm8k_test.parquet"
    columns = inspect_parquet(file_path)
    
    if columns:
        print(f"\n💡 建议的配置:")
        if 'data_source' in columns:
            print(f"   data.data_source_key=data_source")
        else:
            print(f"   可能需要设置正确的 data_source_key")
            
        # 检查其他必要的列
        required_keys = ['responses', 'reward_model']
        print(f"\n🔧 检查必要的列:")
        for key in required_keys:
            if key in columns:
                print(f"   ✅ {key}: 存在")
            else:
                print(f"   ❌ {key}: 缺失")