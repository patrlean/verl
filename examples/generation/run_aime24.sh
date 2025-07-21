set -x

data_path=$HOME/data/aime24/test.parquet
save_path=$HOME/data/aime24/qwen3_8b_base_gen_aime24_test.parquet
model_path=/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path\
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=4 \
    rollout.max_num_batched_tokens=16384 \
    rollout.gpu_memory_utilization=0.6 2>&1 | tee gen_qwen3_8b_base_aime24_test.log


data_path=$HOME/data/aime24/test.parquet
save_path=$HOME/data/aime24/qwen3_8b_base_sft_aime24_test.parquet
model_path=/workspace/qwen3-8b-base-sft/merged_hf

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path\
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=4 \
    rollout.max_num_batched_tokens=16384 \
    rollout.gpu_memory_utilization=0.6 2>&1 | tee gen_qwen3_8b_base_sft_aime24_test.log

data_path=$HOME/data/aime24/test.parquet
save_path=$HOME/data/aime24/qwen3_8b_base_sft_grpo_aime24_test.parquet
model_path=/workspace/merged_qwen3_grpo_model

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path\
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=4 \
    rollout.max_num_batched_tokens=16384 \
    rollout.gpu_memory_utilization=0.6 2>&1 | tee gen_qwen3_8b_base_sft_grpo_aime24_test.log