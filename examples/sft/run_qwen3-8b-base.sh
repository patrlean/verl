set -x

torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files="[$HOME/data/gsm8k/train.parquet, $HOME/data/QwQ-LongCoT/train.parquet]" \
    data.val_files="[$HOME/data/gsm8k/test.parquet]" \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    model.partial_pretrain=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
    model.lora_rank=32 \
    trainer.project_name=qwen3-8b-base-sft \
    trainer.experiment_name=qwen3-8b-base-sft-2048 \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=8