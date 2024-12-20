# SVD-deposition-etching


## 1. 单卡训练
```bash
accelerate launch train_svd.py \
    --base_folder=data \
    --pretrained_model_name_or_path=stable-video-diffusion-img2vid \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=50000 \
    --width=512 \
    --height=384 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=1 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --num_frames 14 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 
```

## 2.双卡训练
- 40G A100 大概两天

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --num_processes=2 train_svd.py \
    --base_folder=data \
    --pretrained_model_name_or_path=stable-video-diffusion-img2vid \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=50000 \
    --width=512 \
    --height=384 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=1 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --num_frames 12 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 > $(date +%m%d).log 2>&1 &
```


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 train_multi.py \
    --base_folder=data \
    --pretrained_model_name_or_path=stable-video-diffusion-img2vid \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=50000 \
    --width=512 \
    --height=384 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=1 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --num_frames 20 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200
