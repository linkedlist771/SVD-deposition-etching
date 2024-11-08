# SVD-deposition-etching


## 1. 单卡训练
```bash
accelerate launch train_svd.py \
    --base_folder=data \
    --pretrained_model_name_or_path=stable-video-diffusion-img2vid \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=384 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=1 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 
```

## 2.双卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1  accelerate launch --num_processes=2 train_svd.py \
    --base_folder=assets \
    --pretrained_model_name_or_path=models/stable-video-diffusion-img2vid \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=384 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=1 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200
```