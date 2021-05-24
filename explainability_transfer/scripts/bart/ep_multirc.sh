#!/usr/bin/env bash
SEED=$1

python bart/finetune.py \
    --learning_rate 3e-5 \
    --gpus 2 \
    --do_train \
    --do_predict \
    --val_check_interval 0.25 \
    --n_val 1000 \
    --adam_eps 1e-06 \
    --num_train_epochs 12 \
    --data_dir /workspace/data/multi_rc/hf \
    --output_dir "/workspace/models/bart/multi_rc_transfer_expl_seed"${SEED} \
    --max_source_length 1024 --max_target_length 256 --val_max_target_length 256 --test_max_target_length 256 \
    --train_batch_size 4 --eval_batch_size 4 --gradient_accumulation_steps 16 \
    --model_name_or_path facebook/bart-large \
    --task explanation \
    --warmup_steps 500 \
    --sortish_sampler \
    --overwrite_output_dir \
    --seed ${SEED}
