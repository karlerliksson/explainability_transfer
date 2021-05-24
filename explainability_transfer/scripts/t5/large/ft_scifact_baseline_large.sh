#!/usr/bin/env bash
SEED=$1

python scripts/t5/run_scifact_baseline.py \
    --model_name_or_path "t5-large" \
    --ft_steps 4000 \
    --ft_save_steps 160 \
    --ft_batch_size 8 \
    --ft_acc_steps 4 \
    --ft_batch_size_eval 64 \
    --ft_model_dir_suffix "_fbs8" \
    --seed ${SEED}
