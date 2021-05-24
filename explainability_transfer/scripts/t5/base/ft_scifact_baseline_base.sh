#!/usr/bin/env bash
SEED=$1

# Standard learning rate and batch size
python scripts/t5/run_scifact_baseline.py \
    --model_name_or_path "t5-base" \
    --ft_steps 500 \
    --ft_save_steps 20 \
    --ft_batch_size 64 \
    --ft_acc_steps 16 \
    --ft_batch_size_eval 64 \
    --seed ${SEED}

# Smaller batch size (best baseline that is used in the paper)
python scripts/t5/run_scifact_baseline.py \
    --model_name_or_path "t5-base" \
    --ft_steps 4000 \
    --ft_save_steps 160 \
    --ft_batch_size 8 \
    --ft_acc_steps 2 \
    --ft_batch_size_eval 64 \
    --ft_model_dir_suffix "_fbs8" \
    --seed ${SEED}

# Bigger learning rate
python scripts/t5/run_scifact_baseline.py \
    --model_name_or_path "t5-base" \
    --ft_steps 500 \
    --ft_save_steps 20 \
    --ft_batch_size 64 \
    --ft_acc_steps 16 \
    --ft_batch_size_eval 64 \
    --ft_lr 5e-4 \
    --ft_model_dir_suffix "_flr5e-4" \
    --seed ${SEED}