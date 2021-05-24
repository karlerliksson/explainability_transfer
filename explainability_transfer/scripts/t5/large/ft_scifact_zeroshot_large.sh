#!/usr/bin/env bash
SEED=$1
EP_TASK=$2

python scripts/t5/run_t5_exp_transfer_ft_all.py \
    --model_name_or_path "t5-large" \
    --ep_task ${EP_TASK} \
    --ft_task_name "scifact_v011_no_expl" \
    --ft_steps 500 \
    --ft_save_steps 20 \
    --ft_batch_size 64 \
    --ft_acc_steps 32 \
    --ft_batch_size_eval 64 \
    --seed ${SEED}
