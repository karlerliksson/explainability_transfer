#!/usr/bin/env bash
SEED=$1

python scripts/t5/run_t5_exp_transfer_prop.py \
    --model_name_or_path "t5-base" \
    --ep_task "fever" \
    --ep_batch_size 64 \
    --ep_acc_steps 16 \
    --ft_steps 500 \
    --ft_save_steps 20 \
    --ft_batch_size 64 \
    --ft_acc_steps 16 \
    --ft_batch_size_eval 64 \
    --seed ${SEED}
