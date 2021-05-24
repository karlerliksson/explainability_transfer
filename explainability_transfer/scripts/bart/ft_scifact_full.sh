#!/usr/bin/env bash
SEED=$1
EP_TASK=$2

DATA_DIR=/workspace/data/scifact/hf/full_expl
OUTPUT_DIR="/workspace/models/bart/scifact_full_ep_"${EP_TASK}"_seed"${SEED}
EP_DIR="/workspace/models/bart/"${EP_TASK}"_transfer_expl_seed"${SEED}"/best_tfmr"

echo "------- Running experiment: ${OUTPUT_DIR} -------"
python bart/finetune.py \
    --learning_rate 1e-5 \
    --gpus 2 \
    --do_train \
    --do_predict \
    --adam_eps 1e-06 \
    --num_train_epochs 15 \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --max_source_length 1024 --max_target_length 256 --val_max_target_length 256 --test_max_target_length 256 \
    --train_batch_size 2 --eval_batch_size 4 --gradient_accumulation_steps 4 \
    --model_name_or_path ${EP_DIR} \
    --task explanation \
    --warmup_steps 10 \
    --overwrite_output_dir \
    --seed ${SEED}