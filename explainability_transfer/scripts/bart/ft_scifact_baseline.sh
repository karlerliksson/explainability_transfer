#!/usr/bin/env bash
SEED=$1
DATA_DIR=/workspace/data/scifact/hf/full_expl
OUTPUT_DIR=/workspace/models/bart/

echo "------- Running experiment: scifact_full_no_ep -------"
python bart/finetune.py \
    --learning_rate 1e-5 \
    --gpus 2 \
    --do_train \
    --do_predict \
    --adam_eps 1e-06 \
    --num_train_epochs 35 \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}"scifact_full_no_ep_seed"${SEED} \
    --max_source_length 1024 --max_target_length 256 --val_max_target_length 256 --test_max_target_length 256 \
    --train_batch_size 2 --eval_batch_size 2 --gradient_accumulation_steps 4 \
    --model_name_or_path facebook/bart-large \
    --task explanation \
    --warmup_steps 50 \
    --overwrite_output_dir \
    --seed ${SEED}

echo "------- Running experiment: scifact_full_no_ep_bs16 -------"
python bart/finetune.py \
    --learning_rate 1e-5 \
    --gpus 2 \
    --do_train \
    --do_predict \
    --adam_eps 1e-06 \
    --num_train_epochs 35 \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}"scifact_full_no_ep_bs16_seed"${SEED} \
    --max_source_length 1024 --max_target_length 256 --val_max_target_length 256 --test_max_target_length 256 \
    --train_batch_size 2 --eval_batch_size 2 --gradient_accumulation_steps 8 \
    --model_name_or_path facebook/bart-large \
    --task explanation \
    --warmup_steps 50 \
    --overwrite_output_dir \
    --seed ${SEED}