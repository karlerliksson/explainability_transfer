import os
import json
import argparse
import functools
import json
import logging
from pathlib import Path

import t5
import torch
import transformers
import wt5.tasks
import wt5.mixtures
import tensorflow as tf
import numpy as np

from utils import get_best_checkpoint, cleanup_checkpoints

parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", type=str, default="t5-base", required=False)

parser.add_argument("--ft_task", type=str, default="scifact", required=False)
parser.add_argument("--ft_task_name", type=str, default="scifact_v011", required=False)
parser.add_argument("--ft_steps", type=int, default=1200, required=False)
parser.add_argument("--ft_save_steps", type=int, default=50, required=False)
parser.add_argument("--ft_batch_size", type=int, default=24, required=False)
parser.add_argument("--ft_batch_size_eval", type=int, default=64, required=False)
parser.add_argument("--ft_acc_steps", type=int, default=4, required=False)
parser.add_argument("--ft_lr", type=float, default=1e-4, required=False)
parser.add_argument("--ft_model_dir_suffix", type=str, default="", required=False)

parser.add_argument("--model_dir_stem", type=str, default="/workspace/models/t5/", required=False)

parser.add_argument("--seed", type=int, default=123456, required=False)

# Set CPU as only available physical device for tf
# Otherwise the tfds dataset can occupy a lot of space on the GPU
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def parse_args(args):
    # Fine-tuning task
    if args.ft_task == "scifact":
        args.ft_eval_tasks = ["scifact_eval_v011",]
        args.ft_sequence_length = {"inputs": 1024, "targets": 512}
    else:
        raise ValueError("Fine-tuning task not recognized")

    return args


def main(args):
    args = parse_args(args)

    ###### EXPLAINABILITY FINE-TUNING ######
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    logging.info(args.ft_task_name)
    ft_model_dir = Path(args.model_dir_stem) / "{}_no_ep_{}_seed{}{}".format(
        args.ft_task_name,
        Path(args.model_name_or_path).name,
        args.seed,
        args.ft_model_dir_suffix)

    logging.info("Loading model...")
    model = t5.models.HfPyTorchModel(args.model_name_or_path, str(ft_model_dir), device)

    # Run fine-tuning
    logging.info("Starting fine-tuning...")
    model.train(
        mixture_or_task_name=args.ft_task_name,
        steps=args.ft_steps,
        save_steps=args.ft_save_steps,
        sequence_length=args.ft_sequence_length,
        split="train",
        batch_size=args.ft_batch_size,
        optimizer=functools.partial(transformers.AdamW, lr=args.ft_lr),
        accumulation_steps=args.ft_acc_steps
    )
    with open(ft_model_dir / "input_args.json", "w") as f:
        json.dump(args.__dict__, f, default=str)

    # Evaluate checkpoints
    logging.info("Starting evaluation...")
    metric_scores = []
    for eval_task in args.ft_eval_tasks:
        metric_scores += model.eval(
            eval_task,
            sequence_length=args.ft_sequence_length,
            split="validation",
            batch_size=args.ft_batch_size_eval,
            checkpoint_steps="all",
            max_length=args.ft_sequence_length['targets']
        )
    with open(ft_model_dir / "metrics.json", "w") as f:
        json.dump(metric_scores, f)

    cleanup_checkpoints(metric_scores, args.ft_eval_tasks[0], ["f1a", "f1"], ft_model_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
