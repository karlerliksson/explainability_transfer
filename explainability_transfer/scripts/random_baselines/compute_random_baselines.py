from pathlib import Path

import numpy as np
import pandas as pd
import nltk
import tensorflow_datasets as tfds
from tqdm import tqdm

from wt5 import postprocessors
from  wt5 import metrics
from utils import calculate_bleu, calculate_rouge

NLTK_DIR = Path.home() / "workspace" / "data" / "nltk"
OUTPUT_DIR = Path.home() / "workspace" / "models" / "random_baselines"
SEED_START = 123456
N = 5

def compute_metrics(data_dev):
    targets = [postprocessors.extractive_explanations(d["target"], example={'inputs_plaintext': d["input"]}, is_target=True)
               for d in data_dev]
    predictions = [postprocessors.extractive_explanations(d["prediction"], example={'inputs_plaintext': d["input"]}, is_target=False)
                   for d in data_dev]
    extractive_metrics = metrics.extractive_explanations_metric(targets, predictions)
    bleu_score = calculate_bleu([d["prediction"] for d in data_dev], [d["target"] for d in data_dev])
    rouge_scores = calculate_rouge([d["prediction"] for d in data_dev], [d["target"] for d in data_dev])
    extractive_metrics.update(bleu_score)
    extractive_metrics.update(rouge_scores)
    return extractive_metrics

def random_baseline(task, ds_name, input_feat, seed=123456):
    np.random.seed(seed)
    
    ds = tfds.load(ds_name, split="train")
    ds_dev = tfds.load(ds_name, split="validation")

    data = []
    num_explanations = []
    labels = []
    for sample in tqdm(ds):
        d = {}
        d["explanations"] = [e.decode("utf-8") for e in sample["evidences"].numpy()]
        d["num_explanations"] = len(d["explanations"])
        num_explanations.append(d["num_explanations"])
        d["label"] = "True" if sample["label"].numpy() else "False"
        labels.append(d["label"])
        d["input"] = sample[input_feat].numpy().decode("utf-8")
        d["input_sentences"] = nltk.tokenize.sent_tokenize(d["input"])
        data.append(d)

    labels = np.array(labels)
    classes, class_count = np.unique(labels, return_counts=True)
    class_weight = class_count/class_count.sum()

    num_explanations = np.array(num_explanations)
    x_obs, count = np.unique(num_explanations, return_counts=True)
    y_obs = count/count.sum()
    x_obs_ext = np.arange(20)
    y_obs_ext = np.zeros_like(x_obs_ext, dtype=float)
    for (x,y) in zip(x_obs, y_obs):
        y_obs_ext[x] = y
    
    data_dev = []
    separator = " explanation: "
    for sample in tqdm(ds_dev):
        d = {}
        d["explanations"] = [e.decode("utf-8") for e in sample["evidences"].numpy()]
        d["num_explanations"] = len(d["explanations"])
        d["label"] = "True" if sample["label"].numpy() else "False"
        d["input"] = sample[input_feat].numpy().decode("utf-8")
        d["input_sentences"] = nltk.tokenize.sent_tokenize(d["input"])

        d["label_pred"] = np.random.choice(classes, p=class_weight)

        M = np.random.choice(x_obs_ext, p=y_obs_ext)
        if M >= len(d["input_sentences"]):
            M = len(d["input_sentences"]) - 1
        sent_ind = sorted(np.random.permutation(len(d["input_sentences"]))[:M])
        d["explanations_pred"] = [d["input_sentences"][i] for i in sent_ind]

        d["target"] = d["label"]
        if len(d["explanations"]) > 0:
            d["target"] += separator
            d["target"] += separator.join(d["explanations"])

        d["prediction"] = d["label_pred"]
        if len(d["explanations_pred"]) > 0:
            d["prediction"] += separator
            d["prediction"] += separator.join(d["explanations_pred"])
        data_dev.append(d)
        
    extractive_metrics = compute_metrics(data_dev)
    extractive_metrics["seed"] = seed
    extractive_metrics["task"] = task
    return extractive_metrics

def main():
    nltk.download("punkt", download_dir=NLTK_DIR)
    nltk.data.path.insert(0, NLTK_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # SciFact
    baseline_metrics = [random_baseline("scifact", "scifact:0.1.1", "text", seed=seed)
                        for seed in range(SEED_START, SEED_START + N)]

    # MultiRC
    baseline_metrics += [random_baseline("multi_rc", "eraser_multi_rc2:0.1.1", "passage", seed=seed)
                         for seed in range(SEED_START, SEED_START + N)]

    # FEVER
    baseline_metrics += [random_baseline("fever", "fever:0.1.1", "passage", seed=seed)
                         for seed in range(SEED_START, SEED_START + N)]

    df = pd.DataFrame(baseline_metrics).set_index(["task", "seed"])
    df.to_pickle(OUTPUT_DIR / "RandomBaselines.pkl")


if __name__ == "__main__":
    main()
