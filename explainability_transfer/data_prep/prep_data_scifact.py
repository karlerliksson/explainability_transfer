from pathlib import Path

import tqdm
import tensorflow_datasets as tfds
import tensorflow as tf
import t5
import wt5.preprocessors as pp
import wt5.tasks
import wt5.mixtures

DATA_DIR = Path.home() / "workspace" / "data" / "scifact" / "hf"

# Set CPU as only available physical device for tf
# Otherwise the tfds dataset can occupy a lot of space on the GPU
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices)

(DATA_DIR / "full_expl").mkdir(parents=True, exist_ok=True)
for (split, name) in zip(["train", "validation", "test"], ["train", "val", "test"]):
    ds_scifact, ds_info = tfds.load("scifact:0.1.1", split=split, with_info=True)
    ds_expl = pp.scifact(ds_scifact,
                         input_features=[('text', 'passage'), ('claim',)],
                         drop_explanations=False,
                         prefix="explain classification")
    with open(DATA_DIR / "full_expl" / (name + ".source"), "w") as inputs:
        with open(DATA_DIR / "full_expl" / (name + ".target"), "w") as targets:
            for sample in tqdm.tqdm(ds_expl):
                inputs.write(sample['inputs'].numpy().decode("utf-8") + "\n")
                targets.write(sample['targets'].numpy().decode("utf-8") + "\n")

for n in wt5.tasks.n_scifact_explanations:
    mixture_name = "scifact_rationales_{}_explanations".format(n)
    task_expl = "scifact_explanations_take{}_v011".format(n)
    task_no_expl = "scifact_labels_skip{}_v011".format(n)
    evaluation_task = "scifact_eval_v011"
    (DATA_DIR / mixture_name).mkdir(parents=True, exist_ok=True)

    ds_expl = t5.data.get_mixture_or_task(task_expl).get_dataset(
        {'inputs': 1024, 'targets': 512}, split="train", shuffle=False)
    ds_no_expl = t5.data.get_mixture_or_task(task_no_expl).get_dataset(
        {'inputs': 1024, 'targets': 512}, split="train", shuffle=False)
    with open(DATA_DIR / mixture_name / "train.source", "w") as inputs:
        with open(DATA_DIR / mixture_name / "train.target", "w") as targets:
            for ds in [ds_expl, ds_no_expl]:
                for sample in tqdm.tqdm(ds):
                    inputs.write(sample['inputs_plaintext'].numpy().decode("utf-8") + "\n")
                    targets.write(sample['targets_plaintext'].numpy().decode("utf-8") + "\n")

    for (split, name) in zip(["validation", "test"], ["val", "test"]):     
        ds = t5.data.get_mixture_or_task(evaluation_task).get_dataset(
            {'inputs': 1024, 'targets': 512}, split=split, shuffle=False)
        with open(DATA_DIR / mixture_name / (name + ".source"), "w") as inputs:
            with open(DATA_DIR / mixture_name / (name + ".target"), "w") as targets:
                for sample in tqdm.tqdm(ds):
                    inputs.write(sample['inputs_plaintext'].numpy().decode("utf-8") + "\n")
                    targets.write(sample['targets_plaintext'].numpy().decode("utf-8") + "\n")

mixture_name = "scifact_v011_no_expl"
folder_name = "scifact_no_expl"
evaluation_task = "scifact_eval_v011"
(DATA_DIR / folder_name).mkdir(parents=True, exist_ok=True)

ds = t5.data.get_mixture_or_task(mixture_name).get_dataset(
    {'inputs':1024, 'targets': 512}, split="train", shuffle=False)
with open(DATA_DIR / folder_name / "train.source", "w") as inputs:
    with open(DATA_DIR / folder_name / "train.target", "w") as targets:
        for sample in tqdm.tqdm(ds):
            inputs.write(sample['inputs_plaintext'].numpy().decode("utf-8") + "\n")
            targets.write(sample['targets_plaintext'].numpy().decode("utf-8") + "\n")

for (split, name) in zip(["validation", "test"], ["val", "test"]):
    ds = t5.data.get_mixture_or_task(evaluation_task).get_dataset(
        {'inputs': 1024, 'targets': 512}, split=split, shuffle=False)
    with open(DATA_DIR / folder_name / (name + ".source"), "w") as inputs:
        with open(DATA_DIR / folder_name / (name + ".target"), "w") as targets:
            for sample in tqdm.tqdm(ds):
                inputs.write(sample['inputs_plaintext'].numpy().decode("utf-8") + "\n")
                targets.write(sample['targets_plaintext'].numpy().decode("utf-8") + "\n")
