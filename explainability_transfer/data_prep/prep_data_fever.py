from pathlib import Path

import tensorflow_datasets as tfds
import tensorflow as tf
import wt5.preprocessors as pp
import tqdm

DATA_DIR = Path.home() / "workspace" / "data" / "fever" / "hf"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Set CPU as only available physical device for tf
# Otherwise the tfds dataset can occupy a lot of space on the GPU
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices)

for (split, name) in zip(["train", "validation", "test"], ["train", "val", "test"]):
    ds_fever, ds_info = tfds.load("fever:0.1.1", split=split, with_info=True)
    ds_expl = pp.fever(ds_fever, prefix="explain classification", input_features=[('passage',)])
    ds_no_expl = pp.fever(ds_fever, prefix="classification",
                          input_features=[('passage',)], drop_explanations=True)
    ds = ds_expl.concatenate(ds_no_expl)

    with open(DATA_DIR / (name + ".source"), "w") as inputs:
        with open(DATA_DIR / (name + ".target"), "w") as targets:
            for sample in tqdm.tqdm(ds):
                # Append claim to input
                input_str = sample['inputs'].numpy().decode("utf-8").replace("\n", " ")
                input_str += " "
                input_str += sample['claim'].numpy().decode("utf-8").replace("\n", " ")
                inputs.write(input_str + "\n")
                targets.write(sample['targets'].numpy().decode("utf-8").replace("\n", " ") + "\n")
