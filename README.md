# Explainability Transfer
This repository contains the source code used to produce the results in our paper *Cross-Domain Transfer of Generative Explanations using Text-to-Text Models* that is to be presented at NLDB 2021.

## Requirements
It is assumed that you have `docker` and `docker-compose` installed on your system.

## Usage
Start by copying `.env.template` to `.env` and specify the environment variables according to your preferences. This includes setting paths to the folders where the data will be downloaded and models will be stored. Make sure that `${MODEL_DIR}`, `${DATA_DIR}` and `${DATA_DIR}/tensorflow_datasets` exist and are accessible to avoid permission issues after mounting.

Simply run `make build && make up` to build the docker image and spin up the containers. A jupyterlab and tensorboard server will be started and exposed on the ports specified in `.env`.

To run the explainability transfer experiments, follow these steps:
1. Start a new tmux session.
2. Run `make shell` to attach to a `bash` shell on the `repl` container.
3. Move to the `explainability_transfer` folder and install the necessary local packages.

      ```bash
      cd explainability_transfer
      sh startup.sh
      ```
4. The first time you set up the environment, download and prepare the [MultiRC](https://cogcomp.seas.upenn.edu/multirc/), [FEVER](https://fever.ai/resources.html) and [SciFact](https://github.com/allenai/scifact) datasets. This will both cache `tfds` versions of the datasets and save raw text files that can be used with the Huggingface library.

      ```bash
      python data_prep/prep_data_multirc.py
      python data_prep/prep_data_fever.py
      python data_prep/prep_data_scifact.py
      ```
5. Example scripts to perform explainability transfer for T5 and BART can be found in `explainability_transfer/scripts`. This includes all experiments and hyperparameter settings for the results in the paper. For example, to run a full explainability transfer experiment for `T5-base` from MultiRC to SciFact with different proportions of annotated explanations during fine-tuning:

      ```bash
      ./scripts/t5/base/ft_scifact_ep_multirc_base.sh [SEED]
      ```
      All scripts expect a seeding value for the random number generators. Additionally, for the scripts that take an explainability pre-training task as input, simply include `"multi_rc"` or `"fever"` as the second argument. Consult the scripts for the different models for more details.

Training and evaluation metrics are logged in the same folders as where the model checkpoints are saved. The tensorboard server automatically loads the logs that are in the default model locations. Otherwise, just update the `tensorboard` service in `docker/docker-compose.yaml` accordingly.

## Citation
```bibtex
@inproceedings{Erliksson2021ExplainabilityTransfer,
  title={Cross-Domain Transfer of Generative Explanations using Text-to-Text Models},
  author={Karl Fredrik Erliksson and Anders Arpteg and Mihhail Matskin and Amir H. Payberah},
  booktitle={26th International Conference on Natural Language and Information Systems, NLDB 2021},
  year={2021},
  note={to appear}
}
```

## Acknowledgements
This work has been a joint collaboration between Peltarion and KTH Royal Institute of Technology, partly funded by the EIT Digital Doctoral Programme.

## Contact
Email: `karl.erliksson@gmail.com`
