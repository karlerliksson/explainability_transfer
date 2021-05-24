## T5 for generative explanations
This subrepo is used to train T5 models for generative explanations according to the WT5 framework.
The code is adopted from the [WT5 Github repo](https://github.com/google-research/google-research/tree/master/wt5).

Exemplary training scripts that illustrate how to perform explainability transfer,
train baseline models etc. can be found in `explainability_transfer/scripts/t5`.
The training scripts rely on an extended version of the [Text-To-Text Transfer Transformer repo](https://github.com/google-research/text-to-text-transfer-transformer)
that supports data parallel training using the T5 HuggingFace model and some additional utility features. For more details, see the `text-to-text-transfer-transformer` folder. 
