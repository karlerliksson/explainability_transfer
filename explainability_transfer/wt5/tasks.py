# coding=utf-8
#
# Extended and adopted from https://github.com/google-research/google-research/blob/master/wt5/wt5/tasks.py
# 
# Original copyright:
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WT5 tasks."""
import functools

from . import metrics
from . import postprocessors
from . import preprocessors

import t5.data
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics
import tensorflow_datasets as tfds


TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask
Task = t5.data.Task

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS


# ======================== MultiRC Transfer ======================

TaskRegistry.add(
    "eraser_multi_rc_v011_transfer_no_expl",
    TfdsTask,
    tfds_name="eraser_multi_rc2:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.eraser_multi_rc,
        drop_explanations=True,
        prefix="classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=['train'],
    metric_fns=[metrics.extractive_explanations_metric])

TaskRegistry.add(
    "eraser_multi_rc_v011_transfer",
    TfdsTask,
    tfds_name="eraser_multi_rc2:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.eraser_multi_rc,
        prefix="explain classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=['train'],
    metric_fns=[metrics.extractive_explanations_metric])

TaskRegistry.add(
    "eraser_multi_rc_eval_v011_transfer",
    TfdsTask,
    tfds_name="eraser_multi_rc2:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.eraser_multi_rc,
        prefix="explain classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["validation", "test"],
    metric_fns=[metrics.extractive_explanations_metric])

# ======================== SciFact ======================

TaskRegistry.add(
    "scifact_v011_no_expl",
    TfdsTask,
    tfds_name="scifact/claims_binary:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=True,
        prefix="classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["train"],
    metric_fns=[])

TaskRegistry.add(
    "scifact_true_v011_no_expl",
    TfdsTask,
    tfds_name="scifact/true:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=True,
        prefix="classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["train"],
    metric_fns=[])

TaskRegistry.add(
    "scifact_false_v011_no_expl",
    TfdsTask,
    tfds_name="scifact/false:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=True,
        prefix="classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["train"],
    metric_fns=[])

TaskRegistry.add(
    "scifact_v011",
    TfdsTask,
    tfds_name="scifact/claims_binary:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=False,
        prefix="explain classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["train"],
    metric_fns=[])

TaskRegistry.add(
    "scifact_true_v011",
    TfdsTask,
    tfds_name="scifact/true:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=False,
        prefix="explain classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["train"],
    metric_fns=[])

TaskRegistry.add(
    "scifact_false_v011",
    TfdsTask,
    tfds_name="scifact/false:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=False,
        prefix="explain classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["train"],
    metric_fns=[])

TaskRegistry.add(
    "scifact_eval_v011",
    TfdsTask,
    tfds_name="scifact/claims_binary:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        prefix="explain classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["validation", "test"],
    metric_fns=[metrics.extractive_explanations_metric])

TaskRegistry.add(
    "scifact_eval_v011_no_expl",
    TfdsTask,
    tfds_name="scifact/claims_binary:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=True,
        prefix="classification"),
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["validation", "test"],
    metric_fns=[metrics.extractive_explanations_metric])

n_scifact_explanations = [500, 400, 300, 200, 100, 10]
for n in n_scifact_explanations:
  # Take n in train.
  TaskRegistry.add(
      "scifact_explanations_take{}_v011".format(n),
      t5.data.TfdsTask,
      tfds_name="scifact/claims_binary:0.1.1",
      splits={"train": "train[0:{}]".format(n)},
      text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=False,
        prefix="explain classification"),
      postprocess_fn=postprocessors.extractive_explanations,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "scifact_labels_skip{}_v011".format(n),
      t5.data.TfdsTask,
      tfds_name="scifact/claims_binary:0.1.1",
      splits={"train": "train[{}:]".format(n)},
      text_preprocessor=functools.partial(
        preprocessors.scifact,
        input_features=[('text', 'passage'), ('claim',)],
        drop_explanations=True,
        prefix="classification"),
      postprocess_fn=postprocessors.extractive_explanations,
      metric_fns=[])

# ======================== FEVER Transfer ======================

TaskRegistry.add(
    "fever_v011_transfer_no_expl",
    TfdsTask,
    tfds_name="fever:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.fever,
        input_features=[('passage',)],
        drop_explanations=True,
        prefix="classification"),
    token_preprocessor=preprocessors.fever_token,
    output_features=['inputs', 'targets', 'claim'],
    postprocess_fn=postprocessors.extractive_explanations,
    splits=['train'],
    metric_fns=[metrics.extractive_explanations_metric])

TaskRegistry.add(
    "fever_v011_transfer",
    TfdsTask,
    tfds_name="fever:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.fever,
        input_features=[('passage',)],
        drop_explanations=False,
        prefix="explain classification"),
    token_preprocessor=preprocessors.fever_token,
    output_features=['inputs', 'targets', 'claim'],
    postprocess_fn=postprocessors.extractive_explanations,
    splits=['train'],
    metric_fns=[metrics.extractive_explanations_metric])

TaskRegistry.add(
    "fever_eval_v011",
    TfdsTask,
    tfds_name="fever:0.1.1",
    text_preprocessor=functools.partial(
        preprocessors.fever,
        input_features=[('passage',)],
        drop_explanations=False,
        prefix="explain classification"),
    token_preprocessor=preprocessors.fever_token,
    output_features=['inputs', 'targets', 'claim'],
    postprocess_fn=postprocessors.extractive_explanations,
    splits=["validation", "test"],
    metric_fns=[metrics.extractive_explanations_metric])
