# coding=utf-8
#
# Extended and adopted from https://github.com/google-research/google-research/blob/master/wt5/wt5/preprocessors.py
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

"""WT5 preprocessors."""
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import re
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from torch.nn import CosineSimilarity
from spacy.lang.en import English
import numpy as np
from t5.models.hf_model import new_py_function


def _explanation_targets(answer, explanations, prefix='explanation:'):
  # Add prefix before each explanation.
  return tf.strings.reduce_join(
      tf.concat([[answer], explanations], axis=0),
      separator=' %s ' % prefix)


def extractive_explanations(
    dataset,
    prefix='explain sentiment',
    input_feature='review',
    output_classes=('negative', 'positive'),
    drop_explanations=False
    ):
  """Preprocessor to handle extractive rationale prediction datasets.

  The preprocessor expects a dataset with the provided 'input_feature', a label,
  and a list of evidences. E.g. the movie rationale dataset consists of the
  following features.

  {
    review: 'This is a bad movie. Skip it.'
    label: 0,
    evidences: ['bad movie', 'Skip it']
  }

  The example will be transformed to the following format by the preprocessor:
  {
    inputs: 'explain sentiment review: This is a bad movie. Skip it.'
    targets: 'NEG because bad movie explanation: Skip it'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    input_feature: str, feature name in input dataset.
    output_classes: list of output classes in the input dataset. Defaults to
      ['negative', 'positive'] for the movie reviews dataset.
    drop_explanations: bool, whether or not to drop explanations.

  Returns:
    a tf.data.Dataset

  """

  if output_classes is None:
    output_classes = ['negative', 'positive']

  def my_fn(x):
    """Helper function to transform a rationale dataset to inputs/targets."""
    input_label = tf.strings.join([input_feature, ':'], separator='')
    inputs = tf.strings.join(
        [prefix, input_label, x[input_feature]], separator=' ')

    class_label = tf.gather(output_classes, x['label'])
    if drop_explanations:
      targets = class_label
    else:
      targets = _explanation_targets(class_label, x['evidences'])

    return {'inputs': inputs, 'targets': targets}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def eraser_multi_rc(
    dataset,
    prefix='explain multirc',
    input_features=None,
    explanation_separator='explanation:',
    drop_explanations=False,):
  """Preprocessor to handle ERASER MultiRC dataset.

  The preprocessor expects a dataset with the provided `input_features`, a
  label, and a list of evidences. The eraser_multi_rc dataset consists of the
  following features.

  {
    'passage': 'This is a passage. It has sentences',
    'query_and_answer': 'Is this a passage? || yes',
    'label': 'True',
    'evidences': ['This is a passage', 'It has sentences']
  }

  The example will be transformed to the following format by the preprocessor:
  {
    'inputs': 'explain multirc passage: This is a passage. It has sentences '
    'query': Is this a passage? answer: yes'
    'targets': 'True explanation: This is a passage. explanation: It has
                sentences'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, label to prepend to the inputs.
    input_features: list, feature name in input dataset.
    explanation_separator: str, separator string for the list of rationales.
    drop_explanations: bool, whether or not to drop explanations.

  Returns:
    a tf.data.Dataset

  """
  if not input_features:
    input_features = ['passage', 'query', 'answer']

  def my_fn(x):
    """Helper function to transform a eraser_multirc dataset to inputs/targets."""

    # Separate out query and answer components
    split_query_answer = tf.strings.split(x['query_and_answer'], '||').values
    x['query'] = tf.strings.strip(tf.slice(split_query_answer, [0], [1]))
    x['answer'] = tf.strings.strip(tf.slice(split_query_answer, [1], [1]))

    # Creating inputs
    inputs = prefix
    for input_feature in input_features:
      ip_feat_str = tf.strings.join([input_feature+':', x[input_feature]],
                                    separator=' ')
      inputs = tf.strings.join([inputs, ip_feat_str], ' ')

    # Creating targets
    class_label = tf.gather(['False', 'True'], x['label'])
    if drop_explanations:
      targets = class_label
    else:
      targets = _explanation_targets(
          class_label,
          x['evidences'],
          prefix=explanation_separator)

    return {'inputs': inputs[0], 'targets': targets}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def scifact(
    dataset,
    prefix='explain classification',
    input_features=None,
    explanation_separator='explanation:',
    drop_explanations=False,):
  if not input_features:
    input_features = [('text', 'passage'), ('claim',)]

  def my_fn(x):
    """Helper function to transform a scifact dataset to inputs/targets."""
    # Creating inputs
    inputs = prefix
    for input_feature in input_features:
      ip_feat_str = tf.strings.join([input_feature[-1]+':', x[input_feature[0]]],
                                    separator=' ')
      inputs = tf.strings.join([inputs, ip_feat_str], ' ')

    # Creating targets
    class_label = tf.gather(['False', 'True'], x['label'])
    if drop_explanations:
      targets = class_label
    else:
      targets = _explanation_targets(
          class_label,
          x['evidences'],
          prefix=explanation_separator)

    return {'inputs': inputs, 'targets': targets}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def fever(
    dataset,
    prefix='explain classification',
    input_features=None,
    explanation_separator='explanation:',
    drop_explanations=False,):
  if not input_features:
    input_features = [('passage',),]

  def my_fn(x):
    """Helper function to transform a fever dataset to inputs/targets."""
    # Creating inputs
    inputs = prefix
    for input_feature in input_features:
      ip_feat_str = tf.strings.join([input_feature[-1]+':', x[input_feature[0]]],
                                    separator=' ')
      inputs = tf.strings.join([inputs, ip_feat_str], ' ')

    # Creating targets
    class_label = tf.gather(['False', 'True'], x['label'])
    if drop_explanations:
      targets = class_label
    else:
      targets = _explanation_targets(
          class_label,
          x['evidences'],
          prefix=explanation_separator)

    claim = tf.strings.join(['claim: ', x['claim']])
    return {'inputs': inputs, 'targets': targets, 'claim': claim}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def fever_token(dataset, sequence_length, output_features=None):
    # Make sure claim is not truncated if input exceeds sequence length
    def my_fn(x):
        trunc_ind = sequence_length['inputs'] - len(x['claim']) - 1
        inputs = tf.concat([x['inputs'][:trunc_ind], x['claim']], axis=0)
        x['inputs'] = inputs
        x['inputs_plaintext'] = tf.strings.join([x['inputs_plaintext'], x['claim_plaintext']], separator=' ')
        return x
    return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
 