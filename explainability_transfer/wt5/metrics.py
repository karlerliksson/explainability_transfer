# coding=utf-8
#
# Extended and adopted from https://github.com/google-research/google-research/blob/master/wt5/wt5/metrics.py
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

"""WT5 metrics."""

import numpy as np
import sklearn.metrics
import t5.evaluation


def extractive_explanations_metric(targets, predictions):
  """Compute label accuracy and macro F1 score for explanations."""

  def get_labels_spans_and_expls(answers):
    """Gets a list of labels and spans from a list of dicts."""
    labels = []
    spans = []
    span_arrays = []
    explanations = []

    for answer in answers:
      for key, value in answer.items():
        if key == "label":
          labels.append(value)
        elif key == "overlap_spans":
          spans.append(value)
        elif key == "span_array":
          span_arrays.append(value)
        elif key == "explanations":
          explanations.append(value)
        else:
          pass
          #raise ValueError("Unexpected key found in answers dict: %s" % key)

    return labels, spans, span_arrays, explanations

  labels_t, spans_t, arrays_t, _ = get_labels_spans_and_expls(targets)
  labels_p, spans_p, arrays_p, explns_p = get_labels_spans_and_expls(
      predictions)

  # Compute f1 score for each example in the target prediction pair
  f1_scores = []
  f1_scores_correct = [] # Only count explanation as correct if the label is correct
  f1_scores_gt_expl = [] # Only eval explanations if there is a ground truth explanation
  f1_scores_gt_expl_correct = []
  for gt_span, pred_span, gt_label, pred_label in zip(spans_t, spans_p, labels_t, labels_p):
    elem_prec = len(set(gt_span)
                    & set(pred_span)) / len(pred_span) if pred_span else 0
    elem_rec = len(set(gt_span)
                   & set(pred_span)) / len(gt_span) if gt_span else 0

    if elem_prec == 0 or elem_rec == 0:
      elem_f1 = 0
    else:
      elem_f1 = 2 * elem_prec * elem_rec / (elem_prec + elem_rec)
    f1_scores.append(elem_f1)
    f1_scores_correct.append(elem_f1 if gt_label == pred_label else 0)
    if len(gt_span) > 0:
        f1_scores_gt_expl.append(elem_f1)
        f1_scores_gt_expl_correct.append(elem_f1 if gt_label == pred_label else 0)

  exact_match_f1 = np.mean(f1_scores) * 100
  exact_match_f1_correct = np.mean(f1_scores_correct) * 100
  exact_match_f1_gt_expl = np.mean(f1_scores_gt_expl) * 100
  exact_match_f1_gt_expl_correct = np.mean(f1_scores_gt_expl_correct) * 100

  partial_match_f1 = 100 * np.mean(
      [sklearn.metrics.f1_score(t, p) for t, p in zip(arrays_t, arrays_p)]
  )
  partial_match_f1_gt_expl = 100 * np.mean(
      [sklearn.metrics.f1_score(t, p) for t, p in zip(arrays_t, arrays_p) if sum(t)]
  )

  def get_avg_num_explanations(explanations):
    total_explns = 0
    for e in explanations:
      total_explns += len(e)
    return float(total_explns)/len(explanations) if explanations else 0.0

  return {
      "accuracy": 100 * sklearn.metrics.accuracy_score(labels_t, labels_p),
      "f1a": 100 * sklearn.metrics.f1_score([1 if l == "True" else 0 for l in labels_t],
                                            [1 if l == "True" else 0 for l in labels_p]),
      "f1": exact_match_f1,
      "f1 correctly classified": exact_match_f1_correct,
      "f1 gt expl": exact_match_f1_gt_expl,
      "f1 correctly classified gt expl": exact_match_f1_gt_expl_correct,
      "partial match f1": partial_match_f1,
      "partial match f1 gt expl": partial_match_f1_gt_expl,
      "avg_explanation_count": get_avg_num_explanations(explns_p),
  }
