# coding=utf-8
#
# Extended and adopted from https://github.com/google-research/google-research/blob/master/wt5/wt5/mixtures.py
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

"""WT5 mixtures."""
import functools
from . import tasks

import t5

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry


def _rate_num_movies(task, scale=1.0):
  del task
  return scale * 125000000.0


def _rate_num_input_examples(task):
  if "train" in task.splits:
    return float(task.num_input_examples("train"))
  elif "validation" in task.splits:
    return float(task.num_input_examples("validation"))
  else:
    raise ValueError("Task %s does not have a train or validation split." % (
        task.name))


# ----------------------------- MultiRC Explainability Pre-Training ---------------------------------
MixtureRegistry.add(
    "multi_rc_transfer_expl",
    [("eraser_multi_rc_v011_transfer", 1.0),
     ("eraser_multi_rc_v011_transfer_no_expl", 1.0)],
)

# ----------------------------- FEVER Explainability Pre-Training ---------------------------------
MixtureRegistry.add(
    "fever_transfer_expl",
    [("fever_v011_transfer", 1.0),
     ("fever_v011_transfer_no_expl", 1.0)],
)

# ----------------------------- SciFact ---------------------------------
for n in tasks.n_scifact_explanations:
  scifact_size = 564
  scifact_n_explanations_tasks = [
      ("scifact_explanations_take{}_v011".format(n), n),
      ("scifact_labels_skip{}_v011".format(n), scifact_size-n),
  ]
  MixtureRegistry.add(
      "scifact_rationales_{}_explanations".format(n),
      scifact_n_explanations_tasks
  )

MixtureRegistry.add(
    "scifact_simple_mixture",
    [("scifact_v011_no_expl", 1.0),
     ("scifact_v011", 1.0)],
)

MixtureRegistry.add(
    "scifact_balanced_v011",
    [("scifact_true_v011", 1.0), # Upsample false class so that the scifact dataset is balanced
     ("scifact_false_v011", 1.0)],
)

MixtureRegistry.add(
    "scifact_balanced_v011_no_expl",
    [("scifact_true_v011_no_expl", 1.0),
     ("scifact_false_v011_no_expl", 1.0)],
)
