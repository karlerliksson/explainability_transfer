"""fever dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@unpublished{eraser2019,
    title = {ERASER: A Benchmark to Evaluate Rationalized NLP Models},
    author = {Jay DeYoung and Sarthak Jain and Nazneen Fatema Rajani and Eric Lehman and Caiming Xiong and Richard Socher and Byron C. Wallace}
}
@inproceedings{MultiRC2018,
    author = {Daniel Khashabi and Snigdha Chaturvedi and Michael Roth and Shyam Upadhyay and Dan Roth},
    title = {Looking Beyond the Surface:A Challenge Set for Reading Comprehension over Multiple Sentences},
    booktitle = {NAACL},
    year = {2018}
}
"""

_DESCRIPTION = """
FEVER dataset for Dact Extraction and Verification. From the ERASER benchmark.
"""

_DOWNLOAD_URL = 'https://www.eraserbenchmark.com/zipped/fever.tar.gz'


class Fever(tfds.core.GeneratorBasedBuilder):
  """FEVER dataset for Dact Extraction and Verification. From the ERASER benchmark."""

  VERSION = tfds.core.Version('0.1.1')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'passage': tfds.features.Text(),
            'claim': tfds.features.Text(),
            'label': tfds.features.ClassLabel(names=['False', 'True']),
            'evidences': tfds.features.Sequence(tfds.features.Text())
        }),
        supervised_keys=None,
        homepage='https://github.com/awslabs/fever',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)
    data_dir = os.path.join(dl_dir, 'fever')
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'data_dir': data_dir,
                        'filepath': os.path.join(data_dir, 'train.jsonl')},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'data_dir': data_dir,
                        'filepath': os.path.join(data_dir, 'val.jsonl')},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'data_dir': data_dir,
                        'filepath': os.path.join(data_dir, 'test.jsonl')},
        ),
    ]

  def _format_text(self, text):
    # Fix quotations
    new_text = ''
    left = False
    skip_next = False
    len_text = len(text)
    for i, c in enumerate(text):
        if c == '"':
            left = not(left)       
            if left and i<(len_text-1) and text[i+1] == ' ':
                skip_next = True
            elif not left and i>0 and text[i-1] == ' ':
                new_text = new_text[:-1]

            new_text += c
        else:
            if skip_next:
                skip_next = False
            else:
                new_text += c

    return (new_text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" '","'")
            .replace(" :", ":").replace(" ;",";").replace(" = ","=").replace("( ","(").replace(" )",")")).replace('`` ','"').replace("''",'"').replace("-LRB- ", "(").replace("-LSB- ", "[").replace(" -RRB-", ")").replace(" -RSB-", "]")

  def _generate_examples(self, data_dir, filepath):
    """Yields examples."""

    fever_dir = os.path.join(data_dir, 'docs')
    with tf.io.gfile.GFile(filepath) as f:
      for line in f:
        row = json.loads(line)
        evidences = []

        for evidence in row['evidences'][0]:
          docid = evidence['docid']
          evidences.append(self._format_text(evidence['text']))

        passage_file = os.path.join(fever_dir, docid)
        with tf.io.gfile.GFile(passage_file) as f1:
          passage_text = f1.read()

        yield row['annotation_id'], {
            'passage': self._format_text(passage_text),
            'claim': self._format_text(row['query']),
            'label': 'True' if row['classification'] == 'SUPPORTS' else 'False',
            'evidences': evidences
        }

