"""scifact dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

from enum import Enum
import json
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import textwrap


_CITATION = """
@misc{wadden2020fact,
    title={Fact or Fiction: Verifying Scientific Claims},
    author={David Wadden and Shanchuan Lin and Kyle Lo and Lucy Lu Wang and Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
    year={2020},
    eprint={2004.14974},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """
SciFact dataset for verification of scientific claims.
"""

_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


####################

# Utility functions and enums.


def load_jsonl(fname):
    return [json.loads(line) for line in tf.io.gfile.GFile(fname)]


class Label(Enum):
    SUPPORTS = 1
    NEI = 0
    REFUTES = -1


def make_label(label_str):
    lookup = {"SUPPORT": Label.SUPPORTS,
              "NOT_ENOUGH_INFO": Label.NEI,
              "CONTRADICT": Label.REFUTES}
    assert label_str in lookup
    return lookup[label_str]


####################

# Representations for the corpus and abstracts.

@dataclass(repr=False, frozen=True)
class Document:
    id: str
    title: str
    sentences: Tuple[str]

    def __repr__(self):
        return self.title.upper() + "\n" + "\n".join(["- " + entry for entry in self.sentences])

    def __lt__(self, other):
        return self.title.__lt__(other.title)

    def dump(self):
        res = {"doc_id": self.id,
               "title": self.title,
               "abstract": self.sentences,
               "structured": self.is_structured()}
        return json.dumps(res)


@dataclass(repr=False, frozen=True)
class Corpus:
    """
    A Corpus is just a collection of `Document` objects, with methods to look up
    a single document.
    """
    documents: List[Document]

    def __repr__(self):
        return f"Corpus of {len(self.documents)} documents."

    def __getitem__(self, i):
        "Get document by index in list."
        return self.documents[i]

    def get_document(self, doc_id):
        "Get document by ID."
        res = [x for x in self.documents if x.id == doc_id]
        assert len(res) == 1
        return res[0]


####################

# Gold dataset.

class GoldDataset:
    """
    Class to represent a gold dataset, include corpus and claims.
    """
    def __init__(self, corpus_file, data_file):
        self.corpus = self._read_corpus(corpus_file)
        self.claims = self._read_claims(data_file)

    def __repr__(self):
        msg = f"{self.corpus.__repr__()} {len(self.claims)} claims."
        return msg

    def __getitem__(self, i):
        return self.claims[i]

    def _read_corpus(self, corpus_file):
        "Read corpus from file."
        corpus = load_jsonl(corpus_file)
        documents = []
        for entry in corpus:
            doc = Document(entry["doc_id"], entry["title"], entry["abstract"])
            documents.append(doc)

        return Corpus(documents)

    def _read_claims(self, data_file):
        "Read claims from file."
        examples = load_jsonl(data_file)
        res = []
        for this_example in examples:
            entry = copy.deepcopy(this_example)
            entry["release"] = self
            if "cited_doc_ids" in entry:
                entry["cited_docs"] = [self.corpus.get_document(doc)
                                       for doc in entry["cited_doc_ids"]]
                assert len(entry["cited_docs"]) == len(entry["cited_doc_ids"])
                del entry["cited_doc_ids"]
            else:
                entry["cited_docs"] = []
            if "evidence" not in entry:
                entry["evidence"] = {}
            res.append(Claim(**entry))

        res = sorted(res, key=lambda x: x.id)
        return res

    def get_claim(self, example_id):
        "Get a single claim by ID."
        keep = [x for x in self.claims if x.id == example_id]
        assert len(keep) == 1
        return keep[0]


@dataclass
class EvidenceAbstract:
    "A single evidence abstract."
    id: int
    label: Label
    rationales: List[List[int]]


@dataclass(repr=False)
class Claim:
    """
    Class representing a single claim, with a pointer back to the dataset.
    """
    id: int
    claim: str
    evidence: Dict[int, EvidenceAbstract]
    cited_docs: List[Document]
    release: GoldDataset

    def __post_init__(self):
        self.evidence = self._format_evidence(self.evidence)

    @staticmethod
    def _format_evidence(evidence_dict):
        # This function is needed because the data schema is designed so that
        # each rationale can have its own support label. But, in the dataset,
        # all rationales for a given claim / abstract pair all have the same
        # label. So, we store the label at the "abstract level" rather than the
        # "rationale level".
        res = {}
        for doc_id, rationales in evidence_dict.items():
            doc_id = int(doc_id)
            labels = [x["label"] for x in rationales]
            if len(set(labels)) > 1:
                msg = ("In this SciFact release, each claim / abstract pair "
                       "should only have one label.")
                raise Exception(msg)
            label = make_label(labels[0])
            rationale_sents = [x["sentences"] for x in rationales]
            this_abstract = EvidenceAbstract(doc_id, label, rationale_sents)
            res[doc_id] = this_abstract

        return res

    def __repr__(self):
        msg = f"Example {self.id}: {self.claim}"
        return msg

    def pretty_print(self, evidence_doc_id=None, file=None):
        "Pretty-print the claim, together with all evidence."
        msg = self.__repr__()
        print(msg, file=file)
        # Print the evidence
        print("\nEvidence sets:", file=file)
        for doc_id, evidence in self.evidence.items():
            # If asked for a specific evidence doc, only show that one.
            if evidence_doc_id is not None and doc_id != evidence_doc_id:
                continue
            print("\n" + 20 * "#" + "\n", file=file)
            ev_doc = self.release.corpus.get_document(doc_id)
            print(f"{doc_id}: {evidence.label.name}", file=file)
            for i, sents in enumerate(evidence.rationales):
                print(f"Set {i}:", file=file)
                kept = [sent for i, sent in enumerate(ev_doc.sentences) if i in sents]
                for entry in kept:
                    print(f"\t- {entry}", file=file)


class ScifactConfig(tfds.core.BuilderConfig):
  """BuilderConfig for SciFact dataset."""

  @tfds.core.disallow_positional_args
  def __init__(self, **kwargs):
    """BuilderConfig for Hallmarks dataset."""
    super(ScifactConfig, self).__init__(
        version=tfds.core.Version("0.1.1"),
        **kwargs)

class Scifact(tfds.core.GeneratorBasedBuilder):
  """SciFact dataset for verification of scientific claims."""
  BUILDER_CONFIGS = [
      ScifactConfig(
          name="claims_binary",
          description=textwrap.dedent("""\
            Only claims with a supporting or contradicting abstract.""")),
      ScifactConfig(
          name="true",
          description=textwrap.dedent("""\
            Only claims with a supporting abstract.""")),
      ScifactConfig(
          name="false",
          description=textwrap.dedent("""\
            Only claims with a contradicting abstract.""")),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'claim': tfds.features.Text(),
            'text': tfds.features.Text(),
            'label': tfds.features.ClassLabel(names=['False', 'True']),
            'evidences': tfds.features.Sequence(tfds.features.Text()),
        }),
        supervised_keys=None,
        homepage='https://github.com/allenai/scifact',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators.""" 
    data_dir = dl_manager.download_and_extract(_URL)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'data_dir': data_dir,
                'split': 'train'
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'data_dir': data_dir,
                'split': 'validation'
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'data_dir': data_dir,
                'split': 'test'
            },
        ),
    ]

  def _generate_examples(self, data_dir, split):
    """Yields examples."""
    if split == "train":
        data = GoldDataset(os.path.join(data_dir, "data", "corpus.jsonl"),
                           os.path.join(data_dir, "data", "claims_train.jsonl"))
    elif split == "validation":
        data = GoldDataset(os.path.join(data_dir, "data", "corpus.jsonl"),
                           os.path.join(data_dir, "data", "claims_dev.jsonl"))
    elif split == "test":
        data = GoldDataset(os.path.join(data_dir, "data", "corpus.jsonl"),
                           os.path.join(data_dir, "data", "claims_test.jsonl"))
    else:
        raise ValueError("split name not recognized!")

    for claim in data.claims:
        if split != "test":
            for doc_id in claim.evidence.keys():
                ev = claim.evidence[doc_id]
                if ev.label == Label.SUPPORTS:
                    label = "True"
                elif ev.label == Label.REFUTES:
                    label = "False"
                else:
                    continue
                    
                if self.builder_config.name == 'true' and label != "True":
                    continue
                elif self.builder_config.name == 'false' and label != "False":
                    continue

                doc = data.corpus.get_document(doc_id)
                evidences = []
                for i, sents in enumerate(ev.rationales):
                    evidences += [sent.strip() for i, sent in enumerate(doc.sentences) if i in sents]

                yield "{}_{}".format(claim.id, doc_id), {
                    'claim': claim.claim,
                    'text': " ".join([sent.strip() for sent in doc.sentences]),
                    'label': label,
                    'evidences': evidences,
                }
        else:
            # TODO: fix test split for true and false subsets...

            yield "{}".format(claim.id), {
                'claim': claim.claim,
                'text': "",
                'label': "False", # TODO: Consider adding an UNK label
                'evidences': [],
            }