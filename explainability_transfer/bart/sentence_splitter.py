# Extended and adopted from https://github.com/huggingface/transformers/blob/8f07f5c44bf33f10b0075ce770b19de96ab389c0/examples/seq2seq/sentence_splitter.py

import re

from filelock import FileLock


try:
    import nltk
    nltk.data.path.append("/workspace/data/nltk")

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True, download_dir="/workspace/data/nltk")


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert NLTK_AVAILABLE, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))
