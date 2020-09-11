"""Load data from database for training the word chooser."""

import torchtext
import os

raise RuntimeError("Not ready yet")


def load(base_dir):
    """Load data from files in base_dir.
    
    :param str base_dir: the directory containing data files
    :returns: a list of :class:`torchtext.Dataset` objects from the data
    :rtype: list"""
    for file in os.listdir(base_dir):
        with open(file) as f:
            passage = f.read()
            tokenizer = torchtext.data.get_tokenizer("basic_english")
            tokens = []
            tags = []
            for token in tokenizer(passage):
                selected = 0
                if token.startswith("_") and token.endswith("_"):
                    token = token[1:-1]
                    selected = 1
                tokens.append(token)
                tags.append(selected)
            dataset = None

