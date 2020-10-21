"""Load data from database for training the wordchooser."""

import os
import re

from readcomp.utils import tokenize


class WordChooserDataset(torch.utils.data.IterableDataset):
    """
    This class represents a dataset for the wordchooser model.

    :param str filename: a file or a directory containing data files
    :param callable tokenizer: the tokenizer to use
    (default: torchtext basic_english)
    """

    def __init__(self, filename, tokenizer=None):
        self._data = []
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = tokenize

        if os.path.isdir(filename):
            for file in os.path.listdir(filename):
                self._load(file)
        else:
            self._load(filename)

    def _load(self, file):
        data = []
        with open(file) as f:
            for passage in f.readlines():
                for token in self.tokenizer(passage):
                    selected = 0
                    match = re.fullmatch(r"_(.+)_", token)
                    if match:
                        token = match.group(1)
                        selected = 1
                    data.append((token, selected))
        self._data.append(data)

    def __iter__(self):
        yield from self._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]