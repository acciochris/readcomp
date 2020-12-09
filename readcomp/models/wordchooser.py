"""Several models for the wordchooser."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
import nltk


class HMMWordChooser(nltk.tag.hmm.HiddenMarkovModelTagger):
    """
    This is the implementation of a wordchooser based on the
    hidden markov model.
    """

    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        """
        Train a HMMWordChooser model.

        The parameters will be passed to the hmm train classmethod.
        """
        self._model = nltk.tag.hmm.HiddenMarkovModelTagger.train(*args, **kwargs)
        self.test = self._model.test
        self.predict = self._model.tag


class RNNWordChooser(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
    ):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [sent len, batch size]

        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pass embedhttp://nlp.stanford.edu/data/glove.6B.zipdings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))

        return predictions

    def fit(self, data, embeddings):
        model.embedding.weight.data.copy_(embeddings)
