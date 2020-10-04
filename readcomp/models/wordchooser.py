"""Several models for the wordchooser."""

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
