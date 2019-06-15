# coding: utf-8

import numpy as np

from external.pyBPE.pybpe.pybpe import pyBPE
from external.encoders.laser import EncodeLoad
from comment_removal.utils.loaders import RedditDataLoader
from comment_removal.utils.text_processing import (tokenize,
                                                   remove_non_printable,
                                                   normalize_punct)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def preproc_text(input_text, lang, bpe):
    # preprocess text
    text = remove_non_printable(normalize_punct(
        input_text.replace("\n", "  "), lang)
    )
    # tokenization
    tokenized = tokenize(text, lang=lang, descape=False, lower_case=False)
    # apply bpe encoding
    return bpe.apply_bpe(tokenized)


# load the embedded test comments
test_set_embedded = np.load("workdir/test_laser-comments.npy")
N, dim = test_set_embedded.shape

# load the raw data
loader = RedditDataLoader("./data/reddit_train.csv",
                          "./data/reddit_test.csv")
test_text = loader.get('BODY', 'test')

# load the BPE encoder
bpe = pyBPE("external/models/LASER/93langs.fvocab",
            "external/models/LASER/93langs.fcodes")

bpe2 = pyBPE("external/models/LASER/eparl21.fvocab",
             "external/models/LASER/eparl21.fcodes")
bpe.load()
bpe2.load()

# preprocessed text with different vocab and codes
t0 = preproc_text(test_text[0], 'en', bpe)
t0_ = preproc_text(test_text[0], 'en', bpe2)

args = dotdict({
    'max_sentences': None,
    'buffer_size': 100,
    'max_tokens': 12000,
    'cpu': False,
    'encoder': "external/models/LASER/bilstm.93langs.2018-12-26.pt"
})

# Load the LASER encoder
encoder = EncodeLoad(args)

# check a few random embeddings
for idx in np.random.choice(range(N), 10):
    t = preproc_text(test_text[idx], 'en', bpe)
    encoded = encoder.encode_sentences([t])
    assert np.isclose(
        test_set_embedded[idx],
        encoded,
        atol=1e-6
    ).all(), "Idx={} differs in {}".format(
        idx,
        np.mean(test_set_embedded[idx] - encoded)
    )
