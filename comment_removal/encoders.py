import logging
import numpy as np

from tqdm import tqdm
from comment_removal.utils.text_processing import (tokenize,
                                                   remove_non_printable,
                                                   normalize_punct)
# TODO: FIX the pyBPE path mess!
from external.pyBPE.pybpe.pybpe import pyBPE


logger = logging.getLogger(__name__)


def preproc(input_text, lang, bpe):

        # normalize the string and remove non printable characters
        text = remove_non_printable(normalize_punct(
            input_text.replace("\n", "  "),
            lang
        ))
        # Tokenize the input
        tokenized = tokenize(text, lang=lang,
                             descape=False, lower_case=False)
        # BPE encode
        encoded = bpe.apply_bpe(tokenized)
        return encoded


class OneHotLabelEncoder():

    def __init__(self, x):
        from sklearn.preprocessing import OneHotEncoder

        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)

        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(x)

    def encode(self, x):
        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)
        return self.encoder.transform(x).todense()

    def get_labels(self):
        return self.encoder.categories_[0]


class IndexEncoder():

    def __init__(self, x):
        from sklearn import preprocessing

        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)

        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(x)

    def encode(self, x):
        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)
        return self.encoder.transform(x)

    def get_labels(self):
        return self.encoder.classes_


class LaserEncoder():

    """LaserEncoder for the Stance dataset.
    Given a batch of sentences encode the batch as a matrix (N x E)
    using Language Agnostic SEntence Representations:
    https://arxiv.org/abs/1812.10464
    """

    HANDLER_REGEX = r"@[\w\d_-]+"

    def __init__(self, args, lang='en', lower_case=False, descape=False):
        """Text corpus encoder using LASER: https://arxiv.org/abs/1812.10464

        Args:
            args ([type]): [description]
            lang (str, optional): [description]. Defaults to 'en'.
            lower_case (bool, optional): [description]. Defaults to False.
            descape (bool, optional): [description]. Defaults to False.
        """

        # TODO: Currently the external Laser lib. requires an args object.
        #       Change for explicit parameters!
        from external.encoders.laser import EncodeLoad

        # configure path from 'args'
        self.workdir = args.workdir
        self.bpe_codes_file = args.bpe_codes
        self.vocab_file = args.vocab_file

        # BPE encoding
        self.bpe = pyBPE(self.vocab_file, self.bpe_codes_file)
        self.bpe.load()

        # load the LASER sentence encoder
        self.encoder = EncodeLoad(args)

        # tokenization configuration
        self.lang = lang
        self.lower_case = lower_case
        self.descape = descape

    def encode(self, corpus, parallel=False):
        """Encodes a given corpus using LASER encoding.

        Args:
            corpus (iterable): of strings composing the text corpus to encde
        """
        # preprocess all corpus examples:
        N = len(corpus)
        logger.info("Preprocessing text corpus ({})".format(N))
        preproc_questions = []

        if parallel:
            from multiprocessing.pool import ThreadPool

            pool = ThreadPool(4)

            with tqdm(total=N) as progress_bar:
                for e in pool.imap(self._preproc, corpus):
                    preproc_questions.append(e)
                    progress_bar.update(1)  # update progress

            pool.close()
        else:
            preproc_questions = [self._preproc(text) for text in tqdm(corpus)]

        # LASER encode
        logger.info("Laser encoding text corpus")
        mat = self.encoder.encode_sentences(preproc_questions)
        return mat

    def _preproc(self, input_text):

        # normalize the string and remove non printable characters
        text = remove_non_printable(normalize_punct(
            input_text.replace("\n", "  "),
            self.lang
        ))
        # Tokenize the input
        tokenized = tokenize(text, lang=self.lang,
                             descape=False, lower_case=False)
        # BPE encode
        encoded = self.bpe.apply_bpe(tokenized)
        return encoded
