import logging
import numpy as np

from tqdm import tqdm
from gensim import corpora, models
from comment_removal.utils.text_processing import (tokenize,
                                                   tokenize_preproc,
                                                   remove_non_printable,
                                                   normalize_punct)
# TODO: FIX the pyBPE path mess!
from external.pyBPE.pybpe.pybpe import pyBPE


logger = logging.getLogger(__name__)


class MyCorpus(corpora.TextCorpus):

        def get_texts(self):
            for q in self.input:
                words = tokenize_preproc(q)
                yield words


class LSIEncoder:

    def __init__(self, dict_no_below=1,
                 dict_no_above=1, keep_n=5000, num_topics=300):
        self.dictionary = None
        self.tf_idf_model = None
        self.lsi_model = None
        # encoder parameters
        self.dict_no_below = dict_no_below
        self.dict_no_above = dict_no_above
        self.keep_n = keep_n
        self.num_topics = num_topics

    def fit(self, text_corpus, y=None):
        """
        Creates and trains a tf-idf + LSI model
        Args:
            corpus (list): List of strings representing the sentences to embed.
        """
        # Create a dictionary with all the documents (train_questions)
        corpus = MyCorpus(text_corpus)

        logger.info("Creating dictionary...")
        corpus.dictionary.filter_extremes(self.dict_no_below,
                                          self.dict_no_above, self.keep_n)
        self.dictionary = corpus.dictionary

        # create tf-idf model
        logger.info("Creating tf-idf model...")
        self.tf_idf_model = models.tfidfmodel.TfidfModel(corpus)

        # doc2bow transformation
        logger.info("Doc2bow transform of each document...")
        embeddings = []
        for i in range(len(text_corpus)):
            q_words = tokenize_preproc(text_corpus[i])
            embeddings.append(self.dictionary.doc2bow(q_words))

        # compute the tf-idf transform
        logger.info("tf-idf transform of each document..")
        for i in range(len(embeddings)):
            embeddings[i] = self.tf_idf_model[embeddings[i]]

        # Create the LSI model
        logger.info("Creating LSI model...")
        self.lsi_model = models.lsimodel.LsiModel(embeddings,
                                                  num_topics=self.num_topics,
                                                  id2word=self.dictionary)

        # compute the actual LSI vectors
        logger.info("Computing the LSI embeddings...")
        for i in range(len(embeddings)):
            embeddings[i] = [x[1] for x in self.lsi_model[embeddings[i]]]
            if len(embeddings[i]) < self.num_topics:
                embeddings[i] = [0.0] * self.num_topics

        logger.info("Total embeddings: {}".format(len(embeddings)))
        assert all([len(embeddings[i]) == len(embeddings[0])
                    for i in range(len(embeddings))])
        return self

    def transform(self, sentences, y=None):
        """Summary

        Args:
            sentences (list): List of strings representing the
            sentences to embed.

        Returns:
            (numpy.array): LSI embedded sentences as numpy arrays
        """
        embeddings = [[0.0] * self.num_topics for i in range(len(sentences))]
        for i in range(len(sentences)):
            words = tokenize_preproc(sentences[i])
            embeddings[i] = self.tf_idf_model[self.dictionary.doc2bow(words)]
            embeddings[i] = [x[1] for x in self.lsi_model[embeddings[i]]]
            if len(embeddings[i]) < self.num_topics:
                embeddings[i] = [0.0] * self.num_topics
        return np.array(embeddings)

    def fit_transform(self, sentences):
        self.fit(sentences)
        return self.transform(sentences)

    def save(self, dict_path, tfidf_path, lsi_path):
        if not self.dictionary or not self.tf_idf_model or not self.lsi_model:
            raise Exception("dictionary, tf-idf model or LSI not build yet...")
        self.dictionary.save(dict_path)
        self.tf_idf_model.save(tfidf_path)
        self.lsi_model.save(lsi_path)

    def load(self, dict_path, tfidf_path, lsi_path):
        logger.info("Loading dictionary, tf-idf and lsi models...")
        self.dictionary = corpora.TextCorpus.load(dict_path)
        self.tf_idf_model = models.tfidfmodel.TfidfModel.load(tfidf_path)
        self.lsi_model = models.lsimodel.LsiModel.load(lsi_path)


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
        from external.models.laser import EncodeLoad

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
