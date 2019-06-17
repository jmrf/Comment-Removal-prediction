import string


def remove_non_printable(text):
    return ''.join(s for s in text
                   if s in string.printable)


def normalize_punct(text, lang):
    """ Normalize punctuation of a given string
    Punctuation normalizer:
    https://bitbucket.org/luismsgomes/mosestokenizer
    """
    from mosestokenizer import MosesPunctuationNormalizer
    with MosesPunctuationNormalizer(lang) as normalize:
        return normalize(text)


def string_descape(text):
    """ Descape a string with escaped characters """
    return text.decode('string_escape')


def string_unescape(s):
    return (s.replace("&lt;", "<")
             .replace("&apos;", "'")
             .replace("&gt;", ">")
             .replace("&quot;", "\"")
             .replace("&amp;", "&")
             .replace("@-@", "-"))


def _moses_tokenize(text, lang):
    """ Tokenize a given string using moses tokenizer
    Tokenization: https://github.com/alvations/sacremoses
    """
    from sacremoses import MosesTokenizer
    mt = MosesTokenizer(lang)
    return [string_unescape(t)
            for t in mt.tokenize(text)]


def tokenize(text, lang='en', lower_case=True, descape=False):
    """Tokenize an input string and return the preprocessed output string.
    Instead of using the external script using python libraries.
    """
    # normalize punct and remove non printable characters
    text = normalize_punct(text, lang)

    # remove non printable chars
    text = remove_non_printable(text)

    if descape:
        text = string_descape(text)

    # moses tokenizer
    text = " ".join(_moses_tokenize(text, lang))

    return text


def tokenize_preproc(sentence, stemm=False, lemm=False):
    """
    Tokenizes a sentence and applies optionaly stemming and lemming

    Args:
        sentence (str): Sentence to tokenize
        stemm (bool, optional): whether to apply stemming or not
        lemm (bool, optional): whether to apply lemming or not

    Returns:
        list: of words
    """
    from nltk.tokenize import word_tokenize
    words = word_tokenize(sentence)
    if lemm:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(x) for x in words]
    if stemm:
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(x) for x in words]
    return words


def clean_text(text, remove_stopwords=True, remove_punct=True, stem_words=False):
    """
    Clean the text, with the option to remove stopwords, remove punctuation and to stem words.
    The function is an adaptation from:
    https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text

    Args:
        text (str): text to clean
        remove_stopwords (bool, optional): Remove stop words
        remove_punct (bool, optional): Remove punctuation signs
        stem_words (bool, optional): Apply stemming

    Returns:
        str: clean string text based on the above cleaning options
    """
    import re
    from nltk.stem import SnowballStemmer

    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        from nltk.corpus import stopwords
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text
