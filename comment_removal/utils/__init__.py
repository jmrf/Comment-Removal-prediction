import random
import torch
import time


from itertools import islice


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("Timeit - '{}' took "
              "{:.4f} seconds".format(method.__name__, te - ts))
        return result
    return timed


def configure_colored_logging(logger, loglevel='info'):
    import coloredlogs
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles['asctime'] = {}
    coloredlogs.install(
        logger=logger,
        level=loglevel,
        use_chroot=False,
        fmt="%(levelname)s - %(filename)s - %(lineno)s: %(message)s",
        field_styles=field_styles)


def chunk(iterable, c_size, stack_func=None):
    """
    Given an iterable yields chunks of size 'c_size'.
    The iterable can be an interator, we do not assume iterable to have
    'len' method.
    Args:
        iterable (iterable): to be partitioned in chunks
        c_size (int): size of the chunks to be produced
    Returns:
        (generator) of elements of size 'c_size' from the given iterable
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, c_size))
        if not chunk:
            return
        if stack_func:
            yield stack_func(chunk)
        else:
            yield chunk


def parallel_shuffle(*args):
    """
    Shuffle n lists concurrently.

    Args:
        *args: list of iterables to shuffle concurrently

    Returns:
        shuffled iterables
    """
    combined = list(zip(*args))
    random.shuffle(combined)
    args = zip(*combined)
    return [list(x) for x in args]


def parallel_split(split_ratio, *args):
    """
    Splits n lists concurrently

    Args:
        *args: list of iterables to split
        split_ratio (float): proportion to split the lists into two parts
    Returns:

    """
    all_outputs = []
    for a in args:
        split_idx = int(len(a) * split_ratio)
        all_outputs.append((a[:split_idx], a[split_idx:]))
    return all_outputs


def to_tensor(ndarray):
    """Converts a np.array into pytorch.tensor

    Args:
        ndarray (np.array): numpy array to convert to tensor
    """
    return torch.from_numpy(ndarray)


def compute_thresholds_ROC(test_labels, prob_vector, weights=None):
    """
    Computes ROC curve for each class and from the ROC curve the Younden's J statistic
    to find the optimal threshold per class.

    Args:
        test_labels (TYPE): Description
        prob_vector (TYPE): Description
        weights (None, optional): Description

    Returns:
        TYPE: Description
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(test_labels, prob_vector, drop_intermediate=True)
    # roc_auc = auc(fpr, tpr)
    # Compute optimal threshold per class (Younden index)
    # Maximize the sum: sentivity + specificity = tpr + 1 - fpr
    ratios = -fpr + tpr + 1
    younden_index = np.argmax(ratios)
    return thresholds[younden_index]
