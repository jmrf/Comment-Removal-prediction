import logging
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve, auc
from comment_removal.utils.plotting import plot_roc


logger = logging.getLogger(__name__)


def compute_roc_curve(y_test, y_score, n_classes=2):

    y_test_ = np.zeros((y_test.shape[0], 2))
    y_test_[y_test] = 1

    roc_score = roc_auc_score(y_test, np.max(y_score, axis=1))
    logger.info("ROC AUC score: {}".format(roc_score))

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plot_roc(fpr, tpr, roc_auc, cls=2)
