import logging

from sklearn.metrics import roc_auc_score, roc_curve, auc
from comment_removal.utils.plotting import plot_roc


logger = logging.getLogger(__name__)


def compute_roc_curve(y_test, y_score):

    roc_score = roc_auc_score(y_test, y_score)
    logger.info("ROC AUC score: {}".format(roc_score))

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    fpr['removed'], tpr['removed'], _ = roc_curve(y_test, y_score)
    roc_auc['removed'] = auc(fpr['removed'], tpr['removed'])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plot_roc(fpr, tpr, roc_auc, class_labels=['removed'])
