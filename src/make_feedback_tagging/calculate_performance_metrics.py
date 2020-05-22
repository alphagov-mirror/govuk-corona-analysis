from functools import partial
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, roc_auc_score, matthews_corrcoef
)
from src.make_feedback_tagging.generate_confusion_matrix_dataframe import confusion_matrix_dataframe
# from src.tools.logger import logger, Log
from typing import Any, Callable, Dict, List, Optional
import numpy
import pandas

# Define scikit-learn metrics that accept true, and prediction labels
FUNCS_METRICS_PRED = [
    accuracy_score,
    confusion_matrix_dataframe,
    partial(classification_report, output_dict=True),
    matthews_corrcoef,
]

# Define scikit-learn metrics that accept true labels, and prediction probabilities
FUNCS_METRICS_PRED_PROB = [
    average_precision_score,
    roc_auc_score,
]


# @Log(logger, level="debug")
def calculate_metrics(y_true: numpy.ndarray, y_pred: numpy.ndarray, y_pred_prob: numpy.ndarray,
                      metrics_pred: Optional[List[Callable]] = None,
                      metrics_pred_prob: Optional[List[Callable]] = None,
                      verbose: bool = True) -> Dict[str, Any]:
    """Calculate a range of scikit-learn metrics on given true labels, and predicted labels and probabilities.

    :param y_true: An NumPy array of true labels.
    :param y_pred: An NumPy array of prediction labels.
    :param y_pred_prob: An NumPy array of prediction probabilities. For binary classifications, this is a 1-D array.
    :param metrics_pred: Default: None. A list of scikit-learn metrics that accept true labels, and prediction labels
        as their first two arguments.
    :param metrics_pred_prob: Default: None. A list of scikit-learn metrics that accept true labels, and prediction
        probabilities as their first two arguments.
    :param verbose: Default: True. If True, will print the outputs from the metrics.
    :return: A dictionary with all metric outputs, where each key is a metric function name, and the corresponding
        value is the metric function output.

    """

    # Initialise an empty dictionary to store the metric outputs
    clf_metrics = {}

    # If `metrics_pred` is None, set to `FUNCS_METRICS_PRED`; repeat for `metrics_pred_prob` to
    # `FUNCS_METRICS_PRED_PROB`
    if metrics_pred is None:
        metrics_pred = []
    if metrics_pred_prob is None:
        metrics_pred_prob = []

    # Coerce `y_true` to a 1-D NumPy array
    y_true = y_true.ravel()

    # Iterate over all the metric functions
    for metric_func in metrics_pred + metrics_pred_prob:

        # Get the metric name; if it is partially complete, then extract the nested function's name
        metric_name = metric_func.func.__name__ if isinstance(metric_func, partial) else metric_func.__name__

        # If the function name already exists in `clf_metrics`, raise an error
        if metric_name in clf_metrics.keys():
            raise KeyError(f"Function `{metric_name}` already exists! Check your input arguments `metrics_pred`, "
                           "and `metrics_pred_prob` for duplicate functions.")

        # Add a message to the log
        # logger.debug(f"`calculate_metrics`: Calculating {metric_name} metric")

        # If the function is in `metrics_pred`, then call it with the test and prediction labels, otherwise call
        # it with the prediction probabilities
        clf_metrics[metric_name] = metric_func(y_true, y_pred if metric_func in metrics_pred else y_pred_prob)

        # If `verbose` is True, print the metrics; use different methods depending on the type of
        # `clf_metrics[metric_name]`
        if verbose:
            if isinstance(clf_metrics[metric_name], pandas.DataFrame):
                print(f"{metric_name}:\n{clf_metrics[metric_name].to_string()}\n")
            elif isinstance(clf_metrics[metric_name], dict):
                print(f"{metric_name}:\n{pandas.DataFrame(clf_metrics[metric_name])}\n")
            else:
                print(f"{metric_name}: {clf_metrics[metric_name]:,.2f}\n")

    # Return the metric outputs
    return clf_metrics
