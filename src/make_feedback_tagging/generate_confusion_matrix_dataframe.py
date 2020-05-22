from sklearn.metrics import confusion_matrix
# from src.tools.logger import logger, Log
from typing import Any
import pandas


# @Log(logger, level="debug")
def confusion_matrix_dataframe(*args: Any, **kwargs: Any) -> pandas.DataFrame:
    """Calculate a confusion matrix for classification models, and return a correctly labelled pandas DataFrame.

    :param args: Arguments for the `sklearn.metrics.confusion_matrix` function.
    :param kwargs: Keyword arguments for the `sklearn.metrics.confusion_matrix` function.
    :return: A pandas DataFrame of the confusion matrix with correctly labelled actual and predicted axes. If
        `labels` is in `kwargs`, then this is used in the index and column labels.

    """

    # If `labels` is in `kwargs` use this to label the axes of the pandas DataFrame
    if "labels" in kwargs:
        class_labels = kwargs.get("labels")
    else:
        class_labels = list(range(len(set(args[0]))))

    # Render the confusion matrix as a correctly labelled pandas DataFrame
    df = pandas.DataFrame(confusion_matrix(*args, **kwargs),
                          index=pandas.MultiIndex.from_arrays([["actual"] * len(class_labels), class_labels]),
                          columns=pandas.MultiIndex.from_arrays([["predicted"] * len(class_labels), class_labels]))

    # Return the confusion matrix as a pandas DataFrame
    return df
