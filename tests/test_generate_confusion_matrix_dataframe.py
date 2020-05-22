from pandas.testing import assert_frame_equal
from src.make_feedback_tagging.generate_confusion_matrix_dataframe import confusion_matrix_dataframe
import pandas as pd
import pytest

# Define a binary classification test case (true labels, predicted labels, expected output)
test_binary_case = (
    [1, 0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 1],
    pd.DataFrame([[2, 1], [1, 2]], index=pd.MultiIndex.from_arrays([["actual"] * 2, [0, 1]]),
                 columns=pd.MultiIndex.from_arrays([["predicted"] * 2, [0, 1]]))
)

# Define a multiclass classification test case (true labels, predicted labels, expected output)
test_multi_case = (
    [2, 0, 2, 2, 0, 1],
    [0, 0, 2, 2, 0, 2],
    pd.DataFrame([[2, 0, 0], [0, 0, 1], [1, 0, 2]], index=pd.MultiIndex.from_arrays([["actual"] * 3, [0, 1, 2]]),
                 columns=pd.MultiIndex.from_arrays([["predicted"] * 3, [0, 1, 2]]))
)

# Define a binary classification test case with class labels (true labels, predicted labels, expected output,
# labels)
test_binary_case_with_labels = (
    ["b", "a", "b", "a", "a", "b"],
    ["a", "a", "b", "b", "a", "b"],
    pd.DataFrame([[2, 1], [1, 2]], index=pd.MultiIndex.from_arrays([["actual"] * 2, ["a", "b"]]),
                 columns=pd.MultiIndex.from_arrays([["predicted"] * 2, ["a", "b"]])),
    ["a", "b"]
)

# Define a multiclass classification test case with class labels (true labels, predicted labels, expected output,
# class labels)
test_multi_case_with_labels = (
    ["c", "a", "c", "c", "a", "b"],
    ["a", "a", "c", "c", "a", "c"],
    pd.DataFrame([[2, 0, 0], [0, 0, 1], [1, 0, 2]], index=pd.MultiIndex.from_arrays([["actual"] * 3, ["a", "b", "c"]]),
                 columns=pd.MultiIndex.from_arrays([["predicted"] * 3, ["a", "b", "c"]])),
    ["a", "b", "c"]
)


@pytest.mark.parametrize("test_input_true, test_input_pred, test_expected", [test_binary_case, test_multi_case])
def test_confusion_matrix_dataframe_returns_correctly(test_input_true, test_input_pred, test_expected):
    """Test the confusion_matrix_dataframe function returns correctly."""
    assert_frame_equal(test_expected, confusion_matrix_dataframe(test_input_true, test_input_pred))


@pytest.mark.parametrize("test_input_true, test_input_pred, test_expected, test_labels",
                         [test_binary_case_with_labels, test_multi_case_with_labels])
def test_confusion_matrix_dataframe_returns_correctly_with_labels(test_input_true, test_input_pred, test_expected,
                                                                  test_labels):
    """Test the confusion_matrix_dataframe function returns correctly with class labels."""
    assert_frame_equal(test_expected, confusion_matrix_dataframe(test_input_true, test_input_pred, labels=test_labels))
