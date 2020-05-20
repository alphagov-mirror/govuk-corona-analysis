from pandas.testing import assert_frame_equal, assert_series_equal
from src.make_feedback_tagging.tagging_preprocessing import standardise_columns
from typing import Callable, Union
import pandas as pd
import pytest

# Define arguments for to test `standardise_columns` in the `test_function_returns_correctly` test
args_function_returns_correctly_standardise_columns_returns_correctly = [
    ([pd.DataFrame(columns=["a", "b"])], pd.DataFrame(columns=["a", "b"])),
    ([pd.DataFrame(columns=["a*!&b£)C_d", "e_f"])], pd.DataFrame(columns=["a_b_c_d", "e_f"])),
    ([pd.DataFrame({"A": [0, 1], "b c%^d(e!F": [2, 3]})], pd.DataFrame({"a": [0, 1], "b_c_d_e_f": [2, 3]}))
]

# Create the test cases for the `test_function_returns_correctly` test
args_function_returns_correctly = [
    *[(standardise_columns, *a) for a in args_function_returns_correctly_standardise_columns_returns_correctly],
]


@pytest.mark.parametrize("test_func, test_input, test_expected", args_function_returns_correctly)
def test_function_returns_correctly(test_func: Callable[..., Union[pd.DataFrame, pd.Series]], test_input,
                                    test_expected):
    """Test the a function, test_func, returns correctly, as long as test_func outputs a pandas DataFrame or Series."""
    if isinstance(test_expected, pd.DataFrame):
        assert_frame_equal(test_func(*test_input), test_expected)
    else:
        assert_series_equal(test_func(*test_input), test_expected)
