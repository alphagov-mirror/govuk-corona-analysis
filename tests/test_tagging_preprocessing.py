from datetime import datetime
from pandas.testing import assert_frame_equal, assert_series_equal
from src.make_feedback_tagging.tagging_preprocessing import (
    convert_object_to_datetime,
    standardise_columns,
)
from typing import Callable, Union
import pandas as pd
import pytest

# Define arguments for to test `standardise_columns` in the `test_function_returns_correctly` test
args_function_returns_correctly_standardise_columns_returns_correctly = [
    ([pd.DataFrame(columns=["a", "b"])], pd.DataFrame(columns=["a", "b"])),
    ([pd.DataFrame(columns=["a*!&b£)C_d", "e_f"])], pd.DataFrame(columns=["a_b_c_d", "e_f"])),
    ([pd.DataFrame({"A": [0, 1], "b c%^d(e!F": [2, 3]})], pd.DataFrame({"a": [0, 1], "b_c_d_e_f": [2, 3]}))
]

# Define arguments for to test `convert_object_datetime_formats_to_datetime` in the `test_function_returns_correctly`
# test
args_function_returns_correctly_convert_object_to_datetime = [
    ([pd.DataFrame({"col_a": ["2020-01-01 01:13:48 foo", "2020-04-23 19:49:23+00"], "col_b": [0, 1]}), "col_a"],
     pd.DataFrame({"col_a": [datetime(2020, 1, 1, 1, 13, 48), datetime(2020, 4, 23, 19, 49, 23)], "col_b": [0, 1]})),
    ([pd.DataFrame({"text_date": ["2020-12-29 18:28:43", "2020-02-17 12:07:04UTC"], "data": [2, 3]}), "text_date"],
     pd.DataFrame({"text_date": [datetime(2020, 12, 29, 18, 28, 43), datetime(2020, 2, 17, 12, 7, 4)], "data": [2, 3]}))
]

# Create the test cases for the `test_function_returns_correctly` test
args_function_returns_correctly = [
    *[(standardise_columns, *a) for a in args_function_returns_correctly_standardise_columns_returns_correctly],
    *[(convert_object_to_datetime, *a) for a in args_function_returns_correctly_convert_object_to_datetime],
]


@pytest.mark.parametrize("test_func, test_input, test_expected", args_function_returns_correctly)
def test_function_returns_correctly(test_func: Callable[..., Union[pd.DataFrame, pd.Series]], test_input,
                                    test_expected):
    """Test the a function, test_func, returns correctly, as long as test_func outputs a pandas DataFrame or Series."""
    if isinstance(test_expected, pd.DataFrame):
        assert_frame_equal(test_func(*test_input), test_expected)
    else:
        assert_series_equal(test_func(*test_input), test_expected)


def test_convert_object_to_datetime_raises_error_for_nat():
    """Test that the convert_object_to_datetime function returns an AssertionError if not all times could be parsed."""
    with pytest.raises(AssertionError):
        _ = convert_object_to_datetime(pd.DataFrame({"text_date": ["2020-12-29 18:28:43", None], "data": [2, 3]}),
                                       "text_date")
