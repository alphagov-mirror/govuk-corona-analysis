from datetime import datetime
from pandas.testing import assert_frame_equal, assert_series_equal
from src.make_feedback_tagging.tagging_preprocessing import (
    COLS_TAGS,
    ORDER_TAGS,
    concat_identical_columns,
    convert_object_to_datetime,
    extract_unique_tags,
    find_duplicated_rows,
    get_rank_statistic,
    rank_multiple_tags,
    rank_rows,
    rank_tags,
    remove_pii,
    sort_and_drop_duplicates,
    standardise_columns,
    tagging_preprocessing
)
from src.make_feedback_tool_data.preprocess import PII_FILTERED
from typing import Callable, Union
import numpy as np
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

# Define arguments for to test `find_duplicated_rows` in the `test_function_returns_correctly` test
args_function_returns_correctly_find_duplicated_rows = [
    ([pd.DataFrame({"id": [0, 1, 2], "col_a": [3, 4, 5], "col_b": [6, 7, 8]}), ["col_a", "col_b"]],
     pd.DataFrame(index=pd.Int64Index([]), columns=["id", "col_a", "col_b"], dtype="int64")),
    ([pd.DataFrame({"id": [0, 1, 2], "col_a": [3, 4, 4], "col_b": [6, 7, 7]}), ["col_a", "col_b"]],
     pd.DataFrame({"id": [1, 2], "col_a": [4, 4], "col_b": [7, 7]}, index=pd.Int64Index([1, 2])))
]

# Define arguments for to test `rank_rows` in the `test_function_returns_correctly` test
args_function_returns_correctly_rank_rows = [
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 9, 6, 7]}), "col_rank"],
     pd.Series([2, 1, 4, 3], dtype="float64", name="col_rank")),
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 9, 6, 7]}), "col_rank", "first"],
     pd.Series([2, 1, 4, 3], dtype="float64", name="col_rank")),
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 8, 6, 7]}), "col_rank", "first"],
     pd.Series([1, 2, 4, 3], dtype="float64", name="col_rank")),
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 8, 6, 7]}), "col_rank", "average"],
     pd.Series([1.5, 1.5, 4, 3], dtype="float64", name="col_rank")),
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 8, 6, 7]}), "col_rank", "min"],
     pd.Series([1, 1, 4, 3], dtype="float64", name="col_rank")),
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 8, 6, 7]}), "col_rank", "max"],
     pd.Series([2, 2, 4, 3], dtype="float64", name="col_rank")),
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 8, 6, 7]}), "col_rank", "dense"],
     pd.Series([1, 1, 3, 2], dtype="float64", name="col_rank")),
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 9, 6, 7]}), "col_rank", "first", False],
     pd.Series([2, 1, 4, 3], dtype="float64", name="col_rank")),
    ([pd.DataFrame({"id": [0, 1, 2, 3], "col_rank": [8, 9, 6, 7]}), "col_rank", "first", True],
     pd.Series([3, 4, 1, 2], dtype="float64", name="col_rank")),
]

# Define arguments for to test `rank_tags` in the `test_function_returns_correctly` test
args_function_returns_correctly_rank_tags = [
    ([pd.DataFrame({"col_tag": ["a", "b", "c", "d"], "data": [0, 1, 2, 3]}), "col_tag", pd.Series([4, 3, 2, 1]),
      {"b": -1, "d": -2}],
     pd.Series([7, 2, 5, 1])),
    ([pd.DataFrame({"col_tag": ["a", "b", "c", "d"], "data": [0, 1, 2, 3]}), "col_tag", pd.Series([4, 3, 2, 1]),
      {"a": -6, "b": -1, "d": -2}],
     pd.Series([1, 6, 9, 5]))
]

# Define arguments for to test `get_rank_statistic` in the `test_function_returns_correctly` test
args_function_returns_correctly_get_rank_statistic = [
    ([[pd.Series([1, 2, 3]), pd.Series([3, 2, 1])]], pd.Series([1.7320508075688772, 2.0, 1.7320508075688772])),
    ([[pd.Series([1, 2, 3]), pd.Series([3, 2, 1]), pd.Series([1, 3, 2])]],
     pd.Series([1.4422495703074083, 2.2894284851066637, 1.8171205928321397])),
    ([[pd.Series([1, 2, 3]), pd.Series([3, 2, 1]), pd.Series([1, 3, 2]), pd.Series([2, 1, 3])]],
     pd.Series([1.5650845800732873, 1.8612097182041991, 2.0597671439071177])),
]

# Define arguments for to test `sort_and_drop_duplicates` in the `test_function_returns_correctly` test
args_function_returns_correctly_sort_and_drop_duplicates = [
    ([pd.DataFrame({"rank": [1, 3, 2, 4, 5], "data": [4, 6, 6, 1, 4]}), "rank", ["data"], True],
     pd.DataFrame({"rank": [1, 2, 4], "data": [4, 6, 1]}, index=pd.Int64Index([0, 2, 3]))),
    ([pd.DataFrame({"rank": [1, 3, 2, 4, 5], "data": [4, 6, 6, 1, 4]}), "rank", ["data"], False],
     pd.DataFrame({"rank": [3, 4, 5], "data": [6, 1, 4]}, index=pd.Int64Index([1, 3, 4])))
]

# Define arguments for to test `concat_identical_columns` in the `test_function_returns_correctly` test
args_function_returns_correctly_concat_identical_columns = [
    ([pd.DataFrame({"data": [3, 7, 9]}, index=pd.Int64Index([1, 4, 5])),
      pd.DataFrame({"rank": [1, 2, 4], "data": [4, 6, 1]}, index=pd.Int64Index([0, 2, 3]))],
     pd.DataFrame({"data": [4, 3, 6, 1, 7, 9]})),
    ([pd.DataFrame({"rank": [1, 2, 4], "data": [4, 6, 1]}, index=pd.Int64Index([0, 2, 3])),
      pd.DataFrame({"data": [3, 7, 9]}, index=pd.Int64Index([1, 4, 5]))],
     pd.DataFrame({"data": [4, 3, 6, 1, 7, 9]})),
    ([pd.DataFrame({"data1": [3, 7, 9], "data2": [1, 3, 5]}, index=pd.Int64Index([1, 4, 5])),
      pd.DataFrame({"rank": [1, 2, 4], "data1": [4, 6, 1], "data2": [7, 2, 4]}, index=pd.Int64Index([0, 2, 3]))],
     pd.DataFrame({"data1": [4, 3, 6, 1, 7, 9], "data2": [7, 1, 2, 4, 3, 5]})),
    ([pd.DataFrame({"rank": [1, 2, 4], "data1": [4, 6, 1], "data2": [7, 2, 4]}, index=pd.Int64Index([0, 2, 3])),
      pd.DataFrame({"data1": [3, 7, 9], "data2": [1, 3, 5]}, index=pd.Int64Index([1, 4, 5]))],
     pd.DataFrame({"data1": [4, 3, 6, 1, 7, 9], "data2": [7, 1, 2, 4, 3, 5]}))
]

# Define arguments for to test `extract_unique_tags` in the `test_function_returns_correctly` test
args_function_returns_correctly_extract_unique_tags = [
    ([pd.DataFrame({"text_date": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0),
                                  datetime(2020, 1, 1, 2, 0, 0), datetime(2020, 1, 1, 3, 0, 0),
                                  datetime(2020, 1, 1, 4, 0, 0), datetime(2020, 1, 1, 5, 0, 0),
                                  datetime(2020, 1, 1, 6, 0, 0), datetime(2020, 1, 1, 7, 0, 0),
                                  datetime(2020, 1, 1, 8, 0, 0), datetime(2020, 1, 1, 9, 0, 0)],
                    "this_response_relates_to_": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                    "coronavirus_theme": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "data": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}),
      "text_date", None, None, "rank"],
     pd.DataFrame({"text_date": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0),
                                 datetime(2020, 1, 1, 2, 0, 0), datetime(2020, 1, 1, 3, 0, 0),
                                 datetime(2020, 1, 1, 4, 0, 0), datetime(2020, 1, 1, 5, 0, 0),
                                 datetime(2020, 1, 1, 6, 0, 0), datetime(2020, 1, 1, 7, 0, 0),
                                 datetime(2020, 1, 1, 8, 0, 0), datetime(2020, 1, 1, 9, 0, 0)],
                   "this_response_relates_to_": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                   "coronavirus_theme": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                   "data": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})),
    ([pd.DataFrame({"text_date": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0),
                                  datetime(2020, 1, 1, 2, 0, 0), datetime(2020, 1, 1, 3, 0, 0),
                                  datetime(2020, 1, 1, 4, 0, 0), datetime(2020, 1, 1, 5, 0, 0),
                                  datetime(2020, 1, 1, 6, 0, 0), datetime(2020, 1, 1, 7, 0, 0),
                                  datetime(2020, 1, 1, 8, 0, 0), datetime(2020, 1, 1, 9, 0, 0)],
                    "this_response_relates_to_": ["a", "b", np.nan, "internal", "e", "f", "g", "h", "i", "j"],
                    "coronavirus_theme": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "none"],
                    "data": [0, 1, 2, 2, 3, 4, 6, 5, 0, 6]}),
      "text_date", None, None, "rank"],
     pd.DataFrame({"text_date": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0),
                                 datetime(2020, 1, 1, 3, 0, 0), datetime(2020, 1, 1, 4, 0, 0),
                                 datetime(2020, 1, 1, 5, 0, 0), datetime(2020, 1, 1, 6, 0, 0),
                                 datetime(2020, 1, 1, 7, 0, 0)],
                   "this_response_relates_to_": ["a", "b", "internal", "e", "f", "g", "h"],
                   "coronavirus_theme": ["A", "B", "D", "E", "F", "G", "H"],
                   "data": [0, 1, 2, 3, 4, 6, 5]}, index=pd.Int64Index([0, 1, 3, 4, 5, 6, 7]))),
    ([pd.DataFrame({"text_date": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0),
                                  datetime(2020, 1, 1, 2, 0, 0), datetime(2020, 1, 1, 3, 0, 0),
                                  datetime(2020, 1, 1, 4, 0, 0), datetime(2020, 1, 1, 5, 0, 0),
                                  datetime(2020, 1, 1, 6, 0, 0), datetime(2020, 1, 1, 7, 0, 0),
                                  datetime(2020, 1, 1, 8, 0, 0), datetime(2020, 1, 1, 9, 0, 0)],
                    "this_response_relates_to_": ["a", "b", np.nan, "internal", "e", "f", "g", "h", "i", "j"],
                    "coronavirus_theme": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "none"],
                    "data": [0, 1, 2, 2, 3, 4, 6, 5, 0, 6]}),
      "text_date", COLS_TAGS, ORDER_TAGS, "rank"],
     pd.DataFrame({"text_date": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0),
                                 datetime(2020, 1, 1, 3, 0, 0), datetime(2020, 1, 1, 4, 0, 0),
                                 datetime(2020, 1, 1, 5, 0, 0), datetime(2020, 1, 1, 6, 0, 0),
                                 datetime(2020, 1, 1, 7, 0, 0)],
                   "this_response_relates_to_": ["a", "b", "internal", "e", "f", "g", "h"],
                   "coronavirus_theme": ["A", "B", "D", "E", "F", "G", "H"],
                   "data": [0, 1, 2, 3, 4, 6, 5]}, index=pd.Int64Index([0, 1, 3, 4, 5, 6, 7])))
]

# Define arguments for to test `remove_pii` in the `test_function_returns_correctly` test
args_function_returns_correctly_remove_pii = [
    ([pd.Series(["Text with no PII", *[f"Text with [{p}]" for p in PII_FILTERED]])],
     pd.Series(["text with no pii", *["text with "] * len(PII_FILTERED)]))
]

# Define arguments for to test `tagging_preprocessing` in the `test_function_returns_correctly` test
args_function_returns_correctly_tagging_preprocessing = [
    ([pd.DataFrame({"text_date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00UTC", "2020-01-01 02:00:00 UTC",
                                  "2020-01-01 03:00:00 GMT", "2020-01-01 04:00:00 ", "2020-01-01 05:00:00  ",
                                  "2020-01-01 06:00:00  CET", "2020-01-01 07:00:00 Time", "2020-01-01 08:00:00 time",
                                  "2020-01-01 09:00:00"],
                    "this_response_relates_to_": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                    "coronavirus_theme": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "data": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}),
      "text_date", None, None, "rank"],
     pd.DataFrame({"text_date": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0),
                                 datetime(2020, 1, 1, 2, 0, 0), datetime(2020, 1, 1, 3, 0, 0),
                                 datetime(2020, 1, 1, 4, 0, 0), datetime(2020, 1, 1, 5, 0, 0),
                                 datetime(2020, 1, 1, 6, 0, 0), datetime(2020, 1, 1, 7, 0, 0),
                                 datetime(2020, 1, 1, 8, 0, 0), datetime(2020, 1, 1, 9, 0, 0)],
                   "this_response_relates_to_": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                   "coronavirus_theme": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                   "data": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})),
    ([pd.DataFrame({"text_date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00UTC", "2020-01-01 02:00:00 UTC",
                                  "2020-01-01 03:00:00 GMT", "2020-01-01 04:00:00 ", "2020-01-01 05:00:00  ",
                                  "2020-01-01 06:00:00  CET", "2020-01-01 07:00:00 Time", "2020-01-01 08:00:00 time",
                                  "2020-01-01 09:00:00"],
                    "this_response_relates_to_": ["a", "b", np.nan, "internal", "e", "f", "g", "h", "i", "j"],
                    "coronavirus_theme": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "none"],
                    "data": [0, 1, 2, 2, 3, 4, 6, 5, 0, 6]}),
      "text_date", None, None, "rank"],
     pd.DataFrame({"text_date": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0),
                                 datetime(2020, 1, 1, 3, 0, 0), datetime(2020, 1, 1, 4, 0, 0),
                                 datetime(2020, 1, 1, 5, 0, 0), datetime(2020, 1, 1, 6, 0, 0),
                                 datetime(2020, 1, 1, 7, 0, 0)],
                   "this_response_relates_to_": ["a", "b", "internal", "e", "f", "g", "h"],
                   "coronavirus_theme": ["A", "B", "D", "E", "F", "G", "H"],
                   "data": [0, 1, 2, 3, 4, 6, 5]}, index=pd.Int64Index([0, 1, 3, 4, 5, 6, 7])))
]


# Create the test cases for the `test_function_returns_correctly` test
args_function_returns_correctly = [
    *[(standardise_columns, *a) for a in args_function_returns_correctly_standardise_columns_returns_correctly],
    *[(convert_object_to_datetime, *a) for a in args_function_returns_correctly_convert_object_to_datetime],
    *[(find_duplicated_rows, *a) for a in args_function_returns_correctly_find_duplicated_rows],
    *[(rank_rows, *a) for a in args_function_returns_correctly_rank_rows],
    *[(rank_tags, *a) for a in args_function_returns_correctly_rank_tags],
    *[(get_rank_statistic, *a) for a in args_function_returns_correctly_get_rank_statistic],
    *[(sort_and_drop_duplicates, *a) for a in args_function_returns_correctly_sort_and_drop_duplicates],
    *[(concat_identical_columns, *a) for a in args_function_returns_correctly_concat_identical_columns],
    *[(extract_unique_tags, *a) for a in args_function_returns_correctly_extract_unique_tags],
    *[(remove_pii, *a) for a in args_function_returns_correctly_remove_pii],
    *[(tagging_preprocessing, *a) for a in args_function_returns_correctly_tagging_preprocessing]
]


@pytest.mark.parametrize("test_func, test_input, test_expected", args_function_returns_correctly)
def test_function_returns_correctly(test_func: Callable[..., Union[pd.DataFrame, pd.Series]], test_input,
                                    test_expected):
    """Test the a function, test_func, returns correctly, as long as test_func outputs a pandas DataFrame or Series."""
    if isinstance(test_expected, pd.DataFrame):
        print(test_func(*test_input))
        print(test_expected)
        assert_frame_equal(test_func(*test_input), test_expected)
    else:
        assert_series_equal(test_func(*test_input), test_expected)


def test_convert_object_to_datetime_raises_error_for_nat():
    """Test that the convert_object_to_datetime function returns an AssertionError if not all times could be parsed."""
    with pytest.raises(AssertionError):
        _ = convert_object_to_datetime(pd.DataFrame({"text_date": ["2020-12-29 18:28:43", None], "data": [2, 3]}),
                                       "text_date")


# Define some test cases for the `test_rank_tags_raises_assertion_error` test
args_rank_tags_raises_assertion_error = [a for args in args_function_returns_correctly_rank_tags for a in args[:-1]]


@pytest.mark.parametrize("test_input_df, test_input_col_tag, test_input_s_ranked, test_input_set_tag_ranks",
                         args_rank_tags_raises_assertion_error)
def test_rank_tags_raises_assertion_error(test_input_df, test_input_col_tag, test_input_s_ranked,
                                          test_input_set_tag_ranks):
    """Test rank_tags raises an assertion error if the lengths of df and s_ranked do not match."""

    # Iterate over the length of `test_input_s_ranked`, and execute `rank_tags` with a filtered `test_input_s_ranked`
    # less than its original length - should always raise an `AssertionError` with a specific message
    for ii in range(len(test_input_s_ranked) - 1):
        with pytest.raises(AssertionError, match=f"'s_ranked', and 'df' must be the same length!: {ii:,} != "
                                                 f"{len(test_input_df):,}"):
            rank_tags(test_input_df, test_input_col_tag, test_input_s_ranked.iloc[:ii], test_input_set_tag_ranks)


# Define test cases for to the `TestRankMulitpleTags` test class
args_rank_multiple_tags = [
    (pd.DataFrame({"col_tag1": ["a", "b", "c"], "col_tag2": ["b", "c", "d"], "data": [0, 1, 2]}),
     ["col_tag1", "col_tag2"], pd.Series([1, 3, 2]), {"b": -1, "d": -2},
     [pd.Series([3, 1, 4]), pd.Series([2, 6, 1])]),
    (pd.DataFrame({"col_tag1": ["a", "b", "a", "c"], "col_tag2": ["b", "c", "d", "d"], "data": [0, 1, 2, 2]}),
     ["col_tag1", "col_tag2"], pd.Series([1, 3, 2, 4]), {"b": -1, "d": -2},
     [pd.Series([3, 1, 4, 6]), pd.Series([2, 6, 1, 1])])
]


@pytest.fixture
def patch_rank_tags(mocker):
    """Patch the rank_tags function."""
    return mocker.patch("src.make_feedback_tagging.tagging_preprocessing.rank_tags")


class TestRankMultipleTags:

    @pytest.mark.parametrize("test_input_df, test_input_col_tags, test_input_s_ranked, test_set_tag_ranks, "
                             "test_expected", args_rank_multiple_tags)
    def test_returns_correctly(self, test_input_df, test_input_col_tags, test_input_s_ranked, test_set_tag_ranks,
                               test_expected):
        """Check that rank_multiple_tags returns correctly."""

        # Invoke the `rank_multiple_tags` function
        test_output = rank_multiple_tags(test_input_df, test_input_col_tags, test_input_s_ranked, test_set_tag_ranks)

        # Assert `test_output`, and `test_expected` are the same length
        assert len(test_output) == len(test_expected)

        # Assert the values are as expected
        for e, o in zip(test_expected, test_output):
            assert_series_equal(e, o)

    @pytest.mark.parametrize("test_input_df, test_input_col_tags, test_input_s_ranked, test_input_set_tag_ranks",
                             [(*[a[:-1] for a in args_rank_multiple_tags])])
    def test_calls_rank_tags_correctly(self, patch_rank_tags, test_input_df, test_input_col_tags, test_input_s_ranked,
                                       test_input_set_tag_ranks):
        """Check that rank_multiple_tags calls rank_tags correctly."""

        # Invoke the `rank_multiple_tags` function
        _ = rank_multiple_tags(test_input_df, test_input_col_tags, test_input_s_ranked, test_input_set_tag_ranks)

        # Assert `rank_tags` is called the same number of times as the length of `test_input_col_tags`
        assert patch_rank_tags.call_count == len(test_input_col_tags)

        # Get the called arguments for `rank_tags`
        test_output = patch_rank_tags.call_args_list

        # Iterate over the arguments, and keyword arguments
        for ii, (test_output_args, test_output_kwargs) in enumerate(test_output):

            # Assert there are no keyword arguments
            assert not test_output_kwargs

            # Assert there are four keyword arguments
            assert len(test_output_args) == 4

            # Assert the first keyword argument is `test_input_df`
            assert_frame_equal(test_output_args[0], test_input_df)

            # Assert the remaining keyword arguments are the `ii`-th element of `test_input_col_tags`,
            # `test_input_s_ranked`, and `test_input_set_tag_ranks`
            assert test_output_args[1:] == (test_input_col_tags[ii], test_input_s_ranked, test_input_set_tag_ranks)


@pytest.fixture
def resource_extract_unique_tags_integration(mocker, patch_rank_tags):
    """Define a pytest fixture for the TestExtractUniqueTagsIntegration test class."""

    # Patch various functions used by the `extract_unique_tags` function
    patch_standardise_columns = mocker.patch("src.make_feedback_tagging.tagging_preprocessing.standardise_columns")
    patch_find_duplicated_rows = mocker.patch("src.make_feedback_tagging.tagging_preprocessing.find_duplicated_rows")
    patch_rank_rows = mocker.patch("src.make_feedback_tagging.tagging_preprocessing.rank_rows")
    patch_rank_multiple_tags = mocker.patch("src.make_feedback_tagging.tagging_preprocessing.rank_multiple_tags")
    patch_get_rank_statistic = mocker.patch("src.make_feedback_tagging.tagging_preprocessing.get_rank_statistic")
    patch_sort_and_drop_duplicates = mocker.patch(
        "src.make_feedback_tagging.tagging_preprocessing.sort_and_drop_duplicates"
    )
    patch_concat_identical_columns = mocker.patch(
        "src.make_feedback_tagging.tagging_preprocessing.concat_identical_columns",
    )

    # Ensure the last function has a return value for its `.duplicated().any()` method of False - this ensures the
    # AssertionError is not tripped by the patches
    patch_concat_identical_columns.return_value.duplicated().any.return_value = False

    return {"patch_standardise_columns": patch_standardise_columns,
            "patch_find_duplicated_rows": patch_find_duplicated_rows, "patch_rank_rows": patch_rank_rows,
            "patch_rank_tags": patch_rank_tags, "patch_rank_multiple_tags": patch_rank_multiple_tags,
            "patch_get_rank_statistic": patch_get_rank_statistic,
            "patch_sort_and_drop_duplicates": patch_sort_and_drop_duplicates,
            "patch_concat_identical_columns": patch_concat_identical_columns}


# Define the test cases for the majority of arguments in the `TestExtractUniqueTagsIntegration` test class
args_extract_unique_tags_integration = [
    (pd.DataFrame({"col_key": ["a", "b", "c"], "col_a": [0, 1, 1], "col_b": [3, 4, 4], "col_tag1": [6, 7, 8],
                   "col_tag2": [9, 10, 11]}), "col_key", None),
    (pd.DataFrame({"col_key": ["a", "b", "c"], "col_a": [0, 0, 1], "col_b": [3, 3, 4], "col_tag1": [6, 7, 8],
                   "col_tag2": [9, 10, 11]}), "col_key", ["col_tag1", "col_tag2"]),
]

# Define the test cases for the `test_input_out_col_rank_label` argument of the `TestExtractUniqueTagsIntegration` test
# class
args_extract_unique_tags_integration_out_col_rank_label = [
    "rank", "hello", "world"
]

# Define the test cases for the `test_input_set_tag_ranks` argument of the `TestExtractUniqueTagsIntegration` test class
args_extract_unique_tags_integration_set_tag_ranks = [
    None, {"b": -1}, {"b": -2, "a": -1}
]


@pytest.mark.parametrize("test_input_df, test_input_col_key, test_input_col_tags", args_extract_unique_tags_integration)
@pytest.mark.parametrize("test_input_out_col_rank_label", args_extract_unique_tags_integration_out_col_rank_label)
@pytest.mark.parametrize("test_input_set_tag_ranks", args_extract_unique_tags_integration_set_tag_ranks)
class TestExtractUniqueTagsIntegration:

    def test_find_duplicated_rows_called_once_correctly(self, resource_extract_unique_tags_integration,
                                                        test_input_df, test_input_col_key, test_input_col_tags,
                                                        test_input_set_tag_ranks, test_input_out_col_rank_label):
        """Test the extract_unique_tags calls find_duplicated_rows once correctly."""

        # Call the `extract_unique_tags` function
        _ = extract_unique_tags(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                test_input_out_col_rank_label)

        # Assert `find_duplicated_rows` is called once
        resource_extract_unique_tags_integration["patch_find_duplicated_rows"].assert_called_once()

        # Get the call argument list, and extract the arguments and keyword arguments
        test_output = resource_extract_unique_tags_integration["patch_find_duplicated_rows"].call_args_list
        test_output_args, test_output_kwargs = test_output[0]

        # Assert there are two arguments, and no keyword arguments
        assert len(test_output_args) == 2
        assert not test_output_kwargs

        # Assert the first argument is the correct pandas DataFrame
        assert_frame_equal(test_output_args[0], test_input_df)

        # Define the expected second argument; if `test_input_col_tags` is None, this should be `COLS_TAGS`
        if test_input_col_tags:
            test_expected_arg_1 = [c for c in test_input_df.columns if c not in [test_input_col_key,
                                                                                 *test_input_col_tags]]
        else:
            test_expected_arg_1 = [c for c in test_input_df.columns if c not in [test_input_col_key, *COLS_TAGS]]

        # Assert the second argument is all columns in `test_input_df` that are not in `test_input_col_tags`
        assert test_output_args[1] == test_expected_arg_1

    def test_rank_rows_called_once_correctly(self, resource_extract_unique_tags_integration, test_input_df,
                                             test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                             test_input_out_col_rank_label):
        """Test the extract_unique_tags calls rank_rows once correctly."""

        # Call the `extract_unique_tags` function
        _ = extract_unique_tags(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                test_input_out_col_rank_label)

        # Assert the `rank_rows` function is called once with the expected call arguments
        resource_extract_unique_tags_integration["patch_rank_rows"].assert_called_once_with(
            resource_extract_unique_tags_integration["patch_find_duplicated_rows"].return_value,
            test_input_col_key
        )

    def test_rank_multiple_tags_called_once_correctly(self, resource_extract_unique_tags_integration, test_input_df,
                                                      test_input_col_key, test_input_col_tags,
                                                      test_input_set_tag_ranks, test_input_out_col_rank_label):
        """Test the extract_unique_tags calls rank_multiple_tags once correctly."""

        # Set `test_input_set_tag_ranks` to `ORDER_TAGS` if it is None
        if not test_input_set_tag_ranks:
            test_input_set_tag_ranks = ORDER_TAGS

        # Call the `extract_unique_tags` function
        _ = extract_unique_tags(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                test_input_out_col_rank_label)

        # Assert the `rank_multiple_tags` function is called once with the expected call arguments
        resource_extract_unique_tags_integration["patch_rank_multiple_tags"].assert_called_once_with(
            resource_extract_unique_tags_integration["patch_find_duplicated_rows"].return_value,
            test_input_col_tags if test_input_col_tags else COLS_TAGS,
            resource_extract_unique_tags_integration["patch_rank_rows"].return_value,
            test_input_set_tag_ranks
        )

    def test_get_rank_statistic_called_once_correctly(self, resource_extract_unique_tags_integration, test_input_df,
                                                      test_input_col_key, test_input_col_tags,
                                                      test_input_set_tag_ranks, test_input_out_col_rank_label):
        """Test the extract_unique_tags calls get_rank_statistic once correctly."""

        # Call the `extract_unique_tags` function
        _ = extract_unique_tags(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                test_input_out_col_rank_label)

        # Assert the `get_rank_statistic` function is called once with the expected call arguments
        resource_extract_unique_tags_integration["patch_get_rank_statistic"].assert_called_once_with(
            resource_extract_unique_tags_integration["patch_rank_multiple_tags"].return_value
        )

    def test_sort_and_drop_duplicates_called_once_correctly(self, resource_extract_unique_tags_integration,
                                                            test_input_df, test_input_col_key, test_input_col_tags,
                                                            test_input_set_tag_ranks, test_input_out_col_rank_label):
        """Test the extract_unique_tags calls sort_and_drop_duplicates once correctly."""

        # Call the `extract_unique_tags` function
        _ = extract_unique_tags(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                test_input_out_col_rank_label)

        # Get the return value from `get_rank_statistic`
        test_expected_grs = resource_extract_unique_tags_integration["patch_find_duplicated_rows"].return_value

        # Get the return value from `find_duplicated_rows`
        test_expected_fdr = resource_extract_unique_tags_integration["patch_find_duplicated_rows"].return_value

        # Compile the expected pandas DataFrame argument `df` for the `sort_and_drop_duplicates` function
        test_expected_df = test_expected_fdr.assign(**{test_input_out_col_rank_label: test_expected_grs})

        # Define the expected `col_duplicates` arguments of the `sort_and_drop_duplicates` function; if
        # test_input_col_tags` is None, this should be `COLS_TAGS`
        if test_input_col_tags:
            test_expected_col_duplicates = [c for c in test_input_df.columns if c not in [test_input_col_key,
                                                                                          *test_input_col_tags]]
        else:
            test_expected_col_duplicates = [c for c in test_input_df.columns if c not in [test_input_col_key,
                                                                                          *COLS_TAGS]]

        # Assert the `sort_and_drop_duplicates` function is called once with the expected call arguments
        resource_extract_unique_tags_integration["patch_sort_and_drop_duplicates"].assert_called_once_with(
            test_expected_df,
            test_input_out_col_rank_label,
            test_expected_col_duplicates
        )

    def test_patch_concat_identical_columns_called_once_correctly(self, resource_extract_unique_tags_integration,
                                                                  test_input_df, test_input_col_key,
                                                                  test_input_col_tags, test_input_set_tag_ranks,
                                                                  test_input_out_col_rank_label):
        """Test the extract_unique_tags calls concat_identical_columns once correctly."""

        # Call the `extract_unique_tags` function
        _ = extract_unique_tags(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                test_input_out_col_rank_label)

        # Assert the `concat_identical_columns` function is called once
        resource_extract_unique_tags_integration["patch_concat_identical_columns"].assert_called_once()

        # Get the arguments and keyword arguments used to call the `concat_identical_columns` function of the
        # singular call to it
        test_output = resource_extract_unique_tags_integration["patch_concat_identical_columns"].call_args_list
        test_output_args, test_output_kwargs = test_output[0]

        # Assert that there are only two arguments and no keyword arguments
        assert len(test_output_args) == 2
        assert not test_output_kwargs

        # Define the expected pandas DataFrame argument `df1` for the `concat_identical_columns` function; note this
        # is an integration test, so does not actually check if the subset is is performed correctly - this will be
        # done in unit/systems testing
        test_expected_df1_bool = test_input_df.index.isin(
            resource_extract_unique_tags_integration["patch_find_duplicated_rows"].return_value.index
        )
        test_expected_df1 = test_input_df[~test_expected_df1_bool]

        # Assert the first keyword argument is the expected pandas DataFrame
        assert_frame_equal(test_output_args[0], test_expected_df1)

        # Define the expected second argument
        test_expected_args_1 = resource_extract_unique_tags_integration["patch_sort_and_drop_duplicates"].return_value

        # Assert the second keyword argument is the expected mock of a pandas DataFrame
        assert test_output_args[1] == test_expected_args_1

    def test_full_integration(self, resource_extract_unique_tags_integration, test_input_df, test_input_col_key,
                              test_input_col_tags, test_input_set_tag_ranks, test_input_out_col_rank_label):
        """Test the extract_unique_tags returns the correct function output."""

        # Call the `extract_unique_tags` function
        test_output = extract_unique_tags(test_input_df, test_input_col_key, test_input_col_tags,
                                          test_input_set_tag_ranks, test_input_out_col_rank_label)

        # Assert `test_output` is the return value from a specific function
        assert test_output == resource_extract_unique_tags_integration["patch_concat_identical_columns"].return_value


# Define the test cases for the `TestExtractUniqueTagsRaisesAssertionError` test class
args_extract_unique_tags_raises_assertion_if_duplicates_remain = [
    (pd.DataFrame({"col_key": ["a", "b", "c"], "col_a": [0, 1, 1], "col_b": [3, 4, 4], "col_tag1": [6, 7, 8],
                   "col_tag2": [9, 10, 11]}), "col_key", ["col_tag1", "col_tag2"], {"b": -1}),
    (pd.DataFrame({"col_key": ["a", "b", "c"], "col_a": [0, 0, 1], "col_b": [3, 3, 4], "col_tag1": [6, 7, 8],
                   "col_tag2": [9, 10, 11]}), "col_key", ["col_tag1", "col_tag2"], {"b": -2, "a": -1}),
]


@pytest.mark.parametrize("test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks",
                         args_extract_unique_tags_raises_assertion_if_duplicates_remain)
class TestExtractUniqueTagsRaisesAssertionError:

    @pytest.mark.parametrize("test_input_out_col_rank_label", ["rank", "test"])
    def test_raises_if_col_key_in_df(self, resource_extract_unique_tags_integration, test_input_df,
                                     test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                     test_input_out_col_rank_label):
        """Test extract_unique_tags raises an AssertionError if col_key is a column in df."""

        # Add a column called `test_input_out_col_rank_label` into `test_input_df`
        test_input_df_with_out_col_rank_label = test_input_df.assign(**{test_input_out_col_rank_label: None})

        # Call the `extract_unique_tags` function, and check it raises an AssertionError
        with pytest.raises(AssertionError, match="`out_col_rank_label` cannot be a column in `df`; please change this "
                                                 f"input argument: {test_input_out_col_rank_label}"):
            _ = extract_unique_tags(test_input_df_with_out_col_rank_label, test_input_col_key, test_input_col_tags,
                                    test_input_set_tag_ranks, test_input_out_col_rank_label)

    def test_raises_if_duplicates_remain(self, resource_extract_unique_tags_integration, test_input_df,
                                         test_input_col_key, test_input_col_tags, test_input_set_tag_ranks):
        """Test extract_unique_tags raises an AssertionError if there are still duplicates after processing."""

        # Set the return value of the last function to `test_input_df`, assuming it has duplicate values
        resource_extract_unique_tags_integration["patch_concat_identical_columns"].return_value = test_input_df

        # Call the `extract_unique_tags` function, and check it raises an AssertionError
        with pytest.raises(AssertionError, match="Duplicate values remain after processing!"):
            _ = extract_unique_tags(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                    "rank")


@pytest.fixture
def resource_tagging_preprocessing_integration(mocker):
    """Define the patches for the tagging_preprocessing integration tests."""

    # Patch functions used by `tagging_preprocessing`
    patch_standardise_columns = mocker.patch("src.make_feedback_tagging.tagging_preprocessing.standardise_columns")
    patch_convert_object_to_datetime = mocker.patch(
        "src.make_feedback_tagging.tagging_preprocessing.convert_object_to_datetime"
    )
    patch_extract_unique_tags = mocker.patch("src.make_feedback_tagging.tagging_preprocessing.extract_unique_tags")

    return {"patch_standardise_columns": patch_standardise_columns,
            "patch_convert_object_to_datetime": patch_convert_object_to_datetime,
            "patch_extract_unique_tags": patch_extract_unique_tags}


@pytest.mark.parametrize("test_input_df, test_input_col_key, test_input_col_tags", args_extract_unique_tags_integration)
@pytest.mark.parametrize("test_input_out_col_rank_label", args_extract_unique_tags_integration_out_col_rank_label)
@pytest.mark.parametrize("test_input_set_tag_ranks", args_extract_unique_tags_integration_set_tag_ranks)
class TestTaggingPreProcessingIntegration:

    def test_standardise_columns_called_once_correctly(self, resource_tagging_preprocessing_integration, test_input_df,
                                                       test_input_col_key, test_input_col_tags,
                                                       test_input_set_tag_ranks, test_input_out_col_rank_label):
        """Test that tagging_preprocessing calls standardise_columns once correctly."""

        # Call the `tagging_preprocessing` function
        _ = tagging_preprocessing(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                  test_input_out_col_rank_label)

        # Assert `standardise_columns` is called once
        resource_tagging_preprocessing_integration["patch_standardise_columns"].assert_called_once()

        # Get the call arguments list, and then extract the arguments and keyword arguments for the first and only call
        test_output = resource_tagging_preprocessing_integration["patch_standardise_columns"].call_args_list
        test_output_args, test_output_kwargs = test_output[0]

        # Assert that there is only one argument, and no keyword arguments
        assert len(test_output_args) == 1
        assert not test_output_kwargs

        # Assert the argument is as expected
        assert_frame_equal(test_output_args[0], test_input_df)

    def test_convert_object_to_datetime_called_once_correctly(self, resource_tagging_preprocessing_integration,
                                                              test_input_df, test_input_col_key, test_input_col_tags,
                                                              test_input_set_tag_ranks, test_input_out_col_rank_label):
        """Test that tagging_preprocessing calls convert_object_to_datetime once correctly."""

        # Call the `tagging_preprocessing` function
        _ = tagging_preprocessing(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                  test_input_out_col_rank_label)

        # Assert `convert_object_to_datetime` is called once with the correct arguments
        resource_tagging_preprocessing_integration["patch_convert_object_to_datetime"].assert_called_once_with(
            resource_tagging_preprocessing_integration["patch_standardise_columns"].return_value, test_input_col_key
        )

    def test_extract_unique_tags_called_once_correctly(self, resource_tagging_preprocessing_integration,
                                                       test_input_df, test_input_col_key, test_input_col_tags,
                                                       test_input_set_tag_ranks, test_input_out_col_rank_label):
        """Test that tagging_preprocessing calls extract_unique_tags once correctly."""

        # Call the `tagging_preprocessing` function
        _ = tagging_preprocessing(test_input_df, test_input_col_key, test_input_col_tags, test_input_set_tag_ranks,
                                  test_input_out_col_rank_label)

        # Assert `extract_unique_tags` is called once with the correct arguments
        resource_tagging_preprocessing_integration["patch_extract_unique_tags"].assert_called_once_with(
            resource_tagging_preprocessing_integration["patch_convert_object_to_datetime"].return_value,
            test_input_col_key, test_input_col_tags, test_input_set_tag_ranks, test_input_out_col_rank_label
        )
