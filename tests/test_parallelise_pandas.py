from pandas.testing import assert_frame_equal, assert_series_equal
from src.utils.parallelise_pandas import COUNT_CPU, parallelise_pandas
from typing import Union
import multiprocessing as mp
import pandas as pd
import pytest


def example_function(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """An example function that multiples values by two."""
    return data.multiply(2)


# Define input data for the `TestParallelisePandas` test class
args_parallelise_pandas_input_data = [
    pd.Series([0, 1, 2, 3]),
    pd.DataFrame({"data": [0, 1, 2, 3]}),
    pd.DataFrame({"col_1": [0, 1, 2, 3], "col_2": ["a", "b", "c", "d"]})
]

# Define the expected outputs of `args_parallelise_pandas_input_data`
args_parallelise_pandas_expected = [
    pd.Series([0, 2, 4, 6]),
    pd.DataFrame({"data": [0, 2, 4, 6]}),
    pd.DataFrame({"col_1": [0, 2, 4, 6], "col_2": ["aa", "bb", "cc", "dd"]})
]


@pytest.mark.parametrize("test_input_n_cores", [None, 8, 16])
class TestParallelisePandas:

    @pytest.mark.parametrize("test_input_data", args_parallelise_pandas_input_data)
    def test_shows_warning_if_n_cores_is_incorrect(self, test_input_data, test_input_n_cores):
        """Test that parallelise_pandas shows a warning message if n_cores is greater than COUNT_CPU."""

        # If `test_input_n_cores` is greater than `COUNT_CPU`, a warning should be shown
        if test_input_n_cores is not None and test_input_n_cores > COUNT_CPU:
            with pytest.warns(RuntimeWarning, match="`n_cores` is larger than available processors"):
                _ = parallelise_pandas(test_input_data, example_function, test_input_n_cores)
        else:
            with pytest.warns(None) as record:
                _ = parallelise_pandas(test_input_data, example_function, test_input_n_cores)
                if record:
                    pytest.fail("Unexpected warning raised!")

    @pytest.mark.parametrize("test_input_data", args_parallelise_pandas_input_data)
    def test_multiprocessing_pool_called_once_correctly(self, mocker, test_input_data, test_input_n_cores):
        """Test that parallelise_pandas calls the multiprocessing.Pool class once correctly."""

        # Patch the `multiprocessing.Pool` class
        patch_pool = mocker.patch("multiprocessing.Pool", wraps=mp.Pool)

        # Call the `parallelise_pandas` function
        _ = parallelise_pandas(test_input_data, example_function, test_input_n_cores)

        # Assert `multiprocessing.Pool` is called once with the correct arguments
        patch_pool.assert_called_once_with(
            COUNT_CPU if test_input_n_cores is None or test_input_n_cores > COUNT_CPU else test_input_n_cores
        )

    @pytest.mark.parametrize("test_input_data, test_expected",
                             zip(args_parallelise_pandas_input_data, args_parallelise_pandas_expected))
    def test_returns_correctly(self, test_input_data, test_input_n_cores, test_expected):
        """Test parallelise_pandas returns the correct output."""

        # Call the `parallelise_pandas` function
        test_output = parallelise_pandas(test_input_data, example_function, test_input_n_cores)

        # Assert `test_output` is correct, depending if `test_input_data` is a pandas Series or DataFrame
        if isinstance(test_input_data, pd.Series):
            assert_series_equal(test_output, test_expected)
        else:
            assert_frame_equal(test_output, test_expected)
