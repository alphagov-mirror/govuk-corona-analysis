from src.make_feedback_tagging.calculate_performance_metrics import calculate_metrics
import numpy as np
import pandas as pd
import pytest


def example_metrics_pred(y_true, y_pred):
    """Define a scikit-learn-like metric function that uses prediction labels."""
    return y_true.sum() + y_pred.sum()


def example_metrics_pred_prob(y_true, y_pred_prob):
    """Define a scikit-learn-like metric function that uses prediction probabilities."""
    return y_true.sum() * y_pred_prob.sum()


# Define some argument functions for the test
args_metric_funcs = [
    (None, None),
    ([example_metrics_pred], None),
    (None, [example_metrics_pred_prob]),
    ([example_metrics_pred], [example_metrics_pred_prob])
]

# Define some argument functions for the duplicate test
args_duplicate_metric_funcs = [
    ([example_metrics_pred, example_metrics_pred], None),
    (None, [example_metrics_pred_prob, example_metrics_pred_prob]),
    ([example_metrics_pred], [example_metrics_pred]),
    ([example_metrics_pred_prob], [example_metrics_pred_prob]),
    ([example_metrics_pred, example_metrics_pred_prob], [example_metrics_pred, example_metrics_pred_prob])
]

# Define some true labels, and some prediction labels, and probabilities
test_input_y_true = np.array([0, 0, 1, 1])
test_input_y_pred = np.array([0, 1, 1, 0])
test_input_y_pred_prob = np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]])


@pytest.mark.parametrize("test_label", [0, 1])
class TestCalculateMetrics:

    @pytest.mark.parametrize("test_dup_metric_preds, test_dup_metric_probs", args_duplicate_metric_funcs)
    def test_raises_keyerror_for_duplicate_metrics(self, test_dup_metric_preds, test_dup_metric_probs, test_label):
        """Check that duplicate metrics raise a KeyError."""

        # Call the `calculate_metrics` function, and check it raises a KeyError
        with pytest.raises(KeyError):
            _ = calculate_metrics(test_input_y_true, test_input_y_pred, test_input_y_pred_prob[:, test_label],
                                  test_dup_metric_preds, test_dup_metric_probs)

    @pytest.mark.parametrize("test_metric_preds, test_metric_probs", args_metric_funcs)
    def test_metrics_called_correctly(self, mocker, test_metric_preds, test_metric_probs, test_label):
        """Assert the metric functions are called correctly."""

        # Mock the two functions
        mock_example_metrics_pred = mocker.MagicMock(return_value=4.2, __name__="mock_example_metrics_pred")
        mock_example_metrics_pred_prob = mocker.MagicMock(return_value=42, __name__="mock_example_metrics_pred_prob")

        # Replace `test_metric_preds`, and `test_metric_probs` with the two mocks
        test_metric_preds = None if test_metric_preds is None else [mock_example_metrics_pred]
        test_metric_probs = None if test_metric_probs is None else [mock_example_metrics_pred_prob]

        # Call the `calculate_metrics` function
        _ = calculate_metrics(test_input_y_true, test_input_y_pred, test_input_y_pred_prob[:, test_label],
                              test_metric_preds, test_metric_probs)

        # Define the expected `test_input_y_true` - this should coerced to a 1-D NumPy array by the `calculate_metrics`
        # function
        test_expected_y_true = test_input_y_true.ravel()

        # Check `example_metrics_pred` is not called if `test_metric_preds` is None, otherwise that it is called
        # correctly
        if test_metric_preds is None:
            assert not mock_example_metrics_pred.called
        else:

            # Check that the `mock_example_metrics_pred` function is called once
            mock_example_metrics_pred.assert_called_once()

            # Iterate over the call arguments and keyword arguments
            for test_output_preds_args, test_output_preds_kwargs in mock_example_metrics_pred.call_args_list:

                # Assert that there are two arguments, and no keyword arguments
                assert len(test_output_preds_args) == 2
                assert not test_output_preds_kwargs

                # Iterate over the output call arguments, and check they are as expected
                for e, o in zip([test_expected_y_true, test_input_y_pred], test_output_preds_args):
                    assert (o == e).all()

        # Check `example_metrics_pred_prob` is not called if `test_metric_probs` is None, otherwise that it is called
        # correctly
        if test_metric_probs is None:
            assert not mock_example_metrics_pred_prob.called
        else:

            # Check that the `mock_example_metrics_pred_prob` function is called once
            mock_example_metrics_pred_prob.assert_called_once()

            # Get the called arguments
            test_output_probs_calls = mock_example_metrics_pred_prob.call_args_list

            # Iterate over the call arguments and keyword arguments
            for test_output_probs_args, test_output_probs_kwargs in test_output_probs_calls:

                # Assert that there are no arguments, and the correct keyword arguments
                assert len(test_output_probs_args) == 2
                assert not test_output_probs_kwargs

                # Iterate over the output call arguments, and check they are as expected
                for e, o in zip([test_expected_y_true, test_input_y_pred_prob[:, test_label]], test_output_probs_args):
                    assert (o == e).all()

    @pytest.mark.parametrize("test_verbose", [True, False])
    @pytest.mark.parametrize("test_metric_preds, test_metric_probs", args_metric_funcs)
    def test_verbose_prints_correctly(self, mocker, test_metric_preds, test_metric_probs, test_label, test_verbose):
        """Test the function prints the correct values."""

        # Patch the print function
        patch_print = mocker.patch("src.make_feedback_tagging.calculate_performance_metrics.print")

        # Call the `calculate_metrics` function
        test_output = calculate_metrics(test_input_y_true, test_input_y_pred, test_input_y_pred_prob[:, test_label],
                                        test_metric_preds, test_metric_probs, verbose=test_verbose)

        # If `verbose` is True, and `test_metric_preds` and/or `test_metric_probs` are not None, check that `print`
        # is called the corrent number of times
        if test_verbose and (test_metric_preds is not None or test_metric_probs is not None):
            assert patch_print.call_count == (0 if test_metric_preds is None else len(test_metric_preds)) + \
                (0 if test_metric_probs is None else len(test_metric_probs))
        else:
            assert not patch_print.called

        # Check the arguments for `print`
        if test_verbose:

            # Instantiate a list to store the expected call outputs
            test_expected_calls = []

            # Iterate over the `test_output` dictionary, and get the expected print call arguments
            for k, v in test_output.items():
                if isinstance(v, pd.DataFrame):
                    test_expected_calls.append(mocker.call(f"{k}:\n{v.to_string()}\n"))
                elif isinstance(v, dict):
                    test_expected_calls.append(mocker.call(f"{k}:\n{pd.DataFrame(v)}\n"))
                else:
                    test_expected_calls.append(mocker.call(f"{k}: {v:,.2f}\n"))

            # Check the output is as expected
            assert patch_print.call_args_list == test_expected_calls

    @pytest.mark.parametrize("test_metric_preds, test_metric_probs", args_metric_funcs)
    def test_returns_correctly(self, test_metric_preds, test_metric_probs, test_label):
        """Test the calculate_metrics returns the correct output."""

        # Define the expected `test_input_y_true` - this should coerced to a 1-D NumPy array by the `calculate_metrics`
        # function
        test_expected_y_true = test_input_y_true.ravel()

        # Initialise an empty dictionary to store the expected test values
        test_expected = {}

        # Calculate the expected dictionary items for each of the functions
        if test_metric_preds is not None:
            test_expected["example_metrics_pred"] = test_expected_y_true.sum() + test_input_y_pred.sum()
        if test_metric_probs is not None:
            test_expected["example_metrics_pred_prob"] = test_expected_y_true.sum() * \
                test_input_y_pred_prob[:, test_label].sum()

        # Call the `calculate_metrics` function
        test_output = calculate_metrics(test_input_y_true, test_input_y_pred, test_input_y_pred_prob[:, test_label],
                                        test_metric_preds, test_metric_probs)

        # Assert the output is as expected
        assert test_output == test_expected
