from src.make_feedback_tool_data.make_data_for_feedback_tool import preproccess_filter_comment_text
from src.make_feedback_tool_data.preprocess import PreProcess
from pandas.testing import assert_frame_equal
import pandas as pd
import pytest
import re


# Create an example pandas DataFrame of data
DF_EXAMPLE_RAW = pd.DataFrame.from_dict({
    "primary_key": [*range(7)],
    "Q3_x": [
        "", "-", "These words are in English.", "Cet mots sont en français.",
        "This is in English, but there is a word here in اَلْعَرَبِيَّةُ",
        "Η Βικιπαίδεια είναι διεθνής, παγκόσμια, ψηφιακή, διαδικτυακή, ελεύθερου περιεχομένου, εγκυκλοπαίδεια, που "
        "βασίζεται σε ένα μοντέλο ανοικτό στη σύνταξη του περιεχομένου της. It is the largest and most popular general "
        "reference work on the World Wide Web, and is one of the 20 most popular websites ranked by Alexa, as of March "
        "2020.",
        "維基百科 是维基媒体基金会运营的一个多语言的線上百科全書，并以创建和维护作为开放式协同合作项目，特点是自由內容、自由编辑、自由版权"
    ],
})

# Define example personally identifiable information for `EXAMPLE_PARAGRAPHS`
EXAMPLE_PII_REGEX = r"(English)|(World Wide Web)|(mots)"

# Create a pre-processed version of `DF_EXAMPLE`
DF_EXAMPLE_PRE_PROCESSED = DF_EXAMPLE_RAW \
    .assign(Q3_pii_removed=DF_EXAMPLE_RAW["Q3_x"].str.replace(EXAMPLE_PII_REGEX, ""),
            language=["un", "-", "en", "fr", "en", "el", "zh"],
            is_en=[True, True, True, False, True, False, False]) \
    .query("is_en")


@pytest.fixture
def patch_preprocess_pii_regex(mocker):
    """Patch the replace_pii_regex method of the PreProcess class with EXAMPLE_PII_REGEX."""
    return mocker.patch.object(PreProcess, "replace_pii_regex", side_effect=lambda s: re.sub(EXAMPLE_PII_REGEX, "", s))


@pytest.fixture
def patch_preprocess_detect_language(mocker):
    """Patch the detect_language method of the PreProcess class"""
    return mocker.patch.object(PreProcess, "detect_language")


@pytest.mark.parametrize("test_input_threshold", [*range(60, 110, 10)])
class TestPreProcessFilterCommentText:

    def test_returns_correctly(self, patch_preprocess_pii_regex, test_input_threshold):
        """Test that the preproccess_filter_comment_text function returns the correct output."""

        # Define the expected output
        test_expected = DF_EXAMPLE_PRE_PROCESSED.query(f"Q3_pii_removed.str.len() < {test_input_threshold}")

        # Call the `preproccess_filter_comment_text` function
        test_output = preproccess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Assert the same columns exist in both
        assert set(test_output.columns) == set(test_expected.columns)

        # Assert the output is as expected
        assert_frame_equal(test_output, test_expected)

    def test_preprocess_replace_pii_regex_call_count(self, patch_preprocess_pii_regex, test_input_threshold):
        """Test that preproccess_filter_comment_text calls PreProcess.replace_pii_regex the correct number of times."""

        # Call the `preproccess_filter_comment_text` function
        _ = preproccess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Assert that `PreProcess.replace_pii_regex` is called the correct number of times
        assert patch_preprocess_pii_regex.call_count == len(DF_EXAMPLE_RAW)

    def test_preprocess_replace_pii_regex_called_correctly(self, mocker, patch_preprocess_pii_regex,
                                                           test_input_threshold):
        """Test that preproccess_filter_comment_text calls PreProcess.replace_pii_regex with the correct arguments."""

        # Call the `preproccess_filter_comment_text` function
        _ = preproccess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Assert that `PreProcess.replace_pii_regex` is called with the correct arguments
        assert patch_preprocess_pii_regex.call_args_list == [mocker.call(v) for v in DF_EXAMPLE_RAW["Q3_x"]]

    def test_preprocess_detect_language_call_count(self, patch_preprocess_pii_regex, patch_preprocess_detect_language,
                                                   test_input_threshold):
        """Test that preproccess_filter_comment_text calls PreProcess.detect_language the correct number of times."""

        # Call the `preproccess_filter_comment_text` function
        _ = preproccess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Get the expected call count
        test_expected = DF_EXAMPLE_RAW \
            .assign(Q3_pii_removed=DF_EXAMPLE_RAW["Q3_x"].str.replace(EXAMPLE_PII_REGEX, "")) \
            .query(f"Q3_pii_removed.str.len() < {test_input_threshold}") \
            .shape[0]

        # Assert that `PreProcess.replace_pii_regex` is called the correct number of times
        assert patch_preprocess_detect_language.call_count == test_expected

    def test_preprocess_detect_language_called_correctly(self, mocker, patch_preprocess_pii_regex,
                                                         patch_preprocess_detect_language, test_input_threshold):
        """Test that preproccess_filter_comment_text calls PreProcess.detect_language with the correct arguments."""

        # Call the `preproccess_filter_comment_text` function
        _ = preproccess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Define the expected values of the call arguments
        text_expected_values = DF_EXAMPLE_RAW \
            .assign(Q3_pii_removed=DF_EXAMPLE_RAW["Q3_x"].str.replace(EXAMPLE_PII_REGEX, "")) \
            .query(f"Q3_pii_removed.str.len() < {test_input_threshold}") \
            .Q3_pii_removed \
            .to_list()

        # Assert that `PreProcess.detect_language` is called with the correct arguments
        assert patch_preprocess_detect_language.call_args_list == [mocker.call(v) for v in text_expected_values]
