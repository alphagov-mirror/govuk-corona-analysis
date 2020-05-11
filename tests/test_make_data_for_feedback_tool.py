from ast import literal_eval
from src.make_feedback_tool_data.make_data_for_feedback_tool import (
    preproccess_filter_comment_text,
    save_intermediate_df
)
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


# Define input arguments for the `TestSaveIntermediateDf` test class; the first text is 'I am going to go and test to
# see if this example is correct.', the second text is 'If this test passes, we should be able to extract lemma and
# words.', the third text is a combination of the first and second text into a two-sentence text, and the fourth text
# is 'I tried to signed up for advice due to the ongoing COVID 19 outbreak with specific concern about vulnerable
# people. I could not!'
args_save_intermediate_df_inputs = [
    {"pos_tag": [[[("I", "PRP", "-PRON-"), ("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"),
                  ("go", "VB", "go"), ("and", "CC", "and"), ("test", "VB", "test"), ("to", "TO", "to"),
                  ("see", "VB", "see"), ("if", "IN", "if"), ("this", "DT", "this"), ("example", "NN", "example"),
                  ("is", "VBZ", "be"), ("correct", "JJ", "correct"), (".", ".", ".")]]]},
    {"pos_tag": [[[("If", "IN", "if"), ("this", "DT", "this"), ("test", "NN", "test"), ("passes", "VBZ", "pass"),
                  (",", ",", ","), ("we", "PRP", "-PRON-"), ("should", "MD", "should"), ("be", "VB", "be"),
                  ("able", "JJ", "able"), ("to", "TO", "to"), ("extract", "VB", "extract"), ("lemma", "NN", "lemma"),
                  ("and", "CC", "and"), ("words", "NNS", "word"), (".", ".", ".")]]]},
    {"pos_tag": [[[("I", "PRP", "-PRON-"), ("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"),
                  ("go", "VB", "go"), ("and", "CC", "and"), ("test", "VB", "test"), ("to", "TO", "to"),
                  ("see", "VB", "see"), ("if", "IN", "if"), ("this", "DT", "this"), ("example", "NN", "example"),
                  ("is", "VBZ", "be"), ("correct", "JJ", "correct"), (".", ".", ".")],
                 [("If", "IN", "if"), ("this", "DT", "this"), ("test", "NN", "test"), ("passes", "VBZ", "pass"),
                  (",", ",", ","), ("we", "PRP", "-PRON-"), ("should", "MD", "should"), ("be", "VB", "be"),
                  ("able", "JJ", "able"), ("to", "TO", "to"), ("extract", "VB", "extract"), ("lemma", "NN", "lemma"),
                  ("and", "CC", "and"), ("words", "NNS", "word"), (".", ".", ".")]]]},
    {"pos_tag": [[[("I", "PRP", "-PRON-"), ("tried", "VBD", "try"), ("to", "TO", "to"), ("signed", "VBN", "sign"),
                  ("up", "RP", "up"), ("for", "IN", "for"), ("advice", "NN", "advice"), ("due", "IN", "due"),
                  ("to", "IN", "to"), ("the", "DT", "the"), ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"),
                  ("19", "CD", "19"), ("outbreak", "NN", "outbreak"), ("with", "IN", "with"),
                  ("specific", "JJ", "specific"), ("concern", "NN", "concern"), ("about", "IN", "about"),
                  ("vulnerable", "JJ", "vulnerable"), ("people", "NNS", "people")],
                 [("I", "PRP", "-PRON-"), ("could", "MD", "could"), ("not", "RB", "not"), ("!", ".", "!")]]]}
]


# Define the additional expected columns in the outputted CSV by the `save_intermediate_df` function - this will be
# in addition to the columns in `args_save_intermediate_df_inputs`
args_save_intermediate_df_expected = [
    {"lemmas": [["-PRON-", "be", "go", "to", "go", "and", "test", "to", "see", "if", "this", "example", "be",
                 "correct", "."]],
     "words": [["I", "am", "going", "to", "go", "and", "test", "to", "see", "if", "this", "example", "is", "correct",
                "."]]},
    {"lemmas": [["if", "this", "test", "pass", ",", "-PRON-", "should", "be", "able", "to", "extract", "lemma",
                 "and", "word", "."]],
     "words": [["If", "this", "test", "passes", ",", "we", "should", "be", "able", "to", "extract", "lemma", "and",
                "words", "."]]},
    {"lemmas": [["-PRON-", "be", "go", "to", "go", "and", "test", "to", "see", "if", "this", "example", "be",
                 "correct", ".", "if", "this", "test", "pass", ",", "-PRON-", "should", "be", "able", "to",
                 "extract", "lemma", "and", "word", "."]],
     "words": [["I", "am", "going", "to", "go", "and", "test", "to", "see", "if", "this", "example", "is", "correct",
                ".", "If", "this", "test", "passes", ",", "we", "should", "be", "able", "to", "extract", "lemma", "and",
                "words", "."]]},
    {"lemmas": [["-PRON-", "try", "to", "sign", "up", "for", "advice", "due", "to", "the", "ongoing", "COVID", "19",
                 "outbreak", "with", "specific", "concern", "about", "vulnerable", "people", "-PRON-", "could", "not",
                 "!"]],
     "words": [["I", "tried", "to", "signed", "up", "for", "advice", "due", "to", "the", "ongoing", "COVID", "19",
                "outbreak", "with", "specific", "concern", "about", "vulnerable", "people", "I", "could", "not", "!"]]}
]

# Create the test cases for the `TestSaveIntermediateDf` test class, where each tuple in the list consists of two
# elements; the first is each element of `args_save_intermediate_df_inputs` as a pandas DataFrame, and the second is
# a pandas DataFrame of the corresponding elements from `args_save_intermediate_df_inputs` and
# `args_save_intermediate_df_expected` as the expected outputs
args_save_intermediate_df = [
    (pd.DataFrame(i), pd.DataFrame({**i, **e})) for i, e in zip(args_save_intermediate_df_inputs,
                                                                args_save_intermediate_df_expected)
]


@pytest.fixture
def patch_pandas_dataframe_to_csv(mocker):
    """Patch the pandas.DataFrame.to_csv method for the TestSaveIntermediateDf test class."""
    return mocker.patch("pandas.DataFrame.to_csv")


@pytest.fixture
def temp_intermediate_folder(tmpdir_factory):
    """Create a temporary directory to store the output from save_intermediate_df."""
    return tmpdir_factory.mktemp("temp")


@pytest.mark.parametrize("test_input_cache_pos_filename", ["foo.csv", "bar.csv"])
class TestSaveIntermediateDf:

    @pytest.mark.parametrize("test_input_df", [a[0] for a in args_save_intermediate_df])
    def test_calls_to_csv_correctly(self, patch_pandas_dataframe_to_csv, test_input_df, test_input_cache_pos_filename):
        """Test save_intermediate_df calls pandas.DataFrame.to_csv correctly."""

        # Call the `save_intermediate_df` function
        _ = save_intermediate_df(test_input_df, test_input_cache_pos_filename)

        # Assert `pandas.DataFrame.to_csv` is called with the correct arguments
        patch_pandas_dataframe_to_csv.assert_called_once_with(test_input_cache_pos_filename, index=False)

    @pytest.mark.parametrize("test_input_df, test_expected_df", args_save_intermediate_df)
    def test_returns_correctly(self, temp_intermediate_folder, test_input_df, test_input_cache_pos_filename,
                               test_expected_df):
        """Test the outputted CSV from save_intermediate_df is correct."""

        # Define the file path for the CSV
        test_input_file_path = temp_intermediate_folder.join(test_input_cache_pos_filename)

        # Call the `save_intermediate_df` function
        _ = save_intermediate_df(test_input_df, test_input_file_path)

        # Assert the CSV output is correct; need to apply `ast.literal_eval` element-wise, as the CSV will contain
        # strings of the lists, rather than the lists themselves
        assert_frame_equal(pd.read_csv(test_input_file_path).applymap(literal_eval), test_expected_df)
